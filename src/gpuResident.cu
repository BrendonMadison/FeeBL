#include "gpuResident.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace feebl {
namespace gpu {

namespace {

const int kThreads = 256;

bool cuda_ok(cudaError_t status, const char* operation, std::string& error)
{
  if (status == cudaSuccess) return true;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status)
          << " (CUDA error " << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

bool cufft_ok(cufftResult status, const char* operation, std::string& error)
{
  if (status == CUFFT_SUCCESS) return true;
  std::ostringstream message;
  message << operation << " failed (cuFFT error "
          << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

bool checked_product(std::size_t left,
                     std::size_t right,
                     std::size_t& result)
{
  if (left && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

bool checked_sum(std::size_t left,
                 std::size_t right,
                 std::size_t& result)
{
  if (right > std::numeric_limits<std::size_t>::max() - left)
    return false;
  result = left + right;
  return true;
}

int block_count(std::size_t work)
{
  if (!work) return 1;
  const std::size_t needed = (work + kThreads - 1) / kThreads;
  return static_cast<int>(std::min<std::size_t>(needed, 65535));
}

bool ranges_overlap(const void* first,
                    const void* second,
                    std::size_t bytes)
{
  if (!bytes) return false;
  const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(second);
  if (a > std::numeric_limits<std::uintptr_t>::max() - bytes ||
      b > std::numeric_limits<std::uintptr_t>::max() - bytes)
    return true;
  return a < b + bytes && b < a + bytes;
}

bool ranges_overlap(const void* first,
                    std::size_t first_bytes,
                    const void* second,
                    std::size_t second_bytes)
{
  if (!first_bytes || !second_bytes) return false;
  const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(second);
  if (a > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      b > std::numeric_limits<std::uintptr_t>::max() - second_bytes)
    return true;
  return a < b + second_bytes && b < a + first_bytes;
}

std::string version_text(int encoded)
{
  if (encoded < 0) return "unavailable";
  std::ostringstream value;
  // CUDA and cuFFT both expose integer versions in major*1000+minor*10 form.
  value << encoded / 1000 << '.' << (encoded % 1000) / 10;
  const int patch = encoded % 10;
  if (patch) value << '.' << patch;
  return value.str();
}

__global__ void pack_rho_kernel(const double* rho1,
                                const double* rho2,
                                int nx,
                                int ny,
                                int padded_ny,
                                cufftDoubleComplex* spectrum)
{
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t cell =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells;
       cell += stride)
    {
      const int ix = static_cast<int>(cell / static_cast<std::size_t>(ny));
      const int iy = static_cast<int>(cell -
                                      static_cast<std::size_t>(ix) * ny);
      cufftDoubleComplex value;
      value.x = rho1[cell];
      value.y = rho2[cell];
      spectrum[static_cast<std::size_t>(ix) * padded_ny + iy] = value;
    }
}

__global__ void multiply_spectrum_kernel(cufftDoubleComplex* spectrum,
                                         const cufftDoubleComplex* dist,
                                         std::size_t values)
{
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < values;
       index += stride)
    {
      const cufftDoubleComplex a = spectrum[index];
      const cufftDoubleComplex b = dist[index];
      cufftDoubleComplex product;
      product.x = a.x * b.x - a.y * b.y;
      product.y = a.y * b.x + a.x * b.y;
      spectrum[index] = product;
    }
}

__global__ void unpack_phi_kernel(const cufftDoubleComplex* spectrum,
                                  int nx,
                                  int ny,
                                  int padded_ny,
                                  double* phi1,
                                  double* phi2)
{
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t cell =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells;
       cell += stride)
    {
      const int ix = static_cast<int>(cell / static_cast<std::size_t>(ny));
      const int iy = static_cast<int>(cell -
                                      static_cast<std::size_t>(ix) * ny);
      const cufftDoubleComplex value =
          spectrum[static_cast<std::size_t>(ix) * padded_ny + iy];
      phi1[cell] = value.x;
      phi2[cell] = value.y;
    }
}

struct DeviceDepositMetrics
{
  unsigned long long beam1_in;
  unsigned long long beam1_out;
  unsigned long long beam2_in;
  unsigned long long beam2_out;
  double fine_lumi_dot;
};

struct DeviceFieldGatherMetrics
{
  unsigned long long beam1_in;
  unsigned long long beam1_out;
  unsigned long long beam2_in;
  unsigned long long beam2_out;
};

__device__ void electric_field_cic_legacy(const float x,
                                          const float y,
                                          const double* phi,
                                          const int ny,
                                          const float delta_x_inverse,
                                          const float delta_y_inverse,
                                          const float offset_x,
                                          const float offset_y,
                                          float& field_x,
                                          float& field_y)
{
  // Keep the exact arithmetic types and narrowing points used by
  // GENERAL_GRID::interpolePotential/ElectricFieldCIC.  In particular,
  // absolute coordinates and localization weights are floats, potential
  // interpolation and differentiation are doubles, the TRIVECTOR assignment
  // narrows to float, and its subsequent *=0.5 is a second float operation.
  const float absolute_x = x * delta_x_inverse + offset_x;
  const float absolute_y = y * delta_y_inverse + offset_y;

  const int y_i1 = static_cast<int>(floorf(absolute_x));
  float y_weight = absolute_y - 0.5f;
  const int y_i2 = static_cast<int>(floorf(y_weight));
  y_weight -= static_cast<float>(y_i2);
  const float y_weight_complement = 1.0f - y_weight;
  const std::size_t y_center =
      static_cast<std::size_t>(y_i1) * ny + y_i2;
  const double phi1_y =
      static_cast<double>(y_weight) * phi[y_center + ny + 1] +
      static_cast<double>(y_weight_complement) * phi[y_center + ny];
  const double phi2_y =
      static_cast<double>(y_weight) * phi[y_center + 1] +
      static_cast<double>(y_weight_complement) * phi[y_center];
  const double phi3_y =
      static_cast<double>(y_weight) * phi[y_center - ny + 1] +
      static_cast<double>(y_weight_complement) * phi[y_center - ny];

  const int x_i2 = static_cast<int>(floorf(absolute_y));
  float x_weight = absolute_x - 0.5f;
  const int x_i1 = static_cast<int>(floorf(x_weight));
  x_weight -= static_cast<float>(x_i1);
  const float x_weight_complement = 1.0f - x_weight;
  const std::size_t x_center =
      static_cast<std::size_t>(x_i1) * ny + x_i2;
  const double phi1_x =
      static_cast<double>(x_weight) * phi[x_center + ny + 1] +
      static_cast<double>(x_weight_complement) * phi[x_center + 1];
  const double phi2_x =
      static_cast<double>(x_weight) * phi[x_center + ny] +
      static_cast<double>(x_weight_complement) * phi[x_center];
  const double phi3_x =
      static_cast<double>(x_weight) * phi[x_center + ny - 1] +
      static_cast<double>(x_weight_complement) * phi[x_center - 1];

  double fractional_x = static_cast<double>(absolute_x);
  double fractional_y = static_cast<double>(absolute_y);
  fractional_x -= floor(fractional_x);
  fractional_y -= floor(fractional_y);
  const double raw_x =
      (fractional_x * (phi1_y - phi2_y) +
       (1.0 - fractional_x) * (phi2_y - phi3_y)) *
      static_cast<double>(delta_x_inverse);
  const double raw_y =
      (fractional_y * (phi1_x - phi2_x) +
       (1.0 - fractional_y) * (phi2_x - phi3_x)) *
      static_cast<double>(delta_y_inverse);
  const float trivec_x = static_cast<float>(raw_x);
  const float trivec_y = static_cast<float>(raw_y);
  field_x = trivec_x * 0.5f;
  field_y = trivec_y * 0.5f;
}

__global__ void gather_opposing_field_kernel(
    const float* xy,
    std::size_t particle_count,
    const double* opposing_phi,
    int beam,
    int nx,
    int ny,
    float delta_x_inverse,
    float delta_y_inverse,
    float offset_x,
    float offset_y,
    float physical_min_x,
    float physical_max_x,
    float physical_min_y,
    float physical_max_y,
    bool zero_field,
    float* fields,
    DeviceFieldGatherMetrics* metrics)
{
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      const float x = xy[2 * particle];
      const float y = xy[2 * particle + 1];
      const float absolute_x = x * delta_x_inverse + offset_x;
      const float absolute_y = y * delta_y_inverse + offset_y;
      // Preserve the exact physical predicate and additionally require the
      // localized stencil to be safe.  The latter deliberately excludes a
      // legacy one-ULP upper boundary layer that could otherwise address
      // beyond phi after float coordinate rounding.
      const bool in_grid =
          x > physical_min_x && x < physical_max_x &&
          y > physical_min_y && y < physical_max_y &&
          absolute_x >= 1.0f &&
          absolute_x < static_cast<float>(nx - 1) &&
          absolute_y >= 1.0f &&
          absolute_y < static_cast<float>(ny - 1);
      if (!in_grid)
        {
          fields[2 * particle] = 0.0f;
          fields[2 * particle + 1] = 0.0f;
          if (beam == 0) atomicAdd(&metrics->beam1_out, 1ULL);
          else atomicAdd(&metrics->beam2_out, 1ULL);
          continue;
        }
      if (beam == 0) atomicAdd(&metrics->beam1_in, 1ULL);
      else atomicAdd(&metrics->beam2_in, 1ULL);
      if (zero_field)
        {
          fields[2 * particle] = 0.0f;
          fields[2 * particle + 1] = 0.0f;
        }
      else
        {
          electric_field_cic_legacy(
              x, y, opposing_phi, ny, delta_x_inverse,
              delta_y_inverse, offset_x, offset_y,
              fields[2 * particle], fields[2 * particle + 1]);
        }
    }
}

__device__ bool locate_cic_particle(float x,
                                    float y,
                                    float delta_x_inverse,
                                    float delta_y_inverse,
                                    float offset_x,
                                    float offset_y,
                                    int nx,
                                    int ny,
                                    int& ix,
                                    int& iy,
                                    float& hx,
                                    float& hy)
{
  // This grouping intentionally matches gpuCic.cu and the fast-math host
  // GRID::particleInGrid boundary behavior.
  hx = x * delta_x_inverse + (offset_x - 0.5f);
  ix = static_cast<int>(floorf(hx));
  hx -= static_cast<float>(ix);
  hy = y * delta_y_inverse + (offset_y - 0.5f);
  iy = static_cast<int>(floorf(hy));
  hy -= static_cast<float>(iy);
  return ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1;
}

__global__ void resident_cic_kernel(const float* xy,
                                    std::size_t particle_count,
                                    double charge,
                                    int beam,
                                    int nx,
                                    int ny,
                                    float delta_x_inverse,
                                    float delta_y_inverse,
                                    float offset_x,
                                    float offset_y,
                                    double* rho,
                                    DeviceDepositMetrics* metrics)
{
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      int ix, iy;
      float hx, hy;
      if (!locate_cic_particle(xy[2 * particle], xy[2 * particle + 1],
                               delta_x_inverse, delta_y_inverse,
                               offset_x, offset_y, nx, ny,
                               ix, iy, hx, hy))
        {
          if (beam == 0) atomicAdd(&metrics->beam1_out, 1ULL);
          else atomicAdd(&metrics->beam2_out, 1ULL);
          continue;
        }
      if (beam == 0) atomicAdd(&metrics->beam1_in, 1ULL);
      else atomicAdd(&metrics->beam2_in, 1ULL);

      const double dx = static_cast<double>(hx);
      const double dy = static_cast<double>(hy);
      atomicAdd(rho + ix * ny + iy,
                (1.0 - dx) * (1.0 - dy) * charge);
      atomicAdd(rho + (ix + 1) * ny + iy,
                dx * (1.0 - dy) * charge);
      atomicAdd(rho + ix * ny + iy + 1,
                (1.0 - dx) * dy * charge);
      atomicAdd(rho + (ix + 1) * ny + iy + 1, dx * dy * charge);
    }
}

__global__ void symmetrize_four_way_kernel(double* rho1,
                                           double* rho2,
                                           int nx,
                                           int ny)
{
  const int half_x = nx / 2;
  const int half_y = ny / 2;
  const std::size_t representatives =
      static_cast<std::size_t>(half_x) * half_y;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t representative =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       representative < representatives;
       representative += stride)
    {
      const int ix =
          static_cast<int>(representative / static_cast<std::size_t>(half_y));
      const int iy = static_cast<int>(representative -
                                      static_cast<std::size_t>(ix) * half_y);
      const std::size_t j1 = static_cast<std::size_t>(ix) * ny + iy;
      const std::size_t j2 = static_cast<std::size_t>(nx - 1 - ix) * ny + iy;
      const std::size_t j3 =
          static_cast<std::size_t>(ix) * ny + (ny - 1 - iy);
      const std::size_t j4 = static_cast<std::size_t>(nx - 1 - ix) * ny +
                             (ny - 1 - iy);
      const double mean1 = 0.25 *
          (rho1[j1] + rho1[j2] + rho1[j3] + rho1[j4]);
      const double mean2 = 0.25 *
          (rho2[j1] + rho2[j2] + rho2[j3] + rho2[j4]);
      rho1[j1] = mean1;
      rho1[j2] = mean1;
      rho1[j3] = mean1;
      rho1[j4] = mean1;
      rho2[j1] = mean2;
      rho2[j2] = mean2;
      rho2[j3] = mean2;
      rho2[j4] = mean2;
    }
}

__global__ void fine_lumi_dot_kernel(const double* rho1,
                                     const double* rho2,
                                     std::size_t cells,
                                     DeviceDepositMetrics* metrics)
{
  __shared__ double partial[kThreads];
  double sum = 0.0;
  for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
    sum += rho1[cell] * rho2[cell];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
      if (threadIdx.x < offset)
        partial[threadIdx.x] += partial[threadIdx.x + offset];
      __syncthreads();
    }
  if (threadIdx.x == 0) metrics->fine_lumi_dot = partial[0];
}

}  // namespace

ResidentFieldOptions::ResidentFieldOptions()
    : device(0), pinned_io(false), memory_fraction(0.80)
{
}

ResidentFieldStats::ResidentFieldStats()
    : calls(0),
      device_work_calls(0),
      deposition_calls(0),
      field_solve_calls(0),
      field_gather_calls(0),
      failed_device_calls(0),
      failure_drains(0),
      failure_drain_failures(0),
      setup_ms(0.0),
      h2d_ms(0.0),
      kernel_ms(0.0),
      d2h_ms(0.0),
      total_ms(0.0),
      last_h2d_ms(0.0),
      last_kernel_ms(0.0),
      last_d2h_ms(0.0),
      last_total_ms(0.0),
      last_device_work(false),
      setup_h2d_bytes(0),
      h2d_bytes(0),
      d2h_bytes(0),
      last_h2d_bytes(0),
      last_d2h_bytes(0),
      setup_h2d_operations(0),
      h2d_operations(0),
      d2h_operations(0),
      last_h2d_operations(0),
      last_d2h_operations(0),
      setup_synchronizations(0),
      synchronizations(0),
      last_synchronizations(0),
      device_allocations(0),
      pinned_host_allocations(0),
      plan_creations(0),
      resident_bytes(0),
      field_required_bytes(0),
      cufft_workspace_bytes(0),
      device_free_bytes_at_init(0),
      device_total_bytes(0),
      memory_budget_bytes(0),
      particle_capacity1(0),
      particle_capacity2(0),
      nx(0),
      ny(0),
      padded_nx(0),
      padded_ny(0),
      cuda_runtime_version(-1),
      cuda_driver_version(-1),
      cufft_runtime_version(-1),
      pinned_io(false),
      context_poisoned(false),
      memory_fraction(0.0)
{
}

ResidentDepositResult::ResidentDepositResult()
    : beam1_in(0),
      beam1_out(0),
      beam2_in(0),
      beam2_out(0),
      fine_lumi_dot(0.0)
{
}

ResidentFieldGatherResult::ResidentFieldGatherResult()
    : beam1_in(0), beam1_out(0), beam2_in(0), beam2_out(0)
{
}

struct ResidentFieldSolver::Impl
{
  int device;
  bool initialized;
  bool capacity_reserved;
  bool resident_rho_ready;
  bool resident_particles_ready;
  bool poisoned;
  bool work_may_be_pending;
  std::string poison_reason;
  std::size_t resident_particle_count1;
  std::size_t resident_particle_count2;
  float resident_delta_x_inverse;
  float resident_delta_y_inverse;
  float resident_offset_x;
  float resident_offset_y;
  int nx;
  int ny;
  int padded_nx;
  int padded_ny;
  std::size_t cells;
  std::size_t padded_cells;
  std::size_t cell_bytes;
  std::size_t spectrum_bytes;
  std::string device_text;
  std::string runtime_text;
  int compute_major;
  int compute_minor;

  cudaStream_t stream;
  cudaEvent_t events[4];
  cufftHandle plan;
  bool plan_valid;
  cufftDoubleComplex* spectrum;
  cufftDoubleComplex* dist;
  double* rho1;
  double* rho2;
  double* phi1;
  double* phi2;
  void* workspace;
  float* xy1;
  float* xy2;
  float* fields1;
  float* fields2;
  std::size_t xy_capacity1;
  std::size_t xy_capacity2;
  DeviceDepositMetrics* deposit_metrics;
  DeviceFieldGatherMetrics* field_gather_metrics;

  double* pinned_rho1;
  double* pinned_rho2;
  double* pinned_phi1;
  double* pinned_phi2;
  float* pinned_xy1;
  float* pinned_xy2;
  float* pinned_fields1;
  float* pinned_fields2;
  DeviceDepositMetrics* pinned_deposit_metrics;
  DeviceFieldGatherMetrics* pinned_field_gather_metrics;

  ResidentFieldStats statistics;

  Impl()
      : device(-1),
        initialized(false),
        capacity_reserved(false),
        resident_rho_ready(false),
        resident_particles_ready(false),
        poisoned(false),
        work_may_be_pending(false),
        resident_particle_count1(0),
        resident_particle_count2(0),
        resident_delta_x_inverse(0.0f),
        resident_delta_y_inverse(0.0f),
        resident_offset_x(0.0f),
        resident_offset_y(0.0f),
        nx(0),
        ny(0),
        padded_nx(0),
        padded_ny(0),
        cells(0),
        padded_cells(0),
        cell_bytes(0),
        spectrum_bytes(0),
        compute_major(0),
        compute_minor(0),
        stream(0),
        plan(0),
        plan_valid(false),
        spectrum(0),
        dist(0),
        rho1(0),
        rho2(0),
        phi1(0),
        phi2(0),
        workspace(0),
        xy1(0),
        xy2(0),
        fields1(0),
        fields2(0),
        xy_capacity1(0),
        xy_capacity2(0),
        deposit_metrics(0),
        field_gather_metrics(0),
        pinned_rho1(0),
        pinned_rho2(0),
        pinned_phi1(0),
        pinned_phi2(0),
        pinned_xy1(0),
        pinned_xy2(0),
        pinned_fields1(0),
        pinned_fields2(0),
        pinned_deposit_metrics(0),
        pinned_field_gather_metrics(0)
  {
    for (int index = 0; index < 4; ++index) events[index] = 0;
  }

  void release()
  {
    if (device >= 0) cudaSetDevice(device);
    // A failed asynchronous call must not leave work targeting caller-owned
    // or internal staging storage while teardown frees the context.  This is
    // intentionally best effort because a reset/lost device can also make
    // synchronization fail.
    bool handles_valid = true;
    if (stream)
      {
        if (cudaStreamSynchronize(stream) == cudaSuccess)
          work_may_be_pending = false;
        else
          handles_valid = false;
      }
    // A reset/lost context invalidates every CUDA/cuFFT handle and allocation;
    // invoking their destroy/free entry points is not safe on all WSL/WDDM
    // stacks.  The driver has already reclaimed a reset context, while a lost
    // context is reclaimed by runtime teardown.  Abandon only those invalid
    // handles after the required drain attempt.
    if (!handles_valid)
      {
        plan_valid = false;
        plan = 0;
        workspace = 0;
        fields2 = 0;
        fields1 = 0;
        xy2 = 0;
        xy1 = 0;
        field_gather_metrics = 0;
        deposit_metrics = 0;
        phi2 = 0;
        phi1 = 0;
        rho2 = 0;
        rho1 = 0;
        dist = 0;
        spectrum = 0;
        pinned_field_gather_metrics = 0;
        pinned_deposit_metrics = 0;
        pinned_fields2 = 0;
        pinned_fields1 = 0;
        pinned_xy2 = 0;
        pinned_xy1 = 0;
        pinned_phi2 = 0;
        pinned_phi1 = 0;
        pinned_rho2 = 0;
        pinned_rho1 = 0;
        for (int index = 0; index < 4; ++index) events[index] = 0;
        stream = 0;
        initialized = false;
        capacity_reserved = false;
        resident_rho_ready = false;
        resident_particles_ready = false;
        work_may_be_pending = false;
        device = -1;
        return;
      }
    if (plan_valid) cufftDestroy(plan);
    plan_valid = false;
    plan = 0;
    if (workspace) cudaFree(workspace);
    if (fields2) cudaFree(fields2);
    if (fields1) cudaFree(fields1);
    if (xy2) cudaFree(xy2);
    if (xy1) cudaFree(xy1);
    if (field_gather_metrics) cudaFree(field_gather_metrics);
    if (deposit_metrics) cudaFree(deposit_metrics);
    if (phi2) cudaFree(phi2);
    if (phi1) cudaFree(phi1);
    if (rho2) cudaFree(rho2);
    if (rho1) cudaFree(rho1);
    if (dist) cudaFree(dist);
    if (spectrum) cudaFree(spectrum);
    workspace = 0;
    phi2 = 0;
    phi1 = 0;
    rho2 = 0;
    rho1 = 0;
    dist = 0;
    spectrum = 0;
    xy2 = 0;
    xy1 = 0;
    fields2 = 0;
    fields1 = 0;
    xy_capacity2 = 0;
    xy_capacity1 = 0;
    deposit_metrics = 0;
    field_gather_metrics = 0;
    if (pinned_field_gather_metrics)
      cudaFreeHost(pinned_field_gather_metrics);
    if (pinned_deposit_metrics) cudaFreeHost(pinned_deposit_metrics);
    if (pinned_fields2) cudaFreeHost(pinned_fields2);
    if (pinned_fields1) cudaFreeHost(pinned_fields1);
    if (pinned_xy2) cudaFreeHost(pinned_xy2);
    if (pinned_xy1) cudaFreeHost(pinned_xy1);
    if (pinned_phi2) cudaFreeHost(pinned_phi2);
    if (pinned_phi1) cudaFreeHost(pinned_phi1);
    if (pinned_rho2) cudaFreeHost(pinned_rho2);
    if (pinned_rho1) cudaFreeHost(pinned_rho1);
    pinned_phi2 = 0;
    pinned_phi1 = 0;
    pinned_rho2 = 0;
    pinned_rho1 = 0;
    pinned_xy2 = 0;
    pinned_xy1 = 0;
    pinned_fields2 = 0;
    pinned_fields1 = 0;
    pinned_deposit_metrics = 0;
    pinned_field_gather_metrics = 0;
    for (int index = 0; index < 4; ++index)
      {
        if (events[index]) cudaEventDestroy(events[index]);
        events[index] = 0;
      }
    if (stream) cudaStreamDestroy(stream);
    stream = 0;
    initialized = false;
    capacity_reserved = false;
    resident_rho_ready = false;
    resident_particles_ready = false;
    poisoned = false;
    work_may_be_pending = false;
    poison_reason.clear();
    resident_particle_count1 = 0;
    resident_particle_count2 = 0;
    device = -1;
  }

  bool reject_if_poisoned(std::string& error) const
  {
    if (!poisoned) return false;
    error = "GPU2 resident context is poisoned after a device failure; "
            "retry and CPU fallback are forbidden";
    if (!poison_reason.empty()) error += ": " + poison_reason;
    return true;
  }

  void poison_device_context(std::string& error)
  {
    const std::string primary = error.empty()
        ? "unspecified CUDA/cuFFT failure" : error;
    if (!poisoned) poison_reason = primary;
    poisoned = true;
    statistics.context_poisoned = true;
    ++statistics.failed_device_calls;

    std::ostringstream detail;
    detail << primary << "; GPU2 context poisoned";
    if (work_may_be_pending && stream)
      {
        ++statistics.failure_drains;
        const cudaError_t drain_status = cudaStreamSynchronize(stream);
        if (drain_status == cudaSuccess)
          {
            work_may_be_pending = false;
            detail << "; queued work drained";
          }
        else
          {
            ++statistics.failure_drain_failures;
            detail << "; best-effort stream drain failed: "
                   << cudaGetErrorString(drain_status)
                   << " (CUDA error " << static_cast<int>(drain_status)
                   << ')';
          }
      }
    detail << "; outputs are undefined and retry/fallback is forbidden";
    error = detail.str();
  }

  struct DeviceWorkGuard
  {
    Impl& owner;
    std::string& error;
    ResidentFieldStats* output_statistics;
    bool armed;
    bool succeeded;

    DeviceWorkGuard(Impl& owner_in,
                    std::string& error_in,
                    ResidentFieldStats* output_statistics_in)
        : owner(owner_in),
          error(error_in),
          output_statistics(output_statistics_in),
          armed(false),
          succeeded(false)
    {
    }

    ~DeviceWorkGuard()
    {
      if (armed && !succeeded)
        {
          owner.poison_device_context(error);
          if (output_statistics) *output_statistics = owner.statistics;
        }
    }

    void arm()
    {
      armed = true;
      owner.work_may_be_pending = true;
    }

    void mark_drained()
    {
      owner.work_may_be_pending = false;
    }

    void complete()
    {
      owner.work_may_be_pending = false;
      succeeded = true;
    }
  };

  bool allocate_device(void** pointer,
                       std::size_t bytes,
                       const char* label,
                       std::string& error)
  {
    if (!cuda_ok(cudaMalloc(pointer, bytes), label, error)) return false;
    ++statistics.device_allocations;
    statistics.resident_bytes += bytes;
    return true;
  }

  bool allocate_pinned(void** pointer,
                       std::size_t bytes,
                       const char* label,
                       std::string& error)
  {
    if (!cuda_ok(cudaHostAlloc(pointer, bytes, cudaHostAllocDefault),
                 label, error))
      return false;
    ++statistics.pinned_host_allocations;
    return true;
  }

  bool ensure_particle_capacity(int beam,
                                std::size_t particles,
                                std::string& error)
  {
    std::size_t& capacity = beam == 0 ? xy_capacity1 : xy_capacity2;
    if (particles <= capacity) return true;
    std::size_t elements = 0;
    std::size_t bytes = 0;
    std::size_t total_device_bytes = 0;
    if (!checked_product(particles, static_cast<std::size_t>(2), elements) ||
        !checked_product(elements, sizeof(float), bytes) ||
        !checked_product(bytes, static_cast<std::size_t>(2),
                         total_device_bytes))
      {
        error = "GPU2 resident particle/field allocation overflows size_t";
        return false;
      }

    float* xy_replacement = 0;
    float* field_replacement = 0;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&xy_replacement), bytes),
                 beam == 0 ? "cudaMalloc(GPU2 xy1)"
                           : "cudaMalloc(GPU2 xy2)", error))
      return false;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&field_replacement),
                            bytes),
                 beam == 0 ? "cudaMalloc(GPU2 compact fields1)"
                           : "cudaMalloc(GPU2 compact fields2)", error))
      {
        cudaFree(xy_replacement);
        return false;
      }
    float* pinned_xy_replacement = 0;
    float* pinned_field_replacement = 0;
    if (statistics.pinned_io &&
        !cuda_ok(cudaHostAlloc(
                               reinterpret_cast<void**>(&pinned_xy_replacement),
                               bytes, cudaHostAllocDefault),
                 beam == 0 ? "cudaHostAlloc(GPU2 xy1 staging)"
                           : "cudaHostAlloc(GPU2 xy2 staging)", error))
      {
        cudaFree(field_replacement);
        cudaFree(xy_replacement);
        return false;
      }
    if (statistics.pinned_io &&
        !cuda_ok(cudaHostAlloc(
                               reinterpret_cast<void**>(&pinned_field_replacement),
                               bytes, cudaHostAllocDefault),
                 beam == 0 ? "cudaHostAlloc(GPU2 compact fields1 staging)"
                           : "cudaHostAlloc(GPU2 compact fields2 staging)",
                 error))
      {
        cudaFreeHost(pinned_xy_replacement);
        cudaFree(field_replacement);
        cudaFree(xy_replacement);
        return false;
      }

    float*& current_xy = beam == 0 ? xy1 : xy2;
    float*& current_field = beam == 0 ? fields1 : fields2;
    float*& pinned_current_xy = beam == 0 ? pinned_xy1 : pinned_xy2;
    float*& pinned_current_field =
        beam == 0 ? pinned_fields1 : pinned_fields2;
    const std::size_t old_bytes =
        4 * capacity * sizeof(float);
    if (current_field) cudaFree(current_field);
    if (current_xy) cudaFree(current_xy);
    if (pinned_current_field) cudaFreeHost(pinned_current_field);
    if (pinned_current_xy) cudaFreeHost(pinned_current_xy);
    current_xy = xy_replacement;
    current_field = field_replacement;
    pinned_current_xy = pinned_xy_replacement;
    pinned_current_field = pinned_field_replacement;
    capacity = particles;
    statistics.particle_capacity1 = xy_capacity1;
    statistics.particle_capacity2 = xy_capacity2;
    statistics.device_allocations += 2;
    statistics.resident_bytes += total_device_bytes - old_bytes;
    if (statistics.pinned_io) statistics.pinned_host_allocations += 2;
    return true;
  }
};

ResidentFieldSolver::ResidentFieldSolver() : impl_(new Impl)
{
}

ResidentFieldSolver::~ResidentFieldSolver()
{
  impl_->release();
  delete impl_;
}

bool ResidentFieldSolver::initialize(const ResidentFieldOptions& options,
                                     int nx,
                                     int ny,
                                     const double* dist_interleaved,
                                     std::string& error)
{
  error.clear();
  if (impl_->initialized || impl_->device >= 0)
    {
      error = "GPU2 resident field solver is already initialized";
      return false;
    }
  if (options.device < 0 || nx <= 0 || ny <= 0 || !dist_interleaved ||
      !std::isfinite(options.memory_fraction) ||
      options.memory_fraction <= 0.0 || options.memory_fraction > 1.0)
    {
      error = "invalid GPU2 device, grid dimensions, Green spectrum, or memory fraction (required range: 0 < fraction <= 1)";
      return false;
    }

  impl_->statistics = ResidentFieldStats();
  if (nx > std::numeric_limits<int>::max() / 2 ||
      ny > std::numeric_limits<int>::max() / 2)
    {
      error = "GPU2 padded field dimensions overflow cuFFT int dimensions";
      return false;
    }

  std::size_t cells = 0;
  std::size_t padded_cells = 0;
  std::size_t cell_bytes = 0;
  std::size_t spectrum_bytes = 0;
  if (!checked_product(static_cast<std::size_t>(nx),
                       static_cast<std::size_t>(ny), cells) ||
      !checked_product(static_cast<std::size_t>(2 * nx),
                       static_cast<std::size_t>(2 * ny), padded_cells) ||
      !checked_product(cells, sizeof(double), cell_bytes) ||
      !checked_product(padded_cells, sizeof(cufftDoubleComplex),
                       spectrum_bytes))
    {
      error = "GPU2 field dimensions overflow host size_t byte counts";
      return false;
    }

  const std::chrono::steady_clock::time_point setup_begin =
      std::chrono::steady_clock::now();

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error))
    return false;
  if (options.device >= device_count)
    {
      std::ostringstream message;
      message << "CUDA device " << options.device
              << " is unavailable; found " << device_count << " device(s)";
      error = message.str();
      return false;
    }
  if (!cuda_ok(cudaSetDevice(options.device), "cudaSetDevice", error))
    return false;
  impl_->device = options.device;

  cudaDeviceProp properties;
  if (!cuda_ok(cudaGetDeviceProperties(&properties, options.device),
               "cudaGetDeviceProperties", error))
    {
      impl_->release();
      return false;
    }

  impl_->nx = nx;
  impl_->ny = ny;
  impl_->padded_nx = 2 * nx;
  impl_->padded_ny = 2 * ny;
  impl_->cells = cells;
  impl_->padded_cells = padded_cells;
  impl_->cell_bytes = cell_bytes;
  impl_->spectrum_bytes = spectrum_bytes;
  impl_->statistics.nx = nx;
  impl_->statistics.ny = ny;
  impl_->statistics.padded_nx = 2 * nx;
  impl_->statistics.padded_ny = 2 * ny;
  impl_->statistics.pinned_io = options.pinned_io;
  impl_->statistics.memory_fraction = options.memory_fraction;
  impl_->compute_major = properties.major;
  impl_->compute_minor = properties.minor;

  if (!cuda_ok(cudaRuntimeGetVersion(
                   &impl_->statistics.cuda_runtime_version),
               "cudaRuntimeGetVersion", error) ||
      !cuda_ok(cudaDriverGetVersion(
                   &impl_->statistics.cuda_driver_version),
               "cudaDriverGetVersion", error) ||
      !cufft_ok(cufftGetVersion(
                    &impl_->statistics.cufft_runtime_version),
                "cufftGetVersion", error))
    {
      impl_->release();
      return false;
    }
  std::size_t free_device_bytes = 0;
  std::size_t total_device_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
               "cudaMemGetInfo(GPU2 preflight)", error))
    {
      impl_->release();
      return false;
    }
  impl_->statistics.device_free_bytes_at_init = free_device_bytes;
  impl_->statistics.device_total_bytes = total_device_bytes;
  impl_->statistics.memory_budget_bytes = static_cast<std::size_t>(
      options.memory_fraction * static_cast<double>(total_device_bytes));

  if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(GPU2 fields)", error))
    {
      impl_->release();
      return false;
    }
  for (int index = 0; index < 4; ++index)
    if (!cuda_ok(cudaEventCreate(&impl_->events[index]),
                 "cudaEventCreate(GPU2 fields)", error))
      {
        impl_->release();
        return false;
      }

  if (!cufft_ok(cufftCreate(&impl_->plan), "cufftCreate(GPU2 fields)",
                error))
    {
      impl_->release();
      return false;
    }
  impl_->plan_valid = true;
  ++impl_->statistics.plan_creations;
  if (!cufft_ok(cufftSetAutoAllocation(impl_->plan, 0),
                "cufftSetAutoAllocation(GPU2 fields)", error))
    {
      impl_->release();
      return false;
    }

  std::size_t workspace_bytes = 0;
  if (!cufft_ok(cufftMakePlan2d(impl_->plan, 2 * nx, 2 * ny, CUFFT_Z2Z,
                                &workspace_bytes),
                "cufftMakePlan2d(2*nx,2*ny,CUFFT_Z2Z)", error))
    {
      impl_->release();
      return false;
    }
  impl_->statistics.cufft_workspace_bytes = workspace_bytes;

  std::size_t field_required_bytes = 0;
  std::size_t component_bytes = 0;
  if (!checked_product(spectrum_bytes, static_cast<std::size_t>(2),
                       field_required_bytes) ||
      !checked_product(cell_bytes, static_cast<std::size_t>(4),
                       component_bytes) ||
      !checked_sum(field_required_bytes, component_bytes,
                   field_required_bytes) ||
      !checked_sum(field_required_bytes, sizeof(DeviceDepositMetrics),
                   field_required_bytes) ||
      !checked_sum(field_required_bytes, sizeof(DeviceFieldGatherMetrics),
                   field_required_bytes) ||
      !checked_sum(field_required_bytes, workspace_bytes,
                   field_required_bytes))
    {
      error = "GPU2 field allocation total overflows size_t";
      impl_->release();
      return false;
    }
  impl_->statistics.field_required_bytes = field_required_bytes;
  if (field_required_bytes > impl_->statistics.memory_budget_bytes ||
      field_required_bytes > free_device_bytes)
    {
      std::ostringstream message;
      message << "GPU2 field context requires " << field_required_bytes
              << " device bytes before allocation, exceeding "
              << (field_required_bytes > impl_->statistics.memory_budget_bytes
                      ? "gpu2_memory_fraction budget " : "currently free memory ")
              << (field_required_bytes > impl_->statistics.memory_budget_bytes
                      ? impl_->statistics.memory_budget_bytes
                      : free_device_bytes)
              << " bytes; reduce the grid or raise the allowed fraction";
      error = message.str();
      impl_->release();
      return false;
    }

  if (!impl_->allocate_device(reinterpret_cast<void**>(&impl_->spectrum),
                              spectrum_bytes,
                              "cudaMalloc(GPU2 FFT spectrum)", error) ||
      !impl_->allocate_device(reinterpret_cast<void**>(&impl_->dist),
                              spectrum_bytes,
                              "cudaMalloc(GPU2 Green spectrum)", error) ||
      !impl_->allocate_device(reinterpret_cast<void**>(&impl_->rho1),
                              cell_bytes,
                              "cudaMalloc(GPU2 rho1)", error) ||
      !impl_->allocate_device(reinterpret_cast<void**>(&impl_->rho2),
                              cell_bytes,
                              "cudaMalloc(GPU2 rho2)", error) ||
      !impl_->allocate_device(reinterpret_cast<void**>(&impl_->phi1),
                              cell_bytes,
                              "cudaMalloc(GPU2 phi1)", error) ||
      !impl_->allocate_device(reinterpret_cast<void**>(&impl_->phi2),
                              cell_bytes,
                              "cudaMalloc(GPU2 phi2)", error) ||
      !impl_->allocate_device(
          reinterpret_cast<void**>(&impl_->deposit_metrics),
          sizeof(DeviceDepositMetrics),
          "cudaMalloc(GPU2 compact deposition metrics)", error) ||
      !impl_->allocate_device(
          reinterpret_cast<void**>(&impl_->field_gather_metrics),
          sizeof(DeviceFieldGatherMetrics),
          "cudaMalloc(GPU2 compact field-gather metrics)", error) ||
      (workspace_bytes &&
       !impl_->allocate_device(&impl_->workspace, workspace_bytes,
                               "cudaMalloc(GPU2 cuFFT workspace)", error)))
    {
      impl_->release();
      return false;
    }
  if (workspace_bytes)
    {
      if (!cufft_ok(cufftSetWorkArea(impl_->plan, impl_->workspace),
                    "cufftSetWorkArea(GPU2 fields)", error))
        {
          impl_->release();
          return false;
        }
    }
  if (impl_->statistics.resident_bytes != field_required_bytes)
    {
      error = "GPU2 internal resident-memory accounting mismatch";
      impl_->release();
      return false;
    }
  if (!cufft_ok(cufftSetStream(impl_->plan, impl_->stream),
                "cufftSetStream(GPU2 fields)", error))
    {
      impl_->release();
      return false;
    }

  if (options.pinned_io &&
      (!impl_->allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_rho1),
                               cell_bytes,
                               "cudaHostAlloc(GPU2 rho1 staging)", error) ||
       !impl_->allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_rho2),
                               cell_bytes,
                               "cudaHostAlloc(GPU2 rho2 staging)", error) ||
       !impl_->allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_phi1),
                               cell_bytes,
                               "cudaHostAlloc(GPU2 phi1 staging)", error) ||
       !impl_->allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_phi2),
                               cell_bytes,
                               "cudaHostAlloc(GPU2 phi2 staging)", error)))
    {
      impl_->release();
      return false;
    }

  if (!cuda_ok(cudaMemcpyAsync(impl_->dist, dist_interleaved,
                               spectrum_bytes, cudaMemcpyHostToDevice,
                               impl_->stream),
               "cudaMemcpyAsync(GPU2 Green spectrum H2D)", error) ||
      !cuda_ok(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize(GPU2 initialization)", error))
    {
      impl_->release();
      return false;
    }
  impl_->statistics.setup_h2d_bytes = spectrum_bytes;
  impl_->statistics.setup_h2d_operations = 1;
  impl_->statistics.setup_synchronizations = 1;

  std::ostringstream device_description;
  device_description << properties.name << " (device " << options.device
                     << ", compute " << properties.major << '.'
                     << properties.minor << ", "
                     << properties.totalGlobalMem / (1024 * 1024)
                     << " MiB VRAM)";
  impl_->device_text = device_description.str();

  std::ostringstream runtime_description;
  runtime_description
      << "CUDA runtime "
      << version_text(impl_->statistics.cuda_runtime_version)
      << ", CUDA driver "
      << version_text(impl_->statistics.cuda_driver_version)
      << ", cuFFT runtime "
      << version_text(impl_->statistics.cufft_runtime_version)
      << ", CUFFT_Z2Z FP64, plan=2D " << 2 * nx << 'x' << 2 * ny
      << ", explicit_workspace=" << workspace_bytes << " bytes"
      << ", field_resident=" << impl_->statistics.field_required_bytes
      << " bytes, device_total=" << total_device_bytes << " bytes"
      << ", memory_fraction=" << options.memory_fraction
      << ", local_budget=" << impl_->statistics.memory_budget_bytes
      << " bytes"
      << ", pinned_staging=" << (options.pinned_io ? "on" : "off")
      << ", normalization=none";
  impl_->runtime_text = runtime_description.str();
  impl_->statistics.setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - setup_begin).count();
  impl_->initialized = true;
  return true;
}

bool ResidentFieldSolver::initialize(int device,
                                     int nx,
                                     int ny,
                                     const double* dist_interleaved,
                                     bool pinned_io,
                                     std::string& error)
{
  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  return initialize(options, nx, ny, dist_interleaved, error);
}

bool ResidentFieldSolver::is_available() const
{
  return impl_->initialized && !impl_->poisoned;
}

std::string ResidentFieldSolver::device_description() const
{
  if (!impl_->initialized) return "CUDA not initialized";
  return impl_->poisoned ? impl_->device_text + " [POISONED]"
                         : impl_->device_text;
}

std::string ResidentFieldSolver::runtime_description() const
{
  if (!impl_->initialized) return "CUDA/cuFFT runtime not initialized";
  return impl_->poisoned ? impl_->runtime_text + ", context=POISONED"
                         : impl_->runtime_text;
}

ResidentFieldStats ResidentFieldSolver::stats() const
{
  return impl_->statistics;
}

bool ResidentFieldSolver::reserve_particle_capacity(
    std::size_t maximum_particles1,
    std::size_t maximum_particles2,
    std::string& error)
{
  error.clear();
  if (!impl_->initialized)
    {
      error = "GPU2 resident field solver is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error)) return false;
  if (impl_->statistics.calls != 0)
    {
      error = "GPU2 particle capacity must be reserved before the first execution";
      return false;
    }
  // Direct CIC uses native FP64 atomicAdd. Qualify this while pre-collision
  // fallback is still legal, rather than discovering it after host background
  // preparation has advanced the legacy RNG stream.
  if (impl_->compute_major < 6)
    {
      std::ostringstream message;
      message << "GPU2 resident FP64 atomic CIC requires compute capability "
              << "6.0 or newer; selected device is " << impl_->compute_major
              << '.' << impl_->compute_minor;
      error = message.str();
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 capacity reservation)", error))
    {
      impl_->poison_device_context(error);
      return false;
    }

  std::size_t elements1 = 0;
  std::size_t elements2 = 0;
  std::size_t requested_bytes1 = 0;
  std::size_t requested_bytes2 = 0;
  if (!checked_product(maximum_particles1, static_cast<std::size_t>(2),
                       elements1) ||
      !checked_product(maximum_particles2, static_cast<std::size_t>(2),
                       elements2) ||
      !checked_product(elements1, sizeof(float), requested_bytes1) ||
      !checked_product(elements2, sizeof(float), requested_bytes2))
    {
      error = "GPU2 particle reservation overflows size_t byte counts";
      return false;
    }
  const std::size_t old_bytes =
      4 * (impl_->xy_capacity1 + impl_->xy_capacity2) * sizeof(float);
  if (requested_bytes1 > std::numeric_limits<std::size_t>::max() -
                             requested_bytes2)
    {
      error = "GPU2 combined particle reservation overflows size_t";
      return false;
    }
  const std::size_t requested_xy_total = requested_bytes1 + requested_bytes2;
  std::size_t requested_total = 0;
  if (!checked_product(requested_xy_total, static_cast<std::size_t>(2),
                       requested_total))
    {
      error = "GPU2 combined particle/field reservation overflows size_t";
      return false;
    }
  if (impl_->statistics.resident_bytes < old_bytes ||
      requested_total > std::numeric_limits<std::size_t>::max() -
                            (impl_->statistics.resident_bytes - old_bytes))
    {
      error = "GPU2 particle reservation overflows resident memory accounting";
      return false;
    }
  const std::size_t proposed_resident =
      impl_->statistics.resident_bytes - old_bytes + requested_total;
  if (proposed_resident > impl_->statistics.memory_budget_bytes)
    {
      std::ostringstream message;
      message << "GPU2 reservation requires " << proposed_resident
              << " device bytes, exceeding gpu2_memory_fraction budget "
              << impl_->statistics.memory_budget_bytes;
      error = message.str();
      return false;
    }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(GPU2 capacity reservation)", error))
    {
      impl_->poison_device_context(error);
      return false;
    }
  // ensure_particle_capacity allocates each xy/field replacement pair before
  // releasing the old pair, so use the conservative full replacement
  // footprint rather than only the net growth.
  if (requested_total > free_bytes)
    {
      std::ostringstream message;
      message << "GPU2 particle reservation needs up to " << requested_total
              << " temporary device bytes but only " << free_bytes
              << " bytes are free";
      error = message.str();
      return false;
    }

  if (!impl_->ensure_particle_capacity(0, maximum_particles1, error) ||
      !impl_->ensure_particle_capacity(1, maximum_particles2, error))
    return false;
  if (impl_->statistics.pinned_io && !impl_->pinned_deposit_metrics &&
      !impl_->allocate_pinned(
          reinterpret_cast<void**>(&impl_->pinned_deposit_metrics),
          sizeof(DeviceDepositMetrics),
          "cudaHostAlloc(GPU2 compact metrics staging)", error))
    return false;
  if (impl_->statistics.pinned_io && !impl_->pinned_field_gather_metrics &&
      !impl_->allocate_pinned(
          reinterpret_cast<void**>(&impl_->pinned_field_gather_metrics),
          sizeof(DeviceFieldGatherMetrics),
          "cudaHostAlloc(GPU2 compact field metrics staging)", error))
    return false;
  impl_->capacity_reserved = true;
  return true;
}

bool ResidentFieldSolver::solve(const double* rho1,
                                const double* rho2,
                                float charge_sign,
                                double* phi1,
                                double* phi2,
                                ResidentFieldStats& statistics,
                                std::string& error)
{
  error.clear();
  if (!impl_->initialized)
    {
      error = "GPU2 resident field solver is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error))
    {
      statistics = impl_->statistics;
      return false;
    }
  if (!rho1 || !rho2 || !phi1 || !phi2)
    {
      error = "GPU2 field solve received a null rho or phi buffer";
      return false;
    }
  const void* buffers[] = {rho1, rho2, phi1, phi2};
  for (int first = 0; first < 4; ++first)
    for (int second = first + 1; second < 4; ++second)
      if (ranges_overlap(buffers[first], buffers[second], impl_->cell_bytes))
        {
          error = "GPU2 field solve requires four non-overlapping host arrays";
          return false;
        }

  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 solve)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }
  impl_->resident_rho_ready = false;
  impl_->resident_particles_ready = false;
  impl_->resident_particle_count1 = 0;
  impl_->resident_particle_count2 = 0;

  ResidentFieldStats& totals = impl_->statistics;
  totals.last_h2d_ms = 0.0;
  totals.last_kernel_ms = 0.0;
  totals.last_d2h_ms = 0.0;
  totals.last_total_ms = 0.0;
  totals.last_device_work = false;
  totals.last_h2d_bytes = 0;
  totals.last_d2h_bytes = 0;
  totals.last_h2d_operations = 0;
  totals.last_d2h_operations = 0;
  totals.last_synchronizations = 0;

  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();
  // Preserve FIELD::fold_fft's historical zero-field threshold and semantics.
  if (std::fabs(charge_sign) < 1.0e-5f)
    {
      std::fill(phi1, phi1 + impl_->cells, 0.0);
      std::fill(phi2, phi2 + impl_->cells, 0.0);
      totals.last_total_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - total_begin).count();
      totals.total_ms += totals.last_total_ms;
      ++totals.calls;
      ++totals.field_solve_calls;
      statistics = totals;
      return true;
    }

  const double* h2d_rho1 = rho1;
  const double* h2d_rho2 = rho2;
  double* d2h_phi1 = phi1;
  double* d2h_phi2 = phi2;
  if (totals.pinned_io)
    {
      std::memcpy(impl_->pinned_rho1, rho1, impl_->cell_bytes);
      std::memcpy(impl_->pinned_rho2, rho2, impl_->cell_bytes);
      h2d_rho1 = impl_->pinned_rho1;
      h2d_rho2 = impl_->pinned_rho2;
      d2h_phi1 = impl_->pinned_phi1;
      d2h_phi2 = impl_->pinned_phi2;
    }

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 start)", error) ||
      !cuda_ok(cudaMemcpyAsync(impl_->rho1, h2d_rho1, impl_->cell_bytes,
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 rho1 H2D)", error) ||
      !cuda_ok(cudaMemcpyAsync(impl_->rho2, h2d_rho2, impl_->cell_bytes,
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 rho2 H2D)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 after H2D)", error))
    return false;

  if (!cuda_ok(cudaMemsetAsync(impl_->spectrum, 0, impl_->spectrum_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 padded spectrum)", error))
    return false;
  pack_rho_kernel<<<block_count(impl_->cells), kThreads, 0, impl_->stream>>>(
      impl_->rho1, impl_->rho2, impl_->nx, impl_->ny, impl_->padded_ny,
      impl_->spectrum);
  if (!cuda_ok(cudaGetLastError(), "pack_rho_kernel launch", error) ||
      !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum, impl_->spectrum,
                             CUFFT_FORWARD),
                "cufftExecZ2Z(GPU2 forward)", error))
    return false;
  multiply_spectrum_kernel<<<block_count(impl_->padded_cells), kThreads, 0,
                             impl_->stream>>>(
      impl_->spectrum, impl_->dist, impl_->padded_cells);
  if (!cuda_ok(cudaGetLastError(),
               "multiply_spectrum_kernel launch", error) ||
      !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum, impl_->spectrum,
                             CUFFT_INVERSE),
                "cufftExecZ2Z(GPU2 inverse)", error))
    return false;
  unpack_phi_kernel<<<block_count(impl_->cells), kThreads, 0, impl_->stream>>>(
      impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
      impl_->phi1, impl_->phi2);
  if (!cuda_ok(cudaGetLastError(), "unpack_phi_kernel launch", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(GPU2 after kernels)", error) ||
      !cuda_ok(cudaMemcpyAsync(d2h_phi1, impl_->phi1, impl_->cell_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 phi1 D2H)", error) ||
      !cuda_ok(cudaMemcpyAsync(d2h_phi2, impl_->phi2, impl_->cell_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 phi2 D2H)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(GPU2 after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize(GPU2 host boundary)", error))
    return false;
  work_guard.mark_drained();

  if (totals.pinned_io)
    {
      std::memcpy(phi1, impl_->pinned_phi1, impl_->cell_bytes);
      std::memcpy(phi2, impl_->pinned_phi2, impl_->cell_bytes);
    }

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&h2d_ms, impl_->events[0],
                                    impl_->events[1]),
               "cudaEventElapsedTime(GPU2 H2D)", error) ||
      !cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(GPU2 kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(GPU2 D2H)", error))
    return false;

  totals.last_h2d_ms = static_cast<double>(h2d_ms);
  totals.last_kernel_ms = static_cast<double>(kernel_ms);
  totals.last_d2h_ms = static_cast<double>(d2h_ms);
  totals.last_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  totals.last_h2d_bytes = 2 * impl_->cell_bytes;
  totals.last_d2h_bytes = 2 * impl_->cell_bytes;
  totals.last_h2d_operations = 2;
  totals.last_d2h_operations = 2;
  totals.last_synchronizations = 1;
  totals.h2d_ms += totals.last_h2d_ms;
  totals.kernel_ms += totals.last_kernel_ms;
  totals.d2h_ms += totals.last_d2h_ms;
  totals.total_ms += totals.last_total_ms;
  totals.h2d_bytes += totals.last_h2d_bytes;
  totals.d2h_bytes += totals.last_d2h_bytes;
  totals.h2d_operations += totals.last_h2d_operations;
  totals.d2h_operations += totals.last_d2h_operations;
  totals.synchronizations += totals.last_synchronizations;
  totals.last_device_work = true;
  ++totals.device_work_calls;
  ++totals.calls;
  ++totals.field_solve_calls;
  statistics = totals;
  work_guard.complete();
  return true;
}

bool ResidentFieldSolver::deposit_and_solve(
    const float* xy1,
    std::size_t particle_count1,
    double charge1,
    const float* xy2,
    std::size_t particle_count2,
    double charge2,
    float delta_x_inverse,
    float delta_y_inverse,
    float offset_x,
    float offset_y,
    bool four_way_symmetry,
    float charge_sign,
    double* phi1,
    double* phi2,
    ResidentDepositResult& result,
    ResidentFieldStats& statistics,
    std::string& error)
{
  error.clear();
  result = ResidentDepositResult();
  impl_->resident_rho_ready = false;
  impl_->resident_particles_ready = false;
  impl_->resident_particle_count1 = 0;
  impl_->resident_particle_count2 = 0;
  if (!impl_->initialized)
    {
      error = "GPU2 resident CIC/field solver is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error))
    {
      statistics = impl_->statistics;
      return false;
    }
  if (!phi1 || !phi2 || (particle_count1 && !xy1) ||
      (particle_count2 && !xy2))
    {
      error = "GPU2 resident CIC/field solve received a null active buffer";
      return false;
    }
  if (ranges_overlap(phi1, phi2, impl_->cell_bytes))
    {
      error = "GPU2 resident CIC/field solve requires non-overlapping phi arrays";
      return false;
    }
  if (!std::isfinite(charge1) || !std::isfinite(charge2) ||
      !std::isfinite(delta_x_inverse) ||
      !std::isfinite(delta_y_inverse) ||
      !std::isfinite(offset_x) || !std::isfinite(offset_y) ||
      delta_x_inverse == 0.0f || delta_y_inverse == 0.0f)
    {
      error = "GPU2 resident CIC parameters must be finite with nonzero inverse mesh spacing";
      return false;
    }
  if (impl_->compute_major < 6)
    {
      std::ostringstream message;
      message << "GPU2 resident FP64 atomic CIC requires compute capability "
              << "6.0 or newer; selected device is " << impl_->compute_major
              << '.' << impl_->compute_minor;
      error = message.str();
      return false;
    }
  if (!impl_->capacity_reserved)
    {
      error = "GPU2 resident CIC requires reserve_particle_capacity() during preflight";
      return false;
    }
  if (particle_count1 > impl_->xy_capacity1 ||
      particle_count2 > impl_->xy_capacity2)
    {
      std::ostringstream message;
      message << "GPU2 resident CIC particle count exceeds reserved capacity: "
              << particle_count1 << '/' << impl_->xy_capacity1 << " and "
              << particle_count2 << '/' << impl_->xy_capacity2;
      error = message.str();
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 resident CIC solve)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentFieldStats& totals = impl_->statistics;
  totals.last_h2d_ms = 0.0;
  totals.last_kernel_ms = 0.0;
  totals.last_d2h_ms = 0.0;
  totals.last_total_ms = 0.0;
  totals.last_device_work = false;
  totals.last_h2d_bytes = 0;
  totals.last_d2h_bytes = 0;
  totals.last_h2d_operations = 0;
  totals.last_d2h_operations = 0;
  totals.last_synchronizations = 0;

  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();
  const float* h2d_xy1 = xy1;
  const float* h2d_xy2 = xy2;
  if (totals.pinned_io)
    {
      if (particle_count1)
        {
          std::memcpy(impl_->pinned_xy1, xy1,
                      2 * particle_count1 * sizeof(float));
          h2d_xy1 = impl_->pinned_xy1;
        }
      if (particle_count2)
        {
          std::memcpy(impl_->pinned_xy2, xy2,
                      2 * particle_count2 * sizeof(float));
          h2d_xy2 = impl_->pinned_xy2;
        }
    }

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 resident CIC start)", error))
    return false;
  if (particle_count1 &&
      !cuda_ok(cudaMemcpyAsync(impl_->xy1, h2d_xy1,
                               2 * particle_count1 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 xy1 H2D)", error))
    return false;
  if (particle_count2 &&
      !cuda_ok(cudaMemcpyAsync(impl_->xy2, h2d_xy2,
                               2 * particle_count2 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 xy2 H2D)", error))
    return false;
  if (!cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 resident CIC after H2D)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->rho1, 0, impl_->cell_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 resident rho1)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->rho2, 0, impl_->cell_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 resident rho2)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->deposit_metrics, 0,
                               sizeof(DeviceDepositMetrics), impl_->stream),
               "cudaMemsetAsync(GPU2 compact metrics)", error))
    return false;

  if (particle_count1)
    resident_cic_kernel<<<block_count(particle_count1), kThreads, 0,
                          impl_->stream>>>(
        impl_->xy1, particle_count1, charge1, 0, impl_->nx, impl_->ny,
        delta_x_inverse, delta_y_inverse, offset_x, offset_y,
        impl_->rho1, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "resident_cic_kernel beam1 launch", error))
    return false;
  if (particle_count2)
    resident_cic_kernel<<<block_count(particle_count2), kThreads, 0,
                          impl_->stream>>>(
        impl_->xy2, particle_count2, charge2, 1, impl_->nx, impl_->ny,
        delta_x_inverse, delta_y_inverse, offset_x, offset_y,
        impl_->rho2, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "resident_cic_kernel beam2 launch", error))
    return false;

  const std::size_t symmetry_representatives =
      static_cast<std::size_t>(impl_->nx / 2) * (impl_->ny / 2);
  if (four_way_symmetry && symmetry_representatives)
    {
      symmetrize_four_way_kernel<<<block_count(symmetry_representatives),
                                    kThreads, 0, impl_->stream>>>(
          impl_->rho1, impl_->rho2, impl_->nx, impl_->ny);
      if (!cuda_ok(cudaGetLastError(),
                   "symmetrize_four_way_kernel launch", error))
        return false;
    }
  fine_lumi_dot_kernel<<<1, kThreads, 0, impl_->stream>>>(
      impl_->rho1, impl_->rho2, impl_->cells, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "fine_lumi_dot_kernel launch", error))
    return false;

  const bool zero_field = std::fabs(charge_sign) < 1.0e-5f;
  if (!zero_field)
    {
      if (!cuda_ok(cudaMemsetAsync(impl_->spectrum, 0,
                                   impl_->spectrum_bytes, impl_->stream),
                   "cudaMemsetAsync(GPU2 resident padded spectrum)", error))
        return false;
      pack_rho_kernel<<<block_count(impl_->cells), kThreads, 0,
                        impl_->stream>>>(
          impl_->rho1, impl_->rho2, impl_->nx, impl_->ny,
          impl_->padded_ny, impl_->spectrum);
      if (!cuda_ok(cudaGetLastError(),
                   "pack resident rho kernel launch", error) ||
          !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                 impl_->spectrum, CUFFT_FORWARD),
                    "cufftExecZ2Z(GPU2 resident forward)", error))
        return false;
      multiply_spectrum_kernel<<<block_count(impl_->padded_cells), kThreads,
                                 0, impl_->stream>>>(
          impl_->spectrum, impl_->dist, impl_->padded_cells);
      if (!cuda_ok(cudaGetLastError(),
                   "multiply resident spectrum kernel launch", error) ||
          !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                 impl_->spectrum, CUFFT_INVERSE),
                    "cufftExecZ2Z(GPU2 resident inverse)", error))
        return false;
      unpack_phi_kernel<<<block_count(impl_->cells), kThreads, 0,
                          impl_->stream>>>(
          impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
          impl_->phi1, impl_->phi2);
      if (!cuda_ok(cudaGetLastError(),
                   "unpack resident phi kernel launch", error))
        return false;
    }

  if (!cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(GPU2 resident after kernels)", error))
    return false;
  DeviceDepositMetrics host_metrics = {};
  DeviceDepositMetrics* d2h_metrics = totals.pinned_io
      ? impl_->pinned_deposit_metrics : &host_metrics;
  double* d2h_phi1 = totals.pinned_io ? impl_->pinned_phi1 : phi1;
  double* d2h_phi2 = totals.pinned_io ? impl_->pinned_phi2 : phi2;
  if (!cuda_ok(cudaMemcpyAsync(d2h_metrics, impl_->deposit_metrics,
                               sizeof(DeviceDepositMetrics),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 compact metrics D2H)", error))
    return false;
  if (!zero_field &&
      (!cuda_ok(cudaMemcpyAsync(d2h_phi1, impl_->phi1, impl_->cell_bytes,
                                cudaMemcpyDeviceToHost, impl_->stream),
                "cudaMemcpyAsync(GPU2 resident phi1 D2H)", error) ||
       !cuda_ok(cudaMemcpyAsync(d2h_phi2, impl_->phi2, impl_->cell_bytes,
                                cudaMemcpyDeviceToHost, impl_->stream),
                "cudaMemcpyAsync(GPU2 resident phi2 D2H)", error)))
    return false;
  if (!cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(GPU2 resident after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize(GPU2 resident host boundary)", error))
    return false;
  work_guard.mark_drained();

  if (totals.pinned_io)
    {
      host_metrics = *impl_->pinned_deposit_metrics;
      if (!zero_field)
        {
          std::memcpy(phi1, impl_->pinned_phi1, impl_->cell_bytes);
          std::memcpy(phi2, impl_->pinned_phi2, impl_->cell_bytes);
        }
    }
  if (zero_field)
    {
      std::fill(phi1, phi1 + impl_->cells, 0.0);
      std::fill(phi2, phi2 + impl_->cells, 0.0);
    }

  result.beam1_in = static_cast<std::size_t>(host_metrics.beam1_in);
  result.beam1_out = static_cast<std::size_t>(host_metrics.beam1_out);
  result.beam2_in = static_cast<std::size_t>(host_metrics.beam2_in);
  result.beam2_out = static_cast<std::size_t>(host_metrics.beam2_out);
  result.fine_lumi_dot = host_metrics.fine_lumi_dot;

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&h2d_ms, impl_->events[0],
                                    impl_->events[1]),
               "cudaEventElapsedTime(GPU2 resident H2D)", error) ||
      !cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(GPU2 resident kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(GPU2 resident D2H)", error))
    return false;

  std::size_t xy_bytes1 = 0;
  std::size_t xy_bytes2 = 0;
  checked_product(2 * particle_count1, sizeof(float), xy_bytes1);
  checked_product(2 * particle_count2, sizeof(float), xy_bytes2);
  totals.last_h2d_ms = static_cast<double>(h2d_ms);
  totals.last_kernel_ms = static_cast<double>(kernel_ms);
  totals.last_d2h_ms = static_cast<double>(d2h_ms);
  totals.last_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  totals.last_h2d_bytes = xy_bytes1 + xy_bytes2;
  totals.last_d2h_bytes = sizeof(DeviceDepositMetrics) +
      (zero_field ? 0 : 2 * impl_->cell_bytes);
  totals.last_h2d_operations =
      static_cast<std::uint64_t>(particle_count1 != 0) +
      static_cast<std::uint64_t>(particle_count2 != 0);
  totals.last_d2h_operations = zero_field ? 1 : 3;
  totals.last_synchronizations = 1;
  totals.h2d_ms += totals.last_h2d_ms;
  totals.kernel_ms += totals.last_kernel_ms;
  totals.d2h_ms += totals.last_d2h_ms;
  totals.total_ms += totals.last_total_ms;
  totals.h2d_bytes += totals.last_h2d_bytes;
  totals.d2h_bytes += totals.last_d2h_bytes;
  totals.h2d_operations += totals.last_h2d_operations;
  totals.d2h_operations += totals.last_d2h_operations;
  totals.synchronizations += totals.last_synchronizations;
  totals.last_device_work = true;
  ++totals.device_work_calls;
  ++totals.calls;
  ++totals.deposition_calls;
  ++totals.field_solve_calls;
  impl_->resident_rho_ready = true;
  impl_->resident_particles_ready = true;
  impl_->resident_particle_count1 = particle_count1;
  impl_->resident_particle_count2 = particle_count2;
  impl_->resident_delta_x_inverse = delta_x_inverse;
  impl_->resident_delta_y_inverse = delta_y_inverse;
  impl_->resident_offset_x = offset_x;
  impl_->resident_offset_y = offset_y;
  statistics = totals;
  work_guard.complete();
  return true;
}

bool ResidentFieldSolver::deposit(const float* xy1,
                                  std::size_t particle_count1,
                                  double charge1,
                                  const float* xy2,
                                  std::size_t particle_count2,
                                  double charge2,
                                  float delta_x_inverse,
                                  float delta_y_inverse,
                                  float offset_x,
                                  float offset_y,
                                  bool four_way_symmetry,
                                  ResidentDepositResult& result,
                                  ResidentFieldStats& statistics,
                                  std::string& error)
{
  error.clear();
  result = ResidentDepositResult();
  impl_->resident_rho_ready = false;
  impl_->resident_particles_ready = false;
  impl_->resident_particle_count1 = 0;
  impl_->resident_particle_count2 = 0;
  if (!impl_->initialized)
    {
      error = "GPU2 resident CIC is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error))
    {
      statistics = impl_->statistics;
      return false;
    }
  if ((particle_count1 && !xy1) || (particle_count2 && !xy2))
    {
      error = "GPU2 resident CIC received a null active xy buffer";
      return false;
    }
  if (!std::isfinite(charge1) || !std::isfinite(charge2) ||
      !std::isfinite(delta_x_inverse) ||
      !std::isfinite(delta_y_inverse) ||
      !std::isfinite(offset_x) || !std::isfinite(offset_y) ||
      delta_x_inverse == 0.0f || delta_y_inverse == 0.0f)
    {
      error = "GPU2 resident CIC parameters must be finite with nonzero inverse mesh spacing";
      return false;
    }
  if (impl_->compute_major < 6)
    {
      std::ostringstream message;
      message << "GPU2 resident FP64 atomic CIC requires compute capability "
              << "6.0 or newer; selected device is " << impl_->compute_major
              << '.' << impl_->compute_minor;
      error = message.str();
      return false;
    }
  if (!impl_->capacity_reserved)
    {
      error = "GPU2 resident CIC requires reserve_particle_capacity() during preflight";
      return false;
    }
  if (particle_count1 > impl_->xy_capacity1 ||
      particle_count2 > impl_->xy_capacity2)
    {
      std::ostringstream message;
      message << "GPU2 resident CIC particle count exceeds reserved capacity: "
              << particle_count1 << '/' << impl_->xy_capacity1 << " and "
              << particle_count2 << '/' << impl_->xy_capacity2;
      error = message.str();
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 resident deposit)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentFieldStats& totals = impl_->statistics;
  totals.last_h2d_ms = 0.0;
  totals.last_kernel_ms = 0.0;
  totals.last_d2h_ms = 0.0;
  totals.last_total_ms = 0.0;
  totals.last_device_work = false;
  totals.last_h2d_bytes = 0;
  totals.last_d2h_bytes = 0;
  totals.last_h2d_operations = 0;
  totals.last_d2h_operations = 0;
  totals.last_synchronizations = 0;
  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();

  const float* h2d_xy1 = xy1;
  const float* h2d_xy2 = xy2;
  if (totals.pinned_io)
    {
      if (particle_count1)
        {
          std::memcpy(impl_->pinned_xy1, xy1,
                      2 * particle_count1 * sizeof(float));
          h2d_xy1 = impl_->pinned_xy1;
        }
      if (particle_count2)
        {
          std::memcpy(impl_->pinned_xy2, xy2,
                      2 * particle_count2 * sizeof(float));
          h2d_xy2 = impl_->pinned_xy2;
        }
    }

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 deposit start)", error))
    return false;
  if (particle_count1 &&
      !cuda_ok(cudaMemcpyAsync(impl_->xy1, h2d_xy1,
                               2 * particle_count1 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 deposit xy1 H2D)", error))
    return false;
  if (particle_count2 &&
      !cuda_ok(cudaMemcpyAsync(impl_->xy2, h2d_xy2,
                               2 * particle_count2 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(GPU2 deposit xy2 H2D)", error))
    return false;
  if (!cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 deposit after H2D)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->rho1, 0, impl_->cell_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 deposit rho1)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->rho2, 0, impl_->cell_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 deposit rho2)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->deposit_metrics, 0,
                               sizeof(DeviceDepositMetrics), impl_->stream),
               "cudaMemsetAsync(GPU2 deposit metrics)", error))
    return false;

  if (particle_count1)
    resident_cic_kernel<<<block_count(particle_count1), kThreads, 0,
                          impl_->stream>>>(
        impl_->xy1, particle_count1, charge1, 0, impl_->nx, impl_->ny,
        delta_x_inverse, delta_y_inverse, offset_x, offset_y,
        impl_->rho1, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "resident_cic_kernel split beam1 launch", error))
    return false;
  if (particle_count2)
    resident_cic_kernel<<<block_count(particle_count2), kThreads, 0,
                          impl_->stream>>>(
        impl_->xy2, particle_count2, charge2, 1, impl_->nx, impl_->ny,
        delta_x_inverse, delta_y_inverse, offset_x, offset_y,
        impl_->rho2, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "resident_cic_kernel split beam2 launch", error))
    return false;
  const std::size_t symmetry_representatives =
      static_cast<std::size_t>(impl_->nx / 2) * (impl_->ny / 2);
  if (four_way_symmetry && symmetry_representatives)
    {
      symmetrize_four_way_kernel<<<block_count(symmetry_representatives),
                                    kThreads, 0, impl_->stream>>>(
          impl_->rho1, impl_->rho2, impl_->nx, impl_->ny);
      if (!cuda_ok(cudaGetLastError(),
                   "split symmetrize_four_way_kernel launch", error))
        return false;
    }
  fine_lumi_dot_kernel<<<1, kThreads, 0, impl_->stream>>>(
      impl_->rho1, impl_->rho2, impl_->cells, impl_->deposit_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "split fine_lumi_dot_kernel launch", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(GPU2 deposit after kernels)", error))
    return false;

  DeviceDepositMetrics host_metrics = {};
  DeviceDepositMetrics* d2h_metrics = totals.pinned_io
      ? impl_->pinned_deposit_metrics : &host_metrics;
  if (!cuda_ok(cudaMemcpyAsync(d2h_metrics, impl_->deposit_metrics,
                               sizeof(DeviceDepositMetrics),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 split metrics D2H)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(GPU2 deposit after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize(GPU2 deposit host boundary)", error))
    return false;
  work_guard.mark_drained();
  if (totals.pinned_io) host_metrics = *impl_->pinned_deposit_metrics;

  result.beam1_in = static_cast<std::size_t>(host_metrics.beam1_in);
  result.beam1_out = static_cast<std::size_t>(host_metrics.beam1_out);
  result.beam2_in = static_cast<std::size_t>(host_metrics.beam2_in);
  result.beam2_out = static_cast<std::size_t>(host_metrics.beam2_out);
  result.fine_lumi_dot = host_metrics.fine_lumi_dot;

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&h2d_ms, impl_->events[0],
                                    impl_->events[1]),
               "cudaEventElapsedTime(GPU2 split deposit H2D)", error) ||
      !cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(GPU2 split deposit kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(GPU2 split deposit D2H)", error))
    return false;
  const std::size_t xy_bytes1 = 2 * particle_count1 * sizeof(float);
  const std::size_t xy_bytes2 = 2 * particle_count2 * sizeof(float);
  totals.last_h2d_ms = static_cast<double>(h2d_ms);
  totals.last_kernel_ms = static_cast<double>(kernel_ms);
  totals.last_d2h_ms = static_cast<double>(d2h_ms);
  totals.last_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  totals.last_h2d_bytes = xy_bytes1 + xy_bytes2;
  totals.last_d2h_bytes = sizeof(DeviceDepositMetrics);
  totals.last_h2d_operations =
      static_cast<std::uint64_t>(particle_count1 != 0) +
      static_cast<std::uint64_t>(particle_count2 != 0);
  totals.last_d2h_operations = 1;
  totals.last_synchronizations = 1;
  totals.h2d_ms += totals.last_h2d_ms;
  totals.kernel_ms += totals.last_kernel_ms;
  totals.d2h_ms += totals.last_d2h_ms;
  totals.total_ms += totals.last_total_ms;
  totals.h2d_bytes += totals.last_h2d_bytes;
  totals.d2h_bytes += totals.last_d2h_bytes;
  totals.h2d_operations += totals.last_h2d_operations;
  totals.d2h_operations += totals.last_d2h_operations;
  totals.synchronizations += totals.last_synchronizations;
  totals.last_device_work = true;
  ++totals.device_work_calls;
  ++totals.calls;
  ++totals.deposition_calls;
  impl_->resident_rho_ready = true;
  impl_->resident_particles_ready = true;
  impl_->resident_particle_count1 = particle_count1;
  impl_->resident_particle_count2 = particle_count2;
  impl_->resident_delta_x_inverse = delta_x_inverse;
  impl_->resident_delta_y_inverse = delta_y_inverse;
  impl_->resident_offset_x = offset_x;
  impl_->resident_offset_y = offset_y;
  statistics = totals;
  work_guard.complete();
  return true;
}

bool ResidentFieldSolver::solve_deposited(float charge_sign,
                                          double* phi1,
                                          double* phi2,
                                          ResidentFieldStats& statistics,
                                          std::string& error)
{
  error.clear();
  if (!impl_->initialized)
    {
      error = "GPU2 solve_deposited is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error))
    {
      statistics = impl_->statistics;
      return false;
    }
  if (!impl_->resident_rho_ready)
    {
      error = "GPU2 solve_deposited requires a successful resident deposit";
      return false;
    }
  if (!phi1 || !phi2 || ranges_overlap(phi1, phi2, impl_->cell_bytes))
    {
      error = "GPU2 solve_deposited requires two non-overlapping phi arrays";
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 solve_deposited)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentFieldStats& totals = impl_->statistics;
  totals.last_h2d_ms = 0.0;
  totals.last_kernel_ms = 0.0;
  totals.last_d2h_ms = 0.0;
  totals.last_total_ms = 0.0;
  totals.last_device_work = false;
  totals.last_h2d_bytes = 0;
  totals.last_d2h_bytes = 0;
  totals.last_h2d_operations = 0;
  totals.last_d2h_operations = 0;
  totals.last_synchronizations = 0;
  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();
  if (std::fabs(charge_sign) < 1.0e-5f)
    {
      std::fill(phi1, phi1 + impl_->cells, 0.0);
      std::fill(phi2, phi2 + impl_->cells, 0.0);
      totals.last_total_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - total_begin).count();
      totals.total_ms += totals.last_total_ms;
      ++totals.calls;
      ++totals.field_solve_calls;
      statistics = totals;
      return true;
    }

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 solve_deposited start)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 solve_deposited after H2D)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->spectrum, 0, impl_->spectrum_bytes,
                               impl_->stream),
               "cudaMemsetAsync(GPU2 deposited padded spectrum)", error))
    return false;
  pack_rho_kernel<<<block_count(impl_->cells), kThreads, 0, impl_->stream>>>(
      impl_->rho1, impl_->rho2, impl_->nx, impl_->ny,
      impl_->padded_ny, impl_->spectrum);
  if (!cuda_ok(cudaGetLastError(),
               "pack deposited rho kernel launch", error) ||
      !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                             impl_->spectrum, CUFFT_FORWARD),
                "cufftExecZ2Z(GPU2 deposited forward)", error))
    return false;
  multiply_spectrum_kernel<<<block_count(impl_->padded_cells), kThreads, 0,
                             impl_->stream>>>(
      impl_->spectrum, impl_->dist, impl_->padded_cells);
  if (!cuda_ok(cudaGetLastError(),
               "multiply deposited spectrum kernel launch", error) ||
      !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                             impl_->spectrum, CUFFT_INVERSE),
                "cufftExecZ2Z(GPU2 deposited inverse)", error))
    return false;
  unpack_phi_kernel<<<block_count(impl_->cells), kThreads, 0, impl_->stream>>>(
      impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
      impl_->phi1, impl_->phi2);
  if (!cuda_ok(cudaGetLastError(),
               "unpack deposited phi kernel launch", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(GPU2 solve_deposited after kernels)", error))
    return false;

  double* d2h_phi1 = totals.pinned_io ? impl_->pinned_phi1 : phi1;
  double* d2h_phi2 = totals.pinned_io ? impl_->pinned_phi2 : phi2;
  if (!cuda_ok(cudaMemcpyAsync(d2h_phi1, impl_->phi1, impl_->cell_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 deposited phi1 D2H)", error) ||
      !cuda_ok(cudaMemcpyAsync(d2h_phi2, impl_->phi2, impl_->cell_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 deposited phi2 D2H)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(GPU2 solve_deposited after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize(GPU2 solve_deposited host boundary)",
               error))
    return false;
  work_guard.mark_drained();
  if (totals.pinned_io)
    {
      std::memcpy(phi1, impl_->pinned_phi1, impl_->cell_bytes);
      std::memcpy(phi2, impl_->pinned_phi2, impl_->cell_bytes);
    }

  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(GPU2 deposited kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(GPU2 deposited D2H)", error))
    return false;
  totals.last_kernel_ms = static_cast<double>(kernel_ms);
  totals.last_d2h_ms = static_cast<double>(d2h_ms);
  totals.last_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  totals.last_d2h_bytes = 2 * impl_->cell_bytes;
  totals.last_d2h_operations = 2;
  totals.last_synchronizations = 1;
  totals.kernel_ms += totals.last_kernel_ms;
  totals.d2h_ms += totals.last_d2h_ms;
  totals.total_ms += totals.last_total_ms;
  totals.d2h_bytes += totals.last_d2h_bytes;
  totals.d2h_operations += totals.last_d2h_operations;
  totals.synchronizations += totals.last_synchronizations;
  totals.last_device_work = true;
  ++totals.device_work_calls;
  ++totals.calls;
  ++totals.field_solve_calls;
  statistics = totals;
  work_guard.complete();
  return true;
}

bool ResidentFieldSolver::solve_deposited_and_gather(
    float charge_sign,
    float physical_min_x,
    float physical_max_x,
    float physical_min_y,
    float physical_max_y,
    float* field_xy1_interleaved,
    std::size_t output_capacity1,
    float* field_xy2_interleaved,
    std::size_t output_capacity2,
    ResidentFieldGatherResult& result,
    ResidentFieldStats& statistics,
    std::string& error)
{
  error.clear();
  if (!impl_->initialized)
    {
      error = "GPU2 compact field gather is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error)) return false;
  if (!impl_->resident_rho_ready || !impl_->resident_particles_ready)
    {
      error = "GPU2 compact field gather requires a successful resident particle deposit";
      return false;
    }
  if (!std::isfinite(physical_min_x) ||
      !std::isfinite(physical_max_x) ||
      !std::isfinite(physical_min_y) ||
      !std::isfinite(physical_max_y) ||
      !(physical_min_x < physical_max_x) ||
      !(physical_min_y < physical_max_y))
    {
      error = "GPU2 compact field gather requires finite, strictly ordered physical bounds";
      return false;
    }
  const std::size_t particle_count1 = impl_->resident_particle_count1;
  const std::size_t particle_count2 = impl_->resident_particle_count2;
  if (output_capacity1 < particle_count1 ||
      output_capacity2 < particle_count2)
    {
      std::ostringstream message;
      message << "GPU2 compact field output capacity is too small: "
              << output_capacity1 << '/' << particle_count1 << " and "
              << output_capacity2 << '/' << particle_count2;
      error = message.str();
      return false;
    }
  if ((particle_count1 && !field_xy1_interleaved) ||
      (particle_count2 && !field_xy2_interleaved))
    {
      error = "GPU2 compact field gather received a null active output buffer";
      return false;
    }
  std::size_t field_bytes1 = 0;
  std::size_t field_bytes2 = 0;
  std::size_t elements1 = 0;
  std::size_t elements2 = 0;
  if (!checked_product(particle_count1, static_cast<std::size_t>(2),
                       elements1) ||
      !checked_product(particle_count2, static_cast<std::size_t>(2),
                       elements2) ||
      !checked_product(elements1, sizeof(float), field_bytes1) ||
      !checked_product(elements2, sizeof(float), field_bytes2))
    {
      error = "GPU2 compact field output byte count overflows size_t";
      return false;
    }
  if (ranges_overlap(field_xy1_interleaved, field_bytes1,
                     field_xy2_interleaved, field_bytes2))
    {
      error = "GPU2 compact field gather requires non-overlapping output arrays";
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 compact field gather)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentFieldStats& totals = impl_->statistics;
  totals.last_h2d_ms = 0.0;
  totals.last_kernel_ms = 0.0;
  totals.last_d2h_ms = 0.0;
  totals.last_total_ms = 0.0;
  totals.last_device_work = false;
  totals.last_h2d_bytes = 0;
  totals.last_d2h_bytes = 0;
  totals.last_h2d_operations = 0;
  totals.last_d2h_operations = 0;
  totals.last_synchronizations = 0;
  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();
  const bool zero_field = std::fabs(charge_sign) < 1.0e-5f;

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 compact gather start)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 compact gather after H2D)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->field_gather_metrics, 0,
                               sizeof(DeviceFieldGatherMetrics),
                               impl_->stream),
               "cudaMemsetAsync(GPU2 compact field metrics)", error))
    return false;

  if (!zero_field)
    {
      if (!cuda_ok(cudaMemsetAsync(impl_->spectrum, 0,
                                   impl_->spectrum_bytes, impl_->stream),
                   "cudaMemsetAsync(GPU2 compact padded spectrum)", error))
        return false;
      pack_rho_kernel<<<block_count(impl_->cells), kThreads, 0,
                        impl_->stream>>>(
          impl_->rho1, impl_->rho2, impl_->nx, impl_->ny,
          impl_->padded_ny, impl_->spectrum);
      if (!cuda_ok(cudaGetLastError(),
                   "pack compact-gather rho kernel launch", error) ||
          !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                 impl_->spectrum, CUFFT_FORWARD),
                    "cufftExecZ2Z(GPU2 compact forward)", error))
        return false;
      multiply_spectrum_kernel<<<block_count(impl_->padded_cells), kThreads,
                                 0, impl_->stream>>>(
          impl_->spectrum, impl_->dist, impl_->padded_cells);
      if (!cuda_ok(cudaGetLastError(),
                   "multiply compact-gather spectrum kernel launch", error) ||
          !cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                 impl_->spectrum, CUFFT_INVERSE),
                    "cufftExecZ2Z(GPU2 compact inverse)", error))
        return false;
      unpack_phi_kernel<<<block_count(impl_->cells), kThreads, 0,
                          impl_->stream>>>(
          impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
          impl_->phi1, impl_->phi2);
      if (!cuda_ok(cudaGetLastError(),
                   "unpack compact-gather phi kernel launch", error))
        return false;
    }

  if (particle_count1)
    gather_opposing_field_kernel<<<block_count(particle_count1), kThreads, 0,
                                    impl_->stream>>>(
        impl_->xy1, particle_count1, impl_->phi2, 0, impl_->nx, impl_->ny,
        impl_->resident_delta_x_inverse, impl_->resident_delta_y_inverse,
        impl_->resident_offset_x, impl_->resident_offset_y,
        physical_min_x, physical_max_x, physical_min_y, physical_max_y,
        zero_field,
        impl_->fields1, impl_->field_gather_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "compact opposing-field beam1 kernel launch", error))
    return false;
  if (particle_count2)
    gather_opposing_field_kernel<<<block_count(particle_count2), kThreads, 0,
                                    impl_->stream>>>(
        impl_->xy2, particle_count2, impl_->phi1, 1, impl_->nx, impl_->ny,
        impl_->resident_delta_x_inverse, impl_->resident_delta_y_inverse,
        impl_->resident_offset_x, impl_->resident_offset_y,
        physical_min_x, physical_max_x, physical_min_y, physical_max_y,
        zero_field,
        impl_->fields2, impl_->field_gather_metrics);
  if (!cuda_ok(cudaGetLastError(),
               "compact opposing-field beam2 kernel launch", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(GPU2 compact gather after kernels)", error))
    return false;

  DeviceFieldGatherMetrics host_metrics = {};
  DeviceFieldGatherMetrics* d2h_metrics = totals.pinned_io
      ? impl_->pinned_field_gather_metrics : &host_metrics;
  float* d2h_fields1 = totals.pinned_io
      ? impl_->pinned_fields1 : field_xy1_interleaved;
  float* d2h_fields2 = totals.pinned_io
      ? impl_->pinned_fields2 : field_xy2_interleaved;
  if (!cuda_ok(cudaMemcpyAsync(d2h_metrics, impl_->field_gather_metrics,
                               sizeof(DeviceFieldGatherMetrics),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 compact field metrics D2H)", error))
    return false;
  if (field_bytes1 &&
      !cuda_ok(cudaMemcpyAsync(d2h_fields1, impl_->fields1, field_bytes1,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 compact fields1 D2H)", error))
    return false;
  if (field_bytes2 &&
      !cuda_ok(cudaMemcpyAsync(d2h_fields2, impl_->fields2, field_bytes2,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(GPU2 compact fields2 D2H)", error))
    return false;
  if (!cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(GPU2 compact gather after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize(GPU2 compact gather host boundary)",
               error))
    return false;
  work_guard.mark_drained();

  if (totals.pinned_io)
    {
      host_metrics = *impl_->pinned_field_gather_metrics;
      if (field_bytes1)
        std::memcpy(field_xy1_interleaved, impl_->pinned_fields1,
                    field_bytes1);
      if (field_bytes2)
        std::memcpy(field_xy2_interleaved, impl_->pinned_fields2,
                    field_bytes2);
    }
  ResidentFieldGatherResult successful_result;
  successful_result.beam1_in =
      static_cast<std::size_t>(host_metrics.beam1_in);
  successful_result.beam1_out =
      static_cast<std::size_t>(host_metrics.beam1_out);
  successful_result.beam2_in =
      static_cast<std::size_t>(host_metrics.beam2_in);
  successful_result.beam2_out =
      static_cast<std::size_t>(host_metrics.beam2_out);

  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(GPU2 compact gather kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(GPU2 compact gather D2H)", error))
    return false;
  totals.last_kernel_ms = static_cast<double>(kernel_ms);
  totals.last_d2h_ms = static_cast<double>(d2h_ms);
  totals.last_total_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  totals.last_d2h_bytes = sizeof(DeviceFieldGatherMetrics) +
                          field_bytes1 + field_bytes2;
  totals.last_d2h_operations = 1 +
      static_cast<std::uint64_t>(field_bytes1 != 0) +
      static_cast<std::uint64_t>(field_bytes2 != 0);
  totals.last_synchronizations = 1;
  totals.kernel_ms += totals.last_kernel_ms;
  totals.d2h_ms += totals.last_d2h_ms;
  totals.total_ms += totals.last_total_ms;
  totals.d2h_bytes += totals.last_d2h_bytes;
  totals.d2h_operations += totals.last_d2h_operations;
  totals.synchronizations += totals.last_synchronizations;
  totals.last_device_work = true;
  ++totals.device_work_calls;
  ++totals.calls;
  ++totals.field_solve_calls;
  ++totals.field_gather_calls;
  result = successful_result;
  statistics = totals;
  work_guard.complete();
  return true;
}

#if defined(FEEBL_GPU_RESIDENT_TEST_HOOKS)
bool ResidentFieldSolver::inject_compact_gather_failure_for_test(
    ResidentFieldStats& statistics,
    std::string& error)
{
  error.clear();
  if (!impl_->initialized)
    {
      error = "GPU2 compact gather failure injection is not initialized";
      return false;
    }
  if (impl_->reject_if_poisoned(error)) return false;
  if (!impl_->resident_rho_ready || !impl_->resident_particles_ready)
    {
      error = "GPU2 compact gather failure injection requires a successful "
              "resident particle deposit";
      return false;
    }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(GPU2 compact gather failure injection)", error))
    {
      impl_->poison_device_context(error);
      statistics = impl_->statistics;
      return false;
    }

  Impl::DeviceWorkGuard work_guard(*impl_, error, &statistics);
  work_guard.arm();
  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(GPU2 injected compact gather start)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(GPU2 injected compact gather after H2D)",
               error) ||
      !cuda_ok(cudaMemsetAsync(impl_->field_gather_metrics, 0,
                               sizeof(DeviceFieldGatherMetrics),
                               impl_->stream),
               "cudaMemsetAsync(GPU2 injected compact gather metrics)",
               error))
    return false;

  error = "injected compact-gather device failure after valid queued work";
  // The armed guard deliberately observes this unsuccessful return, drains
  // the still-valid stream, and poisons the solver.  No caller output has
  // been touched and no stale CUDA handle is ever dereferenced.
  return false;
}
#endif

}  // namespace gpu
}  // namespace feebl
