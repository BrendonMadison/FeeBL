#include "gpuWaveBatch.h"

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

struct DeviceEncounterMetrics {
  unsigned long long beam1_in;
  unsigned long long beam1_out;
  unsigned long long beam2_in;
  unsigned long long beam2_out;
  unsigned long long gather_beam1_in;
  unsigned long long gather_beam1_out;
  unsigned long long gather_beam2_in;
  unsigned long long gather_beam2_out;
  double fine_lumi_dot;
};

bool checked_product(std::size_t a, std::size_t b, std::size_t& result) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
  result = a * b;
  return true;
}

int blocks_for(std::size_t work) {
  if (!work) return 1;
  return static_cast<int>(std::min<std::size_t>(
      65535, (work + static_cast<std::size_t>(kThreads) - 1) /
                 static_cast<std::size_t>(kThreads)));
}

bool cuda_ok(cudaError_t status, const char* operation, std::string& error) {
  if (status == cudaSuccess) return true;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status)
          << " (CUDA error " << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

bool cufft_ok(cufftResult status, const char* operation, std::string& error) {
  if (status == CUFFT_SUCCESS) return true;
  std::ostringstream message;
  message << operation << " failed (cuFFT error "
          << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

__device__ bool locate_cic(float x, float y, float inv_x, float inv_y,
                           float offset_x, float offset_y, int nx, int ny,
                           int& ix, int& iy, float& hx, float& hy) {
  hx = x * inv_x + (offset_x - 0.5f);
  ix = static_cast<int>(floorf(hx));
  hx -= static_cast<float>(ix);
  hy = y * inv_y + (offset_y - 0.5f);
  iy = static_cast<int>(floorf(hy));
  hy -= static_cast<float>(iy);
  return ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1;
}

__global__ void batch_cic_kernel(
    const float* xy, const unsigned long long* counts,
    std::size_t maximum_particles, double charge, int beam,
    int nx, int ny, float inv_x, float inv_y, float offset_x, float offset_y,
    std::size_t cells, double* rho, DeviceEncounterMetrics* metrics) {
  const std::size_t slot = blockIdx.y;
  const std::size_t count = static_cast<std::size_t>(counts[slot]);
  const std::size_t particle_stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const float* slot_xy = xy + 2 * slot * maximum_particles;
  double* slot_rho = rho + slot * cells;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < count; particle += particle_stride) {
    int ix = 0;
    int iy = 0;
    float hx = 0.0f;
    float hy = 0.0f;
    if (!locate_cic(slot_xy[2 * particle], slot_xy[2 * particle + 1],
                    inv_x, inv_y, offset_x, offset_y, nx, ny,
                    ix, iy, hx, hy)) {
      if (beam == 0) atomicAdd(&metrics[slot].beam1_out, 1ULL);
      else atomicAdd(&metrics[slot].beam2_out, 1ULL);
      continue;
    }
    if (beam == 0) atomicAdd(&metrics[slot].beam1_in, 1ULL);
    else atomicAdd(&metrics[slot].beam2_in, 1ULL);
    const double dx = static_cast<double>(hx);
    const double dy = static_cast<double>(hy);
    const std::size_t base = static_cast<std::size_t>(ix) * ny + iy;
    atomicAdd(slot_rho + base, (1.0 - dx) * (1.0 - dy) * charge);
    atomicAdd(slot_rho + base + ny, dx * (1.0 - dy) * charge);
    atomicAdd(slot_rho + base + 1, (1.0 - dx) * dy * charge);
    atomicAdd(slot_rho + base + ny + 1, dx * dy * charge);
  }
}

__global__ void batch_symmetry_kernel(double* rho1, double* rho2,
                                      int nx, int ny, std::size_t cells,
                                      std::size_t active_slots) {
  const std::size_t representatives =
      static_cast<std::size_t>(nx / 2) * (ny / 2);
  const std::size_t total = representatives * active_slots;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const std::size_t slot = work / representatives;
    const std::size_t representative = work - slot * representatives;
    const int half_y = ny / 2;
    const int ix = static_cast<int>(representative /
                                    static_cast<std::size_t>(half_y));
    const int iy = static_cast<int>(representative -
                                    static_cast<std::size_t>(ix) * half_y);
    const std::size_t base = slot * cells;
    const std::size_t j1 = base + static_cast<std::size_t>(ix) * ny + iy;
    const std::size_t j2 = base +
        static_cast<std::size_t>(nx - 1 - ix) * ny + iy;
    const std::size_t j3 = base +
        static_cast<std::size_t>(ix) * ny + (ny - 1 - iy);
    const std::size_t j4 = base +
        static_cast<std::size_t>(nx - 1 - ix) * ny + (ny - 1 - iy);
    const double mean1 = 0.25 *
        (rho1[j1] + rho1[j2] + rho1[j3] + rho1[j4]);
    const double mean2 = 0.25 *
        (rho2[j1] + rho2[j2] + rho2[j3] + rho2[j4]);
    rho1[j1] = mean1; rho1[j2] = mean1; rho1[j3] = mean1; rho1[j4] = mean1;
    rho2[j1] = mean2; rho2[j2] = mean2; rho2[j3] = mean2; rho2[j4] = mean2;
  }
}

__global__ void batch_fine_dot_kernel(const double* rho1,
                                      const double* rho2,
                                      std::size_t cells,
                                      DeviceEncounterMetrics* metrics) {
  const std::size_t slot = blockIdx.x;
  const std::size_t base = slot * cells;
  // Mode 4 is explicitly qualified under reduction_mode=fast.  This density
  // overlap contributes only the aggregate fine-luminosity diagnostic; it is
  // not consumed by the particle mover, RNG, beamstrahlung, or event-sampling
  // decisions.  Use the same FP64 tree as ResidentFieldSolver rather than
  // serializing O(nx*ny) dependent products in every encounter.
  __shared__ double partial[kThreads];
  double sum = 0.0;
  for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
    sum += rho1[base + cell] * rho2[base + cell];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) metrics[slot].fine_lumi_dot = partial[0];
}

__global__ void batch_pack_kernel(const double* rho1, const double* rho2,
                                  int nx, int ny, int padded_ny,
                                  std::size_t cells,
                                  std::size_t padded_cells,
                                  std::size_t active_slots,
                                  cufftDoubleComplex* spectrum) {
  const std::size_t total = active_slots * cells;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const std::size_t slot = work / cells;
    const std::size_t cell = work - slot * cells;
    const int ix = static_cast<int>(cell / static_cast<std::size_t>(ny));
    const int iy = static_cast<int>(cell - static_cast<std::size_t>(ix) * ny);
    cufftDoubleComplex value;
    value.x = rho1[work];
    value.y = rho2[work];
    spectrum[slot * padded_cells +
             static_cast<std::size_t>(ix) * padded_ny + iy] = value;
  }
}

__global__ void batch_multiply_kernel(cufftDoubleComplex* spectrum,
                                      const cufftDoubleComplex* dist,
                                      std::size_t padded_cells,
                                      std::size_t batch_width) {
  const std::size_t total = padded_cells * batch_width;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const cufftDoubleComplex a = spectrum[work];
    const cufftDoubleComplex b = dist[work % padded_cells];
    cufftDoubleComplex product;
    product.x = a.x * b.x - a.y * b.y;
    product.y = a.y * b.x + a.x * b.y;
    spectrum[work] = product;
  }
}

__global__ void batch_unpack_kernel(const cufftDoubleComplex* spectrum,
                                    int nx, int ny, int padded_ny,
                                    std::size_t cells,
                                    std::size_t padded_cells,
                                    std::size_t active_slots,
                                    double* phi1, double* phi2) {
  const std::size_t total = active_slots * cells;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const std::size_t slot = work / cells;
    const std::size_t cell = work - slot * cells;
    const int ix = static_cast<int>(cell / static_cast<std::size_t>(ny));
    const int iy = static_cast<int>(cell - static_cast<std::size_t>(ix) * ny);
    const cufftDoubleComplex value =
        spectrum[slot * padded_cells +
                 static_cast<std::size_t>(ix) * padded_ny + iy];
    phi1[work] = value.x;
    phi2[work] = value.y;
  }
}

__device__ void batch_electric_field(
    float x, float y, const double* phi, int ny,
    float inv_x, float inv_y, float offset_x, float offset_y,
    float& field_x, float& field_y) {
  const float absolute_x = x * inv_x + offset_x;
  const float absolute_y = y * inv_y + offset_y;
  const int y_i1 = static_cast<int>(floorf(absolute_x));
  float y_weight = absolute_y - 0.5f;
  const int y_i2 = static_cast<int>(floorf(y_weight));
  y_weight -= static_cast<float>(y_i2);
  const float ywc = 1.0f - y_weight;
  const std::size_t yc = static_cast<std::size_t>(y_i1) * ny + y_i2;
  const double p1y = static_cast<double>(y_weight) * phi[yc + ny + 1] +
                     static_cast<double>(ywc) * phi[yc + ny];
  const double p2y = static_cast<double>(y_weight) * phi[yc + 1] +
                     static_cast<double>(ywc) * phi[yc];
  const double p3y = static_cast<double>(y_weight) * phi[yc - ny + 1] +
                     static_cast<double>(ywc) * phi[yc - ny];
  const int x_i2 = static_cast<int>(floorf(absolute_y));
  float x_weight = absolute_x - 0.5f;
  const int x_i1 = static_cast<int>(floorf(x_weight));
  x_weight -= static_cast<float>(x_i1);
  const float xwc = 1.0f - x_weight;
  const std::size_t xc = static_cast<std::size_t>(x_i1) * ny + x_i2;
  const double p1x = static_cast<double>(x_weight) * phi[xc + ny + 1] +
                     static_cast<double>(xwc) * phi[xc + 1];
  const double p2x = static_cast<double>(x_weight) * phi[xc + ny] +
                     static_cast<double>(xwc) * phi[xc];
  const double p3x = static_cast<double>(x_weight) * phi[xc + ny - 1] +
                     static_cast<double>(xwc) * phi[xc - 1];
  double fx = static_cast<double>(absolute_x);
  double fy = static_cast<double>(absolute_y);
  fx -= floor(fx);
  fy -= floor(fy);
  const double raw_x =
      (fx * (p1y - p2y) + (1.0 - fx) * (p2y - p3y)) *
      static_cast<double>(inv_x);
  const double raw_y =
      (fy * (p1x - p2x) + (1.0 - fy) * (p2x - p3x)) *
      static_cast<double>(inv_y);
  field_x = static_cast<float>(raw_x) * 0.5f;
  field_y = static_cast<float>(raw_y) * 0.5f;
}

__global__ void batch_gather_kernel(
    const float* xy, const unsigned long long* counts,
    std::size_t maximum_particles, const double* opposing_phi,
    int beam, int nx, int ny, float inv_x, float inv_y,
    float offset_x, float offset_y, float min_x, float max_x,
    float min_y, float max_y, bool zero_field, std::size_t cells,
    float* fields, DeviceEncounterMetrics* metrics) {
  const std::size_t slot = blockIdx.y;
  const std::size_t count = static_cast<std::size_t>(counts[slot]);
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const float* slot_xy = xy + 2 * slot * maximum_particles;
  float* slot_fields = fields + 2 * slot * maximum_particles;
  const double* slot_phi = opposing_phi + slot * cells;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < count; particle += stride) {
    const float x = slot_xy[2 * particle];
    const float y = slot_xy[2 * particle + 1];
    const float absolute_x = x * inv_x + offset_x;
    const float absolute_y = y * inv_y + offset_y;
    const bool in_grid =
        x > min_x && x < max_x && y > min_y && y < max_y &&
        absolute_x >= 1.0f && absolute_x < static_cast<float>(nx - 1) &&
        absolute_y >= 1.0f && absolute_y < static_cast<float>(ny - 1);
    if (!in_grid) {
      slot_fields[2 * particle] = 0.0f;
      slot_fields[2 * particle + 1] = 0.0f;
      if (beam == 0) atomicAdd(&metrics[slot].gather_beam1_out, 1ULL);
      else atomicAdd(&metrics[slot].gather_beam2_out, 1ULL);
      continue;
    }
    if (beam == 0) atomicAdd(&metrics[slot].gather_beam1_in, 1ULL);
    else atomicAdd(&metrics[slot].gather_beam2_in, 1ULL);
    if (zero_field) {
      slot_fields[2 * particle] = 0.0f;
      slot_fields[2 * particle + 1] = 0.0f;
    } else {
      batch_electric_field(x, y, slot_phi, ny, inv_x, inv_y,
                           offset_x, offset_y,
                           slot_fields[2 * particle],
                           slot_fields[2 * particle + 1]);
    }
  }
}

}  // namespace

WaveBatchEncounterOutput::WaveBatchEncounterOutput()
    : fields1(0), capacity1(0), fields2(0), capacity2(0),
      beam1_in(0), beam1_out(0), beam2_in(0), beam2_out(0),
      gather_beam1_in(0), gather_beam1_out(0),
      gather_beam2_in(0), gather_beam2_out(0), fine_lumi_dot(0.0) {}

WaveBatchOptions::WaveBatchOptions()
    : device(0), batch_width(0), maximum_particles1(0),
      maximum_particles2(0), memory_fraction(0.80), pinned_io(true) {}

WaveBatchStats::WaveBatchStats()
    : batch_calls(0), encounters(0), padded_fft_slots(0),
      h2d_operations(0), d2h_operations(0), h2d_bytes(0), d2h_bytes(0),
      synchronizations(0), kernel_launches(0), cufft_executes(0),
      setup_h2d_operations(0), setup_h2d_bytes(0),
      setup_synchronizations(0), plan_creations(0), device_allocations(0),
      pinned_host_allocations(0), deposit_beam1_in(0),
      deposit_beam1_out(0), deposit_beam2_in(0), deposit_beam2_out(0),
      gather_beam1_in(0), gather_beam1_out(0), gather_beam2_in(0),
      gather_beam2_out(0), setup_ms(0.0), pack_ms(0.0), h2d_ms(0.0),
      kernel_ms(0.0), d2h_ms(0.0), unpack_ms(0.0), boundary_ms(0.0),
      resident_bytes(0), cufft_workspace_bytes(0),
      device_free_bytes_at_init(0), device_total_bytes(0),
      memory_budget_bytes(0), batch_width(0), maximum_particles1(0),
      maximum_particles2(0), nx(0), ny(0), pinned_io(false),
      context_poisoned(false) {}

struct WaveBatchedFieldSolver::Impl {
  WaveBatchStats stats;
  bool initialized;
  bool poisoned;
  int device;
  int nx;
  int ny;
  int padded_nx;
  int padded_ny;
  std::size_t cells;
  std::size_t padded_cells;
  std::size_t batch_width;
  std::size_t maximum_particles1;
  std::size_t maximum_particles2;
  cudaStream_t stream;
  cudaEvent_t events[4];
  cufftHandle plan;
  void* workspace;
  cufftDoubleComplex* spectrum;
  cufftDoubleComplex* dist;
  double* rho1;
  double* rho2;
  double* phi1;
  double* phi2;
  float* xy1;
  float* xy2;
  float* fields1;
  float* fields2;
  unsigned long long* counts;
  DeviceEncounterMetrics* metrics;
  float* host_xy1;
  float* host_xy2;
  float* host_fields1;
  float* host_fields2;
  unsigned long long* host_counts;
  DeviceEncounterMetrics* host_metrics;
  std::string device_text;
  std::string runtime_text;

  Impl()
      : initialized(false), poisoned(false), device(0), nx(0), ny(0),
        padded_nx(0), padded_ny(0), cells(0), padded_cells(0), batch_width(0),
        maximum_particles1(0), maximum_particles2(0), stream(0), plan(0),
        workspace(0), spectrum(0), dist(0), rho1(0), rho2(0), phi1(0),
        phi2(0), xy1(0), xy2(0), fields1(0), fields2(0), counts(0),
        metrics(0), host_xy1(0), host_xy2(0), host_fields1(0),
        host_fields2(0), host_counts(0), host_metrics(0) {
    for (int i = 0; i < 4; ++i) events[i] = 0;
  }

  ~Impl() {
    if (device >= 0) cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (plan) cufftDestroy(plan);
    if (events[3]) cudaEventDestroy(events[3]);
    if (events[2]) cudaEventDestroy(events[2]);
    if (events[1]) cudaEventDestroy(events[1]);
    if (events[0]) cudaEventDestroy(events[0]);
    if (workspace) cudaFree(workspace);
    if (metrics) cudaFree(metrics);
    if (counts) cudaFree(counts);
    if (fields2) cudaFree(fields2);
    if (fields1) cudaFree(fields1);
    if (xy2) cudaFree(xy2);
    if (xy1) cudaFree(xy1);
    if (phi2) cudaFree(phi2);
    if (phi1) cudaFree(phi1);
    if (rho2) cudaFree(rho2);
    if (rho1) cudaFree(rho1);
    if (dist) cudaFree(dist);
    if (spectrum) cudaFree(spectrum);
    if (host_metrics) cudaFreeHost(host_metrics);
    if (host_counts) cudaFreeHost(host_counts);
    if (host_fields2) cudaFreeHost(host_fields2);
    if (host_fields1) cudaFreeHost(host_fields1);
    if (host_xy2) cudaFreeHost(host_xy2);
    if (host_xy1) cudaFreeHost(host_xy1);
    if (stream) cudaStreamDestroy(stream);
  }
};

WaveBatchedFieldSolver::WaveBatchedFieldSolver() : impl_(new Impl) {}
WaveBatchedFieldSolver::~WaveBatchedFieldSolver() { delete impl_; }

bool WaveBatchedFieldSolver::initialize(const WaveBatchOptions& options,
                                        int nx, int ny,
                                        const double* dist_interleaved,
                                        std::string& error) {
  error.clear();
  if (impl_->initialized) {
    error = "gpu_mode=4 wave-batched solver initialized twice";
    return false;
  }
  if (options.device < 0 || options.batch_width == 0 ||
      options.batch_width > 65535 || nx < 4 || ny < 4 ||
      !dist_interleaved || !(options.memory_fraction > 0.0) ||
      options.memory_fraction > 1.0 || !options.pinned_io) {
    error = "gpu_mode=4 requires a valid device, 1..65535 batch width, "
            "grid >=4x4, Green spectrum, 0<memory_fraction<=1, and pinned I/O";
    return false;
  }
  if (nx > std::numeric_limits<int>::max() / 2 ||
      ny > std::numeric_limits<int>::max() / 2) {
    error = "gpu_mode=4 doubled FFT dimensions overflow int";
    return false;
  }
  impl_->device = options.device;
  impl_->nx = nx;
  impl_->ny = ny;
  impl_->padded_nx = 2 * nx;
  impl_->padded_ny = 2 * ny;
  impl_->batch_width = options.batch_width;
  impl_->maximum_particles1 = options.maximum_particles1;
  impl_->maximum_particles2 = options.maximum_particles2;
  if (!checked_product(static_cast<std::size_t>(nx),
                       static_cast<std::size_t>(ny), impl_->cells) ||
      !checked_product(static_cast<std::size_t>(impl_->padded_nx),
                       static_cast<std::size_t>(impl_->padded_ny),
                       impl_->padded_cells)) {
    error = "gpu_mode=4 grid dimensions overflow size_t";
    return false;
  }
  std::size_t green_values = 0;
  if (!checked_product(2, impl_->padded_cells, green_values) ||
      impl_->padded_cells > static_cast<std::size_t>(
                                  std::numeric_limits<int>::max())) {
    error = "gpu_mode=4 Green spectrum/PlanMany stride exceeds its ABI";
    return false;
  }
  for (std::size_t value = 0; value < green_values; ++value) {
    if (!std::isfinite(dist_interleaved[value])) {
      std::ostringstream message;
      message << "gpu_mode=4 Green spectrum contains a nonfinite value at "
              << value;
      error = message.str();
      return false;
    }
  }

  const std::chrono::steady_clock::time_point setup_start =
      std::chrono::steady_clock::now();
  if (!cuda_ok(cudaSetDevice(options.device),
               "cudaSetDevice(gpu_mode=4)", error)) return false;
  cudaDeviceProp properties;
  if (!cuda_ok(cudaGetDeviceProperties(&properties, options.device),
               "cudaGetDeviceProperties(gpu_mode=4)", error)) return false;
  if (properties.major < 6) {
    error = "gpu_mode=4 FP64 atomic CIC requires compute capability 6.0+";
    return false;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(gpu_mode=4)", error)) return false;
  impl_->stats.device_free_bytes_at_init = free_bytes;
  impl_->stats.device_total_bytes = total_bytes;
  impl_->stats.memory_budget_bytes = static_cast<std::size_t>(
      options.memory_fraction * static_cast<long double>(total_bytes));

  std::size_t batch_cells = 0;
  std::size_t batch_padded = 0;
  std::size_t xy_elements1 = 0;
  std::size_t xy_elements2 = 0;
  if (!checked_product(options.batch_width, impl_->cells, batch_cells) ||
      !checked_product(options.batch_width, impl_->padded_cells,
                       batch_padded) ||
      !checked_product(2 * options.batch_width,
                       options.maximum_particles1, xy_elements1) ||
      !checked_product(2 * options.batch_width,
                       options.maximum_particles2, xy_elements2)) {
    error = "gpu_mode=4 batch dimensions overflow size_t";
    return false;
  }
  const std::size_t spectrum_bytes = batch_padded * sizeof(cufftDoubleComplex);
  const std::size_t dist_bytes = impl_->padded_cells * sizeof(cufftDoubleComplex);
  const std::size_t one_grid_bytes = batch_cells * sizeof(double);
  const std::size_t xy_bytes1 = xy_elements1 * sizeof(float);
  const std::size_t xy_bytes2 = xy_elements2 * sizeof(float);
  const std::size_t counts_bytes = 2 * options.batch_width *
                                   sizeof(unsigned long long);
  const std::size_t metrics_bytes = options.batch_width *
                                    sizeof(DeviceEncounterMetrics);

  if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(gpu_mode=4)", error)) return false;
  for (int i = 0; i < 4; ++i)
    if (!cuda_ok(cudaEventCreate(&impl_->events[i]),
                 "cudaEventCreate(gpu_mode=4)", error)) return false;
  if (!cufft_ok(cufftCreate(&impl_->plan), "cufftCreate(gpu_mode=4)", error) ||
      !cufft_ok(cufftSetAutoAllocation(impl_->plan, 0),
                "cufftSetAutoAllocation(gpu_mode=4)", error)) return false;
  int dimensions[2] = {impl_->padded_nx, impl_->padded_ny};
  std::size_t workspace_bytes = 0;
  if (!cufft_ok(cufftMakePlanMany(
                    impl_->plan, 2, dimensions, 0, 1,
                    static_cast<int>(impl_->padded_cells), 0, 1,
                    static_cast<int>(impl_->padded_cells), CUFFT_Z2Z,
                    static_cast<int>(options.batch_width), &workspace_bytes),
                "cufftMakePlanMany(gpu_mode=4)", error)) return false;
  impl_->stats.plan_creations = 1;
  impl_->stats.cufft_workspace_bytes = workspace_bytes;

  std::size_t required = spectrum_bytes + dist_bytes + 4 * one_grid_bytes +
                         2 * xy_bytes1 + 2 * xy_bytes2 + counts_bytes +
                         metrics_bytes + workspace_bytes;
  if (required > impl_->stats.memory_budget_bytes || required > free_bytes) {
    std::ostringstream message;
    message << "gpu_mode=4 requires " << required
            << " resident bytes, exceeding "
            << (required > impl_->stats.memory_budget_bytes
                    ? "gpu2_memory_fraction budget " : "free device memory ")
            << (required > impl_->stats.memory_budget_bytes
                    ? impl_->stats.memory_budget_bytes : free_bytes);
    error = message.str();
    return false;
  }
  impl_->stats.resident_bytes = required;

#define FEEBL_MODE4_MALLOC(pointer, bytes, label)                           \
  do {                                                                      \
    if ((bytes) && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&(pointer)), \
                                       (bytes)), label, error)) return false; \
    if (bytes) ++impl_->stats.device_allocations;                           \
  } while (0)
  FEEBL_MODE4_MALLOC(impl_->spectrum, spectrum_bytes,
                     "cudaMalloc(gpu_mode=4 spectrum)");
  FEEBL_MODE4_MALLOC(impl_->dist, dist_bytes,
                     "cudaMalloc(gpu_mode=4 Green spectrum)");
  FEEBL_MODE4_MALLOC(impl_->rho1, one_grid_bytes,
                     "cudaMalloc(gpu_mode=4 rho1)");
  FEEBL_MODE4_MALLOC(impl_->rho2, one_grid_bytes,
                     "cudaMalloc(gpu_mode=4 rho2)");
  FEEBL_MODE4_MALLOC(impl_->phi1, one_grid_bytes,
                     "cudaMalloc(gpu_mode=4 phi1)");
  FEEBL_MODE4_MALLOC(impl_->phi2, one_grid_bytes,
                     "cudaMalloc(gpu_mode=4 phi2)");
  FEEBL_MODE4_MALLOC(impl_->xy1, xy_bytes1,
                     "cudaMalloc(gpu_mode=4 xy1)");
  FEEBL_MODE4_MALLOC(impl_->xy2, xy_bytes2,
                     "cudaMalloc(gpu_mode=4 xy2)");
  FEEBL_MODE4_MALLOC(impl_->fields1, xy_bytes1,
                     "cudaMalloc(gpu_mode=4 fields1)");
  FEEBL_MODE4_MALLOC(impl_->fields2, xy_bytes2,
                     "cudaMalloc(gpu_mode=4 fields2)");
  FEEBL_MODE4_MALLOC(impl_->counts, counts_bytes,
                     "cudaMalloc(gpu_mode=4 counts)");
  FEEBL_MODE4_MALLOC(impl_->metrics, metrics_bytes,
                     "cudaMalloc(gpu_mode=4 metrics)");
  FEEBL_MODE4_MALLOC(impl_->workspace, workspace_bytes,
                     "cudaMalloc(gpu_mode=4 cuFFT workspace)");
#undef FEEBL_MODE4_MALLOC
  if (!cufft_ok(cufftSetWorkArea(impl_->plan, impl_->workspace),
                "cufftSetWorkArea(gpu_mode=4)", error) ||
      !cufft_ok(cufftSetStream(impl_->plan, impl_->stream),
                "cufftSetStream(gpu_mode=4)", error)) return false;

#define FEEBL_MODE4_PINNED(pointer, bytes, label)                         \
  do {                                                                    \
    if ((bytes) && !cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&(pointer)), \
                                          (bytes), cudaHostAllocDefault),   \
                             label, error)) return false;                  \
    if (bytes) ++impl_->stats.pinned_host_allocations;                    \
  } while (0)
  FEEBL_MODE4_PINNED(impl_->host_xy1, xy_bytes1,
                     "cudaHostAlloc(gpu_mode=4 xy1)");
  FEEBL_MODE4_PINNED(impl_->host_xy2, xy_bytes2,
                     "cudaHostAlloc(gpu_mode=4 xy2)");
  FEEBL_MODE4_PINNED(impl_->host_fields1, xy_bytes1,
                     "cudaHostAlloc(gpu_mode=4 fields1)");
  FEEBL_MODE4_PINNED(impl_->host_fields2, xy_bytes2,
                     "cudaHostAlloc(gpu_mode=4 fields2)");
  FEEBL_MODE4_PINNED(impl_->host_counts, counts_bytes,
                     "cudaHostAlloc(gpu_mode=4 counts)");
  FEEBL_MODE4_PINNED(impl_->host_metrics, metrics_bytes,
                     "cudaHostAlloc(gpu_mode=4 metrics)");
#undef FEEBL_MODE4_PINNED

  if (!cuda_ok(cudaMemcpyAsync(impl_->dist, dist_interleaved, dist_bytes,
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(gpu_mode=4 Green H2D)", error) ||
      !cuda_ok(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize(gpu_mode=4 setup)", error)) return false;
  impl_->stats.setup_h2d_operations = 1;
  impl_->stats.setup_h2d_bytes = dist_bytes;
  impl_->stats.setup_synchronizations = 1;
  impl_->stats.batch_width = options.batch_width;
  impl_->stats.maximum_particles1 = options.maximum_particles1;
  impl_->stats.maximum_particles2 = options.maximum_particles2;
  impl_->stats.nx = nx;
  impl_->stats.ny = ny;
  impl_->stats.pinned_io = true;
  impl_->stats.setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  std::ostringstream device;
  device << properties.name << " cc " << properties.major << '.'
         << properties.minor;
  impl_->device_text = device.str();
  std::ostringstream runtime;
  runtime << "gpu_mode=4 PlanMany batch=" << options.batch_width
          << " grid=" << nx << 'x' << ny
          << " padded=" << impl_->padded_nx << 'x' << impl_->padded_ny
          << " workspace=" << workspace_bytes
          << " resident=" << required
          << " pinned_io=on stream=1 sync=batch";
  impl_->runtime_text = runtime.str();
  impl_->initialized = true;
  return true;
}

bool WaveBatchedFieldSolver::execute(
    const WaveBatchEncounterInput* inputs, WaveBatchEncounterOutput* outputs,
    std::size_t encounter_count, double charge1, double charge2,
    float inv_x, float inv_y, float offset_x, float offset_y,
    float min_x, float max_x, float min_y, float max_y,
    bool four_way_symmetry, float charge_sign, std::string& error) {
  error.clear();
  if (!impl_->initialized || impl_->poisoned) {
    error = impl_->poisoned
        ? "gpu_mode=4 context is poisoned after a device failure"
        : "gpu_mode=4 context is not initialized";
    return false;
  }
  if (!inputs || !outputs || encounter_count == 0 ||
      encounter_count > impl_->batch_width) {
    error = "gpu_mode=4 execute requires 1..batch_width encounters";
    return false;
  }
  if (!std::isfinite(charge1) || !std::isfinite(charge2) ||
      !std::isfinite(inv_x) || !std::isfinite(inv_y) ||
      !std::isfinite(offset_x) || !std::isfinite(offset_y) ||
      !std::isfinite(min_x) || !std::isfinite(max_x) ||
      !std::isfinite(min_y) || !std::isfinite(max_y) ||
      !std::isfinite(charge_sign) ||
      inv_x == 0.0f || inv_y == 0.0f || !(max_x > min_x) ||
      !(max_y > min_y)) {
    error = "gpu_mode=4 received invalid grid/deposition parameters";
    return false;
  }
  for (std::size_t slot = 0; slot < encounter_count; ++slot) {
    if (inputs[slot].count1 > impl_->maximum_particles1 ||
        inputs[slot].count2 > impl_->maximum_particles2 ||
        (inputs[slot].count1 && !inputs[slot].xy1) ||
        (inputs[slot].count2 && !inputs[slot].xy2) ||
        outputs[slot].capacity1 < inputs[slot].count1 ||
        outputs[slot].capacity2 < inputs[slot].count2 ||
        (inputs[slot].count1 && !outputs[slot].fields1) ||
        (inputs[slot].count2 && !outputs[slot].fields2)) {
      std::ostringstream message;
      message << "gpu_mode=4 encounter " << slot
              << " violates the preflight particle/output capacity";
      error = message.str();
      return false;
    }
    for (std::size_t value = 0; value < 2 * inputs[slot].count1; ++value) {
      if (!std::isfinite(inputs[slot].xy1[value])) {
        std::ostringstream message;
        message << "gpu_mode=4 encounter " << slot
                << " beam1 xy contains a nonfinite value at " << value;
        error = message.str();
        return false;
      }
    }
    for (std::size_t value = 0; value < 2 * inputs[slot].count2; ++value) {
      if (!std::isfinite(inputs[slot].xy2[value])) {
        std::ostringstream message;
        message << "gpu_mode=4 encounter " << slot
                << " beam2 xy contains a nonfinite value at " << value;
        error = message.str();
        return false;
      }
    }
  }
  if (!cuda_ok(cudaSetDevice(impl_->device),
               "cudaSetDevice(gpu_mode=4 execute)", error)) return false;

  const std::chrono::steady_clock::time_point boundary_start =
      std::chrono::steady_clock::now();
  const std::chrono::steady_clock::time_point pack_start = boundary_start;
  for (std::size_t slot = 0; slot < encounter_count; ++slot) {
    impl_->host_counts[slot] =
        static_cast<unsigned long long>(inputs[slot].count1);
    impl_->host_counts[impl_->batch_width + slot] =
        static_cast<unsigned long long>(inputs[slot].count2);
    if (inputs[slot].count1)
      std::memcpy(impl_->host_xy1 + 2 * slot * impl_->maximum_particles1,
                  inputs[slot].xy1, 2 * inputs[slot].count1 * sizeof(float));
    if (inputs[slot].count2)
      std::memcpy(impl_->host_xy2 + 2 * slot * impl_->maximum_particles2,
                  inputs[slot].xy2, 2 * inputs[slot].count2 * sizeof(float));
  }
  const double pack_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - pack_start).count();

  const std::size_t active_xy_bytes1 =
      2 * encounter_count * impl_->maximum_particles1 * sizeof(float);
  const std::size_t active_xy_bytes2 =
      2 * encounter_count * impl_->maximum_particles2 * sizeof(float);
  const std::size_t counts_bytes = 2 * impl_->batch_width *
                                   sizeof(unsigned long long);
  const std::size_t metrics_bytes = encounter_count *
                                    sizeof(DeviceEncounterMetrics);
  const std::size_t active_grid_bytes =
      encounter_count * impl_->cells * sizeof(double);
  const std::size_t all_spectrum_bytes =
      impl_->batch_width * impl_->padded_cells * sizeof(cufftDoubleComplex);
  bool work_enqueued = false;
  const auto fail = [&](const char* operation, cudaError_t status) {
    if (status == cudaSuccess) return false;
    cuda_ok(status, operation, error);
    if (work_enqueued) {
      cudaStreamSynchronize(impl_->stream);
      impl_->poisoned = true;
      impl_->stats.context_poisoned = true;
    }
    return true;
  };

  if (fail("cudaEventRecord(gpu_mode=4 start)",
           cudaEventRecord(impl_->events[0], impl_->stream))) return false;
  work_enqueued = true;
  if (fail("cudaMemcpyAsync(gpu_mode=4 xy1 H2D)",
           cudaMemcpyAsync(impl_->xy1, impl_->host_xy1, active_xy_bytes1,
                           cudaMemcpyHostToDevice, impl_->stream)) ||
      fail("cudaMemcpyAsync(gpu_mode=4 xy2 H2D)",
           cudaMemcpyAsync(impl_->xy2, impl_->host_xy2, active_xy_bytes2,
                           cudaMemcpyHostToDevice, impl_->stream)) ||
      fail("cudaMemcpyAsync(gpu_mode=4 counts H2D)",
           cudaMemcpyAsync(impl_->counts, impl_->host_counts, counts_bytes,
                           cudaMemcpyHostToDevice, impl_->stream)) ||
      fail("cudaEventRecord(gpu_mode=4 after H2D)",
           cudaEventRecord(impl_->events[1], impl_->stream)) ||
      fail("cudaMemsetAsync(gpu_mode=4 rho1)",
           cudaMemsetAsync(impl_->rho1, 0, active_grid_bytes, impl_->stream)) ||
      fail("cudaMemsetAsync(gpu_mode=4 rho2)",
           cudaMemsetAsync(impl_->rho2, 0, active_grid_bytes, impl_->stream)) ||
      fail("cudaMemsetAsync(gpu_mode=4 metrics)",
           cudaMemsetAsync(impl_->metrics, 0, metrics_bytes, impl_->stream)))
    return false;

  const dim3 cic_grid1(blocks_for(impl_->maximum_particles1),
                       static_cast<unsigned int>(encounter_count), 1);
  const dim3 cic_grid2(blocks_for(impl_->maximum_particles2),
                       static_cast<unsigned int>(encounter_count), 1);
  batch_cic_kernel<<<cic_grid1, kThreads, 0, impl_->stream>>>(
      impl_->xy1, impl_->counts, impl_->maximum_particles1, charge1, 0,
      impl_->nx, impl_->ny, inv_x, inv_y, offset_x, offset_y,
      impl_->cells, impl_->rho1, impl_->metrics);
  if (fail("batch_cic_kernel beam1 launch", cudaGetLastError())) return false;
  batch_cic_kernel<<<cic_grid2, kThreads, 0, impl_->stream>>>(
      impl_->xy2, impl_->counts + impl_->batch_width,
      impl_->maximum_particles2, charge2, 1, impl_->nx, impl_->ny,
      inv_x, inv_y, offset_x, offset_y, impl_->cells,
      impl_->rho2, impl_->metrics);
  if (fail("batch_cic_kernel beam2 launch", cudaGetLastError())) return false;
  std::uint64_t launches = 2;
  if (four_way_symmetry) {
    batch_symmetry_kernel<<<blocks_for(
        encounter_count * static_cast<std::size_t>(impl_->nx / 2) *
        static_cast<std::size_t>(impl_->ny / 2)), kThreads, 0,
        impl_->stream>>>(impl_->rho1, impl_->rho2, impl_->nx, impl_->ny,
                         impl_->cells, encounter_count);
    if (fail("batch_symmetry_kernel launch", cudaGetLastError())) return false;
    ++launches;
  }
  batch_fine_dot_kernel<<<static_cast<unsigned int>(encounter_count),
                          kThreads, 0, impl_->stream>>>(
      impl_->rho1, impl_->rho2, impl_->cells, impl_->metrics);
  if (fail("batch_fine_dot_kernel launch", cudaGetLastError())) return false;
  ++launches;

  const bool zero_field = std::fabs(charge_sign) < 1.0e-5f;
  std::uint64_t fft_executes = 0;
  if (!zero_field) {
    if (fail("cudaMemsetAsync(gpu_mode=4 spectrum)",
             cudaMemsetAsync(impl_->spectrum, 0, all_spectrum_bytes,
                             impl_->stream))) return false;
    batch_pack_kernel<<<blocks_for(encounter_count * impl_->cells),
                        kThreads, 0, impl_->stream>>>(
        impl_->rho1, impl_->rho2, impl_->nx, impl_->ny, impl_->padded_ny,
        impl_->cells, impl_->padded_cells, encounter_count, impl_->spectrum);
    if (fail("batch_pack_kernel launch", cudaGetLastError())) return false;
    ++launches;
    cufftResult fft_status = cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                          impl_->spectrum, CUFFT_FORWARD);
    if (fft_status != CUFFT_SUCCESS) {
      cufft_ok(fft_status, "cufftExecZ2Z(gpu_mode=4 forward)", error);
      cudaStreamSynchronize(impl_->stream);
      impl_->poisoned = true;
      impl_->stats.context_poisoned = true;
      return false;
    }
    ++fft_executes;
    batch_multiply_kernel<<<blocks_for(
        impl_->batch_width * impl_->padded_cells), kThreads, 0,
        impl_->stream>>>(impl_->spectrum, impl_->dist, impl_->padded_cells,
                         impl_->batch_width);
    if (fail("batch_multiply_kernel launch", cudaGetLastError())) return false;
    ++launches;
    fft_status = cufftExecZ2Z(impl_->plan, impl_->spectrum,
                              impl_->spectrum, CUFFT_INVERSE);
    if (fft_status != CUFFT_SUCCESS) {
      cufft_ok(fft_status, "cufftExecZ2Z(gpu_mode=4 inverse)", error);
      cudaStreamSynchronize(impl_->stream);
      impl_->poisoned = true;
      impl_->stats.context_poisoned = true;
      return false;
    }
    ++fft_executes;
    batch_unpack_kernel<<<blocks_for(encounter_count * impl_->cells),
                          kThreads, 0, impl_->stream>>>(
        impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
        impl_->cells, impl_->padded_cells, encounter_count,
        impl_->phi1, impl_->phi2);
    if (fail("batch_unpack_kernel launch", cudaGetLastError())) return false;
    ++launches;
  }

  batch_gather_kernel<<<cic_grid1, kThreads, 0, impl_->stream>>>(
      impl_->xy1, impl_->counts, impl_->maximum_particles1, impl_->phi2, 0,
      impl_->nx, impl_->ny, inv_x, inv_y, offset_x, offset_y,
      min_x, max_x, min_y, max_y, zero_field, impl_->cells,
      impl_->fields1, impl_->metrics);
  if (fail("batch_gather_kernel beam1 launch", cudaGetLastError())) return false;
  batch_gather_kernel<<<cic_grid2, kThreads, 0, impl_->stream>>>(
      impl_->xy2, impl_->counts + impl_->batch_width,
      impl_->maximum_particles2, impl_->phi1, 1, impl_->nx, impl_->ny,
      inv_x, inv_y, offset_x, offset_y, min_x, max_x, min_y, max_y,
      zero_field, impl_->cells, impl_->fields2, impl_->metrics);
  if (fail("batch_gather_kernel beam2 launch", cudaGetLastError())) return false;
  launches += 2;

  if (fail("cudaEventRecord(gpu_mode=4 after kernels)",
           cudaEventRecord(impl_->events[2], impl_->stream)) ||
      fail("cudaMemcpyAsync(gpu_mode=4 fields1 D2H)",
           cudaMemcpyAsync(impl_->host_fields1, impl_->fields1,
                           active_xy_bytes1, cudaMemcpyDeviceToHost,
                           impl_->stream)) ||
      fail("cudaMemcpyAsync(gpu_mode=4 fields2 D2H)",
           cudaMemcpyAsync(impl_->host_fields2, impl_->fields2,
                           active_xy_bytes2, cudaMemcpyDeviceToHost,
                           impl_->stream)) ||
      fail("cudaMemcpyAsync(gpu_mode=4 metrics D2H)",
           cudaMemcpyAsync(impl_->host_metrics, impl_->metrics, metrics_bytes,
                           cudaMemcpyDeviceToHost, impl_->stream)) ||
      fail("cudaEventRecord(gpu_mode=4 after D2H)",
           cudaEventRecord(impl_->events[3], impl_->stream)) ||
      fail("cudaEventSynchronize(gpu_mode=4 batch boundary)",
           cudaEventSynchronize(impl_->events[3]))) return false;
  work_enqueued = false;

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&h2d_ms, impl_->events[0],
                                    impl_->events[1]),
               "cudaEventElapsedTime(gpu_mode=4 H2D)", error) ||
      !cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                    impl_->events[2]),
               "cudaEventElapsedTime(gpu_mode=4 kernels)", error) ||
      !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                    impl_->events[3]),
               "cudaEventElapsedTime(gpu_mode=4 D2H)", error)) {
    impl_->poisoned = true;
    impl_->stats.context_poisoned = true;
    return false;
  }

  // Validate the complete staging result before publishing a single caller
  // field/counter.  Device corruption or a future kernel contract regression
  // therefore cannot expose a partially updated batch.
  for (std::size_t slot = 0; slot < encounter_count; ++slot) {
    const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
    if (metric.beam1_in > inputs[slot].count1 ||
        metric.beam1_out != inputs[slot].count1 - metric.beam1_in ||
        metric.beam2_in > inputs[slot].count2 ||
        metric.beam2_out != inputs[slot].count2 - metric.beam2_in ||
        metric.gather_beam1_in > inputs[slot].count1 ||
        metric.gather_beam1_out !=
            inputs[slot].count1 - metric.gather_beam1_in ||
        metric.gather_beam2_in > inputs[slot].count2 ||
        metric.gather_beam2_out !=
            inputs[slot].count2 - metric.gather_beam2_in ||
        !std::isfinite(metric.fine_lumi_dot)) {
      std::ostringstream message;
      message << "gpu_mode=4 staged metrics failed validation at encounter "
              << slot;
      error = message.str();
      impl_->poisoned = true;
      impl_->stats.context_poisoned = true;
      return false;
    }
    const float* fields1 =
        impl_->host_fields1 + 2 * slot * impl_->maximum_particles1;
    const float* fields2 =
        impl_->host_fields2 + 2 * slot * impl_->maximum_particles2;
    for (std::size_t value = 0; value < 2 * inputs[slot].count1; ++value) {
      if (!std::isfinite(fields1[value])) {
        error = "gpu_mode=4 staged beam1 field is nonfinite";
        impl_->poisoned = true;
        impl_->stats.context_poisoned = true;
        return false;
      }
    }
    for (std::size_t value = 0; value < 2 * inputs[slot].count2; ++value) {
      if (!std::isfinite(fields2[value])) {
        error = "gpu_mode=4 staged beam2 field is nonfinite";
        impl_->poisoned = true;
        impl_->stats.context_poisoned = true;
        return false;
      }
    }
  }

  const std::chrono::steady_clock::time_point unpack_start =
      std::chrono::steady_clock::now();
  for (std::size_t slot = 0; slot < encounter_count; ++slot) {
    if (inputs[slot].count1)
      std::memcpy(outputs[slot].fields1,
                  impl_->host_fields1 + 2 * slot * impl_->maximum_particles1,
                  2 * inputs[slot].count1 * sizeof(float));
    if (inputs[slot].count2)
      std::memcpy(outputs[slot].fields2,
                  impl_->host_fields2 + 2 * slot * impl_->maximum_particles2,
                  2 * inputs[slot].count2 * sizeof(float));
    const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
    outputs[slot].beam1_in = static_cast<std::size_t>(metric.beam1_in);
    outputs[slot].beam1_out = static_cast<std::size_t>(metric.beam1_out);
    outputs[slot].beam2_in = static_cast<std::size_t>(metric.beam2_in);
    outputs[slot].beam2_out = static_cast<std::size_t>(metric.beam2_out);
    outputs[slot].gather_beam1_in =
        static_cast<std::size_t>(metric.gather_beam1_in);
    outputs[slot].gather_beam1_out =
        static_cast<std::size_t>(metric.gather_beam1_out);
    outputs[slot].gather_beam2_in =
        static_cast<std::size_t>(metric.gather_beam2_in);
    outputs[slot].gather_beam2_out =
        static_cast<std::size_t>(metric.gather_beam2_out);
    outputs[slot].fine_lumi_dot = metric.fine_lumi_dot;
  }
  const double unpack_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - unpack_start).count();

  ++impl_->stats.batch_calls;
  impl_->stats.encounters += encounter_count;
  impl_->stats.padded_fft_slots += impl_->batch_width - encounter_count;
  impl_->stats.h2d_operations += 3;
  impl_->stats.d2h_operations += 3;
  impl_->stats.h2d_bytes += active_xy_bytes1 + active_xy_bytes2 + counts_bytes;
  impl_->stats.d2h_bytes += active_xy_bytes1 + active_xy_bytes2 + metrics_bytes;
  ++impl_->stats.synchronizations;
  impl_->stats.kernel_launches += launches;
  impl_->stats.cufft_executes += fft_executes;
  impl_->stats.pack_ms += pack_ms;
  impl_->stats.h2d_ms += h2d_ms;
  impl_->stats.kernel_ms += kernel_ms;
  impl_->stats.d2h_ms += d2h_ms;
  impl_->stats.unpack_ms += unpack_ms;
  for (std::size_t slot = 0; slot < encounter_count; ++slot) {
    const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
    impl_->stats.deposit_beam1_in += metric.beam1_in;
    impl_->stats.deposit_beam1_out += metric.beam1_out;
    impl_->stats.deposit_beam2_in += metric.beam2_in;
    impl_->stats.deposit_beam2_out += metric.beam2_out;
    impl_->stats.gather_beam1_in += metric.gather_beam1_in;
    impl_->stats.gather_beam1_out += metric.gather_beam1_out;
    impl_->stats.gather_beam2_in += metric.gather_beam2_in;
    impl_->stats.gather_beam2_out += metric.gather_beam2_out;
  }
  impl_->stats.boundary_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - boundary_start).count();
  return true;
}

bool WaveBatchedFieldSolver::is_available() const {
  return impl_->initialized && !impl_->poisoned;
}

std::string WaveBatchedFieldSolver::device_description() const {
  if (!impl_->initialized) return "CUDA not initialized";
  return impl_->poisoned ? impl_->device_text + " [POISONED]"
                         : impl_->device_text;
}

std::string WaveBatchedFieldSolver::runtime_description() const {
  if (!impl_->initialized) return "CUDA/cuFFT not initialized";
  return impl_->poisoned ? impl_->runtime_text + ", context=POISONED"
                         : impl_->runtime_text;
}

WaveBatchStats WaveBatchedFieldSolver::stats() const { return impl_->stats; }

}  // namespace gpu
}  // namespace feebl
