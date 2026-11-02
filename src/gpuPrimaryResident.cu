#include "gpuPrimaryResident.h"

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
  if (right > std::numeric_limits<std::size_t>::max() - left) return false;
  result = left + right;
  return true;
}

bool checked_u64_sum(std::uint64_t left,
                     std::uint64_t right,
                     std::uint64_t& result)
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return false;
  result = left + right;
  return true;
}

bool size_to_u64(std::size_t value, std::uint64_t& result)
{
  if (value > static_cast<std::size_t>(
                  std::numeric_limits<std::uint64_t>::max()))
    return false;
  result = static_cast<std::uint64_t>(value);
  return true;
}

bool can_add_u64(std::uint64_t base, std::uint64_t increment)
{
  std::uint64_t unused = 0;
  return checked_u64_sum(base, increment, unused);
}

bool valid_grid(const ResidentPrimaryGrid& grid)
{
  return std::isfinite(grid.delta_x_inverse) &&
         std::isfinite(grid.delta_y_inverse) &&
         std::isfinite(grid.offset_x) && std::isfinite(grid.offset_y) &&
         std::isfinite(grid.physical_min_x) &&
         std::isfinite(grid.physical_max_x) &&
         std::isfinite(grid.physical_min_y) &&
         std::isfinite(grid.physical_max_y) &&
         grid.delta_x_inverse > 0.0f && grid.delta_y_inverse > 0.0f &&
         grid.physical_min_x < grid.physical_max_x &&
         grid.physical_min_y < grid.physical_max_y;
}

bool same_grid(const ResidentPrimaryGrid& left,
               const ResidentPrimaryGrid& right)
{
  return left.delta_x_inverse == right.delta_x_inverse &&
         left.delta_y_inverse == right.delta_y_inverse &&
         left.offset_x == right.offset_x &&
         left.offset_y == right.offset_y &&
         left.physical_min_x == right.physical_min_x &&
         left.physical_max_x == right.physical_max_x &&
         left.physical_min_y == right.physical_min_y &&
         left.physical_max_y == right.physical_max_y;
}

bool finite_particle_grid_arithmetic(const ResidentPrimaryGrid& grid,
                                     float x,
                                     float y)
{
  // Check both legacy coordinate transforms used by deposit and gather.  All
  // operands are deliberately float, matching the device kernels.
  const float deposit_x =
      x * grid.delta_x_inverse + (grid.offset_x - 0.5f);
  const float deposit_y =
      y * grid.delta_y_inverse + (grid.offset_y - 0.5f);
  const float gather_x = x * grid.delta_x_inverse + grid.offset_x;
  const float gather_y = y * grid.delta_y_inverse + grid.offset_y;
  return std::isfinite(deposit_x) && std::isfinite(deposit_y) &&
         std::isfinite(gather_x) && std::isfinite(gather_y);
}

class ScopedCudaDevice
{
 public:
  ScopedCudaDevice(int target, const char* operation, std::string& error)
      : previous_(-1), restore_(false), selected_(false)
  {
    if (!cuda_ok(cudaGetDevice(&previous_), "cudaGetDevice", error)) return;
    if (previous_ != target)
      {
        if (!cuda_ok(cudaSetDevice(target), operation, error)) return;
        restore_ = true;
      }
    selected_ = true;
  }

  ~ScopedCudaDevice()
  {
    if (restore_) (void)cudaSetDevice(previous_);
  }

  bool selected() const { return selected_; }

 private:
  int previous_;
  bool restore_;
  bool selected_;
};

bool range_fits(std::size_t offset,
                std::size_t count,
                std::size_t extent)
{
  return offset <= extent && count <= extent - offset;
}

bool index_ranges_overlap(std::size_t offset1,
                          std::size_t count1,
                          std::size_t offset2,
                          std::size_t count2)
{
  if (!count1 || !count2) return false;
  return offset1 < offset2 + count2 && offset2 < offset1 + count1;
}

bool byte_ranges_overlap(const void* first,
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

int block_count(std::size_t work)
{
  if (!work) return 1;
  const std::size_t needed = (work + kThreads - 1) / kThreads;
  return static_cast<int>(std::min<std::size_t>(needed, 65535));
}

std::string version_text(int encoded)
{
  if (encoded < 0) return "unavailable";
  std::ostringstream value;
  value << encoded / 1000 << '.' << (encoded % 1000) / 10;
  const int patch = encoded % 10;
  if (patch) value << '.' << patch;
  return value.str();
}

struct DevicePrimaryMetrics
{
  unsigned long long deposit_beam1_in;
  unsigned long long deposit_beam1_out;
  unsigned long long deposit_beam2_in;
  unsigned long long deposit_beam2_out;
  unsigned long long gather_beam1_in;
  unsigned long long gather_beam1_out;
  unsigned long long gather_beam2_in;
  unsigned long long gather_beam2_out;
  unsigned long long nonfinite_state_events;
  double fine_lumi_dot_sum;
};

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
                                    float& hy,
                                    bool& arithmetic_finite)
{
  // Match gpuCic.cu and GRID::particleInGrid: the historical arithmetic is
  // float and the half-cell shift is grouped with the offset.
  hx = x * delta_x_inverse + (offset_x - 0.5f);
  hy = y * delta_y_inverse + (offset_y - 0.5f);
  arithmetic_finite = isfinite(hx) && isfinite(hy);
  if (!arithmetic_finite) return false;
  ix = static_cast<int>(floorf(hx));
  hx -= static_cast<float>(ix);
  iy = static_cast<int>(floorf(hy));
  hy -= static_cast<float>(iy);
  return ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1;
}

__global__ void primary_cic_kernel(const float* xy,
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
                                   DevicePrimaryMetrics* metrics)
{
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      int ix = 0;
      int iy = 0;
      float hx = 0.0f;
      float hy = 0.0f;
      bool arithmetic_finite = true;
      if (!locate_cic_particle(xy[2 * particle], xy[2 * particle + 1],
                               delta_x_inverse, delta_y_inverse,
                               offset_x, offset_y, nx, ny,
                               ix, iy, hx, hy, arithmetic_finite))
        {
          if (!arithmetic_finite)
            atomicAdd(&metrics->nonfinite_state_events, 1ULL);
          if (beam == 0)
            atomicAdd(&metrics->deposit_beam1_out, 1ULL);
          else
            atomicAdd(&metrics->deposit_beam2_out, 1ULL);
          continue;
        }
      if (beam == 0)
        atomicAdd(&metrics->deposit_beam1_in, 1ULL);
      else
        atomicAdd(&metrics->deposit_beam2_in, 1ULL);

      const double dx = static_cast<double>(hx);
      const double dy = static_cast<double>(hy);
      atomicAdd(rho + static_cast<std::size_t>(ix) * ny + iy,
                (1.0 - dx) * (1.0 - dy) * charge);
      atomicAdd(rho + static_cast<std::size_t>(ix + 1) * ny + iy,
                dx * (1.0 - dy) * charge);
      atomicAdd(rho + static_cast<std::size_t>(ix) * ny + iy + 1,
                (1.0 - dx) * dy * charge);
      atomicAdd(rho + static_cast<std::size_t>(ix + 1) * ny + iy + 1,
                dx * dy * charge);
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
      const int ix = static_cast<int>(
          representative / static_cast<std::size_t>(half_y));
      const int iy = static_cast<int>(
          representative - static_cast<std::size_t>(ix) * half_y);
      const std::size_t j1 = static_cast<std::size_t>(ix) * ny + iy;
      const std::size_t j2 =
          static_cast<std::size_t>(nx - 1 - ix) * ny + iy;
      const std::size_t j3 =
          static_cast<std::size_t>(ix) * ny + (ny - 1 - iy);
      const std::size_t j4 =
          static_cast<std::size_t>(nx - 1 - ix) * ny + (ny - 1 - iy);
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
                                     DevicePrimaryMetrics* metrics)
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
  // Every encounter is ordered on one stream, so this cumulative addition is
  // deterministic for a fixed encounter order and needs no atomic operation.
  if (threadIdx.x == 0) metrics->fine_lumi_dot_sum += partial[0];
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
  // Preserve GENERAL_GRID::ElectricFieldCIC's types and narrowing points.
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

__global__ void gather_and_push_kernel(
    float* xy,
    float* velocity_xy,
    const float* energy,
    float* upsilon,
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
    float step,
    float scale,
    bool zero_field,
    DevicePrimaryMetrics* metrics)
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
      const bool arithmetic_finite =
          isfinite(x) && isfinite(y) && isfinite(absolute_x) &&
          isfinite(absolute_y) && isfinite(energy[particle]) &&
          energy[particle] > 0.0f;
      const bool in_grid =
          arithmetic_finite &&
          x > physical_min_x && x < physical_max_x &&
          y > physical_min_y && y < physical_max_y &&
          absolute_x >= 1.0f &&
          absolute_x < static_cast<float>(nx - 1) &&
          absolute_y >= 1.0f &&
          absolute_y < static_cast<float>(ny - 1);

      float field_x = 0.0f;
      float field_y = 0.0f;
      if (in_grid)
        {
          if (beam == 0)
            atomicAdd(&metrics->gather_beam1_in, 1ULL);
          else
            atomicAdd(&metrics->gather_beam2_in, 1ULL);
          if (!zero_field)
            electric_field_cic_legacy(
                x, y, opposing_phi, ny, delta_x_inverse,
                delta_y_inverse, offset_x, offset_y, field_x, field_y);
        }
      else if (beam == 0)
        atomicAdd(&metrics->gather_beam1_out, 1ULL);
      else
        atomicAdd(&metrics->gather_beam2_out, 1ULL);

      // Exact no-radiation PARTICLE::advanceDueToEBfield orbit contract:
      // advanceVelocities(-2E, step), then advancePosition(step*scale).
      const float delta_vx = step * (-2.0f * field_x) / energy[particle];
      const float delta_vy = step * (-2.0f * field_y) / energy[particle];
      const float new_vx = velocity_xy[2 * particle] + delta_vx;
      const float new_vy = velocity_xy[2 * particle + 1] + delta_vy;
      const float position_step = step * scale;
      const float new_x = x + new_vx * position_step;
      const float new_y = y + new_vy * position_step;
      // Preserve PARTICLE::advanceDueToEBfield's legacy narrowing points.
      // `1e-9`, LAMBDA_BAR and EMASS are unsuffixed double constants in the
      // legacy source, while delta-v squares, sqrt and radius are float.
      const float dz = static_cast<float>(
          static_cast<double>(step) * 1.0e-9 *
          static_cast<double>(scale));
      const float delta_v_squared = __fadd_rn(
          __fmul_rn(delta_vx, delta_vx),
          __fmul_rn(delta_vy, delta_vy));
      const float dz_on_radius = sqrtf(delta_v_squared);
      const float inverse_radius = __fdiv_rn(dz_on_radius, dz);
      const double electron_mass_gev = 0.51099906e-3;
      const double upsilon_double =
          3.86159323e-13 /
          (electron_mass_gev * electron_mass_gev) *
          static_cast<double>(energy[particle]) *
          static_cast<double>(energy[particle]) *
          static_cast<double>(inverse_radius);
      const float new_upsilon = static_cast<float>(upsilon_double);
      if (!arithmetic_finite || !isfinite(field_x) || !isfinite(field_y) ||
          !isfinite(delta_vx) || !isfinite(delta_vy) ||
          !isfinite(new_vx) || !isfinite(new_vy) ||
          !isfinite(position_step) || !isfinite(new_x) || !isfinite(new_y) ||
          !isfinite(dz) || dz <= 0.0f ||
          !isfinite(delta_v_squared) || delta_v_squared < 0.0f ||
          !isfinite(dz_on_radius) || !isfinite(inverse_radius) ||
          !isfinite(new_upsilon) || new_upsilon < 0.0f)
        {
          atomicAdd(&metrics->nonfinite_state_events, 1ULL);
          continue;
        }
      velocity_xy[2 * particle] = new_vx;
      velocity_xy[2 * particle + 1] = new_vy;
      xy[2 * particle] = new_x;
      xy[2 * particle + 1] = new_y;
      upsilon[particle] = new_upsilon;
    }
}

}  // namespace

ResidentPrimaryContract::ResidentPrimaryContract()
    : interpolation(2),
      extra_grids(0),
      generated_primaries(true),
      beamstrahlung(false),
      photon_emission(false),
      spin_transport(false),
      sokolov_ternov(false),
      coherent_secondaries(false),
      trident_secondaries(false),
      tracked_pairs(false),
      tracked_muons(false),
      full_electron_luminosity(false)
{
}

bool resident_primary_contract_supported(
    const ResidentPrimaryContract& contract,
    std::string& reason)
{
  reason.clear();
  if (contract.interpolation != 2)
    reason = "resident primary pipeline requires CIC interpolation=2";
  else if (contract.extra_grids != 0)
    reason = "resident primary pipeline requires extra_grids=0";
  else if (!contract.generated_primaries)
    reason = "resident primary pipeline currently requires generated primaries";
  else if (contract.beamstrahlung)
    reason = "resident primary beamstrahlung is not implemented";
  else if (contract.photon_emission)
    reason = "resident primary photon emission is not implemented";
  else if (contract.spin_transport)
    reason = "resident primary spin transport is not implemented";
  else if (contract.sokolov_ternov)
    reason = "resident primary Sokolov-Ternov transport is not implemented";
  else if (contract.coherent_secondaries)
    reason = "resident primary coherent secondaries are not implemented";
  else if (contract.trident_secondaries)
    reason = "resident primary trident secondaries are not implemented";
  else if (contract.tracked_pairs)
    reason = "resident primary pair tracking is not implemented";
  else if (contract.tracked_muons)
    reason = "resident primary muon tracking is not implemented";
  else if (contract.full_electron_luminosity)
    reason = "resident primary full electron-luminosity events and spectra are not implemented; only the raw fine-grid dot product is available";
  return reason.empty();
}

ResidentPrimaryOptions::ResidentPrimaryOptions()
    : device(0), pinned_io(false), memory_fraction(0.80), contract()
{
}

ResidentPrimaryEncounter::ResidentPrimaryEncounter()
    : wave_index(0),
      beam1_offset(0),
      beam1_count(0),
      beam2_offset(0),
      beam2_count(0),
      charge1(0.0),
      charge2(0.0),
      step(0.0f),
      scale1(1.0f),
      scale2(1.0f),
      charge_sign(-1.0f),
      four_way_symmetry(false)
{
}

ResidentPrimaryGrid::ResidentPrimaryGrid()
    : delta_x_inverse(0.0f),
      delta_y_inverse(0.0f),
      offset_x(0.0f),
      offset_y(0.0f),
      physical_min_x(0.0f),
      physical_max_x(0.0f),
      physical_min_y(0.0f),
      physical_max_y(0.0f)
{
}

ResidentPrimaryStats::ResidentPrimaryStats()
    : upload_calls(0),
      batch_calls(0),
      download_calls(0),
      waves(0),
      encounters(0),
      particle_updates(0),
      deposition_calls(0),
      field_solve_calls(0),
      gather_push_calls(0),
      deposit_beam1_in(0),
      deposit_beam1_out(0),
      deposit_beam2_in(0),
      deposit_beam2_out(0),
      gather_beam1_in(0),
      gather_beam1_out(0),
      gather_beam2_in(0),
      gather_beam2_out(0),
      h2d_bytes(0),
      d2h_bytes(0),
      h2d_operations(0),
      d2h_operations(0),
      synchronizations(0),
      setup_h2d_bytes(0),
      setup_h2d_operations(0),
      setup_synchronizations(0),
      intermediate_particle_h2d_bytes(0),
      intermediate_particle_d2h_bytes(0),
      host_field_materializations(0),
      nonfinite_state_events(0),
      state_epoch(0),
      particle_capacity1(0),
      particle_capacity2(0),
      particle_count1(0),
      particle_count2(0),
      resident_bytes(0),
      device_free_bytes_at_init(0),
      device_total_bytes(0),
      memory_budget_bytes(0),
      upload_ms(0.0),
      kernel_ms(0.0),
      download_ms(0.0),
      fine_lumi_dot_sum(0.0),
      state_resident(false),
      context_poisoned(false)
{
}

struct ResidentPrimaryPipeline::Impl
{
  int device;
  bool initialized;
  bool poisoned;
  bool state_ready;
  bool pinned_io;
  int nx;
  int ny;
  int padded_nx;
  int padded_ny;
  std::size_t cells;
  std::size_t padded_cells;
  std::size_t cell_bytes;
  std::size_t spectrum_bytes;
  std::size_t capacity1;
  std::size_t capacity2;
  std::size_t count1;
  std::size_t count2;
  ResidentPrimaryGrid bound_grid;
  float spectrum_charge_sign;
  std::string device_text;
  std::string runtime_text;

  cudaStream_t stream;
  cudaEvent_t event_start;
  cudaEvent_t event_end;
  cufftHandle plan;
  bool plan_valid;
  void* workspace;
  cufftDoubleComplex* spectrum;
  cufftDoubleComplex* dist;
  double* rho1;
  double* rho2;
  double* phi1;
  double* phi2;
  float* xy1;
  float* velocity1;
  float* energy1;
  float* upsilon1;
  float* xy2;
  float* velocity2;
  float* energy2;
  float* upsilon2;
  DevicePrimaryMetrics* device_metrics;

  float* pinned_xy1;
  float* pinned_velocity1;
  float* pinned_energy1;
  float* pinned_upsilon1;
  float* pinned_xy2;
  float* pinned_velocity2;
  float* pinned_energy2;
  float* pinned_upsilon2;
  DevicePrimaryMetrics* pinned_metrics;

  ResidentPrimaryStats statistics;

  Impl()
      : device(-1),
        initialized(false),
        poisoned(false),
        state_ready(false),
        pinned_io(false),
        nx(0),
        ny(0),
        padded_nx(0),
        padded_ny(0),
        cells(0),
        padded_cells(0),
        cell_bytes(0),
        spectrum_bytes(0),
        capacity1(0),
        capacity2(0),
        count1(0),
        count2(0),
        bound_grid(),
        spectrum_charge_sign(-1.0f),
        stream(0),
        event_start(0),
        event_end(0),
        plan(0),
        plan_valid(false),
        workspace(0),
        spectrum(0),
        dist(0),
        rho1(0),
        rho2(0),
        phi1(0),
        phi2(0),
        xy1(0),
        velocity1(0),
        energy1(0),
        upsilon1(0),
        xy2(0),
        velocity2(0),
        energy2(0),
        upsilon2(0),
        device_metrics(0),
        pinned_xy1(0),
        pinned_velocity1(0),
        pinned_energy1(0),
        pinned_upsilon1(0),
        pinned_xy2(0),
        pinned_velocity2(0),
        pinned_energy2(0),
        pinned_upsilon2(0),
        pinned_metrics(0)
  {
  }

  void release()
  {
    int previous_device = -1;
    bool restore_device = false;
    bool handles_valid = true;
    if (device >= 0)
      {
        if (cudaGetDevice(&previous_device) != cudaSuccess)
          handles_valid = false;
        else if (previous_device != device)
          {
            if (cudaSetDevice(device) != cudaSuccess)
              handles_valid = false;
            else
              restore_device = true;
          }
      }
    // Never free device or pinned DMA storage unless all queued work is known
    // quiescent.  A reset/lost WSL context invalidates handles and the runtime
    // reclaims them; abandoning is safer than invoking destroy/free on invalid
    // objects or freeing host storage still targeted by DMA.
    if (handles_valid && stream &&
        cudaStreamSynchronize(stream) != cudaSuccess)
      handles_valid = false;
    if (!handles_valid)
      {
        if (restore_device) (void)cudaSetDevice(previous_device);
        *this = Impl();
        return;
      }
    if (plan_valid) cufftDestroy(plan);
    if (workspace) cudaFree(workspace);
    if (device_metrics) cudaFree(device_metrics);
    if (upsilon2) cudaFree(upsilon2);
    if (energy2) cudaFree(energy2);
    if (velocity2) cudaFree(velocity2);
    if (xy2) cudaFree(xy2);
    if (upsilon1) cudaFree(upsilon1);
    if (energy1) cudaFree(energy1);
    if (velocity1) cudaFree(velocity1);
    if (xy1) cudaFree(xy1);
    if (phi2) cudaFree(phi2);
    if (phi1) cudaFree(phi1);
    if (rho2) cudaFree(rho2);
    if (rho1) cudaFree(rho1);
    if (dist) cudaFree(dist);
    if (spectrum) cudaFree(spectrum);
    if (pinned_metrics) cudaFreeHost(pinned_metrics);
    if (pinned_upsilon2) cudaFreeHost(pinned_upsilon2);
    if (pinned_energy2) cudaFreeHost(pinned_energy2);
    if (pinned_velocity2) cudaFreeHost(pinned_velocity2);
    if (pinned_xy2) cudaFreeHost(pinned_xy2);
    if (pinned_upsilon1) cudaFreeHost(pinned_upsilon1);
    if (pinned_energy1) cudaFreeHost(pinned_energy1);
    if (pinned_velocity1) cudaFreeHost(pinned_velocity1);
    if (pinned_xy1) cudaFreeHost(pinned_xy1);
    if (event_end) cudaEventDestroy(event_end);
    if (event_start) cudaEventDestroy(event_start);
    if (stream) cudaStreamDestroy(stream);
    if (restore_device) (void)cudaSetDevice(previous_device);
    *this = Impl();
  }

  bool reject_if_unavailable(std::string& error)
  {
    if (!initialized)
      {
        error = "resident primary pipeline is not initialized";
        return true;
      }
    if (poisoned)
      {
        error = "resident primary CUDA context is poisoned; retry or CPU fallback is forbidden";
        return true;
      }
    return false;
  }

  void poison(std::string cause, std::string& error)
  {
    poisoned = true;
    state_ready = false;
    statistics.context_poisoned = true;
    statistics.state_resident = false;
    std::ostringstream message;
    message << cause << "; resident primary CUDA context poisoned";
    if (stream)
      {
        const cudaError_t drained = cudaStreamSynchronize(stream);
        message << (drained == cudaSuccess ? "; queued work drained"
                                           : "; stream drain failed");
      }
    message << "; outputs are undefined and retry/fallback is forbidden";
    error = message.str();
  }

  void poison_without_drain(std::string cause, std::string& error)
  {
    poisoned = true;
    state_ready = false;
    statistics.context_poisoned = true;
    statistics.state_resident = false;
    error = cause +
        "; resident primary CUDA context poisoned; stream state could not be "
        "established, so retry/fallback is forbidden";
  }
};

ResidentPrimaryPipeline::ResidentPrimaryPipeline() : impl_(new Impl)
{
}

ResidentPrimaryPipeline::~ResidentPrimaryPipeline()
{
  impl_->release();
  delete impl_;
}

bool ResidentPrimaryPipeline::initialize(
    const ResidentPrimaryOptions& options,
    int nx,
    int ny,
    const double* dist_interleaved,
    std::size_t dist_interleaved_values,
    const ResidentPrimaryGrid& spectrum_grid,
    float spectrum_charge_sign,
    std::size_t maximum_particles1,
    std::size_t maximum_particles2,
    std::string& error)
{
  error.clear();
  if (impl_->initialized || impl_->device >= 0)
    {
      error = "resident primary pipeline is already initialized";
      return false;
    }
  if (!resident_primary_contract_supported(options.contract, error))
    return false;
  if (options.device < 0 || nx < 3 || ny < 3 || !dist_interleaved ||
      !std::isfinite(options.memory_fraction) ||
      options.memory_fraction <= 0.0 || options.memory_fraction > 1.0)
    {
      error = "resident primary initialization requires a valid device, grid >=3x3, Green spectrum, and 0<memory_fraction<=1";
      return false;
    }
  if (!valid_grid(spectrum_grid) ||
      !(spectrum_charge_sign == -1.0f ||
        spectrum_charge_sign == 0.0f))
    {
      error = "resident primary initialization requires valid immutable mesh provenance and spectrum charge_sign=-1 or 0";
      return false;
    }
  if (nx > std::numeric_limits<int>::max() / 2 ||
      ny > std::numeric_limits<int>::max() / 2)
    {
      error = "resident primary padded field dimensions overflow cuFFT int dimensions";
      return false;
    }

  std::size_t particle_total = 0;
  std::size_t particle_values = 0;
  std::size_t particle_bytes = 0;
  std::size_t cells = 0;
  std::size_t padded_cells = 0;
  std::size_t cell_bytes = 0;
  std::size_t spectrum_bytes = 0;
  std::size_t expected_dist_values = 0;
  if (!checked_sum(maximum_particles1, maximum_particles2, particle_total) ||
      !checked_product(particle_total, static_cast<std::size_t>(6),
                       particle_values) ||
      !checked_product(particle_values, sizeof(float), particle_bytes) ||
      !checked_product(static_cast<std::size_t>(nx),
                       static_cast<std::size_t>(ny), cells) ||
      !checked_product(static_cast<std::size_t>(2 * nx),
                       static_cast<std::size_t>(2 * ny), padded_cells) ||
      !checked_product(cells, sizeof(double), cell_bytes) ||
      !checked_product(padded_cells, sizeof(cufftDoubleComplex),
                       spectrum_bytes) ||
      !checked_product(padded_cells, static_cast<std::size_t>(2),
                       expected_dist_values))
    {
      error = "resident primary dimensions overflow size_t";
      return false;
    }
  if (dist_interleaved_values != expected_dist_values)
    {
      std::ostringstream message;
      message << "resident primary Green spectrum requires exactly "
              << expected_dist_values << " interleaved FP64 values; received "
              << dist_interleaved_values;
      error = message.str();
      return false;
    }
  for (std::size_t value = 0; value < expected_dist_values; ++value)
    if (!std::isfinite(dist_interleaved[value]))
      {
        error = "resident primary Green spectrum contains a non-finite value";
        return false;
      }

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error))
    return false;
  if (options.device >= device_count)
    {
      std::ostringstream message;
      message << "CUDA device " << options.device << " is unavailable; found "
              << device_count << " device(s)";
      error = message.str();
      return false;
    }
  ScopedCudaDevice selected_device(
      options.device, "cudaSetDevice(resident primary initialize)", error);
  if (!selected_device.selected())
    return false;
  impl_->device = options.device;

  cudaDeviceProp properties;
  if (!cuda_ok(cudaGetDeviceProperties(&properties, options.device),
               "cudaGetDeviceProperties", error))
    {
      impl_->release();
      return false;
    }
  if (properties.major < 6)
    {
      std::ostringstream message;
      message << "resident primary FP64 atomic CIC requires compute capability 6.0 or newer; selected device is "
              << properties.major << '.' << properties.minor;
      error = message.str();
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
  impl_->capacity1 = maximum_particles1;
  impl_->capacity2 = maximum_particles2;
  impl_->pinned_io = options.pinned_io;
  impl_->bound_grid = spectrum_grid;
  impl_->spectrum_charge_sign = spectrum_charge_sign;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(resident primary preflight)", error))
    {
      impl_->release();
      return false;
    }
  impl_->statistics.device_free_bytes_at_init = free_bytes;
  impl_->statistics.device_total_bytes = total_bytes;
  impl_->statistics.memory_budget_bytes = static_cast<std::size_t>(
      options.memory_fraction * static_cast<double>(total_bytes));
  impl_->statistics.particle_capacity1 = maximum_particles1;
  impl_->statistics.particle_capacity2 = maximum_particles2;

  if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(resident primary)", error) ||
      !cuda_ok(cudaEventCreate(&impl_->event_start),
               "cudaEventCreate(resident primary start)", error) ||
      !cuda_ok(cudaEventCreate(&impl_->event_end),
               "cudaEventCreate(resident primary end)", error) ||
      !cufft_ok(cufftCreate(&impl_->plan),
                "cufftCreate(resident primary)", error))
    {
      impl_->release();
      return false;
    }
  impl_->plan_valid = true;
  if (!cufft_ok(cufftSetAutoAllocation(impl_->plan, 0),
                "cufftSetAutoAllocation(resident primary)", error))
    {
      impl_->release();
      return false;
    }
  std::size_t workspace_bytes = 0;
  if (!cufft_ok(cufftMakePlan2d(impl_->plan, 2 * nx, 2 * ny, CUFFT_Z2Z,
                                &workspace_bytes),
                "cufftMakePlan2d(resident primary)", error))
    {
      impl_->release();
      return false;
    }

  std::size_t required_bytes = 0;
  std::size_t field_arrays = 0;
  if (!checked_product(spectrum_bytes, static_cast<std::size_t>(2),
                       required_bytes) ||
      !checked_product(cell_bytes, static_cast<std::size_t>(4),
                       field_arrays) ||
      !checked_sum(required_bytes, field_arrays, required_bytes) ||
      !checked_sum(required_bytes, particle_bytes, required_bytes) ||
      !checked_sum(required_bytes, sizeof(DevicePrimaryMetrics),
                   required_bytes) ||
      !checked_sum(required_bytes, workspace_bytes, required_bytes))
    {
      error = "resident primary allocation total overflows size_t";
      impl_->release();
      return false;
    }
  if (required_bytes > free_bytes ||
      required_bytes > impl_->statistics.memory_budget_bytes)
    {
      std::ostringstream message;
      message << "resident primary context requires " << required_bytes
              << " device bytes, exceeding "
              << (required_bytes > impl_->statistics.memory_budget_bytes
                      ? "memory_fraction budget " : "currently free memory ")
              << (required_bytes > impl_->statistics.memory_budget_bytes
                      ? impl_->statistics.memory_budget_bytes : free_bytes);
      error = message.str();
      impl_->release();
      return false;
    }

  const std::size_t xy_bytes1 = 2 * maximum_particles1 * sizeof(float);
  const std::size_t xy_bytes2 = 2 * maximum_particles2 * sizeof(float);
  const std::size_t energy_bytes1 = maximum_particles1 * sizeof(float);
  const std::size_t energy_bytes2 = maximum_particles2 * sizeof(float);
  const auto allocate_device = [&error](void** pointer,
                                        std::size_t bytes,
                                        const char* label) -> bool {
    if (!bytes)
      {
        *pointer = 0;
        return true;
      }
    return cuda_ok(cudaMalloc(pointer, bytes), label, error);
  };
  if (!allocate_device(reinterpret_cast<void**>(&impl_->spectrum),
                       spectrum_bytes, "cudaMalloc(primary spectrum)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->dist),
                       spectrum_bytes, "cudaMalloc(primary Green spectrum)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->rho1),
                       cell_bytes, "cudaMalloc(primary rho1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->rho2),
                       cell_bytes, "cudaMalloc(primary rho2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->phi1),
                       cell_bytes, "cudaMalloc(primary phi1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->phi2),
                       cell_bytes, "cudaMalloc(primary phi2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->xy1),
                       xy_bytes1, "cudaMalloc(primary xy1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->velocity1),
                       xy_bytes1, "cudaMalloc(primary velocity1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->energy1),
                       energy_bytes1, "cudaMalloc(primary energy1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->upsilon1),
                       energy_bytes1, "cudaMalloc(primary upsilon1)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->xy2),
                       xy_bytes2, "cudaMalloc(primary xy2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->velocity2),
                       xy_bytes2, "cudaMalloc(primary velocity2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->energy2),
                       energy_bytes2, "cudaMalloc(primary energy2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->upsilon2),
                       energy_bytes2, "cudaMalloc(primary upsilon2)") ||
      !allocate_device(reinterpret_cast<void**>(&impl_->device_metrics),
                       sizeof(DevicePrimaryMetrics),
                       "cudaMalloc(primary metrics)") ||
      (workspace_bytes &&
       !allocate_device(&impl_->workspace, workspace_bytes,
                        "cudaMalloc(primary cuFFT workspace)")))
    {
      impl_->release();
      return false;
    }
  if (workspace_bytes &&
      !cufft_ok(cufftSetWorkArea(impl_->plan, impl_->workspace),
                "cufftSetWorkArea(resident primary)", error))
    {
      impl_->release();
      return false;
    }
  if (!cufft_ok(cufftSetStream(impl_->plan, impl_->stream),
                "cufftSetStream(resident primary)", error))
    {
      impl_->release();
      return false;
    }

  const auto allocate_pinned = [&error](void** pointer,
                                        std::size_t bytes,
                                        const char* label) -> bool {
    if (!bytes)
      {
        *pointer = 0;
        return true;
      }
    return cuda_ok(cudaHostAlloc(pointer, bytes, cudaHostAllocDefault),
                   label, error);
  };
  // The compact metrics copy is asynchronous for every supported I/O mode.
  // Keep its destination page-locked even when the bulk state uses pageable
  // memory; otherwise cudaMemcpyAsync may perform an implicit host-side stage
  // and the explicit synchronization telemetry would be misleading.
  if (!allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_metrics),
                       sizeof(DevicePrimaryMetrics),
                       "cudaHostAlloc(primary metrics)") ||
      (options.pinned_io &&
       (!allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_xy1),
                         xy_bytes1, "cudaHostAlloc(primary xy1)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_velocity1),
                        xy_bytes1, "cudaHostAlloc(primary velocity1)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_energy1),
                        energy_bytes1, "cudaHostAlloc(primary energy1)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_upsilon1),
                        energy_bytes1, "cudaHostAlloc(primary upsilon1)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_xy2),
                        xy_bytes2, "cudaHostAlloc(primary xy2)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_velocity2),
                        xy_bytes2, "cudaHostAlloc(primary velocity2)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_energy2),
                        energy_bytes2, "cudaHostAlloc(primary energy2)") ||
       !allocate_pinned(reinterpret_cast<void**>(&impl_->pinned_upsilon2),
                        energy_bytes2, "cudaHostAlloc(primary upsilon2)"))))
    {
      impl_->release();
      return false;
    }

  if (!cuda_ok(cudaMemcpyAsync(impl_->dist, dist_interleaved,
                               spectrum_bytes, cudaMemcpyHostToDevice,
                               impl_->stream),
               "cudaMemcpyAsync(primary Green spectrum H2D)", error) ||
      !cuda_ok(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize(primary initialization)", error))
    {
      impl_->release();
      return false;
    }

  impl_->statistics.resident_bytes = required_bytes;
  impl_->statistics.setup_h2d_bytes = spectrum_bytes;
  impl_->statistics.setup_h2d_operations = 1;
  impl_->statistics.setup_synchronizations = 1;
  std::ostringstream device_text;
  device_text << properties.name << " (device " << options.device
              << ", compute " << properties.major << '.' << properties.minor
              << ", " << properties.totalGlobalMem / (1024 * 1024)
              << " MiB VRAM)";
  impl_->device_text = device_text.str();
  int runtime_version = -1;
  int driver_version = -1;
  int cufft_version = -1;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  cufftGetVersion(&cufft_version);
  std::ostringstream runtime_text;
  runtime_text << "CUDA runtime " << version_text(runtime_version)
               << ", driver " << version_text(driver_version)
               << ", cuFFT " << version_text(cufft_version)
               << ", CUFFT_Z2Z FP64 " << 2 * nx << 'x' << 2 * ny
               << ", whole-bunch particle state resident"
               << ", deposition=atomic-fp64"
               << ", host_field_materialization=none"
               << ", radiation=unsupported"
               << ", full_luminosity=unsupported"
               << ", pinned_staging="
               << (options.pinned_io ? "on" : "off");
  impl_->runtime_text = runtime_text.str();
  impl_->initialized = true;
  return true;
}

bool ResidentPrimaryPipeline::is_available() const
{
  return impl_->initialized && !impl_->poisoned;
}

bool ResidentPrimaryPipeline::state_is_resident() const
{
  return is_available() && impl_->state_ready;
}

std::string ResidentPrimaryPipeline::device_description() const
{
  if (!impl_->initialized) return "CUDA not initialized";
  return impl_->poisoned ? impl_->device_text + " [POISONED]"
                         : impl_->device_text;
}

std::string ResidentPrimaryPipeline::runtime_description() const
{
  if (!impl_->initialized) return "CUDA/cuFFT runtime not initialized";
  return impl_->poisoned ? impl_->runtime_text + ", context=POISONED"
                         : impl_->runtime_text;
}

ResidentPrimaryStats ResidentPrimaryPipeline::stats() const
{
  return impl_->statistics;
}

bool ResidentPrimaryPipeline::upload(
    const float* xy1,
    const float* velocity_xy1,
    const float* energy1,
    const float* upsilon1,
    std::size_t particle_count1,
    const float* xy2,
    const float* velocity_xy2,
    const float* energy2,
    const float* upsilon2,
    std::size_t particle_count2,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  error.clear();
  if (impl_->reject_if_unavailable(error)) return false;
  if (particle_count1 > impl_->capacity1 ||
      particle_count2 > impl_->capacity2)
    {
      std::ostringstream message;
      message << "resident primary upload exceeds reserved capacity: "
              << particle_count1 << '/' << impl_->capacity1 << " and "
              << particle_count2 << '/' << impl_->capacity2;
      error = message.str();
      return false;
    }
  if ((particle_count1 &&
       (!xy1 || !velocity_xy1 || !energy1 || !upsilon1)) ||
      (particle_count2 &&
       (!xy2 || !velocity_xy2 || !energy2 || !upsilon2)))
    {
      error = "resident primary upload received a null active state array";
      return false;
    }
  for (std::size_t particle = 0; particle < particle_count1; ++particle)
    if (!std::isfinite(xy1[2 * particle]) ||
        !std::isfinite(xy1[2 * particle + 1]) ||
        !std::isfinite(velocity_xy1[2 * particle]) ||
        !std::isfinite(velocity_xy1[2 * particle + 1]) ||
        !std::isfinite(energy1[particle]) || energy1[particle] <= 0.0f ||
        !std::isfinite(upsilon1[particle]) || upsilon1[particle] < 0.0f ||
        !finite_particle_grid_arithmetic(
            impl_->bound_grid, xy1[2 * particle], xy1[2 * particle + 1]))
      {
        error = "resident primary beam1 state and mesh transforms must be finite with positive energy and nonnegative Upsilon";
        return false;
      }
  for (std::size_t particle = 0; particle < particle_count2; ++particle)
    if (!std::isfinite(xy2[2 * particle]) ||
        !std::isfinite(xy2[2 * particle + 1]) ||
        !std::isfinite(velocity_xy2[2 * particle]) ||
        !std::isfinite(velocity_xy2[2 * particle + 1]) ||
        !std::isfinite(energy2[particle]) || energy2[particle] <= 0.0f ||
        !std::isfinite(upsilon2[particle]) || upsilon2[particle] < 0.0f ||
        !finite_particle_grid_arithmetic(
            impl_->bound_grid, xy2[2 * particle], xy2[2 * particle + 1]))
      {
        error = "resident primary beam2 state and mesh transforms must be finite with positive energy and nonnegative Upsilon";
        return false;
      }

  const std::size_t xy_bytes1 = 2 * particle_count1 * sizeof(float);
  const std::size_t xy_bytes2 = 2 * particle_count2 * sizeof(float);
  const std::size_t energy_bytes1 = particle_count1 * sizeof(float);
  const std::size_t energy_bytes2 = particle_count2 * sizeof(float);
  std::size_t beam1_transfer_bytes = 0;
  std::size_t beam2_transfer_bytes = 0;
  std::size_t transfer_bytes_size = 0;
  std::uint64_t transfer_bytes = 0;
  if (!checked_sum(xy_bytes1, xy_bytes1, beam1_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, energy_bytes1,
                   beam1_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, energy_bytes1,
                   beam1_transfer_bytes) ||
      !checked_sum(xy_bytes2, xy_bytes2, beam2_transfer_bytes) ||
      !checked_sum(beam2_transfer_bytes, energy_bytes2,
                   beam2_transfer_bytes) ||
      !checked_sum(beam2_transfer_bytes, energy_bytes2,
                   beam2_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, beam2_transfer_bytes,
                   transfer_bytes_size) ||
      !size_to_u64(transfer_bytes_size, transfer_bytes))
    {
      error = "resident primary upload byte accounting overflows";
      return false;
    }
  const std::uint64_t transfer_operations =
      4 * static_cast<std::uint64_t>(particle_count1 != 0) +
      4 * static_cast<std::uint64_t>(particle_count2 != 0);
  const ResidentPrimaryStats& prior = impl_->statistics;
  if (!can_add_u64(prior.upload_calls, 1) ||
      !can_add_u64(prior.state_epoch, 1) ||
      !can_add_u64(prior.h2d_bytes, transfer_bytes) ||
      !can_add_u64(prior.h2d_operations, transfer_operations) ||
      !can_add_u64(prior.synchronizations, 1))
    {
      error = "resident primary upload telemetry counter would overflow";
      return false;
    }
  ScopedCudaDevice selected_device(
      impl_->device, "cudaSetDevice(resident primary upload)", error);
  if (!selected_device.selected())
    {
      impl_->poison_without_drain(error, error);
      statistics = impl_->statistics;
      return false;
    }

  const float* source_xy1 = xy1;
  const float* source_velocity1 = velocity_xy1;
  const float* source_energy1 = energy1;
  const float* source_upsilon1 = upsilon1;
  const float* source_xy2 = xy2;
  const float* source_velocity2 = velocity_xy2;
  const float* source_energy2 = energy2;
  const float* source_upsilon2 = upsilon2;
  if (impl_->pinned_io)
    {
      if (xy_bytes1)
        {
          std::memcpy(impl_->pinned_xy1, xy1, xy_bytes1);
          std::memcpy(impl_->pinned_velocity1, velocity_xy1, xy_bytes1);
          std::memcpy(impl_->pinned_energy1, energy1, energy_bytes1);
          std::memcpy(impl_->pinned_upsilon1, upsilon1, energy_bytes1);
        }
      if (xy_bytes2)
        {
          std::memcpy(impl_->pinned_xy2, xy2, xy_bytes2);
          std::memcpy(impl_->pinned_velocity2, velocity_xy2, xy_bytes2);
          std::memcpy(impl_->pinned_energy2, energy2, energy_bytes2);
          std::memcpy(impl_->pinned_upsilon2, upsilon2, energy_bytes2);
        }
      source_xy1 = impl_->pinned_xy1;
      source_velocity1 = impl_->pinned_velocity1;
      source_energy1 = impl_->pinned_energy1;
      source_upsilon1 = impl_->pinned_upsilon1;
      source_xy2 = impl_->pinned_xy2;
      source_velocity2 = impl_->pinned_velocity2;
      source_energy2 = impl_->pinned_energy2;
      source_upsilon2 = impl_->pinned_upsilon2;
    }

  const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  bool ok = cuda_ok(cudaEventRecord(impl_->event_start, impl_->stream),
                    "cudaEventRecord(primary upload start)", error);
  const auto copy_h2d = [&error, this](void* destination,
                                      const void* source,
                                      std::size_t bytes,
                                      const char* label) -> bool {
    return !bytes || cuda_ok(cudaMemcpyAsync(destination, source, bytes,
                                              cudaMemcpyHostToDevice,
                                              impl_->stream), label, error);
  };
  ok = ok && copy_h2d(impl_->xy1, source_xy1, xy_bytes1,
                      "cudaMemcpyAsync(primary xy1 H2D)");
  ok = ok && copy_h2d(impl_->velocity1, source_velocity1, xy_bytes1,
                      "cudaMemcpyAsync(primary velocity1 H2D)");
  ok = ok && copy_h2d(impl_->energy1, source_energy1, energy_bytes1,
                      "cudaMemcpyAsync(primary energy1 H2D)");
  ok = ok && copy_h2d(impl_->upsilon1, source_upsilon1, energy_bytes1,
                      "cudaMemcpyAsync(primary upsilon1 H2D)");
  ok = ok && copy_h2d(impl_->xy2, source_xy2, xy_bytes2,
                      "cudaMemcpyAsync(primary xy2 H2D)");
  ok = ok && copy_h2d(impl_->velocity2, source_velocity2, xy_bytes2,
                      "cudaMemcpyAsync(primary velocity2 H2D)");
  ok = ok && copy_h2d(impl_->energy2, source_energy2, energy_bytes2,
                      "cudaMemcpyAsync(primary energy2 H2D)");
  ok = ok && copy_h2d(impl_->upsilon2, source_upsilon2, energy_bytes2,
                      "cudaMemcpyAsync(primary upsilon2 H2D)");
  ok = ok && cuda_ok(cudaEventRecord(impl_->event_end, impl_->stream),
                     "cudaEventRecord(primary upload end)", error);
  ok = ok && cuda_ok(cudaStreamSynchronize(impl_->stream),
                     "cudaStreamSynchronize(primary upload)", error);
  if (!ok)
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }
  float elapsed_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, impl_->event_start,
                                    impl_->event_end),
               "cudaEventElapsedTime(primary upload)", error))
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }

  impl_->count1 = particle_count1;
  impl_->count2 = particle_count2;
  impl_->state_ready = true;
  ResidentPrimaryStats& totals = impl_->statistics;
  ++totals.upload_calls;
  ++totals.state_epoch;
  totals.particle_count1 = particle_count1;
  totals.particle_count2 = particle_count2;
  totals.state_resident = true;
  totals.h2d_bytes += transfer_bytes;
  totals.h2d_operations += transfer_operations;
  ++totals.synchronizations;
  totals.upload_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  statistics = totals;
  return true;
}

bool ResidentPrimaryPipeline::run(
    const ResidentPrimaryGrid& grid,
    const ResidentPrimaryEncounter* encounters,
    std::size_t encounter_count,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  error.clear();
  if (impl_->reject_if_unavailable(error)) return false;
  if (!impl_->state_ready)
    {
      error = "resident primary run requires an uploaded whole-bunch state";
      return false;
    }
  if (!encounters || !encounter_count)
    {
      error = "resident primary run requires a nonempty encounter schedule";
      return false;
    }
  if (!same_grid(grid, impl_->bound_grid))
    {
      error = "resident primary run mesh differs from the immutable mesh used to build its Green spectrum";
      return false;
    }

  std::uint64_t waves = 0;
  std::uint64_t particle_updates = 0;
  std::uint64_t nonzero_field_solves = 0;
  std::uint64_t encounter_count_u64 = 0;
  if (!size_to_u64(encounter_count, encounter_count_u64))
    {
      error = "resident primary encounter count cannot be represented in telemetry";
      return false;
    }
  std::size_t wave_begin = 0;
  for (std::size_t index = 0; index < encounter_count; ++index)
    {
      const ResidentPrimaryEncounter& encounter = encounters[index];
      if (!range_fits(encounter.beam1_offset, encounter.beam1_count,
                      impl_->count1) ||
          !range_fits(encounter.beam2_offset, encounter.beam2_count,
                      impl_->count2))
        {
          std::ostringstream message;
          message << "resident primary encounter " << index
                  << " exceeds uploaded particle ranges";
          error = message.str();
          return false;
        }
      const float position_step1 = encounter.step * encounter.scale1;
      const float position_step2 = encounter.step * encounter.scale2;
      const float dz1 = static_cast<float>(
          static_cast<double>(encounter.step) * 1.0e-9 *
          static_cast<double>(encounter.scale1));
      const float dz2 = static_cast<float>(
          static_cast<double>(encounter.step) * 1.0e-9 *
          static_cast<double>(encounter.scale2));
      if (!std::isfinite(encounter.charge1) || encounter.charge1 < 0.0 ||
          !std::isfinite(encounter.charge2) || encounter.charge2 < 0.0 ||
          (encounter.beam1_count && encounter.charge1 <= 0.0) ||
          (encounter.beam2_count && encounter.charge2 <= 0.0) ||
          !std::isfinite(encounter.step) || encounter.step <= 0.0f ||
          !std::isfinite(encounter.scale1) || encounter.scale1 <= 0.0f ||
          !std::isfinite(encounter.scale2) || encounter.scale2 <= 0.0f ||
          !std::isfinite(position_step1) ||
          !std::isfinite(position_step2) ||
          !std::isfinite(dz1) || dz1 <= 0.0f ||
          !std::isfinite(dz2) || dz2 <= 0.0f ||
          encounter.charge_sign != impl_->spectrum_charge_sign ||
          (encounter.four_way_symmetry &&
           ((impl_->nx % 2) != 0 || (impl_->ny % 2) != 0)))
        {
          std::ostringstream message;
          message << "resident primary encounter " << index
                  << " violates the nonnegative-charge, finite-step, immutable-spectrum-sign, or even-symmetry-grid contract";
          error = message.str();
          return false;
        }
      if (index == 0 ||
          encounter.wave_index != encounters[index - 1].wave_index)
        {
          if (index && encounter.wave_index < encounters[index - 1].wave_index)
            {
              error = "resident primary wave indices must be nondecreasing";
              return false;
            }
          wave_begin = index;
          if (!can_add_u64(waves, 1))
            {
              error = "resident primary wave counter overflows uint64";
              return false;
            }
          ++waves;
        }
      for (std::size_t prior = wave_begin; prior < index; ++prior)
        if (index_ranges_overlap(encounter.beam1_offset,
                                 encounter.beam1_count,
                                 encounters[prior].beam1_offset,
                                 encounters[prior].beam1_count) ||
            index_ranges_overlap(encounter.beam2_offset,
                                 encounter.beam2_count,
                                 encounters[prior].beam2_offset,
                                 encounters[prior].beam2_count))
          {
            std::ostringstream message;
            message << "resident primary wave " << encounter.wave_index
                    << " reuses a particle range between encounters "
                    << prior << " and " << index;
            error = message.str();
            return false;
          }
      std::uint64_t beam1_updates = 0;
      std::uint64_t beam2_updates = 0;
      std::uint64_t next_updates = 0;
      if (!size_to_u64(encounter.beam1_count, beam1_updates) ||
          !size_to_u64(encounter.beam2_count, beam2_updates) ||
          !checked_u64_sum(particle_updates, beam1_updates, next_updates) ||
          !checked_u64_sum(next_updates, beam2_updates, next_updates))
        {
          error = "resident primary particle-update counter overflows uint64";
          return false;
        }
      particle_updates = next_updates;
      if (std::fabs(encounter.charge_sign) >= 1.0e-5f)
        {
          if (!can_add_u64(nonzero_field_solves, 1))
            {
              error = "resident primary field-solve counter overflows uint64";
              return false;
            }
          ++nonzero_field_solves;
        }
    }

  std::uint64_t maximum_nonfinite_events = 0;
  if (!checked_u64_sum(particle_updates, particle_updates,
                       maximum_nonfinite_events))
    {
      error = "resident primary non-finite-event counter bound overflows uint64";
      return false;
    }
  const ResidentPrimaryStats& prior = impl_->statistics;
  if (!can_add_u64(prior.batch_calls, 1) ||
      !can_add_u64(prior.state_epoch, 1) ||
      !can_add_u64(prior.waves, waves) ||
      !can_add_u64(prior.encounters, encounter_count_u64) ||
      !can_add_u64(prior.particle_updates, particle_updates) ||
      !can_add_u64(prior.deposition_calls, encounter_count_u64) ||
      !can_add_u64(prior.field_solve_calls, nonzero_field_solves) ||
      !can_add_u64(prior.gather_push_calls, encounter_count_u64) ||
      !can_add_u64(prior.deposit_beam1_in, particle_updates) ||
      !can_add_u64(prior.deposit_beam1_out, particle_updates) ||
      !can_add_u64(prior.deposit_beam2_in, particle_updates) ||
      !can_add_u64(prior.deposit_beam2_out, particle_updates) ||
      !can_add_u64(prior.gather_beam1_in, particle_updates) ||
      !can_add_u64(prior.gather_beam1_out, particle_updates) ||
      !can_add_u64(prior.gather_beam2_in, particle_updates) ||
      !can_add_u64(prior.gather_beam2_out, particle_updates) ||
      !can_add_u64(prior.nonfinite_state_events,
                   maximum_nonfinite_events) ||
      !can_add_u64(prior.d2h_bytes,
                   static_cast<std::uint64_t>(sizeof(DevicePrimaryMetrics))) ||
      !can_add_u64(prior.d2h_operations, 1) ||
      !can_add_u64(prior.synchronizations, 1))
    {
      error = "resident primary batch telemetry counter would overflow";
      return false;
    }

  ScopedCudaDevice selected_device(
      impl_->device, "cudaSetDevice(resident primary run)", error);
  if (!selected_device.selected())
    {
      impl_->poison_without_drain(error, error);
      statistics = impl_->statistics;
      return false;
    }
  bool ok = cuda_ok(cudaEventRecord(impl_->event_start, impl_->stream),
                    "cudaEventRecord(primary batch start)", error) &&
            cuda_ok(cudaMemsetAsync(impl_->device_metrics, 0,
                                    sizeof(DevicePrimaryMetrics),
                                    impl_->stream),
                    "cudaMemsetAsync(primary batch metrics)", error);

  for (std::size_t index = 0; ok && index < encounter_count; ++index)
    {
      const ResidentPrimaryEncounter& encounter = encounters[index];
      ok = cuda_ok(cudaMemsetAsync(impl_->rho1, 0, impl_->cell_bytes,
                                   impl_->stream),
                   "cudaMemsetAsync(primary rho1)", error) &&
           cuda_ok(cudaMemsetAsync(impl_->rho2, 0, impl_->cell_bytes,
                                   impl_->stream),
                   "cudaMemsetAsync(primary rho2)", error);
      if (!ok) break;
      if (encounter.beam1_count)
        primary_cic_kernel<<<block_count(encounter.beam1_count), kThreads, 0,
                             impl_->stream>>>(
            impl_->xy1 + 2 * encounter.beam1_offset,
            encounter.beam1_count, encounter.charge1, 0,
            impl_->nx, impl_->ny, grid.delta_x_inverse,
            grid.delta_y_inverse, grid.offset_x, grid.offset_y,
            impl_->rho1, impl_->device_metrics);
      ok = cuda_ok(cudaGetLastError(),
                   "primary_cic_kernel beam1 launch", error);
      if (ok && encounter.beam2_count)
        primary_cic_kernel<<<block_count(encounter.beam2_count), kThreads, 0,
                             impl_->stream>>>(
            impl_->xy2 + 2 * encounter.beam2_offset,
            encounter.beam2_count, encounter.charge2, 1,
            impl_->nx, impl_->ny, grid.delta_x_inverse,
            grid.delta_y_inverse, grid.offset_x, grid.offset_y,
            impl_->rho2, impl_->device_metrics);
      ok = ok && cuda_ok(cudaGetLastError(),
                         "primary_cic_kernel beam2 launch", error);
      const std::size_t representatives =
          static_cast<std::size_t>(impl_->nx / 2) * (impl_->ny / 2);
      if (ok && encounter.four_way_symmetry && representatives)
        {
          symmetrize_four_way_kernel<<<block_count(representatives), kThreads,
                                       0, impl_->stream>>>(
              impl_->rho1, impl_->rho2, impl_->nx, impl_->ny);
          ok = cuda_ok(cudaGetLastError(),
                       "primary symmetrize kernel launch", error);
        }
      if (!ok) break;
      fine_lumi_dot_kernel<<<1, kThreads, 0, impl_->stream>>>(
          impl_->rho1, impl_->rho2, impl_->cells, impl_->device_metrics);
      ok = cuda_ok(cudaGetLastError(),
                   "primary fine-luminosity kernel launch", error);

      const bool zero_field =
          std::fabs(encounter.charge_sign) < 1.0e-5f;
      if (ok && !zero_field)
        {
          ok = cuda_ok(cudaMemsetAsync(impl_->spectrum, 0,
                                       impl_->spectrum_bytes, impl_->stream),
                       "cudaMemsetAsync(primary padded spectrum)", error);
          if (ok)
            pack_rho_kernel<<<block_count(impl_->cells), kThreads, 0,
                              impl_->stream>>>(
                impl_->rho1, impl_->rho2, impl_->nx, impl_->ny,
                impl_->padded_ny, impl_->spectrum);
          ok = ok && cuda_ok(cudaGetLastError(),
                             "primary pack-rho kernel launch", error) &&
               cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                      impl_->spectrum, CUFFT_FORWARD),
                        "cufftExecZ2Z(primary forward)", error);
          if (ok)
            multiply_spectrum_kernel<<<block_count(impl_->padded_cells),
                                       kThreads, 0, impl_->stream>>>(
                impl_->spectrum, impl_->dist, impl_->padded_cells);
          ok = ok && cuda_ok(cudaGetLastError(),
                             "primary multiply-spectrum kernel launch",
                             error) &&
               cufft_ok(cufftExecZ2Z(impl_->plan, impl_->spectrum,
                                      impl_->spectrum, CUFFT_INVERSE),
                        "cufftExecZ2Z(primary inverse)", error);
          if (ok)
            unpack_phi_kernel<<<block_count(impl_->cells), kThreads, 0,
                                impl_->stream>>>(
                impl_->spectrum, impl_->nx, impl_->ny, impl_->padded_ny,
                impl_->phi1, impl_->phi2);
          ok = ok && cuda_ok(cudaGetLastError(),
                             "primary unpack-phi kernel launch", error);
        }
      if (!ok) break;
      if (encounter.beam1_count)
        gather_and_push_kernel<<<block_count(encounter.beam1_count), kThreads,
                                 0, impl_->stream>>>(
            impl_->xy1 + 2 * encounter.beam1_offset,
            impl_->velocity1 + 2 * encounter.beam1_offset,
            impl_->energy1 + encounter.beam1_offset,
            impl_->upsilon1 + encounter.beam1_offset,
            encounter.beam1_count, impl_->phi2, 0, impl_->nx, impl_->ny,
            grid.delta_x_inverse, grid.delta_y_inverse,
            grid.offset_x, grid.offset_y,
            grid.physical_min_x, grid.physical_max_x,
            grid.physical_min_y, grid.physical_max_y,
            encounter.step, encounter.scale1, zero_field,
            impl_->device_metrics);
      ok = cuda_ok(cudaGetLastError(),
                   "primary gather/push beam1 launch", error);
      if (ok && encounter.beam2_count)
        gather_and_push_kernel<<<block_count(encounter.beam2_count), kThreads,
                                 0, impl_->stream>>>(
            impl_->xy2 + 2 * encounter.beam2_offset,
            impl_->velocity2 + 2 * encounter.beam2_offset,
            impl_->energy2 + encounter.beam2_offset,
            impl_->upsilon2 + encounter.beam2_offset,
            encounter.beam2_count, impl_->phi1, 1, impl_->nx, impl_->ny,
            grid.delta_x_inverse, grid.delta_y_inverse,
            grid.offset_x, grid.offset_y,
            grid.physical_min_x, grid.physical_max_x,
            grid.physical_min_y, grid.physical_max_y,
            encounter.step, encounter.scale2, zero_field,
            impl_->device_metrics);
      ok = ok && cuda_ok(cudaGetLastError(),
                         "primary gather/push beam2 launch", error);
    }

  DevicePrimaryMetrics host_metrics = {};
  ok = ok && cuda_ok(cudaEventRecord(impl_->event_end, impl_->stream),
                     "cudaEventRecord(primary batch kernels end)", error) &&
       cuda_ok(cudaMemcpyAsync(impl_->pinned_metrics, impl_->device_metrics,
                               sizeof(DevicePrimaryMetrics),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(primary compact metrics D2H)", error) &&
       cuda_ok(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize(primary batch boundary)", error);
  if (!ok)
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }
  host_metrics = *impl_->pinned_metrics;
  float elapsed_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, impl_->event_start,
                                    impl_->event_end),
               "cudaEventElapsedTime(primary batch)", error))
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentPrimaryStats& totals = impl_->statistics;
  ++totals.batch_calls;
  ++totals.state_epoch;
  totals.waves += waves;
  totals.encounters += encounter_count;
  totals.particle_updates += particle_updates;
  totals.deposition_calls += encounter_count;
  totals.field_solve_calls += nonzero_field_solves;
  totals.gather_push_calls += encounter_count;
  totals.deposit_beam1_in += host_metrics.deposit_beam1_in;
  totals.deposit_beam1_out += host_metrics.deposit_beam1_out;
  totals.deposit_beam2_in += host_metrics.deposit_beam2_in;
  totals.deposit_beam2_out += host_metrics.deposit_beam2_out;
  totals.gather_beam1_in += host_metrics.gather_beam1_in;
  totals.gather_beam1_out += host_metrics.gather_beam1_out;
  totals.gather_beam2_in += host_metrics.gather_beam2_in;
  totals.gather_beam2_out += host_metrics.gather_beam2_out;
  totals.nonfinite_state_events += host_metrics.nonfinite_state_events;
  totals.d2h_bytes += sizeof(DevicePrimaryMetrics);
  ++totals.d2h_operations;
  ++totals.synchronizations;
  totals.kernel_ms += static_cast<double>(elapsed_ms);
  const double updated_fine_lumi =
      totals.fine_lumi_dot_sum + host_metrics.fine_lumi_dot_sum;
  if (host_metrics.nonfinite_state_events != 0 ||
      !std::isfinite(host_metrics.fine_lumi_dot_sum) ||
      !std::isfinite(updated_fine_lumi))
    {
      std::ostringstream cause;
      cause << "resident primary device detected "
            << host_metrics.nonfinite_state_events
            << " non-finite state event(s) or a non-finite fine-luminosity accumulator";
      impl_->poison(cause.str(), error);
      statistics = totals;
      return false;
    }
  totals.fine_lumi_dot_sum = updated_fine_lumi;
  totals.state_resident = true;
  statistics = totals;
  return true;
}

bool ResidentPrimaryPipeline::download(
    float* xy1,
    float* velocity_xy1,
    float* energy1,
    float* upsilon1,
    std::size_t output_capacity1,
    float* xy2,
    float* velocity_xy2,
    float* energy2,
    float* upsilon2,
    std::size_t output_capacity2,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  error.clear();
  if (impl_->reject_if_unavailable(error)) return false;
  if (!impl_->state_ready)
    {
      error = "resident primary download requires uploaded state";
      return false;
    }
  if (output_capacity1 < impl_->count1 || output_capacity2 < impl_->count2)
    {
      error = "resident primary download output capacity is too small";
      return false;
    }
  if ((impl_->count1 && (!xy1 || !velocity_xy1 || !energy1 || !upsilon1)) ||
      (impl_->count2 && (!xy2 || !velocity_xy2 || !energy2 || !upsilon2)))
    {
      error = "resident primary download received a null active output array";
      return false;
    }
  const std::size_t xy_bytes1 = 2 * impl_->count1 * sizeof(float);
  const std::size_t xy_bytes2 = 2 * impl_->count2 * sizeof(float);
  const std::size_t energy_bytes1 = impl_->count1 * sizeof(float);
  const std::size_t energy_bytes2 = impl_->count2 * sizeof(float);
  const void* pointers[8] = {xy1, velocity_xy1, energy1, upsilon1,
                             xy2, velocity_xy2, energy2, upsilon2};
  const std::size_t bytes[8] = {
      xy_bytes1, xy_bytes1, energy_bytes1, energy_bytes1,
      xy_bytes2, xy_bytes2, energy_bytes2, energy_bytes2};
  for (int first = 0; first < 8; ++first)
    for (int second = first + 1; second < 8; ++second)
      if (byte_ranges_overlap(pointers[first], bytes[first],
                              pointers[second], bytes[second]))
        {
          error = "resident primary download requires non-overlapping output arrays";
          return false;
        }
  std::size_t beam1_transfer_bytes = 0;
  std::size_t beam2_transfer_bytes = 0;
  std::size_t transfer_bytes_size = 0;
  std::uint64_t transfer_bytes = 0;
  if (!checked_sum(xy_bytes1, xy_bytes1, beam1_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, energy_bytes1,
                   beam1_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, energy_bytes1,
                   beam1_transfer_bytes) ||
      !checked_sum(xy_bytes2, xy_bytes2, beam2_transfer_bytes) ||
      !checked_sum(beam2_transfer_bytes, energy_bytes2,
                   beam2_transfer_bytes) ||
      !checked_sum(beam2_transfer_bytes, energy_bytes2,
                   beam2_transfer_bytes) ||
      !checked_sum(beam1_transfer_bytes, beam2_transfer_bytes,
                   transfer_bytes_size) ||
      !size_to_u64(transfer_bytes_size, transfer_bytes))
    {
      error = "resident primary download byte accounting overflows";
      return false;
    }
  const std::uint64_t transfer_operations =
      4 * static_cast<std::uint64_t>(impl_->count1 != 0) +
      4 * static_cast<std::uint64_t>(impl_->count2 != 0);
  const ResidentPrimaryStats& prior = impl_->statistics;
  if (!can_add_u64(prior.download_calls, 1) ||
      !can_add_u64(prior.d2h_bytes, transfer_bytes) ||
      !can_add_u64(prior.d2h_operations, transfer_operations) ||
      !can_add_u64(prior.synchronizations, 1))
    {
      error = "resident primary download telemetry counter would overflow";
      return false;
    }
  ScopedCudaDevice selected_device(
      impl_->device, "cudaSetDevice(resident primary download)", error);
  if (!selected_device.selected())
    {
      impl_->poison_without_drain(error, error);
      statistics = impl_->statistics;
      return false;
    }

  float* destination_xy1 = impl_->pinned_io ? impl_->pinned_xy1 : xy1;
  float* destination_velocity1 =
      impl_->pinned_io ? impl_->pinned_velocity1 : velocity_xy1;
  float* destination_energy1 =
      impl_->pinned_io ? impl_->pinned_energy1 : energy1;
  float* destination_upsilon1 =
      impl_->pinned_io ? impl_->pinned_upsilon1 : upsilon1;
  float* destination_xy2 = impl_->pinned_io ? impl_->pinned_xy2 : xy2;
  float* destination_velocity2 =
      impl_->pinned_io ? impl_->pinned_velocity2 : velocity_xy2;
  float* destination_energy2 =
      impl_->pinned_io ? impl_->pinned_energy2 : energy2;
  float* destination_upsilon2 =
      impl_->pinned_io ? impl_->pinned_upsilon2 : upsilon2;
  const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  bool ok = cuda_ok(cudaEventRecord(impl_->event_start, impl_->stream),
                    "cudaEventRecord(primary download start)", error);
  const auto copy_d2h = [&error, this](void* destination,
                                      const void* source,
                                      std::size_t count,
                                      const char* label) -> bool {
    return !count || cuda_ok(cudaMemcpyAsync(destination, source, count,
                                              cudaMemcpyDeviceToHost,
                                              impl_->stream), label, error);
  };
  ok = ok && copy_d2h(destination_xy1, impl_->xy1, xy_bytes1,
                      "cudaMemcpyAsync(primary xy1 D2H)");
  ok = ok && copy_d2h(destination_velocity1, impl_->velocity1, xy_bytes1,
                      "cudaMemcpyAsync(primary velocity1 D2H)");
  ok = ok && copy_d2h(destination_energy1, impl_->energy1, energy_bytes1,
                      "cudaMemcpyAsync(primary energy1 D2H)");
  ok = ok && copy_d2h(destination_upsilon1, impl_->upsilon1, energy_bytes1,
                      "cudaMemcpyAsync(primary upsilon1 D2H)");
  ok = ok && copy_d2h(destination_xy2, impl_->xy2, xy_bytes2,
                      "cudaMemcpyAsync(primary xy2 D2H)");
  ok = ok && copy_d2h(destination_velocity2, impl_->velocity2, xy_bytes2,
                      "cudaMemcpyAsync(primary velocity2 D2H)");
  ok = ok && copy_d2h(destination_energy2, impl_->energy2, energy_bytes2,
                      "cudaMemcpyAsync(primary energy2 D2H)");
  ok = ok && copy_d2h(destination_upsilon2, impl_->upsilon2, energy_bytes2,
                      "cudaMemcpyAsync(primary upsilon2 D2H)");
  ok = ok && cuda_ok(cudaEventRecord(impl_->event_end, impl_->stream),
                     "cudaEventRecord(primary download end)", error) &&
       cuda_ok(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize(primary download)", error);
  if (!ok)
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }
  if (impl_->pinned_io)
    {
      if (xy_bytes1)
        {
          std::memcpy(xy1, impl_->pinned_xy1, xy_bytes1);
          std::memcpy(velocity_xy1, impl_->pinned_velocity1, xy_bytes1);
          std::memcpy(energy1, impl_->pinned_energy1, energy_bytes1);
          std::memcpy(upsilon1, impl_->pinned_upsilon1, energy_bytes1);
        }
      if (xy_bytes2)
        {
          std::memcpy(xy2, impl_->pinned_xy2, xy_bytes2);
          std::memcpy(velocity_xy2, impl_->pinned_velocity2, xy_bytes2);
          std::memcpy(energy2, impl_->pinned_energy2, energy_bytes2);
          std::memcpy(upsilon2, impl_->pinned_upsilon2, energy_bytes2);
        }
    }
  float elapsed_ms = 0.0f;
  if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, impl_->event_start,
                                    impl_->event_end),
               "cudaEventElapsedTime(primary download)", error))
    {
      impl_->poison(error, error);
      statistics = impl_->statistics;
      return false;
    }

  ResidentPrimaryStats& totals = impl_->statistics;
  ++totals.download_calls;
  totals.d2h_bytes += transfer_bytes;
  totals.d2h_operations += transfer_operations;
  ++totals.synchronizations;
  totals.download_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  totals.state_resident = true;
  statistics = totals;
  return true;
}

}  // namespace gpu
}  // namespace feebl
