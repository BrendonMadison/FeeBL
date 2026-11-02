#include "gpuAdaptiveWaveBatch.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace feebl {
namespace adaptive_grid {
namespace gpu5 {

namespace {

const int kThreads = 256;

struct Hash64 {
  std::uint64_t value;

  Hash64() : value(UINT64_C(14695981039346656037)) {}

  void bytes(const void* data, std::size_t size) noexcept {
    const unsigned char* input = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
      value ^= static_cast<std::uint64_t>(input[index]);
      value *= UINT64_C(1099511628211);
    }
  }

  void byte(std::uint8_t input) noexcept {
    value ^= static_cast<std::uint64_t>(input);
    value *= UINT64_C(1099511628211);
  }

  void boolean(bool input) noexcept { byte(input ? 1U : 0U); }

  void u32(std::uint32_t input) noexcept {
    for (unsigned int shift = 0; shift < 32U; shift += 8U)
      byte(static_cast<std::uint8_t>((input >> shift) & UINT32_C(0xff)));
  }

  void i32(std::int32_t input) noexcept {
    u32(static_cast<std::uint32_t>(input));
  }

  void u64(std::uint64_t input) noexcept {
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
      byte(static_cast<std::uint8_t>((input >> shift) & UINT64_C(0xff)));
  }

  void i64(std::int64_t input) noexcept {
    u64(static_cast<std::uint64_t>(input));
  }

  void size(std::size_t input) noexcept {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "mode-5 hashes require size_t no wider than uint64_t");
    u64(static_cast<std::uint64_t>(input));
  }
};

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4U,
              "mode-5 provenance requires IEEE-754 binary32");
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8U,
              "mode-5 provenance requires IEEE-754 binary64");

std::uint32_t float_bits(float value) noexcept {
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint64_t double_bits(double value) noexcept {
  std::uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

bool checked_product(std::size_t first, std::size_t second,
                     std::size_t& result) noexcept {
  if (first != 0 &&
      second > std::numeric_limits<std::size_t>::max() / first)
    return false;
  result = first * second;
  return true;
}

bool checked_add(std::size_t first, std::size_t second,
                 std::size_t& result) noexcept {
  if (second > std::numeric_limits<std::size_t>::max() - first) return false;
  result = first + second;
  return true;
}

int blocks_for(std::size_t work) noexcept {
  if (work == 0) return 1;
  return static_cast<int>(std::min<std::size_t>(
      65535, (work + static_cast<std::size_t>(kThreads) - 1) /
                 static_cast<std::size_t>(kThreads)));
}

bool cuda_ok(cudaError_t status, const char* operation,
             std::string& error) {
  if (status == cudaSuccess) return true;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status)
          << " (CUDA error " << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

bool cufft_ok(cufftResult status, const char* operation,
              std::string& error) {
  if (status == CUFFT_SUCCESS) return true;
  std::ostringstream message;
  message << operation << " failed (cuFFT error "
          << static_cast<int>(status) << ')';
  error = message.str();
  return false;
}

double f_potential_2(double x0, double y0, double dx, double dy) {
  // Keep FIELD::f_potential_2's statement and operation order literal.  This
  // translation unit is compiled with --fmad=false/-fno-fast-math.
  double x, y, sum;
  x = x0 + dx;
  y = y0 + dy;
  sum = x * y * (std::log(x * x + y * y) - 3.0) +
        x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 + dx;
  y = y0 - dy;
  sum -= x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 - dx;
  y = y0 + dy;
  sum -= x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 - dx;
  y = y0 - dy;
  sum += x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  return sum;
}

struct DeviceDepositDescriptor {
  long long base_x;
  long long base_y;
  float weight_x;
  float weight_y;
  unsigned int accepted;
};

struct DeviceGatherDescriptor {
  long long y_base_x;
  long long y_base_y;
  long long x_base_x;
  long long x_base_y;
  float y_weight;
  float x_weight;
  double fractional_x;
  double fractional_y;
  unsigned int accepted;
};

struct DeviceSlotMetadata {
  long long origin_x;
  long long origin_y;
  unsigned long long beam1_count;
  unsigned long long beam2_count;
};

struct DeviceEncounterMetrics {
  unsigned long long beam1_in;
  unsigned long long beam1_out;
  unsigned long long beam2_in;
  unsigned long long beam2_out;
  unsigned long long gather_beam1_in;
  unsigned long long gather_beam1_out;
  unsigned long long gather_beam2_in;
  unsigned long long gather_beam2_out;
  unsigned long long containment_failures;
  double fine_lumi_dot;
};

__device__ void count_deposit(DeviceEncounterMetrics* metric, int beam,
                              bool accepted) {
  if (beam == 0) {
    if (accepted)
      atomicAdd(&metric->beam1_in, 1ULL);
    else
      atomicAdd(&metric->beam1_out, 1ULL);
  } else {
    if (accepted)
      atomicAdd(&metric->beam2_in, 1ULL);
    else
      atomicAdd(&metric->beam2_out, 1ULL);
  }
}

__device__ void count_gather(DeviceEncounterMetrics* metric, int beam,
                             bool accepted) {
  if (beam == 0) {
    if (accepted)
      atomicAdd(&metric->gather_beam1_in, 1ULL);
    else
      atomicAdd(&metric->gather_beam1_out, 1ULL);
  } else {
    if (accepted)
      atomicAdd(&metric->gather_beam2_in, 1ULL);
    else
      atomicAdd(&metric->gather_beam2_out, 1ULL);
  }
}

__global__ void deposit_descriptor_kernel(
    const DeviceDepositDescriptor* descriptors,
    std::size_t maximum_descriptors, const DeviceSlotMetadata* slots,
    int beam, int local_n_x, int local_n_y, std::size_t local_cells,
    double macro_charge, double* rho, DeviceEncounterMetrics* metrics) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.y);
  const std::size_t count = static_cast<std::size_t>(
      beam == 0 ? slots[slot].beam1_count : slots[slot].beam2_count);
  const DeviceDepositDescriptor* slot_descriptors =
      descriptors + slot * maximum_descriptors;
  double* slot_rho = rho + slot * local_cells;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < count; particle += stride) {
    const DeviceDepositDescriptor descriptor = slot_descriptors[particle];
    if (!descriptor.accepted) {
      count_deposit(metrics + slot, beam, false);
      continue;
    }
    const long long local_x = descriptor.base_x - slots[slot].origin_x;
    const long long local_y = descriptor.base_y - slots[slot].origin_y;
    if (local_x < 0 || local_x >= static_cast<long long>(local_n_x - 1) ||
        local_y < 0 || local_y >= static_cast<long long>(local_n_y - 1)) {
      atomicAdd(&metrics[slot].containment_failures, 1ULL);
      continue;
    }
    count_deposit(metrics + slot, beam, true);
    const double weight_x = static_cast<double>(descriptor.weight_x);
    const double weight_y = static_cast<double>(descriptor.weight_y);
    const double complement_x = 1.0 - weight_x;
    const double complement_y = 1.0 - weight_y;
    const std::size_t base = static_cast<std::size_t>(local_x) *
                                 static_cast<std::size_t>(local_n_y) +
                             static_cast<std::size_t>(local_y);
    // Global FP64 atomics are the only admitted deposition implementation.
    atomicAdd(slot_rho + base,
              complement_x * complement_y * macro_charge);
    atomicAdd(slot_rho + base + static_cast<std::size_t>(local_n_y),
              weight_x * complement_y * macro_charge);
    atomicAdd(slot_rho + base + 1,
              complement_x * weight_y * macro_charge);
    atomicAdd(slot_rho + base + static_cast<std::size_t>(local_n_y) + 1,
              static_cast<double>(descriptor.weight_x *
                                  descriptor.weight_y) *
                  macro_charge);
  }
}

__global__ void fine_dot_kernel(const double* rho1, const double* rho2,
                                std::size_t local_cells,
                                DeviceEncounterMetrics* metrics) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x);
  const std::size_t base = slot * local_cells;
  __shared__ double partial[kThreads];
  double sum = 0.0;
  for (std::size_t cell = threadIdx.x; cell < local_cells;
       cell += blockDim.x)
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

__global__ void pack_spectrum_kernel(
    const double* rho1, const double* rho2, int local_n_y,
    int padded_n_y, std::size_t local_cells, std::size_t padded_cells,
    std::size_t active_slots, cufftDoubleComplex* spectrum) {
  const std::size_t total = active_slots * local_cells;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const std::size_t slot = work / local_cells;
    const std::size_t cell = work - slot * local_cells;
    const std::size_t local_x = cell / static_cast<std::size_t>(local_n_y);
    const std::size_t local_y =
        cell - local_x * static_cast<std::size_t>(local_n_y);
    cufftDoubleComplex value;
    value.x = rho1[work];
    value.y = rho2[work];
    spectrum[slot * padded_cells +
             local_x * static_cast<std::size_t>(padded_n_y) + local_y] =
        value;
  }
}

__global__ void multiply_spectrum_kernel(
    cufftDoubleComplex* spectrum, const cufftDoubleComplex* green,
    std::size_t padded_cells, std::size_t batch_capacity) {
  const std::size_t total = padded_cells * batch_capacity;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const cufftDoubleComplex input = spectrum[work];
    const cufftDoubleComplex kernel = green[work % padded_cells];
    cufftDoubleComplex product;
    product.x = input.x * kernel.x - input.y * kernel.y;
    product.y = input.y * kernel.x + input.x * kernel.y;
    spectrum[work] = product;
  }
}

__global__ void scale_spectrum_kernel(cufftDoubleComplex* spectrum,
                                      std::size_t cells,
                                      double normalization_denominator) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t cell =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cells; cell += stride) {
    // FIELD::dist_init case 2 stores temp[k]/(double)(4*nx*ny).
    spectrum[cell].x /= normalization_denominator;
    spectrum[cell].y /= normalization_denominator;
  }
}

__global__ void unpack_spectrum_kernel(
    const cufftDoubleComplex* spectrum, int local_n_y, int padded_n_y,
    std::size_t local_cells, std::size_t padded_cells,
    std::size_t active_slots, double* phi1, double* phi2) {
  const std::size_t total = active_slots * local_cells;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < total; work += stride) {
    const std::size_t slot = work / local_cells;
    const std::size_t cell = work - slot * local_cells;
    const std::size_t local_x = cell / static_cast<std::size_t>(local_n_y);
    const std::size_t local_y =
        cell - local_x * static_cast<std::size_t>(local_n_y);
    const cufftDoubleComplex value =
        spectrum[slot * padded_cells +
                 local_x * static_cast<std::size_t>(padded_n_y) + local_y];
    phi1[work] = value.x;
    phi2[work] = value.y;
  }
}

__device__ bool gather_contained(const DeviceGatherDescriptor& descriptor,
                                 long long origin_x, long long origin_y,
                                 int local_n_x, int local_n_y) {
  const long long y_x = descriptor.y_base_x - origin_x;
  const long long y_y = descriptor.y_base_y - origin_y;
  const long long x_x = descriptor.x_base_x - origin_x;
  const long long x_y = descriptor.x_base_y - origin_y;
  return y_x - 1 >= 0 && y_x + 1 < local_n_x && y_y >= 0 &&
         y_y + 1 < local_n_y && x_x >= 0 && x_x + 1 < local_n_x &&
         x_y - 1 >= 0 && x_y + 1 < local_n_y;
}

__device__ std::size_t local_index(long long canonical_x,
                                   long long canonical_y,
                                   long long origin_x, long long origin_y,
                                   int local_n_y) {
  return static_cast<std::size_t>(canonical_x - origin_x) *
             static_cast<std::size_t>(local_n_y) +
         static_cast<std::size_t>(canonical_y - origin_y);
}

__device__ void gather_field(const DeviceGatherDescriptor& descriptor,
                             const double* phi, long long origin_x,
                             long long origin_y, int local_n_y,
                             float inverse_delta_x, float inverse_delta_y,
                             float& field_x, float& field_y) {
  float h_p = 1.0 - descriptor.y_weight;
  double phi1_y = descriptor.y_weight *
                      phi[local_index(descriptor.y_base_x + 1,
                                      descriptor.y_base_y + 1,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.y_base_x + 1,
                                        descriptor.y_base_y,
                                        origin_x, origin_y, local_n_y)];
  double phi2_y = descriptor.y_weight *
                      phi[local_index(descriptor.y_base_x,
                                      descriptor.y_base_y + 1,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.y_base_x,
                                        descriptor.y_base_y,
                                        origin_x, origin_y, local_n_y)];
  double phi3_y = descriptor.y_weight *
                      phi[local_index(descriptor.y_base_x - 1,
                                      descriptor.y_base_y + 1,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.y_base_x - 1,
                                        descriptor.y_base_y,
                                        origin_x, origin_y, local_n_y)];

  h_p = 1.0 - descriptor.x_weight;
  double phi1_x = descriptor.x_weight *
                      phi[local_index(descriptor.x_base_x + 1,
                                      descriptor.x_base_y + 1,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.x_base_x,
                                        descriptor.x_base_y + 1,
                                        origin_x, origin_y, local_n_y)];
  double phi2_x = descriptor.x_weight *
                      phi[local_index(descriptor.x_base_x + 1,
                                      descriptor.x_base_y,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.x_base_x,
                                        descriptor.x_base_y,
                                        origin_x, origin_y, local_n_y)];
  double phi3_x = descriptor.x_weight *
                      phi[local_index(descriptor.x_base_x + 1,
                                      descriptor.x_base_y - 1,
                                      origin_x, origin_y, local_n_y)] +
                  h_p * phi[local_index(descriptor.x_base_x,
                                        descriptor.x_base_y - 1,
                                        origin_x, origin_y, local_n_y)];

  double raw_x =
      (descriptor.fractional_x * (phi1_y - phi2_y) +
       (1.0 - descriptor.fractional_x) * (phi2_y - phi3_y)) *
      static_cast<double>(inverse_delta_x);
  double raw_y =
      (descriptor.fractional_y * (phi1_x - phi2_x) +
       (1.0 - descriptor.fractional_y) * (phi2_x - phi3_x)) *
      static_cast<double>(inverse_delta_y);
  field_x = static_cast<float>(raw_x);
  field_y = static_cast<float>(raw_y);
  // TRIVECTOR stores float: GENERAL_GRID narrows on component assignment,
  // then its `field *= 0.5f` performs the historical second float operation.
  field_x *= 0.5f;
  field_y *= 0.5f;
}

__global__ void gather_descriptor_kernel(
    const DeviceGatherDescriptor* descriptors,
    std::size_t maximum_descriptors, const DeviceSlotMetadata* slots,
    const double* opposing_phi, int beam, int local_n_x, int local_n_y,
    std::size_t local_cells, float inverse_delta_x,
    float inverse_delta_y, bool zero_field, float* fields,
    DeviceEncounterMetrics* metrics) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.y);
  const std::size_t count = static_cast<std::size_t>(
      beam == 0 ? slots[slot].beam1_count : slots[slot].beam2_count);
  const DeviceGatherDescriptor* slot_descriptors =
      descriptors + slot * maximum_descriptors;
  const double* slot_phi = opposing_phi + slot * local_cells;
  float* slot_fields = fields + 2 * slot * maximum_descriptors;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < count; particle += stride) {
    const DeviceGatherDescriptor descriptor = slot_descriptors[particle];
    if (!descriptor.accepted) {
      slot_fields[2 * particle] = 0.0f;
      slot_fields[2 * particle + 1] = 0.0f;
      count_gather(metrics + slot, beam, false);
      continue;
    }
    if (!gather_contained(descriptor, slots[slot].origin_x,
                          slots[slot].origin_y, local_n_x, local_n_y)) {
      slot_fields[2 * particle] = 0.0f;
      slot_fields[2 * particle + 1] = 0.0f;
      atomicAdd(&metrics[slot].containment_failures, 1ULL);
      continue;
    }
    count_gather(metrics + slot, beam, true);
    if (zero_field) {
      slot_fields[2 * particle] = 0.0f;
      slot_fields[2 * particle + 1] = 0.0f;
    } else {
      gather_field(descriptor, slot_phi, slots[slot].origin_x,
                   slots[slot].origin_y, local_n_y, inverse_delta_x,
                   inverse_delta_y, slot_fields[2 * particle],
                   slot_fields[2 * particle + 1]);
    }
  }
}

DeviceDepositDescriptor pack_descriptor(
    const CanonicalDepositDescriptor& descriptor) noexcept {
  DeviceDepositDescriptor result;
  result.base_x = static_cast<long long>(descriptor.base_x());
  result.base_y = static_cast<long long>(descriptor.base_y());
  result.weight_x = descriptor.weight_x();
  result.weight_y = descriptor.weight_y();
  result.accepted = descriptor.accepted() ? 1U : 0U;
  return result;
}

DeviceGatherDescriptor pack_descriptor(
    const CanonicalGatherDescriptor& descriptor) noexcept {
  DeviceGatherDescriptor result;
  result.y_base_x = static_cast<long long>(descriptor.y_base_x());
  result.y_base_y = static_cast<long long>(descriptor.y_base_y());
  result.x_base_x = static_cast<long long>(descriptor.x_base_x());
  result.x_base_y = static_cast<long long>(descriptor.x_base_y());
  result.y_weight = descriptor.y_weight();
  result.x_weight = descriptor.x_weight();
  result.fractional_x = descriptor.fractional_x();
  result.fractional_y = descriptor.fractional_y();
  result.accepted = descriptor.accepted() ? 1U : 0U;
  return result;
}

bool host_gather_contained(const CanonicalGatherDescriptor& descriptor,
                           const UniformWindowPlan& plan) noexcept {
  if (!descriptor.accepted()) return true;
  for (const CanonicalNode& node : descriptor.nodes())
    if (!plan.contains_canonical_node(node)) return false;
  return true;
}

bool host_deposit_contained(const CanonicalDepositDescriptor& descriptor,
                            const UniformWindowPlan& plan) noexcept {
  if (!descriptor.accepted()) return true;
  for (const CanonicalNode& node : descriptor.nodes())
    if (!plan.contains_canonical_node(node)) return false;
  return true;
}

template <typename Descriptor>
std::size_t accepted_count(const std::vector<Descriptor>& descriptors) {
  std::size_t count = 0;
  for (const Descriptor& descriptor : descriptors)
    if (descriptor.accepted()) ++count;
  return count;
}

}  // namespace

bool DeviceIdentity::operator==(const DeviceIdentity& other) const noexcept {
  return ordinal == other.ordinal && uuid == other.uuid &&
         compute_major == other.compute_major &&
         compute_minor == other.compute_minor &&
         cuda_runtime_version == other.cuda_runtime_version &&
         cuda_driver_version == other.cuda_driver_version &&
         cufft_version == other.cufft_version;
}

bool DeviceIdentity::operator<(const DeviceIdentity& other) const noexcept {
  return std::tie(ordinal, uuid, compute_major, compute_minor,
                  cuda_runtime_version, cuda_driver_version, cufft_version) <
         std::tie(other.ordinal, other.uuid, other.compute_major,
                  other.compute_minor, other.cuda_runtime_version,
                  other.cuda_driver_version, other.cufft_version);
}

bool FftPlanCacheKey::operator==(const FftPlanCacheKey& other) const noexcept {
  return device == other.device && precision == other.precision &&
         transform_type == other.transform_type &&
         logical_n_x == other.logical_n_x &&
         logical_n_y == other.logical_n_y &&
         padded_n_x == other.padded_n_x &&
         padded_n_y == other.padded_n_y &&
         batch_capacity == other.batch_capacity && rank == other.rank &&
         dimensions == other.dimensions && input_embed == other.input_embed &&
         output_embed == other.output_embed &&
         input_stride == other.input_stride &&
         output_stride == other.output_stride &&
         input_distance == other.input_distance &&
         output_distance == other.output_distance &&
         forward_direction == other.forward_direction &&
         inverse_direction == other.inverse_direction &&
         in_place == other.in_place &&
         explicit_workspace == other.explicit_workspace &&
         one_nonblocking_stream == other.one_nonblocking_stream &&
         build_abi_hash == other.build_abi_hash;
}

bool FftPlanCacheKey::operator<(const FftPlanCacheKey& other) const noexcept {
  return std::tie(device, precision, transform_type, logical_n_x, logical_n_y,
                  padded_n_x, padded_n_y, batch_capacity, rank, dimensions,
                  input_embed, output_embed, input_stride, output_stride,
                  input_distance, output_distance, forward_direction,
                  inverse_direction, in_place, explicit_workspace,
                  one_nonblocking_stream, build_abi_hash) <
         std::tie(other.device, other.precision, other.transform_type,
                  other.logical_n_x, other.logical_n_y, other.padded_n_x,
                  other.padded_n_y, other.batch_capacity, other.rank,
                  other.dimensions, other.input_embed, other.output_embed,
                  other.input_stride, other.output_stride,
                  other.input_distance, other.output_distance,
                  other.forward_direction, other.inverse_direction,
                  other.in_place, other.explicit_workspace,
                  other.one_nonblocking_stream, other.build_abi_hash);
}

std::uint64_t FftPlanCacheKey::stable_hash() const noexcept {
  Hash64 hash;
  hash.i32(device.ordinal);
  hash.bytes(device.uuid.data(), device.uuid.size());
  hash.i32(device.compute_major);
  hash.i32(device.compute_minor);
  hash.i32(device.cuda_runtime_version);
  hash.i32(device.cuda_driver_version);
  hash.i32(device.cufft_version);
  hash.u32(static_cast<std::uint32_t>(precision));
  hash.u32(static_cast<std::uint32_t>(transform_type));
  hash.size(logical_n_x);
  hash.size(logical_n_y);
  hash.size(padded_n_x);
  hash.size(padded_n_y);
  hash.size(batch_capacity);
  hash.u32(rank);
  for (std::size_t value : dimensions) hash.size(value);
  for (std::size_t value : input_embed) hash.size(value);
  for (std::size_t value : output_embed) hash.size(value);
  hash.size(input_stride);
  hash.size(output_stride);
  hash.size(input_distance);
  hash.size(output_distance);
  hash.i32(forward_direction);
  hash.i32(inverse_direction);
  hash.boolean(in_place);
  hash.boolean(explicit_workspace);
  hash.boolean(one_nonblocking_stream);
  hash.u64(build_abi_hash);
  return hash.value;
}

bool GreenSpectrumCacheKey::operator==(
    const GreenSpectrumCacheKey& other) const noexcept {
  return device == other.device && solver == other.solver &&
         normalization == other.normalization &&
         stored_charge == other.stored_charge && precision == other.precision &&
         logical_n_x == other.logical_n_x &&
         logical_n_y == other.logical_n_y &&
         padded_n_x == other.padded_n_x &&
         padded_n_y == other.padded_n_y &&
         delta_x_bits == other.delta_x_bits &&
         delta_y_bits == other.delta_y_bits &&
         delta_z_bits == other.delta_z_bits &&
         charge_sign_bits == other.charge_sign_bits &&
         field_kernel_factor_bits == other.field_kernel_factor_bits &&
         spectrum_generator_id == other.spectrum_generator_id &&
         spectrum_generator_version == other.spectrum_generator_version &&
         explicit_workspace == other.explicit_workspace &&
         build_abi_hash == other.build_abi_hash;
}

bool GreenSpectrumCacheKey::operator<(
    const GreenSpectrumCacheKey& other) const noexcept {
  return std::tie(device, solver, normalization, stored_charge, precision,
                  logical_n_x, logical_n_y, padded_n_x, padded_n_y,
                  delta_x_bits, delta_y_bits, delta_z_bits, charge_sign_bits,
                  field_kernel_factor_bits, spectrum_generator_id,
                  spectrum_generator_version, explicit_workspace,
                  build_abi_hash) <
         std::tie(other.device, other.solver, other.normalization,
                  other.stored_charge, other.precision, other.logical_n_x,
                  other.logical_n_y, other.padded_n_x, other.padded_n_y,
                  other.delta_x_bits, other.delta_y_bits, other.delta_z_bits,
                  other.charge_sign_bits, other.field_kernel_factor_bits,
                  other.spectrum_generator_id,
                  other.spectrum_generator_version,
                  other.explicit_workspace, other.build_abi_hash);
}

std::uint64_t GreenSpectrumCacheKey::stable_hash() const noexcept {
  Hash64 hash;
  hash.i32(device.ordinal);
  hash.bytes(device.uuid.data(), device.uuid.size());
  hash.i32(device.compute_major);
  hash.i32(device.compute_minor);
  hash.i32(device.cuda_runtime_version);
  hash.i32(device.cuda_driver_version);
  hash.i32(device.cufft_version);
  hash.u32(static_cast<std::uint32_t>(solver));
  hash.u32(static_cast<std::uint32_t>(normalization));
  hash.u32(static_cast<std::uint32_t>(stored_charge));
  hash.u32(static_cast<std::uint32_t>(precision));
  hash.size(logical_n_x);
  hash.size(logical_n_y);
  hash.size(padded_n_x);
  hash.size(padded_n_y);
  hash.u32(delta_x_bits);
  hash.u32(delta_y_bits);
  hash.u32(delta_z_bits);
  hash.u32(charge_sign_bits);
  hash.u64(field_kernel_factor_bits);
  hash.u32(spectrum_generator_id);
  hash.i32(spectrum_generator_version);
  hash.boolean(explicit_workspace);
  hash.u64(build_abi_hash);
  return hash.value;
}

void AdaptiveEncounterResult::reset() noexcept {
  valid = false;
  binding = AdaptiveResultBinding();
  fields1.clear();
  fields2.clear();
  beam1_in = beam1_out = beam2_in = beam2_out = 0;
  gather_beam1_in = gather_beam1_out = 0;
  gather_beam2_in = gather_beam2_out = 0;
  fine_lumi_dot = 0.0;
}

void AdaptiveEncounterResult::swap(AdaptiveEncounterResult& other) noexcept {
  using std::swap;
  swap(valid, other.valid);
  swap(binding, other.binding);
  fields1.swap(other.fields1);
  fields2.swap(other.fields2);
  swap(beam1_in, other.beam1_in);
  swap(beam1_out, other.beam1_out);
  swap(beam2_in, other.beam2_in);
  swap(beam2_out, other.beam2_out);
  swap(gather_beam1_in, other.gather_beam1_in);
  swap(gather_beam1_out, other.gather_beam1_out);
  swap(gather_beam2_in, other.gather_beam2_in);
  swap(gather_beam2_out, other.gather_beam2_out);
  swap(fine_lumi_dot, other.fine_lumi_dot);
}

const char* deposition_variant_name(DepositionVariantId variant) noexcept {
  switch (variant) {
    case DepositionVariantId::GlobalAtomic:
      return "global-atomic";
    case DepositionVariantId::SharedTile:
      return "shared-tile";
    case DepositionVariantId::SegmentedReduction:
      return "segmented-reduction";
  }
  return "unknown";
}

struct AdaptiveWaveBatchExecutor::Impl {
  struct PlanRecord {
    FftPlanCacheKey key;
    cufftHandle handle;
    void* workspace;
    std::size_t workspace_bytes;

    PlanRecord() : handle(0), workspace(nullptr), workspace_bytes(0) {}
  };

  struct GreenRecord {
    GreenSpectrumCacheKey key;
    cufftDoubleComplex* spectrum;
    std::size_t spectrum_bytes;

    GreenRecord() : spectrum(nullptr), spectrum_bytes(0) {}
  };

  struct ClassContext {
    WindowClassShape shape;
    std::size_t local_cells;
    std::size_t padded_cells;
    cufftDoubleComplex* spectrum;
    double* rho1;
    double* rho2;
    double* phi1;
    double* phi2;
    std::size_t spectrum_bytes;
    std::size_t one_real_batch_bytes;
    PlanRecord* plan;
    GreenRecord* green;

    ClassContext()
        : local_cells(0), padded_cells(0), spectrum(nullptr), rho1(nullptr),
          rho2(nullptr), phi1(nullptr), phi2(nullptr), spectrum_bytes(0),
          one_real_batch_bytes(0), plan(nullptr), green(nullptr) {}
  };

  AdaptiveGpuOptions options;
  std::unique_ptr<CanonicalLatticeSnapshot> lattice;
  DeviceIdentity identity;
  AdaptiveGpuTelemetry stats;
  bool initialization_attempted;
  bool initialized;
  bool prebuild_attempted;
  bool poisoned;
  std::size_t free_bytes_at_initialize;
  std::size_t shared_device_bytes;
  std::string device_text;
  cudaStream_t stream;
  cudaEvent_t events[4];

  DeviceDepositDescriptor* device_deposit1;
  DeviceDepositDescriptor* device_deposit2;
  DeviceGatherDescriptor* device_gather1;
  DeviceGatherDescriptor* device_gather2;
  DeviceSlotMetadata* device_slots;
  DeviceEncounterMetrics* device_metrics;
  float* device_fields1;
  float* device_fields2;

  DeviceDepositDescriptor* host_deposit1;
  DeviceDepositDescriptor* host_deposit2;
  DeviceGatherDescriptor* host_gather1;
  DeviceGatherDescriptor* host_gather2;
  DeviceSlotMetadata* host_slots;
  DeviceEncounterMetrics* host_metrics;
  float* host_fields1;
  float* host_fields2;

  std::size_t deposit1_bytes;
  std::size_t deposit2_bytes;
  std::size_t gather1_bytes;
  std::size_t gather2_bytes;
  std::size_t slot_bytes;
  std::size_t metric_bytes;
  std::size_t field1_bytes;
  std::size_t field2_bytes;

  std::map<FftPlanCacheKey, std::unique_ptr<PlanRecord> > plan_cache;
  std::map<GreenSpectrumCacheKey, std::unique_ptr<GreenRecord> > green_cache;
  std::map<WindowClassShape, std::unique_ptr<ClassContext> > class_contexts;

  Impl()
      : initialization_attempted(false), initialized(false),
        prebuild_attempted(false), poisoned(false),
        free_bytes_at_initialize(0), shared_device_bytes(0), stream(nullptr),
        device_deposit2(nullptr), device_gather1(nullptr),
        device_gather2(nullptr), device_slots(nullptr),
        device_metrics(nullptr), device_fields1(nullptr),
        device_fields2(nullptr), host_deposit1(nullptr),
        host_deposit2(nullptr), host_gather1(nullptr), host_gather2(nullptr),
        host_slots(nullptr), host_metrics(nullptr), host_fields1(nullptr),
        host_fields2(nullptr), deposit1_bytes(0), deposit2_bytes(0),
        gather1_bytes(0), gather2_bytes(0), slot_bytes(0), metric_bytes(0),
        field1_bytes(0), field2_bytes(0) {
    for (int index = 0; index < 4; ++index) events[index] = nullptr;
  }

  ~Impl() {
    if (identity.ordinal >= 0) cudaSetDevice(identity.ordinal);
    if (stream) cudaStreamSynchronize(stream);
    for (auto& entry : plan_cache) {
      if (entry.second->handle) cufftDestroy(entry.second->handle);
    }
    for (int index = 3; index >= 0; --index)
      if (events[index]) cudaEventDestroy(events[index]);
    for (auto& entry : class_contexts) {
      ClassContext& context = *entry.second;
      if (context.phi2) cudaFree(context.phi2);
      if (context.phi1) cudaFree(context.phi1);
      if (context.rho2) cudaFree(context.rho2);
      if (context.rho1) cudaFree(context.rho1);
      if (context.spectrum) cudaFree(context.spectrum);
    }
    for (auto& entry : green_cache)
      if (entry.second->spectrum) cudaFree(entry.second->spectrum);
    for (auto& entry : plan_cache)
      if (entry.second->workspace) cudaFree(entry.second->workspace);
    if (device_fields2) cudaFree(device_fields2);
    if (device_fields1) cudaFree(device_fields1);
    if (device_metrics) cudaFree(device_metrics);
    if (device_slots) cudaFree(device_slots);
    if (device_gather2) cudaFree(device_gather2);
    if (device_gather1) cudaFree(device_gather1);
    if (device_deposit2) cudaFree(device_deposit2);
    if (device_deposit1) cudaFree(device_deposit1);
    if (host_fields2) cudaFreeHost(host_fields2);
    if (host_fields1) cudaFreeHost(host_fields1);
    if (host_metrics) cudaFreeHost(host_metrics);
    if (host_slots) cudaFreeHost(host_slots);
    if (host_gather2) cudaFreeHost(host_gather2);
    if (host_gather1) cudaFreeHost(host_gather1);
    if (host_deposit2) cudaFreeHost(host_deposit2);
    if (host_deposit1) cudaFreeHost(host_deposit1);
    if (stream) cudaStreamDestroy(stream);
  }

  void mark_poisoned() noexcept {
    poisoned = true;
    stats.context_poisoned = true;
  }

  bool refresh_high_water(std::string& error) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
                 "cudaMemGetInfo(gpu_mode=5 high-water)", error))
      return false;
    const std::size_t used = free_bytes_at_initialize > free_bytes
                                 ? free_bytes_at_initialize - free_bytes
                                 : 0;
    stats.measured_device_high_water_bytes =
        std::max(stats.measured_device_high_water_bytes, used);
    return true;
  }

  bool allocate_device(void** pointer, std::size_t bytes, const char* label,
                       std::string& error) {
    *pointer = nullptr;
    if (bytes == 0) return true;
    if (stats.tracked_device_bytes > stats.memory_budget_bytes ||
        bytes > stats.memory_budget_bytes - stats.tracked_device_bytes) {
      std::ostringstream message;
      message << label << " would exceed the gpu_mode=5 device budget: "
              << stats.tracked_device_bytes << " resident + " << bytes
              << " requested > " << stats.memory_budget_bytes;
      error = message.str();
      return false;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), label, error))
      return false;
    if (bytes > free_bytes) {
      std::ostringstream message;
      message << label << " needs " << bytes << " bytes but only "
              << free_bytes << " CUDA bytes are free";
      error = message.str();
      return false;
    }
    if (!cuda_ok(cudaMalloc(pointer, bytes), label, error)) return false;
    stats.tracked_device_bytes += bytes;
    stats.peak_tracked_device_bytes =
        std::max(stats.peak_tracked_device_bytes, stats.tracked_device_bytes);
    if (!refresh_high_water(error)) return false;
    return true;
  }

  bool release_device(void* pointer, std::size_t bytes, const char* label,
                      std::string& error) {
    if (!pointer) return true;
    if (!cuda_ok(cudaFree(pointer), label, error)) {
      mark_poisoned();
      return false;
    }
    if (stats.tracked_device_bytes < bytes) {
      error = std::string(label) +
              " succeeded but device-byte accounting underflowed";
      mark_poisoned();
      return false;
    }
    stats.tracked_device_bytes -= bytes;
    return true;
  }

  void release_during_failure(void* pointer, std::size_t bytes,
                              const char* label, std::string& error) {
    std::string cleanup_error;
    if (!release_device(pointer, bytes, label, cleanup_error)) {
      if (!error.empty()) error += "; cleanup also failed: ";
      error += cleanup_error;
    }
  }

  bool allocate_pinned(void** pointer, std::size_t bytes, const char* label,
                       std::string& error) {
    *pointer = nullptr;
    if (bytes == 0) return true;
    if (!cuda_ok(cudaHostAlloc(pointer, bytes, cudaHostAllocDefault), label,
                 error))
      return false;
    stats.pinned_host_bytes += bytes;
    return true;
  }

  bool shape_is_valid(const WindowClassShape& shape,
                      std::string& error) const {
    if (!lattice || shape.local_n_x < 4 || shape.local_n_y < 4 ||
        shape.local_n_x > lattice->n_x() ||
        shape.local_n_y > lattice->n_y() ||
        shape.local_n_x >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 2) ||
        shape.local_n_y >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
      error = "gpu_mode=5 window class must be a canonical 4x4-or-larger "
              "shape with doubled dimensions representable by cuFFT";
      return false;
    }
    std::size_t padded_cells = 0;
    if (!checked_product(2 * shape.local_n_x, 2 * shape.local_n_y,
                         padded_cells) ||
        padded_cells >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      error = "gpu_mode=5 padded class exceeds the cuFFT PlanMany ABI";
      return false;
    }
    return true;
  }

  bool make_fft_key(const WindowClassShape& shape, FftPlanCacheKey& key,
                    std::string& error) const {
    if (!initialized || !shape_is_valid(shape, error)) {
      if (!initialized) error = "gpu_mode=5 executor is not initialized";
      return false;
    }
    std::size_t padded_cells = 0;
    if (!checked_product(2 * shape.local_n_x, 2 * shape.local_n_y,
                         padded_cells)) {
      error = "gpu_mode=5 padded class product overflow";
      return false;
    }
    key = FftPlanCacheKey();
    key.device = identity;
    key.logical_n_x = shape.local_n_x;
    key.logical_n_y = shape.local_n_y;
    key.padded_n_x = 2 * shape.local_n_x;
    key.padded_n_y = 2 * shape.local_n_y;
    key.batch_capacity = options.batch_capacity;
    key.dimensions = {{key.padded_n_x, key.padded_n_y}};
    key.input_embed = key.dimensions;
    key.output_embed = key.dimensions;
    key.input_distance = padded_cells;
    key.output_distance = padded_cells;
    key.build_abi_hash = options.build_abi_hash;
    return true;
  }

  bool make_green_key(const WindowClassShape& shape,
                      GreenSpectrumCacheKey& key,
                      std::string& error) const {
    if (!initialized || !shape_is_valid(shape, error)) {
      if (!initialized) error = "gpu_mode=5 executor is not initialized";
      return false;
    }
    key = GreenSpectrumCacheKey();
    key.device = identity;
    key.logical_n_x = shape.local_n_x;
    key.logical_n_y = shape.local_n_y;
    key.padded_n_x = 2 * shape.local_n_x;
    key.padded_n_y = 2 * shape.local_n_y;
    key.delta_x_bits = float_bits(lattice->delta_x());
    key.delta_y_bits = float_bits(lattice->delta_y());
    key.delta_z_bits = float_bits(lattice->delta_z());
    key.charge_sign_bits = float_bits(lattice->charge_sign());
    key.field_kernel_factor_bits =
        double_bits(lattice->field_kernel_factor());
    key.spectrum_generator_version = identity.cufft_version;
    key.build_abi_hash = options.build_abi_hash;
    return true;
  }

  bool create_plan(const FftPlanCacheKey& key, PlanRecord*& output,
                   std::string& error) {
    auto found = plan_cache.find(key);
    if (found != plan_cache.end()) {
      ++stats.plan_cache_hits;
      output = found->second.get();
      return true;
    }
    ++stats.plan_cache_misses;
    const std::chrono::steady_clock::time_point build_start =
        std::chrono::steady_clock::now();
    std::unique_ptr<PlanRecord> record(new PlanRecord());
    record->key = key;
    const auto cleanup = [&]() {
      if (record->handle) {
        const cufftResult destroy_status = cufftDestroy(record->handle);
        if (destroy_status != CUFFT_SUCCESS) {
          std::string cleanup_error;
          cufft_ok(destroy_status,
                   "cufftDestroy(gpu_mode=5 class cleanup)",
                   cleanup_error);
          if (!error.empty()) error += "; cleanup also failed: ";
          error += cleanup_error;
          mark_poisoned();
        } else {
          record->handle = 0;
        }
      }
      if (!record->handle && record->workspace) {
        release_during_failure(record->workspace, record->workspace_bytes,
                               "cudaFree(gpu_mode=5 class workspace)",
                               error);
        record->workspace = nullptr;
      }
    };
    if (!cufft_ok(cufftCreate(&record->handle),
                  "cufftCreate(gpu_mode=5 class)", error) ||
        !cufft_ok(cufftSetAutoAllocation(record->handle, 0),
                  "cufftSetAutoAllocation(gpu_mode=5 class)", error)) {
      cleanup();
      return false;
    }
    int dimensions[2] = {static_cast<int>(key.padded_n_x),
                         static_cast<int>(key.padded_n_y)};
    int input_embed[2] = {static_cast<int>(key.input_embed[0]),
                          static_cast<int>(key.input_embed[1])};
    int output_embed[2] = {static_cast<int>(key.output_embed[0]),
                           static_cast<int>(key.output_embed[1])};
    std::size_t workspace_bytes = 0;
    const cufftResult make_status = cufftMakePlanMany(
        record->handle, 2, dimensions, input_embed,
        static_cast<int>(key.input_stride),
        static_cast<int>(key.input_distance), output_embed,
        static_cast<int>(key.output_stride),
        static_cast<int>(key.output_distance), CUFFT_Z2Z,
        static_cast<int>(key.batch_capacity), &workspace_bytes);
    if (!cufft_ok(make_status, "cufftMakePlanMany(gpu_mode=5 class)", error)) {
      cleanup();
      return false;
    }
    record->workspace_bytes = workspace_bytes;
    if (!allocate_device(&record->workspace, workspace_bytes,
                         "cudaMalloc(gpu_mode=5 class workspace)", error)) {
      cleanup();
      return false;
    }
    if (!cufft_ok(cufftSetWorkArea(record->handle, record->workspace),
                  "cufftSetWorkArea(gpu_mode=5 class)", error) ||
        !cufft_ok(cufftSetStream(record->handle, stream),
                  "cufftSetStream(gpu_mode=5 class)", error)) {
      cleanup();
      return false;
    }
    stats.plan_workspace_bytes += workspace_bytes;
    ++stats.plan_creations;
    stats.plan_build_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    output = record.get();
    plan_cache.emplace(key, std::move(record));
    return true;
  }

  bool create_green(const GreenSpectrumCacheKey& key, GreenRecord*& output,
                    std::string& error) {
    auto found = green_cache.find(key);
    if (found != green_cache.end()) {
      ++stats.green_cache_hits;
      output = found->second.get();
      return true;
    }
    ++stats.green_cache_misses;
    const std::chrono::steady_clock::time_point build_start =
        std::chrono::steady_clock::now();
    const int nx = static_cast<int>(key.logical_n_x);
    const int ny = static_cast<int>(key.logical_n_y);
    const int padded_nx = static_cast<int>(key.padded_n_x);
    const int padded_ny = static_cast<int>(key.padded_n_y);
    const std::size_t padded_cells =
        static_cast<std::size_t>(padded_nx) * padded_ny;
    std::vector<cufftDoubleComplex> host_green(padded_cells);
    for (std::size_t cell = 0; cell < padded_cells; ++cell) {
      host_green[cell].x = 0.0;
      host_green[cell].y = 0.0;
    }
    const double factor = lattice->field_kernel_factor();
    const float deltax = lattice->delta_x();
    const float deltay = lattice->delta_y();
    double phi0 = factor;
    phi0 *= f_potential_2(0.0, 0.0, 0.5 * deltax, 0.5 * deltay);
    for (int i1 = 0; i1 < nx; ++i1) {
      for (int i2 = 0; i2 < ny; ++i2) {
        const std::size_t j0 =
            static_cast<std::size_t>(i1) * padded_ny + i2;
        const double x0 = static_cast<double>(i1);
        const double y0 = static_cast<double>(i2);
        host_green[j0].x =
            factor * f_potential_2(x0 * deltax, y0 * deltay,
                                   0.5 * deltax, 0.5 * deltay) -
            phi0;
        if (i2 >= 1) {
          host_green[static_cast<std::size_t>(i1 + 1) * padded_ny - i2].x =
              host_green[j0].x;
          if (i1 >= 1) {
            host_green[static_cast<std::size_t>(padded_nx - i1) *
                           padded_ny +
                       i2]
                .x = host_green[j0].x;
            host_green[static_cast<std::size_t>(padded_nx - i1) *
                           padded_ny +
                       padded_ny - i2]
                .x = host_green[j0].x;
          }
        } else if (i1 >= 1) {
          host_green[static_cast<std::size_t>(padded_nx - i1) * padded_ny]
              .x = host_green[j0].x;
        }
      }
    }

    std::unique_ptr<GreenRecord> record(new GreenRecord());
    record->key = key;
    record->spectrum_bytes = padded_cells * sizeof(cufftDoubleComplex);
    cufftHandle generator = 0;
    void* generator_workspace = nullptr;
    std::size_t generator_workspace_bytes = 0;
    bool enqueued = false;
    const auto cleanup = [&]() {
      if (enqueued && stream) cudaStreamSynchronize(stream);
      if (generator) {
        const cufftResult destroy_status = cufftDestroy(generator);
        if (destroy_status != CUFFT_SUCCESS) {
          std::string cleanup_error;
          cufft_ok(destroy_status,
                   "cufftDestroy(gpu_mode=5 Green cleanup)",
                   cleanup_error);
          if (!error.empty()) error += "; cleanup also failed: ";
          error += cleanup_error;
          mark_poisoned();
        } else {
          generator = 0;
        }
      }
      if (!generator && generator_workspace)
        release_during_failure(generator_workspace,
                               generator_workspace_bytes,
                               "cudaFree(gpu_mode=5 Green workspace)",
                               error);
      if (!generator && record->spectrum) {
        release_during_failure(record->spectrum, record->spectrum_bytes,
                               "cudaFree(gpu_mode=5 Green spectrum)",
                               error);
        record->spectrum = nullptr;
      }
    };
    if (!cufft_ok(cufftCreate(&generator),
                  "cufftCreate(gpu_mode=5 Green)", error) ||
        !cufft_ok(cufftSetAutoAllocation(generator, 0),
                  "cufftSetAutoAllocation(gpu_mode=5 Green)", error)) {
      cleanup();
      return false;
    }
    int dimensions[2] = {padded_nx, padded_ny};
    if (!cufft_ok(cufftMakePlanMany(
                      generator, 2, dimensions, nullptr, 1,
                      static_cast<int>(padded_cells), nullptr, 1,
                      static_cast<int>(padded_cells), CUFFT_Z2Z, 1,
                      &generator_workspace_bytes),
                  "cufftMakePlanMany(gpu_mode=5 Green)", error) ||
        !allocate_device(reinterpret_cast<void**>(&record->spectrum),
                         record->spectrum_bytes,
                         "cudaMalloc(gpu_mode=5 Green spectrum)", error) ||
        !allocate_device(&generator_workspace, generator_workspace_bytes,
                         "cudaMalloc(gpu_mode=5 Green workspace)", error) ||
        !cufft_ok(cufftSetWorkArea(generator, generator_workspace),
                  "cufftSetWorkArea(gpu_mode=5 Green)", error) ||
        !cufft_ok(cufftSetStream(generator, stream),
                  "cufftSetStream(gpu_mode=5 Green)", error)) {
      cleanup();
      return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(record->spectrum, host_green.data(),
                                 record->spectrum_bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(gpu_mode=5 Green H2D)", error)) {
      cleanup();
      return false;
    }
    enqueued = true;
    if (!cufft_ok(cufftExecZ2Z(generator, record->spectrum,
                               record->spectrum, CUFFT_FORWARD),
                  "cufftExecZ2Z(gpu_mode=5 Green forward)", error)) {
      cleanup();
      return false;
    }
    const int normalization_cells = 4 * nx * ny;
    const double normalization_denominator =
        static_cast<double>(normalization_cells);
    scale_spectrum_kernel<<<blocks_for(padded_cells), kThreads, 0, stream>>>(
        record->spectrum, padded_cells, normalization_denominator);
    if (!cuda_ok(cudaGetLastError(),
                 "scale_spectrum_kernel(gpu_mode=5 Green)", error) ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 "cudaStreamSynchronize(gpu_mode=5 Green)", error)) {
      cleanup();
      return false;
    }
    enqueued = false;
    if (!cufft_ok(cufftDestroy(generator),
                  "cufftDestroy(gpu_mode=5 Green)", error)) {
      mark_poisoned();
      cleanup();
      return false;
    }
    generator = 0;
    if (!release_device(generator_workspace, generator_workspace_bytes,
                        "cudaFree(gpu_mode=5 Green workspace)", error)) {
      cleanup();
      return false;
    }
    generator_workspace = nullptr;
    ++stats.h2d_operations;
    stats.h2d_bytes += record->spectrum_bytes;
    ++stats.kernel_launches;
    ++stats.cufft_executes;
    ++stats.synchronizations;
    ++stats.green_creations;
    stats.green_build_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    output = record.get();
    green_cache.emplace(key, std::move(record));
    return true;
  }

  bool create_class_context(const WindowClassShape& shape, PlanRecord* plan,
                            GreenRecord* green, std::string& error) {
    if (class_contexts.find(shape) != class_contexts.end()) return true;
    std::unique_ptr<ClassContext> context(new ClassContext());
    context->shape = shape;
    context->plan = plan;
    context->green = green;
    if (!checked_product(shape.local_n_x, shape.local_n_y,
                         context->local_cells) ||
        !checked_product(2 * shape.local_n_x, 2 * shape.local_n_y,
                         context->padded_cells)) {
      error = "gpu_mode=5 class scratch dimensions overflow";
      return false;
    }
    std::size_t batch_local_cells = 0;
    std::size_t batch_padded_cells = 0;
    if (!checked_product(options.batch_capacity, context->local_cells,
                         batch_local_cells) ||
        !checked_product(options.batch_capacity, context->padded_cells,
                         batch_padded_cells) ||
        !checked_product(batch_local_cells, sizeof(double),
                         context->one_real_batch_bytes) ||
        !checked_product(batch_padded_cells, sizeof(cufftDoubleComplex),
                         context->spectrum_bytes)) {
      error = "gpu_mode=5 class scratch byte count overflow";
      return false;
    }
    const auto cleanup = [&]() {
      if (context->phi2)
        release_during_failure(context->phi2,
                               context->one_real_batch_bytes,
                               "cudaFree(gpu_mode=5 class phi2)", error);
      if (context->phi1)
        release_during_failure(context->phi1,
                               context->one_real_batch_bytes,
                               "cudaFree(gpu_mode=5 class phi1)", error);
      if (context->rho2)
        release_during_failure(context->rho2,
                               context->one_real_batch_bytes,
                               "cudaFree(gpu_mode=5 class rho2)", error);
      if (context->rho1)
        release_during_failure(context->rho1,
                               context->one_real_batch_bytes,
                               "cudaFree(gpu_mode=5 class rho1)", error);
      if (context->spectrum)
        release_during_failure(context->spectrum, context->spectrum_bytes,
                               "cudaFree(gpu_mode=5 class spectrum)", error);
      context->phi2 = context->phi1 = nullptr;
      context->rho2 = context->rho1 = nullptr;
      context->spectrum = nullptr;
    };
    if (!allocate_device(reinterpret_cast<void**>(&context->spectrum),
                         context->spectrum_bytes,
                         "cudaMalloc(gpu_mode=5 class spectrum)", error) ||
        !allocate_device(reinterpret_cast<void**>(&context->rho1),
                         context->one_real_batch_bytes,
                         "cudaMalloc(gpu_mode=5 class rho1)", error) ||
        !allocate_device(reinterpret_cast<void**>(&context->rho2),
                         context->one_real_batch_bytes,
                         "cudaMalloc(gpu_mode=5 class rho2)", error) ||
        !allocate_device(reinterpret_cast<void**>(&context->phi1),
                         context->one_real_batch_bytes,
                         "cudaMalloc(gpu_mode=5 class phi1)", error) ||
        !allocate_device(reinterpret_cast<void**>(&context->phi2),
                         context->one_real_batch_bytes,
                         "cudaMalloc(gpu_mode=5 class phi2)", error)) {
      cleanup();
      return false;
    }
    class_contexts.emplace(shape, std::move(context));
    return true;
  }

  bool rollback_prebuild(std::string& error) {
    bool success = true;
    const auto append = [&](const std::string& cleanup_error) {
      if (cleanup_error.empty()) return;
      if (!error.empty()) error += "; prebuild rollback also failed: ";
      error += cleanup_error;
      success = false;
    };
    if (stream) {
      std::string cleanup_error;
      if (!cuda_ok(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(gpu_mode=5 prebuild rollback)",
                   cleanup_error)) {
        append(cleanup_error);
        mark_poisoned();
        return false;
      }
    }
    for (auto& entry : plan_cache) {
      PlanRecord& record = *entry.second;
      if (record.handle) {
        std::string cleanup_error;
        if (!cufft_ok(cufftDestroy(record.handle),
                      "cufftDestroy(gpu_mode=5 prebuild rollback)",
                      cleanup_error))
          append(cleanup_error);
        else
          record.handle = 0;
      }
    }
    const auto release = [&](void*& pointer, std::size_t bytes,
                             const char* label) {
      if (!pointer) return;
      std::string cleanup_error;
      if (!release_device(pointer, bytes, label, cleanup_error))
        append(cleanup_error);
      else
        pointer = nullptr;
    };
    for (auto& entry : class_contexts) {
      ClassContext& context = *entry.second;
      void* pointer = context.phi2;
      release(pointer, context.one_real_batch_bytes,
              "cudaFree(gpu_mode=5 rollback phi2)");
      context.phi2 = static_cast<double*>(pointer);
      pointer = context.phi1;
      release(pointer, context.one_real_batch_bytes,
              "cudaFree(gpu_mode=5 rollback phi1)");
      context.phi1 = static_cast<double*>(pointer);
      pointer = context.rho2;
      release(pointer, context.one_real_batch_bytes,
              "cudaFree(gpu_mode=5 rollback rho2)");
      context.rho2 = static_cast<double*>(pointer);
      pointer = context.rho1;
      release(pointer, context.one_real_batch_bytes,
              "cudaFree(gpu_mode=5 rollback rho1)");
      context.rho1 = static_cast<double*>(pointer);
      pointer = context.spectrum;
      release(pointer, context.spectrum_bytes,
              "cudaFree(gpu_mode=5 rollback spectrum)");
      context.spectrum = static_cast<cufftDoubleComplex*>(pointer);
    }
    for (auto& entry : green_cache) {
      GreenRecord& record = *entry.second;
      void* pointer = record.spectrum;
      release(pointer, record.spectrum_bytes,
              "cudaFree(gpu_mode=5 rollback Green)");
      record.spectrum = static_cast<cufftDoubleComplex*>(pointer);
    }
    for (auto& entry : plan_cache) {
      PlanRecord& record = *entry.second;
      if (record.handle) continue;
      void* pointer = record.workspace;
      release(pointer, record.workspace_bytes,
              "cudaFree(gpu_mode=5 rollback plan workspace)");
      record.workspace = pointer;
    }
    if (success) {
      class_contexts.clear();
      green_cache.clear();
      plan_cache.clear();
      stats.plan_workspace_bytes = 0;
      stats.classes_prebuilt = 0;
      stats.prebuild_complete = false;
      if (stats.tracked_device_bytes != shared_device_bytes) {
        std::ostringstream message;
        message << "prebuild rollback left " << stats.tracked_device_bytes
                << " tracked bytes instead of shared baseline "
                << shared_device_bytes;
        append(message.str());
      }
    }
    if (!success) mark_poisoned();
    return success;
  }

  AdaptiveResultBinding make_binding(
      const AdaptiveEncounterSubmission& submission,
      std::uint64_t fft_hash, std::uint64_t green_hash,
      double beam1_macro_charge, double beam2_macro_charge) const {
    AdaptiveResultBinding result;
    result.encounter = submission.binding;
    const UniformWindowPlan& plan = *submission.plan;
    result.lattice_hash = plan.lattice_hash();
    result.geometry_revision = lattice->geometry_revision();
    result.window_plan_hash = plan.plan_hash();
    result.origin_x = plan.origin_x();
    result.origin_y = plan.origin_y();
    result.local_n_x = plan.local_n_x();
    result.local_n_y = plan.local_n_y();
    result.padded_n_x = plan.padded_n_x();
    result.padded_n_y = plan.padded_n_y();
    result.beam1_particle_count = plan.beam1_sources().size();
    result.beam2_particle_count = plan.beam2_sources().size();
    result.beam1_macro_charge_bits = double_bits(beam1_macro_charge);
    result.beam2_macro_charge_bits = double_bits(beam2_macro_charge);
    result.fft_plan_key_hash = fft_hash;
    result.green_spectrum_key_hash = green_hash;
    Hash64 hash;
    hash.size(result.encounter.canonical_result_slot);
    hash.u64(result.encounter.encounter_id);
    hash.i64(result.encounter.beam1_slice);
    hash.i64(result.encounter.beam2_slice);
    hash.u64(result.encounter.beam1_revision);
    hash.u64(result.encounter.beam2_revision);
    hash.u64(result.encounter.beam1_order_token);
    hash.u64(result.encounter.beam2_order_token);
    hash.u64(result.lattice_hash);
    hash.u64(result.geometry_revision);
    hash.u64(result.window_plan_hash);
    hash.i64(result.origin_x);
    hash.i64(result.origin_y);
    hash.size(result.local_n_x);
    hash.size(result.local_n_y);
    hash.size(result.padded_n_x);
    hash.size(result.padded_n_y);
    hash.size(result.beam1_particle_count);
    hash.size(result.beam2_particle_count);
    hash.u64(result.beam1_macro_charge_bits);
    hash.u64(result.beam2_macro_charge_bits);
    hash.u64(result.fft_plan_key_hash);
    hash.u64(result.green_spectrum_key_hash);
    result.binding_hash = hash.value;
    return result;
  }
};

AdaptiveWaveBatchExecutor::AdaptiveWaveBatchExecutor() : impl_(new Impl()) {}

AdaptiveWaveBatchExecutor::~AdaptiveWaveBatchExecutor() { delete impl_; }

bool AdaptiveWaveBatchExecutor::initialize(
    const AdaptiveGpuOptions& options,
    const CanonicalLatticeSnapshot& lattice,
    std::string& error) {
  error.clear();
  if (impl_->initialization_attempted) {
    error = "gpu_mode=5 executor initialization is one-shot";
    return false;
  }
  impl_->initialization_attempted = true;
  const std::chrono::steady_clock::time_point setup_start =
      std::chrono::steady_clock::now();
  if (options.device < 0 || options.batch_capacity == 0 ||
      options.batch_capacity > 65535 ||
      options.maximum_beam1_descriptors == 0 ||
      options.maximum_beam2_descriptors == 0 ||
      !(options.memory_fraction > 0.0) || options.memory_fraction > 1.0 ||
      !std::isfinite(options.memory_fraction) || !options.pinned_io ||
      options.build_abi_hash == 0) {
    error = "gpu_mode=5 requires a valid device, 1..65535 batch capacity, "
            "nonzero descriptor caps/build hash, 0<memory_fraction<=1, and "
            "pinned I/O";
    return false;
  }
  if (options.deposition != DepositionVariantId::GlobalAtomic) {
    std::ostringstream message;
    message << "gpu_mode=5 deposition variant '"
            << deposition_variant_name(options.deposition)
            << "' is enumerated for provenance but is not implemented or "
               "benchmarked; only global-atomic is admitted";
    error = message.str();
    return false;
  }
  if (lattice.n_x() < 4 || lattice.n_y() < 4 ||
      lattice.n_x() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 2) ||
      lattice.n_y() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 2) ||
      !std::isfinite(lattice.delta_x()) || lattice.delta_x() <= 0.0f ||
      !std::isfinite(lattice.delta_y()) || lattice.delta_y() <= 0.0f ||
      !std::isfinite(lattice.delta_z()) || lattice.delta_z() <= 0.0f ||
      !std::isfinite(lattice.inverse_delta_x()) ||
      lattice.inverse_delta_x() <= 0.0f ||
      !std::isfinite(lattice.inverse_delta_y()) ||
      lattice.inverse_delta_y() <= 0.0f ||
      !std::isfinite(lattice.charge_sign()) ||
      !std::isfinite(lattice.field_kernel_factor())) {
    error = "gpu_mode=5 received an invalid canonical lattice snapshot";
    return false;
  }

  impl_->options = options;
  impl_->lattice.reset(new CanonicalLatticeSnapshot(lattice));
  impl_->stats.requested_deposition = options.deposition;
  impl_->stats.actual_deposition = DepositionVariantId::GlobalAtomic;
  if (!cuda_ok(cudaSetDevice(options.device),
               "cudaSetDevice(gpu_mode=5)", error))
    return false;
  impl_->identity.ordinal = options.device;
  cudaDeviceProp properties;
  if (!cuda_ok(cudaGetDeviceProperties(&properties, options.device),
               "cudaGetDeviceProperties(gpu_mode=5)", error))
    return false;
  if (properties.major < 6) {
    error = "gpu_mode=5 global FP64 atomic deposition requires compute "
            "capability 6.0 or newer";
    return false;
  }
  impl_->identity.compute_major = properties.major;
  impl_->identity.compute_minor = properties.minor;
#if CUDART_VERSION >= 10000
  std::memcpy(impl_->identity.uuid.data(), properties.uuid.bytes,
              impl_->identity.uuid.size());
#endif
  if (!cuda_ok(cudaRuntimeGetVersion(&impl_->identity.cuda_runtime_version),
               "cudaRuntimeGetVersion(gpu_mode=5)", error) ||
      !cuda_ok(cudaDriverGetVersion(&impl_->identity.cuda_driver_version),
               "cudaDriverGetVersion(gpu_mode=5)", error) ||
      !cufft_ok(cufftGetVersion(&impl_->identity.cufft_version),
                "cufftGetVersion(gpu_mode=5)", error))
    return false;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
               "cudaMemGetInfo(gpu_mode=5 initialize)", error))
    return false;
  impl_->free_bytes_at_initialize = free_bytes;
  std::size_t budget = static_cast<std::size_t>(
      options.memory_fraction * static_cast<long double>(total_bytes));
  if (options.maximum_device_bytes != 0)
    budget = std::min(budget, options.maximum_device_bytes);
  impl_->stats.memory_budget_bytes = budget;

  std::size_t descriptor_count1 = 0;
  std::size_t descriptor_count2 = 0;
  std::size_t field_values1 = 0;
  std::size_t field_values2 = 0;
  if (!checked_product(options.batch_capacity,
                       options.maximum_beam1_descriptors,
                       descriptor_count1) ||
      !checked_product(options.batch_capacity,
                       options.maximum_beam2_descriptors,
                       descriptor_count2) ||
      !checked_product(descriptor_count1, static_cast<std::size_t>(2),
                       field_values1) ||
      !checked_product(descriptor_count2, static_cast<std::size_t>(2),
                       field_values2) ||
      !checked_product(descriptor_count1, sizeof(DeviceDepositDescriptor),
                       impl_->deposit1_bytes) ||
      !checked_product(descriptor_count2, sizeof(DeviceDepositDescriptor),
                       impl_->deposit2_bytes) ||
      !checked_product(descriptor_count1, sizeof(DeviceGatherDescriptor),
                       impl_->gather1_bytes) ||
      !checked_product(descriptor_count2, sizeof(DeviceGatherDescriptor),
                       impl_->gather2_bytes) ||
      !checked_product(options.batch_capacity,
                       sizeof(DeviceSlotMetadata), impl_->slot_bytes) ||
      !checked_product(options.batch_capacity,
                       sizeof(DeviceEncounterMetrics), impl_->metric_bytes) ||
      !checked_product(field_values1, sizeof(float), impl_->field1_bytes) ||
      !checked_product(field_values2, sizeof(float), impl_->field2_bytes)) {
    error = "gpu_mode=5 shared descriptor buffer sizes overflow size_t";
    return false;
  }
  std::size_t shared_device_bytes = 0;
  if (!checked_add(shared_device_bytes, impl_->deposit1_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->deposit2_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->gather1_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->gather2_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->slot_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->metric_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->field1_bytes,
                   shared_device_bytes) ||
      !checked_add(shared_device_bytes, impl_->field2_bytes,
                   shared_device_bytes)) {
    error = "gpu_mode=5 shared device byte sum overflows size_t";
    return false;
  }
  if (shared_device_bytes > budget || shared_device_bytes > free_bytes) {
    std::ostringstream message;
    message << "gpu_mode=5 shared buffers require " << shared_device_bytes
            << " device bytes, exceeding "
            << (shared_device_bytes > budget ? "configured budget "
                                             : "currently free memory ")
            << (shared_device_bytes > budget ? budget : free_bytes);
    error = message.str();
    return false;
  }

  if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(gpu_mode=5)", error))
    return false;
  for (int index = 0; index < 4; ++index) {
    if (!cuda_ok(cudaEventCreate(&impl_->events[index]),
                 "cudaEventCreate(gpu_mode=5)", error)) {
      impl_->mark_poisoned();
      return false;
    }
  }

#define FEEBL_GPU5_ALLOC_DEVICE(member, bytes, label)                       \
  do {                                                                      \
    if (!impl_->allocate_device(reinterpret_cast<void**>(&(member)), (bytes),\
                                (label), error)) {                           \
      impl_->mark_poisoned();                                                \
      return false;                                                          \
    }                                                                        \
  } while (0)
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_deposit1, impl_->deposit1_bytes,
                          "cudaMalloc(gpu_mode=5 beam1 deposits)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_deposit2, impl_->deposit2_bytes,
                          "cudaMalloc(gpu_mode=5 beam2 deposits)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_gather1, impl_->gather1_bytes,
                          "cudaMalloc(gpu_mode=5 beam1 gathers)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_gather2, impl_->gather2_bytes,
                          "cudaMalloc(gpu_mode=5 beam2 gathers)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_slots, impl_->slot_bytes,
                          "cudaMalloc(gpu_mode=5 slot metadata)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_metrics, impl_->metric_bytes,
                          "cudaMalloc(gpu_mode=5 metrics)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_fields1, impl_->field1_bytes,
                          "cudaMalloc(gpu_mode=5 beam1 fields)");
  FEEBL_GPU5_ALLOC_DEVICE(impl_->device_fields2, impl_->field2_bytes,
                          "cudaMalloc(gpu_mode=5 beam2 fields)");
#undef FEEBL_GPU5_ALLOC_DEVICE

#define FEEBL_GPU5_ALLOC_PINNED(member, bytes, label)                        \
  do {                                                                       \
    if (!impl_->allocate_pinned(reinterpret_cast<void**>(&(member)), (bytes), \
                                (label), error)) {                            \
      impl_->mark_poisoned();                                                 \
      return false;                                                           \
    }                                                                         \
  } while (0)
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_deposit1, impl_->deposit1_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam1 deposits)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_deposit2, impl_->deposit2_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam2 deposits)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_gather1, impl_->gather1_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam1 gathers)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_gather2, impl_->gather2_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam2 gathers)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_slots, impl_->slot_bytes,
                          "cudaHostAlloc(gpu_mode=5 slot metadata)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_metrics, impl_->metric_bytes,
                          "cudaHostAlloc(gpu_mode=5 metrics)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_fields1, impl_->field1_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam1 fields)");
  FEEBL_GPU5_ALLOC_PINNED(impl_->host_fields2, impl_->field2_bytes,
                          "cudaHostAlloc(gpu_mode=5 beam2 fields)");
#undef FEEBL_GPU5_ALLOC_PINNED

  std::memset(impl_->host_deposit1, 0, impl_->deposit1_bytes);
  std::memset(impl_->host_deposit2, 0, impl_->deposit2_bytes);
  std::memset(impl_->host_gather1, 0, impl_->gather1_bytes);
  std::memset(impl_->host_gather2, 0, impl_->gather2_bytes);
  std::memset(impl_->host_slots, 0, impl_->slot_bytes);
  std::memset(impl_->host_metrics, 0, impl_->metric_bytes);
  std::memset(impl_->host_fields1, 0, impl_->field1_bytes);
  std::memset(impl_->host_fields2, 0, impl_->field2_bytes);
  impl_->shared_device_bytes = impl_->stats.tracked_device_bytes;
  if (!impl_->refresh_high_water(error) ||
      impl_->stats.measured_device_high_water_bytes > budget) {
    if (error.empty())
      error = "gpu_mode=5 measured initialization high-water exceeds its "
              "configured device-memory budget";
    impl_->mark_poisoned();
    return false;
  }
  std::ostringstream description;
  description << properties.name << " cc " << properties.major << '.'
              << properties.minor << " CUDA runtime "
              << impl_->identity.cuda_runtime_version << " cuFFT "
              << impl_->identity.cufft_version;
  impl_->device_text = description.str();
  impl_->stats.setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  impl_->initialized = true;
  return true;
}

bool AdaptiveWaveBatchExecutor::prebuild_classes(
    const WindowClassShape* classes, std::size_t class_count,
    std::string& error) {
  error.clear();
  if (!impl_->initialized || impl_->poisoned) {
    error = impl_->poisoned
                ? "gpu_mode=5 context is poisoned"
                : "gpu_mode=5 executor must be initialized before prebuild";
    return false;
  }
  if (!classes || class_count == 0) {
    error = "gpu_mode=5 prebuild requires at least one window class";
    return false;
  }
  if (impl_->prebuild_attempted) {
    error = "gpu_mode=5 class prebuild is one-shot";
    return false;
  }
  impl_->prebuild_attempted = true;
  ++impl_->stats.prebuild_calls;
  const std::chrono::steady_clock::time_point prebuild_start =
      std::chrono::steady_clock::now();
  const auto rollback_and_poison = [&]() {
    impl_->rollback_prebuild(error);
    impl_->mark_poisoned();
    return false;
  };
  try {
    std::set<WindowClassShape> unique_classes;
    for (std::size_t index = 0; index < class_count; ++index) {
      if (!impl_->shape_is_valid(classes[index], error)) {
        impl_->mark_poisoned();
        return false;
      }
      unique_classes.insert(classes[index]);
    }
    if (!cuda_ok(cudaSetDevice(impl_->options.device),
                 "cudaSetDevice(gpu_mode=5 prebuild)", error)) {
      impl_->mark_poisoned();
      return false;
    }
    for (const WindowClassShape& shape : unique_classes) {
      FftPlanCacheKey fft_key;
      GreenSpectrumCacheKey green_key;
      Impl::PlanRecord* plan = nullptr;
      Impl::GreenRecord* green = nullptr;
      if (!impl_->make_fft_key(shape, fft_key, error) ||
          !impl_->make_green_key(shape, green_key, error) ||
          !impl_->create_plan(fft_key, plan, error) ||
          !impl_->create_green(green_key, green, error) ||
          !impl_->create_class_context(shape, plan, green, error)) {
        return rollback_and_poison();
      }
    }
    if (!cuda_ok(cudaStreamSynchronize(impl_->stream),
                 "cudaStreamSynchronize(gpu_mode=5 prebuild boundary)",
                 error) ||
        !impl_->refresh_high_water(error)) {
      return rollback_and_poison();
    }
    ++impl_->stats.synchronizations;
    if (impl_->stats.tracked_device_bytes >
            impl_->stats.memory_budget_bytes ||
        impl_->stats.measured_device_high_water_bytes >
            impl_->stats.memory_budget_bytes) {
      error = "gpu_mode=5 prebuild exceeded its configured memory budget";
      return rollback_and_poison();
    }
    impl_->stats.classes_prebuilt = unique_classes.size();
    impl_->stats.prebuild_complete = true;
    impl_->stats.setup_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prebuild_start).count();
    return true;
  } catch (const std::exception& exception) {
    error = std::string("gpu_mode=5 prebuild failed: ") + exception.what();
    return rollback_and_poison();
  } catch (...) {
    error = "gpu_mode=5 prebuild failed with an unknown host exception";
    return rollback_and_poison();
  }
}

bool AdaptiveWaveBatchExecutor::execute_batch(
    const AdaptiveEncounterSubmission* submissions,
    std::size_t encounter_count, double beam1_macro_charge,
    double beam2_macro_charge, AdaptiveEncounterResult* canonical_results,
    std::size_t canonical_result_capacity, std::string& error) {
  error.clear();
  if (!impl_->initialized || !impl_->stats.prebuild_complete ||
      impl_->poisoned) {
    error = impl_->poisoned
                ? "gpu_mode=5 context is poisoned"
                : (!impl_->initialized
                       ? "gpu_mode=5 executor is not initialized"
                       : "gpu_mode=5 class prebuild is incomplete");
    return false;
  }
  if (!submissions || !canonical_results || encounter_count == 0 ||
      encounter_count > impl_->options.batch_capacity ||
      canonical_result_capacity == 0) {
    error = "gpu_mode=5 execute requires 1..batch_capacity submissions and "
            "a nonempty canonical result table";
    return false;
  }
  if (!std::isfinite(beam1_macro_charge) ||
      !std::isfinite(beam2_macro_charge)) {
    error = "gpu_mode=5 macro charges must be finite";
    return false;
  }

  try {
    std::set<std::size_t> result_slots;
    bool saw_empty = false;
    bool saw_nonempty = false;
    WindowClassShape batch_shape;
    Impl::ClassContext* context = nullptr;
    std::vector<std::size_t> expected_deposit1_in(encounter_count, 0);
    std::vector<std::size_t> expected_deposit2_in(encounter_count, 0);
    std::vector<std::size_t> expected_gather1_in(encounter_count, 0);
    std::vector<std::size_t> expected_gather2_in(encounter_count, 0);

    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      const AdaptiveEncounterSubmission& submission = submissions[slot];
      if (!submission.plan) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot << " has no plan";
        error = message.str();
        return false;
      }
      const std::size_t result_slot =
          submission.binding.canonical_result_slot;
      if (result_slot >= canonical_result_capacity) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot
                << " result slot exceeds canonical result capacity";
        error = message.str();
        return false;
      }
      if (!result_slots.insert(result_slot).second) {
        std::ostringstream message;
        message << "gpu_mode=5 duplicate canonical result slot "
                << result_slot
                << " rejected before any device work or publication";
        error = message.str();
        return false;
      }
      const UniformWindowPlan& plan = *submission.plan;
      if (plan.lattice_hash() != impl_->lattice->hash()) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot
                << " was built from a different canonical lattice";
        error = message.str();
        return false;
      }
      if (plan.beam1_sources().size() != plan.beam1_targets().size() ||
          plan.beam2_sources().size() != plan.beam2_targets().size() ||
          plan.beam1_sources().size() >
              impl_->options.maximum_beam1_descriptors ||
          plan.beam2_sources().size() >
              impl_->options.maximum_beam2_descriptors) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot
                << " violates canonical source/target order binding or "
                   "descriptor caps";
        error = message.str();
        return false;
      }
      expected_deposit1_in[slot] = accepted_count(plan.beam1_sources());
      expected_deposit2_in[slot] = accepted_count(plan.beam2_sources());
      expected_gather1_in[slot] = accepted_count(plan.beam1_targets());
      expected_gather2_in[slot] = accepted_count(plan.beam2_targets());
      if (expected_deposit1_in[slot] + expected_deposit2_in[slot] !=
              plan.accepted_source_count() ||
          expected_gather1_in[slot] + expected_gather2_in[slot] !=
              plan.accepted_target_count()) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot
                << " acceptance totals disagree with its immutable plan";
        error = message.str();
        return false;
      }

      if (plan.empty()) {
        saw_empty = true;
        if (expected_deposit1_in[slot] || expected_deposit2_in[slot] ||
            expected_gather1_in[slot] || expected_gather2_in[slot] ||
            !plan.required_nodes().empty()) {
          error = "gpu_mode=5 empty plan contains accepted descriptors or "
                  "required nodes";
          return false;
        }
        continue;
      }
      saw_nonempty = true;
      const WindowClassShape shape{plan.local_n_x(), plan.local_n_y()};
      std::string shape_error;
      if (!impl_->shape_is_valid(shape, shape_error) ||
          plan.padded_n_x() != 2 * plan.local_n_x() ||
          plan.padded_n_y() != 2 * plan.local_n_y() ||
          plan.origin_x() < 0 || plan.origin_y() < 0 ||
          static_cast<std::uint64_t>(plan.origin_x()) + plan.local_n_x() >
              impl_->lattice->n_x() ||
          static_cast<std::uint64_t>(plan.origin_y()) + plan.local_n_y() >
              impl_->lattice->n_y()) {
        std::ostringstream message;
        message << "gpu_mode=5 submission " << slot
                << " has invalid local/padded dimensions or origin";
        error = message.str();
        return false;
      }
      if (!context) {
        batch_shape = shape;
        auto found = impl_->class_contexts.find(shape);
        if (found == impl_->class_contexts.end()) {
          std::ostringstream message;
          message << "gpu_mode=5 class " << shape.local_n_x << 'x'
                  << shape.local_n_y
                  << " was not explicitly prebuilt; cold fallback is "
                     "forbidden";
          error = message.str();
          return false;
        }
        context = found->second.get();
      } else if (!(shape == batch_shape)) {
        error = "gpu_mode=5 execute_batch accepts exactly one prebucketed "
                "window class per call";
        return false;
      }
      for (const CanonicalNode& node : plan.required_nodes()) {
        if (!plan.contains_canonical_node(node)) {
          error = "gpu_mode=5 immutable required-node set escapes its plan";
          return false;
        }
      }
      for (const CanonicalDepositDescriptor& descriptor :
           plan.beam1_sources()) {
        if (!host_deposit_contained(descriptor, plan)) {
          error = "gpu_mode=5 beam1 deposit descriptor escapes its plan";
          return false;
        }
      }
      for (const CanonicalDepositDescriptor& descriptor :
           plan.beam2_sources()) {
        if (!host_deposit_contained(descriptor, plan)) {
          error = "gpu_mode=5 beam2 deposit descriptor escapes its plan";
          return false;
        }
      }
      for (const CanonicalGatherDescriptor& descriptor :
           plan.beam1_targets()) {
        if (!host_gather_contained(descriptor, plan)) {
          error = "gpu_mode=5 beam1 gather descriptor escapes its plan";
          return false;
        }
      }
      for (const CanonicalGatherDescriptor& descriptor :
           plan.beam2_targets()) {
        if (!host_gather_contained(descriptor, plan)) {
          error = "gpu_mode=5 beam2 gather descriptor escapes its plan";
          return false;
        }
      }
    }
    if (saw_empty && saw_nonempty) {
      error = "gpu_mode=5 empty and nonempty plans must be placed in "
              "separate scheduler batches";
      return false;
    }

    std::vector<AdaptiveEncounterResult> staged(encounter_count);
    if (!saw_nonempty) {
      for (std::size_t slot = 0; slot < encounter_count; ++slot) {
        const UniformWindowPlan& plan = *submissions[slot].plan;
        AdaptiveEncounterResult& result = staged[slot];
        result.fields1.assign(2 * plan.beam1_targets().size(), 0.0f);
        result.fields2.assign(2 * plan.beam2_targets().size(), 0.0f);
        result.beam1_out = plan.beam1_sources().size();
        result.beam2_out = plan.beam2_sources().size();
        result.gather_beam1_out = plan.beam1_targets().size();
        result.gather_beam2_out = plan.beam2_targets().size();
        result.binding = impl_->make_binding(
            submissions[slot], 0, 0, beam1_macro_charge,
            beam2_macro_charge);
        result.valid = true;
      }
      for (std::size_t slot = 0; slot < encounter_count; ++slot)
        canonical_results[submissions[slot].binding.canonical_result_slot]
            .swap(staged[slot]);
      ++impl_->stats.batch_calls;
      impl_->stats.encounters += encounter_count;
      impl_->stats.empty_encounters += encounter_count;
      return true;
    }

    // Reserve every publishable field vector before enqueuing work.  Host
    // allocation failure therefore cannot occur between device validation
    // and the no-throw canonical-slot swaps.
    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      staged[slot].fields1.resize(
          2 * submissions[slot].plan->beam1_targets().size());
      staged[slot].fields2.resize(
          2 * submissions[slot].plan->beam2_targets().size());
    }

    ++impl_->stats.plan_cache_hits;
    ++impl_->stats.green_cache_hits;
    const std::chrono::steady_clock::time_point pack_start =
        std::chrono::steady_clock::now();
    std::size_t active_deposit1_bytes = 0;
    std::size_t active_deposit2_bytes = 0;
    std::size_t active_gather1_bytes = 0;
    std::size_t active_gather2_bytes = 0;
    std::size_t active_field1_bytes = 0;
    std::size_t active_field2_bytes = 0;
    checked_product(encounter_count,
                    impl_->options.maximum_beam1_descriptors,
                    active_deposit1_bytes);
    active_gather1_bytes = active_deposit1_bytes;
    checked_product(encounter_count,
                    impl_->options.maximum_beam2_descriptors,
                    active_deposit2_bytes);
    active_gather2_bytes = active_deposit2_bytes;
    active_deposit1_bytes *= sizeof(DeviceDepositDescriptor);
    active_deposit2_bytes *= sizeof(DeviceDepositDescriptor);
    active_gather1_bytes *= sizeof(DeviceGatherDescriptor);
    active_gather2_bytes *= sizeof(DeviceGatherDescriptor);
    active_field1_bytes =
        2 * encounter_count * impl_->options.maximum_beam1_descriptors *
        sizeof(float);
    active_field2_bytes =
        2 * encounter_count * impl_->options.maximum_beam2_descriptors *
        sizeof(float);
    const std::size_t active_slot_bytes =
        encounter_count * sizeof(DeviceSlotMetadata);
    const std::size_t active_metric_bytes =
        encounter_count * sizeof(DeviceEncounterMetrics);
    std::memset(impl_->host_deposit1, 0, active_deposit1_bytes);
    std::memset(impl_->host_deposit2, 0, active_deposit2_bytes);
    std::memset(impl_->host_gather1, 0, active_gather1_bytes);
    std::memset(impl_->host_gather2, 0, active_gather2_bytes);
    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      const UniformWindowPlan& plan = *submissions[slot].plan;
      impl_->host_slots[slot].origin_x =
          static_cast<long long>(plan.origin_x());
      impl_->host_slots[slot].origin_y =
          static_cast<long long>(plan.origin_y());
      impl_->host_slots[slot].beam1_count =
          static_cast<unsigned long long>(plan.beam1_sources().size());
      impl_->host_slots[slot].beam2_count =
          static_cast<unsigned long long>(plan.beam2_sources().size());
      DeviceDepositDescriptor* deposit1 =
          impl_->host_deposit1 +
          slot * impl_->options.maximum_beam1_descriptors;
      DeviceDepositDescriptor* deposit2 =
          impl_->host_deposit2 +
          slot * impl_->options.maximum_beam2_descriptors;
      DeviceGatherDescriptor* gather1 =
          impl_->host_gather1 +
          slot * impl_->options.maximum_beam1_descriptors;
      DeviceGatherDescriptor* gather2 =
          impl_->host_gather2 +
          slot * impl_->options.maximum_beam2_descriptors;
      for (std::size_t particle = 0;
           particle < plan.beam1_sources().size(); ++particle) {
        deposit1[particle] = pack_descriptor(plan.beam1_sources()[particle]);
        gather1[particle] = pack_descriptor(plan.beam1_targets()[particle]);
      }
      for (std::size_t particle = 0;
           particle < plan.beam2_sources().size(); ++particle) {
        deposit2[particle] = pack_descriptor(plan.beam2_sources()[particle]);
        gather2[particle] = pack_descriptor(plan.beam2_targets()[particle]);
      }
    }
    impl_->stats.host_pack_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pack_start).count();

    if (!cuda_ok(cudaSetDevice(impl_->options.device),
                 "cudaSetDevice(gpu_mode=5 execute)", error)) {
      impl_->mark_poisoned();
      return false;
    }
    bool work_enqueued = false;
    const auto fail_cuda = [&](cudaError_t status, const char* operation) {
      if (status == cudaSuccess) return false;
      cuda_ok(status, operation, error);
      if (work_enqueued) cudaStreamSynchronize(impl_->stream);
      impl_->mark_poisoned();
      return true;
    };
    const auto fail_cufft = [&](cufftResult status, const char* operation) {
      if (status == CUFFT_SUCCESS) return false;
      cufft_ok(status, operation, error);
      if (work_enqueued) cudaStreamSynchronize(impl_->stream);
      impl_->mark_poisoned();
      return true;
    };
    if (fail_cuda(cudaEventRecord(impl_->events[0], impl_->stream),
                  "cudaEventRecord(gpu_mode=5 start)"))
      return false;
    work_enqueued = true;
    if (fail_cuda(cudaMemcpyAsync(impl_->device_deposit1,
                                  impl_->host_deposit1,
                                  active_deposit1_bytes,
                                  cudaMemcpyHostToDevice, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam1 deposits H2D)") ||
        fail_cuda(cudaMemcpyAsync(impl_->device_deposit2,
                                  impl_->host_deposit2,
                                  active_deposit2_bytes,
                                  cudaMemcpyHostToDevice, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam2 deposits H2D)") ||
        fail_cuda(cudaMemcpyAsync(impl_->device_gather1, impl_->host_gather1,
                                  active_gather1_bytes,
                                  cudaMemcpyHostToDevice, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam1 gathers H2D)") ||
        fail_cuda(cudaMemcpyAsync(impl_->device_gather2, impl_->host_gather2,
                                  active_gather2_bytes,
                                  cudaMemcpyHostToDevice, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam2 gathers H2D)") ||
        fail_cuda(cudaMemcpyAsync(impl_->device_slots, impl_->host_slots,
                                  active_slot_bytes,
                                  cudaMemcpyHostToDevice, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 slot metadata H2D)") ||
        fail_cuda(cudaEventRecord(impl_->events[1], impl_->stream),
                  "cudaEventRecord(gpu_mode=5 after H2D)"))
      return false;

    const std::size_t active_real_bytes =
        encounter_count * context->local_cells * sizeof(double);
    if (fail_cuda(cudaMemsetAsync(context->rho1, 0, active_real_bytes,
                                  impl_->stream),
                  "cudaMemsetAsync(gpu_mode=5 rho1)") ||
        fail_cuda(cudaMemsetAsync(context->rho2, 0, active_real_bytes,
                                  impl_->stream),
                  "cudaMemsetAsync(gpu_mode=5 rho2)") ||
        fail_cuda(cudaMemsetAsync(impl_->device_metrics, 0,
                                  active_metric_bytes, impl_->stream),
                  "cudaMemsetAsync(gpu_mode=5 metrics)"))
      return false;

    const dim3 deposit_grid1(
        blocks_for(impl_->options.maximum_beam1_descriptors),
        static_cast<unsigned int>(encounter_count), 1);
    const dim3 deposit_grid2(
        blocks_for(impl_->options.maximum_beam2_descriptors),
        static_cast<unsigned int>(encounter_count), 1);
    deposit_descriptor_kernel<<<deposit_grid1, kThreads, 0, impl_->stream>>>(
        impl_->device_deposit1,
        impl_->options.maximum_beam1_descriptors, impl_->device_slots, 0,
        static_cast<int>(batch_shape.local_n_x),
        static_cast<int>(batch_shape.local_n_y), context->local_cells,
        beam1_macro_charge, context->rho1, impl_->device_metrics);
    if (fail_cuda(cudaGetLastError(),
                  "deposit_descriptor_kernel(gpu_mode=5 beam1)"))
      return false;
    deposit_descriptor_kernel<<<deposit_grid2, kThreads, 0, impl_->stream>>>(
        impl_->device_deposit2,
        impl_->options.maximum_beam2_descriptors, impl_->device_slots, 1,
        static_cast<int>(batch_shape.local_n_x),
        static_cast<int>(batch_shape.local_n_y), context->local_cells,
        beam2_macro_charge, context->rho2, impl_->device_metrics);
    if (fail_cuda(cudaGetLastError(),
                  "deposit_descriptor_kernel(gpu_mode=5 beam2)"))
      return false;
    std::uint64_t launches = 2;
    fine_dot_kernel<<<static_cast<unsigned int>(encounter_count), kThreads, 0,
                      impl_->stream>>>(context->rho1, context->rho2,
                                       context->local_cells,
                                       impl_->device_metrics);
    if (fail_cuda(cudaGetLastError(), "fine_dot_kernel(gpu_mode=5)"))
      return false;
    ++launches;

    const bool zero_field =
        std::fabs(impl_->lattice->charge_sign()) < 1.0e-5f;
    std::uint64_t fft_executes = 0;
    if (!zero_field) {
      if (fail_cuda(cudaMemsetAsync(context->spectrum, 0,
                                    context->spectrum_bytes, impl_->stream),
                    "cudaMemsetAsync(gpu_mode=5 spectrum)"))
        return false;
      pack_spectrum_kernel<<<blocks_for(encounter_count *
                                        context->local_cells),
                             kThreads, 0, impl_->stream>>>(
          context->rho1, context->rho2,
          static_cast<int>(batch_shape.local_n_y),
          static_cast<int>(2 * batch_shape.local_n_y), context->local_cells,
          context->padded_cells, encounter_count, context->spectrum);
      if (fail_cuda(cudaGetLastError(),
                    "pack_spectrum_kernel(gpu_mode=5)"))
        return false;
      ++launches;
      if (fail_cufft(cufftExecZ2Z(context->plan->handle,
                                  context->spectrum, context->spectrum,
                                  CUFFT_FORWARD),
                     "cufftExecZ2Z(gpu_mode=5 forward)"))
        return false;
      ++fft_executes;
      multiply_spectrum_kernel<<<
          blocks_for(impl_->options.batch_capacity * context->padded_cells),
          kThreads, 0, impl_->stream>>>(
          context->spectrum, context->green->spectrum,
          context->padded_cells, impl_->options.batch_capacity);
      if (fail_cuda(cudaGetLastError(),
                    "multiply_spectrum_kernel(gpu_mode=5)"))
        return false;
      ++launches;
      if (fail_cufft(cufftExecZ2Z(context->plan->handle,
                                  context->spectrum, context->spectrum,
                                  CUFFT_INVERSE),
                     "cufftExecZ2Z(gpu_mode=5 inverse)"))
        return false;
      ++fft_executes;
      unpack_spectrum_kernel<<<blocks_for(encounter_count *
                                          context->local_cells),
                               kThreads, 0, impl_->stream>>>(
          context->spectrum, static_cast<int>(batch_shape.local_n_y),
          static_cast<int>(2 * batch_shape.local_n_y), context->local_cells,
          context->padded_cells, encounter_count, context->phi1,
          context->phi2);
      if (fail_cuda(cudaGetLastError(),
                    "unpack_spectrum_kernel(gpu_mode=5)"))
        return false;
      ++launches;
    }

    gather_descriptor_kernel<<<deposit_grid1, kThreads, 0, impl_->stream>>>(
        impl_->device_gather1,
        impl_->options.maximum_beam1_descriptors, impl_->device_slots,
        context->phi2, 0, static_cast<int>(batch_shape.local_n_x),
        static_cast<int>(batch_shape.local_n_y), context->local_cells,
        impl_->lattice->inverse_delta_x(),
        impl_->lattice->inverse_delta_y(), zero_field,
        impl_->device_fields1, impl_->device_metrics);
    if (fail_cuda(cudaGetLastError(),
                  "gather_descriptor_kernel(gpu_mode=5 beam1)"))
      return false;
    gather_descriptor_kernel<<<deposit_grid2, kThreads, 0, impl_->stream>>>(
        impl_->device_gather2,
        impl_->options.maximum_beam2_descriptors, impl_->device_slots,
        context->phi1, 1, static_cast<int>(batch_shape.local_n_x),
        static_cast<int>(batch_shape.local_n_y), context->local_cells,
        impl_->lattice->inverse_delta_x(),
        impl_->lattice->inverse_delta_y(), zero_field,
        impl_->device_fields2, impl_->device_metrics);
    if (fail_cuda(cudaGetLastError(),
                  "gather_descriptor_kernel(gpu_mode=5 beam2)"))
      return false;
    launches += 2;

    if (fail_cuda(cudaEventRecord(impl_->events[2], impl_->stream),
                  "cudaEventRecord(gpu_mode=5 after kernels)") ||
        fail_cuda(cudaMemcpyAsync(impl_->host_fields1,
                                  impl_->device_fields1,
                                  active_field1_bytes,
                                  cudaMemcpyDeviceToHost, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam1 fields D2H)") ||
        fail_cuda(cudaMemcpyAsync(impl_->host_fields2,
                                  impl_->device_fields2,
                                  active_field2_bytes,
                                  cudaMemcpyDeviceToHost, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 beam2 fields D2H)") ||
        fail_cuda(cudaMemcpyAsync(impl_->host_metrics,
                                  impl_->device_metrics,
                                  active_metric_bytes,
                                  cudaMemcpyDeviceToHost, impl_->stream),
                  "cudaMemcpyAsync(gpu_mode=5 metrics D2H)") ||
        fail_cuda(cudaEventRecord(impl_->events[3], impl_->stream),
                  "cudaEventRecord(gpu_mode=5 after D2H)") ||
        fail_cuda(cudaEventSynchronize(impl_->events[3]),
                  "cudaEventSynchronize(gpu_mode=5 batch boundary)"))
      return false;
    work_enqueued = false;

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    if (!cuda_ok(cudaEventElapsedTime(&h2d_ms, impl_->events[0],
                                      impl_->events[1]),
                 "cudaEventElapsedTime(gpu_mode=5 H2D)", error) ||
        !cuda_ok(cudaEventElapsedTime(&kernel_ms, impl_->events[1],
                                      impl_->events[2]),
                 "cudaEventElapsedTime(gpu_mode=5 kernels)", error) ||
        !cuda_ok(cudaEventElapsedTime(&d2h_ms, impl_->events[2],
                                      impl_->events[3]),
                 "cudaEventElapsedTime(gpu_mode=5 D2H)", error)) {
      impl_->mark_poisoned();
      return false;
    }

    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      const UniformWindowPlan& plan = *submissions[slot].plan;
      const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
      const std::size_t count1 = plan.beam1_sources().size();
      const std::size_t count2 = plan.beam2_sources().size();
      if (metric.containment_failures != 0 ||
          metric.beam1_in != expected_deposit1_in[slot] ||
          metric.beam1_out != count1 - expected_deposit1_in[slot] ||
          metric.beam2_in != expected_deposit2_in[slot] ||
          metric.beam2_out != count2 - expected_deposit2_in[slot] ||
          metric.gather_beam1_in != expected_gather1_in[slot] ||
          metric.gather_beam1_out != count1 - expected_gather1_in[slot] ||
          metric.gather_beam2_in != expected_gather2_in[slot] ||
          metric.gather_beam2_out != count2 - expected_gather2_in[slot] ||
          !std::isfinite(metric.fine_lumi_dot)) {
        std::ostringstream message;
        message << "gpu_mode=5 staged metrics/containment failed at batch "
                   "slot "
                << slot;
        error = message.str();
        impl_->stats.containment_failures += metric.containment_failures;
        impl_->mark_poisoned();
        return false;
      }
      const float* fields1 =
          impl_->host_fields1 +
          2 * slot * impl_->options.maximum_beam1_descriptors;
      const float* fields2 =
          impl_->host_fields2 +
          2 * slot * impl_->options.maximum_beam2_descriptors;
      for (std::size_t value = 0; value < 2 * count1; ++value) {
        if (!std::isfinite(fields1[value])) {
          error = "gpu_mode=5 staged beam1 field is nonfinite";
          impl_->mark_poisoned();
          return false;
        }
      }
      for (std::size_t value = 0; value < 2 * count2; ++value) {
        if (!std::isfinite(fields2[value])) {
          error = "gpu_mode=5 staged beam2 field is nonfinite";
          impl_->mark_poisoned();
          return false;
        }
      }
    }

    const std::chrono::steady_clock::time_point publish_start =
        std::chrono::steady_clock::now();
    const std::uint64_t fft_hash = context->plan->key.stable_hash();
    const std::uint64_t green_hash = context->green->key.stable_hash();
    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      const UniformWindowPlan& plan = *submissions[slot].plan;
      const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
      AdaptiveEncounterResult& result = staged[slot];
      const std::size_t count1 = plan.beam1_sources().size();
      const std::size_t count2 = plan.beam2_sources().size();
      if (count1)
        std::memcpy(
            result.fields1.data(),
            impl_->host_fields1 +
                2 * slot * impl_->options.maximum_beam1_descriptors,
            2 * count1 * sizeof(float));
      if (count2)
        std::memcpy(
            result.fields2.data(),
            impl_->host_fields2 +
                2 * slot * impl_->options.maximum_beam2_descriptors,
            2 * count2 * sizeof(float));
      result.beam1_in = static_cast<std::size_t>(metric.beam1_in);
      result.beam1_out = static_cast<std::size_t>(metric.beam1_out);
      result.beam2_in = static_cast<std::size_t>(metric.beam2_in);
      result.beam2_out = static_cast<std::size_t>(metric.beam2_out);
      result.gather_beam1_in =
          static_cast<std::size_t>(metric.gather_beam1_in);
      result.gather_beam1_out =
          static_cast<std::size_t>(metric.gather_beam1_out);
      result.gather_beam2_in =
          static_cast<std::size_t>(metric.gather_beam2_in);
      result.gather_beam2_out =
          static_cast<std::size_t>(metric.gather_beam2_out);
      result.fine_lumi_dot = metric.fine_lumi_dot;
      result.binding =
          impl_->make_binding(submissions[slot], fft_hash, green_hash,
                              beam1_macro_charge, beam2_macro_charge);
      result.valid = true;
    }
    for (std::size_t slot = 0; slot < encounter_count; ++slot)
      canonical_results[submissions[slot].binding.canonical_result_slot]
          .swap(staged[slot]);
    impl_->stats.host_publish_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - publish_start)
            .count();

    ++impl_->stats.batch_calls;
    impl_->stats.encounters += encounter_count;
    impl_->stats.padded_fft_slots +=
        zero_field ? 0 : impl_->options.batch_capacity - encounter_count;
    if (!zero_field)
      impl_->stats.actual_planmany_cell_visits +=
          impl_->options.batch_capacity * context->padded_cells;
    impl_->stats.h2d_operations += 5;
    impl_->stats.d2h_operations += 3;
    impl_->stats.h2d_bytes +=
        active_deposit1_bytes + active_deposit2_bytes +
        active_gather1_bytes + active_gather2_bytes + active_slot_bytes;
    impl_->stats.d2h_bytes +=
        active_field1_bytes + active_field2_bytes + active_metric_bytes;
    impl_->stats.kernel_launches += launches;
    impl_->stats.cufft_executes += fft_executes;
    ++impl_->stats.synchronizations;
    impl_->stats.h2d_ms += h2d_ms;
    impl_->stats.kernel_ms += kernel_ms;
    impl_->stats.d2h_ms += d2h_ms;
    for (std::size_t slot = 0; slot < encounter_count; ++slot) {
      const DeviceEncounterMetrics& metric = impl_->host_metrics[slot];
      const UniformWindowPlan& plan = *submissions[slot].plan;
      impl_->stats.deposit_beam1_in += metric.beam1_in;
      impl_->stats.deposit_beam1_out += metric.beam1_out;
      impl_->stats.deposit_beam2_in += metric.beam2_in;
      impl_->stats.deposit_beam2_out += metric.beam2_out;
      impl_->stats.gather_beam1_in += metric.gather_beam1_in;
      impl_->stats.gather_beam1_out += metric.gather_beam1_out;
      impl_->stats.gather_beam2_in += metric.gather_beam2_in;
      impl_->stats.gather_beam2_out += metric.gather_beam2_out;
      impl_->stats.required_node_visits += plan.required_node_count();
      impl_->stats.selected_local_cell_visits += context->local_cells;
      impl_->stats.padded_fft_cell_visits += context->padded_cells;
      impl_->stats.canonical_cells_avoided +=
          impl_->lattice->n_x() * impl_->lattice->n_y() -
          context->local_cells;
      if (plan.full_window())
        ++impl_->stats.full_window_encounters;
    }
    return true;
  } catch (const std::exception& exception) {
    error = std::string("gpu_mode=5 execute failed before publication: ") +
            exception.what();
    return false;
  } catch (...) {
    error = "gpu_mode=5 execute failed before publication with an unknown "
            "host exception";
    return false;
  }
}

bool AdaptiveWaveBatchExecutor::validate_result_binding(
    const AdaptiveEncounterResult& result,
    const AdaptiveEncounterSubmission& expected,
    double expected_beam1_macro_charge,
    double expected_beam2_macro_charge, std::string& error) const {
  error.clear();
  const auto fail = [&](const std::string& message) {
    error = message;
    ++const_cast<Impl*>(impl_)->stats.binding_failures;
    return false;
  };
  if (!impl_->initialized || impl_->poisoned)
    return fail(impl_->poisoned ? "gpu_mode=5 context is poisoned"
                                : "gpu_mode=5 executor is not initialized");
  if (!result.valid || !expected.plan)
    return fail("gpu_mode=5 result or expected immutable plan is invalid");
  if (!std::isfinite(expected_beam1_macro_charge) ||
      !std::isfinite(expected_beam2_macro_charge))
    return fail("gpu_mode=5 expected macro charges are nonfinite");
  if (expected.plan->lattice_hash() != impl_->lattice->hash())
    return fail("gpu_mode=5 expected plan belongs to another lattice");
  std::uint64_t fft_hash = 0;
  std::uint64_t green_hash = 0;
  if (!expected.plan->empty()) {
    const WindowClassShape shape{expected.plan->local_n_x(),
                                 expected.plan->local_n_y()};
    auto found = impl_->class_contexts.find(shape);
    if (found == impl_->class_contexts.end())
      return fail("gpu_mode=5 expected result class is not prebuilt");
    fft_hash = found->second->plan->key.stable_hash();
    green_hash = found->second->green->key.stable_hash();
  }
  const AdaptiveResultBinding wanted =
      impl_->make_binding(expected, fft_hash, green_hash,
                          expected_beam1_macro_charge,
                          expected_beam2_macro_charge);
  const AdaptiveResultBinding& actual = result.binding;
  const AdaptiveEncounterBinding& a = actual.encounter;
  const AdaptiveEncounterBinding& w = wanted.encounter;
  const bool equal =
      a.canonical_result_slot == w.canonical_result_slot &&
      a.encounter_id == w.encounter_id && a.beam1_slice == w.beam1_slice &&
      a.beam2_slice == w.beam2_slice &&
      a.beam1_revision == w.beam1_revision &&
      a.beam2_revision == w.beam2_revision &&
      a.beam1_order_token == w.beam1_order_token &&
      a.beam2_order_token == w.beam2_order_token &&
      actual.lattice_hash == wanted.lattice_hash &&
      actual.geometry_revision == wanted.geometry_revision &&
      actual.window_plan_hash == wanted.window_plan_hash &&
      actual.origin_x == wanted.origin_x && actual.origin_y == wanted.origin_y &&
      actual.local_n_x == wanted.local_n_x &&
      actual.local_n_y == wanted.local_n_y &&
      actual.padded_n_x == wanted.padded_n_x &&
      actual.padded_n_y == wanted.padded_n_y &&
      actual.beam1_particle_count == wanted.beam1_particle_count &&
      actual.beam2_particle_count == wanted.beam2_particle_count &&
      actual.beam1_macro_charge_bits == wanted.beam1_macro_charge_bits &&
      actual.beam2_macro_charge_bits == wanted.beam2_macro_charge_bits &&
      actual.fft_plan_key_hash == wanted.fft_plan_key_hash &&
      actual.green_spectrum_key_hash == wanted.green_spectrum_key_hash &&
      actual.binding_hash == wanted.binding_hash;
  if (!equal)
    return fail("gpu_mode=5 result binding is stale or belongs to another "
                "encounter/order/revision/plan");
  if (result.fields1.size() != 2 * wanted.beam1_particle_count ||
      result.fields2.size() != 2 * wanted.beam2_particle_count)
    return fail("gpu_mode=5 result field sizes violate the bound particle "
                "counts");
  if (result.beam1_in + result.beam1_out !=
          wanted.beam1_particle_count ||
      result.beam2_in + result.beam2_out !=
          wanted.beam2_particle_count ||
      result.gather_beam1_in + result.gather_beam1_out !=
          wanted.beam1_particle_count ||
      result.gather_beam2_in + result.gather_beam2_out !=
          wanted.beam2_particle_count ||
      !std::isfinite(result.fine_lumi_dot))
    return fail("gpu_mode=5 result counters or fine-luminosity value are "
                "invalid");
  for (float value : result.fields1)
    if (!std::isfinite(value))
      return fail("gpu_mode=5 bound beam1 result contains a nonfinite field");
  for (float value : result.fields2)
    if (!std::isfinite(value))
      return fail("gpu_mode=5 bound beam2 result contains a nonfinite field");
  return true;
}

bool AdaptiveWaveBatchExecutor::fft_plan_key(
    const WindowClassShape& shape, FftPlanCacheKey& key,
    std::string& error) const {
  error.clear();
  return impl_->make_fft_key(shape, key, error);
}

bool AdaptiveWaveBatchExecutor::green_spectrum_key(
    const WindowClassShape& shape, GreenSpectrumCacheKey& key,
    std::string& error) const {
  error.clear();
  return impl_->make_green_key(shape, key, error);
}

bool AdaptiveWaveBatchExecutor::initialized() const noexcept {
  return impl_->initialized;
}

bool AdaptiveWaveBatchExecutor::prebuild_complete() const noexcept {
  return impl_->stats.prebuild_complete;
}

bool AdaptiveWaveBatchExecutor::poisoned() const noexcept {
  return impl_->poisoned;
}

DeviceIdentity AdaptiveWaveBatchExecutor::device_identity() const {
  return impl_->identity;
}

AdaptiveGpuTelemetry AdaptiveWaveBatchExecutor::telemetry() const {
  return impl_->stats;
}

std::string AdaptiveWaveBatchExecutor::device_description() const {
  if (!impl_->initialized)
    return impl_->poisoned ? "CUDA mode-5 initialization failed [POISONED]"
                           : "CUDA mode-5 not initialized";
  return impl_->poisoned ? impl_->device_text + " [POISONED]"
                         : impl_->device_text;
}

}  // namespace gpu5
}  // namespace adaptive_grid
}  // namespace feebl
