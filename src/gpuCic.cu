#include "gpuCic.h"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>

namespace feebl {
namespace gpu {

namespace {

const int kThreads = 256;
const int kSharedSlots = 512;

__device__ bool locate_particle(float x,
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
  // Keep FeeBL's operation order and single-precision localization.  Charge
  // weights are evaluated in double below, as in assignChargeToCIC.
  // GCC's -ffast-math build of GRID::particleInGrid hoists
  // (offset - 0.5f) and then performs one add after the multiplication.  Spell
  // that grouping explicitly so points within one float ULP of a cell boundary
  // select the same cell on CPU and GPU.
  hx = x * delta_x_inverse + (offset_x - 0.5f);
  ix = static_cast<int>(floorf(hx));
  hx -= static_cast<float>(ix);
  hy = y * delta_y_inverse + (offset_y - 0.5f);
  iy = static_cast<int>(floorf(hy));
  hy -= static_cast<float>(iy);
  return ix >= 0 && ix < nx - 1 && iy >= 0 && iy < ny - 1;
}

__device__ void cic_indices_values(int ix,
                                   int iy,
                                   float hx,
                                   float hy,
                                   int ny,
                                   double charge,
                                   int* keys,
                                   double* values)
{
  const double dx = static_cast<double>(hx);
  const double dy = static_cast<double>(hy);
  keys[0] = ix * ny + iy;
  keys[1] = (ix + 1) * ny + iy;
  keys[2] = ix * ny + iy + 1;
  keys[3] = (ix + 1) * ny + iy + 1;
  values[0] = (1.0 - dx) * (1.0 - dy) * charge;
  values[1] = dx * (1.0 - dy) * charge;
  values[2] = (1.0 - dx) * dy * charge;
  values[3] = dx * dy * charge;
}

__global__ void cic_atomic_kernel(const float* xy,
                                  std::size_t particle_count,
                                  double charge,
                                  int nx,
                                  int ny,
                                  float delta_x_inverse,
                                  float delta_y_inverse,
                                  float offset_x,
                                  float offset_y,
                                  double* rho,
                                  unsigned long long* counts)
{
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      int ix, iy;
      float hx, hy;
      if (!locate_particle(xy[2 * particle], xy[2 * particle + 1],
                           delta_x_inverse, delta_y_inverse,
                           offset_x, offset_y, nx, ny, ix, iy, hx, hy))
        {
          atomicAdd(counts + 1, 1ULL);
          continue;
        }

      atomicAdd(counts, 1ULL);
      int keys[4];
      double values[4];
      cic_indices_values(ix, iy, hx, hy, ny, charge, keys, values);
      for (int contribution = 0; contribution < 4; ++contribution)
        atomicAdd(rho + keys[contribution], values[contribution]);
    }
}

__device__ void shared_hash_add(int key,
                                double value,
                                int* shared_keys,
                                double* shared_values,
                                double* rho)
{
  const unsigned int first =
      (static_cast<unsigned int>(key) * 2654435761u) & (kSharedSlots - 1);
  for (int probe = 0; probe < kSharedSlots; ++probe)
    {
      const int slot = static_cast<int>((first + probe) & (kSharedSlots - 1));
      const int previous = atomicCAS(shared_keys + slot, -1, key);
      if (previous == -1 || previous == key)
        {
          atomicAdd(shared_values + slot, value);
          return;
        }
    }

  // A block spanning a very broad distribution can exhaust the small shared
  // table.  Falling back to a global atomic preserves correctness rather than
  // dropping charge or requiring workload-specific table sizing.
  atomicAdd(rho + key, value);
}

__global__ void cic_shared_kernel(const float* xy,
                                  std::size_t particle_count,
                                  double charge,
                                  int nx,
                                  int ny,
                                  float delta_x_inverse,
                                  float delta_y_inverse,
                                  float offset_x,
                                  float offset_y,
                                  double* rho,
                                  unsigned long long* counts)
{
  __shared__ int shared_keys[kSharedSlots];
  __shared__ double shared_values[kSharedSlots];

  for (int slot = threadIdx.x; slot < kSharedSlots; slot += blockDim.x)
    {
      shared_keys[slot] = -1;
      shared_values[slot] = 0.0;
    }
  __syncthreads();

  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      int ix, iy;
      float hx, hy;
      if (!locate_particle(xy[2 * particle], xy[2 * particle + 1],
                           delta_x_inverse, delta_y_inverse,
                           offset_x, offset_y, nx, ny, ix, iy, hx, hy))
        {
          atomicAdd(counts + 1, 1ULL);
          continue;
        }

      atomicAdd(counts, 1ULL);
      int keys[4];
      double values[4];
      cic_indices_values(ix, iy, hx, hy, ny, charge, keys, values);
      for (int contribution = 0; contribution < 4; ++contribution)
        shared_hash_add(keys[contribution], values[contribution],
                        shared_keys, shared_values, rho);
    }
  __syncthreads();

  for (int slot = threadIdx.x; slot < kSharedSlots; slot += blockDim.x)
    if (shared_keys[slot] >= 0)
      atomicAdd(rho + shared_keys[slot], shared_values[slot]);
}

__global__ void cic_contributions_kernel(const float* xy,
                                         std::size_t particle_count,
                                         double charge,
                                         int nx,
                                         int ny,
                                         float delta_x_inverse,
                                         float delta_y_inverse,
                                         float offset_x,
                                         float offset_y,
                                         int* keys,
                                         double* values,
                                         unsigned long long* counts)
{
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t particle =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       particle < particle_count;
       particle += stride)
    {
      const std::size_t base = 4 * particle;
      int ix, iy;
      float hx, hy;
      if (!locate_particle(xy[2 * particle], xy[2 * particle + 1],
                           delta_x_inverse, delta_y_inverse,
                           offset_x, offset_y, nx, ny, ix, iy, hx, hy))
        {
          atomicAdd(counts + 1, 1ULL);
          for (int contribution = 0; contribution < 4; ++contribution)
            {
              keys[base + contribution] = -1;
              values[base + contribution] = 0.0;
            }
          continue;
        }

      atomicAdd(counts, 1ULL);
      int local_keys[4];
      double local_values[4];
      cic_indices_values(ix, iy, hx, hy, ny, charge,
                         local_keys, local_values);
      for (int contribution = 0; contribution < 4; ++contribution)
        {
          keys[base + contribution] = local_keys[contribution];
          values[base + contribution] = local_values[contribution];
        }
    }
}

__global__ void cic_scatter_kernel(const int* keys,
                                   const double* values,
                                   std::size_t unique_count,
                                   double* rho)
{
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < unique_count;
       index += stride)
    if (keys[index] >= 0)
      rho[keys[index]] = values[index];
}

int block_count(std::size_t work)
{
  if (work == 0) return 1;
  const std::size_t needed = (work + kThreads - 1) / kThreads;
  return static_cast<int>(std::min<std::size_t>(needed, 65535));
}

bool cuda_ok(cudaError_t status, const char* operation, std::string& error)
{
  if (status == cudaSuccess) return true;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status);
  error = message.str();
  return false;
}

double elapsed_ms(cudaEvent_t begin, cudaEvent_t end)
{
  float value = 0.0f;
  cudaEventElapsedTime(&value, begin, end);
  return static_cast<double>(value);
}

}  // namespace

CicStats::CicStats()
    : particles_in(0),
      particles_out(0),
      setup_ms(0.0),
      h2d_ms(0.0),
      kernel_ms(0.0),
      d2h_ms(0.0),
      total_ms(0.0)
{
}

const char* cic_variant_name(CicVariant variant)
{
  switch (variant)
    {
    case CicVariant::Atomic: return "atomic";
    case CicVariant::Shared: return "shared";
    case CicVariant::Sorted: return "sorted";
    }
  return "unknown";
}

bool parse_cic_variant(const std::string& text, CicVariant& variant)
{
  if (text == "atomic")
    {
      variant = CicVariant::Atomic;
      return true;
    }
  if (text == "shared" || text == "tiled")
    {
      variant = CicVariant::Shared;
      return true;
    }
  if (text == "sorted" || text == "binning")
    {
      variant = CicVariant::Sorted;
      return true;
    }
  return false;
}

struct CicDepositor::Impl
{
  int device;
  bool initialized;
  std::string description;
  cudaStream_t stream;
  cudaEvent_t events[4];
  float* xy;
  double* rho;
  unsigned long long* counts;
  int* keys;
  double* values;
  int* unique_keys;
  double* unique_values;
  std::size_t particle_capacity;
  std::size_t cell_capacity;
  std::size_t contribution_capacity;

  Impl()
      : device(0),
        initialized(false),
        stream(0),
        xy(0),
        rho(0),
        counts(0),
        keys(0),
        values(0),
        unique_keys(0),
        unique_values(0),
        particle_capacity(0),
        cell_capacity(0),
        contribution_capacity(0)
  {
    for (int index = 0; index < 4; ++index) events[index] = 0;
  }

  void release()
  {
    if (initialized) cudaSetDevice(device);
    cudaFree(xy);
    cudaFree(rho);
    cudaFree(counts);
    cudaFree(keys);
    cudaFree(values);
    cudaFree(unique_keys);
    cudaFree(unique_values);
    for (int index = 0; index < 4; ++index)
      {
        if (events[index]) cudaEventDestroy(events[index]);
        events[index] = 0;
      }
    if (stream) cudaStreamDestroy(stream);
    xy = 0;
    rho = 0;
    counts = 0;
    keys = 0;
    values = 0;
    unique_keys = 0;
    unique_values = 0;
    stream = 0;
    particle_capacity = 0;
    cell_capacity = 0;
    contribution_capacity = 0;
    description.clear();
    initialized = false;
  }

  bool ensure_capacity(std::size_t particle_count,
                       std::size_t cells,
                       bool need_sorted,
                       std::string& error)
  {
    if (particle_count > particle_capacity)
      {
        if (particle_count > std::numeric_limits<std::size_t>::max() /
                                 (2 * sizeof(float)))
          {
            error = "particle count overflows the CUDA xy allocation";
            return false;
          }
        float* replacement = 0;
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&replacement),
                                2 * particle_count * sizeof(float)),
                     "cudaMalloc(xy)", error))
          return false;
        cudaFree(xy);
        xy = replacement;
        particle_capacity = particle_count;
      }

    if (cells > cell_capacity)
      {
        if (cells > std::numeric_limits<std::size_t>::max() / sizeof(double))
          {
            error = "grid size overflows the CUDA rho allocation";
            return false;
          }
        double* replacement = 0;
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&replacement),
                                cells * sizeof(double)),
                     "cudaMalloc(rho)", error))
          return false;
        cudaFree(rho);
        rho = replacement;
        cell_capacity = cells;
      }

    if (!counts &&
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&counts),
                            2 * sizeof(unsigned long long)),
                 "cudaMalloc(counts)", error))
      return false;

    if (need_sorted)
      {
        if (particle_count > std::numeric_limits<std::size_t>::max() / 4)
          {
            error = "particle count overflows the sorted CIC contribution array";
            return false;
          }
        const std::size_t contributions = 4 * particle_count;
        if (contributions > contribution_capacity)
          {
            if (contributions > std::numeric_limits<std::size_t>::max() /
                                    sizeof(double))
              {
                error = "sorted CIC allocation byte count overflows";
                return false;
              }
            const std::size_t key_bytes = contributions * sizeof(int);
            const std::size_t value_bytes = contributions * sizeof(double);
            int* replacement_keys = 0;
            double* replacement_values = 0;
            int* replacement_unique_keys = 0;
            double* replacement_unique_values = 0;
            if (!cuda_ok(cudaMalloc(
                             reinterpret_cast<void**>(&replacement_keys),
                             key_bytes),
                         "cudaMalloc(sorted keys)", error) ||
                !cuda_ok(cudaMalloc(
                             reinterpret_cast<void**>(&replacement_values),
                             value_bytes),
                         "cudaMalloc(sorted values)", error) ||
                !cuda_ok(cudaMalloc(
                             reinterpret_cast<void**>(&replacement_unique_keys),
                             key_bytes),
                         "cudaMalloc(unique keys)", error) ||
                !cuda_ok(cudaMalloc(
                             reinterpret_cast<void**>(&replacement_unique_values),
                             value_bytes),
                         "cudaMalloc(unique values)", error))
              {
                cudaFree(replacement_keys);
                cudaFree(replacement_values);
                cudaFree(replacement_unique_keys);
                cudaFree(replacement_unique_values);
                return false;
              }
            cudaFree(keys);
            cudaFree(values);
            cudaFree(unique_keys);
            cudaFree(unique_values);
            keys = replacement_keys;
            values = replacement_values;
            unique_keys = replacement_unique_keys;
            unique_values = replacement_unique_values;
            contribution_capacity = contributions;
          }
      }
    return true;
  }
};

CicDepositor::CicDepositor() : impl_(new Impl)
{
}

CicDepositor::~CicDepositor()
{
  impl_->release();
  delete impl_;
}

bool CicDepositor::initialize(int device, std::string& error)
{
  if (impl_->initialized)
    {
      if (device == impl_->device) return true;
      error = "a CicDepositor cannot change CUDA devices after initialization";
      return false;
    }

  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error))
    return false;
  if (device < 0 || device >= device_count)
    {
      std::ostringstream message;
      message << "CUDA device " << device << " is unavailable; found "
              << device_count << " device(s)";
      error = message.str();
      return false;
    }
  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice", error)) return false;

  cudaDeviceProp properties;
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties", error))
    return false;
  if (properties.major < 6)
    {
      std::ostringstream message;
      message << "CUDA CIC requires native FP64 atomicAdd (compute capability "
              << "6.0 or newer), but device " << device << " is "
              << properties.major << '.' << properties.minor;
      error = message.str();
      return false;
    }

  if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags", error))
    return false;
  for (int index = 0; index < 4; ++index)
    if (!cuda_ok(cudaEventCreate(&impl_->events[index]),
                 "cudaEventCreate", error))
      {
        impl_->release();
        return false;
      }

  impl_->device = device;
  impl_->initialized = true;
  std::ostringstream description;
  description << properties.name << " (device " << device << ", compute "
              << properties.major << '.' << properties.minor << ", "
              << properties.totalGlobalMem / (1024 * 1024) << " MiB)";
  impl_->description = description.str();
  return true;
}

bool CicDepositor::is_available() const
{
  return impl_->initialized;
}

std::string CicDepositor::device_description() const
{
  return impl_->initialized ? impl_->description : "CUDA not initialized";
}

bool CicDepositor::deposit(const float* xy,
                           std::size_t particle_count,
                           double charge,
                           int nx,
                           int ny,
                           float delta_x_inverse,
                           float delta_y_inverse,
                           float offset_x,
                           float offset_y,
                           CicVariant variant,
                           double* rho,
                           CicStats& stats,
                           std::string& error)
{
  stats = CicStats();
  if (!impl_->initialized)
    {
      error = "CUDA CIC depositor is not initialized";
      return false;
    }
  if (nx < 2 || ny < 2 || !rho || (particle_count && !xy))
    {
      error = "invalid CUDA CIC grid or particle buffer";
      return false;
    }
  if (static_cast<std::size_t>(nx) >
      std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(ny))
    {
      error = "CUDA CIC grid dimensions overflow size_t";
      return false;
    }

  const std::chrono::steady_clock::time_point total_begin =
      std::chrono::steady_clock::now();
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  if (!impl_->ensure_capacity(particle_count, cells,
                              variant == CicVariant::Sorted, error))
    return false;

  if (!cuda_ok(cudaEventRecord(impl_->events[0], impl_->stream),
               "cudaEventRecord(start)", error))
    return false;
  if (particle_count &&
      !cuda_ok(cudaMemcpyAsync(impl_->xy, xy,
                               2 * particle_count * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync(xy H2D)", error))
    return false;
  if (!cuda_ok(cudaEventRecord(impl_->events[1], impl_->stream),
               "cudaEventRecord(after H2D)", error))
    return false;

  if (!cuda_ok(cudaMemsetAsync(impl_->rho, 0, cells * sizeof(double),
                               impl_->stream),
               "cudaMemsetAsync(rho)", error) ||
      !cuda_ok(cudaMemsetAsync(impl_->counts, 0,
                               2 * sizeof(unsigned long long), impl_->stream),
               "cudaMemsetAsync(counts)", error))
    return false;

  const int blocks = block_count(particle_count);
  if (particle_count)
    {
      if (variant == CicVariant::Atomic)
        {
          cic_atomic_kernel<<<blocks, kThreads, 0, impl_->stream>>>(
              impl_->xy, particle_count, charge, nx, ny,
              delta_x_inverse, delta_y_inverse, offset_x, offset_y,
              impl_->rho, impl_->counts);
        }
      else if (variant == CicVariant::Shared)
        {
          cic_shared_kernel<<<blocks, kThreads, 0, impl_->stream>>>(
              impl_->xy, particle_count, charge, nx, ny,
              delta_x_inverse, delta_y_inverse, offset_x, offset_y,
              impl_->rho, impl_->counts);
        }
      else
        {
          const std::size_t contributions = 4 * particle_count;
          cic_contributions_kernel<<<blocks, kThreads, 0, impl_->stream>>>(
              impl_->xy, particle_count, charge, nx, ny,
              delta_x_inverse, delta_y_inverse, offset_x, offset_y,
              impl_->keys, impl_->values, impl_->counts);
          if (!cuda_ok(cudaGetLastError(),
                       "cic_contributions_kernel launch", error))
            return false;

          thrust::device_ptr<int> key_begin(impl_->keys);
          thrust::device_ptr<double> value_begin(impl_->values);
          thrust::stable_sort_by_key(thrust::cuda::par.on(impl_->stream),
                                     key_begin, key_begin + contributions,
                                     value_begin);
          thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<double> > end =
              thrust::reduce_by_key(
                  thrust::cuda::par.on(impl_->stream),
                  key_begin, key_begin + contributions, value_begin,
                  thrust::device_pointer_cast(impl_->unique_keys),
                  thrust::device_pointer_cast(impl_->unique_values));
          const std::size_t unique_count =
              static_cast<std::size_t>(
                  end.first - thrust::device_pointer_cast(impl_->unique_keys));
          cic_scatter_kernel<<<block_count(unique_count), kThreads, 0,
                               impl_->stream>>>(impl_->unique_keys,
                                               impl_->unique_values,
                                               unique_count, impl_->rho);
        }
      if (!cuda_ok(cudaGetLastError(), "CUDA CIC kernel launch", error))
        return false;
    }

  if (!cuda_ok(cudaEventRecord(impl_->events[2], impl_->stream),
               "cudaEventRecord(after kernel)", error))
    return false;

  unsigned long long host_counts[2] = {0, 0};
  if (!cuda_ok(cudaMemcpyAsync(rho, impl_->rho, cells * sizeof(double),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(rho D2H)", error) ||
      !cuda_ok(cudaMemcpyAsync(host_counts, impl_->counts,
                               2 * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync(counts D2H)", error) ||
      !cuda_ok(cudaEventRecord(impl_->events[3], impl_->stream),
               "cudaEventRecord(after D2H)", error) ||
      !cuda_ok(cudaEventSynchronize(impl_->events[3]),
               "cudaEventSynchronize", error))
    return false;

  stats.particles_in = static_cast<std::size_t>(host_counts[0]);
  stats.particles_out = static_cast<std::size_t>(host_counts[1]);
  stats.h2d_ms = elapsed_ms(impl_->events[0], impl_->events[1]);
  stats.kernel_ms = elapsed_ms(impl_->events[1], impl_->events[2]);
  stats.d2h_ms = elapsed_ms(impl_->events[2], impl_->events[3]);
  stats.total_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - total_begin).count();
  stats.setup_ms = std::max(0.0, stats.total_ms - stats.h2d_ms -
                                  stats.kernel_ms - stats.d2h_ms);
  return true;
}

}  // namespace gpu
}  // namespace feebl
