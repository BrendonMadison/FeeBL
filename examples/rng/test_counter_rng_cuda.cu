#include "counterRngCPP.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

__global__ void generate(const feebl::RandomKey key, std::uint32_t* output,
                         std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = feebl::CounterRng::uint32(key, i);
}

int main() {
  constexpr std::size_t count = 4096;
  feebl::RandomKey key;
  key.run_seed = 20260825;
  key.domain = feebl::RandomDomain::Diagnostics;
  key.entity = 123456789;

  std::uint32_t* device = nullptr;
  if (cudaMalloc(&device, count * sizeof(*device)) != cudaSuccess) return 2;
  generate<<<(count + 255) / 256, 256>>>(key, device, count);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;

  std::vector<std::uint32_t> host(count);
  if (cudaMemcpy(host.data(), device, count * sizeof(*device),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return 4;
  cudaFree(device);

  for (std::size_t i = 0; i < count; ++i) {
    if (host[i] != feebl::CounterRng::uint32(key, i)) {
      std::cerr << "CPU/CUDA mismatch at draw " << i << '\n';
      return 1;
    }
  }
  std::cout << "counter RNG CUDA parity: PASS (" << count << " draws)\n";
  return 0;
}
