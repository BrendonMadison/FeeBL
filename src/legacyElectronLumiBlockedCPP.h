#ifndef FEEBL_LEGACY_ELECTRON_LUMI_BLOCKED_CPP_H
#define FEEBL_LEGACY_ELECTRON_LUMI_BLOCKED_CPP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include "electronLumiSamplerCPP.h"

namespace feebl {
namespace legacy_electron_lumi_blocked {

// One block keeps the beam-2 inputs and all pair outputs comfortably inside a
// typical L1 data cache.  A power of two also means that every production
// block starts on the same 64-byte alignment as the underlying SoA arrays.
constexpr std::size_t kBlockSize = 128U;
constexpr std::size_t kScratchAlignment = 64U;

template <typename T, std::size_t Alignment = kScratchAlignment>
class AlignedAllocator {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t count) {
    static_assert(Alignment >= alignof(T),
                  "scratch alignment must satisfy the element alignment");
    static_assert((Alignment & (Alignment - 1U)) == 0U,
                  "scratch alignment must be a power of two");
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    return static_cast<T*>(
        ::operator new(count * sizeof(T), std::align_val_t(Alignment)));
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t(Alignment));
  }
};

template <typename T, typename U, std::size_t Alignment>
inline bool operator==(const AlignedAllocator<T, Alignment>&,
                       const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
inline bool operator!=(const AlignedAllocator<T, Alignment>&,
                       const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

template <typename T>
using AlignedVector =
    std::vector<T, AlignedAllocator<T, kScratchAlignment> >;

// Materialized particle fields are deliberately SoA.  GRID fills these once
// per occupied cell after reserving every component to the largest cell in the
// encounter.  Consequently append() performs no allocation during cell or
// pair processing.
struct BeamSoA {
  AlignedVector<float> signed_energy;
  AlignedVector<float> absolute_energy;
  AlignedVector<float> vx;
  AlignedVector<float> vy;
  AlignedVector<float> vz;
  AlignedVector<float> weight;
  AlignedVector<double> massive_momentum;
  AlignedVector<std::uint8_t> charge_class;
  AlignedVector<std::uint64_t> stable_id;

  void reserve(std::size_t particles);
  void clear() noexcept;
  void append(float signed_energy_value, float absolute_energy_value,
              float vx_value, float vy_value, float vz_value,
              float weight_value, double massive_momentum_value,
              std::uint8_t charge_class_value,
              std::uint64_t stable_id_value =
                  std::numeric_limits<std::uint64_t>::max());
  std::size_t size() const noexcept { return signed_energy.size(); }
  bool component_sizes_match() const noexcept;
  bool capacities_cover(std::size_t particles) const noexcept;
};

// Pair data are computed without touching RESULTS.  The subsequent scalar
// commit loop consumes indices [0,count) in historical order.
struct alignas(kScratchAlignment) BlockKinematics {
  alignas(kScratchAlignment) std::array<float, kBlockSize> weight;
  alignas(kScratchAlignment) std::array<float, kBlockSize> ecm;
  alignas(kScratchAlignment) std::array<double, kBlockSize> invariant_mass;
  alignas(kScratchAlignment) std::array<std::uint32_t, kBlockSize> same_sign;
  alignas(kScratchAlignment) std::array<std::uint32_t, kBlockSize> above_threshold;
};

struct Workspace {
  BeamSoA beam1;
  BeamSoA beam2;
  BlockKinematics block;
  std::vector<BlockKinematics> parallel_blocks;
  std::vector<feebl::electron_lumi_sampling::WeightedReservoir>
      parallel_samplers;
  feebl::electron_lumi_sampling::WeightedReservoir merged_sampler;
  std::vector<feebl::electron_lumi_sampling::KeyedBernoulliSampler>
      parallel_bernoulli_samplers;
  feebl::electron_lumi_sampling::KeyedBernoulliSampler
      merged_bernoulli_sampler;
  // Reused row-local output from the geometric-gap sampler.  Keeping this
  // buffer in the persistent workspace avoids an allocation for every
  // beam-1 particle while retaining a transactional capacity preflight.
  std::vector<std::uint64_t> geometric_success_ordinals;
  std::size_t parallel_lanes = 0U;
  std::size_t blocks_per_lane = 0U;
  std::size_t sampler_capacity = 0U;
  int sampler_mode = 0;

  // This is the only production allocation point for the workspace.  GRID
  // invokes it during preflight and before changing any RESULTS accumulator.
  void reserve(std::size_t beam1_particles, std::size_t beam2_particles);
  void reserve_parallel(std::size_t lanes, std::size_t beam2_particles,
                        std::size_t sampling_capacity,
                        int sampling_mode = 1);
  bool parallel_capacity_covers(std::size_t lanes,
                                std::size_t beam2_particles,
                                std::size_t sampling_capacity,
                                int sampling_mode = 1) const noexcept;
  BlockKinematics& parallel_block(std::size_t lane,
                                  std::size_t block_index) noexcept;
};

// Evaluate one beam-1 particle against a fixed-size, aligned beam-2 block.
// The function is out of line so compiler vectorization can be inspected in
// the shipped binary.  It is pure, has no RNG or global state, and returns
// false without writing RESULTS if the caller violates the block contract.
bool evaluate_block(const BeamSoA& beam1, std::size_t beam1_index,
                    const BeamSoA& beam2, std::size_t beam2_begin,
                    std::size_t count, float ecm_min,
                    double electron_mass,
                    BlockKinematics& output) noexcept;

}  // namespace legacy_electron_lumi_blocked
}  // namespace feebl

#endif  // FEEBL_LEGACY_ELECTRON_LUMI_BLOCKED_CPP_H
