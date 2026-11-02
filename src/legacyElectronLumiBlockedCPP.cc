#include "legacyElectronLumiBlockedCPP.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace feebl {
namespace legacy_electron_lumi_blocked {

#if defined(__GNUC__) || defined(__clang__)
#define FEEBL_LUMI_RESTRICT __restrict__
#else
#define FEEBL_LUMI_RESTRICT
#endif

void BeamSoA::reserve(std::size_t particles) {
  signed_energy.reserve(particles);
  absolute_energy.reserve(particles);
  vx.reserve(particles);
  vy.reserve(particles);
  vz.reserve(particles);
  weight.reserve(particles);
  massive_momentum.reserve(particles);
  charge_class.reserve(particles);
  stable_id.reserve(particles);
}

void BeamSoA::clear() noexcept {
  signed_energy.clear();
  absolute_energy.clear();
  vx.clear();
  vy.clear();
  vz.clear();
  weight.clear();
  massive_momentum.clear();
  charge_class.clear();
  stable_id.clear();
}

void BeamSoA::append(float signed_energy_value,
                     float absolute_energy_value,
                     float vx_value, float vy_value, float vz_value,
                     float weight_value, double massive_momentum_value,
                     std::uint8_t charge_class_value,
                     std::uint64_t stable_id_value) {
  signed_energy.push_back(signed_energy_value);
  absolute_energy.push_back(absolute_energy_value);
  vx.push_back(vx_value);
  vy.push_back(vy_value);
  vz.push_back(vz_value);
  weight.push_back(weight_value);
  massive_momentum.push_back(massive_momentum_value);
  charge_class.push_back(charge_class_value);
  stable_id.push_back(stable_id_value);
}

bool BeamSoA::component_sizes_match() const noexcept {
  const std::size_t particles = signed_energy.size();
  return absolute_energy.size() == particles && vx.size() == particles &&
         vy.size() == particles && vz.size() == particles &&
         weight.size() == particles &&
         massive_momentum.size() == particles &&
         charge_class.size() == particles && stable_id.size() == particles;
}

bool BeamSoA::capacities_cover(std::size_t particles) const noexcept {
  return signed_energy.capacity() >= particles &&
         absolute_energy.capacity() >= particles &&
         vx.capacity() >= particles && vy.capacity() >= particles &&
         vz.capacity() >= particles && weight.capacity() >= particles &&
         massive_momentum.capacity() >= particles &&
         charge_class.capacity() >= particles &&
         stable_id.capacity() >= particles;
}

void Workspace::reserve(std::size_t beam1_particles,
                        std::size_t beam2_particles) {
  beam1.reserve(beam1_particles);
  beam2.reserve(beam2_particles);
  geometric_success_ordinals.reserve(beam2_particles);
}

void Workspace::reserve_parallel(std::size_t lanes,
                                 std::size_t beam2_particles,
                                 std::size_t sampling_capacity_value,
                                 int sampling_mode_value) {
  const std::size_t blocks =
      (beam2_particles + kBlockSize - 1U) / kBlockSize;
  if (lanes != 0U && blocks >
          std::numeric_limits<std::size_t>::max() / lanes)
    throw std::length_error("electron-luminosity parallel scratch overflow");
  const std::size_t required = lanes * blocks;
  if (parallel_blocks.size() < required) parallel_blocks.resize(required);
  if (sampling_mode_value == 1 && parallel_samplers.size() < lanes) {
    parallel_samplers.reserve(lanes);
    while (parallel_samplers.size() < lanes)
      parallel_samplers.push_back(
          feebl::electron_lumi_sampling::WeightedReservoir(0U));
  }
  const std::size_t lane_sampling_capacity =
      std::min(sampling_capacity_value, beam2_particles);
  if (sampling_mode_value == 1)
    for (std::size_t lane = 0; lane < lanes; ++lane)
      parallel_samplers[lane].reset(lane_sampling_capacity);
  if (sampling_mode_value == 2) {
    if (parallel_bernoulli_samplers.size() < lanes) {
      parallel_bernoulli_samplers.reserve(lanes);
      while (parallel_bernoulli_samplers.size() < lanes)
        parallel_bernoulli_samplers.push_back(
            feebl::electron_lumi_sampling::KeyedBernoulliSampler(0U));
    }
    for (std::size_t lane = 0; lane < lanes; ++lane)
      parallel_bernoulli_samplers[lane].reset(
          sampling_capacity_value,
          std::min(sampling_capacity_value, beam2_particles));
  }
  std::size_t wave_candidates = 0U;
  if (lanes != 0U && beam2_particles >
          std::numeric_limits<std::size_t>::max() / lanes)
    throw std::length_error("electron-luminosity sampler scratch overflow");
  wave_candidates = lanes * beam2_particles;
  if (sampling_mode_value == 1)
    merged_sampler.reset(std::min(sampling_capacity_value, wave_candidates));
  else if (sampling_mode_value == 2)
    merged_bernoulli_sampler.reset(
        sampling_capacity_value,
        std::min(sampling_capacity_value, wave_candidates));
  parallel_lanes = lanes;
  blocks_per_lane = blocks;
  sampler_capacity = sampling_capacity_value;
  sampler_mode = sampling_mode_value;
}

bool Workspace::parallel_capacity_covers(
    std::size_t lanes, std::size_t beam2_particles,
    std::size_t sampling_capacity_value,
    int sampling_mode_value) const noexcept {
  const std::size_t blocks =
      (beam2_particles + kBlockSize - 1U) / kBlockSize;
  if (lanes != 0U && blocks >
          std::numeric_limits<std::size_t>::max() / lanes)
    return false;
  if (!(parallel_lanes >= lanes && blocks_per_lane >= blocks &&
        parallel_blocks.size() >= lanes * blocks &&
        sampler_capacity >= sampling_capacity_value &&
        sampler_mode == sampling_mode_value))
    return false;
  if (sampling_capacity_value == 0U) return true;
  if (sampling_mode_value == 1) {
    if (parallel_samplers.size() < lanes || !merged_sampler.healthy())
      return false;
    const std::size_t lane_capacity =
        std::min(sampling_capacity_value, beam2_particles);
    for (std::size_t lane = 0; lane < lanes; ++lane)
      if (!parallel_samplers[lane].healthy() ||
          parallel_samplers[lane].capacity() < lane_capacity)
        return false;
  } else if (sampling_mode_value == 2) {
    if (parallel_bernoulli_samplers.size() < lanes ||
        !merged_bernoulli_sampler.healthy() ||
        merged_bernoulli_sampler.capacity() < sampling_capacity_value)
      return false;
    for (std::size_t lane = 0; lane < lanes; ++lane)
      if (!parallel_bernoulli_samplers[lane].healthy() ||
          parallel_bernoulli_samplers[lane].capacity() <
              sampling_capacity_value)
        return false;
  }
  return true;
}

BlockKinematics& Workspace::parallel_block(
    std::size_t lane, std::size_t block_index) noexcept {
  return parallel_blocks[lane * blocks_per_lane + block_index];
}

namespace {

#if defined(__GNUC__) || defined(__clang__)
template <typename T>
inline T* assume_aligned_64(T* pointer) noexcept {
  return static_cast<T*>(__builtin_assume_aligned(pointer,
                                                  kScratchAlignment));
}

template <typename T>
inline const T* assume_aligned_64(const T* pointer) noexcept {
  return static_cast<const T*>(__builtin_assume_aligned(
      pointer, kScratchAlignment));
}
#else
template <typename T>
inline T* assume_aligned_64(T* pointer) noexcept { return pointer; }
template <typename T>
inline const T* assume_aligned_64(const T* pointer) noexcept {
  return pointer;
}
#endif

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool evaluate_block(const BeamSoA& beam1, std::size_t beam1_index,
                    const BeamSoA& beam2, std::size_t beam2_begin,
                    std::size_t count, float ecm_min,
                    double electron_mass,
                    BlockKinematics& output) noexcept {
  if (!beam1.component_sizes_match() || !beam2.component_sizes_match() ||
      beam1_index >= beam1.size() || count > kBlockSize ||
      beam2_begin > beam2.size() || count > beam2.size() - beam2_begin ||
      beam2_begin % kBlockSize != 0U)
    return false;
  if (count == 0U) return true;

  // beam2_begin is a multiple of kBlockSize, and all component arrays start at
  // 64-byte boundaries.  The assumptions below are therefore checked by the
  // public contract rather than being speculative alignment promises.
  const float* FEEBL_LUMI_RESTRICT signed_energy2 = assume_aligned_64(
      beam2.signed_energy.data() + beam2_begin);
  const float* FEEBL_LUMI_RESTRICT absolute_energy2 = assume_aligned_64(
      beam2.absolute_energy.data() + beam2_begin);
  const float* FEEBL_LUMI_RESTRICT vx2 =
      assume_aligned_64(beam2.vx.data() + beam2_begin);
  const float* FEEBL_LUMI_RESTRICT vy2 =
      assume_aligned_64(beam2.vy.data() + beam2_begin);
  const float* FEEBL_LUMI_RESTRICT vz2 =
      assume_aligned_64(beam2.vz.data() + beam2_begin);
  const float* FEEBL_LUMI_RESTRICT particle_weight2 = assume_aligned_64(
      beam2.weight.data() + beam2_begin);
  const double* FEEBL_LUMI_RESTRICT massive_momentum2 = assume_aligned_64(
      beam2.massive_momentum.data() + beam2_begin);

  float* FEEBL_LUMI_RESTRICT pair_weight =
      assume_aligned_64(output.weight.data());
  float* FEEBL_LUMI_RESTRICT ecm =
      assume_aligned_64(output.ecm.data());
  double* FEEBL_LUMI_RESTRICT invariant_mass =
      assume_aligned_64(output.invariant_mass.data());
  std::uint32_t* FEEBL_LUMI_RESTRICT same_sign =
      assume_aligned_64(output.same_sign.data());
  std::uint32_t* FEEBL_LUMI_RESTRICT above_threshold =
      assume_aligned_64(output.above_threshold.data());

  const float signed_energy1 = beam1.signed_energy[beam1_index];
  const float absolute_energy1 = beam1.absolute_energy[beam1_index];
  const float vx1 = beam1.vx[beam1_index];
  const float vy1 = beam1.vy[beam1_index];
  const float vz1 = beam1.vz[beam1_index];
  const float particle_weight1 = beam1.weight[beam1_index];
  const double massive_momentum1 =
      beam1.massive_momentum[beam1_index];

  // Keep the source operation spelling of GRID::step_lumi_precomputed.  Each
  // loop has no carried dependency. Separating numeric outputs from integer
  // predicates avoids making byte packing a precondition for vectorizing the
  // two square-root-heavy calculations.
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (std::size_t index = 0; index < count; ++index) {
    pair_weight[index] = particle_weight1 * particle_weight2[index];
    const float cosinus =
        vx1 * vx2[index] + vy1 * vy2[index] - vz1 * vz2[index];
    ecm[index] = static_cast<float>(std::sqrt(
        2.0 * absolute_energy1 * absolute_energy2[index] *
        (1. - cosinus)));
  }

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const float signed_product = signed_energy1 * signed_energy2[index];
    same_sign[index] = static_cast<std::uint32_t>(signed_product > 0.0);
    above_threshold[index] =
        static_cast<std::uint32_t>(ecm[index] > ecm_min);
  }

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const double cosang =
        static_cast<double>(vx1) * static_cast<double>(vx2[index]) +
        static_cast<double>(vy1) * static_cast<double>(vy2[index]) +
        static_cast<double>(vz1) * -static_cast<double>(vz2[index]);
    const double w2 =
        2.0 * electron_mass * electron_mass +
        2.0 * (static_cast<double>(signed_energy1) *
                   static_cast<double>(signed_energy2[index]) -
               massive_momentum1 * massive_momentum2[index] * cosang);
    invariant_mass[index] = (w2 > 0.0) ? std::sqrt(w2) : 0.0;
  }
  return true;
}

}  // namespace legacy_electron_lumi_blocked
}  // namespace feebl

#undef FEEBL_LUMI_RESTRICT
