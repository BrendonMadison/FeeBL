#include "legacyHostWaveConcurrencyCPP.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {

using feebl::legacy_host_wave::Aggregate;
using feebl::legacy_host_wave::EncounterResultLane;
using feebl::modern::EncounterId;
using feebl::modern::EncounterLogicalTime;
using feebl::modern::EncounterNode;

static_assert(std::is_trivially_copyable<EncounterResultLane>::value,
              "host-wave result lanes must remain trivially copyable");
static_assert(std::is_standard_layout<EncounterResultLane>::value,
              "host-wave result lanes must remain standard-layout");

bool add_u64(std::uint64_t first, std::uint64_t second,
             std::uint64_t* result) noexcept {
  if (first > std::numeric_limits<std::uint64_t>::max() - second)
    return false;
  *result = first + second;
  return true;
}

void add_saturating(std::atomic<std::uint64_t>& target,
                    std::uint64_t increment) noexcept {
  std::uint64_t observed = target.load(std::memory_order_relaxed);
  for (;;) {
    const std::uint64_t desired =
        observed > std::numeric_limits<std::uint64_t>::max() - increment
            ? std::numeric_limits<std::uint64_t>::max()
            : observed + increment;
    if (target.compare_exchange_weak(observed, desired,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed))
      return;
  }
}

std::uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point& begin) noexcept {
  const std::chrono::steady_clock::duration elapsed =
      std::chrono::steady_clock::now() - begin;
  const long double count =
      std::chrono::duration<long double, std::nano>(elapsed).count();
  if (!(count > 0.0L)) return 0;
  if (count >= static_cast<long double>(
                   std::numeric_limits<std::uint64_t>::max()))
    return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(count);
}

enum class MergeFailure : std::uint8_t {
  None = 0,
  LaneInvalid = 1,
  EncounterOrder = 2,
  CounterOverflow = 3,
  NonFiniteSum = 4
};

const char* merge_failure_text(MergeFailure failure) noexcept {
  switch (failure) {
    case MergeFailure::None:
      return "none";
    case MergeFailure::LaneInvalid:
      return "encounter result lane failed merge-time validation";
    case MergeFailure::EncounterOrder:
      return "encounter result lanes were not merged in canonical ID order";
    case MergeFailure::CounterOverflow:
      return "encounter result aggregate counter overflow";
    case MergeFailure::NonFiniteSum:
      return "encounter result aggregate became non-finite";
  }
  return "unknown encounter result merge failure";
}

bool full_values_finite(const EncounterResultLane& lane) noexcept {
  return std::isfinite(lane.ee_luminosity) &&
         std::isfinite(lane.ee_high_luminosity) &&
         std::isfinite(lane.ecm_weighted_sum) &&
         std::isfinite(lane.ecm_squared_weighted_sum) &&
         std::isfinite(lane.photon_luminosity_eg) &&
         std::isfinite(lane.photon_luminosity_ge) &&
         std::isfinite(lane.photon_luminosity_gg) &&
         std::isfinite(lane.beam_energy_loss) &&
         std::isfinite(lane.emitted_photon_energy) &&
         std::isfinite(lane.maximum_upsilon) &&
         std::isfinite(lane.luminosity_weighted_helicity1) &&
         std::isfinite(lane.luminosity_weighted_helicity2) &&
         std::isfinite(lane.luminosity_helicity_weight);
}

bool full_values_nonnegative(const EncounterResultLane& lane) noexcept {
  return lane.ee_luminosity >= 0.0 && lane.ee_high_luminosity >= 0.0 &&
         lane.ecm_weighted_sum >= 0.0 &&
         lane.ecm_squared_weighted_sum >= 0.0 &&
         lane.photon_luminosity_eg >= 0.0 &&
         lane.photon_luminosity_ge >= 0.0 &&
         lane.photon_luminosity_gg >= 0.0 &&
         lane.beam_energy_loss >= 0.0 &&
         lane.emitted_photon_energy >= 0.0 && lane.maximum_upsilon >= 0.0 &&
         lane.luminosity_helicity_weight >= 0.0;
}

bool full_values_zero(const EncounterResultLane& lane) noexcept {
  return lane.ee_luminosity == 0.0 && lane.ee_high_luminosity == 0.0 &&
         lane.ecm_weighted_sum == 0.0 &&
         lane.ecm_squared_weighted_sum == 0.0 &&
         lane.photon_luminosity_eg == 0.0 &&
         lane.photon_luminosity_ge == 0.0 &&
         lane.photon_luminosity_gg == 0.0 && lane.beam_energy_loss == 0.0 &&
         lane.emitted_photon_energy == 0.0 && lane.maximum_upsilon == 0.0 &&
         lane.luminosity_weighted_helicity1 == 0.0 &&
         lane.luminosity_weighted_helicity2 == 0.0 &&
         lane.luminosity_helicity_weight == 0.0 &&
         lane.electron_pair_evaluations == 0 &&
         lane.photon_lumi_opportunities == 0 && lane.emitted_photons == 0 &&
         lane.counter_rng_streams == 0 && lane.full_result_terms == 0 &&
         lane.lumi_candidates_seen == 0 &&
         lane.lumi_candidates_retained == 0 &&
         lane.lumi_reservoir_digest == 0 && lane.emission_records == 0;
}

bool lane_valid_noexcept(
    const EncounterResultLane& lane, const EncounterNode& node,
    const EncounterLogicalTime& logical_time,
    feebl::legacy_host_wave::ContractTier tier,
    std::uint64_t lumi_reservoir_capacity) noexcept {
  if (lane.schema_version !=
          feebl::legacy_host_wave::kResultLaneSchemaVersion ||
      lane.contract_tier != static_cast<std::uint32_t>(tier) ||
      lane.encounter != node.id || lane.logical_tick != logical_time.tick ||
      !std::isfinite(lane.fine_lumi_dot) ||
      !std::isfinite(lane.fine_luminosity) || !full_values_finite(lane))
    return false;
  std::uint64_t expected = 0;
  if (!add_u64(lane.beam1_in, lane.beam1_out, &expected) ||
      expected != lane.moved_beam1 ||
      !add_u64(lane.beam2_in, lane.beam2_out, &expected) ||
      expected != lane.moved_beam2)
    return false;
  if (tier == feebl::legacy_host_wave::ContractTier::
                  OrbitBmtFineLuminosityOnly)
    return full_values_zero(lane);
  if (tier != feebl::legacy_host_wave::ContractTier::
                  FcczPrimaryBeamstrahlungPhotonFullEeLuminosity ||
      !full_values_nonnegative(lane) || lumi_reservoir_capacity == 0 ||
      lane.lumi_candidates_retained > lane.lumi_candidates_seen ||
      lane.lumi_candidates_retained > lumi_reservoir_capacity)
    return false;
  std::uint64_t expected_emissions = 0;
  return add_u64(lane.emitted_photons, lane.lumi_candidates_retained,
                 &expected_emissions) &&
         expected_emissions == lane.emission_records;
}

bool merge_lane_noexcept(Aggregate* aggregate,
                         const EncounterResultLane& lane,
                         const EncounterNode& node,
                         const EncounterLogicalTime& logical_time,
                         feebl::legacy_host_wave::ContractTier tier,
                         std::uint64_t lumi_reservoir_capacity,
                         MergeFailure* failure) noexcept {
  if (!lane_valid_noexcept(lane, node, logical_time, tier,
                           lumi_reservoir_capacity)) {
    *failure = MergeFailure::LaneInvalid;
    return false;
  }
  if (aggregate->tier != tier || node.id != aggregate->encounters) {
    *failure = MergeFailure::EncounterOrder;
    return false;
  }

  Aggregate candidate = *aggregate;
  if (!add_u64(candidate.encounters, 1, &candidate.encounters) ||
      !add_u64(candidate.beam1_in, lane.beam1_in, &candidate.beam1_in) ||
      !add_u64(candidate.beam1_out, lane.beam1_out,
               &candidate.beam1_out) ||
      !add_u64(candidate.beam2_in, lane.beam2_in, &candidate.beam2_in) ||
      !add_u64(candidate.beam2_out, lane.beam2_out,
               &candidate.beam2_out) ||
      !add_u64(candidate.moved_beam1, lane.moved_beam1,
               &candidate.moved_beam1) ||
      !add_u64(candidate.moved_beam2, lane.moved_beam2,
               &candidate.moved_beam2) ||
      !add_u64(candidate.electron_pair_evaluations,
               lane.electron_pair_evaluations,
               &candidate.electron_pair_evaluations) ||
      !add_u64(candidate.photon_lumi_opportunities,
               lane.photon_lumi_opportunities,
               &candidate.photon_lumi_opportunities) ||
      !add_u64(candidate.emitted_photons, lane.emitted_photons,
               &candidate.emitted_photons) ||
      !add_u64(candidate.counter_rng_streams, lane.counter_rng_streams,
               &candidate.counter_rng_streams) ||
      !add_u64(candidate.full_result_terms, lane.full_result_terms,
               &candidate.full_result_terms) ||
      !add_u64(candidate.lumi_candidates_seen, lane.lumi_candidates_seen,
               &candidate.lumi_candidates_seen) ||
      !add_u64(candidate.lumi_candidates_retained,
               lane.lumi_candidates_retained,
               &candidate.lumi_candidates_retained) ||
      !add_u64(candidate.emission_records, lane.emission_records,
               &candidate.emission_records)) {
    *failure = MergeFailure::CounterOverflow;
    return false;
  }
  candidate.fine_lumi_dot_sum += lane.fine_lumi_dot;
  candidate.fine_luminosity_sum += lane.fine_luminosity;
  candidate.ee_luminosity += lane.ee_luminosity;
  candidate.ee_high_luminosity += lane.ee_high_luminosity;
  candidate.ecm_weighted_sum += lane.ecm_weighted_sum;
  candidate.ecm_squared_weighted_sum += lane.ecm_squared_weighted_sum;
  candidate.photon_luminosity_eg += lane.photon_luminosity_eg;
  candidate.photon_luminosity_ge += lane.photon_luminosity_ge;
  candidate.photon_luminosity_gg += lane.photon_luminosity_gg;
  candidate.beam_energy_loss += lane.beam_energy_loss;
  candidate.emitted_photon_energy += lane.emitted_photon_energy;
  candidate.maximum_upsilon =
      std::max(candidate.maximum_upsilon, lane.maximum_upsilon);
  candidate.luminosity_weighted_helicity1 +=
      lane.luminosity_weighted_helicity1;
  candidate.luminosity_weighted_helicity2 +=
      lane.luminosity_weighted_helicity2;
  candidate.luminosity_helicity_weight +=
      lane.luminosity_helicity_weight;
  if (tier == feebl::legacy_host_wave::ContractTier::
                  FcczPrimaryBeamstrahlungPhotonFullEeLuminosity)
    candidate.lumi_reservoir_digest =
        (candidate.lumi_reservoir_digest * 1099511628211ULL) ^
        lane.lumi_reservoir_digest ^ node.id;
  if (!std::isfinite(candidate.fine_lumi_dot_sum) ||
      !std::isfinite(candidate.fine_luminosity_sum) ||
      !std::isfinite(candidate.ee_luminosity) ||
      !std::isfinite(candidate.ee_high_luminosity) ||
      !std::isfinite(candidate.ecm_weighted_sum) ||
      !std::isfinite(candidate.ecm_squared_weighted_sum) ||
      !std::isfinite(candidate.photon_luminosity_eg) ||
      !std::isfinite(candidate.photon_luminosity_ge) ||
      !std::isfinite(candidate.photon_luminosity_gg) ||
      !std::isfinite(candidate.beam_energy_loss) ||
      !std::isfinite(candidate.emitted_photon_energy) ||
      !std::isfinite(candidate.maximum_upsilon) ||
      !std::isfinite(candidate.luminosity_weighted_helicity1) ||
      !std::isfinite(candidate.luminosity_weighted_helicity2) ||
      !std::isfinite(candidate.luminosity_helicity_weight)) {
    *failure = MergeFailure::NonFiniteSum;
    return false;
  }
  *aggregate = candidate;
  return true;
}

struct ExecutionState {
  feebl::legacy_host_wave::ComputeCallback callback = nullptr;
  void* user_data = nullptr;
  std::atomic<std::uint64_t> attempted{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> callback_nanoseconds{0};
  Aggregate candidate;
  feebl::legacy_host_wave::ContractTier tier =
      feebl::legacy_host_wave::ContractTier::Rejected;
  std::uint64_t lumi_reservoir_capacity = 0;
  MergeFailure merge_failure = MergeFailure::None;
  EncounterId merge_failure_encounter = feebl::modern::kInvalidEncounter;
  std::uint64_t merged = 0;
  std::uint64_t merge_nanoseconds = 0;
};

void compute_bridge(const EncounterNode& node,
                    const EncounterLogicalTime& logical_time,
                    feebl::modern::StepContext& context,
                    feebl::modern::MutableByteLane result_lane,
                    feebl::modern::MutableByteLane emission_lane,
                    void* opaque) {
  ExecutionState& state = *static_cast<ExecutionState*>(opaque);
  state.attempted.fetch_add(1, std::memory_order_relaxed);
  const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  try {
    if (!emission_lane.empty())
      throw std::logic_error(
          "fine-luminosity host-wave contract received an emission lane");
    EncounterResultLane result;
    result.contract_tier = static_cast<std::uint32_t>(state.tier);
    result.encounter = node.id;
    result.logical_tick = logical_time.tick;
    state.callback(node, logical_time, context, result, state.user_data);
    const feebl::legacy_host_wave::LaneValidation validation =
        feebl::legacy_host_wave::validate_lane(
            result, node, logical_time, state.tier,
            state.lumi_reservoir_capacity);
    if (!validation.ok)
      throw std::runtime_error(validation.message);
    result_lane.store(result);
    state.completed.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    add_saturating(state.callback_nanoseconds, elapsed_nanoseconds(begin));
    throw;
  }
  add_saturating(state.callback_nanoseconds, elapsed_nanoseconds(begin));
}

void merge_bridge(const EncounterNode& node,
                  const EncounterLogicalTime& logical_time,
                  feebl::modern::ConstByteLane result_lane,
                  feebl::modern::ConstByteLane emission_lane,
                  void* opaque) noexcept {
  ExecutionState& state = *static_cast<ExecutionState*>(opaque);
  const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  if (state.merge_failure != MergeFailure::None) {
    state.merge_nanoseconds += elapsed_nanoseconds(begin);
    return;
  }
  if (!emission_lane.empty() ||
      result_lane.size() != sizeof(EncounterResultLane)) {
    state.merge_failure = MergeFailure::LaneInvalid;
    state.merge_failure_encounter = node.id;
    state.merge_nanoseconds += elapsed_nanoseconds(begin);
    return;
  }
  const EncounterResultLane lane = result_lane.load<EncounterResultLane>();
  if (!merge_lane_noexcept(&state.candidate, lane, node, logical_time,
                           state.tier, state.lumi_reservoir_capacity,
                           &state.merge_failure)) {
    state.merge_failure_encounter = node.id;
    state.merge_nanoseconds += elapsed_nanoseconds(begin);
    return;
  }
  ++state.merged;
  const std::uint64_t elapsed = elapsed_nanoseconds(begin);
  if (state.merge_nanoseconds >
      std::numeric_limits<std::uint64_t>::max() - elapsed)
    state.merge_nanoseconds = std::numeric_limits<std::uint64_t>::max();
  else
    state.merge_nanoseconds += elapsed;
}

feebl::modern::CpuExecutorConfig make_cpu_config(
    const feebl::legacy_host_wave::ExecutorConfig& config) {
  feebl::modern::CpuExecutorConfig result;
  result.execution = config.execution;
  result.strategy = config.strategy;
  result.context_reset = config.context_reset;
  result.worker_count = config.worker_count;
  result.context_count = config.context_count;
  result.context_bytes = config.context_bytes;
  result.result_lane_bytes = sizeof(EncounterResultLane);
  result.emission_lane_bytes = 0;
  result.allow_serial_fallback_without_openmp =
      config.allow_serial_fallback_without_openmp;
  result.limits = config.limits;
  return result;
}

}  // namespace

namespace feebl {
namespace legacy_host_wave {

FeatureSet FeatureSet::orbit_bmt_fine_luminosity_only() noexcept {
  FeatureSet result;
  result.requested_tier = ContractTier::OrbitBmtFineLuminosityOnly;
  result.prefetched_fields = true;
  result.integration_method = 2;
  result.electron_distribution_rho = 2;
  result.interpolation = 2;
  result.force_symmetric = false;
  result.extra_grids = 0;
  result.electron_ratio = 0.0;
  result.encounter_local_grid_scratch = true;
  result.encounter_local_results = true;
  result.encounter_local_timers = true;
  result.deterministic_canonical_merge = true;
  result.exclusive_slices_within_wave = true;
  result.rng_policy = RngPolicy::None;
  result.result_policy = ResultPolicy::FineLuminosityLaneOnly;
  result.emission_policy = EmissionPolicy::None;
  return result;
}

FeatureSet
FeatureSet::fccz_primary_beamstrahlung_photon_full_ee_luminosity() noexcept {
  FeatureSet result;
  result.requested_tier =
      ContractTier::FcczPrimaryBeamstrahlungPhotonFullEeLuminosity;
  result.prefetched_fields = true;
  result.integration_method = 2;
  result.electron_distribution_rho = 2;
  result.interpolation = 2;
  result.force_symmetric = false;
  result.extra_grids = 0;
  result.electron_ratio = 1.0;
  result.beamstrahlung = true;
  result.do_photons_beam1 = true;
  result.do_photons_beam2 = true;
  result.do_photon_luminosity = true;
  result.do_full_electron_luminosity = true;
  result.store_native_luminosity = true;
  result.encounter_local_grid_scratch = true;
  result.encounter_local_results = true;
  result.encounter_local_timers = true;
  result.deterministic_canonical_merge = true;
  result.exclusive_slices_within_wave = true;
  result.counter_rng_v2_derived_streams = true;
  result.stable_particle_ids = true;
  result.deterministic_full_results_merge = true;
  result.canonical_bounded_lumi_reservoir_merge = true;
  result.thread_safe_distinct_slice_photon_insertion = true;
  result.lumi_event_reservoir_capacity = 50000;
  result.rng_policy = RngPolicy::CounterPerEncounter;
  result.result_policy = ResultPolicy::EncounterLocalFull;
  result.emission_policy = EmissionPolicy::EncounterLocalCanonicalMerge;
  return result;
}

const std::string& EligibilityReport::first_reason() const noexcept {
  static const std::string empty;
  return reasons.empty() ? empty : reasons.front();
}

EligibilityReport audit(const FeatureSet& features) {
  EligibilityReport report;
  const auto reject = [&](Blocker blocker, const char* reason) {
    report.blockers |= blocker_mask(blocker);
    report.reasons.push_back(reason);
  };

  if (features.schema_version != kContractSchemaVersion)
    reject(Blocker::UnsupportedContractSchema,
           "unsupported host-wave contract schema");
  if (!features.prefetched_fields)
    reject(Blocker::MissingPrefetchedFields,
           "host-wave concurrency requires immutable prefetched fields");
  if (features.integration_method != 2 ||
      features.electron_distribution_rho != 2 ||
      features.interpolation != 2)
    reject(Blocker::UnsupportedGridContract,
           "host-wave concurrency requires integration_method=2, CIC rho, "
           "and CIC interpolation");
  if (features.force_symmetric)
    reject(Blocker::ForceSymmetric,
           "force_symmetric mutates shared charge state");
  if (features.extra_grids != 0)
    reject(Blocker::ExtraGrids,
           "extra grids have no encounter-local host adapter");
  if (!std::isfinite(features.electron_ratio))
    reject(Blocker::NonFiniteConfiguration,
           "electron_ratio must be finite");
  if (features.stochastic_spin_flip)
    reject(Blocker::StochasticSpinFlip,
           "stochastic spin flip is outside both admitted host-wave tiers");
  if (features.load_photons)
    reject(Blocker::PrimaryPhotons,
           "loaded photon state has no encounter-local ownership adapter");
  if (features.do_prod || features.do_trident || features.do_coherent ||
      features.do_pairs || features.do_hadrons || features.do_compton ||
      features.do_muons)
    reject(Blocker::PairOrStrongFieldProduction,
           "pair/strong-field/event production has shared secondary state");
  if (features.track_pairs || features.track_muons)
    reject(Blocker::TrackedSecondaries,
           "tracked secondaries cross encounter and diagonal boundaries");
  if (features.do_bhabhas || features.do_jets || features.do_isr)
    reject(Blocker::EventGenerators,
           "event generators have shared RNG and output state");
  if (features.has_external_output_sink)
    reject(Blocker::SharedOutputSink,
           "encounter callbacks may not write shared files or streams");
  if (!features.encounter_local_grid_scratch)
    reject(Blocker::MissingEncounterLocalGrid,
           "adapter must own encounter-local GRID scratch and cell lists");
  if (!features.encounter_local_results)
    reject(Blocker::MissingEncounterLocalResults,
           "adapter must own encounter-local RESULTS and provenance state");
  if (!features.encounter_local_timers)
    reject(Blocker::MissingEncounterLocalTimers,
           "adapter must use encounter-local timing counters");
  if (!features.deterministic_canonical_merge)
    reject(Blocker::MissingCanonicalMerge,
           "adapter must merge validated lanes in canonical encounter order");
  if (!features.exclusive_slices_within_wave)
    reject(Blocker::MissingExclusiveSliceProof,
           "scheduler must prove exclusive beam slices within every wave");

  switch (features.requested_tier) {
    case ContractTier::OrbitBmtFineLuminosityOnly:
      if (std::isfinite(features.electron_ratio) &&
          features.electron_ratio != 0.0)
        reject(Blocker::ElectronSampling,
               "fine-luminosity-only tier requires electron_ratio=0");
      if (features.rng_policy != RngPolicy::None)
        reject(Blocker::SharedOrUnqualifiedRng,
               "fine-luminosity-only tier is RNG-free");
      if (features.beamstrahlung)
        reject(Blocker::Beamstrahlung,
               "beamstrahlung requires the explicit FCC-Z full-physics tier");
      if (features.do_full_electron_luminosity)
        reject(Blocker::FullElectronLuminosity,
               "full electron luminosity requires the explicit FCC-Z tier");
      if (features.store_native_luminosity)
        reject(Blocker::NativeLuminosityOutput,
               "native luminosity output requires the explicit FCC-Z tier");
      if (features.do_photons_beam1 || features.do_photons_beam2)
        reject(Blocker::PrimaryPhotons,
               "generated photons require the explicit FCC-Z tier");
      if (features.do_photon_luminosity)
        reject(Blocker::PhotonLuminosity,
               "photon luminosity requires the explicit FCC-Z tier");
      if (features.result_policy != ResultPolicy::FineLuminosityLaneOnly)
        reject(Blocker::SharedResults,
               "fine-luminosity tier requires its typed result lane");
      if (features.emission_policy != EmissionPolicy::None)
        reject(Blocker::SharedEmissions,
               "fine-luminosity tier permits no encounter emissions");
      break;

    case ContractTier::FcczPrimaryBeamstrahlungPhotonFullEeLuminosity:
      if (std::isfinite(features.electron_ratio) &&
          (!(features.electron_ratio > 0.0) ||
           features.electron_ratio > 1.0))
        reject(Blocker::ElectronSampling,
               "FCC-Z full-physics tier requires 0 < electron_ratio <= 1");
      if (features.rng_policy != RngPolicy::CounterPerEncounter)
        reject(Blocker::SharedOrUnqualifiedRng,
               "FCC-Z full-physics tier requires per-encounter counter RNG");
      if (!features.counter_rng_v2_derived_streams)
        reject(Blocker::MissingCounterRngV2Streams,
               "adapter must derive every stochastic stream from CounterRngV2 "
               "and canonical encounter/particle/event identities");
      if (!features.stable_particle_ids)
        reject(Blocker::MissingStableParticleIds,
               "CounterRngV2 host physics requires stable particle IDs");
      if (!features.beamstrahlung)
        reject(Blocker::Beamstrahlung,
               "FCC-Z full-physics tier requires beamstrahlung enabled");
      if (!features.do_photons_beam1 || !features.do_photons_beam2)
        reject(Blocker::PrimaryPhotons,
               "FCC-Z full-physics tier requires photons for both beams");
      if (!features.do_photon_luminosity)
        reject(Blocker::PhotonLuminosity,
               "FCC-Z full-physics tier requires photon luminosity handling");
      if (!features.do_full_electron_luminosity)
        reject(Blocker::FullElectronLuminosity,
               "FCC-Z tier requires full electron luminosity");
      if (!features.store_native_luminosity)
        reject(Blocker::NativeLuminosityOutput,
               "FCC-Z tier requires bounded native luminosity sampling");
      if (features.result_policy != ResultPolicy::EncounterLocalFull)
        reject(Blocker::SharedResults,
               "FCC-Z tier requires encounter-local full result lanes");
      if (features.emission_policy !=
          EmissionPolicy::EncounterLocalCanonicalMerge)
        reject(Blocker::SharedEmissions,
               "FCC-Z tier requires encounter-local canonical emissions");
      if (!features.deterministic_full_results_merge)
        reject(Blocker::MissingFullResultsMerge,
               "adapter must provide deterministic full-RESULTS merge");
      if (!features.canonical_bounded_lumi_reservoir_merge)
        reject(Blocker::MissingBoundedReservoirMerge,
               "adapter must provide canonical bounded luminosity-reservoir "
               "merge");
      if (features.lumi_event_reservoir_capacity == 0)
        reject(Blocker::InvalidReservoirCapacity,
               "FCC-Z tier requires a positive bounded luminosity reservoir");
      if (!features.thread_safe_distinct_slice_photon_insertion)
        reject(Blocker::MissingThreadSafePhotonInsertion,
               "adapter must prove thread-safe photon insertion into distinct "
               "beam slices");
      break;

    case ContractTier::Rejected:
    default:
      reject(Blocker::UnsupportedContractTier,
             "host-wave contract tier is not explicitly supported");
      break;
  }

  if (report.blockers == 0) report.tier = features.requested_tier;
  return report;
}

const char* contract_tier_name(ContractTier tier) noexcept {
  switch (tier) {
    case ContractTier::Rejected:
      return "rejected";
    case ContractTier::OrbitBmtFineLuminosityOnly:
      return "orbit_bmt_fine_luminosity_only_v1";
    case ContractTier::FcczPrimaryBeamstrahlungPhotonFullEeLuminosity:
      return "fccz_primary_beamstrahlung_photon_full_ee_luminosity_v1";
  }
  return "unknown";
}

LaneValidation validate_lane(const EncounterResultLane& lane,
                             const modern::EncounterNode& node,
                             const modern::EncounterLogicalTime& logical_time,
                             ContractTier tier,
                             std::uint64_t lumi_reservoir_capacity) {
  LaneValidation result;
  if (lane.schema_version != kResultLaneSchemaVersion)
    result.message = "host-wave result lane schema mismatch";
  else if (lane.contract_tier != static_cast<std::uint32_t>(tier))
    result.message = "host-wave result lane contract-tier mismatch";
  else if (lane.encounter != node.id)
    result.message = "host-wave result lane encounter identity mismatch";
  else if (lane.logical_tick != logical_time.tick)
    result.message = "host-wave result lane logical-time mismatch";
  else if (!std::isfinite(lane.fine_lumi_dot) ||
           !std::isfinite(lane.fine_luminosity))
    result.message = "host-wave result lane contains non-finite luminosity";
  else if (!full_values_finite(lane))
    result.message =
        "host-wave result lane contains a non-finite full-physics value";
  else {
    std::uint64_t expected = 0;
    if (!add_u64(lane.beam1_in, lane.beam1_out, &expected))
      result.message = "host-wave beam-1 distribution count overflow";
    else if (expected != lane.moved_beam1)
      result.message =
          "host-wave beam-1 moved count does not match distribution count";
    else if (!add_u64(lane.beam2_in, lane.beam2_out, &expected))
      result.message = "host-wave beam-2 distribution count overflow";
    else if (expected != lane.moved_beam2)
      result.message =
          "host-wave beam-2 moved count does not match distribution count";
    else if (tier == ContractTier::OrbitBmtFineLuminosityOnly) {
      if (!full_values_zero(lane))
        result.message =
            "fine-luminosity tier received full-physics results or emissions";
      else
        result.ok = true;
    } else if (tier == ContractTier::
                           FcczPrimaryBeamstrahlungPhotonFullEeLuminosity) {
      std::uint64_t expected_emissions = 0;
      if (!full_values_nonnegative(lane))
        result.message =
            "FCC-Z result lane contains a negative positive-definite value";
      else if (lumi_reservoir_capacity == 0)
        result.message = "FCC-Z result lane has no bounded reservoir capacity";
      else if (lane.lumi_candidates_retained > lane.lumi_candidates_seen)
        result.message =
            "FCC-Z retained luminosity candidates exceed candidates seen";
      else if (lane.lumi_candidates_retained > lumi_reservoir_capacity)
        result.message =
            "FCC-Z encounter luminosity candidates exceed reservoir capacity";
      else if (!add_u64(lane.emitted_photons,
                        lane.lumi_candidates_retained,
                        &expected_emissions))
        result.message = "FCC-Z emission count overflow";
      else if (expected_emissions != lane.emission_records)
        result.message =
            "FCC-Z emission records do not match photons plus retained lumi "
            "candidates";
      else
        result.ok = true;
    } else {
      result.message = "host-wave result lane uses an unsupported tier";
    }
  }
  return result;
}

HostWaveExecutor::HostWaveExecutor(const modern::EncounterDAG& dag,
                                   const FeatureSet& features,
                                   const ExecutorConfig& config)
    : eligibility_(audit(features)), features_(features) {
  if (!eligibility_.eligible()) {
    std::ostringstream message;
    message << "host-wave concurrency preflight rejected: "
            << eligibility_.first_reason() << " (blocker_mask=0x" << std::hex
            << eligibility_.blockers << ")";
    throw HostWaveError(message.str());
  }
  if (config.worker_count == 0 || config.context_count == 0)
    throw HostWaveError(
        "host-wave concurrency requires positive worker/context counts");
  executor_.reset(
      new modern::CpuEncounterExecutor(dag, make_cpu_config(config)));
  telemetry_.tier = eligibility_.tier;
  telemetry_.waves = executor_->dag().wave_count();
  telemetry_.planned_encounters = executor_->dag().encounter_count();
  telemetry_.requested_workers = config.worker_count;
  telemetry_.context_capacity = config.context_count;
  telemetry_.result_lane_bytes = sizeof(EncounterResultLane);
  telemetry_.emission_lane_bytes = 0;
  telemetry_.lumi_event_reservoir_capacity =
      features_.lumi_event_reservoir_capacity;
  telemetry_.counter_rng_v2_contract =
      features_.counter_rng_v2_derived_streams;
  telemetry_.deterministic_full_results_merge_contract =
      features_.deterministic_full_results_merge;
  telemetry_.canonical_bounded_reservoir_merge_contract =
      features_.canonical_bounded_lumi_reservoir_merge;
  telemetry_.distinct_slice_photon_insertion_contract =
      features_.thread_safe_distinct_slice_photon_insertion;
}

HostWaveExecutor::~HostWaveExecutor() = default;

RunResult HostWaveExecutor::execute(ComputeCallback callback, void* user_data) {
  if (!callback)
    throw HostWaveError("host-wave compute callback must not be null");

  output_valid_ = false;
  aggregate_ = Aggregate{};
  const ContractTier tier = telemetry_.tier;
  const std::uint64_t waves = telemetry_.waves;
  const std::uint64_t planned = telemetry_.planned_encounters;
  const std::uint32_t requested_workers = telemetry_.requested_workers;
  const std::uint32_t context_capacity = telemetry_.context_capacity;
  telemetry_ = Telemetry{};
  telemetry_.tier = tier;
  telemetry_.waves = waves;
  telemetry_.planned_encounters = planned;
  telemetry_.requested_workers = requested_workers;
  telemetry_.context_capacity = context_capacity;
  telemetry_.result_lane_bytes = sizeof(EncounterResultLane);
  telemetry_.lumi_event_reservoir_capacity =
      features_.lumi_event_reservoir_capacity;
  telemetry_.counter_rng_v2_contract =
      features_.counter_rng_v2_derived_streams;
  telemetry_.deterministic_full_results_merge_contract =
      features_.deterministic_full_results_merge;
  telemetry_.canonical_bounded_reservoir_merge_contract =
      features_.canonical_bounded_lumi_reservoir_merge;
  telemetry_.distinct_slice_photon_insertion_contract =
      features_.thread_safe_distinct_slice_photon_insertion;

  ExecutionState state;
  state.callback = callback;
  state.user_data = user_data;
  state.tier = tier;
  state.lumi_reservoir_capacity =
      features_.lumi_event_reservoir_capacity;
  state.candidate.tier = tier;
  const std::chrono::steady_clock::time_point wall_begin =
      std::chrono::steady_clock::now();

  modern::CpuExecutionReport report;
  try {
    report = executor_->execute(compute_bridge, &state, merge_bridge, &state);
  } catch (const modern::CpuExecutorError& error) {
    telemetry_.attempted_encounters =
        state.attempted.load(std::memory_order_relaxed);
    telemetry_.completed_encounters =
        state.completed.load(std::memory_order_relaxed);
    telemetry_.callback_cpu_nanoseconds =
        state.callback_nanoseconds.load(std::memory_order_relaxed);
    telemetry_.merge_nanoseconds = state.merge_nanoseconds;
    telemetry_.execute_wall_nanoseconds = elapsed_nanoseconds(wall_begin);
    throw HostWaveError(std::string("host-wave execution failed: ") +
                            error.what(),
                        error.encounter());
  } catch (const std::exception& error) {
    telemetry_.attempted_encounters =
        state.attempted.load(std::memory_order_relaxed);
    telemetry_.completed_encounters =
        state.completed.load(std::memory_order_relaxed);
    telemetry_.callback_cpu_nanoseconds =
        state.callback_nanoseconds.load(std::memory_order_relaxed);
    telemetry_.merge_nanoseconds = state.merge_nanoseconds;
    telemetry_.execute_wall_nanoseconds = elapsed_nanoseconds(wall_begin);
    throw HostWaveError(std::string("host-wave execution failed: ") +
                        error.what());
  }

  telemetry_.attempted_encounters = report.attempted_encounters;
  telemetry_.completed_encounters = report.completed_encounters;
  telemetry_.merged_encounters = state.merged;
  telemetry_.actual_workers = report.actual_workers;
  telemetry_.context_high_water = report.context_high_water;
  telemetry_.parallel_regions = report.parallel_regions;
  telemetry_.used_serial_fallback = report.used_serial_fallback;
  telemetry_.callback_cpu_nanoseconds =
      state.callback_nanoseconds.load(std::memory_order_relaxed);
  telemetry_.merge_nanoseconds = state.merge_nanoseconds;
  telemetry_.execute_wall_nanoseconds = elapsed_nanoseconds(wall_begin);

  if (state.merge_failure != MergeFailure::None) {
    std::ostringstream message;
    message << "host-wave canonical merge rejected encounter "
            << state.merge_failure_encounter << ": "
            << merge_failure_text(state.merge_failure)
            << "; no aggregate was published";
    throw HostWaveError(message.str(), state.merge_failure_encounter);
  }
  if (!report.success || state.merged != planned ||
      state.candidate.encounters != planned)
    throw HostWaveError(
        "host-wave execution completed without an exhaustive canonical merge");

  aggregate_ = state.candidate;
  output_valid_ = true;
  telemetry_.counter_rng_streams = aggregate_.counter_rng_streams;
  telemetry_.emitted_photons = aggregate_.emitted_photons;
  telemetry_.photon_lumi_opportunities =
      aggregate_.photon_lumi_opportunities;
  telemetry_.full_result_terms = aggregate_.full_result_terms;
  telemetry_.lumi_candidates_seen = aggregate_.lumi_candidates_seen;
  telemetry_.lumi_candidates_retained = aggregate_.lumi_candidates_retained;
  telemetry_.lumi_reservoir_digest = aggregate_.lumi_reservoir_digest;
  telemetry_.output_valid = true;
  RunResult result;
  result.aggregate = aggregate_;
  result.telemetry = telemetry_;
  return result;
}

}  // namespace legacy_host_wave
}  // namespace feebl
