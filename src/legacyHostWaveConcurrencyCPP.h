#ifndef FEEBL_LEGACY_HOST_WAVE_CONCURRENCY_CPP_H
#define FEEBL_LEGACY_HOST_WAVE_CONCURRENCY_CPP_H

#include "modernCpuExecutorCPP.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace feebl {
namespace legacy_host_wave {

// This module is an integration boundary, not permission to call the legacy
// FEEBL::make_step concurrently.  The latter owns shared GRID scratch, a shared
// variable-draw RNG, RESULTS, luminosity heaps and output sinks.  A production
// adapter must describe every one of those capabilities here and pass the
// fail-closed audit before it may use the typed executor below.
constexpr std::uint32_t kContractSchemaVersion = 1;
constexpr std::uint32_t kResultLaneSchemaVersion = 1;

enum class RngPolicy : std::uint8_t {
  None = 0,
  LegacyShared = 1,
  CounterPerEncounter = 2
};

enum class ResultPolicy : std::uint8_t {
  SharedLegacy = 0,
  FineLuminosityLaneOnly = 1,
  EncounterLocalFull = 2
};

enum class EmissionPolicy : std::uint8_t {
  None = 0,
  SharedLegacy = 1,
  EncounterLocalCanonicalMerge = 2
};

enum class ContractTier : std::uint8_t {
  Rejected = 0,
  OrbitBmtFineLuminosityOnly = 1,
  FcczPrimaryBeamstrahlungPhotonFullEeLuminosity = 2
};

enum class Blocker : std::uint64_t {
  None = 0,
  MissingPrefetchedFields = 1ULL << 0,
  UnsupportedGridContract = 1ULL << 1,
  ForceSymmetric = 1ULL << 2,
  ExtraGrids = 1ULL << 3,
  ElectronSampling = 1ULL << 4,
  SharedOrUnqualifiedRng = 1ULL << 5,
  Beamstrahlung = 1ULL << 6,
  StochasticSpinFlip = 1ULL << 7,
  FullElectronLuminosity = 1ULL << 8,
  NativeLuminosityOutput = 1ULL << 9,
  PrimaryPhotons = 1ULL << 10,
  PhotonLuminosity = 1ULL << 11,
  PairOrStrongFieldProduction = 1ULL << 12,
  TrackedSecondaries = 1ULL << 13,
  EventGenerators = 1ULL << 14,
  SharedResults = 1ULL << 15,
  SharedEmissions = 1ULL << 16,
  SharedOutputSink = 1ULL << 17,
  MissingEncounterLocalGrid = 1ULL << 18,
  MissingEncounterLocalTimers = 1ULL << 19,
  MissingCanonicalMerge = 1ULL << 20,
  MissingExclusiveSliceProof = 1ULL << 21,
  NonFiniteConfiguration = 1ULL << 22,
  UnsupportedContractSchema = 1ULL << 23,
  MissingEncounterLocalResults = 1ULL << 24,
  MissingCounterRngV2Streams = 1ULL << 25,
  MissingStableParticleIds = 1ULL << 26,
  MissingFullResultsMerge = 1ULL << 27,
  MissingBoundedReservoirMerge = 1ULL << 28,
  MissingThreadSafePhotonInsertion = 1ULL << 29,
  InvalidReservoirCapacity = 1ULL << 30,
  UnsupportedContractTier = 1ULL << 31
};

constexpr std::uint64_t blocker_mask(Blocker blocker) noexcept {
  return static_cast<std::uint64_t>(blocker);
}

struct FeatureSet {
  std::uint32_t schema_version = kContractSchemaVersion;
  ContractTier requested_tier = ContractTier::Rejected;

  // Field and geometry boundary.
  bool prefetched_fields = false;
  int integration_method = 0;
  int electron_distribution_rho = 0;
  int interpolation = 0;
  bool force_symmetric = false;
  int extra_grids = 0;

  // A zero ratio deliberately selects rho-overlap (fine luminosity) only.  A
  // positive ratio makes legacy particle-pair sampling consume the shared RNG.
  double electron_ratio = 0.0;

  // Primary/spin/radiation physics.  Deterministic BMT rotation is admitted;
  // stochastic Sokolov-Ternov flips and radiation are not.
  bool beamstrahlung = false;
  bool stochastic_spin_flip = false;
  bool bmt_precession = false;
  bool do_prod = false;
  bool do_trident = false;
  bool do_coherent = false;

  // Photon and secondary families.
  bool do_photons_beam1 = false;
  bool do_photons_beam2 = false;
  bool load_photons = false;
  bool do_photon_luminosity = false;
  bool do_pairs = false;
  bool do_hadrons = false;
  bool do_compton = false;
  bool do_muons = false;
  bool track_pairs = false;
  bool track_muons = false;

  // Electron-luminosity heaps and optional event generators/sinks.
  bool do_full_electron_luminosity = false;
  bool store_native_luminosity = false;
  bool do_bhabhas = false;
  bool do_jets = false;
  bool do_isr = false;
  bool has_external_output_sink = false;

  // Adapter properties, not merely card settings.  They must be asserted by
  // the implementation that owns the task-local objects.
  bool encounter_local_grid_scratch = false;
  bool encounter_local_results = false;
  bool encounter_local_timers = false;
  bool deterministic_canonical_merge = false;
  bool exclusive_slices_within_wave = false;
  bool counter_rng_v2_derived_streams = false;
  bool stable_particle_ids = false;
  bool deterministic_full_results_merge = false;
  bool canonical_bounded_lumi_reservoir_merge = false;
  bool thread_safe_distinct_slice_photon_insertion = false;
  std::uint64_t lumi_event_reservoir_capacity = 0;
  RngPolicy rng_policy = RngPolicy::LegacyShared;
  ResultPolicy result_policy = ResultPolicy::SharedLegacy;
  EmissionPolicy emission_policy = EmissionPolicy::SharedLegacy;

  static FeatureSet orbit_bmt_fine_luminosity_only() noexcept;
  static FeatureSet
  fccz_primary_beamstrahlung_photon_full_ee_luminosity() noexcept;
};

struct EligibilityReport {
  ContractTier tier = ContractTier::Rejected;
  std::uint64_t blockers = 0;
  std::vector<std::string> reasons;

  bool eligible() const noexcept {
    return tier != ContractTier::Rejected && blockers == 0 && reasons.empty();
  }
  bool has(Blocker blocker) const noexcept {
    return (blockers & blocker_mask(blocker)) != 0;
  }
  const std::string& first_reason() const noexcept;
};

EligibilityReport audit(const FeatureSet& features);
const char* contract_tier_name(ContractTier tier) noexcept;

// One fixed-size lane is owned by one encounter ID.  It deliberately contains
// no pointer, vector, RNG, heap, or output stream.  A callback may mutate only
// the two exclusive beam slices named by its EncounterNode and then publishes
// this value.  No legacy shared state is reachable through the lane.
struct EncounterResultLane {
  std::uint32_t schema_version = kResultLaneSchemaVersion;
  std::uint32_t contract_tier =
      static_cast<std::uint32_t>(ContractTier::Rejected);
  modern::EncounterId encounter = modern::kInvalidEncounter;
  modern::WaveIndex logical_tick = 0;
  double fine_lumi_dot = 0.0;
  double fine_luminosity = 0.0;
  double ee_luminosity = 0.0;
  double ee_high_luminosity = 0.0;
  double ecm_weighted_sum = 0.0;
  double ecm_squared_weighted_sum = 0.0;
  double photon_luminosity_eg = 0.0;
  double photon_luminosity_ge = 0.0;
  double photon_luminosity_gg = 0.0;
  double beam_energy_loss = 0.0;
  double emitted_photon_energy = 0.0;
  double maximum_upsilon = 0.0;
  double luminosity_weighted_helicity1 = 0.0;
  double luminosity_weighted_helicity2 = 0.0;
  double luminosity_helicity_weight = 0.0;
  std::uint64_t beam1_in = 0;
  std::uint64_t beam1_out = 0;
  std::uint64_t beam2_in = 0;
  std::uint64_t beam2_out = 0;
  std::uint64_t moved_beam1 = 0;
  std::uint64_t moved_beam2 = 0;
  std::uint64_t electron_pair_evaluations = 0;
  std::uint64_t photon_lumi_opportunities = 0;
  std::uint64_t emitted_photons = 0;
  std::uint64_t counter_rng_streams = 0;
  std::uint64_t full_result_terms = 0;
  std::uint64_t lumi_candidates_seen = 0;
  std::uint64_t lumi_candidates_retained = 0;
  std::uint64_t lumi_reservoir_digest = 0;
  std::uint64_t emission_records = 0;
};

struct LaneValidation {
  bool ok = false;
  std::string message;
};

LaneValidation validate_lane(const EncounterResultLane& lane,
                             const modern::EncounterNode& node,
                             const modern::EncounterLogicalTime& logical_time,
                             ContractTier tier =
                                 ContractTier::OrbitBmtFineLuminosityOnly,
                             std::uint64_t lumi_reservoir_capacity = 0);

struct Aggregate {
  ContractTier tier = ContractTier::Rejected;
  std::uint64_t encounters = 0;
  double fine_lumi_dot_sum = 0.0;
  double fine_luminosity_sum = 0.0;
  double ee_luminosity = 0.0;
  double ee_high_luminosity = 0.0;
  double ecm_weighted_sum = 0.0;
  double ecm_squared_weighted_sum = 0.0;
  double photon_luminosity_eg = 0.0;
  double photon_luminosity_ge = 0.0;
  double photon_luminosity_gg = 0.0;
  double beam_energy_loss = 0.0;
  double emitted_photon_energy = 0.0;
  double maximum_upsilon = 0.0;
  double luminosity_weighted_helicity1 = 0.0;
  double luminosity_weighted_helicity2 = 0.0;
  double luminosity_helicity_weight = 0.0;
  std::uint64_t beam1_in = 0;
  std::uint64_t beam1_out = 0;
  std::uint64_t beam2_in = 0;
  std::uint64_t beam2_out = 0;
  std::uint64_t moved_beam1 = 0;
  std::uint64_t moved_beam2 = 0;
  std::uint64_t electron_pair_evaluations = 0;
  std::uint64_t photon_lumi_opportunities = 0;
  std::uint64_t emitted_photons = 0;
  std::uint64_t counter_rng_streams = 0;
  std::uint64_t full_result_terms = 0;
  std::uint64_t lumi_candidates_seen = 0;
  std::uint64_t lumi_candidates_retained = 0;
  std::uint64_t lumi_reservoir_digest = 0;
  std::uint64_t emission_records = 0;
};

struct Telemetry {
  std::uint32_t schema_version = kContractSchemaVersion;
  ContractTier tier = ContractTier::Rejected;
  std::uint64_t waves = 0;
  std::uint64_t planned_encounters = 0;
  std::uint64_t attempted_encounters = 0;
  std::uint64_t completed_encounters = 0;
  std::uint64_t merged_encounters = 0;
  std::uint32_t requested_workers = 0;
  std::uint32_t actual_workers = 0;
  std::uint32_t context_capacity = 0;
  std::uint32_t context_high_water = 0;
  std::uint32_t parallel_regions = 0;
  std::uint64_t result_lane_bytes = 0;
  std::uint64_t emission_lane_bytes = 0;
  std::uint64_t callback_cpu_nanoseconds = 0;
  std::uint64_t merge_nanoseconds = 0;
  std::uint64_t execute_wall_nanoseconds = 0;
  std::uint64_t counter_rng_streams = 0;
  std::uint64_t emitted_photons = 0;
  std::uint64_t photon_lumi_opportunities = 0;
  std::uint64_t full_result_terms = 0;
  std::uint64_t lumi_candidates_seen = 0;
  std::uint64_t lumi_candidates_retained = 0;
  std::uint64_t lumi_reservoir_digest = 0;
  std::uint64_t lumi_event_reservoir_capacity = 0;
  bool counter_rng_v2_contract = false;
  bool deterministic_full_results_merge_contract = false;
  bool canonical_bounded_reservoir_merge_contract = false;
  bool distinct_slice_photon_insertion_contract = false;
  bool used_serial_fallback = false;
  bool output_valid = false;
};

struct ExecutorConfig {
  modern::CpuExecutionKind execution =
      modern::CpuExecutionKind::SerialReference;
  modern::CpuScheduleStrategy strategy =
      modern::CpuScheduleStrategy::Static;
  modern::CpuContextResetPolicy context_reset =
      modern::CpuContextResetPolicy::ZeroBeforeCallback;
  std::uint32_t worker_count = 1;
  std::uint32_t context_count = 1;
  std::uint64_t context_bytes = 0;
  bool allow_serial_fallback_without_openmp = false;
  modern::CpuExecutorLimits limits;
};

using ComputeCallback = void (*)(
    const modern::EncounterNode& node,
    const modern::EncounterLogicalTime& logical_time,
    modern::StepContext& context,
    EncounterResultLane& result,
    void* user_data);

struct RunResult {
  Aggregate aggregate;
  Telemetry telemetry;
};

class HostWaveError : public std::runtime_error {
 public:
  explicit HostWaveError(
      const std::string& message,
      modern::EncounterId encounter = modern::kInvalidEncounter)
      : std::runtime_error(message), encounter_(encounter) {}
  modern::EncounterId encounter() const noexcept { return encounter_; }

 private:
  modern::EncounterId encounter_;
};

// Typed adapter over CpuEncounterExecutor.  The scheduler supplies wave
// barriers and exclusive contexts; this layer supplies the physics contract,
// typed encounter lanes, complete validation, transactional aggregate merge,
// and provenance telemetry.  It intentionally has no production FEEBL dispatch
// until a real adapter can satisfy FeatureSet without lying about shared state.
class HostWaveExecutor {
 public:
  HostWaveExecutor(const modern::EncounterDAG& dag,
                   const FeatureSet& features,
                   const ExecutorConfig& config);
  ~HostWaveExecutor();

  HostWaveExecutor(const HostWaveExecutor&) = delete;
  HostWaveExecutor& operator=(const HostWaveExecutor&) = delete;

  RunResult execute(ComputeCallback callback, void* user_data);

  bool output_valid() const noexcept { return output_valid_; }
  const Aggregate& aggregate() const noexcept { return aggregate_; }
  const Telemetry& telemetry() const noexcept { return telemetry_; }
  const EligibilityReport& eligibility() const noexcept { return eligibility_; }

 private:
  EligibilityReport eligibility_;
  FeatureSet features_;
  std::unique_ptr<modern::CpuEncounterExecutor> executor_;
  Aggregate aggregate_;
  Telemetry telemetry_;
  bool output_valid_ = false;
};

}  // namespace legacy_host_wave
}  // namespace feebl

#endif  // FEEBL_LEGACY_HOST_WAVE_CONCURRENCY_CPP_H
