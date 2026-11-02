#include "legacyHostWaveConcurrencyCPP.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace host_wave = feebl::legacy_host_wave;
namespace modern = feebl::modern;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (condition)
    std::printf("  PASS: %s\n", message);
  else {
    std::printf("  FAIL: %s\n", message);
    ++failures;
  }
}

template <class Callback>
bool throws_host_wave(Callback&& callback) {
  try {
    callback();
  } catch (const host_wave::HostWaveError&) {
    return true;
  }
  return false;
}

host_wave::FeatureSet eligible_features() {
  return host_wave::FeatureSet::orbit_bmt_fine_luminosity_only();
}

void test_eligibility_contract() {
  std::printf("\nFail-closed feature contract\n");
  host_wave::FeatureSet safe = eligible_features();
  check(host_wave::audit(safe).eligible(),
        "explicit orbit/BMT/fine-luminosity tier is admitted");
  safe.bmt_precession = true;
  check(host_wave::audit(safe).eligible(),
        "deterministic BMT transport remains inside the narrow tier");

  struct RejectionProbe {
    host_wave::Blocker expected;
    const char* label;
    std::function<void(host_wave::FeatureSet&)> enable;
  };
  const std::vector<RejectionProbe> probes = {
      {host_wave::Blocker::MissingPrefetchedFields, "missing fields",
       [](host_wave::FeatureSet& f) { f.prefetched_fields = false; }},
      {host_wave::Blocker::UnsupportedGridContract, "non-CIC grid",
       [](host_wave::FeatureSet& f) { f.interpolation = 1; }},
      {host_wave::Blocker::ForceSymmetric, "symmetric shared rho",
       [](host_wave::FeatureSet& f) { f.force_symmetric = true; }},
      {host_wave::Blocker::ExtraGrids, "extra grids",
       [](host_wave::FeatureSet& f) { f.extra_grids = 1; }},
      {host_wave::Blocker::ElectronSampling, "electron sampling",
       [](host_wave::FeatureSet& f) { f.electron_ratio = 1.0; }},
      {host_wave::Blocker::SharedOrUnqualifiedRng, "shared RNG",
       [](host_wave::FeatureSet& f) {
         f.rng_policy = host_wave::RngPolicy::LegacyShared;
       }},
      {host_wave::Blocker::Beamstrahlung, "beamstrahlung",
       [](host_wave::FeatureSet& f) { f.beamstrahlung = true; }},
      {host_wave::Blocker::StochasticSpinFlip, "stochastic spin flip",
       [](host_wave::FeatureSet& f) { f.stochastic_spin_flip = true; }},
      {host_wave::Blocker::FullElectronLuminosity, "full electron lumi",
       [](host_wave::FeatureSet& f) {
         f.do_full_electron_luminosity = true;
       }},
      {host_wave::Blocker::NativeLuminosityOutput, "native lumi output",
       [](host_wave::FeatureSet& f) { f.store_native_luminosity = true; }},
      {host_wave::Blocker::PrimaryPhotons, "primary photons",
       [](host_wave::FeatureSet& f) { f.do_photons_beam1 = true; }},
      {host_wave::Blocker::PhotonLuminosity, "photon luminosity",
       [](host_wave::FeatureSet& f) { f.do_photon_luminosity = true; }},
      {host_wave::Blocker::PairOrStrongFieldProduction, "pair production",
       [](host_wave::FeatureSet& f) { f.do_pairs = true; }},
      {host_wave::Blocker::TrackedSecondaries, "tracked secondaries",
       [](host_wave::FeatureSet& f) { f.track_pairs = true; }},
      {host_wave::Blocker::EventGenerators, "event generators",
       [](host_wave::FeatureSet& f) { f.do_jets = true; }},
      {host_wave::Blocker::SharedResults, "shared results",
       [](host_wave::FeatureSet& f) {
         f.result_policy = host_wave::ResultPolicy::SharedLegacy;
       }},
      {host_wave::Blocker::SharedEmissions, "shared emissions",
       [](host_wave::FeatureSet& f) {
         f.emission_policy = host_wave::EmissionPolicy::SharedLegacy;
       }},
      {host_wave::Blocker::SharedOutputSink, "shared output",
       [](host_wave::FeatureSet& f) { f.has_external_output_sink = true; }},
      {host_wave::Blocker::MissingEncounterLocalGrid, "shared grid scratch",
       [](host_wave::FeatureSet& f) {
         f.encounter_local_grid_scratch = false;
       }},
      {host_wave::Blocker::MissingEncounterLocalTimers, "shared timers",
       [](host_wave::FeatureSet& f) { f.encounter_local_timers = false; }},
      {host_wave::Blocker::MissingCanonicalMerge, "unordered merge",
       [](host_wave::FeatureSet& f) {
         f.deterministic_canonical_merge = false;
       }},
      {host_wave::Blocker::MissingExclusiveSliceProof, "unproven slices",
       [](host_wave::FeatureSet& f) {
         f.exclusive_slices_within_wave = false;
       }},
      {host_wave::Blocker::NonFiniteConfiguration, "non-finite ratio",
       [](host_wave::FeatureSet& f) {
         f.electron_ratio = std::numeric_limits<double>::quiet_NaN();
       }},
      {host_wave::Blocker::UnsupportedContractSchema, "schema mismatch",
       [](host_wave::FeatureSet& f) { f.schema_version = 999; }}};

  bool every_probe_rejected = true;
  for (const RejectionProbe& probe : probes) {
    host_wave::FeatureSet features = eligible_features();
    probe.enable(features);
    const host_wave::EligibilityReport report = host_wave::audit(features);
    every_probe_rejected = every_probe_rejected && !report.eligible() &&
                           report.has(probe.expected) &&
                           !report.first_reason().empty();
  }
  check(every_probe_rejected,
        "every audited shared-state feature has a stable fail-closed blocker");

  host_wave::FeatureSet fccz = eligible_features();
  fccz.electron_ratio = 1.0;
  fccz.rng_policy = host_wave::RngPolicy::LegacyShared;
  fccz.beamstrahlung = true;
  fccz.do_photons_beam1 = true;
  fccz.do_photon_luminosity = true;
  fccz.do_full_electron_luminosity = true;
  fccz.store_native_luminosity = true;
  fccz.result_policy = host_wave::ResultPolicy::SharedLegacy;
  fccz.emission_policy = host_wave::EmissionPolicy::SharedLegacy;
  const host_wave::EligibilityReport report = host_wave::audit(fccz);
  check(!report.eligible() && report.reasons.size() >= 8,
        "the supplied FCC-Z feature combination is explicitly rejected");
}

struct CallbackState {
  explicit CallbackState(std::size_t beam1_slices,
                         std::size_t beam2_slices)
      : active1(new std::atomic<int>[beam1_slices]),
        active2(new std::atomic<int>[beam2_slices]),
        visits1(new std::atomic<std::uint64_t>[beam1_slices]),
        visits2(new std::atomic<std::uint64_t>[beam2_slices]),
        beam1_count(beam1_slices), beam2_count(beam2_slices) {
    reset();
  }

  void reset() {
    for (std::size_t i = 0; i < beam1_count; ++i) {
      active1[i].store(0);
      visits1[i].store(0);
    }
    for (std::size_t i = 0; i < beam2_count; ++i) {
      active2[i].store(0);
      visits2[i].store(0);
    }
    exclusivity_ok.store(true);
    context_alignment_ok.store(true);
    fail_encounter = modern::kInvalidEncounter;
    invalid_lane_encounter = modern::kInvalidEncounter;
    overflow_mode = false;
  }

  std::unique_ptr<std::atomic<int>[]> active1;
  std::unique_ptr<std::atomic<int>[]> active2;
  std::unique_ptr<std::atomic<std::uint64_t>[]> visits1;
  std::unique_ptr<std::atomic<std::uint64_t>[]> visits2;
  std::size_t beam1_count;
  std::size_t beam2_count;
  std::atomic<bool> exclusivity_ok{true};
  std::atomic<bool> context_alignment_ok{true};
  modern::EncounterId fail_encounter = modern::kInvalidEncounter;
  modern::EncounterId invalid_lane_encounter = modern::kInvalidEncounter;
  bool overflow_mode = false;
};

void physics_callback(const modern::EncounterNode& node,
                      const modern::EncounterLogicalTime& logical_time,
                      modern::StepContext& context,
                      host_wave::EncounterResultLane& lane,
                      void* opaque) {
  CallbackState& state = *static_cast<CallbackState*>(opaque);
  const std::size_t first = static_cast<std::size_t>(node.beam1_slice);
  const std::size_t second = static_cast<std::size_t>(node.beam2_slice);
  const bool first_busy = state.active1[first].fetch_add(1) != 0;
  const bool second_busy = state.active2[second].fetch_add(1) != 0;
  if (first_busy || second_busy)
    state.exclusivity_ok.store(false, std::memory_order_relaxed);

  const modern::MutableByteLane scratch = context.scratch();
  if (!scratch.empty() &&
      reinterpret_cast<std::uintptr_t>(scratch.data()) %
              modern::kCpuContextAlignment !=
          0)
    state.context_alignment_ok.store(false, std::memory_order_relaxed);

  // Deliberately vary completion order without tying physics to a worker ID.
  for (std::uint64_t spin = 0; spin < (node.id * 17U) % 31U; ++spin)
    std::this_thread::yield();

  state.visits1[first].fetch_add(1, std::memory_order_relaxed);
  state.visits2[second].fetch_add(1, std::memory_order_relaxed);

  if (state.overflow_mode) {
    lane.beam1_in = node.id == 0
                        ? std::numeric_limits<std::uint64_t>::max()
                        : 1;
    lane.moved_beam1 = lane.beam1_in;
  } else {
    lane.beam1_in = static_cast<std::uint64_t>(node.beam1_slice) + 1U;
    lane.beam1_out = 1U;
    lane.moved_beam1 = lane.beam1_in + lane.beam1_out;
  }
  lane.beam2_in = state.overflow_mode
                      ? 0U
                      : static_cast<std::uint64_t>(node.beam2_slice) + 2U;
  lane.beam2_out = state.overflow_mode ? 0U : 2U;
  lane.moved_beam2 = lane.beam2_in + lane.beam2_out;
  lane.fine_lumi_dot = 0.25 * static_cast<double>(node.id + 1U);
  lane.fine_luminosity =
      lane.fine_lumi_dot * static_cast<double>(logical_time.tick + 1U);

  state.active1[first].fetch_sub(1);
  state.active2[second].fetch_sub(1);

  if (node.id == state.invalid_lane_encounter)
    lane.fine_lumi_dot = std::numeric_limits<double>::quiet_NaN();
  if (node.id == state.fail_encounter)
    throw std::runtime_error("injected encounter failure");
}

#ifdef _OPENMP
bool aggregate_equal(const host_wave::Aggregate& first,
                     const host_wave::Aggregate& second) {
  return first.encounters == second.encounters &&
         first.fine_lumi_dot_sum == second.fine_lumi_dot_sum &&
         first.fine_luminosity_sum == second.fine_luminosity_sum &&
         first.beam1_in == second.beam1_in &&
         first.beam1_out == second.beam1_out &&
         first.beam2_in == second.beam2_in &&
         first.beam2_out == second.beam2_out &&
         first.moved_beam1 == second.moved_beam1 &&
         first.moved_beam2 == second.moved_beam2 &&
         first.emission_records == second.emission_records;
}
#endif

host_wave::RunResult run(const modern::EncounterDAG& dag,
                         modern::CpuExecutionKind execution,
                         std::uint32_t workers,
                         CallbackState* state) {
  host_wave::ExecutorConfig config;
  config.execution = execution;
  config.worker_count = workers;
  config.context_count = workers;
  config.context_bytes = 64;
  host_wave::HostWaveExecutor executor(
      dag, eligible_features(), config);
  return executor.execute(physics_callback, state);
}

void test_deterministic_execution() {
  std::printf("\nTyped execution, barriers, merge and telemetry\n");
  const modern::EncounterDAG dag(8, 8, 2);
  CallbackState serial_state(8, 8);
  const host_wave::RunResult serial = run(
      dag, modern::CpuExecutionKind::SerialReference, 1, &serial_state);
  check(serial.telemetry.output_valid &&
            serial.telemetry.planned_encounters == 128 &&
            serial.telemetry.completed_encounters == 128 &&
            serial.telemetry.merged_encounters == 128 &&
            serial.telemetry.result_lane_bytes ==
                sizeof(host_wave::EncounterResultLane) &&
            serial.telemetry.emission_lane_bytes == 0,
        "serial reference reports exhaustive typed-lane publication");
  check(serial_state.exclusivity_ok.load() &&
            serial_state.context_alignment_ok.load(),
        "serial reference respects slice and aligned-context contracts");

#ifdef _OPENMP
  bool all_parallel_equal = true;
  bool all_parallel_telemetry = true;
  for (std::uint32_t workers : {2U, 4U}) {
    CallbackState parallel_state(8, 8);
    const host_wave::RunResult parallel = run(
        dag, modern::CpuExecutionKind::OpenMP, workers, &parallel_state);
    all_parallel_equal = all_parallel_equal &&
                         aggregate_equal(serial.aggregate,
                                         parallel.aggregate) &&
                         parallel_state.exclusivity_ok.load() &&
                         parallel_state.context_alignment_ok.load();
    all_parallel_telemetry = all_parallel_telemetry &&
        parallel.telemetry.actual_workers == workers &&
        parallel.telemetry.context_high_water <= workers &&
        parallel.telemetry.parallel_regions == 1 &&
        !parallel.telemetry.used_serial_fallback &&
        parallel.telemetry.callback_cpu_nanoseconds > 0 &&
        parallel.telemetry.execute_wall_nanoseconds > 0;
  }
  check(all_parallel_equal,
        "1/2/4-worker runs are bit-identical under completion jitter");
  check(all_parallel_telemetry,
        "parallel telemetry proves actual team, contexts and timed callbacks");
#else
  std::printf("  SKIP: OpenMP worker-identity checks (OpenMP not enabled)\n");
#endif
}

void test_failure_and_recovery() {
  std::printf("\nTransactional failure and recovery\n");
  const modern::EncounterDAG dag(4, 4, 1);
  host_wave::ExecutorConfig config;
  config.execution = modern::CpuExecutionKind::SerialReference;
  config.worker_count = 1;
  config.context_count = 1;
  host_wave::HostWaveExecutor executor(dag, eligible_features(), config);
  CallbackState state(4, 4);
  state.fail_encounter = 5;
  check(throws_host_wave([&] { executor.execute(physics_callback, &state); }) &&
            !executor.output_valid() &&
            !executor.telemetry().output_valid &&
            executor.telemetry().merged_encounters == 0,
        "callback failure publishes no aggregate lanes");

  state.reset();
  const host_wave::RunResult recovered =
      executor.execute(physics_callback, &state);
  check(recovered.telemetry.output_valid && recovered.aggregate.encounters == 16,
        "the bounded executor recovers after a failed transaction");

  state.reset();
  state.invalid_lane_encounter = 3;
  check(throws_host_wave([&] { executor.execute(physics_callback, &state); }) &&
            !executor.output_valid() &&
            executor.telemetry().merged_encounters == 0,
        "non-finite encounter output fails before canonical publication");
}

void test_merge_overflow() {
  std::printf("\nTransactional merge overflow\n");
  const modern::EncounterDAG dag(1, 2, 1);
  host_wave::ExecutorConfig config;
  host_wave::HostWaveExecutor executor(dag, eligible_features(), config);
  CallbackState state(1, 2);
  state.overflow_mode = true;
  host_wave::HostWaveError captured("placeholder");
  bool threw = false;
  try {
    executor.execute(physics_callback, &state);
  } catch (const host_wave::HostWaveError& error) {
    threw = true;
    captured = error;
  }
  check(threw && !executor.output_valid() &&
            executor.aggregate().encounters == 0 &&
            executor.telemetry().merged_encounters == 1 &&
            captured.encounter() == 1,
        "overflow rejects the candidate aggregate without partial publication");
}

void test_constructor_refuses_unsafe_contract() {
  std::printf("\nConstructor preflight\n");
  const modern::EncounterDAG dag(2, 2, 1);
  host_wave::FeatureSet unsafe = eligible_features();
  unsafe.beamstrahlung = true;
  host_wave::ExecutorConfig config;
  check(throws_host_wave([&] {
          host_wave::HostWaveExecutor executor(dag, unsafe, config);
        }),
        "executor construction refuses an unsafe physics feature set");
}

}  // namespace

int main() {
  test_eligibility_contract();
  test_deterministic_execution();
  test_failure_and_recovery();
  test_merge_overflow();
  test_constructor_refuses_unsafe_contract();
  if (failures != 0) {
    std::fprintf(stderr, "legacy host-wave concurrency failures: %d\n",
                 failures);
    return 1;
  }
  std::printf("\nAll legacy host-wave concurrency tests passed.\n");
  return 0;
}
