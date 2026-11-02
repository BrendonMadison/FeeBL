#include "encounterDAGCPP.h"
#include "modernCpuExecutorCPP.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using feebl::modern::ConstByteLane;
using feebl::modern::CpuEncounterExecutor;
using feebl::modern::CpuContextResetPolicy;
using feebl::modern::CpuExecutionKind;
using feebl::modern::CpuExecutionReport;
using feebl::modern::CpuExecutorConfig;
using feebl::modern::CpuExecutorError;
using feebl::modern::CpuScheduleStrategy;
using feebl::modern::EncounterDAG;
using feebl::modern::EncounterId;
using feebl::modern::EncounterLogicalTime;
using feebl::modern::EncounterNode;
using feebl::modern::MutableByteLane;
using feebl::modern::StepContext;
using feebl::modern::kCpuContextAlignment;
using feebl::modern::kInvalidEncounter;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-76s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

template <class Function>
bool throws(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

using ResultValue = std::array<std::uint64_t, 6>;
using EmissionValue = std::array<std::uint64_t, 3>;

std::uint64_t mix(std::uint64_t state, std::uint64_t value) noexcept {
  state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
  state ^= state >> 29U;
  state *= 0xbf58476d1ce4e5b9ULL;
  return state ^ (state >> 31U);
}

struct CallbackState {
  explicit CallbackState(const EncounterDAG& graph,
                          std::uint32_t context_capacity = 1)
      : dag(graph),
        beam1(static_cast<std::size_t>(graph.beam1_slice_count())),
        beam2(static_cast<std::size_t>(graph.beam2_slice_count())),
        seen(static_cast<std::size_t>(graph.encounter_count()), 0),
        beam1_active(graph.beam1_slice_count() == 0
                         ? nullptr
                         : new std::atomic<int>[static_cast<std::size_t>(
                               graph.beam1_slice_count())]),
        beam2_active(graph.beam2_slice_count() == 0
                         ? nullptr
                         : new std::atomic<int>[static_cast<std::size_t>(
                               graph.beam2_slice_count())]),
        context_active(context_capacity == 0
                            ? nullptr
                            : new std::atomic<int>[context_capacity]),
        context_count(context_capacity) {
    for (std::size_t i = 0; i < beam1.size(); ++i) {
      beam1[i] = 0x100000000ULL + i;
      beam1_active[i].store(0);
    }
    for (std::size_t i = 0; i < beam2.size(); ++i) {
      beam2[i] = 0x200000000ULL + i;
      beam2_active[i].store(0);
    }
    for (std::uint32_t context = 0; context < context_count; ++context)
      context_active[context].store(0);
  }

  const EncounterDAG& dag;
  std::vector<std::uint64_t> beam1;
  std::vector<std::uint64_t> beam2;
  std::vector<std::uint8_t> seen;
  std::unique_ptr<std::atomic<int>[]> beam1_active;
  std::unique_ptr<std::atomic<int>[]> beam2_active;
  std::unique_ptr<std::atomic<int>[]> context_active;
  std::uint32_t context_count = 0;
  std::atomic<bool> exclusivity_ok{true};
  std::atomic<bool> context_exclusivity_ok{true};
  std::atomic<bool> logical_time_ok{true};
  std::atomic<bool> scratch_reset_ok{true};
  std::atomic<bool> context_alignment_ok{true};
  std::atomic<unsigned> max_parallel_level{0};
  std::atomic<std::uint64_t> callback_count{0};
  EncounterId throw_at = kInvalidEncounter;
  EncounterId throw_at_secondary = kInvalidEncounter;
  bool jitter = false;
};

#ifdef _OPENMP
void update_max(std::atomic<unsigned>& target, unsigned value) noexcept {
  unsigned observed = target.load(std::memory_order_relaxed);
  while (observed < value &&
         !target.compare_exchange_weak(observed, value,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}
#endif

void encounter_callback(const EncounterNode& node,
                        const EncounterLogicalTime& time,
                        StepContext& context,
                        MutableByteLane result_lane,
                        MutableByteLane emission_lane,
                        void* opaque) {
  CallbackState& state = *static_cast<CallbackState*>(opaque);
  state.callback_count.fetch_add(1, std::memory_order_relaxed);
  if (time.encounter != node.id || time.tick != node.wave ||
      time.wave != node.wave || time.diagonal != node.diagonal ||
      time.substep != node.substep ||
      context.current_encounter() != node.id)
    state.logical_time_ok.store(false, std::memory_order_relaxed);

#ifdef _OPENMP
  update_max(state.max_parallel_level,
             static_cast<unsigned>(omp_get_level()));
#endif

  MutableByteLane scratch = context.scratch();
  if (!scratch.empty() &&
      reinterpret_cast<std::uintptr_t>(scratch.data()) %
              kCpuContextAlignment !=
          0)
    state.context_alignment_ok.store(false, std::memory_order_relaxed);
  if (context.id() >= state.context_count)
    state.context_exclusivity_ok.store(false, std::memory_order_relaxed);
  const bool context_was_active =
      context.id() < state.context_count &&
      state.context_active[context.id()].fetch_add(
          1, std::memory_order_acq_rel) != 0;
  if (context_was_active)
    state.context_exclusivity_ok.store(false, std::memory_order_relaxed);
  for (std::size_t byte = 0; byte < scratch.size(); ++byte) {
    if (scratch.data()[byte] != std::byte{0})
      state.scratch_reset_ok.store(false, std::memory_order_relaxed);
  }
  if (scratch.size() >= sizeof(std::uint64_t)) scratch.store(node.id);

  const std::size_t i = static_cast<std::size_t>(node.beam1_slice);
  const std::size_t j = static_cast<std::size_t>(node.beam2_slice);
  const bool beam1_was_active =
      state.beam1_active[i].fetch_add(1, std::memory_order_acq_rel) != 0;
  const bool beam2_was_active =
      state.beam2_active[j].fetch_add(1, std::memory_order_acq_rel) != 0;
  if (beam1_was_active || beam2_was_active)
    state.exclusivity_ok.store(false, std::memory_order_relaxed);

  if (state.jitter) {
    const auto delay = static_cast<int>((node.id * 17U + 3U) % 11U);
    std::this_thread::sleep_for(std::chrono::microseconds(20 * delay));
  }

  state.seen[static_cast<std::size_t>(node.id)]++;
  state.beam1[i] = mix(state.beam1[i], node.id ^ (time.tick << 32U));
  state.beam2[j] = mix(state.beam2[j], node.id ^ (time.substep << 48U));

  const ResultValue result = {
      {node.id, time.tick, node.beam1_slice, node.beam2_slice,
       state.beam1[i], state.beam2[j]}};
  const EmissionValue emission =
      {{node.id, mix(node.id, time.wave), mix(time.diagonal, time.substep)}};
  result_lane.store(result);
  emission_lane.store(emission);

  state.beam2_active[j].fetch_sub(1, std::memory_order_acq_rel);
  state.beam1_active[i].fetch_sub(1, std::memory_order_acq_rel);
  if (context.id() < state.context_count)
    state.context_active[context.id()].fetch_sub(1, std::memory_order_acq_rel);
  if (node.id == state.throw_at || node.id == state.throw_at_secondary)
    throw std::runtime_error("injected failure");
}

struct MergeState {
  explicit MergeState(std::size_t count) { ids.reserve(count); }
  std::vector<EncounterId> ids;
  bool lanes_ok = true;
};

void merge_callback(const EncounterNode& node,
                    const EncounterLogicalTime& time,
                    ConstByteLane result_lane,
                    ConstByteLane emission_lane,
                    void* opaque) noexcept {
  MergeState& state = *static_cast<MergeState*>(opaque);
  const ResultValue result = result_lane.load<ResultValue>();
  const EmissionValue emission = emission_lane.load<EmissionValue>();
  if (result[0] != node.id || result[1] != time.tick ||
      emission[0] != node.id)
    state.lanes_ok = false;
  state.ids.push_back(node.id);
}

struct PreserveState {
  std::uint64_t expected = 0;
  bool ok = true;
};

void preserve_callback(const EncounterNode& node,
                       const EncounterLogicalTime&,
                       StepContext& context,
                       MutableByteLane,
                       MutableByteLane,
                       void* opaque) {
  PreserveState& state = *static_cast<PreserveState*>(opaque);
  MutableByteLane scratch = context.scratch();
  if (scratch.size() != sizeof(std::uint64_t) ||
      ConstByteLane(scratch).load<std::uint64_t>() != state.expected ||
      context.lease_count() != node.id + 1)
    state.ok = false;
  state.expected = node.id + 100;
  scratch.store(state.expected);
}

struct Snapshot {
  std::vector<std::uint64_t> beam1;
  std::vector<std::uint64_t> beam2;
  std::vector<ResultValue> results;
  std::vector<EmissionValue> emissions;
  std::vector<EncounterId> merge_ids;
  std::vector<std::uint8_t> seen;
  CpuExecutionReport report;
  bool exclusivity_ok = false;
  bool logical_time_ok = false;
  bool scratch_reset_ok = false;
  bool context_alignment_ok = false;
  bool context_exclusivity_ok = false;
  bool merge_lanes_ok = false;
  unsigned max_parallel_level = 0;

  bool physics_equal(const Snapshot& other) const {
    return beam1 == other.beam1 && beam2 == other.beam2 &&
           results == other.results && emissions == other.emissions &&
           merge_ids == other.merge_ids && seen == other.seen;
  }
};

CpuExecutorConfig make_config(const EncounterDAG& dag,
                              CpuExecutionKind execution,
                              CpuScheduleStrategy strategy,
                              std::uint32_t workers) {
  CpuExecutorConfig config;
  config.execution = execution;
  config.strategy = strategy;
  config.context_reset = CpuContextResetPolicy::ZeroBeforeCallback;
  config.worker_count = workers;
  config.context_count = workers;
  config.context_bytes = 64;
  config.result_lane_bytes = sizeof(ResultValue);
  config.emission_lane_bytes = sizeof(EmissionValue);
#ifndef _OPENMP
  if (execution == CpuExecutionKind::OpenMP)
    config.allow_serial_fallback_without_openmp = true;
#endif
  if (strategy == CpuScheduleStrategy::CostAwareDeterministic) {
    config.encounter_costs.resize(static_cast<std::size_t>(dag.encounter_count()));
    for (EncounterId id = 0; id < dag.encounter_count(); ++id)
      config.encounter_costs[static_cast<std::size_t>(id)] =
          1U + ((id * 37U + 11U) % 23U);
  }
  return config;
}

Snapshot run_case(const EncounterDAG& dag,
                  CpuExecutionKind execution,
                  CpuScheduleStrategy strategy,
                  std::uint32_t workers,
                  bool jitter) {
  CpuEncounterExecutor executor(dag,
                                make_config(dag, execution, strategy, workers));
  CallbackState state(dag, workers);
  state.jitter = jitter;
  MergeState merge(static_cast<std::size_t>(dag.encounter_count()));
  Snapshot snapshot;
  snapshot.report = executor.execute(encounter_callback, &state,
                                     merge_callback, &merge);
  snapshot.beam1 = state.beam1;
  snapshot.beam2 = state.beam2;
  snapshot.seen = state.seen;
  snapshot.merge_ids = merge.ids;
  snapshot.exclusivity_ok = state.exclusivity_ok.load();
  snapshot.logical_time_ok = state.logical_time_ok.load();
  snapshot.scratch_reset_ok = state.scratch_reset_ok.load();
  snapshot.context_alignment_ok = state.context_alignment_ok.load();
  snapshot.context_exclusivity_ok = state.context_exclusivity_ok.load();
  snapshot.merge_lanes_ok = merge.lanes_ok;
  snapshot.max_parallel_level = state.max_parallel_level.load();
  snapshot.results.reserve(static_cast<std::size_t>(dag.encounter_count()));
  snapshot.emissions.reserve(static_cast<std::size_t>(dag.encounter_count()));
  for (EncounterId id = 0; id < dag.encounter_count(); ++id) {
    snapshot.results.push_back(executor.result_lane(id).load<ResultValue>());
    snapshot.emissions.push_back(
        executor.emission_lane(id).load<EmissionValue>());
  }
  return snapshot;
}

bool canonical_ids(const std::vector<EncounterId>& ids) {
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != i) return false;
  }
  return true;
}

void test_reference_and_worker_identity() {
  std::printf("\nPersistent CPU encounter waves and deterministic publication\n");
  const EncounterDAG dag = EncounterDAG::build_legacy(6, 7, 3);
  const Snapshot reference =
      run_case(dag, CpuExecutionKind::SerialReference,
               CpuScheduleStrategy::Static, 1, false);
  check(reference.report.success &&
            reference.report.attempted_encounters == dag.encounter_count() &&
            reference.report.completed_encounters == dag.encounter_count() &&
            reference.report.published_encounters == dag.encounter_count() &&
            std::all_of(reference.seen.begin(), reference.seen.end(),
                        [](std::uint8_t count) { return count == 1; }),
        "serial reference executes and publishes every encounter exactly once");
  check(reference.exclusivity_ok && reference.context_exclusivity_ok &&
            reference.logical_time_ok && reference.scratch_reset_ok &&
            reference.context_alignment_ok,
        "slice/context exclusivity, logical time, reset, and 64-byte alignment hold");
  check(reference.merge_lanes_ok && canonical_ids(reference.merge_ids),
        "fixed result/emission lanes merge in increasing node-ID order");

  bool all_workers_identical = true;
  bool all_caps_respected = true;
  bool one_outer_region_only = true;
  const std::array<std::uint32_t, 5> workers = {{1, 2, 4, 8, 10}};
  for (std::uint32_t worker_count : workers) {
    const Snapshot parallel =
        run_case(dag, CpuExecutionKind::OpenMP,
                 CpuScheduleStrategy::Static, worker_count, true);
    all_workers_identical =
        all_workers_identical && reference.physics_equal(parallel);
    all_caps_respected =
        all_caps_respected && parallel.report.context_high_water <= worker_count &&
        parallel.report.context_capacity == worker_count &&
        parallel.context_exclusivity_ok;
#ifdef _OPENMP
    one_outer_region_only =
        one_outer_region_only && parallel.report.parallel_regions == 1 &&
        parallel.max_parallel_level == 1;
#else
    one_outer_region_only =
        one_outer_region_only && parallel.report.parallel_regions == 0 &&
        parallel.report.used_serial_fallback &&
        parallel.max_parallel_level == 0;
#endif
  }
  check(all_workers_identical,
        "serial and 1/2/4/8/10-worker executions are bit-identical under jitter");
  check(all_caps_respected,
        "exclusive reusable StepContext high-water never exceeds its cap");
  check(one_outer_region_only,
        "execution uses one outer parallel region and no nested stage regions");

  const Snapshot cost_aware =
      run_case(dag, CpuExecutionKind::OpenMP,
               CpuScheduleStrategy::CostAwareDeterministic, 10, true);
  check(reference.physics_equal(cost_aware),
        "static and deterministic cost-aware scheduling are bit-identical");
  check(cost_aware.report.context_high_water <= 10 &&
            cost_aware.exclusivity_ok,
        "cost-aware ownership retains bounded contexts and slice exclusivity");
}

void test_shapes_and_limits() {
  std::printf("\nDegenerate shapes and fail-closed resource preflight\n");
  const EncounterDAG unequal = EncounterDAG::build_legacy(2, 5, 4);
  const Snapshot unequal_result =
      run_case(unequal, CpuExecutionKind::OpenMP,
               CpuScheduleStrategy::CostAwareDeterministic, 4, true);
  check(unequal_result.report.completed_encounters == 40 &&
            unequal_result.exclusivity_ok &&
            canonical_ids(unequal_result.merge_ids),
        "unequal slice counts and n_t>1 execute exhaustively and canonically");

  CpuExecutorConfig odd_context = make_config(
      unequal, CpuExecutionKind::OpenMP, CpuScheduleStrategy::Static, 4);
  odd_context.context_bytes = 65;
  CpuEncounterExecutor odd_executor(unequal, odd_context);
  CallbackState odd_state(unequal, 4);
  const CpuExecutionReport odd_report =
      odd_executor.execute(encounter_callback, &odd_state);
  check(odd_report.success && odd_state.context_alignment_ok.load() &&
            odd_executor.preflight_bytes() >= 4 * 128,
        "odd-sized contexts retain 64-byte SIMD/cache-line alignment and accounting");
  check(throws([] { (void)EncounterDAG::build_legacy(1, 1, 0); }),
        "zero n_t is rejected before an executor can be constructed");

  const EncounterDAG zero_left = EncounterDAG::build_legacy(0, 5, 3);
  CpuExecutorConfig zero_config;
  zero_config.context_count = 0;
  zero_config.worker_count = 1;
  CpuEncounterExecutor zero_executor(zero_left, zero_config);
  CallbackState zero_state(zero_left);
  const CpuExecutionReport zero_report =
      zero_executor.execute(encounter_callback, &zero_state);
  const EncounterDAG zero_right = EncounterDAG::build_legacy(5, 0, 7);
  CpuEncounterExecutor zero_right_executor(zero_right, zero_config);
  CallbackState zero_right_state(zero_right);
  const CpuExecutionReport zero_right_report =
      zero_right_executor.execute(encounter_callback, &zero_right_state);
  check(zero_report.success && zero_right_report.success &&
            zero_report.completed_encounters == 0 &&
            zero_right_report.completed_encounters == 0 &&
            zero_state.callback_count.load() == 0 &&
            zero_right_state.callback_count.load() == 0,
        "either zero slice count is a successful zero-work execution");

  const EncounterDAG preserve_dag = EncounterDAG::build_legacy(1, 3, 1);
  CpuExecutorConfig preserve_config;
  preserve_config.context_bytes = sizeof(std::uint64_t);
  preserve_config.context_reset = CpuContextResetPolicy::Preserve;
  CpuEncounterExecutor preserve_executor(preserve_dag, preserve_config);
  PreserveState preserve_state;
  const CpuExecutionReport preserve_report =
      preserve_executor.execute(preserve_callback, &preserve_state);
  check(preserve_report.success && preserve_state.ok &&
            preserve_state.expected == 102,
        "preserve policy reuses context storage without forced clearing");

  CpuExecutorConfig insufficient_contexts = make_config(
      unequal, CpuExecutionKind::OpenMP, CpuScheduleStrategy::Static, 4);
  insufficient_contexts.context_count = 3;
  check(throws([&] {
          CpuEncounterExecutor rejected(unequal, insufficient_contexts);
        }),
        "insufficient context capacity fails before executor construction");

  CpuExecutorConfig context_memory = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 2);
  context_memory.context_bytes = 16;
  context_memory.limits.max_context_bytes = 31;
  check(throws([&] { CpuEncounterExecutor rejected(unequal, context_memory); }),
        "context byte cap is checked before allocation or callbacks");

  CpuExecutorConfig lane_memory = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  lane_memory.limits.max_lane_bytes =
      unequal.encounter_count() *
          (lane_memory.result_lane_bytes + lane_memory.emission_lane_bytes) -
      1;
  check(throws([&] { CpuEncounterExecutor rejected(unequal, lane_memory); }),
        "per-encounter lane byte cap is checked before allocation or callbacks");

  CpuExecutorConfig scheduler_memory = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  scheduler_memory.limits.max_scheduler_bytes = 1;
  check(throws(
            [&] { CpuEncounterExecutor rejected(unequal, scheduler_memory); }),
        "scheduler memory cap fails closed during preflight");

  CpuExecutorConfig encounter_limit = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  encounter_limit.limits.max_encounters = unequal.encounter_count() - 1;
  check(throws([&] { CpuEncounterExecutor rejected(unequal, encounter_limit); }),
        "encounter-count cap fails closed during preflight");

  CpuExecutorConfig graph_memory = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  graph_memory.limits.max_graph_bytes = 1;
  check(throws([&] { CpuEncounterExecutor rejected(unequal, graph_memory); }),
        "owned encounter-graph byte cap is checked during preflight");

  CpuExecutorConfig total_memory = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  total_memory.limits.max_total_bytes = 1;
  check(throws([&] { CpuEncounterExecutor rejected(unequal, total_memory); }),
        "aggregate peak-memory cap fails closed during preflight");

  CpuExecutorConfig bad_costs = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::CostAwareDeterministic, 1);
  bad_costs.encounter_costs.pop_back();
  check(throws([&] { CpuEncounterExecutor rejected(unequal, bad_costs); }),
        "non-exhaustive cost model is rejected before scheduling");

  CpuExecutorConfig bad_strategy = make_config(
      unequal, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  bad_strategy.strategy = static_cast<CpuScheduleStrategy>(255);
  check(throws([&] { CpuEncounterExecutor rejected(unequal, bad_strategy); }),
        "unknown scheduling strategy fails closed during preflight");

  bool missing_openmp_fails_closed = true;
#ifndef _OPENMP
  CpuExecutorConfig no_fallback = make_config(
      unequal, CpuExecutionKind::OpenMP, CpuScheduleStrategy::Static, 2);
  no_fallback.allow_serial_fallback_without_openmp = false;
  missing_openmp_fails_closed = throws(
      [&] { CpuEncounterExecutor rejected(unequal, no_fallback); });
#endif
  check(missing_openmp_fails_closed,
        "OpenMP absence requires explicit serial-fallback authorization");

  EncounterDAG replaceable = EncounterDAG::build_legacy(2, 3, 1);
  CpuExecutorConfig owned_config = make_config(
      replaceable, CpuExecutionKind::SerialReference,
      CpuScheduleStrategy::Static, 1);
  CpuEncounterExecutor owned_executor(replaceable, owned_config);
  replaceable = EncounterDAG::build_legacy(1, 1, 1);
  CallbackState owned_state(owned_executor.dag());
  const CpuExecutionReport owned_report =
      owned_executor.execute(encounter_callback, &owned_state);
  check(owned_report.completed_encounters == 6 &&
            owned_executor.dag().encounter_count() == 6,
        "executor-owned immutable DAG survives caller replacement");
}

void test_failure_transaction_and_reuse() {
  std::printf("\nCallback failure boundary and persistent-executor reuse\n");
  const EncounterDAG dag = EncounterDAG::build_legacy(4, 4, 2);
  CpuEncounterExecutor executor(
      dag, make_config(dag, CpuExecutionKind::OpenMP,
                       CpuScheduleStrategy::CostAwareDeterministic, 4));
  CallbackState failing_state(dag, 4);
  failing_state.throw_at = dag.node_id(1, 1, 1);
  MergeState failing_merge(static_cast<std::size_t>(dag.encounter_count()));
  bool correct_failure = false;
  try {
    (void)executor.execute(encounter_callback, &failing_state,
                           merge_callback, &failing_merge);
  } catch (const CpuExecutorError& error) {
    correct_failure = error.encounter() == failing_state.throw_at;
  }
  check(correct_failure && failing_merge.ids.empty() &&
            !executor.has_successful_output() &&
            throws([&] { (void)executor.result_lane(0); }),
        "callback exception identifies its node and publishes no partial lanes");

  CallbackState multi_failure_state(dag, 4);
  multi_failure_state.throw_at = dag.node_id(0, 2, 0);
  multi_failure_state.throw_at_secondary = dag.node_id(2, 0, 0);
  const EncounterId expected_lowest =
      std::min(multi_failure_state.throw_at,
               multi_failure_state.throw_at_secondary);
  bool lowest_failure_selected = false;
  try {
    (void)executor.execute(encounter_callback, &multi_failure_state);
  } catch (const CpuExecutorError& error) {
    lowest_failure_selected = error.encounter() == expected_lowest;
  }
  check(lowest_failure_selected &&
            multi_failure_state.seen[static_cast<std::size_t>(
                multi_failure_state.throw_at)] == 1 &&
            multi_failure_state.seen[static_cast<std::size_t>(
                multi_failure_state.throw_at_secondary)] == 1,
        "simultaneous worker failures report the deterministic lowest encounter ID");

  CallbackState recovered_state(dag, 4);
  MergeState recovered_merge(static_cast<std::size_t>(dag.encounter_count()));
  const CpuExecutionReport recovered =
      executor.execute(encounter_callback, &recovered_state,
                       merge_callback, &recovered_merge);
  check(recovered.success &&
            recovered.completed_encounters == dag.encounter_count() &&
            canonical_ids(recovered_merge.ids) &&
            std::all_of(recovered_state.seen.begin(), recovered_state.seen.end(),
                        [](std::uint8_t count) { return count == 1; }),
        "executor clears failed lanes/status and is reusable after state reload");

  check(throws([&] { (void)executor.execute(nullptr, nullptr); }),
        "null callback is rejected before invalidating published output");

  bool nested_rejected = true;
#ifdef _OPENMP
  CallbackState nested_state(dag, 4);
  nested_rejected = false;
#pragma omp parallel num_threads(2) shared(nested_rejected, nested_state)
  {
#pragma omp single
    {
      try {
        (void)executor.execute(encounter_callback, &nested_state);
      } catch (const CpuExecutorError&) {
        nested_rejected = true;
      }
    }
  }
  nested_rejected = nested_rejected && nested_state.callback_count.load() == 0;
#endif
  check(nested_rejected,
        "entry from an existing OpenMP region is rejected before callbacks");

  bool nested_serial_completed = true;
#ifdef _OPENMP
  CpuExecutorConfig serial_config = make_config(
      dag, CpuExecutionKind::SerialReference, CpuScheduleStrategy::Static, 1);
  CpuEncounterExecutor serial_executor(dag, serial_config);
  CallbackState serial_nested_state(dag, 1);
  nested_serial_completed = false;
#pragma omp parallel num_threads(2) shared(nested_serial_completed, serial_nested_state)
  {
#pragma omp single
    {
      try {
        const CpuExecutionReport nested_report =
            serial_executor.execute(encounter_callback, &serial_nested_state);
        nested_serial_completed =
            nested_report.completed_encounters == dag.encounter_count();
      } catch (...) {
        nested_serial_completed = false;
      }
    }
  }
#endif
  check(nested_serial_completed,
        "serial-reference execution is valid inside an outer OpenMP region");
}

}  // namespace

int main() {
  std::printf("FeeBL modern CPU encounter executor tests\n");
  test_reference_and_worker_identity();
  test_shapes_and_limits();
  test_failure_transaction_and_reuse();
  std::printf("\nmodern CPU executor failures: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
