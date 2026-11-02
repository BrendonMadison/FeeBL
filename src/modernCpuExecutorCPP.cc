#include "modernCpuExecutorCPP.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace feebl {
namespace modern {
namespace {

std::uint64_t add_or_throw(std::uint64_t a, std::uint64_t b,
                           const char* description) {
  std::uint64_t result = 0;
  if (!checked_add(a, b, result))
    throw CpuExecutorError(std::string(description) + " addition overflow");
  return result;
}

std::uint64_t multiply_or_throw(std::uint64_t a, std::uint64_t b,
                                const char* description) {
  std::uint64_t result = 0;
  if (!checked_multiply(a, b, result))
    throw CpuExecutorError(std::string(description) +
                           " multiplication overflow");
  return result;
}

std::uint64_t align_up_or_throw(std::uint64_t value,
                                std::uint64_t alignment,
                                const char* description) {
  if (value == 0) return 0;
  const std::uint64_t padded =
      add_or_throw(value, alignment - 1, description);
  return (padded / alignment) * alignment;
}

void require_representable(std::uint64_t value, const char* description) {
  if (!fits_size_t(value))
    throw CpuExecutorError(std::string(description) +
                           " is not representable by size_t");
}

#ifdef _OPENMP
void zero_worker_partition(std::vector<std::byte>& storage,
                           std::uint32_t worker,
                           std::uint32_t workers) noexcept {
  const std::size_t quotient = storage.size() / workers;
  const std::size_t remainder = storage.size() % workers;
  const std::size_t begin = quotient * worker +
                            std::min<std::size_t>(worker, remainder);
  const std::size_t size = quotient + (worker < remainder ? 1u : 0u);
  std::fill(storage.begin() + static_cast<std::ptrdiff_t>(begin),
            storage.begin() + static_cast<std::ptrdiff_t>(begin + size),
            std::byte{0});
}
#endif

}  // namespace

void StepContext::begin(EncounterId encounter, bool zero_scratch) noexcept {
  current_encounter_ = encounter;
  if (lease_count_ != std::numeric_limits<std::uint64_t>::max()) ++lease_count_;
  if (zero_scratch && size_ != 0)
    std::fill(data_, data_ + size_, std::byte{0});
}

CpuEncounterExecutor::OwnedInputs CpuEncounterExecutor::checked_copy(
    const EncounterDAG& dag, const CpuExecutorConfig& config) {
  const std::uint64_t bytes = validate_and_preflight(dag, config);
  return OwnedInputs{dag, config, bytes};
}

CpuEncounterExecutor::OwnedInputs CpuEncounterExecutor::checked_move(
    EncounterDAG&& dag, CpuExecutorConfig&& config) {
  const std::uint64_t bytes = validate_and_preflight(dag, config);
  return OwnedInputs{std::move(dag), std::move(config), bytes};
}

CpuEncounterExecutor::CpuEncounterExecutor(
    const EncounterDAG& dag, const CpuExecutorConfig& config)
    : CpuEncounterExecutor(checked_copy(dag, config)) {}

CpuEncounterExecutor::CpuEncounterExecutor(EncounterDAG&& dag,
                                           CpuExecutorConfig&& config)
    : CpuEncounterExecutor(checked_move(std::move(dag), std::move(config))) {}

CpuEncounterExecutor::CpuEncounterExecutor(OwnedInputs inputs)
    : dag_(std::move(inputs.dag)),
      config_(std::move(inputs.config)),
      preflight_bytes_(inputs.preflight_bytes) {

  const std::size_t context_bytes =
      static_cast<std::size_t>(config_.context_bytes);
  const std::size_t context_stride = static_cast<std::size_t>(
      align_up_or_throw(config_.context_bytes, kCpuContextAlignment,
                        "aligned context stride"));
  const std::size_t context_blocks = context_stride == 0
      ? 0
      : static_cast<std::size_t>(config_.context_count) *
            context_stride / kCpuContextAlignment;
  context_storage_.resize(context_blocks);
  contexts_.reserve(config_.context_count);
  std::byte* context_base = context_storage_.empty()
      ? nullptr
      : reinterpret_cast<std::byte*>(context_storage_.data());
  for (std::uint32_t context = 0; context < config_.context_count; ++context) {
    std::byte* data = context_bytes == 0
                          ? nullptr
                          : context_base +
                                static_cast<std::size_t>(context) * context_stride;
    contexts_.push_back(StepContext(context, data, context_bytes));
  }

  const std::size_t encounter_count =
      static_cast<std::size_t>(dag_.encounter_count());
  result_storage_.resize(encounter_count *
                         static_cast<std::size_t>(config_.result_lane_bytes));
  emission_storage_.resize(
      encounter_count *
      static_cast<std::size_t>(config_.emission_lane_bytes));
  worker_states_.resize(config_.worker_count);
  build_schedule();
}

std::uint64_t CpuEncounterExecutor::validate_and_preflight(
    const EncounterDAG& dag, const CpuExecutorConfig& config) {
  if (config.execution != CpuExecutionKind::SerialReference &&
      config.execution != CpuExecutionKind::OpenMP)
    throw CpuExecutorError("CPU executor execution kind is invalid");
  if (config.strategy != CpuScheduleStrategy::Static &&
      config.strategy != CpuScheduleStrategy::CostAwareDeterministic)
    throw CpuExecutorError("CPU executor schedule strategy is invalid");
  if (config.context_reset != CpuContextResetPolicy::Preserve &&
      config.context_reset != CpuContextResetPolicy::ZeroBeforeCallback)
    throw CpuExecutorError("CPU executor context reset policy is invalid");
  if (config.worker_count == 0)
    throw CpuExecutorError("CPU executor worker_count must be positive");
  if (config.worker_count > config.limits.max_workers)
    throw CpuExecutorError("CPU executor worker_count exceeds its limit");
  if (config.context_count > config.limits.max_contexts)
    throw CpuExecutorError("CPU executor context_count exceeds its limit");
  if (dag.encounter_count() > config.limits.max_encounters)
    throw CpuExecutorError("CPU executor encounter count exceeds its limit");
#ifndef _OPENMP
  if (config.execution == CpuExecutionKind::OpenMP &&
      !config.allow_serial_fallback_without_openmp)
    throw CpuExecutorError(
        "OpenMP execution was requested from a build without OpenMP; "
        "serial fallback was not authorized");
#endif
  if (config.execution == CpuExecutionKind::OpenMP &&
      config.context_count < config.worker_count)
    throw CpuExecutorError(
        "OpenMP execution requires at least one context per requested worker");
  if (dag.encounter_count() != 0 && config.context_count == 0)
    throw CpuExecutorError("non-empty execution requires at least one context");
  if (!config.encounter_costs.empty() &&
      config.encounter_costs.size() != dag.encounter_count())
    throw CpuExecutorError(
        "encounter_costs must be empty or have one entry per encounter");

  std::uint64_t graph_total = multiply_or_throw(
      dag.encounter_count(), sizeof(EncounterNode), "owned graph nodes");
  graph_total = add_or_throw(
      graph_total,
      multiply_or_throw(dag.encounter_count(), 2 * sizeof(EncounterId),
                        "owned graph encounter indexes"),
      "owned graph storage");
  graph_total = add_or_throw(
      graph_total,
      multiply_or_throw(dag.wave_count(), sizeof(EncounterWave),
                        "owned graph waves"),
      "owned graph storage");
  if (graph_total > config.limits.max_graph_bytes)
    throw CpuExecutorError("owned encounter graph exceeds its memory limit");

  std::uint64_t validation_scratch = dag.encounter_count();
  validation_scratch = add_or_throw(
      validation_scratch,
      multiply_or_throw(dag.beam1_slice_count(), sizeof(WaveIndex),
                        "beam-1 graph validation scratch"),
      "graph validation scratch");
  validation_scratch = add_or_throw(
      validation_scratch,
      multiply_or_throw(dag.beam2_slice_count(), sizeof(WaveIndex),
                        "beam-2 graph validation scratch"),
      "graph validation scratch");
  const std::uint64_t graph_peak = add_or_throw(
      graph_total, validation_scratch, "graph validation peak");
  if (graph_peak > config.limits.max_graph_bytes)
    throw CpuExecutorError(
        "owned encounter graph plus validation scratch exceeds its memory limit");

  // DAG validation allocates scratch proportional to the graph.  The
  // allocation-free shape/byte gates above therefore run first and include
  // that scratch in the configured peak-memory contract.
  const ValidationResult graph = dag.validate();
  if (!graph) {
    std::ostringstream message;
    message << "invalid EncounterDAG at index " << graph.index << ": "
            << graph.message;
    throw CpuExecutorError(message.str());
  }

  const std::uint64_t context_stride = align_up_or_throw(
      config.context_bytes, kCpuContextAlignment, "aligned context stride");
  const std::uint64_t context_scratch_total = multiply_or_throw(
      config.context_count, context_stride, "aligned context storage");
  const std::uint64_t context_object_total = multiply_or_throw(
      config.context_count, sizeof(StepContext), "context objects");
  const std::uint64_t context_total = add_or_throw(
      context_scratch_total, context_object_total, "context storage");
  if (context_total > config.limits.max_context_bytes)
    throw CpuExecutorError("context storage exceeds its memory limit");

  const std::uint64_t lane_width = add_or_throw(
      config.result_lane_bytes, config.emission_lane_bytes, "lane width");
  const std::uint64_t lane_total = multiply_or_throw(
      dag.encounter_count(), lane_width, "encounter lane storage");
  if (lane_total > config.limits.max_lane_bytes)
    throw CpuExecutorError("encounter lane storage exceeds its memory limit");

  const std::uint64_t worker_slots =
      add_or_throw(config.worker_count, 1, "wave worker slots");
  const std::uint64_t offset_count = multiply_or_throw(
      dag.wave_count(), worker_slots, "schedule offset count");
  std::uint64_t scheduler_total = multiply_or_throw(
      offset_count, sizeof(std::uint64_t), "schedule offsets");
  scheduler_total = add_or_throw(
      scheduler_total,
      multiply_or_throw(dag.encounter_count(), sizeof(EncounterId),
                        "schedule encounter IDs"),
      "scheduler storage");
  scheduler_total = add_or_throw(
      scheduler_total,
      multiply_or_throw(config.worker_count, sizeof(WorkerExecutionState),
                        "worker execution status"),
      "scheduler storage");
  if (!config.encounter_costs.empty()) {
    scheduler_total = add_or_throw(
        scheduler_total,
        multiply_or_throw(dag.encounter_count(), sizeof(std::uint64_t),
                          "cost estimates"),
        "scheduler storage");
  }

  std::uint64_t max_wave_width = 0;
  for (const EncounterWave& wave : dag.waves())
    max_wave_width = std::max<std::uint64_t>(max_wave_width,
                                             wave.encounters.size());
  std::uint64_t schedule_temporary = multiply_or_throw(
      max_wave_width, sizeof(std::uint32_t) + sizeof(EncounterId),
      "schedule construction workspace");
  schedule_temporary = add_or_throw(
      schedule_temporary,
      multiply_or_throw(config.worker_count, sizeof(std::uint64_t),
                        "cost-balancing workspace"),
      "schedule construction workspace");
  const std::uint64_t scheduler_peak = add_or_throw(
      scheduler_total, schedule_temporary, "peak scheduler storage");
  if (scheduler_peak > config.limits.max_scheduler_bytes)
    throw CpuExecutorError("scheduler storage exceeds its memory limit");

  std::uint64_t preflight_bytes = add_or_throw(
      graph_peak, context_total, "total executor storage");
  preflight_bytes = add_or_throw(preflight_bytes, lane_total,
                                 "total executor storage");
  preflight_bytes = add_or_throw(preflight_bytes, scheduler_peak,
                                 "total executor storage");
  if (preflight_bytes > config.limits.max_total_bytes)
    throw CpuExecutorError("executor storage exceeds its total memory limit");

  require_representable(context_scratch_total, "context scratch storage");
  require_representable(lane_total, "lane storage");
  require_representable(offset_count, "schedule offset count");
  require_representable(dag.encounter_count(), "encounter count");

  const std::uint64_t result_total = multiply_or_throw(
      dag.encounter_count(), config.result_lane_bytes,
      "result lane storage");
  const std::uint64_t emission_total = multiply_or_throw(
      dag.encounter_count(), config.emission_lane_bytes,
      "emission lane storage");
  if (context_scratch_total / kCpuContextAlignment >
          static_cast<std::uint64_t>(
              std::vector<ContextStorageBlock>().max_size()) ||
      result_total >
          static_cast<std::uint64_t>(std::vector<std::byte>().max_size()) ||
      emission_total >
          static_cast<std::uint64_t>(std::vector<std::byte>().max_size()))
    throw CpuExecutorError("requested executor byte storage exceeds vector max_size");
  if (config.context_count >
          static_cast<std::uint64_t>(std::vector<StepContext>().max_size()) ||
      offset_count > static_cast<std::uint64_t>(
                         std::vector<std::uint64_t>().max_size()) ||
      dag.encounter_count() > static_cast<std::uint64_t>(
                                  std::vector<EncounterId>().max_size()) ||
      config.worker_count > static_cast<std::uint64_t>(
                                std::vector<WorkerExecutionState>().max_size()))
    throw CpuExecutorError("requested executor element count exceeds vector max_size");
  return preflight_bytes;
}

void CpuEncounterExecutor::build_schedule() {
  const std::size_t workers = config_.worker_count;
  const std::size_t stride = workers + 1;
  assignment_offsets_.resize(static_cast<std::size_t>(dag_.wave_count()) *
                             stride);
  assigned_encounters_.reserve(static_cast<std::size_t>(dag_.encounter_count()));

  std::vector<std::uint32_t> owners;
  std::vector<std::uint64_t> loads;
  std::vector<EncounterId> ranked;
  for (const EncounterWave& wave : dag_.waves()) {
    owners.assign(wave.encounters.size(), 0);
    if (config_.strategy == CpuScheduleStrategy::Static) {
      for (std::size_t position = 0; position < wave.encounters.size();
           ++position)
        owners[position] = static_cast<std::uint32_t>(position % workers);
    } else {
      loads.assign(workers, 0);
      ranked = wave.encounters;
      std::sort(
          ranked.begin(), ranked.end(),
          [this](EncounterId left, EncounterId right) {
            const std::uint64_t left_cost = config_.encounter_costs.empty()
                                                ? 1
                                                : config_.encounter_costs[
                                                      static_cast<std::size_t>(left)];
            const std::uint64_t right_cost = config_.encounter_costs.empty()
                                                 ? 1
                                                 : config_.encounter_costs[
                                                       static_cast<std::size_t>(right)];
            if (left_cost != right_cost) return left_cost > right_cost;
            return left < right;
          });
      for (EncounterId id : ranked) {
        const std::size_t worker = static_cast<std::size_t>(
            std::min_element(loads.begin(), loads.end()) - loads.begin());
        const std::uint64_t cost = config_.encounter_costs.empty()
                                       ? 1
                                       : config_.encounter_costs[
                                             static_cast<std::size_t>(id)];
        loads[worker] =
            loads[worker] > std::numeric_limits<std::uint64_t>::max() - cost
                ? std::numeric_limits<std::uint64_t>::max()
                : loads[worker] + cost;
        const auto position = std::lower_bound(wave.encounters.begin(),
                                               wave.encounters.end(), id);
        owners[static_cast<std::size_t>(position - wave.encounters.begin())] =
            static_cast<std::uint32_t>(worker);
      }
    }

    const std::size_t base = static_cast<std::size_t>(wave.index) * stride;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      assignment_offsets_[base + worker] = assigned_encounters_.size();
      for (std::size_t position = 0; position < wave.encounters.size();
           ++position) {
        if (owners[position] == worker)
          assigned_encounters_.push_back(wave.encounters[position]);
      }
    }
    assignment_offsets_[base + workers] = assigned_encounters_.size();
    std::uint32_t active_workers = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      if (assignment_offsets_[base + worker] !=
          assignment_offsets_[base + worker + 1])
        ++active_workers;
    }
    context_high_water_bound_ =
        std::max(context_high_water_bound_, active_workers);
  }
  if (assigned_encounters_.size() != dag_.encounter_count())
    throw CpuExecutorError("internal schedule is not exhaustive");
}

EncounterLogicalTime CpuEncounterExecutor::logical_time(
    const EncounterNode& node) const noexcept {
  EncounterLogicalTime time;
  time.tick = node.wave;
  time.wave = node.wave;
  time.diagonal = node.diagonal;
  time.substep = node.substep;
  time.encounter = node.id;
  return time;
}

MutableByteLane CpuEncounterExecutor::mutable_result_lane(
    EncounterId encounter) noexcept {
  const std::size_t width =
      static_cast<std::size_t>(config_.result_lane_bytes);
  return MutableByteLane(width == 0
                             ? nullptr
                             : result_storage_.data() +
                                   static_cast<std::size_t>(encounter) * width,
                         width);
}

MutableByteLane CpuEncounterExecutor::mutable_emission_lane(
    EncounterId encounter) noexcept {
  const std::size_t width =
      static_cast<std::size_t>(config_.emission_lane_bytes);
  return MutableByteLane(width == 0
                             ? nullptr
                             : emission_storage_.data() +
                                   static_cast<std::size_t>(encounter) * width,
                         width);
}

ConstByteLane CpuEncounterExecutor::result_lane(EncounterId encounter) const {
  if (!output_valid_)
    throw CpuExecutorError("no successful executor output is available");
  if (encounter >= dag_.encounter_count())
    throw std::out_of_range("result lane encounter ID is out of range");
  const std::size_t width =
      static_cast<std::size_t>(config_.result_lane_bytes);
  return ConstByteLane(width == 0
                           ? nullptr
                           : result_storage_.data() +
                                 static_cast<std::size_t>(encounter) * width,
                       width);
}

ConstByteLane CpuEncounterExecutor::emission_lane(EncounterId encounter) const {
  if (!output_valid_)
    throw CpuExecutorError("no successful executor output is available");
  if (encounter >= dag_.encounter_count())
    throw std::out_of_range("emission lane encounter ID is out of range");
  const std::size_t width =
      static_cast<std::size_t>(config_.emission_lane_bytes);
  return ConstByteLane(width == 0
                           ? nullptr
                           : emission_storage_.data() +
                                 static_cast<std::size_t>(encounter) * width,
                       width);
}

CpuExecutionReport CpuEncounterExecutor::execute(
    EncounterCallback callback, void* callback_data,
    EncounterMergeCallback merge, void* merge_data) {
  if (callback == nullptr)
    throw CpuExecutorError("encounter callback must not be null");
#ifdef _OPENMP
  if (config_.execution == CpuExecutionKind::OpenMP && omp_in_parallel())
    throw CpuExecutorError(
        "CpuEncounterExecutor refuses nested OpenMP execution");
#endif

  output_valid_ = false;
  for (WorkerExecutionState& state : worker_states_)
    state = WorkerExecutionState{};

  CpuExecutionReport report;
  report.requested_workers = config_.worker_count;
  report.context_capacity = config_.context_count;
  if (dag_.encounter_count() == 0) {
    report.actual_workers = 0;
    report.success = true;
    output_valid_ = true;
    return report;
  }

  std::atomic<bool> callback_failed{false};
  bool team_mismatch = false;
  std::uint32_t actual_workers = 1;

  const auto run_one = [&](EncounterId id, std::uint32_t context_id) noexcept {
    const EncounterNode& node = dag_.node(id);
    StepContext& context = contexts_[context_id];
    WorkerExecutionState& state = worker_states_[context_id];
    ++state.attempted;
    context.begin(id, config_.context_reset ==
                          CpuContextResetPolicy::ZeroBeforeCallback);
    try {
      callback(node, logical_time(node), context, mutable_result_lane(id),
               mutable_emission_lane(id), callback_data);
      ++state.completed;
    } catch (...) {
      if (id < state.lowest_failure) {
        state.lowest_failure = id;
        state.failure = std::current_exception();
      }
      callback_failed.store(true, std::memory_order_release);
    }
    context.end();
  };

  bool use_serial = config_.execution == CpuExecutionKind::SerialReference;
#ifndef _OPENMP
  if (config_.execution == CpuExecutionKind::OpenMP) {
    use_serial = true;
    report.used_serial_fallback = true;
  }
#endif

  if (use_serial) {
    std::fill(result_storage_.begin(), result_storage_.end(), std::byte{0});
    std::fill(emission_storage_.begin(), emission_storage_.end(), std::byte{0});
    for (const EncounterWave& wave : dag_.waves()) {
      for (EncounterId id : wave.encounters) run_one(id, 0);
      if (callback_failed.load(std::memory_order_acquire)) break;
    }
    actual_workers = 1;
  } else {
#ifdef _OPENMP
    report.parallel_regions = 1;
    const std::uint32_t requested_workers = config_.worker_count;
    const std::size_t stride = static_cast<std::size_t>(requested_workers) + 1;
#pragma omp parallel num_threads(requested_workers) shared(team_mismatch, actual_workers)
    {
      const std::uint32_t worker =
          static_cast<std::uint32_t>(omp_get_thread_num());
#pragma omp single
      {
        actual_workers = static_cast<std::uint32_t>(omp_get_num_threads());
        team_mismatch = actual_workers != requested_workers;
      }
      if (!team_mismatch) {
        zero_worker_partition(result_storage_, worker, requested_workers);
        zero_worker_partition(emission_storage_, worker, requested_workers);
#pragma omp barrier
        for (const EncounterWave& wave : dag_.waves()) {
          const std::size_t base = static_cast<std::size_t>(wave.index) * stride;
          const std::uint64_t begin = assignment_offsets_[base + worker];
          const std::uint64_t end = assignment_offsets_[base + worker + 1];
          for (std::uint64_t position = begin; position < end; ++position)
            run_one(assigned_encounters_[static_cast<std::size_t>(position)],
                    worker);
#pragma omp barrier
          if (callback_failed.load(std::memory_order_acquire)) break;
        }
      }
    }
#endif
  }

  report.actual_workers = actual_workers;
  report.context_high_water = use_serial ? 1 : context_high_water_bound_;
  for (const WorkerExecutionState& state : worker_states_) {
    report.attempted_encounters += state.attempted;
    report.completed_encounters += state.completed;
  }

  if (team_mismatch)
    throw CpuExecutorError(
        "OpenMP runtime did not create the requested fixed worker team");
  const WorkerExecutionState* first_failure = nullptr;
  for (const WorkerExecutionState& state : worker_states_) {
    if (state.lowest_failure == kInvalidEncounter) continue;
    if (first_failure == nullptr ||
        state.lowest_failure < first_failure->lowest_failure)
      first_failure = &state;
  }
  if (first_failure != nullptr) {
    try {
      std::rethrow_exception(first_failure->failure);
    } catch (const std::exception& error) {
      throw CpuExecutorError(std::string("encounter callback failed: ") +
                                 error.what(),
                             first_failure->lowest_failure);
    } catch (...) {
      throw CpuExecutorError("encounter callback failed with a non-standard exception",
                             first_failure->lowest_failure);
    }
  }
  if (report.completed_encounters != dag_.encounter_count())
    throw CpuExecutorError("executor completed a non-exhaustive encounter set");

  // This is the only publication point.  Node storage order is validated to be
  // identical to increasing EncounterId order by EncounterDAG::validate().
  if (merge != nullptr) {
    for (const EncounterNode& node : dag_.nodes()) {
      merge(node, logical_time(node),
            ConstByteLane(mutable_result_lane(node.id)),
            ConstByteLane(mutable_emission_lane(node.id)), merge_data);
      ++report.published_encounters;
    }
  }
  report.success = true;
  output_valid_ = true;
  return report;
}

}  // namespace modern
}  // namespace feebl
