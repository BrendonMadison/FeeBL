#ifndef FEEBL_MODERN_CPU_EXECUTOR_CPP_H
#define FEEBL_MODERN_CPU_EXECUTOR_CPP_H

#include "encounterDAGCPP.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace feebl {
namespace modern {

constexpr std::size_t kCpuContextAlignment = 64;

// The reference path is deliberately available in an OpenMP build.  It is the
// executable oracle used to distinguish scheduling defects from kernel defects.
enum class CpuExecutionKind : std::uint8_t {
  SerialReference = 0,
  OpenMP = 1
};

enum class CpuScheduleStrategy : std::uint8_t {
  Static = 0,
  CostAwareDeterministic = 1
};

enum class CpuContextResetPolicy : std::uint8_t {
  // Preserved bytes are worker-local cache state only.  They must never affect
  // physics, reductions, emissions, or counter-RNG identity because encounter-
  // to-worker ownership may change with worker count or schedule strategy.
  Preserve = 0,
  ZeroBeforeCallback = 1
};

// Logical time is a property of the encounter graph, never of wall-clock or
// worker progress.  The integer tick is the canonical wave index.  A future
// geometry builder may attach a separately validated physical-time map without
// changing this scheduling contract.
struct EncounterLogicalTime {
  WaveIndex tick = 0;
  WaveIndex wave = 0;
  WaveIndex diagonal = 0;
  SubstepIndex substep = 0;
  EncounterId encounter = kInvalidEncounter;
};

class MutableByteLane {
 public:
  MutableByteLane() noexcept = default;
  MutableByteLane(std::byte* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  std::byte* data() noexcept { return data_; }
  const std::byte* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  template <class T>
  void store(const T& value, std::size_t offset = 0) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "lane values must be trivially copyable");
    if (offset > size_ || sizeof(T) > size_ - offset)
      throw std::out_of_range("fixed lane store exceeds its capacity");
    std::memcpy(data_ + offset, &value, sizeof(T));
  }

 private:
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

class ConstByteLane {
 public:
  ConstByteLane() noexcept = default;
  ConstByteLane(const std::byte* data, std::size_t size) noexcept
      : data_(data), size_(size) {}
  ConstByteLane(MutableByteLane lane) noexcept
      : data_(lane.data()), size_(lane.size()) {}

  const std::byte* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  template <class T>
  T load(std::size_t offset = 0) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "lane values must be trivially copyable");
    if (offset > size_ || sizeof(T) > size_ - offset)
      throw std::out_of_range("fixed lane load exceeds its capacity");
    T value{};
    std::memcpy(&value, data_ + offset, sizeof(T));
    return value;
  }

 private:
  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

// One StepContext is exclusively leased to one logical worker for a callback.
// Its storage is allocated once by CpuEncounterExecutor.  Kernels may use it for
// field tiles, secondary staging, or other bounded transient scratch, but must
// not retain the view after the callback.  Zeroing is an explicit policy because
// blindly clearing a large grid-sized context at every encounter is expensive.
class StepContext {
 public:
  std::uint32_t id() const noexcept { return id_; }
  std::uint64_t lease_count() const noexcept { return lease_count_; }
  EncounterId current_encounter() const noexcept { return current_encounter_; }
  MutableByteLane scratch() noexcept { return MutableByteLane(data_, size_); }
  ConstByteLane scratch() const noexcept { return ConstByteLane(data_, size_); }

 private:
  friend class CpuEncounterExecutor;
  StepContext(std::uint32_t id, std::byte* data, std::size_t size) noexcept
      : id_(id), data_(data), size_(size) {}
  void begin(EncounterId encounter, bool zero_scratch) noexcept;
  void end() noexcept { current_encounter_ = kInvalidEncounter; }

  std::uint32_t id_ = 0;
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
  std::uint64_t lease_count_ = 0;
  EncounterId current_encounter_ = kInvalidEncounter;
};

struct CpuExecutorLimits {
  std::uint64_t max_encounters = 100000000ULL;
  std::uint64_t max_workers = 1024ULL;
  std::uint64_t max_contexts = 1024ULL;
  std::uint64_t max_graph_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t max_context_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t max_lane_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t max_scheduler_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t max_total_bytes = 12ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct CpuExecutorConfig {
  CpuExecutionKind execution = CpuExecutionKind::SerialReference;
  CpuScheduleStrategy strategy = CpuScheduleStrategy::Static;
  CpuContextResetPolicy context_reset = CpuContextResetPolicy::Preserve;
  std::uint32_t worker_count = 1;
  std::uint32_t context_count = 1;
  std::uint64_t context_bytes = 0;
  std::uint64_t result_lane_bytes = 0;
  std::uint64_t emission_lane_bytes = 0;

  // Requested OpenMP execution fails closed when this translation unit was
  // built without OpenMP.  Diagnostic/portable unit builds may authorize and
  // must report an explicit serial fallback; qualified production plans do not.
  bool allow_serial_fallback_without_openmp = false;

  // Empty means unit cost.  Otherwise there must be exactly one non-negative
  // integral estimate per encounter.  Cost affects ownership only, never merge
  // order, logical time, RNG identity, or numerical reduction order.
  std::vector<std::uint64_t> encounter_costs;
  CpuExecutorLimits limits;
};

struct CpuExecutionReport {
  std::uint64_t attempted_encounters = 0;
  std::uint64_t completed_encounters = 0;
  std::uint64_t published_encounters = 0;
  std::uint32_t requested_workers = 0;
  std::uint32_t actual_workers = 0;
  std::uint32_t context_capacity = 0;
  std::uint32_t context_high_water = 0;
  std::uint32_t parallel_regions = 0;
  bool used_serial_fallback = false;
  bool success = false;
};

class CpuExecutorError : public std::runtime_error {
 public:
  CpuExecutorError(const std::string& message,
                   EncounterId encounter = kInvalidEncounter)
      : std::runtime_error(message), encounter_(encounter) {}
  EncounterId encounter() const noexcept { return encounter_; }

 private:
  EncounterId encounter_;
};

// The encounter callback may mutate beam slices named by node because the DAG
// proves that those slices are disjoint within a wave.  It must use stable IDs
// and counter-based RNG keys supplied by the future physics adapter; neither a
// global RNG nor a global RESULTS object is part of this API.
using EncounterCallback = void (*)(const EncounterNode& node,
                                   const EncounterLogicalTime& logical_time,
                                   StepContext& context,
                                   MutableByteLane result_lane,
                                   MutableByteLane emission_lane,
                                   void* user_data);

// Publication is invoked only after every encounter callback has succeeded and
// is always in increasing EncounterId order.  noexcept is intentional: it makes
// the lane-to-sink phase incapable of reporting a recoverable half-publication.
// A sink that performs allocation must pre-reserve it and enforce its own
// no-throw contract.
using EncounterMergeCallback = void (*)(
    const EncounterNode& node,
    const EncounterLogicalTime& logical_time,
    ConstByteLane result_lane,
    ConstByteLane emission_lane,
    void* user_data) noexcept;

// Persistent, bounded executor for one immutable encounter graph. This
// intentionally provides no FeeBL production dispatch: integration happens only
// after this scheduling primitive and its eventual physics adapter pass gates.
//
// Transaction boundary:
//   * configuration, DAG validation, and all executor-owned memory limits and
//     allocations are resolved by the constructor before a callback can run
//     (the OpenMP runtime may initialize its own worker pool lazily);
//   * execute() publishes no lanes if any encounter callback throws;
//   * result/emission lanes and contexts are reusable without allocation;
//   * external beam-slice mutations made by a throwing callback cannot be
//     rolled back.  The caller must discard/reload that step state before reuse.
//
// The object is intentionally non-copyable and not safe for concurrent calls to
// execute().  A successful execution leaves immutable node-indexed lanes
// readable until the next execute() call.
class CpuEncounterExecutor {
 public:
  // The graph and cost model are owned so schedules cannot outlive or drift
  // from caller-owned inputs.  The lvalue overload validates every configured
  // byte/count limit before cloning either input.  Moving both inputs avoids
  // the clone while preserving the same fail-closed preflight.
  CpuEncounterExecutor(const EncounterDAG& dag,
                       const CpuExecutorConfig& config);
  CpuEncounterExecutor(EncounterDAG&& dag, CpuExecutorConfig&& config);
  ~CpuEncounterExecutor() = default;
  CpuEncounterExecutor(const CpuEncounterExecutor&) = delete;
  CpuEncounterExecutor& operator=(const CpuEncounterExecutor&) = delete;

  CpuExecutionReport execute(EncounterCallback callback,
                             void* callback_data,
                             EncounterMergeCallback merge = nullptr,
                             void* merge_data = nullptr);

  const EncounterDAG& dag() const noexcept { return dag_; }
  const CpuExecutorConfig& config() const noexcept { return config_; }
  // Conservative peak (persistent storage plus schedule-construction scratch).
  std::uint64_t preflight_bytes() const noexcept { return preflight_bytes_; }
  bool has_successful_output() const noexcept { return output_valid_; }
  ConstByteLane result_lane(EncounterId encounter) const;
  ConstByteLane emission_lane(EncounterId encounter) const;

 private:
  struct OwnedInputs {
    EncounterDAG dag;
    CpuExecutorConfig config;
    std::uint64_t preflight_bytes = 0;
  };

  struct alignas(64) WorkerExecutionState {
    std::uint64_t attempted = 0;
    std::uint64_t completed = 0;
    EncounterId lowest_failure = kInvalidEncounter;
    std::exception_ptr failure;
  };

  struct alignas(kCpuContextAlignment) ContextStorageBlock {
    std::byte bytes[kCpuContextAlignment];
  };

  explicit CpuEncounterExecutor(OwnedInputs inputs);
  static OwnedInputs checked_copy(const EncounterDAG& dag,
                                  const CpuExecutorConfig& config);
  static OwnedInputs checked_move(EncounterDAG&& dag,
                                  CpuExecutorConfig&& config);
  static std::uint64_t validate_and_preflight(
      const EncounterDAG& dag, const CpuExecutorConfig& config);
  EncounterLogicalTime logical_time(const EncounterNode& node) const noexcept;
  MutableByteLane mutable_result_lane(EncounterId encounter) noexcept;
  MutableByteLane mutable_emission_lane(EncounterId encounter) noexcept;
  void build_schedule();

  const EncounterDAG dag_;
  CpuExecutorConfig config_;
  std::uint64_t preflight_bytes_ = 0;

  std::vector<ContextStorageBlock> context_storage_;
  std::vector<StepContext> contexts_;
  std::vector<std::byte> result_storage_;
  std::vector<std::byte> emission_storage_;
  std::vector<std::uint64_t> assignment_offsets_;
  std::vector<EncounterId> assigned_encounters_;
  std::vector<WorkerExecutionState> worker_states_;
  std::uint32_t context_high_water_bound_ = 0;
  bool output_valid_ = false;
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_CPU_EXECUTOR_CPP_H
