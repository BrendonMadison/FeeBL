#ifndef FEEBL_LEGACY_WAVEFRONT_SCHEDULER_CPP_H
#define FEEBL_LEGACY_WAVEFRONT_SCHEDULER_CPP_H

#include "encounterDAGCPP.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace legacy_wavefront {

// `strict` is a real production execution mode: it changes the implicit
// nested loops into an explicit EncounterDAG traversal while retaining their
// exact diagonal/substep/encounter order.  `parallel` is a reserved execution
// request and must pass the resource gate below before a caller may dispatch
// same-wave encounters concurrently.
enum class ScheduleMode : std::uint8_t {
  Legacy = 0,
  Strict = 1,
  Parallel = 2
};

ScheduleMode parse_schedule_mode(const std::string& value);
const char* schedule_mode_name(ScheduleMode mode) noexcept;

struct DiagonalDescriptor {
  modern::WaveIndex diagonal = 0;
  modern::SliceIndex first_beam1_slice = 0;
  modern::SliceIndex last_beam1_slice = 0;
  modern::SliceIndex first_beam2_slice = 0;
  modern::SliceIndex last_beam2_slice = 0;
  std::size_t encounter_count_per_substep = 0;
};

// Contiguous work unit within one wave.  EncounterDAG assigns node IDs in
// canonical wave order, so [first_encounter, first_encounter+count) is both a
// stable launch range and the strict publication order.  GPU/CPU consumers
// can choose a maximum batch width without rebuilding collision geometry.
struct BatchDescriptor {
  modern::WaveIndex wave = 0;
  modern::WaveIndex diagonal = 0;
  modern::SubstepIndex substep = 0;
  std::size_t position_begin = 0;
  std::size_t encounter_count = 0;
  modern::EncounterId first_encounter = modern::kInvalidEncounter;
};

struct StrictCallbacks {
  void* context = nullptr;
  void (*begin_diagonal)(void*, const DiagonalDescriptor&) = nullptr;
  void (*begin_wave)(void*, const modern::EncounterWave&) = nullptr;
  void (*encounter)(void*, const modern::EncounterNode&) = nullptr;
  void (*end_wave)(void*, const modern::EncounterWave&) = nullptr;
  void (*end_diagonal)(void*, const DiagonalDescriptor&) = nullptr;
};

struct ExecutionReport {
  ScheduleMode requested = ScheduleMode::Legacy;
  ScheduleMode actual = ScheduleMode::Legacy;
  std::uint64_t diagonals = 0;
  std::uint64_t waves = 0;
  std::uint64_t batches = 0;
  std::uint64_t encounters = 0;
  std::uint64_t parallel_physics_encounters = 0;
  std::size_t maximum_wave_width = 0;
  bool canonical_order = false;
};

// Every mutable object named here is shared by the present legacy FEEBL
// make_step.  Slice exclusivity alone is therefore insufficient for safe
// concurrency.  This explicit contract prevents a future caller from
// mistaking a DAG descriptor for permission to execute it in parallel.
struct ParallelResources {
  bool worker_private_grid = false;
  bool worker_private_field_solver = false;
  bool schedule_independent_rng = false;
  bool deterministic_result_lanes = false;
  bool worker_private_timers = false;
};

struct ParallelCapability {
  bool admitted = false;
  std::string reason;
};

ParallelCapability assess_parallel_capability(
    const ParallelResources& resources);

class Scheduler {
 public:
  Scheduler(modern::SliceIndex beam1_slices,
            modern::SliceIndex beam2_slices,
            modern::SubstepIndex substeps);

  const modern::EncounterDAG& dag() const noexcept { return dag_; }
  const std::vector<DiagonalDescriptor>& diagonals() const noexcept {
    return diagonals_;
  }

  // max_batch_width==0 means one batch containing each complete wave.
  std::vector<BatchDescriptor> make_batches(
      std::size_t max_batch_width = 0) const;

  // Execute the production-compatible ordered traversal.  The encounter
  // callback is mandatory; all boundary hooks are optional.  Exceptions are
  // deliberately propagated, so a partially completed collision is never
  // reported as successful.
  ExecutionReport execute_strict(const StrictCallbacks& callbacks,
                                 std::size_t max_batch_width = 0) const;

 private:
  modern::EncounterDAG dag_;
  std::vector<DiagonalDescriptor> diagonals_;
};

}  // namespace legacy_wavefront
}  // namespace feebl

#endif  // FEEBL_LEGACY_WAVEFRONT_SCHEDULER_CPP_H
