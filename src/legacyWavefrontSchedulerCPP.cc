#include "legacyWavefrontSchedulerCPP.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace feebl {
namespace legacy_wavefront {

ScheduleMode parse_schedule_mode(const std::string& value) {
  if (value == "legacy") return ScheduleMode::Legacy;
  if (value == "wavefront_strict") return ScheduleMode::Strict;
  if (value == "wavefront_parallel") return ScheduleMode::Parallel;
  throw std::invalid_argument(
      "collision_schedule must be legacy, wavefront_strict, or "
      "wavefront_parallel");
}

const char* schedule_mode_name(ScheduleMode mode) noexcept {
  switch (mode) {
    case ScheduleMode::Legacy: return "legacy";
    case ScheduleMode::Strict: return "wavefront_strict";
    case ScheduleMode::Parallel: return "wavefront_parallel";
  }
  return "invalid";
}

ParallelCapability assess_parallel_capability(
    const ParallelResources& resources) {
  std::vector<const char*> missing;
  if (!resources.worker_private_grid)
    missing.push_back("worker-private GRID/cell scratch");
  if (!resources.worker_private_field_solver)
    missing.push_back("worker-private or batched FIELD/FFT state");
  if (!resources.schedule_independent_rng)
    missing.push_back("schedule-independent RNG streams");
  if (!resources.deterministic_result_lanes)
    missing.push_back("task-local deterministic result/event lanes");
  if (!resources.worker_private_timers)
    missing.push_back("worker-private timing counters");

  ParallelCapability result;
  result.admitted = missing.empty();
  if (result.admitted) {
    result.reason = "all same-wave mutable-state resources are isolated";
    return result;
  }

  std::ostringstream reason;
  reason << "same-wave make_step execution is unavailable; missing ";
  for (std::size_t index = 0; index < missing.size(); ++index) {
    if (index != 0) reason << ", ";
    reason << missing[index];
  }
  reason << ". No fallback was performed";
  result.reason = reason.str();
  return result;
}

Scheduler::Scheduler(modern::SliceIndex beam1_slices,
                     modern::SliceIndex beam2_slices,
                     modern::SubstepIndex substeps)
    : dag_(modern::EncounterDAG::build_legacy(
          beam1_slices, beam2_slices, substeps)) {
  const modern::ValidationResult validation = dag_.validate();
  if (!validation) {
    std::ostringstream message;
    message << "invalid legacy EncounterDAG at index " << validation.index
            << ": " << validation.message;
    throw std::logic_error(message.str());
  }

  if (beam1_slices == 0 || beam2_slices == 0) return;
  const std::uint64_t diagonal_count = beam1_slices + beam2_slices - 1;
  if (diagonal_count > diagonals_.max_size())
    throw std::length_error("legacy wavefront diagonal storage is too large");
  diagonals_.reserve(static_cast<std::size_t>(diagonal_count));
  for (modern::WaveIndex diagonal = 0; diagonal < diagonal_count;
       ++diagonal) {
    DiagonalDescriptor descriptor;
    descriptor.diagonal = diagonal;
    descriptor.first_beam1_slice =
        diagonal >= beam2_slices - 1 ? diagonal - (beam2_slices - 1) : 0;
    descriptor.last_beam1_slice =
        std::min<modern::SliceIndex>(diagonal, beam1_slices - 1);
    descriptor.first_beam2_slice =
        diagonal - descriptor.last_beam1_slice;
    descriptor.last_beam2_slice =
        diagonal - descriptor.first_beam1_slice;
    descriptor.encounter_count_per_substep = static_cast<std::size_t>(
        descriptor.last_beam1_slice - descriptor.first_beam1_slice + 1);
    diagonals_.push_back(descriptor);
  }
}

std::vector<BatchDescriptor> Scheduler::make_batches(
    std::size_t max_batch_width) const {
  std::vector<BatchDescriptor> batches;
  std::uint64_t batch_count = 0;
  for (const modern::EncounterWave& wave : dag_.waves()) {
    const std::size_t width = wave.encounters.size();
    const std::size_t effective_width =
        max_batch_width == 0 ? width : max_batch_width;
    const std::size_t count = width == 0
        ? 0 : 1 + (width - 1) / effective_width;
    if (batch_count > std::numeric_limits<std::uint64_t>::max() - count)
      throw std::overflow_error("legacy wavefront batch count overflows");
    batch_count += count;
  }
  if (batch_count > batches.max_size())
    throw std::length_error("legacy wavefront batch storage is too large");
  batches.reserve(static_cast<std::size_t>(batch_count));

  for (const modern::EncounterWave& wave : dag_.waves()) {
    const std::size_t width = wave.encounters.size();
    const std::size_t effective_width =
        max_batch_width == 0 ? width : max_batch_width;
    for (std::size_t begin = 0; begin < width; begin += effective_width) {
      BatchDescriptor descriptor;
      descriptor.wave = wave.index;
      descriptor.diagonal = wave.diagonal;
      descriptor.substep = wave.substep;
      descriptor.position_begin = begin;
      descriptor.encounter_count = std::min(effective_width, width - begin);
      descriptor.first_encounter = wave.encounters[begin];
      batches.push_back(descriptor);
    }
  }
  return batches;
}

ExecutionReport Scheduler::execute_strict(
    const StrictCallbacks& callbacks,
    std::size_t max_batch_width) const {
  if (!callbacks.encounter)
    throw std::invalid_argument(
        "legacy wavefront strict execution requires an encounter callback");

  const std::vector<BatchDescriptor> batches =
      make_batches(max_batch_width);
  ExecutionReport report;
  report.requested = ScheduleMode::Strict;
  report.actual = ScheduleMode::Strict;
  report.diagonals = diagonals_.size();
  report.waves = dag_.wave_count();
  report.batches = batches.size();
  report.canonical_order = true;

  std::size_t batch_index = 0;
  for (const DiagonalDescriptor& diagonal : diagonals_) {
    if (callbacks.begin_diagonal)
      callbacks.begin_diagonal(callbacks.context, diagonal);
    for (modern::SubstepIndex substep = 0;
         substep < dag_.substep_count(); ++substep) {
      const modern::WaveIndex wave_index =
          diagonal.diagonal * dag_.substep_count() + substep;
      const modern::EncounterWave& wave =
          dag_.waves()[static_cast<std::size_t>(wave_index)];
      if (callbacks.begin_wave)
        callbacks.begin_wave(callbacks.context, wave);

      while (batch_index < batches.size() &&
             batches[batch_index].wave == wave.index) {
        const BatchDescriptor& batch = batches[batch_index++];
        const std::size_t end =
            batch.position_begin + batch.encounter_count;
        for (std::size_t position = batch.position_begin;
             position < end; ++position) {
          const modern::EncounterNode& node =
              dag_.node(wave.encounters[position]);
          callbacks.encounter(callbacks.context, node);
          ++report.encounters;
          report.maximum_wave_width = std::max(
              report.maximum_wave_width, wave.encounters.size());
        }
      }

      if (callbacks.end_wave)
        callbacks.end_wave(callbacks.context, wave);
    }
    if (callbacks.end_diagonal)
      callbacks.end_diagonal(callbacks.context, diagonal);
  }

  if (batch_index != batches.size() ||
      report.encounters != dag_.encounter_count())
    throw std::logic_error(
        "legacy wavefront strict execution was not exhaustive");
  // Strict mode intentionally publishes every stateful effect in historical
  // order.  It does not claim same-wave physics concurrency.
  report.parallel_physics_encounters = 0;
  return report;
}

}  // namespace legacy_wavefront
}  // namespace feebl
