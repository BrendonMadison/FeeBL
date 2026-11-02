#include "legacyWavefrontSchedulerCPP.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using feebl::legacy_wavefront::BatchDescriptor;
using feebl::legacy_wavefront::DiagonalDescriptor;
using feebl::legacy_wavefront::ExecutionReport;
using feebl::legacy_wavefront::ParallelResources;
using feebl::legacy_wavefront::ScheduleMode;
using feebl::legacy_wavefront::Scheduler;
using feebl::legacy_wavefront::StrictCallbacks;
using feebl::legacy_wavefront::assess_parallel_capability;
using feebl::legacy_wavefront::parse_schedule_mode;
using feebl::modern::EncounterNode;
using feebl::modern::EncounterWave;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-78s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

template <class Function>
bool throws(Function function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

struct Trace {
  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> nodes;
  std::vector<std::string> boundaries;
};

void begin_diagonal(void* opaque, const DiagonalDescriptor& diagonal) {
  Trace& trace = *static_cast<Trace*>(opaque);
  trace.boundaries.push_back("D+" + std::to_string(diagonal.diagonal));
}

void begin_wave(void* opaque, const EncounterWave& wave) {
  Trace& trace = *static_cast<Trace*>(opaque);
  trace.boundaries.push_back("W+" + std::to_string(wave.index));
}

void encounter(void* opaque, const EncounterNode& node) {
  Trace& trace = *static_cast<Trace*>(opaque);
  trace.nodes.emplace_back(node.beam1_slice, node.beam2_slice, node.substep);
}

void end_wave(void* opaque, const EncounterWave& wave) {
  Trace& trace = *static_cast<Trace*>(opaque);
  trace.boundaries.push_back("W-" + std::to_string(wave.index));
}

void end_diagonal(void* opaque, const DiagonalDescriptor& diagonal) {
  Trace& trace = *static_cast<Trace*>(opaque);
  trace.boundaries.push_back("D-" + std::to_string(diagonal.diagonal));
}

void test_fccz_geometry_and_batches() {
  std::printf("\nFCC-Z wavefront geometry and GPU-consumable batches\n");
  const Scheduler scheduler(160, 160, 1);
  const std::vector<BatchDescriptor> full = scheduler.make_batches();
  const std::vector<BatchDescriptor> width32 = scheduler.make_batches(32);
  std::uint64_t full_encounters = 0;
  std::size_t maximum_width = 0;
  for (const BatchDescriptor& batch : full) {
    full_encounters += batch.encounter_count;
    maximum_width = std::max(maximum_width, batch.encounter_count);
  }
  check(scheduler.diagonals().size() == 319 &&
            scheduler.dag().wave_count() == 319,
        "n_z=160 has 319 anti-diagonals/waves");
  check(scheduler.dag().encounter_count() == 25600 &&
            full_encounters == 25600,
        "n_z=160 has exactly 25,600 exhaustive encounters");
  check(full.size() == 319 && maximum_width == 160,
        "full-wave descriptors expose 319 launches with maximum width 160");
  check(width32.size() > full.size() &&
            std::all_of(width32.begin(), width32.end(),
                        [](const BatchDescriptor& batch) {
                          return batch.encounter_count > 0 &&
                                 batch.encounter_count <= 32;
                        }),
        "bounded descriptors split waves without exceeding the launch width");
}

void test_strict_order_and_report() {
  std::printf("\nStrict production traversal and provenance\n");
  const Scheduler scheduler(4, 3, 2);
  Trace trace;
  StrictCallbacks callbacks;
  callbacks.context = &trace;
  callbacks.begin_diagonal = begin_diagonal;
  callbacks.begin_wave = begin_wave;
  callbacks.encounter = encounter;
  callbacks.end_wave = end_wave;
  callbacks.end_diagonal = end_diagonal;
  const ExecutionReport report = scheduler.execute_strict(callbacks, 2);

  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> expected;
  for (std::uint64_t diagonal = 0; diagonal < 6; ++diagonal) {
    const std::uint64_t first_i = diagonal >= 2 ? diagonal - 2 : 0;
    const std::uint64_t last_i = std::min<std::uint64_t>(diagonal, 3);
    for (std::uint64_t substep = 0; substep < 2; ++substep)
      for (std::uint64_t i = first_i; i <= last_i; ++i)
        expected.emplace_back(i, diagonal - i, substep);
  }
  check(trace.nodes == expected,
        "strict traversal is diagonal/substep/increasing-beam1 canonical order");
  check(report.actual == ScheduleMode::Strict && report.diagonals == 6 &&
            report.waves == 12 && report.encounters == 24 &&
            report.parallel_physics_encounters == 0 &&
            report.canonical_order,
        "report distinguishes ordered execution from parallel physics work");
  check(trace.boundaries.front() == "D+0" &&
            trace.boundaries[1] == "W+0" &&
            trace.boundaries[2] == "W-0" &&
            trace.boundaries.back() == "D-5",
        "diagonal and wave hooks bracket encounter work deterministically");
  check(throws([&] {
          StrictCallbacks missing;
          (void)scheduler.execute_strict(missing);
        }),
        "a missing production encounter callback fails closed");
}

void test_selectors_and_parallel_gate() {
  std::printf("\nExplicit selectors and mutable-state capability gate\n");
  check(parse_schedule_mode("legacy") == ScheduleMode::Legacy &&
            parse_schedule_mode("wavefront_strict") == ScheduleMode::Strict &&
            parse_schedule_mode("wavefront_parallel") == ScheduleMode::Parallel,
        "all documented collision_schedule selectors parse distinctly");
  check(throws([] { (void)parse_schedule_mode("parallel-ish"); }),
        "an unknown scheduler selector is rejected");

  const auto rejected = assess_parallel_capability(ParallelResources{});
  check(!rejected.admitted &&
            rejected.reason.find("GRID") != std::string::npos &&
            rejected.reason.find("RNG") != std::string::npos &&
            rejected.reason.find("No fallback") != std::string::npos,
        "current shared GRID/RNG/result state gives an actionable refusal");
  ParallelResources isolated;
  isolated.worker_private_grid = true;
  isolated.worker_private_field_solver = true;
  isolated.schedule_independent_rng = true;
  isolated.deterministic_result_lanes = true;
  isolated.worker_private_timers = true;
  check(assess_parallel_capability(isolated).admitted,
        "the gate documents the complete resource contract for future dispatch");
}

}  // namespace

int main() {
  test_fccz_geometry_and_batches();
  test_strict_order_and_report();
  test_selectors_and_parallel_gate();
  if (failures != 0) {
    std::fprintf(stderr, "legacy wavefront scheduler failures: %d\n", failures);
    return 1;
  }
  std::printf("\nAll legacy wavefront scheduler tests passed.\n");
  return 0;
}
