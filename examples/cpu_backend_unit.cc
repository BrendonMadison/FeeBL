#include "parallelBackendCPP.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct STATIC_RANGE_PROBE
{
  std::vector<int>* owner;
  std::vector<int>* lane_owner;
  std::vector<int>* lane_calls;
  std::vector<std::thread::id>* thread_ids;
  int increment;
  bool throw_on_seven;
};

void probe_static_range(void* opaque, std::size_t lane,
                        std::size_t begin, std::size_t end)
{
  STATIC_RANGE_PROBE& probe = *static_cast<STATIC_RANGE_PROBE*>(opaque);
  ++(*probe.lane_calls)[lane];
  (*probe.thread_ids)[lane] = std::this_thread::get_id();
  if (probe.throw_on_seven)
    {
      if (begin <= 7 && 7 < end)
        throw std::runtime_error("expected static-range worker error");
      return;
    }
  for (std::size_t index = begin; index < end; ++index)
    {
      (*probe.owner)[index] += static_cast<int>(begin) + probe.increment;
      (*probe.lane_owner)[index] = static_cast<int>(lane);
    }
}

void no_op_static_range(void*, std::size_t, std::size_t, std::size_t) {}

struct NESTED_RANGE_PROBE
{
  PARALLEL_BACKEND* backend;
  bool attempted;
};

void probe_nested_range(void* opaque, std::size_t lane,
                        std::size_t, std::size_t)
{
  if (lane != 0) return;
  NESTED_RANGE_PROBE& probe = *static_cast<NESTED_RANGE_PROBE*>(opaque);
  probe.attempted = true;
  probe.backend->run_static_ranges(1, NULL, &no_op_static_range);
}

}  // namespace

int main()
{
  PARALLEL_BACKEND backend;
  backend.configure("cpu", 4, "pool_private", "deterministic", false);
  if (!backend.cpu_enabled() || backend.actual_cpu_threads() != 4 ||
      backend.actual_cpu_variant() != "pool_private" ||
      backend.requested_gpu_mode() != 1 || backend.actual_gpu_mode() != 0)
    {
      std::cerr << "initial backend contract mismatch: cpu_enabled="
                << backend.cpu_enabled()
                << " actual_cpu_threads=" << backend.actual_cpu_threads()
                << " actual_cpu_variant=" << backend.actual_cpu_variant()
                << " requested_gpu_mode=" << backend.requested_gpu_mode()
                << " actual_gpu_mode=" << backend.actual_gpu_mode()
                << std::endl;
      return 1;
    }

  std::vector<int> seen(4096, 0);
  backend.run_tasks(seen.size(), [&](std::size_t i) { seen[i] += 1; });
  for (std::size_t i = 0; i < seen.size(); ++i)
    if (seen[i] != 1) return 2;

  // A second generation exercises worker reuse rather than construction.
  backend.run_tasks(seen.size(), [&](std::size_t i) { seen[i] += 2; });
  for (std::size_t i = 0; i < seen.size(); ++i)
    if (seen[i] != 3) return 3;

  bool propagated = false;
  try
    {
      backend.run_tasks(32, [](std::size_t i) {
        if (i == 7) throw std::runtime_error("expected worker error");
      });
    }
  catch (const std::runtime_error& error)
    {
      propagated = std::string(error.what()) == "expected worker error";
    }
  if (!propagated) return 4;

  // The pool must remain usable after propagating an exception.
  int recovered = 0;
  backend.run_tasks(1, [&](std::size_t) { recovered = 1; });
  if (recovered != 1) return 5;

  // The production mover uses this allocation-free dispatch form. With ten
  // items and four workers the stable balanced ranges are [0,3), [3,6),
  // [6,8), and [8,10); each item records its range owner below.
  std::vector<int> range_owner(10, 0);
  std::vector<int> lane_owner(10, -1);
  std::vector<int> lane_calls(4, 0);
  std::vector<std::thread::id> first_thread_ids(4);
  STATIC_RANGE_PROBE range_probe = {
      &range_owner, &lane_owner, &lane_calls, &first_thread_ids, 1, false};
  backend.run_static_ranges(range_owner.size(), &range_probe,
                            &probe_static_range);
  const int expected_owner[] = {1, 1, 1, 4, 4, 4, 7, 7, 9, 9};
  const int expected_lane[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
  for (std::size_t i = 0; i < range_owner.size(); ++i)
    if (range_owner[i] != expected_owner[i] ||
        lane_owner[i] != expected_lane[i]) return 20;
  for (std::size_t lane = 0; lane < lane_calls.size(); ++lane)
    if (lane_calls[lane] != 1 ||
        first_thread_ids[lane] == std::thread::id()) return 25;

  // A later generation reuses the same worker team, lane identity, and
  // boundaries.  Lane identity is reusable for preallocated per-lane state.
  std::vector<std::thread::id> second_thread_ids(4);
  range_probe.thread_ids = &second_thread_ids;
  backend.run_static_ranges(range_owner.size(), &range_probe,
                            &probe_static_range);
  for (std::size_t i = 0; i < range_owner.size(); ++i)
    if (range_owner[i] != 2 * expected_owner[i]) return 21;
  for (std::size_t lane = 0; lane < lane_calls.size(); ++lane)
    if (lane_calls[lane] != 2 ||
        first_thread_ids[lane] != second_thread_ids[lane]) return 26;

  // A memory-limited caller can own fewer lane-private workspaces than the
  // persistent pool has threads.  Partition the whole input once across only
  // those admitted stable lanes; inactive pool workers must not call back.
  std::vector<int> limited_owner(160, 0);
  std::vector<int> limited_lanes(160, -1);
  std::vector<int> limited_calls(4, 0);
  std::vector<std::thread::id> limited_thread_ids(4);
  STATIC_RANGE_PROBE limited_probe = {
      &limited_owner, &limited_lanes, &limited_calls, &limited_thread_ids,
      1, false};
  backend.run_static_ranges_limited(
      limited_owner.size(), 2, &limited_probe, &probe_static_range);
  for (std::size_t i = 0; i < limited_owner.size(); ++i)
    {
      const int expected_limited_lane = i < 80 ? 0 : 1;
      const int expected_limited_owner = i < 80 ? 1 : 81;
      if (limited_lanes[i] != expected_limited_lane ||
          limited_owner[i] != expected_limited_owner) return 32;
    }
  if (limited_calls[0] != 1 || limited_calls[1] != 1 ||
      limited_calls[2] != 0 || limited_calls[3] != 0) return 33;

  bool invalid_limited_count_rejected = false;
  try
    {
      backend.run_static_ranges_limited(
          1, 5, &limited_probe, &probe_static_range);
    }
  catch (const std::invalid_argument&)
    {
      invalid_limited_count_rejected = true;
    }
  if (!invalid_limited_count_rejected) return 34;

  invalid_limited_count_rejected = false;
  try
    {
      backend.run_static_ranges_limited(
          1, 0, &limited_probe, &probe_static_range);
    }
  catch (const std::invalid_argument&)
    {
      invalid_limited_count_rejected = true;
    }
  if (!invalid_limited_count_rejected) return 35;

  // Empty dispatch is a true no-op: no callback and no new generation work.
  const std::vector<int> calls_before_empty = lane_calls;
  backend.run_static_ranges(0, &range_probe, &probe_static_range);
  if (lane_calls != calls_before_empty) return 27;

  range_probe.throw_on_seven = true;
  bool static_error_propagated = false;
  try
    {
      backend.run_static_ranges(range_owner.size(), &range_probe,
                                &probe_static_range);
    }
  catch (const std::runtime_error& error)
    {
      static_error_propagated =
          std::string(error.what()) == "expected static-range worker error";
    }
  if (!static_error_propagated) return 22;
  range_probe.throw_on_seven = false;
  backend.run_static_ranges(1, &range_probe, &probe_static_range);
  if (range_owner[0] != 2 * expected_owner[0] + 1) return 23;

  bool null_static_task_rejected = false;
  try
    {
      backend.run_static_ranges(1, &range_probe, NULL);
    }
  catch (const std::invalid_argument&)
    {
      null_static_task_rejected = true;
    }
  if (!null_static_task_rejected) return 24;

  bool nested_static_task_rejected = false;
  NESTED_RANGE_PROBE nested_probe = {&backend, false};
  try
    {
      backend.run_static_ranges(4, &nested_probe, &probe_nested_range);
    }
  catch (const std::logic_error& error)
    {
      nested_static_task_rejected =
          std::string(error.what()).find("nested CPU thread-pool dispatch") !=
          std::string::npos;
    }
  if (!nested_probe.attempted || !nested_static_task_rejected) return 28;

  // The serial implementation uses the same API contract with stable lane 0.
  PARALLEL_BACKEND serial_ranges;
  std::vector<int> serial_owner(3, 0);
  std::vector<int> serial_lanes(3, -1);
  std::vector<int> serial_calls(1, 0);
  std::vector<std::thread::id> serial_threads(1);
  STATIC_RANGE_PROBE serial_probe = {
      &serial_owner, &serial_lanes, &serial_calls, &serial_threads, 1, false};
  serial_ranges.run_static_ranges(serial_owner.size(), &serial_probe,
                                  &probe_static_range);
  if (serial_calls[0] != 1) return 29;
  for (std::size_t i = 0; i < serial_lanes.size(); ++i)
    if (serial_lanes[i] != 0) return 30;

  // Repeated construction, dispatch, and destruction exercises worker-team
  // lifecycle and join behavior independently of the long-lived backend.
  for (int lifecycle = 0; lifecycle < 3; ++lifecycle)
    {
      PARALLEL_BACKEND scoped;
      scoped.configure("cpu", 2, "pool_private", "deterministic", false);
      std::vector<int> scoped_owner(5, 0);
      std::vector<int> scoped_lanes(5, -1);
      std::vector<int> scoped_calls(2, 0);
      std::vector<std::thread::id> scoped_threads(2);
      STATIC_RANGE_PROBE scoped_probe = {
          &scoped_owner, &scoped_lanes, &scoped_calls, &scoped_threads,
          1, false};
      scoped.run_static_ranges(scoped_owner.size(), &scoped_probe,
                               &probe_static_range);
      if (scoped_calls[0] != 1 || scoped_calls[1] != 1) return 31;
    }

  // Strict mode is the byte-compatible contract: a regrouped private-grid
  // reduction must never be selected silently.
  bool strict_private_rejected = false;
  try
    {
      PARALLEL_BACKEND invalid;
      invalid.configure("cpu", 4, "pool_private", "strict", false);
    }
  catch (const std::runtime_error& error)
    {
      strict_private_rejected =
          std::string(error.what()).find("strict reduction requires") !=
          std::string::npos;
    }
  if (!strict_private_rejected) return 6;

  PARALLEL_BACKEND strict_fallback;
  strict_fallback.configure("cpu", 4, "pool_private", "strict", true);
  if (strict_fallback.actual_cpu_variant() != "pool_dual" ||
      strict_fallback.fallback_reason().empty())
    return 7;

  PARALLEL_BACKEND strict_auto;
  strict_auto.configure("cpu", 2, "auto", "strict", false);
  if (strict_auto.actual_cpu_variant() != "pool_dual") return 8;

  // Production auto follows the preregistered selection gate.  None of the
  // current CIC-only candidates demonstrated a >=10% end-to-end speedup, so
  // it must retain the serial oracle rather than manufacture a winner.
  PARALLEL_BACKEND production_auto;
  production_auto.configure("auto", 0, "auto", "strict", true);
  if (!production_auto.serial_enabled() ||
      production_auto.fallback_reason().find("10% speedup gate") ==
          std::string::npos)
    return 9;

  // Mode-2-only controls must not be silently consumed by legacy mode 1.
  bool mixed_modes_rejected = false;
  try
    {
      GPU2_OPTIONS mixed;
      mixed.streams = 2;
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, mixed);
    }
  catch (const std::invalid_argument& error)
    {
      mixed_modes_rejected =
          std::string(error.what()).find("gpu2_* controls require") !=
          std::string::npos;
    }
  if (!mixed_modes_rejected) return 10;

  bool bad_fraction_rejected = false;
  try
    {
      GPU2_OPTIONS bad;
      bad.mode = 2;
      bad.memory_fraction = 1.01;
      PARALLEL_BACKEND invalid;
      invalid.configure("gpu", 0, "auto", "auto", 0, 262144,
                        "legacy", "fast", false, bad);
    }
  catch (const std::invalid_argument& error)
    {
      bad_fraction_rejected =
          std::string(error.what()).find("gpu2_memory_fraction") !=
          std::string::npos;
    }
  if (!bad_fraction_rejected) return 11;

  // Modern defaults must be entirely invisible to legacy provenance.  This is
  // the byte-compatibility guard for old cards and output parsers.
  if (backend.provenance().find("engine requested=") != std::string::npos ||
      backend.modern_provenance_active())
    return 12;

  bool bad_algorithm_rejected = false;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.algorithm = "approximate";
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, GPU2_OPTIONS(), modern);
    }
  catch (const std::invalid_argument& error)
    {
      bad_algorithm_rejected =
          std::string(error.what()).find("algorithm must be") !=
          std::string::npos;
    }
  if (!bad_algorithm_rejected) return 13;

  bool legacy_modern_control_rejected = false;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.batch_width = 64;
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, GPU2_OPTIONS(), modern);
    }
  catch (const std::invalid_argument& error)
    {
      legacy_modern_control_rejected =
          std::string(error.what()).find("engine=legacy accepts only") !=
          std::string::npos;
    }
  if (!legacy_modern_control_rejected) return 14;

  bool modern_requires_counter = false;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.engine = "modern_cpu";
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, GPU2_OPTIONS(), modern);
    }
  catch (const std::invalid_argument& error)
    {
      modern_requires_counter =
          std::string(error.what()).find("requires rng_backend=counter") !=
          std::string::npos;
    }
  if (!modern_requires_counter) return 15;

  bool modern_cpu_unwired = false;
  PARALLEL_BACKEND unavailable_cpu;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.engine = "MODERN_CPU";  // Values are deliberately case-insensitive.
      modern.algorithm = "factored_exact";
      modern.batch_width = 128;
      modern.cpu_workers = 4;
      modern.cpu_runtime = "pool";
      modern.cpu_affinity = "compact";
      unavailable_cpu.configure("serial", 0, "auto", "auto", 0, 262144,
                                "counter", "strict", false,
                                GPU2_OPTIONS(), modern);
    }
  catch (const std::runtime_error& error)
    {
      modern_cpu_unwired =
          std::string(error.what()).find("no modern physics engine is wired") !=
          std::string::npos;
    }
  if (!modern_cpu_unwired || unavailable_cpu.actual_engine() != "unavailable" ||
      unavailable_cpu.provenance().find(
          "engine requested=modern_cpu actual=unavailable") ==
          std::string::npos ||
      unavailable_cpu.provenance().find(
          "modern_reduction requested=deterministic actual=inactive") ==
          std::string::npos)
    return 16;

  bool legacy_mode3_rejected = false;
  try
    {
      GPU2_OPTIONS gpu3;
      gpu3.mode = 3;
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, gpu3);
    }
  catch (const std::invalid_argument& error)
    {
      legacy_mode3_rejected =
          std::string(error.what()).find("requires engine=modern_gpu") !=
          std::string::npos;
    }
  if (!legacy_mode3_rejected) return 17;

  bool modern_gpu_requires_mode3 = false;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.engine = "modern_gpu";
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, GPU2_OPTIONS(), modern);
    }
  catch (const std::invalid_argument& error)
    {
      modern_gpu_requires_mode3 =
          std::string(error.what()).find("requires gpu_mode=3") !=
          std::string::npos;
    }
  if (!modern_gpu_requires_mode3) return 18;

  bool modern_gpu_cuda_unavailable = false;
  try
    {
      MODERN_RUNTIME_OPTIONS modern;
      modern.engine = "modern_gpu";
      modern.algorithm = "exact";
      modern.precision = "mixed_qualified";
      modern.reduction = "fast_qualified";
      modern.gpu_streams = 1;
      modern.gpu_graphs = "off";
      modern.trace = "summary";
      GPU2_OPTIONS gpu3;
      gpu3.mode = 3;
      gpu3.scope = "primary";
      gpu3.deposition = "atomic";
      gpu3.fft = "cufft";
      gpu3.streams = 1;
      gpu3.graphs = "off";
      gpu3.fusion = "off";
      gpu3.batch_slices = 0;
      gpu3.pinned_io = "on";
      gpu3.host_threads = 1;
      gpu3.primary_mover = "serial";
      gpu3.profile = "summary";
      PARALLEL_BACKEND invalid;
      invalid.configure("serial", 0, "auto", "auto", 0, 262144,
                        "legacy", "strict", false, gpu3, modern);
    }
  catch (const std::runtime_error& error)
    {
      modern_gpu_cuda_unavailable =
          std::string(error.what()).find("no resident CUDA primary pipeline") !=
          std::string::npos;
    }
  if (!modern_gpu_cuda_unavailable) return 19;

  std::cout << backend.provenance() << std::endl;
  return 0;
}
