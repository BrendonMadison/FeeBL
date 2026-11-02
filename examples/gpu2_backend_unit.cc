#include "parallelBackendCPP.h"

#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct RANGE_PROBE
{
  std::vector<int>* owners;
  std::vector<int>* calls;
  std::vector<std::thread::id>* threads;
};

void record_range(void* opaque, std::size_t lane,
                  std::size_t begin, std::size_t end)
{
  RANGE_PROBE& probe = *static_cast<RANGE_PROBE*>(opaque);
  ++(*probe.calls)[lane];
  (*probe.threads)[lane] = std::this_thread::get_id();
  for (std::size_t index = begin; index < end; ++index)
    (*probe.owners)[index] = static_cast<int>(lane);
}

}  // namespace

int main()
{
  GPU2_OPTIONS requested;
  requested.mode = 2;
  requested.scope = "auto";
  requested.deposition = "auto";
  requested.graphs = "auto";
  requested.fusion = "auto";
  requested.pinned_io = "auto";
  requested.profile = "summary";

  PARALLEL_BACKEND backend;
  backend.configure("gpu", 0, "auto", "auto", 0, 262144,
                    "legacy", "fast", false, requested);
  if (!backend.gpu_mode2_enabled() || backend.gpu_mode1_enabled() ||
      backend.gpu2_plan_committed() || backend.gpu2_calls() != 0)
    return 1;

  bool premature_execution_rejected = false;
  try
    {
      backend.record_gpu2_execution("test device");
    }
  catch (const std::logic_error&)
    {
      premature_execution_rejected = true;
    }
  if (!premature_execution_rejected) return 2;

  GPU2_OPTIONS plan = requested;
  plan.scope = "fields";
  plan.deposition = "host_rho";
  plan.streams = 2;
  plan.graphs = "off";
  plan.fusion = "off";
  plan.batch_slices = 4;
  plan.pinned_io = "on";
  plan.host_threads = 1;
  backend.record_gpu2_plan(plan);
  backend.record_gpu2_execution("test device");
  if (!backend.gpu2_plan_committed() || backend.gpu2_calls() != 1 ||
      backend.actual_gpu2_options().scope != "fields" ||
      backend.actual_gpu2_options().deposition != "host_rho")
    return 3;

  const std::string provenance = backend.provenance();
  if (provenance.find("gpu_mode requested=2 actual=2") == std::string::npos ||
      provenance.find("gpu2_plan=committed gpu2_calls=1") == std::string::npos ||
      provenance.find("gpu2_host_boundary=phi_materialization") ==
          std::string::npos)
    return 4;

  bool wider_host_rho_rejected = false;
  try
    {
      PARALLEL_BACKEND wider;
      wider.configure("gpu", 0, "auto", "auto", 0, 262144,
                      "legacy", "fast", false, requested);
      GPU2_OPTIONS invalid = plan;
      invalid.scope = "primary";
      wider.record_gpu2_plan(invalid);
    }
  catch (const std::invalid_argument&)
    {
      wider_host_rho_rejected = true;
    }
  if (!wider_host_rho_rejected) return 5;

  PARALLEL_BACKEND direct;
  direct.configure("gpu", 0, "auto", "auto", 0, 262144,
                   "legacy", "fast", false, requested);
  GPU2_OPTIONS direct_plan = plan;
  direct_plan.deposition = "atomic";
  direct.record_gpu2_plan(direct_plan);
  direct.record_gpu2_execution("test device");
  if (direct.provenance().find(
          "gpu2_host_boundary=compact_metrics_then_phi_materialization") ==
      std::string::npos)
    return 6;

  PARALLEL_BACKEND primary;
  GPU2_OPTIONS primary_request = requested;
  primary_request.scope = "primary";
  primary.configure("gpu", 0, "auto", "auto", 0, 262144,
                    "legacy", "fast", false, primary_request);
  GPU2_OPTIONS primary_plan = primary_request;
  primary_plan.deposition = "atomic";
  primary_plan.streams = 1;
  primary_plan.graphs = "off";
  primary_plan.fusion = "off";
  primary_plan.batch_slices = 1;
  primary_plan.pinned_io = "off";
  primary_plan.host_threads = 1;
  primary.record_gpu2_plan(primary_plan);
  primary.record_gpu2_execution("test device");
  const std::string primary_provenance = primary.provenance();
  if (primary_provenance.find("gpu2_scope requested=primary actual=primary") ==
          std::string::npos ||
      primary_provenance.find("gpu2_deposition requested=auto actual=atomic") ==
          std::string::npos ||
      primary_provenance.find(
          "gpu2_host_boundary=compact_primary_fields") == std::string::npos ||
      primary_provenance.find("gpu2_primary_update=host_serial") ==
          std::string::npos ||
      primary_provenance.find("gpu2_spin_transport=host_serial") ==
          std::string::npos ||
      primary_provenance.find("gpu2_beamstrahlung=host_serial") ==
          std::string::npos ||
      primary_provenance.find("gpu2_rng=legacy_host_ordered") ==
          std::string::npos ||
      primary_provenance.find("gpu2_host_phi_materialized=0") ==
          std::string::npos)
    return 7;

  bool serial_pool_dispatch_rejected = false;
  try
    {
      primary.run_gpu2_primary_static_ranges(1, NULL, &record_range);
    }
  catch (const std::logic_error&)
    {
      serial_pool_dispatch_rejected = true;
    }
  if (!serial_pool_dispatch_rejected) return 10;

  // A GPU backend can own a separate persistent host pool only after a
  // concrete primary plan commits it.  Its stable combined-beam ranges and
  // lane identities are reused across generations, and observed execution is
  // counted independently from the CUDA calls.
  GPU2_OPTIONS pool_request = primary_request;
  pool_request.primary_mover = "PoOl";  // selectors are case-insensitive
  pool_request.host_threads = 4;
  PARALLEL_BACKEND pool_primary;
  pool_primary.configure("gpu", 0, "auto", "auto", 0, 262144,
                         "legacy", "fast", false, pool_request);
  GPU2_OPTIONS pool_plan = primary_plan;
  pool_plan.primary_mover = "pool";
  pool_plan.host_threads = 4;
  pool_primary.record_gpu2_plan(pool_plan);

  std::vector<int> owners(10, -1);
  std::vector<int> calls(4, 0);
  std::vector<std::thread::id> first_threads(4);
  RANGE_PROBE probe = {&owners, &calls, &first_threads};
  pool_primary.run_gpu2_primary_static_ranges(
      owners.size(), &probe, &record_range);
  const int expected_lanes[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
  for (std::size_t index = 0; index < owners.size(); ++index)
    if (owners[index] != expected_lanes[index]) return 11;
  pool_primary.record_gpu2_primary_mover_execution(
      "pool", owners.size(), 4, 4);

  std::fill(owners.begin(), owners.end(), -1);
  std::vector<std::thread::id> second_threads(4);
  probe.threads = &second_threads;
  pool_primary.run_gpu2_primary_static_ranges(
      owners.size(), &probe, &record_range);
  pool_primary.record_gpu2_primary_mover_execution(
      "POOL", owners.size(), 4, 4);
  for (std::size_t lane = 0; lane < calls.size(); ++lane)
    if (calls[lane] != 2 || first_threads[lane] != second_threads[lane])
      return 12;
  const std::string pool_provenance = pool_primary.provenance();
  if (pool_primary.actual_cpu_variant() != "gpu2_pool_primary" ||
      pool_primary.gpu2_primary_mover_calls() != 2 ||
      pool_primary.gpu2_primary_mover_particles() != 20 ||
      pool_primary.gpu2_primary_mover_ranges() != 8 ||
      pool_primary.gpu2_primary_mover_observed_threads() != 4 ||
      pool_provenance.find(
          "gpu2_primary_mover requested=pool actual=pool") ==
          std::string::npos ||
      pool_provenance.find(
          "gpu2_primary_update=host_pool_static_combined") ==
          std::string::npos ||
      pool_provenance.find(
          "gpu2_rng=unused_by_parallel_contract") == std::string::npos)
    return 13;

  bool wrong_counter_rejected = false;
  try
    {
      pool_primary.record_gpu2_primary_mover_execution("openmp", 1, 4, 1);
    }
  catch (const std::logic_error&)
    {
      wrong_counter_rejected = true;
    }
  if (!wrong_counter_rejected) return 14;

  bool wrong_team_rejected = false;
  try
    {
      pool_primary.record_gpu2_primary_mover_execution("pool", 1, 3, 1);
    }
  catch (const std::logic_error&)
    {
      wrong_team_rejected = true;
    }
  if (!wrong_team_rejected) return 18;

  bool serial_threads_rejected = false;
  try
    {
      PARALLEL_BACKEND invalid_serial;
      invalid_serial.configure("gpu", 0, "auto", "auto", 0, 262144,
                               "legacy", "fast", false, primary_request);
      GPU2_OPTIONS invalid_plan = primary_plan;
      invalid_plan.host_threads = 2;
      invalid_serial.record_gpu2_plan(invalid_plan);
    }
  catch (const std::invalid_argument&)
    {
      serial_threads_rejected = true;
    }
  if (!serial_threads_rejected) return 15;

  bool pool_fields_rejected = false;
  try
    {
      PARALLEL_BACKEND invalid_fields;
      invalid_fields.configure("gpu", 0, "auto", "auto", 0, 262144,
                               "legacy", "fast", false, requested);
      GPU2_OPTIONS invalid_plan = plan;
      invalid_plan.primary_mover = "pool";
      invalid_fields.record_gpu2_plan(invalid_plan);
    }
  catch (const std::invalid_argument&)
    {
      pool_fields_rejected = true;
    }
  if (!pool_fields_rejected) return 16;

#ifdef FEEBL_HAVE_OPENMP
  GPU2_OPTIONS openmp_request = primary_request;
  openmp_request.primary_mover = "openmp";
  openmp_request.host_threads = 2;
  PARALLEL_BACKEND openmp_primary;
  openmp_primary.configure("gpu", 0, "auto", "auto", 0, 262144,
                           "legacy", "fast", false, openmp_request);
  GPU2_OPTIONS openmp_plan = primary_plan;
  openmp_plan.primary_mover = "openmp";
  openmp_plan.host_threads = 2;
  openmp_primary.record_gpu2_plan(openmp_plan);
  openmp_primary.record_gpu2_primary_mover_execution("openmp", 8, 2, 2);
  if (openmp_primary.actual_cpu_variant() != "gpu2_openmp_primary" ||
      openmp_primary.provenance().find(
          "gpu2_primary_update=host_openmp_static_combined") ==
          std::string::npos)
    return 17;
#else
  bool unavailable_openmp_rejected = false;
  try
    {
      GPU2_OPTIONS openmp_request = primary_request;
      openmp_request.primary_mover = "openmp";
      openmp_request.host_threads = 2;
      PARALLEL_BACKEND unavailable_openmp;
      unavailable_openmp.configure(
          "gpu", 0, "auto", "auto", 0, 262144,
          "legacy", "fast", true, openmp_request);
    }
  catch (const std::runtime_error& error)
    {
      unavailable_openmp_rejected =
          std::string(error.what()).find("OpenMP-enabled build") !=
          std::string::npos;
    }
  if (!unavailable_openmp_rejected) return 17;
#endif

  // With explicit fallback authority, the selector may request sort_reduce,
  // but the application must not silently recommit atomic while claiming a
  // deterministic run.  Model FeeBL's preflight catch here and assert the
  // only safe current result: one coherent serial trajectory, zero GPU calls.
  GPU2_OPTIONS deterministic_request = requested;
  deterministic_request.deposition = "atomic";
  PARALLEL_BACKEND deterministic_atomic;
  deterministic_atomic.configure("gpu", 0, "auto", "auto", 0, 262144,
                                 "legacy", "deterministic", true,
                                 deterministic_request);
  GPU2_OPTIONS forbidden_atomic = plan;
  forbidden_atomic.deposition = "atomic";
  bool deterministic_atomic_rejected = false;
  try
    {
      deterministic_atomic.record_gpu2_plan(forbidden_atomic);
    }
  catch (const std::invalid_argument&)
    {
      deterministic_atomic_rejected = true;
      deterministic_atomic.runtime_fallback_to_serial(
          "GPU2 preflight: sort_reduce is not implemented; atomic refused");
    }
  if (!deterministic_atomic_rejected ||
      !deterministic_atomic.serial_enabled() ||
      deterministic_atomic.actual_gpu_mode() != 0 ||
      deterministic_atomic.gpu2_calls() != 0 ||
      deterministic_atomic.provenance().find(
          "sort_reduce is not implemented; atomic refused") ==
          std::string::npos)
    return 8;

  // Mode 3 is a distinct, abort-only whole-crossing contract.  It uses
  // legacy RNG only during initialization and no RNG in the qualified
  // collision; FP32 phase state plus unordered FP64 atomic CIC must be named
  // mixed_qualified/fast_qualified rather than deterministic FP64.
  GPU2_OPTIONS mode3_request;
  mode3_request.mode = 3;
  mode3_request.scope = "primary";
  mode3_request.deposition = "atomic";
  mode3_request.fft = "cufft";
  mode3_request.streams = 1;
  mode3_request.graphs = "off";
  mode3_request.fusion = "off";
  mode3_request.batch_slices = 0;
  mode3_request.pinned_io = "on";
  mode3_request.host_threads = 1;
  mode3_request.primary_mover = "serial";
  mode3_request.profile = "summary";

  MODERN_RUNTIME_OPTIONS mode3_modern;
  mode3_modern.engine = "modern_gpu";
  mode3_modern.algorithm = "exact";
  mode3_modern.precision = "mixed_qualified";
  mode3_modern.reduction = "fast_qualified";
  mode3_modern.gpu_streams = 1;
  mode3_modern.gpu_graphs = "off";
  mode3_modern.trace = "summary";

  PARALLEL_BACKEND mode3;
  mode3.configure("serial", 0, "auto", "auto", 0, 262144,
                  "legacy", "strict", false, mode3_request,
                  mode3_modern);
  if (!mode3.gpu_mode3_enabled() || mode3.gpu_mode2_enabled() ||
      mode3.actual_engine() != "modern_gpu" ||
      mode3.actual_gpu_mode() != 3 || mode3.gpu3_calls() != 0)
    return 20;
  mode3.record_gpu_mode3_execution("mode3 test device");
  const std::string mode3_provenance = mode3.provenance();
  if (mode3.gpu3_calls() != 1 ||
      mode3_provenance.find("gpu_mode requested=3 actual=3") ==
          std::string::npos ||
      mode3_provenance.find("gpu3_calls=1") == std::string::npos ||
      mode3_provenance.find(
          "gpu2_primary_update=resident_device_gather_push_whole_crossing") ==
          std::string::npos ||
      mode3_provenance.find("initialization_rng=legacy_sequential") ==
          std::string::npos ||
      mode3_provenance.find("mode3_collision_rng=unused") ==
          std::string::npos ||
      mode3_provenance.find(
          "precision requested=mixed_qualified actual=mixed_qualified") ==
          std::string::npos ||
      mode3_provenance.find(
          "modern_reduction requested=fast_qualified actual=fast_qualified") ==
          std::string::npos)
    return 21;
  bool duplicate_mode3_record_rejected = false;
  try
    {
      mode3.record_gpu_mode3_execution("mode3 test device");
    }
  catch (const std::logic_error&)
    {
      duplicate_mode3_record_rejected = true;
    }
  if (!duplicate_mode3_record_rejected) return 22;

  bool false_fp64_rejected = false;
  try
    {
      MODERN_RUNTIME_OPTIONS bad_modern = mode3_modern;
      bad_modern.precision = "fp64";
      PARALLEL_BACKEND bad;
      bad.configure("serial", 0, "auto", "auto", 0, 262144,
                    "legacy", "strict", false, mode3_request, bad_modern);
    }
  catch (const std::invalid_argument&)
    {
      false_fp64_rejected = true;
    }
  if (!false_fp64_rejected) return 23;

  bool counter_claim_rejected = false;
  try
    {
      PARALLEL_BACKEND bad;
      bad.configure("serial", 0, "auto", "auto", 0, 262144,
                    "counter", "strict", false, mode3_request,
                    mode3_modern);
    }
  catch (const std::invalid_argument&)
    {
      counter_claim_rejected = true;
    }
  if (!counter_claim_rejected) return 24;

  bool fallback_rejected = false;
  try
    {
      PARALLEL_BACKEND bad;
      bad.configure("serial", 0, "auto", "auto", 0, 262144,
                    "legacy", "strict", true, mode3_request,
                    mode3_modern);
    }
  catch (const std::invalid_argument&)
    {
      fallback_rejected = true;
    }
  if (!fallback_rejected) return 25;

  bool inert_trace_off_rejected = false;
  try
    {
      MODERN_RUNTIME_OPTIONS bad_modern = mode3_modern;
      bad_modern.trace = "off";
      PARALLEL_BACKEND bad;
      bad.configure("serial", 0, "auto", "auto", 0, 262144,
                    "legacy", "strict", false, mode3_request, bad_modern);
    }
  catch (const std::invalid_argument& error)
    {
      inert_trace_off_rejected =
          std::string(error.what()).find("modern_trace=summary") !=
          std::string::npos;
    }
  if (!inert_trace_off_rejected) return 26;

  bool inert_profile_off_rejected = false;
  try
    {
      GPU2_OPTIONS bad_gpu = mode3_request;
      bad_gpu.profile = "off";
      PARALLEL_BACKEND bad;
      bad.configure("serial", 0, "auto", "auto", 0, 262144,
                    "legacy", "strict", false, bad_gpu, mode3_modern);
    }
  catch (const std::invalid_argument& error)
    {
      inert_profile_off_rejected =
          std::string(error.what()).find("gpu2_profile=summary") !=
          std::string::npos;
    }
  if (!inert_profile_off_rejected) return 27;

  GPU2_OPTIONS mode4_request;
  mode4_request.mode = 4;
  mode4_request.scope = "primary";
  mode4_request.deposition = "atomic";
  mode4_request.streams = 1;
  mode4_request.graphs = "off";
  mode4_request.fusion = "on";
  mode4_request.batch_slices = 32;
  mode4_request.pinned_io = "on";
  mode4_request.host_threads = 1;
  mode4_request.primary_mover = "serial";
  mode4_request.profile = "summary";
  PARALLEL_BACKEND mode4;
  mode4.configure("gpu", 0, "auto", "auto", 0, 262144,
                  "legacy", "fast", false, mode4_request);
  if (!mode4.gpu_mode4_enabled() || mode4.gpu_mode2_enabled() ||
      mode4.gpu_mode3_enabled() || mode4.gpu2_plan_committed() ||
      mode4.gpu4_calls() != 0)
    return 28;
  bool mode4_premature_rejected = false;
  try
    {
      mode4.record_gpu_mode4_execution("test wave device");
    }
  catch (const std::logic_error&)
    {
      mode4_premature_rejected = true;
    }
  if (!mode4_premature_rejected) return 29;
  mode4.record_gpu_mode4_plan(mode4_request);
  mode4.record_gpu_mode4_execution("test wave device");
  mode4.record_gpu_mode4_execution("test wave device");
  const std::string mode4_provenance = mode4.provenance();
  if (!mode4.gpu2_plan_committed() || mode4.gpu4_calls() != 2 ||
      mode4_provenance.find("gpu_mode requested=4 actual=4") ==
          std::string::npos ||
      mode4_provenance.find("gpu4_calls=2") == std::string::npos ||
      mode4_provenance.find(
          "gpu2_host_boundary=one_completion_boundary_per_wave_bucket") ==
          std::string::npos ||
      mode4_provenance.find(
          "gpu2_primary_update=host_serial_ordered_after_wave_batch") ==
          std::string::npos ||
      mode4_provenance.find("gpu2_rng=legacy_host_ordered") ==
          std::string::npos)
    return 30;

  bool mode4_deterministic_rejected = false;
  try
    {
      PARALLEL_BACKEND invalid_mode4;
      invalid_mode4.configure("gpu", 0, "auto", "auto", 0, 262144,
                              "legacy", "deterministic", false,
                              mode4_request);
    }
  catch (const std::invalid_argument& error)
    {
      mode4_deterministic_rejected =
          std::string(error.what()).find("reduction_mode=fast") !=
          std::string::npos;
    }
  if (!mode4_deterministic_rejected) return 31;

  // Mode 4 can opt into the backend-owned custom pool without enabling
  // OpenMP.  The plan is fail-closed, creates the team only when committed,
  // preserves stable lane/range assignment, and reuses the same worker
  // identities across dispatch generations.
  GPU2_OPTIONS mode4_pool_request = mode4_request;
  mode4_pool_request.host_threads = 4;
  mode4_pool_request.primary_mover = "PoOl";
  PARALLEL_BACKEND mode4_pool;
  mode4_pool.configure("gpu", 0, "auto", "auto", 0, 262144,
                       "legacy", "fast", false, mode4_pool_request);
  bool mode4_uncommitted_pool_rejected = false;
  try
    {
      mode4_pool.run_gpu2_primary_static_ranges(1, NULL, &record_range);
    }
  catch (const std::logic_error&)
    {
      mode4_uncommitted_pool_rejected = true;
    }
  if (!mode4_uncommitted_pool_rejected) return 32;

  GPU2_OPTIONS mode4_pool_plan = mode4_pool_request;
  mode4_pool_plan.primary_mover = "pool";
  mode4_pool.record_gpu_mode4_plan(mode4_pool_plan);
  std::vector<int> mode4_owners(10, -1);
  std::vector<int> mode4_calls(4, 0);
  std::vector<std::thread::id> mode4_first_threads(4);
  RANGE_PROBE mode4_probe = {
      &mode4_owners, &mode4_calls, &mode4_first_threads};
  mode4_pool.run_gpu2_primary_static_ranges(
      mode4_owners.size(), &mode4_probe, &record_range);
  for (std::size_t index = 0; index < mode4_owners.size(); ++index)
    if (mode4_owners[index] != expected_lanes[index]) return 33;
  mode4_pool.record_gpu2_primary_mover_execution(
      "POOL", mode4_owners.size(), 4, 4);

  std::fill(mode4_owners.begin(), mode4_owners.end(), -1);
  std::vector<std::thread::id> mode4_second_threads(4);
  mode4_probe.threads = &mode4_second_threads;
  mode4_pool.run_gpu2_primary_static_ranges(
      mode4_owners.size(), &mode4_probe, &record_range);
  mode4_pool.record_gpu2_primary_mover_execution(
      "pool", mode4_owners.size(), 4, 4);
  for (std::size_t lane = 0; lane < mode4_calls.size(); ++lane)
    if (mode4_calls[lane] != 2 ||
        mode4_first_threads[lane] != mode4_second_threads[lane])
      return 34;

  const std::string mode4_pool_provenance = mode4_pool.provenance();
  if (mode4_pool.actual_cpu_variant() != "gpu4_pool_static_ranges" ||
      mode4_pool.actual_cpu_threads() != 4 ||
      mode4_pool.gpu2_primary_mover_calls() != 2 ||
      mode4_pool.gpu2_primary_mover_particles() != 20 ||
      mode4_pool.gpu2_primary_mover_ranges() != 8 ||
      mode4_pool.gpu2_primary_mover_observed_threads() != 4 ||
      mode4_pool_provenance.find(
          "gpu2_primary_mover requested=pool actual=pool") ==
          std::string::npos ||
      mode4_pool_provenance.find(
          "gpu2_primary_update=host_pool_static_ranges_after_wave_batch") ==
          std::string::npos ||
      mode4_pool_provenance.find(
          "gpu2_rng=caller_owned_lane_contract") == std::string::npos)
    return 35;

  bool mode4_pool_one_thread_rejected = false;
  try
    {
      GPU2_OPTIONS bad = mode4_request;
      bad.primary_mover = "pool";
      bad.host_threads = 1;
      PARALLEL_BACKEND invalid_mode4_pool;
      invalid_mode4_pool.configure(
          "gpu", 0, "auto", "auto", 0, 262144,
          "legacy", "fast", false, bad);
    }
  catch (const std::invalid_argument& error)
    {
      mode4_pool_one_thread_rejected =
          std::string(error.what()).find("gpu2_host_threads>=2") !=
          std::string::npos;
    }
  if (!mode4_pool_one_thread_rejected) return 36;

  bool mode4_serial_many_threads_rejected = false;
  try
    {
      GPU2_OPTIONS bad = mode4_request;
      bad.host_threads = 2;
      PARALLEL_BACKEND invalid_mode4_serial;
      invalid_mode4_serial.configure(
          "gpu", 0, "auto", "auto", 0, 262144,
          "legacy", "fast", false, bad);
    }
  catch (const std::invalid_argument& error)
    {
      mode4_serial_many_threads_rejected =
          std::string(error.what()).find("gpu2_host_threads=1") !=
          std::string::npos;
    }
  if (!mode4_serial_many_threads_rejected) return 37;

  bool mode4_host_plan_mismatch_rejected = false;
  try
    {
      GPU2_OPTIONS request = mode4_request;
      request.primary_mover = "pool";
      request.host_threads = 4;
      PARALLEL_BACKEND mismatch;
      mismatch.configure("gpu", 0, "auto", "auto", 0, 262144,
                         "legacy", "fast", false, request);
      GPU2_OPTIONS mismatched_plan = request;
      mismatched_plan.host_threads = 3;
      mismatch.record_gpu_mode4_plan(mismatched_plan);
    }
  catch (const std::invalid_argument& error)
    {
      mode4_host_plan_mismatch_rejected =
          std::string(error.what()).find("does not match the requested") !=
          std::string::npos;
    }
  if (!mode4_host_plan_mismatch_rejected) return 38;

  bool mode1_record_rejected = false;
  try
    {
      backend.record_gpu_execution("wrong path");
    }
  catch (const std::logic_error&)
    {
      mode1_record_rejected = true;
    }
  return mode1_record_rejected ? 0 : 9;
}
