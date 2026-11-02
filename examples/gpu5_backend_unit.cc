#include "parallelBackendCPP.h"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

GPU2_OPTIONS valid_options() {
  GPU2_OPTIONS options;
  options.mode = 5;
  options.scope = "primary";
  options.deposition = "atomic";
  options.fft = "cufft";
  options.streams = 1;
  options.graphs = "off";
  options.fusion = "on";
  options.batch_slices = 8;
  options.pinned_io = "on";
  options.host_threads = 2;
  options.primary_mover = "pool";
  options.memory_fraction = 0.5;
  options.profile = "summary";
  return options;
}

void configure(PARALLEL_BACKEND& backend, const GPU2_OPTIONS& options,
               const std::string& compute = "gpu",
               const std::string& reduction = "fast",
               bool fallback = false) {
  backend.configure(compute, 0, "auto", "auto", 0, 262144,
                    "legacy", reduction, fallback, options,
                    MODERN_RUNTIME_OPTIONS());
}

template <class Function>
void require_throws(Function function, const std::string& needle) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(needle) == std::string::npos)
      throw std::runtime_error("unexpected diagnostic: " +
                               std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection containing: " + needle);
}

struct RangeContext {
  std::vector<std::atomic<unsigned int> >* visits;
  std::vector<std::atomic<unsigned int> >* lane_calls;
};

void visit_range(void* opaque, std::size_t lane, std::size_t begin,
                 std::size_t end) {
  RangeContext& context = *static_cast<RangeContext*>(opaque);
  ++(*context.lane_calls)[lane];
  for (std::size_t item = begin; item < end; ++item)
    ++(*context.visits)[item];
}

}  // namespace

int main() {
  try {
    const GPU2_OPTIONS options = valid_options();
    PARALLEL_BACKEND backend;
    configure(backend, options);
    if (!backend.gpu_mode5_enabled() || backend.gpu_mode4_enabled() ||
        backend.actual_gpu_mode() != 5 || backend.allow_fallback() ||
        backend.gpu2_plan_committed())
      throw std::runtime_error("mode-5 selector/precommit state is wrong");
    require_throws(
        [&]() { backend.record_gpu_mode5_execution("unit-device"); },
        "committed adaptive batch plan");
    backend.record_gpu_mode5_plan(options);
    if (!backend.gpu2_plan_committed() ||
        backend.actual_cpu_variant() != "gpu5_pool_static_ranges")
      throw std::runtime_error("mode-5 pool plan was not committed");

    std::vector<std::atomic<unsigned int> > visits(17);
    std::vector<std::atomic<unsigned int> > lane_calls(2);
    for (std::atomic<unsigned int>& visit : visits) visit.store(0);
    for (std::atomic<unsigned int>& call : lane_calls) call.store(0);
    RangeContext context{&visits, &lane_calls};
    backend.run_static_ranges(visits.size(), &context, &visit_range);
    for (const std::atomic<unsigned int>& visit : visits)
      if (visit.load() != 1U)
        throw std::runtime_error("mode-5 pool did not visit exactly once");
    if (lane_calls[0].load() != 1U || lane_calls[1].load() != 1U)
      throw std::runtime_error("mode-5 full pool lane accounting is wrong");

    for (std::atomic<unsigned int>& visit : visits) visit.store(0);
    for (std::atomic<unsigned int>& call : lane_calls) call.store(0);
    backend.run_static_ranges_limited(
        visits.size(), 1, &context, &visit_range);
    for (const std::atomic<unsigned int>& visit : visits)
      if (visit.load() != 1U)
        throw std::runtime_error(
            "mode-5 bounded pool did not visit exactly once");
    if (lane_calls[0].load() != 1U || lane_calls[1].load() != 0U)
      throw std::runtime_error("mode-5 bounded pool used an inactive lane");
    backend.record_gpu_mode5_execution("unit-device");
    const std::string provenance = backend.provenance();
    if (backend.gpu5_calls() != 1U ||
        provenance.find("gpu_mode requested=5 actual=5") ==
            std::string::npos ||
        provenance.find("gpu5_calls=1") == std::string::npos ||
        provenance.find("fallback=") != std::string::npos ||
        provenance.find("adaptive_class_planmany_fields") ==
            std::string::npos)
      throw std::runtime_error("mode-5 execution provenance is incomplete");

    require_throws([&]() {
      PARALLEL_BACKEND bad;
      configure(bad, options, "cpu");
    }, "requires compute_backend=gpu");
    require_throws([&]() {
      PARALLEL_BACKEND bad;
      configure(bad, options, "gpu", "fast", true);
    }, "allow_backend_fallback must be 0");
    require_throws([&]() {
      PARALLEL_BACKEND bad;
      configure(bad, options, "gpu", "strict", false);
    }, "requires reduction_mode=fast");
    require_throws([&]() {
      GPU2_OPTIONS bad_options = options;
      bad_options.deposition = "shared_tile";
      PARALLEL_BACKEND bad;
      configure(bad, bad_options);
    }, "shared_tile and sort_reduce");
    require_throws([&]() {
      GPU2_OPTIONS bad_options = options;
      bad_options.deposition = "sort_reduce";
      PARALLEL_BACKEND bad;
      configure(bad, bad_options);
    }, "shared_tile and sort_reduce");
    require_throws([&]() {
      GPU2_OPTIONS bad_options = options;
      bad_options.host_threads = 1;
      PARALLEL_BACKEND bad;
      configure(bad, bad_options);
    }, "gpu2_host_threads>=2");

    std::cout << "gpu mode 5 backend tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
