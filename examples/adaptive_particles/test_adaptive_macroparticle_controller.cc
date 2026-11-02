#include "adaptiveMacroparticleControllerCPP.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

using feebl::adaptive_particles::Configuration;
using feebl::adaptive_particles::ControllerAction;
using feebl::adaptive_particles::ControllerObservation;
using feebl::adaptive_particles::CoupledController;
using feebl::adaptive_particles::Mode;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Configuration configuration(Mode mode) {
  Configuration value;
  value.mode = mode;
  value.error_target = mode == Mode::Diagnostics ? 0.01 : 0.01;
  value.resample_bias_target = 0.001;
  value.minimum_root_support = 8.0;
  value.minimum_clone_support = 8.0;
  return value;
}

ControllerObservation low(std::uint64_t epoch) {
  ControllerObservation value;
  value.epoch = epoch;
  value.error.source_sampling_known = true;
  value.error.mesh = 0.001;
  value.error.source_sampling = 0.001;
  value.error.resampling_bias = 0.0001;
  value.error.resampling_variance = 0.0001;
  value.error.floating_point = 0.00001;
  value.support.distinct_roots = 100.0;
  value.support.clone_support_proxy = 100.0;
  value.support.protected_tail_support = 20.0;
  value.support.independent_complete_runs = 8U;
  value.canonical_publication_complete = true;
  return value;
}

void test_decision_table() {
  {
    CoupledController controller(configuration(Mode::SplitOnly));
    ControllerObservation value = low(1U);
    value.error.mesh = 0.02;
    const auto decision = controller.observe(value);
    require(decision.action == ControllerAction::RefineMeshAndShapeSplit,
            "high mesh/low sampling did not request mesh refinement");
  }
  {
    CoupledController controller(configuration(Mode::SplitOnly));
    ControllerObservation value = low(1U);
    value.error.source_sampling = 0.02;
    value.initial_state_replay_available = true;
    const auto decision = controller.observe(value);
    require(decision.action == ControllerAction::RequestInitialStateReplay &&
                decision.replay_required,
            "low mesh/high sampling did not request replay");
  }
  {
    CoupledController controller(configuration(Mode::SplitOnly));
    ControllerObservation value = low(1U);
    value.error.mesh = 0.02;
    value.error.source_sampling = 0.02;
    value.initial_state_replay_available = true;
    const auto decision = controller.observe(value);
    require(decision.action ==
                ControllerAction::JointReplayAndMeshRefinement,
            "high mesh/high sampling did not request a joint action");
  }
}

void test_unknown_and_clone_negative_control() {
  CoupledController controller(configuration(Mode::SplitOnly));
  ControllerObservation value = low(1U);
  value.error.source_sampling_known = false;
  value.support.distinct_roots = 4.0;
  value.support.clone_support_proxy = 10000.0;
  value.support.protected_tail_support = 4.0;
  const auto decision = controller.observe(value);
  require(decision.action == ControllerAction::UnmetFailClosed &&
              decision.replay_required &&
              !decision.total_error_bound_met,
          "clones or unknown sampling incorrectly cleared the source gate");
  require(std::isinf(decision.conservative_total_error),
          "unknown source error did not make total uncertainty unavailable");

  CoupledController one_run_controller(configuration(Mode::SplitOnly));
  ControllerObservation one_run = low(2U);
  one_run.support.independent_complete_runs = 1U;
  const auto one_run_decision = one_run_controller.observe(one_run);
  require(one_run_decision.action == ControllerAction::UnmetFailClosed &&
              one_run_decision.replay_required &&
              !one_run_decision.total_error_bound_met,
          "one complete run incorrectly cleared the source-sampling gate");

  CoupledController ancestry_controller(configuration(Mode::SplitOnly));
  ControllerObservation inexact_ancestry = low(3U);
  inexact_ancestry.support.ancestry_exact = false;
  const auto ancestry_decision = ancestry_controller.observe(inexact_ancestry);
  require(ancestry_decision.action == ControllerAction::UnmetFailClosed &&
              ancestry_decision.replay_required &&
              !ancestry_decision.total_error_bound_met,
          "inexact ancestry incorrectly cleared configured support gates");
}

void test_mode3_dwell_and_mode1_dry_run() {
  CoupledController controller(configuration(Mode::CoupledSplitMerge));
  ControllerObservation first = low(1U);
  first.conservative_merge_feasible = true;
  const auto first_decision = controller.observe(first);
  require(first_decision.action == ControllerAction::Hold,
          "Mode 3 coarsened before dwell");
  ControllerObservation second = low(2U);
  second.conservative_merge_feasible = true;
  const auto second_decision = controller.observe(second);
  require(second_decision.action == ControllerAction::ConservativeMerge,
          "Mode 3 did not coarsen after dwell");

  CoupledController diagnostic(configuration(Mode::Diagnostics));
  ControllerObservation high = low(1U);
  high.error.mesh = 0.02;
  const auto dry = diagnostic.observe(high);
  require(dry.action == ControllerAction::DiagnosticsOnly &&
              dry.recommended_action ==
                  ControllerAction::RefineMeshAndShapeSplit &&
              !dry.commit_allowed,
          "Mode 1 did not preserve dry-run semantics");
}

void test_validation_and_repeatability() {
  CoupledController premature(configuration(Mode::SplitOnly));
  ControllerObservation before_publication = low(1U);
  before_publication.canonical_publication_complete = false;
  bool rejected = false;
  try {
    (void)premature.observe(before_publication);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected,
          "controller accepted an observation before canonical publication");

  CoupledController first(configuration(Mode::SplitOnly));
  CoupledController second(configuration(Mode::SplitOnly));
  const auto a = first.observe(low(9U));
  const auto b = second.observe(low(9U));
  require(a.decision_hash != 0U && a.decision_hash == b.decision_hash,
          "controller decision hash is not deterministic");
  rejected = false;
  try {
    (void)first.observe(low(9U));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "controller accepted a repeated epoch");

  CoupledController invalid_values(configuration(Mode::SplitOnly));
  ControllerObservation malformed = low(10U);
  malformed.error.pair_sampling = -1.0;
  rejected = false;
  try {
    (void)invalid_values.observe(malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "controller accepted a negative error component");
}

void test_process_replay_recommendation_aggregation() {
  using feebl::adaptive_particles::combine_replay_recommendation;
  const ControllerAction replay =
      ControllerAction::RequestInitialStateReplay;
  const ControllerAction joint =
      ControllerAction::JointReplayAndMeshRefinement;
  require(combine_replay_recommendation(ControllerAction::Hold, replay) ==
              replay,
          "empty process accumulator did not accept replay");
  require(combine_replay_recommendation(replay, replay) == replay,
          "repeated replay recommendation changed action");
  require(combine_replay_recommendation(replay, joint) == joint,
          "later joint recommendation was masked by early replay");
  require(combine_replay_recommendation(joint, replay) == joint,
          "later replay recommendation downgraded an existing joint action");

  bool rejected = false;
  try {
    (void)combine_replay_recommendation(
        replay, ControllerAction::RefineMeshAndShapeSplit);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "replay aggregation accepted a non-replay candidate");
}

}  // namespace

int main() {
  try {
    test_decision_table();
    test_unknown_and_clone_negative_control();
    test_mode3_dwell_and_mode1_dry_run();
    test_validation_and_repeatability();
    test_process_replay_recommendation_aggregation();
  } catch (const std::exception& error) {
    std::cerr << "adaptive macroparticle controller test failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "adaptive macroparticle controller tests passed\n";
  return EXIT_SUCCESS;
}
