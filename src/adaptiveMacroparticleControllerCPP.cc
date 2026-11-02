#include "adaptiveMacroparticleControllerCPP.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr std::uint64_t kFnvOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

void hash_u64(std::uint64_t& state, std::uint64_t value) {
  for (unsigned int byte = 0U; byte < 8U; ++byte) {
    state ^= (value >> (8U * byte)) & UINT64_C(0xff);
    state *= kFnvPrime;
  }
}

void hash_double(std::uint64_t& state, double value) {
  std::uint64_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(state, bits);
}

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

namespace feebl {
namespace adaptive_particles {

void ErrorComponents::validate() const {
  const double values[] = {
      mesh, source_sampling, resampling_bias, resampling_variance,
      pair_sampling, stochastic_physics, solver_integration,
      floating_point, reference};
  for (double value : values)
    if (!finite_nonnegative(value))
      throw std::invalid_argument(
          "adaptive particle error components must be finite and nonnegative");
}

double ErrorComponents::conservative_total() const {
  validate();
  if (!source_sampling_known)
    return std::numeric_limits<double>::infinity();
  return mesh + source_sampling + resampling_bias + resampling_variance +
         pair_sampling + stochastic_physics + solver_integration +
         floating_point + reference;
}

const char* controller_action_name(ControllerAction action) noexcept {
  switch (action) {
    case ControllerAction::Hold: return "hold";
    case ControllerAction::RefineMeshAndShapeSplit:
      return "refine_mesh_and_shape_split";
    case ControllerAction::RequestInitialStateReplay:
      return "request_initial_state_replay";
    case ControllerAction::JointReplayAndMeshRefinement:
      return "joint_replay_and_mesh_refinement";
    case ControllerAction::ConservativeMerge: return "conservative_merge";
    case ControllerAction::DiagnosticsOnly: return "diagnostics_only";
    case ControllerAction::UnmetFailClosed: return "unmet_fail_closed";
  }
  return "invalid";
}

ControllerAction combine_replay_recommendation(
    ControllerAction accumulated, ControllerAction candidate) {
  const bool accumulated_valid =
      accumulated == ControllerAction::Hold ||
      accumulated == ControllerAction::RequestInitialStateReplay ||
      accumulated == ControllerAction::JointReplayAndMeshRefinement;
  const bool candidate_valid =
      candidate == ControllerAction::RequestInitialStateReplay ||
      candidate == ControllerAction::JointReplayAndMeshRefinement;
  if (!accumulated_valid || !candidate_valid)
    throw std::invalid_argument(
        "root replay aggregation accepts only Hold, replay, or joint actions");
  if (accumulated == ControllerAction::JointReplayAndMeshRefinement ||
      candidate == ControllerAction::JointReplayAndMeshRefinement)
    return ControllerAction::JointReplayAndMeshRefinement;
  return ControllerAction::RequestInitialStateReplay;
}

CoupledController::CoupledController(Configuration configuration)
    : configuration_(std::move(configuration)) {
  validate_configuration(configuration_);
}

void CoupledController::reset() noexcept {
  consecutive_low_epochs_ = 0U;
  observations_ = 0U;
  have_last_epoch_ = false;
  last_epoch_ = 0U;
}

ControllerDecision CoupledController::observe(
    const ControllerObservation& observation) {
  observation.error.validate();
  if (!observation.canonical_publication_complete)
    throw std::invalid_argument(
        "adaptive particle controller observation precedes canonical wave "
        "publication");
  if (!finite_nonnegative(observation.support.distinct_roots) ||
      !finite_nonnegative(observation.support.clone_support_proxy) ||
      !finite_nonnegative(observation.support.protected_tail_support))
    throw std::invalid_argument(
        "adaptive particle support diagnostics must be finite and nonnegative");
  if (have_last_epoch_ && observation.epoch <= last_epoch_)
    throw std::invalid_argument(
        "adaptive particle controller epochs must be strictly increasing");
  have_last_epoch_ = true;
  last_epoch_ = observation.epoch;
  ++observations_;

  ControllerDecision decision;
  decision.diagnostics_due =
      ((observations_ - 1U) % configuration_.update_interval) == 0U;
  decision.commit_allowed =
      configuration_.mode == Mode::SplitOnly ||
      configuration_.mode == Mode::CoupledSplitMerge;
  decision.conservative_total_error =
      observation.error.conservative_total();
  const double target = configuration_.error_target;
  decision.mesh_ratio = target > 0.0
      ? observation.error.mesh / target
      : (observation.error.mesh == 0.0
             ? 0.0 : std::numeric_limits<double>::infinity());
  decision.source_sampling_ratio =
      observation.error.source_sampling_known && target > 0.0
          ? observation.error.source_sampling / target
          : std::numeric_limits<double>::infinity();

  const bool root_support_low =
      configuration_.minimum_root_support > 0.0 &&
      (observation.support.distinct_roots <
           configuration_.minimum_root_support ||
       observation.support.protected_tail_support <
           configuration_.minimum_root_support);
  const bool clone_support_low =
      configuration_.minimum_clone_support > 0.0 &&
      observation.support.clone_support_proxy <
          configuration_.minimum_clone_support;
  const bool independent_replication_unavailable =
      observation.support.independent_complete_runs < 2U;
  const bool root_diagnostics_unavailable =
      !observation.support.ancestry_exact &&
      (configuration_.minimum_root_support > 0.0 ||
       configuration_.minimum_clone_support > 0.0);
  decision.mesh_error_high =
      decision.mesh_ratio >= configuration_.refine_threshold;
  decision.source_sampling_high_or_unknown =
      !observation.error.source_sampling_known ||
      independent_replication_unavailable || root_diagnostics_unavailable ||
      root_support_low || clone_support_low ||
      decision.source_sampling_ratio >= configuration_.refine_threshold;
  decision.total_error_bound_met =
      std::isfinite(decision.conservative_total_error) && target > 0.0 &&
      decision.conservative_total_error <= target && !root_support_low &&
      !clone_support_low && !independent_replication_unavailable &&
      !root_diagnostics_unavailable;

  if (!decision.diagnostics_due) {
    decision.reason = "diagnostics_not_due";
    decision.decision_hash = kFnvOffset;
    hash_u64(decision.decision_hash, observation.epoch);
    hash_u64(decision.decision_hash,
             static_cast<std::uint64_t>(decision.action));
    hash_u64(decision.decision_hash,
             observation.canonical_publication_complete
                 ? UINT64_C(1) : UINT64_C(0));
    return decision;
  }

  if (decision.mesh_error_high &&
      decision.source_sampling_high_or_unknown) {
    decision.recommended_action =
        ControllerAction::JointReplayAndMeshRefinement;
  } else if (decision.source_sampling_high_or_unknown) {
    decision.recommended_action =
        ControllerAction::RequestInitialStateReplay;
  } else if (decision.mesh_error_high) {
    decision.recommended_action =
        ControllerAction::RefineMeshAndShapeSplit;
  } else {
    const bool safely_low =
        decision.mesh_ratio <= configuration_.coarsen_threshold &&
        decision.source_sampling_ratio <= configuration_.coarsen_threshold &&
        observation.error.resampling_bias <=
            configuration_.resample_bias_target &&
        observation.error.resampling_variance <=
            configuration_.resample_bias_target;
    consecutive_low_epochs_ = safely_low
        ? consecutive_low_epochs_ + 1U : 0U;
    if (configuration_.mode == Mode::CoupledSplitMerge &&
        observation.conservative_merge_feasible &&
        consecutive_low_epochs_ >= configuration_.dwell)
      decision.recommended_action = ControllerAction::ConservativeMerge;
    else
      decision.recommended_action = ControllerAction::Hold;
  }
  if (decision.mesh_error_high ||
      decision.source_sampling_high_or_unknown)
    consecutive_low_epochs_ = 0U;
  decision.consecutive_low_epochs = consecutive_low_epochs_;

  if (configuration_.mode == Mode::Diagnostics) {
    decision.action = ControllerAction::DiagnosticsOnly;
    decision.reason = std::string("dry_run_recommends:") +
        controller_action_name(decision.recommended_action);
  } else if ((decision.recommended_action ==
                  ControllerAction::RequestInitialStateReplay ||
              decision.recommended_action ==
                  ControllerAction::JointReplayAndMeshRefinement) &&
             !observation.initial_state_replay_available) {
    decision.action = ControllerAction::UnmetFailClosed;
    decision.replay_required = true;
    decision.reason = observation.error.source_sampling_known
        ? "source_sampling_or_support_target_requires_initial_state_replay"
        : "source_sampling_uncertainty_unknown_requires_independent_runs_or_replay";
  } else if (decision.recommended_action ==
                 ControllerAction::RefineMeshAndShapeSplit &&
             !observation.mesh_refinement_available) {
    decision.action = ControllerAction::UnmetFailClosed;
    decision.reason = "mesh_error_high_but_refinement_unavailable";
  } else {
    decision.action = decision.recommended_action;
    decision.replay_required =
        decision.action == ControllerAction::RequestInitialStateReplay ||
        decision.action == ControllerAction::JointReplayAndMeshRefinement;
    decision.reason = controller_action_name(decision.action);
  }

  decision.decision_hash = kFnvOffset;
  hash_u64(decision.decision_hash, UINT64_C(0x414d50434f4e5431));
  hash_u64(decision.decision_hash, observation.epoch);
  hash_u64(decision.decision_hash,
           static_cast<std::uint64_t>(decision.action));
  hash_u64(decision.decision_hash,
           static_cast<std::uint64_t>(decision.recommended_action));
  hash_double(decision.decision_hash, decision.mesh_ratio);
  hash_double(decision.decision_hash, decision.source_sampling_ratio);
  hash_double(decision.decision_hash, decision.conservative_total_error);
  hash_double(decision.decision_hash, observation.support.distinct_roots);
  hash_double(decision.decision_hash,
              observation.support.clone_support_proxy);
  hash_double(decision.decision_hash,
              observation.support.protected_tail_support);
  hash_u64(decision.decision_hash,
           static_cast<std::uint64_t>(
               observation.support.independent_complete_runs));
  hash_u64(decision.decision_hash,
           observation.support.ancestry_exact ? UINT64_C(1) : UINT64_C(0));
  hash_u64(decision.decision_hash,
           observation.canonical_publication_complete
               ? UINT64_C(1) : UINT64_C(0));
  hash_u64(decision.decision_hash,
           observation.mesh_refinement_available ? UINT64_C(1) : UINT64_C(0));
  hash_u64(decision.decision_hash,
           observation.initial_state_replay_available
               ? UINT64_C(1) : UINT64_C(0));
  hash_u64(decision.decision_hash,
           observation.conservative_merge_feasible
               ? UINT64_C(1) : UINT64_C(0));
  hash_u64(decision.decision_hash, decision.consecutive_low_epochs);
  return decision;
}

}  // namespace adaptive_particles
}  // namespace feebl
