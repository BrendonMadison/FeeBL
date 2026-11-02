#ifndef FEEBL_ADAPTIVE_MACROPARTICLE_CONTROLLER_CPP_H
#define FEEBL_ADAPTIVE_MACROPARTICLE_CONTROLLER_CPP_H

#include "adaptiveMacroparticleCPP.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace feebl {
namespace adaptive_particles {

// Error terms stay separate so callers cannot accidentally clear a missing
// source-sampling estimate with a small mesh residual.  The v1 controller uses
// a conservative sum; covariance-aware RSS is intentionally not available
// until independent-run covariance evidence is supplied.
struct ErrorComponents {
  double mesh = 0.0;                 // eta_h
  double source_sampling = 0.0;      // eta_N
  double resampling_bias = 0.0;      // eta_R,b
  double resampling_variance = 0.0;  // eta_R,v
  double pair_sampling = 0.0;        // eta_pair
  double stochastic_physics = 0.0;   // eta_phys
  double solver_integration = 0.0;   // eta_S
  double floating_point = 0.0;       // eta_fp
  double reference = 0.0;
  bool source_sampling_known = false;

  double conservative_total() const;
  void validate() const;
};

struct SupportDiagnostics {
  double distinct_roots = 0.0;
  double clone_support_proxy = 0.0;
  double protected_tail_support = 0.0;
  std::size_t independent_complete_runs = 0U;
  bool ancestry_exact = true;
};

struct ControllerObservation {
  std::uint64_t epoch = 0U;
  ErrorComponents error;
  SupportDiagnostics support;
  // The collision integrator must freeze diagnostics before the kick but may
  // expose them to this stateful policy only after canonical wave publication.
  bool canonical_publication_complete = false;
  bool mesh_refinement_available = true;
  bool initial_state_replay_available = false;
  bool conservative_merge_feasible = false;
};

enum class ControllerAction : int {
  Hold = 0,
  RefineMeshAndShapeSplit = 1,
  RequestInitialStateReplay = 2,
  JointReplayAndMeshRefinement = 3,
  ConservativeMerge = 4,
  DiagnosticsOnly = 5,
  UnmetFailClosed = 6
};

struct ControllerDecision {
  ControllerAction action = ControllerAction::Hold;
  ControllerAction recommended_action = ControllerAction::Hold;
  bool diagnostics_due = false;
  bool commit_allowed = false;
  bool mesh_error_high = false;
  bool source_sampling_high_or_unknown = false;
  bool total_error_bound_met = false;
  bool replay_required = false;
  double mesh_ratio = 0.0;
  double source_sampling_ratio = 0.0;
  double conservative_total_error = 0.0;
  std::size_t consecutive_low_epochs = 0U;
  std::uint64_t decision_hash = 0U;
  std::string reason;
};

// Deterministic hysteretic policy mechanism.  It makes no population mutation;
// the caller may commit only at its separately audited wave/crossing barrier.
class CoupledController {
 public:
  explicit CoupledController(Configuration configuration);

  ControllerDecision observe(const ControllerObservation& observation);
  void reset() noexcept;

 private:
  Configuration configuration_;
  std::size_t consecutive_low_epochs_ = 0U;
  std::uint64_t observations_ = 0U;
  bool have_last_epoch_ = false;
  std::uint64_t last_epoch_ = 0U;
};

const char* controller_action_name(ControllerAction action) noexcept;

// Accumulate a process-level fresh-root recommendation without allowing an
// early replay-only encounter to mask a later joint grid+root requirement.
// Hold is the empty accumulator; only replay-capable candidates are accepted.
ControllerAction combine_replay_recommendation(
    ControllerAction accumulated, ControllerAction candidate);

}  // namespace adaptive_particles
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_MACROPARTICLE_CONTROLLER_CPP_H
