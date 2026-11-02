#ifndef FEEBL_ADAPTIVE_ROOT_REPLAY_SIGNAL_CPP_H
#define FEEBL_ADAPTIVE_ROOT_REPLAY_SIGNAL_CPP_H

#include <stdexcept>
#include <string>

namespace feebl {
namespace adaptive_particles {

// Process-level handoff contract for a controller that needs genuinely new
// initial roots.  Exit 75 is deliberately distinct from FeeBL's ordinary
// fatal-error exits: the collision has completed, but the pre-output
// qualification gate refuses final summary-output publication.
constexpr int kRootReplayRequiredExitCode = 75;
constexpr const char kRootReplaySignalPrefix[] =
    "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2";
constexpr const char kRootReplaySignalSchema[] =
    "feebl.adaptive-root-replay-signal.v2";
constexpr const char kRootReplayRequestSchema[] =
    "feebl.adaptive-root-replay-request.v1";
constexpr const char kRootReplayGeneratorId[] =
    "feebl_legacy_mc_complete_run_v1";
constexpr const char kRootReplayNormalizationId[] =
    "feebl_n_m_attempted_legacy_markers_v1";
constexpr unsigned int kRootReplayMinimumCompleteRuns = 4U;

class RootReplayRequired : public std::runtime_error {
 public:
  explicit RootReplayRequired(const std::string& reason)
      : std::runtime_error(reason) {}
};

}  // namespace adaptive_particles
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_ROOT_REPLAY_SIGNAL_CPP_H
