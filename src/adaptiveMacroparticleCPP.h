#ifndef FEEBL_ADAPTIVE_MACROPARTICLE_CPP_H
#define FEEBL_ADAPTIVE_MACROPARTICLE_CPP_H

#include "adaptiveResolutionControllerCPP.h"
#include "primaryParticleIdCPP.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace adaptive_particles {

// This module deliberately represents only the disposable field-source
// ensemble.  Physical primaries remain authoritative for kicks, particle-pair
// luminosity, radiation, spin and stored events.  Consequently a source
// representative is an integrated charge atom on a particular field lattice,
// not a new physical particle and not a new statistically independent root.

constexpr std::uint32_t kSourceRepresentativeSchema = 1U;

struct RepresentativeId {
  std::uint64_t high = 0U;
  std::uint64_t low = 0U;

  bool operator==(const RepresentativeId& other) const noexcept {
    return high == other.high && low == other.low;
  }
  bool operator<(const RepresentativeId& other) const noexcept {
    return high < other.high || (high == other.high && low < other.low);
  }
};

// Exact sparse root influence for an atom.  contribution is positive
// physical population and entries are kept in ascending root-ID order.
struct AncestryContribution {
  PRIMARY_PARTICLE_ID root_id = 0U;
  double contribution = 0.0;
};

struct SourceRepresentative {
  RepresentativeId representative_id;
  std::uint64_t statistical_cohort_id = 0U;
  std::uint64_t lineage_path = 0U;
  std::uint64_t birth_epoch = 0U;
  std::uint32_t generation = 0U;
  std::uint32_t flags = 0U;
  std::size_t global_cell_x = 0U;
  std::size_t global_cell_y = 0U;
  double x = 0.0;
  double y = 0.0;
  double physical_weight = 0.0;
  // Initialized to root_weight^2 for a root and distributed in proportion
  // to represented child mass.  It adds under merging.  Therefore splitting
  // cannot manufacture clone support.
  double ancestry_variance_mass = 0.0;
  std::vector<AncestryContribution> ancestry;
};

struct RootProjection {
  PRIMARY_PARTICLE_ID root_id = 0U;
  std::uint64_t statistical_cohort_id = 0U;
  double x = 0.0;
  double y = 0.0;
  double physical_weight = 0.0;
  adaptive_grid::LevelDepositDescriptor cic;
};

enum class Mode : int {
  LegacyOff = 0,
  Diagnostics = 1,
  SplitOnly = 2,
  CoupledSplitMerge = 3
};

struct Configuration {
  Mode mode = Mode::LegacyOff;
  std::string domain = "field_source";
  std::string source_evolution = "projected_current_state";
  std::string decision_scope = "encounter";
  std::string commit_scope = "wave";
  std::string strategy = "coupled_error_budget_v1";
  std::string split_algorithm = "assignment_preserving_v1";
  std::string merge_algorithm = "exact_cell_projection_v1";
  std::string root_source = "physical_primary_roots_v1";
  std::uint64_t root_seed = 0U;
  std::uint64_t scramble_id = 0U;
  double error_target = 0.0;
  double resample_bias_target = 0.0;
  double minimum_root_support = 0.0;
  double minimum_clone_support = 0.0;
  std::size_t statistical_cohorts = 1U;
  double refine_threshold = 1.0;
  double coarsen_threshold = 0.35;
  std::size_t update_interval = 1U;
  std::size_t cooldown = 0U;
  std::size_t dwell = 2U;
  double maximum_change_factor = 4.0;
  std::uint32_t maximum_generation = 32U;
  std::size_t maximum_count = 0U;
  std::size_t maximum_bytes = 0U;
  double maximum_weight_ratio = 1.0e6;
  std::string policy_save;
  std::string policy_load;
  std::string trace_save;
  std::string trace_load;
  std::string profile = "summary";
  bool fail_on_unmet_error = true;
  std::string determinism = "strict";
};

struct Metrics {
  std::size_t representative_count = 0U;
  std::size_t distinct_root_count = 0U;
  std::size_t statistical_cohort_count = 0U;
  std::size_t many_root_representatives = 0U;
  double total_weight = 0.0;
  double minimum_weight = 0.0;
  double maximum_weight = 0.0;
  double weight_coefficient_of_variation = 0.0;
  double weight_ess = 0.0;
  double clone_support_proxy = 0.0;
  bool exact_ancestry = true;
};

struct ProposalReceipt {
  std::size_t roots_seen = 0U;
  std::size_t roots_accepted = 0U;
  std::size_t roots_rejected = 0U;
  std::size_t representatives_before_merge = 0U;
  std::size_t representatives_after_merge = 0U;
  std::size_t split_children_created = 0U;
  // Number of (cohort, cell) groups containing at least two inputs.
  std::size_t merge_groups = 0U;
  std::size_t rollback_count = 0U;
  double relative_weight_residual = 0.0;
  double normalized_first_moment_residual = 0.0;
  double maximum_projection_residual = 0.0;
  std::uint64_t content_hash = 0U;
  bool diagnostics_only = false;
  bool committed = false;
  bool exact_projection = false;
  std::string qualification = "unqualified";
  std::string reason;
};

struct SourceEnsemble {
  std::vector<SourceRepresentative> representatives;
  Metrics metrics;
  ProposalReceipt receipt;
  std::uint64_t population_revision = 0U;
};

// Validate the card-level semantics without touching a beam or RNG.
void validate_configuration(const Configuration& configuration);

// Convert accepted root CIC projections into exact cell-centred charge atoms.
// Mode 1 executes and verifies the complete proposal but returns no active
// representatives. Mode 2 retains one atom per nonzero CIC contribution.
// Mode 3 additionally merges atoms with identical cell and statistical cohort
// while retaining their exact sparse ancestry ledger.
SourceEnsemble build_assignment_preserving_source(
    const Configuration& configuration,
    const adaptive_grid::FieldLevelGeometry& geometry,
    const std::vector<RootProjection>& roots,
    std::uint64_t beam_label,
    std::uint64_t epoch,
    std::uint64_t prior_population_revision = 0U);

// Deposit an already verified source ensemble.  The destination stores
// integrated physical population per control volume.  The S1/S2 arrays use
// root-grouped contributions, so a cell merge never inflates a sampling ESS.
void deposit_source(const SourceEnsemble& source,
                    const adaptive_grid::FieldLevelGeometry& geometry,
                    std::vector<double>& density,
                    std::vector<double>& root_grouped_s1,
                    std::vector<double>& root_grouped_s2);

Metrics compute_metrics(const std::vector<SourceRepresentative>& source);
std::uint64_t source_content_hash(
    const std::vector<SourceRepresentative>& source,
    std::uint64_t beam_label,
    std::uint64_t epoch,
    std::uint64_t population_revision);

const char* mode_name(Mode mode) noexcept;

}  // namespace adaptive_particles
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_MACROPARTICLE_CPP_H
