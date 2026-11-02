#ifndef FEEBL_WAVEFRONT_FIELD_WORKSPACE_CPP_H
#define FEEBL_WAVEFRONT_FIELD_WORKSPACE_CPP_H

#include "beamCPP.h"
#include "adaptiveResolutionControllerCPP.h"
#include "adaptiveMacroparticleCPP.h"
#include "gridCPP.h"
#include "sparseDirectFieldCPP.h"
#include "uniformWindowPlanCPP.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace feebl {
namespace wavefront_fields {

struct Configuration {
  int n_x = 0;
  int n_y = 0;
  int n_z = 0;
  int n_t = 0;
  float cut_x = 0.0f;
  float cut_y = 0.0f;
  float cut_z = 0.0f;
  float charge_sign = 0.0f;
  PHI_FLOAT macro_charge1 = 0.0;
  PHI_FLOAT macro_charge2 = 0.0;
  double physical_particles1 = 0.0;
  double physical_particles2 = 0.0;
  std::size_t maximum_slice_particles1 = 0U;
  std::size_t maximum_slice_particles2 = 0U;
  std::size_t lumi_event_capacity = 0U;
  bool sparse_cells = false;
  sparse_direct_fields::Mode field_solver = sparse_direct_fields::Mode::Fft;
  std::uint64_t sparse_direct_interaction_limit = 262144U;
  int adaptive_grid_mode = 0;
  std::string adaptive_profile = "summary";
  bool retain_full_window_descriptor_payload = true;
  adaptive_grid::UniformWindowPolicy adaptive_window_policy;
  std::uint64_t adaptive_max_cells = 0U;
  std::uint64_t canonical_geometry_revision = 1U;
  std::vector<std::size_t> adaptive_spacing_factors;
  std::string adaptive_field_operator = "legacy_cic12_v1";
  std::size_t adaptive_max_refinement_factor = 0U;
  double adaptive_anchor_x = 0.0;
  double adaptive_anchor_y = 0.0;
  double adaptive_error_target = 0.0;
  std::string adaptive_error_norm_policy = "component_rms_linf_v1";
  std::string adaptive_error_budget_sha256;
  double adaptive_error_quantile = 0.999;
  double adaptive_error_catastrophe_limit = 0.5;
  double adaptive_error_reference_field = 0.0;
  double adaptive_error_absolute_floor_fraction = 0.0;
  double adaptive_error_catastrophe_absolute_fraction = 0.0;
  double adaptive_error_exposure_limit = 0.0;
  bool adaptive_noise_aware_stop = false;
  std::string adaptive_source_model;
  std::size_t adaptive_source_reference_nx = 0U;
  std::size_t adaptive_source_reference_ny = 0U;
  double adaptive_source_width_x = 0.0;
  double adaptive_source_width_y = 0.0;
  double adaptive_source_error_target = 0.0;
  double adaptive_solver_error_target = 0.0;
  double adaptive_lumi_error_target = 0.0;
  double adaptive_lumi_floor = 0.0;
  double adaptive_tau_curvature = 0.0;
  double adaptive_tau_field_gradient = 0.0;
  double adaptive_tau_richardson = 0.0;
  double adaptive_tau_tail = 0.0;
  // Fraction of the initial longitudinal population-product distribution
  // eligible to consume the (separate) adaptive_tau_tail weighted-error
  // budget. Zero inherits adaptive_tau_tail for card compatibility.
  double adaptive_encounter_tail_fraction = 0.0;
  double adaptive_rho_floor = 0.0;
  double adaptive_field_floor = 0.0;
  unsigned int adaptive_indicator_smoothing_passes = 0U;
  double adaptive_refine_threshold = 1.0;
  double adaptive_coarsen_threshold = 0.35;
  std::string adaptive_richardson_order_source = "calibrated_table";
  std::string adaptive_richardson_order_table;
  double adaptive_min_neff = 0.0;
  bool adaptive_fail_on_unmet_error = true;
  std::size_t adaptive_refinement_ratio = 2U;
  std::size_t adaptive_blocking_factor = 2U;
  std::size_t adaptive_error_buffer_cells = 1U;
  std::size_t adaptive_transition_cells = 4U;
  double adaptive_fine_residual_tolerance = 0.0;
  // Default-off certification that may extend a terminal Mode-3 result with
  // two local fixed-width-source patches.  Validation rejects every other
  // source/operator/norm combination.
  bool adaptive_terminal_patch_certification = false;
  adaptive_particles::Configuration adaptive_macroparticles;
};

// Out-of-line finite predicates implemented under a strict floating-point
// contract.  The historical field solver keeps its legacy fast-math contract,
// while all validation and run-time finite guards remain effective.
bool finite_value(float value) noexcept;
bool finite_value(double value) noexcept;
bool finite_value(long double value) noexcept;

namespace runtime_safety {

// Strict-FP result for the rolling three-level luminosity convergence audit.
// An unreliable triplet always carries relative_error >= 1, so the production
// source targets (which are required to lie in (0,1)) fail closed.
struct LuminosityOracleEstimate {
  double relative_error = 1.0;
  double observed_order = 0.0;
  bool reliable = false;
};

LuminosityOracleEstimate estimate_fixed_width_luminosity_error(
    double luminosity0, double luminosity1, double luminosity2,
    std::size_t target_slot, double refinement_ratio,
    double luminosity_floor);

// Bound the largest cell array for which both one retained FFT context and
// the simultaneous controller scratch can fit in a worker lane.  FeeBL's
// production wavefront contract requires at least two lanes, so max_bytes is
// divided by minimum_worker_lanes before the bound is formed.
std::size_t reachable_adaptive_cell_bound(
    std::uint64_t max_cells, std::size_t max_bytes,
    std::size_t bootstrap_context_bytes,
    std::size_t fixed_transient_bytes,
    std::size_t context_bytes_per_cell,
    std::size_t scratch_bytes_per_cell,
    std::size_t minimum_worker_lanes = 2U);

// Strict final-run consistency decision for the canonical adaptive-grid
// certificate.  Counter relationships are validated before any unsigned
// subtraction, and an outstanding external replay request can never coexist
// with all_encounters_certified=true.
struct AdaptiveGridCertificateDecision {
  std::uint64_t certified_encounters = 0U;
  // Accurate name for the historical `resource_unmet_encounters` quantity:
  // it is every unresolved encounter other than an early root-noise stop and
  // does not, by itself, prove a cell/byte/factor resource exhaustion.
  std::uint64_t unresolved_non_noise_encounters = 0U;
  // Compatibility alias retained for the V1 certificate record.  New code
  // and diagnostics must use unresolved_non_noise_encounters.
  std::uint64_t resource_unmet_encounters = 0U;
  bool external_replay_recommended = false;
  bool all_encounters_certified = false;
};

AdaptiveGridCertificateDecision evaluate_adaptive_grid_certificate(
    std::uint64_t encounters,
    std::uint64_t unmet_encounters,
    std::uint64_t noise_stop_encounters,
    std::uint64_t vector_catastrophe_failure_encounters,
    double maximum_vector_difference,
    double catastrophe_limit,
    std::uint64_t terminal_patch_attempts,
    std::uint64_t terminal_patch_accepts,
    std::uint64_t terminal_patch_used,
    std::uint64_t terminal_patch_rejects,
    std::uint64_t terminal_patch_uniform_fallbacks,
    std::uint64_t terminal_patch_resource_rejects,
    std::uint64_t terminal_patch_nonresource_rejects,
    std::uint32_t terminal_patch_rejection_mask_or,
    bool root_replay_required);

}  // namespace runtime_safety

// Beam-independent production preflight.  Exposed for embedders and focused
// tests; its implementation shares the strict floating-point contract above.
void validate_workspace_configuration(const Configuration& configuration);

// Stable logical identity used to bind all variable-draw legacy kernels in an
// encounter to schedule-independent CounterRngV2 streams.  No worker or
// thread identifier is permitted in this structure.
struct PhysicsIdentity {
  std::uint64_t run_seed = 1U;
  std::uint64_t crossing = 0U;
  std::uint64_t diagonal = 0U;
  std::uint64_t wave = 0U;
  std::uint64_t substep = 0U;
  std::uint64_t encounter = 0U;
  int legacy_time_counter = 0;
};

enum NestedPatchRejection : std::uint32_t {
  NestedPatchRejectionNone = 0U,
  NestedPatchRejectionGeometry = 1U << 0,
  NestedPatchRejectionTail = 1U << 1,
  NestedPatchRejectionStraddler = 1U << 2,
  NestedPatchRejectionEmptySupport = 1U << 3,
  NestedPatchRejectionSampling = 1U << 4,
  NestedPatchRejectionResidual = 1U << 5,
  NestedPatchRejectionFieldOracle = 1U << 6,
  NestedPatchRejectionLumiOracle = 1U << 7
};

struct FieldPrefetch {
  std::vector<float> fields1;
  std::vector<float> fields2;
  double fine_lumi_dot = 0.0;
  std::size_t beam1_in = 0;
  std::size_t beam1_out = 0;
  std::size_t beam2_in = 0;
  std::size_t beam2_out = 0;
  std::shared_ptr<const adaptive_grid::UniformWindowPlan> window_plan;
  std::uint64_t lattice_hash = 0U;
  // Binds the selected field spacing/topology in Modes 2/3.  It is zero for
  // fixed topology and exact-crop Mode 1.
  std::uint64_t result_geometry_hash = 0U;
  std::uint64_t result_binding_hash = 0U;
  std::size_t selected_spacing_factor = 1U;
  std::size_t selected_grid_n_x = 0U;
  std::size_t selected_grid_n_y = 0U;
  double selected_delta_x = 0.0;
  double selected_delta_y = 0.0;
  double estimated_field_error = 0.0;
  double field_difference_rms = 0.0;
  double field_difference_linf = 0.0;
  // Opt-in vector_rms_quantile_v1 diagnostics.  Component Linf remains in
  // field_difference_linf for comparison, but is not an acceptance gate for
  // this policy.
  double vector_difference_rms = 0.0;
  double vector_difference_quantile = 0.0;
  double vector_difference_maximum = 0.0;
  double vector_mixed_difference_maximum = 0.0;
  double vector_absolute_difference_rms = 0.0;
  double vector_absolute_difference_quantile = 0.0;
  double vector_absolute_difference_maximum = 0.0;
  std::size_t vector_catastrophe_targets = 0U;
  double vector_catastrophe_exposure = 0.0;
  double vector_quantile_probability = 0.999;
  std::size_t vector_accepted_targets = 0U;
  std::size_t vector_above_target = 0U;
  double vector_above_target_exposure = 0.0;
  double vector_maximum_beam_above_target_exposure = 0.0;
  bool vector_catastrophe_guard_passed = true;
  bool noise_aware_stop_triggered = false;
  std::size_t noise_aware_stop_factor = 0U;
  bool noise_aware_stop_quantile_root_limited = false;
  bool noise_aware_stop_noncontracting = false;
  std::size_t vector_distinct_roots = 0U;
  std::size_t vector_minimum_quantile_roots = 0U;
  // Tail-only bounded sensitivity.  Unlike the Richardson estimate above,
  // this quantity never divides by r^p-1 and therefore remains meaningful
  // when a sparse tail encounter is demonstrably outside the asymptotic
  // regime.  The two pair values bind the terminal same-regime triplet; the
  // published sensitivity is their conservative maximum.
  double tail_field_sensitivity = 0.0;
  double tail_field_sensitivity_previous_pair = 0.0;
  double tail_field_sensitivity_latest_pair = 0.0;
  // Uniform-grid density-overlap measure at the selected terminal level.
  // This is kept separate from fine_lumi_dot because Mode 3 may subsequently
  // replace the latter with a composite-patch value.
  double tail_uniform_lumi_measure = 0.0;
  double charge_curvature_score = 0.0;
  double field_gradient_score = 0.0;
  double effective_particles_per_cell = 0.0;
  double minimum_effective_particles_per_cell = 0.0;
  std::size_t source_reference_factor = 1U;
  std::uint64_t source_model_tag = 0U;
  std::uint64_t source_density_hash = 0U;
  std::uint64_t source_sampling_hash = 0U;
  std::uint64_t source_sampling_hierarchy_hash = 0U;
  int adaptive_macroparticle_mode = 0;
  std::size_t adaptive_macroparticle_roots1 = 0U;
  std::size_t adaptive_macroparticle_roots2 = 0U;
  std::size_t adaptive_macroparticle_representatives1 = 0U;
  std::size_t adaptive_macroparticle_representatives2 = 0U;
  std::size_t adaptive_macroparticle_premerge1 = 0U;
  std::size_t adaptive_macroparticle_premerge2 = 0U;
  std::size_t adaptive_macroparticle_many_root1 = 0U;
  std::size_t adaptive_macroparticle_many_root2 = 0U;
  double adaptive_macroparticle_weight_ess1 = 0.0;
  double adaptive_macroparticle_weight_ess2 = 0.0;
  double adaptive_macroparticle_clone_support1 = 0.0;
  double adaptive_macroparticle_clone_support2 = 0.0;
  double adaptive_macroparticle_projection_residual = 0.0;
  std::uint64_t adaptive_macroparticle_hash1 = 0U;
  std::uint64_t adaptive_macroparticle_hash2 = 0U;
  std::uint64_t adaptive_macroparticle_revision1 = 0U;
  std::uint64_t adaptive_macroparticle_revision2 = 0U;
  bool adaptive_macroparticle_committed = false;
  bool source_budget_runtime_verified = false;
  double source_build_ms = 0.0;
  double source_restrict_ms = 0.0;
  double source_prolong_ms = 0.0;
  double source_maximum_charge_closure = 0.0;
  double source_maximum_projection_moment_shift = 0.0;
  double source_lumi_oracle_error = 0.0;
  double source_maximum_lumi_oracle_error = 0.0;
  bool error_target_met = true;
  bool noise_limited = false;
  // Modes 2/3 may publish the terminal uniform solution for a longitudinal
  // slice pair that belongs to the precomputed whole-bunch tail budget.  Such
  // an encounter remains explicitly *unmet*; it is never relabelled as
  // locally converged.  The run-level weighted-error audit is authoritative.
  bool encounter_tail_budget_eligible = false;
  bool encounter_tail_budget_used = false;
  double encounter_population_weight = 0.0;
  double encounter_tail_selected_fraction = 0.0;
  bool nested_patch_requested = false;
  bool nested_patch_used = false;
  bool nested_patch_promoted_to_uniform = false;
  std::uint32_t nested_patch_rejection_mask = NestedPatchRejectionNone;
  std::size_t nested_patch_coarse_cells = 0U;
  std::size_t nested_patch_fine_cells = 0U;
  double nested_patch_coarse_delta_x = 0.0;
  double nested_patch_coarse_delta_y = 0.0;
  double nested_patch_fine_delta_x = 0.0;
  double nested_patch_fine_delta_y = 0.0;
  double nested_patch_effective_particles_per_cell = 0.0;
  double nested_patch_minimum_effective_particles_per_cell = 0.0;
  double nested_patch_residual = 0.0;
  double nested_patch_field_error_rms = 0.0;
  double nested_patch_field_error_linf = 0.0;
  double nested_patch_lumi_error = 0.0;
  double nested_patch_tail_fraction = 0.0;
  std::size_t nested_patch_straddling_sources = 0U;
  std::size_t nested_patch_iterations = 0U;
  bool terminal_patch_certification_attempted = false;
  bool terminal_patch_certification_accepted = false;
  bool terminal_patch_used = false;
  bool terminal_patch_uniform_fallback = false;
  bool terminal_patch_external_replay_recommended = false;
  std::uint32_t terminal_patch_rejection_mask = 0U;
  std::size_t terminal_patch_factor_r = 0U;
  std::size_t terminal_patch_factor_r2 = 0U;
  std::size_t terminal_patch_cells_r = 0U;
  std::size_t terminal_patch_cells_r2 = 0U;
  std::uint64_t terminal_patch_geometry_hash_r = 0U;
  std::uint64_t terminal_patch_geometry_hash_r2 = 0U;
  std::uint64_t terminal_patch_source_hash_r = 0U;
  std::uint64_t terminal_patch_source_hash_r2 = 0U;
  std::uint64_t terminal_patch_local_target_hash1 = 0U;
  std::uint64_t terminal_patch_local_target_hash2 = 0U;
  std::uint64_t terminal_patch_outside_target_hash1 = 0U;
  std::uint64_t terminal_patch_outside_target_hash2 = 0U;
  double terminal_patch_outside_unresolved_exposure = 0.0;
  double terminal_patch_outside_field_error_contribution = 0.0;
  double terminal_patch_outside_field_maximum = 0.0;
  double terminal_patch_outside_luminosity_fraction = 0.0;
  double terminal_patch_outside_luminosity_earlier = 0.0;
  double terminal_patch_outside_luminosity_previous = 0.0;
  double terminal_patch_outside_luminosity_parent = 0.0;
  double terminal_patch_observed_field_order = 0.0;
  double terminal_patch_observed_outside_field_order = 0.0;
  double terminal_patch_composite_field_rms = 0.0;
  double terminal_patch_composite_field_quantile = 0.0;
  double terminal_patch_composite_field_maximum = 0.0;
  double terminal_patch_composite_field_bound = 0.0;
  double terminal_patch_observed_luminosity_order = 0.0;
  double terminal_patch_observed_outside_luminosity_order = 0.0;
  double terminal_patch_outside_luminosity_error = 0.0;
  double terminal_patch_composite_luminosity_bound = 0.0;
  double terminal_patch_boundary_flux_rms = 0.0;
  double terminal_patch_boundary_flux_quantile = 0.0;
  double terminal_patch_boundary_flux_maximum = 0.0;
  int bound_slice1 = -1;
  int bound_slice2 = -1;
  std::size_t bound_particle_count1 = 0U;
  std::size_t bound_particle_count2 = 0U;
};

// Validate immediately before any consuming collision mutates a slice.
// Fixed/GPU-mode-4 prefetches have no window_plan and retain their established
// contract; adaptive results are strictly bound to their planned slice view.
void validate_field_result_binding(
    const FieldPrefetch& fields, int slice1, int slice2,
    const std::vector<PARTICLE*>& particles1,
    const std::vector<PARTICLE*>& particles2);

// Publish the authoritative pointer/order/transverse-coordinate snapshot
// guard after planning or after an external backend has filled a prefetched
// result. CPU adaptive modes use the same private hash through
// Workspace::prepare; GPU Mode 5 calls this boundary only after its stronger
// encounter/revision/plan binding has also passed.
void bind_field_result(
    FieldPrefetch& fields, int slice1, int slice2,
    const std::vector<PARTICLE*>& particles1,
    const std::vector<PARTICLE*>& particles2);

// One instance is permanently leased to one persistent worker lane.  It owns
// all mutable charge, FIELD, FFT plan, and temporary gather state used during
// the parallel half of a staged wave.  Shared BEAM objects are read-only until
// the wave barrier completes.
class Workspace {
 public:
  Workspace();
  ~Workspace();

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  static std::size_t estimated_bytes(const Configuration& configuration);
  static std::size_t estimated_adaptive_context_bytes(
      const Configuration& configuration);
  static std::size_t estimated_adaptive_transient_bytes(
      const Configuration& configuration);
  void initialize(const Configuration& configuration,
                  BEAM* beam1, BEAM* beam2);
  void initialize_physics(SWITCHES* switches,
                          PAIR_BEAM* secondaries,
                          PAIR_BEAM* muons,
                          double pair_s4,
                          double pair_lns4,
                          double spectrum_lo,
                          double spectrum_hi,
                          std::size_t spectrum_bins,
                          std::uint64_t run_seed,
                          const feebl::design_axis::Spectrum* design_axis_spectrum = NULL);
  std::shared_ptr<const adaptive_grid::UniformWindowPlan>
  make_window_plan(int slice1, int slice2);
  // Plan construction is completed before this call.  A first lane may build
  // authoritative FFTW wisdom; later lanes require that exact wisdom.
  void prebuild_adaptive_class(std::size_t n_x, std::size_t n_y,
                               bool wisdom_only,
                               std::size_t refinement_factor = 1U);
  bool has_adaptive_class(std::size_t n_x, std::size_t n_y,
                          std::size_t refinement_factor = 1U) const noexcept;
  void prepare(int slice1, int slice2, FieldPrefetch& output);
  void process_primary_photon_collision(
      int slice1, int slice2, const FieldPrefetch& fields,
      const PhysicsIdentity& identity);
  bool publish_primary_photon_collision(GRID& target,
                                        std::string* reason);
  const GRID& ready_primary_photon_lane() const;
  void acknowledge_primary_photon_publication();
  // Valid after all ready collisions have been published.  The GRID retains
  // only its crossing-local deterministic luminosity reservoir at that point.
  const GRID& completed_primary_photon_lane() const;
  bool initialized() const noexcept { return initialized_; }
  bool physics_initialized() const noexcept { return physics_initialized_; }
  bool collision_ready() const noexcept { return collision_ready_; }
  std::uint64_t canonical_lattice_hash() const noexcept {
    return canonical_lattice_ ? canonical_lattice_->hash() : 0U;
  }
  std::uint64_t dense_clear_visits() const noexcept
  { return dense_clear_visits_; }
  std::uint64_t actual_clear_visits() const noexcept
  { return actual_clear_visits_; }
  std::uint64_t dense_dot_visits() const noexcept
  { return dense_dot_visits_; }
  std::uint64_t actual_dot_visits() const noexcept
  { return actual_dot_visits_; }
  std::uint64_t direct_encounters() const noexcept
  { return direct_encounters_; }
  std::uint64_t fft_encounters() const noexcept { return fft_encounters_; }
  std::uint64_t direct_active_sources() const noexcept
  { return direct_active_sources_; }
  std::uint64_t direct_target_nodes() const noexcept
  { return direct_target_nodes_; }
  std::uint64_t estimated_interactions() const noexcept
  { return estimated_interactions_; }
  std::uint64_t direct_interactions() const noexcept
  { return direct_interactions_; }
  std::uint64_t maximum_estimated_interactions() const noexcept
  { return maximum_estimated_interactions_; }
  double direct_ms() const noexcept { return direct_ms_; }
  double fft_ms() const noexcept { return fft_ms_; }
  std::uint64_t adaptive_encounters() const noexcept
  { return adaptive_encounters_; }
  std::uint64_t adaptive_full_window_encounters() const noexcept
  { return adaptive_full_window_encounters_; }
  std::uint64_t adaptive_required_cells() const noexcept
  { return adaptive_required_cells_; }
  std::uint64_t adaptive_local_cells() const noexcept
  { return adaptive_local_cells_; }
  std::uint64_t adaptive_padded_cells() const noexcept
  { return adaptive_padded_cells_; }
  std::uint64_t adaptive_global_cells_avoided() const noexcept
  { return adaptive_global_cells_avoided_; }
  std::uint64_t adaptive_plan_builds() const noexcept
  { return adaptive_plan_builds_; }
  std::uint64_t adaptive_context_builds() const noexcept
  { return adaptive_context_builds_; }
  std::uint64_t adaptive_context_hits() const noexcept
  { return adaptive_context_hits_; }
  std::uint64_t adaptive_uniform_context_misses() const noexcept
  { return adaptive_uniform_context_misses_; }
  std::uint64_t adaptive_patch_context_misses() const noexcept
  { return adaptive_patch_context_misses_; }
  std::uint64_t adaptive_context_evictions() const noexcept
  { return adaptive_context_evictions_; }
  std::size_t adaptive_context_bytes() const noexcept
  { return adaptive_context_bytes_; }
  double adaptive_bounds_ms() const noexcept { return adaptive_bounds_ms_; }
  double adaptive_context_build_ms() const noexcept
  { return adaptive_context_build_ms_; }
  double adaptive_deposit_ms() const noexcept { return adaptive_deposit_ms_; }
  double adaptive_solve_ms() const noexcept { return adaptive_solve_ms_; }
  double adaptive_gather_ms() const noexcept { return adaptive_gather_ms_; }
  std::uint64_t mode2_encounters() const noexcept { return mode2_encounters_; }
  std::uint64_t mode2_field_solves() const noexcept { return mode2_field_solves_; }
  std::uint64_t mode2_refined_encounters() const noexcept
  { return mode2_refined_encounters_; }
  std::uint64_t mode2_unmet_error_encounters() const noexcept
  { return mode2_unmet_error_encounters_; }
  std::uint64_t mode2_noise_aware_stop_encounters() const noexcept
  { return mode2_noise_aware_stop_encounters_; }
  std::uint64_t vector_catastrophe_failure_encounters() const noexcept
  { return vector_catastrophe_failure_encounters_; }
  double maximum_vector_difference() const noexcept
  { return maximum_vector_difference_; }
  double maximum_beam_above_target_exposure() const noexcept
  { return maximum_beam_above_target_exposure_; }
  std::uint64_t mode2_tail_budget_eligible_encounters() const noexcept
  { return mode2_tail_budget_eligible_encounters_; }
  std::uint64_t mode2_tail_budget_used_encounters() const noexcept
  { return mode2_tail_budget_used_encounters_; }
  std::size_t encounter_tail_budget_selected_pairs() const noexcept
  { return encounter_tail_budget_selected_pairs_; }
  double encounter_tail_budget_selected_fraction() const noexcept
  { return encounter_tail_budget_selected_fraction_; }
  double encounter_tail_weighted_field_error() const noexcept
  { return encounter_tail_weighted_field_error_; }
  long double encounter_tail_weighted_field_sensitivity() const noexcept
  { return encounter_tail_weighted_field_sensitivity_; }
  double encounter_tail_weighted_source_lumi_error() const noexcept
  { return encounter_tail_weighted_source_lumi_error_; }
  long double encounter_all_uniform_lumi_measure() const noexcept
  { return encounter_all_uniform_lumi_measure_; }
  long double encounter_tail_uniform_lumi_measure() const noexcept
  { return encounter_tail_uniform_lumi_measure_; }
  const std::vector<std::uint64_t>& mode2_spacing_selection_counts() const
      noexcept { return mode2_spacing_selection_counts_; }
  std::uint64_t mode3_patch_encounters() const noexcept
  { return mode3_patch_encounters_; }
  std::uint64_t mode3_patch_requests() const noexcept
  { return mode3_patch_requests_; }
  std::uint64_t mode3_uniform_promotions() const noexcept
  { return mode3_uniform_promotions_; }
  std::uint64_t mode3_terminal_uniform_encounters() const noexcept
  { return mode3_terminal_uniform_encounters_; }
  std::uint64_t mode3_terminal_patch_attempts() const noexcept
  { return mode3_terminal_patch_attempts_; }
  std::uint64_t mode3_terminal_patch_accepts() const noexcept
  { return mode3_terminal_patch_accepts_; }
  std::uint64_t mode3_terminal_patch_used() const noexcept
  { return mode3_terminal_patch_used_; }
  std::uint64_t mode3_terminal_patch_rejects() const noexcept
  { return mode3_terminal_patch_rejects_; }
  std::uint64_t mode3_terminal_patch_uniform_fallbacks() const noexcept
  { return mode3_terminal_patch_uniform_fallbacks_; }
  std::uint64_t mode3_terminal_patch_resource_rejects() const noexcept
  { return mode3_terminal_patch_resource_rejects_; }
  std::uint64_t mode3_terminal_patch_nonresource_rejects() const noexcept
  { return mode3_terminal_patch_nonresource_rejects_; }
  std::uint32_t mode3_terminal_patch_rejection_mask_or() const noexcept
  { return mode3_terminal_patch_rejection_mask_or_; }
  std::uint64_t mode3_no_patch_encounters() const noexcept
  { return mode3_no_patch_encounters_; }
  std::uint64_t mode3_empty_encounters() const noexcept
  { return mode3_empty_encounters_; }
  std::uint32_t mode3_rejection_mask_or() const noexcept
  { return mode3_rejection_mask_or_; }
  std::uint64_t mode3_patch_coarse_cells() const noexcept
  { return mode3_patch_coarse_cells_; }
  std::uint64_t mode3_patch_fine_cells() const noexcept
  { return mode3_patch_fine_cells_; }
  std::uint64_t mode3_attempted_patch_coarse_cells() const noexcept
  { return mode3_attempted_patch_coarse_cells_; }
  std::uint64_t mode3_attempted_patch_fine_cells() const noexcept
  { return mode3_attempted_patch_fine_cells_; }
  std::uint64_t mode3_harmonic_iterations() const noexcept
  { return mode3_harmonic_iterations_; }
  double mode3_controller_ms() const noexcept
  { return mode3_controller_ms_; }
  double mode3_harmonic_ms() const noexcept
  { return mode3_harmonic_ms_; }
  double maximum_estimated_field_error() const noexcept
  { return maximum_estimated_field_error_; }
  double maximum_patch_residual() const noexcept
  { return maximum_patch_residual_; }
  double maximum_patch_field_error() const noexcept
  { return maximum_patch_field_error_; }
  double maximum_examined_patch_lumi_oracle_error() const noexcept
  { return maximum_examined_patch_lumi_oracle_error_; }
  double maximum_accepted_patch_lumi_oracle_error() const noexcept
  { return maximum_accepted_patch_lumi_oracle_error_; }
  std::uint64_t patch_lumi_oracle_evaluations() const noexcept
  { return patch_lumi_oracle_evaluations_; }
  std::uint64_t accepted_patch_lumi_oracle_evaluations() const noexcept
  { return accepted_patch_lumi_oracle_evaluations_; }
  double maximum_source_lumi_oracle_error() const noexcept
  { return maximum_source_lumi_oracle_error_; }
  double maximum_selected_source_lumi_oracle_error() const noexcept
  { return maximum_selected_source_lumi_oracle_error_; }
  double maximum_source_charge_closure() const noexcept
  { return maximum_source_charge_closure_; }
  double maximum_source_projection_moment_shift() const noexcept
  { return maximum_source_projection_moment_shift_; }
  std::uint64_t source_hierarchy_encounters() const noexcept
  { return source_hierarchy_encounters_; }
  std::uint64_t source_density_hash_nonzero_encounters() const noexcept
  { return source_density_hash_nonzero_encounters_; }
  std::uint64_t source_sampling_hash_nonzero_encounters() const noexcept
  { return source_sampling_hash_nonzero_encounters_; }
  std::uint64_t source_sampling_hierarchy_hash_nonzero_encounters() const noexcept
  { return source_sampling_hierarchy_hash_nonzero_encounters_; }
  std::uint64_t source_beam1_in_visits() const noexcept
  { return source_beam1_in_visits_; }
  std::uint64_t source_beam1_out_visits() const noexcept
  { return source_beam1_out_visits_; }
  std::uint64_t source_beam2_in_visits() const noexcept
  { return source_beam2_in_visits_; }
  std::uint64_t source_beam2_out_visits() const noexcept
  { return source_beam2_out_visits_; }
  double source_build_ms() const noexcept { return source_build_ms_; }
  double source_restrict_ms() const noexcept { return source_restrict_ms_; }
  double source_prolong_ms() const noexcept { return source_prolong_ms_; }

 private:
  void deposit_slice(const std::vector<PARTICLE*>& particles,
                     PHI_FLOAT macro_charge,
                     PHI_FLOAT* rho,
                     std::vector<std::size_t>& active_cells,
                     std::vector<unsigned char>& active_marks,
                     std::size_t& particles_in,
                     std::size_t& particles_out);
  void clear_rho(PHI_FLOAT* rho,
                 std::vector<std::size_t>& active_cells,
                 std::vector<unsigned char>& active_marks,
                 std::size_t cells);
  void collect_target_nodes(const std::vector<PARTICLE*>& particles,
                            std::vector<std::size_t>& target_nodes,
                            std::vector<unsigned char>& target_marks);
  void prepare_adaptive(int slice1, int slice2, FieldPrefetch& output);
  void prepare_resolution_adaptive(int slice1, int slice2,
                                   FieldPrefetch& output);
  void prepare_fixed(int slice1, int slice2, FieldPrefetch& output);
  adaptive_grid::CanonicalWindowInputs build_window_inputs(
      int slice1, int slice2);
  struct AdaptiveFieldContext;
  struct CoherentSourceHierarchy;
  struct LevelSolution;
  struct ResolutionEncounterBinding;
  struct ResolutionControllerState;
  struct ResolutionSelection;
  struct TerminalSourceSupport;
  struct TerminalCompositeLevel;
  void reset_resolution_output(ResolutionEncounterBinding& binding);
  void complete_empty_resolution(ResolutionEncounterBinding& binding);
  adaptive_grid::FieldLevelGeometry resolution_geometry(
      const ResolutionEncounterBinding& binding,
      std::size_t factor) const;
  void publish_uniform_resolution(
      ResolutionEncounterBinding& binding,
      const std::vector<LevelSolution>& levels,
      std::size_t index);
  void record_resolution_result(
      ResolutionEncounterBinding& binding,
      const CoherentSourceHierarchy& source_hierarchy,
      std::size_t selected,
      std::size_t spacing_class_count,
      bool vector_error_policy);
  void deposit_coherent_source(
      CoherentSourceHierarchy& source_hierarchy,
      const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
      const std::vector<PARTICLE*>& particles,
      PHI_FLOAT charge,
      std::vector<PHI_FLOAT>& rho,
      std::vector<PHI_FLOAT>& sampling_s1,
      std::vector<PHI_FLOAT>& sampling_s2,
      bool beam1,
      std::size_t& particles_in,
      std::size_t& particles_out,
      long double& accepted_charge,
      long double& accepted_x_moment,
      long double& accepted_y_moment) const;
  std::vector<adaptive_particles::RootProjection> make_adaptive_source_roots(
      const CoherentSourceHierarchy& source_hierarchy,
      const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
      const std::vector<PARTICLE*>& particles,
      const std::vector<PRIMARY_PARTICLE_ID>& ids,
      double macro_weight,
      std::uint64_t cohort) const;
  void publish_adaptive_source_receipt(
      FieldPrefetch& output,
      const adaptive_particles::SourceEnsemble& source,
      bool first) const;
  double verify_adaptive_source_projection(
      const std::vector<double>& candidate,
      std::vector<PHI_FLOAT>& oracle,
      const char* quantity_label,
      bool commit_candidate) const;
  void record_fixed_width_sources(
      std::uint64_t& density_hash,
      const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
      const std::vector<PARTICLE*>& particles,
      PHI_FLOAT charge,
      std::uint64_t beam_tag,
      std::size_t& particles_in,
      std::size_t& particles_out,
      long double& accepted_charge,
      long double& accepted_x_moment,
      long double& accepted_y_moment) const;
  void audit_coherent_source(
      const CoherentSourceHierarchy& source_hierarchy,
      const std::vector<PHI_FLOAT>& rho,
      long double accepted_charge,
      long double accepted_x_moment,
      long double accepted_y_moment) const;
  bool initialize_resolution_source(
      ResolutionEncounterBinding& binding,
      const std::vector<std::size_t>& spacing_factors,
      CoherentSourceHierarchy& source_hierarchy) const;
  void project_adaptive_source(
      ResolutionEncounterBinding& binding,
      CoherentSourceHierarchy& source_hierarchy) const;
  void hash_reference_source(CoherentSourceHierarchy& source_hierarchy) const;
  void publish_source_binding(
      FieldPrefetch& output,
      const CoherentSourceHierarchy& source_hierarchy) const;
  CoherentSourceHierarchy build_resolution_source(
      ResolutionEncounterBinding& binding,
      const std::vector<std::size_t>& spacing_factors);
  void solve_resolution_level(ResolutionControllerState& state);
  void ensure_resolution_levels(
      ResolutionControllerState& state,
      std::size_t required);
  void compare_resolution_pair(
      ResolutionControllerState& state,
      std::size_t index);
  void certify_resolution_pair(
      ResolutionControllerState& state,
      std::size_t pair);
  bool resolution_sampling_ok(const LevelSolution& level);
  bool resolution_indicators_ok(const LevelSolution& level);
  bool resolution_pair_sampling_ok(
      ResolutionControllerState& state,
      std::size_t pair);
  bool resolution_quantile_support_ok(
      ResolutionControllerState& state,
      std::size_t pair);
  double resolution_luminosity_error(
      ResolutionControllerState& state,
      std::size_t level_index);
  bool resolution_luminosity_ok(
      ResolutionControllerState& state,
      std::size_t level_index);
  double resolution_coarse_error(
      ResolutionControllerState& state,
      std::size_t pair);
  double resolution_fine_error(
      ResolutionControllerState& state,
      std::size_t pair);
  bool resolution_catastrophe_ok(
      ResolutionControllerState& state,
      std::size_t pair);
  bool resolution_exposure_ok(
      ResolutionControllerState& state,
      std::size_t pair);
  void publish_resolution_vector_error(
      ResolutionControllerState& state,
      std::size_t pair);
  std::vector<std::size_t> plan_resolution_spacing(
      ResolutionEncounterBinding& binding) const;
  void reconcile_resolution_contexts(
      ResolutionEncounterBinding& binding,
      const std::vector<std::size_t>& spacing_factors);
  void select_resolution_coarse(
      ResolutionControllerState& state,
      ResolutionSelection& selection);
  void select_resolution_terminal(
      ResolutionControllerState& state,
      ResolutionSelection& selection);
  void handle_resolution_unmet_error(
      ResolutionControllerState& state,
      ResolutionSelection& selection,
      bool terminal_patch_requested_by_card);
  void throw_resolution_failure(
      ResolutionControllerState& state,
      ResolutionSelection& selection);
  TerminalSourceSupport terminal_source_support(
      ResolutionEncounterBinding& binding,
      const LevelSolution& parent,
      const adaptive_grid::NestedPatchBox& candidate,
      std::vector<unsigned char>* mask1,
      std::vector<unsigned char>* mask2);
  void mask_terminal_targets(
      adaptive_grid::FieldOperator field_operator,
      const adaptive_grid::FieldLevelGeometry& patch_r,
      const adaptive_grid::FieldLevelGeometry& patch_r2,
      const std::vector<PARTICLE*>& particles,
      const std::vector<adaptive_grid::CanonicalGatherDescriptor>& descriptors,
      std::vector<unsigned char>& mask);
  void accumulate_terminal_localization(
      long double& outside_error_squared_sum,
      std::size_t& outside_unresolved,
      std::size_t& accepted_target_count,
      double& outside_error_maximum,
      double target,
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>& canonical_mask,
      const std::vector<unsigned char>& local_mask,
      std::vector<double>& outside_ratios);
  void collect_terminal_outside_ratios(
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>& canonical_mask,
      const std::vector<unsigned char>& local_mask,
      std::vector<double>& ratios);
  double terminal_outside_luminosity(
      long double patch_low_x,
      long double patch_high_x,
      long double patch_low_y,
      long double patch_high_y,
      const LevelSolution& level);
  void gather_terminal_composite(
      adaptive_grid::FieldOperator field_operator,
      const std::vector<PARTICLE*>& particles,
      const std::vector<unsigned char>& mask,
      const adaptive_grid::FieldLevelGeometry& geometry,
      const std::vector<double>& phi,
      std::vector<float>& fields);
  TerminalCompositeLevel build_terminal_composite(
      ResolutionEncounterBinding& binding,
      const LevelSolution& parent,
      const long double& parent_outside_lumi_dot,
      const std::vector<unsigned char>& target_mask1,
      const std::vector<unsigned char>& target_mask2,
      std::size_t operator_guard,
      const LevelSolution& local);
  void collect_terminal_vector_ratios(
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>& mask,
      std::vector<double>& ratios);
  void certify_terminal_resolution_patch(
      ResolutionControllerState& state,
      ResolutionSelection& selection,
      bool terminal_patch_requested_by_card,
      bool& terminal_patch_accepted);
  void apply_nested_resolution_patch(
      ResolutionControllerState& state,
      ResolutionSelection& selection,
      bool& mode3_promotion_required);
  void promote_resolution_uniform(
      ResolutionControllerState& state,
      ResolutionSelection& selection,
      bool mode3_promotion_required);
  AdaptiveFieldContext* find_adaptive_context(
      std::size_t n_x, std::size_t n_y,
      std::size_t refinement_factor = 1U) noexcept;
  AdaptiveFieldContext* require_adaptive_context(
      const adaptive_grid::FieldLevelGeometry& geometry,
      bool transient_patch_context = false);
  LevelSolution solve_field_level(
      int slice1, int slice2,
      const adaptive_grid::UniformWindowPlan& canonical_plan,
      const adaptive_grid::FieldLevelGeometry& geometry,
      const std::vector<unsigned char>* source_mask1 = NULL,
      const std::vector<unsigned char>* source_mask2 = NULL,
      const std::vector<unsigned char>* target_mask1 = NULL,
      const std::vector<unsigned char>* target_mask2 = NULL,
      bool transient_patch_context = false,
      const CoherentSourceHierarchy* source_hierarchy = NULL,
      const LevelSolution* registered_density_source = NULL);

  FFT_SERVER fourier_;
  GRID grid_;
  std::unique_ptr<RNDM> physics_rng_;
  SWITCHES* switches_;
  PAIR_BEAM* secondaries_;
  PAIR_BEAM* muons_;
  std::vector<GENERAL_GRID*> local_grids_;
  double pair_s4_;
  double pair_lns4_;
  std::uint64_t run_seed_;
  BEAM* beam1_;
  BEAM* beam2_;
  Configuration configuration_;
  std::vector<std::size_t> active_cells1_;
  std::vector<std::size_t> active_cells2_;
  std::vector<unsigned char> active_marks1_;
  std::vector<unsigned char> active_marks2_;
  std::vector<std::size_t> target_nodes1_;
  std::vector<std::size_t> target_nodes2_;
  std::vector<unsigned char> target_marks1_;
  std::vector<unsigned char> target_marks2_;
  std::unique_ptr<adaptive_grid::CanonicalLatticeSnapshot>
      canonical_lattice_;
  std::vector<std::unique_ptr<AdaptiveFieldContext> > adaptive_contexts_;
  std::unique_ptr<AdaptiveFieldContext> adaptive_patch_context_;
  std::uint64_t dense_clear_visits_;
  std::uint64_t actual_clear_visits_;
  std::uint64_t dense_dot_visits_;
  std::uint64_t actual_dot_visits_;
  std::uint64_t direct_encounters_;
  std::uint64_t fft_encounters_;
  std::uint64_t direct_active_sources_;
  std::uint64_t direct_target_nodes_;
  std::uint64_t estimated_interactions_;
  std::uint64_t direct_interactions_;
  std::uint64_t maximum_estimated_interactions_;
  double direct_ms_;
  double fft_ms_;
  std::uint64_t adaptive_encounters_;
  std::uint64_t adaptive_full_window_encounters_;
  std::uint64_t adaptive_required_cells_;
  std::uint64_t adaptive_local_cells_;
  std::uint64_t adaptive_padded_cells_;
  std::uint64_t adaptive_global_cells_avoided_;
  std::uint64_t adaptive_plan_builds_;
  std::uint64_t adaptive_context_builds_;
  std::uint64_t adaptive_context_hits_;
  std::uint64_t adaptive_uniform_context_misses_;
  std::uint64_t adaptive_patch_context_misses_;
  std::uint64_t adaptive_context_evictions_;
  std::size_t adaptive_context_bytes_;
  double adaptive_bounds_ms_;
  double adaptive_context_build_ms_;
  double adaptive_deposit_ms_;
  double adaptive_solve_ms_;
  double adaptive_gather_ms_;
  std::uint64_t mode2_encounters_;
  std::uint64_t mode2_field_solves_;
  std::uint64_t mode2_refined_encounters_;
  std::uint64_t mode2_unmet_error_encounters_;
  std::uint64_t mode2_noise_aware_stop_encounters_;
  std::uint64_t vector_catastrophe_failure_encounters_;
  double maximum_vector_difference_;
  // Maximum over beams within an encounter, then over all encounters handled
  // by this worker.  This prevents pooled exposure from hiding one bad beam.
  double maximum_beam_above_target_exposure_;
  std::uint64_t mode2_tail_budget_eligible_encounters_;
  std::uint64_t mode2_tail_budget_used_encounters_;
  std::vector<std::uint64_t> mode2_spacing_selection_counts_;
  std::vector<unsigned char> encounter_tail_exemption_mask_;
  std::vector<double> encounter_population_weights_;
  std::size_t encounter_tail_budget_selected_pairs_;
  double encounter_tail_budget_selected_fraction_;
  // Retained only as legacy diagnostic telemetry.  Non-asymptotic tail
  // qualification uses the bounded symmetric sensitivity below.
  double encounter_tail_weighted_field_error_;
  long double encounter_tail_weighted_field_sensitivity_;
  double encounter_tail_weighted_source_lumi_error_;
  long double encounter_all_uniform_lumi_measure_;
  long double encounter_tail_uniform_lumi_measure_;
  std::uint64_t mode3_patch_requests_;
  std::uint64_t mode3_patch_encounters_;
  std::uint64_t mode3_uniform_promotions_;
  std::uint64_t mode3_terminal_uniform_encounters_;
  std::uint64_t mode3_terminal_patch_attempts_;
  std::uint64_t mode3_terminal_patch_accepts_;
  std::uint64_t mode3_terminal_patch_used_;
  std::uint64_t mode3_terminal_patch_rejects_;
  std::uint64_t mode3_terminal_patch_uniform_fallbacks_;
  std::uint64_t mode3_terminal_patch_resource_rejects_;
  std::uint64_t mode3_terminal_patch_nonresource_rejects_;
  std::uint32_t mode3_terminal_patch_rejection_mask_or_;
  std::uint64_t mode3_no_patch_encounters_;
  std::uint64_t mode3_empty_encounters_;
  std::uint32_t mode3_rejection_mask_or_;
  std::uint64_t mode3_patch_coarse_cells_;
  std::uint64_t mode3_patch_fine_cells_;
  std::uint64_t mode3_attempted_patch_coarse_cells_;
  std::uint64_t mode3_attempted_patch_fine_cells_;
  std::uint64_t mode3_harmonic_iterations_;
  double mode3_controller_ms_;
  double mode3_harmonic_ms_;
  double maximum_estimated_field_error_;
  double maximum_patch_residual_;
  double maximum_patch_field_error_;
  double maximum_examined_patch_lumi_oracle_error_;
  double maximum_accepted_patch_lumi_oracle_error_;
  std::uint64_t patch_lumi_oracle_evaluations_;
  std::uint64_t accepted_patch_lumi_oracle_evaluations_;
  double maximum_source_lumi_oracle_error_;
  double maximum_selected_source_lumi_oracle_error_;
  double maximum_source_charge_closure_;
  double maximum_source_projection_moment_shift_;
  std::uint64_t source_hierarchy_encounters_;
  std::uint64_t source_density_hash_nonzero_encounters_;
  std::uint64_t source_sampling_hash_nonzero_encounters_;
  std::uint64_t source_sampling_hierarchy_hash_nonzero_encounters_;
  std::uint64_t source_beam1_in_visits_;
  std::uint64_t source_beam1_out_visits_;
  std::uint64_t source_beam2_in_visits_;
  std::uint64_t source_beam2_out_visits_;
  double source_build_ms_;
  double source_restrict_ms_;
  double source_prolong_ms_;
  bool initialized_;
  bool physics_initialized_;
  bool collision_ready_;
};

}  // namespace wavefront_fields
}  // namespace feebl

#endif  // FEEBL_WAVEFRONT_FIELD_WORKSPACE_CPP_H
