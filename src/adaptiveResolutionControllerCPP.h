#ifndef FEEBL_ADAPTIVE_RESOLUTION_CONTROLLER_CPP_H
#define FEEBL_ADAPTIVE_RESOLUTION_CONTROLLER_CPP_H

#include "uniformWindowPlanCPP.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace adaptive_grid {

// The changed-spacing CPU path may retain the historical FeeBL gather or use
// a genuinely fourth-order field operator.  Versioned names are part of the
// numerical contract; callers must not infer an operator from a nominal
// interpolation order alone.
enum class FieldOperator {
  LegacyCic12V1 = 0,
  Fd4Lagrange4V1 = 1
};

FieldOperator parse_field_operator(const std::string& value);
const char* field_operator_name(FieldOperator value);

// The first value is the additional per-face window margin required when the
// canonical window was planned from the legacy twelve-read footprint.  The
// second is the complete fine-grid guard that must remain outside a Mode-3
// potential transition before this operator is gathered.
std::size_t field_operator_additional_window_guard_cells(
    FieldOperator value);
std::size_t field_operator_transition_guard_cells(FieldOperator value);

// Mode-2 spacing classes are integer refinements of the authoritative
// collision/luminosity lattice.  A factor r means dx/r and dy/r.  Keeping the
// two directions commensurate is required by the scalar Richardson estimator.
std::vector<std::size_t> parse_refinement_factors(const std::string& value);

// Parse either a single positive order ("2") or a comma-separated table of
// pair entries ("1:2=1.85,2:4=1.92").  The latter is keyed by refinement
// factors and deliberately has no interpolation/extrapolation semantics.
double richardson_order_for_pair(const std::string& table,
                                 std::size_t coarse_factor,
                                 std::size_t fine_factor);

struct EncounterTailBudget {
  // Row-major beam-1-slice x beam-2-slice mask.  A value of one exempts the
  // complete encounter pair; fractional-pair exemptions are forbidden.
  std::vector<unsigned char> exemption_mask;
  std::size_t selected_pairs = 0U;
  long double selected_fraction = 0.0L;
};

// Select the least-populated complete longitudinal encounter pairs without
// exceeding fractional_budget.  Pair (i,j) has normalized statistical weight
//
//   count1[i] count2[j] / (sum(count1) sum(count2)).
//
// Integer products are never formed, so size_t counts cannot overflow the
// weight calculation.  Pairs are considered in strict ascending
// (weight,i,j) order; the returned mask remains row-major.  Zero-weight pairs
// are selected even for a zero budget.  Empty beams, zero totals, and a
// non-finite or out-of-range budget fail closed.
EncounterTailBudget make_encounter_tail_budget(
    const std::vector<std::size_t>& beam1_slice_counts,
    const std::vector<std::size_t>& beam2_slice_counts,
    double fractional_budget);

struct FieldLevelGeometry {
  std::size_t refinement_factor = 1U;
  std::size_t global_n_x = 0U;
  std::size_t global_n_y = 0U;
  CanonicalIndex origin_x = 0;
  CanonicalIndex origin_y = 0;
  std::size_t n_x = 0U;
  std::size_t n_y = 0U;
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float inverse_delta_x = 0.0f;
  float inverse_delta_y = 0.0f;
  float offset_x = 0.0f;
  float offset_y = 0.0f;

  std::size_t cells() const;
  bool valid() const noexcept;
};

// A physical coordinate has one independent list of integrated source
// weights per transverse axis.  The Cartesian product of the x and y lists
// is therefore the complete separable two-dimensional source stencil.
enum class FieldAxis {
  X = 0,
  Y = 1
};

struct BSplineCellIntegral {
  std::size_t local_index = 0U;
  double weight = 0.0;
};

// Analytically integrate the normalized, fixed-physical-width linear
// B-spline
//
//   S_ell(u) = max(1 - |u| / ell, 0) / ell
//
// over every intersected cell control volume of one FieldLevelGeometry axis.
// The result contains strictly positive, non-negative-to-roundoff weights in
// ascending local-index order and closes to unity.  The call fails closed if
// the inputs are invalid or if any positive-measure part of the compact
// support lies outside either the global lattice or the local window.  A
// support endpoint exactly on a boundary is permitted.
void integrate_first_order_bspline_cells(
    double coordinate,
    double half_width,
    FieldAxis axis,
    const FieldLevelGeometry& geometry,
    std::vector<BSplineCellIntegral>& result);

// Convenience value-returning overload.  Performance-sensitive deposition
// loops should use the overload above so each thread can retain vector
// capacity in reusable scratch storage.
std::vector<BSplineCellIntegral> integrate_first_order_bspline_cells(
    double coordinate,
    double half_width,
    FieldAxis axis,
    const FieldLevelGeometry& geometry);

// Scale an exact Mode-1 active window onto a commensurate finer field
// lattice.  Extra fine-level guards absorb the phase/stencil difference
// between cell-centred lattices; no source or target is silently clipped.
FieldLevelGeometry make_refined_level_geometry(
    const CanonicalLatticeSnapshot& lattice,
    const UniformWindowPlan& window,
    std::size_t refinement_factor,
    std::size_t extra_fine_guard_cells,
    double anchor_x = 0.0,
    double anchor_y = 0.0,
    FieldOperator field_operator = FieldOperator::LegacyCic12V1);

struct LevelDepositDescriptor {
  bool accepted = false;
  int base_x = 0;
  int base_y = 0;
  float weight_x = 0.0f;
  float weight_y = 0.0f;
};

struct LevelGatherDescriptor {
  bool accepted = false;
  FieldOperator field_operator = FieldOperator::LegacyCic12V1;
  int y_base_x = 0;
  int y_base_y = 0;
  float y_weight = 0.0f;
  int x_base_x = 0;
  int x_base_y = 0;
  float x_weight = 0.0f;
  double fractional_x = 0.0;
  double fractional_y = 0.0;
  // Fd4Lagrange4V1 uses cell-centre coordinates.  The four in-domain
  // interpolation nodes are base-1, base, base+1, base+2 on each axis.  In
  // the interior each nodal field uses the radius-two fourth-order central
  // derivative.  At a physical global face the base may be shifted so all
  // interpolation nodes remain in-domain, and nodes 0/1 or n-2/n-1 use the
  // matching five-point fourth-order one-sided derivative closure.
  int centred_base_x = 0;
  int centred_base_y = 0;
  std::array<double, 4> lagrange_weights_x{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> lagrange_weights_y{{0.0, 0.0, 0.0, 0.0}};
};

LevelDepositDescriptor map_deposit_to_level(
    float x, float y, const FieldLevelGeometry& geometry);
LevelGatherDescriptor map_gather_to_level(
    float x, float y, const FieldLevelGeometry& geometry);
LevelGatherDescriptor map_gather_to_level(
    float x, float y, const FieldLevelGeometry& geometry,
    FieldOperator field_operator);

// Diagnostics for a coherent finite-volume charge transfer.  Each value is
// the integrated charge in one rectangular FIELD::dist_init control volume,
// not a pointwise or nodal density.  Consequently the transfer conserves the
// plain sum directly; callers apply no refinement-factor rescaling.
struct DensityTransferReport {
  double relative_charge_error = 0.0;
  double normalized_first_moment_error = 0.0;
  bool sign_preserving = true;
};

// Prolong the immutable canonical CIC control-volume charges onto a registered
// finer lattice without re-depositing particles.  Every canonical cell q is
// partitioned into the aligned r by r fine children q/r^2.  Thus every level
// represents exactly the same piecewise-constant physical source accepted by
// FIELD::dist_init.  This is deliberately different from applying CIC to each
// particle again at finer spacing, which shrinks its effective source support
// as resolution changes and invalidates a Richardson comparison.
//
// The destination refinement factor must be an integer multiple of the source
// factor, have the exactly scaled lattice offset, and contain every child of
// every nonzero source control volume.  A cropped local window that clips a
// nonzero parent cell is rejected.
DensityTransferReport prolong_cic_charge_density(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination);

// Copy a registered same-spacing subwindow.  Mode 3 uses this to obtain its
// patch RHS from the already-built uniform member of the coherent hierarchy,
// rather than performing a patch-specific particle deposition.
void extract_registered_density_window(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination);

// Conservative finite-volume restriction from a registered finer source
// lattice.  Each destination cell is the exact sum of its aligned source
// children.  Charge and sign are strict invariants; first-moment drift is
// reported because cell-average projection legitimately moves charge to the
// coarser cell centre.
DensityTransferReport restrict_cic_charge_density(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination);

// Add one particle's actual reference-CIC contributions to a registered
// source-or-coarser sampling level.  Contributions that land in the same
// parent control volume are summed before squaring, which is the particlewise
// definition required by Kish S2.  Restricting an already accumulated S2
// array would incorrectly omit the within-particle cross terms.
void accumulate_grouped_cic_sampling(
    const LevelDepositDescriptor& reference_descriptor,
    double absolute_charge,
    const FieldLevelGeometry& reference_geometry,
    const FieldLevelGeometry& sampling_geometry,
    std::vector<double>& sum_weight,
    std::vector<double>& sum_weight_squared);

enum class SourcePairRegime {
  Invalid = 0,
  SourceProjection = 1,
  FieldOnly = 2
};

SourcePairRegime source_pair_regime(
    std::size_t coarse_factor, std::size_t fine_factor,
    std::size_t source_factor) noexcept;

struct SameRegimeTriplet {
  bool valid = false;
  SourcePairRegime regime = SourcePairRegime::Invalid;
  std::size_t first_pair = 0U;
  std::size_t second_pair = 0U;
};

// Select two consecutive ratio-two difference pairs without crossing the
// source-projection/field-only transition.  Prefer a forward triplet and use
// the backward triplet at a regime's upper edge.
SameRegimeTriplet select_same_regime_triplet(
    const std::vector<std::size_t>& factors,
    std::size_t candidate_pair,
    std::size_t source_factor);

// Evaluate the historical FeeBL twelve-potential-node CIC field stencil on a
// non-canonical uniform field level.  The ordering intentionally mirrors
// GENERAL_GRID::ElectricFieldCIC.
std::array<float, 2> gather_level_field(
    const double* phi, const FieldLevelGeometry& geometry,
    const LevelGatherDescriptor& descriptor);

// Dispatching overload for changed-spacing Modes 2/3.  The legacy branch
// calls the preceding function without changing its arithmetic.  The FD4
// branch applies fourth-order derivatives at a 4x4 cell-centre neighborhood
// and tensor-product cubic Lagrange gathering in binary64.  Radius-two
// central differences are used in the interior; true global faces use
// five-point fourth-order one-sided closures without relaxing cropped-window
// support checks.
std::array<float, 2> gather_level_field(
    const double* phi, const FieldLevelGeometry& geometry,
    const LevelGatherDescriptor& descriptor,
    FieldOperator field_operator);

struct RichardsonEstimate {
  bool reliable = false;
  // Components are ordered beam-1 x/y, beam-2 x/y.  Acceptance uses the
  // worst component so neither the stronger beam nor the wider plane can
  // hide an unconverged kick in another channel.
  std::array<double, 4> absolute_difference_rms_components{{0.0, 0.0, 0.0,
                                                             0.0}};
  std::array<double, 4> absolute_difference_linf_components{{0.0, 0.0, 0.0,
                                                              0.0}};
  std::array<double, 4> observed_difference_rms_components{{0.0, 0.0, 0.0,
                                                             0.0}};
  std::array<double, 4> observed_difference_linf_components{{0.0, 0.0, 0.0,
                                                              0.0}};
  double absolute_difference_rms = 0.0;
  double observed_difference_rms = 0.0;
  double observed_difference_linf = 0.0;
  double coarse_error_rms = 0.0;
  double coarse_error_linf = 0.0;
  double fine_error_rms = 0.0;
  double fine_error_linf = 0.0;
  // Conservative aggregate retained for the runtime/output contract.
  double coarse_error = 0.0;
  double fine_error = 0.0;
  double order = 0.0;
  double ratio = 0.0;
};

RichardsonEstimate estimate_same_particle_field_error(
    const std::vector<float>& coarse_fields1,
    const std::vector<float>& coarse_fields2,
    const std::vector<float>& fine_fields1,
    const std::vector<float>& fine_fields2,
    std::size_t refinement_ratio,
    double order,
    double field_floor,
    const std::vector<unsigned char>* accepted_targets1 = NULL,
    const std::vector<unsigned char>* accepted_targets2 = NULL);

// A bounded, non-extrapolated sensitivity measure for one pair of field
// levels.  Each particle contributes the Euclidean norm of its two transverse
// field components.  RMS and Linf differences are normalized independently
// for each beam by the symmetric scale
//
//   max(norm(coarse), norm(fine), field_floor).
//
// The triangle inequality therefore bounds every reported ratio and the
// aggregate maximum to [0,2], including when one level is arbitrarily close
// to zero.  This deliberately contains no Richardson factor: callers must not
// present it as a continuum-error estimate when a refinement sequence is not
// asymptotic.
struct SymmetricVectorFieldSensitivity {
  bool reliable = false;
  std::array<double, 2> rms_ratios{{0.0, 0.0}};
  std::array<double, 2> linf_ratios{{0.0, 0.0}};
  std::array<std::size_t, 2> accepted_targets{{0U, 0U}};
  double aggregate = 0.0;
};

SymmetricVectorFieldSensitivity estimate_symmetric_vector_field_sensitivity(
    const std::vector<float>& coarse_fields1,
    const std::vector<float>& coarse_fields2,
    const std::vector<float>& fine_fields1,
    const std::vector<float>& fine_fields2,
    double field_floor,
    const std::vector<unsigned char>* accepted_targets1 = NULL,
    const std::vector<unsigned char>* accepted_targets2 = NULL);

// Same-target empirical error statistics for the opt-in
// vector_rms_quantile_v1 policy.  For each accepted particle i, the bounded
// dimensionless change is
//
//   q_i = ||E_f,i-E_c,i||_2 / max(||E_f,i||_2, ||E_c,i||_2).
//
// q_i is defined as zero only when both vectors are exactly zero.  No
// user-selected or numerical floor enters this metric, so weak non-zero
// fields cannot be hidden by a denominator regularizer.  The triangle
// inequality gives 0 <= q_i <= 2.  Quantiles use the conservative nearest-
// rank empirical definition ceil(p*n)-1 independently per beam; aggregates
// are the worst beam so a low-statistics or weak beam is not averaged away.
struct VectorFieldErrorStatistics {
  bool reliable = false;
  // 1 is the historical unregularized relative policy above.  2 is the
  // mixed absolute/relative policy implemented by
  // estimate_mixed_vector_field_error_statistics_v2.  Keeping the version in
  // the evidence prevents a rolling Richardson triplet from silently mixing
  // two mathematical norms.
  unsigned int policy_version = 1U;
  double quantile_probability = 0.999;
  // V2-only frozen physical scale contract.  The mixed denominator floor is
  // absolute_floor_fraction * reference_field.  Catastrophes require both
  // an unregularized relative discrepancy above the configured relative
  // limit and an absolute discrepancy above catastrophe_absolute_threshold.
  double reference_field = 0.0;
  double absolute_floor_fraction = 0.0;
  double absolute_floor = 0.0;
  double catastrophe_relative_limit = 0.0;
  double catastrophe_absolute_fraction = 0.0;
  double catastrophe_absolute_threshold = 0.0;
  std::array<double, 2> rms{{0.0, 0.0}};
  std::array<double, 2> quantile{{0.0, 0.0}};
  std::array<double, 2> maximum{{0.0, 0.0}};
  // Absolute same-target differences are retained independently of the
  // normalized metric.  V2 Richardson order is estimated from these values,
  // never from a locally normalized ratio.
  std::array<double, 2> absolute_rms{{0.0, 0.0}};
  std::array<double, 2> absolute_quantile{{0.0, 0.0}};
  std::array<double, 2> absolute_maximum{{0.0, 0.0}};
  // Largest unregularized relative discrepancy among absolutely significant
  // targets.  Weak-field targets remain visible through the separate raw
  // diagnostics but cannot alone trip the V2 catastrophe gate.
  std::array<double, 2> significant_unregularized_maximum{{0.0, 0.0}};
  std::array<std::size_t, 2> accepted_targets{{0U, 0U}};
  // Stable hash of the full accepted-target mask (including rejected slots)
  // and, for V2 runtime evidence, every ordered stable physical root ID for
  // each beam.  Rolling comparisons require exact equality, not merely equal
  // counts.  V2 Richardson certification also requires the explicit ID-bound
  // bits; mask-only hashes remain available for historical V1 tests.
  std::array<std::uint64_t, 2> accepted_target_hash{{0U, 0U}};
  std::array<bool, 2> accepted_target_ids_bound{{false, false}};
  std::array<std::size_t, 2> above_target{{0U, 0U}};
  std::array<std::size_t, 2> catastrophe_targets{{0U, 0U}};
  double aggregate_rms = 0.0;
  double aggregate_quantile = 0.0;
  double aggregate_maximum = 0.0;
  double aggregate_absolute_rms = 0.0;
  double aggregate_absolute_quantile = 0.0;
  double aggregate_absolute_maximum = 0.0;
  double aggregate_significant_unregularized_maximum = 0.0;
  std::size_t total_accepted_targets = 0U;
  std::size_t total_above_target = 0U;
  std::size_t total_catastrophe_targets = 0U;
  double above_target_exposure = 0.0;
  std::array<double, 2> above_target_exposure_by_beam{{0.0, 0.0}};
  double maximum_beam_above_target_exposure = 0.0;
  double catastrophe_exposure = 0.0;
};

std::uint64_t accepted_target_identity_hash(
    const std::vector<unsigned char>& accepted_targets);

VectorFieldErrorStatistics estimate_vector_field_error_statistics(
    const std::vector<float>& coarse_fields1,
    const std::vector<float>& coarse_fields2,
    const std::vector<float>& fine_fields1,
    const std::vector<float>& fine_fields2,
    double quantile_probability,
    double error_target,
    const std::vector<unsigned char>* accepted_targets1 = NULL,
    const std::vector<unsigned char>* accepted_targets2 = NULL);

// V2 mixed absolute/relative policy.  For each accepted target i,
//
//   e_i = ||E_f-E_c|| /
//         max(||E_f||, ||E_c||, absolute_floor_fraction*reference_field).
//
// A catastrophe is recorded only when the unregularized ratio exceeds
// catastrophe_relative_limit AND the absolute discrepancy exceeds
// catastrophe_absolute_fraction*reference_field.  Both physical scales must
// be positive and are intended to be preregistered from a kick/QoI budget.
VectorFieldErrorStatistics estimate_mixed_vector_field_error_statistics_v2(
    const std::vector<float>& coarse_fields1,
    const std::vector<float>& coarse_fields2,
    const std::vector<float>& fine_fields1,
    const std::vector<float>& fine_fields2,
    double quantile_probability,
    double error_target,
    double reference_field,
    double absolute_floor_fraction,
    double catastrophe_relative_limit,
    double catastrophe_absolute_fraction,
    const std::vector<unsigned char>* accepted_targets1 = NULL,
    const std::vector<unsigned char>* accepted_targets2 = NULL,
    const std::vector<std::uint64_t>* target_ids1 = NULL,
    const std::vector<std::uint64_t>* target_ids2 = NULL);

// Three-level Richardson evidence for the vector RMS/high-quantile policy.
// Order is evaluated independently for every active beam x {RMS, quantile}
// component and the smallest observed order is used for every extrapolated
// error.  This prevents a changing aggregate argmax from hiding a growing
// beam or metric.  Empirical p-quantile support is also a certification gate:
// both colliding beams need at least ceil(1/(1-p)) accepted targets.
struct VectorFieldRichardsonEstimate {
  bool reliable = false;
  bool asymptotic = false;
  bool quantile_support_met = false;
  bool order_from_absolute_differences = false;
  bool roundoff_limited = false;
  bool plausible_order = false;
  double observed_order = 0.0;
  std::size_t minimum_quantile_targets = 0U;
  std::size_t minimum_accepted_targets = 0U;
  std::array<double, 2> coarse_rms_error{{0.0, 0.0}};
  std::array<double, 2> coarse_quantile_error{{0.0, 0.0}};
  std::array<double, 2> fine_rms_error{{0.0, 0.0}};
  std::array<double, 2> fine_quantile_error{{0.0, 0.0}};
  double coarse_error = 0.0;
  double fine_error = 0.0;
};

VectorFieldRichardsonEstimate estimate_vector_field_richardson_error(
    const VectorFieldErrorStatistics& earlier,
    const VectorFieldErrorStatistics& later,
    const VectorFieldErrorStatistics& current,
    std::size_t refinement_ratio,
    double minimum_observed_order = 0.25);

// Kish effective sample size for actual non-negative CIC contributions.
// Zero total weight has zero effective support.
double kish_effective_sample_size(long double sum_weight,
                                  long double sum_weight_squared);

// Half-open parent-level node box used by Mode 3.  A valid box is always
// contained in the parent geometry and has at least four nodes per axis.
struct NestedPatchBox {
  bool valid = false;
  std::size_t begin_x = 0U;
  std::size_t end_x = 0U;
  std::size_t begin_y = 0U;
  std::size_t end_y = 0U;

  std::size_t n_x() const noexcept { return end_x - begin_x; }
  std::size_t n_y() const noexcept { return end_y - begin_y; }
};

// Build the connected component containing maximum opposing-beam overlap.
// The indicator is a smoothed non-negative absolute-charge field.  Curvature
// may enlarge the component but never removes hard overlap support.
NestedPatchBox select_nested_patch(
    const std::vector<double>& rho1,
    const std::vector<double>& rho2,
    std::size_t n_x,
    std::size_t n_y,
    double overlap_fraction,
    double curvature_threshold,
    double rho_floor,
    unsigned int smoothing_passes,
    std::size_t buffer_cells,
    std::size_t transition_cells,
    std::size_t blocking_factor);

FieldLevelGeometry make_patch_geometry(
    const FieldLevelGeometry& parent, const NestedPatchBox& patch);
FieldLevelGeometry refine_patch_geometry(
    const FieldLevelGeometry& patch, std::size_t refinement_ratio);

// C1 smoothstep blend: zero at the fine-patch interface and one in its
// interior.  Stencil_guard is excluded before transition weighting.
double patch_transition_weight(float x, float y,
                               const FieldLevelGeometry& patch,
                               std::size_t transition_cells,
                               std::size_t stencil_guard = 2U);

// Bilinear nodal prolongation used to impose the coarse potential on a fine
// Dirichlet boundary.  Potential, rather than independently blended field
// components, is transferred so the resulting kick remains conservative.
double interpolate_level_potential(
    const double* phi, const FieldLevelGeometry& geometry,
    float x, float y);

struct HarmonicSolveReport {
  bool converged = false;
  std::size_t iterations = 0U;
  double normalized_residual_l2 = 0.0;
  double normalized_residual_linf = 0.0;
  double normalized_residual = 0.0;
};

// correction contains fixed Dirichlet values on its outermost nodes.  The
// interior is replaced by the deterministic red-black Gauss-Seidel solution
// of the anisotropic five-point harmonic equation.
HarmonicSolveReport solve_harmonic_dirichlet(
    std::vector<double>& correction,
    std::size_t n_x,
    std::size_t n_y,
    double delta_x,
    double delta_y,
    double tolerance,
    std::size_t maximum_iterations);

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_RESOLUTION_CONTROLLER_CPP_H
