#ifndef FEEBL_ADAPTIVE_TERMINAL_PATCH_CPP_H
#define FEEBL_ADAPTIVE_TERMINAL_PATCH_CPP_H

#include "adaptiveResolutionControllerCPP.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace feebl {
namespace adaptive_grid {

// Rejection bits are deliberately independent of the runtime Mode-3 bits.
// This makes the terminal certificate usable as a small, deterministic audit
// kernel and prevents a caller from accidentally treating a conventional
// one-level patch rejection as a terminal two-level certification failure.
enum TerminalPatchRejection : std::uint32_t {
  TerminalPatchRejectionNone = 0U,
  TerminalPatchRejectionInvalidEvidence = 1U << 0,
  TerminalPatchRejectionNonlocalized = 1U << 1,
  TerminalPatchRejectionSampling = 1U << 2,
  TerminalPatchRejectionChargeClosure = 1U << 3,
  TerminalPatchRejectionMomentClosure = 1U << 4,
  TerminalPatchRejectionHarmonicResidual = 1U << 5,
  TerminalPatchRejectionBoundaryFlux = 1U << 6,
  TerminalPatchRejectionTransitionBuffer = 1U << 7,
  TerminalPatchRejectionNoncontracting = 1U << 8,
  TerminalPatchRejectionFieldBudget = 1U << 9,
  TerminalPatchRejectionLuminosityBudget = 1U << 10,
  TerminalPatchRejectionCatastrophe = 1U << 11,
  TerminalPatchRejectionResource = 1U << 12,
  TerminalPatchRejectionSourceSupport = 1U << 13
};

// Allocation gate for the two terminal refinement patches.  Geometry may be
// formed for diagnostics before this decision, but callers must not allocate
// a patch field context or invoke a patch solve unless permits_allocation()
// returns true.  A zero refinement-factor or cell cap is unbounded; the byte
// ceiling remains an explicit positive run-wide capacity.
struct TerminalPatchResourceDecision {
  bool factor_capacity_ok = false;
  bool cell_capacity_ok = false;
  bool byte_capacity_ok = false;
  std::uint32_t rejection_mask = TerminalPatchRejectionResource;

  bool permits_allocation() const noexcept {
    return rejection_mask == TerminalPatchRejectionNone;
  }
};

TerminalPatchResourceDecision evaluate_terminal_patch_resources(
    std::size_t factor_r, std::size_t factor_r2,
    std::size_t cells_r, std::size_t cells_r2,
    std::size_t bytes_r, std::size_t bytes_r2,
    std::size_t retained_context_bytes,
    std::size_t maximum_refinement_factor,
    std::size_t maximum_cells,
    std::size_t maximum_bytes);

// Evidence for an opt-in terminal Mode-3 patch.  The parent-to-r and
// r-to-r2 statistics must have been evaluated on the exact same accepted
// particle targets.  luminosity_parent/r/r2 are the same physical local
// ownership region on the parent, r-patch and r^2-patch meshes.  The outside
// luminosities are the disjoint complement on the rolling uniform triplet.
struct TerminalPatchEvidence {
  VectorFieldErrorStatistics parent_to_r;
  VectorFieldErrorStatistics r_to_r2;
  double luminosity_parent = 0.0;
  double luminosity_r = 0.0;
  double luminosity_r2 = 0.0;
  double outside_luminosity_earlier = 0.0;
  double outside_luminosity_previous = 0.0;
  double outside_luminosity_parent = 0.0;
  double luminosity_floor = 0.0;
  double outside_unresolved_exposure = 0.0;
  double outside_field_error_contribution = 0.0;
  // Per-target, no-floor symmetric-vector ratios.  The local arrays are the
  // r-to-r^2 differences on the identical patch support.  The two outside
  // arrays are a rolling same-target uniform triplet on every accepted target
  // not owned by the patch.  Both local and outside latest differences are
  // Richardson-scaled by their independently observed orders before their
  // disjoint union forms the complete-population RMS/quantile/max.  Thus a
  // slowly converging or rare catastrophic outside target cannot disappear
  // into a scalar arithmetic mean.
  std::array<std::vector<double>, 2> local_r_to_r2_ratios;
  std::array<std::uint64_t, 2> local_target_hash{{0U, 0U}};
  std::array<std::vector<double>, 2>
      outside_earlier_to_previous_ratios;
  std::array<std::vector<double>, 2> outside_error_surrogate_ratios;
  std::array<std::uint64_t, 2> outside_earlier_target_hash{{0U, 0U}};
  std::array<std::uint64_t, 2> outside_latest_target_hash{{0U, 0U}};
  double outside_luminosity_fraction = 0.0;
  double outside_source_tail_fraction = 0.0;
  double minimum_kish_neff_r = 0.0;
  double minimum_kish_neff_r2 = 0.0;
  double maximum_charge_closure = 0.0;
  double maximum_moment_closure = 0.0;
  double maximum_harmonic_residual = 0.0;
  double maximum_boundary_flux_rms = 0.0;
  double maximum_boundary_flux_quantile = 0.0;
  double maximum_boundary_flux_maximum = 0.0;
  std::size_t transition_cells_r = 0U;
  std::size_t transition_cells_r2 = 0U;
  std::size_t stencil_guard_r = 0U;
  std::size_t stencil_guard_r2 = 0U;
  std::size_t refinement_ratio = 2U;
};

struct TerminalPatchPolicy {
  double field_error_target = 0.0;
  double luminosity_error_target = 0.0;
  double outside_exposure_target = 0.0;
  double outside_source_tail_target = 0.0;
  double outside_luminosity_exposure_target = 0.0;
  double minimum_kish_neff = 0.0;
  double charge_closure_tolerance = 5.0e-12;
  double moment_closure_tolerance = 0.5;
  double harmonic_residual_tolerance = 0.0;
  double boundary_flux_tolerance = 0.0;
  double catastrophe_limit = 0.0;
  double minimum_observed_order = 0.25;
  double quantile_probability = 0.999;
};

struct TerminalPatchCertificate {
  bool accepted = false;
  std::uint32_t rejection_mask = TerminalPatchRejectionNone;
  double observed_field_order = 0.0;
  double observed_outside_field_order = 0.0;
  double local_field_error = 0.0;
  double outside_field_rms = 0.0;
  double outside_field_quantile = 0.0;
  double outside_field_maximum = 0.0;
  double outside_unresolved_exposure = 0.0;
  double composite_field_rms = 0.0;
  double composite_field_quantile = 0.0;
  double composite_field_maximum = 0.0;
  std::size_t composite_accepted_targets = 0U;
  std::size_t composite_above_target = 0U;
  double composite_above_target_exposure = 0.0;
  double maximum_vector_difference = 0.0;
  double composite_field_bound = 0.0;
  double observed_luminosity_order = 0.0;
  double observed_outside_luminosity_order = 0.0;
  double local_luminosity_error = 0.0;
  double outside_luminosity_error = 0.0;
  double composite_luminosity_bound = 0.0;
};

struct PatchBoundaryFluxStatistics {
  bool reliable = false;
  double rms = 0.0;
  double quantile = 0.0;
  double maximum = 0.0;
  std::size_t samples = 0U;
};

// Compare the normal derivative of a corrected nested potential with the
// parent potential prolongated to the same two cell-centre samples adjacent
// to every patch face.  This is a flux-continuity diagnostic, separate from
// the Dirichlet potential residual.  Corners are counted once per face, which
// is conservative for the high-quantile/max guards.
PatchBoundaryFluxStatistics estimate_patch_boundary_flux_mismatch(
    const double* parent_phi,
    const FieldLevelGeometry& parent_geometry,
    const std::vector<double>& corrected_patch_phi,
    const FieldLevelGeometry& patch_geometry,
    double quantile_probability,
    double field_floor);

// Pure fail-closed certificate used by the runtime and manufactured tests.
// It cannot make a patch acceptable merely because one metric is absent:
// missing/non-finite evidence sets InvalidEvidence and acceptance remains
// false.  The two total bounds are conservative sums, not RSS estimates.
TerminalPatchCertificate certify_terminal_patch(
    const TerminalPatchEvidence& evidence,
    const TerminalPatchPolicy& policy);

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_TERMINAL_PATCH_CPP_H
