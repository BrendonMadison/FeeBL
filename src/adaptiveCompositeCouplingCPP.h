#ifndef FEEBL_ADAPTIVE_COMPOSITE_COUPLING_CPP_H
#define FEEBL_ADAPTIVE_COMPOSITE_COUPLING_CPP_H

#include "adaptiveCompositePoissonCPP.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace feebl {
namespace adaptive_grid {

// A physical, tensor-product first-order B-spline (tent) particle.  In one
// dimension the normalized shape is
//
//   S(x; xp, h) = max(1 - |x-xp|/h, 0) / h,
//
// so its compact support is [xp-h, xp+h] and its integral is one.  The support
// half-widths are physical inputs: they are deliberately not inferred from the
// leaf containing the particle.  This keeps the particle model continuous as
// a particle crosses a coarse/fine interface.
struct CompositeTentParticleR2 {
  std::uint64_t stable_id = 0U;
  double position_x = 0.0;
  double position_y = 0.0;
  double signed_charge = 0.0;
  double support_half_width_x = 0.0;
  double support_half_width_y = 0.0;
};

enum class CompositeTentStatus : std::uint8_t {
  Accepted = 0U,
  OutsidePhysicalDomain = 1U,
  SupportOutsidePhysicalDomain = 2U,
  FixedBoundaryCenter = 3U,
  FixedBoundarySupport = 4U,
  IncompleteLeafPartition = 5U,
  // The compact shape overlaps a coarse/fine boundary.  The production
  // contract rejects this stencil: the layout must first grow/move the patch
  // until the complete shape is owned by exactly one level.
  SupportCrossesRefinementBoundary = 6U
};

struct CompositeChargeMomentsR2 {
  double signed_charge = 0.0;
  double absolute_charge = 0.0;
  double signed_first_x = 0.0;
  double signed_first_y = 0.0;
};

struct CompositeTentContributionR2 {
  std::size_t leaf = kCompositeNoIndex;

  // Integral of the normalized physical particle shape over this leaf.
  double weight = 0.0;

  // Exact derivatives of weight with respect to particle position.
  double derivative_x = 0.0;
  double derivative_y = 0.0;

  // Exact first spatial moments of the shape over this leaf.  These are
  // telemetry, not a cell-centre approximation.
  double shape_first_x = 0.0;
  double shape_first_y = 0.0;
};

struct CompositeTentStencilR2 {
  CompositeTentParticleR2 particle;
  CompositeTentStatus status = CompositeTentStatus::IncompleteLeafPartition;
  std::size_t owner_leaf = kCompositeNoIndex;
  bool touches_coarse = false;
  bool touches_fine = false;
  // Must be false for every Accepted production stencil.  It remains in the
  // receipt so a stale or forged split-level stencil is detectable.
  bool crosses_coarse_fine_interface = false;
  std::vector<CompositeTentContributionR2> contributions;

  double weight_sum = 0.0;
  double derivative_x_sum = 0.0;
  double derivative_y_sum = 0.0;
  double shape_first_x_sum = 0.0;
  double shape_first_y_sum = 0.0;
};

struct CompositeTentStencilSetR2 {
  // Fingerprint binds the stencils to the exact leaf topology and geometry.
  // Deposit and gather reject a mismatch rather than applying stale stencils.
  std::uint64_t topology_fingerprint = 0U;
  std::size_t leaf_count = 0U;
  double closure_tolerance = 0.0;
  std::vector<CompositeTentStencilR2> stencils;
};

struct CompositeDepositTelemetryR2 {
  std::size_t input_particles = 0U;
  std::size_t accepted_particles = 0U;
  std::size_t rejected_particles = 0U;
  std::size_t outside_particles = 0U;

  // Accepted, rejected, and outside form a disjoint partition of finite input
  // particles.  "Outside" includes a center outside the domain or a compact
  // support extending beyond it.  "Rejected" covers fixed-boundary contact or
  // failure of the geometric leaf-partition closure checks.
  CompositeChargeMomentsR2 input;
  CompositeChargeMomentsR2 accepted;
  CompositeChargeMomentsR2 rejected;
  CompositeChargeMomentsR2 outside;

  // The first entry integrates the analytic tent moments over leaves.  The
  // second interprets each integrated leaf charge at its cell centre and
  // therefore exposes, rather than hides, the finite-volume centroid error.
  CompositeChargeMomentsR2 deposited_exact_shape;
  CompositeChargeMomentsR2 deposited_cell_centres;

  double integrated_source_charge = 0.0;
  double maximum_weight_closure_defect = 0.0;
  double maximum_scaled_derivative_closure_defect = 0.0;
  double maximum_scaled_first_moment_defect = 0.0;
};

struct CompositeDepositResultR2 {
  std::vector<double> integrated_source;
  CompositeDepositTelemetryR2 telemetry;
};

struct CompositeAdjointFieldSampleR2 {
  std::uint64_t stable_id = 0U;
  CompositeTentStatus status = CompositeTentStatus::IncompleteLeafPartition;
  std::size_t owner_leaf = kCompositeNoIndex;
  double interpolated_potential = 0.0;
  double electric_x = 0.0;
  double electric_y = 0.0;
};

struct CompositeAdjointGatherResultR2 {
  std::vector<CompositeAdjointFieldSampleR2> samples;
};

// Build stencils in stable-id order.  Duplicate stable IDs and any non-finite
// particle datum are hard errors.  A stencil is accepted only when its full
// compact support is contained in one level (the finest level containing its
// centre, otherwise the unrefined coarse level).  A shape that crosses the
// refinement boundary is rejected so the layout controller can grow/move the
// patch; source mass is never split across levels or renormalized.
CompositeTentStencilSetR2 build_composite_tent_stencils_r2(
    const CompositePoissonR2& topology,
    const std::vector<CompositeTentParticleR2>& particles,
    double closure_tolerance = 2.0e-12);

// Deposit each accepted particle exactly once onto the finest-owner composite
// leaf hierarchy.  The resulting values are integrated signed charge per leaf.
CompositeDepositResultR2 deposit_composite_tent_source_once_r2(
    const CompositePoissonR2& topology,
    const CompositeTentStencilSetR2& stencil_set);

// Reciprocal gather for the same integrated weights:
//
//   Phi_p = sum_l w_l(xp) phi_l,
//   E_p   = -sum_l phi_l grad_xp w_l(xp).
//
// Thus, for fixed leaf potentials, phi^T delta(rho) = -sum_p q_p E_p.dx.
// This is the exact scatter/gather work identity used by the tests.  It is not
// a claim that a finite, bounded, nonuniform mesh has zero physical self-force.
CompositeAdjointGatherResultR2 gather_composite_tent_adjoint_r2(
    const CompositePoissonR2& topology,
    const CompositeTentStencilSetR2& stencil_set,
    const std::vector<double>& unknown_potential,
    const std::vector<double>& dirichlet_leaf_values);

std::uint64_t composite_coupling_topology_fingerprint_r2(
    const CompositePoissonR2& topology);

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_COMPOSITE_COUPLING_CPP_H
