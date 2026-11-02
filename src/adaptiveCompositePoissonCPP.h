#ifndef FEEBL_ADAPTIVE_COMPOSITE_POISSON_CPP_H
#define FEEBL_ADAPTIVE_COMPOSITE_POISSON_CPP_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace feebl {
namespace adaptive_grid {

// A deliberately small, deterministic two-level algebra kernel for the first
// flux-matched Mode-3 prototype.  It owns no particles, FFT state, or runtime
// policy.  The only supported hierarchy is one aligned rectangular patch with
// refinement ratio two.  Coarse cells covered by the patch are removed from
// the leaf space rather than retained as overlapping unknowns.

constexpr std::size_t kCompositeNoIndex =
    std::numeric_limits<std::size_t>::max();

enum class CompositeLevel : std::uint8_t {
  Coarse = 0U,
  Fine = 1U
};

enum class CompositeFaceAxis : std::uint8_t {
  XNormal = 0U,
  YNormal = 1U
};

enum class CompositeFaceKind : std::uint8_t {
  CoarseCoarse = 0U,
  FineFine = 1U,
  CoarseFine = 2U
};

struct CompositePatchGeometryR2 {
  std::size_t coarse_n_x = 0U;
  std::size_t coarse_n_y = 0U;
  double coarse_delta_x = 0.0;
  double coarse_delta_y = 0.0;
  double physical_origin_x = 0.0;
  double physical_origin_y = 0.0;

  // Half-open coarse-cell range replaced by the ratio-two fine patch.
  std::size_t patch_begin_x = 0U;
  std::size_t patch_end_x = 0U;
  std::size_t patch_begin_y = 0U;
  std::size_t patch_end_y = 0U;

  // The outer coarse-cell ring is prescribed Dirichlet data and is excluded
  // from the linear unknowns.  A further proper-nesting margin keeps the fine
  // patch from touching that authority ring.
  std::size_t fixed_boundary_layers = 1U;
  std::size_t proper_nesting_cells = 1U;
  std::size_t refinement_ratio = 2U;
};

struct CompositeLeafCell {
  CompositeLevel level = CompositeLevel::Coarse;
  std::size_t level_i = 0U;
  std::size_t level_j = 0U;
  double center_x = 0.0;
  double center_y = 0.0;
  double delta_x = 0.0;
  double delta_y = 0.0;
  double area = 0.0;
  bool fixed_dirichlet = false;
  std::size_t dof = kCompositeNoIndex;
};

// A geometric face is stored exactly once.  Same-level faces retain the
// familiar two-point coefficients (+1,-1,0).  A ratio-two coarse/fine
// subface uses a three-point variational jump:
//
//   q = c_a phi_a + c_b phi_b + c_c phi_c,
//   energy = 0.5 * transmissibility * q^2.
//
// leaf_c is the coarse tangential neighbour needed to reconstruct the coarse
// potential at the fine-subface centre.  Its coefficient is +/-1/4 and the
// primary coarse coefficient is +/-3/4.  This makes q exactly zero for a
// potential linear in the tangential coordinate.  Applying the Hessian of
// the energy gives a symmetric positive-semidefinite face contribution;
// prescribed outer data make the assembled operator positive definite.
struct CompositeFace {
  std::size_t leaf_a = kCompositeNoIndex;
  std::size_t leaf_b = kCompositeNoIndex;
  std::size_t leaf_c = kCompositeNoIndex;
  CompositeFaceAxis axis = CompositeFaceAxis::XNormal;
  CompositeFaceKind kind = CompositeFaceKind::CoarseCoarse;
  double area = 0.0;
  double center_distance = 0.0;
  double transmissibility = 0.0;
  double coefficient_a = 1.0;
  double coefficient_b = -1.0;
  double coefficient_c = 0.0;
};

struct CompositeFaceFlux {
  std::size_t face = kCompositeNoIndex;
  // Physical subface accounting is oriented from geometric side a to side b.
  // These values are structurally equal and opposite.  For a multipoint
  // interface, they are deliberately not the individual algebraic row
  // contributions of the primary coarse cell.
  double flux_a_to_b = 0.0;
  double outward_from_a = 0.0;
  double outward_from_b = 0.0;
  // Variational row contributions.  Their three-term sum is exactly zero;
  // at an interface the coarse-side physical flux is partitioned between
  // the primary coarse cell and its tangential neighbour.
  double algebraic_outward_from_a = 0.0;
  double algebraic_outward_from_b = 0.0;
  double algebraic_outward_from_c = 0.0;
};

// A diagnostic independent of the structural equal/opposite face accounting.
// Each side uses its own quadratic one-sided normal reconstruction at the
// geometric interface.  Both fluxes are expressed in the face a->b direction.
// Equality is therefore a numerical consistency property, not an identity.
struct CompositeInterfaceFluxConsistency {
  std::size_t face = kCompositeNoIndex;
  double variational_flux_a_to_b = 0.0;
  double coarse_one_sided_flux_a_to_b = 0.0;
  double fine_one_sided_flux_a_to_b = 0.0;
  double absolute_one_sided_defect = 0.0;
  double relative_one_sided_defect = 0.0;
};

struct CompositeResidualNorms {
  double l2 = 0.0;
  double rms = 0.0;
  double linf = 0.0;
};

struct CompositeGaussReport {
  double integrated_source = 0.0;
  double boundary_outward_flux = 0.0;
  double global_balance_defect = 0.0;
  double maximum_cell_balance_defect = 0.0;
  double cell_balance_l2 = 0.0;
  double maximum_two_sided_flux_defect = 0.0;
  double maximum_interface_two_sided_flux_defect = 0.0;
  double maximum_interface_one_sided_absolute_defect = 0.0;
  double maximum_interface_one_sided_relative_defect = 0.0;
  std::size_t interface_faces = 0U;
  std::size_t interface_one_sided_audited_faces = 0U;
  std::size_t interface_one_sided_unaudited_faces = 0U;
};

enum class CompositePcgFailure : std::uint8_t {
  None = 0U,
  MaximumIterations = 1U,
  NonFiniteArithmetic = 2U,
  NonPositiveCurvature = 3U,
  PreconditionerBreakdown = 4U
};

struct CompositePcgOptions {
  double relative_tolerance = 1.0e-10;
  double absolute_tolerance = 0.0;
  std::size_t maximum_iterations = 10000U;
  std::size_t residual_recompute_interval = 32U;
};

struct CompositePcgReport {
  bool converged = false;
  CompositePcgFailure failure = CompositePcgFailure::None;
  std::size_t iterations = 0U;
  double rhs_l2 = 0.0;
  double rhs_linf = 0.0;
  double stopping_linf = 0.0;
  CompositeResidualNorms initial_residual;
  CompositeResidualNorms final_residual;
};

struct CompositePcgResult {
  std::vector<double> solution;
  CompositePcgReport report;

  // Callers must not publish an unconverged iterate.  Keeping this check on
  // the result object makes the fail-closed contract explicit at integration
  // sites.
  // Algebraic convergence is necessary but is not a physics-qualification
  // statement.  Keep the API name explicit so callers cannot accidentally
  // promote a residual-only solve into a Mode-3 scientific certificate.
  bool solver_converged() const noexcept { return report.converged; }
};

class CompositePoissonR2 {
 public:
  explicit CompositePoissonR2(const CompositePatchGeometryR2& geometry);

  const CompositePatchGeometryR2& geometry() const noexcept {
    return geometry_;
  }
  const std::vector<CompositeLeafCell>& leaves() const noexcept {
    return leaves_;
  }
  const std::vector<CompositeFace>& faces() const noexcept { return faces_; }
  const std::vector<std::size_t>& dof_to_leaf() const noexcept {
    return dof_to_leaf_;
  }
  const std::vector<double>& diagonal() const noexcept { return diagonal_; }

  std::size_t leaf_count() const noexcept { return leaves_.size(); }
  std::size_t unknown_count() const noexcept { return dof_to_leaf_.size(); }
  std::size_t fine_n_x() const noexcept { return fine_n_x_; }
  std::size_t fine_n_y() const noexcept { return fine_n_y_; }
  std::size_t interface_face_count() const noexcept {
    return interface_face_count_;
  }
  bool has_complete_interface_flux_audit_stencil() const noexcept {
    return interface_flux_audit_available_;
  }

  std::size_t coarse_leaf(std::size_t i, std::size_t j) const;
  std::size_t fine_leaf(std::size_t i, std::size_t j) const;

  // Apply the homogeneous-Dirichlet SPD matrix to unknown potentials.  Fixed
  // boundary values enter only through assemble_integrated_rhs().
  std::vector<double> apply(const std::vector<double>& potential) const;

  // integrated_source is indexed by leaf and stores charge/source integrated
  // over that leaf control volume.  source_scale supplies the single physical
  // Poisson normalization.  Sources on prescribed boundary leaves are
  // rejected rather than silently discarded.
  std::vector<double> assemble_integrated_rhs(
      const std::vector<double>& integrated_source,
      double source_scale,
      const std::vector<double>& dirichlet_leaf_values) const;

  std::vector<double> residual(
      const std::vector<double>& rhs,
      const std::vector<double>& potential) const;
  CompositeResidualNorms residual_norms(
      const std::vector<double>& residual) const;

  std::vector<CompositeFaceFlux> compute_face_fluxes(
      const std::vector<double>& potential,
      const std::vector<double>& dirichlet_leaf_values) const;

  std::vector<CompositeInterfaceFluxConsistency>
  compute_interface_flux_consistency(
      const std::vector<double>& potential,
      const std::vector<double>& dirichlet_leaf_values) const;

  CompositeGaussReport compute_gauss_report(
      const std::vector<double>& potential,
      const std::vector<double>& dirichlet_leaf_values,
      const std::vector<double>& integrated_source,
      double source_scale) const;

  CompositePcgResult solve_pcg(
      const std::vector<double>& rhs,
      const std::vector<double>& initial_potential,
      const CompositePcgOptions& options) const;

 private:
  CompositePatchGeometryR2 geometry_;
  std::size_t fine_n_x_ = 0U;
  std::size_t fine_n_y_ = 0U;
  std::size_t interface_face_count_ = 0U;
  bool interface_flux_audit_available_ = false;
  std::vector<CompositeLeafCell> leaves_;
  std::vector<CompositeFace> faces_;
  std::vector<std::size_t> coarse_leaf_map_;
  std::vector<std::size_t> fine_leaf_map_;
  std::vector<std::size_t> dof_to_leaf_;
  std::vector<double> diagonal_;

  void build_cells();
  void build_faces();
  void validate_topology_and_build_diagonal();
  void add_face(std::size_t leaf_a, std::size_t leaf_b,
                CompositeFaceAxis axis, CompositeFaceKind kind,
                double area, double center_distance);
  void add_interface_face(std::size_t leaf_a, std::size_t leaf_b,
                          std::size_t coarse_tangential_neighbor,
                          double coarse_primary_sign,
                          CompositeFaceAxis axis,
                          double area, double center_distance);
  double face_jump(
      const CompositeFace& face,
      const std::vector<double>& potential,
      const std::vector<double>& dirichlet_leaf_values) const;
  double leaf_potential(
      std::size_t leaf,
      const std::vector<double>& potential,
      const std::vector<double>& dirichlet_leaf_values) const;
  void validate_potential(const std::vector<double>& potential) const;
  void validate_dirichlet(
      const std::vector<double>& dirichlet_leaf_values) const;
};

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_COMPOSITE_POISSON_CPP_H
