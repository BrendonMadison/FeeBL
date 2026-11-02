#include "adaptiveCompositePoissonCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using feebl::adaptive_grid::CompositeFaceKind;
using feebl::adaptive_grid::CompositeFaceAxis;
using feebl::adaptive_grid::CompositeGaussReport;
using feebl::adaptive_grid::CompositeInterfaceFluxConsistency;
using feebl::adaptive_grid::CompositeLeafCell;
using feebl::adaptive_grid::CompositeLevel;
using feebl::adaptive_grid::CompositePatchGeometryR2;
using feebl::adaptive_grid::CompositePcgFailure;
using feebl::adaptive_grid::CompositePcgOptions;
using feebl::adaptive_grid::CompositePcgResult;
using feebl::adaptive_grid::CompositePoissonR2;
using feebl::adaptive_grid::kCompositeNoIndex;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance,
                   const char* message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance)
    throw std::runtime_error(message);
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size() == b.size(), "test dot-product size mismatch");
  double sum = 0.0;
  for (std::size_t i = 0U; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

double maximum_absolute(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::fabs(value));
  return result;
}

CompositePatchGeometryR2 reference_geometry() {
  CompositePatchGeometryR2 geometry;
  geometry.coarse_n_x = 12U;
  geometry.coarse_n_y = 10U;
  geometry.coarse_delta_x = 0.75;
  geometry.coarse_delta_y = 1.25;
  geometry.physical_origin_x = -2.0;
  geometry.physical_origin_y = 3.0;
  geometry.patch_begin_x = 4U;
  geometry.patch_end_x = 8U;
  geometry.patch_begin_y = 3U;
  geometry.patch_end_y = 7U;
  geometry.fixed_boundary_layers = 1U;
  geometry.proper_nesting_cells = 1U;
  geometry.refinement_ratio = 2U;
  return geometry;
}

void test_geometry_leaf_map_and_proper_nesting() {
  const CompositePatchGeometryR2 geometry = reference_geometry();
  const CompositePoissonR2 system(geometry);
  require(system.has_complete_interface_flux_audit_stencil(),
          "reference hierarchy lacks complete one-sided audit stencil");
  require(system.fine_n_x() == 8U && system.fine_n_y() == 8U,
          "ratio-two fine extent is wrong");
  require(system.leaf_count() == 168U,
          "covered coarse cells were not replaced by fine leaves");
  require(system.unknown_count() == 128U,
          "Dirichlet boundary ring has the wrong unknown count");
  require(system.interface_face_count() == 32U,
          "ratio-two interface subface count is wrong");

  std::vector<unsigned char> seen(system.unknown_count(), 0U);
  std::size_t coarse_leaves = 0U;
  std::size_t fine_leaves = 0U;
  std::size_t fixed_leaves = 0U;
  double area = 0.0;
  for (std::size_t leaf_index = 0U;
       leaf_index < system.leaves().size(); ++leaf_index) {
    const CompositeLeafCell& leaf = system.leaves()[leaf_index];
    area += leaf.area;
    if (leaf.level == CompositeLevel::Coarse) ++coarse_leaves;
    else ++fine_leaves;
    if (leaf.fixed_dirichlet) {
      ++fixed_leaves;
      require(leaf.dof == kCompositeNoIndex,
              "fixed leaf unexpectedly owns a DOF");
    } else {
      require(leaf.dof < seen.size() && !seen[leaf.dof],
              "duplicate leaf DOF detected");
      seen[leaf.dof] = 1U;
      require(system.dof_to_leaf()[leaf.dof] == leaf_index,
              "DOF-to-leaf inverse map is wrong");
    }
  }
  require(coarse_leaves == 104U && fine_leaves == 64U,
          "leaf level populations are wrong");
  require(fixed_leaves == 40U,
          "fixed coarse boundary population is wrong");
  require(std::find(seen.begin(), seen.end(), 0U) == seen.end(),
          "leaf DOF map has a hole");
  require_close(area,
                static_cast<double>(geometry.coarse_n_x) *
                    static_cast<double>(geometry.coarse_n_y) *
                    geometry.coarse_delta_x * geometry.coarse_delta_y,
                2.0e-13,
                "composite leaves do not tile the coarse physical area");

  for (std::size_t i = geometry.patch_begin_x;
       i < geometry.patch_end_x; ++i)
    for (std::size_t j = geometry.patch_begin_y;
         j < geometry.patch_end_y; ++j)
      require(system.coarse_leaf(i, j) == kCompositeNoIndex,
              "covered coarse leaf remains in composite topology");
  for (std::size_t i = 0U; i < system.fine_n_x(); ++i)
    for (std::size_t j = 0U; j < system.fine_n_y(); ++j)
      require(system.fine_leaf(i, j) != kCompositeNoIndex,
              "fine patch has a missing leaf");

  std::set<std::pair<std::size_t, std::size_t> > unique_faces;
  for (const auto& face : system.faces()) {
    const std::pair<std::size_t, std::size_t> key =
        std::minmax(face.leaf_a, face.leaf_b);
    require(unique_faces.insert(key).second,
            "face list contains a duplicate leaf pair");
  }

  bool rejected = false;
  try {
    CompositePatchGeometryR2 invalid = geometry;
    invalid.refinement_ratio = 3U;
    (void)CompositePoissonR2(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "non-ratio-two geometry was accepted");

  rejected = false;
  try {
    CompositePatchGeometryR2 invalid = geometry;
    invalid.patch_begin_x = 1U;
    (void)CompositePoissonR2(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "improperly nested patch was accepted");
}

void test_symmetric_spd_operator_and_two_sided_flux() {
  const CompositePoissonR2 system(reference_geometry());
  std::vector<double> x(system.unknown_count(), 0.0);
  std::vector<double> y(system.unknown_count(), 0.0);
  for (std::size_t i = 0U; i < x.size(); ++i) {
    x[i] = std::sin(0.37 * static_cast<double>(i + 1U));
    y[i] = std::cos(0.19 * static_cast<double>(i + 2U));
  }
  const std::vector<double> ax = system.apply(x);
  const std::vector<double> ay = system.apply(y);
  const double xtay = dot(x, ay);
  const double ytax = dot(y, ax);
  const double symmetry_scale = std::max(
      1.0, std::max(std::fabs(xtay), std::fabs(ytax)));
  require(std::fabs(xtay - ytax) <= 2.0e-13 * symmetry_scale,
          "composite operator is not symmetric to roundoff");
  require(dot(x, ax) > 0.0, "composite operator is not positive definite");

  // A constant potential with the same prescribed boundary value must have
  // zero physical flux, even though the homogeneous matrix alone has a
  // boundary diagonal.
  std::vector<double> constant(system.unknown_count(), 1.25);
  std::vector<double> boundary(system.leaf_count(), 1.25);
  std::vector<double> no_source(system.leaf_count(), 0.0);
  const std::vector<double> constant_rhs =
      system.assemble_integrated_rhs(no_source, 1.0, boundary);
  require(maximum_absolute(system.residual(constant_rhs, constant)) < 1.0e-14,
          "constant Dirichlet solution has nonzero residual");

  std::vector<double> potential(system.unknown_count(), 0.0);
  for (std::size_t dof = 0U; dof < potential.size(); ++dof) {
    const CompositeLeafCell& leaf =
        system.leaves()[system.dof_to_leaf()[dof]];
    potential[dof] = leaf.center_x + 2.0 * leaf.center_y;
  }
  for (std::size_t leaf = 0U; leaf < system.leaf_count(); ++leaf)
    boundary[leaf] = system.leaves()[leaf].center_x +
                     2.0 * system.leaves()[leaf].center_y;
  const auto fluxes = system.compute_face_fluxes(potential, boundary);
  require(fluxes.size() == system.faces().size(),
          "face-flux report has wrong size");
  std::size_t interface_faces = 0U;
  for (std::size_t i = 0U; i < fluxes.size(); ++i) {
    require(fluxes[i].outward_from_a == -fluxes[i].outward_from_b,
            "one face does not make equal/opposite physical accounting");
    const double algebraic_sum =
        fluxes[i].algebraic_outward_from_a +
        fluxes[i].algebraic_outward_from_b +
        fluxes[i].algebraic_outward_from_c;
    require(std::fabs(algebraic_sum) <=
                2.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::fabs(fluxes[i].flux_a_to_b)),
            "variational face contribution is not conservative");
    if (system.faces()[i].kind == CompositeFaceKind::CoarseFine) {
      ++interface_faces;
      require(system.faces()[i].transmissibility > 0.0,
              "coarse/fine face has non-positive transmissibility");
    }
  }
  require(interface_faces == system.interface_face_count(),
          "coarse/fine flux population is incomplete");
}

template <typename Function>
void polynomial_leaf_values(const CompositePoissonR2& system,
                            const Function& function,
                            std::vector<double>* potential,
                            std::vector<double>* boundary) {
  require(potential != nullptr && boundary != nullptr,
          "polynomial output pointer is null");
  potential->assign(system.unknown_count(), 0.0);
  boundary->assign(system.leaf_count(), 0.0);
  for (std::size_t leaf_index = 0U;
       leaf_index < system.leaf_count(); ++leaf_index) {
    const CompositeLeafCell& leaf = system.leaves()[leaf_index];
    const double value = function(leaf.center_x, leaf.center_y);
    (*boundary)[leaf_index] = value;
    if (!leaf.fixed_dirichlet) (*potential)[leaf.dof] = value;
  }
}

void test_multipoint_interface_linear_and_quadratic_flux() {
  const CompositePoissonR2 system(reference_geometry());
  std::size_t interface_faces = 0U;
  for (const auto& face : system.faces()) {
    if (face.kind != CompositeFaceKind::CoarseFine) continue;
    ++interface_faces;
    require(face.leaf_c != kCompositeNoIndex,
            "multipoint interface has no tangential coarse neighbour");
    require(system.leaves()[face.leaf_c].level == CompositeLevel::Coarse,
            "multipoint neighbour is not coarse");
    require_close(face.coefficient_a + face.coefficient_b +
                      face.coefficient_c,
                  0.0, 0.0,
                  "multipoint interface coefficients do not conserve");
    const bool coarse_is_a =
        system.leaves()[face.leaf_a].level == CompositeLevel::Coarse;
    const double primary = coarse_is_a
        ? face.coefficient_a : face.coefficient_b;
    require_close(std::fabs(primary), 0.75, 0.0,
                  "coarse primary interpolation weight is wrong");
    require_close(std::fabs(face.coefficient_c), 0.25, 0.0,
                  "coarse neighbour interpolation weight is wrong");
  }
  require(interface_faces == system.interface_face_count(),
          "multipoint coefficient audit omitted an interface");

  std::vector<double> potential;
  std::vector<double> boundary;
  const double linear_x = 0.7;
  const double linear_y = -1.1;
  polynomial_leaf_values(
      system,
      [linear_x, linear_y](double x, double y) {
        return 0.4 + linear_x * x + linear_y * y;
      },
      &potential, &boundary);
  const auto linear_fluxes = system.compute_face_fluxes(
      potential, boundary);
  const auto linear_audit = system.compute_interface_flux_consistency(
      potential, boundary);
  require(linear_audit.size() == interface_faces,
          "linear one-sided audit has wrong interface count");
  for (const CompositeInterfaceFluxConsistency& audit : linear_audit) {
    const auto& face = system.faces()[audit.face];
    const double gradient = face.axis == CompositeFaceAxis::XNormal
        ? linear_x : linear_y;
    const double expected = -face.area * gradient;
    require_close(audit.variational_flux_a_to_b, expected, 3.0e-15,
                  "linear variational interface flux is inconsistent");
    require_close(audit.coarse_one_sided_flux_a_to_b, expected, 1.0e-14,
                  "linear coarse one-sided interface flux is wrong");
    require_close(audit.fine_one_sided_flux_a_to_b, expected, 1.0e-14,
                  "linear fine one-sided interface flux is wrong");
    require(audit.absolute_one_sided_defect < 1.0e-14,
            "linear one-sided interface fluxes disagree");
    require_close(linear_fluxes[audit.face].outward_from_a +
                      linear_fluxes[audit.face].outward_from_b,
                  0.0, 0.0,
                  "structural two-sided interface accounting changed");
  }

  // This piecewise selector is a particularly sensitive regression: the
  // continuum normal derivative is exactly zero on every tested face.  The
  // old centre-to-centre TPFA produced a spurious |gradient|=1/3 on each
  // fine subface and only cancelled after summing a coarse parent face.
  polynomial_leaf_values(
      system,
      [](double x, double y) { return x + y; },
      &potential, &boundary);
  const auto tangential_fluxes = system.compute_face_fluxes(
      potential, boundary);
  for (std::size_t face_index = 0U;
       face_index < system.faces().size(); ++face_index) {
    const auto& face = system.faces()[face_index];
    if (face.kind != CompositeFaceKind::CoarseFine) continue;
    // Subtract the known normal component.  The remainder must vanish on
    // every subface, rather than merely cancelling between siblings.
    const double expected = -face.area;
    require_close(tangential_fluxes[face_index].flux_a_to_b,
                  expected, 3.0e-15,
                  "linear tangential reconstruction aliases normal flux");
  }

  const double q_xx = 0.6;
  const double q_xy = -0.35;
  const double q_yy = 0.9;
  const double q_x = -0.2;
  const double q_y = 0.45;
  polynomial_leaf_values(
      system,
      [q_xx, q_xy, q_yy, q_x, q_y](double x, double y) {
        return q_xx * x * x + q_xy * x * y + q_yy * y * y +
               q_x * x + q_y * y + 0.8;
      },
      &potential, &boundary);
  const auto quadratic_audit = system.compute_interface_flux_consistency(
      potential, boundary);
  double maximum_quadratic_one_sided_defect = 0.0;
  for (const CompositeInterfaceFluxConsistency& audit : quadratic_audit) {
    const auto& face = system.faces()[audit.face];
    const CompositeLeafCell& fine =
        system.leaves()[system.leaves()[face.leaf_a].level ==
                                CompositeLevel::Fine
                            ? face.leaf_a : face.leaf_b];
    const CompositeLeafCell& coarse =
        system.leaves()[system.leaves()[face.leaf_a].level ==
                                CompositeLevel::Coarse
                            ? face.leaf_a : face.leaf_b];
    const double interface_x = face.axis == CompositeFaceAxis::XNormal
        ? (coarse.center_x + 2.0 * fine.center_x) / 3.0
        : fine.center_x;
    const double interface_y = face.axis == CompositeFaceAxis::YNormal
        ? (coarse.center_y + 2.0 * fine.center_y) / 3.0
        : fine.center_y;
    const double derivative = face.axis == CompositeFaceAxis::XNormal
        ? 2.0 * q_xx * interface_x + q_xy * interface_y + q_x
        : q_xy * interface_x + 2.0 * q_yy * interface_y + q_y;
    const double expected = -face.area * derivative;
    require_close(audit.coarse_one_sided_flux_a_to_b, expected, 8.0e-14,
                  "quadratic coarse one-sided flux is not exact");
    require_close(audit.fine_one_sided_flux_a_to_b, expected, 8.0e-14,
                  "quadratic fine one-sided flux is not exact");
    maximum_quadratic_one_sided_defect = std::max(
        maximum_quadratic_one_sided_defect,
        audit.absolute_one_sided_defect);
  }
  require(maximum_quadratic_one_sided_defect < 8.0e-14,
          "quadratic independent one-sided flux consistency failed");
  std::cout << "multipoint_interface quadratic_one_sided_max_defect="
            << maximum_quadratic_one_sided_defect << '\n';
}

struct AlgebraicCase {
  CompositePoissonR2 system;
  std::vector<double> exact;
  std::vector<double> boundary;
  std::vector<double> integrated_source;
  std::vector<double> rhs;

  explicit AlgebraicCase(const CompositePatchGeometryR2& geometry)
      : system(geometry) {}
};

AlgebraicCase make_algebraic_case() {
  AlgebraicCase result(reference_geometry());
  result.exact.assign(result.system.unknown_count(), 0.0);
  result.boundary.assign(result.system.leaf_count(), 0.0);
  result.integrated_source.assign(result.system.leaf_count(), 0.0);
  for (std::size_t leaf_index = 0U;
       leaf_index < result.system.leaf_count(); ++leaf_index) {
    const CompositeLeafCell& leaf = result.system.leaves()[leaf_index];
    const double value = 0.35 + 0.07 * leaf.center_x -
                         0.025 * leaf.center_y +
                         std::sin(0.11 * leaf.center_x * leaf.center_y);
    result.boundary[leaf_index] = value;
    if (!leaf.fixed_dirichlet) result.exact[leaf.dof] = value;
  }
  const std::vector<double> applied = result.system.apply(result.exact);
  const std::vector<double> zero_source(result.system.leaf_count(), 0.0);
  const std::vector<double> boundary_rhs =
      result.system.assemble_integrated_rhs(
          zero_source, 1.0, result.boundary);
  for (std::size_t dof = 0U; dof < result.system.unknown_count(); ++dof) {
    const std::size_t leaf = result.system.dof_to_leaf()[dof];
    result.integrated_source[leaf] = applied[dof] - boundary_rhs[dof];
  }
  result.rhs = result.system.assemble_integrated_rhs(
      result.integrated_source, 1.0, result.boundary);
  for (std::size_t dof = 0U; dof < result.system.unknown_count(); ++dof)
    require_close(result.rhs[dof], applied[dof], 2.0e-15,
                  "manufactured integrated RHS does not equal A*phi");
  return result;
}

void test_algebraic_manufactured_solution_pcg_and_gauss() {
  AlgebraicCase problem = make_algebraic_case();
  CompositePcgOptions options;
  options.relative_tolerance = 2.0e-12;
  options.absolute_tolerance = 2.0e-13;
  options.maximum_iterations = 50000U;
  options.residual_recompute_interval = 8U;
  const std::vector<double> zero(problem.system.unknown_count(), 0.0);
  const CompositePcgResult first = problem.system.solve_pcg(
      problem.rhs, zero, options);
  const CompositePcgResult replay = problem.system.solve_pcg(
      problem.rhs, zero, options);
  require(first.solver_converged(), "PCG failed algebraic manufactured solution");
  require(replay.solver_converged(), "PCG deterministic replay did not converge");
  require(first.report.iterations == replay.report.iterations &&
              first.solution == replay.solution,
          "serial binary64 PCG replay changed its result");
  require(first.report.final_residual.linf <= first.report.stopping_linf,
          "PCG published a solution outside mixed residual tolerance");
  double maximum_error = 0.0;
  for (std::size_t i = 0U; i < first.solution.size(); ++i)
    maximum_error = std::max(
        maximum_error, std::fabs(first.solution[i] - problem.exact[i]));
  require(maximum_error < 2.0e-9,
          "PCG did not recover algebraic manufactured potential");

  const CompositeGaussReport gauss = problem.system.compute_gauss_report(
      problem.exact, problem.boundary, problem.integrated_source, 1.0);
  require(gauss.interface_faces == problem.system.interface_face_count(),
          "Gauss report omitted interface subfaces");
  require(gauss.maximum_two_sided_flux_defect == 0.0 &&
              gauss.maximum_interface_two_sided_flux_defect == 0.0,
          "stored face flux is not exactly two-sided");
  require(gauss.maximum_cell_balance_defect < 3.0e-14,
          "manufactured solution violates leaf Gauss balance");
  require(std::fabs(gauss.global_balance_defect) < 2.0e-13,
          "manufactured solution violates global Gauss balance");

  CompositePcgOptions one_step = options;
  one_step.maximum_iterations = 1U;
  const CompositePcgResult rejected = problem.system.solve_pcg(
      problem.rhs, zero, one_step);
  require(!rejected.solver_converged() &&
              rejected.report.failure ==
                  CompositePcgFailure::MaximumIterations,
          "unconverged PCG iterate did not fail closed");
}

double continuum_error(std::size_t n) {
  require(n >= 12U && n % 4U == 0U,
          "continuum test needs n divisible by four");
  CompositePatchGeometryR2 geometry;
  geometry.coarse_n_x = n;
  geometry.coarse_n_y = n;
  geometry.coarse_delta_x = 1.0 / static_cast<double>(n);
  geometry.coarse_delta_y = geometry.coarse_delta_x;
  geometry.patch_begin_x = n / 4U;
  geometry.patch_end_x = 3U * n / 4U;
  geometry.patch_begin_y = n / 4U;
  geometry.patch_end_y = 3U * n / 4U;
  geometry.fixed_boundary_layers = 1U;
  geometry.proper_nesting_cells = 1U;
  const CompositePoissonR2 system(geometry);

  const double pi = std::acos(-1.0);
  const double left = 0.5 * geometry.coarse_delta_x;
  const double right = 1.0 - 0.5 * geometry.coarse_delta_x;
  const double length = right - left;
  const double eigenvalue = 2.0 * pi * pi / (length * length);
  const auto exact_value = [left, length, pi](double x, double y) {
    return std::sin(pi * (x - left) / length) *
           std::sin(pi * (y - left) / length);
  };

  std::vector<double> boundary(system.leaf_count(), 0.0);
  std::vector<double> integrated_source(system.leaf_count(), 0.0);
  for (std::size_t leaf_index = 0U;
       leaf_index < system.leaf_count(); ++leaf_index) {
    const CompositeLeafCell& leaf = system.leaves()[leaf_index];
    const double exact = exact_value(leaf.center_x, leaf.center_y);
    boundary[leaf_index] = exact;
    if (!leaf.fixed_dirichlet)
      integrated_source[leaf_index] = eigenvalue * exact * leaf.area;
  }
  const std::vector<double> rhs = system.assemble_integrated_rhs(
      integrated_source, 1.0, boundary);
  CompositePcgOptions options;
  options.relative_tolerance = 2.0e-11;
  options.absolute_tolerance = 1.0e-13;
  options.maximum_iterations = 200000U;
  options.residual_recompute_interval = 16U;
  const CompositePcgResult solved = system.solve_pcg(
      rhs, std::vector<double>(), options);
  require(solved.solver_converged(),
          "PCG failed continuum convergence scaffold");

  double error_squared = 0.0;
  double exact_squared = 0.0;
  for (std::size_t leaf_index = 0U;
       leaf_index < system.leaf_count(); ++leaf_index) {
    const CompositeLeafCell& leaf = system.leaves()[leaf_index];
    const double exact = exact_value(leaf.center_x, leaf.center_y);
    const double computed = leaf.fixed_dirichlet
        ? boundary[leaf_index]
        : solved.solution[leaf.dof];
    const double difference = computed - exact;
    error_squared += leaf.area * difference * difference;
    exact_squared += leaf.area * exact * exact;
  }
  require(error_squared >= 0.0 && exact_squared > 0.0 &&
              std::isfinite(error_squared) && std::isfinite(exact_squared),
          "invalid continuum error norm");
  const double error = std::sqrt(error_squared / exact_squared);
  std::cout << "continuum_r2 n=" << n
            << " unknowns=" << system.unknown_count()
            << " iterations=" << solved.report.iterations
            << " relative_L2=" << error << '\n';
  return error;
}

void test_continuum_convergence_scaffold() {
  const double e12 = continuum_error(12U);
  const double e24 = continuum_error(24U);
  const double e48 = continuum_error(48U);
  require(e24 < e12 && e48 < e24,
          "continuum composite error does not contract");
  const double p1 = std::log(e12 / e24) / std::log(2.0);
  const double p2 = std::log(e24 / e48) / std::log(2.0);
  std::cout << "continuum_r2 observed_orders=" << p1 << ',' << p2 << '\n';
  // TPFA is only the first auditable interface baseline.  This modest gate
  // proves contraction without disguising a possible first-order interface;
  // production qualification must impose the stricter uniform-fine,
  // self-force, and interface-translation gates described by the audit.
  require(p1 > 0.35 && p2 > 0.35,
          "continuum composite convergence order is unacceptably weak");
}

void test_fail_closed_inputs() {
  const CompositePoissonR2 system(reference_geometry());
  std::vector<double> boundary(system.leaf_count(), 0.0);
  std::vector<double> source(system.leaf_count(), 0.0);

  std::size_t fixed_leaf = kCompositeNoIndex;
  for (std::size_t leaf = 0U; leaf < system.leaf_count(); ++leaf)
    if (system.leaves()[leaf].fixed_dirichlet) {
      fixed_leaf = leaf;
      break;
    }
  require(fixed_leaf != kCompositeNoIndex,
          "test topology has no fixed boundary leaf");
  source[fixed_leaf] = 1.0;
  bool rejected = false;
  try {
    (void)system.assemble_integrated_rhs(source, 1.0, boundary);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "source on fixed boundary was silently discarded");

  rejected = false;
  try {
    std::vector<double> invalid(system.unknown_count(), 0.0);
    invalid.front() = std::numeric_limits<double>::quiet_NaN();
    (void)system.apply(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "non-finite composite potential was accepted");

  rejected = false;
  try {
    CompositePcgOptions invalid;
    invalid.relative_tolerance = 0.0;
    invalid.absolute_tolerance = 0.0;
    (void)system.solve_pcg(
        std::vector<double>(system.unknown_count(), 0.0),
        std::vector<double>(), invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "zero-tolerance PCG policy was accepted");

  rejected = false;
  try {
    CompositePatchGeometryR2 insufficient = reference_geometry();
    insufficient.patch_begin_x = 2U;
    const CompositePoissonR2 unauditable(insufficient);
    require(!unauditable.has_complete_interface_flux_audit_stencil(),
            "insufficient hierarchy advertised a complete audit stencil");
    (void)unauditable.compute_interface_flux_consistency(
        std::vector<double>(unauditable.unknown_count(), 0.0),
        std::vector<double>(unauditable.leaf_count(), 0.0));
  } catch (const std::logic_error&) {
    rejected = true;
  }
  require(rejected,
          "incomplete one-sided interface audit did not fail closed");

  rejected = false;
  try {
    CompositePatchGeometryR2 too_narrow = reference_geometry();
    too_narrow.patch_end_x = too_narrow.patch_begin_x + 1U;
    const CompositePoissonR2 unauditable(too_narrow);
    (void)unauditable.compute_interface_flux_consistency(
        std::vector<double>(unauditable.unknown_count(), 0.0),
        std::vector<double>(unauditable.leaf_count(), 0.0));
  } catch (const std::logic_error&) {
    rejected = true;
  }
  require(rejected,
          "narrow-patch one-sided audit did not fail closed");
}

}  // namespace

int main() {
  try {
    test_geometry_leaf_map_and_proper_nesting();
    test_symmetric_spd_operator_and_two_sided_flux();
    test_multipoint_interface_linear_and_quadratic_flux();
    test_algebraic_manufactured_solution_pcg_and_gauss();
    test_continuum_convergence_scaffold();
    test_fail_closed_inputs();
    std::cout << "adaptive composite Poisson tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "adaptive composite Poisson test failure: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
