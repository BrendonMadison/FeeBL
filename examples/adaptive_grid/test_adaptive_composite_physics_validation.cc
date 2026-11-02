#include "adaptiveCompositeCouplingCPP.h"
#include "adaptiveCompositePoissonCPP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Standalone, deliberately independent physics-validation harness for the
// ratio-two composite prototype.  CTest runs the diagnostic form, while the
// release audit invokes --require-stage4 as a hard scientific gate. Invariants
// and fail-closed behavior are hard assertions. Scientific Stage-4 metrics
// are evaluated against fixed limits and reported without disguising a failed
// gate as a passing unit test.  --require-stage4 returns 2 unless all of the
// standalone scientific gates pass.

namespace {

using feebl::adaptive_grid::CompositeAdjointGatherResultR2;
using feebl::adaptive_grid::CompositeDepositResultR2;
using feebl::adaptive_grid::CompositeFaceAxis;
using feebl::adaptive_grid::CompositeFaceKind;
using feebl::adaptive_grid::CompositeGaussReport;
using feebl::adaptive_grid::CompositeLeafCell;
using feebl::adaptive_grid::CompositeLevel;
using feebl::adaptive_grid::CompositePatchGeometryR2;
using feebl::adaptive_grid::CompositePcgOptions;
using feebl::adaptive_grid::CompositePcgResult;
using feebl::adaptive_grid::CompositePoissonR2;
using feebl::adaptive_grid::CompositeTentParticleR2;
using feebl::adaptive_grid::CompositeTentStatus;
using feebl::adaptive_grid::CompositeTentStencilSetR2;
using feebl::adaptive_grid::build_composite_tent_stencils_r2;
using feebl::adaptive_grid::deposit_composite_tent_source_once_r2;
using feebl::adaptive_grid::gather_composite_tent_adjoint_r2;
using feebl::adaptive_grid::kCompositeNoIndex;

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function function, const std::string& message) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, message);
}

bool finite(double value) { return std::isfinite(value); }

double square(double value) { return value * value; }

double vector_l2(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) sum += value * value;
  return std::sqrt(sum);
}

double vector_linf(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) result = std::max(result, std::fabs(value));
  return result;
}

bool bitwise_equal(const std::vector<double>& a,
                   const std::vector<double>& b) {
  return a.size() == b.size() &&
      (a.empty() || std::memcmp(a.data(), b.data(),
                                a.size() * sizeof(double)) == 0);
}

struct ErrorNorms {
  double absolute_rms = 0.0;
  double absolute_linf = 0.0;
  double relative_l2 = 0.0;
  double mixed_linf = 0.0;
};

ErrorNorms error_norms(const std::vector<double>& candidate,
                       const std::vector<double>& reference,
                       double absolute_floor) {
  require(candidate.size() == reference.size() && !candidate.empty(),
          "error-norm vectors must be nonempty and equal-sized");
  require(finite(absolute_floor) && absolute_floor > 0.0,
          "error-norm absolute floor must be positive");
  double error_sum = 0.0;
  double reference_sum = 0.0;
  ErrorNorms result;
  for (std::size_t i = 0U; i < candidate.size(); ++i) {
    require(finite(candidate[i]) && finite(reference[i]),
            "non-finite value in error norm");
    const double difference = candidate[i] - reference[i];
    error_sum += difference * difference;
    reference_sum += reference[i] * reference[i];
    result.absolute_linf = std::max(result.absolute_linf,
                                    std::fabs(difference));
    result.mixed_linf = std::max(
        result.mixed_linf,
        std::fabs(difference) /
            std::max(absolute_floor, std::fabs(reference[i])));
  }
  result.absolute_rms = std::sqrt(
      error_sum / static_cast<double>(candidate.size()));
  result.relative_l2 = std::sqrt(error_sum) /
      std::max(std::sqrt(reference_sum), absolute_floor);
  return result;
}

struct Integral1D {
  double weight = 0.0;
  double derivative = 0.0;
};

double tent_primitive(double u, double half_width) {
  if (u <= -half_width) return 0.0;
  if (u < 0.0) {
    const double shifted = u + half_width;
    return 0.5 * shifted * shifted /
        (half_width * half_width);
  }
  if (u < half_width)
    return 0.5 + u / half_width -
        0.5 * u * u / (half_width * half_width);
  return 1.0;
}

double tent_value(double u, double half_width) {
  const double distance = std::fabs(u);
  if (!(distance < half_width)) return 0.0;
  return (1.0 - distance / half_width) / half_width;
}

Integral1D integrate_tent(double begin, double end, double position,
                          double half_width) {
  Integral1D result;
  result.weight = tent_primitive(end - position, half_width) -
                  tent_primitive(begin - position, half_width);
  result.derivative = tent_value(begin - position, half_width) -
                      tent_value(end - position, half_width);
  if (result.weight < 0.0 && result.weight > -2.0e-15)
    result.weight = 0.0;
  return result;
}

struct UniformSample {
  double potential = 0.0;
  double electric_x = 0.0;
  double electric_y = 0.0;
};

struct UniformSolveReport {
  bool converged = false;
  std::size_t iterations = 0U;
  double residual_l2 = 0.0;
  double residual_linf = 0.0;
  double gauss_defect = 0.0;
};

class UniformFineOracle {
 public:
  UniformFineOracle(std::size_t n_x, std::size_t n_y, double origin_x,
                    double origin_y, double delta_x, double delta_y,
                    std::size_t fixed_layers)
      : n_x_(n_x), n_y_(n_y), origin_x_(origin_x), origin_y_(origin_y),
        delta_x_(delta_x), delta_y_(delta_y),
        fixed_layers_(fixed_layers), map_(n_x * n_y, kCompositeNoIndex) {
    require(n_x_ >= 5U && n_y_ >= 5U && fixed_layers_ > 0U &&
                2U * fixed_layers_ < n_x_ &&
                2U * fixed_layers_ < n_y_ &&
                finite(origin_x_) && finite(origin_y_) &&
                finite(delta_x_) && finite(delta_y_) &&
                delta_x_ > 0.0 && delta_y_ > 0.0,
            "invalid uniform-fine oracle geometry");
    for (std::size_t i = 0U; i < n_x_; ++i)
      for (std::size_t j = 0U; j < n_y_; ++j) {
        if (!fixed(i, j)) {
          map_[cell(i, j)] = dof_i_.size();
          dof_i_.push_back(i);
          dof_j_.push_back(j);
        }
      }
    require(!dof_i_.empty(), "uniform-fine oracle has no unknowns");
  }

  std::size_t n_x() const noexcept { return n_x_; }
  std::size_t n_y() const noexcept { return n_y_; }
  std::size_t cell_count() const noexcept { return n_x_ * n_y_; }
  std::size_t unknown_count() const noexcept { return dof_i_.size(); }
  double delta_x() const noexcept { return delta_x_; }
  double delta_y() const noexcept { return delta_y_; }
  double origin_x() const noexcept { return origin_x_; }
  double origin_y() const noexcept { return origin_y_; }

  bool fixed(std::size_t i, std::size_t j) const {
    return i < fixed_layers_ || j < fixed_layers_ ||
           i >= n_x_ - fixed_layers_ || j >= n_y_ - fixed_layers_;
  }

  std::size_t cell(std::size_t i, std::size_t j) const {
    require(i < n_x_ && j < n_y_, "uniform-fine cell index out of range");
    return i * n_y_ + j;
  }

  double center_x(std::size_t i) const {
    return origin_x_ + (static_cast<double>(i) + 0.5) * delta_x_;
  }

  double center_y(std::size_t j) const {
    return origin_y_ + (static_cast<double>(j) + 0.5) * delta_y_;
  }

  std::vector<double> apply(const std::vector<double>& potential) const {
    require(potential.size() == unknown_count(),
            "uniform-fine apply size mismatch");
    std::vector<double> result(unknown_count(), 0.0);
    for (std::size_t dof = 0U; dof < unknown_count(); ++dof) {
      const std::size_t i = dof_i_[dof];
      const std::size_t j = dof_j_[dof];
      add_apply_neighbor(i, j, i - 1U, j, delta_y_ / delta_x_,
                         potential, &result[dof]);
      add_apply_neighbor(i, j, i + 1U, j, delta_y_ / delta_x_,
                         potential, &result[dof]);
      add_apply_neighbor(i, j, i, j - 1U, delta_x_ / delta_y_,
                         potential, &result[dof]);
      add_apply_neighbor(i, j, i, j + 1U, delta_x_ / delta_y_,
                         potential, &result[dof]);
    }
    return result;
  }

  std::vector<double> assemble_rhs(
      const std::vector<double>& integrated_source,
      const std::vector<double>& dirichlet) const {
    require(integrated_source.size() == cell_count() &&
                dirichlet.size() == cell_count(),
            "uniform-fine RHS size mismatch");
    std::vector<double> rhs(unknown_count(), 0.0);
    for (std::size_t dof = 0U; dof < unknown_count(); ++dof) {
      const std::size_t i = dof_i_[dof];
      const std::size_t j = dof_j_[dof];
      rhs[dof] = integrated_source[cell(i, j)];
      add_rhs_boundary(i - 1U, j, delta_y_ / delta_x_, dirichlet,
                       &rhs[dof]);
      add_rhs_boundary(i + 1U, j, delta_y_ / delta_x_, dirichlet,
                       &rhs[dof]);
      add_rhs_boundary(i, j - 1U, delta_x_ / delta_y_, dirichlet,
                       &rhs[dof]);
      add_rhs_boundary(i, j + 1U, delta_x_ / delta_y_, dirichlet,
                       &rhs[dof]);
    }
    return rhs;
  }

  std::pair<std::vector<double>, UniformSolveReport> solve(
      const std::vector<double>& integrated_source,
      const std::vector<double>& dirichlet,
      double relative_tolerance = 1.0e-12,
      std::size_t maximum_iterations = 50000U) const {
    require(relative_tolerance > 0.0 && finite(relative_tolerance),
            "invalid uniform-fine solver tolerance");
    const std::vector<double> rhs = assemble_rhs(integrated_source,
                                                  dirichlet);
    std::vector<double> x(unknown_count(), 0.0);
    std::vector<double> r = rhs;
    std::vector<double> z(unknown_count(), 0.0);
    std::vector<double> p(unknown_count(), 0.0);
    const double diagonal = 2.0 * delta_y_ / delta_x_ +
                            2.0 * delta_x_ / delta_y_;
    for (std::size_t i = 0U; i < r.size(); ++i) {
      z[i] = r[i] / diagonal;
      p[i] = z[i];
    }
    double rz = dot(r, z);
    const double rhs_l2 = vector_l2(rhs);
    const double threshold = relative_tolerance * std::max(rhs_l2, 1.0);
    UniformSolveReport report;
    if (vector_l2(r) <= threshold) {
      report.converged = true;
    } else {
      for (std::size_t iteration = 0U;
           iteration < maximum_iterations; ++iteration) {
        const std::vector<double> ap = apply(p);
        const double curvature = dot(p, ap);
        require(finite(curvature) && curvature > 0.0,
                "uniform-fine PCG lost positive curvature");
        const double alpha = rz / curvature;
        for (std::size_t i = 0U; i < x.size(); ++i) {
          x[i] += alpha * p[i];
          r[i] -= alpha * ap[i];
        }
        report.iterations = iteration + 1U;
        if (vector_l2(r) <= threshold) {
          report.converged = true;
          break;
        }
        for (std::size_t i = 0U; i < r.size(); ++i) z[i] = r[i] / diagonal;
        const double next_rz = dot(r, z);
        require(finite(next_rz) && next_rz >= 0.0,
                "uniform-fine PCG preconditioner failure");
        const double beta = next_rz / rz;
        for (std::size_t i = 0U; i < p.size(); ++i)
          p[i] = z[i] + beta * p[i];
        rz = next_rz;
      }
    }
    const std::vector<double> residual = residual_vector(rhs, x);
    report.residual_l2 = vector_l2(residual);
    report.residual_linf = vector_linf(residual);
    report.gauss_defect = gauss_defect(x, integrated_source, dirichlet);
    return std::make_pair(x, report);
  }

  std::vector<double> deposit_tents(
      const std::vector<CompositeTentParticleR2>& particles) const {
    std::vector<double> source(cell_count(), 0.0);
    for (const CompositeTentParticleR2& particle : particles) {
      double sum = 0.0;
      for (std::size_t i = 0U; i < n_x_; ++i)
        for (std::size_t j = 0U; j < n_y_; ++j) {
          const Integral1D x = integrate_tent(
              origin_x_ + static_cast<double>(i) * delta_x_,
              origin_x_ + static_cast<double>(i + 1U) * delta_x_,
              particle.position_x, particle.support_half_width_x);
          if (!(x.weight > 0.0)) continue;
          const Integral1D y = integrate_tent(
              origin_y_ + static_cast<double>(j) * delta_y_,
              origin_y_ + static_cast<double>(j + 1U) * delta_y_,
              particle.position_y, particle.support_half_width_y);
          if (!(y.weight > 0.0)) continue;
          require(!fixed(i, j),
                  "uniform-fine tent source touches fixed boundary");
          const double weight = x.weight * y.weight;
          source[cell(i, j)] += particle.signed_charge * weight;
          sum += weight;
        }
      require(std::fabs(sum - 1.0) <= 3.0e-12,
              "uniform-fine tent deposition lacks partition closure");
    }
    return source;
  }

  UniformSample gather_tent(const CompositeTentParticleR2& probe,
                            const std::vector<double>& potential,
                            const std::vector<double>& dirichlet) const {
    require(potential.size() == unknown_count() &&
                dirichlet.size() == cell_count(),
            "uniform-fine gather size mismatch");
    UniformSample result;
    double sum = 0.0;
    for (std::size_t i = 0U; i < n_x_; ++i)
      for (std::size_t j = 0U; j < n_y_; ++j) {
        const Integral1D x = integrate_tent(
            origin_x_ + static_cast<double>(i) * delta_x_,
            origin_x_ + static_cast<double>(i + 1U) * delta_x_,
            probe.position_x, probe.support_half_width_x);
        if (!(x.weight > 0.0)) continue;
        const Integral1D y = integrate_tent(
            origin_y_ + static_cast<double>(j) * delta_y_,
            origin_y_ + static_cast<double>(j + 1U) * delta_y_,
            probe.position_y, probe.support_half_width_y);
        if (!(y.weight > 0.0)) continue;
        const std::size_t index = cell(i, j);
        const std::size_t dof = map_[index];
        const double phi = dof == kCompositeNoIndex
            ? dirichlet[index] : potential[dof];
        result.potential += x.weight * y.weight * phi;
        result.electric_x -= x.derivative * y.weight * phi;
        result.electric_y -= x.weight * y.derivative * phi;
        sum += x.weight * y.weight;
      }
    require(std::fabs(sum - 1.0) <= 3.0e-12,
            "uniform-fine gather lacks partition closure");
    return result;
  }

  double potential_at_cell(std::size_t i, std::size_t j,
                           const std::vector<double>& potential,
                           const std::vector<double>& dirichlet) const {
    const std::size_t index = cell(i, j);
    const std::size_t dof = map_[index];
    return dof == kCompositeNoIndex ? dirichlet[index] : potential[dof];
  }

  double bilinear_potential(double x, double y,
                            const std::vector<double>& potential,
                            const std::vector<double>& dirichlet) const {
    const double gx = (x - origin_x_) / delta_x_ - 0.5;
    const double gy = (y - origin_y_) / delta_y_ - 0.5;
    require(gx >= 0.0 && gy >= 0.0 &&
                gx <= static_cast<double>(n_x_ - 1U) &&
                gy <= static_cast<double>(n_y_ - 1U),
            "uniform-fine bilinear point outside cell-center domain");
    std::size_t i0 = static_cast<std::size_t>(std::floor(gx));
    std::size_t j0 = static_cast<std::size_t>(std::floor(gy));
    if (i0 + 1U >= n_x_) i0 = n_x_ - 2U;
    if (j0 + 1U >= n_y_) j0 = n_y_ - 2U;
    const double tx = gx - static_cast<double>(i0);
    const double ty = gy - static_cast<double>(j0);
    const double p00 = potential_at_cell(i0, j0, potential, dirichlet);
    const double p10 = potential_at_cell(i0 + 1U, j0, potential, dirichlet);
    const double p01 = potential_at_cell(i0, j0 + 1U, potential, dirichlet);
    const double p11 = potential_at_cell(i0 + 1U, j0 + 1U,
                                         potential, dirichlet);
    return (1.0 - tx) * (1.0 - ty) * p00 +
           tx * (1.0 - ty) * p10 +
           (1.0 - tx) * ty * p01 + tx * ty * p11;
  }

 private:
  std::size_t n_x_;
  std::size_t n_y_;
  double origin_x_;
  double origin_y_;
  double delta_x_;
  double delta_y_;
  std::size_t fixed_layers_;
  std::vector<std::size_t> map_;
  std::vector<std::size_t> dof_i_;
  std::vector<std::size_t> dof_j_;

  static double dot(const std::vector<double>& a,
                    const std::vector<double>& b) {
    require(a.size() == b.size(), "uniform-fine dot size mismatch");
    double result = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) result += a[i] * b[i];
    return result;
  }

  void add_apply_neighbor(std::size_t i, std::size_t j,
                          std::size_t neighbor_i,
                          std::size_t neighbor_j, double transmissibility,
                          const std::vector<double>& potential,
                          double* value) const {
    const std::size_t current_dof = map_[cell(i, j)];
    const std::size_t neighbor_dof = map_[cell(neighbor_i, neighbor_j)];
    *value += transmissibility * potential[current_dof];
    if (neighbor_dof != kCompositeNoIndex)
      *value -= transmissibility * potential[neighbor_dof];
  }

  void add_rhs_boundary(std::size_t i, std::size_t j,
                        double transmissibility,
                        const std::vector<double>& dirichlet,
                        double* value) const {
    const std::size_t index = cell(i, j);
    if (map_[index] == kCompositeNoIndex)
      *value += transmissibility * dirichlet[index];
  }

  std::vector<double> residual_vector(
      const std::vector<double>& rhs,
      const std::vector<double>& potential) const {
    const std::vector<double> applied = apply(potential);
    std::vector<double> result(rhs.size(), 0.0);
    for (std::size_t i = 0U; i < rhs.size(); ++i)
      result[i] = rhs[i] - applied[i];
    return result;
  }

  double gauss_defect(const std::vector<double>& potential,
                      const std::vector<double>& integrated_source,
                      const std::vector<double>& dirichlet) const {
    double source = 0.0;
    double outward_flux = 0.0;
    for (std::size_t dof = 0U; dof < unknown_count(); ++dof) {
      const std::size_t i = dof_i_[dof];
      const std::size_t j = dof_j_[dof];
      source += integrated_source[cell(i, j)];
      const std::pair<std::size_t, std::size_t> neighbors[4] = {
          std::make_pair(i - 1U, j), std::make_pair(i + 1U, j),
          std::make_pair(i, j - 1U), std::make_pair(i, j + 1U)};
      const double transmissibilities[4] = {
          delta_y_ / delta_x_, delta_y_ / delta_x_,
          delta_x_ / delta_y_, delta_x_ / delta_y_};
      for (std::size_t n = 0U; n < 4U; ++n) {
        const std::size_t index = cell(neighbors[n].first,
                                       neighbors[n].second);
        if (map_[index] == kCompositeNoIndex)
          outward_flux += transmissibilities[n] *
              (potential[dof] - dirichlet[index]);
      }
    }
    return outward_flux - source;
  }
};

CompositeTentParticleR2 particle(std::uint64_t stable_id, double x, double y,
                                 double charge, double half_x,
                                 double half_y) {
  CompositeTentParticleR2 result;
  result.stable_id = stable_id;
  result.position_x = x;
  result.position_y = y;
  result.signed_charge = charge;
  result.support_half_width_x = half_x;
  result.support_half_width_y = half_y;
  return result;
}

CompositePatchGeometryR2 geometry(std::size_t coarse_n, double origin,
                                  double coarse_delta,
                                  std::size_t begin_x,
                                  std::size_t end_x,
                                  std::size_t begin_y,
                                  std::size_t end_y,
                                  std::size_t proper_nesting = 1U) {
  CompositePatchGeometryR2 result;
  result.coarse_n_x = coarse_n;
  result.coarse_n_y = coarse_n;
  result.coarse_delta_x = coarse_delta;
  result.coarse_delta_y = coarse_delta;
  result.physical_origin_x = origin;
  result.physical_origin_y = origin;
  result.patch_begin_x = begin_x;
  result.patch_end_x = end_x;
  result.patch_begin_y = begin_y;
  result.patch_end_y = end_y;
  result.fixed_boundary_layers = 1U;
  result.proper_nesting_cells = proper_nesting;
  result.refinement_ratio = 2U;
  return result;
}

std::vector<double> composite_dirichlet(
    const CompositePoissonR2& topology,
    double (*function)(double, double)) {
  std::vector<double> result(topology.leaf_count(), 0.0);
  for (std::size_t leaf = 0U; leaf < topology.leaf_count(); ++leaf)
    if (topology.leaves()[leaf].fixed_dirichlet)
      result[leaf] = function(topology.leaves()[leaf].center_x,
                              topology.leaves()[leaf].center_y);
  return result;
}

std::vector<double> uniform_dirichlet(
    const UniformFineOracle& topology,
    double (*function)(double, double)) {
  std::vector<double> result(topology.cell_count(), 0.0);
  for (std::size_t i = 0U; i < topology.n_x(); ++i)
    for (std::size_t j = 0U; j < topology.n_y(); ++j)
      if (topology.fixed(i, j))
        result[topology.cell(i, j)] =
            function(topology.center_x(i), topology.center_y(j));
  return result;
}

struct CompositeSolved {
  std::vector<double> source;
  std::vector<double> dirichlet;
  CompositePcgResult solve;
  CompositeGaussReport gauss;
};

CompositeSolved solve_composite(const CompositePoissonR2& topology,
                                const std::vector<double>& source,
                                const std::vector<double>& dirichlet) {
  CompositeSolved result;
  result.source = source;
  result.dirichlet = dirichlet;
  const std::vector<double> rhs = topology.assemble_integrated_rhs(
      source, 1.0, dirichlet);
  CompositePcgOptions options;
  options.relative_tolerance = 2.0e-13;
  options.absolute_tolerance = 1.0e-15;
  options.maximum_iterations = 100000U;
  options.residual_recompute_interval = 16U;
  result.solve = topology.solve_pcg(rhs, std::vector<double>(), options);
  require(result.solve.solver_converged(),
          "composite physics solve did not converge");
  result.gauss = topology.compute_gauss_report(
      result.solve.solution, dirichlet, source, 1.0);
  return result;
}

double leaf_potential(const CompositePoissonR2& topology,
                      const CompositeSolved& solved,
                      std::size_t leaf_index) {
  const CompositeLeafCell& leaf = topology.leaves()[leaf_index];
  return leaf.dof == kCompositeNoIndex
      ? solved.dirichlet[leaf_index] : solved.solve.solution[leaf.dof];
}

struct StageGate {
  bool smooth_potential_one_percent = true;
  bool smooth_field_one_percent = true;
  bool smooth_order_plausible = true;
  bool localized_uniform_one_percent = true;
  bool interface_one_percent = true;
  bool gaussian_one_percent = true;
  bool algebraic_closure = true;

  bool pass() const {
    return smooth_potential_one_percent && smooth_field_one_percent &&
           smooth_order_plausible && localized_uniform_one_percent &&
           interface_one_percent && gaussian_one_percent &&
           algebraic_closure;
  }
};

void report_metric(const std::string& case_name, const std::string& metric,
                   double value) {
  std::cout << "METRIC case=" << case_name << " name=" << metric
            << " value=" << std::setprecision(17) << value << '\n';
}

double smooth_phi(double x, double y) {
  const double u = 0.5 * (x + 1.0);
  const double v = 0.5 * (y + 1.0);
  return std::sin(kPi * u) * std::sin(kPi * v);
}

double smooth_source_density(double x, double y) {
  return 0.5 * kPi * kPi * smooth_phi(x, y);
}

double tent_fourier_factor(double wave_number, double half_width) {
  const double argument = 0.5 * wave_number * half_width;
  if (std::fabs(argument) < 1.0e-12) return 1.0;
  return square(std::sin(argument) / argument);
}

UniformSample smooth_exact_tent_sample(
    const CompositeTentParticleR2& probe) {
  const double wave_number = 0.5 * kPi;
  const double filter = tent_fourier_factor(
      wave_number, probe.support_half_width_x) *
      tent_fourier_factor(wave_number, probe.support_half_width_y);
  const double sx = std::sin(wave_number * (probe.position_x + 1.0));
  const double sy = std::sin(wave_number * (probe.position_y + 1.0));
  const double cx = std::cos(wave_number * (probe.position_x + 1.0));
  const double cy = std::cos(wave_number * (probe.position_y + 1.0));
  UniformSample result;
  result.potential = filter * sx * sy;
  result.electric_x = -filter * wave_number * cx * sy;
  result.electric_y = -filter * wave_number * sx * cy;
  return result;
}

std::vector<double> smooth_composite_source(
    const CompositePoissonR2& topology) {
  std::vector<double> source(topology.leaf_count(), 0.0);
  for (std::size_t leaf = 0U; leaf < topology.leaf_count(); ++leaf) {
    const CompositeLeafCell& cell = topology.leaves()[leaf];
    if (!cell.fixed_dirichlet)
      source[leaf] = smooth_source_density(cell.center_x, cell.center_y) *
                     cell.area;
  }
  return source;
}

std::vector<double> smooth_uniform_source(const UniformFineOracle& topology) {
  std::vector<double> source(topology.cell_count(), 0.0);
  for (std::size_t i = 0U; i < topology.n_x(); ++i)
    for (std::size_t j = 0U; j < topology.n_y(); ++j)
      if (!topology.fixed(i, j))
        source[topology.cell(i, j)] =
            smooth_source_density(topology.center_x(i),
                                  topology.center_y(j)) *
            topology.delta_x() * topology.delta_y();
  return source;
}

std::vector<CompositeTentParticleR2> smooth_probes() {
  std::vector<CompositeTentParticleR2> result;
  // Keep complete support outside the coarsest fixed boundary ring while
  // Keep every compact probe shape wholly inside the fine owner.  Interface
  // flux itself is tested independently below; a production source/gather
  // shape that crosses a level boundary must make the layout grow first.
  const double coordinates[5] = {-0.25, -0.125, 0.0, 0.125, 0.25};
  std::uint64_t id = 1000U;
  for (double x : coordinates)
    for (double y : coordinates)
      result.push_back(particle(id++, x, y, 0.0, 0.22, 0.22));
  return result;
}

struct SmoothLevelResult {
  std::size_t coarse_n = 0U;
  double composite_potential_relative_l2 = 0.0;
  double composite_potential_absolute_linf = 0.0;
  double composite_field_relative_l2 = 0.0;
  double composite_field_absolute_linf = 0.0;
  double composite_uniform_potential_relative_l2 = 0.0;
  double composite_uniform_field_relative_l2 = 0.0;
  double residual_relative = 0.0;
  double gauss_relative = 0.0;
};

SmoothLevelResult run_smooth_level(std::size_t coarse_n) {
  require(coarse_n >= 8U && coarse_n % 4U == 0U,
          "smooth level requires a multiple of four");
  const double coarse_delta = 2.0 / static_cast<double>(coarse_n);
  const CompositePoissonR2 composite(geometry(
      coarse_n, -1.0, coarse_delta, coarse_n / 4U,
      3U * coarse_n / 4U, coarse_n / 4U,
      3U * coarse_n / 4U));
  const UniformFineOracle uniform(
      2U * coarse_n, 2U * coarse_n, -1.0, -1.0,
      0.5 * coarse_delta, 0.5 * coarse_delta, 2U);
  const std::vector<double> composite_boundary =
      composite_dirichlet(composite, smooth_phi);
  const std::vector<double> uniform_boundary =
      uniform_dirichlet(uniform, smooth_phi);
  const CompositeSolved composite_solved = solve_composite(
      composite, smooth_composite_source(composite), composite_boundary);
  const auto uniform_solved = uniform.solve(
      smooth_uniform_source(uniform), uniform_boundary);
  require(uniform_solved.second.converged,
          "uniform-fine smooth solve did not converge");

  std::vector<double> composite_potential;
  std::vector<double> exact_potential;
  std::vector<double> uniform_potential;
  for (std::size_t leaf = 0U; leaf < composite.leaf_count(); ++leaf) {
    const CompositeLeafCell& cell = composite.leaves()[leaf];
    if (cell.fixed_dirichlet) continue;
    composite_potential.push_back(leaf_potential(
        composite, composite_solved, leaf));
    exact_potential.push_back(smooth_phi(cell.center_x, cell.center_y));
    uniform_potential.push_back(uniform.bilinear_potential(
        cell.center_x, cell.center_y, uniform_solved.first,
        uniform_boundary));
  }
  const ErrorNorms potential_exact = error_norms(
      composite_potential, exact_potential, 1.0e-10);
  const ErrorNorms potential_uniform = error_norms(
      composite_potential, uniform_potential, 1.0e-10);

  const std::vector<CompositeTentParticleR2> probes = smooth_probes();
  const CompositeTentStencilSetR2 probe_stencils =
      build_composite_tent_stencils_r2(composite, probes);
  const CompositeAdjointGatherResultR2 composite_samples =
      gather_composite_tent_adjoint_r2(
          composite, probe_stencils, composite_solved.solve.solution,
          composite_boundary);
  std::vector<double> composite_field;
  std::vector<double> exact_field;
  std::vector<double> uniform_field;
  for (std::size_t i = 0U; i < probes.size(); ++i) {
    require(composite_samples.samples[i].status ==
                CompositeTentStatus::Accepted,
            "smooth probe rejected by composite topology");
    const UniformSample exact = smooth_exact_tent_sample(probes[i]);
    const UniformSample reference = uniform.gather_tent(
        probes[i], uniform_solved.first, uniform_boundary);
    composite_field.push_back(composite_samples.samples[i].electric_x);
    composite_field.push_back(composite_samples.samples[i].electric_y);
    exact_field.push_back(exact.electric_x);
    exact_field.push_back(exact.electric_y);
    uniform_field.push_back(reference.electric_x);
    uniform_field.push_back(reference.electric_y);
  }
  const ErrorNorms field_exact = error_norms(
      composite_field, exact_field, 1.0e-8);
  const ErrorNorms field_uniform = error_norms(
      composite_field, uniform_field, 1.0e-8);
  const double rhs_norm = vector_l2(composite.assemble_integrated_rhs(
      composite_solved.source, 1.0, composite_boundary));
  const double source_magnitude = std::max(
      1.0e-15, std::fabs(composite_solved.gauss.integrated_source));

  SmoothLevelResult result;
  result.coarse_n = coarse_n;
  result.composite_potential_relative_l2 = potential_exact.relative_l2;
  result.composite_potential_absolute_linf = potential_exact.absolute_linf;
  result.composite_field_relative_l2 = field_exact.relative_l2;
  result.composite_field_absolute_linf = field_exact.absolute_linf;
  result.composite_uniform_potential_relative_l2 =
      potential_uniform.relative_l2;
  result.composite_uniform_field_relative_l2 = field_uniform.relative_l2;
  result.residual_relative =
      composite_solved.solve.report.final_residual.l2 /
      std::max(rhs_norm, 1.0e-15);
  result.gauss_relative = std::fabs(
      composite_solved.gauss.global_balance_defect) / source_magnitude;

  const std::string label = "smooth_n" + std::to_string(coarse_n);
  report_metric(label, "potential_exact_relative_l2",
                result.composite_potential_relative_l2);
  report_metric(label, "potential_exact_absolute_linf",
                result.composite_potential_absolute_linf);
  report_metric(label, "field_exact_relative_l2",
                result.composite_field_relative_l2);
  report_metric(label, "field_exact_absolute_linf",
                result.composite_field_absolute_linf);
  report_metric(label, "potential_uniform_relative_l2",
                result.composite_uniform_potential_relative_l2);
  report_metric(label, "field_uniform_relative_l2",
                result.composite_uniform_field_relative_l2);
  report_metric(label, "residual_relative_l2", result.residual_relative);
  report_metric(label, "gauss_relative", result.gauss_relative);
  report_metric(label, "interface_flux_two_sided_max",
      composite_solved.gauss.maximum_interface_two_sided_flux_defect);
  return result;
}

double observed_order(double coarse_error, double fine_error) {
  require(coarse_error > 0.0 && fine_error > 0.0,
          "observed order requires positive errors");
  return std::log(coarse_error / fine_error) / std::log(2.0);
}

void test_smooth_convergence(StageGate* gate) {
  const std::size_t levels[4] = {8U, 16U, 32U, 64U};
  std::vector<SmoothLevelResult> results;
  for (std::size_t level : levels) results.push_back(run_smooth_level(level));
  for (std::size_t i = 1U; i < results.size(); ++i) {
    const double potential_order = observed_order(
        results[i - 1U].composite_potential_relative_l2,
        results[i].composite_potential_relative_l2);
    const double field_order = observed_order(
        results[i - 1U].composite_field_relative_l2,
        results[i].composite_field_relative_l2);
    const std::string label = "smooth_" +
        std::to_string(results[i - 1U].coarse_n) + "_to_" +
        std::to_string(results[i].coarse_n);
    report_metric(label, "potential_observed_order", potential_order);
    report_metric(label, "field_observed_order", field_order);
    if (i >= 2U &&
        (!(potential_order >= 1.5 && potential_order <= 3.0) ||
         !(field_order >= 1.5 && field_order <= 3.0)))
      gate->smooth_order_plausible = false;
  }
  const SmoothLevelResult& finest = results.back();
  gate->smooth_potential_one_percent =
      finest.composite_potential_relative_l2 <= 0.01 &&
      finest.composite_uniform_potential_relative_l2 <= 0.01;
  gate->smooth_field_one_percent =
      finest.composite_field_relative_l2 <= 0.01 &&
      finest.composite_uniform_field_relative_l2 <= 0.01;
  for (const SmoothLevelResult& result : results) {
    if (!(result.residual_relative <= 1.0e-10 &&
          result.gauss_relative <= 1.0e-10))
      gate->algebraic_closure = false;
  }
}

struct GaussianModel {
  double center_x = 0.35;
  double center_y = -0.45;
  double sigma = 0.85;
};

const GaussianModel kGaussian;

double exponential_integral_e1(double x) {
  require(x > 0.0 && finite(x), "E1 requires a positive finite argument");
  constexpr double kEulerGamma =
      0.577215664901532860606512090082402431;
  if (x <= 1.0) {
    double term = -x;
    double series = term;
    for (std::size_t k = 2U; k <= 200U; ++k) {
      term *= -x / static_cast<double>(k);
      const double addition = term / static_cast<double>(k);
      series += addition;
      if (std::fabs(addition) <= 2.0e-16 * std::max(1.0, std::fabs(series)))
        break;
    }
    return -kEulerGamma - std::log(x) - series;
  }
  constexpr double kTiny = 1.0e-300;
  double b = x + 1.0;
  double c = 1.0 / kTiny;
  double d = 1.0 / b;
  double result = d;
  for (std::size_t k = 1U; k <= 200U; ++k) {
    const double index = static_cast<double>(k);
    const double a = -index * index;
    b += 2.0;
    d = 1.0 / std::max(kTiny, a * d + b);
    c = b + a / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    const double delta = c * d;
    result *= delta;
    if (std::fabs(delta - 1.0) <= 2.0e-15) break;
  }
  return result * std::exp(-x);
}

double gaussian_phi(double x, double y) {
  const double dx = x - kGaussian.center_x;
  const double dy = y - kGaussian.center_y;
  const double radius_squared = dx * dx + dy * dy;
  const double sigma_squared = square(kGaussian.sigma);
  if (radius_squared <= 1.0e-28 * sigma_squared)
    return (0.577215664901532860606512090082402431 -
            std::log(2.0 * sigma_squared)) / (4.0 * kPi);
  const double argument = radius_squared / (2.0 * sigma_squared);
  const double ei_negative = -exponential_integral_e1(argument);
  return -(0.5 / kPi) *
      (0.5 * std::log(radius_squared) - 0.5 * ei_negative);
}

UniformSample gaussian_point_sample(double x, double y) {
  const double dx = x - kGaussian.center_x;
  const double dy = y - kGaussian.center_y;
  const double radius_squared = dx * dx + dy * dy;
  UniformSample result;
  result.potential = gaussian_phi(x, y);
  if (radius_squared > 1.0e-28 * square(kGaussian.sigma)) {
    const double factor =
        (1.0 - std::exp(-radius_squared /
                        (2.0 * square(kGaussian.sigma)))) /
        (2.0 * kPi * radius_squared);
    result.electric_x = factor * dx;
    result.electric_y = factor * dy;
  }
  return result;
}

double normal_fraction(double begin, double end, double mean,
                       double sigma) {
  const double scale = std::sqrt(2.0) * sigma;
  return 0.5 * (std::erf((end - mean) / scale) -
                std::erf((begin - mean) / scale));
}

std::vector<double> gaussian_composite_source(
    const CompositePoissonR2& topology) {
  std::vector<double> result(topology.leaf_count(), 0.0);
  for (std::size_t index = 0U; index < topology.leaf_count(); ++index) {
    const CompositeLeafCell& leaf = topology.leaves()[index];
    if (leaf.fixed_dirichlet) continue;
    const double x_fraction = normal_fraction(
        leaf.center_x - 0.5 * leaf.delta_x,
        leaf.center_x + 0.5 * leaf.delta_x,
        kGaussian.center_x, kGaussian.sigma);
    const double y_fraction = normal_fraction(
        leaf.center_y - 0.5 * leaf.delta_y,
        leaf.center_y + 0.5 * leaf.delta_y,
        kGaussian.center_y, kGaussian.sigma);
    result[index] = x_fraction * y_fraction;
  }
  return result;
}

std::vector<double> gaussian_uniform_source(
    const UniformFineOracle& topology) {
  std::vector<double> result(topology.cell_count(), 0.0);
  for (std::size_t i = 0U; i < topology.n_x(); ++i)
    for (std::size_t j = 0U; j < topology.n_y(); ++j) {
      if (topology.fixed(i, j)) continue;
      const double x_fraction = normal_fraction(
          topology.origin_x() + static_cast<double>(i) * topology.delta_x(),
          topology.origin_x() +
              static_cast<double>(i + 1U) * topology.delta_x(),
          kGaussian.center_x, kGaussian.sigma);
      const double y_fraction = normal_fraction(
          topology.origin_y() + static_cast<double>(j) * topology.delta_y(),
          topology.origin_y() +
              static_cast<double>(j + 1U) * topology.delta_y(),
          kGaussian.center_y, kGaussian.sigma);
      result[topology.cell(i, j)] = x_fraction * y_fraction;
    }
  return result;
}

UniformSample gaussian_tent_convolution(
    const CompositeTentParticleR2& probe) {
  constexpr double nodes[8] = {
      -0.9602898564975363, -0.7966664774136267,
      -0.5255324099163290, -0.1834346424956498,
       0.1834346424956498,  0.5255324099163290,
       0.7966664774136267,  0.9602898564975363};
  constexpr double weights[8] = {
      0.1012285362903763, 0.2223810344533745,
      0.3137066458778873, 0.3626837833783620,
      0.3626837833783620, 0.3137066458778873,
      0.2223810344533745, 0.1012285362903763};
  UniformSample result;
  for (std::size_t sx = 0U; sx < 2U; ++sx)
    for (std::size_t nx = 0U; nx < 8U; ++nx) {
      const double x_begin = sx == 0U
          ? -probe.support_half_width_x : 0.0;
      const double x_end = sx == 0U
          ? 0.0 : probe.support_half_width_x;
      const double ux = 0.5 * ((x_end - x_begin) * nodes[nx] +
                               x_end + x_begin);
      const double wx = 0.5 * (x_end - x_begin) * weights[nx] *
                        tent_value(ux, probe.support_half_width_x);
      for (std::size_t sy = 0U; sy < 2U; ++sy)
        for (std::size_t ny = 0U; ny < 8U; ++ny) {
          const double y_begin = sy == 0U
              ? -probe.support_half_width_y : 0.0;
          const double y_end = sy == 0U
              ? 0.0 : probe.support_half_width_y;
          const double uy = 0.5 * ((y_end - y_begin) * nodes[ny] +
                                   y_end + y_begin);
          const double wy = 0.5 * (y_end - y_begin) * weights[ny] *
                            tent_value(uy, probe.support_half_width_y);
          const UniformSample point = gaussian_point_sample(
              probe.position_x + ux, probe.position_y + uy);
          result.potential += wx * wy * point.potential;
          result.electric_x += wx * wy * point.electric_x;
          result.electric_y += wx * wy * point.electric_y;
        }
    }
  return result;
}

void test_round_gaussian_oracle(StageGate* gate) {
  const std::size_t coarse_n = 40U;
  const double origin = -8.0;
  const double coarse_delta = 0.4;
  const CompositePoissonR2 composite(geometry(
      coarse_n, origin, coarse_delta, 10U, 31U, 9U, 30U));
  const UniformFineOracle uniform(80U, 80U, origin, origin, 0.2, 0.2, 2U);
  const std::vector<double> composite_boundary =
      composite_dirichlet(composite, gaussian_phi);
  const std::vector<double> uniform_boundary =
      uniform_dirichlet(uniform, gaussian_phi);
  const CompositeSolved composite_solved = solve_composite(
      composite, gaussian_composite_source(composite), composite_boundary);
  const auto uniform_solved = uniform.solve(
      gaussian_uniform_source(uniform), uniform_boundary);
  require(uniform_solved.second.converged,
          "uniform round-Gaussian solve did not converge");

  const double locations[8][2] = {
      {-2.1, -0.45}, {-1.0, 0.2}, {-0.25, -1.4}, {0.8, -0.45},
      {1.5, 0.7}, {2.4, -1.1}, {0.35, 1.8}, {-1.8, 1.55}};
  std::vector<CompositeTentParticleR2> probes;
  for (std::size_t i = 0U; i < 8U; ++i)
    probes.push_back(particle(3000U + i, locations[i][0], locations[i][1],
                              0.0, 0.35, 0.35));
  const CompositeTentStencilSetR2 stencils =
      build_composite_tent_stencils_r2(composite, probes);
  const CompositeAdjointGatherResultR2 samples =
      gather_composite_tent_adjoint_r2(
          composite, stencils, composite_solved.solve.solution,
          composite_boundary);
  std::vector<double> composite_potential;
  std::vector<double> uniform_potential;
  std::vector<double> analytic_potential;
  std::vector<double> composite_field;
  std::vector<double> uniform_field;
  std::vector<double> analytic_field;
  for (std::size_t i = 0U; i < probes.size(); ++i) {
    require(samples.samples[i].status == CompositeTentStatus::Accepted,
            "round-Gaussian probe rejected");
    const UniformSample fine = uniform.gather_tent(
        probes[i], uniform_solved.first, uniform_boundary);
    const UniformSample direct = gaussian_tent_convolution(probes[i]);
    composite_potential.push_back(samples.samples[i].interpolated_potential);
    uniform_potential.push_back(fine.potential);
    analytic_potential.push_back(direct.potential);
    composite_field.push_back(samples.samples[i].electric_x);
    composite_field.push_back(samples.samples[i].electric_y);
    uniform_field.push_back(fine.electric_x);
    uniform_field.push_back(fine.electric_y);
    analytic_field.push_back(direct.electric_x);
    analytic_field.push_back(direct.electric_y);
  }
  const ErrorNorms potential_direct = error_norms(
      composite_potential, analytic_potential, 1.0e-8);
  const ErrorNorms potential_uniform = error_norms(
      composite_potential, uniform_potential, 1.0e-8);
  const ErrorNorms uniform_potential_direct = error_norms(
      uniform_potential, analytic_potential, 1.0e-8);
  const ErrorNorms field_direct = error_norms(
      composite_field, analytic_field, 1.0e-7);
  const ErrorNorms field_uniform = error_norms(
      composite_field, uniform_field, 1.0e-7);
  const ErrorNorms uniform_field_direct = error_norms(
      uniform_field, analytic_field, 1.0e-7);
  report_metric("round_gaussian", "potential_direct_relative_l2",
                potential_direct.relative_l2);
  report_metric("round_gaussian", "potential_uniform_relative_l2",
                potential_uniform.relative_l2);
  report_metric("round_gaussian", "uniform_potential_direct_relative_l2",
                uniform_potential_direct.relative_l2);
  report_metric("round_gaussian", "field_direct_relative_l2",
                field_direct.relative_l2);
  report_metric("round_gaussian", "field_uniform_relative_l2",
                field_uniform.relative_l2);
  report_metric("round_gaussian", "uniform_field_direct_relative_l2",
                uniform_field_direct.relative_l2);
  report_metric("round_gaussian", "field_direct_absolute_linf",
                field_direct.absolute_linf);
  report_metric("round_gaussian", "field_direct_mixed_linf",
                field_direct.mixed_linf);
  report_metric("round_gaussian", "gauss_absolute",
                std::fabs(composite_solved.gauss.global_balance_defect));
  gate->gaussian_one_percent =
      potential_direct.relative_l2 <= 0.01 &&
      potential_uniform.relative_l2 <= 0.01 &&
      field_direct.relative_l2 <= 0.01 &&
      field_uniform.relative_l2 <= 0.01;
  if (!(std::fabs(composite_solved.gauss.global_balance_defect) <= 1.0e-10 &&
        composite_solved.gauss.maximum_interface_two_sided_flux_defect <=
            1.0e-14))
    gate->algebraic_closure = false;
}

std::vector<CompositeTentParticleR2> localized_sources() {
  std::vector<CompositeTentParticleR2> result;
  result.push_back(particle(4101U, 0.35, -0.45, 1.0, 0.62, 0.54));
  result.push_back(particle(4102U, 0.75, 0.40, -0.22, 0.55, 0.70));
  result.push_back(particle(4103U, -0.25, 0.25, 0.31, 0.70, 0.48));
  return result;
}

std::vector<CompositeTentParticleR2> localized_probes() {
  const double locations[10][2] = {
      {-0.50, -0.50}, {-0.35, 0.65}, {-0.25, -0.20}, {-0.10, 0.90},
      {0.15, -0.40}, {0.35, 0.10}, {0.55, 0.70}, {0.70, -0.25},
      {0.85, 0.45}, {1.00, -0.50}};
  std::vector<CompositeTentParticleR2> result;
  for (std::size_t i = 0U; i < 10U; ++i)
    result.push_back(particle(5000U + i, locations[i][0], locations[i][1],
                              0.0, 0.40, 0.40));
  return result;
}

CompositeSolved solve_composite_tents(
    const CompositePoissonR2& topology,
    const std::vector<CompositeTentParticleR2>& sources,
    CompositeTentStencilSetR2* source_stencils,
    CompositeDepositResultR2* deposit) {
  *source_stencils = build_composite_tent_stencils_r2(topology, sources);
  *deposit = deposit_composite_tent_source_once_r2(topology,
                                                    *source_stencils);
  require(deposit->telemetry.accepted_particles == sources.size() &&
              deposit->telemetry.rejected_particles == 0U &&
              deposit->telemetry.outside_particles == 0U,
          "localized source was not deposited exactly once");
  const std::vector<double> dirichlet(topology.leaf_count(), 0.0);
  return solve_composite(topology, deposit->integrated_source, dirichlet);
}

std::vector<double> gather_composite_vector(
    const CompositePoissonR2& topology,
    const CompositeSolved& solved,
    const std::vector<CompositeTentParticleR2>& probes,
    std::vector<double>* potential) {
  const CompositeTentStencilSetR2 stencils =
      build_composite_tent_stencils_r2(topology, probes);
  const CompositeAdjointGatherResultR2 samples =
      gather_composite_tent_adjoint_r2(
          topology, stencils, solved.solve.solution, solved.dirichlet);
  std::vector<double> field;
  for (const auto& sample : samples.samples) {
    require(sample.status == CompositeTentStatus::Accepted,
            "localized composite probe was rejected");
    potential->push_back(sample.interpolated_potential);
    field.push_back(sample.electric_x);
    field.push_back(sample.electric_y);
  }
  return field;
}

std::vector<double> gather_uniform_vector(
    const UniformFineOracle& topology,
    const std::vector<double>& solved,
    const std::vector<double>& dirichlet,
    const std::vector<CompositeTentParticleR2>& probes,
    std::vector<double>* potential) {
  std::vector<double> field;
  for (const CompositeTentParticleR2& probe : probes) {
    const UniformSample sample = topology.gather_tent(
        probe, solved, dirichlet);
    potential->push_back(sample.potential);
    field.push_back(sample.electric_x);
    field.push_back(sample.electric_y);
  }
  return field;
}

struct LocalizedResult {
  std::vector<double> solution;
  std::vector<double> deposited;
  std::vector<double> potential;
  std::vector<double> field;
  double potential_relative_l2 = 0.0;
  double field_relative_l2 = 0.0;
  double field_mixed_linf = 0.0;
  double gauss_absolute = 0.0;
  std::size_t crossing_sources = 0U;
};

LocalizedResult run_localized_layout(
    const CompositePatchGeometryR2& layout,
    const std::vector<CompositeTentParticleR2>& sources,
    const std::vector<CompositeTentParticleR2>& probes,
    const std::vector<double>& reference_potential,
    const std::vector<double>& reference_field) {
  const CompositePoissonR2 topology(layout);
  CompositeTentStencilSetR2 source_stencils;
  CompositeDepositResultR2 deposit;
  const CompositeSolved solved = solve_composite_tents(
      topology, sources, &source_stencils, &deposit);
  LocalizedResult result;
  result.solution = solved.solve.solution;
  result.deposited = deposit.integrated_source;
  result.field = gather_composite_vector(topology, solved, probes,
                                         &result.potential);
  const ErrorNorms potential_error = error_norms(
      result.potential, reference_potential, 1.0e-8);
  const ErrorNorms field_error = error_norms(
      result.field, reference_field, 1.0e-7);
  result.potential_relative_l2 = potential_error.relative_l2;
  result.field_relative_l2 = field_error.relative_l2;
  result.field_mixed_linf = field_error.mixed_linf;
  result.gauss_absolute = std::fabs(solved.gauss.global_balance_defect);
  for (const auto& stencil : source_stencils.stencils)
    if (stencil.crosses_coarse_fine_interface) ++result.crossing_sources;
  require(deposit.telemetry.maximum_weight_closure_defect <= 3.0e-12 &&
              deposit.telemetry.maximum_scaled_derivative_closure_defect <=
                  3.0e-12 &&
              deposit.telemetry.maximum_scaled_first_moment_defect <=
                  3.0e-12,
          "localized composite source closure failed");
  return result;
}

void test_buffer_and_patch_translation(StageGate* gate) {
  const std::size_t coarse_n = 32U;
  const double origin = -8.0;
  const double coarse_delta = 0.5;
  const UniformFineOracle uniform(64U, 64U, origin, origin, 0.25, 0.25, 2U);
  const std::vector<double> uniform_boundary(uniform.cell_count(), 0.0);
  const std::vector<CompositeTentParticleR2> sources = localized_sources();
  const std::vector<CompositeTentParticleR2> probes = localized_probes();
  const auto uniform_solved = uniform.solve(
      uniform.deposit_tents(sources), uniform_boundary);
  require(uniform_solved.second.converged,
          "localized complete-uniform-fine solve did not converge");
  std::vector<double> reference_potential;
  const std::vector<double> reference_field = gather_uniform_vector(
      uniform, uniform_solved.first, uniform_boundary, probes,
      &reference_potential);

  const std::size_t buffers[4] = {0U, 2U, 4U, 8U};
  LocalizedResult buffered_results[4];
  for (std::size_t i = 0U; i < 4U; ++i) {
    const std::size_t buffer = buffers[i];
    buffered_results[i] = run_localized_layout(
        geometry(coarse_n, origin, coarse_delta,
                 14U - buffer, 19U + buffer,
                 14U - buffer, 19U + buffer),
        sources, probes, reference_potential, reference_field);
    const std::string label = "localized_buffer" + std::to_string(buffer);
    report_metric(label, "potential_uniform_relative_l2",
                  buffered_results[i].potential_relative_l2);
    report_metric(label, "field_uniform_relative_l2",
                  buffered_results[i].field_relative_l2);
    report_metric(label, "field_uniform_mixed_linf",
                  buffered_results[i].field_mixed_linf);
    report_metric(label, "gauss_absolute",
                  buffered_results[i].gauss_absolute);
    report_metric(label, "crossing_sources",
                  static_cast<double>(buffered_results[i].crossing_sources));
  }
  gate->localized_uniform_one_percent =
      buffered_results[3].potential_relative_l2 <= 0.01 &&
      buffered_results[3].field_relative_l2 <= 0.01;

  std::vector<std::vector<double> > translated_fields;
  double maximum_reference_error = 0.0;
  const int translations[5] = {-4, -2, 0, 2, 4};
  for (int shift : translations) {
    const std::size_t begin_x = static_cast<std::size_t>(8 + shift);
    const std::size_t end_x = static_cast<std::size_t>(24 + shift);
    const LocalizedResult translated = run_localized_layout(
        geometry(coarse_n, origin, coarse_delta,
                 begin_x, end_x, 8U, 24U),
        sources, probes, reference_potential, reference_field);
    translated_fields.push_back(translated.field);
    maximum_reference_error = std::max(
        maximum_reference_error, translated.field_relative_l2);
    const std::string label = "patch_translation" +
        (shift >= 0 ? std::string("_plus") : std::string("_minus")) +
        std::to_string(std::abs(shift));
    report_metric(label, "field_uniform_relative_l2",
                  translated.field_relative_l2);
    report_metric(label, "field_uniform_mixed_linf",
                  translated.field_mixed_linf);
  }
  double maximum_translation_relative_l2 = 0.0;
  for (std::size_t i = 0U; i < translated_fields.size(); ++i)
    for (std::size_t j = i + 1U; j < translated_fields.size(); ++j) {
      std::vector<double> difference(translated_fields[i].size(), 0.0);
      for (std::size_t k = 0U; k < difference.size(); ++k)
        difference[k] = translated_fields[i][k] - translated_fields[j][k];
      maximum_translation_relative_l2 = std::max(
          maximum_translation_relative_l2,
          vector_l2(difference) / std::max(vector_l2(reference_field),
                                            1.0e-12));
    }
  report_metric("patch_translation_sweep", "maximum_pair_relative_l2",
                maximum_translation_relative_l2);
  report_metric("patch_translation_sweep", "maximum_uniform_relative_l2",
                maximum_reference_error);
  gate->interface_one_percent =
      maximum_translation_relative_l2 <= 0.01 &&
      maximum_reference_error <= 0.01;

  const LocalizedResult replay_a = run_localized_layout(
      geometry(coarse_n, origin, coarse_delta, 10U, 27U, 10U, 27U),
      sources, probes, reference_potential, reference_field);
  const LocalizedResult replay_b = run_localized_layout(
      geometry(coarse_n, origin, coarse_delta, 10U, 27U, 10U, 27U),
      sources, probes, reference_potential, reference_field);
  require(bitwise_equal(replay_a.solution, replay_b.solution) &&
              bitwise_equal(replay_a.deposited, replay_b.deposited) &&
              bitwise_equal(replay_a.potential, replay_b.potential) &&
              bitwise_equal(replay_a.field, replay_b.field),
          "composite deterministic replay changed bits");
  report_metric("deterministic_replay", "bitwise_equal", 1.0);
}

void test_source_interface_phase_sweep(StageGate* gate) {
  const std::size_t coarse_n = 32U;
  const double origin = -8.0;
  const double coarse_delta = 0.5;
  const CompositePatchGeometryR2 layout = geometry(
      coarse_n, origin, coarse_delta, 10U, 26U, 8U, 24U);
  const UniformFineOracle uniform(64U, 64U, origin, origin, 0.25, 0.25, 2U);
  const std::vector<double> uniform_boundary(uniform.cell_count(), 0.0);
  const std::vector<CompositeTentParticleR2> probes = localized_probes();
  double worst_field_relative = 0.0;
  double worst_potential_relative = 0.0;
  double maximum_adjacent_jump = 0.0;
  double maximum_adjacent_error_jump = 0.0;
  std::vector<double> previous_field;
  std::vector<double> previous_error;
  std::size_t rejected_crossing_phases = 0U;
  std::size_t accepted_complete_shape_phases = 0U;
  for (std::size_t phase = 0U; phase <= 12U; ++phase) {
    const double x = -3.75 + 0.125 * static_cast<double>(phase);
    const std::vector<CompositeTentParticleR2> source(
        1U, particle(7001U, x, -0.35, 1.0, 0.62, 0.57));
    const CompositePoissonR2 phase_topology(layout);
    const CompositeTentStencilSetR2 ownership =
        build_composite_tent_stencils_r2(phase_topology, source);
    if (ownership.stencils[0].status ==
        CompositeTentStatus::SupportCrossesRefinementBoundary) {
      ++rejected_crossing_phases;
      report_metric("source_interface_phase" + std::to_string(phase),
                    "complete_shape_rejected", 1.0);
      previous_field.clear();
      previous_error.clear();
      continue;
    }
    require(ownership.stencils[0].status == CompositeTentStatus::Accepted,
            "interface phase had an unexpected ownership rejection");
    ++accepted_complete_shape_phases;
    const auto uniform_solved = uniform.solve(
        uniform.deposit_tents(source), uniform_boundary);
    require(uniform_solved.second.converged,
            "interface-phase uniform solve did not converge");
    std::vector<double> reference_potential;
    const std::vector<double> reference_field = gather_uniform_vector(
        uniform, uniform_solved.first, uniform_boundary, probes,
        &reference_potential);
    const LocalizedResult result = run_localized_layout(
        layout, source, probes, reference_potential, reference_field);
    worst_field_relative = std::max(worst_field_relative,
                                    result.field_relative_l2);
    worst_potential_relative = std::max(worst_potential_relative,
                                        result.potential_relative_l2);
    if (!previous_field.empty()) {
      std::vector<double> delta(result.field.size(), 0.0);
      for (std::size_t i = 0U; i < delta.size(); ++i)
        delta[i] = result.field[i] - previous_field[i];
      maximum_adjacent_jump = std::max(
          maximum_adjacent_jump,
          vector_l2(delta) / std::max(vector_l2(reference_field), 1.0e-12));
    }
    std::vector<double> current_error(result.field.size(), 0.0);
    for (std::size_t i = 0U; i < current_error.size(); ++i)
      current_error[i] = result.field[i] - reference_field[i];
    if (!previous_error.empty()) {
      std::vector<double> delta_error(current_error.size(), 0.0);
      for (std::size_t i = 0U; i < delta_error.size(); ++i)
        delta_error[i] = current_error[i] - previous_error[i];
      maximum_adjacent_error_jump = std::max(
          maximum_adjacent_error_jump,
          vector_l2(delta_error) /
              std::max(vector_l2(reference_field), 1.0e-12));
    }
    previous_field = result.field;
    previous_error = current_error;
    const std::string label = "source_interface_phase" +
                              std::to_string(phase);
    report_metric(label, "field_uniform_relative_l2",
                  result.field_relative_l2);
    report_metric(label, "potential_uniform_relative_l2",
                  result.potential_relative_l2);
    report_metric(label, "crossing_source",
                  static_cast<double>(result.crossing_sources));
  }
  require(rejected_crossing_phases >= 5U &&
              accepted_complete_shape_phases >= 2U,
          "source phase sweep did not exercise the strict ownership band");
  report_metric("source_interface_sweep",
                "rejected_crossing_phases",
                static_cast<double>(rejected_crossing_phases));
  report_metric("source_interface_sweep",
                "accepted_complete_shape_phases",
                static_cast<double>(accepted_complete_shape_phases));
  report_metric("source_interface_sweep", "worst_field_relative_l2",
                worst_field_relative);
  report_metric("source_interface_sweep", "worst_potential_relative_l2",
                worst_potential_relative);
  report_metric("source_interface_sweep", "maximum_adjacent_relative_step",
                maximum_adjacent_jump);
  report_metric("source_interface_sweep",
                "maximum_adjacent_numerical_error_step",
                maximum_adjacent_error_jump);
  // Until the runtime layout grow/retry loop removes every rejected support,
  // this source-interface route is intentionally not a Stage-4 candidate.
  if (rejected_crossing_phases != 0U ||
      !(worst_field_relative <= 0.01 &&
        worst_potential_relative <= 0.01))
    gate->interface_one_percent = false;
}

void test_tangential_linear_interface_flux(StageGate* gate) {
  const CompositePoissonR2 topology(geometry(
      16U, -4.0, 0.5, 4U, 12U, 4U, 12U));
  double maximum_spurious_normal_gradient = 0.0;
  double squared_gradient_sum = 0.0;
  double maximum_one_sided_gradient_defect = 0.0;
  double maximum_structural_two_sided_defect = 0.0;
  std::size_t subfaces = 0U;
  typedef std::pair<unsigned int, std::size_t> ParentKey;
  std::map<ParentKey, double> parent_outward_flux;
  const CompositeFaceAxis axes[2] = {
      CompositeFaceAxis::XNormal, CompositeFaceAxis::YNormal};
  for (CompositeFaceAxis selected_axis : axes) {
    std::vector<double> potential(topology.unknown_count(), 0.0);
    std::vector<double> boundary(topology.leaf_count(), 0.0);
    for (std::size_t leaf_index = 0U;
         leaf_index < topology.leaf_count(); ++leaf_index) {
      const CompositeLeafCell& leaf = topology.leaves()[leaf_index];
      // phi=y on x-normal faces and phi=x on y-normal faces.  Each exact
      // continuum field has identically zero normal derivative.
      const double value = selected_axis == CompositeFaceAxis::XNormal
          ? leaf.center_y : leaf.center_x;
      boundary[leaf_index] = value;
      if (!leaf.fixed_dirichlet) potential[leaf.dof] = value;
    }
    const std::vector<feebl::adaptive_grid::CompositeFaceFlux> fluxes =
        topology.compute_face_fluxes(potential, boundary);
    const std::vector<
        feebl::adaptive_grid::CompositeInterfaceFluxConsistency> audits =
        topology.compute_interface_flux_consistency(potential, boundary);
    for (const auto& audit : audits) {
      const auto& face = topology.faces()[audit.face];
      if (face.axis != selected_axis) continue;
      const double spurious_gradient =
          std::fabs(audit.variational_flux_a_to_b) / face.area;
      maximum_spurious_normal_gradient = std::max(
          maximum_spurious_normal_gradient, spurious_gradient);
      squared_gradient_sum += spurious_gradient * spurious_gradient;
      maximum_one_sided_gradient_defect = std::max(
          maximum_one_sided_gradient_defect,
          audit.absolute_one_sided_defect / face.area);
      const auto& structural = fluxes[audit.face];
      maximum_structural_two_sided_defect = std::max(
          maximum_structural_two_sided_defect,
          std::fabs(structural.outward_from_a +
                    structural.outward_from_b));
      ++subfaces;
      const CompositeLeafCell& a = topology.leaves()[face.leaf_a];
      const bool a_is_coarse = a.level == CompositeLevel::Coarse;
      const std::size_t coarse_leaf =
          a_is_coarse ? face.leaf_a : face.leaf_b;
      const double outward_from_coarse = a_is_coarse
          ? structural.outward_from_a : structural.outward_from_b;
      parent_outward_flux[ParentKey(
          static_cast<unsigned int>(face.axis), coarse_leaf)] +=
          outward_from_coarse;
    }
  }
  require(subfaces > 0U, "tangential-flux test found no interface subfaces");
  double maximum_parent_cancellation_defect = 0.0;
  for (const auto& entry : parent_outward_flux)
    maximum_parent_cancellation_defect = std::max(
        maximum_parent_cancellation_defect, std::fabs(entry.second));
  const double rms_spurious_normal_gradient = std::sqrt(
      squared_gradient_sum / static_cast<double>(subfaces));
  report_metric("tangential_linear_interface",
                "maximum_spurious_normal_gradient",
                maximum_spurious_normal_gradient);
  report_metric("tangential_linear_interface",
                "rms_spurious_normal_gradient",
                rms_spurious_normal_gradient);
  report_metric("tangential_linear_interface",
                "maximum_parent_cancellation_defect",
                maximum_parent_cancellation_defect);
  report_metric("tangential_linear_interface",
                "maximum_one_sided_gradient_defect",
                maximum_one_sided_gradient_defect);
  report_metric("tangential_linear_interface",
                "maximum_structural_two_sided_defect",
                maximum_structural_two_sided_defect);
  require(maximum_spurious_normal_gradient <= 5.0e-14,
          "multipoint tangential reconstruction aliases normal flux");
  require(maximum_parent_cancellation_defect <= 2.0e-15,
          "tangential subfaces are not conservatively accounted");
  require(maximum_one_sided_gradient_defect <= 5.0e-14,
          "independent one-sided tangential fluxes disagree");
  require(maximum_structural_two_sided_defect == 0.0,
          "structural two-sided interface accounting is not exact");
  // Unit tangential gradient is the reference scale.
  if (maximum_spurious_normal_gradient > 0.01)
    gate->interface_one_percent = false;
}

void test_deliberate_fail_closed_paths() {
  require_throws([]() {
    (void)CompositePoissonR2(geometry(16U, -4.0, 0.5,
                                      1U, 8U, 4U, 12U));
  }, "improperly nested patch did not fail closed");

  const CompositePoissonR2 a(geometry(16U, -4.0, 0.5,
                                      4U, 12U, 4U, 12U));
  const CompositePoissonR2 b(geometry(16U, -4.0, 0.5,
                                      5U, 13U, 4U, 12U));
  const std::vector<CompositeTentParticleR2> central(
      1U, particle(8001U, 0.1, -0.2, 1.0, 0.55, 0.55));
  const CompositeTentStencilSetR2 stencils =
      build_composite_tent_stencils_r2(a, central);
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(b, stencils);
  }, "stale topology fingerprint did not fail closed");

  CompositeTentStencilSetR2 corrupted = stencils;
  require(!corrupted.stencils[0].contributions.empty(),
          "corruption fixture has no contributions");
  corrupted.stencils[0].contributions[0].weight *= 1.01;
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(a, corrupted);
  }, "corrupted authenticated stencil did not fail closed");

  CompositeTentStencilSetR2 balanced_corruption = stencils;
  require(balanced_corruption.stencils[0].contributions.size() >= 2U,
          "balanced-corruption fixture lacks two contributions");
  const double delta = 0.25 * std::min(
      balanced_corruption.stencils[0].contributions[0].weight,
      balanced_corruption.stencils[0].contributions[1].weight);
  balanced_corruption.stencils[0].contributions[0].weight += delta;
  balanced_corruption.stencils[0].contributions[1].weight -= delta;
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(a, balanced_corruption);
  }, "balanced spatial stencil corruption passed aggregate closure");

  CompositeTentStencilSetR2 status_forgery = stencils;
  status_forgery.stencils[0].status =
      CompositeTentStatus::OutsidePhysicalDomain;
  status_forgery.stencils[0].owner_leaf = kCompositeNoIndex;
  status_forgery.stencils[0].touches_coarse = false;
  status_forgery.stencils[0].touches_fine = false;
  status_forgery.stencils[0].contributions.clear();
  status_forgery.stencils[0].weight_sum = 0.0;
  status_forgery.stencils[0].derivative_x_sum = 0.0;
  status_forgery.stencils[0].derivative_y_sum = 0.0;
  status_forgery.stencils[0].shape_first_x_sum = 0.0;
  status_forgery.stencils[0].shape_first_y_sum = 0.0;
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(a, status_forgery);
  }, "forged rejection status silently discarded an interior source");

  std::vector<double> boundary_source(a.leaf_count(), 0.0);
  std::size_t fixed_leaf = kCompositeNoIndex;
  for (std::size_t leaf = 0U; leaf < a.leaf_count(); ++leaf)
    if (a.leaves()[leaf].fixed_dirichlet) {
      fixed_leaf = leaf;
      break;
    }
  require(fixed_leaf != kCompositeNoIndex,
          "failure fixture has no fixed leaf");
  boundary_source[fixed_leaf] = 1.0e-30;
  require_throws([&]() {
    (void)a.assemble_integrated_rhs(
        boundary_source, 1.0, std::vector<double>(a.leaf_count(), 0.0));
  }, "nonzero source on fixed boundary was silently discarded");

  CompositeDepositResultR2 deposit;
  CompositeTentStencilSetR2 source_stencils;
  const CompositeSolved solved = solve_composite_tents(
      a, central, &source_stencils, &deposit);
  const std::vector<double> rhs = a.assemble_integrated_rhs(
      solved.source, 1.0, solved.dirichlet);
  CompositePcgOptions insufficient;
  insufficient.relative_tolerance = 1.0e-16;
  insufficient.absolute_tolerance = 0.0;
  insufficient.maximum_iterations = 1U;
  insufficient.residual_recompute_interval = 1U;
  const CompositePcgResult failed = a.solve_pcg(
      rhs, std::vector<double>(), insufficient);
  require(!failed.solver_converged() && !failed.report.converged,
          "one-iteration solver failure was publishable");

  std::vector<double> nonfinite_dirichlet(a.leaf_count(), 0.0);
  nonfinite_dirichlet[fixed_leaf] =
      std::numeric_limits<double>::quiet_NaN();
  require_throws([&]() {
    (void)a.assemble_integrated_rhs(solved.source, 1.0,
                                    nonfinite_dirichlet);
  }, "non-finite Dirichlet datum was accepted");

  std::vector<double> nonfinite_source(a.leaf_count(), 0.0);
  const std::size_t active_leaf = a.dof_to_leaf()[0U];
  nonfinite_source[active_leaf] =
      std::numeric_limits<double>::quiet_NaN();
  require_throws([&]() {
    (void)a.assemble_integrated_rhs(nonfinite_source, 1.0, solved.dirichlet);
  }, "non-finite integrated source was accepted");

  std::vector<double> overflow_source(a.leaf_count(), 0.0);
  overflow_source[active_leaf] = std::numeric_limits<double>::max();
  require_throws([&]() {
    (void)a.assemble_integrated_rhs(overflow_source, 2.0,
                                    solved.dirichlet);
  }, "scaled integrated-source overflow was accepted");

  std::vector<CompositeTentParticleR2> overflowing_particles;
  for (std::size_t i = 0U; i < 32U; ++i)
    overflowing_particles.push_back(particle(
        UINT64_C(8101) + static_cast<std::uint64_t>(i),
        0.1, -0.2, std::numeric_limits<double>::max(), 0.55, 0.55));
  const CompositeTentStencilSetR2 overflowing_stencils =
      build_composite_tent_stencils_r2(a, overflowing_particles);
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(a, overflowing_stencils);
  }, "deposit accumulation overflow was accepted");
  report_metric("deliberate_failure_modes", "fail_closed_cases", 11.0);
}

void print_gate(const char* name, bool pass) {
  std::cout << "GATE name=" << name << " pass=" << (pass ? 1 : 0) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  bool require_stage4 = false;
  if (argc == 2 && std::string(argv[1]) == "--require-stage4")
    require_stage4 = true;
  else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--require-stage4]\n";
    return 64;
  }

  try {
    StageGate gate;
    test_deliberate_fail_closed_paths();
    test_tangential_linear_interface_flux(&gate);
    test_smooth_convergence(&gate);
    test_round_gaussian_oracle(&gate);
    test_buffer_and_patch_translation(&gate);
    test_source_interface_phase_sweep(&gate);

    print_gate("smooth_potential_one_percent",
               gate.smooth_potential_one_percent);
    print_gate("smooth_field_one_percent", gate.smooth_field_one_percent);
    print_gate("smooth_order_plausible", gate.smooth_order_plausible);
    print_gate("localized_uniform_one_percent",
               gate.localized_uniform_one_percent);
    print_gate("interface_one_percent", gate.interface_one_percent);
    print_gate("gaussian_one_percent", gate.gaussian_one_percent);
    print_gate("algebraic_closure", gate.algebraic_closure);
    print_gate("standalone_stage4_candidate", gate.pass());

    // Standalone algebraic tests do not establish runtime field-kick and
    // luminosity agreement, or CPU/GPU equivalence. Preserve the overall
    // qualification status separately from the local numerical gates.
    std::cout << "physics_qualification_claim=0\n";
    std::cout << "master_stage4_pass=0\n";
    if (require_stage4 && !gate.pass()) return 2;
    std::cout << "adaptive composite physics validation harness: COMPLETE\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "adaptive composite physics validation harness: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
