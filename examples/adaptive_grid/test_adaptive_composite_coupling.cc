#include "adaptiveCompositeCouplingCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using feebl::adaptive_grid::CompositeAdjointGatherResultR2;
using feebl::adaptive_grid::CompositeDepositResultR2;
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

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance,
                   const std::string& message) {
  const double scale = std::max(1.0, std::max(std::fabs(actual),
                                              std::fabs(expected)));
  if (std::fabs(actual - expected) > tolerance * scale)
    throw std::runtime_error(message + ": actual=" +
                             std::to_string(actual) + " expected=" +
                             std::to_string(expected));
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

CompositePatchGeometryR2 test_geometry() {
  CompositePatchGeometryR2 geometry;
  geometry.coarse_n_x = 12U;
  geometry.coarse_n_y = 12U;
  geometry.coarse_delta_x = 1.0;
  geometry.coarse_delta_y = 1.0;
  geometry.physical_origin_x = -6.0;
  geometry.physical_origin_y = -6.0;
  geometry.patch_begin_x = 4U;
  geometry.patch_end_x = 8U;
  geometry.patch_begin_y = 4U;
  geometry.patch_end_y = 8U;
  geometry.fixed_boundary_layers = 1U;
  geometry.proper_nesting_cells = 1U;
  geometry.refinement_ratio = 2U;
  return geometry;
}

CompositeTentParticleR2 particle(std::uint64_t id, double x, double y,
                                 double charge, double hx, double hy) {
  CompositeTentParticleR2 result;
  result.stable_id = id;
  result.position_x = x;
  result.position_y = y;
  result.signed_charge = charge;
  result.support_half_width_x = hx;
  result.support_half_width_y = hy;
  return result;
}

double vector_sum(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values) result += value;
  return result;
}

double vector_l1_difference(const std::vector<double>& a,
                            const std::vector<double>& b) {
  require(a.size() == b.size(), "L1 vectors have unequal length");
  double result = 0.0;
  for (std::size_t i = 0U; i < a.size(); ++i)
    result += std::fabs(a[i] - b[i]);
  return result;
}

bool bitwise_equal(const std::vector<double>& a,
                   const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  if (a.empty()) return true;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

std::vector<double> deterministic_unknown_potential(
    const CompositePoissonR2& topology) {
  std::vector<double> potential(topology.unknown_count(), 0.0);
  for (std::size_t dof = 0U; dof < potential.size(); ++dof) {
    const CompositeLeafCell& leaf =
        topology.leaves()[topology.dof_to_leaf()[dof]];
    potential[dof] = 0.17 * leaf.center_x - 0.09 * leaf.center_y +
        0.03 * std::sin(0.7 * leaf.center_x + 0.2 * leaf.center_y);
  }
  return potential;
}

double source_potential_dot(const CompositePoissonR2& topology,
                            const std::vector<double>& source,
                            const std::vector<double>& unknown_potential,
                            const std::vector<double>& dirichlet) {
  require(source.size() == topology.leaf_count(),
          "source potential dot source size");
  double result = 0.0;
  for (std::size_t leaf_index = 0U;
       leaf_index < topology.leaf_count(); ++leaf_index) {
    const CompositeLeafCell& leaf = topology.leaves()[leaf_index];
    const double phi = leaf.dof == kCompositeNoIndex
        ? dirichlet[leaf_index]
        : unknown_potential[leaf.dof];
    result += source[leaf_index] * phi;
  }
  return result;
}

struct SolvedSource {
  CompositeTentStencilSetR2 stencils;
  CompositeDepositResultR2 deposit;
  std::vector<double> potential;
  std::vector<double> dirichlet;
  CompositePcgResult solve;
};

SolvedSource solve_particles(const CompositePoissonR2& topology,
                             const std::vector<CompositeTentParticleR2>& p) {
  SolvedSource result;
  result.stencils = build_composite_tent_stencils_r2(topology, p);
  result.deposit = deposit_composite_tent_source_once_r2(
      topology, result.stencils);
  require(result.deposit.telemetry.accepted_particles == p.size(),
          "solver source particle was rejected");
  result.dirichlet.assign(topology.leaf_count(), 0.0);
  const std::vector<double> rhs = topology.assemble_integrated_rhs(
      result.deposit.integrated_source, 1.0, result.dirichlet);
  CompositePcgOptions options;
  options.relative_tolerance = 1.0e-13;
  options.absolute_tolerance = 1.0e-15;
  options.maximum_iterations = 20000U;
  options.residual_recompute_interval = 16U;
  result.solve = topology.solve_pcg(rhs, std::vector<double>(), options);
  require(result.solve.solver_converged(),
          "coupling test Poisson solve failed");
  result.potential = result.solve.solution;
  return result;
}

void test_source_once_and_charge_moments(const CompositePoissonR2& topology) {
  std::vector<CompositeTentParticleR2> particles;
  particles.push_back(particle(30U, 0.35, -0.45, 2.25, 0.72, 0.63));
  particles.push_back(particle(10U, -3.25, 0.20, -0.75, 0.70, 0.70));
  particles.push_back(particle(20U, 3.15, -0.35, 1.50, 0.70, 0.60));
  const CompositeTentStencilSetR2 stencils =
      build_composite_tent_stencils_r2(topology, particles);
  require(stencils.stencils.size() == particles.size(),
          "source-once stencil count");
  require(stencils.stencils[0].particle.stable_id == 10U &&
          stencils.stencils[1].particle.stable_id == 20U &&
          stencils.stencils[2].particle.stable_id == 30U,
          "stencils are not canonical stable-ID order");

  for (const auto& stencil : stencils.stencils) {
    require(stencil.status == CompositeTentStatus::Accepted,
            "interior stencil unexpectedly rejected");
    require(stencil.owner_leaf != kCompositeNoIndex,
            "accepted stencil lacks exactly one owner");
    require_close(stencil.weight_sum, 1.0, 3.0e-13,
                  "tent partition of unity");
    require(std::fabs(stencil.derivative_x_sum) *
                stencil.particle.support_half_width_x < 3.0e-13,
            "x derivative partition closure");
    require(std::fabs(stencil.derivative_y_sum) *
                stencil.particle.support_half_width_y < 3.0e-13,
            "y derivative partition closure");
    std::vector<std::size_t> leaf_ids;
    for (const auto& contribution : stencil.contributions)
      leaf_ids.push_back(contribution.leaf);
    std::sort(leaf_ids.begin(), leaf_ids.end());
    require(std::adjacent_find(leaf_ids.begin(), leaf_ids.end()) ==
                leaf_ids.end(),
            "one particle deposits twice to a leaf");
    require(!stencil.crosses_coarse_fine_interface,
            "accepted source shape was split across refinement levels");
  }

  const CompositeDepositResultR2 deposit =
      deposit_composite_tent_source_once_r2(topology, stencils);
  require(deposit.telemetry.accepted_particles == 3U &&
          deposit.telemetry.rejected_particles == 0U &&
          deposit.telemetry.outside_particles == 0U,
          "accepted telemetry partition");
  const double expected_charge = 2.25 - 0.75 + 1.50;
  const double expected_first_x =
      2.25 * 0.35 - 0.75 * -3.25 + 1.50 * 3.15;
  const double expected_first_y =
      2.25 * -0.45 - 0.75 * 0.20 + 1.50 * -0.35;
  require_close(vector_sum(deposit.integrated_source), expected_charge,
                2.0e-13, "integrated signed charge closure");
  require_close(deposit.telemetry.integrated_source_charge,
                expected_charge, 2.0e-13,
                "telemetry signed charge closure");
  require_close(deposit.telemetry.deposited_exact_shape.signed_first_x,
                expected_first_x, 3.0e-13,
                "analytic deposited x first moment");
  require_close(deposit.telemetry.deposited_exact_shape.signed_first_y,
                expected_first_y, 3.0e-13,
                "analytic deposited y first moment");
}

void test_translation_and_phase_sweep(const CompositePoissonR2& topology) {
  std::vector<double> previous;
  double maximum_l1_step = 0.0;
  std::size_t coarse_accept_count = 0U;
  std::size_t fine_accept_count = 0U;
  std::size_t interface_reject_count = 0U;
  for (std::size_t phase = 0U; phase <= 100U; ++phase) {
    const double x = -3.25 + 0.025 * static_cast<double>(phase);
    const std::vector<CompositeTentParticleR2> particles(1U,
        particle(900U, x, 0.125, 1.0, 0.40, 0.40));
    const CompositeTentStencilSetR2 stencils =
        build_composite_tent_stencils_r2(topology, particles);
    const auto& stencil = stencils.stencils[0];
    if (stencil.status ==
        CompositeTentStatus::SupportCrossesRefinementBoundary) {
      ++interface_reject_count;
      require(stencil.contributions.empty(),
              "rejected interface shape retained source contributions");
      previous.clear();
      continue;
    }
    require(stencil.status == CompositeTentStatus::Accepted,
            "phase sweep had an unexpected rejection");
    require(!stencil.crosses_coarse_fine_interface,
            "phase sweep accepted a split-level source shape");
    require_close(stencils.stencils[0].weight_sum, 1.0, 4.0e-13,
                  "phase sweep charge closure");
    require_close(stencils.stencils[0].shape_first_x_sum, x, 4.0e-13,
                  "phase sweep first moment closure");
    if (topology.leaves()[stencil.owner_leaf].level == CompositeLevel::Fine)
      ++fine_accept_count;
    else
      ++coarse_accept_count;
    const CompositeDepositResultR2 deposit =
        deposit_composite_tent_source_once_r2(topology, stencils);
    if (!previous.empty())
      maximum_l1_step = std::max(maximum_l1_step,
          vector_l1_difference(previous, deposit.integrated_source));
    previous = deposit.integrated_source;
  }
  require(coarse_accept_count > 10U && fine_accept_count > 10U &&
          interface_reject_count > 20U,
          "phase sweep did not exercise both owners and the growth band");
  require(maximum_l1_step < 0.14,
          "phase sweep source changed discontinuously");
  std::cout << "translation_phase_max_l1_step=" << maximum_l1_step
            << " coarse_accepted=" << coarse_accept_count
            << " fine_accepted=" << fine_accept_count
            << " interface_rejected=" << interface_reject_count << '\n';
}

void test_permutation_and_rejection(const CompositePoissonR2& topology) {
  std::vector<CompositeTentParticleR2> forward;
  forward.push_back(particle(1U, -3.25, 0.25, 1.0, 0.7, 0.7));
  forward.push_back(particle(2U, 0.25, 0.75, -2.0, 0.6, 0.5));
  forward.push_back(particle(3U, 3.25, -0.35, 0.5, 0.7, 0.6));
  std::vector<CompositeTentParticleR2> reverse = forward;
  std::reverse(reverse.begin(), reverse.end());
  const CompositeDepositResultR2 a = deposit_composite_tent_source_once_r2(
      topology, build_composite_tent_stencils_r2(topology, forward));
  const CompositeDepositResultR2 b = deposit_composite_tent_source_once_r2(
      topology, build_composite_tent_stencils_r2(topology, reverse));
  require(bitwise_equal(a.integrated_source, b.integrated_source),
          "particle permutation changed source bits");
  require(std::memcmp(&a.telemetry, &b.telemetry,
                      sizeof(a.telemetry)) == 0,
          "particle permutation changed telemetry bits");

  std::vector<CompositeTentParticleR2> rejected;
  rejected.push_back(particle(11U, -6.1, 0.0, 1.0, 0.5, 0.5));
  rejected.push_back(particle(12U, -4.9, 0.0, 2.0, 1.3, 0.5));
  rejected.push_back(particle(13U, -5.5, 0.0, 3.0, 0.3, 0.3));
  rejected.push_back(particle(14U, -4.4, 0.0, 4.0, 0.8, 0.4));
  rejected.push_back(particle(15U, -2.1, 0.0, 5.0, 0.5, 0.3));
  const CompositeTentStencilSetR2 rejected_stencils =
      build_composite_tent_stencils_r2(topology, rejected);
  require(rejected_stencils.stencils[0].status ==
              CompositeTentStatus::OutsidePhysicalDomain,
          "outside center status");
  require(rejected_stencils.stencils[1].status ==
              CompositeTentStatus::SupportOutsidePhysicalDomain,
          "outside support status");
  require(rejected_stencils.stencils[2].status ==
              CompositeTentStatus::FixedBoundaryCenter,
          "fixed boundary center status");
  require(rejected_stencils.stencils[3].status ==
              CompositeTentStatus::FixedBoundarySupport,
          "fixed boundary support status");
  require(rejected_stencils.stencils[4].status ==
              CompositeTentStatus::SupportCrossesRefinementBoundary,
          "coarse/fine complete-shape ownership status");
  const CompositeDepositResultR2 rejected_deposit =
      deposit_composite_tent_source_once_r2(topology, rejected_stencils);
  require(rejected_deposit.telemetry.accepted_particles == 0U &&
          rejected_deposit.telemetry.outside_particles == 2U &&
          rejected_deposit.telemetry.rejected_particles == 3U,
          "rejection telemetry count partition");
  require_close(rejected_deposit.telemetry.outside.signed_charge,
                3.0, 0.0, "outside charge telemetry");
  require_close(rejected_deposit.telemetry.rejected.signed_charge,
                12.0, 0.0, "boundary-rejected charge telemetry");
  require_close(vector_sum(rejected_deposit.integrated_source),
                0.0, 0.0, "rejected particles deposited charge");

  std::vector<CompositeTentParticleR2> duplicate;
  duplicate.push_back(particle(77U, 0.0, 0.0, 1.0, 0.5, 0.5));
  duplicate.push_back(particle(77U, 0.5, 0.0, 1.0, 0.5, 0.5));
  require_throws([&]() {
    (void)build_composite_tent_stencils_r2(topology, duplicate);
  }, "duplicate stable IDs did not fail closed");

  std::vector<CompositeTentParticleR2> nonfinite(1U,
      particle(88U, 0.0, 0.0, 1.0, 0.5, 0.5));
  nonfinite[0].position_x = std::numeric_limits<double>::quiet_NaN();
  require_throws([&]() {
    (void)build_composite_tent_stencils_r2(topology, nonfinite);
  }, "non-finite coordinate did not fail closed");
  nonfinite[0] = particle(89U, 0.0, 0.0, 1.0,
                          std::numeric_limits<double>::infinity(), 0.5);
  require_throws([&]() {
    (void)build_composite_tent_stencils_r2(topology, nonfinite);
  }, "non-finite support did not fail closed");

  std::vector<CompositeTentParticleR2> moment_overflow(1U,
      particle(90U, 3.0, 0.0, std::numeric_limits<double>::max(),
               0.25, 0.25));
  const CompositeTentStencilSetR2 moment_overflow_stencils =
      build_composite_tent_stencils_r2(topology, moment_overflow);
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(
        topology, moment_overflow_stencils);
  }, "finite charge-position product overflow did not fail closed");

  std::vector<CompositeTentParticleR2> accumulation_overflow;
  accumulation_overflow.push_back(particle(
      91U, 0.0, 0.0, std::numeric_limits<double>::max(), 0.25, 0.25));
  accumulation_overflow.push_back(particle(
      92U, 0.0, 0.0, std::numeric_limits<double>::max(), 0.25, 0.25));
  const CompositeTentStencilSetR2 accumulation_overflow_stencils =
      build_composite_tent_stencils_r2(topology, accumulation_overflow);
  require_throws([&]() {
    (void)deposit_composite_tent_source_once_r2(
        topology, accumulation_overflow_stencils);
  }, "finite charge accumulation overflow did not fail closed");
}

void test_adjoint_work_identity(const CompositePoissonR2& topology) {
  std::vector<CompositeTentParticleR2> particles;
  particles.push_back(particle(101U, -3.15, 0.15, 1.25, 0.70, 0.65));
  particles.push_back(particle(102U, 0.35, -0.55, -0.80, 0.62, 0.71));
  particles.push_back(particle(103U, 3.10, 0.40, 0.55, 0.70, 0.57));
  const std::vector<double> velocity_x{0.31, -0.27, 0.18};
  const std::vector<double> velocity_y{-0.22, 0.19, 0.33};
  const CompositeTentStencilSetR2 stencils =
      build_composite_tent_stencils_r2(topology, particles);
  const std::vector<double> potential =
      deterministic_unknown_potential(topology);
  const std::vector<double> dirichlet(topology.leaf_count(), 0.0);
  const CompositeAdjointGatherResultR2 gather =
      gather_composite_tent_adjoint_r2(
          topology, stencils, potential, dirichlet);

  double direct_variation = 0.0;
  double gathered_work = 0.0;
  for (std::size_t p = 0U; p < stencils.stencils.size(); ++p) {
    const auto& stencil = stencils.stencils[p];
    require(stencil.status == CompositeTentStatus::Accepted,
            "work identity stencil rejected");
    require(gather.samples[p].stable_id == stencil.particle.stable_id,
            "gather stable-ID ordering");
    for (const auto& contribution : stencil.contributions) {
      const CompositeLeafCell& leaf =
          topology.leaves()[contribution.leaf];
      const double phi = leaf.dof == kCompositeNoIndex
          ? dirichlet[contribution.leaf]
          : potential[leaf.dof];
      direct_variation += stencil.particle.signed_charge * phi *
          (contribution.derivative_x * velocity_x[p] +
           contribution.derivative_y * velocity_y[p]);
    }
    gathered_work -= stencil.particle.signed_charge *
        (gather.samples[p].electric_x * velocity_x[p] +
         gather.samples[p].electric_y * velocity_y[p]);
  }
  require_close(direct_variation, gathered_work, 2.0e-14,
                "analytic adjoint scatter/gather work identity");

  const double epsilon = 2.0e-6;
  std::vector<CompositeTentParticleR2> plus = particles;
  std::vector<CompositeTentParticleR2> minus = particles;
  for (std::size_t i = 0U; i < particles.size(); ++i) {
    plus[i].position_x += epsilon * velocity_x[i];
    plus[i].position_y += epsilon * velocity_y[i];
    minus[i].position_x -= epsilon * velocity_x[i];
    minus[i].position_y -= epsilon * velocity_y[i];
  }
  const CompositeDepositResultR2 plus_deposit =
      deposit_composite_tent_source_once_r2(
          topology, build_composite_tent_stencils_r2(topology, plus));
  const CompositeDepositResultR2 minus_deposit =
      deposit_composite_tent_source_once_r2(
          topology, build_composite_tent_stencils_r2(topology, minus));
  const double finite_difference =
      (source_potential_dot(topology, plus_deposit.integrated_source,
                            potential, dirichlet) -
       source_potential_dot(topology, minus_deposit.integrated_source,
                            potential, dirichlet)) /
      (2.0 * epsilon);
  require_close(finite_difference, direct_variation, 3.0e-8,
                "finite-difference adjoint work identity");
}

void test_solver_reciprocity_and_force_diagnostics(
    const CompositePoissonR2& topology) {
  const CompositeTentParticleR2 left =
      particle(201U, -0.75, 0.0, 1.0, 0.55, 0.55);
  const CompositeTentParticleR2 right =
      particle(202U, 0.75, 0.0, 1.0, 0.55, 0.55);
  const SolvedSource a = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, left));
  const SolvedSource b = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, right));
  const double energy_ab = source_potential_dot(
      topology, a.deposit.integrated_source, b.potential, b.dirichlet);
  const double energy_ba = source_potential_dot(
      topology, b.deposit.integrated_source, a.potential, a.dirichlet);
  require_close(energy_ab, energy_ba, 2.0e-10,
                "symmetric Poisson cross-energy reciprocity");

  const CompositeAdjointGatherResultR2 field_a_from_b =
      gather_composite_tent_adjoint_r2(
          topology, a.stencils, b.potential, b.dirichlet);
  const CompositeAdjointGatherResultR2 field_b_from_a =
      gather_composite_tent_adjoint_r2(
          topology, b.stencils, a.potential, a.dirichlet);
  const double action_reaction_x =
      left.signed_charge * field_a_from_b.samples[0].electric_x +
      right.signed_charge * field_b_from_a.samples[0].electric_x;
  const double action_reaction_y =
      left.signed_charge * field_a_from_b.samples[0].electric_y +
      right.signed_charge * field_b_from_a.samples[0].electric_y;
  require(std::fabs(action_reaction_x) < 2.0e-9 &&
          std::fabs(action_reaction_y) < 2.0e-9,
          "mirror-pair action/reaction diagnostic");

  const CompositeTentParticleR2 self =
      particle(301U, 0.31, -0.27, 1.3, 0.61, 0.53);
  const SolvedSource center = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, self));
  const CompositeAdjointGatherResultR2 self_field =
      gather_composite_tent_adjoint_r2(
          topology, center.stencils, center.potential, center.dirichlet);
  const double force_x = self.signed_charge *
      self_field.samples[0].electric_x;
  const double epsilon = 2.0e-5;
  CompositeTentParticleR2 plus_particle = self;
  CompositeTentParticleR2 minus_particle = self;
  plus_particle.position_x += epsilon;
  minus_particle.position_x -= epsilon;
  const SolvedSource plus = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, plus_particle));
  const SolvedSource minus = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, minus_particle));
  const double plus_energy = 0.5 * source_potential_dot(
      topology, plus.deposit.integrated_source,
      plus.potential, plus.dirichlet);
  const double minus_energy = 0.5 * source_potential_dot(
      topology, minus.deposit.integrated_source,
      minus.potential, minus.dirichlet);
  const double force_finite_difference =
      -(plus_energy - minus_energy) / (2.0 * epsilon);
  require_close(force_x, force_finite_difference, 2.0e-5,
                "self-force equals energy-gradient diagnostic");

  // A centered source in this exactly reflection-symmetric domain and patch
  // has an analytic zero self-force.  This is a physical magnitude oracle,
  // independent of the energy-gradient identity above.
  const CompositeTentParticleR2 symmetric_self =
      particle(302U, 0.0, 0.0, 1.0, 0.50, 0.50);
  const SolvedSource symmetric = solve_particles(
      topology, std::vector<CompositeTentParticleR2>(1U, symmetric_self));
  const CompositeAdjointGatherResultR2 symmetric_field =
      gather_composite_tent_adjoint_r2(
          topology, symmetric.stencils, symmetric.potential,
          symmetric.dirichlet);
  const double symmetric_self_force = std::hypot(
      symmetric_field.samples[0].electric_x,
      symmetric_field.samples[0].electric_y);
  require(symmetric_self_force < 2.0e-12,
          "reflection-symmetric source has non-negligible self-force");
  std::cout << "reciprocity_defect=" << std::fabs(energy_ab - energy_ba)
            << " action_reaction_x=" << action_reaction_x
            << " self_force_defect="
            << std::fabs(force_x - force_finite_difference)
            << " symmetric_self_force=" << symmetric_self_force << '\n';
}

}  // namespace

int main() {
  try {
    const CompositePoissonR2 topology(test_geometry());
    test_source_once_and_charge_moments(topology);
    test_translation_and_phase_sweep(topology);
    test_permutation_and_rejection(topology);
    test_adjoint_work_identity(topology);
    test_solver_reciprocity_and_force_diagnostics(topology);
    std::cout << "adaptive composite coupling tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "adaptive composite coupling tests: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
