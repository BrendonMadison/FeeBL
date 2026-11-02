#include "electronLumiSpatialCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace spatial = feebl::electron_lumi_spatial;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void check_close(double actual, double expected, double relative_tolerance,
                 const char* message) {
  const double scale = std::max(std::fabs(actual), std::fabs(expected));
  const double error = std::fabs(actual - expected);
  if (error > relative_tolerance * std::max(1.0, scale)) {
    std::cerr << "FAIL: " << message << " actual=" << actual
              << " expected=" << expected << " error=" << error
              << std::endl;
    std::exit(1);
  }
}

template <typename Callable>
void expect_error(Callable&& callable, spatial::SpatialErrorCode expected,
                  const char* message) {
  try {
    callable();
  } catch (const spatial::SpatialError& error) {
    if (error.code() == expected) return;
    std::cerr << "FAIL: " << message << " wrong error code, message: "
              << error.what() << std::endl;
    std::exit(1);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << message << " wrong exception: " << error.what()
              << std::endl;
    std::exit(1);
  }
  std::cerr << "FAIL: " << message << " did not throw" << std::endl;
  std::exit(1);
}

spatial::PhysicalCicGeometry unit_geometry(std::size_t n_x = 6U,
                                           std::size_t n_y = 6U) {
  return spatial::PhysicalCicGeometry(
      {n_x, n_y, 0.0, static_cast<double>(n_x - 1U), 0.0,
       static_cast<double>(n_y - 1U)});
}

double pair_sum(const spatial::CollisionPlan& plan) {
  long double result = 0.0L;
  for (const spatial::CollisionPair& pair : plan.pairs())
    result += static_cast<long double>(pair.weight);
  return static_cast<double>(result);
}

const spatial::CollisionPair& find_pair(const spatial::CollisionPlan& plan,
                                        spatial::RootId beam1,
                                        spatial::RootId beam2) {
  const auto found = std::find_if(
      plan.pairs().begin(), plan.pairs().end(),
      [beam1, beam2](const spatial::CollisionPair& pair) {
        return pair.beam1_id == beam1 && pair.beam2_id == beam2;
      });
  check(found != plan.pairs().end(), "expected physical root pair is absent");
  return *found;
}

struct ManualNodeFraction {
  std::size_t x;
  std::size_t y;
  long double fraction;
};

std::vector<ManualNodeFraction> manual_stencil(
    const spatial::PhysicalCicGeometry& geometry,
    const spatial::PhysicalRoot& root) {
  const auto axis = [](double coordinate, double minimum, double maximum,
                       double spacing, std::size_t nodes) {
    std::pair<std::size_t, long double> result;
    if (coordinate == maximum) {
      result.first = nodes - 2U;
      result.second = 1.0L;
    } else {
      const long double scaled =
          (static_cast<long double>(coordinate) - minimum) / spacing;
      const long double base = std::floor(scaled);
      result.first = static_cast<std::size_t>(base);
      result.second = scaled - base;
    }
    return result;
  };
  const auto x = axis(root.x, geometry.x_min(), geometry.x_max(),
                      geometry.delta_x(), geometry.n_x());
  const auto y = axis(root.y, geometry.y_min(), geometry.y_max(),
                      geometry.delta_y(), geometry.n_y());
  const long double lx = 1.0L - x.second;
  const long double ly = 1.0L - y.second;
  return {{x.first, y.first, lx * ly},
          {x.first + 1U, y.first, x.second * ly},
          {x.first, y.first + 1U, lx * y.second},
          {x.first + 1U, y.first + 1U, x.second * y.second}};
}

long double manual_overlap(const spatial::PhysicalCicGeometry& geometry,
                           const spatial::PhysicalRoot& left,
                           const spatial::PhysicalRoot& right) {
  const std::vector<ManualNodeFraction> a = manual_stencil(geometry, left);
  const std::vector<ManualNodeFraction> b = manual_stencil(geometry, right);
  long double result = 0.0L;
  for (const ManualNodeFraction& x : a)
    for (const ManualNodeFraction& y : b)
      if (x.x == y.x && x.y == y.y) result += x.fraction * y.fraction;
  return result;
}

void test_geometry_and_support_vectors() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry();
  check(geometry.n_x() == 6U && geometry.n_y() == 6U,
        "geometry dimensions changed");
  check(geometry.node_count() == 36U, "geometry node count changed");
  check(geometry.delta_x() == 1.0 && geometry.delta_y() == 1.0,
        "unit geometry spacing changed");
  check(geometry.control_area() == 1.0,
        "unit geometry control area changed");
  check(geometry.linear_node(2U, 3U) == 20U,
        "linear node convention changed");

  const struct Fixture {
    spatial::PhysicalRoot root;
    std::array<double, 4U> fractions;
  } fixtures[] = {
      {{11U, 2.0, 3.0, 1.0}, {1.0, 0.0, 0.0, 0.0}},
      {{12U, 2.5, 3.0, 1.0}, {0.5, 0.5, 0.0, 0.0}},
      {{13U, 2.0, 3.5, 1.0}, {0.5, 0.0, 0.5, 0.0}},
      {{14U, 2.25, 3.75, 1.0},
       {3.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0, 3.0 / 16.0}}};
  for (const Fixture& fixture : fixtures) {
    const spatial::AdmissionAudit audit =
        spatial::audit_admission(geometry, {fixture.root});
    check(audit.accepted.size() == 1U && audit.rejected.empty(),
          "in-domain support fixture was rejected");
    const spatial::RootStencil& stencil = audit.accepted.front();
    check(stencil.base_x == 2U && stencil.base_y == 3U,
          "support fixture base cell changed");
    check(stencil.fraction_sum() == 1.0,
          "dyadic CIC support does not sum exactly to one");
    for (std::size_t index = 0U; index < 4U; ++index)
      check(stencil.nodes[index].fraction == fixture.fractions[index],
            "CIC support fraction changed");
  }

  const spatial::PhysicalCicGeometry positive_zero(
      {4U, 4U, 0.0, 3.0, 0.0, 3.0});
  const spatial::PhysicalCicGeometry signed_zero(
      {4U, 4U, -0.0, 3.0, -0.0, 3.0});
  check(positive_zero.geometry_hash() == signed_zero.geometry_hash(),
        "signed zero forked physical geometry identity");
  const spatial::PhysicalCicGeometry changed(
      {4U, 4U, 0.0, 3.0, 0.0, 4.0});
  check(positive_zero.geometry_hash() != changed.geometry_hash(),
        "physical geometry hash ignored a bound change");
}

void test_full_support_boundaries_and_accounting() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry(4U, 4U);
  const double below = std::nextafter(0.0, -1.0);
  const double above = std::nextafter(3.0, 4.0);
  const std::vector<spatial::PhysicalRoot> roots = {
      {1U, 0.0, 0.0, 1.0}, {2U, below, 1.0, 2.0},
      {3U, above, 1.0, 4.0}, {4U, 1.25, 2.5, 8.0}};
  const spatial::AdmissionAudit audit =
      spatial::audit_admission(geometry, roots);
  check(audit.accounting.input_roots == 4U &&
            audit.accounting.accepted_roots == 2U &&
            audit.accounting.rejected_roots == 2U,
        "whole-root admission counts do not close");
  check(audit.accounting.input_population == 15.0 &&
            audit.accounting.accepted_population == 9.0 &&
            audit.accounting.rejected_population == 6.0,
        "whole-root admission population accounting changed");
  check(audit.rejected[0].id == 2U &&
            spatial::has_reason(audit.rejected[0].reason,
                                spatial::RejectionReason::BelowX),
        "low-bound rejection reason changed");
  check(audit.rejected[1].id == 3U &&
            spatial::has_reason(audit.rejected[1].reason,
                                spatial::RejectionReason::AboveX),
        "high-bound rejection reason changed");
  expect_error(
      [&]() { spatial::CollisionPlan::build(geometry, roots, {}); },
      spatial::SpatialErrorCode::OutOfDomainRoot,
      "strict plan did not fail closed on a rejected root");

  const double just_inside_low = std::nextafter(0.0, 1.0);
  const double just_inside_high = std::nextafter(3.0, 2.0);
  const std::vector<spatial::PhysicalRoot> boundary_roots = {
      {10U, 0.0, 0.0, 1.0},
      {11U, 3.0, 3.0, 1.0},
      {12U, just_inside_low, just_inside_low, 1.0},
      {13U, just_inside_high, just_inside_high, 1.0}};
  const spatial::AdmissionAudit boundary =
      spatial::audit_admission(geometry, boundary_roots);
  check(boundary.accepted.size() == boundary_roots.size() &&
            boundary.rejected.empty(),
        "inclusive/nextafter-inside boundary was rejected");
  check(boundary.accepted[0].base_x == 0U &&
            boundary.accepted[0].base_y == 0U &&
            boundary.accepted[0].nodes[0].fraction == 1.0,
        "lower endpoint CIC mapping changed");
  check(boundary.accepted[1].base_x == 2U &&
            boundary.accepted[1].base_y == 2U &&
            boundary.accepted[1].nodes[3].fraction == 1.0,
        "upper endpoint was not canonicalized to the last complete stencil");

  expect_error(
      [&]() {
        spatial::audit_admission(
            geometry, {{50U, below, 0.0, 1.0}, {50U, above, 0.0, 1.0}});
      },
      spatial::SpatialErrorCode::DuplicateRootId,
      "duplicate rejected roots escaped identity preflight");
  expect_error(
      [&]() {
        spatial::audit_admission(
            geometry,
            {{0U, below, 0.0, std::numeric_limits<double>::quiet_NaN()}});
      },
      spatial::SpatialErrorCode::InvalidRoot,
      "invalid rejected root escaped fail-closed validation");
}

void test_exact_aggregate_and_density_closure() {
  const spatial::PhysicalCicGeometry geometry(
      {4U, 4U, 10.0, 16.0, -4.0, 8.0});
  check(geometry.delta_x() == 2.0 && geometry.delta_y() == 4.0 &&
            geometry.control_area() == 8.0,
        "exact fixture geometry changed");
  const spatial::RootId a1 = 190U;
  const spatial::RootId a2 = 110U;
  const spatial::RootId b1 = 280U;
  const spatial::RootId b2 = 220U;
  const std::vector<spatial::PhysicalRoot> beam1 = {
      {a1, 12.5, 3.0, 2.0}, {a2, 14.0, 4.0, 3.0}};
  const std::vector<spatial::PhysicalRoot> beam2 = {
      {b1, 13.5, 1.0, 5.0}, {b2, 14.0, 4.0, 7.0}};
  const spatial::CollisionPlan plan =
      spatial::CollisionPlan::build(geometry, beam1, beam2);
  check(plan.pairs().size() == 4U,
        "four-pair exact fixture did not emit exactly four root pairs");
  check(find_pair(plan, a1, b1).weight == 45.0 / 256.0,
        "four-node aggregate pair weight changed");
  check(find_pair(plan, a1, b2).weight == 21.0 / 64.0,
        "A1-B2 exact pair weight changed");
  check(find_pair(plan, a2, b1).weight == 45.0 / 128.0,
        "A2-B1 exact pair weight changed");
  check(find_pair(plan, a2, b2).weight == 21.0 / 8.0,
        "A2-B2 exact pair weight changed");
  check(find_pair(plan, a1, b1).stencil_overlap == 9.0 / 64.0,
        "four common nodes were not aggregated once per physical pair");
  check(pair_sum(plan) == 891.0 / 256.0 &&
            plan.density_dot_oracle().weight == 891.0 / 256.0,
        "exact pair sum and independently deposited density dot diverged");
  check(plan.closure().within_tolerance &&
            plan.closure().relative_error == 0.0,
        "dyadic exact fixture did not close bit-for-bit");

  for (std::size_t index = 1U; index < plan.pairs().size(); ++index) {
    const spatial::CollisionPair& previous = plan.pairs()[index - 1U];
    const spatial::CollisionPair& current = plan.pairs()[index];
    check(previous.beam1_id < current.beam1_id ||
              (previous.beam1_id == current.beam1_id &&
               previous.beam2_id < current.beam2_id),
          "collision pair stream is not in canonical ordered-root order");
  }

  const spatial::CollisionPlan coincident_half =
      spatial::CollisionPlan::build(unit_geometry(4U, 4U),
                                    {{1U, 0.5, 0.5, 1.0}},
                                    {{2U, 0.5, 0.5, 1.0}});
  const spatial::CollisionPlan coincident_node =
      spatial::CollisionPlan::build(unit_geometry(4U, 4U),
                                    {{1U, 1.0, 1.0, 1.0}},
                                    {{2U, 1.0, 1.0, 1.0}});
  check(coincident_half.pairs()[0].spatial_kernel == 1.0 / 4.0 &&
            coincident_node.pairs()[0].spatial_kernel == 1.0,
        "raster CIC kernel identity was confused with a continuous-shape kernel");
}

void test_duplicate_free_3x3_enumeration() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry();
  std::vector<spatial::PhysicalRoot> beam2;
  spatial::RootId id = 100U;
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx)
      beam2.push_back({id++, 2.5 + static_cast<double>(dx),
                       2.5 + static_cast<double>(dy), 1.0});
  const spatial::CollisionPlan plan = spatial::CollisionPlan::build(
      geometry, {{1U, 2.5, 2.5, 1.0}}, beam2);
  check(plan.pairs().size() == 9U,
        "3x3 base-neighbor enumeration emitted the wrong pair count");
  std::size_t center = 0U;
  std::size_t axial = 0U;
  std::size_t diagonal = 0U;
  std::set<std::pair<spatial::RootId, spatial::RootId>> keys;
  double overlap_sum = 0.0;
  for (const spatial::CollisionPair& pair : plan.pairs()) {
    check(keys.insert({pair.beam1_id, pair.beam2_id}).second,
          "3x3 enumeration duplicated a physical root pair");
    overlap_sum += pair.stencil_overlap;
    if (pair.stencil_overlap == 1.0 / 4.0)
      ++center;
    else if (pair.stencil_overlap == 1.0 / 8.0)
      ++axial;
    else if (pair.stencil_overlap == 1.0 / 16.0)
      ++diagonal;
    else
      check(false, "3x3 enumeration produced an unexpected overlap");
  }
  check(center == 1U && axial == 4U && diagonal == 4U &&
            overlap_sum == 1.0,
        "3x3 overlap multiplicities changed");

  const spatial::CollisionPlan low_corner = spatial::CollisionPlan::build(
      geometry, {{1U, 0.5, 0.5, 1.0}},
      {{10U, 0.5, 0.5, 1.0}, {11U, 1.5, 0.5, 1.0},
       {12U, 0.5, 1.5, 1.0}, {13U, 1.5, 1.5, 1.0},
       {14U, 2.5, 2.5, 1.0}});
  check(low_corner.pairs().size() == 4U,
        "low-corner neighbor bounds wrapped or missed a valid bucket");
  const spatial::CollisionPlan high_corner = spatial::CollisionPlan::build(
      geometry, {{1U, 4.5, 4.5, 1.0}},
      {{10U, 4.5, 4.5, 1.0}, {11U, 3.5, 4.5, 1.0},
       {12U, 4.5, 3.5, 1.0}, {13U, 3.5, 3.5, 1.0},
       {14U, 1.5, 1.5, 1.0}});
  check(high_corner.pairs().size() == 4U,
        "high-corner neighbor bounds overflowed or missed a valid bucket");
}

void test_exhaustive_neighbor_algebra() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry(7U, 7U);
  const double fractions[] = {0.0, 0.25, 0.5, 0.75};
  std::size_t cases = 0U;
  for (double u1 : fractions) {
    for (double v1 : fractions) {
      for (double u2 : fractions) {
        for (double v2 : fractions) {
          for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
              const auto axis_overlap = [](double a, double b, int offset) {
                if (offset == 0)
                  return (1.0 - a) * (1.0 - b) + a * b;
                if (offset == 1) return a * (1.0 - b);
                if (offset == -1) return (1.0 - a) * b;
                return 0.0;
              };
              const double expected =
                  axis_overlap(u1, u2, dx) * axis_overlap(v1, v2, dy);
              const spatial::CollisionPlan plan =
                  spatial::CollisionPlan::build(
                      geometry, {{1U, 2.0 + u1, 2.0 + v1, 1.0}},
                      {{2U, 2.0 + static_cast<double>(dx) + u2,
                        2.0 + static_cast<double>(dy) + v2, 1.0}});
              if (expected == 0.0) {
                check(plan.pairs().empty(),
                      "non-neighbor/zero-support pair was materialized");
              } else {
                check(plan.pairs().size() == 1U,
                      "positive CIC neighbor was not emitted exactly once");
                check(plan.pairs()[0].stencil_overlap == expected,
                      "exhaustive separable CIC overlap changed");
              }
              ++cases;
            }
          }
        }
      }
    }
  }
  check(cases == 6400U, "exhaustive CIC table did not cover 6400 cases");
}

void test_identity_permutation_and_scaling() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry(8U, 8U);
  std::vector<spatial::PhysicalRoot> beam1 = {
      {19U, 2.25, 3.75, 2.0}, {11U, 1.5, 1.5, 3.0},
      {17U, 5.0, 4.5, 4.0}, {13U, 3.25, 2.0, 5.0}};
  std::vector<spatial::PhysicalRoot> beam2 = {
      {29U, 2.75, 3.25, 7.0}, {23U, 1.0, 1.25, 11.0},
      {31U, 5.5, 4.25, 13.0}, {27U, 3.0, 2.5, 17.0}};
  const spatial::CollisionPlan reference =
      spatial::CollisionPlan::build(geometry, beam1, beam2);
  std::mt19937 generator(0x5eedU);
  for (unsigned int repetition = 0U; repetition < 100U; ++repetition) {
    std::shuffle(beam1.begin(), beam1.end(), generator);
    std::shuffle(beam2.begin(), beam2.end(), generator);
    const spatial::CollisionPlan shuffled =
        spatial::CollisionPlan::build(geometry, beam1, beam2);
    check(shuffled.plan_hash() == reference.plan_hash(),
          "input permutation changed the physical plan hash");
    check(shuffled.pairs() == reference.pairs(),
          "input permutation changed the canonical pair stream");
    check(shuffled.closure().pair_weight_sum ==
              reference.closure().pair_weight_sum,
          "input permutation changed the deterministic reduction");
  }

  const spatial::CollisionPlan swapped =
      spatial::CollisionPlan::build(geometry, beam2, beam1);
  check(swapped.closure().pair_weight_sum ==
            reference.closure().pair_weight_sum &&
            swapped.density_dot_oracle().weight ==
                reference.density_dot_oracle().weight,
        "beam exchange changed the scalar physical overlap");

  for (spatial::PhysicalRoot& root : beam1) root.population *= 3.0;
  for (spatial::PhysicalRoot& root : beam2) root.population *= 5.0;
  const spatial::CollisionPlan scaled =
      spatial::CollisionPlan::build(geometry, beam1, beam2);
  check_close(scaled.closure().pair_weight_sum,
              15.0 * reference.closure().pair_weight_sum, 1.0e-15,
              "beam population scaling is not bilinear");

  const std::vector<spatial::PhysicalRoot> base1 =
      {{1U, 0.5, 0.5, 2.0}};
  const std::vector<spatial::PhysicalRoot> base2 =
      {{2U, 0.5, 0.5, 3.0}};
  const spatial::CollisionPlan unit = spatial::CollisionPlan::build(
      unit_geometry(4U, 4U), base1, base2);
  const spatial::PhysicalCicGeometry double_x(
      {4U, 4U, 0.0, 6.0, 0.0, 3.0});
  const spatial::CollisionPlan stretched = spatial::CollisionPlan::build(
      double_x, {{1U, 1.0, 0.5, 2.0}}, {{2U, 1.0, 0.5, 3.0}});
  check(stretched.closure().pair_weight_sum ==
            unit.closure().pair_weight_sum / 2.0,
        "doubling physical x spacing did not halve the collision kernel");

  const spatial::PhysicalCicGeometry translated(
      {4U, 4U, 32.0, 35.0, -16.0, -13.0});
  const spatial::CollisionPlan moved = spatial::CollisionPlan::build(
      translated, {{1U, 32.5, -15.5, 2.0}},
      {{2U, 32.5, -15.5, 3.0}});
  check(moved.pairs()[0].stencil_overlap ==
            unit.pairs()[0].stencil_overlap &&
            moved.closure().pair_weight_sum ==
                unit.closure().pair_weight_sum,
        "exact physical translation changed CIC weights");

  const spatial::CollisionPlan zero_a = spatial::CollisionPlan::build(
      unit_geometry(4U, 4U), {{1U, -0.0, -0.0, 1.0}},
      {{2U, 0.0, 0.0, 1.0}});
  const spatial::CollisionPlan zero_b = spatial::CollisionPlan::build(
      unit_geometry(4U, 4U), {{1U, 0.0, 0.0, 1.0}},
      {{2U, -0.0, -0.0, 1.0}});
  check(zero_a.plan_hash() == zero_b.plan_hash(),
        "signed-zero root coordinates forked plan identity");
}

void test_empty_and_disjoint_populations() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry(8U, 8U);
  const spatial::CollisionPlan empty =
      spatial::CollisionPlan::build(geometry, {}, {});
  check(empty.pairs().empty() && empty.closure().pair_weight_sum == 0.0 &&
            empty.density_dot_oracle().weight == 0.0 &&
            empty.closure().within_tolerance,
        "empty collision plan did not close exactly");
  const spatial::CollisionPlan one_empty = spatial::CollisionPlan::build(
      geometry, {{1U, 1.5, 1.5, 1.0}}, {});
  check(one_empty.pairs().empty() &&
            one_empty.density_dot_oracle().weight == 0.0,
        "one-empty-beam plan retained stale collision pairs");
  const spatial::CollisionPlan disjoint = spatial::CollisionPlan::build(
      geometry, {{1U, 0.25, 0.25, 1.0}},
      {{2U, 6.75, 6.75, 1.0}});
  check(disjoint.pairs().empty() &&
            disjoint.density_dot_oracle().weight == 0.0,
        "spatially disjoint roots produced luminosity");
}

std::uint64_t lcg_next(std::uint64_t& state) {
  state = state * UINT64_C(6364136223846793005) +
          UINT64_C(1442695040888963407);
  return state;
}

void test_randomized_independent_oracles() {
  const spatial::PhysicalCicGeometry geometry = unit_geometry(33U, 29U);
  std::uint64_t state = UINT64_C(0x3141592653589793);
  const auto make_roots = [&state](std::size_t count, spatial::RootId first_id,
                                   std::size_t max_base_x,
                                   std::size_t max_base_y) {
    std::vector<spatial::PhysicalRoot> roots;
    roots.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
      const std::size_t base_x = lcg_next(state) % max_base_x;
      const std::size_t base_y = lcg_next(state) % max_base_y;
      const double u = static_cast<double>(lcg_next(state) % 8U) / 8.0;
      const double v = static_cast<double>(lcg_next(state) % 8U) / 8.0;
      const double population =
          static_cast<double>(1U + lcg_next(state) % 32U) / 8.0;
      roots.push_back({first_id + index, static_cast<double>(base_x) + u,
                       static_cast<double>(base_y) + v, population});
    }
    return roots;
  };
  const std::vector<spatial::PhysicalRoot> beam1 =
      make_roots(257U, 1000U, 32U, 28U);
  const std::vector<spatial::PhysicalRoot> beam2 =
      make_roots(263U, 1000000U, 32U, 28U);
  spatial::CollisionPlanOptions options;
  options.normalization = 8.0;
  options.closure_relative_tolerance = 1.0e-14;
  const spatial::CollisionPlan plan =
      spatial::CollisionPlan::build(geometry, beam1, beam2, options);

  using PairKey = std::pair<spatial::RootId, spatial::RootId>;
  std::map<PairKey, double> brute_pairs;
  long double brute_sum = 0.0L;
  for (const spatial::PhysicalRoot& left : beam1) {
    for (const spatial::PhysicalRoot& right : beam2) {
      const long double overlap = manual_overlap(geometry, left, right);
      if (overlap == 0.0L) continue;
      const long double value =
          overlap * left.population * right.population *
          options.normalization / geometry.control_area();
      brute_pairs[{left.id, right.id}] = static_cast<double>(value);
      brute_sum += value;
    }
  }
  check(brute_pairs.size() == plan.pairs().size(),
        "3x3 production enumeration differs from O(N1*N2) brute force");
  std::set<PairKey> production_keys;
  for (const spatial::CollisionPair& pair : plan.pairs()) {
    const PairKey key{pair.beam1_id, pair.beam2_id};
    check(production_keys.insert(key).second,
          "randomized production stream duplicated a root pair");
    const auto expected = brute_pairs.find(key);
    check(expected != brute_pairs.end(),
          "randomized production stream emitted a false-positive pair");
    check_close(pair.weight, expected->second, 1.0e-15,
                "randomized aggregate pair weight differs from brute force");
  }

  std::vector<long double> density1(geometry.node_count(), 0.0L);
  std::vector<long double> density2(geometry.node_count(), 0.0L);
  const auto deposit = [&geometry](const std::vector<spatial::PhysicalRoot>& roots,
                                   std::vector<long double>* density) {
    for (const spatial::PhysicalRoot& root : roots) {
      for (const ManualNodeFraction& node : manual_stencil(geometry, root))
        (*density)[geometry.linear_node(node.x, node.y)] +=
            root.population * node.fraction;
    }
  };
  deposit(beam1, &density1);
  deposit(beam2, &density2);
  long double dense_dot = 0.0L;
  for (std::size_t node = 0U; node < density1.size(); ++node)
    dense_dot += density1[node] * density2[node];
  dense_dot *= options.normalization / geometry.control_area();
  check_close(plan.closure().pair_weight_sum,
              static_cast<double>(brute_sum), 2.0e-14,
              "production pair reduction differs from independent brute sum");
  check_close(plan.density_dot_oracle().weight,
              static_cast<double>(dense_dot), 2.0e-14,
              "sparse density-dot oracle differs from test-only dense deposit");
  check(plan.closure().within_tolerance,
        "randomized physical plan failed pair/density closure");
}

void test_numeric_resource_and_transaction_guards() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  expect_error(
      []() { spatial::PhysicalCicGeometry({1U, 2U, 0.0, 1.0, 0.0, 1.0}); },
      spatial::SpatialErrorCode::InvalidGeometry,
      "one-node geometry was accepted");
  expect_error(
      [nan]() { spatial::PhysicalCicGeometry({2U, 2U, nan, 1.0, 0.0, 1.0}); },
      spatial::SpatialErrorCode::InvalidGeometry,
      "NaN geometry was accepted");
  expect_error(
      [inf]() { spatial::PhysicalCicGeometry({2U, 2U, 0.0, inf, 0.0, 1.0}); },
      spatial::SpatialErrorCode::InvalidGeometry,
      "infinite geometry was accepted");
  expect_error(
      []() {
        spatial::PhysicalCicGeometry(
            {2U, 2U, -std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max(), 0.0, 1.0});
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "overflowing geometry width was accepted");
  expect_error(
      []() {
        spatial::PhysicalCicGeometry({2U, 2U, 0.0, 1.0e200, 0.0, 1.0e200});
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "overflowing physical control area was accepted");
  expect_error(
      []() {
        spatial::PhysicalCicGeometry({2U, 2U, 0.0, 1.0e-160, 0.0, 1.0e-160});
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "overflowing inverse physical area was accepted");
  expect_error(
      []() {
        const double origin = 9007199254740992.0;
        spatial::PhysicalCicGeometry(
            {3U, 2U, origin, origin + 2.0, 0.0, 1.0});
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "unresolvable adjacent physical nodes were accepted");
  if (sizeof(std::size_t) >= 8U) {
    expect_error(
        []() {
          spatial::PhysicalCicGeometry(
              {static_cast<std::size_t>(UINT64_C(1) << 40U),
               static_cast<std::size_t>(UINT64_C(1) << 40U),
               0.0, 1.0, 0.0, 1.0});
        },
        spatial::SpatialErrorCode::ArithmeticOverflow,
        "physical node-count product overflow was accepted");
    expect_error(
        []() {
          spatial::PhysicalCicGeometry(
              {static_cast<std::size_t>((UINT64_C(1) << 53U) + 2U),
               2U, 0.0, 1.0, 0.0, 1.0});
        },
        spatial::SpatialErrorCode::InvalidGeometry,
        "unrepresentable physical dimension was accepted");
  }

  const spatial::PhysicalCicGeometry geometry = unit_geometry(4U, 4U);
  const spatial::PhysicalRoot valid{1U, 0.5, 0.5, 1.0};
  const struct InvalidRootFixture {
    spatial::PhysicalRoot root;
  } invalid_roots[] = {{{0U, 0.5, 0.5, 1.0}},
                       {{1U, nan, 0.5, 1.0}},
                       {{1U, 0.5, inf, 1.0}},
                       {{1U, 0.5, 0.5, 0.0}},
                       {{1U, 0.5, 0.5, -1.0}},
                       {{1U, 0.5, 0.5, nan}},
                       {{1U, 0.5, 0.5, inf}}};
  for (const InvalidRootFixture& fixture : invalid_roots) {
    expect_error(
        [&]() { spatial::audit_admission(geometry, {fixture.root}); },
        spatial::SpatialErrorCode::InvalidRoot,
        "invalid physical root was accepted");
  }
  expect_error(
      [&]() { spatial::audit_admission(geometry, {valid, valid}); },
      spatial::SpatialErrorCode::DuplicateRootId,
      "within-beam duplicate root ID was accepted");
  expect_error(
      [&]() { spatial::CollisionPlan::build(geometry, {valid}, {valid}); },
      spatial::SpatialErrorCode::DuplicateRootId,
      "cross-beam duplicate root ID was accepted");

  spatial::CollisionPlanOptions zero_cap;
  zero_cap.max_pairs = 0U;
  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {valid}, {{2U, 0.5, 0.5, 1.0}}, zero_cap);
      },
      spatial::SpatialErrorCode::ResourceLimitExceeded,
      "pair resource cap committed a partial plan");

  spatial::CollisionPlanOptions bad_normalization;
  bad_normalization.normalization = 0.0;
  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {valid}, {{2U, 0.5, 0.5, 1.0}},
            bad_normalization);
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "zero collision normalization was accepted");
  bad_normalization.normalization = inf;
  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {valid}, {{2U, 0.5, 0.5, 1.0}},
            bad_normalization);
      },
      spatial::SpatialErrorCode::InvalidGeometry,
      "infinite collision normalization was accepted");

  spatial::CollisionPlanOptions bad_tolerance;
  bad_tolerance.closure_relative_tolerance = -1.0;
  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {valid}, {{2U, 0.5, 0.5, 1.0}}, bad_tolerance);
      },
      spatial::SpatialErrorCode::ClosureFailure,
      "negative closure tolerance was accepted");

  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {{1U, 0.5, 0.5, 1.0e308}},
            {{2U, 0.5, 0.5, 1.0e308}});
      },
      spatial::SpatialErrorCode::ArithmeticOverflow,
      "overflowing physical pair weight was accepted");
  expect_error(
      [&]() {
        spatial::audit_admission(
            geometry,
            {{1U, 0.5, 0.5, 1.0e308}, {2U, 0.5, 0.5, 1.0e308}});
      },
      spatial::SpatialErrorCode::ArithmeticOverflow,
      "overflowing beam population receipt was accepted");

  spatial::CollisionPlanOptions underflow;
  underflow.normalization = std::numeric_limits<double>::denorm_min();
  expect_error(
      [&]() {
        spatial::CollisionPlan::build(
            geometry, {valid}, {{2U, 0.5, 0.5, 1.0}}, underflow);
      },
      spatial::SpatialErrorCode::ArithmeticUnderflow,
      "positive physical pair-weight underflow was silently rounded to zero");

  const spatial::ClosureReport mismatch =
      spatial::evaluate_closure(1.0, 2.0, 0.1);
  check(!mismatch.within_tolerance && mismatch.relative_error == 0.5,
        "closure evaluator accepted a known mismatch");
  expect_error(
      [nan]() { spatial::evaluate_closure(nan, 1.0, 1.0e-12); },
      spatial::SpatialErrorCode::ClosureFailure,
      "NaN closure endpoint was accepted");

  spatial::AdmissionAudit audit = spatial::audit_admission(geometry, {valid});
  audit.accepted[0].nodes[0].fraction = nan;
  expect_error(
      [&]() {
        spatial::independently_deposited_density_dot(
            geometry, audit.accepted, {}, 1.0);
      },
      spatial::SpatialErrorCode::InternalInvariant,
      "malformed public CIC stencil was accepted by density oracle");

  audit = spatial::audit_admission(geometry, {valid});
  audit.accepted[0].nodes[0].fraction -= 0.125;
  audit.accepted[0].nodes[1].fraction += 0.125;
  expect_error(
      [&]() {
        spatial::independently_deposited_density_dot(
            geometry, audit.accepted, {}, 1.0);
      },
      spatial::SpatialErrorCode::InternalInvariant,
      "normalized but non-CIC public stencil was accepted by density oracle");
}

}  // namespace

int main() {
  test_geometry_and_support_vectors();
  test_full_support_boundaries_and_accounting();
  test_exact_aggregate_and_density_closure();
  test_duplicate_free_3x3_enumeration();
  test_exhaustive_neighbor_algebra();
  test_identity_permutation_and_scaling();
  test_empty_and_disjoint_populations();
  test_randomized_independent_oracles();
  test_numeric_resource_and_transaction_guards();
  std::cout << "electron luminosity spatial CIC unit: PASS" << std::endl;
  return 0;
}
