#include "adaptiveResolutionControllerCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void require_failure(Function function, const std::string& message) {
  bool failed = false;
  try {
    function();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, message);
}

feebl::adaptive_grid::CanonicalLatticeSnapshot lattice() {
  feebl::adaptive_grid::CanonicalLatticeInput input;
  input.n_x = 32U;
  input.n_y = 32U;
  input.delta_x = 2.0f;
  input.delta_y = 1.0f;
  input.delta_z = 3.0f;
  input.inverse_delta_x = 0.5f;
  input.inverse_delta_y = 1.0f;
  input.offset_x = 16.0f;
  input.offset_y = 16.0f;
  input.intended_min_x = -30.0f;
  input.intended_max_x = 30.0f;
  input.intended_min_y = -15.0f;
  input.intended_max_y = 15.0f;
  input.charge_sign = -1.0f;
  input.field_kernel_factor = 4.0;
  input.geometry_revision = 7U;
  return feebl::adaptive_grid::CanonicalLatticeSnapshot(input);
}

feebl::adaptive_grid::CanonicalWindowInputs window_inputs() {
  feebl::adaptive_grid::CanonicalWindowInputs inputs;
  inputs.beam1_sources.push_back(
      feebl::adaptive_grid::CanonicalDepositDescriptor::from_canonical(
          14, 14, 0.25f, 0.75f));
  inputs.beam2_sources.push_back(
      feebl::adaptive_grid::CanonicalDepositDescriptor::from_canonical(
          17, 16, 0.5f, 0.5f));
  inputs.beam1_targets.push_back(
      feebl::adaptive_grid::CanonicalGatherDescriptor::
          from_legacy_absolute_coordinates(15.25, 15.75));
  inputs.beam2_targets.push_back(
      feebl::adaptive_grid::CanonicalGatherDescriptor::
          from_legacy_absolute_coordinates(17.5, 16.5));
  return inputs;
}

feebl::adaptive_grid::UniformWindowPlan window(
    const feebl::adaptive_grid::CanonicalLatticeSnapshot& grid) {
  feebl::adaptive_grid::UniformWindowPolicy policy;
  policy.size_classes_x = {8U, 16U, 32U};
  policy.size_classes_y = {8U, 16U, 32U};
  policy.min_n_x = 8U;
  policy.min_n_y = 8U;
  policy.max_n_x = 32U;
  policy.max_n_y = 32U;
  return feebl::adaptive_grid::UniformWindowPlan::build(
      grid, window_inputs(), policy);
}

feebl::adaptive_grid::FieldLevelGeometry density_geometry(
    std::size_t factor, std::size_t global_n_x, std::size_t global_n_y,
    std::size_t origin_x, std::size_t origin_y,
    std::size_t n_x, std::size_t n_y,
    float delta_x, float delta_y, float offset_x, float offset_y) {
  feebl::adaptive_grid::FieldLevelGeometry geometry;
  geometry.refinement_factor = factor;
  geometry.global_n_x = global_n_x;
  geometry.global_n_y = global_n_y;
  geometry.origin_x = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      origin_x);
  geometry.origin_y = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      origin_y);
  geometry.n_x = n_x;
  geometry.n_y = n_y;
  geometry.delta_x = delta_x;
  geometry.delta_y = delta_y;
  geometry.inverse_delta_x = 1.0f / delta_x;
  geometry.inverse_delta_y = 1.0f / delta_y;
  geometry.offset_x = offset_x;
  geometry.offset_y = offset_y;
  require(geometry.valid(), "density test geometry is invalid");
  return geometry;
}

std::array<long double, 3> density_moments(
    const std::vector<double>& rho,
    const feebl::adaptive_grid::FieldLevelGeometry& geometry) {
  require(rho.size() == geometry.cells(),
          "density moment fixture has the wrong extent");
  std::array<long double, 3> result{{0.0L, 0.0L, 0.0L}};
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
    const long double x =
        (static_cast<long double>(geometry.origin_x) + ix + 0.5L -
         geometry.offset_x) * geometry.delta_x;
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
      const long double y =
          (static_cast<long double>(geometry.origin_y) + iy + 0.5L -
           geometry.offset_y) * geometry.delta_y;
      const long double q = rho[ix * geometry.n_y + iy];
      result[0] += q;
      result[1] += q * x;
      result[2] += q * y;
    }
  }
  return result;
}

long double tent_cdf(long double displacement, long double half_width) {
  if (displacement <= -half_width) return 0.0L;
  if (displacement < 0.0L) {
    const long double t = (displacement + half_width) / half_width;
    return 0.5L * t * t;
  }
  if (displacement < half_width) {
    const long double t = (half_width - displacement) / half_width;
    return 1.0L - 0.5L * t * t;
  }
  return 1.0L;
}

void check_bspline_axis(
    double coordinate,
    double half_width,
    feebl::adaptive_grid::FieldAxis axis,
    const feebl::adaptive_grid::FieldLevelGeometry& geometry) {
  const auto cells =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          coordinate, half_width, axis, geometry);
  const bool use_x = axis == feebl::adaptive_grid::FieldAxis::X;
  const std::size_t origin = static_cast<std::size_t>(
      use_x ? geometry.origin_x : geometry.origin_y);
  const long double delta = use_x ? geometry.delta_x : geometry.delta_y;
  const long double offset = use_x ? geometry.offset_x : geometry.offset_y;
  long double sum = 0.0L;
  long double first_moment = 0.0L;
  std::size_t previous = 0U;
  for (std::size_t entry = 0U; entry < cells.size(); ++entry) {
    require(cells[entry].weight > 0.0 &&
                std::isfinite(cells[entry].weight),
            "B-spline emitted a non-positive or non-finite weight");
    if (entry != 0U)
      require(cells[entry].local_index > previous,
              "B-spline cell indices are not strictly ascending");
    previous = cells[entry].local_index;
    const std::size_t global = origin + cells[entry].local_index;
    const long double cell_left =
        (static_cast<long double>(global) - offset) * delta;
    const long double cell_right = cell_left + delta;
    const long double expected =
        tent_cdf(cell_right - coordinate, half_width) -
        tent_cdf(cell_left - coordinate, half_width);
    require(std::fabs(static_cast<long double>(cells[entry].weight) -
                      expected) < 3.0e-16L,
            "B-spline cell weight disagrees with analytic CDF integral");
    sum += cells[entry].weight;
    first_moment += cells[entry].weight *
        (cell_left + 0.5L * delta);
  }
  require(std::fabs(sum - 1.0L) < 8.0e-16L,
          "B-spline weights do not normalize to unity");
  require(std::fabs(first_moment - coordinate) <=
              0.5L * delta + 2.0e-15L,
          "B-spline cell-centre first moment exceeds its geometric bound");
}

void test_fixed_width_bspline_cell_integration() {
  using feebl::adaptive_grid::FieldAxis;
  const auto geometry = density_geometry(
      1U, 64U, 48U, 8U, 6U, 48U, 36U,
      1.0f, 0.75f, 32.0f, 24.0f);

  // Independent CDF landmarks establish the normalization used by the
  // per-cell oracle below.
  const long double ell = 2.4L;
  require(tent_cdf(-ell, ell) == 0.0L &&
              std::fabs(tent_cdf(-0.5L * ell, ell) - 0.125L) < 1e-18L &&
              tent_cdf(0.0L, ell) == 0.5L &&
              std::fabs(tent_cdf(0.5L * ell, ell) - 0.875L) < 1e-18L &&
              tent_cdf(ell, ell) == 1.0L,
          "linear B-spline CDF landmarks are inconsistent");

  // Sweep particle phase across x and y cell faces, including exact faces
  // and cell centres.  The checker validates every individual CDF integral,
  // normalization, order, positivity, and the cell-centre moment bound.
  for (int phase = 0; phase <= 16; ++phase) {
    const double x = -3.0 + static_cast<double>(phase) / 16.0;
    const double y = 1.5 + 0.75 * static_cast<double>(phase) / 16.0;
    check_bspline_axis(x, 2.4, FieldAxis::X, geometry);
    check_bspline_axis(y, 1.35, FieldAxis::Y, geometry);
  }

  // Cell-centred and cell-face clouds are mirror symmetric.  The two 1-D
  // lists also form an explicitly normalized separable 2-D corner stencil.
  const double centred_x = -2.5;
  const auto centred =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          centred_x, 2.75, FieldAxis::X, geometry);
  for (std::size_t i = 0U; i < centred.size(); ++i)
    require(std::fabs(centred[i].weight -
                      centred[centred.size() - 1U - i].weight) < 2e-16,
            "cell-centred B-spline is not mirror symmetric");
  const double face_x = -2.0;
  const auto face =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          face_x, 2.75, FieldAxis::X, geometry);
  for (std::size_t i = 0U; i < face.size(); ++i)
    require(std::fabs(face[i].weight -
                      face[face.size() - 1U - i].weight) < 2e-16,
            "cell-face B-spline is not mirror symmetric");
  const auto y_cells =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          1.5, 1.35, FieldAxis::Y, geometry);
  long double product_sum = 0.0L;
  for (const auto& x_cell : face)
    for (const auto& y_cell : y_cells) {
      const long double corner_weight =
          static_cast<long double>(x_cell.weight) * y_cell.weight;
      require(corner_weight > 0.0L,
              "separable B-spline product has a non-positive corner");
      product_sum += corner_weight;
    }
  require(std::fabs(product_sum - 1.0L) < 1e-15L,
          "separable 2-D B-spline products do not normalize");

  // Refining a registered lattice must only partition each coarse control
  // volume: summing the two fine children recovers its coarse integral.
  const auto fine = density_geometry(
      2U, 128U, 96U, 16U, 12U, 96U, 72U,
      0.5f, 0.375f, 64.0f, 48.0f);
  const double refinement_coordinate = 0.371;
  const double refinement_width = 2.137;
  const auto coarse_weights =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          refinement_coordinate, refinement_width, FieldAxis::X, geometry);
  const auto fine_weights =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          refinement_coordinate, refinement_width, FieldAxis::X, fine);
  for (const auto& coarse_cell : coarse_weights) {
    const std::size_t coarse_global =
        static_cast<std::size_t>(geometry.origin_x) +
        coarse_cell.local_index;
    double children = 0.0;
    for (const auto& fine_cell : fine_weights) {
      const std::size_t fine_global =
          static_cast<std::size_t>(fine.origin_x) + fine_cell.local_index;
      if (fine_global / 2U == coarse_global) children += fine_cell.weight;
    }
    require(std::fabs(children - coarse_cell.weight) < 4e-16,
            "registered refinement changed a B-spline cell integral");
  }

  // Identical calls must return bit-identical weights and ordering.
  const auto repeated =
      feebl::adaptive_grid::integrate_first_order_bspline_cells(
          refinement_coordinate, refinement_width, FieldAxis::X, geometry);
  require(repeated.size() == coarse_weights.size(),
          "repeated B-spline call changed stencil size");
  for (std::size_t i = 0U; i < repeated.size(); ++i)
    require(repeated[i].local_index == coarse_weights[i].local_index &&
                repeated[i].weight == coarse_weights[i].weight,
            "repeated B-spline call was not deterministic");
  std::vector<feebl::adaptive_grid::BSplineCellIntegral> reusable(17U);
  feebl::adaptive_grid::integrate_first_order_bspline_cells(
      refinement_coordinate, refinement_width, FieldAxis::X, geometry,
      reusable);
  require(reusable.size() == coarse_weights.size(),
          "reusable B-spline output retained stale entries");
  for (std::size_t i = 0U; i < reusable.size(); ++i)
    require(reusable[i].local_index == coarse_weights[i].local_index &&
                reusable[i].weight == coarse_weights[i].weight,
            "reusable B-spline overload changed deterministic output");

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  for (double bad_coordinate : {nan, infinity, -infinity})
    require_failure([&] {
      (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
          bad_coordinate, 1.0, FieldAxis::X, geometry);
    }, "B-spline accepted a non-finite particle coordinate");
  for (double bad_width : {0.0, -1.0, nan, infinity})
    require_failure([&] {
      (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
          0.0, bad_width, FieldAxis::X, geometry);
    }, "B-spline accepted an invalid physical half-width");
  auto invalid_geometry = geometry;
  invalid_geometry.delta_x = 0.0f;
  require_failure([&] {
    (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
        0.0, 1.0, FieldAxis::X, invalid_geometry);
  }, "B-spline accepted invalid field geometry");
  require_failure([&] {
    (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
        0.0, 1.0, static_cast<FieldAxis>(17), geometry);
  }, "B-spline accepted an invalid axis");

  const auto global_geometry = density_geometry(
      1U, 16U, 16U, 0U, 0U, 16U, 16U,
      1.0f, 1.0f, 8.0f, 8.0f);
  require_failure([&] {
    (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
        -7.5, 0.75, FieldAxis::X, global_geometry);
  }, "B-spline accepted support escaping the global geometry");
  // Exact contact with the global boundary has no escaped positive support.
  check_bspline_axis(-7.5, 0.5, FieldAxis::X, global_geometry);

  const auto cropped_geometry = density_geometry(
      1U, 16U, 16U, 4U, 4U, 8U, 8U,
      1.0f, 1.0f, 8.0f, 8.0f);
  require_failure([&] {
    (void)feebl::adaptive_grid::integrate_first_order_bspline_cells(
        -3.5, 0.75, FieldAxis::X, cropped_geometry);
  }, "B-spline accepted support escaping the local window");
  // Exact contact with the local boundary is likewise valid.
  check_bspline_axis(-3.5, 0.5, FieldAxis::X, cropped_geometry);
}

void test_parsers() {
  using feebl::adaptive_grid::FieldOperator;
  require(feebl::adaptive_grid::parse_field_operator("legacy_cic12_v1") ==
              FieldOperator::LegacyCic12V1 &&
              feebl::adaptive_grid::parse_field_operator(
                  " fd4_lagrange4_v1 \n") ==
                  FieldOperator::Fd4Lagrange4V1,
          "field-operator parser changed a versioned name");
  require(std::string(feebl::adaptive_grid::field_operator_name(
                          FieldOperator::Fd4Lagrange4V1)) ==
              "fd4_lagrange4_v1" &&
              feebl::adaptive_grid::
                      field_operator_additional_window_guard_cells(
                          FieldOperator::Fd4Lagrange4V1) == 3U &&
              feebl::adaptive_grid::field_operator_transition_guard_cells(
                  FieldOperator::Fd4Lagrange4V1) == 4U,
          "field-operator name/support contract changed");
  require_failure([] {
    (void)feebl::adaptive_grid::parse_field_operator("fd4");
  }, "unversioned field operator was accepted");
  require_failure([] {
    (void)feebl::adaptive_grid::field_operator_name(
        static_cast<FieldOperator>(99));
  }, "unknown field operator enum was accepted");

  const std::vector<std::size_t> factors =
      feebl::adaptive_grid::parse_refinement_factors("1,2,4");
  require(factors == std::vector<std::size_t>({1U, 2U, 4U}),
          "spacing factor parser changed order");
  require_failure([] {
    (void)feebl::adaptive_grid::parse_refinement_factors("1,3");
  }, "non-power-of-two spacing factor was accepted");
  require_failure([] {
    (void)feebl::adaptive_grid::parse_refinement_factors("2,4");
  }, "spacing factors without canonical class were accepted");
  require(std::fabs(feebl::adaptive_grid::richardson_order_for_pair(
                        "1:2=1.75,2:4=1.9", 2U, 4U) - 1.9) < 1e-15,
          "pair-specific Richardson order lookup failed");
  require(std::fabs(feebl::adaptive_grid::richardson_order_for_pair(
                        "1:1.8,2:1.9,4:2.0", 2U, 4U) - 1.9) < 1e-15,
          "factor-specific Richardson order lookup failed");
  require(std::fabs(feebl::adaptive_grid::richardson_order_for_pair(
                        "2.0", 1U, 2U) - 2.0) < 1e-15,
          "scalar Richardson order lookup failed");

  const char* malformed_factors[] = {
      "", "1", "1,2,2", "1,4,2", "1,0,2", "1,-2", "1,+2",
      "1,,2", "1,2,", "1,184467440737095516160"};
  for (const char* value : malformed_factors)
    require_failure([value] {
      (void)feebl::adaptive_grid::parse_refinement_factors(value);
    }, std::string("malformed spacing factors were accepted: ") + value);

  require_failure([] {
    (void)feebl::adaptive_grid::richardson_order_for_pair(
        "1:2=2.0", 2U, 4U);
  }, "missing pair-specific Richardson order was accepted");
  require_failure([] {
    (void)feebl::adaptive_grid::richardson_order_for_pair(
        "1:0", 1U, 2U);
  }, "non-positive Richardson order was accepted");
  require_failure([] {
    (void)feebl::adaptive_grid::richardson_order_for_pair(
        "1:2=2.0,broken", 4U, 8U);
  }, "malformed Richardson order table was accepted");
}

void test_longitudinal_encounter_tail_budget() {
  using feebl::adaptive_grid::make_encounter_tail_budget;

  const auto zero_budget = make_encounter_tail_budget(
      std::vector<std::size_t>{1U, 2U},
      std::vector<std::size_t>{3U}, 0.0);
  require(zero_budget.exemption_mask ==
              std::vector<unsigned char>({0U, 0U}) &&
          zero_budget.selected_pairs == 0U &&
          zero_budget.selected_fraction == 0.0L,
          "zero tail budget selected a positive-weight pair");

  const auto zeros = make_encounter_tail_budget(
      std::vector<std::size_t>{0U, 2U},
      std::vector<std::size_t>{3U, 0U}, 0.0);
  require(zeros.exemption_mask ==
              std::vector<unsigned char>({1U, 1U, 0U, 1U}) &&
          zeros.selected_pairs == 3U &&
          zeros.selected_fraction == 0.0L,
          "zero-weight encounter pairs were not selected deterministically");

  const auto exact_fit = make_encounter_tail_budget(
      std::vector<std::size_t>{1U, 3U},
      std::vector<std::size_t>{1U}, 0.25);
  require(exact_fit.exemption_mask ==
              std::vector<unsigned char>({1U, 0U}) &&
          exact_fit.selected_pairs == 1U &&
          exact_fit.selected_fraction == 0.25L,
          "an encounter pair that exactly fit the budget was not selected");

  const auto next_excluded = make_encounter_tail_budget(
      std::vector<std::size_t>{1U, 1U, 2U},
      std::vector<std::size_t>{1U}, 0.30);
  require(next_excluded.exemption_mask ==
              std::vector<unsigned char>({1U, 0U, 0U}) &&
          next_excluded.selected_pairs == 1U &&
          next_excluded.selected_fraction == 0.25L,
          "selector admitted the next whole pair past its tail budget");

  const auto tied = make_encounter_tail_budget(
      std::vector<std::size_t>{1U, 1U},
      std::vector<std::size_t>{1U, 1U}, 0.50);
  require(tied.exemption_mask ==
              std::vector<unsigned char>({1U, 1U, 0U, 0U}) &&
          tied.selected_pairs == 2U &&
          tied.selected_fraction == 0.50L,
          "equal-weight encounter ties were not resolved in row-major order");

  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const auto overflow_safe = make_encounter_tail_budget(
      std::vector<std::size_t>{maximum, maximum},
      std::vector<std::size_t>{maximum, maximum}, 0.50);
  require(overflow_safe.exemption_mask ==
              std::vector<unsigned char>({1U, 1U, 0U, 0U}) &&
          overflow_safe.selected_pairs == 2U &&
          std::fabs(overflow_safe.selected_fraction - 0.50L) < 1.0e-18L,
          "maximum-size counts overflowed or changed deterministic ties");

  require_failure([] {
    (void)make_encounter_tail_budget(
        std::vector<std::size_t>(), std::vector<std::size_t>{1U}, 0.0);
  }, "empty beam 1 was accepted by the tail-budget selector");
  require_failure([] {
    (void)make_encounter_tail_budget(
        std::vector<std::size_t>{1U}, std::vector<std::size_t>(), 0.0);
  }, "empty beam 2 was accepted by the tail-budget selector");
  require_failure([] {
    (void)make_encounter_tail_budget(
        std::vector<std::size_t>{0U, 0U},
        std::vector<std::size_t>{1U}, 0.0);
  }, "zero beam 1 total was accepted by the tail-budget selector");
  require_failure([] {
    (void)make_encounter_tail_budget(
        std::vector<std::size_t>{1U},
        std::vector<std::size_t>{0U, 0U}, 0.0);
  }, "zero beam 2 total was accepted by the tail-budget selector");
  const double invalid_budgets[] = {
      -1.0e-6, 1.0, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for (double budget : invalid_budgets)
    require_failure([budget] {
      (void)make_encounter_tail_budget(
          std::vector<std::size_t>{1U},
          std::vector<std::size_t>{1U}, budget);
    }, "invalid longitudinal tail budget was accepted");
}

void test_exact_window_memory_cap_boundary() {
  const auto grid = lattice();
  feebl::adaptive_grid::UniformWindowPolicy policy;
  policy.size_classes_x = {8U, 16U, 32U};
  policy.size_classes_y = {8U, 16U, 32U};
  policy.min_n_x = 8U;
  policy.min_n_y = 8U;
  policy.max_n_x = 32U;
  policy.max_n_y = 32U;
  const auto reference = feebl::adaptive_grid::UniformWindowPlan::build(
      grid, window_inputs(), policy);
  require(!reference.empty() && reference.estimated_bytes() > 0U,
          "memory-cap fixture did not produce an allocated window");

  policy.max_bytes = reference.estimated_bytes();
  const auto exact = feebl::adaptive_grid::UniformWindowPlan::build(
      grid, window_inputs(), policy);
  require(exact.estimated_bytes() == policy.max_bytes,
          "plan at its exact declared memory cap was rejected or changed");

  --policy.max_bytes;
  require_failure([&] {
    (void)feebl::adaptive_grid::UniformWindowPlan::build(
        grid, window_inputs(), policy);
  }, "plan one byte over its declared memory cap was accepted");
}

void test_level_mapping_and_moments() {
  const auto grid = lattice();
  const auto plan = window(grid);
  const auto level = feebl::adaptive_grid::make_refined_level_geometry(
      grid, plan, 2U, 2U);
  require(level.valid() && level.refinement_factor == 2U,
          "refined level was not constructed");
  for (int i = -20; i <= 20; ++i) {
    const float x = 0.37f * static_cast<float>(i);
    const float y = 0.13f * static_cast<float>(i);
    const auto d = feebl::adaptive_grid::map_deposit_to_level(x, y, level);
    require(d.accepted, "central test particle escaped refined level");
    const double weights[4] = {
        (1.0 - d.weight_x) * (1.0 - d.weight_y),
        d.weight_x * (1.0 - d.weight_y),
        (1.0 - d.weight_x) * d.weight_y,
        d.weight_x * d.weight_y};
    require(std::fabs(weights[0] + weights[1] + weights[2] + weights[3] -
                      1.0) < 2e-7,
            "refined CIC weights do not close");
    const long gx = static_cast<long>(level.origin_x) + d.base_x;
    const long gy = static_cast<long>(level.origin_y) + d.base_y;
    const double x0 = (gx + 0.5 - level.offset_x) * level.delta_x;
    const double y0 = (gy + 0.5 - level.offset_y) * level.delta_y;
    const double deposited_x = x0 * (weights[0] + weights[2]) +
                               (x0 + level.delta_x) *
                                   (weights[1] + weights[3]);
    const double deposited_y = y0 * (weights[0] + weights[1]) +
                               (y0 + level.delta_y) *
                                   (weights[2] + weights[3]);
    require(std::fabs(deposited_x - x) < 2e-5 &&
            std::fabs(deposited_y - y) < 2e-5,
            "refined CIC failed first-moment reproduction");
    require(feebl::adaptive_grid::map_gather_to_level(x, y, level).accepted,
            "central gather stencil escaped refined level");
  }
}

void test_field_operator_mapping_guard_and_legacy_preservation() {
  using feebl::adaptive_grid::FieldOperator;
  const auto geometry = density_geometry(
      1U, 32U, 32U, 0U, 0U, 32U, 32U,
      1.0f, 1.0f, 0.0f, 0.0f);
  std::vector<double> phi(geometry.cells(), 0.0);
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy)
      phi[ix * geometry.n_y + iy] =
          std::sin(0.17 * static_cast<double>(ix)) +
          std::cos(0.11 * static_cast<double>(iy)) +
          0.003 * static_cast<double>(ix * iy);

  const float x = 10.125f;
  const float y = 12.375f;
  const auto legacy = feebl::adaptive_grid::map_gather_to_level(
      x, y, geometry);
  const auto legacy_dispatch = feebl::adaptive_grid::map_gather_to_level(
      x, y, geometry, FieldOperator::LegacyCic12V1);
  require(legacy.accepted && legacy_dispatch.accepted &&
              legacy.y_base_x == legacy_dispatch.y_base_x &&
              legacy.y_base_y == legacy_dispatch.y_base_y &&
              legacy.y_weight == legacy_dispatch.y_weight &&
              legacy.x_base_x == legacy_dispatch.x_base_x &&
              legacy.x_base_y == legacy_dispatch.x_base_y &&
              legacy.x_weight == legacy_dispatch.x_weight &&
              legacy.fractional_x == legacy_dispatch.fractional_x &&
              legacy.fractional_y == legacy_dispatch.fractional_y,
          "legacy gather dispatch changed its localization");
  const auto legacy_field = feebl::adaptive_grid::gather_level_field(
      phi.data(), geometry, legacy);
  const auto dispatched_field = feebl::adaptive_grid::gather_level_field(
      phi.data(), geometry, legacy_dispatch,
      FieldOperator::LegacyCic12V1);
  require(legacy_field[0] == dispatched_field[0] &&
              legacy_field[1] == dispatched_field[1],
          "legacy gather dispatch changed a field bit pattern");

  // The established central path remains valid at its exact footprint
  // extrema.  True global faces now use shifted interpolation and one-sided
  // FD4 closure instead of rejecting an otherwise physical target.
  require(feebl::adaptive_grid::map_gather_to_level(
              3.5f, 3.5f, geometry,
              FieldOperator::Fd4Lagrange4V1).accepted,
          "FD4/cubic rejected its exact lower support boundary");
  require(feebl::adaptive_grid::map_gather_to_level(
              27.5f, 27.5f, geometry,
              FieldOperator::Fd4Lagrange4V1).accepted,
          "FD4/cubic rejected its exact upper support boundary");
  require(feebl::adaptive_grid::map_gather_to_level(
              0.0f, 0.0f, geometry,
              FieldOperator::Fd4Lagrange4V1).accepted &&
              feebl::adaptive_grid::map_gather_to_level(
              std::nextafter(32.0f, 0.0f),
              std::nextafter(32.0f, 0.0f), geometry,
              FieldOperator::Fd4Lagrange4V1).accepted,
          "FD4/cubic rejected its global-face closure");
  const auto cropped = density_geometry(
      1U, 32U, 32U, 4U, 4U, 24U, 24U,
      1.0f, 1.0f, 0.0f, 0.0f);
  require(!feebl::adaptive_grid::map_gather_to_level(
               4.0f, 12.5f, cropped,
               FieldOperator::Fd4Lagrange4V1).accepted &&
              !feebl::adaptive_grid::map_gather_to_level(
               std::nextafter(28.0f, 4.0f), 12.5f, cropped,
               FieldOperator::Fd4Lagrange4V1).accepted,
          "FD4/cubic treated a cropped local face as a global boundary");

  const auto grid = lattice();
  const auto plan = window(grid);
  const auto legacy_level =
      feebl::adaptive_grid::make_refined_level_geometry(
          grid, plan, 1U, 0U);
  const auto fd4_level =
      feebl::adaptive_grid::make_refined_level_geometry(
          grid, plan, 1U, 0U, 0.0, 0.0,
          FieldOperator::Fd4Lagrange4V1);
  require(fd4_level.origin_x + 3 <= legacy_level.origin_x &&
              fd4_level.origin_y + 3 <= legacy_level.origin_y &&
              static_cast<std::size_t>(fd4_level.origin_x) + fd4_level.n_x >=
                  static_cast<std::size_t>(legacy_level.origin_x) +
                      legacy_level.n_x + 3U &&
              static_cast<std::size_t>(fd4_level.origin_y) + fd4_level.n_y >=
                  static_cast<std::size_t>(legacy_level.origin_y) +
                      legacy_level.n_y + 3U,
          "FD4/cubic refined geometry omitted its additional support guard");

  const auto high_descriptor = feebl::adaptive_grid::map_gather_to_level(
      x, y, geometry, FieldOperator::Fd4Lagrange4V1);
  require(high_descriptor.accepted,
          "central FD4/cubic descriptor was rejected");
  std::vector<double> nonfinite_phi(phi);
  nonfinite_phi[(high_descriptor.centred_base_x - 3) * geometry.n_y +
                high_descriptor.centred_base_y] =
      std::numeric_limits<double>::quiet_NaN();
  require_failure([&] {
    (void)feebl::adaptive_grid::gather_level_field(
        nonfinite_phi.data(), geometry, high_descriptor,
        FieldOperator::Fd4Lagrange4V1);
  }, "FD4/cubic gather accepted a non-finite potential");
  auto invalid_weights = high_descriptor;
  invalid_weights.lagrange_weights_x[0] =
      std::numeric_limits<double>::infinity();
  require_failure([&] {
    (void)feebl::adaptive_grid::gather_level_field(
        phi.data(), geometry, invalid_weights,
        FieldOperator::Fd4Lagrange4V1);
  }, "FD4/cubic gather accepted a non-finite interpolation weight");
}

void test_fd4_lagrange_polynomial_and_phase_reproduction() {
  using feebl::adaptive_grid::FieldOperator;
  const auto geometry = density_geometry(
      1U, 64U, 64U, 0U, 0U, 64U, 64U,
      0.25f, 0.375f, 0.0f, 0.0f);
  const auto potential = [](double x, double y) {
    const double px = 1.0 + 0.2 * x + 0.03 * x * x -
        0.004 * x * x * x + 0.0002 * x * x * x * x;
    const double py = -0.3 * y + 0.02 * y * y +
        0.001 * y * y * y - 0.0001 * y * y * y * y;
    return px + py + 2.0e-6 * x * x * x * y * y * y;
  };
  const auto exact_field = [](double x, double y) {
    const double derivative_x = 0.2 + 0.06 * x - 0.012 * x * x +
        0.0008 * x * x * x + 6.0e-6 * x * x * y * y * y;
    const double derivative_y = -0.3 + 0.04 * y + 0.003 * y * y -
        0.0004 * y * y * y + 6.0e-6 * x * x * x * y * y;
    return std::array<double, 2>{{0.5 * derivative_x,
                                  0.5 * derivative_y}};
  };
  std::vector<double> phi(geometry.cells(), 0.0);
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
    const double x = (static_cast<double>(ix) + 0.5) * geometry.delta_x;
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
      const double y = (static_cast<double>(iy) + 0.5) * geometry.delta_y;
      phi[ix * geometry.n_y + iy] = potential(x, y);
    }
  }

  // Sweep independent x/y phases through a cell.  Fourth-degree univariate
  // terms exercise FD4; the x^3 y^3 term exercises both cubic gather axes.
  for (int phase = 1; phase <= 9; ++phase) {
    const float x = static_cast<float>(
        (20.0 + 0.1 * static_cast<double>(phase) + 0.5) *
        geometry.delta_x);
    const float y = static_cast<float>(
        (17.0 + 0.1 * static_cast<double>(10 - phase) + 0.5) *
        geometry.delta_y);
    const auto descriptor = feebl::adaptive_grid::map_gather_to_level(
        x, y, geometry, FieldOperator::Fd4Lagrange4V1);
    require(descriptor.accepted,
            "polynomial phase sweep escaped the FD4/cubic level");
    const auto field = feebl::adaptive_grid::gather_level_field(
        phi.data(), geometry, descriptor,
        FieldOperator::Fd4Lagrange4V1);
    const auto exact = exact_field(static_cast<double>(x),
                                   static_cast<double>(y));
    require(std::fabs(static_cast<double>(field[0]) - exact[0]) < 2.0e-6 &&
                std::fabs(static_cast<double>(field[1]) - exact[1]) < 2.0e-6,
            "FD4/cubic did not reproduce its polynomial contract");
  }
}

void test_fd4_lagrange_global_boundary_closure() {
  using feebl::adaptive_grid::FieldOperator;
  const auto geometry = density_geometry(
      1U, 32U, 32U, 0U, 0U, 32U, 32U,
      0.25f, 0.375f, 0.0f, 0.0f);
  const auto potential = [](double x, double y) {
    const double px = 1.0 + 0.2 * x + 0.03 * x * x -
        0.004 * x * x * x + 0.0002 * x * x * x * x;
    const double py = -0.3 * y + 0.02 * y * y +
        0.001 * y * y * y - 0.0001 * y * y * y * y;
    return px + py + 2.0e-6 * x * x * x * y * y * y;
  };
  const auto exact_field = [](double x, double y) {
    const double derivative_x = 0.2 + 0.06 * x - 0.012 * x * x +
        0.0008 * x * x * x + 6.0e-6 * x * x * y * y * y;
    const double derivative_y = -0.3 + 0.04 * y + 0.003 * y * y -
        0.0004 * y * y * y + 6.0e-6 * x * x * x * y * y;
    return std::array<double, 2>{{0.5 * derivative_x,
                                  0.5 * derivative_y}};
  };
  std::vector<double> phi(geometry.cells(), 0.0);
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
    const double x = (static_cast<double>(ix) + 0.5) * geometry.delta_x;
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
      const double y = (static_cast<double>(iy) + 0.5) * geometry.delta_y;
      phi[ix * geometry.n_y + iy] = potential(x, y);
    }
  }

  // Coordinates are expressed in cell-centre index units.  This covers the
  // physical lower face (-0.5), both one-sided derivative nodes (0/1),
  // shifted fractional phases, and their reflected upper-face counterparts.
  const double n = static_cast<double>(geometry.n_x);
  const double phases[][2] = {
      {-0.5, -0.5}, {-0.25, 0.0}, {0.0, 1.0}, {0.375, n - 2.0},
      {1.0, n - 1.0}, {1.625, n - 0.875},
      {n - 2.0, 0.375}, {n - 1.0, 1.25},
      {n - 0.875, n - 0.625}};
  for (const auto& phase : phases) {
    const float x = static_cast<float>(
        (phase[0] + 0.5) * geometry.delta_x);
    const float y = static_cast<float>(
        (phase[1] + 0.5) * geometry.delta_y);
    const auto descriptor = feebl::adaptive_grid::map_gather_to_level(
        x, y, geometry, FieldOperator::Fd4Lagrange4V1);
    require(descriptor.accepted,
            "global-edge polynomial phase escaped FD4 closure");
    const auto field = feebl::adaptive_grid::gather_level_field(
        phi.data(), geometry, descriptor,
        FieldOperator::Fd4Lagrange4V1);
    const auto exact = exact_field(static_cast<double>(x),
                                   static_cast<double>(y));
    require(std::fabs(static_cast<double>(field[0]) - exact[0]) < 1.0e-5 &&
                std::fabs(static_cast<double>(field[1]) - exact[1]) < 1.0e-5,
            "global-face FD4/cubic closure failed polynomial reproduction");
  }
}

void test_fd4_lagrange_refinement_contraction() {
  using feebl::adaptive_grid::FieldOperator;
  const auto potential = [](double x, double y) {
    return std::sin(0.9 * x) * std::cos(0.7 * y) +
        0.2 * std::cos(0.4 * x + 0.3 * y);
  };
  const float target_x = 7.123f;
  const float target_y = 6.417f;
  const double exact_x = 0.5 *
      (0.9 * std::cos(0.9 * target_x) * std::cos(0.7 * target_y) -
       0.08 * std::sin(0.4 * target_x + 0.3 * target_y));
  const double exact_y = 0.5 *
      (-0.7 * std::sin(0.9 * target_x) * std::sin(0.7 * target_y) -
       0.06 * std::sin(0.4 * target_x + 0.3 * target_y));
  const float spacings[] = {0.8f, 0.4f, 0.2f};
  std::array<double, 3> errors{{0.0, 0.0, 0.0}};
  for (std::size_t level = 0U; level < 3U; ++level) {
    const auto geometry = density_geometry(
        1U << level, 128U, 128U, 0U, 0U, 128U, 128U,
        spacings[level], spacings[level], 0.0f, 0.0f);
    std::vector<double> phi(geometry.cells(), 0.0);
    for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
      const double x = (static_cast<double>(ix) + 0.5) * geometry.delta_x;
      for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
        const double y = (static_cast<double>(iy) + 0.5) * geometry.delta_y;
        phi[ix * geometry.n_y + iy] = potential(x, y);
      }
    }
    const auto descriptor = feebl::adaptive_grid::map_gather_to_level(
        target_x, target_y, geometry, FieldOperator::Fd4Lagrange4V1);
    require(descriptor.accepted,
            "refinement fixture escaped the FD4/cubic level");
    const auto field = feebl::adaptive_grid::gather_level_field(
        phi.data(), geometry, descriptor,
        FieldOperator::Fd4Lagrange4V1);
    errors[level] = std::hypot(
        static_cast<double>(field[0]) - exact_x,
        static_cast<double>(field[1]) - exact_y);
  }
  const double contraction_1 = errors[0] / errors[1];
  const double contraction_2 = errors[1] / errors[2];
  require(errors[0] > errors[1] && errors[1] > errors[2] &&
              contraction_1 > 8.0 && contraction_1 < 32.0 &&
              contraction_2 > 8.0 && contraction_2 < 32.0,
          "FD4/cubic manufactured field did not contract at fourth order");
}

void test_coherent_density_conservation_alignment_and_scale() {
  const auto source_geometry = density_geometry(
      1U, 40U, 32U, 5U, 4U, 28U, 22U,
      1.25f, 0.75f, 19.3f, 15.6f);
  std::vector<double> source(source_geometry.cells(), 0.0);
  // A sparse, deliberately asymmetric and globally shifted source exercises
  // nonzero local origins while leaving guard cells on every window face.
  const std::size_t sparse_nodes[][2] = {
      {3U, 4U}, {6U, 13U}, {11U, 7U}, {17U, 15U}, {23U, 5U}};
  for (std::size_t i = 0U; i < 5U; ++i)
    source[sparse_nodes[i][0] * source_geometry.n_y + sparse_nodes[i][1]] =
        0.25 + 0.37 * static_cast<double>(i + 1U);
  for (std::size_t ix = 2U; ix + 2U < source_geometry.n_x; ++ix)
    for (std::size_t iy = 2U; iy + 2U < source_geometry.n_y; ++iy) {
      const double dx = static_cast<double>(ix) - 12.35;
      const double dy = static_cast<double>(iy) - 9.2;
      source[ix * source_geometry.n_y + iy] +=
          0.03 * std::exp(-(dx * dx / 27.0 + dy * dy / 11.0));
    }

  std::vector<double> identity;
  const auto identity_report =
      feebl::adaptive_grid::prolong_cic_charge_density(
          source, source_geometry, source_geometry, identity);
  require(identity == source && identity_report.relative_charge_error == 0.0 &&
              identity_report.normalized_first_moment_error == 0.0,
          "canonical density member was not a bit-preserving identity");

  const auto source_moments = density_moments(source, source_geometry);
  std::vector<double> source2(source.size(), 0.0);
  for (std::size_t ix = 0U; ix < source_geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < source_geometry.n_y; ++iy)
      source2[ix * source_geometry.n_y + iy] =
          0.4 * source[(source_geometry.n_x - 1U - ix) *
                       source_geometry.n_y + iy] +
          0.1 * source[ix * source_geometry.n_y + iy];
  long double canonical_dot = 0.0L;
  for (std::size_t cell = 0U; cell < source.size(); ++cell)
    canonical_dot += static_cast<long double>(source[cell]) * source2[cell];
  for (std::size_t factor : {2U, 4U, 8U}) {
    const std::size_t guard = 2U * factor;
    const std::size_t origin_x =
        static_cast<std::size_t>(source_geometry.origin_x) * factor - guard;
    const std::size_t origin_y =
        static_cast<std::size_t>(source_geometry.origin_y) * factor - guard;
    const std::size_t n_x = source_geometry.n_x * factor + 2U * guard;
    const std::size_t n_y = source_geometry.n_y * factor + 2U * guard;
    const float delta_x = source_geometry.delta_x /
                          static_cast<float>(factor);
    const float delta_y = source_geometry.delta_y /
                          static_cast<float>(factor);
    const auto destination_geometry = density_geometry(
        factor, source_geometry.global_n_x * factor,
        source_geometry.global_n_y * factor,
        origin_x, origin_y, n_x, n_y, delta_x, delta_y,
        source_geometry.offset_x * static_cast<float>(factor),
        source_geometry.offset_y * static_cast<float>(factor));
    std::vector<double> destination;
    const auto report =
        feebl::adaptive_grid::prolong_cic_charge_density(
            source, source_geometry, destination_geometry, destination);
    require(report.sign_preserving && report.relative_charge_error < 2e-13 &&
                report.normalized_first_moment_error < 2e-13,
            "coherent density hierarchy did not conserve charge/centroid");
    require(*std::min_element(destination.begin(), destination.end()) >= 0.0,
            "positive canonical source acquired a negative fine density");
    const auto destination_moments = density_moments(
        destination, destination_geometry);
    const long double charge_scale = std::max(1.0L,
                                               std::fabs(source_moments[0]));
    require(std::fabs(destination_moments[0] - source_moments[0]) /
                    charge_scale < 2e-13,
            "fine density charge does not close independently");
    const double inverse_children = 1.0 /
        static_cast<double>(factor * factor);
    const double coarse_area = source_geometry.delta_x *
                               source_geometry.delta_y;
    const double fine_area = destination_geometry.delta_x *
                             destination_geometry.delta_y;
    for (std::size_t ix = 0U; ix < source_geometry.n_x; ++ix)
      for (std::size_t iy = 0U; iy < source_geometry.n_y; ++iy) {
        const std::size_t global_x =
            (static_cast<std::size_t>(source_geometry.origin_x) + ix) *
            factor;
        const std::size_t global_y =
            (static_cast<std::size_t>(source_geometry.origin_y) + iy) *
            factor;
        for (std::size_t child_x = 0U; child_x < factor; ++child_x)
          for (std::size_t child_y = 0U; child_y < factor; ++child_y) {
            const std::size_t local_x =
                global_x + child_x - destination_geometry.origin_x;
            const std::size_t local_y =
                global_y + child_y - destination_geometry.origin_y;
            const double child =
                destination[local_x * destination_geometry.n_y + local_y];
            require(std::fabs(child -
                              source[ix * source_geometry.n_y + iy] *
                                  inverse_children) < 2e-15,
                    "canonical cell was not split equally among children");
            require(std::fabs(child / fine_area -
                              source[ix * source_geometry.n_y + iy] /
                                  coarse_area) < 2e-13,
                    "fine child changed the physical piecewise-constant "
                    "density");
          }
      }

    std::vector<double> destination2;
    (void)feebl::adaptive_grid::prolong_cic_charge_density(
        source2, source_geometry, destination_geometry, destination2);
    long double fine_dot = 0.0L;
    for (std::size_t cell = 0U; cell < destination.size(); ++cell)
      fine_dot += static_cast<long double>(destination[cell]) *
                  destination2[cell];
    require(std::fabs(static_cast<long double>(factor * factor) * fine_dot -
                      canonical_dot) /
                    std::max(1.0L, std::fabs(canonical_dot)) < 2e-13L,
            "factor-squared luminosity dot normalization is not invariant");

    // The production hierarchy transfers S1 as an extensive quantity and S2
    // with one additional inverse-child-area factor.  Verify that this keeps
    // per-control-volume Kish occupancy invariant for every refinement class.
    std::vector<double> canonical_s2(source.size(), 0.0);
    for (std::size_t cell = 0U; cell < source.size(); ++cell)
      canonical_s2[cell] = source[cell] * source[cell] / 3.0;
    std::vector<double> fine_s2;
    (void)feebl::adaptive_grid::prolong_cic_charge_density(
        canonical_s2, source_geometry, destination_geometry, fine_s2);
    for (double& value : fine_s2) value *= inverse_children;
    for (std::size_t cell = 0U; cell < destination.size(); ++cell)
      if (destination[cell] > 0.0)
        require(std::fabs(
                    feebl::adaptive_grid::kish_effective_sample_size(
                        destination[cell], fine_s2[cell]) - 3.0) < 2e-12,
                "sampling hierarchy changed Kish effective occupancy");

    std::vector<double> scaled_source(source);
    for (double& value : scaled_source) value *= -7.25;
    std::vector<double> scaled_destination;
    const auto scaled_report =
        feebl::adaptive_grid::prolong_cic_charge_density(
            scaled_source, source_geometry, destination_geometry,
            scaled_destination);
    require(scaled_report.sign_preserving,
            "negative scaled source lost sign preservation");
    double maximum_scale_error = 0.0;
    for (std::size_t i = 0U; i < destination.size(); ++i)
      maximum_scale_error = std::max(
          maximum_scale_error,
          std::fabs(scaled_destination[i] + 7.25 * destination[i]));
    require(maximum_scale_error < 2e-13,
            "density hierarchy is not scale invariant");
  }

  auto shifted_geometry = density_geometry(
      2U, source_geometry.global_n_x * 2U,
      source_geometry.global_n_y * 2U,
      static_cast<std::size_t>(source_geometry.origin_x) * 2U,
      static_cast<std::size_t>(source_geometry.origin_y) * 2U,
      source_geometry.n_x * 2U, source_geometry.n_y * 2U,
      source_geometry.delta_x * 0.5f, source_geometry.delta_y * 0.5f,
      source_geometry.offset_x * 2.0f + 0.25f,
      source_geometry.offset_y * 2.0f);
  std::vector<double> rejected;
  bool rejected_nonregistered_phase = false;
  try {
    (void)feebl::adaptive_grid::prolong_cic_charge_density(
        source, source_geometry, shifted_geometry, rejected);
  } catch (const std::invalid_argument&) {
    rejected_nonregistered_phase = true;
  }
  require(rejected_nonregistered_phase,
          "nonregistered fine-grid phase did not fail closed");
}

void test_coherent_density_patch_ownership() {
  const auto source_geometry = density_geometry(
      1U, 24U, 20U, 2U, 2U, 20U, 16U,
      1.0f, 1.0f, 12.0f, 10.0f);
  std::vector<double> inside(source_geometry.cells(), 0.0);
  std::vector<double> outside(source_geometry.cells(), 0.0);
  // The two sources occupy adjacent canonical control volumes on opposite
  // sides of the later patch boundary.  Aligned subdivision must give each
  // source exactly one owner, with no fine-CIC tail across that boundary.
  inside[7U * source_geometry.n_y + 8U] = 2.0;
  outside[6U * source_geometry.n_y + 8U] = 1.25;
  std::vector<double> combined(source_geometry.cells(), 0.0);
  for (std::size_t i = 0U; i < combined.size(); ++i)
    combined[i] = inside[i] + outside[i];
  const auto fine_geometry = density_geometry(
      2U, 48U, 40U, 2U, 2U, 44U, 36U,
      0.5f, 0.5f, 24.0f, 20.0f);
  std::vector<double> fine_inside;
  std::vector<double> fine_outside;
  std::vector<double> fine_combined;
  (void)feebl::adaptive_grid::prolong_cic_charge_density(
      inside, source_geometry, fine_geometry, fine_inside);
  (void)feebl::adaptive_grid::prolong_cic_charge_density(
      outside, source_geometry, fine_geometry, fine_outside);
  (void)feebl::adaptive_grid::prolong_cic_charge_density(
      combined, source_geometry, fine_geometry, fine_combined);

  auto patch_geometry = fine_geometry;
  patch_geometry.origin_x += 16;
  patch_geometry.origin_y += 8;
  patch_geometry.n_x = 18U;
  patch_geometry.n_y = 20U;
  require(patch_geometry.valid(), "ownership patch fixture is invalid");
  std::vector<double> patch_inside;
  std::vector<double> patch_outside;
  std::vector<double> patch_combined;
  feebl::adaptive_grid::extract_registered_density_window(
      fine_inside, fine_geometry, patch_geometry, patch_inside);
  feebl::adaptive_grid::extract_registered_density_window(
      fine_outside, fine_geometry, patch_geometry, patch_outside);
  feebl::adaptive_grid::extract_registered_density_window(
      fine_combined, fine_geometry, patch_geometry, patch_combined);
  const std::size_t exterior_face_x = static_cast<std::size_t>(
      patch_geometry.origin_x - fine_geometry.origin_x - 1);
  const std::size_t exterior_source_y = 18U;
  require(fine_outside[exterior_face_x * fine_geometry.n_y +
                       exterior_source_y] > 0.0,
          "exterior source was not adjacent to the patch boundary");
  require(std::accumulate(patch_inside.begin(), patch_inside.end(), 0.0) ==
              2.0,
          "boundary-adjacent interior source was not wholly owned by patch");
  require(std::accumulate(patch_outside.begin(), patch_outside.end(), 0.0) ==
              0.0,
          "exterior control volume leaked across the aligned patch boundary");
  double linearity_error = 0.0;
  for (std::size_t i = 0U; i < patch_combined.size(); ++i)
    linearity_error = std::max(
        linearity_error,
        std::fabs(patch_combined[i] -
                  (patch_inside[i] + patch_outside[i])));
  require(linearity_error < 2e-15,
          "registered patch source is not the linear restriction of the "
          "uniform coherent hierarchy");
}

void test_reference_restriction_sampling_regimes_and_admission() {
  const auto reference = density_geometry(
      4U, 32U, 32U, 0U, 0U, 32U, 32U,
      0.25f, 0.125f, 16.0f, 16.0f);
  const auto factor2 = density_geometry(
      2U, 16U, 16U, 0U, 0U, 16U, 16U,
      0.5f, 0.25f, 8.0f, 8.0f);
  const auto factor1 = density_geometry(
      1U, 8U, 8U, 0U, 0U, 8U, 8U,
      1.0f, 0.5f, 4.0f, 4.0f);
  std::vector<double> rho(reference.cells(), 0.0);
  rho[3U * reference.n_y + 5U] = 0.75;
  rho[11U * reference.n_y + 14U] = 1.25;
  rho[26U * reference.n_y + 21U] = 2.5;
  std::vector<double> restricted2;
  const auto report2 = feebl::adaptive_grid::restrict_cic_charge_density(
      rho, reference, factor2, restricted2);
  std::vector<double> restricted1;
  const auto report1 = feebl::adaptive_grid::restrict_cic_charge_density(
      rho, reference, factor1, restricted1);
  require(report2.sign_preserving && report1.sign_preserving &&
              report2.relative_charge_error < 2e-13 &&
              report1.relative_charge_error < 2e-13,
          "reference-CIC restriction lost charge or sign");
  require(std::fabs(restricted1[0U * factor1.n_y + 1U] - 0.75) < 1e-15 &&
              std::fabs(restricted1[2U * factor1.n_y + 3U] - 1.25) <
                  1e-15 &&
              std::fabs(restricted1[6U * factor1.n_y + 5U] - 2.5) <
                  1e-15,
          "reference-CIC restriction did not use exact aligned child sums");
  auto shifted = factor1;
  shifted.offset_x += 0.25f;
  require_failure([&] {
    std::vector<double> rejected;
    (void)feebl::adaptive_grid::restrict_cic_charge_density(
        rho, reference, shifted, rejected);
  }, "phase-shifted reference restriction was accepted");

  feebl::adaptive_grid::LevelDepositDescriptor descriptor;
  descriptor.accepted = true;
  descriptor.base_x = 9;
  descriptor.base_y = 10;
  descriptor.weight_x = 0.25f;
  descriptor.weight_y = 0.75f;
  std::vector<double> s1(factor1.cells(), 0.0);
  std::vector<double> s2(factor1.cells(), 0.0);
  feebl::adaptive_grid::accumulate_grouped_cic_sampling(
      descriptor, 2.0, reference, factor1, s1, s2);
  const std::size_t parent = 2U * factor1.n_y + 2U;
  require(std::fabs(s1[parent] - 2.0) < 1e-15 &&
              std::fabs(s2[parent] - 4.0) < 1e-15,
          "actual-source Kish S2 omitted within-particle parent cross terms");
  require(std::fabs(std::accumulate(s1.begin(), s1.end(), 0.0) - 2.0) <
              1e-15 &&
              std::fabs(std::accumulate(s2.begin(), s2.end(), 0.0) - 4.0) <
              1e-15,
          "grouped reference-CIC sampling did not close");
  auto shifted_sampling = factor1;
  shifted_sampling.offset_y += 0.25f;
  require_failure([&] {
    std::vector<double> bad_s1(factor1.cells(), 0.0);
    std::vector<double> bad_s2(factor1.cells(), 0.0);
    feebl::adaptive_grid::accumulate_grouped_cic_sampling(
        descriptor, 2.0, reference, shifted_sampling, bad_s1, bad_s2);
  }, "phase-shifted grouped sampling lattice was accepted");

  using feebl::adaptive_grid::SourcePairRegime;
  require(feebl::adaptive_grid::source_pair_regime(1U, 2U, 4U) ==
              SourcePairRegime::SourceProjection &&
              feebl::adaptive_grid::source_pair_regime(2U, 4U, 4U) ==
              SourcePairRegime::SourceProjection &&
              feebl::adaptive_grid::source_pair_regime(4U, 8U, 4U) ==
              SourcePairRegime::FieldOnly,
          "source/field Richardson regime tags are incorrect");
  const std::vector<std::size_t> n256{1U, 2U, 4U, 8U, 16U};
  const auto projection_upper =
      feebl::adaptive_grid::select_same_regime_triplet(n256, 1U, 4U);
  const auto field_lower =
      feebl::adaptive_grid::select_same_regime_triplet(n256, 2U, 4U);
  require(projection_upper.valid && projection_upper.first_pair == 0U &&
              projection_upper.second_pair == 1U &&
              projection_upper.regime == SourcePairRegime::SourceProjection,
          "projection candidate did not use its backward same-regime triplet");
  require(field_lower.valid && field_lower.first_pair == 2U &&
              field_lower.second_pair == 3U &&
              field_lower.regime == SourcePairRegime::FieldOnly,
          "field-only candidate did not use its forward same-regime triplet");
  const std::vector<std::size_t> n512{1U, 2U, 4U, 8U};
  require(!feebl::adaptive_grid::select_same_regime_triplet(
               n512, 0U, 2U).valid,
          "single projection pair was incorrectly certified across the "
          "source transition");
  require(feebl::adaptive_grid::select_same_regime_triplet(
              n512, 1U, 2U).valid,
          "supported field-only N512 campaign triplet was rejected");

  // At a common physical upper boundary, the finer lattice has one extra
  // complete CIC base-cell phase.  Nominal admission remains authoritative:
  // a nominal reject must be skipped even when the reference mapping accepts.
  const float boundary_tail_x = 3.7f;
  const float boundary_tail_y = 0.0f;
  require(!feebl::adaptive_grid::map_deposit_to_level(
               boundary_tail_x, boundary_tail_y, factor1).accepted &&
              feebl::adaptive_grid::map_deposit_to_level(
                  boundary_tail_x, boundary_tail_y, reference).accepted,
          "boundary fixture does not exercise finer-only CIC admission");
  require(feebl::adaptive_grid::map_deposit_to_level(
              0.1f, -0.2f, factor1).accepted &&
              feebl::adaptive_grid::map_deposit_to_level(
                  0.1f, -0.2f, reference).accepted,
          "reference grid failed to contain a nominally admitted particle");
}

void test_richardson() {
  std::vector<float> coarse1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> coarse2 = {2.0f, 1.0f};
  std::vector<float> fine1 = {1.1f, 2.2f, 3.3f, 4.4f};
  std::vector<float> fine2 = {2.2f, 1.1f};
  const auto estimate =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          coarse1, coarse2, fine1, fine2, 2U, 2.0, 1e-12);
  require(estimate.reliable && estimate.coarse_error > estimate.fine_error,
          "Richardson estimator did not distinguish accepted-grid errors");
  require(std::fabs(estimate.coarse_error /
                    estimate.fine_error - 4.0) < 1e-12,
          "Richardson coarse/fine error ratio is wrong");
  require(estimate.coarse_error == std::max(estimate.coarse_error_rms,
                                             estimate.coarse_error_linf),
          "Richardson acceptance is not conservative in RMS/Linf");

  std::vector<float> masked_coarse1 = {1.0f, 1.0f, 1000.0f, -1000.0f};
  std::vector<float> masked_fine1 = {1.01f, 1.02f, -1000.0f, 1000.0f};
  std::vector<float> masked_coarse2 = {2.0f, 3.0f};
  std::vector<float> masked_fine2 = {2.02f, 3.03f};
  std::vector<unsigned char> mask1 = {1U, 0U};
  std::vector<unsigned char> mask2 = {1U};
  const auto masked =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          masked_coarse1, masked_coarse2, masked_fine1, masked_fine2,
          2U, 2.0, 1e-12, &mask1, &mask2);
  require(masked.reliable && masked.observed_difference_linf < 0.03,
          "rejected targets diluted or contaminated Richardson acceptance");
  require(masked.observed_difference_rms_components[1] >
              masked.observed_difference_rms_components[0],
          "per-component Richardson diagnostics were pooled");

  require(std::fabs(feebl::adaptive_grid::kish_effective_sample_size(
                        2.0L, 2.0L) - 2.0) < 1e-15,
          "Kish N_eff failed equal CIC contributions");
  require(feebl::adaptive_grid::kish_effective_sample_size(0.0L, 0.0L) ==
              0.0,
          "Kish N_eff did not preserve empty support");
}

void test_richardson_contraction_and_validation() {
  // An exact p=2 model, Q_h = Q + C h^2, contracts successive absolute
  // differences by four under ratio-two refinement.
  const std::vector<float> h1(8U, 11.0f);
  const std::vector<float> h2(8U, 10.25f);
  const std::vector<float> h4(8U, 10.0625f);
  const auto first = feebl::adaptive_grid::estimate_same_particle_field_error(
      h1, h1, h2, h2, 2U, 2.0, 1e-12);
  const auto second = feebl::adaptive_grid::estimate_same_particle_field_error(
      h2, h2, h4, h4, 2U, 2.0, 1e-12);
  require(first.reliable && second.reliable,
          "contracting three-level fixture was marked unreliable");
  for (std::size_t component = 0U; component < 4U; ++component)
    require(std::fabs(first.absolute_difference_rms_components[component] /
                          second.absolute_difference_rms_components[component] -
                      4.0) < 1e-14,
            "p=2 three-level differences did not contract by four");
  const double observed_order = std::log(
      first.absolute_difference_rms_components[0] /
      second.absolute_difference_rms_components[0]) / std::log(2.0);
  require(std::fabs(observed_order - 2.0) < 1e-14,
          "three-level difference sequence did not recover order two");
  require(std::fabs(first.coarse_error / first.fine_error - 4.0) < 1e-14,
          "Richardson coarse/fine errors do not obey r^p contraction");

  // One localized outlier has a small RMS but a large Linf error.  The public
  // aggregate must retain the latter, rather than allowing RMS dilution.
  std::vector<float> fine(32U, 100.0f);
  std::vector<float> coarse(fine);
  coarse[0] = 90.0f;
  const std::vector<float> unchanged(fine);
  const auto outlier =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          coarse, unchanged, fine, unchanged, 2U, 2.0, 1e-12);
  require(outlier.reliable &&
              outlier.observed_difference_linf >
                  outlier.observed_difference_rms,
          "localized field error was not distinguished by Linf");
  require(outlier.coarse_error == outlier.coarse_error_linf &&
              outlier.fine_error == outlier.fine_error_linf,
          "conservative Richardson aggregate did not retain Linf");

  const std::vector<float> two = {1.0f, 2.0f};
  const std::vector<float> odd = {1.0f};
  const std::vector<float> mismatched = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<unsigned char> wrong_mask = {1U, 1U};
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        two, two, mismatched, two, 2U, 2.0, 1e-12);
  }, "Richardson estimator accepted unequal coarse/fine storage");
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        odd, two, odd, two, 2U, 2.0, 1e-12);
  }, "Richardson estimator accepted an odd field-vector length");
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        two, two, two, two, 2U, 2.0, 1e-12, &wrong_mask, NULL);
  }, "Richardson estimator accepted a mismatched target mask");
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        two, two, two, two, 3U, 2.0, 1e-12);
  }, "Richardson estimator accepted a non-power-of-two ratio");
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        two, two, two, two, 2U, 0.0, 1e-12);
  }, "Richardson estimator accepted a non-positive order");
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        two, two, two, two, 2U, 2.0, -1.0);
  }, "Richardson estimator accepted a negative field floor");

  std::vector<float> nonfinite(two);
  nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
  require_failure([&] {
    (void)feebl::adaptive_grid::estimate_same_particle_field_error(
        nonfinite, two, two, two, 2U, 2.0, 1e-12);
  }, "Richardson estimator accepted a non-finite gathered field");
  const auto exponent_overflow =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          two, two, two, two, 2U,
          std::numeric_limits<double>::max(), 1e-12);
  require(!exponent_overflow.reliable,
          "overflowed Richardson exponent was marked reliable");
  const std::vector<float> no_fields;
  const auto empty_comparison =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          no_fields, no_fields, no_fields, no_fields, 2U, 2.0, 1e-12);
  require(!empty_comparison.reliable,
          "empty Richardson comparison was marked reliable");
  const std::vector<unsigned char> reject_all = {0U};
  const auto fully_masked =
      feebl::adaptive_grid::estimate_same_particle_field_error(
          two, two, two, two, 2U, 2.0, 1e-12,
          &reject_all, &reject_all);
  require(!fully_masked.reliable,
          "fully masked Richardson comparison was marked reliable");

  require_failure([] {
    (void)feebl::adaptive_grid::kish_effective_sample_size(-1.0L, 1.0L);
  }, "Kish N_eff accepted a negative weight sum");
  require_failure([] {
    (void)feebl::adaptive_grid::kish_effective_sample_size(
        std::numeric_limits<long double>::max(),
        std::numeric_limits<long double>::min());
  }, "overflowed Kish N_eff was accepted");
}

void test_symmetric_vector_field_sensitivity() {
  using feebl::adaptive_grid::estimate_symmetric_vector_field_sensitivity;

  // A fine field near zero must not turn a finite coarse field into an
  // unbounded one-sided relative error.
  const std::vector<float> coarse1 = {3.0f, 4.0f};
  const std::vector<float> near_zero1 = {1.0e-30f, 0.0f};
  const std::vector<float> coarse2 = {0.0f, 2.0f};
  const std::vector<float> near_zero2 = {0.0f, 0.0f};
  const auto near_zero = estimate_symmetric_vector_field_sensitivity(
      coarse1, coarse2, near_zero1, near_zero2, 1.0e-30);
  require(near_zero.reliable && near_zero.accepted_targets[0] == 1U &&
              near_zero.accepted_targets[1] == 1U,
          "symmetric vector sensitivity lost accepted targets");
  require(std::fabs(near_zero.rms_ratios[0] - 1.0) < 1.0e-14 &&
              std::fabs(near_zero.linf_ratios[0] - 1.0) < 1.0e-14 &&
              std::fabs(near_zero.aggregate - 1.0) < 1.0e-14 &&
              std::isfinite(near_zero.aggregate) &&
              near_zero.aggregate >= 0.0 && near_zero.aggregate <= 2.0,
          "near-zero fine field produced an unbounded sensitivity");

  const std::vector<float> subfloor = {1.0e-20f, 0.0f};
  const std::vector<float> vector_zero = {0.0f, 0.0f};
  const auto floor_limited = estimate_symmetric_vector_field_sensitivity(
      subfloor, subfloor, vector_zero, vector_zero, 1.0e-10);
  require(floor_limited.aggregate > 0.0 &&
              floor_limited.aggregate < 2.0e-10,
          "absolute field floor did not regularize a sub-floor difference");

  // Reversing a nonzero vector saturates the exact triangle-inequality bound.
  const std::vector<float> reversed = {-3.0f, -4.0f};
  const auto sign_reversal = estimate_symmetric_vector_field_sensitivity(
      coarse1, coarse1, reversed, reversed, 1.0e-30);
  require(sign_reversal.reliable && sign_reversal.aggregate == 2.0 &&
              sign_reversal.rms_ratios[0] == 2.0 &&
              sign_reversal.linf_ratios[1] == 2.0,
          "sign reversal did not saturate the bounded sensitivity");

  // The metric is invariant under a common field rescaling while the floor
  // remains inactive.
  const std::vector<float> scale_coarse = {
      1.0f, 2.0f, -2.0f, 1.0f};
  const std::vector<float> scale_fine = {
      1.5f, 1.5f, -1.5f, 0.5f};
  std::vector<float> large_coarse(scale_coarse);
  std::vector<float> large_fine(scale_fine);
  for (float& value : large_coarse) value *= 1.0e20f;
  for (float& value : large_fine) value *= 1.0e20f;
  const auto unit_scale = estimate_symmetric_vector_field_sensitivity(
      scale_coarse, scale_coarse, scale_fine, scale_fine, 1.0e-30);
  const auto large_scale = estimate_symmetric_vector_field_sensitivity(
      large_coarse, large_coarse, large_fine, large_fine, 1.0e-30);
  require(std::fabs(unit_scale.aggregate - large_scale.aggregate) < 2.0e-8,
          "symmetric vector sensitivity is not scale invariant");

  // A rotation is measured as a change of the physical transverse vector,
  // without a singular component-by-component denominator.
  const std::vector<float> x_axis = {1.0f, 0.0f};
  const std::vector<float> y_axis = {0.0f, 1.0f};
  const auto rotation = estimate_symmetric_vector_field_sensitivity(
      x_axis, x_axis, y_axis, y_axis, 1.0e-30);
  require(std::fabs(rotation.aggregate - std::sqrt(2.0)) < 1.0e-14,
          "transverse field rotation has the wrong sensitivity");

  const std::vector<float> zeros(4U, 0.0f);
  const auto exact_zero = estimate_symmetric_vector_field_sensitivity(
      zeros, zeros, zeros, zeros, 1.0e-30);
  require(exact_zero.reliable && exact_zero.aggregate == 0.0,
          "exact zero fields did not produce zero sensitivity");

  // Masks apply at particle granularity to both vector components.
  const std::vector<unsigned char> first_only = {1U, 0U};
  const std::vector<float> masked_coarse = {
      1.0f, 0.0f, 1.0e20f, -1.0e20f};
  const std::vector<float> masked_fine = {
      0.5f, 0.0f, -1.0e20f, 1.0e20f};
  const auto masked = estimate_symmetric_vector_field_sensitivity(
      masked_coarse, masked_coarse, masked_fine, masked_fine,
      1.0e-30, &first_only, &first_only);
  require(masked.reliable && masked.accepted_targets[0] == 1U &&
              masked.accepted_targets[1] == 1U &&
              std::fabs(masked.aggregate - 0.5) < 1.0e-14,
          "rejected targets contaminated symmetric vector sensitivity");

  const std::vector<unsigned char> reject_all = {0U, 0U};
  const auto fully_masked = estimate_symmetric_vector_field_sensitivity(
      zeros, zeros, zeros, zeros, 1.0e-30,
      &reject_all, &reject_all);
  require(!fully_masked.reliable && fully_masked.aggregate == 0.0,
          "fully masked sensitivity was marked reliable");

  const std::vector<float> odd = {1.0f};
  const std::vector<unsigned char> wrong_mask = {1U};
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        odd, zeros, odd, zeros, 1.0e-30);
  }, "symmetric vector sensitivity accepted odd field storage");
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        zeros, zeros, vector_zero, zeros, 1.0e-30);
  }, "symmetric vector sensitivity accepted unequal level storage");
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        zeros, zeros, zeros, zeros, 1.0e-30,
        &wrong_mask, NULL);
  }, "symmetric vector sensitivity accepted a mismatched mask");
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        zeros, zeros, zeros, zeros, 0.0);
  }, "symmetric vector sensitivity accepted a zero field floor");
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        zeros, zeros, zeros, zeros, -1.0);
  }, "symmetric vector sensitivity accepted a negative field floor");
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        zeros, zeros, zeros, zeros,
        std::numeric_limits<double>::infinity());
  }, "symmetric vector sensitivity accepted a non-finite field floor");
  std::vector<float> nonfinite(zeros);
  nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
  require_failure([&] {
    (void)estimate_symmetric_vector_field_sensitivity(
        nonfinite, zeros, zeros, zeros, 1.0e-30);
  }, "symmetric vector sensitivity accepted a non-finite field");
}

void test_vector_rms_quantile_field_error() {
  using feebl::adaptive_grid::estimate_vector_field_error_statistics;

  std::vector<float> coarse(2000U, 0.0f);
  std::vector<float> fine(2000U, 0.0f);
  for (std::size_t particle = 0U; particle < 1000U; ++particle) {
    coarse[2U * particle] = 1.0f;
    fine[2U * particle] = 0.999f;
  }
  fine[2U * 998U] = 0.9f;
  fine[2U * 999U] = -1.0f;
  const auto statistics = estimate_vector_field_error_statistics(
      coarse, coarse, fine, fine, 0.999, 0.01);
  require(statistics.reliable &&
              statistics.accepted_targets[0] == 1000U &&
              statistics.accepted_targets[1] == 1000U,
          "vector RMS/quantile estimator lost accepted targets");
  require(std::fabs(statistics.aggregate_quantile - 0.1) < 1.0e-6,
          "vector error does not use conservative nearest-rank p99.9");
  require(statistics.aggregate_maximum == 2.0,
          "vector catastrophe diagnostic lost a sign reversal");
  require(statistics.total_above_target == 4U &&
              std::fabs(statistics.above_target_exposure - 0.002) < 1e-15,
          "vector target exceedance exposure/count is wrong");

  // Unlike the legacy component diagnostic, the policy metric has no field
  // floor: a weak but non-zero field disappearing is still a unit relative
  // change, while an exact zero-to-zero comparison is exactly zero.
  const std::vector<float> tiny = {1.0e-30f, 0.0f};
  const std::vector<float> zero = {0.0f, 0.0f};
  const auto weak = estimate_vector_field_error_statistics(
      tiny, tiny, zero, zero, 0.999, 0.01);
  require(weak.aggregate_rms == 1.0 && weak.aggregate_quantile == 1.0 &&
              weak.aggregate_maximum == 1.0,
          "weak non-zero field was hidden by an implicit denominator floor");
  const auto exact_zero = estimate_vector_field_error_statistics(
      zero, zero, zero, zero, 0.999, 0.01);
  require(exact_zero.reliable && exact_zero.aggregate_rms == 0.0 &&
              exact_zero.aggregate_quantile == 0.0 &&
              exact_zero.aggregate_maximum == 0.0,
          "exact zero vector comparison was not well-defined");

  const std::vector<unsigned char> rejected = {0U};
  const auto none = estimate_vector_field_error_statistics(
      tiny, tiny, zero, zero, 0.999, 0.01, &rejected, &rejected);
  require(!none.reliable && none.total_accepted_targets == 0U,
          "fully masked vector error comparison was marked reliable");
  require_failure([&] {
    (void)estimate_vector_field_error_statistics(
        tiny, tiny, zero, zero, 0.90, 0.01);
  }, "vector error accepted a non-high quantile");
  require_failure([&] {
    (void)estimate_vector_field_error_statistics(
        tiny, tiny, zero, zero, 1.0, 0.01);
  }, "vector error accepted quantile one");
}

void test_mixed_vector_field_error_v2() {
  using feebl::adaptive_grid::VectorFieldErrorStatistics;
  using feebl::adaptive_grid::estimate_mixed_vector_field_error_statistics_v2;
  using feebl::adaptive_grid::estimate_vector_field_richardson_error;

  // A tiny nonzero field may have an O(1) unregularized relative change, but
  // it is neither a mixed-metric failure nor a physical catastrophe when the
  // absolute discrepancy is far below the preregistered kick scale.
  const std::vector<float> tiny = {1.0e-9f, 0.0f};
  const std::vector<float> zero = {0.0f, 0.0f};
  const auto weak = estimate_mixed_vector_field_error_statistics_v2(
      tiny, tiny, zero, zero, 0.999, 0.01,
      1.0, 1.0e-3, 0.5, 1.0e-2);
  require(weak.reliable && weak.policy_version == 2U &&
              std::fabs(weak.aggregate_rms - 1.0e-6) < 2.0e-13 &&
              weak.aggregate_absolute_maximum > 9.9e-10 &&
              weak.aggregate_absolute_maximum < 1.01e-9 &&
              weak.total_catastrophe_targets == 0U &&
              weak.aggregate_significant_unregularized_maximum == 0.0,
          "mixed V2 metric misclassified an absolutely negligible kick");

  // The same relative reversal at a physically significant scale must trip
  // the catastrophe gate, and the separately retained mixed and absolute
  // diagnostics must remain finite.
  const std::vector<float> positive = {1.0f, 0.0f};
  const std::vector<float> negative = {-1.0f, 0.0f};
  const auto strong = estimate_mixed_vector_field_error_statistics_v2(
      positive, positive, negative, negative, 0.999, 0.01,
      1.0, 1.0e-3, 0.5, 1.0e-2);
  require(strong.aggregate_maximum == 2.0 &&
              strong.aggregate_absolute_maximum == 2.0 &&
              strong.aggregate_significant_unregularized_maximum == 2.0 &&
              strong.total_catastrophe_targets == 2U &&
              strong.catastrophe_exposure == 1.0,
          "mixed V2 metric lost a physically significant reversal");

  const auto make_v2 = [](double normalized, double absolute) {
    VectorFieldErrorStatistics value;
    value.reliable = true;
    value.policy_version = 2U;
    value.quantile_probability = 0.999;
    value.reference_field = 1.0;
    value.absolute_floor_fraction = 1.0e-3;
    value.absolute_floor = 1.0e-3;
    value.catastrophe_relative_limit = 0.5;
    value.catastrophe_absolute_fraction = 1.0e-2;
    value.catastrophe_absolute_threshold = 1.0e-2;
    value.rms = {{normalized, normalized}};
    value.quantile = {{normalized, normalized}};
    value.maximum = {{normalized, normalized}};
    value.absolute_rms = {{absolute, absolute}};
    value.absolute_quantile = {{absolute, absolute}};
    value.absolute_maximum = {{absolute, absolute}};
    value.accepted_targets = {{1000U, 1000U}};
    value.accepted_target_hash = {{UINT64_C(0x11111111),
                                   UINT64_C(0x22222222)}};
    value.accepted_target_ids_bound = {{true, true}};
    value.aggregate_rms = normalized;
    value.aggregate_quantile = normalized;
    value.aggregate_maximum = normalized;
    value.aggregate_absolute_rms = absolute;
    value.aggregate_absolute_quantile = absolute;
    value.aggregate_absolute_maximum = absolute;
    value.total_accepted_targets = 2000U;
    return value;
  };

  // Deliberately keep the normalized pair differences flat.  V2 must obtain
  // p=2 from absolute same-target differences and then apply that order to
  // the current mixed error, rather than estimating order from the ratios.
  const auto absolute_coarse = make_v2(0.02, 0.04);
  const auto absolute_fine = make_v2(0.02, 0.01);
  const auto richardson = estimate_vector_field_richardson_error(
      absolute_coarse, absolute_fine, absolute_fine, 2U, 0.25);
  require(richardson.reliable && richardson.asymptotic &&
              richardson.order_from_absolute_differences &&
              richardson.plausible_order && !richardson.roundoff_limited &&
              std::fabs(richardson.observed_order - 2.0) < 1.0e-14 &&
              std::fabs(richardson.fine_error - 0.02 / 3.0) < 1.0e-14,
          "V2 Richardson order was not derived from absolute differences");

  const auto roundoff = make_v2(0.0, 0.0);
  const auto no_false_order = estimate_vector_field_richardson_error(
      roundoff, roundoff, roundoff, 2U, 0.25);
  require(no_false_order.reliable && !no_false_order.asymptotic &&
              no_false_order.roundoff_limited &&
              !no_false_order.plausible_order &&
              no_false_order.observed_order == 0.0,
          "V2 treated a zero/roundoff plateau as high-order convergence");

  VectorFieldErrorStatistics mixed_version = absolute_fine;
  mixed_version.policy_version = 1U;
  require_failure([&] {
    (void)estimate_vector_field_richardson_error(
        absolute_coarse, mixed_version, absolute_fine, 2U, 0.25);
  }, "Richardson accepted a triplet mixing V1 and V2 norms");

  require_failure([&] {
    (void)estimate_mixed_vector_field_error_statistics_v2(
        tiny, tiny, zero, zero, 0.999, 0.01,
        0.0, 1.0e-3, 0.5, 1.0e-2);
  }, "mixed V2 metric accepted a zero reference field");
  require_failure([&] {
    (void)estimate_mixed_vector_field_error_statistics_v2(
        tiny, tiny, zero, zero, 0.999, 0.01,
        1.0, 0.0, 0.5, 1.0e-2);
  }, "mixed V2 metric accepted a zero absolute floor fraction");
  require_failure([&] {
    (void)estimate_mixed_vector_field_error_statistics_v2(
        tiny, tiny, zero, zero, 0.999, 0.01,
        1.0, 1.0e-3, 0.5, 0.0);
  }, "mixed V2 metric accepted a zero catastrophe absolute threshold");

  // Stable target identity is part of the V2 same-particle proof.  Equal
  // masks and counts with a permuted physical-root order must hash
  // differently and cannot form a Richardson triplet.
  const std::vector<float> two_zero(4U, 0.0f);
  const std::vector<unsigned char> two_accept(2U, 1U);
  const std::vector<std::uint64_t> ids12 = {1U, 2U};
  const std::vector<std::uint64_t> ids21 = {2U, 1U};
  const auto ordered = estimate_mixed_vector_field_error_statistics_v2(
      two_zero, two_zero, two_zero, two_zero, 0.999, 0.01,
      1.0, 1.0e-3, 0.5, 1.0e-2, &two_accept, &two_accept,
      &ids12, &ids12);
  const auto permuted = estimate_mixed_vector_field_error_statistics_v2(
      two_zero, two_zero, two_zero, two_zero, 0.999, 0.01,
      1.0, 1.0e-3, 0.5, 1.0e-2, &two_accept, &two_accept,
      &ids21, &ids12);
  require(ordered.accepted_target_ids_bound[0] &&
              ordered.accepted_target_hash[0] !=
                  permuted.accepted_target_hash[0],
          "V2 target identity hash ignored stable-root ordering");
  require_failure([&] {
    (void)estimate_vector_field_richardson_error(
        ordered, permuted, ordered, 2U, 0.25);
  }, "V2 Richardson accepted a permuted stable-root identity");

  // A poorly resolved small beam must not be diluted by a much larger quiet
  // beam in the exposure gate.
  std::vector<float> quiet(198U, 0.0f);
  std::vector<std::uint64_t> quiet_ids(99U, 0U);
  for (std::size_t i = 0U; i < quiet_ids.size(); ++i) quiet_ids[i] = 100U + i;
  const std::vector<std::uint64_t> one_id = {7U};
  const std::vector<float> one_large = {1.0f, 0.0f};
  const auto imbalanced = estimate_mixed_vector_field_error_statistics_v2(
      one_large, quiet, zero, quiet, 0.99, 0.01,
      1.0, 1.0e-3, 0.5, 1.0e-2, NULL, NULL,
      &one_id, &quiet_ids);
  require(imbalanced.above_target_exposure < 0.02 &&
              imbalanced.maximum_beam_above_target_exposure == 1.0,
          "mixed V2 exposure pooled away a failing beam");
}

void test_vector_richardson_per_beam_order_and_quantile_support() {
  using feebl::adaptive_grid::VectorFieldErrorStatistics;
  using feebl::adaptive_grid::estimate_vector_field_richardson_error;
  const auto make_statistics = [](
      double rms0, double rms1, double quantile0, double quantile1,
      std::size_t targets) {
    VectorFieldErrorStatistics value;
    value.reliable = true;
    value.quantile_probability = 0.999;
    value.rms = {{rms0, rms1}};
    value.quantile = {{quantile0, quantile1}};
    value.maximum = {{std::max(rms0, quantile0),
                      std::max(rms1, quantile1)}};
    value.accepted_targets = {{targets, targets}};
    value.accepted_target_hash = {{UINT64_C(0x11111111),
                                    UINT64_C(0x22222222)}};
    value.aggregate_rms = std::max(rms0, rms1);
    value.aggregate_quantile = std::max(quantile0, quantile1);
    value.aggregate_maximum = std::max(value.maximum[0], value.maximum[1]);
    value.total_accepted_targets = 2U * targets;
    return value;
  };

  // Aggregate maxima contract from 0.08 to 0.02, but beam 0's quantile
  // grows from 0.001 to 0.02.  Reducing over beams before estimating order
  // would falsely infer p=2; the componentwise gate must reject it.
  const VectorFieldErrorStatistics switched_earlier =
      make_statistics(0.08, 0.04, 0.001, 0.04, 1000U);
  const VectorFieldErrorStatistics switched_later =
      make_statistics(0.02, 0.01, 0.02, 0.01, 1000U);
  const auto switched = estimate_vector_field_richardson_error(
      switched_earlier, switched_later, switched_later, 2U, 0.25);
  require(switched.reliable && !switched.asymptotic &&
              switched.observed_order == 0.0,
          "aggregate argmax switching hid a growing beam/metric");

  const VectorFieldErrorStatistics contracting_999_earlier =
      make_statistics(0.04, 0.02, 0.08, 0.04, 999U);
  const VectorFieldErrorStatistics contracting_999_later =
      make_statistics(0.01, 0.005, 0.02, 0.01, 999U);
  const auto under_sampled = estimate_vector_field_richardson_error(
      contracting_999_earlier, contracting_999_later,
      contracting_999_later, 2U, 0.25);
  require(under_sampled.reliable && under_sampled.asymptotic &&
              !under_sampled.quantile_support_met &&
              under_sampled.minimum_quantile_targets == 1000U &&
              under_sampled.minimum_accepted_targets == 999U,
          "p99.9 Richardson policy accepted fewer than 1000 targets/beam");

  const VectorFieldErrorStatistics contracting_1000_earlier =
      make_statistics(0.04, 0.02, 0.08, 0.04, 1000U);
  const VectorFieldErrorStatistics contracting_1000_later =
      make_statistics(0.01, 0.005, 0.02, 0.01, 1000U);
  const auto supported = estimate_vector_field_richardson_error(
      contracting_1000_earlier, contracting_1000_later,
      contracting_1000_later, 2U, 0.25);
  require(supported.reliable && supported.asymptotic &&
              supported.quantile_support_met &&
              std::fabs(supported.observed_order - 2.0) < 1.0e-14 &&
              std::fabs(supported.fine_error - 0.02 / 3.0) < 1.0e-14,
          "supported componentwise Richardson estimate is incorrect");

  VectorFieldErrorStatistics missing_beam = contracting_1000_later;
  missing_beam.accepted_targets[1] = 0U;
  missing_beam.total_accepted_targets = 1000U;
  require_failure([&] {
    (void)estimate_vector_field_richardson_error(
        missing_beam, missing_beam, missing_beam, 2U, 0.25);
  }, "vector Richardson certificate accepted one unevaluated beam");

  VectorFieldErrorStatistics different_identity =
      contracting_1000_later;
  different_identity.accepted_target_hash[0] ^= UINT64_C(1);
  require_failure([&] {
    (void)estimate_vector_field_richardson_error(
        contracting_1000_earlier, different_identity,
        contracting_1000_later, 2U, 0.25);
  }, "equal target counts hid a different Richardson target set");
}

void test_patch_and_harmonic_solver() {
  const std::size_t nx = 64U;
  const std::size_t ny = 48U;
  std::vector<double> rho1(nx * ny, 0.0);
  std::vector<double> rho2(nx * ny, 0.0);
  for (std::size_t x = 0; x < nx; ++x)
    for (std::size_t y = 0; y < ny; ++y) {
      const double dx = static_cast<double>(x) - 31.0;
      const double dy = static_cast<double>(y) - 23.0;
      rho1[x * ny + y] = std::exp(-(dx * dx / 20.0 + dy * dy / 8.0));
      rho2[x * ny + y] = std::exp(-(dx * dx / 16.0 + dy * dy / 10.0));
    }
  const auto patch = feebl::adaptive_grid::select_nested_patch(
      rho1, rho2, nx, ny, 0.25, 0.15, 1e-15, 1U, 2U, 2U, 4U);
  require(patch.valid && patch.n_x() < nx && patch.n_y() < ny,
          "connected overlap selector did not produce a local patch");
  require(patch.begin_x % 4U == 0U && patch.begin_y % 4U == 0U &&
          patch.end_x % 4U == 0U && patch.end_y % 4U == 0U,
          "nested patch does not satisfy blocking alignment");

  const std::size_t hx = 24U;
  const std::size_t hy = 18U;
  std::vector<double> harmonic(hx * hy, 0.0);
  for (std::size_t x = 0; x < hx; ++x) {
    const double value_low = static_cast<double>(x);
    harmonic[x * hy] = value_low;
    harmonic[x * hy + hy - 1U] = value_low + static_cast<double>(hy - 1U);
  }
  for (std::size_t y = 0; y < hy; ++y) {
    harmonic[y] = static_cast<double>(y);
    harmonic[(hx - 1U) * hy + y] = static_cast<double>(hx - 1U + y);
  }
  const auto report = feebl::adaptive_grid::solve_harmonic_dirichlet(
      harmonic, hx, hy, 1.0, 1.0, 1e-10, 20000U);
  require(report.converged, "manufactured linear harmonic solve did not converge");
  double maximum_error = 0.0;
  for (std::size_t x = 0; x < hx; ++x)
    for (std::size_t y = 0; y < hy; ++y)
      maximum_error = std::max(
          maximum_error,
          std::fabs(harmonic[x * hy + y] - static_cast<double>(x + y)));
  require(maximum_error < 2e-7,
          "harmonic Dirichlet solver failed its linear manufactured solution");

  std::vector<double> shifted(hx * hy, 0.0);
  const double gauge = 1.0e12;
  for (std::size_t x = 0; x < hx; ++x) {
    shifted[x * hy] = gauge + static_cast<double>(x);
    shifted[x * hy + hy - 1U] =
        gauge + static_cast<double>(x + hy - 1U);
  }
  for (std::size_t y = 0; y < hy; ++y) {
    shifted[y] = gauge + static_cast<double>(y);
    shifted[(hx - 1U) * hy + y] =
        gauge + static_cast<double>(hx - 1U + y);
  }
  const auto shifted_report =
      feebl::adaptive_grid::solve_harmonic_dirichlet(
          shifted, hx, hy, 1.0, 1.0, 1e-10, 20000U);
  require(shifted_report.converged &&
              std::fabs(shifted_report.normalized_residual -
                         report.normalized_residual) < 1e-10,
          "harmonic residual changed under an additive potential gauge");
  require(std::fabs(shifted_report.normalized_residual_l2 -
                        report.normalized_residual_l2) < 1e-10 &&
              std::fabs(shifted_report.normalized_residual_linf -
                        report.normalized_residual_linf) < 1e-10,
          "harmonic residual components changed under a potential gauge");
  require(report.normalized_residual ==
                  std::max(report.normalized_residual_l2,
                           report.normalized_residual_linf) &&
              shifted_report.normalized_residual ==
                  std::max(shifted_report.normalized_residual_l2,
                           shifted_report.normalized_residual_linf),
          "harmonic convergence did not use the conservative residual norm");
  double maximum_shifted_error = 0.0;
  for (std::size_t x = 0; x < hx; ++x)
    for (std::size_t y = 0; y < hy; ++y)
      maximum_shifted_error = std::max(
          maximum_shifted_error,
          std::fabs(shifted[x * hy + y] -
                    (gauge + static_cast<double>(x + y))));
  require(maximum_shifted_error < 2e-4,
          "gauge-centred harmonic solve lost its manufactured solution");
}

void test_nested_geometry_prolongation_and_transition() {
  const auto grid = lattice();
  const auto plan = window(grid);
  const auto parent = feebl::adaptive_grid::make_refined_level_geometry(
      grid, plan, 2U, 12U);
  feebl::adaptive_grid::NestedPatchBox box;
  box.valid = true;
  box.begin_x = 4U;
  box.begin_y = 4U;
  box.end_x = parent.n_x - 4U;
  box.end_y = parent.n_y - 4U;
  const auto coarse_patch = feebl::adaptive_grid::make_patch_geometry(
      parent, box);
  const auto fine_patch = feebl::adaptive_grid::refine_patch_geometry(
      coarse_patch, 2U);
  require(fine_patch.refinement_factor == 4U &&
              fine_patch.global_n_x == 2U * coarse_patch.global_n_x &&
              fine_patch.global_n_y == 2U * coarse_patch.global_n_y &&
              fine_patch.origin_x == 2 * coarse_patch.origin_x &&
              fine_patch.origin_y == 2 * coarse_patch.origin_y &&
              fine_patch.n_x == 2U * coarse_patch.n_x &&
              fine_patch.n_y == 2U * coarse_patch.n_y &&
              fine_patch.delta_x == 0.5f * coarse_patch.delta_x &&
              fine_patch.delta_y == 0.5f * coarse_patch.delta_y &&
              fine_patch.inverse_delta_x ==
                  2.0f * coarse_patch.inverse_delta_x &&
              fine_patch.inverse_delta_y ==
                  2.0f * coarse_patch.inverse_delta_y &&
              fine_patch.offset_x == 2.0f * coarse_patch.offset_x &&
              fine_patch.offset_y == 2.0f * coarse_patch.offset_y,
          "nested patch did not create genuinely smaller cells");
  const auto physical_boundary = [](
      const feebl::adaptive_grid::FieldLevelGeometry& geometry,
      bool upper, bool x_axis) {
    const double origin = x_axis ? geometry.origin_x : geometry.origin_y;
    const double extent = x_axis ? geometry.n_x : geometry.n_y;
    const double offset = x_axis ? geometry.offset_x : geometry.offset_y;
    const double delta = x_axis ? geometry.delta_x : geometry.delta_y;
    return (origin + (upper ? extent : 0.0) - offset) * delta;
  };
  for (bool x_axis : {false, true})
    for (bool upper : {false, true})
      require(std::fabs(physical_boundary(coarse_patch, upper, x_axis) -
                        physical_boundary(fine_patch, upper, x_axis)) < 1e-6,
              "ratio-two refinement changed a physical patch boundary");

  require_failure([&] {
    (void)feebl::adaptive_grid::refine_patch_geometry(coarse_patch, 1U);
  }, "nested patch accepted a refinement ratio below two");
  require_failure([&] {
    (void)feebl::adaptive_grid::refine_patch_geometry(coarse_patch, 3U);
  }, "nested patch accepted a non-power-of-two refinement ratio");
  feebl::adaptive_grid::NestedPatchBox outside = box;
  outside.end_x = parent.n_x + 1U;
  require_failure([&] {
    (void)feebl::adaptive_grid::make_patch_geometry(parent, outside);
  }, "nested patch escaped its parent geometry");

  feebl::adaptive_grid::FieldLevelGeometry overflowing_patch(coarse_patch);
  overflowing_patch.refinement_factor =
      std::numeric_limits<std::size_t>::max() / 2U + 1U;
  require(overflowing_patch.valid(),
          "overflow fixture is not a valid parent geometry");
  require_failure([&] {
    (void)feebl::adaptive_grid::refine_patch_geometry(
        overflowing_patch, 2U);
  }, "nested refinement-factor multiplication wrapped around");

  feebl::adaptive_grid::FieldLevelGeometry overflowing_cells(coarse_patch);
  overflowing_cells.origin_x = 0;
  overflowing_cells.origin_y = 0;
  overflowing_cells.n_x = std::numeric_limits<std::size_t>::max();
  overflowing_cells.n_y = 4U;
  overflowing_cells.global_n_x = overflowing_cells.n_x;
  overflowing_cells.global_n_y = overflowing_cells.n_y;
  require(!overflowing_cells.valid(),
          "unrepresentable field geometry passed validation");

  feebl::adaptive_grid::FieldLevelGeometry nonfinite_offset(coarse_patch);
  nonfinite_offset.offset_x = std::numeric_limits<float>::infinity();
  require(!nonfinite_offset.valid(),
          "non-finite field offset passed geometry validation");
  require_failure([&] {
    (void)feebl::adaptive_grid::map_deposit_to_level(
        0.0f, 0.0f, nonfinite_offset);
  }, "deposit mapping accepted a non-finite field offset");
  require_failure([&] {
    (void)feebl::adaptive_grid::map_gather_to_level(
        0.0f, 0.0f, nonfinite_offset);
  }, "gather mapping accepted a non-finite field offset");

  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  require_failure([&] {
    (void)feebl::adaptive_grid::map_deposit_to_level(
        quiet_nan, 0.0f, coarse_patch);
  }, "deposit mapping accepted a NaN particle coordinate");
  require_failure([&] {
    (void)feebl::adaptive_grid::map_gather_to_level(
        0.0f, infinity, coarse_patch);
  }, "gather mapping accepted an infinite particle coordinate");

  feebl::adaptive_grid::FieldLevelGeometry overflowing_coordinate(coarse_patch);
  overflowing_coordinate.inverse_delta_x =
      std::numeric_limits<float>::max();
  overflowing_coordinate.inverse_delta_y =
      std::numeric_limits<float>::max();
  require(overflowing_coordinate.valid(),
          "affine-overflow mapper fixture is not valid");
  require_failure([&] {
    (void)feebl::adaptive_grid::map_deposit_to_level(
        2.0f, 2.0f, overflowing_coordinate);
  }, "deposit mapping accepted an overflowed affine coordinate");
  require_failure([&] {
    (void)feebl::adaptive_grid::map_gather_to_level(
        2.0f, 2.0f, overflowing_coordinate);
  }, "gather mapping accepted an overflowed affine coordinate");

  feebl::adaptive_grid::FieldLevelGeometry distant_window(coarse_patch);
  const std::size_t long_limit = static_cast<std::size_t>(
      std::numeric_limits<long>::max());
  distant_window.global_n_x = long_limit;
  distant_window.global_n_y = long_limit;
  distant_window.origin_x = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      long_limit - 16U);
  distant_window.origin_y = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      long_limit - 16U);
  distant_window.n_x = 16U;
  distant_window.n_y = 16U;
  distant_window.delta_x = 1.0f;
  distant_window.delta_y = 1.0f;
  distant_window.inverse_delta_x = 1.0f;
  distant_window.inverse_delta_y = 1.0f;
  distant_window.offset_x = 0.0f;
  distant_window.offset_y = 0.0f;
  require(distant_window.valid(),
          "large-origin mapper fixture is not valid");
  require(!feebl::adaptive_grid::map_deposit_to_level(
               2.0f, 2.0f, distant_window).accepted,
          "outside deposit was accepted by a distant adaptive window");
  require(!feebl::adaptive_grid::map_gather_to_level(
               2.0f, 2.0f, distant_window).accepted,
          "outside gather was accepted by a distant adaptive window");
  distant_window.offset_x = static_cast<float>(long_limit);
  distant_window.offset_y = static_cast<float>(long_limit);
  require(distant_window.valid(),
          "rounded-global-boundary mapper fixture is not valid");
  require(!feebl::adaptive_grid::map_deposit_to_level(
               0.0f, 0.0f, distant_window).accepted,
          "rounded global deposit boundary was converted to an integer");
  require(!feebl::adaptive_grid::map_gather_to_level(
               0.0f, 0.0f, distant_window).accepted,
          "rounded global gather boundary was converted to an integer");

  std::vector<double> linear(parent.cells(), 0.0);
  for (std::size_t ix = 0U; ix < parent.n_x; ++ix)
    for (std::size_t iy = 0U; iy < parent.n_y; ++iy) {
      const double x =
          (static_cast<double>(parent.origin_x) + ix + 0.5 -
           parent.offset_x) * parent.delta_x;
      const double y =
          (static_cast<double>(parent.origin_y) + iy + 0.5 -
           parent.offset_y) * parent.delta_y;
      linear[ix * parent.n_y + iy] = 3.0 * x - 2.0 * y + 7.0;
    }
  for (std::size_t ix = 2U; ix + 2U < fine_patch.n_x; ix += 3U)
    for (std::size_t iy = 2U; iy + 2U < fine_patch.n_y; iy += 3U) {
      const float x = static_cast<float>(
          (static_cast<double>(fine_patch.origin_x) + ix + 0.5 -
           fine_patch.offset_x) * fine_patch.delta_x);
      const float y = static_cast<float>(
          (static_cast<double>(fine_patch.origin_y) + iy + 0.5 -
           fine_patch.offset_y) * fine_patch.delta_y);
      const double prolonged =
          feebl::adaptive_grid::interpolate_level_potential(
              linear.data(), parent, x, y);
      require(std::fabs(prolonged - (3.0 * x - 2.0 * y + 7.0)) < 2e-5,
              "coarse-to-fine prolongation failed a linear potential");
    }

  const float boundary_x = static_cast<float>(
      (static_cast<double>(fine_patch.origin_x) - fine_patch.offset_x) *
      fine_patch.delta_x);
  const float centre_x = static_cast<float>(
      (static_cast<double>(fine_patch.origin_x) +
       0.5 * static_cast<double>(fine_patch.n_x) -
       fine_patch.offset_x) * fine_patch.delta_x);
  const float centre_y = static_cast<float>(
      (static_cast<double>(fine_patch.origin_y) +
       0.5 * static_cast<double>(fine_patch.n_y) -
       fine_patch.offset_y) * fine_patch.delta_y);
  require(feebl::adaptive_grid::patch_transition_weight(
              boundary_x, centre_y, fine_patch, 8U, 2U) == 0.0,
          "patch transition is not zero at its boundary");
  require(feebl::adaptive_grid::patch_transition_weight(
              centre_x, centre_y, fine_patch, 8U, 2U) == 1.0,
          "patch transition does not reach its fine interior");
  require_failure([&] {
    (void)feebl::adaptive_grid::patch_transition_weight(
        centre_x, centre_y, fine_patch, 0U, 2U);
  }, "patch transition accepted a zero-width non-C1 step");

  const auto transition_at_local = [&fine_patch](double local_x,
                                                  double local_y) {
    const float x = static_cast<float>(
        (static_cast<double>(fine_patch.origin_x) + local_x -
         fine_patch.offset_x) * fine_patch.delta_x);
    const float y = static_cast<float>(
        (static_cast<double>(fine_patch.origin_y) + local_y -
         fine_patch.offset_y) * fine_patch.delta_y);
    return feebl::adaptive_grid::patch_transition_weight(
        x, y, fine_patch, 8U, 2U);
  };
  require(fine_patch.n_x > 20U && fine_patch.n_y > 20U,
          "C1 transition fixture has no saturated upper/right interior");
  const auto transition_gradient = [&transition_at_local](double local_x,
                                                           double local_y) {
    const double h = 1.0e-2;
    return std::array<double, 2>{{
        (transition_at_local(local_x + h, local_y) -
         transition_at_local(local_x - h, local_y)) / (2.0 * h),
        (transition_at_local(local_x, local_y + h) -
         transition_at_local(local_x, local_y - h)) / (2.0 * h)}};
  };
  const double diagonal = 6.0;
  const double diagonal_offset = 1.0e-2;
  const std::array<double, 2> gradient_above = transition_gradient(
      diagonal + diagonal_offset, diagonal - diagonal_offset);
  const std::array<double, 2> gradient_below = transition_gradient(
      diagonal - diagonal_offset, diagonal + diagonal_offset);
  const double diagonal_gradient_jump = std::hypot(
      gradient_above[0] - gradient_below[0],
      gradient_above[1] - gradient_below[1]);
  require(diagonal_gradient_jump < 2.0e-2,
          "patch transition has a diagonal gradient kink");
  for (std::size_t ix = 0U; ix < fine_patch.n_x; ++ix) {
    const float lower_x = static_cast<float>(
        (static_cast<double>(fine_patch.origin_x) + ix + 0.5 -
         fine_patch.offset_x) * fine_patch.delta_x);
    const float upper_x = static_cast<float>(
        (static_cast<double>(fine_patch.origin_x) + fine_patch.n_x - ix -
         0.5 - fine_patch.offset_x) * fine_patch.delta_x);
    const double lower = feebl::adaptive_grid::patch_transition_weight(
        lower_x, centre_y, fine_patch, 8U, 2U);
    const double upper = feebl::adaptive_grid::patch_transition_weight(
        upper_x, centre_y, fine_patch, 8U, 2U);
    require(std::fabs(lower - upper) < 1e-12,
            "patch transition is not mirror-symmetric at cell centres");
  }
}

}  // namespace

int main() {
  try {
    test_parsers();
    test_longitudinal_encounter_tail_budget();
    test_exact_window_memory_cap_boundary();
    test_level_mapping_and_moments();
    test_field_operator_mapping_guard_and_legacy_preservation();
    test_fd4_lagrange_polynomial_and_phase_reproduction();
    test_fd4_lagrange_global_boundary_closure();
    test_fd4_lagrange_refinement_contraction();
    test_fixed_width_bspline_cell_integration();
    test_coherent_density_conservation_alignment_and_scale();
    test_coherent_density_patch_ownership();
    test_reference_restriction_sampling_regimes_and_admission();
    test_richardson();
    test_richardson_contraction_and_validation();
    test_symmetric_vector_field_sensitivity();
    test_vector_rms_quantile_field_error();
    test_mixed_vector_field_error_v2();
    test_vector_richardson_per_beam_order_and_quantile_support();
    test_patch_and_harmonic_solver();
    test_nested_geometry_prolongation_and_transition();
    std::cout << "adaptive resolution controller: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "adaptive resolution controller: FAIL: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
