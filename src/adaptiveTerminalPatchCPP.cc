#include "adaptiveTerminalPatchCPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace feebl {
namespace adaptive_grid {
namespace {

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool finite_probability(double value) {
  return finite_nonnegative(value) && value <= 1.0;
}

double observed_order(double earlier, double later, double ratio) {
  if (earlier == 0.0 && later == 0.0)
    return 16.0;
  if (!(earlier > 0.0) || !(later >= 0.0) || !(later < earlier) ||
      !(ratio > 1.0))
    return 0.0;
  if (later == 0.0)
    return 16.0;
  const double value = std::log(earlier / later) / std::log(ratio);
  return std::isfinite(value) && value > 0.0
      ? std::min(16.0, value) : 0.0;
}

double richardson_fine_error(double difference, double ratio,
                             double order) {
  if (!finite_nonnegative(difference) || !(ratio > 1.0) || !(order > 0.0))
    return std::numeric_limits<double>::infinity();
  const double divisor = std::pow(ratio, order) - 1.0;
  if (!(divisor > 0.0) || !std::isfinite(divisor))
    return std::numeric_limits<double>::infinity();
  return difference / divisor;
}

}  // namespace

TerminalPatchResourceDecision evaluate_terminal_patch_resources(
    std::size_t factor_r, std::size_t factor_r2,
    std::size_t cells_r, std::size_t cells_r2,
    std::size_t bytes_r, std::size_t bytes_r2,
    std::size_t retained_context_bytes,
    std::size_t maximum_refinement_factor,
    std::size_t maximum_cells,
    std::size_t maximum_bytes) {
  if (factor_r == 0U || factor_r2 == 0U || factor_r2 < factor_r ||
      cells_r == 0U || cells_r2 == 0U || maximum_bytes == 0U)
    throw std::invalid_argument(
        "invalid terminal-patch resource decision input");

  TerminalPatchResourceDecision result;
  result.factor_capacity_ok =
      maximum_refinement_factor == 0U ||
      (factor_r <= maximum_refinement_factor &&
       factor_r2 <= maximum_refinement_factor);
  result.cell_capacity_ok =
      maximum_cells == 0U ||
      (cells_r <= maximum_cells && cells_r2 <= maximum_cells);
  result.byte_capacity_ok =
      retained_context_bytes <= maximum_bytes &&
      std::max(bytes_r, bytes_r2) <=
          maximum_bytes - retained_context_bytes;
  result.rejection_mask =
      result.factor_capacity_ok && result.cell_capacity_ok &&
              result.byte_capacity_ok
          ? TerminalPatchRejectionNone
          : TerminalPatchRejectionResource;
  return result;
}

PatchBoundaryFluxStatistics estimate_patch_boundary_flux_mismatch(
    const double* parent_phi,
    const FieldLevelGeometry& parent_geometry,
    const std::vector<double>& patch_phi,
    const FieldLevelGeometry& patch_geometry,
    double quantile_probability,
    double field_floor) {
  PatchBoundaryFluxStatistics result;
  if (!parent_phi || !parent_geometry.valid() || !patch_geometry.valid() ||
      patch_phi.size() != patch_geometry.cells() ||
      patch_geometry.refinement_factor <= parent_geometry.refinement_factor ||
      patch_geometry.refinement_factor %
              parent_geometry.refinement_factor != 0U ||
      !std::isfinite(quantile_probability) || quantile_probability < 0.5 ||
      quantile_probability >= 1.0 || !std::isfinite(field_floor) ||
      !(field_floor > 0.0))
    return result;
  const std::size_t nx = patch_geometry.n_x;
  const std::size_t ny = patch_geometry.n_y;
  if (nx < 4U || ny < 4U) return result;
  std::vector<double> ratios;
  ratios.reserve(2U * (nx + ny));
  long double squared_sum = 0.0L;
  const auto coordinate_x = [&patch_geometry](std::size_t ix) {
    return static_cast<float>(
        (static_cast<double>(patch_geometry.origin_x) + ix + 0.5 -
         patch_geometry.offset_x) * patch_geometry.delta_x);
  };
  const auto coordinate_y = [&patch_geometry](std::size_t iy) {
    return static_cast<float>(
        (static_cast<double>(patch_geometry.origin_y) + iy + 0.5 -
         patch_geometry.offset_y) * patch_geometry.delta_y);
  };
  const auto append = [&ratios, &squared_sum, field_floor](
      double patch_flux, double parent_flux) {
    if (!std::isfinite(patch_flux) || !std::isfinite(parent_flux))
      return false;
    const double scale = std::max(
        field_floor, std::max(std::fabs(patch_flux),
                             std::fabs(parent_flux)));
    const double ratio = std::fabs(patch_flux - parent_flux) / scale;
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 2.0)
      return false;
    ratios.push_back(ratio);
    squared_sum += static_cast<long double>(ratio) * ratio;
    return true;
  };
  const auto at = [&patch_phi, ny](std::size_t ix, std::size_t iy) {
    return patch_phi[ix * ny + iy];
  };
  for (std::size_t iy = 0U; iy < ny; ++iy) {
    const float y = coordinate_y(iy);
    const float x0 = coordinate_x(0U);
    const float x1 = coordinate_x(1U);
    const float xn0 = coordinate_x(nx - 1U);
    const float xn1 = coordinate_x(nx - 2U);
    if (!append((at(1U, iy) - at(0U, iy)) / patch_geometry.delta_x,
                (interpolate_level_potential(parent_phi, parent_geometry,
                                             x1, y) -
                 interpolate_level_potential(parent_phi, parent_geometry,
                                             x0, y)) /
                    patch_geometry.delta_x) ||
        !append((at(nx - 2U, iy) - at(nx - 1U, iy)) /
                    patch_geometry.delta_x,
                (interpolate_level_potential(parent_phi, parent_geometry,
                                             xn1, y) -
                 interpolate_level_potential(parent_phi, parent_geometry,
                                             xn0, y)) /
                    patch_geometry.delta_x))
      return PatchBoundaryFluxStatistics();
  }
  for (std::size_t ix = 0U; ix < nx; ++ix) {
    const float x = coordinate_x(ix);
    const float y0 = coordinate_y(0U);
    const float y1 = coordinate_y(1U);
    const float yn0 = coordinate_y(ny - 1U);
    const float yn1 = coordinate_y(ny - 2U);
    if (!append((at(ix, 1U) - at(ix, 0U)) / patch_geometry.delta_y,
                (interpolate_level_potential(parent_phi, parent_geometry,
                                             x, y1) -
                 interpolate_level_potential(parent_phi, parent_geometry,
                                             x, y0)) /
                    patch_geometry.delta_y) ||
        !append((at(ix, ny - 2U) - at(ix, ny - 1U)) /
                    patch_geometry.delta_y,
                (interpolate_level_potential(parent_phi, parent_geometry,
                                             x, yn1) -
                 interpolate_level_potential(parent_phi, parent_geometry,
                                             x, yn0)) /
                    patch_geometry.delta_y))
      return PatchBoundaryFluxStatistics();
  }
  if (ratios.empty()) return result;
  result.samples = ratios.size();
  result.rms = std::sqrt(static_cast<double>(
      squared_sum / static_cast<long double>(ratios.size())));
  result.maximum = *std::max_element(ratios.begin(), ratios.end());
  const std::size_t rank = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(
              quantile_probability * static_cast<double>(ratios.size()))));
  std::nth_element(ratios.begin(), ratios.begin() + rank - 1U, ratios.end());
  result.quantile = ratios[rank - 1U];
  result.reliable = std::isfinite(result.rms) &&
                    std::isfinite(result.quantile) &&
                    std::isfinite(result.maximum);
  return result;
}

TerminalPatchCertificate certify_terminal_patch(
    const TerminalPatchEvidence& evidence,
    const TerminalPatchPolicy& policy) {
  TerminalPatchCertificate result;
  const double ratio = static_cast<double>(evidence.refinement_ratio);
  const auto ratios_are_valid = [](const std::vector<double>& values) {
    for (double value : values)
      if (!std::isfinite(value) || value < 0.0 || value > 2.0)
        return false;
    return true;
  };
  bool complete_target_partition = true;
  std::array<std::size_t, 2> composite_targets_per_beam{{0U, 0U}};
  std::size_t composite_target_count = 0U;
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    const std::size_t local_count =
        evidence.local_r_to_r2_ratios[beam].size();
    const std::size_t outside_count =
        evidence.outside_error_surrogate_ratios[beam].size();
    const std::size_t outside_earlier_count =
        evidence.outside_earlier_to_previous_ratios[beam].size();
    if (local_count == 0U ||
        local_count != evidence.r_to_r2.accepted_targets[beam] ||
        local_count != evidence.parent_to_r.accepted_targets[beam] ||
        evidence.parent_to_r.accepted_target_hash[beam] == 0U ||
        evidence.parent_to_r.accepted_target_hash[beam] !=
            evidence.r_to_r2.accepted_target_hash[beam] ||
        evidence.local_target_hash[beam] !=
            evidence.r_to_r2.accepted_target_hash[beam] ||
        evidence.outside_earlier_target_hash[beam] == 0U ||
        evidence.outside_earlier_target_hash[beam] !=
            evidence.outside_latest_target_hash[beam] ||
        outside_count != outside_earlier_count ||
        local_count > std::numeric_limits<std::size_t>::max() -
                          outside_count ||
        composite_target_count > std::numeric_limits<std::size_t>::max() -
                                     local_count - outside_count ||
        !ratios_are_valid(evidence.local_r_to_r2_ratios[beam]) ||
        !ratios_are_valid(
            evidence.outside_earlier_to_previous_ratios[beam]) ||
        !ratios_are_valid(evidence.outside_error_surrogate_ratios[beam])) {
      complete_target_partition = false;
      break;
    }
    composite_targets_per_beam[beam] = local_count + outside_count;
    composite_target_count += local_count + outside_count;
  }
  const bool valid_policy =
      evidence.refinement_ratio >= 2U &&
      (evidence.refinement_ratio & (evidence.refinement_ratio - 1U)) == 0U &&
      policy.field_error_target > 0.0 &&
      policy.field_error_target < 1.0 &&
      policy.luminosity_error_target > 0.0 &&
      policy.luminosity_error_target < 1.0 &&
      finite_probability(policy.outside_exposure_target) &&
      finite_probability(policy.outside_source_tail_target) &&
      finite_probability(policy.outside_luminosity_exposure_target) &&
      policy.minimum_kish_neff >= 1.0 &&
      finite_nonnegative(policy.charge_closure_tolerance) &&
      finite_nonnegative(policy.moment_closure_tolerance) &&
      policy.harmonic_residual_tolerance > 0.0 &&
      policy.harmonic_residual_tolerance < 1.0 &&
      policy.boundary_flux_tolerance > 0.0 &&
      policy.boundary_flux_tolerance < 1.0 &&
      policy.catastrophe_limit > 0.0 &&
      policy.catastrophe_limit <= 2.0 &&
      policy.minimum_observed_order > 0.0 &&
      policy.minimum_observed_order <= 16.0 &&
      std::isfinite(policy.quantile_probability) &&
      policy.quantile_probability >= 0.95 &&
      policy.quantile_probability < 1.0 &&
      evidence.parent_to_r.reliable && evidence.r_to_r2.reliable &&
      evidence.parent_to_r.total_accepted_targets != 0U &&
      evidence.parent_to_r.total_accepted_targets ==
          evidence.r_to_r2.total_accepted_targets &&
      evidence.parent_to_r.accepted_targets ==
          evidence.r_to_r2.accepted_targets &&
      finite_nonnegative(evidence.parent_to_r.aggregate_rms) &&
      finite_nonnegative(evidence.parent_to_r.aggregate_quantile) &&
      finite_nonnegative(evidence.r_to_r2.aggregate_rms) &&
      finite_nonnegative(evidence.r_to_r2.aggregate_quantile) &&
      finite_nonnegative(evidence.r_to_r2.aggregate_maximum) &&
      finite_nonnegative(evidence.luminosity_parent) &&
      finite_nonnegative(evidence.luminosity_r) &&
      finite_nonnegative(evidence.luminosity_r2) &&
      finite_nonnegative(evidence.outside_luminosity_earlier) &&
      finite_nonnegative(evidence.outside_luminosity_previous) &&
      finite_nonnegative(evidence.outside_luminosity_parent) &&
      evidence.luminosity_floor > 0.0 &&
      std::isfinite(evidence.luminosity_floor) &&
      finite_probability(evidence.outside_unresolved_exposure) &&
      finite_nonnegative(evidence.outside_field_error_contribution) &&
      finite_probability(evidence.outside_luminosity_fraction) &&
      finite_probability(evidence.outside_source_tail_fraction) &&
      finite_nonnegative(evidence.minimum_kish_neff_r) &&
      finite_nonnegative(evidence.minimum_kish_neff_r2) &&
      finite_nonnegative(evidence.maximum_charge_closure) &&
      finite_nonnegative(evidence.maximum_moment_closure) &&
      finite_nonnegative(evidence.maximum_harmonic_residual) &&
      finite_nonnegative(evidence.maximum_boundary_flux_rms) &&
      finite_nonnegative(evidence.maximum_boundary_flux_quantile) &&
      finite_nonnegative(evidence.maximum_boundary_flux_maximum) &&
      complete_target_partition && composite_target_count != 0U;
  if (!valid_policy) {
    result.rejection_mask |= TerminalPatchRejectionInvalidEvidence;
    return result;
  }
  result.composite_accepted_targets = composite_target_count;

  if (evidence.outside_unresolved_exposure >
          policy.outside_exposure_target ||
      evidence.outside_source_tail_fraction >
          policy.outside_source_tail_target)
    result.rejection_mask |= TerminalPatchRejectionNonlocalized;
  if (evidence.minimum_kish_neff_r < policy.minimum_kish_neff ||
      evidence.minimum_kish_neff_r2 < policy.minimum_kish_neff)
    result.rejection_mask |= TerminalPatchRejectionSampling;
  const long double tail_probability =
      1.0L - static_cast<long double>(policy.quantile_probability);
  const std::size_t minimum_quantile_targets = static_cast<std::size_t>(
      std::ceil(1.0L / tail_probability));
  if (composite_targets_per_beam[0] < minimum_quantile_targets ||
      composite_targets_per_beam[1] < minimum_quantile_targets)
    result.rejection_mask |= TerminalPatchRejectionSampling;
  if (evidence.maximum_charge_closure >
      policy.charge_closure_tolerance)
    result.rejection_mask |= TerminalPatchRejectionChargeClosure;
  if (evidence.maximum_moment_closure >
      policy.moment_closure_tolerance)
    result.rejection_mask |= TerminalPatchRejectionMomentClosure;
  if (evidence.maximum_harmonic_residual >
      policy.harmonic_residual_tolerance)
    result.rejection_mask |= TerminalPatchRejectionHarmonicResidual;
  if (std::max(evidence.maximum_boundary_flux_maximum,
               std::max(evidence.maximum_boundary_flux_rms,
                        evidence.maximum_boundary_flux_quantile)) >
      policy.boundary_flux_tolerance)
    result.rejection_mask |= TerminalPatchRejectionBoundaryFlux;
  if (evidence.transition_cells_r <= evidence.stencil_guard_r ||
      evidence.transition_cells_r2 <= evidence.stencil_guard_r2)
    result.rejection_mask |= TerminalPatchRejectionTransitionBuffer;

  struct DistributionStatistics {
    double rms = 0.0;
    double quantile = 0.0;
    double maximum = 0.0;
  };
  const auto summarize = [&policy](std::vector<double> values) {
    DistributionStatistics summary;
    if (values.empty()) return summary;
    long double squared_sum = 0.0L;
    for (double value : values) {
      squared_sum += static_cast<long double>(value) * value;
      summary.maximum = std::max(summary.maximum, value);
    }
    summary.rms = std::sqrt(static_cast<double>(
        squared_sum / static_cast<long double>(values.size())));
    const std::size_t rank = std::max<std::size_t>(
        1U, static_cast<std::size_t>(std::ceil(
                policy.quantile_probability *
                static_cast<double>(values.size()))));
    std::nth_element(values.begin(), values.begin() + rank - 1U,
                     values.end());
    summary.quantile = values[rank - 1U];
    return summary;
  };

  const double later_field = std::max(
      evidence.r_to_r2.aggregate_rms,
      evidence.r_to_r2.aggregate_quantile);
  result.observed_field_order = 16.0;
  bool have_local_component = false;
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    if (evidence.r_to_r2.accepted_targets[beam] == 0U) continue;
    have_local_component = true;
    result.observed_field_order = std::min(
        result.observed_field_order,
        observed_order(evidence.parent_to_r.rms[beam],
                       evidence.r_to_r2.rms[beam], ratio));
    result.observed_field_order = std::min(
        result.observed_field_order,
        observed_order(evidence.parent_to_r.quantile[beam],
                       evidence.r_to_r2.quantile[beam], ratio));
  }
  if (!have_local_component) {
    result.rejection_mask |= TerminalPatchRejectionInvalidEvidence;
    return result;
  }

  std::size_t outside_target_count = 0U;
  double raw_outside_maximum = 0.0;
  result.observed_outside_field_order = 16.0;
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    const DistributionStatistics earlier = summarize(
        evidence.outside_earlier_to_previous_ratios[beam]);
    const DistributionStatistics later = summarize(
        evidence.outside_error_surrogate_ratios[beam]);
    if (!evidence.outside_error_surrogate_ratios[beam].empty()) {
      result.observed_outside_field_order = std::min(
          result.observed_outside_field_order,
          observed_order(earlier.rms, later.rms, ratio));
      result.observed_outside_field_order = std::min(
          result.observed_outside_field_order,
          observed_order(earlier.quantile, later.quantile, ratio));
    }
    raw_outside_maximum = std::max(raw_outside_maximum, later.maximum);
    outside_target_count +=
        evidence.outside_error_surrogate_ratios[beam].size();
  }
  result.maximum_vector_difference = std::max(
      evidence.r_to_r2.aggregate_maximum, raw_outside_maximum);
  if (result.maximum_vector_difference > policy.catastrophe_limit)
    result.rejection_mask |= TerminalPatchRejectionCatastrophe;

  const double order_roundoff = 64.0 *
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, policy.minimum_observed_order);
  const bool local_order_ok = result.observed_field_order + order_roundoff >=
      policy.minimum_observed_order;
  const bool outside_order_ok =
      result.observed_outside_field_order + order_roundoff >=
          policy.minimum_observed_order;
  if (!local_order_ok || !outside_order_ok) {
    result.rejection_mask |= TerminalPatchRejectionNoncontracting;
    result.composite_field_bound =
        std::numeric_limits<double>::infinity();
  } else {
    result.local_field_error = richardson_fine_error(
        later_field, ratio, result.observed_field_order);
    const double local_divisor =
        std::pow(ratio, result.observed_field_order) - 1.0;
    const double outside_divisor = outside_target_count == 0U
        ? 1.0
        : std::pow(ratio, result.observed_outside_field_order) - 1.0;
    for (std::size_t beam = 0U; beam < 2U; ++beam) {
      std::size_t outside_unresolved = 0U;
      std::vector<double> composite_errors;
      std::vector<double> outside_errors;
      composite_errors.reserve(
          evidence.local_r_to_r2_ratios[beam].size() +
          evidence.outside_error_surrogate_ratios[beam].size());
      outside_errors.reserve(
          evidence.outside_error_surrogate_ratios[beam].size());
      for (double difference : evidence.local_r_to_r2_ratios[beam]) {
        const double estimated_error = difference / local_divisor;
        composite_errors.push_back(estimated_error);
        if (estimated_error > policy.field_error_target)
          ++result.composite_above_target;
      }
      for (double difference :
           evidence.outside_error_surrogate_ratios[beam]) {
        const double estimated_error = difference / outside_divisor;
        composite_errors.push_back(estimated_error);
        outside_errors.push_back(estimated_error);
        if (estimated_error > policy.field_error_target)
          ++result.composite_above_target;
        if (estimated_error > policy.field_error_target)
          ++outside_unresolved;
      }
      const DistributionStatistics composite = summarize(
          composite_errors);
      const DistributionStatistics outside = summarize(outside_errors);
      result.composite_field_rms = std::max(
          result.composite_field_rms, composite.rms);
      result.composite_field_quantile = std::max(
          result.composite_field_quantile, composite.quantile);
      result.composite_field_maximum = std::max(
          result.composite_field_maximum, composite.maximum);
      result.outside_field_rms = std::max(
          result.outside_field_rms, outside.rms);
      result.outside_field_quantile = std::max(
          result.outside_field_quantile, outside.quantile);
      result.outside_field_maximum = std::max(
          result.outside_field_maximum, outside.maximum);
      const std::size_t beam_composite_targets =
          evidence.local_r_to_r2_ratios[beam].size() +
          evidence.outside_error_surrogate_ratios[beam].size();
      result.outside_unresolved_exposure = std::max(
          result.outside_unresolved_exposure,
          static_cast<double>(outside_unresolved) /
              static_cast<double>(beam_composite_targets));
    }
    result.composite_field_bound = std::max(
        result.composite_field_rms,
        result.composite_field_quantile);
    result.composite_above_target_exposure =
        static_cast<double>(result.composite_above_target) /
        static_cast<double>(result.composite_accepted_targets);
    if (result.composite_field_maximum > policy.catastrophe_limit)
      result.rejection_mask |= TerminalPatchRejectionCatastrophe;
    if (result.outside_unresolved_exposure >
        policy.outside_exposure_target)
      result.rejection_mask |= TerminalPatchRejectionNonlocalized;
  }
  if (!std::isfinite(result.composite_field_bound) ||
      result.composite_field_bound > policy.field_error_target)
    result.rejection_mask |= TerminalPatchRejectionFieldBudget;

  const double local_d0 = std::fabs(
      evidence.luminosity_r - evidence.luminosity_parent);
  const double local_d1 = std::fabs(
      evidence.luminosity_r2 - evidence.luminosity_r);
  const double outside_d0 = std::fabs(
      evidence.outside_luminosity_previous -
      evidence.outside_luminosity_earlier);
  const double outside_d1 = std::fabs(
      evidence.outside_luminosity_parent -
      evidence.outside_luminosity_previous);
  result.observed_luminosity_order = observed_order(
      local_d0, local_d1, ratio);
  result.observed_outside_luminosity_order = observed_order(
      outside_d0, outside_d1, ratio);
  const bool local_luminosity_order_ok =
      result.observed_luminosity_order + order_roundoff >=
          policy.minimum_observed_order;
  const bool outside_luminosity_order_ok =
      result.observed_outside_luminosity_order + order_roundoff >=
          policy.minimum_observed_order;
  if (!local_luminosity_order_ok || !outside_luminosity_order_ok) {
    result.rejection_mask |= TerminalPatchRejectionNoncontracting;
    result.composite_luminosity_bound =
        std::numeric_limits<double>::infinity();
  } else {
    const double final_luminosity =
        evidence.luminosity_r2 + evidence.outside_luminosity_parent;
    const double luminosity_scale = std::max(
        std::fabs(final_luminosity), evidence.luminosity_floor);
    result.local_luminosity_error = richardson_fine_error(
        local_d1, ratio, result.observed_luminosity_order) /
        luminosity_scale;
    result.outside_luminosity_error = richardson_fine_error(
        outside_d1, ratio,
        result.observed_outside_luminosity_order) /
        luminosity_scale;
    result.composite_luminosity_bound =
        result.local_luminosity_error +
        result.outside_luminosity_error;
  }
  if (!std::isfinite(result.composite_luminosity_bound) ||
      result.composite_luminosity_bound >
          policy.luminosity_error_target ||
      evidence.outside_luminosity_fraction >
          policy.outside_luminosity_exposure_target)
    result.rejection_mask |= TerminalPatchRejectionLuminosityBudget;

  result.accepted = result.rejection_mask == TerminalPatchRejectionNone;
  return result;
}

}  // namespace adaptive_grid
}  // namespace feebl
