#include "wavefrontFieldWorkspaceCPP.h"
#include "adaptiveTerminalPatchCPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace feebl {
namespace wavefront_fields {

bool finite_value(float value) noexcept { return std::isfinite(value); }

bool finite_value(double value) noexcept { return std::isfinite(value); }

bool finite_value(long double value) noexcept { return std::isfinite(value); }

namespace runtime_safety {

LuminosityOracleEstimate estimate_fixed_width_luminosity_error(
    double luminosity0, double luminosity1, double luminosity2,
    std::size_t target_slot, double refinement_ratio,
    double luminosity_floor) {
  if (!std::isfinite(luminosity0) || !std::isfinite(luminosity1) ||
      !std::isfinite(luminosity2) || luminosity0 < 0.0 ||
      luminosity1 < 0.0 || luminosity2 < 0.0 || target_slot > 2U ||
      !std::isfinite(refinement_ratio) || !(refinement_ratio > 1.0) ||
      !std::isfinite(luminosity_floor) || !(luminosity_floor > 0.0))
    throw std::invalid_argument(
        "invalid fixed-width luminosity-oracle input");

  LuminosityOracleEstimate result;
  const double scale = std::max(
      std::max(luminosity0, std::max(luminosity1, luminosity2)),
      luminosity_floor);
  const double difference1 = std::fabs(luminosity1 - luminosity0);
  const double difference2 = std::fabs(luminosity2 - luminosity1);
  const double roundoff =
      256.0 * std::numeric_limits<double>::epsilon() * scale;
  const double relative_roundoff = roundoff / scale;

  // Three values that agree to the arithmetic noise floor are stable only to
  // that floor, never exactly error-free.  Conversely, a significant first
  // change followed by a roundoff plateau has no measurable convergence
  // order and must not be interpreted as superconvergence.
  if (difference1 <= roundoff && difference2 <= roundoff) {
    result.relative_error = relative_roundoff;
    result.reliable = true;
    return result;
  }
  if (difference1 <= roundoff || difference2 <= roundoff ||
      !(difference2 < difference1)) {
    result.relative_error = std::max(
        1.0, std::max(difference1, difference2) / scale);
    return result;
  }

  const double raw_order =
      std::log(difference1 / difference2) / std::log(refinement_ratio);
  // The most accurate selectable operator is fourth order.  A larger order
  // inferred from only three luminosity values is cancellation or accidental
  // superconvergence, not evidence for a stronger production error bound.
  const double minimum_order = 0.25;
  const double maximum_order = 4.0;
  if (!std::isfinite(raw_order) || raw_order < minimum_order) {
    result.relative_error = std::max(1.0, difference2 / scale);
    return result;
  }
  result.observed_order = std::min(maximum_order, raw_order);
  const double ratio_to_order =
      std::pow(refinement_ratio, result.observed_order);
  const double divisor = ratio_to_order - 1.0;
  if (!std::isfinite(ratio_to_order) || !(divisor > 0.0)) {
    result.relative_error = std::max(1.0, difference2 / scale);
    return result;
  }

  double extrapolated = 0.0;
  double direct_difference = difference2;
  if (target_slot == 0U) {
    extrapolated = ratio_to_order * difference1 / divisor / scale;
    direct_difference = difference1;
  } else if (target_slot == 1U) {
    extrapolated = ratio_to_order * difference2 / divisor / scale;
  } else {
    extrapolated = difference2 / divisor / scale;
  }
  result.relative_error = std::max(
      std::max(extrapolated, direct_difference / scale),
      relative_roundoff);
  if (!std::isfinite(result.relative_error) || result.relative_error < 0.0) {
    result.relative_error = 1.0;
    return result;
  }
  result.reliable = true;
  return result;
}

std::size_t reachable_adaptive_cell_bound(
    std::uint64_t max_cells, std::size_t max_bytes,
    std::size_t bootstrap_context_bytes,
    std::size_t fixed_transient_bytes,
    std::size_t context_bytes_per_cell,
    std::size_t scratch_bytes_per_cell,
    std::size_t minimum_worker_lanes) {
  if (max_cells == 0U || max_bytes == 0U ||
      context_bytes_per_cell == 0U || scratch_bytes_per_cell == 0U ||
      minimum_worker_lanes == 0U)
    throw std::invalid_argument(
        "invalid adaptive reachable-cell capacity input");
  if (max_cells > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error(
        "adaptive maximum cell count does not fit host size_t");
  if (context_bytes_per_cell >
      std::numeric_limits<std::size_t>::max() - scratch_bytes_per_cell)
    throw std::overflow_error(
        "adaptive per-cell context/scratch sum overflows size_t");

  const std::size_t declared_cells = static_cast<std::size_t>(max_cells);
  const std::size_t per_lane_ceiling = max_bytes / minimum_worker_lanes;
  const std::size_t per_cell_combined =
      context_bytes_per_cell + scratch_bytes_per_cell;
  // Dividing the complete bootstrap estimate by the variable per-cell cost
  // deliberately overestimates the largest bootstrap member because that
  // estimate also contains every other bootstrap context and fixed overhead.
  const std::size_t bootstrap_cell_bound = std::min(
      declared_cells,
      bootstrap_context_bytes / context_bytes_per_cell +
          (bootstrap_context_bytes % context_bytes_per_cell != 0U ? 1U : 0U));
  if (fixed_transient_bytes >= per_lane_ceiling ||
      bootstrap_context_bytes >= per_lane_ceiling - fixed_transient_bytes)
    return bootstrap_cell_bound;
  const std::size_t available = per_lane_ceiling - fixed_transient_bytes -
                                bootstrap_context_bytes;
  const std::size_t dynamic_cell_bound = available / per_cell_combined;
  return std::min(
      declared_cells, std::max(bootstrap_cell_bound, dynamic_cell_bound));
}

AdaptiveGridCertificateDecision evaluate_adaptive_grid_certificate(
    std::uint64_t encounters,
    std::uint64_t unmet_encounters,
    std::uint64_t noise_stop_encounters,
    std::uint64_t vector_catastrophe_failure_encounters,
    double maximum_vector_difference,
    double catastrophe_limit,
    std::uint64_t terminal_patch_attempts,
    std::uint64_t terminal_patch_accepts,
    std::uint64_t terminal_patch_used,
    std::uint64_t terminal_patch_rejects,
    std::uint64_t terminal_patch_uniform_fallbacks,
    std::uint64_t terminal_patch_resource_rejects,
    std::uint64_t terminal_patch_nonresource_rejects,
    std::uint32_t terminal_patch_rejection_mask_or,
    bool root_replay_required) {
  if (unmet_encounters > encounters)
    throw std::logic_error(
        "adaptive unmet-error count exceeds encounter count");
  if (noise_stop_encounters > unmet_encounters)
    throw std::logic_error(
        "adaptive noise-stop count exceeds unmet-error count");
  if (vector_catastrophe_failure_encounters > encounters)
    throw std::logic_error(
        "adaptive vector-catastrophe count exceeds encounter count");
  const std::uint32_t terminal_resource_bit =
      adaptive_grid::TerminalPatchRejectionResource;
  const std::uint32_t terminal_nonresource_mask =
      terminal_patch_rejection_mask_or & ~terminal_resource_bit;
  const bool terminal_rejection_reasons_cover_all =
      terminal_patch_resource_rejects >
          std::numeric_limits<std::uint64_t>::max() -
              terminal_patch_nonresource_rejects ||
      terminal_patch_resource_rejects +
              terminal_patch_nonresource_rejects >=
          terminal_patch_rejects;
  if (terminal_patch_accepts > terminal_patch_attempts ||
      terminal_patch_used != terminal_patch_accepts ||
      terminal_patch_rejects > terminal_patch_attempts ||
      terminal_patch_uniform_fallbacks != terminal_patch_rejects ||
      terminal_patch_resource_rejects > terminal_patch_rejects ||
      terminal_patch_nonresource_rejects > terminal_patch_rejects ||
      !terminal_rejection_reasons_cover_all ||
      terminal_patch_accepts >
          std::numeric_limits<std::uint64_t>::max() -
              terminal_patch_rejects ||
      terminal_patch_accepts + terminal_patch_rejects !=
          terminal_patch_attempts ||
      ((terminal_patch_resource_rejects != 0U) !=
       ((terminal_patch_rejection_mask_or & terminal_resource_bit) != 0U)) ||
      ((terminal_patch_nonresource_rejects != 0U) !=
       (terminal_nonresource_mask != 0U)) ||
      ((terminal_patch_rejects != 0U) !=
       (terminal_patch_rejection_mask_or != 0U)))
    throw std::logic_error(
        "adaptive terminal-patch counters are inconsistent");
  if (!finite_value(maximum_vector_difference) ||
      maximum_vector_difference < 0.0 ||
      maximum_vector_difference > 2.0 ||
      !finite_value(catastrophe_limit) || catastrophe_limit <= 0.0 ||
      catastrophe_limit > 2.0)
    throw std::logic_error(
        "adaptive catastrophe summary bounds are invalid");

  AdaptiveGridCertificateDecision result;
  result.certified_encounters = encounters - unmet_encounters;
  result.unresolved_non_noise_encounters =
      unmet_encounters - noise_stop_encounters;
  result.resource_unmet_encounters =
      result.unresolved_non_noise_encounters;
  result.external_replay_recommended =
      root_replay_required || unmet_encounters != 0U ||
      noise_stop_encounters != 0U ||
      vector_catastrophe_failure_encounters != 0U ||
      terminal_patch_rejects != 0U ||
      terminal_patch_rejection_mask_or != 0U;
  result.all_encounters_certified =
      encounters != 0U && result.certified_encounters == encounters &&
      unmet_encounters == 0U && noise_stop_encounters == 0U &&
      result.unresolved_non_noise_encounters == 0U &&
      vector_catastrophe_failure_encounters == 0U &&
      terminal_patch_rejection_mask_or == 0U &&
      maximum_vector_difference <= catastrophe_limit &&
      terminal_patch_attempts == terminal_patch_accepts &&
      terminal_patch_rejects == 0U &&
      !result.external_replay_recommended;
  return result;
}

}  // namespace runtime_safety

void validate_workspace_configuration(const Configuration& configuration) {
  if (configuration.n_x < 4 || configuration.n_y < 4 ||
      configuration.n_z <= 0 || configuration.n_t <= 0 ||
      !(configuration.cut_x > 0.0f) || !(configuration.cut_y > 0.0f) ||
      !(configuration.cut_z > 0.0f))
    throw std::invalid_argument(
        "wavefront field workspace received invalid grid geometry");
  if (!finite_value(configuration.macro_charge1) ||
      !finite_value(configuration.macro_charge2) ||
      !finite_value(configuration.physical_particles1) ||
      !finite_value(configuration.physical_particles2) ||
      configuration.physical_particles1 < 0.0 ||
      configuration.physical_particles2 < 0.0)
    throw std::invalid_argument(
        "wavefront field workspace received non-finite macro charge");
  if (configuration.field_solver != sparse_direct_fields::Mode::Fft &&
      (!configuration.sparse_cells ||
       configuration.sparse_direct_interaction_limit == 0U))
    throw std::invalid_argument(
        "sparse-direct wavefront fields require sparse cells and a positive "
        "interaction crossover");
  if (configuration.adaptive_field_operator != "legacy_cic12_v1" &&
      configuration.adaptive_field_operator != "fd4_lagrange4_v1")
    throw std::invalid_argument(
        "adaptive field operator must be legacy_cic12_v1 or "
        "fd4_lagrange4_v1");
  if (configuration.adaptive_error_norm_policy !=
          "component_rms_linf_v1" &&
      configuration.adaptive_error_norm_policy !=
          "vector_rms_quantile_v1" &&
      configuration.adaptive_error_norm_policy !=
          "vector_mixed_rms_quantile_v2")
    throw std::invalid_argument(
        "adaptive error norm policy must be component_rms_linf_v1, "
        "vector_rms_quantile_v1, or vector_mixed_rms_quantile_v2");
  if (!finite_value(configuration.adaptive_error_quantile) ||
      configuration.adaptive_error_quantile < 0.95 ||
      configuration.adaptive_error_quantile >= 1.0)
    throw std::invalid_argument(
        "adaptive vector error quantile must be finite in [0.95,1)");
  if (!finite_value(configuration.adaptive_error_catastrophe_limit) ||
      !(configuration.adaptive_error_catastrophe_limit > 0.0) ||
      configuration.adaptive_error_catastrophe_limit > 2.0)
    throw std::invalid_argument(
        "adaptive vector catastrophe limit must be finite in (0,2]");
  const bool mixed_vector_policy =
      configuration.adaptive_error_norm_policy ==
          "vector_mixed_rms_quantile_v2";
  if (mixed_vector_policy) {
    const auto canonical_budget_sha256 = [](const std::string& value) {
      if (value.size() != 64U) return false;
      for (char character : value)
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f')))
          return false;
      return true;
    };
    if (!canonical_budget_sha256(
            configuration.adaptive_error_budget_sha256) ||
        !finite_value(configuration.adaptive_error_reference_field) ||
        !(configuration.adaptive_error_reference_field > 0.0) ||
        !finite_value(
            configuration.adaptive_error_absolute_floor_fraction) ||
        !(configuration.adaptive_error_absolute_floor_fraction > 0.0) ||
        configuration.adaptive_error_absolute_floor_fraction > 1.0 ||
        !finite_value(
            configuration.adaptive_error_catastrophe_absolute_fraction) ||
        !(configuration.adaptive_error_catastrophe_absolute_fraction > 0.0) ||
        configuration.adaptive_error_catastrophe_absolute_fraction > 2.0 ||
        !finite_value(configuration.adaptive_error_exposure_limit) ||
        !(configuration.adaptive_error_exposure_limit > 0.0) ||
        !(configuration.adaptive_error_exposure_limit < 1.0))
      throw std::invalid_argument(
          "mixed V2 field-error policy requires an exact lowercase SHA-256 "
          "budget binding, a positive reference field, floor/catastrophe "
          "fractions in range, and exposure limit in (0,1)");
  } else if (!configuration.adaptive_error_budget_sha256.empty() ||
             configuration.adaptive_error_reference_field != 0.0 ||
             configuration.adaptive_error_absolute_floor_fraction != 0.0 ||
             configuration.adaptive_error_catastrophe_absolute_fraction !=
                 0.0 ||
             configuration.adaptive_error_exposure_limit != 0.0) {
    throw std::invalid_argument(
        "mixed V2 field-error budget/scale settings are nonempty or nonzero "
        "while another error policy is selected");
  }
  if (configuration.adaptive_grid_mode < 2 &&
      (configuration.adaptive_field_operator != "legacy_cic12_v1" ||
       configuration.adaptive_max_refinement_factor != 0U ||
       configuration.adaptive_error_norm_policy !=
           "component_rms_linf_v1" ||
       configuration.adaptive_noise_aware_stop ||
       configuration.adaptive_terminal_patch_certification))
    throw std::invalid_argument(
        "adaptive field operator, error policy, noise stop, terminal patch, "
        "and maximum refinement factor are only meaningful in adaptive grid "
        "modes 2/3");
  if (configuration.adaptive_grid_mode >= 2) {
    if (configuration.adaptive_terminal_patch_certification &&
        (configuration.adaptive_grid_mode != 3 ||
         configuration.adaptive_source_model !=
             "fixed_width_bspline1_v1" ||
         configuration.adaptive_field_operator != "fd4_lagrange4_v1" ||
         configuration.adaptive_error_norm_policy !=
             "vector_rms_quantile_v1" ||
         configuration.adaptive_richardson_order_source != "three_level" ||
         configuration.adaptive_macroparticles.mode !=
             adaptive_particles::Mode::LegacyOff ||
         configuration.adaptive_profile != "encounter"))
      throw std::invalid_argument(
          "terminal Mode-3 two-patch certification requires the continuous "
          "fixed-width source, fd4/vector-quantile/three-level controls, "
          "adaptive macroparticles off, and encounter telemetry");
    if ((configuration.adaptive_error_norm_policy ==
             "vector_rms_quantile_v1" ||
         configuration.adaptive_error_norm_policy ==
             "vector_mixed_rms_quantile_v2") &&
        (configuration.adaptive_richardson_order_source != "three_level" ||
         !configuration.adaptive_richardson_order_table.empty()))
      throw std::invalid_argument(
          "vector quantile policies require rolling three_level order and "
          "an empty calibrated-order table");
    if (configuration.adaptive_noise_aware_stop &&
        ((configuration.adaptive_error_norm_policy !=
              "vector_rms_quantile_v1" &&
          configuration.adaptive_error_norm_policy !=
              "vector_mixed_rms_quantile_v2") ||
         (configuration.adaptive_macroparticles.mode !=
              adaptive_particles::Mode::SplitOnly &&
          configuration.adaptive_macroparticles.mode !=
              adaptive_particles::Mode::CoupledSplitMerge) ||
         configuration.adaptive_fail_on_unmet_error))
      throw std::invalid_argument(
          "adaptive noise-aware stopping requires a vector quantile policy "
          "with non-strict pilot publication and a replay-capable adaptive "
          "macroparticle mode");
    if (configuration.adaptive_max_refinement_factor != 0U &&
        (configuration.adaptive_max_refinement_factor &
         (configuration.adaptive_max_refinement_factor - 1U)) != 0U)
      throw std::invalid_argument(
          "adaptive maximum refinement factor must be zero (no user factor "
          "cap) or a power of two");
    if (configuration.adaptive_max_refinement_factor != 0U &&
        (configuration.adaptive_spacing_factors.empty() ||
         configuration.adaptive_max_refinement_factor <
             configuration.adaptive_spacing_factors.back()))
      throw std::invalid_argument(
          "adaptive maximum refinement factor must be at least the final "
          "bootstrap spacing factor");
    if (!finite_value(configuration.adaptive_encounter_tail_fraction) ||
        configuration.adaptive_encounter_tail_fraction < 0.0 ||
        configuration.adaptive_encounter_tail_fraction >= 1.0)
      throw std::invalid_argument(
          "Mode 2/3 encounter-tail eligibility fraction must be zero "
          "(inherit tau_tail) or finite in (0,1)");
    if (configuration.adaptive_source_model !=
            "canonical_cell_average_v1" &&
        configuration.adaptive_source_model != "reference_cic_v1" &&
        configuration.adaptive_source_model != "fixed_width_bspline1_v1")
      throw std::invalid_argument(
          "Mode 2/3 requires an explicit supported adaptive source model");
    if (configuration.adaptive_source_model == "reference_cic_v1") {
      const std::size_t nominal_x = static_cast<std::size_t>(configuration.n_x);
      const std::size_t nominal_y = static_cast<std::size_t>(configuration.n_y);
      const std::size_t reference_x =
          configuration.adaptive_source_reference_nx;
      const std::size_t reference_y =
          configuration.adaptive_source_reference_ny;
      const std::size_t int_max = static_cast<std::size_t>(
          std::numeric_limits<int>::max());
      if (reference_x == 0U || reference_y == 0U ||
          reference_x % nominal_x != 0U ||
          reference_y % nominal_y != 0U)
        throw std::invalid_argument(
            "reference_cic_v1 source dimensions must be integer multiples "
            "of the nominal grid");
      if (reference_x > int_max || reference_y > int_max ||
          reference_x > int_max / reference_y)
        throw std::invalid_argument(
            "reference_cic_v1 source cells exceed the signed-int linear "
            "index contract of legacy CIC deposition");
      const std::size_t factor_x = reference_x / nominal_x;
      const std::size_t factor_y = reference_y / nominal_y;
      if (factor_x == 0U || factor_x != factor_y ||
          (factor_x & (factor_x - 1U)) != 0U)
        throw std::invalid_argument(
            "reference_cic_v1 requires equal power-of-two source factors");
      if (configuration.adaptive_anchor_x != 0.0 ||
          configuration.adaptive_anchor_y != 0.0)
        throw std::invalid_argument(
            "reference_cic_v1 requires exactly zero registered anchors");
      if (!(configuration.adaptive_source_error_target > 0.0) ||
          !(configuration.adaptive_solver_error_target > 0.0) ||
          !(configuration.adaptive_lumi_error_target > 0.0) ||
          !finite_value(configuration.adaptive_source_error_target) ||
          !finite_value(configuration.adaptive_solver_error_target) ||
          !finite_value(configuration.adaptive_lumi_error_target) ||
          !finite_value(configuration.adaptive_lumi_floor) ||
          configuration.adaptive_source_error_target >= 1.0 ||
          configuration.adaptive_solver_error_target >= 1.0 ||
          configuration.adaptive_lumi_error_target >= 1.0 ||
          !(configuration.adaptive_lumi_floor > 0.0))
        throw std::invalid_argument(
            "reference_cic_v1 requires finite source/solver/luminosity "
            "targets in (0,1) and a positive luminosity floor");
      if (configuration.adaptive_richardson_order_source != "three_level" ||
          !configuration.adaptive_richardson_order_table.empty())
        throw std::invalid_argument(
            "reference_cic_v1 requires measured three_level Richardson order");
      for (std::size_t field_factor :
           configuration.adaptive_spacing_factors)
        if (field_factor == 0U ||
            (field_factor < factor_x && factor_x % field_factor != 0U) ||
            (field_factor > factor_x && field_factor % factor_x != 0U))
          throw std::invalid_argument(
              "reference_cic_v1 field classes are not registered with the "
              "source lattice");
      bool has_identity_level = false;
      for (std::size_t field_factor :
           configuration.adaptive_spacing_factors)
        has_identity_level = has_identity_level || field_factor == factor_x;
      if (!has_identity_level)
        throw std::invalid_argument(
            "reference_cic_v1 field classes must contain the source-factor "
            "identity level");
      if (configuration.adaptive_source_width_x != 0.0 ||
          configuration.adaptive_source_width_y != 0.0)
        throw std::invalid_argument(
            "reference_cic_v1 requires zero fixed source widths");
    } else if (configuration.adaptive_source_model ==
               "fixed_width_bspline1_v1") {
      if (configuration.adaptive_macroparticles.mode !=
          adaptive_particles::Mode::LegacyOff)
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 currently requires adaptive "
            "macroparticles off; adaptive source projection is qualified "
            "only for reference_cic_v1");
      if (configuration.adaptive_source_reference_nx != 0U ||
          configuration.adaptive_source_reference_ny != 0U)
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 requires zero reference dimensions");
      if (!finite_value(configuration.adaptive_source_width_x) ||
          !finite_value(configuration.adaptive_source_width_y) ||
          !(configuration.adaptive_source_width_x > 0.0) ||
          !(configuration.adaptive_source_width_y > 0.0))
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 requires finite positive physical "
            "half-widths");
      if (configuration.adaptive_anchor_x != 0.0 ||
          configuration.adaptive_anchor_y != 0.0)
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 requires exactly zero anchors");
      if (!(configuration.adaptive_source_error_target > 0.0) ||
          !(configuration.adaptive_solver_error_target > 0.0) ||
          !(configuration.adaptive_lumi_error_target > 0.0) ||
          !finite_value(configuration.adaptive_source_error_target) ||
          !finite_value(configuration.adaptive_solver_error_target) ||
          !finite_value(configuration.adaptive_lumi_error_target) ||
          !finite_value(configuration.adaptive_lumi_floor) ||
          configuration.adaptive_source_error_target >= 1.0 ||
          configuration.adaptive_solver_error_target >= 1.0 ||
          configuration.adaptive_lumi_error_target >= 1.0 ||
          !(configuration.adaptive_lumi_floor > 0.0))
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 requires finite source/solver/"
            "luminosity targets in (0,1) and a positive luminosity floor");
      if (configuration.adaptive_richardson_order_source != "three_level" ||
          !configuration.adaptive_richardson_order_table.empty())
        throw std::invalid_argument(
            "fixed_width_bspline1_v1 requires measured three_level "
            "Richardson order");
    } else if (configuration.adaptive_source_reference_nx != 0U ||
               configuration.adaptive_source_reference_ny != 0U ||
               configuration.adaptive_source_width_x != 0.0 ||
               configuration.adaptive_source_width_y != 0.0 ||
               configuration.adaptive_source_error_target != 0.0 ||
               configuration.adaptive_solver_error_target != 0.0 ||
               configuration.adaptive_lumi_error_target != 0.0 ||
               configuration.adaptive_lumi_floor != 0.0) {
      throw std::invalid_argument(
          "canonical_cell_average_v1 cannot consume reference-source or "
          "fixed-width-source controls");
    }
  }
}

}  // namespace wavefront_fields
}  // namespace feebl
