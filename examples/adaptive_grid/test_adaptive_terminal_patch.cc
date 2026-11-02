#include "adaptiveTerminalPatchCPP.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

feebl::adaptive_grid::VectorFieldErrorStatistics statistics(
    double rms, double quantile, double maximum,
    std::size_t targets_per_beam = 1000U) {
  feebl::adaptive_grid::VectorFieldErrorStatistics value;
  value.reliable = true;
  value.rms = {{rms, 0.9 * rms}};
  value.quantile = {{quantile, 0.9 * quantile}};
  value.maximum = {{maximum, 0.9 * maximum}};
  value.accepted_targets = {{targets_per_beam, targets_per_beam}};
  value.accepted_target_hash = {{UINT64_C(0x11111111),
                                  UINT64_C(0x22222222)}};
  value.aggregate_rms = rms;
  value.aggregate_quantile = quantile;
  value.aggregate_maximum = maximum;
  value.total_accepted_targets = 2U * targets_per_beam;
  return value;
}

feebl::adaptive_grid::TerminalPatchPolicy policy() {
  feebl::adaptive_grid::TerminalPatchPolicy value;
  value.field_error_target = 0.01;
  value.luminosity_error_target = 0.01;
  value.outside_exposure_target = 0.002;
  value.outside_source_tail_target = 0.002;
  value.outside_luminosity_exposure_target = 0.002;
  value.minimum_kish_neff = 8.0;
  value.harmonic_residual_tolerance = 1.0e-5;
  value.boundary_flux_tolerance = 0.01;
  value.catastrophe_limit = 0.25;
  return value;
}

feebl::adaptive_grid::TerminalPatchEvidence smooth_gaussian() {
  feebl::adaptive_grid::TerminalPatchEvidence value;
  // Ratio-two, approximately second-order contraction.
  value.parent_to_r = statistics(0.012, 0.016, 0.06);
  value.r_to_r2 = statistics(0.003, 0.004, 0.02);
  value.luminosity_parent = 0.9996;
  value.luminosity_r = 1.0056;
  value.luminosity_r2 = 1.0071;
  value.outside_luminosity_earlier = 0.00038;
  value.outside_luminosity_previous = 0.000395;
  value.outside_luminosity_parent = 0.0004;
  value.luminosity_floor = 1.0e-30;
  value.outside_unresolved_exposure = 0.0005;
  value.outside_field_error_contribution = 0.0002;
  value.local_r_to_r2_ratios[0] = std::vector<double>(1000U, 0.004);
  value.local_r_to_r2_ratios[1] = std::vector<double>(1000U, 0.0036);
  value.local_target_hash = value.r_to_r2.accepted_target_hash;
  value.outside_earlier_to_previous_ratios[0] =
      std::vector<double>(1000U, 0.0008);
  value.outside_earlier_to_previous_ratios[1] =
      std::vector<double>(1000U, 0.0004);
  value.outside_error_surrogate_ratios[0] =
      std::vector<double>(1000U, 0.0002);
  value.outside_error_surrogate_ratios[1] =
      std::vector<double>(1000U, 0.0001);
  value.outside_earlier_target_hash = {{UINT64_C(0x33333333),
                                         UINT64_C(0x44444444)}};
  value.outside_latest_target_hash =
      value.outside_earlier_target_hash;
  value.outside_luminosity_fraction = 0.0004;
  value.outside_source_tail_fraction = 0.0004;
  value.minimum_kish_neff_r = 20.0;
  value.minimum_kish_neff_r2 = 12.0;
  value.maximum_charge_closure = 2.0e-13;
  value.maximum_moment_closure = 0.1;
  value.maximum_harmonic_residual = 1.0e-7;
  value.maximum_boundary_flux_rms = 0.001;
  value.maximum_boundary_flux_quantile = 0.003;
  value.maximum_boundary_flux_maximum = 0.006;
  value.transition_cells_r = 8U;
  value.transition_cells_r2 = 16U;
  value.stencil_guard_r = 2U;
  value.stencil_guard_r2 = 4U;
  return value;
}

void test_terminal_patch_resource_gate_honours_factor_cap() {
  using feebl::adaptive_grid::TerminalPatchRejectionResource;
  using feebl::adaptive_grid::evaluate_terminal_patch_resources;

  const auto capped = evaluate_terminal_patch_resources(
      8U, 16U, 4096U, 16384U, 1U << 20, 4U << 20,
      2U << 20, 4U, 65536U, 64U << 20);
  require(!capped.factor_capacity_ok && capped.cell_capacity_ok &&
              capped.byte_capacity_ok && !capped.permits_allocation(),
          "factor-4 cap permitted factor-8/factor-16 patch allocation");
  require(capped.rejection_mask == TerminalPatchRejectionResource,
          "factor-cap rejection did not report the resource bit");

  const auto unbounded = evaluate_terminal_patch_resources(
      8U, 16U, 4096U, 16384U, 1U << 20, 4U << 20,
      2U << 20, 0U, 65536U, 64U << 20);
  require(unbounded.factor_capacity_ok && unbounded.cell_capacity_ok &&
              unbounded.byte_capacity_ok && unbounded.permits_allocation(),
          "zero factor cap did not permit resource-admissible extension");

  const auto bounded = evaluate_terminal_patch_resources(
      8U, 16U, 4096U, 16384U, 1U << 20, 4U << 20,
      2U << 20, 16U, 65536U, 64U << 20);
  require(bounded.permits_allocation(),
          "inclusive factor-16 cap rejected factor-16 terminal patch");
}

void test_localized_smooth_gaussian_accepts() {
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      smooth_gaussian(), policy());
  require(result.accepted, "localized smooth Gaussian should certify");
  require(result.rejection_mask == 0U,
          "accepted Gaussian has rejection bits");
  require(result.observed_field_order > 1.9,
          "Gaussian field order was not recovered");
  require(result.composite_field_bound < 0.01,
          "Gaussian composite field budget did not close");
  require(result.composite_accepted_targets == 4000U &&
              result.composite_above_target == 0U &&
              result.composite_above_target_exposure == 0.0,
          "Gaussian complete-population telemetry is incorrect");
  require(result.composite_luminosity_bound < 0.01,
          "Gaussian luminosity budget did not close");
}

void test_nonlocalized_rejects() {
  auto evidence = smooth_gaussian();
  evidence.outside_unresolved_exposure = 0.01;
  evidence.outside_luminosity_fraction = 0.02;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted, "nonlocalized error was accepted");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionNonlocalized) != 0U,
          "nonlocalized rejection bit missing");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionLuminosityBudget) !=
              0U,
          "outside luminosity rejection bit missing");
}

void test_noncontracting_noise_rejects() {
  auto evidence = smooth_gaussian();
  evidence.r_to_r2 = statistics(0.013, 0.018, 0.08);
  evidence.luminosity_r2 = 1.02;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted, "noncontracting/noisy sequence was accepted");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionNoncontracting) != 0U,
          "noncontracting rejection bit missing");
}

void test_rare_outside_catastrophe_rejects() {
  auto evidence = smooth_gaussian();
  evidence.outside_error_surrogate_ratios[0].assign(500U, 0.0);
  evidence.outside_error_surrogate_ratios[1].assign(500U, 0.0);
  evidence.outside_earlier_to_previous_ratios[0].assign(500U, 0.0);
  evidence.outside_earlier_to_previous_ratios[1].assign(500U, 0.0);
  evidence.outside_error_surrogate_ratios[1].back() = 1.0;
  evidence.outside_earlier_to_previous_ratios[1].back() = 2.0;
  // One catastrophic target is below the localization exposure allowance
  // and its arithmetic mean would be only ~0.002.  The complete-population
  // maximum and RMS must nevertheless prevent certification.
  evidence.outside_unresolved_exposure = 1.0 / 1010.0;
  evidence.outside_field_error_contribution =
      1.0 / std::sqrt(1010.0);
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted,
          "rare outside catastrophic target was accepted");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionCatastrophe) != 0U,
          "outside catastrophe rejection bit missing");
  require(result.composite_field_maximum == 1.0,
          "outside maximum was not preserved in composite telemetry");
  require(result.composite_field_rms > 0.02,
          "outside RMS was incorrectly collapsed to an arithmetic mean");
}

void test_slow_outside_order_is_richardson_scaled() {
  auto evidence = smooth_gaussian();
  const double latest = 0.005;
  const double earlier = latest * std::pow(2.0, 0.25);
  evidence.outside_earlier_to_previous_ratios[0].assign(20U, earlier);
  evidence.outside_earlier_to_previous_ratios[1].assign(20U, earlier);
  evidence.outside_error_surrogate_ratios[0].assign(20U, latest);
  evidence.outside_error_surrogate_ratios[1].assign(20U, latest);
  evidence.outside_unresolved_exposure = 0.0;
  evidence.outside_field_error_contribution = latest;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted,
          "slowly converging outside error passed without scaling");
  require(result.observed_outside_field_order >= 0.249 &&
              result.observed_outside_field_order <= 0.251,
          "outside observed order was not recovered");
  require(result.outside_field_rms > 0.026,
          "outside error was not Richardson scaled");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionFieldBudget) != 0U,
          "scaled outside field-budget rejection bit missing");
}

void test_boundary_and_tail_reject() {
  auto evidence = smooth_gaussian();
  evidence.maximum_boundary_flux_quantile = 0.02;
  evidence.outside_source_tail_fraction = 0.01;
  evidence.transition_cells_r2 = evidence.stencil_guard_r2;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted, "boundary/tail violation was accepted");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionBoundaryFlux) != 0U,
          "boundary-flux rejection bit missing");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionNonlocalized) != 0U,
          "source-tail rejection bit missing");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionTransitionBuffer) !=
              0U,
          "transition-buffer rejection bit missing");
}

void test_argmax_switch_cannot_hide_growing_component() {
  auto evidence = smooth_gaussian();
  // The old aggregate-first logic saw max(0.08, 0.04) ->
  // max(0.02, 0.01) and inferred p=2.  Beam 0's quantile actually grows
  // 20-fold, so the per-beam/per-metric sequence is not asymptotic.
  evidence.parent_to_r.rms = {{0.08, 0.04}};
  evidence.parent_to_r.quantile = {{0.001, 0.04}};
  evidence.parent_to_r.aggregate_rms = 0.08;
  evidence.parent_to_r.aggregate_quantile = 0.04;
  evidence.r_to_r2.rms = {{0.02, 0.01}};
  evidence.r_to_r2.quantile = {{0.02, 0.01}};
  evidence.r_to_r2.aggregate_rms = 0.02;
  evidence.r_to_r2.aggregate_quantile = 0.02;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted,
          "aggregate argmax switching hid a growing beam/metric");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionNoncontracting) != 0U,
          "argmax-switch sequence missed noncontracting rejection");
  require(result.observed_field_order == 0.0,
          "growing beam/metric did not force zero observed order");
}

void test_boundary_maximum_is_a_hard_gate() {
  auto evidence = smooth_gaussian();
  evidence.maximum_boundary_flux_rms = 0.001;
  evidence.maximum_boundary_flux_quantile = 0.003;
  evidence.maximum_boundary_flux_maximum = 0.02;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted,
          "rare boundary-flux maximum was hidden by RMS/quantile");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionBoundaryFlux) != 0U,
          "boundary maximum rejection bit missing");
}

void test_outside_luminosity_requires_contracting_triplet() {
  auto evidence = smooth_gaussian();
  evidence.outside_luminosity_earlier = 0.0001;
  evidence.outside_luminosity_previous = 0.0002;
  evidence.outside_luminosity_parent = 0.0005;
  const auto result = feebl::adaptive_grid::certify_terminal_patch(
      evidence, policy());
  require(!result.accepted,
          "noncontracting outside-luminosity triplet was accepted");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionNoncontracting) != 0U,
          "outside-luminosity contraction rejection bit missing");
  require((result.rejection_mask &
           feebl::adaptive_grid::TerminalPatchRejectionLuminosityBudget) !=
              0U,
          "unbounded outside luminosity did not fail the luminosity budget");
}

void test_both_beams_and_empirical_quantile_support_are_required() {
  auto missing_beam = smooth_gaussian();
  missing_beam.parent_to_r.accepted_targets[1] = 0U;
  missing_beam.r_to_r2.accepted_targets[1] = 0U;
  missing_beam.parent_to_r.total_accepted_targets = 1000U;
  missing_beam.r_to_r2.total_accepted_targets = 1000U;
  missing_beam.local_r_to_r2_ratios[1].clear();
  const auto absent = feebl::adaptive_grid::certify_terminal_patch(
      missing_beam, policy());
  require(!absent.accepted &&
              (absent.rejection_mask &
               feebl::adaptive_grid::TerminalPatchRejectionInvalidEvidence),
          "terminal certificate accepted one unevaluated local beam");

  auto under_sampled = smooth_gaussian();
  under_sampled.parent_to_r = statistics(0.012, 0.016, 0.06, 499U);
  under_sampled.r_to_r2 = statistics(0.003, 0.004, 0.02, 499U);
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    under_sampled.local_r_to_r2_ratios[beam].resize(499U);
    under_sampled.outside_earlier_to_previous_ratios[beam].resize(500U);
    under_sampled.outside_error_surrogate_ratios[beam].resize(500U);
  }
  const auto insufficient = feebl::adaptive_grid::certify_terminal_patch(
      under_sampled, policy());
  require(!insufficient.accepted &&
              (insufficient.rejection_mask &
               feebl::adaptive_grid::TerminalPatchRejectionSampling),
          "terminal p99.9 certificate accepted 999 targets/beam");

  auto supported = smooth_gaussian();
  supported.parent_to_r = statistics(0.012, 0.016, 0.06, 500U);
  supported.r_to_r2 = statistics(0.003, 0.004, 0.02, 500U);
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    supported.local_r_to_r2_ratios[beam].resize(500U);
    supported.outside_earlier_to_previous_ratios[beam].resize(500U);
    supported.outside_error_surrogate_ratios[beam].resize(500U);
  }
  const auto enough = feebl::adaptive_grid::certify_terminal_patch(
      supported, policy());
  require(enough.accepted,
          "terminal p99.9 certificate rejected 1000 targets/beam");
}

void test_determinism_and_missing_evidence() {
  const auto first = feebl::adaptive_grid::certify_terminal_patch(
      smooth_gaussian(), policy());
  const auto second = feebl::adaptive_grid::certify_terminal_patch(
      smooth_gaussian(), policy());
  require(first.accepted == second.accepted &&
              first.rejection_mask == second.rejection_mask &&
              first.observed_field_order == second.observed_field_order &&
              first.observed_outside_field_order ==
                  second.observed_outside_field_order &&
              first.local_field_error == second.local_field_error &&
              first.composite_field_rms == second.composite_field_rms &&
              first.composite_field_quantile ==
                  second.composite_field_quantile &&
              first.composite_field_maximum ==
                  second.composite_field_maximum &&
              first.composite_accepted_targets ==
                  second.composite_accepted_targets &&
              first.composite_above_target ==
                  second.composite_above_target &&
              first.composite_above_target_exposure ==
                  second.composite_above_target_exposure &&
              first.maximum_vector_difference ==
                  second.maximum_vector_difference &&
              first.composite_field_bound == second.composite_field_bound &&
              first.observed_luminosity_order ==
                  second.observed_luminosity_order &&
              first.local_luminosity_error ==
                  second.local_luminosity_error &&
              first.composite_luminosity_bound ==
                  second.composite_luminosity_bound,
          "certificate is not bitwise-deterministic in scalar outputs");
  auto missing = smooth_gaussian();
  missing.r_to_r2.reliable = false;
  const auto rejected = feebl::adaptive_grid::certify_terminal_patch(
      missing, policy());
  require(!rejected.accepted &&
              (rejected.rejection_mask &
               feebl::adaptive_grid::TerminalPatchRejectionInvalidEvidence),
          "missing evidence did not fail closed");

  auto mismatched_identity = smooth_gaussian();
  mismatched_identity.local_target_hash[1] ^= UINT64_C(1);
  const auto identity_rejected =
      feebl::adaptive_grid::certify_terminal_patch(
          mismatched_identity, policy());
  require(!identity_rejected.accepted &&
              (identity_rejected.rejection_mask &
               feebl::adaptive_grid::TerminalPatchRejectionInvalidEvidence),
          "equal target counts hid a different target identity set");
}

void test_boundary_flux_manufactured_linear_field() {
  feebl::adaptive_grid::FieldLevelGeometry parent;
  parent.refinement_factor = 1U;
  parent.global_n_x = 12U;
  parent.global_n_y = 12U;
  parent.origin_x = 0;
  parent.origin_y = 0;
  parent.n_x = 12U;
  parent.n_y = 12U;
  parent.delta_x = 1.0;
  parent.delta_y = 1.0;
  parent.inverse_delta_x = 1.0;
  parent.inverse_delta_y = 1.0;
  parent.offset_x = 0.0;
  parent.offset_y = 0.0;
  feebl::adaptive_grid::NestedPatchBox box;
  box.valid = true;
  box.begin_x = 2U;
  box.end_x = 10U;
  box.begin_y = 2U;
  box.end_y = 10U;
  const auto patch = feebl::adaptive_grid::refine_patch_geometry(
      feebl::adaptive_grid::make_patch_geometry(parent, box), 2U);
  std::vector<double> parent_phi(parent.cells(), 0.0);
  for (std::size_t ix = 0U; ix < parent.n_x; ++ix)
    for (std::size_t iy = 0U; iy < parent.n_y; ++iy) {
      const double x = ix + 0.5;
      const double y = iy + 0.5;
      parent_phi[ix * parent.n_y + iy] = 2.0 * x + 3.0 * y;
    }
  std::vector<double> patch_phi(patch.cells(), 0.0);
  for (std::size_t ix = 0U; ix < patch.n_x; ++ix)
    for (std::size_t iy = 0U; iy < patch.n_y; ++iy) {
      const double x = (patch.origin_x + ix + 0.5 - patch.offset_x) *
                       patch.delta_x;
      const double y = (patch.origin_y + iy + 0.5 - patch.offset_y) *
                       patch.delta_y;
      patch_phi[ix * patch.n_y + iy] = 2.0 * x + 3.0 * y;
    }
  const auto matched =
      feebl::adaptive_grid::estimate_patch_boundary_flux_mismatch(
          parent_phi.data(), parent, patch_phi, patch, 0.99, 1.0e-12);
  require(matched.reliable && matched.maximum < 1.0e-12,
          "linear manufactured patch did not close its boundary flux");
  for (std::size_t ix = 0U; ix < patch.n_x; ++ix)
    for (std::size_t iy = 0U; iy < patch.n_y; ++iy)
      patch_phi[ix * patch.n_y + iy] += 0.1 * ix;
  const auto mismatched =
      feebl::adaptive_grid::estimate_patch_boundary_flux_mismatch(
          parent_phi.data(), parent, patch_phi, patch, 0.99, 1.0e-12);
  require(mismatched.reliable && mismatched.quantile > 0.01,
          "manufactured normal-gradient mismatch was not detected");
}

}  // namespace

int main() {
  try {
    test_terminal_patch_resource_gate_honours_factor_cap();
    test_localized_smooth_gaussian_accepts();
    test_nonlocalized_rejects();
    test_noncontracting_noise_rejects();
    test_rare_outside_catastrophe_rejects();
    test_slow_outside_order_is_richardson_scaled();
    test_boundary_and_tail_reject();
    test_argmax_switch_cannot_hide_growing_component();
    test_boundary_maximum_is_a_hard_gate();
    test_outside_luminosity_requires_contracting_triplet();
    test_both_beams_and_empirical_quantile_support_are_required();
    test_determinism_and_missing_evidence();
    test_boundary_flux_manufactured_linear_field();
  } catch (const std::exception& error) {
    std::cerr << "adaptive terminal patch test failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "adaptive terminal patch tests passed\n";
  return EXIT_SUCCESS;
}
