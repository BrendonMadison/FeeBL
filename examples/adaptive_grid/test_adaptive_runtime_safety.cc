#include "wavefrontFieldWorkspaceCPP.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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

void test_fixed_width_luminosity_oracle() {
  using feebl::wavefront_fields::runtime_safety::
      estimate_fixed_width_luminosity_error;

  const auto stable = estimate_fixed_width_luminosity_error(
      3.0, 3.0, 3.0, 2U, 2.0, 1.0e-300);
  require(stable.reliable, "roundoff-stable luminosities were rejected");
  require(stable.relative_error > 0.0 && stable.relative_error < 1.0e-12,
          "roundoff-stable luminosities were labelled exactly error-free");

  // L(h)=1+h^2 at h={1,1/2,1/4} has exact observed order two.
  const auto quadratic = estimate_fixed_width_luminosity_error(
      2.0, 1.25, 1.0625, 2U, 2.0, 1.0e-300);
  require(quadratic.reliable,
          "exact second-order luminosity sequence was rejected");
  require(std::fabs(quadratic.observed_order - 2.0) < 1.0e-12,
          "exact second-order luminosity sequence has the wrong order");
  require(quadratic.relative_error >= 0.1875 / 2.0,
          "luminosity oracle dropped below its adjacent-difference floor");

  // A significant first change followed only by representational noise is a
  // plateau, not evidence for an arbitrarily high convergence order.
  const double plateau = std::nextafter(1.01,
                                        std::numeric_limits<double>::infinity());
  const auto roundoff_plateau = estimate_fixed_width_luminosity_error(
      1.0, 1.01, plateau, 2U, 2.0, 1.0e-300);
  require(!roundoff_plateau.reliable &&
              roundoff_plateau.relative_error >= 1.0,
          "roundoff plateau falsely certified luminosity convergence");

  // A resolvable apparent superconvergence is capped at the formal order of
  // the highest selectable field operator, and the raw final change remains
  // an independent lower bound.
  const auto capped = estimate_fixed_width_luminosity_error(
      1.0, 1.01, 1.0100000001, 2U, 2.0, 1.0e-300);
  require(capped.reliable,
          "resolvable contracting luminosity sequence was rejected");
  require(std::fabs(capped.observed_order - 4.0) < 1.0e-12,
          "luminosity superconvergence order was not capped at four");
  require(capped.relative_error >=
              std::fabs(1.0100000001 - 1.01) / 1.0100000001,
          "capped luminosity oracle dropped below the direct difference");

  const auto noncontracting = estimate_fixed_width_luminosity_error(
      1.0, 1.1, 1.3, 1U, 2.0, 1.0e-300);
  require(!noncontracting.reliable && noncontracting.relative_error >= 1.0,
          "noncontracting luminosity sequence was certified");
  require_failure([&] {
    (void)estimate_fixed_width_luminosity_error(
        1.0, 1.0, 1.0, 3U, 2.0, 1.0e-300);
  }, "luminosity oracle accepted an invalid target slot");
}

void test_reachable_cell_bound() {
  using feebl::wavefront_fields::runtime_safety::
      reachable_adaptive_cell_bound;
  const std::uint64_t declared_cells = UINT64_C(1000000000000);
  const std::size_t max_bytes = std::size_t(4U) * 1024U * 1024U * 1024U;
  const std::size_t bootstrap = std::size_t(64U) * 1024U * 1024U;
  const std::size_t fixed = std::size_t(16U) * 1024U * 1024U;
  const std::size_t context_per_cell = 224U;
  const std::size_t scratch_per_cell = 256U;
  const std::size_t bound = reachable_adaptive_cell_bound(
      declared_cells, max_bytes, bootstrap, fixed,
      context_per_cell, scratch_per_cell, 2U);
  const std::size_t per_lane = max_bytes / 2U;
  const std::size_t expected_dynamic =
      (per_lane - fixed - bootstrap) /
      (context_per_cell + scratch_per_cell);
  require(bound == expected_dynamic,
          "reachable-cell bound does not honor the combined byte budget");
  require(bound < static_cast<std::size_t>(declared_cells),
          "declared cell safety cap leaked into transient reservation");
  require(bound <=
              (per_lane - fixed - bootstrap) /
                  (context_per_cell + scratch_per_cell),
          "reachable-cell bound can exceed a worker-lane high-water");

  const std::size_t explicit_small = reachable_adaptive_cell_bound(
      4096U, max_bytes, bootstrap, fixed,
      context_per_cell, scratch_per_cell, 2U);
  require(explicit_small == 4096U,
          "explicit smaller cell cap was not authoritative");
  require_failure([&] {
    (void)reachable_adaptive_cell_bound(
        4096U, max_bytes, bootstrap, fixed,
        0U, scratch_per_cell, 2U);
  }, "reachable-cell bound accepted zero context bytes per cell");
}

void test_vector_policy_runtime_preflight() {
  using feebl::wavefront_fields::Configuration;
  using feebl::wavefront_fields::validate_workspace_configuration;
  Configuration base;
  base.n_x = 4;
  base.n_y = 4;
  base.n_z = 1;
  base.n_t = 1;
  base.cut_x = 1.0f;
  base.cut_y = 1.0f;
  base.cut_z = 1.0f;

  Configuration bad_policy = base;
  bad_policy.adaptive_error_norm_policy = "mean_only_guess";
  require_failure([&] { validate_workspace_configuration(bad_policy); },
                  "runtime preflight accepted an unknown error norm policy");

  Configuration bad_quantile = base;
  bad_quantile.adaptive_error_quantile = 0.90;
  require_failure([&] { validate_workspace_configuration(bad_quantile); },
                  "runtime preflight accepted a non-high quantile");

  Configuration bad_catastrophe = base;
  bad_catastrophe.adaptive_error_catastrophe_limit = 2.01;
  require_failure(
      [&] { validate_workspace_configuration(bad_catastrophe); },
      "runtime preflight accepted a catastrophe limit above the exact bound");

  Configuration wrong_mode = base;
  wrong_mode.adaptive_grid_mode = 0;
  wrong_mode.adaptive_error_norm_policy = "vector_rms_quantile_v1";
  require_failure([&] { validate_workspace_configuration(wrong_mode); },
                  "vector error policy was accepted outside Modes 2/3");

  Configuration mixed = base;
  mixed.adaptive_grid_mode = 2;
  mixed.adaptive_spacing_factors = {1U, 2U, 4U};
  mixed.adaptive_source_model = "canonical_cell_average_v1";
  mixed.adaptive_richardson_order_source = "three_level";
  mixed.adaptive_error_norm_policy = "vector_mixed_rms_quantile_v2";
  mixed.adaptive_error_budget_sha256 = std::string(64U, 'a');
  mixed.adaptive_error_reference_field = 12.5;
  mixed.adaptive_error_absolute_floor_fraction = 1.0e-3;
  mixed.adaptive_error_catastrophe_absolute_fraction = 1.0e-2;
  mixed.adaptive_error_exposure_limit = 1.0e-3;
  validate_workspace_configuration(mixed);

  Configuration missing_budget = mixed;
  missing_budget.adaptive_error_budget_sha256.clear();
  require_failure(
      [&] { validate_workspace_configuration(missing_budget); },
      "mixed V2 policy accepted a missing prospective budget digest");
  Configuration short_budget = mixed;
  short_budget.adaptive_error_budget_sha256 = std::string(63U, 'a');
  require_failure(
      [&] { validate_workspace_configuration(short_budget); },
      "mixed V2 policy accepted a short prospective budget digest");
  Configuration uppercase_budget = mixed;
  uppercase_budget.adaptive_error_budget_sha256 = std::string(64U, 'A');
  require_failure(
      [&] { validate_workspace_configuration(uppercase_budget); },
      "mixed V2 policy accepted a noncanonical uppercase budget digest");

  Configuration zero_reference = mixed;
  zero_reference.adaptive_error_reference_field = 0.0;
  require_failure(
      [&] { validate_workspace_configuration(zero_reference); },
      "mixed V2 policy accepted a zero physical reference field");
  Configuration zero_floor = mixed;
  zero_floor.adaptive_error_absolute_floor_fraction = 0.0;
  require_failure([&] { validate_workspace_configuration(zero_floor); },
                  "mixed V2 policy accepted a zero absolute floor");
  Configuration zero_absolute_catastrophe = mixed;
  zero_absolute_catastrophe
      .adaptive_error_catastrophe_absolute_fraction = 0.0;
  require_failure(
      [&] { validate_workspace_configuration(zero_absolute_catastrophe); },
      "mixed V2 policy accepted a zero absolute catastrophe threshold");
  Configuration bad_exposure = mixed;
  bad_exposure.adaptive_error_exposure_limit = 1.0;
  require_failure([&] { validate_workspace_configuration(bad_exposure); },
                  "mixed V2 policy accepted an exposure limit of one");

  Configuration dead_mixed_controls = base;
  dead_mixed_controls.adaptive_error_reference_field = 1.0;
  require_failure(
      [&] { validate_workspace_configuration(dead_mixed_controls); },
      "a legacy policy accepted a dead V2 reference-field control");
  Configuration dead_budget = base;
  dead_budget.adaptive_error_budget_sha256 = std::string(64U, 'a');
  require_failure(
      [&] { validate_workspace_configuration(dead_budget); },
      "a legacy policy accepted a dead V2 budget digest");
}

void test_adaptive_grid_certificate_decision_invariants() {
  using feebl::wavefront_fields::runtime_safety::
      evaluate_adaptive_grid_certificate;
  const auto clean = evaluate_adaptive_grid_certificate(
      8U, 0U, 0U, 0U, 0.1, 0.5,
      2U, 2U, 2U, 0U, 0U, 0U, 0U, 0U, false);
  require(clean.certified_encounters == 8U &&
              clean.unresolved_non_noise_encounters == 0U &&
              clean.resource_unmet_encounters == 0U &&
              !clean.external_replay_recommended &&
              clean.all_encounters_certified,
          "clean adaptive certificate decision was not accepted");

  const auto replay = evaluate_adaptive_grid_certificate(
      8U, 0U, 0U, 0U, 0.1, 0.5,
      2U, 2U, 2U, 0U, 0U, 0U, 0U, 0U, true);
  require(replay.external_replay_recommended &&
              !replay.all_encounters_certified,
          "external replay coexisted with all-encounters certification");

  require_failure([&] {
    (void)evaluate_adaptive_grid_certificate(
        8U, 1U, 2U, 0U, 0.1, 0.5,
        0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, false);
  }, "noise-stop counter underflow was not rejected");
  require_failure([&] {
    (void)evaluate_adaptive_grid_certificate(
        8U, 0U, 0U, 9U, 0.1, 0.5,
        0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, false);
  }, "catastrophe counter above encounters was not rejected");
  require_failure([&] {
    (void)evaluate_adaptive_grid_certificate(
        8U, 0U, 0U, 0U, 0.1, 0.5,
        2U, 1U, 1U, 0U, 0U, 0U, 0U, 0U, false);
  }, "incomplete terminal-patch disposition was not rejected");

  const auto unresolved = evaluate_adaptive_grid_certificate(
      8U, 3U, 1U, 0U, 0.1, 0.5,
      2U, 0U, 0U, 2U, 2U, 1U, 2U,
      (1U << 12) | (1U << 6), false);
  require(unresolved.unresolved_non_noise_encounters == 2U &&
              unresolved.resource_unmet_encounters == 2U &&
              !unresolved.all_encounters_certified,
          "explicit unresolved-non-noise semantics are incorrect");

  require_failure([&] {
    (void)evaluate_adaptive_grid_certificate(
        8U, 1U, 0U, 0U, 0.1, 0.5,
        1U, 1U, 0U, 0U, 0U, 0U, 0U, 0U, false);
  }, "accepted-but-unused terminal patch was not rejected");
  require_failure([&] {
    (void)evaluate_adaptive_grid_certificate(
        8U, 1U, 0U, 0U, 0.1, 0.5,
        1U, 0U, 0U, 1U, 0U, 1U, 0U, 1U << 12, false);
  }, "terminal patch rejection without uniform fallback was not rejected");
}

}  // namespace

int main() {
  try {
    test_fixed_width_luminosity_oracle();
    test_reachable_cell_bound();
    test_vector_policy_runtime_preflight();
    test_adaptive_grid_certificate_decision_invariants();
    std::cout << "adaptive runtime safety: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "adaptive runtime safety: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
