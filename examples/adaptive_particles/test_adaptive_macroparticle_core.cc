#include "adaptiveMacroparticleCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using feebl::adaptive_particles::Configuration;
using feebl::adaptive_particles::Mode;
using feebl::adaptive_particles::RootProjection;
using feebl::adaptive_particles::SourceEnsemble;
using feebl::adaptive_particles::SourceRepresentative;
using feebl::adaptive_grid::FieldLevelGeometry;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void require_throws(Function function, const char* message) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception&) {
    rejected = true;
  }
  require(rejected, message);
}

FieldLevelGeometry geometry() {
  FieldLevelGeometry result;
  result.n_x = 8U;
  result.n_y = 8U;
  result.global_n_x = 8U;
  result.global_n_y = 8U;
  result.origin_x = 0;
  result.origin_y = 0;
  result.refinement_factor = 1U;
  result.delta_x = 0.5f;
  result.delta_y = 0.25f;
  result.inverse_delta_x = 2.0f;
  result.inverse_delta_y = 4.0f;
  result.offset_x = 4.0f;
  result.offset_y = 4.0f;
  return result;
}

RootProjection root(std::uint64_t ordinal, double x, double y,
                    double weight, std::uint64_t cohort = 11U) {
  RootProjection result;
  result.root_id = make_generated_primary_particle_id(1U, ordinal);
  result.statistical_cohort_id = cohort;
  result.x = x;
  result.y = y;
  result.physical_weight = weight;
  result.cic = feebl::adaptive_grid::map_deposit_to_level(
      static_cast<float>(x), static_cast<float>(y), geometry());
  require(result.cic.accepted, "test root must lie in geometry");
  return result;
}

Configuration config(Mode mode) {
  Configuration result;
  result.mode = mode;
  result.error_target = mode == Mode::Diagnostics ? 0.0 : 0.005;
  result.resample_bias_target = 1.0e-12;
  result.maximum_change_factor = 4.0;
  result.maximum_generation = 8U;
  return result;
}

void test_split_and_clone_support() {
  const std::vector<RootProjection> roots = {
      root(0U, -0.63, -0.19, 2.0),
      root(1U, 0.31, 0.27, 1.0)};
  const SourceEnsemble source =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), roots, 1U, 7U);
  require(source.receipt.committed, "split proposal did not commit");
  require(source.receipt.exact_projection,
          "split proposal is not exact projection");
  require(source.metrics.distinct_root_count == 2U,
          "split changed distinct roots");
  require(source.metrics.representative_count >= 4U,
          "split did not create representatives");
  const double root_ess = 9.0 / 5.0;
  require(std::fabs(source.metrics.clone_support_proxy - root_ess) < 1e-12,
          "split manufactured clone support");
  require(source.metrics.weight_ess > source.metrics.clone_support_proxy,
          "negative control did not distinguish weight and clone ESS");

  std::vector<double> density;
  std::vector<double> s1;
  std::vector<double> s2;
  feebl::adaptive_particles::deposit_source(
      source, geometry(), density, s1, s2);
  double density_sum = 0.0;
  for (double value : density) density_sum += value;
  require(std::fabs(density_sum - 3.0) < 1e-13,
          "projected density lost charge");
  for (std::size_t cell = 0U; cell < density.size(); ++cell)
    require(std::fabs(density[cell] - s1[cell]) < 1e-13,
            "root grouped S1 differs from density");
}

void test_exact_cell_merge() {
  // Coincident roots exercise a many-to-many same-cell merge while the exact
  // ancestry ledger retains their individual S2 contributions.
  const std::vector<RootProjection> roots = {
      root(2U, -0.13, 0.08, 1.0),
      root(3U, -0.13, 0.08, 3.0)};
  const SourceEnsemble split =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), roots, 1U, 9U);
  const SourceEnsemble merged =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::CoupledSplitMerge), geometry(), roots, 1U, 9U);
  require(merged.receipt.committed, "merge proposal did not commit");
  require(merged.metrics.many_root_representatives > 0U,
          "merge lost many-root provenance");
  require(merged.metrics.representative_count <
              split.metrics.representative_count,
          "merge did not reduce representative count");
  require(merged.receipt.merge_groups > 0U,
          "merge did not count groups with an actual reduction");
  require(std::fabs(merged.metrics.clone_support_proxy - 1.6) < 1e-12,
          "merge changed clone support");

  std::vector<double> split_rho, split_s1, split_s2;
  std::vector<double> merged_rho, merged_s1, merged_s2;
  feebl::adaptive_particles::deposit_source(
      split, geometry(), split_rho, split_s1, split_s2);
  feebl::adaptive_particles::deposit_source(
      merged, geometry(), merged_rho, merged_s1, merged_s2);
  for (std::size_t cell = 0U; cell < split_rho.size(); ++cell) {
    require(std::fabs(split_rho[cell] - merged_rho[cell]) < 2e-15,
            "exact cell merge changed deposited density");
    require(std::fabs(split_s1[cell] - merged_s1[cell]) < 2e-15,
            "exact cell merge changed root-grouped S1");
    require(std::fabs(split_s2[cell] - merged_s2[cell]) < 2e-15,
            "exact cell merge changed root-grouped S2");
  }
}

void test_randomized_projection_property() {
  // A fixed integer generator makes this a repeatable property test without
  // consuming any FeeBL physics RNG state.
  std::uint64_t state = UINT64_C(0x6a09e667f3bcc909);
  const auto uniform = [&state]() {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return static_cast<double>(state >> 11U) /
           static_cast<double>(UINT64_C(1) << 53U);
  };
  std::vector<RootProjection> roots;
  for (std::uint64_t index = 0U; index < 257U; ++index) {
    const double x = -1.35 + 2.7 * uniform();
    const double y = -0.65 + 1.3 * uniform();
    const double weight = 0.2 + 4.8 * uniform();
    roots.push_back(root(1000U + index, x, y, weight));
  }
  const SourceEnsemble split =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), roots, 1U, 100U);
  const SourceEnsemble merged =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::CoupledSplitMerge), geometry(), roots, 1U, 100U);
  std::vector<double> split_rho, split_s1, split_s2;
  std::vector<double> merged_rho, merged_s1, merged_s2;
  feebl::adaptive_particles::deposit_source(
      split, geometry(), split_rho, split_s1, split_s2);
  feebl::adaptive_particles::deposit_source(
      merged, geometry(), merged_rho, merged_s1, merged_s2);

  std::vector<long double> oracle_rho(split_rho.size(), 0.0L);
  std::vector<long double> oracle_s2(split_rho.size(), 0.0L);
  long double input_weight = 0.0L;
  for (const RootProjection& projected : roots) {
    const double wx = projected.cic.weight_x;
    const double wy = projected.cic.weight_y;
    const double fractions[4] = {
        (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy),
        (1.0 - wx) * wy, wx * wy};
    const std::size_t x0 = static_cast<std::size_t>(projected.cic.base_x);
    const std::size_t y0 = static_cast<std::size_t>(projected.cic.base_y);
    const std::size_t cells[4] = {
        x0 * geometry().n_y + y0,
        (x0 + 1U) * geometry().n_y + y0,
        x0 * geometry().n_y + y0 + 1U,
        (x0 + 1U) * geometry().n_y + y0 + 1U};
    for (std::size_t corner = 0U; corner < 4U; ++corner) {
      const long double contribution =
          projected.physical_weight * fractions[corner];
      oracle_rho[cells[corner]] += contribution;
      oracle_s2[cells[corner]] += contribution * contribution;
    }
    input_weight += projected.physical_weight;
  }
  long double split_total = 0.0L;
  long double merged_total = 0.0L;
  for (std::size_t cell = 0U; cell < split_rho.size(); ++cell) {
    const double scale = std::max(
        1.0, static_cast<double>(std::fabs(oracle_rho[cell])));
    require(std::fabs(split_rho[cell] - oracle_rho[cell]) < 2e-13 * scale,
            "random split density differs from direct CIC oracle");
    require(std::fabs(merged_rho[cell] - oracle_rho[cell]) < 2e-13 * scale,
            "random merged density differs from direct CIC oracle");
    require(std::fabs(split_rho[cell] - split_s1[cell]) < 2e-13 * scale,
            "random split root-grouped S1 is incorrect");
    require(std::fabs(merged_rho[cell] - merged_s1[cell]) < 2e-13 * scale,
            "random merge root-grouped S1 is incorrect");
    require(std::fabs(split_s2[cell] - oracle_s2[cell]) < 2e-13 * scale,
            "random split root-grouped S2 is incorrect");
    require(std::fabs(merged_s2[cell] - oracle_s2[cell]) < 2e-13 * scale,
            "random merge root-grouped S2 is incorrect");
    split_total += split_rho[cell];
    merged_total += merged_rho[cell];
  }
  require(std::fabs(split_total - input_weight) < 2e-12 * input_weight,
          "random split property lost total weight");
  require(std::fabs(merged_total - input_weight) < 2e-12 * input_weight,
          "random merge property lost total weight");
  require(std::fabs(split.metrics.clone_support_proxy -
                    merged.metrics.clone_support_proxy) < 2e-12,
          "random merge changed conservative clone support");
  const long double constant_kernel_before = input_weight * input_weight;
  const long double constant_kernel_after = split_total * merged_total;
  require(std::fabs(constant_kernel_before - constant_kernel_after) <
              4e-12 * constant_kernel_before,
          "constant cross-beam product-weight normalization changed");
}

void test_empty_boundary_and_cohort_cases() {
  const std::vector<RootProjection> empty;
  const SourceEnsemble empty_source =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), empty, 1U, 33U);
  require(empty_source.receipt.committed &&
              empty_source.representatives.empty(),
          "empty active source was not a valid zero transaction");
  std::vector<double> rho, s1, s2;
  feebl::adaptive_particles::deposit_source(
      empty_source, geometry(), rho, s1, s2);
  require(rho.size() == geometry().cells(),
          "empty source deposit has the wrong shape");
  for (double value : rho)
    require(value == 0.0, "empty source deposit is nonzero");

  RootProjection rejected = root(2000U, 0.0, 0.0, 1.0);
  rejected.cic.accepted = false;
  const SourceEnsemble rejected_source =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), {rejected}, 1U, 34U);
  require(rejected_source.receipt.roots_rejected == 1U &&
              rejected_source.receipt.roots_accepted == 0U,
          "rejected root accounting is wrong");

  Configuration cohorts = config(Mode::CoupledSplitMerge);
  cohorts.statistical_cohorts = 2U;
  const std::vector<RootProjection> separate = {
      root(2001U, 0.125, -0.125, 1.0, 21U),
      root(2002U, 0.125, -0.125, 1.0, 22U)};
  const SourceEnsemble separated =
      feebl::adaptive_particles::build_assignment_preserving_source(
          cohorts, geometry(), separate, 1U, 35U);
  require(separated.metrics.statistical_cohort_count == 2U,
          "cohort count is wrong");
  require(separated.metrics.many_root_representatives == 0U,
          "merge crossed an independent cohort boundary");

  // Exercise an exact grid-node/edge fractional coordinate.  Zero-weight CIC
  // corners must be omitted without changing normalization or IDs.
  const RootProjection boundary = root(2003U, 0.0, 0.0, 2.0);
  const SourceEnsemble boundary_source =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::SplitOnly), geometry(), {boundary}, 1U, 36U);
  feebl::adaptive_particles::deposit_source(
      boundary_source, geometry(), rho, s1, s2);
  long double total = 0.0L;
  for (double value : rho) total += value;
  require(std::fabs(total - 2.0L) < 1e-14L,
          "boundary projection lost weight");
}

void test_resource_and_invalid_state_guards() {
  const std::vector<RootProjection> roots = {
      root(3000U, -0.23, 0.17, 1.0),
      root(3001U, 0.27, -0.13, 3.0)};
  Configuration root_ratio = config(Mode::SplitOnly);
  root_ratio.maximum_weight_ratio = 2.0;
  require_throws<std::runtime_error>([&]() {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        root_ratio, geometry(), roots, 1U, 40U);
  }, "input-root weight-ratio guard did not fail closed");

  Configuration bytes = config(Mode::SplitOnly);
  bytes.maximum_bytes = 1U;
  require_throws<std::runtime_error>([&]() {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        bytes, geometry(), roots, 1U, 40U);
  }, "transactional byte guard did not fail closed");

  Configuration generation = config(Mode::CoupledSplitMerge);
  generation.maximum_generation = 1U;
  require_throws<std::runtime_error>([&]() {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        generation, geometry(), roots, 1U, 40U);
  }, "generation guard did not fail closed");

  RootProjection negative = roots.front();
  negative.physical_weight = -1.0;
  require_throws<std::invalid_argument>([&]() {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        config(Mode::SplitOnly), geometry(), {negative}, 1U, 40U);
  }, "negative root weight was accepted");

  Configuration invalid = config(Mode::SplitOnly);
  invalid.coarsen_threshold = 2.0;
  require_throws<std::invalid_argument>([&]() {
    feebl::adaptive_particles::validate_configuration(invalid);
  }, "inverted controller hysteresis was accepted");

  SourceRepresentative malformed;
  malformed.physical_weight = 2.0;
  malformed.ancestry_variance_mass = 2.0;
  malformed.ancestry = {
      {make_generated_primary_particle_id(1U, 2U), 1.0},
      {make_generated_primary_particle_id(1U, 1U), 1.0}};
  require_throws<std::runtime_error>([&]() {
    (void)feebl::adaptive_particles::compute_metrics({malformed});
  }, "unordered ancestry ledger was accepted");
}

void test_diagnostics_and_fail_closed() {
  const std::vector<RootProjection> roots = {
      root(4U, 0.11, -0.07, 1.0)};
  const SourceEnsemble diagnostic =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::Diagnostics), geometry(), roots, 1U, 1U, 5U);
  require(diagnostic.receipt.diagnostics_only,
          "mode 1 did not report diagnostics-only");
  require(!diagnostic.receipt.committed &&
              !diagnostic.representatives.empty() &&
              diagnostic.population_revision == 5U,
          "mode 1 proposal/revision semantics are wrong");
  require(diagnostic.metrics.distinct_root_count == 1U,
          "mode 1 discarded proposal diagnostics");
  require_throws<std::invalid_argument>([&]() {
    std::vector<double> rho, s1, s2;
    feebl::adaptive_particles::deposit_source(
        diagnostic, geometry(), rho, s1, s2);
  }, "mode 1 dry-run proposal was deposited as an active source");

  Configuration unsupported = config(Mode::SplitOnly);
  unsupported.domain = "full_physics";
  bool rejected = false;
  try {
    feebl::adaptive_particles::validate_configuration(unsupported);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unqualified full-physics domain was accepted");

  Configuration capped = config(Mode::SplitOnly);
  capped.maximum_count = 1U;
  rejected = false;
  try {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        capped, geometry(), roots, 1U, 1U);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "source count limit did not fail closed");

  Configuration one_cohort = config(Mode::CoupledSplitMerge);
  const std::vector<RootProjection> mixed_cohorts = {
      root(40U, 0.11, -0.07, 1.0, 1U),
      root(41U, 0.11, -0.07, 1.0, 2U)};
  require_throws<std::runtime_error>([&]() {
    (void)feebl::adaptive_particles::build_assignment_preserving_source(
        one_cohort, geometry(), mixed_cohorts, 1U, 1U);
  }, "undeclared statistical cohort was accepted");
}

void test_identity_repeatability() {
  const std::vector<RootProjection> roots = {
      root(5U, -0.41, 0.19, 1.0), root(6U, 0.22, -0.21, 1.0)};
  const SourceEnsemble first =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::CoupledSplitMerge), geometry(), roots, 1U, 17U);
  const SourceEnsemble second =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::CoupledSplitMerge), geometry(), roots, 1U, 17U);
  require(first.receipt.content_hash == second.receipt.content_hash,
          "strict source build is not repeatable");
  require(first.receipt.content_hash != 0U,
          "source content hash is not bound");
  const SourceEnsemble new_epoch =
      feebl::adaptive_particles::build_assignment_preserving_source(
          config(Mode::CoupledSplitMerge), geometry(), roots, 1U, 18U);
  require(first.receipt.content_hash != new_epoch.receipt.content_hash,
          "source content hash did not bind the epoch");
}

}  // namespace

int main() {
  try {
    test_split_and_clone_support();
    test_exact_cell_merge();
    test_randomized_projection_property();
    test_empty_boundary_and_cohort_cases();
    test_resource_and_invalid_state_guards();
    test_diagnostics_and_fail_closed();
    test_identity_repeatability();
  } catch (const std::exception& error) {
    std::cerr << "adaptive macroparticle core test failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "adaptive macroparticle core tests passed\n";
  return EXIT_SUCCESS;
}
