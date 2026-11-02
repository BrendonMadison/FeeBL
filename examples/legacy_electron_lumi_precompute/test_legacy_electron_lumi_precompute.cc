#include "legacyElectronLumiPrecomputeCPP.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace lee = feebl::modern::legacy_ee;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-78s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double bit width changed");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float bit width changed");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_double(double left, double right) {
  return double_bits(left) == double_bits(right);
}

bool same_float(float left, float right) {
  return float_bits(left) == float_bits(right);
}

bool same_moment(const lee::Moment1Raw& left,
                 const lee::Moment1Raw& right) {
  return same_double(left.weight, right.weight) &&
         same_double(left.weight2, right.weight2) &&
         same_double(left.weighted_x, right.weighted_x) &&
         same_double(left.weighted_x2, right.weighted_x2);
}

bool same_helicity(const lee::PairHelicityRaw& left,
                   const lee::PairHelicityRaw& right) {
  return same_moment(left.h1, right.h1) &&
         same_moment(left.h2, right.h2) &&
         same_double(left.weight, right.weight) &&
         same_double(left.weighted_product, right.weighted_product);
}

bool same_spectrum(const lee::SpectrumRaw& left,
                   const lee::SpectrumRaw& right) {
  if (!same_double(left.low, right.low) ||
      !same_double(left.high, right.high) ||
      !same_double(left.width, right.width) ||
      !same_moment(left.w_moment, right.w_moment) ||
      !same_double(left.underflow, right.underflow) ||
      !same_double(left.overflow, right.overflow) ||
      left.bin_weight.size() != right.bin_weight.size() ||
      left.bin_helicity.size() != right.bin_helicity.size())
    return false;
  for (std::size_t index = 0; index < left.bin_weight.size(); ++index) {
    if (!same_double(left.bin_weight[index], right.bin_weight[index]) ||
        !same_helicity(left.bin_helicity[index],
                       right.bin_helicity[index]))
      return false;
  }
  return true;
}

bool same_results(const lee::AggregateResults& left,
                  const lee::AggregateResults& right) {
  for (std::size_t index = 0; index < left.class_luminosity.size(); ++index)
    if (!same_double(left.class_luminosity[index],
                     right.class_luminosity[index]))
      return false;
  return same_double(left.lumi_ee, right.lumi_ee) &&
         same_double(left.step_lumi_ee, right.step_lumi_ee) &&
         same_double(left.lumi_ee_high, right.lumi_ee_high) &&
         same_double(left.step_lumi_ee_high, right.step_lumi_ee_high) &&
         same_double(left.lumi_pp, right.lumi_pp) &&
         same_double(left.lumi_weight_sum, right.lumi_weight_sum) &&
         same_double(left.lumi_weighted_helicity1,
                     right.lumi_weighted_helicity1) &&
         same_double(left.lumi_weighted_helicity2,
                     right.lumi_weighted_helicity2) &&
         same_double(left.lumi_ecm, right.lumi_ecm) &&
         same_double(left.lumi_ecm2, right.lumi_ecm2) &&
         left.invalid_velocity_observed == right.invalid_velocity_observed &&
         left.nonfinite_input_observed == right.nonfinite_input_observed &&
         same_spectrum(left.spectrum, right.spectrum);
}

bool runs_match(const lee::KernelRun& reference,
                const lee::KernelRun& candidate) {
  return reference.status.code == candidate.status.code &&
         reference.status.message == candidate.status.message &&
         (!reference.status.ok() || same_results(reference.results,
                                                  candidate.results));
}

lee::ParticleSample particle(float energy, float vx, float vy, float weight,
                             bool spin = false, double sx = 0.0,
                             double sy = 0.0, double sz = 0.0) {
  lee::ParticleSample result;
  result.signed_energy = energy;
  result.vx = vx;
  result.vy = vy;
  result.weight = weight;
  result.has_spin = spin;
  result.spin_x = sx;
  result.spin_y = sy;
  result.spin_z = sz;
  return result;
}

lee::KernelConfig base_config(bool spin = false) {
  lee::KernelConfig config;
  config.eligibility.bmt_spin = spin;
  config.eligibility.spin_records_complete = true;
  config.ecm_min = 1.0F;
  config.spectrum.low = 0.5;
  config.spectrum.high = 500.5;
  config.spectrum.bins = 64;
  return config;
}

void test_eligibility() {
  std::printf("eligibility gates\n");
  struct Gate {
    lee::StatusCode expected;
    void (*enable)(lee::EligibilityOptions&);
  };
  const std::array<Gate, 6> gates = {{
      {lee::StatusCode::IsrUnsupported,
       [](lee::EligibilityOptions& value) { value.do_isr = true; }},
      {lee::StatusCode::EeLumiHeapUnsupported,
       [](lee::EligibilityOptions& value) { value.do_ee_lumi_heap = true; }},
      {lee::StatusCode::CrossEventsUnsupported,
       [](lee::EligibilityOptions& value) { value.do_cross_events = true; }},
      {lee::StatusCode::BhabhasUnsupported,
       [](lee::EligibilityOptions& value) { value.do_bhabhas = true; }},
      {lee::StatusCode::JetsUnsupported,
       [](lee::EligibilityOptions& value) { value.do_jets = true; }},
      {lee::StatusCode::ProductionHookUnsupported,
       [](lee::EligibilityOptions& value) { value.do_prod_hook = true; }},
  }};
  for (const Gate& gate : gates) {
    lee::EligibilityOptions options;
    gate.enable(options);
    check(lee::check_eligibility(options).code == gate.expected,
          "unsupported side-effect mode fails closed");
  }
  lee::EligibilityOptions missing;
  missing.bmt_spin = true;
  missing.spin_records_complete = false;
  check(lee::check_eligibility(missing).code == lee::StatusCode::MissingSpin,
        "BMT mode rejects an incomplete spin view");
  check(lee::check_eligibility(lee::EligibilityOptions{}).ok(),
        "pure aggregate mode is eligible");
}

void test_precomputed_fields() {
  std::printf("precomputed particle fields\n");
  lee::KernelConfig config = base_config(true);
  const std::vector<lee::ParticleSample> input = {
      particle(-12.5F, 0.125F, -0.25F, 1.75F, true,
               0.2, -0.3, 0.4)};
  const lee::PrecomputeRun first = lee::precompute_beam(input, false, config);
  const lee::PrecomputeRun second = lee::precompute_beam(input, true, config);
  check(first.status.ok() && second.status.ok() && first.beam.size() == 1,
        "precomputation succeeds transactionally");
  const lee::PrecomputedParticle& cached = first.beam.front();
  check(same_float(cached.signed_energy, -12.5F) &&
            same_float(cached.absolute_energy, 12.5F) &&
            same_float(cached.vx, 0.125F) &&
            same_float(cached.vy, -0.25F) &&
            same_float(cached.weight, 1.75F),
        "signed/absolute energy, velocity, and weight are materialized");
  check(cached.massive_momentum == 0.0 && cached.charge_class == 1U &&
            second.beam.front().charge_class == 0U,
        "negative energy preserves massive-momentum and charge-class rules");
  check(cached.has_spin && cached.helicity_valid &&
            same_float(cached.legacy_spin_z, 0.4F),
        "optional legacy spin-z and helicity are materialized once");
}

void compare_case(const std::vector<lee::ParticleSample>& beam1,
                  const std::vector<lee::ParticleSample>& beam2,
                  const lee::KernelConfig& config,
                  const char* description) {
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate), description);
}

void test_basic_and_charge() {
  std::printf("aggregate, charge, zero, and negative-energy equivalence\n");
  const std::vector<lee::ParticleSample> beam1 = {
      particle(125.0F, 0.001F, -0.002F, 1.25F),
      particle(-90.0F, -0.003F, 0.004F, 0.75F),
      particle(0.0F, 0.0F, 0.0F, 2.0F)};
  const std::vector<lee::ParticleSample> beam2 = {
      particle(110.0F, -0.002F, 0.003F, 0.5F),
      particle(-80.0F, 0.004F, -0.001F, 1.5F),
      particle(-0.0F, 0.0F, 0.0F, 0.25F)};
  const lee::KernelConfig config = base_config(false);
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate),
        "all aggregates and spectrum state are bit-identical");
  check(reference.operations.qualifying_ee_pairs == 2U,
        "only positive-positive and negative-negative pairs enter ee totals");
  check(reference.results.class_luminosity[1] != 0.0 &&
            reference.results.class_luminosity[3] != 0.0 &&
            reference.results.class_luminosity[0] != 0.0 &&
            reference.results.class_luminosity[4] != 0.0,
        "legacy charge-sign matrix receives all expected classes");
}

void test_spin() {
  std::printf("BMT spin and helicity equivalence\n");
  const std::vector<lee::ParticleSample> beam1 = {
      particle(125.0F, 0.01F, -0.02F, 1.0F, true, 0.2, 0.3, 0.9),
      particle(80.0F, -0.03F, 0.01F, 0.7F, true, -0.4, 0.1, 0.5)};
  const std::vector<lee::ParticleSample> beam2 = {
      particle(120.0F, -0.02F, 0.01F, 1.1F, true, 0.1, -0.2, -0.8),
      particle(70.0F, 0.01F, 0.04F, 0.6F, true, 0.3, 0.2, -0.6)};
  const lee::KernelConfig config = base_config(true);
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate),
        "lumi_pp, luminosity helicity, and per-bin moments are bit-identical");
  check(reference.results.lumi_weight_sum > 0.0 &&
            reference.results.spectrum.bin_helicity.size() == 64U,
        "spin-bearing pairs populate weighted helicity moments");

  std::vector<lee::ParticleSample> missing = beam1;
  missing.front().has_spin = false;
  const lee::KernelRun rejected =
      lee::run_precomputed_candidate(missing, beam2, config);
  check(rejected.status.code == lee::StatusCode::MissingSpin &&
            rejected.operations.cartesian_pairs == 0U,
        "missing per-particle spin fails before the Cartesian loop");
}

void test_spectrum_edges() {
  std::printf("spectrum bins, underflow, overflow, and raw moments\n");
  lee::KernelConfig config = base_config(true);
  config.spectrum.low = 1.0;
  config.spectrum.high = 4.0;
  config.spectrum.bins = 3;
  const std::vector<lee::ParticleSample> beam1 = {
      particle(0.2F, 0.0F, 0.0F, 1.0F, true, 0.0, 0.0, 1.0),
      particle(1.0F, 0.0F, 0.0F, 2.0F, true, 0.0, 0.0, 0.5),
      particle(3.0F, 0.0F, 0.0F, 0.5F, true, 0.0, 0.0, -0.5)};
  const std::vector<lee::ParticleSample> beam2 = beam1;
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate),
        "every bin, flow, W moment, and helicity moment is bit-identical");
  check(reference.results.spectrum.underflow > 0.0 &&
            reference.results.spectrum.overflow > 0.0 &&
            reference.results.spectrum.bin_weight[0] > 0.0,
        "fixture exercises underflow, overflow, and in-range bins");
}

void test_threshold_ulps() {
  std::printf("high-luminosity threshold ULP behavior\n");
  const std::vector<lee::ParticleSample> beam1 = {
      particle(125.0F, 0.001F, 0.002F, 1.0F)};
  const std::vector<lee::ParticleSample> beam2 = {
      particle(124.0F, -0.003F, 0.001F, 1.0F)};
  lee::KernelConfig probe = base_config(false);
  probe.ecm_min = 0.0F;
  const lee::KernelRun base =
      lee::run_cartesian_reference(beam1, beam2, probe);
  const float ecm = static_cast<float>(base.results.lumi_ecm);
  lee::KernelConfig below = probe;
  below.ecm_min = std::nextafter(ecm,
                                 -std::numeric_limits<float>::infinity());
  lee::KernelConfig exact = probe;
  exact.ecm_min = ecm;
  lee::KernelConfig above = probe;
  above.ecm_min = std::nextafter(ecm,
                                 std::numeric_limits<float>::infinity());
  compare_case(beam1, beam2, below,
               "candidate matches one ULP below the high threshold");
  compare_case(beam1, beam2, exact,
               "candidate matches the strict comparison at exact threshold");
  compare_case(beam1, beam2, above,
               "candidate matches one ULP above the high threshold");
  check(lee::run_cartesian_reference(beam1, beam2, below)
                .results.lumi_ee_high == 1.0 &&
            lee::run_cartesian_reference(beam1, beam2, exact)
                .results.lumi_ee_high == 0.0 &&
            lee::run_cartesian_reference(beam1, beam2, above)
                .results.lumi_ee_high == 0.0,
        "legacy strict-greater-than threshold is preserved at adjacent ULPs");
}

void test_invalid_and_nonfinite() {
  std::printf("invalid velocity, overflow, and nonfinite propagation\n");
  const std::vector<lee::ParticleSample> invalid_velocity = {
      particle(10.0F, 1.1F, 0.2F, 1.0F)};
  const std::vector<lee::ParticleSample> ordinary = {
      particle(12.0F, 0.0F, 0.0F, 1.0F)};
  const lee::KernelRun invalid_reference = lee::run_cartesian_reference(
      invalid_velocity, ordinary, base_config(false));
  const lee::KernelRun invalid_candidate = lee::run_precomputed_candidate(
      invalid_velocity, ordinary, base_config(false));
  check(runs_match(invalid_reference, invalid_candidate),
        "invalid transverse velocity preserves legacy NaN/W=0 behavior");
  check(invalid_reference.results.invalid_velocity_observed,
        "invalid transverse velocity is explicitly diagnosed");

  const float infinity = std::numeric_limits<float>::infinity();
  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const float maximum = std::numeric_limits<float>::max();
  const std::vector<lee::ParticleSample> nonfinite1 = {
      particle(infinity, 0.0F, 0.0F, maximum),
      particle(quiet_nan, 0.1F, 0.0F, 2.0F)};
  const std::vector<lee::ParticleSample> nonfinite2 = {
      particle(maximum, 0.0F, 0.0F, maximum)};
  const lee::KernelRun reference = lee::run_cartesian_reference(
      nonfinite1, nonfinite2, base_config(false));
  const lee::KernelRun candidate = lee::run_precomputed_candidate(
      nonfinite1, nonfinite2, base_config(false));
  check(runs_match(reference, candidate),
        "overflow and nonfinite results retain identical IEEE-754 bit patterns");
  check(reference.results.nonfinite_input_observed,
        "nonfinite input is explicitly reported without changing legacy math");
}

void test_randomized() {
  std::printf("randomized Cartesian oracle equivalence\n");
  std::mt19937 generator(0x5eed1234U);
  std::uniform_real_distribution<float> energy(0.01F, 250.0F);
  std::uniform_real_distribution<float> velocity(-0.35F, 0.35F);
  std::uniform_real_distribution<float> weight(0.01F, 3.0F);
  std::uniform_real_distribution<double> spin(-1.0, 1.0);
  bool all_match = true;
  for (int trial = 0; trial < 80 && all_match; ++trial) {
    const bool with_spin = trial % 2 != 0;
    const std::size_t n1 = static_cast<std::size_t>(trial % 9 + 1);
    const std::size_t n2 = static_cast<std::size_t>((trial * 5) % 11 + 1);
    std::vector<lee::ParticleSample> beam1;
    std::vector<lee::ParticleSample> beam2;
    for (std::size_t index = 0; index < n1; ++index) {
      float value = energy(generator);
      if ((trial + static_cast<int>(index)) % 3 == 0) value = -value;
      beam1.push_back(particle(value, velocity(generator), velocity(generator),
                               weight(generator), with_spin,
                               spin(generator), spin(generator), spin(generator)));
    }
    for (std::size_t index = 0; index < n2; ++index) {
      float value = energy(generator);
      if ((trial + static_cast<int>(index)) % 4 == 0) value = -value;
      beam2.push_back(particle(value, velocity(generator), velocity(generator),
                               weight(generator), with_spin,
                               spin(generator), spin(generator), spin(generator)));
    }
    lee::KernelConfig config = base_config(with_spin);
    config.ecm_min = energy(generator);
    const lee::KernelRun reference =
        lee::run_cartesian_reference(beam1, beam2, config);
    const lee::KernelRun candidate =
        lee::run_precomputed_candidate(beam1, beam2, config);
    all_match = runs_match(reference, candidate);
  }
  check(all_match,
        "80 randomized spin/no-spin, sign, weight, and threshold cases match bits");
}

void test_empty_high_occupancy_and_counts() {
  std::printf("empty/high-occupancy and operation counts\n");
  const lee::KernelConfig config = base_config(false);
  compare_case({}, {}, config,
               "empty beams produce identical configured empty spectra");
  const std::vector<lee::ParticleSample> one_sided = {
      particle(125.0F, 0.001F, -0.002F, 1.0F)};
  const lee::KernelRun one_sided_reference =
      lee::run_cartesian_reference(one_sided, {}, config);
  const lee::KernelRun one_sided_candidate =
      lee::run_precomputed_candidate(one_sided, {}, config);
  check(runs_match(one_sided_reference, one_sided_candidate) &&
            one_sided_reference.operations.cartesian_pairs == 0U &&
            one_sided_candidate.operations.precomputed_particles == 1U,
        "one-sided empty input has no pairs and bounded one-time preprocessing");

  constexpr std::size_t n1 = 96;
  constexpr std::size_t n2 = 80;
  std::vector<lee::ParticleSample> beam1;
  std::vector<lee::ParticleSample> beam2;
  beam1.reserve(n1);
  beam2.reserve(n2);
  for (std::size_t index = 0; index < n1; ++index)
    beam1.push_back(particle(125.0F + static_cast<float>(index) * 0.01F,
                             0.001F, -0.002F,
                             0.5F + static_cast<float>(index % 7) * 0.1F));
  for (std::size_t index = 0; index < n2; ++index)
    beam2.push_back(particle(120.0F + static_cast<float>(index) * 0.02F,
                             -0.002F, 0.001F,
                             0.4F + static_cast<float>(index % 5) * 0.2F));
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate),
        "7,680-pair high-occupancy cell preserves every output bit");
  const std::uint64_t pairs = n1 * n2;
  const std::uint64_t particles = n1 + n2;
  check(reference.operations.pair_sqrt_calls == 6U * pairs,
        "legacy oracle records six square roots per ordinary qualifying pair");
  check(candidate.operations.pair_sqrt_calls == 2U * pairs &&
            candidate.operations.precompute_sqrt_calls == 2U * particles,
        "candidate retains two pair roots and moves four particle roots to cache");
  check(reference.operations.particle_energy_reads == 4U * pairs &&
            reference.operations.particle_velocity_reads == 2U * pairs &&
            reference.operations.particle_weight_reads == 2U * pairs,
        "legacy oracle records repeated per-pair particle accessors");
  check(candidate.operations.particle_energy_reads == particles &&
            candidate.operations.particle_velocity_reads == particles &&
            candidate.operations.particle_weight_reads == particles,
        "candidate reads each cached particle field group once");
}

void test_seeded_cell_order() {
  std::printf("seeded multi-cell floating-point update order\n");
  const lee::KernelConfig config = base_config(false);
  const std::vector<lee::ParticleSample> first_beam1 = {
      particle(20.0F, 0.0F, 0.0F, 1.0e10F)};
  const std::vector<lee::ParticleSample> first_beam2 = {
      particle(20.0F, 0.0F, 0.0F, 1.0e10F)};
  const std::vector<lee::ParticleSample> second_beam1 = {
      particle(20.0F, 0.0F, 0.0F, 1.0F)};
  const std::vector<lee::ParticleSample> second_beam2 = {
      particle(20.0F, 0.0F, 0.0F, -1.0e20F),
      particle(20.0F, 0.0F, 0.0F, 1.0F)};

  const lee::KernelRun reference_first = lee::run_cartesian_reference(
      first_beam1, first_beam2, config);
  const lee::KernelRun candidate_first = lee::run_precomputed_candidate(
      first_beam1, first_beam2, config);
  const lee::KernelRun reference_final = lee::run_cartesian_reference_seeded(
      second_beam1, second_beam2, config, reference_first.results);
  const lee::KernelRun candidate_final = lee::run_precomputed_candidate_seeded(
      second_beam1, second_beam2, config, candidate_first.results);
  check(runs_match(reference_first, candidate_first) &&
            runs_match(reference_final, candidate_final),
        "candidate preserves every accumulator bit across ordered cells");

  const lee::KernelRun zero_seeded_second = lee::run_cartesian_reference(
      second_beam1, second_beam2, config);
  const double grouped_lumi =
      reference_first.results.lumi_ee + zero_seeded_second.results.lumi_ee;
  check(!same_double(grouped_lumi, reference_final.results.lumi_ee),
        "fixture proves merging zero-seeded cell totals changes rounding");

  lee::AggregateResults mismatched = reference_first.results;
  mismatched.spectrum.width = std::nextafter(
      mismatched.spectrum.width, std::numeric_limits<double>::infinity());
  const lee::KernelRun rejected = lee::run_precomputed_candidate_seeded(
      second_beam1, second_beam2, config, mismatched);
  check(rejected.status.code ==
            lee::StatusCode::AccumulatorConfigurationMismatch &&
            rejected.operations.cartesian_pairs == 0U,
        "mismatched seeded spectrum fails before any pair update");
}

void test_adversarial_configuration() {
  std::printf("fail-closed spectrum and exact-physics configuration\n");
  lee::KernelConfig config = base_config(false);
  config.electron_mass = std::nextafter(
      lee::kLegacyElectronMassGeV,
      std::numeric_limits<double>::infinity());
  check(lee::run_precomputed_candidate({}, {}, config).status.code ==
            lee::StatusCode::InvalidElectronMass,
        "one-ULP nonlegacy electron mass is rejected");
  config.electron_mass = std::nextafter(
      lee::kLegacyElectronMassGeV, 0.0);
  check(lee::run_cartesian_reference({}, {}, config).status.code ==
            lee::StatusCode::InvalidElectronMass,
        "opposite-adjacent nonlegacy electron mass is also rejected");

  config = base_config(false);
  config.spectrum.low = -std::numeric_limits<double>::infinity();
  check(lee::run_precomputed_candidate({}, {}, config).status.code ==
            lee::StatusCode::InvalidSpectrumConfig,
        "infinite enabled spectrum bound is rejected");
  config = base_config(false);
  config.spectrum.low = std::numeric_limits<double>::quiet_NaN();
  check(lee::run_precomputed_candidate({}, {}, config).status.code ==
            lee::StatusCode::InvalidSpectrumConfig,
        "NaN enabled spectrum bound is rejected");
  config = base_config(false);
  config.spectrum.low = -std::numeric_limits<double>::max();
  config.spectrum.high = std::numeric_limits<double>::max();
  check(lee::run_precomputed_candidate({}, {}, config).status.code ==
            lee::StatusCode::InvalidSpectrumConfig,
        "overflowed spectrum span is rejected");

  config = base_config(false);
  config.limits.max_spectrum_bins =
      std::numeric_limits<std::uint64_t>::max();
  config.spectrum.bins = std::numeric_limits<std::size_t>::max();
  const lee::KernelRun excessive =
      lee::run_precomputed_candidate({}, {}, config);
  check(!excessive.status.ok() &&
            excessive.operations.cartesian_pairs == 0U,
        "SIZE_MAX bins fail closed even when caller widens the configured cap");

  lee::SpectrumRaw transactional;
  const lee::SpectrumConfig small_spectrum{0.0, 10.0, 2U};
  check(transactional.configure(small_spectrum).ok(),
        "small standalone spectrum configuration succeeds");
  transactional.bin_weight[0] = 7.0;
  lee::SpectrumConfig over_budget = small_spectrum;
  over_budget.bins = static_cast<std::size_t>(
      lee::kAbsoluteSpectrumStorageBytes /
      (sizeof(double) + sizeof(lee::PairHelicityRaw)) + 1U);
  const lee::Status allocation_gate = transactional.configure(over_budget);
  check(allocation_gate.code == lee::StatusCode::SpectrumBinLimitExceeded &&
            transactional.bin_weight.size() == 2U &&
            same_double(transactional.bin_weight[0], 7.0),
        "absolute storage gate rejects transactionally before allocation");

  config = base_config(false);
  config.spectrum.bins = 0U;
  config.spectrum.low = std::numeric_limits<double>::quiet_NaN();
  check(lee::run_precomputed_candidate({}, {}, config).status.ok(),
        "zero bins explicitly disable spectrum allocation before bound use");
}

void test_bmt_invalid_velocity_counts() {
  std::printf("BMT invalid-velocity source-count coverage\n");
  const lee::KernelConfig config = base_config(true);
  const std::vector<lee::ParticleSample> beam1 = {
      particle(40.0F, 1.1F, 0.0F, 1.0F, true, 0.1, 0.2, 0.3)};
  const std::vector<lee::ParticleSample> beam2 = {
      particle(40.0F, 0.0F, 0.0F, 1.0F, true, -0.1, 0.2, -0.3)};
  const lee::KernelRun reference =
      lee::run_cartesian_reference(beam1, beam2, config);
  const lee::KernelRun candidate =
      lee::run_precomputed_candidate(beam1, beam2, config);
  check(runs_match(reference, candidate) &&
            reference.results.invalid_velocity_observed &&
            reference.operations.qualifying_ee_pairs == 1U,
        "invalid-q BMT pair matches without fabricating helicity");
}

void test_limits() {
  std::printf("bounded allocation and overflow gates\n");
  lee::KernelLimits limits;
  limits.max_particles_per_beam = std::numeric_limits<std::uint64_t>::max();
  limits.max_cartesian_pairs = std::numeric_limits<std::uint64_t>::max();
  limits.max_spectrum_bins = 4;
  check(lee::validate_problem_size(
            std::numeric_limits<std::size_t>::max(), 2U, 0U, limits).code ==
            lee::StatusCode::PairCountOverflow,
        "unrepresentable Cartesian pair product is rejected before allocation");
  check(lee::validate_problem_size(2U, 2U, 5U, limits).code ==
            lee::StatusCode::SpectrumBinLimitExceeded,
        "spectrum allocation is bounded");
  limits.max_cartesian_pairs = 3;
  check(lee::validate_problem_size(2U, 2U, 0U, limits).code ==
            lee::StatusCode::PairLimitExceeded,
        "representable but excessive Cartesian product is rejected");

  lee::KernelConfig invalid = base_config(false);
  invalid.electron_mass = std::numeric_limits<double>::infinity();
  check(lee::run_precomputed_candidate({}, {}, invalid).status.code ==
            lee::StatusCode::InvalidElectronMass,
        "nonfinite electron mass fails closed");
}

}  // namespace

int main() {
  test_eligibility();
  test_precomputed_fields();
  test_basic_and_charge();
  test_spin();
  test_spectrum_edges();
  test_threshold_ulps();
  test_invalid_and_nonfinite();
  test_randomized();
  test_empty_high_occupancy_and_counts();
  test_seeded_cell_order();
  test_adversarial_configuration();
  test_bmt_invalid_velocity_counts();
  test_limits();
  std::printf("\nlegacy electron luminosity precompute: %s (%d failures)\n",
              failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
