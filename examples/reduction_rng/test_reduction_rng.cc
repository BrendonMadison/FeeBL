#include "pairMomentsCPP.h"
#include "resultsCPP.h"
#include "rndmCPP.h"
#include "nativeOutputCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool same_double(double lhs, double rhs, double tolerance = 1.0e-14) {
  const double scale = std::max(1.0, std::max(std::fabs(lhs), std::fabs(rhs)));
  return std::fabs(lhs - rhs) <= tolerance * scale;
}

std::uint32_t float_bits(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t),
                "test requires binary32-sized float");
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct SpectrumSnapshot {
  bool configured = false;
  std::size_t bins = 0;
  double lower = 0.0;
  double upper = 0.0;
  double total_weight = 0.0;
  double total_mean = 0.0;
  double total_variance = 0.0;
  double total_neff = 0.0;
  double underflow = 0.0;
  double overflow = 0.0;
  std::vector<double> bin_weight;
  std::vector<double> helicity_weight;
  std::vector<double> h1_mean;
  std::vector<double> h2_mean;
  std::vector<double> product_mean;
};

SpectrumSnapshot snapshot(const pairmoments::SpectrumW& spectrum) {
  SpectrumSnapshot value;
  value.configured = spectrum.configured();
  value.bins = spectrum.bins();
  value.lower = spectrum.lower_edge();
  value.upper = spectrum.upper_edge();
  value.total_weight = spectrum.w_moment().weight();
  value.total_mean = spectrum.w_moment().mean();
  value.total_variance = spectrum.w_moment().variance();
  value.total_neff = spectrum.w_moment().n_eff();
  value.underflow = spectrum.underflow();
  value.overflow = spectrum.overflow();
  for (std::size_t bin = 0; bin < spectrum.bins(); ++bin) {
    value.bin_weight.push_back(spectrum.bin_weight(bin));
    value.helicity_weight.push_back(spectrum.bin(bin).weight());
    value.h1_mean.push_back(spectrum.bin(bin).h1().mean());
    value.h2_mean.push_back(spectrum.bin(bin).h2().mean());
    value.product_mean.push_back(spectrum.bin(bin).mean_product());
  }
  return value;
}

bool same_snapshot(const SpectrumSnapshot& lhs, const SpectrumSnapshot& rhs) {
  if (lhs.configured != rhs.configured || lhs.bins != rhs.bins ||
      !same_double(lhs.lower, rhs.lower) ||
      !same_double(lhs.upper, rhs.upper) ||
      !same_double(lhs.total_weight, rhs.total_weight) ||
      !same_double(lhs.total_mean, rhs.total_mean) ||
      !same_double(lhs.total_variance, rhs.total_variance) ||
      !same_double(lhs.total_neff, rhs.total_neff) ||
      !same_double(lhs.underflow, rhs.underflow) ||
      !same_double(lhs.overflow, rhs.overflow) ||
      lhs.bin_weight.size() != rhs.bin_weight.size()) {
    return false;
  }
  for (std::size_t bin = 0; bin < lhs.bin_weight.size(); ++bin) {
    if (!same_double(lhs.bin_weight[bin], rhs.bin_weight[bin]) ||
        !same_double(lhs.helicity_weight[bin], rhs.helicity_weight[bin]) ||
        !same_double(lhs.h1_mean[bin], rhs.h1_mean[bin]) ||
        !same_double(lhs.h2_mean[bin], rhs.h2_mean[bin]) ||
        !same_double(lhs.product_mean[bin], rhs.product_mean[bin])) {
      return false;
    }
  }
  return true;
}

struct ResultsSnapshot {
  double lumi_fine = 0.0;
  double step = 0.0;
  double step_high = 0.0;
  double helicity_weight = 0.0;
  double h1 = 0.0;
  double h2 = 0.0;
  SpectrumSnapshot spectrum;
};

ResultsSnapshot snapshot(const RESULTS& results) {
  ResultsSnapshot value;
  value.lumi_fine = results.get_lumi_fine();
  value.step = results.get_step_lumi_ee();
  value.step_high = results.get_step_lumi_ee_high();
  value.helicity_weight = results.get_lumi_weight_sum();
  value.h1 = results.get_lumi_weighted_helicity1();
  value.h2 = results.get_lumi_weighted_helicity2();
  value.spectrum = snapshot(results.pair_spectrum());
  return value;
}

bool same_snapshot(const ResultsSnapshot& lhs, const ResultsSnapshot& rhs) {
  return same_double(lhs.lumi_fine, rhs.lumi_fine) &&
         same_double(lhs.step, rhs.step) &&
         same_double(lhs.step_high, rhs.step_high) &&
         same_double(lhs.helicity_weight, rhs.helicity_weight) &&
         same_double(lhs.h1, rhs.h1) && same_double(lhs.h2, rhs.h2) &&
         same_snapshot(lhs.spectrum, rhs.spectrum);
}

void add_left_sample(pairmoments::SpectrumW& spectrum) {
  spectrum.add(2.0, 1.0, 0.5, -0.25);
  spectrum.add(4.0, 5.0, -0.5, 0.75);
  spectrum.add(1.0, -1.0);
}

void add_right_sample(pairmoments::SpectrumW& spectrum) {
  spectrum.add(8.0, 9.0, 0.25, 0.5);
  spectrum.add(16.0, 3.0);
  spectrum.add(2.0, 11.0);
}

void test_moment_and_spectrum_reduction() {
  pairmoments::Moment1 left_moment;
  pairmoments::Moment1 right_moment;
  pairmoments::Moment1 oracle_moment;
  left_moment.add(1.0, -2.0);
  left_moment.add(2.0, 4.0);
  right_moment.add(4.0, 8.0);
  right_moment.add(8.0, -1.0);
  oracle_moment.add(1.0, -2.0);
  oracle_moment.add(2.0, 4.0);
  oracle_moment.add(4.0, 8.0);
  oracle_moment.add(8.0, -1.0);
  check(left_moment.merge(right_moment), "Moment1 split-lane merge succeeds");
  check(same_double(left_moment.weight(), oracle_moment.weight()) &&
            same_double(left_moment.mean(), oracle_moment.mean()) &&
            same_double(left_moment.variance(), oracle_moment.variance()) &&
            same_double(left_moment.n_eff(), oracle_moment.n_eff()) &&
            same_double(left_moment.error(), oracle_moment.error()),
        "Moment1 split-lane merge retains sufficient statistics");

  pairmoments::PairHelicity left_helicity;
  pairmoments::PairHelicity right_helicity;
  pairmoments::PairHelicity oracle_helicity;
  left_helicity.add(2.0, 0.5, -0.25);
  right_helicity.add(4.0, -0.5, 0.75);
  oracle_helicity.add(2.0, 0.5, -0.25);
  oracle_helicity.add(4.0, -0.5, 0.75);
  check(left_helicity.merge(right_helicity),
        "PairHelicity split-lane merge succeeds");
  check(same_double(left_helicity.weight(), oracle_helicity.weight()) &&
            same_double(left_helicity.h1().mean(),
                        oracle_helicity.h1().mean()) &&
            same_double(left_helicity.h2().mean(),
                        oracle_helicity.h2().mean()) &&
            same_double(left_helicity.mean_product(),
                        oracle_helicity.mean_product()) &&
            same_double(left_helicity.covariance(),
                        oracle_helicity.covariance()),
        "PairHelicity split-lane merge retains joint moments");

  pairmoments::SpectrumW left;
  pairmoments::SpectrumW right;
  pairmoments::SpectrumW oracle;
  left.configure(0.0, 10.0, 5);
  right.configure(0.0, 10.0, 5);
  oracle.configure(0.0, 10.0, 5);
  add_left_sample(left);
  add_right_sample(right);
  add_left_sample(oracle);
  add_right_sample(oracle);
  check(left.merge(right), "SpectrumW split-lane merge succeeds");
  check(same_snapshot(snapshot(left), snapshot(oracle)),
        "SpectrumW split-lane merge matches serial sufficient statistics");
}

void test_spectrum_reset_and_transactionality() {
  pairmoments::SpectrumW spectrum;
  spectrum.configure(0.0, 10.0, 5);
  add_left_sample(spectrum);
  add_right_sample(spectrum);
  spectrum.configure(20.0, 30.0, 2);
  check(spectrum.configured() && spectrum.bins() == 2U &&
            spectrum.lower_edge() == 20.0 && spectrum.upper_edge() == 30.0,
        "SpectrumW reconfigure installs the new geometry");
  check(spectrum.w_moment().weight() == 0.0 && spectrum.underflow() == 0.0 &&
            spectrum.overflow() == 0.0 && spectrum.bin_weight(0) == 0.0 &&
            spectrum.bin_weight(1) == 0.0 &&
            spectrum.bin(0).weight() == 0.0 &&
            spectrum.bin(1).weight() == 0.0,
        "SpectrumW reconfigure clears every accumulated statistic");

  pairmoments::SpectrumW destination;
  pairmoments::SpectrumW mismatch;
  destination.configure(0.0, 10.0, 5);
  mismatch.configure(0.0, 12.0, 6);
  add_left_sample(destination);
  add_right_sample(mismatch);
  const SpectrumSnapshot before_mismatch = snapshot(destination);
  check(!destination.merge(mismatch), "SpectrumW rejects mismatched binning");
  check(same_snapshot(snapshot(destination), before_mismatch),
        "SpectrumW mismatch rejection is transactional");

  pairmoments::SpectrumW nonfinite;
  nonfinite.configure(0.0, 10.0, 5);
  nonfinite.add(std::numeric_limits<double>::infinity(), 1.0);
  const SpectrumSnapshot before_nonfinite = snapshot(destination);
  check(!destination.merge(nonfinite), "SpectrumW rejects non-finite lanes");
  check(same_snapshot(snapshot(destination), before_nonfinite),
        "SpectrumW non-finite rejection is transactional");
}

void populate_result_lane(RESULTS& results, double scale) {
  results.add_lumi_fine(1.0 * scale);
  results.add_lumi_ee(2.0 * scale);
  results.add_lumi_ee_high(3.0 * scale);
  results.add_lumi_pp(4.0 * scale);
  results.add_lumi_eg(5.0 * scale);
  results.add_lumi_ge(6.0 * scale);
  results.add_lumi_gg(7.0 * scale);
  results.add_lumi_gg_high(8.0 * scale);
  results.add_lumi_ecm(9.0 * scale);
  results.add_lumi_ecm2(10.0 * scale);
  results.add_lumi_weighted_helicity(2.0 * scale, 0.5, -0.25);
  results.add_step_lumi_ee(11.0 * scale);
  results.add_step_lumi_ee_high(12.0 * scale);
  results.add_lumi(0, 0, 13.0 * scale);
  results.updateUpsmax(static_cast<float>(14.0 * scale));
  results.add_pair_W(2.0 * scale, 4.0, 0.5, -0.25);
}

void test_results_merge_reset_and_transactionality() {
  RESULTS destination;
  RESULTS lane;
  destination.configure_pair_spectrum(0.0, 10.0, 5);
  lane.configure_pair_spectrum(0.0, 10.0, 5);
  populate_result_lane(destination, 1.0);
  populate_result_lane(lane, 2.0);
  check(destination.merge_primary_photon_collision_from(lane),
        "RESULTS lane merge succeeds");
  check(destination.get_lumi_fine() == 3.0 &&
            destination.get_step_lumi_ee() == 33.0 &&
            destination.get_step_lumi_ee_high() == 36.0 &&
            destination.get_lumi_weight_sum() == 6.0 &&
            destination.get_lumi_weighted_helicity1() == 0.5 &&
            destination.get_lumi_weighted_helicity2() == -0.25 &&
            destination.pair_spectrum().w_moment().weight() == 6.0 &&
            destination.pair_spectrum().bin_weight(2) == 6.0,
        "RESULTS merge reduces scalar steps, moments, and spectrum");

  lane.reset_primary_photon_collision_lane();
  check(lane.get_lumi_fine() == 0.0 && lane.get_step_lumi_ee() == 0.0 &&
            lane.get_step_lumi_ee_high() == 0.0 &&
            lane.get_lumi_weight_sum() == 0.0 &&
            lane.pair_spectrum().configured() &&
            lane.pair_spectrum().bins() == 5U &&
            lane.pair_spectrum().w_moment().weight() == 0.0,
        "RESULTS lane reset clears state but preserves spectrum geometry");

  RESULTS mismatch;
  mismatch.configure_pair_spectrum(0.0, 12.0, 6);
  populate_result_lane(mismatch, 4.0);
  const ResultsSnapshot before_mismatch = snapshot(destination);
  check(!destination.merge_primary_photon_collision_from(mismatch),
        "RESULTS rejects a worker with mismatched spectrum geometry");
  check(same_snapshot(snapshot(destination), before_mismatch),
        "RESULTS spectrum-mismatch rejection leaves observables unchanged");

  RESULTS nonfinite;
  nonfinite.configure_pair_spectrum(0.0, 10.0, 5);
  populate_result_lane(nonfinite, 4.0);
  nonfinite.add_lumi_fine(std::numeric_limits<double>::infinity());
  const ResultsSnapshot before_nonfinite = snapshot(destination);
  check(!destination.merge_primary_photon_collision_from(nonfinite),
        "RESULTS rejects a non-finite worker lane");
  check(same_snapshot(snapshot(destination), before_nonfinite),
        "RESULTS non-finite rejection leaves observables unchanged");

  RESULTS mode_mismatch;
  mode_mismatch.configure_pair_spectrum(0.0, 10.0, 5);
  mode_mismatch.set_primary_only_mode3(true);
  const ResultsSnapshot before_mode_mismatch = snapshot(destination);
  check(!destination.merge_primary_photon_collision_from(mode_mismatch),
        "RESULTS rejects incompatible result-scope modes");
  check(same_snapshot(snapshot(destination), before_mode_mismatch),
        "RESULTS mode rejection leaves observables unchanged");
}

feebl::PhysicalRandomKeyV2 production_key(feebl::RandomDomain domain) {
  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = 0x0123456789abcdefULL;
  key.domain = domain;
  key.crossing = 3U;
  key.diagonal = 17U;
  key.wave = 9U;
  key.substep = 2U;
  key.encounter = 0xfedcba9876543210ULL;
  key.beam = 1U;
  key.parent_a = 0x1111222233334444ULL;
  key.parent_b = 0xaaaabbbbccccddddULL;
  key.interaction = 23U;
  key.emission = 5U;
  key.rejection_attempt = 7U;
  return key;
}

void test_counter_rng_adapter() {
  const feebl::PhysicalRandomKeyV2 key =
      production_key(feebl::RandomDomain::Beamstrahlung);
  RNDM adapter(991U);
  adapter.begin_counter_stream(key);
  check(adapter.counter_stream_active(), "RNDM reports active keyed mode");
  std::vector<std::uint32_t> first_pass;
  for (std::uint64_t draw = 0; draw < 64U; ++draw) {
    const float actual = adapter.rndm();
    const float expected = feebl::CounterRngV2::uniform_float(key, draw);
    first_pass.push_back(float_bits(actual));
    check(float_bits(actual) == float_bits(expected),
          "RNDM keyed adapter matches direct CounterRngV2 draw");
  }

  adapter.begin_counter_stream(key);
  for (std::size_t draw = 0; draw < first_pass.size(); ++draw) {
    check(float_bits(adapter.rndm()) == first_pass[draw],
          "RNDM keyed stream repeats after rebinding the same identity");
  }

  const feebl::PhysicalRandomKeyV2 other_key =
      production_key(feebl::RandomDomain::PhotonSampling);
  adapter.begin_counter_stream(other_key);
  bool any_domain_difference = false;
  for (std::size_t draw = 0; draw < first_pass.size(); ++draw) {
    if (float_bits(adapter.rndm()) != first_pass[draw])
      any_domain_difference = true;
  }
  check(any_domain_difference,
        "RNDM separates keyed streams belonging to different domains");

  feebl::PhysicalRandomKeyV2 invalid = key;
  invalid.schema_version = 1U;
  bool invalid_rejected = false;
  try {
    adapter.begin_counter_stream(invalid);
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  check(invalid_rejected, "RNDM rejects an invalid physical-key schema");

  adapter.begin_counter_stream(key);
  (void)adapter.gasdev();  // Leave a pending Gaussian variate in the cache.
  (void)adapter.rndm_save("reduction_rng_v2.save");
  const float expected_normal = adapter.gasdev();
  const float expected_uniform = adapter.rndm();
  (void)adapter.rndm_load("reduction_rng_v2.save");
  check(float_bits(adapter.gasdev()) == float_bits(expected_normal),
        "RNDM restores a CounterRngV2 Gaussian cache");
  check(float_bits(adapter.rndm()) == float_bits(expected_uniform),
        "RNDM restores a CounterRngV2 uniform cursor");
}

void test_legacy_rng_known_answer() {
  // These are the binary32 outputs of the historical 64-bit LCG path for
  // seed 0x123456789abcdef0.  They pin the pre-adapter stream independently
  // of CounterRngV2 and catch accidental changes to reseed/dispatch behavior.
  static const std::uint32_t expected[] = {
      0x3f0ddb19U, 0x3eb2a1d1U, 0x3ee47e81U, 0x3f017de4U,
      0x3c8f023eU, 0x3ef6639eU, 0x3ea3ff98U, 0x3edafa2eU};
  RNDM legacy(0x123456789abcdef0ULL);
  for (std::size_t index = 0; index < sizeof(expected) / sizeof(expected[0]);
       ++index) {
    const std::uint32_t actual = float_bits(legacy.rndm());
    if (actual != expected[index]) {
      std::fprintf(stderr,
                   "legacy known-answer draw %zu: got 0x%08x expected 0x%08x\n",
                   index, actual, expected[index]);
      ++failures;
    }
  }

  RNDM reseeded(7U);
  reseeded.begin_counter_stream(
      production_key(feebl::RandomDomain::Beamstrahlung));
  (void)reseeded.rndm();
  reseeded.reseed_encounter(0x123456789abcdef0ULL);
  check(!reseeded.counter_stream_active(),
        "legacy reseed exits CounterRngV2 adapter mode");
  for (std::size_t index = 0; index < sizeof(expected) / sizeof(expected[0]);
       ++index) {
    check(float_bits(reseeded.rndm()) == expected[index],
          "legacy encounter reseed restores the pinned LCG stream");
  }
}

void test_encounter_error_proxy_metadata() {
  pairmoments::Moment1 independent, repeated;
  independent.add(1.0, 0.5);
  independent.add(1.0, 0.9);
  for (int repeat = 0; repeat < 16; ++repeat) {
    repeated.add(1.0, 0.5);
    repeated.add(1.0, 0.9);
  }
  // Duplicating observations does not add independent histories. The reported
  // proxy nevertheless shrinks, which is why the file must declare its model.
  check(same_double(repeated.mean(), independent.mean()), "repeated-history mean");
  check(same_double(repeated.error(), independent.error() / 4.0),
        "encounter proxy exposes repeated-history dependence");
  const std::string header = native_header(NativeRunMeta(), "pair_spectrum", "h1_err h2_err");
  check(header.find("independent_encounter_SEM_proxy") != std::string::npos,
        "native spectrum declares error-proxy model");
  check(header.find("shared_macroparticle_correlations") != std::string::npos,
        "native spectrum declares omitted ancestry covariance");
  check(header.find("zero_is_not a_calibrated_zero_uncertainty") != std::string::npos,
        "native spectrum declares zero-error semantics");
}

}  // namespace

int main() {
  test_moment_and_spectrum_reduction();
  test_spectrum_reset_and_transactionality();
  test_results_merge_reset_and_transactionality();
  test_counter_rng_adapter();
  test_legacy_rng_known_answer();
  test_encounter_error_proxy_metadata();
  if (failures != 0) {
    std::fprintf(stderr, "%d reduction/RNG checks failed\n", failures);
    return 1;
  }
  std::printf("PASS: pair moments, RESULTS reduction/reset, and RNDM adapter\n");
  return 0;
}
