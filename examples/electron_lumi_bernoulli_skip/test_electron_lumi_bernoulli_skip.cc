#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "electronLumiBernoulliSkipCPP.h"

namespace sampling = feebl::electron_lumi_sampling;

namespace {

using Cursor = sampling::ElectronLumiBernoulliSkipCursorV1;
using Error = sampling::ElectronLumiBernoulliSkipErrorV1;
using Limits = sampling::ElectronLumiBernoulliSkipLimitsV1;
using Prefix = sampling::ElectronLumiBernoulliRowPrefixV1;
using Result = sampling::ElectronLumiBernoulliSkipResultV1;

void fail(const std::string& message) {
  std::cerr << "FAIL: " << message << std::endl;
  std::exit(1);
}

void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

Prefix prefix(std::uint64_t seed = 0x123456789abcdef0ull,
              std::uint64_t parent = 0x1100000000000042ull,
              std::uint64_t row = 7U) {
  Prefix value;
  value.run_seed = seed;
  value.crossing = 3U;
  value.diagonal = 5U;
  value.wave = 11U;
  value.substep = 2U;
  value.encounter = 17U;
  value.beam = 1U;
  value.row_parent = parent;
  value.row_ordinal = row;
  return value;
}

std::string indices_string(const std::vector<std::uint64_t>& values) {
  std::ostringstream output;
  output << "{";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) output << ", ";
    output << values[index] << "U";
  }
  output << "}";
  return output.str();
}

std::vector<std::uint64_t> drain(Cursor* cursor) {
  std::vector<std::uint64_t> selected;
  for (;;) {
    std::uint64_t opportunity = 0U;
    const Result result = cursor->next_success(&opportunity);
    if (result == Result::Success) {
      selected.push_back(opportunity);
      continue;
    }
    if (result == Result::End) break;
    fail(std::string("cursor drain failed: ") +
         sampling::electron_lumi_bernoulli_skip_error_name_v1(
             cursor->error()));
  }
  return selected;
}

void test_edges_and_multiplicity() {
  Cursor empty;
  check(empty.initialize(prefix(), 0.0, 0U),
        "empty zero-probability row failed initialization");
  std::uint64_t opportunity = 123U;
  check(empty.next_success(&opportunity) == Result::End,
        "empty row did not end immediately");
  check(empty.telemetry().row_opportunities == 0U &&
            empty.telemetry().rng_calls == 0U,
        "empty row telemetry is incorrect");

  Cursor integer;
  check(integer.initialize(prefix(), 3.0, 11U),
        "integer expected_store failed initialization");
  check(integer.integer_copies_per_opportunity() == 3U,
        "integer copy count is incorrect");
  check(integer.fractional_probability() == 0.0,
        "integer expected_store retained a fractional probability");
  check(integer.next_success(&opportunity) == Result::End,
        "integer expected_store produced a Bernoulli success");
  check(integer.telemetry().integer_copies_total == 33U &&
            integer.telemetry().selected_copies_total == 33U &&
            integer.telemetry().rng_calls == 0U,
        "integer-copy telemetry is incorrect");

  Cursor fractional;
  constexpr std::uint64_t row_length = 257U;
  check(fractional.initialize(prefix(), 2.375, row_length),
        "fractional expected_store failed initialization");
  const std::vector<std::uint64_t> successes = drain(&fractional);
  std::vector<std::uint64_t> multiplicity(
      row_length, fractional.integer_copies_per_opportunity());
  for (std::size_t item = 0; item < successes.size(); ++item) {
    check(successes[item] < row_length, "success index escaped row bounds");
    ++multiplicity[successes[item]];
  }
  for (std::size_t item = 0; item < multiplicity.size(); ++item)
    check(multiplicity[item] == 2U || multiplicity[item] == 3U,
          "opportunity multiplicity is not floor(s) plus Bernoulli");
  check(fractional.telemetry().selected_copies_total ==
            2U * row_length + successes.size(),
        "fractional total-copy telemetry is incorrect");
  check(fractional.telemetry().rng_calls >= successes.size() &&
            fractional.telemetry().rng_calls <= successes.size() + 1U,
        "geometric cursor did not use O(successes) RNG calls");

  Cursor tiny_probability;
  check(tiny_probability.initialize(prefix(),
                                    std::numeric_limits<double>::min(),
                                    1000U),
        "smallest normal probability failed initialization");
  check(drain(&tiny_probability).empty(),
        "smallest normal probability unexpectedly selected this fixed row");
  check(tiny_probability.telemetry().rng_calls == 1U,
        "terminal geometric gap should use exactly one RNG call; got " +
            std::to_string(tiny_probability.telemetry().rng_calls));
}

void test_known_answer() {
  Cursor cursor;
  check(cursor.initialize(prefix(0x0123456789abcdefull,
                                 0x11000000000000a5ull, 19U),
                          4.3125, 96U),
        "known-answer row failed initialization");
  const std::vector<std::uint64_t> actual = drain(&cursor);
  // Frozen V1 physical-key and inverse-CDF fixture.  Any change requires a
  // new cursor version, never replacement constants under the V1 name.
  const std::uint64_t expected_values[] = {
      2U, 3U, 5U, 7U, 11U, 14U, 15U, 16U, 17U, 18U, 19U,
      23U, 24U, 25U, 26U, 30U, 37U, 38U, 41U, 42U, 43U,
      49U, 50U, 56U, 57U, 59U, 62U, 64U, 65U, 67U, 69U,
      70U, 76U, 78U, 84U, 85U, 86U, 91U, 92U, 93U, 95U};
  const std::vector<std::uint64_t> expected(
      expected_values,
      expected_values + sizeof(expected_values) / sizeof(expected_values[0]));
  if (actual != expected)
    fail("V1 known-answer sequence changed; actual=" + indices_string(actual));
  check(cursor.telemetry().row_opportunities == 96U &&
            cursor.telemetry().opportunities_decided == 96U &&
            cursor.telemetry().rng_calls == 41U &&
            cursor.telemetry().bernoulli_successes == 41U &&
            cursor.telemetry().integer_copies_per_opportunity == 4U &&
            cursor.telemetry().integer_copies_total == 384U &&
            cursor.telemetry().selected_copies_total == 425U,
        "V1 known-answer telemetry changed");
}

void test_partition_and_row_order_invariance() {
  const Prefix row_prefix = prefix(0xfeedfacecafebeefull,
                                   0x1100000000000066ull, 23U);
  Cursor whole;
  check(whole.initialize(row_prefix, 0.2175, 257U),
        "whole-row cursor failed initialization");
  const std::vector<std::uint64_t> expected = drain(&whole);

  Cursor partitioned;
  check(partitioned.initialize(row_prefix, 0.2175, 257U),
        "partitioned cursor failed initialization");
  const std::uint64_t boundaries[] = {0U, 1U, 7U, 19U, 64U, 129U, 257U};
  std::vector<std::uint64_t> actual;
  for (std::size_t boundary = 0;
       boundary < sizeof(boundaries) / sizeof(boundaries[0]); ++boundary) {
    for (;;) {
      std::uint64_t opportunity = 0U;
      const Result result = partitioned.next_success_before(
          boundaries[boundary], &opportunity);
      if (result == Result::Success) {
        actual.push_back(opportunity);
        continue;
      }
      if (boundaries[boundary] == 257U)
        check(result == Result::End,
              "final partition did not report row end");
      else
        check(result == Result::Boundary,
              "intermediate partition did not report its boundary");
      break;
    }
  }
  check(actual == expected,
        "monotone partitioning changed the Bernoulli success sequence");
  check(partitioned.telemetry().rng_calls == whole.telemetry().rng_calls,
        "partitioning changed the number of physical RNG calls");

  std::vector<Prefix> rows;
  for (std::uint64_t row = 0U; row < 9U; ++row)
    rows.push_back(prefix(0xa5a5a5a500000000ull + row,
                          0x1100000000001000ull + 5U * row, row));
  std::map<std::uint64_t, std::vector<std::uint64_t> > forward;
  std::map<std::uint64_t, std::vector<std::uint64_t> > reverse;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    Cursor cursor;
    check(cursor.initialize(rows[row], 0.41, 113U),
          "forward row failed initialization");
    forward[rows[row].row_parent] = drain(&cursor);
  }
  for (std::size_t reverse_index = rows.size(); reverse_index-- > 0U;) {
    Cursor cursor;
    check(cursor.initialize(rows[reverse_index], 0.41, 113U),
          "reverse row failed initialization");
    reverse[rows[reverse_index].row_parent] = drain(&cursor);
  }
  check(forward == reverse,
        "row traversal/scheduling order changed a physical row stream");
}

void expect_initialize_error(const Prefix& row_prefix, double expected_store,
                             std::uint64_t row_length, const Limits& limits,
                             Error expected_error, const char* label) {
  Cursor cursor;
  check(!cursor.initialize(row_prefix, expected_store, row_length, limits),
        std::string(label) + " did not fail closed");
  check(cursor.error() == expected_error,
        std::string(label) + " reported " +
            sampling::electron_lumi_bernoulli_skip_error_name_v1(
                cursor.error()));
  std::uint64_t ignored = 0U;
  check(cursor.next_success(&ignored) == Result::Error,
        std::string(label) + " allowed use after failure");
}

void test_invalid_and_hard_bounds() {
  Limits defaults;
  expect_initialize_error(prefix(), -0.25, 10U, defaults,
                          Error::InvalidExpectedStore,
                          "negative expected_store");
  expect_initialize_error(prefix(),
                          std::numeric_limits<double>::quiet_NaN(), 10U,
                          defaults, Error::InvalidExpectedStore,
                          "NaN expected_store");
  expect_initialize_error(prefix(),
                          std::numeric_limits<double>::infinity(), 10U,
                          defaults, Error::InvalidExpectedStore,
                          "infinite expected_store");
  expect_initialize_error(prefix(), std::ldexp(1.0, 64), 1U, defaults,
                          Error::IntegerCopiesOverflow,
                          "integer-copy conversion overflow");

  Prefix bad_version = prefix();
  bad_version.cursor_version = 2U;
  expect_initialize_error(bad_version, 0.5, 10U, defaults,
                          Error::UnsupportedVersion, "unknown version");
  Prefix bad_parent = prefix();
  bad_parent.row_parent = feebl::kNoParentStableIdV2;
  expect_initialize_error(bad_parent, 0.5, 10U, defaults,
                          Error::InvalidPhysicalPrefix,
                          "missing stable row parent");
  Prefix bad_beam = prefix();
  bad_beam.beam = 0U;
  expect_initialize_error(bad_beam, 0.5, 10U, defaults,
                          Error::InvalidPhysicalPrefix, "invalid beam");

  Limits row_limit;
  row_limit.max_opportunities = 9U;
  expect_initialize_error(prefix(), 0.5, 10U, row_limit,
                          Error::RowLengthLimit, "row hard bound");
  expect_initialize_error(prefix(), 2.0, feebl::kMaxCounterOrdinalV2,
                          defaults, Error::IntegerCopyProductOverflow,
                          "integer-copy product overflow");
  Limits total_limit;
  total_limit.max_total_copies = 11U;
  expect_initialize_error(prefix(), 3.0, 4U, total_limit,
                          Error::TotalCopyLimit, "base-copy hard bound");
  Limits invalid_rng_limit;
  invalid_rng_limit.max_rng_calls =
      sampling::ElectronLumiBernoulliSkipLimitsV1().max_rng_calls + 1U;
  expect_initialize_error(prefix(), 0.5, 10U, invalid_rng_limit,
                          Error::InvalidLimits, "unrepresentable RNG limit");
  Limits zero_rng;
  zero_rng.max_rng_calls = 0U;
  expect_initialize_error(prefix(), 0.5, 10U, zero_rng,
                          Error::RngLimit, "zero RNG-call bound");

  const double almost_one = std::nextafter(1.0, 0.0);
  Limits success_limit;
  success_limit.max_bernoulli_successes = 0U;
  Cursor capped_success;
  check(capped_success.initialize(prefix(), almost_one, 10U, success_limit),
        "success-capped cursor failed initialization too early");
  std::uint64_t opportunity = 0U;
  check(capped_success.next_success(&opportunity) == Result::Error &&
            capped_success.error() == Error::SuccessLimit,
        "success hard bound did not poison before publication");

  Limits fractional_total_limit;
  fractional_total_limit.max_total_copies = 20U;
  Cursor capped_total;
  check(capped_total.initialize(prefix(), 2.75, 10U,
                                fractional_total_limit),
        "fractional total-capped cursor failed initialization too early");
  check(capped_total.next_success(&opportunity) == Result::Error &&
            capped_total.error() == Error::TotalCopyLimit,
        "fractional total-copy bound did not poison before publication");

  Limits one_rng;
  one_rng.max_rng_calls = 1U;
  Cursor capped_rng;
  check(capped_rng.initialize(prefix(), almost_one, 10U, one_rng),
        "one-draw cursor failed initialization");
  check(capped_rng.next_success(&opportunity) == Result::Success,
        "one-draw cursor did not publish its first known success");
  check(capped_rng.next_success(&opportunity) == Result::Error &&
            capped_rng.error() == Error::RngLimit,
        "RNG-call bound did not fail before a second draw");

  Cursor boundary;
  check(boundary.initialize(prefix(), 0.4, 20U),
        "boundary test failed initialization");
  check(boundary.next_success_before(21U, &opportunity) == Result::Error &&
            boundary.error() == Error::InvalidBoundary,
        "out-of-row boundary did not fail closed");
  Cursor decreasing;
  check(decreasing.initialize(prefix(), 0.4, 20U),
        "decreasing-boundary test failed initialization");
  const Result first_partition =
      decreasing.next_success_before(10U, &opportunity);
  check(first_partition == Result::Success || first_partition == Result::Boundary,
        "first monotone boundary unexpectedly failed");
  check(decreasing.next_success_before(9U, &opportunity) == Result::Error &&
            decreasing.error() == Error::InvalidBoundary,
        "decreasing boundary did not fail closed");

  Cursor null_output;
  check(null_output.initialize(prefix(), 0.4, 20U),
        "null-output test failed initialization");
  check(null_output.next_success(0) == Result::Error &&
            null_output.error() == Error::NullOutput,
        "null success destination did not poison the cursor");
}

void test_many_seed_mean_and_variance() {
  constexpr std::uint64_t seeds = 4096U;
  constexpr std::uint64_t row_length = 257U;
  constexpr double probability = 0.2;
  std::vector<double> counts;
  counts.reserve(seeds);
  std::uint64_t total_successes = 0U;
  std::uint64_t total_rng_calls = 0U;
  for (std::uint64_t seed = 0U; seed < seeds; ++seed) {
    Cursor cursor;
    check(cursor.initialize(prefix(0x9e3779b97f4a7c15ull + seed,
                                   0x1100000010000000ull + seed, 31U),
                            probability, row_length),
          "many-seed cursor failed initialization");
    const std::vector<std::uint64_t> selected = drain(&cursor);
    counts.push_back(static_cast<double>(selected.size()));
    total_successes += selected.size();
    total_rng_calls += cursor.telemetry().rng_calls;
  }
  double mean = 0.0;
  for (std::size_t item = 0; item < counts.size(); ++item) mean += counts[item];
  mean /= static_cast<double>(counts.size());
  double variance = 0.0;
  for (std::size_t item = 0; item < counts.size(); ++item) {
    const double residual = counts[item] - mean;
    variance += residual * residual;
  }
  variance /= static_cast<double>(counts.size() - 1U);

  const double expected_mean = row_length * probability;
  const double expected_variance =
      row_length * probability * (1.0 - probability);
  const double mean_standard_error =
      std::sqrt(expected_variance / static_cast<double>(seeds));
  check(std::fabs(mean - expected_mean) <= 6.0 * mean_standard_error,
        "many-seed mean is inconsistent with Binomial(N,p)");
  check(std::fabs(variance - expected_variance) <=
            0.15 * expected_variance,
        "many-seed variance is inconsistent with Binomial(N,p)");
  check(total_rng_calls >= total_successes &&
            total_rng_calls <= total_successes + seeds,
        "aggregate RNG calls are not K or K+1 per row");
  std::cout << "bernoulli_skip_statistics rows=" << seeds
            << " row_length=" << row_length
            << " p=" << probability
            << " mean=" << mean
            << " expected_mean=" << expected_mean
            << " variance=" << variance
            << " expected_variance=" << expected_variance
            << " successes=" << total_successes
            << " rng_calls=" << total_rng_calls << std::endl;
}

}  // namespace

int main() {
  test_edges_and_multiplicity();
  test_known_answer();
  test_partition_and_row_order_invariance();
  test_invalid_and_hard_bounds();
  test_many_seed_mean_and_variance();
  std::cout << "PASS: versioned geometric-gap electron-luminosity cursor"
            << std::endl;
  return 0;
}
