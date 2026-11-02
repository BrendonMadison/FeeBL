#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "counterRngCPP.h"
#include "electronLumiSamplerCPP.h"

namespace sampling = feebl::electron_lumi_sampling;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

sampling::PairIdentity identity(std::uint64_t ordinal) {
  sampling::PairIdentity value;
  value.run_seed = 0x5a17c0decafebeefull;
  value.crossing = ordinal / 200003U;
  value.diagonal = ordinal / 1009U;
  value.wave = ordinal / 251U;
  value.substep = ordinal / 67U;
  value.encounter = ordinal / 17U;
  value.parent_a = 0x1000000000000000ull + ordinal;
  value.parent_b = 0x2000000000000000ull + 3U * ordinal;
  value.interaction = ordinal % 13U;
  return value;
}

feebl::PhysicalRandomKeyV2 key(const sampling::PairIdentity& value) {
  feebl::PhysicalRandomKeyV2 result;
  result.run_seed = value.run_seed;
  result.domain = feebl::RandomDomain::LuminosityEeSampling;
  result.crossing = value.crossing;
  result.diagonal = value.diagonal;
  result.wave = value.wave;
  result.substep = value.substep;
  result.encounter = value.encounter;
  result.beam = 0U;
  result.parent_a = value.parent_a;
  result.parent_b = value.parent_b;
  result.interaction = value.interaction;
  result.emission = 0U;
  result.rejection_attempt = 0U;
  return result;
}

bool same(const std::vector<sampling::BernoulliSelection>& left,
          const std::vector<sampling::BernoulliSelection>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!(left[index].identity == right[index].identity) ||
        left[index].multiplicity != right[index].multiplicity ||
        left[index].payload_token != right[index].payload_token)
      return false;
  return true;
}

void test_zero_probability_is_the_minimal_fast_path() {
  sampling::KeyedBernoulliSampler sampler(8192U);
  for (std::uint64_t item = 0; item < 4096U; ++item)
    check(sampler.offer(identity(item), 0.0, item).status ==
              sampling::OfferStatus::Rejected,
          "zero-probability pair was accepted");
  check(sampler.opportunities() == 4096U,
        "fast-path opportunity telemetry is wrong");
  check(sampler.rejected() == 4096U,
        "fast-path rejection telemetry is wrong");
  check(sampler.accepted_identity_lookups() == 0U,
        "rejected pairs performed accepted-identity hash work");
  check(sampler.empty() && sampler.copies() == 0U,
        "rejected pairs consumed sample storage");
}

void test_legacy_multiplicity_law_and_statistics() {
  constexpr std::uint64_t trials = 200000U;
  constexpr double probability = 0.075;
  sampling::KeyedBernoulliSampler sampler(trials);
  for (std::uint64_t item = 0; item < trials; ++item)
    sampler.offer(identity(item), probability, item);
  const double observed =
      static_cast<double>(sampler.copies()) / static_cast<double>(trials);
  const double sigma = std::sqrt(probability * (1.0 - probability) /
                                 static_cast<double>(trials));
  check(std::fabs(observed - probability) < 6.0 * sigma,
        "store<1 acceptance rate is inconsistent with Bernoulli(weight*p)");

  sampling::KeyedBernoulliSampler general(1024U);
  for (std::uint64_t item = 0; item < 128U; ++item) {
    const sampling::PairIdentity value = identity(300000U + item);
    const bool extra = feebl::CounterRngV2::uniform_float(key(value), 0U) <=
                       0.375;
    const sampling::OfferResult offered = general.offer(value, 2.375, item);
    check(offered.accepted(), "general finite multiplicity was rejected");
    check(offered.multiplicity == static_cast<std::size_t>(2U + extra),
          "floor(store)+Bernoulli(frac(store)) law changed");
  }
}

void test_order_and_lane_invariance() {
  constexpr std::size_t candidates = 4096U;
  constexpr std::size_t capacity = 8192U;
  sampling::KeyedBernoulliSampler forward(capacity);
  sampling::KeyedBernoulliSampler reverse(capacity);
  for (std::size_t item = 0; item < candidates; ++item)
    forward.offer(identity(item), 0.31, item);
  for (std::size_t item = candidates; item-- > 0U;)
    reverse.offer(identity(item), 0.31, item);
  const std::vector<sampling::BernoulliSelection> reference =
      forward.canonical_selection();
  check(same(reference, reverse.canonical_selection()),
        "offer traversal changed keyed Bernoulli selection");

  const std::size_t worker_counts[] = {1U, 2U, 3U, 8U, 17U};
  for (std::size_t w = 0; w < sizeof(worker_counts) / sizeof(worker_counts[0]);
       ++w) {
    const std::size_t workers = worker_counts[w];
    std::vector<sampling::KeyedBernoulliSampler> lanes;
    lanes.reserve(workers);
    for (std::size_t lane = 0; lane < workers; ++lane)
      lanes.push_back(sampling::KeyedBernoulliSampler(capacity));
    for (std::size_t lane = 0; lane < workers; ++lane)
      for (std::size_t item = lane; item < candidates; item += workers)
        lanes[lane].offer(identity(item), 0.31, item);
    sampling::KeyedBernoulliSampler merged;
    std::string reason;
    check(sampling::KeyedBernoulliSampler::merge_lanes(
              lanes, capacity, &merged, &reason),
          reason.empty() ? "keyed Bernoulli lane merge failed"
                         : reason.c_str());
    check(same(reference, merged.canonical_selection()),
          "worker assignment changed keyed Bernoulli selection");
  }
}

void test_fail_closed_capacity_and_inputs() {
  sampling::KeyedBernoulliSampler bounded(2U);
  check(bounded.offer(identity(1U), 2.0).multiplicity == 2U,
        "integral multiplicity was not retained");
  const sampling::OfferResult overflow = bounded.offer(identity(2U), 1.0);
  check(overflow.status == sampling::OfferStatus::CapacityExceeded,
        "num_lumi overflow did not fail closed");
  check(!bounded.healthy(), "capacity overflow did not poison the sampler");
  check(bounded.copies() == 2U,
        "capacity failure partially modified selected records");

  sampling::KeyedBernoulliSampler invalid(8U);
  sampling::PairIdentity missing = identity(4U);
  missing.parent_a = feebl::kNoParentStableIdV2;
  check(invalid.offer(missing, 0.5).status ==
            sampling::OfferStatus::InvalidIdentity,
        "missing stable ID did not fail closed");
  check(invalid.offer(identity(5U), -0.1).status ==
            sampling::OfferStatus::InvalidWeight,
        "negative expected store did not fail closed");
  check(invalid.offer(identity(6U),
                      std::numeric_limits<double>::infinity()).status ==
            sampling::OfferStatus::InvalidWeight,
        "infinite expected store did not fail closed");

  sampling::KeyedBernoulliSampler duplicate(8U);
  check(duplicate.offer(identity(7U), 1.0).accepted(),
        "first accepted identity failed");
  check(duplicate.offer(identity(7U), 1.0).status ==
            sampling::OfferStatus::DuplicateSelectedIdentity,
        "duplicate accepted identity did not fail closed");
  check(!duplicate.healthy(), "duplicate identity did not poison sampler");
}

}  // namespace

int main() {
  test_zero_probability_is_the_minimal_fast_path();
  test_legacy_multiplicity_law_and_statistics();
  test_order_and_lane_invariance();
  test_fail_closed_capacity_and_inputs();
  std::cout << "fast keyed Bernoulli electron-luminosity tests passed"
            << std::endl;
  return 0;
}
