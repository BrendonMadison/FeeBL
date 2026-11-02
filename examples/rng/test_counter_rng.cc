#include "counterRngCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

feebl::RandomKey key_for(std::uint64_t entity) {
  feebl::RandomKey key;
  key.run_seed = 20260825;
  key.domain = feebl::RandomDomain::Beamstrahlung;
  key.beam = 2;
  key.overlap = 17;
  key.timestep = 4;
  key.slice = 9;
  key.entity = entity;
  key.interaction = 3;
  return key;
}

}  // namespace

int main() {
  {
    feebl::PhiloxBlock zero = {{0u, 0u, 0u, 0u}};
    const feebl::PhiloxBlock got = feebl::Philox4x32_10::eval(zero, 0u, 0u);
    // Published Random123 known-answer vector for Philox4x32-10.
    check(got.value[0] == 0x6627e8d5u, "Philox KAT lane 0");
    check(got.value[1] == 0xe169c58du, "Philox KAT lane 1");
    check(got.value[2] == 0xbc57ac4cu, "Philox KAT lane 2");
    check(got.value[3] == 0x9b00dbd8u, "Philox KAT lane 3");
  }

  const feebl::RandomKey key = key_for(42);
  for (std::uint64_t draw = 0; draw < 1000; ++draw) {
    check(feebl::CounterRng::uint32(key, draw) ==
              feebl::CounterRng::uint32(key, draw),
          "same key and draw must repeat");
    const double u = feebl::CounterRng::uniform_double(key, draw);
    check(u > 0.0 && u < 1.0, "uniform_double is in the open unit interval");
  }

  {
    feebl::RandomKey other = key;
    other.domain = feebl::RandomDomain::SpinFlip;
    check(feebl::CounterRng::uint32(key, 0) !=
              feebl::CounterRng::uint32(other, 0),
          "process domains must be separated");
    other = key;
    other.entity += 1;
    check(feebl::CounterRng::uint32(key, 0) !=
              feebl::CounterRng::uint32(other, 0),
          "entity streams must be separated");
  }

  // Processing order must not affect the value assigned to an entity.
  std::vector<std::uint64_t> entities(2048);
  std::iota(entities.begin(), entities.end(), 0);
  std::vector<std::uint32_t> forward(entities.size());
  std::vector<std::uint32_t> reverse(entities.size());
  for (std::uint64_t entity : entities)
    forward[entity] = feebl::CounterRng::uint32(key_for(entity), 11);
  std::reverse(entities.begin(), entities.end());
  for (std::uint64_t entity : entities)
    reverse[entity] = feebl::CounterRng::uint32(key_for(entity), 11);
  check(forward == reverse, "entity traversal order changes no draws");

  // A broad deterministic sanity check of the stateless normal transform.
  const std::size_t n = 200000;
  double sum = 0.0;
  double sum2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = feebl::CounterRng::normal(key_for(i), 0);
    check(std::isfinite(x), "normal draw is finite");
    sum += x;
    sum2 += x * x;
  }
  const double mean = sum / n;
  const double variance = sum2 / n - mean * mean;
  check(std::fabs(mean) < 0.01, "normal mean sanity bound");
  check(std::fabs(variance - 1.0) < 0.02, "normal variance sanity bound");

  if (failures != 0) {
    std::cerr << failures << " counter-RNG checks failed\n";
    return 1;
  }
  std::cout << "counter RNG: PASS; mean=" << mean
            << " variance=" << variance << '\n';
  return 0;
}
