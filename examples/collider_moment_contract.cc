// Stable beam moments: narrow energy peaks, unequal weights and interval merging.
#include "colliderDiagnosticsCPP.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using feebl::diagnostics::Moment;
  const double energy_GeV = 45.6;
  const double low = energy_GeV - 1e-8, high = energy_GeV + 1e-8;
  Moment first, second, total, mono;
  for (int i = 0; i < 50000; ++i) {
    first.add(1.0, low); second.add(3.0, high);
    total.add(1.0, low); total.add(3.0, high);
    mono.add(1.0 + i % 3, energy_GeV);
  }
  Moment merged = first; merged.merge(second);
  const double expected_mean = low + .75 * (high - low);
  const double expected_rms = std::sqrt(3.0) * (high - low) / 4.0;
  for (const Moment* value : {&total, &merged}) {
    if (!value->finite() || std::abs(value->mean() - expected_mean) > 1e-14 ||
        std::abs(value->rms() - expected_rms) > expected_rms * 1e-10)
      throw std::runtime_error("narrow weighted moment failed");
  }
  const Moment remainder = merged.minus(first);
  if (mono.rms() != 0.0 || mono.mean() != energy_GeV ||
      std::abs(remainder.mean() - high) > 1e-14 || remainder.rms() > 1e-15)
    throw std::runtime_error("monoenergetic/subset moment failed");
  Moment zero; zero.add(0.0, 1e100); zero.merge(mono);
  if (zero.rms() != 0.0 || zero.mean() != energy_GeV)
    throw std::runtime_error("zero-weight input changed origin");
  std::cout << "PASS: shifted weighted mean/RMS, unequal weights, merged intervals, zero spread\n";
}
