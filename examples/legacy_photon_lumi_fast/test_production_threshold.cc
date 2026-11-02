#include "legacyPhotonLumiFastCPP.h"

#include <cmath>
#include <cstdio>
#include <limits>

// Source-level copy of the frozen GRID::collide_gg spelling.  This function
// deliberately does not call the shared helper: under the production
// -ffast-math contract it is the independent oracle for the compiler's
// sqrt(s)*sqrt(s) folding.
static bool frozen_source_threshold(float first_energy, float second_energy,
                                    double gg_smin) {
  const double ecm = std::sqrt(4.0 * first_energy * second_energy);
  return ecm * ecm > gg_smin;
}

int main() {
  const float ecm_min_gg = 0.3f;
  const float cut = 4.0f * ecm_min_gg * ecm_min_gg;
  const float equal_partner = cut * 0.25f;
  const float below = std::nextafter(equal_partner, 0.0f);
  const float above = std::nextafter(
      equal_partner, std::numeric_limits<float>::infinity());

  bool pass =
      !feebl::legacy_photon_lumi::gg_above_threshold(1.0f, equal_partner,
                                                     cut) &&
      !feebl::legacy_photon_lumi::gg_above_threshold(1.0f, below, cut) &&
      feebl::legacy_photon_lumi::gg_above_threshold(1.0f, above, cut);

  const float energies[] = {
      std::numeric_limits<float>::min(), 1.0e-20f, 0.125f, 0.3f,
      1.0f, 2.0f, 9.5f, 1000.0f, 1.0e10f,
      std::numeric_limits<float>::max() * 0.125f};
  std::size_t comparisons = 0;
  for (float first : energies) {
    for (float second : energies) {
      const double boundary =
          (4.0 * static_cast<double>(first)) *
          static_cast<double>(second);
      const double probes[] = {
          std::nextafter(boundary, 0.0), boundary,
          std::nextafter(boundary,
                         std::numeric_limits<double>::infinity())};
      for (double probe : probes) {
        const bool historical =
            frozen_source_threshold(first, second, probe);
        const bool shared =
            feebl::legacy_photon_lumi::gg_above_threshold(first, second,
                                                          probe);
        pass = pass && historical == shared;
        ++comparisons;
      }
    }
  }

  std::printf("production -ffast-math threshold equality/ULP probe "
              "(%zu independent comparisons): %s\n", comparisons,
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
