// Emission-probability probe of FeeBL's spin-dependent radiation
// sampler, PHYSTOOLS::synrad_0_spin_flip, at fixed field.
//
// One call is one proposal interval. The proposal probability p0 is fixed by
// choosing dzOnRadius; spin is set along e2 (s2 = +1, -1) or zero. Each run
// uses its own seed so settings are statistically independent, unlike a
// reused-seed design. Output is one whitespace-separated record per run:
// chi exposure s2 seed trials emitted sum_x sum_x2 p0_code
#include "physicalTools.h"
#include "rndmCPP.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
  if (argc != 6) {
    std::fprintf(stderr, "usage: chi exposure s2 seed trials\n");
    return 2;
  }
  const float chi = std::strtof(argv[1], nullptr);
  const double exposure = std::strtod(argv[2], nullptr);
  const double s2 = std::strtod(argv[3], nullptr);
  const unsigned long seed = std::strtoul(argv[4], nullptr, 10);
  const long trials = std::strtol(argv[5], nullptr, 10);

  // Reproduce the proposal probability exactly as synrad_0_spin_flip forms it.
  const float energy_gev = 250.0f;
  const double gamma = energy_gev / EMASS;
  const double upsilon_bar = 1.5 * static_cast<double>(chi);
  const double factor = std::pow(1.0 + 0.5 * upsilon_bar, 0.33333333333);
  const float dz_on_radius = static_cast<float>(exposure * factor / (CONST1 * gamma));
  const double p0_code = CONST1 * static_cast<double>(dz_on_radius) * gamma / factor;

  RNDM rng(seed);
  const TRIDVECTOR e1(1, 0, 0), e2(0, 1, 0), e3(0, 0, 1);
  long emitted = 0;
  double sum_x = 0.0, sum_x2 = 0.0;
  for (long i = 0; i < trials; ++i) {
    TRIDVECTOR polar(0.0, s2, 0.0);
    TRIDVECTOR stokes;
    float photon_gev = 0.0f;
    if (PHYSTOOLS::synrad_0_spin_flip(chi, energy_gev, e1, e2, e3, polar, stokes,
                                      dz_on_radius, &photon_gev, rng)) {
      ++emitted;
      const double x = photon_gev / energy_gev;
      sum_x += x;
      sum_x2 += x * x;
    }
  }
  std::printf("%.9g %.9g %g %lu %ld %ld %.17g %.17g %.17g\n", static_cast<double>(chi),
              exposure, s2, seed, trials, emitted, sum_x, sum_x2, p0_code);
  return 0;
}
