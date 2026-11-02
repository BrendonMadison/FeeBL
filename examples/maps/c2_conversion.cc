// How much does the C2 conversion actually matter?
//
// doc/CONVENTIONS.md C2 states that FEEBL's vx_ is the trajectory angle x',
// while the canonical maps act on px = x'(1+delta). That is a warning until
// somebody puts a number on it, so: apply the reference crab map both ways to
// the same particle and measure the divergence error.
//
//   correct : px = x'(1+delta) -> map -> x' = px/(1+delta)
//   naive   : feed x' to the map as if it were px, read the result back as x'
//
// The leading error is a missing (1+delta) on the transverse kick. The kick is
// K(zeta) ~ kappa*zeta, NOT kappa, so the error scales with distance along the
// bunch as well as with delta -- it is small at the core and largest in the
// tails. It vanishes identically at delta = 0, which is why a suite built only
// from on-momentum particles would never see it.
#include "reference_maps.hpp"
#include <cstdio>
#include <cmath>

int main()
{
  using gpgeom::CrabRFCavityMap;
  using gpgeom::Harmonic;
  using gpgeom::PhaseSpace6D;

  const double k     = 8.377580409572781;      // 400 MHz, 2*pi*f/c
  const double theta = 0.030;                  // full crossing angle [rad]
  const double kappa = theta / 2.0;            // dx/dzeta at the IP
  const auto   a     = CrabRFCavityMap::cubic_linearizer(kappa, k, 2);
  CrabRFCavityMap crab(k, {{1, a[0], 0.0}, {2, a[1], 0.0}});

  // FCC-ee-Z-like scales, for context.
  const double eps_geom = 0.71e-9;             // geometric emittance [m.rad]
  const double beta_x   = 0.10;                // [m]
  const double sig_xp   = std::sqrt(eps_geom / beta_x);
  const double sig_z    = 14.5e-3;             // bunch length [m]
  const double xp0      = 1.0e-5;

  printf("crab tilt kappa = %.4f, sigma_x' = %.1f urad, sigma_z = %.1f mm\n",
         kappa, sig_xp * 1e6, sig_z * 1e3);

  printf("\n-- error vs delta, at zeta = 1 sigma_z --\n");
  printf("  %10s %14s %14s %12s\n", "delta", "err [rad]", "of sigma_x'", "of kick");
  for (double delta : {0.0, 1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3}) {
    PhaseSpace6D pc{0.0, xp0*(1.0+delta), 0.0, 0.0, sig_z, delta};
    crab.apply(pc);
    const double xp_correct = pc.px / (1.0 + delta);
    PhaseSpace6D pn{0.0, xp0, 0.0, 0.0, sig_z, delta};
    crab.apply(pn);
    const double err  = pn.px - xp_correct;
    const double kick = xp_correct - xp0;
    printf("  %10.1e %14.3e %13.3f%% %11.3f%%\n", delta, err,
           100.0*std::fabs(err)/sig_xp,
           kick != 0.0 ? 100.0*std::fabs(err/kick) : 0.0);
  }

  printf("\n-- error vs position along the bunch, at delta = 1e-3 --\n");
  printf("  %12s %14s %14s\n", "zeta", "err [rad]", "of sigma_x'");
  const double delta = 1.0e-3;
  for (double nz : {0.5, 1.0, 2.0, 3.0}) {
    const double zeta = nz * sig_z;
    PhaseSpace6D pc{0.0, xp0*(1.0+delta), 0.0, 0.0, zeta, delta};
    crab.apply(pc);
    const double xp_correct = pc.px / (1.0 + delta);
    PhaseSpace6D pn{0.0, xp0, 0.0, 0.0, zeta, delta};
    crab.apply(pn);
    const double err = pn.px - xp_correct;
    printf("  %10.1f sz %14.3e %13.3f%%\n", nz, err,
           100.0*std::fabs(err)/sig_xp);
  }

  printf("\n  Zero at delta = 0, linear in delta, and growing along the bunch.\n"
         "  Systematic, not random: it biases the crabbed tilt rather than\n"
         "  adding noise, so it does not average away over macroparticles.\n");
  return 0;
}
