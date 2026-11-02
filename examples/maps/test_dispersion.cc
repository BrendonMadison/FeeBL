// IP dispersion: symplecticity, exact inversion, and the spectrum it narrows.
//
// Three arms. The first two are properties of the map alone and hold whatever
// the physics is. The third checks the claim that motivates the map existing at
// all.
//
// The symplecticity arm carries a deliberate failure case. A map that shears x
// with delta but omits the zeta companion is exactly the mistake a hurried
// implementation makes, so the test builds that broken map explicitly and
// REQUIRES it to fail. Without that, M^T J M = J would pass on any
// near-identity map and this file would prove nothing.

#include "gpgeom/maps.hpp"
#include "gpgeom/phase_space.hpp"
#include "pairMomentsCPP.h"

#include <cmath>
#include <cstdio>
#include <random>

using gpgeom::Matrix6;
using gpgeom::PhaseSpace6D;

static int failures = 0;

static void ok(bool cond, const char* what)
{
  std::printf("  %-58s %s\n", what, cond ? "PASS" : "FAIL");
  if (!cond) failures++;
}

// max |(M^T J M - J)_ij| : zero exactly when M is symplectic.
static double symplectic_residual(const Matrix6& m)
{
  const Matrix6 j = gpgeom::canonical_symplectic_matrix();
  double worst = 0.0;
  for (int a = 0; a < 6; a++)
    for (int b = 0; b < 6; b++)
      {
        double s = 0.0;
        for (int c = 0; c < 6; c++)
          for (int d = 0; d < 6; d++)
            s += m[c][a] * j[c][d] * m[d][b];
        const double diff = std::fabs(s - j[a][b]);
        if (diff > worst) worst = diff;
      }
  return worst;
}

int main()
{
  const double D = 0.012;

  std::printf("=== the map is symplectic, and the test can tell ===\n");

  PhaseSpace6D around;
  around.x = 3.0e-6; around.px = 1.1e-5;
  around.y = -2.0e-6; around.py = 4.0e-6;
  around.zeta = 1.5e-4; around.delta = 7.0e-4;

  for (int vertical = 0; vertical < 2; vertical++)
    {
      const gpgeom::IPDispersionMap map(D, vertical != 0);
      const Matrix6 m = gpgeom::numerical_jacobian(
          [&](PhaseSpace6D& p) { map.apply(p); }, around);
      const double r = symplectic_residual(m);
      std::printf("      %s plane residual %.3e\n",
                  vertical ? "vertical  " : "horizontal", r);
      ok(r < 1.0e-9, vertical ? "vertical map is symplectic"
                              : "horizontal map is symplectic");
    }

  // The broken map: shear x with delta, drop the zeta companion. This is the
  // whole point of the arm -- it must be DETECTED, not tolerated.
  {
    const Matrix6 m = gpgeom::numerical_jacobian(
        [&](PhaseSpace6D& p) { p.x += D * p.delta; }, around);
    const double r = symplectic_residual(m);
    std::printf("      companion dropped -> residual %.3e (must be large)\n", r);
    ok(r > 1.0e-6, "dropping the zeta companion is CAUGHT as non-symplectic");
  }

  std::printf("\n=== the inverse is exact, not first order ===\n");
  {
    std::mt19937 rng(20260823u);
    std::normal_distribution<double> g(0.0, 1.0);
    double worst = 0.0;
    for (int i = 0; i < 20000; i++)
      {
        PhaseSpace6D p;
        p.x = 1.0e-5 * g(rng);  p.px = 1.0e-4 * g(rng);
        p.y = 1.0e-5 * g(rng);  p.py = 1.0e-4 * g(rng);
        p.zeta = 1.0e-3 * g(rng); p.delta = 1.0e-2 * g(rng);
        const PhaseSpace6D orig = p;

        const gpgeom::IPDispersionMap map(D, (i % 2) != 0);
        map.apply(p);
        map.apply_inverse(p);

        const std::array<double, 6> a = p.as_array();
        const std::array<double, 6> b = orig.as_array();
        // Scale by the largest intermediate the coordinate passed through, not
        // by the coordinate itself. x + D*delta - D*delta is exact in real
        // arithmetic but cancels in floating point, and a Gaussian draw will
        // sooner or later put x near zero while D*delta is not -- dividing by
        // that x measures the cancellation, not the map.
        const double shear = std::fabs(D * orig.delta) + std::fabs(D * orig.px);
        for (int k = 0; k < 6; k++)
          {
            const double scale = std::fmax(std::fabs(b[k]), shear);
            const double rel = std::fabs(a[k] - b[k]) / (scale > 0.0 ? scale : 1.0);
            if (rel > worst) worst = rel;
          }
      }
    std::printf("      worst relative round-trip residual %.3e\n", worst);
    ok(worst < 1.0e-14, "apply then apply_inverse returns the original");
  }

  // The physics the map exists for, and the reason the SIGN is not a free
  // choice.
  //
  // Overlap selects on the transverse separation x1 - x2 = D*(d1 -+ d2), while
  // W depends on d1 + d2. For Gaussian energy spreads the sum and the
  // difference are INDEPENDENT, so:
  //
  //   opposite sign -> separation tracks (d1 + d2), the same combination W
  //                    depends on, and the spectrum narrows;
  //   same sign     -> separation tracks (d1 - d2), which is independent of
  //                    W, so the spectrum is left alone.
  //
  // Same-sign dispersion therefore does NOT broaden dL/dW -- it does nothing to
  // it, to first order. An earlier version of this test asserted broadening and
  // was wrong; the measurement below is what corrected it.
  //
  // SCOPE. This is a toy model: Gaussian energy spreads, a Gaussian overlap
  // weight, no beamstrahlung and no dilution. Within it, same-sign dispersion
  // provably does nothing to dL/dW, and that is what the assertions below
  // check.
  //
  // In a FULL simulation that is no longer true on its own, because separating
  // the beams transversely reduces the beam-beam field and therefore the
  // beamstrahlung, which narrows W for EITHER sign. Measured on the quick deck
  // at D*sigma_delta = 4.6 sigma_x: no dispersion 5.361 GeV rms, opposite
  // 0.405, same 0.690 -- with the opposite and same cases having identical
  // luminosity and beamstrahlung, since only the relative sign differs.
  //
  // So the monochromatization figure is opposite-against-same at equal |D|
  // (1.70x), never against the undispersed case (13x, mostly dilution). See
  // Opposite IP dispersions select anticorrelated beam-energy deviations.
  //
  // Sizes matter as much as the sign: the effect scales with D*sigma_delta
  // measured in beam sigmas, so this uses ILC250-like numbers.
  std::printf("\n=== opposite dispersion narrows dL/dW; same-sign cannot ===\n");
  {
    const double E0 = 125.0;
    const double m_e = 0.000510999;
    const double sigma_d = 0.00124;    // ILC250 relative energy spread
    const double sigma_x = 515.0e-9;   // ILC250 horizontal IP size, metres
    const double Dphys = 3.0 * sigma_x / sigma_d;

    std::printf("      D = %.4f mm, so D*sigma_delta = %.2f sigma_x\n",
                Dphys * 1000.0, Dphys * sigma_d / sigma_x);

    double rms[3] = {0.0, 0.0, 0.0};
    double vsum[3] = {0.0, 0.0, 0.0};
    double vdif[3] = {0.0, 0.0, 0.0};
    const double dsign[3] = {0.0, -1.0, +1.0};
    const char* label[3] = {"no dispersion", "opposite sign", "same sign"};

    for (int mode = 0; mode < 3; mode++)
      {
        const gpgeom::IPDispersionMap map1(Dphys, false);
        const gpgeom::IPDispersionMap map2(dsign[mode] * Dphys, false);
        double sw = 0.0, swW = 0.0, swWW = 0.0;
        double ss = 0.0, sss = 0.0, sd = 0.0, sdd = 0.0;

        std::mt19937 rng(7u);
        std::normal_distribution<double> g(0.0, 1.0);
        for (int i = 0; i < 400000; i++)
          {
            PhaseSpace6D p1, p2;
            p1.delta = sigma_d * g(rng);
            p2.delta = sigma_d * g(rng);
            if (mode == 0) { p1.x = 0.0; p2.x = 0.0; }
            else           { map1.apply(p1); map2.apply(p2); }

            const double dx = p1.x - p2.x;
            // Two beams of size sigma_x overlap with a width of sqrt(2)*sigma_x.
            const double w = std::exp(-0.25 * dx * dx / (sigma_x * sigma_x));

            const double e1 = E0 * (1.0 + p1.delta);
            const double e2 = E0 * (1.0 + p2.delta);
            const double W = pairmoments::invariant_mass(
                e1, 0.0, 0.0, 1.0, e2, 0.0, 0.0, -1.0, m_e);

            const double su = p1.delta + p2.delta;
            const double di = p1.delta - p2.delta;
            sw += w; swW += w * W; swWW += w * W * W;
            ss += w * su; sss += w * su * su;
            sd += w * di; sdd += w * di * di;
          }
        const double mean = swW / sw;
        rms[mode] = std::sqrt(std::fmax(swWW / sw - mean * mean, 0.0));
        vsum[mode] = std::sqrt(std::fmax(sss / sw - (ss / sw) * (ss / sw), 0.0));
        vdif[mode] = std::sqrt(std::fmax(sdd / sw - (sd / sw) * (sd / sw), 0.0));
        std::printf("      %-14s  rms W = %.5f GeV   sd(d1+d2) = %.2e"
                    "   sd(d1-d2) = %.2e\n",
                    label[mode], rms[mode], vsum[mode], vdif[mode]);
      }

    const double narrow = rms[1] / rms[0];
    const double same = rms[2] / rms[0];
    std::printf("      narrowing factor %.4f (opposite), %.4f (same)\n",
                narrow, same);

    ok(narrow < 0.5, "opposite-sign dispersion narrows dL/dW by more than 2x");
    // The claim is that same-sign leaves the spectrum ALONE, not that it
    // broadens it. Anything outside a percent would mean the sum/difference
    // independence argument above is wrong.
    ok(std::fabs(same - 1.0) < 0.01,
       "same-sign dispersion leaves dL/dW essentially unchanged");

    // And the mechanism, not just the outcome: each sign must squeeze the
    // combination it actually selects on, and leave the other one alone.
    ok(vsum[1] < 0.5 * vsum[0], "opposite sign squeezes (d1+d2), which W sees");
    ok(std::fabs(vdif[1] / vdif[0] - 1.0) < 0.01,
       "opposite sign leaves (d1-d2) alone");
    ok(vdif[2] < 0.5 * vdif[0], "same sign squeezes (d1-d2), which W does not see");
    ok(std::fabs(vsum[2] / vsum[0] - 1.0) < 0.01,
       "same sign leaves (d1+d2) alone, so W cannot change");
  }

  // The commutator, measured rather than asserted.
  //
  // Both maps are Lie transformations exp(:G:) with quadratic generators, so
  // they live in sp(6,R). BCH makes the composition order matter exactly when
  // the Poisson bracket is non-zero:
  //
  //   F = t*zeta*px   G = D*delta*px
  //   {F,G} = (t*px)(D*px) - 0 = t*D*px^2
  //
  // and L*px^2/2 generates a drift of length L, so t*D*px^2 = (2tD)*px^2/2 is
  // a drift of length 2*t*D. The two orderings should therefore differ by
  // exactly that drift, and by nothing else: px, y, py, zeta and delta must
  // come out bit-identical, with the whole disagreement in x.
  //
  // The factor of 2 is the point of measuring. Reading "L*px^2/2 generates a
  // drift of length L" too quickly gives t*D; the generator is t*D*px^2 with
  // no half, so it is 2*t*D. This arm is what caught that.
  std::printf("\n=== the crab/dispersion commutator IS a drift, of length 2tD ===\n");
  {
    const double t = std::tan(0.007);   // ILC250: 14 mrad full crossing
    const double Dc = 1.246e-3;         // the monochromatizing dispersion
    const double L = 2.0 * t * Dc;      // predicted drift length

    std::mt19937 rng(31u);
    std::normal_distribution<double> g(0.0, 1.0);
    double worst_x = 0.0, worst_other = 0.0;

    for (int i = 0; i < 20000; i++)
      {
        PhaseSpace6D p0;
        p0.x = 1.0e-6 * g(rng);   p0.px = 1.0e-4 * g(rng);
        p0.y = 1.0e-8 * g(rng);   p0.py = 1.0e-5 * g(rng);
        p0.zeta = 3.0e-4 * g(rng); p0.delta = 1.24e-3 * g(rng);

        // dispersion then crab
        PhaseSpace6D a = p0;
        a.x += Dc * a.delta;  a.zeta += Dc * a.px;
        a.x += t * a.zeta;    a.delta -= t * a.px;

        // crab then dispersion
        PhaseSpace6D b = p0;
        b.x += t * b.zeta;    b.delta -= t * b.px;
        b.x += Dc * b.delta;  b.zeta += Dc * b.px;

        // The x disagreement must be the drift L*px, to rounding.
        //
        // Measured in ulps of the quantities actually being differenced, NOT
        // as a relative error against the prediction. a.x and b.x are around
        // 5e-6 and their difference is around 1e-9, so three and a half digits
        // are lost to cancellation before the comparison starts; and a Gaussian
        // draw will sooner or later make px -- and with it the prediction --
        // almost zero, at which point dividing by it reports the rounding of
        // a.x rather than any disagreement about the drift.
        const double predicted = L * p0.px;
        const double got = a.x - b.x;
        // The largest INTERMEDIATE, not the largest output. x + D*delta +
        // t*zeta can cancel toward zero by chance, and when it does the
        // outputs go tiny while the rounding that produced them came from
        // terms around 5e-6. Scaling by the output then divides real rounding
        // by an accidentally small number and reports thousands of ulps.
        double mag = std::fabs(predicted);
        const double terms[3] = {std::fabs(p0.x), std::fabs(Dc * p0.delta),
                                 std::fabs(t * p0.zeta)};
        for (int k = 0; k < 3; k++) mag = std::fmax(mag, terms[k]);
        const double eps = 2.220446049250313e-16;
        const double ulps = std::fabs(got - predicted) / (eps * mag);
        if (ulps > worst_x) worst_x = ulps;

        // And nothing else may differ at all.
        const double others[5] = {a.px - b.px, a.y - b.y, a.py - b.py,
                                  a.zeta - b.zeta, a.delta - b.delta};
        for (int k = 0; k < 5; k++)
          if (std::fabs(others[k]) > worst_other) worst_other = std::fabs(others[k]);
      }

    std::printf("      predicted drift length 2tD = %.3f um\n", L * 1.0e6);
    std::printf("      worst error on the x difference: %.1f ulps of the\n"
                "                differenced magnitude\n", worst_x);
    std::printf("      worst disagreement in any other coordinate %.3e\n",
                worst_other);

    ok(worst_x < 100.0,
       "the two orderings differ by exactly a drift of length 2tD");
    ok(worst_other == 0.0, "and differ in nothing but x");

    // Scale check: against ILC250 beta*_x this is a negligible but real effect,
    // and it grows linearly with D -- which monochromatization exists to raise.
    const double beta_x = 0.013;
    const double dsigma = 0.5 * (L / beta_x) * (L / beta_x);
    std::printf("      relative beam-size effect at beta*_x = 13 mm: %.1e\n",
                dsigma);
    ok(dsigma < 1.0e-5, "and is negligible at the ILC250 working point");
  }

  std::printf("\n  %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
