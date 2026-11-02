// Stage E gate, part 1: the geometry and the analytic reference. Part 2 is the
// rigid FEEBL comparison, which needs actual runs.
#include "collisionGeometryCPP.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

int main()
{
  // ---- the convention must be declared, not guessed --------------------
  {
    COLLISION_GEOMETRY g;
    check(!g.configure(0, 0.030, 0) && !g.declared(),
          "refuses an undeclared angle convention");
    check(g.error().find("factor of two") != std::string::npos,
          "and says why: half and full differ by two");
  }

  // ---- half vs full actually differ by two -----------------------------
  {
    COLLISION_GEOMETRY h, f;
    h.configure(ANGLE_HALF, 0.015, 0);
    f.configure(ANGLE_FULL, 0.030, 0);
    check(std::fabs(h.full_crossing_angle() - f.full_crossing_angle()) < 1e-15,
          "half 15 mrad and full 30 mrad describe the same geometry");
    check(std::fabs(h.half_crossing_angle() - 0.015) < 1e-15,
          "half angle recovered consistently");
  }

  // ---- the frames counter-propagate ------------------------------------
  {
    COLLISION_GEOMETRY g;
    g.configure(ANGLE_FULL, 0.030, 0);
    check(g.frame(1).longitudinal_sign == +1 && g.frame(2).longitudinal_sign == -1,
          "the two beams get opposite longitudinal signs");
    check(std::fabs(g.frame(1).half_angle_rad + g.frame(2).half_angle_rad) < 1e-18,
          "and opposite tilts");
    check(g.frame(1).declared && g.frame(2).declared,
          "both frames are marked declared");
    check(g.frame(1).longitudinal().declared,
          "a frame yields a usable longitudinal convention for the maps");
  }

  // ---- Piwinski and the geometric reduction ----------------------------
  {
    // FCC-ee Z: sigma_x 8.4 um, sigma_z 14.5 mm, 30 mrad full.
    const double sx = 8.4023e-6, sz = 14.5e-3, th = 0.030;
    const double phi = piwinski_angle(sz, sx, th);
    const double rg  = 1.0 / std::sqrt(1.0 + phi*phi);
    printf("       FCC-ee Z: Piwinski = %.3f, R_geom = %.5f\n", phi, rg);
    check(phi > 20.0 && phi < 30.0, "FCC-ee Z Piwinski angle is ~26");
    check(rg < 0.05, "the geometric reduction is severe at that Piwinski angle");

    // Head-on must be exactly unity, not merely close.
    check(std::fabs(piwinski_angle(sz, sx, 0.0)) < 1e-18,
          "zero crossing angle gives exactly zero Piwinski angle");
    RigidLuminosity h = rigid_luminosity(1e10, 1e10, 1.0, sx, 1e-8, sz, 1.0, 0.0);
    check(std::fabs(h.r_geom - 1.0) < 1e-15,
          "and exactly unity geometric reduction");
  }

  // ---- hourglass -------------------------------------------------------
  {
    // sigma_z well below beta*: no reduction.
    const double a = hourglass_factor(1e-4, 1.0);
    check(a > 0.9999, "hourglass is negligible when sigma_z << beta*");
    // sigma_z comparable to beta*: real reduction, and bounded.
    const double b = hourglass_factor(14.5e-3, 0.8e-3);
    printf("       hourglass at sigma_z=14.5 mm, beta*=0.8 mm: %.4f\n", b);
    check(b > 0.0 && b < 1.0, "hourglass reduces when sigma_z exceeds beta*");
    check(b < a, "and reduces more as the ratio worsens");
  }

  // ---- the assembled luminosity ----------------------------------------
  {
    // ILC 250 head-on, from the benchmark deck.
    const double N = 2e10, sx = 515.48e-9, sy = 7.65916e-9;
    RigidLuminosity r = rigid_luminosity(N, N, 1.0, sx, sy, 300e-6, 0.41e-3, 0.0);
    printf("       ILC 250 head-on geometric L = %.4e m^-2 per crossing\n",
           r.l_headon);
    // N^2/(4 pi sx sy) = 8.06e33 for these numbers.
    check(r.l_headon > 7.5e33 && r.l_headon < 8.5e33,
          "ILC 250 head-on geometric L is 8.1e33");
    // Cross-check against the code itself: FEEBL reports lumi_fine = 1.615e34
    // for this deck, so the ratio is the pinch enhancement at D_y = 34.5. A
    // geometric formula coming out ABOVE the simulated luminosity would be the
    // real warning -- rigid overlap is a floor, not a target.
    const double enhancement = 1.615e34 / r.l_headon;
    printf("       FEEBL lumi_fine / geometric = %.2f (pinch enhancement)\n",
           enhancement);
    check(enhancement > 1.5 && enhancement < 2.5,
          "implied pinch enhancement is sensible for D_y = 34.5");
    check(std::fabs(r.l_total - r.l_headon * r.hourglass) < 1e20,
          "head-on total is the head-on value times hourglass only");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT" : "all Stage E geometry checks passed");
  return fails ? 1 : 0;
}
