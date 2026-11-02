// The head-on -> lab transformation.
//
// FEEBL collides in the head-on frame, so every direction it reports is a
// head-on-frame direction. This is the transformation back, and these are the
// properties that make it the RIGHT transformation rather than a plausible one.
//
// The sharpest test is #3/#4: the intuitive "rotate by the crossing angle"
// gives the backward beam the WRONG SIGN, and nothing downstream would notice.
#include "headOnFrameCPP.h"
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
  const double PHI = 0.007;             // ILC250 half crossing angle
  const double E0  = 125.0;
  HEAD_ON_FRAME f(PHI);

  // 1. Round trip.
  {
    double E = 37.0, px = 0.13, py = -0.07, pz = -36.0;
    const double E1 = E, x1 = px, y1 = py, z1 = pz;
    f.to_lab(E, px, py, pz);
    f.to_headon(E, px, py, pz);
    check(std::fabs(E - E1) < 1e-10 && std::fabs(px - x1) < 1e-10 &&
          py == y1 && pz == z1,
          "to_lab then to_headon is the identity");
  }

  // 2. It is a Lorentz transformation, so the invariant mass is preserved.
  {
    double E = 37.0, px = 0.13, py = -0.07, pz = -36.0;
    const double m2a = E*E - px*px - py*py - pz*pz;
    f.to_lab(E, px, py, pz);
    const double m2b = E*E - px*px - py*py - pz*pz;
    check(std::fabs(m2b - m2a) < 1e-8 * (1.0 + std::fabs(m2a)),
          "the invariant mass is preserved");
  }

  // 3. THE POINT. In the lab BOTH beams carry +x momentum, so both must come
  //    out at the SAME sign of x-angle.
  double a1 = 0.0, a2 = 0.0;
  {
    double E = E0, px = 0.0, py = 0.0, pz = E0;
    f.to_lab(E, px, py, pz); a1 = px / E;
    double E2 = E0, px2 = 0.0, py2 = 0.0, pz2 = -E0;
    f.to_lab(E2, px2, py2, pz2); a2 = px2 / E2;
    check(a1 > 0.0 && a2 > 0.0, "both beams tilt the SAME way in the lab");
    check(std::fabs(a1 - std::sin(PHI)) < 1e-9 &&
          std::fabs(a2 - std::sin(PHI)) < 1e-9,
          "and both by exactly sin(phi)");
  }

  // 4. The rejected alternative, asserted so the reason stays in the suite: a
  //    rotation about y mirrors the backward beam.
  {
    const double c = std::cos(PHI), s = std::sin(PHI);
    const double r1 = (0.0 * c + ( E0) * s) / E0;   // beam 1, pz = +E
    const double r2 = (0.0 * c + (-E0) * s) / E0;   // beam 2, pz = -E
    check(r1 > 0.0 && r2 < 0.0,
          "a rotation about y would mirror beam 2 (so it is not used)");
    check(std::fabs(r1 - a1) < 1e-9,
          "rotation and boost agree for the FORWARD beam only");
  }

  // 5. py and pz are untouched: the boost is along x.
  {
    double E = 12.0, px = 0.5, py = 0.25, pz = -11.0;
    f.to_lab(E, px, py, pz);
    check(py == 0.25 && pz == -11.0, "py and pz are untouched");
  }

  // 6. No crossing angle must be EXACTLY the identity, or every head-on run
  //    would shift in its last bits and bitcheck would break.
  {
    HEAD_ON_FRAME none(0.0);
    double E = 37.0, px = 0.13, py = -0.07, pz = -36.0;
    none.to_lab(E, px, py, pz);
    check(E == 37.0 && px == 0.13 && py == -0.07 && pz == -36.0,
          "zero crossing angle is bit-for-bit the identity");
    check(!none.active(), "and it reports itself inactive");
  }

  // 7. A backward-going SECONDARY -- the pair background case -- also picks up
  //    the same +phi, which is the whole reason this matters for cone counts.
  {
    double E = 0.5, px = 0.0, py = 0.0, pz = -0.5;   // 500 MeV, straight back
    f.to_lab(E, px, py, pz);
    check(px / E > 0.0, "a backward secondary also tilts +x in the lab");
  }

  // 8. A crossing in the VERTICAL plane. Same construction, boost along y.
  //    Without this the generalisation is untested surface.
  {
    HEAD_ON_FRAME fy(PHI, 0.0, 1.0);
    double E = E0, px = 0.0, py = 0.0, pz = E0;
    fy.to_lab(E, px, py, pz);
    const double ay1 = py / E;
    double E2 = E0, px2 = 0.0, py2 = 0.0, pz2 = -E0;
    fy.to_lab(E2, px2, py2, pz2);
    const double ay2 = py2 / E2;
    check(ay1 > 0.0 && ay2 > 0.0 &&
          std::fabs(ay1 - std::sin(PHI)) < 1e-9 &&
          std::fabs(ay2 - std::sin(PHI)) < 1e-9,
          "a vertical crossing tilts both beams +y by sin(phi)");
    check(px == 0.0 && px2 == 0.0, "and leaves px untouched");
  }

  // 9. An oblique plane: the tilt must land along the declared direction and
  //    have the right magnitude, not merely be nonzero.
  {
    const double rt = 1.0 / std::sqrt(2.0);
    HEAD_ON_FRAME fd(PHI, 1.0, 1.0);
    double E = E0, px = 0.0, py = 0.0, pz = E0;
    fd.to_lab(E, px, py, pz);
    check(std::fabs(px / E - std::sin(PHI) * rt) < 1e-9 &&
          std::fabs(py / E - std::sin(PHI) * rt) < 1e-9,
          "an oblique crossing tilts along the declared direction");
  }

  // 10. A degenerate direction must disable the transform, not emit NaNs.
  {
    HEAD_ON_FRAME bad(PHI, 0.0, 0.0);
    double E = 37.0, px = 0.13, py = -0.07, pz = -36.0;
    bad.to_lab(E, px, py, pz);
    check(!bad.active() && E == 37.0 && px == 0.13,
          "a degenerate crossing direction disables the transform");
  }

  printf("\n  %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
