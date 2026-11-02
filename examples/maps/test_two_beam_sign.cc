// The two-beam longitudinal sign (C5), and what it actually has to produce.
//
// FEEBL has no per-beam zeta: zpos_ is one shared lab coordinate, so each beam
// must DECLARE how its zeta relates to it. CANONICAL_BUNCH::from_feebl applies
// that sign, and INCOMING_MAP_CHAIN::apply must not apply it a second time --
// it used to, and because every earlier test used sign = +1 the squaring went
// unnoticed.
//
// What the correct lab-frame behaviour IS took measurement to establish, and an
// earlier version of this test asserted the opposite of the truth. Both facts
// below are now measured, not assumed:
//
//   * Given the SAME EXPLICIT tilt on both beams, the lab tilts come out
//     OPPOSITE, because zeta_2 = -zeta_1 at the same lab z. This is the C5
//     hazard: the sign is live and reverses the response.
//
//   * But cancelling a crossing angle needs both beams tilted the SAME way in
//     the lab. FEEBL applies the crossing angle as a per-slice position shear
//     with an identical formula for both beams (set_angle_particles), so both
//     bunches acquire the same lab tilt, and only a common counter-tilt undoes
//     it. So the DERIVED tilt is t_b = -sign_b * theta/2, whose lab effect is
//     deliberately sign-INdependent.
//
// Measured on the ILC deck at 30 mrad: the derived pair recovers 13.55x
// (5.0% -> 68.4% of head-on at n_z=16, rising to 99.1% at n_z=64), while the
// same explicit tilt on both beams gives 0.75x and the reversed pair 0.37x --
// i.e. both wrong choices are WORSE than not crabbing at all.
#include "incomingMapsCPP.h"
#include <cstdio>
#include <cmath>
#include <string>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

static IncomingMapConfig base(int sign)
{
  IncomingMapConfig c;
  c.crab_mode = CRAB_LINEAR_IP_TILT;
  c.crab_frequency_Hz = 400e6;
  c.full_crossing_angle_rad = 0.030;     // theta/2 = 15 mrad
  c.longitudinal_sign = sign;
  return c;
}

// Build a bunch spanning a range of LAB z, apply the chain, and return the
// induced tilt dx/dz measured against the LAB coordinate.
static double lab_tilt(int chain_sign, int bunch_sign, double explicit_tilt,
                       bool& applied)
{
  IncomingMapConfig c = base(chain_sign);
  if (explicit_tilt != -1.0) c.crab_target_tilt = explicit_tilt;

  INCOMING_MAP_CHAIN m;
  if (!m.configure(c)) { applied = false; return 0.0; }

  CANONICAL_BUNCH b;
  b.set_longitudinal(LongitudinalConvention(bunch_sign, 0.0));
  b.set_reference_energy(125.0);
  const int N = 21;
  b.resize(N);
  double zs[21];
  for (int i = 0; i < N; i++) {
    zs[i] = (i - 10) * 2.0e4;                       // nm, +/- 200 um
    b.from_feebl(i, 0.0f, 0.0f, (float)zs[i], 0.0f, 0.0f, 125.0f);
  }
  std::string why;
  applied = m.apply(b, why);
  if (!applied) return 0.0;

  double sxy = 0.0, sxx = 0.0;
  for (int i = 0; i < N; i++) {
    const double dz = zs[i] * 1e-9;                 // metres, lab
    sxy += dz * b.x(i); sxx += dz * dz;
  }
  return sxy / sxx;
}

int main()
{
  // ---- an EXPLICIT tilt: the sign is live, lab tilts are opposite --------
  {
    bool a1 = false, a2 = false;
    const double t1 = lab_tilt(+1, +1, 0.015, a1);
    const double t2 = lab_tilt(-1, -1, 0.015, a2);
    printf("       explicit t=+0.015:  beam 1 lab %+.6f   beam 2 lab %+.6f\n", t1, t2);
    check(a1 && a2, "the chain applies to both beams");
    check(t1 * t2 < 0.0, "an explicit tilt gives OPPOSITE lab tilts");
    check(std::fabs(t1 + t2) < 1e-12, "and they are equal and opposite");
    check(std::fabs(std::fabs(t1) - 0.015) < 1e-9, "of the requested magnitude");
  }

  // ---- the DERIVED tilt: lab tilts are equal, which is what cancels ------
  {
    bool a1 = false, a2 = false;
    const double t1 = lab_tilt(+1, +1, -1.0, a1);
    const double t2 = lab_tilt(-1, -1, -1.0, a2);
    printf("       derived:            beam 1 lab %+.6f   beam 2 lab %+.6f\n", t1, t2);
    check(a1 && a2, "the derived chain applies to both beams");
    check(std::fabs(t1 - t2) < 1e-12, "the derived tilt gives EQUAL lab tilts");
    check(std::fabs(t1 + 0.015) < 1e-9,
          "of -theta/2, the sign that cancels FEEBL's crossing-angle shear");
  }

  // ---- refusals ---------------------------------------------------------
  {
    bool applied = true;
    lab_tilt(+1, -1, -1.0, applied);
    check(!applied, "refuses a bunch whose declared sign contradicts the config");
  }
  {
    INCOMING_MAP_CHAIN m;
    m.configure(base(+1));
    CANONICAL_BUNCH b;                       // never declared
    b.set_reference_energy(125.0); b.resize(1);
    std::string why;
    check(!m.apply(b, why) && why.find("declared") != std::string::npos,
          "refuses a bunch with no declared convention at all");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT"
                           : "two-beam longitudinal sign is correct (C5)");
  return fails ? 1 : 0;
}
