// Verify incoming maps without beam-beam fields: round trips, the symplectic
// defect and rejection of invalid configurations.
#include "incomingMapsCPP.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

static IncomingMapConfig base()
{
  IncomingMapConfig c;
  c.crab_mode = CRAB_RF_HARMONICS;
  c.crab_frequency_Hz = 400e6;
  c.full_crossing_angle_rad = 0.030;
  c.longitudinal_sign = +1;
  return c;
}

int main()
{
  // ---- refusals: each of these would otherwise run and be quietly wrong ----
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.longitudinal_sign = 0;
    check(!m.configure(c) && m.error().find("longitudinal") != std::string::npos,
          "refuses without a declared longitudinal sign (C5)");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.include_panofsky_wenzel = 0;
    check(!m.configure(c) && m.error().find("canonical") != std::string::npos,
          "refuses crab_include_panofsky_wenzel=0 as non-canonical");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.crab_frequency_Hz = 0.0;
    check(!m.configure(c), "refuses a non-positive crab frequency");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.waist_mode = WAIST_IDEAL_LOCAL;
    c.full_crossing_angle_rad = 0.0;
    c.crab_mode = CRAB_NONE;
    check(!m.configure(c), "refuses crab waist with no crossing angle");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c;                    // all modes off
    check(m.configure(c) && !m.active(),
          "a fully-off configuration is valid and inert");
  }

  // ---- symplecticity, the Level 2 gate ----------------------------------
  {
    INCOMING_MAP_CHAIN m;
    check(m.configure(base()), "configures RF harmonics crab");
    double worst = 0.0;
    const bool ok = m.verify_symplectic(worst);
    printf("       worst symplectic defect %.3e (tolerance %.0e)\n", worst, 1e-11);
    check(ok, "crab map is symplectic within tolerance");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.crab_mode = CRAB_NONE;
    c.waist_mode = WAIST_IDEAL_LOCAL;
    check(m.configure(c), "configures ideal local crab waist");
    double worst = 0.0;
    const bool ok = m.verify_symplectic(worst);
    printf("       worst symplectic defect %.3e\n", worst);
    check(ok, "waist map is symplectic within tolerance");
  }
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.waist_mode = WAIST_IDEAL_LOCAL;
    check(m.configure(c), "configures crab AND waist together");
    double worst = 0.0;
    check(m.verify_symplectic(worst), "combined chain is symplectic");
  }

  // ---- the crab actually tilts the bunch, and by the right amount --------
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.crab_mode = CRAB_LINEAR_IP_TILT;
    m.configure(c);
    // dx/dzeta at the IP should approach theta/2 for small zeta.
    const double dz = 1.0e-4;
    const double slope = (m.kick_at(dz) - m.kick_at(-dz)) / (2.0 * dz);
    printf("       lab dx/dz at IP = %.6f  (-theta/2 = %.6f)\n", slope, -0.015);
    // kick_at takes a LAB z offset, so this is the LAB-frame tilt. With the
    // tilt derived from the crossing angle it is -theta/2 for BOTH beams: the
    // derivation carries the per-beam sign precisely so that the lab tilt comes
    // out common, which is what cancels FEEBL's crossing-angle shear.
    check(std::fabs(slope + 0.015) < 1e-6,
          "derived tilt gives a lab tilt of -theta/2");
  }

  // ---- the longitudinal sign reverses an EXPLICIT tilt ------------------
  // With an explicit tilt the sign reverses the lab response, which is the C5
  // hazard. With a DERIVED tilt it deliberately does not, because the
  // derivation cancels it -- so this has to be tested with an explicit value.
  {
    INCOMING_MAP_CHAIN p, m;
    IncomingMapConfig cp = base();
    cp.crab_mode = CRAB_LINEAR_IP_TILT; cp.crab_target_tilt = 0.015;
    IncomingMapConfig cm = cp;          cm.longitudinal_sign = -1;
    p.configure(cp); m.configure(cm);
    const double a = p.kick_at(1.0e-3), b = m.kick_at(1.0e-3);
    check(std::fabs(a + b) < 1e-15 && std::fabs(a) > 0.0,
          "longitudinal sign reverses an explicit tilt, as C5 warns");
  }

  // ---- and the DERIVED tilt is deliberately sign-INdependent in the lab --
  {
    INCOMING_MAP_CHAIN p, m;
    IncomingMapConfig cp = base(); cp.crab_mode = CRAB_LINEAR_IP_TILT;
    IncomingMapConfig cm = cp;     cm.longitudinal_sign = -1;
    p.configure(cp); m.configure(cm);
    const double a = p.kick_at(1.0e-3), b = m.kick_at(1.0e-3);
    check(std::fabs(a - b) < 1e-15 && std::fabs(a) > 0.0,
          "derived tilt gives BOTH beams the same lab tilt");
  }

  // ---- the CAVITY changes px and leaves x alone --------------------------
  // base() is CRAB_RF_HARMONICS. This is cavity behaviour and belongs to mode 2
  // only: mode 1 is an IP tilt, which moves x instead (tested below).
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    m.configure(c);

    CANONICAL_BUNCH b;
    b.resize(3);
    b.set_reference_energy(62.5);
    b.set_longitudinal(LongitudinalConvention(+1, 0.0));
    b.set(0, 1e-4, 0.0, 2e-6, 0.0,  5e-3, 0.0, 1.0);
    b.set(1, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0, 1.0);   // at the origin
    b.set(2, -1e-4,0.0, 1e-6, 0.0, -5e-3, 1e-3, 1.0);

    const double x0 = b.x(0), px0 = b.px(0);
    { std::string why; check(m.apply(b, why), "apply accepts a matching convention"); }
    check(std::fabs(b.x(0) - x0) < 1e-18, "a thin cavity leaves x unchanged");
    check(std::fabs(b.px(0) - px0) > 1e-9, "px is kicked away from the origin");
    check(std::fabs(b.px(1)) < 1e-18, "a particle at zeta=0 receives no kick");
    check(b.px(0) * b.px(2) < 0.0,
          "head and tail are kicked in opposite directions");
  }

  // ---- the IP TILT moves x, leaves px alone, and carries its PW companion --
  //
  // CRAB_LINEAR_IP_TILT imposes x -> x + t*zeta, delta -> delta - t*px at the
  // IP. That distinction is not cosmetic: a cavity kick applied at the IP
  // recovers NOTHING of the crossing-angle luminosity loss, because a momentum
  // kick only becomes a position tilt after transport FEEBL does not have.
  {
    INCOMING_MAP_CHAIN m;
    IncomingMapConfig c = base();
    c.crab_mode = CRAB_LINEAR_IP_TILT;
    c.crab_target_tilt = 0.015;          // dx/dzeta
    check(m.configure(c), "IP tilt configures");

    CANONICAL_BUNCH b;
    b.resize(3);
    b.set_reference_energy(62.5);
    b.set_longitudinal(LongitudinalConvention(+1, 0.0));
    b.set(0, 1e-4, 3e-5, 2e-6, 0.0,  5e-3, 0.0, 1.0);
    b.set(1, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 1.0);   // at zeta = 0
    b.set(2, -1e-4,3e-5, 1e-6, 0.0, -5e-3, 0.0, 1.0);

    const double x0 = b.x(0), px0 = b.px(0), d0 = b.delta(0), x2 = b.x(2);
    { std::string why; check(m.apply(b, why), "IP tilt applies"); }
    check(std::fabs(b.x(0) - (x0 + 0.015 * 5e-3)) < 1e-15,
          "x is displaced by exactly t*zeta");
    check(std::fabs(b.px(0) - px0) < 1e-18, "an IP tilt leaves px unchanged");
    check(std::fabs(b.delta(0) - (d0 - 0.015 * px0)) < 1e-18,
          "delta carries the Panofsky-Wenzel companion -t*px");
    check(std::fabs(b.x(1)) < 1e-18, "a particle at zeta=0 is not displaced");
    check((b.x(0) - x0) * (b.x(2) - x2) < 0.0,
          "head and tail are displaced in opposite directions");

    double defect = 0.0;
    check(m.verify_symplectic(defect) && defect < 1e-11,
          "the IP tilt is symplectic to the required tolerance");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT" : "all Stage D map checks passed");
  return fails ? 1 : 0;
}
