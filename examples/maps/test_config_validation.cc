// Configuration validation for the incoming map chain.
//
// Reject configurations that would otherwise silently select a wrong map:
//   * enum-like keys were tested for "nonzero", so crab_mode=3 counted as
//     active, matched no branch, and became a logged no-op;
//   * any nonzero waist_mode enabled the ideal map;
//   * only crab_plane==1 was vertical, so crab_plane=2 tilted the wrong plane;
//   * NaN passes every ordinary comparison, including the symplectic gate;
//   * the RF linearizer solves for zero phase, so a nonzero crab_phase
//     delivered cos(phase) times the requested slope.
//
// Every one of those produces a plausible number rather than an error, which is
// the failure mode this whole stage exists to prevent.
#include "incomingMapsCPP.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <limits>

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

// Measure the lab-frame slope the chain actually delivers near zeta = 0.
static double delivered_slope(const INCOMING_MAP_CHAIN& m)
{
  const double dz = 1.0e-5;
  return (m.kick_at(dz) - m.kick_at(-dz)) / (2.0 * dz);
}

int main()
{
  const double NaN = std::numeric_limits<double>::quiet_NaN();

  // ---- enum-like keys ---------------------------------------------------
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base(); c.crab_mode = 3;
    check(!m.configure(c) && m.error().find("crab_mode") != std::string::npos,
          "crab_mode=3 is refused, not silently inert");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
    c.waist_mode = 2;
    check(!m.configure(c) && m.error().find("waist_mode") != std::string::npos,
          "waist_mode=2 is refused");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base(); c.crab_plane = 2;
    check(!m.configure(c) && m.error().find("crab_plane") != std::string::npos,
          "crab_plane=2 is refused, not silently horizontal");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base(); c.crab_plane = 1;
    check(m.configure(c), "crab_plane=1 (vertical) is still accepted");
  }

  // ---- non-finite inputs -------------------------------------------------
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base(); c.crab_frequency_Hz = NaN;
    check(!m.configure(c) && m.error().find("finite") != std::string::npos,
          "a NaN frequency is refused");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base(); c.crab_target_tilt = NaN;
    check(!m.configure(c) && m.error().find("finite") != std::string::npos,
          "a NaN target tilt is refused");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
    c.full_crossing_angle_rad = std::numeric_limits<double>::infinity();
    check(!m.configure(c) && m.error().find("finite") != std::string::npos,
          "an infinite crossing angle is refused");
  }

  // ---- the RF phase must not quietly rescale the requested slope --------
  //
  // K(z) = A1 sin(kz+phi) + An sin(nkz+phi) has K'(0) = cos(phi) * slope, so
  // before the correction a phase of 1 rad delivered only 54% of the tilt asked
  // for. The amplitudes are now solved for slope/cos(phi).
  {
    const double want = 0.015;
    double at_zero = 0.0;
    for (int i = 0; i < 4; i++)
      {
        const double phi = 0.0 + 0.4 * i;          // 0, 0.4, 0.8, 1.2 rad
        INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
        c.crab_target_tilt = want;
        c.crab_phase_rad = phi;
        if (!m.configure(c)) { check(false, "phase sweep configures"); break; }
        const double got = delivered_slope(m);
        if (i == 0) at_zero = got;
        char label[96];
        snprintf(label, sizeof label,
                 "phase %.1f rad still delivers the requested slope", phi);
        check(std::fabs(got - want) / want < 1e-6, label);
      }
    printf("       slope at zero phase = %.6f (requested %.6f)\n", at_zero, want);
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
    c.crab_phase_rad = 1.55;                        // ~1.6 degrees off pi/2
    check(!m.configure(c) && m.error().find("pi/2") != std::string::npos,
          "a phase where the cavity delivers no slope is refused");
  }

  // ---- the COMPOSED chain, not only its parts ---------------------------
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
    c.waist_mode = WAIST_IDEAL_LOCAL;
    check(m.configure(c), "crab and waist configure together");
    double defect = 0.0;
    const bool okd = m.verify_symplectic(defect);
    printf("       worst defect over components AND the composed map = %.3e\n", defect);
    check(okd && std::isfinite(defect) && defect <= 1e-11,
          "the composed chain is symplectic, not just its parts");
  }
  {
    INCOMING_MAP_CHAIN m; IncomingMapConfig c = base();
    c.crab_mode = CRAB_LINEAR_IP_TILT; c.crab_plane = 1;
    c.waist_mode = WAIST_IDEAL_LOCAL;
    check(m.configure(c), "vertical IP tilt with waist configures");
    double defect = 0.0;
    check(m.verify_symplectic(defect) && defect <= 1e-11,
          "and that composition is symplectic too");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT"
                           : "configuration validation holds");
  return fails ? 1 : 0;
}
