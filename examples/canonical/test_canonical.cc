// Stage C gate: the canonical tier must round-trip FEEBL state exactly, must
// refuse to invent a longitudinal convention, and must reject bad weights.
//
// The off-momentum cases matter most: px = x'(1+delta) is the identity at
// delta = 0, so a test built only from on-momentum particles would pass even
// with px and x' conflated (doc/CONVENTIONS.md C2).
#include "canonicalBunchCPP.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

int main()
{
  const double Eref = 125.0;

  // --- refuses to convert without a declared convention (C5) -----------
  {
    CANONICAL_BUNCH b;
    b.resize(1);
    b.set_reference_energy(Eref);
    float x, y, z, vx, vy, e;
    check(!b.from_feebl(0, 1.0f, 2.0f, 3.0f, 1e-5f, 2e-5f, 125.0f),
          "from_feebl refuses without a longitudinal convention");
    check(!b.to_feebl(0, x, y, z, vx, vy, e),
          "to_feebl refuses without a longitudinal convention");
  }

  // --- refuses without a reference energy ------------------------------
  {
    CANONICAL_BUNCH b;
    b.resize(1);
    b.set_longitudinal(LongitudinalConvention(+1, 0.0));
    check(!b.from_feebl(0, 1.0f, 2.0f, 3.0f, 1e-5f, 2e-5f, 125.0f),
          "from_feebl refuses without a reference energy");
  }

  // --- round trip, on and off momentum ---------------------------------
  {
    CANONICAL_BUNCH b;
    b.resize(4);
    b.set_reference_energy(Eref);
    b.set_longitudinal(LongitudinalConvention(+1, 0.0));

    const float X[4]  = { 1.0e3f, -2.5e3f,  0.0f,  7.5e2f };   // nm
    const float Y[4]  = { -50.0f,  12.5f,  0.0f,  -3.0f  };
    const float Z[4]  = { 1.0e6f, -2.0e6f, 0.0f,  5.0e5f };
    const float VX[4] = { 1.0e-5f, -2e-5f, 0.0f,  3e-6f  };   // rad
    const float VY[4] = { 2.0e-6f,  1e-6f, 0.0f, -4e-6f  };
    const float E[4]  = { 125.0f, 125.125f, 124.375f, 125.625f }; // GeV

    bool all = true, exact = true;
    for (std::size_t i = 0; i < 4; i++)
      all = all && b.from_feebl(i, X[i], Y[i], Z[i], VX[i], VY[i], E[i]);
    check(all, "from_feebl succeeds once conventions are declared");

    for (std::size_t i = 0; i < 4; i++)
      {
        float x = 0, y = 0, z = 0, vx = 0, vy = 0, e = 0;
        if (!b.to_feebl(i, x, y, z, vx, vy, e))
          { exact = false; continue; }
        const double tol = 1e-6;   // float round-trip through double
        if (std::fabs(x - X[i]) > tol * std::fmax(1.0f, std::fabs(X[i]))) exact = false;
        if (std::fabs(y - Y[i]) > tol * std::fmax(1.0f, std::fabs(Y[i]))) exact = false;
        if (std::fabs(z - Z[i]) > tol * std::fmax(1.0f, std::fabs(Z[i]))) exact = false;
        if (std::fabs(vx - VX[i]) > tol * std::fmax(1e-9f, std::fabs(VX[i]))) exact = false;
        if (std::fabs(vy - VY[i]) > tol * std::fmax(1e-9f, std::fabs(VY[i]))) exact = false;
        if (std::fabs(e - E[i]) > tol * std::fmax(1.0f, std::fabs(E[i]))) exact = false;
      }
    check(exact, "round trip is exact on and off momentum");

    // px must differ from x' off momentum -- the C2 relation
    bool separated = false;
    for (std::size_t i = 0; i < 4; i++)
      if (std::fabs(b.delta(i)) > 1e-6 && VX[i] != 0.0f)
        if (std::fabs(b.px(i) - VX[i]) > 1e-12) separated = true;
    check(separated, "px differs from x' off momentum (C2 applied, not skipped)");
  }

  // --- the negative-sign convention actually flips zeta ----------------
  {
    CANONICAL_BUNCH p, m;
    p.resize(1); m.resize(1);
    p.set_reference_energy(Eref); m.set_reference_energy(Eref);
    p.set_longitudinal(LongitudinalConvention(+1, 0.0));
    m.set_longitudinal(LongitudinalConvention(-1, 0.0));
    p.from_feebl(0, 0.0f, 0.0f, 1.0e6f, 0.0f, 0.0f, 125.0f);
    m.from_feebl(0, 0.0f, 0.0f, 1.0e6f, 0.0f, 0.0f, 125.0f);
    check(std::fabs(p.zeta(0) + m.zeta(0)) < 1e-15 && p.zeta(0) != 0.0,
          "longitudinal sign flips zeta as declared");
  }

  // --- weight validation ------------------------------------------------
  {
    CANONICAL_BUNCH b;
    b.resize(3);
    b.set_reference_energy(Eref);
    b.set_longitudinal(LongitudinalConvention(+1, 0.0));
    b.set(0, 0,0,0,0,0,0,  1.0);
    b.set(1, 0,0,0,0,0,0, -1.0);            // negative weight
    b.set(2, 0,0,0,0,0,0,  0.0);            // zero weight
    CANONICAL_BUNCH::Validation v = b.validate();
    check(v.bad_weight == 2 && !v.ok(), "rejects non-positive weights");
    check(!b.uniform_weight(), "non-uniform weights are flagged");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT" : "all canonical-tier checks passed");
  return fails ? 1 : 0;
}
