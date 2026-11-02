// The spin precession must respond to the relative beam charge.
//
// charge_sign is the RELATIVE sign of the two beam charges (switchesCPP.h:77),
// and it reaches the physics twice:
//
//   gridCPP.cc:149    factor = -charge_sign * 2*RE/deltaz * EMASSeV
//                     -> dist_init -> Green's function -> potential -> E,B
//   particlesCPP.cc   omega  = -(arg) * (ga1*Btrans + a1*Blong - ce1*g*betaE)
//
// The call site used to pass charge_sign as that argument, so the product was
// cs^2 = +1 and the precession did not depend on the relative beam charge --
// while the orbit did. Measured on one deck varying only charge_sign: e+e-
// pinches to 1.29387e34 and e-e- defocuses to 1.79356e33 against a force-free
// 7.16844e33, yet the spin rotation was bit-identical between the two.
//
// The call site now passes the constant -1.0f. Note it is -1 and NOT +1:
// rotateBMT applies its own leading minus, so +1 would invert the e+e- case
// (measured: beam-1 depolarization moved from -0.0096 to -0.0111). See
// CHARGE_SIGN_DERIVATION.md.
//
// This gates both halves of that: the production composition must now flip with
// charge_sign, and rotateBMT must still honour its own argument in isolation.
#include "particlesCPP.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

namespace {
class Probe : public PARTICLE_WITH_SPIN
{
 public:
  Probe(double sx, double sy, double sz, float energy_GeV)
  {
    TRIDVECTOR s(sx, sy, sz);
    setParticle(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, energy_GeV, &s);
  }
  double SX() const { return spin_(0); }
};

// One step through the PRODUCTION composition: the field carries -cs, and the
// call site passes the fixed multiplier.
double production_step(float cs, double Bgeom, float multiplier)
{
  TRIDVECTOR E(0.0, 0.0, 0.0);
  TRIDVECTOR B(0.0, -(double)cs * Bgeom, 0.0);   // gridCPP.cc:149
  Probe p(0.0, 0.0, 1.0, 125.0f);
  p.rotateBMT(E, B, multiplier, 1.0e3f, 1.0f);
  return p.SX();
}
}  // namespace

int main()
{
  const double G = 1.0e-12;          // a sign-free geometric field

  printf("=== the production composition must follow the beam charge ===\n");

  const double now_m = production_step(-1.0f, G, -1.0f);   // e+e-, as shipped
  const double now_p = production_step(+1.0f, G, -1.0f);   // e-e-, as shipped
  printf("    multiplier -1 (shipped): cs=-1 -> %+.6e   cs=+1 -> %+.6e\n",
         now_m, now_p);
  check(std::fabs(now_m + now_p) < 1e-18 && std::fabs(now_m) > 0.0,
        "shipped: the precession REVERSES with charge_sign");

  // The defect this replaced.
  const double old_m = production_step(-1.0f, G, -1.0f * -1.0f);
  const double old_p = production_step(+1.0f, G, +1.0f * -1.0f);
  // reproduce the old call literally: multiplier == charge_sign
  const double was_m = production_step(-1.0f, G, -1.0f);
  const double was_p = production_step(+1.0f, G, +1.0f);
  (void)old_m; (void)old_p;
  printf("    multiplier = charge_sign (old): cs=-1 -> %+.6e   cs=+1 -> %+.6e\n",
         was_m, was_p);
  check(std::fabs(was_m - was_p) < 1e-18,
        "old: it did NOT (this is the defect, reproduced)");

  // e+e- must be untouched by the change: the shipped multiplier reproduces
  // exactly what the old call produced at cs = -1.
  check(std::fabs(now_m - was_m) < 1e-18,
        "and e+e- is bit-identical before and after the fix");

  // +1 was the wrong constant, and this records why.
  const double plus_one = production_step(-1.0f, G, +1.0f);
  check(std::fabs(plus_one + now_m) < 1e-18 && std::fabs(plus_one) > 0.0,
        "passing +1 would INVERT e+e- (why the constant is -1)");

  printf("\n=== rotateBMT still honours its own argument in isolation ===\n");
  {
    TRIDVECTOR E(0.0, 0.0, 0.0), B(0.0, G, 0.0);
    Probe a(0.0, 0.0, 1.0, 125.0f), b(0.0, 0.0, 1.0, 125.0f);
    a.rotateBMT(E, B, -1.0f, 1.0e3f, 1.0f);
    b.rotateBMT(E, B, +1.0f, 1.0e3f, 1.0f);
    check(std::fabs(a.SX() + b.SX()) < 1e-18 && std::fabs(a.SX()) > 0.0,
          "an unfactored field still reverses with the argument");
  }

  printf("\n  %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
