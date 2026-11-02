// Analytic gate for Thomas-BMT precession.
//
// Verify the production rotation against analytic constant-field limits.
//
// Everything here is analytic and independent of the code's internal unit
// bookkeeping, because each property follows from ds/dt = omega x s alone:
//
//   * a rotation preserves |s|;
//   * rotations about a fixed axis compose additively, so N steps of dt/N must
//     equal one step of dt -- this is precisely what the old Euler update could
//     not do, and it is the sharpest statement of integration accuracy;
//   * a spin parallel to omega does not move;
//   * reversing the charge reverses the sense;
//   * a zero field does nothing.
//
// The final case does check the physics formula rather than only the
// integrator: for B perpendicular to the momentum the precession angle is
// (1 + gamma*a)*|B|*dt/E, with a from PHYSTOOLS::BformFunction, computed here
// independently of rotateBMT.
#include "particlesCPP.h"
#include "physicalTools.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

namespace {

// A spinning electron travelling essentially along +z.
class Probe : public PARTICLE_WITH_SPIN
{
 public:
  Probe(double sx, double sy, double sz, float energy_GeV,
        float vx = 0.0f, float vy = 0.0f, float ups = 0.0f)
  {
    TRIDVECTOR s; s.setComponents((float)sx, (float)sy, (float)sz);
    setParticle(0.0f, 0.0f, 0.0f, vx, vy, energy_GeV, &s);
    setUps(ups);
  }
  double norm() const
  {
    const TRIDVECTOR& s = getSpin();
    return std::sqrt(s.getComponent(0)*s.getComponent(0) +
                     s.getComponent(1)*s.getComponent(1) +
                     s.getComponent(2)*s.getComponent(2));
  }
  double comp(int i) const { return getSpin().getComponent(i); }
};

TRIDVECTOR vec(double x, double y, double z)
{
  TRIDVECTOR v; v.setComponents((float)x, (float)y, (float)z); return v;
}

const TRIDVECTOR ZERO = vec(0.0, 0.0, 0.0);

}  // namespace

int main()
{
  const float  E    = 125.0f;         // GeV
  const double By   = 1.0e-9;         // field unit as the routine defines it
  const float  dt   = 100.0f;         // nm
  const float  qs   = -1.0f;          // charge_sign

  // ---- 1. a rotation preserves the norm ---------------------------------
  {
    Probe p(1.0, 0.0, 0.0, E);
    const double n0 = p.norm();
    for (int i = 0; i < 4000; i++) p.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt);
    printf("       |s| after 4000 steps: %.15f\n", p.norm());
    check(std::fabs(p.norm() - n0) < 1e-12,
          "4000 precession steps leave |s| unchanged to 1e-12");
  }

  // ---- 2. rotations compose: N small steps == one big step --------------
  //
  // The old Euler update failed this badly, and in a direction that grew the
  // norm. For a genuine rotation about a fixed axis the two must agree to
  // rounding.
  {
    Probe one(1.0, 0.0, 0.0, E);
    one.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt * 100.0f);

    Probe many(1.0, 0.0, 0.0, E);
    for (int i = 0; i < 100; i++) many.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt);

    double worst = 0.0;
    for (int k = 0; k < 3; k++)
      worst = std::max(worst, std::fabs(one.comp(k) - many.comp(k)));
    printf("       worst component difference, 1 x 100dt vs 100 x dt: %.3e\n", worst);
    check(worst < 1e-12, "100 steps of dt equal one step of 100*dt");
  }

  // ---- 3. a spin along the precession axis does not move ----------------
  //
  // With v along z and B along y, B has no longitudinal projection, so omega
  // lies along y. A spin along y is then invariant.
  {
    Probe p(0.0, 1.0, 0.0, E);
    for (int i = 0; i < 500; i++) p.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt);
    check(std::fabs(p.comp(1) - 1.0) < 1e-12 &&
          std::fabs(p.comp(0)) < 1e-12 && std::fabs(p.comp(2)) < 1e-12,
          "a spin along the precession axis is invariant");
  }

  // ---- 4. reversing the charge reverses the sense ------------------------
  {
    Probe a(1.0, 0.0, 0.0, E), b(1.0, 0.0, 0.0, E);
    a.rotateBMT(ZERO, vec(0.0, By, 0.0), +1.0f, dt * 50.0f);
    b.rotateBMT(ZERO, vec(0.0, By, 0.0), -1.0f, dt * 50.0f);
    check(std::fabs(a.comp(0) - b.comp(0)) < 1e-12 &&
          std::fabs(a.comp(2) + b.comp(2)) < 1e-12 &&
          std::fabs(a.comp(2)) > 1e-9,
          "reversing charge_sign reverses the rotation sense");
  }

  // ---- 5. a zero field does nothing --------------------------------------
  {
    Probe p(0.6, 0.0, 0.8, E);
    // TRIDVECTOR::setComponents takes floats, so the stored value is float(0.6),
    // not the double 0.6. Compare against what was actually stored -- otherwise
    // this measures the storage precision rather than the routine.
    const double x0 = p.comp(0), z0 = p.comp(2);
    p.rotateBMT(ZERO, ZERO, qs, dt * 1000.0f);
    check(p.comp(0) == x0 && p.comp(2) == z0,
          "a zero field leaves the spin bit-for-bit alone");
  }

  // ---- 6. the precession ANGLE matches the BMT formula -------------------
  //
  // For B perpendicular to the momentum the routine's own documented omega is
  //     |omega| = (1 + gamma*a) * |B| * dt / E
  // with a = BformFunction(upsilon). Computed here from scratch and compared
  // with the angle rotateBMT actually produced.
  {
    const double gamma = (double)E / (double)EMASS;
    const double a     = PHYSTOOLS::BformFunction(0.0f);
    // Use the field as STORED: setComponents narrows to float, and float(1e-9)
    // carries 2.7e-8 of relative error -- far larger than anything this test is
    // trying to resolve. Reading it back keeps the comparison about the physics.
    const TRIDVECTOR B = vec(0.0, By, 0.0);
    const double Bstored = B.getComponent(1);
    const double want  = (1.0 + gamma * a) * Bstored * (double)(dt * 10.0f) / (double)E;

    Probe p(1.0, 0.0, 0.0, E);
    p.rotateBMT(ZERO, B, qs, dt * 10.0f);
    // started along x; the turned angle is atan2 of the z component against x
    const double got = std::atan2(std::fabs(p.comp(2)), p.comp(0));
    printf("       precession angle: analytic %.12e, code %.12e\n", want, got);
    const double rel = std::fabs(got - want) / want;
    check(rel < 1e-9, "the precession angle matches (1+gamma*a)|B|dt/E");
  }

  // ---- 7. the spin basis must never be NaN -------------------------------
  //
  // referenceSpin normalises the transverse Lorentz force to get e1. A particle
  // on the axis of a symmetric bunch, or any run with the field off, has
  // exactly zero transverse force, and normalising a zero vector gave
  // 0 * inf = NaN across the whole basis -- which then reached synrad_spin_flip
  // and produced NaN spins that no comparison downstream could catch.
  {
    struct Case { const char* label; TRIDVECTOR E, B; };
    const Case cases[3] = {
      { "zero field", ZERO,                ZERO },
      { "pure B",     ZERO,                vec(0.0, 1e-9, 0.0) },
      { "pure E",     vec(1e-9, 0.0, 0.0), ZERO },
    };
    bool all_ok = true;
    for (int i = 0; i < 3; i++)
      {
        TRIDVECTOR e1, e2, e3;
        PHYSTOOLS::referenceSpin(0.0, 0.0, e1, e2, e3, cases[i].E, cases[i].B, -1.0f);
        bool ok = true;
        for (int k = 0; k < 3; k++)
          ok = ok && std::isfinite(e1(k)) && std::isfinite(e2(k)) && std::isfinite(e3(k));
        // and it must be a genuine orthonormal frame, not merely finite
        const double n1 = std::sqrt(e1(0)*e1(0)+e1(1)*e1(1)+e1(2)*e1(2));
        const double n2 = std::sqrt(e2(0)*e2(0)+e2(1)*e2(1)+e2(2)*e2(2));
        const double d12 = e1(0)*e2(0)+e1(1)*e2(1)+e1(2)*e2(2);
        const double d13 = e1(0)*e3(0)+e1(1)*e3(1)+e1(2)*e3(2);
        ok = ok && std::fabs(n1-1.0) < 1e-12 && std::fabs(n2-1.0) < 1e-12
                && std::fabs(d12) < 1e-12 && std::fabs(d13) < 1e-12;
        printf("       %-11s |e1|=%.6f |e2|=%.6f e1.e2=%+.1e e1.e3=%+.1e\n",
               cases[i].label, n1, n2, d12, d13);
        all_ok = all_ok && ok;
      }
    check(all_ok, "the spin basis is finite and orthonormal, even at zero force");
  }

  // ---- 8. beam 2 travels the other way, and beta_z must say so ----------
  //
  // The routine used to hard-code beta_z = +1 for every particle. That is right
  // for beam 1 and wrong for beam 2. With E_z zeroed by the field
  // reconstruction, the beta x E term is proportional to beta_z in x and y, so
  // it FLIPS for beam 2; a pure electric field therefore precesses the two
  // beams in opposite senses. A pure B field along y with no transverse slope
  // is insensitive to it, which is why the analytic gate above is unchanged.
  {
    const TRIDVECTOR Ex = vec(1.0e-9, 0.0, 0.0);
    Probe fwd(0.0, 0.0, 1.0, E), bwd(0.0, 0.0, 1.0, E);
    fwd.rotateBMT(Ex, ZERO, qs, dt * 50.0f, +1.0f);
    bwd.rotateBMT(Ex, ZERO, qs, dt * 50.0f, -1.0f);
    printf("       pure E, beta_z=+1: sx=%+.6e   beta_z=-1: sx=%+.6e\n",
           fwd.comp(0), bwd.comp(0));
    check(std::fabs(fwd.comp(0)) > 1e-12,
          "a pure E field does precess the spin");
    check(std::fabs(fwd.comp(0) + bwd.comp(0)) < 1e-15 * (1.0 + std::fabs(fwd.comp(0))),
          "and reversing beta_z reverses that precession exactly");

    // a pure transverse B with no transverse slope is beta_z-insensitive
    Probe bf(1.0, 0.0, 0.0, E), bb(1.0, 0.0, 0.0, E);
    bf.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt * 50.0f, +1.0f);
    bb.rotateBMT(ZERO, vec(0.0, By, 0.0), qs, dt * 50.0f, -1.0f);
    check(std::fabs(bf.comp(2) - bb.comp(2)) < 1e-15,
          "while a pure transverse B is insensitive to beta_z");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT"
                           : "BMT precession is an exact rotation at the analytic angle");
  return fails ? 1 : 0;
}
