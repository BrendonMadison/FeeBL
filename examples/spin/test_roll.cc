// The transverse roll must carry the spin.
//
// PARTICLE_BEAM stores PARTICLE*, and PARTICLE::rotate was NOT virtual. So
// rotate_particles() -- which implements angle_phi -- dispatched statically to
// the base version, which rotates pos_ and (vx_,vy_) and never touches spin_.
// PARTICLE_WITH_SPIN overrode rotateBMT but not rotate, so nothing caught it.
//
// Why it went unnoticed: a roll is a rotation about z, and s_z is invariant
// under it, so the reported longitudinal polarization is untouched *directly*.
// The damage is downstream -- velocity and fields end up in the rolled frame
// while the spin is still in the unrolled one, and rotateBMT then combines
// them, so the precession is computed from mismatched frames.
//
// The invariant that actually matters is therefore not "the spin rotated" but
// "the spin and the velocity stayed in the same frame". That is what test 1
// asserts, and it is the one a future refactor is most likely to break.
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
  Probe(double sx, double sy, double sz, float vx, float vy,
        float x = 1000.0f, float y = 0.0f)
  {
    TRIDVECTOR s(sx, sy, sz);        // the double ctor; setComponents narrows
    setParticle(x, y, 0.0f, vx, vy, 125.0f, &s);
  }
  float X()  const { return pos_[0]; }
  float Y()  const { return pos_[1]; }
  float VX() const { return vx_; }
  float VY() const { return vy_; }
  double SX() const { return spin_(0); }
  double SY() const { return spin_(1); }
  double SZ() const { return spin_(2); }
  double norm() const
    { return std::sqrt(SX()*SX() + SY()*SY() + SZ()*SZ()); }
};

void roll(Probe& p, double theta)
{
  p.rotate((float)std::cos(theta), (float)std::sin(theta));
}

}  // namespace

int main()
{
  const double TH = 0.7;   // radians, a general angle: no accidental symmetry

  printf("=== the roll carries the spin ===\n");

  // 1. THE INVARIANT. A spin and a velocity that start at the same azimuth
  //    must still share an azimuth afterwards. This is frame consistency, and
  //    it is what rotateBMT depends on.
  {
    Probe p(1.0, 0.0, 0.0, 3.0e-4f, 0.0f);
    const double before = std::atan2(p.SY(), p.SX())
                        - std::atan2((double)p.VY(), (double)p.VX());
    roll(p, TH);
    const double after  = std::atan2(p.SY(), p.SX())
                        - std::atan2((double)p.VY(), (double)p.VX());
    check(std::fabs(after - before) < 1e-6,
          "spin and velocity stay in the same azimuthal frame");
  }

  // 2. NON-VACUITY. If the override is removed, the spin simply does not move
  //    and test 1 fails -- but so would a test that rotated nothing at all.
  //    Assert the spin genuinely turned by the requested angle.
  {
    Probe p(1.0, 0.0, 0.0, 0.0f, 0.0f);
    roll(p, TH);
    check(std::fabs(p.SX() - std::cos(TH)) < 1e-6 &&
          std::fabs(p.SY() + std::sin(TH)) < 1e-6,
          "a spin along +x turns to (cos th, -sin th, 0)");
    check(std::fabs(p.SX() - 1.0) > 0.2,
          "and it really moved (guards against a no-op regression)");
  }

  // 3. The rotation is about z, so s_z cannot change. This is exactly why the
  //    bug was invisible in the reported longitudinal polarization.
  {
    Probe p(0.6, 0.0, 0.8, 0.0f, 0.0f);
    const double sz = p.SZ();
    roll(p, TH);
    check(std::fabs(p.SZ() - sz) < 1e-12,
          "s_z is invariant under a roll (why P looked fine)");
  }

  // 4. A rotation preserves length.
  {
    Probe p(0.36, -0.48, 0.8, 0.0f, 0.0f);
    const double n0 = p.norm();
    roll(p, TH);
    // PARTICLE::rotate takes FLOAT cos/sin, so cos^2+sin^2 differs from 1
    // by ~1e-7 and |s| by ~5e-8. That is the existing convention -- the
    // base class rotates position and velocity with the same floats -- so
    // the bound here is float epsilon, not double. Tightening it below
    // that would be testing the float literal, not the rotation.
    printf("       |s| drift across one roll: %.3e\n", std::fabs(p.norm() - n0));
    check(std::fabs(p.norm() - n0) < 1e-6, "the roll preserves |s| to float epsilon");
  }

  // 5. Rotations about a fixed axis compose additively.
  {
    Probe a(0.36, -0.48, 0.8, 0.0f, 0.0f);
    Probe b(0.36, -0.48, 0.8, 0.0f, 0.0f);
    roll(a, TH);
    roll(b, TH * 0.5); roll(b, TH * 0.5);
    check(std::fabs(a.SX() - b.SX()) < 1e-6 &&
          std::fabs(a.SY() - b.SY()) < 1e-6,
          "two half rolls equal one whole roll");
  }

  // 6. A spin along the rotation axis is fixed.
  {
    Probe p(0.0, 0.0, 1.0, 0.0f, 0.0f);
    roll(p, TH);
    check(std::fabs(p.SX()) < 1e-12 && std::fabs(p.SY()) < 1e-12 &&
          std::fabs(p.SZ() - 1.0) < 1e-12,
          "a spin along z is untouched by any roll");
  }

  // 7. The base class still does its job: position and velocity rotate the
  //    same way as the spin, not merely somewhere.
  {
    Probe p(1.0, 0.0, 0.0, 1.0f, 0.0f, 1.0f, 0.0f);
    roll(p, TH);
    check(std::fabs((double)p.X() - std::cos(TH)) < 1e-6 &&
          std::fabs((double)p.Y() + std::sin(TH)) < 1e-6,
          "position rotates by the same angle as the spin");
    check(std::fabs((double)p.VX() - std::cos(TH)) < 1e-6 &&
          std::fabs((double)p.VY() + std::sin(TH)) < 1e-6,
          "velocity rotates by the same angle as the spin");
  }

  // 8. A zero roll is exactly the identity.
  {
    Probe p(0.36, -0.48, 0.8, 0.0f, 0.0f);
    roll(p, 0.0);
    check(p.SX() == 0.36 && p.SY() == -0.48 && p.SZ() == 0.8,
          "a zero roll leaves the spin bit-for-bit alone");
  }

  printf("\n  %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
