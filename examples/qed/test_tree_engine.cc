// Unit tests of qedbg_tree.h and dump of random 2->4 points for the FeynCalc oracle.
// CMake target feebl_qed_tree_oracle uses the integrated source headers.
//   ./a.out points.txt        (writes points + C++ |A|^2 for benchmarks/qed_validation/examples/numerical/compare_oracle.wl)
#define QEDBG_USE_QUADMATH
#include "qedbg_tree.h"

#include <cstdio>
#include <random>
#include <string>

using namespace qedbg;

static int failures = 0;
static void check(bool ok, const char* what, double val) {
  std::printf("%-62s %s  (%.3e)\n", what, ok ? "PASS" : "FAIL", val);
  if (!ok) ++failures;
}

static const double ME = 0.51099895000e-3, MMU = 0.1056583755;

// Cartesian -> light-cone (fine for generic, non-collinear test points)
static LCq lc_cart(Q E, Q px, Q py, Q pz, Q mass) {
  // use the larger of E+pz, E-pz and the mass shell for the other one
  if (pz >= 0) return onshell_plus(E + pz, px, py, mass);
  return onshell_minus(E - pz, px, py, mass);
}

struct Cart { Q E, x, y, z; };
static Cart boost(const Cart& p, Q bx, Q by, Q bz) {
  const Q b2 = bx * bx + by * by + bz * bz;
  if (b2 == 0) return p;
  const Q g = 1 / sqrtq(1 - b2), bp = bx * p.x + by * p.y + bz * p.z, f = (g - 1) * bp / b2 + g * p.E;
  return {g * (p.E + bp), p.x + f * bx, p.y + f * by, p.z + f * bz};
}
static Q twobody(Q M, Q m1, Q m2) { return sqrtq((M * M - (m1 + m2) * (m1 + m2)) * (M * M - (m1 - m2) * (m1 - m2))) / (2 * M); }
static void decay(const Cart& P, Q m1, Q m2, Q cth, Q phi, Cart& k1, Cart& k2) {
  const Q M = sqrtq(P.E * P.E - P.x * P.x - P.y * P.y - P.z * P.z), q = twobody(M, m1, m2), sth = sqrtq(1 - cth * cth);
  Cart a{sqrtq(q * q + m1 * m1), q * sth * cosq(phi), q * sth * sinq(phi), q * cth};
  Cart b{sqrtq(q * q + m2 * m2), -a.x, -a.y, -a.z};
  k1 = boost(a, P.x / P.E, P.y / P.E, P.z / P.E);
  k2 = boost(b, P.x / P.E, P.y / P.E, P.z / P.E);
}

// e+(p1) e-(p2) -> e+(p3) e-(p4) l+(p5) l-(p6), e- beam along +z
static std::vector<Fermion> random_point(std::mt19937_64& rng, Q sqrts, double ml) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const Q me = ME, m = ml, E = sqrts / 2, pz = sqrtq(E * E - me * me);
  const Q M1 = (me + 2 * m) + (sqrts - me - (me + 2 * m)) * (Q)U(rng);
  const Q M2 = 2 * m + (M1 - me - 2 * m) * (Q)U(rng);
  Cart P{sqrts, 0, 0, 0}, k3, r1, k4, r2, k5, k6;
  decay(P, me, M1, 2 * (Q)U(rng) - 1, 2 * M_PIq * (Q)U(rng), k3, r1);
  decay(r1, me, M2, 2 * (Q)U(rng) - 1, 2 * M_PIq * (Q)U(rng), k4, r2);
  decay(r2, m, m, 2 * (Q)U(rng) - 1, 2 * M_PIq * (Q)U(rng), k5, k6);
  std::vector<Fermion> f(6);
  f[0] = {onshell_minus(E + pz, 0, 0, me), ME, 0, true, true};    // e+ in (along -z)
  f[1] = {onshell_plus(E + pz, 0, 0, me), ME, 0, true, false};    // e- in (along +z)
  f[2] = {lc_cart(k3.E, k3.x, k3.y, k3.z, me), ME, 0, false, true};
  f[3] = {lc_cart(k4.E, k4.x, k4.y, k4.z, me), ME, 0, false, false};
  const int fl = (ml == ME) ? 0 : 1;
  f[4] = {lc_cart(k5.E, k5.x, k5.y, k5.z, m), ml, fl, false, true};
  f[5] = {lc_cart(k6.E, k6.x, k6.y, k6.z, m), ml, fl, false, false};
  return f;
}
// same-sign beams: e-(p1) e-(p2) -> e-(p3) e-(p4) l+(p5) l-(p6)
static std::vector<Fermion> random_point_same_sign(std::mt19937_64& rng, Q sqrts, double ml) {
  auto f = random_point(rng, sqrts, ml);
  f[0].anti = false;
  f[2].anti = false;
  return f;
}

static double m2(const std::vector<Fermion>& f, bool shift = true) {
  std::array<C, 64> a;
  amplitude_3lines(f, a, shift);
  return sum_sq(a);
}

// exact light-front transformations (Lorentz): z boost, transverse Galilean boost, z rotation
static LCq lf_transform(const LCq& a, Q eta, Q vx, Q vy, Q phi) {
  LCq b{eta * a.p, a.m / eta, a.x, a.y};
  LCq c{b.p, b.m + 2 * (vx * b.x + vy * b.y) + (vx * vx + vy * vy) * b.p, b.x + vx * b.p, b.y + vy * b.p};
  return {c.p, c.m, cosq(phi) * c.x - sinq(phi) * c.y, sinq(phi) * c.x + cosq(phi) * c.y};
}

int main(int argc, char** argv) {
  std::mt19937_64 rng(20261029);
  // ---- spinor identities
  {
    LCq p = onshell_plus(37.3, 1.7, -2.2, ME);
    double worst = 0;
    for (int h = 0; h < 2; ++h) {
      Sp u = u_spinor(p, ME, h), v = v_spinor(p, ME, h);
      Sp du = slash(cv(p), u), dv = slash(cv(p), v);
      for (int i = 0; i < 4; ++i) {
        worst = std::max(worst, std::abs(du.c[i] - ME * u.c[i]) / std::sqrt((double)p.p));
        worst = std::max(worst, std::abs(dv.c[i] + ME * v.c[i]) / std::sqrt((double)p.p));
      }
    }
    check(worst < 1e-13, "Dirac equation (pslash-m)u = 0, (pslash+m)v = 0", worst);
    // completeness: Sum_h (psibar u_h)(ubar_h chi) = psibar (pslash + m) chi for random psi, chi
    Sp psi{{C(0.3, 0.1), C(-0.7, 0.2), C(0.5, -0.4), C(0.1, 0.9)}}, chi{{C(-0.2, 0.6), C(0.8, 0.3), C(-0.1, -0.5), C(0.4, 0.2)}};
    C lhs_u(0, 0), lhs_v(0, 0);
    for (int h = 0; h < 2; ++h) {
      Sp u = u_spinor(p, ME, h), v = v_spinor(p, ME, h);
      lhs_u += bar_dot(psi, u) * bar_dot(u, chi);
      lhs_v += bar_dot(psi, v) * bar_dot(v, chi);
    }
    const C rhs_u = bar_dot(psi, slash(cv(p), chi) + C(ME, 0) * chi), rhs_v = bar_dot(psi, slash(cv(p), chi) + C(-ME, 0) * chi);
    check(std::abs(lhs_u - rhs_u) / std::abs(rhs_u) < 1e-13, "completeness Sum u ubar = pslash + m", std::abs(lhs_u - rhs_u) / std::abs(rhs_u));
    check(std::abs(lhs_v - rhs_v) / std::abs(rhs_v) < 1e-13, "completeness Sum v vbar = pslash - m", std::abs(lhs_v - rhs_v) / std::abs(rhs_v));
    // J.a = psibar aslash chi
    CV a{C(1.3, 0.2), C(-0.4, 0.7), C(0.9, -1.1), C(0.2, 0.5)};
    const C x1 = dot(current(psi, chi), a), x2 = bar_dot(psi, slash(a, chi));
    check(std::abs(x1 - x2) / std::abs(x2) < 1e-14, "current identity J.a = psibar aslash chi", std::abs(x1 - x2) / std::abs(x2));
  }
  // ---- pairing counts
  {
    auto f = random_point(rng, 91.1876, ME);
    auto g = random_point(rng, 91.1876, MMU);
    check(pairings(f).size() == 6 && pairings(g).size() == 2, "pairings: 6 for eeee (36 diagrams), 2 for eemm (12)", (double)pairings(f).size());
  }
  // ---- invariances
  double worstShift = 0, worstLorentz = 0, worstSym = 0;
  for (int i = 0; i < 20; ++i) {
    for (double ml : {ME, MMU}) {
      auto f = random_point(rng, 91.1876, ml);
      const double v = m2(f);
      worstShift = std::max(worstShift, std::fabs(m2(f, false) / v - 1));
      auto g = f;
      for (auto& x : g) x.p = lf_transform(x.p, 1.7, 0.31, -0.22, 0.4);
      worstLorentz = std::max(worstLorentz, std::fabs(m2(g) / v - 1));
      if (ml == ME) {
        auto s = f;
        std::swap(s[3].p, s[5].p);
        worstSym = std::max(worstSym, std::fabs(m2(s) / v - 1));
        s = f;
        std::swap(s[2].p, s[4].p);
        worstSym = std::max(worstSym, std::fabs(m2(s) / v - 1));
      }
    }
  }
  check(worstShift < 1e-10, "gauge-shifted currents == Feynman-gauge currents", worstShift);
  check(worstLorentz < 1e-10, "Lorentz invariance (light-front boosts, z rotation)", worstLorentz);
  check(worstSym < 1e-10, "identical-particle symmetry p4<->p6, p3<->p5 (eeee)", worstSym);

  // ---- dump points for the FeynCalc oracle
  if (argc > 1) {
    FILE* out = std::fopen(argv[1], "w");
    char buf[64];
    for (int i = 0; i < 6; ++i)
      for (double ml : {ME, MMU})
        for (int same = 0; same < 2; ++same) {
        auto f = same ? random_point_same_sign(rng, (i < 3) ? 91.1876 : 500.0, ml) : random_point(rng, (i < 3) ? 91.1876 : 500.0, ml);
        std::fprintf(out, "%s %.17e", same ? (ml == ME ? "mmee" : "mmmm") : (ml == ME ? "eeee" : "eemm"), m2(f));
        for (const auto& x : f)
          for (Q c : {x.p.p, x.p.m, x.p.x, x.p.y}) {
            quadmath_snprintf(buf, sizeof buf, "%.36Qe", c);
            std::fprintf(out, " %s", buf);
          }
        std::fprintf(out, "\n");
      }
    std::fclose(out);
    std::printf("wrote %s\n", argv[1]);
  }
  std::printf("%s (%d failures)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
  return failures ? 1 : 0;
}
