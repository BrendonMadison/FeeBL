// Unit tests of the external-photon amplitudes in qedbg_tree.h:
//  - gamma gamma -> l+l- (Breit-Wheeler) against the analytic massive |M|^2 and photon Ward identity;
//  - e gamma -> e l+l- (Bethe-Heitler in GUINEA-PIG naming): photon Ward identity, gauge-shift
//    equivalence, Lorentz invariance, identical-electron symmetry;
//  - dump of random e gamma -> e l+l- points for the FeynCalc oracle (benchmarks/qed_validation/examples/numerical/compare_oracle_bh.wl).
// CMake target feebl_qed_bh_bw_oracle uses the integrated source headers.
#define QEDBG_USE_QUADMATH
#include "qedbg_tree.h"

#include <cstdio>
#include <random>

using namespace qedbg;

static int failures = 0;
static void check(bool ok, const char* what, double val) {
  std::printf("%-66s %s  (%.3e)\n", what, ok ? "PASS" : "FAIL", val);
  if (!ok) ++failures;
}
static const double ME = 0.51099895000e-3, MMU = 0.1056583755;

struct Cart { Q E, x, y, z; };
static Cart boost(const Cart& p, Q bx, Q by, Q bz) {
  const Q b2 = bx * bx + by * by + bz * bz;
  if (b2 == 0) return p;
  const Q g = 1 / sqrtq(1 - b2), bp = bx * p.x + by * p.y + bz * p.z, f = (g - 1) * bp / b2 + g * p.E;
  return {g * (p.E + bp), p.x + f * bx, p.y + f * by, p.z + f * bz};
}
static LCq lc_cart(const Cart& c, Q mass) {
  if (c.z >= 0) return onshell_plus(c.E + c.z, c.x, c.y, mass);
  return onshell_minus(c.E - c.z, c.x, c.y, mass);
}
static Q twobody(Q M, Q m1, Q m2) { return sqrtq((M * M - (m1 + m2) * (m1 + m2)) * (M * M - (m1 - m2) * (m1 - m2))) / (2 * M); }
static void decay(const Cart& P, Q m1, Q m2, Q cth, Q phi, Cart& k1, Cart& k2) {
  const Q M = sqrtq(P.E * P.E - P.x * P.x - P.y * P.y - P.z * P.z), q = twobody(M, m1, m2), sth = sqrtq(1 - cth * cth);
  Cart a{sqrtq(q * q + m1 * m1), q * sth * cosq(phi), q * sth * sinq(phi), q * cth};
  Cart b{sqrtq(q * q + m2 * m2), -a.x, -a.y, -a.z};
  k1 = boost(a, P.x / P.E, P.y / P.E, P.z / P.E);
  k2 = boost(b, P.x / P.E, P.y / P.E, P.z / P.E);
}

// e-(p0, +z, energy E) gamma(k, -z, energy w) -> e-(f1) l+(f2) l-(f3); fermion indices 0..3
static std::vector<Fermion> bh_point(std::mt19937_64& rng, Q E, Q w, double ml, LCq& k, bool positron = false) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const Q me = ME, m = ml;
  const Q pz = sqrtq(E * E - me * me);
  Cart P{E + w, 0, 0, pz - w};
  const Q s = P.E * P.E - P.z * P.z, rs = sqrtq(s);
  const Q M1 = 2 * m + (rs - me - 2 * m) * (Q)U(rng);
  Cart k1, r1, k2, k3;
  decay(P, me, M1, 2 * (Q)U(rng) - 1, 2 * M_PIq * (Q)U(rng), k1, r1);
  decay(r1, m, m, 2 * (Q)U(rng) - 1, 2 * M_PIq * (Q)U(rng), k2, k3);
  k = {0, 2 * w, 0, 0};   // massless photon along -z: k^+ = 0, k^- = 2w
  const int fl = (ml == ME) ? 0 : 1;
  std::vector<Fermion> f(4);
  f[0] = {onshell_plus(E + pz, 0, 0, me), ME, 0, true, positron};
  f[1] = {lc_cart(k1, me), ME, 0, false, positron};
  f[2] = {lc_cart(k2, m), ml, fl, false, true};
  f[3] = {lc_cart(k3, m), ml, fl, false, false};
  return f;
}
static double bh_m2(const std::vector<Fermion>& f, const LCq& k, bool shift = true) {
  CV eps[2];
  photon_polarisations(k, eps);
  std::array<C, 32> a;
  amplitude_2lines_photon(f, k, eps, a, shift);
  return sum_sq(a);
}
static LCq lf_transform(const LCq& a, Q eta, Q vx, Q vy, Q phi) {
  LCq b{eta * a.p, a.m / eta, a.x, a.y};
  LCq c{b.p, b.m + 2 * (vx * b.x + vy * b.y) + (vx * vx + vy * vy) * b.p, b.x + vx * b.p, b.y + vy * b.p};
  return {c.p, c.m, cosq(phi) * c.x - sinq(phi) * c.y, sinq(phi) * c.x + cosq(phi) * c.y};
}

int main(int argc, char** argv) {
  std::mt19937_64 rng(271828);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  // ---------------- Breit-Wheeler: gamma(k1,+z) gamma(k2,-z) -> l+(f0) l-(f1), CM frame
  {
    double worst = 0, worstW = 0;
    for (double ml : {ME, MMU})
      for (int i = 0; i < 20; ++i) {
        const Q m = ml, w = m * (1 + 5 * (Q)U(rng) * (Q)U(rng) * 100);   // photon energy in CM
        const Q pz = sqrtq(w * w - m * m), cth = 2 * (Q)U(rng) - 1, sth = sqrtq(1 - cth * cth), phi = 2 * M_PIq * (Q)U(rng);
        const LCq k1{2 * w, 0, 0, 0}, k2{0, 2 * w, 0, 0};
        Cart a{w, pz * sth * cosq(phi), pz * sth * sinq(phi), pz * cth}, b{w, -a.x, -a.y, -a.z};
        std::vector<Fermion> f = {{lc_cart(a, m), ml, 0, false, true}, {lc_cart(b, m), ml, 0, false, false}};
        CV e1[2], e2[2];
        photon_polarisations(k1, e1);
        photon_polarisations(k2, e2);
        std::array<C, 16> amp;
        amplitude_1line_2photons(f, k1, e1, k2, e2, amp);
        double s2 = 0;
        for (auto& z : amp) s2 += std::norm(z);
        // analytic (crossed Compton, Peskin 5.105 with p.k1 = -(t-m^2)/2):
        // Sum|M|^2/e^4 = 8[(u-m2)/(t-m2) + (t-m2)/(u-m2) - 4m2(1/(t-m2)+1/(u-m2)) - 4m4(1/(t-m2)+1/(u-m2))^2]
        const double m2 = ml * ml;
        const double t = (double)sqq(k1 - f[1].p), u = (double)sqq(k1 - f[0].p);
        const double T = t - m2, Uu = u - m2;
        const double ref = 8 * (Uu / T + T / Uu - 4 * m2 * (1 / T + 1 / Uu) - 4 * m2 * m2 * (1 / T + 1 / Uu) * (1 / T + 1 / Uu));
        worst = std::max(worst, std::fabs(s2 / ref - 1));
        // Ward identity: eps1 -> k1
        CV kk[2] = {cv(k1), cv(k1)};
        amplitude_1line_2photons(f, k1, kk, k2, e2, amp);
        double sw = 0;
        for (auto& z : amp) sw += std::norm(z);
        worstW = std::max(worstW, std::sqrt(sw / s2) / (double)w);
      }
    check(worst < 1e-12, "Breit-Wheeler helicity sum == analytic massive |M|^2", worst);
    check(worstW < 1e-12, "Breit-Wheeler photon Ward identity (eps -> k)", worstW);
  }
  // ---------------- e gamma -> e l+ l-
  {
    double wWard = 0, wShift = 0, wLor = 0, wSym = 0;
    for (double ml : {ME, MMU})
      for (int i = 0; i < 20; ++i) {
        LCq k;
        auto f = bh_point(rng, 250, 3 + 50 * (Q)U(rng), ml, k, i % 2 == 1);
        const double v = bh_m2(f, k);
        CV kk[2] = {cv(k), cv(k)};
        std::array<C, 32> a;
        amplitude_2lines_photon(f, k, kk, a, true);
        wWard = std::max(wWard, std::sqrt(sum_sq(a) / v) / (double)energy(k));
        wShift = std::max(wShift, std::fabs(bh_m2(f, k, false) / v - 1));
        auto g = f;
        for (auto& x : g) x.p = lf_transform(x.p, 1.9, -0.27, 0.41, 0.8);
        wLor = std::max(wLor, std::fabs(bh_m2(g, lf_transform(k, 1.9, -0.27, 0.41, 0.8)) / v - 1));
        if (ml == ME) {
          auto s = f;
          // the outgoing lepton with the beam's charge is identical to the pair lepton of that charge
          if (i % 2 == 1) std::swap(s[1].p, s[2].p); else std::swap(s[1].p, s[3].p);
          wSym = std::max(wSym, std::fabs(bh_m2(s, k) / v - 1));
        }
      }
    check(wWard < 1e-12, "e gamma -> e l l: photon Ward identity (eps -> k)", wWard);
    check(wShift < 1e-10, "e gamma -> e l l: gauge-shifted == Feynman-gauge currents", wShift);
    check(wLor < 1e-10, "e gamma -> e l l: Lorentz invariance (light-front transforms)", wLor);
    check(wSym < 1e-10, "e gamma -> e e e: identical outgoing-electron symmetry", wSym);
  }
  if (argc > 1) {
    FILE* out = std::fopen(argv[1], "w");
    char buf[64];
    for (int i = 0; i < 8; ++i)
      for (double ml : {ME, MMU}) {
        LCq k;
        auto f = bh_point(rng, (i < 4) ? 45.6 : 250.0, 0.5 + 30 * (Q)U(rng), ml, k);
        std::fprintf(out, "%s %.17e", ml == ME ? "egeee" : "egemm", bh_m2(f, k));
        for (Q c : {k.p, k.m, k.x, k.y}) { quadmath_snprintf(buf, sizeof buf, "%.36Qe", c); std::fprintf(out, " %s", buf); }
        for (const auto& x : f)
          for (Q c : {x.p.p, x.p.m, x.p.x, x.p.y}) { quadmath_snprintf(buf, sizeof buf, "%.36Qe", c); std::fprintf(out, " %s", buf); }
        std::fprintf(out, "\n");
      }
    std::fclose(out);
    std::printf("wrote %s\n", argv[1]);
  }
  std::printf("%s (%d failures)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
  return failures ? 1 : 0;
}
