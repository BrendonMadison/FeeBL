// qedbg_tree.h -- exact tree-level QED helicity amplitudes for processes with
// three fermion lines and two internal photons (e.g. e+e- -> e+e- l+l-, e-e- -> e-e- l+l-),
// two fermion lines with one internal and one external photon (e gamma -> e l+l-), and one
// fermion line with two external photons (gamma gamma -> l+l-).
//
// Conventions
//  * Metric (+,-,-,-).  Light-cone components a^+ = a^0 + a^3, a^- = a^0 - a^3,
//    a_T = a^1 + i a^2, abar_T = a^1 - i a^2 (independent for complex vectors).
//  * Chiral (Weyl) representation, gamma^mu = [[0, sigma^mu], [sigmabar^mu, 0]].
//    Spinors are light-cone helicity spinors built from (p^+, p_T, m) only, so no
//    cancellation occurs for particles moving close to the beam axis.
//  * External kinematics use the extended real type Q (long double by default):
//    every propagator denominator and photon virtuality is formed from light-cone
//    components, which keeps Q^2 ~ m_e^2 x^2 ~ 1e-27 GeV^2 accurate at TeV energies.
//    The Dirac algebra itself is done in double precision.
//  * The amplitude returned is the reduced amplitude A with |M|^2 = e^8 |A|^2 (2->4),
//    e^6 |A|^2 (2->3, one external photon) or e^4 |A|^2 (2->2 with two photons): every
//    diagram of a given process carries the same coupling and i-factors, so only the
//    relative fermion-permutation signs matter.  Fermion lines are built from all
//    flavour-conserving pairings of barred ("left") and unbarred ("right") external
//    spinors; the relative sign of a pairing is the parity of the permutation.
//  * End-line currents may be gauge shifted, J -> J - (J.n / Q.n) Q, which is exact by
//    the per-line abelian Ward identity and removes the large E/m cancellations of
//    Feynman gauge for quasi-real photons.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "qedbg_real.h"

namespace qedbg {

typedef std::complex<double> C;

// ---------------------------------------------------------------- kinematics
struct LCq {               // extended-precision four-vector in light-cone components
  Q p, m, x, y;            // plus, minus, px, py
};
inline LCq operator+(const LCq& a, const LCq& b) { return {a.p + b.p, a.m + b.m, a.x + b.x, a.y + b.y}; }
inline LCq operator-(const LCq& a, const LCq& b) { return {a.p - b.p, a.m - b.m, a.x - b.x, a.y - b.y}; }
inline LCq operator-(const LCq& a) { return {-a.p, -a.m, -a.x, -a.y}; }
inline LCq operator*(Q s, const LCq& a) { return {s * a.p, s * a.m, s * a.x, s * a.y}; }
inline Q dotq(const LCq& a, const LCq& b) { return (a.p * b.m + a.m * b.p) / 2 - a.x * b.x - a.y * b.y; }
inline Q sqq(const LCq& a) { return a.p * a.m - a.x * a.x - a.y * a.y; }
inline Q energy(const LCq& a) { return (a.p + a.m) / 2; }
inline Q pzq(const LCq& a) { return (a.p - a.m) / 2; }
// on-shell vectors: the small light-cone component follows from the mass shell
inline LCq onshell_plus(Q plus, Q px, Q py, Q mass) { return {plus, (mass * mass + px * px + py * py) / plus, px, py}; }
inline LCq onshell_minus(Q minus, Q px, Q py, Q mass) { return {(mass * mass + px * px + py * py) / minus, minus, px, py}; }

// ---------------------------------------------------------------- Dirac algebra
struct CV {                // complex four-vector, light-cone components
  C p, m, t, tb;
};
inline CV cv(const LCq& a) {
  const double px = (double)a.x, py = (double)a.y;
  return {C((double)a.p, 0.0), C((double)a.m, 0.0), C(px, py), C(px, -py)};
}
inline CV operator-(const CV& a, const CV& b) { return {a.p - b.p, a.m - b.m, a.t - b.t, a.tb - b.tb}; }
inline CV operator+(const CV& a, const CV& b) { return {a.p + b.p, a.m + b.m, a.t + b.t, a.tb + b.tb}; }
inline CV operator*(C s, const CV& a) { return {s * a.p, s * a.m, s * a.t, s * a.tb}; }
inline C dot(const CV& a, const CV& b) { return 0.5 * (a.p * b.m + a.m * b.p) - 0.5 * (a.t * b.tb + a.tb * b.t); }

struct Sp {                // Dirac spinor (L1, L2, R1, R2) in the chiral representation
  C c[4];
};
inline Sp operator+(const Sp& a, const Sp& b) { return {{a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2], a.c[3] + b.c[3]}}; }
inline Sp operator*(C s, const Sp& a) { return {{s * a.c[0], s * a.c[1], s * a.c[2], s * a.c[3]}}; }

// light-cone helicity spinors; Sum_h u ubar = pslash + m, Sum_h v vbar = pslash - m
inline Sp u_spinor(const LCq& p, double mass, int h) {
  const double sq = std::sqrt((double)p.p);
  const C pt((double)p.x, (double)p.y);
  if (h == 0) return {{C(mass / sq, 0), C(0, 0), C(sq, 0), pt / sq}};
  return {{-std::conj(pt) / sq, C(sq, 0), C(0, 0), C(mass / sq, 0)}};
}
inline Sp v_spinor(const LCq& p, double mass, int h) {
  const double sq = std::sqrt((double)p.p);
  const C pt((double)p.x, (double)p.y);
  if (h == 0) return {{C(-mass / sq, 0), C(0, 0), C(sq, 0), pt / sq}};
  return {{-std::conj(pt) / sq, C(sq, 0), C(0, 0), C(-mass / sq, 0)}};
}
// aslash * s
inline Sp slash(const CV& a, const Sp& s) {
  return {{a.m * s.c[2] - a.tb * s.c[3], -a.t * s.c[2] + a.p * s.c[3],
           a.p * s.c[0] + a.tb * s.c[1], a.t * s.c[0] + a.m * s.c[1]}};
}
// psibar chi
inline C bar_dot(const Sp& psi, const Sp& chi) {
  return std::conj(psi.c[0]) * chi.c[2] + std::conj(psi.c[1]) * chi.c[3] +
         std::conj(psi.c[2]) * chi.c[0] + std::conj(psi.c[3]) * chi.c[1];
}
// psibar gamma^mu chi
inline CV current(const Sp& psi, const Sp& chi) {
  const C l1 = std::conj(psi.c[0]), l2 = std::conj(psi.c[1]), r1 = std::conj(psi.c[2]), r2 = std::conj(psi.c[3]);
  return {2.0 * (r1 * chi.c[2] + l2 * chi.c[1]), 2.0 * (r2 * chi.c[3] + l1 * chi.c[0]),
          2.0 * (r1 * chi.c[3] - l1 * chi.c[1]), 2.0 * (r2 * chi.c[2] - l2 * chi.c[0])};
}
// gauge shift J -> J - (J.n/Q.n) Q with n chosen to maximise |Q.n|; n_- gives J'^+ = 0
inline CV gauge_shift(const CV& J, const LCq& q) {
  if (q_fabs(q.p) >= q_fabs(q.m)) return J - (J.p / C((double)q.p, 0.0)) * cv(q);
  return J - (J.m / C((double)q.m, 0.0)) * cv(q);
}

// ---------------------------------------------------------------- external states
struct Fermion {
  LCq p;          // physical momentum (positive energy)
  double mass;
  int flavor;
  bool incoming;
  bool anti;
  bool left() const { return incoming == anti; }  // barred end: outgoing particle or incoming antiparticle
  // momentum flowing along the fermion arrow at this end (out of the line at the left end,
  // into the line at the right end): +p for particles, -p for antiparticles
  LCq flow() const { return anti ? -p : p; }
  Sp spinor(int h) const { return anti ? v_spinor(p, mass, h) : u_spinor(p, mass, h); }
};

struct Pairing {           // a set of fermion lines (left particle, right particle) with sign
  std::vector<std::array<int, 2>> lines;
  int sign;
};

// all flavour-conserving pairings of left ends with right ends; sign = permutation parity
inline std::vector<Pairing> pairings(const std::vector<Fermion>& f) {
  std::vector<int> L, R;
  for (int i = 0; i < (int)f.size(); ++i) (f[i].left() ? L : R).push_back(i);
  std::vector<Pairing> out;
  std::vector<int> perm(R.size());
  for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int)i;
  do {
    bool ok = true;
    for (size_t i = 0; i < L.size(); ++i) ok = ok && f[L[i]].flavor == f[R[perm[i]]].flavor;
    if (!ok) continue;
    int inv = 0;
    for (size_t i = 0; i < perm.size(); ++i)
      for (size_t j = i + 1; j < perm.size(); ++j) inv += perm[i] > perm[j];
    Pairing P;
    P.sign = (inv % 2) ? -1 : 1;
    for (size_t i = 0; i < L.size(); ++i) P.lines.push_back({L[i], R[perm[i]]});
    out.push_back(P);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return out;
}

// One fermion line prepared for amplitude assembly.
struct Line {
  int l, r;                // particle indices of left and right ends
  LCq kin, kout;           // momentum entering at the right end / leaving at the left end
  LCq Qabs;                // momentum absorbed from photons: kout - kin
  double mass;
  Sp psi[2], chi[2];
  CV J[2][2];              // (gauge-shifted) current psibar gamma chi / Q^2, indices [hl][hr]
};

inline Line make_line(const std::vector<Fermion>& f, int l, int r, bool shift) {
  Line L;
  L.l = l; L.r = r;
  L.kin = f[r].flow(); L.kout = f[l].flow();
  L.Qabs = L.kout - L.kin;
  L.mass = f[l].mass;
  for (int h = 0; h < 2; ++h) { L.psi[h] = f[l].spinor(h); L.chi[h] = f[r].spinor(h); }
  const double q2 = (double)sqq(L.Qabs);
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b) {
      CV j = current(L.psi[a], L.chi[b]);
      if (shift) j = gauge_shift(j, L.Qabs);
      L.J[a][b] = (1.0 / q2) * j;
    }
  return L;
}

// Propagator numerator (kslash + m) applied to s, with denominator evaluated in Q.
inline Sp propagate(const LCq& k, double m, const Sp& s) {
  const double D = (double)(sqq(k) - (Q)m * (Q)m);
  return (C(1.0 / D, 0)) * (slash(cv(k), s) + C(m, 0) * s);
}

// ---------------------------------------------------------------- 2 -> 4 (three lines)
// amp[h] with helicity bit i of h belonging to particle i (0..5).
inline void amplitude_3lines(const std::vector<Fermion>& f, std::array<C, 64>& amp, bool shift = true) {
  amp.fill(C(0, 0));
  for (const Pairing& P : pairings(f)) {
    if (P.lines.size() != 3) continue;
    Line Ls[3];
    for (int i = 0; i < 3; ++i) Ls[i] = make_line(f, P.lines[i][0], P.lines[i][1], shift);
    for (int s = 0; s < 3; ++s) {            // spine (middle) line
      const Line& S = Ls[s];
      const Line& A = Ls[(s + 1) % 3];
      const Line& B = Ls[(s + 2) % 3];
      for (int ord = 0; ord < 2; ++ord) {     // which end line attaches next to the right end
        const Line& F1 = ord ? B : A;         // first insertion (near chi)
        const Line& F2 = ord ? A : B;         // second insertion (near psibar)
        // the photon from line F carries -F.Qabs into the spine
        const LCq k1 = S.kin - F1.Qabs;
        for (int hr = 0; hr < 2; ++hr)
          for (int a1 = 0; a1 < 2; ++a1)
            for (int b1 = 0; b1 < 2; ++b1) {
              const Sp v1 = propagate(k1, S.mass, slash(F1.J[a1][b1], S.chi[hr]));
              for (int a2 = 0; a2 < 2; ++a2)
                for (int b2 = 0; b2 < 2; ++b2) {
                  const Sp v2 = slash(F2.J[a2][b2], v1);
                  for (int hl = 0; hl < 2; ++hl) {
                    const int idx = (hl << S.l) | (hr << S.r) | (a1 << F1.l) | (b1 << F1.r) | (a2 << F2.l) | (b2 << F2.r);
                    amp[idx] += (double)P.sign * bar_dot(S.psi[hl], v2);
                  }
                }
            }
      }
    }
  }
}

inline double sum_sq(const std::array<C, 64>& a) {
  double s = 0;
  for (const C& z : a) s += std::norm(z);
  return s;
}

// ---------------------------------------------------------------- external real photons
// Two real linear polarisation vectors of a massless photon with momentum k, in light-cone gauge
// with respect to the opposite light-cone direction (eps^- = 0 if the photon moves towards -z,
// eps^+ = 0 otherwise); eps.k = 0, eps.eps = -1, and Sum eps eps is -g up to gauge terms.
inline void photon_polarisations(const LCq& k, CV eps[2]) {
  const double kx = (double)k.x, ky = (double)k.y;
  if (q_fabs(k.m) >= q_fabs(k.p)) {
    const double km = (double)k.m;
    eps[0] = {C(2 * kx / km, 0), C(0, 0), C(1, 0), C(1, 0)};
    eps[1] = {C(2 * ky / km, 0), C(0, 0), C(0, 1), C(0, -1)};
  } else {
    const double kp = (double)k.p;
    eps[0] = {C(0, 0), C(2 * kx / kp, 0), C(1, 0), C(1, 0)};
    eps[1] = {C(0, 0), C(2 * ky / kp, 0), C(0, 1), C(0, -1)};
  }
}

// ---------------------------------------------------------------- 2 -> 3: two lines + one incoming photon
// e(+-) gamma -> e(+-) l+ l- and crossings.  Four external fermions (helicity bits 0..3) and one
// incoming photon of momentum k with polarisation vectors eps[0..1] (bit 4).  For each pairing the
// photon attaches to either line, in either order relative to the internal photon (4 diagrams).
inline void amplitude_2lines_photon(const std::vector<Fermion>& f, const LCq& k, const CV eps[2],
                                    std::array<C, 32>& amp, bool shift = true) {
  amp.fill(C(0, 0));
  for (const Pairing& P : pairings(f)) {
    if (P.lines.size() != 2) continue;
    Line Ls[2];
    for (int i = 0; i < 2; ++i) Ls[i] = make_line(f, P.lines[i][0], P.lines[i][1], shift);
    for (int X = 0; X < 2; ++X) {
      const Line& S = Ls[X];
      const Line& E = Ls[1 - X];
      const LCq kA = S.kin + k;          // external photon first (next to chi)
      const LCq kB = S.kin - E.Qabs;     // internal photon first
      for (int hr = 0; hr < 2; ++hr)
        for (int lam = 0; lam < 2; ++lam) {
          const Sp vA = propagate(kA, S.mass, slash(eps[lam], S.chi[hr]));
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
              const Sp wA = slash(E.J[a][b], vA);
              const Sp wB = slash(eps[lam], propagate(kB, S.mass, slash(E.J[a][b], S.chi[hr])));
              for (int hl = 0; hl < 2; ++hl) {
                const int idx = (hl << S.l) | (hr << S.r) | (a << E.l) | (b << E.r) | (lam << 4);
                amp[idx] += (double)P.sign * (bar_dot(S.psi[hl], wA) + bar_dot(S.psi[hl], wB));
              }
            }
        }
    }
  }
}

inline double sum_sq(const std::array<C, 32>& a) {
  double s = 0;
  for (const C& z : a) s += std::norm(z);
  return s;
}

// ---------------------------------------------------------------- 2 -> 2: one line + two photons
// gamma(k1) gamma(k2) -> l+ l- (Breit-Wheeler) for arbitrary (also off-shell, complex) polarisation
// vectors; helicity bits 0..1 for the fermions, bit 2 and bit 3 for the two photons.
inline void amplitude_1line_2photons(const std::vector<Fermion>& f, const LCq& k1, const CV e1[2], const LCq& k2,
                                     const CV e2[2], std::array<C, 16>& amp) {
  amp.fill(C(0, 0));
  for (const Pairing& P : pairings(f)) {
    if (P.lines.size() != 1) continue;
    const int l = P.lines[0][0], r = P.lines[0][1];
    const Fermion& L = f[l];
    const Fermion& R = f[r];
    const LCq kin = R.flow();
    for (int hl = 0; hl < 2; ++hl)
      for (int hr = 0; hr < 2; ++hr)
        for (int a = 0; a < 2; ++a)
          for (int b = 0; b < 2; ++b) {
            const Sp chi = R.spinor(hr), psi = L.spinor(hl);
            const Sp o1 = slash(e2[b], propagate(kin + k1, R.mass, slash(e1[a], chi)));
            const Sp o2 = slash(e1[a], propagate(kin + k2, R.mass, slash(e2[b], chi)));
            amp[(hl << l) | (hr << r) | (a << 2) | (b << 3)] += (double)P.sign * (bar_dot(psi, o1) + bar_dot(psi, o2));
          }
  }
}

}  // namespace qedbg
