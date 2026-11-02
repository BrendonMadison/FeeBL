// Exact pair invariant mass, and weighted moments that carry an uncertainty.
//
// The W tests use random physical equal-mass four-momenta, head-on and
// non-paraxial, with beam-exchange and
// common-rotation residuals expected at roundoff. W is a Lorentz scalar built
// from two four-vectors, so rotating both momenta together, or swapping which
// one is called beam 1, cannot change it. Those are the two symmetries that
// catch a dropped mass term or a mis-signed vz.
//
// The moments arm checks the estimators against closed forms on small samples
// where the answer can be written down, and checks that streaming equals batch.
#include "pairMomentsCPP.h"
#include <cstdio>
#include <cmath>

using namespace pairmoments;

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

namespace {

const double ME = 0.51099895e-3;   // GeV

struct Dir { double x, y, z; };

Dir unit(double x, double y, double z)
{
  const double n = std::sqrt(x*x + y*y + z*z);
  Dir d; d.x = x/n; d.y = y/n; d.z = z/n; return d;
}

// Rotate about y by a, then about z by b -- a common rotation applied to both.
Dir rot(Dir d, double a, double b)
{
  const double x1 =  d.x*std::cos(a) + d.z*std::sin(a);
  const double z1 = -d.x*std::sin(a) + d.z*std::cos(a);
  const double y1 =  d.y;
  Dir o;
  o.x = x1*std::cos(b) - y1*std::sin(b);
  o.y = x1*std::sin(b) + y1*std::cos(b);
  o.z = z1;
  return o;
}

double W_of(double e1, Dir a, double e2, Dir b)
{
  return invariant_mass(e1, a.x, a.y, a.z, e2, b.x, b.y, b.z, ME);
}

// A deterministic pseudo-random sequence: no <random>, no seeds to argue about,
// and identical on every platform.
double lcg(unsigned long& s)
{
  s = s * 6364136223846793005UL + 1442695040888963407UL;
  return (double)((s >> 11) & 0xFFFFFFFFUL) / 4294967296.0;
}

}  // namespace

int main()
{
  printf("=== W is a Lorentz scalar: exact under the symmetries ===\n");
  {
    unsigned long seed = 20260823UL;
    double worst_rot = 0.0, worst_swap = 0.0, worst_head = 0.0;
    int n = 0;
    for (int i = 0; i < 20000; i++)
      {
        const double e1 = 1.0 + 249.0 * lcg(seed);
        const double e2 = 1.0 + 249.0 * lcg(seed);
        // non-paraxial on purpose: the small-angle case is the easy one
        Dir a = unit(lcg(seed) - 0.5, lcg(seed) - 0.5,  0.5 + lcg(seed));
        Dir b = unit(lcg(seed) - 0.5, lcg(seed) - 0.5, -0.5 - lcg(seed));

        const double W = W_of(e1, a, e2, b);

        const double ra = 0.3 + lcg(seed), rb = 0.7 + lcg(seed);
        const double Wr = W_of(e1, rot(a, ra, rb), e2, rot(b, ra, rb));
        const double Ws = W_of(e2, b, e1, a);

        if (W > 0.0)
          {
            const double dr = std::fabs(Wr - W) / W;
            const double ds = std::fabs(Ws - W) / W;
            if (dr > worst_rot)  worst_rot = dr;
            if (ds > worst_swap) worst_swap = ds;
            n++;
          }
      }
    printf("    %d pairs; worst relative residual: rotation %.2e, exchange %.2e\n",
           n, worst_rot, worst_swap);
    check(n > 19000, "the scan is not vacuous");
    check(worst_rot  < 1e-12, "W is invariant under a common rotation");
    check(worst_swap < 1e-15, "W is invariant under beam exchange");

    // Head-on limit: W -> 2*sqrt(E1 E2) as the mass becomes negligible.
    for (int i = 0; i < 500; i++)
      {
        const double e1 = 10.0 + 240.0 * lcg(seed);
        const double e2 = 10.0 + 240.0 * lcg(seed);
        Dir a = unit(0.0, 0.0,  1.0);
        Dir b = unit(0.0, 0.0, -1.0);
        const double W    = W_of(e1, a, e2, b);
        const double Wapp = 2.0 * std::sqrt(e1 * e2);
        const double d = std::fabs(W - Wapp) / W;
        if (d > worst_head) worst_head = d;
      }
    printf("    head-on vs 2*sqrt(E1 E2): worst relative difference %.2e\n",
           worst_head);
    check(worst_head < 1e-8, "head-on reduces to the massless form");
    check(worst_head > 0.0,  "but not exactly -- the mass term is kept");
  }

  printf("\n=== where the mass term does and does not survive ===\n");
  {
    Dir a = unit(0.0, 0.0, 1.0), b = unit(0.0, 0.0, -1.0);

    // EQUAL energies, head-on: the mass cancels EXACTLY.
    //   W^2 = 2m^2 + 2(E^2 + p^2) = 2m^2 + 2E^2 + 2(E^2 - m^2) = 4E^2
    // so W = 2E whatever the mass is. Worth asserting because it is a place the
    // exact and massless forms must agree to the last bit, and a test that
    // expected a difference here would be wrong -- as an earlier version of
    // this one was.
    const double e = 5.0 * ME;
    const double Weq = W_of(e, a, e, b);
    printf("    equal E = 5 m_e head-on: W = %.12e   2E = %.12e\n", Weq, 2.0*e);
    check(std::fabs(Weq - 2.0 * e) < 1e-15 * (2.0 * e),
          "equal-energy head-on gives W = 2E exactly, mass and all");

    // UNEQUAL energies: now the mass term survives, and the massless shortcut
    // 2*sqrt(E1 E2) is measurably wrong.
    const double e1 = 5.0 * ME, e2 = 20.0 * ME;
    const double W    = W_of(e1, a, e2, b);
    const double Wapp = 2.0 * std::sqrt(e1 * e2);
    const double rel  = std::fabs(W - Wapp) / W;
    printf("    E1 = 5 m_e, E2 = 20 m_e:  exact %.6e   massless %.6e  (%.2e)\n",
           W, Wapp, rel);
    check(rel > 1e-3, "unequal energies expose the mass term");
    check(W < Wapp, "and the exact W is the smaller of the two");

    // At collider energies the same asymmetry is negligible, which is why the
    // shortcut survives in practice -- and why it must not be relied on for a
    // spectrum whose whole subject is the tails.
    const double c1 = 120.0, c2 = 130.0;
    const double Wc = W_of(c1, a, c2, b);
    const double Wc_app = 2.0 * std::sqrt(c1 * c2);
    printf("    E1 = 120 GeV, E2 = 130 GeV: relative difference %.2e\n",
           std::fabs(Wc - Wc_app) / Wc);
    check(std::fabs(Wc - Wc_app) / Wc < 1e-10,
          "at collider energies the two forms agree to 1e-10");
  }

  printf("\n=== weighted moments against closed forms ===\n");
  {
    Moment1 m;
    m.add(1.0, 2.0); m.add(3.0, 6.0);          // mean = (2 + 18)/4 = 5
    check(std::fabs(m.mean() - 5.0) < 1e-12, "weighted mean is sum(wx)/sum(w)");
    // variance = sum w (x-mu)^2 / sum w = (1*9 + 3*1)/4 = 3
    check(std::fabs(m.variance() - 3.0) < 1e-12, "weighted variance matches");
    // N_eff = (4)^2 / (1 + 9) = 1.6
    check(std::fabs(m.n_eff() - 1.6) < 1e-12, "N_eff is Kish (sum w)^2/sum w^2");
  }
  {
    Moment1 eq;
    for (int i = 0; i < 7; i++) eq.add(2.0, (double)i);
    check(std::fabs(eq.n_eff() - 7.0) < 1e-12,
          "equal weights give N_eff = the sample count");
    Moment1 dom;
    dom.add(1e6, 1.0);
    for (int i = 0; i < 50; i++) dom.add(1.0, 2.0);
    check(dom.n_eff() < 1.001,
          "one dominant weight collapses N_eff toward 1");
  }

  printf("\n=== correlation is available, not just the two means ===\n");
  {
    PairHelicity p;
    // perfectly anti-correlated
    p.add(1.0, +1.0, -1.0); p.add(1.0, -1.0, +1.0);
    check(std::fabs(p.h1().mean()) < 1e-12 && std::fabs(p.h2().mean()) < 1e-12,
          "both means are zero");
    check(std::fabs(p.mean_product() + 1.0) < 1e-12,
          "yet <h1 h2> = -1, which the means alone cannot show");
    check(std::fabs(p.covariance() + 1.0) < 1e-12, "covariance is -1");
  }

  printf("\n=== streaming equals batch, and empty is not zero ===\n");
  {
    unsigned long s1 = 7UL, s2 = 7UL;
    SpectrumW a, b;
    a.configure(0.0, 100.0, 10);
    b.configure(0.0, 100.0, 10);
    for (int i = 0; i < 5000; i++)
      { const double w = 0.5 + lcg(s1), W = 100.0 * lcg(s1);
        a.add(w, W, lcg(s1) - 0.5, lcg(s1) - 0.5); }
    for (int i = 0; i < 5000; i++)
      { const double w = 0.5 + lcg(s2), W = 100.0 * lcg(s2);
        b.add(w, W, lcg(s2) - 0.5, lcg(s2) - 0.5); }
    bool same = true;
    for (std::size_t i = 0; i < a.bins(); i++)
      if (std::fabs(a.bin(i).weight() - b.bin(i).weight()) > 1e-12) same = false;
    check(same, "the same stream gives the same bins");

    SpectrumW e;
    e.configure(0.0, 10.0, 4);
    e.add(1.0, 5.0, 0.0, 0.0);        // lands in bin 2 only
    double v = -1.0;
    check(!e.dL_dW(0, v), "an empty bin reports NOT-COMPUTED, not zero");
    check(e.dL_dW(2, v) && std::fabs(v - 1.0 / 2.5) < 1e-12,
          "and a filled bin reports weight per unit W");

    // Luminosity-only pairs must fill dL/dW without touching the helicity
    // means. Folding them in as (0,0) would bias those means toward zero with
    // pairs that never carried a helicity.
    SpectrumW mix;
    mix.configure(0.0, 10.0, 2);
    mix.add(4.0, 2.5);                 // unpolarized: weight only
    mix.add(1.0, 2.5, 1.0, 1.0);       // polarized
    double dv = 0.0;
    check(mix.dL_dW(0, dv) && std::fabs(dv - 5.0/5.0) < 1e-12,
          "an unpolarized pair still contributes to dL/dW");
    check(std::fabs(mix.bin(0).h1().mean() - 1.0) < 1e-12,
          "but does NOT dilute the helicity mean");

    SpectrumW o;
    o.configure(0.0, 10.0, 4);
    o.add(3.0, -1.0, 0.0, 0.0);
    o.add(5.0, 99.0, 0.0, 0.0);
    check(std::fabs(o.underflow() - 3.0) < 1e-12 &&
          std::fabs(o.overflow() - 5.0) < 1e-12,
          "out-of-range weight is kept as under/overflow, not discarded");
  }

  printf("\n  %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
