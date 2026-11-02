// Independent check of the hourglass factor, corrected.
//
// The first version of this test compared against sqrt(pi) u e^{u^2} erfc(u),
// which is the ROUND-beam hourglass: both sigma_x and sigma_y grow with s, so
// the luminosity density falls as 1/(1+(s/beta)^2). The implementation uses
// 1/sqrt(1+(s/beta)^2), the FLAT-beam form where only sigma_y varies. Those
// differ by 21% at sigma_z/beta = 0.73, which is what the failure showed. The
// reference was wrong, not the code.
//
// ILC 250 is flat: beta_x = 13 mm gives sigma_z/beta_x = 0.023, so sigma_x
// varies by 0.03% across the collision, while beta_y = 0.41 mm gives 0.73.
//
// So verify two things separately:
//   * the QUADRATURE machinery, by feeding it the round integrand and checking
//     it against the round closed form;
//   * the FLAT integrand, against a much finer independent integration.
#include "collisionGeometryCPP.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

// Round-beam closed form, for validating the machinery only.
static double round_closed(double sigma_z, double beta)
{
  const double u = beta / sigma_z;   // NOT beta/(sqrt(2) sigma_z): the sqrt(2)
  // is already carried by the collision-point width in the substitution
  return std::sqrt(M_PI) * u * std::exp(u * u) * std::erfc(u);
}

// Same weight as the implementation, but with a selectable integrand and a
// much finer rule, as an independent numerical reference.
static double quad(double sigma_z, double beta, bool round_beam, int n)
{
  const double zmax = 10.0 * sigma_z;
  const double h = 2.0 * zmax / (n - 1);
  double sum = 0.0, norm = 0.0;
  for (int i = 0; i < n; i++) {
    const double z = -zmax + i * h;
    const double w = (i==0||i==n-1)?1.0:((i&1)?4.0:2.0);
    const double g = std::exp(-z*z/(sigma_z*sigma_z));
    const double t = (z/beta)*(z/beta);
    const double r = round_beam ? 1.0/(1.0+t) : 1.0/std::sqrt(1.0+t);
    sum += w*g*r; norm += w*g;
  }
  return sum/norm;
}

int main()
{
  const double beta = 410e-6;

  // 1. the collision-point weight really is sigma_z/sqrt(2)
  {
    const double sz = 1.0; const int n = 200001; const double zmax = 12.0;
    const double h = 2.0*zmax/(n-1);
    double m0=0, m2=0;
    for (int i=0;i<n;i++){ const double z=-zmax+i*h;
      const double w=(i==0||i==n-1)?1.0:((i&1)?4.0:2.0);
      const double g=std::exp(-z*z/(sz*sz)); m0+=w*g; m2+=w*g*z*z; }
    const double rms = std::sqrt(m2/m0);
    printf("       weight RMS = %.6f sigma_z (expect 0.707107)\n", rms);
    check(std::fabs(rms - 1.0/std::sqrt(2.0)) < 1e-6,
          "collision points are Gaussian of RMS sigma_z/sqrt(2)");
  }

  // 2. machinery: the ROUND integrand must reproduce the round closed form
  {
    printf("\n       machinery check, round integrand vs closed form:\n");
    printf("       sigma_z/beta    quadrature    closed form      diff\n");
    bool ok = true;
    const double szs[4] = {41e-6, 123e-6, 300e-6, 615e-6};
    for (int i=0;i<4;i++){
      const double q = quad(szs[i], beta, true, 200001);
      const double c = round_closed(szs[i], beta);
      const double d = (q-c)/c;
      printf("       %10.3f    %10.6f    %10.6f   %+8.4f%%\n",
             szs[i]/beta, q, c, 100.0*d);
      if (std::fabs(d) > 3e-3) ok = false;
    }
    check(ok, "quadrature machinery reproduces the round closed form");
  }

  // 3. the shipped FLAT implementation against a much finer independent rule
  {
    printf("\n       flat integrand, shipped (n=4001) vs fine (n=200001):\n");
    printf("       sigma_z/beta       shipped          fine       diff\n");
    bool ok = true;
    const double szs[4] = {41e-6, 123e-6, 300e-6, 615e-6};
    for (int i=0;i<4;i++){
      const double s = hourglass_factor(szs[i], beta);
      const double f = quad(szs[i], beta, false, 200001);
      const double d = (s-f)/f;
      printf("       %10.3f    %10.6f    %10.6f   %+8.4f%%\n",
             szs[i]/beta, s, f, 100.0*d);
      if (std::fabs(d) > 1e-3) ok = false;
    }
    check(ok, "shipped flat quadrature matches a 50x finer integration");
  }

  // 4. flat must always exceed round: 1/sqrt(1+t) > 1/(1+t)
  {
    const double f = hourglass_factor(300e-6, beta);
    const double r = round_closed(300e-6, beta);
    printf("\n       at sigma_z/beta = 0.732: flat %.5f, round %.5f\n", f, r);
    check(f > r, "flat hourglass exceeds round, as the integrands require");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT" : "hourglass implementation verified");
  return fails ? 1 : 0;
}
