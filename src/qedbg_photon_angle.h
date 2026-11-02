// Out-of-plane emission angle of beamstrahlung photons: photon_angle = 1, 2.
//
// A lepton (energy E, gamma = E/m) in a locally constant field emits a photon of energy fraction u at the
// quantum parameter chi.  On the classical trajectory the photon's angle in the orbit plane is the
// direction of the lepton at the moment of emission (the in-plane angle and the emission time are the same
// variable), so the only angle a trajectory code must add is psi, out of the orbit plane, i.e. along the
// transverse direction perpendicular to the force.  From the Baier-Katkov formula, with the in-plane angle
// integrated as the emission time, t = gamma psi and xi = xi0 (1+t^2)^(3/2), xi0 = u/(3 chi (1-u)):
//   dW/(du dt) ~ (1+t^2) [ (1+(1-u)^2) ((1+t^2) K_{2/3}(xi)^2 + t^2 K_{1/3}(xi)^2) + u^2 K_{1/3}(xi)^2 ] ,
// the quantum form of Jackson's (14.79) (u -> 0: classical; the u^2 term is the spin-flip emission).  Its
// t-integral is the standard LCFA spectrum (1-u+1/(1-u)) K_{2/3}(2 xi0) - int_{2 xi0} K_{1/3}, term by term:
//   orbital   g_A = (1+t^2) ((1+t^2) K_{2/3}^2 + t^2 K_{1/3}^2),  int g_A dt = (pi/(sqrt3 xi0)) int_{2 xi0} K_{5/3},
//   spin flip g_B = (1+t^2) K_{1/3}^2,                          int g_B dt = (pi/(sqrt3 xi0)) int_{2 xi0} K_{1/3},
// weighted (1+(1-u)^2) and u^2.  Width in t: xi0^(-1/3) for soft photons (<|psi|> = 0.4923/gamma xi0^(-1/3),
// the classical gamma^-1 (omega_c/omega)^(1/3)), (6 xi0)^(-1/2) for hard ones.
//
// Sampling: the term with its share (closed form above, tabulated in ln xi0 with step 1/50), then |t| from an
// inverse-cdf table of that shape in s = t/width(xi0) (both shapes depend on xi0 only; 209 rows
// -40 <= ln xi0 <= 12, outside which the scaled shape has reached its limit; 513 points 0 <= s <= 8),
// interpolated in ln xi0, and a random sign.  The three uniforms per photon come from a stream of their own
// (splitmix64_uniform), seeded once from the main generator's next draws without advancing it (seed_from_copy),
// so that the main random sequence is that of collinear emission.
#ifndef QEDBG_PHOTON_ANGLE_H
#define QEDBG_PHOTON_ANGLE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qedbg {

// Trapezoidal rule for e^z int_0^inf exp(-z cosh s) f(s) ds with f = cosh(nu s) (K_nu) or cosh(nu s)/cosh s
// (int_z^inf K_nu): geometric convergence for these entire integrands once the step resolves the width
// 1/sqrt(z) of the integrand, h = min(1/10, 1/(2 sqrt z)) (exact to rounding); the factor e^z keeps
// hard-photon arguments representable.  Returns the two orders nu1, nu2 at once.
inline void scaled_k_trapezoid(double z, double nu1, double nu2, bool over_cosh, double& r1, double& r2)
{
  const double h = std::min(0.1, 0.5/std::sqrt(z));
  double s1 = 0.5, s2 = 0.5;   // s = 0, weight 1/2
  for (int i = 1; i < 1000000; ++i)
    {
      const double s = i*h;
      const double sh = std::sinh(0.5*s);
      double e = std::exp(-2.0*z*sh*sh);   // exp(-z (cosh s - 1))
      if (over_cosh) e /= std::cosh(s);
      const double a = e*std::cosh(nu1*s), b = e*std::cosh(nu2*s);
      s1 += a;
      s2 += b;
      if (2.0*z*sh*sh > 1.0 && a < 1e-18*s1 && b < 1e-18*s2) break;
    }
  r1 = h*s1;
  r2 = h*s2;
}

// e^z K_{1/3}(z), e^z K_{2/3}(z)
inline void scaled_bessel_k13_k23(double z, double& k13, double& k23)
{
  scaled_k_trapezoid(z, 1.0/3.0, 2.0/3.0, false, k13, k23);
}

// e^z int_z^inf K_{1/3}, e^z int_z^inf K_{5/3}
inline void scaled_integral_k13_k53(double z, double& i13, double& i53)
{
  scaled_k_trapezoid(z, 1.0/3.0, 5.0/3.0, true, i13, i53);
}

class PhotonAngleProfile
{
 public:
  static const PhotonAngleProfile& instance()
  {
    static const PhotonAngleProfile profile;
    return profile;
  }

  // xi0 of a photon of energy fraction u emitted at quantum parameter chi
  static double xi0(double u, double chi) { return u/(3.0*chi*(1.0 - u)); }

  // width of the distribution of t = gamma psi: xi0^(-1/3) (soft), (6 xi0)^(-1/2) (hard)
  static double width(double x0) { return 1.0/std::sqrt(std::cbrt(x0*x0) + 6.0*x0); }

  // share of spin-flip emission among the photons at (u, chi): u^2 I13/((1+(1-u)^2) I53 + u^2 I13)
  double spin_flip_share(double u, double chi) const
  {
    const double c1 = 1.0 + (1.0 - u)*(1.0 - u), c2 = u*u;
    const double p = std::min(std::max((std::log(xi0(u, chi)) - kLnLow)/kRatioStep, 0.0), double(kRatioRows - 1));
    const int i = std::min(int(p), kRatioRows - 2);
    const double r = std::exp((1.0 - (p - i))*lnratio_[i] + (p - i)*lnratio_[i+1]);
    return c2*r/(c1 + c2*r);
  }

  // t = gamma psi for uniforms r1 (term), r2 (|t|) and r3 (sign) in [0, 1); 0 < u < 1, chi > 0
  double sample_t(double u, double chi, double r1, double r2, double r3) const
  {
    const double x0 = xi0(u, chi);
    const int term = (r1 < spin_flip_share(u, chi)) ? 1 : 0;
    const double p = std::min(std::max((std::log(x0) - kLnLow)/kLnStep, 0.0), double(kRows - 1));
    const int row = std::min(int(p), kRows - 2);
    const double f = p - row;
    const double s = (1.0 - f)*inverse_cdf(term, row, r2) + f*inverse_cdf(term, row + 1, r2);
    const double t = s*width(x0);
    return (r3 < 0.5) ? -t : t;
  }

 private:
  static const int kRows = 209, kPoints = 513, kRatioRows = 2601;
  static constexpr double kLnLow = -40.0, kLnStep = 0.25, kRatioStep = 0.02, kSMax = 8.0;

  PhotonAngleProfile() : lnratio_(kRatioRows)
  {
    for (int i = 0; i < kRatioRows; ++i)
      {
	double i13, i53;
	scaled_integral_k13_k53(2.0*std::exp(kLnLow + kRatioStep*i), i13, i53);
	lnratio_[i] = std::log(i13/i53);
      }
    cdf_[0].resize(std::size_t(kRows)*kPoints);
    cdf_[1].resize(std::size_t(kRows)*kPoints);
    const double ds = kSMax/(kPoints - 1);
    std::vector<double> g[2] = {std::vector<double>(kPoints), std::vector<double>(kPoints)};
    for (int r = 0; r < kRows; ++r)
      {
	const double x0 = std::exp(kLnLow + kLnStep*r), w = width(x0);
	for (int j = 0; j < kPoints; ++j)
	  {
	    const double t = j*ds*w, q = 1.0 + t*t, xi = x0*q*std::sqrt(q);
	    double k13, k23;
	    scaled_bessel_k13_k23(xi, k13, k23);
	    const double damp = std::exp(-2.0*(xi - x0));   // common factor exp(-2 xi0) taken out of the row
	    g[0][j] = q*(q*k23*k23 + t*t*k13*k13)*damp;
	    g[1][j] = q*k13*k13*damp;
	  }
	for (int a = 0; a < 2; ++a)
	  {
	    double* c = &cdf_[a][std::size_t(r)*kPoints];
	    c[0] = 0.0;
	    for (int j = 1; j < kPoints; ++j) c[j] = c[j-1] + 0.5*(g[a][j-1] + g[a][j])*ds;
	    const double norm = c[kPoints-1];
	    for (int j = 1; j < kPoints; ++j) c[j] /= norm;
	  }
      }
  }

  // s of quantile q in one row: linear interpolation between the cdf points
  double inverse_cdf(int term, int row, double q) const
  {
    const double* c = &cdf_[term][std::size_t(row)*kPoints];
    const int j = int(std::upper_bound(c, c + kPoints, q) - c);   // c[j-1] <= q < c[j]
    if (j >= kPoints) return kSMax;
    const double ds = kSMax/(kPoints - 1);
    return ds*((j - 1) + (q - c[j-1])/(c[j] - c[j-1]));
  }

  std::vector<double> lnratio_;   // ln(int_{2 xi0} K_{1/3} / int_{2 xi0} K_{5/3}), step kRatioStep in ln xi0
  std::vector<double> cdf_[2];    // cdf of |t|/width per row: orbital (0), spin flip (1)
};

// Uniform in (0, 1) from the 64-bit state of splitmix64 (Steele, Lea and Flood, OOPSLA 2014)
inline double splitmix64_uniform(std::uint64_t& state)
{
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
  z ^= z >> 31;
  return (double(z >> 12) + 0.5)*(1.0/4503599627370496.0);   // exact: 2^-53 <= u <= 1 - 2^-53
}

// Seed of that state from the next three draws (24 bits each) of a copy of generator g; g is not advanced
template <class G> std::uint64_t seed_from_copy(const G& g)
{
  G probe(g);
  std::uint64_t s = 0;
  for (int k = 0; k < 3; ++k) s = s*16777216ULL + std::uint64_t(probe.rndm()*16777216.0);
  return s;
}

}  // namespace qedbg

#endif
