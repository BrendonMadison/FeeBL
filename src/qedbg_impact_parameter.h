// Impact-parameter profile of the equivalent (virtual) photons of a beam lepton: beam_size = 3.
//
// The splitting l -> l gamma* at light-cone fraction x and photon transverse momentum q_T (relative to
// the emitter; (1-x) Q^2 = q_T^2 + x^2 m^2 =: D exactly) has the photon density, at Q^2 << m^2 W^2-scales,
//     dN/(dx d^2q_T) = alpha/(2 pi^2 x) [ (1+(1-x)^2) q_T^2 + x^4 m^2 ] / D^2
// (helicity-conserving and helicity-flip terms; their sum is the Budnev/finite-mass flux, the conserving
// term alone with q_T^2 -> D is the leading-log flux of mequiv iflag 6).  Its Fourier image, term by
// term, in the separation r of the photon from the outgoing lepton is
//     dN/(dx d^2r) = alpha/(2 pi^2 x) eps^2 [ (1+(1-x)^2) K1(eps r)^2 + x^2 K0(eps r)^2 ],   eps = x m,
// and the photon lies at b = (1-x) r from the emitter's transverse centre (the P+-weighted centre of the
// photon-lepton system, the position the beam-beam code tracks).  For x -> 0 this is the classical
// Weizsaecker-Williams profile (omega/gamma)^2 K1(omega b/gamma)^2.
//
// beam_size = 1 (GUINEA-PIG) puts each photon on the ring r = 1/sqrt(D): its b-distribution is 1/b^2 with
// a sharp edge at 1/eps.  beam_size = 3 draws r = xi/sqrt(D) with xi from one of two universal densities,
//     helicity-conserving:  p(xi) = xi^3 (K0^2 + K1^2),   cdf F(xi) = 1 - xi^2 K1^2 - xi^3 K0 K1,
//     helicity-flip:        p(xi) = 2 xi^2 K0 K1,          cdf F(xi) = 1 - xi^2 K1^2,
// the term being chosen with its share of the density at the photon's q_T.  Mixed over q_T with the
// weights above, the scaled shapes give exactly the K1^2/K0^2 profile for every x (Mellin convolution of
// q_T^2/D^2, resp. eps^2/D^2, with these densities); conditionally on q_T the photon stays at b ~ 1/sqrt(D),
// like the ring.  The conditional shape is a model choice (the quantum joint distribution of q_T and b,
// the Wigner function of the splitting amplitude, is not positive); it matters only through the q_T-b
// correlation seen by the pair kernels at q_T ~ 1/sigma_y << m, where they do not depend on q_T.
// Moments: <ln xi> = ln 2 - gamma_E = 0.1159 (conserving; the ring's shape offset), ln 2 - gamma_E - 1/2
// (flip); <xi> = 1.3879, 0.9253; <xi^2> = 8/3, 4/3.
#ifndef QEDBG_IMPACT_PARAMETER_H
#define QEDBG_IMPACT_PARAMETER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qedbg {

// K0(z) and K1(z), z > 0, from K_nu(z) = int_0^inf exp(-z cosh t) cosh(nu t) dt by the trapezoidal rule,
// which converges geometrically for this entire, double-exponentially decaying integrand: step 1/20 is
// exact to rounding for 1e-12 < z < 700
inline void bessel_k01(double z, double& k0, double& k1)
{
  const double h = 0.05;
  double s0 = 0.5*std::exp(-z), s1 = s0;
  for (int i = 1; i < 100000; ++i)
    {
      const double c = std::cosh(i*h);
      const double e = std::exp(-z*c);
      s0 += e;
      s1 += e*c;
      if (z*c > 1.0 && e*c < 1e-18*s1 && e < 1e-18*s0) break;
    }
  k0 = h*s0;
  k1 = h*s1;
}

// inverse-cdf tables of xi for the helicity-conserving (term 0) and helicity-flip (term 1) shapes,
// built once from the closed-form cdfs on 4096 logarithmic points 1e-4 <= xi <= 40
class ImpactParameterProfile
{
 public:
  static const ImpactParameterProfile& instance()
  {
    static const ImpactParameterProfile profile;
    return profile;
  }

  // xi for a uniform deviate u in [0, 1); linear interpolation in the table (relative error < 1e-5);
  // below xi = 1e-4 (probability 5e-9, resp. 1e-7) xi is drawn from the leading xi^2 behaviour
  double xi(int term, double u) const
  {
    const std::vector<double>& F = cdf_[term];
    if (!(u > F.front())) return xi_.front()*std::sqrt(std::max(u, 0.0)/F.front());
    const std::size_t i = std::upper_bound(F.begin(), F.end(), u) - F.begin();   // F[i-1] <= u < F[i]
    if (i >= F.size()) return xi_.back();
    return xi_[i-1] + (xi_[i] - xi_[i-1])*(u - F[i-1])/(F[i] - F[i-1]);
  }

  // the cdfs themselves (for tests)
  static double cdf(int term, double xi)
  {
    double k0, k1;
    bessel_k01(xi, k0, k1);
    const double s = xi*xi*k1*k1;
    return (term == 1) ? 1.0 - s : 1.0 - s - xi*xi*xi*k0*k1;
  }

 private:
  ImpactParameterProfile() : xi_(kPoints)
  {
    cdf_[0].resize(kPoints);
    cdf_[1].resize(kPoints);
    const double lo = std::log(1e-4), hi = std::log(40.0);
    for (std::size_t i = 0; i < kPoints; ++i)
      {
	xi_[i] = std::exp(lo + (hi - lo)*double(i)/double(kPoints - 1));
	cdf_[0][i] = cdf(0, xi_[i]);
	cdf_[1][i] = cdf(1, xi_[i]);
      }
  }

  static const std::size_t kPoints = 4096;
  std::vector<double> xi_;
  std::vector<double> cdf_[2];
};

}  // namespace qedbg

#endif
