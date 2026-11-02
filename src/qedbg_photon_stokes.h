// qedbg_photon_stokes.h -- Stokes parameters of beamstrahlung photons for GUINEA-PIG++/FeeBL with
// photon_polarisation = 1: emission by unpolarised or polarised leptons (quantum-synchrotron radiation in the
// locally-constant-field approximation), conversion from the emission frame to the lab frame, and the
// polarisation dependence of coherent pair creation (nonlinear Breit-Wheeler), including the change of
// polarisation of photons that do not convert.  No dependencies.
//
// Lab-frame Stokes parameters s = (s1, s2, s3), basis (x, y) for the photons of both beams:
//      rho = 1/2 [[1 + s3, s1 - i s2], [s1 + i s2, 1 - s3]]   in the basis (x, y);
// s3 = +1 (-1): linear along x (y); s1 = +1 (-1): linear along (x + y)/sqrt2 ((x - y)/sqrt2); s2 = +1: the
// vector (x + i y)/sqrt2, helicity +1 for a photon moving along +z and -1 for one moving along -z.
// Emission-frame Stokes parameters xi = (xi1, xi2, xi3) refer to e1 = transverse acceleration of the emitter,
// e2 = e3 x e1, e3 = emitter velocity (CAIN; Chen et al., below): xi3 = +1 along e1, xi1 = +1 along
// (e1 + e2)/sqrt2, xi2 = helicity.
#pragma once
#include <algorithm>
#include <cmath>

namespace qedbg {

// ---------------------------------------------------------------- emission (quantum-synchrotron radiation)
// Linear polarisation degree K23/F of a photon of energy fraction x from an unpolarised emitter, given
// Ki53(z) = int_z^inf K_{5/3} and K23(z) = K_{2/3}(z), F = Ki53 + x^2/(1-x) K23 (Seipt & King, PRA 102 (2020)
// 052805, eq. (38) averaged over the initial and summed over the final spin).
inline double synchrotron_linear_degree(double x, double Ki53, double K23) {
  if (!(x > 0.0 && x < 1.0)) return 0.0;
  const double F = Ki53 + x * x / (1.0 - x) * K23;
  if (!(F > 0.0) || !(K23 > 0.0)) return 0.0;
  return std::min(K23 / F, 1.0);
}

// Emission-frame Stokes parameters xi of a photon of energy fraction x emitted by a lepton (electron or
// positron) with spin vector zeta, s1 = zeta.e1, s2 = zeta.e2, s3 = zeta.e3 (all 0: unpolarised), final spin
// summed; z = 2x/(3 Upsilon (1-x)), K13 = K_{1/3}(z), K23 = K_{2/3}(z), Ki13 = int_z^inf K_{1/3},
// Ki53 = int_z^inf K_{5/3} (Chen, Hatsagortsyan, Keitel, Shaisultanov, arXiv:2201.10863, eq. (17)):
//      F*     = Ki53 + x^2/(1-x) K23 - x K13 s2        (emission rate; F00star of PHYSTOOLS::synrad_0_spin_flip)
//      xi1 F* = x/(1-x) K13 s1
//      xi2 F* = [x(2-x)/(1-x) K23 - x Ki13] s3
//      xi3 F* = K23 - x/(1-x) K13 s2 .
// F* and xi3 agree with Seipt & King eq. (38) (spin along e2) and with CAIN; xi1 agrees with CAIN.  For xi2
// the CAIN expression commented out in PHYSTOOLS::synrad_0_spin_flip lacks the term -x Ki13 s3 and, in this
// convention, has the opposite sign; the form used here gives xi2 -> s3 for x -> 1 (the hard photon takes the
// helicity) and xi2 = x s3 + O(x^2) for x -> 0, the soft-photon helicity transfer of bremsstrahlung (Olsen &
// Maximon 1959).  |xi| <= 1 holds exactly for |zeta| <= 1 (the rate into every photon state is non-negative).
// Returns F*; xi is scaled into the unit ball if rounding of the Bessel fits pushes it out.
inline double synchrotron_stokes(double x, double Ki53, double K23, double K13, double Ki13, double s1, double s2,
                                 double s3, float xi[3]) {
  xi[0] = xi[1] = xi[2] = 0.0f;
  if (!(x > 0.0 && x < 1.0)) return 0.0;
  const double r = x / (1.0 - x);
  const double F = Ki53 + x * r * K23 - x * K13 * s2;
  if (!(F > 0.0)) return 0.0;
  double a1 = r * K13 * s1 / F;
  double a2 = (x * (2.0 - x) / (1.0 - x) * K23 - x * Ki13) * s3 / F;
  double a3 = (K23 - r * K13 * s2) / F;
  const double n = std::sqrt(a1 * a1 + a2 * a2 + a3 * a3);
  if (n > 1.0) { a1 /= n; a2 /= n; a3 /= n; }
  xi[0] = (float)a1;
  xi[1] = (float)a2;
  xi[2] = (float)a3;
  return F;
}

// Lab-frame Stokes s of a photon with emission-frame Stokes xi; e1 along the transverse vector (fx, fy) (either
// sign: rotating the frame by pi leaves Stokes parameters unchanged); handed = +1 for a photon moving along +z
// (e2 = z x e1), -1 for one moving along -z (e2 = -z x e1).  No field: unpolarised.
inline void stokes_frame_to_lab(const float xi[3], double fx, double fy, int handed, float s[3]) {
  const double f2 = fx * fx + fy * fy;
  if (!(f2 > 0.0)) { s[0] = s[1] = s[2] = 0.0f; return; }
  // cos 2psi, sin 2psi of the azimuth psi of e1
  const double cos2 = (fx * fx - fy * fy) / f2, sin2 = 2.0 * fx * fy / f2, h = (handed < 0) ? -1.0 : 1.0;
  s[2] = (float)(xi[2] * cos2 - h * xi[0] * sin2);
  s[0] = (float)(xi[2] * sin2 + h * xi[0] * cos2);
  s[1] = (float)(h * xi[1]);
}

// Lab Stokes of a photon linearly polarised with degree P along the transverse vector (fx, fy).
inline void stokes_linear(double P, double fx, double fy, float s[3]) {
  const float xi[3] = {0.0f, 0.0f, (float)std::min(std::max(P, 0.0), 1.0)};
  stokes_frame_to_lab(xi, fx, fy, +1, s);
}

// ---------------------------------------------------------------- coherent pair creation
// Linear Stokes parameter of a photon with respect to the transverse field direction (fx, fy) (tau = +1:
// polarised along the field's E, the polarisation that beamstrahlung photons are emitted with).
inline double stokes_along(const float s[3], double fx, double fy) {
  const double f2 = fx * fx + fy * fy;
  if (!(f2 > 0.0)) return 0.0;
  return (s[2] * (fx * fx - fy * fy) + s[0] * 2.0 * fx * fy) / f2;
}

// For a photon with quantum parameter chi and pair energy split x (eta = 2/(3 chi x(1-x))) the pair spectrum,
// summed over the pair spins, is (Seipt & King eq. (60) in Bessel form)
//      dW/dx ~ fcp(x) - tau K23(eta),   fcp(x) = Ki13(eta) + (x/(1-x) + (1-x)/x) K23(eta)   (PHYSTOOLS::fcp).
// The circular polarisation does not enter.  Since fcp >= 2 K23 the factor (1 - tau K23/fcp) lies in
// [1 - |tau|/2, 1 + |tau|/2]; this returns it divided by 1 + |tau|/2, an acceptance probability in [0, 1].
inline double coherent_pair_split_acceptance(double tau, double fcp, double k23) {
  if (!(fcp > 0.0)) return 0.0;
  const double f = 1.0 - tau * k23 / fcp;
  return std::min(std::max(f, 0.0), 1.0 + 0.5 * std::fabs(tau)) / (1.0 + 0.5 * std::fabs(tau));
}

namespace detail {
// ln(chi) grid from ln(0.01) to ln(1000), 47 points; exact Bessel functions (scripts/coherent_pol_theory.py),
// linear interpolation in ln(chi): |dA| <= 6e-5, |dR/R| <= 2e-4
const double kCohLnChiMin = -4.6051701860, kCohLnChiMax = 6.9077552790;
const int kCohN = 47;
// A(chi) = -int K23 dx / int fcp dx: total rate W(tau) = W_unpol (1 + tau A); A -> -1/3 for chi -> 0
const double kCohA[kCohN] = {-0.332784, -0.332629, -0.332432, -0.332180, -0.331859, -0.331452, -0.330937, -0.330288,
                             -0.329475, -0.328460, -0.327203, -0.325659, -0.323781, -0.321523, -0.318842, -0.315703,
                             -0.312082, -0.307974, -0.303391, -0.298369, -0.292962, -0.287244, -0.281300, -0.275228,
                             -0.269123, -0.263082, -0.257189, -0.251521, -0.246138, -0.241086, -0.236395, -0.232084,
                             -0.228158, -0.224612, -0.221433, -0.218602, -0.216096, -0.213890, -0.211957, -0.210272,
                             -0.208807, -0.207538, -0.206442, -0.205499, -0.204688, -0.203993, -0.203398};
// R(chi) = 1.36 a int fcp dx / u(chi) = int fcp dx / (sqrt(3) pi u), a = 0.13513 = 1/(1.36 sqrt(3) pi),
// u = PHYSTOOLS::u: the conversion probability per step realised by GRID::move_photons2 + pick_coherent_energy
// (trial probability 1.36 tmp; y = 2 rndm - 1 with density 1/2 and acceptance fcp a/u dxdy, where dxdy is twice
// the Jacobian dx/dy) divided by tmp, i.e. the exact LCFA rate over the interpolation u (0.91 ... 1.0)
const double kCohR[kCohN] = {0.997457, 0.997183, 0.996833, 0.996387, 0.995820, 0.995099, 0.994188, 0.993041, 0.991603,
                             0.989812, 0.987598, 0.984888, 0.981610, 0.977692, 0.973086, 0.967771, 0.961767, 0.955152,
                             0.948065, 0.940715, 0.933370, 0.926345, 0.919970, 0.914562, 0.910387, 0.907634, 0.906393,
                             0.906654, 0.908313, 0.911197, 0.915086, 0.919738, 0.924917, 0.930403, 0.936004, 0.941565,
                             0.946962, 0.952106, 0.956936, 0.961413, 0.965519, 0.969250, 0.972615, 0.975628, 0.978311,
                             0.980688, 0.982783};
inline double coherent_interp(const double* t, double chi) {
  if (!(chi > 0.0)) return t[0];
  const double l = (std::log(chi) - kCohLnChiMin) / (kCohLnChiMax - kCohLnChiMin) * (kCohN - 1);
  if (l <= 0.0) return t[0];
  if (l >= kCohN - 1) return t[kCohN - 1];
  const int i = (int)l;
  const double f = l - i;
  return t[i] * (1.0 - f) + t[i + 1] * f;
}
}  // namespace detail

// Asymmetry A(chi) of the total coherent pair rate (clamped outside chi = 0.01 ... 1000, where it is flat).
inline double coherent_pair_asymmetry(double chi) { return detail::coherent_interp(detail::kCohA, chi); }

// Ratio R(chi) of the conversion probability per step that the codes' sampler realises to tmp, the probability
// of GUINEA-PIG's rate formula u (GRID::move_photons2): the exact LCFA rate over the interpolation u.
inline double coherent_pair_rate_ratio(double chi) { return detail::coherent_interp(detail::kCohR, chi); }

// Updates the lab Stokes parameters s of a photon that did not convert during a step.  The step is sampled as
// nsub sub-steps (move_photons2 uses nsub > 1 for trial probabilities >= 0.1); in each, an unpolarised photon
// converts with probability p0/nsub and the two linear eigen-polarisations, along and across the field
// direction (fx, fy), with p0 (1 + A)/nsub and p0 (1 - A)/nsub (for nsub = 1, p0 is the conversion probability
// of the step).  The eigen-polarisations survive with Q+- = (1 - p0 (1 +- A)/nsub)^nsub, and
// rho -> S rho S / Tr(rho S^2) with S = diag(sqrt Q+, sqrt Q-) in the field frame (Lueders' rule).
// Birefringence, a phase between the two eigenstates, is not included: it leaves the component along the
// field, and so the conversion rate, unchanged while the field direction is fixed.  move_photons2 keeps the
// photon's tau from the start of the step for all its sub-steps; to second order in p0 A its survival
// probability is then lower than Tr(rho S^2) by (1 - 1/nsub)(1 - p0/nsub)^(nsub-2) (1 - tau^2)(p0 A)^2/2,
// which is zero for nsub = 1.
inline void stokes_survival_update(float s[3], double fx, double fy, double p0, double A, int nsub = 1) {
  const double f2 = fx * fx + fy * fy;
  if (!(f2 > 0.0) || !(p0 > 0.0)) return;
  if (nsub < 1) nsub = 1;
  // conversion probability per sub-step, capped so that both survival factors stay positive
  const double q = std::min(p0 / nsub, 0.999 / (1.0 + std::fabs(A)));
  // field frame: t3 along / across the field, t1 at 45 degrees (the circular s[1] is unchanged by the rotation)
  const double cos2 = (fx * fx - fy * fy) / f2, sin2 = 2.0 * fx * fy / f2;
  const double t3 = s[2] * cos2 + s[0] * sin2, t1 = -s[2] * sin2 + s[0] * cos2;
  const double Qpar = std::pow(1.0 - q * (1.0 + A), nsub), Qperp = std::pow(1.0 - q * (1.0 - A), nsub);
  const double N = 0.5 * ((1.0 + t3) * Qpar + (1.0 - t3) * Qperp);   // survival probability, Tr(rho S^2)
  if (!(N > 0.0)) return;
  const double t3new = 0.5 * ((1.0 + t3) * Qpar - (1.0 - t3) * Qperp) / N;
  const double offdiag = std::sqrt(Qpar * Qperp) / N;   // factor of the off-diagonal elements (t1, s2)
  const double t1new = offdiag * t1;
  s[2] = (float)(t3new * cos2 - t1new * sin2);
  s[0] = (float)(t3new * sin2 + t1new * cos2);
  s[1] = (float)(offdiag * s[1]);
}

}  // namespace qedbg
