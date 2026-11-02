// qedbg_photon_polarisation.h -- polarisation of beamstrahlung photons and polarised Breit-Wheeler pair
// creation, used by GUINEA-PIG++/FeeBL with the opt-in switch photon_polarisation = 1.
//
// Stokes parameters (s1, s2, s3) of a photon are kept in the LAB basis (x, y), the same for the photons
// of both beams: the polarisation density matrix in the basis (x, y) is
//      rho = 1/2 [[1 + s3, s1 - i s2], [s1 + i s2, 1 - s3]],
// i.e. s3 = +1 (-1) is linear along x (y), s1 = +1 (-1) linear along (x + y)/sqrt2 ((x - y)/sqrt2), and
// s2 = +1 the polarisation vector (x + i y)/sqrt2, which is helicity +1 for a photon moving along +z and
// helicity -1 for one moving along -z.  Linear polarisation of degree P along the azimuth psi:
// s3 = P cos 2psi, s1 = P sin 2psi.
//
// Beamstrahlung: in the locally-constant-field approximation (the spectrum F = Ki53(z) + x^2/(1-x) K23(z),
// z = 2x/(3 Upsilon (1-x)), that both codes sample) an unpolarised emitter radiates photons linearly
// polarised along its transverse acceleration (the Lorentz force) with degree
//      xi = K23(z) / F
// (Seipt & King, PRA 102 (2020) 052805, eq. (38) averaged over the initial and summed over the final
// spin; CAIN's photon Stokes parameter).  xi -> 1/2 for x << Upsilon.  With spin tracking the emitter's
// spin adds the terms of qedbg::synchrotron_stokes (qedbg_photon_stokes.h): a transverse-spin correction to
// the linear polarisation and the rate, linear polarisation at 45 degrees, and circular polarisation from
// the helicity (FeeBL; GUINEA-PIG++ keeps the unpolarised-emitter degree, see synrad_0_spin_flip).
// Emission, lab-frame conversion and coherent pair creation: qedbg_photon_stokes.h.
//
// Breit-Wheeler gamma(k1, along +z) gamma(k2, along -z) -> l- l+, g = m^2/(w1 w2) < 1, beta = sqrt(1-g),
// L = ln((1+beta)/(1-beta)), lab Stokes s, t of photons 1 and 2:
//      sigma = sigma_np [1 + tau_lin (s3 t3 + s1 t1) - tau_circ s2 t2],
//      tau_lin  = -g (beta + g L/2) / (2 [(1 + g - g^2/2) L - (1 + g) beta]),
//      tau_circ = (2 - g)(beta + g L/2) / [(1 + g - g^2/2) L - (1 + g) beta] - 1
// (parallel linear polarisations: sigma_np (1 + tau_lin) -> 0 at threshold; equal helicities, J_z = 0:
// sigma_np (1 + tau_circ)).  The direction (cos theta*, phi) of l- in the gamma-gamma frame has the
// density, up to a constant,
//      [N_A + (s3 t3 + s1 t1) N_B - s2 t2 N_C + ((s3 + t3) cos 2phi + (s1 + t1) sin 2phi) N_D
//       + ((s3 t3 - s1 t1) cos 4phi + (s3 t1 + s1 t3) sin 4phi) N_E] / (1 - beta^2 cos^2 theta*)^2,
//      N_A = 1 - beta^4 + beta^2 sin^2 (2 - beta^2 sin^2),  N_C = 1 - beta^4 - beta^2 sin^2 (2 - beta^2 sin^2),
//      N_B = -(1 - beta^2)^2,  N_D = 2 beta^2 (1 - beta^2) sin^2,  N_E = -beta^4 sin^4
// (sin = sin theta*).  Derivation: the helicity amplitudes contracted with the photons' density matrices,
// Appendix A.5 of docs/PHOTON_POLARISATION.md in the deliverables (check: scripts/bw_helicity_derivation_check.py);
// tested against the amplitudes of qedbg_tree.h (tests/unit/test_photon_polarisation.cc).
#pragma once
#include <algorithm>
#include <cmath>

#include "qedbg_pair_kernels.h"
#include "qedbg_photon_stokes.h"

namespace qedbg {

// Asymmetries tau_lin and tau_circ of the Breit-Wheeler cross section, g = m^2/(w1 w2) in (0, 1).
inline void bw_polarisation_asymmetries(double g, double& tau_lin, double& tau_circ) {
  const double b = std::sqrt(1.0 - g), L = 2.0 * std::log1p(b) - std::log(g);
  const double np = (1.0 + g - 0.5 * g * g) * L - (1.0 + g) * b, j = b + 0.5 * g * L;
  tau_lin = -g * j / (2.0 * np);
  tau_circ = (2.0 - g) * j / np - 1.0;
}

// sigma(s, t) / sigma_np for photons with lab Stokes s (photon 1, +z) and t (photon 2, -z).
inline double bw_polarisation_factor(double g, const float s[3], const float t[3]) {
  double tl, tc;
  bw_polarisation_asymmetries(g, tl, tc);
  const double f = 1.0 + tl * ((double)s[2] * t[2] + (double)s[0] * t[0]) - tc * (double)s[1] * t[1];
  return f > 0.0 ? f : 0.0;
}

// Ratio of the polarised to the unpolarised density of the l- direction (cos theta*, phi) in the
// gamma-gamma frame (photon 1 along +z); sth = sin theta*, (cphi, sphi) = (cos phi, sin phi).
inline double bw_polarisation_ratio(double g, double sth, double cphi, double sphi, const float s[3], const float t[3]) {
  const double b2 = 1.0 - g, sn2 = sth * sth, bs = b2 * sn2;
  const double NA = g * (2.0 - g) + bs * (2.0 - bs);           // 1 - beta^4 + beta^2 sin^2 (2 - beta^2 sin^2)
  const double NC = g * (2.0 - g) - bs * (2.0 - bs);
  const double NB = -g * g, ND = 2.0 * b2 * g * sn2, NE = -bs * bs;
  const double cos2p = cphi * cphi - sphi * sphi, sin2p = 2.0 * sphi * cphi;
  const double cos4p = cos2p * cos2p - sin2p * sin2p, sin4p = 2.0 * sin2p * cos2p;
  const double s1 = s[0], s2 = s[1], s3 = s[2], t1 = t[0], t2 = t[1], t3 = t[2];
  const double num = NA + (s3 * t3 + s1 * t1) * NB - s2 * t2 * NC + ((s3 + t3) * cos2p + (s1 + t1) * sin2p) * ND +
                     ((s3 * t3 - s1 * t1) * cos4p + (s3 * t1 + s1 * t3) * sin4p) * NE;
  return (NA > 0.0 && num > 0.0) ? num / NA : 0.0;
}

// Polar and azimuthal angle of l- in the gamma-gamma frame for polarised photons: the exact unpolarised
// sampler (bw_sample_angle) and a uniform azimuth, accepted with the ratio above divided by its bound
// (1 + |s|)(1 + |t|).  Bound: |M|^2 summed over the lepton spins is a positive form T in the two photons'
// polarisations, so its contraction with rho1 x rho2 is at most lambda_max(rho1) lambda_max(rho2) Tr T, and
// the unpolarised value is Tr T / 4; lambda_max(rho) = (1 + |s|)/2.
template <class Uniform>
inline void bw_sample_angle_polarised(double g, const float s[3], const float t[3], Uniform&& rnd, double& cth, double& sth,
                                      double& cphi, double& sphi,
                                      unsigned max_trials = kMaximumAngleTrials) {
  const double ps = std::sqrt((double)s[0] * s[0] + (double)s[1] * s[1] + (double)s[2] * s[2]);
  const double pt = std::sqrt((double)t[0] * t[0] + (double)t[1] * t[1] + (double)t[2] * t[2]);
  const double rmax = (1.0 + std::min(ps, 1.0)) * (1.0 + std::min(pt, 1.0));
  for (unsigned trial=0; trial<max_trials; ++trial) {
    bw_sample_angle(g, rnd, cth, sth);
    const double phi = 2.0 * M_PI * rnd();
    cphi = std::cos(phi);
    sphi = std::sin(phi);
    if (rmax <= 1.0) return;
    if (rnd() * rmax <= bw_polarisation_ratio(g, sth, cphi, sphi, s, t)) return;
  }
  throw std::runtime_error("Polarised Breit-Wheeler angle sampler exhausted its trial limit");
}

}  // namespace qedbg
