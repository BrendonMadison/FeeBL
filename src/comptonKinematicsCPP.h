#ifndef FEEBL_COMPTON_KINEMATICS_CPP_H
#define FEEBL_COMPTON_KINEMATICS_CPP_H

#include "physconst.h"
#include "mathconst.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace feebl { namespace compton {

// Unpolarized, real-photon Klein-Nishina scattering. The incident photon is
// antiparallel to the electron axis. Virtual-photon fluxes supplied by the
// caller retain their equivalent-photon approximation.
struct HeadOnIncident {
  double energy_GeV;
  double photon_GeV;
  double mass_GeV;
  double momentum_GeV;
  double plus_GeV;
  double minus_GeV;
  double x;                         // 2 p.k / m^2, not total Mandelstam s/m^2
  int direction;
  std::array<double, 3> axis;
};

struct FinalState {
  double electron_energy_GeV;
  double photon_energy_GeV;
  std::array<double, 3> electron_momentum_GeV;
  std::array<double, 3> photon_momentum_GeV;
  double electron_angle_rad;
  double photon_angle_rad;
};

// The native primary transverse coordinates are direction cosines. Form a
// unit incident axis, keeping the beam/charge identifier separate from recoil.
inline HeadOnIncident incident(double energy, double photon, double vx,
                              double vy, int direction, double mass = EMASS) {
  const double transverse2 = vx*vx + vy*vy;
  if (!std::isfinite(energy) || !std::isfinite(photon) ||
      !std::isfinite(mass) || !std::isfinite(transverse2) ||
      !(mass > 0.0) || energy < mass || !(photon > 0.0) ||
      transverse2 >= 1.0 || (direction != 1 && direction != -1))
    throw std::domain_error("Invalid head-on Compton incident state");
  const double momentum = std::sqrt((energy-mass)*(energy+mass));
  const double plus = energy+momentum;
  const double minus = mass*mass/plus;
  const double x = 2.0*photon/minus;
  if (!std::isfinite(x) || !(x > 0.0))
    throw std::domain_error("Compton invariant is outside the finite domain");
  return {energy, photon, mass, momentum, plus, minus, x, direction,
          {{vx, vy, direction*std::sqrt(1.0-transverse2)}}};
}

struct NativeIncident {
  HeadOnIncident kinematics;
  int charge_sign; // PAIR_PARTICLE signed-energy convention, independent of pz.
};

// GRID branch 1 is gamma(beam1) + particle(beam2); branch 2 interchanges them.
// The target beam determines the physical axis: beam1 travels +z, beam2 -z.
// Negative source energy identifies an opposite-charge coherent/trident
// particle travelling with that same beam. Klein-Nishina is charge symmetric.
inline NativeIncident incident_for_native_branch(double signed_energy, double photon,
                                                  double vx, double vy, int branch) {
  if (branch != 1 && branch != 2)
    throw std::domain_error("Compton collision branch must be 1 (ge) or 2 (eg)");
  const int direction=branch == 1 ? -1 : 1;
  const int relative_charge=signed_energy < 0.0 ? -1 : 1;
  return {incident(std::abs(signed_energy),photon,vx,vy,direction),
          direction*relative_charge};
}

// With t=-log(1-y), the density is bounded even at the high-energy endpoint.
// Here y=1-omega_out*/omega_in* in the electron rest frame and
// r=y/[x(1-y)]=(1-cos(theta*))/2.
inline double density_log_recoil(double x, double t) {
  const double u = std::exp(-t);
  const double r = std::expm1(t)/x;
  return 1.0 + u*u - 4.0*r*(1.0-r)*u;
}

// Integral of the Klein-Nishina bracket from y=0 to y=1-exp(-t).
inline double integral_log_recoil(double x, double t) {
  const double y = -std::expm1(-t);
  if (t < 0.1) {
    // Evaluate (y-t)/t^2 and (sinh(t)-t)/t^3 by their Taylor series.
    // Factoring the powers as t*(t/x)^k avoids both cancellation near the
    // Thomson limit and underflow of t^3 or x^2 for very soft photons.
    double alternating=-0.5, term=-0.5;
    for (int order=3; order<=14; ++order) {
      term *= -t/order;
      alternating += term;
    }
    double odd=1.0/6.0;
    term=odd;
    for (int order=5; order<=15; order+=2) {
      term *= t*t/((order-1)*order);
      odd += term;
    }
    const double ratio=t/x;
    return t+y-0.5*y*y+4.0*t*ratio*alternating+
           8.0*t*ratio*ratio*odd;
  }
  return t+y-0.5*y*y+4.0*(y-t)/x+
         8.0*((std::sinh(t)-t)/x)/x;
}

// Total unpolarized cross section in square metres.
inline double cross_section_m2(double x) {
  if (!std::isfinite(x) || !(x > 0.0))
    throw std::domain_error("Compton cross section requires positive finite x");
  return 2.0*PI*RE*RE*integral_log_recoil(x,std::log1p(x))/x;
}

// Invert the same normalized density using exactly one supplied uniform draw.
// Safeguarded Newton steps use a bracket in t, so the stopping scale does not
// discard the narrow y~1 endpoint at large x.
inline double sample_log_recoil(double x, double uniform) {
  if (!std::isfinite(x) || !(x > 0.0) || !std::isfinite(uniform) ||
      uniform < 0.0 || uniform > 1.0)
    throw std::domain_error("Invalid Compton inverse-CDF input");
  const double maximum = std::log1p(x);
  if (uniform == 0.0) return 0.0;
  if (uniform == 1.0) return maximum;
  const double total = integral_log_recoil(x,maximum);
  const double target = uniform*total;
  double low=0.0, high=maximum, t=uniform*maximum;
  for (int iteration=0; iteration<80; ++iteration) {
    const double value = integral_log_recoil(x,t);
    if (std::abs(value-target) <= 32.0*std::numeric_limits<double>::epsilon()*total)
      return t;
    if (value < target) low=t; else high=t;
    if (std::nextafter(low,high) >= high) return 0.5*(low+high);
    double trial=t-(value-target)/density_log_recoil(x,t);
    if (!std::isfinite(trial) || trial <= low || trial >= high)
      trial=0.5*(low+high);
    t=trial;
  }
  throw std::runtime_error("Compton inverse CDF did not converge");
}

// Stable light-cone reconstruction conserves both four-momenta and permits
// backward electrons. In particular, no positive sqrt(pz^2) is imposed.
inline FinalState final_state(const HeadOnIncident& in, double t, double phi) {
  const double maximum=std::log1p(in.x);
  if (!std::isfinite(t) || !std::isfinite(phi) || t < 0.0 || t > maximum)
    throw std::domain_error("Compton recoil is outside its sampled support");
  const double u=std::exp(-t), y=-std::expm1(-t);
  // d=(1+x)exp(-t)-1. This form stays nonnegative at the endpoint
  // without subtracting two nearly equal numbers or clipping a recoil.
  const double d=std::expm1(maximum-t);
  if (!std::isfinite(d) || d < 0.0)
    throw std::domain_error("Compton recoil violates the Klein-Nishina support");
  const double pt=in.mass_GeV*std::sqrt(y*d);
  const double electron_plus=in.plus_GeV*u;
  const double electron_minus=in.minus_GeV*(1.0+y)+2.0*in.photon_GeV*y;
  const double photon_plus=in.plus_GeV*y;
  const double photon_minus=in.minus_GeV*d;
  const double electron_parallel=0.5*(electron_plus-electron_minus);
  const double photon_parallel=0.5*(photon_plus-photon_minus);
  FinalState result;
  result.electron_energy_GeV=0.5*(electron_plus+electron_minus);
  result.photon_energy_GeV=0.5*(photon_plus+photon_minus);
  result.electron_angle_rad=std::atan2(pt,electron_parallel);
  result.photon_angle_rad=std::atan2(pt,photon_parallel);

  const auto& n=in.axis;
  const std::array<double,3> reference = std::abs(n[0]) < 0.9
      ? std::array<double,3>{{1.0,0.0,0.0}}
      : std::array<double,3>{{0.0,1.0,0.0}};
  const double projection=n[0]*reference[0]+n[1]*reference[1]+n[2]*reference[2];
  std::array<double,3> first;
  double norm2=0.0;
  for (int axis=0; axis<3; ++axis) {
    first[axis]=reference[axis]-projection*n[axis];
    norm2+=first[axis]*first[axis];
  }
  const double inverse_norm=1.0/std::sqrt(norm2);
  for (double& component:first) component*=inverse_norm;
  const std::array<double,3> second{{
    in.direction*(n[1]*first[2]-n[2]*first[1]),
    in.direction*(n[2]*first[0]-n[0]*first[2]),
    in.direction*(n[0]*first[1]-n[1]*first[0])}};
  for (int axis=0; axis<3; ++axis) {
    const double transverse=pt*(std::sin(phi)*first[axis]+std::cos(phi)*second[axis]);
    result.electron_momentum_GeV[axis]=electron_parallel*n[axis]+transverse;
    result.photon_momentum_GeV[axis]=photon_parallel*n[axis]-transverse;
  }
  return result;
}

} }
#endif
