#ifndef FEEBL_SECONDARY_MOMENTUM_CPP_H
#define FEEBL_SECONDARY_MOMENTUM_CPP_H

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace feebl { namespace secondary_transport {

// A bit test remains effective when a production build enables fast-math.
inline bool finite(double value) {
  static_assert(sizeof(double)==sizeof(std::uint64_t), "IEEE binary64 required");
  static_assert(std::numeric_limits<double>::is_iec559, "IEEE binary64 required");
  std::uint64_t bits;
  std::memcpy(&bits,&value,sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

// Persistent momentum in GeV/c, mass in GeV/c^2 and energy in GeV (c=1).
// Energy is derived from momentum at initialization and electric impulses.
// A magnetic rotation preserves its entry energy and momentum magnitude.
// Float particle members are output mirrors, never inputs to another step.
class Momentum {
  std::array<double,3> p_{{0,0,0}};
  double mass_=0, energy_=0;
  bool initialized_=false;

  static double norm(const std::array<double,3>& p) {
    return std::hypot(std::hypot(p[0],p[1]),p[2]);
  }
  static void validate(double mass,const std::array<double,3>& p,double energy) {
    if (!finite(mass) || !(mass>0) || !finite(energy) || energy<mass ||
        !finite(p[0]) || !finite(p[1]) || !finite(p[2]))
      throw std::domain_error("Invalid secondary canonical momentum or mass");
  }

public:
  bool initialized() const { return initialized_; }
  double mass() const { return mass_; }
  double energy() const { return energy_; }
  double component(int axis) const { return p_.at(axis); }
  double beta(int axis) const { return p_.at(axis)/energy_; }
  double magnitude() const { return norm(p_); }
  // Unlike E-m, this retains sub-ULP kinetic energy close to rest.
  double kinetic_energy() const {
    const double momentum=norm(p_);
    return momentum*(momentum/(energy_+mass_));
  }

  void initialize(double mass,double px,double py,double pz) {
    const std::array<double,3> proposed{{px,py,pz}};
    const double energy=std::hypot(mass,norm(proposed));
    validate(mass,proposed,energy);
    p_=proposed;mass_=mass;energy_=energy;initialized_=true;
  }

  // Imported E and direction determine an on-shell state. The direction's
  // length is not used as a second, inconsistent estimate of the energy.
  void initialize_energy_direction(double mass,double energy,
                                   double dx,double dy,double dz) {
    const std::array<double,3> direction{{dx,dy,dz}};
    validate(mass,direction,energy);
    const double length=norm(direction);
    if (!finite(length) || (energy>mass && !(length>0)))
      throw std::domain_error("Secondary energy requires a nonzero direction");
    const double magnitude=std::sqrt(energy-mass)*std::sqrt(energy+mass);
    std::array<double,3> proposed{{0,0,0}};
    if (magnitude>0)
      for(int axis=0;axis<3;++axis) proposed[axis]=magnitude*(direction[axis]/length);
    validate(mass,proposed,energy);
    p_=proposed;mass_=mass;energy_=energy;initialized_=true;
  }

  // q E ds has momentum units because E is GV/nm and ds=c dt is nm.
  // Return E_before/E_after for the radiation curvature's old-velocity map.
  double electric_impulse(double dpx,double dpy) {
    if (!initialized_ || !finite(dpx) || !finite(dpy))
      throw std::domain_error("Invalid secondary electric impulse");
    if (dpx==0 && dpy==0) return 1;
    const std::array<double,3> proposed{{p_[0]+dpx,p_[1]+dpy,p_[2]}};
    const double energy=std::hypot(mass_,norm(proposed));
    validate(mass_,proposed,energy);
    const double scale=energy_/energy;
    p_=proposed;energy_=energy;
    return scale;
  }

  // Rodrigues rotation about a transverse axis. Renormalize the rotated
  // momentum to its entry magnitude to prevent magnetic work from norm
  // roundoff. This never feeds the rounded float velocity back into energy.
  double magnetic_rotation(double angle_x,double angle_y) {
    if (!initialized_ || !finite(angle_x) || !finite(angle_y))
      throw std::domain_error("Invalid secondary magnetic rotation");
    const double angle=std::hypot(angle_x,angle_y), before=norm(p_);
    if (angle==0 || before==0) return 0;
    const double nx=angle_x/angle,ny=angle_y/angle;
    const double c=std::cos(angle),s=std::sin(angle);
    const double one_minus_c=2*std::sin(0.5*angle)*std::sin(0.5*angle);
    const double projection=nx*p_[0]+ny*p_[1];
    std::array<double,3> proposed{{
      p_[0]*c+ny*p_[2]*s+nx*projection*one_minus_c,
      p_[1]*c-nx*p_[2]*s+ny*projection*one_minus_c,
      p_[2]*c+(nx*p_[1]-ny*p_[0])*s}};
    const double after=norm(proposed);
    if (!finite(after) || !(after>0))
      throw std::domain_error("Invalid rotated secondary momentum norm");
    for(double& value:proposed) value*=before/after;
    validate(mass_,proposed,energy_);
    // Keep the legacy step diagnostic: angle times transverse speed.
    const double pitch=std::hypot(p_[2],nx*p_[1]-ny*p_[0])/energy_;
    p_=proposed;
    return angle*pitch;
  }

  // Remove a checked sum of photon energies without changing direction.
  // No photon is clipped or discarded; an impossible recoil fails closed.
  void radiate(double photon_energy) {
    if (!initialized_ || !finite(photon_energy) || photon_energy<0)
      throw std::domain_error("Invalid secondary radiated energy");
    if (photon_energy==0) return;
    const double kinetic=kinetic_energy();
    if (photon_energy>kinetic)
      throw std::domain_error("Secondary radiation exceeds available kinetic energy");
    const double remaining=kinetic-photon_energy;
    const double magnitude=std::sqrt(remaining)*std::sqrt(remaining+2*mass_);
    const double before=norm(p_);
    std::array<double,3> proposed=p_;
    for(double& value:proposed) value*=magnitude/before;
    const double energy=mass_+remaining;
    validate(mass_,proposed,energy);
    p_=proposed;energy_=energy;
  }
};

} }
#endif
