#include "particlesCPP.h"
#include "spinTransportCPP.h"

#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
bool secondary_finite(double value) { return feebl::secondary_transport::finite(value); }

void validate_secondary_output_range(const feebl::secondary_transport::Momentum& momentum) {
  if (momentum.energy()>std::numeric_limits<float>::max())
    throw std::domain_error("Secondary energy exceeds float output range");
}
}

void PARTICLE_WITH_SPIN::rotateBMT(TRIDVECTOR Efield, TRIDVECTOR Bfield, float charge_sign, float dt, float betaz) 
{
  // Efield : electric field (lab. frame) : GV/nm
  // Bfield : magnetic field X c (lab. frame) : GV/nm
  // dt : time step in nanometers (c.dt)

  // Thomas-BMT in the ultrarelativistic primary-particle model.  Tests in
  // examples/spin/test_spin_physics.cc compare oblique fields against an
  // independently evaluated full-vector BMT expression.
  int k;
  TRIDVECTOR Blong, Btrans, betaE, omega;
  
  // gamma = E[GeV] / M[Gev/c2]
  double gamma = (double)energy_ / (double)EMASS;


  double vitx = (double)vx_;
  double vity = (double)vy_;

  // vx and vy are transverse direction cosines in the primary-particle
  // approximation |beta|=1.  The argument selects the beam direction; using
  // (vx,vy,+/-1) here is not a unit direction and corrupts both the B parallel
  // projection and beta x E at finite transverse angle.
  const double transverse2 = vitx*vitx + vity*vity;
  if (!(transverse2 < 1.0))
    throw std::domain_error("BMT requires a subluminal transverse direction");
  const double vitz = std::copysign(std::sqrt(1.0-transverse2), (double)betaz);

  // projection of B on the momentum
  double BlongProjection = vitx*Bfield(0) + vity*Bfield(1) + vitz*Bfield(2);
  Blong.setComponents(BlongProjection*vitx, BlongProjection*vity, BlongProjection*vitz);
  Btrans.setComponents(Bfield(0) - Blong(0), Bfield(1) - Blong(1), Bfield(2) - Blong(2));
  // beta x E
  betaE(0) = vity*Efield(2) - vitz*Efield(1);
  betaE(1) = vitz*Efield(0) - vitx*Efield(2);
  betaE(2) = vitx*Efield(1) - vity*Efield(0);

  // Keep the anomalous moment in double precision: gamma amplifies its
  // contribution to the transverse precession rate.
  double anom = PHYSTOOLS::BformFunction(ups_);
  double a1 = 1 + anom;
  double ga1 = 1 + gamma*anom;
  double ce1 = anom + 1./(1.0 + gamma);

  for (k = 0; k< 3; k++)
    {
      omega(k) = -(double)charge_sign*(ga1*Btrans(k) + a1*Blong(k) - ce1*gamma*betaE(k));

    }

  // With E and cB in GV/nm, dt in nm and energy in GeV, dt/energy
  // converts the precession vector to the rotation angle for this step.
  omega *= (double)dt/(double)energy_;
  feebl::spin_transport::rotate(spin_,omega);
}

void PAIR_PARTICLE::sync_transport_mirrors()
{
  // The mirrors retain the existing file and field-gather interfaces.
  // They are never used to reconstruct an already initialized momentum.
  validate_secondary_output_range(transport_);
  energy_=static_cast<float>(transport_.energy());
  vx_=static_cast<float>(transport_.beta(0));
  vy_=static_cast<float>(transport_.beta(1));
  velocityz_=static_cast<float>(transport_.beta(2));
}

void PAIR_PARTICLE::validate_transport_mirrors() const
{
  if (!transport_.initialized())
    throw std::logic_error("Secondary canonical momentum is not initialized");
  if (!secondary_finite(energy_) || !secondary_finite(vx_) || !secondary_finite(vy_) || !secondary_finite(velocityz_) ||
      energy_!=static_cast<float>(transport_.energy()) ||
      vx_!=static_cast<float>(transport_.beta(0)) ||
      vy_!=static_cast<float>(transport_.beta(1)) ||
      velocityz_!=static_cast<float>(transport_.beta(2)))
    throw std::logic_error("Secondary kinematics changed through an unsynchronized base interface");
}

void PAIR_PARTICLE::initialize_transport_momentum(double mass,double px,double py,double pz)
{
  feebl::secondary_transport::Momentum proposed;
  proposed.initialize(mass,px,py,pz);
  validate_secondary_output_range(proposed);
  transport_=proposed;
  sync_transport_mirrors();
}

void PAIR_PARTICLE::initialize_transport_from_energy(double mass)
{
  if (transport_.initialized()) {
    validate_transport_mirrors();
    if (mass!=transport_.mass()) throw std::domain_error("Secondary transport mass changed");
    return;
  }
  const double speed2=double(vx_)*vx_+double(vy_)*vy_+double(velocityz_)*velocityz_;
  if (!secondary_finite(speed2) || speed2>1+8*std::numeric_limits<float>::epsilon())
    throw std::domain_error("Imported secondary has an invalid velocity direction");
  double initial_energy=energy_;
  // A float record cannot distinguish rest energy from its rounded mass bin.
  // Only that exact bin, with unresolved beta^2, maps to the rest state.
  // Lower energies or a resolved momentum at that energy are rejected.
  if (energy_==static_cast<float>(mass)) {
    if (speed2>8*std::numeric_limits<float>::epsilon())
      throw std::domain_error("Rest-energy secondary has a resolved input momentum");
    initial_energy=mass;
  }
  feebl::secondary_transport::Momentum proposed;
  proposed.initialize_energy_direction(mass,initial_energy,vx_,vy_,velocityz_);
  validate_secondary_output_range(proposed);
  transport_=proposed;
  sync_transport_mirrors();
}

void PAIR_PARTICLE::ensure_transport(double mass)
{
  initialize_transport_from_energy(mass);
}

double PAIR_PARTICLE::transport_energy() const {
  validate_transport_mirrors();return transport_.energy();
}
double PAIR_PARTICLE::transport_momentum(int axis) const {
  validate_transport_mirrors();return transport_.component(axis);
}
double PAIR_PARTICLE::transport_kinetic_energy() const {
  validate_transport_mirrors();return transport_.kinetic_energy();
}

void PAIR_PARTICLE::setEnergy(float energy)
{
  if (!secondary_finite(energy) || !(energy>0))
    throw std::domain_error("Secondary total energy must be finite and positive");
  if (!transport_.initialized()) { energy_=energy;return; }
  validate_transport_mirrors();
  auto proposed=transport_;
  const double value=energy==static_cast<float>(transport_.mass()) ? transport_.mass() : double(energy);
  proposed.initialize_energy_direction(transport_.mass(),value,
    transport_.component(0),transport_.component(1),transport_.component(2));
  validate_secondary_output_range(proposed);
  transport_=proposed;sync_transport_mirrors();
}

void PAIR_PARTICLE::setVelocities(float vx,float vy)
{
  ensure_transport(transport_.initialized()?transport_.mass():EMASS);
  auto proposed=transport_;
  proposed.initialize_energy_direction(transport_.mass(),transport_.energy(),vx,vy,velocityz_);
  validate_secondary_output_range(proposed);
  transport_=proposed;sync_transport_mirrors();
}

void PAIR_PARTICLE::advanceVelocities(float fx,float fy,float step)
{
  ensure_transport(transport_.initialized()?transport_.mass():EMASS);
  auto proposed=transport_;
  const double scale=proposed.electric_impulse(double(step)*fx,double(step)*fy);
  validate_secondary_output_range(proposed);
  transport_=proposed;
  last_electric_scale_=scale;
  sync_transport_mirrors();
}

void PAIR_PARTICLE::advanceVelocities(float fx,float fy,float step,float& dvx,float& dvy)
{
  const float before_x=vx_,before_y=vy_;
  advanceVelocities(fx,fy,step);
  dvx=vx_-before_x;dvy=vy_-before_y;
}

void PAIR_PARTICLE::razEnergy()
{
  throw std::logic_error("A massive secondary cannot be zeroed through razEnergy");
}

void PAIR_PARTICLE::apply_radiation_recoil(double photon_energy_GeV)
{
  ensure_transport(transport_.initialized()?transport_.mass():EMASS);
  auto proposed=transport_;
  proposed.radiate(photon_energy_GeV);
  validate_secondary_output_range(proposed);
  transport_=proposed;
  sync_transport_mirrors();
}

void PAIR_PARTICLE::initialize_imported_spin(const TRIDVECTOR& spin)
{
  ensure_transport(EMASS);
  const double norm2=spin.norm2();
  if (!secondary_finite(norm2) || norm2>1.0+1e-10)
    throw std::domain_error("Imported spin requires finite |S|<=1 and an electron/positron momentum");
  imported_spin_=spin;
  imported_spin_enabled_=true;
}

double PAIR_PARTICLE::imported_helicity() const
{
  validate_transport_mirrors();
  const double norm=transport_.magnitude();
  if (!imported_spin_enabled_ || !(norm>0.0))
    throw std::logic_error("Imported helicity requires initialized spin and momentum");
  return (imported_spin_(0)*transport_.component(0)+imported_spin_(1)*transport_.component(1)+
          imported_spin_(2)*transport_.component(2))/norm;
}

void PAIR_PARTICLE::prepare_imported_spin_fields(float ex,float ey,float bx,float by)
{
  if (!imported_spin_enabled_) return;
  imported_electric_=TRIDVECTOR(ex,ey,0);
  imported_magnetic_=TRIDVECTOR(bx,by,0);
  ++imported_spin_steps_;
}

void PAIR_PARTICLE::rotate_imported_spin(double ds_nm,const TRIDVECTOR& electric,
                                       const TRIDVECTOR& magnetic,double mass)
{
  if (!imported_spin_enabled_) return;
  if (mass!=EMASS) throw std::domain_error("Imported spin transport supports electrons and positrons only");
  ensure_transport(mass);
  const double gamma=transport_.energy()/mass;
  if (!secondary_finite(gamma) || gamma<1.0)
    throw std::domain_error("Invalid imported momentum during BMT transport");
  // beta=0 is valid in Thomas-BMT; only helicity is undefined at rest.
  const TRIDVECTOR beta(transport_.beta(0),transport_.beta(1),transport_.beta(2));
  const TRIDVECTOR force(imported_electric_(0)+beta(1)*imported_magnetic_(2)-beta(2)*imported_magnetic_(1),
                        imported_electric_(1)+beta(2)*imported_magnetic_(0)-beta(0)*imported_magnetic_(2),
                        imported_electric_(2)+beta(0)*imported_magnetic_(1)-beta(1)*imported_magnetic_(0));
  const double longitudinal=beta(0)*imported_electric_(0)+beta(1)*imported_electric_(1)+beta(2)*imported_electric_(2);
  const double critical_field=EMASS/(LAMBDA_BAR*1e9); // GV/nm
  const double chi=gamma*std::sqrt(std::max(0.0,force.norm2()-longitudinal*longitudinal))/critical_field;
  imported_max_chi_=std::max(imported_max_chi_,chi);
  const double anomaly=PHYSTOOLS::BformFunction(chi);
  feebl::spin_transport::rotate(imported_spin_,feebl::spin_transport::bmt_angle(
    beta,electric,magnetic,gamma,transport_.energy(),force_charge_sign(),ds_nm,anomaly));
}

double PAIR_PARTICLE::electric_kick(float ds_nm,double mass,float ex,float ey)
{
  ensure_transport(mass);
  if (!secondary_finite(ds_nm) || !secondary_finite(ex) || !secondary_finite(ey))
    throw std::domain_error("Invalid secondary electric field or step");
  const TRIDVECTOR field(ex,ey,0),zero;
  const auto before=transport_;
  const auto spin_before=imported_spin_;
  const double chi_before=imported_max_chi_,scale_before=last_electric_scale_;
  try {
    rotate_imported_spin(0.5*ds_nm,field,zero,mass);
    apply_electric_field_on_pair(ds_nm,ex,ey);
    rotate_imported_spin(0.5*ds_nm,field,zero,mass);
  } catch (...) {
    transport_=before;imported_spin_=spin_before;imported_max_chi_=chi_before;
    last_electric_scale_=scale_before;sync_transport_mirrors();throw;
  }
  return last_electric_scale_;
}

void PAIR_PARTICLE::apply_electric_field_on_pair(float step_2, float ex, float ey) 
{
  // The native signed-energy convention fixes the charge sign of q E ds.
  if (icharge_)
    {
      ex = -ex;
      ey = -ey;
    }
  advanceVelocities(ex, ey, step_2);
}

float PAIR_PARTICLE::apply_magnetic_field_on_pair(float fac_theta, float step_q, float bx, float by) 
{
  ensure_transport(transport_.initialized()?transport_.mass():EMASS);
  if (!secondary_finite(fac_theta) || !secondary_finite(step_q) || fac_theta<0 || step_q<0 ||
      !secondary_finite(bx) || !secondary_finite(by))
    throw std::domain_error("Invalid secondary magnetic field or step");
  const double ds_nm=std::sqrt(double(fac_theta)*step_q);
  const TRIDVECTOR field(bx,by,0),zero;
  const auto before=transport_;
  const auto spin_before=imported_spin_;
  const double chi_before=imported_max_chi_;
  double theta=0;
  try {
    rotate_imported_spin(0.5*ds_nm,zero,field,transport_.mass());
    const double factor=-force_charge_sign()*ds_nm/transport_.energy();
    theta=transport_.magnetic_rotation(factor*bx,factor*by);
    sync_transport_mirrors();
    rotate_imported_spin(0.5*ds_nm,zero,field,transport_.mass());
  } catch (...) {
    transport_=before;imported_spin_=spin_before;imported_max_chi_=chi_before;
    sync_transport_mirrors();throw;
  }
  return static_cast<float>(theta);
}

float PAIR_PARTICLE::scale_pair(float /*vold2*/, double mass)
{
  // Compatibility entry point: the electric impulse already establishes the
  // mass shell. A second normalization would corrupt the canonical state.
  ensure_transport(mass);
  return static_cast<float>(last_electric_scale_);
}

void PAIR_PARTICLE::synchrotron_radiation(float step, double mass, float step_fraction,
  double vx0, double vy0, double vz0, std::vector<float>* photon_e, RNDM& rng)
{
  ensure_transport(mass);
  if (imported_spin_enabled_) return; // Imported spin transport is BMT-only.
#ifdef PAIR_SYN
  if (!secondary_finite(step) || !(step>0) || !secondary_finite(step_fraction) || !(step_fraction>0))
    throw std::domain_error("Secondary radiation requires a positive step");
  const double dx=transport_.beta(0)-vx0,dy=transport_.beta(1)-vy0,dz=transport_.beta(2)-vz0;
  const double bend=std::hypot(std::hypot(dx,dy),dz)/step_fraction;
  const double inverse_radius=bend/(double(step)*1e-9);
  const double energy=transport_.energy();
  const double chi=LAMBDA_BAR*EMASS/(mass*mass*mass)*energy*energy*inverse_radius;
  if (!secondary_finite(chi) || chi>std::numeric_limits<float>::max())
    throw std::domain_error("Secondary radiation parameter is outside the sampler domain");
  std::vector<float> photons;
  PHYSTOOLS::synrad_no_spin_flip(static_cast<float>(chi),static_cast<float>(energy),
    static_cast<float>(bend*EMASS/mass),photons,rng);
  double emitted=0;
  for (float photon:photons) {
    if (!secondary_finite(photon) || !(photon>0))
      throw std::domain_error("Secondary radiation emitted an invalid photon energy");
    emitted+=double(photon);
  }
  // Commit the orbit and photon output only after the complete recoil passes.
  auto proposed=transport_;
  proposed.radiate(emitted);
  validate_secondary_output_range(proposed);
  if (photon_e) photon_e->insert(photon_e->end(),photons.begin(),photons.end());
  transport_=proposed;
  sync_transport_mirrors();
#else
  (void)step;(void)step_fraction;(void)vx0;(void)vy0;(void)vz0;(void)photon_e;(void)rng;
#endif
}

void PAIR_PARTICLE::synchrotron_radiation(float step, double mass, float step_fraction,
  double vx0, double vy0, double vz0, RNDM& rng)
{
  synchrotron_radiation(step,mass,step_fraction,vx0,vy0,vz0,nullptr,rng);
}

float PAIR_PARTICLE::apply_final_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, float /*thetamx*/, RNDM& /*rndm_generator*/)
{
  float theta;
  ensure_transport(mass);
  prepare_imported_spin_fields(ex,ey,bx,by);
  electric_kick(0.5f*step,mass,ex,ey);
  theta = apply_magnetic_field_on_pair(0.25, step*step, bx, by); 
  update_energy(mass);
  return theta;
}

float PAIR_PARTICLE::apply_initial_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, std::vector<float>* photon_e, RNDM& rndm_generator)
{
  double vx0 = 0.0, vy0 = 0.0, vz0 = 0.0;
  float step_2, step_q;
  double scal;
  float theta;
  ensure_transport(mass);
  prepare_imported_spin_fields(ex,ey,bx,by);
  step_2=0.5*step;
  step_q=step*step;
#ifdef PAIR_SYN
  // Keep the entry beta in double for the curvature estimate.
  vx0=transport_.beta(0);vy0=transport_.beta(1);vz0=transport_.beta(2);
#endif
  theta = apply_magnetic_field_on_pair(0.25, step_q, bx, by); 


  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0); 
  
  synchrotron_radiation(step, mass, 0.5, vx0, vy0, vz0, photon_e, rndm_generator);
  return theta;
}
float PAIR_PARTICLE::apply_initial_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, RNDM& rndm_generator)
{
  double vx0 = 0.0, vy0 = 0.0, vz0 = 0.0;
  float step_2, step_q;
  double scal;
  float theta;
  ensure_transport(mass);
  prepare_imported_spin_fields(ex,ey,bx,by);
  step_2=0.5*step;
  step_q=step*step;
#ifdef PAIR_SYN
  // Keep the entry beta in double for the curvature estimate.
  vx0=transport_.beta(0);vy0=transport_.beta(1);vz0=transport_.beta(2);
#endif
  theta = apply_magnetic_field_on_pair(0.25, step_q, bx, by); 


  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0); 
  
  synchrotron_radiation(step, mass, 0.5, vx0, vy0, vz0, rndm_generator);
  return theta;
}
float PAIR_PARTICLE::apply_full_step_fields(float step, double mass, float ex,float ey, float bx, float by, std::vector<float>* photon_e, RNDM& rndm_generator) 
{
  double vx0=0.0,vy0=0.0,vz0=0.0;
  float step_2, step_q;
  double scal;
  float theta;
  ensure_transport(mass);
  prepare_imported_spin_fields(ex,ey,bx,by);
  step_2=0.5*step;
  step_q=step*step;

#ifdef PAIR_SYN
  // Keep the entry beta in double for the curvature estimate.
  vx0=transport_.beta(0);vy0=transport_.beta(1);vz0=transport_.beta(2);
#endif
  // Symmetric electric half-kicks bracket the magnetic rotation.
  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0);   
  
  theta = apply_magnetic_field_on_pair(1.0, step_q, bx, by); 
  
  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0);  
  
  synchrotron_radiation(step, mass, 1.0, vx0, vy0, vz0, photon_e, rndm_generator);
  return theta;
}

float PAIR_PARTICLE::apply_full_step_fields(float step, double mass, float ex,float ey, float bx, float by, RNDM& rndm_generator) 
{
  double vx0=0.0,vy0=0.0,vz0=0.0;
  float step_2, step_q;
  double scal;
  float theta;
  ensure_transport(mass);
  prepare_imported_spin_fields(ex,ey,bx,by);
  step_2=0.5*step;
  step_q=step*step;

#ifdef PAIR_SYN
  // Keep the entry beta in double for the curvature estimate.
  vx0=transport_.beta(0);vy0=transport_.beta(1);vz0=transport_.beta(2);
#endif
  // Symmetric electric half-kicks bracket the magnetic rotation.
  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0);   
  
  theta = apply_magnetic_field_on_pair(1.0, step_q, bx, by); 
  
  scal = electric_kick(step_2,mass,ex,ey);   
  scale_velocities_sync(scal, vx0, vy0, vz0);  
  
  synchrotron_radiation(step, mass, 1.0, vx0, vy0, vz0, rndm_generator);
  return theta;
}


void PAIR_PARTICLE::update_energy(double mass)
{
  // Finalization verifies the canonical state; it must not erase momentum
  // merely because total energy rounds to the float rest-mass bin.
  ensure_transport(mass);
  last_rescaling_ok_=true;
  sync_transport_mirrors();
}
