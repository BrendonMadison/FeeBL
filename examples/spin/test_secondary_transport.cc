// Regression tests of the production PAIR_PARTICLE entry points.
#include "particlesCPP.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures=0;
void check(bool condition,const char* message) {
  std::cout<<(condition?"PASS ":"FAIL ")<<message<<'\n';
  failures+=!condition;
}
template<class Function> void rejects(Function function,const char* message) {
  bool rejected=false;
  try {function();} catch(const std::exception&) {rejected=true;}
  check(rejected,message);
}
double norm(const PAIR_PARTICLE& p) {
  return std::hypot(std::hypot(p.transport_momentum(0),p.transport_momentum(1)),p.transport_momentum(2));
}
PAIR_PARTICLE particle(double px,double py,double pz,int sign=1,double mass=EMASS) {
  const double energy=std::hypot(mass,std::hypot(std::hypot(px,py),pz));
  PAIR_PARTICLE p(1,2,0,0,0,px/energy,py/energy,pz/energy,sign*energy);
  p.initialize_transport_momentum(mass,px,py,pz);
  return p;
}
void advance(PAIR_PARTICLE& p,float step,int count,float ex,float ey,float bx,float by) {
  RNDM rng(130010);
  p.apply_initial_half_step_fields(step,EMASS,ex,ey,bx,by,rng);
  for(int i=1;i<count;++i) {
    p.advancePosition(step);
    p.apply_full_step_fields(step,EMASS,ex,ey,bx,by,rng);
  }
  p.advancePosition(step);
  p.apply_final_half_step_fields(step,EMASS,ex,ey,bx,by,0,rng);
}
}

int main() {
  // A canonical momentum smaller than the energy ULP must survive finalization.
  auto tiny=particle(1e-14,0,0);
  check(tiny.transport_energy()==EMASS,"Total energy may round to rest while momentum remains nonzero");
  const double tiny_p=tiny.transport_momentum(0);
  tiny.update_energy(EMASS);
  check(tiny.transport_momentum(0)==tiny_p && tiny.transport_kinetic_energy()>0,
        "Finalization preserves sub-ULP kinetic momentum");

  // Uniform electric impulse, including a passage through rest and reversal.
  for(int charge:{-1,1}) {
    const float impulse=-1e-7f;
    const double initial=-100*double(impulse);
    auto p=particle(charge*initial,0,0,charge);
    p.initialize_imported_spin(TRIDVECTOR(0,0,1)); // deterministic BMT-only orbit
    advance(p,1.0f,200,impulse,0,0,0);
    const double expected=charge*(initial+200*double(impulse));
    check(std::abs(p.transport_momentum(0)-expected)<5e-18,
          "Initial/full/final electric stepping crosses rest with the analytic impulse");
    check(p.transport_energy()>=EMASS && p.transport_momentum(0)*charge<0,
          "Deceleration reverses the momentum without crossing below rest energy");
    check(std::abs(p.getSpin().norm2()-1)<1e-13,"BMT remains defined through rest");
  }

  // The previously rejected float re-projection loses MeV in this regime.
  PAIR_PARTICLE high(1,2,0,0,0,.06669305f,-.59881073f,.79810882f,125);
  high.initialize_transport_from_energy(EMASS);
  const double high_e=high.transport_energy(),high_px=high.transport_momentum(0);
  for(int i=0;i<1000;++i) high.apply_electric_field_on_pair(1,1e-15f,0);
  check(std::abs(high.transport_energy()-high_e)<1e-9,
        "A thousand weak high-energy impulses do not cause MeV energy drift");
  check(std::abs(high.transport_momentum(0)-high_px-1000*double(1e-15f))<2e-12,
        "Weak high-energy momentum increments remain bounded by double rounding");
  const double zero_e=high.transport_energy(),zero_px=high.transport_momentum(0);
  for(int i=0;i<1000;++i) high.apply_electric_field_on_pair(1,0,0);
  check(high.transport_energy()==zero_e && high.transport_momentum(0)==zero_px,
        "Zero electric field preserves canonical state exactly");

  auto magnetic=particle(.3,.4,5);
  const double magnetic_e=magnetic.transport_energy(),magnetic_p=norm(magnetic);
  for(int i=0;i<10000;++i) magnetic.apply_magnetic_field_on_pair(1,1,.001f,.002f);
  check(magnetic.transport_energy()==magnetic_e,"Magnetic rotations do no work");
  check(std::abs(norm(magnetic)/magnetic_p-1)<1e-11,
        "Repeated arbitrary-pitch rotation retains momentum magnitude");

  auto recoil=particle(.2,.3,4);
  const double before=recoil.transport_energy(),px=recoil.transport_momentum(0),pz=recoil.transport_momentum(2);
  recoil.apply_radiation_recoil(.1);
  check(std::abs(before-recoil.transport_energy()-.1)<2e-15,
        "Canonical recoil subtracts the declared photon energy");
  check(std::abs(recoil.transport_momentum(0)/recoil.transport_momentum(2)-px/pz)<1e-15,
        "Collinear recoil preserves momentum direction");
  const double saved=recoil.transport_energy(),saved_p=recoil.transport_momentum(0);
  rejects([&]{recoil.apply_radiation_recoil(saved);},"An excessive photon recoil is rejected");
  check(recoil.transport_energy()==saved && recoil.transport_momentum(0)==saved_p,
        "Rejected recoil leaves the state unchanged");
  auto stopped=particle(1e-14,0,0);
  stopped.apply_radiation_recoil(stopped.transport_kinetic_energy());
  check(stopped.transport_energy()==EMASS && norm(stopped)==0,
        "Removal of the available sub-ULP kinetic energy gives rest");

  auto copied=high;
  copied.apply_electric_field_on_pair(1,.01f,0);
  check(high.transport_momentum(0)==zero_px && copied.transport_momentum(0)!=zero_px,
        "Copied particles have independent canonical momentum");
  copied.setEnergy(50);
  check(copied.transport_energy()==50,"Direct energy update synchronizes canonical state");
  const double old=copied.transport_momentum(0);
  copied.advanceVelocities(.01f,0,1);
  check(std::abs(copied.transport_momentum(0)-old-double(.01f))<1e-14,
        "Direct velocity-impulse API updates canonical momentum");
  ABSTRACT_PARTICLE& base=copied;
  base.setEnergy(40);
  rejects([&]{copied.update_energy(EMASS);},"Unsynchronized base mutation is detected");

  PAIR_PARTICLE invalid(1,2,0,0,0,0,0,0,
    std::nextafter(static_cast<float>(EMASS),0.0f));
  rejects([&]{invalid.initialize_transport_from_energy(EMASS);},"A genuinely submass float input is rejected");
  auto rest=particle(0,0,0);
  rest.initialize_imported_spin(TRIDVECTOR(0,0,1));
  advance(rest,1,2,0,0,0,.01f);
  check(norm(rest)==0 && rest.transport_energy()==EMASS,"A particle at rest supports BMT magnetic rotation");
  rejects([&]{rest.imported_helicity();},"Helicity is explicitly undefined at rest");
  auto muon=particle(0,0,1,1,MUMASS);
  rejects([&]{muon.update_energy(EMASS);},"A secondary cannot silently change rest mass");
  auto finite_probe=particle(0,0,1);
  rejects([&]{finite_probe.apply_electric_field_on_pair(1,std::numeric_limits<float>::infinity(),0);},
          "Nonfinite fields fail in the production entry point");
  auto overflow_probe=particle(.1,.2,1);
  const double overflow_energy=overflow_probe.transport_energy();
  const double overflow_px=overflow_probe.transport_momentum(0);
  const double overflow_py=overflow_probe.transport_momentum(1);
  const double overflow_pz=overflow_probe.transport_momentum(2);
  const float overflow_scale=overflow_probe.scale_pair(0,EMASS);
  const float huge=std::numeric_limits<float>::max();
  rejects([&]{overflow_probe.advanceVelocities(huge,0,huge);},
          "Finite electric impulse beyond output range is rejected");
  check(overflow_probe.transport_energy()==overflow_energy &&
        overflow_probe.transport_momentum(0)==overflow_px &&
        overflow_probe.transport_momentum(1)==overflow_py &&
        overflow_probe.transport_momentum(2)==overflow_pz &&
        overflow_probe.scale_pair(0,EMASS)==overflow_scale,
        "Rejected direct impulse preserves canonical state, mirrors and scale");
  return failures?1:0;
}
