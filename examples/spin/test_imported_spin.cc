#include "particlesCPP.h"
#include "importedSpinCPP.h"
#include "comptonKinematicsCPP.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <list>

namespace {
using Vector=std::array<double,3>;
int failures=0;
double dot(Vector a,Vector b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vector cross(Vector a,Vector b) { return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}}; }
Vector scaled(Vector a,double factor) { for (auto& v:a) v*=factor;return a; }
Vector rotation(Vector vector,Vector angle) {
  const double magnitude=std::sqrt(dot(angle,angle));
  if (magnitude==0) return vector;
  const auto axis=scaled(angle,1/magnitude), perpendicular=cross(axis,vector);
  Vector result;
  for (int i=0;i<3;++i) result[i]=vector[i]*std::cos(magnitude)
    +perpendicular[i]*std::sin(magnitude)+axis[i]*dot(axis,vector)*(1-std::cos(magnitude));
  return result;
}
Vector direction(const PAIR_PARTICLE& particle) {
  float vx,vy;particle.velocities(vx,vy);Vector v={{vx,vy,particle.Zvelocity()}};
  return scaled(v,1/std::sqrt(dot(v,v)));
}
Vector spin(const PAIR_PARTICLE& particle) {
  const auto& s=particle.getSpin();return {{s(0),s(1),s(2)}};
}
double error(Vector a,Vector b) {
  double result=0;for (int i=0;i<3;++i) result=std::max(result,std::abs(a[i]-b[i]));return result;
}
void check(bool ok,const char* label,double value=0) {
  std::printf("%s: %s (%.12g)\n",label,ok?"PASS":"FAIL",value);failures+=!ok;
}

// Use exactly the initial/full/final entry points used by GRID::step_pair_1.
void advance(PAIR_PARTICLE& p,float step,int count,float ex,float ey,float bx,float by) {
  RNDM rng(12781);
  p.apply_initial_half_step_fields(step,EMASS,ex,ey,bx,by,rng);
  for (int i=1;i<count;++i) {
    p.advancePosition(step);
    p.apply_full_step_fields(step,EMASS,ex,ey,bx,by,rng);
  }
  p.advancePosition(step);
  p.apply_final_half_step_fields(step,EMASS,ex,ey,bx,by,0,rng);
}

double magnetic_case(int charge,int direction_sign,int count) {
  const float energy=5.1099906f, field=2e-11f;
  const double gamma=double(energy)/EMASS;
  const double beta=std::sqrt(1-1/(gamma*gamma));
  const Vector nominal=scaled(Vector{{.2,.6,std::sqrt(.6)}},direction_sign);
  PAIR_PARTICLE particle(1,-99,0,0,0,beta*nominal[0],beta*nominal[1],beta*nominal[2],charge*energy);
  const auto n0=direction(particle);const Vector s0={{.3,-.4,std::sqrt(.75)}};
  particle.initialize_imported_spin(TRIDVECTOR(s0[0],s0[1],s0[2]));
  const float step=static_cast<float>(.25*energy/(double(field)*count));
  const double length=double(step)*count;
  const Vector magnetic={{0,field,0}};
  const Vector orbit_angle=scaled(magnetic,-charge*length/energy);
  // Independent exact solution in the frame rotating with the orbit.
  // Pitch is constant in a uniform magnetic field, so chi and a(chi) are too.
  const double critical=EMASS/(LAMBDA_BAR*1e9);
  const double chi=gamma*beta*field*std::sqrt(1-n0[1]*n0[1])/critical;
  const double anomaly=PHYSTOOLS::BformFunction(chi);
  Vector relative;
  for (int i=0;i<3;++i) relative[i]=-charge*anomaly*length/energy*
    (gamma*magnetic[i]-(gamma-1)*dot(n0,magnetic)*n0[i]);
  const auto exact=rotation(rotation(s0,relative),orbit_angle);
  advance(particle,step,count,0,0,0,field);
  const double orbit_error=error(direction(particle),rotation(n0,orbit_angle));
  const double spin_error=error(spin(particle),exact);
  check(orbit_error<5e-6,"Secondary orbit obeys exact arbitrary-pitch cyclotron angle",orbit_error);
  check(std::abs(particle.getSpin().norm2()-1)<5e-12,"Imported BMT preserves spin norm",particle.getSpin().norm2()-1);
  check(std::abs(particle.unsigned_energy()/energy-1)<1e-5,"Uniform B does no work",particle.unsigned_energy()/energy-1);
  return spin_error;
}

double electric_case(int charge,int count) {
  const float energy=.051099906f,field=2e-11f;
  const double mass=EMASS, beta=std::sqrt(1-mass*mass/(double(energy)*energy));
  PAIR_PARTICLE p(1,-99,0,0,0,.3*beta,0,std::sqrt(.91)*beta,charge*energy);
  const auto n0=direction(p);p.initialize_imported_spin(TRIDVECTOR(n0[0],n0[1],n0[2]));
  const double momentum=std::sqrt(double(energy)*energy-mass*mass);
  const double px0=n0[0]*momentum,pz=n0[2]*momentum;
  const float step=static_cast<float>(.04/(double(field)*count));
  const double delta_px=charge*double(field)*double(step)*count;
  const double chi=std::abs(field)*std::sqrt(mass*mass+pz*pz)/mass/(EMASS/(LAMBDA_BAR*1e9));
  const double anomaly=PHYSTOOLS::BformFunction(chi);
  // Exact electric orbit: px=px0+q E t, pz constant. Integrate the scalar
  // BMT angle on that analytic orbit by independent, fine Simpson quadrature.
  const int intervals=4096;
  double integral=0;
  for (int i=0;i<=intervals;++i) {
    const double px=px0+delta_px*i/intervals;
    const double current_energy=std::sqrt(mass*mass+px*px+pz*pz);
    const double integrand=(anomaly+1/(current_energy/mass+1))*pz/(mass*current_energy);
    integral+=(i==0 || i==intervals ? 1 : (i%2 ? 4 : 2))*integrand;
  }
  const double angle=integral*delta_px/(3*intervals);
  const auto exact_spin=rotation(n0,Vector{{0,angle,0}});
  Vector exact_direction={{px0+delta_px,0,pz}};
  exact_direction=scaled(exact_direction,1/std::sqrt(dot(exact_direction,exact_direction)));
  advance(p,step,count,field,0,0,0);
  const double orbit_error=error(direction(p),exact_direction);
  check(orbit_error<5e-6,"Electric orbit agrees with linear momentum impulse",orbit_error);
  return error(spin(p),exact_spin);
}
}

int main() {
  for (int charge:{-1,1}) for (int direction_sign:{-1,1}) {
    PAIR_PARTICLE p(1,-99,0,0,0,.017452406f,0,direction_sign*.9998477f,charge*125.f);
    const auto n=direction(p);
    p.initialize_imported_spin(TRIDVECTOR(n[0],n[1],n[2]));
    const auto s=spin(p);const double h=p.imported_helicity();
    advance(p,100.f,100,0,0,0,0);
    check(error(spin(p),s)==0,"Zero-field imported spin is unchanged",error(spin(p),s));
    check(std::abs(p.imported_helicity()-h)<1e-12,"Zero-field imported helicity is unchanged",p.imported_helicity()-h);
    check(p.imported_max_chi()==0,"Zero-field chi is zero",p.imported_max_chi());
    const double coarse=magnetic_case(charge,direction_sign,4);
    const double fine=magnetic_case(charge,direction_sign,32);
    check(fine<2e-4 && fine<coarse/12,"Imported BMT converges to constant-field analytic spin",fine);
  }
  for (int charge:{-1,1}) {
    const double coarse=electric_case(charge,4),fine=electric_case(charge,64);
    check(fine<2e-5 && fine<coarse/20,"Electric BMT converges to analytic-orbit quadrature",fine);
  }
  // A recoil or external energy update must affect the next magnetic kick.
  // Reading a cached inverse of the original energy fails this comparison.
  PAIR_PARTICLE recoil(1,-99,0,0,0,0,.6f,.8f,125);
  recoil.setEnergy(50);
  const auto n0=direction(recoil);
  recoil.apply_magnetic_field_on_pair(1,1e12f,0,1e-5f);
  const double recoil_error=error(direction(recoil),rotation(n0,Vector{{0,-1e-5f*std::sqrt(double(1e12f))/50,0}}));
  check(recoil_error<2e-7,"Magnetic kick uses current post-recoil energy",recoil_error);
  bool rejected=false;
  try { recoil.initialize_imported_spin(TRIDVECTOR(0,0,1.1)); }
  catch (const std::domain_error&) { rejected=true; }
  check(rejected,"Unphysical input polarization is rejected");
  // Trace the Compton caller's ge/eg branch through the actual secondary
  // pusher. The field_pair source1 convention is (Ex,By)=(F,F), and source2
  // is (-F,F). A target co-moving with its own source must feel no force;
  // the opposing bunch gives the same focusing kick as its primary beam.
  for (int branch:{1,2}) for (int relative_charge:{-1,1}) {
    const auto source=feebl::compton::incident_for_native_branch(relative_charge*125,0.01,0,0,branch);
    const int direction=source.kinematics.direction;
    const float field=2e-4f, step=10.0f;
    RNDM random(771);
    PAIR_PARTICLE own(1,0,0,0,0,0,0,direction,source.charge_sign*125.f);
    PAIR_PARTICLE opposing=own;
    own.apply_final_half_step_fields(step,EMASS,direction*field,0,0,field,0,random);
    opposing.apply_final_half_step_fields(step,EMASS,-direction*field,0,0,field,0,random);
    float own_x,own_y,opposing_x,opposing_y;
    own.velocities(own_x,own_y);opposing.velocities(opposing_x,opposing_y);
    check(std::abs(own_x)<1e-10 && own_y==0,
          "Compton target charge/direction cancels its co-moving bunch field",own_x);
    const double expected=-relative_charge*double(field)*step/125;
    check(std::abs(opposing_x/expected-1)<2e-6 && opposing_y==0,
          "Compton charge conjugation reverses focusing to defocusing",opposing_x);
  }
  // Exact Compton recoils may reach rest. Native energies are float, so the
  // nearest representation of m is the rest state even if it rounds down.
  for (int charge:{-1,1}) {
    RNDM random(91);
    RNDM unchanged_sequence(91);
    const float rest_energy=static_cast<float>(EMASS);
    PAIR_PARTICLE at_rest(1,0,0,0,0,0,0,0,charge*rest_energy);
    at_rest.apply_initial_half_step_fields(1000,EMASS,0,0,0,0,random);
    std::vector<float> photons;
    at_rest.apply_full_step_fields(1000,EMASS,0,0,0,0,&photons,random);
    at_rest.apply_final_half_step_fields(1000,EMASS,0,0,0,0,0,random);
#ifdef PAIR_SYN
    // Each zero-probability radiation trial retains its legacy uniform draw.
    unchanged_sequence.rndm_synrad();
    unchanged_sequence.rndm_synrad();
#endif
    check(random.rndm()==unchanged_sequence.rndm(),
          "Rest propagation retains the zero-field radiation RNG sequence");
    check(photons.empty(),"Rest propagation emits no photons");
    float vx,vy;at_rest.velocities(vx,vy);
    check(vx==0 && vy==0 && at_rest.Zvelocity()==0,
          "Representable rest state stays finite with zero fields");
    at_rest.apply_final_half_step_fields(1000,EMASS,2e-6f,0,0,0,0,random);
    const double expected_energy=std::sqrt(EMASS*EMASS+1e-6);
    check(std::abs(at_rest.unsigned_energy()/expected_energy-1)<3e-7,
          "Electric kick accelerates a resting secondary on shell");
    PAIR_PARTICLE invalid(1,0,0,0,0,0,0,0,
                          charge*std::nextafter(rest_energy,0.0f));
    rejected=false;
    try { invalid.apply_final_half_step_fields(1000,EMASS,0,0,0,0,0,random); }
    catch (const std::domain_error&) { rejected=true; }
    check(rejected,"Genuinely submass secondary is rejected, not clipped");
  }
  return failures ? 1 : 0;
}
