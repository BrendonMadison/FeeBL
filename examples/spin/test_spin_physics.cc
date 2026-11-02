#include "particlesCPP.h"
#include "physicalTools.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
using V=std::array<double,3>;
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V as_v(const TRIDVECTOR& a){return {a.getComponent(0),a.getComponent(1),a.getComponent(2)};}
int failures=0;
void check(bool ok,const char* name,double err){printf("%s: %s (error %.12g)\n",name,ok?"PASS":"FAIL",err);failures+=!ok;}
struct RadiationMean{double energy,energy_sem,spin,spin_sem;};
RadiationMean fixed_field_radiation(float chi0,bool with_spin,int steps) {
  const long count=50000;const float initial_energy=250;
  RNDM rng(918273);double e=0,e2=0,s=0,s2=0;
  // Same total field exposure, regrouped into 1 or 64 external intervals.
  // chi(E)=chi0*E/E0 and curvature(E)=curvature0*E0/E follow the
  // Lorentz-force and quantum-parameter definitions, independently of
  // the radiation wrapper's internal subdivision bookkeeping.
  const double total_bend=10./(CONST1*(initial_energy/EMASS));
  for(long i=0;i<count;i++) {
    float energy=initial_energy;TRIDVECTOR polar(0,0,1);std::vector<float> photons;
    for(int j=0;j<steps;j++) {
      const float chi=chi0*energy/initial_energy;
      const float bend=total_bend/steps*initial_energy/energy;
      if(with_spin)energy=PHYSTOOLS::synrad_spin_flip(chi,energy,TRIDVECTOR(1,0,0),TRIDVECTOR(0,1,0),TRIDVECTOR(0,0,1),polar,bend,photons,rng);
      else energy=PHYSTOOLS::synrad_no_spin_flip(chi,energy,bend,photons,rng);
    }
    const double f=energy/initial_energy;e+=f;e2+=f*f;s+=polar(2);s2+=polar(2)*polar(2);
  }
  return {e/count,std::sqrt((e2/count-(e/count)*(e/count))/count),s/count,std::sqrt(std::max(0.,s2/count-(s/count)*(s/count))/count)};
}
}

int main(){
  // CAIN 2.1e Sec.5.8.3: e1 follows acceleration, e2 = e3 x e1.
  // This oracle uses the real primary orbit kick, independent of spin code.
  for(int direction: {-1,1}) {
    PARTICLE p;p.setParticle(0,0,0,0,0,250);
    TRIVECTOR field(1e-6f,2e-6f,0);p.advanceDueToEBfield(field,100,1);
    float vx,vy;p.velocities(vx,vy);
    TRIDVECTOR e1,e2,e3;
    PHYSTOOLS::referenceSpin(0,0,e1,e2,e3,TRIDVECTOR(field),TRIDVECTOR(direction*field(1),-direction*field(0),0),-1,direction);
    double alignment=(e1(0)*vx+e1(1)*vy)/std::hypot(vx,vy);
    check(std::abs(alignment-1)<1e-6,"ST basis aligned with orbit acceleration",std::abs(alignment-1));
    for(double tx: {0.,.001,.15}) {
      const double ty=-.08*(tx/.15);
      PHYSTOOLS::referenceSpin(tx,ty,e1,e2,e3,TRIDVECTOR(1e-6,2e-6,3e-7),TRIDVECTOR(-4e-7,5e-7,2e-6),-1,direction);
      const V n1=as_v(e1),n2=as_v(e2),n3=as_v(e3);const V handed=cross(n3,n1);
      double error=std::max(std::abs(dot(n1,n3)),std::abs(dot(n2,n3)));
      for(const V n: {n1,n2,n3})error=std::max(error,std::abs(dot(n,n)-1));
      for(int k=0;k<3;k++)error=std::max(error,std::abs(handed[k]-n2[k]));
      check(error<2e-14,"ST basis orthonormal and right handed",error);
      PHYSTOOLS::referenceSpin(tx,ty,e1,e2,e3,TRIDVECTOR(0,0,0),TRIDVECTOR(0,0,0),-1,direction);
      error=std::max(std::abs(dot(as_v(e1),as_v(e3))),std::abs(e2.norm2()-1));
      check(error<2e-14,"Zero-field spin basis finite and orthonormal",error);
    }
    // Independent full-vector BMT form, not the production B_parallel split.
    // beta=1 is the primary model; finite-mass corrections are not claimed.
    const float vx0=.15f,vy0=-.08f;const V n={vx0,vy0,direction*std::sqrt(1.-vx0*(double)vx0-vy0*(double)vy0)};
    const V E={1e-6,2e-6,3e-7},B={-4e-7,5e-7,2e-6},s0={.3,.4,std::sqrt(.75)};
    const double g=250./EMASS,a=1.15965221e-3;
    V omega;const V nxE=cross(n,E);
    for(int k=0;k<3;k++)omega[k]=((1+g*a)*B[k]-(g-1)*a*dot(n,B)*n[k]-(g*a+g/(g+1))*nxE[k])*1000/250;
    const double angle=std::sqrt(dot(omega,omega));V axis=omega;for(double& c:axis)c/=angle;
    const V axs=cross(axis,s0);V exact;for(int k=0;k<3;k++)exact[k]=s0[k]*std::cos(angle)+axs[k]*std::sin(angle)+axis[k]*dot(axis,s0)*(1-std::cos(angle));
    PARTICLE_WITH_SPIN spin;TRIDVECTOR s(s0[0],s0[1],s0[2]);spin.setParticle(0,0,0,vx0,vy0,250,&s);spin.setUps(0);
    spin.rotateBMT(TRIDVECTOR(E[0],E[1],E[2]),TRIDVECTOR(B[0],B[1],B[2]),-1,1000,direction);
    double error=0;for(int k=0;k<3;k++)error=std::max(error,std::abs(spin.getSpin().getComponent(k)-exact[k]));
    check(error<2e-14,"Oblique BMT equals full-vector analytic rotation",error);
  }
  // Invalid directions must fail explicitly, not inject NaNs into spectra.
  bool rejected=false;TRIDVECTOR e1,e2,e3;
  try{PHYSTOOLS::referenceSpin(1.,.1,e1,e2,e3,TRIDVECTOR(),TRIDVECTOR(),-1,1);}catch(const std::domain_error&){rejected=true;}
  check(rejected,"Unphysical transverse direction rejected",rejected?0:1);
  // Recoil must not leave chi frozen across an interval containing many
  // photons. A six-SEM gate includes an absolute .002 finite-step allowance;
  // the old implementation differs by .014--.024 in energy and fails it.
  for(float chi: {.1f,1.f})for(bool with_spin: {false,true}) {
    const RadiationMean coarse=fixed_field_radiation(chi,with_spin,1),fine=fixed_field_radiation(chi,with_spin,64);
    double error=std::abs(coarse.energy-fine.energy);
    double tolerance=6*std::hypot(coarse.energy_sem,fine.energy_sem)+.002;
    check(error<tolerance,"Radiative recoil energy independent of interval grouping",error);
    if(with_spin){error=std::abs(coarse.spin-fine.spin);tolerance=6*std::hypot(coarse.spin_sem,fine.spin_sem)+.002;
      check(error<tolerance,"Radiative recoil spin independent of interval grouping",error);}
  }
  return failures?1:0;
}
