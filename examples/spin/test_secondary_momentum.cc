#include "secondaryMomentumCPP.h"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
  using feebl::secondary_transport::Momentum;
  constexpr double mass=.00051099906;
  Momentum p;
  p.initialize(mass,1e-14,0,0);
  if(p.energy()!=mass || !(p.kinetic_energy()>0)) return 1;
  const double energy=p.energy(),momentum=p.component(0);
  for(int i=0;i<1000;++i) p.electric_impulse(0,0);
  if(p.energy()!=energy || p.component(0)!=momentum) return 2;
  p.radiate(p.kinetic_energy());
  if(p.energy()!=mass || p.magnitude()!=0) return 3;
  p.initialize(mass,.3,.4,125);
  const double original_energy=p.energy(),original_momentum=p.magnitude();
  for(int i=0;i<10000;++i) p.magnetic_rotation(.001,.002);
  if(p.energy()!=original_energy || std::abs(p.magnitude()/original_momentum-1)>1e-11) return 4;
  bool rejected=false;
  try { p.radiate(p.energy()); } catch (...) { rejected=true; }
  if(!rejected || p.energy()!=original_energy) return 5;
  rejected=false;
  try { p.electric_impulse(std::numeric_limits<double>::infinity(),0); } catch (...) { rejected=true; }
  if(!rejected) return 6;
  std::cout<<"PASS canonical momentum arithmetic; full PAIR_PARTICLE and collision validation remain separate\n";
}
