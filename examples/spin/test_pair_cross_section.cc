#include "pairCrossSectionCPP.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
int failures=0;
void check(bool ok,const char* label,double error=0) {
  std::printf("%s: %s (%.12g)\n",label,ok?"PASS":"FAIL",error);failures+=!ok;
}
// Independent beta-form evaluation of Gould & Schreder Eq. (1), with extended
// precision. Production uses inverse gamma squared and a stable logarithm.
long double reference(long double product) {
  const long double rho=static_cast<long double>(EMASS*EMASS)/product;
  const long double beta=std::sqrt(1-rho),pi=std::acos(-1.L),radius=2.81794092e-15L;
  return pi*radius*radius/2*(1-beta*beta)*
    ((3-beta*beta*beta*beta)*std::log((1+beta)/(1-beta))-2*beta*(2-beta*beta));
}
}
int main() {
  using feebl::pair_production::breit_wheeler_m2;
  const double threshold=EMASS*EMASS;
  check(breit_wheeler_m2(0)==0 && breit_wheeler_m2(threshold)==0 &&
        breit_wheeler_m2(.99*threshold)==0,"Pair production vanishes at and below threshold");
  for (double beta:{.001,.01,.1,.5,.9,.99,.9999}) {
    const double product=threshold/(1-beta*beta);
    const double expected=static_cast<double>(reference(product));
    const double relative=breit_wheeler_m2(product)/expected-1;
    check(std::abs(relative)<2e-10,"Cross section matches independent beta-form oracle",relative);
  }
  const double known=1.381899451234991e-29;
  const double fixed_error=breit_wheeler_m2(threshold/.75)/known-1;
  check(std::abs(fixed_error)<3e-14,"Absolute cross section at beta=0.5 in m^2",fixed_error);
  for (double ratio:{2.,10.,1000.}) {
    const double expected=(EMASS/MUMASS)*(EMASS/MUMASS);
    const double actual=breit_wheeler_m2(ratio*MUMASS*MUMASS,MUMASS)/breit_wheeler_m2(ratio*threshold);
    check(std::abs(actual/expected-1)<1e-13,"Lepton mass scaling at fixed gamma",actual/expected-1);
  }
  const double rho=1e-24;
  const double asymptotic=PI*RE*RE*rho*(std::log(4/rho)-1);
  const double high_energy=breit_wheeler_m2(threshold/rho);
  check(std::isfinite(high_energy) && std::abs(high_energy/asymptotic-1)<1e-13,
        "High-energy logarithm remains finite when beta rounds to one",high_energy/asymptotic-1);
  for (double bad:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;
    try { (void)breit_wheeler_m2(bad); } catch (const std::domain_error&) { rejected=true; }
    check(rejected,"Invalid pair-production invariant rejected");
  }
  return failures ? 1 : 0;
}
