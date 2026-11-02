#ifndef FEEBL_PAIR_CROSS_SECTION_CPP_H
#define FEEBL_PAIR_CROSS_SECTION_CPP_H

#include "physconst.h"
#include "mathconst.h"
#include <cmath>
#include <stdexcept>

namespace feebl { namespace pair_production {

// Unpolarized tree-level gamma gamma -> lepton+ lepton- total cross section.
// energy_product_GeV2 = E_gamma1*E_gamma2 = s/4 for head-on photons.
// The returned area is m^2; luminosity and event sampling are caller concerns.
// Gould & Schreder, Phys. Rev. 155, 1404 (1967), Eq. (1),
// https://doi.org/10.1103/PhysRev.155.1404 .
inline double breit_wheeler_m2(double energy_product_GeV2,double mass_GeV=EMASS) {
  if (!std::isfinite(energy_product_GeV2) || energy_product_GeV2<0 ||
      !std::isfinite(mass_GeV) || !(mass_GeV>0))
    throw std::domain_error("Breit-Wheeler requires finite nonnegative energy product and positive mass");
  const double mass2=mass_GeV*mass_GeV;
  if (energy_product_GeV2<=mass2) return 0.0;
  const double inverse_gamma2=mass2/energy_product_GeV2;
  if (inverse_gamma2==0.0) return 0.0; // Underflowed, vanishing high-energy limit.
  const double beta=std::sqrt(1.0-inverse_gamma2);
  // (1-beta)(1+beta)=1/gamma^2: avoid subtracting beta from one at
  // high energy. The algebraically equivalent bracket is finite at threshold.
  const double logarithm=2*std::log1p(beta)-std::log(inverse_gamma2);
  const double bracket=(2+2*inverse_gamma2-inverse_gamma2*inverse_gamma2)*logarithm
                       -2*(1+inverse_gamma2)*beta;
  const double classical_radius_m=RE*EMASS/mass_GeV;
  return 0.5*PI*classical_radius_m*classical_radius_m*inverse_gamma2*bracket;
}

} }
#endif
