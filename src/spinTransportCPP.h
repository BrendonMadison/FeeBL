#ifndef FEEBL_SPIN_TRANSPORT_CPP_H
#define FEEBL_SPIN_TRANSPORT_CPP_H

#include "mathematicalEntities.h"
#include <cmath>
#include <stdexcept>

namespace feebl { namespace spin_transport {

// Exact rotation for a frozen angular velocity. No renormalization is used:
// conservation of the spin length remains an independent integration check.
inline void rotate(TRIDVECTOR& spin, const TRIDVECTOR& angle) {
  const double theta2 = angle(0)*angle(0)+angle(1)*angle(1)+angle(2)*angle(2);
  if (theta2 == 0.0) return;
  if (!std::isfinite(theta2)) throw std::domain_error("Nonfinite spin rotation");
  const double theta=std::sqrt(theta2), inverse=1.0/theta;
  const double nx=angle(0)*inverse, ny=angle(1)*inverse, nz=angle(2)*inverse;
  const double c=std::cos(theta), s=std::sin(theta);
  const double sx=spin(0), sy=spin(1), sz=spin(2), projection=nx*sx+ny*sy+nz*sz;
  spin(0)=sx*c+(ny*sz-nz*sy)*s+nx*projection*(1.0-c);
  spin(1)=sy*c+(nz*sx-nx*sz)*s+ny*projection*(1.0-c);
  spin(2)=sz*c+(nx*sy-ny*sx)*s+nz*projection*(1.0-c);
}

// Full finite-mass Thomas-BMT equation for rest-frame polarization expressed
// on fixed laboratory axes. E and cB are in GV/nm, energy in GeV, c*dt in nm.
// The signed force convention is d(beta)/ds = q*(E+beta x B)/energy,
// with the electric acceleration's longitudinal energy term handled by the
// orbit integrator. The same q and field signs must be used by orbit and spin.
inline TRIDVECTOR bmt_angle(const TRIDVECTOR& beta, const TRIDVECTOR& electric,
                           const TRIDVECTOR& magnetic, double gamma,
                           double energy_GeV, double charge, double ds_nm,
                           double anomaly) {
  const double beta_dot_B=beta(0)*magnetic(0)+beta(1)*magnetic(1)+beta(2)*magnetic(2);
  const TRIDVECTOR beta_cross_E(beta(1)*electric(2)-beta(2)*electric(1),
                              beta(2)*electric(0)-beta(0)*electric(2),
                              beta(0)*electric(1)-beta(1)*electric(0));
  TRIDVECTOR angle;
  for (int axis=0; axis<3; ++axis)
    angle(axis)=-charge*ds_nm/energy_GeV*((1.0+anomaly*gamma)*magnetic(axis)
      -anomaly*gamma*gamma/(gamma+1.0)*beta_dot_B*beta(axis)
      -gamma*(anomaly+1.0/(gamma+1.0))*beta_cross_E(axis));
  return angle;
}

} }
#endif
