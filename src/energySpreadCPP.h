#ifndef ENERGYSPREAD_SEEN
#define ENERGYSPREAD_SEEN

#include "rndmCPP.h"

// The beam energy spread, as a multiplicative factor on a particle's energy.
//
// This used to live in GRID::spread_energy and was applied at COLLISION time,
// inside collide_ee, to the two energies entering the luminosity calculation.
// That made the spread a property of the luminosity spectrum rather than of the
// beam: particles travelled through the whole beam-beam interaction carrying
// exactly ebeam, and delta was identically zero everywhere upstream.
//
// It is now applied once, per particle, at beam setup -- so delta is a real
// beam property that the incoming maps, the deflection and the beamstrahlung
// all see. That is what IP dispersion needs to exist at all: a dispersive shear
// x += D*delta moves nothing on a monoenergetic bunch, however large D is.
//
// which_spread selects the distribution, and the numbering is inherited:
//   0  none
//   1  flat, over the interval that makes the RMS equal to `spread`
//   2  two peaks at +-spread
//   3  Gaussian of RMS `spread`
//
// The flat case uses dx = 3.4, xmin = -1.7, i.e. uniform on [-1.7, +1.7].
// That interval has RMS 1.7/sqrt(3) = 0.9815, so case 1 delivers an RMS of
// 0.98*spread rather than exactly spread. Kept as it is: changing it would
// silently move every result that has ever used which_espread=1, and the
// discrepancy belongs in the documentation rather than in a quiet correction.
inline float energy_spread_factor(int which_spread, float spread,
                                  RNDM& rndm_generator)
{
  const double dx = 3.4, xmin = -1.7;
  switch (which_spread)
    {
    case 0:
      return 1.0f;
    case 1:
      return (float)(1.0 + spread * (rndm_generator.rndm() * dx + xmin));
    case 2:
      return (rndm_generator.rndm() < 0.5) ? (float)(1.0 + spread)
                                           : (float)(1.0 - spread);
    case 3:
      return (float)(1.0 + spread * rndm_generator.gasdev());
    default:
      return 1.0f;
    }
}

#endif
