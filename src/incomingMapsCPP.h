#ifndef INCOMINGMAPS_SEEN
#define INCOMINGMAPS_SEEN

#include <string>
#include <vector>
#include "canonicalBunchCPP.h"
#include "gpgeom/maps.hpp"

// Incoming canonical optics maps, applied once before beam-beam tracking.
// Coordinates are (x, px, y, py, zeta, delta): positions in metres,
// normalized canonical momenta and fractional momentum deviation.
// CANONICAL_BUNCH converts the legacy direction variables to this basis.
// zeta is beam-local; the longitudinal sign must be declared explicitly.
//
// CRAB_LINEAR_IP_TILT imposes x += t*zeta and delta -= t*px. It is an
// ideal symplectic IP shear, with no cavity or lattice errors.
// CRAB_RF_HARMONICS supplies the cavity momentum kick only. A cavity kick
// needs transport to the IP to become a position tilt; this class does
// not provide that transport. The waist map likewise represents incoming
// optics, not an additional beam-beam force.
//
// The input-card dispersion is horizontal. Vertical dispersion and
// longitudinal/chromatic-waist schemes are outside the current study.
enum CrabMode  { CRAB_NONE = 0, CRAB_LINEAR_IP_TILT = 1, CRAB_RF_HARMONICS = 2 };
enum WaistMode { WAIST_NONE = 0, WAIST_IDEAL_LOCAL = 1 };

struct IncomingMapConfig
{
  int    crab_mode;
  int    crab_plane;              // 0 = x, 1 = y
  double crab_frequency_Hz;
  double crab_phase_rad;
  double crab_target_tilt;        // dx/dzeta at the IP; exactly -1.0 = derive
  int    crab_harmonic_n;         // second harmonic for the cubic linearizer
  int    include_panofsky_wenzel; // must be 1 for a canonical map
  int    waist_mode;
  double waist_fraction;
  double ip_dispersion_m;         // IP dispersion D, metres per unit delta
  double full_crossing_angle_rad;
  double symplecticity_tolerance;
  int    longitudinal_sign;       // 0 = undeclared

  IncomingMapConfig()
    : crab_mode(CRAB_NONE), crab_plane(0), crab_frequency_Hz(0.0),
      crab_phase_rad(0.0), crab_target_tilt(-1.0), crab_harmonic_n(2),
      include_panofsky_wenzel(1), waist_mode(WAIST_NONE), waist_fraction(1.0),
      ip_dispersion_m(0.0),
      full_crossing_angle_rad(0.0), symplecticity_tolerance(1e-11),
      longitudinal_sign(0) {}

  inline bool active() const
    { return crab_mode != CRAB_NONE || waist_mode != WAIST_NONE
             || ip_dispersion_m != 0.0; }
};

class INCOMING_MAP_CHAIN
{
  IncomingMapConfig cfg_;
  std::vector<gpgeom::Harmonic> harmonics_;
  bool   have_crab_, have_waist_;
  bool   have_dispersion_;    // IP dispersion for monochromatization
  bool   have_tilt_;          // CRAB_LINEAR_IP_TILT: a tilt AT the IP
  double tilt_slope_;         // dx/dzeta imposed at the IP
  double k_;
  std::string error_;

 public:

  INCOMING_MAP_CHAIN() : have_crab_(false), have_waist_(false),
                        have_dispersion_(false),
                        have_tilt_(false), tilt_slope_(0.0), k_(0.0) {}

  inline const std::string& error() const { return error_; }
  inline bool active() const
    { return have_crab_ || have_tilt_ || have_waist_ || have_dispersion_; }
  inline double ip_dispersion() const { return have_dispersion_ ? cfg_.ip_dispersion_m : 0.0; }

  // Returns false and sets error() rather than proceeding on a bad
  // configuration. Every refusal here is a case where continuing would produce
  // a plausible wrong answer rather than a visible failure.
  bool configure(const IncomingMapConfig& c);

  // Modify all particles in place, in the order dispersion, crab, waist.
  // These maps do not generally commute. The bunch already has beam-local
  // zeta, so no second longitudinal-sign conversion is applied here.
  // Return false and set why if its declared convention is incompatible.
  bool apply(CANONICAL_BUNCH& b, std::string& why) const;

  // Sample the Jacobian and check the symplectic defect
  // against the configured tolerance.
  bool verify_symplectic(double& worst_defect) const;

  // Kick as a function of LAB z offset from the reference plane, in metres.
  // The sign IS applied here, because the argument is a lab coordinate rather
  // than a beam-local zeta -- this is the conversion apply() must not repeat.
  double kick_at(double z_lab_offset_m) const;

  // The resolved IP tilt dx/dzeta, after any derivation from the crossing
  // angle; 0 if this chain imposes no tilt. Callers need it to check that the
  // longitudinal slicing can actually resolve the tilt -- see
  // FEEBL::apply_incoming_maps.
  inline double ip_tilt() const { return have_tilt_ ? tilt_slope_ : 0.0; }
  inline int    tilt_plane() const { return cfg_.crab_plane; }
};

#endif
