#ifndef COLLISIONGEOMETRY_SEEN
#define COLLISIONGEOMETRY_SEEN

#include <cmath>
#include <string>
#include "canonicalBunchCPP.h"

// Beam frames for rigid-overlap geometry, with an explicit reference plane
// for each beam's zeta. The caller must declare a half- or full-angle input;
// inferring this convention would introduce a factor-of-two ambiguity.
// The native angle_x input separately describes the per-beam position shear
// dx/dz = +angle_x. See doc/CONVENTIONS.md for the coordinate conversion.

enum AngleConvention { ANGLE_UNDECLARED = 0, ANGLE_HALF = 1, ANGLE_FULL = 2 };

// Rigid orthonormal map between beam-local and laboratory coordinates. Kept
// deliberately thin: at the small angles these machines use, the transverse
// geometry is a rotation in the crossing plane plus the longitudinal sign.
struct BeamFrame
{
  double half_angle_rad;   // signed: +theta/2 for one beam, -theta/2 for the other
  int    crossing_plane;   // 0 = x, 1 = y
  int    longitudinal_sign;// +1 travelling toward +z, -1 otherwise
  double z_ref_nm;         // this bunch's reference plane on the shared axis
  bool   declared;

  BeamFrame()
    : half_angle_rad(0.0), crossing_plane(0), longitudinal_sign(0),
      z_ref_nm(0.0), declared(false) {}

  inline LongitudinalConvention longitudinal() const
    { return declared ? LongitudinalConvention(longitudinal_sign, z_ref_nm)
                      : LongitudinalConvention(); }
};

class COLLISION_GEOMETRY
{
  AngleConvention conv_;
  double          angle_in_;      // as given, in the declared convention
  int             plane_;
  BeamFrame       f_[2];
  std::string     error_;

 public:

  COLLISION_GEOMETRY() : conv_(ANGLE_UNDECLARED), angle_in_(0.0), plane_(0) {}

  inline const std::string& error() const { return error_; }
  inline const BeamFrame& frame(int beam) const { return f_[beam-1]; }

  // The full crossing angle, whatever convention it was supplied in. Every
  // downstream formula uses this, so the conversion happens exactly once.
  inline double full_crossing_angle() const
    { return (conv_ == ANGLE_HALF) ? 2.0 * angle_in_ : angle_in_; }
  inline double half_crossing_angle() const
    { return 0.5 * full_crossing_angle(); }

  // Refuses an undeclared convention rather than guessing (see above).
  bool configure(int convention, double angle, int crossing_plane,
                 double z_ref_nm_beam1 = 0.0, double z_ref_nm_beam2 = 0.0)
    {
      error_.clear();
      if (convention != ANGLE_HALF && convention != ANGLE_FULL)
        { error_ = "geometry_angle_convention must be declared as half (1) or "
                   "full (2); the two differ by a factor of two and a silent "
                   "choice gives a plausible wrong luminosity";
          conv_ = ANGLE_UNDECLARED; return false; }
      if (angle < 0.0)
        { error_ = "crossing angle must be non-negative"; return false; }
      if (crossing_plane != 0 && crossing_plane != 1)
        { error_ = "crossing plane must be 0 (x) or 1 (y)"; return false; }

      conv_ = (AngleConvention)convention;
      angle_in_ = angle;
      plane_ = crossing_plane;

      const double h = half_crossing_angle();
      // Beams counter-propagate: opposite longitudinal sign, opposite tilt.
      f_[0].half_angle_rad = +h; f_[0].crossing_plane = plane_;
      f_[0].longitudinal_sign = +1; f_[0].z_ref_nm = z_ref_nm_beam1;
      f_[0].declared = true;
      f_[1].half_angle_rad = -h; f_[1].crossing_plane = plane_;
      f_[1].longitudinal_sign = -1; f_[1].z_ref_nm = z_ref_nm_beam2;
      f_[1].declared = true;
      return true;
    }

  inline bool declared() const { return conv_ != ANGLE_UNDECLARED; }
};

// ---- analytic rigid-bunch luminosity, the Level 3 reference -------------
//
// For rigid Gaussian bunches crossing at a full angle theta, the geometric
// reduction is the Piwinski factor
//
//     Phi     = sigma_z tan(theta/2) / sigma_x
//     R_geom  = 1 / sqrt(1 + Phi^2)
//
// and the head-on luminosity is N1 N2 f / (4 pi sigma_x sigma_y). The hourglass
// factor is a separate reduction when sigma_z is not small against beta*.
//
// These are the numbers a rigid FEEBL run must reproduce with the beam-beam
// force switched off; any disagreement is geometry or convention, not physics.
struct RigidLuminosity
{
  double piwinski;
  double r_geom;
  double hourglass;
  double l_headon;      // per crossing, m^-2
  double l_total;       // l_headon * r_geom * hourglass
};

inline double piwinski_angle(double sigma_z_m, double sigma_x_m,
                             double full_angle_rad)
{
  if (sigma_x_m <= 0.0) return 0.0;
  return sigma_z_m * std::tan(0.5 * full_angle_rad) / sigma_x_m;
}

// Hourglass reduction for equal beta functions, evaluated by quadrature of
// 1/sqrt(1+(z/beta)^2) weighted by the longitudinal overlap. The closed form
// involves a modified Bessel function; a short Simpson rule is accurate enough
// here and keeps the dependency surface at zero.
inline double hourglass_factor(double sigma_z_m, double beta_y_m, int n = 4001)
{
  if (beta_y_m <= 0.0 || sigma_z_m <= 0.0) return 1.0;
  const double zmax = 6.0 * sigma_z_m;
  const double h = 2.0 * zmax / (n - 1);
  double sum = 0.0, norm = 0.0;
  for (int i = 0; i < n; i++)
    {
      const double z = -zmax + i * h;
      const double w = (i == 0 || i == n-1) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
      const double g = std::exp(-z*z / (sigma_z_m*sigma_z_m));   // overlap density
      const double r = 1.0 / std::sqrt(1.0 + (z/beta_y_m)*(z/beta_y_m));
      sum  += w * g * r;
      norm += w * g;
    }
  return (norm > 0.0) ? sum / norm : 1.0;
}

inline RigidLuminosity rigid_luminosity(double n1, double n2, double freq,
                                        double sigma_x_m, double sigma_y_m,
                                        double sigma_z_m, double beta_y_m,
                                        double full_angle_rad)
{
  RigidLuminosity r;
  r.piwinski = piwinski_angle(sigma_z_m, sigma_x_m, full_angle_rad);
  r.r_geom   = 1.0 / std::sqrt(1.0 + r.piwinski * r.piwinski);
  r.hourglass= hourglass_factor(sigma_z_m, beta_y_m);
  const double denom = 4.0 * M_PI * sigma_x_m * sigma_y_m;
  r.l_headon = (denom > 0.0) ? n1 * n2 * freq / denom : 0.0;
  r.l_total  = r.l_headon * r.r_geom * r.hourglass;
  return r;
}

#endif
