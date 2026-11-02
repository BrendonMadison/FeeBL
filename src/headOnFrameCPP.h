#ifndef HEADONFRAME_SEEN
#define HEADONFRAME_SEEN

#include <cmath>

// THE HEAD-ON FRAME, AND HOW TO GET OUT OF IT.
//
// FEEBL collides in the frame where both beams travel along +-z. A crossing
// angle is not carried by the momenta at all: it is applied as a z-dependent
// transverse POSITION shear (PARTICLE_BEAM::set_angle_particles), identically
// for both beams. That is exactly the Lorentz-boosted frame construction --
// with beams at half-angle phi crossing in the x-z plane the lab momenta are
//
//     p1 = E (sin phi, 0, +cos phi)      p2 = E (sin phi, 0, -cos phi)
//
// (note BOTH carry +x), and a boost along x with beta = sin phi sends both px
// to zero simultaneously, leaving the crossing angle visible only as a tilt of
// the bunches. FEEBL's shear IS that tilt, to O(phi^2).
//
// So luminosity and every energy-domain result are computed in a legitimate
// frame and are valid. What is NOT valid is reading an ANGLE off the output and
// calling it a detector angle: every direction in the run is a head-on-frame
// direction. At ILC250 the half-angle is 7 mrad, the same order as the forward
// acceptance the pair background is binned against -- a leading correction
// there, not a refinement.
//
// THE TRAP THIS CLASS EXISTS TO AVOID.
//
// The intuitive fix -- "rotate the momenta by the crossing angle" -- is WRONG,
// and wrong silently. A rotation about y sends px -> px + phi*pz, whose sign
// follows the direction of travel: beam 1 gets +phi, beam 2 gets -phi. The
// boost sends px -> px + beta*E, and E > 0 always, so BOTH get +phi, which is
// what the lab geometry above actually says. Verified numerically at 7 mrad:
// the rotation gives beam 2 -6.999943 mrad where the boost gives +6.999943.
// Beam 2 and every backward-going secondary would come out mirrored.
//
// The boost direction is a unit vector in the transverse plane, so a crossing
// in y (or a mixed one) is handled by the same code rather than by a second
// implementation that could disagree with this one.
class HEAD_ON_FRAME
{
  double beta_;
  double gamma_;
  double nx_, ny_;      // unit vector along the crossing plane, transverse

 public:
  // half_crossing_angle_rad is phi, i.e. HALF the full crossing angle.
  // (nx, ny) need not be normalised; it is normalised here. A degenerate
  // direction disables the transformation rather than producing NaNs.
  HEAD_ON_FRAME(double half_crossing_angle_rad = 0.0,
                double nx = 1.0, double ny = 0.0)
    : beta_(0.0), gamma_(1.0), nx_(1.0), ny_(0.0)
    {
      const double n = std::sqrt(nx * nx + ny * ny);
      if (!std::isfinite(half_crossing_angle_rad) ||
          half_crossing_angle_rad == 0.0 || !(n > 0.0)) return;
      const double b = std::sin(half_crossing_angle_rad);
      if (!(std::fabs(b) < 1.0)) return;          // refuse a nonsense angle
      beta_  = b;
      gamma_ = 1.0 / std::sqrt(1.0 - b * b);
      nx_ = nx / n;
      ny_ = ny / n;
    }

  inline bool   active() const { return beta_ != 0.0; }
  inline double beta()   const { return beta_; }
  inline double gamma()  const { return gamma_; }
  inline double nx()     const { return nx_; }
  inline double ny()     const { return ny_; }

  // Head-on frame -> lab. pz is unchanged: the boost is transverse.
  inline void to_lab(double& E, double& px, double& py, double& /*pz*/) const
    { apply(E, px, py, beta_); }

  // Lab -> head-on frame.
  inline void to_headon(double& E, double& px, double& py, double& /*pz*/) const
    { apply(E, px, py, -beta_); }

 private:
  inline void apply(double& E, double& px, double& py, double b) const
    {
      if (beta_ == 0.0) return;          // exactly the identity, bit for bit
      const double pn  = px * nx_ + py * ny_;
      const double e   = gamma_ * (E  + b * pn);
      const double pn2 = gamma_ * (pn + b * E);
      const double d   = pn2 - pn;
      px += d * nx_;
      py += d * ny_;
      E   = e;
    }
};

#endif
