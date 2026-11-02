#include "incomingMapsCPP.h"
#include <cmath>
#include <memory>
#include <utility>
#include <cstdio>

namespace {
const double C_LIGHT = 299792458.0;
}

bool INCOMING_MAP_CHAIN::configure(const IncomingMapConfig& c)
{
  cfg_ = c;
  have_crab_ = have_waist_ = have_tilt_ = false;
  tilt_slope_ = 0.0;
  harmonics_.clear();
  error_.clear();

  if (!c.active()) return true;

  // Enum-like keys were previously tested for "nonzero", so crab_mode=3 counted
  // as active but matched neither branch and became a silently logged no-op,
  // and any nonzero waist_mode enabled the ideal map. A deck typo must not
  // quietly disable the physics it asked for.
  if (c.crab_mode != CRAB_NONE && c.crab_mode != CRAB_LINEAR_IP_TILT &&
      c.crab_mode != CRAB_RF_HARMONICS)
    { error_ = "crab_mode must be 0 (off), 1 (IP position tilt) or 2 (RF "
               "harmonics); any other value would be a silent no-op";
      return false; }
  if (c.waist_mode != WAIST_NONE && c.waist_mode != WAIST_IDEAL_LOCAL)
    { error_ = "waist_mode must be 0 (off) or 1 (ideal local)"; return false; }

  // Only crab_plane==1 was treated as vertical; every other value silently
  // became horizontal, so crab_plane=2 would tilt the wrong plane.
  if (c.crab_plane != 0 && c.crab_plane != 1)
    { error_ = "crab_plane must be 0 (horizontal) or 1 (vertical)"; return false; }

  // NaN passes every ordinary comparison, so a non-finite input would flow
  // through the guards below and into the beam.
  if (!std::isfinite(c.crab_frequency_Hz) || !std::isfinite(c.crab_phase_rad) ||
      !std::isfinite(c.crab_target_tilt)  || !std::isfinite(c.waist_fraction) ||
      !std::isfinite(c.ip_dispersion_m)   ||
      !std::isfinite(c.full_crossing_angle_rad) ||
      !std::isfinite(c.symplecticity_tolerance))
    { error_ = "a crab/waist parameter is not finite (NaN or inf)"; return false; }

  // C5: without a declared longitudinal sign there is no defensible zeta, and
  // sin(k*zeta) changes sign with it. Refuse rather than pick one.
  if (c.longitudinal_sign != +1 && c.longitudinal_sign != -1)
    { error_ = "crab_longitudinal_sign must be +1 or -1; FEEBL has no per-beam "
               "zeta to fall back on (CONVENTIONS.md C5)"; return false; }

  if (c.crab_mode != CRAB_NONE)
    {
      if (!c.include_panofsky_wenzel)
        { error_ = "crab_include_panofsky_wenzel=0 gives a NON-canonical map; "
                   "allowed only for explicit diagnostic tests, not a physics run";
          return false; }
      // Only the cavity needs a frequency: an IP tilt is a pure geometric
      // shear with no RF in it.
      if (c.crab_mode == CRAB_RF_HARMONICS && !(c.crab_frequency_Hz > 0.0))
        { error_ = "crab_frequency must be positive for crab_mode=2 (RF harmonics)";
          return false; }

      // "Derive" is signalled by exactly the registered default, not by any
      // negative value: a crossing-angle sign convention can perfectly well
      // call for a NEGATIVE tilt, and treating every negative number as
      // "derive" made that impossible to ask for.
      double slope = c.crab_target_tilt;
      if (slope == -1.0)
        {
          // Derive the tilt that restores head-on overlap.
          //
          // The magnitude is theta/2, but the SIGN is per beam and is not
          // optional. FEEBL applies the crossing angle as a per-slice position
          // shear with an identical formula for both beams
          // (PARTICLE_BEAM::set_angle_particles), but because zeta_2 = -zeta_1
          // at the same lab z the two bunches acquire OPPOSITE lab-frame
          // tilts. An earlier version of this comment said they acquire the
          // SAME tilt; the formula below survived that because a second sign
          // cancels it in canonicalBunchCPP.h, but the explanation was wrong
          // and it is the explanation the next person reasons from.
          // Cancelling them therefore needs opposite per-beam tilts:
          //
          //     t_b = -longitudinal_sign_b * theta/2
          //
          // Measured on the ILC deck at 30 mrad: opposite tilts recover 13.55x
          // (5.0% -> 68.4% of head-on), while the SAME tilt on both beams gives
          // 0.75x and the reversed pair 0.37x. Getting this wrong makes crab
          // crossing worse than not crabbing at all.
          if (!(c.full_crossing_angle_rad > 0.0))
            { error_ = "crab_target_tilt not given and no crossing angle to "
                       "derive it from"; return false; }
          slope = -c.longitudinal_sign * 0.5 * c.full_crossing_angle_rad;
        }

      k_ = 2.0 * M_PI * c.crab_frequency_Hz / C_LIGHT;

      if (c.crab_mode == CRAB_LINEAR_IP_TILT)
        {
          // The tilt is imposed directly at the IP; no cavity, no transport.
          tilt_slope_ = slope;
          have_tilt_  = true;
        }
      else if (c.crab_mode == CRAB_RF_HARMONICS)
        {
          if (c.crab_harmonic_n < 2)
            { error_ = "crab_harmonic_n must be >= 2 for the cubic linearizer";
              return false; }
          // Two harmonics chosen so K'(0) = slope and K'''(0) = 0, i.e. the
          // leading nonlinearity of a single sine is cancelled.
          // The linearizer solves for K'(0) = slope at ZERO phase, but an
          // arbitrary common phase is accepted. With
          //     K(z) = A1 sin(kz+phi) + An sin(nkz+phi)
          // the delivered slope is cos(phi) * slope, so a nonzero phase used to
          // quietly under-deliver the requested tilt. Solve for the slope that
          // comes out right, and refuse where the correction blows up.
          const double cphi = std::cos(c.crab_phase_rad);
          if (std::fabs(cphi) < 0.1)
            { error_ = "crab_phase is within ~6 degrees of +-pi/2, where the "
                       "cavity delivers almost no transverse slope";
              return false; }
          const std::array<double,2> a =
            gpgeom::CrabRFCavityMap::cubic_linearizer(slope / cphi, k_,
                                                      c.crab_harmonic_n);
          harmonics_.push_back(gpgeom::Harmonic{1, a[0], c.crab_phase_rad});
          harmonics_.push_back(gpgeom::Harmonic{c.crab_harmonic_n, a[1], c.crab_phase_rad});
          have_crab_ = true;
        }
    }

  if (c.waist_mode != WAIST_NONE)
    {
      if (!(c.full_crossing_angle_rad > 0.0))
        { error_ = "waist_mode requires a non-zero crossing angle: the ideal "
                   "map is g(x) = -fraction*x/tan(theta)"; return false; }
      if (c.waist_fraction < 0.0)
        { error_ = "waist_fraction must be >= 0"; return false; }
      have_waist_ = true;
    }

  // IP dispersion is independent of BOTH the crab and the waist: a beam may be
  // dispersive with neither. So this sits at function scope, outside both
  // blocks above.
  //
  // It was originally written inside the waist block, where it was dead for
  // every deck that did not also enable a crab waist -- and silently so, since
  // an unset have_dispersion_ just makes apply() skip the shear. The unit test
  // could not see it, because that exercises IPDispersionMap directly rather
  // than through the chain. What caught it was running FEEBL with D = 1 m, a
  // displacement of thousands of sigma_x, and getting byte-identical output.
  have_dispersion_ = (c.ip_dispersion_m != 0.0);

  return true;
}

bool INCOMING_MAP_CHAIN::apply(CANONICAL_BUNCH& b, std::string& why) const
{
  why.clear();
  if (!active()) return true;

  // The bunch must carry the same longitudinal convention the chain was
  // configured with, or its zeta means something other than what the map
  // assumes. Refuse instead of tilting the beam the wrong way (C5).
  if (!b.longitudinal().declared)
    { why = "bunch has no declared longitudinal convention (C5)"; return false; }
  if (b.longitudinal().sign != cfg_.longitudinal_sign)
    { why = "bunch longitudinal sign disagrees with crab_longitudinal_sign";
      return false; }

  const bool vertical = (cfg_.crab_plane == 1);

  // Construct only the active maps: they validate their own arguments and throw,
  // so building a crab map with k_ = 0 because the crab is off would abort.
  std::unique_ptr<gpgeom::CrabRFCavityMap> crab;
  std::unique_ptr<gpgeom::CrabWaistMap>    waist;
  if (have_crab_)  crab.reset(new gpgeom::CrabRFCavityMap(k_, harmonics_));
  if (have_waist_) waist.reset(new gpgeom::CrabWaistMap(cfg_.full_crossing_angle_rad,
                                                        cfg_.waist_fraction));

  for (std::size_t i = 0; i < b.size(); i++)
    {
      gpgeom::PhaseSpace6D p;
      p.x = b.x(i); p.px = b.px(i);
      p.y = b.y(i); p.py = b.py(i);
      p.zeta = b.zeta(i);          // already beam-local; do NOT re-sign
      p.delta = b.delta(i);

      // IP DISPERSION FIRST, and the order is declared rather than
      // incidental, because dispersion and the crab do NOT commute.
      //
      // Both are Lie transformations exp(:G:) with quadratic generators, so
      // they sit in sp(6,R) -- which is why each comes out exactly linear and
      // exactly symplectic. BCH makes the order matter exactly when the
      // Poisson bracket is non-zero, and here it is:
      //
      //   F = t*zeta*px   (crab)        G = D*delta*px   (dispersion)
      //   {F,G} = dF/dzeta * dG/ddelta - dF/ddelta * dG/dzeta
      //         = (t*px)(D*px) - 0  =  t*D*px^2
      //
      // The (x,px) and (y,py) pairs contribute nothing; the whole bracket comes
      // from (zeta,delta). A drift of length L has generator L*px^2/2, so
      // t*D*px^2 = (2tD)*px^2/2 is a drift of length 2*t*D -- note the factor
      // of two, which reading "L*px^2/2 generates a drift of length L" too
      // quickly loses. examples/maps/test_dispersion.cc measures it: the two
      // orderings differ by exactly (2tD)*px in x, to 3.5 ulps, and by nothing
      // whatever in the other five coordinates.
      //
      // That makes the size concrete rather than asserted: for ILC250,
      // t = tan(7 mrad) = 0.0070 and D = 1.246 mm give 2tD = 17.4 um. Against
      // beta*_x = 13 mm the beam size moves by (1/2)(s/beta*)^2 = 9e-7, which
      // is negligible -- but it is negligible as a MEASURED quantity, and it
      // grows linearly with D, which monochromatization exists to increase.
      //
      // Dispersion is applied first because that is the order the beam meets
      // them travelling toward the IP: dispersion is generated by the upstream
      // bending optics, while the crab is imposed closest to the IP.
      //
      // Horizontal only. The map itself handles either plane and is tested in
      // both, but the input card and the present monochromatization study
      // expose only horizontal dispersion.
      if (have_dispersion_)
        {
          const double D = cfg_.ip_dispersion_m;
          p.x    += D * p.delta;
          p.zeta += D * p.px;      // the symplectic companion, not optional
        }

      // A vertical crab acts in (y,py); swap the planes around the map rather
      // than duplicating it.
      if (have_crab_)
        {
          if (vertical) { std::swap(p.x, p.y); std::swap(p.px, p.py); }
          crab->apply(p);
          if (vertical) { std::swap(p.x, p.y); std::swap(p.px, p.py); }
        }
      // The IP tilt: a symplectic shear, with its Panofsky-Wenzel companion.
      if (have_tilt_)
        {
          if (vertical) { p.y += tilt_slope_ * p.zeta; p.delta -= tilt_slope_ * p.py; }
          else          { p.x += tilt_slope_ * p.zeta; p.delta -= tilt_slope_ * p.px; }
        }
      if (have_waist_) waist->apply(p);

      b.set(i, p.x, p.px, p.y, p.py, p.zeta, p.delta, b.weight(i));
    }
  return true;
}

bool INCOMING_MAP_CHAIN::verify_symplectic(double& worst) const
{
  worst = 0.0;
  if (!active()) return true;

  std::unique_ptr<gpgeom::CrabRFCavityMap> crab;
  std::unique_ptr<gpgeom::CrabWaistMap>    waist;
  if (have_crab_)  crab.reset(new gpgeom::CrabRFCavityMap(k_, harmonics_));
  if (have_waist_) waist.reset(new gpgeom::CrabWaistMap(cfg_.full_crossing_angle_rad,
                                                        cfg_.waist_fraction));

  // Sample away from the origin and off momentum: several defects vanish at
  // zero, so an origin-only check would pass a broken map.
  const gpgeom::PhaseSpace6D pts[4] = {
    { 1.5e-3,  2.0e-4, -3.0e-4, 1.2e-3,  8.0e-4,  2.0e-5},
    {-2.0e-3, -1.0e-4,  5.0e-4,-8.0e-4, -1.2e-2, -1.0e-3},
    { 5.0e-4,  3.0e-5,  1.0e-5, 2.0e-5,  2.5e-2,  5.0e-4},
    { 0.0,     0.0,     0.0,    0.0,     1.0e-3,  0.0   }
  };

  for (int i = 0; i < 4; i++)
    {
      if (have_crab_)
        {
          const gpgeom::Matrix6 j = gpgeom::numerical_jacobian(
            [&crab](gpgeom::PhaseSpace6D& q){ crab->apply(q); }, pts[i], 2.0e-7);
          const double d = gpgeom::symplectic_defect_inf_norm(j);
          // NaN fails every ordinary comparison, so "if (d > worst)" would step
          // straight past it and the gate would pass on a broken map.
          if (!std::isfinite(d)) { worst = d; return false; }
          if (d > worst) worst = d;
        }
      if (have_dispersion_)
        {
          const double D = cfg_.ip_dispersion_m;
          const gpgeom::Matrix6 j = gpgeom::numerical_jacobian(
            [D](gpgeom::PhaseSpace6D& q){
              q.x += D * q.delta; q.zeta += D * q.px; }, pts[i], 2.0e-7);
          const double d = gpgeom::symplectic_defect_inf_norm(j);
          if (!std::isfinite(d)) { worst = d; return false; }
          if (d > worst) worst = d;
        }
      if (have_tilt_)
        {
          const double t = tilt_slope_;
          const bool vert = (cfg_.crab_plane == 1);
          const gpgeom::Matrix6 j = gpgeom::numerical_jacobian(
            [t, vert](gpgeom::PhaseSpace6D& q){
              if (vert) { q.y += t * q.zeta; q.delta -= t * q.py; }
              else      { q.x += t * q.zeta; q.delta -= t * q.px; } },
            pts[i], 2.0e-7);
          const double d = gpgeom::symplectic_defect_inf_norm(j);
          if (!std::isfinite(d)) { worst = d; return false; }
          if (d > worst) worst = d;
        }
      if (have_waist_)
        {
          const gpgeom::Matrix6 j = gpgeom::numerical_jacobian(
            [&waist](gpgeom::PhaseSpace6D& q){ waist->apply(q); }, pts[i], 2.0e-7);
          const double d = gpgeom::symplectic_defect_inf_norm(j);
          if (!std::isfinite(d)) { worst = d; return false; }
          if (d > worst) worst = d;
        }
    }
  // Everything above samples the components separately. A chain can be built
  // from symplectic pieces and still be applied wrongly -- a plane swap that
  // is not undone, a term applied twice -- so sample the COMPOSED map too, in
  // exactly the order apply() uses.
  if (have_crab_ || have_tilt_ || have_waist_ || have_dispersion_)
    {
      const bool vertical = (cfg_.crab_plane == 1);
      const double t = tilt_slope_;
      const bool tilt = have_tilt_;
      const bool disp = have_dispersion_;
      const double D = cfg_.ip_dispersion_m;
      for (int i = 0; i < 4; i++)
        {
          const gpgeom::Matrix6 j = gpgeom::numerical_jacobian(
            [&](gpgeom::PhaseSpace6D& q){
              // Dispersion FIRST, matching apply(). Omitting it here verified a
              // composition the code never applies -- and did so precisely for
              // the dispersive runs, which are the ones the map exists for.
              if (disp) { q.x += D * q.delta; q.zeta += D * q.px; }
              if (crab)
                { if (vertical) { std::swap(q.x, q.y); std::swap(q.px, q.py); }
                  crab->apply(q);
                  if (vertical) { std::swap(q.x, q.y); std::swap(q.px, q.py); } }
              if (tilt)
                { if (vertical) { q.y += t * q.zeta; q.delta -= t * q.py; }
                  else          { q.x += t * q.zeta; q.delta -= t * q.px; } }
              if (waist) waist->apply(q);
            }, pts[i], 2.0e-7);
          const double d = gpgeom::symplectic_defect_inf_norm(j);
          if (!std::isfinite(d)) { worst = d; return false; }
          if (d > worst) worst = d;
        }
    }

  return worst <= cfg_.symplecticity_tolerance;
}

// Argument is a LAB z offset from the reference plane, in metres. Converting it
// to the beam-local zeta the cavity acts on is precisely what the sign is for.
double INCOMING_MAP_CHAIN::kick_at(double z_lab_offset_m) const
{
  const double zeta = cfg_.longitudinal_sign * z_lab_offset_m;
  // For an IP tilt the zeta-dependent response is a DISPLACEMENT t*zeta; for a
  // cavity it is a momentum kick K(zeta). Both reverse with the longitudinal
  // sign, which is what callers use this for.
  if (have_tilt_) return tilt_slope_ * zeta;
  if (!have_crab_) return 0.0;
  const gpgeom::CrabRFCavityMap crab(k_, harmonics_);
  return crab.kick(zeta);
}
