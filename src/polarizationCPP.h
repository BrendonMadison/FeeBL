#ifndef POLARIZATIONCPP_SEEN
#define POLARIZATIONCPP_SEEN

#include "abstractIOclass.h"
#include "beamCPP.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>

// FEEBL: longitudinal bunch polarization, reported alongside luminosity.
//
// P_long = < s . p_hat >, the projection of each particle's spin onto its own
// direction of motion, averaged over the bunch. It is the quantity a polarised
// collider actually quotes, and it is the one thing the spin machinery could
// compute but never reported: the output exposed only a scalar lumi_pp.
//
// Two conventions matter here and neither is optional.
//
//   * p_hat is the particle's OWN direction, not the z axis. FEEBL stores vx
//     and vy as transverse slopes and carries no longitudinal component, so
//     p_hat = (vx, vy, sign*sqrt(1 - vx^2 - vy^2)) with sign = +1 for beam 1
//     and -1 for beam 2 (BEAM::sign_label, set in beamCPP.h). Using +z for both
//     beams would report beam 2's polarization with the wrong sign -- a plausible
//     number, which is the failure mode this project keeps finding.
//
//   * The uncertainty is the statistical error on the mean over MACROPARTICLES,
//     sqrt(Var/N). It describes how well this run resolves the bunch mean. It is
//     not a physics uncertainty and says nothing about model error.
//
// Sampled at three points: as the bunch enters, at maximum overlap, and after
// the collision. The middle one is taken at the instant every slice of beam 1
// overlaps every slice of beam 2, which is where the beam-beam field is largest.

struct BunchPolarization
{
  double        p;        // helicity, < s . p_hat >
  double        p_lab;    // lab-frame projection, < s . zhat >
  double        err;      // statistical error on the mean, sqrt(Var/N)
  unsigned long n;        // macroparticles sampled
  bool          valid;    // false when no spin is stored

  BunchPolarization() : p(0.0), p_lab(0.0), err(0.0), n(0), valid(false) {}
};

inline BunchPolarization longitudinal_polarization(const BEAM& beam)
{
  BunchPolarization r;
  if (!beam.has_spin()) return r;          // valid stays false

  const double sign = (double)beam.sign_label();
  double sum = 0.0, sumsq = 0.0, sumlab = 0.0;
  unsigned long n = 0;

  const int nsl = beam.number_of_slices();
  for (int sl = 0; sl < nsl; sl++)
    {
      const std::vector<PARTICLE*>& v = beam.getParticleVector(sl);
      for (std::size_t k = 0; k < v.size(); k++)
        {
          float vx = 0.0f, vy = 0.0f;
          v[k]->velocities(vx, vy);
          const double t2 = (double)vx*(double)vx + (double)vy*(double)vy;
          if (!(t2 < 1.0)) continue;       // also rejects NaN
          const double vz = sign * std::sqrt(1.0 - t2);
          const TRIDVECTOR& s = v[k]->getSpin();
          const double proj = s.getComponent(0)*(double)vx
                            + s.getComponent(1)*(double)vy
                            + s.getComponent(2)*vz;
          if (!std::isfinite(proj)) continue;
          sum += proj; sumsq += proj*proj; n++;
          sumlab += s.getComponent(2);      // lab-frame z projection
        }
    }

  if (n == 0) return r;
  r.n = n;
  r.p = sum / (double)n;
  r.p_lab = sumlab / (double)n;
  if (n > 1)
    {
      // population variance of the projection, then the error on its mean
      double var = sumsq/(double)n - r.p*r.p;
      if (var < 0.0) var = 0.0;            // rounding at near-zero spread
      r.err = std::sqrt(var / (double)n);
    }
  r.valid = true;
  return r;
}

// The three states, both beams, formatted for the output listing.
class POLARIZATION_REPORT : public ABSTRACT_IO_CLASS
{
 public:
  enum State { INITIAL = 0, COLLISION = 1, FINAL = 2, NSTATE = 3 };

  POLARIZATION_REPORT() : have_lw_(false), lw1_(0.0), lw2_(0.0), lwsum_(0.0), enabled_(false)
    { for (int i = 0; i < NSTATE; i++) sampled_[i] = false; }

  inline void set_enabled(bool e) { enabled_ = e; }
  inline bool enabled() const { return enabled_; }
  inline bool sampled(State s) const { return sampled_[s]; }

  inline void sample(State s, const BEAM& b1, const BEAM& b2)
    {
      if (!enabled_) return;
      p1_[s] = longitudinal_polarization(b1);
      p2_[s] = longitudinal_polarization(b2);
      sampled_[s] = true;
    }

  // Luminosity-weighted polarization, supplied by the grid after the
  // collision. This is the quantity a physics analysis needs: the polarization
  // of the particles that actually collided, weighted by the luminosity they
  // produced, rather than the average over the whole outgoing bunch.
  inline void set_luminosity_weighted(double h1, double h2, double wsum)
    { lw1_ = h1; lw2_ = h2; lwsum_ = wsum; have_lw_ = (wsum > 0.0); }

  inline const BunchPolarization& beam1(State s) const { return p1_[s]; }
  inline const BunchPolarization& beam2(State s) const { return p2_[s]; }

  virtual std::string output_flow() const
    {
      std::ostringstream out;
      // Emit nothing at all when no spin is stored. Printing "not computed"
      // would add two lines to every existing run's listing and break the
      // bit-identity check against the v0.5.0 reference, which is a standing
      // invariant of this project -- a new diagnostic must not silently
      // invalidate the differential test. The beam-parameter block already
      // says polarization applies only when bmt_precession = 1.
      if (!enabled_) return out.str();
      out << " ---------- longitudinal bunch polarization ---------- " << std::endl;
      out << " P is HELICITY, <s.p_hat>: the spin projected on each particle's OWN"
             " direction of motion." << std::endl
          << " Beam 2 travels toward -z, so a lab spin along +z gives it NEGATIVE"
             " helicity. The lab-frame" << std::endl
          << " projection <s.zhat> is printed alongside so the two cannot be"
             " confused." << std::endl
          << " The +- is the statistical error on the mean over macroparticles --"
             " the resolution of this" << std::endl
          << " run, not a physics uncertainty." << std::endl;
      static const char* label[NSTATE] = { "initial  ", "collision", "final    " };
      for (int i = 0; i < NSTATE; i++)
        {
          out << " (P.1 , P.2) at " << label[i] << " = ";
          if (!sampled_[i]) { out << "(not sampled)" << std::endl; continue; }
          out << "(" << fmt(p1_[i]) << " , " << fmt(p2_[i]) << ")"
              << "   [lab <s.zhat>: " << fmtlab(p1_[i]) << " , "
              << fmtlab(p2_[i]) << "]" << std::endl;
        }
      if (have_lw_)
        {
          out << " (P.1 , P.2) luminosity-weighted = ("
              << std::showpos << std::fixed << std::setprecision(4) << lw1_
              << " , " << lw2_ << std::noshowpos << ")" << std::endl;
          if (sampled_[INITIAL] && p1_[INITIAL].valid)
            {
              const double d1 = lw1_ - p1_[INITIAL].p;
              const double d2 = lw2_ - p2_[INITIAL].p;
              out << " shift from initial, luminosity-weighted : beam 1 "
                  << std::showpos << d1 << " , beam 2 " << d2 << std::noshowpos
                  << std::endl
                  << " This is the number a physics analysis uses, and it is"
                     " normally much smaller than the" << std::endl
                  << " whole-bunch figure below: the particles that collide are"
                     " not a fair sample of the bunch." << std::endl;
            }
        }
      if (sampled_[INITIAL] && sampled_[FINAL] &&
          p1_[INITIAL].valid && p1_[FINAL].valid)
        {
          out << " depolarization (final - initial) : beam 1 "
              << std::showpos << (p1_[FINAL].p - p1_[INITIAL].p)
              << " , beam 2 " << (p2_[FINAL].p - p2_[INITIAL].p)
              << std::noshowpos << std::endl;
        }
      return out.str();
    }

 private:
  static std::string fmt(const BunchPolarization& b)
    {
      std::ostringstream o;
      if (!b.valid) { o << "n/a"; return o.str(); }
      o << std::showpos << std::fixed << std::setprecision(4) << b.p
        << std::noshowpos << " +- " << std::setprecision(4) << b.err;
      return o.str();
    }

  static std::string fmtlab(const BunchPolarization& b)
    {
      std::ostringstream o;
      if (!b.valid) { o << "n/a"; return o.str(); }
      o << std::showpos << std::fixed << std::setprecision(4) << b.p_lab
        << std::noshowpos;
      return o.str();
    }

  bool              have_lw_;
  double            lw1_, lw2_, lwsum_;
  bool              enabled_;
  bool              sampled_[NSTATE];
  BunchPolarization p1_[NSTATE], p2_[NSTATE];
};

#endif
