#ifndef PAIRMOMENTSCPP_SEEN
#define PAIRMOMENTSCPP_SEEN

#include <cmath>
#include <vector>
#include <cstddef>

// Pair invariant mass and weighted moments for collision-energy and helicity
// diagnostics. Energies and masses use GeV; directions are unit three-vectors.
// Finite mass and the collision angle are retained in W. They cannot generally
// be dropped when resolving a narrow monochromatized peak.
//
// The Kish effective size, (sum w)^2/sum(w^2), describes weight concentration.
// The resulting moment SEM assumes independent weighted observations; shared
// macroparticle ancestry and encounter correlations require separate treatment.
// Legacy lumi_pp remains the product of laboratory Sz components, distinct from
// the helicities projected onto the local particle momenta.

namespace pairmoments {

// Invariant mass of a pair, finite mass kept.
//
//   W^2 = (p1 + p2)^2 = m1^2 + m2^2 + 2(E1 E2 - p1.p2)
//
// The direction vectors are unit. |p| is formed as sqrt((E-m)(E+m)) rather than
// sqrt(E^2 - m^2): the two agree analytically, but the factored form does not
// lose the difference to cancellation when E approaches m.
//
// In a collider the two beams are nearly anti-parallel, so E1E2 - p1.p2 tends to
// E1E2 + |p1||p2| and there is no cancellation. The dangerous direction is
// nearly PARALLEL momenta, which this expression handles correctly but with the
// usual loss of significance -- worth knowing if it is ever reused for
// same-beam pairs.
inline double invariant_mass(double e1, double vx1, double vy1, double vz1,
                             double e2, double vx2, double vy2, double vz2,
                             double m)
{
  const double p1 = (e1 > m) ? std::sqrt((e1 - m) * (e1 + m)) : 0.0;
  const double p2 = (e2 > m) ? std::sqrt((e2 - m) * (e2 + m)) : 0.0;
  const double cosang = vx1 * vx2 + vy1 * vy2 + vz1 * vz2;
  const double w2 = 2.0 * m * m + 2.0 * (e1 * e2 - p1 * p2 * cosang);
  // W^2 is non-negative analytically; a small negative is roundoff on a nearly
  // massless collinear pair. Clamp rather than return NaN, and let the caller
  // see zero.
  return (w2 > 0.0) ? std::sqrt(w2) : 0.0;
}

// Weighted first and second moments of one scalar.
class Moment1
{
  double w_, w2_, sx_, sxx_;
 public:
  Moment1() : w_(0.0), w2_(0.0), sx_(0.0), sxx_(0.0) {}
  inline void add(double w, double x)
    { w_ += w; w2_ += w * w; sx_ += w * x; sxx_ += w * x * x; }
  // Exact sufficient-statistic merge used by encounter-local collision
  // lanes.  No means or variances are averaged: doing that would lose the
  // weight-squared term needed for N_eff and would make the answer depend on
  // the number of workers.
  inline bool finite() const
    {
      return std::isfinite(w_) && std::isfinite(w2_) &&
             std::isfinite(sx_) && std::isfinite(sxx_);
    }
  inline bool merge(const Moment1& other)
    {
      if (!finite() || !other.finite()) return false;
      const double w = w_ + other.w_;
      const double w2 = w2_ + other.w2_;
      const double sx = sx_ + other.sx_;
      const double sxx = sxx_ + other.sxx_;
      if (!std::isfinite(w) || !std::isfinite(w2) ||
          !std::isfinite(sx) || !std::isfinite(sxx)) return false;
      w_ = w;
      w2_ = w2;
      sx_ = sx;
      sxx_ = sxx;
      return true;
    }
  inline double weight() const { return w_; }
  inline bool   empty()  const { return !(w_ > 0.0); }
  inline double mean()   const { return empty() ? 0.0 : sx_ / w_; }

  // Effective sample size, Kish. Equal weights give the count; one dominant
  // weight collapses it toward 1.
  inline double n_eff() const
    { return (w2_ > 0.0) ? (w_ * w_) / w2_ : 0.0; }

  // Population variance of the weighted sample.
  inline double variance() const
    {
      if (empty()) return 0.0;
      const double mu = mean();
      const double v = sxx_ / w_ - mu * mu;
      return (v > 0.0) ? v : 0.0;      // roundoff can make it slightly negative
    }

  // Independent-encounter SEM proxy. Reusing a macroparticle in multiple
  // encounters violates this independence assumption. The value is not an
  // independent-seed uncertainty and excludes grid, slice and step convergence.
  inline double error() const
    {
      const double n = n_eff();
      return (n > 1.0) ? std::sqrt(variance() / n) : 0.0;
    }
};

// Joint moments of the two helicities, so their correlation is available and
// not merely their means.
class PairHelicity
{
  Moment1 h1_, h2_;
  double  w_, sh1h2_;
 public:
  PairHelicity() : w_(0.0), sh1h2_(0.0) {}
  inline void add(double w, double a, double b)
    { h1_.add(w, a); h2_.add(w, b); w_ += w; sh1h2_ += w * a * b; }
  inline bool finite() const
    {
      return h1_.finite() && h2_.finite() && std::isfinite(w_) &&
             std::isfinite(sh1h2_);
    }
  inline bool merge(const PairHelicity& other)
    {
      if (!finite() || !other.finite()) return false;
      PairHelicity candidate = *this;
      if (!candidate.h1_.merge(other.h1_) ||
          !candidate.h2_.merge(other.h2_)) return false;
      candidate.w_ += other.w_;
      candidate.sh1h2_ += other.sh1h2_;
      if (!candidate.finite()) return false;
      *this = candidate;
      return true;
    }

  inline const Moment1& h1() const { return h1_; }
  inline const Moment1& h2() const { return h2_; }
  inline double weight()     const { return w_; }
  inline bool   empty()      const { return !(w_ > 0.0); }

  // <h1 h2>, the quantity a spin-dependent cross section multiplies. Note this
  // is NOT <h1><h2> unless the helicities are uncorrelated.
  inline double mean_product() const
    { return empty() ? 0.0 : sh1h2_ / w_; }

  inline double covariance() const
    { return empty() ? 0.0 : mean_product() - h1_.mean() * h2_.mean(); }
};

// dL/dW: the pair weight binned in invariant mass, each bin carrying its own
// helicity moments.
class SpectrumW
{
  double lo_, hi_, dw_;
  // Luminosity weight and helicity moments are kept SEPARATE on purpose.
  // dL/dW is wanted whether or not spin is stored, and folding an unpolarized
  // pair in as (h1, h2) = (0, 0) would pull the helicity means toward zero with
  // pairs that never carried a helicity at all.
  std::vector<double>       binw_;
  std::vector<PairHelicity> binh_;
  Moment1 wmom_;                 // W itself, over all pairs
  double  under_, over_;
 public:
  SpectrumW() : lo_(0.0), hi_(0.0), dw_(0.0), under_(0.0), over_(0.0) {}

  inline void reset()
    {
      lo_ = hi_ = dw_ = 0.0;
      binw_.clear();
      binh_.clear();
      wmom_ = Moment1();
      under_ = over_ = 0.0;
    }

  inline void configure(double lo, double hi, std::size_t n)
    {
      reset();
      if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi) || n == 0)
        return;
      lo_ = lo; hi_ = hi; dw_ = (hi - lo) / (double)n;
      binw_.assign(n, 0.0);
      binh_.assign(n, PairHelicity());
      under_ = over_ = 0.0;
    }

  inline bool configured() const { return !binw_.empty() && dw_ > 0.0; }
  inline std::size_t bins() const { return binw_.size(); }
  inline double lower_edge() const { return lo_; }
  inline double upper_edge() const { return hi_; }
  inline double bin_low(std::size_t i)  const { return lo_ + dw_ * (double)i; }
  inline double bin_high(std::size_t i) const { return lo_ + dw_ * (double)(i + 1); }
  inline double bin_weight(std::size_t i) const
    { return (i < binw_.size()) ? binw_[i] : 0.0; }
  inline const PairHelicity& bin(std::size_t i) const { return binh_[i]; }
  inline const Moment1& w_moment() const { return wmom_; }
  inline double underflow() const { return under_; }
  inline double overflow()  const { return over_; }

  inline bool finite() const
    {
      if (!std::isfinite(lo_) || !std::isfinite(hi_) ||
          !std::isfinite(dw_) || !std::isfinite(under_) ||
          !std::isfinite(over_) || !wmom_.finite()) return false;
      for (std::size_t i = 0; i < binw_.size(); ++i)
        if (!std::isfinite(binw_[i]) || !binh_[i].finite()) return false;
      return true;
    }

  // Luminosity only. Use for every counted pair.
  inline void add(double weight, double W)
    {
      wmom_.add(weight, W);
      if (!configured()) return;
      if (W < lo_)  { under_ += weight; return; }
      if (W >= hi_) { over_  += weight; return; }
      std::size_t i = (std::size_t)((W - lo_) / dw_);
      if (i >= binw_.size()) i = binw_.size() - 1;   // guard the top edge
      binw_[i] += weight;
    }

  // Luminosity and helicity. Use only for a pair that actually carries spin.
  inline void add(double weight, double W, double h1, double h2)
    {
      add(weight, W);
      if (!configured() || W < lo_ || W >= hi_) return;
      std::size_t i = (std::size_t)((W - lo_) / dw_);
      if (i >= binh_.size()) i = binh_.size() - 1;
      binh_[i].add(weight, h1, h2);
    }

  // Merge only identically configured spectra.  This is intentionally a
  // checked operation: silently combining different binning contracts would
  // produce a plausible but physically meaningless dL/dW file.
  inline bool merge(const SpectrumW& other)
    {
      std::vector<const SpectrumW*> spectra(1U, &other);
      return merge_many(spectra);
    }

  // Transactional multi-lane reduction.  Spectrum storage is copied once per
  // publication batch, not once for every encounter in that batch.
  inline bool merge_many(const std::vector<const SpectrumW*>& others)
    {
      if (!finite()) return false;
      SpectrumW candidate = *this;
      for (std::size_t lane = 0; lane < others.size(); ++lane)
        {
          const SpectrumW* other = others[lane];
          if (!other || !other->finite()) return false;
          if (configured() != other->configured()) return false;
          if (configured() &&
              (lo_ != other->lo_ || hi_ != other->hi_ ||
               dw_ != other->dw_ ||
               binw_.size() != other->binw_.size()))
            return false;
          if (!candidate.wmom_.merge(other->wmom_)) return false;
          candidate.under_ += other->under_;
          candidate.over_ += other->over_;
          for (std::size_t i = 0; i < binw_.size(); ++i)
            {
              candidate.binw_[i] += other->binw_[i];
              if (!candidate.binh_[i].merge(other->binh_[i])) return false;
            }
          if (!candidate.finite()) return false;
        }
      *this = candidate;
      return true;
    }

  // dL/dW in whatever units the weights carry, per unit W. An EMPTY bin returns
  // false rather than zero: zero luminosity and no sample are different
  // statements, and a spectrum that reports the second as the first invites
  // exactly the wrong conclusion.
  inline bool dL_dW(std::size_t i, double& value) const
    {
      if (i >= binw_.size() || !(binw_[i] > 0.0)) return false;
      value = binw_[i] / dw_;
      return true;
    }
};

}  // namespace pairmoments

#endif
