#ifndef FEEBL_COLLIDER_DIAGNOSTICS_CPP_H
#define FEEBL_COLLIDER_DIAGNOSTICS_CPP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>

// Read-only sufficient statistics for the optional collider viewer. These
// never feed a particle, field, acceptance decision, or random-number stream.
namespace feebl { namespace diagnostics {
// Shifted weighted moments avoid subtracting two O(E^2) numbers for a narrow
// energy distribution. Long-double accumulators retain small interval changes.
// Inputs use the native observable units and nonnegative luminosity weights.
struct Moment {
  long double weight = 0.0L, origin = 0.0L, sum = 0.0L, square = 0.0L;

  void add(double w, double x) {
    if (!std::isfinite(w) || w < 0.0 || !std::isfinite(x))
      throw std::domain_error("invalid native diagnostic moment input");
    if (w == 0.0) return;
    if (weight == 0.0L) origin = x;
    const long double delta = static_cast<long double>(x) - origin;
    weight += w;
    sum += w * delta;
    square += w * delta * delta;
  }

  bool finite() const {
    return std::isfinite(weight) && std::isfinite(origin) &&
           std::isfinite(sum) && std::isfinite(square);
  }

  // Translate the second accumulator to this origin before combining it.
  void merge(const Moment& other) {
    if (other.weight == 0.0L) return;
    if (weight == 0.0L) { *this = other; return; }
    const long double delta = other.origin - origin;
    square += other.square + 2.0L * delta * other.sum +
              other.weight * delta * delta;
    sum += other.sum + other.weight * delta;
    weight += other.weight;
  }

  // Remove a known subset. Empty differences have the canonical zero state.
  Moment minus(const Moment& other) const {
    if (other.weight == 0.0L) return *this;
    if (other.weight > weight)
      throw std::domain_error("diagnostic subset exceeds total weight");
    if (other.weight == weight) return Moment();
    Moment result;
    result.origin = origin;
    result.weight = weight - other.weight;
    const long double delta = other.origin - origin;
    result.sum = sum - other.sum - other.weight * delta;
    result.square = square - other.square - 2.0L * delta * other.sum -
                    other.weight * delta * delta;
    return result;
  }

  double mean() const {
    return weight > 0.0L ? static_cast<double>(origin + sum / weight) : 0.0;
  }

  double rms() const {
    if (weight == 0.0L) return 0.0;
    const long double shifted_mean = sum / weight;
    const long double variance = square / weight - shifted_mean * shifted_mean;
    // Only permit cancellation at long-double rounding scale. A materially
    // negative variance indicates inconsistent subset arithmetic, not zero spread.
    const long double tolerance = 128.0L * std::numeric_limits<long double>::epsilon() *
        std::max(std::fabs(square / weight), shifted_mean * shifted_mean);
    if (variance < -tolerance)
      throw std::domain_error("negative native diagnostic centered variance");
    return static_cast<double>(std::sqrt(std::max(0.0L, variance)));
  }
};
struct BeamMoments {
  // Absolute native energy, transverse positions, transverse unit-direction
  // components. rms is centered population standard deviation, not an error.
  std::array<Moment,5> values;
  void add(double w,double energy,double x,double y,double vx,double vy) {
    const double v[5]={std::fabs(energy),x,y,vx,vy};
    for (std::size_t i=0;i<5;++i) values[i].add(w,v[i]);
  }
  bool finite() const {
    for (const Moment& v:values) if (!v.finite()) return false;
    return true;
  }
  void merge(const BeamMoments& other) {
    for (std::size_t i=0;i<5;++i) values[i].merge(other.values[i]);
  }
  BeamMoments minus(const BeamMoments& other) const {
    BeamMoments r;
    for (std::size_t i=0;i<5;++i) r.values[i]=values[i].minus(other.values[i]);
    return r;
  }
  double weight() const { return values[0].weight; }
};
// Fixed bounded logarithmic geometry; zero belongs to underflow. Bin contents
// are integrals, not density per logarithmic interval.
struct Histogram {
  static const std::size_t bins = 128;
  std::array<double,bins> weights{};
  double log_low=-8.0, log_high=4.0, underflow=0.0, overflow=0.0;
  unsigned long long samples=0;
  Histogram() = default;
  Histogram(double lo,double hi):log_low(lo),log_high(hi) {}
  double low(std::size_t i) const {
    return std::pow(10.0,log_low+(log_high-log_low)*double(i)/double(bins));
  }
  void add(double x,double weight=1.0) {
    if (!std::isfinite(x) || x < 0.0 || !std::isfinite(weight) || weight < 0.0)
      throw std::domain_error("invalid native diagnostic histogram input");
    ++samples;
    if (x < low(0)) underflow+=weight;
    else if (x >= low(bins)) overflow+=weight;
    else {
      const std::size_t index=std::min(bins-1,static_cast<std::size_t>(
          (std::log10(x)-log_low)*double(bins)/(log_high-log_low)));
      weights[index]+=weight;
    }
  }
  bool finite() const {
    if (!std::isfinite(underflow) || !std::isfinite(overflow)) return false;
    for (double w:weights) if (!std::isfinite(w)) return false;
    return true;
  }
  void merge(const Histogram& other) {
    if (log_low!=other.log_low || log_high!=other.log_high)
      throw std::logic_error("native diagnostic histogram geometry mismatch");
    for (std::size_t i=0;i<bins;++i) weights[i]+=other.weights[i];
    underflow+=other.underflow; overflow+=other.overflow; samples+=other.samples;
  }
  Histogram minus(const Histogram& other) const {
    Histogram r(log_low,log_high);
    for (std::size_t i=0;i<bins;++i) r.weights[i]=weights[i]-other.weights[i];
    r.underflow=underflow-other.underflow; r.overflow=overflow-other.overflow;
    r.samples=samples-other.samples; return r;
  }
};
// Canonical native e+e- adapter convention, not inferred from energy sign or
// measured per-particle species. See gpuMode3ProductionCPP.cc import contract.
// charge_sign is only the relative beam-charge product; other modes do not
// authorize the absolute assignment here (including the force-free case).
inline int canonical_primary_pdg(int beam,double charge_sign) {
  if(beam!=1 && beam!=2)throw std::invalid_argument("invalid primary diagnostic beam identity");
  if(charge_sign!=-1.0)return 0;
  return beam==1?11:-11;
}
struct PrimaryChiHistograms {
  Histogram combined{-10.0,3.0};
  std::array<Histogram,2> by_beam{{Histogram(-10.0,3.0),Histogram(-10.0,3.0)}};
  void add(int beam,double value) {
    // Reject identity before mutating any accumulator. The combined histogram
    // keeps exactly its former update order and unit-weight semantics.
    if(beam!=1 && beam!=2)throw std::invalid_argument("invalid primary chi beam identity");
    combined.add(value);
    by_beam[static_cast<std::size_t>(beam-1)].add(value);
  }
  const Histogram& beam(int identity) const {
    if(identity!=1 && identity!=2)throw std::invalid_argument("invalid primary chi beam identity");
    return by_beam[static_cast<std::size_t>(identity-1)];
  }
  bool finite() const {return combined.finite()&&by_beam[0].finite()&&by_beam[1].finite();}
  void merge(const PrimaryChiHistograms& other) {
    combined.merge(other.combined);
    by_beam[0].merge(other.by_beam[0]);by_beam[1].merge(other.by_beam[1]);
  }
};
struct CollisionMoments {
  BeamMoments beam1,beam2;
  bool finite() const { return beam1.finite() && beam2.finite(); }
  void merge(const CollisionMoments& other) {
    beam1.merge(other.beam1); beam2.merge(other.beam2);
  }
  CollisionMoments minus(const CollisionMoments& other) const {
    CollisionMoments r; r.beam1=beam1.minus(other.beam1);
    r.beam2=beam2.minus(other.beam2); return r;
  }
};
} }
#endif
