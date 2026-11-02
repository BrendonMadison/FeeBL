#ifndef FEEBL_LEGACY_PHOTON_LUMI_FAST_CPP_H
#define FEEBL_LEGACY_PHOTON_LUMI_FAST_CPP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace legacy_photon_lumi {

// The legacy GRID stores signed lepton energy and a sampled macro weight in
// pointer wrappers.  These deliberately small records are the audited
// boundary into the luminosity-only kernel; no particle or RNG state is owned
// or changed here.
struct PhotonSample {
  float energy;
  float weight;
};

struct ChargedSample {
  float signed_energy;
  float weight;
};

// Non-owning, type-erased views let the production GRID expose its private
// BEAM_*_POINTER vectors without first copying every energy/weight pair into
// adapter vectors.  The callback is selected once at the GRID boundary; the
// kernel never retains either the context or a returned sample.
class PhotonView {
 public:
  typedef PhotonSample (*Loader)(const void*, std::size_t);

  PhotonView();
  PhotonView(const void* context, std::size_t size, Loader loader);

  std::size_t size() const;
  bool valid() const;
  PhotonSample at(std::size_t index) const;

 private:
  const void* context_;
  std::size_t size_;
  Loader loader_;
};

class ChargedView {
 public:
  typedef ChargedSample (*Loader)(const void*, std::size_t);

  ChargedView();
  ChargedView(const void* context, std::size_t size, Loader loader);

  std::size_t size() const;
  bool valid() const;
  ChargedSample at(std::size_t index) const;

 private:
  const void* context_;
  std::size_t size_;
  Loader loader_;
};

enum class Algorithm : unsigned char {
  CartesianReference = 1,
  FactoredExact = 2
};

struct Result {
  std::array<double, 9> class_luminosity;
  double lumi_ge;
  double lumi_eg;
  double lumi_gg;
  double lumi_gg_high;
  std::uint64_t ge_pair_opportunities;
  std::uint64_t eg_pair_opportunities;
  std::uint64_t gg_pair_opportunities;
  std::uint64_t cartesian_pairs_visited;

  Result();
  void merge(const Result& other);
};

// Reused across all cells and all photon_lumi calls owned by one GRID.  Only
// the two positive-photon lists needed for the strict gamma-gamma threshold
// are materialized.  One stays in sample order and one is sorted.  reserve()
// is called once with the largest cell occupancies, so none of these vectors
// allocates in the per-cell loop.
class Workspace {
 public:
  struct WeightedPhoton {
    float energy;
    float weight;
    std::size_t ordinal;
  };

  Workspace();
  void reserve(std::size_t maximum_query_photons,
               std::size_t maximum_sorted_photons);
  std::size_t query_capacity() const;
  std::size_t weighted_capacity() const;
  std::size_t suffix_capacity() const;

  // Public only to keep this small C++11 workspace usable by the separately
  // compiled audited kernel.  Callers must treat these as scratch state.
  std::vector<WeightedPhoton> positive_query_photons;
  std::vector<WeightedPhoton> sorted_positive_photons;
  std::vector<long double> suffix_weights;
};

// This is the complete switch-level proof obligation for selecting the
// factorized production path.  An explicit request that fails this predicate
// is an error, never a silent fallback.
struct Features {
  int do_photon_lumi;
  int do_lumi;
  int do_jets;
  int do_compt;
  int do_compt_phot;
  int do_pairs;
  int do_muons;
  int do_hadrons;
  int do_coherent;
  int do_trident;
  int do_cross;
  int do_bhabhas;
  int do_prod;
  int do_tertphot;

  Features();
};

bool eligible(const Features& features, std::string* reason);

// This is the explicit production GRID::collide_gg threshold definition.
// The audited GNU legacy build used -ffast-math, which folded the historical
// sqrt(s)*sqrt(s) spelling to this direct product.  Keep the definition in one
// header so the Cartesian GRID path and both factorization oracles cannot
// drift at equality/ULP boundaries again.
inline bool gg_above_threshold(float first_energy, float second_energy,
                               double gg_smin) {
  const double first = static_cast<double>(first_energy);
  const double second = static_cast<double>(second_energy);
  return (4.0 * first) * second > gg_smin;
}

// Exact here means an exact algebraic factorization of the same selected
// physical pairs, weights, charge classes, and strict production predicate
// 4 E1 E2 > gg_smin.  Reassociation removes the Cartesian visits, so last-bit
// floating-point equality with sequential legacy addition is neither promised
// nor used as a validation criterion.
Result compute_cell(const std::vector<PhotonSample>& beam1_photons,
                    const std::vector<ChargedSample>& beam2_particles,
                    const std::vector<PhotonSample>& beam2_photons,
                    const std::vector<ChargedSample>& beam1_particles,
                    double gg_smin,
                    Algorithm algorithm);

// Production entry point.  Accumulation is into caller-owned transactional
// state: GRID publishes the Result only after every cell validates.  Views and
// workspace are never retained, and this function performs no RNG operations.
void accumulate_cell(const PhotonView& beam1_photons,
                     const ChargedView& beam2_particles,
                     const PhotonView& beam2_photons,
                     const ChargedView& beam1_particles,
                     double gg_smin,
                     Algorithm algorithm,
                     Workspace& workspace,
                     Result* result);

const char* algorithm_name(Algorithm algorithm);

}  // namespace legacy_photon_lumi
}  // namespace feebl

#endif  // FEEBL_LEGACY_PHOTON_LUMI_FAST_CPP_H
