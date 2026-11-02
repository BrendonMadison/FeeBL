#include "legacyPhotonLumiFastCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace feebl {
namespace legacy_photon_lumi {

namespace {

void add_product(std::uint64_t left, std::uint64_t right,
                 std::uint64_t* destination) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
    throw std::overflow_error("legacy photon-luminosity pair count overflow");
  const std::uint64_t product = left * right;
  if (*destination > std::numeric_limits<std::uint64_t>::max() - product)
    throw std::overflow_error("legacy photon-luminosity pair count overflow");
  *destination += product;
}

void count_cartesian_visit(Result* result) {
  if (result->cartesian_pairs_visited ==
      std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(
        "legacy photon-luminosity Cartesian visit count overflow");
  ++result->cartesian_pairs_visited;
}

void require_finite_nonnegative(float value, const char* label) {
  if (!std::isfinite(static_cast<double>(value)) || value < 0.0f) {
    std::ostringstream out;
    out << label << " must be finite and non-negative";
    throw std::invalid_argument(out.str());
  }
  // The historical executable links with -ffast-math, which enables DAZ/FTZ
  // on x86 even though this translation unit is compiled with
  // -fno-fast-math.  Inspect the representation so a subnormal deck weight
  // cannot be silently observed as zero before the proof obligation runs.
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = (bits >> 23u) & 0xffu;
  const std::uint32_t fraction = bits & 0x7fffffu;
  if (exponent == 0u && fraction != 0u) {
    std::ostringstream out;
    out << label << " must not be a subnormal float";
    throw std::underflow_error(out.str());
  }
}

struct WeightRange {
  bool have_positive;
  float minimum_positive;
  float maximum;
};

WeightRange empty_weight_range() {
  const WeightRange empty = {
      false, std::numeric_limits<float>::max(), 0.0f};
  return empty;
}

void observe_weight(float weight, WeightRange* range) {
  range->maximum = std::max(range->maximum, weight);
  if (weight > 0.0f) {
    range->have_positive = true;
    range->minimum_positive = std::min(range->minimum_positive, weight);
  }
}

struct PhotonStats {
  WeightRange all_weights;
  WeightRange positive_energy_weights;
  long double all_weight;
  long double positive_energy_weight;
  std::size_t positive_energy_count;
};

struct ChargedStats {
  WeightRange all_weights;
  long double class_weight[2];
};

PhotonStats scan_photons(
    const PhotonView& photons,
    const char* label,
    std::vector<Workspace::WeightedPhoton>* materialized_positive) {
  PhotonStats stats = {empty_weight_range(), empty_weight_range(),
                       0.0L, 0.0L, 0u};
  for (std::size_t i = 0; i < photons.size(); ++i) {
    const PhotonSample sample = photons.at(i);
    require_finite_nonnegative(sample.energy, label);
    require_finite_nonnegative(sample.weight, label);
    observe_weight(sample.weight, &stats.all_weights);
    stats.all_weight += sample.weight;
    if (sample.energy > 0.0f) {
      observe_weight(sample.weight, &stats.positive_energy_weights);
      stats.positive_energy_weight += sample.weight;
      ++stats.positive_energy_count;
      if (materialized_positive != 0) {
        const Workspace::WeightedPhoton weighted = {
            sample.energy, sample.weight, i};
        materialized_positive->push_back(weighted);
      }
    }
  }
  return stats;
}

ChargedStats scan_particles(const ChargedView& particles,
                            const char* label) {
  ChargedStats stats = {empty_weight_range(), {0.0L, 0.0L}};
  for (std::size_t i = 0; i < particles.size(); ++i) {
    const ChargedSample sample = particles.at(i);
    const float signed_energy = sample.signed_energy;
    if (!std::isfinite(static_cast<double>(signed_energy))) {
      std::ostringstream out;
      out << label << " energy must be finite";
      throw std::invalid_argument(out.str());
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &signed_energy, sizeof(bits));
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0u && fraction != 0u) {
      std::ostringstream out;
      out << label << " energy must not be a subnormal float";
      throw std::underflow_error(out.str());
    }
    require_finite_nonnegative(sample.weight, label);
    observe_weight(sample.weight, &stats.all_weights);
    // Preserve collide_ge/collide_eg exactly: zero belongs to class one.
    stats.class_weight[signed_energy < 0.0f ? 0u : 1u] += sample.weight;
  }
  return stats;
}

void require_representable_pair_weights(const WeightRange& left,
                                        const WeightRange& right,
                                        const char* label) {
  if (!left.have_positive || !right.have_positive) return;
  const float maximum_product = left.maximum * right.maximum;
  if (!std::isfinite(static_cast<double>(maximum_product))) {
    std::ostringstream out;
    out << label << " pair weight overflows legacy float multiplication";
    throw std::overflow_error(out.str());
  }
  const float minimum_product =
      left.minimum_positive * right.minimum_positive;
  if (!(minimum_product >= std::numeric_limits<float>::min())) {
    std::ostringstream out;
    out << label
        << " pair weight enters legacy float underflow/subnormal range";
    throw std::underflow_error(out.str());
  }
}

std::size_t charge_class(float signed_energy) {
  // Preserve collide_ge/collide_eg exactly: zero belongs to legacy class 1.
  return signed_energy < 0.0f ? 0u : 1u;
}

void add_reference_ge(const PhotonView& photons,
                      const ChargedView& particles,
                      bool ge, Result* result) {
  add_product(static_cast<std::uint64_t>(photons.size()),
              static_cast<std::uint64_t>(particles.size()),
              ge ? &result->ge_pair_opportunities
                 : &result->eg_pair_opportunities);
  for (std::size_t p = 0; p < photons.size(); ++p) {
    const PhotonSample photon = photons.at(p);
    for (std::size_t e = 0; e < particles.size(); ++e) {
      count_cartesian_visit(result);
      if (photon.energy <= 0.0f) continue;
      const ChargedSample particle = particles.at(e);
      // collide_ge/collide_eg rounds this product to float before adding it to
      // the double RESULTS accumulators.
      const float pair_weight = photon.weight * particle.weight;
      const std::size_t signed_bucket =
          charge_class(particle.signed_energy);
      // collide_ge initializes j2=1 and maps negative energy to zero, while
      // collide_eg initializes j1=0 and maps negative energy to one.
      const std::size_t charged_class = ge ? signed_bucket
                                            : 1u - signed_bucket;
      result->class_luminosity[ge ? 6u + charged_class
                                    : 3u * charged_class + 2u] += pair_weight;
      if (ge) result->lumi_ge += pair_weight;
      else result->lumi_eg += pair_weight;
    }
  }
}

void add_factored_ge(std::size_t photon_count,
                     std::size_t particle_count,
                     const PhotonStats& photon_stats,
                     const ChargedStats& charged_stats,
                     bool ge, Result* result) {
  add_product(static_cast<std::uint64_t>(photon_count),
              static_cast<std::uint64_t>(particle_count),
              ge ? &result->ge_pair_opportunities
                 : &result->eg_pair_opportunities);

  for (std::size_t signed_bucket = 0; signed_bucket < 2; ++signed_bucket) {
    const double value = static_cast<double>(
        photon_stats.positive_energy_weight *
        charged_stats.class_weight[signed_bucket]);
    const std::size_t charged_class = ge ? signed_bucket
                                         : 1u - signed_bucket;
    result->class_luminosity[ge ? 6u + charged_class
                                  : 3u * charged_class + 2u] += value;
  }
  const double total = static_cast<double>(
      photon_stats.positive_energy_weight *
      (charged_stats.class_weight[0] + charged_stats.class_weight[1]));
  if (ge) result->lumi_ge += total;
  else result->lumi_eg += total;
}

void add_reference_gg(const PhotonView& first,
                      const PhotonView& second,
                      double gg_smin, Result* result) {
  add_product(static_cast<std::uint64_t>(first.size()),
              static_cast<std::uint64_t>(second.size()),
              &result->gg_pair_opportunities);
  for (std::size_t a = 0; a < first.size(); ++a) {
    const PhotonSample first_sample = first.at(a);
    for (std::size_t b = 0; b < second.size(); ++b) {
      count_cartesian_visit(result);
      const PhotonSample second_sample = second.at(b);
      const float pair_weight = first_sample.weight * second_sample.weight;
      result->class_luminosity[8] += pair_weight;
      const double s_gg =
          (4.0 * static_cast<double>(first_sample.energy)) *
          static_cast<double>(second_sample.energy);
      if (!(s_gg > 0.0)) continue;
      result->lumi_gg += pair_weight;
      if (gg_above_threshold(first_sample.energy, second_sample.energy,
                             gg_smin))
        result->lumi_gg_high += pair_weight;
    }
  }
}

void add_factored_gg(const PhotonView& first,
                     const PhotonStats& first_stats,
                     const PhotonView& second,
                     const PhotonStats& second_stats,
                     double gg_smin,
                     Workspace& workspace,
                     Result* result) {
  add_product(static_cast<std::uint64_t>(first.size()),
              static_cast<std::uint64_t>(second.size()),
              &result->gg_pair_opportunities);
  result->class_luminosity[8] +=
      static_cast<double>(first_stats.all_weight * second_stats.all_weight);

  const double positive_total = static_cast<double>(
      first_stats.positive_energy_weight *
      second_stats.positive_energy_weight);
  result->lumi_gg += positive_total;
  if (first_stats.positive_energy_count == 0u ||
      second_stats.positive_energy_count == 0u)
    return;
  if (gg_smin == 0.0) {
    result->lumi_gg_high += positive_total;
    return;
  }

  std::sort(workspace.sorted_positive_photons.begin(),
            workspace.sorted_positive_photons.end(),
            [](const Workspace::WeightedPhoton& left,
               const Workspace::WeightedPhoton& right) {
              return left.energy < right.energy ||
                     (left.energy == right.energy &&
                      left.ordinal < right.ordinal);
            });
  workspace.suffix_weights.resize(
      workspace.sorted_positive_photons.size() + 1u, 0.0L);
  for (std::size_t i = workspace.sorted_positive_photons.size(); i-- > 0;)
    workspace.suffix_weights[i] = workspace.suffix_weights[i + 1u] +
                                  workspace.sorted_positive_photons[i].weight;

  long double high = 0.0L;
  for (std::size_t i = 0; i < workspace.positive_query_photons.size(); ++i) {
    const Workspace::WeightedPhoton& left =
        workspace.positive_query_photons[i];
    const double threshold = gg_smin / (4.0 * left.energy);
    const std::vector<Workspace::WeightedPhoton>::const_iterator selected =
        std::upper_bound(
            workspace.sorted_positive_photons.begin(),
            workspace.sorted_positive_photons.end(), threshold,
            [](double value, const Workspace::WeightedPhoton& right) {
              return value < right.energy;
            });
    std::size_t offset = static_cast<std::size_t>(
        selected - workspace.sorted_positive_photons.begin());
    // Division is only a boundary estimate.  Correct adjacent values using
    // the shared direct-product predicate evaluated by GRID::collide_gg.
    while (offset > 0u &&
           gg_above_threshold(left.energy,
                              workspace.sorted_positive_photons[offset - 1u]
                                  .energy,
                              gg_smin))
      --offset;
    while (offset < workspace.sorted_positive_photons.size() &&
           !gg_above_threshold(left.energy,
                               workspace.sorted_positive_photons[offset].energy,
                               gg_smin))
      ++offset;
    high += static_cast<long double>(left.weight) *
            workspace.suffix_weights[offset];
  }
  result->lumi_gg_high += static_cast<double>(high);
}

PhotonSample load_vector_photon(const void* context, std::size_t index) {
  const std::vector<PhotonSample>& samples =
      *static_cast<const std::vector<PhotonSample>*>(context);
  return samples[index];
}

ChargedSample load_vector_particle(const void* context, std::size_t index) {
  const std::vector<ChargedSample>& samples =
      *static_cast<const std::vector<ChargedSample>*>(context);
  return samples[index];
}

}  // namespace

PhotonView::PhotonView() : context_(0), size_(0), loader_(0) {}

PhotonView::PhotonView(const void* context, std::size_t size, Loader loader)
    : context_(context), size_(size), loader_(loader) {}

std::size_t PhotonView::size() const { return size_; }

bool PhotonView::valid() const {
  return size_ == 0u || (context_ != 0 && loader_ != 0);
}

PhotonSample PhotonView::at(std::size_t index) const {
  return loader_(context_, index);
}

ChargedView::ChargedView() : context_(0), size_(0), loader_(0) {}

ChargedView::ChargedView(const void* context, std::size_t size, Loader loader)
    : context_(context), size_(size), loader_(loader) {}

std::size_t ChargedView::size() const { return size_; }

bool ChargedView::valid() const {
  return size_ == 0u || (context_ != 0 && loader_ != 0);
}

ChargedSample ChargedView::at(std::size_t index) const {
  return loader_(context_, index);
}

Workspace::Workspace()
    : positive_query_photons(),
      sorted_positive_photons(),
      suffix_weights() {}

void Workspace::reserve(std::size_t maximum_query_photons,
                        std::size_t maximum_sorted_photons) {
  if (maximum_query_photons == std::numeric_limits<std::size_t>::max() ||
      maximum_sorted_photons == std::numeric_limits<std::size_t>::max())
    throw std::length_error(
        "legacy photon-luminosity workspace size overflow");
  positive_query_photons.reserve(maximum_query_photons);
  sorted_positive_photons.reserve(maximum_sorted_photons);
  suffix_weights.reserve(maximum_sorted_photons + 1u);
}

std::size_t Workspace::query_capacity() const {
  return positive_query_photons.capacity();
}

std::size_t Workspace::weighted_capacity() const {
  return sorted_positive_photons.capacity();
}

std::size_t Workspace::suffix_capacity() const {
  return suffix_weights.capacity();
}

Result::Result()
    : class_luminosity{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
      lumi_ge(0.0),
      lumi_eg(0.0),
      lumi_gg(0.0),
      lumi_gg_high(0.0),
      ge_pair_opportunities(0),
      eg_pair_opportunities(0),
      gg_pair_opportunities(0),
      cartesian_pairs_visited(0) {}

void Result::merge(const Result& other) {
  for (std::size_t i = 0; i < class_luminosity.size(); ++i)
    class_luminosity[i] += other.class_luminosity[i];
  lumi_ge += other.lumi_ge;
  lumi_eg += other.lumi_eg;
  lumi_gg += other.lumi_gg;
  lumi_gg_high += other.lumi_gg_high;
  add_product(other.ge_pair_opportunities, 1u, &ge_pair_opportunities);
  add_product(other.eg_pair_opportunities, 1u, &eg_pair_opportunities);
  add_product(other.gg_pair_opportunities, 1u, &gg_pair_opportunities);
  add_product(other.cartesian_pairs_visited, 1u, &cartesian_pairs_visited);
}

Features::Features()
    : do_photon_lumi(1),
      do_lumi(0),
      do_jets(0),
      do_compt(0),
      do_compt_phot(0),
      do_pairs(0),
      do_muons(0),
      do_hadrons(0),
      do_coherent(0),
      do_trident(0),
      do_cross(0),
      do_bhabhas(0),
      do_prod(0),
      do_tertphot(0) {}

bool eligible(const Features& f, std::string* reason) {
  const char* failure = 0;
  if (!f.do_photon_lumi) failure = "photon luminosity is disabled";
  else if ((f.do_lumi & 2) != 0)
    failure = "gamma-e luminosity heap is enabled";
  else if ((f.do_lumi & 4) != 0)
    failure = "gamma-gamma luminosity heap is enabled";
  else if (f.do_jets) failure = "jet production is enabled";
  else if (f.do_compt) failure = "Compton production is enabled";
  else if (f.do_compt_phot)
    failure = "virtual-photon Compton production is enabled";
  else if (f.do_pairs) failure = "Breit-Wheeler pair production is enabled";
  else if (f.do_muons) failure = "muon-pair production is enabled";
  else if (f.do_hadrons) failure = "hadron production is enabled";
  else if (f.do_coherent) failure = "coherent background production is enabled";
  else if (f.do_trident) failure = "trident background production is enabled";
  else if (f.do_cross) failure = "cross-section event production is enabled";
  else if (f.do_bhabhas) failure = "Bhabha production is enabled";
  else if (f.do_prod) failure = "legacy production hook is enabled";
  else if (f.do_tertphot) failure = "tertiary-photon production is enabled";
  if (reason) *reason = failure ? failure : std::string();
  return failure == 0;
}

Result compute_cell(const std::vector<PhotonSample>& beam1_photons,
                    const std::vector<ChargedSample>& beam2_particles,
                    const std::vector<PhotonSample>& beam2_photons,
                    const std::vector<ChargedSample>& beam1_particles,
                    double gg_smin,
                    Algorithm algorithm) {
  const PhotonView first_photons(
      &beam1_photons, beam1_photons.size(), &load_vector_photon);
  const ChargedView second_particles(
      &beam2_particles, beam2_particles.size(), &load_vector_particle);
  const PhotonView second_photons(
      &beam2_photons, beam2_photons.size(), &load_vector_photon);
  const ChargedView first_particles(
      &beam1_particles, beam1_particles.size(), &load_vector_particle);
  Workspace workspace;
  Result result;
  accumulate_cell(first_photons, second_particles, second_photons,
                  first_particles, gg_smin, algorithm, workspace, &result);
  return result;
}

void accumulate_cell(const PhotonView& beam1_photons,
                     const ChargedView& beam2_particles,
                     const PhotonView& beam2_photons,
                     const ChargedView& beam1_particles,
                     double gg_smin,
                     Algorithm algorithm,
                     Workspace& workspace,
                     Result* result) {
  if (result == 0)
    throw std::invalid_argument(
        "legacy photon-luminosity result must not be null");
  if (algorithm != Algorithm::CartesianReference &&
      algorithm != Algorithm::FactoredExact)
    throw std::invalid_argument("unknown legacy photon-luminosity algorithm");
  if (!std::isfinite(gg_smin) || gg_smin < 0.0)
    throw std::invalid_argument("gg_smin must be finite and non-negative");
  if (!beam1_photons.valid() || !beam2_photons.valid() ||
      !beam1_particles.valid() || !beam2_particles.valid())
    throw std::invalid_argument(
        "legacy photon-luminosity view is invalid");

  workspace.positive_query_photons.clear();
  workspace.sorted_positive_photons.clear();
  workspace.suffix_weights.clear();
  // Standalone callers get the same high-water allocation behavior as GRID;
  // GRID itself has already reserved the largest cell before this loop.
  if (algorithm == Algorithm::FactoredExact && gg_smin > 0.0)
    workspace.reserve(beam1_photons.size(), beam2_photons.size());

  const PhotonStats beam1_photon_stats = scan_photons(
      beam1_photons, "beam-1 photon",
      algorithm == Algorithm::FactoredExact && gg_smin > 0.0
          ? &workspace.positive_query_photons
          : 0);
  const ChargedStats beam2_particle_stats =
      scan_particles(beam2_particles, "beam-2 charged particle");
  const PhotonStats beam2_photon_stats = scan_photons(
      beam2_photons, "beam-2 photon",
      algorithm == Algorithm::FactoredExact && gg_smin > 0.0
          ? &workspace.sorted_positive_photons
          : 0);
  const ChargedStats beam1_particle_stats =
      scan_particles(beam1_particles, "beam-1 charged particle");
  // The legacy routines explicitly round every pair weight to float before
  // adding it to RESULTS.  Reassociation is a harmless last-bit change only
  // while those pair products remain finite normal floats.  Refuse exotic
  // weights for which overflow/underflow would change the represented model.
  require_representable_pair_weights(
      beam1_photon_stats.positive_energy_weights,
      beam2_particle_stats.all_weights, "gamma-e");
  require_representable_pair_weights(
      beam2_photon_stats.positive_energy_weights,
      beam1_particle_stats.all_weights, "e-gamma");
  require_representable_pair_weights(
      beam1_photon_stats.all_weights,
      beam2_photon_stats.all_weights, "gamma-gamma");

  if (algorithm == Algorithm::CartesianReference) {
    add_reference_ge(beam1_photons, beam2_particles, true, result);
    add_reference_ge(beam2_photons, beam1_particles, false, result);
    add_reference_gg(beam1_photons, beam2_photons, gg_smin, result);
  } else {
    add_factored_ge(beam1_photons.size(), beam2_particles.size(),
                    beam1_photon_stats, beam2_particle_stats, true, result);
    add_factored_ge(beam2_photons.size(), beam1_particles.size(),
                    beam2_photon_stats, beam1_particle_stats, false, result);
    add_factored_gg(beam1_photons, beam1_photon_stats,
                    beam2_photons, beam2_photon_stats,
                    gg_smin, workspace, result);
  }
}

const char* algorithm_name(Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::CartesianReference: return "cartesian-reference";
    case Algorithm::FactoredExact: return "factorized-exact-v2";
  }
  return "unknown";
}

}  // namespace legacy_photon_lumi
}  // namespace feebl
