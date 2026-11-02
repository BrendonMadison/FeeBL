#include "modernPhotonLumiCPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace feebl {
namespace modern {

namespace {

std::size_t class_index(LuminositySpeciesClass first,
                        LuminositySpeciesClass second) noexcept {
  return 3u * static_cast<std::size_t>(first) +
         static_cast<std::size_t>(second);
}

LuminositySpeciesClass lepton_class(ParticleSpecies species) {
  if (species == ParticleSpecies::Electron)
    return LuminositySpeciesClass::Electron;
  if (species == ParticleSpecies::Positron)
    return LuminositySpeciesClass::Positron;
  throw std::invalid_argument(
      "photon-luminosity particle owner contains a non-lepton species");
}

void require_same_grid(const UniformGrid2D& reference,
                       const UniformGrid2D& candidate) {
  if (reference.xmin() != candidate.xmin() ||
      reference.xmax() != candidate.xmax() ||
      reference.ymin() != candidate.ymin() ||
      reference.ymax() != candidate.ymax() ||
      reference.nx() != candidate.nx() ||
      reference.ny() != candidate.ny() ||
      reference.active_nx() != candidate.active_nx() ||
      reference.active_ny() != candidate.active_ny() ||
      reference.cell_count() != candidate.cell_count() ||
      reference.mapping_mode() != candidate.mapping_mode())
    throw std::invalid_argument(
        "photon-luminosity CSR grids are not identical");
}

void require_owner(const BeamSoA& beam, const CellCSR& cells,
                   std::uint32_t beam_id, std::int8_t direction,
                   bool photons) {
  const ValidationResult beam_validation = beam.validate();
  if (!beam_validation)
    throw std::invalid_argument("invalid photon-luminosity BeamSoA: " +
                                beam_validation.message);
  const ValidationResult cell_validation = cells.validate(beam);
  if (!cell_validation)
    throw std::invalid_argument("invalid/stale photon-luminosity CellCSR: " +
                                cell_validation.message);
  for (std::size_t i = 0; i < static_cast<std::size_t>(beam.size()); ++i) {
    if (beam.beam_id()[i] != beam_id || beam.direction()[i] != direction)
      throw std::invalid_argument(
          "photon-luminosity beam identity/direction mismatch");
    if (photons) {
      if (beam.species()[i] != ParticleSpecies::Photon ||
          beam.charge_e()[i] != 0)
        throw std::invalid_argument(
            "photon owner contains a charged or non-photon object");
    } else {
      (void)lepton_class(beam.species()[i]);
    }
  }
}

void require_globally_distinct_ids(const BeamSoA& a, const BeamSoA& b,
                                   const BeamSoA& c, const BeamSoA& d) {
  std::unordered_set<StableParticleId> ids;
  std::uint64_t total64 = 0;
  const BeamSoA* owners[] = {&a, &b, &c, &d};
  for (const BeamSoA* owner : owners) {
    std::uint64_t updated = 0;
    if (!checked_add(total64, owner->size(), updated))
      throw std::overflow_error("global stable-ID count overflow");
    total64 = updated;
  }
  if (!fits_size_t(total64))
    throw std::length_error("global stable-ID set is not representable");
  ids.reserve(static_cast<std::size_t>(total64));
  for (const BeamSoA* owner : owners)
    for (StableParticleId id : owner->stable_ids())
      if (!ids.insert(id).second)
        throw std::invalid_argument(
            "stable particle ID is duplicated across luminosity owners");
}

void count_cartesian_visit(PhotonLuminosityResult& result) {
  if (result.cartesian_pairs_visited ==
      std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(
        "photon-luminosity Cartesian visit accounting overflow");
  ++result.cartesian_pairs_visited;
}

void add_checked_product(std::uint64_t left, std::uint64_t right,
                         std::uint64_t& destination) {
  std::uint64_t product = 0;
  std::uint64_t updated = 0;
  if (!checked_multiply(left, right, product) ||
      !checked_add(destination, product, updated))
    throw std::overflow_error(
        "photon-luminosity logical pair accounting overflow");
  destination = updated;
}

struct WeightedEnergy {
  double energy;
  double weight;
  StableParticleId id;
};

bool legacy_gg_high(double first_energy, double second_energy,
                    double smin) noexcept {
  // Preserve GRID::collide_gg's evaluated predicate, including its rounded
  // sqrt-and-square, rather than silently substituting the algebraically
  // simplified 4*E1*E2 comparison at a threshold boundary.
  const double ecm = std::sqrt(4.0 * first_energy * second_energy);
  return ecm * ecm > smin;
}

std::vector<WeightedEnergy> positive_photons(const BeamSoA& photons,
                                             const CellCSR& cells,
                                             CellKey cell) {
  const CellRange range = cells.cell_range(cell);
  std::vector<WeightedEnergy> result;
  result.reserve(static_cast<std::size_t>(range.size()));
  const std::vector<ParticleIndex>& members = cells.particle_indices();
  for (ArrayOffset offset = range.begin; offset < range.end; ++offset) {
    const std::size_t index = static_cast<std::size_t>(
        members[static_cast<std::size_t>(offset)]);
    if (photons.energy()[index] > 0.0)
      result.push_back({photons.energy()[index],
                        photons.macro_weight()[index],
                        photons.stable_ids()[index]});
  }
  return result;
}

void add_cartesian_ge(const BeamSoA& photons, const CellCSR& photon_cells,
                      const BeamSoA& particles, const CellCSR& particle_cells,
                      CellKey cell, bool ge,
                      PhotonLuminosityResult& result) {
  const CellRange photon_range = photon_cells.cell_range(cell);
  const CellRange particle_range = particle_cells.cell_range(cell);
  add_checked_product(photon_range.size(), particle_range.size(),
                      ge ? result.ge_pair_opportunities
                         : result.eg_pair_opportunities);
  const std::vector<ParticleIndex>& photon_members =
      photon_cells.particle_indices();
  const std::vector<ParticleIndex>& particle_members =
      particle_cells.particle_indices();
  for (ArrayOffset po = photon_range.begin; po < photon_range.end; ++po) {
    const std::size_t pi = static_cast<std::size_t>(
        photon_members[static_cast<std::size_t>(po)]);
    for (ArrayOffset eo = particle_range.begin;
         eo < particle_range.end; ++eo) {
      count_cartesian_visit(result);
      if (!(photons.energy()[pi] > 0.0)) continue;
      const std::size_t ei = static_cast<std::size_t>(
          particle_members[static_cast<std::size_t>(eo)]);
      const double weight = photons.macro_weight()[pi] *
                            particles.macro_weight()[ei];
      const LuminositySpeciesClass charged =
          lepton_class(particles.species()[ei]);
      if (ge) {
        result.add_class(LuminositySpeciesClass::Photon, charged, weight);
        result.lumi_ge += weight;
      } else {
        result.add_class(charged, LuminositySpeciesClass::Photon, weight);
        result.lumi_eg += weight;
      }
    }
  }
}

void add_factored_ge(const BeamSoA& photons, const CellCSR& photon_cells,
                     const BeamSoA& particles, const CellCSR& particle_cells,
                     CellKey cell, bool ge,
                     PhotonLuminosityResult& result) {
  const CellRange photon_range = photon_cells.cell_range(cell);
  const CellRange particle_range = particle_cells.cell_range(cell);
  add_checked_product(photon_range.size(), particle_range.size(),
                      ge ? result.ge_pair_opportunities
                         : result.eg_pair_opportunities);
  const std::vector<ParticleIndex>& photon_members =
      photon_cells.particle_indices();
  const std::vector<ParticleIndex>& particle_members =
      particle_cells.particle_indices();
  double photon_weight = 0.0;
  for (ArrayOffset offset = photon_range.begin; offset < photon_range.end;
       ++offset) {
    const std::size_t index = static_cast<std::size_t>(
        photon_members[static_cast<std::size_t>(offset)]);
    if (photons.energy()[index] > 0.0)
      photon_weight += photons.macro_weight()[index];
  }
  std::array<double, 2> charged_weight = {{0.0, 0.0}};
  for (ArrayOffset offset = particle_range.begin; offset < particle_range.end;
       ++offset) {
    const std::size_t index = static_cast<std::size_t>(
        particle_members[static_cast<std::size_t>(offset)]);
    const std::size_t species = static_cast<std::size_t>(
        lepton_class(particles.species()[index]));
    charged_weight[species] += particles.macro_weight()[index];
  }
  for (std::size_t species = 0; species < charged_weight.size(); ++species) {
    const double weight = photon_weight * charged_weight[species];
    const LuminositySpeciesClass charged =
        static_cast<LuminositySpeciesClass>(species);
    if (ge)
      result.add_class(LuminositySpeciesClass::Photon, charged, weight);
    else
      result.add_class(charged, LuminositySpeciesClass::Photon, weight);
  }
  const double total_charged = charged_weight[0] + charged_weight[1];
  if (ge) result.lumi_ge += photon_weight * total_charged;
  else result.lumi_eg += photon_weight * total_charged;
}

void add_cartesian_gg(const BeamSoA& first, const CellCSR& first_cells,
                      const BeamSoA& second, const CellCSR& second_cells,
                      CellKey cell, double smin,
                      PhotonLuminosityResult& result) {
  const CellRange first_range = first_cells.cell_range(cell);
  const CellRange second_range = second_cells.cell_range(cell);
  add_checked_product(first_range.size(), second_range.size(),
                      result.gg_pair_opportunities);
  const std::vector<ParticleIndex>& first_members =
      first_cells.particle_indices();
  const std::vector<ParticleIndex>& second_members =
      second_cells.particle_indices();
  for (ArrayOffset ao = first_range.begin; ao < first_range.end; ++ao) {
    const std::size_t ai = static_cast<std::size_t>(
        first_members[static_cast<std::size_t>(ao)]);
    for (ArrayOffset bo = second_range.begin; bo < second_range.end; ++bo) {
      count_cartesian_visit(result);
      const std::size_t bi = static_cast<std::size_t>(
          second_members[static_cast<std::size_t>(bo)]);
      const double weight = first.macro_weight()[ai] *
                            second.macro_weight()[bi];
      result.add_class(LuminositySpeciesClass::Photon,
                       LuminositySpeciesClass::Photon, weight);
      const double ecm =
          std::sqrt(4.0 * first.energy()[ai] * second.energy()[bi]);
      if (!(ecm > 0.0)) continue;
      result.lumi_gg += weight;
      if (ecm * ecm > smin) result.lumi_gg_high += weight;
    }
  }
}

void add_factored_gg(const BeamSoA& first, const CellCSR& first_cells,
                     const BeamSoA& second, const CellCSR& second_cells,
                     CellKey cell, double smin,
                     PhotonLuminosityResult& result) {
  const CellRange first_range = first_cells.cell_range(cell);
  const CellRange second_range = second_cells.cell_range(cell);
  add_checked_product(first_range.size(), second_range.size(),
                      result.gg_pair_opportunities);
  std::vector<WeightedEnergy> a = positive_photons(first, first_cells, cell);
  std::vector<WeightedEnergy> b = positive_photons(second, second_cells, cell);
  double all_wa = 0.0;
  double all_wb = 0.0;
  const std::vector<ParticleIndex>& first_members =
      first_cells.particle_indices();
  const std::vector<ParticleIndex>& second_members =
      second_cells.particle_indices();
  for (ArrayOffset offset = first_range.begin; offset < first_range.end;
       ++offset)
    all_wa += first.macro_weight()[static_cast<std::size_t>(
        first_members[static_cast<std::size_t>(offset)])];
  for (ArrayOffset offset = second_range.begin; offset < second_range.end;
       ++offset)
    all_wb += second.macro_weight()[static_cast<std::size_t>(
        second_members[static_cast<std::size_t>(offset)])];
  double wa = 0.0;
  double wb = 0.0;
  for (const WeightedEnergy& value : a) wa += value.weight;
  for (const WeightedEnergy& value : b) wb += value.weight;
  const double class_total = all_wa * all_wb;
  const double total = wa * wb;
  result.add_class(LuminositySpeciesClass::Photon,
                   LuminositySpeciesClass::Photon, class_total);
  result.lumi_gg += total;
  if (a.empty() || b.empty()) return;
  if (smin == 0.0) {
    result.lumi_gg_high += total;
    return;
  }

  // The strict legacy predicate is 4*Ea*Eb > smin.  Sort the second list by
  // (energy, stable ID), then upper_bound the exact threshold and use a suffix
  // weight.  This preserves the physical selected pair set while reducing the
  // work to O((Na+Nb) log Nb).
  std::sort(b.begin(), b.end(), [](const WeightedEnergy& left,
                                   const WeightedEnergy& right) {
    return left.energy < right.energy ||
           (left.energy == right.energy && left.id < right.id);
  });
  std::vector<double> suffix(b.size() + 1, 0.0);
  for (std::size_t i = b.size(); i-- > 0;)
    suffix[i] = suffix[i + 1] + b[i].weight;
  for (const WeightedEnergy& left : a) {
    const double threshold = smin / (4.0 * left.energy);
    const auto first_selected = std::upper_bound(
        b.begin(), b.end(), threshold,
        [](double value, const WeightedEnergy& right) {
          return value < right.energy;
        });
    std::size_t offset = static_cast<std::size_t>(
        first_selected - b.begin());
    // Division gives a fast boundary estimate, but its rounding need not be
    // the inverse of the legacy sqrt-and-square predicate.  Correct the
    // adjacent monotone boundary using that predicate so the selected pair set
    // is exact even for nextafter probes.
    while (offset > 0 &&
           legacy_gg_high(left.energy, b[offset - 1].energy, smin))
      --offset;
    while (offset < b.size() &&
           !legacy_gg_high(left.energy, b[offset].energy, smin))
      ++offset;
    result.lumi_gg_high += left.weight * suffix[offset];
  }
}

}  // namespace

double PhotonLuminosityResult::class_value(
    LuminositySpeciesClass first,
    LuminositySpeciesClass second) const noexcept {
  return class_luminosity[class_index(first, second)];
}

void PhotonLuminosityResult::add_class(
    LuminositySpeciesClass first,
    LuminositySpeciesClass second, double value) noexcept {
  class_luminosity[class_index(first, second)] += value;
}

PhotonLuminosityResult compute_photon_luminosity_only(
    const BeamSoA& beam1_particles,
    const CellCSR& beam1_particle_cells,
    const BeamSoA& beam2_particles,
    const CellCSR& beam2_particle_cells,
    const BeamSoA& beam1_photons,
    const CellCSR& beam1_photon_cells,
    const BeamSoA& beam2_photons,
    const CellCSR& beam2_photon_cells,
    const PhotonLumiRequest& request) {
  if (request.algorithm != PhotonLumiAlgorithm::CartesianReference &&
      request.algorithm != PhotonLumiAlgorithm::FactoredExact)
    throw std::invalid_argument("unknown photon-luminosity algorithm");
  if (!std::isfinite(request.gg_smin) || request.gg_smin < 0.0)
    throw std::invalid_argument("gg_smin must be finite and non-negative");

  require_owner(beam1_particles, beam1_particle_cells, 1u, 1, false);
  require_owner(beam2_particles, beam2_particle_cells, 2u, -1, false);
  require_owner(beam1_photons, beam1_photon_cells, 1u, 1, true);
  require_owner(beam2_photons, beam2_photon_cells, 2u, -1, true);
  require_globally_distinct_ids(beam1_particles, beam2_particles,
                                beam1_photons, beam2_photons);
  const UniformGrid2D& grid = beam1_particle_cells.grid();
  require_same_grid(grid, beam2_particle_cells.grid());
  require_same_grid(grid, beam1_photon_cells.grid());
  require_same_grid(grid, beam2_photon_cells.grid());

  PhotonLuminosityResult result;
  for (CellKey cell = 0; cell < grid.cell_count(); ++cell) {
    if (request.algorithm == PhotonLumiAlgorithm::CartesianReference) {
      add_cartesian_ge(beam1_photons, beam1_photon_cells,
                       beam2_particles, beam2_particle_cells,
                       cell, true, result);
      add_cartesian_ge(beam2_photons, beam2_photon_cells,
                       beam1_particles, beam1_particle_cells,
                       cell, false, result);
      add_cartesian_gg(beam1_photons, beam1_photon_cells,
                       beam2_photons, beam2_photon_cells,
                       cell, request.gg_smin, result);
    } else {
      add_factored_ge(beam1_photons, beam1_photon_cells,
                      beam2_particles, beam2_particle_cells,
                      cell, true, result);
      add_factored_ge(beam2_photons, beam2_photon_cells,
                      beam1_particles, beam1_particle_cells,
                      cell, false, result);
      add_factored_gg(beam1_photons, beam1_photon_cells,
                      beam2_photons, beam2_photon_cells,
                      cell, request.gg_smin, result);
    }
  }
  return result;
}

}  // namespace modern
}  // namespace feebl
