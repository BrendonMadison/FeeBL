#include "modernCollisionCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using feebl::modern::BeamSoA;
using feebl::modern::CellCSR;
using feebl::modern::CellOrder;
using feebl::modern::CollisionEvaluation;
using feebl::modern::CollisionResultLane;
using feebl::modern::EePairPolicy;
using feebl::modern::ModernCollisionConfig;
using feebl::modern::ParticleRecord;
using feebl::modern::ParticleSpecies;
using feebl::modern::StableParticleId;
using feebl::modern::UniformGrid2D;
using feebl::modern::WeightedMomentLane;
using feebl::modern::evaluate_collision_luminosity;
using feebl::modern::kFeeBLElectronMassGeV;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-72s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

bool near(double a, double b, double relative = 2.0e-13,
          double absolute = 2.0e-13) {
  return std::fabs(a - b) <=
         absolute + relative * std::max(std::fabs(a), std::fabs(b));
}

template <class Function>
bool throws(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

ParticleRecord particle(StableParticleId id, double x, double energy,
                        double weight, ParticleSpecies species,
                        std::int8_t charge, std::int8_t direction,
                        double vx = 0.0, double vy = 0.0) {
  ParticleRecord p;
  p.stable_id = id;
  p.x = x;
  p.y = 0.0;
  p.z = static_cast<double>(id % 11);
  p.vx = vx;
  p.vy = vy;
  p.energy = energy;
  p.macro_weight = weight;
  p.species = species;
  p.charge_e = charge;
  p.direction = direction;
  p.beam_id = direction > 0 ? 1u : 2u;
  return p;
}

ParticleRecord electron(StableParticleId id, double x, double energy,
                        double weight, std::int8_t direction,
                        double vx = 0.0, double vy = 0.0) {
  return particle(id, x, energy, weight, ParticleSpecies::Electron, -1,
                  direction, vx, vy);
}

ParticleRecord positron(StableParticleId id, double x, double energy,
                        double weight, std::int8_t direction,
                        double vx = 0.0, double vy = 0.0) {
  return particle(id, x, energy, weight, ParticleSpecies::Positron, 1,
                  direction, vx, vy);
}

ParticleRecord neutral(StableParticleId id, double x, double weight,
                       std::int8_t direction) {
  return particle(id, x, 1.0, weight, ParticleSpecies::Other, 0, direction);
}

BeamSoA beam_of(std::initializer_list<ParticleRecord> particles) {
  BeamSoA beam(1);
  beam.append_slice(particles);
  return beam;
}

ModernCollisionConfig config(EePairPolicy policy, double threshold = 0.0,
                             std::uint64_t tile = 2) {
  ModernCollisionConfig c;
  c.ee_policy = policy;
  c.high_ecm_threshold = threshold;
  c.tile_width = tile;
  return c;
}

bool lane_exact(const CollisionResultLane& a,
                const CollisionResultLane& b) {
  return a.provenance == b.provenance &&
         a.charge_class.weight == b.charge_class.weight &&
         a.ee_weight == b.ee_weight &&
         a.ee_high_weight == b.ee_high_weight &&
         a.ecm_weighted_sum == b.ecm_weighted_sum &&
         a.ecm_squared_weighted_sum == b.ecm_squared_weighted_sum &&
         a.cartesian_pair_count == b.cartesian_pair_count &&
         a.ee_pair_count == b.ee_pair_count &&
         a.w_spectrum.lower == b.w_spectrum.lower &&
         a.w_spectrum.upper == b.w_spectrum.upper &&
         a.w_spectrum.bin_width == b.w_spectrum.bin_width &&
         a.w_spectrum.bin_weight == b.w_spectrum.bin_weight &&
         a.w_spectrum.underflow_weight == b.w_spectrum.underflow_weight &&
         a.w_spectrum.overflow_weight == b.w_spectrum.overflow_weight &&
         a.w_spectrum.moments.weight_sum ==
             b.w_spectrum.moments.weight_sum &&
         a.w_spectrum.moments.weight_squared_sum ==
             b.w_spectrum.moments.weight_squared_sum &&
         a.w_spectrum.moments.weighted_value_sum ==
             b.w_spectrum.moments.weighted_value_sum &&
         a.w_spectrum.moments.weighted_value_squared_sum ==
             b.w_spectrum.moments.weighted_value_squared_sum;
}

bool lane_near(const CollisionResultLane& a,
                const CollisionResultLane& b) {
  if (a.provenance.schema_version != b.provenance.schema_version ||
      a.provenance.ee_policy != b.provenance.ee_policy ||
      a.provenance.high_ecm_threshold !=
          b.provenance.high_ecm_threshold ||
      a.provenance.electron_mass != b.provenance.electron_mass)
    return false;
  for (std::size_t i = 0; i < a.charge_class.weight.size(); ++i)
    if (!near(a.charge_class.weight[i], b.charge_class.weight[i]))
      return false;
  if (!near(a.ee_weight, b.ee_weight) ||
      !near(a.ee_high_weight, b.ee_high_weight) ||
      !near(a.ecm_weighted_sum, b.ecm_weighted_sum) ||
      !near(a.ecm_squared_weighted_sum,
            b.ecm_squared_weighted_sum) ||
      a.cartesian_pair_count != b.cartesian_pair_count ||
      a.ee_pair_count != b.ee_pair_count ||
      a.w_spectrum.bin_weight.size() != b.w_spectrum.bin_weight.size() ||
      !near(a.w_spectrum.underflow_weight,
            b.w_spectrum.underflow_weight) ||
      !near(a.w_spectrum.overflow_weight,
            b.w_spectrum.overflow_weight) ||
      !near(a.w_spectrum.moments.weight_sum,
            b.w_spectrum.moments.weight_sum) ||
      !near(a.w_spectrum.moments.weight_squared_sum,
            b.w_spectrum.moments.weight_squared_sum) ||
      !near(a.w_spectrum.moments.weighted_value_sum,
            b.w_spectrum.moments.weighted_value_sum) ||
      !near(a.w_spectrum.moments.weighted_value_squared_sum,
            b.w_spectrum.moments.weighted_value_squared_sum))
    return false;
  for (std::size_t i = 0; i < a.w_spectrum.bin_weight.size(); ++i)
    if (!near(a.w_spectrum.bin_weight[i], b.w_spectrum.bin_weight[i]))
      return false;
  return true;
}

bool selected(EePairPolicy policy, const ParticleRecord& a,
              const ParticleRecord& b) {
  const bool family1 = a.species == ParticleSpecies::Electron ||
                       a.species == ParticleSpecies::Positron;
  const bool family2 = b.species == ParticleSpecies::Electron ||
                       b.species == ParticleSpecies::Positron;
  if (!family1 || !family2) return false;
  if (policy == EePairPolicy::AllElectronFamily) return true;
  if (policy == EePairPolicy::OppositeElectricCharge)
    return a.charge_e == -b.charge_e;
  if (policy == EePairPolicy::SameElectricCharge)
    return a.charge_e == b.charge_e;
  throw std::logic_error("direct test received unspecified policy");
}

// Independent, deliberately un-tiled direct enumeration used as an oracle for
// mixed energies, unequal weights and non-collinear momenta.
CollisionResultLane direct_enumeration(const BeamSoA& beam1,
                                       const CellCSR& cells1,
                                       const BeamSoA& beam2,
                                       const CellCSR& cells2,
                                       const ModernCollisionConfig& c) {
  CollisionResultLane result;
  result.provenance.schema_version =
      feebl::modern::kModernCollisionResultSchemaVersion;
  result.provenance.ee_policy = c.ee_policy;
  result.provenance.evaluation =
      CollisionEvaluation::TiledCartesianReference;
  result.provenance.high_ecm_threshold = c.high_ecm_threshold;
  result.provenance.electron_mass = c.electron_mass;
  if (c.w_spectrum.enabled()) {
    result.w_spectrum.lower = c.w_spectrum.lower;
    result.w_spectrum.upper = c.w_spectrum.upper;
    result.w_spectrum.bin_width =
        (c.w_spectrum.upper - c.w_spectrum.lower) /
        static_cast<double>(c.w_spectrum.bins);
    result.w_spectrum.bin_weight.assign(c.w_spectrum.bins, 0.0);
  }
  for (std::uint64_t cell = 0; cell < cells1.grid().cell_count(); ++cell) {
    const auto a = cells1.cell_range(cell);
    const auto b = cells2.cell_range(cell);
    for (std::uint64_t ia = a.begin; ia < a.end; ++ia) {
      const std::size_t p1 = static_cast<std::size_t>(
          cells1.particle_indices()[static_cast<std::size_t>(ia)]);
      const ParticleRecord r1 = beam1.particle(p1);
      for (std::uint64_t ib = b.begin; ib < b.end; ++ib) {
        const std::size_t p2 = static_cast<std::size_t>(
            cells2.particle_indices()[static_cast<std::size_t>(ib)]);
        const ParticleRecord r2 = beam2.particle(p2);
        ++result.cartesian_pair_count;
        const double weight = r1.macro_weight * r2.macro_weight;
        const std::size_t matrix =
            static_cast<std::size_t>(r1.charge_e + 1) * 3u +
            static_cast<std::size_t>(r2.charge_e + 1);
        result.charge_class.weight[matrix] += weight;
        if (!selected(c.ee_policy, r1, r2)) continue;
        ++result.ee_pair_count;
        result.ee_weight += weight;

        const long double z1 =
            static_cast<long double>(r1.direction) *
            std::sqrt(1.0L - static_cast<long double>(r1.vx) * r1.vx -
                      static_cast<long double>(r1.vy) * r1.vy);
        const long double z2 =
            static_cast<long double>(r2.direction) *
            std::sqrt(1.0L - static_cast<long double>(r2.vx) * r2.vx -
                      static_cast<long double>(r2.vy) * r2.vy);
        const long double dot =
            static_cast<long double>(r1.vx) * r2.vx +
            static_cast<long double>(r1.vy) * r2.vy + z1 * z2;
        const long double ecm2 =
            2.0L * static_cast<long double>(r1.energy) * r2.energy *
            (1.0L - dot);
        const double ecm = static_cast<double>(std::sqrt(ecm2));
        const long double p1mag = std::sqrt(
            (static_cast<long double>(r1.energy) - c.electron_mass) *
            (static_cast<long double>(r1.energy) + c.electron_mass));
        const long double p2mag = std::sqrt(
            (static_cast<long double>(r2.energy) - c.electron_mass) *
            (static_cast<long double>(r2.energy) + c.electron_mass));
        const long double w2 =
            2.0L * c.electron_mass * c.electron_mass +
            2.0L * (static_cast<long double>(r1.energy) * r2.energy -
                    p1mag * p2mag * dot);
        const double w = static_cast<double>(std::sqrt(w2));
        if (ecm > c.high_ecm_threshold) result.ee_high_weight += weight;
        result.ecm_weighted_sum += weight * ecm;
        result.ecm_squared_weighted_sum += weight * ecm * ecm;
        auto& moment = result.w_spectrum.moments;
        moment.weight_sum += weight;
        moment.weight_squared_sum += weight * weight;
        moment.weighted_value_sum += weight * w;
        moment.weighted_value_squared_sum += weight * w * w;
        if (!result.w_spectrum.configured()) continue;
        if (w < result.w_spectrum.lower) {
          result.w_spectrum.underflow_weight += weight;
        } else if (w >= result.w_spectrum.upper) {
          result.w_spectrum.overflow_weight += weight;
        } else {
          std::size_t bin = static_cast<std::size_t>(
              (w - result.w_spectrum.lower) /
              result.w_spectrum.bin_width);
          if (bin >= result.w_spectrum.bin_weight.size())
            bin = result.w_spectrum.bin_weight.size() - 1;
          result.w_spectrum.bin_weight[bin] += weight;
        }
      }
    }
  }
  return result;
}

void test_empty_and_one_pair() {
  std::printf("\nEmpty and one-pair hand checks\n");
  UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 2, 1);
  BeamSoA empty = beam_of({});
  BeamSoA one1 = beam_of({electron(9, -0.5, 5.0, 2.0, 1)});
  BeamSoA one2 = beam_of({positron(4, -0.5, 5.0, 3.0, -1)});
  CellCSR ce(empty, grid);
  CellCSR c1(one1, grid);
  CellCSR c2(one2, grid);
  ModernCollisionConfig c = config(EePairPolicy::OppositeElectricCharge,
                                   10.0, 1);
  c.w_spectrum = {9.0, 11.0, 4};

  const CollisionResultLane zero = evaluate_collision_luminosity(
      empty, ce, one2, c2, c,
      CollisionEvaluation::TiledCartesianReference);
  check(zero.cartesian_pair_count == 0 && zero.ee_pair_count == 0 &&
            zero.ee_weight == 0.0 && zero.w_spectrum.bin_weight.size() == 4 &&
            static_cast<bool>(zero.validate()),
        "empty cell pair produces a shaped but entirely zero result lane");

  const CollisionResultLane one = evaluate_collision_luminosity(
      one1, c1, one2, c2, c,
      CollisionEvaluation::TiledCartesianReference);
  check(one.cartesian_pair_count == 1 && one.ee_pair_count == 1,
        "one occupied cell enumerates exactly one Cartesian/e-e pair");
  check(one.charge_class.at(-1, 1) == 6.0 && one.ee_weight == 6.0,
        "charge matrix and e-e total carry the hand weight 2*3");
  check(one.ecm_weighted_sum == 60.0 &&
            one.ecm_squared_weighted_sum == 600.0,
        "head-on 5+5 GeV pair gives exact weighted ECM moments");
  check(one.ee_high_weight == 0.0,
        "high-energy cut is strict at the exact ECM threshold edge");
  check(near(one.w_spectrum.moments.mean(), 10.0) &&
            one.w_spectrum.moments.variance() == 0.0 &&
            one.w_spectrum.moments.effective_count() == 1.0,
        "finite-mass head-on equal-energy W and its moments are exact");
  check(one.w_spectrum.bin_weight[2] == 6.0,
        "W=10 enters the half-open bin beginning at 10");

  c.high_ecm_threshold = std::nextafter(10.0, 0.0);
  const CollisionResultLane below_edge = evaluate_collision_luminosity(
      one1, c1, one2, c2, c,
      CollisionEvaluation::TiledCartesianReference);
  check(below_edge.ee_high_weight == 6.0,
        "moving threshold one double below the edge includes the pair");

  BeamSoA parallel1 = beam_of(
      {electron(31, -0.5, 5.0, 1.0, 1, 0.6, 0.2)});
  BeamSoA parallel2 = beam_of(
      {positron(32, -0.5, 5.0, 1.0, 1, 0.6, 0.2)});
  CellCSR parallel_cells1(parallel1, grid);
  CellCSR parallel_cells2(parallel2, grid);
  c.high_ecm_threshold = 0.0;
  c.w_spectrum = {0.0, 1.0, 2};
  const CollisionResultLane parallel = evaluate_collision_luminosity(
      parallel1, parallel_cells1, parallel2, parallel_cells2, c,
      CollisionEvaluation::TiledCartesianReference);
  check(near(parallel.ecm_weighted_sum, 0.0, 0.0, 1.0e-7) &&
            near(parallel.w_spectrum.moments.mean(),
                 2.0 * kFeeBLElectronMassGeV, 1.0e-9, 1.0e-12),
        "near-cancelling parallel invariant retains the finite 2m threshold");
}

void test_spectrum_boundaries() {
  std::printf("\nExact finite-mass spectrum underflow, bins and overflow\n");
  UniformGrid2D grid(-1.5, 1.5, -1.0, 1.0, 3, 1);
  BeamSoA a = beam_of({electron(30, -1.0, 2.0, 2.0, 1),
                       electron(10, 0.0, 5.0, 3.0, 1),
                       electron(20, 1.0, 8.0, 4.0, 1)});
  BeamSoA b = beam_of({positron(60, -1.0, 2.0, 1.0, -1),
                       positron(40, 0.0, 5.0, 1.0, -1),
                       positron(50, 1.0, 8.0, 1.0, -1)});
  CellCSR ca(a, grid);
  CellCSR cb(b, grid);
  ModernCollisionConfig c = config(EePairPolicy::OppositeElectricCharge);
  c.w_spectrum = {5.0, 15.0, 5};
  const CollisionResultLane lane = evaluate_collision_luminosity(
      a, ca, b, cb, c, CollisionEvaluation::TiledCartesianReference);
  check(lane.cartesian_pair_count == 3 && lane.ee_pair_count == 3 &&
            lane.ee_weight == 9.0,
        "three isolated cells produce three pairs with total weight 9");
  check(lane.w_spectrum.underflow_weight == 2.0 &&
            lane.w_spectrum.bin_weight[2] == 3.0 &&
            lane.w_spectrum.overflow_weight == 4.0,
        "W=4,10,16 populate underflow, central bin and overflow exactly");
  check(near(lane.w_spectrum.moments.mean(), 102.0 / 9.0),
        "spectrum retains weighted W moments across under/overflow");
}

struct MixedFixture {
  UniformGrid2D grid;
  BeamSoA beam1;
  BeamSoA beam2;

  MixedFixture()
      : grid(-1.0, 1.0, -1.0, 1.0, 2, 1),
        beam1(beam_of({electron(40, -0.7, 4.0, 2.0, 1, 0.10, 0.02),
                       positron(10, -0.7, 6.0, 0.5, 1, -0.15, 0.03),
                       neutral(30, -0.7, 1.25, 1),
                       electron(20, 0.6, 9.0, 1.5, 1, 0.04, -0.06)})),
        beam2(beam_of({positron(8, -0.7, 5.0, 3.0, -1, -0.20, 0.01),
                       electron(50, -0.7, 7.0, 4.0, -1, 0.05, -0.08),
                       neutral(2, -0.7, 2.0, -1),
                       positron(70, 0.6, 10.0, 0.25, -1, -0.03, 0.02)})) {}
};

void test_policy_mapping_and_direct_enumeration() {
  std::printf("\nExplicit legacy provenance policy and mixed-charge oracle\n");
  MixedFixture f;
  CellCSR a(f.beam1, f.grid);
  CellCSR b(f.beam2, f.grid);

  ModernCollisionConfig all =
      config(EePairPolicy::AllElectronFamily, 12.0, 2);
  all.w_spectrum = {0.0, 30.0, 12};
  const CollisionResultLane actual = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, all,
      CollisionEvaluation::TiledCartesianReference);
  const CollisionResultLane direct =
      direct_enumeration(f.beam1, a, f.beam2, b, all);
  check(lane_near(actual, direct),
        "tiled kernel matches independent direct mixed-angle enumeration");
  check(actual.cartesian_pair_count == 10 && actual.ee_pair_count == 5 &&
            near(actual.ee_weight, 17.875),
        "primary-like policy accepts all five electron-family pairs only");
  check(near(actual.charge_class.at(-1, 1), 6.375) &&
            near(actual.charge_class.at(-1, -1), 8.0) &&
            near(actual.charge_class.at(0, 0), 2.5),
        "3x3 physical charge matrix includes charged and neutral classes");

  ModernCollisionConfig same = all;
  same.ee_policy = EePairPolicy::SameElectricCharge;
  const CollisionResultLane same_lane = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, same,
      CollisionEvaluation::TiledCartesianReference);
  check(same_lane.ee_pair_count == 2 && near(same_lane.ee_weight, 9.5),
        "same-electric-charge policy selects e-e- and e+e+ explicitly");

  ModernCollisionConfig opposite = all;
  opposite.ee_policy = EePairPolicy::OppositeElectricCharge;
  const CollisionResultLane opposite_lane = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, opposite,
      CollisionEvaluation::TiledCartesianReference);
  check(opposite_lane.ee_pair_count == 3 && near(opposite_lane.ee_weight, 8.375),
        "opposite-electric-charge policy selects the physical e+e- category");
  check(near(all.ee_policy == EePairPolicy::AllElectronFamily
                 ? actual.ee_weight
                 : 0.0,
             same_lane.ee_weight + opposite_lane.ee_weight),
        "explicit same/opposite categories partition the legacy primary-like total");
}

void test_factoring_and_tile_invariance() {
  std::printf("\nFactored sufficient statistics and canonical tile order\n");
  MixedFixture f;
  CellCSR a(f.beam1, f.grid);
  CellCSR b(f.beam2, f.grid);
  ModernCollisionConfig c =
      config(EePairPolicy::AllElectronFamily, 8.0, 1);
  c.w_spectrum = {0.0, 30.0, 16};

  const CollisionResultLane tile1 = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, c,
      CollisionEvaluation::TiledCartesianReference);
  c.tile_width = 2;
  const CollisionResultLane tile2 = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, c,
      CollisionEvaluation::TiledCartesianReference);
  c.tile_width = 4096;
  const CollisionResultLane tile4096 = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, c,
      CollisionEvaluation::TiledCartesianReference);
  check(lane_exact(tile1, tile2) && lane_exact(tile1, tile4096),
        "tile widths 1, 2 and 4096 preserve every result bit");

  const CollisionResultLane factored = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, c,
      CollisionEvaluation::FactoredExactTotals);
  check(lane_near(tile1, factored),
        "factored charge/e-e/ECM2 totals agree with Cartesian reference");
  check(tile1.ee_high_weight == factored.ee_high_weight &&
            tile1.ecm_weighted_sum == factored.ecm_weighted_sum &&
            tile1.w_spectrum.bin_weight == factored.w_spectrum.bin_weight &&
            tile1.w_spectrum.moments.weighted_value_sum ==
                factored.w_spectrum.moments.weighted_value_sum,
        "nonseparable threshold, ECM first moment and W remain Cartesian");
  check(near(tile1.ecm_squared_weighted_sum,
             factored.ecm_squared_weighted_sum),
        "separable ECM second moment is reproduced by sufficient statistics");

  std::vector<ParticleRecord> many1;
  std::vector<ParticleRecord> many2;
  for (StableParticleId id = 1; id <= 50; ++id) {
    many1.push_back(electron(id, -0.7, 5.0, 0.1, 1));
    many2.push_back(positron(100 + id, -0.7, 5.0, 0.1, -1));
  }
  BeamSoA roundoff1(1);
  BeamSoA roundoff2(1);
  roundoff1.append_slice(many1);
  roundoff2.append_slice(many2);
  CellCSR roundoff_cells1(roundoff1, f.grid);
  CellCSR roundoff_cells2(roundoff2, f.grid);
  ModernCollisionConfig all_high =
      config(EePairPolicy::OppositeElectricCharge, 0.0, 17);
  const CollisionResultLane roundoff_factored = evaluate_collision_luminosity(
      roundoff1, roundoff_cells1, roundoff2, roundoff_cells2, all_high,
      CollisionEvaluation::FactoredExactTotals);
  check(static_cast<bool>(roundoff_factored.validate()) &&
            near(roundoff_factored.ee_high_weight,
                 roundoff_factored.ee_weight, 2.0e-13, 2.0e-13),
        "factored/Cartesian rounding cannot invalidate an all-high cell");

  std::vector<ParticleRecord> parallel_many1;
  std::vector<ParticleRecord> parallel_many2;
  parallel_many1.reserve(1000);
  parallel_many2.reserve(1000);
  for (StableParticleId id = 1; id <= 1000; ++id) {
    parallel_many1.push_back(
        electron(id, -0.7, 5.0, 0.1, 1, 0.6, 0.2));
    parallel_many2.push_back(
        positron(2000 + id, -0.7, 5.0, 0.1, 1, 0.6, 0.2));
  }
  BeamSoA parallel_many_beam1(1);
  BeamSoA parallel_many_beam2(1);
  parallel_many_beam1.append_slice(parallel_many1);
  parallel_many_beam2.append_slice(parallel_many2);
  CellCSR parallel_many_cells1(parallel_many_beam1, f.grid);
  CellCSR parallel_many_cells2(parallel_many_beam2, f.grid);
  ModernCollisionConfig none_high =
      config(EePairPolicy::OppositeElectricCharge, 100.0, 31);
  const CollisionResultLane parallel_factored = evaluate_collision_luminosity(
      parallel_many_beam1, parallel_many_cells1,
      parallel_many_beam2, parallel_many_cells2, none_high,
      CollisionEvaluation::FactoredExactTotals);
  check(static_cast<bool>(parallel_factored.validate()) &&
            parallel_factored.ee_pair_count == 1000000u &&
            parallel_factored.ecm_squared_weighted_sum == 0.0,
        "factored cancellation bound accepts a large co-propagating cell");
}

BeamSoA mirrored(const BeamSoA& source) {
  BeamSoA result(source.slice_count());
  for (std::uint64_t slice = 0; slice < source.slice_count(); ++slice) {
    const auto range = source.slice_range(slice);
    std::vector<ParticleRecord> records;
    records.reserve(static_cast<std::size_t>(range.second - range.first));
    for (std::uint64_t p = range.first; p < range.second; ++p) {
      ParticleRecord record = source.particle(p);
      record.x = -record.x;
      record.y = -record.y;
      record.vx = -record.vx;
      record.vy = -record.vy;
      record.direction = static_cast<std::int8_t>(-record.direction);
      records.push_back(record);
    }
    result.append_slice(records);
  }
  return result;
}

void test_beam_exchange_and_mirror() {
  std::printf("\nBeam exchange plus spatial/momentum mirror\n");
  MixedFixture f;
  CellCSR a(f.beam1, f.grid);
  CellCSR b(f.beam2, f.grid);
  ModernCollisionConfig c =
      config(EePairPolicy::AllElectronFamily, 9.0, 3);
  c.w_spectrum = {0.0, 30.0, 10};
  const CollisionResultLane original = evaluate_collision_luminosity(
      f.beam1, a, f.beam2, b, c,
      CollisionEvaluation::TiledCartesianReference);

  BeamSoA exchanged1 = mirrored(f.beam2);
  BeamSoA exchanged2 = mirrored(f.beam1);
  CellCSR e1(exchanged1, f.grid);
  CellCSR e2(exchanged2, f.grid);
  const CollisionResultLane exchanged = evaluate_collision_luminosity(
      exchanged1, e1, exchanged2, e2, c,
      CollisionEvaluation::TiledCartesianReference);
  bool transpose = true;
  for (int q1 = -1; q1 <= 1; ++q1)
    for (int q2 = -1; q2 <= 1; ++q2)
      transpose = transpose &&
                  near(original.charge_class.at(static_cast<std::int8_t>(q1),
                                                static_cast<std::int8_t>(q2)),
                       exchanged.charge_class.at(static_cast<std::int8_t>(q2),
                                                 static_cast<std::int8_t>(q1)));
  check(transpose, "beam exchange transposes the charge-class matrix");
  check(near(original.ee_weight, exchanged.ee_weight) &&
            near(original.ee_high_weight, exchanged.ee_high_weight) &&
            near(original.ecm_weighted_sum, exchanged.ecm_weighted_sum) &&
            near(original.ecm_squared_weighted_sum,
                 exchanged.ecm_squared_weighted_sum) &&
            original.w_spectrum.bin_weight == exchanged.w_spectrum.bin_weight,
        "exchange plus full mirror preserves scalar luminosity/ECM/W observables");
}

void test_result_lane_merge() {
  std::printf("\nDeterministic result-lane merge\n");
  UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 1, 1);
  BeamSoA a = beam_of({electron(1, 0.0, 5.0, 2.0, 1)});
  BeamSoA b = beam_of({positron(2, 0.0, 5.0, 3.0, -1)});
  CellCSR ca(a, grid);
  CellCSR cb(b, grid);
  ModernCollisionConfig c = config(EePairPolicy::OppositeElectricCharge);
  c.w_spectrum = {0.0, 20.0, 4};
  const CollisionResultLane source = evaluate_collision_luminosity(
      a, ca, b, cb, c, CollisionEvaluation::TiledCartesianReference);
  CollisionResultLane merged = source;
  merged.merge_from(source);
  check(merged.ee_weight == 12.0 && merged.cartesian_pair_count == 2 &&
            merged.ee_pair_count == 2 &&
            merged.w_spectrum.moments.weight_sum == 12.0 &&
            static_cast<bool>(merged.validate()),
        "fixed-order merge adds totals, counts, bins and moments coherently");

  ModernCollisionConfig different = c;
  different.w_spectrum = {0.0, 10.0, 4};
  const CollisionResultLane incompatible = evaluate_collision_luminosity(
      a, ca, b, cb, different,
      CollisionEvaluation::TiledCartesianReference);
  check(throws([&] { merged.merge_from(incompatible); }),
        "merging incompatible W histogram shapes fails loudly");

  ModernCollisionConfig different_cut = c;
  different_cut.high_ecm_threshold = 9.0;
  const CollisionResultLane incompatible_cut = evaluate_collision_luminosity(
      a, ca, b, cb, different_cut,
      CollisionEvaluation::TiledCartesianReference);
  check(throws([&] { merged.merge_from(incompatible_cut); }),
        "merging different threshold/model provenance fails loudly");

  const CollisionResultLane incompatible_algorithm =
      evaluate_collision_luminosity(
          a, ca, b, cb, c, CollisionEvaluation::FactoredExactTotals);
  check(throws([&] { merged.merge_from(incompatible_algorithm); }),
        "merging different evaluation-algorithm provenance fails loudly");

  CollisionResultLane misclassified = source;
  misclassified.w_spectrum.bin_weight[0] += 1.0;
  check(!static_cast<bool>(misclassified.validate()),
        "lane validation detects lost or duplicated W classification weight");

  CollisionResultLane disabled_flow = source;
  disabled_flow.w_spectrum = {};
  disabled_flow.w_spectrum.moments.weight_sum = disabled_flow.ee_weight;
  disabled_flow.w_spectrum.moments.weight_squared_sum = 36.0;
  disabled_flow.w_spectrum.moments.weighted_value_sum = 60.0;
  disabled_flow.w_spectrum.moments.weighted_value_squared_sum = 600.0;
  disabled_flow.w_spectrum.underflow_weight = 1.0;
  check(!static_cast<bool>(disabled_flow.validate()),
        "disabled spectrum rejects a spurious classified under/overflow");

  CollisionResultLane very_large;
  very_large.provenance = source.provenance;
  very_large.charge_class.weight[0] = 1.0e308;
  very_large.ee_weight = 1.0e308;
  very_large.cartesian_pair_count = 1;
  very_large.ee_pair_count = 1;
  very_large.w_spectrum.moments.weight_sum = 1.0e308;
  very_large.w_spectrum.moments.weight_squared_sum = 1.0;
  const CollisionResultLane before_overflow = very_large;
  check(static_cast<bool>(very_large.validate()) &&
            throws([&] { very_large.merge_from(before_overflow); }) &&
            lane_exact(very_large, before_overflow),
        "overflowing merge fails transactionally without partial publication");

  WeightedMomentLane scaled_moment;
  scaled_moment.weight_sum = 1.0e200;
  scaled_moment.weight_squared_sum = 1.0e300;
  check(std::isfinite(scaled_moment.effective_count()) &&
            near(scaled_moment.effective_count(), 1.0e100),
        "effective-count calculation avoids an avoidable double overflow");
}

void test_invalid_inputs() {
  std::printf("\nFail-closed configuration, ordering and invariant checks\n");
  UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 2, 1);
  BeamSoA a = beam_of({electron(1, -0.5, 5.0, 1.0, 1)});
  BeamSoA b = beam_of({positron(2, -0.5, 5.0, 1.0, -1)});
  CellCSR ca(a, grid);
  CellCSR cb(b, grid);
  ModernCollisionConfig good = config(EePairPolicy::AllElectronFamily);

  ModernCollisionConfig bad = good;
  bad.ee_policy = EePairPolicy::Unspecified;
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, ca, b, cb, bad,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "unspecified legacy/physical e-e category is rejected");

  CellCSR dense(a, grid, CellOrder::DenseInput);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, dense, b, cb, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "noncanonical dense-input cell order is rejected");

  UniformGrid2D other_grid(-2.0, 2.0, -1.0, 1.0, 2, 1);
  CellCSR other(b, other_grid);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, ca, b, other, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "nonidentical beam grids are rejected");

  bad = good;
  bad.tile_width = 0;
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, ca, b, cb, bad,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "zero tile width is rejected");
  bad = good;
  bad.high_ecm_threshold = std::numeric_limits<double>::quiet_NaN();
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, ca, b, cb, bad,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "non-finite threshold is rejected");
  bad = good;
  bad.w_spectrum = {10.0, 10.0, 2};
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              a, ca, b, cb, bad,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "non-increasing W spectrum bounds are rejected");

  BeamSoA superluminal = beam_of(
      {electron(3, -0.5, 5.0, 1.0, 1, 1.0, 0.1)});
  CellCSR superluminal_cells(superluminal, grid);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              superluminal, superluminal_cells, b, cb, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "transverse direction norm above unity is rejected");

  BeamSoA submass = beam_of(
      {electron(4, -0.5, 0.5 * kFeeBLElectronMassGeV, 1.0, 1)});
  CellCSR submass_cells(submass, grid);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              submass, submass_cells, b, cb, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "electron-family energy below rest mass is rejected");

  BeamSoA bad_charge = beam_of(
      {particle(5, -0.5, 1.0, 1.0, ParticleSpecies::Other, 2, 1)});
  CellCSR bad_charge_cells(bad_charge, grid);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              bad_charge, bad_charge_cells, b, cb, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "charge outside the explicit {-1,0,+1} matrix is rejected");

  BeamSoA huge1 = beam_of(
      {electron(6, -0.5, 5.0, 1.0e308, 1)});
  BeamSoA huge2 = beam_of(
      {positron(7, -0.5, 5.0, 1.0e308, -1)});
  CellCSR huge_cells1(huge1, grid);
  CellCSR huge_cells2(huge2, grid);
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              huge1, huge_cells1, huge2, huge_cells2, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "non-finite macro-weight product fails before accumulation");

  BeamSoA stale = beam_of({electron(8, -0.5, 5.0, 1.0, 1)});
  CellCSR stale_cells(stale, grid);
  stale.x_data()[0] = 0.5;
  check(throws([&] {
          (void)evaluate_collision_luminosity(
              stale, stale_cells, b, cb, good,
              CollisionEvaluation::TiledCartesianReference);
        }),
        "coordinate mutation after CSR construction is detected");

  BeamSoA reserved_id = beam_of({electron(
      std::numeric_limits<StableParticleId>::max(), -0.5, 5.0, 1.0, 1)});
  check(!static_cast<bool>(reserved_id.validate()),
        "counter-RNG no-parent stable-ID sentinel is invalid input");
}

}  // namespace

int main() {
  std::printf("FeeBL standalone collision/luminosity kernel\n");
  test_empty_and_one_pair();
  test_spectrum_boundaries();
  test_policy_mapping_and_direct_enumeration();
  test_factoring_and_tile_invariance();
  test_beam_exchange_and_mirror();
  test_result_lane_merge();
  test_invalid_inputs();
  std::printf("\nmodern collision result: %s (%d failure%s)\n",
              failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
