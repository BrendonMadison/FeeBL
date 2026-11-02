#include "modernPhotonLumiCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using feebl::modern::BeamSoA;
using feebl::modern::CellCSR;
using feebl::modern::CellOrder;
using feebl::modern::LuminositySpeciesClass;
using feebl::modern::ParticleRecord;
using feebl::modern::ParticleSpecies;
using feebl::modern::PhotonLumiAlgorithm;
using feebl::modern::PhotonLumiRequest;
using feebl::modern::PhotonLuminosityResult;
using feebl::modern::StableParticleId;
using feebl::modern::UniformGrid2D;
using feebl::modern::compute_photon_luminosity_only;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-72s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

bool close(double actual, double expected, double relative = 2.0e-14,
           double absolute = 1.0e-14) {
  return std::fabs(actual - expected) <=
         absolute + relative * std::fabs(expected);
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

ParticleRecord make_particle(StableParticleId id, std::uint32_t beam,
                             ParticleSpecies species, double weight,
                             double x, double y, double energy = 45.6) {
  ParticleRecord value;
  value.stable_id = id;
  value.x = x;
  value.y = y;
  value.energy = energy;
  value.macro_weight = weight;
  value.species = species;
  value.charge_e = species == ParticleSpecies::Electron ? -1 : 1;
  value.direction = beam == 1 ? 1 : -1;
  value.beam_id = beam;
  return value;
}

ParticleRecord make_photon(StableParticleId id, std::uint32_t beam,
                           double energy, double weight,
                           double x, double y) {
  ParticleRecord value;
  value.stable_id = id;
  value.x = x;
  value.y = y;
  value.energy = energy;
  value.macro_weight = weight;
  value.species = ParticleSpecies::Photon;
  value.charge_e = 0;
  value.direction = beam == 1 ? 1 : -1;
  value.beam_id = beam;
  return value;
}

BeamSoA owner(std::vector<ParticleRecord> records) {
  BeamSoA result(1);
  result.append_slice(records);
  return result;
}

struct Fixture {
  UniformGrid2D grid{-1.0, 1.0, -1.0, 1.0, 2, 2};
  BeamSoA e1;
  BeamSoA e2;
  BeamSoA g1;
  BeamSoA g2;
  CellCSR e1c;
  CellCSR e2c;
  CellCSR g1c;
  CellCSR g2c;

  Fixture()
      : e1(owner({
            make_particle(101, 1, ParticleSpecies::Electron, 0.5, -0.5, -0.5),
            make_particle(102, 1, ParticleSpecies::Positron, 1.5, -0.5, -0.5),
            make_particle(103, 1, ParticleSpecies::Electron, 4.0, 0.5, 0.5)})),
        e2(owner({
            make_particle(201, 2, ParticleSpecies::Electron, 2.0, -0.5, -0.5),
            make_particle(202, 2, ParticleSpecies::Positron, 3.0, 0.5, 0.5)})),
        g1(owner({
            make_photon(301, 1, 2.0, 0.25, -0.5, -0.5),
            make_photon(302, 1, 0.0, 7.0, -0.5, -0.5),
            make_photon(303, 1, 4.0, 0.75, 0.5, 0.5)})),
        g2(owner({
            make_photon(401, 2, 2.0, 5.0, -0.5, -0.5),
            make_photon(402, 2, 3.0, 0.5, -0.5, -0.5),
            make_photon(403, 2, 1.0, 2.0, 0.5, 0.5)})),
        e1c(e1, grid, CellOrder::ByStableParticleId),
        e2c(e2, grid, CellOrder::ByStableParticleId),
        g1c(g1, grid, CellOrder::ByStableParticleId),
        g2c(g2, grid, CellOrder::ByStableParticleId) {}
};

PhotonLuminosityResult run(const Fixture& f, PhotonLumiAlgorithm algorithm,
                           double smin = 16.0) {
  PhotonLumiRequest request;
  request.algorithm = algorithm;
  request.gg_smin = smin;
  return compute_photon_luminosity_only(
      f.e1, f.e1c, f.e2, f.e2c, f.g1, f.g1c, f.g2, f.g2c,
      request);
}

void test_direct_known_answer() {
  std::printf("\nDirect luminosity-only known answers\n");
  Fixture f;
  const PhotonLuminosityResult value =
      run(f, PhotonLumiAlgorithm::CartesianReference);

  // Cell (-,-): positive g1 weight .25 against beam-2 electron weight 2;
  // positive g2 weight 5.5 against beam-1 electron/positron .5/1.5.
  // Cell (+,+): g1 .75 against beam-2 positron 3; g2 2 against beam-1
  // electron 4.  The zero-energy g1 participates only in class/opportunity
  // accounting, exactly as GRID::collide_gg does before its ecm<=0 return.
  check(close(value.lumi_ge, 0.5 + 2.25),
        "gamma-e total matches hand enumeration");
  check(close(value.lumi_eg, 5.5 * 2.0 + 2.0 * 4.0),
        "e-gamma total matches hand enumeration");
  check(close(value.class_value(LuminositySpeciesClass::Photon,
                                LuminositySpeciesClass::Electron), 0.5),
        "gamma-electron class is explicit");
  check(close(value.class_value(LuminositySpeciesClass::Photon,
                                LuminositySpeciesClass::Positron), 2.25),
        "gamma-positron class is explicit");
  check(close(value.class_value(LuminositySpeciesClass::Electron,
                                LuminositySpeciesClass::Photon),
              5.5 * 0.5 + 2.0 * 4.0),
        "electron-gamma class is explicit");
  check(close(value.class_value(LuminositySpeciesClass::Positron,
                                LuminositySpeciesClass::Photon),
              5.5 * 1.5),
        "positron-gamma class is explicit");

  const double gg_positive = 0.25 * (5.0 + 0.5) + 0.75 * 2.0;
  const double gg_class = (0.25 + 7.0) * (5.0 + 0.5) + 0.75 * 2.0;
  check(close(value.lumi_gg, gg_positive),
        "gamma-gamma total excludes zero-energy return paths");
  check(close(value.class_value(LuminositySpeciesClass::Photon,
                                LuminositySpeciesClass::Photon), gg_class),
        "gamma-gamma class includes pre-ecm zero-energy accounting");
  // smin=16 is strict: (2,2) and (4,1) are exactly on the boundary and
  // excluded; only (2,3) is included.
  check(close(value.lumi_gg_high, 0.25 * 0.5),
        "strict gamma-gamma threshold handles equality exactly");
  check(value.ge_pair_opportunities == 3 &&
            value.eg_pair_opportunities == 5 &&
            value.gg_pair_opportunities == 5 &&
            value.cartesian_pairs_visited == 13,
        "logical opportunities and visited Cartesian pairs are exact");
}

void test_factored_equivalence() {
  std::printf("\nProven exact factorization\n");
  Fixture f;
  const PhotonLuminosityResult direct =
      run(f, PhotonLumiAlgorithm::CartesianReference);
  const PhotonLuminosityResult factored =
      run(f, PhotonLumiAlgorithm::FactoredExact);
  bool equal = true;
  for (std::size_t i = 0; i < direct.class_luminosity.size(); ++i)
    equal = equal && close(factored.class_luminosity[i],
                           direct.class_luminosity[i]);
  equal = equal && close(factored.lumi_ge, direct.lumi_ge) &&
          close(factored.lumi_eg, direct.lumi_eg) &&
          close(factored.lumi_gg, direct.lumi_gg) &&
          close(factored.lumi_gg_high, direct.lumi_gg_high);
  check(equal, "factored sufficient statistics match direct enumeration");
  check(factored.ge_pair_opportunities == direct.ge_pair_opportunities &&
            factored.eg_pair_opportunities == direct.eg_pair_opportunities &&
            factored.gg_pair_opportunities == direct.gg_pair_opportunities,
        "factorization preserves logical work-driving opportunities");
  check(factored.cartesian_pairs_visited == 0,
        "factored path performs no redundant Cartesian pair visits");

  const PhotonLuminosityResult zero_threshold =
      run(f, PhotonLumiAlgorithm::FactoredExact, 0.0);
  check(close(zero_threshold.lumi_gg_high, zero_threshold.lumi_gg),
        "zero threshold uses the exact all-positive-photon factorization");

  // Very large finite threshold selects no pair without overflow/NaN.
  const PhotonLuminosityResult unreachable = run(
      f, PhotonLumiAlgorithm::FactoredExact,
      std::numeric_limits<double>::max());
  check(unreachable.lumi_gg_high == 0.0,
        "unreachable finite threshold produces exactly zero high luminosity");
}

void test_threshold_boundary_property() {
  std::printf("\nThreshold-boundary property sweep\n");
  const UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 2, 2);
  BeamSoA e1 = owner({make_particle(
      1101, 1, ParticleSpecies::Electron, 1.0, -0.5, -0.5)});
  BeamSoA e2 = owner({make_particle(
      1201, 2, ParticleSpecies::Positron, 1.0, -0.5, -0.5)});
  std::mt19937_64 random(20260826u);
  std::uniform_real_distribution<double> log_energy(-12.0, 8.0);
  std::uniform_real_distribution<double> weight(0.01, 3.0);
  std::vector<ParticleRecord> first;
  std::vector<ParticleRecord> second;
  for (std::uint64_t i = 0; i < 31; ++i) {
    first.push_back(make_photon(1301 + i, 1, std::exp(log_energy(random)),
                                weight(random), -0.5, -0.5));
    second.push_back(make_photon(1401 + i, 2, std::exp(log_energy(random)),
                                 weight(random), -0.5, -0.5));
  }
  BeamSoA g1 = owner(first);
  BeamSoA g2 = owner(second);
  CellCSR e1c(e1, grid), e2c(e2, grid), g1c(g1, grid), g2c(g2, grid);

  bool all_equal = true;
  for (std::size_t probe = 0; probe < 60; ++probe) {
    const double product =
        4.0 * first[probe % first.size()].energy *
        second[(probe * 7) % second.size()].energy;
    const double ecm = std::sqrt(product);
    const double boundary = ecm * ecm;
    const int direction = static_cast<int>(probe % 3) - 1;
    const double smin = direction < 0
        ? std::nextafter(boundary, 0.0)
        : (direction > 0
               ? std::nextafter(boundary,
                                std::numeric_limits<double>::infinity())
               : boundary);
    PhotonLumiRequest direct_request;
    direct_request.algorithm = PhotonLumiAlgorithm::CartesianReference;
    direct_request.gg_smin = smin;
    PhotonLumiRequest factored_request = direct_request;
    factored_request.algorithm = PhotonLumiAlgorithm::FactoredExact;
    const PhotonLuminosityResult direct = compute_photon_luminosity_only(
        e1, e1c, e2, e2c, g1, g1c, g2, g2c, direct_request);
    const PhotonLuminosityResult factored = compute_photon_luminosity_only(
        e1, e1c, e2, e2c, g1, g1c, g2, g2c, factored_request);
    all_equal = all_equal && close(factored.lumi_gg_high,
                                    direct.lumi_gg_high, 5.0e-14, 1.0e-13);
  }
  check(all_equal,
        "sorted suffix boundary correction matches legacy predicate at 60 ULP probes");
}

void test_empty_and_exchange() {
  std::printf("\nEmpty owners and beam exchange\n");
  const UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 2, 2);
  BeamSoA e1 = owner({});
  BeamSoA e2 = owner({});
  BeamSoA g1 = owner({});
  BeamSoA g2 = owner({});
  CellCSR e1c(e1, grid), e2c(e2, grid), g1c(g1, grid), g2c(g2, grid);
  PhotonLumiRequest request;
  request.algorithm = PhotonLumiAlgorithm::FactoredExact;
  const PhotonLuminosityResult empty = compute_photon_luminosity_only(
      e1, e1c, e2, e2c, g1, g1c, g2, g2c, request);
  bool zero = empty.lumi_ge == 0.0 && empty.lumi_eg == 0.0 &&
              empty.lumi_gg == 0.0 && empty.lumi_gg_high == 0.0;
  for (double value : empty.class_luminosity) zero = zero && value == 0.0;
  check(zero, "four empty owners produce an exact all-zero result");

  Fixture original;
  const PhotonLuminosityResult forward =
      run(original, PhotonLumiAlgorithm::FactoredExact);

  // Recreate a physically exchanged fixture with canonical beam IDs and
  // directions; stable IDs remain globally unique.
  auto exchange_owner = [](const BeamSoA& source, std::uint32_t beam) {
    std::vector<ParticleRecord> values;
    for (std::uint64_t i = 0; i < source.size(); ++i) {
      ParticleRecord value = source.particle(i);
      value.stable_id += 1000;
      value.beam_id = beam;
      value.direction = beam == 1 ? 1 : -1;
      values.push_back(value);
    }
    return owner(values);
  };
  BeamSoA xe1 = exchange_owner(original.e2, 1);
  BeamSoA xe2 = exchange_owner(original.e1, 2);
  BeamSoA xg1 = exchange_owner(original.g2, 1);
  BeamSoA xg2 = exchange_owner(original.g1, 2);
  CellCSR xe1c(xe1, original.grid), xe2c(xe2, original.grid);
  CellCSR xg1c(xg1, original.grid), xg2c(xg2, original.grid);
  request.gg_smin = 16.0;
  const PhotonLuminosityResult reverse = compute_photon_luminosity_only(
      xe1, xe1c, xe2, xe2c, xg1, xg1c, xg2, xg2c, request);
  check(close(forward.lumi_ge, reverse.lumi_eg) &&
            close(forward.lumi_eg, reverse.lumi_ge) &&
            close(forward.lumi_gg, reverse.lumi_gg) &&
            close(forward.lumi_gg_high, reverse.lumi_gg_high),
        "beam exchange swaps ge/eg and preserves gamma-gamma observables");
  bool transpose = true;
  for (std::size_t first = 0; first < 3; ++first)
    for (std::size_t second = 0; second < 3; ++second)
      transpose = transpose && close(
          forward.class_luminosity[3 * first + second],
          reverse.class_luminosity[3 * second + first]);
  check(transpose, "beam exchange transposes the explicit 3x3 class matrix");
}

void test_fail_closed_inputs() {
  std::printf("\nFail-closed capability and ownership checks\n");
  Fixture f;
  PhotonLumiRequest request;
  request.algorithm = PhotonLumiAlgorithm::FactoredExact;
  request.gg_smin = -1.0;
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c, f.g1, f.g1c, f.g2, f.g2c,
              request);
        }),
        "negative threshold is rejected");
  request.gg_smin = std::numeric_limits<double>::quiet_NaN();
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c, f.g1, f.g1c, f.g2, f.g2c,
              request);
        }),
        "non-finite threshold is rejected");

  BeamSoA wrong_direction = f.g1;
  // BeamSoA intentionally exposes no mutable identity/direction arrays, so
  // construct the malformed boundary owner explicitly.
  ParticleRecord bad = make_photon(901, 1, 1.0, 1.0, 0.0, 0.0);
  bad.direction = -1;
  wrong_direction = owner({bad});
  CellCSR wrong_cells(wrong_direction, f.grid);
  request.gg_smin = 0.0;
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c,
              wrong_direction, wrong_cells, f.g2, f.g2c, request);
        }),
        "beam identity/direction mismatch is rejected");

  ParticleRecord nonphoton = make_particle(
      902, 1, ParticleSpecies::Electron, 1.0, 0.0, 0.0);
  BeamSoA wrong_species = owner({nonphoton});
  CellCSR wrong_species_cells(wrong_species, f.grid);
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c,
              wrong_species, wrong_species_cells, f.g2, f.g2c, request);
        }),
        "charged object in photon owner is rejected");

  ParticleRecord duplicate = make_photon(101, 1, 1.0, 1.0, 0.0, 0.0);
  BeamSoA duplicate_across = owner({duplicate});
  CellCSR duplicate_cells(duplicate_across, f.grid);
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c,
              duplicate_across, duplicate_cells, f.g2, f.g2c, request);
        }),
        "cross-owner stable-ID collision is rejected");

  const UniformGrid2D other_grid(-2.0, 2.0, -1.0, 1.0, 2, 2);
  CellCSR mismatched(f.g1, other_grid);
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c,
              f.g1, mismatched, f.g2, f.g2c, request);
        }),
        "mismatched grid geometry is rejected");

  f.g1.x_data()[0] = 0.75;
  check(throws([&] {
          (void)compute_photon_luminosity_only(
              f.e1, f.e1c, f.e2, f.e2c,
              f.g1, f.g1c, f.g2, f.g2c, request);
        }),
        "stale CSR after particle movement is rejected");
}

}  // namespace

int main() {
  test_direct_known_answer();
  test_factored_equivalence();
  test_threshold_boundary_property();
  test_empty_and_exchange();
  test_fail_closed_inputs();
  std::printf("\n%s (%d failure%s)\n",
              failures ? "FAILURES PRESENT" : "all photon-luminosity checks passed",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
