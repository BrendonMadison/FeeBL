#include "modernLegacyAdapterCPP.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using feebl::modern::BeamSoA;
using feebl::modern::LegacyBeamImportContract;
using feebl::modern::LegacyBeamImportSpan;
using feebl::modern::ParticleIndex;
using feebl::modern::ParticleSpecies;
using feebl::modern::UnitConvention;
using feebl::modern::append_legacy_particle_beam_slices;
using feebl::modern::compare_legacy_particle_beam_import;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-72s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

template <class Exception, class Function>
bool throws_as(Function&& function) {
  try {
    function();
  } catch (const Exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

void initialize_generated(PARTICLE_BEAM& beam, unsigned long count,
                          int charge_symmetric) {
  beam.init_particles(count,
                      100.0f, 10.0f, 1.0f,
                      0, 0,
                      1.0e-6f, 1.0e-6f,
                      10.0f, 125.0f,
                      charge_symmetric, 0);
}

LegacyBeamImportContract contract_for(const PARTICLE_BEAM& beam,
                                      ParticleSpecies species,
                                      std::int8_t charge,
                                      ParticleIndex declared_count,
                                      double macro_weight) {
  LegacyBeamImportContract contract;
  contract.beam_id = beam.beam_label();
  contract.direction = beam.beam_label() == 1u ? 1 : -1;
  contract.species = species;
  contract.charge_e = charge;
  contract.declared_macro_particle_count = declared_count;
  contract.macro_weight = macro_weight;
  contract.bunch_population =
      macro_weight * static_cast<double>(declared_count);
  return contract;
}

struct LegacyValue {
  PRIMARY_PARTICLE_ID id = 0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float vx = 0.0f;
  float vy = 0.0f;
  float energy = 0.0f;
  float upsilon = 0.0f;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
};

std::vector<LegacyValue> snapshot(const PARTICLE_BEAM& beam) {
  std::vector<LegacyValue> result;
  for (int slice = 0; slice < beam.number_of_slices(); ++slice) {
    const std::vector<PARTICLE*>& particles = beam.getParticleVector(slice);
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        beam.getParticleIdVectorStrict(slice);
    for (std::size_t i = 0; i < particles.size(); ++i) {
      LegacyValue value;
      value.id = ids[i];
      particles[i]->XYposition(value.x, value.y);
      particles[i]->velocities(value.vx, value.vy);
      value.z = particles[i]->z();
      value.energy = particles[i]->energy();
      value.upsilon = particles[i]->getUps();
      if (beam.has_spin()) {
        const TRIDVECTOR& spin = particles[i]->getSpin();
        value.sx = spin.getComponent(0);
        value.sy = spin.getComponent(1);
        value.sz = spin.getComponent(2);
      }
      result.push_back(value);
    }
  }
  return result;
}

bool same_snapshot(const std::vector<LegacyValue>& left,
                   const std::vector<LegacyValue>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i].id != right[i].id || left[i].x != right[i].x ||
        left[i].y != right[i].y || left[i].z != right[i].z ||
        left[i].vx != right[i].vx || left[i].vy != right[i].vy ||
        left[i].energy != right[i].energy ||
        left[i].upsilon != right[i].upsilon ||
        left[i].sx != right[i].sx || left[i].sy != right[i].sy ||
        left[i].sz != right[i].sz)
      return false;
  }
  return true;
}

void test_ordinary_generated_read_only_import() {
  std::printf("\nOrdinary generated beam: exact, read-only full import\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(12345u);
  PARTICLE_BEAM source(8, 0, polarization, &random, 1u);
  initialize_generated(source, 96u, 0);
  for (int slice = 0; slice < source.number_of_slices(); ++slice) {
    std::vector<PARTICLE*>& particles = source.getParticleVector(slice);
    for (std::size_t i = 0; i < particles.size(); ++i)
      particles[i]->setUps(0.001f * static_cast<float>(i + 1u));
  }

  const std::vector<LegacyValue> before = snapshot(source);
  const LegacyBeamImportContract contract = contract_for(
      source, ParticleSpecies::Electron, -1, 96u, 2.5e8);
  BeamSoA imported(8, false);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      source, 0, 8, contract, imported);

  check(imported.complete() && imported.size() == source.numberOfParticles(),
        "full ordinary import has exact slice and particle counts");
  check(static_cast<bool>(imported.validate()),
        "ordinary imported BeamSoA passes structural/physics validation");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            source, contract, span, imported)),
        "read-only comparison proves all ordinary values and IDs exact");
  check(same_snapshot(before, snapshot(source)),
        "import and comparison do not mutate legacy phase space, spin, or IDs");
  check(imported.metadata().units == UnitConvention::LegacyFeeBLNmGeV &&
            imported.metadata().bunch_population == contract.bunch_population,
        "unit and physical-population normalization cross the boundary");

  bool ordinary_ids = true;
  for (std::size_t i = 0; i < imported.stable_ids().size(); ++i) {
    PRIMARY_PARTICLE_ID_FIELDS fields;
    ordinary_ids = ordinary_ids &&
        decode_primary_particle_id(imported.stable_ids()[i], fields) &&
        fields.provenance ==
            PRIMARY_PARTICLE_PROVENANCE::GENERATED_ORDINARY &&
        fields.beam_label == 1u && imported.direction()[i] == 1 &&
        imported.beam_id()[i] == 1u && imported.charge_e()[i] == -1 &&
        imported.species()[i] == ParticleSpecies::Electron;
  }
  check(ordinary_ids,
        "ordinary generator provenance and beam-1 +z/electron conventions survive");

  imported.x_data()[0] = imported.x()[0] + 1.0;
  check(!static_cast<bool>(compare_legacy_particle_beam_import(
            source, contract, span, imported)),
        "comparison detects a deliberately changed imported phase-space value");
}

void test_symmetric_generated_spin_import() {
  std::printf("\nSymmetric generated beam: exact spin and beam-2 conventions\n");
  TRIDVECTOR polarization;
  polarization.setComponents(0.2f, -0.1f, 0.8f);
  RNDM random(24680u);
  PARTICLE_BEAM source(8, 1, polarization, &random, 2u);
  initialize_generated(source, 40u, 1);
  const std::vector<LegacyValue> before = snapshot(source);

  const LegacyBeamImportContract contract = contract_for(
      source, ParticleSpecies::Positron, 1, 40u, 5.0e8);
  BeamSoA imported(8, true);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      source, 0, 8, contract, imported);

  check(imported.spin_enabled() && imported.sx().size() == imported.size(),
        "spin-enabled import owns one dense exact spin triple per particle");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            source, contract, span, imported)) &&
            same_snapshot(before, snapshot(source)),
        "symmetric spin import compares exactly and leaves source unchanged");
  bool symmetric_ids = imported.size() == 40u;
  for (std::size_t i = 0; i < imported.size(); ++i) {
    PRIMARY_PARTICLE_ID_FIELDS fields;
    symmetric_ids = symmetric_ids &&
        decode_primary_particle_id(imported.stable_ids()[i], fields) &&
        fields.provenance ==
            PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC &&
        fields.symmetric_lane < 4u && fields.beam_label == 2u &&
        imported.direction()[i] == -1 && imported.beam_id()[i] == 2u &&
        imported.charge_e()[i] == 1 &&
        imported.species()[i] == ParticleSpecies::Positron;
  }
  check(symmetric_ids,
        "symmetric source/lane IDs and beam-2 -z/positron conventions survive");

  BeamSoA wrong_spin_layout(8, false);
  check(throws_as<std::invalid_argument>([&] {
          (void)append_legacy_particle_beam_slices(
              source, 0, 8, contract, wrong_spin_layout);
        }),
        "discarding a present legacy spin component fails closed");
}

void test_empty_slices_ranges_and_existing_target() {
  std::printf("\nEmpty slices, subranges, and append into an existing BeamSoA\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(9u);
  PARTICLE_BEAM source(3, 0, polarization, &random, 1u);
  initialize_generated(source, 2u, 0);
  // Make the empty-slice fixture deterministic while retaining genuine
  // generated objects and source-stable generator IDs. The legacy API exposes
  // particle vectors as mutable; its strict ID vectors are const views of this
  // non-const test object, so the const_cast only keeps the two parallel public
  // views aligned while relocating ownership between slices.
  std::vector<PARTICLE*> generated_particles;
  std::vector<PRIMARY_PARTICLE_ID> generated_ids;
  std::vector<std::vector<PRIMARY_PARTICLE_ID>*> id_slices;
  for (int slice = 0; slice < 3; ++slice) {
    std::vector<PARTICLE*>& particles = source.getParticleVector(slice);
    generated_particles.insert(generated_particles.end(), particles.begin(),
                               particles.end());
    std::vector<PRIMARY_PARTICLE_ID>& ids =
        const_cast<std::vector<PRIMARY_PARTICLE_ID>&>(
            source.getParticleIdVectorStrict(slice));
    generated_ids.insert(generated_ids.end(), ids.begin(), ids.end());
    id_slices.push_back(&ids);
  }
  for (int slice = 0; slice < 3; ++slice) {
    source.getParticleVector(slice).clear();
    id_slices[static_cast<std::size_t>(slice)]->clear();
  }
  source.getParticleVector(0).push_back(generated_particles.at(0));
  id_slices[0]->push_back(generated_ids.at(0));
  source.getParticleVector(2).push_back(generated_particles.at(1));
  id_slices[2]->push_back(generated_ids.at(1));
  source.require_valid_particle_id_sidecar();
  const LegacyBeamImportContract contract = contract_for(
      source, ParticleSpecies::Electron, -1, 2u, 10.0);

  BeamSoA full(3, false);
  const LegacyBeamImportSpan full_span = append_legacy_particle_beam_slices(
      source, 0, 3, contract, full);
  check(full.slice_offsets() ==
            std::vector<feebl::modern::ArrayOffset>({0, 1, 1, 2}),
        "empty legacy slice becomes an exact zero-width CSR slice");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            source, contract, full_span, full)),
        "full uneven beam compares exactly after import");

  BeamSoA ranged(2, false);
  const LegacyBeamImportSpan range_span = append_legacy_particle_beam_slices(
      source, 1, 3, contract, ranged);
  check(ranged.slice_offsets() ==
            std::vector<feebl::modern::ArrayOffset>({0, 0, 1}) &&
            range_span.source_first_slice == 1u &&
            range_span.target_first_slice == 0u,
        "selected source slice range preserves its empty leading slice");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            source, contract, range_span, ranged)),
        "selected slice-range receipt audits the correct source/target mapping");

  BeamSoA incremental(3, false);
  const LegacyBeamImportSpan first = append_legacy_particle_beam_slices(
      source, 0, 1, contract, incremental);
  const LegacyBeamImportSpan rest = append_legacy_particle_beam_slices(
      source, 1, 3, contract, incremental);
  check(first.target_first_slice == 0u && rest.target_first_slice == 1u &&
            incremental.complete() &&
            static_cast<bool>(compare_legacy_particle_beam_import(
                source, contract, first, incremental)) &&
            static_cast<bool>(compare_legacy_particle_beam_import(
                source, contract, rest, incremental)),
        "two appends into one existing BeamSoA retain auditable disjoint spans");
}

void test_invalid_contracts_capacity_and_sidecar() {
  std::printf("\nFail-closed metadata, capacity, physics, and sidecar guards\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(77u);
  PARTICLE_BEAM source(1, 0, polarization, &random, 1u);
  initialize_generated(source, 1u, 0);
  const LegacyBeamImportContract good = contract_for(
      source, ParticleSpecies::Electron, -1, 1u, 100.0);

  auto rejects_contract = [&](LegacyBeamImportContract bad) {
    BeamSoA target(1, false);
    return throws_as<std::invalid_argument>([&] {
      (void)append_legacy_particle_beam_slices(source, 0, 1, bad, target);
    }) && target.size() == 0u && target.appended_slice_count() == 0u;
  };

  LegacyBeamImportContract bad = good;
  bad.macro_weight = -1.0;
  check(rejects_contract(bad), "negative macro weight is rejected before append");
  bad = good;
  bad.macro_weight = std::numeric_limits<double>::infinity();
  check(rejects_contract(bad), "non-finite macro weight is rejected before append");
  bad = good;
  bad.bunch_population += 1.0;
  check(rejects_contract(bad),
        "inconsistent bunch population/denominator/weight is rejected");
  bad = good;
  bad.declared_macro_particle_count = 0u;
  bad.macro_weight = 0.0;
  bad.bunch_population = 0.0;
  check(rejects_contract(bad),
        "non-empty source cannot use a zero-count/zero-weight normalization");
  bad = good;
  bad.declared_macro_particle_count = 0u;
  bad.bunch_population = 0.0;
  check(rejects_contract(bad),
        "non-empty source cannot use a zero declared macro count");
  bad = good;
  bad.direction = -1;
  check(rejects_contract(bad), "beam-1 direction other than +z is rejected");
  bad = good;
  bad.beam_id = 2u;
  bad.direction = -1;
  check(rejects_contract(bad), "contract/source beam-label mismatch is rejected");
  bad = good;
  bad.charge_e = 1;
  check(rejects_contract(bad), "electron with positive electric charge is rejected");
  bad = good;
  bad.species = ParticleSpecies::Photon;
  bad.charge_e = 0;
  check(rejects_contract(bad), "photon metadata cannot label a primary lepton beam");
  bad = good;
  bad.units = static_cast<UnitConvention>(99);
  check(rejects_contract(bad), "unknown boundary unit convention is rejected");

  bad = good;
  bad.maximum_total_particles = 0u;
  BeamSoA capacity_target(1, false);
  check(throws_as<std::length_error>([&] {
          (void)append_legacy_particle_beam_slices(
              source, 0, 1, bad, capacity_target);
        }) && capacity_target.size() == 0u &&
            capacity_target.appended_slice_count() == 0u,
        "caller particle-capacity overflow is rejected atomically");

  BeamSoA no_slice_capacity(0, false);
  check(throws_as<std::length_error>([&] {
          (void)append_legacy_particle_beam_slices(
              source, 0, 1, good, no_slice_capacity);
        }),
        "target slice-capacity overflow is rejected");

  std::vector<PARTICLE*>& exposed = source.getParticleVector(0);
  PARTICLE* extra = new PARTICLE();
  exposed.push_back(extra);
  BeamSoA sidecar_target(1, false);
  check(throws_as<std::logic_error>([&] {
          (void)append_legacy_particle_beam_slices(
              source, 0, 1, good, sidecar_target);
        }) && throws_as<std::logic_error>([&] {
          (void)source.getParticleIdSlicesStrict();
        }) && sidecar_target.size() == 0u,
        "outer strict sidecar view rejects deliberate shape misalignment");
  exposed.pop_back();
  delete extra;
  check(source.validate_particle_id_sidecar(),
        "restoring deliberate sidecar misalignment restores validity");

  RNDM duplicate_random(78u);
  PARTICLE_BEAM duplicate_source(1, 0, polarization, &duplicate_random, 1u);
  initialize_generated(duplicate_source, 2u, 0);
  std::vector<PRIMARY_PARTICLE_ID>& duplicate_ids =
      const_cast<std::vector<PRIMARY_PARTICLE_ID>&>(
          duplicate_source.getParticleIdSlicesStrict().at(0));
  const PRIMARY_PARTICLE_ID original_second = duplicate_ids.at(1);
  duplicate_ids[1] = duplicate_ids[0];
  check(throws_as<std::logic_error>([&] {
          (void)duplicate_source.getParticleIdSlicesStrict();
        }),
        "outer strict sidecar view rejects duplicate source-stable IDs");
  duplicate_ids[1] = original_second;
  check(duplicate_source.validate_particle_id_sidecar(),
        "restoring duplicate-ID corruption restores strict sidecar validity");
}

void test_invalid_legacy_particle_values() {
  std::printf("\nFail-closed legacy particle scalar checks\n");
  TRIDVECTOR polarization;
  polarization = 0.0;

  RNDM random_nan(88u);
  PARTICLE_BEAM nan_source(1, 0, polarization, &random_nan, 1u);
  initialize_generated(nan_source, 1u, 0);
  nan_source.getParticleVector(0).at(0)->setParticle(
      std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f,
      0.0f, 0.0f, 45.0f);
  const LegacyBeamImportContract nan_contract = contract_for(
      nan_source, ParticleSpecies::Electron, -1, 1u, 1.0);
  BeamSoA nan_target(1, false);
  check(throws_as<std::domain_error>([&] {
          (void)append_legacy_particle_beam_slices(
              nan_source, 0, 1, nan_contract, nan_target);
        }) && nan_target.size() == 0u,
        "non-finite legacy phase-space value is rejected atomically");

  RNDM random_speed(99u);
  PARTICLE_BEAM speed_source(1, 0, polarization, &random_speed, 1u);
  initialize_generated(speed_source, 1u, 0);
  speed_source.getParticleVector(0).at(0)->setParticle(
      0.0f, 0.0f, 0.0f, 0.8f, 0.6f, 45.0f);
  const LegacyBeamImportContract speed_contract = contract_for(
      speed_source, ParticleSpecies::Electron, -1, 1u, 1.0);
  BeamSoA speed_target(1, false);
  check(throws_as<std::domain_error>([&] {
          (void)append_legacy_particle_beam_slices(
              speed_source, 0, 1, speed_contract, speed_target);
        }),
        "legacy |v_transverse| equal to one is rejected");
}

}  // namespace

int main() {
  test_ordinary_generated_read_only_import();
  test_symmetric_generated_spin_import();
  test_empty_slices_ranges_and_existing_target();
  test_invalid_contracts_capacity_and_sidecar();
  test_invalid_legacy_particle_values();

  if (failures != 0) {
    std::fprintf(stderr, "%d modern legacy-adapter checks failed\n", failures);
    return 1;
  }
  std::printf("\nmodern legacy boundary adapter: PASS\n");
  return 0;
}
