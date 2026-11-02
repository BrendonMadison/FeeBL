#include "modernLegacyAdapterCPP.h"
#include "modernLegacyWritebackCPP.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using feebl::modern::BeamSoA;
using feebl::modern::LegacyBeamImportContract;
using feebl::modern::LegacyBeamImportSpan;
using feebl::modern::LegacyBeamWritebackReceipt;
using feebl::modern::LegacyCrossingWritebackReceipt;
using feebl::modern::ParticleSpecies;
using feebl::modern::append_legacy_particle_beam_slices;
using feebl::modern::compare_legacy_particle_beam_import;
using feebl::modern::preflight_legacy_particle_beam_writeback;
using feebl::modern::preflight_legacy_crossing_writeback;
using feebl::modern::write_back_legacy_particle_beam_phase_space;
using feebl::modern::write_back_legacy_crossing_phase_space;

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-78s %s\n", description,
              condition ? "PASS" : "FAIL");
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

void initialize_generated(PARTICLE_BEAM& beam,
                          unsigned long count,
                          int symmetric) {
  beam.init_particles(count,
                      100.0f, 10.0f, 1.0f,
                      0, 0,
                      1.0e-6f, 1.0e-6f,
                      10.0f, 45.6f,
                      symmetric, 0);
}

LegacyBeamImportContract make_contract(const PARTICLE_BEAM& beam,
                                       ParticleSpecies species,
                                       std::int8_t charge) {
  LegacyBeamImportContract contract;
  contract.beam_id = beam.beam_label();
  contract.direction = beam.beam_label() == 1u ? 1 : -1;
  contract.species = species;
  contract.charge_e = charge;
  contract.declared_macro_particle_count = 25000;
  contract.macro_weight = 9720000.0;
  contract.bunch_population = 2.43e11;
  return contract;
}

struct LegacyValue {
  PRIMARY_PARTICLE_ID id = 0;
  const PARTICLE* address = nullptr;
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

struct LegacySnapshot {
  std::vector<LegacyValue> particles;
  std::vector<const float*> slice_slots;
  std::vector<std::size_t> slice_counts;
  bool xy_adopted = false;
};

LegacySnapshot snapshot(const PARTICLE_BEAM& beam) {
  LegacySnapshot result;
  result.xy_adopted = beam.xy_adopted();
  for (int slice = 0; slice < beam.number_of_slices(); ++slice) {
    result.slice_slots.push_back(
        beam.xy_adopted() ? beam.slice_xy(static_cast<std::size_t>(slice))
                          : nullptr);
    result.slice_counts.push_back(
        beam.numberOfParticlesOfSlice(slice));
    const std::vector<PARTICLE*>& particles = beam.getParticleVector(slice);
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        beam.getParticleIdVectorStrict(slice);
    for (std::size_t index = 0; index < particles.size(); ++index) {
      LegacyValue value;
      value.id = ids[index];
      value.address = particles[index];
      particles[index]->XYposition(value.x, value.y);
      particles[index]->velocities(value.vx, value.vy);
      value.z = particles[index]->z();
      value.energy = particles[index]->energy();
      value.upsilon = particles[index]->getUps();
      if (beam.has_spin()) {
        const TRIDVECTOR& spin = particles[index]->getSpin();
        value.sx = spin.getComponent(0);
        value.sy = spin.getComponent(1);
        value.sz = spin.getComponent(2);
      }
      result.particles.push_back(value);
    }
  }
  return result;
}

bool exactly_same(const LegacySnapshot& first,
                  const LegacySnapshot& second) {
  if (first.xy_adopted != second.xy_adopted ||
      first.slice_slots != second.slice_slots ||
      first.slice_counts != second.slice_counts ||
      first.particles.size() != second.particles.size())
    return false;
  for (std::size_t index = 0; index < first.particles.size(); ++index) {
    const LegacyValue& a = first.particles[index];
    const LegacyValue& b = second.particles[index];
    if (a.id != b.id || a.address != b.address || a.x != b.x ||
        a.y != b.y || a.z != b.z || a.vx != b.vx || a.vy != b.vy ||
        a.energy != b.energy || a.upsilon != b.upsilon ||
        a.sx != b.sx || a.sy != b.sy || a.sz != b.sz)
      return false;
  }
  return true;
}

bool invariant_same(const LegacySnapshot& first,
                    const LegacySnapshot& second) {
  if (first.xy_adopted != second.xy_adopted ||
      first.slice_slots != second.slice_slots ||
      first.slice_counts != second.slice_counts ||
      first.particles.size() != second.particles.size())
    return false;
  for (std::size_t index = 0; index < first.particles.size(); ++index) {
    const LegacyValue& a = first.particles[index];
    const LegacyValue& b = second.particles[index];
    if (a.id != b.id || a.address != b.address || a.z != b.z ||
        a.sx != b.sx || a.sy != b.sy ||
        a.sz != b.sz)
      return false;
  }
  return true;
}

void change_supported_phase_space(BeamSoA& beam) {
  for (std::size_t index = 0; index < beam.x().size(); ++index) {
    float x = static_cast<float>(beam.x()[index]);
    float y = static_cast<float>(beam.y()[index]);
    float vx = static_cast<float>(beam.vx()[index]);
    float vy = static_cast<float>(beam.vy()[index]);
    float energy = static_cast<float>(beam.energy()[index]);
    float upsilon = static_cast<float>(beam.upsilon()[index]);
    x += (index & 1u) ? 0.25f : -0.125f;
    y += (index & 1u) ? -0.0625f : 0.125f;
    vx += 1.0e-8f * static_cast<float>(index + 1u);
    vy -= 2.0e-8f * static_cast<float>(index + 1u);
    energy -= 0.001f * static_cast<float>((index % 3u) + 1u);
    upsilon += 1.0e-6f * static_cast<float>(index + 1u);
    beam.x_data()[index] = static_cast<double>(x);
    beam.y_data()[index] = static_cast<double>(y);
    beam.vx_data()[index] = static_cast<double>(vx);
    beam.vy_data()[index] = static_cast<double>(vy);
    beam.energy_data()[index] = static_cast<double>(energy);
    beam.upsilon_data()[index] = static_cast<double>(upsilon);
  }
}

void ordinary_exact_round_trip() {
  std::printf("\nOrdinary generated beam: transactional adopted-slot write-back\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(9173u);
  PARTICLE_BEAM legacy(7, 0, polarization, &random, 1u);
  initialize_generated(legacy, 128u, 0);
  for (int slice = 0; slice < legacy.number_of_slices(); ++slice) {
    std::vector<PARTICLE*>& particles = legacy.getParticleVector(slice);
    for (std::size_t index = 0; index < particles.size(); ++index)
      particles[index]->setUps(
          1.0e-5f * static_cast<float>(index + slice + 1));
  }
  legacy.adopt_xy();
  const LegacySnapshot before = snapshot(legacy);

  const LegacyBeamImportContract contract = make_contract(
      legacy, ParticleSpecies::Electron, -1);
  BeamSoA modern(7, false);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      legacy, 0, 7, contract, modern);
  change_supported_phase_space(modern);

  const feebl::modern::ValidationResult preflight =
      preflight_legacy_particle_beam_writeback(
          modern, contract, span, legacy);
  check(static_cast<bool>(preflight),
        "preflight proves full CSR/ID/invariant/write-back compatibility");
  check(exactly_same(before, snapshot(legacy)),
        "read-only preflight leaves legacy object graph and adopted slots exact");

  const LegacyBeamWritebackReceipt receipt =
      write_back_legacy_particle_beam_phase_space(
          modern, contract, span, legacy);
  const LegacySnapshot after = snapshot(legacy);
  check(receipt.particle_count == modern.size() &&
            receipt.target_xy_was_adopted &&
            receipt.slice_shape_and_order_preserved &&
            receipt.stable_ids_preserved &&
            receipt.z_and_spin_preserved &&
            receipt.upsilon_matches_beam_soa &&
            receipt.adopted_xy_slot_addresses_preserved &&
            receipt.phase_space_matches_beam_soa &&
            receipt.invariant_fingerprint_before ==
                receipt.invariant_fingerprint_after,
        "receipt proves exact in-place commit and invariant preservation");
  check(invariant_same(before, after),
        "particle pointers, adopted XY bases, z, spin, IDs and slices survive");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            legacy, contract, span, modern)),
        "legacy phase state now exactly equals the mode-3 BeamSoA result");
}

void invalid_output_is_transactional() {
  std::printf("\nInvalid modern output: fail before mutation\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(1823u);
  PARTICLE_BEAM legacy(5, 0, polarization, &random, 1u);
  initialize_generated(legacy, 80u, 0);
  legacy.adopt_xy();
  const LegacyBeamImportContract contract = make_contract(
      legacy, ParticleSpecies::Electron, -1);
  BeamSoA modern(5, false);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      legacy, 0, 5, contract, modern);
  modern.x_data()[0] = 0.1;  // Not exactly representable by legacy float.
  const LegacySnapshot before = snapshot(legacy);
  const feebl::modern::ValidationResult preflight =
      preflight_legacy_particle_beam_writeback(
          modern, contract, span, legacy);
  check(!static_cast<bool>(preflight) &&
            preflight.message.find("float-exact") != std::string::npos,
        "preflight rejects a non-float-exact modern result");
  check(throws_as<std::invalid_argument>([&]() {
          (void)write_back_legacy_particle_beam_phase_space(
              modern, contract, span, legacy);
        }),
        "commit API repeats the gate and throws before changing a particle");
  check(exactly_same(before, snapshot(legacy)),
        "failed write-back preserves every legacy value, pointer, slot and ID");
}

void spin_invariants_survive_generic_bridge() {
  std::printf("\nSpin beam: generic bridge changes phase space only\n");
  TRIDVECTOR polarization;
  polarization.setComponents(0.2f, -0.1f, 0.8f);
  RNDM random(771u);
  PARTICLE_BEAM legacy(6, 1, polarization, &random, 2u);
  initialize_generated(legacy, 96u, 1);
  legacy.adopt_xy();
  const LegacySnapshot before = snapshot(legacy);
  const LegacyBeamImportContract contract = make_contract(
      legacy, ParticleSpecies::Positron, 1);
  BeamSoA modern(6, true);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      legacy, 0, 6, contract, modern);
  change_supported_phase_space(modern);
  const LegacyBeamWritebackReceipt receipt =
      write_back_legacy_particle_beam_phase_space(
          modern, contract, span, legacy);
  check(receipt.z_and_spin_preserved &&
            receipt.upsilon_matches_beam_soa &&
            invariant_same(before, snapshot(legacy)),
        "spin components and all non-phase invariants remain bit-for-value exact");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            legacy, contract, span, modern)),
        "spin BeamSoA and legacy target compare exact after phase write-back");
}

void paired_crossing_transaction() {
  std::printf("\nTwo-beam crossing: all-or-nothing phase/Upsilon write-back\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random1(291u);
  RNDM random2(292u);
  PARTICLE_BEAM legacy1(5, 0, polarization, &random1, 1u);
  PARTICLE_BEAM legacy2(5, 0, polarization, &random2, 2u);
  initialize_generated(legacy1, 80u, 0);
  initialize_generated(legacy2, 80u, 0);
  legacy1.adopt_xy();
  legacy2.adopt_xy();
  const LegacyBeamImportContract contract1 = make_contract(
      legacy1, ParticleSpecies::Electron, -1);
  const LegacyBeamImportContract contract2 = make_contract(
      legacy2, ParticleSpecies::Positron, 1);
  BeamSoA modern1(5, false);
  BeamSoA modern2(5, false);
  const LegacyBeamImportSpan span1 = append_legacy_particle_beam_slices(
      legacy1, 0, 5, contract1, modern1);
  const LegacyBeamImportSpan span2 = append_legacy_particle_beam_slices(
      legacy2, 0, 5, contract2, modern2);
  change_supported_phase_space(modern1);
  change_supported_phase_space(modern2);
  const LegacySnapshot before1 = snapshot(legacy1);
  const LegacySnapshot before2 = snapshot(legacy2);

  const feebl::modern::ValidationResult preflight =
      preflight_legacy_crossing_writeback(
          modern1, contract1, span1, legacy1,
          modern2, contract2, span2, legacy2);
  check(static_cast<bool>(preflight) &&
            exactly_same(before1, snapshot(legacy1)) &&
            exactly_same(before2, snapshot(legacy2)),
        "paired preflight stages both beams without mutating either target");
  const LegacyCrossingWritebackReceipt receipt =
      write_back_legacy_crossing_phase_space(
          modern1, contract1, span1, legacy1,
          modern2, contract2, span2, legacy2);
  check(receipt.distinct_particle_graphs &&
            receipt.crossing_transaction_committed &&
            receipt.beam1.upsilon_matches_beam_soa &&
            receipt.beam2.upsilon_matches_beam_soa &&
            receipt.beam1.z_and_spin_preserved &&
            receipt.beam2.z_and_spin_preserved,
        "paired receipt proves both Upsilon states and host invariants");
  check(static_cast<bool>(compare_legacy_particle_beam_import(
            legacy1, contract1, span1, modern1)) &&
            static_cast<bool>(compare_legacy_particle_beam_import(
                legacy2, contract2, span2, modern2)),
        "both legacy beams exactly equal their nonzero-kick modern states");

  change_supported_phase_space(modern1);
  change_supported_phase_space(modern2);
  modern2.x_data()[0] = 0.1;
  const LegacySnapshot committed1 = snapshot(legacy1);
  const LegacySnapshot committed2 = snapshot(legacy2);
  const feebl::modern::ValidationResult rejected =
      preflight_legacy_crossing_writeback(
          modern1, contract1, span1, legacy1,
          modern2, contract2, span2, legacy2);
  check(!static_cast<bool>(rejected) &&
            throws_as<std::invalid_argument>([&]() {
              (void)write_back_legacy_crossing_phase_space(
                  modern1, contract1, span1, legacy1,
                  modern2, contract2, span2, legacy2);
            }),
        "invalid beam 2 rejects the whole crossing before beam 1 commit");
  check(exactly_same(committed1, snapshot(legacy1)) &&
            exactly_same(committed2, snapshot(legacy2)),
        "failed paired transaction preserves phase space and Upsilon in both beams");
}

void nonadopted_target_is_rejected() {
  std::printf("\nNon-adopted target: unsupported production boundary\n");
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(44u);
  PARTICLE_BEAM legacy(4, 0, polarization, &random, 1u);
  initialize_generated(legacy, 64u, 0);
  const LegacyBeamImportContract contract = make_contract(
      legacy, ParticleSpecies::Electron, -1);
  BeamSoA modern(4, false);
  const LegacyBeamImportSpan span = append_legacy_particle_beam_slices(
      legacy, 0, 4, contract, modern);
  const LegacySnapshot before = snapshot(legacy);
  const feebl::modern::ValidationResult preflight =
      preflight_legacy_particle_beam_writeback(
          modern, contract, span, legacy);
  check(!static_cast<bool>(preflight) &&
            preflight.message.find("adopted XY") != std::string::npos,
        "production bridge refuses a target without stable adopted XY slots");
  check(exactly_same(before, snapshot(legacy)),
        "non-adopted rejection is read-only");
}

}  // namespace

int main() {
  ordinary_exact_round_trip();
  invalid_output_is_transactional();
  spin_invariants_survive_generic_bridge();
  paired_crossing_transaction();
  nonadopted_target_is_rejected();
  std::printf("\nmodern legacy write-back: %s (%d failure%s)\n",
              failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
