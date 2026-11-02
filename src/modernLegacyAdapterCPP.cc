#include "modernLegacyAdapterCPP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace feebl {
namespace modern {

namespace {

constexpr Scalar kLegacyNormalizationRelativeTolerance =
    8.0 * static_cast<Scalar>(std::numeric_limits<float>::epsilon());

bool species_charge_matches_primary(ParticleSpecies species,
                                    std::int8_t charge_e) noexcept {
  return (species == ParticleSpecies::Electron && charge_e == -1) ||
         (species == ParticleSpecies::Positron && charge_e == 1);
}

ValidationResult validate_contract(const PARTICLE_BEAM& source,
                                   const LegacyBeamImportContract& contract) {
  if (contract.units != UnitConvention::LegacyFeeBLNmGeV)
    return ValidationResult::failure(
        "legacy adapter supports only FeeBL nm/GeV units");
  if (contract.normalization !=
      LegacyMacroNormalization::UniformPhysicalParticlesPerMacro)
    return ValidationResult::failure(
        "unknown legacy macro-particle normalization");
  if (!std::isfinite(contract.bunch_population) ||
      contract.bunch_population < 0.0)
    return ValidationResult::failure(
        "bunch population must be finite and non-negative");
  if (!std::isfinite(contract.macro_weight) || contract.macro_weight < 0.0)
    return ValidationResult::failure(
        "macro-particle weight must be finite and non-negative");

  const ParticleIndex stored_particle_count =
      static_cast<ParticleIndex>(source.numberOfParticles());
  if (stored_particle_count != 0u) {
    if (contract.declared_macro_particle_count == 0u ||
        contract.macro_weight == 0.0 || contract.bunch_population == 0.0)
      return ValidationResult::failure(
          "a non-empty legacy beam requires a positive declared count, "
          "macro weight, and bunch population");
    if (contract.declared_macro_particle_count < stored_particle_count)
      return ValidationResult::failure(
          "declared macro-particle count is smaller than the stored beam");
  }

  Scalar reconstructed = 0.0;
  if (contract.declared_macro_particle_count != 0) {
    reconstructed = contract.macro_weight *
        static_cast<Scalar>(contract.declared_macro_particle_count);
  }
  if (!std::isfinite(reconstructed))
    return ValidationResult::failure(
        "macro-particle normalization product overflows");
  const Scalar scale = std::max(
      Scalar{1.0}, std::max(std::fabs(contract.bunch_population),
                            std::fabs(reconstructed)));
  if (std::fabs(contract.bunch_population - reconstructed) >
      kLegacyNormalizationRelativeTolerance * scale)
    return ValidationResult::failure(
        "bunch population disagrees with uniform macro normalization");

  if (contract.beam_id != source.beam_label() ||
      (contract.beam_id != 1u && contract.beam_id != 2u))
    return ValidationResult::failure(
        "import beam ID must equal the source-stable beam label (1 or 2)");
  const std::int8_t expected_direction =
      contract.beam_id == 1u ? std::int8_t{1} : std::int8_t{-1};
  if (contract.direction != expected_direction)
    return ValidationResult::failure(
        "FeeBL direction convention is beam 1 -> +z, beam 2 -> -z");
  if (!species_charge_matches_primary(contract.species, contract.charge_e))
    return ValidationResult::failure(
        "legacy primary species and electric charge disagree");
  return ValidationResult::success();
}

ValidationResult validate_record(const ParticleRecord& record,
                                 bool spin_enabled,
                                 std::uint64_t index) {
  if (!std::isfinite(record.x) || !std::isfinite(record.y) ||
      !std::isfinite(record.z) || !std::isfinite(record.vx) ||
      !std::isfinite(record.vy) || !std::isfinite(record.energy) ||
      !std::isfinite(record.upsilon))
    return ValidationResult::failure(
        "legacy particle contains a non-finite phase-space value", index);
  if (record.energy < 0.0)
    return ValidationResult::failure(
        "legacy particle energy is negative", index);
  if (record.upsilon < 0.0)
    return ValidationResult::failure(
        "legacy particle Upsilon is negative", index);
  const Scalar transverse_speed2 =
      record.vx * record.vx + record.vy * record.vy;
  if (!std::isfinite(transverse_speed2) || transverse_speed2 >= 1.0)
    return ValidationResult::failure(
        "legacy particle requires |v_transverse| < 1", index);
  if (spin_enabled) {
    if (!std::isfinite(record.sx) || !std::isfinite(record.sy) ||
        !std::isfinite(record.sz))
      return ValidationResult::failure(
          "legacy particle spin contains a non-finite component", index);
    const Scalar norm2 = record.sx * record.sx + record.sy * record.sy +
                         record.sz * record.sz;
    if (!std::isfinite(norm2) || norm2 > 1.0 + 1.0e-9)
      return ValidationResult::failure(
          "legacy particle spin/polarization norm exceeds unity", index);
  }
  return ValidationResult::success();
}

ParticleRecord legacy_record(const PARTICLE& particle,
                             StableParticleId stable_id,
                             const LegacyBeamImportContract& contract,
                             bool spin_enabled) {
  float x = 0.0f;
  float y = 0.0f;
  float vx = 0.0f;
  float vy = 0.0f;
  particle.XYposition(x, y);
  particle.velocities(vx, vy);

  ParticleRecord record;
  record.stable_id = stable_id;
  record.x = static_cast<Scalar>(x);
  record.y = static_cast<Scalar>(y);
  record.z = static_cast<Scalar>(particle.z());
  record.vx = static_cast<Scalar>(vx);
  record.vy = static_cast<Scalar>(vy);
  record.energy = static_cast<Scalar>(particle.energy());
  record.macro_weight = contract.macro_weight;
  record.upsilon = static_cast<Scalar>(particle.getUps());
  if (spin_enabled) {
    const TRIDVECTOR& spin = particle.getSpin();
    record.sx = spin.getComponent(0);
    record.sy = spin.getComponent(1);
    record.sz = spin.getComponent(2);
  }
  record.species = contract.species;
  record.charge_e = contract.charge_e;
  record.direction = contract.direction;
  record.beam_id = contract.beam_id;
  return record;
}

ValidationResult compare_record(const ParticleRecord& expected,
                                const ParticleRecord& actual,
                                bool spin_enabled,
                                ParticleIndex index) {
  if (actual.stable_id != expected.stable_id)
    return ValidationResult::failure("stable particle ID changed", index);
  if (actual.x != expected.x || actual.y != expected.y ||
      actual.z != expected.z || actual.vx != expected.vx ||
      actual.vy != expected.vy || actual.energy != expected.energy ||
      actual.upsilon != expected.upsilon)
    return ValidationResult::failure(
        "legacy phase-space value changed at the adapter boundary", index);
  if (actual.macro_weight != expected.macro_weight ||
      actual.species != expected.species ||
      actual.charge_e != expected.charge_e ||
      actual.direction != expected.direction ||
      actual.beam_id != expected.beam_id || actual.flags != 0u)
    return ValidationResult::failure(
        "legacy import metadata changed at the adapter boundary", index);
  if (spin_enabled &&
      (actual.sx != expected.sx || actual.sy != expected.sy ||
       actual.sz != expected.sz))
    return ValidationResult::failure(
        "legacy spin changed at the adapter boundary", index);
  return ValidationResult::success();
}

}  // namespace

LegacyBeamImportSpan append_legacy_particle_beam_slices(
    const PARTICLE_BEAM& source,
    SliceIndex source_first_slice,
    SliceIndex source_last_slice,
    const LegacyBeamImportContract& contract,
    BeamSoA& target) {
  const ValidationResult contract_check = validate_contract(source, contract);
  if (!contract_check)
    throw std::invalid_argument(contract_check.message);

  // One strict global sidecar audit, then O(1) const slice views. Repeating the
  // strict per-slice accessor here would turn a boundary conversion into
  // O(n_slice * N log N) work.
  const std::vector<std::vector<PRIMARY_PARTICLE_ID>>& id_slices =
      source.getParticleIdSlicesStrict();

  const SliceIndex source_slice_count =
      static_cast<SliceIndex>(source.number_of_slices());
  if (source_first_slice > source_last_slice ||
      source_last_slice > source_slice_count)
    throw std::out_of_range("legacy source slice range is invalid");
  const SliceIndex import_slice_count =
      source_last_slice - source_first_slice;

  std::uint64_t target_slice_end = 0;
  if (!checked_add(target.appended_slice_count(), import_slice_count,
                   target_slice_end) ||
      target_slice_end > target.slice_count())
    throw std::length_error("legacy slice range exceeds BeamSoA capacity");
  if (target.spin_enabled() != source.has_spin())
    throw std::invalid_argument(
        "legacy and BeamSoA optional-spin layouts must agree");

  BeamMetadata metadata;
  metadata.units = contract.units;
  metadata.bunch_population = contract.bunch_population;
  const bool initialize_metadata =
      target.appended_slice_count() == 0 && target.size() == 0;
  if (!initialize_metadata &&
      (target.metadata().units != metadata.units ||
       target.metadata().bunch_population != metadata.bunch_population)) {
    throw std::invalid_argument(
        "legacy import normalization disagrees with existing BeamSoA");
  }

  // Count first so a caller capacity limit fails before allocating/staging any
  // particle records. The strict outer-sidecar audit already proved each ID
  // slice has the same shape as its particle slice.
  ParticleIndex imported_count = 0;
  for (SliceIndex source_slice = source_first_slice;
       source_slice < source_last_slice; ++source_slice) {
    const std::size_t particle_count = source.getParticleVector(
        static_cast<int>(source_slice)).size();
    if (!checked_add(imported_count,
                     static_cast<ParticleIndex>(particle_count),
                     imported_count))
      throw std::length_error("legacy imported particle count overflows");
  }
  ParticleIndex final_particle_count = 0;
  if (!checked_add(target.size(), imported_count, final_particle_count) ||
      !fits_size_t(final_particle_count) ||
      final_particle_count > contract.maximum_total_particles)
    throw std::length_error("legacy import exceeds BeamSoA particle capacity");

  std::vector<std::vector<ParticleRecord>> staged;
  if (!fits_size_t(import_slice_count) ||
      import_slice_count > staged.max_size())
    throw std::length_error("legacy source slice range is not representable");
  staged.resize(static_cast<std::size_t>(import_slice_count));

  std::uint64_t validation_index = 0;
  for (SliceIndex source_slice = source_first_slice;
       source_slice < source_last_slice; ++source_slice) {
    const int legacy_slice = static_cast<int>(source_slice);
    const std::vector<PARTICLE*>& particles =
        source.getParticleVector(legacy_slice);
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        id_slices[static_cast<std::size_t>(legacy_slice)];
    if (particles.size() != ids.size())
      throw std::logic_error("legacy primary-ID sidecar changed during import");
    if (particles.size() > staged[static_cast<std::size_t>(
                               source_slice - source_first_slice)].max_size())
      throw std::length_error("legacy slice particle count is not representable");

    std::vector<ParticleRecord>& records =
        staged[static_cast<std::size_t>(source_slice - source_first_slice)];
    records.reserve(particles.size());
    for (std::size_t index = 0; index < particles.size(); ++index) {
      if (particles[index] == nullptr)
        throw std::logic_error("legacy primary particle pointer is null");
      const ParticleRecord record = legacy_record(
          *particles[index], ids[index], contract, source.has_spin());
      const ValidationResult record_check =
          validate_record(record, source.has_spin(), validation_index);
      if (!record_check)
        throw std::domain_error(record_check.message);
      records.push_back(record);
      ++validation_index;
    }
  }

  // Reserve before the first logical append so primitive component pushes do
  // not allocate between slices. BeamSoA owns the already-reserved offset
  // vector for its declared slice count.
  target.reserve(final_particle_count);
  // Do this only after the final potentially-throwing allocation. Validation
  // and capacity failures then leave both logical contents and metadata intact.
  if (initialize_metadata) target.set_metadata(metadata);

  LegacyBeamImportSpan span;
  span.source_first_slice = source_first_slice;
  span.target_first_slice = target.appended_slice_count();
  span.slice_count = import_slice_count;
  span.target_particle_begin = target.size();
  for (const std::vector<ParticleRecord>& records : staged)
    target.append_slice(records);
  span.target_particle_end = target.size();
  return span;
}

ValidationResult compare_legacy_particle_beam_import(
    const PARTICLE_BEAM& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    const BeamSoA& imported) {
  const ValidationResult contract_check = validate_contract(source, contract);
  if (!contract_check) return contract_check;

  const std::vector<std::vector<PRIMARY_PARTICLE_ID>>* id_slices = nullptr;
  try {
    id_slices = &source.getParticleIdSlicesStrict();
  } catch (const std::exception& error) {
    return ValidationResult::failure(
        std::string("legacy primary-ID sidecar is invalid: ") + error.what());
  }
  const ValidationResult imported_check = imported.validate();
  if (!imported_check) return imported_check;
  if (imported.spin_enabled() != source.has_spin())
    return ValidationResult::failure(
        "legacy and BeamSoA optional-spin layouts differ");
  if (imported.metadata().units != contract.units ||
      imported.metadata().bunch_population != contract.bunch_population)
    return ValidationResult::failure(
        "BeamSoA normalization metadata differs from import contract");

  std::uint64_t source_end = 0;
  std::uint64_t target_end = 0;
  if (!checked_add(span.source_first_slice, span.slice_count, source_end) ||
      source_end > static_cast<SliceIndex>(source.number_of_slices()) ||
      !checked_add(span.target_first_slice, span.slice_count, target_end) ||
      target_end > imported.slice_count())
    return ValidationResult::failure("legacy import receipt has invalid slices");
  if (span.target_particle_begin > span.target_particle_end ||
      span.target_particle_end > imported.size())
    return ValidationResult::failure(
        "legacy import receipt has invalid particle bounds");

  ParticleIndex target_particle = span.target_particle_begin;
  for (SliceIndex offset = 0; offset < span.slice_count; ++offset) {
    const int source_slice =
        static_cast<int>(span.source_first_slice + offset);
    const SliceIndex target_slice = span.target_first_slice + offset;
    const std::vector<PARTICLE*>& particles =
        source.getParticleVector(source_slice);
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        (*id_slices)[static_cast<std::size_t>(source_slice)];
    const std::pair<ArrayOffset, ArrayOffset> target_range =
        imported.slice_range(target_slice);
    if (target_range.first != target_particle ||
        target_range.second - target_range.first != particles.size())
      return ValidationResult::failure(
          "legacy and imported slice offsets/counts differ", target_slice);

    for (std::size_t index = 0; index < particles.size(); ++index) {
      if (particles[index] == nullptr)
        return ValidationResult::failure(
            "legacy primary particle pointer is null", target_particle);
      const ParticleRecord expected = legacy_record(
          *particles[index], ids[index], contract, source.has_spin());
      const ValidationResult expected_check = validate_record(
          expected, source.has_spin(), target_particle);
      if (!expected_check) return expected_check;
      const ValidationResult record_check = compare_record(
          expected, imported.particle(target_particle), source.has_spin(),
          target_particle);
      if (!record_check) return record_check;
      ++target_particle;
    }
  }
  if (target_particle != span.target_particle_end)
    return ValidationResult::failure(
        "legacy import receipt particle end does not match contents",
        target_particle);
  return ValidationResult::success();
}

}  // namespace modern
}  // namespace feebl
