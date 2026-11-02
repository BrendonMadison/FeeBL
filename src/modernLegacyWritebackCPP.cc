#include "modernLegacyWritebackCPP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace feebl {
namespace modern {

namespace {

constexpr Scalar kNormalizationTolerance =
    8.0 * static_cast<Scalar>(std::numeric_limits<float>::epsilon());
const std::uint64_t kFnvOffset = 1469598103934665603ull;
const std::uint64_t kFnvPrime = 1099511628211ull;

struct StagedParticle {
  PARTICLE* target = nullptr;
  PRIMARY_PARTICLE_ID stable_id = 0;
  ParticleIndex source_index = 0;
  float new_x = 0.0f;
  float new_y = 0.0f;
  float new_vx = 0.0f;
  float new_vy = 0.0f;
  float new_energy = 0.0f;
  float new_upsilon = 0.0f;
  float old_x = 0.0f;
  float old_y = 0.0f;
  float old_z = 0.0f;
  float old_vx = 0.0f;
  float old_vy = 0.0f;
  float old_energy = 0.0f;
  float old_upsilon = 0.0f;
  bool has_spin = false;
  double old_sx = 0.0;
  double old_sy = 0.0;
  double old_sz = 0.0;
};

struct StagedWriteback {
  std::vector<StagedParticle> particles;
  std::vector<const float*> slice_slots;
  LegacyBeamWritebackReceipt receipt;
};

void hash_byte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= static_cast<std::uint64_t>(value);
  hash *= kFnvPrime;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) {
  for (unsigned int byte = 0; byte < 8; ++byte)
    hash_byte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
}

void hash_float(std::uint64_t& hash, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

void hash_double(std::uint64_t& hash, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

bool float_exact(Scalar value, float& result) {
  result = static_cast<float>(value);
  return std::isfinite(value) && std::isfinite(result) &&
         static_cast<Scalar>(result) == value;
}

bool contract_species_charge_matches(const LegacyBeamImportContract& contract) {
  return (contract.species == ParticleSpecies::Electron &&
          contract.charge_e == -1) ||
         (contract.species == ParticleSpecies::Positron &&
          contract.charge_e == 1);
}

ValidationResult validate_contract(const BeamSoA& source,
                                   const LegacyBeamImportContract& contract,
                                   const PARTICLE_BEAM& target) {
  if (contract.units != UnitConvention::LegacyFeeBLNmGeV ||
      source.metadata().units != contract.units)
    return ValidationResult::failure(
        "legacy write-back supports only FeeBL nm/GeV units");
  if (contract.normalization !=
      LegacyMacroNormalization::UniformPhysicalParticlesPerMacro)
    return ValidationResult::failure(
        "legacy write-back requires uniform macro normalization");
  if (!std::isfinite(contract.bunch_population) ||
      contract.bunch_population <= 0.0 ||
      !std::isfinite(contract.macro_weight) ||
      contract.macro_weight <= 0.0 ||
      contract.declared_macro_particle_count == 0)
    return ValidationResult::failure(
        "legacy write-back requires positive population, weight and declared count");
  const Scalar reconstructed = contract.macro_weight *
      static_cast<Scalar>(contract.declared_macro_particle_count);
  const Scalar scale = std::max(
      Scalar{1.0}, std::max(std::fabs(contract.bunch_population),
                            std::fabs(reconstructed)));
  if (!std::isfinite(reconstructed) ||
      std::fabs(contract.bunch_population - reconstructed) >
          kNormalizationTolerance * scale)
    return ValidationResult::failure(
        "legacy write-back normalization product is inconsistent");
  if (source.metadata().bunch_population != contract.bunch_population)
    return ValidationResult::failure(
        "BeamSoA population differs from the legacy import contract");
  if (contract.beam_id != target.beam_label() ||
      (contract.beam_id != 1u && contract.beam_id != 2u))
    return ValidationResult::failure(
        "legacy write-back beam label differs from import contract");
  const std::int8_t expected_direction =
      contract.beam_id == 1u ? std::int8_t{1} : std::int8_t{-1};
  if (contract.direction != expected_direction ||
      !contract_species_charge_matches(contract))
    return ValidationResult::failure(
        "legacy write-back primary direction/species/charge is inconsistent");
  if (source.spin_enabled() != target.has_spin())
    return ValidationResult::failure(
        "BeamSoA and legacy target spin layouts differ");
  if (!target.xy_adopted())
    return ValidationResult::failure(
        "legacy write-back requires the target beam to have adopted XY slots");
  return ValidationResult::success();
}

std::uint64_t invariant_fingerprint(const std::vector<StagedParticle>& staged,
                                    bool read_current) {
  std::uint64_t hash = kFnvOffset;
  hash_u64(hash, staged.size());
  for (const StagedParticle& item : staged) {
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    item.target->XYposition(x, y);
    item.target->velocities(vx, vy);
    (void)x;
    (void)y;
    (void)vx;
    (void)vy;
    hash_u64(hash, item.stable_id);
    hash_float(hash, read_current ? item.target->z() : item.old_z);
    if (read_current && item.has_spin) {
      const TRIDVECTOR& spin = item.target->getSpin();
      hash_double(hash, spin.getComponent(0));
      hash_double(hash, spin.getComponent(1));
      hash_double(hash, spin.getComponent(2));
    } else {
      hash_double(hash, item.old_sx);
      hash_double(hash, item.old_sy);
      hash_double(hash, item.old_sz);
    }
  }
  return hash;
}

bool build_stage(const BeamSoA& source,
                 const LegacyBeamImportContract& contract,
                 const LegacyBeamImportSpan& span,
                 const PARTICLE_BEAM& target,
                 StagedWriteback& staged,
                 ValidationResult& validation) {
  staged = StagedWriteback();
  validation = source.validate();
  if (!validation) return false;
  validation = validate_contract(source, contract, target);
  if (!validation) return false;

  std::uint64_t source_slice_end = 0;
  std::uint64_t modern_slice_end = 0;
  if (!checked_add(span.source_first_slice, span.slice_count,
                   source_slice_end) ||
      source_slice_end >
          static_cast<SliceIndex>(target.number_of_slices()) ||
      !checked_add(span.target_first_slice, span.slice_count,
                   modern_slice_end) ||
      modern_slice_end > source.slice_count()) {
    validation = ValidationResult::failure(
        "legacy write-back receipt has invalid slice bounds");
    return false;
  }
  if (span.target_particle_begin > span.target_particle_end ||
      span.target_particle_end > source.size()) {
    validation = ValidationResult::failure(
        "legacy write-back receipt has invalid BeamSoA particle bounds");
    return false;
  }
  const ParticleIndex span_particles =
      span.target_particle_end - span.target_particle_begin;
  if (!fits_size_t(span_particles) ||
      span_particles > staged.particles.max_size() ||
      !fits_size_t(span.slice_count) ||
      span.slice_count > staged.slice_slots.max_size()) {
    validation = ValidationResult::failure(
        "legacy write-back staging extent is not representable");
    return false;
  }

  try {
    staged.particles.reserve(static_cast<std::size_t>(span_particles));
    staged.slice_slots.reserve(static_cast<std::size_t>(span.slice_count));
    const std::vector<std::vector<PRIMARY_PARTICLE_ID>>& ids =
        target.getParticleIdSlicesStrict();
    ParticleIndex modern_particle = span.target_particle_begin;
    for (SliceIndex offset = 0; offset < span.slice_count; ++offset) {
      const int legacy_slice =
          static_cast<int>(span.source_first_slice + offset);
      const SliceIndex modern_slice = span.target_first_slice + offset;
      const std::vector<PARTICLE*>& legacy_particles =
          target.getParticleVector(legacy_slice);
      const std::pair<ArrayOffset, ArrayOffset> modern_range =
          source.slice_range(modern_slice);
      if (modern_range.first != modern_particle ||
          modern_range.second - modern_range.first !=
              legacy_particles.size() ||
          ids[static_cast<std::size_t>(legacy_slice)].size() !=
              legacy_particles.size() ||
          target.slice_xy_count(static_cast<std::size_t>(legacy_slice)) !=
              legacy_particles.size()) {
        validation = ValidationResult::failure(
            "legacy write-back slice shape/order differs from import receipt",
            offset);
        return false;
      }
      staged.slice_slots.push_back(
          target.slice_xy(static_cast<std::size_t>(legacy_slice)));

      for (std::size_t local = 0; local < legacy_particles.size(); ++local) {
        PARTICLE* particle = legacy_particles[local];
        if (!particle) {
          validation = ValidationResult::failure(
              "legacy write-back target contains a null particle",
              modern_particle);
          return false;
        }
        const std::size_t modern_index =
            static_cast<std::size_t>(modern_particle);
        if (source.stable_ids()[modern_index] !=
            ids[static_cast<std::size_t>(legacy_slice)][local]) {
          validation = ValidationResult::failure(
              "legacy write-back stable ID/order differs from import receipt",
              modern_particle);
          return false;
        }
        if (source.macro_weight()[modern_index] != contract.macro_weight ||
            source.species()[modern_index] != contract.species ||
            source.charge_e()[modern_index] != contract.charge_e ||
            source.direction()[modern_index] != contract.direction ||
            source.beam_id()[modern_index] != contract.beam_id ||
            source.flags()[modern_index] != 0u) {
          validation = ValidationResult::failure(
              "legacy write-back BeamSoA metadata differs from import contract",
              modern_particle);
          return false;
        }

        StagedParticle item;
        item.target = particle;
        item.stable_id = source.stable_ids()[modern_index];
        item.source_index = modern_particle;
        if (!float_exact(source.x()[modern_index], item.new_x) ||
            !float_exact(source.y()[modern_index], item.new_y) ||
            !float_exact(source.vx()[modern_index], item.new_vx) ||
            !float_exact(source.vy()[modern_index], item.new_vy) ||
            !float_exact(source.energy()[modern_index], item.new_energy) ||
            item.new_energy <= 0.0f ||
            !float_exact(source.upsilon()[modern_index], item.new_upsilon) ||
            item.new_upsilon < 0.0f) {
          validation = ValidationResult::failure(
              "legacy write-back phase space/Upsilon is not finite, positive-energy, nonnegative-Upsilon and float-exact",
              modern_particle);
          return false;
        }
        particle->XYposition(item.old_x, item.old_y);
        particle->velocities(item.old_vx, item.old_vy);
        item.old_z = particle->z();
        item.old_energy = particle->energy();
        item.old_upsilon = particle->getUps();
        if (source.z()[modern_index] !=
                static_cast<Scalar>(item.old_z)) {
          validation = ValidationResult::failure(
              "legacy write-back z invariant differs from imported BeamSoA",
              modern_particle);
          return false;
        }
        if (source.spin_enabled()) {
          item.has_spin = true;
          const TRIDVECTOR& spin = particle->getSpin();
          item.old_sx = spin.getComponent(0);
          item.old_sy = spin.getComponent(1);
          item.old_sz = spin.getComponent(2);
          if (source.sx()[modern_index] != item.old_sx ||
              source.sy()[modern_index] != item.old_sy ||
              source.sz()[modern_index] != item.old_sz) {
            validation = ValidationResult::failure(
                "legacy write-back spin invariant differs from imported BeamSoA",
                modern_particle);
            return false;
          }
        }
        staged.particles.push_back(item);
        ++modern_particle;
      }
    }
    if (modern_particle != span.target_particle_end ||
        staged.particles.size() != span_particles) {
      validation = ValidationResult::failure(
          "legacy write-back particle bounds differ from slice contents");
      return false;
    }
  } catch (const std::exception& exception) {
    validation = ValidationResult::failure(
        std::string("legacy write-back staging failed: ") + exception.what());
    return false;
  }

  staged.receipt.span = span;
  staged.receipt.particle_count = span_particles;
  staged.receipt.target_xy_was_adopted = true;
  staged.receipt.invariant_fingerprint_before =
      invariant_fingerprint(staged.particles, false);
  validation = ValidationResult::success();
  return true;
}

void restore_originals(const std::vector<StagedParticle>& staged,
                       std::size_t count) noexcept {
  const std::size_t restore_count = std::min(count, staged.size());
  for (std::size_t index = 0; index < restore_count; ++index) {
    const StagedParticle& item = staged[index];
    item.target->setParticle(item.old_x, item.old_y, item.old_z,
                             item.old_vx, item.old_vy, item.old_energy);
    item.target->setUps(item.old_upsilon);
  }
}

void commit_stage(StagedWriteback& staged, std::size_t& committed) {
  for (; committed < staged.particles.size(); ++committed) {
    const StagedParticle& item = staged.particles[committed];
    item.target->setParticle(item.new_x, item.new_y, item.old_z,
                             item.new_vx, item.new_vy, item.new_energy);
    item.target->setUps(item.new_upsilon);
  }
}

bool finalize_stage(const BeamSoA& source,
                    const LegacyBeamImportSpan& span,
                    const PARTICLE_BEAM& target,
                    StagedWriteback& staged) {
  bool ids_preserved = true;
  bool invariants_preserved = true;
  bool phase_matches = true;
  bool upsilon_matches = true;
  for (const StagedParticle& item : staged.particles) {
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    item.target->XYposition(x, y);
    item.target->velocities(vx, vy);
    phase_matches = phase_matches && x == item.new_x && y == item.new_y &&
                    vx == item.new_vx && vy == item.new_vy &&
                    item.target->energy() == item.new_energy;
    upsilon_matches = upsilon_matches &&
        item.target->getUps() == item.new_upsilon;
    invariants_preserved = invariants_preserved &&
        item.target->z() == item.old_z;
    if (source.spin_enabled()) {
      const TRIDVECTOR& spin = item.target->getSpin();
      invariants_preserved = invariants_preserved &&
          spin.getComponent(0) == item.old_sx &&
          spin.getComponent(1) == item.old_sy &&
          spin.getComponent(2) == item.old_sz;
    }
    ids_preserved = ids_preserved && item.stable_id ==
        source.stable_ids()[static_cast<std::size_t>(item.source_index)];
  }
  bool slots_preserved = target.xy_adopted();
  for (SliceIndex offset = 0; offset < span.slice_count; ++offset) {
    const std::size_t legacy_slice = static_cast<std::size_t>(
        span.source_first_slice + offset);
    slots_preserved = slots_preserved &&
        target.slice_xy(legacy_slice) ==
            staged.slice_slots[static_cast<std::size_t>(offset)];
  }
  staged.receipt.invariant_fingerprint_after =
      invariant_fingerprint(staged.particles, true);
  staged.receipt.slice_shape_and_order_preserved = true;
  staged.receipt.stable_ids_preserved = ids_preserved;
  staged.receipt.z_and_spin_preserved = invariants_preserved;
  staged.receipt.upsilon_matches_beam_soa = upsilon_matches;
  staged.receipt.adopted_xy_slot_addresses_preserved = slots_preserved;
  staged.receipt.phase_space_matches_beam_soa = phase_matches;
  return ids_preserved && invariants_preserved && upsilon_matches &&
         slots_preserved && phase_matches &&
         staged.receipt.invariant_fingerprint_before ==
             staged.receipt.invariant_fingerprint_after;
}

ValidationResult distinct_staged_particle_graphs(
    const StagedWriteback& first,
    const StagedWriteback& second) {
  try {
    std::unordered_set<const PARTICLE*> addresses;
    if (first.particles.size() > addresses.max_size() -
            second.particles.size())
      return ValidationResult::failure(
          "crossing write-back particle-address extent overflows");
    addresses.reserve(first.particles.size() + second.particles.size());
    for (const StagedParticle& item : first.particles)
      addresses.insert(item.target);
    for (const StagedParticle& item : second.particles)
      if (!addresses.insert(item.target).second)
        return ValidationResult::failure(
            "crossing write-back beams share a legacy particle object");
  } catch (const std::exception& exception) {
    return ValidationResult::failure(
        std::string("crossing write-back overlap preflight failed: ") +
        exception.what());
  }
  return ValidationResult::success();
}

}  // namespace

ValidationResult preflight_legacy_particle_beam_writeback(
    const BeamSoA& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    const PARTICLE_BEAM& target) {
  StagedWriteback staged;
  ValidationResult validation;
  (void)build_stage(source, contract, span, target, staged, validation);
  return validation;
}

LegacyBeamWritebackReceipt write_back_legacy_particle_beam_phase_space(
    const BeamSoA& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    PARTICLE_BEAM& target) {
  StagedWriteback staged;
  ValidationResult validation;
  if (!build_stage(source, contract, span, target, staged, validation))
    throw std::invalid_argument(validation.message);

  std::size_t committed = 0;
  try {
    commit_stage(staged, committed);
  } catch (...) {
    restore_originals(staged.particles, committed + 1);
    throw;
  }
  if (!finalize_stage(source, span, target, staged)) {
    restore_originals(staged.particles, staged.particles.size());
    throw std::logic_error(
        "legacy write-back post-commit invariant failed; originals restored");
  }
  return staged.receipt;
}

ValidationResult preflight_legacy_crossing_writeback(
    const BeamSoA& source1,
    const LegacyBeamImportContract& contract1,
    const LegacyBeamImportSpan& span1,
    const PARTICLE_BEAM& target1,
    const BeamSoA& source2,
    const LegacyBeamImportContract& contract2,
    const LegacyBeamImportSpan& span2,
    const PARTICLE_BEAM& target2) {
  StagedWriteback staged1;
  StagedWriteback staged2;
  ValidationResult validation;
  if (!build_stage(source1, contract1, span1, target1, staged1, validation))
    return validation;
  if (!build_stage(source2, contract2, span2, target2, staged2, validation))
    return validation;
  if (contract1.beam_id == contract2.beam_id || &target1 == &target2)
    return ValidationResult::failure(
        "crossing write-back requires two distinct beam labels and targets");
  return distinct_staged_particle_graphs(staged1, staged2);
}

LegacyCrossingWritebackReceipt write_back_legacy_crossing_phase_space(
    const BeamSoA& source1,
    const LegacyBeamImportContract& contract1,
    const LegacyBeamImportSpan& span1,
    PARTICLE_BEAM& target1,
    const BeamSoA& source2,
    const LegacyBeamImportContract& contract2,
    const LegacyBeamImportSpan& span2,
    PARTICLE_BEAM& target2) {
  StagedWriteback staged1;
  StagedWriteback staged2;
  ValidationResult validation;
  if (!build_stage(source1, contract1, span1, target1, staged1, validation) ||
      !build_stage(source2, contract2, span2, target2, staged2, validation))
    throw std::invalid_argument(validation.message);
  if (contract1.beam_id == contract2.beam_id || &target1 == &target2)
    throw std::invalid_argument(
        "crossing write-back requires two distinct beam labels and targets");
  validation = distinct_staged_particle_graphs(staged1, staged2);
  if (!validation) throw std::invalid_argument(validation.message);

  std::size_t committed1 = 0;
  try {
    commit_stage(staged1, committed1);
  } catch (...) {
    restore_originals(staged1.particles, committed1 + 1);
    throw;
  }
  std::size_t committed2 = 0;
  try {
    commit_stage(staged2, committed2);
  } catch (...) {
    restore_originals(staged2.particles, committed2 + 1);
    restore_originals(staged1.particles, staged1.particles.size());
    throw;
  }

  if (!finalize_stage(source1, span1, target1, staged1) ||
      !finalize_stage(source2, span2, target2, staged2)) {
    restore_originals(staged2.particles, staged2.particles.size());
    restore_originals(staged1.particles, staged1.particles.size());
    throw std::logic_error(
        "crossing write-back post-commit invariant failed; both beams restored");
  }

  LegacyCrossingWritebackReceipt receipt;
  receipt.beam1 = staged1.receipt;
  receipt.beam2 = staged2.receipt;
  receipt.distinct_particle_graphs = true;
  receipt.crossing_transaction_committed = true;
  return receipt;
}

}  // namespace modern
}  // namespace feebl
