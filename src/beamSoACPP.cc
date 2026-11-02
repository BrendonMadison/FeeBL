#include "beamSoACPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace feebl {
namespace modern {

namespace {

bool species_charge_matches(ParticleSpecies species, std::int8_t charge) {
  switch (species) {
    case ParticleSpecies::Electron:
    case ParticleSpecies::MuonMinus:
      return charge == -1;
    case ParticleSpecies::Positron:
    case ParticleSpecies::MuonPlus:
      return charge == 1;
    case ParticleSpecies::Photon:
      return charge == 0;
    case ParticleSpecies::Other:
      return true;
  }
  return false;
}

}  // namespace

BeamSoA::BeamSoA(SliceIndex slice_count, bool spin_enabled) {
  reset(slice_count, spin_enabled);
}

void BeamSoA::reset(SliceIndex slice_count, bool spin_enabled) {
  std::uint64_t offset_count = 0;
  if (!checked_add(slice_count, 1, offset_count) ||
      !fits_size_t(offset_count) ||
      offset_count > slice_offsets_.max_size()) {
    throw std::length_error("BeamSoA slice-offset count is not representable");
  }

  slice_count_ = slice_count;
  spin_enabled_ = spin_enabled;
  metadata_ = BeamMetadata{};

  slice_offsets_.clear();
  stable_id_.clear();
  x_.clear();
  y_.clear();
  z_.clear();
  vx_.clear();
  vy_.clear();
  energy_.clear();
  macro_weight_.clear();
  upsilon_.clear();
  sx_.clear();
  sy_.clear();
  sz_.clear();
  species_.clear();
  charge_e_.clear();
  direction_.clear();
  beam_id_.clear();
  flags_.clear();

  slice_offsets_.reserve(static_cast<std::size_t>(offset_count));
  slice_offsets_.push_back(0);
}

void BeamSoA::reserve(ParticleIndex particle_capacity) {
  if (!fits_size_t(particle_capacity))
    throw std::length_error("BeamSoA particle capacity is not representable");
  reserve_all(static_cast<std::size_t>(particle_capacity));
}

void BeamSoA::reserve_all(std::size_t capacity) {
  if (capacity > stable_id_.max_size() || capacity > x_.max_size() ||
      capacity > y_.max_size() || capacity > z_.max_size() ||
      capacity > vx_.max_size() || capacity > vy_.max_size() ||
      capacity > energy_.max_size() || capacity > macro_weight_.max_size() ||
      capacity > upsilon_.max_size() || capacity > species_.max_size() ||
      capacity > charge_e_.max_size() || capacity > direction_.max_size() ||
      capacity > beam_id_.max_size() || capacity > flags_.max_size() ||
      (spin_enabled_ &&
       (capacity > sx_.max_size() || capacity > sy_.max_size() ||
        capacity > sz_.max_size()))) {
    throw std::length_error("BeamSoA particle capacity exceeds vector limit");
  }

  // Reserve every component before changing any size.  Appending primitive
  // records is then exception-safe with respect to the equal-length invariant.
  stable_id_.reserve(capacity);
  x_.reserve(capacity);
  y_.reserve(capacity);
  z_.reserve(capacity);
  vx_.reserve(capacity);
  vy_.reserve(capacity);
  energy_.reserve(capacity);
  macro_weight_.reserve(capacity);
  upsilon_.reserve(capacity);
  species_.reserve(capacity);
  charge_e_.reserve(capacity);
  direction_.reserve(capacity);
  beam_id_.reserve(capacity);
  flags_.reserve(capacity);
  if (spin_enabled_) {
    sx_.reserve(capacity);
    sy_.reserve(capacity);
    sz_.reserve(capacity);
  }
}

void BeamSoA::append_record(const ParticleRecord& particle) {
  stable_id_.push_back(particle.stable_id);
  x_.push_back(particle.x);
  y_.push_back(particle.y);
  z_.push_back(particle.z);
  vx_.push_back(particle.vx);
  vy_.push_back(particle.vy);
  energy_.push_back(particle.energy);
  macro_weight_.push_back(particle.macro_weight);
  upsilon_.push_back(particle.upsilon);
  species_.push_back(particle.species);
  charge_e_.push_back(particle.charge_e);
  direction_.push_back(particle.direction);
  beam_id_.push_back(particle.beam_id);
  flags_.push_back(particle.flags);
  if (spin_enabled_) {
    sx_.push_back(particle.sx);
    sy_.push_back(particle.sy);
    sz_.push_back(particle.sz);
  }
}

void BeamSoA::append_slice(const std::vector<ParticleRecord>& particles) {
  if (appended_slice_count() >= slice_count_)
    throw std::logic_error("BeamSoA already contains all declared slices");

  const std::uint64_t add = static_cast<std::uint64_t>(particles.size());
  std::uint64_t new_size = 0;
  if (!checked_add(size(), add, new_size) || !fits_size_t(new_size))
    throw std::length_error("BeamSoA particle count overflow");

  reserve_all(static_cast<std::size_t>(new_size));
  for (const ParticleRecord& particle : particles) append_record(particle);
  slice_offsets_.push_back(new_size);
}

void BeamSoA::append_slice(
    std::initializer_list<ParticleRecord> particles) {
  append_slice(std::vector<ParticleRecord>(particles));
}

bool BeamSoA::complete() const noexcept {
  // reset() proves the addition and conversion are safe.
  return slice_offsets_.size() == static_cast<std::size_t>(slice_count_ + 1);
}

std::pair<ArrayOffset, ArrayOffset> BeamSoA::slice_range(
    SliceIndex slice) const {
  if (!complete())
    throw std::logic_error("BeamSoA slice ranges requested before completion");
  if (slice >= slice_count_)
    throw std::out_of_range("BeamSoA slice index out of range");
  const std::size_t index = static_cast<std::size_t>(slice);
  return std::make_pair(slice_offsets_[index], slice_offsets_[index + 1]);
}

SliceIndex BeamSoA::slice_of(ParticleIndex particle) const {
  if (!complete())
    throw std::logic_error("BeamSoA slice lookup requested before completion");
  if (particle >= size())
    throw std::out_of_range("BeamSoA particle index out of range");
  const auto it = std::upper_bound(slice_offsets_.begin(), slice_offsets_.end(),
                                   static_cast<ArrayOffset>(particle));
  return static_cast<SliceIndex>((it - slice_offsets_.begin()) - 1);
}

ParticleRecord BeamSoA::particle(ParticleIndex index) const {
  if (index >= size())
    throw std::out_of_range("BeamSoA particle index out of range");
  const std::size_t i = static_cast<std::size_t>(index);
  ParticleRecord result;
  result.stable_id = stable_id_[i];
  result.x = x_[i];
  result.y = y_[i];
  result.z = z_[i];
  result.vx = vx_[i];
  result.vy = vy_[i];
  result.energy = energy_[i];
  result.macro_weight = macro_weight_[i];
  result.upsilon = upsilon_[i];
  if (spin_enabled_) {
    result.sx = sx_[i];
    result.sy = sy_[i];
    result.sz = sz_[i];
  }
  result.species = species_[i];
  result.charge_e = charge_e_[i];
  result.direction = direction_[i];
  result.beam_id = beam_id_[i];
  result.flags = flags_[i];
  return result;
}

ValidationResult BeamSoA::validate() const {
  if (!complete())
    return ValidationResult::failure("slice construction is incomplete");

  const std::size_t n = stable_id_.size();
  if (x_.size() != n || y_.size() != n || z_.size() != n ||
      vx_.size() != n || vy_.size() != n || energy_.size() != n ||
      macro_weight_.size() != n || upsilon_.size() != n ||
      species_.size() != n || charge_e_.size() != n ||
      direction_.size() != n || beam_id_.size() != n || flags_.size() != n)
    return ValidationResult::failure("authoritative SoA components differ in length");

  if (spin_enabled_) {
    if (sx_.size() != n || sy_.size() != n || sz_.size() != n)
      return ValidationResult::failure("enabled spin component differs in length");
  } else if (!sx_.empty() || !sy_.empty() || !sz_.empty()) {
    return ValidationResult::failure("disabled spin component is not empty");
  }

  if (slice_offsets_.empty() || slice_offsets_.front() != 0)
    return ValidationResult::failure("slice offsets must begin at zero");
  ArrayOffset previous = 0;
  for (std::size_t s = 0; s < slice_offsets_.size(); ++s) {
    const ArrayOffset offset = slice_offsets_[s];
    if (offset < previous || offset > size())
      return ValidationResult::failure(
          "slice offsets are not monotonic and bounded",
          static_cast<std::uint64_t>(s));
    previous = offset;
  }
  if (slice_offsets_.back() != size())
    return ValidationResult::failure("last slice offset does not equal particle count");

  if (metadata_.units != UnitConvention::LegacyFeeBLNmGeV)
    return ValidationResult::failure("unknown BeamSoA unit convention");
  if (!std::isfinite(metadata_.bunch_population) ||
      metadata_.bunch_population < 0.0)
    return ValidationResult::failure("bunch population must be finite and non-negative");

  std::unordered_set<StableParticleId> ids;
  ids.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    // UINT64_MAX is the version-2 counter-RNG no-parent sentinel
    // (kNoParentStableIdV2).  Letting a physical object carry it would make a
    // unary/binary process key ambiguous even though the SoA shape itself is
    // otherwise valid.  Keep this storage-layer check independent of the RNG
    // header so the fundamental modern containers do not acquire a physics
    // implementation dependency.
    if (stable_id_[i] == std::numeric_limits<StableParticleId>::max())
      return ValidationResult::failure(
          "stable particle ID aliases the reserved no-parent sentinel",
          static_cast<std::uint64_t>(i));
    if (!ids.insert(stable_id_[i]).second)
      return ValidationResult::failure("stable particle ID is duplicated",
                                       static_cast<std::uint64_t>(i));
    if (!std::isfinite(x_[i]) || !std::isfinite(y_[i]) ||
        !std::isfinite(z_[i]) || !std::isfinite(vx_[i]) ||
        !std::isfinite(vy_[i]) || !std::isfinite(energy_[i]) ||
        !std::isfinite(macro_weight_[i]) || !std::isfinite(upsilon_[i]))
      return ValidationResult::failure("particle contains a non-finite scalar",
                                       static_cast<std::uint64_t>(i));
    if (energy_[i] < 0.0)
      return ValidationResult::failure("energy must be non-negative",
                                       static_cast<std::uint64_t>(i));
    if (macro_weight_[i] < 0.0)
      return ValidationResult::failure("macro-particle weight must be non-negative",
                                       static_cast<std::uint64_t>(i));
    if (upsilon_[i] < 0.0)
      return ValidationResult::failure("Upsilon must be non-negative",
                                       static_cast<std::uint64_t>(i));
    if (direction_[i] != -1 && direction_[i] != 1)
      return ValidationResult::failure("longitudinal direction must be -1 or +1",
                                       static_cast<std::uint64_t>(i));
    if (!species_charge_matches(species_[i], charge_e_[i]))
      return ValidationResult::failure("species and electric charge disagree",
                                       static_cast<std::uint64_t>(i));
    if (spin_enabled_) {
      if (!std::isfinite(sx_[i]) || !std::isfinite(sy_[i]) ||
          !std::isfinite(sz_[i]))
        return ValidationResult::failure("spin contains a non-finite scalar",
                                         static_cast<std::uint64_t>(i));
      const Scalar spin_norm2 =
          sx_[i] * sx_[i] + sy_[i] * sy_[i] + sz_[i] * sz_[i];
      if (spin_norm2 > 1.0 + 1.0e-9)
        return ValidationResult::failure("spin/Stokes norm exceeds unity",
                                         static_cast<std::uint64_t>(i));
    }
  }
  return ValidationResult::success();
}

}  // namespace modern
}  // namespace feebl
