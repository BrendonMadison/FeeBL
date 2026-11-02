#include "modernPhotonSoACPP.h"

#include "counterRngCPP.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace feebl {
namespace modern {

namespace {

bool parent_is_explicitly_absent(const PhotonRecord& photon) noexcept {
  return (photon.flags & kPhotonParentAbsent) != 0u;
}

ValidationResult validate_photon_record(const PhotonRecord& photon,
                                        SliceIndex slice_count,
                                        std::uint32_t owner_beam_id,
                                        std::int8_t owner_direction,
                                        bool stokes_enabled,
                                        std::uint64_t index) {
  if (photon.stable_id == kNoParentStableIdV2)
    return ValidationResult::failure(
        "photon stable ID aliases the reserved no-parent sentinel", index);

  const bool absent = parent_is_explicitly_absent(photon);
  const bool sentinel_parent =
      photon.parent_stable_id == kNoParentStableIdV2;
  if (absent != sentinel_parent)
    return ValidationResult::failure(
        "no-parent flag and parent stable-ID sentinel disagree", index);

  if (!absent && photon.parent_stable_id == photon.stable_id)
    return ValidationResult::failure(
        "photon cannot identify itself as its physical parent", index);

  if (photon.beam_id != owner_beam_id)
    return ValidationResult::failure(
        "photon beam identity disagrees with its beam-owned pool", index);
  if (photon.direction != owner_direction)
    return ValidationResult::failure(
        "photon direction disagrees with the FeeBL beam convention", index);
  if (photon.slice >= slice_count)
    return ValidationResult::failure("photon slice is out of range", index);

  if (!std::isfinite(photon.x) || !std::isfinite(photon.y) ||
      !std::isfinite(photon.z) || !std::isfinite(photon.vx) ||
      !std::isfinite(photon.vy) || !std::isfinite(photon.energy) ||
      !std::isfinite(photon.macro_weight) ||
      !std::isfinite(photon.stokes_1) ||
      !std::isfinite(photon.stokes_2) ||
      !std::isfinite(photon.stokes_3))
    return ValidationResult::failure("photon contains a non-finite scalar",
                                     index);
  if (photon.energy < 0.0)
    return ValidationResult::failure("photon energy must be non-negative",
                                     index);
  if (photon.macro_weight < 0.0)
    return ValidationResult::failure(
        "photon macro-particle weight must be non-negative", index);

  // vx and vy are transverse direction cosines in the retained FeeBL
  // convention.  hypot avoids both overflow and underflow in this gate.
  if (std::hypot(photon.vx, photon.vy) >= 1.0)
    return ValidationResult::failure(
        "beam-owned photon requires a strictly positive longitudinal component",
        index);
  if (!stokes_enabled &&
      (photon.stokes_1 != 0.0 || photon.stokes_2 != 0.0 ||
       photon.stokes_3 != 0.0))
    return ValidationResult::failure(
        "photon Stokes state was supplied to a polarization-disabled pool",
        index);
  if (stokes_enabled &&
      std::hypot(photon.stokes_1, photon.stokes_2, photon.stokes_3) >
          1.0 + 1.0e-12)
    return ValidationResult::failure(
        "photon Stokes norm exceeds the physical density-matrix bound",
        index);
  return ValidationResult::success();
}

void throw_record_error(const ValidationResult& result) {
  throw std::invalid_argument(result.message + " at staged photon index " +
                              std::to_string(result.index));
}

void preflight_logical_limits(SliceIndex slice_count,
                              ParticleIndex photon_count,
                              const PhotonStorageLimits& limits,
                              bool stokes_enabled) {
  std::uint64_t offset_count = 0;
  if (!checked_add(slice_count, 1u, offset_count) ||
      !fits_size_t(offset_count))
    throw std::length_error(
        "PhotonSoA slice-offset count is not representable");
  if (!fits_size_t(photon_count))
    throw std::length_error("PhotonSoA photon count is not representable");
  if (photon_count > limits.max_photons)
    throw std::length_error("PhotonSoA exceeds the total photon limit");

  std::uint64_t bytes = 0;
  if (!checked_photon_authoritative_bytes(slice_count, photon_count, bytes,
                                          stokes_enabled))
    throw std::length_error("PhotonSoA authoritative byte count overflows");
  if (bytes > limits.max_authoritative_bytes)
    throw std::length_error("PhotonSoA exceeds the authoritative byte limit");
  if (bytes > limits.max_transaction_bytes)
    throw std::length_error(
        "PhotonSoA persistent state exceeds the transaction peak-byte limit");
}

std::uint64_t bytes_for_records_or_throw(ParticleIndex count,
                                         const char* description) {
  std::uint64_t bytes = 0;
  if (!checked_multiply(count,
                        static_cast<std::uint64_t>(sizeof(PhotonRecord)),
                        bytes))
    throw std::length_error(std::string(description) + " byte count overflows");
  return bytes;
}

void preflight_pool_append(const PhotonSoA& current,
                           ParticleIndex staged_count,
                           std::uint64_t additional_staging_copies = 0) {
  std::uint64_t candidate_count = 0;
  if (!checked_add(current.size(), staged_count, candidate_count))
    throw std::length_error("PhotonPool append count overflows");
  preflight_logical_limits(current.slice_count(), candidate_count,
                           current.limits(), current.stokes_enabled());
  if (candidate_count > std::vector<PhotonRecord>().max_size())
    throw std::length_error("PhotonPool candidate exceeds vector max_size");

  std::uint64_t candidate_authoritative = 0;
  std::uint64_t replacement_empty = 0;
  std::uint64_t identity_scratch = 0;
  if (!checked_photon_authoritative_bytes(
          current.slice_count(), candidate_count, candidate_authoritative,
          current.stokes_enabled()) ||
      !checked_photon_authoritative_bytes(
          current.slice_count(), 0, replacement_empty,
          current.stokes_enabled()) ||
      !checked_multiply(candidate_count,
                        static_cast<std::uint64_t>(sizeof(StableParticleId)),
                        identity_scratch))
    throw std::length_error("PhotonPool transaction byte count overflows");

  // The append keeps the current SoA and submitted staging live while it
  // builds one combined AoS and one replacement SoA plus identity scratch.
  std::uint64_t peak = current.authoritative_bytes();
  const std::uint64_t submitted =
      bytes_for_records_or_throw(staged_count, "submitted photon staging");
  const std::uint64_t combined =
      bytes_for_records_or_throw(candidate_count, "combined photon staging");
  if (!checked_add(peak, submitted, peak) ||
      !checked_add(peak, combined, peak) ||
      !checked_add(peak, identity_scratch, peak) ||
      !checked_add(peak, replacement_empty, peak) ||
      !checked_add(peak, candidate_authoritative, peak))
    throw std::length_error("PhotonPool transaction peak bytes overflow");
  for (std::uint64_t copy = 0; copy < additional_staging_copies; ++copy) {
    if (!checked_add(peak, submitted, peak))
      throw std::length_error("PhotonPool transaction peak bytes overflow");
  }
  if (peak > current.limits().max_transaction_bytes)
    throw std::length_error("PhotonPool exceeds its transaction peak-byte limit");
}

void preflight_pool_replace(const PhotonSoA& current,
                            ParticleIndex candidate_count,
                            std::uint64_t additional_staging_copies = 0) {
  preflight_logical_limits(current.slice_count(), candidate_count,
                           current.limits(), current.stokes_enabled());
  if (candidate_count > std::vector<PhotonRecord>().max_size())
    throw std::length_error("PhotonPool replacement exceeds vector max_size");
  std::uint64_t candidate_authoritative = 0;
  std::uint64_t replacement_empty = 0;
  std::uint64_t identity_scratch = 0;
  if (!checked_photon_authoritative_bytes(
          current.slice_count(), candidate_count, candidate_authoritative,
          current.stokes_enabled()) ||
      !checked_photon_authoritative_bytes(
          current.slice_count(), 0, replacement_empty,
          current.stokes_enabled()) ||
      !checked_multiply(candidate_count,
                        static_cast<std::uint64_t>(sizeof(StableParticleId)),
                        identity_scratch))
    throw std::length_error("PhotonPool replacement byte count overflows");
  std::uint64_t peak = current.authoritative_bytes();
  if (!checked_add(peak,
                   bytes_for_records_or_throw(candidate_count,
                                              "replacement photon staging"),
                   peak) ||
      !checked_add(peak, identity_scratch, peak) ||
      !checked_add(peak, replacement_empty, peak) ||
      !checked_add(peak, candidate_authoritative, peak))
    throw std::length_error("PhotonPool replacement peak bytes overflow");
  const std::uint64_t staging = bytes_for_records_or_throw(
      candidate_count, "replacement photon staging");
  for (std::uint64_t copy = 0; copy < additional_staging_copies; ++copy) {
    if (!checked_add(peak, staging, peak))
      throw std::length_error("PhotonPool replacement peak bytes overflow");
  }
  if (peak > current.limits().max_transaction_bytes)
    throw std::length_error("PhotonPool exceeds its transaction peak-byte limit");
}

template <class BatchVisitor>
void preflight_pool_records(const PhotonSoA& current,
                            ParticleIndex staged_count,
                            bool retain_current,
                            BatchVisitor visit_batches) {
  const ParticleIndex candidate_count =
      retain_current ? current.size() + staged_count : staged_count;
  std::vector<ArrayOffset> slice_counts(
      static_cast<std::size_t>(current.slice_count()), 0u);
  std::vector<StableParticleId> ids;
  ids.reserve(static_cast<std::size_t>(candidate_count));
  if (retain_current) {
    for (SliceIndex slice = 0; slice < current.slice_count(); ++slice) {
      const std::pair<ArrayOffset, ArrayOffset> range =
          current.slice_range(slice);
      slice_counts[static_cast<std::size_t>(slice)] =
          range.second - range.first;
    }
    ids.insert(ids.end(), current.stable_ids().begin(),
               current.stable_ids().end());
  }

  std::uint64_t staged_index = 0;
  visit_batches([&](const PhotonRecord& photon) {
    const ValidationResult result = validate_photon_record(
        photon, current.slice_count(), current.beam_id(), current.direction(),
        current.stokes_enabled(), staged_index++);
    if (!result) throw_record_error(result);
    ArrayOffset& count =
        slice_counts[static_cast<std::size_t>(photon.slice)];
    if (count == current.limits().max_photons_per_slice)
      throw std::length_error("PhotonPool exceeds a per-slice photon limit");
    ++count;
    ids.push_back(photon.stable_id);
  });
  if (staged_index != staged_count || ids.size() != candidate_count)
    throw std::logic_error("PhotonPool staged count changed during preflight");
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::invalid_argument("photon stable child ID is duplicated");
}

}  // namespace

bool checked_photon_authoritative_bytes(SliceIndex slice_count,
                                        ParticleIndex photon_count,
                                        std::uint64_t& bytes,
                                        bool stokes_enabled) noexcept {
  std::uint64_t offset_count = 0;
  std::uint64_t offset_bytes = 0;
  std::uint64_t photon_bytes = 0;
  std::uint64_t bytes_per_photon = kPhotonBaseBytesPerPhoton;
  if (stokes_enabled &&
      !checked_add(bytes_per_photon, kPhotonStokesBytesPerPhoton,
                   bytes_per_photon)) {
    bytes = 0;
    return false;
  }
  if (!checked_add(slice_count, 1u, offset_count) ||
      !checked_multiply(offset_count,
                        static_cast<std::uint64_t>(sizeof(ArrayOffset)),
                        offset_bytes) ||
      !checked_multiply(photon_count, bytes_per_photon,
                        photon_bytes) ||
      !checked_add(offset_bytes, photon_bytes, bytes)) {
    bytes = 0;
    return false;
  }
  return true;
}

PhotonSoA::PhotonSoA(SliceIndex slice_count, std::uint32_t beam_id,
                     const PhotonStorageLimits& limits,
                     bool stokes_enabled)
    : slice_count_(slice_count),
      beam_id_(beam_id),
      direction_(beam_id == 1u ? std::int8_t{1} : std::int8_t{-1}),
      stokes_enabled_(stokes_enabled),
      limits_(limits) {
  if (beam_id_ != 1u && beam_id_ != 2u)
    throw std::invalid_argument(
        "PhotonSoA owner beam ID must be 1 or 2");
  validate_publication_shape(0);
  const std::size_t offset_count =
      static_cast<std::size_t>(slice_count_ + 1u);
  slice_offsets_.assign(offset_count, 0u);
}

void PhotonSoA::validate_publication_shape(
    ParticleIndex photon_count) const {
  preflight_logical_limits(slice_count_, photon_count, limits_,
                           stokes_enabled_);

  const std::uint64_t offset_count = slice_count_ + 1u;
  if (offset_count > slice_offsets_.max_size() ||
      photon_count > stable_id_.max_size() ||
      photon_count > parent_stable_id_.max_size() ||
      photon_count > slice_.max_size() || photon_count > x_.max_size() ||
      photon_count > y_.max_size() || photon_count > z_.max_size() ||
      photon_count > vx_.max_size() || photon_count > vy_.max_size() ||
      photon_count > energy_.max_size() ||
      photon_count > macro_weight_.max_size() ||
      photon_count > stokes_1_.max_size() ||
      photon_count > stokes_2_.max_size() ||
      photon_count > stokes_3_.max_size() ||
      photon_count > flags_.max_size())
    throw std::length_error("PhotonSoA exceeds a component vector limit");
}

void PhotonSoA::validate_transaction_peak(
    ParticleIndex photon_count, std::uint64_t staging_copies) const {
  std::uint64_t candidate_authoritative = 0;
  if (!checked_photon_authoritative_bytes(
          slice_count_, photon_count, candidate_authoritative,
          stokes_enabled_))
    throw std::length_error("PhotonSoA transaction byte count overflows");
  std::uint64_t staging = 0;
  std::uint64_t identity_scratch = 0;
  if (!checked_multiply(photon_count,
                        static_cast<std::uint64_t>(sizeof(PhotonRecord)),
                        staging) ||
      !checked_multiply(photon_count,
                        static_cast<std::uint64_t>(sizeof(StableParticleId)),
                        identity_scratch))
    throw std::length_error("PhotonSoA transaction staging bytes overflow");
  std::uint64_t peak = authoritative_bytes();
  for (std::uint64_t copy = 0; copy < staging_copies; ++copy) {
    if (!checked_add(peak, staging, peak))
      throw std::length_error("PhotonSoA transaction peak bytes overflow");
  }
  if (!checked_add(peak, identity_scratch, peak) ||
      !checked_add(peak, candidate_authoritative, peak))
    throw std::length_error("PhotonSoA transaction peak bytes overflow");
  if (peak > limits_.max_transaction_bytes)
    throw std::length_error("PhotonSoA exceeds its transaction peak-byte limit");
}

void PhotonSoA::validate_candidate_records(
    const std::vector<PhotonRecord>& photons) const {
  std::vector<ArrayOffset> slice_counts(
      static_cast<std::size_t>(slice_count_), 0u);
  std::vector<StableParticleId> ids;
  ids.reserve(photons.size());
  for (std::size_t i = 0; i < photons.size(); ++i) {
    const PhotonRecord& photon = photons[i];
    const ValidationResult result = validate_photon_record(
        photon, slice_count_, beam_id_, direction_, stokes_enabled_,
        static_cast<std::uint64_t>(i));
    if (!result) throw_record_error(result);
    ArrayOffset& count =
        slice_counts[static_cast<std::size_t>(photon.slice)];
    if (count == limits_.max_photons_per_slice)
      throw std::length_error("PhotonSoA exceeds a per-slice photon limit");
    ++count;
    ids.push_back(photon.stable_id);
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::invalid_argument("photon stable child ID is duplicated");
}

void PhotonSoA::assign_canonical(
    const std::vector<PhotonRecord>& photons) {
  const ParticleIndex photon_count =
      static_cast<ParticleIndex>(photons.size());
  validate_publication_shape(photon_count);
  validate_transaction_peak(photon_count, 2);
  validate_candidate_records(photons);
  std::vector<PhotonRecord> owned(photons);
  assign_canonical_owned(std::move(owned));
}

void PhotonSoA::assign_canonical(std::vector<PhotonRecord>&& photons) {
  const ParticleIndex photon_count =
      static_cast<ParticleIndex>(photons.size());
  validate_publication_shape(photon_count);
  validate_transaction_peak(photon_count, 1);
  assign_canonical_owned(std::move(photons));
}

void PhotonSoA::assign_canonical_owned(std::vector<PhotonRecord>&& photons) {
  const ParticleIndex photon_count =
      static_cast<ParticleIndex>(photons.size());

  for (std::size_t i = 0; i < photons.size(); ++i) {
    const ValidationResult result = validate_photon_record(
        photons[i], slice_count_, beam_id_, direction_, stokes_enabled_,
        static_cast<std::uint64_t>(i));
    if (!result) throw_record_error(result);
  }

  // Count first so a pathological single-slice candidate fails before any
  // O(N) identity scratch or component allocation.
  const std::size_t offset_count =
      static_cast<std::size_t>(slice_count_ + 1u);
  std::vector<ArrayOffset> next_offsets(offset_count, 0u);
  for (const PhotonRecord& photon : photons) {
    const std::size_t slot = static_cast<std::size_t>(photon.slice + 1u);
    if (next_offsets[slot] == limits_.max_photons_per_slice)
      throw std::length_error("PhotonSoA exceeds a per-slice photon limit");
    ++next_offsets[slot];
  }

  // Detect duplicates globally, including a reused child ID in two slices.
  // This occurs before constructing any replacement component array.
  std::vector<StableParticleId> ids;
  ids.reserve(photons.size());
  for (const PhotonRecord& photon : photons) ids.push_back(photon.stable_id);
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::invalid_argument("photon stable child ID is duplicated");

  // A physical identity, never the completion order of an emission task,
  // determines the canonical order.
  std::sort(photons.begin(), photons.end(),
            [](const PhotonRecord& left, const PhotonRecord& right) {
              if (left.slice != right.slice) return left.slice < right.slice;
              return left.stable_id < right.stable_id;
            });

  for (std::size_t i = 1; i < next_offsets.size(); ++i) {
    std::uint64_t prefix = 0;
    if (!checked_add(next_offsets[i - 1], next_offsets[i], prefix))
      throw std::length_error("PhotonSoA slice prefix sum overflows");
    next_offsets[i] = prefix;
  }
  if (next_offsets.back() != photon_count)
    throw std::logic_error("PhotonSoA internal prefix sum is inconsistent");

  std::vector<StableParticleId> next_stable_id;
  std::vector<StableParticleId> next_parent_stable_id;
  std::vector<SliceIndex> next_slice;
  std::vector<Scalar> next_x;
  std::vector<Scalar> next_y;
  std::vector<Scalar> next_z;
  std::vector<Scalar> next_vx;
  std::vector<Scalar> next_vy;
  std::vector<Scalar> next_energy;
  std::vector<Scalar> next_macro_weight;
  std::vector<Scalar> next_stokes_1;
  std::vector<Scalar> next_stokes_2;
  std::vector<Scalar> next_stokes_3;
  std::vector<std::uint32_t> next_flags;

  const std::size_t n = photons.size();
  next_stable_id.reserve(n);
  next_parent_stable_id.reserve(n);
  next_slice.reserve(n);
  next_x.reserve(n);
  next_y.reserve(n);
  next_z.reserve(n);
  next_vx.reserve(n);
  next_vy.reserve(n);
  next_energy.reserve(n);
  next_macro_weight.reserve(n);
  if (stokes_enabled_) {
    next_stokes_1.reserve(n);
    next_stokes_2.reserve(n);
    next_stokes_3.reserve(n);
  }
  next_flags.reserve(n);

  for (const PhotonRecord& photon : photons) {
    next_stable_id.push_back(photon.stable_id);
    next_parent_stable_id.push_back(photon.parent_stable_id);
    next_slice.push_back(photon.slice);
    next_x.push_back(photon.x);
    next_y.push_back(photon.y);
    next_z.push_back(photon.z);
    next_vx.push_back(photon.vx);
    next_vy.push_back(photon.vy);
    next_energy.push_back(photon.energy);
    next_macro_weight.push_back(photon.macro_weight);
    if (stokes_enabled_) {
      next_stokes_1.push_back(photon.stokes_1);
      next_stokes_2.push_back(photon.stokes_2);
      next_stokes_3.push_back(photon.stokes_3);
    }
    next_flags.push_back(photon.flags);
  }

  // std::vector::swap is noexcept for the default allocator.  Publication is
  // therefore all-or-nothing after every allocation and check has succeeded.
  slice_offsets_.swap(next_offsets);
  stable_id_.swap(next_stable_id);
  parent_stable_id_.swap(next_parent_stable_id);
  slice_.swap(next_slice);
  x_.swap(next_x);
  y_.swap(next_y);
  z_.swap(next_z);
  vx_.swap(next_vx);
  vy_.swap(next_vy);
  energy_.swap(next_energy);
  macro_weight_.swap(next_macro_weight);
  stokes_1_.swap(next_stokes_1);
  stokes_2_.swap(next_stokes_2);
  stokes_3_.swap(next_stokes_3);
  flags_.swap(next_flags);
}

void PhotonSoA::swap(PhotonSoA& other) noexcept {
  using std::swap;
  swap(slice_count_, other.slice_count_);
  swap(beam_id_, other.beam_id_);
  swap(direction_, other.direction_);
  swap(stokes_enabled_, other.stokes_enabled_);
  swap(limits_, other.limits_);
  slice_offsets_.swap(other.slice_offsets_);
  stable_id_.swap(other.stable_id_);
  parent_stable_id_.swap(other.parent_stable_id_);
  slice_.swap(other.slice_);
  x_.swap(other.x_);
  y_.swap(other.y_);
  z_.swap(other.z_);
  vx_.swap(other.vx_);
  vy_.swap(other.vy_);
  energy_.swap(other.energy_);
  macro_weight_.swap(other.macro_weight_);
  stokes_1_.swap(other.stokes_1_);
  stokes_2_.swap(other.stokes_2_);
  stokes_3_.swap(other.stokes_3_);
  flags_.swap(other.flags_);
}

std::uint64_t PhotonSoA::authoritative_bytes() const noexcept {
  std::uint64_t bytes = 0;
  if (!checked_photon_authoritative_bytes(slice_count_, size(), bytes,
                                          stokes_enabled_))
    return std::numeric_limits<std::uint64_t>::max();
  return bytes;
}

std::pair<ArrayOffset, ArrayOffset> PhotonSoA::slice_range(
    SliceIndex slice) const {
  if (slice >= slice_count_)
    throw std::out_of_range("PhotonSoA slice index out of range");
  const std::size_t index = static_cast<std::size_t>(slice);
  return std::make_pair(slice_offsets_[index], slice_offsets_[index + 1u]);
}

PhotonRecord PhotonSoA::photon(ParticleIndex index) const {
  if (index >= size())
    throw std::out_of_range("PhotonSoA photon index out of range");
  const std::size_t i = static_cast<std::size_t>(index);
  PhotonRecord result;
  result.stable_id = stable_id_[i];
  result.parent_stable_id = parent_stable_id_[i];
  result.beam_id = beam_id_;
  result.direction = direction_;
  result.slice = slice_[i];
  result.x = x_[i];
  result.y = y_[i];
  result.z = z_[i];
  result.vx = vx_[i];
  result.vy = vy_[i];
  result.energy = energy_[i];
  result.macro_weight = macro_weight_[i];
  if (stokes_enabled_) {
    result.stokes_1 = stokes_1_[i];
    result.stokes_2 = stokes_2_[i];
    result.stokes_3 = stokes_3_[i];
  }
  result.flags = flags_[i];
  return result;
}

ValidationResult PhotonSoA::validate() const {
  if (beam_id_ != 1u && beam_id_ != 2u)
    return ValidationResult::failure(
        "photon owner beam identity must be 1 or 2");
  if (direction_ != (beam_id_ == 1u ? std::int8_t{1} : std::int8_t{-1}))
    return ValidationResult::failure(
        "photon owner direction contradicts its beam identity");
  std::uint64_t expected_offset_count = 0;
  if (!checked_add(slice_count_, 1u, expected_offset_count) ||
      !fits_size_t(expected_offset_count) ||
      slice_offsets_.size() !=
          static_cast<std::size_t>(expected_offset_count))
    return ValidationResult::failure(
        "photon slice-offset shape is not representable or complete");

  const std::size_t n = stable_id_.size();
  if (parent_stable_id_.size() != n || slice_.size() != n || x_.size() != n ||
      y_.size() != n || z_.size() != n || vx_.size() != n ||
      vy_.size() != n || energy_.size() != n ||
      macro_weight_.size() != n || flags_.size() != n)
    return ValidationResult::failure(
        "authoritative photon SoA components differ in length");
  const bool stokes_shape = stokes_enabled_
      ? stokes_1_.size() == n && stokes_2_.size() == n &&
            stokes_3_.size() == n
      : stokes_1_.empty() && stokes_2_.empty() && stokes_3_.empty();
  if (!stokes_shape)
    return ValidationResult::failure(
        "photon Stokes arrays violate the all-or-none owner layout");

  if (size() > limits_.max_photons)
    return ValidationResult::failure("published photon count exceeds its limit");
  std::uint64_t bytes = 0;
  if (!checked_photon_authoritative_bytes(slice_count_, size(), bytes,
                                          stokes_enabled_) ||
      bytes > limits_.max_authoritative_bytes ||
      bytes > limits_.max_transaction_bytes)
    return ValidationResult::failure(
        "published photon storage exceeds its byte limit");

  if (slice_offsets_.empty() || slice_offsets_.front() != 0u ||
      slice_offsets_.back() != size())
    return ValidationResult::failure(
        "photon slice offsets must span the complete dense population");

  std::unordered_set<StableParticleId> ids;
  ids.reserve(n);
  for (std::size_t s = 0; s < static_cast<std::size_t>(slice_count_); ++s) {
    const ArrayOffset begin = slice_offsets_[s];
    const ArrayOffset end = slice_offsets_[s + 1u];
    if (end < begin || end > size())
      return ValidationResult::failure(
          "photon slice offsets are not monotonic and bounded",
          static_cast<std::uint64_t>(s));
    if (end - begin > limits_.max_photons_per_slice)
      return ValidationResult::failure(
          "published slice exceeds its photon limit",
          static_cast<std::uint64_t>(s));

    StableParticleId previous = 0;
    bool have_previous = false;
    for (ArrayOffset offset = begin; offset < end; ++offset) {
      const std::size_t i = static_cast<std::size_t>(offset);
      if (slice_[i] != static_cast<SliceIndex>(s))
        return ValidationResult::failure(
            "dense photon slice label disagrees with CSR offsets", offset);
      if (have_previous && stable_id_[i] <= previous)
        return ValidationResult::failure(
            "photon child IDs are not strictly canonical within a slice",
            offset);
      previous = stable_id_[i];
      have_previous = true;
      if (!ids.insert(stable_id_[i]).second)
        return ValidationResult::failure(
            "photon stable child ID is duplicated", offset);

      const ValidationResult record = validate_photon_record(
          photon(offset), slice_count_, beam_id_, direction_, stokes_enabled_,
          offset);
      if (!record) return record;
    }
  }
  return ValidationResult::success();
}

PhotonPool::PhotonPool(SliceIndex slice_count, std::uint32_t beam_id,
                       const PhotonStorageLimits& limits,
                       bool stokes_enabled)
    : photons_(slice_count, beam_id, limits, stokes_enabled) {}

void PhotonPool::append_transaction(
    const std::vector<PhotonRecord>& staged) {
  const ParticleIndex staged_count =
      static_cast<ParticleIndex>(staged.size());
  preflight_pool_append(photons_, staged_count, 1);
  preflight_pool_records(photons_, staged_count, true,
                         [&](const auto& accept) {
                           for (const PhotonRecord& photon : staged)
                             accept(photon);
                         });
  std::vector<PhotonRecord> owned(staged);
  append_transaction_owned(std::move(owned));
}

void PhotonPool::append_transaction(std::vector<PhotonRecord>&& staged) {
  const ParticleIndex staged_count =
      static_cast<ParticleIndex>(staged.size());
  preflight_pool_append(photons_, staged_count);
  preflight_pool_records(photons_, staged_count, true,
                         [&](const auto& accept) {
                           for (const PhotonRecord& photon : staged)
                             accept(photon);
                         });
  append_transaction_owned(std::move(staged));
}

void PhotonPool::append_transaction_owned(
    std::vector<PhotonRecord>&& staged) {
  const std::uint64_t combined_count =
      photons_.size() + static_cast<std::uint64_t>(staged.size());
  std::vector<PhotonRecord> combined;
  combined.reserve(static_cast<std::size_t>(combined_count));
  for (ParticleIndex i = 0; i < photons_.size(); ++i)
    combined.push_back(photons_.photon(i));
  combined.insert(combined.end(),
                  std::make_move_iterator(staged.begin()),
                  std::make_move_iterator(staged.end()));

  PhotonSoA replacement(photons_.slice_count(), photons_.beam_id(),
                        photons_.limits(), photons_.stokes_enabled());
  replacement.assign_canonical(std::move(combined));
  photons_.swap(replacement);
}

void PhotonPool::append_batches_transaction(
    const std::vector<std::vector<PhotonRecord>>& staged_batches) {
  std::uint64_t staged_count = 0;
  for (const std::vector<PhotonRecord>& batch : staged_batches) {
    if (!checked_add(staged_count,
                     static_cast<std::uint64_t>(batch.size()),
                     staged_count))
      throw std::length_error("PhotonPool batch count overflows");
  }
  // Caller-owned worker batches and the flattened staging vector coexist
  // until the one generation-level commit completes.
  preflight_pool_append(photons_, staged_count, 1);
  preflight_pool_records(photons_, staged_count, true,
                         [&](const auto& accept) {
                           for (const std::vector<PhotonRecord>& batch :
                                staged_batches)
                             for (const PhotonRecord& photon : batch)
                               accept(photon);
                         });
  std::vector<PhotonRecord> staged;
  staged.reserve(static_cast<std::size_t>(staged_count));
  for (const std::vector<PhotonRecord>& batch : staged_batches)
    staged.insert(staged.end(), batch.begin(), batch.end());
  append_transaction_owned(std::move(staged));
}

void PhotonPool::replace_transaction(
    const std::vector<PhotonRecord>& staged) {
  preflight_pool_replace(photons_,
                         static_cast<ParticleIndex>(staged.size()), 1);
  preflight_pool_records(
      photons_, static_cast<ParticleIndex>(staged.size()), false,
      [&](const auto& accept) {
        for (const PhotonRecord& photon : staged) accept(photon);
      });
  std::vector<PhotonRecord> owned(staged);
  replace_transaction(std::move(owned));
}

void PhotonPool::replace_transaction(std::vector<PhotonRecord>&& staged) {
  preflight_pool_replace(photons_,
                         static_cast<ParticleIndex>(staged.size()));
  preflight_pool_records(
      photons_, static_cast<ParticleIndex>(staged.size()), false,
      [&](const auto& accept) {
        for (const PhotonRecord& photon : staged) accept(photon);
      });
  PhotonSoA replacement(photons_.slice_count(), photons_.beam_id(),
                        photons_.limits(), photons_.stokes_enabled());
  replacement.assign_canonical(std::move(staged));
  photons_.swap(replacement);
}

void PhotonPool::clear_transaction() { replace_transaction({}); }

}  // namespace modern
}  // namespace feebl
