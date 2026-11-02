#ifndef FEEBL_MODERN_PHOTON_SOA_CPP_H
#define FEEBL_MODERN_PHOTON_SOA_CPP_H

#include "modernTypesCPP.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace feebl {
namespace modern {

// Bit zero has storage semantics: it distinguishes a genuinely parentless
// photon from a corrupt/missing parent identity.  Remaining bits are opaque
// provenance bits owned by later, separately validated emission models.
constexpr std::uint32_t kPhotonParentAbsent = 1u << 0u;

struct PhotonRecord {
  // The emission model owns the versioned mapping from physical parent and
  // emission ordinal to this child ID.  Storage deliberately accepts the
  // completed identity instead of deriving it from a thread, allocator, or
  // append ordinal, so a later CounterRNG mapping remains schedule invariant.
  StableParticleId stable_id = 0;
  StableParticleId parent_stable_id = 0;
  std::uint32_t beam_id = 0;
  std::int8_t direction = 1;
  SliceIndex slice = 0;
  Scalar x = 0.0;
  Scalar y = 0.0;
  Scalar z = 0.0;
  Scalar vx = 0.0;
  Scalar vy = 0.0;
  Scalar energy = 0.0;
  Scalar macro_weight = 0.0;
  Scalar stokes_1 = 0.0;
  Scalar stokes_2 = 0.0;
  Scalar stokes_3 = 0.0;
  std::uint32_t flags = 0;
};

// Limits are logical publication limits, not merely reserve hints.  They are
// checked before any replacement storage is made visible.  The byte limit is
// the exact sum of the authoritative component and CSR-offset array lengths;
// vector allocator metadata and temporary coordinator-side staging are not
// part of that persistent device/host-state budget.
struct PhotonStorageLimits {
  ParticleIndex max_photons = std::numeric_limits<ParticleIndex>::max();
  ParticleIndex max_photons_per_slice =
      std::numeric_limits<ParticleIndex>::max();
  std::uint64_t max_authoritative_bytes =
      std::numeric_limits<std::uint64_t>::max();
  // Peak bytes owned or retained across one strong publication transaction:
  // current authoritative arrays + candidate PhotonRecord staging + identity
  // scratch + replacement authoritative arrays.  Allocator metadata and
  // caller capacity beyond the logical candidate length are excluded.
  std::uint64_t max_transaction_bytes =
      std::numeric_limits<std::uint64_t>::max();
};

constexpr std::uint64_t kPhotonBaseBytesPerPhoton =
    2u * sizeof(StableParticleId) + sizeof(SliceIndex) +
    7u * sizeof(Scalar) + sizeof(std::uint32_t);
constexpr std::uint64_t kPhotonStokesBytesPerPhoton = 3u * sizeof(Scalar);

// Computes (slice_count + 1) * sizeof(ArrayOffset) plus the exact byte count
// of all per-photon authoritative arrays.  False means a uint64_t overflow;
// no saturated or wrapped result is returned.
bool checked_photon_authoritative_bytes(SliceIndex slice_count,
                                        ParticleIndex photon_count,
                                        std::uint64_t& bytes,
                                        bool stokes_enabled = false) noexcept;

// Photon species and zero electric charge are implicit in this type.  One
// owner represents exactly one FeeBL beam: beam 1 travels toward +z and beam 2
// toward -z.  Beam/direction are therefore immutable owner metadata rather
// than redundant per-photon arrays that could contradict the slice CSR.
// Dense authoritative photon storage is otherwise complete for the selected
// all-or-none Stokes layout:
// records are grouped by slice and strictly ordered by stable child ID within
// each slice, and slice_offsets contains exactly slice_count + 1 entries.
//
// assign_canonical is a strong transaction.  It validates and builds a full
// replacement off to the side, then publishes it with noexcept vector swaps.
// Consequently malformed input, capacity failure, allocation failure, or a
// duplicate identity cannot leave a partly extended SoA.
class PhotonSoA {
 public:
  explicit PhotonSoA(
      SliceIndex slice_count,
      std::uint32_t beam_id,
      const PhotonStorageLimits& limits = PhotonStorageLimits{},
      bool stokes_enabled = false);

  // Lvalue publication checks all logical/peak limits before cloning input.
  // The rvalue overload checks the same limits before taking ownership.
  void assign_canonical(const std::vector<PhotonRecord>& photons);
  void assign_canonical(std::vector<PhotonRecord>&& photons);
  void swap(PhotonSoA& other) noexcept;

  SliceIndex slice_count() const noexcept { return slice_count_; }
  std::uint32_t beam_id() const noexcept { return beam_id_; }
  std::int8_t direction() const noexcept { return direction_; }
  bool stokes_enabled() const noexcept { return stokes_enabled_; }
  ParticleIndex size() const noexcept {
    return static_cast<ParticleIndex>(stable_id_.size());
  }
  bool empty() const noexcept { return stable_id_.empty(); }
  const PhotonStorageLimits& limits() const noexcept { return limits_; }
  std::uint64_t authoritative_bytes() const noexcept;

  std::pair<ArrayOffset, ArrayOffset> slice_range(SliceIndex slice) const;
  PhotonRecord photon(ParticleIndex index) const;

  const std::vector<ArrayOffset>& slice_offsets() const noexcept {
    return slice_offsets_;
  }
  const std::vector<StableParticleId>& stable_ids() const noexcept {
    return stable_id_;
  }
  const std::vector<StableParticleId>& parent_stable_ids() const noexcept {
    return parent_stable_id_;
  }
  const std::vector<SliceIndex>& slices() const noexcept { return slice_; }
  const std::vector<Scalar>& x() const noexcept { return x_; }
  const std::vector<Scalar>& y() const noexcept { return y_; }
  const std::vector<Scalar>& z() const noexcept { return z_; }
  const std::vector<Scalar>& vx() const noexcept { return vx_; }
  const std::vector<Scalar>& vy() const noexcept { return vy_; }
  const std::vector<Scalar>& energy() const noexcept { return energy_; }
  const std::vector<Scalar>& macro_weight() const noexcept {
    return macro_weight_;
  }
  const std::vector<Scalar>& stokes_1() const noexcept { return stokes_1_; }
  const std::vector<Scalar>& stokes_2() const noexcept { return stokes_2_; }
  const std::vector<Scalar>& stokes_3() const noexcept { return stokes_3_; }
  const std::vector<std::uint32_t>& flags() const noexcept { return flags_; }

  ValidationResult validate() const;

 private:
  void validate_publication_shape(ParticleIndex photon_count) const;
  void validate_transaction_peak(ParticleIndex photon_count,
                                 std::uint64_t staging_copies) const;
  void validate_candidate_records(
      const std::vector<PhotonRecord>& photons) const;
  void assign_canonical_owned(std::vector<PhotonRecord>&& photons);

  SliceIndex slice_count_ = 0;
  std::uint32_t beam_id_ = 0;
  std::int8_t direction_ = 0;
  bool stokes_enabled_ = false;
  PhotonStorageLimits limits_;
  std::vector<ArrayOffset> slice_offsets_;
  std::vector<StableParticleId> stable_id_;
  std::vector<StableParticleId> parent_stable_id_;
  std::vector<SliceIndex> slice_;
  std::vector<Scalar> x_;
  std::vector<Scalar> y_;
  std::vector<Scalar> z_;
  std::vector<Scalar> vx_;
  std::vector<Scalar> vy_;
  std::vector<Scalar> energy_;
  std::vector<Scalar> macro_weight_;
  std::vector<Scalar> stokes_1_;
  std::vector<Scalar> stokes_2_;
  std::vector<Scalar> stokes_3_;
  std::vector<std::uint32_t> flags_;
};

// Coordinator-side bounded pool.  Worker tasks should stage PhotonRecord
// vectors locally and submit their union in any order.  Stable IDs, rather
// than worker/device completion order, define the published order.  Calls
// mutate the pool only after the full candidate generation has passed every
// invariant and limit; shared concurrent calls themselves must be serialized.
class PhotonPool {
 public:
  explicit PhotonPool(
      SliceIndex slice_count,
      std::uint32_t beam_id,
      const PhotonStorageLimits& limits = PhotonStorageLimits{},
      bool stokes_enabled = false);

  void append_transaction(const std::vector<PhotonRecord>& staged);
  void append_transaction(std::vector<PhotonRecord>&& staged);
  // Production coordinator API: flatten and canonicalize all worker/device
  // batches once per generation, never once per completion callback.
  void append_batches_transaction(
      const std::vector<std::vector<PhotonRecord>>& staged_batches);
  void replace_transaction(const std::vector<PhotonRecord>& staged);
  void replace_transaction(std::vector<PhotonRecord>&& staged);
  void clear_transaction();

  const PhotonSoA& photons() const noexcept { return photons_; }
  SliceIndex slice_count() const noexcept { return photons_.slice_count(); }
  ParticleIndex size() const noexcept { return photons_.size(); }

 private:
  void append_transaction_owned(std::vector<PhotonRecord>&& staged);
  PhotonSoA photons_;
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_PHOTON_SOA_CPP_H
