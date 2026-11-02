#ifndef FEEBL_BEAM_SOA_CPP_H
#define FEEBL_BEAM_SOA_CPP_H

#include "modernTypesCPP.h"

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace feebl {
namespace modern {

struct ParticleRecord {
  StableParticleId stable_id = 0;
  Scalar x = 0.0;
  Scalar y = 0.0;
  Scalar z = 0.0;
  Scalar vx = 0.0;
  Scalar vy = 0.0;
  Scalar energy = 0.0;
  Scalar macro_weight = 1.0;
  Scalar upsilon = 0.0;
  Scalar sx = 0.0;
  Scalar sy = 0.0;
  Scalar sz = 0.0;
  ParticleSpecies species = ParticleSpecies::Other;
  std::int8_t charge_e = 0;
  std::int8_t direction = 1;
  std::uint32_t beam_id = 0;
  std::uint32_t flags = 0;
};

struct BeamMetadata {
  UnitConvention units = UnitConvention::LegacyFeeBLNmGeV;
  Scalar bunch_population = 1.0;
};

// Owning, dense particle storage for the modern engine.  Particle arrays are
// authoritative and equal-sized; slice_offsets is CSR-style and groups the
// dense order into exactly slice_count ranges.  Construction is deliberately
// slice-at-a-time so empty slices are represented without placeholder objects.
//
// Spin is an optional component: its three arrays either all have particle
// length or are all absent.  Stable IDs and slice offsets are never exposed as
// mutable containers, because changing either silently would invalidate RNG
// identity and scheduler ownership.
class BeamSoA {
 public:
  explicit BeamSoA(SliceIndex slice_count = 0, bool spin_enabled = false);

  void reset(SliceIndex slice_count, bool spin_enabled = false);
  void reserve(ParticleIndex particle_capacity);
  void append_slice(const std::vector<ParticleRecord>& particles);
  void append_slice(std::initializer_list<ParticleRecord> particles);

  SliceIndex slice_count() const noexcept { return slice_count_; }
  SliceIndex appended_slice_count() const noexcept {
    return static_cast<SliceIndex>(slice_offsets_.size() - 1);
  }
  ParticleIndex size() const noexcept {
    return static_cast<ParticleIndex>(stable_id_.size());
  }
  bool empty() const noexcept { return stable_id_.empty(); }
  bool spin_enabled() const noexcept { return spin_enabled_; }
  bool complete() const noexcept;

  const BeamMetadata& metadata() const noexcept { return metadata_; }
  void set_metadata(const BeamMetadata& metadata) { metadata_ = metadata; }

  const std::vector<ArrayOffset>& slice_offsets() const noexcept {
    return slice_offsets_;
  }
  std::pair<ArrayOffset, ArrayOffset> slice_range(SliceIndex slice) const;
  SliceIndex slice_of(ParticleIndex particle) const;
  ParticleRecord particle(ParticleIndex index) const;

  const std::vector<StableParticleId>& stable_ids() const noexcept {
    return stable_id_;
  }
  const std::vector<Scalar>& x() const noexcept { return x_; }
  const std::vector<Scalar>& y() const noexcept { return y_; }
  const std::vector<Scalar>& z() const noexcept { return z_; }
  const std::vector<Scalar>& vx() const noexcept { return vx_; }
  const std::vector<Scalar>& vy() const noexcept { return vy_; }
  const std::vector<Scalar>& energy() const noexcept { return energy_; }
  const std::vector<Scalar>& macro_weight() const noexcept {
    return macro_weight_;
  }
  const std::vector<Scalar>& upsilon() const noexcept { return upsilon_; }
  const std::vector<Scalar>& sx() const noexcept { return sx_; }
  const std::vector<Scalar>& sy() const noexcept { return sy_; }
  const std::vector<Scalar>& sz() const noexcept { return sz_; }
  const std::vector<ParticleSpecies>& species() const noexcept {
    return species_;
  }
  const std::vector<std::int8_t>& charge_e() const noexcept {
    return charge_e_;
  }
  const std::vector<std::int8_t>& direction() const noexcept {
    return direction_;
  }
  const std::vector<std::uint32_t>& beam_id() const noexcept {
    return beam_id_;
  }
  const std::vector<std::uint32_t>& flags() const noexcept { return flags_; }

  // Non-resizing data access for future vectorized kernels.  Array ownership
  // and lengths remain with BeamSoA, so these cannot break structural shape.
  Scalar* x_data() noexcept { return x_.data(); }
  Scalar* y_data() noexcept { return y_.data(); }
  Scalar* z_data() noexcept { return z_.data(); }
  Scalar* vx_data() noexcept { return vx_.data(); }
  Scalar* vy_data() noexcept { return vy_.data(); }
  Scalar* energy_data() noexcept { return energy_.data(); }
  Scalar* macro_weight_data() noexcept { return macro_weight_.data(); }
  Scalar* upsilon_data() noexcept { return upsilon_.data(); }
  Scalar* sx_data() noexcept { return sx_.data(); }
  Scalar* sy_data() noexcept { return sy_.data(); }
  Scalar* sz_data() noexcept { return sz_.data(); }
  std::uint32_t* flags_data() noexcept { return flags_.data(); }

  ValidationResult validate() const;

 private:
  void reserve_all(std::size_t capacity);
  void append_record(const ParticleRecord& particle);

  SliceIndex slice_count_ = 0;
  bool spin_enabled_ = false;
  BeamMetadata metadata_;
  std::vector<ArrayOffset> slice_offsets_;

  std::vector<StableParticleId> stable_id_;
  std::vector<Scalar> x_;
  std::vector<Scalar> y_;
  std::vector<Scalar> z_;
  std::vector<Scalar> vx_;
  std::vector<Scalar> vy_;
  std::vector<Scalar> energy_;
  std::vector<Scalar> macro_weight_;
  std::vector<Scalar> upsilon_;
  std::vector<Scalar> sx_;
  std::vector<Scalar> sy_;
  std::vector<Scalar> sz_;
  std::vector<ParticleSpecies> species_;
  std::vector<std::int8_t> charge_e_;
  std::vector<std::int8_t> direction_;
  std::vector<std::uint32_t> beam_id_;
  std::vector<std::uint32_t> flags_;
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_BEAM_SOA_CPP_H
