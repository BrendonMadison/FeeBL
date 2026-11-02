#ifndef FEEBL_PRIMARY_PARTICLE_ID_CPP_H
#define FEEBL_PRIMARY_PARTICLE_ID_CPP_H

#include <cstdint>
#include <stdexcept>

// Source-stable primary-particle identity for migration from the legacy object
// model to the modern SoA/counter-RNG core.  This sidecar is deliberately not
// stored in PARTICLE: it cannot affect legacy object layout, physics, output or
// sequential RNG call order.
//
// Schema 1 is an injective bit allocation, not a hash:
//
//   63       60 59       58 57       56 55                    0
//  +-----------+-----------+-----------+-----------------------+
//  | schema=1  | provenance| beam=1/2  | provenance payload    |
//  +-----------+-----------+-----------+-----------------------+
//
// Ordinary payload = attempted generator ordinal (0 based).  It intentionally
// has gaps when an attempted z lies outside the slicing window.
// Symmetric payload = 4 * attempted generator ordinal + mirror lane [0,3].
// Loaded payload = original physical input-record/source-line ordinal (1
// based), including gaps caused by malformed/rejected or out-of-window lines.
// Slice and vector indices never enter the ID.

using PRIMARY_PARTICLE_ID = std::uint64_t;

enum class PRIMARY_PARTICLE_PROVENANCE : std::uint8_t {
  GENERATED_ORDINARY = 1,
  GENERATED_SYMMETRIC = 2,
  LOADED_SOURCE_RECORD = 3
};

constexpr std::uint8_t PRIMARY_PARTICLE_ID_SCHEMA = 1u;
constexpr unsigned PRIMARY_PARTICLE_ID_PAYLOAD_BITS = 56u;
constexpr PRIMARY_PARTICLE_ID PRIMARY_PARTICLE_ID_PAYLOAD_MASK =
    0x00ffffffffffffffull;
constexpr std::uint64_t PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL =
    PRIMARY_PARTICLE_ID_PAYLOAD_MASK;
constexpr std::uint64_t PRIMARY_PARTICLE_MAX_SYMMETRIC_GENERATOR_ORDINAL =
    PRIMARY_PARTICLE_ID_PAYLOAD_MASK >> 2u;

struct PRIMARY_PARTICLE_ID_FIELDS {
  std::uint8_t schema = 0;
  PRIMARY_PARTICLE_PROVENANCE provenance =
      PRIMARY_PARTICLE_PROVENANCE::GENERATED_ORDINARY;
  std::uint8_t beam_label = 0;
  std::uint64_t source_ordinal = 0;
  std::uint8_t symmetric_lane = 0;
};

inline bool primary_particle_beam_label_is_valid(unsigned int beam_label) {
  return beam_label == 1u || beam_label == 2u;
}

inline PRIMARY_PARTICLE_ID make_primary_particle_id_payload(
    unsigned int beam_label, PRIMARY_PARTICLE_PROVENANCE provenance,
    std::uint64_t payload) {
  if (!primary_particle_beam_label_is_valid(beam_label))
    throw std::invalid_argument("primary particle ID requires beam label 1 or 2");
  if (payload > PRIMARY_PARTICLE_ID_PAYLOAD_MASK)
    throw std::overflow_error("primary particle ID payload overflow");
  const std::uint64_t provenance_number =
      static_cast<std::uint8_t>(provenance);
  if (provenance_number < 1u || provenance_number > 3u)
    throw std::invalid_argument("invalid primary particle provenance");
  return (static_cast<std::uint64_t>(PRIMARY_PARTICLE_ID_SCHEMA) << 60u) |
         (provenance_number << 58u) |
         (static_cast<std::uint64_t>(beam_label) << 56u) | payload;
}

inline PRIMARY_PARTICLE_ID make_generated_primary_particle_id(
    unsigned int beam_label, std::uint64_t attempted_generator_ordinal) {
  if (attempted_generator_ordinal > PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL)
    throw std::overflow_error("ordinary primary generator ordinal overflow");
  return make_primary_particle_id_payload(
      beam_label, PRIMARY_PARTICLE_PROVENANCE::GENERATED_ORDINARY,
      attempted_generator_ordinal);
}

inline PRIMARY_PARTICLE_ID make_symmetric_primary_particle_id(
    unsigned int beam_label, std::uint64_t attempted_generator_ordinal,
    unsigned int lane) {
  if (attempted_generator_ordinal >
      PRIMARY_PARTICLE_MAX_SYMMETRIC_GENERATOR_ORDINAL)
    throw std::overflow_error("symmetric primary generator ordinal overflow");
  if (lane > 3u)
    throw std::invalid_argument("symmetric primary lane must be in [0,3]");
  const std::uint64_t payload =
      (attempted_generator_ordinal << 2u) | static_cast<std::uint64_t>(lane);
  return make_primary_particle_id_payload(
      beam_label, PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC, payload);
}

inline PRIMARY_PARTICLE_ID make_loaded_primary_particle_id(
    unsigned int beam_label, std::uint64_t source_record_ordinal) {
  if (source_record_ordinal == 0u)
    throw std::invalid_argument("loaded primary source-record ordinal is 1 based");
  if (source_record_ordinal > PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL)
    throw std::overflow_error("loaded primary source-record ordinal overflow");
  return make_primary_particle_id_payload(
      beam_label, PRIMARY_PARTICLE_PROVENANCE::LOADED_SOURCE_RECORD,
      source_record_ordinal);
}

inline bool decode_primary_particle_id(PRIMARY_PARTICLE_ID id,
                                       PRIMARY_PARTICLE_ID_FIELDS& fields) {
  fields.schema = static_cast<std::uint8_t>((id >> 60u) & 0x0full);
  const std::uint8_t provenance_number =
      static_cast<std::uint8_t>((id >> 58u) & 0x03ull);
  fields.beam_label = static_cast<std::uint8_t>((id >> 56u) & 0x03ull);
  if (fields.schema != PRIMARY_PARTICLE_ID_SCHEMA ||
      !primary_particle_beam_label_is_valid(fields.beam_label) ||
      provenance_number < 1u || provenance_number > 3u)
    return false;
  fields.provenance =
      static_cast<PRIMARY_PARTICLE_PROVENANCE>(provenance_number);
  const std::uint64_t payload = id & PRIMARY_PARTICLE_ID_PAYLOAD_MASK;
  if (fields.provenance ==
      PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC) {
    fields.source_ordinal = payload >> 2u;
    fields.symmetric_lane = static_cast<std::uint8_t>(payload & 0x03ull);
  } else {
    fields.source_ordinal = payload;
    fields.symmetric_lane = 0u;
    if (fields.provenance ==
            PRIMARY_PARTICLE_PROVENANCE::LOADED_SOURCE_RECORD &&
        fields.source_ordinal == 0u)
      return false;
  }
  return true;
}

#endif  // FEEBL_PRIMARY_PARTICLE_ID_CPP_H
