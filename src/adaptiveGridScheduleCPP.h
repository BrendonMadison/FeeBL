#ifndef FEEBL_ADAPTIVE_GRID_SCHEDULE_CPP_H
#define FEEBL_ADAPTIVE_GRID_SCHEDULE_CPP_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace feebl {
namespace adaptive_grid {

// Increment this value only for an intentionally incompatible on-disk schema.
// Readers reject every other value rather than guessing how to interpret it.
constexpr std::uint32_t kAdaptiveGridScheduleFormatVersion = 1U;

// Shared SHA-256 implementation for schedule-adjacent canonical receipts.
// The input may contain embedded NUL bytes (particle-ID streams do).
std::string sha256_hex(std::string_view input);

class AdaptiveGridScheduleError : public std::runtime_error {
 public:
  explicit AdaptiveGridScheduleError(const std::string& message)
      : std::runtime_error(message) {}
};

// Immutable through its public interface.  The card and particle-ordering
// fields are SHA-256 digests.  particle_ordering_sha256 identifies an ordering
// contract or an ID stream; it must never be computed from particle coordinates.
class AdaptiveGridScheduleHeader {
 public:
  AdaptiveGridScheduleHeader(std::string card_sha256,
                             std::uint64_t canonical_lattice_hash,
                             std::string source_id,
                             std::string build_id,
                             std::string controller_id,
                             std::string particle_ordering_sha256);

  std::uint32_t format_version() const noexcept {
    return kAdaptiveGridScheduleFormatVersion;
  }
  const std::string& card_sha256() const noexcept { return card_sha256_; }
  std::uint64_t canonical_lattice_hash() const noexcept {
    return canonical_lattice_hash_;
  }
  const std::string& source_id() const noexcept { return source_id_; }
  const std::string& build_id() const noexcept { return build_id_; }
  const std::string& controller_id() const noexcept { return controller_id_; }
  const std::string& particle_ordering_sha256() const noexcept {
    return particle_ordering_sha256_;
  }

  bool operator==(const AdaptiveGridScheduleHeader& other) const noexcept;
  bool operator!=(const AdaptiveGridScheduleHeader& other) const noexcept {
    return !(*this == other);
  }

 private:
  std::string card_sha256_;
  std::uint64_t canonical_lattice_hash_;
  std::string source_id_;
  std::string build_id_;
  std::string controller_id_;
  std::string particle_ordering_sha256_;
};

// The scheduler supplies these monotonically increasing canonical coordinates.
// No worker number, GPU batch number, allocator state, or timing is present.
struct AdaptiveGridEncounterKey {
  std::uint64_t crossing = 0U;
  std::uint64_t diagonal = 0U;
  std::uint64_t wave = 0U;
  std::uint64_t substep = 0U;
  std::uint64_t encounter = 0U;

  bool operator==(const AdaptiveGridEncounterKey& other) const noexcept;
  bool operator!=(const AdaptiveGridEncounterKey& other) const noexcept {
    return !(*this == other);
  }
  bool operator<(const AdaptiveGridEncounterKey& other) const noexcept;
};

// An encounter record is immutable after construction.  The dimensions are
// the selected physical window and its doubled/padded transform.  class_id is
// a versioned class-bank label; plan_hash is the exact deterministic planner
// hash.  Counts describe accepted sources/targets, never coordinate digests.
class AdaptiveGridEncounterRecord {
 public:
  AdaptiveGridEncounterRecord(AdaptiveGridEncounterKey key,
                              std::int64_t beam1_slice,
                              std::int64_t beam2_slice,
                              std::int64_t origin_x,
                              std::int64_t origin_y,
                              std::uint64_t n_x,
                              std::uint64_t n_y,
                              std::uint64_t padded_n_x,
                              std::uint64_t padded_n_y,
                              std::uint64_t plan_hash,
                              std::string class_id,
                              std::uint64_t source_count,
                              std::uint64_t target_count,
                              bool full_window);

  const AdaptiveGridEncounterKey& key() const noexcept { return key_; }
  std::int64_t beam1_slice() const noexcept { return beam1_slice_; }
  std::int64_t beam2_slice() const noexcept { return beam2_slice_; }
  std::int64_t origin_x() const noexcept { return origin_x_; }
  std::int64_t origin_y() const noexcept { return origin_y_; }
  std::uint64_t n_x() const noexcept { return n_x_; }
  std::uint64_t n_y() const noexcept { return n_y_; }
  std::uint64_t padded_n_x() const noexcept { return padded_n_x_; }
  std::uint64_t padded_n_y() const noexcept { return padded_n_y_; }
  std::uint64_t plan_hash() const noexcept { return plan_hash_; }
  const std::string& class_id() const noexcept { return class_id_; }
  std::uint64_t source_count() const noexcept { return source_count_; }
  std::uint64_t target_count() const noexcept { return target_count_; }
  bool full_window() const noexcept { return full_window_; }

  bool operator==(const AdaptiveGridEncounterRecord& other) const noexcept;
  bool operator!=(const AdaptiveGridEncounterRecord& other) const noexcept {
    return !(*this == other);
  }

 private:
  AdaptiveGridEncounterKey key_;
  std::int64_t beam1_slice_;
  std::int64_t beam2_slice_;
  std::int64_t origin_x_;
  std::int64_t origin_y_;
  std::uint64_t n_x_;
  std::uint64_t n_y_;
  std::uint64_t padded_n_x_;
  std::uint64_t padded_n_y_;
  std::uint64_t plan_hash_;
  std::string class_id_;
  std::uint64_t source_count_;
  std::uint64_t target_count_;
  bool full_window_;
};

class AdaptiveGridSchedule {
 public:
  explicit AdaptiveGridSchedule(AdaptiveGridScheduleHeader header);

  const AdaptiveGridScheduleHeader& header() const noexcept { return header_; }
  const std::vector<AdaptiveGridEncounterRecord>& records() const noexcept {
    return records_;
  }

  // Appends are accepted only in strict canonical key order.  Equal keys are
  // duplicates and are rejected, even if the remaining record differs.
  void append(const AdaptiveGridEncounterRecord& record);

  // SHA-256 of the canonical payload (everything except the checksum line).
  // This value is stable across CPU/GPU backends and operating systems.
  std::string stable_hash() const;

  // Writes a temporary sibling, closes it, and replaces the destination with
  // one rename operation on POSIX or MoveFileExW on Windows.  A failed replace
  // leaves the previous destination intact wherever the platform guarantees it.
  void save_atomic(const std::filesystem::path& path) const;

  // Strict reader: malformed numbers, noncanonical encoding, bad checksum,
  // truncation, unknown version, reordered or duplicate records all fail.
  static AdaptiveGridSchedule load(const std::filesystem::path& path);

  // Exact replay validation.  This compares the complete header and every
  // immutable record and rejects missing or surplus encounters.
  void validate_exact(const AdaptiveGridSchedule& expected) const;

  bool operator==(const AdaptiveGridSchedule& other) const noexcept;
  bool operator!=(const AdaptiveGridSchedule& other) const noexcept {
    return !(*this == other);
  }

 private:
  AdaptiveGridScheduleHeader header_;
  std::vector<AdaptiveGridEncounterRecord> records_;

  std::string canonical_payload() const;
};

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_ADAPTIVE_GRID_SCHEDULE_CPP_H
