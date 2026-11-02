#ifndef FEEBL_MODERN_TYPES_CPP_H
#define FEEBL_MODERN_TYPES_CPP_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

// Fundamental, storage-independent types for the replacement FeeBL core.
//
// The modern core deliberately uses explicit-width identifiers and offsets.
// They are written to traces, used in counter-RNG keys, and eventually shared
// by CPU and GPU implementations; their meaning must therefore not depend on
// pointer size, allocation order, or a worker/thread identifier.
namespace feebl {
namespace modern {

using Scalar = double;
using StableParticleId = std::uint64_t;
using ParticleIndex = std::uint64_t;
using SliceIndex = std::uint64_t;
using SubstepIndex = std::uint64_t;
using ArrayOffset = std::uint64_t;
using CellKey = std::uint64_t;
using EncounterId = std::uint64_t;
using WaveIndex = std::uint64_t;

constexpr CellKey kOutsideCell = std::numeric_limits<CellKey>::max();
constexpr EncounterId kInvalidEncounter =
    std::numeric_limits<EncounterId>::max();

// The first modern-core bridge intentionally retains FeeBL's externally
// visible units.  Conversion to SI belongs at a validated boundary rather than
// in hot kernels: positions are nm, energy is GeV, vx/vy are dimensionless
// slopes, macro_weight is dimensionless, and charge_e is in units of |e|.
enum class UnitConvention : std::uint8_t {
  LegacyFeeBLNmGeV = 1
};

enum class ParticleSpecies : std::uint8_t {
  Electron = 1,
  Positron = 2,
  Photon = 3,
  MuonMinus = 4,
  MuonPlus = 5,
  Other = 255
};

// A compact first-error result keeps validation usable in tests and at future
// adapter boundaries without forcing an exception-only interface.
struct ValidationResult {
  bool valid = true;
  std::string message;
  std::uint64_t index = 0;

  explicit operator bool() const noexcept { return valid; }

  static ValidationResult success() { return ValidationResult{}; }

  static ValidationResult failure(const std::string& why,
                                  std::uint64_t at = 0) {
    ValidationResult result;
    result.valid = false;
    result.message = why;
    result.index = at;
    return result;
  }
};

inline bool checked_add(std::uint64_t a, std::uint64_t b,
                        std::uint64_t& result) noexcept {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) return false;
  result = a + b;
  return true;
}

inline bool checked_multiply(std::uint64_t a, std::uint64_t b,
                             std::uint64_t& result) noexcept {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    return false;
  result = a * b;
  return true;
}

inline bool fits_size_t(std::uint64_t value) noexcept {
  return value <= static_cast<std::uint64_t>(
      std::numeric_limits<std::size_t>::max());
}

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_TYPES_CPP_H
