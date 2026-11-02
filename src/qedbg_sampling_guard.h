#ifndef FEEBL_QEDBG_SAMPLING_GUARD_H
#define FEEBL_QEDBG_SAMPLING_GUARD_H

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qedbg {

// Bit inspection remains valid under -ffast-math, unlike std::isfinite.
// The native interchange and kernel weights use IEEE binary64.
inline bool finite_nonnegative(double value) {
  static_assert(sizeof(double)==sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559,
                "QED weight validation requires IEEE binary64");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return (bits >> 63) == 0 &&
         (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

// Computational limits per photon-pair proposal. They never truncate a
// physical contribution: exceeding a limit rejects the run with context.
constexpr int kMaximumPairWork = 1000000;
constexpr unsigned kMaximumAngleTrials = 1000000;

inline void reject_weight(double value, const char* quantity, int process) {
  std::ostringstream message;
  message << std::setprecision(17) << "Invalid or excessive QED sampling work: "
          << quantity << '=' << value << " process=" << process
          << " limit=" << kMaximumPairWork;
  throw std::domain_error(message.str());
}

// Number of independent kernel evaluations, floor(expected yield)+1.
// Validate before converting to int, so overflow cannot escape a loop bound.
inline int checked_repetitions(double expected, const char* quantity, int process) {
  if (!finite_nonnegative(expected) || expected >= kMaximumPairWork)
    reject_weight(expected, quantity, process);
  return static_cast<int>(expected)+1;
}

// The chunk loop splits an existing weighted event; its copies are correlated.
inline void check_chunk_weight(double weight, const char* quantity, int process) {
  if (!finite_nonnegative(weight) || weight > kMaximumPairWork)
    reject_weight(weight, quantity, process);
}

}  // namespace qedbg
#endif
