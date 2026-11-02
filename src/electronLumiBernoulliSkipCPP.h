#ifndef FEEBL_ELECTRON_LUMI_BERNOULLI_SKIP_CPP_H
#define FEEBL_ELECTRON_LUMI_BERNOULLI_SKIP_CPP_H

#include <cstdint>

#include "counterRngCPP.h"

namespace feebl {
namespace electron_lumi_sampling {

// Version 1 freezes the physical-key mapping and binary64 geometric inverse
// transform described in electronLumiBernoulliSkipCPP.cc.  A future mapping or
// numerical algorithm must use a new version; changing this constant in place
// would silently invalidate reproducibility.
constexpr std::uint32_t kElectronLumiBernoulliSkipVersionV1 = 1U;

// Physical identity of one canonically ordered luminosity row.  The row's
// opportunity ordinal is deliberately not stored here: geometric gaps advance
// through the canonical order without one RNG lookup per opportunity.
struct ElectronLumiBernoulliRowPrefixV1 {
  std::uint32_t cursor_version = kElectronLumiBernoulliSkipVersionV1;
  std::uint64_t run_seed = 1U;
  std::uint64_t crossing = 0U;
  std::uint64_t diagonal = 0U;
  std::uint64_t wave = 0U;
  std::uint64_t substep = 0U;
  std::uint64_t encounter = 0U;
  std::uint32_t beam = 1U;
  std::uint64_t row_parent = kNoParentStableIdV2;
  std::uint64_t row_ordinal = 0U;
};

struct ElectronLumiBernoulliSkipLimitsV1 {
  std::uint64_t max_opportunities = kMaxCounterOrdinalV2;
  std::uint64_t max_rng_calls =
      CounterRngV2::kMaxUniformDoubleIndex + 1U;
  std::uint64_t max_bernoulli_successes = kMaxCounterOrdinalV2;
  std::uint64_t max_total_copies = kMaxCounterOrdinalV2;
};

struct ElectronLumiBernoulliSkipTelemetryV1 {
  std::uint64_t row_opportunities = 0U;
  std::uint64_t opportunities_decided = 0U;
  std::uint64_t rng_calls = 0U;
  std::uint64_t bernoulli_successes = 0U;
  std::uint64_t integer_copies_per_opportunity = 0U;
  std::uint64_t integer_copies_total = 0U;
  std::uint64_t selected_copies_total = 0U;
};

enum class ElectronLumiBernoulliSkipErrorV1 : std::uint8_t {
  None = 0,
  Uninitialized = 1,
  UnsupportedVersion = 2,
  InvalidPhysicalPrefix = 3,
  InvalidExpectedStore = 4,
  IntegerCopiesOverflow = 5,
  InvalidLimits = 6,
  RowLengthLimit = 7,
  IntegerCopyProductOverflow = 8,
  TotalCopyLimit = 9,
  RngLimit = 10,
  SuccessLimit = 11,
  InvalidBoundary = 12,
  NullOutput = 13,
  GapMathFailure = 14
};

enum class ElectronLumiBernoulliSkipResultV1 : std::uint8_t {
  Success = 0,
  Boundary = 1,
  End = 2,
  Error = 3
};

// Allocation-free cursor over the fractional Bernoulli successes in one row.
// The caller emits integer_copies_per_opportunity() copies for every row item,
// and one additional copy for each opportunity returned by next_success().
//
// next_success_before() supports monotonically increasing partition boundaries
// without changing the stream.  Row scheduling order and worker assignment are
// absent from the physical key.  Opportunity order, however, is semantic and
// must be the same canonical stable-ID order on every run.
class ElectronLumiBernoulliSkipCursorV1 {
 public:
  ElectronLumiBernoulliSkipCursorV1() noexcept;

  bool initialize(
      const ElectronLumiBernoulliRowPrefixV1& prefix,
      double expected_store,
      std::uint64_t row_length,
      const ElectronLumiBernoulliSkipLimitsV1& limits =
          ElectronLumiBernoulliSkipLimitsV1()) noexcept;

  ElectronLumiBernoulliSkipResultV1 next_success(
      std::uint64_t* opportunity) noexcept;

  // Return Success for the next index below end_exclusive, Boundary when this
  // partition contains no further success, End at the complete row end, and
  // Error after any fail-closed contract violation. Boundaries must be
  // nondecreasing and may not exceed row_length().
  ElectronLumiBernoulliSkipResultV1 next_success_before(
      std::uint64_t end_exclusive,
      std::uint64_t* opportunity) noexcept;

  bool healthy() const noexcept {
    return initialized_ && error_ == ElectronLumiBernoulliSkipErrorV1::None;
  }
  bool exhausted() const noexcept { return exhausted_ && healthy(); }
  ElectronLumiBernoulliSkipErrorV1 error() const noexcept { return error_; }
  std::uint64_t row_length() const noexcept { return row_length_; }
  std::uint64_t integer_copies_per_opportunity() const noexcept {
    return telemetry_.integer_copies_per_opportunity;
  }
  double fractional_probability() const noexcept {
    return fractional_probability_;
  }
  const ElectronLumiBernoulliSkipTelemetryV1& telemetry() const noexcept {
    return telemetry_;
  }

 private:
  ElectronLumiBernoulliRowPrefixV1 prefix_;
  ElectronLumiBernoulliSkipLimitsV1 limits_;
  ElectronLumiBernoulliSkipTelemetryV1 telemetry_;
  PhysicalRandomKeyV2 random_key_;
  double fractional_probability_;
  std::uint64_t row_length_;
  std::uint64_t next_origin_;
  std::uint64_t pending_success_;
  std::uint64_t largest_boundary_;
  bool initialized_;
  bool exhausted_;
  bool have_pending_success_;
  ElectronLumiBernoulliSkipErrorV1 error_;

  bool schedule_from(std::uint64_t origin) noexcept;
  bool poison(ElectronLumiBernoulliSkipErrorV1 error) noexcept;
};

const char* electron_lumi_bernoulli_skip_error_name_v1(
    ElectronLumiBernoulliSkipErrorV1 error) noexcept;

}  // namespace electron_lumi_sampling
}  // namespace feebl

#endif  // FEEBL_ELECTRON_LUMI_BERNOULLI_SKIP_CPP_H
