#include "electronLumiBernoulliSkipCPP.h"

#include <cmath>
#include <limits>

namespace feebl {
namespace electron_lumi_sampling {

namespace {

constexpr double kTwoTo64 = 18446744073709551616.0;

bool multiply_u64(std::uint64_t left, std::uint64_t right,
                  std::uint64_t* product) noexcept {
  if (left != 0U && right > kMaxCounterOrdinalV2 / left) return false;
  *product = left * right;
  return true;
}

}  // namespace

ElectronLumiBernoulliSkipCursorV1::ElectronLumiBernoulliSkipCursorV1()
    noexcept
    : fractional_probability_(0.0),
      row_length_(0U),
      next_origin_(0U),
      pending_success_(0U),
      largest_boundary_(0U),
      initialized_(false),
      exhausted_(false),
      have_pending_success_(false),
      error_(ElectronLumiBernoulliSkipErrorV1::Uninitialized) {}

bool ElectronLumiBernoulliSkipCursorV1::poison(
    ElectronLumiBernoulliSkipErrorV1 error) noexcept {
  error_ = error;
  exhausted_ = false;
  have_pending_success_ = false;
  return false;
}

bool ElectronLumiBernoulliSkipCursorV1::initialize(
    const ElectronLumiBernoulliRowPrefixV1& prefix,
    double expected_store,
    std::uint64_t row_length,
    const ElectronLumiBernoulliSkipLimitsV1& limits) noexcept {
  prefix_ = prefix;
  limits_ = limits;
  telemetry_ = ElectronLumiBernoulliSkipTelemetryV1();
  telemetry_.row_opportunities = row_length;
  fractional_probability_ = 0.0;
  row_length_ = row_length;
  next_origin_ = 0U;
  pending_success_ = 0U;
  largest_boundary_ = 0U;
  initialized_ = false;
  exhausted_ = false;
  have_pending_success_ = false;
  error_ = ElectronLumiBernoulliSkipErrorV1::Uninitialized;

  if (prefix.cursor_version != kElectronLumiBernoulliSkipVersionV1)
    return poison(ElectronLumiBernoulliSkipErrorV1::UnsupportedVersion);
  if ((prefix.beam != 1U && prefix.beam != 2U) ||
      prefix.row_parent == kNoParentStableIdV2)
    return poison(ElectronLumiBernoulliSkipErrorV1::InvalidPhysicalPrefix);
  if (!std::isfinite(expected_store) || expected_store < 0.0)
    return poison(ElectronLumiBernoulliSkipErrorV1::InvalidExpectedStore);
  if (expected_store >= kTwoTo64)
    return poison(ElectronLumiBernoulliSkipErrorV1::IntegerCopiesOverflow);
  const std::uint64_t max_uniform_double_calls =
      CounterRngV2::kMaxUniformDoubleIndex + 1U;
  if (limits.max_rng_calls > max_uniform_double_calls)
    return poison(ElectronLumiBernoulliSkipErrorV1::InvalidLimits);
  if (row_length > limits.max_opportunities)
    return poison(ElectronLumiBernoulliSkipErrorV1::RowLengthLimit);

  const double integer_part = std::floor(expected_store);
  telemetry_.integer_copies_per_opportunity =
      static_cast<std::uint64_t>(integer_part);
  fractional_probability_ = expected_store - integer_part;
  if (!(fractional_probability_ > 0.0)) fractional_probability_ = 0.0;
  if (!(fractional_probability_ >= 0.0) ||
      !(fractional_probability_ < 1.0))
    return poison(ElectronLumiBernoulliSkipErrorV1::InvalidExpectedStore);

  if (!multiply_u64(telemetry_.integer_copies_per_opportunity,
                    row_length, &telemetry_.integer_copies_total))
    return poison(
        ElectronLumiBernoulliSkipErrorV1::IntegerCopyProductOverflow);
  telemetry_.selected_copies_total = telemetry_.integer_copies_total;
  if (telemetry_.integer_copies_total > limits.max_total_copies)
    return poison(ElectronLumiBernoulliSkipErrorV1::TotalCopyLimit);

  random_key_ = PhysicalRandomKeyV2();
  random_key_.run_seed = prefix.run_seed;
  random_key_.domain = RandomDomain::LuminosityEventSampling;
  random_key_.crossing = prefix.crossing;
  random_key_.diagonal = prefix.diagonal;
  random_key_.wave = prefix.wave;
  random_key_.substep = prefix.substep;
  random_key_.encounter = prefix.encounter;
  random_key_.beam = prefix.beam;
  random_key_.parent_a = prefix.row_parent;
  random_key_.parent_b = kNoParentStableIdV2;
  random_key_.interaction = prefix.row_ordinal;
  random_key_.emission = kElectronLumiBernoulliSkipVersionV1;
  random_key_.rejection_attempt = 0U;

  initialized_ = true;
  error_ = ElectronLumiBernoulliSkipErrorV1::None;
  if (row_length == 0U || fractional_probability_ == 0.0) {
    exhausted_ = true;
    telemetry_.opportunities_decided = row_length;
    return true;
  }
  return schedule_from(0U);
}

// If U is uniform on (0,1), then
//
//   G = floor(log(U) / log(1-p))
//
// satisfies P(G=k)=(1-p)^k p.  Independent CounterRngV2 draws therefore give
// the failure gaps of the same iid Bernoulli(p) sequence that a scalar loop
// would generate, but require only K plus (usually) one terminal draw for K
// successes.  The comparisons against remaining happen before integer
// conversion, so extremely small p cannot overflow a gap ordinal.
bool ElectronLumiBernoulliSkipCursorV1::schedule_from(
    std::uint64_t origin) noexcept {
  if (origin >= row_length_) {
    exhausted_ = true;
    have_pending_success_ = false;
    telemetry_.opportunities_decided = row_length_;
    return true;
  }
  if (telemetry_.rng_calls >= limits_.max_rng_calls ||
      telemetry_.rng_calls > CounterRngV2::kMaxUniformDoubleIndex)
    return poison(ElectronLumiBernoulliSkipErrorV1::RngLimit);

  double uniform = 0.0;
  try {
    uniform = CounterRngV2::uniform_double(random_key_, telemetry_.rng_calls);
  } catch (...) {
    return poison(ElectronLumiBernoulliSkipErrorV1::RngLimit);
  }
  ++telemetry_.rng_calls;
  const double numerator = std::log(uniform);
  const double denominator = std::log1p(-fractional_probability_);
  if (!(numerator < 0.0) || !(denominator < 0.0) ||
      !std::isfinite(numerator) || !std::isfinite(denominator))
    return poison(ElectronLumiBernoulliSkipErrorV1::GapMathFailure);
  const long double quotient =
      static_cast<long double>(numerator) /
      static_cast<long double>(denominator);
  if (!(quotient >= 0.0L))
    return poison(ElectronLumiBernoulliSkipErrorV1::GapMathFailure);

  const std::uint64_t remaining = row_length_ - origin;
  if (!std::isfinite(quotient) ||
      quotient >= static_cast<long double>(remaining)) {
    exhausted_ = true;
    have_pending_success_ = false;
    telemetry_.opportunities_decided = row_length_;
    return true;
  }
  const std::uint64_t failures = static_cast<std::uint64_t>(quotient);
  if (failures >= remaining)
    return poison(ElectronLumiBernoulliSkipErrorV1::GapMathFailure);
  pending_success_ = origin + failures;
  next_origin_ = pending_success_ + 1U;
  have_pending_success_ = true;
  if (telemetry_.opportunities_decided < pending_success_)
    telemetry_.opportunities_decided = pending_success_;
  return true;
}

ElectronLumiBernoulliSkipResultV1
ElectronLumiBernoulliSkipCursorV1::next_success(
    std::uint64_t* opportunity) noexcept {
  return next_success_before(row_length_, opportunity);
}

ElectronLumiBernoulliSkipResultV1
ElectronLumiBernoulliSkipCursorV1::next_success_before(
    std::uint64_t end_exclusive,
    std::uint64_t* opportunity) noexcept {
  if (opportunity == 0) {
    poison(ElectronLumiBernoulliSkipErrorV1::NullOutput);
    return ElectronLumiBernoulliSkipResultV1::Error;
  }
  if (!initialized_ || error_ != ElectronLumiBernoulliSkipErrorV1::None)
    return ElectronLumiBernoulliSkipResultV1::Error;
  if (end_exclusive > row_length_ || end_exclusive < largest_boundary_) {
    poison(ElectronLumiBernoulliSkipErrorV1::InvalidBoundary);
    return ElectronLumiBernoulliSkipResultV1::Error;
  }
  largest_boundary_ = end_exclusive;

  if (!have_pending_success_ && !exhausted_ &&
      !schedule_from(next_origin_))
    return ElectronLumiBernoulliSkipResultV1::Error;
  if (exhausted_) {
    if (telemetry_.opportunities_decided < end_exclusive)
      telemetry_.opportunities_decided = end_exclusive;
    return end_exclusive == row_length_
               ? ElectronLumiBernoulliSkipResultV1::End
               : ElectronLumiBernoulliSkipResultV1::Boundary;
  }
  if (pending_success_ >= end_exclusive) {
    if (telemetry_.opportunities_decided < end_exclusive)
      telemetry_.opportunities_decided = end_exclusive;
    return ElectronLumiBernoulliSkipResultV1::Boundary;
  }
  if (telemetry_.bernoulli_successes >=
      limits_.max_bernoulli_successes) {
    poison(ElectronLumiBernoulliSkipErrorV1::SuccessLimit);
    return ElectronLumiBernoulliSkipResultV1::Error;
  }
  if (telemetry_.selected_copies_total >= limits_.max_total_copies) {
    poison(ElectronLumiBernoulliSkipErrorV1::TotalCopyLimit);
    return ElectronLumiBernoulliSkipResultV1::Error;
  }

  *opportunity = pending_success_;
  ++telemetry_.bernoulli_successes;
  ++telemetry_.selected_copies_total;
  if (telemetry_.opportunities_decided < pending_success_ + 1U)
    telemetry_.opportunities_decided = pending_success_ + 1U;
  have_pending_success_ = false;
  if (next_origin_ >= row_length_) {
    exhausted_ = true;
    telemetry_.opportunities_decided = row_length_;
  }
  return ElectronLumiBernoulliSkipResultV1::Success;
}

const char* electron_lumi_bernoulli_skip_error_name_v1(
    ElectronLumiBernoulliSkipErrorV1 error) noexcept {
  switch (error) {
    case ElectronLumiBernoulliSkipErrorV1::None: return "none";
    case ElectronLumiBernoulliSkipErrorV1::Uninitialized:
      return "uninitialized";
    case ElectronLumiBernoulliSkipErrorV1::UnsupportedVersion:
      return "unsupported-version";
    case ElectronLumiBernoulliSkipErrorV1::InvalidPhysicalPrefix:
      return "invalid-physical-prefix";
    case ElectronLumiBernoulliSkipErrorV1::InvalidExpectedStore:
      return "invalid-expected-store";
    case ElectronLumiBernoulliSkipErrorV1::IntegerCopiesOverflow:
      return "integer-copies-overflow";
    case ElectronLumiBernoulliSkipErrorV1::InvalidLimits:
      return "invalid-limits";
    case ElectronLumiBernoulliSkipErrorV1::RowLengthLimit:
      return "row-length-limit";
    case ElectronLumiBernoulliSkipErrorV1::IntegerCopyProductOverflow:
      return "integer-copy-product-overflow";
    case ElectronLumiBernoulliSkipErrorV1::TotalCopyLimit:
      return "total-copy-limit";
    case ElectronLumiBernoulliSkipErrorV1::RngLimit: return "rng-limit";
    case ElectronLumiBernoulliSkipErrorV1::SuccessLimit:
      return "success-limit";
    case ElectronLumiBernoulliSkipErrorV1::InvalidBoundary:
      return "invalid-boundary";
    case ElectronLumiBernoulliSkipErrorV1::NullOutput: return "null-output";
    case ElectronLumiBernoulliSkipErrorV1::GapMathFailure:
      return "gap-math-failure";
  }
  return "unknown";
}

}  // namespace electron_lumi_sampling
}  // namespace feebl
