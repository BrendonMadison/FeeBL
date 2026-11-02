#include "electronLumiSpatialCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <utility>

namespace feebl {
namespace electron_lumi_spatial {
namespace {

static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8U,
              "physical collision raster requires IEEE-754 binary64");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "physical collision raster requires size_t no wider than uint64_t");

constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
constexpr std::uint64_t kLargestExactlyRepresentableInteger =
    UINT64_C(1) << 53U;

[[noreturn]] void fail(SpatialErrorCode code, const std::string& message) {
  throw SpatialError(code, message);
}

std::uint64_t binary64_bits(double value) noexcept {
  std::uint64_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool finite_value(double value) noexcept {
  const std::uint64_t bits = binary64_bits(value);
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

bool exact_zero(double value) noexcept {
  return (binary64_bits(value) & UINT64_C(0x7fffffffffffffff)) == 0U;
}

double canonical_zero(double value) noexcept {
  return exact_zero(value) ? 0.0 : value;
}

// IEEE bit ordering avoids process-wide DAZ/FTZ state inherited from the
// historical FeeBL link from turning the two subnormal nextafter boundary
// sentinels into zero during admission comparisons.
bool finite_less(double left, double right) noexcept {
  left = canonical_zero(left);
  right = canonical_zero(right);
  const std::uint64_t left_bits = binary64_bits(left);
  const std::uint64_t right_bits = binary64_bits(right);
  if (left_bits == right_bits) return false;
  const bool left_negative = (left_bits >> 63U) != 0U;
  const bool right_negative = (right_bits >> 63U) != 0U;
  if (left_negative != right_negative) return left_negative;
  return left_negative ? left_bits > right_bits : left_bits < right_bits;
}

bool finite_equal(double left, double right) noexcept {
  return binary64_bits(canonical_zero(left)) ==
         binary64_bits(canonical_zero(right));
}

bool strictly_positive_finite(double value) noexcept {
  const std::uint64_t bits = binary64_bits(value);
  return finite_value(value) && (bits >> 63U) == 0U &&
         (bits & UINT64_C(0x7fffffffffffffff)) != 0U;
}

bool negative_finite(double value) noexcept {
  const std::uint64_t bits = binary64_bits(value);
  return finite_value(value) && (bits >> 63U) != 0U && !exact_zero(value);
}

bool finite_value(long double value) noexcept {
  // This translation unit is compiled with -fno-fast-math.  Unlike binary64,
  // long-double representation is ABI dependent, so use the standard
  // classifier here instead of assuming an exponent layout.
  return std::isfinite(value);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* quantity) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    std::ostringstream message;
    message << quantity << " multiplication overflows size_t";
    fail(SpatialErrorCode::ArithmeticOverflow, message.str());
  }
  return left * right;
}

double checked_nonnegative_double(long double value, const char* quantity) {
  if (!finite_value(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<double>::max())) {
    std::ostringstream message;
    message << quantity << " is not representable as a finite non-negative "
            << "binary64 value";
    fail(SpatialErrorCode::ArithmeticOverflow, message.str());
  }
  const double result = static_cast<double>(value);
  if (value > 0.0L && exact_zero(result)) {
    std::ostringstream message;
    message << quantity << " underflows binary64";
    fail(SpatialErrorCode::ArithmeticUnderflow, message.str());
  }
  if (!finite_value(result)) {
    std::ostringstream message;
    message << quantity << " conversion produced a non-finite value";
    fail(SpatialErrorCode::ArithmeticOverflow, message.str());
  }
  return result;
}

class PositiveAccumulator {
 public:
  void add(long double value, const char* quantity) {
    if (!finite_value(value) || value < 0.0L) {
      std::ostringstream message;
      message << quantity << " contribution is non-finite or negative";
      fail(SpatialErrorCode::InternalInvariant, message.str());
    }
    // Neumaier compensation is deterministic in canonical traversal order.
    const long double next = sum_ + value;
    if (!finite_value(next)) {
      std::ostringstream message;
      message << quantity << " accumulation overflows long double";
      fail(SpatialErrorCode::ArithmeticOverflow, message.str());
    }
    if (std::fabs(sum_) >= std::fabs(value))
      correction_ += (sum_ - next) + value;
    else
      correction_ += (value - next) + sum_;
    sum_ = next;
    if (!finite_value(correction_)) {
      std::ostringstream message;
      message << quantity << " compensation overflows long double";
      fail(SpatialErrorCode::ArithmeticOverflow, message.str());
    }
  }

  long double value(const char* quantity) const {
    const long double result = sum_ + correction_;
    if (!finite_value(result) || result < 0.0L) {
      std::ostringstream message;
      message << quantity << " accumulated value is invalid";
      fail(SpatialErrorCode::InternalInvariant, message.str());
    }
    return result;
  }

 private:
  long double sum_ = 0.0L;
  long double correction_ = 0.0L;
};

class StableHash {
 public:
  void add_byte(std::uint8_t value) noexcept {
    value_ ^= value;
    value_ *= kFnvPrime;
  }

  void add_u32(std::uint32_t value) noexcept {
    for (unsigned int shift = 0U; shift < 32U; shift += 8U)
      add_byte(static_cast<std::uint8_t>((value >> shift) & UINT32_C(0xff)));
  }

  void add_u64(std::uint64_t value) noexcept {
    for (unsigned int shift = 0U; shift < 64U; shift += 8U)
      add_byte(static_cast<std::uint8_t>((value >> shift) & UINT64_C(0xff)));
  }

  void add_size(std::size_t value) noexcept {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "hash requires size_t no wider than uint64_t");
    add_u64(static_cast<std::uint64_t>(value));
  }

  void add_double(double value) noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    add_u64(bits);
  }

  void add_string(const char* value) noexcept {
    std::size_t length = 0U;
    while (value[length] != '\0') ++length;
    add_size(length);
    for (std::size_t index = 0U; index < length; ++index)
      add_byte(static_cast<std::uint8_t>(value[index]));
  }

  std::uint64_t value() const noexcept { return value_; }

 private:
  std::uint64_t value_ = kFnvOffset;
};

struct AxisStencil {
  std::size_t base = 0U;
  double fraction_upper = 0.0;
};

AxisStencil map_axis(double coordinate, double minimum, double maximum,
                     double spacing, std::size_t nodes,
                     const char* axis_name) {
  if (finite_less(coordinate, minimum) || finite_less(maximum, coordinate)) {
    std::ostringstream message;
    message << axis_name << " coordinate lies outside the physical raster";
    fail(SpatialErrorCode::OutOfDomainRoot, message.str());
  }

  AxisStencil result;
  if (finite_equal(coordinate, maximum)) {
    result.base = nodes - 2U;
    result.fraction_upper = 1.0;
    return result;
  }

  const double scaled = (coordinate - minimum) / spacing;
  if (!finite_value(scaled) || scaled < 0.0) {
    std::ostringstream message;
    message << axis_name << " CIC coordinate mapping is non-finite";
    fail(SpatialErrorCode::ArithmeticOverflow, message.str());
  }
  const double last_node = static_cast<double>(nodes - 1U);
  if (scaled >= last_node) {
    // A representable in-domain coordinate may round to the inclusive upper
    // node.  Give it the same complete stencil as the exact endpoint.
    result.base = nodes - 2U;
    result.fraction_upper = 1.0;
    return result;
  }

  const double floored = std::floor(scaled);
  if (!finite_value(floored) || floored < 0.0 || floored > last_node - 1.0) {
    std::ostringstream message;
    message << axis_name << " CIC base index is invalid";
    fail(SpatialErrorCode::InternalInvariant, message.str());
  }
  result.base = static_cast<std::size_t>(floored);
  result.fraction_upper = scaled - floored;
  if (result.fraction_upper < 0.0 || result.fraction_upper > 1.0 ||
      !finite_value(result.fraction_upper)) {
    std::ostringstream message;
    message << axis_name << " CIC fraction is outside [0,1]";
    fail(SpatialErrorCode::InternalInvariant, message.str());
  }
  return result;
}

void validate_root(const PhysicalRoot& root) {
  if (root.id == 0U)
    fail(SpatialErrorCode::InvalidRoot,
         "physical collision root ID zero is reserved for missing identity");
  if (!finite_value(root.x) || !finite_value(root.y))
    fail(SpatialErrorCode::InvalidRoot,
         "physical collision root coordinates must be finite");
  if (!strictly_positive_finite(root.population))
    fail(SpatialErrorCode::InvalidRoot,
         "physical collision root population must be finite and positive");
}

RootStencil make_stencil(const PhysicalCicGeometry& geometry,
                         const PhysicalRoot& root) {
  const AxisStencil x = map_axis(root.x, geometry.x_min(), geometry.x_max(),
                                 geometry.delta_x(), geometry.n_x(), "x");
  const AxisStencil y = map_axis(root.y, geometry.y_min(), geometry.y_max(),
                                 geometry.delta_y(), geometry.n_y(), "y");
  const double lower_x = 1.0 - x.fraction_upper;
  const double lower_y = 1.0 - y.fraction_upper;

  RootStencil stencil;
  stencil.root = root;
  stencil.base_x = x.base;
  stencil.base_y = y.base;
  stencil.nodes = {{{x.base, y.base, lower_x * lower_y},
                    {x.base + 1U, y.base,
                     x.fraction_upper * lower_y},
                    {x.base, y.base + 1U,
                     lower_x * y.fraction_upper},
                    {x.base + 1U, y.base + 1U,
                     x.fraction_upper * y.fraction_upper}}};

  PositiveAccumulator sum;
  for (const CicNodeFraction& node : stencil.nodes) {
    if (node.x >= geometry.n_x() || node.y >= geometry.n_y() ||
        !finite_value(node.fraction) || node.fraction < 0.0 ||
        node.fraction > 1.0) {
      fail(SpatialErrorCode::InternalInvariant,
           "CIC construction produced an invalid full-support stencil");
    }
    sum.add(static_cast<long double>(node.fraction), "CIC stencil");
  }
  const long double error = std::fabs(sum.value("CIC stencil") - 1.0L);
  const long double tolerance =
      16.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());
  if (error > tolerance)
    fail(SpatialErrorCode::InternalInvariant,
         "CIC stencil fractions do not conserve one root");
  return stencil;
}

RejectionReason rejection_reason(const PhysicalCicGeometry& geometry,
                                 const PhysicalRoot& root) noexcept {
  RejectionReason result = RejectionReason::None;
  if (finite_less(root.x, geometry.x_min()))
    result = result | RejectionReason::BelowX;
  if (finite_less(geometry.x_max(), root.x))
    result = result | RejectionReason::AboveX;
  if (finite_less(root.y, geometry.y_min()))
    result = result | RejectionReason::BelowY;
  if (finite_less(geometry.y_max(), root.y))
    result = result | RejectionReason::AboveY;
  return result;
}

struct BaseCell {
  std::size_t x = 0U;
  std::size_t y = 0U;

  bool operator<(const BaseCell& other) const noexcept {
    return x < other.x || (x == other.x && y < other.y);
  }
};

long double stencil_overlap(const RootStencil& left,
                            const RootStencil& right) {
  PositiveAccumulator result;
  for (const CicNodeFraction& left_node : left.nodes) {
    for (const CicNodeFraction& right_node : right.nodes) {
      if (left_node.x == right_node.x && left_node.y == right_node.y) {
        const long double contribution =
            static_cast<long double>(left_node.fraction) *
            static_cast<long double>(right_node.fraction);
        result.add(contribution, "CIC pair overlap");
      }
    }
  }
  return result.value("CIC pair overlap");
}

double scaled_pair_weight(long double overlap,
                          const PhysicalCicGeometry& geometry,
                          const PhysicalRoot& beam1,
                          const PhysicalRoot& beam2,
                          double normalization) {
  long double value = overlap;
  value /= static_cast<long double>(geometry.control_area());
  value *= static_cast<long double>(beam1.population);
  value *= static_cast<long double>(beam2.population);
  value *= static_cast<long double>(normalization);
  return checked_nonnegative_double(value, "physical collision pair weight");
}

void validate_stencils(const PhysicalCicGeometry& geometry,
                       std::vector<RootStencil>* stencils,
                       const char* beam_name) {
  std::sort(stencils->begin(), stencils->end(),
            [](const RootStencil& left, const RootStencil& right) {
              return left.root.id < right.root.id;
            });
  RootId previous = 0U;
  bool have_previous = false;
  for (RootStencil& stencil : *stencils) {
    stencil.root.x = canonical_zero(stencil.root.x);
    stencil.root.y = canonical_zero(stencil.root.y);
    validate_root(stencil.root);
    if (have_previous && stencil.root.id == previous) {
      std::ostringstream message;
      message << beam_name << " contains duplicate root ID "
              << stencil.root.id;
      fail(SpatialErrorCode::DuplicateRootId, message.str());
    }
    have_previous = true;
    previous = stencil.root.id;
    if (rejection_reason(geometry, stencil.root) != RejectionReason::None)
      fail(SpatialErrorCode::OutOfDomainRoot,
           "density-dot stencil root lies outside the physical raster");
    const RootStencil reconstructed = make_stencil(geometry, stencil.root);
    if (stencil.base_x >= geometry.n_x() - 1U ||
        stencil.base_y >= geometry.n_y() - 1U)
      fail(SpatialErrorCode::InternalInvariant,
           "density-dot stencil has an invalid base cell");
    const std::array<std::pair<std::size_t, std::size_t>, 4U> expected =
        {{{stencil.base_x, stencil.base_y},
          {stencil.base_x + 1U, stencil.base_y},
          {stencil.base_x, stencil.base_y + 1U},
          {stencil.base_x + 1U, stencil.base_y + 1U}}};
    PositiveAccumulator sum;
    for (std::size_t node = 0U; node < stencil.nodes.size(); ++node) {
      if (stencil.nodes[node].x != expected[node].first ||
          stencil.nodes[node].y != expected[node].second ||
          stencil.nodes[node].x != reconstructed.nodes[node].x ||
          stencil.nodes[node].y != reconstructed.nodes[node].y ||
          !finite_value(stencil.nodes[node].fraction) ||
          stencil.nodes[node].fraction < 0.0 ||
          stencil.nodes[node].fraction > 1.0 ||
          stencil.nodes[node].fraction !=
              reconstructed.nodes[node].fraction)
        fail(SpatialErrorCode::InternalInvariant,
             "density-dot stencil differs from the root's canonical CIC projection");
      sum.add(static_cast<long double>(stencil.nodes[node].fraction),
              "density-dot CIC stencil");
    }
    const long double error =
        std::fabs(sum.value("density-dot CIC stencil") - 1.0L);
    if (error > 16.0L *
                    static_cast<long double>(
                        std::numeric_limits<double>::epsilon()))
      fail(SpatialErrorCode::InternalInvariant,
           "density-dot CIC stencil does not conserve one root");
  }
}

using NodeDensity = std::map<std::size_t, long double>;

NodeDensity deposit_density(const PhysicalCicGeometry& geometry,
                            const std::vector<RootStencil>& stencils,
                            const char* beam_name) {
  NodeDensity density;
  for (const RootStencil& stencil : stencils) {
    for (const CicNodeFraction& node : stencil.nodes) {
      if (node.fraction == 0.0) continue;
      const std::size_t key = geometry.linear_node(node.x, node.y);
      const long double contribution =
          static_cast<long double>(stencil.root.population) *
          static_cast<long double>(node.fraction);
      long double& destination = density[key];
      destination += contribution;
      if (!finite_value(destination) || destination < 0.0L ||
          destination >
              static_cast<long double>(std::numeric_limits<double>::max())) {
        std::ostringstream message;
        message << beam_name << " deposited node density overflows binary64";
        fail(SpatialErrorCode::ArithmeticOverflow, message.str());
      }
    }
  }
  return density;
}

void ensure_disjoint_ids(const std::vector<RootStencil>& beam1,
                         const std::vector<RootStencil>& beam2) {
  std::size_t left = 0U;
  std::size_t right = 0U;
  while (left < beam1.size() && right < beam2.size()) {
    if (beam1[left].root.id < beam2[right].root.id) {
      ++left;
    } else if (beam2[right].root.id < beam1[left].root.id) {
      ++right;
    } else {
      std::ostringstream message;
      message << "root ID " << beam1[left].root.id
              << " occurs in both physical beams";
      fail(SpatialErrorCode::DuplicateRootId, message.str());
    }
  }
}

std::uint64_t make_plan_hash(const PhysicalCicGeometry& geometry,
                             const std::vector<RootStencil>& beam1,
                             const std::vector<RootStencil>& beam2,
                             const std::vector<CollisionPair>& pairs,
                             double normalization) noexcept {
  StableHash hash;
  hash.add_string("FeeBLPhysicalCicCollisionPlan");
  hash.add_u32(kPhysicalCicKernelVersion);
  hash.add_u64(geometry.geometry_hash());
  hash.add_double(normalization);
  const auto add_stencils = [&hash](const std::vector<RootStencil>& values) {
    hash.add_size(values.size());
    for (const RootStencil& stencil : values) {
      hash.add_u64(stencil.root.id);
      hash.add_double(stencil.root.x);
      hash.add_double(stencil.root.y);
      hash.add_double(stencil.root.population);
      hash.add_size(stencil.base_x);
      hash.add_size(stencil.base_y);
      for (const CicNodeFraction& node : stencil.nodes) {
        hash.add_size(node.x);
        hash.add_size(node.y);
        hash.add_double(node.fraction);
      }
    }
  };
  add_stencils(beam1);
  add_stencils(beam2);
  hash.add_size(pairs.size());
  for (const CollisionPair& pair : pairs) {
    hash.add_u64(pair.beam1_id);
    hash.add_u64(pair.beam2_id);
    hash.add_double(pair.stencil_overlap);
    hash.add_double(pair.spatial_kernel);
    hash.add_double(pair.weight);
  }
  return hash.value();
}

}  // namespace

SpatialError::SpatialError(SpatialErrorCode code, const std::string& message)
    : std::runtime_error(message), code_(code) {}

PhysicalCicGeometry::PhysicalCicGeometry(
    const PhysicalCicGeometryInput& input)
    : n_x_(input.n_x),
      n_y_(input.n_y),
      node_count_(0U),
      x_min_(input.x_min),
      x_max_(input.x_max),
      y_min_(input.y_min),
      y_max_(input.y_max),
      delta_x_(0.0),
      delta_y_(0.0),
      control_area_(0.0),
      geometry_hash_(0U) {
  // Signed zero is not a distinct physical geometry and must not fork hashes.
  x_min_ = canonical_zero(x_min_);
  x_max_ = canonical_zero(x_max_);
  y_min_ = canonical_zero(y_min_);
  y_max_ = canonical_zero(y_max_);
  if (n_x_ < 2U || n_y_ < 2U)
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC raster requires at least two nodes on each axis");
  if (static_cast<std::uint64_t>(n_x_ - 1U) >
          kLargestExactlyRepresentableInteger ||
      static_cast<std::uint64_t>(n_y_ - 1U) >
          kLargestExactlyRepresentableInteger)
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC raster dimension cannot be represented exactly in binary64");
  if (!finite_value(x_min_) || !finite_value(x_max_) ||
      !finite_value(y_min_) || !finite_value(y_max_) ||
      !finite_less(x_min_, x_max_) || !finite_less(y_min_, y_max_))
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC bounds must be finite and strictly ordered");

  node_count_ = checked_multiply(n_x_, n_y_, "physical CIC node count");
  const double width_x = x_max_ - x_min_;
  const double width_y = y_max_ - y_min_;
  if (!finite_value(width_x) || !finite_value(width_y))
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC bound width is not finite");
  delta_x_ = width_x / static_cast<double>(n_x_ - 1U);
  delta_y_ = width_y / static_cast<double>(n_y_ - 1U);
  control_area_ = delta_x_ * delta_y_;
  if (!finite_value(delta_x_) || !finite_value(delta_y_) ||
      !strictly_positive_finite(delta_x_) ||
      !strictly_positive_finite(delta_y_))
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC spacing is not finite and positive");
  if (!strictly_positive_finite(control_area_))
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC control-cell area is not finite and positive");
  const double inverse_area = 1.0 / control_area_;
  if (!strictly_positive_finite(inverse_area))
    fail(SpatialErrorCode::InvalidGeometry,
         "physical CIC inverse control-cell area is not finite and positive");
  if (!finite_less(x_min_, x_min_ + delta_x_) ||
      !finite_less(x_max_ - delta_x_, x_max_) ||
      !finite_less(y_min_, y_min_ + delta_y_) ||
      !finite_less(y_max_ - delta_y_, y_max_))
    fail(SpatialErrorCode::InvalidGeometry,
         "adjacent physical CIC nodes are not distinct in binary64");

  StableHash hash;
  hash.add_string("FeeBLPhysicalCicGeometry");
  hash.add_u32(kPhysicalCicKernelVersion);
  hash.add_size(n_x_);
  hash.add_size(n_y_);
  hash.add_double(x_min_);
  hash.add_double(x_max_);
  hash.add_double(y_min_);
  hash.add_double(y_max_);
  hash.add_double(delta_x_);
  hash.add_double(delta_y_);
  hash.add_double(control_area_);
  geometry_hash_ = hash.value();
}

std::size_t PhysicalCicGeometry::linear_node(std::size_t x,
                                             std::size_t y) const {
  if (x >= n_x_ || y >= n_y_)
    fail(SpatialErrorCode::InternalInvariant,
         "physical CIC node lies outside the configured geometry");
  const std::size_t row = checked_multiply(y, n_x_, "physical CIC node index");
  if (x > std::numeric_limits<std::size_t>::max() - row)
    fail(SpatialErrorCode::ArithmeticOverflow,
         "physical CIC linear node index overflows size_t");
  return row + x;
}

double RootStencil::fraction_sum() const noexcept {
  return ((nodes[0].fraction + nodes[1].fraction) + nodes[2].fraction) +
         nodes[3].fraction;
}

RejectionReason operator|(RejectionReason left,
                          RejectionReason right) noexcept {
  return static_cast<RejectionReason>(static_cast<std::uint8_t>(left) |
                                      static_cast<std::uint8_t>(right));
}

bool has_reason(RejectionReason value, RejectionReason reason) noexcept {
  return (static_cast<std::uint8_t>(value) &
          static_cast<std::uint8_t>(reason)) != 0U;
}

AdmissionAudit audit_admission(const PhysicalCicGeometry& geometry,
                               const std::vector<PhysicalRoot>& roots) {
  std::vector<PhysicalRoot> canonical(roots);
  for (PhysicalRoot& root : canonical) {
    // Signed zero is likewise not a distinct root position or plan identity.
    root.x = canonical_zero(root.x);
    root.y = canonical_zero(root.y);
    validate_root(root);
  }
  std::sort(canonical.begin(), canonical.end(),
            [](const PhysicalRoot& left, const PhysicalRoot& right) {
              return left.id < right.id;
            });
  for (std::size_t index = 1U; index < canonical.size(); ++index) {
    if (canonical[index - 1U].id == canonical[index].id) {
      std::ostringstream message;
      message << "physical beam contains duplicate root ID "
              << canonical[index].id;
      fail(SpatialErrorCode::DuplicateRootId, message.str());
    }
  }

  AdmissionAudit result;
  result.accounting.input_roots = canonical.size();
  result.accepted.reserve(canonical.size());
  PositiveAccumulator input_population;
  PositiveAccumulator accepted_population;
  PositiveAccumulator rejected_population;
  for (const PhysicalRoot& root : canonical) {
    input_population.add(static_cast<long double>(root.population),
                         "input root population");
    const RejectionReason reason = rejection_reason(geometry, root);
    if (reason == RejectionReason::None) {
      result.accepted.push_back(make_stencil(geometry, root));
      accepted_population.add(static_cast<long double>(root.population),
                              "accepted root population");
    } else {
      result.rejected.push_back({root.id, root.population, reason});
      rejected_population.add(static_cast<long double>(root.population),
                              "rejected root population");
    }
  }
  result.accounting.accepted_roots = result.accepted.size();
  result.accounting.rejected_roots = result.rejected.size();
  result.accounting.input_population = checked_nonnegative_double(
      input_population.value("input root population"),
      "input root population");
  result.accounting.accepted_population = checked_nonnegative_double(
      accepted_population.value("accepted root population"),
      "accepted root population");
  result.accounting.rejected_population = checked_nonnegative_double(
      rejected_population.value("rejected root population"),
      "rejected root population");
  if (result.accounting.accepted_roots + result.accounting.rejected_roots !=
      result.accounting.input_roots)
    fail(SpatialErrorCode::InternalInvariant,
         "physical root admission count does not close");
  return result;
}

bool CollisionPair::operator==(const CollisionPair& other) const noexcept {
  return beam1_id == other.beam1_id && beam2_id == other.beam2_id &&
         stencil_overlap == other.stencil_overlap &&
         spatial_kernel == other.spatial_kernel && weight == other.weight;
}

DensityDotOracle independently_deposited_density_dot(
    const PhysicalCicGeometry& geometry,
    const std::vector<RootStencil>& beam1_input,
    const std::vector<RootStencil>& beam2_input,
    double normalization) {
  if (!strictly_positive_finite(normalization))
    fail(SpatialErrorCode::InvalidGeometry,
         "collision normalization must be finite and positive");
  std::vector<RootStencil> beam1(beam1_input);
  std::vector<RootStencil> beam2(beam2_input);
  validate_stencils(geometry, &beam1, "beam 1");
  validate_stencils(geometry, &beam2, "beam 2");
  ensure_disjoint_ids(beam1, beam2);

  const NodeDensity density1 = deposit_density(geometry, beam1, "beam 1");
  const NodeDensity density2 = deposit_density(geometry, beam2, "beam 2");
  DensityDotOracle result;
  result.beam1_nonzero_nodes = density1.size();
  result.beam2_nonzero_nodes = density2.size();
  PositiveAccumulator dot;
  auto left = density1.begin();
  auto right = density2.begin();
  while (left != density1.end() && right != density2.end()) {
    if (left->first < right->first) {
      ++left;
    } else if (right->first < left->first) {
      ++right;
    } else {
      const long double product = left->second * right->second;
      if (!finite_value(product))
        fail(SpatialErrorCode::ArithmeticOverflow,
             "density-dot node product overflows long double");
      dot.add(product, "independent density dot");
      ++result.common_nonzero_nodes;
      ++left;
      ++right;
    }
  }
  const long double unscaled = dot.value("independent density dot");
  result.unscaled_node_dot =
      checked_nonnegative_double(unscaled, "independent density dot");
  long double weighted = unscaled;
  weighted /= static_cast<long double>(geometry.control_area());
  weighted *= static_cast<long double>(normalization);
  result.weight =
      checked_nonnegative_double(weighted, "scaled independent density dot");
  return result;
}

ClosureReport evaluate_closure(double pair_weight_sum,
                               double density_dot_weight,
                               double relative_tolerance) {
  if (!finite_value(pair_weight_sum) || negative_finite(pair_weight_sum) ||
      !finite_value(density_dot_weight) || negative_finite(density_dot_weight))
    fail(SpatialErrorCode::ClosureFailure,
         "closure endpoints must be finite and non-negative");
  if (!finite_value(relative_tolerance) || negative_finite(relative_tolerance))
    fail(SpatialErrorCode::ClosureFailure,
         "closure tolerance must be finite and non-negative");
  ClosureReport result;
  result.pair_weight_sum = pair_weight_sum;
  result.density_dot_weight = density_dot_weight;
  result.relative_tolerance = relative_tolerance;
  const long double difference =
      std::fabs(static_cast<long double>(pair_weight_sum) -
                static_cast<long double>(density_dot_weight));
  const long double scale =
      std::max(static_cast<long double>(pair_weight_sum),
               static_cast<long double>(density_dot_weight));
  result.absolute_error =
      checked_nonnegative_double(difference, "collision closure error");
  const long double relative = scale == 0.0L ? 0.0L : difference / scale;
  result.relative_error =
      checked_nonnegative_double(relative, "relative collision closure error");
  result.within_tolerance = result.relative_error <= relative_tolerance;
  return result;
}

CollisionPlan CollisionPlan::build(
    const PhysicalCicGeometry& geometry,
    const std::vector<PhysicalRoot>& beam1,
    const std::vector<PhysicalRoot>& beam2,
    const CollisionPlanOptions& options) {
  if (!strictly_positive_finite(options.normalization))
    fail(SpatialErrorCode::InvalidGeometry,
         "collision normalization must be finite and positive");
  if (!finite_value(options.closure_relative_tolerance) ||
      negative_finite(options.closure_relative_tolerance))
    fail(SpatialErrorCode::ClosureFailure,
         "collision closure tolerance must be finite and non-negative");

  AdmissionAudit audit1 = audit_admission(geometry, beam1);
  AdmissionAudit audit2 = audit_admission(geometry, beam2);
  if (!audit1.rejected.empty() || !audit2.rejected.empty()) {
    std::ostringstream message;
    message << "physical CIC full-support admission rejected "
            << audit1.rejected.size() << " beam-1 and "
            << audit2.rejected.size()
            << " beam-2 roots; physics plan construction is fail-closed";
    fail(SpatialErrorCode::OutOfDomainRoot, message.str());
  }
  ensure_disjoint_ids(audit1.accepted, audit2.accepted);

  std::map<BaseCell, std::vector<std::size_t>> beam2_buckets;
  for (std::size_t index = 0U; index < audit2.accepted.size(); ++index) {
    const RootStencil& root = audit2.accepted[index];
    beam2_buckets[{root.base_x, root.base_y}].push_back(index);
  }

  std::vector<CollisionPair> pairs;
  PositiveAccumulator pair_weight_sum;
  for (const RootStencil& root1 : audit1.accepted) {
    const std::size_t min_x = root1.base_x == 0U ? 0U : root1.base_x - 1U;
    const std::size_t max_x =
        root1.base_x < geometry.n_x() - 2U ? root1.base_x + 1U
                                           : root1.base_x;
    const std::size_t min_y = root1.base_y == 0U ? 0U : root1.base_y - 1U;
    const std::size_t max_y =
        root1.base_y < geometry.n_y() - 2U ? root1.base_y + 1U
                                           : root1.base_y;
    std::vector<std::size_t> candidates;
    for (std::size_t y = min_y;; ++y) {
      for (std::size_t x = min_x;; ++x) {
        const auto bucket = beam2_buckets.find({x, y});
        if (bucket != beam2_buckets.end())
          candidates.insert(candidates.end(), bucket->second.begin(),
                            bucket->second.end());
        if (x == max_x) break;
      }
      if (y == max_y) break;
    }
    std::sort(candidates.begin(), candidates.end(),
              [&audit2](std::size_t left, std::size_t right) {
                return audit2.accepted[left].root.id <
                       audit2.accepted[right].root.id;
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    for (std::size_t candidate : candidates) {
      const RootStencil& root2 = audit2.accepted[candidate];
      const long double overlap = stencil_overlap(root1, root2);
      if (overlap == 0.0L) continue;
      if (pairs.size() >= options.max_pairs)
        fail(SpatialErrorCode::ResourceLimitExceeded,
             "physical collision pair count exceeds configured maximum");
      const double overlap_double =
          checked_nonnegative_double(overlap, "CIC stencil overlap");
      const long double kernel =
          overlap / static_cast<long double>(geometry.control_area());
      const double kernel_double =
          checked_nonnegative_double(kernel, "physical CIC spatial kernel");
      const double weight = scaled_pair_weight(
          overlap, geometry, root1.root, root2.root, options.normalization);
      pairs.push_back({root1.root.id, root2.root.id, overlap_double,
                       kernel_double, weight});
      pair_weight_sum.add(static_cast<long double>(weight),
                          "physical collision pair-weight sum");
    }
  }

  for (std::size_t index = 1U; index < pairs.size(); ++index) {
    const CollisionPair& previous = pairs[index - 1U];
    const CollisionPair& current = pairs[index];
    if (current.beam1_id < previous.beam1_id ||
        (current.beam1_id == previous.beam1_id &&
         current.beam2_id <= previous.beam2_id))
      fail(SpatialErrorCode::InternalInvariant,
           "physical collision pairs are not unique canonical root-ID pairs");
  }

  const DensityDotOracle oracle = independently_deposited_density_dot(
      geometry, audit1.accepted, audit2.accepted, options.normalization);
  const double pair_sum = checked_nonnegative_double(
      pair_weight_sum.value("physical collision pair-weight sum"),
      "physical collision pair-weight sum");
  const ClosureReport closure = evaluate_closure(
      pair_sum, oracle.weight, options.closure_relative_tolerance);
  if (!closure.within_tolerance) {
    std::ostringstream message;
    message << "physical pair sum/density-dot closure failed: pair="
            << closure.pair_weight_sum << " oracle="
            << closure.density_dot_weight << " relative_error="
            << closure.relative_error << " tolerance="
            << closure.relative_tolerance;
    fail(SpatialErrorCode::ClosureFailure, message.str());
  }

  const std::uint64_t plan_hash = make_plan_hash(
      geometry, audit1.accepted, audit2.accepted, pairs,
      options.normalization);
  return CollisionPlan(geometry, std::move(audit1.accepted),
                       std::move(audit2.accepted), audit1.accounting,
                       audit2.accounting, std::move(pairs), oracle, closure,
                       options.normalization, plan_hash);
}

CollisionPlan::CollisionPlan(
    const PhysicalCicGeometry& geometry,
    std::vector<RootStencil> beam1_stencils,
    std::vector<RootStencil> beam2_stencils,
    const BeamAdmissionAccounting& beam1_accounting,
    const BeamAdmissionAccounting& beam2_accounting,
    std::vector<CollisionPair> pairs,
    const DensityDotOracle& density_dot_oracle,
    const ClosureReport& closure,
    double normalization,
    std::uint64_t plan_hash)
    : geometry_(geometry),
      beam1_stencils_(std::move(beam1_stencils)),
      beam2_stencils_(std::move(beam2_stencils)),
      beam1_accounting_(beam1_accounting),
      beam2_accounting_(beam2_accounting),
      pairs_(std::move(pairs)),
      density_dot_oracle_(density_dot_oracle),
      closure_(closure),
      normalization_(normalization),
      plan_hash_(plan_hash) {}

}  // namespace electron_lumi_spatial
}  // namespace feebl
