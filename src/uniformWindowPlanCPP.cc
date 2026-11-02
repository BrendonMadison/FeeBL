#include "uniformWindowPlanCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <climits>
#include <iterator>
#include <sstream>
#include <utility>

namespace feebl {
namespace adaptive_grid {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4U,
              "canonical mapping hashes require IEEE-754 binary32");
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8U,
              "canonical mapping hashes require IEEE-754 binary64");

constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

class StableHash {
 public:
  StableHash() noexcept : value_(kFnvOffset) {}

  void add_byte(std::uint8_t value) noexcept {
    value_ ^= value;
    value_ *= kFnvPrime;
  }

  void add_bool(bool value) noexcept { add_byte(value ? 1U : 0U); }

  void add_u32(std::uint32_t value) noexcept {
    for (unsigned int shift = 0; shift < 32U; shift += 8U)
      add_byte(static_cast<std::uint8_t>((value >> shift) & UINT32_C(0xff)));
  }

  void add_u64(std::uint64_t value) noexcept {
    for (unsigned int shift = 0; shift < 64U; shift += 8U)
      add_byte(static_cast<std::uint8_t>((value >> shift) & UINT64_C(0xff)));
  }

  void add_index(CanonicalIndex value) noexcept {
    add_u64(static_cast<std::uint64_t>(value));
  }

  void add_size(std::size_t value) noexcept {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "plan hashes require size_t no wider than uint64_t");
    add_u64(static_cast<std::uint64_t>(value));
  }

  void add_float(float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    add_u32(bits);
  }

  void add_double(double value) noexcept {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
    std::memcpy(&bits, &value, sizeof(bits));
    add_u64(bits);
  }

  void add_string(const char* value) noexcept {
    std::size_t length = 0;
    while (value[length] != '\0') ++length;
    add_size(length);
    for (std::size_t i = 0; i < length; ++i)
      add_byte(static_cast<std::uint8_t>(value[i]));
  }

  std::uint64_t value() const noexcept { return value_; }

 private:
  std::uint64_t value_;
};

[[noreturn]] void fail(WindowPlanErrorCode code, const std::string& message) {
  throw WindowPlanError(code, message);
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* quantity) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    std::ostringstream message;
    message << quantity << " addition overflows size_t";
    fail(WindowPlanErrorCode::ArithmeticOverflow, message.str());
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* quantity) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    std::ostringstream message;
    message << quantity << " multiplication overflows size_t";
    fail(WindowPlanErrorCode::ArithmeticOverflow, message.str());
  }
  return left * right;
}

CanonicalIndex checked_index_add(CanonicalIndex value, CanonicalIndex delta,
                                 const char* quantity) {
  if ((delta > 0 && value > std::numeric_limits<CanonicalIndex>::max() - delta) ||
      (delta < 0 && value < std::numeric_limits<CanonicalIndex>::min() - delta)) {
    std::ostringstream message;
    message << quantity << " index arithmetic overflows";
    fail(WindowPlanErrorCode::InvalidDescriptor, message.str());
  }
  return value + delta;
}

// std::isfinite may be folded to true when this translation unit inherits the
// historical project-wide -ffast-math flags.  Classification by IEEE exponent
// bits preserves the input-card fail-closed contract under either flag set.
bool finite_value(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

bool finite_value(double value) noexcept {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

bool unit_fraction(float value) noexcept {
  return finite_value(value) && value >= 0.0f && value < 1.0f;
}

bool unit_fraction(double value) noexcept {
  return finite_value(value) && value >= 0.0 && value < 1.0;
}

void validate_node(const CanonicalNode& node,
                   const CanonicalLatticeSnapshot& lattice,
                   const char* role) {
  if (node.x < 0 || node.y < 0 ||
      static_cast<std::uint64_t>(node.x) >= lattice.n_x() ||
      static_cast<std::uint64_t>(node.y) >= lattice.n_y()) {
    std::ostringstream message;
    message << role << " node (" << node.x << ',' << node.y
            << ") lies outside canonical " << lattice.n_x() << 'x'
            << lattice.n_y() << " storage";
    fail(WindowPlanErrorCode::OutOfCanonicalBank, message.str());
  }
}

void sort_unique(std::vector<CanonicalNode>& nodes) {
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
}

CanonicalNodeBox make_box(const std::vector<CanonicalNode>& nodes) noexcept {
  CanonicalNodeBox box;
  if (nodes.empty()) return box;
  box.valid = true;
  box.min_x = box.max_x = nodes.front().x;
  box.min_y = box.max_y = nodes.front().y;
  for (const CanonicalNode& node : nodes) {
    box.min_x = std::min(box.min_x, node.x);
    box.max_x = std::max(box.max_x, node.x);
    box.min_y = std::min(box.min_y, node.y);
    box.max_y = std::max(box.max_y, node.y);
  }
  return box;
}

std::vector<std::size_t> normalize_class_bank(
    const std::vector<std::size_t>& classes, const char* axis) {
  if (classes.empty()) {
    std::ostringstream message;
    message << axis << " size-class bank is empty";
    fail(WindowPlanErrorCode::InvalidClassBank, message.str());
  }
  std::vector<std::size_t> normalized(classes);
  for (std::size_t value : normalized) {
    if (value < 4U || value > static_cast<std::size_t>(INT_MAX)) {
      std::ostringstream message;
      message << axis << " size class " << value
              << " is outside the supported [4,INT_MAX] range";
      fail(WindowPlanErrorCode::InvalidClassBank, message.str());
    }
  }
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  return normalized;
}

struct NormalizedPolicy {
  std::vector<std::size_t> classes_x;
  std::vector<std::size_t> classes_y;
  std::vector<std::size_t> eligible_x;
  std::vector<std::size_t> eligible_y;
  std::size_t min_n_x;
  std::size_t min_n_y;
  std::size_t max_n_x;
  std::size_t max_n_y;
  std::size_t guard;
  bool allow_full_window;
  std::size_t bytes_per_local_cell;
  std::size_t bytes_per_padded_cell;
  std::size_t fixed_bytes;
  std::size_t max_bytes;
};

NormalizedPolicy normalize_policy(const CanonicalLatticeSnapshot& lattice,
                                  const UniformWindowPolicy& policy) {
  NormalizedPolicy normalized;
  normalized.classes_x = normalize_class_bank(policy.size_classes_x, "x");
  normalized.classes_y = normalize_class_bank(policy.size_classes_y, "y");
  normalized.min_n_x = policy.min_n_x;
  normalized.min_n_y = policy.min_n_y;
  normalized.max_n_x = policy.max_n_x == 0U ? lattice.n_x() : policy.max_n_x;
  normalized.max_n_y = policy.max_n_y == 0U ? lattice.n_y() : policy.max_n_y;
  normalized.allow_full_window = policy.allow_full_window;
  normalized.bytes_per_local_cell = policy.bytes_per_local_cell;
  normalized.bytes_per_padded_cell = policy.bytes_per_padded_cell;
  normalized.fixed_bytes = policy.fixed_bytes;
  normalized.max_bytes = policy.max_bytes;

  if (normalized.min_n_x < 4U || normalized.min_n_y < 4U ||
      normalized.max_n_x > lattice.n_x() ||
      normalized.max_n_y > lattice.n_y() ||
      normalized.min_n_x > normalized.max_n_x ||
      normalized.min_n_y > normalized.max_n_y) {
    fail(WindowPlanErrorCode::InvalidPolicy,
         "window min/max dimensions are inconsistent with the canonical bank");
  }
  if (policy.extra_guard_cells < 0) {
    fail(WindowPlanErrorCode::InvalidPolicy,
         "adaptive extra guard cells must be non-negative");
  }
  if (static_cast<std::uint64_t>(policy.extra_guard_cells) >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    fail(WindowPlanErrorCode::ArithmeticOverflow,
         "adaptive extra guard cells do not fit size_t");
  }
  normalized.guard = static_cast<std::size_t>(policy.extra_guard_cells);

  const auto eligible = [](const std::vector<std::size_t>& values,
                           std::size_t minimum, std::size_t maximum) {
    std::vector<std::size_t> result;
    for (std::size_t value : values)
      if (value >= minimum && value <= maximum) result.push_back(value);
    return result;
  };
  normalized.eligible_x = eligible(normalized.classes_x,
                                   normalized.min_n_x, normalized.max_n_x);
  normalized.eligible_y = eligible(normalized.classes_y,
                                   normalized.min_n_y, normalized.max_n_y);
  if (normalized.eligible_x.empty() || normalized.eligible_y.empty()) {
    fail(WindowPlanErrorCode::InvalidClassBank,
         "no size class remains inside the configured canonical min/max bank");
  }
  return normalized;
}

std::size_t choose_class(const std::vector<std::size_t>& classes,
                         std::size_t required, const char* axis) {
  const auto selected = std::lower_bound(classes.begin(), classes.end(), required);
  if (selected == classes.end()) {
    std::ostringstream message;
    message << "no " << axis << " size class contains " << required
            << " required canonical nodes";
    fail(WindowPlanErrorCode::NoFittingClass, message.str());
  }
  return *selected;
}

void hash_node(StableHash& hash, const CanonicalNode& node) noexcept {
  hash.add_index(node.x);
  hash.add_index(node.y);
}

void hash_box(StableHash& hash, const CanonicalNodeBox& box) noexcept {
  hash.add_bool(box.valid);
  hash.add_index(box.min_x);
  hash.add_index(box.max_x);
  hash.add_index(box.min_y);
  hash.add_index(box.max_y);
}

void hash_deposits(StableHash& hash,
                   const std::vector<CanonicalDepositDescriptor>& values) {
  hash.add_size(values.size());
  for (const CanonicalDepositDescriptor& value : values) {
    hash.add_bool(value.accepted());
    hash.add_index(value.base_x());
    hash.add_index(value.base_y());
    hash.add_float(value.weight_x());
    hash.add_float(value.weight_y());
  }
}

void hash_gathers(StableHash& hash,
                  const std::vector<CanonicalGatherDescriptor>& values) {
  hash.add_size(values.size());
  for (const CanonicalGatherDescriptor& value : values) {
    hash.add_bool(value.accepted());
    hash.add_index(value.y_base_x());
    hash.add_index(value.y_base_y());
    hash.add_float(value.y_weight());
    hash.add_index(value.x_base_x());
    hash.add_index(value.x_base_y());
    hash.add_float(value.x_weight());
    hash.add_double(value.fractional_x());
    hash.add_double(value.fractional_y());
  }
}

template <class Descriptor>
void append_nodes(const std::vector<Descriptor>& descriptors,
                  const CanonicalLatticeSnapshot& lattice,
                  const char* role,
                  std::vector<CanonicalNode>& output,
                  std::size_t& accepted_count) {
  for (const Descriptor& descriptor : descriptors) {
    if (!descriptor.accepted()) continue;
    accepted_count = checked_add(accepted_count, 1U,
                                 "accepted descriptor count");
    for (const CanonicalNode& node : descriptor.nodes()) {
      validate_node(node, lattice, role);
      output.push_back(node);
    }
  }
}

void hash_nodes(StableHash& hash,
                const std::vector<CanonicalNode>& nodes) noexcept {
  hash.add_size(nodes.size());
  for (const CanonicalNode& node : nodes) hash_node(hash, node);
}

}  // namespace

std::size_t CanonicalNodeBox::width() const noexcept {
  if (!valid) return 0U;
  return static_cast<std::size_t>(max_x - min_x) + 1U;
}

std::size_t CanonicalNodeBox::height() const noexcept {
  if (!valid) return 0U;
  return static_cast<std::size_t>(max_y - min_y) + 1U;
}

bool CanonicalNodeBox::contains(const CanonicalNode& node) const noexcept {
  return valid && node.x >= min_x && node.x <= max_x &&
         node.y >= min_y && node.y <= max_y;
}

WindowPlanError::WindowPlanError(WindowPlanErrorCode code,
                                 const std::string& message)
    : std::runtime_error(message), code_(code) {}

CanonicalLatticeSnapshot::CanonicalLatticeSnapshot(
    const CanonicalLatticeInput& input)
    : n_x_(input.n_x), n_y_(input.n_y),
      delta_x_(input.delta_x), delta_y_(input.delta_y),
      delta_z_(input.delta_z), inverse_delta_x_(input.inverse_delta_x),
      inverse_delta_y_(input.inverse_delta_y), offset_x_(input.offset_x),
      offset_y_(input.offset_y), intended_min_x_(input.intended_min_x),
      intended_max_x_(input.intended_max_x),
      intended_min_y_(input.intended_min_y),
      intended_max_y_(input.intended_max_y),
      charge_sign_(input.charge_sign),
      field_kernel_factor_(input.field_kernel_factor),
      geometry_revision_(input.geometry_revision), hash_(0) {
  if (n_x_ < 4U || n_y_ < 4U ||
      n_x_ > static_cast<std::size_t>(INT_MAX) ||
      n_y_ > static_cast<std::size_t>(INT_MAX)) {
    fail(WindowPlanErrorCode::InvalidLattice,
         "canonical lattice dimensions must be in [4,INT_MAX]");
  }
  (void)checked_multiply(n_x_, n_y_, "canonical cell count");
  if (!finite_value(delta_x_) || !finite_value(delta_y_) ||
      !finite_value(delta_z_) || delta_x_ <= 0.0f || delta_y_ <= 0.0f ||
      delta_z_ <= 0.0f || !finite_value(inverse_delta_x_) ||
      !finite_value(inverse_delta_y_) || inverse_delta_x_ <= 0.0f ||
      inverse_delta_y_ <= 0.0f || !finite_value(offset_x_) ||
      !finite_value(offset_y_) || !finite_value(intended_min_x_) ||
      !finite_value(intended_max_x_) || !finite_value(intended_min_y_) ||
      !finite_value(intended_max_y_) ||
      !(intended_min_x_ < intended_max_x_) ||
      !(intended_min_y_ < intended_max_y_) ||
      !finite_value(charge_sign_) ||
      !finite_value(field_kernel_factor_)) {
    fail(WindowPlanErrorCode::InvalidLattice,
         "canonical lattice contains invalid or non-finite geometry");
  }

  StableHash hash;
  hash.add_string("FeeBL/CanonicalLatticeSnapshot/v1");
  hash.add_size(n_x_);
  hash.add_size(n_y_);
  hash.add_float(delta_x_);
  hash.add_float(delta_y_);
  hash.add_float(delta_z_);
  hash.add_float(inverse_delta_x_);
  hash.add_float(inverse_delta_y_);
  hash.add_float(offset_x_);
  hash.add_float(offset_y_);
  hash.add_float(intended_min_x_);
  hash.add_float(intended_max_x_);
  hash.add_float(intended_min_y_);
  hash.add_float(intended_max_y_);
  hash.add_float(charge_sign_);
  hash.add_double(field_kernel_factor_);
  hash.add_u64(geometry_revision_);
  hash_ = hash.value();
}

CanonicalDepositDescriptor::CanonicalDepositDescriptor() noexcept
    : accepted_(false), base_x_(0), base_y_(0), weight_x_(0.0f),
      weight_y_(0.0f), nodes_{} {}

CanonicalDepositDescriptor CanonicalDepositDescriptor::rejected() noexcept {
  return CanonicalDepositDescriptor();
}

CanonicalDepositDescriptor CanonicalDepositDescriptor::from_canonical(
    CanonicalIndex base_x, CanonicalIndex base_y,
    float weight_x, float weight_y) {
  if (!unit_fraction(weight_x) || !unit_fraction(weight_y)) {
    fail(WindowPlanErrorCode::InvalidDescriptor,
         "canonical CIC weights must be finite members of [0,1)");
  }
  const CanonicalIndex next_x =
      checked_index_add(base_x, 1, "canonical CIC x");
  const CanonicalIndex next_y =
      checked_index_add(base_y, 1, "canonical CIC y");
  CanonicalDepositDescriptor result;
  result.accepted_ = true;
  result.base_x_ = base_x;
  result.base_y_ = base_y;
  result.weight_x_ = weight_x;
  result.weight_y_ = weight_y;
  result.nodes_ = {{{base_x, base_y},
                    {next_x, base_y},
                    {base_x, next_y},
                    {next_x, next_y}}};
  return result;
}

CanonicalGatherDescriptor::CanonicalGatherDescriptor() noexcept
    : accepted_(false), y_base_x_(0), y_base_y_(0), y_weight_(0.0f),
      x_base_x_(0), x_base_y_(0), x_weight_(0.0f), fractional_x_(0.0),
      fractional_y_(0.0), nodes_{} {}

CanonicalGatherDescriptor CanonicalGatherDescriptor::rejected() noexcept {
  return CanonicalGatherDescriptor();
}

CanonicalGatherDescriptor CanonicalGatherDescriptor::from_canonical(
    CanonicalIndex y_base_x, CanonicalIndex y_base_y, float y_weight,
    CanonicalIndex x_base_x, CanonicalIndex x_base_y, float x_weight,
    double fractional_x, double fractional_y) {
  if (!unit_fraction(y_weight) || !unit_fraction(x_weight) ||
      !unit_fraction(fractional_x) || !unit_fraction(fractional_y)) {
    fail(WindowPlanErrorCode::InvalidDescriptor,
         "canonical gather weights/fractions must be finite members of [0,1)");
  }
  const CanonicalIndex y_plus_x =
      checked_index_add(y_base_x, 1, "canonical gather Y x+1");
  const CanonicalIndex y_minus_x =
      checked_index_add(y_base_x, -1, "canonical gather Y x-1");
  const CanonicalIndex y_plus_y =
      checked_index_add(y_base_y, 1, "canonical gather Y y+1");
  const CanonicalIndex x_plus_x =
      checked_index_add(x_base_x, 1, "canonical gather X x+1");
  const CanonicalIndex x_plus_y =
      checked_index_add(x_base_y, 1, "canonical gather X y+1");
  const CanonicalIndex x_minus_y =
      checked_index_add(x_base_y, -1, "canonical gather X y-1");

  CanonicalGatherDescriptor result;
  result.accepted_ = true;
  result.y_base_x_ = y_base_x;
  result.y_base_y_ = y_base_y;
  result.y_weight_ = y_weight;
  result.x_base_x_ = x_base_x;
  result.x_base_y_ = x_base_y;
  result.x_weight_ = x_weight;
  result.fractional_x_ = fractional_x;
  result.fractional_y_ = fractional_y;
  result.nodes_ = {{{y_plus_x, y_plus_y},
                    {y_plus_x, y_base_y},
                    {y_base_x, y_plus_y},
                    {y_base_x, y_base_y},
                    {y_minus_x, y_plus_y},
                    {y_minus_x, y_base_y},
                    {x_plus_x, x_plus_y},
                    {x_base_x, x_plus_y},
                    {x_plus_x, x_base_y},
                    {x_base_x, x_base_y},
                    {x_plus_x, x_minus_y},
                    {x_base_x, x_minus_y}}};
  return result;
}

CanonicalGatherDescriptor
CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
    double absolute_x, double absolute_y) {
  if (!finite_value(absolute_x) || !finite_value(absolute_y)) {
    fail(WindowPlanErrorCode::InvalidDescriptor,
         "legacy canonical absolute coordinates must be finite");
  }
  // interpolePotential stores absoluteCoordinates in PHI_FLOAT but passes the
  // values to cellLocalization with a local float weight.  Overload resolution
  // therefore selects the float overload for both Y and swapped-X calls.
  const float grid_x = static_cast<float>(absolute_x);
  const float grid_y = static_cast<float>(absolute_y);
  if (!finite_value(grid_x) || !finite_value(grid_y)) {
    fail(WindowPlanErrorCode::InvalidDescriptor,
         "legacy float-narrowed absolute coordinates must be finite");
  }
  const auto floor_index = [](float value, const char* role) {
    const long double floored = std::floor(static_cast<long double>(value));
    if (floored < static_cast<long double>(
                      std::numeric_limits<CanonicalIndex>::min()) ||
        floored > static_cast<long double>(
                      std::numeric_limits<CanonicalIndex>::max())) {
      std::ostringstream message;
      message << role << " floor does not fit a canonical index";
      fail(WindowPlanErrorCode::InvalidDescriptor, message.str());
    }
    return static_cast<CanonicalIndex>(floored);
  };

  const CanonicalIndex y_base_x = floor_index(grid_x, "legacy gather Y x");
  float shifted_y = grid_y - 0.5f;
  const CanonicalIndex y_base_y =
      floor_index(shifted_y, "legacy gather Y y");
  const float y_weight = shifted_y - static_cast<float>(y_base_y);

  const CanonicalIndex x_base_y = floor_index(grid_y, "legacy gather X y");
  float shifted_x = grid_x - 0.5f;
  const CanonicalIndex x_base_x =
      floor_index(shifted_x, "legacy gather X x");
  const float x_weight = shifted_x - static_cast<float>(x_base_x);

  const double fractional_x = absolute_x - std::floor(absolute_x);
  const double fractional_y = absolute_y - std::floor(absolute_y);
  return from_canonical(y_base_x, y_base_y, y_weight,
                        x_base_x, x_base_y, x_weight,
                        fractional_x, fractional_y);
}

std::vector<std::size_t> parse_size_classes(const std::string& value) {
  std::vector<std::size_t> classes;
  std::size_t begin = 0U;
  while (begin <= value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::size_t end = comma == std::string::npos ? value.size() : comma;
    std::size_t first = begin;
    std::size_t last = end;
    while (first < last &&
           std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1U]))) --last;
    if (first == last) {
      fail(WindowPlanErrorCode::InvalidClassBank,
           "size-class list contains an empty entry");
    }
    std::size_t parsed = 0U;
    for (std::size_t position = first; position < last; ++position) {
      const unsigned char character =
          static_cast<unsigned char>(value[position]);
      if (!std::isdigit(character)) {
        fail(WindowPlanErrorCode::InvalidClassBank,
             "size-class entries must contain decimal digits only");
      }
      const std::size_t digit = static_cast<std::size_t>(character - '0');
      if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
        fail(WindowPlanErrorCode::ArithmeticOverflow,
             "size-class decimal value overflows size_t");
      }
      parsed = parsed * 10U + digit;
    }
    if (parsed < 4U || parsed > static_cast<std::size_t>(INT_MAX)) {
      fail(WindowPlanErrorCode::InvalidClassBank,
           "size classes must lie in the supported [4,INT_MAX] range");
    }
    classes.push_back(parsed);
    if (comma == std::string::npos) break;
    begin = comma + 1U;
  }
  std::sort(classes.begin(), classes.end());
  classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
  return classes;
}

UniformWindowPlan UniformWindowPlan::build(
    const CanonicalLatticeSnapshot& lattice,
    CanonicalWindowInputs inputs,
    const UniformWindowPolicy& policy) {
  const NormalizedPolicy normalized = normalize_policy(lattice, policy);
  UniformWindowPlan plan;
  plan.lattice_hash_ = lattice.hash();
  plan.inputs_ = std::move(inputs);

  const std::size_t source_descriptor_count = checked_add(
      plan.inputs_.beam1_sources.size(), plan.inputs_.beam2_sources.size(),
      "source descriptor count");
  const std::size_t gather_descriptor_count = checked_add(
      plan.inputs_.beam1_targets.size(), plan.inputs_.beam2_targets.size(),
      "gather descriptor count");
  plan.source_nodes_.reserve(checked_multiply(
      source_descriptor_count, 4U, "source-node reserve"));
  plan.gather_nodes_.reserve(checked_multiply(
      gather_descriptor_count, 12U, "gather-node reserve"));

  append_nodes(
      plan.inputs_.beam1_sources, lattice, "beam-1 source",
      plan.source_nodes_, plan.accepted_source_count_);
  append_nodes(
      plan.inputs_.beam2_sources, lattice, "beam-2 source",
      plan.source_nodes_, plan.accepted_source_count_);
  append_nodes(
      plan.inputs_.beam1_targets, lattice, "beam-1 target",
      plan.gather_nodes_, plan.accepted_target_count_);
  append_nodes(
      plan.inputs_.beam2_targets, lattice, "beam-2 target",
      plan.gather_nodes_, plan.accepted_target_count_);

  sort_unique(plan.source_nodes_);
  sort_unique(plan.gather_nodes_);
  plan.source_box_ = make_box(plan.source_nodes_);
  plan.gather_box_ = make_box(plan.gather_nodes_);
  plan.required_nodes_.reserve(checked_add(
      plan.source_nodes_.size(), plan.gather_nodes_.size(),
      "required-node reserve"));
  // The two inputs are already sorted and unique.  A linear union preserves
  // the exact canonical x-major/y-minor representation without paying for a
  // third comparison sort in every encounter.
  std::set_union(plan.source_nodes_.begin(), plan.source_nodes_.end(),
                 plan.gather_nodes_.begin(), plan.gather_nodes_.end(),
                 std::back_inserter(plan.required_nodes_));
  plan.required_node_count_ = plan.required_nodes_.size();
  plan.required_box_ = make_box(plan.required_nodes_);

  if (!plan.required_box_.valid) {
    plan.status_ = WindowPlanStatus::Empty;
  } else {
    plan.status_ = WindowPlanStatus::Ready;
    plan.guarded_box_ = plan.required_box_;
    const std::size_t required_min_x =
        static_cast<std::size_t>(plan.required_box_.min_x);
    const std::size_t required_max_x =
        static_cast<std::size_t>(plan.required_box_.max_x);
    const std::size_t required_min_y =
        static_cast<std::size_t>(plan.required_box_.min_y);
    const std::size_t required_max_y =
        static_cast<std::size_t>(plan.required_box_.max_y);
    const std::size_t high_distance_x =
        lattice.n_x() - 1U - required_max_x;
    const std::size_t high_distance_y =
        lattice.n_y() - 1U - required_max_y;

    plan.guard_clipped_low_x_ = normalized.guard > required_min_x;
    plan.guard_clipped_high_x_ = normalized.guard > high_distance_x;
    plan.guard_clipped_low_y_ = normalized.guard > required_min_y;
    plan.guard_clipped_high_y_ = normalized.guard > high_distance_y;
    plan.guarded_box_.min_x = static_cast<CanonicalIndex>(
        plan.guard_clipped_low_x_ ? 0U : required_min_x - normalized.guard);
    plan.guarded_box_.max_x = static_cast<CanonicalIndex>(
        plan.guard_clipped_high_x_
            ? lattice.n_x() - 1U
            : required_max_x + normalized.guard);
    plan.guarded_box_.min_y = static_cast<CanonicalIndex>(
        plan.guard_clipped_low_y_ ? 0U : required_min_y - normalized.guard);
    plan.guarded_box_.max_y = static_cast<CanonicalIndex>(
        plan.guard_clipped_high_y_
            ? lattice.n_y() - 1U
            : required_max_y + normalized.guard);

    plan.local_n_x_ = choose_class(normalized.eligible_x,
                                   plan.guarded_box_.width(), "x");
    plan.local_n_y_ = choose_class(normalized.eligible_y,
                                   plan.guarded_box_.height(), "y");
    plan.full_window_ = plan.local_n_x_ == lattice.n_x() &&
                        plan.local_n_y_ == lattice.n_y();
    if (plan.full_window_ && !normalized.allow_full_window) {
      fail(WindowPlanErrorCode::FullWindowForbidden,
           "the selected exact window is canonical-sized but full windows "
           "are disabled");
    }

    const auto place_origin = [](CanonicalIndex guarded_min,
                                 std::size_t guarded_width,
                                 std::size_t local_width,
                                 std::size_t canonical_width) {
      const std::size_t slack = local_width - guarded_width;
      // Version-1 convention: the odd slack cell belongs to the low side.
      const std::size_t low_padding = slack / 2U + slack % 2U;
      const std::size_t minimum = static_cast<std::size_t>(guarded_min);
      std::size_t origin = low_padding > minimum
                               ? 0U
                               : minimum - low_padding;
      const std::size_t maximum_origin = canonical_width - local_width;
      if (origin > maximum_origin) origin = maximum_origin;
      return origin;
    };
    const std::size_t origin_x = place_origin(
        plan.guarded_box_.min_x, plan.guarded_box_.width(),
        plan.local_n_x_, lattice.n_x());
    const std::size_t origin_y = place_origin(
        plan.guarded_box_.min_y, plan.guarded_box_.height(),
        plan.local_n_y_, lattice.n_y());
    plan.origin_x_ = static_cast<CanonicalIndex>(origin_x);
    plan.origin_y_ = static_cast<CanonicalIndex>(origin_y);

    const CanonicalNode window_high{
        static_cast<CanonicalIndex>(origin_x + plan.local_n_x_ - 1U),
        static_cast<CanonicalIndex>(origin_y + plan.local_n_y_ - 1U)};
    if (plan.origin_x_ > plan.guarded_box_.min_x ||
        plan.origin_y_ > plan.guarded_box_.min_y) {
      fail(WindowPlanErrorCode::NodeNotInWindow,
           "internal low-side window placement validation failed");
    }
    if (plan.guarded_box_.max_x > window_high.x ||
        plan.guarded_box_.max_y > window_high.y) {
      fail(WindowPlanErrorCode::NodeNotInWindow,
           "selected class does not contain the guarded canonical box");
    }
    for (const CanonicalNode& node : plan.required_nodes_)
      if (!plan.contains_canonical_node(node))
        fail(WindowPlanErrorCode::NodeNotInWindow,
             "selected class lost a required canonical node");

    plan.padded_n_x_ = checked_multiply(plan.local_n_x_, 2U,
                                        "doubled FFT x dimension");
    plan.padded_n_y_ = checked_multiply(plan.local_n_y_, 2U,
                                        "doubled FFT y dimension");
    const std::size_t local_cells = checked_multiply(
        plan.local_n_x_, plan.local_n_y_, "local window cell count");
    const std::size_t padded_cells = checked_multiply(
        plan.padded_n_x_, plan.padded_n_y_, "padded FFT cell count");
    const std::size_t local_bytes = checked_multiply(
        local_cells, normalized.bytes_per_local_cell,
        "local window byte estimate");
    const std::size_t padded_bytes = checked_multiply(
        padded_cells, normalized.bytes_per_padded_cell,
        "padded FFT byte estimate");
    plan.estimated_bytes_ = checked_add(
        checked_add(local_bytes, padded_bytes, "window byte estimate"),
        normalized.fixed_bytes, "window fixed-byte estimate");
    if (plan.estimated_bytes_ > normalized.max_bytes) {
      std::ostringstream message;
      message << "window estimate " << plan.estimated_bytes_
              << " exceeds configured budget " << normalized.max_bytes;
      fail(WindowPlanErrorCode::MemoryBudgetExceeded, message.str());
    }
  }

  StableHash hash;
  hash.add_string("FeeBL/UniformWindowPlan/v1");
  hash.add_u64(lattice.hash());
  hash.add_size(normalized.classes_x.size());
  for (std::size_t value : normalized.classes_x) hash.add_size(value);
  hash.add_size(normalized.classes_y.size());
  for (std::size_t value : normalized.classes_y) hash.add_size(value);
  hash.add_size(normalized.min_n_x);
  hash.add_size(normalized.min_n_y);
  hash.add_size(normalized.max_n_x);
  hash.add_size(normalized.max_n_y);
  hash.add_size(normalized.guard);
  hash.add_bool(normalized.allow_full_window);
  hash.add_size(normalized.bytes_per_local_cell);
  hash.add_size(normalized.bytes_per_padded_cell);
  hash.add_size(normalized.fixed_bytes);
  // `SIZE_MAX` means "unlimited within this process", not a literal policy
  // capacity.  Serialize that semantic sentinel as uint64 all-ones so a
  // 32-bit planning/test host and the 64-bit production host produce the same
  // schedule hash for the same controls.  Finite budgets retain their exact
  // numeric byte value.
  hash.add_u64(normalized.max_bytes ==
                       std::numeric_limits<std::size_t>::max()
                   ? std::numeric_limits<std::uint64_t>::max()
                   : static_cast<std::uint64_t>(normalized.max_bytes));
  hash.add_string("beam1-sources");
  hash_deposits(hash, plan.inputs_.beam1_sources);
  hash.add_string("beam2-sources");
  hash_deposits(hash, plan.inputs_.beam2_sources);
  hash.add_string("beam1-targets");
  hash_gathers(hash, plan.inputs_.beam1_targets);
  hash.add_string("beam2-targets");
  hash_gathers(hash, plan.inputs_.beam2_targets);
  hash.add_u32(static_cast<std::uint32_t>(plan.status_));
  hash.add_size(plan.local_n_x_);
  hash.add_size(plan.local_n_y_);
  hash.add_size(plan.padded_n_x_);
  hash.add_size(plan.padded_n_y_);
  hash.add_index(plan.origin_x_);
  hash.add_index(plan.origin_y_);
  hash.add_bool(plan.full_window_);
  hash.add_bool(plan.guard_clipped_low_x_);
  hash.add_bool(plan.guard_clipped_high_x_);
  hash.add_bool(plan.guard_clipped_low_y_);
  hash.add_bool(plan.guard_clipped_high_y_);
  hash.add_size(plan.estimated_bytes_);
  hash.add_size(plan.accepted_source_count_);
  hash.add_size(plan.accepted_target_count_);
  hash_nodes(hash, plan.source_nodes_);
  hash_nodes(hash, plan.gather_nodes_);
  hash_nodes(hash, plan.required_nodes_);
  hash_box(hash, plan.source_box_);
  hash_box(hash, plan.gather_box_);
  hash_box(hash, plan.required_box_);
  hash_box(hash, plan.guarded_box_);
  plan.plan_hash_ = hash.value();
  return plan;
}

UniformWindowPlan UniformWindowPlan::compact_full_window_identity() const {
  if (empty() || !full_window_)
    throw std::logic_error(
        "only a ready full-window plan may discard its execution payload");
  UniformWindowPlan compact;
  compact.status_ = status_;
  compact.lattice_hash_ = lattice_hash_;
  compact.plan_hash_ = plan_hash_;
  compact.local_n_x_ = local_n_x_;
  compact.local_n_y_ = local_n_y_;
  compact.padded_n_x_ = padded_n_x_;
  compact.padded_n_y_ = padded_n_y_;
  compact.origin_x_ = origin_x_;
  compact.origin_y_ = origin_y_;
  compact.full_window_ = full_window_;
  compact.guard_clipped_low_x_ = guard_clipped_low_x_;
  compact.guard_clipped_high_x_ = guard_clipped_high_x_;
  compact.guard_clipped_low_y_ = guard_clipped_low_y_;
  compact.guard_clipped_high_y_ = guard_clipped_high_y_;
  compact.estimated_bytes_ = estimated_bytes_;
  compact.accepted_source_count_ = accepted_source_count_;
  compact.accepted_target_count_ = accepted_target_count_;
  compact.required_node_count_ = required_node_count_;
  compact.descriptor_payload_retained_ = false;
  compact.source_box_ = source_box_;
  compact.gather_box_ = gather_box_;
  compact.required_box_ = required_box_;
  compact.guarded_box_ = guarded_box_;
  return compact;
}

bool UniformWindowPlan::contains_canonical_node(
    const CanonicalNode& node) const noexcept {
  if (empty()) return false;
  if (node.x < origin_x_ || node.y < origin_y_) return false;
  const std::uint64_t local_x =
      static_cast<std::uint64_t>(node.x - origin_x_);
  const std::uint64_t local_y =
      static_cast<std::uint64_t>(node.y - origin_y_);
  return local_x < local_n_x_ && local_y < local_n_y_;
}

CanonicalNode UniformWindowPlan::to_local(const CanonicalNode& node) const {
  if (!contains_canonical_node(node)) {
    std::ostringstream message;
    message << "canonical node (" << node.x << ',' << node.y
            << ") is not in this active window";
    fail(WindowPlanErrorCode::NodeNotInWindow, message.str());
  }
  return {node.x - origin_x_, node.y - origin_y_};
}

std::size_t UniformWindowPlan::local_linear_index(
    const CanonicalNode& node) const {
  const CanonicalNode local = to_local(node);
  return checked_add(
      checked_multiply(static_cast<std::size_t>(local.x), local_n_y_,
                       "local row offset"),
      static_cast<std::size_t>(local.y), "local linear index");
}

}  // namespace adaptive_grid
}  // namespace feebl
