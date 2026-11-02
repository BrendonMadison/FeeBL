#ifndef FEEBL_UNIFORM_WINDOW_PLAN_CPP_H
#define FEEBL_UNIFORM_WINDOW_PLAN_CPP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace feebl {
namespace adaptive_grid {

using CanonicalIndex = std::int64_t;

struct CanonicalNode {
  CanonicalIndex x = 0;
  CanonicalIndex y = 0;

  bool operator==(const CanonicalNode& other) const noexcept {
    return x == other.x && y == other.y;
  }
  bool operator!=(const CanonicalNode& other) const noexcept {
    return !(*this == other);
  }
  bool operator<(const CanonicalNode& other) const noexcept {
    return x < other.x || (x == other.x && y < other.y);
  }
};

// Inclusive canonical-node bounds.  Invalid is the only representation of an
// empty node set; valid boxes always have ordered, non-negative endpoints.
struct CanonicalNodeBox {
  bool valid = false;
  CanonicalIndex min_x = 0;
  CanonicalIndex max_x = 0;
  CanonicalIndex min_y = 0;
  CanonicalIndex max_y = 0;

  std::size_t width() const noexcept;
  std::size_t height() const noexcept;
  bool contains(const CanonicalNode& node) const noexcept;
};

enum class WindowPlanStatus {
  Empty,
  Ready
};

enum class WindowPlanErrorCode {
  InvalidLattice,
  InvalidDescriptor,
  InvalidClassBank,
  InvalidPolicy,
  NoFittingClass,
  FullWindowForbidden,
  OutOfCanonicalBank,
  ArithmeticOverflow,
  MemoryBudgetExceeded,
  NodeNotInWindow
};

class WindowPlanError : public std::runtime_error {
 public:
  WindowPlanError(WindowPlanErrorCode code, const std::string& message);
  WindowPlanErrorCode code() const noexcept { return code_; }

 private:
  WindowPlanErrorCode code_;
};

// The caller supplies the already-established canonical GRID values.  They
// are retained and hashed bit-for-bit; this module never recomputes spacing,
// offsets, intended bounds, or the case-2 Green-kernel factor.
struct CanonicalLatticeInput {
  std::size_t n_x = 0;
  std::size_t n_y = 0;
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float delta_z = 0.0f;
  float inverse_delta_x = 0.0f;
  float inverse_delta_y = 0.0f;
  float offset_x = 0.0f;
  float offset_y = 0.0f;
  float intended_min_x = 0.0f;
  float intended_max_x = 0.0f;
  float intended_min_y = 0.0f;
  float intended_max_y = 0.0f;
  float charge_sign = 0.0f;
  double field_kernel_factor = 0.0;
  std::uint64_t geometry_revision = 0;
};

// Immutable through its public interface.  Copy construction is intentional:
// a wave may retain the exact snapshot used to build its encounter plans.
class CanonicalLatticeSnapshot {
 public:
  explicit CanonicalLatticeSnapshot(const CanonicalLatticeInput& input);

  std::size_t n_x() const noexcept { return n_x_; }
  std::size_t n_y() const noexcept { return n_y_; }
  float delta_x() const noexcept { return delta_x_; }
  float delta_y() const noexcept { return delta_y_; }
  float delta_z() const noexcept { return delta_z_; }
  float inverse_delta_x() const noexcept { return inverse_delta_x_; }
  float inverse_delta_y() const noexcept { return inverse_delta_y_; }
  float offset_x() const noexcept { return offset_x_; }
  float offset_y() const noexcept { return offset_y_; }
  float intended_min_x() const noexcept { return intended_min_x_; }
  float intended_max_x() const noexcept { return intended_max_x_; }
  float intended_min_y() const noexcept { return intended_min_y_; }
  float intended_max_y() const noexcept { return intended_max_y_; }
  float charge_sign() const noexcept { return charge_sign_; }
  double field_kernel_factor() const noexcept { return field_kernel_factor_; }
  std::uint64_t geometry_revision() const noexcept {
    return geometry_revision_;
  }
  std::uint64_t hash() const noexcept { return hash_; }

 private:
  std::size_t n_x_;
  std::size_t n_y_;
  float delta_x_;
  float delta_y_;
  float delta_z_;
  float inverse_delta_x_;
  float inverse_delta_y_;
  float offset_x_;
  float offset_y_;
  float intended_min_x_;
  float intended_max_x_;
  float intended_min_y_;
  float intended_max_y_;
  float charge_sign_;
  double field_kernel_factor_;
  std::uint64_t geometry_revision_;
  std::uint64_t hash_;
};

// A source descriptor is created only after GENERAL_GRID::particleInGrid has
// made the canonical admission and returned its exact base indices/weights.
// Rejected particles carry no coordinates and require no local storage.
class CanonicalDepositDescriptor {
 public:
  static CanonicalDepositDescriptor rejected() noexcept;
  static CanonicalDepositDescriptor from_canonical(
      CanonicalIndex base_x, CanonicalIndex base_y,
      float weight_x, float weight_y);

  bool accepted() const noexcept { return accepted_; }
  CanonicalIndex base_x() const noexcept { return base_x_; }
  CanonicalIndex base_y() const noexcept { return base_y_; }
  float weight_x() const noexcept { return weight_x_; }
  float weight_y() const noexcept { return weight_y_; }
  const std::array<CanonicalNode, 4>& nodes() const noexcept { return nodes_; }

 private:
  CanonicalDepositDescriptor() noexcept;

  bool accepted_;
  CanonicalIndex base_x_;
  CanonicalIndex base_y_;
  float weight_x_;
  float weight_y_;
  std::array<CanonicalNode, 4> nodes_;
};

// The two localization pairs deliberately remain distinct.  They are the
// float-narrowed Y-then-X decisions made by interpolePotential, not values
// reconstructed from a local origin.  The twelve nodes preserve production
// read order and may contain duplicates.
class CanonicalGatherDescriptor {
 public:
  static CanonicalGatherDescriptor rejected() noexcept;
  static CanonicalGatherDescriptor from_canonical(
      CanonicalIndex y_base_x, CanonicalIndex y_base_y, float y_weight,
      CanonicalIndex x_base_x, CanonicalIndex x_base_y, float x_weight,
      double fractional_x, double fractional_y);
  // absolute_x/y are the PHI_FLOAT outputs already produced by
  // GENERAL_GRID::absoluteCoordinates for a target already accepted by
  // coordinatesInGridRange.  This helper intentionally casts them back to
  // float before both legacy localization decisions; it does not accept
  // physical coordinates and does not repeat the canonical affine map.
  static CanonicalGatherDescriptor from_legacy_absolute_coordinates(
      double absolute_x, double absolute_y);

  bool accepted() const noexcept { return accepted_; }
  CanonicalIndex y_base_x() const noexcept { return y_base_x_; }
  CanonicalIndex y_base_y() const noexcept { return y_base_y_; }
  float y_weight() const noexcept { return y_weight_; }
  CanonicalIndex x_base_x() const noexcept { return x_base_x_; }
  CanonicalIndex x_base_y() const noexcept { return x_base_y_; }
  float x_weight() const noexcept { return x_weight_; }
  double fractional_x() const noexcept { return fractional_x_; }
  double fractional_y() const noexcept { return fractional_y_; }
  const std::array<CanonicalNode, 12>& nodes() const noexcept { return nodes_; }

 private:
  CanonicalGatherDescriptor() noexcept;

  bool accepted_;
  CanonicalIndex y_base_x_;
  CanonicalIndex y_base_y_;
  float y_weight_;
  CanonicalIndex x_base_x_;
  CanonicalIndex x_base_y_;
  float x_weight_;
  double fractional_x_;
  double fractional_y_;
  std::array<CanonicalNode, 12> nodes_;
};

// Beam separation is part of the plan hash.  Vector order is canonical
// particle order and is likewise significant even when node unions coincide.
struct CanonicalWindowInputs {
  std::vector<CanonicalDepositDescriptor> beam1_sources;
  std::vector<CanonicalDepositDescriptor> beam2_sources;
  std::vector<CanonicalGatherDescriptor> beam1_targets;
  std::vector<CanonicalGatherDescriptor> beam2_targets;
};

// Parse a comma-separated class bank, reject malformed/out-of-range values,
// and return its sorted unique representation.
std::vector<std::size_t> parse_size_classes(const std::string& value);

struct UniformWindowPolicy {
  std::vector<std::size_t> size_classes_x;
  std::vector<std::size_t> size_classes_y;
  std::size_t min_n_x = 4;
  std::size_t min_n_y = 4;
  // Zero means the matching canonical dimension.
  std::size_t max_n_x = 0;
  std::size_t max_n_y = 0;
  std::int64_t extra_guard_cells = 0;
  bool allow_full_window = true;

  // The planner checks both native and doubled-domain products.  These byte
  // coefficients let the caller represent its exact context layout without
  // coupling this geometry module to FIELD/FFTW ownership.  Defaults count
  // four local real arrays and three padded complex-double arrays.
  std::size_t bytes_per_local_cell = 4U * sizeof(double);
  std::size_t bytes_per_padded_cell = 6U * sizeof(double);
  std::size_t fixed_bytes = 0;
  std::size_t max_bytes = std::numeric_limits<std::size_t>::max();
};

class UniformWindowPlan {
 public:
  static UniformWindowPlan build(
      const CanonicalLatticeSnapshot& lattice,
      CanonicalWindowInputs inputs,
      const UniformWindowPolicy& policy);

  WindowPlanStatus status() const noexcept { return status_; }
  bool empty() const noexcept { return status_ == WindowPlanStatus::Empty; }
  std::uint64_t lattice_hash() const noexcept { return lattice_hash_; }
  std::uint64_t plan_hash() const noexcept { return plan_hash_; }

  std::size_t local_n_x() const noexcept { return local_n_x_; }
  std::size_t local_n_y() const noexcept { return local_n_y_; }
  std::size_t padded_n_x() const noexcept { return padded_n_x_; }
  std::size_t padded_n_y() const noexcept { return padded_n_y_; }
  CanonicalIndex origin_x() const noexcept { return origin_x_; }
  CanonicalIndex origin_y() const noexcept { return origin_y_; }
  bool full_window() const noexcept { return full_window_; }
  bool guard_clipped_low_x() const noexcept { return guard_clipped_low_x_; }
  bool guard_clipped_high_x() const noexcept { return guard_clipped_high_x_; }
  bool guard_clipped_low_y() const noexcept { return guard_clipped_low_y_; }
  bool guard_clipped_high_y() const noexcept { return guard_clipped_high_y_; }
  std::size_t estimated_bytes() const noexcept { return estimated_bytes_; }

  std::size_t accepted_source_count() const noexcept {
    return accepted_source_count_;
  }
  std::size_t accepted_target_count() const noexcept {
    return accepted_target_count_;
  }
  std::size_t required_node_count() const noexcept {
    return required_node_count_;
  }
  bool descriptor_payload_retained() const noexcept {
    return descriptor_payload_retained_;
  }
  // The canonical CPU full-window path reuses its established GRID/FIELD
  // implementation and needs only the immutable plan/result identity after
  // planning.  Return that identity without copying descriptor/node vectors.
  // Cropped and GPU plans retain the complete payload used for execution.
  UniformWindowPlan compact_full_window_identity() const;
  const std::vector<CanonicalDepositDescriptor>& beam1_sources() const noexcept {
    return inputs_.beam1_sources;
  }
  const std::vector<CanonicalDepositDescriptor>& beam2_sources() const noexcept {
    return inputs_.beam2_sources;
  }
  const std::vector<CanonicalGatherDescriptor>& beam1_targets() const noexcept {
    return inputs_.beam1_targets;
  }
  const std::vector<CanonicalGatherDescriptor>& beam2_targets() const noexcept {
    return inputs_.beam2_targets;
  }
  // All three node sets are sorted canonical x-major/y-minor and unique.
  const std::vector<CanonicalNode>& source_nodes() const noexcept {
    return source_nodes_;
  }
  const std::vector<CanonicalNode>& gather_nodes() const noexcept {
    return gather_nodes_;
  }
  const std::vector<CanonicalNode>& required_nodes() const noexcept {
    return required_nodes_;
  }
  const CanonicalNodeBox& source_box() const noexcept { return source_box_; }
  const CanonicalNodeBox& gather_box() const noexcept { return gather_box_; }
  const CanonicalNodeBox& required_box() const noexcept {
    return required_box_;
  }
  const CanonicalNodeBox& guarded_box() const noexcept { return guarded_box_; }

  bool contains_canonical_node(const CanonicalNode& node) const noexcept;
  CanonicalNode to_local(const CanonicalNode& node) const;
  std::size_t local_linear_index(const CanonicalNode& node) const;

 private:
  UniformWindowPlan() = default;

  WindowPlanStatus status_ = WindowPlanStatus::Empty;
  std::uint64_t lattice_hash_ = 0;
  std::uint64_t plan_hash_ = 0;
  std::size_t local_n_x_ = 0;
  std::size_t local_n_y_ = 0;
  std::size_t padded_n_x_ = 0;
  std::size_t padded_n_y_ = 0;
  CanonicalIndex origin_x_ = 0;
  CanonicalIndex origin_y_ = 0;
  bool full_window_ = false;
  bool guard_clipped_low_x_ = false;
  bool guard_clipped_high_x_ = false;
  bool guard_clipped_low_y_ = false;
  bool guard_clipped_high_y_ = false;
  std::size_t estimated_bytes_ = 0;
  std::size_t accepted_source_count_ = 0;
  std::size_t accepted_target_count_ = 0;
  std::size_t required_node_count_ = 0;
  bool descriptor_payload_retained_ = true;
  CanonicalWindowInputs inputs_;
  std::vector<CanonicalNode> source_nodes_;
  std::vector<CanonicalNode> gather_nodes_;
  std::vector<CanonicalNode> required_nodes_;
  CanonicalNodeBox source_box_;
  CanonicalNodeBox gather_box_;
  CanonicalNodeBox required_box_;
  CanonicalNodeBox guarded_box_;
};

}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_UNIFORM_WINDOW_PLAN_CPP_H
