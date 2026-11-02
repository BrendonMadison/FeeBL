#ifndef FEEBL_ELECTRON_LUMI_SPATIAL_CPP_H
#define FEEBL_ELECTRON_LUMI_SPATIAL_CPP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace feebl {
namespace electron_lumi_spatial {

using RootId = std::uint64_t;

// This is a versioned, physical collision raster.  Its geometry is unrelated
// to GRID::mesh_: the acceleration grid may change without changing this
// estimator.  The bounds are inclusive CIC node coordinates, so nx,ny >= 2.
constexpr std::uint32_t kPhysicalCicKernelVersion = 1U;

enum class SpatialErrorCode {
  InvalidGeometry,
  InvalidRoot,
  DuplicateRootId,
  OutOfDomainRoot,
  ArithmeticOverflow,
  ArithmeticUnderflow,
  ResourceLimitExceeded,
  ClosureFailure,
  InternalInvariant
};

class SpatialError : public std::runtime_error {
 public:
  SpatialError(SpatialErrorCode code, const std::string& message);
  SpatialErrorCode code() const noexcept { return code_; }

 private:
  SpatialErrorCode code_;
};

struct PhysicalCicGeometryInput {
  std::size_t n_x = 0U;
  std::size_t n_y = 0U;
  double x_min = 0.0;
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;
};

class PhysicalCicGeometry {
 public:
  explicit PhysicalCicGeometry(const PhysicalCicGeometryInput& input);

  std::size_t n_x() const noexcept { return n_x_; }
  std::size_t n_y() const noexcept { return n_y_; }
  std::size_t node_count() const noexcept { return node_count_; }
  double x_min() const noexcept { return x_min_; }
  double x_max() const noexcept { return x_max_; }
  double y_min() const noexcept { return y_min_; }
  double y_max() const noexcept { return y_max_; }
  double delta_x() const noexcept { return delta_x_; }
  double delta_y() const noexcept { return delta_y_; }
  double control_area() const noexcept { return control_area_; }
  std::uint64_t geometry_hash() const noexcept { return geometry_hash_; }

  std::size_t linear_node(std::size_t x, std::size_t y) const;

 private:
  std::size_t n_x_;
  std::size_t n_y_;
  std::size_t node_count_;
  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;
  double delta_x_;
  double delta_y_;
  double control_area_;
  std::uint64_t geometry_hash_;
};

// population is the physical number represented by this independent root.
// Charge sign does not enter an electron-electron luminosity overlap weight.
struct PhysicalRoot {
  RootId id = 0U;
  double x = 0.0;
  double y = 0.0;
  double population = 0.0;
};

struct CicNodeFraction {
  std::size_t x = 0U;
  std::size_t y = 0U;
  double fraction = 0.0;
};

struct RootStencil {
  PhysicalRoot root;
  std::size_t base_x = 0U;
  std::size_t base_y = 0U;
  // Canonical order: (0,0), (1,0), (0,1), (1,1) relative to the base.
  std::array<CicNodeFraction, 4U> nodes;

  double fraction_sum() const noexcept;
};

enum class RejectionReason : std::uint8_t {
  None = 0U,
  BelowX = 1U,
  AboveX = 2U,
  BelowY = 4U,
  AboveY = 8U
};

RejectionReason operator|(RejectionReason left,
                          RejectionReason right) noexcept;
bool has_reason(RejectionReason value, RejectionReason reason) noexcept;

struct RejectedRoot {
  RootId id = 0U;
  double population = 0.0;
  RejectionReason reason = RejectionReason::None;
};

struct BeamAdmissionAccounting {
  std::size_t input_roots = 0U;
  std::size_t accepted_roots = 0U;
  std::size_t rejected_roots = 0U;
  double input_population = 0.0;
  double accepted_population = 0.0;
  double rejected_population = 0.0;
};

// audit_admission is deliberately usable before strict plan construction so a
// caller can publish complete tail accounting.  Invalid/non-finite roots and
// duplicate IDs throw; finite roots outside the physical raster are returned
// as whole-root rejections.  No boundary stencil is silently renormalized.
struct AdmissionAudit {
  std::vector<RootStencil> accepted;
  std::vector<RejectedRoot> rejected;
  BeamAdmissionAccounting accounting;
};

AdmissionAudit audit_admission(const PhysicalCicGeometry& geometry,
                               const std::vector<PhysicalRoot>& roots);

struct CollisionPair {
  RootId beam1_id = 0U;
  RootId beam2_id = 0U;
  // Dimensionless dot product of the two normalized four-node stencils.
  double stencil_overlap = 0.0;
  // stencil_overlap / physical control-cell area.
  double spatial_kernel = 0.0;
  // normalization * population1 * population2 * spatial_kernel.
  double weight = 0.0;

  bool operator==(const CollisionPair& other) const noexcept;
};

struct DensityDotOracle {
  // Sum over physical raster nodes of deposited beam populations' product.
  double unscaled_node_dot = 0.0;
  // normalization * unscaled_node_dot / physical control-cell area.
  double weight = 0.0;
  std::size_t beam1_nonzero_nodes = 0U;
  std::size_t beam2_nonzero_nodes = 0U;
  std::size_t common_nonzero_nodes = 0U;
};

struct ClosureReport {
  double pair_weight_sum = 0.0;
  double density_dot_weight = 0.0;
  double absolute_error = 0.0;
  double relative_error = 0.0;
  double relative_tolerance = 0.0;
  bool within_tolerance = false;
};

struct CollisionPlanOptions {
  // The FeeBL runtime can later supply 1e18/timestep (and any explicitly
  // audited sampling factor).  Keeping it explicit makes this module unit- and
  // dimension-testable and prevents nominal grid area from leaking in.
  double normalization = 1.0;
  std::size_t max_pairs = std::numeric_limits<std::size_t>::max();
  double closure_relative_tolerance = 1.0e-12;
};

DensityDotOracle independently_deposited_density_dot(
    const PhysicalCicGeometry& geometry,
    const std::vector<RootStencil>& beam1,
    const std::vector<RootStencil>& beam2,
    double normalization);

ClosureReport evaluate_closure(double pair_weight_sum,
                               double density_dot_weight,
                               double relative_tolerance);

class CollisionPlan {
 public:
  // Physics construction is strict: every root must have complete support in
  // the fixed physical raster.  Call audit_admission first if rejected-tail
  // details are needed for a failure receipt.
  static CollisionPlan build(const PhysicalCicGeometry& geometry,
                             const std::vector<PhysicalRoot>& beam1,
                             const std::vector<PhysicalRoot>& beam2,
                             const CollisionPlanOptions& options = {});

  const PhysicalCicGeometry& geometry() const noexcept { return geometry_; }
  const std::vector<RootStencil>& beam1_stencils() const noexcept {
    return beam1_stencils_;
  }
  const std::vector<RootStencil>& beam2_stencils() const noexcept {
    return beam2_stencils_;
  }
  const BeamAdmissionAccounting& beam1_accounting() const noexcept {
    return beam1_accounting_;
  }
  const BeamAdmissionAccounting& beam2_accounting() const noexcept {
    return beam2_accounting_;
  }
  const std::vector<CollisionPair>& pairs() const noexcept { return pairs_; }
  const DensityDotOracle& density_dot_oracle() const noexcept {
    return density_dot_oracle_;
  }
  const ClosureReport& closure() const noexcept { return closure_; }
  double normalization() const noexcept { return normalization_; }
  std::uint64_t plan_hash() const noexcept { return plan_hash_; }

 private:
  CollisionPlan(const PhysicalCicGeometry& geometry,
                std::vector<RootStencil> beam1_stencils,
                std::vector<RootStencil> beam2_stencils,
                const BeamAdmissionAccounting& beam1_accounting,
                const BeamAdmissionAccounting& beam2_accounting,
                std::vector<CollisionPair> pairs,
                const DensityDotOracle& density_dot_oracle,
                const ClosureReport& closure,
                double normalization,
                std::uint64_t plan_hash);

  PhysicalCicGeometry geometry_;
  std::vector<RootStencil> beam1_stencils_;
  std::vector<RootStencil> beam2_stencils_;
  BeamAdmissionAccounting beam1_accounting_;
  BeamAdmissionAccounting beam2_accounting_;
  std::vector<CollisionPair> pairs_;
  DensityDotOracle density_dot_oracle_;
  ClosureReport closure_;
  double normalization_;
  std::uint64_t plan_hash_;
};

}  // namespace electron_lumi_spatial
}  // namespace feebl

#endif  // FEEBL_ELECTRON_LUMI_SPATIAL_CPP_H
