#include "adaptiveCompositeCouplingCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace feebl {
namespace adaptive_grid {
namespace {

struct Integral1D {
  double weight = 0.0;
  double derivative = 0.0;
  double first_moment = 0.0;
};

bool finite(double value) { return std::isfinite(value); }

double checked_product(double left, double right, const char* label) {
  const double value = left * right;
  if (!finite(value))
    throw std::overflow_error(std::string(label) + " overflowed");
  return value;
}

void checked_accumulate(double* total, double increment, const char* label) {
  if (!total || !finite(increment))
    throw std::overflow_error(std::string(label) + " is non-finite");
  const double value = *total + increment;
  if (!finite(value))
    throw std::overflow_error(std::string(label) + " accumulation overflowed");
  *total = value;
}

std::size_t checked_size_product(std::size_t left, std::size_t right,
                                 const char* label) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflowed");
  return left * right;
}

struct CandidateIndexRange {
  std::size_t begin = 0U;
  std::size_t end = 0U;
};

CandidateIndexRange candidate_index_range(
    double support_begin, double support_end, double origin,
    double spacing, std::size_t count) {
  CandidateIndexRange result;
  if (count == 0U || support_end <= origin ||
      support_begin >= origin + static_cast<double>(count) * spacing)
    return result;
  const double clipped_begin = std::max(support_begin, origin);
  const double clipped_end = std::min(
      support_end, origin + static_cast<double>(count) * spacing);
  const double raw_begin = (clipped_begin - origin) / spacing;
  const double raw_end = (clipped_end - origin) / spacing;
  if (!finite(raw_begin) || !finite(raw_end) || raw_begin < 0.0 ||
      raw_end < 0.0)
    throw std::overflow_error("composite stencil candidate range overflowed");
  const std::size_t nominal_begin = std::min(
      count, static_cast<std::size_t>(std::floor(raw_begin)));
  const std::size_t nominal_end = std::min(
      count, static_cast<std::size_t>(std::ceil(raw_end)));
  // One conservative neighbor on either side protects exact-boundary cases
  // from floating-point quotient rounding.  Exact tent integration below
  // still decides whether the intersection has positive measure.
  result.begin = nominal_begin == 0U ? 0U : nominal_begin - 1U;
  result.end = nominal_end == count ? count : nominal_end + 1U;
  return result;
}

double tent_cdf(double t) {
  if (t <= -1.0) return 0.0;
  if (t < 0.0) {
    const double u = t + 1.0;
    return 0.5 * u * u;
  }
  if (t < 1.0) {
    const double u = 1.0 - t;
    return 1.0 - 0.5 * u * u;
  }
  return 1.0;
}

// Integral from minus infinity to t of u*max(1-|u|,0) du.
double tent_relative_first_primitive(double t) {
  if (t <= -1.0 || t >= 1.0) return 0.0;
  if (t < 0.0)
    return 0.5 * t * t + t * t * t / 3.0 - 1.0 / 6.0;
  return 0.5 * t * t - t * t * t / 3.0 - 1.0 / 6.0;
}

double normalized_tent_at(double x, double center, double half_width) {
  const double distance = std::fabs(x - center);
  if (!(distance < half_width)) return 0.0;
  return (1.0 - distance / half_width) / half_width;
}

Integral1D integrate_tent_1d(double cell_begin, double cell_end,
                             double center, double half_width) {
  Integral1D result;
  const double support_begin = center - half_width;
  const double support_end = center + half_width;
  const double begin = std::max(cell_begin, support_begin);
  const double end = std::min(cell_end, support_end);
  if (!(end > begin)) return result;

  const double t_begin = std::max(
      -1.0, std::min(1.0, (begin - center) / half_width));
  const double t_end = std::max(
      -1.0, std::min(1.0, (end - center) / half_width));
  result.weight = tent_cdf(t_end) - tent_cdf(t_begin);
  result.derivative =
      normalized_tent_at(cell_begin, center, half_width) -
      normalized_tent_at(cell_end, center, half_width);
  result.first_moment = center * result.weight + half_width *
      (tent_relative_first_primitive(t_end) -
       tent_relative_first_primitive(t_begin));
  if (!finite(result.weight) || !finite(result.derivative) ||
      !finite(result.first_moment) || result.weight < 0.0)
    throw std::overflow_error("non-finite composite tent integral");
  return result;
}

void hash_bytes(std::uint64_t* state, const void* bytes, std::size_t count) {
  const unsigned char* data = static_cast<const unsigned char*>(bytes);
  for (std::size_t i = 0U; i < count; ++i) {
    *state ^= static_cast<std::uint64_t>(data[i]);
    *state *= UINT64_C(1099511628211);
  }
}

template <typename T>
void hash_scalar(std::uint64_t* state, const T& value) {
  hash_bytes(state, &value, sizeof(value));
}

void add_particle_moments(CompositeChargeMomentsR2* moments,
                          const CompositeTentParticleR2& particle) {
  checked_accumulate(&moments->signed_charge, particle.signed_charge,
                     "composite signed charge");
  checked_accumulate(&moments->absolute_charge,
                     std::fabs(particle.signed_charge),
                     "composite absolute charge");
  checked_accumulate(
      &moments->signed_first_x,
      checked_product(particle.signed_charge, particle.position_x,
                      "composite signed x moment"),
      "composite signed x moment");
  checked_accumulate(
      &moments->signed_first_y,
      checked_product(particle.signed_charge, particle.position_y,
                      "composite signed y moment"),
      "composite signed y moment");
}

void validate_finite_particle(const CompositeTentParticleR2& particle) {
  if (!finite(particle.position_x) || !finite(particle.position_y) ||
      !finite(particle.signed_charge) ||
      !finite(particle.support_half_width_x) ||
      !finite(particle.support_half_width_y))
    throw std::invalid_argument(
        "composite tent particle contains a non-finite datum");
  if (!(particle.support_half_width_x > 0.0) ||
      !(particle.support_half_width_y > 0.0))
    throw std::invalid_argument(
        "composite tent support half-width must be positive");
  if (!finite(particle.position_x - particle.support_half_width_x) ||
      !finite(particle.position_x + particle.support_half_width_x) ||
      !finite(particle.position_y - particle.support_half_width_y) ||
      !finite(particle.position_y + particle.support_half_width_y))
    throw std::overflow_error("composite tent support bounds overflowed");
}

void leaf_bounds(const CompositePatchGeometryR2& geometry,
                 const CompositeLeafCell& leaf,
                 double* begin_x, double* end_x,
                 double* begin_y, double* end_y) {
  if (leaf.level == CompositeLevel::Coarse) {
    *begin_x = geometry.physical_origin_x +
        static_cast<double>(leaf.level_i) * geometry.coarse_delta_x;
    *begin_y = geometry.physical_origin_y +
        static_cast<double>(leaf.level_j) * geometry.coarse_delta_y;
  } else {
    const double fine_delta_x = 0.5 * geometry.coarse_delta_x;
    const double fine_delta_y = 0.5 * geometry.coarse_delta_y;
    *begin_x = geometry.physical_origin_x +
        static_cast<double>(geometry.patch_begin_x) *
            geometry.coarse_delta_x +
        static_cast<double>(leaf.level_i) * fine_delta_x;
    *begin_y = geometry.physical_origin_y +
        static_cast<double>(geometry.patch_begin_y) *
            geometry.coarse_delta_y +
        static_cast<double>(leaf.level_j) * fine_delta_y;
  }
  *end_x = *begin_x + leaf.delta_x;
  *end_y = *begin_y + leaf.delta_y;
}

std::size_t owner_leaf(const CompositePoissonR2& topology,
                       const CompositeTentParticleR2& particle) {
  const CompositePatchGeometryR2& geometry = topology.geometry();
  const double domain_end_x = geometry.physical_origin_x +
      static_cast<double>(geometry.coarse_n_x) * geometry.coarse_delta_x;
  const double domain_end_y = geometry.physical_origin_y +
      static_cast<double>(geometry.coarse_n_y) * geometry.coarse_delta_y;
  if (particle.position_x < geometry.physical_origin_x ||
      !(particle.position_x < domain_end_x) ||
      particle.position_y < geometry.physical_origin_y ||
      !(particle.position_y < domain_end_y))
    return kCompositeNoIndex;

  const double patch_begin_x = geometry.physical_origin_x +
      static_cast<double>(geometry.patch_begin_x) * geometry.coarse_delta_x;
  const double patch_end_x = geometry.physical_origin_x +
      static_cast<double>(geometry.patch_end_x) * geometry.coarse_delta_x;
  const double patch_begin_y = geometry.physical_origin_y +
      static_cast<double>(geometry.patch_begin_y) * geometry.coarse_delta_y;
  const double patch_end_y = geometry.physical_origin_y +
      static_cast<double>(geometry.patch_end_y) * geometry.coarse_delta_y;
  const bool in_patch = particle.position_x >= patch_begin_x &&
      particle.position_x < patch_end_x &&
      particle.position_y >= patch_begin_y &&
      particle.position_y < patch_end_y;
  if (in_patch) {
    const double fine_delta_x = 0.5 * geometry.coarse_delta_x;
    const double fine_delta_y = 0.5 * geometry.coarse_delta_y;
    const std::size_t i = static_cast<std::size_t>(
        std::floor((particle.position_x - patch_begin_x) / fine_delta_x));
    const std::size_t j = static_cast<std::size_t>(
        std::floor((particle.position_y - patch_begin_y) / fine_delta_y));
    return topology.fine_leaf(i, j);
  }

  const std::size_t i = static_cast<std::size_t>(std::floor(
      (particle.position_x - geometry.physical_origin_x) /
      geometry.coarse_delta_x));
  const std::size_t j = static_cast<std::size_t>(std::floor(
      (particle.position_y - geometry.physical_origin_y) /
      geometry.coarse_delta_y));
  return topology.coarse_leaf(i, j);
}

bool outside_status(CompositeTentStatus status) {
  return status == CompositeTentStatus::OutsidePhysicalDomain ||
      status == CompositeTentStatus::SupportOutsidePhysicalDomain;
}

bool same_contribution(const CompositeTentContributionR2& left,
                       const CompositeTentContributionR2& right) {
  return left.leaf == right.leaf && left.weight == right.weight &&
      left.derivative_x == right.derivative_x &&
      left.derivative_y == right.derivative_y &&
      left.shape_first_x == right.shape_first_x &&
      left.shape_first_y == right.shape_first_y;
}

bool same_stencil(const CompositeTentStencilR2& left,
                  const CompositeTentStencilR2& right) {
  if (left.particle.stable_id != right.particle.stable_id ||
      left.particle.position_x != right.particle.position_x ||
      left.particle.position_y != right.particle.position_y ||
      left.particle.signed_charge != right.particle.signed_charge ||
      left.particle.support_half_width_x !=
          right.particle.support_half_width_x ||
      left.particle.support_half_width_y !=
          right.particle.support_half_width_y ||
      left.status != right.status || left.owner_leaf != right.owner_leaf ||
      left.touches_coarse != right.touches_coarse ||
      left.touches_fine != right.touches_fine ||
      left.crosses_coarse_fine_interface !=
          right.crosses_coarse_fine_interface ||
      left.weight_sum != right.weight_sum ||
      left.derivative_x_sum != right.derivative_x_sum ||
      left.derivative_y_sum != right.derivative_y_sum ||
      left.shape_first_x_sum != right.shape_first_x_sum ||
      left.shape_first_y_sum != right.shape_first_y_sum ||
      left.contributions.size() != right.contributions.size())
    return false;
  for (std::size_t index = 0U; index < left.contributions.size(); ++index)
    if (!same_contribution(left.contributions[index],
                           right.contributions[index]))
      return false;
  return true;
}

void verify_stencil_topology(const CompositePoissonR2& topology,
                             const CompositeTentStencilSetR2& stencil_set) {
  if (stencil_set.leaf_count != topology.leaf_count() ||
      stencil_set.topology_fingerprint !=
          composite_coupling_topology_fingerprint_r2(topology))
    throw std::invalid_argument(
        "composite tent stencils do not match this topology");
  if (!finite(stencil_set.closure_tolerance) ||
      !(stencil_set.closure_tolerance > 0.0))
    throw std::invalid_argument("invalid composite stencil tolerance");
  std::vector<CompositeTentParticleR2> particles;
  particles.reserve(stencil_set.stencils.size());
  for (const CompositeTentStencilR2& stencil : stencil_set.stencils)
    particles.push_back(stencil.particle);
  const CompositeTentStencilSetR2 canonical =
      build_composite_tent_stencils_r2(
          topology, particles, stencil_set.closure_tolerance);
  if (canonical.stencils.size() != stencil_set.stencils.size())
    throw std::logic_error(
        "canonical composite stencil recomputation returned wrong shape");
  for (std::size_t index = 0U; index < canonical.stencils.size(); ++index)
    if (!same_stencil(stencil_set.stencils[index],
                      canonical.stencils[index]))
      throw std::invalid_argument(
          "composite tent stencil set differs from canonical recomputation");
}

void validate_stencil(const CompositePoissonR2& topology,
                      const CompositeTentStencilSetR2& stencil_set,
                      const CompositeTentStencilR2& stencil) {
  validate_finite_particle(stencil.particle);
  if (stencil.status != CompositeTentStatus::Accepted) {
    if (!stencil.contributions.empty())
      throw std::invalid_argument(
          "rejected composite tent stencil retains source contributions");
    return;
  }
  if (stencil.owner_leaf >= topology.leaf_count() ||
      stencil.owner_leaf != owner_leaf(topology, stencil.particle) ||
      topology.leaves()[stencil.owner_leaf].fixed_dirichlet ||
      stencil.contributions.empty())
    throw std::invalid_argument(
        "accepted composite tent stencil has invalid finest owner");

  double weight_sum = 0.0;
  double derivative_x_sum = 0.0;
  double derivative_y_sum = 0.0;
  double first_x_sum = 0.0;
  double first_y_sum = 0.0;
  bool touches_coarse = false;
  bool touches_fine = false;
  const CompositeLevel owner_level =
      topology.leaves()[stencil.owner_leaf].level;
  std::size_t previous_leaf = kCompositeNoIndex;
  for (const CompositeTentContributionR2& contribution :
       stencil.contributions) {
    double begin_x = 0.0;
    double end_x = 0.0;
    double begin_y = 0.0;
    double end_y = 0.0;
    if (contribution.leaf < topology.leaf_count())
      leaf_bounds(topology.geometry(), topology.leaves()[contribution.leaf],
                  &begin_x, &end_x, &begin_y, &end_y);
    const Integral1D expected_x = contribution.leaf < topology.leaf_count()
        ? integrate_tent_1d(begin_x, end_x, stencil.particle.position_x,
                            stencil.particle.support_half_width_x)
        : Integral1D();
    const Integral1D expected_y = contribution.leaf < topology.leaf_count()
        ? integrate_tent_1d(begin_y, end_y, stencil.particle.position_y,
                            stencil.particle.support_half_width_y)
        : Integral1D();
    const double expected_weight = expected_x.weight * expected_y.weight;
    const double expected_derivative_x =
        expected_x.derivative * expected_y.weight;
    const double expected_derivative_y =
        expected_x.weight * expected_y.derivative;
    const double expected_first_x =
        expected_x.first_moment * expected_y.weight;
    const double expected_first_y =
        expected_x.weight * expected_y.first_moment;
    if (contribution.leaf >= topology.leaf_count() ||
        topology.leaves()[contribution.leaf].fixed_dirichlet ||
        topology.leaves()[contribution.leaf].level != owner_level ||
        (previous_leaf != kCompositeNoIndex &&
         contribution.leaf <= previous_leaf) ||
        !finite(contribution.weight) ||
        !finite(contribution.derivative_x) ||
        !finite(contribution.derivative_y) ||
        !finite(contribution.shape_first_x) ||
        !finite(contribution.shape_first_y) ||
        !(contribution.weight > 0.0) ||
        contribution.weight != expected_weight ||
        contribution.derivative_x != expected_derivative_x ||
        contribution.derivative_y != expected_derivative_y ||
        contribution.shape_first_x != expected_first_x ||
        contribution.shape_first_y != expected_first_y)
      throw std::invalid_argument(
          "accepted composite tent stencil differs from its canonical "
          "analytic shape integral");
    previous_leaf = contribution.leaf;
    weight_sum += contribution.weight;
    derivative_x_sum += contribution.derivative_x;
    derivative_y_sum += contribution.derivative_y;
    first_x_sum += contribution.shape_first_x;
    first_y_sum += contribution.shape_first_y;
    touches_coarse = touches_coarse ||
        topology.leaves()[contribution.leaf].level == CompositeLevel::Coarse;
    touches_fine = touches_fine ||
        topology.leaves()[contribution.leaf].level == CompositeLevel::Fine;
  }

  const double tolerance = stencil_set.closure_tolerance;
  const double first_x_scale = std::max(
      1.0, std::max(std::fabs(stencil.particle.position_x),
                    stencil.particle.support_half_width_x));
  const double first_y_scale = std::max(
      1.0, std::max(std::fabs(stencil.particle.position_y),
                    stencil.particle.support_half_width_y));
  if (std::fabs(weight_sum - stencil.weight_sum) > tolerance ||
      std::fabs(derivative_x_sum - stencil.derivative_x_sum) *
          stencil.particle.support_half_width_x > tolerance ||
      std::fabs(derivative_y_sum - stencil.derivative_y_sum) *
          stencil.particle.support_half_width_y > tolerance ||
      std::fabs(first_x_sum - stencil.shape_first_x_sum) /
          first_x_scale > tolerance ||
      std::fabs(first_y_sum - stencil.shape_first_y_sum) /
          first_y_scale > tolerance ||
      std::fabs(weight_sum - 1.0) > tolerance ||
      std::fabs(derivative_x_sum) *
          stencil.particle.support_half_width_x > tolerance ||
      std::fabs(derivative_y_sum) *
          stencil.particle.support_half_width_y > tolerance ||
      std::fabs(first_x_sum - stencil.particle.position_x) /
          first_x_scale > tolerance ||
      std::fabs(first_y_sum - stencil.particle.position_y) /
          first_y_scale > tolerance ||
      stencil.touches_coarse != touches_coarse ||
      stencil.touches_fine != touches_fine ||
      stencil.crosses_coarse_fine_interface ||
      (owner_level == CompositeLevel::Coarse &&
       (!touches_coarse || touches_fine)) ||
      (owner_level == CompositeLevel::Fine &&
       (!touches_fine || touches_coarse)))
    throw std::invalid_argument(
        "accepted composite tent stencil fails authenticated closure");
}

}  // namespace

std::uint64_t composite_coupling_topology_fingerprint_r2(
    const CompositePoissonR2& topology) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const CompositePatchGeometryR2& geometry = topology.geometry();
  hash_scalar(&hash, geometry.coarse_n_x);
  hash_scalar(&hash, geometry.coarse_n_y);
  hash_scalar(&hash, geometry.coarse_delta_x);
  hash_scalar(&hash, geometry.coarse_delta_y);
  hash_scalar(&hash, geometry.physical_origin_x);
  hash_scalar(&hash, geometry.physical_origin_y);
  hash_scalar(&hash, geometry.patch_begin_x);
  hash_scalar(&hash, geometry.patch_end_x);
  hash_scalar(&hash, geometry.patch_begin_y);
  hash_scalar(&hash, geometry.patch_end_y);
  hash_scalar(&hash, geometry.fixed_boundary_layers);
  hash_scalar(&hash, geometry.proper_nesting_cells);
  hash_scalar(&hash, geometry.refinement_ratio);
  const std::size_t count = topology.leaf_count();
  hash_scalar(&hash, count);
  for (const CompositeLeafCell& leaf : topology.leaves()) {
    const std::uint8_t level = static_cast<std::uint8_t>(leaf.level);
    hash_scalar(&hash, level);
    hash_scalar(&hash, leaf.level_i);
    hash_scalar(&hash, leaf.level_j);
    hash_scalar(&hash, leaf.center_x);
    hash_scalar(&hash, leaf.center_y);
    hash_scalar(&hash, leaf.delta_x);
    hash_scalar(&hash, leaf.delta_y);
    hash_scalar(&hash, leaf.area);
    hash_scalar(&hash, leaf.fixed_dirichlet);
    hash_scalar(&hash, leaf.dof);
  }
  return hash;
}

CompositeTentStencilSetR2 build_composite_tent_stencils_r2(
    const CompositePoissonR2& topology,
    const std::vector<CompositeTentParticleR2>& particles,
    double closure_tolerance) {
  if (!finite(closure_tolerance) || !(closure_tolerance > 0.0) ||
      closure_tolerance > 1.0e-6)
    throw std::invalid_argument(
        "composite tent closure tolerance must be in (0, 1e-6]");

  std::vector<CompositeTentParticleR2> ordered = particles;
  for (const CompositeTentParticleR2& particle : ordered)
    validate_finite_particle(particle);
  std::sort(ordered.begin(), ordered.end(),
            [](const CompositeTentParticleR2& a,
               const CompositeTentParticleR2& b) {
              return a.stable_id < b.stable_id;
            });
  for (std::size_t i = 1U; i < ordered.size(); ++i)
    if (ordered[i - 1U].stable_id == ordered[i].stable_id)
      throw std::invalid_argument(
          "duplicate stable ID in composite tent particles");

  CompositeTentStencilSetR2 result;
  result.topology_fingerprint =
      composite_coupling_topology_fingerprint_r2(topology);
  result.leaf_count = topology.leaf_count();
  result.closure_tolerance = closure_tolerance;
  result.stencils.reserve(ordered.size());

  const CompositePatchGeometryR2& geometry = topology.geometry();
  const double domain_begin_x = geometry.physical_origin_x;
  const double domain_begin_y = geometry.physical_origin_y;
  const double domain_end_x = domain_begin_x +
      static_cast<double>(geometry.coarse_n_x) * geometry.coarse_delta_x;
  const double domain_end_y = domain_begin_y +
      static_cast<double>(geometry.coarse_n_y) * geometry.coarse_delta_y;

  for (const CompositeTentParticleR2& particle : ordered) {
    CompositeTentStencilR2 stencil;
    stencil.particle = particle;
    stencil.owner_leaf = owner_leaf(topology, particle);
    if (stencil.owner_leaf == kCompositeNoIndex) {
      stencil.status = CompositeTentStatus::OutsidePhysicalDomain;
      result.stencils.push_back(stencil);
      continue;
    }
    if (topology.leaves()[stencil.owner_leaf].fixed_dirichlet) {
      stencil.status = CompositeTentStatus::FixedBoundaryCenter;
      result.stencils.push_back(stencil);
      continue;
    }

    const double support_begin_x =
        particle.position_x - particle.support_half_width_x;
    const double support_end_x =
        particle.position_x + particle.support_half_width_x;
    const double support_begin_y =
        particle.position_y - particle.support_half_width_y;
    const double support_end_y =
        particle.position_y + particle.support_half_width_y;
    if (!finite(support_begin_x) || !finite(support_end_x) ||
        !finite(support_begin_y) || !finite(support_end_y))
      throw std::overflow_error(
          "composite tent physical support is not representable");
    if (support_begin_x < domain_begin_x || support_end_x > domain_end_x ||
        support_begin_y < domain_begin_y || support_end_y > domain_end_y) {
      stencil.status = CompositeTentStatus::SupportOutsidePhysicalDomain;
      result.stencils.push_back(stencil);
      continue;
    }

    const double patch_begin_x = domain_begin_x +
        static_cast<double>(geometry.patch_begin_x) *
            geometry.coarse_delta_x;
    const double patch_end_x = domain_begin_x +
        static_cast<double>(geometry.patch_end_x) *
            geometry.coarse_delta_x;
    const double patch_begin_y = domain_begin_y +
        static_cast<double>(geometry.patch_begin_y) *
            geometry.coarse_delta_y;
    const double patch_end_y = domain_begin_y +
        static_cast<double>(geometry.patch_end_y) *
            geometry.coarse_delta_y;
    const CompositeLevel owner_level =
        topology.leaves()[stencil.owner_leaf].level;
    const bool support_inside_patch =
        support_begin_x >= patch_begin_x && support_end_x <= patch_end_x &&
        support_begin_y >= patch_begin_y && support_end_y <= patch_end_y;
    // Touching the patch boundary only at the zero-measure edge of a tent is
    // disjoint for deposition purposes.  Any positive-area overlap requires
    // the patch to grow before this source can be accepted on one level.
    const bool support_disjoint_from_patch =
        support_end_x <= patch_begin_x || support_begin_x >= patch_end_x ||
        support_end_y <= patch_begin_y || support_begin_y >= patch_end_y;
    if ((owner_level == CompositeLevel::Fine && !support_inside_patch) ||
        (owner_level == CompositeLevel::Coarse &&
         !support_disjoint_from_patch)) {
      stencil.status =
          CompositeTentStatus::SupportCrossesRefinementBoundary;
      result.stencils.push_back(stencil);
      continue;
    }

    bool fixed_support = false;
    const CandidateIndexRange coarse_x = candidate_index_range(
        support_begin_x, support_end_x, domain_begin_x,
        geometry.coarse_delta_x, geometry.coarse_n_x);
    const CandidateIndexRange coarse_y = candidate_index_range(
        support_begin_y, support_end_y, domain_begin_y,
        geometry.coarse_delta_y, geometry.coarse_n_y);
    const double fine_spacing_x = 0.5 * geometry.coarse_delta_x;
    const double fine_spacing_y = 0.5 * geometry.coarse_delta_y;
    const double fine_origin_x = domain_begin_x +
        static_cast<double>(geometry.patch_begin_x) *
            geometry.coarse_delta_x;
    const double fine_origin_y = domain_begin_y +
        static_cast<double>(geometry.patch_begin_y) *
            geometry.coarse_delta_y;
    const CandidateIndexRange fine_x = candidate_index_range(
        support_begin_x, support_end_x, fine_origin_x,
        fine_spacing_x, topology.fine_n_x());
    const CandidateIndexRange fine_y = candidate_index_range(
        support_begin_y, support_end_y, fine_origin_y,
        fine_spacing_y, topology.fine_n_y());
    const std::size_t coarse_candidates = checked_size_product(
        coarse_x.end - coarse_x.begin, coarse_y.end - coarse_y.begin,
        "composite coarse candidate count");
    const std::size_t fine_candidates = checked_size_product(
        fine_x.end - fine_x.begin, fine_y.end - fine_y.begin,
        "composite fine candidate count");
    const std::size_t candidate_count =
        owner_level == CompositeLevel::Fine ? fine_candidates
                                            : coarse_candidates;
    stencil.contributions.reserve(
        std::min(candidate_count, topology.leaf_count()));

    const auto append_leaf = [&](std::size_t leaf_index) {
      if (leaf_index == kCompositeNoIndex) return;
      const CompositeLeafCell& leaf = topology.leaves()[leaf_index];
      double begin_x = 0.0;
      double end_x = 0.0;
      double begin_y = 0.0;
      double end_y = 0.0;
      leaf_bounds(geometry, leaf, &begin_x, &end_x, &begin_y, &end_y);
      const Integral1D x = integrate_tent_1d(
          begin_x, end_x, particle.position_x,
          particle.support_half_width_x);
      if (!(x.weight > 0.0)) return;
      const Integral1D y = integrate_tent_1d(
          begin_y, end_y, particle.position_y,
          particle.support_half_width_y);
      if (!(y.weight > 0.0)) return;

      CompositeTentContributionR2 contribution;
      contribution.leaf = leaf_index;
      contribution.weight = x.weight * y.weight;
      contribution.derivative_x = x.derivative * y.weight;
      contribution.derivative_y = x.weight * y.derivative;
      contribution.shape_first_x = x.first_moment * y.weight;
      contribution.shape_first_y = x.weight * y.first_moment;
      if (!finite(contribution.weight) ||
          !finite(contribution.derivative_x) ||
          !finite(contribution.derivative_y) ||
          !finite(contribution.shape_first_x) ||
          !finite(contribution.shape_first_y) ||
          !(contribution.weight > 0.0))
        throw std::overflow_error(
            "composite two-dimensional tent integral overflowed");
      stencil.contributions.push_back(contribution);
      checked_accumulate(&stencil.weight_sum, contribution.weight,
                         "composite stencil weight");
      checked_accumulate(&stencil.derivative_x_sum,
                         contribution.derivative_x,
                         "composite stencil x derivative");
      checked_accumulate(&stencil.derivative_y_sum,
                         contribution.derivative_y,
                         "composite stencil y derivative");
      checked_accumulate(&stencil.shape_first_x_sum,
                         contribution.shape_first_x,
                         "composite stencil x moment");
      checked_accumulate(&stencil.shape_first_y_sum,
                         contribution.shape_first_y,
                         "composite stencil y moment");
      stencil.touches_coarse = stencil.touches_coarse ||
          leaf.level == CompositeLevel::Coarse;
      stencil.touches_fine = stencil.touches_fine ||
          leaf.level == CompositeLevel::Fine;
      fixed_support = fixed_support || leaf.fixed_dirichlet;
    };
    if (owner_level == CompositeLevel::Coarse) {
      for (std::size_t i = coarse_x.begin; i < coarse_x.end; ++i)
        for (std::size_t j = coarse_y.begin; j < coarse_y.end; ++j)
          append_leaf(topology.coarse_leaf(i, j));
    } else {
      for (std::size_t i = fine_x.begin; i < fine_x.end; ++i)
        for (std::size_t j = fine_y.begin; j < fine_y.end; ++j)
          append_leaf(topology.fine_leaf(i, j));
    }
    stencil.crosses_coarse_fine_interface =
        stencil.touches_coarse && stencil.touches_fine;

    if (fixed_support) {
      stencil.status = CompositeTentStatus::FixedBoundarySupport;
      stencil.contributions.clear();
      result.stencils.push_back(stencil);
      continue;
    }

    const double weight_defect = std::fabs(stencil.weight_sum - 1.0);
    const double derivative_defect = std::max(
        std::fabs(stencil.derivative_x_sum) *
            particle.support_half_width_x,
        std::fabs(stencil.derivative_y_sum) *
            particle.support_half_width_y);
    const double first_x_scale = std::max(
        1.0, std::max(std::fabs(particle.position_x),
                      particle.support_half_width_x));
    const double first_y_scale = std::max(
        1.0, std::max(std::fabs(particle.position_y),
                      particle.support_half_width_y));
    const double first_defect = std::max(
        std::fabs(stencil.shape_first_x_sum - particle.position_x) /
            first_x_scale,
        std::fabs(stencil.shape_first_y_sum - particle.position_y) /
            first_y_scale);
    if (stencil.contributions.empty() ||
        weight_defect > closure_tolerance ||
        derivative_defect > closure_tolerance ||
        first_defect > closure_tolerance) {
      stencil.status = CompositeTentStatus::IncompleteLeafPartition;
      stencil.contributions.clear();
    } else {
      stencil.status = CompositeTentStatus::Accepted;
    }
    result.stencils.push_back(stencil);
  }
  return result;
}

CompositeDepositResultR2 deposit_composite_tent_source_once_r2(
    const CompositePoissonR2& topology,
    const CompositeTentStencilSetR2& stencil_set) {
  verify_stencil_topology(topology, stencil_set);
  CompositeDepositResultR2 result;
  result.integrated_source.assign(topology.leaf_count(), 0.0);
  result.telemetry.input_particles = stencil_set.stencils.size();

  std::uint64_t previous_id = 0U;
  bool have_previous_id = false;
  for (const CompositeTentStencilR2& stencil : stencil_set.stencils) {
    validate_stencil(topology, stencil_set, stencil);
    if (have_previous_id && stencil.particle.stable_id <= previous_id)
      throw std::invalid_argument(
          "composite tent stencil set is not in unique stable-ID order");
    previous_id = stencil.particle.stable_id;
    have_previous_id = true;
    add_particle_moments(&result.telemetry.input, stencil.particle);

    const double weight_defect = std::fabs(stencil.weight_sum - 1.0);
    const double derivative_defect = std::max(
        std::fabs(stencil.derivative_x_sum) *
            stencil.particle.support_half_width_x,
        std::fabs(stencil.derivative_y_sum) *
            stencil.particle.support_half_width_y);
    const double first_x_scale = std::max(
        1.0, std::max(std::fabs(stencil.particle.position_x),
                      stencil.particle.support_half_width_x));
    const double first_y_scale = std::max(
        1.0, std::max(std::fabs(stencil.particle.position_y),
                      stencil.particle.support_half_width_y));
    const double first_defect = std::max(
        std::fabs(stencil.shape_first_x_sum -
                  stencil.particle.position_x) / first_x_scale,
        std::fabs(stencil.shape_first_y_sum -
                  stencil.particle.position_y) / first_y_scale);
    // Outside/fixed-center rejections have no evaluated leaf partition.  Do
    // not turn their intentionally empty stencils into a spurious unit closure
    // defect.  Fixed-support and incomplete-partition rejections retain their
    // diagnostic sums even though their contributions were fail-closed.
    if (stencil.status == CompositeTentStatus::Accepted ||
        stencil.status == CompositeTentStatus::FixedBoundarySupport ||
        stencil.status == CompositeTentStatus::IncompleteLeafPartition) {
      result.telemetry.maximum_weight_closure_defect = std::max(
          result.telemetry.maximum_weight_closure_defect, weight_defect);
      result.telemetry.maximum_scaled_derivative_closure_defect = std::max(
          result.telemetry.maximum_scaled_derivative_closure_defect,
          derivative_defect);
      result.telemetry.maximum_scaled_first_moment_defect = std::max(
          result.telemetry.maximum_scaled_first_moment_defect, first_defect);
    }

    if (stencil.status != CompositeTentStatus::Accepted) {
      if (outside_status(stencil.status)) {
        ++result.telemetry.outside_particles;
        add_particle_moments(&result.telemetry.outside, stencil.particle);
      } else {
        ++result.telemetry.rejected_particles;
        add_particle_moments(&result.telemetry.rejected, stencil.particle);
      }
      continue;
    }
    ++result.telemetry.accepted_particles;
    add_particle_moments(&result.telemetry.accepted, stencil.particle);
    for (const CompositeTentContributionR2& contribution :
         stencil.contributions) {
      if (contribution.leaf >= topology.leaf_count() ||
          topology.leaves()[contribution.leaf].fixed_dirichlet)
        throw std::invalid_argument(
            "accepted composite tent stencil targets an invalid leaf");
      const double charge =
          checked_product(stencil.particle.signed_charge,
                          contribution.weight,
                          "composite deposited charge");
      const double absolute_charge = checked_product(
          std::fabs(stencil.particle.signed_charge), contribution.weight,
          "composite deposited absolute charge");
      checked_accumulate(&result.integrated_source[contribution.leaf], charge,
                         "composite integrated source");
      checked_accumulate(
          &result.telemetry.deposited_exact_shape.signed_charge, charge,
          "composite exact-shape signed charge");
      checked_accumulate(
          &result.telemetry.deposited_exact_shape.absolute_charge,
          absolute_charge, "composite exact-shape absolute charge");
      checked_accumulate(
          &result.telemetry.deposited_exact_shape.signed_first_x,
          checked_product(stencil.particle.signed_charge,
                          contribution.shape_first_x,
                          "composite exact-shape x moment"),
          "composite exact-shape x moment");
      checked_accumulate(
          &result.telemetry.deposited_exact_shape.signed_first_y,
          checked_product(stencil.particle.signed_charge,
                          contribution.shape_first_y,
                          "composite exact-shape y moment"),
          "composite exact-shape y moment");
      checked_accumulate(
          &result.telemetry.deposited_cell_centres.signed_charge, charge,
          "composite cell-centre signed charge");
      checked_accumulate(
          &result.telemetry.deposited_cell_centres.absolute_charge,
          absolute_charge, "composite cell-centre absolute charge");
      checked_accumulate(
          &result.telemetry.deposited_cell_centres.signed_first_x,
          checked_product(
              charge, topology.leaves()[contribution.leaf].center_x,
              "composite cell-centre x moment"),
          "composite cell-centre x moment");
      checked_accumulate(
          &result.telemetry.deposited_cell_centres.signed_first_y,
          checked_product(
              charge, topology.leaves()[contribution.leaf].center_y,
              "composite cell-centre y moment"),
          "composite cell-centre y moment");
    }
  }
  for (double source : result.integrated_source) {
    if (!finite(source))
      throw std::overflow_error(
          "composite integrated source accumulation overflowed");
    checked_accumulate(&result.telemetry.integrated_source_charge, source,
                       "composite integrated-source charge");
  }
  const CompositeChargeMomentsR2* const moment_sets[] = {
      &result.telemetry.input, &result.telemetry.accepted,
      &result.telemetry.rejected, &result.telemetry.outside,
      &result.telemetry.deposited_exact_shape,
      &result.telemetry.deposited_cell_centres};
  for (const CompositeChargeMomentsR2* moments : moment_sets)
    if (!finite(moments->signed_charge) ||
        !finite(moments->absolute_charge) ||
        !finite(moments->signed_first_x) ||
        !finite(moments->signed_first_y))
      throw std::overflow_error(
          "composite deposited-moment telemetry is non-finite");
  if (!finite(result.telemetry.integrated_source_charge))
    throw std::overflow_error(
        "composite integrated-source telemetry is non-finite");
  return result;
}

CompositeAdjointGatherResultR2 gather_composite_tent_adjoint_r2(
    const CompositePoissonR2& topology,
    const CompositeTentStencilSetR2& stencil_set,
    const std::vector<double>& unknown_potential,
    const std::vector<double>& dirichlet_leaf_values) {
  verify_stencil_topology(topology, stencil_set);
  if (unknown_potential.size() != topology.unknown_count() ||
      dirichlet_leaf_values.size() != topology.leaf_count())
    throw std::invalid_argument(
        "composite adjoint gather potential has wrong shape");
  for (double value : unknown_potential)
    if (!finite(value))
      throw std::invalid_argument(
          "composite adjoint gather potential is non-finite");
  for (double value : dirichlet_leaf_values)
    if (!finite(value))
      throw std::invalid_argument(
          "composite adjoint gather Dirichlet data is non-finite");

  CompositeAdjointGatherResultR2 result;
  result.samples.reserve(stencil_set.stencils.size());
  std::uint64_t previous_id = 0U;
  bool have_previous_id = false;
  for (const CompositeTentStencilR2& stencil : stencil_set.stencils) {
    validate_stencil(topology, stencil_set, stencil);
    if (have_previous_id && stencil.particle.stable_id <= previous_id)
      throw std::invalid_argument(
          "composite gather stencils are not in unique stable-ID order");
    previous_id = stencil.particle.stable_id;
    have_previous_id = true;
    CompositeAdjointFieldSampleR2 sample;
    sample.stable_id = stencil.particle.stable_id;
    sample.status = stencil.status;
    sample.owner_leaf = stencil.owner_leaf;
    if (stencil.status == CompositeTentStatus::Accepted) {
      for (const CompositeTentContributionR2& contribution :
           stencil.contributions) {
        if (contribution.leaf >= topology.leaf_count())
          throw std::invalid_argument(
              "composite gather stencil has invalid leaf");
        const CompositeLeafCell& leaf = topology.leaves()[contribution.leaf];
        const double phi = leaf.dof == kCompositeNoIndex
            ? dirichlet_leaf_values[contribution.leaf]
            : unknown_potential[leaf.dof];
        sample.interpolated_potential += contribution.weight * phi;
        sample.electric_x -= contribution.derivative_x * phi;
        sample.electric_y -= contribution.derivative_y * phi;
      }
      if (!finite(sample.interpolated_potential) ||
          !finite(sample.electric_x) || !finite(sample.electric_y))
        throw std::overflow_error("composite adjoint gather overflowed");
    }
    result.samples.push_back(sample);
  }
  return result;
}

}  // namespace adaptive_grid
}  // namespace feebl
