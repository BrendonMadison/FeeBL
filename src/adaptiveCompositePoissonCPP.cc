#include "adaptiveCompositePoissonCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace feebl {
namespace adaptive_grid {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* label) {
  if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error(std::string(label) + " overflows size_t");
  return a * b;
}

std::size_t checked_sum(std::size_t a, std::size_t b, const char* label) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error(std::string(label) + " overflows size_t");
  return a + b;
}

bool finite(double value) { return std::isfinite(value); }

double deterministic_dot(const std::vector<double>& a,
                         const std::vector<double>& b) {
  if (a.size() != b.size())
    throw std::invalid_argument("composite dot-product size mismatch");
  double sum = 0.0;
  // Deliberately serial and in index order.  Production parallel reductions
  // must reproduce or explicitly version this arithmetic contract.
  for (std::size_t i = 0U; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace

CompositePoissonR2::CompositePoissonR2(
    const CompositePatchGeometryR2& geometry)
    : geometry_(geometry) {
  if (geometry_.refinement_ratio != 2U)
    throw std::invalid_argument(
        "composite Poisson requires refinement ratio exactly two");
  if (geometry_.coarse_n_x < 5U || geometry_.coarse_n_y < 5U ||
      !finite(geometry_.coarse_delta_x) ||
      !finite(geometry_.coarse_delta_y) ||
      !finite(geometry_.physical_origin_x) ||
      !finite(geometry_.physical_origin_y) ||
      !(geometry_.coarse_delta_x > 0.0) ||
      !(geometry_.coarse_delta_y > 0.0) ||
      geometry_.fixed_boundary_layers == 0U)
    throw std::invalid_argument("invalid composite coarse geometry");
  if (geometry_.patch_begin_x >= geometry_.patch_end_x ||
      geometry_.patch_begin_y >= geometry_.patch_end_y ||
      geometry_.patch_end_x > geometry_.coarse_n_x ||
      geometry_.patch_end_y > geometry_.coarse_n_y)
    throw std::invalid_argument("invalid composite patch bounds");

  const std::size_t margin = checked_sum(
      geometry_.fixed_boundary_layers, geometry_.proper_nesting_cells,
      "composite proper-nesting margin");
  if (margin >= geometry_.coarse_n_x ||
      margin >= geometry_.coarse_n_y ||
      geometry_.patch_begin_x < margin ||
      geometry_.patch_begin_y < margin ||
      geometry_.patch_end_x > geometry_.coarse_n_x - margin ||
      geometry_.patch_end_y > geometry_.coarse_n_y - margin)
    throw std::invalid_argument(
        "composite patch violates proper nesting or Dirichlet margin");
  // A quadratic one-sided interface audit needs three cell-centred samples
  // normal to each side.  The solver itself remains valid for narrower
  // properly nested layouts; the diagnostic is explicitly unavailable and
  // its strict public entry point fails closed for such layouts.
  interface_flux_audit_available_ =
      geometry_.patch_end_x - geometry_.patch_begin_x >= 2U &&
      geometry_.patch_end_y - geometry_.patch_begin_y >= 2U &&
      geometry_.patch_begin_x >= 3U && geometry_.patch_begin_y >= 3U &&
      geometry_.patch_end_x <= geometry_.coarse_n_x - 3U &&
      geometry_.patch_end_y <= geometry_.coarse_n_y - 3U;

  fine_n_x_ = checked_product(
      geometry_.patch_end_x - geometry_.patch_begin_x, 2U,
      "composite fine n_x");
  fine_n_y_ = checked_product(
      geometry_.patch_end_y - geometry_.patch_begin_y, 2U,
      "composite fine n_y");
  build_cells();
  build_faces();
  validate_topology_and_build_diagonal();
}

void CompositePoissonR2::build_cells() {
  const std::size_t coarse_cells = checked_product(
      geometry_.coarse_n_x, geometry_.coarse_n_y,
      "composite coarse cell count");
  const std::size_t fine_cells = checked_product(
      fine_n_x_, fine_n_y_, "composite fine cell count");
  coarse_leaf_map_.assign(coarse_cells, kCompositeNoIndex);
  fine_leaf_map_.assign(fine_cells, kCompositeNoIndex);

  const auto covered = [this](std::size_t i, std::size_t j) {
    return i >= geometry_.patch_begin_x &&
           i < geometry_.patch_end_x &&
           j >= geometry_.patch_begin_y &&
           j < geometry_.patch_end_y;
  };
  const auto fixed = [this](std::size_t i, std::size_t j) {
    const std::size_t layers = geometry_.fixed_boundary_layers;
    return i < layers || j < layers ||
           i >= geometry_.coarse_n_x - layers ||
           j >= geometry_.coarse_n_y - layers;
  };

  leaves_.reserve(coarse_cells + fine_cells);
  for (std::size_t i = 0U; i < geometry_.coarse_n_x; ++i)
    for (std::size_t j = 0U; j < geometry_.coarse_n_y; ++j) {
      if (covered(i, j)) continue;
      CompositeLeafCell leaf;
      leaf.level = CompositeLevel::Coarse;
      leaf.level_i = i;
      leaf.level_j = j;
      leaf.delta_x = geometry_.coarse_delta_x;
      leaf.delta_y = geometry_.coarse_delta_y;
      leaf.area = leaf.delta_x * leaf.delta_y;
      leaf.center_x = geometry_.physical_origin_x +
                      (static_cast<double>(i) + 0.5) * leaf.delta_x;
      leaf.center_y = geometry_.physical_origin_y +
                      (static_cast<double>(j) + 0.5) * leaf.delta_y;
      leaf.fixed_dirichlet = fixed(i, j);
      if (!leaf.fixed_dirichlet) {
        leaf.dof = dof_to_leaf_.size();
        dof_to_leaf_.push_back(leaves_.size());
      }
      coarse_leaf_map_[i * geometry_.coarse_n_y + j] = leaves_.size();
      leaves_.push_back(leaf);
    }

  const double fine_delta_x = 0.5 * geometry_.coarse_delta_x;
  const double fine_delta_y = 0.5 * geometry_.coarse_delta_y;
  const double fine_origin_x = geometry_.physical_origin_x +
      static_cast<double>(geometry_.patch_begin_x) *
          geometry_.coarse_delta_x;
  const double fine_origin_y = geometry_.physical_origin_y +
      static_cast<double>(geometry_.patch_begin_y) *
          geometry_.coarse_delta_y;
  for (std::size_t i = 0U; i < fine_n_x_; ++i)
    for (std::size_t j = 0U; j < fine_n_y_; ++j) {
      CompositeLeafCell leaf;
      leaf.level = CompositeLevel::Fine;
      leaf.level_i = i;
      leaf.level_j = j;
      leaf.delta_x = fine_delta_x;
      leaf.delta_y = fine_delta_y;
      leaf.area = fine_delta_x * fine_delta_y;
      leaf.center_x = fine_origin_x +
                      (static_cast<double>(i) + 0.5) * fine_delta_x;
      leaf.center_y = fine_origin_y +
                      (static_cast<double>(j) + 0.5) * fine_delta_y;
      leaf.fixed_dirichlet = false;
      leaf.dof = dof_to_leaf_.size();
      dof_to_leaf_.push_back(leaves_.size());
      fine_leaf_map_[i * fine_n_y_ + j] = leaves_.size();
      leaves_.push_back(leaf);
    }
}

std::size_t CompositePoissonR2::coarse_leaf(
    std::size_t i, std::size_t j) const {
  if (i >= geometry_.coarse_n_x || j >= geometry_.coarse_n_y)
    throw std::out_of_range("composite coarse leaf index out of range");
  return coarse_leaf_map_[i * geometry_.coarse_n_y + j];
}

std::size_t CompositePoissonR2::fine_leaf(
    std::size_t i, std::size_t j) const {
  if (i >= fine_n_x_ || j >= fine_n_y_)
    throw std::out_of_range("composite fine leaf index out of range");
  return fine_leaf_map_[i * fine_n_y_ + j];
}

void CompositePoissonR2::add_face(
    std::size_t leaf_a, std::size_t leaf_b,
    CompositeFaceAxis axis, CompositeFaceKind kind,
    double area, double center_distance) {
  if (leaf_a >= leaves_.size() || leaf_b >= leaves_.size() ||
      leaf_a == leaf_b || !finite(area) || !finite(center_distance) ||
      !(area > 0.0) || !(center_distance > 0.0))
    throw std::logic_error("invalid composite face construction");
  CompositeFace face;
  face.leaf_a = leaf_a;
  face.leaf_b = leaf_b;
  face.axis = axis;
  face.kind = kind;
  face.area = area;
  face.center_distance = center_distance;
  face.transmissibility = area / center_distance;
  if (!finite(face.transmissibility) || !(face.transmissibility > 0.0))
    throw std::logic_error("invalid composite face transmissibility");
  faces_.push_back(face);
  if (kind == CompositeFaceKind::CoarseFine) ++interface_face_count_;
}

void CompositePoissonR2::add_interface_face(
    std::size_t leaf_a, std::size_t leaf_b,
    std::size_t coarse_tangential_neighbor,
    double coarse_primary_sign, CompositeFaceAxis axis,
    double area, double center_distance) {
  if (leaf_a >= leaves_.size() || leaf_b >= leaves_.size() ||
      coarse_tangential_neighbor >= leaves_.size() ||
      leaf_a == leaf_b || leaf_a == coarse_tangential_neighbor ||
      leaf_b == coarse_tangential_neighbor ||
      (coarse_primary_sign != 1.0 && coarse_primary_sign != -1.0) ||
      !finite(area) || !finite(center_distance) ||
      !(area > 0.0) || !(center_distance > 0.0))
    throw std::logic_error("invalid multipoint interface construction");

  const std::size_t coarse_primary = coarse_primary_sign > 0.0
      ? leaf_a : leaf_b;
  const std::size_t fine = coarse_primary_sign > 0.0
      ? leaf_b : leaf_a;
  if (leaves_[coarse_primary].level != CompositeLevel::Coarse ||
      leaves_[coarse_tangential_neighbor].level != CompositeLevel::Coarse ||
      leaves_[fine].level != CompositeLevel::Fine)
    throw std::logic_error("multipoint interface levels are inconsistent");

  CompositeFace face;
  face.leaf_a = leaf_a;
  face.leaf_b = leaf_b;
  face.leaf_c = coarse_tangential_neighbor;
  face.axis = axis;
  face.kind = CompositeFaceKind::CoarseFine;
  face.area = area;
  face.center_distance = center_distance;
  face.transmissibility = area / center_distance;
  if (coarse_primary_sign > 0.0) {
    face.coefficient_a = 0.75;
    face.coefficient_b = -1.0;
    face.coefficient_c = 0.25;
  } else {
    face.coefficient_a = 1.0;
    face.coefficient_b = -0.75;
    face.coefficient_c = -0.25;
  }
  if (!finite(face.transmissibility) || !(face.transmissibility > 0.0))
    throw std::logic_error("invalid multipoint interface transmissibility");
  faces_.push_back(face);
  ++interface_face_count_;
}

void CompositePoissonR2::build_faces() {
  const double hx = geometry_.coarse_delta_x;
  const double hy = geometry_.coarse_delta_y;
  const double fx = 0.5 * hx;
  const double fy = 0.5 * hy;

  // Coarse same-level faces; covered cells are absent by construction.
  for (std::size_t i = 0U; i + 1U < geometry_.coarse_n_x; ++i)
    for (std::size_t j = 0U; j < geometry_.coarse_n_y; ++j) {
      const std::size_t a = coarse_leaf(i, j);
      const std::size_t b = coarse_leaf(i + 1U, j);
      if (a != kCompositeNoIndex && b != kCompositeNoIndex)
        add_face(a, b, CompositeFaceAxis::XNormal,
                 CompositeFaceKind::CoarseCoarse, hy, hx);
    }
  for (std::size_t i = 0U; i < geometry_.coarse_n_x; ++i)
    for (std::size_t j = 0U; j + 1U < geometry_.coarse_n_y; ++j) {
      const std::size_t a = coarse_leaf(i, j);
      const std::size_t b = coarse_leaf(i, j + 1U);
      if (a != kCompositeNoIndex && b != kCompositeNoIndex)
        add_face(a, b, CompositeFaceAxis::YNormal,
                 CompositeFaceKind::CoarseCoarse, hx, hy);
    }

  // Fine same-level faces.
  for (std::size_t i = 0U; i + 1U < fine_n_x_; ++i)
    for (std::size_t j = 0U; j < fine_n_y_; ++j)
      add_face(fine_leaf(i, j), fine_leaf(i + 1U, j),
               CompositeFaceAxis::XNormal,
               CompositeFaceKind::FineFine, fy, fx);
  for (std::size_t i = 0U; i < fine_n_x_; ++i)
    for (std::size_t j = 0U; j + 1U < fine_n_y_; ++j)
      add_face(fine_leaf(i, j), fine_leaf(i, j + 1U),
               CompositeFaceAxis::YNormal,
               CompositeFaceKind::FineFine, fx, fy);

  // Each coarse interface face is split into its two fine subfaces.  The
  // center distance is H/2+h/2.  A single transmissibility is shared by the
  // coarse and fine residual rows, making the algebraic operator symmetric.
  const double interface_dx = 0.5 * hx + 0.5 * fx;
  const double interface_dy = 0.5 * hy + 0.5 * fy;
  const std::size_t patch_height =
      geometry_.patch_end_y - geometry_.patch_begin_y;
  const std::size_t patch_width =
      geometry_.patch_end_x - geometry_.patch_begin_x;

  for (std::size_t offset = 0U; offset < patch_height; ++offset) {
    const std::size_t coarse_j = geometry_.patch_begin_y + offset;
    const std::size_t fine_j = 2U * offset;
    const std::size_t left_coarse = coarse_leaf(
        geometry_.patch_begin_x - 1U, coarse_j);
    const std::size_t right_coarse = coarse_leaf(
        geometry_.patch_end_x, coarse_j);
    for (std::size_t subface = 0U; subface < 2U; ++subface) {
      const std::size_t neighbor_j = subface == 0U
          ? coarse_j - 1U : coarse_j + 1U;
      // Left orientation is coarse -> fine (increasing x).
      add_interface_face(
          left_coarse, fine_leaf(0U, fine_j + subface),
          coarse_leaf(geometry_.patch_begin_x - 1U, neighbor_j),
          1.0, CompositeFaceAxis::XNormal, fy, interface_dx);
      // Right orientation is fine -> coarse (increasing x).
      add_interface_face(
          fine_leaf(fine_n_x_ - 1U, fine_j + subface), right_coarse,
          coarse_leaf(geometry_.patch_end_x, neighbor_j),
          -1.0, CompositeFaceAxis::XNormal, fy, interface_dx);
    }
  }
  for (std::size_t offset = 0U; offset < patch_width; ++offset) {
    const std::size_t coarse_i = geometry_.patch_begin_x + offset;
    const std::size_t fine_i = 2U * offset;
    const std::size_t lower_coarse = coarse_leaf(
        coarse_i, geometry_.patch_begin_y - 1U);
    const std::size_t upper_coarse = coarse_leaf(
        coarse_i, geometry_.patch_end_y);
    for (std::size_t subface = 0U; subface < 2U; ++subface) {
      const std::size_t neighbor_i = subface == 0U
          ? coarse_i - 1U : coarse_i + 1U;
      // Lower orientation is coarse -> fine (increasing y).
      add_interface_face(
          lower_coarse, fine_leaf(fine_i + subface, 0U),
          coarse_leaf(neighbor_i, geometry_.patch_begin_y - 1U),
          1.0, CompositeFaceAxis::YNormal, fx, interface_dy);
      // Upper orientation is fine -> coarse (increasing y).
      add_interface_face(
          fine_leaf(fine_i + subface, fine_n_y_ - 1U), upper_coarse,
          coarse_leaf(neighbor_i, geometry_.patch_end_y),
          -1.0, CompositeFaceAxis::YNormal, fx, interface_dy);
    }
  }
}

void CompositePoissonR2::validate_topology_and_build_diagonal() {
  if (leaves_.empty() || dof_to_leaf_.empty() || faces_.empty())
    throw std::logic_error("empty composite topology");

  std::vector<unsigned char> seen_dof(dof_to_leaf_.size(), 0U);
  for (std::size_t leaf = 0U; leaf < leaves_.size(); ++leaf) {
    const CompositeLeafCell& cell = leaves_[leaf];
    if (!finite(cell.center_x) || !finite(cell.center_y) ||
        !finite(cell.delta_x) || !finite(cell.delta_y) ||
        !finite(cell.area) || !(cell.delta_x > 0.0) ||
        !(cell.delta_y > 0.0) || !(cell.area > 0.0))
      throw std::logic_error("non-finite composite leaf geometry");
    if (cell.fixed_dirichlet) {
      if (cell.dof != kCompositeNoIndex)
        throw std::logic_error("Dirichlet composite leaf owns a DOF");
    } else {
      if (cell.dof >= dof_to_leaf_.size() ||
          dof_to_leaf_[cell.dof] != leaf || seen_dof[cell.dof])
        throw std::logic_error("duplicate or inconsistent composite DOF");
      seen_dof[cell.dof] = 1U;
    }
  }
  if (std::find(seen_dof.begin(), seen_dof.end(), 0U) != seen_dof.end())
    throw std::logic_error("composite DOF map has a hole");

  std::set<std::pair<std::size_t, std::size_t> > unique_faces;
  diagonal_.assign(dof_to_leaf_.size(), 0.0);
  std::vector<std::vector<std::size_t> > neighbors(dof_to_leaf_.size());
  std::vector<unsigned char> touches_boundary(dof_to_leaf_.size(), 0U);
  std::size_t counted_interfaces = 0U;
  for (const CompositeFace& face : faces_) {
    const std::pair<std::size_t, std::size_t> key =
        std::minmax(face.leaf_a, face.leaf_b);
    if (!unique_faces.insert(key).second)
      throw std::logic_error("duplicate composite face");
    if (face.leaf_a >= leaves_.size() || face.leaf_b >= leaves_.size() ||
        !finite(face.transmissibility) ||
        !(face.transmissibility > 0.0) ||
        !finite(face.coefficient_a) || !finite(face.coefficient_b) ||
        !finite(face.coefficient_c))
      throw std::logic_error("invalid stored composite face");
    const CompositeLeafCell& a = leaves_[face.leaf_a];
    const CompositeLeafCell& b = leaves_[face.leaf_b];
    if (face.kind == CompositeFaceKind::CoarseFine) {
      ++counted_interfaces;
      if (a.level == b.level || face.leaf_c >= leaves_.size() ||
          leaves_[face.leaf_c].level != CompositeLevel::Coarse ||
          std::fabs(face.coefficient_a + face.coefficient_b +
                    face.coefficient_c) > 4.0e-16 ||
          std::fabs(std::fabs(face.coefficient_c) - 0.25) > 0.0)
        throw std::logic_error("invalid multipoint coarse/fine face");
      const bool a_is_coarse = a.level == CompositeLevel::Coarse;
      const CompositeLeafCell& coarse = a_is_coarse ? a : b;
      const CompositeLeafCell& fine = a_is_coarse ? b : a;
      const CompositeLeafCell& tangent = leaves_[face.leaf_c];
      if ((a_is_coarse &&
           (face.coefficient_a != 0.75 ||
            face.coefficient_b != -1.0 ||
            face.coefficient_c != 0.25)) ||
          (!a_is_coarse &&
           (face.coefficient_a != 1.0 ||
            face.coefficient_b != -0.75 ||
            face.coefficient_c != -0.25)))
        throw std::logic_error(
            "multipoint interface orientation weights are inconsistent");
      const double coordinate_scale = std::max(
          1.0, std::max(std::fabs(fine.center_x) +
                            std::fabs(fine.center_y),
                        std::fabs(coarse.center_x) +
                            std::fabs(coarse.center_y)));
      const double coordinate_tolerance =
          64.0 * std::numeric_limits<double>::epsilon() * coordinate_scale;
      if (face.axis == CompositeFaceAxis::XNormal) {
        const std::size_t tangent_difference =
            tangent.level_j > coarse.level_j
                ? tangent.level_j - coarse.level_j
                : coarse.level_j - tangent.level_j;
        if (!(a.center_x < b.center_x) ||
            tangent.level_i != coarse.level_i ||
            tangent_difference != 1U ||
            std::fabs(fine.center_y -
                      (0.75 * coarse.center_y +
                       0.25 * tangent.center_y)) > coordinate_tolerance ||
            std::fabs((b.center_x - a.center_x) -
                      face.center_distance) > coordinate_tolerance)
          throw std::logic_error(
              "x-normal multipoint interface geometry is inconsistent");
      } else {
        const std::size_t tangent_difference =
            tangent.level_i > coarse.level_i
                ? tangent.level_i - coarse.level_i
                : coarse.level_i - tangent.level_i;
        if (!(a.center_y < b.center_y) ||
            tangent.level_j != coarse.level_j ||
            tangent_difference != 1U ||
            std::fabs(fine.center_x -
                      (0.75 * coarse.center_x +
                       0.25 * tangent.center_x)) > coordinate_tolerance ||
            std::fabs((b.center_y - a.center_y) -
                      face.center_distance) > coordinate_tolerance)
          throw std::logic_error(
              "y-normal multipoint interface geometry is inconsistent");
      }
    } else if (a.level != b.level) {
      throw std::logic_error("same-level face crosses composite levels");
    } else if (face.leaf_c != kCompositeNoIndex ||
               face.coefficient_a != 1.0 ||
               face.coefficient_b != -1.0 ||
               face.coefficient_c != 0.0) {
      throw std::logic_error("same-level face has multipoint coefficients");
    }

    const std::array<std::size_t, 3U> term_leaves = {{
        face.leaf_a, face.leaf_b, face.leaf_c}};
    const std::array<double, 3U> coefficients = {{
        face.coefficient_a, face.coefficient_b, face.coefficient_c}};
    bool term_touches_fixed = false;
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      if (term_leaves[i] >= leaves_.size())
        throw std::logic_error("composite term leaf is missing");
      const CompositeLeafCell& cell = leaves_[term_leaves[i]];
      if (cell.dof == kCompositeNoIndex) term_touches_fixed = true;
      else diagonal_[cell.dof] += face.transmissibility *
          coefficients[i] * coefficients[i];
    }
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      const CompositeLeafCell& cell_i = leaves_[term_leaves[i]];
      if (cell_i.dof == kCompositeNoIndex) continue;
      if (term_touches_fixed) touches_boundary[cell_i.dof] = 1U;
      for (std::size_t j = i + 1U; j < term_leaves.size(); ++j) {
        if (coefficients[j] == 0.0) continue;
        const CompositeLeafCell& cell_j = leaves_[term_leaves[j]];
        if (cell_j.dof == kCompositeNoIndex) continue;
        neighbors[cell_i.dof].push_back(cell_j.dof);
        neighbors[cell_j.dof].push_back(cell_i.dof);
      }
    }
  }
  if (counted_interfaces != interface_face_count_ ||
      interface_face_count_ != 4U *
          ((geometry_.patch_end_x - geometry_.patch_begin_x) +
           (geometry_.patch_end_y - geometry_.patch_begin_y)))
    throw std::logic_error("incomplete ratio-two composite interface");
  for (double value : diagonal_)
    if (!finite(value) || !(value > 0.0))
      throw std::logic_error("non-positive composite Jacobi diagonal");

  // SPD requires every unknown component to connect to prescribed data.
  std::deque<std::size_t> work;
  std::vector<unsigned char> reachable(dof_to_leaf_.size(), 0U);
  for (std::size_t dof = 0U; dof < touches_boundary.size(); ++dof)
    if (touches_boundary[dof]) {
      reachable[dof] = 1U;
      work.push_back(dof);
    }
  while (!work.empty()) {
    const std::size_t current = work.front();
    work.pop_front();
    for (std::size_t next : neighbors[current])
      if (!reachable[next]) {
        reachable[next] = 1U;
        work.push_back(next);
      }
  }
  if (std::find(reachable.begin(), reachable.end(), 0U) != reachable.end())
    throw std::logic_error(
        "composite unknown component is disconnected from Dirichlet data");
}

void CompositePoissonR2::validate_potential(
    const std::vector<double>& potential) const {
  if (potential.size() != unknown_count())
    throw std::invalid_argument("composite potential has wrong DOF count");
  for (double value : potential)
    if (!finite(value))
      throw std::invalid_argument("composite potential is non-finite");
}

void CompositePoissonR2::validate_dirichlet(
    const std::vector<double>& dirichlet_leaf_values) const {
  if (dirichlet_leaf_values.size() != leaf_count())
    throw std::invalid_argument(
        "composite Dirichlet data has wrong leaf count");
  for (double value : dirichlet_leaf_values)
    if (!finite(value))
      throw std::invalid_argument("composite Dirichlet data is non-finite");
}

double CompositePoissonR2::leaf_potential(
    std::size_t leaf, const std::vector<double>& potential,
    const std::vector<double>& dirichlet_leaf_values) const {
  if (leaf >= leaves_.size())
    throw std::out_of_range("composite potential leaf out of range");
  const CompositeLeafCell& cell = leaves_[leaf];
  return cell.dof == kCompositeNoIndex
      ? dirichlet_leaf_values[leaf]
      : potential[cell.dof];
}

double CompositePoissonR2::face_jump(
    const CompositeFace& face,
    const std::vector<double>& potential,
    const std::vector<double>& dirichlet_leaf_values) const {
  double result = face.coefficient_a * leaf_potential(
      face.leaf_a, potential, dirichlet_leaf_values);
  result += face.coefficient_b * leaf_potential(
      face.leaf_b, potential, dirichlet_leaf_values);
  if (face.coefficient_c != 0.0)
    result += face.coefficient_c * leaf_potential(
        face.leaf_c, potential, dirichlet_leaf_values);
  if (!finite(result))
    throw std::overflow_error("composite variational jump overflowed");
  return result;
}

std::vector<double> CompositePoissonR2::apply(
    const std::vector<double>& potential) const {
  validate_potential(potential);
  std::vector<double> result(unknown_count(), 0.0);
  for (const CompositeFace& face : faces_) {
    const CompositeLeafCell& a = leaves_[face.leaf_a];
    const CompositeLeafCell& b = leaves_[face.leaf_b];
    double weighted_jump = a.dof == kCompositeNoIndex
        ? 0.0 : face.coefficient_a * potential[a.dof];
    if (b.dof != kCompositeNoIndex)
      weighted_jump += face.coefficient_b * potential[b.dof];
    if (face.coefficient_c != 0.0) {
      const CompositeLeafCell& c = leaves_[face.leaf_c];
      if (c.dof != kCompositeNoIndex)
        weighted_jump += face.coefficient_c * potential[c.dof];
    }
    const double force = face.transmissibility * weighted_jump;
    if (a.dof != kCompositeNoIndex)
      result[a.dof] += face.coefficient_a * force;
    if (b.dof != kCompositeNoIndex)
      result[b.dof] += face.coefficient_b * force;
    if (face.coefficient_c != 0.0) {
      const CompositeLeafCell& c = leaves_[face.leaf_c];
      if (c.dof != kCompositeNoIndex)
        result[c.dof] += face.coefficient_c * force;
    }
  }
  for (double value : result)
    if (!finite(value))
      throw std::overflow_error("composite matrix application overflowed");
  return result;
}

std::vector<double> CompositePoissonR2::assemble_integrated_rhs(
    const std::vector<double>& integrated_source,
    double source_scale,
    const std::vector<double>& dirichlet_leaf_values) const {
  if (integrated_source.size() != leaf_count() || !finite(source_scale))
    throw std::invalid_argument("invalid composite integrated source");
  validate_dirichlet(dirichlet_leaf_values);
  std::vector<double> rhs(unknown_count(), 0.0);
  for (std::size_t leaf = 0U; leaf < leaf_count(); ++leaf) {
    const double source = integrated_source[leaf];
    if (!finite(source))
      throw std::invalid_argument("composite integrated source is non-finite");
    const CompositeLeafCell& cell = leaves_[leaf];
    if (cell.fixed_dirichlet) {
      if (source != 0.0)
        throw std::invalid_argument(
            "source on prescribed composite boundary would be discarded");
    } else {
      rhs[cell.dof] = source_scale * source;
      if (!finite(rhs[cell.dof]))
        throw std::overflow_error("scaled composite source overflowed");
    }
  }
  for (const CompositeFace& face : faces_) {
    const std::array<std::size_t, 3U> term_leaves = {{
        face.leaf_a, face.leaf_b, face.leaf_c}};
    const std::array<double, 3U> coefficients = {{
        face.coefficient_a, face.coefficient_b, face.coefficient_c}};
    double fixed_jump = 0.0;
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      const CompositeLeafCell& cell = leaves_[term_leaves[i]];
      if (cell.fixed_dirichlet)
        fixed_jump += coefficients[i] *
                      dirichlet_leaf_values[term_leaves[i]];
    }
    if (fixed_jump == 0.0) continue;
    const double boundary_force = face.transmissibility * fixed_jump;
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      const CompositeLeafCell& cell = leaves_[term_leaves[i]];
      if (cell.dof != kCompositeNoIndex)
        rhs[cell.dof] -= coefficients[i] * boundary_force;
    }
  }
  for (double value : rhs)
    if (!finite(value))
      throw std::overflow_error("composite RHS assembly overflowed");
  return rhs;
}

std::vector<double> CompositePoissonR2::residual(
    const std::vector<double>& rhs,
    const std::vector<double>& potential) const {
  if (rhs.size() != unknown_count())
    throw std::invalid_argument("composite RHS has wrong DOF count");
  for (double value : rhs)
    if (!finite(value))
      throw std::invalid_argument("composite RHS is non-finite");
  const std::vector<double> applied = apply(potential);
  std::vector<double> result(unknown_count(), 0.0);
  for (std::size_t i = 0U; i < result.size(); ++i)
    result[i] = rhs[i] - applied[i];
  return result;
}

CompositeResidualNorms CompositePoissonR2::residual_norms(
    const std::vector<double>& values) const {
  if (values.size() != unknown_count())
    throw std::invalid_argument("composite residual has wrong DOF count");
  CompositeResidualNorms result;
  double sum_squared = 0.0;
  for (double value : values) {
    if (!finite(value))
      throw std::invalid_argument("composite residual is non-finite");
    sum_squared += value * value;
    result.linf = std::max(result.linf, std::fabs(value));
  }
  if (!finite(sum_squared))
    throw std::overflow_error("composite residual norm overflowed");
  result.l2 = std::sqrt(sum_squared);
  result.rms = values.empty()
      ? 0.0
      : std::sqrt(sum_squared / static_cast<double>(values.size()));
  return result;
}

std::vector<CompositeFaceFlux> CompositePoissonR2::compute_face_fluxes(
    const std::vector<double>& potential,
    const std::vector<double>& dirichlet_leaf_values) const {
  validate_potential(potential);
  validate_dirichlet(dirichlet_leaf_values);
  std::vector<CompositeFaceFlux> result;
  result.reserve(faces_.size());
  for (std::size_t index = 0U; index < faces_.size(); ++index) {
    const CompositeFace& face = faces_[index];
    CompositeFaceFlux flux;
    flux.face = index;
    flux.flux_a_to_b = face.transmissibility * face_jump(
        face, potential, dirichlet_leaf_values);
    flux.outward_from_a = flux.flux_a_to_b;
    flux.outward_from_b = -flux.flux_a_to_b;
    flux.algebraic_outward_from_a =
        face.coefficient_a * flux.flux_a_to_b;
    flux.algebraic_outward_from_b =
        face.coefficient_b * flux.flux_a_to_b;
    flux.algebraic_outward_from_c =
        face.coefficient_c * flux.flux_a_to_b;
    if (!finite(flux.flux_a_to_b))
      throw std::overflow_error("composite face flux overflowed");
    result.push_back(flux);
  }
  return result;
}

std::vector<CompositeInterfaceFluxConsistency>
CompositePoissonR2::compute_interface_flux_consistency(
    const std::vector<double>& potential,
    const std::vector<double>& dirichlet_leaf_values) const {
  validate_potential(potential);
  validate_dirichlet(dirichlet_leaf_values);
  if (!interface_flux_audit_available_)
    throw std::logic_error(
        "composite hierarchy lacks complete one-sided interface audit stencil");
  std::vector<CompositeInterfaceFluxConsistency> result;
  result.reserve(interface_face_count_);

  const auto offset = [](std::size_t base, bool increase,
                         std::size_t step, std::size_t limit) {
    if ((increase && step > limit - 1U - base) ||
        (!increase && step > base))
      throw std::logic_error(
          "composite one-sided interface stencil is out of range");
    return increase ? base + step : base - step;
  };

  for (std::size_t face_index = 0U; face_index < faces_.size();
       ++face_index) {
    const CompositeFace& face = faces_[face_index];
    if (face.kind != CompositeFaceKind::CoarseFine) continue;
    const bool a_is_coarse =
        leaves_[face.leaf_a].level == CompositeLevel::Coarse;
    const std::size_t coarse_leaf_index =
        a_is_coarse ? face.leaf_a : face.leaf_b;
    const std::size_t fine_leaf_index =
        a_is_coarse ? face.leaf_b : face.leaf_a;
    const CompositeLeafCell& coarse = leaves_[coarse_leaf_index];
    const CompositeLeafCell& coarse_neighbor = leaves_[face.leaf_c];
    const CompositeLeafCell& fine = leaves_[fine_leaf_index];
    const bool coarse_on_negative_side = a_is_coarse;
    const bool coarse_away_increases = !coarse_on_negative_side;
    const bool fine_inward_increases = coarse_on_negative_side;

    double coarse_samples[3] = {0.0, 0.0, 0.0};
    double fine_samples[3] = {0.0, 0.0, 0.0};
    for (std::size_t step = 0U; step < 3U; ++step) {
      std::size_t coarse_primary_at_step = kCompositeNoIndex;
      std::size_t coarse_neighbor_at_step = kCompositeNoIndex;
      std::size_t fine_at_step = kCompositeNoIndex;
      if (face.axis == CompositeFaceAxis::XNormal) {
        const std::size_t coarse_i = offset(
            coarse.level_i, coarse_away_increases, step,
            geometry_.coarse_n_x);
        coarse_primary_at_step = coarse_leaf(
            coarse_i, coarse.level_j);
        coarse_neighbor_at_step = coarse_leaf(
            coarse_i, coarse_neighbor.level_j);
        const std::size_t fine_i = offset(
            fine.level_i, fine_inward_increases, step, fine_n_x_);
        fine_at_step = fine_leaf(fine_i, fine.level_j);
      } else {
        const std::size_t coarse_j = offset(
            coarse.level_j, coarse_away_increases, step,
            geometry_.coarse_n_y);
        coarse_primary_at_step = coarse_leaf(
            coarse.level_i, coarse_j);
        coarse_neighbor_at_step = coarse_leaf(
            coarse_neighbor.level_i, coarse_j);
        const std::size_t fine_j = offset(
            fine.level_j, fine_inward_increases, step, fine_n_y_);
        fine_at_step = fine_leaf(fine.level_i, fine_j);
      }
      if (coarse_primary_at_step == kCompositeNoIndex ||
          coarse_neighbor_at_step == kCompositeNoIndex ||
          fine_at_step == kCompositeNoIndex)
        throw std::logic_error(
            "composite one-sided interface stencil crosses a covered cell");
      coarse_samples[step] = 0.75 * leaf_potential(
          coarse_primary_at_step, potential, dirichlet_leaf_values) +
          0.25 * leaf_potential(
              coarse_neighbor_at_step, potential,
              dirichlet_leaf_values);
      fine_samples[step] = leaf_potential(
          fine_at_step, potential, dirichlet_leaf_values);
    }

    const double coarse_spacing = face.axis == CompositeFaceAxis::XNormal
        ? geometry_.coarse_delta_x : geometry_.coarse_delta_y;
    const double fine_spacing = 0.5 * coarse_spacing;
    const double coarse_derivative = coarse_on_negative_side
        ? (2.0 * coarse_samples[0] - 3.0 * coarse_samples[1] +
           coarse_samples[2]) / coarse_spacing
        : (-2.0 * coarse_samples[0] + 3.0 * coarse_samples[1] -
           coarse_samples[2]) / coarse_spacing;
    const double fine_derivative = coarse_on_negative_side
        ? (-2.0 * fine_samples[0] + 3.0 * fine_samples[1] -
           fine_samples[2]) / fine_spacing
        : (2.0 * fine_samples[0] - 3.0 * fine_samples[1] +
           fine_samples[2]) / fine_spacing;

    CompositeInterfaceFluxConsistency audit;
    audit.face = face_index;
    audit.variational_flux_a_to_b = face.transmissibility * face_jump(
        face, potential, dirichlet_leaf_values);
    audit.coarse_one_sided_flux_a_to_b =
        -face.area * coarse_derivative;
    audit.fine_one_sided_flux_a_to_b = -face.area * fine_derivative;
    audit.absolute_one_sided_defect = std::fabs(
        audit.coarse_one_sided_flux_a_to_b -
        audit.fine_one_sided_flux_a_to_b);
    const double scale = std::max(
        1.0e-300,
        std::max(std::fabs(audit.coarse_one_sided_flux_a_to_b),
                 std::fabs(audit.fine_one_sided_flux_a_to_b)));
    audit.relative_one_sided_defect =
        audit.absolute_one_sided_defect / scale;
    if (!finite(audit.variational_flux_a_to_b) ||
        !finite(audit.coarse_one_sided_flux_a_to_b) ||
        !finite(audit.fine_one_sided_flux_a_to_b) ||
        !finite(audit.absolute_one_sided_defect) ||
        !finite(audit.relative_one_sided_defect))
      throw std::overflow_error(
          "composite one-sided interface audit overflowed");
    result.push_back(audit);
  }
  if (result.size() != interface_face_count_)
    throw std::logic_error(
        "composite one-sided interface audit omitted a subface");
  return result;
}

CompositeGaussReport CompositePoissonR2::compute_gauss_report(
    const std::vector<double>& potential,
    const std::vector<double>& dirichlet_leaf_values,
    const std::vector<double>& integrated_source,
    double source_scale) const {
  if (integrated_source.size() != leaf_count() || !finite(source_scale))
    throw std::invalid_argument("invalid composite Gauss source");
  validate_potential(potential);
  validate_dirichlet(dirichlet_leaf_values);
  std::vector<double> balance(unknown_count(), 0.0);
  const std::vector<CompositeFaceFlux> fluxes = compute_face_fluxes(
      potential, dirichlet_leaf_values);
  CompositeGaussReport result;
  for (std::size_t index = 0U; index < faces_.size(); ++index) {
    const CompositeFace& face = faces_[index];
    const CompositeFaceFlux& flux = fluxes[index];
    const double two_sided =
        std::fabs(flux.outward_from_a + flux.outward_from_b);
    result.maximum_two_sided_flux_defect = std::max(
        result.maximum_two_sided_flux_defect, two_sided);
    if (face.kind == CompositeFaceKind::CoarseFine) {
      ++result.interface_faces;
      result.maximum_interface_two_sided_flux_defect = std::max(
          result.maximum_interface_two_sided_flux_defect, two_sided);
    }
    const std::array<std::size_t, 3U> term_leaves = {{
        face.leaf_a, face.leaf_b, face.leaf_c}};
    const std::array<double, 3U> coefficients = {{
        face.coefficient_a, face.coefficient_b, face.coefficient_c}};
    bool touches_fixed = false;
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      if (leaves_[term_leaves[i]].fixed_dirichlet) touches_fixed = true;
    }
    for (std::size_t i = 0U; i < term_leaves.size(); ++i) {
      if (coefficients[i] == 0.0) continue;
      const CompositeLeafCell& cell = leaves_[term_leaves[i]];
      if (cell.dof == kCompositeNoIndex) continue;
      const double algebraic_outward =
          coefficients[i] * flux.flux_a_to_b;
      balance[cell.dof] += algebraic_outward;
      if (touches_fixed)
        result.boundary_outward_flux += algebraic_outward;
    }
  }

  if (interface_flux_audit_available_) {
    const std::vector<CompositeInterfaceFluxConsistency> interface_audit =
        compute_interface_flux_consistency(
            potential, dirichlet_leaf_values);
    result.interface_one_sided_audited_faces = interface_audit.size();
    for (const CompositeInterfaceFluxConsistency& entry : interface_audit) {
      result.maximum_interface_one_sided_absolute_defect = std::max(
          result.maximum_interface_one_sided_absolute_defect,
          entry.absolute_one_sided_defect);
      result.maximum_interface_one_sided_relative_defect = std::max(
          result.maximum_interface_one_sided_relative_defect,
          entry.relative_one_sided_defect);
    }
  } else {
    result.interface_one_sided_unaudited_faces = interface_face_count_;
  }

  double defect_squared = 0.0;
  for (std::size_t leaf = 0U; leaf < leaf_count(); ++leaf) {
    const double source = integrated_source[leaf];
    if (!finite(source))
      throw std::invalid_argument("composite Gauss source is non-finite");
    const CompositeLeafCell& cell = leaves_[leaf];
    if (cell.fixed_dirichlet) {
      if (source != 0.0)
        throw std::invalid_argument(
            "Gauss source on prescribed composite boundary");
      continue;
    }
    const double scaled = source_scale * source;
    if (!finite(scaled))
      throw std::overflow_error("scaled composite Gauss source overflowed");
    result.integrated_source += scaled;
    const double defect = balance[cell.dof] - scaled;
    defect_squared += defect * defect;
    result.maximum_cell_balance_defect = std::max(
        result.maximum_cell_balance_defect, std::fabs(defect));
  }
  if (!finite(defect_squared) || !finite(result.integrated_source) ||
      !finite(result.boundary_outward_flux))
    throw std::overflow_error("composite Gauss report overflowed");
  result.cell_balance_l2 = std::sqrt(defect_squared);
  result.global_balance_defect =
      result.boundary_outward_flux - result.integrated_source;
  return result;
}

CompositePcgResult CompositePoissonR2::solve_pcg(
    const std::vector<double>& rhs,
    const std::vector<double>& initial_potential,
    const CompositePcgOptions& options) const {
  if (rhs.size() != unknown_count())
    throw std::invalid_argument("PCG RHS has wrong composite DOF count");
  for (double value : rhs)
    if (!finite(value)) throw std::invalid_argument("PCG RHS is non-finite");
  if ((!initial_potential.empty() &&
       initial_potential.size() != unknown_count()) ||
      !finite(options.relative_tolerance) ||
      !finite(options.absolute_tolerance) ||
      options.relative_tolerance < 0.0 ||
      options.relative_tolerance > 1.0 ||
      options.absolute_tolerance < 0.0 ||
      (options.relative_tolerance == 0.0 &&
       options.absolute_tolerance == 0.0) ||
      options.maximum_iterations == 0U ||
      options.residual_recompute_interval == 0U)
    throw std::invalid_argument("invalid composite PCG options");
  for (double value : initial_potential)
    if (!finite(value))
      throw std::invalid_argument("PCG initial potential is non-finite");

  CompositePcgResult result;
  result.solution = initial_potential.empty()
      ? std::vector<double>(unknown_count(), 0.0)
      : initial_potential;
  result.report.rhs_l2 = residual_norms(rhs).l2;
  result.report.rhs_linf = residual_norms(rhs).linf;
  result.report.stopping_linf = std::max(
      options.absolute_tolerance,
      options.relative_tolerance * result.report.rhs_linf);

  std::vector<double> r = residual(rhs, result.solution);
  result.report.initial_residual = residual_norms(r);
  result.report.final_residual = result.report.initial_residual;
  if (result.report.initial_residual.linf <=
      result.report.stopping_linf) {
    result.report.converged = true;
    return result;
  }

  std::vector<double> z(unknown_count(), 0.0);
  std::vector<double> p(unknown_count(), 0.0);
  for (std::size_t i = 0U; i < unknown_count(); ++i) {
    if (!finite(diagonal_[i]) || !(diagonal_[i] > 0.0)) {
      result.report.failure = CompositePcgFailure::PreconditionerBreakdown;
      return result;
    }
    z[i] = r[i] / diagonal_[i];
    p[i] = z[i];
  }
  double rz = deterministic_dot(r, z);
  if (!finite(rz) || !(rz > 0.0)) {
    result.report.failure = CompositePcgFailure::PreconditionerBreakdown;
    return result;
  }

  for (std::size_t iteration = 1U;
       iteration <= options.maximum_iterations; ++iteration) {
    const std::vector<double> ap = apply(p);
    const double pap = deterministic_dot(p, ap);
    if (!finite(pap) || !(pap > 0.0)) {
      result.report.failure = finite(pap)
          ? CompositePcgFailure::NonPositiveCurvature
          : CompositePcgFailure::NonFiniteArithmetic;
      result.report.iterations = iteration - 1U;
      result.report.final_residual = residual_norms(
          residual(rhs, result.solution));
      return result;
    }
    const double alpha = rz / pap;
    if (!finite(alpha)) {
      result.report.failure = CompositePcgFailure::NonFiniteArithmetic;
      result.report.iterations = iteration - 1U;
      result.report.final_residual = residual_norms(
          residual(rhs, result.solution));
      return result;
    }
    for (std::size_t i = 0U; i < unknown_count(); ++i) {
      result.solution[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      if (!finite(result.solution[i]) || !finite(r[i])) {
        result.report.failure = CompositePcgFailure::NonFiniteArithmetic;
        result.report.iterations = iteration;
        return result;
      }
    }
    result.report.iterations = iteration;

    bool recomputed = false;
    CompositeResidualNorms current = residual_norms(r);
    if (iteration % options.residual_recompute_interval == 0U ||
        current.linf <= result.report.stopping_linf) {
      r = residual(rhs, result.solution);
      current = residual_norms(r);
      recomputed = true;
    }
    result.report.final_residual = current;
    if (current.linf <= result.report.stopping_linf) {
      // A candidate stop is accepted only from the explicitly recomputed
      // residual, never solely from the recurrence.
      if (!recomputed) {
        r = residual(rhs, result.solution);
        result.report.final_residual = residual_norms(r);
      }
      if (result.report.final_residual.linf <=
          result.report.stopping_linf) {
        result.report.converged = true;
        result.report.failure = CompositePcgFailure::None;
        return result;
      }
    }

    for (std::size_t i = 0U; i < unknown_count(); ++i)
      z[i] = r[i] / diagonal_[i];
    const double new_rz = deterministic_dot(r, z);
    if (!finite(new_rz) || !(new_rz > 0.0) || !finite(rz) || !(rz > 0.0)) {
      result.report.failure = finite(new_rz)
          ? CompositePcgFailure::PreconditionerBreakdown
          : CompositePcgFailure::NonFiniteArithmetic;
      result.report.final_residual = residual_norms(
          residual(rhs, result.solution));
      return result;
    }
    const double beta = new_rz / rz;
    if (!finite(beta)) {
      result.report.failure = CompositePcgFailure::NonFiniteArithmetic;
      result.report.final_residual = residual_norms(
          residual(rhs, result.solution));
      return result;
    }
    for (std::size_t i = 0U; i < unknown_count(); ++i) {
      p[i] = z[i] + beta * p[i];
      if (!finite(p[i])) {
        result.report.failure = CompositePcgFailure::NonFiniteArithmetic;
        result.report.final_residual = residual_norms(
            residual(rhs, result.solution));
        return result;
      }
    }
    rz = new_rz;
  }

  result.report.failure = CompositePcgFailure::MaximumIterations;
  result.report.final_residual = residual_norms(
      residual(rhs, result.solution));
  return result;
}

}  // namespace adaptive_grid
}  // namespace feebl
