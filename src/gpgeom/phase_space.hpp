#pragma once

#include <array>
#include <cstddef>

namespace gpgeom {

struct PhaseSpace6D {
  double x{};
  double px{};
  double y{};
  double py{};
  double zeta{};
  double delta{};

  [[nodiscard]] std::array<double, 6> as_array() const noexcept {
    return {x, px, y, py, zeta, delta};
  }

  static PhaseSpace6D from_array(const std::array<double, 6>& a) noexcept {
    return {a[0], a[1], a[2], a[3], a[4], a[5]};
  }
};

using Matrix6 = std::array<std::array<double, 6>, 6>;

[[nodiscard]] Matrix6 canonical_symplectic_matrix() noexcept;

}  // namespace gpgeom
