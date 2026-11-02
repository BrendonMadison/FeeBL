#include "reference_maps.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool close(double a, double b, double atol = 1.0e-11, double rtol = 1.0e-10) {
  return std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  using gpgeom::CrabRFCavityMap;
  using gpgeom::CrabWaistMap;
  using gpgeom::Harmonic;
  using gpgeom::PhaseSpace6D;

  // Analytic second-harmonic coefficients: K'(0)=kappa and K'''(0)=0.
  constexpr double k = 8.377580409572781;  // 400 MHz / c, approximately
  constexpr double kappa = 0.015 / 16.0;
  const auto a = CrabRFCavityMap::cubic_linearizer(kappa, k, 2);
  require(close(k * (a[0] + 2.0 * a[1]), kappa), "linear slope mismatch");
  require(close(a[0] + 8.0 * a[1], 0.0), "cubic term did not cancel");

  CrabRFCavityMap crab(k, {{1, a[0], 0.0}, {2, a[1], 0.0}});
  PhaseSpace6D p{2.0e-4, 0.0, -1.0e-5, 3.0e-4, 1.0e-3, 0.0};
  const double expected_px = crab.kick(p.zeta);
  const double expected_delta = p.x * crab.kick_derivative(p.zeta);
  crab.apply(p);
  require(close(p.px, expected_px), "RF transverse kick mismatch");
  require(close(p.delta, expected_delta), "Panofsky-Wenzel kick mismatch");

  const PhaseSpace6D around{1.5e-3, 2.0e-4, -3.0e-4, 1.2e-3, 8.0e-4, 2.0e-5};
  const auto j_crab = gpgeom::numerical_jacobian(
      [&crab](PhaseSpace6D& q) { crab.apply(q); }, around, 2.0e-7);
  require(gpgeom::symplectic_defect_inf_norm(j_crab) < 2.0e-8,
          "RF map is not numerically symplectic");

  CrabWaistMap waist(0.030, 1.0);
  PhaseSpace6D w{3.0e-3, 4.0e-5, 2.0e-5, 2.0e-3, 0.0, 0.0};
  const double old_y = w.y;
  const double old_px = w.px;
  const double g = waist.g(w.x);
  const double dg = waist.dg_dx(w.x);
  waist.apply(w);
  require(close(w.y, old_y + g * 2.0e-3), "waist y shift mismatch");
  require(close(w.px, old_px - 0.5 * dg * 4.0e-6), "waist px companion kick mismatch");

  const auto j_waist = gpgeom::numerical_jacobian(
      [&waist](PhaseSpace6D& q) { waist.apply(q); }, around, 2.0e-7);
  require(gpgeom::symplectic_defect_inf_norm(j_waist) < 2.0e-8,
          "waist map is not numerically symplectic");

  std::cout << "All gpgeom reference tests passed.\n";
  std::cout << "RF symplectic defect: " << gpgeom::symplectic_defect_inf_norm(j_crab) << "\n";
  std::cout << "Waist symplectic defect: " << gpgeom::symplectic_defect_inf_norm(j_waist) << "\n";
  return EXIT_SUCCESS;
}
