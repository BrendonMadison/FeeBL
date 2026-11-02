#include "qedbg_photon_domain.h"
#include <iostream>
#include <limits>

// Domain failures must remain visible with the production fast-math flags.
int main() {
  qedbg::validate_photon_angle(0.0, 0.0, 0.0);
  qedbg::validate_photon_angle(-0.05, 0.05, 0.01);
  int rejected = 0;
  const auto check = [&](double angle, double bx, double by) {
    try { qedbg::validate_photon_angle(angle, bx, by); }
    catch (const std::domain_error&) { ++rejected; }
  };
  check(0.11, 0.11, 0.0);
  check(-0.11, 0.11, 0.0);
  check(0.01, 0.8, 0.8);
  check(std::numeric_limits<double>::infinity(), 0.0, 0.0);
  check(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  check(0.01, std::numeric_limits<double>::quiet_NaN(), 0.0);
  if (rejected != 6) {
    std::cerr << "Photon domain rejected " << rejected << " of six invalid inputs\n";
    return 1;
  }
  return 0;
}
