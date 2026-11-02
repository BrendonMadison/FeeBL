#ifndef FEEBL_QEDBG_PHOTON_DOMAIN_H
#define FEEBL_QEDBG_PHOTON_DOMAIN_H

#include "qedbg_sampling_guard.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace qedbg {

// The additive transverse-angle model is a small-angle approximation. At
// 0.1 rad the leading omitted relative angular terms are O(1%). This is a
// validity ceiling, not a physical acceptance cut: exceeding it aborts the
// calculation instead of replacing or resampling the photon. The constrained
// photon_angle=2 model also checks the radiation formation length.
inline void validate_photon_angle(double angle_rad, double beta_x, double beta_y) {
  constexpr double maximum_angle_rad = 0.1;
  const double transverse_beta2 = beta_x*beta_x + beta_y*beta_y;
  if (!finite_nonnegative(std::abs(angle_rad)) ||
      !finite_nonnegative(transverse_beta2) ||
      std::abs(angle_rad) > maximum_angle_rad || !(transverse_beta2 < 1.0)) {
    std::ostringstream message;
    message << "Photon small-angle model outside its domain: angle_rad=" << angle_rad
            << " transverse_beta_squared=" << transverse_beta2
            << "; photon_angle=2 restricts emission to short formation lengths";
    throw std::domain_error(message.str());
  }
}

} // namespace qedbg
#endif
