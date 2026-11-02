#ifndef FEEBL_SPARSE_DIRECT_FIELD_CPP_H
#define FEEBL_SPARSE_DIRECT_FIELD_CPP_H

#include "typeDefs.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace sparse_direct_fields {

enum class Mode {
  Fft,
  Direct,
  Auto
};

Mode parse_mode(const std::string& value);
const char* mode_name(Mode mode) noexcept;

// The estimate is the exact number of multiply/add source contributions made
// by solve_potential_nodes.  Overflow is rejected instead of wrapping into an
// unsafe automatic-selection decision.
std::uint64_t interaction_count(std::size_t active_sources,
                                std::size_t target_nodes);

// Forced modes are literal.  Auto chooses the sparse solver only when its
// measured unit of work is at or below the user-visible crossover.  A zero
// crossover is invalid at input preflight and is not assigned special meaning.
bool choose_direct(Mode mode, std::uint64_t interactions,
                   std::uint64_t interaction_limit) noexcept;

// Return the twelve (not necessarily unique) potential nodes read by
// GENERAL_GRID::ElectricFieldCIC for a point already accepted by the intended
// main-grid domain predicate.  The order mirrors interpolePotential's Y then X
// reads and is kept here so the optimization and its unit test share one
// audited stencil definition.
void electric_field_stencil_nodes(PHI_FLOAT absolute_x,
                                  PHI_FLOAT absolute_y,
                                  int n_x, int n_y,
                                  std::size_t nodes[12]);

// Evaluate the same translation-invariant, cell-integrated discrete Green
// convolution used to construct FIELD's case-2 FFT kernel, but only at the
// requested target nodes.  Source and target vectors must be unique and in
// strictly ascending linear-cell order.  This makes source summation order an
// explicit reproducibility contract.
void solve_potential_nodes(const PHI_FLOAT* rho,
                           const std::vector<std::size_t>& active_sources,
                           const PHI_FLOAT* real_space_kernel,
                           const std::vector<std::size_t>& target_nodes,
                           int n_x, int n_y, PHI_FLOAT* phi);

}  // namespace sparse_direct_fields
}  // namespace feebl

#endif  // FEEBL_SPARSE_DIRECT_FIELD_CPP_H
