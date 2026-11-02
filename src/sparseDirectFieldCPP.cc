#include "sparseDirectFieldCPP.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void require_ascending_unique(const std::vector<std::size_t>& cells,
                              std::size_t cell_count,
                              const char* label)
{
  for (std::size_t position = 0; position < cells.size(); ++position)
    {
      if (cells[position] >= cell_count)
        throw std::out_of_range(std::string(label) +
                                " contains an out-of-grid cell");
      if (position != 0 && cells[position - 1] >= cells[position])
        throw std::invalid_argument(std::string(label) +
                                    " must be unique and ascending");
    }
}

}  // namespace

namespace feebl {
namespace sparse_direct_fields {

Mode parse_mode(const std::string& value)
{
  if (value == "fft") return Mode::Fft;
  if (value == "direct") return Mode::Direct;
  if (value == "auto") return Mode::Auto;
  throw std::invalid_argument(
      "wavefront_field_solver must be fft, direct, or auto");
}

const char* mode_name(Mode mode) noexcept
{
  switch (mode)
    {
    case Mode::Fft: return "fft";
    case Mode::Direct: return "direct";
    case Mode::Auto: return "auto";
    }
  return "invalid";
}

std::uint64_t interaction_count(std::size_t active_sources,
                                std::size_t target_nodes)
{
  if (active_sources != 0 &&
      target_nodes > std::numeric_limits<std::uint64_t>::max() /
                         active_sources)
    throw std::overflow_error("sparse-direct interaction count overflow");
  return static_cast<std::uint64_t>(active_sources) *
         static_cast<std::uint64_t>(target_nodes);
}

bool choose_direct(Mode mode, std::uint64_t interactions,
                   std::uint64_t interaction_limit) noexcept
{
  if (mode == Mode::Direct) return true;
  if (mode == Mode::Fft) return false;
  return interactions <= interaction_limit;
}

void electric_field_stencil_nodes(PHI_FLOAT absolute_x,
                                  PHI_FLOAT absolute_y,
                                  int n_x, int n_y,
                                  std::size_t nodes[12])
{
  if (!nodes || n_x < 3 || n_y < 3)
    throw std::invalid_argument(
        "electric-field stencil received invalid storage or dimensions");
  const auto linear = [n_x, n_y](int ix, int iy) -> std::size_t {
    if (ix < 0 || ix >= n_x || iy < 0 || iy >= n_y)
      throw std::out_of_range(
          "intended-domain particle produced an invalid field stencil");
    return static_cast<std::size_t>(ix) *
               static_cast<std::size_t>(n_y) +
           static_cast<std::size_t>(iy);
  };

  // interpolePotential's local interpolation weight is float, so overload
  // resolution selects GENERAL_GRID::cellLocalization(float,...), even though
  // absoluteCoordinates stores its result in PHI_FLOAT.  Reproduce those two
  // conversions before taking either floor; this matters at one-ULP faces.
  const float grid_x = static_cast<float>(absolute_x);
  const float grid_y = static_cast<float>(absolute_y);
  int ix = static_cast<int>(std::floor(grid_x));
  int iy = static_cast<int>(std::floor(grid_y - 0.5f));
  nodes[0] = linear(ix + 1, iy + 1);
  nodes[1] = linear(ix + 1, iy);
  nodes[2] = linear(ix, iy + 1);
  nodes[3] = linear(ix, iy);
  nodes[4] = linear(ix - 1, iy + 1);
  nodes[5] = linear(ix - 1, iy);

  ix = static_cast<int>(std::floor(grid_x - 0.5f));
  iy = static_cast<int>(std::floor(grid_y));
  nodes[6] = linear(ix + 1, iy + 1);
  nodes[7] = linear(ix, iy + 1);
  nodes[8] = linear(ix + 1, iy);
  nodes[9] = linear(ix, iy);
  nodes[10] = linear(ix + 1, iy - 1);
  nodes[11] = linear(ix, iy - 1);
}

void solve_potential_nodes(const PHI_FLOAT* rho,
                           const std::vector<std::size_t>& active_sources,
                           const PHI_FLOAT* real_space_kernel,
                           const std::vector<std::size_t>& target_nodes,
                           int n_x, int n_y, PHI_FLOAT* phi)
{
  if (!rho || !real_space_kernel || !phi)
    throw std::invalid_argument(
        "sparse-direct convolution received a null array");
  if (n_x <= 0 || n_y <= 0)
    throw std::invalid_argument(
        "sparse-direct convolution received invalid dimensions");
  const std::size_t nx = static_cast<std::size_t>(n_x);
  const std::size_t ny = static_cast<std::size_t>(n_y);
  if (nx > std::numeric_limits<std::size_t>::max() / ny)
    throw std::overflow_error("sparse-direct grid cell count overflow");
  const std::size_t cells = nx * ny;
  require_ascending_unique(active_sources, cells, "active source list");
  require_ascending_unique(target_nodes, cells, "target-node list");

  // Keep the target-major/source-minor order scalar and explicit.  This file
  // is compiled without fast-math or contraction so worker count and SIMD
  // width cannot alter the result for a fixed build.
  for (std::size_t target_position = 0;
       target_position < target_nodes.size(); ++target_position)
    {
      const std::size_t target = target_nodes[target_position];
      const std::size_t target_x = target / ny;
      const std::size_t target_y = target - target_x * ny;
      PHI_FLOAT potential = PHI_FLOAT(0.0);
      for (std::size_t source_position = 0;
           source_position < active_sources.size(); ++source_position)
        {
          const std::size_t source = active_sources[source_position];
          const std::size_t source_x = source / ny;
          const std::size_t source_y = source - source_x * ny;
          const std::size_t delta_x = target_x >= source_x
              ? target_x - source_x : source_x - target_x;
          const std::size_t delta_y = target_y >= source_y
              ? target_y - source_y : source_y - target_y;
          potential += rho[source] *
                       real_space_kernel[delta_x * ny + delta_y];
        }
      phi[target] = potential;
    }
}

}  // namespace sparse_direct_fields
}  // namespace feebl
