#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "gridCPP.h"

namespace {

std::size_t cell_index(int ix, int iy, int ny)
{
  return static_cast<std::size_t>(ix) * static_cast<std::size_t>(ny) +
         static_cast<std::size_t>(iy);
}

std::uint32_t float_bits(float value)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// A focused concrete shell around GENERAL_GRID.  Geometry is injected
// directly so this test exercises the production ElectricFieldCIC method
// without constructing beams, FFTs, event generators, or the FeeBL driver.
class FieldContractGrid final : public GENERAL_GRID
{
 public:
  FieldContractGrid(int nx,
                    int ny,
                    float inverse_x,
                    float inverse_y,
                    float offset_x,
                    float offset_y,
                    float physical_min_x,
                    float physical_max_x,
                    float physical_min_y,
                    float physical_max_y)
  {
    n_cell_x_ = nx;
    n_cell_y_ = ny;
    delta_x_inverse_ = inverse_x;
    delta_y_inverse_ = inverse_y;
    mesh_ = MESH(1.0f / inverse_x, 1.0f / inverse_y, 1.0f,
                 offset_x, offset_y);
    min_x_ = physical_min_x;
    max_x_ = physical_max_x;
    min_y_ = physical_min_y;
    max_y_ = physical_max_y;
  }

  void distribute_coherent_particles(int, int, int) override {}
  void distribute_particles(int, int, int, int) override {}
};

bool independent_stencil_domain(float x,
                                float y,
                                float physical_min_x,
                                float physical_max_x,
                                float physical_min_y,
                                float physical_max_y,
                                float inverse_x,
                                float inverse_y,
                                float offset_x,
                                float offset_y,
                                int nx,
                                int ny)
{
  if (!(x > physical_min_x && x < physical_max_x &&
        y > physical_min_y && y < physical_max_y))
    return false;
  volatile float x_product = x * inverse_x;
  volatile float y_product = y * inverse_y;
  volatile float absolute_x_store = x_product + offset_x;
  volatile float absolute_y_store = y_product + offset_y;
  const float absolute_x = absolute_x_store;
  const float absolute_y = absolute_y_store;
  if (!std::isfinite(absolute_x) || !std::isfinite(absolute_y)) return false;

  // Validate each index touched by phiValuesY and phiValuesX independently
  // of the shared compact predicate.
  const int y_i1 = static_cast<int>(std::floor(absolute_x));
  const int y_i2 = static_cast<int>(std::floor(absolute_y - 0.5f));
  const int x_i1 = static_cast<int>(std::floor(absolute_x - 0.5f));
  const int x_i2 = static_cast<int>(std::floor(absolute_y));
  return y_i1 - 1 >= 0 && y_i1 + 1 < nx &&
         y_i2 >= 0 && y_i2 + 1 < ny &&
         x_i1 >= 0 && x_i1 + 1 < nx &&
         x_i2 - 1 >= 0 && x_i2 + 1 < ny;
}

std::pair<float, float> independent_electric_field(
    float x,
    float y,
    const std::vector<PHI_FLOAT>& phi,
    int ny,
    float inverse_x,
    float inverse_y,
    float offset_x,
    float offset_y)
{
  // This is a direct expression-level transcription of the historical
  // interpolePotential/ElectricFieldCIC arithmetic.  It deliberately does
  // not call any GENERAL_GRID helper.
  float absolute_x = x * inverse_x + offset_x;
  float absolute_y = y * inverse_y + offset_y;

  int y_i1 = static_cast<int>(std::floor(absolute_x));
  float y_weight = absolute_y - 0.5f;
  int y_i2 = static_cast<int>(std::floor(y_weight));
  y_weight -= static_cast<float>(y_i2);
  const float y_complement = 1.0f - y_weight;
  const std::size_t y_center = cell_index(y_i1, y_i2, ny);
  const PHI_FLOAT phi1_y =
      y_weight * phi[y_center + ny + 1] +
      y_complement * phi[y_center + ny];
  const PHI_FLOAT phi2_y =
      y_weight * phi[y_center + 1] + y_complement * phi[y_center];
  const PHI_FLOAT phi3_y =
      y_weight * phi[y_center - ny + 1] +
      y_complement * phi[y_center - ny];

  int x_i2 = static_cast<int>(std::floor(absolute_y));
  float x_weight = absolute_x - 0.5f;
  int x_i1 = static_cast<int>(std::floor(x_weight));
  x_weight -= static_cast<float>(x_i1);
  const float x_complement = 1.0f - x_weight;
  const std::size_t x_center = cell_index(x_i1, x_i2, ny);
  const PHI_FLOAT phi1_x =
      x_weight * phi[x_center + ny + 1] +
      x_complement * phi[x_center + 1];
  const PHI_FLOAT phi2_x =
      x_weight * phi[x_center + ny] + x_complement * phi[x_center];
  const PHI_FLOAT phi3_x =
      x_weight * phi[x_center + ny - 1] +
      x_complement * phi[x_center - 1];

  absolute_x -= std::floor(absolute_x);
  absolute_y -= std::floor(absolute_y);
  float field_x = static_cast<float>(
      (absolute_x * (phi1_y - phi2_y) +
       (1.0 - absolute_x) * (phi2_y - phi3_y)) * inverse_x);
  float field_y = static_cast<float>(
      (absolute_y * (phi1_x - phi2_x) +
       (1.0 - absolute_y) * (phi2_x - phi3_x)) * inverse_y);
  field_x *= 0.5f;
  field_y *= 0.5f;
  return std::make_pair(field_x, field_y);
}

float coordinate_from_normalized(float normalized,
                                 float inverse,
                                 float offset)
{
  return (normalized - offset) / inverse;
}

bool run_scenario(int nx,
                  int ny,
                  float inverse_x,
                  float inverse_y,
                  float offset_x,
                  float offset_y,
                  float physical_min_x,
                  float physical_max_x,
                  float physical_min_y,
                  float physical_max_y,
                  std::uint32_t seed)
{
  FieldContractGrid grid(nx, ny, inverse_x, inverse_y, offset_x, offset_y,
                         physical_min_x, physical_max_x,
                         physical_min_y, physical_max_y);
  std::vector<PHI_FLOAT> phi(static_cast<std::size_t>(nx) * ny);
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      phi[cell_index(ix, iy, ny)] =
          0.173 * ix - 0.291 * iy + 0.037 * ix * iy +
          0.011 * (ix + 2 * iy) * (ix - iy);

  const float safe_x = coordinate_from_normalized(
      0.5f * (2.0f + static_cast<float>(nx - 2)), inverse_x, offset_x);
  const float safe_y = coordinate_from_normalized(
      0.5f * (2.0f + static_cast<float>(ny - 2)), inverse_y, offset_y);
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const float positive_infinity = std::numeric_limits<float>::infinity();
  std::vector<std::pair<float, float> > points;

  const float normalized_x_edges[] = {1.0f, static_cast<float>(nx - 1)};
  for (float edge : normalized_x_edges)
    {
      const float x = coordinate_from_normalized(edge, inverse_x, offset_x);
      points.push_back(std::make_pair(x, safe_y));
      points.push_back(std::make_pair(std::nextafter(x, negative_infinity),
                                      safe_y));
      points.push_back(std::make_pair(std::nextafter(x, positive_infinity),
                                      safe_y));
    }
  const float normalized_y_edges[] = {1.0f, static_cast<float>(ny - 1)};
  for (float edge : normalized_y_edges)
    {
      const float y = coordinate_from_normalized(edge, inverse_y, offset_y);
      points.push_back(std::make_pair(safe_x, y));
      points.push_back(std::make_pair(safe_x,
                                      std::nextafter(y, negative_infinity)));
      points.push_back(std::make_pair(safe_x,
                                      std::nextafter(y, positive_infinity)));
    }
  const float physical_x_edges[] = {physical_min_x, physical_max_x};
  for (float edge : physical_x_edges)
    {
      points.push_back(std::make_pair(edge, safe_y));
      points.push_back(std::make_pair(std::nextafter(edge, negative_infinity),
                                      safe_y));
      points.push_back(std::make_pair(std::nextafter(edge, positive_infinity),
                                      safe_y));
    }
  const float physical_y_edges[] = {physical_min_y, physical_max_y};
  for (float edge : physical_y_edges)
    {
      points.push_back(std::make_pair(safe_x, edge));
      points.push_back(std::make_pair(safe_x,
                                      std::nextafter(edge, negative_infinity)));
      points.push_back(std::make_pair(safe_x,
                                      std::nextafter(edge, positive_infinity)));
    }
  points.push_back(std::make_pair(std::numeric_limits<float>::quiet_NaN(),
                                  safe_y));
  points.push_back(std::make_pair(safe_x, positive_infinity));

  std::mt19937 random(seed);
  std::uniform_real_distribution<float> random_x(
      physical_min_x - 1.0f, physical_max_x + 1.0f);
  std::uniform_real_distribution<float> random_y(
      physical_min_y - 1.0f, physical_max_y + 1.0f);
  for (int sample = 0; sample < 4096; ++sample)
    points.push_back(std::make_pair(random_x(random), random_y(random)));

  std::size_t safe_points = 0;
  for (const std::pair<float, float>& point : points)
    {
      const bool independent = independent_stencil_domain(
          point.first, point.second,
          physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          inverse_x, inverse_y, offset_x, offset_y, nx, ny);
      const bool production = grid.coordinatesInGridRange(
          point.first, point.second);
      if (production != independent)
        {
          std::cerr << "domain mismatch at (" << std::setprecision(9)
                    << point.first << ',' << point.second << ")\n";
          return false;
        }
      if (!independent) continue;
      ++safe_points;
      const TRIVECTOR actual = grid.ElectricFieldCIC(
          point.first, point.second, phi.data());
      const std::pair<float, float> expected = independent_electric_field(
          point.first, point.second, phi, ny,
          inverse_x, inverse_y, offset_x, offset_y);
      if (float_bits(actual(0)) != float_bits(expected.first) ||
          float_bits(actual(1)) != float_bits(expected.second) ||
          float_bits(actual(2)) != float_bits(0.0f))
        {
          std::cerr << "ElectricFieldCIC bit mismatch at ("
                    << std::setprecision(9) << point.first << ','
                    << point.second << ") actual=(" << actual(0) << ','
                    << actual(1) << ") expected=(" << expected.first << ','
                    << expected.second << ")\n";
          return false;
        }

      // Build the smallest integer-aligned canonical window containing the
      // complete twelve-node stencil.  The translated production evaluator
      // must make the same floating mapping decisions before subtracting the
      // integer origin, so even nextafter boundary cases remain bit-identical.
      float absolute_x = point.first * inverse_x + offset_x;
      float absolute_y = point.second * inverse_y + offset_y;
      const int y_i1 = static_cast<int>(std::floor(absolute_x));
      const int y_i2 = static_cast<int>(std::floor(absolute_y - 0.5f));
      const int x_i1 = static_cast<int>(std::floor(absolute_x - 0.5f));
      const int x_i2 = static_cast<int>(std::floor(absolute_y));
      const int origin_x = std::min(y_i1 - 1, x_i1);
      const int end_x = std::max(y_i1 + 1, x_i1 + 1);
      const int origin_y = std::min(y_i2, x_i2 - 1);
      const int end_y = std::max(y_i2 + 1, x_i2 + 1);
      const int local_nx = end_x - origin_x + 1;
      const int local_ny = end_y - origin_y + 1;
      std::vector<PHI_FLOAT> local_phi(
          static_cast<std::size_t>(local_nx) * local_ny);
      for (int local_x = 0; local_x < local_nx; ++local_x)
        for (int local_y = 0; local_y < local_ny; ++local_y)
          local_phi[cell_index(local_x, local_y, local_ny)] =
              phi[cell_index(origin_x + local_x,
                             origin_y + local_y, ny)];
      const TRIVECTOR translated = grid.ElectricFieldCICTranslated(
          point.first, point.second, local_phi.data(),
          local_nx, local_ny, origin_x, origin_y);
      if (float_bits(translated(0)) != float_bits(actual(0)) ||
          float_bits(translated(1)) != float_bits(actual(1)) ||
          float_bits(translated(2)) != float_bits(actual(2)))
        {
          std::cerr << "translated ElectricFieldCIC bit mismatch at ("
                    << std::setprecision(9) << point.first << ','
                    << point.second << ") origin=(" << origin_x << ','
                    << origin_y << ") shape=(" << local_nx << ','
                    << local_ny << ")\n";
          return false;
        }
    }
  if (safe_points < 100)
    {
      std::cerr << "insufficient safe randomized points: " << safe_points
                << '\n';
      return false;
    }
  std::cout << "GENERAL_GRID ElectricFieldCIC actual-host differential: PASS "
            << "points=" << points.size() << " safe=" << safe_points
            << " max_ulp=0\n";
  return true;
}

}  // namespace

int main()
{
  bool passed = GRID::compactPrimaryFieldContractEligible(2, 0, 0);
  if (!passed || GRID::compactPrimaryFieldContractEligible(1, 0, 0) ||
      GRID::compactPrimaryFieldContractEligible(2, 1, 0) ||
      GRID::compactPrimaryFieldContractEligible(2, 0, 1))
    {
      std::cerr << "compact primary mover eligibility invariant failed\n";
      passed = false;
    }
  const bool parallel_contract =
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 0, 0, 0, 0, 0);
  if (!parallel_contract ||
      GRID::compactPrimaryParallelMoverContractEligible(
          1, 0, 0, 0, 0, 0, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 1, 0, 0, 0, 0, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 1, 0, 0, 0, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 1, 0, 0, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 0, 1, 0, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 0, 0, 1, 0, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 0, 0, 0, 1, 0) ||
      GRID::compactPrimaryParallelMoverContractEligible(
          2, 0, 0, 0, 0, 0, 0, 1))
    {
      std::cerr << "parallel compact primary mover eligibility invariant failed\n";
      passed = false;
    }
  passed = run_scenario(8, 9, 2.0f, 4.0f, 4.0f, 4.5f,
                             -1.5f, 1.5f, -0.875f, 0.875f,
                             0x31415926U) && passed;
  passed = run_scenario(7, 10, 1.3f, 0.7f, 3.1f, 4.2f,
                        -1.4f, 2.0f, -4.2f, 6.3f,
                        0x27182818U) && passed;
  return passed ? 0 : 1;
}
