#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fieldCPP.h"
#include "fourierCPP.h"
#include "sparseDirectFieldCPP.h"

namespace {

void require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << "sparse-direct field test failure: " << message << '\n';
    std::exit(1);
  }
}

void require_close(double actual, double expected, const char* message)
{
  const double error = std::fabs(actual - expected);
  const double tolerance = 2.0e-11 *
      std::max(1.0, std::max(std::fabs(actual), std::fabs(expected)));
  if (!(error <= tolerance)) {
    std::cerr << "sparse-direct field test failure: " << message
              << ": actual=" << actual << " expected=" << expected
              << " error=" << error << " tolerance=" << tolerance << '\n';
    std::exit(1);
  }
}

struct Field2 {
  double x;
  double y;
};

Field2 cic_field(PHI_FLOAT absolute_x, PHI_FLOAT absolute_y,
                 double inverse_dx, double inverse_dy,
                 const PHI_FLOAT* phi, int ny)
{
  int ix = static_cast<int>(std::floor(absolute_x));
  int iy = static_cast<int>(std::floor(absolute_y - 0.5));
  double h = absolute_y - 0.5 - iy;
  const std::size_t j_y = static_cast<std::size_t>(ix) * ny + iy;
  const double phi1_y = h * phi[j_y + ny + 1U] +
                        (1.0 - h) * phi[j_y + ny];
  const double phi2_y = h * phi[j_y + 1U] + (1.0 - h) * phi[j_y];
  const double phi3_y = h * phi[j_y - ny + 1U] +
                        (1.0 - h) * phi[j_y - ny];

  iy = static_cast<int>(std::floor(absolute_y));
  ix = static_cast<int>(std::floor(absolute_x - 0.5));
  h = absolute_x - 0.5 - ix;
  const std::size_t j_x = static_cast<std::size_t>(ix) * ny + iy;
  const double phi1_x = h * phi[j_x + ny + 1U] +
                        (1.0 - h) * phi[j_x + 1U];
  const double phi2_x = h * phi[j_x + ny] + (1.0 - h) * phi[j_x];
  const double phi3_x = h * phi[j_x + ny - 1U] +
                        (1.0 - h) * phi[j_x - 1U];

  Field2 field;
  field.x = 0.5 * (absolute_x - std::floor(absolute_x)) *
                (phi1_y - phi2_y) * inverse_dx +
            0.5 * (1.0 - (absolute_x - std::floor(absolute_x))) *
                (phi2_y - phi3_y) * inverse_dx;
  field.y = 0.5 * (absolute_y - std::floor(absolute_y)) *
                (phi1_x - phi2_x) * inverse_dy +
            0.5 * (1.0 - (absolute_y - std::floor(absolute_y))) *
                (phi2_x - phi3_x) * inverse_dy;
  return field;
}

void test_selector_and_crossover()
{
  using namespace feebl::sparse_direct_fields;
  require(parse_mode("fft") == Mode::Fft, "parse fft");
  require(parse_mode("direct") == Mode::Direct, "parse direct");
  require(parse_mode("auto") == Mode::Auto, "parse auto");
  bool rejected = false;
  try { (void)parse_mode("approximate"); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "unknown solver must fail closed");
  require(interaction_count(17U, 31U) == 527U,
          "interaction estimate must be exact");
  require(choose_direct(Mode::Direct, 999U, 1U),
          "forced direct must remain literal");
  require(!choose_direct(Mode::Fft, 0U, 999U),
          "forced FFT must remain literal");
  require(choose_direct(Mode::Auto, 527U, 527U),
          "auto threshold must include equality");
  require(!choose_direct(Mode::Auto, 528U, 527U),
          "auto threshold must reject the first larger estimate");
}

void test_full_interpolation_stencil()
{
  std::size_t nodes[12];
  feebl::sparse_direct_fields::electric_field_stencil_nodes(
      4.2, 5.3, 10, 12, nodes);
  const std::size_t expected[12] = {
      65U, 64U, 53U, 52U, 41U, 40U,
      54U, 42U, 53U, 41U, 52U, 40U};
  for (std::size_t position = 0; position < 12U; ++position)
    require(nodes[position] == expected[position],
            "all phiValuesY/phiValuesX nodes must be represented");

  // Both valid physical extremes must retain an in-array +/- one stencil.
  feebl::sparse_direct_fields::electric_field_stencil_nodes(
      1.0, 1.0, 10, 12, nodes);
  for (std::size_t node : nodes) require(node < 120U, "lower boundary node");
  feebl::sparse_direct_fields::electric_field_stencil_nodes(
      std::nextafter(9.0f, 0.0f), std::nextafter(11.0f, 0.0f),
      10, 12, nodes);
  for (std::size_t node : nodes) require(node < 120U, "upper boundary node");
}

void test_exact_kernel_direct_against_fft()
{
  const int nx = 8;
  const int ny = 10;
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  FFT_SERVER server;
  FIELD field(2, nx, ny, &server);
  field.enable_real_space_kernel(true);
  field.dist_init(-0.375, 0.7f, 0.23f, &server);

  std::vector<PHI_FLOAT> rho1(cells, PHI_FLOAT(0.0));
  std::vector<PHI_FLOAT> rho2(cells, PHI_FLOAT(0.0));
  const std::vector<std::size_t> active1 = {3U, 27U, 51U, 76U};
  const std::vector<std::size_t> active2 = {8U, 14U, 42U, 69U};
  rho1[3] = 1.25;
  rho1[27] = -0.5;
  rho1[51] = 2.0;
  rho1[76] = 0.125;
  rho2[8] = -0.75;
  rho2[14] = 0.25;
  rho2[42] = 1.5;
  rho2[69] = -0.0625;
  field.fold_fft(rho1.data(), rho2.data(), -1.0f);

  const std::vector<std::size_t> targets = {
      0U, 1U, 9U, 10U, 22U, 37U, 58U, 70U, 78U, 79U};
  std::vector<PHI_FLOAT> direct1(cells,
      std::numeric_limits<PHI_FLOAT>::quiet_NaN());
  std::vector<PHI_FLOAT> direct2(cells,
      std::numeric_limits<PHI_FLOAT>::quiet_NaN());
  const PHI_FLOAT* kernel = field.get_real_space_kernel();
  require(kernel != nullptr, "case-2 setup must retain its real kernel");
  feebl::sparse_direct_fields::solve_potential_nodes(
      rho1.data(), active1, kernel, targets, nx, ny, direct1.data());
  feebl::sparse_direct_fields::solve_potential_nodes(
      rho2.data(), active2, kernel, targets, nx, ny, direct2.data());

  for (std::size_t target : targets) {
    require_close(direct1[target], field.get_phi(1)[target],
                  "beam-1 sparse potential must match FFT");
    require_close(direct2[target], field.get_phi(2)[target],
                  "beam-2 sparse potential must match FFT");
  }

  // The optimized production path gathers fields, not potentials.  Build the
  // exact union for several interpolation points, leave every other direct
  // potential as NaN, and prove both ElectricFieldCIC components agree.
  const std::vector<std::pair<PHI_FLOAT, PHI_FLOAT> > points = {
      {1.2, 1.3}, {2.75, 4.125}, {4.2, 5.3}, {6.7, 8.2}};
  std::vector<std::size_t> field_targets;
  for (const auto& point : points) {
    std::size_t stencil[12];
    feebl::sparse_direct_fields::electric_field_stencil_nodes(
        point.first, point.second, nx, ny, stencil);
    field_targets.insert(field_targets.end(), stencil, stencil + 12U);
  }
  std::sort(field_targets.begin(), field_targets.end());
  field_targets.erase(
      std::unique(field_targets.begin(), field_targets.end()),
      field_targets.end());
  std::fill(direct1.begin(), direct1.end(),
            std::numeric_limits<PHI_FLOAT>::quiet_NaN());
  feebl::sparse_direct_fields::solve_potential_nodes(
      rho1.data(), active1, kernel, field_targets, nx, ny, direct1.data());
  for (const auto& point : points) {
    const Field2 direct_field = cic_field(
        point.first, point.second, 1.0 / 0.7, 1.0 / 0.23,
        direct1.data(), ny);
    const Field2 fft_field = cic_field(
        point.first, point.second, 1.0 / 0.7, 1.0 / 0.23,
        field.get_phi(1), ny);
    require_close(direct_field.x, fft_field.x,
                  "sparse direct Ex must match FFT gather");
    require_close(direct_field.y, fft_field.y,
                  "sparse direct Ey must match FFT gather");
  }

  // The absolute-index convolution must implement all kernel reflections.
  std::vector<PHI_FLOAT> impulse(cells, PHI_FLOAT(0.0));
  impulse[3U * ny + 4U] = 1.0;
  const std::vector<std::size_t> source = {3U * ny + 4U};
  const std::vector<std::size_t> reflected_targets = {
      1U * ny + 2U, 1U * ny + 6U, 5U * ny + 2U, 5U * ny + 6U};
  std::vector<PHI_FLOAT> reflected(cells, PHI_FLOAT(0.0));
  feebl::sparse_direct_fields::solve_potential_nodes(
      impulse.data(), source, kernel, reflected_targets,
      nx, ny, reflected.data());
  for (std::size_t position = 1; position < reflected_targets.size(); ++position)
    require(reflected[reflected_targets[position]] ==
                reflected[reflected_targets[0]],
            "Green kernel reflection must be exact");
}

void test_order_and_range_contracts()
{
  const int nx = 4;
  const int ny = 4;
  std::vector<PHI_FLOAT> rho(16U, 0.0);
  std::vector<PHI_FLOAT> kernel(16U, 1.0);
  std::vector<PHI_FLOAT> phi(16U, 0.0);
  phi[5] = 7.0;
  feebl::sparse_direct_fields::solve_potential_nodes(
      rho.data(), std::vector<std::size_t>(), kernel.data(),
      std::vector<std::size_t>({5U}), nx, ny, phi.data());
  require(phi[5] == 0.0, "empty source slice must produce zero potential");
  phi[6] = 9.0;
  feebl::sparse_direct_fields::solve_potential_nodes(
      rho.data(), std::vector<std::size_t>({2U}), kernel.data(),
      std::vector<std::size_t>(), nx, ny, phi.data());
  require(phi[6] == 9.0, "empty target slice must touch no potential nodes");
  bool rejected = false;
  try {
    feebl::sparse_direct_fields::solve_potential_nodes(
        rho.data(), std::vector<std::size_t>({3U, 2U}), kernel.data(),
        std::vector<std::size_t>({5U}), nx, ny, phi.data());
  } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "unsorted source order must fail closed");
  rejected = false;
  try {
    feebl::sparse_direct_fields::solve_potential_nodes(
        rho.data(), std::vector<std::size_t>({2U}), kernel.data(),
        std::vector<std::size_t>({16U}), nx, ny, phi.data());
  } catch (const std::out_of_range&) { rejected = true; }
  require(rejected, "out-of-grid target must fail closed");
}

}  // namespace

int main()
{
  test_selector_and_crossover();
  test_full_interpolation_stencil();
  test_exact_kernel_direct_against_fft();
  test_order_and_range_contracts();
  std::cout << "sparse-direct field tests passed\n";
  return 0;
}
