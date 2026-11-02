#include "fieldCPP.h"
#include "gridCPP.h"
#include "particlesCPP.h"
#include "uniformWindowPlanCPP.h"
#include "wavefrontFieldWorkspaceCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using feebl::adaptive_grid::CanonicalDepositDescriptor;
using feebl::adaptive_grid::CanonicalGatherDescriptor;
using feebl::adaptive_grid::CanonicalLatticeInput;
using feebl::adaptive_grid::CanonicalLatticeSnapshot;
using feebl::adaptive_grid::CanonicalNode;
using feebl::adaptive_grid::CanonicalWindowInputs;
using feebl::adaptive_grid::UniformWindowPlan;
using feebl::adaptive_grid::UniformWindowPolicy;
using feebl::adaptive_grid::WindowPlanError;
using feebl::adaptive_grid::WindowPlanErrorCode;
using feebl::wavefront_fields::Configuration;
using feebl::wavefront_fields::FieldPrefetch;
using feebl::wavefront_fields::validate_workspace_configuration;

int failures = 0;
std::size_t scenarios_checked = 0;
std::size_t nodes_checked = 0;
double maximum_kernel_relative_error = 0.0;
double maximum_fft_relative_error = 0.0;
double maximum_crop_relative_error = 0.0;
double maximum_field_scaled_error = 0.0;

void check(bool condition, const std::string& description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

template <class Function>
void check_throws(Function function, const std::string& expected_fragment,
                  const std::string& description) {
  bool matched = false;
  try {
    function();
  } catch (const std::exception& error) {
    matched = std::string(error.what()).find(expected_fragment) !=
              std::string::npos;
    if (!matched)
      std::cerr << "wrong exception for " << description << ": "
                << error.what() << '\n';
  }
  check(matched, description);
}

std::size_t index(std::int64_t x, std::int64_t y, std::size_t n_y) {
  return static_cast<std::size_t>(x) * n_y + static_cast<std::size_t>(y);
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool close_scaled(double actual, double expected, double relative,
                  double absolute_scale) {
  return std::abs(actual - expected) <=
         relative * std::max({std::abs(actual), std::abs(expected),
                              absolute_scale});
}

// Independent expression of the rectangular cell integral used by FeeBL's
// case-2 Green kernel.  This test never reads FIELD's transformed spectrum.
double primitive(double x, double y) {
  const double radius2 = x * x + y * y;
  return x * y * (std::log(radius2) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
}

double cell_integral(double x0, double y0, double half_x, double half_y) {
  return primitive(x0 + half_x, y0 + half_y) -
         primitive(x0 + half_x, y0 - half_y) -
         primitive(x0 - half_x, y0 + half_y) +
         primitive(x0 - half_x, y0 - half_y);
}

double independent_kernel(std::size_t delta_x_node,
                          std::size_t delta_y_node, double factor,
                          double spacing_x, double spacing_y) {
  const double self = cell_integral(
      0.0, 0.0, 0.5 * spacing_x, 0.5 * spacing_y);
  return factor *
         (cell_integral(delta_x_node * spacing_x,
                        delta_y_node * spacing_y,
                        0.5 * spacing_x, 0.5 * spacing_y) - self);
}

double independent_kernel_signed(std::int64_t delta_x_node,
                                 std::int64_t delta_y_node, double factor,
                                 double spacing_x, double spacing_y) {
  const double self = cell_integral(
      0.0, 0.0, 0.5 * spacing_x, 0.5 * spacing_y);
  return factor *
         (cell_integral(delta_x_node * spacing_x,
                        delta_y_node * spacing_y,
                        0.5 * spacing_x, 0.5 * spacing_y) - self);
}

double direct_potential(const std::vector<PHI_FLOAT>& rho,
                        std::size_t n_x, std::size_t n_y,
                        const CanonicalNode& target, double factor,
                        double spacing_x, double spacing_y) {
  double result = 0.0;
  for (std::size_t source_x = 0; source_x < n_x; ++source_x) {
    for (std::size_t source_y = 0; source_y < n_y; ++source_y) {
      const double charge = rho[index(source_x, source_y, n_y)];
      if (charge == 0.0) continue;
      const std::size_t separation_x = static_cast<std::size_t>(
          std::llabs(static_cast<long long>(target.x) -
                     static_cast<long long>(source_x)));
      const std::size_t separation_y = static_cast<std::size_t>(
          std::llabs(static_cast<long long>(target.y) -
                     static_cast<long long>(source_y)));
      result += charge * independent_kernel(
                           separation_x, separation_y, factor,
                           spacing_x, spacing_y);
    }
  }
  return result;
}

class ContractGrid final : public GENERAL_GRID {
 public:
  ContractGrid(int n_x, int n_y, float spacing_x, float spacing_y) {
    n_cell_x_ = n_x;
    n_cell_y_ = n_y;
    delta_x_inverse_ = 1.0f / spacing_x;
    delta_y_inverse_ = 1.0f / spacing_y;
    mesh_ = MESH(spacing_x, spacing_y, 1.0f,
                 0.5f * static_cast<float>(n_x),
                 0.5f * static_cast<float>(n_y));
    min_x_ = -static_cast<float>(n_x - 2) * spacing_x * 0.5f;
    max_x_ = static_cast<float>(n_x - 2) * spacing_x * 0.5f;
    min_y_ = -static_cast<float>(n_y - 2) * spacing_y * 0.5f;
    max_y_ = static_cast<float>(n_y - 2) * spacing_y * 0.5f;
  }

  void distribute_coherent_particles(int, int, int) override {}
  void distribute_particles(int, int, int, int) override {}
};

struct ChargedDeposit {
  CanonicalDepositDescriptor descriptor;
  double charge = 0.0;
};

CanonicalLatticeSnapshot make_lattice(std::size_t n_x, std::size_t n_y,
                                      float spacing_x, float spacing_y,
                                      double factor) {
  CanonicalLatticeInput input;
  input.n_x = n_x;
  input.n_y = n_y;
  input.delta_x = spacing_x;
  input.delta_y = spacing_y;
  input.delta_z = 0.75f;
  input.inverse_delta_x = 1.0f / spacing_x;
  input.inverse_delta_y = 1.0f / spacing_y;
  input.offset_x = 0.5f * static_cast<float>(n_x);
  input.offset_y = 0.5f * static_cast<float>(n_y);
  input.intended_min_x = -static_cast<float>(n_x - 2) * spacing_x * 0.5f;
  input.intended_max_x = static_cast<float>(n_x - 2) * spacing_x * 0.5f;
  input.intended_min_y = -static_cast<float>(n_y - 2) * spacing_y * 0.5f;
  input.intended_max_y = static_cast<float>(n_y - 2) * spacing_y * 0.5f;
  input.charge_sign = -1.0f;
  input.field_kernel_factor = factor;
  input.geometry_revision = 7U;
  return CanonicalLatticeSnapshot(input);
}

std::vector<std::size_t> class_bank(std::size_t maximum) {
  const std::size_t candidates[] = {4U, 6U, 8U, 12U, 16U,
                                    20U, 24U, 32U, 40U, 48U};
  std::vector<std::size_t> result;
  for (std::size_t value : candidates)
    if (value <= maximum) result.push_back(value);
  if (result.empty() || result.back() != maximum) result.push_back(maximum);
  return result;
}

UniformWindowPolicy make_policy(std::size_t n_x, std::size_t n_y,
                                std::int64_t guard = 0) {
  UniformWindowPolicy policy;
  policy.size_classes_x = class_bank(n_x);
  policy.size_classes_y = class_bank(n_y);
  policy.min_n_x = 4U;
  policy.min_n_y = 4U;
  policy.max_n_x = n_x;
  policy.max_n_y = n_y;
  policy.extra_guard_cells = guard;
  policy.allow_full_window = true;
  policy.max_bytes = 64U * 1024U * 1024U;
  return policy;
}

void deposit_global(ContractGrid& grid,
                    const std::vector<ChargedDeposit>& deposits,
                    std::vector<PHI_FLOAT>* rho) {
  for (const ChargedDeposit& source : deposits) {
    if (!source.descriptor.accepted()) continue;
    grid.assignChargeToCIC(
        rho->data(), static_cast<int>(source.descriptor.base_x()),
        static_cast<int>(source.descriptor.base_y()),
        source.descriptor.weight_x(), source.descriptor.weight_y(),
        source.charge);
  }
}

void deposit_local(ContractGrid& grid,
                   const std::vector<ChargedDeposit>& deposits,
                   const UniformWindowPlan& plan,
                   std::vector<PHI_FLOAT>* rho) {
  for (const ChargedDeposit& source : deposits) {
    if (!source.descriptor.accepted()) continue;
    grid.assignChargeToCICTranslated(
        rho->data(), static_cast<int>(plan.local_n_x()),
        static_cast<int>(plan.local_n_y()),
        static_cast<int>(plan.origin_x()),
        static_cast<int>(plan.origin_y()),
        static_cast<int>(source.descriptor.base_x()),
        static_cast<int>(source.descriptor.base_y()),
        source.descriptor.weight_x(), source.descriptor.weight_y(),
        source.charge);
  }
}

std::pair<double, double> independent_gather(
    const CanonicalGatherDescriptor& descriptor,
    const std::vector<double>& potential, std::size_t n_y,
    double inverse_x, double inverse_y) {
  const auto phi = [&](std::int64_t x, std::int64_t y) {
    return potential[index(x, y, n_y)];
  };
  const double wy = descriptor.y_weight();
  const double cy = 1.0 - wy;
  const std::int64_t yx = descriptor.y_base_x();
  const std::int64_t yy = descriptor.y_base_y();
  const double p1y = wy * phi(yx + 1, yy + 1) + cy * phi(yx + 1, yy);
  const double p2y = wy * phi(yx, yy + 1) + cy * phi(yx, yy);
  const double p3y = wy * phi(yx - 1, yy + 1) + cy * phi(yx - 1, yy);

  const double wx = descriptor.x_weight();
  const double cx = 1.0 - wx;
  const std::int64_t xx = descriptor.x_base_x();
  const std::int64_t xy = descriptor.x_base_y();
  const double p1x = wx * phi(xx + 1, xy + 1) + cx * phi(xx, xy + 1);
  const double p2x = wx * phi(xx + 1, xy) + cx * phi(xx, xy);
  const double p3x = wx * phi(xx + 1, xy - 1) + cx * phi(xx, xy - 1);

  return std::make_pair(
      0.5 * inverse_x *
          (descriptor.fractional_x() * (p1y - p2y) +
           (1.0 - descriptor.fractional_x()) * (p2y - p3y)),
      0.5 * inverse_y *
          (descriptor.fractional_y() * (p1x - p2x) +
           (1.0 - descriptor.fractional_y()) * (p2x - p3x)));
}

void validate_kernel(const FIELD& field, std::size_t n_x, std::size_t n_y,
                     double factor, double spacing_x, double spacing_y,
                     const std::string& label) {
  const PHI_FLOAT* production = field.get_real_space_kernel();
  check(production != nullptr, label + ": production real-space kernel retained");
  if (!production) return;
  for (std::size_t x = 0; x < n_x; ++x) {
    for (std::size_t y = 0; y < n_y; ++y) {
      const double oracle = independent_kernel(
          x, y, factor, spacing_x, spacing_y);
      const double relative = std::abs(production[index(x, y, n_y)] - oracle) /
                              std::max(1.0, std::abs(oracle));
      maximum_kernel_relative_error =
          std::max(maximum_kernel_relative_error, relative);
      if (relative > 64.0 * std::numeric_limits<double>::epsilon()) {
        std::ostringstream message;
        message << label << ": Green kernel mismatch at (" << x << ',' << y
                << ") relative=" << std::setprecision(17) << relative;
        check(false, message.str());
        return;
      }
    }
  }
}

void run_equivalence_scenario(
    std::size_t n_x, std::size_t n_y, float spacing_x, float spacing_y,
    double factor, const std::vector<ChargedDeposit>& beam1,
    const std::vector<ChargedDeposit>& beam2,
    const std::vector<CanonicalGatherDescriptor>& targets1,
    const std::vector<CanonicalGatherDescriptor>& targets2,
    std::int64_t guard, const std::string& label) {
  ContractGrid grid(static_cast<int>(n_x), static_cast<int>(n_y),
                    spacing_x, spacing_y);
  CanonicalWindowInputs inputs;
  for (const ChargedDeposit& value : beam1)
    inputs.beam1_sources.push_back(value.descriptor);
  for (const ChargedDeposit& value : beam2)
    inputs.beam2_sources.push_back(value.descriptor);
  inputs.beam1_targets = targets1;
  inputs.beam2_targets = targets2;
  const CanonicalLatticeSnapshot lattice =
      make_lattice(n_x, n_y, spacing_x, spacing_y, factor);
  const UniformWindowPlan plan = UniformWindowPlan::build(
      lattice, std::move(inputs), make_policy(n_x, n_y, guard));
  check(!plan.empty(), label + ": nonempty plan");
  if (plan.empty()) return;
  check(plan.local_n_x() <= n_x && plan.local_n_y() <= n_y,
        label + ": crop lies within canonical shape");
  for (const CanonicalNode& node : plan.required_nodes())
    check(plan.contains_canonical_node(node),
          label + ": every required node is inside crop");

  std::vector<PHI_FLOAT> global_rho1(n_x * n_y, 0.0);
  std::vector<PHI_FLOAT> global_rho2(n_x * n_y, 0.0);
  std::vector<PHI_FLOAT> local_rho1(
      plan.local_n_x() * plan.local_n_y(), 0.0);
  std::vector<PHI_FLOAT> local_rho2(local_rho1.size(), 0.0);
  deposit_global(grid, beam1, &global_rho1);
  deposit_global(grid, beam2, &global_rho2);
  deposit_local(grid, beam1, plan, &local_rho1);
  deposit_local(grid, beam2, plan, &local_rho2);

  for (std::size_t local_x = 0; local_x < plan.local_n_x(); ++local_x) {
    for (std::size_t local_y = 0; local_y < plan.local_n_y(); ++local_y) {
      const std::size_t global_x =
          static_cast<std::size_t>(plan.origin_x()) + local_x;
      const std::size_t global_y =
          static_cast<std::size_t>(plan.origin_y()) + local_y;
      check(double_bits(local_rho1[index(local_x, local_y, plan.local_n_y())]) ==
                double_bits(global_rho1[index(global_x, global_y, n_y)]),
            label + ": beam-1 translated CIC is bit exact");
      check(double_bits(local_rho2[index(local_x, local_y, plan.local_n_y())]) ==
                double_bits(global_rho2[index(global_x, global_y, n_y)]),
            label + ": beam-2 translated CIC is bit exact");
    }
  }

  FFT_SERVER global_fourier;
  FIELD global_field(2, static_cast<int>(n_x), static_cast<int>(n_y),
                     &global_fourier);
  global_field.enable_real_space_kernel(true);
  global_field.dist_init(factor, spacing_x, spacing_y, &global_fourier);
  validate_kernel(global_field, n_x, n_y, factor, spacing_x, spacing_y,
                  label + "/global");
  global_field.fold_fft(global_rho1.data(), global_rho2.data(), -1.0f);

  FFT_SERVER local_fourier;
  FIELD local_field(2, static_cast<int>(plan.local_n_x()),
                    static_cast<int>(plan.local_n_y()), &local_fourier);
  local_field.enable_real_space_kernel(true);
  local_field.dist_init(factor, spacing_x, spacing_y, &local_fourier);
  validate_kernel(local_field, plan.local_n_x(), plan.local_n_y(), factor,
                  spacing_x, spacing_y, label + "/crop");
  local_field.fold_fft(local_rho1.data(), local_rho2.data(), -1.0f);

  const PHI_FLOAT* global_phi[] = {global_field.get_phi(1),
                                   global_field.get_phi(2)};
  const PHI_FLOAT* local_phi[] = {local_field.get_phi(1),
                                  local_field.get_phi(2)};
  const std::vector<PHI_FLOAT>* global_rho[] = {&global_rho1, &global_rho2};
  std::array<std::vector<double>, 2> oracle_phi = {
      std::vector<double>(n_x * n_y,
                          std::numeric_limits<double>::quiet_NaN()),
      std::vector<double>(n_x * n_y,
                          std::numeric_limits<double>::quiet_NaN())};

  double potential_scale = 1.0;
  for (int beam = 0; beam < 2; ++beam) {
    for (const CanonicalNode& node : plan.required_nodes()) {
      const double oracle = direct_potential(
          *global_rho[beam], n_x, n_y, node, factor,
          spacing_x, spacing_y);
      oracle_phi[beam][index(node.x, node.y, n_y)] = oracle;
      potential_scale = std::max(potential_scale, std::abs(oracle));
      const double full_value = global_phi[beam][index(node.x, node.y, n_y)];
      const CanonicalNode local = plan.to_local(node);
      const double crop_value =
          local_phi[beam][index(local.x, local.y, plan.local_n_y())];
      const double fft_error = std::abs(full_value - oracle) /
                               std::max(1.0, std::abs(oracle));
      const double crop_error = std::abs(crop_value - oracle) /
                                std::max(1.0, std::abs(oracle));
      maximum_fft_relative_error =
          std::max(maximum_fft_relative_error, fft_error);
      maximum_crop_relative_error =
          std::max(maximum_crop_relative_error, crop_error);
      if (fft_error > 2.0e-11 || crop_error > 2.0e-11) {
        std::ostringstream message;
        message << label << ": direct-convolution mismatch beam=" << beam + 1
                << " node=(" << node.x << ',' << node.y << ") full="
                << std::setprecision(17) << full_value << " crop="
                << crop_value << " oracle=" << oracle << " errors=("
                << fft_error << ',' << crop_error << ')';
        check(false, message.str());
        return;
      }
      ++nodes_checked;
    }
  }

  const std::vector<CanonicalGatherDescriptor>* targets[] = {&targets1,
                                                              &targets2};
  for (int beam = 0; beam < 2; ++beam) {
    for (const CanonicalGatherDescriptor& target : *targets[beam]) {
      if (!target.accepted()) continue;
      const std::pair<double, double> oracle = independent_gather(
          target, oracle_phi[beam], n_y, 1.0 / spacing_x,
          1.0 / spacing_y);
      const TRIVECTOR full = grid.ElectricFieldCICTranslatedCanonical(
          global_phi[beam], static_cast<int>(n_x), static_cast<int>(n_y),
          0, 0, static_cast<int>(target.y_base_x()),
          static_cast<int>(target.y_base_y()), target.y_weight(),
          static_cast<int>(target.x_base_x()),
          static_cast<int>(target.x_base_y()), target.x_weight(),
          target.fractional_x(), target.fractional_y());
      const TRIVECTOR crop = grid.ElectricFieldCICTranslatedCanonical(
          local_phi[beam], static_cast<int>(plan.local_n_x()),
          static_cast<int>(plan.local_n_y()),
          static_cast<int>(plan.origin_x()),
          static_cast<int>(plan.origin_y()),
          static_cast<int>(target.y_base_x()),
          static_cast<int>(target.y_base_y()), target.y_weight(),
          static_cast<int>(target.x_base_x()),
          static_cast<int>(target.x_base_y()), target.x_weight(),
          target.fractional_x(), target.fractional_y());
      const double field_scale =
          potential_scale * std::max(1.0 / spacing_x, 1.0 / spacing_y);
      for (int component = 0; component < 2; ++component) {
        const double reference = component == 0 ? oracle.first : oracle.second;
        const double full_error = std::abs(full(component) - reference) /
                                  std::max(1.0, field_scale);
        const double crop_error = std::abs(crop(component) - reference) /
                                  std::max(1.0, field_scale);
        maximum_field_scaled_error = std::max(
            maximum_field_scaled_error, std::max(full_error, crop_error));
        check(float_bits(full(component)) == float_bits(crop(component)),
              label + ": canonical and crop fields are bit-identical");
        // TRIVECTOR is binary32.  The independent oracle retains binary64
        // through convolution and gather, so two binary32 epsilons is a
        // sub-ULP-at-typical-magnitude, representation-aware gate rather
        // than a physics-level percentage tolerance.
        const double binary32_gate =
            2.0 * std::numeric_limits<float>::epsilon();
        if (full_error > binary32_gate || crop_error > binary32_gate) {
          std::ostringstream message;
          message << label << ": field mismatch component=" << component
                  << " full=" << std::setprecision(17) << full(component)
                  << " crop=" << crop(component) << " oracle=" << reference
                  << " scaled_errors=(" << full_error << ',' << crop_error
                  << ')';
          check(false, message.str());
          return;
        }
      }
      check(crop(2) == 0.0 && full(2) == 0.0,
            label + ": longitudinal field remains exactly zero");
    }
  }
  ++scenarios_checked;
}

void test_cic_partition_and_closure() {
  std::cout << "CIC partition and charge closure\n";
  const std::array<float, 7> weights = {
      0.0f, std::nextafter(0.0f, 1.0f), 0.125f, 0.5f, 0.875f,
      std::nextafter(1.0f, 0.0f), 0.33333334f};
  const std::array<double, 5> charges = {0.0, 1.0, -3.25, 1.0e-18, 1.0e18};
  ContractGrid grid(8, 9, 0.25f, 0.75f);
  for (float h_x : weights) {
    for (float h_y : weights) {
      for (double charge : charges) {
        std::vector<PHI_FLOAT> full(8U * 9U, 0.0);
        std::vector<PHI_FLOAT> translated(4U * 5U, 0.0);
        GENERAL_GRID::assignChargeToCICWithStride(
            full.data(), 9, 3, 4, h_x, h_y, charge);
        grid.assignChargeToCICTranslated(
            translated.data(), 4, 5, 2, 3, 3, 4, h_x, h_y, charge);
        // Production performs += into zeroed storage, which canonicalizes a
        // negative zero on IEEE-754 hardware.  Include that addition when
        // requiring bit identity at every corner.
        const std::array<double, 4> expected = {
            0.0 + (1.0 - h_x) * (1.0 - h_y) * charge,
            0.0 + h_x * (1.0 - h_y) * charge,
            0.0 + (1.0 - h_x) * h_y * charge,
            0.0 + h_x * h_y * charge};
        const std::array<double, 4> observed = {
            full[index(3, 4, 9)], full[index(4, 4, 9)],
            full[index(3, 5, 9)], full[index(4, 5, 9)]};
        const std::array<double, 4> shifted = {
            translated[index(1, 1, 5)], translated[index(2, 1, 5)],
            translated[index(1, 2, 5)], translated[index(2, 2, 5)]};
        for (std::size_t corner = 0; corner < 4; ++corner) {
          check(double_bits(observed[corner]) == double_bits(expected[corner]),
                "CIC corner preserves established operand order");
          check(double_bits(shifted[corner]) == double_bits(observed[corner]),
                "translated CIC corner is bit-identical");
        }
        const double total = observed[0] + observed[1] + observed[2] +
                             observed[3];
        // The established fourth corner evaluates h_x*h_y in binary32
        // before multiplying PHI_FLOAT charge; the other three corners are
        // promoted by the 1.0 literals.  Therefore closure is intentionally
        // a binary32-level contract even though rho storage is binary64.
        const double bound = 2.0 * std::numeric_limits<float>::epsilon() *
                             std::max(1.0, std::abs(charge));
        if (!(std::abs(total - charge) <= bound)) {
          std::ostringstream message;
          message << "four CIC weights conserve macro charge to roundoff"
                  << " hx=" << std::setprecision(9) << h_x
                  << " hy=" << h_y << " charge=" << charge
                  << " total=" << total << " error="
                  << std::abs(total - charge) << " bound=" << bound;
          check(false, message.str());
        }
      }
    }
  }
  check_throws(
      [&] {
        std::vector<PHI_FLOAT> rho(16U, 0.0);
        grid.assignChargeToCICTranslated(
            rho.data(), 4, 4, 4, 4, 3, 4, 0.5f, 0.5f, 1.0);
      },
      "outside", "translated deposition outside its crop fails closed");
}

void test_empty_zero_halo_and_out_of_domain() {
  std::cout << "Empty, zero-charge, halo, and out-of-domain cases\n";
  const CanonicalLatticeSnapshot lattice =
      make_lattice(16, 20, 0.25f, 0.75f, -0.03125);
  CanonicalWindowInputs empty;
  empty.beam1_sources.push_back(CanonicalDepositDescriptor::rejected());
  empty.beam2_targets.push_back(CanonicalGatherDescriptor::rejected());
  const UniformWindowPlan empty_plan = UniformWindowPlan::build(
      lattice, std::move(empty), make_policy(16, 20));
  check(empty_plan.empty() && empty_plan.estimated_bytes() == 0,
        "all-rejected encounter is allocation-free");

  CanonicalWindowInputs halo;
  halo.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(7, 9, 0.0f, 0.0f));
  const CanonicalGatherDescriptor far_target =
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(11.25, 12.75);
  halo.beam1_targets.push_back(far_target);
  UniformWindowPolicy halo_policy = make_policy(16, 20, 2);
  const UniformWindowPlan halo_plan = UniformWindowPlan::build(
      lattice, std::move(halo), halo_policy);
  check(halo_plan.accepted_source_count() == 1 &&
            halo_plan.accepted_target_count() == 1 &&
            halo_plan.gather_box().valid && halo_plan.source_box().valid &&
            halo_plan.guarded_box().width() >= halo_plan.required_box().width(),
        "one-particle plus gather halo is retained with explicit guard");
  for (const CanonicalNode& node : far_target.nodes())
    check(halo_plan.contains_canonical_node(node),
          "every halo target stencil node remains in the crop");

  CanonicalWindowInputs outside;
  outside.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(15, 3, 0.0f, 0.0f));
  bool outside_rejected = false;
  try {
    (void)UniformWindowPlan::build(
        lattice, std::move(outside), make_policy(16, 20));
  } catch (const WindowPlanError& error) {
    outside_rejected = error.code() == WindowPlanErrorCode::OutOfCanonicalBank;
  }
  check(outside_rejected,
        "accepted descriptor outside canonical bank fails closed");

  ContractGrid grid(16, 20, 0.25f, 0.75f);
  check(!grid.coordinatesInGridRange(-1000.0f, 0.0f) &&
            !grid.coordinatesInGridRange(0.0f, 1000.0f) &&
            !grid.coordinatesInGridRange(
                std::numeric_limits<float>::quiet_NaN(), 0.0f),
        "physical out-of-domain and NaN targets are rejected");

  std::vector<ChargedDeposit> zero = {{
      CanonicalDepositDescriptor::from_canonical(7, 9, 0.25f, 0.75f), 0.0}};
  run_equivalence_scenario(
      16, 20, 0.25f, 0.75f, -0.03125, zero, zero,
      {CanonicalGatherDescriptor::from_legacy_absolute_coordinates(8.25, 10.25)},
      {}, 0, "zero-charge-single-particle");
}

void test_translation_and_symmetry() {
  std::cout << "Integer translation and Green symmetry\n";
  const CanonicalLatticeSnapshot lattice =
      make_lattice(32, 24, 0.3f, 0.55f, -0.017);
  auto build = [&](std::int64_t shift_x, std::int64_t shift_y) {
    CanonicalWindowInputs inputs;
    inputs.beam1_sources.push_back(CanonicalDepositDescriptor::from_canonical(
        8 + shift_x, 7 + shift_y, 0.25f, 0.625f));
    inputs.beam1_targets.push_back(
        CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
            9.125 + shift_x, 8.375 + shift_y));
    return UniformWindowPlan::build(
        lattice, std::move(inputs), make_policy(32, 24));
  };
  const UniformWindowPlan first = build(0, 0);
  const UniformWindowPlan shifted = build(7, 5);
  check(first.local_n_x() == shifted.local_n_x() &&
            first.local_n_y() == shifted.local_n_y() &&
            shifted.origin_x() - first.origin_x() == 7 &&
            shifted.origin_y() - first.origin_y() == 5,
        "integer translation shifts origin without changing topology");
  check(first.plan_hash() != shifted.plan_hash(),
        "translation remains bound into immutable plan identity");

  for (std::size_t dx = 0; dx < 11; ++dx)
    for (std::size_t dy = 0; dy < 9; ++dy) {
      const double value = independent_kernel_signed(
          static_cast<std::int64_t>(dx), static_cast<std::int64_t>(dy),
          -0.017, 0.3, 0.55);
      const double reflected_x = independent_kernel_signed(
          -static_cast<std::int64_t>(dx), static_cast<std::int64_t>(dy),
          -0.017, 0.3, 0.55);
      const double reflected_y = independent_kernel_signed(
          static_cast<std::int64_t>(dx), -static_cast<std::int64_t>(dy),
          -0.017, 0.3, 0.55);
      check(close_scaled(value, reflected_x,
                         64.0 * std::numeric_limits<double>::epsilon(), 1.0) &&
                close_scaled(value, reflected_y,
                             64.0 * std::numeric_limits<double>::epsilon(),
                             1.0),
            "direct Green oracle is reflection symmetric");
    }
}

void test_wavefront_nonfinite_configuration() {
  std::cout << "Wavefront non-finite configuration rejection\n";
  Configuration valid;
  valid.n_x = 4;
  valid.n_y = 4;
  valid.n_z = 1;
  valid.n_t = 1;
  valid.cut_x = 1.0f;
  valid.cut_y = 1.0f;
  valid.cut_z = 1.0f;
  valid.macro_charge1 = 1.0;
  valid.macro_charge2 = 1.0;
  valid.physical_particles1 = 1.0;
  valid.physical_particles2 = 1.0;
  Configuration nonfinite_charge(valid);
  nonfinite_charge.macro_charge1 =
      std::numeric_limits<double>::quiet_NaN();
  check_throws(
      [&] { validate_workspace_configuration(nonfinite_charge); },
      "non-finite macro charge",
      "production wavefront rejects a NaN macro charge");

  Configuration nonfinite_population(valid);
  nonfinite_population.physical_particles2 =
      std::numeric_limits<double>::infinity();
  check_throws(
      [&] { validate_workspace_configuration(nonfinite_population); },
      "non-finite macro charge",
      "production wavefront rejects an infinite physical population");

  Configuration nonfinite_geometry(valid);
  nonfinite_geometry.cut_x = std::numeric_limits<float>::quiet_NaN();
  check_throws(
      [&] { validate_workspace_configuration(nonfinite_geometry); },
      "invalid grid geometry",
      "production wavefront rejects a NaN grid extent");
}

void test_result_binding_negatives() {
  std::cout << "Result binding negative paths\n";
  const CanonicalLatticeSnapshot lattice =
      make_lattice(16, 16, 0.25f, 0.5f, -0.03125);
  CanonicalWindowInputs inputs;
  inputs.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(7, 7, 0.25f, 0.75f));
  inputs.beam1_targets.push_back(
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          7.25, 7.75));
  const UniformWindowPlan plan = UniformWindowPlan::build(
      lattice, std::move(inputs), make_policy(16, 16));
  FieldPrefetch fields;
  std::vector<PARTICLE*> empty1;
  std::vector<PARTICLE*> empty2;
  feebl::wavefront_fields::validate_field_result_binding(
      fields, 99, 100, empty1, empty2);
  check(true, "fixed result without adaptive plan retains legacy no-op binding");

  fields.window_plan = std::make_shared<const UniformWindowPlan>(plan);
  fields.bound_slice1 = 3;
  fields.bound_slice2 = 4;
  fields.bound_particle_count1 = 1;
  fields.bound_particle_count2 = 0;
  fields.lattice_hash = plan.lattice_hash();
  fields.result_binding_hash = 0;
  PARTICLE particle;
  particle.setParticle(1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 45.6f);
  std::vector<PARTICLE*> bound1(1U, &particle);
  check_throws(
      [&] {
        feebl::wavefront_fields::validate_field_result_binding(
            fields, 4, 4, bound1, empty2);
      },
      "slice/count", "wrong slice cannot consume an adaptive result");
  check_throws(
      [&] {
        feebl::wavefront_fields::bind_field_result(
            fields, 3, 4, empty1, empty2);
      },
      "descriptor count",
      "mismatched descriptor count cannot publish an adaptive result");
  feebl::wavefront_fields::bind_field_result(
      fields, 3, 4, bound1, empty2);
  feebl::wavefront_fields::validate_field_result_binding(
      fields, 3, 4, bound1, empty2);
  check(fields.result_binding_hash != 0U,
        "authoritative external-backend publication binds a valid result");
  particle.setParticle(1.25f, 2.0f, 0.0f, 0.0f, 0.0f, 45.6f);
  check_throws(
      [&] {
        feebl::wavefront_fields::validate_field_result_binding(
            fields, 3, 4, bound1, empty2);
      },
      "particle-order/revision",
      "in-place particle movement invalidates a prefetched field result");
  feebl::wavefront_fields::bind_field_result(
      fields, 3, 4, bound1, empty2);
  fields.lattice_hash ^= 1U;
  check_throws(
      [&] {
        feebl::wavefront_fields::validate_field_result_binding(
            fields, 3, 4, bound1, empty2);
      },
      "lattice descriptor", "wrong lattice cannot consume an adaptive result");
  fields.lattice_hash = plan.lattice_hash();
  fields.result_binding_hash ^= UINT64_C(1);
  check_throws(
      [&] {
        feebl::wavefront_fields::validate_field_result_binding(
            fields, 3, 4, bound1, empty2);
      },
      "particle-order/revision",
      "unbound particle ordering cannot consume an adaptive result");
}

void test_randomized_equivalence() {
  std::cout << "Randomized global/crop/direct-oracle equivalence\n";
  std::mt19937_64 random(0x9e3779b97f4a7c15ULL);
  const std::array<std::pair<std::size_t, std::size_t>, 6> shapes = {
      std::make_pair(16U, 32U), std::make_pair(32U, 16U),
      std::make_pair(24U, 40U), std::make_pair(40U, 24U),
      std::make_pair(24U, 24U), std::make_pair(32U, 40U)};
  for (std::size_t scenario = 0; scenario < 36; ++scenario) {
    const std::size_t n_x = shapes[scenario % shapes.size()].first;
    const std::size_t n_y = shapes[scenario % shapes.size()].second;
    const float spacing_x = scenario % 2 == 0 ? 0.25f : 0.7f;
    const float spacing_y = scenario % 3 == 0 ? 0.8f : 0.375f;
    const double factor = scenario % 2 == 0 ? -0.03125 : 0.017578125;
    const bool low_edge = scenario % 6 == 0;
    const bool high_edge = scenario % 6 == 1;
    const std::int64_t anchor_x =
        low_edge ? 1 : (high_edge ? static_cast<std::int64_t>(n_x) - 5
                                  : 5 + static_cast<std::int64_t>(
                                            random() % (n_x - 10)));
    const std::int64_t anchor_y =
        low_edge ? 1 : (high_edge ? static_cast<std::int64_t>(n_y) - 5
                                  : 5 + static_cast<std::int64_t>(
                                            random() % (n_y - 10)));
    std::vector<ChargedDeposit> beam1;
    std::vector<ChargedDeposit> beam2;
    for (std::size_t particle = 0; particle < 9; ++particle) {
      std::int64_t bx = std::max<std::int64_t>(
          0, std::min<std::int64_t>(n_x - 2,
              anchor_x + static_cast<std::int64_t>(random() % 4)));
      std::int64_t by = std::max<std::int64_t>(
          0, std::min<std::int64_t>(n_y - 2,
              anchor_y + static_cast<std::int64_t>(random() % 4)));
      if (particle == 0 && low_edge) {
        bx = 0;
        by = 0;
      } else if (particle == 0 && high_edge) {
        bx = static_cast<std::int64_t>(n_x) - 2;
        by = static_cast<std::int64_t>(n_y) - 2;
      }
      const float hx = static_cast<float>((random() >> 11) * 0x1.0p-53);
      const float hy = static_cast<float>((random() >> 11) * 0x1.0p-53);
      const double charge =
          (0.125 + static_cast<double>(random() % 1000) / 997.0) *
          (particle % 4 == 0 ? 1.0e6 : 1.0);
      (particle % 2 == 0 ? beam1 : beam2).push_back(
          {CanonicalDepositDescriptor::from_canonical(bx, by, hx, hy),
           charge});
    }
    beam1.push_back({CanonicalDepositDescriptor::rejected(), 99.0});
    const double target_x = std::max(
        1.125, std::min(static_cast<double>(n_x) - 2.125,
                        static_cast<double>(anchor_x) + 1.375));
    const double target_y = std::max(
        1.125, std::min(static_cast<double>(n_y) - 2.125,
                        static_cast<double>(anchor_y) + 1.625));
    const std::vector<CanonicalGatherDescriptor> targets1 = {
        CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
            target_x, target_y),
        CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
            std::nextafter(target_x, 0.0),
            std::nextafter(target_y, std::numeric_limits<double>::infinity()))};
    const std::vector<CanonicalGatherDescriptor> targets2 = {
        CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
            std::min(static_cast<double>(n_x) - 2.125, target_x + 0.375),
            std::min(static_cast<double>(n_y) - 2.125, target_y + 0.25))};
    std::ostringstream label;
    label << "random-" << scenario << '-' << n_x << 'x' << n_y;
    run_equivalence_scenario(
        n_x, n_y, spacing_x, spacing_y, factor, beam1, beam2,
        targets1, targets2, static_cast<std::int64_t>(scenario % 3),
        label.str());
  }
}

}  // namespace

int main() {
  test_cic_partition_and_closure();
  test_empty_zero_halo_and_out_of_domain();
  test_translation_and_symmetry();
  test_wavefront_nonfinite_configuration();
  test_result_binding_negatives();
  test_randomized_equivalence();

  if (failures != 0) {
    std::cerr << "mode-1 CPU field validation FAILED failures=" << failures
              << " scenarios=" << scenarios_checked
              << " nodes=" << nodes_checked << '\n';
    return 1;
  }
  std::cout << std::setprecision(9)
            << "mode-1 CPU field validation PASS scenarios="
            << scenarios_checked << " nodes=" << nodes_checked
            << " max_kernel_relative=" << maximum_kernel_relative_error
            << " max_full_fft_relative=" << maximum_fft_relative_error
            << " max_crop_fft_relative=" << maximum_crop_relative_error
            << " max_field_scaled=" << maximum_field_scaled_error << '\n';
  return 0;
}
