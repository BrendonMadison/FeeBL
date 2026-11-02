#include "gpuAdaptiveWaveBatch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using feebl::adaptive_grid::CanonicalDepositDescriptor;
using feebl::adaptive_grid::CanonicalGatherDescriptor;
using feebl::adaptive_grid::CanonicalLatticeInput;
using feebl::adaptive_grid::CanonicalLatticeSnapshot;
using feebl::adaptive_grid::CanonicalWindowInputs;
using feebl::adaptive_grid::UniformWindowPlan;
using feebl::adaptive_grid::UniformWindowPolicy;
using feebl::adaptive_grid::gpu5::AdaptiveEncounterResult;
using feebl::adaptive_grid::gpu5::AdaptiveEncounterSubmission;
using feebl::adaptive_grid::gpu5::AdaptiveGpuOptions;
using feebl::adaptive_grid::gpu5::AdaptiveWaveBatchExecutor;
using feebl::adaptive_grid::gpu5::DepositionVariantId;
using feebl::adaptive_grid::gpu5::FftPlanCacheKey;
using feebl::adaptive_grid::gpu5::GreenSpectrumCacheKey;
using feebl::adaptive_grid::gpu5::WindowClassShape;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double absolute_tolerance,
                  double relative_tolerance, const std::string& message) {
  const double scale = std::max(std::fabs(actual), std::fabs(expected));
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) >
          absolute_tolerance + relative_tolerance * scale) {
    std::cerr << message << ": actual=" << actual
              << " expected=" << expected << '\n';
    throw std::runtime_error(message);
  }
}

std::uint32_t bits(float value) {
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint64_t bits(double value) {
  std::uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

double f_potential_2(double x0, double y0, double dx, double dy) {
  double x, y, sum;
  x = x0 + dx;
  y = y0 + dy;
  sum = x * y * (std::log(x * x + y * y) - 3.0) +
        x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 + dx;
  y = y0 - dy;
  sum -= x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 - dx;
  y = y0 + dy;
  sum -= x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  x = x0 - dx;
  y = y0 - dy;
  sum += x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
  return sum;
}

CanonicalLatticeSnapshot make_lattice(float charge_sign = -1.0f,
                                      std::uint64_t revision = 17) {
  CanonicalLatticeInput input;
  input.n_x = 12;
  input.n_y = 12;
  input.delta_x = 0.25f;
  input.delta_y = 0.5f;
  input.delta_z = 1.5f;
  input.inverse_delta_x = 4.0f;
  input.inverse_delta_y = 2.0f;
  input.offset_x = 6.0f;
  input.offset_y = 6.0f;
  input.intended_min_x = -1.5f;
  input.intended_max_x = 1.5f;
  input.intended_min_y = -3.0f;
  input.intended_max_y = 3.0f;
  input.charge_sign = charge_sign;
  // This is already the complete historical factor, including -charge_sign.
  input.field_kernel_factor = 0.125;
  input.geometry_revision = revision;
  return CanonicalLatticeSnapshot(input);
}

CanonicalWindowInputs make_inputs(int shift) {
  CanonicalWindowInputs inputs;
  inputs.beam1_sources.push_back(CanonicalDepositDescriptor::from_canonical(
      4 + shift, 4 + shift, 0.25f, 0.75f));
  inputs.beam1_sources.push_back(CanonicalDepositDescriptor::from_canonical(
      5 + shift, 5 + shift, 0.60f, 0.30f));
  inputs.beam1_sources.push_back(CanonicalDepositDescriptor::rejected());
  inputs.beam2_sources.push_back(CanonicalDepositDescriptor::from_canonical(
      4 + shift, 5 + shift, 0.40f, 0.20f));
  inputs.beam2_sources.push_back(CanonicalDepositDescriptor::from_canonical(
      5 + shift, 4 + shift, 0.15f, 0.55f));
  inputs.beam2_sources.push_back(CanonicalDepositDescriptor::rejected());

  inputs.beam1_targets.push_back(
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          5.25 + shift, 5.50 + shift));
  inputs.beam1_targets.push_back(
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          4.25 + shift, 4.50 + shift));
  inputs.beam1_targets.push_back(CanonicalGatherDescriptor::rejected());
  inputs.beam2_targets.push_back(
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          4.50 + shift, 5.25 + shift));
  inputs.beam2_targets.push_back(
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          5.50 + shift, 4.25 + shift));
  inputs.beam2_targets.push_back(CanonicalGatherDescriptor::rejected());
  return inputs;
}

UniformWindowPolicy crop_policy() {
  UniformWindowPolicy policy;
  policy.size_classes_x = {6, 12};
  policy.size_classes_y = {6, 12};
  policy.min_n_x = 4;
  policy.min_n_y = 4;
  policy.max_n_x = 12;
  policy.max_n_y = 12;
  policy.extra_guard_cells = 1;
  policy.allow_full_window = true;
  return policy;
}

UniformWindowPolicy full_policy() {
  UniformWindowPolicy policy = crop_policy();
  policy.size_classes_x.clear();
  policy.size_classes_y.clear();
  policy.size_classes_x.push_back(12U);
  policy.size_classes_y.push_back(12U);
  return policy;
}

std::size_t local_index(const UniformWindowPlan& plan,
                        std::int64_t canonical_x,
                        std::int64_t canonical_y) {
  return static_cast<std::size_t>(canonical_x - plan.origin_x()) *
             plan.local_n_y() +
         static_cast<std::size_t>(canonical_y - plan.origin_y());
}

void deposit(const UniformWindowPlan& plan,
             const std::vector<CanonicalDepositDescriptor>& descriptors,
             double charge, std::vector<double>& rho,
             std::size_t& in, std::size_t& out) {
  for (const CanonicalDepositDescriptor& descriptor : descriptors) {
    if (!descriptor.accepted()) {
      ++out;
      continue;
    }
    ++in;
    const double weight_x = static_cast<double>(descriptor.weight_x());
    const double weight_y = static_cast<double>(descriptor.weight_y());
    const double complement_x = 1.0 - weight_x;
    const double complement_y = 1.0 - weight_y;
    const std::size_t base = local_index(
        plan, descriptor.base_x(), descriptor.base_y());
    rho[base] += complement_x * complement_y * charge;
    rho[base + plan.local_n_y()] += weight_x * complement_y * charge;
    rho[base + 1] += complement_x * weight_y * charge;
    rho[base + plan.local_n_y() + 1] +=
        static_cast<double>(descriptor.weight_x() * descriptor.weight_y()) *
        charge;
  }
}

std::vector<double> direct_potential(const UniformWindowPlan& plan,
                                     const CanonicalLatticeSnapshot& lattice,
                                     const std::vector<double>& rho) {
  const std::size_t nx = plan.local_n_x();
  const std::size_t ny = plan.local_n_y();
  const double factor = lattice.field_kernel_factor();
  double phi0 = factor;
  phi0 *= f_potential_2(0.0, 0.0, 0.5 * lattice.delta_x(),
                       0.5 * lattice.delta_y());
  std::vector<double> kernel(nx * ny, 0.0);
  for (std::size_t x = 0; x < nx; ++x)
    for (std::size_t y = 0; y < ny; ++y)
      kernel[x * ny + y] =
          factor * f_potential_2(
                       static_cast<double>(x) * lattice.delta_x(),
                       static_cast<double>(y) * lattice.delta_y(),
                       0.5 * lattice.delta_x(),
                       0.5 * lattice.delta_y()) -
          phi0;
  std::vector<double> phi(nx * ny, 0.0);
  if (std::fabs(lattice.charge_sign()) < 1.0e-5f) return phi;
  for (std::size_t target_x = 0; target_x < nx; ++target_x) {
    for (std::size_t target_y = 0; target_y < ny; ++target_y) {
      double value = 0.0;
      for (std::size_t source_x = 0; source_x < nx; ++source_x)
        for (std::size_t source_y = 0; source_y < ny; ++source_y) {
          const std::size_t dx = target_x > source_x
                                     ? target_x - source_x
                                     : source_x - target_x;
          const std::size_t dy = target_y > source_y
                                     ? target_y - source_y
                                     : source_y - target_y;
          value += rho[source_x * ny + source_y] * kernel[dx * ny + dy];
        }
      phi[target_x * ny + target_y] = value;
    }
  }
  return phi;
}

void gather_one(const UniformWindowPlan& plan,
                const CanonicalLatticeSnapshot& lattice,
                const CanonicalGatherDescriptor& descriptor,
                const std::vector<double>& phi, float* fields) {
  if (!descriptor.accepted() ||
      std::fabs(lattice.charge_sign()) < 1.0e-5f) {
    fields[0] = fields[1] = 0.0f;
    return;
  }
  float h_p = 1.0 - descriptor.y_weight();
  const double phi1_y =
      descriptor.y_weight() *
          phi[local_index(plan, descriptor.y_base_x() + 1,
                          descriptor.y_base_y() + 1)] +
      h_p * phi[local_index(plan, descriptor.y_base_x() + 1,
                            descriptor.y_base_y())];
  const double phi2_y =
      descriptor.y_weight() *
          phi[local_index(plan, descriptor.y_base_x(),
                          descriptor.y_base_y() + 1)] +
      h_p * phi[local_index(plan, descriptor.y_base_x(),
                            descriptor.y_base_y())];
  const double phi3_y =
      descriptor.y_weight() *
          phi[local_index(plan, descriptor.y_base_x() - 1,
                          descriptor.y_base_y() + 1)] +
      h_p * phi[local_index(plan, descriptor.y_base_x() - 1,
                            descriptor.y_base_y())];
  h_p = 1.0 - descriptor.x_weight();
  const double phi1_x =
      descriptor.x_weight() *
          phi[local_index(plan, descriptor.x_base_x() + 1,
                          descriptor.x_base_y() + 1)] +
      h_p * phi[local_index(plan, descriptor.x_base_x(),
                            descriptor.x_base_y() + 1)];
  const double phi2_x =
      descriptor.x_weight() *
          phi[local_index(plan, descriptor.x_base_x() + 1,
                          descriptor.x_base_y())] +
      h_p * phi[local_index(plan, descriptor.x_base_x(),
                            descriptor.x_base_y())];
  const double phi3_x =
      descriptor.x_weight() *
          phi[local_index(plan, descriptor.x_base_x() + 1,
                          descriptor.x_base_y() - 1)] +
      h_p * phi[local_index(plan, descriptor.x_base_x(),
                            descriptor.x_base_y() - 1)];
  double x =
      (descriptor.fractional_x() * (phi1_y - phi2_y) +
       (1.0 - descriptor.fractional_x()) * (phi2_y - phi3_y)) *
      static_cast<double>(lattice.inverse_delta_x());
  double y =
      (descriptor.fractional_y() * (phi1_x - phi2_x) +
       (1.0 - descriptor.fractional_y()) * (phi2_x - phi3_x)) *
      static_cast<double>(lattice.inverse_delta_y());
  fields[0] = static_cast<float>(x);
  fields[1] = static_cast<float>(y);
  fields[0] *= 0.5f;
  fields[1] *= 0.5f;
}

struct OracleResult {
  std::vector<float> fields1;
  std::vector<float> fields2;
  std::size_t beam1_in = 0;
  std::size_t beam1_out = 0;
  std::size_t beam2_in = 0;
  std::size_t beam2_out = 0;
  std::size_t gather_beam1_in = 0;
  std::size_t gather_beam1_out = 0;
  std::size_t gather_beam2_in = 0;
  std::size_t gather_beam2_out = 0;
  double dot = 0.0;
};

OracleResult oracle(const UniformWindowPlan& plan,
                    const CanonicalLatticeSnapshot& lattice,
                    double charge1, double charge2) {
  OracleResult result;
  const std::size_t cells = plan.local_n_x() * plan.local_n_y();
  std::vector<double> rho1(cells, 0.0);
  std::vector<double> rho2(cells, 0.0);
  deposit(plan, plan.beam1_sources(), charge1, rho1,
          result.beam1_in, result.beam1_out);
  deposit(plan, plan.beam2_sources(), charge2, rho2,
          result.beam2_in, result.beam2_out);
  for (std::size_t cell = 0; cell < cells; ++cell)
    result.dot += rho1[cell] * rho2[cell];
  const std::vector<double> phi1 = direct_potential(plan, lattice, rho1);
  const std::vector<double> phi2 = direct_potential(plan, lattice, rho2);
  result.fields1.assign(2 * plan.beam1_targets().size(), 0.0f);
  result.fields2.assign(2 * plan.beam2_targets().size(), 0.0f);
  for (std::size_t particle = 0; particle < plan.beam1_targets().size();
       ++particle) {
    if (plan.beam1_targets()[particle].accepted())
      ++result.gather_beam1_in;
    else
      ++result.gather_beam1_out;
    gather_one(plan, lattice, plan.beam1_targets()[particle], phi2,
               &result.fields1[2 * particle]);
  }
  for (std::size_t particle = 0; particle < plan.beam2_targets().size();
       ++particle) {
    if (plan.beam2_targets()[particle].accepted())
      ++result.gather_beam2_in;
    else
      ++result.gather_beam2_out;
    gather_one(plan, lattice, plan.beam2_targets()[particle], phi1,
               &result.fields2[2 * particle]);
  }
  return result;
}

void compare_result(const AdaptiveEncounterResult& actual,
                    const OracleResult& expected,
                    const std::string& label) {
  require(actual.valid, label + " result is not valid");
  require(actual.beam1_in == expected.beam1_in &&
              actual.beam1_out == expected.beam1_out &&
              actual.beam2_in == expected.beam2_in &&
              actual.beam2_out == expected.beam2_out &&
              actual.gather_beam1_in == expected.gather_beam1_in &&
              actual.gather_beam1_out == expected.gather_beam1_out &&
              actual.gather_beam2_in == expected.gather_beam2_in &&
              actual.gather_beam2_out == expected.gather_beam2_out,
          label + " acceptance counters differ from CPU oracle");
  require(actual.fields1.size() == expected.fields1.size() &&
              actual.fields2.size() == expected.fields2.size(),
          label + " field sizes differ from CPU oracle");
  for (std::size_t index = 0; index < actual.fields1.size(); ++index)
    require_near(actual.fields1[index], expected.fields1[index], 2.0e-6,
                 2.0e-5, label + " beam1 field");
  for (std::size_t index = 0; index < actual.fields2.size(); ++index)
    require_near(actual.fields2[index], expected.fields2[index], 2.0e-6,
                 2.0e-5, label + " beam2 field");
  require_near(actual.fine_lumi_dot, expected.dot, 2.0e-12, 2.0e-12,
               label + " fine luminosity dot");
}

AdaptiveEncounterSubmission submission(const UniformWindowPlan& plan,
                                       std::size_t result_slot,
                                       std::uint64_t encounter,
                                       std::uint64_t revision1,
                                       std::uint64_t revision2) {
  AdaptiveEncounterSubmission result;
  result.plan = &plan;
  result.binding.canonical_result_slot = result_slot;
  result.binding.encounter_id = encounter;
  result.binding.beam1_slice = static_cast<std::int64_t>(encounter * 2);
  result.binding.beam2_slice = static_cast<std::int64_t>(encounter * 2 + 1);
  result.binding.beam1_revision = revision1;
  result.binding.beam2_revision = revision2;
  result.binding.beam1_order_token = UINT64_C(0x11000000) + encounter;
  result.binding.beam2_order_token = UINT64_C(0x22000000) + encounter;
  return result;
}

void require_same_fields(const AdaptiveEncounterResult& first,
                         const AdaptiveEncounterResult& second,
                         const std::string& label) {
  require(first.fields1.size() == second.fields1.size() &&
              first.fields2.size() == second.fields2.size(),
          label + " size mismatch");
  for (std::size_t index = 0; index < first.fields1.size(); ++index)
    require_near(first.fields1[index], second.fields1[index], 3.0e-6,
                 3.0e-5, label + " beam1");
  for (std::size_t index = 0; index < first.fields2.size(); ++index)
    require_near(first.fields2[index], second.fields2[index], 3.0e-6,
                 3.0e-5, label + " beam2");
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device is available\n";
      cudaGetLastError();
      return 77;
    }

    const CanonicalLatticeSnapshot lattice = make_lattice();
    const CanonicalWindowInputs inputs_a = make_inputs(0);
    const CanonicalWindowInputs inputs_b = make_inputs(3);
    const UniformWindowPlan crop_a =
        UniformWindowPlan::build(lattice, inputs_a, crop_policy());
    const UniformWindowPlan crop_b =
        UniformWindowPlan::build(lattice, inputs_b, crop_policy());
    const UniformWindowPlan full_a =
        UniformWindowPlan::build(lattice, inputs_a, full_policy());
    const UniformWindowPlan full_b =
        UniformWindowPlan::build(lattice, inputs_b, full_policy());
    require(!crop_a.empty() && !crop_b.empty() && !full_a.empty() &&
                !full_b.empty(),
            "fixture unexpectedly produced an empty plan");
    require(crop_a.local_n_x() == 6 && crop_a.local_n_y() == 6 &&
                crop_b.local_n_x() == 6 && crop_b.local_n_y() == 6,
            "fixture did not select the cropped 6x6 class");
    require(crop_a.origin_x() != crop_b.origin_x() &&
                crop_a.origin_y() != crop_b.origin_y(),
            "fixture did not exercise per-encounter origins");
    require(full_a.full_window() && full_b.full_window() &&
                full_a.local_n_x() == 12 && full_a.local_n_y() == 12,
            "fixture did not select the full canonical class");
    for (const auto& node : crop_a.required_nodes())
      require(crop_a.contains_canonical_node(node),
              "cropped plan lost a required node");
    for (const auto& node : crop_b.required_nodes())
      require(crop_b.contains_canonical_node(node),
              "shifted cropped plan lost a required node");

    AdaptiveGpuOptions options;
    options.device = 0;
    options.batch_capacity = 2;
    options.maximum_beam1_descriptors = 3;
    options.maximum_beam2_descriptors = 3;
    options.memory_fraction = 0.50;
    options.pinned_io = true;
    options.deposition = DepositionVariantId::GlobalAtomic;
    AdaptiveWaveBatchExecutor executor;
    std::string error;
    require(executor.initialize(options, lattice, error), error);
    const WindowClassShape classes[2] = {{6, 6}, {12, 12}};
    require(executor.prebuild_classes(classes, 2, error), error);

    FftPlanCacheKey fft_key;
    GreenSpectrumCacheKey green_key;
    require(executor.fft_plan_key(classes[0], fft_key, error), error);
    require(executor.green_spectrum_key(classes[0], green_key, error),
            error);
    require(fft_key.logical_n_x == 6 && fft_key.logical_n_y == 6 &&
                fft_key.padded_n_x == 12 && fft_key.padded_n_y == 12 &&
                fft_key.batch_capacity == 2 && fft_key.input_distance == 144 &&
                fft_key.output_distance == 144,
            "FFT cache key omitted a class/PlanMany invariant");
    require(green_key.logical_n_x == 6 && green_key.logical_n_y == 6 &&
                green_key.padded_n_x == 12 && green_key.padded_n_y == 12 &&
                green_key.delta_x_bits == bits(lattice.delta_x()) &&
                green_key.delta_y_bits == bits(lattice.delta_y()) &&
                green_key.delta_z_bits == bits(lattice.delta_z()) &&
                green_key.charge_sign_bits == bits(lattice.charge_sign()) &&
                green_key.field_kernel_factor_bits ==
                    bits(lattice.field_kernel_factor()),
            "Green cache key omitted exact geometry/kernel provenance");

    const double charge1 = 1.25;
    const double charge2 = -0.75;
    const OracleResult crop_oracle_a = oracle(crop_a, lattice, charge1, charge2);
    const OracleResult crop_oracle_b = oracle(crop_b, lattice, charge1, charge2);
    const OracleResult full_oracle_a = oracle(full_a, lattice, charge1, charge2);
    const OracleResult full_oracle_b = oracle(full_b, lattice, charge1, charge2);

    // Scheduler order and canonical publication order are deliberately
    // opposite; both plans share a class but have distinct integer origins.
    AdaptiveEncounterSubmission crop_submissions[2] = {
        submission(crop_a, 1, 101, 10, 20),
        submission(crop_b, 0, 102, 11, 21)};
    std::vector<AdaptiveEncounterResult> crop_results(2);
    require(executor.execute_batch(crop_submissions, 2, charge1, charge2,
                                   crop_results.data(), crop_results.size(),
                                   error),
            error);
    compare_result(crop_results[1], crop_oracle_a, "cropped origin A");
    compare_result(crop_results[0], crop_oracle_b, "cropped origin B");
    require(executor.validate_result_binding(crop_results[1],
                                             crop_submissions[0], charge1,
                                             charge2, error),
            error);
    require(executor.validate_result_binding(crop_results[0],
                                             crop_submissions[1], charge1,
                                             charge2, error),
            error);

    AdaptiveEncounterSubmission full_submissions[2] = {
        submission(full_a, 1, 201, 30, 40),
        submission(full_b, 0, 202, 31, 41)};
    std::vector<AdaptiveEncounterResult> full_results(2);
    require(executor.execute_batch(full_submissions, 2, charge1, charge2,
                                   full_results.data(), full_results.size(),
                                   error),
            error);
    compare_result(full_results[1], full_oracle_a, "full origin A");
    compare_result(full_results[0], full_oracle_b, "full origin B");
    require_same_fields(crop_results[1], full_results[1],
                        "cropped/full equivalence A");
    require_same_fields(crop_results[0], full_results[0],
                        "cropped/full equivalence B");

    // A 1/2 batch must zero the unused fixed-capacity FFT slot, and a
    // following full batch must not observe stale spectrum or staging data.
    AdaptiveEncounterSubmission partial_submission =
        submission(crop_a, 0, 250, 35, 45);
    AdaptiveEncounterResult partial_result;
    require(executor.execute_batch(&partial_submission, 1, charge1, charge2,
                                   &partial_result, 1, error),
            error);
    compare_result(partial_result, crop_oracle_a, "partial-capacity crop");
    std::vector<AdaptiveEncounterResult> reuse_results(2);
    require(executor.execute_batch(crop_submissions, 2, charge1, charge2,
                                   reuse_results.data(), reuse_results.size(),
                                   error),
            error);
    compare_result(reuse_results[1], crop_oracle_a,
                   "post-partial reuse origin A");
    compare_result(reuse_results[0], crop_oracle_b,
                   "post-partial reuse origin B");

    AdaptiveEncounterSubmission stale = crop_submissions[0];
    ++stale.binding.beam1_revision;
    require(!executor.validate_result_binding(crop_results[1], stale,
                                              charge1, charge2, error),
            "changed slice revision was accepted");
    require(error.find("stale") != std::string::npos,
            "revision rejection did not identify a stale binding");
    require(executor.validate_result_binding(crop_results[1],
                                             crop_submissions[0], charge1,
                                             charge2, error),
            error);
    AdaptiveEncounterSubmission stale_order = crop_submissions[0];
    ++stale_order.binding.beam2_order_token;
    require(!executor.validate_result_binding(crop_results[1], stale_order,
                                              charge1, charge2, error),
            "changed particle order token was accepted");
    AdaptiveEncounterSubmission wrong_plan = crop_submissions[0];
    wrong_plan.plan = &crop_b;
    require(!executor.validate_result_binding(crop_results[1], wrong_plan,
                                              charge1, charge2, error),
            "wrong plan/origin was accepted by result binding");
    require(!executor.validate_result_binding(crop_results[1],
                                              crop_submissions[0],
                                              charge1 + 1.0, charge2, error),
            "changed macro charge was accepted by result binding");

    CanonicalWindowInputs empty_inputs;
    empty_inputs.beam1_sources.assign(
        2, CanonicalDepositDescriptor::rejected());
    empty_inputs.beam2_sources.assign(
        2, CanonicalDepositDescriptor::rejected());
    empty_inputs.beam1_targets.assign(
        2, CanonicalGatherDescriptor::rejected());
    empty_inputs.beam2_targets.assign(
        2, CanonicalGatherDescriptor::rejected());
    const UniformWindowPlan empty_plan = UniformWindowPlan::build(
        lattice, empty_inputs, crop_policy());
    require(empty_plan.empty(), "all-rejected fixture is not empty");
    AdaptiveEncounterSubmission empty_submission =
        submission(empty_plan, 0, 275, 36, 46);
    AdaptiveEncounterResult empty_result;
    require(executor.execute_batch(&empty_submission, 1, charge1, charge2,
                                   &empty_result, 1, error),
            error);
    require(empty_result.valid && empty_result.beam1_in == 0 &&
                empty_result.beam1_out == 2 && empty_result.beam2_in == 0 &&
                empty_result.beam2_out == 2 &&
                empty_result.gather_beam1_out == 2 &&
                empty_result.gather_beam2_out == 2 &&
                std::all_of(empty_result.fields1.begin(),
                            empty_result.fields1.end(),
                            [](float value) { return value == 0.0f; }) &&
                std::all_of(empty_result.fields2.begin(),
                            empty_result.fields2.end(),
                            [](float value) { return value == 0.0f; }),
            "empty plan did not publish exact outside/zero semantics");
    require(executor.validate_result_binding(empty_result, empty_submission,
                                             charge1, charge2, error),
            error);

    // Duplicate canonical slots are rejected before any device work and
    // before either destination object is changed.
    std::vector<AdaptiveEncounterResult> sentinels(2);
    sentinels[0].valid = true;
    sentinels[0].fields1.push_back(123.0f);
    sentinels[1].valid = true;
    sentinels[1].fields2.push_back(456.0f);
    AdaptiveEncounterSubmission duplicates[2] = {
        submission(crop_a, 0, 301, 50, 60),
        submission(crop_b, 0, 302, 51, 61)};
    const std::uint64_t calls_before = executor.telemetry().batch_calls;
    require(!executor.execute_batch(duplicates, 2, charge1, charge2,
                                    sentinels.data(), sentinels.size(), error),
            "duplicate result slots were accepted");
    require(sentinels[0].valid && sentinels[0].fields1.size() == 1 &&
                sentinels[0].fields1[0] == 123.0f && sentinels[1].valid &&
                sentinels[1].fields2.size() == 1 &&
                sentinels[1].fields2[0] == 456.0f &&
                executor.telemetry().batch_calls == calls_before,
            "duplicate-slot rejection was not transactional");

    CanonicalWindowInputs oversized_inputs = inputs_a;
    oversized_inputs.beam1_sources.push_back(
        CanonicalDepositDescriptor::rejected());
    oversized_inputs.beam1_targets.push_back(
        CanonicalGatherDescriptor::rejected());
    oversized_inputs.beam2_sources.push_back(
        CanonicalDepositDescriptor::rejected());
    oversized_inputs.beam2_targets.push_back(
        CanonicalGatherDescriptor::rejected());
    const UniformWindowPlan oversized_plan = UniformWindowPlan::build(
        lattice, oversized_inputs, crop_policy());
    AdaptiveEncounterSubmission oversized_submission =
        submission(oversized_plan, 0, 320, 52, 62);
    AdaptiveEncounterResult oversized_result;
    require(!executor.execute_batch(&oversized_submission, 1, charge1,
                                    charge2, &oversized_result, 1, error),
            "descriptor-cap overflow was accepted");
    require(error.find("descriptor caps") != std::string::npos &&
                !oversized_result.valid,
            "descriptor-cap failure was not precise/transactional");
    AdaptiveEncounterSubmission too_many[3] = {
        submission(crop_a, 0, 321, 53, 63),
        submission(crop_a, 1, 322, 54, 64),
        submission(crop_a, 2, 323, 55, 65)};
    std::vector<AdaptiveEncounterResult> too_many_results(3);
    require(!executor.execute_batch(too_many, 3, charge1, charge2,
                                    too_many_results.data(),
                                    too_many_results.size(), error),
            "batch-cap overflow was accepted");
    require(std::all_of(too_many_results.begin(), too_many_results.end(),
                        [](const AdaptiveEncounterResult& result) {
                          return !result.valid;
                        }),
            "batch-cap rejection published a partial result");

    const auto stats = executor.telemetry();
    require(stats.prebuild_complete && stats.classes_prebuilt == 2 &&
                stats.actual_deposition == DepositionVariantId::GlobalAtomic &&
                stats.tracked_device_bytes > 0 &&
                stats.tracked_device_bytes <= stats.memory_budget_bytes &&
                stats.measured_device_high_water_bytes <=
                    stats.memory_budget_bytes &&
                stats.containment_failures == 0 &&
                stats.full_window_encounters == 2 &&
                stats.required_node_visits > 0 &&
                stats.selected_local_cell_visits == 468 &&
                stats.padded_fft_cell_visits == 1872 &&
                stats.actual_planmany_cell_visits == 2016 &&
                stats.canonical_cells_avoided == 540 &&
                stats.padded_fft_slots == 1 && stats.empty_encounters == 1,
            "mode-5 telemetry/memory/containment invariants failed");

    AdaptiveGpuOptions tiny = options;
    tiny.maximum_device_bytes = 1;
    AdaptiveWaveBatchExecutor tiny_executor;
    require(!tiny_executor.initialize(tiny, lattice, error),
            "one-byte device cap was accepted");
    require(error.find("require") != std::string::npos ||
                error.find("budget") != std::string::npos,
            "memory-cap rejection did not identify its resource bound");

    AdaptiveGpuOptions prebuild_oom_options = options;
    prebuild_oom_options.maximum_device_bytes = 8192;
    AdaptiveWaveBatchExecutor prebuild_oom_executor;
    require(prebuild_oom_executor.initialize(prebuild_oom_options, lattice,
                                             error),
            error);
    const std::size_t shared_bytes =
        prebuild_oom_executor.telemetry().tracked_device_bytes;
    require(!prebuild_oom_executor.prebuild_classes(&classes[0], 1, error),
            "prebuild-time device cap was accepted");
    require(prebuild_oom_executor.poisoned() &&
                !prebuild_oom_executor.prebuild_complete() &&
                prebuild_oom_executor.telemetry().tracked_device_bytes ==
                    shared_bytes,
            "failed prebuild did not poison and roll back to shared buffers");

    AdaptiveGpuOptions unimplemented = options;
    unimplemented.deposition = DepositionVariantId::SharedTile;
    AdaptiveWaveBatchExecutor unimplemented_executor;
    require(!unimplemented_executor.initialize(unimplemented, lattice, error),
            "unimplemented shared-tile deposition was accepted");
    require(error.find("not implemented") != std::string::npos,
            "unimplemented deposition did not fail closed explicitly");

    AdaptiveGpuOptions cold_options = options;
    cold_options.batch_capacity = 1;
    AdaptiveWaveBatchExecutor cold_executor;
    require(cold_executor.initialize(cold_options, lattice, error), error);
    const WindowClassShape crop_only = {6, 6};
    require(cold_executor.prebuild_classes(&crop_only, 1, error), error);
    FftPlanCacheKey batch1_fft_key;
    GreenSpectrumCacheKey batch1_green_key;
    require(cold_executor.fft_plan_key(crop_only, batch1_fft_key, error),
            error);
    require(cold_executor.green_spectrum_key(crop_only, batch1_green_key,
                                             error),
            error);
    require(!(batch1_fft_key == fft_key) && batch1_green_key == green_key,
            "batch capacity leaked into Green identity or was omitted from "
            "FFT identity");
    AdaptiveEncounterSubmission cold_submission =
        submission(full_a, 0, 401, 70, 80);
    AdaptiveEncounterResult cold_result;
    require(!cold_executor.execute_batch(&cold_submission, 1, charge1, charge2,
                                         &cold_result, 1, error),
            "unprebuilt full class was accepted");
    require(error.find("not explicitly prebuilt") != std::string::npos &&
                !cold_result.valid,
            "cold class did not fail closed before publication");

    // The raw sign is only FIELD's exact zero gate.  The already-normalized
    // kernel factor remains nonzero and appears separately in provenance.
    const CanonicalLatticeSnapshot zero_lattice = make_lattice(0.0f, 18);
    const UniformWindowPlan zero_plan = UniformWindowPlan::build(
        zero_lattice, inputs_a, crop_policy());
    AdaptiveGpuOptions zero_options = options;
    zero_options.batch_capacity = 1;
    AdaptiveWaveBatchExecutor zero_executor;
    require(zero_executor.initialize(zero_options, zero_lattice, error), error);
    require(zero_executor.prebuild_classes(&crop_only, 1, error), error);
    AdaptiveEncounterSubmission zero_submission =
        submission(zero_plan, 0, 501, 90, 100);
    AdaptiveEncounterResult zero_result;
    require(zero_executor.execute_batch(&zero_submission, 1, charge1, charge2,
                                        &zero_result, 1, error),
            error);
    for (float value : zero_result.fields1)
      require(value == 0.0f, "charge-sign zero gate left a beam1 field");
    for (float value : zero_result.fields2)
      require(value == 0.0f, "charge-sign zero gate left a beam2 field");
    require(std::fabs(zero_result.fine_lumi_dot) > 0.0,
            "zero-field gate incorrectly suppressed deposition/fine dot");
    GreenSpectrumCacheKey zero_green;
    require(zero_executor.green_spectrum_key(crop_only, zero_green, error),
            error);
    require(zero_green.charge_sign_bits == bits(0.0f) &&
                zero_green.field_kernel_factor_bits == bits(0.125),
            "zero-gate Green provenance conflated sign and kernel factor");

    std::cout << "gpu adaptive wave batch tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
