#include "gpuPrimaryResident.h"
#include "particlesCPP.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const double kPi = 3.141592653589793238462643383279502884;

std::size_t index_2d(int x, int y, int ny)
{
  return static_cast<std::size_t>(x) * ny + y;
}

std::vector<double> normalized_kernel_spectrum(
    const std::vector<double>& kernel,
    int padded_nx,
    int padded_ny)
{
  const std::size_t values =
      static_cast<std::size_t>(padded_nx) * padded_ny;
  const double inverse_size = 1.0 / static_cast<double>(values);
  std::vector<double> spectrum(2 * values, 0.0);
  for (int kx = 0; kx < padded_nx; ++kx)
    for (int ky = 0; ky < padded_ny; ++ky)
      {
        double real = 0.0;
        double imaginary = 0.0;
        for (int x = 0; x < padded_nx; ++x)
          for (int y = 0; y < padded_ny; ++y)
            {
              const double value = kernel[index_2d(x, y, padded_ny)];
              if (value == 0.0) continue;
              const double phase = -2.0 * kPi *
                  (static_cast<double>(kx) * x / padded_nx +
                   static_cast<double>(ky) * y / padded_ny);
              real += value * std::cos(phase);
              imaginary += value * std::sin(phase);
            }
        const std::size_t index = index_2d(kx, ky, padded_ny);
        spectrum[2 * index] = real * inverse_size;
        spectrum[2 * index + 1] = imaginary * inverse_size;
      }
  return spectrum;
}

std::vector<double> direct_convolution(const std::vector<double>& rho,
                                       const std::vector<double>& kernel,
                                       int nx,
                                       int ny)
{
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  std::vector<double> result(static_cast<std::size_t>(nx) * ny, 0.0);
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      {
        double sum = 0.0;
        for (int source_x = 0; source_x < nx; ++source_x)
          for (int source_y = 0; source_y < ny; ++source_y)
            {
              const int kernel_x =
                  (x - source_x + padded_nx) % padded_nx;
              const int kernel_y =
                  (y - source_y + padded_ny) % padded_ny;
              sum += rho[index_2d(source_x, source_y, ny)] *
                     kernel[index_2d(kernel_x, kernel_y, padded_ny)];
            }
        result[index_2d(x, y, ny)] = sum;
      }
  return result;
}

void deposit_reference(const std::vector<float>& xy,
                       std::size_t offset,
                       std::size_t count,
                       double charge,
                       int nx,
                       int ny,
                       const feebl::gpu::ResidentPrimaryGrid& grid,
                       std::vector<double>& rho)
{
  rho.assign(static_cast<std::size_t>(nx) * ny, 0.0);
  for (std::size_t local = 0; local < count; ++local)
    {
      const std::size_t particle = offset + local;
      float hx = xy[2 * particle] * grid.delta_x_inverse +
                 (grid.offset_x - 0.5f);
      const int ix = static_cast<int>(std::floor(hx));
      hx -= static_cast<float>(ix);
      float hy = xy[2 * particle + 1] * grid.delta_y_inverse +
                 (grid.offset_y - 0.5f);
      const int iy = static_cast<int>(std::floor(hy));
      hy -= static_cast<float>(iy);
      if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1) continue;
      const double dx = static_cast<double>(hx);
      const double dy = static_cast<double>(hy);
      rho[index_2d(ix, iy, ny)] +=
          (1.0 - dx) * (1.0 - dy) * charge;
      rho[index_2d(ix + 1, iy, ny)] +=
          dx * (1.0 - dy) * charge;
      rho[index_2d(ix, iy + 1, ny)] +=
          (1.0 - dx) * dy * charge;
      rho[index_2d(ix + 1, iy + 1, ny)] += dx * dy * charge;
    }
}

void electric_field_reference(float x,
                              float y,
                              const std::vector<double>& phi,
                              int ny,
                              const feebl::gpu::ResidentPrimaryGrid& grid,
                              float& field_x,
                              float& field_y)
{
  const float absolute_x = x * grid.delta_x_inverse + grid.offset_x;
  const float absolute_y = y * grid.delta_y_inverse + grid.offset_y;
  const int y_i1 = static_cast<int>(std::floor(absolute_x));
  float y_weight = absolute_y - 0.5f;
  const int y_i2 = static_cast<int>(std::floor(y_weight));
  y_weight -= static_cast<float>(y_i2);
  const float y_complement = 1.0f - y_weight;
  const std::size_t y_center = index_2d(y_i1, y_i2, ny);
  const double phi1_y =
      static_cast<double>(y_weight) * phi[y_center + ny + 1] +
      static_cast<double>(y_complement) * phi[y_center + ny];
  const double phi2_y =
      static_cast<double>(y_weight) * phi[y_center + 1] +
      static_cast<double>(y_complement) * phi[y_center];
  const double phi3_y =
      static_cast<double>(y_weight) * phi[y_center - ny + 1] +
      static_cast<double>(y_complement) * phi[y_center - ny];

  const int x_i2 = static_cast<int>(std::floor(absolute_y));
  float x_weight = absolute_x - 0.5f;
  const int x_i1 = static_cast<int>(std::floor(x_weight));
  x_weight -= static_cast<float>(x_i1);
  const float x_complement = 1.0f - x_weight;
  const std::size_t x_center = index_2d(x_i1, x_i2, ny);
  const double phi1_x =
      static_cast<double>(x_weight) * phi[x_center + ny + 1] +
      static_cast<double>(x_complement) * phi[x_center + 1];
  const double phi2_x =
      static_cast<double>(x_weight) * phi[x_center + ny] +
      static_cast<double>(x_complement) * phi[x_center];
  const double phi3_x =
      static_cast<double>(x_weight) * phi[x_center + ny - 1] +
      static_cast<double>(x_complement) * phi[x_center - 1];

  double fractional_x = static_cast<double>(absolute_x);
  double fractional_y = static_cast<double>(absolute_y);
  fractional_x -= std::floor(fractional_x);
  fractional_y -= std::floor(fractional_y);
  const double raw_x =
      (fractional_x * (phi1_y - phi2_y) +
       (1.0 - fractional_x) * (phi2_y - phi3_y)) *
      static_cast<double>(grid.delta_x_inverse);
  const double raw_y =
      (fractional_y * (phi1_x - phi2_x) +
       (1.0 - fractional_y) * (phi2_x - phi3_x)) *
      static_cast<double>(grid.delta_y_inverse);
  field_x = static_cast<float>(raw_x) * 0.5f;
  field_y = static_cast<float>(raw_y) * 0.5f;
}

bool field_domain(float x,
                  float y,
                  int nx,
                  int ny,
                  const feebl::gpu::ResidentPrimaryGrid& grid)
{
  const float absolute_x = x * grid.delta_x_inverse + grid.offset_x;
  const float absolute_y = y * grid.delta_y_inverse + grid.offset_y;
  return x > grid.physical_min_x && x < grid.physical_max_x &&
         y > grid.physical_min_y && y < grid.physical_max_y &&
         absolute_x >= 1.0f && absolute_x < static_cast<float>(nx - 1) &&
         absolute_y >= 1.0f && absolute_y < static_cast<float>(ny - 1);
}

void push_reference(std::vector<float>& xy,
                    std::vector<float>& velocity,
                    const std::vector<float>& energy,
                    std::vector<float>& upsilon,
                    std::size_t offset,
                    std::size_t count,
                    const std::vector<double>& opposing_phi,
                    int nx,
                    int ny,
                    const feebl::gpu::ResidentPrimaryGrid& grid,
                    float step,
                    float scale,
                    bool zero_field)
{
  for (std::size_t local = 0; local < count; ++local)
    {
      const std::size_t particle = offset + local;
      const float x = xy[2 * particle];
      const float y = xy[2 * particle + 1];
      float field_x = 0.0f;
      float field_y = 0.0f;
      if (!zero_field && field_domain(x, y, nx, ny, grid))
        electric_field_reference(x, y, opposing_phi, ny, grid,
                                 field_x, field_y);
      // Authoritative oracle: this is the exact legacy method invoked by
      // GENERAL_GRID::move_particles, not a test-side transcription of its
      // Upsilon formula.  It proves the resident kernel's narrowing order and
      // the otherwise easy-to-miss update when radiation is disabled.
      PARTICLE legacy;
      legacy.setParticle(x, y, 0.0f,
                         velocity[2 * particle],
                         velocity[2 * particle + 1], energy[particle]);
      TRIVECTOR field(field_x, field_y, 0.0f);
      legacy.advanceDueToEBfield(field, step, scale);
      legacy.XYposition(xy[2 * particle], xy[2 * particle + 1]);
      legacy.velocities(velocity[2 * particle],
                        velocity[2 * particle + 1]);
      upsilon[particle] = legacy.getUps();
    }
}

double run_reference(
    std::vector<float>& xy1,
    std::vector<float>& velocity1,
    const std::vector<float>& energy1,
    std::vector<float>& upsilon1,
    std::vector<float>& xy2,
    std::vector<float>& velocity2,
    const std::vector<float>& energy2,
    std::vector<float>& upsilon2,
    const std::vector<feebl::gpu::ResidentPrimaryEncounter>& encounters,
    const std::vector<double>& kernel,
    int nx,
    int ny,
    const feebl::gpu::ResidentPrimaryGrid& grid)
{
  double fine_lumi = 0.0;
  std::vector<double> rho1;
  std::vector<double> rho2;
  for (std::size_t index = 0; index < encounters.size(); ++index)
    {
      const feebl::gpu::ResidentPrimaryEncounter& encounter =
          encounters[index];
      deposit_reference(xy1, encounter.beam1_offset,
                        encounter.beam1_count, encounter.charge1,
                        nx, ny, grid, rho1);
      deposit_reference(xy2, encounter.beam2_offset,
                        encounter.beam2_count, encounter.charge2,
                        nx, ny, grid, rho2);
      for (std::size_t cell = 0; cell < rho1.size(); ++cell)
        fine_lumi += rho1[cell] * rho2[cell];
      const bool zero_field = std::fabs(encounter.charge_sign) < 1.0e-5f;
      std::vector<double> phi1;
      std::vector<double> phi2;
      if (!zero_field)
        {
          phi1 = direct_convolution(rho1, kernel, nx, ny);
          phi2 = direct_convolution(rho2, kernel, nx, ny);
        }
      else
        {
          phi1.assign(rho1.size(), 0.0);
          phi2.assign(rho2.size(), 0.0);
        }
      push_reference(xy1, velocity1, energy1, upsilon1,
                     encounter.beam1_offset, encounter.beam1_count,
                     phi2, nx, ny, grid, encounter.step,
                     encounter.scale1, zero_field);
      push_reference(xy2, velocity2, energy2, upsilon2,
                     encounter.beam2_offset, encounter.beam2_count,
                     phi1, nx, ny, grid, encounter.step,
                     encounter.scale2, zero_field);
    }
  return fine_lumi;
}

bool close_float_vectors(const std::vector<float>& expected,
                         const std::vector<float>& observed,
                         const char* label)
{
  if (expected.size() != observed.size()) return false;
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  bool all_finite = true;
  for (std::size_t index = 0; index < expected.size(); ++index)
    {
      if (!std::isfinite(expected[index]) ||
          !std::isfinite(observed[index]))
        {
          all_finite = false;
          continue;
        }
      const double difference =
          std::fabs(static_cast<double>(expected[index]) - observed[index]);
      maximum_absolute = std::max(maximum_absolute, difference);
      maximum_relative = std::max(
          maximum_relative,
          difference / std::max(1.0e-12, std::fabs(
              static_cast<double>(expected[index]))));
    }
  const bool passed = all_finite && maximum_absolute <= 2.0e-6;
  std::cout << label << ": " << (passed ? "PASS" : "FAIL")
            << " max_abs=" << std::scientific << maximum_absolute
            << " max_rel=" << maximum_relative
            << " all_finite=" << all_finite << '\n';
  return passed;
}

bool contract_rejection_tests()
{
  using feebl::gpu::ResidentPrimaryContract;
  bool passed = true;
  std::string reason;
  ResidentPrimaryContract supported;
  passed = feebl::gpu::resident_primary_contract_supported(
               supported, reason) && reason.empty() && passed;
  const auto rejected = [&reason](const ResidentPrimaryContract& value,
                                  const char* expected) -> bool {
    return !feebl::gpu::resident_primary_contract_supported(value, reason) &&
           reason.find(expected) != std::string::npos;
  };

  ResidentPrimaryContract candidate;
  candidate.beamstrahlung = true;
  passed = rejected(candidate, "beamstrahlung") && passed;
  candidate = ResidentPrimaryContract();
  candidate.photon_emission = true;
  passed = rejected(candidate, "photon") && passed;
  candidate = ResidentPrimaryContract();
  candidate.spin_transport = true;
  passed = rejected(candidate, "spin") && passed;
  candidate = ResidentPrimaryContract();
  candidate.sokolov_ternov = true;
  passed = rejected(candidate, "Sokolov-Ternov") && passed;
  candidate = ResidentPrimaryContract();
  candidate.coherent_secondaries = true;
  passed = rejected(candidate, "coherent") && passed;
  candidate = ResidentPrimaryContract();
  candidate.trident_secondaries = true;
  passed = rejected(candidate, "trident") && passed;
  candidate = ResidentPrimaryContract();
  candidate.tracked_pairs = true;
  passed = rejected(candidate, "pair tracking") && passed;
  candidate = ResidentPrimaryContract();
  candidate.tracked_muons = true;
  passed = rejected(candidate, "muon") && passed;
  candidate = ResidentPrimaryContract();
  candidate.full_electron_luminosity = true;
  passed = rejected(candidate, "full electron-luminosity") && passed;
  candidate = ResidentPrimaryContract();
  candidate.extra_grids = 1;
  passed = rejected(candidate, "extra_grids=0") && passed;
  candidate = ResidentPrimaryContract();
  candidate.interpolation = 1;
  passed = rejected(candidate, "interpolation=2") && passed;
  candidate = ResidentPrimaryContract();
  candidate.generated_primaries = false;
  passed = rejected(candidate, "generated primaries") && passed;
  std::cout << "fail-closed capability matrix: "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

bool run_suite(int device, bool pinned_io)
{
  using feebl::gpu::ResidentPrimaryEncounter;
  using feebl::gpu::ResidentPrimaryGrid;
  using feebl::gpu::ResidentPrimaryOptions;
  using feebl::gpu::ResidentPrimaryPipeline;
  using feebl::gpu::ResidentPrimaryStats;

  const int nx = 8;
  const int ny = 8;
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  std::vector<double> kernel(
      static_cast<std::size_t>(padded_nx) * padded_ny, 0.0);
  kernel[index_2d(0, 0, padded_ny)] = 0.50;
  kernel[index_2d(1, 0, padded_ny)] = 0.10;
  kernel[index_2d(padded_nx - 1, 0, padded_ny)] = 0.10;
  kernel[index_2d(0, 1, padded_ny)] = -0.08;
  kernel[index_2d(0, padded_ny - 1, padded_ny)] = -0.08;
  kernel[index_2d(1, 1, padded_ny)] = 0.03;
  kernel[index_2d(padded_nx - 1, padded_ny - 1, padded_ny)] = 0.03;
  kernel[index_2d(1, padded_ny - 1, padded_ny)] = 0.03;
  kernel[index_2d(padded_nx - 1, 1, padded_ny)] = 0.03;
  const std::vector<double> spectrum =
      normalized_kernel_spectrum(kernel, padded_nx, padded_ny);

  ResidentPrimaryGrid grid;
  grid.delta_x_inverse = 1.0f;
  grid.delta_y_inverse = 1.0f;
  grid.offset_x = 0.0f;
  grid.offset_y = 0.0f;
  grid.physical_min_x = -0.25f;
  grid.physical_max_x = 7.25f;
  grid.physical_min_y = -0.25f;
  grid.physical_max_y = 7.25f;

  ResidentPrimaryOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  ResidentPrimaryPipeline pipeline;
  std::string error;
  if (!pipeline.initialize(options, nx, ny, spectrum.data(), spectrum.size(),
                           grid, -1.0f, 4, 4, error))
    {
      std::cerr << "resident primary initialization failed: " << error << '\n';
      return false;
    }

  const std::vector<float> initial_xy1 = {
      2.20f, 2.40f, 3.10f, 4.20f,
      4.30f, 2.70f, 5.10f, 4.80f};
  const std::vector<float> initial_xy2 = {
      2.60f, 3.20f, 4.50f, 4.40f,
      3.40f, 2.20f, 5.20f, 3.70f};
  const std::vector<float> initial_velocity1 = {
      0.0010f, -0.0007f, -0.0003f, 0.0009f,
      0.0006f, 0.0002f, -0.0008f, -0.0005f};
  const std::vector<float> initial_velocity2 = {
      -0.0002f, 0.0004f, 0.0007f, -0.0006f,
      -0.0009f, 0.0001f, 0.0003f, 0.0008f};
  const std::vector<float> energy1 = {45.6f, 45.4f, 45.8f, 45.5f};
  const std::vector<float> energy2 = {45.3f, 45.7f, 45.2f, 45.9f};
  const std::vector<float> initial_upsilon1 = {
      0.01f, 0.02f, 0.03f, 0.04f};
  const std::vector<float> initial_upsilon2 = {
      0.05f, 0.06f, 0.07f, 0.08f};
  ResidentPrimaryStats statistics;
  if (!pipeline.upload(initial_xy1.data(), initial_velocity1.data(),
                       energy1.data(), initial_upsilon1.data(), energy1.size(),
                       initial_xy2.data(), initial_velocity2.data(),
                       energy2.data(), initial_upsilon2.data(), energy2.size(),
                       statistics, error))
    {
      std::cerr << "resident primary upload failed: " << error << '\n';
      return false;
    }
  bool passed = pipeline.state_is_resident() &&
                statistics.upload_calls == 1 &&
                statistics.h2d_operations == 8 &&
                statistics.synchronizations == 1 &&
                statistics.setup_h2d_operations == 1 &&
                statistics.setup_synchronizations == 1 &&
                statistics.setup_h2d_bytes ==
                    spectrum.size() * sizeof(double) &&
                statistics.state_epoch == 1;

  std::vector<ResidentPrimaryEncounter> schedule(4);
  schedule[0].wave_index = 0;
  schedule[0].beam1_offset = 0;
  schedule[0].beam1_count = 2;
  schedule[0].beam2_offset = 0;
  schedule[0].beam2_count = 2;
  schedule[1].wave_index = 1;
  schedule[1].beam1_offset = 0;
  schedule[1].beam1_count = 2;
  schedule[1].beam2_offset = 2;
  schedule[1].beam2_count = 2;
  schedule[2].wave_index = 1;
  schedule[2].beam1_offset = 2;
  schedule[2].beam1_count = 2;
  schedule[2].beam2_offset = 0;
  schedule[2].beam2_count = 2;
  schedule[3].wave_index = 2;
  schedule[3].beam1_offset = 2;
  schedule[3].beam1_count = 2;
  schedule[3].beam2_offset = 2;
  schedule[3].beam2_count = 2;
  for (std::size_t index = 0; index < schedule.size(); ++index)
    {
      schedule[index].charge1 = 0.31 + 0.02 * index;
      schedule[index].charge2 = 0.27 + 0.015 * index;
      schedule[index].step = 0.08f;
      schedule[index].scale1 = 1.0f;
      schedule[index].scale2 = 0.95f;
      schedule[index].charge_sign = -1.0f;
    }

  // A same-wave range reuse must fail before any device work and leave both
  // telemetry and resident state unchanged.
  std::vector<ResidentPrimaryEncounter> invalid = schedule;
  invalid[2].beam1_offset = 1;
  const ResidentPrimaryStats before_rejection = pipeline.stats();
  ResidentPrimaryStats rejected_statistics;
  const bool overlap_rejected =
      !pipeline.run(grid, invalid.data(), invalid.size(),
                    rejected_statistics, error) &&
      error.find("reuses a particle range") != std::string::npos &&
      pipeline.state_is_resident() &&
      pipeline.stats().batch_calls == before_rejection.batch_calls &&
      pipeline.stats().synchronizations == before_rejection.synchronizations;
  if (!overlap_rejected)
    std::cerr << "overlap pre-device rejection failed: error=\"" << error
              << "\" batches=" << pipeline.stats().batch_calls
              << " syncs=" << pipeline.stats().synchronizations << '\n';
  passed = overlap_rejected && passed;

  const auto rejected_before_device_work =
      [&pipeline, &grid, &schedule, &error](
          const ResidentPrimaryGrid& candidate_grid,
          const std::vector<ResidentPrimaryEncounter>& candidate_schedule,
          const char* label, const char* expected_text) -> bool {
        const ResidentPrimaryStats before = pipeline.stats();
        ResidentPrimaryStats after;
        const bool rejected = !pipeline.run(
            candidate_grid, candidate_schedule.data(),
            candidate_schedule.size(), after, error);
        const bool ok = rejected &&
               error.find(expected_text) != std::string::npos &&
               pipeline.state_is_resident() &&
               pipeline.stats().batch_calls == before.batch_calls &&
               pipeline.stats().synchronizations == before.synchronizations &&
               pipeline.stats().state_epoch == before.state_epoch;
        if (!ok)
          std::cerr << label << " pre-device rejection failed: rejected="
                    << rejected << " error=\"" << error << "\" batches="
                    << pipeline.stats().batch_calls << " syncs="
                    << pipeline.stats().synchronizations << '\n';
        return ok;
      };

  ResidentPrimaryGrid mismatched_grid = grid;
  // Use a normal, material difference.  The smallest subnormal returned by
  // nextafter(0,+1) may compare equal under a process-wide FTZ/DAZ mode set by
  // a CUDA runtime or legacy fast-math translation unit.
  mismatched_grid.offset_x += 0.25f;
  passed = rejected_before_device_work(
               mismatched_grid, schedule, "mesh mismatch", "mesh differs") && passed;

  std::vector<ResidentPrimaryEncounter> sign_mismatch = schedule;
  sign_mismatch[0].charge_sign = 0.0f;
  passed = rejected_before_device_work(
               grid, sign_mismatch, "sign mismatch",
               "immutable-spectrum-sign") && passed;

  std::vector<ResidentPrimaryEncounter> negative_charge = schedule;
  negative_charge[0].charge1 = -negative_charge[0].charge1;
  passed = rejected_before_device_work(
               grid, negative_charge, "negative charge",
               "nonnegative-charge") && passed;

  std::vector<ResidentPrimaryEncounter> derived_overflow = schedule;
  derived_overflow[0].step = std::numeric_limits<float>::max();
  derived_overflow[0].scale1 = 2.0f;
  passed = rejected_before_device_work(
               grid, derived_overflow, "derived overflow", "finite-step") && passed;

  std::vector<float> expected_xy1 = initial_xy1;
  std::vector<float> expected_velocity1 = initial_velocity1;
  std::vector<float> expected_xy2 = initial_xy2;
  std::vector<float> expected_velocity2 = initial_velocity2;
  std::vector<float> expected_upsilon1 = initial_upsilon1;
  std::vector<float> expected_upsilon2 = initial_upsilon2;
  const double expected_lumi = run_reference(
      expected_xy1, expected_velocity1, energy1, expected_upsilon1,
      expected_xy2, expected_velocity2, energy2, expected_upsilon2,
      schedule, kernel, nx, ny, grid);

  if (!pipeline.run(grid, schedule.data(), schedule.size(),
                    statistics, error))
    {
      std::cerr << "resident primary multi-wave run failed: " << error << '\n';
      return false;
    }
  passed = statistics.batch_calls == 1 && statistics.waves == 3 &&
           statistics.encounters == 4 &&
           statistics.particle_updates == 16 &&
           statistics.deposition_calls == 4 &&
           statistics.field_solve_calls == 4 &&
           statistics.gather_push_calls == 4 &&
           statistics.deposit_beam1_in == 8 &&
           statistics.deposit_beam1_out == 0 &&
           statistics.deposit_beam2_in == 8 &&
           statistics.deposit_beam2_out == 0 &&
           statistics.gather_beam1_in == 8 &&
           statistics.gather_beam1_out == 0 &&
           statistics.gather_beam2_in == 8 &&
           statistics.gather_beam2_out == 0 &&
           statistics.d2h_operations == 1 &&
           statistics.synchronizations == 2 &&
           statistics.intermediate_particle_h2d_bytes == 0 &&
           statistics.intermediate_particle_d2h_bytes == 0 &&
           statistics.host_field_materializations == 0 &&
           statistics.state_epoch == 2 &&
           statistics.state_resident && passed;
  const double lumi_error =
      std::fabs(expected_lumi - statistics.fine_lumi_dot_sum);
  passed = lumi_error <=
               5.0e-11 * std::max(1.0, std::fabs(expected_lumi)) && passed;

  std::vector<float> observed_xy1(8, 0.0f);
  std::vector<float> observed_velocity1(8, 0.0f);
  std::vector<float> observed_energy1(4, 0.0f);
  std::vector<float> observed_upsilon1(4, 0.0f);
  std::vector<float> observed_xy2(8, 0.0f);
  std::vector<float> observed_velocity2(8, 0.0f);
  std::vector<float> observed_energy2(4, 0.0f);
  std::vector<float> observed_upsilon2(4, 0.0f);
  if (!pipeline.download(observed_xy1.data(), observed_velocity1.data(),
                         observed_energy1.data(), observed_upsilon1.data(),
                         observed_energy1.size(),
                         observed_xy2.data(), observed_velocity2.data(),
                         observed_energy2.data(), observed_upsilon2.data(),
                         observed_energy2.size(),
                         statistics, error))
    {
      std::cerr << "resident primary download failed: " << error << '\n';
      return false;
    }
  passed = close_float_vectors(expected_xy1, observed_xy1,
                               "beam1 resident positions") && passed;
  passed = close_float_vectors(expected_velocity1, observed_velocity1,
                               "beam1 resident velocities") && passed;
  passed = close_float_vectors(expected_xy2, observed_xy2,
                               "beam2 resident positions") && passed;
  passed = close_float_vectors(expected_velocity2, observed_velocity2,
                               "beam2 resident velocities") && passed;
  passed = std::memcmp(energy1.data(), observed_energy1.data(),
                       energy1.size() * sizeof(float)) == 0 && passed;
  passed = std::memcmp(energy2.data(), observed_energy2.data(),
                       energy2.size() * sizeof(float)) == 0 && passed;
  passed = close_float_vectors(expected_upsilon1, observed_upsilon1,
                               "beam1 resident Upsilon") && passed;
  passed = close_float_vectors(expected_upsilon2, observed_upsilon2,
                               "beam2 resident Upsilon") && passed;
  passed = statistics.download_calls == 1 &&
           statistics.d2h_operations == 9 &&
           statistics.synchronizations == 3 &&
           statistics.state_resident && pipeline.state_is_resident() &&
           passed;

  if (pinned_io)
    {
      // Both a zero-charge encounter and an out-of-grid particle receive the
      // legacy zero field.  advanceDueToEBfield must overwrite a stale,
      // nonzero Upsilon with exactly zero in both cases.
      const std::vector<double> zero_spectrum(spectrum.size(), 0.0);
      ResidentPrimaryPipeline zero_field_pipeline;
      std::string edge_error;
      passed = zero_field_pipeline.initialize(
                   options, nx, ny, zero_spectrum.data(),
                   zero_spectrum.size(), grid, 0.0f, 2, 2,
                   edge_error) && passed;
      const float edge_xy1[4] = {2.2f, 2.4f, -100.0f, -100.0f};
      const float edge_xy2[4] = {2.6f, 3.2f, 100.0f, 100.0f};
      const float edge_velocity[4] = {0.001f, -0.002f, 0.003f, 0.004f};
      const float edge_energy[2] = {45.6f, 45.6f};
      const float stale_upsilon[2] = {0.75f, 0.5f};
      ResidentPrimaryStats edge_stats;
      passed = zero_field_pipeline.upload(
                   edge_xy1, edge_velocity, edge_energy, stale_upsilon, 2,
                   edge_xy2, edge_velocity, edge_energy, stale_upsilon, 2,
                   edge_stats, edge_error) && passed;
      ResidentPrimaryEncounter edge_encounter;
      edge_encounter.wave_index = 0;
      edge_encounter.beam1_count = 2;
      edge_encounter.beam2_count = 2;
      edge_encounter.charge1 = 1.0;
      edge_encounter.charge2 = 1.0;
      edge_encounter.step = 0.08f;
      edge_encounter.scale1 = 1.0f;
      edge_encounter.scale2 = 1.0f;
      edge_encounter.charge_sign = 0.0f;
      passed = zero_field_pipeline.run(
                   grid, &edge_encounter, 1, edge_stats, edge_error) && passed;
      float edge_out_xy1[4] = {};
      float edge_out_xy2[4] = {};
      float edge_out_velocity1[4] = {};
      float edge_out_velocity2[4] = {};
      float edge_out_energy1[2] = {};
      float edge_out_energy2[2] = {};
      float edge_out_upsilon1[2] = {-1.0f, -1.0f};
      float edge_out_upsilon2[2] = {-1.0f, -1.0f};
      passed = zero_field_pipeline.download(
                   edge_out_xy1, edge_out_velocity1, edge_out_energy1,
                   edge_out_upsilon1, 2,
                   edge_out_xy2, edge_out_velocity2, edge_out_energy2,
                   edge_out_upsilon2, 2, edge_stats, edge_error) && passed;

      PARTICLE legacy_zero;
      legacy_zero.setParticle(edge_xy1[0], edge_xy1[1], 0.0f,
                              edge_velocity[0], edge_velocity[1],
                              edge_energy[0]);
      legacy_zero.setUps(stale_upsilon[0]);
      legacy_zero.advanceDueToEBfield(TRIVECTOR(0.0f, 0.0f, 0.0f),
                                      edge_encounter.step, 1.0f);
      passed = legacy_zero.getUps() == 0.0f &&
               edge_out_upsilon1[0] == legacy_zero.getUps() &&
               edge_out_upsilon1[1] == 0.0f &&
               edge_out_upsilon2[0] == 0.0f &&
               edge_out_upsilon2[1] == 0.0f &&
               edge_stats.gather_beam1_out == 1 &&
               edge_stats.gather_beam2_out == 1 && passed;
      std::cout << "authoritative legacy zero-field/out-of-grid Upsilon oracle: "
                << (passed ? "PASS" : "FAIL") << '\n';
    }

  if (pinned_io)
    {
      std::vector<double> nonfinite_spectrum = spectrum;
      nonfinite_spectrum[7] =
          std::numeric_limits<double>::quiet_NaN();
      ResidentPrimaryPipeline rejected_pipeline;
      std::string rejection;
      passed = !rejected_pipeline.initialize(
                   options, nx, ny, nonfinite_spectrum.data(),
                   nonfinite_spectrum.size(), grid, -1.0f, 1, 1,
                   rejection) &&
               rejection.find("non-finite") != std::string::npos && passed;

      ResidentPrimaryPipeline truncated_pipeline;
      passed = !truncated_pipeline.initialize(
                   options, nx, ny, spectrum.data(), spectrum.size() - 1,
                   grid, -1.0f, 1, 1, rejection) &&
               rejection.find("exactly") != std::string::npos && passed;

      ResidentPrimaryGrid overflow_grid = grid;
      overflow_grid.delta_x_inverse =
          std::numeric_limits<float>::max();
      ResidentPrimaryPipeline overflow_upload;
      passed = overflow_upload.initialize(
                   options, nx, ny, spectrum.data(), spectrum.size(),
                   overflow_grid, -1.0f, 1, 1, rejection) && passed;
      const float overflow_xy[2] = {2.0f, 2.0f};
      const float zero_velocity[2] = {0.0f, 0.0f};
      const float positive_energy[1] = {45.6f};
      const float zero_upsilon[1] = {0.0f};
      ResidentPrimaryStats overflow_stats;
      passed = !overflow_upload.upload(
                   overflow_xy, zero_velocity, positive_energy,
                   zero_upsilon, 1,
                   overflow_xy, zero_velocity, positive_energy,
                   zero_upsilon, 1,
                   overflow_stats, rejection) &&
               rejection.find("mesh transforms") != std::string::npos &&
               overflow_upload.stats().upload_calls == 0 &&
               !overflow_upload.state_is_resident() && passed;

      ResidentPrimaryPipeline poisoned_pipeline;
      passed = poisoned_pipeline.initialize(
                   options, nx, ny, spectrum.data(), spectrum.size(),
                   grid, -1.0f, 1, 1, rejection) && passed;
      const float finite_xy1[2] = {2.2f, 2.4f};
      const float finite_xy2[2] = {2.6f, 3.2f};
      passed = poisoned_pipeline.upload(
                   finite_xy1, zero_velocity, positive_energy,
                   zero_upsilon, 1,
                   finite_xy2, zero_velocity, positive_energy,
                   zero_upsilon, 1,
                   overflow_stats, rejection) && passed;
      ResidentPrimaryEncounter overflow_encounter;
      overflow_encounter.wave_index = 0;
      overflow_encounter.beam1_count = 1;
      overflow_encounter.beam2_count = 1;
      overflow_encounter.charge1 =
          std::numeric_limits<double>::max();
      overflow_encounter.charge2 =
          std::numeric_limits<double>::max();
      overflow_encounter.step = 0.08f;
      overflow_encounter.charge_sign = -1.0f;
      passed = !poisoned_pipeline.run(
                   grid, &overflow_encounter, 1, overflow_stats,
                   rejection) &&
               rejection.find("non-finite") != std::string::npos &&
               overflow_stats.context_poisoned &&
               !overflow_stats.state_resident &&
               !poisoned_pipeline.state_is_resident() && passed;
      std::cout << "spectrum/grid/nonfinite hardening: "
                << (passed ? "PASS" : "FAIL") << '\n';
    }

  std::cout << (pinned_io ? "pinned" : "pageable")
            << " 2-slice/3-wave resident lifecycle: "
            << (passed ? "PASS" : "FAIL")
            << " encounters=" << statistics.encounters
            << " particle_updates=" << statistics.particle_updates
            << " syncs=" << statistics.synchronizations
            << " intermediate_h2d="
            << statistics.intermediate_particle_h2d_bytes
            << " intermediate_d2h="
            << statistics.intermediate_particle_d2h_bytes
            << " host_fields=" << statistics.host_field_materializations
            << " fine_lumi_error=" << std::scientific << lumi_error
            << '\n';
  return passed;
}

}  // namespace

int main(int argc, char** argv)
{
  const int device = argc > 1 ? std::atoi(argv[1]) : 0;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 ||
      device >= count)
    {
      std::cerr << "resident primary CUDA test requires device " << device
                << '\n';
      return 2;
    }
  bool passed = contract_rejection_tests();
  passed = run_suite(device, false) && passed;
  passed = run_suite(device, true) && passed;
  std::cout << "resident primary CUDA component: "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
