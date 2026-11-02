#include "beamSoACPP.h"
#include "encounterDAGCPP.h"
#include "gpuPrimaryModernAdapter.h"
#include "primaryParticleIdCPP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const double kPi = 3.141592653589793238462643383279502884;
const double kClassicalElectronRadiusM = 2.81794092e-15;
const double kElectronMassEv = 0.51099906e6;

std::size_t cell_index(int x, int y, int ny) {
  return static_cast<std::size_t>(x) * ny + y;
}

double legacy_float(double value) {
  return static_cast<double>(static_cast<float>(value));
}

double potential_primitive(double x, double y) {
  return x * y * (std::log(x * x + y * y) - 3.0) +
         x * x * std::atan(y / x) + y * y * std::atan(x / y);
}

// Exact algebra used by FIELD::f_potential_2 for the integration_method=2
// Green kernel.  The reduced fixture keeps dx,dy nonzero at all four corners.
double f_potential_2(double x0, double y0, double dx, double dy) {
  return potential_primitive(x0 + dx, y0 + dy) -
         potential_primitive(x0 + dx, y0 - dy) -
         potential_primitive(x0 - dx, y0 + dy) +
         potential_primitive(x0 - dx, y0 - dy);
}

std::vector<double> feebl_spatial_green(int nx,
                                        int ny,
                                        float delta_x,
                                        float delta_y,
                                        float delta_z,
                                        float charge_sign) {
  const int padded_x = 2 * nx;
  const int padded_y = 2 * ny;
  std::vector<double> kernel(
      static_cast<std::size_t>(padded_x) * padded_y, 0.0);
  double factor = -static_cast<double>(charge_sign) * 2.0 *
                  kClassicalElectronRadiusM * kElectronMassEv /
                  static_cast<double>(delta_z);
  factor /= static_cast<double>(delta_x) * delta_y;
  const double phi0 = factor * f_potential_2(
      0.0, 0.0, 0.5 * delta_x, 0.5 * delta_y);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      const double value = factor * f_potential_2(
          static_cast<double>(ix) * delta_x,
          static_cast<double>(iy) * delta_y,
          0.5 * delta_x, 0.5 * delta_y) - phi0;
      kernel[cell_index(ix, iy, padded_y)] = value;
      if (iy >= 1) {
        kernel[cell_index(ix, padded_y - iy, padded_y)] = value;
        if (ix >= 1) {
          kernel[cell_index(padded_x - ix, iy, padded_y)] = value;
          kernel[cell_index(padded_x - ix, padded_y - iy, padded_y)] =
              value;
        }
      } else if (ix >= 1) {
        kernel[cell_index(padded_x - ix, 0, padded_y)] = value;
      }
    }
  }
  return kernel;
}

std::vector<double> normalized_spectrum(const std::vector<double>& kernel,
                                        int padded_x,
                                        int padded_y) {
  const std::size_t cells =
      static_cast<std::size_t>(padded_x) * padded_y;
  const double inverse_cells = 1.0 / static_cast<double>(cells);
  std::vector<double> spectrum(2 * cells, 0.0);
  for (int kx = 0; kx < padded_x; ++kx) {
    for (int ky = 0; ky < padded_y; ++ky) {
      double real = 0.0;
      double imaginary = 0.0;
      for (int x = 0; x < padded_x; ++x) {
        for (int y = 0; y < padded_y; ++y) {
          const double value = kernel[cell_index(x, y, padded_y)];
          if (value == 0.0) continue;
          const double phase = -2.0 * kPi *
              (static_cast<double>(kx) * x / padded_x +
               static_cast<double>(ky) * y / padded_y);
          real += value * std::cos(phase);
          imaginary += value * std::sin(phase);
        }
      }
      const std::size_t index = cell_index(kx, ky, padded_y);
      spectrum[2 * index] = real * inverse_cells;
      spectrum[2 * index + 1] = imaginary * inverse_cells;
    }
  }
  return spectrum;
}

feebl::modern::ParticleRecord primary_record(unsigned int beam,
                                             std::uint64_t ordinal,
                                             bool loaded,
                                             bool spin,
                                             float x,
                                             float y,
                                             float z,
                                             float vx,
                                             float vy) {
  feebl::modern::ParticleRecord particle;
  particle.stable_id = loaded
      ? make_loaded_primary_particle_id(beam, ordinal + 1)
      : make_generated_primary_particle_id(beam, ordinal);
  particle.x = legacy_float(x);
  particle.y = legacy_float(y);
  particle.z = legacy_float(z);
  particle.vx = legacy_float(vx);
  particle.vy = legacy_float(vy);
  particle.energy = legacy_float(45.6);
  // FeeBL parses `particles=24.3` in units of 1e10 particles.  Therefore the
  // supplied FCC-Z card has 2.43e11 particles and n_m=25000 gives 9.72e6
  // physical particles per macro-particle.
  particle.macro_weight = legacy_float(9720000.0);
  particle.upsilon = 0.0;
  particle.sx = spin ? 0.0 : 0.0;
  particle.sy = spin ? 0.0 : 0.0;
  particle.sz = spin ? (beam == 1 ? 0.8 : -0.8) : 0.0;
  particle.species = beam == 1
      ? feebl::modern::ParticleSpecies::Electron
      : feebl::modern::ParticleSpecies::Positron;
  particle.charge_e = beam == 1 ? -1 : 1;
  particle.direction = beam == 1 ? 1 : -1;
  particle.beam_id = beam;
  particle.flags = 0;
  return particle;
}

feebl::modern::BeamSoA make_beam(unsigned int beam,
                                 bool loaded = false,
                                 bool spin = false) {
  feebl::modern::BeamSoA result(3, spin);
  feebl::modern::BeamMetadata metadata;
  metadata.units = feebl::modern::UnitConvention::LegacyFeeBLNmGeV;
  metadata.bunch_population = 2.43e11;
  result.set_metadata(metadata);
  if (beam == 1) {
    result.append_slice({primary_record(
        beam, 0, loaded, spin, -120000.0f, -300.0f, -900.0f,
        0.0f, 1.0e-7f)});
    result.append_slice({});  // Explicit empty CSR range.
    result.append_slice({primary_record(
        beam, 7, loaded, spin, 100000.0f, 220.0f, 900.0f,
        1.0e-7f, -2.0e-7f)});
  } else {
    result.append_slice({primary_record(
        beam, 1, loaded, spin, -50000.0f, 100.0f, -900.0f,
        -1.0e-7f, 0.0f)});
    result.append_slice({primary_record(
                             beam, 4, loaded, spin, 80000.0f, -250.0f,
                             0.0f, 2.0e-7f, 1.0e-7f),
                         primary_record(
                             beam, 9, loaded, spin, 30000.0f, 350.0f,
                             0.0f, -1.0e-7f, -1.0e-7f)});
    result.append_slice({});
  }
  return result;
}

feebl::gpu::Mode3PrimaryCrossingConfig make_config(
    const std::vector<double>& spectrum) {
  const int nx = 8;
  const int ny = 8;
  const float cut_x = 1389005.7f;
  const float cut_y = 4096.62f;
  const float cut_z = 3.0f * 453.125f;
  const float delta_x = 2.0f * cut_x / nx;
  const float delta_y = 2.0f * cut_y / ny;
  const float delta_z = 2.0f * cut_z / 3.0f;

  feebl::gpu::Mode3PrimaryCrossingConfig config;
  config.requested_gpu_mode = 3;
  config.failure_policy =
      feebl::gpu::Mode3FailurePolicy::AbortNoFallback;
  config.resident_options.device = 0;
  config.resident_options.pinned_io = true;
  config.resident_options.memory_fraction = 0.50;
  config.precision = "mixed_qualified";
  config.modern_reduction = "fast_qualified";
  config.integration_method = 2;
  config.nx = nx;
  config.ny = ny;
  config.green_spectrum_interleaved = spectrum.data();
  config.green_spectrum_interleaved_values = spectrum.size();
  config.green_spectrum_provenance_declared = true;
  config.spectrum_integration_method = 2;
  config.spectrum_charge_sign = -1.0f;
  config.grid.delta_x_inverse = 1.0f / delta_x;
  config.grid.delta_y_inverse = 1.0f / delta_y;
  config.grid.offset_x = 0.5f * nx;
  config.grid.offset_y = 0.5f * ny;
  config.grid.physical_min_x =
      -static_cast<float>(nx - 2) / nx * cut_x;
  config.grid.physical_max_x =
      static_cast<float>(nx - 2) / nx * cut_x;
  config.grid.physical_min_y =
      -static_cast<float>(ny - 2) / ny * cut_y;
  config.grid.physical_max_y =
      static_cast<float>(ny - 2) / ny * cut_y;
  config.spectrum_grid = config.grid;
  config.step = delta_z / 2.0f;  // n_t=1.
  config.scale1 = 1.0f;
  config.scale2 = 1.0f;
  config.charge_sign = -1.0f;
  config.four_way_symmetry = false;
  config.input_card_label =
      "fccz/p_fccz reduced mode3: nx=8 ny=8 nz=3 nt=1 nm=25000 no-radiation no-full-lumi";
  config.crossing_index = 17;
  return config;
}

void deposit(const std::vector<float>& xy,
             std::size_t offset,
             std::size_t count,
             double charge,
             int nx,
             int ny,
             const feebl::gpu::ResidentPrimaryGrid& grid,
             std::vector<double>& rho) {
  rho.assign(static_cast<std::size_t>(nx) * ny, 0.0);
  for (std::size_t local = 0; local < count; ++local) {
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
    rho[cell_index(ix, iy, ny)] +=
        (1.0 - dx) * (1.0 - dy) * charge;
    rho[cell_index(ix + 1, iy, ny)] +=
        dx * (1.0 - dy) * charge;
    rho[cell_index(ix, iy + 1, ny)] +=
        (1.0 - dx) * dy * charge;
    rho[cell_index(ix + 1, iy + 1, ny)] += dx * dy * charge;
  }
}

std::vector<double> direct_convolution(const std::vector<double>& rho,
                                       const std::vector<double>& kernel,
                                       int nx,
                                       int ny) {
  const int padded_x = 2 * nx;
  const int padded_y = 2 * ny;
  std::vector<double> phi(static_cast<std::size_t>(nx) * ny, 0.0);
  for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
      double value = 0.0;
      for (int source_x = 0; source_x < nx; ++source_x) {
        for (int source_y = 0; source_y < ny; ++source_y) {
          const int kernel_x =
              (x - source_x + padded_x) % padded_x;
          const int kernel_y =
              (y - source_y + padded_y) % padded_y;
          value += rho[cell_index(source_x, source_y, ny)] *
                   kernel[cell_index(kernel_x, kernel_y, padded_y)];
        }
      }
      phi[cell_index(x, y, ny)] = value;
    }
  }
  return phi;
}

bool field_domain(float x,
                  float y,
                  int nx,
                  int ny,
                  const feebl::gpu::ResidentPrimaryGrid& grid) {
  const float absolute_x = x * grid.delta_x_inverse + grid.offset_x;
  const float absolute_y = y * grid.delta_y_inverse + grid.offset_y;
  return x > grid.physical_min_x && x < grid.physical_max_x &&
         y > grid.physical_min_y && y < grid.physical_max_y &&
         absolute_x >= 1.0f && absolute_x < static_cast<float>(nx - 1) &&
         absolute_y >= 1.0f && absolute_y < static_cast<float>(ny - 1);
}

void electric_field(float x,
                    float y,
                    const std::vector<double>& phi,
                    int ny,
                    const feebl::gpu::ResidentPrimaryGrid& grid,
                    float& field_x,
                    float& field_y) {
  const float absolute_x = x * grid.delta_x_inverse + grid.offset_x;
  const float absolute_y = y * grid.delta_y_inverse + grid.offset_y;
  const int y_i1 = static_cast<int>(std::floor(absolute_x));
  float y_weight = absolute_y - 0.5f;
  const int y_i2 = static_cast<int>(std::floor(y_weight));
  y_weight -= static_cast<float>(y_i2);
  const float y_complement = 1.0f - y_weight;
  const std::size_t y_center = cell_index(y_i1, y_i2, ny);
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
  const std::size_t x_center = cell_index(x_i1, x_i2, ny);
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

void push(std::vector<float>& xy,
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
          float scale) {
  for (std::size_t local = 0; local < count; ++local) {
    const std::size_t particle = offset + local;
    const float x = xy[2 * particle];
    const float y = xy[2 * particle + 1];
    float ex = 0.0f;
    float ey = 0.0f;
    if (field_domain(x, y, nx, ny, grid))
      electric_field(x, y, opposing_phi, ny, grid, ex, ey);
    const float delta_vx = step * (-2.0f * ex) / energy[particle];
    const float delta_vy = step * (-2.0f * ey) / energy[particle];
    const float vx = velocity[2 * particle] + delta_vx;
    const float vy = velocity[2 * particle + 1] + delta_vy;
    velocity[2 * particle] = vx;
    velocity[2 * particle + 1] = vy;
    xy[2 * particle] = x + vx * (step * scale);
    xy[2 * particle + 1] = y + vy * (step * scale);
    const float dz = static_cast<float>(
        static_cast<double>(step) * 1.0e-9 * static_cast<double>(scale));
    const float dz_on_radius = std::sqrt(
        delta_vx * delta_vx + delta_vy * delta_vy);
    const float inverse_radius = dz_on_radius / dz;
    const double electron_mass_gev = 0.51099906e-3;
    upsilon[particle] = static_cast<float>(
        3.86159323e-13 /
        (electron_mass_gev * electron_mass_gev) *
        static_cast<double>(energy[particle]) *
        static_cast<double>(energy[particle]) *
        static_cast<double>(inverse_radius));
  }
}

struct FloatState {
  std::vector<float> xy1;
  std::vector<float> velocity1;
  std::vector<float> energy1;
  std::vector<float> upsilon1;
  std::vector<float> xy2;
  std::vector<float> velocity2;
  std::vector<float> energy2;
  std::vector<float> upsilon2;
};

void pack(const feebl::modern::BeamSoA& beam,
          std::vector<float>& xy,
          std::vector<float>& velocity,
          std::vector<float>& energy,
          std::vector<float>& upsilon) {
  xy.resize(2 * static_cast<std::size_t>(beam.size()));
  velocity.resize(xy.size());
  energy.resize(static_cast<std::size_t>(beam.size()));
  upsilon.resize(static_cast<std::size_t>(beam.size()));
  for (std::size_t index = 0; index < energy.size(); ++index) {
    xy[2 * index] = static_cast<float>(beam.x()[index]);
    xy[2 * index + 1] = static_cast<float>(beam.y()[index]);
    velocity[2 * index] = static_cast<float>(beam.vx()[index]);
    velocity[2 * index + 1] = static_cast<float>(beam.vy()[index]);
    energy[index] = static_cast<float>(beam.energy()[index]);
    upsilon[index] = static_cast<float>(beam.upsilon()[index]);
  }
}

double serial_oracle(const feebl::modern::BeamSoA& beam1,
                     const feebl::modern::BeamSoA& beam2,
                     const feebl::modern::EncounterDAG& dag,
                     const feebl::gpu::Mode3PrimaryCrossingConfig& config,
                     const std::vector<double>& spatial_green,
                     FloatState& state) {
  pack(beam1, state.xy1, state.velocity1, state.energy1, state.upsilon1);
  pack(beam2, state.xy2, state.velocity2, state.energy2, state.upsilon2);
  double fine_lumi = 0.0;
  std::vector<double> rho1;
  std::vector<double> rho2;
  for (const feebl::modern::EncounterWave& wave : dag.waves()) {
    for (feebl::modern::EncounterId id : wave.encounters) {
      const feebl::modern::EncounterNode& node = dag.node(id);
      const auto range1 = beam1.slice_range(node.beam1_slice);
      const auto range2 = beam2.slice_range(node.beam2_slice);
      deposit(state.xy1, static_cast<std::size_t>(range1.first),
              static_cast<std::size_t>(range1.second - range1.first),
              beam1.macro_weight().front(), config.nx, config.ny,
              config.grid, rho1);
      deposit(state.xy2, static_cast<std::size_t>(range2.first),
              static_cast<std::size_t>(range2.second - range2.first),
              beam2.macro_weight().front(), config.nx, config.ny,
              config.grid, rho2);
      for (std::size_t cell = 0; cell < rho1.size(); ++cell)
        fine_lumi += rho1[cell] * rho2[cell];
      const std::vector<double> phi1 =
          direct_convolution(rho1, spatial_green, config.nx, config.ny);
      const std::vector<double> phi2 =
          direct_convolution(rho2, spatial_green, config.nx, config.ny);
      push(state.xy1, state.velocity1, state.energy1, state.upsilon1,
           static_cast<std::size_t>(range1.first),
           static_cast<std::size_t>(range1.second - range1.first),
           phi2, config.nx, config.ny, config.grid,
           config.step, config.scale1);
      push(state.xy2, state.velocity2, state.energy2, state.upsilon2,
           static_cast<std::size_t>(range2.first),
           static_cast<std::size_t>(range2.second - range2.first),
           phi1, config.nx, config.ny, config.grid,
           config.step, config.scale2);
    }
  }
  return fine_lumi;
}

bool compare_component(const std::vector<float>& expected,
                       const std::vector<double>& observed,
                       double absolute_tolerance,
                       const char* label) {
  if (expected.size() != observed.size()) return false;
  double max_absolute = 0.0;
  double max_relative = 0.0;
  bool finite = true;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double wanted = static_cast<double>(expected[index]);
    const double got = observed[index];
    if (!std::isfinite(wanted) || !std::isfinite(got)) {
      finite = false;
      continue;
    }
    const double difference = std::fabs(wanted - got);
    max_absolute = std::max(max_absolute, difference);
    max_relative = std::max(
        max_relative,
        difference / std::max(1.0e-30, std::fabs(wanted)));
  }
  const bool pass = finite && max_absolute <= absolute_tolerance;
  std::cout << label << ": " << (pass ? "PASS" : "FAIL")
            << " finite=" << finite
            << " max_abs=" << std::scientific << max_absolute
            << " max_rel=" << max_relative
            << " abs_tolerance=" << absolute_tolerance << '\n';
  return pass;
}

bool compare_xy(const FloatState& expected,
                const feebl::modern::BeamSoA& beam1,
                const feebl::modern::BeamSoA& beam2) {
  std::vector<double> expected_axis;
  bool pass = true;
  const auto compare_beam = [&expected_axis, &pass](
      const std::vector<float>& xy,
      const std::vector<float>& velocity,
      const feebl::modern::BeamSoA& beam,
      const char* prefix) {
    std::vector<float> x(beam.size());
    std::vector<float> y(beam.size());
    std::vector<float> vx(beam.size());
    std::vector<float> vy(beam.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
      x[index] = xy[2 * index];
      y[index] = xy[2 * index + 1];
      vx[index] = velocity[2 * index];
      vy[index] = velocity[2 * index + 1];
    }
    pass = compare_component(x, beam.x(), 2.0e-6,
                             (std::string(prefix) + " x").c_str()) && pass;
    pass = compare_component(y, beam.y(), 2.0e-6,
                             (std::string(prefix) + " y").c_str()) && pass;
    pass = compare_component(vx, beam.vx(), 2.0e-6,
                             (std::string(prefix) + " vx").c_str()) && pass;
    pass = compare_component(vy, beam.vy(), 2.0e-6,
                             (std::string(prefix) + " vy").c_str()) && pass;
  };
  compare_beam(expected.xy1, expected.velocity1, beam1, "beam1");
  compare_beam(expected.xy2, expected.velocity2, beam2, "beam2");
  return pass;
}

bool invariants_equal(const feebl::modern::BeamSoA& before,
                      const feebl::modern::BeamSoA& after) {
  return before.slice_offsets() == after.slice_offsets() &&
         before.stable_ids() == after.stable_ids() &&
         before.z() == after.z() &&
         before.macro_weight() == after.macro_weight() &&
         before.sx() == after.sx() && before.sy() == after.sy() &&
         before.sz() == after.sz() && before.species() == after.species() &&
         before.charge_e() == after.charge_e() &&
         before.direction() == after.direction() &&
         before.beam_id() == after.beam_id() &&
         before.flags() == after.flags() &&
         before.metadata().units == after.metadata().units &&
         before.metadata().bunch_population ==
             after.metadata().bunch_population;
}

bool mutable_state_equal(const feebl::modern::BeamSoA& first,
                         const feebl::modern::BeamSoA& second) {
  return first.x() == second.x() && first.y() == second.y() &&
         first.vx() == second.vx() && first.vy() == second.vy() &&
         first.energy() == second.energy() &&
         first.upsilon() == second.upsilon();
}

bool reject_preflight(const feebl::modern::BeamSoA& beam1,
                      const feebl::modern::BeamSoA& beam2,
                      const feebl::modern::EncounterDAG& dag,
                      const feebl::gpu::Mode3PrimaryCrossingConfig& config,
                      const char* expected,
                      const char* label) {
  feebl::gpu::Mode3PrimaryCrossingAdapter adapter;
  feebl::gpu::Mode3PrimaryPreflight receipt;
  std::string error;
  const bool rejected = !adapter.preflight(
      beam1, beam2, dag, config, receipt, error);
  const bool pass = rejected && !receipt.passed &&
                    error.find(expected) != std::string::npos;
  std::cout << "preflight " << label << ": "
            << (pass ? "PASS" : "FAIL") << " reason=" << error << '\n';
  return pass;
}

bool structural_rejection_suite(
    const feebl::modern::BeamSoA& beam1,
    const feebl::modern::BeamSoA& beam2,
    const feebl::modern::EncounterDAG& dag,
    const feebl::gpu::Mode3PrimaryCrossingConfig& base) {
  bool pass = true;
  feebl::gpu::Mode3PrimaryCrossingConfig candidate = base;
  candidate.requested_gpu_mode = 2;
  pass = reject_preflight(beam1, beam2, dag, candidate, "only as gpu_mode=3",
                          "wrong selector") && pass;
  candidate = base;
  candidate.failure_policy = feebl::gpu::Mode3FailurePolicy::AllowFallback;
  pass = reject_preflight(beam1, beam2, dag, candidate, "fallback",
                          "fallback policy") && pass;
  candidate = base;
  candidate.precision = "fp64";
  pass = reject_preflight(beam1, beam2, dag, candidate,
                          "mixed_qualified", "unqualified precision") && pass;
  candidate = base;
  candidate.modern_reduction = "deterministic";
  pass = reject_preflight(beam1, beam2, dag, candidate,
                          "fast_qualified", "unqualified reduction") && pass;
  candidate = base;
  candidate.resident_options.pinned_io = false;
  pass = reject_preflight(beam1, beam2, dag, candidate, "pinned_io=1",
                          "pageable I/O") && pass;
  candidate = base;
  candidate.resident_options.contract.beamstrahlung = true;
  pass = reject_preflight(beam1, beam2, dag, candidate, "beamstrahlung",
                          "beamstrahlung") && pass;
  candidate = base;
  candidate.resident_options.contract.photon_emission = true;
  pass = reject_preflight(beam1, beam2, dag, candidate, "photon",
                          "photon emission") && pass;
  candidate = base;
  candidate.resident_options.contract.full_electron_luminosity = true;
  pass = reject_preflight(beam1, beam2, dag, candidate,
                          "full electron-luminosity",
                          "full luminosity") && pass;
  candidate = base;
  candidate.integration_method = 1;
  pass = reject_preflight(beam1, beam2, dag, candidate,
                          "integration_method=2", "non-FFT") && pass;
  candidate = base;
  --candidate.green_spectrum_interleaved_values;
  pass = reject_preflight(beam1, beam2, dag, candidate, "exactly",
                          "truncated spectrum") && pass;
  candidate = base;
  candidate.spectrum_grid.offset_x += 1.0f;
  pass = reject_preflight(beam1, beam2, dag, candidate, "mesh provenance",
                          "spectrum mesh mismatch") && pass;
  candidate = base;
  candidate.spectrum_charge_sign = 0.0f;
  pass = reject_preflight(beam1, beam2, dag, candidate,
                          "charge-sign provenance", "spectrum sign mismatch") &&
         pass;
  candidate = base;
  candidate.step = std::numeric_limits<float>::max();
  candidate.scale1 = std::numeric_limits<float>::max();
  pass = reject_preflight(beam1, beam2, dag, candidate, "step*scale_step",
                          "derived overflow") && pass;
  candidate = base;
  candidate.charge_symmetric_generation = true;
  pass = reject_preflight(beam1, beam2, dag, candidate, "provenance",
                          "charge-symmetric ID mismatch") && pass;

  candidate = base;
  candidate.four_way_symmetry = true;
  feebl::gpu::Mode3PrimaryCrossingAdapter independent_symmetry;
  feebl::gpu::Mode3PrimaryPreflight symmetry_receipt;
  std::string symmetry_error;
  const bool independent_ok = independent_symmetry.preflight(
      beam1, beam2, dag, candidate, symmetry_receipt, symmetry_error);
  std::cout << "preflight independent charge/force symmetry: "
            << (independent_ok ? "PASS" : "FAIL")
            << " reason=" << symmetry_error << '\n';
  pass = independent_ok && pass;

  feebl::modern::BeamSoA nonexact = beam1;
  nonexact.x_data()[0] = 0.1;
  pass = reject_preflight(nonexact, beam2, dag, base, "round-trip exactly",
                          "non-legacy scalar") && pass;
  const feebl::modern::BeamSoA loaded = make_beam(1, true, false);
  pass = reject_preflight(loaded, beam2, dag, base, "generated",
                          "loaded primary") && pass;
  const feebl::modern::BeamSoA spin = make_beam(1, false, true);
  pass = reject_preflight(spin, beam2, dag, base, "spin transport",
                          "spin state") && pass;
  const feebl::modern::EncounterDAG wrong_dag(2, 3, 1);
  pass = reject_preflight(beam1, beam2, wrong_dag, base, "slice counts",
                          "DAG/CSR mismatch") && pass;
  return pass;
}

}  // namespace

int main() {
  const int nx = 8;
  const int ny = 8;
  const float cut_x = 1389005.7f;
  const float cut_y = 4096.62f;
  const float cut_z = 3.0f * 453.125f;
  const float delta_x = 2.0f * cut_x / nx;
  const float delta_y = 2.0f * cut_y / ny;
  const float delta_z = 2.0f * cut_z / 3.0f;
  const std::vector<double> spatial_green = feebl_spatial_green(
      nx, ny, delta_x, delta_y, delta_z, -1.0f);
  const std::vector<double> spectrum = normalized_spectrum(
      spatial_green, 2 * nx, 2 * ny);

  feebl::modern::BeamSoA beam1 = make_beam(1);
  feebl::modern::BeamSoA beam2 = make_beam(2);
  const feebl::modern::BeamSoA before1 = beam1;
  const feebl::modern::BeamSoA before2 = beam2;
  const feebl::modern::EncounterDAG dag(3, 3, 1);
  const feebl::gpu::Mode3PrimaryCrossingConfig config =
      make_config(spectrum);

  bool pass = structural_rejection_suite(beam1, beam2, dag, config);

  feebl::gpu::Mode3PrimaryCrossingAdapter adapter;
  feebl::gpu::Mode3PrimaryPreflight preflight;
  std::string error;
  const bool preflight_ok = adapter.preflight(
      beam1, beam2, dag, config, preflight, error);
  const bool mapping_ok = preflight_ok && preflight.passed &&
      preflight.particle_count1 == beam1.size() &&
      preflight.particle_count2 == beam2.size() &&
      preflight.slice_count1 == 3 && preflight.slice_count2 == 3 &&
      preflight.substeps == 1 && preflight.waves == 5 &&
      preflight.encounters == 9 &&
      preflight.expected_green_spectrum_values == spectrum.size() &&
      preflight.schedule_fingerprint != 0 &&
      preflight.green_spectrum_fingerprint != 0;
  std::cout << "whole BeamSoA + EncounterDAG mapping: "
            << (mapping_ok ? "PASS" : "FAIL")
            << " error=" << error << '\n';
  pass = mapping_ok && pass;

  FloatState expected;
  const double expected_fine_lumi = serial_oracle(
      beam1, beam2, dag, config, spatial_green, expected);

  feebl::gpu::Mode3PrimaryCrossingReport report;
  const bool executed = adapter.execute(
      beam1, beam2, dag, config, report, error);
  if (!executed && error.find("unavailable") != std::string::npos) {
    const bool transactional = invariants_equal(before1, beam1) &&
                               invariants_equal(before2, beam2) &&
                               before1.x() == beam1.x() &&
                               before2.x() == beam2.x() &&
                               !report.outputs_committed;
    std::cout << "CUDA unavailable no-fallback transaction: "
              << (transactional ? "PASS" : "FAIL")
              << " reason=" << error << '\n';
    return pass && transactional ? 0 : 1;
  }
  std::cout << "mode3 execution: " << (executed ? "PASS" : "FAIL")
            << " reason=" << error << '\n';
  pass = executed && pass;
  if (executed) {
    pass = compare_xy(expected, beam1, beam2) && pass;
    pass = compare_component(expected.energy1, beam1.energy(), 0.0,
                             "beam1 energy") && pass;
    pass = compare_component(expected.energy2, beam2.energy(), 0.0,
                             "beam2 energy") && pass;
    pass = compare_component(expected.upsilon1, beam1.upsilon(), 1.0e-12,
                             "beam1 Upsilon") && pass;
    pass = compare_component(expected.upsilon2, beam2.upsilon(), 1.0e-12,
                             "beam2 Upsilon") && pass;
    const double lumi_difference = std::fabs(
        report.resident.fine_lumi_dot_sum - expected_fine_lumi);
    const double lumi_relative = lumi_difference /
        std::max(1.0, std::fabs(expected_fine_lumi));
    // The explicitly selected fast-qualified path permits unordered FP64
    // atomic CIC plus legacy-FP32 coordinate localization.  This diagnostic
    // threshold (1e-6 relative) is still four orders tighter than FeeBL's 1%
    // nominal physics-accuracy envelope.
    const bool lumi_ok = std::isfinite(report.resident.fine_lumi_dot_sum) &&
                         lumi_relative <= 1.0e-6;
    std::cout << "raw fine-grid luminosity oracle: "
              << (lumi_ok ? "PASS" : "FAIL")
              << " expected=" << std::scientific << expected_fine_lumi
              << " observed=" << report.resident.fine_lumi_dot_sum
              << " rel=" << lumi_relative << '\n';
    pass = lumi_ok && pass;

    const bool receipt_ok =
        report.implementation_id ==
            feebl::gpu::mode3_primary_implementation_id() &&
        report.requested_gpu_mode == 3 && report.selected_gpu_mode == 3 &&
        report.precision == "mixed_qualified" &&
        report.modern_reduction == "fast_qualified" &&
        !report.fallback_permitted && !report.fallback_used &&
        report.structural_preflight_passed &&
        report.device_preflight_passed &&
        report.state_uploaded_once && report.all_waves_submitted_once &&
        report.state_downloaded_once && report.outputs_committed &&
        report.slice_offsets_and_order_preserved &&
        report.host_invariants_preserved &&
        report.schedule_fingerprint == preflight.schedule_fingerprint &&
        report.green_spectrum_fingerprint ==
            preflight.green_spectrum_fingerprint &&
        report.resident.upload_calls == 1 &&
        report.resident.batch_calls == 1 &&
        report.resident.download_calls == 1 &&
        report.resident.synchronizations == 3 &&
        report.resident.waves == dag.wave_count() &&
        report.resident.encounters == dag.encounter_count() &&
        report.resident.intermediate_particle_h2d_bytes == 0 &&
        report.resident.intermediate_particle_d2h_bytes == 0 &&
        report.resident.host_field_materializations == 0 &&
        report.resident.nonfinite_state_events == 0 &&
        report.mapped_mutable_components == 6 &&
        report.host_invariant_components == 10;
    std::cout << "mode3 provenance/residency receipt: "
              << (receipt_ok ? "PASS" : "FAIL")
              << " device='" << report.device_description << "'"
              << " upload/batch/download=" << report.resident.upload_calls
              << '/' << report.resident.batch_calls
              << '/' << report.resident.download_calls
              << " sync=" << report.resident.synchronizations
              << " intermediate_state_bytes="
              << (report.resident.intermediate_particle_h2d_bytes +
                  report.resident.intermediate_particle_d2h_bytes)
              << '\n';
    pass = receipt_ok && pass;

    bool repeated_exact = true;
    for (int repetition = 0; repetition < 3; ++repetition) {
      feebl::modern::BeamSoA repeated_beam1 = before1;
      feebl::modern::BeamSoA repeated_beam2 = before2;
      feebl::gpu::Mode3PrimaryCrossingAdapter repeated_adapter;
      feebl::gpu::Mode3PrimaryCrossingReport repeated_report;
      std::string repeated_error;
      repeated_exact = repeated_adapter.execute(
                           repeated_beam1, repeated_beam2, dag, config,
                           repeated_report, repeated_error) &&
                       mutable_state_equal(beam1, repeated_beam1) &&
                       mutable_state_equal(beam2, repeated_beam2) &&
                       repeated_report.resident.fine_lumi_dot_sum ==
                           report.resident.fine_lumi_dot_sum &&
                       repeated_exact;
    }
    std::cout << "fast-qualified same-device repeated results: "
              << (repeated_exact ? "PASS" : "FAIL")
              << " repeats=3\n";
    pass = repeated_exact && pass;

    const bool invariants_ok = invariants_equal(before1, beam1) &&
                               invariants_equal(before2, beam2);
    std::cout << "slice offsets/order and host invariants: "
              << (invariants_ok ? "PASS" : "FAIL") << '\n';
    pass = invariants_ok && pass;

    const feebl::modern::BeamSoA after_once1 = beam1;
    const feebl::modern::BeamSoA after_once2 = beam2;
    feebl::gpu::Mode3PrimaryCrossingReport second_report;
    const bool reuse_rejected = !adapter.execute(
        beam1, beam2, dag, config, second_report, error) &&
        error.find("one crossing/one attempt") != std::string::npos &&
        after_once1.x() == beam1.x() && after_once1.y() == beam1.y() &&
        after_once2.x() == beam2.x() && after_once2.y() == beam2.y();
    std::cout << "second lifecycle rejected without mutation: "
              << (reuse_rejected ? "PASS" : "FAIL")
              << " reason=" << error << '\n';
    pass = reuse_rejected && pass;
  }

  // An unsupported mode-3 request must fail before touching either BeamSoA.
  feebl::modern::BeamSoA fail_beam1 = before1;
  feebl::modern::BeamSoA fail_beam2 = before2;
  feebl::gpu::Mode3PrimaryCrossingConfig fail_config = config;
  fail_config.resident_options.contract.beamstrahlung = true;
  feebl::gpu::Mode3PrimaryCrossingAdapter fail_adapter;
  feebl::gpu::Mode3PrimaryCrossingReport fail_report;
  const bool fail_closed = !fail_adapter.execute(
      fail_beam1, fail_beam2, dag, fail_config, fail_report, error) &&
      error.find("beamstrahlung") != std::string::npos &&
      !fail_report.outputs_committed && !fail_report.fallback_used &&
      before1.x() == fail_beam1.x() && before1.y() == fail_beam1.y() &&
      before2.x() == fail_beam2.x() && before2.y() == fail_beam2.y();
  std::cout << "unsupported physics fail-closed/no mutation: "
            << (fail_closed ? "PASS" : "FAIL")
            << " reason=" << error << '\n';
  pass = fail_closed && pass;

  std::cout << "gpu primary modern mode3 adapter overall: "
            << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}
