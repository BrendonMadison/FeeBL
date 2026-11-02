#include "gpuResident.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// Construct the exact spectrum contract consumed by FIELD::fold_fft and by
// ResidentFieldSolver: FFT(kernel)/N as interleaved FP64 complex values.  The
// expected answer below is computed independently as a direct spatial cyclic
// convolution, so the test does not merely reproduce the implementation's FFT
// sequence.
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

bool close_vectors(const std::vector<double>& expected,
                   const std::vector<double>& observed,
                   const std::string& label)
{
  double reference_scale = 0.0;
  double maximum_error = 0.0;
  double l1_error = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index)
    {
      reference_scale = std::max(reference_scale, std::fabs(expected[index]));
      const double error = std::fabs(expected[index] - observed[index]);
      maximum_error = std::max(maximum_error, error);
      l1_error += error;
    }
  const double max_limit = 2.0e-11 * std::max(1.0, reference_scale);
  const double l1_limit =
      2.0e-11 * std::max(1.0, reference_scale * expected.size());
  const bool passed = maximum_error <= max_limit && l1_error <= l1_limit;
  std::cout << label << ": " << (passed ? "PASS" : "FAIL")
            << " max_abs=" << std::scientific << std::setprecision(6)
            << maximum_error << " l1=" << l1_error << '\n';
  return passed;
}

std::vector<double> first_rho(int nx, int ny, double phase)
{
  std::vector<double> values(static_cast<std::size_t>(nx) * ny);
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      values[index_2d(x, y, ny)] =
          phase + 0.31 * x - 0.17 * y + 0.07 * x * y;
  return values;
}

struct DepositReference
{
  std::vector<double> rho;
  std::size_t in;
  std::size_t out;
};

float particle_coordinate(float mesh_coordinate,
                          float inverse,
                          float offset)
{
  return (mesh_coordinate - (offset - 0.5f)) / inverse;
}

DepositReference deposit_reference(const std::vector<float>& xy,
                                   double charge,
                                   int nx,
                                   int ny,
                                   float inverse_x,
                                   float inverse_y,
                                   float offset_x,
                                   float offset_y)
{
  DepositReference reference;
  reference.rho.assign(static_cast<std::size_t>(nx) * ny, 0.0);
  reference.in = 0;
  reference.out = 0;
  for (std::size_t particle = 0; particle < xy.size() / 2; ++particle)
    {
      float hx = xy[2 * particle] * inverse_x + (offset_x - 0.5f);
      const int ix = static_cast<int>(std::floor(hx));
      hx -= static_cast<float>(ix);
      float hy = xy[2 * particle + 1] * inverse_y + (offset_y - 0.5f);
      const int iy = static_cast<int>(std::floor(hy));
      hy -= static_cast<float>(iy);
      if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1)
        {
          ++reference.out;
          continue;
        }
      ++reference.in;
      const double dx = static_cast<double>(hx);
      const double dy = static_cast<double>(hy);
      reference.rho[index_2d(ix, iy, ny)] +=
          (1.0 - dx) * (1.0 - dy) * charge;
      reference.rho[index_2d(ix + 1, iy, ny)] +=
          dx * (1.0 - dy) * charge;
      reference.rho[index_2d(ix, iy + 1, ny)] +=
          (1.0 - dx) * dy * charge;
      reference.rho[index_2d(ix + 1, iy + 1, ny)] += dx * dy * charge;
    }
  return reference;
}

void symmetrize_reference(std::vector<double>& rho, int nx, int ny)
{
  for (int ix = 0; ix < nx / 2; ++ix)
    for (int iy = 0; iy < ny / 2; ++iy)
      {
        const std::size_t j1 = index_2d(ix, iy, ny);
        const std::size_t j2 = index_2d(nx - 1 - ix, iy, ny);
        const std::size_t j3 = index_2d(ix, ny - 1 - iy, ny);
        const std::size_t j4 = index_2d(nx - 1 - ix, ny - 1 - iy, ny);
        const double mean =
            0.25 * (rho[j1] + rho[j2] + rho[j3] + rho[j4]);
        rho[j1] = mean;
        rho[j2] = mean;
        rho[j3] = mean;
        rho[j4] = mean;
      }
}

double fine_lumi_reference(const std::vector<double>& rho1,
                           const std::vector<double>& rho2)
{
  double sum = 0.0;
  for (std::size_t cell = 0; cell < rho1.size(); ++cell)
    sum += rho1[cell] * rho2[cell];
  return sum;
}

struct FieldReference
{
  std::vector<float> fields;
  std::size_t in;
  std::size_t out;
  bool shared_domain_agrees;
};

double legacy_weighted_pair(float weight,
                            double first,
                            float complement,
                            double second)
{
  // Volatile stores make the intended no-contraction legacy source sequence
  // explicit even though the repository's global C++ flags include
  // -ffast-math.  gpuResident.cu is compiled with --fmad=false.
  volatile double first_product = static_cast<double>(weight) * first;
  volatile double second_product = static_cast<double>(complement) * second;
  volatile double sum = first_product + second_product;
  return sum;
}

void legacy_electric_field(float x,
                           float y,
                           const std::vector<double>& phi,
                           int ny,
                           float inverse_x,
                           float inverse_y,
                           float offset_x,
                           float offset_y,
                           float& field_x,
                           float& field_y)
{
  volatile float x_product = x * inverse_x;
  volatile float y_product = y * inverse_y;
  volatile float absolute_x_store = x_product + offset_x;
  volatile float absolute_y_store = y_product + offset_y;
  const float absolute_x = absolute_x_store;
  const float absolute_y = absolute_y_store;

  const int y_i1 = static_cast<int>(std::floor(absolute_x));
  volatile float y_shift = absolute_y - 0.5f;
  const int y_i2 = static_cast<int>(std::floor(y_shift));
  volatile float y_weight_store = y_shift - static_cast<float>(y_i2);
  const float y_weight = y_weight_store;
  volatile float y_complement_store = 1.0f - y_weight;
  const float y_complement = y_complement_store;
  const std::size_t y_center = index_2d(y_i1, y_i2, ny);
  const double phi1_y = legacy_weighted_pair(
      y_weight, phi[y_center + ny + 1], y_complement,
      phi[y_center + ny]);
  const double phi2_y = legacy_weighted_pair(
      y_weight, phi[y_center + 1], y_complement, phi[y_center]);
  const double phi3_y = legacy_weighted_pair(
      y_weight, phi[y_center - ny + 1], y_complement,
      phi[y_center - ny]);

  const int x_i2 = static_cast<int>(std::floor(absolute_y));
  volatile float x_shift = absolute_x - 0.5f;
  const int x_i1 = static_cast<int>(std::floor(x_shift));
  volatile float x_weight_store = x_shift - static_cast<float>(x_i1);
  const float x_weight = x_weight_store;
  volatile float x_complement_store = 1.0f - x_weight;
  const float x_complement = x_complement_store;
  const std::size_t x_center = index_2d(x_i1, x_i2, ny);
  const double phi1_x = legacy_weighted_pair(
      x_weight, phi[x_center + ny + 1], x_complement,
      phi[x_center + 1]);
  const double phi2_x = legacy_weighted_pair(
      x_weight, phi[x_center + ny], x_complement, phi[x_center]);
  const double phi3_x = legacy_weighted_pair(
      x_weight, phi[x_center + ny - 1], x_complement,
      phi[x_center - 1]);

  volatile double fractional_x_store =
      static_cast<double>(absolute_x) - std::floor(absolute_x);
  volatile double fractional_y_store =
      static_cast<double>(absolute_y) - std::floor(absolute_y);
  const double fractional_x = fractional_x_store;
  const double fractional_y = fractional_y_store;
  volatile double x_difference1 = phi1_y - phi2_y;
  volatile double x_difference2 = phi2_y - phi3_y;
  volatile double x_term1 = fractional_x * x_difference1;
  volatile double x_term2 = (1.0 - fractional_x) * x_difference2;
  volatile double x_sum = x_term1 + x_term2;
  volatile double raw_x = x_sum * static_cast<double>(inverse_x);
  volatile double y_difference1 = phi1_x - phi2_x;
  volatile double y_difference2 = phi2_x - phi3_x;
  volatile double y_term1 = fractional_y * y_difference1;
  volatile double y_term2 = (1.0 - fractional_y) * y_difference2;
  volatile double y_sum = y_term1 + y_term2;
  volatile double raw_y = y_sum * static_cast<double>(inverse_y);
  volatile float trivec_x = static_cast<float>(raw_x);
  volatile float trivec_y = static_cast<float>(raw_y);
  volatile float final_x = trivec_x * 0.5f;
  volatile float final_y = trivec_y * 0.5f;
  field_x = final_x;
  field_y = final_y;
}

bool independent_stencil_safe_domain(float x,
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

  // Independently validate every index touched by phiValuesY/phiValuesX,
  // rather than restating the compact normalized inequalities.
  const int y_i1 = static_cast<int>(std::floor(absolute_x));
  const int y_i2 = static_cast<int>(std::floor(absolute_y - 0.5f));
  const int x_i1 = static_cast<int>(std::floor(absolute_x - 0.5f));
  const int x_i2 = static_cast<int>(std::floor(absolute_y));
  return y_i1 - 1 >= 0 && y_i1 + 1 < nx &&
         y_i2 >= 0 && y_i2 + 1 < ny &&
         x_i1 >= 0 && x_i1 + 1 < nx &&
         x_i2 - 1 >= 0 && x_i2 + 1 < ny;
}

FieldReference field_reference(const std::vector<float>& xy,
                               const std::vector<double>& opposing_phi,
                               int nx,
                               int ny,
                               float physical_min_x,
                               float physical_max_x,
                               float physical_min_y,
                               float physical_max_y,
                               float inverse_x,
                               float inverse_y,
                               float offset_x,
                               float offset_y,
                               bool zero_field)
{
  FieldReference reference;
  reference.fields.assign(xy.size(), 0.0f);
  reference.in = 0;
  reference.out = 0;
  reference.shared_domain_agrees = true;
  for (std::size_t particle = 0; particle < xy.size() / 2; ++particle)
    {
      const bool independent = independent_stencil_safe_domain(
          xy[2 * particle], xy[2 * particle + 1],
          physical_min_x, physical_max_x, physical_min_y, physical_max_y,
          inverse_x, inverse_y, offset_x, offset_y, nx, ny);
      const bool shared = feebl::gpu::resident_field_point_in_intended_domain(
          xy[2 * particle], xy[2 * particle + 1],
          physical_min_x, physical_max_x, physical_min_y, physical_max_y,
          inverse_x, inverse_y, offset_x, offset_y, nx, ny);
      reference.shared_domain_agrees =
          reference.shared_domain_agrees && independent == shared;
      if (!independent)
        {
          ++reference.out;
          continue;
        }
      ++reference.in;
      if (!zero_field)
        legacy_electric_field(
            xy[2 * particle], xy[2 * particle + 1], opposing_phi, ny,
            inverse_x, inverse_y, offset_x, offset_y,
            reference.fields[2 * particle],
            reference.fields[2 * particle + 1]);
    }
  return reference;
}

std::uint32_t ordered_float_bits(float value)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

bool exact_field_vectors(const std::vector<float>& expected,
                         const std::vector<float>& observed,
                         const std::string& label)
{
  bool exact = expected.size() == observed.size();
  double maximum_absolute = 0.0;
  std::uint64_t maximum_ulp = 0;
  const std::size_t values = std::min(expected.size(), observed.size());
  for (std::size_t index = 0; index < values; ++index)
    {
      exact = std::memcmp(&expected[index], &observed[index],
                          sizeof(float)) == 0 && exact;
      maximum_absolute = std::max(
          maximum_absolute,
          std::fabs(static_cast<double>(expected[index]) - observed[index]));
      const std::uint32_t expected_ordered =
          ordered_float_bits(expected[index]);
      const std::uint32_t observed_ordered =
          ordered_float_bits(observed[index]);
      const std::uint64_t ulp = expected_ordered > observed_ordered
          ? static_cast<std::uint64_t>(expected_ordered - observed_ordered)
          : static_cast<std::uint64_t>(observed_ordered - expected_ordered);
      maximum_ulp = std::max(maximum_ulp, ulp);
    }
  std::cout << label << ": " << (exact ? "PASS" : "FAIL")
            << " bit_identical=" << (exact ? "yes" : "no")
            << " max_abs=" << std::scientific << maximum_absolute
            << " max_ulp=" << maximum_ulp << '\n';
  return exact;
}

template <class T>
std::vector<unsigned char> object_bytes(const T& value)
{
  std::vector<unsigned char> bytes(sizeof(T));
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

float field_particle_coordinate(float absolute_coordinate,
                                float inverse,
                                float offset)
{
  return (absolute_coordinate - offset) / inverse;
}

std::vector<float> field_test_particles(int nx,
                                        int ny,
                                        float physical_min_x,
                                        float physical_max_x,
                                        float physical_min_y,
                                        float physical_max_y,
                                        float inverse_x,
                                        float inverse_y,
                                        float offset_x,
                                        float offset_y,
                                        std::uint32_t seed)
{
  std::vector<float> xy;
  const float x_low = 1.0f;
  const float x_high = static_cast<float>(nx - 1);
  const float y_low = 1.0f;
  const float y_high = static_cast<float>(ny - 1);
  const float x_mid = 0.5f * (x_low + x_high);
  const float y_mid = 0.5f * (y_low + y_high);
  const float physical_x_low =
      field_particle_coordinate(x_low, inverse_x, offset_x);
  const float physical_x_high =
      field_particle_coordinate(x_high, inverse_x, offset_x);
  const float physical_y_low =
      field_particle_coordinate(y_low, inverse_y, offset_y);
  const float physical_y_high =
      field_particle_coordinate(y_high, inverse_y, offset_y);
  const float physical_x_mid =
      field_particle_coordinate(x_mid, inverse_x, offset_x);
  const float physical_y_mid =
      field_particle_coordinate(y_mid, inverse_y, offset_y);
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const float positive_infinity = std::numeric_limits<float>::infinity();
  const float boundary_cases[][2] = {
      {physical_x_low, physical_y_mid},
      {std::nextafter(physical_x_low, negative_infinity), physical_y_mid},
      {std::nextafter(physical_x_low, positive_infinity), physical_y_mid},
      {physical_x_high, physical_y_mid},
      {std::nextafter(physical_x_high, negative_infinity), physical_y_mid},
      {std::nextafter(physical_x_high, positive_infinity), physical_y_mid},
      {physical_x_mid, physical_y_low},
      {physical_x_mid, std::nextafter(physical_y_low, negative_infinity)},
      {physical_x_mid, std::nextafter(physical_y_low, positive_infinity)},
      {physical_x_mid, physical_y_high},
      {physical_x_mid, std::nextafter(physical_y_high, negative_infinity)},
      {physical_x_mid, std::nextafter(physical_y_high, positive_infinity)}};
  for (std::size_t index = 0;
       index < sizeof(boundary_cases) / sizeof(boundary_cases[0]); ++index)
    {
      xy.push_back(boundary_cases[index][0]);
      xy.push_back(boundary_cases[index][1]);
    }
  const float physical_mid_x =
      0.5f * (physical_min_x + physical_max_x);
  const float physical_mid_y =
      0.5f * (physical_min_y + physical_max_y);
  const float physical_range_cases[][2] = {
      {physical_min_x, physical_mid_y},
      {std::nextafter(physical_min_x, negative_infinity), physical_mid_y},
      {std::nextafter(physical_min_x, positive_infinity), physical_mid_y},
      {physical_max_x, physical_mid_y},
      {std::nextafter(physical_max_x, negative_infinity), physical_mid_y},
      {std::nextafter(physical_max_x, positive_infinity), physical_mid_y},
      {physical_mid_x, physical_min_y},
      {physical_mid_x, std::nextafter(physical_min_y, negative_infinity)},
      {physical_mid_x, std::nextafter(physical_min_y, positive_infinity)},
      {physical_mid_x, physical_max_y},
      {physical_mid_x, std::nextafter(physical_max_y, negative_infinity)},
      {physical_mid_x, std::nextafter(physical_max_y, positive_infinity)}};
  for (std::size_t index = 0;
       index < sizeof(physical_range_cases) /
                   sizeof(physical_range_cases[0]); ++index)
    {
      xy.push_back(physical_range_cases[index][0]);
      xy.push_back(physical_range_cases[index][1]);
    }
  xy.push_back(field_particle_coordinate(1.25f, inverse_x, offset_x));
  xy.push_back(field_particle_coordinate(1.75f, inverse_y, offset_y));
  xy.push_back(field_particle_coordinate(static_cast<float>(nx) - 1.25f,
                                         inverse_x, offset_x));
  xy.push_back(field_particle_coordinate(static_cast<float>(ny) - 1.75f,
                                         inverse_y, offset_y));
  std::uint32_t state = seed;
  while (xy.size() / 2 < 60)
    {
      state = 1664525U * state + 1013904223U;
      const float unit_x = static_cast<float>(state & 0xffffU) / 65535.0f;
      state = 1664525U * state + 1013904223U;
      const float unit_y = static_cast<float>(state & 0xffffU) / 65535.0f;
      const float absolute_x = 0.25f +
          unit_x * (static_cast<float>(nx) - 0.5f);
      const float absolute_y = 0.25f +
          unit_y * (static_cast<float>(ny) - 0.5f);
      xy.push_back(field_particle_coordinate(absolute_x, inverse_x,
                                             offset_x));
      xy.push_back(field_particle_coordinate(absolute_y, inverse_y,
                                             offset_y));
    }
  return xy;
}

std::vector<float> deposit_particles(int nx,
                                     int ny,
                                     float inverse_x,
                                     float inverse_y,
                                     float offset_x,
                                     float offset_y,
                                     float shift)
{
  const float positions[][2] = {
      {0.0f + shift, 0.0f},
      {0.25f + shift, 0.75f},
      {1.10f + shift, 2.20f},
      {2.80f + shift, 3.70f},
      {1.35f + shift, 1.45f},
      {1.35f + shift, 1.45f},
      {std::nextafter(0.0f, -std::numeric_limits<float>::infinity()), 1.0f},
      {std::nextafter(0.0f, std::numeric_limits<float>::infinity()), 1.0f},
      {std::nextafter(static_cast<float>(nx - 1),
                      -std::numeric_limits<float>::infinity()), 1.0f},
      {std::nextafter(static_cast<float>(nx - 1),
                      std::numeric_limits<float>::infinity()), 1.0f},
      {1.0f, std::nextafter(0.0f,
                           -std::numeric_limits<float>::infinity())},
      {1.0f, std::nextafter(static_cast<float>(ny - 1),
                           -std::numeric_limits<float>::infinity())},
      {-0.10f, 1.0f},
      {static_cast<float>(nx - 1), 1.0f},
      {1.0f, static_cast<float>(ny - 1)}};
  std::vector<float> xy;
  for (std::size_t index = 0;
       index < sizeof(positions) / sizeof(positions[0]); ++index)
    {
      xy.push_back(particle_coordinate(positions[index][0], inverse_x,
                                       offset_x));
      xy.push_back(particle_coordinate(positions[index][1], inverse_y,
                                       offset_y));
    }
  return xy;
}

bool run_compact_gather_suite(int device, bool pinned_io)
{
  using feebl::gpu::ResidentDepositResult;
  using feebl::gpu::ResidentFieldGatherResult;
  using feebl::gpu::ResidentFieldOptions;
  using feebl::gpu::ResidentFieldSolver;
  using feebl::gpu::ResidentFieldStats;

  const int nx = 8;
  const int ny = 9;
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  const std::size_t padded_cells =
      static_cast<std::size_t>(padded_nx) * padded_ny;
  const float inverse_x = 2.0f;
  const float inverse_y = 4.0f;
  const float offset_x = 4.0f;
  const float offset_y = 4.5f;
  const float physical_min_x = -1.5f;
  const float physical_max_x = 1.5f;
  const float physical_min_y = -0.875f;
  const float physical_max_y = 0.875f;

  std::vector<double> kernel(padded_cells, 0.0);
  kernel[index_2d(0, 0, padded_ny)] = 0.375;
  kernel[index_2d(1, 0, padded_ny)] = -0.0625;
  kernel[index_2d(0, 1, padded_ny)] = 0.1875;
  kernel[index_2d(2, 3, padded_ny)] = -0.03125;
  kernel[index_2d(padded_nx - 1, padded_ny - 2, padded_ny)] = 0.015625;
  const std::vector<double> dist =
      normalized_kernel_spectrum(kernel, padded_nx, padded_ny);

  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  ResidentFieldSolver solver;
  std::string error;
  if (!solver.initialize(options, nx, ny, dist.data(), error) ||
      !solver.reserve_particle_capacity(64, 64, error))
    {
      std::cerr << "compact gather preflight failed: " << error << '\n';
      return false;
    }
  bool passed = true;
  const float legacy_ub_x = std::nextafter(
      physical_max_x, -std::numeric_limits<float>::infinity());
  const float safe_mid_y = 0.0f;
  passed = legacy_ub_x < physical_max_x && passed;
  passed = feebl::gpu::resident_field_normalized_coordinate(
               legacy_ub_x, inverse_x, offset_x) ==
               static_cast<float>(nx - 1) && passed;
  passed = !feebl::gpu::resident_field_point_in_intended_domain(
               legacy_ub_x, safe_mid_y,
               physical_min_x, physical_max_x,
               physical_min_y, physical_max_y,
               inverse_x, inverse_y, offset_x, offset_y, nx, ny) && passed;
  const float safe_lower_x = std::nextafter(
      physical_min_x, std::numeric_limits<float>::infinity());
  const float safe_lower_ax =
      feebl::gpu::resident_field_normalized_coordinate(
          safe_lower_x, inverse_x, offset_x);
  passed = safe_lower_ax >= 1.0f &&
           safe_lower_ax < static_cast<float>(nx - 1) && passed;
  passed = feebl::gpu::resident_field_point_in_intended_domain(
               safe_lower_x, safe_mid_y,
               physical_min_x, physical_max_x,
               physical_min_y, physical_max_y,
               inverse_x, inverse_y, offset_x, offset_y, nx, ny) && passed;
  const ResidentFieldStats initialized = solver.stats();
  passed = initialized.particle_capacity1 == 64 && passed;
  passed = initialized.particle_capacity2 == 64 && passed;
  passed = initialized.device_allocations >= 12 && passed;
  passed = initialized.pinned_host_allocations ==
               static_cast<std::uint64_t>(pinned_io ? 10 : 0) && passed;

  std::vector<float> fields1(2 * 64, 7.0f);
  std::vector<float> fields2(2 * 64, -7.0f);
  ResidentFieldGatherResult rejected_gather;
  rejected_gather.beam1_in = 101;
  rejected_gather.beam1_out = 102;
  rejected_gather.beam2_in = 103;
  rejected_gather.beam2_out = 104;
  ResidentFieldStats rejected_stats = initialized;
  const std::vector<unsigned char> rejected_gather_sentinel =
      object_bytes(rejected_gather);
  const std::vector<unsigned char> rejected_stats_sentinel =
      object_bytes(rejected_stats);
  const std::vector<float> rejected_fields1_sentinel = fields1;
  const std::vector<float> rejected_fields2_sentinel = fields2;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), 64, fields2.data(), 64,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "pre-deposit compact gather unexpectedly succeeded\n";
      return false;
    }
  passed = !error.empty() && passed;
  passed = fields1 == rejected_fields1_sentinel && passed;
  passed = fields2 == rejected_fields2_sentinel && passed;
  passed = object_bytes(rejected_gather) == rejected_gather_sentinel && passed;
  passed = object_bytes(rejected_stats) == rejected_stats_sentinel && passed;
  passed = solver.stats().calls == 0 && passed;

  const std::vector<float> xy1 = field_test_particles(
      nx, ny, physical_min_x, physical_max_x,
      physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, 0x12345678U);
  const std::vector<float> xy2 = field_test_particles(
      nx, ny, physical_min_x, physical_max_x,
      physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, 0x9abcdef0U);
  const double charge1 = 1.25;
  const double charge2 = -0.75;
  const DepositReference deposited1 = deposit_reference(
      xy1, charge1, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  const DepositReference deposited2 = deposit_reference(
      xy2, charge2, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  ResidentDepositResult deposit_result;
  ResidentFieldStats statistics;
  if (!solver.deposit(xy1.data(), xy1.size() / 2, charge1,
                      xy2.data(), xy2.size() / 2, charge2,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      deposit_result, statistics, error))
    {
      std::cerr << "compact gather resident deposit failed: " << error
                << '\n';
      return false;
    }
  passed = deposit_result.beam1_in == deposited1.in && passed;
  passed = deposit_result.beam1_out == deposited1.out && passed;
  passed = deposit_result.beam2_in == deposited2.in && passed;
  passed = deposit_result.beam2_out == deposited2.out && passed;
  const std::uint64_t allocations_before_gather =
      statistics.device_allocations;
  const std::uint64_t pinned_allocations_before_gather =
      statistics.pinned_host_allocations;

  fields1.assign(xy1.size(), std::numeric_limits<float>::quiet_NaN());
  fields2.assign(xy2.size(), std::numeric_limits<float>::quiet_NaN());
  ResidentFieldGatherResult gather_result;
  if (!solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          gather_result, statistics, error))
    {
      std::cerr << "compact opposing-field gather failed: " << error << '\n';
      return false;
    }
  const std::uint64_t compact_bytes =
      static_cast<std::uint64_t>((xy1.size() + xy2.size()) * sizeof(float) +
                                 4 * sizeof(std::uint64_t));
  passed = statistics.last_h2d_bytes == 0 && passed;
  passed = statistics.last_h2d_operations == 0 && passed;
  passed = statistics.last_d2h_bytes == compact_bytes && passed;
  passed = statistics.last_d2h_operations == 3 && passed;
  passed = statistics.last_synchronizations == 1 && passed;
  passed = statistics.last_device_work && passed;
  passed = statistics.field_gather_calls == 1 && passed;
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;

  // Materialize the same resident potentials only for the independent test
  // oracle.  The production compact path above did not transfer them.
  std::vector<double> phi1(cells, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> phi2(cells, std::numeric_limits<double>::quiet_NaN());
  if (!solver.solve_deposited(1.0f, phi1.data(), phi2.data(),
                              statistics, error))
    {
      std::cerr << "compact gather oracle solve failed: " << error << '\n';
      return false;
    }
  passed = close_vectors(direct_convolution(deposited1.rho, kernel, nx, ny),
                         phi1, "compact oracle beam1 potential") && passed;
  passed = close_vectors(direct_convolution(deposited2.rho, kernel, nx, ny),
                         phi2, "compact oracle beam2 potential") && passed;
  const FieldReference expected_fields1 = field_reference(
      xy1, phi2, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  const FieldReference expected_fields2 = field_reference(
      xy2, phi1, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  passed = expected_fields1.shared_domain_agrees && passed;
  passed = expected_fields2.shared_domain_agrees && passed;
  passed = exact_field_vectors(expected_fields1.fields, fields1,
                               "compact beam1 opposing field") && passed;
  passed = exact_field_vectors(expected_fields2.fields, fields2,
                               "compact beam2 opposing field") && passed;
  passed = gather_result.beam1_in == expected_fields1.in && passed;
  passed = gather_result.beam1_out == expected_fields1.out && passed;
  passed = gather_result.beam2_in == expected_fields2.in && passed;
  passed = gather_result.beam2_out == expected_fields2.out && passed;
  passed = gather_result.beam1_in + gather_result.beam1_out ==
               xy1.size() / 2 && passed;
  passed = gather_result.beam2_in + gather_result.beam2_out ==
               xy2.size() / 2 && passed;

  // A negative nonzero sign is the same historical nonzero gate.  Exercise
  // an exactly adjacent pair of output ranges and a repeated gather from the
  // unchanged resident rho/xy state.
  std::vector<float> adjacent_fields(xy1.size() + xy2.size(), 19.0f);
  ResidentFieldGatherResult adjacent_result;
  if (!solver.solve_deposited_and_gather(
          -1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          adjacent_fields.data(), xy1.size() / 2,
          adjacent_fields.data() + xy1.size(), xy2.size() / 2,
          adjacent_result, statistics, error))
    {
      std::cerr << "adjacent negative-sign compact gather failed: " << error
                << '\n';
      return false;
    }
  const std::vector<float> adjacent1(
      adjacent_fields.begin(), adjacent_fields.begin() + xy1.size());
  const std::vector<float> adjacent2(
      adjacent_fields.begin() + xy1.size(), adjacent_fields.end());
  passed = exact_field_vectors(expected_fields1.fields, adjacent1,
                               "adjacent negative-sign beam1 field") && passed;
  passed = exact_field_vectors(expected_fields2.fields, adjacent2,
                               "adjacent negative-sign beam2 field") && passed;
  passed = adjacent_result.beam1_in == expected_fields1.in && passed;
  passed = adjacent_result.beam2_in == expected_fields2.in && passed;

  // Capacities may exceed the resident particle counts, but the gather must
  // write exactly two floats per resident particle and leave tail canaries.
  const float canary1 = 123456.25f;
  const float canary2 = -654321.5f;
  std::vector<float> excess1(xy1.size() + 6, canary1);
  std::vector<float> excess2(xy2.size() + 10, canary2);
  ResidentFieldGatherResult excess_result;
  if (!solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          excess1.data(), xy1.size() / 2 + 3,
          excess2.data(), xy2.size() / 2 + 5,
          excess_result, statistics, error))
    {
      std::cerr << "excess-capacity compact gather failed: " << error << '\n';
      return false;
    }
  const std::vector<float> excess_observed1(
      excess1.begin(), excess1.begin() + xy1.size());
  const std::vector<float> excess_observed2(
      excess2.begin(), excess2.begin() + xy2.size());
  passed = exact_field_vectors(expected_fields1.fields, excess_observed1,
                               "excess-capacity beam1 field") && passed;
  passed = exact_field_vectors(expected_fields2.fields, excess_observed2,
                               "excess-capacity beam2 field") && passed;
  passed = std::find_if(excess1.begin() + xy1.size(), excess1.end(),
                        [canary1](float value) { return value != canary1; }) ==
               excess1.end() && passed;
  passed = std::find_if(excess2.begin() + xy2.size(), excess2.end(),
                        [canary2](float value) { return value != canary2; }) ==
               excess2.end() && passed;
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;
  passed = statistics.field_gather_calls == 3 && passed;

  const std::uint64_t calls_before_rejection = statistics.calls;
  rejected_stats = statistics;
  const std::vector<unsigned char> post_result_sentinel =
      object_bytes(rejected_gather);
  const std::vector<unsigned char> post_stats_sentinel =
      object_bytes(rejected_stats);
  const std::vector<float> post_fields1_sentinel = fields1;
  const std::vector<float> post_fields2_sentinel = fields2;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2 - 1,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "undersized compact output unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          0, xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "null compact output unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  std::vector<float> overlapping(xy1.size() + xy2.size(), 0.0f);
  const std::vector<float> overlapping_sentinel = overlapping;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          overlapping.data(), xy1.size() / 2,
          overlapping.data() + 1, xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "overlapping compact outputs unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_max_x, physical_min_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "reversed compact physical bounds unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_min_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "equal compact x bounds unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_max_y, physical_min_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "reversed compact y bounds unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, std::numeric_limits<float>::quiet_NaN(), physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "nonfinite compact physical bounds unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, std::numeric_limits<float>::infinity(),
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "infinite compact physical bounds unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  passed = fields1 == post_fields1_sentinel && passed;
  passed = fields2 == post_fields2_sentinel && passed;
  passed = overlapping == overlapping_sentinel && passed;
  passed = object_bytes(rejected_gather) == post_result_sentinel && passed;
  passed = object_bytes(rejected_stats) == post_stats_sentinel && passed;
  passed = solver.stats().calls == calls_before_rejection && passed;

  fields1.assign(xy1.size(), std::numeric_limits<float>::quiet_NaN());
  fields2.assign(xy2.size(), std::numeric_limits<float>::quiet_NaN());
  ResidentFieldGatherResult zero_result;
  if (!solver.solve_deposited_and_gather(
          0.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          zero_result, statistics, error))
    {
      std::cerr << "zero-gated compact gather failed: " << error << '\n';
      return false;
    }
  const FieldReference zero_fields1 = field_reference(
      xy1, phi2, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, true);
  const FieldReference zero_fields2 = field_reference(
      xy2, phi1, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, true);
  passed = exact_field_vectors(zero_fields1.fields, fields1,
                               "zero-gated compact beam1 field") && passed;
  passed = exact_field_vectors(zero_fields2.fields, fields2,
                               "zero-gated compact beam2 field") && passed;
  passed = zero_result.beam1_in == expected_fields1.in && passed;
  passed = zero_result.beam1_out == expected_fields1.out && passed;
  passed = zero_result.beam2_in == expected_fields2.in && passed;
  passed = zero_result.beam2_out == expected_fields2.out && passed;
  passed = statistics.last_d2h_bytes == compact_bytes && passed;
  passed = statistics.last_d2h_operations == 3 && passed;
  passed = statistics.last_h2d_bytes == 0 && passed;
  passed = statistics.field_gather_calls == 4 && passed;
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;

  ResidentDepositResult zero_charge_deposit;
  if (!solver.deposit(xy1.data(), xy1.size() / 2, 0.0,
                      xy2.data(), xy2.size() / 2, 0.0,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      zero_charge_deposit, statistics, error))
    {
      std::cerr << "zero-charge compact resident deposit failed: " << error
                << '\n';
      return false;
    }
  fields1.assign(xy1.size(), std::numeric_limits<float>::quiet_NaN());
  fields2.assign(xy2.size(), std::numeric_limits<float>::quiet_NaN());
  ResidentFieldGatherResult zero_charge_result;
  if (!solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          zero_charge_result, statistics, error))
    {
      std::cerr << "zero-charge compact gather failed: " << error << '\n';
      return false;
    }
  passed = std::find_if(fields1.begin(), fields1.end(),
                        [](float value) { return value != 0.0f; }) ==
               fields1.end() && passed;
  passed = std::find_if(fields2.begin(), fields2.end(),
                        [](float value) { return value != 0.0f; }) ==
               fields2.end() && passed;
  passed = zero_charge_result.beam1_in == expected_fields1.in && passed;
  passed = zero_charge_result.beam1_out == expected_fields1.out && passed;
  passed = zero_charge_result.beam2_in == expected_fields2.in && passed;
  passed = zero_charge_result.beam2_out == expected_fields2.out && passed;
  passed = statistics.last_d2h_bytes == compact_bytes && passed;
  passed = statistics.field_gather_calls == 5 && passed;
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;

  ResidentDepositResult empty_deposit;
  if (!solver.deposit(0, 0, 0.0, 0, 0, 0.0,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      empty_deposit, statistics, error))
    {
      std::cerr << "empty compact resident deposit failed: " << error << '\n';
      return false;
    }
  ResidentFieldGatherResult empty_result;
  if (!solver.solve_deposited_and_gather(
          0.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          0, 0, 0, 0, empty_result, statistics, error))
    {
      std::cerr << "empty compact gather failed: " << error << '\n';
      return false;
    }
  passed = empty_result.beam1_in == 0 && empty_result.beam1_out == 0 && passed;
  passed = empty_result.beam2_in == 0 && empty_result.beam2_out == 0 && passed;
  passed = statistics.last_d2h_bytes == 4 * sizeof(std::uint64_t) && passed;
  passed = statistics.last_d2h_operations == 1 && passed;
  passed = statistics.last_synchronizations == 1 && passed;
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;

  // The combined legacy-compatible entry point must retain the same particle
  // state as split deposit(), so compact gather is legal immediately after it.
  ResidentDepositResult combined_result;
  if (!solver.deposit_and_solve(
          xy1.data(), xy1.size() / 2, charge1,
          xy2.data(), xy2.size() / 2, charge2,
          inverse_x, inverse_y, offset_x, offset_y, false, 1.0f,
          phi1.data(), phi2.data(), combined_result, statistics, error))
    {
      std::cerr << "combined state-transition solve failed: " << error << '\n';
      return false;
    }
  const FieldReference combined_expected1 = field_reference(
      xy1, phi2, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  const FieldReference combined_expected2 = field_reference(
      xy2, phi1, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  fields1.assign(xy1.size(), 0.0f);
  fields2.assign(xy2.size(), 0.0f);
  ResidentFieldGatherResult combined_gather;
  if (!solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          combined_gather, statistics, error))
    {
      std::cerr << "gather after deposit_and_solve failed: " << error << '\n';
      return false;
    }
  passed = exact_field_vectors(combined_expected1.fields, fields1,
                               "combined-state beam1 field") && passed;
  passed = exact_field_vectors(combined_expected2.fields, fields2,
                               "combined-state beam2 field") && passed;

  // A host-rho solve deliberately invalidates resident particle/deposit
  // state.  Its subsequent gather rejection must be side-effect free.
  std::vector<double> host_rho1 = first_rho(nx, ny, 0.4);
  std::vector<double> host_rho2 = first_rho(nx, ny, -0.6);
  if (!solver.solve(host_rho1.data(), host_rho2.data(), 1.0f,
                    phi1.data(), phi2.data(), statistics, error))
    {
      std::cerr << "state-invalidation host solve failed: " << error << '\n';
      return false;
    }
  fields1.assign(xy1.size(), 31.0f);
  fields2.assign(xy2.size(), -31.0f);
  const std::vector<float> invalidated_fields1 = fields1;
  const std::vector<float> invalidated_fields2 = fields2;
  rejected_stats = statistics;
  const std::vector<unsigned char> invalidated_stats =
      object_bytes(rejected_stats);
  const std::vector<unsigned char> invalidated_result =
      object_bytes(rejected_gather);
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "gather after host solve unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && fields1 == invalidated_fields1 &&
           fields2 == invalidated_fields2 && passed;
  passed = object_bytes(rejected_stats) == invalidated_stats && passed;
  passed = object_bytes(rejected_gather) == invalidated_result && passed;

  // A failed deposit invalidates any preceding valid deposit so callers
  // cannot gather stale rho/positions.
  if (!solver.deposit(xy1.data(), xy1.size() / 2, charge1,
                      xy2.data(), xy2.size() / 2, charge2,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      deposit_result, statistics, error))
    {
      std::cerr << "valid deposit before failed-state test failed: " << error
                << '\n';
      return false;
    }
  std::vector<float> over_capacity(2 * 65, 0.0f);
  ResidentDepositResult failed_deposit_result;
  if (solver.deposit(over_capacity.data(), 65, charge1,
                     xy2.data(), xy2.size() / 2, charge2,
                     inverse_x, inverse_y, offset_x, offset_y, false,
                     failed_deposit_result, statistics, error))
    {
      std::cerr << "over-capacity state-transition deposit unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  rejected_stats = solver.stats();
  const std::vector<unsigned char> failed_state_stats =
      object_bytes(rejected_stats);
  if (solver.solve_deposited_and_gather(
          1.0f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          rejected_gather, rejected_stats, error))
    {
      std::cerr << "gather after failed deposit unexpectedly succeeded\n";
      passed = false;
    }
  passed = !error.empty() && passed;
  passed = object_bytes(rejected_stats) == failed_state_stats && passed;

  // Differentially exercise every exact/nextafter normalized and physical
  // boundary case one particle at a time, so aggregate counts cannot mask a
  // per-boundary GPU predicate error.  charge_sign==0 still launches the
  // device predicate/zero-output kernel without an FFT.
  const float safe_opponent_xy[] = {
      field_particle_coordinate(4.0f, inverse_x, offset_x),
      field_particle_coordinate(4.5f, inverse_y, offset_y)};
  for (std::size_t boundary = 0; boundary < 24; ++boundary)
    {
      const float one_xy[] = {xy1[2 * boundary], xy1[2 * boundary + 1]};
      ResidentDepositResult boundary_deposit;
      if (!solver.deposit(one_xy, 1, charge1,
                          safe_opponent_xy, 1, charge2,
                          inverse_x, inverse_y, offset_x, offset_y, false,
                          boundary_deposit, statistics, error))
        {
          std::cerr << "boundary predicate deposit failed at " << boundary
                    << ": " << error << '\n';
          return false;
        }
      float boundary_field1[2] = {77.0f, 78.0f};
      float boundary_field2[2] = {79.0f, 80.0f};
      ResidentFieldGatherResult boundary_result;
      if (!solver.solve_deposited_and_gather(
              0.0f, physical_min_x, physical_max_x,
              physical_min_y, physical_max_y,
              boundary_field1, 1, boundary_field2, 1,
              boundary_result, statistics, error))
        {
          std::cerr << "boundary predicate gather failed at " << boundary
                    << ": " << error << '\n';
          return false;
        }
      const bool expected_boundary = independent_stencil_safe_domain(
          one_xy[0], one_xy[1],
          physical_min_x, physical_max_x, physical_min_y, physical_max_y,
          inverse_x, inverse_y, offset_x, offset_y, nx, ny);
      const bool expected_opponent = independent_stencil_safe_domain(
          safe_opponent_xy[0], safe_opponent_xy[1],
          physical_min_x, physical_max_x, physical_min_y, physical_max_y,
          inverse_x, inverse_y, offset_x, offset_y, nx, ny);
      passed = boundary_result.beam1_in ==
                   static_cast<std::size_t>(expected_boundary) && passed;
      passed = boundary_result.beam1_out ==
                   static_cast<std::size_t>(!expected_boundary) && passed;
      passed = boundary_result.beam2_in ==
                   static_cast<std::size_t>(expected_opponent) && passed;
      passed = boundary_result.beam2_out ==
                   static_cast<std::size_t>(!expected_opponent) && passed;
      passed = boundary_field1[0] == 0.0f &&
               boundary_field1[1] == 0.0f &&
               boundary_field2[0] == 0.0f &&
               boundary_field2[1] == 0.0f && passed;
    }
  passed = statistics.device_allocations == allocations_before_gather && passed;
  passed = statistics.pinned_host_allocations ==
               pinned_allocations_before_gather && passed;

  const std::uint64_t full_phi_bytes =
      static_cast<std::uint64_t>(2 * cells * sizeof(double));
  std::cout << (pinned_io ? "pinned" : "pageable")
            << " compact field gather: boundary/random beam counts="
            << gather_result.beam1_in << '/' << gather_result.beam1_out
            << " and " << gather_result.beam2_in << '/'
            << gather_result.beam2_out
            << " D2H=" << compact_bytes
            << " bytes vs full-phi=" << full_phi_bytes << " bytes\n";
  if (!passed)
    std::cerr << (pinned_io ? "pinned" : "pageable")
              << " compact field gather invariant failed\n";
  return passed;
}

bool run_non_dyadic_gather_case(int device, bool pinned_io, int scenario)
{
  using feebl::gpu::ResidentDepositResult;
  using feebl::gpu::ResidentFieldGatherResult;
  using feebl::gpu::ResidentFieldOptions;
  using feebl::gpu::ResidentFieldSolver;
  using feebl::gpu::ResidentFieldStats;

  const int nx = scenario == 0 ? 7 : 11;
  const int ny = scenario == 0 ? 10 : 6;
  const float inverse_x = scenario == 0 ? 1.3f : 0.37f;
  const float inverse_y = scenario == 0 ? 0.7f : 1.7f;
  const float offset_x = scenario == 0 ? 3.1f : 5.3f;
  const float offset_y = scenario == 0 ? 4.2f : 2.4f;
  const float physical_min_x = scenario == 0 ? -1.4f : -10.0f;
  const float physical_max_x = scenario == 0 ? 2.0f : 12.0f;
  const float physical_min_y = scenario == 0 ? -4.2f : -0.7f;
  const float physical_max_y = scenario == 0 ? 6.3f : 1.45f;
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  const std::size_t padded_cells =
      static_cast<std::size_t>(padded_nx) * padded_ny;
  std::vector<double> kernel(padded_cells, 0.0);
  kernel[index_2d(0, 0, padded_ny)] = 0.371 + 0.013 * scenario;
  kernel[index_2d(1, 2, padded_ny)] = -0.119 + 0.007 * scenario;
  kernel[index_2d(3, 1, padded_ny)] = 0.083 - 0.011 * scenario;
  kernel[index_2d(padded_nx - 2, padded_ny - 1, padded_ny)] = -0.027;
  const std::vector<double> dist =
      normalized_kernel_spectrum(kernel, padded_nx, padded_ny);

  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  ResidentFieldSolver solver;
  std::string error;
  if (!solver.initialize(options, nx, ny, dist.data(), error) ||
      !solver.reserve_particle_capacity(64, 64, error))
    {
      std::cerr << "non-dyadic gather preflight failed: " << error << '\n';
      return false;
    }
  const std::vector<float> xy1 = field_test_particles(
      nx, ny, physical_min_x, physical_max_x,
      physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y,
      scenario == 0 ? 0x0badc0deU : 0x31415926U);
  const std::vector<float> xy2 = field_test_particles(
      nx, ny, physical_min_x, physical_max_x,
      physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y,
      scenario == 0 ? 0xfeedfaceU : 0x27182818U);
  const double charge1 = scenario == 0 ? -1.17 : 0.93;
  const double charge2 = scenario == 0 ? 0.81 : -1.29;
  const DepositReference deposited1 = deposit_reference(
      xy1, charge1, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  const DepositReference deposited2 = deposit_reference(
      xy2, charge2, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  ResidentDepositResult deposit_result;
  ResidentFieldStats statistics;
  if (!solver.deposit(xy1.data(), xy1.size() / 2, charge1,
                      xy2.data(), xy2.size() / 2, charge2,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      deposit_result, statistics, error))
    {
      std::cerr << "non-dyadic deposit failed: " << error << '\n';
      return false;
    }
  const std::uint64_t allocations_before = statistics.device_allocations;
  const std::uint64_t pinned_allocations_before =
      statistics.pinned_host_allocations;
  std::vector<float> fields1(xy1.size(), 0.0f);
  std::vector<float> fields2(xy2.size(), 0.0f);
  ResidentFieldGatherResult gather_result;
  if (!solver.solve_deposited_and_gather(
          -0.25f, physical_min_x, physical_max_x,
          physical_min_y, physical_max_y,
          fields1.data(), xy1.size() / 2,
          fields2.data(), xy2.size() / 2,
          gather_result, statistics, error))
    {
      std::cerr << "non-dyadic compact gather failed: " << error << '\n';
      return false;
    }
  std::vector<double> phi1(cells, 0.0);
  std::vector<double> phi2(cells, 0.0);
  if (!solver.solve_deposited(1.0f, phi1.data(), phi2.data(),
                              statistics, error))
    {
      std::cerr << "non-dyadic oracle solve failed: " << error << '\n';
      return false;
    }
  const FieldReference expected1 = field_reference(
      xy1, phi2, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  const FieldReference expected2 = field_reference(
      xy2, phi1, nx, ny,
      physical_min_x, physical_max_x, physical_min_y, physical_max_y,
      inverse_x, inverse_y, offset_x, offset_y, false);
  bool passed = expected1.shared_domain_agrees &&
                expected2.shared_domain_agrees;
  passed = exact_field_vectors(expected1.fields, fields1,
                               "non-dyadic beam1 field") && passed;
  passed = exact_field_vectors(expected2.fields, fields2,
                               "non-dyadic beam2 field") && passed;
  passed = gather_result.beam1_in == expected1.in &&
           gather_result.beam1_out == expected1.out && passed;
  passed = gather_result.beam2_in == expected2.in &&
           gather_result.beam2_out == expected2.out && passed;
  passed = close_vectors(direct_convolution(deposited1.rho, kernel, nx, ny),
                         phi1, "non-dyadic beam1 potential") && passed;
  passed = close_vectors(direct_convolution(deposited2.rho, kernel, nx, ny),
                         phi2, "non-dyadic beam2 potential") && passed;
  passed = statistics.device_allocations == allocations_before &&
           statistics.pinned_host_allocations ==
               pinned_allocations_before && passed;
  std::cout << (pinned_io ? "pinned" : "pageable")
            << " non-dyadic scenario " << scenario
            << ": grid=" << nx << 'x' << ny
            << " in/out=" << gather_result.beam1_in << '/'
            << gather_result.beam1_out << " and "
            << gather_result.beam2_in << '/' << gather_result.beam2_out
            << '\n';
  return passed;
}

#if defined(FEEBL_GPU_RESIDENT_TEST_HOOKS)
bool run_injected_failure_case(int device, bool pinned_io)
{
  using feebl::gpu::ResidentDepositResult;
  using feebl::gpu::ResidentFieldGatherResult;
  using feebl::gpu::ResidentFieldOptions;
  using feebl::gpu::ResidentFieldSolver;
  using feebl::gpu::ResidentFieldStats;

  const int nx = 4;
  const int ny = 5;
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  std::vector<double> kernel(
      static_cast<std::size_t>(padded_nx) * padded_ny, 0.0);
  kernel[0] = 0.5;
  const std::vector<double> dist =
      normalized_kernel_spectrum(kernel, padded_nx, padded_ny);
  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  ResidentFieldSolver solver;
  std::string error;
  if (!solver.initialize(options, nx, ny, dist.data(), error) ||
      !solver.reserve_particle_capacity(2, 2, error))
    {
      std::cerr << "poison test preflight failed: " << error << '\n';
      return false;
    }
  const float inverse_x = 1.0f;
  const float inverse_y = 1.0f;
  const float offset_x = 2.0f;
  const float offset_y = 2.5f;
  const float xy1[] = {0.0f, 0.0f};
  const float xy2[] = {0.25f, -0.25f};
  ResidentDepositResult deposit_result;
  ResidentFieldStats statistics;
  if (!solver.deposit(xy1, 1, 1.0, xy2, 1, -1.0,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      deposit_result, statistics, error))
    {
      std::cerr << "poison test deposit failed: " << error << '\n';
      return false;
    }
  float fields1[] = {41.0f, 42.0f};
  float fields2[] = {43.0f, 44.0f};
  ResidentFieldGatherResult result;
  result.beam1_in = 51;
  result.beam1_out = 52;
  result.beam2_in = 53;
  result.beam2_out = 54;
  const std::vector<unsigned char> result_sentinel = object_bytes(result);
  if (solver.inject_compact_gather_failure_for_test(statistics, error))
    {
      std::cerr << "deterministic compact-gather failure injection "
                   "unexpectedly succeeded\n";
      return false;
    }
  bool passed = !error.empty();
  passed = error.find("injected compact-gather") != std::string::npos && passed;
  passed = error.find("poisoned") != std::string::npos && passed;
  passed = error.find("fallback") != std::string::npos && passed;
  passed = !solver.is_available() && passed;
  const ResidentFieldStats poisoned = solver.stats();
  passed = object_bytes(statistics) == object_bytes(poisoned) && passed;
  passed = poisoned.context_poisoned && passed;
  passed = poisoned.failed_device_calls == 1 && passed;
  passed = poisoned.failure_drains == 1 && passed;
  passed = poisoned.failure_drain_failures == 0 && passed;
  passed = object_bytes(result) == result_sentinel && passed;
  passed = fields1[0] == 41.0f && fields1[1] == 42.0f &&
           fields2[0] == 43.0f && fields2[1] == 44.0f && passed;

  const float retry_fields1[] = {fields1[0], fields1[1]};
  const float retry_fields2[] = {fields2[0], fields2[1]};
  ResidentFieldStats retry_statistics = statistics;
  const std::vector<unsigned char> retry_stats_sentinel =
      object_bytes(retry_statistics);
  const std::vector<unsigned char> poisoned_stats_sentinel =
      object_bytes(poisoned);
  if (solver.solve_deposited_and_gather(
          1.0f, -1.0f, 1.0f, -1.5f, 1.5f,
          fields1, 1, fields2, 1, result, retry_statistics, error))
    {
      std::cerr << "poisoned-context retry unexpectedly succeeded\n";
      return false;
    }
  passed = !error.empty() &&
           error.find("retry") != std::string::npos && passed;
  passed = fields1[0] == retry_fields1[0] &&
           fields1[1] == retry_fields1[1] &&
           fields2[0] == retry_fields2[0] &&
           fields2[1] == retry_fields2[1] && passed;
  passed = object_bytes(result) == result_sentinel && passed;
  passed = object_bytes(retry_statistics) == retry_stats_sentinel && passed;
  passed = object_bytes(solver.stats()) == poisoned_stats_sentinel && passed;
  passed = solver.stats().failed_device_calls == 1 &&
           solver.stats().failure_drains == 1 && passed;
  std::cout << (pinned_io ? "pinned" : "pageable")
            << " deterministic queued-work failure: "
            << (passed ? "PASS" : "FAIL")
            << " drains=" << poisoned.failure_drains
            << " drain_failures=" << poisoned.failure_drain_failures << '\n';
  return passed;
}
#endif

bool run_suite(int device, bool pinned_io)
{
  using feebl::gpu::ResidentFieldOptions;
  using feebl::gpu::ResidentFieldSolver;
  using feebl::gpu::ResidentFieldStats;

  const int nx = 4;
  const int ny = 5;
  const int padded_nx = 2 * nx;
  const int padded_ny = 2 * ny;
  const std::size_t cells = static_cast<std::size_t>(nx) * ny;
  const std::size_t padded_cells =
      static_cast<std::size_t>(padded_nx) * padded_ny;
  const std::uint64_t one_direction_bytes =
      static_cast<std::uint64_t>(2 * cells * sizeof(double));

  std::vector<double> kernel(padded_cells, 0.0);
  kernel[index_2d(0, 0, padded_ny)] = 0.5;
  kernel[index_2d(1, 0, padded_ny)] = -0.125;
  kernel[index_2d(0, 2, padded_ny)] = 0.25;
  kernel[index_2d(padded_nx - 1, padded_ny - 1, padded_ny)] = 0.0625;
  const std::vector<double> dist =
      normalized_kernel_spectrum(kernel, padded_nx, padded_ny);

  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  ResidentFieldSolver solver;
  std::string error;
  if (!solver.initialize(options, nx, ny, dist.data(), error))
    {
      std::cerr << "GPU2 resident FFT initialization failed: " << error
                << '\n';
      return false;
    }
  if (!solver.reserve_particle_capacity(16, 16, error))
    {
      std::cerr << "GPU2 particle-capacity preflight failed: " << error
                << '\n';
      return false;
    }

  const ResidentFieldStats initialized = solver.stats();
  bool passed = true;
  passed = solver.is_available() && passed;
  passed = initialized.calls == 0 && passed;
  passed = initialized.device_work_calls == 0 && passed;
  passed = !initialized.last_device_work && passed;
  passed = initialized.plan_creations == 1 && passed;
  passed = initialized.device_allocations >= 9 && passed;
  passed = initialized.pinned_host_allocations ==
               static_cast<std::uint64_t>(pinned_io ? 10 : 0) && passed;
  passed = initialized.setup_h2d_operations == 1 && passed;
  passed = initialized.setup_synchronizations == 1 && passed;
  passed = initialized.setup_h2d_bytes ==
               2 * padded_cells * sizeof(double) && passed;
  passed = initialized.cuda_runtime_version > 0 && passed;
  passed = initialized.cuda_driver_version > 0 && passed;
  passed = initialized.cufft_runtime_version > 0 && passed;
  passed = initialized.device_total_bytes > 0 && passed;
  passed = initialized.device_free_bytes_at_init > 0 && passed;
  passed = initialized.field_required_bytes > 0 && passed;
  passed = initialized.resident_bytes >= initialized.field_required_bytes
               && passed;
  passed = initialized.particle_capacity1 == 16 && passed;
  passed = initialized.particle_capacity2 == 16 && passed;
  if (!passed)
    {
      std::cerr << "GPU2 initialization provenance/accounting invariant failed\n";
      return false;
    }
  std::cout << "device: " << solver.device_description() << '\n'
            << "runtime: " << solver.runtime_description() << '\n';

  std::vector<double> rho1 = first_rho(nx, ny, 1.25);
  std::vector<double> rho2 = first_rho(nx, ny, -0.8);
  std::vector<double> phi1(cells, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> phi2(cells, std::numeric_limits<double>::quiet_NaN());
  ResidentFieldStats statistics;
  if (!solver.solve(rho1.data(), rho2.data(), 1.0f,
                    phi1.data(), phi2.data(), statistics, error))
    {
      std::cerr << "first GPU2 resident FFT solve failed: " << error << '\n';
      return false;
    }
  passed = close_vectors(direct_convolution(rho1, kernel, nx, ny), phi1,
                         pinned_io ? "pinned beam1 convolution"
                                   : "pageable beam1 convolution") && passed;
  passed = close_vectors(direct_convolution(rho2, kernel, nx, ny), phi2,
                         pinned_io ? "pinned beam2 convolution"
                                   : "pageable beam2 convolution") && passed;
  passed = statistics.calls == 1 && passed;
  passed = statistics.device_work_calls == 1 && passed;
  passed = statistics.last_device_work && passed;
  passed = statistics.last_h2d_bytes == one_direction_bytes && passed;
  passed = statistics.last_d2h_bytes == one_direction_bytes && passed;
  passed = statistics.last_h2d_operations == 2 && passed;
  passed = statistics.last_d2h_operations == 2 && passed;
  passed = statistics.last_synchronizations == 1 && passed;
  passed = statistics.plan_creations == 1 && passed;

  rho1 = first_rho(nx, ny, 3.0);
  rho2 = first_rho(nx, ny, -2.5);
  if (!solver.solve(rho1.data(), rho2.data(), -1.0f,
                    phi1.data(), phi2.data(), statistics, error))
    {
      std::cerr << "negative-sign GPU2 resident FFT solve failed: "
                << error << '\n';
      return false;
    }
  const std::vector<double> expected1 =
      direct_convolution(rho1, kernel, nx, ny);
  const std::vector<double> expected2 =
      direct_convolution(rho2, kernel, nx, ny);
  passed = close_vectors(expected1, phi1, "negative-sign gate beam1") && passed;
  passed = close_vectors(expected2, phi2, "negative-sign gate beam2") && passed;
  const std::vector<double> repeat1 = phi1;
  const std::vector<double> repeat2 = phi2;

  if (!solver.solve(rho1.data(), rho2.data(), -1.0f,
                    phi1.data(), phi2.data(), statistics, error))
    {
      std::cerr << "reused-plan GPU2 resident FFT solve failed: "
                << error << '\n';
      return false;
    }
  passed = close_vectors(repeat1, phi1, "reused plan beam1") && passed;
  passed = close_vectors(repeat2, phi2, "reused plan beam2") && passed;
  passed = std::memcmp(repeat1.data(), phi1.data(), cells * sizeof(double)) == 0
               && passed;
  passed = std::memcmp(repeat2.data(), phi2.data(), cells * sizeof(double)) == 0
               && passed;
  passed = statistics.calls == 3 && passed;
  passed = statistics.device_work_calls == 3 && passed;
  passed = statistics.plan_creations == 1 && passed;
  passed = statistics.h2d_bytes == 3 * one_direction_bytes && passed;
  passed = statistics.d2h_bytes == 3 * one_direction_bytes && passed;
  passed = statistics.h2d_operations == 6 && passed;
  passed = statistics.d2h_operations == 6 && passed;
  passed = statistics.synchronizations == 3 && passed;

  const std::uint64_t h2d_before_zero = statistics.h2d_bytes;
  const std::uint64_t d2h_before_zero = statistics.d2h_bytes;
  const std::uint64_t sync_before_zero = statistics.synchronizations;
  const std::uint64_t work_before_zero = statistics.device_work_calls;
  std::fill(phi1.begin(), phi1.end(), 123.0);
  std::fill(phi2.begin(), phi2.end(), -456.0);
  if (!solver.solve(rho1.data(), rho2.data(), 0.0f,
                    phi1.data(), phi2.data(), statistics, error))
    {
      std::cerr << "zero-gate GPU2 resident FFT solve failed: "
                << error << '\n';
      return false;
    }
  passed = std::find_if(phi1.begin(), phi1.end(),
                        [](double value) { return value != 0.0; }) == phi1.end()
               && passed;
  passed = std::find_if(phi2.begin(), phi2.end(),
                        [](double value) { return value != 0.0; }) == phi2.end()
               && passed;
  passed = statistics.calls == 4 && passed;
  passed = statistics.last_h2d_bytes == 0 && passed;
  passed = statistics.last_d2h_bytes == 0 && passed;
  passed = statistics.last_synchronizations == 0 && passed;
  passed = !statistics.last_device_work && passed;
  passed = statistics.device_work_calls == work_before_zero && passed;
  passed = statistics.h2d_bytes == h2d_before_zero && passed;
  passed = statistics.d2h_bytes == d2h_before_zero && passed;
  passed = statistics.synchronizations == sync_before_zero && passed;
  passed = statistics.plan_creations == 1 && passed;

  const float inverse_x = 1.25f;
  const float inverse_y = 0.75f;
  const float offset_x = 2.5f;
  const float offset_y = 3.0f;
  const std::vector<float> xy1 = deposit_particles(
      nx, ny, inverse_x, inverse_y, offset_x, offset_y, 0.0f);
  const std::vector<float> xy2 = deposit_particles(
      nx, ny, inverse_x, inverse_y, offset_x, offset_y, 0.08f);
  const double charge1 = 3.25;
  const double charge2 = 1.75;
  const DepositReference unsym1 = deposit_reference(
      xy1, charge1, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  const DepositReference unsym2 = deposit_reference(
      xy2, charge2, nx, ny, inverse_x, inverse_y, offset_x, offset_y);
  std::uint64_t allocations_after_first_deposit = 0;
  std::uint64_t pinned_allocations_after_first_deposit = 0;
  for (int symmetry = 0; symmetry < 2; ++symmetry)
    {
      DepositReference reference1 = unsym1;
      DepositReference reference2 = unsym2;
      if (symmetry)
        {
          symmetrize_reference(reference1.rho, nx, ny);
          symmetrize_reference(reference2.rho, nx, ny);
        }
      feebl::gpu::ResidentDepositResult deposit_result;
      if (!solver.deposit_and_solve(
              xy1.data(), xy1.size() / 2, charge1,
              xy2.data(), xy2.size() / 2, charge2,
              inverse_x, inverse_y, offset_x, offset_y,
              symmetry != 0, 1.0f, phi1.data(), phi2.data(),
              deposit_result, statistics, error))
        {
          std::cerr << "resident CIC/field solve failed: " << error << '\n';
          return false;
        }
      passed = deposit_result.beam1_in == reference1.in && passed;
      passed = deposit_result.beam1_out == reference1.out && passed;
      passed = deposit_result.beam2_in == reference2.in && passed;
      passed = deposit_result.beam2_out == reference2.out && passed;
      const double expected_dot =
          fine_lumi_reference(reference1.rho, reference2.rho);
      const double dot_error =
          std::fabs(deposit_result.fine_lumi_dot - expected_dot);
      passed = dot_error <= 2.0e-11 * std::max(1.0, std::fabs(expected_dot))
                   && passed;
      passed = close_vectors(
          direct_convolution(reference1.rho, kernel, nx, ny), phi1,
          symmetry ? "resident symmetric CIC+FFT beam1"
                   : "resident CIC+FFT beam1") && passed;
      passed = close_vectors(
          direct_convolution(reference2.rho, kernel, nx, ny), phi2,
          symmetry ? "resident symmetric CIC+FFT beam2"
                   : "resident CIC+FFT beam2") && passed;
      passed = statistics.last_h2d_bytes ==
                   (xy1.size() + xy2.size()) * sizeof(float) && passed;
      passed = statistics.last_h2d_operations == 2 && passed;
      passed = statistics.last_d2h_operations == 3 && passed;
      passed = statistics.last_synchronizations == 1 && passed;
      passed = statistics.plan_creations == 1 && passed;
      if (!symmetry)
        {
          allocations_after_first_deposit = statistics.device_allocations;
          pinned_allocations_after_first_deposit =
              statistics.pinned_host_allocations;
        }
      else
        {
          passed = statistics.device_allocations ==
                       allocations_after_first_deposit && passed;
          passed = statistics.pinned_host_allocations ==
                       pinned_allocations_after_first_deposit && passed;
        }
      std::cout << (symmetry ? "symmetric" : "ordinary")
                << " resident CIC: beam1=" << deposit_result.beam1_in
                << '/' << deposit_result.beam1_out
                << " beam2=" << deposit_result.beam2_in
                << '/' << deposit_result.beam2_out
                << " fine_lumi_dot=" << std::setprecision(17)
                << deposit_result.fine_lumi_dot
                << " dot_error=" << dot_error << '\n';
    }

  feebl::gpu::ResidentDepositResult zero_deposit_result;
  if (!solver.deposit_and_solve(
          xy1.data(), xy1.size() / 2, charge1,
          xy2.data(), xy2.size() / 2, charge2,
          inverse_x, inverse_y, offset_x, offset_y, false, 0.0f,
          phi1.data(), phi2.data(), zero_deposit_result, statistics, error))
    {
      std::cerr << "zero-gated resident CIC/field solve failed: "
                << error << '\n';
      return false;
    }
  passed = zero_deposit_result.beam1_in == unsym1.in && passed;
  passed = zero_deposit_result.beam1_out == unsym1.out && passed;
  passed = zero_deposit_result.beam2_in == unsym2.in && passed;
  passed = zero_deposit_result.beam2_out == unsym2.out && passed;
  const double zero_expected_dot = fine_lumi_reference(unsym1.rho, unsym2.rho);
  passed = std::fabs(zero_deposit_result.fine_lumi_dot - zero_expected_dot) <=
               2.0e-11 * std::max(1.0, std::fabs(zero_expected_dot)) && passed;
  passed = std::find_if(phi1.begin(), phi1.end(),
                        [](double value) { return value != 0.0; }) == phi1.end()
               && passed;
  passed = std::find_if(phi2.begin(), phi2.end(),
                        [](double value) { return value != 0.0; }) == phi2.end()
               && passed;
  passed = statistics.last_d2h_operations == 1 && passed;
  passed = statistics.last_synchronizations == 1 && passed;
  passed = statistics.deposition_calls == 3 && passed;
  passed = statistics.calls == 7 && passed;
  passed = statistics.plan_creations == 1 && passed;

  const std::uint64_t allocations_before_split = statistics.device_allocations;
  feebl::gpu::ResidentDepositResult split_result;
  if (!solver.deposit(xy1.data(), xy1.size() / 2, charge1,
                      xy2.data(), xy2.size() / 2, charge2,
                      inverse_x, inverse_y, offset_x, offset_y, false,
                      split_result, statistics, error))
    {
      std::cerr << "split resident deposit failed: " << error << '\n';
      return false;
    }
  passed = split_result.beam1_in == unsym1.in && passed;
  passed = split_result.beam1_out == unsym1.out && passed;
  passed = split_result.beam2_in == unsym2.in && passed;
  passed = split_result.beam2_out == unsym2.out && passed;
  passed = std::fabs(split_result.fine_lumi_dot - zero_expected_dot) <=
               2.0e-11 * std::max(1.0, std::fabs(zero_expected_dot)) && passed;
  passed = statistics.last_d2h_operations == 1 && passed;
  passed = statistics.last_d2h_bytes > 0 && passed;
  passed = statistics.device_allocations == allocations_before_split && passed;
  // This is where FeeBL performs collide_ee and update_slice_charge on the
  // host; the next call must consume the unchanged resident rho arrays.
  if (!solver.solve_deposited(1.0f, phi1.data(), phi2.data(),
                              statistics, error))
    {
      std::cerr << "split resident field solve failed: " << error << '\n';
      return false;
    }
  passed = close_vectors(direct_convolution(unsym1.rho, kernel, nx, ny),
                         phi1, "split resident deposit/FFT beam1") && passed;
  passed = close_vectors(direct_convolution(unsym2.rho, kernel, nx, ny),
                         phi2, "split resident deposit/FFT beam2") && passed;
  passed = statistics.last_h2d_operations == 0 && passed;
  passed = statistics.last_h2d_bytes == 0 && passed;
  passed = statistics.last_d2h_operations == 2 && passed;
  passed = statistics.device_allocations == allocations_before_split && passed;
  passed = statistics.deposition_calls == 4 && passed;
  passed = statistics.field_solve_calls == 8 && passed;
  passed = statistics.calls == 9 && passed;
  passed = statistics.device_work_calls == 8 && passed;
  passed = statistics.last_device_work && passed;

  std::string expected_failure;
  if (solver.reserve_particle_capacity(32, 32, expected_failure))
    {
      std::cerr << "post-execution particle reservation unexpectedly succeeded\n";
      passed = false;
    }
  std::vector<float> too_many(2 * 17, 0.0f);
  feebl::gpu::ResidentDepositResult rejected_result;
  ResidentFieldStats rejected_stats;
  if (solver.deposit(too_many.data(), 17, 1.0, xy2.data(), xy2.size() / 2,
                     charge2, inverse_x, inverse_y, offset_x, offset_y,
                     false, rejected_result, rejected_stats,
                     expected_failure))
    {
      std::cerr << "over-capacity resident deposit unexpectedly succeeded\n";
      passed = false;
    }

  std::cout << (pinned_io ? "pinned" : "pageable")
            << " accounting: calls=" << statistics.calls
            << " setup_ms=" << statistics.setup_ms
            << " h2d_ms=" << statistics.h2d_ms
            << " kernel_ms=" << statistics.kernel_ms
            << " d2h_ms=" << statistics.d2h_ms
            << " total_ms=" << statistics.total_ms
            << " explicit_syncs=" << statistics.synchronizations << '\n';
  if (!passed)
    std::cerr << (pinned_io ? "pinned" : "pageable")
              << " GPU2 resident FFT invariant failed\n";
  return passed;
}

}  // namespace

int main(int argc, char** argv)
{
  int device = 0;
  if (argc == 3 && std::string(argv[1]) == "--device")
    device = std::atoi(argv[2]);
  else if (argc != 1)
    {
      std::cerr << "usage: " << argv[0] << " [--device N]\n";
      return 2;
    }
  if (device < 0)
    {
      std::cerr << "device must be non-negative\n";
      return 2;
    }

  const bool pageable = run_suite(device, false);
  const bool pinned = run_suite(device, true);
  const bool compact_pageable = run_compact_gather_suite(device, false);
  const bool compact_pinned = run_compact_gather_suite(device, true);
  const bool nondyadic_pageable =
      run_non_dyadic_gather_case(device, false, 0);
  const bool nondyadic_pinned =
      run_non_dyadic_gather_case(device, true, 1);
#if defined(FEEBL_GPU_RESIDENT_TEST_HOOKS)
  const bool injected_pageable = run_injected_failure_case(device, false);
  const bool injected_pinned = run_injected_failure_case(device, true);
#else
  std::cout << "deterministic queued-work failure: SKIP "
               "(configure FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS=ON for "
               "qualification)\n";
  const bool injected_pageable = true;
  const bool injected_pinned = true;
#endif
  return pageable && pinned && compact_pageable && compact_pinned &&
         nondyadic_pageable && nondyadic_pinned &&
         injected_pageable && injected_pinned ? 0 : 1;
}
