#include "gpuCic.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using feebl::gpu::CicDepositor;
using feebl::gpu::CicStats;
using feebl::gpu::CicVariant;

struct Grid
{
  int nx;
  int ny;
  float inverse_x;
  float inverse_y;
  float offset_x;
  float offset_y;
};

struct Reference
{
  std::vector<double> rho;
  std::size_t in;
  std::size_t out;
  double elapsed_ms;
};

struct ErrorMetrics
{
  double max_absolute;
  double max_relative;
  double l1;
  double l2;
  double reference_l1;
  double reference_max;
  double reference_sum;
  double candidate_sum;
};

bool locate(float x,
            float y,
            const Grid& grid,
            int& ix,
            int& iy,
            float& hx,
            float& hy)
{
  hx = x * grid.inverse_x + grid.offset_x - 0.5f;
  ix = static_cast<int>(std::floor(hx));
  hx -= static_cast<float>(ix);
  hy = y * grid.inverse_y + grid.offset_y - 0.5f;
  iy = static_cast<int>(std::floor(hy));
  hy -= static_cast<float>(iy);
  return ix >= 0 && ix < grid.nx - 1 && iy >= 0 && iy < grid.ny - 1;
}

Reference deposit_reference(const std::vector<float>& xy,
                            const Grid& grid,
                            double charge)
{
  const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  Reference result;
  result.rho.assign(static_cast<std::size_t>(grid.nx) * grid.ny, 0.0);
  result.in = 0;
  result.out = 0;
  for (std::size_t particle = 0; particle < xy.size() / 2; ++particle)
    {
      int ix, iy;
      float hx, hy;
      if (!locate(xy[2 * particle], xy[2 * particle + 1],
                  grid, ix, iy, hx, hy))
        {
          ++result.out;
          continue;
        }
      ++result.in;
      const double dx = static_cast<double>(hx);
      const double dy = static_cast<double>(hy);
      result.rho[ix * grid.ny + iy] +=
          (1.0 - dx) * (1.0 - dy) * charge;
      result.rho[(ix + 1) * grid.ny + iy] +=
          dx * (1.0 - dy) * charge;
      result.rho[ix * grid.ny + iy + 1] +=
          (1.0 - dx) * dy * charge;
      result.rho[(ix + 1) * grid.ny + iy + 1] += dx * dy * charge;
    }
  result.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin).count();
  return result;
}

ErrorMetrics compare(const std::vector<double>& reference,
                     const std::vector<double>& candidate)
{
  ErrorMetrics metrics = {};
  double squared = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index)
    {
      const double absolute = std::fabs(reference[index] - candidate[index]);
      metrics.max_absolute = std::max(metrics.max_absolute, absolute);
      if (std::fabs(reference[index]) > 1.0e-300)
        metrics.max_relative = std::max(
            metrics.max_relative, absolute / std::fabs(reference[index]));
      metrics.l1 += absolute;
      squared += absolute * absolute;
      metrics.reference_l1 += std::fabs(reference[index]);
      metrics.reference_max = std::max(metrics.reference_max,
                                       std::fabs(reference[index]));
      metrics.reference_sum += reference[index];
      metrics.candidate_sum += candidate[index];
    }
  metrics.l2 = std::sqrt(squared);
  return metrics;
}

float physical_coordinate(float grid_coordinate,
                          float inverse,
                          float offset)
{
  return (grid_coordinate - offset + 0.5f) / inverse;
}

std::vector<float> boundary_particles(const Grid& grid)
{
  const float points[][2] = {
      {0.0f, 0.0f},
      {0.25f, 0.75f},
      {1.0f, 1.0f},
      {static_cast<float>(grid.nx - 2),
       static_cast<float>(grid.ny - 2)},
      {static_cast<float>(grid.nx - 1) - 1.0e-4f,
       static_cast<float>(grid.ny - 1) - 1.0e-4f},
      {-1.0e-4f, 2.0f},
      {static_cast<float>(grid.nx - 1), 2.0f},
      {2.0f, -0.01f},
      {2.0f, static_cast<float>(grid.ny - 1)},
      {3.5f, 4.5f},
      {3.5f, 4.5f}};
  std::vector<float> xy;
  for (std::size_t index = 0; index < sizeof(points) / sizeof(points[0]); ++index)
    {
      xy.push_back(physical_coordinate(points[index][0], grid.inverse_x,
                                       grid.offset_x));
      xy.push_back(physical_coordinate(points[index][1], grid.inverse_y,
                                       grid.offset_y));
    }
  return xy;
}

std::vector<float> random_particles(std::size_t count,
                                    const Grid& grid,
                                    unsigned int seed,
                                    bool clustered)
{
  std::mt19937 generator(seed);
  std::vector<float> xy(2 * count);
  if (clustered)
    {
      std::normal_distribution<float> x_distribution(
          0.5f * (grid.nx - 1), 0.08f * grid.nx);
      std::normal_distribution<float> y_distribution(
          0.5f * (grid.ny - 1), 0.08f * grid.ny);
      for (std::size_t particle = 0; particle < count; ++particle)
        {
          xy[2 * particle] = physical_coordinate(
              x_distribution(generator), grid.inverse_x, grid.offset_x);
          xy[2 * particle + 1] = physical_coordinate(
              y_distribution(generator), grid.inverse_y, grid.offset_y);
        }
    }
  else
    {
      std::uniform_real_distribution<float> x_distribution(
          -1.0f, static_cast<float>(grid.nx));
      std::uniform_real_distribution<float> y_distribution(
          -1.0f, static_cast<float>(grid.ny));
      for (std::size_t particle = 0; particle < count; ++particle)
        {
          xy[2 * particle] = physical_coordinate(
              x_distribution(generator), grid.inverse_x, grid.offset_x);
          xy[2 * particle + 1] = physical_coordinate(
              y_distribution(generator), grid.inverse_y, grid.offset_y);
        }
    }
  return xy;
}

std::string csv_number(double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

bool accuracy_passes(const Reference& reference,
                     const CicStats& stats,
                     const ErrorMetrics& metrics,
                     std::string& reason)
{
  if (stats.particles_in != reference.in ||
      stats.particles_out != reference.out)
    {
      reason = "in/out particle count differs";
      return false;
    }
  const double max_limit = 2.0e-11 * std::max(1.0, metrics.reference_max);
  const double l1_limit = 2.0e-11 * std::max(1.0, metrics.reference_l1);
  const double charge_limit =
      2.0e-11 * std::max(1.0, std::fabs(metrics.reference_sum));
  if (metrics.max_absolute > max_limit)
    {
      reason = "maximum cell error exceeds preregistered tolerance";
      return false;
    }
  if (metrics.l1 > l1_limit)
    {
      reason = "L1 grid error exceeds preregistered tolerance";
      return false;
    }
  if (std::fabs(metrics.reference_sum - metrics.candidate_sum) > charge_limit)
    {
      reason = "total deposited charge exceeds preregistered tolerance";
      return false;
    }
  reason = "pass";
  return true;
}

bool run_correctness(CicDepositor& depositor,
                     const std::string& csv_path)
{
  const Grid edge_grid = {9, 11, 1.25f, 0.75f, 4.5f, 5.5f};
  const Grid random_grid = {129, 97, 0.8125f, 1.375f, 64.5f, 48.5f};
  const double charge = 3.25e6;

  struct Case
  {
    std::string name;
    Grid grid;
    std::vector<float> xy;
  };
  std::vector<Case> cases;
  cases.push_back(Case{"boundaries", edge_grid,
                       boundary_particles(edge_grid)});
  cases.push_back(Case{"uniform_with_outliers", random_grid,
                       random_particles(200000, random_grid, 42017, false)});
  cases.push_back(Case{"clustered", random_grid,
                       random_particles(200000, random_grid, 7331, true)});

  std::ofstream csv;
  if (!csv_path.empty())
    {
      csv.open(csv_path.c_str());
      if (!csv)
        {
          std::cerr << "cannot open correctness CSV: " << csv_path << '\n';
          return false;
        }
      csv << "case,variant,particles,in,out,max_abs,max_rel,l1,l2,"
             "reference_sum,candidate_sum,h2d_ms,kernel_ms,d2h_ms,total_ms,result\n";
    }

  const CicVariant variants[] = {CicVariant::Atomic,
                                 CicVariant::Shared,
                                 CicVariant::Sorted};
  bool all_passed = true;
  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index)
    {
      const Case& test_case = cases[case_index];
      const Reference reference =
          deposit_reference(test_case.xy, test_case.grid, charge);
      for (std::size_t variant_index = 0;
           variant_index < sizeof(variants) / sizeof(variants[0]);
           ++variant_index)
        {
          const CicVariant variant = variants[variant_index];
          std::vector<double> candidate(reference.rho.size(),
                                        std::numeric_limits<double>::quiet_NaN());
          CicStats stats;
          std::string error;
          if (!depositor.deposit(test_case.xy.empty() ? 0 : &test_case.xy[0],
                                 test_case.xy.size() / 2, charge,
                                 test_case.grid.nx, test_case.grid.ny,
                                 test_case.grid.inverse_x,
                                 test_case.grid.inverse_y,
                                 test_case.grid.offset_x,
                                 test_case.grid.offset_y,
                                 variant, &candidate[0], stats, error))
            {
              std::cerr << "correctness deposit failed for "
                        << feebl::gpu::cic_variant_name(variant) << ": "
                        << error << '\n';
              return false;
            }
          if (variant == CicVariant::Sorted)
            {
              // Thrust's reduction order is not a cross-device contract. It
              // must nevertheless be bitwise repeatable for an unchanged
              // binary, toolkit, device, and input before FeeBL admits it as
              // the local deterministic reduction mode.
              for (int repeat = 0; repeat < 2; ++repeat)
                {
                  std::vector<double> repeated(candidate.size());
                  CicStats repeated_stats;
                  if (!depositor.deposit(
                          test_case.xy.empty() ? 0 : &test_case.xy[0],
                          test_case.xy.size() / 2, charge,
                          test_case.grid.nx, test_case.grid.ny,
                          test_case.grid.inverse_x,
                          test_case.grid.inverse_y,
                          test_case.grid.offset_x,
                          test_case.grid.offset_y,
                          variant, &repeated[0], repeated_stats, error))
                    {
                      std::cerr << "sorted repeat deposit failed: "
                                << error << '\n';
                      return false;
                    }
                  if (repeated_stats.particles_in != stats.particles_in ||
                      repeated_stats.particles_out != stats.particles_out ||
                      std::memcmp(&candidate[0], &repeated[0],
                                  candidate.size() * sizeof(double)) != 0)
                    {
                      std::cerr << "sorted deposition is not bitwise "
                                   "repeatable on this device/toolkit\n";
                      return false;
                    }
                }
              std::cout << "correctness " << test_case.name
                        << " sorted-repeat: PASS (same binary/toolkit/device)\n";
            }
          const ErrorMetrics metrics = compare(reference.rho, candidate);
          std::string reason;
          const bool passed = accuracy_passes(reference, stats, metrics, reason);
          all_passed = all_passed && passed;
          std::cout << "correctness " << test_case.name << ' '
                    << feebl::gpu::cic_variant_name(variant) << ": "
                    << (passed ? "PASS" : "FAIL")
                    << " max_abs=" << std::setprecision(6)
                    << std::scientific << metrics.max_absolute
                    << " l1=" << metrics.l1
                    << " charge_delta="
                    << (metrics.candidate_sum - metrics.reference_sum)
                    << " (" << reason << ")\n";
          if (csv)
            csv << test_case.name << ','
                << feebl::gpu::cic_variant_name(variant) << ','
                << test_case.xy.size() / 2 << ','
                << stats.particles_in << ',' << stats.particles_out << ','
                << csv_number(metrics.max_absolute) << ','
                << csv_number(metrics.max_relative) << ','
                << csv_number(metrics.l1) << ','
                << csv_number(metrics.l2) << ','
                << csv_number(metrics.reference_sum) << ','
                << csv_number(metrics.candidate_sum) << ','
                << csv_number(stats.h2d_ms) << ','
                << csv_number(stats.kernel_ms) << ','
                << csv_number(stats.d2h_ms) << ','
                << csv_number(stats.total_ms) << ','
                << (passed ? "PASS" : "FAIL") << '\n';
        }
    }
  return all_passed;
}

void write_benchmark_header(std::ofstream& csv)
{
  csv << "backend,phase,particles,nx,ny,repeat,in,out,setup_ms,h2d_ms,"
         "kernel_ms,d2h_ms,total_ms,max_abs,l1,charge_delta\n";
}

bool run_benchmark(CicDepositor& depositor,
                   int device,
                   const std::string& csv_path,
                   int repetitions,
                   std::size_t max_particles)
{
  std::ofstream csv(csv_path.c_str());
  if (!csv)
    {
      std::cerr << "cannot open benchmark CSV: " << csv_path << '\n';
      return false;
    }
  write_benchmark_header(csv);

  const Grid grid = {256, 256, 1.0f, 1.0f, 128.0f, 128.0f};
  const double charge = 1.0;
  std::set<std::size_t> size_set;
  const std::size_t candidates[] = {1024, 16384, 262144,
                                    1048576, 4194304};
  for (std::size_t index = 0;
       index < sizeof(candidates) / sizeof(candidates[0]); ++index)
    if (candidates[index] <= max_particles) size_set.insert(candidates[index]);
  size_set.insert(max_particles);

  const CicVariant variants[] = {CicVariant::Atomic,
                                 CicVariant::Shared,
                                 CicVariant::Sorted};
  volatile double checksum = 0.0;
  for (std::set<std::size_t>::const_iterator size = size_set.begin();
       size != size_set.end(); ++size)
    {
      const std::vector<float> xy = random_particles(*size, grid,
                                                     918273 + *size, true);
      Reference reference;
      for (int repeat = 0; repeat < repetitions; ++repeat)
        {
          reference = deposit_reference(xy, grid, charge);
          checksum += reference.rho[reference.rho.size() / 2];
          csv << "cpu_serial,steady," << *size << ',' << grid.nx << ','
              << grid.ny << ',' << repeat << ',' << reference.in << ','
              << reference.out << ",0,0," << reference.elapsed_ms
              << ",0," << reference.elapsed_ms << ",0,0,0\n";
        }

      for (std::size_t variant_index = 0;
           variant_index < sizeof(variants) / sizeof(variants[0]);
           ++variant_index)
        {
          const CicVariant variant = variants[variant_index];
          // A fresh context captures allocation and first-use costs.
          CicDepositor cold;
          std::string error;
          if (!cold.initialize(device, error))
            {
              std::cerr << "cold benchmark initialization failed: "
                        << error << '\n';
              return false;
            }
          std::vector<double> candidate(reference.rho.size());
          CicStats stats;
          if (!cold.deposit(&xy[0], *size, charge, grid.nx, grid.ny,
                            grid.inverse_x, grid.inverse_y,
                            grid.offset_x, grid.offset_y, variant,
                            &candidate[0], stats, error))
            {
              std::cerr << "cold benchmark failed: " << error << '\n';
              return false;
            }
          ErrorMetrics metrics = compare(reference.rho, candidate);
          csv << feebl::gpu::cic_variant_name(variant) << ",cold,"
              << *size << ',' << grid.nx << ',' << grid.ny << ",-1,"
              << stats.particles_in << ',' << stats.particles_out << ','
              << stats.setup_ms << ',' << stats.h2d_ms << ','
              << stats.kernel_ms << ',' << stats.d2h_ms << ','
              << stats.total_ms << ',' << metrics.max_absolute << ','
              << metrics.l1 << ','
              << metrics.candidate_sum - metrics.reference_sum << '\n';

          // Warm the long-lived context once before recording steady state.
          if (!depositor.deposit(&xy[0], *size, charge, grid.nx, grid.ny,
                                 grid.inverse_x, grid.inverse_y,
                                 grid.offset_x, grid.offset_y, variant,
                                 &candidate[0], stats, error))
            {
              std::cerr << "warmup failed: " << error << '\n';
              return false;
            }
          for (int repeat = 0; repeat < repetitions; ++repeat)
            {
              if (!depositor.deposit(&xy[0], *size, charge,
                                     grid.nx, grid.ny,
                                     grid.inverse_x, grid.inverse_y,
                                     grid.offset_x, grid.offset_y, variant,
                                     &candidate[0], stats, error))
                {
                  std::cerr << "steady benchmark failed: " << error << '\n';
                  return false;
                }
              metrics = compare(reference.rho, candidate);
              checksum += candidate[candidate.size() / 2];
              csv << feebl::gpu::cic_variant_name(variant) << ",steady,"
                  << *size << ',' << grid.nx << ',' << grid.ny << ','
                  << repeat << ',' << stats.particles_in << ','
                  << stats.particles_out << ',' << stats.setup_ms << ','
                  << stats.h2d_ms << ',' << stats.kernel_ms << ','
                  << stats.d2h_ms << ',' << stats.total_ms << ','
                  << metrics.max_absolute << ',' << metrics.l1 << ','
                  << metrics.candidate_sum - metrics.reference_sum << '\n';
            }
        }
      std::cout << "benchmarked " << *size << " particles\n";
    }
  std::cout << "benchmark checksum=" << std::setprecision(17) << checksum
            << " results=" << csv_path << '\n';
  return true;
}

bool parse_positive(const char* text, std::size_t& value)
{
  char* end = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (!end || *end != '\0' || parsed == 0) return false;
  value = static_cast<std::size_t>(parsed);
  return static_cast<unsigned long long>(value) == parsed;
}

}  // namespace

int main(int argc, char** argv)
{
  bool correctness = argc == 1;
  std::string correctness_csv;
  std::string benchmark_csv;
  int device = 0;
  int repetitions = 7;
  std::size_t max_particles = 4194304;

  for (int argument = 1; argument < argc; ++argument)
    {
      const std::string option(argv[argument]);
      if (option == "--correctness") correctness = true;
      else if (option == "--correctness-csv" && argument + 1 < argc)
        correctness_csv = argv[++argument];
      else if (option == "--benchmark" && argument + 1 < argc)
        benchmark_csv = argv[++argument];
      else if (option == "--device" && argument + 1 < argc)
        device = std::atoi(argv[++argument]);
      else if (option == "--repetitions" && argument + 1 < argc)
        repetitions = std::atoi(argv[++argument]);
      else if (option == "--max-particles" && argument + 1 < argc)
        {
          if (!parse_positive(argv[++argument], max_particles))
            {
              std::cerr << "invalid --max-particles value\n";
              return 2;
            }
        }
      else
        {
          std::cerr << "usage: " << argv[0]
                    << " [--correctness] [--correctness-csv FILE]"
                    << " [--benchmark FILE] [--device N]"
                    << " [--repetitions N] [--max-particles N]\n";
          return 2;
        }
    }
  if (!correctness_csv.empty()) correctness = true;
  if (device < 0 || repetitions < 1)
    {
      std::cerr << "device must be non-negative and repetitions positive\n";
      return 2;
    }

  CicDepositor depositor;
  std::string error;
  if (!depositor.initialize(device, error))
    {
      std::cerr << "CUDA initialization failed: " << error << '\n';
      return 1;
    }
  std::cout << "device: " << depositor.device_description() << '\n';

  bool passed = true;
  if (correctness)
    passed = run_correctness(depositor, correctness_csv) && passed;
  if (!benchmark_csv.empty())
    passed = run_benchmark(depositor, device, benchmark_csv,
                           repetitions, max_particles) && passed;
  return passed ? 0 : 1;
}
