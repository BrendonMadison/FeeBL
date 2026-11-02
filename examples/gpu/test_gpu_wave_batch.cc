#include "gpuResident.h"
#include "gpuWaveBatch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool close_field(float a, float b) {
  const double difference = std::fabs(static_cast<double>(a) - b);
  const double scale = std::max(std::fabs(static_cast<double>(a)),
                                std::fabs(static_cast<double>(b)));
  return difference <= 2.0e-6 * std::max(1.0, scale);
}

bool close_scalar(double a, double b) {
  const double difference = std::fabs(a - b);
  const double scale = std::max(std::fabs(a), std::fabs(b));
  return difference <= 2.0e-6 * std::max(1.0, scale);
}

}  // namespace

int main() {
  const int nx = 8;
  const int ny = 8;
  const std::size_t padded_cells =
      static_cast<std::size_t>(2 * nx) * static_cast<std::size_t>(2 * ny);
  std::vector<double> spectrum(2 * padded_cells, 0.0);
  for (std::size_t value = 0; value < padded_cells; ++value) {
    spectrum[2 * value] =
        (1.0 + static_cast<double>(value % 17)) /
        static_cast<double>(padded_cells);
    spectrum[2 * value + 1] =
        static_cast<double>(static_cast<int>(value % 7) - 3) /
        (100.0 * static_cast<double>(padded_cells));
  }

  const float xy11[] = {-1.2f, -0.7f, 0.1f, 0.3f, 3.5f, -1.0f};
  const float xy12[] = {-0.9f, 0.6f, 0.8f, -0.4f};
  const float xy21[] = {-1.0f, -0.4f, 0.4f, 0.2f};
  const float xy22[] = {-0.3f, 0.8f, 0.9f, -0.6f, -3.0f, 0.5f};
  const float* beam1[] = {xy11, xy12};
  const float* beam2[] = {xy21, xy22};
  const std::size_t count1[] = {3, 2};
  const std::size_t count2[] = {2, 3};

  feebl::gpu::WaveBatchOptions options;
  options.device = 0;
  options.batch_width = 3;
  options.maximum_particles1 = 3;
  options.maximum_particles2 = 3;
  options.memory_fraction = 0.8;
  options.pinned_io = true;

  // Initialization must reject a poisoned Green spectrum before it creates
  // a CUDA plan or copies any device state.
  std::vector<double> bad_spectrum = spectrum;
  bad_spectrum[7] = std::numeric_limits<double>::quiet_NaN();
  feebl::gpu::WaveBatchedFieldSolver bad_green;
  std::string error;
  if (bad_green.initialize(options, nx, ny, &bad_spectrum[0], error) ||
      error.find("nonfinite") == std::string::npos) {
    std::cerr << "nonfinite Green spectrum did not fail preflight\n";
    return 1;
  }

  feebl::gpu::WaveBatchedFieldSolver batch;
  if (!batch.initialize(options, nx, ny, &spectrum[0], error)) {
    std::cerr << "SKIP: wave-batched CUDA unavailable: " << error << '\n';
    return 77;
  }

  feebl::gpu::WaveBatchEncounterInput inputs[2];
  feebl::gpu::WaveBatchEncounterOutput outputs[2];
  std::vector<float> batch_fields1[2];
  std::vector<float> batch_fields2[2];
  for (std::size_t slot = 0; slot < 2; ++slot) {
    inputs[slot].xy1 = beam1[slot];
    inputs[slot].count1 = count1[slot];
    inputs[slot].xy2 = beam2[slot];
    inputs[slot].count2 = count2[slot];
    batch_fields1[slot].resize(2 * count1[slot]);
    batch_fields2[slot].resize(2 * count2[slot]);
    outputs[slot].fields1 = &batch_fields1[slot][0];
    outputs[slot].capacity1 = count1[slot];
    outputs[slot].fields2 = &batch_fields2[slot][0];
    outputs[slot].capacity2 = count2[slot];
  }
  if (!batch.execute(inputs, outputs, 2, 1.25, -0.75,
                     1.0f, 1.0f, 4.0f, 4.0f,
                     -3.0f, 3.0f, -3.0f, 3.0f,
                     false, -1.0f, error)) {
    std::cerr << "wave-batched execute failed: " << error << '\n';
    return 1;
  }

  for (std::size_t slot = 0; slot < 2; ++slot) {
    feebl::gpu::ResidentFieldOptions scalar_options;
    scalar_options.device = 0;
    scalar_options.pinned_io = true;
    scalar_options.memory_fraction = 0.8;
    feebl::gpu::ResidentFieldSolver scalar;
    if (!scalar.initialize(scalar_options, nx, ny, &spectrum[0], error) ||
        !scalar.reserve_particle_capacity(3, 3, error)) {
      std::cerr << "scalar resident preflight failed: " << error << '\n';
      return 1;
    }
    feebl::gpu::ResidentDepositResult deposit;
    feebl::gpu::ResidentFieldStats scalar_stats;
    if (!scalar.deposit(beam1[slot], count1[slot], 1.25,
                        beam2[slot], count2[slot], -0.75,
                        1.0f, 1.0f, 4.0f, 4.0f, false,
                        deposit, scalar_stats, error)) {
      std::cerr << "scalar deposit failed: " << error << '\n';
      return 1;
    }
    std::vector<float> scalar_fields1(2 * count1[slot]);
    std::vector<float> scalar_fields2(2 * count2[slot]);
    feebl::gpu::ResidentFieldGatherResult gather;
    if (!scalar.solve_deposited_and_gather(
            -1.0f, -3.0f, 3.0f, -3.0f, 3.0f,
            &scalar_fields1[0], count1[slot],
            &scalar_fields2[0], count2[slot],
            gather, scalar_stats, error)) {
      std::cerr << "scalar gather failed: " << error << '\n';
      return 1;
    }
    if (outputs[slot].beam1_in != deposit.beam1_in ||
        outputs[slot].beam1_out != deposit.beam1_out ||
        outputs[slot].beam2_in != deposit.beam2_in ||
        outputs[slot].beam2_out != deposit.beam2_out ||
        outputs[slot].gather_beam1_in != gather.beam1_in ||
        outputs[slot].gather_beam1_out != gather.beam1_out ||
        outputs[slot].gather_beam2_in != gather.beam2_in ||
        outputs[slot].gather_beam2_out != gather.beam2_out) {
      std::cerr << "batch/scalar population counters differ at slot "
                << slot << '\n';
      return 1;
    }
    if (!std::isfinite(outputs[slot].fine_lumi_dot) ||
        !close_scalar(outputs[slot].fine_lumi_dot,
                      deposit.fine_lumi_dot)) {
      std::cerr << "batch/scalar fine-luminosity dot differs at slot "
                << slot << '\n';
      return 1;
    }
    for (std::size_t value = 0; value < scalar_fields1.size(); ++value)
      if (!close_field(batch_fields1[slot][value], scalar_fields1[value])) {
        std::cerr << "beam1 field mismatch at slot/value " << slot << '/'
                  << value << '\n';
        return 1;
      }
    for (std::size_t value = 0; value < scalar_fields2.size(); ++value)
      if (!close_field(batch_fields2[slot][value], scalar_fields2[value])) {
        std::cerr << "beam2 field mismatch at slot/value " << slot << '/'
                  << value << '\n';
        return 1;
      }
  }

  // Exercise the zero-field shortcut in a one-encounter partial batch.  The
  // fields must be initialized to exact zero, not stale values from the first
  // submission, while deposition/fine luminosity remains active.
  feebl::gpu::WaveBatchEncounterInput zero_input = inputs[0];
  std::vector<float> zero_fields1(2 * count1[0], 99.0f);
  std::vector<float> zero_fields2(2 * count2[0], 99.0f);
  feebl::gpu::WaveBatchEncounterOutput zero_output;
  zero_output.fields1 = &zero_fields1[0];
  zero_output.capacity1 = count1[0];
  zero_output.fields2 = &zero_fields2[0];
  zero_output.capacity2 = count2[0];
  if (!batch.execute(&zero_input, &zero_output, 1, 1.25, -0.75,
                     1.0f, 1.0f, 4.0f, 4.0f,
                     -3.0f, 3.0f, -3.0f, 3.0f,
                     false, 0.0f, error)) {
    std::cerr << "zero-field partial batch failed: " << error << '\n';
    return 1;
  }
  for (float value : zero_fields1)
    if (value != 0.0f) {
      std::cerr << "zero-field beam1 output retained stale data\n";
      return 1;
    }
  for (float value : zero_fields2)
    if (value != 0.0f) {
      std::cerr << "zero-field beam2 output retained stale data\n";
      return 1;
    }
  if (!std::isfinite(zero_output.fine_lumi_dot)) {
    std::cerr << "zero-field fine-luminosity dot is nonfinite\n";
    return 1;
  }

  feebl::gpu::WaveBatchEncounterInput empty_input;
  feebl::gpu::WaveBatchEncounterOutput empty_output;
  empty_output.beam1_in = 99;
  empty_output.gather_beam2_out = 99;
  empty_output.fine_lumi_dot = 99.0;
  if (!batch.execute(&empty_input, &empty_output, 1, 1.25, -0.75,
                     1.0f, 1.0f, 4.0f, 4.0f,
                     -3.0f, 3.0f, -3.0f, 3.0f,
                     false, 0.0f, error) ||
      empty_output.beam1_in != 0 || empty_output.beam1_out != 0 ||
      empty_output.beam2_in != 0 || empty_output.beam2_out != 0 ||
      empty_output.gather_beam1_in != 0 ||
      empty_output.gather_beam1_out != 0 ||
      empty_output.gather_beam2_in != 0 ||
      empty_output.gather_beam2_out != 0 ||
      empty_output.fine_lumi_dot != 0.0) {
    std::cerr << "empty encounter did not produce an exact empty result\n";
    return 1;
  }

  // Preflight rejection is transactional: neither caller-owned fields and
  // counters nor observed-work telemetry may change.
  const float nonfinite_xy[] = {
      std::numeric_limits<float>::quiet_NaN(), 0.0f};
  feebl::gpu::WaveBatchEncounterInput rejected_input;
  rejected_input.xy1 = nonfinite_xy;
  rejected_input.count1 = 1;
  float rejected_fields[] = {17.0f, 23.0f};
  feebl::gpu::WaveBatchEncounterOutput rejected_output;
  rejected_output.fields1 = rejected_fields;
  rejected_output.capacity1 = 1;
  rejected_output.beam1_in = 41;
  rejected_output.fine_lumi_dot = 43.0;
  const feebl::gpu::WaveBatchStats before_rejection = batch.stats();
  if (batch.execute(&rejected_input, &rejected_output, 1, 1.0, 1.0,
                    1.0f, 1.0f, 4.0f, 4.0f,
                    -3.0f, 3.0f, -3.0f, 3.0f,
                    false, -1.0f, error) ||
      error.find("nonfinite") == std::string::npos ||
      rejected_fields[0] != 17.0f || rejected_fields[1] != 23.0f ||
      rejected_output.beam1_in != 41 ||
      rejected_output.fine_lumi_dot != 43.0 ||
      batch.stats().batch_calls != before_rejection.batch_calls) {
    std::cerr << "nonfinite-xy rejection violated API atomicity\n";
    return 1;
  }
  if (batch.execute(&zero_input, &zero_output, 1, 1.0, 1.0,
                    1.0f, 1.0f, 4.0f, 4.0f,
                    -3.0f, 3.0f, -3.0f, 3.0f,
                    false, std::numeric_limits<float>::infinity(), error) ||
      error.find("invalid grid/deposition") == std::string::npos ||
      batch.stats().batch_calls != before_rejection.batch_calls) {
    std::cerr << "nonfinite charge-sign rejection failed\n";
    return 1;
  }

  const feebl::gpu::WaveBatchStats stats = batch.stats();
  if (stats.batch_calls != 3 || stats.encounters != 4 ||
      stats.padded_fft_slots != 5 || stats.synchronizations != 3 ||
      stats.h2d_operations != 9 || stats.d2h_operations != 9 ||
      stats.kernel_launches != 18 || stats.cufft_executes != 2 ||
      stats.plan_creations != 1 || stats.setup_synchronizations != 1 ||
      stats.context_poisoned) {
    std::cerr << "wave-batch telemetry invariant failed\n";
    return 1;
  }
  std::cout << "PASS: PlanMany wave batch matches field/fine-luminosity "
               "oracles, isolates padding and zero-field work, and rejects "
               "nonfinite inputs transactionally\n";
  return 0;
}
