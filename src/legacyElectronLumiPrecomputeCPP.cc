#include "legacyElectronLumiPrecomputeCPP.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace feebl {
namespace modern {
namespace legacy_ee {
namespace {

Status failure(StatusCode code, const char* message) {
  Status result;
  result.code = code;
  result.message = message;
  return result;
}

bool finite_sample(const ParticleSample& particle) noexcept {
  return std::isfinite(particle.signed_energy) &&
         std::isfinite(particle.vx) && std::isfinite(particle.vy) &&
         std::isfinite(particle.weight) &&
         (!particle.has_spin ||
          (std::isfinite(particle.spin_x) &&
           std::isfinite(particle.spin_y) &&
           std::isfinite(particle.spin_z)));
}

bool same_double_bits(double first, double second) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "the exact legacy oracle requires binary64 doubles");
  std::uint64_t first_bits = 0;
  std::uint64_t second_bits = 0;
  std::memcpy(&first_bits, &first, sizeof(first_bits));
  std::memcpy(&second_bits, &second, sizeof(second_bits));
  return first_bits == second_bits;
}

bool spectrum_enabled(const SpectrumConfig& spectrum) noexcept {
  return spectrum.bins != 0 && spectrum.high > spectrum.low;
}

Status validate_spectrum_config(const SpectrumConfig& spectrum,
                                const KernelLimits& limits) {
  if (spectrum.bins == 0) return Status::success();
  if (!std::isfinite(spectrum.low) || !std::isfinite(spectrum.high))
    return failure(StatusCode::InvalidSpectrumConfig,
                   "enabled spectrum bounds must be finite");
  if (!(spectrum.high > spectrum.low)) return Status::success();
  const double span = spectrum.high - spectrum.low;
  const double width = span / static_cast<double>(spectrum.bins);
  if (!std::isfinite(span) || !std::isfinite(width) || !(width > 0.0))
    return failure(StatusCode::InvalidSpectrumConfig,
                   "enabled spectrum span and width must be finite and positive");
  if (spectrum.bins > limits.max_spectrum_bins)
    return failure(StatusCode::SpectrumBinLimitExceeded,
                   "spectrum bin count exceeds configured limit");
  constexpr std::uint64_t bytes_per_bin =
      sizeof(double) + sizeof(PairHelicityRaw);
  if (spectrum.bins >
      kAbsoluteSpectrumStorageBytes / bytes_per_bin)
    return failure(StatusCode::SpectrumBinLimitExceeded,
                   "spectrum storage exceeds the non-overridable byte budget");
  const std::vector<double> weight_probe;
  const std::vector<PairHelicityRaw> helicity_probe;
  if (spectrum.bins > weight_probe.max_size() ||
      spectrum.bins > helicity_probe.max_size())
    return failure(StatusCode::SpectrumBinLimitExceeded,
                   "spectrum bin count exceeds vector max_size");
  return Status::success();
}

Status validate_legacy_mass(double mass) {
  if (!same_double_bits(mass, kLegacyElectronMassGeV))
    return failure(StatusCode::InvalidElectronMass,
                   "electron mass must exactly match GRID's legacy EMASS");
  return Status::success();
}

Status validate_inputs(const std::vector<ParticleSample>& beam1,
                       const std::vector<ParticleSample>& beam2,
                       const KernelConfig& config) {
  const Status eligible = check_eligibility(config.eligibility);
  if (!eligible) return eligible;
  const Status mass = validate_legacy_mass(config.electron_mass);
  if (!mass) return mass;
  const Status spectrum =
      validate_spectrum_config(config.spectrum, config.limits);
  if (!spectrum) return spectrum;
  const Status size = validate_problem_size(
      beam1.size(), beam2.size(), config.spectrum.bins, config.limits);
  if (!size) return size;
  if (config.eligibility.bmt_spin) {
    for (const ParticleSample& particle : beam1)
      if (!particle.has_spin)
        return failure(StatusCode::MissingSpin,
                       "BMT mode requires spin on every beam-1 particle");
    for (const ParticleSample& particle : beam2)
      if (!particle.has_spin)
        return failure(StatusCode::MissingSpin,
                       "BMT mode requires spin on every beam-2 particle");
  }
  return Status::success();
}

float legacy_longitudinal_velocity(float vx, float vy,
                                   OperationCounts& operations,
                                   bool preprocessing) {
  if (preprocessing)
    ++operations.precompute_sqrt_calls;
  else
    ++operations.pair_sqrt_calls;
  // Keep the double literal, float products, evaluation order, double sqrt,
  // and final float narrowing of GRID::collide_ee.
  return static_cast<float>(std::sqrt(1. - vx * vx - vy * vy));
}

float legacy_ecm(float e1, float e2,
                 float vx1, float vy1, float vz1,
                 float vx2, float vy2, float vz2,
                 OperationCounts& operations) {
  const float cosinus = vx1 * vx2 + vy1 * vy2 - vz1 * vz2;
  ++operations.pair_sqrt_calls;
  return static_cast<float>(
      std::sqrt(2.0 * e1 * e2 * (1. - cosinus)));
}

double legacy_massive_momentum(double energy, double mass,
                               OperationCounts& operations,
                               bool preprocessing) {
  if (!(energy > mass)) return 0.0;
  if (preprocessing)
    ++operations.precompute_sqrt_calls;
  else
    ++operations.pair_sqrt_calls;
  return std::sqrt((energy - mass) * (energy + mass));
}

double finish_invariant_mass(double energy1, double vx1, double vy1,
                             double vz1, double momentum1,
                             double energy2, double vx2, double vy2,
                             double vz2, double momentum2, double mass,
                             OperationCounts& operations) {
  const double cosang = vx1 * vx2 + vy1 * vy2 + vz1 * vz2;
  const double w2 = 2.0 * mass * mass +
                    2.0 * (energy1 * energy2 -
                           momentum1 * momentum2 * cosang);
  if (!(w2 > 0.0)) return 0.0;
  ++operations.pair_sqrt_calls;
  return std::sqrt(w2);
}

void inspect_inputs(AggregateResults& results,
                    const std::vector<ParticleSample>& beam1,
                    const std::vector<ParticleSample>& beam2) {
  const auto inspect = [&results](const ParticleSample& particle) {
    if (!finite_sample(particle)) results.nonfinite_input_observed = true;
    const double transverse2 =
        static_cast<double>(particle.vx) * particle.vx +
        static_cast<double>(particle.vy) * particle.vy;
    if (!(transverse2 < 1.0)) results.invalid_velocity_observed = true;
  };
  for (const ParticleSample& particle : beam1) inspect(particle);
  for (const ParticleSample& particle : beam2) inspect(particle);
}

Status validate_accumulator(const AggregateResults& results,
                            const SpectrumConfig& spectrum) {
  if (!spectrum_enabled(spectrum)) {
    if (!results.spectrum.bin_weight.empty() ||
        !results.spectrum.bin_helicity.empty() ||
        results.spectrum.width != 0.0)
      return failure(StatusCode::AccumulatorConfigurationMismatch,
                     "seeded accumulator has an enabled spectrum for a disabled config");
    return Status::success();
  }
  const double expected_width =
      (spectrum.high - spectrum.low) / static_cast<double>(spectrum.bins);
  if (results.spectrum.bin_weight.size() != spectrum.bins ||
      results.spectrum.bin_helicity.size() != spectrum.bins ||
      !same_double_bits(results.spectrum.low, spectrum.low) ||
      !same_double_bits(results.spectrum.high, spectrum.high) ||
      !same_double_bits(results.spectrum.width, expected_width))
    return failure(StatusCode::AccumulatorConfigurationMismatch,
                   "seeded accumulator spectrum does not match KernelConfig");
  return Status::success();
}

Status make_zero_results(AggregateResults& results,
                         const KernelConfig& config) {
  const Status spectrum =
      validate_spectrum_config(config.spectrum, config.limits);
  if (!spectrum) return spectrum;
  return results.spectrum.configure(config.spectrum);
}

Status copy_seeded_results(AggregateResults& destination,
                           const AggregateResults& source) {
  try {
    destination = source;
  } catch (const std::bad_alloc&) {
    return failure(StatusCode::AllocationFailure,
                   "allocation failed while copying the seeded accumulator");
  } catch (const std::length_error&) {
    return failure(StatusCode::AllocationFailure,
                   "invalid vector length while copying the seeded accumulator");
  }
  return Status::success();
}

void add_spin_and_spectrum(AggregateResults& results, float pair_weight,
                           double invariant_mass, float legacy_spin1,
                           float legacy_spin2, bool helicity_valid,
                           double helicity1, double helicity2) {
  float help = static_cast<float>(
      0.5 * (1.0 + legacy_spin1 * legacy_spin2));
  results.lumi_pp += pair_weight * help;
  if (helicity_valid) {
    const double weight = static_cast<double>(pair_weight);
    results.lumi_weight_sum += weight;
    results.lumi_weighted_helicity1 += weight * helicity1;
    results.lumi_weighted_helicity2 += weight * helicity2;
    results.spectrum.add(weight, invariant_mass, helicity1, helicity2);
  } else {
    results.spectrum.add(static_cast<double>(pair_weight), invariant_mass);
  }
}

void add_nonspin_spectrum(AggregateResults& results, float pair_weight,
                          double invariant_mass) {
  results.spectrum.add(static_cast<double>(pair_weight), invariant_mass);
}

void add_common_pair_results(AggregateResults& results, float pair_weight,
                             float ecm, float ecm_min) {
  if (ecm > ecm_min) {
    results.lumi_ee_high += pair_weight;
    results.step_lumi_ee_high += pair_weight;
  }
  const float help = pair_weight * ecm;
  results.lumi_ecm += help;
  results.lumi_ecm2 += help * ecm;
}

}  // namespace

void Moment1Raw::add(double w, double x) noexcept {
  weight += w;
  weight2 += w * w;
  weighted_x += w * x;
  weighted_x2 += w * x * x;
}

void PairHelicityRaw::add(double w, double first, double second) noexcept {
  h1.add(w, first);
  h2.add(w, second);
  weight += w;
  weighted_product += w * first * second;
}

Status SpectrumRaw::configure(const SpectrumConfig& config) {
  KernelLimits allocation_limits;
  allocation_limits.max_spectrum_bins =
      std::numeric_limits<std::uint64_t>::max();
  const Status safe = validate_spectrum_config(config, allocation_limits);
  if (!safe) return safe;
  SpectrumRaw replacement;
  if (!(config.high > config.low) || config.bins == 0) {
    *this = std::move(replacement);
    return Status::success();
  }
  replacement.low = config.low;
  replacement.high = config.high;
  replacement.width =
      (replacement.high - replacement.low) /
      static_cast<double>(config.bins);
  try {
    replacement.bin_weight.assign(config.bins, 0.0);
    replacement.bin_helicity.assign(config.bins, PairHelicityRaw{});
  } catch (const std::bad_alloc&) {
    return failure(StatusCode::AllocationFailure,
                   "allocation failed while configuring spectrum storage");
  } catch (const std::length_error&) {
    return failure(StatusCode::AllocationFailure,
                   "invalid vector length while configuring spectrum storage");
  }
  *this = std::move(replacement);
  return Status::success();
}

bool SpectrumRaw::configured() const noexcept {
  return !bin_weight.empty() && width > 0.0;
}

void SpectrumRaw::add(double pair_weight, double invariant_mass) {
  w_moment.add(pair_weight, invariant_mass);
  if (!configured()) return;
  if (invariant_mass < low) {
    underflow += pair_weight;
    return;
  }
  if (invariant_mass >= high) {
    overflow += pair_weight;
    return;
  }
  std::size_t bin = static_cast<std::size_t>(
      (invariant_mass - low) / width);
  if (bin >= bin_weight.size()) bin = bin_weight.size() - 1;
  bin_weight[bin] += pair_weight;
}

void SpectrumRaw::add(double pair_weight, double invariant_mass,
                      double helicity1, double helicity2) {
  add(pair_weight, invariant_mass);
  if (!configured() || invariant_mass < low || invariant_mass >= high) return;
  std::size_t bin = static_cast<std::size_t>(
      (invariant_mass - low) / width);
  if (bin >= bin_helicity.size()) bin = bin_helicity.size() - 1;
  bin_helicity[bin].add(pair_weight, helicity1, helicity2);
}

Status check_eligibility(const EligibilityOptions& options) {
  if (options.do_isr)
    return failure(StatusCode::IsrUnsupported,
                   "ISR changes energies with RNG draws and is unsupported");
  if (options.do_ee_lumi_heap)
    return failure(StatusCode::EeLumiHeapUnsupported,
                   "electron luminosity heap sampling is unsupported");
  if (options.do_cross_events)
    return failure(StatusCode::CrossEventsUnsupported,
                   "cross-event side effects are unsupported");
  if (options.do_bhabhas)
    return failure(StatusCode::BhabhasUnsupported,
                   "Bhabha generation and RNG side effects are unsupported");
  if (options.do_jets)
    return failure(StatusCode::JetsUnsupported,
                   "jet generation and RNG side effects are unsupported");
  if (options.do_prod_hook)
    return failure(StatusCode::ProductionHookUnsupported,
                   "the unimplemented production hook is unsupported");
  if (options.bmt_spin && !options.spin_records_complete)
    return failure(StatusCode::MissingSpin,
                   "BMT mode requires complete particle spin records");
  return Status::success();
}

Status validate_problem_size(std::size_t beam1_particles,
                             std::size_t beam2_particles,
                             std::size_t spectrum_bins,
                             const KernelLimits& limits) {
  if (beam1_particles > limits.max_particles_per_beam ||
      beam2_particles > limits.max_particles_per_beam)
    return failure(StatusCode::ParticleLimitExceeded,
                   "particle count exceeds configured per-beam limit");
  if (spectrum_bins > limits.max_spectrum_bins)
    return failure(StatusCode::SpectrumBinLimitExceeded,
                   "spectrum bin count exceeds configured limit");
  if (beam1_particles != 0 &&
      beam2_particles > std::numeric_limits<std::size_t>::max() /
                            beam1_particles)
    return failure(StatusCode::PairCountOverflow,
                   "Cartesian pair count overflows size_t");
  const std::size_t pairs = beam1_particles * beam2_particles;
  if (pairs > limits.max_cartesian_pairs)
    return failure(StatusCode::PairLimitExceeded,
                   "Cartesian pair count exceeds configured limit");
  return Status::success();
}

PrecomputeRun precompute_beam(const std::vector<ParticleSample>& particles,
                              bool beam2_direction,
                              const KernelConfig& config) {
  PrecomputeRun run;
  const Status eligible = check_eligibility(config.eligibility);
  if (!eligible) {
    run.status = eligible;
    return run;
  }
  run.status = validate_legacy_mass(config.electron_mass);
  if (!run.status) {
    return run;
  }
  if (particles.size() > config.limits.max_particles_per_beam) {
    run.status = failure(StatusCode::ParticleLimitExceeded,
                         "particle count exceeds configured per-beam limit");
    return run;
  }
  if (config.eligibility.bmt_spin) {
    for (const ParticleSample& particle : particles) {
      if (!particle.has_spin) {
        run.status = failure(StatusCode::MissingSpin,
                             "BMT mode requires spin on every particle");
        return run;
      }
    }
  }

  try {
    run.beam.reserve(particles.size());
    for (const ParticleSample& source : particles) {
      PrecomputedParticle particle;
      particle.signed_energy = source.signed_energy;
      ++run.operations.particle_energy_reads;
      particle.absolute_energy = std::fabs(particle.signed_energy);
      ++run.operations.absolute_value_calls;
      particle.vx = source.vx;
      particle.vy = source.vy;
      ++run.operations.particle_velocity_reads;
      particle.legacy_vz = legacy_longitudinal_velocity(
          particle.vx, particle.vy, run.operations, true);
      particle.massive_momentum = legacy_massive_momentum(
          static_cast<double>(particle.signed_energy), config.electron_mass,
          run.operations, true);
      particle.charge_class = static_cast<std::uint8_t>(
          beam2_direction ? (particle.signed_energy < 0.0F ? 0 : 1)
                          : (particle.signed_energy < 0.0F ? 1 : 0));
      particle.weight = source.weight;
      ++run.operations.particle_weight_reads;
      particle.has_spin = source.has_spin;
      if (config.eligibility.bmt_spin) {
        ++run.operations.particle_spin_reads;
        particle.legacy_spin_z = static_cast<float>(source.spin_z);
        const double q = static_cast<double>(particle.vx) * particle.vx +
                         static_cast<double>(particle.vy) * particle.vy;
        if (q < 1.0) {
          ++run.operations.precompute_sqrt_calls;
          const double vz_magnitude = std::sqrt(1.0 - q);
          const double vz = beam2_direction ? -vz_magnitude : vz_magnitude;
          particle.helicity = source.spin_x * static_cast<double>(particle.vx) +
                              source.spin_y * static_cast<double>(particle.vy) +
                              source.spin_z * vz;
          particle.helicity_valid = true;
        }
      }
      run.beam.push_back(particle);
      ++run.operations.precomputed_particles;
    }
  } catch (const std::bad_alloc&) {
    run.beam.clear();
    run.status = failure(StatusCode::AllocationFailure,
                         "allocation failed while precomputing particle cache");
    return run;
  } catch (const std::length_error&) {
    run.beam.clear();
    run.status = failure(StatusCode::AllocationFailure,
                         "invalid vector length while precomputing particle cache");
    return run;
  }
  run.status = Status::success();
  return run;
}

KernelRun run_cartesian_reference_seeded(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config,
    const AggregateResults& initial_results) {
  KernelRun run;
  run.status = validate_inputs(beam1, beam2, config);
  if (!run.status) return run;
  run.status = validate_accumulator(initial_results, config.spectrum);
  if (!run.status) return run;
  run.status = copy_seeded_results(run.results, initial_results);
  if (!run.status) return run;
  inspect_inputs(run.results, beam1, beam2);

  for (const ParticleSample& pointer1 : beam1) {
    for (const ParticleSample& pointer2 : beam2) {
      ++run.operations.cartesian_pairs;
      ++run.operations.particle_weight_reads;
      ++run.operations.particle_weight_reads;
      const float weight = pointer1.weight * pointer2.weight;
      ++run.operations.particle_energy_reads;
      ++run.operations.particle_energy_reads;
      ++run.operations.absolute_value_calls;
      ++run.operations.absolute_value_calls;
      const float e1 = std::fabs(pointer1.signed_energy);
      const float e2 = std::fabs(pointer2.signed_energy);
      ++run.operations.particle_velocity_reads;
      ++run.operations.particle_velocity_reads;
      const float vx1 = pointer1.vx;
      const float vy1 = pointer1.vy;
      const float vx2 = pointer2.vx;
      const float vy2 = pointer2.vy;
      run.operations.pair_sqrt_calls += 2;
      const float vz1 = static_cast<float>(
          std::sqrt(1. - vx1 * vx1 - vy1 * vy1));
      const float vz2 = static_cast<float>(
          std::sqrt(1. - vx2 * vx2 - vy2 * vy2));
      const float cosinus = vx1 * vx2 + vy1 * vy2 - vz1 * vz2;
      ++run.operations.pair_sqrt_calls;
      const float ecm = static_cast<float>(
          std::sqrt(2.0 * e1 * e2 * (1. - cosinus)));
      ++run.operations.particle_energy_reads;
      ++run.operations.particle_energy_reads;
      const float energy1 = pointer1.signed_energy;
      const float energy2 = pointer2.signed_energy;
      const std::size_t class1 = energy1 < 0.0F ? 1U : 0U;
      const std::size_t class2 = energy2 < 0.0F ? 0U : 1U;
      run.results.class_luminosity[3U * class1 + class2] += weight;

      if (energy1 * energy2 > 0.0F) {
        ++run.operations.qualifying_ee_pairs;
        run.results.lumi_ee += weight;
        run.results.step_lumi_ee += weight;
        double momentum1 = 0.0;
        double momentum2 = 0.0;
        if (static_cast<double>(energy1) > config.electron_mass) {
          ++run.operations.pair_sqrt_calls;
          momentum1 = std::sqrt(
              (static_cast<double>(energy1) - config.electron_mass) *
              (static_cast<double>(energy1) + config.electron_mass));
        }
        if (static_cast<double>(energy2) > config.electron_mass) {
          ++run.operations.pair_sqrt_calls;
          momentum2 = std::sqrt(
              (static_cast<double>(energy2) - config.electron_mass) *
              (static_cast<double>(energy2) + config.electron_mass));
        }
        const double massive_cosinus =
            static_cast<double>(vx1) * static_cast<double>(vx2) +
            static_cast<double>(vy1) * static_cast<double>(vy2) +
            static_cast<double>(vz1) * -static_cast<double>(vz2);
        const double w2 = 2.0 * config.electron_mass * config.electron_mass +
                          2.0 * (static_cast<double>(energy1) *
                                     static_cast<double>(energy2) -
                                 momentum1 * momentum2 * massive_cosinus);
        double invariant_mass = 0.0;
        if (w2 > 0.0) {
          ++run.operations.pair_sqrt_calls;
          invariant_mass = std::sqrt(w2);
        }

        if (config.eligibility.bmt_spin) {
          ++run.operations.particle_spin_reads;
          ++run.operations.particle_spin_reads;
          const float spin1 = static_cast<float>(pointer1.spin_z);
          const float spin2 = static_cast<float>(pointer2.spin_z);
          ++run.operations.particle_velocity_reads;
          ++run.operations.particle_velocity_reads;
          const float w1x = pointer1.vx;
          const float w1y = pointer1.vy;
          const float w2x = pointer2.vx;
          const float w2y = pointer2.vy;
          const double q1 = static_cast<double>(w1x) * w1x +
                            static_cast<double>(w1y) * w1y;
          const double q2 = static_cast<double>(w2x) * w2x +
                            static_cast<double>(w2y) * w2y;
          bool helicity_valid = false;
          double helicity1 = 0.0;
          double helicity2 = 0.0;
          if (q1 < 1.0 && q2 < 1.0) {
            run.operations.pair_sqrt_calls += 2;
            const double spin_vz1 = std::sqrt(1.0 - q1);
            const double spin_vz2 = -std::sqrt(1.0 - q2);
            ++run.operations.particle_spin_reads;
            ++run.operations.particle_spin_reads;
            helicity1 = pointer1.spin_x * static_cast<double>(w1x) +
                        pointer1.spin_y * static_cast<double>(w1y) +
                        pointer1.spin_z * spin_vz1;
            helicity2 = pointer2.spin_x * static_cast<double>(w2x) +
                        pointer2.spin_y * static_cast<double>(w2y) +
                        pointer2.spin_z * spin_vz2;
            helicity_valid = true;
          }
          float help = static_cast<float>(
              0.5 * (1.0 + spin1 * spin2));
          run.results.lumi_pp += weight * help;
          if (helicity_valid) {
            const double double_weight = static_cast<double>(weight);
            run.results.lumi_weight_sum += double_weight;
            run.results.lumi_weighted_helicity1 +=
                double_weight * helicity1;
            run.results.lumi_weighted_helicity2 +=
                double_weight * helicity2;
            run.results.spectrum.add(double_weight, invariant_mass,
                                     helicity1, helicity2);
          } else {
            run.results.spectrum.add(static_cast<double>(weight),
                                     invariant_mass);
          }
        } else {
          run.results.spectrum.add(static_cast<double>(weight),
                                   invariant_mass);
        }
        if (ecm > config.ecm_min) {
          run.results.lumi_ee_high += weight;
          run.results.step_lumi_ee_high += weight;
        }
        const float help = weight * ecm;
        run.results.lumi_ecm += help;
        run.results.lumi_ecm2 += help * ecm;
      }
    }
  }
  return run;
}

KernelRun run_cartesian_reference(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config) {
  KernelRun run;
  run.status = validate_inputs(beam1, beam2, config);
  if (!run.status) return run;
  AggregateResults initial_results;
  run.status = make_zero_results(initial_results, config);
  if (!run.status) return run;
  return run_cartesian_reference_seeded(
      beam1, beam2, config, initial_results);
}

KernelRun run_precomputed_candidate_seeded(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config,
    const AggregateResults& initial_results) {
  KernelRun run;
  run.status = validate_inputs(beam1, beam2, config);
  if (!run.status) return run;
  run.status = validate_accumulator(initial_results, config.spectrum);
  if (!run.status) return run;
  run.status = copy_seeded_results(run.results, initial_results);
  if (!run.status) return run;
  inspect_inputs(run.results, beam1, beam2);
  PrecomputeRun cached1 = precompute_beam(beam1, false, config);
  if (!cached1.status) {
    run.status = cached1.status;
    return run;
  }
  PrecomputeRun cached2 = precompute_beam(beam2, true, config);
  if (!cached2.status) {
    run.status = cached2.status;
    return run;
  }
  run.operations = cached1.operations;
  run.operations.particle_energy_reads += cached2.operations.particle_energy_reads;
  run.operations.particle_velocity_reads += cached2.operations.particle_velocity_reads;
  run.operations.particle_weight_reads += cached2.operations.particle_weight_reads;
  run.operations.particle_spin_reads += cached2.operations.particle_spin_reads;
  run.operations.absolute_value_calls += cached2.operations.absolute_value_calls;
  run.operations.precompute_sqrt_calls += cached2.operations.precompute_sqrt_calls;
  run.operations.precomputed_particles += cached2.operations.precomputed_particles;

  for (const PrecomputedParticle& particle1 : cached1.beam) {
    for (const PrecomputedParticle& particle2 : cached2.beam) {
      ++run.operations.cartesian_pairs;
      const float weight = particle1.weight * particle2.weight;
      const float ecm = legacy_ecm(
          particle1.absolute_energy, particle2.absolute_energy,
          particle1.vx, particle1.vy, particle1.legacy_vz,
          particle2.vx, particle2.vy, particle2.legacy_vz,
          run.operations);
      run.results.class_luminosity[
          3U * particle1.charge_class + particle2.charge_class] += weight;

      if (particle1.signed_energy * particle2.signed_energy > 0.0F) {
        ++run.operations.qualifying_ee_pairs;
        run.results.lumi_ee += weight;
        run.results.step_lumi_ee += weight;
        const double invariant_mass = finish_invariant_mass(
            static_cast<double>(particle1.signed_energy),
            static_cast<double>(particle1.vx),
            static_cast<double>(particle1.vy),
            static_cast<double>(particle1.legacy_vz),
            particle1.massive_momentum,
            static_cast<double>(particle2.signed_energy),
            static_cast<double>(particle2.vx),
            static_cast<double>(particle2.vy),
            -static_cast<double>(particle2.legacy_vz),
            particle2.massive_momentum,
            config.electron_mass, run.operations);
        if (config.eligibility.bmt_spin) {
          add_spin_and_spectrum(
              run.results, weight, invariant_mass,
              particle1.legacy_spin_z, particle2.legacy_spin_z,
              particle1.helicity_valid && particle2.helicity_valid,
              particle1.helicity, particle2.helicity);
        } else {
          add_nonspin_spectrum(run.results, weight, invariant_mass);
        }
        add_common_pair_results(run.results, weight, ecm, config.ecm_min);
      }
    }
  }
  return run;
}

KernelRun run_precomputed_candidate(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config) {
  KernelRun run;
  run.status = validate_inputs(beam1, beam2, config);
  if (!run.status) return run;
  AggregateResults initial_results;
  run.status = make_zero_results(initial_results, config);
  if (!run.status) return run;
  return run_precomputed_candidate_seeded(
      beam1, beam2, config, initial_results);
}

}  // namespace legacy_ee
}  // namespace modern
}  // namespace feebl
