#ifndef FEEBL_LEGACY_ELECTRON_LUMI_PRECOMPUTE_CPP_H
#define FEEBL_LEGACY_ELECTRON_LUMI_PRECOMPUTE_CPP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace modern {
namespace legacy_ee {

constexpr double kLegacyElectronMassGeV = 0.51099906e-3;

enum class StatusCode : std::uint8_t {
  Ok = 0,
  IsrUnsupported,
  EeLumiHeapUnsupported,
  CrossEventsUnsupported,
  BhabhasUnsupported,
  JetsUnsupported,
  ProductionHookUnsupported,
  MissingSpin,
  InvalidElectronMass,
  InvalidSpectrumConfig,
  AllocationFailure,
  AccumulatorConfigurationMismatch,
  ParticleLimitExceeded,
  PairCountOverflow,
  PairLimitExceeded,
  SpectrumBinLimitExceeded
};

struct Status {
  StatusCode code = StatusCode::Ok;
  std::string message;

  bool ok() const noexcept { return code == StatusCode::Ok; }
  explicit operator bool() const noexcept { return ok(); }
  static Status success() { return Status{}; }
};

// These gates are deliberately stricter than collide_ee itself. The pure
// kernel has no RNG, heap sampling, event generator, or production-hook side
// effects, so a future dispatcher must reject every such mode before entering
// this path. BMT-derived aggregate spin/helicity quantities are supported.
struct EligibilityOptions {
  bool do_isr = false;
  bool do_ee_lumi_heap = false;
  bool do_cross_events = false;
  bool do_bhabhas = false;
  bool do_jets = false;
  bool do_prod_hook = false;
  bool bmt_spin = false;
  bool spin_records_complete = true;
};

struct KernelLimits {
  std::uint64_t max_particles_per_beam = 10000000ULL;
  std::uint64_t max_cartesian_pairs = 100000000ULL;
  std::uint64_t max_spectrum_bins = 10000000ULL;
};

// A caller cannot relax this allocation ceiling.  It is intentionally
// independent of KernelLimits::max_spectrum_bins so a widened input-card limit
// cannot turn malformed input into an unbounded allocation attempt.
constexpr std::uint64_t kAbsoluteSpectrumStorageBytes =
    256ULL * 1024ULL * 1024ULL;

struct SpectrumConfig {
  double low = 0.0;
  double high = 0.0;
  std::size_t bins = 0;
};

struct KernelConfig {
  EligibilityOptions eligibility;
  KernelLimits limits;
  SpectrumConfig spectrum;
  float ecm_min = 0.0F;
  double electron_mass = kLegacyElectronMassGeV;
};

// A dependency-free materialized view of the particle fields read by the
// eligible subset of GRID::collide_ee. Beam direction remains a property of
// which input vector contains the record, exactly as in the legacy routine.
struct ParticleSample {
  float signed_energy = 0.0F;
  float vx = 0.0F;
  float vy = 0.0F;
  float weight = 0.0F;
  bool has_spin = false;
  double spin_x = 0.0;
  double spin_y = 0.0;
  double spin_z = 0.0;
};

// Candidate-owned particle invariants. legacy_vz is intentionally float and
// massive_momentum intentionally double to reproduce the two legacy formulas.
struct PrecomputedParticle {
  float signed_energy = 0.0F;
  float absolute_energy = 0.0F;
  float vx = 0.0F;
  float vy = 0.0F;
  float legacy_vz = 0.0F;  // positive magnitude; beam-2 sign is applied per use
  double massive_momentum = 0.0;
  std::uint8_t charge_class = 0;
  float weight = 0.0F;
  bool has_spin = false;
  float legacy_spin_z = 0.0F;
  bool helicity_valid = false;
  double helicity = 0.0;
};

struct OperationCounts {
  std::uint64_t cartesian_pairs = 0;
  std::uint64_t qualifying_ee_pairs = 0;
  std::uint64_t particle_energy_reads = 0;
  std::uint64_t particle_velocity_reads = 0;
  std::uint64_t particle_weight_reads = 0;
  std::uint64_t particle_spin_reads = 0;
  std::uint64_t absolute_value_calls = 0;
  std::uint64_t precompute_sqrt_calls = 0;
  std::uint64_t pair_sqrt_calls = 0;
  std::uint64_t precomputed_particles = 0;
};

// Raw accumulators are exposed so an ablation test can compare every update at
// the bit level, including quantities normally visible only through derived
// means/errors in pairMomentsCPP.h.
struct Moment1Raw {
  double weight = 0.0;
  double weight2 = 0.0;
  double weighted_x = 0.0;
  double weighted_x2 = 0.0;

  void add(double w, double x) noexcept;
};

struct PairHelicityRaw {
  Moment1Raw h1;
  Moment1Raw h2;
  double weight = 0.0;
  double weighted_product = 0.0;

  void add(double w, double first, double second) noexcept;
};

struct SpectrumRaw {
  double low = 0.0;
  double high = 0.0;
  double width = 0.0;
  std::vector<double> bin_weight;
  std::vector<PairHelicityRaw> bin_helicity;
  Moment1Raw w_moment;
  double underflow = 0.0;
  double overflow = 0.0;

  Status configure(const SpectrumConfig& config);
  bool configured() const noexcept;
  void add(double weight, double invariant_mass);
  void add(double weight, double invariant_mass,
           double helicity1, double helicity2);
};

struct AggregateResults {
  std::array<double, 9> class_luminosity{};
  double lumi_ee = 0.0;
  double step_lumi_ee = 0.0;
  double lumi_ee_high = 0.0;
  double step_lumi_ee_high = 0.0;
  double lumi_pp = 0.0;
  double lumi_weight_sum = 0.0;
  double lumi_weighted_helicity1 = 0.0;
  double lumi_weighted_helicity2 = 0.0;
  double lumi_ecm = 0.0;
  double lumi_ecm2 = 0.0;
  SpectrumRaw spectrum;
  bool invalid_velocity_observed = false;
  bool nonfinite_input_observed = false;
};

struct KernelRun {
  Status status;
  AggregateResults results;
  OperationCounts operations;
};

struct PrecomputeRun {
  Status status;
  std::vector<PrecomputedParticle> beam;
  OperationCounts operations;
};

Status check_eligibility(const EligibilityOptions& options);
Status validate_problem_size(std::size_t beam1_particles,
                             std::size_t beam2_particles,
                             std::size_t spectrum_bins,
                             const KernelLimits& limits);

PrecomputeRun precompute_beam(const std::vector<ParticleSample>& particles,
                              bool beam2_direction,
                              const KernelConfig& config);

// Direct Cartesian transcription of the eligible, deterministic subset of
// GRID::collide_ee. It deliberately repeats particle-owned work for each pair
// and is the executable legacy oracle for this ablation.
KernelRun run_cartesian_reference(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config);

// Continue directly from a previously accumulated state.  This is required
// to preserve legacy pair-by-pair floating-point update order across cells;
// merging independently zero-seeded per-cell totals is not equivalent.
KernelRun run_cartesian_reference_seeded(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config,
    const AggregateResults& initial_results);

// Same pair order and update order, with immutable per-particle invariants
// materialized before the Cartesian loop. No production dispatcher calls this
// function yet.
KernelRun run_precomputed_candidate(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config);

KernelRun run_precomputed_candidate_seeded(
    const std::vector<ParticleSample>& beam1,
    const std::vector<ParticleSample>& beam2,
    const KernelConfig& config,
    const AggregateResults& initial_results);

}  // namespace legacy_ee
}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_LEGACY_ELECTRON_LUMI_PRECOMPUTE_CPP_H
