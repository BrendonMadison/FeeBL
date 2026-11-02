#include "wavefrontFieldWorkspaceCPP.h"
#include "adaptiveTerminalPatchCPP.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void checked_add_counter(std::uint64_t& counter, std::size_t increment,
                         const char* label) {
  if ((sizeof(std::size_t) > sizeof(std::uint64_t) &&
       increment > static_cast<std::size_t>(
                       std::numeric_limits<std::uint64_t>::max())) ||
      counter > std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(increment))
    throw std::overflow_error(std::string(label) + " counter overflow");
  counter += static_cast<std::uint64_t>(increment);
}

std::size_t checked_product(std::size_t first, std::size_t second,
                            const char* label) {
  if (first != 0U && second >
          std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error(std::string(label) + " product overflow");
  return first * second;
}

std::size_t checked_sum(std::size_t first, std::size_t second,
                        const char* label) {
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error(std::string(label) + " sum overflow");
  return first + second;
}

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

void validate_resolution_factor_sequence(
    const feebl::wavefront_fields::Configuration& configuration) {
  const std::vector<std::size_t>& factors =
      configuration.adaptive_spacing_factors;
  if (factors.size() < 3U || factors.front() != 1U)
    throw std::invalid_argument(
        "Mode 2/3 adaptive spacing factors must begin with 1 and contain "
        "at least three classes");
  for (std::size_t index = 0U; index < factors.size(); ++index) {
    if (!is_power_of_two(factors[index]) ||
        (index != 0U && factors[index] <= factors[index - 1U]))
      throw std::invalid_argument(
          "Mode 2/3 adaptive spacing factors must be strictly increasing "
          "powers of two");
    if (index != 0U &&
        factors[index] != checked_product(
            factors[index - 1U], configuration.adaptive_refinement_ratio,
            "adaptive adjacent spacing factor"))
      throw std::invalid_argument(
          "Mode 2/3 adaptive spacing factors must be consecutive ratio-two "
          "classes");
  }
}

std::size_t adaptive_context_estimate(
    std::size_t n_x, std::size_t n_y,
    const feebl::adaptive_grid::UniformWindowPolicy& policy) {
  const std::size_t local_cells = checked_product(
      n_x, n_y, "adaptive context estimate cells");
  const std::size_t padded_cells = checked_product(
      checked_product(n_x, 2U, "adaptive context estimate padded x"),
      checked_product(n_y, 2U, "adaptive context estimate padded y"),
      "adaptive context estimate padded cells");
  return checked_sum(
      checked_sum(
          checked_product(local_cells, policy.bytes_per_local_cell,
                          "adaptive context estimate local bytes"),
          checked_product(padded_cells, policy.bytes_per_padded_cell,
                          "adaptive context estimate padded bytes"),
          "adaptive context estimate arrays"),
      policy.fixed_bytes, "adaptive context estimate");
}

int checked_int_dimension(std::size_t value, std::size_t other,
                          const char* label) {
  const std::size_t int_max = static_cast<std::size_t>(
      std::numeric_limits<int>::max());
  // Legacy FIELD stores 2*n, n_x*n_y and 8*n_x*n_y in signed int
  // temporaries.  Validate that complete arithmetic contract before its
  // constructor can evaluate any of those expressions.
  if (value < 2U || other < 2U || value > int_max / 2U ||
      value > int_max / other ||
      checked_product(value, other, "FIELD cell") > int_max / 8U)
    throw std::overflow_error(std::string(label) +
                               " exceeds FIELD signed-integer products");
  return static_cast<int>(value);
}

int checked_int_index(std::size_t value, const char* label) {
  if (value > static_cast<std::size_t>(
                  std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) +
                              " does not fit an integer index");
  return static_cast<int>(value);
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (unsigned int byte = 0; byte < 8U; ++byte) {
    hash ^= (value >> (8U * byte)) & 0xffU;
    hash *= UINT64_C(1099511628211);
  }
}

void hash_float_bits(std::uint64_t& hash, float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value),
                "unexpected FeeBL float representation");
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, static_cast<std::uint64_t>(bits));
}

void hash_double_bits(std::uint64_t& hash, double value) noexcept {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value),
                "unexpected FeeBL double representation");
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

void hash_field_geometry(
    std::uint64_t& hash,
    const feebl::adaptive_grid::FieldLevelGeometry& geometry) noexcept {
  hash_u64(hash, static_cast<std::uint64_t>(geometry.refinement_factor));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.global_n_x));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.global_n_y));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.origin_x));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.origin_y));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.n_x));
  hash_u64(hash, static_cast<std::uint64_t>(geometry.n_y));
  hash_float_bits(hash, geometry.delta_x);
  hash_float_bits(hash, geometry.delta_y);
  hash_float_bits(hash, geometry.offset_x);
  hash_float_bits(hash, geometry.offset_y);
}

feebl::adaptive_grid::FieldLevelGeometry make_full_registered_geometry(
    const feebl::adaptive_grid::CanonicalLatticeSnapshot& lattice,
    std::size_t factor) {
  if (factor == 0U || !is_power_of_two(factor))
    throw std::invalid_argument(
        "reference CIC geometry requires a power-of-two factor");
  feebl::adaptive_grid::FieldLevelGeometry geometry;
  geometry.refinement_factor = factor;
  geometry.global_n_x = checked_product(
      lattice.n_x(), factor, "reference CIC global n_x");
  geometry.global_n_y = checked_product(
      lattice.n_y(), factor, "reference CIC global n_y");
  geometry.origin_x = 0;
  geometry.origin_y = 0;
  geometry.n_x = geometry.global_n_x;
  geometry.n_y = geometry.global_n_y;
  geometry.delta_x = lattice.delta_x() / static_cast<float>(factor);
  geometry.delta_y = lattice.delta_y() / static_cast<float>(factor);
  geometry.inverse_delta_x =
      lattice.inverse_delta_x() * static_cast<float>(factor);
  geometry.inverse_delta_y =
      lattice.inverse_delta_y() * static_cast<float>(factor);
  geometry.offset_x = lattice.offset_x() * static_cast<float>(factor);
  geometry.offset_y = lattice.offset_y() * static_cast<float>(factor);
  if (!geometry.valid())
    throw std::invalid_argument(
        "constructed full registered reference CIC geometry is invalid");
  return geometry;
}

feebl::adaptive_grid::FieldLevelGeometry align_geometry_to_source_cells(
    feebl::adaptive_grid::FieldLevelGeometry geometry,
    std::size_t source_factor) {
  if (geometry.refinement_factor <= source_factor) return geometry;
  if (source_factor == 0U ||
      geometry.refinement_factor % source_factor != 0U)
    throw std::invalid_argument(
        "field geometry is not registered with the reference source");
  const std::size_t ratio = geometry.refinement_factor / source_factor;
  const std::size_t begin_x = static_cast<std::size_t>(geometry.origin_x);
  const std::size_t begin_y = static_cast<std::size_t>(geometry.origin_y);
  const std::size_t end_x = checked_sum(
      begin_x, geometry.n_x, "reference-aligned field end x");
  const std::size_t end_y = checked_sum(
      begin_y, geometry.n_y, "reference-aligned field end y");
  const std::size_t aligned_begin_x = begin_x - begin_x % ratio;
  const std::size_t aligned_begin_y = begin_y - begin_y % ratio;
  const std::size_t aligned_end_x = std::min(
      geometry.global_n_x,
      checked_product((checked_sum(end_x, ratio - 1U,
                                   "reference-aligned field round x") / ratio),
                      ratio, "reference-aligned field end x"));
  const std::size_t aligned_end_y = std::min(
      geometry.global_n_y,
      checked_product((checked_sum(end_y, ratio - 1U,
                                   "reference-aligned field round y") / ratio),
                      ratio, "reference-aligned field end y"));
  geometry.origin_x = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      aligned_begin_x);
  geometry.origin_y = static_cast<feebl::adaptive_grid::CanonicalIndex>(
      aligned_begin_y);
  geometry.n_x = aligned_end_x - aligned_begin_x;
  geometry.n_y = aligned_end_y - aligned_begin_y;
  if (!geometry.valid() || geometry.n_x % ratio != 0U ||
      geometry.n_y % ratio != 0U)
    throw std::runtime_error(
        "reference-aligned field geometry is invalid");
  return geometry;
}

std::size_t fixed_width_support_guard(double half_width, float delta,
                                      const char* label) {
  if (!std::isfinite(half_width) || !(half_width > 0.0) ||
      !std::isfinite(delta) || !(delta > 0.0f))
    throw std::invalid_argument(std::string(label) +
                                " has invalid width or spacing");
  const double cells = std::ceil(half_width / static_cast<double>(delta));
  if (!std::isfinite(cells) || cells < 0.0 ||
      cells > static_cast<double>(
                  std::numeric_limits<std::size_t>::max() - 1U))
    throw std::overflow_error(std::string(label) +
                              " support guard overflows size_t");
  return checked_sum(static_cast<std::size_t>(cells), 1U, label);
}

feebl::adaptive_grid::FieldLevelGeometry expand_geometry_for_fixed_width(
    feebl::adaptive_grid::FieldLevelGeometry geometry,
    double half_width_x, double half_width_y) {
  if (!geometry.valid())
    throw std::invalid_argument(
        "cannot expand an invalid fixed-width source geometry");
  const std::size_t guard_x = fixed_width_support_guard(
      half_width_x, geometry.delta_x, "fixed-width x guard");
  const std::size_t guard_y = fixed_width_support_guard(
      half_width_y, geometry.delta_y, "fixed-width y guard");
  const auto expand_axis = [](std::size_t origin, std::size_t extent,
                              std::size_t global, std::size_t guard,
                              std::size_t& expanded_origin,
                              std::size_t& expanded_extent,
                              const char* label) {
    const std::size_t end = checked_sum(origin, extent, label);
    expanded_origin = origin > guard ? origin - guard : 0U;
    std::size_t expanded_end = std::min(
        global, checked_sum(end, guard, label));
    if (((expanded_end - expanded_origin) & 1U) != 0U) {
      if (expanded_end < global)
        ++expanded_end;
      else if (expanded_origin > 0U)
        --expanded_origin;
    }
    expanded_extent = expanded_end - expanded_origin;
  };
  std::size_t origin_x = 0U;
  std::size_t origin_y = 0U;
  expand_axis(
      static_cast<std::size_t>(geometry.origin_x), geometry.n_x,
      geometry.global_n_x, guard_x, origin_x, geometry.n_x,
      "fixed-width expanded x end");
  expand_axis(
      static_cast<std::size_t>(geometry.origin_y), geometry.n_y,
      geometry.global_n_y, guard_y, origin_y, geometry.n_y,
      "fixed-width expanded y end");
  geometry.origin_x =
      static_cast<feebl::adaptive_grid::CanonicalIndex>(origin_x);
  geometry.origin_y =
      static_cast<feebl::adaptive_grid::CanonicalIndex>(origin_y);
  if (!geometry.valid() || (geometry.n_x & 1U) != 0U ||
      (geometry.n_y & 1U) != 0U)
    throw std::runtime_error(
        "fixed-width support expansion produced invalid geometry");
  return geometry;
}

std::uint64_t field_binding_hash(
    int slice1, int slice2,
    const std::vector<PARTICLE*>& particles1,
    const std::vector<PARTICLE*>& particles2,
    std::uint64_t lattice_hash, std::uint64_t plan_hash,
    std::uint64_t result_geometry_hash, std::uint64_t source_model_tag,
    std::size_t source_reference_factor,
    std::uint64_t source_density_hash,
    std::uint64_t source_sampling_hash,
    std::uint64_t source_sampling_hierarchy_hash) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash_u64(hash, static_cast<std::uint64_t>(
                     static_cast<std::uint32_t>(slice1)));
  hash_u64(hash, static_cast<std::uint64_t>(
                     static_cast<std::uint32_t>(slice2)));
  hash_u64(hash, static_cast<std::uint64_t>(particles1.size()));
  hash_u64(hash, static_cast<std::uint64_t>(particles2.size()));
  // Binding is crossing-local and never serialized.  Hash the complete
  // pointer/order plus transverse-coordinate snapshot so a plan or field
  // result cannot be consumed after an in-place particle kick.  Mode 2/3
  // already traverse every particle to build immutable descriptors, making
  // this O(N) fail-closed check a small and deterministic part of planning.
  hash_u64(hash, static_cast<std::uint64_t>(
                     reinterpret_cast<std::uintptr_t>(particles1.data())));
  hash_u64(hash, static_cast<std::uint64_t>(
                     reinterpret_cast<std::uintptr_t>(particles2.data())));
  const auto add_particles = [&hash](
      const std::vector<PARTICLE*>& particles) noexcept {
    for (const PARTICLE* particle : particles) {
      hash_u64(hash, static_cast<std::uint64_t>(
                         reinterpret_cast<std::uintptr_t>(particle)));
      if (!particle) {
        hash_u64(hash, UINT64_C(0x4e554c4c50415254));  // "NULLPART"
        continue;
      }
      float x = 0.0f;
      float y = 0.0f;
      particle->XYposition(x, y);
      hash_float_bits(hash, x);
      hash_float_bits(hash, y);
    }
  };
  add_particles(particles1);
  add_particles(particles2);
  hash_u64(hash, lattice_hash);
  hash_u64(hash, plan_hash);
  hash_u64(hash, result_geometry_hash);
  hash_u64(hash, source_model_tag);
  hash_u64(hash, static_cast<std::uint64_t>(source_reference_factor));
  hash_u64(hash, source_density_hash);
  hash_u64(hash, source_sampling_hash);
  hash_u64(hash, source_sampling_hierarchy_hash);
  return hash;
}

std::uint64_t adaptive_result_geometry_hash(
    int mode, const feebl::adaptive_grid::FieldLevelGeometry& selected,
    const feebl::adaptive_grid::FieldLevelGeometry* nested,
    feebl::adaptive_grid::FieldOperator field_operator) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash_u64(hash, UINT64_C(0x464545424c47524d));  // "FEEBLGRM"
  hash_u64(hash, static_cast<std::uint64_t>(mode));
  // A result produced by the legacy and higher-order gather must never share
  // a binding identity even when every geometric quantity is identical.
  hash_u64(hash, static_cast<std::uint64_t>(field_operator));
  hash_field_geometry(hash, selected);
  hash_u64(hash, nested ? 1U : 0U);
  if (nested) hash_field_geometry(hash, *nested);
  return hash;
}

}  // namespace

namespace feebl {
namespace wavefront_fields {

struct Workspace::AdaptiveFieldContext {
  AdaptiveFieldContext(std::size_t requested_n_x,
                       std::size_t requested_n_y,
                       PHI_FLOAT kernel_factor,
                       float delta_x, float delta_y,
                       bool wisdom_only,
                       bool force_estimate,
                       std::size_t requested_refinement_factor)
      : n_x(requested_n_x), n_y(requested_n_y),
        refinement_factor(requested_refinement_factor),
        fourier(wisdom_only, force_estimate),
        field(2, checked_int_dimension(requested_n_x, requested_n_y,
                                       "adaptive n_x"),
              checked_int_dimension(requested_n_y, requested_n_x,
                                       "adaptive n_y"), &fourier),
        rho1(checked_product(requested_n_x, requested_n_y,
                             "adaptive rho"), PHI_FLOAT(0.0)),
        rho2(rho1.size(), PHI_FLOAT(0.0)) {
    field.dist_init(kernel_factor, delta_x, delta_y, &fourier);
  }

  std::size_t n_x;
  std::size_t n_y;
  std::size_t refinement_factor;
  FFT_SERVER fourier;
  FIELD field;
  std::vector<PHI_FLOAT> rho1;
  std::vector<PHI_FLOAT> rho2;
  std::size_t estimated_bytes = 0U;
};

// Encounter-local and constructed only by Modes 2/3.  The canonical CIC
// deposition is immutable after construction; every uniform resolution class
// is a conservative subdivision of these same control-volume charges.  The
// parallel S1/S2 arrays carry particle-sampling statistics through the same
// hierarchy without performing a resolution-dependent CIC re-deposition.
// Keeping this
// object on the Mode-2/3 stack preserves the established Mode-0/1 workspace
// object layout and allocation behavior.
struct SamplingHierarchyLevel {
  std::size_t factor = 1U;
  adaptive_grid::FieldLevelGeometry geometry;
  std::vector<PHI_FLOAT> s1_1;
  std::vector<PHI_FLOAT> s2_1;
  std::vector<PHI_FLOAT> s1_2;
  std::vector<PHI_FLOAT> s2_2;
};

struct Workspace::CoherentSourceHierarchy {
  std::string model = "canonical_cell_average_v1";
  std::size_t source_factor = 1U;
  double source_width_x = 0.0;
  double source_width_y = 0.0;
  adaptive_grid::FieldLevelGeometry canonical_geometry;
  std::vector<PHI_FLOAT> rho1;
  std::vector<PHI_FLOAT> rho2;
  std::vector<PHI_FLOAT> sampling_s1_1;
  std::vector<PHI_FLOAT> sampling_s2_1;
  std::vector<PHI_FLOAT> sampling_s1_2;
  std::vector<PHI_FLOAT> sampling_s2_2;
  std::vector<SamplingHierarchyLevel> sampling_levels;
  std::uint64_t density_hash = 0U;
  std::uint64_t sampling_hash = 0U;
  std::uint64_t sampling_hierarchy_hash = 0U;
  std::size_t beam1_in = 0U;
  std::size_t beam1_out = 0U;
  std::size_t beam2_in = 0U;
  std::size_t beam2_out = 0U;
  long double accepted_charge1 = 0.0L;
  long double accepted_x_moment1 = 0.0L;
  long double accepted_y_moment1 = 0.0L;
  long double accepted_charge2 = 0.0L;
  long double accepted_x_moment2 = 0.0L;
  long double accepted_y_moment2 = 0.0L;
  adaptive_particles::SourceEnsemble adaptive_source1;
  adaptive_particles::SourceEnsemble adaptive_source2;
};

struct Workspace::LevelSolution {
  adaptive_grid::FieldLevelGeometry geometry;
  AdaptiveFieldContext* context = NULL;
  std::vector<float> fields1;
  std::vector<float> fields2;
  std::vector<PHI_FLOAT> sampling_s1_1;
  std::vector<PHI_FLOAT> sampling_s2_1;
  std::vector<PHI_FLOAT> sampling_s1_2;
  std::vector<PHI_FLOAT> sampling_s2_2;
  std::size_t beam1_in = 0U;
  std::size_t beam1_out = 0U;
  std::size_t beam2_in = 0U;
  std::size_t beam2_out = 0U;
  double raw_lumi_dot = 0.0;
  double charge_curvature_score = 0.0;
  double field_gradient_score = 0.0;
  double effective_particles_per_cell = 0.0;
  double minimum_effective_particles_per_cell = 0.0;
  double maximum_charge_closure = 0.0;
  double maximum_first_moment_error = 0.0;
  double source_build_ms = 0.0;
  double source_restrict_ms = 0.0;
  double source_prolong_ms = 0.0;
};

// Per-encounter references never outlive prepare_resolution_adaptive. The
// owning Workspace and immutable beam slices remain private to their lane.
struct Workspace::ResolutionEncounterBinding {
  int slice1, slice2;
  const adaptive_grid::UniformWindowPlan& plan;
  const std::vector<PARTICLE*>& particles1;
  const std::vector<PARTICLE*>& particles2;
  adaptive_grid::FieldOperator field_operator;
  FieldPrefetch& output;
};

// Borrowed selection state is confined to one encounter and one worker lane.
// The caller owns every vector; helpers neither retain nor publish references.
struct Workspace::ResolutionControllerState {
  ResolutionEncounterBinding& binding;
  const std::vector<std::size_t>& spacing_factors;
  CoherentSourceHierarchy& source_hierarchy;
  std::vector<LevelSolution>& levels;
  std::vector<adaptive_grid::RichardsonEstimate>& raw_pairs;
  std::vector<adaptive_grid::SymmetricVectorFieldSensitivity>& raw_pair_sensitivities;
  std::vector<adaptive_grid::VectorFieldErrorStatistics>& raw_pair_vector_statistics;
  std::vector<adaptive_grid::VectorFieldRichardsonEstimate>& vector_pair_estimates;
  std::vector<adaptive_grid::RichardsonEstimate>& pair_estimates;
  std::vector<double>& vector_coarse_errors;
  std::vector<double>& vector_fine_errors;
  std::vector<unsigned char>& raw_pair_ready;
  std::vector<unsigned char>& pair_estimate_ready;
  std::vector<double>& observed_orders;
  std::vector<unsigned char>& asymptotic_pairs;
  std::vector<std::size_t>& triplet_first_pairs;
  std::vector<std::size_t>& triplet_second_pairs;
  std::vector<unsigned char>& triplet_valid;
  const std::vector<unsigned char>& accepted_targets1;
  const std::vector<unsigned char>& accepted_targets2;
  const std::vector<PRIMARY_PARTICLE_ID>& target_ids1;
  const std::vector<PRIMARY_PARTICLE_ID>& target_ids2;
  double target;
  bool vector_error_policy;
  bool mixed_vector_error_policy;
  std::size_t source_level_index;
  bool& noise_aware_stopped;
  std::vector<double>& source_lumi_oracle_errors;
  std::vector<unsigned char>& source_lumi_oracle_ready;
};

// Mutable decision values are owned by the encounter orchestration scope.
struct Workspace::ResolutionSelection {
  std::size_t& selected;
  bool& met;
  bool& finest_asymptotic;
  bool& mode3_terminal_uniform;
  std::size_t& noncontracting_triplet_streak;
  std::size_t& terminal_pair;
  adaptive_grid::RichardsonEstimate& accepted_estimate;
};

struct Workspace::TerminalSourceSupport {
  double tail_fraction = 1.0;
  std::size_t straddlers = 0U;
  std::size_t admitted1 = 0U;
  std::size_t admitted2 = 0U;
};

struct Workspace::TerminalCompositeLevel {
  std::vector<float> fields1;
  std::vector<float> fields2;
  double luminosity = 0.0;
  double local_luminosity = 0.0;
  double residual = 1.0;
  double boundary_rms = 1.0;
  double boundary_quantile = 1.0;
  double boundary_maximum = 1.0;
  double neff = 0.0;
  double charge_closure = 1.0;
  double moment_closure = 1.0;
  std::size_t harmonic_iterations = 0U;
  std::uint64_t source_hash = 0U;
  bool reliable = false;
};

Workspace::Workspace()
    : fourier_(true), switches_(NULL), secondaries_(NULL), muons_(NULL),
      pair_s4_(0.0), pair_lns4_(0.0), run_seed_(1U),
      beam1_(NULL), beam2_(NULL), dense_clear_visits_(0),
      actual_clear_visits_(0), dense_dot_visits_(0),
      actual_dot_visits_(0), direct_encounters_(0), fft_encounters_(0),
      direct_active_sources_(0), direct_target_nodes_(0),
      estimated_interactions_(0), direct_interactions_(0),
      maximum_estimated_interactions_(0), direct_ms_(0.0), fft_ms_(0.0),
      adaptive_encounters_(0), adaptive_full_window_encounters_(0),
      adaptive_required_cells_(0), adaptive_local_cells_(0),
      adaptive_padded_cells_(0), adaptive_global_cells_avoided_(0),
      adaptive_plan_builds_(0), adaptive_context_builds_(0),
      adaptive_context_hits_(0), adaptive_uniform_context_misses_(0),
      adaptive_patch_context_misses_(0), adaptive_context_evictions_(0),
      adaptive_context_bytes_(0),
      adaptive_bounds_ms_(0.0), adaptive_context_build_ms_(0.0),
      adaptive_deposit_ms_(0.0), adaptive_solve_ms_(0.0),
      adaptive_gather_ms_(0.0), mode2_encounters_(0),
      mode2_field_solves_(0), mode2_refined_encounters_(0),
      mode2_unmet_error_encounters_(0),
      mode2_noise_aware_stop_encounters_(0),
      vector_catastrophe_failure_encounters_(0),
      maximum_vector_difference_(0.0),
      maximum_beam_above_target_exposure_(0.0),
      mode2_tail_budget_eligible_encounters_(0),
      mode2_tail_budget_used_encounters_(0),
      encounter_tail_budget_selected_pairs_(0U),
      encounter_tail_budget_selected_fraction_(0.0),
      encounter_tail_weighted_field_error_(0.0),
      encounter_tail_weighted_field_sensitivity_(0.0L),
      encounter_tail_weighted_source_lumi_error_(0.0),
      encounter_all_uniform_lumi_measure_(0.0L),
      encounter_tail_uniform_lumi_measure_(0.0L),
      mode3_patch_requests_(0),
      mode3_patch_encounters_(0), mode3_uniform_promotions_(0),
      mode3_terminal_uniform_encounters_(0),
      mode3_terminal_patch_attempts_(0),
      mode3_terminal_patch_accepts_(0),
      mode3_terminal_patch_used_(0),
      mode3_terminal_patch_rejects_(0),
      mode3_terminal_patch_uniform_fallbacks_(0),
      mode3_terminal_patch_resource_rejects_(0),
      mode3_terminal_patch_nonresource_rejects_(0),
      mode3_terminal_patch_rejection_mask_or_(0U),
      mode3_no_patch_encounters_(0), mode3_empty_encounters_(0),
      mode3_rejection_mask_or_(NestedPatchRejectionNone),
      mode3_patch_coarse_cells_(0), mode3_patch_fine_cells_(0),
      mode3_attempted_patch_coarse_cells_(0),
      mode3_attempted_patch_fine_cells_(0),
      mode3_harmonic_iterations_(0), mode3_controller_ms_(0.0),
      mode3_harmonic_ms_(0.0), maximum_estimated_field_error_(0.0),
      maximum_patch_residual_(0.0), maximum_patch_field_error_(0.0),
      maximum_examined_patch_lumi_oracle_error_(0.0),
      maximum_accepted_patch_lumi_oracle_error_(0.0),
      patch_lumi_oracle_evaluations_(0U),
      accepted_patch_lumi_oracle_evaluations_(0U),
      maximum_source_lumi_oracle_error_(0.0),
      maximum_selected_source_lumi_oracle_error_(0.0),
      maximum_source_charge_closure_(0.0),
      maximum_source_projection_moment_shift_(0.0),
      source_hierarchy_encounters_(0U),
      source_density_hash_nonzero_encounters_(0U),
      source_sampling_hash_nonzero_encounters_(0U),
      source_sampling_hierarchy_hash_nonzero_encounters_(0U),
      source_beam1_in_visits_(0U), source_beam1_out_visits_(0U),
      source_beam2_in_visits_(0U), source_beam2_out_visits_(0U),
      source_build_ms_(0.0), source_restrict_ms_(0.0),
      source_prolong_ms_(0.0),
      initialized_(false), physics_initialized_(false),
      collision_ready_(false) {}

Workspace::~Workspace() = default;

void validate_field_result_binding(
    const FieldPrefetch& fields, int slice1, int slice2,
    const std::vector<PARTICLE*>& particles1,
    const std::vector<PARTICLE*>& particles2) {
  if (!fields.window_plan) return;
  if (fields.bound_slice1 != slice1 || fields.bound_slice2 != slice2 ||
      fields.bound_particle_count1 != particles1.size() ||
      fields.bound_particle_count2 != particles2.size())
    throw std::runtime_error(
        "adaptive field result rejected: slice/count binding mismatch");
  if (fields.lattice_hash != fields.window_plan->lattice_hash())
    throw std::runtime_error(
        "adaptive field result rejected: lattice descriptor mismatch");
  if (fields.window_plan->descriptor_payload_retained() &&
      (fields.window_plan->beam1_sources().size() != particles1.size() ||
       fields.window_plan->beam2_sources().size() != particles2.size() ||
       fields.window_plan->beam1_targets().size() != particles1.size() ||
       fields.window_plan->beam2_targets().size() != particles2.size()))
    throw std::runtime_error(
        "adaptive field result rejected: descriptor count mismatch");
  const std::uint64_t expected = field_binding_hash(
      slice1, slice2, particles1, particles2, fields.lattice_hash,
      fields.window_plan->plan_hash(), fields.result_geometry_hash,
      fields.source_model_tag, fields.source_reference_factor,
      fields.source_density_hash, fields.source_sampling_hash,
      fields.source_sampling_hierarchy_hash);
  if (expected != fields.result_binding_hash)
    throw std::runtime_error(
        "adaptive field result rejected: particle-order/revision binding "
        "mismatch");
}

void bind_field_result(
    FieldPrefetch& fields, int slice1, int slice2,
    const std::vector<PARTICLE*>& particles1,
    const std::vector<PARTICLE*>& particles2) {
  if (!fields.window_plan)
    throw std::invalid_argument(
        "adaptive field publication requires an immutable window plan");
  if (fields.window_plan->lattice_hash() == 0U)
    throw std::invalid_argument(
        "adaptive field publication requires a valid lattice identity");
  if (fields.window_plan->descriptor_payload_retained() &&
      (fields.window_plan->beam1_sources().size() != particles1.size() ||
       fields.window_plan->beam2_sources().size() != particles2.size() ||
       fields.window_plan->beam1_targets().size() != particles1.size() ||
       fields.window_plan->beam2_targets().size() != particles2.size()))
    throw std::invalid_argument(
        "adaptive field publication descriptor count mismatch");
  fields.lattice_hash = fields.window_plan->lattice_hash();
  fields.bound_slice1 = slice1;
  fields.bound_slice2 = slice2;
  fields.bound_particle_count1 = particles1.size();
  fields.bound_particle_count2 = particles2.size();
  fields.result_geometry_hash = 0U;
  fields.result_binding_hash = field_binding_hash(
      slice1, slice2, particles1, particles2, fields.lattice_hash,
      fields.window_plan->plan_hash(), fields.result_geometry_hash,
      fields.source_model_tag, fields.source_reference_factor,
      fields.source_density_hash, fields.source_sampling_hash,
      fields.source_sampling_hierarchy_hash);
}

std::size_t Workspace::estimated_bytes(
    const Configuration& configuration) {
  if (configuration.n_x <= 0 || configuration.n_y <= 0)
    throw std::invalid_argument(
        "cannot estimate a non-positive wavefront grid");
  const std::size_t nx = static_cast<std::size_t>(configuration.n_x);
  const std::size_t ny = static_cast<std::size_t>(configuration.n_y);
  if (nx > std::numeric_limits<std::size_t>::max() / ny)
    throw std::overflow_error("wavefront grid cell count overflows size_t");
  const std::size_t cells = nx * ny;

  // Conservative accounting for rho/phi/Green spectrum, two complex FFT
  // work arrays, four legacy per-cell vector headers allocated by GRID, the
  // sparse marks/indices, and allocator/plan headroom.  Compact per-primary
  // fields are wave-owned rather than workspace-owned and are bounded by the
  // already allocated bunch population.
  const std::size_t bytes_per_cell = 384U;
  const std::size_t fixed_headroom = 1024U * 1024U;
  if (cells > (std::numeric_limits<std::size_t>::max() - fixed_headroom) /
                  bytes_per_cell)
    throw std::overflow_error(
        "wavefront workspace memory estimate overflows size_t");
  const std::size_t grid_bytes = cells * bytes_per_cell + fixed_headroom;
  // Conservative combined accounting for one keyed reservoir node, hash-set
  // entry, and LUMI_PAIR_EE payload.  The exact implementation is allocator
  // dependent; overestimating is preferable to an OOM after beam mutation.
  const std::size_t bytes_per_lumi_event = 320U;
  if (configuration.lumi_event_capacity >
      (std::numeric_limits<std::size_t>::max() - grid_bytes) /
          bytes_per_lumi_event)
    throw std::overflow_error(
        "wavefront luminosity-reservoir estimate overflows size_t");
  return grid_bytes +
      configuration.lumi_event_capacity * bytes_per_lumi_event;
}

std::size_t Workspace::estimated_adaptive_context_bytes(
    const Configuration& configuration) {
  if (configuration.adaptive_grid_mode < 2) return 0U;
  validate_workspace_configuration(configuration);
  if (configuration.n_x < 4 || configuration.n_y < 4 ||
      configuration.adaptive_spacing_factors.empty())
    throw std::invalid_argument(
        "cannot estimate unresolved adaptive resolution contexts");
  validate_resolution_factor_sequence(configuration);
  const adaptive_grid::UniformWindowPolicy& policy =
      configuration.adaptive_window_policy;
  const std::size_t canonical_x = static_cast<std::size_t>(configuration.n_x);
  const std::size_t canonical_y = static_cast<std::size_t>(configuration.n_y);
  const std::size_t window_x = std::min(
      canonical_x, policy.max_n_x == 0U ? canonical_x : policy.max_n_x);
  const std::size_t window_y = std::min(
      canonical_y, policy.max_n_y == 0U ? canonical_y : policy.max_n_y);
  const adaptive_grid::FieldOperator field_operator =
      adaptive_grid::parse_field_operator(
          configuration.adaptive_field_operator);
  const std::size_t operator_guard =
      adaptive_grid::field_operator_additional_window_guard_cells(
          field_operator);
  std::size_t total = 0U;
  std::size_t patch_parent_x = 0U;
  std::size_t patch_parent_y = 0U;
  std::size_t patch_parent_factor = 1U;
  for (std::size_t factor : configuration.adaptive_spacing_factors) {
    std::size_t explicit_guard = checked_product(
        static_cast<std::size_t>(std::max<std::int64_t>(
            0, policy.extra_guard_cells)), factor,
        "adaptive context estimate explicit guard");
    if (configuration.adaptive_grid_mode == 3)
      explicit_guard = std::max(
          explicit_guard,
          checked_sum(
              checked_sum(
                  checked_product(2U, configuration.adaptive_blocking_factor,
                                  "adaptive context estimate block guard"),
                  configuration.adaptive_error_buffer_cells,
                  "adaptive context estimate error guard"),
              configuration.adaptive_transition_cells,
              "adaptive context estimate transition guard"));
    explicit_guard = checked_sum(
        explicit_guard, operator_guard,
        "adaptive context estimate operator guard");
    const std::size_t guard = factor == 1U
        ? explicit_guard
        : checked_sum(std::max<std::size_t>(2U, factor), explicit_guard,
                      "adaptive context estimate fine guard");
    const std::size_t global_x = checked_product(
        canonical_x, factor, "adaptive context estimate global x");
    const std::size_t global_y = checked_product(
        canonical_y, factor, "adaptive context estimate global y");
    std::size_t n_x = std::min(
        global_x, checked_sum(checked_product(window_x, factor,
                                              "adaptive context window x"),
                              checked_product(2U, guard,
                                              "adaptive context guards x"),
                              "adaptive context extent x"));
    std::size_t n_y = std::min(
        global_y, checked_sum(checked_product(window_y, factor,
                                              "adaptive context window y"),
                              checked_product(2U, guard,
                                              "adaptive context guards y"),
                              "adaptive context extent y"));
    if (configuration.adaptive_source_model ==
        "fixed_width_bspline1_v1") {
      const float delta_x =
          (2.0f * configuration.cut_x /
           static_cast<float>(configuration.n_x)) /
          static_cast<float>(factor);
      const float delta_y =
          (2.0f * configuration.cut_y /
           static_cast<float>(configuration.n_y)) /
          static_cast<float>(factor);
      const std::size_t source_guard_x = fixed_width_support_guard(
          configuration.adaptive_source_width_x, delta_x,
          "fixed-width context estimate x guard");
      const std::size_t source_guard_y = fixed_width_support_guard(
          configuration.adaptive_source_width_y, delta_y,
          "fixed-width context estimate y guard");
      n_x = std::min(
          global_x,
          checked_sum(
              n_x, checked_product(2U, source_guard_x,
                                   "fixed-width context x guards"),
              "fixed-width context extent x"));
      n_y = std::min(
          global_y,
          checked_sum(
              n_y, checked_product(2U, source_guard_y,
                                   "fixed-width context y guards"),
              "fixed-width context extent y"));
    }
    if (configuration.adaptive_source_model == "reference_cic_v1") {
      const std::size_t source_factor =
          configuration.adaptive_source_reference_nx / canonical_x;
      if (factor > source_factor) {
        const std::size_t ratio = factor / source_factor;
        const std::size_t alignment_headroom = checked_product(
            2U, ratio - 1U,
            "reference-aligned context boundary headroom");
        n_x = std::min(
            global_x, checked_sum(n_x, alignment_headroom,
                                  "reference-aligned context x"));
        n_y = std::min(
            global_y, checked_sum(n_y, alignment_headroom,
                                  "reference-aligned context y"));
      }
    }
    if (configuration.adaptive_max_cells != 0U &&
        checked_product(n_x, n_y, "adaptive context class cells") >
            configuration.adaptive_max_cells)
      throw std::runtime_error(
          "declared adaptive class bank cannot fit adaptive_grid_max_cells");
    total = checked_sum(total, adaptive_context_estimate(n_x, n_y, policy),
                        "adaptive context bank estimate");
    if (configuration.adaptive_grid_mode == 3 &&
        factor == configuration.adaptive_spacing_factors[
                      configuration.adaptive_spacing_factors.size() - 2U]) {
      patch_parent_x = n_x;
      patch_parent_y = n_y;
      patch_parent_factor = factor;
    }
  }
  if (configuration.adaptive_grid_mode == 3) {
    const std::size_t patch_x = checked_product(
        patch_parent_x, configuration.adaptive_refinement_ratio,
        "adaptive patch estimate x");
    const std::size_t patch_y = checked_product(
        patch_parent_y, configuration.adaptive_refinement_ratio,
        "adaptive patch estimate y");
    const std::size_t patch_factor = checked_product(
        patch_parent_factor, configuration.adaptive_refinement_ratio,
        "adaptive patch estimate factor");
    (void)patch_factor;
    if (configuration.adaptive_max_cells != 0U &&
        checked_product(patch_x, patch_y, "adaptive patch estimate cells") >
            configuration.adaptive_max_cells)
      throw std::runtime_error(
          "declared Mode-3 patch can exceed adaptive_grid_max_cells");
    total = checked_sum(total,
                        adaptive_context_estimate(patch_x, patch_y, policy),
                        "adaptive context plus patch estimate");
  }
  return total;
}

std::size_t Workspace::estimated_adaptive_transient_bytes(
    const Configuration& configuration) {
  if (configuration.adaptive_grid_mode < 2) return 0U;
  validate_workspace_configuration(configuration);
  if (configuration.n_x < 4 || configuration.n_y < 4 ||
      configuration.adaptive_spacing_factors.empty())
    throw std::invalid_argument(
        "cannot estimate unresolved adaptive transient storage");
  validate_resolution_factor_sequence(configuration);
  const std::size_t canonical_x =
      static_cast<std::size_t>(configuration.n_x);
  const std::size_t canonical_y =
      static_cast<std::size_t>(configuration.n_y);
  // A zero factor cap permits runtime extension beyond the bootstrap list,
  // so bootstrap.back() is not a valid transient high-water bound.  Modes
  // 2/3 require an explicit positive cell capacity; reserve against that
  // physical allocation boundary directly.  An optional factor cap can only
  // lower the reachable cell count, making this conservative.
  std::size_t maximum_cells = 0U;
  if (configuration.adaptive_max_cells != 0U) {
    if (configuration.adaptive_max_cells >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error(
          "adaptive_grid_max_cells does not fit host size_t");
    maximum_cells =
        static_cast<std::size_t>(configuration.adaptive_max_cells);
  } else {
    throw std::invalid_argument(
        "dynamic adaptive transient estimate requires max_cells");
  }

  const std::size_t particle_count = checked_sum(
      configuration.maximum_slice_particles1,
      configuration.maximum_slice_particles2,
      "adaptive transient particle count");
  // Bound the count of retained per-particle level vectors by the smallest
  // executable cropped window.  Ignoring guards here can only admit extra
  // theoretical factors and therefore over-reserve; using every size_t bit
  // would be needlessly catastrophic for high-macro-particle cards.
  const std::size_t minimum_x = std::max<std::size_t>(
      4U, configuration.adaptive_window_policy.min_n_x);
  const std::size_t minimum_y = std::max<std::size_t>(
      4U, configuration.adaptive_window_policy.min_n_y);
  std::size_t dynamic_level_slots = 0U;
  for (std::size_t factor = 1U;;) {
    if (configuration.adaptive_max_refinement_factor != 0U &&
        factor > configuration.adaptive_max_refinement_factor)
      break;
    if (minimum_x > std::numeric_limits<std::size_t>::max() / factor ||
        minimum_y > std::numeric_limits<std::size_t>::max() / factor)
      break;
    const std::size_t extent_x = minimum_x * factor;
    const std::size_t extent_y = minimum_y * factor;
    if (extent_x != 0U &&
        extent_y > maximum_cells / extent_x)
      break;
    ++dynamic_level_slots;
    if (factor > std::numeric_limits<std::size_t>::max() /
                     configuration.adaptive_refinement_ratio)
      break;
    factor *= configuration.adaptive_refinement_ratio;
  }
  dynamic_level_slots = std::max(
      dynamic_level_slots, configuration.adaptive_spacing_factors.size());
  const std::size_t maximum_level_slots = checked_sum(
      dynamic_level_slots, 3U,
      "adaptive transient dynamic particle levels");
  const std::size_t particle_bytes_each = checked_product(
      maximum_level_slots,
      16U, "adaptive transient bytes per particle");
  const std::size_t particle_scratch = checked_product(
      particle_count, particle_bytes_each,
      "adaptive transient particle scratch");
  std::size_t reference_storage = 0U;
  if (configuration.adaptive_source_model == "reference_cic_v1") {
    const std::size_t reference_cells = checked_product(
        configuration.adaptive_source_reference_nx,
        configuration.adaptive_source_reference_ny,
        "reference CIC transient cells");
    std::size_t reference_values = checked_product(
        reference_cells, 2U, "reference CIC rho values");
    const std::size_t source_factor =
        configuration.adaptive_source_reference_nx / canonical_x;
    for (std::size_t factor : configuration.adaptive_spacing_factors) {
      if (factor > source_factor) break;
      const std::size_t sampling_cells = checked_product(
          checked_product(canonical_x, factor,
                          "reference CIC sampling x"),
          checked_product(canonical_y, factor,
                          "reference CIC sampling y"),
          "reference CIC sampling cells");
      reference_values = checked_sum(
          reference_values,
          checked_product(sampling_cells, 4U,
                          "reference CIC S1/S2 values"),
          "reference CIC hierarchy values");
    }
    // Four simultaneous cropped S1/S2 arrays are the largest transfer
    // scratch above the reference level.  Account the full reference extent
    // even though encounter crops are normally much smaller.
    reference_values = checked_sum(
        reference_values,
        checked_product(reference_cells, 4U,
                        "reference CIC transfer scratch values"),
        "reference CIC hierarchy plus transfer values");
    reference_storage = checked_product(
        reference_values, sizeof(PHI_FLOAT),
        "reference CIC transient bytes");
  }
  const std::size_t fixed_headroom = 8U * 1024U * 1024U;
  const std::size_t fixed_transient = checked_sum(
      checked_sum(particle_scratch, reference_storage,
                  "adaptive fixed particle/reference scratch"),
      fixed_headroom, "adaptive fixed transient high-water");
  const std::size_t context_bytes_per_cell = checked_sum(
      configuration.adaptive_window_policy.bytes_per_local_cell,
      checked_product(
          4U, configuration.adaptive_window_policy.bytes_per_padded_cell,
          "adaptive padded context bytes per local cell"),
      "adaptive context bytes per local cell");
  const std::size_t scratch_bytes_per_cell = 256U;
  const std::size_t bootstrap_context_bytes =
      estimated_adaptive_context_bytes(configuration);
  // The user cell cap is an allocation safety limit, not a promise that every
  // cell can coexist with FFT state and controller scratch.  Bound scratch by
  // the largest cell array reachable under both that cap and the explicit
  // byte budget.  FeeBL requires at least two wavefront workers, so no actual
  // lane can receive more than half of the run-wide adaptive byte capacity.
  const std::size_t reachable_cells =
      runtime_safety::reachable_adaptive_cell_bound(
          configuration.adaptive_max_cells,
          configuration.adaptive_window_policy.max_bytes,
          bootstrap_context_bytes, fixed_transient,
          context_bytes_per_cell, scratch_bytes_per_cell, 2U);
  const std::size_t cell_scratch = checked_product(
      reachable_cells, scratch_bytes_per_cell,
      "reachable adaptive transient cell scratch");
  return checked_sum(
      fixed_transient, cell_scratch,
      "adaptive transient high-water reservation");
}

void Workspace::initialize(const Configuration& configuration,
                           BEAM* beam1, BEAM* beam2) {
  if (initialized_)
    throw std::logic_error("wavefront field workspace initialized twice");
  if (!beam1 || !beam2)
    throw std::invalid_argument(
        "wavefront field workspace requires two beams");
  validate_workspace_configuration(configuration);

  configuration_ = configuration;
  if (configuration_.adaptive_macroparticles.mode !=
      adaptive_particles::Mode::LegacyOff) {
    beam1->particle_beam().require_valid_particle_id_sidecar();
    beam2->particle_beam().require_valid_particle_id_sidecar();
  }
  mode2_spacing_selection_counts_.assign(
      configuration.adaptive_spacing_factors.size(), 0U);
  beam1_ = beam1;
  beam2_ = beam2;
  if (configuration_.adaptive_grid_mode >= 2) {
    if (configuration_.n_t <= 0)
      throw std::invalid_argument(
          "Mode 2/3 encounter-tail accounting requires positive n_t");
    std::vector<std::size_t> slice_counts1(
        static_cast<std::size_t>(beam1_->number_of_slices()), 0U);
    std::vector<std::size_t> slice_counts2(
        static_cast<std::size_t>(beam2_->number_of_slices()), 0U);
    for (std::size_t slice = 0U; slice < slice_counts1.size(); ++slice)
      slice_counts1[slice] = beam1_->getParticleVector(
          static_cast<int>(slice)).size();
    for (std::size_t slice = 0U; slice < slice_counts2.size(); ++slice)
      slice_counts2[slice] = beam2_->getParticleVector(
          static_cast<int>(slice)).size();
    const adaptive_grid::EncounterTailBudget tail_budget =
        adaptive_grid::make_encounter_tail_budget(
            slice_counts1, slice_counts2,
            configuration_.adaptive_encounter_tail_fraction > 0.0
                ? configuration_.adaptive_encounter_tail_fraction
                : configuration_.adaptive_tau_tail);
    encounter_tail_exemption_mask_ = tail_budget.exemption_mask;
    encounter_tail_budget_selected_pairs_ = tail_budget.selected_pairs;
    encounter_tail_budget_selected_fraction_ = static_cast<double>(
        tail_budget.selected_fraction);
    if (!finite_value(encounter_tail_budget_selected_fraction_) ||
        encounter_tail_budget_selected_fraction_ < 0.0 ||
        encounter_tail_budget_selected_fraction_ >
            (configuration_.adaptive_encounter_tail_fraction > 0.0
                 ? configuration_.adaptive_encounter_tail_fraction
                 : configuration_.adaptive_tau_tail))
      throw std::logic_error(
          "Mode 2/3 encounter-tail selector exceeded its declared budget");

    long double total1 = 0.0L;
    long double total2 = 0.0L;
    for (std::size_t count : slice_counts1)
      total1 += static_cast<long double>(count);
    for (std::size_t count : slice_counts2)
      total2 += static_cast<long double>(count);
    const std::size_t pair_count = checked_product(
        slice_counts1.size(), slice_counts2.size(),
        "Mode 2/3 encounter-tail pair count");
    encounter_population_weights_.assign(pair_count, 0.0);
    const long double substeps =
        static_cast<long double>(configuration_.n_t);
    long double normalized_sum = 0.0L;
    for (std::size_t first = 0U; first < slice_counts1.size(); ++first)
      for (std::size_t second = 0U; second < slice_counts2.size(); ++second) {
        const long double pair_fraction =
            (static_cast<long double>(slice_counts1[first]) / total1) *
            (static_cast<long double>(slice_counts2[second]) / total2);
        const double encounter_weight = static_cast<double>(
            pair_fraction / substeps);
        if (!finite_value(encounter_weight) || encounter_weight < 0.0)
          throw std::logic_error(
              "Mode 2/3 encounter-tail weight is not representable");
        encounter_population_weights_[
            first * slice_counts2.size() + second] = encounter_weight;
        normalized_sum += pair_fraction;
      }
    if (std::fabs(normalized_sum - 1.0L) > 1.0e-12L)
      throw std::logic_error(
          "Mode 2/3 encounter-tail weights do not close to unity");
  }
  grid_.set_n_cell_z(configuration.n_z);
  grid_.set_timestep(configuration.n_t);
  grid_.init_grid_comp(configuration.n_x, configuration.n_y, 2, &fourier_);
  grid_.enable_real_space_field_kernel(
      configuration.field_solver != sparse_direct_fields::Mode::Fft);
  grid_.connect_beams(beam1_, beam2_);
  // A fresh worker GRID has no legacy slice population metadata.  Establish
  // the exact generated macroparticle counts before init_grid_phys converts
  // physical bunch populations into per-macroparticle charge/weight.
  grid_.set_nb_macroparticles(
      1, beam1_->particle_beam().numberOfParticles());
  grid_.set_nb_macroparticles(
      2, beam2_->particle_beam().numberOfParticles());
  // Field deposition uses the audited macro charges directly.  The physical
  // populations additionally initialize the collision weights used when the
  // same workspace is opted into full encounter processing.
  grid_.init_grid_phys(static_cast<float>(configuration.physical_particles1),
                       static_cast<float>(configuration.physical_particles2),
                       configuration.cut_x, configuration.cut_y,
                       configuration.cut_z, configuration.charge_sign,
                       &fourier_);
  if (configuration.adaptive_grid_mode != 0) {
    if (configuration.field_solver != sparse_direct_fields::Mode::Fft)
      throw std::invalid_argument(
          "adaptive grids require the exact FFT field solver");
    if (configuration.adaptive_grid_mode >= 2) {
      validate_resolution_factor_sequence(configuration);
      if (!(configuration.adaptive_error_target > 0.0) ||
          !finite_value(configuration.adaptive_error_target) ||
          !(configuration.adaptive_field_floor > 0.0) ||
          !finite_value(configuration.adaptive_field_floor))
        throw std::invalid_argument(
            "Mode 2/3 requires resolved spacing classes, error target, and "
            "positive field floor");
    }
    if (configuration.adaptive_grid_mode == 3 &&
        (!(configuration.adaptive_fine_residual_tolerance > 0.0) ||
         !finite_value(configuration.adaptive_fine_residual_tolerance) ||
         !(configuration.adaptive_tau_tail > 0.0) ||
         !finite_value(configuration.adaptive_tau_tail) ||
         !finite_value(configuration.adaptive_coarsen_threshold) ||
         configuration.adaptive_coarsen_threshold < 0.0 ||
         configuration.adaptive_coarsen_threshold > 1.0 ||
         configuration.adaptive_blocking_factor == 0U ||
         (configuration.adaptive_blocking_factor &
          (configuration.adaptive_blocking_factor - 1U)) != 0U ||
         configuration.adaptive_refinement_ratio != 2U ||
         configuration.adaptive_blocking_factor %
             configuration.adaptive_refinement_ratio != 0U ||
         configuration.adaptive_error_buffer_cells == 0U ||
         configuration.adaptive_transition_cells == 0U ||
         (configuration.adaptive_transition_cells & 1U) != 0U))
      throw std::invalid_argument(
          "Mode 3 requires positive finite tail/residual tolerances, a "
          "normalized overlap fraction, and valid blocking/refinement/"
          "buffer/transition controls");
    adaptive_grid::CanonicalLatticeInput lattice;
    lattice.n_x = static_cast<std::size_t>(configuration.n_x);
    lattice.n_y = static_cast<std::size_t>(configuration.n_y);
    lattice.delta_x = grid_.get_delta_x();
    lattice.delta_y = grid_.get_delta_y();
    lattice.delta_z = grid_.get_delta_z();
    lattice.inverse_delta_x = grid_.get_delta_x_inverse();
    lattice.inverse_delta_y = grid_.get_delta_y_inverse();
    lattice.offset_x = grid_.get_offset_x();
    lattice.offset_y = grid_.get_offset_y();
    lattice.intended_min_x = grid_.get_min_x();
    lattice.intended_max_x = grid_.get_max_x();
    lattice.intended_min_y = grid_.get_min_y();
    lattice.intended_max_y = grid_.get_max_y();
    lattice.charge_sign = configuration.charge_sign;
    lattice.field_kernel_factor = grid_.get_field_kernel_factor();
    lattice.geometry_revision = configuration.canonical_geometry_revision;
    canonical_lattice_.reset(
        new adaptive_grid::CanonicalLatticeSnapshot(lattice));
  }
  if (configuration.sparse_cells) {
    const std::size_t cells =
        static_cast<std::size_t>(configuration.n_x) *
        static_cast<std::size_t>(configuration.n_y);
    active_marks1_.assign(cells, 0U);
    active_marks2_.assign(cells, 0U);
    active_cells1_.reserve(cells);
    active_cells2_.reserve(cells);
    if (configuration.field_solver != sparse_direct_fields::Mode::Fft) {
      target_marks1_.assign(cells, 0U);
      target_marks2_.assign(cells, 0U);
      target_nodes1_.reserve(cells);
      target_nodes2_.reserve(cells);
    }
  }
  initialized_ = true;
}

void Workspace::initialize_physics(
    SWITCHES* switches, PAIR_BEAM* secondaries, PAIR_BEAM* muons,
    double pair_s4, double pair_lns4, double spectrum_lo,
    double spectrum_hi, std::size_t spectrum_bins, std::uint64_t run_seed,
    const feebl::design_axis::Spectrum* design_axis_spectrum) {
  if (!initialized_)
    throw std::logic_error(
        "wavefront physics workspace requires initialized grid fields");
  if (physics_initialized_)
    throw std::logic_error("wavefront physics workspace initialized twice");
  if (!switches || !secondaries || !muons ||
      !finite_value(pair_s4) || !finite_value(pair_lns4) ||
      !(spectrum_hi > spectrum_lo) || spectrum_bins == 0U)
    throw std::invalid_argument(
        "wavefront physics workspace received an invalid collision contract");

  switches_ = switches;
  secondaries_ = secondaries;
  muons_ = muons;
  pair_s4_ = pair_s4;
  pair_lns4_ = pair_lns4;
  run_seed_ = run_seed;
  physics_rng_.reset(new RNDM(static_cast<unsigned long>(run_seed_)));
  grid_.connect_random_generator(physics_rng_.get());
  grid_.generalInit(*switches_, "", "", "", "", "", NULL);
  grid_.get_results().configure_pair_spectrum(
      spectrum_lo, spectrum_hi, spectrum_bins);
  if (design_axis_spectrum)
    grid_.design_axis_spectrum_.configure_like(*design_axis_spectrum);
  grid_.enable_sparse_cells(switches_->get_sparse_cells() != 0);
  grid_.set_photon_polarisation(switches_->get_photon_polarisation());
  grid_.set_coherent_model(switches_->get_coherent_model());
  grid_.set_wavefront_outer_parallel_lane(true);
  local_grids_.clear();
  local_grids_.push_back(&grid_);
  physics_initialized_ = true;
}

namespace {

feebl::PhysicalRandomKeyV2 physics_key(
    const feebl::wavefront_fields::PhysicsIdentity& identity,
    feebl::RandomDomain domain) {
  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = identity.run_seed;
  key.domain = domain;
  key.crossing = identity.crossing;
  key.diagonal = identity.diagonal;
  key.wave = identity.wave;
  key.substep = identity.substep;
  key.encounter = identity.encounter;
  key.beam = 0U;
  key.parent_a = feebl::kNoParentStableIdV2;
  key.parent_b = feebl::kNoParentStableIdV2;
  key.interaction = 0U;
  key.emission = 0U;
  key.rejection_attempt = 0U;
  return key;
}

}  // namespace

void Workspace::process_primary_photon_collision(
    int slice1, int slice2, const FieldPrefetch& fields,
    const PhysicsIdentity& identity) {
  if (!physics_initialized_ || !physics_rng_)
    throw std::logic_error(
        "wavefront collision entered an uninitialized physics workspace");
  if (collision_ready_)
    throw std::logic_error(
        "wavefront collision lane was not published before reuse");
  if (identity.run_seed != run_seed_)
    throw std::invalid_argument("wavefront collision RNG seed drifted");

  validate_field_result_binding(
      fields, slice1, slice2,
      beam1_->getParticleVector(slice1),
      beam2_->getParticleVector(slice2));

  const float min_z = 0.5f * static_cast<float>(slice2 - slice1 - 1);
  grid_.all_distribute_background_only(
      slice1, slice2, *switches_, pair_s4_, pair_lns4_);
  grid_.add_external_distribution_counts(
      fields.beam1_in, fields.beam1_out,
      fields.beam2_in, fields.beam2_out);
  feebl::electron_lumi_sampling::PairIdentity sampling_identity;
  sampling_identity.run_seed = identity.run_seed;
  sampling_identity.crossing = identity.crossing;
  sampling_identity.diagonal = identity.diagonal;
  sampling_identity.wave = identity.wave;
  sampling_identity.substep = identity.substep;
  sampling_identity.encounter = identity.encounter;
  grid_.step_lumi_with_fine_dot(
      min_z, *secondaries_, identity.legacy_time_counter, *switches_,
      slice1, slice2, fields.fine_lumi_dot, &sampling_identity);
  grid_.update_slice_charge(slice1, slice2);

  physics_rng_->begin_counter_stream(
      physics_key(identity, feebl::RandomDomain::Beamstrahlung));
  grid_.movePrimaryParticlesWithPrefetchedFields(
      local_grids_, slice1, slice2, switches_->get_interpolation(),
      switches_->get_do_beamstrahlung(), switches_->get_ST_spin_flip(),
      switches_->get_emin(), switches_->get_do_prod(),
      switches_->get_extra_grids(), switches_->get_charge_sign(),
      switches_->get_bmt_precession(),
      fields.fields1.empty() ? NULL : &fields.fields1[0],
      fields.fields1.size() / 2U,
      fields.fields2.empty() ? NULL : &fields.fields2[0],
      fields.fields2.size() / 2U,
      switches_->get_do_trident(), switches_->get_do_coherent());

  if (switches_->get_do_photons(1) || switches_->get_do_photons(2) ||
      switches_->get_load_photon()) {
    physics_rng_->begin_counter_stream(
        physics_key(identity,
                    feebl::RandomDomain::PhotonMacroparticleSampling));
    grid_.distribute_photons(
        slice1, slice2, switches_->get_photon_ratio(), *physics_rng_);
    if (switches_->get_do_photon_lumi()) {
      physics_rng_->begin_counter_stream(
          physics_key(identity, feebl::RandomDomain::LuminosityEventSampling));
      grid_.photon_lumi(
          min_z, *switches_, *secondaries_, *muons_, *physics_rng_);
    }
    grid_.move_photons(*beam1_, 1, slice1);
    grid_.move_photons(*beam2_, 2, slice2);
  }
  collision_ready_ = true;
}

bool Workspace::publish_primary_photon_collision(
    GRID& target, std::string* reason) {
  if (!collision_ready_) {
    if (reason) *reason = "wavefront collision lane has no completed result";
    return false;
  }
  if (!target.merge_primary_photon_collision_lane(grid_, reason))
    return false;
  acknowledge_primary_photon_publication();
  if (reason) reason->clear();
  return true;
}

const GRID& Workspace::ready_primary_photon_lane() const {
  if (!collision_ready_)
    throw std::logic_error(
        "wavefront collision lane has no completed result");
  return grid_;
}

void Workspace::acknowledge_primary_photon_publication() {
  if (!collision_ready_)
    throw std::logic_error(
        "wavefront collision lane was acknowledged without a result");
  grid_.reset_primary_photon_collision_lane(*switches_);
  collision_ready_ = false;
}

const GRID& Workspace::completed_primary_photon_lane() const {
  if (!physics_initialized_)
    throw std::logic_error(
        "wavefront luminosity publication requires initialized physics");
  if (collision_ready_)
    throw std::logic_error(
        "wavefront luminosity publication found an unpublished collision");
  return grid_;
}

void Workspace::collect_target_nodes(
    const std::vector<PARTICLE*>& particles,
    std::vector<std::size_t>& target_nodes,
    std::vector<unsigned char>& target_marks)
{
  for (std::size_t position = 0; position < target_nodes.size(); ++position)
    target_marks[target_nodes[position]] = 0U;
  target_nodes.clear();

  const int ny = configuration_.n_y;
  const auto add_node = [&](std::size_t node) {
    if (!target_marks[node]) {
      target_marks[node] = 1U;
      target_nodes.push_back(node);
    }
  };

  for (std::size_t particle = 0; particle < particles.size(); ++particle) {
    if (!particles[particle])
      throw std::logic_error(
          "wavefront target stencil encountered a null primary particle");
    float x = 0.0f;
    float y = 0.0f;
    particles[particle]->XYposition(x, y);
    // With grids=1 the established out-of-main-grid contract is a zero field;
    // no potential nodes are needed for those particles.
    if (!grid_.coordinatesInGridRange(x, y)) continue;

    PHI_FLOAT absolute_x = 0.0;
    PHI_FLOAT absolute_y = 0.0;
    grid_.absoluteCoordinates(x, y, absolute_x, absolute_y);

    std::size_t stencil[12];
    sparse_direct_fields::electric_field_stencil_nodes(
        absolute_x, absolute_y, configuration_.n_x, ny, stencil);
    for (std::size_t node = 0; node < 12U; ++node)
      add_node(stencil[node]);
  }
  std::sort(target_nodes.begin(), target_nodes.end());
}

void Workspace::clear_rho(
    PHI_FLOAT* rho, std::vector<std::size_t>& active_cells,
    std::vector<unsigned char>& active_marks, std::size_t cells) {
  checked_add_counter(dense_clear_visits_, cells, "dense-clear visit");
  if (!configuration_.sparse_cells) {
    std::fill(rho, rho + cells, PHI_FLOAT(0.0));
    checked_add_counter(actual_clear_visits_, cells,
                        "actual-clear visit");
    return;
  }
  checked_add_counter(actual_clear_visits_, active_cells.size(),
                      "actual-clear visit");
  for (std::size_t position = 0; position < active_cells.size(); ++position) {
    const std::size_t cell = active_cells[position];
    rho[cell] = PHI_FLOAT(0.0);
    active_marks[cell] = 0U;
  }
  active_cells.clear();
}

void Workspace::deposit_slice(const std::vector<PARTICLE*>& particles,
                              PHI_FLOAT macro_charge,
                              PHI_FLOAT* rho,
                              std::vector<std::size_t>& active_cells,
                              std::vector<unsigned char>& active_marks,
                              std::size_t& particles_in,
                              std::size_t& particles_out) {
  particles_in = 0;
  particles_out = 0;
  for (std::size_t particle = 0; particle < particles.size(); ++particle) {
    if (!particles[particle])
      throw std::logic_error(
          "wavefront field workspace encountered a null primary particle");
    float x = 0.0f;
    float y = 0.0f;
    float hx = 0.0f;
    float hy = 0.0f;
    int ix = 0;
    int iy = 0;
    particles[particle]->XYposition(x, y);
    if (grid_.particleInGrid(x, y, hx, hy, ix, iy)) {
      if (configuration_.sparse_cells) {
        const std::size_t ny =
            static_cast<std::size_t>(configuration_.n_y);
        const std::size_t base = static_cast<std::size_t>(ix) * ny +
                                 static_cast<std::size_t>(iy);
        const std::size_t touched[4] = {
            base, base + ny, base + 1U, base + ny + 1U};
        for (std::size_t corner = 0; corner < 4U; ++corner) {
          if (!active_marks[touched[corner]]) {
            active_marks[touched[corner]] = 1U;
            active_cells.push_back(touched[corner]);
          }
        }
      }
      // Same function, operand types, and particle order as
      // SLICE_ON_GRID::assignChargeToCIC on the legacy path.
      grid_.assignChargeToCIC(rho, ix, iy, hx, hy, macro_charge);
      ++particles_in;
    } else {
      ++particles_out;
    }
  }
}

adaptive_grid::CanonicalWindowInputs Workspace::build_window_inputs(
    int slice1, int slice2) {
  if (!canonical_lattice_)
    throw std::logic_error(
        "adaptive window planning entered without a canonical lattice");
  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  adaptive_grid::CanonicalWindowInputs inputs;
  inputs.beam1_sources.reserve(particles1.size());
  inputs.beam2_sources.reserve(particles2.size());
  inputs.beam1_targets.reserve(particles1.size());
  inputs.beam2_targets.reserve(particles2.size());

  const auto append_particle = [this](
      PARTICLE* particle,
      std::vector<adaptive_grid::CanonicalDepositDescriptor>& sources,
      std::vector<adaptive_grid::CanonicalGatherDescriptor>& targets) {
    if (!particle)
      throw std::logic_error(
          "adaptive window planner encountered a null primary particle");
    float x = 0.0f;
    float y = 0.0f;
    particle->XYposition(x, y);

    float weight_x = 0.0f;
    float weight_y = 0.0f;
    int base_x = 0;
    int base_y = 0;
    if (grid_.particleInGrid(
            x, y, weight_x, weight_y, base_x, base_y))
      sources.push_back(
          adaptive_grid::CanonicalDepositDescriptor::from_canonical(
              base_x, base_y, weight_x, weight_y));
    else
      sources.push_back(
          adaptive_grid::CanonicalDepositDescriptor::rejected());

    if (grid_.coordinatesInGridRange(x, y)) {
      PHI_FLOAT absolute_x = 0.0;
      PHI_FLOAT absolute_y = 0.0;
      grid_.absoluteCoordinates(x, y, absolute_x, absolute_y);
      targets.push_back(
          adaptive_grid::CanonicalGatherDescriptor::
              from_legacy_absolute_coordinates(
                  static_cast<double>(absolute_x),
                  static_cast<double>(absolute_y)));
    } else {
      targets.push_back(
          adaptive_grid::CanonicalGatherDescriptor::rejected());
    }
  };

  for (PARTICLE* particle : particles1)
    append_particle(
        particle, inputs.beam1_sources, inputs.beam1_targets);
  for (PARTICLE* particle : particles2)
    append_particle(
        particle, inputs.beam2_sources, inputs.beam2_targets);
  return inputs;
}

std::shared_ptr<const adaptive_grid::UniformWindowPlan>
Workspace::make_window_plan(int slice1, int slice2) {
  if (!initialized_ || configuration_.adaptive_grid_mode < 1 ||
      !canonical_lattice_)
    throw std::logic_error(
        "adaptive window planning requires an initialized adaptive mode");
  if (slice1 < 0 || slice1 >= beam1_->number_of_slices() ||
      slice2 < 0 || slice2 >= beam2_->number_of_slices())
    throw std::out_of_range(
        "adaptive window planner received an invalid slice index");
  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  adaptive_grid::CanonicalWindowInputs inputs =
      build_window_inputs(slice1, slice2);
  adaptive_grid::UniformWindowPlan built =
      adaptive_grid::UniformWindowPlan::build(
          *canonical_lattice_, std::move(inputs),
          configuration_.adaptive_window_policy);
  if (built.full_window() &&
      !configuration_.retain_full_window_descriptor_payload)
    built = built.compact_full_window_identity();
  std::shared_ptr<const adaptive_grid::UniformWindowPlan> result(
      new adaptive_grid::UniformWindowPlan(std::move(built)));
  if (!result->empty() && configuration_.adaptive_max_cells != 0U) {
    const std::size_t cells = checked_product(
        result->local_n_x(), result->local_n_y(),
        "adaptive local cell count");
    if (cells > configuration_.adaptive_max_cells) {
      std::ostringstream message;
      message << "adaptive window has " << cells
              << " cells, exceeding adaptive_grid_max_cells="
              << configuration_.adaptive_max_cells;
      throw adaptive_grid::WindowPlanError(
          adaptive_grid::WindowPlanErrorCode::MemoryBudgetExceeded,
          message.str());
    }
  }
  ++adaptive_plan_builds_;
  adaptive_bounds_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}

Workspace::AdaptiveFieldContext* Workspace::find_adaptive_context(
    std::size_t n_x, std::size_t n_y,
    std::size_t refinement_factor) noexcept {
  for (std::size_t index = 0; index < adaptive_contexts_.size(); ++index)
    if (adaptive_contexts_[index]->n_x == n_x &&
        adaptive_contexts_[index]->n_y == n_y &&
        adaptive_contexts_[index]->refinement_factor == refinement_factor)
      return adaptive_contexts_[index].get();
  return NULL;
}

bool Workspace::has_adaptive_class(
    std::size_t n_x, std::size_t n_y,
    std::size_t refinement_factor) const noexcept {
  for (std::size_t index = 0; index < adaptive_contexts_.size(); ++index)
    if (adaptive_contexts_[index]->n_x == n_x &&
        adaptive_contexts_[index]->n_y == n_y &&
        adaptive_contexts_[index]->refinement_factor == refinement_factor)
      return true;
  return false;
}

void Workspace::prebuild_adaptive_class(
    std::size_t n_x, std::size_t n_y, bool wisdom_only,
    std::size_t refinement_factor) {
  if (!initialized_ || configuration_.adaptive_grid_mode < 1 ||
      !canonical_lattice_)
    throw std::logic_error(
        "adaptive context prebuild requires an initialized adaptive mode");
  if (refinement_factor == 0U)
    throw std::invalid_argument("adaptive context refinement factor is zero");
  if (find_adaptive_context(n_x, n_y, refinement_factor)) return;
  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  if (configuration_.adaptive_grid_mode == 3 && adaptive_patch_context_) {
    if (adaptive_patch_context_->estimated_bytes > adaptive_context_bytes_)
      throw std::logic_error(
          "Mode-3 patch eviction byte accounting underflow");
    adaptive_context_bytes_ -= adaptive_patch_context_->estimated_bytes;
    adaptive_patch_context_.reset();
    ++adaptive_context_evictions_;
  }
  const std::size_t global_n_x = checked_product(
      canonical_lattice_->n_x(), refinement_factor,
      "adaptive refined global n_x");
  const std::size_t global_n_y = checked_product(
      canonical_lattice_->n_y(), refinement_factor,
      "adaptive refined global n_y");
  if (n_x > global_n_x || n_y > global_n_y)
    throw std::invalid_argument(
        "adaptive context class exceeds its refined global lattice");
  const std::size_t local_cells = checked_product(
      n_x, n_y, "adaptive context local cells");
  if (configuration_.adaptive_max_cells != 0U &&
      local_cells > configuration_.adaptive_max_cells)
    throw std::runtime_error(
        "adaptive context exceeds adaptive_grid_max_cells before allocation");
  const adaptive_grid::UniformWindowPolicy& policy =
      configuration_.adaptive_window_policy;
  const std::size_t estimate = adaptive_context_estimate(n_x, n_y, policy);
  std::size_t replacement_index = adaptive_contexts_.size();
  std::size_t retained = adaptive_context_bytes_;
  for (std::size_t index = 0U;
       configuration_.adaptive_grid_mode >= 2 &&
       index < adaptive_contexts_.size(); ++index)
    if (adaptive_contexts_[index]->refinement_factor == refinement_factor) {
      replacement_index = index;
      if (adaptive_contexts_[index]->estimated_bytes > retained)
        throw std::logic_error(
            "adaptive uniform context byte accounting underflow");
      retained -= adaptive_contexts_[index]->estimated_bytes;
      break;
    }
  if (estimate > policy.max_bytes || retained > policy.max_bytes - estimate)
    throw std::runtime_error(
        "adaptive FFT context pool exceeds adaptive_grid_max_bytes; "
        "no allocation or fallback was performed");

  // Modes 2/3 retain at most one variable shape for each spacing factor.
  // This makes the preflight bank estimate a hard bound and prevents an
  // unbounded sequence of boundary-clipped encounter shapes from accumulating
  // plans.  Mode 1 retains its established multi-shape prebuilt cache.
  if (replacement_index != adaptive_contexts_.size()) {
    adaptive_contexts_.erase(adaptive_contexts_.begin() + replacement_index);
    adaptive_context_bytes_ = retained;
    ++adaptive_context_evictions_;
  }

  std::unique_ptr<AdaptiveFieldContext> context(
      new AdaptiveFieldContext(
          n_x, n_y,
          grid_.get_field_kernel_factor() *
              static_cast<PHI_FLOAT>(refinement_factor) *
              static_cast<PHI_FLOAT>(refinement_factor),
          grid_.get_delta_x() / static_cast<float>(refinement_factor),
          grid_.get_delta_y() / static_cast<float>(refinement_factor),
          wisdom_only, configuration_.adaptive_grid_mode >= 2,
          refinement_factor));
  context->estimated_bytes = estimate;
  adaptive_contexts_.push_back(std::move(context));
  adaptive_context_bytes_ = retained + estimate;
  ++adaptive_context_builds_;
  adaptive_context_build_ms_ +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
}

Workspace::AdaptiveFieldContext* Workspace::require_adaptive_context(
    const adaptive_grid::FieldLevelGeometry& geometry,
    bool transient_patch_context) {
  if (transient_patch_context) {
    if (adaptive_patch_context_ &&
        adaptive_patch_context_->n_x == geometry.n_x &&
        adaptive_patch_context_->n_y == geometry.n_y &&
        adaptive_patch_context_->refinement_factor ==
            geometry.refinement_factor) {
      ++adaptive_context_hits_;
      return adaptive_patch_context_.get();
    }
    ++adaptive_patch_context_misses_;
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    const std::size_t estimate = adaptive_context_estimate(
        geometry.n_x, geometry.n_y, configuration_.adaptive_window_policy);
    if (adaptive_patch_context_ &&
        adaptive_patch_context_->estimated_bytes > adaptive_context_bytes_)
      throw std::logic_error(
          "Mode-3 transient context byte accounting underflow");
    const std::size_t retained = adaptive_patch_context_
        ? adaptive_context_bytes_ - adaptive_patch_context_->estimated_bytes
        : adaptive_context_bytes_;
    if (estimate > configuration_.adaptive_window_policy.max_bytes ||
        retained > configuration_.adaptive_window_policy.max_bytes - estimate)
      throw std::runtime_error(
          "Mode-3 transient patch context exceeds adaptive_grid_max_bytes");
    // Retire the previous variable-shape context before allocating its
    // replacement.  This makes the configured cap a true allocation
    // high-water bound rather than merely an accounting bound.
    if (adaptive_patch_context_) ++adaptive_context_evictions_;
    adaptive_patch_context_.reset();
    adaptive_context_bytes_ = retained;
    std::unique_ptr<AdaptiveFieldContext> replacement(
        new AdaptiveFieldContext(
            geometry.n_x, geometry.n_y,
            grid_.get_field_kernel_factor() *
                static_cast<PHI_FLOAT>(geometry.refinement_factor) *
                static_cast<PHI_FLOAT>(geometry.refinement_factor),
            grid_.get_delta_x() /
                static_cast<float>(geometry.refinement_factor),
            grid_.get_delta_y() /
                static_cast<float>(geometry.refinement_factor),
            false, true, geometry.refinement_factor));
    replacement->estimated_bytes = estimate;
    adaptive_patch_context_.swap(replacement);
    adaptive_context_bytes_ = retained + estimate;
    ++adaptive_context_builds_;
    adaptive_context_build_ms_ +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    return adaptive_patch_context_.get();
  }
  AdaptiveFieldContext* existing = find_adaptive_context(
      geometry.n_x, geometry.n_y, geometry.refinement_factor);
  if (existing) {
    ++adaptive_context_hits_;
    return existing;
  }
  // Mode 2/3 can discover a refined shape only after the immutable particle
  // snapshot exists. These lazy contexts use FFTW_ESTIMATE: construction is
  // deterministic across lanes and avoids repeated serialized measurements.
  // FFTW's process-global create/destroy lock still protects its planner.
  ++adaptive_uniform_context_misses_;
  prebuild_adaptive_class(geometry.n_x, geometry.n_y, false,
                          geometry.refinement_factor);
  existing = find_adaptive_context(
      geometry.n_x, geometry.n_y, geometry.refinement_factor);
  if (!existing)
    throw std::runtime_error(
        "adaptive refined context construction did not publish a context");
  return existing;
}

Workspace::LevelSolution Workspace::solve_field_level(
    int slice1, int slice2,
    const adaptive_grid::UniformWindowPlan& canonical_plan,
    const adaptive_grid::FieldLevelGeometry& geometry,
    const std::vector<unsigned char>* source_mask1,
    const std::vector<unsigned char>* source_mask2,
    const std::vector<unsigned char>* target_mask1,
    const std::vector<unsigned char>* target_mask2,
    bool transient_patch_context,
    const CoherentSourceHierarchy* source_hierarchy,
    const LevelSolution* registered_density_source) {
  if (!canonical_lattice_ || canonical_plan.empty() ||
      !canonical_plan.descriptor_payload_retained() || !geometry.valid())
    throw std::invalid_argument(
        "changed-spacing field solve received an invalid immutable plan");
  if (canonical_plan.lattice_hash() != canonical_lattice_->hash())
    throw std::runtime_error(
        "changed-spacing field solve rejected a stale canonical lattice");
  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  if (canonical_plan.beam1_sources().size() != particles1.size() ||
      canonical_plan.beam2_sources().size() != particles2.size() ||
      canonical_plan.beam1_targets().size() != particles1.size() ||
      canonical_plan.beam2_targets().size() != particles2.size())
    throw std::runtime_error(
        "changed-spacing plan lost its particle-order descriptor binding");
  const auto validate_mask = [](const std::vector<unsigned char>* mask,
                                std::size_t size, const char* label) {
    if (mask && mask->size() != size)
      throw std::invalid_argument(std::string(label) +
                                  " mask has the wrong particle count");
  };
  validate_mask(source_mask1, particles1.size(), "beam1 source");
  validate_mask(source_mask2, particles2.size(), "beam2 source");
  validate_mask(target_mask1, particles1.size(), "beam1 target");
  validate_mask(target_mask2, particles2.size(), "beam2 target");
  if (!source_hierarchy)
    throw std::invalid_argument(
        "changed-spacing field solve requires an immutable canonical "
        "charge-density hierarchy");
  if (registered_density_source &&
      (!registered_density_source->context ||
       !registered_density_source->geometry.valid()))
    throw std::invalid_argument(
        "changed-spacing patch solve received an invalid registered "
        "density source");

  LevelSolution result;
  result.geometry = geometry;
  const std::size_t cells = geometry.cells();
  if (configuration_.adaptive_max_cells != 0U &&
      cells > configuration_.adaptive_max_cells)
    throw std::runtime_error(
        "changed-spacing field level exceeds adaptive_grid_max_cells");
  result.context = require_adaptive_context(geometry,
                                             transient_patch_context);
  result.fields1.assign(2U * particles1.size(), 0.0f);
  result.fields2.assign(2U * particles2.size(), 0.0f);
  AdaptiveFieldContext& context = *result.context;

  const std::chrono::steady_clock::time_point deposit_start =
      std::chrono::steady_clock::now();
  // reference_cic_v1 is deposited once on its canonical lattice and moved
  // through an exact finite-volume hierarchy. fixed_width_bspline1_v1 instead
  // analytically integrates the same continuous, fixed-physical-width source
  // over every requested lattice. A Mode-3 patch always extracts a registered
  // subwindow from its already-built uniform-fine oracle.
  if (registered_density_source) {
    adaptive_grid::extract_registered_density_window(
        registered_density_source->context->rho1,
        registered_density_source->geometry,
        geometry, context.rho1);
    adaptive_grid::extract_registered_density_window(
        registered_density_source->context->rho2,
        registered_density_source->geometry,
        geometry, context.rho2);
    adaptive_grid::extract_registered_density_window(
        registered_density_source->sampling_s1_1,
        registered_density_source->geometry,
        geometry, result.sampling_s1_1);
    adaptive_grid::extract_registered_density_window(
        registered_density_source->sampling_s2_1,
        registered_density_source->geometry,
        geometry, result.sampling_s2_1);
    adaptive_grid::extract_registered_density_window(
        registered_density_source->sampling_s1_2,
        registered_density_source->geometry,
        geometry, result.sampling_s1_2);
    adaptive_grid::extract_registered_density_window(
        registered_density_source->sampling_s2_2,
        registered_density_source->geometry,
        geometry, result.sampling_s2_2);
  } else if (source_hierarchy->model == "fixed_width_bspline1_v1") {
    context.rho1.assign(cells, PHI_FLOAT(0.0));
    context.rho2.assign(cells, PHI_FLOAT(0.0));
    result.sampling_s1_1.assign(cells, PHI_FLOAT(0.0));
    result.sampling_s2_1.assign(cells, PHI_FLOAT(0.0));
    result.sampling_s1_2.assign(cells, PHI_FLOAT(0.0));
    result.sampling_s2_2.assign(cells, PHI_FLOAT(0.0));

    const auto deposit_fixed_width = [
        &geometry, source_hierarchy](
        const std::vector<adaptive_grid::CanonicalDepositDescriptor>&
            descriptors,
        const std::vector<PARTICLE*>& particles,
        PHI_FLOAT charge,
        const std::vector<unsigned char>* source_mask,
        std::vector<PHI_FLOAT>& rho,
        std::vector<PHI_FLOAT>& sampling_s1,
        std::vector<PHI_FLOAT>& sampling_s2,
        long double& expected_charge,
        long double& expected_x_moment,
        long double& expected_y_moment) {
      if (descriptors.size() != particles.size())
        throw std::logic_error(
            "fixed-width source lost its particle-order binding");
      std::vector<adaptive_grid::BSplineCellIntegral> weights_x;
      std::vector<adaptive_grid::BSplineCellIntegral> weights_y;
      const long double magnitude = std::fabs(
          static_cast<long double>(charge));
      for (std::size_t particle_index = 0U;
           particle_index < descriptors.size(); ++particle_index) {
        if (!descriptors[particle_index].accepted() ||
            (source_mask && !(*source_mask)[particle_index]))
          continue;
        if (!particles[particle_index])
          throw std::logic_error(
              "fixed-width source encountered a null particle");
        float x = 0.0f;
        float y = 0.0f;
        particles[particle_index]->XYposition(x, y);
        if (!std::isfinite(x) || !std::isfinite(y))
          throw std::runtime_error(
              "fixed-width source has a non-finite particle position");
        adaptive_grid::integrate_first_order_bspline_cells(
            static_cast<double>(x), source_hierarchy->source_width_x,
            adaptive_grid::FieldAxis::X, geometry, weights_x);
        adaptive_grid::integrate_first_order_bspline_cells(
            static_cast<double>(y), source_hierarchy->source_width_y,
            adaptive_grid::FieldAxis::Y, geometry, weights_y);
        long double sum_x = 0.0L;
        long double sum_y = 0.0L;
        for (const adaptive_grid::BSplineCellIntegral& weight : weights_x)
          sum_x += weight.weight;
        for (const adaptive_grid::BSplineCellIntegral& weight : weights_y)
          sum_y += weight.weight;
        // A masked local patch may consume only complete fixed-width shapes.
        // Any support crossing its boundary must have been detected while
        // constructing the mask and is rejected here as a second independent
        // guard; accepting a clipped source would break both charge closure
        // and the local Richardson comparison.
        const long double support_tolerance =
            512.0L * std::numeric_limits<double>::epsilon();
        if (std::fabs(sum_x - 1.0L) > support_tolerance ||
            std::fabs(sum_y - 1.0L) > support_tolerance)
          throw std::runtime_error(
              "fixed-width local patch clipped an accepted source shape");
        for (const adaptive_grid::BSplineCellIntegral& weight_x : weights_x)
          for (const adaptive_grid::BSplineCellIntegral& weight_y :
               weights_y) {
            const std::size_t cell =
                weight_x.local_index * geometry.n_y + weight_y.local_index;
            const long double normalized_weight =
                static_cast<long double>(weight_x.weight) * weight_y.weight;
            const PHI_FLOAT signed_contribution = static_cast<PHI_FLOAT>(
                static_cast<long double>(charge) * normalized_weight);
            const PHI_FLOAT absolute_contribution = static_cast<PHI_FLOAT>(
                magnitude * normalized_weight);
            rho[cell] += signed_contribution;
            sampling_s1[cell] += absolute_contribution;
            sampling_s2[cell] +=
                absolute_contribution * absolute_contribution;
          }
        expected_charge += charge;
        expected_x_moment += static_cast<long double>(charge) * x;
        expected_y_moment += static_cast<long double>(charge) * y;
      }
    };
    long double expected_charge1 = 0.0L;
    long double expected_x_moment1 = 0.0L;
    long double expected_y_moment1 = 0.0L;
    long double expected_charge2 = 0.0L;
    long double expected_x_moment2 = 0.0L;
    long double expected_y_moment2 = 0.0L;
    deposit_fixed_width(
        canonical_plan.beam1_sources(), particles1,
        configuration_.macro_charge1, source_mask1, context.rho1,
        result.sampling_s1_1, result.sampling_s2_1,
        expected_charge1, expected_x_moment1, expected_y_moment1);
    deposit_fixed_width(
        canonical_plan.beam2_sources(), particles2,
        configuration_.macro_charge2, source_mask2, context.rho2,
        result.sampling_s1_2, result.sampling_s2_2,
        expected_charge2, expected_x_moment2, expected_y_moment2);

    // The density arrays store integrated cell charge. Their cell-centre
    // quadrature conserves charge exactly to roundoff, while its centroid can
    // differ from the analytic symmetric source centroid by at most half a
    // cell in each axis. Report that projection displacement; fail closed if
    // either the conservation or geometric bound is violated.
    const auto audit_fixed_width = [
        &geometry, source_hierarchy](
        const std::vector<PHI_FLOAT>& rho,
        long double accepted_charge,
        long double accepted_x_moment,
        long double accepted_y_moment,
        double& charge_error,
        double& normalized_moment_error) {
      long double deposited_charge = 0.0L;
      long double deposited_x_moment = 0.0L;
      long double deposited_y_moment = 0.0L;
      for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
        const long double x =
            (static_cast<long double>(geometry.origin_x) + ix + 0.5L -
             geometry.offset_x) * geometry.delta_x;
        for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
          const PHI_FLOAT value = rho[ix * geometry.n_y + iy];
          if (!std::isfinite(value))
            throw std::runtime_error(
                "fixed-width B-spline deposit produced non-finite charge");
          const long double y =
              (static_cast<long double>(geometry.origin_y) + iy + 0.5L -
               geometry.offset_y) * geometry.delta_y;
          deposited_charge += value;
          deposited_x_moment += static_cast<long double>(value) * x;
          deposited_y_moment += static_cast<long double>(value) * y;
        }
      }
      const long double charge_scale = std::max(
          std::fabs(accepted_charge),
          static_cast<long double>(std::numeric_limits<double>::min()));
      const long double relative_charge_error =
          std::fabs(deposited_charge - accepted_charge) / charge_scale;
      const long double x_centroid_error =
          std::fabs(deposited_x_moment - accepted_x_moment) / charge_scale;
      const long double y_centroid_error =
          std::fabs(deposited_y_moment - accepted_y_moment) / charge_scale;
      const long double roundoff_guard =
          256.0L * std::numeric_limits<double>::epsilon();
      if (relative_charge_error > 5.0e-12L ||
          x_centroid_error >
              0.5L * geometry.delta_x * (1.0L + roundoff_guard) ||
          y_centroid_error >
              0.5L * geometry.delta_y * (1.0L + roundoff_guard))
        throw std::runtime_error(
            "fixed-width B-spline deposit failed charge/centroid closure");
      charge_error = static_cast<double>(relative_charge_error);
      normalized_moment_error = static_cast<double>(std::max(
          x_centroid_error /
              std::max(static_cast<long double>(
                           source_hierarchy->source_width_x),
                       static_cast<long double>(geometry.delta_x)),
          y_centroid_error /
              std::max(static_cast<long double>(
                           source_hierarchy->source_width_y),
                       static_cast<long double>(geometry.delta_y))));
    };
    double charge_error1 = 0.0;
    double charge_error2 = 0.0;
    double moment_error1 = 0.0;
    double moment_error2 = 0.0;
    audit_fixed_width(
        context.rho1, expected_charge1,
        expected_x_moment1, expected_y_moment1,
        charge_error1, moment_error1);
    audit_fixed_width(
        context.rho2, expected_charge2,
        expected_x_moment2, expected_y_moment2,
        charge_error2, moment_error2);
    result.maximum_charge_closure = std::max(charge_error1, charge_error2);
    result.maximum_first_moment_error =
        std::max(moment_error1, moment_error2);
  } else {
    const auto transfer_density = [&geometry, source_hierarchy](
        const std::vector<PHI_FLOAT>& source,
        std::vector<PHI_FLOAT>& destination) {
      if (geometry.refinement_factor < source_hierarchy->source_factor)
        return adaptive_grid::restrict_cic_charge_density(
            source, source_hierarchy->canonical_geometry,
            geometry, destination);
      if (geometry.refinement_factor == source_hierarchy->source_factor) {
        adaptive_grid::extract_registered_density_window(
            source, source_hierarchy->canonical_geometry,
            geometry, destination);
        return adaptive_grid::DensityTransferReport();
      }
      if (source_hierarchy->model == "reference_cic_v1") {
        const std::size_t ratio = geometry.refinement_factor /
                                  source_hierarchy->source_factor;
        if (static_cast<std::size_t>(geometry.origin_x) % ratio != 0U ||
            static_cast<std::size_t>(geometry.origin_y) % ratio != 0U ||
            geometry.n_x % ratio != 0U || geometry.n_y % ratio != 0U)
          throw std::runtime_error(
              "reference CIC prolongation window is not source-cell aligned");
        adaptive_grid::FieldLevelGeometry source_window =
            source_hierarchy->canonical_geometry;
        source_window.origin_x = geometry.origin_x /
                                 static_cast<adaptive_grid::CanonicalIndex>(
                                     ratio);
        source_window.origin_y = geometry.origin_y /
                                 static_cast<adaptive_grid::CanonicalIndex>(
                                     ratio);
        source_window.n_x = geometry.n_x / ratio;
        source_window.n_y = geometry.n_y / ratio;
        std::vector<PHI_FLOAT> cropped_source;
        adaptive_grid::extract_registered_density_window(
            source, source_hierarchy->canonical_geometry,
            source_window, cropped_source);
        return adaptive_grid::prolong_cic_charge_density(
            cropped_source, source_window, geometry, destination);
      }
      return adaptive_grid::prolong_cic_charge_density(
          source, source_hierarchy->canonical_geometry,
          geometry, destination);
    };
    const adaptive_grid::DensityTransferReport transfer1 =
        transfer_density(source_hierarchy->rho1, context.rho1);
    const adaptive_grid::DensityTransferReport transfer2 =
        transfer_density(source_hierarchy->rho2, context.rho2);
    result.maximum_charge_closure = std::max(
        transfer1.relative_charge_error, transfer2.relative_charge_error);
    result.maximum_first_moment_error = std::max(
        transfer1.normalized_first_moment_error,
        transfer2.normalized_first_moment_error);
    // Below/at a reference source, S1/S2 were built from the actual grouped
    // particle contributions for that field factor.  Above it, subdivision
    // gives each child S1/r^2 and S2/r^4, preserving Kish Neff exactly.
    const auto prolong_sampling = [&geometry](
        const std::vector<PHI_FLOAT>& canonical_s1,
        const std::vector<PHI_FLOAT>& canonical_s2,
        const adaptive_grid::FieldLevelGeometry& canonical_geometry,
        std::vector<PHI_FLOAT>& level_s1,
        std::vector<PHI_FLOAT>& level_s2) {
      (void)adaptive_grid::prolong_cic_charge_density(
          canonical_s1, canonical_geometry, geometry, level_s1);
      (void)adaptive_grid::prolong_cic_charge_density(
          canonical_s2, canonical_geometry, geometry, level_s2);
      const std::size_t ratio = geometry.refinement_factor /
                                canonical_geometry.refinement_factor;
      const double inverse_area = 1.0 /
          static_cast<double>(checked_product(
              ratio, ratio, "sampling hierarchy child count"));
      for (PHI_FLOAT& value : level_s2) value *= inverse_area;
    };
    if (source_hierarchy->model == "reference_cic_v1" &&
        geometry.refinement_factor <= source_hierarchy->source_factor) {
      const SamplingHierarchyLevel* sampling = NULL;
      for (const SamplingHierarchyLevel& candidate :
           source_hierarchy->sampling_levels)
        if (candidate.factor == geometry.refinement_factor) {
          sampling = &candidate;
          break;
        }
      if (!sampling)
        throw std::runtime_error(
            "reference CIC lacks actual-source sampling for field factor");
      adaptive_grid::extract_registered_density_window(
          sampling->s1_1, sampling->geometry,
          geometry, result.sampling_s1_1);
      adaptive_grid::extract_registered_density_window(
          sampling->s2_1, sampling->geometry,
          geometry, result.sampling_s2_1);
      adaptive_grid::extract_registered_density_window(
          sampling->s1_2, sampling->geometry,
          geometry, result.sampling_s1_2);
      adaptive_grid::extract_registered_density_window(
          sampling->s2_2, sampling->geometry,
          geometry, result.sampling_s2_2);
    } else if (source_hierarchy->model == "reference_cic_v1") {
      const SamplingHierarchyLevel* source_sampling = NULL;
      for (const SamplingHierarchyLevel& candidate :
           source_hierarchy->sampling_levels)
        if (candidate.factor == source_hierarchy->source_factor) {
          source_sampling = &candidate;
          break;
        }
      if (!source_sampling)
        throw std::runtime_error(
            "reference CIC lacks source-level sampling statistics");
      const std::size_t ratio = geometry.refinement_factor /
                                source_hierarchy->source_factor;
      adaptive_grid::FieldLevelGeometry source_window =
          source_sampling->geometry;
      source_window.origin_x = geometry.origin_x /
                               static_cast<adaptive_grid::CanonicalIndex>(
                                   ratio);
      source_window.origin_y = geometry.origin_y /
                               static_cast<adaptive_grid::CanonicalIndex>(
                                   ratio);
      source_window.n_x = geometry.n_x / ratio;
      source_window.n_y = geometry.n_y / ratio;
      std::vector<PHI_FLOAT> s1_1;
      std::vector<PHI_FLOAT> s2_1;
      std::vector<PHI_FLOAT> s1_2;
      std::vector<PHI_FLOAT> s2_2;
      adaptive_grid::extract_registered_density_window(
          source_sampling->s1_1, source_sampling->geometry,
          source_window, s1_1);
      adaptive_grid::extract_registered_density_window(
          source_sampling->s2_1, source_sampling->geometry,
          source_window, s2_1);
      adaptive_grid::extract_registered_density_window(
          source_sampling->s1_2, source_sampling->geometry,
          source_window, s1_2);
      adaptive_grid::extract_registered_density_window(
          source_sampling->s2_2, source_sampling->geometry,
          source_window, s2_2);
      prolong_sampling(
          s1_1, s2_1, source_window,
          result.sampling_s1_1, result.sampling_s2_1);
      prolong_sampling(
          s1_2, s2_2, source_window,
          result.sampling_s1_2, result.sampling_s2_2);
    } else {
      prolong_sampling(
          source_hierarchy->sampling_s1_1,
          source_hierarchy->sampling_s2_1,
          source_hierarchy->canonical_geometry,
          result.sampling_s1_1, result.sampling_s2_1);
      prolong_sampling(
          source_hierarchy->sampling_s1_2,
          source_hierarchy->sampling_s2_2,
          source_hierarchy->canonical_geometry,
          result.sampling_s1_2, result.sampling_s2_2);
    }
  }
  if (result.sampling_s1_1.size() != cells ||
      result.sampling_s2_1.size() != cells ||
      result.sampling_s1_2.size() != cells ||
      result.sampling_s2_2.size() != cells)
    throw std::runtime_error(
        "coherent sampling hierarchy has the wrong field-level extent");
  const double source_transfer_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - deposit_start).count();
  if (!registered_density_source &&
      source_hierarchy->model == "fixed_width_bspline1_v1")
    result.source_build_ms = source_transfer_ms;
  else if (!registered_density_source &&
      geometry.refinement_factor < source_hierarchy->source_factor)
    result.source_restrict_ms = source_transfer_ms;
  else if (!registered_density_source &&
           geometry.refinement_factor > source_hierarchy->source_factor)
    result.source_prolong_ms = source_transfer_ms;

  // For registered raster sources, masks control encounter accounting only:
  // the patch RHS and S1/S2 arrays are exact control-volume restrictions.  A
  // terminal fixed-width patch instead uses the same mask for its direct
  // continuous-source integration; the earlier complete-support check makes
  // that local ownership mathematically closed at both patch spacings.
  const auto count_sources = [](
      const std::vector<adaptive_grid::CanonicalDepositDescriptor>&
          canonical_descriptors,
      const std::vector<unsigned char>* mask,
      std::size_t& in, std::size_t& out) {
    for (std::size_t particle_index = 0;
         particle_index < canonical_descriptors.size(); ++particle_index) {
      if (!canonical_descriptors[particle_index].accepted()) {
        ++out;
        continue;
      }
      if (mask && !(*mask)[particle_index]) continue;
      ++in;
    }
  };
  count_sources(canonical_plan.beam1_sources(), source_mask1,
                result.beam1_in, result.beam1_out);
  count_sources(canonical_plan.beam2_sources(), source_mask2,
                result.beam2_in, result.beam2_out);
  adaptive_deposit_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - deposit_start).count();

  long double fine_lumi_dot = 0.0L;
  for (std::size_t cell = 0U; cell < cells; ++cell)
    fine_lumi_dot += static_cast<long double>(context.rho1[cell]) *
                     static_cast<long double>(context.rho2[cell]);
  if (!finite_value(fine_lumi_dot) ||
      std::fabs(fine_lumi_dot) >
          static_cast<long double>(std::numeric_limits<double>::max()))
    throw std::runtime_error(
        "changed-spacing luminosity dot product overflowed");
  result.raw_lumi_dot = static_cast<double>(fine_lumi_dot);

  std::vector<double> indicator_charge(cells, 0.0);
  std::vector<double> indicator_overlap(cells, 0.0);
  for (std::size_t cell = 0U; cell < cells; ++cell) {
    indicator_charge[cell] = std::fabs(context.rho1[cell]) +
                             std::fabs(context.rho2[cell]);
    indicator_overlap[cell] = std::sqrt(std::fabs(
        static_cast<double>(context.rho1[cell]) *
        static_cast<double>(context.rho2[cell])));
  }
  const auto smooth_indicator = [&geometry](std::vector<double>& values) {
    std::vector<double> next(values.size(), 0.0);
    for (std::size_t ix = 0U; ix < geometry.n_x; ++ix)
      for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
        double sum = 0.0;
        double normalization = 0.0;
        for (int dx = -1; dx <= 1; ++dx) {
          const long x = static_cast<long>(ix) + dx;
          if (x < 0 || x >= static_cast<long>(geometry.n_x)) continue;
          const double wx = dx == 0 ? 2.0 : 1.0;
          for (int dy = -1; dy <= 1; ++dy) {
            const long y = static_cast<long>(iy) + dy;
            if (y < 0 || y >= static_cast<long>(geometry.n_y)) continue;
            const double wy = dy == 0 ? 2.0 : 1.0;
            const double weight = wx * wy;
            sum += weight * values[static_cast<std::size_t>(x) *
                                   geometry.n_y +
                                   static_cast<std::size_t>(y)];
            normalization += weight;
          }
        }
        next[ix * geometry.n_y + iy] = sum / normalization;
      }
    values.swap(next);
  };
  for (unsigned int pass = 0U;
       pass < configuration_.adaptive_indicator_smoothing_passes; ++pass) {
    smooth_indicator(indicator_charge);
    smooth_indicator(indicator_overlap);
  }
  const double maximum_overlap = *std::max_element(
      indicator_overlap.begin(), indicator_overlap.end());
  std::vector<double> occupied_counts;
  double critical_minimum_neff = std::numeric_limits<double>::infinity();
  const double rho_floor = configuration_.adaptive_rho_floor;
  for (std::size_t ix = 1U; ix + 1U < geometry.n_x; ++ix) {
    for (std::size_t iy = 1U; iy + 1U < geometry.n_y; ++iy) {
      const std::size_t cell = ix * geometry.n_y + iy;
      const double charge = indicator_charge[cell];
      const double overlap = indicator_overlap[cell];
      const double curvature =
          std::fabs(indicator_charge[(ix + 1U) * geometry.n_y + iy] -
                    2.0 * charge +
                    indicator_charge[(ix - 1U) * geometry.n_y + iy]) +
          std::fabs(indicator_charge[cell + 1U] - 2.0 * charge +
                    indicator_charge[cell - 1U]);
      const double overlap_weight = maximum_overlap > rho_floor
          ? overlap / maximum_overlap : 0.0;
      result.charge_curvature_score = std::max(
          result.charge_curvature_score,
          overlap_weight * curvature /
              (charge + rho_floor + std::numeric_limits<double>::min()));
      if (overlap > 0.1 * maximum_overlap &&
          result.sampling_s1_1[cell] > 0.0 &&
          result.sampling_s1_2[cell] > 0.0) {
        const double neff1 = adaptive_grid::kish_effective_sample_size(
            result.sampling_s1_1[cell], result.sampling_s2_1[cell]);
        const double neff2 = adaptive_grid::kish_effective_sample_size(
            result.sampling_s1_2[cell], result.sampling_s2_2[cell]);
        const double neff = std::min(neff1, neff2);
        occupied_counts.push_back(neff);
        if (overlap > 0.5 * maximum_overlap)
          critical_minimum_neff = std::min(critical_minimum_neff, neff);
      }
    }
  }
  if (!occupied_counts.empty()) {
    const std::size_t quantile = std::min(
        occupied_counts.size() - 1U, occupied_counts.size() / 10U);
    std::nth_element(occupied_counts.begin(),
                     occupied_counts.begin() + quantile,
                     occupied_counts.end());
    result.effective_particles_per_cell = occupied_counts[quantile];
    result.minimum_effective_particles_per_cell =
        finite_value(critical_minimum_neff)
            ? critical_minimum_neff
            : result.effective_particles_per_cell;
  }

  const std::chrono::steady_clock::time_point solve_start =
      std::chrono::steady_clock::now();
  context.field.fold_fft(context.rho1.data(), context.rho2.data(),
                         configuration_.charge_sign);
  adaptive_solve_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - solve_start).count();
  ++fft_encounters_;
  if (configuration_.adaptive_grid_mode >= 2) ++mode2_field_solves_;

  // A normalized Hessian indicator complements the a-posteriori Richardson
  // gate.  The common global field scale avoids singular pointwise ratios at
  // symmetry zeros, while overlap weighting prevents empty tails from
  // driving refinement.
  const auto field_gradient_indicator = [this, &geometry,
                                         &indicator_overlap,
                                         maximum_overlap](const PHI_FLOAT* phi) {
    double maximum_field = 0.0;
    for (std::size_t ix = 1U; ix + 1U < geometry.n_x; ++ix)
      for (std::size_t iy = 1U; iy + 1U < geometry.n_y; ++iy) {
        const std::size_t cell = ix * geometry.n_y + iy;
        const double gx = (phi[(ix + 1U) * geometry.n_y + iy] -
                           phi[(ix - 1U) * geometry.n_y + iy]) /
                          (2.0 * geometry.delta_x);
        const double gy = (phi[cell + 1U] - phi[cell - 1U]) /
                          (2.0 * geometry.delta_y);
        maximum_field = std::max(maximum_field, std::hypot(gx, gy));
      }
    const double scale = std::max(
        maximum_field, configuration_.adaptive_field_floor);
    const double inverse_dx2 = 1.0 /
        (static_cast<double>(geometry.delta_x) * geometry.delta_x);
    const double inverse_dy2 = 1.0 /
        (static_cast<double>(geometry.delta_y) * geometry.delta_y);
    double score = 0.0;
    for (std::size_t ix = 1U; ix + 1U < geometry.n_x; ++ix)
      for (std::size_t iy = 1U; iy + 1U < geometry.n_y; ++iy) {
        const std::size_t cell = ix * geometry.n_y + iy;
        const double overlap = indicator_overlap[cell];
        const double overlap_weight = maximum_overlap > 0.0
            ? overlap / maximum_overlap : 0.0;
        const double dxx =
            (phi[(ix + 1U) * geometry.n_y + iy] - 2.0 * phi[cell] +
             phi[(ix - 1U) * geometry.n_y + iy]) * inverse_dx2;
        const double dyy =
            (phi[cell + 1U] - 2.0 * phi[cell] + phi[cell - 1U]) *
            inverse_dy2;
        const double dxy =
            (phi[(ix + 1U) * geometry.n_y + iy + 1U] -
             phi[(ix + 1U) * geometry.n_y + iy - 1U] -
             phi[(ix - 1U) * geometry.n_y + iy + 1U] +
             phi[(ix - 1U) * geometry.n_y + iy - 1U]) /
            (4.0 * geometry.delta_x * geometry.delta_y);
        const double x_variation = geometry.delta_x * std::hypot(dxx, dxy);
        const double y_variation = geometry.delta_y * std::hypot(dxy, dyy);
        score = std::max(score, overlap_weight *
            std::max(x_variation, y_variation) / scale);
      }
    return score;
  };
  result.field_gradient_score = std::max(
      field_gradient_indicator(context.field.get_phi(1)),
      field_gradient_indicator(context.field.get_phi(2)));

  const std::chrono::steady_clock::time_point gather_start =
      std::chrono::steady_clock::now();
  const adaptive_grid::FieldOperator field_operator =
      adaptive_grid::parse_field_operator(
          configuration_.adaptive_field_operator);
  const auto gather = [&geometry, field_operator](
      const std::vector<PARTICLE*>& particles,
      const std::vector<adaptive_grid::CanonicalGatherDescriptor>&
          canonical_descriptors,
      const std::vector<unsigned char>* mask,
      const PHI_FLOAT* phi, std::vector<float>& fields) {
    for (std::size_t particle_index = 0U;
         particle_index < particles.size(); ++particle_index) {
      if (!canonical_descriptors[particle_index].accepted() ||
          (mask && !(*mask)[particle_index]))
        continue;
      float x = 0.0f;
      float y = 0.0f;
      particles[particle_index]->XYposition(x, y);
      const adaptive_grid::LevelGatherDescriptor descriptor =
          adaptive_grid::map_gather_to_level(
              x, y, geometry, field_operator);
      if (!descriptor.accepted)
        throw std::runtime_error(
            "changed-spacing target stencil escaped its planned level");
      const std::array<float, 2> field =
          adaptive_grid::gather_level_field(
              phi, geometry, descriptor, field_operator);
      fields[2U * particle_index] = field[0];
      fields[2U * particle_index + 1U] = field[1];
    }
  };
  gather(particles1, canonical_plan.beam1_targets(), target_mask1,
         context.field.get_phi(2), result.fields1);
  gather(particles2, canonical_plan.beam2_targets(), target_mask2,
         context.field.get_phi(1), result.fields2);
  adaptive_gather_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - gather_start).count();
  checked_add_counter(adaptive_local_cells_, cells,
                      "adaptive solved field-cell visit");
  checked_add_counter(
      adaptive_padded_cells_,
      checked_product(checked_product(geometry.n_x, 2U,
                                      "adaptive solved padded x"),
                      checked_product(geometry.n_y, 2U,
                                      "adaptive solved padded y"),
                      "adaptive solved padded cells"),
      "adaptive solved padded-cell visit");
  return result;
}

void Workspace::prepare_adaptive(
    int slice1, int slice2, FieldPrefetch& output) {
  if (!output.window_plan)
    output.window_plan = make_window_plan(slice1, slice2);
  const adaptive_grid::UniformWindowPlan& plan = *output.window_plan;
  if (!plan.descriptor_payload_retained())
    throw std::runtime_error(
        "cropped adaptive execution received a compact identity-only plan");
  if (!canonical_lattice_ ||
      plan.lattice_hash() != canonical_lattice_->hash())
    throw std::runtime_error(
        "adaptive result rejected: canonical lattice hash mismatch");

  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  output.fields1.assign(2U * particles1.size(), 0.0f);
  output.fields2.assign(2U * particles2.size(), 0.0f);
  output.beam1_in = 0U;
  output.beam1_out = 0U;
  output.beam2_in = 0U;
  output.beam2_out = 0U;
  output.fine_lumi_dot = 0.0;
  output.result_geometry_hash = 0U;

  if (!plan.empty()) {
    AdaptiveFieldContext* context = find_adaptive_context(
        plan.local_n_x(), plan.local_n_y());
    if (!context)
      throw std::runtime_error(
          "adaptive FFT context cold miss inside an encounter; prebuild "
          "the complete wave class set before field dispatch");
    ++adaptive_context_hits_;
    const std::size_t cells = checked_product(
        plan.local_n_x(), plan.local_n_y(),
        "adaptive execution cells");
    const std::size_t global_cells = checked_product(
        canonical_lattice_->n_x(), canonical_lattice_->n_y(),
        "canonical execution cells");

    const std::chrono::steady_clock::time_point deposit_start =
        std::chrono::steady_clock::now();
    std::fill(context->rho1.begin(), context->rho1.end(), PHI_FLOAT(0.0));
    std::fill(context->rho2.begin(), context->rho2.end(), PHI_FLOAT(0.0));
    checked_add_counter(dense_clear_visits_, global_cells,
                        "adaptive dense-clear visit");
    checked_add_counter(dense_clear_visits_, global_cells,
                        "adaptive dense-clear visit");
    checked_add_counter(actual_clear_visits_, cells,
                        "adaptive actual-clear visit");
    checked_add_counter(actual_clear_visits_, cells,
                        "adaptive actual-clear visit");

    const auto deposit = [this, &plan](
        const std::vector<adaptive_grid::CanonicalDepositDescriptor>&
            descriptors,
        PHI_FLOAT charge,
        PHI_FLOAT* rho, std::size_t& in, std::size_t& out) {
      for (const adaptive_grid::CanonicalDepositDescriptor& descriptor :
           descriptors) {
        if (descriptor.accepted()) {
          grid_.assignChargeToCICTranslated(
              rho, checked_int_dimension(plan.local_n_x(), plan.local_n_y(),
                                         "local n_x"),
              checked_int_dimension(plan.local_n_y(), plan.local_n_x(),
                                         "local n_y"),
              checked_int_index(
                  static_cast<std::size_t>(plan.origin_x()), "origin x"),
              checked_int_index(
                  static_cast<std::size_t>(plan.origin_y()), "origin y"),
              static_cast<int>(descriptor.base_x()),
              static_cast<int>(descriptor.base_y()),
              descriptor.weight_x(), descriptor.weight_y(), charge);
          ++in;
        } else {
          ++out;
        }
      }
    };
    if (plan.beam1_sources().size() != particles1.size() ||
        plan.beam2_sources().size() != particles2.size())
      throw std::runtime_error(
          "adaptive source snapshot particle-order binding is inconsistent");
    deposit(plan.beam1_sources(), configuration_.macro_charge1,
            context->rho1.data(), output.beam1_in, output.beam1_out);
    deposit(plan.beam2_sources(), configuration_.macro_charge2,
            context->rho2.data(), output.beam2_in, output.beam2_out);
    adaptive_deposit_ms_ +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - deposit_start).count();

    float fine_lumi_dot = 0.0f;
    checked_add_counter(dense_dot_visits_, global_cells,
                        "adaptive dense-dot visit");
    checked_add_counter(actual_dot_visits_, cells,
                        "adaptive actual-dot visit");
    for (std::size_t cell = 0; cell < cells; ++cell)
      fine_lumi_dot += context->rho1[cell] * context->rho2[cell];
    output.fine_lumi_dot = static_cast<double>(fine_lumi_dot);

    const std::chrono::steady_clock::time_point solve_start =
        std::chrono::steady_clock::now();
    context->field.fold_fft(
        context->rho1.data(), context->rho2.data(),
        configuration_.charge_sign);
    adaptive_solve_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - solve_start).count();
    ++fft_encounters_;

    const int local_n_x = checked_int_dimension(
        plan.local_n_x(), plan.local_n_y(), "local n_x");
    const int local_n_y = checked_int_dimension(
        plan.local_n_y(), plan.local_n_x(), "local n_y");
    const int origin_x = static_cast<int>(plan.origin_x());
    const int origin_y = static_cast<int>(plan.origin_y());
    const std::chrono::steady_clock::time_point gather_start =
        std::chrono::steady_clock::now();
    const auto gather = [this, local_n_x, local_n_y, origin_x, origin_y](
        const std::vector<adaptive_grid::CanonicalGatherDescriptor>&
            descriptors,
        const PHI_FLOAT* phi,
        float* fields) {
      for (std::size_t particle_index = 0;
           particle_index < descriptors.size(); ++particle_index) {
        const adaptive_grid::CanonicalGatherDescriptor& descriptor =
            descriptors[particle_index];
        if (!descriptor.accepted()) continue;
        const TRIVECTOR field =
            grid_.ElectricFieldCICTranslatedCanonical(
                phi, local_n_x, local_n_y, origin_x, origin_y,
                static_cast<int>(descriptor.y_base_x()),
                static_cast<int>(descriptor.y_base_y()),
                descriptor.y_weight(),
                static_cast<int>(descriptor.x_base_x()),
                static_cast<int>(descriptor.x_base_y()),
                descriptor.x_weight(), descriptor.fractional_x(),
                descriptor.fractional_y());
        fields[2U * particle_index] = field(0);
        fields[2U * particle_index + 1U] = field(1);
      }
    };
    if (plan.beam1_targets().size() != particles1.size() ||
        plan.beam2_targets().size() != particles2.size())
      throw std::runtime_error(
          "adaptive target snapshot particle-order binding is inconsistent");
    gather(plan.beam1_targets(), context->field.get_phi(2),
           output.fields1.data());
    gather(plan.beam2_targets(), context->field.get_phi(1),
           output.fields2.data());
    adaptive_gather_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gather_start).count();

    ++adaptive_encounters_;
    if (plan.full_window()) ++adaptive_full_window_encounters_;
    checked_add_counter(adaptive_required_cells_,
                        plan.required_node_count(),
                        "adaptive required-cell visit");
    checked_add_counter(adaptive_local_cells_, cells,
                        "adaptive local-cell visit");
    const std::size_t padded_cells = checked_product(
        plan.padded_n_x(), plan.padded_n_y(),
        "adaptive padded-cell visit");
    checked_add_counter(adaptive_padded_cells_, padded_cells,
                        "adaptive padded-cell visit");
    checked_add_counter(adaptive_global_cells_avoided_,
                        global_cells - cells,
                        "adaptive avoided-cell visit");
  } else {
    output.beam1_out = particles1.size();
    output.beam2_out = particles2.size();
    ++adaptive_encounters_;
  }

  if (output.beam1_in + output.beam2_in !=
      plan.accepted_source_count())
    throw std::runtime_error(
        "adaptive deposition acceptance disagrees with immutable plan");
  output.lattice_hash = plan.lattice_hash();
  output.bound_slice1 = slice1;
  output.bound_slice2 = slice2;
  output.bound_particle_count1 = particles1.size();
  output.bound_particle_count2 = particles2.size();
  output.result_binding_hash = field_binding_hash(
      slice1, slice2, particles1, particles2,
      output.lattice_hash, plan.plan_hash(), output.result_geometry_hash,
      output.source_model_tag, output.source_reference_factor,
      output.source_density_hash, output.source_sampling_hash,
      output.source_sampling_hierarchy_hash);
}

// Initialize result telemetry and bind the encounter-tail budget.
void Workspace::reset_resolution_output(ResolutionEncounterBinding& binding) {
  const int slice1 = binding.slice1, slice2 = binding.slice2;
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  auto& output = binding.output;
  output.fields1.assign(2U * particles1.size(), 0.0f);
  output.fields2.assign(2U * particles2.size(), 0.0f);
  output.fine_lumi_dot = 0.0;
  output.selected_spacing_factor = 1U;
  output.selected_grid_n_x = 0U;
  output.selected_grid_n_y = 0U;
  output.selected_delta_x = 0.0;
  output.selected_delta_y = 0.0;
  output.estimated_field_error = 0.0;
  output.field_difference_rms = 0.0;
  output.field_difference_linf = 0.0;
  output.vector_difference_rms = 0.0;
  output.vector_difference_quantile = 0.0;
  output.vector_difference_maximum = 0.0;
  output.vector_mixed_difference_maximum = 0.0;
  output.vector_absolute_difference_rms = 0.0;
  output.vector_absolute_difference_quantile = 0.0;
  output.vector_absolute_difference_maximum = 0.0;
  output.vector_catastrophe_targets = 0U;
  output.vector_catastrophe_exposure = 0.0;
  output.vector_quantile_probability =
      configuration_.adaptive_error_quantile;
  output.vector_accepted_targets = 0U;
  output.vector_above_target = 0U;
  output.vector_above_target_exposure = 0.0;
  output.vector_maximum_beam_above_target_exposure = 0.0;
  output.vector_catastrophe_guard_passed = true;
  output.noise_aware_stop_triggered = false;
  output.noise_aware_stop_factor = 0U;
  output.noise_aware_stop_quantile_root_limited = false;
  output.noise_aware_stop_noncontracting = false;
  output.vector_distinct_roots = 0U;
  output.vector_minimum_quantile_roots = 0U;
  output.tail_field_sensitivity = 0.0;
  output.tail_field_sensitivity_previous_pair = 0.0;
  output.tail_field_sensitivity_latest_pair = 0.0;
  output.tail_uniform_lumi_measure = 0.0;
  output.charge_curvature_score = 0.0;
  output.field_gradient_score = 0.0;
  output.effective_particles_per_cell = 0.0;
  output.minimum_effective_particles_per_cell = 0.0;
  output.source_reference_factor =
      configuration_.adaptive_source_model == "reference_cic_v1"
          ? configuration_.adaptive_source_reference_nx /
                static_cast<std::size_t>(configuration_.n_x)
          : configuration_.adaptive_source_model ==
                    "fixed_width_bspline1_v1"
                ? 0U
                : 1U;
  output.source_model_tag =
      configuration_.adaptive_source_model == "reference_cic_v1"
          ? UINT64_C(0x5245464349435631)  // "REFCICV1"
          : configuration_.adaptive_source_model == "fixed_width_bspline1_v1"
                ? UINT64_C(0x4253504c31563100)  // "BSPL1V1\0"
                : UINT64_C(0x43414e43454c4c31); // "CANCELL1"
  output.source_density_hash = 0U;
  output.source_sampling_hash = 0U;
  output.source_sampling_hierarchy_hash = 0U;
  output.adaptive_macroparticle_mode = static_cast<int>(
      configuration_.adaptive_macroparticles.mode);
  output.adaptive_macroparticle_roots1 = 0U;
  output.adaptive_macroparticle_roots2 = 0U;
  output.adaptive_macroparticle_representatives1 = 0U;
  output.adaptive_macroparticle_representatives2 = 0U;
  output.adaptive_macroparticle_premerge1 = 0U;
  output.adaptive_macroparticle_premerge2 = 0U;
  output.adaptive_macroparticle_many_root1 = 0U;
  output.adaptive_macroparticle_many_root2 = 0U;
  output.adaptive_macroparticle_weight_ess1 = 0.0;
  output.adaptive_macroparticle_weight_ess2 = 0.0;
  output.adaptive_macroparticle_clone_support1 = 0.0;
  output.adaptive_macroparticle_clone_support2 = 0.0;
  output.adaptive_macroparticle_projection_residual = 0.0;
  output.adaptive_macroparticle_hash1 = 0U;
  output.adaptive_macroparticle_hash2 = 0U;
  output.adaptive_macroparticle_revision1 = 0U;
  output.adaptive_macroparticle_revision2 = 0U;
  output.adaptive_macroparticle_committed = false;
  output.source_budget_runtime_verified = false;
  output.source_build_ms = 0.0;
  output.source_restrict_ms = 0.0;
  output.source_prolong_ms = 0.0;
  output.source_maximum_charge_closure = 0.0;
  output.source_maximum_projection_moment_shift = 0.0;
  output.source_lumi_oracle_error = 0.0;
  output.source_maximum_lumi_oracle_error = 0.0;
  output.error_target_met = true;
  output.noise_limited = false;
  const std::size_t beam2_slices = static_cast<std::size_t>(
      beam2_->number_of_slices());
  const std::size_t encounter_tail_index =
      static_cast<std::size_t>(slice1) * beam2_slices +
      static_cast<std::size_t>(slice2);
  if (encounter_tail_index >= encounter_tail_exemption_mask_.size() ||
      encounter_tail_index >= encounter_population_weights_.size())
    throw std::logic_error(
        "Mode 2/3 encounter-tail binding exceeds its initialized mask");
  output.encounter_tail_budget_eligible =
      encounter_tail_exemption_mask_[encounter_tail_index] != 0U;
  output.encounter_tail_budget_used = false;
  output.encounter_population_weight =
      encounter_population_weights_[encounter_tail_index];
  output.encounter_tail_selected_fraction =
      encounter_tail_budget_selected_fraction_;
  if (output.encounter_tail_budget_eligible)
    ++mode2_tail_budget_eligible_encounters_;
  output.nested_patch_requested = false;
  output.nested_patch_used = false;
  output.nested_patch_promoted_to_uniform = false;
  output.nested_patch_rejection_mask = NestedPatchRejectionNone;
  output.nested_patch_coarse_cells = 0U;
  output.nested_patch_fine_cells = 0U;
  output.nested_patch_coarse_delta_x = 0.0;
  output.nested_patch_coarse_delta_y = 0.0;
  output.nested_patch_fine_delta_x = 0.0;
  output.nested_patch_fine_delta_y = 0.0;
  output.nested_patch_effective_particles_per_cell = 0.0;
  output.nested_patch_minimum_effective_particles_per_cell = 0.0;
  output.nested_patch_residual = 0.0;
  output.nested_patch_field_error_rms = 0.0;
  output.nested_patch_field_error_linf = 0.0;
  output.nested_patch_lumi_error = 0.0;
  output.nested_patch_tail_fraction = 0.0;
  output.nested_patch_straddling_sources = 0U;
  output.nested_patch_iterations = 0U;
  output.terminal_patch_certification_attempted = false;
  output.terminal_patch_certification_accepted = false;
  output.terminal_patch_used = false;
  output.terminal_patch_uniform_fallback = false;
  output.terminal_patch_external_replay_recommended = false;
  output.terminal_patch_rejection_mask = 0U;
  output.terminal_patch_factor_r = 0U;
  output.terminal_patch_factor_r2 = 0U;
  output.terminal_patch_cells_r = 0U;
  output.terminal_patch_cells_r2 = 0U;
  output.terminal_patch_geometry_hash_r = 0U;
  output.terminal_patch_geometry_hash_r2 = 0U;
  output.terminal_patch_source_hash_r = 0U;
  output.terminal_patch_source_hash_r2 = 0U;
  output.terminal_patch_local_target_hash1 = 0U;
  output.terminal_patch_local_target_hash2 = 0U;
  output.terminal_patch_outside_target_hash1 = 0U;
  output.terminal_patch_outside_target_hash2 = 0U;
  output.terminal_patch_outside_unresolved_exposure = 0.0;
  output.terminal_patch_outside_field_error_contribution = 0.0;
  output.terminal_patch_outside_field_maximum = 0.0;
  output.terminal_patch_outside_luminosity_fraction = 0.0;
  output.terminal_patch_outside_luminosity_earlier = 0.0;
  output.terminal_patch_outside_luminosity_previous = 0.0;
  output.terminal_patch_outside_luminosity_parent = 0.0;
  output.terminal_patch_observed_field_order = 0.0;
  output.terminal_patch_observed_outside_field_order = 0.0;
  output.terminal_patch_composite_field_rms = 0.0;
  output.terminal_patch_composite_field_quantile = 0.0;
  output.terminal_patch_composite_field_maximum = 0.0;
  output.terminal_patch_composite_field_bound = 0.0;
  output.terminal_patch_observed_luminosity_order = 0.0;
  output.terminal_patch_observed_outside_luminosity_order = 0.0;
  output.terminal_patch_outside_luminosity_error = 0.0;
  output.terminal_patch_composite_luminosity_bound = 0.0;
  output.terminal_patch_boundary_flux_rms = 0.0;
  output.terminal_patch_boundary_flux_quantile = 0.0;
  output.terminal_patch_boundary_flux_maximum = 0.0;
  output.result_geometry_hash = 0U;
}

// Complete an empty encounter without allocating a field hierarchy.
void Workspace::complete_empty_resolution(
    ResolutionEncounterBinding& binding) {
  const int slice1 = binding.slice1, slice2 = binding.slice2;
  const auto& plan = binding.plan;
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  const auto field_operator = binding.field_operator;
  auto& output = binding.output;
  output.beam1_in = 0U;
  output.beam2_in = 0U;
  output.beam1_out = particles1.size();
  output.beam2_out = particles2.size();
  ++adaptive_encounters_;
  ++mode2_encounters_;
  if (configuration_.adaptive_grid_mode == 3)
    ++mode3_empty_encounters_;
  if (mode2_spacing_selection_counts_.empty())
    throw std::logic_error(
        "empty Mode-2 encounter has no spacing telemetry class");
  ++mode2_spacing_selection_counts_.front();
  checked_add_counter(source_beam1_in_visits_, output.beam1_in,
                      "Mode-2 source beam-1 admitted visit");
  checked_add_counter(source_beam1_out_visits_, output.beam1_out,
                      "Mode-2 source beam-1 rejected visit");
  checked_add_counter(source_beam2_in_visits_, output.beam2_in,
                      "Mode-2 source beam-2 admitted visit");
  checked_add_counter(source_beam2_out_visits_, output.beam2_out,
                      "Mode-2 source beam-2 rejected visit");
  output.lattice_hash = plan.lattice_hash();
  output.bound_slice1 = slice1;
  output.bound_slice2 = slice2;
  output.bound_particle_count1 = particles1.size();
  output.bound_particle_count2 = particles2.size();
  output.result_geometry_hash =
      UINT64_C(0x464545424c454d50) ^
      static_cast<std::uint64_t>(configuration_.adaptive_grid_mode) ^
      (static_cast<std::uint64_t>(field_operator) << 32U);
  output.result_binding_hash = field_binding_hash(
      slice1, slice2, particles1, particles2,
      output.lattice_hash, plan.plan_hash(), output.result_geometry_hash,
      output.source_model_tag, output.source_reference_factor,
      output.source_density_hash, output.source_sampling_hash,
      output.source_sampling_hierarchy_hash);
  return;
}

// Build one declared field level with the source and operator guards.
adaptive_grid::FieldLevelGeometry Workspace::resolution_geometry(
    const ResolutionEncounterBinding& binding,
    std::size_t factor) const {
  const auto& plan = binding.plan;
  const auto field_operator = binding.field_operator;
  std::size_t extra_guard = factor == 1U
      ? 0U
      : checked_product(
            static_cast<std::size_t>(std::max<std::int64_t>(
                0, configuration_.adaptive_window_policy.extra_guard_cells)),
            factor, "adaptive dynamic fine guard");
  if (configuration_.adaptive_grid_mode == 3)
    extra_guard = std::max(
        extra_guard,
        checked_sum(
            checked_sum(
                checked_product(2U, configuration_.adaptive_blocking_factor,
                                "adaptive dynamic blocking guard"),
                configuration_.adaptive_error_buffer_cells,
                "adaptive dynamic error guard"),
            configuration_.adaptive_transition_cells,
            "adaptive dynamic transition guard"));
  adaptive_grid::FieldLevelGeometry geometry =
      adaptive_grid::make_refined_level_geometry(
          *canonical_lattice_, plan, factor, extra_guard,
          configuration_.adaptive_anchor_x,
          configuration_.adaptive_anchor_y, field_operator);
  if (configuration_.adaptive_source_model == "reference_cic_v1") {
    const std::size_t source_factor =
        configuration_.adaptive_source_reference_nx /
        static_cast<std::size_t>(configuration_.n_x);
    geometry = align_geometry_to_source_cells(geometry, source_factor);
  } else if (configuration_.adaptive_source_model ==
             "fixed_width_bspline1_v1") {
    geometry = expand_geometry_for_fixed_width(
        geometry, configuration_.adaptive_source_width_x,
        configuration_.adaptive_source_width_y);
  }
  return geometry;
}

// Publish a selected level after validating its luminosity normalization.
void Workspace::publish_uniform_resolution(
    ResolutionEncounterBinding& binding,
    const std::vector<LevelSolution>& levels,
    std::size_t index) {
  auto& output = binding.output;
  const auto field_operator = binding.field_operator;
  const LevelSolution& level = levels[index];
  output.selected_spacing_factor = level.geometry.refinement_factor;
  output.selected_grid_n_x = level.geometry.n_x;
  output.selected_grid_n_y = level.geometry.n_y;
  output.selected_delta_x = level.geometry.delta_x;
  output.selected_delta_y = level.geometry.delta_y;
  output.charge_curvature_score = level.charge_curvature_score;
  output.field_gradient_score = level.field_gradient_score;
  output.effective_particles_per_cell =
      level.effective_particles_per_cell;
  output.minimum_effective_particles_per_cell =
      level.minimum_effective_particles_per_cell;
  output.fields1 = level.fields1;
  output.fields2 = level.fields2;
  output.beam1_in = levels.front().beam1_in;
  output.beam1_out = levels.front().beam1_out;
  output.beam2_in = levels.front().beam2_in;
  output.beam2_out = levels.front().beam2_out;
  const double factor = static_cast<double>(
      level.geometry.refinement_factor);
  const double raw_roundoff_floor = -64.0 *
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::fabs(level.raw_lumi_dot));
  if (!finite_value(level.raw_lumi_dot) ||
      level.raw_lumi_dot < raw_roundoff_floor)
    throw std::runtime_error(
        "adaptive uniform luminosity dot product is non-finite or negative");
  const double normalized_lumi_dot =
      std::max(0.0, level.raw_lumi_dot) * factor * factor;
  if (!finite_value(normalized_lumi_dot))
    throw std::runtime_error(
        "adaptive uniform luminosity normalization overflowed");
  output.fine_lumi_dot = normalized_lumi_dot;
  output.tail_uniform_lumi_measure = normalized_lumi_dot;
  output.result_geometry_hash = adaptive_result_geometry_hash(
      configuration_.adaptive_grid_mode, level.geometry, NULL,
      field_operator);
}

// Account for the executed encounter and bind the final selected fields.
void Workspace::record_resolution_result(
    ResolutionEncounterBinding& binding,
    const CoherentSourceHierarchy& source_hierarchy,
    std::size_t selected,
    std::size_t spacing_class_count,
    bool vector_error_policy) {
  const int slice1 = binding.slice1, slice2 = binding.slice2;
  const auto& plan = binding.plan;
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  auto& output = binding.output;
  if (!finite_value(output.tail_uniform_lumi_measure) ||
      output.tail_uniform_lumi_measure < 0.0)
    throw std::runtime_error(
        "Mode 2/3 uniform luminosity exposure is non-finite or negative");
  encounter_all_uniform_lumi_measure_ +=
      static_cast<long double>(output.tail_uniform_lumi_measure);
  if (output.encounter_tail_budget_used)
    encounter_tail_uniform_lumi_measure_ +=
        static_cast<long double>(output.tail_uniform_lumi_measure);
  if (!finite_value(encounter_all_uniform_lumi_measure_) ||
      !finite_value(encounter_tail_uniform_lumi_measure_))
    throw std::overflow_error(
        "Mode 2/3 cumulative luminosity exposure overflowed");

  maximum_estimated_field_error_ = std::max(
      maximum_estimated_field_error_, output.estimated_field_error);
  if (!finite_value(output.vector_difference_maximum) ||
      output.vector_difference_maximum < 0.0)
    throw std::runtime_error(
        "Mode 2/3 final vector-maximum telemetry is invalid");
  maximum_vector_difference_ = std::max(
      maximum_vector_difference_, output.vector_difference_maximum);
  if (!finite_value(
          output.vector_maximum_beam_above_target_exposure) ||
      output.vector_maximum_beam_above_target_exposure < 0.0 ||
      output.vector_maximum_beam_above_target_exposure > 1.0)
    throw std::runtime_error(
        "Mode 2/3 per-beam above-target exposure telemetry is invalid");
  maximum_beam_above_target_exposure_ = std::max(
      maximum_beam_above_target_exposure_,
      output.vector_maximum_beam_above_target_exposure);
  if (vector_error_policy &&
      !output.vector_catastrophe_guard_passed)
    ++vector_catastrophe_failure_encounters_;
  if (source_hierarchy.model == "reference_cic_v1" ||
      source_hierarchy.model == "fixed_width_bspline1_v1") {
    ++source_hierarchy_encounters_;
    if (output.source_density_hash != 0U)
      ++source_density_hash_nonzero_encounters_;
    if (output.source_sampling_hash != 0U)
      ++source_sampling_hash_nonzero_encounters_;
    if (output.source_sampling_hierarchy_hash != 0U)
      ++source_sampling_hierarchy_hash_nonzero_encounters_;
    source_build_ms_ += output.source_build_ms;
    source_restrict_ms_ += output.source_restrict_ms;
    source_prolong_ms_ += output.source_prolong_ms;
    maximum_source_charge_closure_ = std::max(
        maximum_source_charge_closure_,
        output.source_maximum_charge_closure);
    maximum_source_projection_moment_shift_ = std::max(
        maximum_source_projection_moment_shift_,
        output.source_maximum_projection_moment_shift);
    maximum_selected_source_lumi_oracle_error_ = std::max(
        maximum_selected_source_lumi_oracle_error_,
        output.source_lumi_oracle_error);
  }
  ++mode2_encounters_;
  if (selected != 0U) ++mode2_refined_encounters_;
  if (mode2_spacing_selection_counts_.size() < spacing_class_count)
    mode2_spacing_selection_counts_.resize(spacing_class_count, 0U);
  if (selected >= mode2_spacing_selection_counts_.size())
    throw std::logic_error(
        "Mode-2 selected dynamic spacing class has no telemetry slot");
  ++mode2_spacing_selection_counts_[selected];
  checked_add_counter(source_beam1_in_visits_, output.beam1_in,
                      "Mode-2 source beam-1 admitted visit");
  checked_add_counter(source_beam1_out_visits_, output.beam1_out,
                      "Mode-2 source beam-1 rejected visit");
  checked_add_counter(source_beam2_in_visits_, output.beam2_in,
                      "Mode-2 source beam-2 admitted visit");
  checked_add_counter(source_beam2_out_visits_, output.beam2_out,
                      "Mode-2 source beam-2 rejected visit");

  ++adaptive_encounters_;
  if (plan.full_window()) ++adaptive_full_window_encounters_;
  checked_add_counter(adaptive_required_cells_, plan.required_node_count(),
                      "adaptive required-cell visit");
  // adaptive_local/padded counters are updated for every executed solve in
  // solve_field_level, including rejected Richardson levels and Mode-3 patch
  // work.  Do not report selected-only work as an actual-cost metric here.
  output.lattice_hash = plan.lattice_hash();
  output.bound_slice1 = slice1;
  output.bound_slice2 = slice2;
  output.bound_particle_count1 = particles1.size();
  output.bound_particle_count2 = particles2.size();
  output.result_binding_hash = field_binding_hash(
      slice1, slice2, particles1, particles2,
      output.lattice_hash, plan.plan_hash(), output.result_geometry_hash,
      output.source_model_tag, output.source_reference_factor,
      output.source_density_hash, output.source_sampling_hash,
      output.source_sampling_hierarchy_hash);
}

// Deposit one accepted beam with its root-grouped sampling moments.
void Workspace::deposit_coherent_source(
    CoherentSourceHierarchy& source_hierarchy,
    const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
    const std::vector<PARTICLE*>& particles,
    PHI_FLOAT charge,
    std::vector<PHI_FLOAT>& rho,
    std::vector<PHI_FLOAT>& sampling_s1,
    std::vector<PHI_FLOAT>& sampling_s2,
    bool beam1,
    std::size_t& particles_in,
    std::size_t& particles_out,
    long double& accepted_charge,
    long double& accepted_x_moment,
    long double& accepted_y_moment) const {
  const adaptive_grid::FieldLevelGeometry& geometry =
      source_hierarchy.canonical_geometry;
  if (descriptors.size() != particles.size())
    throw std::logic_error(
        "canonical coherent source lost its particle binding");
  for (std::size_t particle_index = 0U;
       particle_index < descriptors.size(); ++particle_index) {
    const adaptive_grid::CanonicalDepositDescriptor& descriptor =
        descriptors[particle_index];
    if (!particles[particle_index])
      throw std::logic_error(
          "canonical coherent source encountered a null particle");
    float x = 0.0f;
    float y = 0.0f;
    particles[particle_index]->XYposition(x, y);
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::runtime_error(
          "coherent source has a non-finite particle position");
    // The immutable nominal plan is authoritative.  A finer reference grid
    // can accept extra boundary-tail particles, but those must remain out;
    // only nominally admitted particles are projected onto the reference.
    if (!descriptor.accepted()) {
      ++particles_out;
      continue;
    }
    adaptive_grid::LevelDepositDescriptor reference_descriptor;
    if (source_hierarchy.model == "reference_cic_v1") {
      reference_descriptor = adaptive_grid::map_deposit_to_level(
          x, y, geometry);
      if (descriptor.accepted() && !reference_descriptor.accepted)
        throw std::runtime_error(
            "reference CIC cannot represent a particle admitted by the "
            "nominal immutable window plan");
    }
    const adaptive_grid::CanonicalIndex base_x =
        source_hierarchy.model == "reference_cic_v1"
            ? geometry.origin_x + reference_descriptor.base_x
            : descriptor.base_x();
    const adaptive_grid::CanonicalIndex base_y =
        source_hierarchy.model == "reference_cic_v1"
            ? geometry.origin_y + reference_descriptor.base_y
            : descriptor.base_y();
    const float weight_x = source_hierarchy.model == "reference_cic_v1"
        ? reference_descriptor.weight_x : descriptor.weight_x();
    const float weight_y = source_hierarchy.model == "reference_cic_v1"
        ? reference_descriptor.weight_y : descriptor.weight_y();
    if (base_x < geometry.origin_x || base_y < geometry.origin_y)
      throw std::runtime_error(
          "coherent CIC descriptor escaped the source window");
    const std::size_t local_x = static_cast<std::size_t>(
        base_x - geometry.origin_x);
    const std::size_t local_y = static_cast<std::size_t>(
        base_y - geometry.origin_y);
    if (local_x + 1U >= geometry.n_x ||
        local_y + 1U >= geometry.n_y)
      throw std::runtime_error(
          "coherent CIC cloud escaped the source window");
    GENERAL_GRID::assignChargeToCICWithStride(
        rho.data(), static_cast<int>(geometry.n_y),
        checked_int_index(local_x, "canonical coherent source x"),
        checked_int_index(local_y, "canonical coherent source y"),
        weight_x, weight_y, charge);
    const std::size_t base = local_x * geometry.n_y + local_y;
    const std::size_t touched[4] = {
        base, base + geometry.n_y, base + 1U,
        base + geometry.n_y + 1U};
    const long double wx = weight_x;
    const long double wy = weight_y;
    const long double magnitude = std::fabs(
        static_cast<long double>(charge));
    const long double contributions[4] = {
        magnitude * (1.0L - wx) * (1.0L - wy),
        magnitude * wx * (1.0L - wy),
        magnitude * (1.0L - wx) * wy,
        magnitude * wx * wy};
    if (source_hierarchy.model == "reference_cic_v1") {
      for (SamplingHierarchyLevel& sampling :
           source_hierarchy.sampling_levels) {
        std::vector<PHI_FLOAT>& level_s1 = beam1
            ? sampling.s1_1 : sampling.s1_2;
        std::vector<PHI_FLOAT>& level_s2 = beam1
            ? sampling.s2_1 : sampling.s2_2;
        adaptive_grid::accumulate_grouped_cic_sampling(
            reference_descriptor, static_cast<double>(magnitude), geometry,
            sampling.geometry, level_s1, level_s2);
      }
    } else {
      for (std::size_t corner = 0U; corner < 4U; ++corner) {
        const PHI_FLOAT contribution = static_cast<PHI_FLOAT>(
            contributions[corner]);
        sampling_s1[touched[corner]] += contribution;
        sampling_s2[touched[corner]] += contribution * contribution;
      }
    }
    accepted_charge += charge;
    accepted_x_moment += static_cast<long double>(charge) * x;
    accepted_y_moment += static_cast<long double>(charge) * y;
    ++particles_in;
  }
}

// Bind immutable physical roots to the declared source geometry.
std::vector<adaptive_particles::RootProjection> Workspace::make_adaptive_source_roots(
    const CoherentSourceHierarchy& source_hierarchy,
    const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
    const std::vector<PARTICLE*>& particles,
    const std::vector<PRIMARY_PARTICLE_ID>& ids,
    double macro_weight,
    std::uint64_t cohort) const {
  if (descriptors.size() != particles.size() ||
      ids.size() != particles.size())
    throw std::logic_error(
        "adaptive source root/particle/descriptor shape mismatch");
  std::vector<adaptive_particles::RootProjection> roots;
  roots.reserve(particles.size());
  for (std::size_t index = 0U; index < particles.size(); ++index) {
    if (!particles[index])
      throw std::logic_error(
          "adaptive source encountered a null physical root");
    adaptive_particles::RootProjection root;
    root.root_id = ids[index];
    root.statistical_cohort_id = cohort;
    float x = 0.0f;
    float y = 0.0f;
    particles[index]->XYposition(x, y);
    root.x = x;
    root.y = y;
    root.physical_weight = macro_weight;
    if (descriptors[index].accepted()) {
      root.cic = adaptive_grid::map_deposit_to_level(
          x, y, source_hierarchy.canonical_geometry);
      if (!root.cic.accepted)
        throw std::runtime_error(
            "adaptive reference source cannot represent a root "
            "admitted by the immutable nominal plan");
    }
    roots.push_back(root);
  }
  return roots;
}

// Publish source counts and ancestry metrics without changing particles.
void Workspace::publish_adaptive_source_receipt(
    FieldPrefetch& output,
    const adaptive_particles::SourceEnsemble& source,
    bool first) const {
  if (first) {
    output.adaptive_macroparticle_roots1 =
        source.receipt.roots_accepted;
    output.adaptive_macroparticle_premerge1 =
        source.receipt.representatives_before_merge;
    output.adaptive_macroparticle_representatives1 =
        source.receipt.representatives_after_merge;
    output.adaptive_macroparticle_many_root1 =
        source.metrics.many_root_representatives;
    output.adaptive_macroparticle_weight_ess1 =
        source.metrics.weight_ess;
    output.adaptive_macroparticle_clone_support1 =
        source.metrics.clone_support_proxy;
    output.adaptive_macroparticle_hash1 = source.receipt.content_hash;
    output.adaptive_macroparticle_revision1 =
        source.population_revision;
  } else {
    output.adaptive_macroparticle_roots2 =
        source.receipt.roots_accepted;
    output.adaptive_macroparticle_premerge2 =
        source.receipt.representatives_before_merge;
    output.adaptive_macroparticle_representatives2 =
        source.receipt.representatives_after_merge;
    output.adaptive_macroparticle_many_root2 =
        source.metrics.many_root_representatives;
    output.adaptive_macroparticle_weight_ess2 =
        source.metrics.weight_ess;
    output.adaptive_macroparticle_clone_support2 =
        source.metrics.clone_support_proxy;
    output.adaptive_macroparticle_hash2 = source.receipt.content_hash;
    output.adaptive_macroparticle_revision2 =
        source.population_revision;
  }
}

// Check the complete projection against its independent legacy oracle.
double Workspace::verify_adaptive_source_projection(
    const std::vector<double>& candidate,
    std::vector<PHI_FLOAT>& oracle,
    const char* quantity_label,
    bool commit_candidate) const {
  if (candidate.size() != oracle.size())
    throw std::logic_error(
        "adaptive source projection changed oracle shape");
  long double absolute_oracle = 0.0L;
  long double absolute_difference = 0.0L;
  double maximum_difference = 0.0;
  for (std::size_t cell = 0U; cell < oracle.size(); ++cell) {
    if (!std::isfinite(candidate[cell]))
      throw std::runtime_error(
          "adaptive source projection produced a non-finite value");
    absolute_oracle += std::fabs(
        static_cast<long double>(oracle[cell]));
    const double difference = std::fabs(
        candidate[cell] - static_cast<double>(oracle[cell]));
    absolute_difference += difference;
    maximum_difference = std::max(maximum_difference, difference);
  }
  const double residual = static_cast<double>(
      absolute_difference /
      std::max(absolute_oracle, static_cast<long double>(1.0)));
  const double roundoff_allowance =
      1024.0 * std::numeric_limits<double>::epsilon();
  // This path advertises an exact assignment projection.  The user
  // bias allocation may tighten its numerical check but may never
  // loosen it into an approximate resampler.  The fixed ceiling
  // covers the independently accumulated binary32 legacy oracle;
  // observed production-card residuals are about 1.4e-9.
  constexpr double exact_projection_ceiling = 1.0e-8;
  const double allowed = std::min(
      std::max(
          configuration_.adaptive_macroparticles.resample_bias_target,
          roundoff_allowance),
      exact_projection_ceiling);
  if (!finite_value(residual) || residual > allowed) {
    std::ostringstream error;
    error << "adaptive source " << quantity_label
          << " failed exact projection: relative_l1=" << residual
          << " max_abs=" << maximum_difference
          << " allowed=" << allowed;
    throw std::runtime_error(error.str());
  }
  if (commit_candidate)
    for (std::size_t cell = 0U; cell < oracle.size(); ++cell)
      oracle[cell] = static_cast<PHI_FLOAT>(candidate[cell]);
  return residual;
}

// Record admitted fixed-width sources and their physical moments.
void Workspace::record_fixed_width_sources(
    std::uint64_t& density_hash,
    const std::vector<adaptive_grid::CanonicalDepositDescriptor>& descriptors,
    const std::vector<PARTICLE*>& particles,
    PHI_FLOAT charge,
    std::uint64_t beam_tag,
    std::size_t& particles_in,
    std::size_t& particles_out,
    long double& accepted_charge,
    long double& accepted_x_moment,
    long double& accepted_y_moment) const {
  if (descriptors.size() != particles.size())
    throw std::logic_error(
        "fixed-width source lost its particle binding");
  hash_u64(density_hash, beam_tag);
  hash_double_bits(density_hash, charge);
  for (std::size_t particle_index = 0U;
       particle_index < descriptors.size(); ++particle_index) {
    hash_u64(density_hash, static_cast<std::uint64_t>(particle_index));
    if (!particles[particle_index])
      throw std::logic_error(
          "fixed-width source encountered a null particle");
    if (!descriptors[particle_index].accepted()) {
      ++particles_out;
      hash_u64(density_hash, 0U);
      continue;
    }
    float x = 0.0f;
    float y = 0.0f;
    particles[particle_index]->XYposition(x, y);
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::runtime_error(
          "fixed-width source has a non-finite particle position");
    hash_u64(density_hash, 1U);
    hash_double_bits(density_hash, static_cast<double>(x));
    hash_double_bits(density_hash, static_cast<double>(y));
    accepted_charge += charge;
    accepted_x_moment += static_cast<long double>(charge) * x;
    accepted_y_moment += static_cast<long double>(charge) * y;
    ++particles_in;
  }
}

// Audit deposited charge and first moments before any hierarchy transfer.
void Workspace::audit_coherent_source(
    const CoherentSourceHierarchy& source_hierarchy,
    const std::vector<PHI_FLOAT>& rho,
    long double accepted_charge,
    long double accepted_x_moment,
    long double accepted_y_moment) const {
  const adaptive_grid::FieldLevelGeometry& geometry =
      source_hierarchy.canonical_geometry;
  long double deposited_charge = 0.0L;
  long double deposited_x_moment = 0.0L;
  long double deposited_y_moment = 0.0L;
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
    const long double x =
        (static_cast<long double>(geometry.origin_x) + ix + 0.5L -
         geometry.offset_x) * geometry.delta_x;
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
      const PHI_FLOAT value = rho[ix * geometry.n_y + iy];
      if (!std::isfinite(value))
        throw std::runtime_error(
            "canonical CIC deposit produced non-finite charge");
      const long double y =
          (static_cast<long double>(geometry.origin_y) + iy + 0.5L -
           geometry.offset_y) * geometry.delta_y;
      deposited_charge += value;
      deposited_x_moment += static_cast<long double>(value) * x;
      deposited_y_moment += static_cast<long double>(value) * y;
    }
  }
  const long double charge_scale = std::max(
      std::fabs(accepted_charge), static_cast<long double>(1.0));
  const long double x_scale = std::max(
      std::fabs(accepted_x_moment), charge_scale * geometry.delta_x);
  const long double y_scale = std::max(
      std::fabs(accepted_y_moment), charge_scale * geometry.delta_y);
  const long double charge_error =
      std::fabs(deposited_charge - accepted_charge) / charge_scale;
  const long double moment_error = std::max(
      std::fabs(deposited_x_moment - accepted_x_moment) / x_scale,
      std::fabs(deposited_y_moment - accepted_y_moment) / y_scale);
  if (charge_error > 5.0e-6L || moment_error > 5.0e-5L)
    throw std::runtime_error(
        "canonical CIC deposit failed accepted-particle charge/first-"
        "moment closure");
}

// Build and certify the immutable source before any field level is solved.
// Allocate the declared raster or continuous-source representation.
bool Workspace::initialize_resolution_source(
    ResolutionEncounterBinding& binding,
    const std::vector<std::size_t>& spacing_factors,
    CoherentSourceHierarchy& source_hierarchy) const {
  source_hierarchy.model = configuration_.adaptive_source_model;
  if (source_hierarchy.model == "reference_cic_v1") {
    source_hierarchy.source_factor =
        configuration_.adaptive_source_reference_nx /
        static_cast<std::size_t>(configuration_.n_x);
    source_hierarchy.canonical_geometry = make_full_registered_geometry(
        *canonical_lattice_, source_hierarchy.source_factor);
    if (source_hierarchy.canonical_geometry.global_n_x !=
            configuration_.adaptive_source_reference_nx ||
        source_hierarchy.canonical_geometry.global_n_y !=
            configuration_.adaptive_source_reference_ny)
      throw std::logic_error(
          "reference CIC geometry disagrees with its validated dimensions");
  } else if (source_hierarchy.model == "fixed_width_bspline1_v1") {
    source_hierarchy.source_factor = 0U;
    source_hierarchy.source_width_x = configuration_.adaptive_source_width_x;
    source_hierarchy.source_width_y = configuration_.adaptive_source_width_y;
  } else {
    source_hierarchy.source_factor = 1U;
    // The authoritative canonical source must cover the exact factor-one
    // field level that will consume it.  Mode 3 adds blocking/error/transition
    // guard cells even at factor one; rebuilding an unguarded geometry here
    // made a valid cropped field level extend beyond its registered density
    // hierarchy before the terminal-uniform policy could be reached.
    source_hierarchy.canonical_geometry = resolution_geometry(binding,1U);
  }
  const bool fixed_width_source =
      source_hierarchy.model == "fixed_width_bspline1_v1";
  const std::size_t canonical_cells = fixed_width_source
      ? 0U : source_hierarchy.canonical_geometry.cells();
  if (!fixed_width_source) {
    source_hierarchy.rho1.assign(canonical_cells, PHI_FLOAT(0.0));
    source_hierarchy.rho2.assign(canonical_cells, PHI_FLOAT(0.0));
  }
  if (source_hierarchy.model == "reference_cic_v1") {
    for (std::size_t factor : spacing_factors) {
      if (factor > source_hierarchy.source_factor) break;
      SamplingHierarchyLevel sampling;
      sampling.factor = factor;
      sampling.geometry = make_full_registered_geometry(
          *canonical_lattice_, factor);
      const std::size_t sampling_cells = sampling.geometry.cells();
      sampling.s1_1.assign(sampling_cells, PHI_FLOAT(0.0));
      sampling.s2_1.assign(sampling_cells, PHI_FLOAT(0.0));
      sampling.s1_2.assign(sampling_cells, PHI_FLOAT(0.0));
      sampling.s2_2.assign(sampling_cells, PHI_FLOAT(0.0));
      source_hierarchy.sampling_levels.push_back(std::move(sampling));
    }
  } else if (!fixed_width_source) {
    source_hierarchy.sampling_s1_1.assign(canonical_cells, PHI_FLOAT(0.0));
    source_hierarchy.sampling_s2_1.assign(canonical_cells, PHI_FLOAT(0.0));
    source_hierarchy.sampling_s1_2.assign(canonical_cells, PHI_FLOAT(0.0));
    source_hierarchy.sampling_s2_2.assign(canonical_cells, PHI_FLOAT(0.0));
  }
  return fixed_width_source;
}

// Certify all source observables before committing an adaptive projection.
void Workspace::project_adaptive_source(
    ResolutionEncounterBinding& binding,
    CoherentSourceHierarchy& source_hierarchy) const {
  const int slice1 = binding.slice1, slice2 = binding.slice2;
  const auto& plan = binding.plan;
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  auto& output = binding.output;
  if (configuration_.adaptive_macroparticles.mode !=
      adaptive_particles::Mode::LegacyOff) {
    if (source_hierarchy.model != "reference_cic_v1")
      throw std::logic_error(
          "adaptive source projection entered without reference_cic_v1");

    const std::vector<PRIMARY_PARTICLE_ID>& ids1 =
        beam1_->particle_beam().getParticleIdVectorShapeChecked(slice1);
    const std::vector<PRIMARY_PARTICLE_ID>& ids2 =
        beam2_->particle_beam().getParticleIdVectorShapeChecked(slice2);
    const std::uint64_t epoch =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(slice1))
             << 32U) |
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(slice2));
    const std::vector<adaptive_particles::RootProjection> roots1 =
        make_adaptive_source_roots(source_hierarchy,plan.beam1_sources(), particles1, ids1,
                    static_cast<double>(configuration_.macro_charge1),
                    configuration_.adaptive_macroparticles.scramble_id);
    const std::vector<adaptive_particles::RootProjection> roots2 =
        make_adaptive_source_roots(source_hierarchy,plan.beam2_sources(), particles2, ids2,
                    static_cast<double>(configuration_.macro_charge2),
                    configuration_.adaptive_macroparticles.scramble_id);
    source_hierarchy.adaptive_source1 =
        adaptive_particles::build_assignment_preserving_source(
            configuration_.adaptive_macroparticles,
            source_hierarchy.canonical_geometry, roots1, 1U, epoch);
    source_hierarchy.adaptive_source2 =
        adaptive_particles::build_assignment_preserving_source(
            configuration_.adaptive_macroparticles,
            source_hierarchy.canonical_geometry, roots2, 2U, epoch);

    publish_adaptive_source_receipt(output,source_hierarchy.adaptive_source1, true);
    publish_adaptive_source_receipt(output,source_hierarchy.adaptive_source2, false);

    if (configuration_.adaptive_macroparticles.mode !=
        adaptive_particles::Mode::Diagnostics) {
      std::vector<double> rho1;
      std::vector<double> s1_1;
      std::vector<double> s2_1;
      std::vector<double> rho2;
      std::vector<double> s1_2;
      std::vector<double> s2_2;
      adaptive_particles::deposit_source(
          source_hierarchy.adaptive_source1,
          source_hierarchy.canonical_geometry, rho1, s1_1, s2_1);
      adaptive_particles::deposit_source(
          source_hierarchy.adaptive_source2,
          source_hierarchy.canonical_geometry, rho2, s1_2, s2_2);

      SamplingHierarchyLevel* source_sampling = NULL;
      for (SamplingHierarchyLevel& sampling :
           source_hierarchy.sampling_levels)
        if (sampling.factor == source_hierarchy.source_factor) {
          source_sampling = &sampling;
          break;
        }
      if (!source_sampling)
        throw std::logic_error(
            "adaptive source has no legacy source-level sampling oracle");
      // First validate every independently accumulated observable.  Density
      // is committed only after all six arrays pass, so a failed S1/S2
      // ancestry check cannot leave a partially updated field source.
      const double residual1 = verify_adaptive_source_projection(
          rho1, source_hierarchy.rho1, "beam1_density", false);
      const double residual2 = verify_adaptive_source_projection(
          rho2, source_hierarchy.rho2, "beam2_density", false);
      const double residual_s1_1 = verify_adaptive_source_projection(
          s1_1, source_sampling->s1_1, "beam1_root_grouped_s1", false);
      const double residual_s2_1 = verify_adaptive_source_projection(
          s2_1, source_sampling->s2_1, "beam1_root_grouped_s2", false);
      const double residual_s1_2 = verify_adaptive_source_projection(
          s1_2, source_sampling->s1_2, "beam2_root_grouped_s1", false);
      const double residual_s2_2 = verify_adaptive_source_projection(
          s2_2, source_sampling->s2_2, "beam2_root_grouped_s2", false);
      verify_adaptive_source_projection(rho1, source_hierarchy.rho1,
                        "beam1_density_commit", true);
      verify_adaptive_source_projection(rho2, source_hierarchy.rho2,
                        "beam2_density_commit", true);
      output.adaptive_macroparticle_projection_residual =
          std::max(std::max(residual1, residual2),
                   std::max(std::max(residual_s1_1, residual_s2_1),
                            std::max(residual_s1_2, residual_s2_2)));
      output.adaptive_macroparticle_committed = true;
    }
  }
}

// Bind the density and every root-grouped sampling level to the source.
void Workspace::hash_reference_source(
    CoherentSourceHierarchy& source_hierarchy) const {
  if (source_hierarchy.model == "reference_cic_v1") {
    std::uint64_t density_hash = UINT64_C(1469598103934665603);
    hash_u64(density_hash, UINT64_C(0x5245464349435631));  // "REFCICV1"
    hash_u64(density_hash, source_hierarchy.canonical_geometry.global_n_x);
    hash_u64(density_hash, source_hierarchy.canonical_geometry.global_n_y);
    hash_u64(density_hash, source_hierarchy.beam1_in);
    hash_u64(density_hash, source_hierarchy.beam2_in);
    for (PHI_FLOAT value : source_hierarchy.rho1)
      hash_double_bits(density_hash, value);
    for (PHI_FLOAT value : source_hierarchy.rho2)
      hash_double_bits(density_hash, value);
    source_hierarchy.density_hash = density_hash;
    const SamplingHierarchyLevel* source_sampling = NULL;
    for (const SamplingHierarchyLevel& sampling :
         source_hierarchy.sampling_levels)
      if (sampling.factor == source_hierarchy.source_factor) {
        source_sampling = &sampling;
        break;
      }
    if (!source_sampling)
      throw std::runtime_error(
          "reference CIC source factor has no sampling hierarchy member");
    std::uint64_t sampling_hash = UINT64_C(1469598103934665603);
    hash_u64(sampling_hash, UINT64_C(0x52454653414d5031));  // "REFSAMP1"
    hash_u64(sampling_hash, source_sampling->geometry.global_n_x);
    hash_u64(sampling_hash, source_sampling->geometry.global_n_y);
    for (PHI_FLOAT value : source_sampling->s1_1)
      hash_double_bits(sampling_hash, value);
    for (PHI_FLOAT value : source_sampling->s2_1)
      hash_double_bits(sampling_hash, value);
    for (PHI_FLOAT value : source_sampling->s1_2)
      hash_double_bits(sampling_hash, value);
    for (PHI_FLOAT value : source_sampling->s2_2)
      hash_double_bits(sampling_hash, value);
    source_hierarchy.sampling_hash = sampling_hash;
    std::uint64_t hierarchy_hash = UINT64_C(1469598103934665603);
    hash_u64(hierarchy_hash, UINT64_C(0x52454653414c4c31));  // "REFSALL1"
    for (const SamplingHierarchyLevel& sampling :
         source_hierarchy.sampling_levels) {
      hash_u64(hierarchy_hash, sampling.factor);
      hash_u64(hierarchy_hash, sampling.geometry.global_n_x);
      hash_u64(hierarchy_hash, sampling.geometry.global_n_y);
      for (PHI_FLOAT value : sampling.s1_1)
        hash_double_bits(hierarchy_hash, value);
      for (PHI_FLOAT value : sampling.s2_1)
        hash_double_bits(hierarchy_hash, value);
      for (PHI_FLOAT value : sampling.s1_2)
        hash_double_bits(hierarchy_hash, value);
      for (PHI_FLOAT value : sampling.s2_2)
        hash_double_bits(hierarchy_hash, value);
    }
    source_hierarchy.sampling_hierarchy_hash = hierarchy_hash;
  }
}

// Publish immutable source provenance and adaptive population identity.
void Workspace::publish_source_binding(
    FieldPrefetch& output,
    const CoherentSourceHierarchy& source_hierarchy) const {
  output.source_reference_factor = source_hierarchy.source_factor;
  output.source_density_hash = source_hierarchy.density_hash;
  output.source_sampling_hash = source_hierarchy.sampling_hash;
  output.source_sampling_hierarchy_hash =
      source_hierarchy.sampling_hierarchy_hash;
  if (configuration_.adaptive_macroparticles.mode !=
      adaptive_particles::Mode::LegacyOff) {
    hash_u64(output.source_sampling_hierarchy_hash,
             UINT64_C(0x414d5042494e4431));  // "AMPBIND1"
    hash_u64(output.source_sampling_hierarchy_hash,
             static_cast<std::uint64_t>(
                 configuration_.adaptive_macroparticles.mode));
    hash_u64(output.source_sampling_hierarchy_hash,
             output.adaptive_macroparticle_hash1);
    hash_u64(output.source_sampling_hierarchy_hash,
             output.adaptive_macroparticle_hash2);
    hash_u64(output.source_sampling_hierarchy_hash,
             output.adaptive_macroparticle_revision1);
    hash_u64(output.source_sampling_hierarchy_hash,
             output.adaptive_macroparticle_revision2);
  }
  output.source_budget_runtime_verified = false;
}

Workspace::CoherentSourceHierarchy Workspace::build_resolution_source(
    ResolutionEncounterBinding& binding,
    const std::vector<std::size_t>& spacing_factors) {
  const auto& plan = binding.plan;
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  auto& output = binding.output;
  const std::chrono::steady_clock::time_point canonical_deposit_start =
      std::chrono::steady_clock::now();
  CoherentSourceHierarchy source_hierarchy;
  const bool fixed_width_source =
      initialize_resolution_source(binding, spacing_factors, source_hierarchy);
  long double accepted_charge1 = 0.0L;
  long double accepted_x_moment1 = 0.0L;
  long double accepted_y_moment1 = 0.0L;
  long double accepted_charge2 = 0.0L;
  long double accepted_x_moment2 = 0.0L;
  long double accepted_y_moment2 = 0.0L;
  if (!fixed_width_source) {
    deposit_coherent_source(source_hierarchy,
        plan.beam1_sources(), particles1, configuration_.macro_charge1,
        source_hierarchy.rho1, source_hierarchy.sampling_s1_1,
        source_hierarchy.sampling_s2_1, true, source_hierarchy.beam1_in,
        source_hierarchy.beam1_out, accepted_charge1,
        accepted_x_moment1, accepted_y_moment1);
    deposit_coherent_source(source_hierarchy,
        plan.beam2_sources(), particles2, configuration_.macro_charge2,
        source_hierarchy.rho2, source_hierarchy.sampling_s1_2,
        source_hierarchy.sampling_s2_2, false, source_hierarchy.beam2_in,
        source_hierarchy.beam2_out, accepted_charge2,
        accepted_x_moment2, accepted_y_moment2);

    // Project each accepted physical root onto a disposable source ensemble.
    // Physical PARTICLE objects remain
    // immutable here and are still the only inputs to luminosity, radiation,
    // spin and stored-event physics.  The independently implemented legacy
    // deposit above is retained as an oracle; the transaction commits only
    // after density and root-grouped S1/S2 agree within the declared budget.
    project_adaptive_source(binding,source_hierarchy);
  } else {
    std::uint64_t density_hash = UINT64_C(1469598103934665603);
    hash_u64(density_hash, UINT64_C(0x4253504c31563100));  // "BSPL1V1\0"
    hash_double_bits(density_hash, source_hierarchy.source_width_x);
    hash_double_bits(density_hash, source_hierarchy.source_width_y);

    record_fixed_width_sources(density_hash,
        plan.beam1_sources(), particles1, configuration_.macro_charge1,
        UINT64_C(1), source_hierarchy.beam1_in,
        source_hierarchy.beam1_out, accepted_charge1,
        accepted_x_moment1, accepted_y_moment1);
    record_fixed_width_sources(density_hash,
        plan.beam2_sources(), particles2, configuration_.macro_charge2,
        UINT64_C(2), source_hierarchy.beam2_in,
        source_hierarchy.beam2_out, accepted_charge2,
        accepted_x_moment2, accepted_y_moment2);
    source_hierarchy.density_hash = density_hash;
    std::uint64_t sampling_hash = density_hash;
    hash_u64(sampling_hash, UINT64_C(0x4253504c53414d50));  // "BSPLSAMP"
    source_hierarchy.sampling_hash = sampling_hash;
    std::uint64_t hierarchy_hash = sampling_hash;
    hash_u64(hierarchy_hash, UINT64_C(0x4253504c48494552)); // "BSPLHIER"
    for (std::size_t factor : spacing_factors)
      hash_u64(hierarchy_hash, static_cast<std::uint64_t>(factor));
    source_hierarchy.sampling_hierarchy_hash = hierarchy_hash;
  }
  source_hierarchy.accepted_charge1 = accepted_charge1;
  source_hierarchy.accepted_x_moment1 = accepted_x_moment1;
  source_hierarchy.accepted_y_moment1 = accepted_y_moment1;
  source_hierarchy.accepted_charge2 = accepted_charge2;
  source_hierarchy.accepted_x_moment2 = accepted_x_moment2;
  source_hierarchy.accepted_y_moment2 = accepted_y_moment2;
  if (source_hierarchy.beam1_in + source_hierarchy.beam2_in !=
      plan.accepted_source_count())
    throw std::runtime_error(
        "coherent-source deposition disagrees with its immutable "
        "window plan");

  // Independently audit the authoritative CIC deposit against the accepted
  // particle list before any hierarchy transfer or FFT can consume it.  The
  // cell-centre moment is exactly the CIC interpolation moment by construction.

  if (!fixed_width_source) {
    audit_coherent_source(source_hierarchy,
        source_hierarchy.rho1, accepted_charge1,
        accepted_x_moment1, accepted_y_moment1);
    audit_coherent_source(source_hierarchy,
        source_hierarchy.rho2, accepted_charge2,
        accepted_x_moment2, accepted_y_moment2);
  }
  hash_reference_source(source_hierarchy);
  adaptive_deposit_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - canonical_deposit_start).count();
  publish_source_binding(output,source_hierarchy);
  output.source_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - canonical_deposit_start).count();

  return source_hierarchy;
}

// Solve the next resource-admitted level and bind its source diagnostics.
void Workspace::solve_resolution_level(ResolutionControllerState& state) {
  auto& levels = state.levels;
  auto& spacing_factors = state.spacing_factors;
  auto& binding = state.binding;
  auto& slice1 = state.binding.slice1;
  auto& slice2 = state.binding.slice2;
  auto& plan = state.binding.plan;
  auto& source_hierarchy = state.source_hierarchy;
  auto& output = state.binding.output;
  if (levels.size() >= spacing_factors.size())
    throw std::logic_error(
        "Mode-2 controller attempted to solve past its resource-admissible "
        "spacing ladder");
  const std::size_t factor = spacing_factors[levels.size()];
  adaptive_grid::FieldLevelGeometry geometry =
      resolution_geometry(binding,factor);
  levels.push_back(solve_field_level(
      slice1, slice2, plan, geometry,
      NULL, NULL, NULL, NULL, false, &source_hierarchy, NULL));
  if (source_hierarchy.model == "fixed_width_bspline1_v1") {
    const LevelSolution& level = levels.back();
    if (!level.context)
      throw std::logic_error(
          "fixed-width level hash has no field context");
    std::uint64_t density_level_hash = UINT64_C(1469598103934665603);
    hash_u64(density_level_hash, UINT64_C(0x4253504c44454e53));
    hash_field_geometry(density_level_hash, level.geometry);
    for (PHI_FLOAT value : level.context->rho1)
      hash_double_bits(density_level_hash, value);
    for (PHI_FLOAT value : level.context->rho2)
      hash_double_bits(density_level_hash, value);
    std::uint64_t sampling_level_hash = UINT64_C(1469598103934665603);
    hash_u64(sampling_level_hash, UINT64_C(0x4253504c53414d50));
    hash_field_geometry(sampling_level_hash, level.geometry);
    for (PHI_FLOAT value : level.sampling_s1_1)
      hash_double_bits(sampling_level_hash, value);
    for (PHI_FLOAT value : level.sampling_s2_1)
      hash_double_bits(sampling_level_hash, value);
    for (PHI_FLOAT value : level.sampling_s1_2)
      hash_double_bits(sampling_level_hash, value);
    for (PHI_FLOAT value : level.sampling_s2_2)
      hash_double_bits(sampling_level_hash, value);
    hash_u64(output.source_density_hash, density_level_hash);
    hash_u64(output.source_sampling_hash, sampling_level_hash);
    hash_u64(output.source_sampling_hierarchy_hash,
             static_cast<std::uint64_t>(factor));
    hash_u64(output.source_sampling_hierarchy_hash, density_level_hash);
    hash_u64(output.source_sampling_hierarchy_hash, sampling_level_hash);
  }
  output.source_build_ms += levels.back().source_build_ms;
  output.source_restrict_ms += levels.back().source_restrict_ms;
  output.source_prolong_ms += levels.back().source_prolong_ms;
  output.source_maximum_charge_closure = std::max(
      output.source_maximum_charge_closure,
      levels.back().maximum_charge_closure);
  output.source_maximum_projection_moment_shift = std::max(
      output.source_maximum_projection_moment_shift,
      levels.back().maximum_first_moment_error);
}

// Solve only the levels required by the current comparison.
void Workspace::ensure_resolution_levels(
    ResolutionControllerState& state,
    std::size_t required) {
  auto& levels = state.levels;
  while (levels.size() < required) solve_resolution_level(state);
}

// Compare the same physical targets at adjacent resolutions.
void Workspace::compare_resolution_pair(
    ResolutionControllerState& state,
    std::size_t index) {
  auto& levels = state.levels;
  auto& raw_pairs = state.raw_pairs;
  auto& raw_pair_sensitivities = state.raw_pair_sensitivities;
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& raw_pair_ready = state.raw_pair_ready;
  auto& accepted_targets1 = state.accepted_targets1;
  auto& accepted_targets2 = state.accepted_targets2;
  auto& target = state.target;
  auto& target_ids1 = state.target_ids1;
  auto& target_ids2 = state.target_ids2;
  auto& vector_error_policy = state.vector_error_policy;
  auto& mixed_vector_error_policy = state.mixed_vector_error_policy;
  if (index >= raw_pairs.size())
    throw std::out_of_range("Mode-2 Richardson pair index is invalid");
  if (raw_pair_ready[index]) return;
  ensure_resolution_levels(state, index + 2U);
  const std::size_t coarse_factor =
      levels[index].geometry.refinement_factor;
  const std::size_t fine_factor =
      levels[index + 1U].geometry.refinement_factor;
  if (fine_factor % coarse_factor != 0U)
    throw std::logic_error("noncommensurate Mode-2 spacing classes");
  raw_pairs[index] = adaptive_grid::estimate_same_particle_field_error(
      levels[index].fields1, levels[index].fields2,
      levels[index + 1U].fields1, levels[index + 1U].fields2,
      fine_factor / coarse_factor, 1.0,
      configuration_.adaptive_field_floor,
      &accepted_targets1, &accepted_targets2);
  raw_pair_sensitivities[index] =
      adaptive_grid::estimate_symmetric_vector_field_sensitivity(
          levels[index].fields1, levels[index].fields2,
          levels[index + 1U].fields1, levels[index + 1U].fields2,
          configuration_.adaptive_field_floor,
          &accepted_targets1, &accepted_targets2);
  if (vector_error_policy) {
    if (mixed_vector_error_policy)
      raw_pair_vector_statistics[index] =
          adaptive_grid::estimate_mixed_vector_field_error_statistics_v2(
              levels[index].fields1, levels[index].fields2,
              levels[index + 1U].fields1, levels[index + 1U].fields2,
              configuration_.adaptive_error_quantile, target,
              configuration_.adaptive_error_reference_field,
              configuration_.adaptive_error_absolute_floor_fraction,
              configuration_.adaptive_error_catastrophe_limit,
              configuration_.adaptive_error_catastrophe_absolute_fraction,
              &accepted_targets1, &accepted_targets2,
              &target_ids1, &target_ids2);
    else
      raw_pair_vector_statistics[index] =
          adaptive_grid::estimate_vector_field_error_statistics(
              levels[index].fields1, levels[index].fields2,
              levels[index + 1U].fields1, levels[index + 1U].fields2,
              configuration_.adaptive_error_quantile, target,
              &accepted_targets1, &accepted_targets2);
  }
  if (!raw_pairs[index].reliable)
    throw std::runtime_error(
        "Mode-2 same-particle field comparison has no reliable targets");
  if (!raw_pair_sensitivities[index].reliable)
    throw std::runtime_error(
        "Mode-2 symmetric vector field comparison has no reliable "
        "targets");
  if (vector_error_policy &&
      !raw_pair_vector_statistics[index].reliable)
    throw std::runtime_error(
        "Mode-2 vector RMS/quantile comparison has no reliable targets");
  raw_pair_ready[index] = 1U;
}

// Require a three-level contraction estimate before certifying a pair.
void Workspace::certify_resolution_pair(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& levels = state.levels;
  auto& raw_pairs = state.raw_pairs;
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& vector_pair_estimates = state.vector_pair_estimates;
  auto& pair_estimates = state.pair_estimates;
  auto& vector_coarse_errors = state.vector_coarse_errors;
  auto& vector_fine_errors = state.vector_fine_errors;
  auto& pair_estimate_ready = state.pair_estimate_ready;
  auto& observed_orders = state.observed_orders;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& triplet_first_pairs = state.triplet_first_pairs;
  auto& triplet_second_pairs = state.triplet_second_pairs;
  auto& triplet_valid = state.triplet_valid;
  auto& accepted_targets1 = state.accepted_targets1;
  auto& accepted_targets2 = state.accepted_targets2;
  auto& spacing_factors = state.spacing_factors;
  auto& source_hierarchy = state.source_hierarchy;
  auto& target = state.target;
  auto& vector_error_policy = state.vector_error_policy;
  if (pair >= raw_pairs.size())
    throw std::out_of_range("Mode-2 estimate pair index is invalid");
  if (pair_estimate_ready[pair]) return;
  std::size_t first_pair = pair + 1U < raw_pairs.size()
      ? pair : pair - 1U;
  std::size_t second_pair = pair + 1U < raw_pairs.size()
      ? pair + 1U : pair;
  if (source_hierarchy.model == "reference_cic_v1") {
    const adaptive_grid::SameRegimeTriplet triplet =
        adaptive_grid::select_same_regime_triplet(
            spacing_factors, pair, source_hierarchy.source_factor);
    if (!triplet.valid) {
      // A transition pair (or a regime represented by only one pair) has
      // no mathematically admissible three-level order estimate.  Retain a
      // conservative diagnostic error, but make the candidate explicitly
      // uncertifiable so the controller refines to a same-regime triplet.
      compare_resolution_pair(state, pair);
      const std::size_t coarse_factor =
          levels[pair].geometry.refinement_factor;
      const std::size_t fine_factor =
          levels[pair + 1U].geometry.refinement_factor;
      observed_orders[pair] = 0.25;
      asymptotic_pairs[pair] = 0U;
      pair_estimates[pair] =
          adaptive_grid::estimate_same_particle_field_error(
              levels[pair].fields1, levels[pair].fields2,
              levels[pair + 1U].fields1, levels[pair + 1U].fields2,
              fine_factor / coarse_factor, observed_orders[pair],
              configuration_.adaptive_field_floor,
              &accepted_targets1, &accepted_targets2);
      if (!pair_estimates[pair].reliable)
        throw std::runtime_error(
            "Mode-2 transition-pair field diagnostic is unreliable");
      if (vector_error_policy) {
        const double ratio_to_order = std::pow(
            static_cast<double>(fine_factor / coarse_factor),
            observed_orders[pair]);
        const double divisor = ratio_to_order - 1.0;
        const adaptive_grid::VectorFieldErrorStatistics& statistics =
            raw_pair_vector_statistics[pair];
        vector_coarse_errors[pair] = ratio_to_order / divisor *
            std::max(statistics.aggregate_rms,
                     statistics.aggregate_quantile);
        vector_fine_errors[pair] = 1.0 / divisor *
            std::max(statistics.aggregate_rms,
                     statistics.aggregate_quantile);
      }
      pair_estimate_ready[pair] = 1U;
      return;
    }
    first_pair = triplet.first_pair;
    second_pair = triplet.second_pair;
    triplet_valid[pair] = 1U;
  } else {
    triplet_valid[pair] = 1U;
  }
  triplet_first_pairs[pair] = first_pair;
  triplet_second_pairs[pair] = second_pair;
  compare_resolution_pair(state, first_pair);
  compare_resolution_pair(state, second_pair);
  if (vector_error_policy) {
    const adaptive_grid::VectorFieldErrorStatistics& first =
        raw_pair_vector_statistics[first_pair];
    const adaptive_grid::VectorFieldErrorStatistics& second =
        raw_pair_vector_statistics[second_pair];
    const std::size_t coarse_factor =
        levels[pair].geometry.refinement_factor;
    const std::size_t fine_factor =
        levels[pair + 1U].geometry.refinement_factor;
    const adaptive_grid::VectorFieldErrorStatistics& current =
        raw_pair_vector_statistics[pair];
    vector_pair_estimates[pair] =
        adaptive_grid::estimate_vector_field_richardson_error(
            first, second, current, fine_factor / coarse_factor, 0.25);
    const adaptive_grid::VectorFieldRichardsonEstimate& vector_estimate =
        vector_pair_estimates[pair];
    if (!vector_estimate.reliable)
      throw std::runtime_error(
          "Mode-2 vector RMS/quantile Richardson estimate is non-finite");
    observed_orders[pair] = vector_estimate.observed_order;
    asymptotic_pairs[pair] = vector_estimate.asymptotic ? 1U : 0U;
    vector_coarse_errors[pair] = vector_estimate.coarse_error;
    vector_fine_errors[pair] = vector_estimate.fine_error;

    // Preserve the complete component RMS/Linf estimator as telemetry, but
    // its Linf value is deliberately not an acceptance gate for this
    // opt-in policy.
    pair_estimates[pair] =
        adaptive_grid::estimate_same_particle_field_error(
            levels[pair].fields1, levels[pair].fields2,
            levels[pair + 1U].fields1, levels[pair + 1U].fields2,
            fine_factor / coarse_factor,
            std::max(0.25, vector_estimate.observed_order),
            configuration_.adaptive_field_floor,
            &accepted_targets1, &accepted_targets2);
    if (!pair_estimates[pair].reliable)
      throw std::runtime_error(
          "Mode-2 component field diagnostic is unreliable");
    pair_estimate_ready[pair] = 1U;
    return;
  }
  double conservative_order = std::numeric_limits<double>::infinity();
  bool active_component = false;
  bool asymptotic = true;
  const double minimum_supported_order = 0.25;
  const double ratio = static_cast<double>(
      levels[pair + 1U].geometry.refinement_factor) /
      static_cast<double>(levels[pair].geometry.refinement_factor);
  const double ratio_to_floor_order = std::pow(
      ratio, minimum_supported_order);
  for (std::size_t component = 0U; component < 4U; ++component) {
    // If a component already satisfies the requested coarse-level error
    // even under the deliberately weak p=0.25 floor, its observed order is
    // immaterial to acceptance.  Requiring strict contraction for such a
    // numerically negligible component turns roundoff/PIC noise into a
    // false global veto.  Components that need Richardson extrapolation
    // remain subject to the full three-level contraction gate below.
    const bool fixed_width_projection =
        source_hierarchy.model == "fixed_width_bspline1_v1";
    const double conservative_component_difference =
        fixed_width_projection
            ? raw_pairs[pair].
                  observed_difference_rms_components[component]
            : std::max(
                  raw_pairs[pair].
                      observed_difference_rms_components[component],
                  raw_pairs[pair].
                      observed_difference_linf_components[component]);
    const double conservative_component_error =
        ratio_to_floor_order /
        (ratio_to_floor_order - 1.0) *
        conservative_component_difference;
    if (finite_value(conservative_component_error) &&
        conservative_component_error <= target)
      continue;
    const double first_norms[2] = {
        raw_pairs[first_pair].
            absolute_difference_rms_components[component],
        raw_pairs[first_pair].
            absolute_difference_linf_components[component]};
    const double second_norms[2] = {
        raw_pairs[second_pair].
            absolute_difference_rms_components[component],
        raw_pairs[second_pair].
            absolute_difference_linf_components[component]};
    // A fixed continuous particle shape gives an L2-convergent stochastic
    // finite-volume source, but the single largest particle kick can move
    // between samples and need not exhibit a stable asymptotic order. Use
    // component RMS differences to estimate p for that model and retain the
    // measured maximum change below as a direct, non-extrapolated guard.
    const std::size_t norm_count = fixed_width_projection ? 1U : 2U;
    for (std::size_t norm = 0U; norm < norm_count; ++norm) {
      const double first = first_norms[norm];
      const double second = second_norms[norm];
      if (first == 0.0 && second == 0.0) continue;
      active_component = true;
      if (!(first > 0.0) || !(second >= 0.0) || !(second < first)) {
        asymptotic = false;
        continue;
      }
      const double observed_order = second == 0.0
          ? 16.0
          : std::log(first / second) / std::log(ratio);
      if (!finite_value(observed_order) || !(observed_order > 0.0)) {
        asymptotic = false;
        continue;
      }
      conservative_order = std::min(
          conservative_order, std::min(16.0, observed_order));
    }
  }
  if (!active_component) conservative_order = 2.0;
  if (!finite_value(conservative_order)) {
    asymptotic = false;
    conservative_order = minimum_supported_order;
  }
  const std::size_t coarse_factor =
      levels[pair].geometry.refinement_factor;
  const std::size_t fine_factor =
      levels[pair + 1U].geometry.refinement_factor;
  const double calibrated_order =
      configuration_.adaptive_richardson_order_source == "calibrated_table"
          ? adaptive_grid::richardson_order_for_pair(
                configuration_.adaptive_richardson_order_table,
                coarse_factor, fine_factor)
          : conservative_order;
  const double minimum_observed_order = std::max(
      0.25, 0.25 * calibrated_order);
  if (!(conservative_order >= minimum_observed_order)) asymptotic = false;
  observed_orders[pair] = conservative_order;
  asymptotic_pairs[pair] = asymptotic ? 1U : 0U;
  const double order =
      configuration_.adaptive_richardson_order_source == "three_level"
          ? observed_orders[pair]
          : adaptive_grid::richardson_order_for_pair(
                configuration_.adaptive_richardson_order_table,
                coarse_factor, fine_factor);
  pair_estimates[pair] = adaptive_grid::estimate_same_particle_field_error(
      levels[pair].fields1, levels[pair].fields2,
      levels[pair + 1U].fields1, levels[pair + 1U].fields2,
      fine_factor / coarse_factor, order,
      configuration_.adaptive_field_floor,
      &accepted_targets1, &accepted_targets2);
  if (!pair_estimates[pair].reliable)
    throw std::runtime_error(
        "Mode-2 same-particle Richardson estimator is unreliable");
  if (source_hierarchy.model == "fixed_width_bspline1_v1") {
    pair_estimates[pair].coarse_error = std::max(
        pair_estimates[pair].coarse_error_rms,
        pair_estimates[pair].observed_difference_linf);
    pair_estimates[pair].fine_error = std::max(
        pair_estimates[pair].fine_error_rms,
        pair_estimates[pair].observed_difference_linf);
  }
  pair_estimate_ready[pair] = 1U;
}

// Require both sampling diagnostics to meet their declared floor.
bool Workspace::resolution_sampling_ok(const LevelSolution& level) {
  return configuration_.adaptive_min_neff <= 0.0 ||
      (level.effective_particles_per_cell >=
           configuration_.adaptive_min_neff &&
       level.minimum_effective_particles_per_cell >=
           configuration_.adaptive_min_neff);
}

// Check charge curvature and field gradient refinement indicators.
bool Workspace::resolution_indicators_ok(const LevelSolution& level) {
  return level.charge_curvature_score <=
             configuration_.adaptive_tau_curvature *
                 configuration_.adaptive_refine_threshold &&
         level.field_gradient_score <=
             configuration_.adaptive_tau_field_gradient *
                 configuration_.adaptive_refine_threshold;
}

// Check the independent sampling support of the certifying triplet.
bool Workspace::resolution_pair_sampling_ok(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& levels = state.levels;
  auto& triplet_first_pairs = state.triplet_first_pairs;
  auto& triplet_second_pairs = state.triplet_second_pairs;
  auto& triplet_valid = state.triplet_valid;
  if (!triplet_valid[pair])
    return resolution_sampling_ok(levels[pair]) && resolution_sampling_ok(levels[pair + 1U]);
  const std::size_t first_pair = triplet_first_pairs[pair];
  const std::size_t second_pair = triplet_second_pairs[pair];
  bool adequate = resolution_sampling_ok(levels[first_pair]) &&
                  resolution_sampling_ok(levels[first_pair + 1U]) &&
                  resolution_sampling_ok(levels[second_pair + 1U]);
  return adequate;
}

// Bound empirical tail support by distinct physical source roots.
bool Workspace::resolution_quantile_support_ok(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& output = state.binding.output;
  auto& vector_error_policy = state.vector_error_policy;
  if (!vector_error_policy) return true;
  const adaptive_grid::VectorFieldErrorStatistics& statistics =
      raw_pair_vector_statistics[pair];
  const long double tail_probability =
      1.0L - static_cast<long double>(
                   configuration_.adaptive_error_quantile);
  const std::size_t minimum_targets = static_cast<std::size_t>(
      std::ceil(1.0L / tail_probability));
  std::size_t independent_targets = std::min(
      statistics.accepted_targets[0], statistics.accepted_targets[1]);
  // Mode 0 has no separate source-population receipt: every accepted
  // same-particle target is an independent physical root.  When adaptive
  // source populations are enabled, cap support by the actual root receipt
  // as well, never by the (possibly cloned) representative count.
  if (output.adaptive_macroparticle_mode != 0) {
    if (output.adaptive_macroparticle_roots1 == 0U ||
        output.adaptive_macroparticle_roots2 == 0U)
      independent_targets = 0U;
    else
      independent_targets = std::min(
          independent_targets,
          std::min(output.adaptive_macroparticle_roots1,
                   output.adaptive_macroparticle_roots2));
  }
  output.vector_distinct_roots = independent_targets;
  output.vector_minimum_quantile_roots = minimum_targets;
  return independent_targets >= minimum_targets;
}

// Certify the source luminosity against its declared resolution oracle.
double Workspace::resolution_luminosity_error(
    ResolutionControllerState& state,
    std::size_t level_index) {
  auto& levels = state.levels;
  auto& source_hierarchy = state.source_hierarchy;
  auto& source_lumi_oracle_errors = state.source_lumi_oracle_errors;
  auto& source_lumi_oracle_ready = state.source_lumi_oracle_ready;
  auto& spacing_factors = state.spacing_factors;
  auto& source_level_index = state.source_level_index;
  auto& output = state.binding.output;
  auto& noise_aware_stopped = state.noise_aware_stopped;
  const bool audited_source =
      source_hierarchy.model == "reference_cic_v1" ||
      source_hierarchy.model == "fixed_width_bspline1_v1";
  if (!audited_source) return 0.0;
  if (level_index >= source_lumi_oracle_errors.size())
    throw std::out_of_range(
        "coherent-source luminosity-oracle level is invalid");
  if (source_lumi_oracle_ready[level_index])
    return source_lumi_oracle_errors[level_index];
  double error = 0.0;
  if (source_hierarchy.model == "reference_cic_v1") {
    ensure_resolution_levels(state, std::max(level_index, source_level_index) + 1U);
    const LevelSolution& candidate = levels[level_index];
    const LevelSolution& identity = levels[source_level_index];
    const double candidate_factor = static_cast<double>(
        candidate.geometry.refinement_factor);
    const double identity_factor = static_cast<double>(
        identity.geometry.refinement_factor);
    const double candidate_lumi = std::max(0.0, candidate.raw_lumi_dot) *
                                  candidate_factor * candidate_factor;
    const double identity_lumi = std::max(0.0, identity.raw_lumi_dot) *
                                 identity_factor * identity_factor;
    error = std::fabs(candidate_lumi - identity_lumi) /
        std::max(std::fabs(identity_lumi),
                 configuration_.adaptive_lumi_floor);
  } else {
    // A continuous fixed-width source has no distinguished raster at the
    // resource boundary.  Certify its luminosity with the same rolling
    // three-level contraction principle as the field instead of forcing
    // every encounter to solve the finest capacity-admitted level.
    if (spacing_factors.size() < 3U)
      throw std::logic_error(
          "fixed-width source luminosity needs three spacing levels");
    const std::size_t available_level_count = noise_aware_stopped
        ? levels.size() : spacing_factors.size();
    if (available_level_count < 3U)
      throw std::logic_error(
          "fixed-width source luminosity has fewer than three computed "
          "levels");
    const std::size_t first = level_index + 2U < available_level_count
        ? level_index : available_level_count - 3U;
    ensure_resolution_levels(state, first + 3U);
    const auto normalized_lumi = [](const LevelSolution& level) {
      const double factor = static_cast<double>(
          level.geometry.refinement_factor);
      const double raw_roundoff_floor = -64.0 *
          std::numeric_limits<double>::epsilon() *
          std::max(1.0, std::fabs(level.raw_lumi_dot));
      if (!finite_value(level.raw_lumi_dot) ||
          level.raw_lumi_dot < raw_roundoff_floor)
        throw std::runtime_error(
            "fixed-width luminosity oracle received a non-finite or "
            "negative dot product");
      const double luminosity =
          std::max(0.0, level.raw_lumi_dot) * factor * factor;
      if (!finite_value(luminosity))
        throw std::runtime_error(
            "fixed-width luminosity oracle normalization overflowed");
      return luminosity;
    };
    const double lumi0 = normalized_lumi(levels[first]);
    const double lumi1 = normalized_lumi(levels[first + 1U]);
    const double lumi2 = normalized_lumi(levels[first + 2U]);
    const std::size_t target_slot = level_index - first;
    const double ratio = static_cast<double>(
        levels[first + 1U].geometry.refinement_factor) /
        static_cast<double>(levels[first].geometry.refinement_factor);
    const runtime_safety::LuminosityOracleEstimate estimate =
        runtime_safety::estimate_fixed_width_luminosity_error(
            lumi0, lumi1, lumi2, target_slot, ratio,
            configuration_.adaptive_lumi_floor);
    error = estimate.relative_error;
  }
  if (!finite_value(error))
    throw std::runtime_error(
        "coherent-source uniform luminosity oracle is non-finite");
  source_lumi_oracle_errors[level_index] = error;
  source_lumi_oracle_ready[level_index] = 1U;
  output.source_maximum_lumi_oracle_error = std::max(
      output.source_maximum_lumi_oracle_error, error);
  maximum_source_lumi_oracle_error_ = std::max(
      maximum_source_lumi_oracle_error_, error);
  return error;
}

// Apply the independent coherent-source luminosity tolerance.
bool Workspace::resolution_luminosity_ok(
    ResolutionControllerState& state,
    std::size_t level_index) {
  auto& source_hierarchy = state.source_hierarchy;
  return (source_hierarchy.model != "reference_cic_v1" &&
          source_hierarchy.model != "fixed_width_bspline1_v1") ||
      resolution_luminosity_error(state, level_index) <=
          configuration_.adaptive_source_error_target;
}

// Select the coarse-level error under the requested norm policy.
double Workspace::resolution_coarse_error(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& pair_estimates = state.pair_estimates;
  auto& vector_coarse_errors = state.vector_coarse_errors;
  auto& vector_error_policy = state.vector_error_policy;
  return vector_error_policy ? vector_coarse_errors[pair]
                             : pair_estimates[pair].coarse_error;
}

// Select the fine-level error under the requested norm policy.
double Workspace::resolution_fine_error(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& pair_estimates = state.pair_estimates;
  auto& vector_fine_errors = state.vector_fine_errors;
  auto& vector_error_policy = state.vector_error_policy;
  return vector_error_policy ? vector_fine_errors[pair]
                             : pair_estimates[pair].fine_error;
}

// Apply the policy-specific guard on extreme target errors.
bool Workspace::resolution_catastrophe_ok(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& vector_error_policy = state.vector_error_policy;
  auto& mixed_vector_error_policy = state.mixed_vector_error_policy;
  if (!vector_error_policy) return true;
  const adaptive_grid::VectorFieldErrorStatistics& statistics =
      raw_pair_vector_statistics[pair];
  return mixed_vector_error_policy
      ? statistics.total_catastrophe_targets == 0U
      : statistics.aggregate_maximum <=
            configuration_.adaptive_error_catastrophe_limit;
}

// Bound the fraction of targets above the declared error target.
bool Workspace::resolution_exposure_ok(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& vector_error_policy = state.vector_error_policy;
  auto& mixed_vector_error_policy = state.mixed_vector_error_policy;
  return !vector_error_policy || !mixed_vector_error_policy ||
      raw_pair_vector_statistics[pair].maximum_beam_above_target_exposure <=
          configuration_.adaptive_error_exposure_limit;
}

// Publish all vector-error components for the accepted comparison.
void Workspace::publish_resolution_vector_error(
    ResolutionControllerState& state,
    std::size_t pair) {
  auto& raw_pair_vector_statistics = state.raw_pair_vector_statistics;
  auto& output = state.binding.output;
  auto& vector_error_policy = state.vector_error_policy;
  auto& mixed_vector_error_policy = state.mixed_vector_error_policy;
  if (!vector_error_policy) return;
  const adaptive_grid::VectorFieldErrorStatistics& statistics =
      raw_pair_vector_statistics[pair];
  output.vector_difference_rms = statistics.aggregate_rms;
  output.vector_difference_quantile = statistics.aggregate_quantile;
  output.vector_mixed_difference_maximum =
      statistics.aggregate_maximum;
  output.vector_absolute_difference_rms =
      statistics.aggregate_absolute_rms;
  output.vector_absolute_difference_quantile =
      statistics.aggregate_absolute_quantile;
  output.vector_absolute_difference_maximum =
      statistics.aggregate_absolute_maximum;
  output.vector_catastrophe_targets =
      statistics.total_catastrophe_targets;
  output.vector_catastrophe_exposure =
      statistics.catastrophe_exposure;
  output.vector_difference_maximum = mixed_vector_error_policy
      ? statistics.aggregate_significant_unregularized_maximum
      : statistics.aggregate_maximum;
  output.vector_quantile_probability = statistics.quantile_probability;
  output.vector_accepted_targets = statistics.total_accepted_targets;
  output.vector_above_target = statistics.total_above_target;
  output.vector_above_target_exposure =
      statistics.above_target_exposure;
  output.vector_maximum_beam_above_target_exposure =
      statistics.maximum_beam_above_target_exposure;
  output.vector_catastrophe_guard_passed =
      mixed_vector_error_policy
          ? statistics.total_catastrophe_targets == 0U
          : statistics.aggregate_maximum <=
                configuration_.adaptive_error_catastrophe_limit;
}

// Admit the resolution ladder against index and allocation limits before solving.
std::vector<std::size_t> Workspace::plan_resolution_spacing(
    ResolutionEncounterBinding& binding) const {
  std::vector<std::size_t> spacing_factors =
      configuration_.adaptive_spacing_factors;
  if (spacing_factors.size() < 3U)
    throw std::logic_error(
        "Mode-2 controller requires at least three spacing levels");
  std::size_t planned_context_bytes = 0U;
  for (std::size_t factor : spacing_factors) {
    const adaptive_grid::FieldLevelGeometry geometry =
        resolution_geometry(binding,factor);
    const std::size_t cells = geometry.cells();
    if (configuration_.adaptive_max_cells != 0U &&
        cells > configuration_.adaptive_max_cells)
      throw std::runtime_error(
          "bootstrap adaptive spacing class exceeds "
          "adaptive_grid_max_cells");
    (void)checked_int_dimension(
        geometry.n_x, geometry.n_y,
        "bootstrap adaptive spacing class");
    planned_context_bytes = checked_sum(
        planned_context_bytes,
        adaptive_context_estimate(
            geometry.n_x, geometry.n_y,
            configuration_.adaptive_window_policy),
        "bootstrap adaptive context bank");
  }
  if (planned_context_bytes >
      configuration_.adaptive_window_policy.max_bytes)
    throw std::runtime_error(
        "bootstrap adaptive spacing classes exceed "
        "adaptive_grid_max_bytes");
  std::size_t reserved_patch_context_bytes = 0U;
  if (configuration_.adaptive_grid_mode == 3) {
    const adaptive_grid::FieldLevelGeometry finest_bootstrap_geometry =
        resolution_geometry(binding,spacing_factors.back());
    reserved_patch_context_bytes = adaptive_context_estimate(
        finest_bootstrap_geometry.n_x, finest_bootstrap_geometry.n_y,
        configuration_.adaptive_window_policy);
    if (reserved_patch_context_bytes >
            configuration_.adaptive_window_policy.max_bytes ||
        planned_context_bytes >
            configuration_.adaptive_window_policy.max_bytes -
                reserved_patch_context_bytes)
      throw std::runtime_error(
          "bootstrap Mode-3 uniform bank plus patch reserve exceeds "
          "adaptive_grid_max_bytes");
  }

  while (configuration_.adaptive_max_refinement_factor == 0U ||
         spacing_factors.back() <
             configuration_.adaptive_max_refinement_factor) {
    if (spacing_factors.back() >
        std::numeric_limits<std::size_t>::max() /
            configuration_.adaptive_refinement_ratio)
      break;
    const std::size_t next_factor = spacing_factors.back() *
        configuration_.adaptive_refinement_ratio;
    if (configuration_.adaptive_max_refinement_factor != 0U &&
        next_factor > configuration_.adaptive_max_refinement_factor)
      break;
    try {
      const adaptive_grid::FieldLevelGeometry geometry =
          resolution_geometry(binding,next_factor);
      const std::size_t cells = geometry.cells();
      if (configuration_.adaptive_max_cells != 0U &&
          cells > configuration_.adaptive_max_cells)
        break;
      (void)checked_int_dimension(
          geometry.n_x, geometry.n_y,
          "dynamic adaptive spacing class");
      const std::size_t bytes = adaptive_context_estimate(
          geometry.n_x, geometry.n_y,
          configuration_.adaptive_window_policy);
      const std::size_t next_patch_reserve =
          configuration_.adaptive_grid_mode == 3 ? bytes : 0U;
      if (bytes > configuration_.adaptive_window_policy.max_bytes ||
          planned_context_bytes >
              configuration_.adaptive_window_policy.max_bytes - bytes ||
          next_patch_reserve >
              configuration_.adaptive_window_policy.max_bytes - bytes -
                  planned_context_bytes)
        break;
      planned_context_bytes += bytes;
      spacing_factors.push_back(next_factor);
    } catch (const std::overflow_error&) {
      break;
    } catch (const std::invalid_argument&) {
      // At extreme factors float spacing or a legacy signed-index contract
      // can cease to be representable.  That is a capacity boundary, not a
      // converged physics result; the normal unmet-error path remains strict.
      break;
    }
  }

  return spacing_factors;
}

// Retire unmatched contexts before any solution borrows a context pointer.
void Workspace::reconcile_resolution_contexts(
    ResolutionEncounterBinding& binding,
    const std::vector<std::size_t>& spacing_factors) {
  // Context objects are cached between encounters.  Reconcile that cache
  // with the ideal bank admitted above before any LevelSolution can hold a
  // raw context pointer.  Otherwise a stale compact-window fine context can
  // consume bytes that a later wider encounter correctly budgeted for its
  // own ladder.  Exact geometry matches remain cached; everything else is
  // deterministically retired.
  for (std::vector<std::unique_ptr<AdaptiveFieldContext> >::iterator context =
           adaptive_contexts_.begin();
       context != adaptive_contexts_.end();) {
    const std::vector<std::size_t>::const_iterator factor = std::find(
        spacing_factors.begin(), spacing_factors.end(),
        (*context)->refinement_factor);
    bool keep = false;
    if (factor != spacing_factors.end()) {
      const adaptive_grid::FieldLevelGeometry geometry =
          resolution_geometry(binding,*factor);
      keep = (*context)->n_x == geometry.n_x &&
             (*context)->n_y == geometry.n_y;
    }
    if (keep) {
      ++context;
      continue;
    }
    if ((*context)->estimated_bytes > adaptive_context_bytes_)
      throw std::logic_error(
          "adaptive context reconciliation byte-accounting underflow");
    adaptive_context_bytes_ -= (*context)->estimated_bytes;
    context = adaptive_contexts_.erase(context);
    ++adaptive_context_evictions_;
  }
  if (adaptive_patch_context_) {
    if (adaptive_patch_context_->estimated_bytes > adaptive_context_bytes_)
      throw std::logic_error(
          "adaptive patch reconciliation byte-accounting underflow");
    adaptive_context_bytes_ -= adaptive_patch_context_->estimated_bytes;
    adaptive_patch_context_.reset();
    ++adaptive_context_evictions_;
  }

}

// Try coarse levels in order, retaining strict noise and contraction guards.
void Workspace::select_resolution_coarse(
    ResolutionControllerState& state,
    ResolutionSelection& selection) {
  auto& levels = state.levels;
  auto& pair_estimates = state.pair_estimates;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& triplet_valid = state.triplet_valid;
  auto& target = state.target;
  auto& vector_error_policy = state.vector_error_policy;
  auto& noise_aware_stopped = state.noise_aware_stopped;
  auto& selected = selection.selected;
  auto& met = selection.met;
  auto& noncontracting_triplet_streak = selection.noncontracting_triplet_streak;
  auto& terminal_pair = selection.terminal_pair;
  auto& accepted_estimate = selection.accepted_estimate;
  auto& output = state.binding.output;
  const std::size_t pair_count = state.raw_pairs.size();
  for (std::size_t index = 0U; index < pair_count; ++index) {
    certify_resolution_pair(state, index);
    const bool empirical_quantile_support =
        resolution_quantile_support_ok(state, index);
    const bool adequate_sampling = resolution_pair_sampling_ok(state, index) &&
                                    empirical_quantile_support;
    if (vector_error_policy)
      output.noise_limited = !adequate_sampling;
    else if (!adequate_sampling)
      output.noise_limited = true;
    if (vector_error_policy && triplet_valid[index] &&
        !asymptotic_pairs[index])
      ++noncontracting_triplet_streak;
    else
      noncontracting_triplet_streak = 0U;
    if (resolution_coarse_error(state, index) <= target &&
        resolution_indicators_ok(levels[index]) && adequate_sampling &&
        asymptotic_pairs[index] && resolution_catastrophe_ok(state, index) &&
        resolution_exposure_ok(state, index) &&
        resolution_luminosity_ok(state, index)) {
      selected = index;
      accepted_estimate = pair_estimates[index];
      output.estimated_field_error = resolution_coarse_error(state, index);
      output.source_lumi_oracle_error =
          resolution_luminosity_error(state, index);
      publish_resolution_vector_error(state, index);
      if (vector_error_policy)
        output.vector_catastrophe_guard_passed =
            resolution_catastrophe_ok(state, index);
      met = true;
      break;
    }
    // This optional pilot-only stop is not a convergence shortcut.  It acts
    // only when a same-regime triplet has too few distinct roots even to
    // resolve the requested empirical tail (n*(1-p)<1), or after two rolling
    // noncontracting triplets also fail the pre-registered N_eff floor.
    // Smaller cells cannot create independent roots.  Stop at the finest
    // already-computed level, retain error_target_met=0, and let the
    // post-publication controller request a fresh-root replay.
    const bool empirical_quantile_root_limited =
        vector_error_policy && !empirical_quantile_support;
    const bool persistent_under_sampled_noncontraction =
        !adequate_sampling && noncontracting_triplet_streak >= 2U;
    if (configuration_.adaptive_noise_aware_stop &&
        triplet_valid[index] &&
        (empirical_quantile_root_limited ||
         persistent_under_sampled_noncontraction)) {
      terminal_pair = levels.size() - 2U;
      certify_resolution_pair(state, terminal_pair);
      noise_aware_stopped = true;
      output.noise_aware_stop_triggered = true;
      output.noise_aware_stop_factor =
          levels.back().geometry.refinement_factor;
      output.noise_aware_stop_quantile_root_limited =
          empirical_quantile_root_limited;
      output.noise_aware_stop_noncontracting =
          persistent_under_sampled_noncontraction;
      output.noise_limited = true;
      ++mode2_noise_aware_stop_encounters_;
      break;
    }
  }
}

// Accept a terminal uniform level only after its independent fine-error audit.
void Workspace::select_resolution_terminal(
    ResolutionControllerState& state,
    ResolutionSelection& selection) {
  auto& levels = state.levels;
  auto& raw_pair_sensitivities = state.raw_pair_sensitivities;
  auto& pair_estimates = state.pair_estimates;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& triplet_first_pairs = state.triplet_first_pairs;
  auto& triplet_second_pairs = state.triplet_second_pairs;
  auto& triplet_valid = state.triplet_valid;
  auto& target = state.target;
  auto& vector_error_policy = state.vector_error_policy;
  auto& noise_aware_stopped = state.noise_aware_stopped;
  auto& selected = selection.selected;
  auto& met = selection.met;
  auto& finest_asymptotic = selection.finest_asymptotic;
  auto& mode3_terminal_uniform = selection.mode3_terminal_uniform;
  auto& terminal_pair = selection.terminal_pair;
  auto& accepted_estimate = selection.accepted_estimate;
  auto& output = state.binding.output;
  if (!met) {
    certify_resolution_pair(state, terminal_pair);
    if (triplet_valid[terminal_pair]) {
      const std::size_t previous_pair =
          triplet_first_pairs[terminal_pair];
      const std::size_t latest_pair =
          triplet_second_pairs[terminal_pair];
      compare_resolution_pair(state, previous_pair);
      compare_resolution_pair(state, latest_pair);
      output.tail_field_sensitivity_previous_pair =
          raw_pair_sensitivities[previous_pair].aggregate;
      output.tail_field_sensitivity_latest_pair =
          raw_pair_sensitivities[latest_pair].aggregate;
      output.tail_field_sensitivity = std::max(
          output.tail_field_sensitivity_previous_pair,
          output.tail_field_sensitivity_latest_pair);
    }
    const bool asymptotic = asymptotic_pairs[terminal_pair] != 0U;
    finest_asymptotic = asymptotic;
    const bool empirical_quantile_support =
        resolution_quantile_support_ok(state, terminal_pair);
    const bool adequate_sampling = resolution_pair_sampling_ok(state, terminal_pair) &&
                                    empirical_quantile_support;
    if (vector_error_policy)
      output.noise_limited = !adequate_sampling || noise_aware_stopped;
    else if (!adequate_sampling)
      output.noise_limited = true;
    if (!noise_aware_stopped &&
        resolution_fine_error(state, terminal_pair) <= target && asymptotic &&
        resolution_indicators_ok(levels.back()) && adequate_sampling &&
        resolution_catastrophe_ok(state, terminal_pair) &&
        resolution_exposure_ok(state, terminal_pair) &&
        resolution_luminosity_ok(state, levels.size() - 1U)) {
      selected = levels.size() - 1U;
      accepted_estimate = pair_estimates[terminal_pair];
      output.estimated_field_error = resolution_fine_error(state, terminal_pair);
      output.source_lumi_oracle_error =
          resolution_luminosity_error(state, selected);
      publish_resolution_vector_error(state, terminal_pair);
      if (vector_error_policy)
        output.vector_catastrophe_guard_passed =
            resolution_catastrophe_ok(state, terminal_pair);
      met = true;
      mode3_terminal_uniform = configuration_.adaptive_grid_mode == 3;
    }
  }
}

// Account for an unresolved encounter and apply its explicit failure policy.
void Workspace::handle_resolution_unmet_error(
    ResolutionControllerState& state,
    ResolutionSelection& selection,
    bool terminal_patch_requested_by_card) {
  auto& levels = state.levels;
  auto& raw_pairs = state.raw_pairs;
  auto& pair_estimates = state.pair_estimates;
  auto& pair_estimate_ready = state.pair_estimate_ready;
  auto& observed_orders = state.observed_orders;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& triplet_valid = state.triplet_valid;
  auto& vector_error_policy = state.vector_error_policy;
  auto& noise_aware_stopped = state.noise_aware_stopped;
  auto& selected = selection.selected;
  auto& met = selection.met;
  auto& finest_asymptotic = selection.finest_asymptotic;
  auto& mode3_terminal_uniform = selection.mode3_terminal_uniform;
  auto& terminal_pair = selection.terminal_pair;
  auto& accepted_estimate = selection.accepted_estimate;
  auto& output = state.binding.output;
  auto& slice1 = state.binding.slice1;
  auto& slice2 = state.binding.slice2;
  if (!met) {
    output.error_target_met = false;
    ++mode2_unmet_error_encounters_;
    output.estimated_field_error = resolution_fine_error(state, terminal_pair);
    selected = levels.size() - 1U;
    output.source_lumi_oracle_error =
        resolution_luminosity_error(state, selected);
    accepted_estimate = pair_estimates[terminal_pair];
    publish_resolution_vector_error(state, terminal_pair);
    if (vector_error_policy)
      output.vector_catastrophe_guard_passed =
          resolution_catastrophe_ok(state, terminal_pair);
    mode3_terminal_uniform = configuration_.adaptive_grid_mode == 3;
    // A requested terminal two-patch certificate gets first refusal.  Do not
    // spend the independent longitudinal tail budget or throw the ordinary
    // uniform-ladder failure until that local proof has either closed or
    // rejected.  This does not weaken the default path because the feature
    // is off unless explicitly requested.
    if (!terminal_patch_requested_by_card &&
        output.encounter_tail_budget_eligible) {
      if (!triplet_valid[terminal_pair] ||
          !finite_value(output.tail_field_sensitivity) ||
          output.tail_field_sensitivity < 0.0 ||
          output.tail_field_sensitivity > 2.0)
        throw std::runtime_error(
            "Mode 2/3 encounter-tail sensitivity has no reliable "
            "same-regime terminal triplet");
      output.encounter_tail_budget_used = true;
      ++mode2_tail_budget_used_encounters_;
      const double weighted_field_error =
          output.encounter_population_weight * output.estimated_field_error;
      const long double weighted_field_sensitivity =
          static_cast<long double>(output.encounter_population_weight) *
          static_cast<long double>(output.tail_field_sensitivity);
      const double weighted_source_lumi_error =
          output.encounter_population_weight *
          output.source_lumi_oracle_error;
      if (!finite_value(weighted_field_error) ||
          !finite_value(weighted_field_sensitivity) ||
          !finite_value(weighted_source_lumi_error) ||
          weighted_field_error < 0.0 ||
          weighted_field_sensitivity < 0.0L ||
          weighted_source_lumi_error < 0.0)
        throw std::runtime_error(
            "Mode 2/3 encounter-tail weighted error is non-finite");
      encounter_tail_weighted_field_error_ += weighted_field_error;
      encounter_tail_weighted_field_sensitivity_ +=
          weighted_field_sensitivity;
      encounter_tail_weighted_source_lumi_error_ +=
          weighted_source_lumi_error;
      if (!finite_value(encounter_tail_weighted_field_error_) ||
          !finite_value(encounter_tail_weighted_field_sensitivity_) ||
          !finite_value(encounter_tail_weighted_source_lumi_error_))
        throw std::overflow_error(
            "Mode 2/3 encounter-tail weighted error overflowed");
    }
    if (configuration_.adaptive_fail_on_unmet_error &&
        !output.encounter_tail_budget_used &&
        !terminal_patch_requested_by_card) {
      throw_resolution_failure(state, selection);
    }
  }
}

// Explain the failed certificate with the complete evaluated ladder.
void Workspace::throw_resolution_failure(
    ResolutionControllerState& state,
    ResolutionSelection& selection) {
  auto& levels = state.levels;
  auto& raw_pairs = state.raw_pairs;
  auto& pair_estimates = state.pair_estimates;
  auto& pair_estimate_ready = state.pair_estimate_ready;
  auto& observed_orders = state.observed_orders;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& vector_error_policy = state.vector_error_policy;
  auto& noise_aware_stopped = state.noise_aware_stopped;
  auto& finest_asymptotic = selection.finest_asymptotic;
  auto& terminal_pair = selection.terminal_pair;
  auto& output = state.binding.output;
  auto& slice1 = state.binding.slice1;
  auto& slice2 = state.binding.slice2;
  const LevelSolution& finest = levels.back();
  std::ostringstream message;
  message
      << "Mode-2 "
      << (noise_aware_stopped
              ? "stopped at a root-sampling-limited triplet "
              : "exhausted its resource-admissible spacing ladder ")
      << "without simultaneous "
      << (vector_error_policy
              ? "vector RMS/high-quantile Richardson, catastrophe, "
                "curvature, gradient, asymptotic, and Kish-N_eff "
              : "component RMS/Linf, curvature, gradient, asymptotic, "
                "and Kish-N_eff ")
      << "acceptance"
      << " error_norm_policy="
      << configuration_.adaptive_error_norm_policy
      << " (finest_factor=" << finest.geometry.refinement_factor
      << " configured_max_factor="
      << configuration_.adaptive_max_refinement_factor
      << " max_cells=" << configuration_.adaptive_max_cells
      << " max_bytes="
      << configuration_.adaptive_window_policy.max_bytes
      << " fine_error=" << resolution_fine_error(state, terminal_pair)
      << " fine_error_rms="
      << pair_estimates[terminal_pair].fine_error_rms
      << " fine_error_linf="
      << pair_estimates[terminal_pair].fine_error_linf
      << " vector_difference_rms="
      << output.vector_difference_rms
      << " vector_difference_quantile="
      << output.vector_difference_quantile
      << " vector_difference_maximum="
      << output.vector_difference_maximum
      << " vector_mixed_difference_maximum="
      << output.vector_mixed_difference_maximum
      << " vector_absolute_difference_rms="
      << output.vector_absolute_difference_rms
      << " vector_absolute_difference_quantile="
      << output.vector_absolute_difference_quantile
      << " vector_absolute_difference_maximum="
      << output.vector_absolute_difference_maximum
      << " vector_catastrophe_targets="
      << output.vector_catastrophe_targets
      << " vector_catastrophe_exposure="
      << output.vector_catastrophe_exposure
      << " vector_above_target_exposure_limit="
      << configuration_.adaptive_error_exposure_limit
      << " vector_quantile_probability="
      << output.vector_quantile_probability
      << " vector_accepted_targets="
      << output.vector_accepted_targets
      << " vector_above_target=" << output.vector_above_target
      << " vector_above_target_exposure="
      << output.vector_above_target_exposure
      << " vector_maximum_beam_above_target_exposure="
      << output.vector_maximum_beam_above_target_exposure
      << " vector_catastrophe_guard_passed="
      << (output.vector_catastrophe_guard_passed ? 1 : 0)
      << " noise_aware_stop_triggered="
      << (output.noise_aware_stop_triggered ? 1 : 0)
      << " noise_aware_stop_quantile_root_limited="
      << (output.noise_aware_stop_quantile_root_limited ? 1 : 0)
      << " noise_aware_stop_noncontracting="
      << (output.noise_aware_stop_noncontracting ? 1 : 0)
      << " vector_distinct_roots=" << output.vector_distinct_roots
      << " vector_minimum_quantile_roots="
      << output.vector_minimum_quantile_roots
      << " tail_field_sensitivity="
      << output.tail_field_sensitivity
      << " tail_field_sensitivity_previous_pair="
      << output.tail_field_sensitivity_previous_pair
      << " tail_field_sensitivity_latest_pair="
      << output.tail_field_sensitivity_latest_pair
      << " slice1=" << slice1
      << " slice2=" << slice2
      << " encounter_population_weight="
      << output.encounter_population_weight
      << " encounter_tail_budget_eligible="
      << (output.encounter_tail_budget_eligible ? 1 : 0)
      << " encounter_tail_selected_fraction="
      << output.encounter_tail_selected_fraction
      << " curvature=" << finest.charge_curvature_score
      << " gradient=" << finest.field_gradient_score
      << " neff_p10=" << finest.effective_particles_per_cell
      << " neff_critical_min="
      << finest.minimum_effective_particles_per_cell
      << " finest_asymptotic=" << (finest_asymptotic ? 1 : 0);
  for (std::size_t index = 0U; index < levels.size(); ++index) {
    message << " level[" << index << "]={factor="
            << levels[index].geometry.refinement_factor
            << ",curvature=" << levels[index].charge_curvature_score
            << ",gradient=" << levels[index].field_gradient_score
            << ",neff_p10=" << levels[index].effective_particles_per_cell
            << ",neff_min="
            << levels[index].minimum_effective_particles_per_cell;
    if (index < pair_estimates.size() &&
        pair_estimate_ready[index])
      message << ",coarse_error=" << pair_estimates[index].coarse_error
              << ",difference_rms="
              << pair_estimates[index].observed_difference_rms
              << ",difference_linf="
              << pair_estimates[index].observed_difference_linf
              << ",absolute_rms_components=";
    if (index < pair_estimates.size() &&
        pair_estimate_ready[index]) {
      message << '[';
      for (std::size_t component = 0U; component < 4U; ++component) {
        if (component != 0U) message << ',';
        message << raw_pairs[index].
            absolute_difference_rms_components[component];
      }
      message << "]"
              << ",absolute_linf_components=[";
      for (std::size_t component = 0U; component < 4U; ++component) {
        if (component != 0U) message << ',';
        message << raw_pairs[index].
            absolute_difference_linf_components[component];
      }
      message << ']'
              << ",observed_order=" << observed_orders[index]
              << ",asymptotic="
              << static_cast<unsigned int>(asymptotic_pairs[index]);
    }
    if (index > 0U && pair_estimate_ready[index - 1U])
      message << ",preceding_fine_error="
              << pair_estimates[index - 1U].fine_error;
    message << '}';
  }
  message << ')';
  throw std::runtime_error(message.str());
}

// Measure whole-source support, straddlers and excluded tails for a patch.
Workspace::TerminalSourceSupport Workspace::terminal_source_support(
    ResolutionEncounterBinding& binding,
    const LevelSolution& parent,
    const adaptive_grid::NestedPatchBox& candidate,
    std::vector<unsigned char>* mask1,
    std::vector<unsigned char>* mask2) {
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  const auto& plan = binding.plan;
  TerminalSourceSupport result;
  if (mask1) mask1->assign(particles1.size(), 0U);
  if (mask2) mask2->assign(particles2.size(), 0U);
  const double low_x =
      (static_cast<double>(parent.geometry.origin_x) +
       candidate.begin_x - parent.geometry.offset_x) *
      parent.geometry.delta_x;
  const double high_x =
      (static_cast<double>(parent.geometry.origin_x) +
       candidate.end_x - parent.geometry.offset_x) *
      parent.geometry.delta_x;
  const double low_y =
      (static_cast<double>(parent.geometry.origin_y) +
       candidate.begin_y - parent.geometry.offset_y) *
      parent.geometry.delta_y;
  const double high_y =
      (static_cast<double>(parent.geometry.origin_y) +
       candidate.end_y - parent.geometry.offset_y) *
      parent.geometry.delta_y;
  const double coordinate_scale = std::max(
      1.0, std::max(std::max(std::fabs(low_x), std::fabs(high_x)),
                    std::max(std::fabs(low_y), std::fabs(high_y))));
  const double tolerance = 128.0 *
      std::numeric_limits<double>::epsilon() * coordinate_scale;
  const auto inspect = [
      low_x, high_x, low_y, high_y, tolerance, this](
      const std::vector<PARTICLE*>& particles,
      const std::vector<adaptive_grid::CanonicalDepositDescriptor>&
          descriptors,
      std::vector<unsigned char>* mask,
      std::size_t& admitted,
      std::size_t& outside,
      std::size_t& total,
      std::size_t& straddlers) {
    for (std::size_t index = 0U; index < particles.size(); ++index) {
      if (!descriptors[index].accepted()) continue;
      ++total;
      float x_value = 0.0f;
      float y_value = 0.0f;
      particles[index]->XYposition(x_value, y_value);
      const double x = x_value;
      const double y = y_value;
      const double sx0 = x - configuration_.adaptive_source_width_x;
      const double sx1 = x + configuration_.adaptive_source_width_x;
      const double sy0 = y - configuration_.adaptive_source_width_y;
      const double sy1 = y + configuration_.adaptive_source_width_y;
      const bool overlaps = sx1 > low_x + tolerance &&
                            sx0 < high_x - tolerance &&
                            sy1 > low_y + tolerance &&
                            sy0 < high_y - tolerance;
      const bool inside = sx0 >= low_x - tolerance &&
                          sx1 <= high_x + tolerance &&
                          sy0 >= low_y - tolerance &&
                          sy1 <= high_y + tolerance;
      if (overlaps && !inside) {
        ++straddlers;
      } else if (inside) {
        ++admitted;
        if (mask) (*mask)[index] = 1U;
      } else {
        ++outside;
      }
    }
  };
  std::size_t outside1 = 0U;
  std::size_t outside2 = 0U;
  std::size_t total1 = 0U;
  std::size_t total2 = 0U;
  inspect(particles1, plan.beam1_sources(), mask1, result.admitted1,
          outside1, total1, result.straddlers);
  inspect(particles2, plan.beam2_sources(), mask2, result.admitted2,
          outside2, total2, result.straddlers);
  const double tail1 = total1 == 0U ? 0.0 :
      static_cast<double>(outside1) / static_cast<double>(total1);
  const double tail2 = total2 == 0U ? 0.0 :
      static_cast<double>(outside2) / static_cast<double>(total2);
  result.tail_fraction = std::max(tail1, tail2);
  return result;
}

// Bind targets to identical gather support at both terminal resolutions.
void Workspace::mask_terminal_targets(
    adaptive_grid::FieldOperator field_operator,
    const adaptive_grid::FieldLevelGeometry& patch_r,
    const adaptive_grid::FieldLevelGeometry& patch_r2,
    const std::vector<PARTICLE*>& particles,
    const std::vector<adaptive_grid::CanonicalGatherDescriptor>& descriptors,
    std::vector<unsigned char>& mask) {
  for (std::size_t index = 0U; index < particles.size(); ++index) {
    if (!descriptors[index].accepted()) continue;
    float x = 0.0f;
    float y = 0.0f;
    particles[index]->XYposition(x, y);
    mask[index] =
        adaptive_grid::map_gather_to_level(
            x, y, patch_r, field_operator).accepted &&
        adaptive_grid::map_gather_to_level(
            x, y, patch_r2, field_operator).accepted;
  }
}

// Accumulate unresolved field changes outside the candidate patch.
void Workspace::accumulate_terminal_localization(
    long double& outside_error_squared_sum,
    std::size_t& outside_unresolved,
    std::size_t& accepted_target_count,
    double& outside_error_maximum,
    double target,
    const std::vector<float>& coarse,
    const std::vector<float>& fine,
    const std::vector<unsigned char>& canonical_mask,
    const std::vector<unsigned char>& local_mask,
    std::vector<double>& outside_ratios) {
  for (std::size_t particle = 0U;
       particle < canonical_mask.size(); ++particle) {
    if (!canonical_mask[particle]) continue;
    ++accepted_target_count;
    const std::size_t index = 2U * particle;
    const double coarse_norm = std::hypot(
        static_cast<double>(coarse[index]),
        static_cast<double>(coarse[index + 1U]));
    const double fine_norm = std::hypot(
        static_cast<double>(fine[index]),
        static_cast<double>(fine[index + 1U]));
    const double difference = std::hypot(
        static_cast<double>(fine[index] - coarse[index]),
        static_cast<double>(fine[index + 1U] - coarse[index + 1U]));
    const double scale = std::max(coarse_norm, fine_norm);
    const double value = scale == 0.0 ? 0.0 : difference / scale;
    if (!finite_value(value) || value < 0.0 || value > 2.0)
      throw std::runtime_error(
          "terminal Mode-3 localization metric is invalid");
    if (!local_mask[particle]) {
      outside_error_squared_sum +=
          static_cast<long double>(value) * value;
      outside_error_maximum = std::max(
          outside_error_maximum, value);
      outside_ratios.push_back(value);
      if (value > target) ++outside_unresolved;
    }
  }
}

// Collect the independent outside-target contraction samples.
void Workspace::collect_terminal_outside_ratios(
    const std::vector<float>& coarse,
    const std::vector<float>& fine,
    const std::vector<unsigned char>& canonical_mask,
    const std::vector<unsigned char>& local_mask,
    std::vector<double>& ratios) {
  for (std::size_t particle = 0U;
       particle < canonical_mask.size(); ++particle) {
    if (!canonical_mask[particle] || local_mask[particle]) continue;
    const std::size_t index = 2U * particle;
    const long double coarse_norm = std::hypotl(
        static_cast<long double>(coarse[index]),
        static_cast<long double>(coarse[index + 1U]));
    const long double fine_norm = std::hypotl(
        static_cast<long double>(fine[index]),
        static_cast<long double>(fine[index + 1U]));
    const long double difference = std::hypotl(
        static_cast<long double>(fine[index]) - coarse[index],
        static_cast<long double>(fine[index + 1U]) -
            coarse[index + 1U]);
    const long double scale = std::max(coarse_norm, fine_norm);
    const long double raw = scale == 0.0L
        ? 0.0L : difference / scale;
    const long double tolerance = 64.0L *
        std::numeric_limits<long double>::epsilon();
    if (!std::isfinite(raw) || raw < 0.0L ||
        raw > 2.0L + tolerance)
      throw std::runtime_error(
          "terminal Mode-3 outside vector ratio is invalid");
    ratios.push_back(static_cast<double>(std::min(2.0L, raw)));
  }
}

// Integrate the parent luminosity outside the physical patch support.
double Workspace::terminal_outside_luminosity(
    long double patch_low_x,
    long double patch_high_x,
    long double patch_low_y,
    long double patch_high_y,
    const LevelSolution& level) {
  long double outside = 0.0L;
  for (std::size_t ix = 0U; ix < level.geometry.n_x; ++ix) {
    const long double x =
        (static_cast<long double>(level.geometry.origin_x) + ix +
         0.5L - level.geometry.offset_x) * level.geometry.delta_x;
    for (std::size_t iy = 0U; iy < level.geometry.n_y; ++iy) {
      const long double y =
          (static_cast<long double>(level.geometry.origin_y) + iy +
           0.5L - level.geometry.offset_y) * level.geometry.delta_y;
      if (x >= patch_low_x && x < patch_high_x &&
          y >= patch_low_y && y < patch_high_y)
        continue;
      const std::size_t cell = ix * level.geometry.n_y + iy;
      outside += std::max(
          0.0L,
          static_cast<long double>(level.context->rho1[cell]) *
              static_cast<long double>(level.context->rho2[cell]));
    }
  }
  const long double factor = level.geometry.refinement_factor;
  const long double normalized = outside * factor * factor;
  if (!finite_value(normalized) || normalized < 0.0L ||
      normalized >
          static_cast<long double>(
              std::numeric_limits<double>::max()))
    throw std::runtime_error(
        "terminal Mode-3 outside luminosity is invalid");
  return static_cast<double>(normalized);
}

// Gather composite fields only onto the certified target support.
void Workspace::gather_terminal_composite(
    adaptive_grid::FieldOperator field_operator,
    const std::vector<PARTICLE*>& particles,
    const std::vector<unsigned char>& mask,
    const adaptive_grid::FieldLevelGeometry& geometry,
    const std::vector<double>& phi,
    std::vector<float>& fields) {
  for (std::size_t index = 0U; index < particles.size(); ++index) {
    if (!mask[index]) continue;
    float x = 0.0f;
    float y = 0.0f;
    particles[index]->XYposition(x, y);
    const adaptive_grid::LevelGatherDescriptor descriptor =
        adaptive_grid::map_gather_to_level(
            x, y, geometry, field_operator);
    if (!descriptor.accepted)
      throw std::runtime_error(
          "terminal Mode-3 target escaped identical patch support");
    const std::array<float, 2> field =
        adaptive_grid::gather_level_field(
            phi.data(), geometry, descriptor, field_operator);
    fields[2U * index] = field[0];
    fields[2U * index + 1U] = field[1];
  }
}

// Solve the boundary correction and certify one terminal composite field.
Workspace::TerminalCompositeLevel Workspace::build_terminal_composite(
    ResolutionEncounterBinding& binding,
    const LevelSolution& parent,
    const long double& parent_outside_lumi_dot,
    const std::vector<unsigned char>& target_mask1,
    const std::vector<unsigned char>& target_mask2,
    std::size_t operator_guard,
    const LevelSolution& local) {
  const auto& particles1 = binding.particles1;
  const auto& particles2 = binding.particles2;
  const auto field_operator = binding.field_operator;
  TerminalCompositeLevel result;
  if (!local.context)
    return result;
  const std::size_t cells = local.geometry.cells();
  result.fields1 = parent.fields1;
  result.fields2 = parent.fields2;
  result.neff = local.minimum_effective_particles_per_cell;
  result.charge_closure = local.maximum_charge_closure;
  result.moment_closure = local.maximum_first_moment_error;
  std::uint64_t source_hash = UINT64_C(1469598103934665603);
  hash_u64(source_hash, UINT64_C(0x5445524d53524331)); // "TERMSRC1"
  hash_field_geometry(source_hash, local.geometry);
  for (PHI_FLOAT value : local.context->rho1)
    hash_double_bits(source_hash, value);
  for (PHI_FLOAT value : local.context->rho2)
    hash_double_bits(source_hash, value);
  for (PHI_FLOAT value : local.sampling_s1_1)
    hash_double_bits(source_hash, value);
  for (PHI_FLOAT value : local.sampling_s2_1)
    hash_double_bits(source_hash, value);
  for (PHI_FLOAT value : local.sampling_s1_2)
    hash_double_bits(source_hash, value);
  for (PHI_FLOAT value : local.sampling_s2_2)
    hash_double_bits(source_hash, value);
  result.source_hash = source_hash;

  std::vector<double> parent_phi1(cells, 0.0);
  std::vector<double> parent_phi2(cells, 0.0);
  std::vector<double> correction1(cells, 0.0);
  std::vector<double> correction2(cells, 0.0);
  const PHI_FLOAT* local_phi1 = local.context->field.get_phi(1);
  const PHI_FLOAT* local_phi2 = local.context->field.get_phi(2);
  const PHI_FLOAT* parent_raw_phi1 = parent.context->field.get_phi(1);
  const PHI_FLOAT* parent_raw_phi2 = parent.context->field.get_phi(2);
  for (std::size_t ix = 0U; ix < local.geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < local.geometry.n_y; ++iy) {
      const std::size_t cell = ix * local.geometry.n_y + iy;
      const float x = static_cast<float>(
          (static_cast<double>(local.geometry.origin_x) + ix + 0.5 -
           local.geometry.offset_x) * local.geometry.delta_x);
      const float y = static_cast<float>(
          (static_cast<double>(local.geometry.origin_y) + iy + 0.5 -
           local.geometry.offset_y) * local.geometry.delta_y);
      parent_phi1[cell] = adaptive_grid::interpolate_level_potential(
          parent_raw_phi1, parent.geometry, x, y);
      parent_phi2[cell] = adaptive_grid::interpolate_level_potential(
          parent_raw_phi2, parent.geometry, x, y);
      if (ix == 0U || iy == 0U ||
          ix + 1U == local.geometry.n_x ||
          iy + 1U == local.geometry.n_y) {
        correction1[cell] = parent_phi1[cell] - local_phi1[cell];
        correction2[cell] = parent_phi2[cell] - local_phi2[cell];
      }
    }
  const std::size_t maximum_iterations = std::max<std::size_t>(
      2000U, std::min<std::size_t>(
          1000000U,
          checked_product(cells, 4U,
                          "terminal Mode-3 harmonic iteration cap")));
  const std::chrono::steady_clock::time_point harmonic_start =
      std::chrono::steady_clock::now();
  const adaptive_grid::HarmonicSolveReport report1 =
      adaptive_grid::solve_harmonic_dirichlet(
          correction1, local.geometry.n_x, local.geometry.n_y,
          local.geometry.delta_x, local.geometry.delta_y,
          configuration_.adaptive_fine_residual_tolerance,
          maximum_iterations);
  const adaptive_grid::HarmonicSolveReport report2 =
      adaptive_grid::solve_harmonic_dirichlet(
          correction2, local.geometry.n_x, local.geometry.n_y,
          local.geometry.delta_x, local.geometry.delta_y,
          configuration_.adaptive_fine_residual_tolerance,
          maximum_iterations);
  mode3_harmonic_ms_ += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - harmonic_start).count();
  result.residual = std::max(
      report1.normalized_residual, report2.normalized_residual);
  checked_add_counter(
      mode3_harmonic_iterations_,
      checked_sum(report1.iterations, report2.iterations,
                  "terminal Mode-3 harmonic iterations"),
      "terminal Mode-3 harmonic iteration");
  result.harmonic_iterations = checked_sum(
      report1.iterations, report2.iterations,
      "terminal Mode-3 per-level harmonic iterations");
  if (!report1.converged || !report2.converged)
    return result;
  std::vector<double> corrected1(cells, 0.0);
  std::vector<double> corrected2(cells, 0.0);
  for (std::size_t cell = 0U; cell < cells; ++cell) {
    corrected1[cell] = local_phi1[cell] + correction1[cell];
    corrected2[cell] = local_phi2[cell] + correction2[cell];
  }
  const adaptive_grid::PatchBoundaryFluxStatistics boundary1 =
      adaptive_grid::estimate_patch_boundary_flux_mismatch(
          parent_raw_phi1, parent.geometry, corrected1,
          local.geometry, configuration_.adaptive_error_quantile,
          configuration_.adaptive_field_floor);
  const adaptive_grid::PatchBoundaryFluxStatistics boundary2 =
      adaptive_grid::estimate_patch_boundary_flux_mismatch(
          parent_raw_phi2, parent.geometry, corrected2,
          local.geometry, configuration_.adaptive_error_quantile,
          configuration_.adaptive_field_floor);
  if (!boundary1.reliable || !boundary2.reliable)
    return result;
  result.boundary_rms = std::max(boundary1.rms, boundary2.rms);
  result.boundary_quantile = std::max(
      boundary1.quantile, boundary2.quantile);
  result.boundary_maximum = std::max(
      boundary1.maximum, boundary2.maximum);
  std::vector<double> composite1(cells, 0.0);
  std::vector<double> composite2(cells, 0.0);
  const std::size_t relative_factor =
      local.geometry.refinement_factor /
      parent.geometry.refinement_factor;
  const std::size_t transition_cells = checked_product(
      configuration_.adaptive_transition_cells, relative_factor,
      "terminal Mode-3 transition width");
  for (std::size_t ix = 0U; ix < local.geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < local.geometry.n_y; ++iy) {
      const std::size_t cell = ix * local.geometry.n_y + iy;
      const float x = static_cast<float>(
          (static_cast<double>(local.geometry.origin_x) + ix + 0.5 -
           local.geometry.offset_x) * local.geometry.delta_x);
      const float y = static_cast<float>(
          (static_cast<double>(local.geometry.origin_y) + iy + 0.5 -
           local.geometry.offset_y) * local.geometry.delta_y);
      const double blend = adaptive_grid::patch_transition_weight(
          x, y, local.geometry, transition_cells, operator_guard);
      composite1[cell] = parent_phi1[cell] +
          blend * (corrected1[cell] - parent_phi1[cell]);
      composite2[cell] = parent_phi2[cell] +
          blend * (corrected2[cell] - parent_phi2[cell]);
    }

  gather_terminal_composite(field_operator, 
      particles1, target_mask1, local.geometry, composite2,
      result.fields1);
  gather_terminal_composite(field_operator, 
      particles2, target_mask2, local.geometry, composite1,
      result.fields2);
  const long double local_factor =
      static_cast<long double>(local.geometry.refinement_factor);
  const long double local_parent_factor =
      static_cast<long double>(parent.geometry.refinement_factor);
  const long double local_luminosity =
      std::max(0.0L, static_cast<long double>(local.raw_lumi_dot)) *
      local_factor * local_factor;
  const long double luminosity =
      parent_outside_lumi_dot * local_parent_factor * local_parent_factor +
      local_luminosity;
  if (!finite_value(luminosity) || luminosity < 0.0L ||
      luminosity >
          static_cast<long double>(
              std::numeric_limits<double>::max()))
    throw std::runtime_error(
        "terminal Mode-3 composite luminosity is invalid");
  result.luminosity = static_cast<double>(luminosity);
  result.local_luminosity = static_cast<double>(local_luminosity);
  result.reliable = true;
  return result;
}

// Collect matched local-target ratios without changing their identities.
void Workspace::collect_terminal_vector_ratios(
    const std::vector<float>& coarse,
    const std::vector<float>& fine,
    const std::vector<unsigned char>& mask,
    std::vector<double>& ratios) {
  ratios.clear();
  ratios.reserve(std::accumulate(
      mask.begin(), mask.end(), std::size_t(0)));
  for (std::size_t particle = 0U;
       particle < mask.size(); ++particle) {
    if (!mask[particle]) continue;
    const std::size_t index = 2U * particle;
    const long double coarse_norm = std::hypotl(
        static_cast<long double>(coarse[index]),
        static_cast<long double>(coarse[index + 1U]));
    const long double fine_norm = std::hypotl(
        static_cast<long double>(fine[index]),
        static_cast<long double>(fine[index + 1U]));
    const long double difference = std::hypotl(
        static_cast<long double>(fine[index]) - coarse[index],
        static_cast<long double>(fine[index + 1U]) -
            coarse[index + 1U]);
    const long double scale = std::max(coarse_norm, fine_norm);
    const long double raw = scale == 0.0L
        ? 0.0L : difference / scale;
    const long double tolerance = 64.0L *
        std::numeric_limits<long double>::epsilon();
    if (!std::isfinite(raw) || raw < 0.0L ||
        raw > 2.0L + tolerance)
      throw std::runtime_error(
          "terminal Mode-3 pointwise vector ratio is invalid");
    ratios.push_back(static_cast<double>(
        std::min(2.0L, raw)));
  }
}

// Certify two finer terminal patches before spending any tail-error budget.
void Workspace::certify_terminal_resolution_patch(
    ResolutionControllerState& state,
    ResolutionSelection& selection,
    bool terminal_patch_requested_by_card,
    bool& terminal_patch_accepted) {
  auto& binding = state.binding;
  auto& source_hierarchy = state.source_hierarchy;
  auto& levels = state.levels;
  auto& accepted_targets1 = state.accepted_targets1;
  auto& accepted_targets2 = state.accepted_targets2;
  auto& target = state.target;
  auto& selected = selection.selected;
  auto& met = selection.met;
  auto& mode3_terminal_uniform = selection.mode3_terminal_uniform;
  auto& output = state.binding.output;
  auto& slice1 = state.binding.slice1;
  auto& slice2 = state.binding.slice2;
  auto& particles1 = state.binding.particles1;
  auto& particles2 = state.binding.particles2;
  auto& plan = state.binding.plan;
  auto& field_operator = state.binding.field_operator;
  if (configuration_.adaptive_grid_mode == 3 && mode3_terminal_uniform &&
      terminal_patch_requested_by_card && !output.error_target_met) {
    output.terminal_patch_certification_attempted = true;
    ++mode3_terminal_patch_attempts_;
    if (selected < 2U)
      throw std::logic_error(
          "terminal Mode-3 patch lacks its rolling outside triplet");
    LevelSolution& parent = levels[selected];
    const LevelSolution& previous = levels[selected - 1U];
    const LevelSolution& preceding = levels[selected - 2U];
    const std::size_t ratio = configuration_.adaptive_refinement_ratio;
    const std::size_t block = configuration_.adaptive_blocking_factor;
    const std::size_t operator_guard =
        adaptive_grid::field_operator_transition_guard_cells(field_operator);
    std::vector<double> rho1(parent.context->rho1.begin(),
                             parent.context->rho1.end());
    std::vector<double> rho2(parent.context->rho2.begin(),
                             parent.context->rho2.end());
    adaptive_grid::NestedPatchBox patch =
        adaptive_grid::select_nested_patch(
            rho1, rho2, parent.geometry.n_x, parent.geometry.n_y,
            configuration_.adaptive_coarsen_threshold,
            configuration_.adaptive_tau_curvature,
            configuration_.adaptive_rho_floor,
            configuration_.adaptive_indicator_smoothing_passes,
            configuration_.adaptive_error_buffer_cells,
            configuration_.adaptive_transition_cells,
            block);
    output.nested_patch_requested = patch.valid;
    if (patch.valid) {
      ++mode3_patch_requests_;
    } else {
      ++mode3_no_patch_encounters_;
      output.terminal_patch_rejection_mask |=
          adaptive_grid::TerminalPatchRejectionNonlocalized;
    }

    const std::size_t interior_begin = std::max(operator_guard, block);
    const std::size_t interior_end_x =
        parent.geometry.n_x > operator_guard
            ? (parent.geometry.n_x - operator_guard) / block * block
            : 0U;
    const std::size_t interior_end_y =
        parent.geometry.n_y > operator_guard
            ? (parent.geometry.n_y - operator_guard) / block * block
            : 0U;
    if (patch.valid) {
      patch.begin_x = std::max(patch.begin_x, interior_begin);
      patch.begin_y = std::max(patch.begin_y, interior_begin);
      patch.end_x = std::min(patch.end_x, interior_end_x);
      patch.end_y = std::min(patch.end_y, interior_end_y);
      patch.valid = patch.end_x > patch.begin_x + 3U &&
                    patch.end_y > patch.begin_y + 3U;
      if (!patch.valid)
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionTransitionBuffer;
    }

    TerminalSourceSupport support;
    if (patch.valid) {
      support = terminal_source_support(binding, parent, patch, NULL, NULL);
      while ((support.straddlers != 0U ||
              support.tail_fraction > configuration_.adaptive_tau_tail)) {
        const std::size_t old_begin_x = patch.begin_x;
        const std::size_t old_begin_y = patch.begin_y;
        const std::size_t old_end_x = patch.end_x;
        const std::size_t old_end_y = patch.end_y;
        const std::size_t lower_guard = checked_sum(
            interior_begin, block, "terminal Mode-3 lower guard");
        patch.begin_x = patch.begin_x > lower_guard
            ? patch.begin_x - block : interior_begin;
        patch.begin_y = patch.begin_y > lower_guard
            ? patch.begin_y - block : interior_begin;
        patch.end_x = std::min(
            interior_end_x,
            checked_sum(patch.end_x, block,
                        "terminal Mode-3 patch end x"));
        patch.end_y = std::min(
            interior_end_y,
            checked_sum(patch.end_y, block,
                        "terminal Mode-3 patch end y"));
        if (patch.begin_x == old_begin_x &&
            patch.begin_y == old_begin_y &&
            patch.end_x == old_end_x && patch.end_y == old_end_y)
          break;
        support = terminal_source_support(binding, parent, patch, NULL, NULL);
      }
      // Keep one complete parent block between the last admitted source
      // shape and the C1 transition whenever the containing level permits.
      const std::size_t lower_guard = checked_sum(
          interior_begin, block, "terminal Mode-3 safety lower guard");
      patch.begin_x = patch.begin_x > lower_guard
          ? patch.begin_x - block : interior_begin;
      patch.begin_y = patch.begin_y > lower_guard
          ? patch.begin_y - block : interior_begin;
      patch.end_x = std::min(
          interior_end_x,
          checked_sum(patch.end_x, block,
                      "terminal Mode-3 safety end x"));
      patch.end_y = std::min(
          interior_end_y,
          checked_sum(patch.end_y, block,
                      "terminal Mode-3 safety end y"));
      support = terminal_source_support(binding, parent, patch, NULL, NULL);
      output.nested_patch_tail_fraction = support.tail_fraction;
      output.nested_patch_straddling_sources = support.straddlers;
      if (support.straddlers != 0U) {
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionSourceSupport;
        patch.valid = false;
      }
      if (support.tail_fraction > configuration_.adaptive_tau_tail) {
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionNonlocalized;
        patch.valid = false;
      }
      const std::size_t minimum_span = checked_product(
          2U,
          checked_sum(configuration_.adaptive_transition_cells,
                      operator_guard,
                      "terminal Mode-3 minimum transition span"),
          "terminal Mode-3 minimum patch span");
      if (patch.n_x() < minimum_span || patch.n_y() < minimum_span) {
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionTransitionBuffer;
        patch.valid = false;
      }
    }

    if (patch.valid) {
      const adaptive_grid::FieldLevelGeometry coarse_patch =
          adaptive_grid::make_patch_geometry(parent.geometry, patch);
      const adaptive_grid::FieldLevelGeometry patch_r =
          adaptive_grid::refine_patch_geometry(coarse_patch, ratio);
      const adaptive_grid::FieldLevelGeometry patch_r2 =
          adaptive_grid::refine_patch_geometry(patch_r, ratio);
      output.terminal_patch_factor_r = patch_r.refinement_factor;
      output.terminal_patch_factor_r2 = patch_r2.refinement_factor;
      output.terminal_patch_cells_r = patch_r.cells();
      output.terminal_patch_cells_r2 = patch_r2.cells();
      output.terminal_patch_geometry_hash_r =
          adaptive_result_geometry_hash(
              configuration_.adaptive_grid_mode, parent.geometry,
              &patch_r, field_operator);
      output.terminal_patch_geometry_hash_r2 =
          adaptive_result_geometry_hash(
              configuration_.adaptive_grid_mode, parent.geometry,
              &patch_r2, field_operator);

      if (adaptive_patch_context_ &&
          adaptive_patch_context_->estimated_bytes > adaptive_context_bytes_)
        throw std::logic_error(
            "terminal Mode-3 patch context byte accounting underflow");
      const std::size_t retained_context_bytes = adaptive_patch_context_
          ? adaptive_context_bytes_ -
                adaptive_patch_context_->estimated_bytes
          : adaptive_context_bytes_;
      const std::size_t bytes_r = adaptive_context_estimate(
          patch_r.n_x, patch_r.n_y,
          configuration_.adaptive_window_policy);
      const std::size_t bytes_r2 = adaptive_context_estimate(
          patch_r2.n_x, patch_r2.n_y,
          configuration_.adaptive_window_policy);
      const adaptive_grid::TerminalPatchResourceDecision resource_decision =
          adaptive_grid::evaluate_terminal_patch_resources(
              patch_r.refinement_factor, patch_r2.refinement_factor,
              patch_r.cells(), patch_r2.cells(), bytes_r, bytes_r2,
              retained_context_bytes,
              configuration_.adaptive_max_refinement_factor,
              configuration_.adaptive_max_cells,
              configuration_.adaptive_window_policy.max_bytes);
      output.terminal_patch_rejection_mask |=
          resource_decision.rejection_mask;

      std::vector<unsigned char> source_mask1;
      std::vector<unsigned char> source_mask2;
      support = terminal_source_support(binding, parent, 
          patch, &source_mask1, &source_mask2);
      std::vector<unsigned char> target_mask1(particles1.size(), 0U);
      std::vector<unsigned char> target_mask2(particles2.size(), 0U);

      mask_terminal_targets(field_operator, patch_r, patch_r2, 
          particles1, plan.beam1_targets(), target_mask1);
      mask_terminal_targets(field_operator, patch_r, patch_r2, 
          particles2, plan.beam2_targets(), target_mask2);
      std::vector<unsigned char> outside_target_mask1(
          target_mask1.size(), 0U);
      std::vector<unsigned char> outside_target_mask2(
          target_mask2.size(), 0U);
      for (std::size_t index = 0U; index < target_mask1.size(); ++index)
        outside_target_mask1[index] =
            accepted_targets1[index] && !target_mask1[index];
      for (std::size_t index = 0U; index < target_mask2.size(); ++index)
        outside_target_mask2[index] =
            accepted_targets2[index] && !target_mask2[index];
      const std::size_t local_sources = checked_sum(
          std::accumulate(source_mask1.begin(), source_mask1.end(),
                          std::size_t(0)),
          std::accumulate(source_mask2.begin(), source_mask2.end(),
                          std::size_t(0)),
          "terminal Mode-3 local source count");
      const std::size_t local_targets = checked_sum(
          std::accumulate(target_mask1.begin(), target_mask1.end(),
                          std::size_t(0)),
          std::accumulate(target_mask2.begin(), target_mask2.end(),
                          std::size_t(0)),
          "terminal Mode-3 local target count");
      if (local_sources == 0U || local_targets == 0U)
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionNonlocalized;

      long double outside_error_squared_sum = 0.0L;
      std::size_t outside_unresolved = 0U;
      std::size_t accepted_target_count = 0U;
      double outside_error_maximum = 0.0;
      std::array<std::vector<double>, 2> outside_error_ratios;
      std::array<std::vector<double>, 2>
          outside_earlier_to_previous_ratios;

      accumulate_terminal_localization(outside_error_squared_sum, outside_unresolved, accepted_target_count, outside_error_maximum, target, 
          previous.fields1, parent.fields1, accepted_targets1,
          target_mask1, outside_error_ratios[0]);
      accumulate_terminal_localization(outside_error_squared_sum, outside_unresolved, accepted_target_count, outside_error_maximum, target, 
          previous.fields2, parent.fields2, accepted_targets2,
          target_mask2, outside_error_ratios[1]);
      const std::array<std::uint64_t, 2> outside_latest_target_hash{{
          adaptive_grid::accepted_target_identity_hash(
              outside_target_mask1),
          adaptive_grid::accepted_target_identity_hash(
              outside_target_mask2)}};

      collect_terminal_outside_ratios(
          preceding.fields1, previous.fields1, accepted_targets1,
          target_mask1, outside_earlier_to_previous_ratios[0]);
      collect_terminal_outside_ratios(
          preceding.fields2, previous.fields2, accepted_targets2,
          target_mask2, outside_earlier_to_previous_ratios[1]);
      const std::array<std::uint64_t, 2> outside_earlier_target_hash{{
          adaptive_grid::accepted_target_identity_hash(
              outside_target_mask1),
          adaptive_grid::accepted_target_identity_hash(
              outside_target_mask2)}};
      if (accepted_target_count == 0U) {
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionInvalidEvidence;
      } else {
        output.terminal_patch_outside_unresolved_exposure =
            static_cast<double>(outside_unresolved) /
            static_cast<double>(accepted_target_count);
        output.terminal_patch_outside_field_error_contribution =
            std::sqrt(static_cast<double>(
                outside_error_squared_sum /
                static_cast<long double>(accepted_target_count)));
        output.terminal_patch_outside_field_maximum =
            outside_error_maximum;
      }

      long double parent_total_lumi_dot = 0.0L;
      long double parent_outside_lumi_dot = 0.0L;
      for (std::size_t ix = 0U; ix < parent.geometry.n_x; ++ix)
        for (std::size_t iy = 0U; iy < parent.geometry.n_y; ++iy) {
          const std::size_t cell = ix * parent.geometry.n_y + iy;
          const long double contribution = std::max(
              0.0L,
              static_cast<long double>(parent.context->rho1[cell]) *
                  static_cast<long double>(parent.context->rho2[cell]));
          parent_total_lumi_dot += contribution;
          if (ix < patch.begin_x || ix >= patch.end_x ||
              iy < patch.begin_y || iy >= patch.end_y)
            parent_outside_lumi_dot += contribution;
        }
      const long double patch_low_x =
          (static_cast<long double>(parent.geometry.origin_x) +
           patch.begin_x - parent.geometry.offset_x) *
          parent.geometry.delta_x;
      const long double patch_high_x =
          (static_cast<long double>(parent.geometry.origin_x) +
           patch.end_x - parent.geometry.offset_x) *
          parent.geometry.delta_x;
      const long double patch_low_y =
          (static_cast<long double>(parent.geometry.origin_y) +
           patch.begin_y - parent.geometry.offset_y) *
          parent.geometry.delta_y;
      const long double patch_high_y =
          (static_cast<long double>(parent.geometry.origin_y) +
           patch.end_y - parent.geometry.offset_y) *
          parent.geometry.delta_y;

      const long double parent_factor =
          parent.geometry.refinement_factor;
      const long double normalized_parent_total =
          parent_total_lumi_dot * parent_factor * parent_factor;
      const long double normalized_parent_outside =
          parent_outside_lumi_dot * parent_factor * parent_factor;
      if (!finite_value(normalized_parent_total) ||
          !finite_value(normalized_parent_outside) ||
          normalized_parent_total < 0.0L ||
          normalized_parent_outside < 0.0L ||
          normalized_parent_outside > normalized_parent_total)
        throw std::runtime_error(
            "terminal Mode-3 parent luminosity partition is invalid");
      output.terminal_patch_outside_luminosity_earlier =
          terminal_outside_luminosity(patch_low_x, patch_high_x, patch_low_y, patch_high_y, preceding);
      output.terminal_patch_outside_luminosity_previous =
          terminal_outside_luminosity(patch_low_x, patch_high_x, patch_low_y, patch_high_y, previous);
      output.terminal_patch_outside_luminosity_parent =
          static_cast<double>(normalized_parent_outside);
      output.terminal_patch_outside_luminosity_fraction =
          normalized_parent_total > 0.0L
              ? static_cast<double>(normalized_parent_outside /
                                    normalized_parent_total)
              : 0.0;
      const double parent_local_luminosity = static_cast<double>(
          normalized_parent_total - normalized_parent_outside);

      if (output.terminal_patch_rejection_mask == 0U) {
        const LevelSolution local_r = solve_field_level(
            slice1, slice2, plan, patch_r,
            &source_mask1, &source_mask2, &target_mask1, &target_mask2,
            true, &source_hierarchy, NULL);
        TerminalCompositeLevel composite_r = build_terminal_composite(binding, parent, parent_outside_lumi_dot, target_mask1, target_mask2, operator_guard, local_r);
        output.source_build_ms += local_r.source_build_ms;
        output.terminal_patch_source_hash_r = composite_r.source_hash;
        const LevelSolution local_r2 = solve_field_level(
            slice1, slice2, plan, patch_r2,
            &source_mask1, &source_mask2, &target_mask1, &target_mask2,
            true, &source_hierarchy, NULL);
        TerminalCompositeLevel composite_r2 = build_terminal_composite(binding, parent, parent_outside_lumi_dot, target_mask1, target_mask2, operator_guard, local_r2);
        output.source_build_ms += local_r2.source_build_ms;
        output.terminal_patch_source_hash_r2 = composite_r2.source_hash;
        output.source_maximum_charge_closure = std::max(
            output.source_maximum_charge_closure,
            std::max(composite_r.charge_closure,
                     composite_r2.charge_closure));
        output.source_maximum_projection_moment_shift = std::max(
            output.source_maximum_projection_moment_shift,
            std::max(composite_r.moment_closure,
                     composite_r2.moment_closure));
        output.nested_patch_effective_particles_per_cell =
            std::min(local_r.effective_particles_per_cell,
                     local_r2.effective_particles_per_cell);
        output.nested_patch_minimum_effective_particles_per_cell =
            std::min(composite_r.neff, composite_r2.neff);
        output.nested_patch_residual = std::max(
            composite_r.residual, composite_r2.residual);
        output.nested_patch_iterations = checked_sum(
            composite_r.harmonic_iterations,
            composite_r2.harmonic_iterations,
            "terminal Mode-3 encounter harmonic iterations");
        output.terminal_patch_boundary_flux_rms = std::max(
            composite_r.boundary_rms, composite_r2.boundary_rms);
        output.terminal_patch_boundary_flux_quantile = std::max(
            composite_r.boundary_quantile,
            composite_r2.boundary_quantile);
        output.terminal_patch_boundary_flux_maximum = std::max(
            composite_r.boundary_maximum,
            composite_r2.boundary_maximum);
        output.nested_patch_coarse_cells = patch_r.cells();
        output.nested_patch_fine_cells = patch_r2.cells();
        output.nested_patch_coarse_delta_x = patch_r.delta_x;
        output.nested_patch_coarse_delta_y = patch_r.delta_y;
        output.nested_patch_fine_delta_x = patch_r2.delta_x;
        output.nested_patch_fine_delta_y = patch_r2.delta_y;
        checked_add_counter(mode3_attempted_patch_coarse_cells_,
                            patch_r.cells(),
                            "terminal Mode-3 attempted r-patch cell");
        checked_add_counter(mode3_attempted_patch_fine_cells_,
                            patch_r2.cells(),
                            "terminal Mode-3 attempted r2-patch cell");
        maximum_patch_residual_ = std::max(
            maximum_patch_residual_, output.nested_patch_residual);

        if (!composite_r.reliable || !composite_r2.reliable)
          output.terminal_patch_rejection_mask |=
              adaptive_grid::TerminalPatchRejectionHarmonicResidual;
        if (output.terminal_patch_rejection_mask == 0U) {
          const adaptive_grid::VectorFieldErrorStatistics parent_to_r =
              adaptive_grid::estimate_vector_field_error_statistics(
                  parent.fields1, parent.fields2,
                  composite_r.fields1, composite_r.fields2,
                  configuration_.adaptive_error_quantile, target,
                  &target_mask1, &target_mask2);
          const adaptive_grid::VectorFieldErrorStatistics r_to_r2 =
              adaptive_grid::estimate_vector_field_error_statistics(
                  composite_r.fields1, composite_r.fields2,
                  composite_r2.fields1, composite_r2.fields2,
                  configuration_.adaptive_error_quantile, target,
                  &target_mask1, &target_mask2);
          std::array<std::vector<double>, 2> local_r_to_r2_ratios;

          collect_terminal_vector_ratios(
              composite_r.fields1, composite_r2.fields1,
              target_mask1, local_r_to_r2_ratios[0]);
          collect_terminal_vector_ratios(
              composite_r.fields2, composite_r2.fields2,
              target_mask2, local_r_to_r2_ratios[1]);
          const std::array<std::uint64_t, 2> local_target_hash{{
              adaptive_grid::accepted_target_identity_hash(target_mask1),
              adaptive_grid::accepted_target_identity_hash(target_mask2)}};
          output.terminal_patch_local_target_hash1 =
              local_target_hash[0];
          output.terminal_patch_local_target_hash2 =
              local_target_hash[1];
          output.terminal_patch_outside_target_hash1 =
              outside_latest_target_hash[0];
          output.terminal_patch_outside_target_hash2 =
              outside_latest_target_hash[1];
          adaptive_grid::TerminalPatchEvidence evidence;
          evidence.parent_to_r = parent_to_r;
          evidence.r_to_r2 = r_to_r2;
          evidence.luminosity_parent = parent_local_luminosity;
          evidence.luminosity_r = composite_r.local_luminosity;
          evidence.luminosity_r2 = composite_r2.local_luminosity;
          evidence.outside_luminosity_earlier =
              output.terminal_patch_outside_luminosity_earlier;
          evidence.outside_luminosity_previous =
              output.terminal_patch_outside_luminosity_previous;
          evidence.outside_luminosity_parent =
              output.terminal_patch_outside_luminosity_parent;
          evidence.luminosity_floor = configuration_.adaptive_lumi_floor;
          evidence.outside_unresolved_exposure =
              output.terminal_patch_outside_unresolved_exposure;
          evidence.outside_field_error_contribution =
              output.terminal_patch_outside_field_error_contribution;
          evidence.local_r_to_r2_ratios.swap(local_r_to_r2_ratios);
          evidence.local_target_hash = local_target_hash;
          evidence.outside_earlier_to_previous_ratios.swap(
              outside_earlier_to_previous_ratios);
          evidence.outside_error_surrogate_ratios.swap(
              outside_error_ratios);
          evidence.outside_earlier_target_hash =
              outside_earlier_target_hash;
          evidence.outside_latest_target_hash =
              outside_latest_target_hash;
          evidence.outside_luminosity_fraction =
              output.terminal_patch_outside_luminosity_fraction;
          evidence.outside_source_tail_fraction = support.tail_fraction;
          evidence.minimum_kish_neff_r = composite_r.neff;
          evidence.minimum_kish_neff_r2 = composite_r2.neff;
          evidence.maximum_charge_closure = std::max(
              composite_r.charge_closure,
              composite_r2.charge_closure);
          evidence.maximum_moment_closure = std::max(
              composite_r.moment_closure,
              composite_r2.moment_closure);
          evidence.maximum_harmonic_residual =
              output.nested_patch_residual;
          evidence.maximum_boundary_flux_rms =
              output.terminal_patch_boundary_flux_rms;
          evidence.maximum_boundary_flux_quantile =
              output.terminal_patch_boundary_flux_quantile;
          evidence.maximum_boundary_flux_maximum =
              output.terminal_patch_boundary_flux_maximum;
          evidence.transition_cells_r = checked_product(
              configuration_.adaptive_transition_cells, ratio,
              "terminal Mode-3 r transition evidence");
          evidence.transition_cells_r2 = checked_product(
              evidence.transition_cells_r, ratio,
              "terminal Mode-3 r2 transition evidence");
          evidence.stencil_guard_r = operator_guard;
          evidence.stencil_guard_r2 = operator_guard;
          evidence.refinement_ratio = ratio;
          adaptive_grid::TerminalPatchPolicy policy;
          policy.field_error_target = target;
          policy.luminosity_error_target = std::min(
              configuration_.adaptive_source_error_target,
              configuration_.adaptive_lumi_error_target);
          policy.outside_exposure_target =
              configuration_.adaptive_tau_tail;
          policy.outside_source_tail_target =
              configuration_.adaptive_tau_tail;
          policy.outside_luminosity_exposure_target = std::min(
              configuration_.adaptive_tau_tail,
              configuration_.adaptive_lumi_error_target);
          policy.minimum_kish_neff = configuration_.adaptive_min_neff;
          policy.harmonic_residual_tolerance =
              configuration_.adaptive_fine_residual_tolerance;
          policy.boundary_flux_tolerance = target;
          policy.catastrophe_limit =
              configuration_.adaptive_error_catastrophe_limit;
          policy.quantile_probability =
              configuration_.adaptive_error_quantile;
          const adaptive_grid::TerminalPatchCertificate certificate =
              adaptive_grid::certify_terminal_patch(evidence, policy);
          output.terminal_patch_rejection_mask |=
              certificate.rejection_mask;
          output.terminal_patch_observed_field_order =
              certificate.observed_field_order;
          output.terminal_patch_observed_outside_field_order =
              certificate.observed_outside_field_order;
          output.terminal_patch_outside_unresolved_exposure =
              certificate.outside_unresolved_exposure;
          output.terminal_patch_outside_field_error_contribution =
              certificate.outside_field_rms;
          output.terminal_patch_outside_field_maximum =
              certificate.outside_field_maximum;
          output.terminal_patch_composite_field_rms =
              certificate.composite_field_rms;
          output.terminal_patch_composite_field_quantile =
              certificate.composite_field_quantile;
          output.terminal_patch_composite_field_maximum =
              certificate.composite_field_maximum;
          if (!certificate.accepted)
            output.vector_difference_maximum = std::max(
                output.vector_difference_maximum,
                certificate.maximum_vector_difference);
          output.terminal_patch_composite_field_bound =
              certificate.composite_field_bound;
          output.terminal_patch_observed_luminosity_order =
              certificate.observed_luminosity_order;
          output.terminal_patch_observed_outside_luminosity_order =
              certificate.observed_outside_luminosity_order;
          output.terminal_patch_outside_luminosity_error =
              certificate.outside_luminosity_error;
          output.terminal_patch_composite_luminosity_bound =
              certificate.composite_luminosity_bound;
          output.nested_patch_field_error_rms =
              r_to_r2.aggregate_rms;
          output.nested_patch_field_error_linf =
              r_to_r2.aggregate_quantile;
          output.nested_patch_lumi_error =
              certificate.composite_luminosity_bound;
          maximum_patch_field_error_ = std::max(
              maximum_patch_field_error_,
              certificate.composite_field_bound);
          maximum_examined_patch_lumi_oracle_error_ = std::max(
              maximum_examined_patch_lumi_oracle_error_,
              certificate.composite_luminosity_bound);
          ++patch_lumi_oracle_evaluations_;
          if (certificate.accepted) {
            if (mode2_unmet_error_encounters_ == 0U)
              throw std::logic_error(
                  "terminal patch recovered an unrecorded uniform miss");
            --mode2_unmet_error_encounters_;
            terminal_patch_accepted = true;
            met = true;
            output.error_target_met = true;
            output.terminal_patch_certification_accepted = true;
            output.terminal_patch_used = true;
            output.estimated_field_error =
                certificate.composite_field_bound;
            output.source_lumi_oracle_error =
                certificate.composite_luminosity_bound;
            output.fields1.swap(composite_r2.fields1);
            output.fields2.swap(composite_r2.fields2);
            output.fine_lumi_dot = composite_r2.luminosity;
            output.nested_patch_used = true;
            output.vector_difference_rms =
                certificate.composite_field_rms;
            output.vector_difference_quantile =
                certificate.composite_field_quantile;
            output.vector_difference_maximum =
                certificate.maximum_vector_difference;
            output.vector_accepted_targets =
                certificate.composite_accepted_targets;
            output.vector_above_target =
                certificate.composite_above_target;
            output.vector_above_target_exposure =
                certificate.composite_above_target_exposure;
            output.vector_catastrophe_guard_passed = true;
            output.result_geometry_hash =
                output.terminal_patch_geometry_hash_r2;
            hash_u64(output.source_sampling_hierarchy_hash,
                     output.terminal_patch_geometry_hash_r);
            hash_u64(output.source_sampling_hierarchy_hash,
                     output.terminal_patch_geometry_hash_r2);
            hash_u64(output.source_sampling_hierarchy_hash,
                     output.terminal_patch_source_hash_r);
            hash_u64(output.source_sampling_hierarchy_hash,
                     output.terminal_patch_source_hash_r2);
            ++mode3_terminal_patch_accepts_;
            ++mode3_terminal_patch_used_;
            ++mode3_patch_encounters_;
            ++accepted_patch_lumi_oracle_evaluations_;
            maximum_accepted_patch_lumi_oracle_error_ = std::max(
                maximum_accepted_patch_lumi_oracle_error_,
                certificate.composite_luminosity_bound);
            checked_add_counter(mode3_patch_coarse_cells_, patch_r.cells(),
                                "terminal Mode-3 accepted r-patch cell");
            checked_add_counter(mode3_patch_fine_cells_, patch_r2.cells(),
                                "terminal Mode-3 accepted r2-patch cell");
          }
        }
      }
    }

    if (!terminal_patch_accepted) {
      ++mode3_terminal_patch_rejects_;
      ++mode3_terminal_patch_uniform_fallbacks_;
      ++mode3_terminal_uniform_encounters_;
      output.terminal_patch_uniform_fallback = true;
      output.terminal_patch_external_replay_recommended = true;
      output.error_target_met = false;
      if (output.terminal_patch_rejection_mask == 0U)
        output.terminal_patch_rejection_mask |=
            adaptive_grid::TerminalPatchRejectionInvalidEvidence;
      mode3_terminal_patch_rejection_mask_or_ |=
          output.terminal_patch_rejection_mask;
      if ((output.terminal_patch_rejection_mask &
           adaptive_grid::TerminalPatchRejectionResource) != 0U)
        ++mode3_terminal_patch_resource_rejects_;
      if ((output.terminal_patch_rejection_mask &
           ~static_cast<std::uint32_t>(
               adaptive_grid::TerminalPatchRejectionResource)) != 0U)
        ++mode3_terminal_patch_nonresource_rejects_;
      // Preserve a coarse compatibility summary while the full terminal mask
      // remains the authoritative diagnostic.
      if (output.terminal_patch_rejection_mask &
          (adaptive_grid::TerminalPatchRejectionNonlocalized |
           adaptive_grid::TerminalPatchRejectionSourceSupport))
        output.nested_patch_rejection_mask |= NestedPatchRejectionTail;
      if (output.terminal_patch_rejection_mask &
          adaptive_grid::TerminalPatchRejectionSampling)
        output.nested_patch_rejection_mask |= NestedPatchRejectionSampling;
      if (output.terminal_patch_rejection_mask &
          (adaptive_grid::TerminalPatchRejectionHarmonicResidual |
           adaptive_grid::TerminalPatchRejectionBoundaryFlux))
        output.nested_patch_rejection_mask |= NestedPatchRejectionResidual;
      if (output.terminal_patch_rejection_mask &
          (adaptive_grid::TerminalPatchRejectionNoncontracting |
           adaptive_grid::TerminalPatchRejectionFieldBudget |
           adaptive_grid::TerminalPatchRejectionCatastrophe))
        output.nested_patch_rejection_mask |=
            NestedPatchRejectionFieldOracle;
      if (output.terminal_patch_rejection_mask &
          adaptive_grid::TerminalPatchRejectionLuminosityBudget)
        output.nested_patch_rejection_mask |=
            NestedPatchRejectionLumiOracle;
      if (configuration_.adaptive_fail_on_unmet_error) {
        std::ostringstream message;
        message
            << "terminal Mode-3 two-patch certificate rejected; "
               "recommended_action=joint_replay_and_mesh_refinement"
            << " rejection_mask="
            << output.terminal_patch_rejection_mask
            << " parent_factor=" << parent.geometry.refinement_factor
            << " patch_factor_r=" << output.terminal_patch_factor_r
            << " patch_factor_r2=" << output.terminal_patch_factor_r2
            << " outside_unresolved_exposure="
            << output.terminal_patch_outside_unresolved_exposure
            << " outside_field_contribution="
            << output.terminal_patch_outside_field_error_contribution
            << " outside_field_maximum="
            << output.terminal_patch_outside_field_maximum
            << " outside_luminosity_fraction="
            << output.terminal_patch_outside_luminosity_fraction
            << " source_tail_fraction="
            << output.nested_patch_tail_fraction
            << " straddling_sources="
            << output.nested_patch_straddling_sources
            << " composite_field_bound="
            << output.terminal_patch_composite_field_bound
            << " outside_observed_order="
            << output.terminal_patch_observed_outside_field_order
            << " composite_field_rms="
            << output.terminal_patch_composite_field_rms
            << " composite_field_quantile="
            << output.terminal_patch_composite_field_quantile
            << " composite_field_maximum="
            << output.terminal_patch_composite_field_maximum
            << " composite_luminosity_bound="
            << output.terminal_patch_composite_luminosity_bound
            << " boundary_flux_rms="
            << output.terminal_patch_boundary_flux_rms
            << " boundary_flux_quantile="
            << output.terminal_patch_boundary_flux_quantile
            << " boundary_flux_maximum="
            << output.terminal_patch_boundary_flux_maximum;
        throw std::runtime_error(message.str());
      }
    }
  }
}

// Apply the ordinary nested patch only while every local certificate holds.
void Workspace::apply_nested_resolution_patch(
    ResolutionControllerState& state,
    ResolutionSelection& selection,
    bool& mode3_promotion_required) {
  auto& source_hierarchy = state.source_hierarchy;
  auto& levels = state.levels;
  auto& accepted_targets1 = state.accepted_targets1;
  auto& accepted_targets2 = state.accepted_targets2;
  auto& target = state.target;
  auto& selected = selection.selected;
  auto& mode3_terminal_uniform = selection.mode3_terminal_uniform;
  auto& output = state.binding.output;
  auto& slice1 = state.binding.slice1;
  auto& slice2 = state.binding.slice2;
  auto& particles1 = state.binding.particles1;
  auto& particles2 = state.binding.particles2;
  auto& plan = state.binding.plan;
  auto& field_operator = state.binding.field_operator;
  if (configuration_.adaptive_grid_mode == 3 && !mode3_terminal_uniform) {
    LevelSolution& accepted = levels[selected];
    const std::size_t operator_transition_guard =
        adaptive_grid::field_operator_transition_guard_cells(
            field_operator);
    std::vector<double> rho1(accepted.context->rho1.begin(),
                             accepted.context->rho1.end());
    std::vector<double> rho2(accepted.context->rho2.begin(),
                             accepted.context->rho2.end());
    adaptive_grid::NestedPatchBox patch =
        adaptive_grid::select_nested_patch(
            rho1, rho2, accepted.geometry.n_x, accepted.geometry.n_y,
            configuration_.adaptive_coarsen_threshold,
            configuration_.adaptive_tau_curvature,
            configuration_.adaptive_rho_floor,
            configuration_.adaptive_indicator_smoothing_passes,
            configuration_.adaptive_error_buffer_cells,
            configuration_.adaptive_transition_cells,
            configuration_.adaptive_blocking_factor);
    output.nested_patch_requested = patch.valid;
    if (output.nested_patch_requested)
      ++mode3_patch_requests_;
    else
      ++mode3_no_patch_encounters_;
    const std::size_t block = configuration_.adaptive_blocking_factor;
    const std::size_t interior_begin = std::max(
        operator_transition_guard, block);
    const std::size_t interior_end_x =
        accepted.geometry.n_x > operator_transition_guard
            ? (accepted.geometry.n_x - operator_transition_guard) /
                  block * block
            : 0U;
    const std::size_t interior_end_y =
        accepted.geometry.n_y > operator_transition_guard
            ? (accepted.geometry.n_y - operator_transition_guard) /
                  block * block
            : 0U;
    double certified_tail_fraction = 0.0;
    std::size_t certified_straddlers = 0U;
    if (patch.valid) {
      patch.begin_x = std::max(patch.begin_x, interior_begin);
      patch.begin_y = std::max(patch.begin_y, interior_begin);
      patch.end_x = std::min(patch.end_x, interior_end_x);
      patch.end_y = std::min(patch.end_y, interior_end_y);
      patch.valid = patch.end_x > patch.begin_x + 3U &&
                    patch.end_y > patch.begin_y + 3U;
      if (!patch.valid)
        output.nested_patch_rejection_mask |=
            NestedPatchRejectionGeometry;
    }
    if (patch.valid) {
      // The connected indicator locates the dense overlap core.  Grow in
      // complete blocking units until the excluded full-charge tail is
      // bounded.  Ownership is exact at parent control-volume boundaries:
      // every accepted parent cell maps to a disjoint aligned r-by-r child
      // block, so a particle-centre/fine-CIC straddler test would describe a
      // different source and must not veto this coherent hierarchy.
      long double total_charge = 0.0L;
      for (std::size_t cell = 0U; cell < rho1.size(); ++cell)
        total_charge += std::fabs(rho1[cell]) + std::fabs(rho2[cell]);
      const auto outside_fraction = [&rho1, &rho2, &patch,
                                     &accepted, total_charge]() {
        if (!(total_charge > 0.0L)) return 0.0;
        long double outside = 0.0L;
        for (std::size_t ix = 0U; ix < accepted.geometry.n_x; ++ix)
          for (std::size_t iy = 0U; iy < accepted.geometry.n_y; ++iy)
            if (ix < patch.begin_x || ix >= patch.end_x ||
                iy < patch.begin_y || iy >= patch.end_y) {
              const std::size_t cell = ix * accepted.geometry.n_y + iy;
              outside += std::fabs(rho1[cell]) + std::fabs(rho2[cell]);
            }
        return static_cast<double>(outside / total_charge);
      };
      double tail_fraction = outside_fraction();
      while (tail_fraction > configuration_.adaptive_tau_tail) {
        const std::size_t old_begin_x = patch.begin_x;
        const std::size_t old_begin_y = patch.begin_y;
        const std::size_t old_end_x = patch.end_x;
        const std::size_t old_end_y = patch.end_y;
        const std::size_t lower_guard = checked_sum(
            interior_begin, block, "Mode-3 lower patch guard");
        patch.begin_x = patch.begin_x > lower_guard
            ? patch.begin_x - block : interior_begin;
        patch.begin_y = patch.begin_y > lower_guard
            ? patch.begin_y - block : interior_begin;
        patch.end_x = std::min(interior_end_x,
                               checked_sum(patch.end_x, block,
                                           "Mode-3 patch end x"));
        patch.end_y = std::min(interior_end_y,
                               checked_sum(patch.end_y, block,
                                           "Mode-3 patch end y"));
        if (patch.begin_x == old_begin_x &&
            patch.begin_y == old_begin_y &&
            patch.end_x == old_end_x && patch.end_y == old_end_y)
          break;
        tail_fraction = outside_fraction();
      }
      // Retain one complete coarse block between the certified source tail
      // and the transition layer whenever the containing level permits it.
      const std::size_t lower_guard = checked_sum(
          interior_begin, block, "Mode-3 safety lower guard");
      patch.begin_x = patch.begin_x > lower_guard
          ? patch.begin_x - block : interior_begin;
      patch.begin_y = patch.begin_y > lower_guard
          ? patch.begin_y - block : interior_begin;
      patch.end_x = std::min(
          interior_end_x,
          checked_sum(patch.end_x, block, "Mode-3 safety end x"));
      patch.end_y = std::min(
          interior_end_y,
          checked_sum(patch.end_y, block, "Mode-3 safety end y"));
      tail_fraction = outside_fraction();
      certified_tail_fraction = tail_fraction;
      certified_straddlers = 0U;
      if (tail_fraction > configuration_.adaptive_tau_tail) {
        output.nested_patch_rejection_mask |= NestedPatchRejectionTail;
        patch.valid = false;
      }
      const std::size_t minimum_span = checked_product(
          2U, checked_sum(configuration_.adaptive_transition_cells,
                          operator_transition_guard,
                          "Mode-3 minimum transition span"),
          "Mode-3 minimum patch span");
      if (patch.n_x() < minimum_span || patch.n_y() < minimum_span) {
        output.nested_patch_rejection_mask |=
            NestedPatchRejectionGeometry;
        patch.valid = false;
      }
    }
    output.nested_patch_tail_fraction = certified_tail_fraction;
    output.nested_patch_straddling_sources = certified_straddlers;
    if (output.nested_patch_requested && !patch.valid)
      mode3_promotion_required = true;
    if (patch.valid) {
      const adaptive_grid::FieldLevelGeometry coarse_patch =
          adaptive_grid::make_patch_geometry(accepted.geometry, patch);
      const adaptive_grid::FieldLevelGeometry fine_patch =
          adaptive_grid::refine_patch_geometry(
              coarse_patch, configuration_.adaptive_refinement_ratio);
      std::vector<unsigned char> source_mask1(particles1.size(), 0U);
      std::vector<unsigned char> source_mask2(particles2.size(), 0U);
      std::vector<unsigned char> target_mask1(particles1.size(), 0U);
      std::vector<unsigned char> target_mask2(particles2.size(), 0U);
      const auto make_source_mask = [](
          const std::vector<PARTICLE*>& particles,
          const std::vector<adaptive_grid::CanonicalDepositDescriptor>& desc,
          const adaptive_grid::FieldLevelGeometry& coarse,
          const adaptive_grid::FieldLevelGeometry& fine,
          std::vector<unsigned char>& mask) {
        for (std::size_t index = 0U; index < particles.size(); ++index) {
          if (!desc[index].accepted()) continue;
          float x = 0.0f;
          float y = 0.0f;
          particles[index]->XYposition(x, y);
          mask[index] =
              adaptive_grid::map_deposit_to_level(x, y, coarse).accepted &&
              adaptive_grid::map_deposit_to_level(x, y, fine).accepted;
        }
      };
      const auto make_target_mask = [field_operator](
          const std::vector<PARTICLE*>& particles,
          const std::vector<adaptive_grid::CanonicalGatherDescriptor>& desc,
          const adaptive_grid::FieldLevelGeometry& coarse,
          const adaptive_grid::FieldLevelGeometry& fine,
          std::vector<unsigned char>& mask) {
        for (std::size_t index = 0U; index < particles.size(); ++index) {
          if (!desc[index].accepted()) continue;
          float x = 0.0f;
          float y = 0.0f;
          particles[index]->XYposition(x, y);
          mask[index] =
              adaptive_grid::map_gather_to_level(
                  x, y, coarse, field_operator).accepted &&
              adaptive_grid::map_gather_to_level(
                  x, y, fine, field_operator).accepted;
        }
      };
      make_source_mask(particles1, plan.beam1_sources(), coarse_patch,
                       fine_patch, source_mask1);
      make_source_mask(particles2, plan.beam2_sources(), coarse_patch,
                       fine_patch, source_mask2);
      make_target_mask(particles1, plan.beam1_targets(), coarse_patch,
                       fine_patch, target_mask1);
      make_target_mask(particles2, plan.beam2_targets(), coarse_patch,
                       fine_patch, target_mask2);
      const std::size_t fine_sources = checked_sum(
          std::accumulate(source_mask1.begin(), source_mask1.end(),
                          std::size_t(0)),
          std::accumulate(source_mask2.begin(), source_mask2.end(),
                          std::size_t(0)),
          "Mode-3 fine source count");
      const std::size_t fine_targets = checked_sum(
          std::accumulate(target_mask1.begin(), target_mask1.end(),
                          std::size_t(0)),
          std::accumulate(target_mask2.begin(), target_mask2.end(),
                          std::size_t(0)),
          "Mode-3 fine target count");
      if (fine_sources != 0U && fine_targets != 0U) {
        if (selected + 1U >= levels.size() ||
            levels[selected + 1U].geometry.refinement_factor !=
                fine_patch.refinement_factor)
          throw std::runtime_error(
              "Mode-3 coherent patch source requires its declared "
              "uniform-fine hierarchy member");
        const LevelSolution fine = solve_field_level(
            slice1, slice2, plan, fine_patch,
            &source_mask1, &source_mask2, &target_mask1, &target_mask2,
            true, &source_hierarchy, &levels[selected + 1U]);
        output.source_restrict_ms += fine.source_restrict_ms;
        output.source_prolong_ms += fine.source_prolong_ms;
        output.source_maximum_charge_closure = std::max(
            output.source_maximum_charge_closure,
            fine.maximum_charge_closure);
        output.source_maximum_projection_moment_shift = std::max(
            output.source_maximum_projection_moment_shift,
            fine.maximum_first_moment_error);
        if (!resolution_sampling_ok(fine)) {
          output.noise_limited = true;
          output.nested_patch_rejection_mask |=
              NestedPatchRejectionSampling;
          mode3_promotion_required = true;
        }
        output.nested_patch_effective_particles_per_cell =
            fine.effective_particles_per_cell;
        output.nested_patch_minimum_effective_particles_per_cell =
            fine.minimum_effective_particles_per_cell;
        const std::size_t fine_cells = fine_patch.cells();
        output.nested_patch_coarse_cells = coarse_patch.cells();
        output.nested_patch_fine_cells = fine_cells;
        output.nested_patch_coarse_delta_x = coarse_patch.delta_x;
        output.nested_patch_coarse_delta_y = coarse_patch.delta_y;
        output.nested_patch_fine_delta_x = fine_patch.delta_x;
        output.nested_patch_fine_delta_y = fine_patch.delta_y;
        checked_add_counter(mode3_attempted_patch_coarse_cells_,
                            output.nested_patch_coarse_cells,
                            "Mode-3 attempted coarse-patch cell");
        checked_add_counter(mode3_attempted_patch_fine_cells_,
                            output.nested_patch_fine_cells,
                            "Mode-3 attempted fine-patch cell");
        // A failed sampling gate already has a safe uniform-fine promotion.
        // Do not run patch-only prolongation/harmonic/oracle work that cannot
        // change that decision and could itself fail before promotion.
        if (!mode3_promotion_required) {
        long double parent_outside_lumi_dot = 0.0L;
        for (std::size_t ix = 0U; ix < accepted.geometry.n_x; ++ix)
          for (std::size_t iy = 0U; iy < accepted.geometry.n_y; ++iy)
            if (ix < patch.begin_x || ix >= patch.end_x ||
                iy < patch.begin_y || iy >= patch.end_y) {
              const std::size_t cell = ix * accepted.geometry.n_y + iy;
              parent_outside_lumi_dot +=
                  static_cast<long double>(accepted.context->rho1[cell]) *
                  static_cast<long double>(accepted.context->rho2[cell]);
            }
        std::vector<double> coarse_phi1(fine_cells, 0.0);
        std::vector<double> coarse_phi2(fine_cells, 0.0);
        std::vector<double> correction1(fine_cells, 0.0);
        std::vector<double> correction2(fine_cells, 0.0);
        const PHI_FLOAT* fine_phi1 = fine.context->field.get_phi(1);
        const PHI_FLOAT* fine_phi2 = fine.context->field.get_phi(2);
        const PHI_FLOAT* accepted_phi1 = accepted.context->field.get_phi(1);
        const PHI_FLOAT* accepted_phi2 = accepted.context->field.get_phi(2);
        for (std::size_t ix = 0U; ix < fine_patch.n_x; ++ix)
          for (std::size_t iy = 0U; iy < fine_patch.n_y; ++iy) {
            const std::size_t cell = ix * fine_patch.n_y + iy;
            const float x = static_cast<float>(
                (static_cast<double>(fine_patch.origin_x) + ix + 0.5 -
                 fine_patch.offset_x) * fine_patch.delta_x);
            const float y = static_cast<float>(
                (static_cast<double>(fine_patch.origin_y) + iy + 0.5 -
                 fine_patch.offset_y) * fine_patch.delta_y);
            coarse_phi1[cell] =
                adaptive_grid::interpolate_level_potential(
                    accepted_phi1, accepted.geometry, x, y);
            coarse_phi2[cell] =
                adaptive_grid::interpolate_level_potential(
                    accepted_phi2, accepted.geometry, x, y);
            if (ix == 0U || iy == 0U || ix + 1U == fine_patch.n_x ||
                iy + 1U == fine_patch.n_y) {
              correction1[cell] = coarse_phi1[cell] - fine_phi1[cell];
              correction2[cell] = coarse_phi2[cell] - fine_phi2[cell];
            }
          }
        const std::size_t maximum_iterations = std::max<std::size_t>(
            2000U, std::min<std::size_t>(
                1000000U,
                checked_product(fine_cells, 4U,
                                "Mode-3 harmonic iteration cap")));
        const std::chrono::steady_clock::time_point harmonic_start =
            std::chrono::steady_clock::now();
        const adaptive_grid::HarmonicSolveReport report1 =
            adaptive_grid::solve_harmonic_dirichlet(
                correction1, fine_patch.n_x, fine_patch.n_y,
                fine_patch.delta_x, fine_patch.delta_y,
                configuration_.adaptive_fine_residual_tolerance,
                maximum_iterations);
        const adaptive_grid::HarmonicSolveReport report2 =
            adaptive_grid::solve_harmonic_dirichlet(
                correction2, fine_patch.n_x, fine_patch.n_y,
                fine_patch.delta_x, fine_patch.delta_y,
                configuration_.adaptive_fine_residual_tolerance,
                maximum_iterations);
        mode3_harmonic_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - harmonic_start).count();
        if (!report1.converged || !report2.converged) {
          output.nested_patch_rejection_mask |=
              NestedPatchRejectionResidual;
          mode3_promotion_required = true;
        }
        output.nested_patch_residual = std::max(
            report1.normalized_residual, report2.normalized_residual);
        maximum_patch_residual_ = std::max(
            maximum_patch_residual_, output.nested_patch_residual);
        output.nested_patch_iterations = checked_sum(
            report1.iterations, report2.iterations,
            "Mode-3 harmonic transform/iteration count");
        checked_add_counter(mode3_harmonic_iterations_,
                            output.nested_patch_iterations,
                            "Mode-3 harmonic iteration");
        // Likewise, a failed Dirichlet residual is terminal for this patch.
        // Preserve its diagnostics and go directly to uniform promotion.
        if (!mode3_promotion_required) {
        std::vector<double> composite1(fine_cells, 0.0);
        std::vector<double> composite2(fine_cells, 0.0);
        const std::size_t fine_transition_cells = checked_product(
            configuration_.adaptive_transition_cells,
            configuration_.adaptive_refinement_ratio,
            "Mode-3 fine transition width");
        for (std::size_t ix = 0U; ix < fine_patch.n_x; ++ix)
          for (std::size_t iy = 0U; iy < fine_patch.n_y; ++iy) {
            const std::size_t cell = ix * fine_patch.n_y + iy;
            const float x = static_cast<float>(
                (static_cast<double>(fine_patch.origin_x) + ix + 0.5 -
                 fine_patch.offset_x) * fine_patch.delta_x);
            const float y = static_cast<float>(
                (static_cast<double>(fine_patch.origin_y) + iy + 0.5 -
                 fine_patch.offset_y) * fine_patch.delta_y);
            const double blend = adaptive_grid::patch_transition_weight(
                x, y, fine_patch,
                fine_transition_cells,
                operator_transition_guard);
            const double nested1 = fine_phi1[cell] + correction1[cell];
            const double nested2 = fine_phi2[cell] + correction2[cell];
            composite1[cell] = coarse_phi1[cell] +
                               blend * (nested1 - coarse_phi1[cell]);
            composite2[cell] = coarse_phi2[cell] +
                               blend * (nested2 - coarse_phi2[cell]);
          }
        const auto replace_gather = [&fine_patch, field_operator](
            const std::vector<PARTICLE*>& particles,
            const std::vector<unsigned char>& mask,
            const std::vector<double>& phi,
            std::vector<float>& fields) {
          for (std::size_t index = 0U; index < particles.size(); ++index) {
            if (!mask[index]) continue;
            float x = 0.0f;
            float y = 0.0f;
            particles[index]->XYposition(x, y);
            const adaptive_grid::LevelGatherDescriptor descriptor =
                adaptive_grid::map_gather_to_level(
                    x, y, fine_patch, field_operator);
            if (!descriptor.accepted)
              throw std::runtime_error(
                  "Mode-3 composite gather escaped its fine patch");
            const std::array<float, 2> field =
                adaptive_grid::gather_level_field(
                    phi.data(), fine_patch, descriptor, field_operator);
            fields[2U * index] = field[0];
            fields[2U * index + 1U] = field[1];
          }
        };
        replace_gather(particles1, target_mask1, composite2, output.fields1);
        replace_gather(particles2, target_mask2, composite1, output.fields2);
        if (selected + 1U >= levels.size() ||
            levels[selected + 1U].geometry.refinement_factor !=
                fine_patch.refinement_factor)
          throw std::runtime_error(
              "Mode-3 requires one declared uniform audit class at its "
              "nested-patch spacing");
        const adaptive_grid::RichardsonEstimate patch_difference =
            adaptive_grid::estimate_same_particle_field_error(
                output.fields1, output.fields2,
                levels[selected + 1U].fields1,
                levels[selected + 1U].fields2,
                configuration_.adaptive_refinement_ratio, 2.0,
                configuration_.adaptive_field_floor,
                &accepted_targets1, &accepted_targets2);
        output.nested_patch_field_error_rms =
            patch_difference.observed_difference_rms;
        output.nested_patch_field_error_linf =
            patch_difference.observed_difference_linf;
        const double patch_field_error = std::max(
            output.nested_patch_field_error_rms,
            output.nested_patch_field_error_linf);
        if (!patch_difference.reliable || patch_field_error > target) {
          output.nested_patch_rejection_mask |=
              NestedPatchRejectionFieldOracle;
          mode3_promotion_required = true;
        }
        maximum_patch_field_error_ = std::max(
            maximum_patch_field_error_, patch_field_error);
        if (!mode3_promotion_required) {
          const double parent_factor = static_cast<double>(
              accepted.geometry.refinement_factor);
          const double ratio = static_cast<double>(
              configuration_.adaptive_refinement_ratio);
          const long double composite_lumi_dot_long =
              parent_outside_lumi_dot +
              static_cast<long double>(ratio * ratio) *
              static_cast<long double>(fine.raw_lumi_dot);
          if (!finite_value(composite_lumi_dot_long) ||
              composite_lumi_dot_long < 0.0L ||
              composite_lumi_dot_long >
                  static_cast<long double>(
                      std::numeric_limits<double>::max()))
            throw std::runtime_error(
                "Mode-3 composite luminosity is non-finite or negative");
          const double composite_lumi_dot =
              static_cast<double>(composite_lumi_dot_long);
          const double normalized_composite_lumi_dot =
              std::max(0.0, composite_lumi_dot) *
              parent_factor * parent_factor;
          if (!finite_value(normalized_composite_lumi_dot))
            throw std::runtime_error(
                "Mode-3 composite luminosity normalization overflowed");
          if (source_hierarchy.model == "reference_cic_v1" ||
              source_hierarchy.model == "fixed_width_bspline1_v1") {
            const LevelSolution& uniform_fine = levels[selected + 1U];
            const double uniform_factor = static_cast<double>(
                uniform_fine.geometry.refinement_factor);
            const double normalized_uniform_lumi =
                std::max(0.0, uniform_fine.raw_lumi_dot) *
                uniform_factor * uniform_factor;
            const double denominator = std::max(
                std::fabs(normalized_uniform_lumi),
                configuration_.adaptive_lumi_floor);
            output.nested_patch_lumi_error = std::fabs(
                normalized_composite_lumi_dot - normalized_uniform_lumi) /
                denominator;
            if (!finite_value(output.nested_patch_lumi_error))
              throw std::runtime_error(
                  "Mode-3 patch luminosity oracle is non-finite");
            ++patch_lumi_oracle_evaluations_;
            maximum_examined_patch_lumi_oracle_error_ = std::max(
                maximum_examined_patch_lumi_oracle_error_,
                output.nested_patch_lumi_error);
            if (output.nested_patch_lumi_error >
                configuration_.adaptive_lumi_error_target) {
              output.nested_patch_rejection_mask |=
                  NestedPatchRejectionLumiOracle;
              mode3_promotion_required = true;
            }
          }
          if (!mode3_promotion_required) {
            if (source_hierarchy.model == "reference_cic_v1" ||
                source_hierarchy.model == "fixed_width_bspline1_v1") {
              ++accepted_patch_lumi_oracle_evaluations_;
              maximum_accepted_patch_lumi_oracle_error_ = std::max(
                  maximum_accepted_patch_lumi_oracle_error_,
                  output.nested_patch_lumi_error);
            }
            output.fine_lumi_dot = normalized_composite_lumi_dot;
            output.nested_patch_used = true;
            output.result_geometry_hash = adaptive_result_geometry_hash(
                configuration_.adaptive_grid_mode, accepted.geometry,
                &fine_patch, field_operator);
            ++mode3_patch_encounters_;
            checked_add_counter(mode3_patch_coarse_cells_,
                                output.nested_patch_coarse_cells,
                                "Mode-3 coarse-patch cell");
            checked_add_counter(mode3_patch_fine_cells_,
                                output.nested_patch_fine_cells,
                                "Mode-3 fine-patch cell");
          }
        }
        }
        }
      } else {
        output.nested_patch_rejection_mask |=
            NestedPatchRejectionEmptySupport;
        mode3_promotion_required = true;
      }
    }
  }
}

// Publish the already certified uniform fallback when a patch is rejected.
void Workspace::promote_resolution_uniform(
    ResolutionControllerState& state,
    ResolutionSelection& selection,
    bool mode3_promotion_required) {
  auto& binding = state.binding;
  auto& levels = state.levels;
  auto& pair_estimates = state.pair_estimates;
  auto& pair_estimate_ready = state.pair_estimate_ready;
  auto& asymptotic_pairs = state.asymptotic_pairs;
  auto& target = state.target;
  auto& selected = selection.selected;
  auto& accepted_estimate = selection.accepted_estimate;
  auto& output = state.binding.output;
  if (mode3_promotion_required) {
    if (output.nested_patch_rejection_mask == NestedPatchRejectionNone)
      throw std::logic_error(
          "Mode-3 uniform promotion has no recorded rejection reason");
    if (selected + 1U >= levels.size())
      throw std::runtime_error(
          "Mode-3 cannot promote an uncertified patch without its declared "
          "uniform-fine oracle");
    const std::size_t parent = selected;
    const std::size_t promoted = selected + 1U;
    if (!pair_estimate_ready[parent] || !asymptotic_pairs[parent] ||
        !resolution_sampling_ok(levels[promoted]) ||
        !resolution_indicators_ok(levels[promoted]) ||
        !resolution_luminosity_ok(state, promoted) ||
        pair_estimates[parent].fine_error > target ||
        levels[promoted].geometry.refinement_factor !=
            checked_product(levels[parent].geometry.refinement_factor,
                            configuration_.adaptive_refinement_ratio,
                            "Mode-3 promotion factor"))
      throw std::runtime_error(
          "Mode-3 requested patch failed and its uniform-fine promotion "
          "did not satisfy the declared field-error contract");
    selected = promoted;
    accepted_estimate = pair_estimates[parent];
    output.estimated_field_error = accepted_estimate.fine_error;
    output.source_lumi_oracle_error =
        resolution_luminosity_error(state, promoted);
    output.field_difference_rms = accepted_estimate.observed_difference_rms;
    output.field_difference_linf = accepted_estimate.observed_difference_linf;
    output.nested_patch_promoted_to_uniform = true;
    publish_uniform_resolution(binding,levels,selected);
    ++mode3_uniform_promotions_;
  }
}

void Workspace::prepare_resolution_adaptive(
    int slice1, int slice2, FieldPrefetch& output) {
  if (!output.window_plan)
    output.window_plan = make_window_plan(slice1, slice2);
  const adaptive_grid::UniformWindowPlan& plan = *output.window_plan;
  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  const std::vector<PRIMARY_PARTICLE_ID>& target_ids1 =
      beam1_->particle_beam().getParticleIdVectorShapeChecked(slice1);
  const std::vector<PRIMARY_PARTICLE_ID>& target_ids2 =
      beam2_->particle_beam().getParticleIdVectorShapeChecked(slice2);
  const adaptive_grid::FieldOperator field_operator =
      adaptive_grid::parse_field_operator(
          configuration_.adaptive_field_operator);
  ResolutionEncounterBinding binding{
      slice1, slice2, plan, particles1, particles2, field_operator, output};
  reset_resolution_output(binding);
  if (plan.empty()) {
    complete_empty_resolution(binding);
    return;
  }

  double target = std::min(
      configuration_.adaptive_error_target,
      configuration_.adaptive_tau_richardson > 0.0
          ? configuration_.adaptive_tau_richardson
          : configuration_.adaptive_error_target);
  if (configuration_.adaptive_source_model == "reference_cic_v1" ||
      configuration_.adaptive_source_model == "fixed_width_bspline1_v1")
    target = std::min(target, configuration_.adaptive_solver_error_target);
  const bool mixed_vector_error_policy =
      configuration_.adaptive_error_norm_policy ==
          "vector_mixed_rms_quantile_v2";
  const bool vector_error_policy =
      configuration_.adaptive_error_norm_policy ==
          "vector_rms_quantile_v1" || mixed_vector_error_policy;
  std::vector<unsigned char> accepted_targets1(particles1.size(), 0U);
  std::vector<unsigned char> accepted_targets2(particles2.size(), 0U);
  for (std::size_t index = 0U; index < accepted_targets1.size(); ++index)
    accepted_targets1[index] = plan.beam1_targets()[index].accepted() ? 1U : 0U;
  for (std::size_t index = 0U; index < accepted_targets2.size(); ++index)
    accepted_targets2[index] = plan.beam2_targets()[index].accepted() ? 1U : 0U;

  // The card list is a bootstrap triplet, not a scientific resolution
  // ceiling.  Extend it geometrically for this encounter until an explicit
  // optional factor cap or a real allocation/indexing capacity binds.  No
  // field context is allocated here; the existing lazy solve path remains
  // authoritative.

  std::vector<std::size_t> spacing_factors = plan_resolution_spacing(binding);
  reconcile_resolution_contexts(binding, spacing_factors);

  CoherentSourceHierarchy source_hierarchy =
      build_resolution_source(binding, spacing_factors);

  std::vector<LevelSolution> levels;
  levels.reserve(spacing_factors.size());

  const std::size_t pair_count = spacing_factors.size() - 1U;
  std::vector<adaptive_grid::RichardsonEstimate> raw_pairs(pair_count);
  std::vector<adaptive_grid::SymmetricVectorFieldSensitivity>
      raw_pair_sensitivities(pair_count);
  std::vector<adaptive_grid::VectorFieldErrorStatistics>
      raw_pair_vector_statistics(pair_count);
  std::vector<adaptive_grid::VectorFieldRichardsonEstimate>
      vector_pair_estimates(pair_count);
  std::vector<adaptive_grid::RichardsonEstimate> pair_estimates(pair_count);
  std::vector<double> vector_coarse_errors(pair_count, 0.0);
  std::vector<double> vector_fine_errors(pair_count, 0.0);
  std::vector<unsigned char> raw_pair_ready(pair_count, 0U);
  std::vector<unsigned char> pair_estimate_ready(pair_count, 0U);
  std::vector<double> observed_orders(pair_count, 0.0);
  std::vector<unsigned char> asymptotic_pairs(pair_count, 0U);
  std::vector<std::size_t> triplet_first_pairs(pair_count, 0U);
  std::vector<std::size_t> triplet_second_pairs(pair_count, 0U);
  std::vector<unsigned char> triplet_valid(pair_count, 0U);

  // A pair can certify a level only when a third level establishes that the
  // same-particle absolute differences contract.  Interior candidates use
  // the following pair; the terminal pair uses its predecessor.  Levels are
  // solved only as this rolling three-level audit demands them.

  std::size_t source_level_index = 0U;
  if (source_hierarchy.model == "reference_cic_v1") {
    const std::vector<std::size_t>::const_iterator source_position =
        std::find(spacing_factors.begin(), spacing_factors.end(),
                  source_hierarchy.source_factor);
    if (source_position == spacing_factors.end())
      throw std::logic_error(
          "reference CIC source identity level disappeared after preflight");
    source_level_index = static_cast<std::size_t>(
        source_position - spacing_factors.begin());
  }
  bool noise_aware_stopped = false;
  std::size_t terminal_pair = pair_count - 1U;
  std::vector<double> source_lumi_oracle_errors(
      spacing_factors.size(), 0.0);
  std::vector<unsigned char> source_lumi_oracle_ready(
      spacing_factors.size(), 0U);

  ResolutionControllerState state{
      binding,
      spacing_factors,
      source_hierarchy,
      levels,
      raw_pairs,
      raw_pair_sensitivities,
      raw_pair_vector_statistics,
      vector_pair_estimates,
      pair_estimates,
      vector_coarse_errors,
      vector_fine_errors,
      raw_pair_ready,
      pair_estimate_ready,
      observed_orders,
      asymptotic_pairs,
      triplet_first_pairs,
      triplet_second_pairs,
      triplet_valid,
      accepted_targets1,
      accepted_targets2,
      target_ids1,
      target_ids2,
      target,
      vector_error_policy,
      mixed_vector_error_policy,
      source_level_index,
      noise_aware_stopped,
      source_lumi_oracle_errors,
      source_lumi_oracle_ready
  };

  std::size_t selected = spacing_factors.size() - 1U;
  bool met = false;
  bool finest_asymptotic = false;
  bool mode3_terminal_uniform = false;
  std::size_t noncontracting_triplet_streak = 0U;
  adaptive_grid::RichardsonEstimate accepted_estimate;
  ResolutionSelection selection{
      selected, met, finest_asymptotic, mode3_terminal_uniform, noncontracting_triplet_streak, terminal_pair, accepted_estimate
  };
  select_resolution_coarse(state, selection);

  // The finest class has no next-finer direct comparison.  It may use the
  // preceding fine extrapolation only after every active component shows a
  // decreasing three-level difference sequence.
  const bool terminal_patch_requested_by_card =
      configuration_.adaptive_grid_mode == 3 &&
      configuration_.adaptive_terminal_patch_certification &&
      source_hierarchy.model == "fixed_width_bspline1_v1";
  select_resolution_terminal(state, selection);

  handle_resolution_unmet_error(state, selection, terminal_patch_requested_by_card);

  output.field_difference_rms = accepted_estimate.observed_difference_rms;
  output.field_difference_linf = accepted_estimate.observed_difference_linf;
  publish_uniform_resolution(binding,levels,selected);

  const std::chrono::steady_clock::time_point mode3_controller_start =
      std::chrono::steady_clock::now();
  bool mode3_promotion_required = false;
  bool terminal_patch_accepted = false;
  certify_terminal_resolution_patch(state, selection, terminal_patch_requested_by_card, terminal_patch_accepted);

  if (configuration_.adaptive_grid_mode == 3 && mode3_terminal_uniform &&
      (!terminal_patch_requested_by_card || output.error_target_met))
    ++mode3_terminal_uniform_encounters_;
  apply_nested_resolution_patch(state, selection, mode3_promotion_required);

  promote_resolution_uniform(state, selection, mode3_promotion_required);

  mode3_rejection_mask_or_ |= output.nested_patch_rejection_mask;
  if (configuration_.adaptive_grid_mode == 3)
    mode3_controller_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mode3_controller_start).count();

  record_resolution_result(binding,source_hierarchy,selected,spacing_factors.size(),vector_error_policy);
}

void Workspace::prepare(int slice1, int slice2, FieldPrefetch& output) {
  if (!initialized_)
    throw std::logic_error(
        "wavefront field workspace used before initialization");
  if (slice1 < 0 || slice1 >= beam1_->number_of_slices() ||
      slice2 < 0 || slice2 >= beam2_->number_of_slices())
    throw std::out_of_range(
        "wavefront field workspace received an invalid slice index");

  if (configuration_.adaptive_grid_mode == 0) {
    prepare_fixed(slice1, slice2, output);
    return;
  }
  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  if (!output.window_plan) {
    output.window_plan = make_window_plan(slice1, slice2);
    bind_field_result(output, slice1, slice2, particles1, particles2);
  }
  const adaptive_grid::UniformWindowPlan& bound_plan = *output.window_plan;
  if (!canonical_lattice_ ||
      bound_plan.lattice_hash() != canonical_lattice_->hash())
    throw std::runtime_error(
        "adaptive plan rejected: canonical lattice hash mismatch");
  if (configuration_.adaptive_grid_mode >= 2) {
    if (!bound_plan.descriptor_payload_retained())
      throw std::runtime_error(
          "Mode 2/3 execution requires a complete descriptor payload");
    if (bound_plan.beam1_sources().size() != particles1.size() ||
        bound_plan.beam2_sources().size() != particles2.size() ||
        bound_plan.beam1_targets().size() != particles1.size() ||
        bound_plan.beam2_targets().size() != particles2.size())
      throw std::runtime_error(
          "Mode 2/3 plan rejected: particle descriptor count mismatch");
  }
  validate_field_result_binding(
      output, slice1, slice2, particles1, particles2);
  if (configuration_.adaptive_grid_mode >= 2) {
    prepare_resolution_adaptive(slice1, slice2, output);
    return;
  }
  const adaptive_grid::UniformWindowPlan& plan = *output.window_plan;
  if (!plan.full_window()) {
    prepare_adaptive(slice1, slice2, output);
    return;
  }

  // A canonical-sized Mode-1 window is an explicit topology class, not a
  // backend fallback.  Reuse the already-initialized canonical FFT context so
  // this identity case has neither a second full-size plan nor duplicate field
  // storage.  Its immutable plan and result binding remain fully adaptive.
  prepare_fixed(slice1, slice2, output);
  if (output.beam1_in + output.beam2_in != plan.accepted_source_count())
    throw std::runtime_error(
        "full-window adaptive deposition disagrees with immutable plan");
  const std::size_t cells = checked_product(
      plan.local_n_x(), plan.local_n_y(),
      "full-window adaptive cell count");
  ++adaptive_encounters_;
  ++adaptive_full_window_encounters_;
  ++adaptive_context_hits_;
  checked_add_counter(adaptive_required_cells_,
                      plan.required_node_count(),
                      "adaptive required-cell visit");
  checked_add_counter(adaptive_local_cells_, cells,
                      "adaptive local-cell visit");
  checked_add_counter(adaptive_padded_cells_,
                      checked_product(plan.padded_n_x(), plan.padded_n_y(),
                                      "adaptive padded-cell visit"),
                      "adaptive padded-cell visit");
  output.lattice_hash = plan.lattice_hash();
  output.bound_slice1 = slice1;
  output.bound_slice2 = slice2;
  output.bound_particle_count1 = particles1.size();
  output.bound_particle_count2 = particles2.size();
  output.result_binding_hash = field_binding_hash(
      slice1, slice2, particles1, particles2,
      output.lattice_hash, plan.plan_hash(), output.result_geometry_hash,
      output.source_model_tag, output.source_reference_factor,
      output.source_density_hash, output.source_sampling_hash,
      output.source_sampling_hierarchy_hash);
}

void Workspace::prepare_fixed(
    int slice1, int slice2, FieldPrefetch& output) {
  if (!initialized_)
    throw std::logic_error(
        "wavefront field workspace used before initialization");
  if (slice1 < 0 || slice1 >= beam1_->number_of_slices() ||
      slice2 < 0 || slice2 >= beam2_->number_of_slices())
    throw std::out_of_range(
        "wavefront field workspace received an invalid slice index");
  output.result_geometry_hash = 0U;

  const std::size_t cells =
      static_cast<std::size_t>(configuration_.n_x) *
      static_cast<std::size_t>(configuration_.n_y);
  PHI_FLOAT* rho1 = grid_.get_rho1_mutable();
  PHI_FLOAT* rho2 = grid_.get_rho2_mutable();
  clear_rho(rho1, active_cells1_, active_marks1_, cells);
  clear_rho(rho2, active_cells2_, active_marks2_, cells);

  const std::vector<PARTICLE*>& particles1 =
      beam1_->getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_->getParticleVector(slice2);
  deposit_slice(particles1, configuration_.macro_charge1, rho1,
                active_cells1_, active_marks1_,
                output.beam1_in, output.beam1_out);
  deposit_slice(particles2, configuration_.macro_charge2, rho2,
                active_cells2_, active_marks2_,
                output.beam2_in, output.beam2_out);

  // GRID::step_lumi historically accumulates this in float and only widens at
  // the external-field boundary.  Retain that exact summation contract.
  float fine_lumi_dot = 0.0f;
  checked_add_counter(dense_dot_visits_, cells, "dense-dot visit");
  if (!configuration_.sparse_cells) {
    for (std::size_t cell = 0; cell < cells; ++cell)
      fine_lumi_dot += rho1[cell] * rho2[cell];
    checked_add_counter(actual_dot_visits_, cells, "actual-dot visit");
  } else {
    // Sorting does not alter charge arithmetic.  Intersecting in increasing
    // linear-cell order omits only products that are exactly zero, retaining
    // the historical nonzero floating accumulation order.
    std::sort(active_cells1_.begin(), active_cells1_.end());
    std::sort(active_cells2_.begin(), active_cells2_.end());
    std::size_t first = 0;
    std::size_t second = 0;
    std::size_t actual_dot_visits = 0;
    while (first < active_cells1_.size() &&
           second < active_cells2_.size()) {
      if (active_cells1_[first] < active_cells2_[second]) {
        ++first;
      } else if (active_cells2_[second] < active_cells1_[first]) {
        ++second;
      } else {
        const std::size_t cell = active_cells1_[first];
        fine_lumi_dot += rho1[cell] * rho2[cell];
        ++actual_dot_visits;
        ++first;
        ++second;
      }
    }
    checked_add_counter(actual_dot_visits_, actual_dot_visits,
                        "actual-dot visit");
  }
  output.fine_lumi_dot = static_cast<double>(fine_lumi_dot);

  bool use_direct = false;
  std::uint64_t encounter_interactions = 0;
  if (configuration_.field_solver != sparse_direct_fields::Mode::Fft) {
    collect_target_nodes(particles1, target_nodes1_, target_marks1_);
    collect_target_nodes(particles2, target_nodes2_, target_marks2_);
    const std::uint64_t interactions_for_beam1 =
        sparse_direct_fields::interaction_count(
            active_cells2_.size(), target_nodes1_.size());
    const std::uint64_t interactions_for_beam2 =
        sparse_direct_fields::interaction_count(
            active_cells1_.size(), target_nodes2_.size());
    if (interactions_for_beam1 >
        std::numeric_limits<std::uint64_t>::max() - interactions_for_beam2)
      throw std::overflow_error(
          "sparse-direct encounter interaction count overflow");
    encounter_interactions = interactions_for_beam1 + interactions_for_beam2;
    use_direct = sparse_direct_fields::choose_direct(
        configuration_.field_solver, encounter_interactions,
        configuration_.sparse_direct_interaction_limit);
    checked_add_counter(direct_active_sources_, active_cells1_.size(),
                        "sparse-direct active-source");
    checked_add_counter(direct_active_sources_, active_cells2_.size(),
                        "sparse-direct active-source");
    checked_add_counter(direct_target_nodes_, target_nodes1_.size(),
                        "sparse-direct target-node");
    checked_add_counter(direct_target_nodes_, target_nodes2_.size(),
                        "sparse-direct target-node");
    if (estimated_interactions_ >
        std::numeric_limits<std::uint64_t>::max() - encounter_interactions)
      throw std::overflow_error(
          "sparse-direct cumulative interaction count overflow");
    estimated_interactions_ += encounter_interactions;
    maximum_estimated_interactions_ = std::max(
        maximum_estimated_interactions_, encounter_interactions);
  }

  if (use_direct) {
    const std::chrono::steady_clock::time_point field_start =
        std::chrono::steady_clock::now();
    const PHI_FLOAT* kernel = grid_.get_real_space_field_kernel();
    if (!kernel)
      throw std::logic_error(
          "sparse-direct field solver has no case-2 real-space kernel");
    // FIELD::fold_fft historically treats |charge_sign| < 1e-5 as an exact
    // zero-field request.  Preserve that gate even though the retained kernel
    // itself contains the small nonzero factor.
    const bool zero_field =
        std::abs(configuration_.charge_sign) < 1.0e-5f;
    if (zero_field) {
      PHI_FLOAT* phi2 = grid_.get_phi2_mutable();
      PHI_FLOAT* phi1 = grid_.get_phi1_mutable();
      for (std::size_t position = 0; position < target_nodes1_.size(); ++position)
        phi2[target_nodes1_[position]] = PHI_FLOAT(0.0);
      for (std::size_t position = 0; position < target_nodes2_.size(); ++position)
        phi1[target_nodes2_[position]] = PHI_FLOAT(0.0);
    } else {
      sparse_direct_fields::solve_potential_nodes(
          rho2, active_cells2_, kernel, target_nodes1_,
          configuration_.n_x, configuration_.n_y,
          grid_.get_phi2_mutable());
      sparse_direct_fields::solve_potential_nodes(
          rho1, active_cells1_, kernel, target_nodes2_,
          configuration_.n_x, configuration_.n_y,
          grid_.get_phi1_mutable());
    }
    direct_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - field_start).count();
    ++direct_encounters_;
    const std::uint64_t executed_interactions =
        zero_field ? 0U : encounter_interactions;
    if (direct_interactions_ >
        std::numeric_limits<std::uint64_t>::max() - executed_interactions)
      throw std::overflow_error(
          "sparse-direct executed interaction count overflow");
    direct_interactions_ += executed_interactions;
  } else {
    const bool time_fft =
        configuration_.field_solver != sparse_direct_fields::Mode::Fft;
    const std::chrono::steady_clock::time_point field_start =
        std::chrono::steady_clock::now();
    grid_.computeFields(2, configuration_.charge_sign, NULL);
    if (time_fft)
      fft_ms_ += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - field_start).count();
    ++fft_encounters_;
  }
  output.fields1.resize(2U * particles1.size());
  output.fields2.resize(2U * particles2.size());
  grid_.gatherPrimaryFieldsForSlices(
      slice1, slice2,
      output.fields1.empty() ? NULL : &output.fields1[0], particles1.size(),
      output.fields2.empty() ? NULL : &output.fields2[0], particles2.size());
}

}  // namespace wavefront_fields
}  // namespace feebl
