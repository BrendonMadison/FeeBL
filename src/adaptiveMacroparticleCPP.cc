#include "adaptiveMacroparticleCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

constexpr std::uint64_t kFnvOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

void hash_u64(std::uint64_t& state, std::uint64_t value) {
  for (unsigned int byte = 0U; byte < 8U; ++byte) {
    state ^= (value >> (8U * byte)) & UINT64_C(0xff);
    state *= kFnvPrime;
  }
}

void hash_double(std::uint64_t& state, double value) {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(state, bits);
}

bool finite_positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

std::size_t checked_cells(
    const feebl::adaptive_grid::FieldLevelGeometry& geometry) {
  if (!geometry.valid())
    throw std::invalid_argument(
        "adaptive source requires a valid field geometry");
  if (geometry.n_x > std::numeric_limits<std::size_t>::max() / geometry.n_y)
    throw std::overflow_error("adaptive source cell count overflows size_t");
  return geometry.n_x * geometry.n_y;
}

feebl::adaptive_particles::RepresentativeId child_id(
    std::uint64_t beam_label, PRIMARY_PARTICLE_ID root_id,
    std::uint64_t epoch, std::uint64_t path) {
  std::uint64_t high = kFnvOffset;
  hash_u64(high, UINT64_C(0x464545424c414d50));  // "FEEBLAMP"
  hash_u64(high, feebl::adaptive_particles::kSourceRepresentativeSchema);
  hash_u64(high, beam_label);
  hash_u64(high, root_id);
  std::uint64_t low = kFnvOffset;
  hash_u64(low, UINT64_C(0x50524f4a45435431));  // "PROJECT1"
  hash_u64(low, epoch);
  hash_u64(low, path);
  return feebl::adaptive_particles::RepresentativeId{high, low};
}

struct MergeKey {
  std::uint64_t cohort = 0U;
  std::size_t x = 0U;
  std::size_t y = 0U;

  bool operator<(const MergeKey& other) const noexcept {
    return std::tie(cohort, x, y) <
           std::tie(other.cohort, other.x, other.y);
  }
};

void append_ancestry(
    std::vector<feebl::adaptive_particles::AncestryContribution>& target,
    const std::vector<feebl::adaptive_particles::AncestryContribution>& add) {
  std::vector<feebl::adaptive_particles::AncestryContribution> combined;
  combined.reserve(target.size() + add.size());
  std::size_t i = 0U;
  std::size_t j = 0U;
  while (i < target.size() || j < add.size()) {
    if (j == add.size() ||
        (i < target.size() && target[i].root_id < add[j].root_id)) {
      combined.push_back(target[i++]);
    } else if (i == target.size() || add[j].root_id < target[i].root_id) {
      combined.push_back(add[j++]);
    } else {
      const double contribution =
          target[i].contribution + add[j].contribution;
      if (!finite_positive(contribution))
        throw std::runtime_error(
            "adaptive source ancestry merge produced invalid mass");
      combined.push_back(
          feebl::adaptive_particles::AncestryContribution{
              target[i].root_id, contribution});
      ++i;
      ++j;
    }
  }
  target.swap(combined);
}

std::uint64_t merged_path_hash(
    const std::vector<feebl::adaptive_particles::AncestryContribution>&
        ancestry,
    std::size_t global_x, std::size_t global_y,
    std::uint64_t cohort) {
  std::uint64_t state = kFnvOffset;
  hash_u64(state, UINT64_C(0x4d4552474543454c));  // "MERGECEL"
  hash_u64(state, global_x);
  hash_u64(state, global_y);
  hash_u64(state, cohort);
  for (const auto& term : ancestry) {
    hash_u64(state, term.root_id);
    hash_double(state, term.contribution);
  }
  return state;
}

}  // namespace

namespace feebl {
namespace adaptive_particles {

const char* mode_name(Mode mode) noexcept {
  switch (mode) {
    case Mode::LegacyOff: return "legacy_off";
    case Mode::Diagnostics: return "diagnostics";
    case Mode::SplitOnly: return "split_only";
    case Mode::CoupledSplitMerge: return "coupled_split_merge";
  }
  return "invalid";
}

void validate_configuration(const Configuration& configuration) {
  const int mode = static_cast<int>(configuration.mode);
  if (mode < 0 || mode > 3)
    throw std::invalid_argument(
        "adaptive_macroparticle_mode must be 0, 1, 2, or 3");
  if (configuration.domain != "field_source" &&
      configuration.domain != "full_physics")
    throw std::invalid_argument(
        "adaptive_macroparticle_domain must be field_source or full_physics");
  if (mode != 0 && configuration.domain != "field_source")
    throw std::invalid_argument(
        "adaptive full_physics domain is not physics-qualified");
  if (configuration.source_evolution != "projected_current_state" &&
      configuration.source_evolution != "persistent_replayed_roots")
    throw std::invalid_argument(
        "adaptive_macroparticle_source_evolution is invalid");
  if (mode != 0 &&
      configuration.source_evolution != "projected_current_state")
    throw std::invalid_argument(
        "persistent_replayed_roots requires the external replay generator");
  if (configuration.decision_scope != "encounter" ||
      configuration.commit_scope != "wave")
    throw std::invalid_argument(
        "adaptive particle decisions must be encounter proposals with wave commits");
  if (configuration.strategy != "coupled_error_budget_v1" ||
      configuration.split_algorithm != "assignment_preserving_v1")
    throw std::invalid_argument(
        "unsupported adaptive macroparticle strategy/split algorithm");
  if (configuration.root_source != "physical_primary_roots_v1" &&
      configuration.root_source != "pilot_replay_rqmc_v1")
    throw std::invalid_argument(
        "unsupported adaptive macroparticle root-source policy");
  if (mode != 0 &&
      configuration.root_source != "physical_primary_roots_v1")
    throw std::invalid_argument(
        "pilot_replay_rqmc_v1 is reserved but not implemented; the current "
        "projected source requires physical_primary_roots_v1");
  if (mode != 0 && configuration.root_source == "physical_primary_roots_v1" &&
      (configuration.root_seed != 0U || configuration.scramble_id != 0U))
    throw std::invalid_argument(
        "physical_primary_roots_v1 does not consume an adaptive root seed or "
        "RQMC scramble; both fields must be zero");
  if (mode != 0 &&
      configuration.merge_algorithm != "exact_cell_projection_v1")
    throw std::invalid_argument(
        "active adaptive macroparticle modes require the canonical "
        "exact_cell_projection_v1 merge identifier");
  const auto finite_nonnegative = [](double value) {
    return std::isfinite(value) && value >= 0.0;
  };
  if (!finite_nonnegative(configuration.error_target) ||
      !finite_nonnegative(configuration.resample_bias_target) ||
      !finite_nonnegative(configuration.minimum_root_support) ||
      !finite_nonnegative(configuration.minimum_clone_support) ||
      !finite_nonnegative(configuration.coarsen_threshold) ||
      !finite_nonnegative(configuration.refine_threshold) ||
      configuration.coarsen_threshold > configuration.refine_threshold ||
      !(configuration.maximum_change_factor >= 1.0) ||
      !std::isfinite(configuration.maximum_change_factor) ||
      !(configuration.maximum_weight_ratio >= 1.0) ||
      !std::isfinite(configuration.maximum_weight_ratio) ||
      configuration.update_interval == 0U ||
      configuration.statistical_cohorts == 0U ||
      configuration.dwell == 0U ||
      configuration.maximum_generation == 0U)
    throw std::invalid_argument(
        "invalid adaptive macroparticle error/controller limits");
  if (mode >= 2 &&
      (!(configuration.error_target > 0.0) ||
       !(configuration.error_target < 1.0) ||
       !(configuration.resample_bias_target < 1.0)))
    throw std::invalid_argument(
        "active adaptive macroparticle modes require error_target in (0,1) "
        "and resample_bias_target in [0,1)");
  if (configuration.determinism != "strict" &&
      configuration.determinism != "fast_qualified")
    throw std::invalid_argument(
        "adaptive_macroparticle_determinism must be strict or fast_qualified");
  if (configuration.profile != "off" &&
      configuration.profile != "summary" &&
      configuration.profile != "detail")
    throw std::invalid_argument(
        "adaptive_macroparticle_profile must be off, summary, or detail");
  if (mode != 0 &&
      (configuration.update_interval != 1U ||
       configuration.cooldown != 0U || configuration.dwell != 2U ||
       configuration.refine_threshold != 1.0 ||
       configuration.coarsen_threshold != 0.35 ||
       configuration.determinism != "strict"))
    throw std::invalid_argument(
        "the v1 exact projected-source model supports only "
        "update_interval=1, cooldown=0, dwell=2, refine_threshold=1, "
        "coarsen_threshold=0.35, and strict determinism");
}

Metrics compute_metrics(const std::vector<SourceRepresentative>& source) {
  Metrics result;
  result.representative_count = source.size();
  if (source.empty()) return result;

  std::set<PRIMARY_PARTICLE_ID> roots;
  std::set<std::uint64_t> cohorts;
  long double total = 0.0L;
  long double square_sum = 0.0L;
  long double variance_mass = 0.0L;
  result.minimum_weight = std::numeric_limits<double>::infinity();
  long double mean = 0.0L;
  for (const SourceRepresentative& representative : source) {
    if (!finite_positive(representative.physical_weight) ||
        !finite_positive(representative.ancestry_variance_mass) ||
        representative.ancestry.empty())
      throw std::runtime_error(
          "adaptive source contains an invalid weight or ancestry record");
    long double ancestry_total = 0.0L;
    PRIMARY_PARTICLE_ID previous = 0U;
    bool first = true;
    for (const AncestryContribution& term : representative.ancestry) {
      if (!finite_positive(term.contribution) ||
          (!first && term.root_id <= previous))
        throw std::runtime_error(
            "adaptive source ancestry must be positive and strictly ordered");
      first = false;
      previous = term.root_id;
      roots.insert(term.root_id);
      ancestry_total += term.contribution;
    }
    const long double scale = std::max(
        static_cast<long double>(representative.physical_weight), 1.0L);
    if (std::fabs(
            ancestry_total - representative.physical_weight) >
        128.0L * std::numeric_limits<double>::epsilon() * scale)
      throw std::runtime_error(
          "adaptive source ancestry does not close to representative weight");
    if (representative.ancestry.size() > 1U)
      ++result.many_root_representatives;
    cohorts.insert(representative.statistical_cohort_id);
    total += representative.physical_weight;
    square_sum += static_cast<long double>(representative.physical_weight) *
                  representative.physical_weight;
    variance_mass += representative.ancestry_variance_mass;
    result.minimum_weight = std::min(
        result.minimum_weight, representative.physical_weight);
    result.maximum_weight = std::max(
        result.maximum_weight, representative.physical_weight);
    mean += representative.physical_weight;
  }
  result.total_weight = static_cast<double>(total);
  result.distinct_root_count = roots.size();
  result.statistical_cohort_count = cohorts.size();
  result.weight_ess = static_cast<double>(total * total / square_sum);
  result.clone_support_proxy =
      static_cast<double>(total * total / variance_mass);
  mean /= static_cast<long double>(source.size());
  long double centered = 0.0L;
  for (const SourceRepresentative& representative : source) {
    const long double delta = representative.physical_weight - mean;
    centered += delta * delta;
  }
  result.weight_coefficient_of_variation = static_cast<double>(
      std::sqrt(centered / static_cast<long double>(source.size())) / mean);
  return result;
}

std::uint64_t source_content_hash(
    const std::vector<SourceRepresentative>& source,
    std::uint64_t beam_label,
    std::uint64_t epoch,
    std::uint64_t population_revision) {
  std::uint64_t state = kFnvOffset;
  hash_u64(state, UINT64_C(0x414d50534f555231));  // "AMPSOUR1"
  hash_u64(state, kSourceRepresentativeSchema);
  hash_u64(state, beam_label);
  hash_u64(state, epoch);
  hash_u64(state, population_revision);
  hash_u64(state, source.size());
  for (const SourceRepresentative& representative : source) {
    hash_u64(state, representative.representative_id.high);
    hash_u64(state, representative.representative_id.low);
    hash_u64(state, representative.statistical_cohort_id);
    hash_u64(state, representative.lineage_path);
    hash_u64(state, representative.birth_epoch);
    hash_u64(state, representative.generation);
    hash_u64(state, representative.flags);
    hash_u64(state, representative.global_cell_x);
    hash_u64(state, representative.global_cell_y);
    hash_double(state, representative.x);
    hash_double(state, representative.y);
    hash_double(state, representative.physical_weight);
    hash_double(state, representative.ancestry_variance_mass);
    hash_u64(state, representative.ancestry.size());
    for (const AncestryContribution& term : representative.ancestry) {
      hash_u64(state, term.root_id);
      hash_double(state, term.contribution);
    }
  }
  return state;
}

SourceEnsemble build_assignment_preserving_source(
    const Configuration& configuration,
    const adaptive_grid::FieldLevelGeometry& geometry,
    const std::vector<RootProjection>& roots,
    std::uint64_t beam_label,
    std::uint64_t epoch,
    std::uint64_t prior_population_revision) {
  validate_configuration(configuration);
  (void)checked_cells(geometry);
  if (beam_label != 1U && beam_label != 2U)
    throw std::invalid_argument("adaptive source beam label must be 1 or 2");

  SourceEnsemble proposed;
  proposed.receipt.roots_seen = roots.size();
  proposed.receipt.diagnostics_only =
      configuration.mode == Mode::Diagnostics;
  if (configuration.mode == Mode::LegacyOff) {
    proposed.receipt.reason = "legacy mode does not construct a source";
    return proposed;
  }

  long double root_weight = 0.0L;
  long double root_x_moment = 0.0L;
  long double root_y_moment = 0.0L;
  double minimum_root_weight = std::numeric_limits<double>::infinity();
  double maximum_root_weight = 0.0;
  std::set<RepresentativeId> unique_ids;
  if (roots.size() > std::numeric_limits<std::size_t>::max() / 4U)
    throw std::overflow_error(
        "adaptive source proposal size overflows size_t");
  const std::size_t maximum_representatives = 4U * roots.size();
  if (configuration.maximum_bytes != 0U) {
    // Bound the transactional high-water before allocating.  Mode 3 briefly
    // owns the split array, ordered map nodes and the merged ancestry ledger
    // simultaneously.  The fixed 128-byte term is a conservative allowance
    // for ordered-container links, allocator metadata and spare capacity.
    const std::size_t bytes_per_candidate =
        sizeof(SourceRepresentative) + sizeof(AncestryContribution) + 128U;
    if (maximum_representatives >
        configuration.maximum_bytes / bytes_per_candidate)
      throw std::runtime_error(
          "adaptive source transactional high-water exceeds "
          "adaptive_macroparticle_max_bytes");
  }
  proposed.representatives.reserve(maximum_representatives);
  for (const RootProjection& root : roots) {
    if (!finite_positive(root.physical_weight) ||
        !std::isfinite(root.x) || !std::isfinite(root.y))
      throw std::invalid_argument(
          "adaptive source root has invalid state/weight");
    if (!root.cic.accepted) {
      ++proposed.receipt.roots_rejected;
      continue;
    }
    if (root.cic.base_x < 0 || root.cic.base_y < 0 ||
        !(root.cic.weight_x >= 0.0f && root.cic.weight_x <= 1.0f) ||
        !(root.cic.weight_y >= 0.0f && root.cic.weight_y <= 1.0f))
      throw std::invalid_argument(
          "adaptive source root has an invalid CIC descriptor");
    const std::size_t base_x = static_cast<std::size_t>(root.cic.base_x);
    const std::size_t base_y = static_cast<std::size_t>(root.cic.base_y);
    if (base_x + 1U >= geometry.n_x || base_y + 1U >= geometry.n_y)
      throw std::invalid_argument(
          "adaptive source CIC descriptor escapes local geometry");
    const double wx = root.cic.weight_x;
    const double wy = root.cic.weight_y;
    const double fractions[4] = {
        (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy),
        (1.0 - wx) * wy, wx * wy};
    const std::size_t local_x[4] = {base_x, base_x + 1U,
                                    base_x, base_x + 1U};
    const std::size_t local_y[4] = {base_y, base_y,
                                    base_y + 1U, base_y + 1U};
    const std::uint32_t generation = 1U;
    if (generation > configuration.maximum_generation)
      throw std::runtime_error(
          "adaptive source split exceeds maximum generation");
    for (std::uint64_t child = 0U; child < 4U; ++child) {
      if (!(fractions[child] > 0.0)) continue;
      SourceRepresentative representative;
      representative.statistical_cohort_id =
          root.statistical_cohort_id;
      representative.lineage_path = child;
      representative.birth_epoch = epoch;
      representative.generation = generation;
      representative.global_cell_x =
          static_cast<std::size_t>(geometry.origin_x) + local_x[child];
      representative.global_cell_y =
          static_cast<std::size_t>(geometry.origin_y) + local_y[child];
      representative.x =
          (static_cast<double>(representative.global_cell_x) + 0.5 -
           geometry.offset_x) * geometry.delta_x;
      representative.y =
          (static_cast<double>(representative.global_cell_y) + 0.5 -
           geometry.offset_y) * geometry.delta_y;
      representative.physical_weight =
          root.physical_weight * fractions[child];
      // Sum over children is root_weight^2.  This is intentionally not
      // child_weight^2, which would manufacture clone support.
      representative.ancestry_variance_mass =
          root.physical_weight * root.physical_weight * fractions[child];
      representative.ancestry.push_back(
          AncestryContribution{root.root_id,
                               representative.physical_weight});
      representative.representative_id = child_id(
          beam_label, root.root_id, epoch, child);
      if (!unique_ids.insert(representative.representative_id).second)
        throw std::runtime_error(
            "adaptive source representative-ID collision");
      proposed.representatives.push_back(std::move(representative));
      ++proposed.receipt.split_children_created;
    }
    root_weight += root.physical_weight;
    minimum_root_weight = std::min(minimum_root_weight,
                                   root.physical_weight);
    maximum_root_weight = std::max(maximum_root_weight,
                                   root.physical_weight);
    // The hard conservation target is the actual FeeBL CIC assignment, whose
    // mapper deliberately rounds its normalized coordinate in binary32.
    // Compare children with that projection—not with an idealized binary64
    // particle coordinate that the production assignment never deposits.
    const double projected_x =
        (static_cast<double>(geometry.origin_x) + base_x + 0.5 + wx -
         geometry.offset_x) * geometry.delta_x;
    const double projected_y =
        (static_cast<double>(geometry.origin_y) + base_y + 0.5 + wy -
         geometry.offset_y) * geometry.delta_y;
    root_x_moment += root.physical_weight * projected_x;
    root_y_moment += root.physical_weight * projected_y;
    ++proposed.receipt.roots_accepted;
  }
  proposed.receipt.representatives_before_merge =
      proposed.representatives.size();

  if (configuration.mode == Mode::CoupledSplitMerge) {
    std::map<MergeKey, SourceRepresentative> merged;
    std::set<MergeKey> groups_with_reduction;
    for (const SourceRepresentative& representative :
         proposed.representatives) {
      const MergeKey key{representative.statistical_cohort_id,
                         representative.global_cell_x,
                         representative.global_cell_y};
      auto insertion = merged.emplace(key, representative);
      if (insertion.second) continue;
      groups_with_reduction.insert(key);
      SourceRepresentative& destination = insertion.first->second;
      destination.physical_weight += representative.physical_weight;
      destination.ancestry_variance_mass +=
          representative.ancestry_variance_mass;
      append_ancestry(destination.ancestry, representative.ancestry);
    }
    // Count only occupied groups that actually reduced the population.  A
    // singleton output cell is not a merge and must not inflate telemetry.
    proposed.receipt.merge_groups = groups_with_reduction.size();
    proposed.representatives.clear();
    proposed.representatives.reserve(merged.size());
    for (auto& entry : merged) {
      SourceRepresentative& representative = entry.second;
      representative.generation = 2U;
      if (representative.generation > configuration.maximum_generation)
        throw std::runtime_error(
            "adaptive source merge exceeds maximum generation");
      representative.lineage_path = merged_path_hash(
          representative.ancestry,
          representative.global_cell_x,
          representative.global_cell_y,
          representative.statistical_cohort_id);
      representative.representative_id = child_id(
          beam_label, representative.ancestry.front().root_id,
          epoch, representative.lineage_path);
      proposed.representatives.push_back(std::move(representative));
    }
  }
  proposed.receipt.representatives_after_merge =
      proposed.representatives.size();

  if (configuration.maximum_count != 0U &&
      proposed.representatives.size() > configuration.maximum_count)
    throw std::runtime_error(
        "adaptive source exceeds adaptive_macroparticle_max_count");
  if (configuration.maximum_bytes != 0U) {
    std::size_t ancestry_capacity = 0U;
    for (const SourceRepresentative& representative :
         proposed.representatives) {
      if (representative.ancestry.capacity() >
          std::numeric_limits<std::size_t>::max() - ancestry_capacity)
        throw std::overflow_error(
            "adaptive source ancestry capacity overflows size_t");
      ancestry_capacity += representative.ancestry.capacity();
    }
    if (proposed.representatives.capacity() >
            std::numeric_limits<std::size_t>::max() /
                sizeof(SourceRepresentative) ||
        ancestry_capacity >
            std::numeric_limits<std::size_t>::max() /
                sizeof(AncestryContribution))
      throw std::overflow_error(
          "adaptive source storage estimate overflows size_t");
    const std::size_t representative_bytes =
        proposed.representatives.capacity() * sizeof(SourceRepresentative);
    const std::size_t ancestry_bytes =
        ancestry_capacity * sizeof(AncestryContribution);
    if (ancestry_bytes >
        std::numeric_limits<std::size_t>::max() - representative_bytes)
      throw std::overflow_error(
          "adaptive source storage total overflows size_t");
    const std::size_t conservative_bytes =
        representative_bytes + ancestry_bytes;
    if (conservative_bytes > configuration.maximum_bytes)
      throw std::runtime_error(
          "adaptive source exceeds adaptive_macroparticle_max_bytes");
  }
  if (!roots.empty() && configuration.maximum_change_factor > 0.0 &&
      proposed.representatives.size() > static_cast<std::size_t>(
          std::ceil(configuration.maximum_change_factor * roots.size())))
    throw std::runtime_error(
        "adaptive source exceeds maximum per-epoch change factor");

  proposed.metrics = compute_metrics(proposed.representatives);
  if (configuration.minimum_root_support > 0.0 &&
      static_cast<double>(proposed.metrics.distinct_root_count) <
          configuration.minimum_root_support)
    throw std::runtime_error(
        "adaptive source does not meet minimum distinct-root support");
  if (configuration.minimum_clone_support > 0.0 &&
      proposed.metrics.clone_support_proxy <
          configuration.minimum_clone_support)
    throw std::runtime_error(
        "adaptive source does not meet minimum clone-support proxy");
  if (proposed.metrics.statistical_cohort_count >
      configuration.statistical_cohorts)
    throw std::runtime_error(
        "adaptive source contains more statistical cohorts than declared");
  // CIC shape coefficients may approach zero continuously at a cell edge.
  // Consequently the atom max/min ratio is unbounded even for a perfectly
  // uniform root ensemble and is not a valid super-particle diagnostic.  The
  // declared guard applies to authoritative root weights for this exact
  // projected-source algorithm; downstream lossy resamplers must define and
  // enforce their own representative impulse/concentration bound.
  if (proposed.receipt.roots_accepted != 0U &&
      minimum_root_weight > 0.0 &&
      maximum_root_weight / minimum_root_weight >
          configuration.maximum_weight_ratio)
    throw std::runtime_error(
        "adaptive source exceeds maximum input-root weight ratio");
  long double representative_weight = 0.0L;
  long double representative_x_moment = 0.0L;
  long double representative_y_moment = 0.0L;
  for (const SourceRepresentative& representative :
       proposed.representatives) {
    representative_weight += representative.physical_weight;
    representative_x_moment +=
        representative.physical_weight * representative.x;
    representative_y_moment +=
        representative.physical_weight * representative.y;
  }
  const long double scale = std::max(std::fabs(root_weight), 1.0L);
  proposed.receipt.relative_weight_residual = static_cast<double>(
      std::fabs(representative_weight - root_weight) / scale);
  const long double x_scale = std::max(
      std::fabs(root_x_moment), scale * geometry.delta_x);
  const long double y_scale = std::max(
      std::fabs(root_y_moment), scale * geometry.delta_y);
  proposed.receipt.normalized_first_moment_residual =
      static_cast<double>(std::max(
          std::fabs(representative_x_moment - root_x_moment) / x_scale,
          std::fabs(representative_y_moment - root_y_moment) / y_scale));
  const double tolerance =
      512.0 * std::numeric_limits<double>::epsilon() *
      std::max<std::size_t>(1U, roots.size());
  if (proposed.receipt.relative_weight_residual > tolerance ||
      proposed.receipt.normalized_first_moment_residual > tolerance) {
    proposed.representatives.clear();
    proposed.metrics = Metrics();
    proposed.receipt.rollback_count = 1U;
    proposed.receipt.reason =
        "assignment-preserving source failed conservation transaction";
    if (configuration.fail_on_unmet_error)
      throw std::runtime_error(proposed.receipt.reason);
    return proposed;
  }
  if (proposed.metrics.distinct_root_count !=
      proposed.receipt.roots_accepted)
    throw std::runtime_error(
        "adaptive source root support changed during projection");
  // The clone proxy must equal the root-weight ESS.  For unequal roots this
  // is not the number of roots, but it must never equal the much larger atom
  // weight ESS merely because a root was split.
  long double root_square_sum = 0.0L;
  for (const RootProjection& root : roots)
    if (root.cic.accepted)
      root_square_sum += root.physical_weight * root.physical_weight;
  const double expected_clone_support = root_square_sum > 0.0L
      ? static_cast<double>(root_weight * root_weight / root_square_sum)
      : 0.0;
  if (std::fabs(proposed.metrics.clone_support_proxy -
                expected_clone_support) >
      1024.0 * std::numeric_limits<double>::epsilon() *
          std::max(1.0, expected_clone_support))
    throw std::runtime_error(
        "adaptive source manufactured clone support");

  // A dry-run proposal has an identity and complete metrics, but does not
  // advance the live population revision.  Keeping the verified proposal in
  // memory makes Mode 1 auditable; `deposit_source` still refuses it because
  // its committed bit is false.
  proposed.population_revision =
      configuration.mode == Mode::Diagnostics
          ? prior_population_revision
          : prior_population_revision + 1U;
  proposed.receipt.content_hash = source_content_hash(
      proposed.representatives, beam_label, epoch,
      proposed.population_revision);
  hash_u64(proposed.receipt.content_hash,
           static_cast<std::uint64_t>(configuration.mode));
  hash_u64(proposed.receipt.content_hash, configuration.root_seed);
  hash_u64(proposed.receipt.content_hash, configuration.scramble_id);
  proposed.receipt.maximum_projection_residual = 0.0;
  proposed.receipt.exact_projection = true;
  proposed.receipt.committed =
      configuration.mode != Mode::Diagnostics;
  proposed.receipt.qualification =
      configuration.mode == Mode::Diagnostics
          ? "diagnostics_only" : "software_verified_field_source";
  proposed.receipt.reason =
      configuration.mode == Mode::Diagnostics
          ? "proposal verified but not activated"
          : "exact field-source projection committed";
  return proposed;
}

void deposit_source(const SourceEnsemble& source,
                    const adaptive_grid::FieldLevelGeometry& geometry,
                    std::vector<double>& density,
                    std::vector<double>& root_grouped_s1,
                    std::vector<double>& root_grouped_s2) {
  const std::size_t cells = checked_cells(geometry);
  density.assign(cells, 0.0);
  root_grouped_s1.assign(cells, 0.0);
  root_grouped_s2.assign(cells, 0.0);
  if (!source.receipt.committed)
    throw std::invalid_argument(
        "cannot deposit an uncommitted adaptive source proposal");
  if (source.receipt.content_hash == 0U)
    throw std::invalid_argument(
        "cannot deposit an unbound adaptive source proposal");

  // Accumulate ancestry by (cell, root), not by representative.  This is
  // what keeps S2—and therefore cell-level root support—unchanged by either
  // splitting or exact same-cell merging.
  std::map<std::pair<std::size_t, PRIMARY_PARTICLE_ID>, long double>
      per_root_cell;
  for (const SourceRepresentative& representative :
       source.representatives) {
    if (representative.global_cell_x <
            static_cast<std::size_t>(geometry.origin_x) ||
        representative.global_cell_y <
            static_cast<std::size_t>(geometry.origin_y))
      throw std::runtime_error(
          "adaptive source atom lies below local geometry");
    const std::size_t local_x = representative.global_cell_x -
        static_cast<std::size_t>(geometry.origin_x);
    const std::size_t local_y = representative.global_cell_y -
        static_cast<std::size_t>(geometry.origin_y);
    if (local_x >= geometry.n_x || local_y >= geometry.n_y)
      throw std::runtime_error(
          "adaptive source atom lies outside local geometry");
    const std::size_t cell = local_x * geometry.n_y + local_y;
    density[cell] += representative.physical_weight;
    for (const AncestryContribution& term : representative.ancestry)
      per_root_cell[std::make_pair(cell, term.root_id)] += term.contribution;
  }
  for (const auto& entry : per_root_cell) {
    const std::size_t cell = entry.first.first;
    const long double value = entry.second;
    root_grouped_s1[cell] += static_cast<double>(value);
    root_grouped_s2[cell] += static_cast<double>(value * value);
  }
}

}  // namespace adaptive_particles
}  // namespace feebl
