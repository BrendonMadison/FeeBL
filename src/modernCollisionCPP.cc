#include "modernCollisionCPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace feebl {
namespace modern {

namespace {

constexpr std::size_t kChargeClassCount = 3;

std::size_t charge_index(std::int8_t charge) {
  if (charge < -1 || charge > 1)
    throw std::invalid_argument(
        "modern collision charge_e must be one of {-1,0,+1}");
  return static_cast<std::size_t>(static_cast<int>(charge) + 1);
}

Scalar checked_sum(Scalar a, Scalar b, const char* quantity) {
  const Scalar result = a + b;
  if (!std::isfinite(result))
    throw std::overflow_error(std::string(quantity) +
                              " accumulation is non-finite");
  return result;
}

Scalar checked_product(Scalar a, Scalar b, const char* quantity) {
  const Scalar result = a * b;
  if (!std::isfinite(result))
    throw std::overflow_error(std::string(quantity) +
                              " product is non-finite");
  return result;
}

void checked_count_add(std::uint64_t add, std::uint64_t& target,
                       const char* quantity) {
  std::uint64_t result = 0;
  if (!checked_add(target, add, result))
    throw std::overflow_error(std::string(quantity) + " count overflow");
  target = result;
}

void checked_count_product_add(std::uint64_t a, std::uint64_t b,
                               std::uint64_t& target,
                               const char* quantity) {
  std::uint64_t product = 0;
  if (!checked_multiply(a, b, product))
    throw std::overflow_error(std::string(quantity) +
                              " count multiplication overflow");
  checked_count_add(product, target, quantity);
}

Scalar positive_sum_roundoff_bound(Scalar scale,
                                   std::uint64_t term_count) noexcept;

bool electron_family(ParticleSpecies species) noexcept {
  return species == ParticleSpecies::Electron ||
         species == ParticleSpecies::Positron;
}

bool selected_pair(EePairPolicy policy, ParticleSpecies species1,
                   std::int8_t charge1, ParticleSpecies species2,
                   std::int8_t charge2) {
  if (!electron_family(species1) || !electron_family(species2)) return false;
  switch (policy) {
    case EePairPolicy::AllElectronFamily:
      return true;
    case EePairPolicy::OppositeElectricCharge:
      return charge1 == -charge2;
    case EePairPolicy::SameElectricCharge:
      return charge1 == charge2;
    case EePairPolicy::Unspecified:
      break;
  }
  throw std::logic_error("unresolved e-e pair policy reached hot kernel");
}

struct ParticleKinematics {
  Scalar energy = 0.0;
  Scalar momentum = 0.0;
  Scalar vx = 0.0;
  Scalar vy = 0.0;
  Scalar vz = 0.0;
};

std::vector<ParticleKinematics> prepare_kinematics(
    const BeamSoA& beam, Scalar electron_mass, const char* beam_name) {
  const std::size_t count = static_cast<std::size_t>(beam.size());
  std::vector<ParticleKinematics> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    (void)charge_index(beam.charge_e()[i]);
    if (!electron_family(beam.species()[i])) continue;

    const Scalar energy = beam.energy()[i];
    if (energy < electron_mass)
      throw std::invalid_argument(std::string(beam_name) +
                                  " electron-family energy is below rest mass at particle " +
                                  std::to_string(i));
    const Scalar vx = beam.vx()[i];
    const Scalar vy = beam.vy()[i];
    const long double transverse2 =
        static_cast<long double>(vx) * vx +
        static_cast<long double>(vy) * vy;
    if (!(transverse2 <= 1.0L))
      throw std::invalid_argument(std::string(beam_name) +
                                  " transverse direction norm exceeds unity at particle " +
                                  std::to_string(i));

    const long double longitudinal2 = 1.0L - transverse2;
    const long double longitudinal = std::sqrt(longitudinal2);
    const long double p2 =
        (static_cast<long double>(energy) - electron_mass) *
        (static_cast<long double>(energy) + electron_mass);
    if (!(p2 >= 0.0L) || !std::isfinite(p2))
      throw std::invalid_argument(std::string(beam_name) +
                                  " electron momentum invariant is invalid at particle " +
                                  std::to_string(i));
    const long double momentum = std::sqrt(p2);
    if (!std::isfinite(longitudinal) || !std::isfinite(momentum) ||
        momentum > std::numeric_limits<Scalar>::max())
      throw std::overflow_error(std::string(beam_name) +
                                " electron kinematics are not representable");

    ParticleKinematics& k = result[i];
    k.energy = energy;
    k.momentum = static_cast<Scalar>(momentum);
    k.vx = vx;
    k.vy = vy;
    k.vz = static_cast<Scalar>(beam.direction()[i]) *
           static_cast<Scalar>(longitudinal);
  }
  return result;
}

void validate_grid_pair(const CellCSR& cells1, const CellCSR& cells2) {
  if (cells1.order() != CellOrder::ByStableParticleId ||
      cells2.order() != CellOrder::ByStableParticleId)
    throw std::invalid_argument(
        "modern collision requires CellOrder::ByStableParticleId");

  const UniformGrid2D& a = cells1.grid();
  const UniformGrid2D& b = cells2.grid();
  if (a.mapping_mode() != b.mapping_mode() || a.nx() != b.nx() ||
      a.ny() != b.ny() || a.active_nx() != b.active_nx() ||
      a.active_ny() != b.active_ny() ||
      a.cell_count() != b.cell_count() || a.xmin() != b.xmin() ||
      a.xmax() != b.xmax() || a.ymin() != b.ymin() ||
      a.ymax() != b.ymax())
    throw std::invalid_argument(
        "modern collision CellCSR grids are not identical");
}

InvariantMassSpectrumLane make_spectrum_lane(
    const InvariantMassSpectrumConfig& config) {
  InvariantMassSpectrumLane result;
  if (!config.enabled()) return result;
  if (!std::isfinite(config.lower) || !std::isfinite(config.upper) ||
      !(config.upper > config.lower))
    throw std::invalid_argument(
        "W spectrum requires finite, strictly increasing bounds");
  if (config.bins > result.bin_weight.max_size())
    throw std::length_error("W spectrum bin count is not representable");
  const Scalar width =
      (config.upper - config.lower) / static_cast<Scalar>(config.bins);
  if (!std::isfinite(width) || !(width > 0.0))
    throw std::invalid_argument("W spectrum bin width is not representable");
  result.lower = config.lower;
  result.upper = config.upper;
  result.bin_width = width;
  result.bin_weight.assign(config.bins, 0.0);
  return result;
}

void add_moment(WeightedMomentLane& moment, Scalar weight, Scalar value) {
  const Scalar w2 = checked_product(weight, weight, "W moment weight-squared");
  const Scalar wx = checked_product(weight, value, "W first moment");
  const Scalar x2 = checked_product(value, value, "W value-squared");
  const Scalar wx2 = checked_product(weight, x2, "W second moment");
  moment.weight_sum =
      checked_sum(moment.weight_sum, weight, "W moment weight");
  moment.weight_squared_sum = checked_sum(
      moment.weight_squared_sum, w2, "W moment weight-squared");
  moment.weighted_value_sum = checked_sum(
      moment.weighted_value_sum, wx, "W first moment");
  moment.weighted_value_squared_sum = checked_sum(
      moment.weighted_value_squared_sum, wx2, "W second moment");
}

void add_spectrum(InvariantMassSpectrumLane& spectrum, Scalar weight,
                  Scalar invariant_mass) {
  add_moment(spectrum.moments, weight, invariant_mass);
  if (!spectrum.configured()) return;
  if (invariant_mass < spectrum.lower) {
    spectrum.underflow_weight = checked_sum(
        spectrum.underflow_weight, weight, "W spectrum underflow");
    return;
  }
  if (invariant_mass >= spectrum.upper) {
    spectrum.overflow_weight = checked_sum(
        spectrum.overflow_weight, weight, "W spectrum overflow");
    return;
  }
  const Scalar scaled =
      (invariant_mass - spectrum.lower) / spectrum.bin_width;
  std::size_t bin = static_cast<std::size_t>(scaled);
  if (bin >= spectrum.bin_weight.size())
    bin = spectrum.bin_weight.size() - 1;
  spectrum.bin_weight[bin] = checked_sum(
      spectrum.bin_weight[bin], weight, "W spectrum bin");
}

struct PairKinematics {
  Scalar ecm = 0.0;
  Scalar invariant_mass = 0.0;
};

PairKinematics pair_kinematics(const ParticleKinematics& a,
                               const ParticleKinematics& b,
                               Scalar electron_mass) {
  const long double dot =
      static_cast<long double>(a.vx) * b.vx +
      static_cast<long double>(a.vy) * b.vy +
      static_cast<long double>(a.vz) * b.vz;
  const long double angular = 1.0L - dot;
  const long double ecm2 =
      2.0L * static_cast<long double>(a.energy) * b.energy * angular;
  const long double m = electron_mass;
  const long double w2 =
      2.0L * m * m +
      2.0L * (static_cast<long double>(a.energy) * b.energy -
              static_cast<long double>(a.momentum) * b.momentum * dot);

  const long double scale =
      1.0L + 2.0L * static_cast<long double>(a.energy) * b.energy;
  const long double tolerance =
      64.0L * std::numeric_limits<Scalar>::epsilon() * scale;
  if (ecm2 < -tolerance || w2 < -tolerance ||
      !std::isfinite(ecm2) || !std::isfinite(w2))
    throw std::domain_error("pair four-momentum invariant is invalid");

  const long double ecm = std::sqrt(std::max(0.0L, ecm2));
  const long double invariant_mass = std::sqrt(std::max(0.0L, w2));
  if (!std::isfinite(ecm) || !std::isfinite(invariant_mass) ||
      ecm > std::numeric_limits<Scalar>::max() ||
      invariant_mass > std::numeric_limits<Scalar>::max())
    throw std::overflow_error("pair energy invariant is not representable");
  return PairKinematics{static_cast<Scalar>(ecm),
                        static_cast<Scalar>(invariant_mass)};
}

void add_pair_observables(CollisionResultLane& lane, Scalar weight,
                          const PairKinematics& kinematics,
                          Scalar high_threshold,
                          bool add_ecm_squared) {
  if (kinematics.ecm > high_threshold)
    lane.ee_high_weight = checked_sum(
        lane.ee_high_weight, weight, "high-ECM e-e luminosity");
  const Scalar wecm =
      checked_product(weight, kinematics.ecm, "ECM first moment");
  const Scalar ecm2 = checked_product(
      kinematics.ecm, kinematics.ecm, "ECM squared");
  lane.ecm_weighted_sum = checked_sum(
      lane.ecm_weighted_sum, wecm, "ECM first moment");
  if (add_ecm_squared) {
    const Scalar wecm2 =
        checked_product(weight, ecm2, "ECM second moment");
    lane.ecm_squared_weighted_sum = checked_sum(
        lane.ecm_squared_weighted_sum, wecm2, "ECM second moment");
  }
  add_spectrum(lane.w_spectrum, weight, kinematics.invariant_mass);
}

struct ElectronSufficientStatistics {
  Scalar weight = 0.0;
  std::uint64_t count = 0;
  Scalar weighted_energy = 0.0;
  Scalar weighted_energy_vx = 0.0;
  Scalar weighted_energy_vy = 0.0;
  Scalar weighted_energy_vz = 0.0;
};

struct CellSufficientStatistics {
  std::array<Scalar, kChargeClassCount> all_weight{};
  std::array<ElectronSufficientStatistics, kChargeClassCount> electron{};
};

CellSufficientStatistics cell_statistics(const BeamSoA& beam,
                                         const CellCSR& cells,
                                         const std::vector<ParticleKinematics>&
                                             kinematics,
                                         CellKey cell) {
  CellSufficientStatistics result;
  const CellRange range = cells.cell_range(cell);
  for (ArrayOffset pos = range.begin; pos < range.end; ++pos) {
    const ParticleIndex particle =
        cells.particle_indices()[static_cast<std::size_t>(pos)];
    const std::size_t p = static_cast<std::size_t>(particle);
    const std::size_t charge = charge_index(beam.charge_e()[p]);
    const Scalar weight = beam.macro_weight()[p];
    result.all_weight[charge] = checked_sum(
        result.all_weight[charge], weight, "cell charge-class weight");
    if (electron_family(beam.species()[p])) {
      ElectronSufficientStatistics& electron = result.electron[charge];
      electron.weight = checked_sum(
          electron.weight, weight, "cell electron-family weight");
      checked_count_add(1, electron.count, "cell electron-family");
      const Scalar weighted_energy = checked_product(
          weight, kinematics[p].energy, "cell weighted electron energy");
      electron.weighted_energy = checked_sum(
          electron.weighted_energy, weighted_energy,
          "cell weighted electron energy");
      electron.weighted_energy_vx = checked_sum(
          electron.weighted_energy_vx,
          checked_product(weighted_energy, kinematics[p].vx,
                          "cell weighted electron x momentum direction"),
          "cell weighted electron x momentum direction");
      electron.weighted_energy_vy = checked_sum(
          electron.weighted_energy_vy,
          checked_product(weighted_energy, kinematics[p].vy,
                          "cell weighted electron y momentum direction"),
          "cell weighted electron y momentum direction");
      electron.weighted_energy_vz = checked_sum(
          electron.weighted_energy_vz,
          checked_product(weighted_energy, kinematics[p].vz,
                          "cell weighted electron z momentum direction"),
          "cell weighted electron z momentum direction");
    }
  }
  return result;
}

void factor_charge_matrix(CollisionResultLane& lane,
                          const CellSufficientStatistics& a,
                          const CellSufficientStatistics& b) {
  for (std::size_t i = 0; i < kChargeClassCount; ++i) {
    for (std::size_t j = 0; j < kChargeClassCount; ++j) {
      const Scalar contribution = checked_product(
          a.all_weight[i], b.all_weight[j],
          "factored charge-class luminosity");
      const std::size_t index = i * kChargeClassCount + j;
      lane.charge_class.weight[index] = checked_sum(
          lane.charge_class.weight[index], contribution,
          "factored charge-class luminosity");
    }
  }
}

ElectronSufficientStatistics combine_statistics(
    const std::array<ElectronSufficientStatistics, kChargeClassCount>&
        categories) {
  ElectronSufficientStatistics result;
  for (const ElectronSufficientStatistics& category : categories) {
    result.weight = checked_sum(result.weight, category.weight,
                                "combined electron weight");
    checked_count_add(category.count, result.count,
                      "combined electron");
    result.weighted_energy = checked_sum(
        result.weighted_energy, category.weighted_energy,
        "combined weighted electron energy");
    result.weighted_energy_vx = checked_sum(
        result.weighted_energy_vx, category.weighted_energy_vx,
        "combined weighted electron x momentum direction");
    result.weighted_energy_vy = checked_sum(
        result.weighted_energy_vy, category.weighted_energy_vy,
        "combined weighted electron y momentum direction");
    result.weighted_energy_vz = checked_sum(
        result.weighted_energy_vz, category.weighted_energy_vz,
        "combined weighted electron z momentum direction");
  }
  return result;
}

void add_factored_category(const ElectronSufficientStatistics& a,
                           const ElectronSufficientStatistics& b,
                           CollisionResultLane& lane) {
  const Scalar contribution = checked_product(
      a.weight, b.weight, "factored e-e luminosity");
  lane.ee_weight =
      checked_sum(lane.ee_weight, contribution, "factored e-e luminosity");
  checked_count_product_add(a.count, b.count, lane.ee_pair_count,
                            "factored e-e pair");

  // For the massless legacy ECM definition,
  //   w*ECM^2 = 2 (w1 E1)(w2 E2) (1 - n1.n2),
  // so the second moment is exactly separable into four dot products.  ECM,
  // the threshold predicate and W itself contain square roots/pairwise bins
  // and intentionally remain Cartesian.
  const long double energy =
      static_cast<long double>(a.weighted_energy) * b.weighted_energy;
  const long double dot =
      static_cast<long double>(a.weighted_energy_vx) *
          b.weighted_energy_vx +
      static_cast<long double>(a.weighted_energy_vy) *
          b.weighted_energy_vy +
      static_cast<long double>(a.weighted_energy_vz) *
          b.weighted_energy_vz;
  long double ecm2_contribution = 2.0L * (energy - dot);
  const long double scale =
      1.0L + 2.0L * (std::fabs(energy) + std::fabs(dot));
  std::uint64_t accumulated_terms = 0;
  if (!checked_add(a.count, b.count, accumulated_terms))
    accumulated_terms = std::numeric_limits<std::uint64_t>::max();
  const long double tolerance = static_cast<long double>(
      positive_sum_roundoff_bound(static_cast<Scalar>(std::min<long double>(
                                       scale,
                                       std::numeric_limits<Scalar>::max())),
                                   accumulated_terms));
  if (ecm2_contribution < -tolerance ||
      !std::isfinite(ecm2_contribution))
    throw std::domain_error(
        "factored ECM second-moment invariant is invalid");
  if (ecm2_contribution < 0.0L) ecm2_contribution = 0.0L;
  if (ecm2_contribution > std::numeric_limits<Scalar>::max())
    throw std::overflow_error(
        "factored ECM second moment is not representable");
  lane.ecm_squared_weighted_sum = checked_sum(
      lane.ecm_squared_weighted_sum,
      static_cast<Scalar>(ecm2_contribution),
      "factored ECM second moment");
}

void factor_ee_category(CollisionResultLane& lane,
                        const CellSufficientStatistics& a,
                        const CellSufficientStatistics& b,
                        EePairPolicy policy) {
  if (policy == EePairPolicy::AllElectronFamily) {
    add_factored_category(combine_statistics(a.electron),
                          combine_statistics(b.electron), lane);
    return;
  }

  constexpr std::size_t negative = 0;
  constexpr std::size_t positive = 2;
  if (policy == EePairPolicy::OppositeElectricCharge) {
    add_factored_category(a.electron[negative], b.electron[positive], lane);
    add_factored_category(a.electron[positive], b.electron[negative], lane);
    return;
  }
  if (policy == EePairPolicy::SameElectricCharge) {
    add_factored_category(a.electron[negative], b.electron[negative], lane);
    add_factored_category(a.electron[positive], b.electron[positive], lane);
    return;
  }
  throw std::logic_error("unresolved e-e policy in factored kernel");
}

bool same_spectrum_shape(const InvariantMassSpectrumLane& a,
                         const InvariantMassSpectrumLane& b) noexcept {
  return a.lower == b.lower && a.upper == b.upper &&
         a.bin_width == b.bin_width &&
         a.bin_weight.size() == b.bin_weight.size();
}

Scalar positive_sum_roundoff_bound(Scalar scale,
                                   std::uint64_t term_count) noexcept {
  // Higham's gamma_n bound for sequential sums of non-negative terms, with a
  // conservative factor for comparing two differently grouped sums.  Above
  // the useful arithmetic range, retain a small fail-closed ceiling rather
  // than allowing an invariant check to become vacuous.
  const long double n = static_cast<long double>(
      std::max<std::uint64_t>(term_count, 1u));
  const long double epsilon =
      static_cast<long double>(std::numeric_limits<Scalar>::epsilon());
  const long double nepsilon = n * epsilon;
  const long double gamma = nepsilon < 0.5L
                                ? nepsilon / (1.0L - nepsilon)
                                : 0.5L;
  const long double relative = std::min(
      1.0e-4L, std::max(64.0L * epsilon, 8.0L * gamma));
  return static_cast<Scalar>(relative * static_cast<long double>(scale));
}

}  // namespace

Scalar WeightedMomentLane::mean() const noexcept {
  return weight_sum > 0.0 ? weighted_value_sum / weight_sum : 0.0;
}

Scalar WeightedMomentLane::variance() const noexcept {
  if (!(weight_sum > 0.0)) return 0.0;
  const long double mu = static_cast<long double>(weighted_value_sum) /
                         static_cast<long double>(weight_sum);
  const long double value =
      static_cast<long double>(weighted_value_squared_sum) /
          static_cast<long double>(weight_sum) -
      mu * mu;
  return value > 0.0L ? static_cast<Scalar>(value) : 0.0;
}

Scalar WeightedMomentLane::effective_count() const noexcept {
  if (!(weight_squared_sum > 0.0)) return 0.0;
  const long double sum = static_cast<long double>(weight_sum);
  return static_cast<Scalar>(
      sum * sum / static_cast<long double>(weight_squared_sum));
}

Scalar ChargeClassLuminosityLane::at(std::int8_t beam1_charge,
                                     std::int8_t beam2_charge) const {
  return weight[charge_index(beam1_charge) * kChargeClassCount +
                 charge_index(beam2_charge)];
}

bool CollisionResultProvenance::operator==(
    const CollisionResultProvenance& other) const noexcept {
  return schema_version == other.schema_version &&
         ee_policy == other.ee_policy && evaluation == other.evaluation &&
         high_ecm_threshold == other.high_ecm_threshold &&
         electron_mass == other.electron_mass;
}

ValidationResult CollisionResultLane::validate() const {
  const auto valid_nonnegative = [](Scalar value) {
    return std::isfinite(value) && value >= 0.0;
  };
  if (provenance.schema_version != kModernCollisionResultSchemaVersion)
    return ValidationResult::failure(
        "collision result provenance schema is missing or unknown");
  if (provenance.ee_policy != EePairPolicy::AllElectronFamily &&
      provenance.ee_policy != EePairPolicy::OppositeElectricCharge &&
      provenance.ee_policy != EePairPolicy::SameElectricCharge)
    return ValidationResult::failure(
        "collision result provenance has an invalid e-e policy");
  if (provenance.evaluation !=
          CollisionEvaluation::TiledCartesianReference &&
      provenance.evaluation != CollisionEvaluation::FactoredExactTotals)
    return ValidationResult::failure(
        "collision result provenance has an invalid evaluation algorithm");
  if (!valid_nonnegative(provenance.high_ecm_threshold) ||
      !std::isfinite(provenance.electron_mass) ||
      !(provenance.electron_mass > 0.0))
    return ValidationResult::failure(
        "collision result provenance has invalid physical constants/cuts");
  for (std::size_t i = 0; i < charge_class.weight.size(); ++i) {
    if (!valid_nonnegative(charge_class.weight[i]))
      return ValidationResult::failure(
          "charge-class lane is negative or non-finite", i);
  }
  const Scalar ee_scale = std::max<Scalar>(1.0, ee_weight);
  if (!valid_nonnegative(ee_weight) ||
      !valid_nonnegative(ee_high_weight) ||
      ee_high_weight >
          ee_weight + positive_sum_roundoff_bound(ee_scale, ee_pair_count) ||
      !valid_nonnegative(ecm_weighted_sum) ||
      !valid_nonnegative(ecm_squared_weighted_sum))
    return ValidationResult::failure("e-e/ECM lane invariant is invalid");
  if (ee_pair_count > cartesian_pair_count)
    return ValidationResult::failure(
        "e-e pair count exceeds Cartesian pair count");

  const WeightedMomentLane& m = w_spectrum.moments;
  if (!valid_nonnegative(m.weight_sum) ||
      !valid_nonnegative(m.weight_squared_sum) ||
      !valid_nonnegative(m.weighted_value_sum) ||
      !valid_nonnegative(m.weighted_value_squared_sum) ||
      !valid_nonnegative(w_spectrum.underflow_weight) ||
      !valid_nonnegative(w_spectrum.overflow_weight))
    return ValidationResult::failure("W spectrum moment is invalid");
  const Scalar weight_scale =
      std::max<Scalar>(1.0, std::max(m.weight_sum, ee_weight));
  if (std::fabs(m.weight_sum - ee_weight) >
      positive_sum_roundoff_bound(weight_scale, ee_pair_count))
    return ValidationResult::failure(
        "W spectrum weight and e-e luminosity disagree");

  if (w_spectrum.configured()) {
    if (!std::isfinite(w_spectrum.lower) ||
        !std::isfinite(w_spectrum.upper) ||
        !std::isfinite(w_spectrum.bin_width) ||
        !(w_spectrum.upper > w_spectrum.lower) ||
        !(w_spectrum.bin_width > 0.0))
      return ValidationResult::failure("W spectrum shape is invalid");
    for (std::size_t i = 0; i < w_spectrum.bin_weight.size(); ++i) {
      if (!valid_nonnegative(w_spectrum.bin_weight[i]))
        return ValidationResult::failure(
            "W spectrum bin is negative or non-finite", i);
    }
    long double classified =
        static_cast<long double>(w_spectrum.underflow_weight) +
        static_cast<long double>(w_spectrum.overflow_weight);
    for (Scalar bin : w_spectrum.bin_weight)
      classified += static_cast<long double>(bin);
    if (!std::isfinite(classified) ||
        std::fabs(classified - static_cast<long double>(m.weight_sum)) >
            static_cast<long double>(positive_sum_roundoff_bound(
                weight_scale, ee_pair_count)))
      return ValidationResult::failure(
          "W spectrum classified weight and moment weight disagree");
  } else if (w_spectrum.lower != 0.0 || w_spectrum.upper != 0.0 ||
             w_spectrum.bin_width != 0.0 ||
             w_spectrum.underflow_weight != 0.0 ||
             w_spectrum.overflow_weight != 0.0) {
    return ValidationResult::failure(
        "disabled W spectrum retains non-zero shape or classified flow");
  }
  return ValidationResult::success();
}

void CollisionResultLane::merge_from(const CollisionResultLane& source) {
  const ValidationResult left = validate();
  const ValidationResult right = source.validate();
  if (!left)
    throw std::invalid_argument("destination result lane is invalid: " +
                                left.message);
  if (!right)
    throw std::invalid_argument("source result lane is invalid: " +
                                right.message);
  if (!same_spectrum_shape(w_spectrum, source.w_spectrum))
    throw std::invalid_argument("cannot merge different W spectrum shapes");
  if (provenance != source.provenance)
    throw std::invalid_argument(
        "cannot merge collision lanes with different physics provenance");

  // Merge transactionally.  Overflow, allocation, or invariant failure must
  // not leave the destination lane partially published.
  CollisionResultLane candidate = *this;

  for (std::size_t i = 0; i < candidate.charge_class.weight.size(); ++i)
    candidate.charge_class.weight[i] = checked_sum(
        candidate.charge_class.weight[i], source.charge_class.weight[i],
        "merged charge-class luminosity");
  candidate.ee_weight = checked_sum(candidate.ee_weight, source.ee_weight,
                                    "merged e-e luminosity");
  candidate.ee_high_weight = checked_sum(
      candidate.ee_high_weight, source.ee_high_weight,
      "merged high-ECM e-e luminosity");
  candidate.ecm_weighted_sum = checked_sum(
      candidate.ecm_weighted_sum, source.ecm_weighted_sum,
      "merged ECM first moment");
  candidate.ecm_squared_weighted_sum = checked_sum(
      candidate.ecm_squared_weighted_sum,
      source.ecm_squared_weighted_sum,
      "merged ECM second moment");
  checked_count_add(source.cartesian_pair_count,
                    candidate.cartesian_pair_count,
                     "merged Cartesian pair");
  checked_count_add(source.ee_pair_count, candidate.ee_pair_count,
                     "merged e-e pair");

  candidate.w_spectrum.underflow_weight = checked_sum(
      candidate.w_spectrum.underflow_weight,
      source.w_spectrum.underflow_weight,
      "merged W underflow");
  candidate.w_spectrum.overflow_weight = checked_sum(
      candidate.w_spectrum.overflow_weight,
      source.w_spectrum.overflow_weight,
      "merged W overflow");
  for (std::size_t i = 0; i < candidate.w_spectrum.bin_weight.size(); ++i)
    candidate.w_spectrum.bin_weight[i] = checked_sum(
        candidate.w_spectrum.bin_weight[i],
        source.w_spectrum.bin_weight[i],
        "merged W bin");
  WeightedMomentLane& a = candidate.w_spectrum.moments;
  const WeightedMomentLane& b = source.w_spectrum.moments;
  a.weight_sum = checked_sum(a.weight_sum, b.weight_sum,
                             "merged W moment weight");
  a.weight_squared_sum = checked_sum(
      a.weight_squared_sum, b.weight_squared_sum,
      "merged W moment weight-squared");
  a.weighted_value_sum = checked_sum(
      a.weighted_value_sum, b.weighted_value_sum,
      "merged W first moment");
  a.weighted_value_squared_sum = checked_sum(
      a.weighted_value_squared_sum, b.weighted_value_squared_sum,
      "merged W second moment");

  const ValidationResult merged = candidate.validate();
  if (!merged)
    throw std::logic_error("merged result lane violates invariant: " +
                           merged.message);
  *this = std::move(candidate);
}

CollisionResultLane evaluate_collision_luminosity(
    const BeamSoA& beam1, const CellCSR& cells1,
    const BeamSoA& beam2, const CellCSR& cells2,
    const ModernCollisionConfig& config,
    CollisionEvaluation evaluation) {
  const ValidationResult valid_beam1 = beam1.validate();
  const ValidationResult valid_beam2 = beam2.validate();
  const ValidationResult valid_cells1 = cells1.validate(beam1);
  const ValidationResult valid_cells2 = cells2.validate(beam2);
  if (!valid_beam1)
    throw std::invalid_argument("invalid beam 1: " + valid_beam1.message);
  if (!valid_beam2)
    throw std::invalid_argument("invalid beam 2: " + valid_beam2.message);
  if (!valid_cells1)
    throw std::invalid_argument("invalid beam-1 CellCSR: " +
                                valid_cells1.message);
  if (!valid_cells2)
    throw std::invalid_argument("invalid beam-2 CellCSR: " +
                                valid_cells2.message);
  validate_grid_pair(cells1, cells2);

  if (config.ee_policy == EePairPolicy::Unspecified)
    throw std::invalid_argument(
        "modern collision requires an explicit e-e pair policy");
  if (config.ee_policy != EePairPolicy::AllElectronFamily &&
      config.ee_policy != EePairPolicy::OppositeElectricCharge &&
      config.ee_policy != EePairPolicy::SameElectricCharge)
    throw std::invalid_argument("unknown e-e pair policy");
  if (evaluation != CollisionEvaluation::TiledCartesianReference &&
      evaluation != CollisionEvaluation::FactoredExactTotals)
    throw std::invalid_argument("unknown collision evaluation algorithm");
  if (!std::isfinite(config.high_ecm_threshold) ||
      config.high_ecm_threshold < 0.0)
    throw std::invalid_argument(
        "high-ECM threshold must be finite and non-negative");
  if (!std::isfinite(config.electron_mass) ||
      !(config.electron_mass > 0.0))
    throw std::invalid_argument(
        "electron rest mass must be finite and positive");
  if (config.tile_width == 0)
    throw std::invalid_argument("collision tile width must be non-zero");
  if (!fits_size_t(config.tile_width))
    throw std::length_error("collision tile width is not representable");

  CollisionResultLane lane;
  lane.provenance.schema_version = kModernCollisionResultSchemaVersion;
  lane.provenance.ee_policy = config.ee_policy;
  lane.provenance.evaluation = evaluation;
  lane.provenance.high_ecm_threshold = config.high_ecm_threshold;
  lane.provenance.electron_mass = config.electron_mass;
  lane.w_spectrum = make_spectrum_lane(config.w_spectrum);
  const std::vector<ParticleKinematics> kinematics1 =
      prepare_kinematics(beam1, config.electron_mass, "beam 1");
  const std::vector<ParticleKinematics> kinematics2 =
      prepare_kinematics(beam2, config.electron_mass, "beam 2");

  // Preflight the full Cartesian count before mutating result state.
  for (CellKey cell = 0; cell < cells1.grid().cell_count(); ++cell) {
    const CellRange range1 = cells1.cell_range(cell);
    const CellRange range2 = cells2.cell_range(cell);
    checked_count_product_add(range1.size(), range2.size(),
                              lane.cartesian_pair_count,
                              "Cartesian collision pair");
  }

  const bool factored =
      evaluation == CollisionEvaluation::FactoredExactTotals;
  if (factored) {
    for (CellKey cell = 0; cell < cells1.grid().cell_count(); ++cell) {
      const CellSufficientStatistics a =
          cell_statistics(beam1, cells1, kinematics1, cell);
      const CellSufficientStatistics b =
          cell_statistics(beam2, cells2, kinematics2, cell);
      factor_charge_matrix(lane, a, b);
      factor_ee_category(lane, a, b, config.ee_policy);
    }
  }

  const std::size_t tile_width =
      static_cast<std::size_t>(config.tile_width);
  for (CellKey cell = 0; cell < cells1.grid().cell_count(); ++cell) {
    const CellRange range1 = cells1.cell_range(cell);
    const CellRange range2 = cells2.cell_range(cell);
    for (ArrayOffset position1 = range1.begin;
         position1 < range1.end; ++position1) {
      const ParticleIndex particle1 =
          cells1.particle_indices()[static_cast<std::size_t>(position1)];
      const std::size_t p1 = static_cast<std::size_t>(particle1);

      // Tiling only the inner stable-ID sequence retains one canonical pair
      // order for every tile width: cell, stable ID 1, stable ID 2.
      ArrayOffset tile_begin = range2.begin;
      while (tile_begin < range2.end) {
        const ArrayOffset remaining = range2.end - tile_begin;
        const ArrayOffset span = std::min<ArrayOffset>(
            remaining, static_cast<ArrayOffset>(tile_width));
        const ArrayOffset tile_end = tile_begin + span;
        for (ArrayOffset position2 = tile_begin;
             position2 < tile_end; ++position2) {
          const ParticleIndex particle2 =
              cells2.particle_indices()[static_cast<std::size_t>(position2)];
          const std::size_t p2 = static_cast<std::size_t>(particle2);
          const std::int8_t charge1 = beam1.charge_e()[p1];
          const std::int8_t charge2 = beam2.charge_e()[p2];
          const Scalar weight = checked_product(
              beam1.macro_weight()[p1], beam2.macro_weight()[p2],
              "Cartesian pair weight");

          if (!factored) {
            const std::size_t matrix_index =
                charge_index(charge1) * kChargeClassCount +
                charge_index(charge2);
            lane.charge_class.weight[matrix_index] = checked_sum(
                lane.charge_class.weight[matrix_index], weight,
                "Cartesian charge-class luminosity");
          }

          if (!selected_pair(config.ee_policy, beam1.species()[p1], charge1,
                             beam2.species()[p2], charge2))
            continue;
          if (!factored) {
            lane.ee_weight = checked_sum(
                lane.ee_weight, weight, "Cartesian e-e luminosity");
            checked_count_add(1, lane.ee_pair_count,
                              "Cartesian e-e pair");
          }
          const PairKinematics pair = pair_kinematics(
              kinematics1[p1], kinematics2[p2], config.electron_mass);
          add_pair_observables(lane, weight, pair,
                               config.high_ecm_threshold, !factored);
        }
        tile_begin = tile_end;
      }
    }
  }

  const ValidationResult valid_lane = lane.validate();
  if (!valid_lane)
    throw std::logic_error("modern collision produced invalid result lane: " +
                           valid_lane.message);
  return lane;
}

}  // namespace modern
}  // namespace feebl
