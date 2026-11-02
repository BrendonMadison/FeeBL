#ifndef FEEBL_MODERN_COLLISION_CPP_H
#define FEEBL_MODERN_COLLISION_CPP_H

#include "beamSoACPP.h"
#include "cellCSRCPP.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace feebl {
namespace modern {

// FeeBL's historical value from physconst.h, in GeV.  It is repeated here so
// the replacement core remains independent of the legacy macro namespace.
constexpr Scalar kFeeBLElectronMassGeV = 0.51099906e-3;

// The legacy GRID::collide_ee predicate is signed-energy provenance, not an
// electric-charge test: it accepts energy1*energy2 > 0.  Generated primary
// particles have positive stored energy, so a primary-only legacy call accepts
// every electron-family Cartesian pair, irrespective of charge_sign.
//
// BeamSoA deliberately stores non-negative energy and explicit physical
// species/charge.  Consequently the caller must state which category it is
// replacing; the pure kernel never silently equates the legacy sign bit with
// physical electric charge.
enum class EePairPolicy : std::uint8_t {
  Unspecified = 0,
  // Primary-only equivalent of the legacy positive-energy call: both records
  // must be Electron or Positron, with no physical-charge restriction.
  AllElectronFamily = 1,
  // Physical e+e- category.
  OppositeElectricCharge = 2,
  // Physical e-e- or e+e+ category.
  SameElectricCharge = 3
};

enum class CollisionEvaluation : std::uint8_t {
  // Canonical cell, beam-1 stable-ID, beam-2 stable-ID pair enumeration.
  TiledCartesianReference = 1,
  // Algebraically separable charge/e-e totals and the ECM second moment are
  // factored. Threshold, ECM first moment and W remain exact Cartesian.
  FactoredExactTotals = 2
};

struct InvariantMassSpectrumConfig {
  Scalar lower = 0.0;
  Scalar upper = 0.0;
  std::size_t bins = 0;

  bool enabled() const noexcept { return bins != 0; }
};

struct ModernCollisionConfig {
  EePairPolicy ee_policy = EePairPolicy::Unspecified;
  Scalar high_ecm_threshold = 0.0;
  Scalar electron_mass = kFeeBLElectronMassGeV;
  InvariantMassSpectrumConfig w_spectrum;
  std::uint64_t tile_width = 64;
};

// Weighted W sufficient statistics.  weight_squared_sum gives a Kish
// effective sample size without retaining pair records.
struct WeightedMomentLane {
  Scalar weight_sum = 0.0;
  Scalar weight_squared_sum = 0.0;
  Scalar weighted_value_sum = 0.0;
  Scalar weighted_value_squared_sum = 0.0;

  Scalar mean() const noexcept;
  Scalar variance() const noexcept;
  Scalar effective_count() const noexcept;
};

struct InvariantMassSpectrumLane {
  Scalar lower = 0.0;
  Scalar upper = 0.0;
  Scalar bin_width = 0.0;
  std::vector<Scalar> bin_weight;
  Scalar underflow_weight = 0.0;
  Scalar overflow_weight = 0.0;
  WeightedMomentLane moments;

  bool configured() const noexcept { return !bin_weight.empty(); }
};

// Rows are beam 1, columns beam 2.  The fixed class order is
// {-1, 0, +1}; index = (charge_e + 1).  Every same-cell Cartesian pair
// contributes, including a neutral class if the caller supplied one.
struct ChargeClassLuminosityLane {
  std::array<Scalar, 9> weight{};

  Scalar at(std::int8_t beam1_charge, std::int8_t beam2_charge) const;
};

constexpr std::uint32_t kModernCollisionResultSchemaVersion = 1;

// Exact provenance is carried with every task-local lane so deterministic
// reduction cannot silently combine different physical selections or models.
// A future production adapter additionally records the enclosing PhysicsPlan
// hash in its run manifest.
struct CollisionResultProvenance {
  std::uint32_t schema_version = 0;
  EePairPolicy ee_policy = EePairPolicy::Unspecified;
  CollisionEvaluation evaluation =
      CollisionEvaluation::TiledCartesianReference;
  Scalar high_ecm_threshold = 0.0;
  Scalar electron_mass = 0.0;

  bool operator==(const CollisionResultProvenance& other) const noexcept;
  bool operator!=(const CollisionResultProvenance& other) const noexcept {
    return !(*this == other);
  }
};

// A task-local, legacy-global-free reduction lane.  merge_from is deterministic
// when the caller merges lanes in canonical encounter order.  All counts use
// checked uint64 arithmetic and every floating update is checked for finiteness.
struct CollisionResultLane {
  CollisionResultProvenance provenance;
  ChargeClassLuminosityLane charge_class;
  Scalar ee_weight = 0.0;
  Scalar ee_high_weight = 0.0;
  Scalar ecm_weighted_sum = 0.0;
  Scalar ecm_squared_weighted_sum = 0.0;
  InvariantMassSpectrumLane w_spectrum;
  std::uint64_t cartesian_pair_count = 0;
  std::uint64_t ee_pair_count = 0;

  ValidationResult validate() const;
  void merge_from(const CollisionResultLane& source);
};

// Evaluate pairs occupying the same cell in two canonical CellCSR objects.
// This standalone kernel is not selected by the production dispatcher.
CollisionResultLane evaluate_collision_luminosity(
    const BeamSoA& beam1, const CellCSR& cells1,
    const BeamSoA& beam2, const CellCSR& cells2,
    const ModernCollisionConfig& config,
    CollisionEvaluation evaluation);

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_COLLISION_CPP_H
