#ifndef FEEBL_MODERN_PHOTON_LUMI_CPP_H
#define FEEBL_MODERN_PHOTON_LUMI_CPP_H

#include "beamSoACPP.h"
#include "cellCSRCPP.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace feebl {
namespace modern {

// Explicit replacement for the historical 3x3 luminosity classes.  The
// legacy implementation inferred lepton class from the sign of energy;
// modern state keeps energy non-negative and uses physical species/charge.
enum class LuminositySpeciesClass : std::uint8_t {
  Electron = 0,
  Positron = 1,
  Photon = 2
};

enum class PhotonLumiAlgorithm : std::uint8_t {
  CartesianReference = 1,
  FactoredExact = 2
};

struct PhotonLuminosityResult {
  std::array<double, 9> class_luminosity = {{0.0}};
  double lumi_ge = 0.0;
  double lumi_eg = 0.0;
  double lumi_gg = 0.0;
  double lumi_gg_high = 0.0;

  // Logical pair opportunities are invariant across implementations even
  // when FactoredExact deliberately avoids visiting the Cartesian product.
  std::uint64_t ge_pair_opportunities = 0;
  std::uint64_t eg_pair_opportunities = 0;
  std::uint64_t gg_pair_opportunities = 0;
  std::uint64_t cartesian_pairs_visited = 0;

  double class_value(LuminositySpeciesClass first,
                     LuminositySpeciesClass second) const noexcept;
  void add_class(LuminositySpeciesClass first,
                 LuminositySpeciesClass second, double value) noexcept;
};

struct PhotonLumiRequest {
  // Legacy GRID::collide_gg applies the strict comparison 4 E1 E2 > gg_smin.
  double gg_smin = 0.0;
  PhotonLumiAlgorithm algorithm = PhotonLumiAlgorithm::CartesianReference;
};

// Exact luminosity-only kernel for the feature family in which heaps, jets,
// Compton, pair/muon/hadron production and other per-pair side effects are
// disabled.  It is intentionally not a generic fallback: callers must select
// this pure family through PhysicsPlan capability validation.
//
// beam1_particles/beam2_particles contain charged Electron/Positron state;
// beam1_photons/beam2_photons contain only Photon state.  Every CSR must have
// been built from its corresponding owner on an identical grid.  Zero-energy
// photons are retained in logical opportunity counts, but—as in the legacy
// ge/eg functions—contribute no luminosity.
PhotonLuminosityResult compute_photon_luminosity_only(
    const BeamSoA& beam1_particles,
    const CellCSR& beam1_particle_cells,
    const BeamSoA& beam2_particles,
    const CellCSR& beam2_particle_cells,
    const BeamSoA& beam1_photons,
    const CellCSR& beam1_photon_cells,
    const BeamSoA& beam2_photons,
    const CellCSR& beam2_photon_cells,
    const PhotonLumiRequest& request);

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_PHOTON_LUMI_CPP_H
