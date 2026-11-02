#ifndef FEEBL_MODERN_LEGACY_ADAPTER_CPP_H
#define FEEBL_MODERN_LEGACY_ADAPTER_CPP_H

#include "beamSoACPP.h"
#include "particleBeamCPP.h"

#include <cstdint>
#include <limits>

namespace feebl {
namespace modern {

// This is the only normalization represented by the first legacy boundary:
// every stored primary carries the same explicit number of physical particles.
// `declared_macro_particle_count` is the denominator used by the legacy grid;
// it is deliberately not inferred from the number that happened to survive a
// longitudinal acceptance cut.
enum class LegacyMacroNormalization : std::uint8_t {
  UniformPhysicalParticlesPerMacro = 1
};

struct LegacyBeamImportContract {
  UnitConvention units = UnitConvention::LegacyFeeBLNmGeV;
  LegacyMacroNormalization normalization =
      LegacyMacroNormalization::UniformPhysicalParticlesPerMacro;

  // Physical bunch population and its legacy normalization denominator.
  // The contract requires
  //   bunch_population ~= macro_weight * declared_macro_particle_count
  // at legacy-float precision.
  Scalar bunch_population = 0.0;
  Scalar macro_weight = 0.0;
  ParticleIndex declared_macro_particle_count = 0;

  // Provenance and kinematics are explicit at the boundary.  FeeBL's native
  // convention is beam 1 -> +z and beam 2 -> -z.
  std::uint32_t beam_id = 0;
  std::int8_t direction = 0;
  ParticleSpecies species = ParticleSpecies::Other;
  std::int8_t charge_e = 0;

  // A caller-controlled allocation/overflow guard.  The adapter refuses before
  // changing the BeamSoA if target.size()+imported_count exceeds this bound.
  ParticleIndex maximum_total_particles =
      std::numeric_limits<ParticleIndex>::max();
};

// Receipt describing exactly where one append landed.  It allows a later
// read-only audit without assuming the target began empty.
struct LegacyBeamImportSpan {
  SliceIndex source_first_slice = 0;
  SliceIndex target_first_slice = 0;
  SliceIndex slice_count = 0;
  ParticleIndex target_particle_begin = 0;
  ParticleIndex target_particle_end = 0;
};

// Append [source_first_slice, source_last_slice) to the next unfilled target
// slices.  All legacy data and identities are staged and validated first; a
// validation failure therefore leaves the target's logical contents unchanged.
// There is no production dispatch in this interface.
LegacyBeamImportSpan append_legacy_particle_beam_slices(
    const PARTICLE_BEAM& source,
    SliceIndex source_first_slice,
    SliceIndex source_last_slice,
    const LegacyBeamImportContract& contract,
    BeamSoA& target);

// Compare a previously imported span against the live legacy objects without
// mutating either representation.  Exact equality means float legacy values
// widened to Scalar compare exactly; spin components, already double in the
// legacy object, compare bit-for-value as doubles.
ValidationResult compare_legacy_particle_beam_import(
    const PARTICLE_BEAM& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    const BeamSoA& imported);

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_LEGACY_ADAPTER_CPP_H
