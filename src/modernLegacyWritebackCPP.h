#ifndef FEEBL_MODERN_LEGACY_WRITEBACK_CPP_H
#define FEEBL_MODERN_LEGACY_WRITEBACK_CPP_H

#include "modernLegacyAdapterCPP.h"

#include <cstdint>

namespace feebl {
namespace modern {

// Receipt for the only mutable part of the first modern/legacy bridge.  The
// resident primary engine may change x/y/vx/vy/energy; every other legacy
// property remains authoritative and is proved unchanged by this operation.
struct LegacyBeamWritebackReceipt {
  LegacyBeamImportSpan span;
  ParticleIndex particle_count = 0;
  bool target_xy_was_adopted = false;
  bool slice_shape_and_order_preserved = false;
  bool stable_ids_preserved = false;
  bool z_and_spin_preserved = false;
  bool upsilon_matches_beam_soa = false;
  bool adopted_xy_slot_addresses_preserved = false;
  bool phase_space_matches_beam_soa = false;
  std::uint64_t invariant_fingerprint_before = 0;
  std::uint64_t invariant_fingerprint_after = 0;
};

// Read-only compatibility gate.  A selector can call this before launching
// CUDA to validate and discard a complete staged write-back, proving that the
// accepted mapping is fail-safe and transactional before device work starts.
// The later production write-back creates its own stage and can still report
// an allocation failure without mutating the targets.  The accepted mapping is
// exactly the one described by the import contract/receipt;
// loaded/generated provenance is preserved verbatim.
ValidationResult preflight_legacy_particle_beam_writeback(
    const BeamSoA& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    const PARTICLE_BEAM& target);

// Validate and stage every output before mutation, then update the existing
// PARTICLE objects in place.  setParticle writes x/y through their already
// adopted SoA slots, so neither particle pointers nor adopted slot addresses
// change.  Upsilon is committed too because every legacy field push updates
// it; z, spin, stable IDs and slice order are checked before and after.
// Validation/allocation failures leave target untouched; unexpected
// commit failures restore every staged original value before throwing.
LegacyBeamWritebackReceipt write_back_legacy_particle_beam_phase_space(
    const BeamSoA& source,
    const LegacyBeamImportContract& contract,
    const LegacyBeamImportSpan& span,
    PARTICLE_BEAM& target);

// Crossing-level receipt and all-or-nothing API.  Both beams are fully staged
// and cross-checked before either target changes.  Any commit or post-commit
// failure restores x/y/vx/vy/energy/Upsilon for both beams.
struct LegacyCrossingWritebackReceipt {
  LegacyBeamWritebackReceipt beam1;
  LegacyBeamWritebackReceipt beam2;
  bool distinct_particle_graphs = false;
  bool crossing_transaction_committed = false;
};

ValidationResult preflight_legacy_crossing_writeback(
    const BeamSoA& source1,
    const LegacyBeamImportContract& contract1,
    const LegacyBeamImportSpan& span1,
    const PARTICLE_BEAM& target1,
    const BeamSoA& source2,
    const LegacyBeamImportContract& contract2,
    const LegacyBeamImportSpan& span2,
    const PARTICLE_BEAM& target2);

LegacyCrossingWritebackReceipt write_back_legacy_crossing_phase_space(
    const BeamSoA& source1,
    const LegacyBeamImportContract& contract1,
    const LegacyBeamImportSpan& span1,
    PARTICLE_BEAM& target1,
    const BeamSoA& source2,
    const LegacyBeamImportContract& contract2,
    const LegacyBeamImportSpan& span2,
    PARTICLE_BEAM& target2);

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_LEGACY_WRITEBACK_CPP_H
