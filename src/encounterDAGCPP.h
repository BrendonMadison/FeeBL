#ifndef FEEBL_ENCOUNTER_DAG_CPP_H
#define FEEBL_ENCOUNTER_DAG_CPP_H

#include "modernTypesCPP.h"

#include <array>
#include <cstdint>
#include <vector>

namespace feebl {
namespace modern {

struct EncounterNode {
  EncounterId id = kInvalidEncounter;
  SliceIndex beam1_slice = 0;
  SliceIndex beam2_slice = 0;
  WaveIndex diagonal = 0;
  SubstepIndex substep = 0;
  WaveIndex wave = 0;
  std::array<EncounterId, 2> dependencies = {
      {kInvalidEncounter, kInvalidEncounter}};
  std::uint8_t dependency_count = 0;
};

struct EncounterWave {
  WaveIndex index = 0;
  WaveIndex diagonal = 0;
  SubstepIndex substep = 0;
  std::vector<EncounterId> encounters;
};

// Dependency graph for the present FeeBL head-on/current-geometry encounter
// semantics.  Every pair (i,j) appears exactly once.  It depends on (i-1,j)
// and (i,j-1), where present, because either predecessor mutates one of the
// same primary slices.  Each legacy diagonal has n_t sequential substep waves:
// W_(d,t)={(i,j,t):i+j=d}.  A t>0 node depends on its own t-1 node; a t=0 node
// depends on the last substep of its prior same-slice encounters.  Tasks within
// a wave have exclusive ownership of both slices.  Unequal slice counts and
// arbitrary positive n_t are supported explicitly.
//
// Arbitrary physical encounter-time geometry will use the same node/wave
// representation but a different builder; it must not assume this shortcut.
class EncounterDAG {
 public:
  EncounterDAG(SliceIndex beam1_slices, SliceIndex beam2_slices,
               SubstepIndex substeps = 1);
  static EncounterDAG build_legacy(SliceIndex beam1_slices,
                                   SliceIndex beam2_slices,
                                   SubstepIndex substeps) {
    return EncounterDAG(beam1_slices, beam2_slices, substeps);
  }

  SliceIndex beam1_slice_count() const noexcept { return beam1_slices_; }
  SliceIndex beam2_slice_count() const noexcept { return beam2_slices_; }
  SubstepIndex substep_count() const noexcept { return substeps_; }
  EncounterId encounter_count() const noexcept {
    return static_cast<EncounterId>(nodes_.size());
  }
  WaveIndex wave_count() const noexcept {
    return static_cast<WaveIndex>(waves_.size());
  }

  const std::vector<EncounterNode>& nodes() const noexcept { return nodes_; }
  const std::vector<EncounterWave>& waves() const noexcept { return waves_; }
  const EncounterNode& node(EncounterId id) const;
  EncounterId node_id(SliceIndex beam1_slice,
                      SliceIndex beam2_slice,
                      SubstepIndex substep = 0) const;

  bool is_legacy_antidiagonal() const noexcept { return true; }
  ValidationResult validate() const;

 private:
  std::size_t pair_index(SliceIndex beam1_slice,
                         SliceIndex beam2_slice,
                         SubstepIndex substep) const;

  SliceIndex beam1_slices_;
  SliceIndex beam2_slices_;
  SubstepIndex substeps_;
  std::vector<EncounterNode> nodes_;      // contiguous, topological ID order
  std::vector<EncounterWave> waves_;
  std::vector<EncounterId> pair_to_node_; // row-major lookup by (i,j,t)
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_ENCOUNTER_DAG_CPP_H
