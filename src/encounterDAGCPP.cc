#include "encounterDAGCPP.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace feebl {
namespace modern {

EncounterDAG::EncounterDAG(SliceIndex beam1_slices,
                           SliceIndex beam2_slices,
                           SubstepIndex substeps)
    : beam1_slices_(beam1_slices),
      beam2_slices_(beam2_slices),
      substeps_(substeps) {
  if (substeps_ == 0)
    throw std::invalid_argument("EncounterDAG n_t/substep count must be positive");

  std::uint64_t pair_count = 0;
  std::uint64_t total = 0;
  if (!checked_multiply(beam1_slices_, beam2_slices_, pair_count) ||
      !checked_multiply(pair_count, substeps_, total))
    throw std::overflow_error("EncounterDAG encounter-count multiplication overflow");
  if (!fits_size_t(total) || total > nodes_.max_size() ||
      total > pair_to_node_.max_size())
    throw std::length_error("EncounterDAG encounter arrays are not representable");

  if (beam1_slices_ == 0 || beam2_slices_ == 0) return;

  std::uint64_t sum = 0;
  if (!checked_add(beam1_slices_, beam2_slices_, sum))
    throw std::overflow_error("EncounterDAG diagonal-count addition overflow");
  const WaveIndex diagonal_total = sum - 1;
  std::uint64_t wave_total = 0;
  if (!checked_multiply(diagonal_total, substeps_, wave_total))
    throw std::overflow_error("EncounterDAG wave-count multiplication overflow");
  if (!fits_size_t(wave_total) || wave_total > waves_.max_size())
    throw std::length_error("EncounterDAG wave array is not representable");

  nodes_.reserve(static_cast<std::size_t>(total));
  pair_to_node_.assign(static_cast<std::size_t>(total), kInvalidEncounter);
  waves_.resize(static_cast<std::size_t>(wave_total));

  for (WaveIndex diagonal = 0; diagonal < diagonal_total; ++diagonal) {
    const SliceIndex first_i = diagonal >= beam2_slices_ - 1
                                   ? diagonal - (beam2_slices_ - 1)
                                   : 0;
    const SliceIndex last_i =
        std::min<SliceIndex>(diagonal, beam1_slices_ - 1);
    const std::size_t width =
        static_cast<std::size_t>(last_i - first_i + 1);

    // This nesting is the legacy execution order: diagonal, then n_t, then
    // increasing beam-1 slice/decreasing beam-2 slice.
    for (SubstepIndex substep = 0; substep < substeps_; ++substep) {
      const WaveIndex wave = diagonal * substeps_ + substep;
      EncounterWave& wave_record = waves_[static_cast<std::size_t>(wave)];
      wave_record.index = wave;
      wave_record.diagonal = diagonal;
      wave_record.substep = substep;
      wave_record.encounters.reserve(width);

      for (SliceIndex i = first_i; i <= last_i; ++i) {
        const SliceIndex j = diagonal - i;
        EncounterNode node;
        node.id = static_cast<EncounterId>(nodes_.size());
        node.beam1_slice = i;
        node.beam2_slice = j;
        node.diagonal = diagonal;
        node.substep = substep;
        node.wave = wave;

        if (substep > 0) {
          // Same two slices, immediately preceding integration substep.
          node.dependencies[node.dependency_count++] =
              pair_to_node_[pair_index(i, j, substep - 1)];
        } else {
          // First substep of a new diagonal: both slice histories must have
          // finished the last substep of their respective prior encounters.
          if (i > 0)
            node.dependencies[node.dependency_count++] =
                pair_to_node_[pair_index(i - 1, j, substeps_ - 1)];
          if (j > 0)
            node.dependencies[node.dependency_count++] =
                pair_to_node_[pair_index(i, j - 1, substeps_ - 1)];
        }

        nodes_.push_back(node);
        pair_to_node_[pair_index(i, j, substep)] = node.id;
        wave_record.encounters.push_back(node.id);
      }
    }
  }
}

std::size_t EncounterDAG::pair_index(SliceIndex beam1_slice,
                                     SliceIndex beam2_slice,
                                     SubstepIndex substep) const {
  // The constructor proved the full rectangular product representable.
  return static_cast<std::size_t>(
      (beam1_slice * beam2_slices_ + beam2_slice) * substeps_ + substep);
}

const EncounterNode& EncounterDAG::node(EncounterId id) const {
  if (id >= nodes_.size())
    throw std::out_of_range("EncounterDAG node ID out of range");
  return nodes_[static_cast<std::size_t>(id)];
}

EncounterId EncounterDAG::node_id(SliceIndex beam1_slice,
                                  SliceIndex beam2_slice,
                                  SubstepIndex substep) const {
  if (beam1_slice >= beam1_slices_ || beam2_slice >= beam2_slices_ ||
      substep >= substeps_)
    throw std::out_of_range("EncounterDAG (slice pair, substep) out of range");
  return pair_to_node_[pair_index(beam1_slice, beam2_slice, substep)];
}

ValidationResult EncounterDAG::validate() const {
  if (substeps_ == 0)
    return ValidationResult::failure("n_t/substep count is zero");

  std::uint64_t expected_pairs = 0;
  std::uint64_t expected_nodes = 0;
  if (!checked_multiply(beam1_slices_, beam2_slices_, expected_pairs) ||
      !checked_multiply(expected_pairs, substeps_, expected_nodes))
    return ValidationResult::failure("encounter-count multiplication overflow");
  if (nodes_.size() != expected_nodes ||
      pair_to_node_.size() != expected_nodes)
    return ValidationResult::failure("(slice pair,substep) prism is not exhaustive");

  WaveIndex expected_diagonals = 0;
  if (beam1_slices_ != 0 && beam2_slices_ != 0) {
    std::uint64_t sum = 0;
    if (!checked_add(beam1_slices_, beam2_slices_, sum))
      return ValidationResult::failure("diagonal-count addition overflow");
    expected_diagonals = sum - 1;
  }
  std::uint64_t expected_waves = 0;
  if (!checked_multiply(expected_diagonals, substeps_, expected_waves))
    return ValidationResult::failure("wave-count multiplication overflow");
  if (waves_.size() != expected_waves)
    return ValidationResult::failure("topological wave count is incorrect");

  if (expected_nodes == 0) return ValidationResult::success();

  std::vector<std::uint8_t> membership(nodes_.size(), 0);
  std::vector<WaveIndex> beam1_owner(
      static_cast<std::size_t>(beam1_slices_),
      std::numeric_limits<WaveIndex>::max());
  std::vector<WaveIndex> beam2_owner(
      static_cast<std::size_t>(beam2_slices_),
      std::numeric_limits<WaveIndex>::max());

  for (WaveIndex diagonal = 0; diagonal < expected_diagonals; ++diagonal) {
    const SliceIndex first_i = diagonal >= beam2_slices_ - 1
                                   ? diagonal - (beam2_slices_ - 1)
                                   : 0;
    const SliceIndex last_i =
        std::min<SliceIndex>(diagonal, beam1_slices_ - 1);
    const std::uint64_t expected_width = last_i - first_i + 1;

    for (SubstepIndex substep = 0; substep < substeps_; ++substep) {
      const WaveIndex wave = diagonal * substeps_ + substep;
      const EncounterWave& record = waves_[static_cast<std::size_t>(wave)];
      if (record.index != wave || record.diagonal != diagonal ||
          record.substep != substep)
        return ValidationResult::failure("wave identity is not canonical", wave);
      if (record.encounters.size() != expected_width)
        return ValidationResult::failure("anti-diagonal/substep wave width is incorrect",
                                         wave);

      for (std::size_t position = 0; position < record.encounters.size();
           ++position) {
        const EncounterId id = record.encounters[position];
        if (id >= nodes_.size())
          return ValidationResult::failure("wave contains out-of-range node ID", id);
        if (membership[static_cast<std::size_t>(id)]++)
          return ValidationResult::failure("node occurs in more than one wave", id);
        const EncounterNode& current = nodes_[static_cast<std::size_t>(id)];
        const SliceIndex expected_i = first_i + position;
        const SliceIndex expected_j = diagonal - expected_i;
        if (current.wave != wave || current.diagonal != diagonal ||
            current.substep != substep ||
            current.beam1_slice != expected_i ||
            current.beam2_slice != expected_j)
          return ValidationResult::failure(
              "wave is not canonical diagonal/substep order", id);
        if (beam1_owner[static_cast<std::size_t>(current.beam1_slice)] == wave ||
            beam2_owner[static_cast<std::size_t>(current.beam2_slice)] == wave)
          return ValidationResult::failure(
              "same-wave encounters do not have exclusive slices", id);
        beam1_owner[static_cast<std::size_t>(current.beam1_slice)] = wave;
        beam2_owner[static_cast<std::size_t>(current.beam2_slice)] = wave;
      }
    }
  }

  for (std::size_t position = 0; position < nodes_.size(); ++position) {
    const EncounterNode& current = nodes_[position];
    const EncounterId id = static_cast<EncounterId>(position);
    if (current.id != id)
      return ValidationResult::failure(
          "node ID does not equal topological storage index", id);
    if (current.beam1_slice >= beam1_slices_ ||
        current.beam2_slice >= beam2_slices_ ||
        current.substep >= substeps_)
      return ValidationResult::failure("node logical index is out of range", id);
    if (current.diagonal != current.beam1_slice + current.beam2_slice)
      return ValidationResult::failure("node diagonal is not i+j", id);
    if (current.wave != current.diagonal * substeps_ + current.substep)
      return ValidationResult::failure("node wave is not diagonal*n_t+t", id);
    if (pair_to_node_[pair_index(current.beam1_slice,
                                 current.beam2_slice,
                                 current.substep)] != id)
      return ValidationResult::failure("tuple-to-node lookup is not bijective", id);

    std::array<EncounterId, 2> expected = {
        {kInvalidEncounter, kInvalidEncounter}};
    std::uint8_t expected_count = 0;
    if (current.substep > 0) {
      expected[expected_count++] =
          pair_to_node_[pair_index(current.beam1_slice,
                                   current.beam2_slice,
                                   current.substep - 1)];
    } else {
      if (current.beam1_slice > 0)
        expected[expected_count++] =
            pair_to_node_[pair_index(current.beam1_slice - 1,
                                     current.beam2_slice,
                                     substeps_ - 1)];
      if (current.beam2_slice > 0)
        expected[expected_count++] =
            pair_to_node_[pair_index(current.beam1_slice,
                                     current.beam2_slice - 1,
                                     substeps_ - 1)];
    }
    if (current.dependency_count != expected_count)
      return ValidationResult::failure("node has incorrect dependency count", id);
    for (std::uint8_t dependency = 0; dependency < expected_count;
         ++dependency) {
      const EncounterId predecessor = expected[dependency];
      if (current.dependencies[dependency] != predecessor ||
          predecessor >= id ||
          nodes_[static_cast<std::size_t>(predecessor)].wave >= current.wave)
        return ValidationResult::failure(
            "node dependency violates topological order", id);
    }
    for (std::uint8_t dependency = expected_count; dependency < 2;
         ++dependency) {
      if (current.dependencies[dependency] != kInvalidEncounter)
        return ValidationResult::failure(
            "unused dependency slot is not sentinel", id);
    }
  }

  for (std::size_t id = 0; id < membership.size(); ++id) {
    if (membership[id] != 1)
      return ValidationResult::failure("node is absent from topological waves",
                                       static_cast<std::uint64_t>(id));
  }
  return ValidationResult::success();
}

}  // namespace modern
}  // namespace feebl
