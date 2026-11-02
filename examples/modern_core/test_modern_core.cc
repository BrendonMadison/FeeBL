#include "beamSoACPP.h"
#include "cellCSRCPP.h"
#include "encounterDAGCPP.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using feebl::modern::ArrayOffset;
using feebl::modern::BeamMetadata;
using feebl::modern::BeamSoA;
using feebl::modern::CellCSR;
using feebl::modern::CellKey;
using feebl::modern::CellOrder;
using feebl::modern::EncounterDAG;
using feebl::modern::EncounterId;
using feebl::modern::GridMappingMode;
using feebl::modern::ParticleIndex;
using feebl::modern::ParticleRecord;
using feebl::modern::ParticleSpecies;
using feebl::modern::SliceIndex;
using feebl::modern::StableParticleId;
using feebl::modern::SubstepIndex;
using feebl::modern::UniformGrid2D;
using feebl::modern::ValidationResult;
using feebl::modern::kOutsideCell;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-69s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

template <class Function>
bool throws(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

ParticleRecord electron(StableParticleId id, double x = 0.0,
                        double y = 0.0) {
  ParticleRecord particle;
  particle.stable_id = id;
  particle.x = x;
  particle.y = y;
  particle.z = 100.0 + static_cast<double>(id % 17);
  particle.vx = 1.0e-6 * static_cast<double>(id % 5);
  particle.vy = -2.0e-6 * static_cast<double>(id % 7);
  particle.energy = 125.0;
  particle.macro_weight = 0.5;
  particle.upsilon = 0.01;
  particle.sx = 0.1;
  particle.sy = -0.2;
  particle.sz = 0.8;
  particle.species = ParticleSpecies::Electron;
  particle.charge_e = -1;
  particle.direction = 1;
  particle.beam_id = 1;
  particle.flags = static_cast<std::uint32_t>(id & 7u);
  return particle;
}

struct DirectLegacyMapping {
  bool inside = false;
  CellKey key = kOutsideCell;
  int ix = 0;
  int iy = 0;
  float weight_x = 0.0f;
  float weight_y = 0.0f;
};

// Independent transcription of GENERAL_GRID::init_grid_phys and the two
// particleInGrid overloads.  Tests below compare the modern mapping against
// this formula at boundaries and their adjacent float values.
DirectLegacyMapping direct_legacy_map(float x, float y, float cut_x,
                                      float cut_y, int nx, int ny,
                                      bool cic) {
  const float dx = static_cast<float>(2.0 * cut_x / static_cast<float>(nx));
  const float dy = static_cast<float>(2.0 * cut_y / static_cast<float>(ny));
  const float inverse_dx = static_cast<float>(1.0 / dx);
  const float inverse_dy = static_cast<float>(1.0 / dy);
  const float offset_x = static_cast<float>(0.5 * static_cast<float>(nx));
  const float offset_y = static_cast<float>(0.5 * static_cast<float>(ny));
  float hx = static_cast<float>(x * inverse_dx + offset_x);
  float hy = static_cast<float>(y * inverse_dy + offset_y);
  if (cic) {
    hx = static_cast<float>(static_cast<double>(hx) - 0.5);
    hy = static_cast<float>(static_cast<double>(hy) - 0.5);
  }
  const int ix = static_cast<int>(std::floor(hx));
  const int iy = static_cast<int>(std::floor(hy));
  DirectLegacyMapping result;
  if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1) return result;
  result.inside = true;
  result.ix = ix;
  result.iy = iy;
  result.key = static_cast<CellKey>(ix) * static_cast<CellKey>(ny) +
               static_cast<CellKey>(iy);
  if (cic) {
    result.weight_x = static_cast<float>(hx - static_cast<float>(ix));
    result.weight_y = static_cast<float>(hy - static_cast<float>(iy));
  }
  return result;
}

bool matches_legacy(const feebl::modern::CellMapping& actual,
                    const DirectLegacyMapping& expected) {
  if (actual.inside != expected.inside || actual.key != expected.key)
    return false;
  if (!actual.inside) return true;
  return actual.ix == static_cast<CellKey>(expected.ix) &&
         actual.iy == static_cast<CellKey>(expected.iy) &&
         static_cast<float>(actual.weight_x) == expected.weight_x &&
         static_cast<float>(actual.weight_y) == expected.weight_y;
}

void test_beam_soa() {
  std::printf("\nBeamSoA ownership and invariants\n");

  BeamSoA zero_slices(0);
  check(zero_slices.complete() && zero_slices.empty() &&
            zero_slices.slice_offsets() == std::vector<ArrayOffset>{0},
        "zero-slice beam is a complete, valid empty owner");
  check(static_cast<bool>(zero_slices.validate()),
        "zero-slice beam passes validation");

  BeamSoA beam(4, true);
  BeamMetadata metadata;
  metadata.bunch_population = 2.0e10;
  beam.set_metadata(metadata);
  beam.reserve(8);
  beam.append_slice({electron(900), electron(12)});
  beam.append_slice({});
  beam.append_slice({electron(44)});
  beam.append_slice({electron(5), electron(99), electron(71)});

  const std::vector<ArrayOffset> expected_offsets = {0, 2, 2, 3, 6};
  check(beam.complete() && beam.size() == 6 && beam.slice_count() == 4,
        "uneven beam completes with the exact particle/slice counts");
  check(beam.slice_offsets() == expected_offsets,
        "empty and uneven slices have canonical CSR offsets");
  check(beam.slice_range(1) == std::make_pair<ArrayOffset, ArrayOffset>(2, 2),
        "empty slice has a zero-width range");
  check(beam.slice_of(0) == 0 && beam.slice_of(1) == 0 &&
            beam.slice_of(2) == 2 && beam.slice_of(5) == 3,
        "particle-to-slice lookup skips repeated empty-slice offsets");
  check(beam.stable_ids() ==
            std::vector<StableParticleId>({900, 12, 44, 5, 99, 71}),
        "stable IDs survive dense slice construction exactly");
  check(beam.spin_enabled() && beam.sx().size() == beam.size() &&
            beam.sy().size() == beam.size() && beam.sz().size() == beam.size(),
        "optional spin component is dense when enabled");
  const ParticleRecord restored = beam.particle(2);
  check(restored.stable_id == 44 && restored.species == ParticleSpecies::Electron &&
            restored.charge_e == -1 && restored.direction == 1 &&
            restored.energy == 125.0 && restored.sz == 0.8,
        "record view preserves identity, phase space, species and spin");
  check(static_cast<bool>(beam.validate()),
        "complete uneven BeamSoA passes all structural/physical invariants");

  check(throws([&beam] { beam.append_slice({electron(123)}); }),
        "append beyond declared slice count fails loudly");
  check(throws([&beam] { (void)beam.slice_range(4); }) &&
            throws([&beam] { (void)beam.slice_of(6); }),
        "slice and particle range violations fail loudly");

  BeamSoA incomplete(2);
  incomplete.append_slice({electron(1)});
  check(!static_cast<bool>(incomplete.validate()) &&
            throws([&incomplete] { (void)incomplete.slice_range(0); }),
        "incomplete slice construction cannot masquerade as a valid beam");

  BeamSoA duplicate(1);
  duplicate.append_slice({electron(17), electron(17)});
  check(!static_cast<bool>(duplicate.validate()),
        "duplicate stable particle IDs are rejected");

  BeamSoA bad_direction(1);
  ParticleRecord bad = electron(18);
  bad.direction = 0;
  bad_direction.append_slice({bad});
  check(!static_cast<bool>(bad_direction.validate()),
        "invalid longitudinal direction is rejected");

  BeamSoA bad_charge(1);
  bad = electron(19);
  bad.charge_e = 1;
  bad_charge.append_slice({bad});
  check(!static_cast<bool>(bad_charge.validate()),
        "species/charge alias is rejected");

  BeamSoA bad_weight(1);
  bad = electron(20);
  bad.macro_weight = -1.0;
  bad_weight.append_slice({bad});
  check(!static_cast<bool>(bad_weight.validate()),
        "negative macro-particle weight is rejected");

  BeamSoA bad_spin(1, true);
  bad = electron(21);
  bad.sx = 1.0;
  bad.sy = 1.0;
  bad.sz = 1.0;
  bad_spin.append_slice({bad});
  check(!static_cast<bool>(bad_spin.validate()),
        "unphysical spin/Stokes norm is rejected");

  BeamSoA max_id(1);
  max_id.append_slice({electron(std::numeric_limits<StableParticleId>::max())});
  check(!static_cast<bool>(max_id.validate()),
        "counter-RNG no-parent sentinel cannot be a physical stable ID");
}

BeamSoA make_csr_beam() {
  BeamSoA beam(3);
  const double upper_inside =
      std::nextafter(1.0, -std::numeric_limits<double>::infinity());
  beam.append_slice({electron(50, -1.0, -1.0),
                     electron(10, -0.25, -0.75),
                     electron(40, 0.0, -0.5)});
  beam.append_slice({});
  beam.append_slice({electron(30, -0.5, 0.0),
                     electron(20, upper_inside, upper_inside),
                     electron(60, 1.0, 0.0),
                     electron(70, -1.0001, 0.0),
                     electron(80, 0.0, 1.0)});
  return beam;
}

void test_cell_csr() {
  std::printf("\nCellCSR mapping, ordering and overflow guards\n");

  BeamSoA beam = make_csr_beam();
  UniformGrid2D grid(-1.0, 1.0, -1.0, 1.0, 2, 2);
  check(grid.cell_count() == 4, "uniform grid derives exact cell count");
  check(grid.map(-1.0, -1.0).inside && grid.map(-1.0, -1.0).key == 0,
        "lower grid boundary is included");
  check(grid.map(0.0, -0.5).key == 1 && grid.map(-0.5, 0.0).key == 2 &&
            grid.map(0.0, 0.0).key == 3,
        "internal boundaries map uniquely to their upper cell");
  check(!grid.map(1.0, 0.0).inside && !grid.map(0.0, 1.0).inside,
        "upper grid boundaries are outside under half-open convention");
  check(!grid.map(std::numeric_limits<double>::quiet_NaN(), 0.0).inside,
        "non-finite coordinates never produce a cell key");

  CellCSR canonical(beam, grid, CellOrder::ByStableParticleId);
  const std::vector<ArrayOffset> expected_offsets = {0, 2, 3, 4, 5};
  const std::vector<ParticleIndex> expected_canonical = {1, 0, 2, 3, 4};
  const std::vector<ParticleIndex> expected_outside = {5, 6, 7};
  check(canonical.cell_offsets() == expected_offsets,
        "two-pass counts/scan produce exact CSR offsets");
  check(canonical.particle_indices() == expected_canonical,
        "canonical cell segments are ordered by stable particle ID");
  check(canonical.outside_particles() == expected_outside,
        "outside population is explicit and deterministically ordered");
  check(canonical.particle_cell_keys() ==
            std::vector<CellKey>({0, 0, 1, 2, 3, kOutsideCell,
                                  kOutsideCell, kOutsideCell}),
        "per-particle cell-key cache includes explicit outside sentinels");
  check(canonical.cell_range(0).size() == 2 &&
            canonical.cell_range(1).size() == 1 &&
            canonical.cell_range(3).size() == 1,
        "cell ranges address compact membership segments");
  check(static_cast<bool>(canonical.validate(beam)),
        "canonical CellCSR passes exhaustive membership validation");

  CellCSR repeated(beam, grid, CellOrder::ByStableParticleId);
  check(repeated.cell_offsets() == canonical.cell_offsets() &&
            repeated.particle_indices() == canonical.particle_indices() &&
            repeated.particle_cell_keys() == canonical.particle_cell_keys() &&
            repeated.outside_particles() == canonical.outside_particles(),
        "repeated canonical builds are byte-order deterministic");

  CellCSR dense(beam, grid, CellOrder::DenseInput);
  check(dense.particle_indices() ==
            std::vector<ParticleIndex>({0, 1, 2, 3, 4}) &&
            static_cast<bool>(dense.validate(beam)),
        "fast two-pass mode preserves dense input order within every cell");

  const UniformGrid2D legacy_ngp = UniformGrid2D::legacy_ngp(4.0, 4.0, 8, 8);
  const UniformGrid2D legacy_cic = UniformGrid2D::legacy_cic(4.0, 4.0, 8, 8);
  check(legacy_ngp.mapping_mode() == GridMappingMode::LegacyNGP &&
            legacy_cic.mapping_mode() == GridMappingMode::LegacyCIC &&
            legacy_ngp.cell_count() == 64 &&
            legacy_ngp.active_nx() == 7 && legacy_ngp.active_ny() == 7,
        "legacy maps retain full 8x8 storage stride and 7x7 active bases");
  check(legacy_ngp.map(0.0, 0.0).key == 4 * 8 + 4 &&
            legacy_cic.map(0.0, 0.0).key == 3 * 8 + 3 &&
            legacy_cic.map(0.0, 0.0).weight_x == 0.5 &&
            legacy_cic.map(0.0, 0.0).weight_y == 0.5,
        "NGP/CIC offsets and x-major legacy storage keys are explicit");

  const float minus_infinity = -std::numeric_limits<float>::infinity();
  const float plus_infinity = std::numeric_limits<float>::infinity();
  const std::vector<float> probes = {
      -4.0f,
      std::nextafter(-4.0f, minus_infinity),
      std::nextafter(-4.0f, plus_infinity),
      -3.5f,
      std::nextafter(-3.5f, minus_infinity),
      std::nextafter(-3.5f, plus_infinity),
      -3.0f,
      std::nextafter(-3.0f, minus_infinity),
      std::nextafter(-3.0f, plus_infinity),
      0.0f,
      3.0f,
      std::nextafter(3.0f, minus_infinity),
      std::nextafter(3.0f, plus_infinity),
      3.5f,
      std::nextafter(3.5f, minus_infinity),
      std::nextafter(3.5f, plus_infinity)};
  bool legacy_formula_exact = true;
  for (float coordinate : probes) {
    legacy_formula_exact = legacy_formula_exact &&
        matches_legacy(legacy_ngp.map(coordinate, 0.0),
                       direct_legacy_map(coordinate, 0.0f, 4.0f, 4.0f,
                                         8, 8, false)) &&
        matches_legacy(legacy_ngp.map(0.0, coordinate),
                       direct_legacy_map(0.0f, coordinate, 4.0f, 4.0f,
                                         8, 8, false)) &&
        matches_legacy(legacy_cic.map(coordinate, 0.0),
                       direct_legacy_map(coordinate, 0.0f, 4.0f, 4.0f,
                                         8, 8, true)) &&
        matches_legacy(legacy_cic.map(0.0, coordinate),
                       direct_legacy_map(0.0f, coordinate, 4.0f, 4.0f,
                                         8, 8, true));
  }
  check(legacy_formula_exact,
        "legacy NGP/CIC maps match direct float formulas at boundaries and one ULP");

  CellCSR legacy_csr(beam, UniformGrid2D::legacy_ngp(1.0, 1.0, 4, 4),
                     CellOrder::ByStableParticleId);
  bool guard_stride = legacy_csr.cell_offsets().size() == 17;
  for (CellKey ix = 0; ix < 4; ++ix) {
    guard_stride = guard_stride && legacy_csr.cell_range(ix * 4 + 3).empty();
  }
  for (CellKey iy = 0; iy < 4; ++iy) {
    guard_stride = guard_stride && legacy_csr.cell_range(3 * 4 + iy).empty();
  }
  check(guard_stride && static_cast<bool>(legacy_csr.validate(beam)),
        "legacy CSR preserves empty terminal guard column/row in full storage stride");

  BeamSoA cluster(1);
  std::vector<ParticleRecord> particles;
  for (std::uint64_t i = 0; i < 128; ++i)
    particles.push_back(electron(128 - i, -0.2, 0.2));
  cluster.append_slice(particles);
  CellCSR clustered(cluster, grid, CellOrder::ByStableParticleId);
  bool cluster_sorted = clustered.cell_range(2).size() == 128;
  for (std::size_t position = 0;
       position < clustered.particle_indices().size(); ++position) {
    const ParticleIndex particle = clustered.particle_indices()[position];
    cluster_sorted = cluster_sorted &&
        cluster.stable_ids()[static_cast<std::size_t>(particle)] == position + 1;
  }
  check(cluster_sorted && static_cast<bool>(clustered.validate(cluster)),
        "heavily clustered cell remains complete and stable-ID ordered");

  beam.x_data()[0] = 2.0;
  check(!static_cast<bool>(canonical.validate(beam)),
        "validation detects stale CSR after particle motion");
  beam.x_data()[0] = -1.0;
  check(static_cast<bool>(canonical.validate(beam)),
        "restored coordinates restore cached-membership validity");

  check(throws([&canonical] { (void)canonical.cell_range(4); }),
        "out-of-range cell lookup fails loudly");
  check(throws([] { UniformGrid2D bad(0.0, 0.0, -1.0, 1.0, 2, 2); }),
        "degenerate grid bounds are rejected");
  check(throws([] { UniformGrid2D bad(-1.0, 1.0, -1.0, 1.0, 0, 2); }),
        "zero grid dimension is rejected");
  check(throws([] {
          UniformGrid2D bad(-1.0, 1.0, -1.0, 1.0,
                            std::numeric_limits<CellKey>::max(), 2);
        }),
        "grid cell-count overflow is rejected before allocation");
  check(throws([] { (void)UniformGrid2D::legacy_cic(1.0, 1.0, 1, 8); }),
        "legacy map rejects dimensions without an active plus guard cell");
}

void verify_wave_ownership(const EncounterDAG& dag, bool& unique,
                           bool& exclusive, bool& topological) {
  std::set<EncounterId> all;
  unique = true;
  exclusive = true;
  topological = true;
  for (const auto& wave : dag.waves()) {
    std::set<SliceIndex> beam1;
    std::set<SliceIndex> beam2;
    for (EncounterId id : wave.encounters) {
      const auto& node = dag.node(id);
      unique = unique && all.insert(id).second;
      exclusive = exclusive && beam1.insert(node.beam1_slice).second;
      exclusive = exclusive && beam2.insert(node.beam2_slice).second;
      for (std::uint8_t d = 0; d < node.dependency_count; ++d) {
        const auto& predecessor = dag.node(node.dependencies[d]);
        topological = topological && predecessor.wave < node.wave &&
                      predecessor.id < node.id;
      }
    }
  }
  unique = unique && all.size() == dag.encounter_count();
}

void test_encounter_dag() {
  std::printf("\nEncounterDAG counts, ownership and topological order\n");

  EncounterDAG empty_left(0, 5);
  EncounterDAG empty_right(7, 0);
  check(empty_left.encounter_count() == 0 && empty_left.wave_count() == 0 &&
            empty_right.encounter_count() == 0 && empty_right.wave_count() == 0,
        "zero slices on either beam produce the empty DAG");
  check(static_cast<bool>(empty_left.validate()) &&
            static_cast<bool>(empty_right.validate()),
        "both empty-DAG orientations validate");

  EncounterDAG chain(1, 4);
  bool chain_ok = chain.encounter_count() == 4 && chain.wave_count() == 4;
  for (SliceIndex j = 0; j < 4; ++j) {
    const auto& node = chain.node(chain.node_id(0, j));
    chain_ok = chain_ok && node.wave == j &&
               node.dependency_count == (j == 0 ? 0 : 1);
    if (j > 0)
      chain_ok = chain_ok && node.dependencies[0] == chain.node_id(0, j - 1);
  }
  check(chain_ok && static_cast<bool>(chain.validate()),
        "one-by-many DAG reduces to a correctly ordered dependency chain");

  EncounterDAG uneven(3, 5);
  const std::vector<std::size_t> expected_widths = {1, 2, 3, 3, 3, 2, 1};
  bool widths = uneven.encounter_count() == 15 && uneven.wave_count() == 7;
  for (std::size_t wave = 0; wave < expected_widths.size(); ++wave)
    widths = widths && uneven.waves()[wave].encounters.size() ==
                          expected_widths[wave];
  check(widths, "unequal 3x5 slice DAG has exact encounter and wave counts");

  bool exhaustive_pairs = true;
  std::set<EncounterId> pair_ids;
  for (SliceIndex i = 0; i < 3; ++i) {
    for (SliceIndex j = 0; j < 5; ++j) {
      const EncounterId id = uneven.node_id(i, j);
      const auto& node = uneven.node(id);
      exhaustive_pairs = exhaustive_pairs && pair_ids.insert(id).second &&
                         node.beam1_slice == i && node.beam2_slice == j &&
                         node.wave == i + j;
    }
  }
  check(exhaustive_pairs && pair_ids.size() == 15,
        "every unequal-grid slice pair maps bijectively to one logical encounter");

  bool unique = false;
  bool exclusive = false;
  bool topological = false;
  verify_wave_ownership(uneven, unique, exclusive, topological);
  check(unique, "each encounter occurs in exactly one topological wave");
  check(exclusive, "same-wave encounters have exclusive beam-1 and beam-2 slices");
  check(topological, "all slice-mutation dependencies precede their consumer");

  const auto& interior = uneven.node(uneven.node_id(1, 2));
  check(interior.dependency_count == 2 &&
            interior.dependencies[0] == uneven.node_id(0, 2) &&
            interior.dependencies[1] == uneven.node_id(1, 1),
        "interior node names both and only prior same-slice encounters");
  check(static_cast<bool>(uneven.validate()) &&
            uneven.is_legacy_antidiagonal(),
        "unequal legacy specialization passes exhaustive DAG validation");

  EncounterDAG staged = EncounterDAG::build_legacy(2, 3, 4);
  bool staged_counts = staged.substep_count() == 4 &&
                       staged.encounter_count() == 2 * 3 * 4 &&
                       staged.wave_count() == (2 + 3 - 1) * 4;
  std::set<std::uint64_t> staged_tuples;
  bool staged_order = true;
  for (const auto& wave : staged.waves()) {
    staged_order = staged_order && wave.index == wave.diagonal * 4 + wave.substep;
    SliceIndex previous_i = 0;
    bool first = true;
    for (EncounterId id : wave.encounters) {
      const auto& node = staged.node(id);
      const std::uint64_t tuple =
          (node.beam1_slice * 3 + node.beam2_slice) * 4 + node.substep;
      staged_order = staged_order && node.diagonal == wave.diagonal &&
                     node.substep == wave.substep &&
                     node.wave == wave.index &&
                     node.beam1_slice + node.beam2_slice == wave.diagonal &&
                     (first || node.beam1_slice == previous_i + 1);
      first = false;
      previous_i = node.beam1_slice;
      staged_tuples.insert(tuple);
    }
  }
  for (SliceIndex i = 0; i < 2; ++i) {
    for (SliceIndex j = 0; j < 3; ++j) {
      for (SubstepIndex t = 0; t < 4; ++t) {
        const auto& node = staged.node(staged.node_id(i, j, t));
        staged_counts = staged_counts && node.beam1_slice == i &&
                        node.beam2_slice == j && node.substep == t;
      }
    }
  }
  check(staged_counts && staged_tuples.size() == 24,
        "n_t DAG contains every (beam1 slice,beam2 slice,substep) exactly once");
  check(staged_order,
        "n_t DAG uses canonical legacy diagonal/substep/slice-pair order");

  const auto& continued = staged.node(staged.node_id(1, 1, 2));
  const auto& new_diagonal = staged.node(staged.node_id(1, 1, 0));
  check(continued.dependency_count == 1 &&
            continued.dependencies[0] == staged.node_id(1, 1, 1),
        "substep t>0 depends on the same encounter at t-1");
  check(new_diagonal.dependency_count == 2 &&
            new_diagonal.dependencies[0] == staged.node_id(0, 1, 3) &&
            new_diagonal.dependencies[1] == staged.node_id(1, 0, 3),
        "new diagonal waits for last substep of both prior same-slice encounters");
  verify_wave_ownership(staged, unique, exclusive, topological);
  check(unique && exclusive && topological &&
            static_cast<bool>(staged.validate()),
        "all n_t waves are unique, ownership-exclusive and topologically valid");

  EncounterDAG square(4, 4);
  const std::vector<std::vector<std::pair<SliceIndex, SliceIndex>>> expected = {
      {{0, 0}},
      {{0, 1}, {1, 0}},
      {{0, 2}, {1, 1}, {2, 0}},
      {{0, 3}, {1, 2}, {2, 1}, {3, 0}},
      {{1, 3}, {2, 2}, {3, 1}},
      {{2, 3}, {3, 2}},
      {{3, 3}}};
  bool legacy_order = square.waves().size() == expected.size();
  for (std::size_t wave = 0; wave < expected.size(); ++wave) {
    legacy_order = legacy_order &&
                   square.waves()[wave].encounters.size() == expected[wave].size();
    for (std::size_t p = 0;
         legacy_order && p < expected[wave].size(); ++p) {
      const auto& node = square.node(square.waves()[wave].encounters[p]);
      legacy_order = legacy_order &&
          node.beam1_slice == expected[wave][p].first &&
          node.beam2_slice == expected[wave][p].second;
    }
  }
  check(legacy_order && static_cast<bool>(square.validate()),
        "equal-slice specialization exactly reproduces legacy anti-diagonal order");

  check(throws([&uneven] { (void)uneven.node(15); }) &&
            throws([&uneven] { (void)uneven.node_id(3, 0); }) &&
            throws([&uneven] { (void)uneven.node_id(0, 5); }),
        "invalid encounter and slice-pair lookups fail loudly");
  check(throws([&staged] { (void)staged.node_id(0, 0, 4); }) &&
            throws([] { EncounterDAG zero_substeps(2, 2, 0); }),
        "invalid substep lookup and n_t=0 fail loudly");
  check(throws([] {
          EncounterDAG impossible(std::numeric_limits<SliceIndex>::max(), 2);
        }),
        "encounter-count overflow is rejected before allocation");
}

}  // namespace

int main() {
  test_beam_soa();
  test_cell_csr();
  test_encounter_dag();
  std::printf("\n%s (%d failure%s)\n", failures ? "FAILURES PRESENT" : "all modern-core checks passed",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
