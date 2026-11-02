#include "particleBeamCPP.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Exception, typename Callable>
bool throws_as(Callable callable) {
  try {
    callable();
  } catch (const Exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

std::vector<PRIMARY_PARTICLE_ID> flatten_ids(const PARTICLE_BEAM& beam,
                                             int slices) {
  beam.require_valid_particle_id_sidecar();
  std::vector<PRIMARY_PARTICLE_ID> result;
  for (int slice = 0; slice < slices; ++slice) {
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        beam.getParticleIdVectorStrict(slice);
    result.insert(result.end(), ids.begin(), ids.end());
  }
  return result;
}

std::set<PRIMARY_PARTICLE_ID> id_set(const PARTICLE_BEAM& beam, int slices) {
  const std::vector<PRIMARY_PARTICLE_ID> ids = flatten_ids(beam, slices);
  return std::set<PRIMARY_PARTICLE_ID>(ids.begin(), ids.end());
}

void initialize_generated(PARTICLE_BEAM& beam, unsigned long count,
                          float sigma_z, float delta_z,
                          int charge_symmetric) {
  beam.init_particles(count,
                      100.0f,   // sigma_x [nm]
                      10.0f,    // sigma_y [nm]
                      sigma_z,
                      0, 0,     // normal transverse and longitudinal laws
                      1.0e-6f, 1.0e-6f,
                      delta_z,
                      100.0f,   // beam energy [GeV]
                      charge_symmetric,
                      0);
}

void test_schema_known_answers_and_bounds() {
  check(make_generated_primary_particle_id(1u, 7u) ==
            0x1500000000000007ull,
        "ordinary primary-ID fixed vector");
  check(make_symmetric_primary_particle_id(2u, 3u, 2u) ==
            0x1a0000000000000eull,
        "symmetric primary-ID fixed vector");
  check(make_loaded_primary_particle_id(1u, 5u) ==
            0x1d00000000000005ull,
        "loaded primary-ID fixed vector");

  PRIMARY_PARTICLE_ID_FIELDS fields;
  check(decode_primary_particle_id(
            make_symmetric_primary_particle_id(2u, 123456u, 3u), fields) &&
            fields.schema == PRIMARY_PARTICLE_ID_SCHEMA &&
            fields.provenance ==
                PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC &&
            fields.beam_label == 2u && fields.source_ordinal == 123456u &&
            fields.symmetric_lane == 3u,
        "symmetric primary-ID round trip");

  check(make_generated_primary_particle_id(
            1u, PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL) != 0u,
        "ordinary maximum source ordinal is accepted");
  check(make_symmetric_primary_particle_id(
            2u, PRIMARY_PARTICLE_MAX_SYMMETRIC_GENERATOR_ORDINAL, 3u) != 0u,
        "symmetric maximum source ordinal/lane is accepted");
  check(make_loaded_primary_particle_id(
            2u, PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL) != 0u,
        "loaded maximum source ordinal is accepted");
  check(throws_as<std::overflow_error>([]() {
          (void)make_generated_primary_particle_id(
              1u, PRIMARY_PARTICLE_MAX_SOURCE_ORDINAL + 1u);
        }),
        "ordinary source ordinal overflow is loud");
  check(throws_as<std::overflow_error>([]() {
          (void)make_symmetric_primary_particle_id(
              1u, PRIMARY_PARTICLE_MAX_SYMMETRIC_GENERATOR_ORDINAL + 1u, 0u);
        }),
        "symmetric source ordinal overflow is loud");
  check(throws_as<std::invalid_argument>([]() {
          (void)make_symmetric_primary_particle_id(1u, 0u, 4u);
        }),
        "symmetric lane overflow is loud");
  check(throws_as<std::invalid_argument>([]() {
          (void)make_loaded_primary_particle_id(1u, 0u);
        }),
        "loaded source record zero is rejected");
  check(throws_as<std::invalid_argument>([]() {
          (void)make_generated_primary_particle_id(0u, 1u);
        }),
        "missing beam label is rejected");

  const PRIMARY_PARTICLE_ID ordinary =
      make_generated_primary_particle_id(1u, 99u);
  check(ordinary != make_generated_primary_particle_id(2u, 99u),
        "beam label separates otherwise identical IDs");
  check(ordinary != make_loaded_primary_particle_id(1u, 99u),
        "provenance separates otherwise identical IDs");
}

void test_ordinary_attempted_ordinals() {
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(12345u);
  PARTICLE_BEAM beam(8, 0, polarization, &random, 1u);
  initialize_generated(beam, 64u, 1.0f, 10.0f, 0);

  const std::vector<PRIMARY_PARTICLE_ID> ids = flatten_ids(beam, 8);
  check(ids.size() == 64u, "wide ordinary window accepts all attempts");
  std::set<std::uint64_t> source_ordinals;
  for (PRIMARY_PARTICLE_ID id : ids) {
    PRIMARY_PARTICLE_ID_FIELDS fields;
    check(decode_primary_particle_id(id, fields),
          "ordinary ID decodes");
    check(fields.provenance ==
              PRIMARY_PARTICLE_PROVENANCE::GENERATED_ORDINARY,
          "ordinary provenance is recorded");
    check(fields.beam_label == 1u, "ordinary beam label is recorded");
    source_ordinals.insert(fields.source_ordinal);
  }
  check(source_ordinals.size() == 64u && *source_ordinals.begin() == 0u &&
            *source_ordinals.rbegin() == 63u,
        "ordinary IDs map exactly to attempted generator ordinals");
  check(std::set<PRIMARY_PARTICLE_ID>(ids.begin(), ids.end()).size() ==
            ids.size(),
        "ordinary IDs are unique");

  // A narrow physical window rejects many attempted z values.  Retained IDs
  // must preserve the attempted loop ordinal, not be renumbered by insertion.
  RNDM narrow_random(67890u);
  PARTICLE_BEAM narrow(1, 0, polarization, &narrow_random, 1u);
  initialize_generated(narrow, 512u, 10.0f, 4.0f, 0);
  const std::vector<PRIMARY_PARTICLE_ID> narrow_ids = flatten_ids(narrow, 1);
  check(!narrow_ids.empty() && narrow_ids.size() < 400u,
        "narrow ordinary window has accepted and rejected attempts");
  bool observed_gap = false;
  for (std::size_t stored = 0; stored < narrow_ids.size(); ++stored) {
    PRIMARY_PARTICLE_ID_FIELDS decoded;
    check(decode_primary_particle_id(narrow_ids[stored], decoded),
          "narrow ordinary ID decodes");
    check(decoded.source_ordinal < 512u,
          "ordinary attempted ordinal stays in source range");
    if (decoded.source_ordinal != stored) observed_gap = true;
  }
  check(observed_gap,
        "rejected attempts leave source-ordinal gaps (no slice/index IDs)");
}

void test_symmetric_generator_ordinal_and_lane() {
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(24680u);
  PARTICLE_BEAM beam(8, 0, polarization, &random, 2u);
  initialize_generated(beam, 40u, 1.0f, 10.0f, 1);

  const std::vector<PRIMARY_PARTICLE_ID> ids = flatten_ids(beam, 8);
  check(ids.size() == 40u,
        "normal symmetric generation makes four lanes per source attempt");
  std::set<std::pair<std::uint64_t, unsigned int>> identities;
  for (PRIMARY_PARTICLE_ID id : ids) {
    PRIMARY_PARTICLE_ID_FIELDS fields;
    check(decode_primary_particle_id(id, fields),
          "symmetric ID decodes");
    check(fields.provenance ==
              PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC,
          "symmetric provenance is recorded");
    check(fields.beam_label == 2u, "symmetric beam label is recorded");
    check(fields.source_ordinal < 10u && fields.symmetric_lane < 4u,
          "symmetric source ordinal/lane are bounded");
    identities.emplace(fields.source_ordinal, fields.symmetric_lane);
  }
  check(identities.size() == 40u,
        "symmetric generator ordinal plus lane is one-to-one");
  for (std::uint64_t ordinal = 0; ordinal < 10u; ++ordinal)
    for (unsigned int lane = 0; lane < 4u; ++lane)
      check(identities.count(std::make_pair(ordinal, lane)) == 1u,
            "every symmetric source/lane identity is present exactly once");
}

void test_slicing_stability() {
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random4(13579u);
  RNDM random8(13579u);
  PARTICLE_BEAM four_slices(4, 0, polarization, &random4, 1u);
  PARTICLE_BEAM eight_slices(8, 0, polarization, &random8, 1u);

  // Both grids cover the same [-40,40) longitudinal interval.  They consume
  // the same legacy draws but assign accepted sources to different slice bins.
  initialize_generated(four_slices, 512u, 15.0f, 20.0f, 0);
  initialize_generated(eight_slices, 512u, 15.0f, 10.0f, 0);
  const std::set<PRIMARY_PARTICLE_ID> ids4 = id_set(four_slices, 4);
  const std::set<PRIMARY_PARTICLE_ID> ids8 = id_set(eight_slices, 8);
  check(ids4 == ids8 && !ids4.empty(),
        "changing slicing at fixed physical window preserves source IDs");
}

void test_loaded_record_ordinals(const std::string& fixture) {
  BEAM_FROM_FILE* from_file = new BEAM_FROM_FILE(fixture);
  const std::vector<std::uint64_t>& accepted =
      from_file->source_record_ordinals();
  const std::vector<std::uint64_t> expected_accepted = {1u, 3u, 4u, 5u};
  check(accepted == expected_accepted,
        "loaded reader retains physical line ordinals and rejected-line gap");
  check(from_file->accepted_particle_count() == 4u,
        "loaded accepted-particle/source sidecars align");
  check(from_file->pick_last_source_record_ordinal() == 5u,
        "reverse-pop path begins with the last accepted source line");

  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(11223u);
  PARTICLE_BEAM beam(4, 0, polarization, &random, 1u);
  const unsigned int loaded = beam.load_particles(
      from_file, 0.0f, -2000.0f, 1000.0f, 1.0f, 1.0f, 1.0f);
  check(from_file == NULL, "legacy loaded-beam transfer consumes its owner");
  check(loaded == 3u,
        "loaded transfer rejects the accepted input line outside z window");
  beam.require_valid_particle_id_sidecar();

  std::map<int, std::uint64_t> energy_to_source;
  std::set<PRIMARY_PARTICLE_ID> unique;
  for (int slice = 0; slice < 4; ++slice) {
    const std::vector<PARTICLE*>& particles = beam.getParticleVector(slice);
    const std::vector<PRIMARY_PARTICLE_ID>& ids =
        beam.getParticleIdVectorStrict(slice);
    check(particles.size() == ids.size(),
          "loaded particle and ID vectors are shape aligned");
    for (std::size_t index = 0; index < ids.size(); ++index) {
      PRIMARY_PARTICLE_ID_FIELDS fields;
      check(decode_primary_particle_id(ids[index], fields),
            "loaded ID decodes");
      check(fields.provenance ==
                PRIMARY_PARTICLE_PROVENANCE::LOADED_SOURCE_RECORD,
            "loaded provenance is recorded");
      energy_to_source[static_cast<int>(particles[index]->energy())] =
          fields.source_ordinal;
      unique.insert(ids[index]);
    }
  }
  check(energy_to_source[11] == 1u && energy_to_source[33] == 3u &&
            energy_to_source[55] == 5u,
        "reverse-pop transfer keeps each particle's original source line");
  check(energy_to_source.count(0) == 0u && energy_to_source.count(44) == 0u,
        "bad and out-of-window loaded records never receive primary IDs");
  check(unique.size() == loaded, "loaded primary IDs are unique");
}

void test_strict_alignment_guard() {
  TRIDVECTOR polarization;
  polarization = 0.0;
  RNDM random(99887u);
  PARTICLE_BEAM beam(4, 0, polarization, &random, 1u);
  initialize_generated(beam, 16u, 1.0f, 10.0f, 0);
  check(beam.validate_particle_id_sidecar(),
        "valid generated sidecar passes strict shape/uniqueness audit");

  std::vector<PARTICLE*>& exposed_legacy_vector = beam.getParticleVector(0);
  PARTICLE* extra = new PARTICLE();
  exposed_legacy_vector.push_back(extra);
  std::string reason;
  check(!beam.validate_particle_id_sidecar(&reason) && !reason.empty(),
        "strict audit detects legacy-vector/sidecar shape corruption");
  check(throws_as<std::logic_error>(
            [&]() { (void)beam.getParticleIdVectorStrict(0); }),
        "strict ID accessor refuses a misaligned legacy vector");
  exposed_legacy_vector.pop_back();
  delete extra;
  check(beam.validate_particle_id_sidecar(),
        "restoring legacy vector shape restores a valid sidecar");

  check(throws_as<std::logic_error>([&]() { beam.set_beam_label(2u); }),
        "beam provenance label cannot change after IDs exist");
  beam.set_beam_label(1u);
  check(throws_as<std::out_of_range>(
            [&]() { (void)beam.getParticleIdVectorStrict(99); }),
        "strict ID accessor bounds-checks the slice");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: test_primary_particle_ids LOADED_FIXTURE\n";
    return 2;
  }

  test_schema_known_answers_and_bounds();
  test_ordinary_attempted_ordinals();
  test_symmetric_generator_ordinal_and_lane();
  test_slicing_stability();
  test_loaded_record_ordinals(argv[1]);
  test_strict_alignment_guard();

  if (failures != 0) {
    std::cerr << failures << " primary-particle ID checks failed\n";
    return 1;
  }
  std::cout << "primary particle IDs: PASS\n";
  return 0;
}
