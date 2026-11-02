#include "counterRngCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void check_u32(std::uint32_t got, std::uint32_t expected,
               const char* message) {
  if (got != expected) {
    std::cerr << "FAIL: " << message << ": got=0x" << std::hex << got
              << " expected=0x" << expected << std::dec << '\n';
    ++failures;
  }
}

void check_u64(std::uint64_t got, std::uint64_t expected,
               const char* message) {
  if (got != expected) {
    std::cerr << "FAIL: " << message << ": got=0x" << std::hex << got
              << " expected=0x" << expected << std::dec << '\n';
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

feebl::PhysicalRandomKeyV2 known_key() {
  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = 0x0123456789abcdefull;
  key.domain = feebl::RandomDomain::Beamstrahlung;
  key.crossing = 0x1020304050607080ull;
  key.diagonal = 0x1112131415161718ull;
  key.wave = 0x2122232425262728ull;
  key.substep = 0x3132333435363738ull;
  key.encounter = 0x4142434445464748ull;
  key.beam = 2u;
  key.parent_a = 0x5152535455565758ull;
  key.parent_b = 0x6162636465666768ull;
  key.interaction = 0x7172737475767778ull;
  key.emission = 0x8182838485868788ull;
  key.rejection_attempt = 0x9192939495969798ull;
  return key;
}

const std::array<feebl::RandomDomain, 50> kAllDomains = {{
    feebl::RandomDomain::BeamLoading,
    feebl::RandomDomain::EnergySpread,
    feebl::RandomDomain::ISR,
    feebl::RandomDomain::Beamstrahlung,
    feebl::RandomDomain::SpinFlip,
    feebl::RandomDomain::PhotonSampling,
    feebl::RandomDomain::PairProduction,
    feebl::RandomDomain::TridentProduction,
    feebl::RandomDomain::BhabhaProduction,
    feebl::RandomDomain::JetProduction,
    feebl::RandomDomain::Diagnostics,
    feebl::RandomDomain::BeamTransversePhaseSpace,
    feebl::RandomDomain::BeamLongitudinalPhaseSpace,
    feebl::RandomDomain::BeamAngularDivergence,
    feebl::RandomDomain::BeamEnergyTail,
    feebl::RandomDomain::BeamPolarizationInitialization,
    feebl::RandomDomain::BeamParticleShuffle,
    feebl::RandomDomain::EquivalentPhotonSampling,
    feebl::RandomDomain::PhotonMacroparticleSampling,
    feebl::RandomDomain::LuminosityEeSampling,
    feebl::RandomDomain::LuminosityEgSampling,
    feebl::RandomDomain::LuminosityGeSampling,
    feebl::RandomDomain::LuminosityGgSampling,
    feebl::RandomDomain::IncoherentPairProduction,
    feebl::RandomDomain::CoherentPairProduction,
    feebl::RandomDomain::TridentVirtualPhoton,
    feebl::RandomDomain::TridentConversion,
    feebl::RandomDomain::BreitWheelerPairProduction,
    feebl::RandomDomain::BetheHeitlerPairProduction,
    feebl::RandomDomain::LandauLifshitzPairProduction,
    feebl::RandomDomain::MuonPairProduction,
    feebl::RandomDomain::ComptonProduction,
    feebl::RandomDomain::BhabhaEventSelection,
    feebl::RandomDomain::BhabhaPhotonSelection,
    feebl::RandomDomain::HadronProduction,
    feebl::RandomDomain::HadronVertexSampling,
    feebl::RandomDomain::MinijetProduction,
    feebl::RandomDomain::SecondarySynchrotronRadiation,
    feebl::RandomDomain::TertiaryPhotonProduction,
    feebl::RandomDomain::PairMacroparticleSampling,
    feebl::RandomDomain::MuonMacroparticleSampling,
    feebl::RandomDomain::CoherentEnergySampling,
    feebl::RandomDomain::TridentEnergySampling,
    feebl::RandomDomain::LuminosityEventSampling,
    feebl::RandomDomain::DiagnosticSampling,
    feebl::RandomDomain::DiagnosticParticleSampling,
    feebl::RandomDomain::OutputSampling,
    feebl::RandomDomain::OutputShuffle,
    feebl::RandomDomain::LoadedSampleSelection,
    feebl::RandomDomain::BackgroundEventSampling,
}};

void test_v1_stays_frozen() {
  feebl::PhiloxBlock zero = {{0u, 0u, 0u, 0u}};
  const feebl::PhiloxBlock got = feebl::Philox4x32_10::eval(zero, 0u, 0u);
  check_u32(got.value[0], 0x6627e8d5u, "V1 Philox KAT lane 0");
  check_u32(got.value[1], 0xe169c58du, "V1 Philox KAT lane 1");
  check_u32(got.value[2], 0xbc57ac4cu, "V1 Philox KAT lane 2");
  check_u32(got.value[3], 0x9b00dbd8u, "V1 Philox KAT lane 3");
}

void test_v2_known_answers() {
  const feebl::PhysicalRandomKeyV2 key = known_key();
  const feebl::PhysicalRandomDigestV2 digest =
      feebl::CounterRngV2::identity_digest(key);

  // Frozen schema-2 values.  A deliberate schema change requires new symbols,
  // new vectors and a migration note--never an in-place update of these values.
  check_u64(digest.low, 0xe49beedd2b347487ull,
            "V2 identity digest low KAT");
  check_u64(digest.high, 0x157f2b3ffd5d764aull,
            "V2 identity digest high KAT");
  const std::array<std::uint32_t, 8> expected = {{
      0x5e95f333u, 0xbb6b259bu, 0xb42b996bu, 0x926a76fcu,
      0x2a91cb34u, 0x7f912796u, 0xd3725fc8u, 0x8392c081u,
  }};
  for (std::size_t i = 0; i < expected.size(); ++i)
    check_u32(feebl::CounterRngV2::uint32(key, i), expected[i],
              "V2 uint32 KAT");

  // This vector checks both high draw-counter words and lane selection.
  check_u32(feebl::CounterRngV2::uint32(key, 0xfedcba9876543210ull),
            0xce4fe800u, "V2 high draw-index KAT");
}

void test_domain_registry_and_separation() {
  std::set<std::pair<std::uint64_t, std::uint64_t>> digests;
  for (std::size_t i = 0; i < kAllDomains.size(); ++i) {
    const feebl::RandomDomain domain = kAllDomains[i];
    check(feebl::random_domain_number(domain) == i + 1,
          "domain registry must be contiguous and permanently numbered");
    check(feebl::random_domain_is_valid_v2(domain),
          "registered domain must validate");
    feebl::PhysicalRandomKeyV2 key = known_key();
    key.domain = domain;
    const feebl::PhysicalRandomDigestV2 digest =
        feebl::CounterRngV2::identity_digest(key);
    check(digests.emplace(digest.low, digest.high).second,
          "domain-separated identity digest alias");
  }
  check(!feebl::random_domain_is_valid_v2(
            static_cast<feebl::RandomDomain>(0u)),
        "domain zero is invalid");
  check(!feebl::random_domain_is_valid_v2(
            static_cast<feebl::RandomDomain>(51u)),
        "unassigned domain is invalid");
}

void test_parent_ordering() {
  feebl::PhysicalRandomKeyV2 ab = known_key();
  ab.parent_a = 101;
  ab.parent_b = 707;
  feebl::PhysicalRandomKeyV2 ba = ab;
  std::swap(ba.parent_a, ba.parent_b);

  check(feebl::CounterRngV2::uint32(ab, 0) !=
            feebl::CounterRngV2::uint32(ba, 0),
        "ordered parent reversal must change an asymmetric stream");

  const feebl::PhysicalRandomKeyV2 canonical_ab =
      feebl::canonical_symmetric_pair_key_v2(ab);
  const feebl::PhysicalRandomKeyV2 canonical_ba =
      feebl::canonical_symmetric_pair_key_v2(ba);
  check(canonical_ab.parent_a == 101 && canonical_ab.parent_b == 707,
        "canonical symmetric pair is ascending");
  check(feebl::CounterRngV2::uint32(canonical_ab, 99) ==
            feebl::CounterRngV2::uint32(canonical_ba, 99),
        "explicit symmetric canonicalization must be order invariant");

  feebl::PhysicalRandomKeyV2 unary = known_key();
  unary.parent_a = 88;
  unary.parent_b = feebl::kNoParentStableIdV2;
  const feebl::PhysicalRandomKeyV2 canonical_unary =
      feebl::canonical_symmetric_pair_key_v2(unary);
  check(canonical_unary.parent_a == 88 &&
            canonical_unary.parent_b == feebl::kNoParentStableIdV2,
        "unary sentinel remains the second parent");
}

feebl::PhysicalRandomKeyV2 job_key(std::uint64_t job) {
  feebl::PhysicalRandomKeyV2 key = known_key();
  key.encounter = job * 13u + 5u;
  key.parent_a = job * 2u;
  key.parent_b = job * 2u + 1u;
  key.interaction = job % 17u;
  key.emission = job % 23u;
  return key;
}

void test_traversal_and_worker_invariance() {
  constexpr std::size_t count = 8192;
  std::vector<std::uint32_t> reference(count);
  for (std::size_t job = 0; job < count; ++job)
    reference[job] = feebl::CounterRngV2::uint32(job_key(job), 31);

  const std::array<std::size_t, 7> worker_counts = {{1, 2, 3, 4, 8, 10, 37}};
  for (std::size_t workers : worker_counts) {
    std::vector<std::uint32_t> scheduled(count, 0u);
    // Simulate a strided worker assignment with worker order reversed.  The
    // worker number deliberately never enters job_key().
    for (std::size_t reverse_worker = 0; reverse_worker < workers;
         ++reverse_worker) {
      const std::size_t worker = workers - 1u - reverse_worker;
      for (std::size_t job = worker; job < count; job += workers)
        scheduled[job] = feebl::CounterRngV2::uint32(job_key(job), 31);
    }
    check(scheduled == reference,
          "simulated worker count/order must not reassign draws");
  }

  std::vector<std::uint32_t> reverse(count, 0u);
  for (std::size_t job = count; job-- > 0;)
    reverse[job] = feebl::CounterRngV2::uint32(job_key(job), 31);
  check(reverse == reference, "reverse traversal must not reassign draws");
}

void test_max_ranges_and_overflow() {
  const std::uint32_t maximum_float_mantissa =
      (std::uint32_t{1} << 24) - 1U;
  const float float_endpoint =
      feebl::counter_rng_v2_detail::open_unit_float_from_mantissa(
          maximum_float_mantissa);
  check(float_endpoint == std::nextafter(1.0f, 0.0f),
        "all-ones float mantissa clamps below one");
  check(float_endpoint > 0.0f && float_endpoint < 1.0f,
        "uniform-float conversion is strictly open at its upper endpoint");

  const std::uint64_t maximum_mantissa =
      (std::uint64_t{1} << 53) - 1U;
  const double endpoint =
      feebl::counter_rng_v2_detail::open_unit_double_from_mantissa(
          maximum_mantissa);
  check(endpoint == std::nextafter(1.0, 0.0),
        "all-ones mantissa clamps to the largest binary64 below one");
  check(endpoint > 0.0 && endpoint < 1.0,
        "uniform-double conversion is strictly open at its upper endpoint");

  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = std::numeric_limits<std::uint64_t>::max();
  key.domain = feebl::RandomDomain::BackgroundEventSampling;
  key.crossing = std::numeric_limits<std::uint64_t>::max();
  key.diagonal = std::numeric_limits<std::uint64_t>::max();
  key.wave = std::numeric_limits<std::uint64_t>::max();
  key.substep = std::numeric_limits<std::uint64_t>::max();
  key.encounter = std::numeric_limits<std::uint64_t>::max();
  key.beam = std::numeric_limits<std::uint32_t>::max();
  key.parent_a = std::numeric_limits<std::uint64_t>::max();
  key.parent_b = std::numeric_limits<std::uint64_t>::max();
  key.interaction = std::numeric_limits<std::uint64_t>::max();
  key.emission = std::numeric_limits<std::uint64_t>::max();
  key.rejection_attempt = std::numeric_limits<std::uint64_t>::max();
  check(feebl::physical_random_key_is_valid_v2(key),
        "complete fixed-width key range is valid");
  check(feebl::CounterRngV2::uint32(
            key, std::numeric_limits<std::uint64_t>::max()) ==
            feebl::CounterRngV2::uint32(
                key, std::numeric_limits<std::uint64_t>::max()),
        "maximum draw index repeats without wrap");

  const double upper_uniform = feebl::CounterRngV2::uniform_double(
      key, feebl::CounterRngV2::kMaxUniformDoubleIndex);
  const double upper_normal = feebl::CounterRngV2::normal(
      key, feebl::CounterRngV2::kMaxNormalIndex);
  check(upper_uniform > 0.0 && upper_uniform < 1.0,
        "maximum double index is valid");
  check(std::isfinite(upper_normal), "maximum normal index is valid");

  check(throws_as<std::out_of_range>([&]() {
          (void)feebl::CounterRngV2::uniform_double(
              key, feebl::CounterRngV2::kMaxUniformDoubleIndex + 1u);
        }),
        "uniform-double index overflow must fail loudly");
  check(throws_as<std::out_of_range>([&]() {
          (void)feebl::CounterRngV2::normal(
              key, feebl::CounterRngV2::kMaxNormalIndex + 1u);
        }),
        "normal index overflow must fail loudly");

  check(!feebl::try_advance_rejection_attempt_v2(key),
        "maximum rejection attempt cannot wrap");
  check(key.rejection_attempt == std::numeric_limits<std::uint64_t>::max(),
        "failed rejection advance leaves identity unchanged");
  check(throws_as<std::out_of_range>(
            [&]() { feebl::advance_rejection_attempt_v2(key); }),
        "checked rejection overflow must fail loudly");
  key.rejection_attempt = std::numeric_limits<std::uint64_t>::max() - 1u;
  check(feebl::try_advance_rejection_attempt_v2(key) &&
            key.rejection_attempt == std::numeric_limits<std::uint64_t>::max(),
        "last rejection attempt advances exactly to maximum");

  feebl::PhysicalRandomKeyV2 invalid = key;
  invalid.schema_version = 1u;
  check(!feebl::physical_random_key_is_valid_v2(invalid),
        "wrong schema version is invalid");
  check(throws_as<std::out_of_range>(
            [&]() { (void)feebl::CounterRngV2::uint32(invalid, 0); }),
        "wrong schema version must fail loudly");
  invalid = key;
  invalid.domain = static_cast<feebl::RandomDomain>(0u);
  check(throws_as<std::out_of_range>(
            [&]() { (void)feebl::CounterRngV2::uint32(invalid, 0); }),
        "invalid domain must fail loudly");
}

void test_diagnostic_independence() {
  feebl::PhysicalRandomKeyV2 physics = known_key();
  physics.domain = feebl::RandomDomain::Beamstrahlung;
  feebl::PhysicalRandomKeyV2 diagnostic = physics;
  diagnostic.domain = feebl::RandomDomain::DiagnosticParticleSampling;

  std::array<std::uint32_t, 64> before{};
  std::array<std::uint32_t, 64> after{};
  for (std::size_t i = 0; i < before.size(); ++i)
    before[i] = feebl::CounterRngV2::uint32(physics, i);

  std::uint32_t diagnostic_sink = 0;
  for (std::uint64_t budget = 0; budget < 100000; ++budget)
    diagnostic_sink ^= feebl::CounterRngV2::uint32(diagnostic, budget);
  check(diagnostic_sink != 0u, "diagnostic stream was actually exercised");

  for (std::size_t i = 0; i < after.size(); ++i)
    after[i] = feebl::CounterRngV2::uint32(physics, i);
  check(before == after,
        "changing diagnostic sampling budget cannot move physics draws");
  check(feebl::CounterRngV2::uint32(physics, 0) !=
            feebl::CounterRngV2::uint32(diagnostic, 0),
        "diagnostic and physics domains are disjoint");
  check(feebl::random_domain_is_diagnostic_v2(diagnostic.domain),
        "diagnostic domain classification");
  check(!feebl::random_domain_is_diagnostic_v2(physics.domain),
        "physics domain classification");
}

void test_generated_child_ids() {
  feebl::GeneratedChildIdSpecV2 spec;
  spec.parent_a = 0x0123456789abcdefull;
  spec.parent_b = 0xfedcba9876543210ull;
  spec.process = feebl::RandomDomain::BreitWheelerPairProduction;
  spec.generation = 7;
  spec.emission_ordinal = 123456;

  const std::uint64_t child = feebl::make_generated_child_id_v2(spec);
  check_u64(child, 0x1fe676e3ab72eeaeull, "generated child-ID KAT");
  check(child != feebl::kNoParentStableIdV2,
        "generated child ID cannot be the parent sentinel");
  check(child == feebl::make_generated_child_id_v2(spec),
        "generated child ID repeats");

  feebl::GeneratedChildIdSpecV2 reversed = spec;
  std::swap(reversed.parent_a, reversed.parent_b);
  check(feebl::make_generated_child_id_v2(reversed) != child,
        "ordered child parents matter by default");
  spec.symmetric_parents = true;
  reversed.symmetric_parents = true;
  check(feebl::make_generated_child_id_v2(spec) ==
            feebl::make_generated_child_id_v2(reversed),
        "symmetric child parents canonicalize explicitly");

  feebl::GeneratedChildIdSpecV2 changed = spec;
  ++changed.generation;
  check(feebl::make_generated_child_id_v2(changed) !=
            feebl::make_generated_child_id_v2(spec),
        "child generation separates IDs");
  changed = spec;
  ++changed.emission_ordinal;
  check(feebl::make_generated_child_id_v2(changed) !=
            feebl::make_generated_child_id_v2(spec),
        "child emission ordinal separates IDs");
  changed = spec;
  changed.process = feebl::RandomDomain::MuonPairProduction;
  check(feebl::make_generated_child_id_v2(changed) !=
            feebl::make_generated_child_id_v2(spec),
        "child process domain separates IDs");

  feebl::GeneratedChildIdSpecV2 unary = spec;
  unary.parent_b = feebl::kNoParentStableIdV2;
  unary.symmetric_parents = false;
  check(feebl::make_generated_child_id_v2(unary) !=
            feebl::kNoParentStableIdV2,
        "unary child uses the sentinel only as unused parent");

  feebl::GeneratedChildIdSpecV2 maximum = unary;
  maximum.generation = feebl::kMaxGeneratedIdGenerationV2;
  maximum.emission_ordinal = feebl::kMaxGeneratedIdEmissionV2;
  check(feebl::make_generated_child_id_v2(maximum) !=
            feebl::kNoParentStableIdV2,
        "maximum child ordinals are supported");

  std::uint64_t output = 0;
  feebl::GeneratedChildIdSpecV2 invalid = unary;
  invalid.generation = feebl::kMaxGeneratedIdGenerationV2 + 1u;
  check(feebl::try_make_generated_child_id_v2(invalid, &output) ==
            feebl::GeneratedChildIdStatusV2::GenerationOutOfRange,
        "child generation overflow is reported");
  check(throws_as<std::overflow_error>(
            [&]() { (void)feebl::make_generated_child_id_v2(invalid); }),
        "checked child generation overflow throws");
  invalid = unary;
  invalid.emission_ordinal = feebl::kMaxGeneratedIdEmissionV2 + 1u;
  check(feebl::try_make_generated_child_id_v2(invalid, &output) ==
            feebl::GeneratedChildIdStatusV2::EmissionOutOfRange,
        "child emission overflow is reported");
  invalid = unary;
  invalid.parent_a = feebl::kNoParentStableIdV2;
  check(feebl::try_make_generated_child_id_v2(invalid, &output) ==
            feebl::GeneratedChildIdStatusV2::MissingPrimaryParent,
        "missing primary parent is rejected");
  invalid = unary;
  invalid.process = static_cast<feebl::RandomDomain>(0u);
  check(feebl::try_make_generated_child_id_v2(invalid, &output) ==
            feebl::GeneratedChildIdStatusV2::InvalidProcessDomain,
        "invalid child process is rejected");
  invalid = unary;
  invalid.process = feebl::RandomDomain::DiagnosticSampling;
  check(feebl::try_make_generated_child_id_v2(invalid, &output) ==
            feebl::GeneratedChildIdStatusV2::InvalidProcessDomain,
        "diagnostics cannot create physics stable IDs");
  check(feebl::try_make_generated_child_id_v2(unary, nullptr) ==
            feebl::GeneratedChildIdStatusV2::NullOutput,
        "null child output is rejected");

  // Exercise the documented uniqueness check on a sizable deterministic
  // namespace.  Production insertion must do the same against live IDs.
  constexpr std::uint64_t sample_count = 200000;
  std::set<std::uint64_t> ids;
  feebl::GeneratedChildIdSpecV2 many = unary;
  for (std::uint64_t ordinal = 0; ordinal < sample_count; ++ordinal) {
    many.emission_ordinal = ordinal;
    const std::uint64_t id = feebl::make_generated_child_id_v2(many);
    check(id != feebl::kNoParentStableIdV2,
          "generated child aliases reserved sentinel");
    check(ids.insert(id).second, "generated child-ID collision in test range");
  }
  check(ids.size() == sample_count,
        "generated child uniqueness accounting is exact");
}

}  // namespace

int main() {
  test_v1_stays_frozen();
  test_v2_known_answers();
  test_domain_registry_and_separation();
  test_parent_ordering();
  test_traversal_and_worker_invariance();
  test_max_ranges_and_overflow();
  test_diagnostic_independence();
  test_generated_child_ids();

  if (failures != 0) {
    std::cerr << failures << " counter-RNG V2 checks failed\n";
    return 1;
  }
  std::cout << "counter RNG V2: PASS (" << kAllDomains.size()
            << " domains, checked child-ID namespace)\n";
  return 0;
}
