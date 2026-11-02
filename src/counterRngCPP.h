#ifndef FEEBL_COUNTER_RNG_CPP_H
#define FEEBL_COUNTER_RNG_CPP_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// A small, dependency-free Philox4x32-10 implementation for schedule-
// independent CPU/GPU random streams.  The algorithm and constants are from
// Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC11).
//
// This does not replace the legacy RNDM generator.  It is the reference engine
// for accelerated backends, where a draw must be a pure function of physical
// event identity instead of the order in which worker threads happen to run.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FEEBL_COUNTER_HD __host__ __device__
#else
#define FEEBL_COUNTER_HD
#endif

namespace feebl {

enum class RandomDomain : std::uint32_t {
  // Values 1--11 are the original V1 assignments.  They are part of the
  // replay format and must never be renumbered or reused.
  BeamLoading = 1,
  EnergySpread = 2,
  ISR = 3,
  Beamstrahlung = 4,
  SpinFlip = 5,
  PhotonSampling = 6,
  PairProduction = 7,
  TridentProduction = 8,
  BhabhaProduction = 9,
  JetProduction = 10,
  Diagnostics = 11,

  // Process-specific V2 assignments.  This is an append-only registry: a
  // retired process keeps its number and new processes receive new numbers.
  BeamTransversePhaseSpace = 12,
  BeamLongitudinalPhaseSpace = 13,
  BeamAngularDivergence = 14,
  BeamEnergyTail = 15,
  BeamPolarizationInitialization = 16,
  BeamParticleShuffle = 17,
  EquivalentPhotonSampling = 18,
  PhotonMacroparticleSampling = 19,
  LuminosityEeSampling = 20,
  LuminosityEgSampling = 21,
  LuminosityGeSampling = 22,
  LuminosityGgSampling = 23,
  IncoherentPairProduction = 24,
  CoherentPairProduction = 25,
  TridentVirtualPhoton = 26,
  TridentConversion = 27,
  BreitWheelerPairProduction = 28,
  BetheHeitlerPairProduction = 29,
  LandauLifshitzPairProduction = 30,
  MuonPairProduction = 31,
  ComptonProduction = 32,
  BhabhaEventSelection = 33,
  BhabhaPhotonSelection = 34,
  HadronProduction = 35,
  HadronVertexSampling = 36,
  MinijetProduction = 37,
  SecondarySynchrotronRadiation = 38,
  TertiaryPhotonProduction = 39,
  PairMacroparticleSampling = 40,
  MuonMacroparticleSampling = 41,
  CoherentEnergySampling = 42,
  TridentEnergySampling = 43,
  LuminosityEventSampling = 44,
  DiagnosticSampling = 45,
  DiagnosticParticleSampling = 46,
  OutputSampling = 47,
  OutputShuffle = 48,
  LoadedSampleSelection = 49,
  BackgroundEventSampling = 50
};

// Every field is a stable, canonical identifier.  worker/thread/device IDs do
// not belong here: changing execution resources must not change the stream.
struct RandomKey {
  std::uint64_t run_seed = 1;
  RandomDomain domain = RandomDomain::Diagnostics;
  std::uint32_t beam = 0;
  std::uint32_t overlap = 0;
  std::uint32_t timestep = 0;
  std::uint32_t slice = 0;
  std::uint64_t entity = 0;       // particle, pair, or cell-pair stable ID
  std::uint64_t interaction = 0;  // emission/collision/secondary ordinal
};

struct PhiloxBlock {
  std::uint32_t value[4];
};

class Philox4x32_10 {
 public:
  FEEBL_COUNTER_HD static PhiloxBlock eval(PhiloxBlock counter,
                                           std::uint32_t key0,
                                           std::uint32_t key1) {
    for (int round = 0; round < 10; ++round) {
      const std::uint32_t lo0 = multiply_low(kMul0, counter.value[0]);
      const std::uint32_t hi0 = multiply_high(kMul0, counter.value[0]);
      const std::uint32_t lo1 = multiply_low(kMul1, counter.value[2]);
      const std::uint32_t hi1 = multiply_high(kMul1, counter.value[2]);

      PhiloxBlock next = {{
          hi1 ^ counter.value[1] ^ key0,
          lo1,
          hi0 ^ counter.value[3] ^ key1,
          lo0
      }};
      counter = next;
      key0 += kWeyl0;
      key1 += kWeyl1;
    }
    return counter;
  }

 private:
  static constexpr std::uint32_t kMul0 = 0xD2511F53u;
  static constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
  static constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
  static constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;

  FEEBL_COUNTER_HD static std::uint32_t multiply_low(std::uint32_t a,
                                                     std::uint32_t b) {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(a) * b);
  }

  FEEBL_COUNTER_HD static std::uint32_t multiply_high(std::uint32_t a,
                                                      std::uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __umulhi(a, b);
#else
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(a) * b) >> 32);
#endif
  }
};

class CounterRng {
 public:
  FEEBL_COUNTER_HD static std::uint32_t uint32(const RandomKey& key,
                                               std::uint64_t draw_index) {
    const std::uint64_t block_index = draw_index >> 2;
    const unsigned lane = static_cast<unsigned>(draw_index & 3u);
    const std::uint64_t stream = stream_key(key);
    PhiloxBlock counter = {{
        static_cast<std::uint32_t>(key.entity),
        static_cast<std::uint32_t>(key.entity >> 32),
        static_cast<std::uint32_t>(block_index),
        static_cast<std::uint32_t>(block_index >> 32)
    }};
    const PhiloxBlock block = Philox4x32_10::eval(
        counter, static_cast<std::uint32_t>(stream),
        static_cast<std::uint32_t>(stream >> 32));
    return block.value[lane];
  }

  // Open interval (0,1), with 24 random mantissa bits.  Avoiding exact zero is
  // important for logarithms in rejection and Box-Muller samplers.
  FEEBL_COUNTER_HD static float uniform_float(const RandomKey& key,
                                              std::uint64_t draw_index) {
    const std::uint32_t mantissa = uint32(key, draw_index) >> 8;
    return (static_cast<float>(mantissa) + 0.5f) * 0x1.0p-24f;
  }

  // Open interval (0,1), with 53 random mantissa bits.
  FEEBL_COUNTER_HD static double uniform_double(const RandomKey& key,
                                                std::uint64_t draw_index) {
    const std::uint64_t a = uint32(key, draw_index * 2u) >> 5;
    const std::uint64_t b = uint32(key, draw_index * 2u + 1u) >> 6;
    const std::uint64_t mantissa = (a << 26) | b;
    return (static_cast<double>(mantissa) + 0.5) * 0x1.0p-53;
  }

  // Stateless Box-Muller. normal_index identifies a pair-independent normal
  // draw and therefore has no hidden cache whose value depends on call order.
  FEEBL_COUNTER_HD static double normal(const RandomKey& key,
                                        std::uint64_t normal_index) {
    const double u1 = uniform_double(key, normal_index * 2u);
    const double u2 = uniform_double(key, normal_index * 2u + 1u);
    const double radius = ::sqrt(-2.0 * ::log(u1));
    return radius * ::cos(kTwoPi * u2);
  }

 private:
  static constexpr double kTwoPi = 6.283185307179586476925286766559;

  FEEBL_COUNTER_HD static std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }

  FEEBL_COUNTER_HD static std::uint64_t combine(std::uint64_t state,
                                                std::uint64_t word) {
    return splitmix64(state ^ splitmix64(word));
  }

  FEEBL_COUNTER_HD static std::uint64_t stream_key(const RandomKey& key) {
    std::uint64_t state = splitmix64(key.run_seed);
    state = combine(state, static_cast<std::uint32_t>(key.domain));
    state = combine(state, key.beam);
    state = combine(state, key.overlap);
    state = combine(state, key.timestep);
    state = combine(state, key.slice);
    state = combine(state, key.interaction);
    return state;
  }
};

// ---------------------------------------------------------------------------
// Physical random identity schema V2
// ---------------------------------------------------------------------------
//
// V1 above is intentionally frozen: changing its mapping would invalidate its
// published known-answer values and the forensic replay stream.  V2 is the
// production identity scheme for the modern encounter engine.
//
// The fields below are fixed-width integer *values*, not a serialized struct.
// identity_digest() absorbs them one at a time in the documented order, using
// explicit shifts and fixed constants.  Consequently padding, host byte order,
// object layout, std::hash, pointers, workers and device IDs cannot affect the
// result.  StableParticleId and EncounterId in modernTypesCPP.h are uint64_t,
// so parent_a/parent_b and encounter are representation-compatible without
// coupling this dependency-free header to modern-core storage classes.
//
// Mapping contract (schema 2):
//   root -> schema -> seed/domain -> crossing/encounter hierarchy
//        -> beam -> ordered parents -> interaction/emission/rejection.
// The resulting 128-bit digest supplies 64 Philox key bits and 64 counter bits;
// the remaining 64 counter bits hold floor(draw/4).  The lane is draw mod 4.
// Under the standard random-mapping model, the probability that any two of N
// distinct physical identities alias is at most N(N-1)/2^129 (about 1.5e-15
// even for N=10^12).  Qualification traces must nevertheless compare the full
// logical tuple when diagnosing a duplicate digest.
//
// All identity fields accept their complete fixed-width range.  uint32() also
// accepts every uint64_t draw index.  uniform_double() accepts indices through
// (UINT64_MAX-1)/2, and normal() through (UINT64_MAX-3)/4; exceeding those
// ranges throws on the host and asserts/traps in device debug builds.  A
// rejection attempt may equal UINT64_MAX, but advancing it further is an
// error--try_advance_rejection_attempt_v2 reports false and the checked helper
// fails loudly.  No ordinal is ever permitted to wrap.

constexpr std::uint32_t kPhysicalRandomKeySchemaV2 = 2u;
constexpr std::uint64_t kMaxCounterOrdinalV2 = 0xffffffffffffffffull;
constexpr std::uint64_t kNoParentStableIdV2 = kMaxCounterOrdinalV2;

FEEBL_COUNTER_HD inline std::uint32_t random_domain_number(
    RandomDomain domain) {
  return static_cast<std::uint32_t>(domain);
}

FEEBL_COUNTER_HD inline bool random_domain_is_valid_v2(RandomDomain domain) {
  const std::uint32_t number = random_domain_number(domain);
  return number >= random_domain_number(RandomDomain::BeamLoading) &&
         number <= random_domain_number(RandomDomain::BackgroundEventSampling);
}

FEEBL_COUNTER_HD inline bool random_domain_is_diagnostic_v2(
    RandomDomain domain) {
  return domain == RandomDomain::Diagnostics ||
         domain == RandomDomain::DiagnosticSampling ||
         domain == RandomDomain::DiagnosticParticleSampling ||
         domain == RandomDomain::OutputSampling ||
         domain == RandomDomain::OutputShuffle;
}

struct PhysicalRandomKeyV2 {
  std::uint32_t schema_version = kPhysicalRandomKeySchemaV2;
  std::uint64_t run_seed = 1;
  RandomDomain domain = RandomDomain::Diagnostics;

  std::uint64_t crossing = 0;
  std::uint64_t diagonal = 0;
  std::uint64_t wave = 0;
  std::uint64_t substep = 0;
  std::uint64_t encounter = 0;

  // Zero denotes a run-global/no-beam process.  FeeBL beams normally use 1
  // and 2, but the full uint32_t range is encoded without truncation.
  std::uint32_t beam = 0;

  // Parent order is significant.  Unary processes use kNoParentStableIdV2 for
  // parent_b.  Only a physically symmetric process may call the explicit
  // canonical_symmetric_pair_key_v2() helper.
  std::uint64_t parent_a = kNoParentStableIdV2;
  std::uint64_t parent_b = kNoParentStableIdV2;

  std::uint64_t interaction = 0;
  std::uint64_t emission = 0;
  std::uint64_t rejection_attempt = 0;
};

FEEBL_COUNTER_HD inline bool physical_random_key_is_valid_v2(
    const PhysicalRandomKeyV2& key) {
  return key.schema_version == kPhysicalRandomKeySchemaV2 &&
         random_domain_is_valid_v2(key.domain);
}

FEEBL_COUNTER_HD inline PhysicalRandomKeyV2
canonical_symmetric_pair_key_v2(PhysicalRandomKeyV2 key) {
  if (key.parent_b < key.parent_a) {
    const std::uint64_t temporary = key.parent_a;
    key.parent_a = key.parent_b;
    key.parent_b = temporary;
  }
  return key;
}

struct PhysicalRandomDigestV2 {
  std::uint64_t low;
  std::uint64_t high;
};

namespace counter_rng_v2_detail {

FEEBL_COUNTER_HD inline std::uint64_t rotate_left(std::uint64_t value,
                                                   unsigned shift) {
  shift &= 63u;
  return shift == 0u ? value
                     : ((value << shift) | (value >> (64u - shift)));
}

// SplitMix64's bijective 64-bit finalizer.  This is an integer permutation;
// every operation has defined unsigned wrap semantics.
FEEBL_COUNTER_HD inline std::uint64_t avalanche(std::uint64_t value) {
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31);
}

// A field tag is mixed independently into both lanes before the lanes are
// cross-coupled.  Tags are schema constants, not native byte encodings.
FEEBL_COUNTER_HD inline void absorb(std::uint64_t& low, std::uint64_t& high,
                                    std::uint64_t field_tag,
                                    std::uint64_t word) {
  const std::uint64_t tagged_low = avalanche(
      word ^ (0x9E3779B97F4A7C15ull * field_tag));
  const std::uint64_t tagged_high = avalanche(
      word ^ (0xD1B54A32D192ED03ull * field_tag));
  const std::uint64_t old_low = low;
  const std::uint64_t old_high = high;
  low = avalanche(old_low ^ tagged_low ^ rotate_left(old_high, 17u));
  high = avalanche(old_high ^ tagged_high ^ rotate_left(old_low, 41u));
}

FEEBL_COUNTER_HD inline PhysicalRandomDigestV2 identity_digest_unchecked(
    const PhysicalRandomKeyV2& key) {
  std::uint64_t low = 0x243F6A8885A308D3ull;
  std::uint64_t high = 0x13198A2E03707344ull;
  absorb(low, high, 1u, key.schema_version);
  absorb(low, high, 2u, key.run_seed);
  absorb(low, high, 3u, random_domain_number(key.domain));
  absorb(low, high, 4u, key.crossing);
  absorb(low, high, 5u, key.diagonal);
  absorb(low, high, 6u, key.wave);
  absorb(low, high, 7u, key.substep);
  absorb(low, high, 8u, key.encounter);
  absorb(low, high, 9u, key.beam);
  absorb(low, high, 10u, key.parent_a);
  absorb(low, high, 11u, key.parent_b);
  absorb(low, high, 12u, key.interaction);
  absorb(low, high, 13u, key.emission);
  absorb(low, high, 14u, key.rejection_attempt);
  return PhysicalRandomDigestV2{low, high};
}

FEEBL_COUNTER_HD inline void require(bool condition) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  assert(condition);
#else
  if (!condition)
    throw std::out_of_range("FeeBL counter-RNG V2 identity overflow/invalid key");
#endif
}

// Convert integer mantissas to open-unit variates while preserving the
// historical midpoint mapping for every ordinary input. For the single
// all-ones mantissa, the exact midpoint lies halfway between the largest
// representable value below one and one; round-to-nearest-even would otherwise
// return exactly 1. Clamp only that rounded endpoint so logarithmic and
// half-open-cell consumers can rely on the documented 0 < U < 1 contract.
FEEBL_COUNTER_HD inline float open_unit_float_from_mantissa(
    std::uint32_t mantissa) {
  require(mantissa < (std::uint32_t{1} << 24));
  const float candidate =
      (static_cast<float>(mantissa) + 0.5f) * 0x1.0p-24f;
  return candidate < 1.0f ? candidate : 0x1.fffffep-1f;
}

FEEBL_COUNTER_HD inline double open_unit_double_from_mantissa(
    std::uint64_t mantissa) {
  require(mantissa < (std::uint64_t{1} << 53));
  const double candidate =
      (static_cast<double>(mantissa) + 0.5) * 0x1.0p-53;
  return candidate < 1.0 ? candidate : 0x1.fffffffffffffp-1;
}

}  // namespace counter_rng_v2_detail

class CounterRngV2 {
 public:
  static constexpr std::uint64_t kMaxUniformDoubleIndex =
      (kMaxCounterOrdinalV2 - 1u) / 2u;
  static constexpr std::uint64_t kMaxNormalIndex =
      (kMaxCounterOrdinalV2 - 3u) / 4u;

  FEEBL_COUNTER_HD static PhysicalRandomDigestV2 identity_digest(
      const PhysicalRandomKeyV2& key) {
    counter_rng_v2_detail::require(physical_random_key_is_valid_v2(key));
    return counter_rng_v2_detail::identity_digest_unchecked(key);
  }

  FEEBL_COUNTER_HD static std::uint32_t uint32(
      const PhysicalRandomKeyV2& key, std::uint64_t draw_index) {
    counter_rng_v2_detail::require(physical_random_key_is_valid_v2(key));
    return uint32_unchecked(key, draw_index);
  }

  FEEBL_COUNTER_HD static float uniform_float(
      const PhysicalRandomKeyV2& key, std::uint64_t draw_index) {
    const std::uint32_t mantissa = uint32(key, draw_index) >> 8;
    return counter_rng_v2_detail::open_unit_float_from_mantissa(mantissa);
  }

  FEEBL_COUNTER_HD static double uniform_double(
      const PhysicalRandomKeyV2& key, std::uint64_t sample_index) {
    counter_rng_v2_detail::require(sample_index <= kMaxUniformDoubleIndex);
    counter_rng_v2_detail::require(physical_random_key_is_valid_v2(key));
    const std::uint64_t first_draw = sample_index * 2u;
    const std::uint64_t a = uint32_unchecked(key, first_draw) >> 5;
    const std::uint64_t b = uint32_unchecked(key, first_draw + 1u) >> 6;
    const std::uint64_t mantissa = (a << 26) | b;
    return counter_rng_v2_detail::open_unit_double_from_mantissa(mantissa);
  }

  FEEBL_COUNTER_HD static double normal(const PhysicalRandomKeyV2& key,
                                        std::uint64_t normal_index) {
    counter_rng_v2_detail::require(normal_index <= kMaxNormalIndex);
    counter_rng_v2_detail::require(physical_random_key_is_valid_v2(key));
    const std::uint64_t first_sample = normal_index * 2u;
    const double u1 = uniform_double(key, first_sample);
    const double u2 = uniform_double(key, first_sample + 1u);
    const double radius = ::sqrt(-2.0 * ::log(u1));
    return radius * ::cos(kTwoPi * u2);
  }

 private:
  static constexpr double kTwoPi = 6.283185307179586476925286766559;

  FEEBL_COUNTER_HD static std::uint32_t uint32_unchecked(
      const PhysicalRandomKeyV2& key, std::uint64_t draw_index) {
    const std::uint64_t block_index = draw_index >> 2;
    const unsigned lane = static_cast<unsigned>(draw_index & 3u);
    const PhysicalRandomDigestV2 digest =
        counter_rng_v2_detail::identity_digest_unchecked(key);
    PhiloxBlock counter = {{
        static_cast<std::uint32_t>(digest.low),
        static_cast<std::uint32_t>(digest.low >> 32),
        static_cast<std::uint32_t>(block_index),
        static_cast<std::uint32_t>(block_index >> 32)}};
    const PhiloxBlock block = Philox4x32_10::eval(
        counter, static_cast<std::uint32_t>(digest.high),
        static_cast<std::uint32_t>(digest.high >> 32));
    return block.value[lane];
  }
};

FEEBL_COUNTER_HD inline bool try_advance_rejection_attempt_v2(
    PhysicalRandomKeyV2& key) {
  if (key.rejection_attempt == kMaxCounterOrdinalV2)
    return false;
  ++key.rejection_attempt;
  return true;
}

FEEBL_COUNTER_HD inline void advance_rejection_attempt_v2(
    PhysicalRandomKeyV2& key) {
  counter_rng_v2_detail::require(try_advance_rejection_attempt_v2(key));
}

// Generated stable IDs are a separate, fixed namespace derived only from the
// ordered parents, process domain, generation and emission ordinal.  They do
// not depend on seed, scheduling, encounter or diagnostics.  The 64-bit fold
// has birthday bound M(M-1)/2^65 (2.7e-8 for M=10^6 generated objects in one
// checked namespace).  Because 64 bits cannot injectively encode two arbitrary
// 64-bit parents plus ordinals, every population insertion MUST compare IDs in
// an exact uniqueness table and abort on a duplicate; a collision may never be
// resolved by schedule-dependent probing.  kNoParentStableIdV2 is reserved and
// is never returned.  Generation/ordinal bounds are checked before mixing, so
// truncation and wrap are impossible.
constexpr std::uint64_t kMaxGeneratedIdGenerationV2 = 0xffffffffull;
constexpr std::uint64_t kMaxGeneratedIdEmissionV2 = 0x0000ffffffffffffull;
constexpr std::uint64_t kMaxGeneratedIdsPerCheckedNamespaceV2 = 1000000ull;

struct GeneratedChildIdSpecV2 {
  std::uint64_t parent_a = kNoParentStableIdV2;
  std::uint64_t parent_b = kNoParentStableIdV2;
  RandomDomain process = RandomDomain::PairProduction;
  std::uint64_t generation = 0;
  std::uint64_t emission_ordinal = 0;
  bool symmetric_parents = false;
};

enum class GeneratedChildIdStatusV2 : std::uint8_t {
  Success = 0,
  NullOutput = 1,
  MissingPrimaryParent = 2,
  InvalidProcessDomain = 3,
  GenerationOutOfRange = 4,
  EmissionOutOfRange = 5,
  ReservedIdGenerated = 6
};

FEEBL_COUNTER_HD inline GeneratedChildIdStatusV2
try_make_generated_child_id_v2(const GeneratedChildIdSpecV2& specification,
                               std::uint64_t* child_id) {
  if (child_id == nullptr) return GeneratedChildIdStatusV2::NullOutput;
  if (specification.parent_a == kNoParentStableIdV2)
    return GeneratedChildIdStatusV2::MissingPrimaryParent;
  if (!random_domain_is_valid_v2(specification.process) ||
      random_domain_is_diagnostic_v2(specification.process))
    return GeneratedChildIdStatusV2::InvalidProcessDomain;
  if (specification.generation > kMaxGeneratedIdGenerationV2)
    return GeneratedChildIdStatusV2::GenerationOutOfRange;
  if (specification.emission_ordinal > kMaxGeneratedIdEmissionV2)
    return GeneratedChildIdStatusV2::EmissionOutOfRange;

  PhysicalRandomKeyV2 key;
  key.run_seed = 0x4348494c44494432ull;  // ASCII "CHILDID2"
  key.domain = specification.process;
  key.parent_a = specification.parent_a;
  key.parent_b = specification.parent_b;
  if (specification.symmetric_parents && key.parent_b < key.parent_a) {
    const std::uint64_t temporary = key.parent_a;
    key.parent_a = key.parent_b;
    key.parent_b = temporary;
  }
  key.interaction = specification.generation;
  key.emission = specification.emission_ordinal;

  const PhysicalRandomDigestV2 digest =
      counter_rng_v2_detail::identity_digest_unchecked(key);
  const std::uint64_t folded = counter_rng_v2_detail::avalanche(
      digest.low ^ counter_rng_v2_detail::rotate_left(digest.high, 29u) ^
      0xA0761D6478BD642Full);
  if (folded == kNoParentStableIdV2)
    return GeneratedChildIdStatusV2::ReservedIdGenerated;
  *child_id = folded;
  return GeneratedChildIdStatusV2::Success;
}

inline std::uint64_t make_generated_child_id_v2(
    const GeneratedChildIdSpecV2& specification) {
  std::uint64_t child_id = kNoParentStableIdV2;
  const GeneratedChildIdStatusV2 status =
      try_make_generated_child_id_v2(specification, &child_id);
  if (status == GeneratedChildIdStatusV2::GenerationOutOfRange ||
      status == GeneratedChildIdStatusV2::EmissionOutOfRange)
    throw std::overflow_error("FeeBL generated stable-ID ordinal overflow");
  if (status != GeneratedChildIdStatusV2::Success)
    throw std::invalid_argument("FeeBL invalid generated stable-ID identity");
  return child_id;
}

// A convenient local stream for one physical entity.  It is safe to create one
// per particle/event because its cursor is never shared between workers.
class CounterStream {
 public:
  FEEBL_COUNTER_HD explicit CounterStream(const RandomKey& key,
                                          std::uint64_t first_draw = 0)
      : key_(key), draw_(first_draw) {}

  FEEBL_COUNTER_HD std::uint32_t next_uint32() {
    return CounterRng::uint32(key_, draw_++);
  }

  FEEBL_COUNTER_HD float uniform_float() {
    return CounterRng::uniform_float(key_, draw_++);
  }

  FEEBL_COUNTER_HD double uniform_double() {
    return CounterRng::uniform_double(key_, draw_++);
  }

  FEEBL_COUNTER_HD std::uint64_t draw_index() const { return draw_; }
  FEEBL_COUNTER_HD void skip(std::uint64_t draws) { draw_ += draws; }

 private:
  RandomKey key_;
  std::uint64_t draw_;
};

}  // namespace feebl

#undef FEEBL_COUNTER_HD

#endif
