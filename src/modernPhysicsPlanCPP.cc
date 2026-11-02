#include "modernPhysicsPlanCPP.h"

#include <algorithm>
#include <stdexcept>

namespace feebl {
namespace modern {

namespace {

constexpr std::uint64_t kKnownProcesses =
    bits(ProcessBit::Luminosity) | bits(ProcessBit::BeamBeamField) |
    bits(ProcessBit::Beamstrahlung) | bits(ProcessBit::SpinBmt) |
    bits(ProcessBit::SpinFlip) | bits(ProcessBit::PhotonTracking) |
    bits(ProcessBit::CoherentPairs) | bits(ProcessBit::IncoherentPairs) |
    bits(ProcessBit::Trident) | bits(ProcessBit::Bhabha) |
    bits(ProcessBit::Hadrons) | bits(ProcessBit::Compton) |
    bits(ProcessBit::Muons) | bits(ProcessBit::CrossingGeometry) |
    bits(ProcessBit::CrabCrossing) | bits(ProcessBit::CrabWaist) |
    bits(ProcessBit::Monochromatization) |
    bits(ProcessBit::ArbitraryGeometry);

constexpr std::uint64_t kKnownCapabilities =
    bits(CapabilityBit::ModernSoA) | bits(CapabilityBit::CellCSR) |
    bits(CapabilityBit::CounterRngV2) |
    bits(CapabilityBit::FixedReduction) |
    bits(CapabilityBit::ScalarReference) |
    bits(CapabilityBit::PersistentCpu) | bits(CapabilityBit::ResidentGpu) |
    bits(CapabilityBit::FactoredExact) |
    bits(CapabilityBit::SampledEstimator) |
    bits(CapabilityBit::AdaptiveResolution) |
    bits(CapabilityBit::MixedPrecision) |
    bits(CapabilityBit::FastReduction) |
    bits(CapabilityBit::ArbitraryEncounterDAG) |
    bits(CapabilityBit::GenericKernel) | bits(CapabilityBit::TraceOutput);

constexpr std::uint64_t kKnownQualifications =
    bits(QualificationBit::FactoredExact) |
    bits(QualificationBit::SampledEstimator) |
    bits(QualificationBit::AdaptiveResolution) |
    bits(QualificationBit::MixedPrecision) |
    bits(QualificationBit::FastReduction) |
    bits(QualificationBit::GenericKernel);

constexpr std::uint64_t kKnownStates =
    bits(StateComponentBit::PrimaryBeam) | bits(StateComponentBit::Spin) |
    bits(StateComponentBit::PhotonPool) |
    bits(StateComponentBit::SecondaryPool) |
    bits(StateComponentBit::PhotonStokes);

constexpr std::uint64_t kKnownOutputs =
    bits(OutputCapabilityBit::Summary) |
    bits(OutputCapabilityBit::LuminositySpectrum) |
    bits(OutputCapabilityBit::BeamRecords) |
    bits(OutputCapabilityBit::SpinRecords) |
    bits(OutputCapabilityBit::PhotonRecords) |
    bits(OutputCapabilityBit::SecondaryRecords) |
    bits(OutputCapabilityBit::EncounterTrace);

constexpr std::uint64_t kModernCoreCapabilities =
    bits(CapabilityBit::ModernSoA) | bits(CapabilityBit::CellCSR) |
    bits(CapabilityBit::CounterRngV2) |
    bits(CapabilityBit::FixedReduction);

constexpr std::uint64_t kBackgroundProcesses =
    bits(ProcessBit::CoherentPairs) | bits(ProcessBit::IncoherentPairs) |
    bits(ProcessBit::Trident) | bits(ProcessBit::Bhabha) |
    bits(ProcessBit::Hadrons) | bits(ProcessBit::Compton) |
    bits(ProcessBit::Muons);

constexpr std::uint64_t kSpinProcesses =
    bits(ProcessBit::SpinBmt) | bits(ProcessBit::SpinFlip);

constexpr std::uint64_t kRadiationProcesses =
    bits(ProcessBit::Beamstrahlung) | bits(ProcessBit::PhotonTracking);

ModelIdentity identity(std::initializer_list<std::uint8_t> bytes) {
  ModelIdentity result;
  result.version = 1;
  std::copy(bytes.begin(), bytes.end(), result.sha256.begin());
  return result;
}

const ModelIdentity& unit_identity() {
  // SHA-256("FeeBL.units.legacy-nm-GeV.v1")
  static const ModelIdentity value = identity({
      0xf8,0x78,0x62,0x81,0xf3,0xc0,0x07,0x16,
      0x3e,0xbd,0xed,0x4b,0x00,0xbf,0xf7,0xc2,
      0x8c,0x6e,0x0a,0x41,0x48,0x2e,0xdb,0xef,
      0xb8,0x2c,0x16,0x77,0x60,0x25,0x6f,0xd8});
  return value;
}

const ModelIdentity& physics_identity() {
  // SHA-256("FeeBL.physics.developer-5bd2117.v1")
  static const ModelIdentity value = identity({
      0x72,0x41,0xde,0xdd,0x21,0xf7,0x7c,0x08,
      0x4c,0x00,0x59,0xaa,0x0c,0xb5,0x14,0xe0,
      0x14,0x33,0x86,0xa0,0xab,0xdd,0xd7,0x06,
      0xa6,0x2f,0xe0,0xbd,0x5a,0x55,0xb0,0xf9});
  return value;
}

const ModelIdentity& observable_identity() {
  // SHA-256("FeeBL.observables.developer-5bd2117.v1")
  static const ModelIdentity value = identity({
      0x1c,0xfa,0xde,0x74,0xb8,0xb4,0xc9,0x98,
      0xc3,0x90,0x4a,0xa0,0x05,0xfe,0xd5,0xc5,
      0xb3,0x66,0x5b,0x95,0x9d,0x0a,0x47,0xc4,
      0x5b,0x85,0xcf,0xf5,0x1c,0x44,0x61,0xab});
  return value;
}

bool inert(const ModelIdentity& value) {
  return value.version == 0 ||
      std::all_of(value.sha256.begin(), value.sha256.end(),
                  [](std::uint8_t byte) { return byte == 0; });
}

PlanValidation failure(PlanErrorCode code, const char* detail) {
  return PlanValidation{code, detail};
}

bool valid(Engine value) {
  switch (value) {
    case Engine::Legacy:
    case Engine::ModernRefCpu:
    case Engine::ModernCpu:
    case Engine::ModernGpu: return true;
    default: return false;
  }
}
bool valid(Algorithm value) {
  switch (value) {
    case Algorithm::Historical:
    case Algorithm::Exact:
    case Algorithm::FactoredExact:
    case Algorithm::SampledValidated:
    case Algorithm::AdaptiveValidated: return true;
    default: return false;
  }
}
bool valid(PrecisionPolicy value) {
  return value == PrecisionPolicy::Historical || value == PrecisionPolicy::Fp64 ||
         value == PrecisionPolicy::MixedQualified;
}
bool valid(ReductionPolicy value) {
  return value == ReductionPolicy::Historical ||
         value == ReductionPolicy::Deterministic ||
         value == ReductionPolicy::FastQualified;
}
bool valid(RngSchema value) {
  return value == RngSchema::LegacySequential || value == RngSchema::CounterV2;
}
bool valid(GeometryMode value) {
  switch (value) {
    case GeometryMode::LegacyCurrent:
    case GeometryMode::HeadOn:
    case GeometryMode::CrossingAngle:
    case GeometryMode::ArbitraryEncounterTime: return true;
    default: return false;
  }
}
bool valid(KernelFamily value) {
  switch (value) {
    case KernelFamily::LegacyMonolith:
    case KernelFamily::NoSpinNoRadiation:
    case KernelFamily::SpinBmt:
    case KernelFamily::Beamstrahlung:
    case KernelFamily::FullBackground:
    case KernelFamily::GenericValidated: return true;
    default: return false;
  }
}

void set_bit(std::uint64_t& destination, std::uint64_t bit, bool enabled) {
  if (enabled) destination |= bit;
  else destination &= ~bit;
}

void append_u8(std::vector<std::uint8_t>& bytes, std::uint8_t value) {
  bytes.push_back(value);
}
void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_identity(std::vector<std::uint8_t>& bytes,
                     const ModelIdentity& value) {
  append_u32(bytes, value.version);
  bytes.insert(bytes.end(), value.sha256.begin(), value.sha256.end());
}

PhysicsPlanData legacy_data() {
  PhysicsPlanData data;
  data.unit_model = unit_identity();
  data.physics_model = physics_identity();
  data.observable_model = observable_identity();
  data.process_bits = bits(ProcessBit::Luminosity) |
                      bits(ProcessBit::BeamBeamField);
  data.required_state_bits = bits(StateComponentBit::PrimaryBeam);
  data.output_bits = bits(OutputCapabilityBit::Summary) |
                     bits(OutputCapabilityBit::LuminositySpectrum);
  return data;
}

PhysicsPlanData modern_data(Engine engine) {
  PhysicsPlanData data = legacy_data();
  data.engine = engine;
  data.algorithm = Algorithm::Exact;
  data.precision = PrecisionPolicy::Fp64;
  data.reduction = ReductionPolicy::Deterministic;
  data.rng = RngSchema::CounterV2;
  data.geometry = GeometryMode::HeadOn;
  data.kernel_family = KernelFamily::NoSpinNoRadiation;
  data.capability_bits = kModernCoreCapabilities;
  if (engine == Engine::ModernRefCpu)
    data.capability_bits |= bits(CapabilityBit::ScalarReference);
  else if (engine == Engine::ModernCpu)
    data.capability_bits |= bits(CapabilityBit::PersistentCpu);
  else if (engine == Engine::ModernGpu) {
    data.capability_bits |= bits(CapabilityBit::ResidentGpu);
    data.limits.gpu_mode = 3;
    data.limits.gpu_devices = 1;
    data.limits.device_memory_bytes = 1ull << 30;
  }
  return data;
}

PlanValidation validate_data(const PhysicsPlanData& data) {
  if (!valid(data.engine))
    return failure(PlanErrorCode::UnsupportedEngineId, "engine");
  if (!valid(data.algorithm))
    return failure(PlanErrorCode::UnsupportedAlgorithmId, "algorithm");
  if (!valid(data.precision))
    return failure(PlanErrorCode::UnsupportedPrecisionId, "precision");
  if (!valid(data.reduction))
    return failure(PlanErrorCode::UnsupportedReductionId, "reduction");
  if (!valid(data.rng))
    return failure(PlanErrorCode::UnsupportedRngId, "rng_schema");
  if (!valid(data.geometry))
    return failure(PlanErrorCode::UnsupportedGeometryId, "geometry");
  if (!valid(data.kernel_family))
    return failure(PlanErrorCode::UnsupportedKernelFamilyId, "kernel_family");

  if (inert(data.unit_model))
    return failure(PlanErrorCode::InertUnitIdentity, "unit_model");
  if (inert(data.physics_model))
    return failure(PlanErrorCode::InertPhysicsIdentity, "physics_model");
  if (inert(data.observable_model))
    return failure(PlanErrorCode::InertObservableIdentity, "observable_model");
  if (data.unit_model != unit_identity())
    return failure(PlanErrorCode::UnsupportedUnitIdentity, "unit_model");
  if (data.physics_model != physics_identity())
    return failure(PlanErrorCode::UnsupportedPhysicsIdentity, "physics_model");
  if (data.observable_model != observable_identity())
    return failure(PlanErrorCode::UnsupportedObservableIdentity,
                   "observable_model");

  if (data.process_bits & ~kKnownProcesses)
    return failure(PlanErrorCode::UnknownProcessBits, "process_bits");
  if (data.capability_bits & ~kKnownCapabilities)
    return failure(PlanErrorCode::UnknownCapabilityBits, "capability_bits");
  if (data.qualification_bits & ~kKnownQualifications)
    return failure(PlanErrorCode::UnknownQualificationBits,
                   "qualification_bits");
  if (data.required_state_bits & ~kKnownStates)
    return failure(PlanErrorCode::UnknownStateBits, "required_state_bits");
  if (data.output_bits & ~kKnownOutputs)
    return failure(PlanErrorCode::UnknownOutputBits, "output_bits");

  const PlanLimits& limits = data.limits;
  if (limits.grid_nx == 0 || limits.grid_ny == 0)
    return failure(PlanErrorCode::InvalidGridLimits, "grid_nx/grid_ny");
  if (limits.beam1_slices == 0 || limits.beam2_slices == 0)
    return failure(PlanErrorCode::InvalidSliceLimits, "beam slice counts");
  if (limits.substeps == 0)
    return failure(PlanErrorCode::InvalidSubstepLimit, "substeps");
  if (limits.batch_width == 0)
    return failure(PlanErrorCode::InvalidBatchLimit, "batch_width");
  if (limits.cpu_threads == 0 || limits.max_step_contexts == 0 ||
      limits.host_memory_bytes == 0)
    return failure(PlanErrorCode::InvalidResourceLimits,
                   "cpu/context/host-memory resource");
  if (limits.max_primary_particles == 0 ||
      ((data.required_state_bits & bits(StateComponentBit::PhotonPool)) &&
       limits.max_photons == 0) ||
      ((data.required_state_bits & bits(StateComponentBit::SecondaryPool)) &&
       limits.max_secondaries == 0))
    return failure(PlanErrorCode::InvalidCapacityLimits,
                   "particle/photon/secondary capacity");
  std::uint64_t scratch = 0;
  std::uint64_t encounters = 0;
  if (!checked_multiply(limits.grid_nx, limits.grid_ny, scratch) ||
      !checked_multiply(limits.beam1_slices, limits.beam2_slices, encounters) ||
      !checked_multiply(encounters, limits.substeps, encounters) ||
      !checked_multiply(encounters, limits.batch_width, scratch))
    return failure(PlanErrorCode::LimitArithmeticOverflow,
                   "grid or encounter resource product");

  struct QualificationPair {
    CapabilityBit capability;
    QualificationBit qualification;
    const char* name;
  };
  static const QualificationPair qualification_pairs[] = {
      {CapabilityBit::FactoredExact, QualificationBit::FactoredExact,
       "factored_exact"},
      {CapabilityBit::SampledEstimator, QualificationBit::SampledEstimator,
       "sampled_estimator"},
      {CapabilityBit::AdaptiveResolution, QualificationBit::AdaptiveResolution,
       "adaptive_resolution"},
      {CapabilityBit::MixedPrecision, QualificationBit::MixedPrecision,
       "mixed_precision"},
      {CapabilityBit::FastReduction, QualificationBit::FastReduction,
       "fast_reduction"},
      {CapabilityBit::GenericKernel, QualificationBit::GenericKernel,
       "generic_kernel"}};
  for (const QualificationPair& pair : qualification_pairs) {
    const bool capability = (data.capability_bits & bits(pair.capability)) != 0;
    const bool qualification =
        (data.qualification_bits & bits(pair.qualification)) != 0;
    if (capability && !qualification)
      return failure(PlanErrorCode::CapabilityQualificationMissing, pair.name);
    if (!capability && qualification)
      return failure(PlanErrorCode::OrphanQualification, pair.name);
  }

  if (!(data.required_state_bits & bits(StateComponentBit::PrimaryBeam)))
    return failure(PlanErrorCode::RequiredPrimaryStateMissing, "primary_beam");
  if ((data.process_bits & kSpinProcesses) &&
      !(data.required_state_bits & bits(StateComponentBit::Spin)))
    return failure(PlanErrorCode::RequiredSpinStateMissing, "spin process");
  if ((data.process_bits & kRadiationProcesses) &&
      !(data.required_state_bits & bits(StateComponentBit::PhotonPool)))
    return failure(PlanErrorCode::RequiredPhotonStateMissing,
                   "radiation/photon process");
  if ((data.process_bits & kBackgroundProcesses) &&
      !(data.required_state_bits & bits(StateComponentBit::SecondaryPool)))
    return failure(PlanErrorCode::RequiredSecondaryStateMissing,
                   "background process");
  if ((data.required_state_bits & bits(StateComponentBit::PhotonStokes)) &&
      !(data.required_state_bits & bits(StateComponentBit::PhotonPool)))
    return failure(PlanErrorCode::RequiredPhotonStateMissing,
                   "photon Stokes component");

  if ((data.output_bits & bits(OutputCapabilityBit::LuminositySpectrum)) &&
      !(data.process_bits & bits(ProcessBit::Luminosity)))
    return failure(PlanErrorCode::OutputProcessMismatch,
                   "luminosity spectrum without luminosity process");
  if ((data.output_bits & bits(OutputCapabilityBit::SpinRecords)) &&
      !(data.required_state_bits & bits(StateComponentBit::Spin)))
    return failure(PlanErrorCode::OutputStateMismatch, "spin records");
  if ((data.output_bits & bits(OutputCapabilityBit::PhotonRecords)) &&
      !(data.required_state_bits & bits(StateComponentBit::PhotonPool)))
    return failure(PlanErrorCode::OutputStateMismatch, "photon records");
  if ((data.output_bits & bits(OutputCapabilityBit::SecondaryRecords)) &&
      !(data.required_state_bits & bits(StateComponentBit::SecondaryPool)))
    return failure(PlanErrorCode::OutputStateMismatch, "secondary records");
  if ((data.output_bits & bits(OutputCapabilityBit::EncounterTrace)) &&
      !(data.capability_bits & bits(CapabilityBit::TraceOutput)))
    return failure(PlanErrorCode::OutputStateMismatch, "encounter trace");

  const std::uint64_t crossing_bits =
      bits(ProcessBit::CrossingGeometry) | bits(ProcessBit::CrabCrossing) |
      bits(ProcessBit::CrabWaist);
  if ((data.process_bits & (bits(ProcessBit::CrabCrossing) |
                            bits(ProcessBit::CrabWaist))) &&
      !(data.process_bits & bits(ProcessBit::CrossingGeometry)))
    return failure(PlanErrorCode::GeometryProcessMismatch,
                   "crab process without crossing geometry");
  if (data.geometry == GeometryMode::HeadOn &&
      (data.process_bits & (crossing_bits | bits(ProcessBit::ArbitraryGeometry))))
    return failure(PlanErrorCode::GeometryProcessMismatch,
                   "head-on with crossing/arbitrary process");
  if (data.geometry == GeometryMode::CrossingAngle &&
      !(data.process_bits & bits(ProcessBit::CrossingGeometry)))
    return failure(PlanErrorCode::GeometryProcessMismatch,
                   "crossing geometry process absent");
  if (data.geometry == GeometryMode::ArbitraryEncounterTime) {
    if (!(data.process_bits & bits(ProcessBit::ArbitraryGeometry)))
      return failure(PlanErrorCode::GeometryProcessMismatch,
                     "arbitrary geometry process absent");
    if (!(data.capability_bits & bits(CapabilityBit::ArbitraryEncounterDAG)))
      return failure(PlanErrorCode::GeometryCapabilityMissing,
                     "arbitrary encounter DAG");
  } else if (data.process_bits & bits(ProcessBit::ArbitraryGeometry)) {
    return failure(PlanErrorCode::GeometryProcessMismatch,
                   "arbitrary process with non-arbitrary geometry mode");
  }

  if (data.engine == Engine::Legacy) {
    if (data.algorithm != Algorithm::Historical ||
        data.precision != PrecisionPolicy::Historical ||
        data.reduction != ReductionPolicy::Historical ||
        data.rng != RngSchema::LegacySequential ||
        data.kernel_family != KernelFamily::LegacyMonolith)
      return failure(PlanErrorCode::LegacyPolicyMismatch,
                     "legacy historical policy tuple");
    if (limits.batch_width != 1)
      return failure(PlanErrorCode::LegacyBatchMismatch, "legacy batch width");
    if (data.geometry != GeometryMode::LegacyCurrent)
      return failure(PlanErrorCode::LegacyGeometryMismatch, "legacy geometry");
    if (data.capability_bits != 0 || data.qualification_bits != 0)
      return failure(PlanErrorCode::LegacyModernCapability,
                     "modern capability on legacy engine");
    if (limits.gpu_mode > 2)
      return failure(PlanErrorCode::GpuModeMismatch, "legacy gpu_mode");
    if ((limits.gpu_mode == 0 &&
         (limits.gpu_devices != 0 || limits.device_memory_bytes != 0)) ||
        (limits.gpu_mode != 0 &&
         (limits.gpu_devices == 0 || limits.device_memory_bytes == 0)))
      return failure(PlanErrorCode::GpuResourceMismatch,
                     "legacy optional GPU resources");
  } else {
    if (data.algorithm == Algorithm::Historical ||
        data.precision == PrecisionPolicy::Historical ||
        data.reduction == ReductionPolicy::Historical)
      return failure(PlanErrorCode::ModernPolicyMismatch,
                     "historical policy on modern engine");
    if (data.rng != RngSchema::CounterV2)
      return failure(PlanErrorCode::CounterRngV2Required, "modern RNG");
    if ((data.capability_bits & kModernCoreCapabilities) !=
        kModernCoreCapabilities)
      return failure(PlanErrorCode::ModernCoreCapabilityMissing,
                     "SoA/CSR/counter/fixed-reduction core");

    if (data.engine == Engine::ModernRefCpu) {
      if (data.algorithm != Algorithm::Exact ||
          data.precision != PrecisionPolicy::Fp64 ||
          data.reduction != ReductionPolicy::Deterministic ||
          limits.batch_width != 1 || limits.cpu_threads != 1 ||
          !(data.capability_bits & bits(CapabilityBit::ScalarReference)))
        return failure(PlanErrorCode::ReferencePolicyMismatch,
                       "scalar exact FP64 deterministic reference");
      if (data.capability_bits &
          (bits(CapabilityBit::PersistentCpu) |
           bits(CapabilityBit::ResidentGpu)))
        return failure(PlanErrorCode::CpuGpuCapabilityConflict,
                       "reference backend capability");
    } else if (data.engine == Engine::ModernCpu) {
      if (!(data.capability_bits & bits(CapabilityBit::PersistentCpu)))
        return failure(PlanErrorCode::ModernCoreCapabilityMissing,
                       "persistent CPU capability");
      if (data.capability_bits &
          (bits(CapabilityBit::ResidentGpu) |
           bits(CapabilityBit::ScalarReference)))
        return failure(PlanErrorCode::CpuGpuCapabilityConflict,
                       "modern CPU capability");
    } else {
      if (!(data.capability_bits & bits(CapabilityBit::ResidentGpu)))
        return failure(PlanErrorCode::ResidentGpuCapabilityMissing,
                       "resident GPU pipeline");
      if (limits.gpu_mode != 3)
        return failure(PlanErrorCode::GpuModeMismatch,
                       "modern GPU requires gpu_mode=3");
      if (limits.gpu_devices == 0 || limits.device_memory_bytes == 0)
        return failure(PlanErrorCode::GpuResourceMismatch,
                       "modern GPU resources");
      if (data.capability_bits &
          (bits(CapabilityBit::PersistentCpu) |
           bits(CapabilityBit::ScalarReference)))
        return failure(PlanErrorCode::CpuGpuCapabilityConflict,
                       "modern GPU capability");
    }
    if (data.engine != Engine::ModernGpu &&
        (limits.gpu_mode != 0 || limits.gpu_devices != 0 ||
         limits.device_memory_bytes != 0))
      return failure(PlanErrorCode::GpuResourceMismatch,
                     "CPU engine has GPU resources");
  }

  if (data.algorithm == Algorithm::FactoredExact &&
      (!(data.capability_bits & bits(CapabilityBit::FactoredExact)) ||
       !(data.qualification_bits & bits(QualificationBit::FactoredExact))))
    return failure(PlanErrorCode::AlgorithmCapabilityMissing,
                   "factored_exact");
  if (data.algorithm == Algorithm::SampledValidated &&
      (!(data.capability_bits & bits(CapabilityBit::SampledEstimator)) ||
       !(data.qualification_bits & bits(QualificationBit::SampledEstimator))))
    return failure(PlanErrorCode::AlgorithmCapabilityMissing,
                   "sampled_validated");
  if (data.algorithm == Algorithm::AdaptiveValidated &&
      (!(data.capability_bits & bits(CapabilityBit::AdaptiveResolution)) ||
       !(data.qualification_bits & bits(QualificationBit::AdaptiveResolution))))
    return failure(PlanErrorCode::AlgorithmCapabilityMissing,
                   "adaptive_validated");
  if (data.precision == PrecisionPolicy::MixedQualified &&
      (!(data.capability_bits & bits(CapabilityBit::MixedPrecision)) ||
       !(data.qualification_bits & bits(QualificationBit::MixedPrecision))))
    return failure(PlanErrorCode::PrecisionCapabilityMissing,
                   "mixed_qualified");
  if (data.reduction == ReductionPolicy::FastQualified &&
      (!(data.capability_bits & bits(CapabilityBit::FastReduction)) ||
       !(data.qualification_bits & bits(QualificationBit::FastReduction))))
    return failure(PlanErrorCode::ReductionCapabilityMissing,
                   "fast_qualified");

  if (data.engine != Engine::Legacy) {
    switch (data.kernel_family) {
      case KernelFamily::NoSpinNoRadiation:
        if (data.process_bits &
            (kSpinProcesses | kRadiationProcesses | kBackgroundProcesses))
          return failure(PlanErrorCode::KernelFamilyMismatch,
                         "no-spin/no-radiation family");
        break;
      case KernelFamily::SpinBmt:
        if (!(data.process_bits & kSpinProcesses) ||
            (data.process_bits & (kRadiationProcesses | kBackgroundProcesses)))
          return failure(PlanErrorCode::KernelFamilyMismatch,
                         "spin/BMT family");
        break;
      case KernelFamily::Beamstrahlung:
        if (!(data.process_bits & bits(ProcessBit::Beamstrahlung)) ||
            (data.process_bits & (kSpinProcesses | kBackgroundProcesses)))
          return failure(PlanErrorCode::KernelFamilyMismatch,
                         "beamstrahlung family");
        break;
      case KernelFamily::FullBackground:
        if (!(data.process_bits & kBackgroundProcesses))
          return failure(PlanErrorCode::KernelFamilyMismatch,
                         "full-background family");
        break;
      case KernelFamily::GenericValidated:
        if (!(data.capability_bits & bits(CapabilityBit::GenericKernel)) ||
            !(data.qualification_bits & bits(QualificationBit::GenericKernel)))
          return failure(PlanErrorCode::GenericKernelUnqualified,
                         "generic kernel");
        break;
      default:
        return failure(PlanErrorCode::KernelFamilyMismatch,
                       "legacy kernel on modern engine");
    }
  }

  return PlanValidation{};
}

}  // namespace

bool PlanLimits::operator==(const PlanLimits& other) const noexcept {
  return grid_nx == other.grid_nx && grid_ny == other.grid_ny &&
      beam1_slices == other.beam1_slices &&
      beam2_slices == other.beam2_slices && substeps == other.substeps &&
      batch_width == other.batch_width && cpu_threads == other.cpu_threads &&
      gpu_devices == other.gpu_devices && gpu_mode == other.gpu_mode &&
      max_step_contexts == other.max_step_contexts &&
      max_primary_particles == other.max_primary_particles &&
      max_photons == other.max_photons &&
      max_secondaries == other.max_secondaries &&
      host_memory_bytes == other.host_memory_bytes &&
      device_memory_bytes == other.device_memory_bytes;
}

const char* plan_error_reason_id(PlanErrorCode code) noexcept {
  switch (code) {
    case PlanErrorCode::None: return "PLAN_OK";
#define FEEBL_PLAN_REASON(name) case PlanErrorCode::name: return "PLAN_E_" #name
    FEEBL_PLAN_REASON(UnsupportedEngineId);
    FEEBL_PLAN_REASON(UnsupportedAlgorithmId);
    FEEBL_PLAN_REASON(UnsupportedPrecisionId);
    FEEBL_PLAN_REASON(UnsupportedReductionId);
    FEEBL_PLAN_REASON(UnsupportedRngId);
    FEEBL_PLAN_REASON(UnsupportedGeometryId);
    FEEBL_PLAN_REASON(UnsupportedKernelFamilyId);
    FEEBL_PLAN_REASON(InertUnitIdentity);
    FEEBL_PLAN_REASON(InertPhysicsIdentity);
    FEEBL_PLAN_REASON(InertObservableIdentity);
    FEEBL_PLAN_REASON(UnsupportedUnitIdentity);
    FEEBL_PLAN_REASON(UnsupportedPhysicsIdentity);
    FEEBL_PLAN_REASON(UnsupportedObservableIdentity);
    FEEBL_PLAN_REASON(UnknownProcessBits);
    FEEBL_PLAN_REASON(UnknownCapabilityBits);
    FEEBL_PLAN_REASON(UnknownQualificationBits);
    FEEBL_PLAN_REASON(UnknownStateBits);
    FEEBL_PLAN_REASON(UnknownOutputBits);
    FEEBL_PLAN_REASON(InvalidGridLimits);
    FEEBL_PLAN_REASON(InvalidSliceLimits);
    FEEBL_PLAN_REASON(InvalidSubstepLimit);
    FEEBL_PLAN_REASON(InvalidBatchLimit);
    FEEBL_PLAN_REASON(InvalidResourceLimits);
    FEEBL_PLAN_REASON(InvalidCapacityLimits);
    FEEBL_PLAN_REASON(LimitArithmeticOverflow);
    FEEBL_PLAN_REASON(LegacyPolicyMismatch);
    FEEBL_PLAN_REASON(LegacyBatchMismatch);
    FEEBL_PLAN_REASON(LegacyGeometryMismatch);
    FEEBL_PLAN_REASON(LegacyModernCapability);
    FEEBL_PLAN_REASON(ReferencePolicyMismatch);
    FEEBL_PLAN_REASON(ModernPolicyMismatch);
    FEEBL_PLAN_REASON(CounterRngV2Required);
    FEEBL_PLAN_REASON(ModernCoreCapabilityMissing);
    FEEBL_PLAN_REASON(GpuModeMismatch);
    FEEBL_PLAN_REASON(ResidentGpuCapabilityMissing);
    FEEBL_PLAN_REASON(GpuResourceMismatch);
    FEEBL_PLAN_REASON(CpuGpuCapabilityConflict);
    FEEBL_PLAN_REASON(CapabilityQualificationMissing);
    FEEBL_PLAN_REASON(OrphanQualification);
    FEEBL_PLAN_REASON(AlgorithmCapabilityMissing);
    FEEBL_PLAN_REASON(PrecisionCapabilityMissing);
    FEEBL_PLAN_REASON(ReductionCapabilityMissing);
    FEEBL_PLAN_REASON(RequiredPrimaryStateMissing);
    FEEBL_PLAN_REASON(RequiredSpinStateMissing);
    FEEBL_PLAN_REASON(RequiredPhotonStateMissing);
    FEEBL_PLAN_REASON(RequiredSecondaryStateMissing);
    FEEBL_PLAN_REASON(OutputStateMismatch);
    FEEBL_PLAN_REASON(OutputProcessMismatch);
    FEEBL_PLAN_REASON(GeometryProcessMismatch);
    FEEBL_PLAN_REASON(GeometryCapabilityMissing);
    FEEBL_PLAN_REASON(KernelFamilyMismatch);
    FEEBL_PLAN_REASON(GenericKernelUnqualified);
#undef FEEBL_PLAN_REASON
  }
  return "PLAN_E_UNKNOWN_REASON";
}

std::vector<std::uint8_t> PhysicsPlan::canonical_hash_input() const {
  static const std::uint8_t magic[] =
      {'F','E','E','B','L','P','L','A','N',0};
  std::vector<std::uint8_t> bytes;
  bytes.reserve(256);
  bytes.insert(bytes.end(), std::begin(magic), std::end(magic));
  append_u32(bytes, canonical_schema_version());
  append_u8(bytes, static_cast<std::uint8_t>(data_.engine));
  append_u8(bytes, static_cast<std::uint8_t>(data_.algorithm));
  append_u8(bytes, static_cast<std::uint8_t>(data_.precision));
  append_u8(bytes, static_cast<std::uint8_t>(data_.reduction));
  append_u8(bytes, static_cast<std::uint8_t>(data_.rng));
  append_u8(bytes, static_cast<std::uint8_t>(data_.geometry));
  append_u8(bytes, static_cast<std::uint8_t>(data_.kernel_family));
  append_identity(bytes, data_.unit_model);
  append_identity(bytes, data_.physics_model);
  append_identity(bytes, data_.observable_model);
  append_u64(bytes, data_.process_bits);
  append_u64(bytes, data_.capability_bits);
  append_u64(bytes, data_.qualification_bits);
  append_u64(bytes, data_.required_state_bits);
  append_u64(bytes, data_.output_bits);
  append_u64(bytes, data_.limits.grid_nx);
  append_u64(bytes, data_.limits.grid_ny);
  append_u64(bytes, data_.limits.beam1_slices);
  append_u64(bytes, data_.limits.beam2_slices);
  append_u64(bytes, data_.limits.substeps);
  append_u64(bytes, data_.limits.batch_width);
  append_u32(bytes, data_.limits.cpu_threads);
  append_u32(bytes, data_.limits.gpu_devices);
  append_u32(bytes, data_.limits.gpu_mode);
  append_u32(bytes, data_.limits.max_step_contexts);
  append_u64(bytes, data_.limits.max_primary_particles);
  append_u64(bytes, data_.limits.max_photons);
  append_u64(bytes, data_.limits.max_secondaries);
  append_u64(bytes, data_.limits.host_memory_bytes);
  append_u64(bytes, data_.limits.device_memory_bytes);
  return bytes;
}

bool PhysicsPlan::operator==(const PhysicsPlan& other) const noexcept {
  return data_.engine == other.data_.engine &&
      data_.algorithm == other.data_.algorithm &&
      data_.precision == other.data_.precision &&
      data_.reduction == other.data_.reduction &&
      data_.rng == other.data_.rng &&
      data_.geometry == other.data_.geometry &&
      data_.kernel_family == other.data_.kernel_family &&
      data_.unit_model == other.data_.unit_model &&
      data_.physics_model == other.data_.physics_model &&
      data_.observable_model == other.data_.observable_model &&
      data_.process_bits == other.data_.process_bits &&
      data_.capability_bits == other.data_.capability_bits &&
      data_.qualification_bits == other.data_.qualification_bits &&
      data_.required_state_bits == other.data_.required_state_bits &&
      data_.output_bits == other.data_.output_bits &&
      data_.limits == other.data_.limits;
}

const PhysicsPlan& PlanBuildResult::plan() const {
  if (!plan_)
    throw std::logic_error("PhysicsPlan requested from failed PlanBuildResult: " +
                           std::string(validation_.reason_id()));
  return *plan_;
}

PlanBuilder::PlanBuilder() : data_(legacy_data()) {}

PlanBuilder PlanBuilder::legacy_defaults() { return PlanBuilder(legacy_data()); }
PlanBuilder PlanBuilder::modern_reference_defaults() {
  return PlanBuilder(modern_data(Engine::ModernRefCpu));
}
PlanBuilder PlanBuilder::modern_cpu_defaults() {
  return PlanBuilder(modern_data(Engine::ModernCpu));
}
PlanBuilder PlanBuilder::modern_gpu_defaults() {
  return PlanBuilder(modern_data(Engine::ModernGpu));
}

#define FEEBL_PLAN_SETTER(method, field, type) \
  PlanBuilder& PlanBuilder::method(type value) { data_.field = value; return *this; }
FEEBL_PLAN_SETTER(set_engine, engine, Engine)
FEEBL_PLAN_SETTER(set_algorithm, algorithm, Algorithm)
FEEBL_PLAN_SETTER(set_precision, precision, PrecisionPolicy)
FEEBL_PLAN_SETTER(set_reduction, reduction, ReductionPolicy)
FEEBL_PLAN_SETTER(set_rng, rng, RngSchema)
FEEBL_PLAN_SETTER(set_geometry, geometry, GeometryMode)
FEEBL_PLAN_SETTER(set_kernel_family, kernel_family, KernelFamily)
FEEBL_PLAN_SETTER(set_process_bits, process_bits, std::uint64_t)
FEEBL_PLAN_SETTER(set_capability_bits, capability_bits, std::uint64_t)
FEEBL_PLAN_SETTER(set_qualification_bits, qualification_bits, std::uint64_t)
FEEBL_PLAN_SETTER(set_required_state_bits, required_state_bits, std::uint64_t)
FEEBL_PLAN_SETTER(set_output_bits, output_bits, std::uint64_t)
#undef FEEBL_PLAN_SETTER

PlanBuilder& PlanBuilder::set_unit_model(const ModelIdentity& value) {
  data_.unit_model = value; return *this;
}
PlanBuilder& PlanBuilder::set_physics_model(const ModelIdentity& value) {
  data_.physics_model = value; return *this;
}
PlanBuilder& PlanBuilder::set_observable_model(const ModelIdentity& value) {
  data_.observable_model = value; return *this;
}
PlanBuilder& PlanBuilder::set_limits(const PlanLimits& value) {
  data_.limits = value; return *this;
}
PlanBuilder& PlanBuilder::enable_process(ProcessBit value, bool enabled) {
  set_bit(data_.process_bits, bits(value), enabled); return *this;
}
PlanBuilder& PlanBuilder::enable_capability(CapabilityBit value, bool enabled) {
  set_bit(data_.capability_bits, bits(value), enabled); return *this;
}
PlanBuilder& PlanBuilder::mark_qualified(QualificationBit value, bool enabled) {
  set_bit(data_.qualification_bits, bits(value), enabled); return *this;
}
PlanBuilder& PlanBuilder::require_state(StateComponentBit value, bool required) {
  set_bit(data_.required_state_bits, bits(value), required); return *this;
}
PlanBuilder& PlanBuilder::enable_output(OutputCapabilityBit value, bool enabled) {
  set_bit(data_.output_bits, bits(value), enabled); return *this;
}

PlanBuildResult PlanBuilder::build() const {
  const PlanValidation validation = validate_data(data_);
  if (!validation) return PlanBuildResult(validation);
  return PlanBuildResult(PhysicsPlan(data_));
}

const ModelIdentity& PlanBuilder::supported_unit_model() noexcept {
  return unit_identity();
}
const ModelIdentity& PlanBuilder::supported_physics_model() noexcept {
  return physics_identity();
}
const ModelIdentity& PlanBuilder::supported_observable_model() noexcept {
  return observable_identity();
}

}  // namespace modern
}  // namespace feebl
