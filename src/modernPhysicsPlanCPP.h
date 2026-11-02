#ifndef FEEBL_MODERN_PHYSICS_PLAN_CPP_H
#define FEEBL_MODERN_PHYSICS_PLAN_CPP_H

#include "modernTypesCPP.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace feebl {
namespace modern {

enum class Engine : std::uint8_t {
  Unspecified = 0,
  Legacy = 1,
  ModernRefCpu = 2,
  ModernCpu = 3,
  ModernGpu = 4
};

enum class Algorithm : std::uint8_t {
  Unspecified = 0,
  Historical = 1,
  Exact = 2,
  FactoredExact = 3,
  SampledValidated = 4,
  AdaptiveValidated = 5
};

enum class PrecisionPolicy : std::uint8_t {
  Unspecified = 0,
  Historical = 1,
  Fp64 = 2,
  MixedQualified = 3
};

enum class ReductionPolicy : std::uint8_t {
  Unspecified = 0,
  Historical = 1,
  Deterministic = 2,
  FastQualified = 3
};

enum class RngSchema : std::uint8_t {
  Unspecified = 0,
  LegacySequential = 1,
  CounterV2 = 2
};

enum class GeometryMode : std::uint8_t {
  Unspecified = 0,
  LegacyCurrent = 1,
  HeadOn = 2,
  CrossingAngle = 3,
  ArbitraryEncounterTime = 4
};

enum class KernelFamily : std::uint8_t {
  Unspecified = 0,
  LegacyMonolith = 1,
  NoSpinNoRadiation = 2,
  SpinBmt = 3,
  Beamstrahlung = 4,
  FullBackground = 5,
  GenericValidated = 6
};

enum class ProcessBit : std::uint64_t {
  Luminosity = 1ull << 0,
  BeamBeamField = 1ull << 1,
  Beamstrahlung = 1ull << 2,
  SpinBmt = 1ull << 3,
  SpinFlip = 1ull << 4,
  PhotonTracking = 1ull << 5,
  CoherentPairs = 1ull << 6,
  IncoherentPairs = 1ull << 7,
  Trident = 1ull << 8,
  Bhabha = 1ull << 9,
  Hadrons = 1ull << 10,
  Compton = 1ull << 11,
  Muons = 1ull << 12,
  CrossingGeometry = 1ull << 13,
  CrabCrossing = 1ull << 14,
  CrabWaist = 1ull << 15,
  Monochromatization = 1ull << 16,
  ArbitraryGeometry = 1ull << 17
};

enum class CapabilityBit : std::uint64_t {
  ModernSoA = 1ull << 0,
  CellCSR = 1ull << 1,
  CounterRngV2 = 1ull << 2,
  FixedReduction = 1ull << 3,
  ScalarReference = 1ull << 4,
  PersistentCpu = 1ull << 5,
  ResidentGpu = 1ull << 6,
  FactoredExact = 1ull << 7,
  SampledEstimator = 1ull << 8,
  AdaptiveResolution = 1ull << 9,
  MixedPrecision = 1ull << 10,
  FastReduction = 1ull << 11,
  ArbitraryEncounterDAG = 1ull << 12,
  GenericKernel = 1ull << 13,
  TraceOutput = 1ull << 14
};

enum class QualificationBit : std::uint64_t {
  FactoredExact = 1ull << 0,
  SampledEstimator = 1ull << 1,
  AdaptiveResolution = 1ull << 2,
  MixedPrecision = 1ull << 3,
  FastReduction = 1ull << 4,
  GenericKernel = 1ull << 5
};

enum class StateComponentBit : std::uint64_t {
  PrimaryBeam = 1ull << 0,
  Spin = 1ull << 1,
  PhotonPool = 1ull << 2,
  SecondaryPool = 1ull << 3,
  PhotonStokes = 1ull << 4
};

enum class OutputCapabilityBit : std::uint64_t {
  Summary = 1ull << 0,
  LuminositySpectrum = 1ull << 1,
  BeamRecords = 1ull << 2,
  SpinRecords = 1ull << 3,
  PhotonRecords = 1ull << 4,
  SecondaryRecords = 1ull << 5,
  EncounterTrace = 1ull << 6
};

constexpr std::uint64_t bits(ProcessBit value) noexcept {
  return static_cast<std::uint64_t>(value);
}
constexpr std::uint64_t bits(CapabilityBit value) noexcept {
  return static_cast<std::uint64_t>(value);
}
constexpr std::uint64_t bits(QualificationBit value) noexcept {
  return static_cast<std::uint64_t>(value);
}
constexpr std::uint64_t bits(StateComponentBit value) noexcept {
  return static_cast<std::uint64_t>(value);
}
constexpr std::uint64_t bits(OutputCapabilityBit value) noexcept {
  return static_cast<std::uint64_t>(value);
}

struct ModelIdentity {
  std::uint32_t version = 0;
  std::array<std::uint8_t, 32> sha256 = {{0}};

  bool operator==(const ModelIdentity& other) const noexcept {
    return version == other.version && sha256 == other.sha256;
  }
  bool operator!=(const ModelIdentity& other) const noexcept {
    return !(*this == other);
  }
};

struct PlanLimits {
  std::uint64_t grid_nx = 32;
  std::uint64_t grid_ny = 32;
  std::uint64_t beam1_slices = 16;
  std::uint64_t beam2_slices = 16;
  std::uint64_t substeps = 1;
  std::uint64_t batch_width = 1;
  std::uint32_t cpu_threads = 1;
  std::uint32_t gpu_devices = 0;
  std::uint32_t gpu_mode = 0;
  std::uint32_t max_step_contexts = 1;
  std::uint64_t max_primary_particles = 200000;
  std::uint64_t max_photons = 0;
  std::uint64_t max_secondaries = 0;
  std::uint64_t host_memory_bytes = 1ull << 30;
  std::uint64_t device_memory_bytes = 0;

  bool operator==(const PlanLimits& other) const noexcept;
  bool operator!=(const PlanLimits& other) const noexcept {
    return !(*this == other);
  }
};

enum class PlanErrorCode : std::uint16_t {
  None = 0,
  UnsupportedEngineId,
  UnsupportedAlgorithmId,
  UnsupportedPrecisionId,
  UnsupportedReductionId,
  UnsupportedRngId,
  UnsupportedGeometryId,
  UnsupportedKernelFamilyId,
  InertUnitIdentity,
  InertPhysicsIdentity,
  InertObservableIdentity,
  UnsupportedUnitIdentity,
  UnsupportedPhysicsIdentity,
  UnsupportedObservableIdentity,
  UnknownProcessBits,
  UnknownCapabilityBits,
  UnknownQualificationBits,
  UnknownStateBits,
  UnknownOutputBits,
  InvalidGridLimits,
  InvalidSliceLimits,
  InvalidSubstepLimit,
  InvalidBatchLimit,
  InvalidResourceLimits,
  InvalidCapacityLimits,
  LimitArithmeticOverflow,
  LegacyPolicyMismatch,
  LegacyBatchMismatch,
  LegacyGeometryMismatch,
  LegacyModernCapability,
  ReferencePolicyMismatch,
  ModernPolicyMismatch,
  CounterRngV2Required,
  ModernCoreCapabilityMissing,
  GpuModeMismatch,
  ResidentGpuCapabilityMissing,
  GpuResourceMismatch,
  CpuGpuCapabilityConflict,
  CapabilityQualificationMissing,
  OrphanQualification,
  AlgorithmCapabilityMissing,
  PrecisionCapabilityMissing,
  ReductionCapabilityMissing,
  RequiredPrimaryStateMissing,
  RequiredSpinStateMissing,
  RequiredPhotonStateMissing,
  RequiredSecondaryStateMissing,
  OutputStateMismatch,
  OutputProcessMismatch,
  GeometryProcessMismatch,
  GeometryCapabilityMissing,
  KernelFamilyMismatch,
  GenericKernelUnqualified
};

const char* plan_error_reason_id(PlanErrorCode code) noexcept;

struct PlanValidation {
  PlanErrorCode code = PlanErrorCode::None;
  std::string detail;

  bool ok() const noexcept { return code == PlanErrorCode::None; }
  explicit operator bool() const noexcept { return ok(); }
  const char* reason_id() const noexcept {
    return plan_error_reason_id(code);
  }
};

struct PhysicsPlanData {
  Engine engine = Engine::Legacy;
  Algorithm algorithm = Algorithm::Historical;
  PrecisionPolicy precision = PrecisionPolicy::Historical;
  ReductionPolicy reduction = ReductionPolicy::Historical;
  RngSchema rng = RngSchema::LegacySequential;
  GeometryMode geometry = GeometryMode::LegacyCurrent;
  KernelFamily kernel_family = KernelFamily::LegacyMonolith;
  ModelIdentity unit_model;
  ModelIdentity physics_model;
  ModelIdentity observable_model;
  std::uint64_t process_bits = 0;
  std::uint64_t capability_bits = 0;
  std::uint64_t qualification_bits = 0;
  std::uint64_t required_state_bits = 0;
  std::uint64_t output_bits = 0;
  PlanLimits limits;
};

class PlanBuilder;

// Immutable after validation: there are no public setters and assignment is
// deliberately disabled.  Copy/move construction is permitted so plans can be
// passed by value to schedulers or uploaded as canonical data.
class PhysicsPlan {
 public:
  PhysicsPlan(const PhysicsPlan&) = default;
  PhysicsPlan(PhysicsPlan&&) noexcept = default;
  PhysicsPlan& operator=(const PhysicsPlan&) = delete;
  PhysicsPlan& operator=(PhysicsPlan&&) = delete;

  Engine engine() const noexcept { return data_.engine; }
  Algorithm algorithm() const noexcept { return data_.algorithm; }
  PrecisionPolicy precision() const noexcept { return data_.precision; }
  ReductionPolicy reduction() const noexcept { return data_.reduction; }
  RngSchema rng() const noexcept { return data_.rng; }
  GeometryMode geometry() const noexcept { return data_.geometry; }
  KernelFamily kernel_family() const noexcept { return data_.kernel_family; }
  const ModelIdentity& unit_model() const noexcept { return data_.unit_model; }
  const ModelIdentity& physics_model() const noexcept {
    return data_.physics_model;
  }
  const ModelIdentity& observable_model() const noexcept {
    return data_.observable_model;
  }
  std::uint64_t process_bits() const noexcept { return data_.process_bits; }
  std::uint64_t capability_bits() const noexcept {
    return data_.capability_bits;
  }
  std::uint64_t qualification_bits() const noexcept {
    return data_.qualification_bits;
  }
  std::uint64_t required_state_bits() const noexcept {
    return data_.required_state_bits;
  }
  std::uint64_t output_bits() const noexcept { return data_.output_bits; }
  const PlanLimits& limits() const noexcept { return data_.limits; }

  bool has_process(ProcessBit value) const noexcept {
    return (data_.process_bits & bits(value)) != 0;
  }
  bool has_capability(CapabilityBit value) const noexcept {
    return (data_.capability_bits & bits(value)) != 0;
  }
  bool has_qualification(QualificationBit value) const noexcept {
    return (data_.qualification_bits & bits(value)) != 0;
  }
  bool requires_state(StateComponentBit value) const noexcept {
    return (data_.required_state_bits & bits(value)) != 0;
  }
  bool has_output(OutputCapabilityBit value) const noexcept {
    return (data_.output_bits & bits(value)) != 0;
  }

  // Versioned, endian-stable bytes to feed SHA-256 or another declared
  // provenance hash.  No object padding, std::hash, locale, or host endian is
  // involved.
  std::vector<std::uint8_t> canonical_hash_input() const;

  bool operator==(const PhysicsPlan& other) const noexcept;
  bool operator!=(const PhysicsPlan& other) const noexcept {
    return !(*this == other);
  }

  static constexpr std::uint32_t canonical_schema_version() noexcept {
    return 1;
  }

 private:
  friend class PlanBuilder;
  explicit PhysicsPlan(const PhysicsPlanData& data) : data_(data) {}
  PhysicsPlanData data_;
};

class PlanBuildResult {
 public:
  bool ok() const noexcept { return validation_.ok(); }
  explicit operator bool() const noexcept { return ok(); }
  const PlanValidation& validation() const noexcept { return validation_; }
  const PhysicsPlan& plan() const;

 private:
  friend class PlanBuilder;
  explicit PlanBuildResult(const PlanValidation& validation)
      : validation_(validation) {}
  explicit PlanBuildResult(PhysicsPlan plan)
      : validation_(), plan_(std::move(plan)) {}

  PlanValidation validation_;
  std::optional<PhysicsPlan> plan_;
};

class PlanBuilder {
 public:
  // The default is the historical, single-batch legacy compatibility plan.
  PlanBuilder();

  static PlanBuilder legacy_defaults();
  static PlanBuilder modern_reference_defaults();
  static PlanBuilder modern_cpu_defaults();
  static PlanBuilder modern_gpu_defaults();

  PlanBuilder& set_engine(Engine value);
  PlanBuilder& set_algorithm(Algorithm value);
  PlanBuilder& set_precision(PrecisionPolicy value);
  PlanBuilder& set_reduction(ReductionPolicy value);
  PlanBuilder& set_rng(RngSchema value);
  PlanBuilder& set_geometry(GeometryMode value);
  PlanBuilder& set_kernel_family(KernelFamily value);
  PlanBuilder& set_unit_model(const ModelIdentity& value);
  PlanBuilder& set_physics_model(const ModelIdentity& value);
  PlanBuilder& set_observable_model(const ModelIdentity& value);
  PlanBuilder& set_process_bits(std::uint64_t value);
  PlanBuilder& set_capability_bits(std::uint64_t value);
  PlanBuilder& set_qualification_bits(std::uint64_t value);
  PlanBuilder& set_required_state_bits(std::uint64_t value);
  PlanBuilder& set_output_bits(std::uint64_t value);
  PlanBuilder& set_limits(const PlanLimits& value);

  PlanBuilder& enable_process(ProcessBit value, bool enabled = true);
  PlanBuilder& enable_capability(CapabilityBit value, bool enabled = true);
  PlanBuilder& mark_qualified(QualificationBit value, bool enabled = true);
  PlanBuilder& require_state(StateComponentBit value, bool required = true);
  PlanBuilder& enable_output(OutputCapabilityBit value, bool enabled = true);

  const PhysicsPlanData& draft() const noexcept { return data_; }
  PlanBuildResult build() const;

  static const ModelIdentity& supported_unit_model() noexcept;
  static const ModelIdentity& supported_physics_model() noexcept;
  static const ModelIdentity& supported_observable_model() noexcept;

 private:
  explicit PlanBuilder(const PhysicsPlanData& data) : data_(data) {}
  PhysicsPlanData data_;
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_MODERN_PHYSICS_PLAN_CPP_H
