#include "modernPhysicsPlanCPP.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace feebl::modern;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-72s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

bool has_code(const PlanBuilder& builder, PlanErrorCode expected) {
  const PlanBuildResult result = builder.build();
  return !result && result.validation().code == expected &&
         std::strncmp(result.validation().reason_id(), "PLAN_E_", 7) == 0;
}

PlanLimits limits_of(const PlanBuilder& builder) {
  return builder.draft().limits;
}

void test_default_and_immutability() {
  std::printf("\nPhysicsPlan default compatibility and immutability\n");
  static_assert(!std::is_default_constructible<PhysicsPlan>::value,
                "PhysicsPlan must only come from a validated builder");
  static_assert(std::is_copy_constructible<PhysicsPlan>::value,
                "immutable plans should be cheaply passable by value");
  static_assert(!std::is_copy_assignable<PhysicsPlan>::value,
                "validated plans must not be mutable by assignment");

  const PlanBuildResult default_result = PlanBuilder().build();
  const PlanBuildResult named_result = PlanBuilder::legacy_defaults().build();
  check(default_result && named_result,
        "default and named legacy compatibility builders both validate");
  if (!default_result || !named_result) return;

  const PhysicsPlan& plan = default_result.plan();
  check(plan.engine() == Engine::Legacy &&
            plan.algorithm() == Algorithm::Historical &&
            plan.precision() == PrecisionPolicy::Historical &&
            plan.reduction() == ReductionPolicy::Historical &&
            plan.rng() == RngSchema::LegacySequential &&
            plan.geometry() == GeometryMode::LegacyCurrent &&
            plan.kernel_family() == KernelFamily::LegacyMonolith,
        "legacy default preserves the complete historical policy tuple");
  check(plan.limits().batch_width == 1 && plan.limits().gpu_mode == 0 &&
            plan.limits().gpu_devices == 0 &&
            plan.has_process(ProcessBit::Luminosity) &&
            plan.has_process(ProcessBit::BeamBeamField) &&
            plan.requires_state(StateComponentBit::PrimaryBeam),
        "legacy default is single-batch CPU with explicit core process/state bits");
  check(plan.unit_model() == PlanBuilder::supported_unit_model() &&
            plan.physics_model() == PlanBuilder::supported_physics_model() &&
            plan.observable_model() == PlanBuilder::supported_observable_model(),
        "default plan carries supported unit, physics and observable identities");

  const PhysicsPlan copied(plan);
  check(plan == named_result.plan() && copied == plan,
        "independently built and copy-constructed plans compare equal");
  const std::vector<std::uint8_t> bytes = plan.canonical_hash_input();
  const std::vector<std::uint8_t> repeated =
      named_result.plan().canonical_hash_input();
  const std::uint8_t magic[] = {'F','E','E','B','L','P','L','A','N',0};
  check(bytes == repeated && bytes.size() == 273,
        "canonical hash input is deterministic and has frozen schema-1 length");
  check(std::equal(std::begin(magic), std::end(magic), bytes.begin()) &&
            bytes[10] == 1 && bytes[11] == 0 && bytes[12] == 0 &&
            bytes[13] == 0 && bytes[14] == static_cast<std::uint8_t>(Engine::Legacy),
        "canonical bytes begin with fixed magic, little-endian schema and engine ID");
}

void test_engine_table() {
  std::printf("\nPhysicsPlan valid engine/algorithm combinations\n");
  const PlanBuildResult reference =
      PlanBuilder::modern_reference_defaults().build();
  const PlanBuildResult cpu = PlanBuilder::modern_cpu_defaults().build();
  const PlanBuildResult gpu = PlanBuilder::modern_gpu_defaults().build();
  check(reference && reference.plan().engine() == Engine::ModernRefCpu &&
            reference.plan().algorithm() == Algorithm::Exact &&
            reference.plan().precision() == PrecisionPolicy::Fp64 &&
            reference.plan().reduction() == ReductionPolicy::Deterministic &&
            reference.plan().rng() == RngSchema::CounterV2 &&
            reference.plan().limits().cpu_threads == 1 &&
            reference.plan().limits().batch_width == 1,
        "modern reference is scalar exact FP64 deterministic counter-V2");
  check(cpu && cpu.plan().has_capability(CapabilityBit::PersistentCpu) &&
            !cpu.plan().has_capability(CapabilityBit::ResidentGpu) &&
            cpu.plan().limits().gpu_mode == 0,
        "modern CPU default selects persistent CPU without GPU state");
  check(gpu && gpu.plan().has_capability(CapabilityBit::ResidentGpu) &&
            gpu.plan().limits().gpu_mode == 3 &&
            gpu.plan().limits().gpu_devices == 1 &&
            gpu.plan().limits().device_memory_bytes > 0,
        "modern GPU default requires resident capability and new gpu_mode=3");

  PlanBuilder factored = PlanBuilder::modern_cpu_defaults();
  factored.set_algorithm(Algorithm::FactoredExact)
      .enable_capability(CapabilityBit::FactoredExact)
      .mark_qualified(QualificationBit::FactoredExact);
  check(factored.build().ok(), "qualified factored-exact algorithm is accepted");

  PlanBuilder sampled = PlanBuilder::modern_cpu_defaults();
  sampled.set_algorithm(Algorithm::SampledValidated)
      .enable_capability(CapabilityBit::SampledEstimator)
      .mark_qualified(QualificationBit::SampledEstimator);
  check(sampled.build().ok(), "separately qualified sampled algorithm is accepted");

  PlanBuilder adaptive = PlanBuilder::modern_cpu_defaults();
  adaptive.set_algorithm(Algorithm::AdaptiveValidated)
      .enable_capability(CapabilityBit::AdaptiveResolution)
      .mark_qualified(QualificationBit::AdaptiveResolution);
  check(adaptive.build().ok(), "separately qualified adaptive algorithm is accepted");

  PlanBuilder mixed_fast = PlanBuilder::modern_gpu_defaults();
  mixed_fast.set_precision(PrecisionPolicy::MixedQualified)
      .set_reduction(ReductionPolicy::FastQualified)
      .enable_capability(CapabilityBit::MixedPrecision)
      .mark_qualified(QualificationBit::MixedPrecision)
      .enable_capability(CapabilityBit::FastReduction)
      .mark_qualified(QualificationBit::FastReduction);
  check(mixed_fast.build().ok(),
        "independently qualified mixed precision and fast reduction are accepted");
}

void test_one_bit_qualification_contract() {
  std::printf("\nPhysicsPlan one-bit qualification gates\n");
  struct Pair {
    CapabilityBit capability;
    QualificationBit qualification;
    const char* description;
  };
  const Pair pairs[] = {
      {CapabilityBit::FactoredExact, QualificationBit::FactoredExact,
       "factored-exact"},
      {CapabilityBit::SampledEstimator, QualificationBit::SampledEstimator,
       "sampled-estimator"},
      {CapabilityBit::AdaptiveResolution, QualificationBit::AdaptiveResolution,
       "adaptive-resolution"},
      {CapabilityBit::MixedPrecision, QualificationBit::MixedPrecision,
       "mixed-precision"},
      {CapabilityBit::FastReduction, QualificationBit::FastReduction,
       "fast-reduction"},
      {CapabilityBit::GenericKernel, QualificationBit::GenericKernel,
       "generic-kernel"}};

  const PlanBuildResult baseline = PlanBuilder::modern_cpu_defaults().build();
  check(baseline.ok(), "one-bit tests start from valid modern CPU baseline");
  for (const Pair& pair : pairs) {
    PlanBuilder capability_only = PlanBuilder::modern_cpu_defaults();
    capability_only.enable_capability(pair.capability);
    check(has_code(capability_only,
                   PlanErrorCode::CapabilityQualificationMissing),
          pair.description);

    PlanBuilder qualification_only = PlanBuilder::modern_cpu_defaults();
    qualification_only.mark_qualified(pair.qualification);
    check(has_code(qualification_only, PlanErrorCode::OrphanQualification),
          pair.description);

    PlanBuilder qualified = PlanBuilder::modern_cpu_defaults();
    qualified.enable_capability(pair.capability)
        .mark_qualified(pair.qualification);
    const PlanBuildResult accepted = qualified.build();
    check(accepted && baseline && accepted.plan() != baseline.plan() &&
              accepted.plan().canonical_hash_input() !=
                  baseline.plan().canonical_hash_input(),
          pair.description);
  }
}

void test_negative_combinations() {
  std::printf("\nPhysicsPlan negative combination table and reason IDs\n");

  PlanBuilder bad = PlanBuilder::legacy_defaults();
  bad.set_algorithm(Algorithm::Exact);
  check(has_code(bad, PlanErrorCode::LegacyPolicyMismatch),
        "legacy rejects a non-historical algorithm");

  bad = PlanBuilder::legacy_defaults();
  PlanLimits limits = limits_of(bad);
  limits.batch_width = 2;
  bad.set_limits(limits);
  check(has_code(bad, PlanErrorCode::LegacyBatchMismatch),
        "legacy rejects batch width greater than one");

  bad = PlanBuilder::modern_reference_defaults();
  limits = limits_of(bad);
  limits.cpu_threads = 2;
  bad.set_limits(limits);
  check(has_code(bad, PlanErrorCode::ReferencePolicyMismatch),
        "modern reference rejects more than one CPU thread");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_rng(RngSchema::LegacySequential);
  check(has_code(bad, PlanErrorCode::CounterRngV2Required),
        "modern CPU rejects legacy call-order RNG");

  bad = PlanBuilder::modern_gpu_defaults();
  bad.enable_capability(CapabilityBit::ResidentGpu, false);
  check(has_code(bad, PlanErrorCode::ResidentGpuCapabilityMissing),
        "modern GPU rejects missing resident-pipeline capability");

  bad = PlanBuilder::modern_gpu_defaults();
  limits = limits_of(bad);
  limits.gpu_mode = 2;
  bad.set_limits(limits);
  check(has_code(bad, PlanErrorCode::GpuModeMismatch),
        "modern GPU refuses reinterpretation of existing gpu_mode=2");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_precision(PrecisionPolicy::MixedQualified);
  check(has_code(bad, PlanErrorCode::PrecisionCapabilityMissing),
        "mixed precision rejects absent capability and qualification");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_reduction(ReductionPolicy::FastQualified);
  check(has_code(bad, PlanErrorCode::ReductionCapabilityMissing),
        "fast reduction rejects absent capability and qualification");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_algorithm(Algorithm::SampledValidated);
  check(has_code(bad, PlanErrorCode::AlgorithmCapabilityMissing),
        "sampled algorithm rejects absent capability and qualification");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_engine(static_cast<Engine>(99));
  const PlanBuildResult unknown_engine = bad.build();
  check(!unknown_engine &&
            unknown_engine.validation().code ==
                PlanErrorCode::UnsupportedEngineId &&
            std::strcmp(unknown_engine.validation().reason_id(),
                        "PLAN_E_UnsupportedEngineId") == 0,
        "unsupported enum has a stable machine-readable reason ID");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_physics_model(ModelIdentity{});
  check(has_code(bad, PlanErrorCode::InertPhysicsIdentity),
        "zero/inert physics identity is rejected");

  bad = PlanBuilder::modern_cpu_defaults();
  ModelIdentity unsupported = PlanBuilder::supported_physics_model();
  unsupported.sha256[31] ^= 1;
  bad.set_physics_model(unsupported);
  check(has_code(bad, PlanErrorCode::UnsupportedPhysicsIdentity),
        "nondefault unsupported physics hash is rejected");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_process_bits(bad.draft().process_bits | (1ull << 63));
  check(has_code(bad, PlanErrorCode::UnknownProcessBits),
        "unknown process/capability namespace bit is rejected");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.require_state(StateComponentBit::PrimaryBeam, false);
  check(has_code(bad, PlanErrorCode::RequiredPrimaryStateMissing),
        "every executable plan requires primary beam state");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.enable_process(ProcessBit::SpinBmt);
  check(has_code(bad, PlanErrorCode::RequiredSpinStateMissing),
        "spin process rejects absent spin state component");

  PlanBuilder spin = PlanBuilder::modern_cpu_defaults();
  spin.enable_process(ProcessBit::SpinBmt)
      .require_state(StateComponentBit::Spin)
      .set_kernel_family(KernelFamily::SpinBmt);
  check(spin.build().ok(), "spin/BMT process, state and kernel family agree");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_geometry(GeometryMode::ArbitraryEncounterTime)
      .enable_process(ProcessBit::ArbitraryGeometry);
  check(has_code(bad, PlanErrorCode::GeometryCapabilityMissing),
        "arbitrary encounter time requires explicit DAG capability");
  bad.enable_capability(CapabilityBit::ArbitraryEncounterDAG);
  check(bad.build().ok(), "arbitrary geometry validates once process and DAG agree");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.enable_output(OutputCapabilityBit::EncounterTrace);
  check(has_code(bad, PlanErrorCode::OutputStateMismatch),
        "trace output rejects absent trace capability");

  bad = PlanBuilder::modern_cpu_defaults();
  bad.set_kernel_family(KernelFamily::GenericValidated);
  check(has_code(bad, PlanErrorCode::GenericKernelUnqualified),
        "generic fallback kernel rejects absent independent qualification");

  bad = PlanBuilder::modern_cpu_defaults();
  limits = limits_of(bad);
  limits.beam1_slices = std::numeric_limits<std::uint64_t>::max();
  limits.beam2_slices = 2;
  bad.set_limits(limits);
  check(has_code(bad, PlanErrorCode::LimitArithmeticOverflow),
        "encounter-resource arithmetic overflow is rejected before execution");

  check([] {
          const PlanBuildResult failed =
              PlanBuilder::modern_cpu_defaults()
                  .set_rng(RngSchema::LegacySequential).build();
          try { (void)failed.plan(); }
          catch (const std::logic_error&) { return true; }
          return false;
        }(),
        "failed builder cannot expose a PhysicsPlan value");
}

}  // namespace

int main() {
  test_default_and_immutability();
  test_engine_table();
  test_one_bit_qualification_contract();
  test_negative_combinations();
  std::printf("\n%s (%d failure%s)\n",
              failures ? "FAILURES PRESENT" : "all PhysicsPlan checks passed",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
