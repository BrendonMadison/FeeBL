#include "legacyElectronLumiBlockedCPP.h"
#include "pairMomentsCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace blocked = feebl::legacy_electron_lumi_blocked;

namespace {

constexpr double kElectronMass = 0.51099906e-3;
int failures = 0;

static_assert(noexcept(blocked::evaluate_block(
                  std::declval<const blocked::BeamSoA&>(), 0U,
                  std::declval<const blocked::BeamSoA&>(), 0U, 0U, 0.0F,
                  kElectronMass,
                  std::declval<blocked::BlockKinematics&>())),
              "the post-preflight pair kernel must remain noexcept");

void check(bool condition, const char* description) {
  std::printf("  %-76s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_double(double first, double second) {
  return double_bits(first) == double_bits(second);
}

struct RawParticle {
  float energy;
  float vx;
  float vy;
  float weight;
};

struct ScalarParticle {
  float signed_energy;
  float absolute_energy;
  float vx;
  float vy;
  float vz;
  float weight;
  double massive_momentum;
  std::uint8_t charge_class;
};

ScalarParticle scalar_precompute(const RawParticle& raw, bool beam2) {
  ScalarParticle result;
  result.signed_energy = raw.energy;
  result.absolute_energy = std::fabs(raw.energy);
  result.vx = raw.vx;
  result.vy = raw.vy;
  result.vz = static_cast<float>(
      std::sqrt(1. - raw.vx * raw.vx - raw.vy * raw.vy));
  result.weight = raw.weight;
  const double energy = static_cast<double>(raw.energy);
  result.massive_momentum =
      (energy > kElectronMass)
          ? std::sqrt((energy - kElectronMass) *
                      (energy + kElectronMass))
          : 0.0;
  result.charge_class = static_cast<std::uint8_t>(
      raw.energy < 0.0F ? (beam2 ? 0 : 1) : (beam2 ? 1 : 0));
  return result;
}

void candidate_append(blocked::BeamSoA& destination,
                      const RawParticle& raw, bool beam2) {
  const float absolute_energy = std::fabs(raw.energy);
  const float vz = static_cast<float>(
      std::sqrt(1. - raw.vx * raw.vx - raw.vy * raw.vy));
  const double energy = static_cast<double>(raw.energy);
  const double massive_momentum =
      (energy > kElectronMass)
          ? std::sqrt((energy - kElectronMass) *
                      (energy + kElectronMass))
          : 0.0;
  const std::uint8_t charge_class = static_cast<std::uint8_t>(
      raw.energy < 0.0F ? (beam2 ? 0 : 1) : (beam2 ? 1 : 0));
  destination.append(raw.energy, absolute_energy, raw.vx, raw.vy, vz,
                     raw.weight, massive_momentum, charge_class);
}

struct ScalarPair {
  float weight;
  float ecm;
  double invariant_mass;
  bool same_sign;
  bool above_threshold;
};

ScalarPair scalar_pair(const ScalarParticle& first,
                       const ScalarParticle& second, float ecm_min) {
  ScalarPair result;
  result.weight = first.weight * second.weight;
  const float cosinus = first.vx * second.vx + first.vy * second.vy -
                        first.vz * second.vz;
  result.ecm = static_cast<float>(std::sqrt(
      2.0 * first.absolute_energy * second.absolute_energy *
      (1. - cosinus)));
  result.same_sign = first.signed_energy * second.signed_energy > 0.0;
  result.above_threshold = result.ecm > ecm_min;
  const double cosang =
      static_cast<double>(first.vx) * static_cast<double>(second.vx) +
      static_cast<double>(first.vy) * static_cast<double>(second.vy) +
      static_cast<double>(first.vz) * -static_cast<double>(second.vz);
  const double w2 =
      2.0 * kElectronMass * kElectronMass +
      2.0 * (static_cast<double>(first.signed_energy) *
                 static_cast<double>(second.signed_energy) -
             first.massive_momentum * second.massive_momentum * cosang);
  result.invariant_mass = (w2 > 0.0) ? std::sqrt(w2) : 0.0;
  return result;
}

struct SpectrumConfig {
  double low;
  double high;
  std::size_t bins;
};

struct RawSpectrum {
  double low = 0.0;
  double high = 0.0;
  double width = 0.0;
  std::vector<double> bins;
  double sum_weight = 0.0;
  double sum_weight2 = 0.0;
  double weighted_w = 0.0;
  double weighted_w2 = 0.0;
  double underflow = 0.0;
  double overflow = 0.0;

  void configure(const SpectrumConfig& config) {
    if (!(config.high > config.low) || config.bins == 0U) return;
    low = config.low;
    high = config.high;
    width = (high - low) / static_cast<double>(config.bins);
    bins.assign(config.bins, 0.0);
  }

  void add(double pair_weight, double invariant_mass) {
    sum_weight += pair_weight;
    sum_weight2 += pair_weight * pair_weight;
    weighted_w += pair_weight * invariant_mass;
    weighted_w2 += pair_weight * invariant_mass * invariant_mass;
    if (bins.empty()) return;
    if (invariant_mass < low) {
      underflow += pair_weight;
      return;
    }
    if (invariant_mass >= high) {
      overflow += pair_weight;
      return;
    }
    std::size_t bin = static_cast<std::size_t>(
        (invariant_mass - low) / width);
    if (bin >= bins.size()) bin = bins.size() - 1U;
    bins[bin] += pair_weight;
  }

  double mean() const {
    return sum_weight > 0.0 ? weighted_w / sum_weight : 0.0;
  }
  double n_eff() const {
    return sum_weight2 > 0.0
               ? (sum_weight * sum_weight) / sum_weight2
               : 0.0;
  }
  double variance() const {
    if (!(sum_weight > 0.0)) return 0.0;
    const double mu = mean();
    const double value = weighted_w2 / sum_weight - mu * mu;
    return value > 0.0 ? value : 0.0;
  }
  double error() const {
    const double effective = n_eff();
    return effective > 1.0 ? std::sqrt(variance() / effective) : 0.0;
  }
};

struct ScalarResults {
  std::array<double, 9> class_luminosity{};
  double lumi_ee = 0.0;
  double step_lumi_ee = 0.0;
  double lumi_ee_high = 0.0;
  double step_lumi_ee_high = 0.0;
  double lumi_ecm = 0.0;
  double lumi_ecm2 = 0.0;
  RawSpectrum spectrum;
};

struct CandidateResults {
  std::array<double, 9> class_luminosity{};
  double lumi_ee = 0.0;
  double step_lumi_ee = 0.0;
  double lumi_ee_high = 0.0;
  double step_lumi_ee_high = 0.0;
  double lumi_ecm = 0.0;
  double lumi_ecm2 = 0.0;
  pairmoments::SpectrumW spectrum;
};

struct WorkspaceSnapshot {
  std::array<const void*, 16> data;
  std::array<std::size_t, 16> capacity;
};

WorkspaceSnapshot snapshot(const blocked::Workspace& workspace) {
  WorkspaceSnapshot result;
  result.data = {{
      workspace.beam1.signed_energy.data(),
      workspace.beam1.absolute_energy.data(),
      workspace.beam1.vx.data(), workspace.beam1.vy.data(),
      workspace.beam1.vz.data(), workspace.beam1.weight.data(),
      workspace.beam1.massive_momentum.data(),
      workspace.beam1.charge_class.data(),
      workspace.beam2.signed_energy.data(),
      workspace.beam2.absolute_energy.data(),
      workspace.beam2.vx.data(), workspace.beam2.vy.data(),
      workspace.beam2.vz.data(), workspace.beam2.weight.data(),
      workspace.beam2.massive_momentum.data(),
      workspace.beam2.charge_class.data(),
  }};
  result.capacity = {{
      workspace.beam1.signed_energy.capacity(),
      workspace.beam1.absolute_energy.capacity(),
      workspace.beam1.vx.capacity(), workspace.beam1.vy.capacity(),
      workspace.beam1.vz.capacity(), workspace.beam1.weight.capacity(),
      workspace.beam1.massive_momentum.capacity(),
      workspace.beam1.charge_class.capacity(),
      workspace.beam2.signed_energy.capacity(),
      workspace.beam2.absolute_energy.capacity(),
      workspace.beam2.vx.capacity(), workspace.beam2.vy.capacity(),
      workspace.beam2.vz.capacity(), workspace.beam2.weight.capacity(),
      workspace.beam2.massive_momentum.capacity(),
      workspace.beam2.charge_class.capacity(),
  }};
  return result;
}

bool same_storage(const WorkspaceSnapshot& first,
                  const WorkspaceSnapshot& second) {
  return first.data == second.data && first.capacity == second.capacity;
}

ScalarResults run_scalar(const std::vector<RawParticle>& beam1,
                         const std::vector<RawParticle>& beam2,
                         float ecm_min, const SpectrumConfig& spectrum) {
  ScalarResults result;
  result.spectrum.configure(spectrum);
  std::vector<ScalarParticle> first;
  std::vector<ScalarParticle> second;
  first.reserve(beam1.size());
  second.reserve(beam2.size());
  for (const RawParticle& particle : beam1)
    first.push_back(scalar_precompute(particle, false));
  for (const RawParticle& particle : beam2)
    second.push_back(scalar_precompute(particle, true));

  for (const ScalarParticle& particle1 : first) {
    for (const ScalarParticle& particle2 : second) {
      const ScalarPair pair = scalar_pair(particle1, particle2, ecm_min);
      result.class_luminosity[
          static_cast<std::size_t>(particle1.charge_class) * 3U +
          particle2.charge_class] += pair.weight;
      if (pair.same_sign) {
        result.lumi_ee += pair.weight;
        result.step_lumi_ee += pair.weight;
        result.spectrum.add(static_cast<double>(pair.weight),
                            pair.invariant_mass);
        if (pair.above_threshold) {
          result.lumi_ee_high += pair.weight;
          result.step_lumi_ee_high += pair.weight;
        }
        const float help = pair.weight * pair.ecm;
        result.lumi_ecm += help;
        result.lumi_ecm2 += help * pair.ecm;
      }
    }
  }
  return result;
}

bool prepare_workspace(blocked::Workspace& workspace,
                       const std::vector<RawParticle>& beam1,
                       const std::vector<RawParticle>& beam2) {
  workspace.reserve(beam1.size(), beam2.size());
  workspace.beam1.clear();
  workspace.beam2.clear();
  for (const RawParticle& particle : beam1)
    candidate_append(workspace.beam1, particle, false);
  for (const RawParticle& particle : beam2)
    candidate_append(workspace.beam2, particle, true);
  return workspace.beam1.component_sizes_match() &&
         workspace.beam2.component_sizes_match();
}

CandidateResults run_candidate(blocked::Workspace& workspace,
                               const std::vector<RawParticle>& beam1,
                               const std::vector<RawParticle>& beam2,
                               float ecm_min,
                               const SpectrumConfig& spectrum,
                               bool& contract_ok) {
  CandidateResults result;
  result.spectrum.configure(spectrum.low, spectrum.high, spectrum.bins);
  contract_ok = prepare_workspace(workspace, beam1, beam2);
  for (std::size_t first = 0; first < workspace.beam1.size(); ++first) {
    for (std::size_t begin = 0; begin < workspace.beam2.size();
         begin += blocked::kBlockSize) {
      const std::size_t count = std::min(
          blocked::kBlockSize, workspace.beam2.size() - begin);
      contract_ok = contract_ok && blocked::evaluate_block(
          workspace.beam1, first, workspace.beam2, begin, count, ecm_min,
          kElectronMass, workspace.block);
      if (!contract_ok) return result;
      for (std::size_t local = 0; local < count; ++local) {
        const std::size_t second = begin + local;
        const float pair_weight = workspace.block.weight[local];
        const float ecm = workspace.block.ecm[local];
        result.class_luminosity[
            static_cast<std::size_t>(
                workspace.beam1.charge_class[first]) * 3U +
            workspace.beam2.charge_class[second]] += pair_weight;
        if (workspace.block.same_sign[local] != 0U) {
          result.lumi_ee += pair_weight;
          result.step_lumi_ee += pair_weight;
          result.spectrum.add(
              static_cast<double>(pair_weight),
              workspace.block.invariant_mass[local]);
          if (workspace.block.above_threshold[local] != 0U) {
            result.lumi_ee_high += pair_weight;
            result.step_lumi_ee_high += pair_weight;
          }
          const float help = pair_weight * ecm;
          result.lumi_ecm += help;
          result.lumi_ecm2 += help * ecm;
        }
      }
    }
  }
  return result;
}

bool same_spectrum(const pairmoments::SpectrumW& candidate,
                   const RawSpectrum& scalar) {
  if (candidate.bins() != scalar.bins.size() ||
      !same_double(candidate.underflow(), scalar.underflow) ||
      !same_double(candidate.overflow(), scalar.overflow))
    return false;
  for (std::size_t bin = 0; bin < scalar.bins.size(); ++bin)
    if (!same_double(candidate.bin_weight(bin), scalar.bins[bin]))
      return false;
  const pairmoments::Moment1& moment = candidate.w_moment();
  return same_double(moment.weight(), scalar.sum_weight) &&
         same_double(moment.mean(), scalar.mean()) &&
         same_double(moment.n_eff(), scalar.n_eff()) &&
         same_double(moment.variance(), scalar.variance()) &&
         same_double(moment.error(), scalar.error());
}

bool same_results(const CandidateResults& candidate,
                  const ScalarResults& scalar) {
  for (std::size_t index = 0; index < scalar.class_luminosity.size(); ++index)
    if (!same_double(candidate.class_luminosity[index],
                     scalar.class_luminosity[index]))
      return false;
  return same_double(candidate.lumi_ee, scalar.lumi_ee) &&
         same_double(candidate.step_lumi_ee, scalar.step_lumi_ee) &&
         same_double(candidate.lumi_ee_high, scalar.lumi_ee_high) &&
         same_double(candidate.step_lumi_ee_high,
                     scalar.step_lumi_ee_high) &&
         same_double(candidate.lumi_ecm, scalar.lumi_ecm) &&
         same_double(candidate.lumi_ecm2, scalar.lumi_ecm2) &&
         same_spectrum(candidate.spectrum, scalar.spectrum);
}

void compare_case(blocked::Workspace& workspace,
                  const std::vector<RawParticle>& beam1,
                  const std::vector<RawParticle>& beam2,
                  float threshold, const SpectrumConfig& spectrum,
                  const char* description) {
  const ScalarResults scalar =
      run_scalar(beam1, beam2, threshold, spectrum);
  bool contract_ok = false;
  const CandidateResults candidate =
      run_candidate(workspace, beam1, beam2, threshold, spectrum,
                    contract_ok);
  check(contract_ok && same_results(candidate, scalar), description);
}

void test_alignment_capacity_and_contracts() {
  std::printf("alignment, retained capacity, and fail-closed API contracts\n");
  blocked::Workspace workspace;
  workspace.reserve(257U, 257U);
  const std::vector<RawParticle> one = {{125.0F, 0.0F, 0.0F, 1.0F}};
  prepare_workspace(workspace, one, one);
  const auto aligned = [](const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) %
               blocked::kScratchAlignment ==
           0U;
  };
  check(aligned(workspace.beam1.signed_energy.data()) &&
            aligned(workspace.beam1.absolute_energy.data()),
        "beam-1 SoA components begin on 64-byte boundaries");
  check(aligned(workspace.beam2.massive_momentum.data()) &&
            aligned(workspace.block.invariant_mass.data()),
        "beam-2 double data and fixed block outputs are 64-byte aligned");

  const float* signed_pointer = workspace.beam1.signed_energy.data();
  const double* momentum_pointer = workspace.beam2.massive_momentum.data();
  workspace.beam1.clear();
  workspace.beam2.clear();
  for (int index = 0; index < 200; ++index) {
    const RawParticle raw = {125.0F + index * 0.01F,
                             0.001F, -0.002F, 1.0F};
    candidate_append(workspace.beam1, raw, false);
    candidate_append(workspace.beam2, raw, true);
  }
  check(signed_pointer == workspace.beam1.signed_energy.data() &&
            momentum_pointer == workspace.beam2.massive_momentum.data(),
        "preflight capacity is reused without cell-time allocation");

  check(!blocked::evaluate_block(
            workspace.beam1, workspace.beam1.size(), workspace.beam2, 0U,
            1U, 0.0F, kElectronMass, workspace.block),
        "out-of-range beam-1 index is rejected before evaluation");
  check(!blocked::evaluate_block(
            workspace.beam1, 0U, workspace.beam2, 1U, 1U, 0.0F,
            kElectronMass, workspace.block),
        "misaligned/non-block beam-2 start is rejected");
  check(!blocked::evaluate_block(
            workspace.beam1, 0U, workspace.beam2, 0U,
            blocked::kBlockSize + 1U, 0.0F, kElectronMass,
            workspace.block),
        "oversize block is rejected");
}

void test_empty_zero_and_signed_energy() {
  std::printf("empty, zero, signed-energy, and non-dyadic cases\n");
  blocked::Workspace workspace;
  const SpectrumConfig spectrum = {0.125, 501.375, 73U};
  compare_case(workspace, {}, {}, 17.3F, spectrum,
               "two empty beams produce identical empty accumulators");
  compare_case(workspace, {{125.0F, 0.0F, 0.0F, 1.0F}}, {}, 17.3F,
               spectrum, "one-sided empty beam produces no pairs");
  const std::vector<RawParticle> beam1 = {
      {125.125F, 0.0031F, -0.0047F, 1.125F},
      {-91.375F, -0.0123F, 0.0077F, 0.625F},
      {0.0F, 0.0F, 0.0F, 2.0F},
      {-0.0F, 0.0F, -0.0F, 0.0F},
      {0.0007F, 0.125F, -0.0625F, 1.75F}};
  const std::vector<RawParticle> beam2 = {
      {110.2F, -0.0029F, 0.0053F, 0.875F},
      {-80.6F, 0.0081F, -0.0067F, 1.375F},
      {0.0F, -0.0F, 0.0F, 0.25F},
      {-0.0F, 0.0F, 0.0F, 0.5F},
      {0.0008F, -0.03125F, 0.09375F, 1.25F}};
  compare_case(workspace, beam1, beam2, 149.03125F, spectrum,
               "signed/zero energies and non-dyadic kinematics match bits");
}

void test_every_materialized_lane() {
  std::printf("independent per-lane kinematics and metadata\n");
  blocked::Workspace workspace;
  const std::vector<RawParticle> beam1 = {
      {125.125F, 0.0031F, -0.0047F, 1.125F},
      {-91.375F, -0.0123F, 0.0077F, 0.625F}};
  std::vector<RawParticle> beam2;
  for (std::size_t index = 0; index < 129U; ++index) {
    const float sign = index % 2U == 0U ? 1.0F : -1.0F;
    beam2.push_back({sign * (80.0F + 0.037F * index),
                     -0.031F + 0.0001F * index,
                     0.021F - 0.00007F * index,
                     0.25F + 0.0625F * (index % 13U)});
  }
  bool exact = prepare_workspace(workspace, beam1, beam2);
  const WorkspaceSnapshot before = snapshot(workspace);
  const float threshold = 174.3125F;
  for (std::size_t first = 0; first < beam1.size() && exact; ++first) {
    const ScalarParticle scalar1 = scalar_precompute(beam1[first], false);
    for (std::size_t begin = 0; begin < beam2.size();
         begin += blocked::kBlockSize) {
      const std::size_t count = std::min(
          blocked::kBlockSize, beam2.size() - begin);
      exact = blocked::evaluate_block(
          workspace.beam1, first, workspace.beam2, begin, count, threshold,
          kElectronMass, workspace.block);
      for (std::size_t local = 0; local < count && exact; ++local) {
        const ScalarPair scalar = scalar_pair(
            scalar1, scalar_precompute(beam2[begin + local], true),
            threshold);
        std::uint32_t candidate_weight_bits = 0;
        std::uint32_t scalar_weight_bits = 0;
        std::uint32_t candidate_ecm_bits = 0;
        std::uint32_t scalar_ecm_bits = 0;
        std::memcpy(&candidate_weight_bits, &workspace.block.weight[local],
                    sizeof(candidate_weight_bits));
        std::memcpy(&scalar_weight_bits, &scalar.weight,
                    sizeof(scalar_weight_bits));
        std::memcpy(&candidate_ecm_bits, &workspace.block.ecm[local],
                    sizeof(candidate_ecm_bits));
        std::memcpy(&scalar_ecm_bits, &scalar.ecm,
                    sizeof(scalar_ecm_bits));
        exact = candidate_weight_bits == scalar_weight_bits &&
                candidate_ecm_bits == scalar_ecm_bits &&
                same_double(workspace.block.invariant_mass[local],
                            scalar.invariant_mass) &&
                (workspace.block.same_sign[local] != 0U) ==
                    scalar.same_sign &&
                (workspace.block.above_threshold[local] != 0U) ==
                    scalar.above_threshold;
      }
    }
  }
  check(exact,
        "all fields match scalar bits across unlike signs and a 128+1 tail");
  const ScalarResults scalar = run_scalar(
      beam1, beam2, threshold, {0.125, 501.375, 73U});
  bool contract_ok = false;
  const CandidateResults candidate = run_candidate(
      workspace, beam1, beam2, threshold, {0.125, 501.375, 73U},
      contract_ok);
  const WorkspaceSnapshot after = snapshot(workspace);
  check(contract_ok && same_results(candidate, scalar),
        "multi-block scalar commits match the legacy ordered accumulator bits");
  check(same_storage(before, after),
        "all sixteen SoA bases/capacities remain unchanged during pair work");
}

void test_threshold_and_spectrum_edges() {
  std::printf("threshold ULPs and spectrum bin edges\n");
  blocked::Workspace workspace;
  const RawParticle raw1 = {125.0F, 0.0013F, -0.0027F, 1.0F};
  const RawParticle raw2 = {124.0F, -0.0037F, 0.0011F, 0.75F};
  const ScalarPair probe = scalar_pair(scalar_precompute(raw1, false),
                                       scalar_precompute(raw2, true), 0.0F);
  const float below = std::nextafter(
      probe.ecm, -std::numeric_limits<float>::infinity());
  const float above = std::nextafter(
      probe.ecm, std::numeric_limits<float>::infinity());
  const SpectrumConfig broad = {0.1, 400.3, 97U};
  compare_case(workspace, {raw1}, {raw2}, below, broad,
               "strict threshold matches one ULP below ecm");
  compare_case(workspace, {raw1}, {raw2}, probe.ecm, broad,
               "strict threshold matches exactly at ecm");
  compare_case(workspace, {raw1}, {raw2}, above, broad,
               "strict threshold matches one ULP above ecm");

  const double W = probe.invariant_mass;
  compare_case(workspace, {raw1}, {raw2}, 0.0F,
               {W, std::nextafter(W, std::numeric_limits<double>::infinity()),
                1U},
               "W exactly on the lower edge enters the first bin");
  compare_case(workspace, {raw1}, {raw2}, 0.0F,
               {std::nextafter(W,
                               -std::numeric_limits<double>::infinity()),
                W, 1U},
               "W exactly on the upper edge is overflow");
  compare_case(workspace, {raw1}, {raw2}, 0.0F,
               {W - 0.03125, W + 0.09375, 7U},
               "non-dyadic spectrum width preserves bin selection");
}

void test_block_boundaries_and_randomized() {
  std::printf("fixed-block boundaries and randomized finite oracle\n");
  blocked::Workspace workspace;
  const SpectrumConfig spectrum = {0.05, 700.125, 211U};
  for (const std::size_t size : {1U, 127U, 128U, 129U, 257U}) {
    std::vector<RawParticle> second;
    second.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
      const float sign = index % 3U == 0U ? -1.0F : 1.0F;
      second.push_back({sign * (70.0F + 0.03125F * index),
                        -0.015F + 0.00003F * index,
                        0.011F - 0.00002F * index,
                        0.25F + 0.125F * (index % 11U)});
    }
    const std::vector<RawParticle> first = {
        {125.0F, 0.003F, -0.004F, 1.0F},
        {-98.0F, -0.006F, 0.002F, 0.75F}};
    const ScalarResults scalar = run_scalar(first, second, 111.25F, spectrum);
    bool contract_ok = false;
    const CandidateResults candidate = run_candidate(
        workspace, first, second, 111.25F, spectrum, contract_ok);
    const std::string description =
        "beam-2 size " + std::to_string(size) +
        " preserves results across block/tail boundaries";
    check(contract_ok && same_results(candidate, scalar),
          description.c_str());
  }

  std::mt19937 generator(0x6c756d69U);
  std::uniform_real_distribution<float> energy(0.001F, 250.0F);
  std::uniform_real_distribution<float> velocity(-0.21F, 0.21F);
  std::uniform_real_distribution<float> weight(0.0F, 2.5F);
  bool all_match = true;
  for (int trial = 0; trial < 48 && all_match; ++trial) {
    const std::size_t first_size = static_cast<std::size_t>(trial % 9);
    const std::size_t second_size = static_cast<std::size_t>(
        (trial * 37) % 260);
    std::vector<RawParticle> first;
    std::vector<RawParticle> second;
    for (std::size_t index = 0; index < first_size; ++index) {
      float value = energy(generator);
      if ((trial + static_cast<int>(index)) % 3 == 0) value = -value;
      first.push_back({value, velocity(generator), velocity(generator),
                       weight(generator)});
    }
    for (std::size_t index = 0; index < second_size; ++index) {
      float value = energy(generator);
      if ((trial + static_cast<int>(index)) % 4 == 0) value = -value;
      second.push_back({value, velocity(generator), velocity(generator),
                        weight(generator)});
    }
    const float threshold = energy(generator);
    const ScalarResults scalar =
        run_scalar(first, second, threshold, spectrum);
    bool contract_ok = false;
    const CandidateResults candidate = run_candidate(
        workspace, first, second, threshold, spectrum, contract_ok);
    all_match = contract_ok && same_results(candidate, scalar);
  }
  check(all_match,
        "48 randomized finite sign/weight/threshold/block cases match bits");
}

}  // namespace

int main() {
  test_alignment_capacity_and_contracts();
  test_empty_zero_and_signed_energy();
  test_every_materialized_lane();
  test_threshold_and_spectrum_edges();
  test_block_boundaries_and_randomized();
  if (failures != 0) {
    std::fprintf(stderr, "%d blocked electron-luminosity checks failed\n",
                 failures);
    return 1;
  }
  std::puts("all blocked electron-luminosity checks passed");
  return 0;
}
