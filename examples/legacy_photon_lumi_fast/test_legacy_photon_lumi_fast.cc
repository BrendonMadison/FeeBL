#include "legacyPhotonLumiFastCPP.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using feebl::legacy_photon_lumi::Algorithm;
using feebl::legacy_photon_lumi::ChargedSample;
using feebl::legacy_photon_lumi::Features;
using feebl::legacy_photon_lumi::PhotonView;
using feebl::legacy_photon_lumi::PhotonSample;
using feebl::legacy_photon_lumi::Result;
using feebl::legacy_photon_lumi::ChargedView;
using feebl::legacy_photon_lumi::Workspace;
using feebl::legacy_photon_lumi::accumulate_cell;
using feebl::legacy_photon_lumi::algorithm_name;
using feebl::legacy_photon_lumi::compute_cell;
using feebl::legacy_photon_lumi::eligible;
using feebl::legacy_photon_lumi::gg_above_threshold;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-76s %s\n", description, condition ? "PASS" : "FAIL");
  if (!condition) ++failures;
}

bool close(double actual, double expected, double relative = 2.0e-6,
           double absolute = 1.0e-12) {
  return std::fabs(actual - expected) <=
         absolute + relative * std::fabs(expected);
}

bool equivalent(const Result& actual, const Result& expected,
                double relative = 2.0e-6) {
  bool ok = close(actual.lumi_ge, expected.lumi_ge, relative) &&
            close(actual.lumi_eg, expected.lumi_eg, relative) &&
            close(actual.lumi_gg, expected.lumi_gg, relative) &&
            close(actual.lumi_gg_high, expected.lumi_gg_high, relative);
  for (std::size_t i = 0; i < actual.class_luminosity.size(); ++i)
    ok = ok && close(actual.class_luminosity[i],
                     expected.class_luminosity[i], relative);
  return ok;
}

struct PhotonProbe {
  const std::vector<PhotonSample>* samples;
  mutable std::size_t loads;
};

struct ChargedProbe {
  const std::vector<ChargedSample>* samples;
  mutable std::size_t loads;
};

PhotonSample load_photon_probe(const void* context, std::size_t index) {
  const PhotonProbe* probe = static_cast<const PhotonProbe*>(context);
  ++probe->loads;
  return (*probe->samples)[index];
}

ChargedSample load_charged_probe(const void* context, std::size_t index) {
  const ChargedProbe* probe = static_cast<const ChargedProbe*>(context);
  ++probe->loads;
  return (*probe->samples)[index];
}

template <class Function>
bool throws(Function function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void test_known_answer() {
  std::printf("\nLegacy semantic known answer\n");
  const std::vector<PhotonSample> g1 = {
      {2.0f, 0.25f}, {0.0f, 7.0f}, {4.0f, 0.75f}};
  const std::vector<ChargedSample> e2 = {
      {-45.6f, 2.0f}, {45.6f, 3.0f}};
  const std::vector<PhotonSample> g2 = {
      {2.0f, 5.0f}, {3.0f, 0.5f}, {1.0f, 2.0f}};
  const std::vector<ChargedSample> e1 = {
      {-45.6f, 0.5f}, {45.6f, 1.5f}};
  const Result direct = compute_cell(
      g1, e2, g2, e1, 16.0, Algorithm::CartesianReference);
  const Result fast = compute_cell(
      g1, e2, g2, e1, 16.0, Algorithm::FactoredExact);

  check(close(direct.lumi_ge, (0.25 + 0.75) * (2.0 + 3.0)),
        "gamma-e normalization matches a hand calculation");
  check(close(direct.lumi_eg, (5.0 + 0.5 + 2.0) * (0.5 + 1.5)),
        "e-gamma normalization matches a hand calculation");
  check(close(direct.class_luminosity[2],
              (5.0 + 0.5 + 2.0) * 1.5) &&
            close(direct.class_luminosity[5],
                  (5.0 + 0.5 + 2.0) * 0.5),
        "e-gamma signed-energy class convention matches collide_eg");
  check(close(direct.class_luminosity[8],
              (0.25 + 7.0 + 0.75) * (5.0 + 0.5 + 2.0)),
        "gamma-gamma class keeps legacy pre-ECM zero-energy accounting");
  check(close(direct.lumi_gg,
              (0.25 + 0.75) * (5.0 + 0.5 + 2.0)),
        "gamma-gamma total keeps only positive-ECM pairs");
  // Strict 16 GeV^2 boundary: (2,2) and (4,1) are excluded.
  check(close(direct.lumi_gg_high,
              0.25 * 0.5 + 0.75 * (5.0 + 0.5)),
        "strict direct-product gg threshold selects the expected physical pairs");
  check(equivalent(fast, direct, 2.0e-7),
        "factorized result agrees with direct legacy enumeration");
  check(direct.cartesian_pairs_visited == 21 &&
            fast.cartesian_pairs_visited == 0,
        "factorized path eliminates every Cartesian visit");
  check(fast.ge_pair_opportunities == 6 &&
            fast.eg_pair_opportunities == 6 &&
            fast.gg_pair_opportunities == 9,
        "logical pair-opportunity counters retain provenance");
}

void test_randomized_oracle() {
  std::printf("\nRandomized Cartesian oracle\n");
  std::mt19937 random(20260826u);
  std::uniform_real_distribution<float> log_energy(-8.0f, 6.0f);
  std::uniform_real_distribution<float> weight(0.01f, 20.0f);
  bool all_match = true;
  for (int trial = 0; trial < 80; ++trial) {
    std::vector<PhotonSample> g1;
    std::vector<PhotonSample> g2;
    std::vector<ChargedSample> e1;
    std::vector<ChargedSample> e2;
    const std::size_t n1 = static_cast<std::size_t>(trial % 19);
    const std::size_t n2 = static_cast<std::size_t>((trial * 7) % 23);
    const std::size_t m1 = static_cast<std::size_t>((trial * 11) % 17);
    const std::size_t m2 = static_cast<std::size_t>((trial * 13) % 29);
    for (std::size_t i = 0; i < n1; ++i)
      g1.push_back({i % 13 == 0 ? 0.0f : std::exp(log_energy(random)),
                    weight(random)});
    for (std::size_t i = 0; i < n2; ++i)
      g2.push_back({i % 11 == 0 ? 0.0f : std::exp(log_energy(random)),
                    weight(random)});
    for (std::size_t i = 0; i < m1; ++i)
      e1.push_back({i % 2 ? 45.6f : -45.6f, weight(random)});
    for (std::size_t i = 0; i < m2; ++i)
      e2.push_back({i % 3 ? 125.0f : -125.0f, weight(random)});
    const double smin = trial % 5 == 0
        ? 0.0 : std::exp(static_cast<double>(log_energy(random)));
    const Result direct = compute_cell(
        g1, e2, g2, e1, smin, Algorithm::CartesianReference);
    const Result fast = compute_cell(
        g1, e2, g2, e1, smin, Algorithm::FactoredExact);
    all_match = all_match && equivalent(fast, direct) &&
                fast.ge_pair_opportunities == direct.ge_pair_opportunities &&
                fast.eg_pair_opportunities == direct.eg_pair_opportunities &&
                fast.gg_pair_opportunities == direct.gg_pair_opportunities &&
                fast.cartesian_pairs_visited == 0;
  }
  check(all_match,
        "80 variable-size cells agree with the direct legacy oracle");
}

void test_nonowning_views_and_workspace_reuse() {
  std::printf("\nNon-owning production view and workspace reuse\n");
  const std::vector<PhotonSample> g1 = {
      {0.0f, 9.0f}, {0.5f, 1.0f}, {3.0f, 2.0f}, {8.0f, 4.0f}};
  const std::vector<PhotonSample> g2 = {
      {1.0f, 5.0f}, {0.0f, 7.0f}, {2.0f, 6.0f}, {7.0f, 8.0f},
      {20.0f, 3.0f}};
  const std::vector<ChargedSample> e1 = {
      {-45.6f, 0.5f}, {0.0f, 1.0f}, {45.6f, 1.5f}};
  const std::vector<ChargedSample> e2 = {
      {-125.0f, 2.0f}, {125.0f, 3.0f}};
  const Result reference = compute_cell(
      g1, e2, g2, e1, 12.0, Algorithm::CartesianReference);

  PhotonProbe pg1 = {&g1, 0u};
  PhotonProbe pg2 = {&g2, 0u};
  ChargedProbe pe1 = {&e1, 0u};
  ChargedProbe pe2 = {&e2, 0u};
  const PhotonView vg1(&pg1, g1.size(), &load_photon_probe);
  const PhotonView vg2(&pg2, g2.size(), &load_photon_probe);
  const ChargedView ve1(&pe1, e1.size(), &load_charged_probe);
  const ChargedView ve2(&pe2, e2.size(), &load_charged_probe);
  Workspace workspace;
  workspace.reserve(g1.size(), g2.size());
  const std::size_t query_capacity = workspace.query_capacity();
  const std::size_t weighted_capacity = workspace.weighted_capacity();
  const std::size_t suffix_capacity = workspace.suffix_capacity();

  Result first;
  accumulate_cell(vg1, ve2, vg2, ve1, 12.0, Algorithm::FactoredExact,
                  workspace, &first);
  check(equivalent(first, reference),
        "non-owning v2 entry point agrees with the Cartesian oracle");
  check(pg1.loads == g1.size() && pg2.loads == g2.size() &&
            pe1.loads == e1.size() && pe2.loads == e2.size(),
        "views make one bounded pass over each sampled cell list");
  check(workspace.query_capacity() == query_capacity &&
            workspace.weighted_capacity() == weighted_capacity &&
            workspace.suffix_capacity() == suffix_capacity,
        "pre-reserved workspace performs no growth in the first cell");

  Result second;
  accumulate_cell(vg1, ve2, vg2, ve1, 12.0, Algorithm::FactoredExact,
                  workspace, &second);
  check(equivalent(second, reference) &&
            workspace.query_capacity() == query_capacity &&
            workspace.weighted_capacity() == weighted_capacity &&
            workspace.suffix_capacity() == suffix_capacity,
        "repeated cells reuse storage without changing the result");
  check(std::string(algorithm_name(Algorithm::FactoredExact)) ==
            "factorized-exact-v2",
        "provenance names the optimized v2 implementation");

  const PhotonView invalid(0, 1u, &load_photon_probe);
  check(throws([&] {
          Result rejected;
          accumulate_cell(invalid, ve2, vg2, ve1, 12.0,
                          Algorithm::FactoredExact, workspace, &rejected);
        }),
        "non-empty view with a null context fails closed");
}

void test_threshold_ulps() {
  std::printf("\nStrict threshold ULP probes\n");
  const std::vector<PhotonSample> g1 = {
      {0.125f, 1.0f}, {1.0f, 2.0f}, {9.5f, 3.0f}, {1000.0f, 4.0f}};
  const std::vector<PhotonSample> g2 = {
      {0.25f, 5.0f}, {2.0f, 6.0f}, {7.0f, 7.0f}, {800.0f, 8.0f}};
  const std::vector<ChargedSample> empty;
  bool all_match = true;
  for (std::size_t a = 0; a < g1.size(); ++a) {
    for (std::size_t b = 0; b < g2.size(); ++b) {
      const double boundary =
          (4.0 * static_cast<double>(g1[a].energy)) *
          static_cast<double>(g2[b].energy);
      const double probes[3] = {
          std::nextafter(boundary, 0.0), boundary,
          std::nextafter(boundary, std::numeric_limits<double>::infinity())};
      for (int p = 0; p < 3; ++p) {
        const Result direct = compute_cell(
            g1, empty, g2, empty, probes[p], Algorithm::CartesianReference);
        const Result fast = compute_cell(
            g1, empty, g2, empty, probes[p], Algorithm::FactoredExact);
        all_match = all_match && close(fast.lumi_gg_high,
                                        direct.lumi_gg_high, 1.0e-14);
      }
    }
  }
  check(all_match,
        "factorized suffix boundary matches 48 direct-product predicates");

  const float real_card_ecm_min_gg = 0.3f;
  const float real_card_cut =
      4.0f * real_card_ecm_min_gg * real_card_ecm_min_gg;
  const float equality_partner = real_card_cut * 0.25f;
  check(!gg_above_threshold(1.0f, equality_partner, real_card_cut) &&
            gg_above_threshold(
                1.0f,
                std::nextafter(equality_partner,
                               std::numeric_limits<float>::infinity()),
                real_card_cut) &&
            !gg_above_threshold(
                1.0f, std::nextafter(equality_partner, 0.0f), real_card_cut),
        "real-card float cut excludes equality and resolves adjacent ULPs");
}

void test_fail_closed() {
  std::printf("\nFail-closed capability and input validation\n");
  Features features;
  std::string reason;
  check(eligible(features, &reason) && reason.empty(),
        "pure luminosity aggregate feature set is eligible");
  Features disabled;
  disabled.do_photon_lumi = 0;
  check(!eligible(disabled, &reason) &&
            reason == "photon luminosity is disabled",
        "selector cannot claim a disabled photon-luminosity path");

  int Features::* prohibited[] = {
      &Features::do_jets, &Features::do_compt, &Features::do_compt_phot,
      &Features::do_pairs, &Features::do_muons, &Features::do_hadrons,
      &Features::do_coherent, &Features::do_trident, &Features::do_cross,
      &Features::do_bhabhas, &Features::do_prod, &Features::do_tertphot};
  bool rejects_all = true;
  for (std::size_t i = 0; i < sizeof(prohibited) / sizeof(prohibited[0]); ++i) {
    Features candidate;
    candidate.*prohibited[i] = 1;
    rejects_all = rejects_all && !eligible(candidate, &reason) &&
                  !reason.empty();
  }
  Features ge_heap;
  ge_heap.do_lumi = 2;
  Features gg_heap;
  gg_heap.do_lumi = 4;
  rejects_all = rejects_all && !eligible(ge_heap, &reason) &&
                !eligible(gg_heap, &reason);
  check(rejects_all,
        "every pair side effect, background generator, and heap is rejected");

  const std::vector<ChargedSample> particles = {{45.6f, 1.0f}};
  const std::vector<PhotonSample> valid = {{1.0f, 1.0f}};
  const std::vector<PhotonSample> negative_energy = {{-1.0f, 1.0f}};
  const std::vector<PhotonSample> nan_weight = {{1.0f,
      std::numeric_limits<float>::quiet_NaN()}};
  check(throws([&] {
          compute_cell(negative_energy, particles, valid, particles, 0.0,
                       Algorithm::FactoredExact);
        }),
        "negative photon energy aborts instead of changing legacy semantics");
  check(throws([&] {
          compute_cell(valid, particles, nan_weight, particles, 0.0,
                       Algorithm::FactoredExact);
        }),
        "non-finite weight is rejected");
  check(throws([&] {
          compute_cell(valid, particles, valid, particles, -1.0,
                       Algorithm::FactoredExact);
        }),
        "invalid gamma-gamma threshold is rejected");
  const std::vector<PhotonSample> overflow_weight = {
      {1.0f, std::numeric_limits<float>::max()}};
  const std::vector<PhotonSample> double_weight = {{1.0f, 2.0f}};
  check(throws([&] {
          compute_cell(overflow_weight, particles, double_weight, particles,
                       0.0, Algorithm::FactoredExact);
        }),
        "legacy float pair-weight overflow is rejected");
  const std::vector<PhotonSample> subnormal_weight = {
      {1.0f, std::numeric_limits<float>::min() * 0.5f}};
  check(throws([&] {
          compute_cell(subnormal_weight, particles, valid, particles, 0.0,
                       Algorithm::FactoredExact);
        }),
        "legacy float pair-weight subnormal range is rejected");
  const std::vector<ChargedSample> subnormal_signed_energy = {
      {-std::numeric_limits<float>::denorm_min(), 1.0f}};
  check(throws([&] {
          compute_cell(valid, subnormal_signed_energy, valid, particles, 0.0,
                       Algorithm::FactoredExact);
        }),
        "charged-energy subnormal sign cannot be changed by DAZ");
}

}  // namespace

int main() {
  test_known_answer();
  test_randomized_oracle();
  test_nonowning_views_and_workspace_reuse();
  test_threshold_ulps();
  test_fail_closed();
  std::printf("\n%s (%d failure%s)\n",
              failures ? "FAILURES PRESENT" : "all legacy fast-path checks passed",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
