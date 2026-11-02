#include "counterRngCPP.h"
#include "modernPhotonSoACPP.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using feebl::kNoParentStableIdV2;
using feebl::modern::ArrayOffset;
using feebl::modern::ParticleIndex;
using feebl::modern::PhotonPool;
using feebl::modern::PhotonRecord;
using feebl::modern::PhotonSoA;
using feebl::modern::PhotonStorageLimits;
using feebl::modern::SliceIndex;
using feebl::modern::StableParticleId;
using feebl::modern::checked_photon_authoritative_bytes;
using feebl::modern::kPhotonBaseBytesPerPhoton;
using feebl::modern::kPhotonParentAbsent;
using feebl::modern::kPhotonStokesBytesPerPhoton;

namespace {

int failures = 0;

void check(bool condition, const char* description) {
  std::printf("  %-72s %s\n", description, condition ? "PASS" : "FAIL");
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

PhotonRecord photon(StableParticleId id, SliceIndex slice,
                    std::uint32_t beam_id = 1u) {
  PhotonRecord result;
  result.stable_id = id;
  result.parent_stable_id = 1000000u + id;
  result.beam_id = beam_id;
  result.direction = beam_id == 1u ? 1 : -1;
  result.slice = slice;
  result.x = 0.125 * static_cast<double>(id % 13u);
  result.y = -0.25 * static_cast<double>(id % 7u);
  result.z = 1.5 * static_cast<double>(slice);
  result.vx = 1.0e-5 * static_cast<double>(id % 17u);
  result.vy = -2.0e-5 * static_cast<double>(id % 11u);
  result.energy = 0.25 + static_cast<double>(id % 101u);
  result.macro_weight = 0.5 + 0.01 * static_cast<double>(id % 19u);
  result.flags = static_cast<std::uint32_t>((id & 7u) << 4u);
  return result;
}

PhotonRecord parentless_photon(StableParticleId id, SliceIndex slice) {
  PhotonRecord result = photon(id, slice);
  result.parent_stable_id = kNoParentStableIdV2;
  result.flags |= kPhotonParentAbsent;
  return result;
}

bool identical(const PhotonSoA& left, const PhotonSoA& right) {
  return left.slice_count() == right.slice_count() &&
         left.beam_id() == right.beam_id() &&
         left.direction() == right.direction() &&
         left.stokes_enabled() == right.stokes_enabled() &&
         left.slice_offsets() == right.slice_offsets() &&
         left.stable_ids() == right.stable_ids() &&
         left.parent_stable_ids() == right.parent_stable_ids() &&
         left.slices() == right.slices() && left.x() == right.x() &&
         left.y() == right.y() && left.z() == right.z() &&
         left.vx() == right.vx() && left.vy() == right.vy() &&
         left.energy() == right.energy() &&
         left.macro_weight() == right.macro_weight() &&
         left.stokes_1() == right.stokes_1() &&
         left.stokes_2() == right.stokes_2() &&
         left.stokes_3() == right.stokes_3() &&
         left.flags() == right.flags();
}

void test_empty_and_canonical_storage() {
  std::printf("\nPhotonSoA complete ownership and canonical CSR\n");

  PhotonSoA empty(4, 1);
  check(empty.empty() && empty.slice_offsets() ==
                             std::vector<ArrayOffset>({0, 0, 0, 0, 0}),
        "empty owner represents every declared slice without placeholders");
  check(static_cast<bool>(empty.validate()),
        "empty multi-slice owner passes exhaustive validation");
  check(empty.slice_range(2) == std::make_pair<ArrayOffset, ArrayOffset>(0, 0),
        "empty slice exposes an exact zero-width CSR range");
  check(throws([&empty] { (void)empty.slice_range(4); }) &&
            throws([&empty] { (void)empty.photon(0); }),
        "slice and photon bounds fail loudly");

  std::vector<PhotonRecord> staged = {
      photon(90, 2), photon(40, 0), parentless_photon(11, 2),
      photon(70, 3), photon(18, 0), photon(3, 3)};
  empty.assign_canonical(staged);
  check(empty.slice_offsets() ==
            std::vector<ArrayOffset>({0, 2, 2, 4, 6}),
        "two-pass counts and scan preserve empty and uneven slices");
  check(empty.stable_ids() ==
            std::vector<StableParticleId>({18, 40, 11, 90, 3, 70}),
        "child identity defines strict canonical order inside each slice");
  check(empty.slices() ==
            std::vector<SliceIndex>({0, 0, 2, 2, 3, 3}),
        "explicit dense slice labels agree with CSR grouping");
  check(empty.photon(2).parent_stable_id == kNoParentStableIdV2 &&
            (empty.photon(2).flags & kPhotonParentAbsent) != 0u &&
            empty.photon(4).stable_id == 3 && empty.photon(4).direction == 1 &&
            empty.beam_id() == 1u,
        "record reconstruction preserves identity, provenance and direction");
  check(static_cast<bool>(empty.validate()),
        "canonical uneven population passes exhaustive validation");

  std::uint64_t expected_bytes = 0;
  check(kPhotonBaseBytesPerPhoton == 84u &&
            checked_photon_authoritative_bytes(4, 6, expected_bytes) &&
            expected_bytes == 5u * sizeof(ArrayOffset) + 6u * 84u &&
            empty.authoritative_bytes() == expected_bytes,
        "persistent byte accounting exactly sums every authoritative array");
}

void test_identity_and_physical_guards() {
  std::printf("\nPhoton identity, lineage and physical validation\n");

  check(throws([] { PhotonSoA invalid_owner(2, 0); }) &&
            throws([] { PhotonSoA invalid_owner(2, 3); }),
        "owner construction rejects invalid FeeBL beam identities");

  PhotonSoA owner(2, 1);
  check(throws([&owner] {
          owner.assign_canonical({photon(7, 0), photon(7, 1)});
        }),
        "duplicate child identity is rejected globally across slices");

  PhotonRecord bad = photon(kNoParentStableIdV2, 0);
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "reserved no-parent sentinel cannot identify a physical photon");

  bad = photon(8, 0);
  bad.parent_stable_id = kNoParentStableIdV2;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "sentinel parent requires an explicit parent-absent provenance flag");

  bad = photon(9, 0);
  bad.flags |= kPhotonParentAbsent;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "parent-absent flag cannot silently discard a real parent identity");

  bad = photon(19, 0);
  bad.parent_stable_id = bad.stable_id;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "self-parenting photon lineage is rejected");

  owner.assign_canonical({parentless_photon(10, 1)});
  check(static_cast<bool>(owner.validate()),
        "explicitly parentless photon is valid and unambiguous");

  bad = photon(11, 0, 2);
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "cross-beam staging is rejected by immutable pool ownership");
  bad = photon(20, 0);
  bad.direction = -1;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "beam/direction contradiction is rejected");
  bad = photon(12, 0);
  bad.slice = 2;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "out-of-range slice is rejected before publication");
  bad = photon(13, 0);
  bad.energy = -1.0;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "negative photon energy is rejected");
  bad = photon(14, 0);
  bad.macro_weight = -0.01;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "negative macro weight is rejected");
  bad = photon(15, 0);
  bad.x = std::numeric_limits<double>::quiet_NaN();
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "non-finite phase-space scalar is rejected");
  bad = photon(16, 0);
  bad.vx = 0.8;
  bad.vy = 0.7;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "superluminal transverse direction norm is rejected");
  bad = photon(17, 0);
  bad.vx = 0.6;
  bad.vy = 0.8;
  check(throws([&owner, bad] { owner.assign_canonical({bad}); }),
        "zero-longitudinal photon is rejected by the beam-owned scheduler");
  bad.vy = 0.79;
  owner.assign_canonical({bad});
  check(static_cast<bool>(owner.validate()),
        "strictly positive longitudinal-component boundary is accepted");

  PhotonRecord polarized = photon(30, 1, 2);
  polarized.stokes_1 = 0.6;
  polarized.stokes_2 = 0.8;
  check(throws([&owner, polarized] {
          PhotonRecord wrong_owner = polarized;
          wrong_owner.beam_id = 1;
          wrong_owner.direction = 1;
          owner.assign_canonical({wrong_owner});
        }),
        "unpolarized owner refuses to discard supplied Stokes state");

  PhotonSoA stokes_owner(2, 2, PhotonStorageLimits{}, true);
  stokes_owner.assign_canonical({polarized});
  check(stokes_owner.stokes_enabled() && stokes_owner.beam_id() == 2u &&
            stokes_owner.direction() == -1 &&
            stokes_owner.stokes_1()[0] == 0.6 &&
            stokes_owner.stokes_2()[0] == 0.8 &&
            stokes_owner.stokes_3()[0] == 0.0 &&
            static_cast<bool>(stokes_owner.validate()),
        "optional all-or-none Stokes layout preserves physical polarization");
  polarized.stokes_2 = 0.81;
  check(throws([&stokes_owner, polarized] {
          stokes_owner.assign_canonical({polarized});
        }),
        "Stokes norm beyond the density-matrix bound is rejected");
  polarized = photon(31, 1, 2);
  polarized.stokes_3 = std::numeric_limits<double>::quiet_NaN();
  check(throws([&stokes_owner, polarized] {
          stokes_owner.assign_canonical({polarized});
        }),
        "non-finite Stokes state is rejected");

  std::uint64_t polarized_bytes = 0;
  check(checked_photon_authoritative_bytes(2, 1, polarized_bytes, true) &&
            polarized_bytes == 3u * sizeof(ArrayOffset) +
                                    kPhotonBaseBytesPerPhoton +
                                    kPhotonStokesBytesPerPhoton,
        "Stokes-enabled layout has explicit exact byte accounting");
}

void test_transactional_limits_and_overflow() {
  std::printf("\nPhotonPool strong transactions and checked bounds\n");

  PhotonStorageLimits limits;
  limits.max_photons = 4;
  limits.max_photons_per_slice = 2;
  limits.max_authoritative_bytes =
      3u * sizeof(ArrayOffset) + 4u * kPhotonBaseBytesPerPhoton;
  PhotonPool pool(2, 1, limits);
  pool.append_transaction({photon(5, 0), photon(1, 1), photon(3, 0)});
  PhotonSoA snapshot = pool.photons();
  check(pool.size() == 3 && static_cast<bool>(pool.photons().validate()),
        "bounded pool publishes a valid canonical first transaction");

  check(throws([&pool] { pool.append_transaction({photon(6, 0)}); }) &&
            identical(pool.photons(), snapshot),
        "per-slice overflow leaves every previously published array unchanged");
  check(throws([&pool] {
          pool.append_transaction({photon(7, 1), photon(8, 1)});
        }) &&
            identical(pool.photons(), snapshot),
        "total-capacity overflow cannot partially publish a batch");
  check(throws([&pool] { pool.append_transaction({photon(1, 1)}); }) &&
            identical(pool.photons(), snapshot),
        "duplicate identity failure preserves the pre-transaction generation");

  PhotonRecord invalid = photon(9, 1);
  invalid.energy = std::numeric_limits<double>::infinity();
  check(throws([&pool, invalid] { pool.replace_transaction({invalid}); }) &&
            identical(pool.photons(), snapshot),
        "failed full replacement also has the strong publication guarantee");

  pool.replace_transaction({photon(22, 1)});
  check(pool.size() == 1 && pool.photons().stable_ids()[0] == 22,
        "successful replacement atomically supersedes the old generation");
  pool.clear_transaction();
  check(pool.size() == 0 && static_cast<bool>(pool.photons().validate()),
        "transactional clear retains a complete empty CSR owner");

  std::uint64_t bytes = 123;
  check(!checked_photon_authoritative_bytes(
            std::numeric_limits<SliceIndex>::max(), 0, bytes) &&
            bytes == 0,
        "slice-offset addition overflow is detected without saturation");
  check(!checked_photon_authoritative_bytes(
            0, std::numeric_limits<ParticleIndex>::max(), bytes) &&
            bytes == 0,
        "per-photon byte multiplication overflow is detected without wrap");
  check(throws([] {
          PhotonSoA impossible(std::numeric_limits<SliceIndex>::max(), 1);
          (void)impossible;
        }),
        "unrepresentable slice count fails before any offset allocation");

  PhotonStorageLimits tiny_bytes;
  tiny_bytes.max_authoritative_bytes = 2u * sizeof(ArrayOffset) - 1u;
  check(throws([tiny_bytes] {
          PhotonSoA impossible(1, 1, tiny_bytes);
          (void)impossible;
        }),
        "even empty storage must fit the explicit persistent byte budget");

  PhotonStorageLimits peak_limited;
  std::uint64_t one_authoritative = 0;
  check(checked_photon_authoritative_bytes(1, 1, one_authoritative),
        "single-photon transaction layout is representable");
  const std::uint64_t empty_authoritative = 2u * sizeof(ArrayOffset);
  peak_limited.max_transaction_bytes =
      2u * empty_authoritative + 2u * sizeof(PhotonRecord) +
      sizeof(StableParticleId) + one_authoritative - 1u;
  PhotonPool peak_pool(1, 1, peak_limited);
  check(throws([&peak_pool] {
          peak_pool.append_transaction({photon(55, 0)});
        }) && peak_pool.size() == 0,
        "one byte below the append peak budget fails before candidate allocation");

  PhotonStorageLimits lvalue_limited = peak_limited;
  lvalue_limited.max_transaction_bytes =
      2u * empty_authoritative + 3u * sizeof(PhotonRecord) +
      sizeof(StableParticleId) + one_authoritative - 1u;
  PhotonPool lvalue_pool(1, 1, lvalue_limited);
  const std::vector<PhotonRecord> lvalue_candidate = {photon(56, 0)};
  check(throws([&] {
          lvalue_pool.append_transaction(lvalue_candidate);
        }) && lvalue_pool.size() == 0,
        "lvalue clone is included in exact append peak preflight");
}

void test_schedule_independent_high_occupancy() {
  std::printf("\nSchedule-independent canonical publication at high occupancy\n");

  constexpr std::size_t kCount = 32768;
  constexpr SliceIndex kSlices = 64;
  std::vector<PhotonRecord> forward;
  forward.reserve(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    const StableParticleId id = static_cast<StableParticleId>(i + 1u);
    const SliceIndex slice = static_cast<SliceIndex>((i * 37u) % kSlices);
    forward.push_back(photon(id, slice));
  }
  std::vector<PhotonRecord> reverse = forward;
  std::reverse(reverse.begin(), reverse.end());
  std::vector<PhotonRecord> rotated = forward;
  std::rotate(rotated.begin(), rotated.begin() + 7919, rotated.end());

  PhotonStorageLimits limits;
  limits.max_photons = kCount;
  limits.max_photons_per_slice = kCount / kSlices;
  std::uint64_t byte_limit = 0;
  check(checked_photon_authoritative_bytes(kSlices, kCount, byte_limit),
        "high-occupancy byte budget is representable");
  limits.max_authoritative_bytes = byte_limit;

  PhotonSoA a(kSlices, 1, limits);
  PhotonSoA b(kSlices, 1, limits);
  PhotonSoA c(kSlices, 1, limits);
  a.assign_canonical(std::move(forward));
  b.assign_canonical(std::move(reverse));
  c.assign_canonical(std::move(rotated));
  check(identical(a, b) && identical(a, c),
        "three completion-order permutations publish byte-identical arrays");
  check(static_cast<bool>(a.validate()) &&
            a.authoritative_bytes() == byte_limit &&
            a.slice_range(0).second - a.slice_range(0).first ==
                kCount / kSlices,
        "high-occupancy owner meets exact shape, budget and validation gates");

  std::vector<std::vector<PhotonRecord>> batches(8);
  for (ParticleIndex i = 0; i < a.size(); ++i)
    batches[static_cast<std::size_t>(i % batches.size())].push_back(a.photon(i));
  PhotonPool ordered(kSlices, 1, limits);
  PhotonPool reordered(kSlices, 1, limits);
  ordered.append_batches_transaction(batches);
  std::vector<std::vector<PhotonRecord>> reversed_batches = batches;
  std::reverse(reversed_batches.begin(), reversed_batches.end());
  reordered.append_batches_transaction(reversed_batches);
  check(identical(ordered.photons(), reordered.photons()) &&
            identical(ordered.photons(), a),
        "one atomic generation transaction is invariant to worker completion order");
}

}  // namespace

int main() {
  std::printf("FeeBL modern photon SoA/pool foundation tests\n");
  test_empty_and_canonical_storage();
  test_identity_and_physical_guards();
  test_transactional_limits_and_overflow();
  test_schedule_independent_high_occupancy();
  std::printf("\nPhoton SoA/pool result: %s (%d failure%s)\n",
              failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
