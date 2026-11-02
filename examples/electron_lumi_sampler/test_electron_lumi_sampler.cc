#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "electronLumiSamplerCPP.h"
#include "counterRngCPP.h"

namespace sampling = feebl::electron_lumi_sampling;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

sampling::PairIdentity identity(std::uint64_t ordinal) {
  sampling::PairIdentity result;
  result.run_seed = 0x123456789abcdef0ull;
  result.crossing = 3U;
  result.diagonal = ordinal / 29U;
  result.wave = ordinal / 11U;
  result.substep = ordinal / 5U;
  result.encounter = ordinal % 17U;
  result.parent_a = 0x1100000000000000ull + ordinal;
  result.parent_b = 0x1200000000010000ull + 3U * ordinal;
  result.interaction = ordinal % 7U;
  return result;
}

double weight(std::uint64_t ordinal) {
  return 0.125 + static_cast<double>((ordinal * 37U) % 101U) / 13.0;
}

double independent_priority(const sampling::PairIdentity& value,
                            double candidate_weight) {
  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = value.run_seed;
  key.domain = feebl::RandomDomain::LuminosityEeSampling;
  key.crossing = value.crossing;
  key.diagonal = value.diagonal;
  key.wave = value.wave;
  key.substep = value.substep;
  key.encounter = value.encounter;
  key.beam = 0U;
  key.parent_a = value.parent_a;
  key.parent_b = value.parent_b;
  key.interaction = value.interaction;
  key.emission = 0U;
  key.rejection_attempt = 0U;
  return -std::log(feebl::CounterRngV2::uniform_double(key, 0U)) /
         candidate_weight;
}

bool same_selection(const std::vector<sampling::Selection>& left,
                    const std::vector<sampling::Selection>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!(left[index].identity == right[index].identity) ||
        left[index].priority != right[index].priority ||
        left[index].payload_token != right[index].payload_token)
      return false;
  return true;
}

bool same_selection_with_slots(
    const std::vector<sampling::Selection>& left,
    const std::vector<sampling::Selection>& right) {
  if (!same_selection(left, right)) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index].storage_slot != right[index].storage_slot) return false;
  return true;
}

void sort_canonically(std::vector<sampling::Selection>* selected) {
  std::sort(selected->begin(), selected->end(),
            [](const sampling::Selection& left,
               const sampling::Selection& right) {
              return sampling::canonical_less(left.identity, right.identity);
            });
}

void test_traversal_and_lane_invariance() {
  constexpr std::size_t candidates = 4096U;
  constexpr std::size_t capacity = 73U;
  sampling::WeightedReservoir forward(capacity);
  sampling::WeightedReservoir reverse(capacity);
  for (std::size_t item = 0; item < candidates; ++item)
    check(forward.offer(identity(item), weight(item), item).status !=
              sampling::OfferStatus::InvalidWeight,
          "forward offer rejected a valid weight");
  for (std::size_t item = candidates; item-- > 0U;)
    reverse.offer(identity(item), weight(item), item);
  const std::vector<sampling::Selection> reference =
      forward.canonical_selection();
  check(reference.size() == capacity, "direct reservoir did not fill");
  check(same_selection(reference, reverse.canonical_selection()),
        "reverse traversal changed the selected stable identities");

  std::vector<sampling::Selection> independent;
  independent.reserve(candidates);
  for (std::size_t item = 0; item < candidates; ++item)
    independent.push_back(sampling::Selection{
        identity(item), independent_priority(identity(item), weight(item)),
        0U, item});
  std::sort(independent.begin(), independent.end(),
            [](const sampling::Selection& left,
               const sampling::Selection& right) {
              if (left.priority < right.priority) return true;
              if (right.priority < left.priority) return false;
              return sampling::canonical_less(left.identity, right.identity);
            });
  independent.resize(capacity);
  std::sort(independent.begin(), independent.end(),
            [](const sampling::Selection& left,
               const sampling::Selection& right) {
              return sampling::canonical_less(left.identity, right.identity);
            });
  for (std::size_t index = 0; index < capacity; ++index) {
    independent[index].storage_slot = reference[index].storage_slot;
  }
  check(same_selection(reference, independent),
        "bounded heap differs from an independent full-sort top-K oracle");

  const std::size_t worker_counts[] = {1U, 2U, 3U, 7U, 16U};
  for (std::size_t wc = 0; wc < sizeof(worker_counts) / sizeof(worker_counts[0]);
       ++wc) {
    const std::size_t workers = worker_counts[wc];
    std::vector<sampling::WeightedReservoir> lanes;
    lanes.reserve(workers);
    for (std::size_t lane = 0; lane < workers; ++lane)
      lanes.push_back(sampling::WeightedReservoir(capacity));
    for (std::size_t reverse_worker = 0; reverse_worker < workers;
         ++reverse_worker) {
      const std::size_t lane = workers - 1U - reverse_worker;
      for (std::size_t item = lane; item < candidates; item += workers)
        lanes[lane].offer(identity(item), weight(item), item);
    }
    sampling::WeightedReservoir merged;
    std::string reason;
    check(sampling::WeightedReservoir::merge_lanes(
              lanes, capacity, &merged, &reason),
          reason.empty() ? "lane merge failed" : reason.c_str());
    check(same_selection(reference, merged.canonical_selection()),
          "worker count/order changed the merged top-K reservoir");
  }
}

void test_fail_closed_contracts() {
  sampling::WeightedReservoir empty(0U);
  check(empty.offer(identity(1U), 1.0).status ==
            sampling::OfferStatus::InvalidCapacity,
        "zero capacity did not reject an offer");

  sampling::WeightedReservoir invalid(4U);
  sampling::PairIdentity missing = identity(1U);
  missing.parent_a = feebl::kNoParentStableIdV2;
  check(invalid.offer(missing, 1.0).status ==
            sampling::OfferStatus::InvalidIdentity,
        "missing stable parent did not fail closed");
  sampling::PairIdentity equal_beam_local_ids = identity(4U);
  equal_beam_local_ids.parent_b = equal_beam_local_ids.parent_a;
  check(invalid.offer(equal_beam_local_ids, 1.0).accepted(),
        "equal beam-local IDs were incorrectly treated as one particle");
  check(invalid.offer(identity(2U), 0.0).status ==
            sampling::OfferStatus::InvalidWeight,
        "zero weight did not fail closed");
  check(invalid.offer(identity(3U),
                      std::numeric_limits<double>::quiet_NaN()).status ==
            sampling::OfferStatus::InvalidWeight,
        "NaN weight did not fail closed");

  sampling::WeightedReservoir duplicate(4U);
  check(duplicate.offer(identity(9U), 2.0).accepted(),
        "first identity was not accepted");
  check(duplicate.offer(identity(9U), 2.0).status ==
            sampling::OfferStatus::DuplicateSelectedIdentity,
        "duplicate selected identity did not fail closed");
  check(!duplicate.healthy(), "duplicate identity did not poison reservoir");
  std::vector<sampling::Selection> poisoned_export;
  poisoned_export.reserve(duplicate.size());
  std::string poisoned_reason;
  check(!duplicate.selected_records_into(&poisoned_export, &poisoned_reason),
        "poisoned reservoir allowed a selected-record export");
  check(!poisoned_reason.empty(),
        "poisoned selected-record export lacked a diagnostic");

  std::vector<sampling::WeightedReservoir> lanes;
  lanes.push_back(sampling::WeightedReservoir(2U));
  lanes.push_back(sampling::WeightedReservoir(2U));
  lanes[0].offer(identity(12U), 1.0);
  lanes[1].offer(identity(12U), 1.0);
  sampling::WeightedReservoir merged;
  std::string reason;
  check(!sampling::WeightedReservoir::merge_lanes(
            lanes, 2U, &merged, &reason),
        "duplicate identity across lanes did not fail merge");
  check(reason.find("duplicate") != std::string::npos,
        "duplicate lane failure lacked an actionable diagnostic");
}

void test_canonical_order_and_bounded_storage() {
  sampling::WeightedReservoir reservoir(11U);
  for (std::size_t item = 0; item < 1000U; ++item)
    reservoir.offer(identity(999U - item), weight(item), 999U - item);
  check(reservoir.size() == 11U, "reservoir exceeded or missed capacity");
  const std::vector<sampling::Selection> selected =
      reservoir.canonical_selection();
  for (std::size_t index = 1; index < selected.size(); ++index)
    check(sampling::canonical_less(selected[index - 1U].identity,
                                   selected[index].identity),
          "output is not in strict canonical identity order");
  for (std::size_t index = 0; index < selected.size(); ++index)
    check(selected[index].storage_slot < reservoir.capacity(),
          "selection references storage outside the bounded payload array");
}

void test_preallocated_hot_path_and_benchmark() {
  constexpr std::size_t candidates = 200000U;
  constexpr std::size_t capacity = 257U;
  sampling::WeightedReservoir reservoir(capacity);
  const std::uint64_t rebuilds_before = reservoir.storage_rebuilds();
  check(reservoir.identity_table_slots() >= 2U * capacity,
        "flat identity table did not establish bounded spare capacity");

  const std::chrono::steady_clock::time_point offer_begin =
      std::chrono::steady_clock::now();
  for (std::size_t item = 0; item < candidates; ++item) {
    const sampling::OfferStatus status =
        reservoir.offer(identity(item), weight(item), item).status;
    check(status == sampling::OfferStatus::Inserted ||
              status == sampling::OfferStatus::Replaced ||
              status == sampling::OfferStatus::Rejected,
          "flat-table offer returned an invalid hot-path status");
  }
  const std::chrono::steady_clock::time_point offer_end =
      std::chrono::steady_clock::now();
  check(reservoir.storage_rebuilds() == rebuilds_before,
        "candidate offers rebuilt sampler storage in the hot path");

  std::vector<sampling::Selection> selected;
  std::string reason;
  check(!reservoir.canonical_selection_into(&selected, &reason),
        "undersized canonical scratch did not fail closed");
  check(reason.find("capacity") != std::string::npos,
        "undersized canonical scratch lacked an actionable diagnostic");
  selected.reserve(capacity);
  check(reservoir.canonical_selection_into(&selected, &reason),
        reason.empty() ? "preallocated canonical selection failed"
                       : reason.c_str());
  check(selected.size() == capacity,
        "preallocated canonical selection returned the wrong size");
  check(reservoir.storage_rebuilds() == rebuilds_before,
        "preallocated canonical selection rebuilt sampler storage");

  reason.clear();
  check(!reservoir.selected_records_into(0, &reason),
        "null selected-record destination did not fail closed");
  check(reason.find("null") != std::string::npos,
        "null selected-record destination lacked an actionable diagnostic");
  std::vector<sampling::Selection> selected_records;
  reason.clear();
  check(!reservoir.selected_records_into(&selected_records, &reason),
        "undersized no-sort scratch did not fail closed");
  check(reason.find("capacity") != std::string::npos,
        "undersized no-sort scratch lacked an actionable diagnostic");
  selected_records.reserve(capacity);
  check(reservoir.selected_records_into(&selected_records, &reason),
        reason.empty() ? "preallocated no-sort export failed"
                       : reason.c_str());
  check(selected_records.size() == capacity,
        "no-sort export returned the wrong number of records");
  std::vector<sampling::Selection> sorted_records = selected_records;
  sort_canonically(&sorted_records);
  check(same_selection_with_slots(selected, sorted_records),
        "no-sort export changed identity, priority, payload, or source slot");
  check(reservoir.storage_rebuilds() == rebuilds_before,
        "no-sort export rebuilt sampler storage");

  constexpr std::size_t workers = 8U;
  std::vector<sampling::WeightedReservoir> lanes;
  lanes.reserve(workers);
  for (std::size_t lane = 0; lane < workers; ++lane)
    lanes.push_back(sampling::WeightedReservoir(capacity));
  for (std::size_t item = 0; item < candidates; ++item)
    lanes[item % workers].offer(identity(item), weight(item), item);
  sampling::WeightedReservoir merged;
  check(sampling::WeightedReservoir::merge_lanes(
            lanes, capacity, &merged, &reason),
        reason.empty() ? "flat-table lane merge failed" : reason.c_str());
  const std::uint64_t merged_rebuilds = merged.storage_rebuilds();
  const std::chrono::steady_clock::time_point merge_begin =
      std::chrono::steady_clock::now();
  check(sampling::WeightedReservoir::merge_lanes(
            lanes, capacity, &merged, &reason),
        reason.empty() ? "repeat flat-table lane merge failed"
                       : reason.c_str());
  const std::chrono::steady_clock::time_point merge_end =
      std::chrono::steady_clock::now();
  check(merged.storage_rebuilds() == merged_rebuilds,
        "repeat lane merge rebuilt sampler storage");

  // Model the crossing-end consumer: export each worker's bounded heap without
  // sorting, then re-offer its records into one bounded reservoir.  This must
  // select exactly the same global top-K as the internal lane merge.
  sampling::WeightedReservoir exported_merge(capacity);
  std::vector<sampling::Selection> lane_records;
  lane_records.reserve(capacity);
  for (std::size_t lane = 0; lane < workers; ++lane) {
    check(lanes[lane].selected_records_into(&lane_records, &reason),
          reason.empty() ? "lane no-sort export failed" : reason.c_str());
    std::vector<sampling::Selection> canonical_lane = lane_records;
    sort_canonically(&canonical_lane);
    check(same_selection_with_slots(canonical_lane,
                                    lanes[lane].canonical_selection()),
          "lane no-sort export did not preserve source payload slots");
    for (std::size_t record = 0; record < lane_records.size(); ++record) {
      const sampling::OfferStatus status = exported_merge.offer_precomputed(
          lane_records[record].identity, lane_records[record].priority,
          lane_records[record].payload_token).status;
      check(status == sampling::OfferStatus::Inserted ||
                status == sampling::OfferStatus::Replaced ||
                status == sampling::OfferStatus::Rejected,
            "crossing-end no-sort re-offer failed");
    }
  }
  check(same_selection(merged.canonical_selection(),
                       exported_merge.canonical_selection()),
        "no-sort crossing-end merge changed the selected global top-K");

  const double offer_ms = std::chrono::duration<double, std::milli>(
      offer_end - offer_begin).count();
  const double merge_ms = std::chrono::duration<double, std::milli>(
      merge_end - merge_begin).count();
  std::cout << "sampler_benchmark candidates=" << candidates
            << " capacity=" << capacity
            << " lanes=" << workers
            << " offer_ms=" << offer_ms
            << " merge_ms=" << merge_ms
            << " hot_path_storage_rebuilds="
            << (reservoir.storage_rebuilds() - rebuilds_before)
            << " repeat_merge_storage_rebuilds="
            << (merged.storage_rebuilds() - merged_rebuilds)
            << std::endl;
}

}  // namespace

int main() {
  test_traversal_and_lane_invariance();
  test_fail_closed_contracts();
  test_canonical_order_and_bounded_storage();
  test_preallocated_hot_path_and_benchmark();
  std::cout << "PASS: deterministic weighted e-e luminosity sampler"
            << std::endl;
  return 0;
}
