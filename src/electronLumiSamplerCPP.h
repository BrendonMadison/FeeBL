#ifndef FEEBL_ELECTRON_LUMI_SAMPLER_CPP_H
#define FEEBL_ELECTRON_LUMI_SAMPLER_CPP_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace electron_lumi_sampling {

// Stable physical identity of one e-e luminosity opportunity.  No worker,
// lane, address, container index, or traversal ordinal is part of this key.
// parent_a is always the beam-1 primary ID and parent_b the beam-2 primary ID,
// so the pair is intentionally ordered rather than symmetrically canonical.
struct PairIdentity {
  std::uint64_t run_seed = 1U;
  std::uint64_t crossing = 0U;
  std::uint64_t diagonal = 0U;
  std::uint64_t wave = 0U;
  std::uint64_t substep = 0U;
  std::uint64_t encounter = 0U;
  std::uint64_t parent_a = 0U;
  std::uint64_t parent_b = 0U;
  std::uint64_t interaction = 0U;
};

bool operator==(const PairIdentity& left, const PairIdentity& right) noexcept;
bool canonical_less(const PairIdentity& left,
                    const PairIdentity& right) noexcept;

enum class OfferStatus : std::uint8_t {
  Rejected = 0,
  Inserted = 1,
  Replaced = 2,
  InvalidIdentity = 3,
  InvalidWeight = 4,
  DuplicateSelectedIdentity = 5,
  InvalidCapacity = 6,
  CapacityExceeded = 7,
  InvalidMultiplicity = 8
};

struct OfferResult {
  OfferStatus status = OfferStatus::Rejected;
  std::size_t storage_slot = 0U;
  std::size_t multiplicity = 0U;

  bool accepted() const noexcept {
    return status == OfferStatus::Inserted || status == OfferStatus::Replaced;
  }
};

// One accepted physical pair in the keyed-Bernoulli policy.  A single pair
// can contribute floor(weight*p) records plus one Bernoulli record, exactly
// matching the multiplicity law of GENERAL_LUMI_HEAP::numberToStore().  The
// records occupy [storage_slot, storage_slot + multiplicity).
struct BernoulliSelection {
  PairIdentity identity;
  std::size_t storage_slot = 0U;
  std::size_t multiplicity = 0U;
  std::uint64_t payload_token = 0U;
};

struct Selection {
  PairIdentity identity;
  double priority = 0.0;
  std::size_t storage_slot = 0U;
  std::uint64_t payload_token = 0U;
};

// Deterministic A-Res weighted reservoir.  Each pair receives the exponential
// race priority -log(U)/weight from CounterRngV2 and the K smallest priorities
// survive.  The result is independent of offer order and worker assignment.
// Storage is O(K), not O(number of pair opportunities).
class WeightedReservoir {
 public:
  explicit WeightedReservoir(std::size_t capacity = 0U);

  void reset(std::size_t capacity);
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t size() const noexcept { return heap_.size(); }
  bool empty() const noexcept { return heap_.empty(); }

  OfferResult offer(const PairIdentity& identity, double weight,
                    std::uint64_t payload_token = 0U) noexcept;

  // Deterministic merge boundary for priorities already evaluated in worker
  // lanes.  It performs the same identity and finite-value checks as offer().
  OfferResult offer_precomputed(const PairIdentity& identity,
                                double priority,
                                std::uint64_t payload_token = 0U) noexcept;

  // Selected records are ordered by the complete stable identity.  This is
  // the only supported output order and is independent of heap layout.
  std::vector<Selection> canonical_selection() const;

  // Allocation-free variant for production barriers.  The caller must
  // reserve at least size() records before entry; otherwise this fails closed
  // without modifying the existing selection.
  bool canonical_selection_into(std::vector<Selection>* selected,
                                std::string* reason = 0) const;

  // Allocation-free, no-sort export for intermediate lane/crossing merges.
  // Records retain their reservoir payload and storage slots.  Heap order is
  // intentionally unspecified: consumers must re-offer identity/priority or
  // sort explicitly, never use this as final output order.
  bool selected_records_into(std::vector<Selection>* selected,
                             std::string* reason = 0) const;

  // Merge already bounded lane reservoirs into one bounded global reservoir.
  // Keeping K candidates per lane is sufficient for the exact global top-K.
  // lane order is deliberately irrelevant; returned storage_slot values refer
  // to the merged reservoir, not to a source lane.
  static bool merge_lanes(const std::vector<WeightedReservoir>& lanes,
                          std::size_t capacity,
                          WeightedReservoir* merged,
                          std::string* reason = 0);

  bool healthy() const noexcept { return healthy_; }
  const std::string& error() const noexcept { return error_; }
  std::size_t identity_table_slots() const noexcept
    { return selected_identity_keys_.size(); }
  std::uint64_t storage_rebuilds() const noexcept
    { return storage_rebuilds_; }

 private:
  struct IdentityHash {
    std::size_t operator()(const PairIdentity& identity) const noexcept;
  };

  struct Node {
    PairIdentity identity;
    double priority;
    std::size_t storage_slot;
    std::uint64_t payload_token;
  };

  std::size_t capacity_;
  std::vector<Node> heap_;
  // Fixed-capacity open-address table.  Generation tags make reset O(1), and
  // backward cluster repair keeps replacements free of accumulating
  // tombstones.  All storage is allocated by reset()/construction, never by
  // offer().
  std::vector<PairIdentity> selected_identity_keys_;
  std::vector<std::uint32_t> selected_identity_generations_;
  std::uint32_t selected_identity_generation_;
  std::size_t selected_identity_size_;
  std::uint64_t storage_rebuilds_;
  bool healthy_;
  std::string error_;

  static bool better(const Node& left, const Node& right) noexcept;
  static bool valid_identity(const PairIdentity& identity) noexcept;
  OfferResult offer_priority(const PairIdentity& identity,
                             double priority,
                             std::uint64_t payload_token) noexcept;
  bool selected_identity_exists(const PairIdentity& identity) const noexcept;
  bool insert_selected_identity(const PairIdentity& identity) noexcept;
  bool erase_selected_identity(const PairIdentity& identity) noexcept;
  void prepare_identity_table(std::size_t capacity);
  void begin_identity_epoch() noexcept;
  void poison(const char* message) noexcept;
};

// Sparse deterministic counterpart of the legacy prepare_store(weight*p)
// path.  The CounterRngV2 decision is made before any identity-table lookup,
// allocation, logarithm, or heap operation.  Consequently the common
// store<1 rejection path is one keyed uniform and a branch.  Accepted pairs
// are retained without replacement; capacity is a hard unbiased limit.  An
// overflow poisons the sampler and fails closed instead of silently changing
// the event distribution.
class KeyedBernoulliSampler {
 public:
  explicit KeyedBernoulliSampler(std::size_t capacity = 0U,
                                 std::size_t maximum_groups = 0U);

  void reset(std::size_t capacity, std::size_t maximum_groups = 0U);
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t size() const noexcept { return selections_.size(); }
  std::size_t copies() const noexcept { return copies_; }
  bool empty() const noexcept { return selections_.empty(); }

  OfferResult offer(const PairIdentity& identity, double expected_store,
                    std::uint64_t payload_token = 0U) noexcept;
  OfferResult offer_precomputed(const PairIdentity& identity,
                                std::size_t multiplicity,
                                std::uint64_t payload_token = 0U) noexcept;

  std::vector<BernoulliSelection> canonical_selection() const;
  bool canonical_selection_into(
      std::vector<BernoulliSelection>* selected,
      std::string* reason = 0) const;
  bool selected_records_into(
      std::vector<BernoulliSelection>* selected,
      std::string* reason = 0) const;

  static bool merge_lanes(const std::vector<KeyedBernoulliSampler>& lanes,
                          std::size_t capacity,
                          KeyedBernoulliSampler* merged,
                          std::string* reason = 0);

  bool healthy() const noexcept { return healthy_; }
  const std::string& error() const noexcept { return error_; }
  std::uint64_t opportunities() const noexcept { return opportunities_; }
  std::uint64_t rejected() const noexcept { return rejected_; }
  std::uint64_t accepted_pairs() const noexcept { return accepted_pairs_; }
  std::uint64_t accepted_copies() const noexcept { return accepted_copies_; }
  std::uint64_t accepted_identity_lookups() const noexcept {
    return accepted_identity_lookups_;
  }

 private:
  struct IdentityHash {
    std::size_t operator()(const PairIdentity& identity) const noexcept;
  };

  std::size_t capacity_;
  std::size_t maximum_groups_;
  std::size_t copies_;
  std::vector<BernoulliSelection> selections_;
  std::vector<PairIdentity> selected_identity_keys_;
  std::vector<std::uint32_t> selected_identity_generations_;
  std::uint32_t selected_identity_generation_;
  std::size_t selected_identity_size_;
  bool healthy_;
  std::string error_;
  std::uint64_t opportunities_;
  std::uint64_t rejected_;
  std::uint64_t accepted_pairs_;
  std::uint64_t accepted_copies_;
  std::uint64_t accepted_identity_lookups_;

  static bool valid_identity(const PairIdentity& identity) noexcept;
  bool selected_identity_exists(const PairIdentity& identity) const noexcept;
  bool insert_selected_identity(const PairIdentity& identity) noexcept;
  void prepare_identity_table(std::size_t maximum_groups);
  void begin_identity_epoch() noexcept;
  OfferResult append_precomputed(const PairIdentity& identity,
                                 std::size_t multiplicity,
                                 std::uint64_t payload_token,
                                 bool count_physical_offer) noexcept;
  void poison(const char* message) noexcept;
};

const char* offer_status_name(OfferStatus status) noexcept;

}  // namespace electron_lumi_sampling
}  // namespace feebl

#endif  // FEEBL_ELECTRON_LUMI_SAMPLER_CPP_H
