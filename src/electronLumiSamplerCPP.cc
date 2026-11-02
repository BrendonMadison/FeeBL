#include "electronLumiSamplerCPP.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>

#include "counterRngCPP.h"

namespace feebl {
namespace electron_lumi_sampling {

namespace {

std::uint64_t hash_mix(std::uint64_t value) noexcept {
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31U);
}

PhysicalRandomKeyV2 counter_key(const PairIdentity& identity) noexcept {
  PhysicalRandomKeyV2 key;
  key.run_seed = identity.run_seed;
  key.domain = RandomDomain::LuminosityEeSampling;
  key.crossing = identity.crossing;
  key.diagonal = identity.diagonal;
  key.wave = identity.wave;
  key.substep = identity.substep;
  key.encounter = identity.encounter;
  key.beam = 0U;
  key.parent_a = identity.parent_a;
  key.parent_b = identity.parent_b;
  key.interaction = identity.interaction;
  key.emission = 0U;
  key.rejection_attempt = 0U;
  return key;
}

}  // namespace

std::size_t WeightedReservoir::IdentityHash::operator()(
    const PairIdentity& identity) const noexcept {
  std::uint64_t hash = hash_mix(identity.run_seed);
  hash = hash_mix(hash ^ identity.crossing);
  hash = hash_mix(hash ^ identity.diagonal);
  hash = hash_mix(hash ^ identity.wave);
  hash = hash_mix(hash ^ identity.substep);
  hash = hash_mix(hash ^ identity.encounter);
  hash = hash_mix(hash ^ identity.parent_a);
  hash = hash_mix(hash ^ identity.parent_b);
  hash = hash_mix(hash ^ identity.interaction);
  return static_cast<std::size_t>(hash);
}

bool operator==(const PairIdentity& left,
                const PairIdentity& right) noexcept {
  return left.run_seed == right.run_seed &&
         left.crossing == right.crossing &&
         left.diagonal == right.diagonal &&
         left.wave == right.wave &&
         left.substep == right.substep &&
         left.encounter == right.encounter &&
         left.parent_a == right.parent_a &&
         left.parent_b == right.parent_b &&
         left.interaction == right.interaction;
}

bool canonical_less(const PairIdentity& left,
                    const PairIdentity& right) noexcept {
  return std::tie(left.run_seed, left.crossing, left.diagonal, left.wave,
                  left.substep, left.encounter, left.parent_a, left.parent_b,
                  left.interaction) <
         std::tie(right.run_seed, right.crossing, right.diagonal, right.wave,
                  right.substep, right.encounter, right.parent_a,
                  right.parent_b, right.interaction);
}

WeightedReservoir::WeightedReservoir(std::size_t capacity)
    : capacity_(capacity), selected_identity_generation_(1U),
      selected_identity_size_(0U), storage_rebuilds_(0U), healthy_(true) {
  try {
    if (capacity_ > heap_.capacity()) {
      heap_.reserve(capacity_);
      ++storage_rebuilds_;
    }
    prepare_identity_table(capacity_);
  } catch (const std::bad_alloc&) {
    poison("weighted reservoir allocation failed");
  } catch (const std::length_error&) {
    poison("weighted reservoir capacity exceeds vector limits");
  }
}

void WeightedReservoir::reset(std::size_t capacity) {
  capacity_ = capacity;
  heap_.clear();
  healthy_ = true;
  error_.clear();
  try {
    if (capacity_ > heap_.capacity()) {
      heap_.reserve(capacity_);
      ++storage_rebuilds_;
    }
    prepare_identity_table(capacity_);
  } catch (const std::bad_alloc&) {
    poison("weighted reservoir allocation failed");
  } catch (const std::length_error&) {
    poison("weighted reservoir capacity exceeds vector limits");
  }
}

bool WeightedReservoir::valid_identity(
    const PairIdentity& identity) noexcept {
  return identity.parent_a != kNoParentStableIdV2 &&
         identity.parent_b != kNoParentStableIdV2;
}

bool WeightedReservoir::better(const Node& left,
                               const Node& right) noexcept {
  if (left.priority < right.priority) return true;
  if (right.priority < left.priority) return false;
  return canonical_less(left.identity, right.identity);
}

bool WeightedReservoir::selected_identity_exists(
    const PairIdentity& identity) const noexcept {
  if (selected_identity_keys_.empty()) return false;
  const std::size_t mask = selected_identity_keys_.size() - 1U;
  std::size_t slot = IdentityHash()(identity) & mask;
  for (std::size_t probe = 0; probe < selected_identity_keys_.size(); ++probe) {
    if (selected_identity_generations_[slot] !=
        selected_identity_generation_)
      return false;
    if (selected_identity_keys_[slot] == identity)
      return true;
    slot = (slot + 1U) & mask;
  }
  return false;
}

bool WeightedReservoir::insert_selected_identity(
    const PairIdentity& identity) noexcept {
  if (selected_identity_keys_.empty()) return false;
  const std::size_t mask = selected_identity_keys_.size() - 1U;
  std::size_t slot = IdentityHash()(identity) & mask;
  for (std::size_t probe = 0; probe < selected_identity_keys_.size(); ++probe) {
    if (selected_identity_generations_[slot] !=
        selected_identity_generation_) {
      selected_identity_keys_[slot] = identity;
      selected_identity_generations_[slot] =
          selected_identity_generation_;
      ++selected_identity_size_;
      return true;
    }
    if (selected_identity_keys_[slot] == identity) return false;
    slot = (slot + 1U) & mask;
  }
  return false;
}

bool WeightedReservoir::erase_selected_identity(
    const PairIdentity& identity) noexcept {
  if (selected_identity_keys_.empty()) return false;
  const std::size_t mask = selected_identity_keys_.size() - 1U;
  std::size_t slot = IdentityHash()(identity) & mask;
  for (std::size_t probe = 0; probe < selected_identity_keys_.size(); ++probe) {
    if (selected_identity_generations_[slot] !=
        selected_identity_generation_)
      return false;
    if (selected_identity_keys_[slot] == identity) {
      // Close the linear-probing cluster instead of leaving tombstones.  A
      // long replacement stream can otherwise fill every generation-tagged
      // slot with tombstones and turn absent lookups into full-table scans.
      selected_identity_generations_[slot] = 0U;
      --selected_identity_size_;
      std::size_t displaced_slot = (slot + 1U) & mask;
      while (selected_identity_generations_[displaced_slot] ==
             selected_identity_generation_) {
        const PairIdentity displaced =
            selected_identity_keys_[displaced_slot];
        selected_identity_generations_[displaced_slot] = 0U;
        --selected_identity_size_;
        if (!insert_selected_identity(displaced)) return false;
        displaced_slot = (displaced_slot + 1U) & mask;
      }
      return true;
    }
    slot = (slot + 1U) & mask;
  }
  return false;
}

void WeightedReservoir::begin_identity_epoch() noexcept {
  selected_identity_size_ = 0U;
  if (selected_identity_keys_.empty()) {
    selected_identity_generation_ = 1U;
    return;
  }
  if (selected_identity_generation_ ==
      std::numeric_limits<std::uint32_t>::max()) {
    std::fill(selected_identity_generations_.begin(),
              selected_identity_generations_.end(), 0U);
    selected_identity_generation_ = 1U;
  } else {
    ++selected_identity_generation_;
  }
}

void WeightedReservoir::prepare_identity_table(std::size_t capacity) {
  if (capacity == 0U) {
    begin_identity_epoch();
    return;
  }
  if (capacity > std::numeric_limits<std::size_t>::max() / 2U)
    throw std::length_error("weighted reservoir identity capacity overflow");
  const std::size_t minimum = std::max<std::size_t>(8U, capacity * 2U);
  std::size_t slots = 8U;
  while (slots < minimum) {
    if (slots > std::numeric_limits<std::size_t>::max() / 2U)
      throw std::length_error("weighted reservoir identity table overflow");
    slots *= 2U;
  }
  if (selected_identity_keys_.size() < slots) {
    std::vector<PairIdentity> keys(slots);
    std::vector<std::uint32_t> generations(slots, 0U);
    selected_identity_keys_.swap(keys);
    selected_identity_generations_.swap(generations);
    selected_identity_generation_ = 1U;
    selected_identity_size_ = 0U;
    ++storage_rebuilds_;
  } else {
    begin_identity_epoch();
  }
}

void WeightedReservoir::poison(const char* message) noexcept {
  healthy_ = false;
  try {
    error_ = message;
  } catch (...) {
    error_.clear();
  }
}

OfferResult WeightedReservoir::offer(const PairIdentity& identity,
                                     double weight,
                                     std::uint64_t payload_token) noexcept {
  if (!healthy_) return OfferResult{OfferStatus::InvalidCapacity, 0U};
  if (capacity_ == 0U)
    return OfferResult{OfferStatus::InvalidCapacity, 0U};
  if (!valid_identity(identity))
    return OfferResult{OfferStatus::InvalidIdentity, 0U};
  if (!(weight > 0.0) || !std::isfinite(weight))
    return OfferResult{OfferStatus::InvalidWeight, 0U};
  if (selected_identity_exists(identity)) {
    poison("duplicate selected luminosity-pair identity");
    return OfferResult{OfferStatus::DuplicateSelectedIdentity, 0U};
  }

  try {
    const double uniform = CounterRngV2::uniform_double(counter_key(identity), 0U);
    const double priority = -std::log(uniform) / weight;
    if (!(priority >= 0.0) || !std::isfinite(priority))
      return OfferResult{OfferStatus::InvalidWeight, 0U};
    return offer_priority(identity, priority, payload_token);
  } catch (...) {
    poison("CounterRngV2 rejected a luminosity-pair identity");
    return OfferResult{OfferStatus::InvalidIdentity, 0U};
  }
}

OfferResult WeightedReservoir::offer_precomputed(
    const PairIdentity& identity, double priority,
    std::uint64_t payload_token) noexcept {
  if (!healthy_) return OfferResult{OfferStatus::InvalidCapacity, 0U};
  if (capacity_ == 0U)
    return OfferResult{OfferStatus::InvalidCapacity, 0U};
  if (!valid_identity(identity))
    return OfferResult{OfferStatus::InvalidIdentity, 0U};
  if (!(priority >= 0.0) || !std::isfinite(priority))
    return OfferResult{OfferStatus::InvalidWeight, 0U};
  if (selected_identity_exists(identity)) {
    poison("duplicate selected luminosity-pair identity");
    return OfferResult{OfferStatus::DuplicateSelectedIdentity, 0U};
  }
  return offer_priority(identity, priority, payload_token);
}

OfferResult WeightedReservoir::offer_priority(
    const PairIdentity& identity, double priority,
    std::uint64_t payload_token) noexcept {
  const Node candidate = {identity, priority, 0U, payload_token};
  try {
    if (heap_.size() < capacity_) {
      Node inserted = candidate;
      inserted.storage_slot = heap_.size();
      heap_.push_back(inserted);
      if (!insert_selected_identity(identity)) {
        heap_.pop_back();
        poison("weighted reservoir identity table insertion failed");
        return OfferResult{OfferStatus::InvalidCapacity, 0U};
      }
      std::push_heap(heap_.begin(), heap_.end(), better);
      return OfferResult{OfferStatus::Inserted, inserted.storage_slot};
    }

    // better(candidate, heap.front()) means the new pair belongs to the
    // global top-K.  Reuse the evicted record slot so payload storage remains
    // bounded and does not depend on heap reordering.
    if (!better(candidate, heap_.front()))
      return OfferResult{OfferStatus::Rejected, 0U};
    std::pop_heap(heap_.begin(), heap_.end(), better);
    const std::size_t replacement_slot = heap_.back().storage_slot;
    if (!erase_selected_identity(heap_.back().identity)) {
      poison("weighted reservoir identity table erase failed");
      return OfferResult{OfferStatus::InvalidCapacity, 0U};
    }
    heap_.back() = candidate;
    heap_.back().storage_slot = replacement_slot;
    if (!insert_selected_identity(identity)) {
      poison("weighted reservoir identity table replacement failed");
      return OfferResult{OfferStatus::InvalidCapacity, 0U};
    }
    std::push_heap(heap_.begin(), heap_.end(), better);
    return OfferResult{OfferStatus::Replaced, replacement_slot};
  } catch (const std::bad_alloc&) {
    poison("weighted reservoir allocation failed while offering a pair");
  } catch (...) {
    poison("weighted reservoir failed while offering a pair");
  }
  return OfferResult{OfferStatus::InvalidCapacity, 0U};
}

std::vector<Selection> WeightedReservoir::canonical_selection() const {
  std::vector<Selection> selected;
  selected.reserve(heap_.size());
  std::string reason;
  if (!canonical_selection_into(&selected, &reason)) selected.clear();
  return selected;
}

bool WeightedReservoir::canonical_selection_into(
    std::vector<Selection>* selected, std::string* reason) const {
  if (!selected_records_into(selected, reason)) return false;
  std::sort(selected->begin(), selected->end(),
            [](const Selection& left, const Selection& right) {
              return canonical_less(left.identity, right.identity);
            });
  if (reason != 0) reason->clear();
  return true;
}

bool WeightedReservoir::selected_records_into(
    std::vector<Selection>* selected, std::string* reason) const {
  if (selected == 0) {
    if (reason != 0) *reason = "null selected-record destination";
    return false;
  }
  if (!healthy_) {
    if (reason != 0) *reason = error_;
    return false;
  }
  if (selected->capacity() < heap_.size()) {
    if (reason != 0)
      *reason = "selected-record destination capacity is too small";
    return false;
  }
  selected->clear();
  for (std::vector<Node>::const_iterator node = heap_.begin();
       node != heap_.end(); ++node)
    selected->push_back(Selection{node->identity, node->priority,
                                  node->storage_slot, node->payload_token});
  if (reason != 0) reason->clear();
  return true;
}

bool WeightedReservoir::merge_lanes(
    const std::vector<WeightedReservoir>& lanes, std::size_t capacity,
    WeightedReservoir* merged, std::string* reason) {
  if (merged == 0) {
    if (reason != 0) *reason = "null merged reservoir";
    return false;
  }
  merged->reset(capacity);
  if (!merged->healthy()) {
    if (reason != 0) *reason = merged->error();
    return false;
  }
  for (std::vector<WeightedReservoir>::const_iterator lane = lanes.begin();
       lane != lanes.end(); ++lane) {
    if (!lane->healthy()) {
      if (reason != 0) *reason = "cannot merge a poisoned lane reservoir";
      return false;
    }
    for (std::vector<Node>::const_iterator node = lane->heap_.begin();
         node != lane->heap_.end(); ++node) {
      if (merged->selected_identity_exists(node->identity)) {
        if (reason != 0) *reason = "duplicate identity across lane reservoirs";
        return false;
      }
      const OfferResult result = merged->offer_priority(
          node->identity, node->priority, node->payload_token);
      if (result.status != OfferStatus::Inserted &&
          result.status != OfferStatus::Replaced &&
          result.status != OfferStatus::Rejected) {
        if (reason != 0) *reason = merged->error();
        return false;
      }
    }
  }
  if (reason != 0) reason->clear();
  return true;
}

const char* offer_status_name(OfferStatus status) noexcept {
  switch (status) {
    case OfferStatus::Rejected: return "rejected";
    case OfferStatus::Inserted: return "inserted";
    case OfferStatus::Replaced: return "replaced";
    case OfferStatus::InvalidIdentity: return "invalid-identity";
    case OfferStatus::InvalidWeight: return "invalid-weight";
    case OfferStatus::DuplicateSelectedIdentity:
      return "duplicate-selected-identity";
    case OfferStatus::InvalidCapacity: return "invalid-capacity";
    case OfferStatus::CapacityExceeded: return "capacity-exceeded";
    case OfferStatus::InvalidMultiplicity: return "invalid-multiplicity";
  }
  return "unknown";
}

std::size_t KeyedBernoulliSampler::IdentityHash::operator()(
    const PairIdentity& identity) const noexcept {
  std::uint64_t hash = hash_mix(identity.run_seed);
  hash = hash_mix(hash ^ identity.crossing);
  hash = hash_mix(hash ^ identity.diagonal);
  hash = hash_mix(hash ^ identity.wave);
  hash = hash_mix(hash ^ identity.substep);
  hash = hash_mix(hash ^ identity.encounter);
  hash = hash_mix(hash ^ identity.parent_a);
  hash = hash_mix(hash ^ identity.parent_b);
  hash = hash_mix(hash ^ identity.interaction);
  return static_cast<std::size_t>(hash);
}

KeyedBernoulliSampler::KeyedBernoulliSampler(
    std::size_t capacity, std::size_t maximum_groups)
    : capacity_(0U), maximum_groups_(0U), copies_(0U),
      selected_identity_generation_(1U), selected_identity_size_(0U),
      healthy_(true), opportunities_(0U), rejected_(0U),
      accepted_pairs_(0U), accepted_copies_(0U),
      accepted_identity_lookups_(0U) {
  reset(capacity, maximum_groups);
}

void KeyedBernoulliSampler::reset(std::size_t capacity,
                                  std::size_t maximum_groups) {
  capacity_ = capacity;
  maximum_groups_ = maximum_groups == 0U
                        ? capacity
                        : std::min(capacity, maximum_groups);
  copies_ = 0U;
  selections_.clear();
  healthy_ = true;
  error_.clear();
  opportunities_ = 0U;
  rejected_ = 0U;
  accepted_pairs_ = 0U;
  accepted_copies_ = 0U;
  accepted_identity_lookups_ = 0U;
  try {
    if (maximum_groups_ > selections_.capacity())
      selections_.reserve(maximum_groups_);
    prepare_identity_table(maximum_groups_);
  } catch (const std::bad_alloc&) {
    poison("keyed Bernoulli sampler allocation failed");
  } catch (const std::length_error&) {
    poison("keyed Bernoulli sampler capacity exceeds vector limits");
  }
}

bool KeyedBernoulliSampler::valid_identity(
    const PairIdentity& identity) noexcept {
  return identity.parent_a != kNoParentStableIdV2 &&
         identity.parent_b != kNoParentStableIdV2;
}

void KeyedBernoulliSampler::begin_identity_epoch() noexcept {
  selected_identity_size_ = 0U;
  if (selected_identity_keys_.empty()) {
    selected_identity_generation_ = 1U;
    return;
  }
  if (selected_identity_generation_ ==
      std::numeric_limits<std::uint32_t>::max()) {
    std::fill(selected_identity_generations_.begin(),
              selected_identity_generations_.end(), 0U);
    selected_identity_generation_ = 1U;
  } else {
    ++selected_identity_generation_;
  }
}

void KeyedBernoulliSampler::prepare_identity_table(
    std::size_t maximum_groups) {
  if (maximum_groups == 0U) {
    begin_identity_epoch();
    return;
  }
  if (maximum_groups > std::numeric_limits<std::size_t>::max() / 2U)
    throw std::length_error("keyed Bernoulli identity capacity overflow");
  const std::size_t minimum = std::max<std::size_t>(8U,
                                                    maximum_groups * 2U);
  std::size_t slots = 8U;
  while (slots < minimum) {
    if (slots > std::numeric_limits<std::size_t>::max() / 2U)
      throw std::length_error("keyed Bernoulli identity table overflow");
    slots *= 2U;
  }
  if (selected_identity_keys_.size() < slots) {
    std::vector<PairIdentity> keys(slots);
    std::vector<std::uint32_t> generations(slots, 0U);
    selected_identity_keys_.swap(keys);
    selected_identity_generations_.swap(generations);
    selected_identity_generation_ = 1U;
    selected_identity_size_ = 0U;
  } else {
    begin_identity_epoch();
  }
}

bool KeyedBernoulliSampler::selected_identity_exists(
    const PairIdentity& identity) const noexcept {
  if (selected_identity_keys_.empty()) return false;
  const std::size_t mask = selected_identity_keys_.size() - 1U;
  std::size_t slot = IdentityHash()(identity) & mask;
  for (std::size_t probe = 0; probe < selected_identity_keys_.size(); ++probe) {
    if (selected_identity_generations_[slot] !=
        selected_identity_generation_)
      return false;
    if (selected_identity_keys_[slot] == identity) return true;
    slot = (slot + 1U) & mask;
  }
  return false;
}

bool KeyedBernoulliSampler::insert_selected_identity(
    const PairIdentity& identity) noexcept {
  if (selected_identity_keys_.empty()) return false;
  const std::size_t mask = selected_identity_keys_.size() - 1U;
  std::size_t slot = IdentityHash()(identity) & mask;
  for (std::size_t probe = 0; probe < selected_identity_keys_.size(); ++probe) {
    if (selected_identity_generations_[slot] !=
        selected_identity_generation_) {
      selected_identity_keys_[slot] = identity;
      selected_identity_generations_[slot] = selected_identity_generation_;
      ++selected_identity_size_;
      return true;
    }
    if (selected_identity_keys_[slot] == identity) return false;
    slot = (slot + 1U) & mask;
  }
  return false;
}

void KeyedBernoulliSampler::poison(const char* message) noexcept {
  healthy_ = false;
  try {
    error_ = message;
  } catch (...) {
    error_.clear();
  }
}

OfferResult KeyedBernoulliSampler::offer(
    const PairIdentity& identity, double expected_store,
    std::uint64_t payload_token) noexcept {
  ++opportunities_;
  if (!healthy_)
    return OfferResult{OfferStatus::InvalidCapacity, 0U, 0U};
  if (capacity_ == 0U)
    return OfferResult{OfferStatus::InvalidCapacity, 0U, 0U};
  if (!valid_identity(identity))
    return OfferResult{OfferStatus::InvalidIdentity, 0U, 0U};
  if (!(expected_store >= 0.0) || !std::isfinite(expected_store) ||
      expected_store >=
          static_cast<double>(std::numeric_limits<std::size_t>::max()))
    return OfferResult{OfferStatus::InvalidWeight, 0U, 0U};

  const double integral = std::floor(expected_store);
  std::size_t multiplicity = static_cast<std::size_t>(integral);
  const double fractional = expected_store - integral;
  try {
    // Match the legacy floor(store)+Bernoulli(frac(store)) law.  A stateless
    // keyed draw cannot perturb any other physics stream.
    if (CounterRngV2::uniform_float(counter_key(identity), 0U) <= fractional)
      ++multiplicity;
  } catch (...) {
    poison("CounterRngV2 rejected a Bernoulli luminosity-pair identity");
    return OfferResult{OfferStatus::InvalidIdentity, 0U, 0U};
  }
  if (multiplicity == 0U) {
    ++rejected_;
    return OfferResult{OfferStatus::Rejected, 0U, 0U};
  }
  return append_precomputed(identity, multiplicity, payload_token, true);
}

OfferResult KeyedBernoulliSampler::offer_precomputed(
    const PairIdentity& identity, std::size_t multiplicity,
    std::uint64_t payload_token) noexcept {
  if (!healthy_)
    return OfferResult{OfferStatus::InvalidCapacity, 0U, 0U};
  if (capacity_ == 0U)
    return OfferResult{OfferStatus::InvalidCapacity, 0U, 0U};
  if (!valid_identity(identity))
    return OfferResult{OfferStatus::InvalidIdentity, 0U, 0U};
  if (multiplicity == 0U)
    return OfferResult{OfferStatus::InvalidMultiplicity, 0U, 0U};
  return append_precomputed(identity, multiplicity, payload_token, false);
}

OfferResult KeyedBernoulliSampler::append_precomputed(
    const PairIdentity& identity, std::size_t multiplicity,
    std::uint64_t payload_token, bool count_physical_offer) noexcept {
  ++accepted_identity_lookups_;
  if (selected_identity_exists(identity)) {
    poison("duplicate selected Bernoulli luminosity-pair identity");
    return OfferResult{OfferStatus::DuplicateSelectedIdentity, 0U, 0U};
  }
  if (multiplicity > capacity_ - copies_) {
    poison("keyed Bernoulli luminosity sample exceeded num_lumi capacity");
    return OfferResult{OfferStatus::CapacityExceeded, 0U, multiplicity};
  }
  if (selections_.size() >= maximum_groups_) {
    poison("keyed Bernoulli accepted-pair scratch capacity was exceeded");
    return OfferResult{OfferStatus::InvalidCapacity, 0U, multiplicity};
  }
  const std::size_t storage_slot = copies_;
  try {
    selections_.push_back(
        BernoulliSelection{identity, storage_slot, multiplicity,
                           payload_token});
  } catch (...) {
    poison("keyed Bernoulli accepted-pair append failed");
    return OfferResult{OfferStatus::InvalidCapacity, 0U, multiplicity};
  }
  if (!insert_selected_identity(identity)) {
    selections_.pop_back();
    poison("keyed Bernoulli identity table insertion failed");
    return OfferResult{OfferStatus::InvalidCapacity, 0U, multiplicity};
  }
  copies_ += multiplicity;
  if (count_physical_offer) {
    ++accepted_pairs_;
    accepted_copies_ += static_cast<std::uint64_t>(multiplicity);
  }
  return OfferResult{OfferStatus::Inserted, storage_slot, multiplicity};
}

std::vector<BernoulliSelection>
KeyedBernoulliSampler::canonical_selection() const {
  std::vector<BernoulliSelection> selected;
  selected.reserve(selections_.size());
  std::string reason;
  if (!canonical_selection_into(&selected, &reason)) selected.clear();
  return selected;
}

bool KeyedBernoulliSampler::selected_records_into(
    std::vector<BernoulliSelection>* selected, std::string* reason) const {
  if (selected == 0) {
    if (reason) *reason = "null Bernoulli selected-record destination";
    return false;
  }
  if (!healthy_) {
    if (reason) *reason = error_;
    return false;
  }
  if (selected->capacity() < selections_.size()) {
    if (reason) *reason =
        "Bernoulli selected-record destination capacity is too small";
    return false;
  }
  selected->clear();
  selected->insert(selected->end(), selections_.begin(), selections_.end());
  if (reason) reason->clear();
  return true;
}

bool KeyedBernoulliSampler::canonical_selection_into(
    std::vector<BernoulliSelection>* selected, std::string* reason) const {
  if (!selected_records_into(selected, reason)) return false;
  std::sort(selected->begin(), selected->end(),
            [](const BernoulliSelection& left,
               const BernoulliSelection& right) {
              return canonical_less(left.identity, right.identity);
            });
  if (reason) reason->clear();
  return true;
}

bool KeyedBernoulliSampler::merge_lanes(
    const std::vector<KeyedBernoulliSampler>& lanes,
    std::size_t capacity, KeyedBernoulliSampler* merged,
    std::string* reason) {
  if (!merged) {
    if (reason) *reason = "null merged Bernoulli sampler";
    return false;
  }
  merged->reset(capacity);
  if (!merged->healthy()) {
    if (reason) *reason = merged->error();
    return false;
  }
  std::vector<BernoulliSelection> selected;
  std::size_t maximum_lane_groups = 0U;
  for (std::vector<KeyedBernoulliSampler>::const_iterator lane = lanes.begin();
       lane != lanes.end(); ++lane) {
    if (!lane->healthy()) {
      if (reason) *reason = "cannot merge a poisoned Bernoulli lane";
      return false;
    }
    maximum_lane_groups = std::max(maximum_lane_groups, lane->size());
  }
  selected.reserve(maximum_lane_groups);
  for (std::vector<KeyedBernoulliSampler>::const_iterator lane = lanes.begin();
       lane != lanes.end(); ++lane) {
    std::string export_reason;
    if (!lane->canonical_selection_into(&selected, &export_reason)) {
      if (reason) *reason = "Bernoulli lane export failed: " + export_reason;
      return false;
    }
    for (std::vector<BernoulliSelection>::const_iterator item =
             selected.begin(); item != selected.end(); ++item) {
      const OfferResult offered = merged->offer_precomputed(
          item->identity, item->multiplicity, item->payload_token);
      if (!offered.accepted()) {
        if (reason)
          *reason = std::string("Bernoulli lane merge failed: ") +
              offer_status_name(offered.status) + " " + merged->error();
        return false;
      }
    }
  }
  if (reason) reason->clear();
  return true;
}

}  // namespace electron_lumi_sampling
}  // namespace feebl
