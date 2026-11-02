#include "lumiCPP.h"
#include "counterRngCPP.h"

#include <limits>

int GENERAL_LUMI_HEAP::stack_vector(int nstore, std::vector<int>& selected_indices)
  {
     int k;
     float store,scal;
     scal=1.0/(float)((int)((nstore+nb_pairs_)/(nmax_))+1);
     p_ *= scal;      
     for (k = 0; k< nb_pairs_; k++)
	{
	  if(rndm_generator_->rndm()<scal)
	    {
              selected_indices.push_back(k);
	    }
	}
      store = nstore*scal;
      nstore = numberToStore(store);
      return nstore;
}

 
void LUMI_HEAP::lumi_store(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float energy2, float weight) 
{
  int k;
  int nstore = prepare_store(weight);
  for (k=0; k < nstore; k++)
    {
      data_.push_back(LUMI_PAIR(energy1, energy2));
      data_.back().random_position(mesh,cellx, celly, min_z,*rndm_generator_);
      nb_pairs_++;
    }
}

int  LUMI_HEAP::prepare_store(float weight)
{
  unsigned int j;
  int nstore;
  float store;
  //float scal;
  store=weight*p_;
  nstore = numberToStore(store);
  if(nstore + nb_pairs_ > nmax_ )
    {
      std::vector<int> selected_indices;
      nstore = stack_vector(nstore, selected_indices);
      nb_pairs_ = 0;
      for (j=0; j<selected_indices.size(); j++)
          {
	      data_[nb_pairs_++] = data_[selected_indices[j]];
          }
      data_.resize(nb_pairs_);
      // at the end of this method: certrin elements of the rndm_generator are
      // stored in the elements of data_ and they are transferred to the first
      // elements of vector. nstore is the number of stored elements.
    }
  return nstore;
}

int  LUMI_HEAP_EE::prepare_store(float weight)
{
  unsigned int j;
  int nstore;
  float store;
  //float scal;
  store=weight*p_;
  nstore = numberToStore(store);
  if(nstore + nb_pairs_ > nmax_ )
    {
      std::vector<int> selected_indices;
      // recuperate the pair indices to store
      nstore = stack_vector(nstore, selected_indices);
      nb_pairs_ = 0;
      for (j=0; j<selected_indices.size(); j++)
	{
	  data_[nb_pairs_++] = data_[selected_indices[j]];
	}
      data_.resize(nb_pairs_);
      // at the end of this method: certrin elements of the rndm_generator are
      // stored in the elements of data_ and they are transferred to the first
      // elements of vector. nstore is the number of stored elements.
    }
  return nstore;
}


void LUMI_HEAP_EE::lumi_store_ee(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float p1Vx, float p1Vy, float energy2, float p2Vx, float p2Vy,float weight,int time_counter) 
{
  int k;
  int nstore = prepare_store(weight);
  for (k=0; k < nstore; k++)
    {
      data_.push_back(LUMI_PAIR_EE(energy1, energy2, p1Vx,p1Vy,p2Vx, p2Vy,time_counter));
      data_.back().random_position(mesh,cellx, celly, min_z,*rndm_generator_);
      nb_pairs_++;
    }
}

void LUMI_HEAP_EE::lumi_store_ee(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float p1Vx, float p1Vy, float energy2, float p2Vx, float p2Vy,float weight, const TRIDVECTOR& s1, const TRIDVECTOR& s2, int time_counter) 
{
  int k;
  int nstore = prepare_store(weight);
  for (k=0; k < nstore; k++)
    {
      data_.push_back(LUMI_PAIR_EE(energy1, energy2, p1Vx,p1Vy,p2Vx, p2Vy,time_counter));
      data_.back().random_position(mesh,cellx, celly, min_z,*rndm_generator_);
      data_.back().set_spins( s1,s2);
      nb_pairs_++;
    }
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::lumi_store_ee_keyed(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy, float weight,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity)
{
  using feebl::electron_lumi_sampling::OfferResult;
  using feebl::electron_lumi_sampling::OfferStatus;
  if (deterministic_sampling_mode_ != 1)
    return OfferStatus::InvalidCapacity;

  const OfferResult offered = deterministic_sampler_.offer(identity, weight);
  return commit_keyed_record(
      mesh, cellx, celly, min_z, energy1, p1Vx, p1Vy,
      energy2, p2Vx, p2Vy, time_counter, identity, offered);
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::lumi_store_ee_keyed_priority(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity,
    double priority)
{
  using feebl::electron_lumi_sampling::OfferResult;
  using feebl::electron_lumi_sampling::OfferStatus;
  if (deterministic_sampling_mode_ != 1)
    return OfferStatus::InvalidCapacity;
  const OfferResult offered =
      deterministic_sampler_.offer_precomputed(identity, priority);
  return commit_keyed_record(
      mesh, cellx, celly, min_z, energy1, p1Vx, p1Vy,
      energy2, p2Vx, p2Vy, time_counter, identity, offered);
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::commit_keyed_record(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity,
    const feebl::electron_lumi_sampling::OfferResult& offered)
{
  using feebl::electron_lumi_sampling::OfferStatus;
  if (!offered.accepted()) return offered.status;

  LUMI_PAIR_EE record(energy1, energy2, p1Vx, p1Vy, p2Vx, p2Vy,
                      time_counter);
  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = identity.run_seed;
  key.domain = feebl::RandomDomain::LuminosityEeSampling;
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
  // Weighted-reservoir selection consumes raw draws 0 and 1 through
  // uniform_double(sample=0). Start position sampling at draw 2 so event
  // coordinates do not reuse selection entropy.
  const float z =
      (min_z + feebl::CounterRngV2::uniform_float(key, 2U)) *
      mesh.get_delta_z();
  const float x =
      (cellx + feebl::CounterRngV2::uniform_float(key, 3U) -
       mesh.get_offset_x()) * mesh.get_delta_x();
  const float y =
      (celly + feebl::CounterRngV2::uniform_float(key, 4U) -
       mesh.get_offset_y()) * mesh.get_delta_y();
  record.set_position(x, y, z);

  if (offered.storage_slot == data_.size())
    data_.push_back(record);
  else if (offered.storage_slot < data_.size())
    data_[offered.storage_slot] = record;
  else
    return OfferStatus::InvalidCapacity;
  nb_pairs_ = static_cast<int>(data_.size());
  return offered.status;
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::lumi_store_ee_keyed_bernoulli(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy, float weight,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity)
{
  using feebl::electron_lumi_sampling::OfferResult;
  using feebl::electron_lumi_sampling::OfferStatus;
  if (deterministic_sampling_mode_ != 2)
    return OfferStatus::InvalidCapacity;
  // Preserve the legacy float multiplication before promoting for the keyed
  // decision, so mode 2 mirrors prepare_store(weight*p_) statistically.
  const double expected_store = static_cast<double>(weight * p_);
  const OfferResult offered =
      deterministic_bernoulli_sampler_.offer(identity, expected_store);
  return commit_keyed_bernoulli_records(
      mesh, cellx, celly, min_z, energy1, p1Vx, p1Vy,
      energy2, p2Vx, p2Vy, time_counter, identity, offered);
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::lumi_store_ee_keyed_bernoulli_multiplicity(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity,
    std::size_t multiplicity)
{
  using feebl::electron_lumi_sampling::OfferResult;
  using feebl::electron_lumi_sampling::OfferStatus;
  if (deterministic_sampling_mode_ != 2)
    return OfferStatus::InvalidCapacity;
  const OfferResult offered = deterministic_bernoulli_sampler_
      .offer_precomputed(identity, multiplicity);
  return commit_keyed_bernoulli_records(
      mesh, cellx, celly, min_z, energy1, p1Vx, p1Vy,
      energy2, p2Vx, p2Vy, time_counter, identity, offered);
}

feebl::electron_lumi_sampling::OfferStatus
LUMI_HEAP_EE::commit_keyed_bernoulli_records(
    const MESH& mesh, int cellx, int celly, float min_z,
    float energy1, float p1Vx, float p1Vy,
    float energy2, float p2Vx, float p2Vy,
    int time_counter,
    const feebl::electron_lumi_sampling::PairIdentity& identity,
    const feebl::electron_lumi_sampling::OfferResult& offered)
{
  using feebl::electron_lumi_sampling::OfferStatus;
  if (!offered.accepted()) return offered.status;
  if (offered.multiplicity == 0U ||
      offered.storage_slot != data_.size() ||
      offered.multiplicity > static_cast<std::size_t>(nmax_) - data_.size())
    return OfferStatus::InvalidCapacity;

  feebl::PhysicalRandomKeyV2 key;
  key.run_seed = identity.run_seed;
  key.domain = feebl::RandomDomain::LuminosityEeSampling;
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

  for (std::size_t replica = 0U; replica < offered.multiplicity; ++replica)
    {
      LUMI_PAIR_EE record(energy1, energy2, p1Vx, p1Vy, p2Vx, p2Vy,
                          time_counter);
      const std::uint64_t first_draw =
          1U + 3U * static_cast<std::uint64_t>(replica);
      const float z =
          (min_z + feebl::CounterRngV2::uniform_float(key, first_draw)) *
          mesh.get_delta_z();
      const float x =
          (cellx + feebl::CounterRngV2::uniform_float(key, first_draw + 1U) -
           mesh.get_offset_x()) * mesh.get_delta_x();
      const float y =
          (celly + feebl::CounterRngV2::uniform_float(key, first_draw + 2U) -
           mesh.get_offset_y()) * mesh.get_delta_y();
      record.set_position(x, y, z);
      data_.push_back(record);
    }
  nb_pairs_ = static_cast<int>(data_.size());
  return offered.status;
}

const std::string& LUMI_HEAP_EE::deterministic_sampling_error() const
{
  return deterministic_sampling_mode_ == 2
      ? deterministic_bernoulli_sampler_.error()
      : deterministic_sampler_.error();
}

std::vector<std::size_t> LUMI_HEAP_EE::canonical_storage_order() const
{
  std::vector<std::size_t> order;
  order.reserve(data_.size());
  if (deterministic_sampling_mode_ == 1)
    {
      const std::vector<feebl::electron_lumi_sampling::Selection> selected =
          deterministic_sampler_.canonical_selection();
      for (std::vector<feebl::electron_lumi_sampling::Selection>::const_iterator
               item = selected.begin(); item != selected.end(); ++item)
        order.push_back(item->storage_slot);
    }
  else if (deterministic_sampling_mode_ == 2)
    {
      const std::vector<feebl::electron_lumi_sampling::BernoulliSelection>
          selected = deterministic_bernoulli_sampler_.canonical_selection();
      for (std::vector<
               feebl::electron_lumi_sampling::BernoulliSelection>::const_iterator
               item = selected.begin(); item != selected.end(); ++item)
        for (std::size_t replica = 0U; replica < item->multiplicity; ++replica)
          {
            if (item->storage_slot > data_.size() ||
                replica >= data_.size() - item->storage_slot)
              {
                std::cerr
                    << " LUMI_HEAP_EE Bernoulli sampler/data slot mismatch"
                    << std::endl;
                std::exit(1);
              }
            order.push_back(item->storage_slot + replica);
          }
    }
  if (order.size() != data_.size())
    {
      std::cerr << " LUMI_HEAP_EE deterministic sampler/data shape mismatch"
                << std::endl;
      std::exit(1);
    }
  return order;
}

bool LUMI_HEAP_EE::merge_deterministic_from(
    const LUMI_HEAP_EE& other, std::string* reason)
{
  std::vector<const LUMI_HEAP_EE*> lanes(1U, &other);
  return merge_deterministic_from_many(lanes, reason);
}

bool LUMI_HEAP_EE::merge_deterministic_from_many(
    const std::vector<const LUMI_HEAP_EE*>& others,
    std::string* reason)
{
  using feebl::electron_lumi_sampling::OfferResult;
  using feebl::electron_lumi_sampling::OfferStatus;
  using feebl::electron_lumi_sampling::Selection;
  if (deterministic_sampling_mode_ == 0)
    {
      if (reason) *reason = "target e-e heap must use deterministic sampling";
      return false;
    }
  const bool target_capacity_valid =
      deterministic_sampling_mode_ == 1
          ? deterministic_sampler_.capacity() ==
                static_cast<std::size_t>(nmax_)
          : deterministic_bernoulli_sampler_.capacity() ==
                static_cast<std::size_t>(nmax_);
  if (nmax_ <= 0 || !target_capacity_valid)
    {
      if (reason) *reason = "target e-e heap has an invalid capacity";
      return false;
    }
  if ((deterministic_sampling_mode_ == 1 &&
       !deterministic_sampler_.healthy()) ||
      (deterministic_sampling_mode_ == 2 &&
       !deterministic_bernoulli_sampler_.healthy()))
    {
      if (reason) *reason = "cannot merge into a poisoned e-e reservoir";
      return false;
    }

  // Construct a pre-reserved authoritative candidate, then treat the current
  // target as the first source.  Copying an empty std::vector would discard
  // its reserved K-record capacity and allocate repeatedly after the beam had
  // already been mutated.
  LUMI_HEAP_EE candidate(
      nmax_, p_, rndm_generator_, deterministic_sampling_mode_);
  std::vector<const LUMI_HEAP_EE*> sources;
  sources.reserve(others.size() + 1U);
  sources.push_back(this);
  sources.insert(sources.end(), others.begin(), others.end());
  std::vector<Selection> selected;
  std::vector<feebl::electron_lumi_sampling::BernoulliSelection>
      bernoulli_selected;
  selected.reserve(static_cast<std::size_t>(nmax_));
  bernoulli_selected.reserve(static_cast<std::size_t>(nmax_));
  for (std::size_t lane = 0; lane < sources.size(); ++lane)
    {
      const LUMI_HEAP_EE* other = sources[lane];
      if (!other)
        {
          if (reason) *reason = "null worker e-e heap";
          return false;
        }
      const bool weighted_capacity_matches =
          deterministic_sampling_mode_ == 1 &&
          deterministic_sampler_.capacity() ==
              other->deterministic_sampler_.capacity();
      const bool bernoulli_capacity_matches =
          deterministic_sampling_mode_ == 2 &&
          deterministic_bernoulli_sampler_.capacity() ==
              other->deterministic_bernoulli_sampler_.capacity();
      if (other->deterministic_sampling_mode_ !=
              deterministic_sampling_mode_ ||
          nmax_ != other->nmax_ ||
          (!weighted_capacity_matches && !bernoulli_capacity_matches))
        {
          if (reason) *reason = "worker e-e reservoir capacities differ";
          return false;
        }
      const bool source_healthy =
          deterministic_sampling_mode_ == 1
              ? other->deterministic_sampler_.healthy()
              : other->deterministic_bernoulli_sampler_.healthy();
      if (!source_healthy)
        {
          if (reason) *reason = "cannot merge a poisoned worker e-e reservoir";
          return false;
        }

      if (deterministic_sampling_mode_ == 1)
        {
          std::string selection_reason;
          if (!other->deterministic_sampler_.selected_records_into(
                  &selected, &selection_reason))
            {
              if (reason)
                *reason = "source e-e reservoir export failed: " +
                    selection_reason;
              return false;
            }
          if (selected.size() != other->data_.size())
            {
              if (reason)
                *reason = "source e-e reservoir/payload shape mismatch";
              return false;
            }
          for (std::vector<Selection>::const_iterator item = selected.begin();
               item != selected.end(); ++item)
            {
              if (item->storage_slot >= other->data_.size())
                {
                  if (reason)
                    *reason = "source e-e payload slot is out of range";
                  return false;
                }
              const OfferResult offered = candidate.deterministic_sampler_
                  .offer_precomputed(item->identity, item->priority);
              if (offered.status == OfferStatus::Rejected) continue;
              if (!offered.accepted())
                {
                  if (reason)
                    *reason =
                        std::string("global e-e reservoir merge failed: ") +
                        feebl::electron_lumi_sampling::offer_status_name(
                            offered.status);
                  return false;
                }
              if (offered.storage_slot == candidate.data_.size())
                candidate.data_.push_back(other->data_[item->storage_slot]);
              else if (offered.storage_slot < candidate.data_.size())
                candidate.data_[offered.storage_slot] =
                    other->data_[item->storage_slot];
              else
                {
                  if (reason)
                    *reason = "global e-e payload slot is out of range";
                  return false;
                }
            }
        }
      else
        {
          std::string selection_reason;
          if (!other->deterministic_bernoulli_sampler_.selected_records_into(
                  &bernoulli_selected, &selection_reason))
            {
              if (reason)
                *reason = "source Bernoulli e-e sample export failed: " +
                    selection_reason;
              return false;
            }
          std::size_t source_copies = 0U;
          for (std::vector<
                   feebl::electron_lumi_sampling::BernoulliSelection>::
                   const_iterator item = bernoulli_selected.begin();
               item != bernoulli_selected.end(); ++item)
            {
              if (item->multiplicity == 0U ||
                  item->storage_slot > other->data_.size() ||
                  item->multiplicity >
                      other->data_.size() - item->storage_slot)
                {
                  if (reason)
                    *reason = "source Bernoulli e-e payload range is invalid";
                  return false;
                }
              if (item->multiplicity >
                  std::numeric_limits<std::size_t>::max() - source_copies)
                {
                  if (reason) *reason = "source Bernoulli copy count overflow";
                  return false;
                }
              source_copies += item->multiplicity;
              const OfferResult offered =
                  candidate.deterministic_bernoulli_sampler_
                      .offer_precomputed(item->identity, item->multiplicity);
              if (!offered.accepted())
                {
                  if (reason)
                    *reason =
                        std::string("global Bernoulli e-e merge failed: ") +
                        feebl::electron_lumi_sampling::offer_status_name(
                            offered.status) + " " +
                        candidate.deterministic_bernoulli_sampler_.error();
                  return false;
                }
              if (offered.storage_slot != candidate.data_.size())
                {
                  if (reason)
                    *reason = "global Bernoulli e-e storage is not append-only";
                  return false;
                }
              candidate.data_.insert(
                  candidate.data_.end(),
                  other->data_.begin() + item->storage_slot,
                  other->data_.begin() + item->storage_slot +
                      item->multiplicity);
            }
          if (source_copies != other->data_.size())
            {
              if (reason)
                *reason = "source Bernoulli e-e sampler/payload shape mismatch";
              return false;
            }
        }
    }
  candidate.nb_pairs_ = static_cast<int>(candidate.data_.size());
  *this = candidate;
  if (reason) reason->clear();
  return true;
}

bool LUMI_HEAP_EE::reset_deterministic_records(std::string* reason)
{
  if (deterministic_sampling_mode_ == 0 || nmax_ <= 0)
    {
      if (reason) *reason = "cannot reset a non-deterministic e-e heap lane";
      return false;
    }
  if (deterministic_sampling_mode_ == 1)
    deterministic_sampler_.reset(static_cast<std::size_t>(nmax_));
  else
    deterministic_bernoulli_sampler_.reset(static_cast<std::size_t>(nmax_));
  const bool healthy = deterministic_sampling_mode_ == 1
      ? deterministic_sampler_.healthy()
      : deterministic_bernoulli_sampler_.healthy();
  if (!healthy)
    {
      if (reason) *reason = deterministic_sampling_error();
      return false;
    }
  data_.clear();
  nb_pairs_ = 0;
  if (reason) reason->clear();
  return true;
}
