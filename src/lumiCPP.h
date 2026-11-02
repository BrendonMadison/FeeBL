#ifndef LUMI_SEEN
#define LUMI_SEEN

#include "nativeOutputCPP.h"
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
//#include "pairsCPP.h"
#include "rndmCPP.h"
#include "meshCPP.h"
#include "abstractParticle.h"
#include "fileInputOutput.h"
#include "mathematicalEntities.h"
#include "abstractIOclass.h"
#include "electronLumiSamplerCPP.h"

class LUMI_PAIR : public ABSTRACT_IO_CLASS
{
 protected:
  
  float e1_,e2_,x_,y_,z_;
  
 public :
    
 LUMI_PAIR() : e1_(0.0),e2_(0.0),x_(0.0),y_(0.0),z_(0.0) {;}
  
 LUMI_PAIR(float energy1, float energy2) : x_(0.0),y_(0.0),z_(0.0)
    {
      e1_ = energy1; 
      e2_ = energy2; 
    }
  
  ~LUMI_PAIR() {;}
  
  inline void random_position(const MESH& mesh, int cellx, int celly,float min_z, RNDM& rndm_generator)
    {
      mesh.guess_position_in_cell(cellx, celly,min_z, x_, y_, z_, rndm_generator);
    }

  inline void set_position(float x, float y, float z)
    {
      x_ = x;
      y_ = y;
      z_ = z;
    }
   
  virtual std::string output_flow() const 
    {
      std::ostringstream out;
      out << " LUMI_PAIR:: no data for output file " << std::endl;
      return out.str();
    }
  
  virtual std::string persistent_flow() const
  {
    std::ostringstream out;
    write_persistent(out);
    return out.str();
  }

  // Stream records directly when writing large luminosity samples.  The
  // historical string-returning interface above remains available for the
  // generic I/O layer, but constructing one ostringstream per event is a
  // substantial cost at O(10^5--10^6) records.
  virtual void write_persistent(std::ostream& out) const
  {
    out << e1_ << " " << e2_ << " " << x_ << " " << y_ << " "
        << z_*1e-3 << " ";
  }

  // Default native row: the base record has no momenta or spin to add, so it is
  // the persistent one. LUMI_PAIR_EE overrides this to append vz and spin.
  virtual std::string native_flow() const
  {
    std::ostringstream out;
    write_native(out);
    return out.str();
  }

  virtual void write_native(std::ostream& out) const
  {
    write_persistent(out);
  }
  
  inline void get_parameters_for_output(float& e1,float& e2,float& x,float& y,float& z) const
    {
      e1 = e1_;
      e2 = e2_;
      x = x_;
      y = y_;
      z = z_*1e-3;
    }
  
  
  inline void get_impulsion_parameters(float& /*vx1*/,float& /*vy1*/,float& /*vx2*/,float& /*vy2*/, int& /*t*/) const {;}
  
};

class LUMI_PAIR_EE  : public LUMI_PAIR

{

protected :

  float vx1_,vy1_,vx2_,vy2_;
  TRIDVECTOR spin1_, spin2_;
  int t_;
  
  
 public:
  
  
 LUMI_PAIR_EE() : vx1_(0.0),vy1_(0.0),vx2_(0.0),vy2_(0.0),t_(0) {;}
  
  LUMI_PAIR_EE(float energy1, float energy2, float p1Vx, float p1Vy, float p2Vx, float p2Vy,int time_counter) : LUMI_PAIR(energy1, energy2)
    {
     vx1_ = p1Vx;
     vx2_ = p2Vx;
     vy1_ = p1Vy;
     vy2_ = p2Vy;
     t_ = time_counter;
    }

  ~LUMI_PAIR_EE() {;}

  inline  void set_spins(const TRIDVECTOR& s1, const TRIDVECTOR& s2 )
    {
      spin1_ = s1;
      spin2_ = s2;
    }

  void get_spins(float& sx1, float& sy1, float& sz1,float& sx2, float& sy2, float& sz2 ) const
    {
      sx1 = spin1_(0);
      sy1 =  spin1_(1);
      sz1 =  spin1_(2);
      
      sx2 =  spin2_(0);
      sy2 =  spin2_(1);
      sz2 =   spin2_(2);
    }
  

  inline void get_impulsion_parameters(float& vx1,float& vy1,float& vx2,float& vy2, int& t) const
    {
      vx1 = vx1_;
      vy1 = vy1_;
      vx2 =vx2_;
      vy2 = vy2_;
      t = t_;
    }
  
  virtual std::string persistent_flow() const
    {
      std::ostringstream out;
      write_persistent(out);
      return out.str();
    }

  virtual void write_persistent(std::ostream& out) const
    {
      LUMI_PAIR::write_persistent(out);
      out << " " << t_ << " " << vx1_ << " " << vy1_ << " " << vx2_
          << " " << vy2_ << " " << spin1_(0) << " " << spin1_(1)
          << " " << spin1_(2) << " " << spin2_(0) << " " << spin2_(1)
          << " " << spin2_(2) << " ";
    }

  // The same record, plus the two longitudinal components the legacy format
  // omits. Without vz a consumer cannot build a four-vector at all, and the
  // sign is not guessable: beam 2 travels toward -z. Recomputed here rather
  // than stored, because FEEBL carries no vz per particle.
  virtual std::string native_flow() const
    {
      std::ostringstream out;
      write_native(out);
      return out.str();
    }

  virtual void write_native(std::ostream& out) const
    {
      const double q1 = (double)vx1_*(double)vx1_ + (double)vy1_*(double)vy1_;
      const double q2 = (double)vx2_*(double)vx2_ + (double)vy2_*(double)vy2_;
      const double vz1 = (q1 < 1.0) ?  std::sqrt(1.0 - q1) : 0.0;
      const double vz2 = (q2 < 1.0) ? -std::sqrt(1.0 - q2) : 0.0;
      LUMI_PAIR::write_persistent(out);
      out << " " << t_
          << " " << vx1_ << " " << vy1_ << " " << vz1
          << " " << vx2_ << " " << vy2_ << " " << vz2
          << " " << spin1_(0) << " " << spin1_(1) << " " << spin1_(2)
          << " " << spin2_(0) << " " << spin2_(1) << " " << spin2_(2);
    }
};

class GENERAL_LUMI_HEAP : public ABSTRACT_LUMI_HEAP
{
  protected : 
    
    RNDM* rndm_generator_;
  
  int nmax_;
  int nb_pairs_;
  float p_;
  
  public : 
 GENERAL_LUMI_HEAP() : nmax_(0),nb_pairs_(0),p_(0.0) { rndm_generator_ = NULL;}
  
  
  GENERAL_LUMI_HEAP(int nmax, float p, RNDM* rndm_generator)
    {
      rndm_generator_ = rndm_generator;
      nmax_ = nmax;
      p_ = p;
      nb_pairs_ = 0;
    }
  
  inline bool random_ok() const
    {
      return rndm_generator_ != NULL;
    }
  
  
  virtual inline int nb_pairs() const {return nb_pairs_;}
  
  
  inline int numberToStore(float store) const
    {
      int nstore;
      nstore=(int)store;
      store -= nstore;
      if (rndm_generator_->rndm() <= store) nstore++;
      return nstore;
    }
  
  int stack_vector(int nstore, std::vector<int>& selected_indices);
  
  
};

class LUMI_HEAP : public GENERAL_LUMI_HEAP
{
  protected :
    
    std::vector<LUMI_PAIR> data_;
  
 public:
  
  LUMI_HEAP(): GENERAL_LUMI_HEAP() {;};
  
  LUMI_HEAP( int nmax, float p, RNDM* rndm_generator) : GENERAL_LUMI_HEAP( nmax,p, rndm_generator)
    {
      data_.reserve(nmax_);
    }
  
  virtual ~LUMI_HEAP() {;}

  void lumi_store(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float energy2, float weight);
  
  
  
  virtual inline void get_parameters_for_output(unsigned int number, float& e1,float& e2,float& x,float& y,float& z) const
   {
     data_[number].get_parameters_for_output(e1,e2,x,y,z);
   }
  
  virtual inline void get_parameters_for_output(unsigned int /*number*/, float& /*e1*/,float& /*e2*/,float& /*x*/,float& /*y*/,float& /*z*/, float& /*vx1*/,float& /*vy1*/,float& /*vx2*/,float& /*vy2*/, float& /*sx1*/, float& /*sy1*/, float& /*sz1*/, float& /*sx2*/, float& /*sy2*/, float& /*sz2*/, int& /*t*/) const {;}
  
  virtual std::string persistent_flow() const
    {
      //unsigned long int k;
      int k;
      std::ostringstream out;
      std::vector<unsigned long int> order;
      //unsigned long int npairs  =  data_.size();
      int npairs = (int) data_.size();
      // check 
      if ( npairs != nb_pairs_) 
	{
	  std::cerr << " LUMI_HEAP_EE problem with the number of pairs: npairs = " << npairs << " nb_pairs_ = " << nb_pairs_ << std::endl;
	  exit(1);
	}
      
      rndm_generator_->getShuffledIntegerSequence(npairs, order);
      for (k = 0; k < npairs; k++)
	{
	  out << data_[order[k] - 1 ].persistent_flow() << " " << order[k] << std::endl;
	}
      return out.str();
    }
  
  
  
  inline void saveLumi(std::string nameOfOutputFile) const
    {
      std::ofstream f(nameOfOutputFile.c_str());
      if (!f) return;
      std::vector<unsigned long int> order;
      const int npairs = (int)data_.size();
      rndm_generator_->getShuffledIntegerSequence(npairs, order);
      for (int k = 0; k < npairs; ++k)
        {
          data_[order[k] - 1].write_persistent(f);
          f << " " << order[k] << '\n';
        }
      // FILE_IN_OUT_ASCII::save_lumi_heap historically appended std::endl to
      // a payload that already ended in a newline.  Preserve that final blank
      // line so existing output remains byte-compatible.
      f << '\n';
    }

  // FEEBL native output: the same records, but readable by someone who did not
  // write the code. Header states units, frame and every sign convention the
  // numbers depend on; columns are named; vz is present.
  inline void saveLumiNative(std::string nameOfOutputFile,
                             const NativeRunMeta& meta) const
    {
      std::ofstream f(nameOfOutputFile.c_str());
      if (!f) return;
      f << native_header(meta, "lumi_ee",
                         "e1 e2 x y z t vx1 vy1 vz1 vx2 vy2 vz2 "
                         "sx1 sy1 sz1 sx2 sy2 sz2 index");
      std::vector<unsigned long int> order;
      const int npairs = (int) data_.size();
      rndm_generator_->getShuffledIntegerSequence(npairs, order);
      for (int k = 0; k < npairs; k++)
        {
          data_[order[k] - 1].write_native(f);
          f << " " << order[k] << '\n';
        }
      f << native_footer("lumi_ee", (unsigned long)npairs);
    }

  int  prepare_store(float weight);
  
  
};

class LUMI_HEAP_EE : public GENERAL_LUMI_HEAP
{
 protected: 
  
  std::vector<LUMI_PAIR_EE> data_;
  int deterministic_sampling_mode_;
  feebl::electron_lumi_sampling::WeightedReservoir deterministic_sampler_;
  feebl::electron_lumi_sampling::KeyedBernoulliSampler
      deterministic_bernoulli_sampler_;

  std::vector<std::size_t> canonical_storage_order() const;

  feebl::electron_lumi_sampling::OfferStatus commit_keyed_record(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity,
      const feebl::electron_lumi_sampling::OfferResult& offered);

  feebl::electron_lumi_sampling::OfferStatus
  commit_keyed_bernoulli_records(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity,
      const feebl::electron_lumi_sampling::OfferResult& offered);
  
  public :
    
    LUMI_HEAP_EE()
        : GENERAL_LUMI_HEAP(), deterministic_sampling_mode_(0),
          deterministic_sampler_(0U), deterministic_bernoulli_sampler_(0U)
      {;}
  
  LUMI_HEAP_EE( int nmax, float p, RNDM* rndm_generator) : GENERAL_LUMI_HEAP( nmax,p, rndm_generator)
    {
      deterministic_sampling_mode_ = 0;
      deterministic_sampler_.reset(0U);
      deterministic_bernoulli_sampler_.reset(0U);
      data_.reserve(nmax_);
    }

  LUMI_HEAP_EE(int nmax, float p, RNDM* rndm_generator,
               int deterministic_sampling_mode)
      : GENERAL_LUMI_HEAP(nmax, p, rndm_generator),
        deterministic_sampling_mode_(deterministic_sampling_mode),
        deterministic_sampler_(deterministic_sampling_mode == 1
                                   ? static_cast<std::size_t>(nmax) : 0U),
        deterministic_bernoulli_sampler_(deterministic_sampling_mode == 2
                                   ? static_cast<std::size_t>(nmax) : 0U)
    {
      data_.reserve(nmax_);
    }
  
  
  virtual inline void get_parameters_for_output(unsigned int /*number*/, float& /*e1*/,float& /*e2*/,float& /*x*/,float& /*y*/,float& /*z*/) const {;}
  
  virtual inline void get_parameters_for_output(unsigned int number, float& e1,float& e2,float& x,float& y,float& z, float& vx1,float& vy1,float& vx2,float& vy2, float& sx1, float& sy1, float& sz1, float& sx2, float& sy2, float& sz2,int& t) const
    {
      data_[number].get_parameters_for_output(e1,e2,x,y,z);
      data_[number].get_impulsion_parameters(vx1,vy1,vx2,vy2,t);
      data_[number].get_spins(sx1, sy1, sz1, sx2, sy2, sz2);
      /*      std::cout << " get_parameters_for_output, the spins :  " << std::endl; */
      /*      std::cout << sx1 << " " << sy1 << " " << sz1 << std::endl; */
      /*      std::cout << sx2 << " " << sy2 << " " << sz2 << std::endl; */
    }
  
  virtual std::string persistent_flow() const
    {
      //unsigned long int k;
      int k;
      std::ostringstream out;
      std::vector<unsigned long int> order;
      //unsigned long int npairs  =  data_.size();
      int npairs = (int) data_.size();
      
      // check 
      if ( npairs != nb_pairs_) 
	{
	  std::cerr << " LUMI_HEAP_EE problem with the number of pairs: npairs = " << npairs << " nb_pairs_ = " << nb_pairs_ << std::endl;
	  exit(1);
	}
      if (deterministic_sampling_mode_ != 0)
        {
          const std::vector<std::size_t> selected = canonical_storage_order();
          for (k = 0; k < npairs; ++k)
            out << data_[selected[static_cast<std::size_t>(k)]]
                       .persistent_flow()
                << " " << (k + 1) << std::endl;
        }
      else
        {
          rndm_generator_->getShuffledIntegerSequence(npairs, order);
          for (k = 0; k < npairs; k++)
	    out << data_[ order[k] - 1 ].persistent_flow() << " " << order[k] << std::endl;
	}
      return out.str();
    }
  
  
  inline void saveLumi(std::string nameOfOutputFile) const
    {
      std::ofstream f(nameOfOutputFile.c_str());
      if (!f) return;
      std::vector<unsigned long int> order;
      const int npairs = (int)data_.size();
      if (deterministic_sampling_mode_ != 0)
        {
          const std::vector<std::size_t> selected = canonical_storage_order();
          for (int n = 0; n < npairs; ++n)
            {
              data_[selected[static_cast<std::size_t>(n)]]
                  .write_persistent(f);
              f << " " << (n + 1) << '\n';
            }
        }
      else
        {
          rndm_generator_->getShuffledIntegerSequence(npairs, order);
          for (int n = 0; n < npairs; ++n)
            {
              data_[order[n] - 1].write_persistent(f);
              f << " " << order[n] << '\n';
            }
        }
      f << '\n';
    }

  // FEEBL native output for the ee heap: same records, but readable by someone
  // who did not write the code. The legacy file has no header, no column names
  // and no vz, so a consumer cannot even build a four-vector from it.
  inline void saveLumiNative(std::string nameOfOutputFile,
                             const NativeRunMeta& meta) const
    {
      std::ofstream f(nameOfOutputFile.c_str());
      if (!f) return;
      f << native_header(meta, "lumi_ee",
                         "e1 e2 x y z t vx1 vy1 vz1 vx2 vy2 vz2 "
                         "sx1 sy1 sz1 sx2 sy2 sz2 index");
      std::vector<unsigned long int> order;
      const int npairs = (int) data_.size();
      if (deterministic_sampling_mode_ != 0)
        {
          const std::vector<std::size_t> selected = canonical_storage_order();
          for (int n = 0; n < npairs; ++n)
            {
              data_[selected[static_cast<std::size_t>(n)]]
                  .write_native(f);
              f << " " << (n + 1) << '\n';
            }
        }
      else
        {
          rndm_generator_->getShuffledIntegerSequence(npairs, order);
          for (int n = 0; n < npairs; n++)
            {
              data_[order[n] - 1].write_native(f);
              f << " " << order[n] << '\n';
            }
        }
      f << native_footer("lumi_ee", (unsigned long)npairs);
    }


  void lumi_store_ee(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float p1Vx, float p1Vy, float energy2, float p2Vx, float p2Vy,float weight,int time_counter);
  
  void lumi_store_ee(const MESH& mesh, int cellx, int celly,float min_z, float energy1, float p1Vx, float p1Vy, float energy2, float p2Vx, float p2Vy,float weight, const TRIDVECTOR& s1, const TRIDVECTOR& s2, int time_counter); 

  feebl::electron_lumi_sampling::OfferStatus lumi_store_ee_keyed(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy, float weight,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity);

  feebl::electron_lumi_sampling::OfferStatus
  lumi_store_ee_keyed_priority(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity,
      double priority);

  feebl::electron_lumi_sampling::OfferStatus
  lumi_store_ee_keyed_bernoulli(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy, float weight,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity);

  feebl::electron_lumi_sampling::OfferStatus
  lumi_store_ee_keyed_bernoulli_multiplicity(
      const MESH& mesh, int cellx, int celly, float min_z,
      float energy1, float p1Vx, float p1Vy,
      float energy2, float p2Vx, float p2Vy,
      int time_counter,
      const feebl::electron_lumi_sampling::PairIdentity& identity,
      std::size_t multiplicity);

  inline bool deterministic_sampling() const
    { return deterministic_sampling_mode_ != 0; }
  inline int deterministic_sampling_mode() const
    { return deterministic_sampling_mode_; }
  const std::string& deterministic_sampling_error() const;

  // Canonical worker-lane publication.  Both the weighted selection and its
  // variable-size event payload are merged transactionally; source storage
  // slots are never exposed after this call.
  bool merge_deterministic_from(const LUMI_HEAP_EE& other,
                                std::string* reason);
  // Crossing-level worker publication.  A single transactional candidate is
  // used for every lane so a large num_lumi reservoir is copied only once,
  // rather than once per lane (or, worse, once per slice encounter).
  bool merge_deterministic_from_many(
      const std::vector<const LUMI_HEAP_EE*>& others,
      std::string* reason);
  bool reset_deterministic_records(std::string* reason);

  int  prepare_store(float weight);
  
  
};

#endif
