#ifndef RNDM_SEEN
#define RNDM_SEEN

#include <cstdio>   //   exit function 
#include <cstdlib>  //   exit function
#include <stdexcept>

#include "define.h"

#define RNDM_EPS 6e-8

#include <vector>
#include <cstdint>
#include <string>

#include "counterRngCPP.h"

/** 
    Random number generator class
    Implements several random number generators.
    Currently number 7 is used in all cases
 */

class RNDM
{

  int m1_;
  int ia1_;
  int ic1_;
  float rm1_;
  int m2_;
  int ia2_;
  int ic2_;
  float rm2_;
  int m3_;
  int ia3_;
  int ic3_;
  int iff_;
  int ix1_;
  int ix2_;
  int ix3_;
  float rm3_;

  int iset_;
  float v1_,v2_;

  /** counter to count how often rndm7 is called */ 
  std::uint64_t counterRndm7_;
  unsigned long rndm7_a_coeff_, rndm7_c_coeff_;
  float rndm7_modulo_dividing_factor_;
  bool counter_mode_;
  feebl::PhysicalRandomKeyV2 counter_key_;
  std::uint64_t counter_draw_;
  bool counter_exhausted_;
  std::string loaded_checkpoint_checksum_;

  struct
  {
    int i;
  } rndm0_store;
  
  struct
  {
    long i,p,is[32];
  } rndm1_store;
  
  struct
  {
    int i1,i2,p,is[32];
  } rndm2_store;
  
  struct
  {
    float u[97],c,cd,cm;
    int i,j;
  } rndm5_store;
  
  struct
  {
    int i;
  } rndm6_store;
  
  struct
  {
    unsigned long i;
  } rndm7_store;
  
  struct
  {
    float r[97];
    int ix1,ix2,ix3;
  } rndm8_store;
  
  
  /** random number generators
      only rndm7 is used at the moment
   */  
  void rndmst0(int i);
  float rndm0();
  void rndmst1(int i);
  float rndm1();
  void rndmst2(int i);
  float rndm2();
  void rndmst5(int na1,int na2,int na3, int nb1);
  float rndm5();
  void rndmst6(int i);
  float rndm6();

  void rndmst7(unsigned long i);
  bool rndm_test7(unsigned long a_coeff, unsigned long c_coeff, const unsigned long* test_values);
  float rndm7();

  void rndmst8(int idummy);
  float rndm8();
  float expdev();
  float exp_dev();
  


  /* choice of random number generators 5 and 7 are good */
 public:
  /** constructor with seed for random number generator 7 */
  RNDM(unsigned long seed = 1);

  // Start a new encounter-local legacy-physics stream without repeating the
  // generator self-test or reconstructing the object.  The seed supplied by
  // the wave adapter is itself derived from CounterRngV2 physical identity,
  // so worker assignment and completion order cannot change the stream.
  inline void reseed_encounter(unsigned long seed)
    {
      counter_mode_ = false;
      loaded_checkpoint_checksum_.clear();
      counter_exhausted_ = false;
      rndm7_store.i = seed;
      counterRndm7_ = 0;
      iset_ = 0;
      v1_ = 0.0f;
      v2_ = 0.0f;
    }

  // Bind the familiar RNDM interface to one CounterRngV2 physical stream.
  // Existing physics kernels therefore need no shared cursor and no worker ID
  // in their random identity.  A caller starts a fresh domain at each audited
  // encounter stage (beamstrahlung, photon sampling, and so on).
  inline void begin_counter_stream(const feebl::PhysicalRandomKeyV2& key)
    {
      if (!feebl::physical_random_key_is_valid_v2(key))
        throw std::invalid_argument("RNDM received an invalid CounterRngV2 key");
      counter_mode_ = true;
      loaded_checkpoint_checksum_.clear();
      counter_key_ = key;
      counter_draw_ = 0U;
      counter_exhausted_ = false;
      counterRndm7_ = 0;
      iset_ = 0;
      v1_ = 0.0f;
      v2_ = 0.0f;
    }

  inline bool counter_stream_active() const { return counter_mode_; }
  inline const std::string& loaded_checkpoint_checksum() const
    { return loaded_checkpoint_checksum_; }
  
  /** gaussian distributed random number*/
  float gasdev();
  // Persist the complete public random stream (uniform cursor, Gaussian
  // cache and CounterRngV2 identity). Saving atomically replaces path only
  // after a checked write. Loading validates a candidate before changing any
  // live state. Missing, corrupt and legacy raw checkpoints throw.
  // These files restart the random stream, not the particle simulation.
  int rndm_save(const char* path = "rndm.save") const;
  int rndm_load(const char* path = "rndm.save");
  
  /** method draws random number theta between [0,2*Pi] and return value is sin(theta), and *c is cos(theta)  */
  float rndm_sincos(float *c);

  /** get a vector containing the sequence of integers from 1 to maxInt 
      randomly shuffled */
  void getShuffledIntegerSequence(int maxInt, std::vector<unsigned long int>& vect);
  
  /** public random number generators */
  inline float rndm() {return rndm7();}
  inline float rndm_hadron(){return rndm7();}
  inline float rndm_synrad() {return rndm7();}
  inline float rndm_equiv() {return rndm7();}
  inline float rndm_jet() {return rndm7();}
  inline float rndm_pairs() {return rndm7();}
  
};
#endif
