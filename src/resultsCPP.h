#ifndef RESULTS_SEEN
#define RESULTS_SEEN
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

#include "qedbg_sampling_guard.h"

#include "pairMomentsCPP.h"
#include "colliderDiagnosticsCPP.h"
#include "colliderPeakDiagnosticsCPP.h"
#include "define.h"
#include "physconst.h"
#include "mathconst.h"
//#include "hitCPP.h"
//#include "meshCPP.h"
//#include "pairsCPP.h"
#include "beamCPP.h"
#include "switchesCPP.h"
#include "abstractParticle.h"

/* Definition of variables to store the results */

class  RESULTS : public ABSTRACT_IO_CLASS
{
  const SWITCHES* switches_;
  // Mode 3 currently transports primaries and the charge-density fine
  // luminosity only.  Pair enumeration aggregates must be serialized as
  // unavailable, never as physically plausible zeroes.
  bool primary_only_mode3_;
  double lumi_0;
  double lumi_fine,lumi_ee,lumi_pp,lumi_eg,lumi_ge,lumi_gg,lumi_gg_high,lumi_ee_high;
  // Luminosity-weighted helicity, the quantity a physics analysis needs.
  // P_lumi = integral L(t) P(t) dt / integral L(t) dt, accumulated pair by
  // pair as the collision proceeds. Distinct from the bunch average, and
  // much smaller: the literature quotes ~0.21 percentage points for an 80%
  // beam against ~1.4% for the whole outgoing bunch.
  double lumi_whel1, lumi_whel2, lumi_wsum;
  // Pair correlations must be accumulated before averaging: <h1 h2> is
  // generally not <h1><h2>. Keep both local helicity and projections onto
  // the two design axes (+z,-z) for source-matched spin observables.
  pairmoments::PairHelicity lumi_helicity_pair_, lumi_axis_pair_;
  pairmoments::PairHelicity gui_interval_helicity_pair_, gui_interval_axis_pair_;
  pairmoments::SpectrumW pair_spectrum_;
  bool gui_diagnostics_enabled_ = false;
  feebl::peak_diagnostics::Accumulator gui_peak_,gui_interval_peak_;
  feebl::diagnostics::CollisionMoments gui_collision_moments_;
  feebl::diagnostics::CollisionMoments gui_interval_collision_moments_;
  double gui_interval_sw_=0.0,gui_interval_h1_=0.0,gui_interval_h2_=0.0;
  feebl::diagnostics::PrimaryChiHistograms gui_chi_e_;
  feebl::diagnostics::Histogram gui_chi_gamma_{-10.0,3.0};
  feebl::diagnostics::Histogram gui_photon_energy_;
  feebl::diagnostics::PrimaryChiHistograms gui_interval_chi_e_;
  feebl::diagnostics::Histogram gui_interval_chi_gamma_{-10.0,3.0};
  feebl::diagnostics::Histogram gui_interval_photon_energy_;
  double lumi[3][3];
  double lumi_ecm,lumi_ecm2,lumi_ecm3,lumi_ecm4;
  double hadrons_ee[3],hadrons_eg[3],hadrons_ge[3],hadrons_gg[3];
  double minijets;
  //  double temp_1,temp_2,temp_3;
  double eloss_1,eloss_2;
  double ephot_1,ephot_2;
  double c_vx_1,sig_vx_1,c_vy_1,sig_vy_1;
  double c_vx_2,sig_vx_2,c_vy_2,sig_vy_2;
  double c_vx_1_coh,c_vx_2_coh,c_vy_1_coh,c_vy_2_coh;
  double upsmax;
  std::vector<double> lumi_ee_step, lumi_ee_high_step;
  double step_lumi[2];
  
 public:
   
  RESULTS();
  
  inline double get_lumi_fine() const {return lumi_fine;} 
  inline void set_primary_only_mode3(bool enabled)
    { primary_only_mode3_ = enabled; }

  inline void cumulate3array(double source[3], double cible[3])
    {
      int k;
      for (k=0; k<3; k++) cible[k] += source[k];
    }
  
  inline void connect_switches(const SWITCHES* sw)
    {
      switches_ = sw;
    }
  
  inline void get_velocity_quantities1(double& c_vx, double& sig_vx, double& c_vy, double& sig_vy)
    {
      c_vx = c_vx_1;
      sig_vx = sig_vx_1;
      c_vy = c_vy_1;
      sig_vy = sig_vy_1;
    }
  inline void get_velocity_quantities2(double& c_vx, double& sig_vx, double& c_vy, double& sig_vy)
    {
      c_vx = c_vx_2;
      sig_vx = sig_vx_2;
      c_vy = c_vy_2;
      sig_vy = sig_vy_2;
    }

  //inline void raz_results_temp()
  //{
  //  temp_1=0.0;
  //  temp_2=0.0;
  //  temp_3=0.0;
  //}
  
  inline void add_lumi(int firstIndex, int secondIndex, double x)
    {
      lumi[firstIndex][secondIndex]+= x;
    }
  
  inline void add_lumi_ee(double x)
    {
      lumi_ee+= x;
    }
  
   inline void add_lumi_ee_step(double x)
    {
      lumi_ee_step.push_back(x);
    }

  inline void add_lumi_ee_high_step(double x)
    {
      lumi_ee_high_step.push_back(x);
    }
  inline void add_step_lumi_ee(double x)
    {
      step_lumi[0] += x;
    }
  inline void add_step_lumi_ee_high(double x)
    {
      step_lumi[1] += x;
    }

  inline void add_lumi_ge(double x)
    {
      lumi_ge += x;
    }
  
  inline void add_lumi_eg(double x)
    {
      lumi_eg += x;
    }
  
  inline void add_lumi_gg(double x)
    {
      lumi_gg += x;
    }
  
  inline void add_lumi_gg_high(double x)
    {
      lumi_gg_high += x;
    }
  
  
  inline void add_lumi_pp(double x)
    {
      lumi_pp+= x;
    }

  inline void add_lumi_weighted_helicity(double w, double h1, double h2)
    {
      lumi_wsum  += w;
      lumi_whel1 += w*h1;
      lumi_whel2 += w*h2;
      lumi_helicity_pair_.add(w,h1,h2);
      if (gui_diagnostics_enabled_) {
        gui_interval_sw_+=w; gui_interval_h1_+=w*h1; gui_interval_h2_+=w*h2;
        gui_interval_helicity_pair_.add(w,h1,h2);
      }
    }

  inline void add_lumi_weighted_helicity(double w,double h1,double h2,
                                         double axis1,double axis2)
    {
      add_lumi_weighted_helicity(w,h1,h2);
      lumi_axis_pair_.add(w,axis1,axis2);
      if(gui_diagnostics_enabled_)gui_interval_axis_pair_.add(w,axis1,axis2);
    }
  inline const pairmoments::PairHelicity& helicity_correlation(bool interval=false) const
    { return interval ? gui_interval_helicity_pair_ : lumi_helicity_pair_; }
  inline const pairmoments::PairHelicity& axis_correlation(bool interval=false) const
    { return interval ? gui_interval_axis_pair_ : lumi_axis_pair_; }

  // EXACT PAIR INVARIANT MASS, BINNED.
  //
  // The three sums above are first moments and nothing else, so the
  // luminosity-weighted helicities could be reported but not their spread,
  // their correlation, or how many pairs carried the weight -- and there was no
  // W at all, so no dL/dW. See src/pairMomentsCPP.h.
  //
  // W is computed from the two four-momenta with the electron mass kept. For
  // equal-energy head-on pairs the mass cancels exactly and W = 2E, so this
  // costs nothing there; it matters for asymmetric pairs, which is precisely
  // what a spectrum's tails are made of.
  inline void configure_pair_spectrum(double lo, double hi, std::size_t n)
    { pair_spectrum_.configure(lo, hi, n); }
  inline void add_pair_W(double w, double W)
    { pair_spectrum_.add(w, W); }
  inline void add_pair_W(double w, double W, double h1, double h2)
    { pair_spectrum_.add(w, W, h1, h2); }
  inline const pairmoments::SpectrumW& pair_spectrum() const
    { return pair_spectrum_; }

  inline void enable_gui_diagnostics(bool enabled) { gui_diagnostics_enabled_ = enabled; }
  inline bool gui_diagnostics_enabled() const { return gui_diagnostics_enabled_; }
  inline void configure_gui_peak(const feebl::peak_diagnostics::Definition& definition) {
    gui_peak_=feebl::peak_diagnostics::Accumulator(definition);
    gui_interval_peak_=feebl::peak_diagnostics::Accumulator(definition);
  }
  inline const feebl::peak_diagnostics::Accumulator& gui_peak() const {return gui_peak_;}
  inline const feebl::peak_diagnostics::Accumulator& gui_interval_peak() const {return gui_interval_peak_;}
  inline void add_gui_peak(double weight,double W,double e1,double e2,
                           bool spin=false,double h1=0,double h2=0) {
    if(!gui_diagnostics_enabled_)return;
    gui_peak_.add(weight,W,e1,e2,spin,h1,h2);
    gui_interval_peak_.add(weight,W,e1,e2,spin,h1,h2);
  }
  inline const feebl::diagnostics::CollisionMoments& gui_collision_moments() const
    { return gui_collision_moments_; }
  inline void add_gui_collision_moments(double weight,
      const ABSTRACT_PARTICLE& p1, const ABSTRACT_PARTICLE& p2)
    {
      if (!gui_diagnostics_enabled_) return;
      PHI_FLOAT x1,y1,x2,y2; double vx1,vy1,vx2,vy2;
      p1.XYposition(x1,y1); p2.XYposition(x2,y2);
      p1.velocities(vx1,vy1); p2.velocities(vx2,vy2);
      gui_collision_moments_.beam1.add(weight,p1.energy(),x1,y1,vx1,vy1);
      gui_collision_moments_.beam2.add(weight,p2.energy(),x2,y2,vx2,vy2);
      gui_interval_collision_moments_.beam1.add(weight,p1.energy(),x1,y1,vx1,vy1);
      gui_interval_collision_moments_.beam2.add(weight,p2.energy(),x2,y2,vx2,vy2);
    }
  inline void add_gui_chi_e(int beam,double value) {
    if (gui_diagnostics_enabled_) { gui_chi_e_.add(beam,value); gui_interval_chi_e_.add(beam,value); }
  }
  inline void add_gui_chi_gamma(double value) {
    if (gui_diagnostics_enabled_) { gui_chi_gamma_.add(value); gui_interval_chi_gamma_.add(value); }
  }
  inline void add_gui_photon_energy(double value) {
    if (gui_diagnostics_enabled_) { gui_photon_energy_.add(std::fabs(value)); gui_interval_photon_energy_.add(std::fabs(value)); }
  }
  inline const feebl::diagnostics::Histogram& gui_chi_e() const { return gui_chi_e_.combined; }
  inline const feebl::diagnostics::Histogram& gui_chi_e_beam(int beam) const { return gui_chi_e_.beam(beam); }
  inline const feebl::diagnostics::Histogram& gui_chi_gamma() const { return gui_chi_gamma_; }
  inline const feebl::diagnostics::Histogram& gui_photon_energy() const { return gui_photon_energy_; }
  inline void reset_gui_interval() {
    gui_interval_peak_=feebl::peak_diagnostics::Accumulator(gui_peak_.definition);
    gui_interval_collision_moments_=feebl::diagnostics::CollisionMoments();
    gui_interval_sw_=gui_interval_h1_=gui_interval_h2_=0.0;
    gui_interval_helicity_pair_=pairmoments::PairHelicity();
    gui_interval_axis_pair_=pairmoments::PairHelicity();
    gui_interval_chi_e_=feebl::diagnostics::PrimaryChiHistograms();
    gui_interval_chi_gamma_=feebl::diagnostics::Histogram(-10.0,3.0);
    gui_interval_photon_energy_=feebl::diagnostics::Histogram();
  }
  inline const feebl::diagnostics::CollisionMoments& gui_interval_collision_moments() const { return gui_interval_collision_moments_; }
  inline const feebl::diagnostics::Histogram& gui_interval_chi_e() const { return gui_interval_chi_e_.combined; }
  inline const feebl::diagnostics::Histogram& gui_interval_chi_e_beam(int beam) const { return gui_interval_chi_e_.beam(beam); }
  inline const feebl::diagnostics::Histogram& gui_interval_chi_gamma() const { return gui_interval_chi_gamma_; }
  inline const feebl::diagnostics::Histogram& gui_interval_photon_energy() const { return gui_interval_photon_energy_; }
  inline double gui_interval_sw() const { return gui_interval_sw_; }
  inline double gui_interval_h1() const { return gui_interval_h1_; }
  inline double gui_interval_h2() const { return gui_interval_h2_; }
  inline double get_lumi_helicity_sum1() const { return lumi_whel1; }
  inline double get_lumi_helicity_sum2() const { return lumi_whel2; }

  // Canonical reduction boundary for worker-local primary/photon collision
  // state.  It covers every RESULTS field written by collide_ee,
  // beamstrahlung and the factorized photon-luminosity path.  Per-diagonal
  // step vectors are deliberately excluded: the wave scheduler publishes
  // those once, after summing the workers for that wave.
  bool merge_primary_photon_collision_from(const RESULTS& other);
  bool merge_primary_photon_collision_from_many(
      const std::vector<const RESULTS*>& others);

  // Reuse a worker-local lane without retaining any result, step, or spectrum
  // state from its previous encounter.  Spectrum geometry and the immutable
  // SWITCHES connection are preserved.
  void reset_primary_photon_collision_lane();

  inline double get_lumi_weighted_helicity1() const
    { return (lumi_wsum > 0.0) ? lumi_whel1/lumi_wsum : 0.0; }
  inline double get_lumi_weighted_helicity2() const
    { return (lumi_wsum > 0.0) ? lumi_whel2/lumi_wsum : 0.0; }
  inline double get_lumi_weight_sum() const { return lumi_wsum; }
  
  inline void add_lumi_ee_high(double x)
    {
      lumi_ee_high += x;
    };

  inline void add_lumi_ecm(double x)
    {
      lumi_ecm += x;
    }
  
  inline void add_lumi_ecm2(double x)
    {
      lumi_ecm2 += x;
    }
  
  void set_step_lumi_ee(double x){ step_lumi[0] = x; };
  void set_step_lumi_ee_high(double x) { step_lumi[1] = x; }   
  double get_step_lumi_ee() const { return step_lumi[0]; }
  double get_step_lumi_ee_high() const { return step_lumi[1]; }
  
  
  inline void updateUpsmax(float x) { upsmax=std::max(upsmax,(double)x);};
  
  // this method isn't great, see how useful it is and how to avoid
  inline void cumulate_hadrons_ee(double temp[3])
    {
      cumulate3array(temp, hadrons_ee);
      //  hadrons_ee[0] = temp_1; 
      //  hadrons_ee[1] = temp_2; 
      //  hadrons_ee[2] = temp_3; 
    }
  
  // this method isn't terrific, see how useful it is and how to avoid
  inline void cumulate_hadrons_ge(double temp[3])
    {
      cumulate3array(temp, hadrons_ge);
      //  hadrons_ge[0] = temp_1; 
      //  hadrons_ge[1] = temp_2; 
      //  hadrons_ge[2] = temp_3; 
    }
  
  // this method isn't terrific, see how useful it is and how to avoid
  inline void  cumulate_hadrons_eg(double temp[3])
    {
      cumulate3array(temp, hadrons_eg);
      //  hadrons_eg[0] = temp_1; 
      //  hadrons_eg[1] = temp_2; 
      //  hadrons_eg[2] = temp_3; 
    }
  
  // this method isn't terrific, see how useful it is and how to avoid
  inline void  cumulate_hadrons_gg(double temp[3])
    {
      cumulate3array(temp, hadrons_gg);
      //  hadrons_gg[0] = temp_1; 
      //  hadrons_gg[1] = temp_2; 
      //  hadrons_gg[2] = temp_3; 
    }
  
  
  
  inline void add_lumi_fine(double x)
    {
      lumi_fine += x;
    }

  void bpm_signal(const BEAM& beam);
  
  void bpm_signal_coherent(const BEAM& beam) ;

  
  void output_flow(std::ostringstream& out ) const ;
  virtual std::string  output_flow() const ;

 };

class PAIRS_RESULTS : public ABSTRACT_IO_CLASS
{
 public:
  // Exact-kernel importance contributions, before integer chunking or thinning.
  // Repeated stored tracks are correlated; these sums are diagnostics, not a
  // substitute for covariance estimated across independent bunch crossings.
  struct ExactWeightDiagnostics {
    std::uint64_t kernel_evaluations = 0;
    double sum_contributions = 0.0;
    double sum_squared_contributions = 0.0;
    double max_contribution = 0.0;
    std::uint64_t accepted_chunks = 0;
    std::uint64_t stored_records = 0;
  };

 private:
  ExactWeightDiagnostics exact_weights_[3]{};
  static void require_exact_process(int process) {
    if (process < 0 || process > 2)
      throw std::out_of_range("Pair diagnostic process must be BW=0, BH=1 or LL=2");
  }
  bool gui_diagnostics_enabled_ = false;
  feebl::diagnostics::Histogram gui_pair_energy_;
  feebl::diagnostics::Histogram gui_interval_pair_energy_;
  int number_;
  double energy_;
  double eproc_[3],nproc_[3];
  // photon_polarisation=1 diagnostic: expected particle numbers of the same photon pairs for unpolarised
  // photons (Breit-Wheeler: exact; Bethe-Heitler: same events, unpolarised weight); printed only if non-zero
  double nproc_unpol_[3];
  double b1_,b2_,n1_,n2_;
  std::string name_;

  double highptsum_, highpteng_;


  void set();
  void update_contribution(double ener,double px,double py,double pz, double wgt);
  
  
 public: 
  
 PAIRS_RESULTS()  {set();}

 // Record one importance contribution in expected pairs per bunch crossing.
 // Zero contributions count as evaluations. Reject invalid values/overflow.
 inline void record_exact_weight(int process, double contribution) {
   require_exact_process(process);
   auto& d = exact_weights_[process];
   const double next_sum = d.sum_contributions + contribution;
   const double next_sum2 = d.sum_squared_contributions + contribution*contribution;
   if (!qedbg::finite_nonnegative(contribution) ||
       !qedbg::finite_nonnegative(next_sum) || !qedbg::finite_nonnegative(next_sum2))
     throw std::domain_error("Invalid exact-pair contribution or diagnostic overflow");
   ++d.kernel_evaluations;
   d.sum_contributions = next_sum;
   d.sum_squared_contributions = next_sum2;
   d.max_contribution = std::max(d.max_contribution, contribution);
 }
 // Count the stochastic chunks and records actually admitted by their cuts.
 inline void record_exact_accepted_chunk(int process) {
   require_exact_process(process);
   ++exact_weights_[process].accepted_chunks;
 }
 inline void record_exact_stored_record(int process) {
   require_exact_process(process);
   ++exact_weights_[process].stored_records;
 }
 inline const ExactWeightDiagnostics& exact_weight_diagnostics(int process) const {
   require_exact_process(process);
   return exact_weights_[process];
 }
 inline bool has_exact_weight_diagnostics() const {
   return exact_weights_[0].kernel_evaluations || exact_weights_[1].kernel_evaluations ||
          exact_weights_[2].kernel_evaluations;
 }
 inline void reset_exact_weight_diagnostics() {
   for (auto& d : exact_weights_) d = ExactWeightDiagnostics{};
 }
 
 //  PAIRS_RESULTS() : secondaries_pointer_(NULL) {set();}
 //  PAIRS_RESULTS(PAIR_BEAM* secondaries) {secondaries_pointer_ = secondaries; set();}

 inline void enable_gui_diagnostics(bool enabled) { gui_diagnostics_enabled_=enabled; }
 inline const feebl::diagnostics::Histogram& gui_pair_energy() const { return gui_pair_energy_; }
 inline const feebl::diagnostics::Histogram& gui_interval_pair_energy() const { return gui_interval_pair_energy_; }
 inline void reset_gui_interval() { gui_interval_pair_energy_=feebl::diagnostics::Histogram(); }
 inline void set_name(std::string name){name_=name;}
 inline int number() const {return number_;}
 inline double energy() const {return energy_;}
 
 void store_full_pair(int index_of_process, double e1,double px1,double py1,double pz1,double e2,double px2,double py2,double pz2, double wgt,  bool luckyPair );
 inline void add_unpolarised(int index_of_process, double wgt) { nproc_unpol_[index_of_process] += wgt; }
 
 inline void storep_(int index_of_process, double e,double wgt, bool lucky)
   {
     update_contribution(e,0.0,0.0,e, wgt);
     eproc_[index_of_process]+= fabs(e) * wgt;
     nproc_[index_of_process]+= wgt;
     if (lucky)
       {
	 number_++;
	 energy_ += fabs(e);
       }
   } /* storep_ */
 
 virtual std::string output_flow() const ;

 
};

class COMPT_RESULTS : public ABSTRACT_IO_CLASS
{
  int number_;
  double energy_;
  double eproc_[2],nproc_[2];
  double b1_,b2_,n1_,n2_;


 public:
  
  COMPT_RESULTS();
  
  int store_compt(int index_of_process,double e,double px,double py,double pz,double wgt,RNDM& rndm_generator);

  virtual  inline std::string output_flow() const 
    {
      std::ostringstream out;
      out << title(std::string("Compton results"));
      out << "compt_npart_0 = " << nproc_[0] << " compt_epart_0 = " << eproc_[0] << std::endl;
      out << "compt_npart_1 = " << nproc_[1] << " compt_epart_1 = " << eproc_[1] << std::endl;
      out << "compt_n.1 = " << n1_ << " compt_b.1 = " << b1_ << std::endl;
      out << "compt_n.2 = " << n2_ << " compt_b.2 = " << b2_ << std::endl;
      return out.str();
    }
};

/*
  class MUON_RESULTS : public ABSTRACT_IO_CLASS
  {
  double eproc[3],nproc[3];
  
  public:
  
  MUON_RESULTS();
  
  inline std::string output_flow() const 
  {
  std::ostringstream out;
  out << title(std::string("Muons results"));
  out << "muon_n.0 = " << nproc[0] << " muon_e.0 = " << eproc[0] << std::endl;
  out << "muon_n.1 = " << nproc[1] << " muon_e.1 = " << eproc[1] << std::endl;
  out << "muon_n.2 = " << nproc[2] << " muon_e.2 = " << eproc[2] << std::endl;
  
  out << "muon_n_sum = " << nproc[0] + nproc[1]+nproc[2] <<  " muon_e_sum = " << eproc[0]+eproc[1] + eproc[2] << std::endl;
  
  return out.str();
  }
  };
*/

class JET_RESULTS : public ABSTRACT_IO_CLASS
{
  double sigma_[3];
  
  
 public:
  
  JET_RESULTS()
    {
      int k;
      for (k=0; k<3; k++) sigma_[k] = 0.0;
    }
  
  inline void increment_sigma(int index, double x)
    {
     sigma_[index] += x;
    }
  
  virtual inline std::string output_flow() const 
    {
      std::ostringstream out;
      out << title(std::string("jets results"));
      out << " nb of minijet evts per bx due to the direct process : jets0 = " << sigma_[0] << std::endl;
      out << "nb of minijet evts per bx due to the once resolved process : jets1 = " << sigma_[1] << std::endl;
      out << "nb of minijet evts per bx due to the twice resolved process : jets2 = " << sigma_[2] << std::endl;
      return out.str();
    }
};


/* class  PHOTON_RESULTS : public ABSTRACT_IO_CLASS */
/* { */
/*     double energy_[2]; */
/*     unsigned long int number_[2];  */
/*  public: */
/* PHOTON_RESULTS() */
/*   { */
/*     energy_[0]=0.0; */
/*     energy_[1]=0.0; */
/*     number_[0]=0; */
/*     number_[1]=0; */
/*   }; */
/*  inline void addPhoton(int nbeam, float energy) */
/*       { */
/* 	number_[nbeam-1]++; */
/* 	energy_[nbeam-1]+=energy; */
/*       }; */
/*  inline std::string output_flow() const  */
/* { */
/*   std::ostringstream out; */
/*   out << title(std::string("photons results")); */
/*   out << "e_phot.1 = " << energy_[0]/std::max(double(1.0),double(number_[0])) << " e_phot.2 = " << energy_[1]/std::max(double(1.0),double(number_[1])) << std::endl; */
/*   return out.str(); */
/* } */
/* }; */

class COHERENT_RESULTS : public ABSTRACT_IO_CLASS 
{
  double sum_,sum2_,sumeng_,upsmax_,probmax_,sumreal_,engreal_;
  double total_energy_;
  long ncall_,count_;
  int do_coherent_;

  public :
    
    COHERENT_RESULTS() 
    {
      sum_          = 0.0;
      sum2_         = 0.0;
      sumeng_       = 0.0;
      ncall_        = 0;
      upsmax_       = 0.0;
      probmax_      = 0.0;
      sumreal_      = 0.0;
      engreal_      = 0.0;
      count_        = 0;
      total_energy_ = 0.0;
      do_coherent_ = 0;
    }

  inline void init() { do_coherent_ = 1;}
  
  inline void updateProbmax(PHI_FLOAT tmp)
    {
      probmax_ = std::max(probmax_,tmp);
    }
  inline void updateSumeng(float ener)
    {
      sumeng_ += ener;
    }
  inline void addCountEnergy(float ener)
    {
      count_++;
      total_energy_ += fabs(ener);
    }
  
  inline void updateSums(float wt, float ener, float ups)
    {
      sum_ += wt;
      sum2_ += wt*wt;
      sumeng_ += wt*ener;
      upsmax_ = std::max(upsmax_, double(ups));
    }
  
  virtual std::string output_flow() const ;


};

class TRIDENT_RESULTS : public ABSTRACT_IO_CLASS 
{
  //  double sum_,sum2_,sumeng_,upsmax_,probmax_,sumreal_,engreal_;
  double total_energy_;
  //  long ncall_,
  long count_;
  int do_trident_;

  public :
    
    TRIDENT_RESULTS() 
      {;
      //      sum_          = 0.0;
      //      sum2_         = 0.0;
      //      sumeng_       = 0.0;
      //      ncall_        = 0;
      //      upsmax_       = 0.0;
      //      probmax_      = 0.0;
      //      sumreal_      = 0.0;
      //      engreal_      = 0.0;
      count_        = 0;
      total_energy_ = 0.0;
      do_trident_ = 0;
    }

  inline void init() { do_trident_ = 1;}

  inline void addCountEnergy(float ener)
    {
      count_++;
      total_energy_ += fabs(ener);
    }
  
  virtual std::string output_flow() const ;

};
#endif
