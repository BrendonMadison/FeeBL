#ifndef PAIRS_SEEN
#define PAIRS_SEEN

#include <vector>
#include <list>
#include <cmath>
#include <algorithm>
#include <string>

#include "resultsCPP.h" 
#include "mathconst.h"
#include "beamCPP.h"
#include "switchesCPP.h"
#include "rndmCPP.h"
#include "particlesCPP.h"
#include "colliderImportedTrajectoriesCPP.h"
#include "importedSpinCPP.h"
#include "fileInputOutput.h"
#include "meshCPP.h"
#include "physicalTools.h"

class PAIR_TRACK
{
  long int step_,call_; 
  
  
  public : 
    
    PAIR_TRACK()
    {
      call_  = 0;
      step_  = 0;
    }
  
  inline void addStep(long int n)
    {
      call_++;
      step_ += n;
    }
  
  inline long call() const {return call_;}
  inline long step() const {return step_;}
};


class PAIR_PARAMETER
{
  double d_eps_1_,d_eps_2_,mass_;
  double s4,lns4,ecut;
  
 public:
  
 PAIR_PARAMETER():d_eps_1_(0.0),d_eps_2_(0.0),mass_(EMASS),s4(0.0),lns4(0.0),ecut(0.0) {;}
  
  void  init(BEAM& beam1, BEAM& beam2, int massflag, float pair_ecut, float pair_step, float step, int timestep);
  
  inline double get_s4() const {return s4;}
  inline double get_lns4() const {return lns4;}
  inline double get_ecut() const {return ecut;}
  
  inline void get_d_eps(double& d_1, double& d_2) const 
    {
      d_1 = d_eps_1_;
      d_2 = d_eps_2_;
    }
  
  inline double get_mass() const {return mass_;}

  void jet_equiv (float xmin,float e,int iflag,float& eph,float& q2,float& wgt, RNDM& rndm_generator) const;
  
  inline float jet_requiv(float xmin,int iflag) const
    {
      float help;
      
      if (xmin>=1.0) return 0.0;
      switch (iflag)
	{
	case 1: 
	  return log(xmin) * -.00464921660700172 * 0.5*lns4;
	case 2:
	case 3:
	case 4:
      return log(xmin) * -.003951834115951462 * 0.5*lns4;
	case 5:
	  help=lns4;
	  return 2.0*.002325*help*help;
	}
      return 0.0;
    } /* jet_requiv */
};


class  PAIR_BEAM : public ABSTRACT_IO_CLASS
{
  
  std::list<PAIR_PARTICLE> reserve_;
  std::vector< std::vector<PAIR_PARTICLE> > active_pairs_;
  std::vector<PAIR_PARTICLE> pairs0_;

  PAIR_TRACK pair_track_;
  PAIR_PARAMETER pair_parameter_;
  PAIRS_RESULTS pairs_results_;
  
  FILE_IN_OUT* file_of_events_;
  std::string event_to_store_;
  std::size_t imported_source_row_=0, pending_source_row_=0;
  bool imported_audit_enabled_=false;
  feebl::imported_trajectories::Registry imported_registry_;
  feebl::imported_spin::Registry imported_spin_registry_;
  bool imported_spin_enabled_=false;
  int count_pairs_;
  
  
  void compute_pairs_calls(long int& n1, double& e1, long int& n2, double& e2) const;
  
  inline void new_event(float e,float x,float y,float z,float vx,float vy,float vz, float ratio, int tracking, RNDM& rndm_generator, std::size_t source_row, int scheduled, int admitted, const TRIDVECTOR* spin)
    {
      int index_of_process = -99;
      // test if particle energy is above the required minumum 
      if (fabs(e) < pair_parameter_.get_ecut())   
	{
	  return;
	}    
      
      // reduce the number of stored particles if requested (to speed up tracking) 
      
      if (rndm_generator.rndm_pairs() > ratio) 
	{
	  return;
	}    

      // store particles for tracking? 
      if (!tracking ) return;
      count_pairs_++;
      PAIR_PARTICLE pair_temp = PAIR_PARTICLE(count_pairs_, index_of_process,x,y,z,vx,vy,vz, e);
      pair_temp.initialize_transport_from_energy(pair_parameter_.get_mass());
      if (spin) {
        pair_temp.initialize_imported_spin(*spin);
        pair_temp.enable_imported_field_audit();
        imported_spin_registry_.add(pair_temp,source_row,scheduled,admitted);
      }
      if(imported_audit_enabled_) {
        pair_temp.enable_imported_field_audit();
        imported_registry_.add(pair_temp,source_row,scheduled,admitted);
      }
      reserve_.push_back(pair_temp);
    }

  void admit_event_line(const std::string& line,std::size_t source_row,
                       int admitted,float ratio,int tracking,RNDM& rndm_generator);
  
 public:
  
  
  PAIR_BEAM()  : file_of_events_(NULL)
    {
      count_pairs_= 0;
    }


    inline void add_pair_steps(long n_pair_steps) 
    {
      pair_track_.addStep(n_pair_steps);
    }
    inline void set_name(std::string name){pairs_results_.set_name((std::string)name);}
  inline void set_pair_parameters(BEAM& beam1, BEAM& beam2, int massflag,float pair_ecut, float pair_step, float step, int timestep)
    {
      /* massflag: 0 for electrons, 1 for muons */
      pair_parameter_.init(beam1, beam2, massflag, pair_ecut, pair_step, step, timestep);
    }
  
  inline const PAIR_PARAMETER& get_pair_parameters() const {return pair_parameter_;}

  inline const PAIRS_RESULTS* get_results() const {return &pairs_results_;}
  inline void enable_gui_diagnostics(bool enabled) { pairs_results_.enable_gui_diagnostics(enabled); imported_audit_enabled_=enabled; }
  inline const feebl::imported_trajectories::Registry& imported_registry() const { return imported_registry_; }
  inline void enable_imported_spin(bool enabled) { imported_spin_enabled_=enabled; }
  inline void write_imported_spin(const char* path) const {
    if (imported_spin_enabled_)
      imported_spin_registry_.write(path,completed_tracking_snapshot());
  }
  inline void reset_gui_interval() { pairs_results_.reset_gui_interval(); }
  
  inline void set_load_file(std::string filename) 
    {
      if ( file_of_events_ != NULL)
	{
	  std::cerr << " PAIR_BEAM::set_load_file: only one file is actually allowed for load events " << std::endl;
	  exit(1);
	}
      file_of_events_ = new FILE_IN_OUT();
      file_of_events_->open_file(filename, "r");
      event_to_store_.clear();
    }
  
  
  inline void resize(int n_cell_z)
    {
      active_pairs_ =  std::vector< std::vector<PAIR_PARTICLE> >(n_cell_z);
      count_pairs_=0;
    }
  
  
  ~PAIR_BEAM() 
    {
      if ( file_of_events_ != NULL) { file_of_events_->close(); delete file_of_events_;}
    }
  
  inline int size_of_particle() const { return active_pairs_.size(); }
  
  inline int n_particles() const { return count_pairs_;}

  // At a completed tracking substep, reserve_ is authoritative, including
  // escaped particles. desactive_current_pairs leaves stale copies in the
  // active vectors, so those MUST NOT be concatenated with this range.
  // Only call after deactivation at the GUI diagonal barrier.
  inline const std::list<PAIR_PARTICLE>& completed_tracking_snapshot() const
    {
      if (count_pairs_ < 0 || reserve_.size() != static_cast<std::size_t>(count_pairs_))
        throw std::logic_error("secondary snapshot requested before tracking deactivation");
      return reserve_;
    }
  
  inline int numberOfParticles(int slice) const {return active_pairs_[slice].size();}
  
  inline std::vector<PAIR_PARTICLE>& get_pairs(int slice) { return active_pairs_[slice];}
  
  // active pairs are pairs which are in currently overlaping slices
  // once a step has been made and pairs moved, active pairs have to
  // desactived for further being redistributed.
  inline void desactive_current_pairs(int n)
    {
      unsigned int k;
      int j;
      for (j=0;j<n;j++) 
	{
	  if (active_pairs_[j].size()>0)
	    {
	      for (k = 0; k < active_pairs_[j].size(); k++)
		{
		  reserve_.push_back(active_pairs_[j][k]);
		}
	      //		reserve_.splice(reserve_.end(),active_pairs_[j]);  
	    }     
	}
    }
  
  void init(int n_cell_z);
  
  void distribute_pairs(float delta_z,unsigned int n);
  void move_unactive_pairs(float step);
  
  void load_events(int time_counter,float ratio, int tracking, RNDM& rndm_generator);
  
  std::string output_flow() const ;
  
  void book_keeping(const MESH& mesh, int index_of_process, double e1,double px1,double py1,double pz1,double e2,double px2,double py2,double pz2, double wgt,int cellx, int celly,float min_z,SWITCHES& switches,RNDM& rndm_generator )
    {
      bool lucky = true;
      if (wgt<rndm_generator.rndm()) lucky = false;
      pairs_results_.store_full_pair(index_of_process,e1,px1,py1,pz1,e2,px2,py2,pz2,wgt,lucky );
      if (lucky)
	{
	  /*       new_pair(mesh, cellx, celly,min_z,index_of_process, (float)e1,(float)px1,(float)py1,(float)pz1, switches.get_pair_ratio(), switches.get_track_secondaries(), switches.get_store_secondaries(), rndm_generator); */
	  /*       new_pair(mesh, cellx, celly,min_z, index_of_process, (float)e2,(float)px2,(float)py2,(float)pz2, switches.get_pair_ratio(), switches.get_track_secondaries(), switches.get_store_secondaries(), rndm_generator); */
	  new_pair(mesh, cellx, celly,min_z,index_of_process, e1,px1,py1,pz1, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	  new_pair(mesh, cellx, celly,min_z, index_of_process, e2,px2,py2,pz2, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	}
    }

  void book_keeping_muon(const MESH& mesh, int index_of_process, double e1,double px1,double py1,double pz1,double e2,double px2,double py2,double pz2, double wgt,int cellx, int celly,float min_z,SWITCHES& switches,RNDM& rndm_generator )
    {
      bool lucky = true;
      if (wgt<rndm_generator.rndm()) lucky = false;
      pairs_results_.store_full_pair(index_of_process,e1,px1,py1,pz1,e2,px2,py2,pz2,wgt/switches.get_muon_scale(),lucky );
      if (lucky)
	{
	  new_pair(mesh, cellx, celly,min_z,index_of_process, e1,px1,py1,pz1, switches.get_muon_ratio(), switches.get_track_muons(), switches.get_store_muons(), rndm_generator);
	  new_pair(mesh, cellx, celly,min_z, index_of_process, e2,px2,py2,pz2, switches.get_muon_ratio(), switches.get_track_muons(), switches.get_store_muons(), rndm_generator);
	}
    }
  
  void book_keeping_p(const MESH& mesh,int index_of_process, double e,double wgt,int cellx, int celly,float min_z,SWITCHES& switches,RNDM& rndm_generator )
    {
      bool lucky = true;
      if (wgt<rndm_generator.rndm()) lucky = false;
      pairs_results_.storep_(index_of_process, e,wgt, lucky);
      if (lucky)
	{
	  new_pair(mesh, cellx, celly, min_z,index_of_process, (e),0.0,0.0,(e), switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	}
    }
  
  // stokes1, stokes2 (photon_polarisation=1, pair_model >= 1): lab-frame Stokes parameters of the two photons
  // (qedbg_photon_polarisation.h); NULL: unpolarised (unchanged code path)
  void make_pair_bw(const MESH& mesh, int cellx, int celly,float min_z,int index_of_process, float eph1,float q2_1,float eorg1, float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1 = NULL, const float* stokes2 = NULL);

  void  make_muon(const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, float eph1,float q2_1,float eorg1,float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1 = NULL, const float* stokes2 = NULL);

  // pair_model = 2: exact lowest-order QED for photon pairs with at least one virtual photon
  // (index_of_process 1: real x virtual, 2: virtual x virtual); photon 1 belongs to beam 1 (+z),
  // photon 2 to beam 2 (-z); eorg < 0 marks a real photon.  muonPair selects l = mu.
  void make_pair_exact(const MESH& mesh, int cellx, int celly,float min_z,int index_of_process, float eph1,float q2_1,float eorg1, float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, bool muonPair, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1 = NULL, const float* stokes2 = NULL);
  
  // Generated total energy and Cartesian momentum are in GeV. Preserve their
  // precision until the canonical massive transport state has been initialized.
  void new_pair(const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, double energy,double px,double py,double pz, float ratio, int tracking, int saving, RNDM& rndm_generator );

 void new_pair(const unsigned index, const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, double energy,double px,double py,double pz, float ratio, int tracking, int saving, RNDM& rndm_generator, int beamslice1, int beamslice2);
 
  inline void save_pairs_on_file(std::string nameOfOutputFile) const
    {
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      
      std::list<PAIR_PARTICLE>::const_iterator point = reserve_.begin();
      for ( point = reserve_.begin(); point != reserve_.end(); point++)
		{
		  //      filout.save_pair_particle(*point);
		  filout.save_object_on_persistent_file( &(*point) );
		}
      filout.close();
    }
  

  inline void save_pairs0_on_file(std::string nameOfOutputFile)
    {
      unsigned int k;
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      for (k=0; k < pairs0_.size(); k++)
		{
		  //     filout.save_pair_particle(pairs0_[k]);
		  filout.save_object_on_persistent_file( &pairs0_[k] );
		}
      filout.close();
    }

  inline void save_bhabhas_on_file(std::string nameOfOutputFile) const
    {
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      std::cout << "Saving tracked Bhabha particles in " << nameOfOutputFile << std::endl;
      std::list<PAIR_PARTICLE>::const_iterator point = reserve_.begin();
      for ( point = reserve_.begin(); point != reserve_.end(); point++)
		{
		  //      filout.save_pair_particle(*point);
		  filout.save_object_on_persistent_file(point->get_label(), &(*point) );
		}
      filout.close();
    }
  
  inline void save_bhabhas0_on_file(std::string nameOfOutputFile)
    {
      unsigned int k;
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      std::cout << "Saving boosted Bhabha particles in " << nameOfOutputFile << std::endl;
      for (k=0; k < pairs0_.size(); k++)
		{
		  //     filout.save_pair_particle(pairs0_[k]);
		  filout.save_object_on_persistent_file(pairs0_[k].get_label() , &pairs0_[k] );
		}
      filout.close();
    }
};


#endif
