#ifndef RUN_SEEN
#define RUN_SEEN

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "rndmCPP.h"
#include "incomingMapsCPP.h"
#include "beamParametersCPP.h"
#include "config.h"
#include "gridCPP.h"
#include "switchesCPP.h"
#include "beamCPP.h"
#include "polarizationCPP.h"
#include "pairsCPP.h"
#include "timerCPP.h"
#include "fileInputOutput.h"
#include "mathematicalEntities.h"
#include "physconst.h"
#include "mathematicalTools.h"
#include "parametersCPP.h"
#include "gpuResident.h"
#include "gpuWaveBatch.h"
#ifdef FEEBL_HAVE_GPU5
#include "gpuAdaptiveWaveBatch.h"
#endif
#include "legacyWavefrontSchedulerCPP.h"
#include "wavefrontFieldWorkspaceCPP.h"
#include "colliderSnapshotFieldsCPP.h"
#include "colliderMeshCoverageCPP.h"
#include "colliderSecondaryFramesCPP.h"


class FEEBL :  public ABSTRACT_IO_CLASS
{

  TIMER time_;
  RNDM generator_;
  FFT_SERVER fourier_server_;

  PARAMETERS params_;

  SWITCHES switches;
  PHOTON_BEAM photon_vector_;

  GRID grid_;
  std::vector<GENERAL_GRID*> gridsPtr_;
  // Owned here rather than by GENERAL_GRID/FIELD: EXTRA_GRID relies on the
  // historical implicit GENERAL_GRID copy constructor, while the resident
  // solver is deliberately non-copyable and owns CUDA resources.
  std::vector<feebl::gpu::ResidentFieldSolver*> gpu2_field_solvers_;
  std::vector<float> gpu2_packed_xy1_;
  std::vector<float> gpu2_packed_xy2_;
  // Preallocated compact device-to-host boundary for gpu2_scope=primary.
  // Entries are (Ex,Ey) in the historical particle order of each active
  // slice.  No host phi array is populated on this plan.
  std::vector<float> gpu2_primary_fields1_;
  std::vector<float> gpu2_primary_fields2_;
  bool gpu2_direct_deposition_;
  bool gpu2_primary_scope_;
  std::uint64_t gpu2_primary_beam1_in_;
  std::uint64_t gpu2_primary_beam1_out_;
  std::uint64_t gpu2_primary_beam2_in_;
  std::uint64_t gpu2_primary_beam2_out_;

  // gpu_mode=4 owns one fixed-capacity PlanMany field pipeline.  Its compact
  // outputs feed the existing ordered host make_step boundary; no stochastic
  // collision state is resident on the device.
  std::unique_ptr<feebl::gpu::WaveBatchedFieldSolver> gpu4_wave_solver_;
  std::vector<float> gpu4_packed_xy1_;
  std::vector<float> gpu4_packed_xy2_;
  std::vector<feebl::wavefront_fields::FieldPrefetch> gpu4_prefetch_;
  std::size_t gpu4_batch_width_;
  std::size_t gpu4_maximum_particles1_;
  std::size_t gpu4_maximum_particles2_;

#ifdef FEEBL_HAVE_GPU5
  // gpu_mode=5 keeps class-keyed cuFFT/Green contexts resident for the whole
  // crossing.  Slice revisions are monotonic host publication epochs; source
  // ID order tokens are stable because the qualified primary domain neither
  // inserts, removes, nor reorders primaries.
  std::unique_ptr<feebl::adaptive_grid::gpu5::AdaptiveWaveBatchExecutor>
      gpu5_wave_solver_;
  std::unique_ptr<feebl::adaptive_grid::CanonicalLatticeSnapshot>
      gpu5_lattice_;
  std::vector<std::uint64_t> gpu5_beam1_revisions_;
  std::vector<std::uint64_t> gpu5_beam2_revisions_;
  std::vector<std::uint64_t> gpu5_beam1_order_tokens_;
  std::vector<std::uint64_t> gpu5_beam2_order_tokens_;
  std::size_t gpu5_batch_capacity_;
#endif

  // collision_schedule=wavefront_parallel leases exactly one isolated
  // GRID/FIELD/FFT workspace to each persistent CPU worker lane.  The default
  // selector parallelizes fields only.  wavefront_parallel_physics=1 also
  // executes the audited primary/photon/luminosity encounter in these lanes
  // and publishes results at each batch barrier in canonical encounter order.
  std::vector<std::unique_ptr<feebl::wavefront_fields::Workspace> >
      wavefront_field_workspaces_;
  std::vector<feebl::wavefront_fields::FieldPrefetch>
      wavefront_field_prefetch_;
  std::vector<std::size_t> wavefront_physics_lane_for_position_;
  std::size_t wavefront_field_workers_;
  std::size_t wavefront_workspace_bytes_each_;
  std::size_t wavefront_workspace_bytes_reserved_;
  double wavefront_field_setup_ms_;

  BEAM beam1_,beam2_;
  POLARIZATION_REPORT polarization_;
  BEAM_PARAMETERS beam_parameters1_,beam_parameters2_;

  PAIR_BEAM secondaries_;
  PAIR_BEAM muons_;

  BESSEL bessel;
  int time_counter_;

  std::string event_input_file_;
  // User-visible PARAMETERS block label used in the mode-3 execution receipt.
  std::string mode3_input_card_label_;
  // A coupled-controller request for genuinely new roots is a process-level
  // handoff, never an in-process population mutation.  The collision records
  // it here and FEEBL::run enforces the pre-output publication gate.
  bool adaptive_root_replay_required_;
  std::string adaptive_root_replay_reason_;
  std::string adaptive_root_replay_recommended_action_;
  std::uint64_t adaptive_root_replay_controller_observations_;
  std::uint64_t adaptive_root_replay_trace_hash_;
  std::string cross_input_file_;
  std::string photon_input_file_;
  std::string bhabha_input_file_;
  std::string bhabha_photon_input_file_;

  std::string hadronfile_;
  std::string photon_file_;
  std::string compton_phot_file_;
  std::string secondaries_file_;
  std::string secondaries0_file_;
  std::string muons_file_;
  std::string muons0_file_;
  std::string bhabha_prod_;
  std::string bhphoton_prod_; // intial bhabhas photons
  std::string bhphotons_; // boosted bhabhas photons


  std::string jet_file_;
  std::string lumi_ee_file_;
  std::string lumi_eg_file_;
  std::string lumi_ge_file_;
  std::string lumi_gg_file_;
  std::string beam1_file_;
  std::string beam2_file_;
  std::string coh1_file_;
  std::string coh2_file_;
  std::string tri1_file_;
  std::string tri2_file_;
  std::string tertphot_file_;

  GENERAL_CROSS* cross_;

  int number_of_stored_particles1_;
  int number_of_stored_particles2_;


  bool check_parameters() const;
  void configure_compute_backend();
  void initialize_collision_state();
  void finalize_collision(const std::string& prot);

  inline void init_input_file_names()
    {
      event_input_file_ = std::string("event.ini");
      cross_input_file_ = std::string("cross.ini");
      photon_input_file_ = std::string("photon.ini");
      bhabha_input_file_ = std::string("bhabha.ini");
      bhabha_photon_input_file_ = std::string("bhabha_photon.ini");
      
    }
  
  inline void init_output_file_names()
    {
      secondaries_file_ = std::string("pairs.dat");
      secondaries0_file_ = std::string("pairs0.dat");
      muons_file_ = std::string("muons.dat");
      muons0_file_ = std::string("muons0.dat");
      jet_file_ = std::string("minijet.dat");
      
      bhabha_prod_ = std::string("bhabha_prod.dat"); // initial bhabhas
      bhphoton_prod_ = std::string("bhphoton_prod.dat"); // initial bhabhas photons
      bhphotons_ = std::string("bhphotons.dat"); // boosted bhabhas photons
      
      lumi_ee_file_ = std::string("lumi.ee.out");
      lumi_eg_file_ = std::string("lumi.eg.out");
      lumi_ge_file_ = std::string("lumi.ge.out");
      lumi_gg_file_ = std::string("lumi.gg.out");
      beam1_file_ = std::string("beam1.dat");
      beam2_file_ = std::string("beam2.dat");
      coh1_file_ = std::string("coh1.dat");
      coh2_file_ = std::string("coh2.dat");
      tri1_file_ = std::string("tri1.dat");
      tri2_file_ = std::string("tri2.dat");
      tertphot_file_ = std::string("tertphot.dat");

      if (switches.get_write_photons())
	{
	  photon_file_ = std::string("photon.dat");
	}
      else photon_file_ = std::string("");
      
      if ( switches.get_do_compt_phot() )
	{
	  compton_phot_file_ = std::string("compton.photon");
	}
      else compton_phot_file_ = std::string("");
      
      
      
      if (switches.get_store_hadrons())
	{
	  hadronfile_ = std::string("hadron.dat");
	}
      else hadronfile_ = std::string("");
    }
  
  
  
  void fprint_beam_parameters(FILE *file);

  inline  float sigma_for_cut_from_data(char dir) const
    {
      float sig, sig2;
      sig  = beam1_.sigma_xyz_from_data(dir);
      sig2 = beam2_.sigma_xyz_from_data(dir);
      if (sig < sig2) sig = sig2;
      return sig;
    }

  
  inline void transfer_loaded_particles_in_beam(BEAM& beam, const BEAM_PARAMETERS& bp, BEAM_FROM_FILE*& bff, float size_z)
    {
      unsigned int k;
      unsigned int nb_particles_read;
      float deltaz, sigx, sigy, sigz;
      float sigxTest, sigyTest, sigzTest;
      float dummy; 
      deltaz=2.0*size_z/((float)grid_.get_n_cell_z());
      sigx = bp.sigma_x();
      sigy = bp.sigma_y();
      sigz = bp.sigma_z();
      if ( sigx <= 0.0 || sigy <= 0.0 || sigz <= 0.0) 
	{
	  bff->beamXyRms(dummy, dummy, sigxTest, sigyTest);
	  bff->beamZRms(dummy, sigzTest);
	  if (sigx <= 0.0) sigx = sigxTest;
	  if (sigy <= 0.0) sigy = sigyTest;
	  if (sigz <= 0.0) sigz = sigzTest;
	}
      nb_particles_read = beam.load_particles(bff,switches.get_emin(), -size_z,deltaz, sigx, sigy, sigz);
      for (k=0; k < gridsPtr_.size(); k ++)
	{
       gridsPtr_[k]->set_nb_macroparticles(beam.label(),nb_particles_read);
	}
    }
  
  inline  void init_grid_phys(float n_particles1, float n_particles2, const std::vector<float>& size_x, const std::vector<float>& size_y, float size_z, int n_cell_x, int n_cell_y)
    {
      int extra_grids;
      unsigned int i1;
      grid_.init_grid_comp (n_cell_x, n_cell_y, switches.get_integration_method(), &fourier_server_);
      
      grid_.init_grid_phys(n_particles1, n_particles2,size_x[0],size_y[0],size_z, switches.get_charge_sign(), &fourier_server_);
      extra_grids = switches.get_extra_grids();
      if ( extra_grids < 0 ) extra_grids = 0;
      gridsPtr_ = std::vector<GENERAL_GRID*>(extra_grids+1);
      gridsPtr_[0] = &grid_;
      for (i1=1; i1 <  gridsPtr_.size() ;i1++)
	{
	  gridsPtr_[i1] = new EXTRA_GRID(grid_);
	  gridsPtr_[i1]->init_grid_phys(n_particles1, n_particles2,size_x[i1],size_y[i1],size_z, switches.get_charge_sign(), &fourier_server_);
	}
      if(switches.get_do_pairs()||switches.get_do_hadrons()||switches.get_do_compt()
	 ||switches.get_do_muons())
	{
	  grid_.init_extra_photons();
	  
	}
    }
  
  inline int adjust_nb_cells_from_cut(float cut, float sigma) const
    {
      float rapp = cut/sigma;
      // for every sigma, 8 cells of grid...
      return TOOLS::nearest_power_of_2(8.0*rapp);
    }
  
  void xycuts_for_grids(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 , int nbgrids, std::vector<float>& size_x, std::vector<float>& size_y, int& updated_n_cell_x, int& updated_n_cell_y ) const;
  float zcut_for_grids(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2) const;
  
  void main_grid_xycuts_from_loaded_beam(const BEAM_FROM_FILE& bff1,const BEAM_FROM_FILE& bff2, float& size_x, float& size_y ) const;
  
  float main_grid_zcut_from_loaded_beam(const BEAM_FROM_FILE& bff1,const BEAM_FROM_FILE& bff2) const;
  
  
  inline void main_grid_xycuts_from_user(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ,float& size_x, float& size_y ) const
    {
      if ( switches.get_load_beam() && switches.get_cuts_from_loaded_beam() ) 
	{
	  // cuts computed as cut_factor*sigma; cut_factor given in input data (defaut : 3) ; sigma computed from read beam
	  main_grid_xycuts_from_loaded_beam(dynamic_cast<const BEAM_FROM_FILE&>(bff1),dynamic_cast<const BEAM_FROM_FILE&>(bff2), size_x, size_y);
	}
      else
	{
	  // cuts given as such by the input file. If not, default values 3*sigma, provided the beam characteristics have been given in input data.
	  size_x = transverse_cut_from_data('x');
	  size_y = transverse_cut_from_data('y');   
	}  
    }
  inline float main_grid_zcut_from_user(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ) const
    {
      float size_z;
      if ( switches.get_load_beam() && switches.get_cuts_from_loaded_beam() ) 
	{
	  // cuts computed as cut_factor*sigma; cut_factor given in input data (defaut : 3) ; sigma computed from read beam
	  size_z = main_grid_zcut_from_loaded_beam(dynamic_cast<const BEAM_FROM_FILE&>(bff1),dynamic_cast<const BEAM_FROM_FILE&>(bff2) );
	}
      else
	{
	  size_z = cut_z_from_data(0);
	}  
      return size_z;
    }
  
  
  void main_grid_automatic_xycuts(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ,float& size_x, float& size_y, int& new_n_cell_x, int& new_n_cell_y) const;
  
  float default_zcut_from_beams(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ) const;
  
  inline void test_size_due_to_cdm(float gamma, float sigmax, float sigmay, float sigmaz, float betax, float betay) const
    {
      // DELTA = f*sigmaz*THETA
      // THETA = 0.5*DD*(sigma/sigmaz)*f_form
      // DELTA = f*0.5*DD*sigma*f_form
      // on choisit A = 1, c-a-d beta = 10^-6 sigmaz
      std::ofstream out1, out2;
      float delta, AA, DD, formfct, factor,  decalage;
      //float sigma, 
      float theta0;
      float npart = beam_parameters1_.n_particles();
      // the sigmas are in nanometres
      theta0 = 2.*npart*RE/(gamma*(sigmax+sigmay)*1.0e-9);
      DD = theta0*sigmaz/sigmax;
      AA = sigmaz/(betax*1.0e6);
      //   std::cout << " ******** Dx = " << DD << " AA = " << AA << " ************ " << std::endl;
      out1.open("disrupx.dat", std::ios::out);
      for (delta = 0.; delta <= 5. ; delta +=  0.01)
	{
	  formfct = form_function(AA, DD, delta);
	  factor = offset_factor(0.5*delta);
	  decalage = 0.5*formfct*DD*factor*sigmax;
	  //	 std::cout << " delta = " << delta << " decal= " << decalage << " form factor : " << formfct << " factor: " << factor << std::endl;
	  out1 << delta << " " << decalage << " " << factor << std::endl;
	}
      out1.close();
      DD = theta0*sigmaz/sigmay;
      AA = sigmaz/(betay*1.0e6);
      //   std::cout << " ******** Dy = " << DD << " AA = " << AA << " ************ " << std::endl;
      out2.open("disrupy.dat", std::ios::out);
      for (delta = 0.; delta <= 5. ; delta +=  0.01)
	{
	  formfct = form_function(AA, DD, delta);
	  factor = offset_factor(0.5*delta);
	  decalage = 0.5*formfct*DD*factor*sigmay;
	  //	 std::cout << " delta = " << delta << " decal= " << decalage << " form factor : " << formfct << " factor: " << factor << std::endl;
	  out2 << delta << " " << decalage << " " << factor << std::endl;
	}
      out2.close();
    }
  
  inline float form_function(float AA,float DD, float delta) const
    {
      // formulas of the Yokoya, Chen's paper (p. 16)
      
      float C1, C2;
      float sqdy = sqrt(DD);
      float formfct;
      C1= 0.6+(sqdy-2.5)*(sqdy-2.5);
      C1 = 1 + 0.5/C1;
      C1 = (1+AA*AA)*C1*C1;
      
      C2 = 1.2*DD*DD/(DD+10);
      C2 = C2*C2;

      //   delta = 2.*dxx/sigmaxx;
      formfct = C1 + C2*delta*delta + delta*delta*delta*delta/(PI*PI);
      formfct = pow(formfct, (float)0.25);
      formfct = delta/formfct;
      return formfct;
    }
  
  // compute a special factor to determine an automatic grid size 
  // (see method automatic_transverse_cuts() )
  // the limits are chosen in a totally empirical way. May be refined
  inline float offset_factor(float hh) const
    {
      //	float test = dxx/sigmaxx;
      float factor;
      float x1 = 0.75;
      float x2 = 1.25;
      float x3 = 2.;
      if (hh > x3) 
	{
	  factor = 3. - (hh - x3)/(x3-x2) ;
	  if (factor <= 0.0) factor = 1.0;
	}
      else
	if (hh > x2 ) 
	  { 
	    factor = 4. - (hh - x2)/(x3 - x2);
	  }
	else
	  if (hh > x1 ) 
	    {
	      factor = 5. - (hh - x1)/(x2 - x1);
	    }
	  else 
	    {
	      factor = 6. - hh/x1;
	    }
      //  factor = 0.;
      factor = 6. - 3.*hh/x3;
      return factor;
    }
  
  // for method automatic_transverse_cuts()
  inline float size_due_to_cdm_deflection(float theta0, float sigmaz, float sigma, float beta, float DD, float dxx) const 
    {
      float AA;
      float factor, formfct;
      float decalage, hh;
      // beta is in mm, sigma in nanometres
      AA = sigmaz/(beta*1.0e6);
      //  std::cout << " A= " << AA << " beta = " << beta << std::endl;
      hh = dxx/sigma;
      formfct = form_function(AA, DD, 2.*hh);
      factor = offset_factor(hh);
      //  std::cout << " factor = " << factor << std::endl;
      decalage = 0.5*formfct*theta0*factor*sigmaz;
      return decalage;
    }
  

  inline float transverse_cut_from_data(char dir) const
    {
      float cut;
      std::string cut_dir("cut_");
      cut_dir += dir;
      cut = params_.readFValue(cut_dir);
      if (cut <= 0.0) 
	{ 
	  std::cerr << " FEEBL::transverse_cut_from_data() : WARNING : non automatic grid sizing, but cut_x or cut_y is not given. default values will be computed from sigma's " << std::endl;
	  cut = 3.0*sigma_for_cut_from_data(dir);
	}
      if (cut <= 0.0) 
	{
	  std::cerr << " FEEBL::transverse_cut_from_data() : ERROR : no cut is given in data, and thera are missing data for beta, emittance, sigma in the direction " << dir << ". So the default value cut = 3*sigma is not available. You should complete these data, or use the automatic_grid_sizing option " << std::endl;
	  exit(1);
	}
      return cut;
    }
  
  
  inline float cut_z_from_data(int automatic) const
    {
      float cutz;
      if (automatic == 1)
	{
	  cutz =  default_zcut(sigma_for_cut_from_data('z'));
	}
      else
	{
	  cutz = params_.readFValue("cut_z");
	  if(cutz>0.0) cutz *= 1e3;
	  else 
	    {
	      std::cerr << " FEEBL::cut_z_from_data() : WARNING : cut_z is not given. default values will be computed from sigma_z " << std::endl;
	      cutz =  default_zcut(sigma_for_cut_from_data('z'));
	    }
	}
      if (cutz <= 0.0) 
	{
	  std::cerr << " FEEBL::cut_z_from_data() :either  cut_z is not given in data, or sigma_z " << std::endl;
	  exit(1);
	}
      return cutz;
    }
  
  inline float default_zcut(float sigmaz ) const
    {
      
      // the sigmas are in nanometres
      // the factor 3.35, is an empirical value, valide for gaussian shape
      return 3.35*sigmaz;
    }
  
  
  // Crab and crab-waist maps describe incoming optics, applied once per
  // bunch before any lab-frame geometry. Placed at the head of this routine so
  // it covers both beam sources -- generated bunches and load_beam=3 -- and so
  // the cavity acts before the crossing-angle shear and the offsets, which is
  // the order the beamline has. A no-op unless a crab or waist mode is set.
  IncomingMapConfig configure_incoming_map(int b);
  void check_incoming_crossing_geometry(int b, const IncomingMapConfig& c);
  void check_incoming_tilt_resolution(
      int b,
      const INCOMING_MAP_CHAIN& chain,
      const BEAM_PARAMETERS& bp,
      float delta_z_nm);
  CANONICAL_BUNCH read_incoming_canonical_bunch(
      int b,
      const IncomingMapConfig& c,
      BEAM& beam,
      const BEAM_PARAMETERS& bp,
      int nsl,
      std::size_t n);
  void report_incoming_dispersion(
      int b,
      const INCOMING_MAP_CHAIN& chain,
      const BEAM_PARAMETERS& bp,
      const CANONICAL_BUNCH& bunch,
      std::size_t n);
  double write_incoming_canonical_bunch(
      int b,
      BEAM& beam,
      const CANONICAL_BUNCH& bunch,
      int nsl);
  void check_incoming_spin_contract(
      int b,
      const IncomingMapConfig& c,
      const BEAM_PARAMETERS& bp,
      double worst_dtheta);
  void apply_incoming_maps(float delta_z_nm);

  // Give every beam particle its energy spread, once, before anything else
  // looks at the beam. Must run BEFORE apply_incoming_maps: a dispersive shear
  // x += D*delta needs delta to exist. A no-op unless do_espread is set.
  void apply_energy_spread();

  // The crossing angle is applied as a per-slice position shear, so the sheared
  // bunch is a staircase of step angle_x*delta_z. Uncrabbed that is a
  // convergence question rather than a cancellation failure, so this warns
  // rather than refusing -- see the note on the definition.
  void warn_on_shear_quantisation(float delta_z_nm) const;

  inline void beam_displacements_from_data(float delta_z) 
    {
      apply_energy_spread();
      apply_incoming_maps(delta_z);
      if (switches.get_incoming_map_reslice())
        {
          for (int b=1; b<=2; ++b)
            {
              BEAM& beam=b==1 ? beam1_ : beam2_;
              const PARTICLE_BEAM::IncomingMapResliceReceipt r=
                  beam.particle_beam().reslice_after_incoming_map(delta_z);
              std::cout << "FEEBL_INCOMING_MAP_RESLICE_V1 beam=" << b
                        << " particles=" << r.particles << " moved=" << r.moved
                        << " moved_fraction=" << (r.particles ? double(r.moved)/r.particles : 0.0)
                        << " maximum_slice_shift=" << r.maximum_slice_shift
                        << " maximum_abs_z_nm=" << r.maximum_abs_z_nm
                        << " delta_z_nm=" << delta_z
                        << " dropped=0 population_renormalized=0" << std::endl;
            }
        }
      warn_on_shear_quantisation(delta_z);
      if ( switches.get_adjust() )
	{
	  beam1_.adjustToMeanPosition();
	  beam2_.adjustToMeanPosition();
	}    
      beam1_.rotate_particles();
      beam2_.rotate_particles();
      beam1_.set_angle_particles(delta_z);
      beam2_.set_angle_particles(delta_z);
      beam1_.set_particles_offset();
      beam2_.set_particles_offset();
    }
  
  void simulate();
  void set_simulation();
  void set_beams_and_grids();
  void set_output_data_and_files();

  // GPU mode 2 preflight occurs only after every grid has its final dimensions
  // and Green spectrum.  `fields` remains the host-phi ablation;
  // `primary` is a stricter resident-deposition/FFT/gather plan whose only
  // collision boundary is compact per-primary (Ex,Ey).
  void initialize_gpu2_fields();
  double deposit_gpu2_fields(int slice1, int slice2);
  void solve_gpu2_fields(int slice1, int slice2);
  std::string gpu2_fields_summary() const;
  void initialize_gpu4_wave_batch();
  void beam_interaction_gpu_mode4(PHI_FLOAT* sor_parameter);
  std::string gpu4_wave_summary() const;
  void initialize_wavefront_fields();
  void beam_interaction_gpu_mode3();
  
  void outputs(std::string nameOfProtokoll);
  
  //void  lumi_init();
  void lumi_exit();

  //  void store_photon2(PHOTON_BEAM *photon1,PHOTON_BEAM *photon2, float energy,float hel,float vx,float vy,float vz,float x, float y,float z);
  void  load_photon(PHOTON_BEAM *photon1,PHOTON_BEAM *photon2);
  void make_step(
      int i1, int i2, PHI_FLOAT* sor_parameter,
      const feebl::wavefront_fields::FieldPrefetch* prefetched_fields = NULL);
  
  void iteration_on_overlaping_slices(int firstSliceOfBeam1, int lastSliceOfBeam2, PHI_FLOAT* sor_parameter);
  
  void iteration_on_overlaping_slices_with_trackpair(PAIR_BEAM& pair_beam_ref,int firstSliceOfBeam1, int lastSliceOfBeam2,PHI_FLOAT* sor_parameter);
  void iteration_on_overlaping_slices_with_trackpair_muon(PAIR_BEAM& pair_beam_ref, PAIR_BEAM& muon_beam_ref, int firstSliceOfBeam1, int lastSliceOfBeam2,PHI_FLOAT* sor_parameter);

  struct LEGACY_WAVEFRONT_CONTEXT;
  static void wavefront_begin_diagonal(
      void* context,
      const feebl::legacy_wavefront::DiagonalDescriptor& diagonal);
  static void wavefront_begin_wave(
      void* context, const feebl::modern::EncounterWave& wave);
  static void wavefront_encounter(
      void* context, const feebl::modern::EncounterNode& encounter);
  static void wavefront_end_wave(
      void* context, const feebl::modern::EncounterWave& wave);
  static void wavefront_end_diagonal(
      void* context,
      const feebl::legacy_wavefront::DiagonalDescriptor& diagonal);
  void beam_interaction_wavefront_strict(PHI_FLOAT* sor_parameter);
  struct WAVEFRONT_FIELD_CONTEXT;
  static void wavefront_plan_field_range(
      void* context, std::size_t lane,
      std::size_t begin, std::size_t end);
  static void wavefront_prepare_field_range(
      void* context, std::size_t lane,
      std::size_t begin, std::size_t end);
  struct WAVEFRONT_PHYSICS_CONTEXT;
  static void wavefront_process_physics_range(
      void* context, std::size_t lane,
      std::size_t begin, std::size_t end);
  void publish_wavefront_physics_batch(
      std::size_t begin, std::size_t count);
  void publish_wavefront_lumi_heaps();
  void beam_interaction_wavefront_parallel(PHI_FLOAT* sor_parameter);
  
  inline void beam_interaction(PHI_FLOAT* sor_parameter )
    {
      if (switches.get_gpu_mode() == 4)
        {
          beam_interaction_gpu_mode4(sor_parameter);
          return;
        }
      if (switches.get_collision_schedule() == "wavefront_strict")
        {
          beam_interaction_wavefront_strict(sor_parameter);
          return;
        }
      if (switches.get_collision_schedule() == "wavefront_parallel")
        {
          beam_interaction_wavefront_parallel(sor_parameter);
          return;
        }
      int k;
      int n_slice=grid_.get_n_cell_z();
      for (k=0; k<n_slice; k++)
		{
		  iteration_on_overlaping_slices(0,k,sor_parameter);
		}
      // Maximum overlap: every slice of beam 1 is now against every slice of
      // beam 2, so this is where the beam-beam field, and any precession it
      // drives, is largest.
      polarization_.sample(POLARIZATION_REPORT::COLLISION, beam1_, beam2_);
      for (k = n_slice; k < 2 * n_slice - 1; k++)
		{
			iteration_on_overlaping_slices(k - n_slice + 1, n_slice - 1, sor_parameter);
		}
    }
  
  inline  void beam_interaction_with_trackpair(PAIR_BEAM& pair_beam_ref,PHI_FLOAT* sor_parameter )
    {
      int k;
      int n_slice=grid_.get_n_cell_z();
      // at the start of the collision
      for (k=0;k<n_slice;k++)
	{
	  iteration_on_overlaping_slices_with_trackpair(pair_beam_ref, 0, k,sor_parameter);
	}
      
      // Maximum overlap. Sampled here for the same reason as in
      // beam_interaction: this is where the beam-beam field, and any precession
      // it drives, is largest. This path used to omit it entirely, so a run
      // with track_pairs reported the collision state as "(not sampled)"
      // without saying why.
      polarization_.sample(POLARIZATION_REPORT::COLLISION, beam1_, beam2_);

      // at the end of the collision
      for (k=n_slice;k<2*n_slice-1;k++)
	{
	  iteration_on_overlaping_slices_with_trackpair(pair_beam_ref, k-n_slice+1,n_slice-1, sor_parameter);
	}
    }

  inline  void beam_interaction_with_trackpair_muon(PAIR_BEAM& pair_beam_ref, PAIR_BEAM& muon_beam_ref, PHI_FLOAT* sor_parameter )
    {
      int k;
      int n_slice=grid_.get_n_cell_z();
      // at the start of the collision
      for (k=0;k<n_slice;k++)
	{
	  iteration_on_overlaping_slices_with_trackpair_muon(pair_beam_ref, muon_beam_ref, 0, k,sor_parameter);
	}
      
      // Maximum overlap -- see beam_interaction_with_trackpair above.
      polarization_.sample(POLARIZATION_REPORT::COLLISION, beam1_, beam2_);

      // at the end of the collision
      for (k=n_slice;k<2*n_slice-1;k++)
	{
	  iteration_on_overlaping_slices_with_trackpair_muon(pair_beam_ref, muon_beam_ref, k-n_slice+1,n_slice-1, sor_parameter);
	}
    }

  inline void  make_time_step_on_slices(unsigned int firstSliceOfBeam1, unsigned int lastSliceOfBeam2, PHI_FLOAT* sor_parameter)
    {
      unsigned int slice_beam_1;
      int slice_beam_2 = lastSliceOfBeam2;
      grid_.check_distribute(1);
      for (slice_beam_1 = firstSliceOfBeam1; slice_beam_1 <= lastSliceOfBeam2; slice_beam_1++)
		{
		  make_step(slice_beam_1, slice_beam_2, sor_parameter);
		  slice_beam_2--;
		}
      grid_.check_distribute(2);
      if (switches.get_do_size_log())
		{
		  beam1_.write_size(firstSliceOfBeam1, lastSliceOfBeam2);
		  beam2_.write_size(firstSliceOfBeam1, lastSliceOfBeam2);
		}
    }
  
  void save_results_on_files();

  void emit_adaptive_root_replay_signal(
      bool publication_blocked, bool final_summary_outputs_published) const;
  
  void print_program_outputs(std::string nameOfProtokoll);
  
  // Read-only primary snapshots are published after committed diagonals.
  void write_gui_snapshot(int diagonal);
  void write_gui_manifest() const;
  std::string gui_secondary_frame_payload(int diagonal) const;
  void write_gui_secondary_frames() const;
  std::vector<std::string> gui_secondary_frames_;
  std::string gui_imported_unavailable_reason() const;
  std::string gui_imported_current() const;
  void write_gui_imported_trajectories() const;
  std::vector<std::string> gui_imported_frames_;
  void capture_gui_fields(int beam1_slice,int beam2_slice,bool host_phi_valid);
  void write_gui_fields() const;
  std::string gui_longitudinal_payload(int diagonal) const;
  std::string gui_snapshot_fields_payload(int diagonal);
  std::string gui_mesh_coverage_reason() const;
  void write_gui_mesh_coverage() const;
  std::vector<feebl::collider_fields::MeshEncounterCoverage> gui_mesh_encounters_;
  std::vector<feebl::collider_fields::MeshSnapshotCoverage> gui_mesh_snapshot_frames_;
  std::vector<int> gui_mesh_slice_visits_;

  std::unique_ptr<feebl::collider_fields::SnapshotGrid> gui_snapshot_grid_;
  std::vector<int> gui_longitudinal_visits_;
  std::vector<std::string> gui_longitudinal_strips_;
  std::vector<bool> gui_longitudinal_available_;
  bool gui_field_any_available_ = false;
  int gui_field_diagonal_ = -1;
  int gui_field_visits_ = 0;
  int gui_field_pending_diagonal_ = -1;
  std::string gui_field_pending_;
  std::vector<std::string> gui_field_frames_;
  std::string gui_diagnostics_payload(int first, int last,
      const RESULTS* previous,
      bool integrated = false) const;
  void write_gui_diagnostics() const;
  RESULTS gui_previous_results_;
  std::vector<std::string> gui_diagnostic_frames_;
  std::vector<int> gui_diagonals_;
  std::vector<int> gui_strides1_, gui_strides2_;

  inline void dump_beams(int timestep,int every_step, int every_particle)
    {
      
      //   if (timestep%every_step==0) 
      //     {
      std::ostringstream name;
      std::string filename;
      int extension = timestep/every_step;
      //
      name << std::string("bp1.") << extension << std::ends;
      filename = name.str();
      beam1_.dump_photons(filename,timestep,every_particle, grid_.get_timestep(), grid_.get_step(), grid_.get_max_z());
      name.seekp(0);
      name << std::string("bp2.") << extension << std::ends;
      filename = name.str();
      beam2_.dump_photons(filename,timestep, every_particle, grid_.get_timestep(), grid_.get_step(), grid_.get_max_z());
      name.seekp(0);
      //	
      name << std::string("b1.") << extension << std::ends;
      filename = name.str();
      
      beam1_.dump_beam(filename, timestep, every_particle, grid_.get_timestep(), grid_.get_step(), grid_.get_max_z());
      name.seekp(0);
      name << std::string("b2.") << extension << std::ends;
      filename = name.str();
      
      beam2_.dump_beam(filename, timestep, every_particle, grid_.get_timestep(), grid_.get_step(), grid_.get_max_z());
      //     }
    }
  
  // FEEBL: transverse sweep of the bunch centroid due to the crossing
  // angle, in nm; folded into the automatic cut_x.
  mutable float size_x_sweep_;
  int optimized_n_x_, optimized_n_y_;
  int optimized_n_z_, optimized_n_t_, optimized_grids_;

  // FEEBL: derive the whole grid/step configuration from the beams.
  void optimize_grid_settings();

  void printInitialBeam(const BEAM& beam);
  
  std::string header() const 
    {
      std::ostringstream out;
      out <<  "****************************************************** " << std::endl;
      out <<  "* FEEBL " << PACKAGE_VERSION << "  (Fast e+e- Beam Luminosity)" << std::endl;
      out <<  "*" << std::endl;
      out <<  "* Derived from GUINEA-PIG++ 1.2.2, used with the permission of" << std::endl;
      out <<  "* the original authors.  Physics reference: D. Schulte," << std::endl;
      out <<  "* TESLA-97-08 (DESY).  See ORIGIN.md." << std::endl;
      out <<  "*" << std::endl;
      out <<  "* Original program written by Daniel Schulte at DESY and CERN " << std::endl;
      out <<  "* object oriented by Guy Le Meur at LAL-Orsay " << std::endl;
      out <<  "* contributions from C. Rimbault at LAL-Orsay " << std::endl;
      out <<  "* B. Dalena, J. Esberg and J. Snuverink at CERN" << std::endl;
      out <<  "*" << std::endl;
      out <<  "* Compiled with FFTW: " 
#ifdef USE_FFT_LOCAL
	  << " FFT LOCAL (consider compiling with FFTW2 or FFTW3 to get cpu speedup) " << std::endl;
#elif defined USE_FFTW2
          << " FFTW2 " << std::endl;
#elif defined USE_FFTW3
	  << " FFTW3 " << std::endl;
#else
          << " UNKNOWN " << std::endl;
#endif
      out <<  "**************************************************** " << std::endl;

      return out.str();
    }
  // private default constructor - not implemented
  FEEBL();
 public:
  
  FEEBL(char *name);
  ~FEEBL()
    {
      unsigned int k;
      for (k=0; k < gpu2_field_solvers_.size(); k++)
        delete gpu2_field_solvers_[k];
      for (k=1; k < gridsPtr_.size(); k++) delete gridsPtr_[k];
      delete cross_;
    };
  void run(char *par,char *prot);
  //  void close();
  
  virtual std::string output_flow() const; 
  
};

#endif
