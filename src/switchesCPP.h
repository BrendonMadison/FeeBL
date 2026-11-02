#ifndef SWITCHES_SEEN
#define SWITCHES_SEEN

#include <cstdint>
#include <string>
#include <vector>

#include "define.h"
#include "abstractIOclass.h"
#include "parametersCPP.h"
#include "adaptiveMacroparticleCPP.h"

/** 
 * class that holds global parameters
 * most parameters are explained in the manual
 */

class SWITCHES : public ABSTRACT_IO_CLASS
{
  struct AdaptiveGridSizeClasses {
    std::vector<int> x,y;
  };
  AdaptiveGridSizeClasses check_adaptive_geometry_consistency() const;
  std::vector<int> check_adaptive_spacing_consistency() const;
  void check_adaptive_precision_consistency(const std::vector<int>& spacing_factors) const;
  void check_adaptive_patch_consistency(const AdaptiveGridSizeClasses& classes) const;
  void check_imported_spin_consistency() const;
  void check_adaptive_error_bounds() const;
  void check_adaptive_error_policy() const;
  void check_adaptive_source_bounds() const;
  void check_adaptive_operator_consistency() const;
  void check_adaptive_patch_bounds() const;
  void check_adaptive_source_consistency() const;
  void check_adaptive_capacity_consistency() const;
  void check_wavefront_dispatch_consistency() const;
  void check_spatial_luminosity_consistency() const;
  void check_precompute_luminosity_consistency() const;
  void check_factorized_photon_consistency() const;
  void check_private_wavefront_consistency() const;
  void check_secondary_output_consistency() const;
  void read_compute_options(const PARAMETERS& param);
  void read_adaptive_grid_options(const PARAMETERS& param);
  void read_adaptive_particle_options(const PARAMETERS& param);
  //  int silent;
  int electron_distribution_rho;
  //   int electron_distribution_scatter;
  int do_lumi,num_lumi,num_lumi_eg,num_lumi_gg;
  float lumi_p,lumi_p_eg,lumi_p_gg;
  float electron_ratio;

  // 2 dimensions was enough, but the original C had 3 dimensions
  // to be seen..
  int do_photons_[2],write_photons;
  int photon_distribution,do_beamstrahlung_;
  float photon_ratio;
  int do_hadrons,store_hadrons;
  float hadron_ratio;
  int do_jets;
  float jet_pstar;
  float jet_ratio;
  //  int do_pairs,track_secondaries,do_muons,store_secondaries;
  int do_pairs,track_pairs,store_pairs;
  int do_muons,track_muons,store_muons;
  int do_tertphot;
  float muon_ratio,muon_scale,muon_ecut;
  float pair_ratio;
  float bhabha_ratio;
  /// pair_ecut: minimal energy in GeV the particles from pair creation need to have to be stored
  /// pair_step: scaling factor for the step size of the pairs; if the value is increased the step size is decreased
  float pair_ecut,pair_step;
  int integration_method; /*!< integration method in transverse plane 1=direct, 2= fast Fourier (default), 3 = iterative */
  int extra_grids;
  int time_order;
  int interpolation;
  int adjust;
  /// ext_field, if not 0 the program takes into account the effect of the magnetic field for the equivalent photon approximation
  int geom,ext_field,beam_pair;
  int twobeam;
  float r_scal;
  int jet_pythia,jet_select,jet_store;
  int pair_q2;
  // incoherent-pair QED model: 0 legacy (default, bit-identical), 1 corrected equivalent-photon
  // model, 2 exact lowest-order kernels for the real x virtual and virtual x virtual processes
  int pair_model;
  // pair_model=2: Russian-roulette threshold on the expected pair count per photon pair (0: off)
  float pair_exact_rr;
  // time level of the beamstrahlung photons in the collisions: 0 legacy (binned one step ahead of the
  // particles and virtual photons), 1 synchronised (start-of-step position, half weight in the step of
  // birth), 2 diagnostic (start-of-step position, full weight in the step of birth), 3 time-centred (as 1,
  // and the emitter's mean direction over the step's kick)
  int photon_timing;
  // direction of beamstrahlung photons: 0 legacy (collinear with the emitter), 1 angle out of the orbit plane
  // from the LCFA spectrum, 2 as 1 only where the formation length is below half the opposing bunch length
  int photon_angle;
  // 1: beamstrahlung photons carry their polarisation (Stokes parameters), used by the Breit-Wheeler and
  // Bethe-Heitler pair and muon channels of pair_model >= 1 and by coherent pair creation; 0 (default):
  // unpolarised photons
  int photon_polarisation;
  // coherent pair creation: 0 (default) legacy sampling; 1 corrected sub-stepping, i.e. a step whose conversion
  // probability (after the factor 1.36) is >= 0.1 is sampled in sub-steps without the legacy extra gate
  // rndm() < tmp, which applied the probability twice for 0.1 <= tmp < 1
  int coherent_model;
  int load_photon,load_beam,load_event;
  int imported_spin_mode_=0;
  int cuts_from_loaded_beam;
  int store_native_;
  int loaded_beam_is_pretransformed_;
  int bmt_precession_;
  int ST_spin_flip_;
  int automatic_grid_sizing;
  int optimize_grids;
  int do_photon_lumi;
  // Strictly opt-in alternatives for the side-effect-free e-e luminosity
  // aggregate path. One is the ordered AoS invariant cache; two adds aligned
  // SoA/block evaluation and ordered scalar commits; three adds parallel pure
  // block evaluation and supports the e-e event heap. Zero preserves
  // GRID::collide_ee byte-for-byte.
  int electron_lumi_precompute_;
  // Precompute-3 event sampling policy: zero retains the historical shared RNG
  // heap; one selects a CounterRngV2 weighted reservoir by stable pair
  // identity; two selects sparse keyed Bernoulli
  // prepare_store(weight*lumi_p) sampling; three selects the full-wave-only
  // fused geometric-gap form of the same Bernoulli multiplicity law. Modes 2
  // and 3 use max(100,num_lumi) as a hard fail-closed capacity.
  int electron_lumi_sampling_;
  // Versioned physical-space e-e collision quadrature.  The raster is an
  // immutable luminosity estimator and is deliberately independent of the
  // mutable field/force mesh.  Zero retains the historical collision path;
  // one selects fixed-physical-CIC-v1 under the restricted capability
  // contract validated in SWITCHES::read.
  int electron_lumi_spatial_kernel_;
  int electron_lumi_spatial_nx_;
  int electron_lumi_spatial_ny_;
  double electron_lumi_spatial_x_min_;
  double electron_lumi_spatial_x_max_;
  double electron_lumi_spatial_y_min_;
  double electron_lumi_spatial_y_max_;
  std::uint64_t electron_lumi_spatial_max_pairs_;
  double electron_lumi_spatial_closure_tolerance_;
  // Strictly opt-in luminosity-only replacement for GRID::photon_lumi.
  // Zero is the historical Cartesian path; one is the audited factorization.
  int photon_lumi_factorized_;
  // Incoming canonical maps (crab crossing, crab waist).
  int crab_mode_[2];
  int crab_plane_[2];
  int crab_longitudinal_sign_[2];
  int waist_mode_[2];
  double crab_frequency_[2];
  double crab_phase_[2];
  double crab_target_tilt_[2];
  double waist_fraction_[2];
  double ip_dispersion_[2];
  int design_axis_spectrum_, design_axis_peak_bins_;
  int incoming_map_reslice_;
  /// dL/dW spectrum range and resolution. hi <= lo means derive the range from
  /// the beam energies, which is the historical behaviour.
  double spectrum_w_min_, spectrum_w_max_;
  /// Read ONLY so the dead-key warning can report what a deck asked for.
  /// Nothing else consumes these -- see the warning in feeblCPP.cc.
  int    hist_ee_bins_declared_;
  double hist_ee_max_declared_;
  int    spectrum_w_bins_;
  int crab_harmonic_n_;
  int crab_include_panofsky_wenzel_;
  double crab_full_crossing_angle_;
  double waist_symplecticity_tolerance_;
  /// charge_sign is the relative sign of the charge of the two beams -1 is for e+e- and +1 for e-e-. If set to 0 no beam-beam force is assumed
  float emin,charge_sign,charge_sign_0;
  // float charge_sign_2;
  int store_beam;
  int do_cross,do_cross_gg,do_isr,do_espread,do_prod;
  /// RMS value of energy spread of beam 1 and 2, distribution set with which_espread
  float espread1,espread2;
  /// energy spreading method, used in gridCPP.h
  /// 0: no energy spread 1: flat distribution 2: two peaks 3: Gaussian
  int which_espread1,which_espread2;
  /// if force_symmetric is not equal to 0, the beams are assumed to be up-down and left-right symmetric
  int force_symmetric,charge_symmetric;
  int do_bds_spin_rotation;
  int do_coherent,do_compt,do_compt_phot,do_trident;
  /// load and save random number generator state
  int rndm_load,rndm_save;
  /// initial random seed
  unsigned long rndm_seed;
  std::string compute_backend_;
  std::string collision_schedule_;
  // Opt-in execution of the complete admitted host-physics encounter body in
  // task-local wavefront workers, followed by deterministic publication.
  int wavefront_parallel_physics_;
  int sparse_cells_;
  std::string wavefront_field_solver_;
  // Total host workspace cap in bytes; zero selects the automatic cap.
  std::uint64_t wavefront_workspace_max_bytes_;
  int sparse_direct_interaction_limit_;
  int cpu_threads_;
  std::string cpu_variant_;
  int cpu_primary_move_;
  std::string gpu_variant_;
  int gpu_device_;
  int gpu_min_particles_;
  int gpu_mode_;
  std::string gpu2_scope_;
  std::string gpu2_deposition_;
  std::string gpu2_fft_;
  int gpu2_streams_;
  std::string gpu2_graphs_;
  std::string gpu2_fusion_;
  int gpu2_batch_slices_;
  std::string gpu2_pinned_io_;
  int gpu2_host_threads_;
  std::string gpu2_primary_mover_;
  double gpu2_memory_fraction_;
  std::string gpu2_profile_;
  std::string rng_backend_;
  std::string reduction_mode_;
  int allow_backend_fallback_;
  std::string engine_;
  std::string algorithm_;
  std::string precision_;
  std::string modern_reduction_;
  int modern_batch_width_;
  int modern_cpu_workers_;
  std::string modern_cpu_runtime_;
  std::string modern_cpu_affinity_;
  int modern_gpu_streams_;
  std::string modern_gpu_graphs_;
  std::string modern_trace_;
  // Adaptive transverse-field-grid input contract.  Mode zero is deliberately
  // inert.  Modes two and three have a validated configuration surface, but
  // remain execution-fail-closed until their solvers are integrated and
  // qualified.
  int adaptive_grid_mode_;
  std::string adaptive_grid_scope_;
  std::string adaptive_grid_size_classes_x_;
  std::string adaptive_grid_size_classes_y_;
  int adaptive_grid_min_nx_;
  int adaptive_grid_min_ny_;
  int adaptive_grid_max_nx_;
  int adaptive_grid_max_ny_;
  int adaptive_grid_extra_guard_cells_;
  int adaptive_grid_allow_full_window_;
  std::uint64_t adaptive_grid_max_cells_;
  std::uint64_t adaptive_grid_max_bytes_;
  std::string adaptive_grid_schedule_save_;
  std::string adaptive_grid_schedule_load_;
  std::string adaptive_grid_profile_;
  int adaptive_grid_pilot_diagnostics_;

  std::string adaptive_grid_field_spacing_classes_;
  std::string adaptive_grid_field_operator_;
  std::uint64_t adaptive_grid_max_refinement_factor_;
  double adaptive_grid_anchor_x_;
  double adaptive_grid_anchor_y_;
  std::string adaptive_grid_monitor_;
  double adaptive_grid_error_target_;
  std::string adaptive_grid_error_norm_policy_;
  // Prospective provenance for the physics kick/error budget that fixes all
  // mixed-V2 scales and gates.  Empty for every other error policy.
  std::string adaptive_grid_error_budget_sha256_;
  double adaptive_grid_error_quantile_;
  double adaptive_grid_error_catastrophe_limit_;
  double adaptive_grid_error_reference_field_;
  double adaptive_grid_error_absolute_floor_fraction_;
  double adaptive_grid_error_catastrophe_absolute_fraction_;
  double adaptive_grid_error_exposure_limit_;
  int adaptive_grid_noise_aware_stop_;
  std::string adaptive_grid_source_model_;
  std::uint64_t adaptive_grid_source_reference_nx_;
  std::uint64_t adaptive_grid_source_reference_ny_;
  double adaptive_grid_source_width_x_;
  double adaptive_grid_source_width_y_;
  double adaptive_grid_source_error_target_;
  double adaptive_grid_solver_error_target_;
  double adaptive_grid_lumi_error_target_;
  double adaptive_grid_lumi_floor_;
  double adaptive_grid_tau_curvature_;
  double adaptive_grid_tau_field_gradient_;
  double adaptive_grid_tau_richardson_;
  double adaptive_grid_tau_tail_;
  double adaptive_grid_encounter_tail_fraction_;
  double adaptive_grid_rho_floor_;
  double adaptive_grid_field_floor_;
  int adaptive_grid_indicator_smoothing_passes_;
  std::string adaptive_grid_indicator_reduction_;
  double adaptive_grid_refine_threshold_;
  double adaptive_grid_coarsen_threshold_;
  int adaptive_grid_shrink_dwell_;
  int adaptive_grid_update_interval_;
  int adaptive_grid_richardson_interval_;
  std::string adaptive_grid_richardson_order_source_;
  std::string adaptive_grid_richardson_order_table_;
  double adaptive_grid_min_neff_;
  int adaptive_grid_fail_on_unmet_error_;

  int adaptive_grid_max_levels_;
  int adaptive_grid_max_patches_;
  int adaptive_grid_refinement_ratio_;
  int adaptive_grid_blocking_factor_;
  int adaptive_grid_error_buffer_cells_;
  int adaptive_grid_transition_cells_;
  double adaptive_grid_fine_residual_tolerance_;
  int adaptive_grid_terminal_patch_certification_;
  // Coupled adaptive macroparticle configuration.  This remains inert at
  // mode zero and is presently production-admitted only for the disposable
  // projected field-source domain.
  feebl::adaptive_particles::Configuration
      adaptive_macroparticle_configuration_;
  std::string actual_compute_backend_;
  std::string actual_collision_schedule_;
  std::uint64_t collision_diagonals_;
  std::uint64_t collision_waves_;
  std::uint64_t collision_batches_;
  std::uint64_t collision_encounters_;
  std::uint64_t collision_parallel_physics_encounters_;
  std::uint64_t collision_maximum_wave_width_;
  std::uint64_t collision_prefetched_field_encounters_;
  std::uint64_t collision_parallel_field_encounters_;
  std::uint64_t collision_full_parallel_make_step_encounters_;
  std::uint64_t collision_maximum_nonempty_field_lanes_;
  int collision_field_workers_;
  double collision_field_setup_ms_;
  double collision_field_prefetch_ms_;
  double collision_ordered_commit_ms_;
  std::uint64_t collision_field_dense_clear_visits_;
  std::uint64_t collision_field_actual_clear_visits_;
  std::uint64_t collision_field_dense_dot_visits_;
  std::uint64_t collision_field_actual_dot_visits_;
  std::uint64_t collision_field_dispatch_batches_;
  std::uint64_t collision_workspace_bytes_each_estimate_;
  std::uint64_t collision_workspace_bytes_reserved_estimate_;
  int collision_backend_threads_;
  std::string actual_cpu_variant_;
  int actual_cpu_threads_;
  std::string backend_fallback_reason_;
  float gg_smin,compt_x_min,compt_emax;
  /* int do_lumi_ee_2,lumi_ee_2_n; */
  /* float lumi_ee_2_xmin,lumi_ee_2_xmax; */
  /* int do_lumi_eg_2,lumi_eg_2_n; */
  /* float lumi_eg_2_xmin,lumi_eg_2_xmax; */
  /* int do_lumi_ge_2,lumi_ge_2_n; */
  /* float lumi_ge_2_xmin,lumi_ge_2_xmax; */
  /* int do_lumi_gg_2,lumi_gg_2_n; */
  /* float lumi_gg_2_xmin,lumi_gg_2_xmax; */
  int do_bhabhas;
  float bhabha_scal, bhabha_ecmload,ecm_min;
  /// values related to do_prod, not implemented
  float prod_e,prod_scal;
  float compt_scale;
  //  int hist_ee_bins,hist_espec_bins;
  //  float hist_ee_min,hist_ee_max,hist_espec_min,hist_espec_max;
  int do_size_log;
  //  float beam_vx_min,beam_vx_max,beam_vy_min,beam_vy_max;
  /// rep rate and number of bunches (only used in some output results)
  double f_rep,n_b;
  int do_dump,dump_step,dump_particle;
  //  int beam_vx_interval,beam_vy_interval;
  int do_edm4hep;

  void check_consistency() const;

 public:

  SWITCHES();

  //  void readFirstBeamParameters(const PARAMETERS& param);
  // void readBeamParametersContinued(const PARAMETERS& param);
  void read(const PARAMETERS& param);
  void readTWOBEAM(const PARAMETERS& param);
  //  void readCharge_sign_2(const PARAMETERS& param);

  virtual  std::string output_flow() const;
  // One-line, machine-readable receipt for an accepted Mode-2/3 card.  It
  // records CPU integration but leaves actual execution pending runtime
  // backend/wavefront preflight; the later adaptive summary is authoritative.
  std::string adaptive_grid_configuration_flow() const;


  inline int getTWOBEAM() const {return twobeam;};
  inline int get_rndm_load() const {return rndm_load;};
  inline int get_rndm_save() const {return rndm_save;};
  inline unsigned long   get_rndm_seed() const {return rndm_seed;};
  inline const std::string& get_compute_backend() const {return compute_backend_;}
  inline const std::string& get_collision_schedule() const
  {return collision_schedule_;}
  inline int get_wavefront_parallel_physics() const
  {return wavefront_parallel_physics_;}
  inline int get_sparse_cells() const {return sparse_cells_;}
  inline const std::string& get_wavefront_field_solver() const
  {return wavefront_field_solver_;}
  inline std::uint64_t get_wavefront_workspace_max_bytes() const
  {return wavefront_workspace_max_bytes_;}
  inline int get_sparse_direct_interaction_limit() const
  {return sparse_direct_interaction_limit_;}
  inline int get_cpu_threads() const {return cpu_threads_;}
  inline const std::string& get_cpu_variant() const {return cpu_variant_;}
  inline int get_cpu_primary_move() const {return cpu_primary_move_;}
  inline const std::string& get_gpu_variant() const {return gpu_variant_;}
  inline int get_gpu_device() const {return gpu_device_;}
  inline int get_gpu_min_particles() const {return gpu_min_particles_;}
  inline int get_gpu_mode() const {return gpu_mode_;}
  inline const std::string& get_gpu2_scope() const {return gpu2_scope_;}
  inline const std::string& get_gpu2_deposition() const {return gpu2_deposition_;}
  inline const std::string& get_gpu2_fft() const {return gpu2_fft_;}
  inline int get_gpu2_streams() const {return gpu2_streams_;}
  inline const std::string& get_gpu2_graphs() const {return gpu2_graphs_;}
  inline const std::string& get_gpu2_fusion() const {return gpu2_fusion_;}
  inline int get_gpu2_batch_slices() const {return gpu2_batch_slices_;}
  inline const std::string& get_gpu2_pinned_io() const {return gpu2_pinned_io_;}
  inline int get_gpu2_host_threads() const {return gpu2_host_threads_;}
  inline const std::string& get_gpu2_primary_mover() const
  {return gpu2_primary_mover_;}
  inline double get_gpu2_memory_fraction() const {return gpu2_memory_fraction_;}
  inline const std::string& get_gpu2_profile() const {return gpu2_profile_;}
  inline const std::string& get_rng_backend() const {return rng_backend_;}
  inline const std::string& get_reduction_mode() const {return reduction_mode_;}
  inline int get_allow_backend_fallback() const {return allow_backend_fallback_;}
  inline const std::string& get_engine() const {return engine_;}
  inline const std::string& get_algorithm() const {return algorithm_;}
  inline const std::string& get_precision() const {return precision_;}
  inline const std::string& get_modern_reduction() const
  {return modern_reduction_;}
  inline int get_modern_batch_width() const {return modern_batch_width_;}
  inline int get_modern_cpu_workers() const {return modern_cpu_workers_;}
  inline const std::string& get_modern_cpu_runtime() const
  {return modern_cpu_runtime_;}
  inline const std::string& get_modern_cpu_affinity() const
  {return modern_cpu_affinity_;}
  inline int get_modern_gpu_streams() const {return modern_gpu_streams_;}
  inline const std::string& get_modern_gpu_graphs() const
  {return modern_gpu_graphs_;}
  inline const std::string& get_modern_trace() const {return modern_trace_;}
  inline int get_adaptive_grid_mode() const {return adaptive_grid_mode_;}
  inline const feebl::adaptive_particles::Configuration&
  get_adaptive_macroparticle_configuration() const
  {return adaptive_macroparticle_configuration_;}
  inline int get_adaptive_macroparticle_mode() const
  {return static_cast<int>(adaptive_macroparticle_configuration_.mode);}
  inline const std::string& get_adaptive_grid_scope() const
  {return adaptive_grid_scope_;}
  inline const std::string& get_adaptive_grid_size_classes_x() const
  {return adaptive_grid_size_classes_x_;}
  inline const std::string& get_adaptive_grid_size_classes_y() const
  {return adaptive_grid_size_classes_y_;}
  inline int get_adaptive_grid_min_nx() const {return adaptive_grid_min_nx_;}
  inline int get_adaptive_grid_min_ny() const {return adaptive_grid_min_ny_;}
  inline int get_adaptive_grid_max_nx() const {return adaptive_grid_max_nx_;}
  inline int get_adaptive_grid_max_ny() const {return adaptive_grid_max_ny_;}
  inline int get_adaptive_grid_extra_guard_cells() const
  {return adaptive_grid_extra_guard_cells_;}
  inline int get_adaptive_grid_allow_full_window() const
  {return adaptive_grid_allow_full_window_;}
  inline std::uint64_t get_adaptive_grid_max_cells() const
  {return adaptive_grid_max_cells_;}
  inline std::uint64_t get_adaptive_grid_max_bytes() const
  {return adaptive_grid_max_bytes_;}
  inline const std::string& get_adaptive_grid_schedule_save() const
  {return adaptive_grid_schedule_save_;}
  inline const std::string& get_adaptive_grid_schedule_load() const
  {return adaptive_grid_schedule_load_;}
  inline const std::string& get_adaptive_grid_profile() const
  {return adaptive_grid_profile_;}
  inline int get_adaptive_grid_pilot_diagnostics() const
  {return adaptive_grid_pilot_diagnostics_;}
  inline const std::string& get_adaptive_grid_field_spacing_classes() const
  {return adaptive_grid_field_spacing_classes_;}
  inline const std::string& get_adaptive_grid_field_operator() const
  {return adaptive_grid_field_operator_;}
  inline std::uint64_t get_adaptive_grid_max_refinement_factor() const
  {return adaptive_grid_max_refinement_factor_;}
  inline double get_adaptive_grid_anchor_x() const
  {return adaptive_grid_anchor_x_;}
  inline double get_adaptive_grid_anchor_y() const
  {return adaptive_grid_anchor_y_;}
  inline const std::string& get_adaptive_grid_monitor() const
  {return adaptive_grid_monitor_;}
  inline double get_adaptive_grid_error_target() const
  {return adaptive_grid_error_target_;}
  inline const std::string& get_adaptive_grid_error_norm_policy() const
  {return adaptive_grid_error_norm_policy_;}
  inline const std::string& get_adaptive_grid_error_budget_sha256() const
  {return adaptive_grid_error_budget_sha256_;}
  inline double get_adaptive_grid_error_quantile() const
  {return adaptive_grid_error_quantile_;}
  inline double get_adaptive_grid_error_catastrophe_limit() const
  {return adaptive_grid_error_catastrophe_limit_;}
  inline double get_adaptive_grid_error_reference_field() const
  {return adaptive_grid_error_reference_field_;}
  inline double get_adaptive_grid_error_absolute_floor_fraction() const
  {return adaptive_grid_error_absolute_floor_fraction_;}
  inline double get_adaptive_grid_error_catastrophe_absolute_fraction() const
  {return adaptive_grid_error_catastrophe_absolute_fraction_;}
  inline double get_adaptive_grid_error_exposure_limit() const
  {return adaptive_grid_error_exposure_limit_;}
  inline int get_adaptive_grid_noise_aware_stop() const
  {return adaptive_grid_noise_aware_stop_;}
  inline const std::string& get_adaptive_grid_source_model() const
  {return adaptive_grid_source_model_;}
  inline std::uint64_t get_adaptive_grid_source_reference_nx() const
  {return adaptive_grid_source_reference_nx_;}
  inline std::uint64_t get_adaptive_grid_source_reference_ny() const
  {return adaptive_grid_source_reference_ny_;}
  inline double get_adaptive_grid_source_width_x() const
  {return adaptive_grid_source_width_x_;}
  inline double get_adaptive_grid_source_width_y() const
  {return adaptive_grid_source_width_y_;}
  inline double get_adaptive_grid_source_error_target() const
  {return adaptive_grid_source_error_target_;}
  inline double get_adaptive_grid_solver_error_target() const
  {return adaptive_grid_solver_error_target_;}
  inline double get_adaptive_grid_lumi_error_target() const
  {return adaptive_grid_lumi_error_target_;}
  inline double get_adaptive_grid_lumi_floor() const
  {return adaptive_grid_lumi_floor_;}
  inline double get_adaptive_grid_tau_curvature() const
  {return adaptive_grid_tau_curvature_;}
  inline double get_adaptive_grid_tau_field_gradient() const
  {return adaptive_grid_tau_field_gradient_;}
  inline double get_adaptive_grid_tau_richardson() const
  {return adaptive_grid_tau_richardson_;}
  inline double get_adaptive_grid_tau_tail() const
  {return adaptive_grid_tau_tail_;}
  inline double get_adaptive_grid_encounter_tail_fraction() const
  {return adaptive_grid_encounter_tail_fraction_;}
  inline double get_adaptive_grid_rho_floor() const
  {return adaptive_grid_rho_floor_;}
  inline double get_adaptive_grid_field_floor() const
  {return adaptive_grid_field_floor_;}
  inline int get_adaptive_grid_indicator_smoothing_passes() const
  {return adaptive_grid_indicator_smoothing_passes_;}
  inline const std::string& get_adaptive_grid_indicator_reduction() const
  {return adaptive_grid_indicator_reduction_;}
  inline double get_adaptive_grid_refine_threshold() const
  {return adaptive_grid_refine_threshold_;}
  inline double get_adaptive_grid_coarsen_threshold() const
  {return adaptive_grid_coarsen_threshold_;}
  inline int get_adaptive_grid_shrink_dwell() const
  {return adaptive_grid_shrink_dwell_;}
  inline int get_adaptive_grid_update_interval() const
  {return adaptive_grid_update_interval_;}
  inline int get_adaptive_grid_richardson_interval() const
  {return adaptive_grid_richardson_interval_;}
  inline const std::string& get_adaptive_grid_richardson_order_source() const
  {return adaptive_grid_richardson_order_source_;}
  inline const std::string& get_adaptive_grid_richardson_order_table() const
  {return adaptive_grid_richardson_order_table_;}
  inline double get_adaptive_grid_min_neff() const
  {return adaptive_grid_min_neff_;}
  inline int get_adaptive_grid_fail_on_unmet_error() const
  {return adaptive_grid_fail_on_unmet_error_;}
  inline int get_adaptive_grid_max_levels() const
  {return adaptive_grid_max_levels_;}
  inline int get_adaptive_grid_max_patches() const
  {return adaptive_grid_max_patches_;}
  inline int get_adaptive_grid_refinement_ratio() const
  {return adaptive_grid_refinement_ratio_;}
  inline int get_adaptive_grid_blocking_factor() const
  {return adaptive_grid_blocking_factor_;}
  inline int get_adaptive_grid_error_buffer_cells() const
  {return adaptive_grid_error_buffer_cells_;}
  inline int get_adaptive_grid_transition_cells() const
  {return adaptive_grid_transition_cells_;}
  inline double get_adaptive_grid_fine_residual_tolerance() const
  {return adaptive_grid_fine_residual_tolerance_;}
  inline int get_adaptive_grid_terminal_patch_certification() const
  {return adaptive_grid_terminal_patch_certification_;}
  inline void set_compute_provenance(const std::string& backend,
                                     const std::string& variant,
                                     int threads,
                                     const std::string& fallback)
  {
    actual_compute_backend_ = backend;
    actual_cpu_variant_ = variant;
    actual_cpu_threads_ = threads;
    backend_fallback_reason_ = fallback;
  }
  inline void set_collision_schedule_provenance(
      const std::string& actual,
      std::uint64_t diagonals,
      std::uint64_t waves,
      std::uint64_t batches,
      std::uint64_t encounters,
      std::uint64_t parallel_physics_encounters,
      std::uint64_t maximum_wave_width)
  {
    actual_collision_schedule_ = actual;
    collision_diagonals_ = diagonals;
    collision_waves_ = waves;
    collision_batches_ = batches;
    collision_encounters_ = encounters;
    collision_parallel_physics_encounters_ = parallel_physics_encounters;
    collision_maximum_wave_width_ = maximum_wave_width;
  }
  inline void set_collision_field_provenance(
      std::uint64_t prefetched_field_encounters,
      std::uint64_t parallel_field_encounters,
      std::uint64_t full_parallel_make_step_encounters,
      std::uint64_t maximum_nonempty_field_lanes,
      int field_workers,
      double field_setup_ms,
      double field_prefetch_ms,
      double ordered_commit_ms,
      std::uint64_t dense_clear_visits,
      std::uint64_t actual_clear_visits,
      std::uint64_t dense_dot_visits,
      std::uint64_t actual_dot_visits,
      std::uint64_t field_dispatch_batches,
      std::uint64_t workspace_bytes_each_estimate,
      std::uint64_t workspace_bytes_reserved_estimate,
      int backend_threads)
  {
    collision_prefetched_field_encounters_ = prefetched_field_encounters;
    collision_parallel_field_encounters_ = parallel_field_encounters;
    collision_full_parallel_make_step_encounters_ =
        full_parallel_make_step_encounters;
    collision_maximum_nonempty_field_lanes_ = maximum_nonempty_field_lanes;
    collision_field_workers_ = field_workers;
    collision_field_setup_ms_ = field_setup_ms;
    collision_field_prefetch_ms_ = field_prefetch_ms;
    collision_ordered_commit_ms_ = ordered_commit_ms;
    collision_field_dense_clear_visits_ = dense_clear_visits;
    collision_field_actual_clear_visits_ = actual_clear_visits;
    collision_field_dense_dot_visits_ = dense_dot_visits;
    collision_field_actual_dot_visits_ = actual_dot_visits;
    collision_field_dispatch_batches_ = field_dispatch_batches;
    collision_workspace_bytes_each_estimate_ =
        workspace_bytes_each_estimate;
    collision_workspace_bytes_reserved_estimate_ =
        workspace_bytes_reserved_estimate;
    collision_backend_threads_ = backend_threads;
  }
  inline int get_write_photons() const {return write_photons;};
  inline int get_do_lumi() const {return do_lumi;};
  /* inline int get_do_lumi_ee_2() const {return do_lumi_ee_2;}; */
  /* inline int get_do_lumi_ge_2() const {return do_lumi_ge_2;}; */

  inline int get_do_bhabhas() const {return do_bhabhas;};

inline int get_do_edm4hep() const {return do_edm4hep;};


  /* inline int get_do_lumi_eg_2() const {return do_lumi_eg_2;}; */
  inline int get_num_lumi() const {return num_lumi;};
  inline int get_num_lumi_eg() const {return num_lumi_eg;};
  inline int get_num_lumi_gg() const {return num_lumi_gg;};
  inline int get_do_compt_phot() const {return do_compt_phot;};
  inline int get_do_size_log() const {return do_size_log;};
  inline int get_do_pairs() const {return do_pairs;};
  inline int get_do_hadrons() const {return do_hadrons;};
  inline int get_do_compt() const {return do_compt;};
  inline int get_do_muons() const {return do_muons;};
  inline int get_integration_method() const {return integration_method;};
  inline int get_extra_grids() const { return extra_grids;};
  inline int get_load_photon() const {return load_photon;};
  inline int get_load_beam() const {return load_beam;};
  inline int get_automatic_grid_sizing() const {return automatic_grid_sizing;}
  inline int get_optimize_grids() const {return optimize_grids;}
  inline int get_do_photon_lumi() const {return do_photon_lumi;}
  inline int get_electron_lumi_precompute() const
  {return electron_lumi_precompute_;}
  inline int get_electron_lumi_sampling() const
  {return electron_lumi_sampling_;}
  inline int get_electron_lumi_spatial_kernel() const
  {return electron_lumi_spatial_kernel_;}
  inline int get_electron_lumi_spatial_nx() const
  {return electron_lumi_spatial_nx_;}
  inline int get_electron_lumi_spatial_ny() const
  {return electron_lumi_spatial_ny_;}
  inline double get_electron_lumi_spatial_x_min() const
  {return electron_lumi_spatial_x_min_;}
  inline double get_electron_lumi_spatial_x_max() const
  {return electron_lumi_spatial_x_max_;}
  inline double get_electron_lumi_spatial_y_min() const
  {return electron_lumi_spatial_y_min_;}
  inline double get_electron_lumi_spatial_y_max() const
  {return electron_lumi_spatial_y_max_;}
  inline std::uint64_t get_electron_lumi_spatial_max_pairs() const
  {return electron_lumi_spatial_max_pairs_;}
  inline double get_electron_lumi_spatial_closure_tolerance() const
  {return electron_lumi_spatial_closure_tolerance_;}
  inline int get_photon_lumi_factorized() const
  {return photon_lumi_factorized_;}
  inline int get_crab_mode(int b) const {return crab_mode_[b-1];}
  inline int get_crab_plane(int b) const {return crab_plane_[b-1];}
  inline int get_crab_longitudinal_sign(int b) const {return crab_longitudinal_sign_[b-1];}
  inline int get_waist_mode(int b) const {return waist_mode_[b-1];}
  inline double get_crab_frequency(int b) const {return crab_frequency_[b-1];}
  inline double get_crab_phase(int b) const {return crab_phase_[b-1];}
  inline double get_crab_target_tilt(int b) const {return crab_target_tilt_[b-1];}
  inline double get_waist_fraction(int b) const {return waist_fraction_[b-1];}
  inline double get_ip_dispersion(int b) const {return ip_dispersion_[b-1];}
  inline int get_incoming_map_reslice()const{return incoming_map_reslice_;}
  inline int get_design_axis_spectrum()const{return design_axis_spectrum_;}
  inline int get_design_axis_peak_bins()const{return design_axis_peak_bins_;}
  inline double get_spectrum_w_min() const {return spectrum_w_min_;}
  inline double get_spectrum_w_max() const {return spectrum_w_max_;}
  inline int    get_spectrum_w_bins() const {return spectrum_w_bins_;}
  inline int    get_hist_ee_bins_declared() const {return hist_ee_bins_declared_;}
  inline double get_hist_ee_max_declared() const {return hist_ee_max_declared_;}
  inline int get_crab_harmonic_n() const {return crab_harmonic_n_;}
  inline int get_crab_include_panofsky_wenzel() const {return crab_include_panofsky_wenzel_;}
  inline double get_crab_full_crossing_angle() const {return crab_full_crossing_angle_;}
  inline double get_waist_symplecticity_tolerance() const {return waist_symplecticity_tolerance_;}
  inline void set_automatic_grid_sizing(int v) {automatic_grid_sizing = v;}
  inline int get_cuts_from_loaded_beam() const {return cuts_from_loaded_beam;}
  inline int get_store_native() const {return store_native_;}
  inline int get_loaded_beam_is_pretransformed() const {return loaded_beam_is_pretransformed_;}
  inline int get_bmt_precession() const {return bmt_precession_;}
  inline int get_ST_spin_flip() const {return ST_spin_flip_;}
  inline int get_charge_symmetric() const {return charge_symmetric;};
  inline int get_do_bds_spin_rotation() const {return do_bds_spin_rotation;};
  //  inline int get_electron_distribution_scatter() const { return electron_distribution_scatter;};
  inline int get_adjust() const {return adjust;};
  inline int get_time_order() const {return time_order;};
  //  inline int get_track_secondaries() const {return track_secondaries;};
  inline int get_track_pairs() const {return track_pairs;};
  inline int get_track_muons() const {return track_muons;};
  inline int get_do_tertphot() const {return do_tertphot;};
  inline int get_load_event() const {return load_event;};
    inline int get_imported_spin_mode() const {return imported_spin_mode_;}
  inline int get_electron_distribution_rho() const {return electron_distribution_rho;};
  inline int get_force_symmetric() const {return force_symmetric;};
  inline int get_geom() const {return geom;};
  inline int get_ext_field() const {return ext_field;};
  inline int get_do_espread() const { return do_espread;};
  inline int get_which_espread1() const { return which_espread1;};
  inline int get_which_espread2() const { return which_espread2;};
  inline int get_do_isr() const { return do_isr;};
  inline int get_do_cross() const {return do_cross;};
  inline int get_do_coherent() const {return do_coherent;};
  inline int get_do_trident() const {return do_trident;};
  inline int get_do_jets() const {return do_jets;};
  inline int get_do_prod() const {return do_prod;};
  inline int get_interpolation() const {return interpolation;};
  inline int get_do_beamstrahlung() const {return do_beamstrahlung_;};
  inline int get_do_photons(int nbeam) const {return do_photons_[nbeam-1];};
  inline int get_jet_pythia() const {return jet_pythia;};
  inline int get_jet_select() const {return jet_select;};
  inline int get_jet_store() const {return jet_store;};
  //  inline int get_store_secondaries() const {return store_secondaries;}
  inline int get_store_pairs() const {return store_pairs;}
  inline int get_store_muons() const {return store_muons;}
  /* inline int get_do_lumi_gg_2() const {return do_lumi_gg_2;} */
  inline int get_store_hadrons() const {return store_hadrons;}
  inline int get_pair_q2() const {return pair_q2;}
  inline int get_pair_model() const {return pair_model;}
  inline float get_pair_exact_rr() const {return pair_exact_rr;}
  inline int get_photon_timing() const {return photon_timing;}
  inline int get_photon_angle() const {return photon_angle;}
  inline int get_photon_polarisation() const {return photon_polarisation;}
  inline int get_coherent_model() const {return coherent_model;}
  inline int get_beam_pair() const {return beam_pair;}
  inline int get_store_beam() const { return store_beam;}
  inline int get_do_dump() const { return do_dump;}
  inline int get_dump_step() const { return dump_step;}
  inline int get_dump_particle() const { return dump_particle;}
  inline float get_lumi_p() const {return lumi_p;};
  inline float get_lumi_p_eg() const {return lumi_p_eg;};
  inline float get_lumi_p_gg() const {return lumi_p_gg;};
  inline float get_charge_sign() const {return charge_sign;};
  inline float get_charge_sign_0() const {return charge_sign_0;}
  inline float get_pair_ecut() const {return pair_ecut;};
  inline float get_muon_ecut() const {return muon_ecut;};
  inline float get_pair_step() const {return pair_step;};
  inline float get_electron_ratio() const {return electron_ratio;};
  inline float get_r_scal() const {return r_scal;};
  inline float get_compt_x_min() const {return compt_x_min;};
  inline float get_espread1() const { return espread1;};
  inline float get_espread2() const { return espread2;};
  inline float get_ecm_min() const {return ecm_min;};
  inline float get_emin() const {return emin;};
  inline float get_photon_ratio() const {return photon_ratio;};
  inline float get_jet_pstar() const {return jet_pstar;};
  inline float get_jet_ratio() const {return jet_ratio;};
  inline float get_pair_ratio() const {return pair_ratio;}
  inline float get_muon_ratio() const {return muon_ratio;}
  inline float get_muon_scale() const {return muon_scale;}
  inline float get_bhabha_ratio() const {return bhabha_ratio;}
  inline float get_compt_scale() const {return compt_scale;}
  inline float get_compt_emax() const {return compt_emax;}
  inline float get_gg_smin() const {return gg_smin;}
  inline float get_hadron_ratio() const {return hadron_ratio;}
  inline float get_bhabha_scal() const {return bhabha_scal;};
  inline float get_bhabha_ecmload() const {return bhabha_ecmload;};
  inline double get_f_rep() const {return f_rep;}
  inline double get_n_b() const {return n_b;}

};

#endif
