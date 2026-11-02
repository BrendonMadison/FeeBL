#include <iostream>
#include <cfloat>
#include "qedbg_real.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>
#include "switchesCPP.h"
#include "physconst.h"
#include "parallelBackendCPP.h"
#include "legacyPhotonLumiFastCPP.h"

namespace
{
bool adaptive_grid_is_finite(double value)
{
  // FeeBL is built with -ffast-math, under which std::isfinite may be folded
  // away.  Inspect the IEEE-754 exponent bits so malformed card values still
  // fail closed in optimized builds.
  static_assert(std::numeric_limits<double>::is_iec559,
                "adaptive-grid input validation requires IEEE-754 binary64");
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "adaptive-grid input validation requires 64-bit double");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

void adaptive_grid_input_error(const std::string& name,
                               const std::string& detail)
{
  std::cerr << "FEEBL: ERROR: " << name << " " << detail << std::endl;
  std::exit(2);
}

std::uint64_t read_adaptive_grid_capacity(double value,
                                          const std::string& name)
{
  // Input real values are represented as double.  Refuse integers above the
  // exact binary64 range instead of silently rounding a byte/cell cap.
  const double max_exact_integer = 9007199254740991.0; // 2^53 - 1
  if (!adaptive_grid_is_finite(value) || value < 0.0 ||
      std::floor(value) != value ||
      value > max_exact_integer)
    adaptive_grid_input_error(
        name, "must be a finite non-negative integer no greater than 2^53-1");
  return static_cast<std::uint64_t>(value);
}

bool is_power_of_two(int value)
{
  return value > 0 && (value & (value - 1)) == 0;
}

bool is_canonical_lowercase_sha256(const std::string& value)
{
  if (value.size() != 64U) return false;
  for (char character : value)
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f')))
      return false;
  return true;
}

void validate_adaptive_grid_dimension_limit(int value,
                                            const std::string& name)
{
  if (value != 0 && (value < 4 || (value & 1) != 0))
    adaptive_grid_input_error(
        name, "must be 0 (derive at preflight) or an even dimension at least 4");
}

std::vector<int> validate_adaptive_grid_size_classes(
    const std::string& text, const std::string& name, int minimum, int maximum)
{
  std::vector<int> result;
  if (text.empty()) return result;

  std::size_t begin = 0;
  long long previous = 0;
  while (begin <= text.size())
    {
      const std::size_t comma = text.find(',', begin);
      const std::string token = text.substr(
          begin, comma == std::string::npos ? std::string::npos : comma - begin);
      if (token.empty())
        adaptive_grid_input_error(
            name, "must be a comma-separated list without empty entries");

      std::istringstream parser(token);
      long long value = 0;
      char trailing = '\0';
      if (!(parser >> value) || (parser >> trailing))
        adaptive_grid_input_error(
            name, "contains a non-integer entry: '" + token + "'");
      if (value < 2 || value > std::numeric_limits<int>::max() ||
          (value & 1LL) != 0)
        adaptive_grid_input_error(
            name, "entries must be positive even integers representable as int");
      if (value <= previous)
        adaptive_grid_input_error(
            name, "entries must be unique and strictly increasing");
      if (minimum > 0 && value < minimum)
        adaptive_grid_input_error(
            name, "contains an entry below the configured minimum");
      if (maximum > 0 && value > maximum)
        adaptive_grid_input_error(
            name, "contains an entry above the configured maximum");
      previous = value;
      result.push_back(static_cast<int>(value));

      if (comma == std::string::npos) break;
      begin = comma + 1;
      if (begin == text.size())
        adaptive_grid_input_error(name, "must not end with a comma");
    }
  return result;
}

std::vector<int> validate_adaptive_grid_spacing_classes(
    const std::string& text)
{
  const std::string name = "adaptive_grid_field_spacing_classes";
  std::vector<int> result;
  if (text.empty()) return result;

  std::size_t begin = 0;
  int previous = 0;
  while (begin <= text.size())
    {
      const std::size_t comma = text.find(',', begin);
      const std::string token = text.substr(
          begin, comma == std::string::npos ? std::string::npos : comma - begin);
      if (token.empty())
        adaptive_grid_input_error(
            name, "must be a comma-separated list without empty entries");
      std::istringstream parser(token);
      long long factor = 0;
      char trailing = '\0';
      if (!(parser >> factor) || (parser >> trailing) || factor < 1 ||
          factor > std::numeric_limits<int>::max())
        adaptive_grid_input_error(
            name, "entries must be positive integer refinement factors");
      if (!is_power_of_two(static_cast<int>(factor)))
        adaptive_grid_input_error(
            name, "entries must be powers of two in the v1 class grammar");
      if (factor <= previous)
        adaptive_grid_input_error(
            name, "entries must be unique and strictly increasing");
      result.push_back(static_cast<int>(factor));
      previous = static_cast<int>(factor);
      if (comma == std::string::npos) break;
      begin = comma + 1;
      if (begin == text.size())
        adaptive_grid_input_error(name, "must not end with a comma");
    }
  if (result.front() != 1)
    adaptive_grid_input_error(
        name, "must begin with canonical spacing factor 1");
  return result;
}

void validate_adaptive_grid_order_table(
    const std::string& text, const std::vector<int>& spacing_factors)
{
  const std::string name = "adaptive_grid_richardson_order_table";
  if (text.empty())
    adaptive_grid_input_error(
        name, "is required when order source is calibrated_table");
  std::size_t begin = 0;
  std::size_t index = 0;
  while (begin <= text.size())
    {
      const std::size_t comma = text.find(',', begin);
      const std::string token = text.substr(
          begin, comma == std::string::npos ? std::string::npos : comma - begin);
      const std::size_t colon = token.find(':');
      if (colon == std::string::npos || colon == 0 ||
          colon + 1 == token.size() || token.find(':', colon + 1) != std::string::npos)
        adaptive_grid_input_error(
            name, "must use factor:observed_order entries");
      std::istringstream factor_parser(token.substr(0, colon));
      long long factor = 0;
      char trailing = '\0';
      if (!(factor_parser >> factor) || (factor_parser >> trailing) ||
          index >= spacing_factors.size() ||
          factor != spacing_factors[index])
        adaptive_grid_input_error(
            name, "factors must exactly match the spacing-class order");
      std::istringstream order_parser(token.substr(colon + 1));
      double order = 0.0;
      if (!(order_parser >> order) || (order_parser >> trailing) ||
          !adaptive_grid_is_finite(order) || order <= 0.0 || order > 16.0)
        adaptive_grid_input_error(
            name, "observed orders must be finite and in (0,16]");
      ++index;
      if (comma == std::string::npos) break;
      begin = comma + 1;
      if (begin == text.size())
        adaptive_grid_input_error(name, "must not end with a comma");
    }
  if (index != spacing_factors.size())
    adaptive_grid_input_error(
        name, "must contain exactly one entry per spacing class");
}

void require_positive_mode2_tolerance(double value, const char* name)
{
  if (!adaptive_grid_is_finite(value) || value <= 0.0 || value >= 1.0)
    adaptive_grid_input_error(
        name, "is required in modes 2/3 and must be finite in (0,1)");
}
}

SWITCHES::SWITCHES() 
{
  electron_distribution_rho=2;
  //    electron_distribution_scatter=1;
  electron_ratio=1.0;
  do_lumi=0;
  num_lumi=100000; num_lumi_eg=100000; num_lumi_gg=100000;
  do_cross=0;
  lumi_p=1e-29; lumi_p_eg=1e-29; lumi_p_gg=1e-29;
  do_photons_[0]=0;
  do_photons_[1]=0;
  write_photons=0; //=store_photons
  photon_distribution=1;
  photon_ratio=1.0;
  do_hadrons=0;
  store_hadrons=0;
  hadron_ratio=1e5;
  do_jets=0;
  jet_store=0; //=store_jets
  jet_pstar=2.0; //=jet_ptmin
  jet_ratio=1e5;
  do_pairs=0;
  load_event=0;
  //track_secondaries=0;
  track_pairs=0;
  track_muons=0;
  store_pairs=0;
  store_muons=0;
  do_tertphot=0;
  pair_ratio=1.0;
  muon_ratio=1.0;
  muon_scale=1.0;
  bhabha_ratio=1.0;
  pair_ecut=EMASS;
  muon_ecut=MUMASS;
  integration_method=2;
  extra_grids=0;
  time_order=2;
  interpolation=2;
  adjust=0;
  geom=1;
  r_scal=1.0;
  jet_pythia=0;
  jet_select=1;
  pair_q2=1;
  pair_model=0;
  pair_exact_rr=0.25;
  photon_timing=0;
  photon_angle=0;
  photon_polarisation=0;
  coherent_model=0;
  load_photon=0;
  load_beam=0;
  cuts_from_loaded_beam = 0;
  bmt_precession_ = 0;
  ST_spin_flip_ = 0;
  automatic_grid_sizing = 0;
  optimize_grids = 0;
  do_photon_lumi = 1;
  electron_lumi_precompute_ = 0;
  electron_lumi_sampling_ = 0;
  electron_lumi_spatial_kernel_ = 0;
  electron_lumi_spatial_nx_ = 0;
  electron_lumi_spatial_ny_ = 0;
  electron_lumi_spatial_x_min_ = 0.0;
  electron_lumi_spatial_x_max_ = 0.0;
  electron_lumi_spatial_y_min_ = 0.0;
  electron_lumi_spatial_y_max_ = 0.0;
  electron_lumi_spatial_max_pairs_ = 0U;
  electron_lumi_spatial_closure_tolerance_ = 1.0e-12;
  photon_lumi_factorized_ = 0;
  store_native_ = 0;
  loaded_beam_is_pretransformed_ = 0;
  crab_mode_[0] = crab_mode_[1] = 0;
  crab_plane_[0] = crab_plane_[1] = 0;
  crab_longitudinal_sign_[0] = crab_longitudinal_sign_[1] = 0;
  waist_mode_[0] = waist_mode_[1] = 0;
  crab_frequency_[0] = crab_frequency_[1] = 0.0;
  crab_phase_[0] = crab_phase_[1] = 0.0;
  crab_target_tilt_[0] = crab_target_tilt_[1] = -1.0;
  waist_fraction_[0] = waist_fraction_[1] = 1.0;
  ip_dispersion_[0] = ip_dispersion_[1] = 0.0;
  design_axis_spectrum_=0;design_axis_peak_bins_=10000;
  incoming_map_reslice_=0;
  spectrum_w_min_ = 0.0; spectrum_w_max_ = 0.0; spectrum_w_bins_ = 200;
  hist_ee_bins_declared_ = 200; hist_ee_max_declared_ = -1.0;
  crab_harmonic_n_ = 2;
  crab_include_panofsky_wenzel_ = 1;
  crab_full_crossing_angle_ = 0.0;
  waist_symplecticity_tolerance_ = 1e-11;
  emin=1.0;
  charge_sign=-1.0; charge_sign_0=-1.0;
  do_beamstrahlung_=1;
  store_beam=0;
  do_cross_gg=0;
  force_symmetric=0;
  do_isr=0;
  do_espread=0;
  espread1=0.0; espread2=0.0; which_espread1=0; which_espread2=0;
  do_coherent=0;
  do_trident=0;
  gg_smin=4.0*150.0*150.0;
  twobeam=0;

  do_edm4hep=0;

  // default values from readData.cc
  beam_pair=0;
  do_bhabhas=0;
  bhabha_ecmload=500;
  bhabha_scal=1.e-29;

  charge_symmetric=0; // to be checked
  do_bds_spin_rotation=0;

  do_compt=0;
  do_compt_phot=0;
  compt_emax=100;
  compt_scale=1.0;
  compt_x_min=1.0;
  
  do_dump=0;
  do_muons=0;
  // do_prod is not implemented
  do_prod=0; prod_e=0.0; prod_scal=1e-29;
  do_size_log=0;

  dump_particle=1;
  dump_step=1;
  ecm_min=0.0;
  ext_field=0;
  pair_step=1.0;
  // CLIC defaults:
  f_rep = 50; n_b=312;

  rndm_load=1; rndm_save=1; 
  rndm_seed=1;
  compute_backend_ = "serial";
  collision_schedule_ = "legacy";
  wavefront_parallel_physics_ = 0;
  sparse_cells_ = 0;
  wavefront_field_solver_ = "fft";
  wavefront_workspace_max_bytes_ = 0;
  sparse_direct_interaction_limit_ = 262144;
  cpu_threads_ = 0;
  cpu_variant_ = "auto";
  cpu_primary_move_ = 0;
  gpu_variant_ = "auto";
  gpu_device_ = 0;
  gpu_min_particles_ = 262144;
  gpu_mode_ = 1;
  gpu2_scope_ = "auto";
  gpu2_deposition_ = "auto";
  gpu2_fft_ = "cufft";
  gpu2_streams_ = 0;
  gpu2_graphs_ = "auto";
  gpu2_fusion_ = "auto";
  gpu2_batch_slices_ = 0;
  gpu2_pinned_io_ = "auto";
  gpu2_host_threads_ = 0;
  gpu2_primary_mover_ = "serial";
  gpu2_memory_fraction_ = 0.80;
  gpu2_profile_ = "summary";
  rng_backend_ = "legacy";
  reduction_mode_ = "strict";
  allow_backend_fallback_ = 0;
  engine_ = "legacy";
  algorithm_ = "exact";
  precision_ = "fp64";
  modern_reduction_ = "deterministic";
  modern_batch_width_ = 0;
  modern_cpu_workers_ = 0;
  modern_cpu_runtime_ = "auto";
  modern_cpu_affinity_ = "auto";
  modern_gpu_streams_ = 0;
  modern_gpu_graphs_ = "auto";
  modern_trace_ = "off";
  adaptive_grid_mode_ = 0;
  adaptive_grid_scope_ = "encounter";
  adaptive_grid_size_classes_x_.clear();
  adaptive_grid_size_classes_y_.clear();
  adaptive_grid_min_nx_ = 0;
  adaptive_grid_min_ny_ = 0;
  adaptive_grid_max_nx_ = 0;
  adaptive_grid_max_ny_ = 0;
  adaptive_grid_extra_guard_cells_ = 0;
  adaptive_grid_allow_full_window_ = 1;
  adaptive_grid_max_cells_ = 0;
  adaptive_grid_max_bytes_ = 0;
  adaptive_grid_schedule_save_.clear();
  adaptive_grid_schedule_load_.clear();
  adaptive_grid_profile_ = "summary";
  adaptive_grid_pilot_diagnostics_ = 0;
  adaptive_grid_field_spacing_classes_.clear();
  adaptive_grid_field_operator_ = "legacy_cic12_v1";
  adaptive_grid_max_refinement_factor_ = 0U;
  adaptive_grid_anchor_x_ = 0.0;
  adaptive_grid_anchor_y_ = 0.0;
  adaptive_grid_monitor_ = "charge_curvature_field_richardson_overlap";
  adaptive_grid_error_target_ = 0.0;
  adaptive_grid_error_norm_policy_ = "component_rms_linf_v1";
  adaptive_grid_error_budget_sha256_.clear();
  adaptive_grid_error_quantile_ = 0.999;
  adaptive_grid_error_catastrophe_limit_ = 0.5;
  adaptive_grid_error_reference_field_ = 0.0;
  adaptive_grid_error_absolute_floor_fraction_ = 0.0;
  adaptive_grid_error_catastrophe_absolute_fraction_ = 0.0;
  adaptive_grid_error_exposure_limit_ = 0.0;
  adaptive_grid_noise_aware_stop_ = 0;
  adaptive_grid_source_model_.clear();
  adaptive_grid_source_reference_nx_ = 0U;
  adaptive_grid_source_reference_ny_ = 0U;
  adaptive_grid_source_width_x_ = 0.0;
  adaptive_grid_source_width_y_ = 0.0;
  adaptive_grid_source_error_target_ = 0.0;
  adaptive_grid_solver_error_target_ = 0.0;
  adaptive_grid_lumi_error_target_ = 0.0;
  adaptive_grid_lumi_floor_ = 0.0;
  adaptive_grid_tau_curvature_ = 0.0;
  adaptive_grid_tau_field_gradient_ = 0.0;
  adaptive_grid_tau_richardson_ = 0.0;
  adaptive_grid_tau_tail_ = 0.0;
  adaptive_grid_encounter_tail_fraction_ = 0.0;
  adaptive_grid_rho_floor_ = 0.0;
  adaptive_grid_field_floor_ = 0.0;
  adaptive_grid_indicator_smoothing_passes_ = 0;
  adaptive_grid_indicator_reduction_ = "connected_max";
  adaptive_grid_refine_threshold_ = 1.0;
  adaptive_grid_coarsen_threshold_ = 0.35;
  adaptive_grid_shrink_dwell_ = 2;
  adaptive_grid_update_interval_ = 1;
  adaptive_grid_richardson_interval_ = 0;
  adaptive_grid_richardson_order_source_ = "calibrated_table";
  adaptive_grid_richardson_order_table_.clear();
  adaptive_grid_min_neff_ = 0.0;
  adaptive_grid_fail_on_unmet_error_ = 1;
  adaptive_grid_max_levels_ = 1;
  adaptive_grid_max_patches_ = 1;
  adaptive_grid_refinement_ratio_ = 2;
  adaptive_grid_blocking_factor_ = 0;
  adaptive_grid_error_buffer_cells_ = 0;
  adaptive_grid_transition_cells_ = 4;
  adaptive_grid_fine_residual_tolerance_ = 0.0;
  adaptive_grid_terminal_patch_certification_ = 0;
  actual_compute_backend_ = "serial";
  actual_collision_schedule_ = "legacy";
  collision_diagonals_ = 0;
  collision_waves_ = 0;
  collision_batches_ = 0;
  collision_encounters_ = 0;
  collision_parallel_physics_encounters_ = 0;
  collision_maximum_wave_width_ = 0;
  collision_prefetched_field_encounters_ = 0;
  collision_parallel_field_encounters_ = 0;
  collision_full_parallel_make_step_encounters_ = 0;
  collision_maximum_nonempty_field_lanes_ = 0;
  collision_field_workers_ = 0;
  collision_field_setup_ms_ = 0.0;
  collision_field_prefetch_ms_ = 0.0;
  collision_ordered_commit_ms_ = 0.0;
  collision_field_dense_clear_visits_ = 0;
  collision_field_actual_clear_visits_ = 0;
  collision_field_dense_dot_visits_ = 0;
  collision_field_actual_dot_visits_ = 0;
  collision_field_dispatch_batches_ = 0;
  collision_workspace_bytes_each_estimate_ = 0;
  collision_workspace_bytes_reserved_estimate_ = 0;
  collision_backend_threads_ = 0;
  actual_cpu_variant_ = "serial";
  actual_cpu_threads_ = 1;
}


// void SWITCHES::readFirstBeamParameters(const PARAMETERS& param)
// {
//    n_b= param.readDValue("n_b");
//    f_rep=param.readDValue("f_rep");
//    charge_sign=param.readFValue("charge_sign");

//   if(charge_sign>0.0){
//     charge_sign_0=1.0;
//   }
//   if(charge_sign<0.0){
//     charge_sign_0=-1.0;
//   }

// }

// Read execution selectors; capability checks are performed after all families.
void SWITCHES::read_compute_options(const PARAMETERS& param)
{
  compute_backend_ = param.readSValue("compute_backend");
  collision_schedule_ = param.readSValue("collision_schedule");
  wavefront_parallel_physics_ =
      param.readIValue("wavefront_parallel_physics");
  sparse_cells_ = param.readIValue("sparse_cells");
  wavefront_field_solver_ = param.readSValue("wavefront_field_solver");
  wavefront_workspace_max_bytes_ = read_adaptive_grid_capacity(
      param.readDValue("wavefront_workspace_max_bytes"),
      "wavefront_workspace_max_bytes");
  sparse_direct_interaction_limit_ =
      param.readIValue("sparse_direct_interaction_limit");
  cpu_threads_ = param.readIValue("cpu_threads");
  cpu_variant_ = param.readSValue("cpu_variant");
  cpu_primary_move_ = param.readIValue("cpu_primary_move");
  gpu_variant_ = param.readSValue("gpu_variant");
  gpu_device_ = param.readIValue("gpu_device");
  gpu_min_particles_ = param.readIValue("gpu_min_particles");
  gpu_mode_ = param.readIValue("gpu_mode");
  gpu2_scope_ = param.readSValue("gpu2_scope");
  gpu2_deposition_ = param.readSValue("gpu2_deposition");
  gpu2_fft_ = param.readSValue("gpu2_fft");
  gpu2_streams_ = param.readIValue("gpu2_streams");
  gpu2_graphs_ = param.readSValue("gpu2_graphs");
  gpu2_fusion_ = param.readSValue("gpu2_fusion");
  gpu2_batch_slices_ = param.readIValue("gpu2_batch_slices");
  gpu2_pinned_io_ = param.readSValue("gpu2_pinned_io");
  gpu2_host_threads_ = param.readIValue("gpu2_host_threads");
  gpu2_primary_mover_ = param.readSValue("gpu2_primary_mover");
  gpu2_memory_fraction_ = param.readDValue("gpu2_memory_fraction");
  gpu2_profile_ = param.readSValue("gpu2_profile");
  rng_backend_ = param.readSValue("rng_backend");
  reduction_mode_ = param.readSValue("reduction_mode");
  allow_backend_fallback_ = param.readIValue("allow_backend_fallback");
  engine_ = param.readSValue("engine");
  algorithm_ = param.readSValue("algorithm");
  precision_ = param.readSValue("precision");
  modern_reduction_ = param.readSValue("modern_reduction");
  modern_batch_width_ = param.readIValue("modern_batch_width");
  modern_cpu_workers_ = param.readIValue("modern_cpu_workers");
  modern_cpu_runtime_ = param.readSValue("modern_cpu_runtime");
  modern_cpu_affinity_ = param.readSValue("modern_cpu_affinity");
  modern_gpu_streams_ = param.readIValue("modern_gpu_streams");
  modern_gpu_graphs_ = param.readSValue("modern_gpu_graphs");
  modern_trace_ = param.readSValue("modern_trace");
}

// Read grid error budgets in the native field/length conventions.
void SWITCHES::read_adaptive_grid_options(const PARAMETERS& param)
{
  adaptive_grid_mode_ = param.readIValue("adaptive_grid_mode");
  adaptive_grid_scope_ = param.readSValue("adaptive_grid_scope");
  adaptive_grid_size_classes_x_ =
      param.readSValue("adaptive_grid_size_classes_x");
  adaptive_grid_size_classes_y_ =
      param.readSValue("adaptive_grid_size_classes_y");
  adaptive_grid_min_nx_ = param.readIValue("adaptive_grid_min_nx");
  adaptive_grid_min_ny_ = param.readIValue("adaptive_grid_min_ny");
  adaptive_grid_max_nx_ = param.readIValue("adaptive_grid_max_nx");
  adaptive_grid_max_ny_ = param.readIValue("adaptive_grid_max_ny");
  adaptive_grid_extra_guard_cells_ =
      param.readIValue("adaptive_grid_extra_guard_cells");
  adaptive_grid_allow_full_window_ =
      param.readIValue("adaptive_grid_allow_full_window");
  adaptive_grid_max_cells_ = read_adaptive_grid_capacity(
      param.readDValue("adaptive_grid_max_cells"),
      "adaptive_grid_max_cells");
  adaptive_grid_max_bytes_ = read_adaptive_grid_capacity(
      param.readDValue("adaptive_grid_max_bytes"),
      "adaptive_grid_max_bytes");
  adaptive_grid_schedule_save_ =
      param.readSValue("adaptive_grid_schedule_save");
  adaptive_grid_schedule_load_ =
      param.readSValue("adaptive_grid_schedule_load");
  adaptive_grid_profile_ = param.readSValue("adaptive_grid_profile");
  adaptive_grid_pilot_diagnostics_ =
      param.readIValue("adaptive_grid_pilot_diagnostics");

  adaptive_grid_field_spacing_classes_ =
      param.readSValue("adaptive_grid_field_spacing_classes");
  adaptive_grid_field_operator_ =
      param.readSValue("adaptive_grid_field_operator");
  adaptive_grid_max_refinement_factor_ = read_adaptive_grid_capacity(
      param.readDValue("adaptive_grid_max_refinement_factor"),
      "adaptive_grid_max_refinement_factor");
  adaptive_grid_anchor_x_ = param.readDValue("adaptive_grid_anchor_x");
  adaptive_grid_anchor_y_ = param.readDValue("adaptive_grid_anchor_y");
  adaptive_grid_monitor_ = param.readSValue("adaptive_grid_monitor");
  adaptive_grid_error_target_ =
      param.readDValue("adaptive_grid_error_target");
  adaptive_grid_error_norm_policy_ =
      param.readSValue("adaptive_grid_error_norm_policy");
  adaptive_grid_error_budget_sha256_ =
      param.readSValue("adaptive_grid_error_budget_sha256");
  adaptive_grid_error_quantile_ =
      param.readDValue("adaptive_grid_error_quantile");
  adaptive_grid_error_catastrophe_limit_ =
      param.readDValue("adaptive_grid_error_catastrophe_limit");
  adaptive_grid_error_reference_field_ =
      param.readDValue("adaptive_grid_error_reference_field");
  adaptive_grid_error_absolute_floor_fraction_ =
      param.readDValue("adaptive_grid_error_absolute_floor_fraction");
  adaptive_grid_error_catastrophe_absolute_fraction_ =
      param.readDValue("adaptive_grid_error_catastrophe_absolute_fraction");
  adaptive_grid_error_exposure_limit_ =
      param.readDValue("adaptive_grid_error_exposure_limit");
  adaptive_grid_noise_aware_stop_ =
      param.readIValue("adaptive_grid_noise_aware_stop");
  adaptive_grid_source_model_ =
      param.readSValue("adaptive_grid_source_model");
  adaptive_grid_source_reference_nx_ = read_adaptive_grid_capacity(
      param.readDValue("adaptive_grid_source_reference_nx"),
      "adaptive_grid_source_reference_nx");
  adaptive_grid_source_reference_ny_ = read_adaptive_grid_capacity(
      param.readDValue("adaptive_grid_source_reference_ny"),
      "adaptive_grid_source_reference_ny");
  adaptive_grid_source_width_x_ =
      param.readDValue("adaptive_grid_source_width_x");
  adaptive_grid_source_width_y_ =
      param.readDValue("adaptive_grid_source_width_y");
  adaptive_grid_source_error_target_ =
      param.readDValue("adaptive_grid_source_error_target");
  adaptive_grid_solver_error_target_ =
      param.readDValue("adaptive_grid_solver_error_target");
  adaptive_grid_lumi_error_target_ =
      param.readDValue("adaptive_grid_lumi_error_target");
  adaptive_grid_lumi_floor_ =
      param.readDValue("adaptive_grid_lumi_floor");
  adaptive_grid_tau_curvature_ =
      param.readDValue("adaptive_grid_tau_curvature");
  adaptive_grid_tau_field_gradient_ =
      param.readDValue("adaptive_grid_tau_field_gradient");
  adaptive_grid_tau_richardson_ =
      param.readDValue("adaptive_grid_tau_richardson");
  adaptive_grid_tau_tail_ = param.readDValue("adaptive_grid_tau_tail");
  adaptive_grid_encounter_tail_fraction_ =
      param.readDValue("adaptive_grid_encounter_tail_fraction");
  adaptive_grid_rho_floor_ = param.readDValue("adaptive_grid_rho_floor");
  adaptive_grid_field_floor_ =
      param.readDValue("adaptive_grid_field_floor");
  adaptive_grid_indicator_smoothing_passes_ =
      param.readIValue("adaptive_grid_indicator_smoothing_passes");
  adaptive_grid_indicator_reduction_ =
      param.readSValue("adaptive_grid_indicator_reduction");
  adaptive_grid_refine_threshold_ =
      param.readDValue("adaptive_grid_refine_threshold");
  adaptive_grid_coarsen_threshold_ =
      param.readDValue("adaptive_grid_coarsen_threshold");
  adaptive_grid_shrink_dwell_ =
      param.readIValue("adaptive_grid_shrink_dwell");
  adaptive_grid_update_interval_ =
      param.readIValue("adaptive_grid_update_interval");
  adaptive_grid_richardson_interval_ =
      param.readIValue("adaptive_grid_richardson_interval");
  adaptive_grid_richardson_order_source_ =
      param.readSValue("adaptive_grid_richardson_order_source");
  adaptive_grid_richardson_order_table_ =
      param.readSValue("adaptive_grid_richardson_order_table");
  adaptive_grid_min_neff_ = param.readDValue("adaptive_grid_min_neff");
  adaptive_grid_fail_on_unmet_error_ =
      param.readIValue("adaptive_grid_fail_on_unmet_error");

  adaptive_grid_max_levels_ =
      param.readIValue("adaptive_grid_max_levels");
  adaptive_grid_max_patches_ =
      param.readIValue("adaptive_grid_max_patches");
  adaptive_grid_refinement_ratio_ =
      param.readIValue("adaptive_grid_refinement_ratio");
  adaptive_grid_blocking_factor_ =
      param.readIValue("adaptive_grid_blocking_factor");
  adaptive_grid_error_buffer_cells_ =
      param.readIValue("adaptive_grid_error_buffer_cells");
  adaptive_grid_transition_cells_ =
      param.readIValue("adaptive_grid_transition_cells");
  adaptive_grid_fine_residual_tolerance_ =
      param.readDValue("adaptive_grid_fine_residual_tolerance");
  adaptive_grid_terminal_patch_certification_ =
      param.readIValue("adaptive_grid_terminal_patch_certification");

}

// Read representative-source controls and validate their typed configuration.
// This cannot change the physical primary population.
void SWITCHES::read_adaptive_particle_options(const PARAMETERS& param)
{
  {
    feebl::adaptive_particles::Configuration& adaptive =
        adaptive_macroparticle_configuration_;
    const int mode = param.readIValue("adaptive_macroparticle_mode");
    if (mode < 0 || mode > 3)
      adaptive_grid_input_error(
          "adaptive_macroparticle_mode", "must be 0, 1, 2, or 3");
    adaptive.mode = static_cast<feebl::adaptive_particles::Mode>(mode);
    adaptive.domain = param.readSValue("adaptive_macroparticle_domain");
    adaptive.source_evolution =
        param.readSValue("adaptive_macroparticle_source_evolution");
    adaptive.decision_scope =
        param.readSValue("adaptive_macroparticle_decision_scope");
    adaptive.commit_scope =
        param.readSValue("adaptive_macroparticle_commit_scope");
    adaptive.strategy =
        param.readSValue("adaptive_macroparticle_strategy");
    adaptive.split_algorithm =
        param.readSValue("adaptive_macroparticle_split");
    adaptive.merge_algorithm =
        param.readSValue("adaptive_macroparticle_merge");
    adaptive.root_source =
        param.readSValue("adaptive_macroparticle_root_source");
    adaptive.root_seed = read_adaptive_grid_capacity(
        param.readDValue("adaptive_macroparticle_root_seed"),
        "adaptive_macroparticle_root_seed");
    adaptive.scramble_id = read_adaptive_grid_capacity(
        param.readDValue("adaptive_macroparticle_scramble_id"),
        "adaptive_macroparticle_scramble_id");
    adaptive.error_target =
        param.readDValue("adaptive_macroparticle_error_target");
    adaptive.resample_bias_target =
        param.readDValue("adaptive_macroparticle_resample_bias_target");
    adaptive.minimum_root_support =
        param.readDValue("adaptive_macroparticle_min_root_support");
    adaptive.minimum_clone_support =
        param.readDValue("adaptive_macroparticle_min_clone_support");
    const auto read_size = [&param](const char* name) {
      const int value = param.readIValue(name);
      if (value < 0)
        adaptive_grid_input_error(name, "must be a non-negative integer");
      return static_cast<std::size_t>(value);
    };
    adaptive.statistical_cohorts =
        read_size("adaptive_macroparticle_statistical_cohorts");
    adaptive.refine_threshold =
        param.readDValue("adaptive_macroparticle_refine_threshold");
    adaptive.coarsen_threshold =
        param.readDValue("adaptive_macroparticle_coarsen_threshold");
    adaptive.update_interval =
        read_size("adaptive_macroparticle_update_interval");
    adaptive.cooldown = read_size("adaptive_macroparticle_cooldown");
    adaptive.dwell = read_size("adaptive_macroparticle_dwell");
    adaptive.maximum_change_factor =
        param.readDValue("adaptive_macroparticle_max_change_factor");
    const std::size_t maximum_generation =
        read_size("adaptive_macroparticle_max_generation");
    if (maximum_generation >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
      adaptive_grid_input_error(
          "adaptive_macroparticle_max_generation",
          "does not fit the representative generation field");
    adaptive.maximum_generation =
        static_cast<std::uint32_t>(maximum_generation);
    const std::uint64_t maximum_count = read_adaptive_grid_capacity(
        param.readDValue("adaptive_macroparticle_max_count"),
        "adaptive_macroparticle_max_count");
    const std::uint64_t maximum_bytes = read_adaptive_grid_capacity(
        param.readDValue("adaptive_macroparticle_max_bytes"),
        "adaptive_macroparticle_max_bytes");
    if (maximum_count > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()) ||
        maximum_bytes > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()))
      adaptive_grid_input_error(
          "adaptive_macroparticle_max_count/max_bytes",
          "does not fit this host size_t");
    adaptive.maximum_count = static_cast<std::size_t>(maximum_count);
    adaptive.maximum_bytes = static_cast<std::size_t>(maximum_bytes);
    adaptive.maximum_weight_ratio =
        param.readDValue("adaptive_macroparticle_max_weight_ratio");
    adaptive.policy_save =
        param.readSValue("adaptive_macroparticle_policy_save");
    adaptive.policy_load =
        param.readSValue("adaptive_macroparticle_policy_load");
    adaptive.trace_save =
        param.readSValue("adaptive_macroparticle_trace_save");
    adaptive.trace_load =
        param.readSValue("adaptive_macroparticle_trace_load");
    adaptive.profile =
        param.readSValue("adaptive_macroparticle_profile");
    const int fail_on_unmet =
        param.readIValue("adaptive_macroparticle_fail_on_unmet_error");
    if (fail_on_unmet != 0 && fail_on_unmet != 1)
      adaptive_grid_input_error(
          "adaptive_macroparticle_fail_on_unmet_error", "must be 0 or 1");
    adaptive.fail_on_unmet_error = fail_on_unmet != 0;
    adaptive.determinism =
        param.readSValue("adaptive_macroparticle_determinism");
    try
      {
        feebl::adaptive_particles::validate_configuration(adaptive);
      }
    catch (const std::exception& error)
      {
        adaptive_grid_input_error(
            "adaptive macroparticle configuration", error.what());
      }
  }

}

void SWITCHES::read(const PARAMETERS& param)
{

   n_b= param.readDValue("n_b");
   f_rep=param.readDValue("f_rep");
   charge_sign=param.readFValue("charge_sign");

  if(charge_sign>0.0){
    charge_sign_0=1.0;
  }
  if(charge_sign<0.0){
    charge_sign_0=-1.0;
  }


  integration_method = param.readIValue("integration_method");
  electron_distribution_rho =
      param.readIValue("electron_distribution_rho");
  //  silent= param.readIValue("silent");


  extra_grids = param.readIValue("grids")-1;


  if (extra_grids<0) extra_grids=0;

  load_photon = param.readIValue("load_photons");
  write_photons = param.readIValue("store_photons");
  do_photons_[0] = param.readIValue("do_photons.1");

  do_photons_[1] = param.readIValue("do_photons.2");
  
  do_beamstrahlung_ = param.readIValue("do_eloss");

  ecm_min = param.readFValue("ecm_min"); 

  float temp = param.readFValue("ecm_min_gg");
  gg_smin=4*temp*temp;

  do_hadrons = param.readIValue("do_hadrons");

  store_hadrons = param.readIValue("store_hadrons");

  hadron_ratio = param.readFValue("hadron_ratio");

  do_jets = param.readIValue("do_jets");

  do_pairs = param.readIValue("do_pairs");

  beam_pair = param.readIValue("beam_pair");

  //store_secondaries = param.readIValue("store_secondaries");
  store_pairs = param.readIValue("store_pairs");

  do_muons = param.readIValue("do_muons");

  store_muons = param.readIValue("store_muons");

  do_coherent = param.readIValue("do_coherent");

  do_trident = param.readIValue("do_trident");

  emin = param.readFValue("emin");

  //track_secondaries = param.readIValue("track_secondaries");
  track_pairs = param.readIValue("track_pairs");

  track_muons = param.readIValue("track_muons");

  do_tertphot = param.readIValue("do_tertphot");

  pair_ecut = param.readFValue("pair_ecut");

  muon_ecut = param.readFValue("muon_ecut");

  pair_step = param.readFValue("pair_step");

  electron_ratio = param.readFValue("electron_ratio");

  do_lumi = param.readIValue("do_lumi");

  bhabha_scal = param.readFValue("bhabha_scal");

  bhabha_ecmload = param.readFValue("bhabha_ecmload");

  do_bhabhas = param.readIValue("do_bhabhas");

  rndm_save = param.readIValue("rndm_save");

  rndm_load = param.readIValue("rndm_load");

  rndm_seed = param.readIValue("rndm_seed");

  read_compute_options(param);
  read_adaptive_grid_options(param);
  read_adaptive_particle_options(param);
  // do_lumi_ee_2 = param.readIValue("do_lumi_ee_2");

  do_size_log = param.readIValue("do_size_log");

  // lumi_ee_2_n = param.readIValue("lumi_ee_2_n");

  // lumi_ee_2_xmin=param.readFValue("lumi_ee_2_min"); 

  // lumi_ee_2_xmax=param.readFValue("lumi_ee_2_max"); 

  // do_lumi_eg_2=param.readIValue("do_lumi_eg_2");

  // lumi_eg_2_n=param.readIValue("lumi_eg_2_n");

  // lumi_eg_2_xmin=param.readFValue("lumi_eg_2_min"); 

  // lumi_eg_2_xmax=param.readFValue("lumi_eg_2_max"); 

  // do_lumi_ge_2=param.readIValue("do_lumi_ge_2");

  // lumi_ge_2_n=param.readIValue("lumi_ge_2_n");

  // lumi_ge_2_xmin=param.readFValue("lumi_ge_2_min"); 

  // lumi_ge_2_xmax=param.readFValue("lumi_ge_2_max"); 

  // do_lumi_gg_2=param.readIValue("do_lumi_gg_2");

  // lumi_gg_2_n=param.readIValue("lumi_gg_2_n");

  // lumi_gg_2_xmin=param.readFValue("lumi_gg_2_min"); 

  // lumi_gg_2_xmax=param.readFValue("lumi_gg_2_max"); 

  do_cross=param.readIValue("do_cross");

  do_prod=param.readIValue("do_prod");

  load_event=param.readIValue("load_events");
  imported_spin_mode_=param.readIValue("imported_spin_mode");

  prod_e=param.readFValue("prod_e");

  prod_scal=param.readFValue("prod_scal");

  do_compt=param.readIValue("do_compt");

  do_compt_phot=param.readIValue("do_compt_phot");

  compt_x_min=param.readFValue("compt_x_min");

  compt_scale=param.readFValue("compt_scale");

  compt_emax=param.readFValue("compt_emax");

  do_isr=param.readIValue("do_isr");

  do_espread=param.readIValue("do_espread");

  espread1=param.readFValue("espread.1");

  which_espread1=param.readIValue("which_espread.1");

  espread2=param.readFValue("espread.2");

  which_espread2=param.readIValue("which_espread.2");

  num_lumi=param.readIValue("num_lumi");

  num_lumi_eg=param.readIValue("num_lumi_eg");

  num_lumi_gg=param.readIValue("num_lumi_gg");

  lumi_p=param.readFValue("lumi_p");

  lumi_p_eg=param.readFValue("lumi_p_eg");

  lumi_p_gg=param.readFValue("lumi_p_gg");

  photon_ratio=param.readFValue("photon_ratio");

  pair_ratio=param.readFValue("pair_ratio");

  muon_ratio=param.readFValue("muon_ratio");

  muon_scale=param.readFValue("muon_scale");
  if (do_muons && (!adaptive_grid_is_finite(muon_scale) || !(muon_scale > 0.0)))
    {
      std::cerr << " SWITCHES: enabled muon production requires finite muon_scale > 0" << std::endl;
      exit(1);
    }

  bhabha_ratio=param.readFValue("bhabha_ratio");

  jet_ratio=param.readFValue("jet_ratio");

  jet_pstar=param.readFValue("jet_ptmin");

  jet_store=param.readIValue("store_jets");

  jet_select=param.readIValue("jet_log");

  geom=param.readIValue("beam_size");

  r_scal=param.readFValue("beam_size_scale"); 

  photon_timing=param.readIValue("photon_timing");
  if (photon_timing < 0 || photon_timing > 3)
    {
      std::cerr << " SWITCHES: photon_timing must be 0 (legacy), 1 (synchronised), 2 (diagnostic: synchronised"
		<< " position, full weight in the step of birth) or 3 (synchronised, time-centred direction); got "
		<< photon_timing << std::endl;
      exit(1);
    }
  photon_angle=param.readIValue("photon_angle");
  if (photon_angle < 0 || photon_angle > 2)
    {
      std::cerr << " SWITCHES: photon_angle must be 0 (collinear), 1 (LCFA angle) or 2 (LCFA angle where the"
		<< " formation length is below half the opposing bunch length); got " << photon_angle << std::endl;
      exit(1);
    }

  ext_field=param.readIValue("ext_field");

//   if (!EXT_FIELD) 
//     {
//       if (ext_field){
// 	//	fprintf(stderr,"EXT_FIELD = false (LesDifines.h) \n");
// 	fprintf(stderr,"Cannot use flag ext_field\n");
// 	//	fprintf(stderr,"Please recompile with EXT_FIELD = true \n");
// 	exit(1);
//       }
      //    }
  pair_q2=param.readIValue("pair_q2");
  if ((do_pairs || do_muons) && (pair_q2 < 0 || pair_q2 > 2))
    {
      // any other value would give every pair and muon zero weight (no default branch)
      std::cerr << " SWITCHES: pair_q2 must be 0, 1 or 2; got " << pair_q2 << std::endl;
      exit(1);
    }
  pair_model=param.readIValue("pair_model");
  if (pair_model < 0 || pair_model > 2)
    {
      std::cerr << " SWITCHES: pair_model must be 0 (legacy), 1 (corrected EPA) or 2 (exact LO); got "
		<< pair_model << std::endl;
      exit(1);
    }
#ifndef QEDBG_USE_QUADMATH
  // the exact kernels need an extended-precision long double (qedbg_real.h); where long double is the
  // IEEE double (e.g. AArch64 macOS) the code builds, and pair_model 0 and 1 run, but 2 is refused
  if (pair_model == 2 && !qedbg::kExtendedLongDouble)
    {
      std::cerr << " SWITCHES: pair_model = 2 needs an extended-precision long double (64-bit mantissa); this build has "
		<< LDBL_MANT_DIG << " bits" << std::endl;
      exit(1);
    }
#endif
  pair_exact_rr=param.readFValue("pair_exact_rr");
  if (!adaptive_grid_is_finite(pair_exact_rr) || pair_exact_rr < 0.0)
    {
      std::cerr << " SWITCHES: pair_exact_rr must be finite and >= 0 (0 disables the Russian roulette); got " << pair_exact_rr << std::endl;
      exit(1);
    }
  photon_polarisation=param.readIValue("photon_polarisation");
  if (photon_polarisation < 0 || photon_polarisation > 1)
    {
      std::cerr << " SWITCHES: photon_polarisation must be 0 (unpolarised photons) or 1 (polarised"
		<< " beamstrahlung photons); got " << photon_polarisation << std::endl;
      exit(1);
    }
  if (photon_polarisation == 1 && pair_model == 0)
    {
      // Mode 0 retains FeeBL's existing unpolarized EPA model. Polarized cross
      // sections are implemented by the corrected (1) and exact (2) models.
      std::cerr << " SWITCHES: photon_polarisation = 1 requires pair_model = 1 or 2" << std::endl;
      exit(1);
    }
  coherent_model=param.readIValue("coherent_model");
  if (coherent_model < 0 || coherent_model > 1)
    {
      std::cerr << " SWITCHES: coherent_model must be 0 (legacy) or 1 (corrected sub-stepping of large coherent"
		<< " conversion probabilities); got " << coherent_model << std::endl;
      exit(1);
    }
  if (photon_polarisation == 1 && do_coherent != 0 && coherent_model == 0)
    {
      // the polarised conversion widens the trial probability by 1 + |tau|/2 and takes it out again in the
      // energy-split acceptance; with the legacy gate the widening would stay in the rate for 0.1 <= tmp < 1
      std::cerr << " SWITCHES: photon_polarisation = 1 with do_coherent = 1 requires coherent_model = 1" << std::endl;
      exit(1);
    }
  if ((photon_timing == 1 || photon_timing == 3) &&
      do_coherent != 0 && coherent_model == 0)
    {
      // Half of the birth-step exposure must scale one conversion probability.
      // The historical extra rejection gate would apply the factor twice.
      std::cerr << " SWITCHES: photon_timing = 1 or 3 with do_coherent = 1"
                << " requires coherent_model = 1" << std::endl;
      exit(1);
    }

  store_beam=param.readIValue("store_beam");

  load_beam=param.readIValue("load_beam");

  cuts_from_loaded_beam = param.readIValue("cuts_from_loaded_beam");

  bmt_precession_   = param.readIValue("bmt_precession");
  ST_spin_flip_   = param.readIValue("ST_spin_flip");

  store_native_ = param.readIValue("store_native");
  loaded_beam_is_pretransformed_ = param.readIValue("loaded_beam_is_pretransformed");
  automatic_grid_sizing = param.readIValue("automatic_grid_sizing");
  optimize_grids = param.readIValue("optimize_grids");
  do_photon_lumi = param.readIValue("do_photon_lumi");
  electron_lumi_precompute_ = param.readIValue("electron_lumi_precompute");
  electron_lumi_sampling_ = param.readIValue("electron_lumi_sampling");
  electron_lumi_spatial_kernel_ =
      param.readIValue("electron_lumi_spatial_kernel");
  electron_lumi_spatial_nx_ = param.readIValue("electron_lumi_spatial_nx");
  electron_lumi_spatial_ny_ = param.readIValue("electron_lumi_spatial_ny");
  electron_lumi_spatial_x_min_ =
      param.readDValue("electron_lumi_spatial_x_min");
  electron_lumi_spatial_x_max_ =
      param.readDValue("electron_lumi_spatial_x_max");
  electron_lumi_spatial_y_min_ =
      param.readDValue("electron_lumi_spatial_y_min");
  electron_lumi_spatial_y_max_ =
      param.readDValue("electron_lumi_spatial_y_max");
  {
    const double maximum_pairs =
        param.readDValue("electron_lumi_spatial_max_pairs");
    electron_lumi_spatial_max_pairs_ =
        electron_lumi_spatial_kernel_ == 1
            ? read_adaptive_grid_capacity(
                  maximum_pairs, "electron_lumi_spatial_max_pairs")
            : 0U;
  }
  electron_lumi_spatial_closure_tolerance_ =
      param.readDValue("electron_lumi_spatial_closure_tolerance");
  photon_lumi_factorized_ = param.readIValue("photon_lumi_factorized");
  crab_mode_[0] = param.readIValue("crab_mode.1");
  crab_mode_[1] = param.readIValue("crab_mode.2");
  crab_plane_[0] = param.readIValue("crab_plane.1");
  crab_plane_[1] = param.readIValue("crab_plane.2");
  crab_longitudinal_sign_[0] = param.readIValue("crab_longitudinal_sign.1");
  crab_longitudinal_sign_[1] = param.readIValue("crab_longitudinal_sign.2");
  waist_mode_[0] = param.readIValue("waist_mode.1");
  waist_mode_[1] = param.readIValue("waist_mode.2");
  crab_frequency_[0] = param.readDValue("crab_frequency.1");
  crab_frequency_[1] = param.readDValue("crab_frequency.2");
  crab_phase_[0] = param.readDValue("crab_phase.1");
  crab_phase_[1] = param.readDValue("crab_phase.2");
  crab_target_tilt_[0] = param.readDValue("crab_target_tilt.1");
  crab_target_tilt_[1] = param.readDValue("crab_target_tilt.2");
  waist_fraction_[0] = param.readDValue("waist_fraction.1");
  waist_fraction_[1] = param.readDValue("waist_fraction.2");
  ip_dispersion_[0] = param.readDValue("ip_dispersion.1");
  ip_dispersion_[1] = param.readDValue("ip_dispersion.2");
  incoming_map_reslice_=param.readIValue("incoming_map_reslice");
  design_axis_spectrum_=param.readIValue("design_axis_spectrum");
  design_axis_peak_bins_=param.readIValue("design_axis_peak_bins");
  spectrum_w_min_  = param.readDValue("spectrum_w_min");
  spectrum_w_max_  = param.readDValue("spectrum_w_max");
  spectrum_w_bins_ = param.readIValue("spectrum_w_bins");
  hist_ee_bins_declared_ = param.readIValue("hist_ee_bins");
  hist_ee_max_declared_  = param.readDValue("hist_ee_max");
  crab_harmonic_n_ = param.readIValue("crab_harmonic_n");
  crab_include_panofsky_wenzel_ = param.readIValue("crab_include_panofsky_wenzel");
  crab_full_crossing_angle_ = param.readDValue("crab_full_crossing_angle");
  waist_symplecticity_tolerance_ = param.readDValue("waist_symplecticity_tolerance");

  load_photon=param.readIValue("load_photons");

  jet_pythia=param.readIValue("jet_pythia");

  force_symmetric=param.readIValue("force_symmetric");

  charge_symmetric=param.readIValue("charge_symmetric");

  do_bds_spin_rotation=param.readIValue("do_bds_spin_rotation");

  // beam_vx_min=param.readFValue("beam_vx_min");

  // beam_vx_max=param.readFValue("beam_vx_max");

  // beam_vx_interval=param.readIValue("beam_vx_interval");

  // beam_vy_min=param.readFValue("beam_vy_min");

  // beam_vy_max=param.readFValue("beam_vy_max");

  // beam_vy_interval=param.readIValue("beam_vy_interval");

  do_dump=param.readIValue("do_dump");

  dump_step=param.readIValue("dump_step");

  dump_particle=param.readIValue("dump_particle");
  if ( dump_step <= 0 ) dump_step = 1;
  if ( dump_particle <= 0 ) dump_particle = 1;
  // beam_size selects the virtual-photon placement (electronScatter switch without a default branch):
  // 0 at the emitter, 1 ring (legacy default), 2 Gaussian, 3 exact impact-parameter profile
  if ((do_pairs || do_muons || do_compt || do_hadrons) && (geom < 0 || geom > 3))
    {
      std::cerr << " SWITCHES: beam_size must be 0, 1, 2 or 3 when virtual photons are used; got " << geom << std::endl;
      exit(1);
    }
  // thinning is unbiased only for ratios in (0, 1]: above 1 every particle is kept but weighted 1/ratio
  if ((do_pairs || do_muons || do_compt || do_hadrons) && (electron_ratio > 1.0 || photon_ratio > 1.0))
    std::cerr << " WARNING: electron_ratio = " << electron_ratio << ", photon_ratio = " << photon_ratio
              << ": values above 1 bias the photon and electron weights low (keep them in (0, 1])." << std::endl;
  // silent-zero configurations of the incoherent pair and muon channels (warnings only)
  if ((do_pairs || do_muons) && !(do_photons_[0] || do_photons_[1] || load_photon) && pair_model == 0)
    std::cerr << " WARNING: pairs/muons without real photons (do_photons.1/2 = 0, load_photons = 0): with"
              << " pair_model = 0 the Landau-Lifshitz channel is not computed (legacy gating); pair_model >= 1 computes it."
              << std::endl;
  if ((do_pairs || do_muons || do_compt || do_hadrons) && electron_ratio <= 0.0)
    std::cerr << " WARNING: electron_ratio = " << electron_ratio << " generates no virtual photons: Bethe-Heitler and"
              << " Landau-Lifshitz pairs and muons, virtual-photon Compton and gamma* hadrons are zero"
              << " (electron_ratio = 1 is the GUINEA-PIG++ default)." << std::endl;
  if ((do_pairs || do_muons) && !do_photon_lumi)
    std::cerr << " WARNING: do_photon_lumi = 0 skips GRID::photon_lumi, the only caller of the Breit-Wheeler pair and"
              << " muon channels: Breit-Wheeler production is zero." << std::endl;
  check_consistency();

  do_edm4hep=param.readFValue("do_edm4hep");
  if (adaptive_grid_mode_ >= 2)
    std::cerr << adaptive_grid_configuration_flow() << std::endl;
    }

void SWITCHES::check_consistency() const
{
  // Order fixes which invalid setting is reported first. Each stage is
  // read-only; parsed class lists remain owned by this validation call.
  check_imported_spin_consistency();
  const AdaptiveGridSizeClasses classes = check_adaptive_geometry_consistency();
  check_adaptive_error_bounds();
  check_adaptive_error_policy();
  check_adaptive_source_bounds();
  check_adaptive_operator_consistency();
  check_adaptive_patch_bounds();
  if (adaptive_grid_mode_ >= 2) {
    const std::vector<int> spacing_factors = check_adaptive_spacing_consistency();
    check_adaptive_source_consistency();
    check_adaptive_precision_consistency(spacing_factors);
    check_adaptive_capacity_consistency();
    if (adaptive_grid_mode_ == 3) {
      check_adaptive_patch_consistency(classes);
    }
  }
  check_wavefront_dispatch_consistency();
  check_spatial_luminosity_consistency();
  check_precompute_luminosity_consistency();
  check_factorized_photon_consistency();
  check_private_wavefront_consistency();
  check_secondary_output_consistency();
}

// Admit only the explicitly supported imported-particle spin model.
void SWITCHES::check_imported_spin_consistency() const
{
  if (imported_spin_mode_!=0 && imported_spin_mode_!=1)
    throw std::runtime_error("imported_spin_mode supports 0 (off) or 1 (BMT only, imported radiation off); secondary radiative spin flips are not implemented");
  if (imported_spin_mode_ && (!load_event || !track_pairs || do_pairs ||
      beam_pair || do_compt || do_prod || do_bhabhas || track_muons))
    throw std::runtime_error("imported_spin_mode=1 requires a pure imported electron/positron population with load_events and track_pairs enabled");
  if (imported_spin_mode_ && (collision_schedule_!="legacy" ||
      compute_backend_!="serial" || wavefront_parallel_physics_ || cpu_primary_move_ ||
      gpu2_primary_mover_!="serial"))
    throw std::runtime_error("imported_spin_mode=1 requires the serial legacy collision tracker");
}

// Validate window dimensions and parse each size-class list once.
SWITCHES::AdaptiveGridSizeClasses SWITCHES::check_adaptive_geometry_consistency() const
{
  AdaptiveGridSizeClasses classes;
  if (adaptive_grid_mode_ < 0 || adaptive_grid_mode_ > 3)
    adaptive_grid_input_error(
        "adaptive_grid_mode", "must be 0, 1, 2, or 3");
  if (adaptive_grid_scope_ != "encounter" && adaptive_grid_scope_ != "wave")
    adaptive_grid_input_error(
        "adaptive_grid_scope", "must be encounter or wave");
  if (adaptive_grid_profile_ != "off" &&
      adaptive_grid_profile_ != "summary" &&
      adaptive_grid_profile_ != "encounter")
    adaptive_grid_input_error(
        "adaptive_grid_profile", "must be off, summary, or encounter");
  if (adaptive_grid_allow_full_window_ != 0 &&
      adaptive_grid_allow_full_window_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_allow_full_window", "must be 0 or 1");
  if (adaptive_grid_fail_on_unmet_error_ != 0 &&
      adaptive_grid_fail_on_unmet_error_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_fail_on_unmet_error", "must be 0 or 1");
  if (adaptive_grid_pilot_diagnostics_ != 0 &&
      adaptive_grid_pilot_diagnostics_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_pilot_diagnostics", "must be 0 or 1");
  if (adaptive_grid_mode_ < 2 && adaptive_grid_pilot_diagnostics_ != 0)
    adaptive_grid_input_error(
        "adaptive_grid_pilot_diagnostics",
        "is only valid in adaptive grid modes 2/3");

  if (adaptive_grid_min_nx_ < 0 || adaptive_grid_min_ny_ < 0 ||
      adaptive_grid_max_nx_ < 0 || adaptive_grid_max_ny_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_min/max_nx/ny",
        "must be non-negative (0 is the derive-at-preflight sentinel)");
  validate_adaptive_grid_dimension_limit(
      adaptive_grid_min_nx_, "adaptive_grid_min_nx");
  validate_adaptive_grid_dimension_limit(
      adaptive_grid_min_ny_, "adaptive_grid_min_ny");
  validate_adaptive_grid_dimension_limit(
      adaptive_grid_max_nx_, "adaptive_grid_max_nx");
  validate_adaptive_grid_dimension_limit(
      adaptive_grid_max_ny_, "adaptive_grid_max_ny");
  if (adaptive_grid_min_nx_ > 0 && adaptive_grid_max_nx_ > 0 &&
      adaptive_grid_min_nx_ > adaptive_grid_max_nx_)
    adaptive_grid_input_error(
        "adaptive_grid_min_nx", "must not exceed adaptive_grid_max_nx");
  if (adaptive_grid_min_ny_ > 0 && adaptive_grid_max_ny_ > 0 &&
      adaptive_grid_min_ny_ > adaptive_grid_max_ny_)
    adaptive_grid_input_error(
        "adaptive_grid_min_ny", "must not exceed adaptive_grid_max_ny");
  if (adaptive_grid_extra_guard_cells_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_extra_guard_cells", "must be non-negative");

  const bool has_x_classes = !adaptive_grid_size_classes_x_.empty();
  const bool has_y_classes = !adaptive_grid_size_classes_y_.empty();
  if (has_x_classes != has_y_classes)
    adaptive_grid_input_error(
        "adaptive_grid_size_classes_x/y",
        "must either both be empty or both contain explicit class lists");
  classes.x = validate_adaptive_grid_size_classes(
      adaptive_grid_size_classes_x_, "adaptive_grid_size_classes_x",
      adaptive_grid_min_nx_, adaptive_grid_max_nx_);
  classes.y = validate_adaptive_grid_size_classes(
      adaptive_grid_size_classes_y_, "adaptive_grid_size_classes_y",
      adaptive_grid_min_ny_, adaptive_grid_max_ny_);
  if (adaptive_grid_mode_ >= 1 && !has_x_classes &&
      adaptive_grid_allow_full_window_ == 0)
    adaptive_grid_input_error(
        "adaptive_grid_size_classes_x/y",
        "cannot both be empty in an adaptive mode when full-window selection is disabled");
  return classes;
}

// Check finite scalar error bounds before policy-specific combinations.
void SWITCHES::check_adaptive_error_bounds() const
{
  const double nonnegative_values[] = {
      adaptive_grid_tau_curvature_, adaptive_grid_tau_field_gradient_,
      adaptive_grid_tau_richardson_, adaptive_grid_tau_tail_,
      adaptive_grid_encounter_tail_fraction_,
      adaptive_grid_rho_floor_, adaptive_grid_field_floor_,
      adaptive_grid_min_neff_, adaptive_grid_error_reference_field_,
      adaptive_grid_error_absolute_floor_fraction_,
      adaptive_grid_error_catastrophe_absolute_fraction_,
      adaptive_grid_error_exposure_limit_};
  const char* const nonnegative_names[] = {
      "adaptive_grid_tau_curvature", "adaptive_grid_tau_field_gradient",
      "adaptive_grid_tau_richardson", "adaptive_grid_tau_tail",
      "adaptive_grid_encounter_tail_fraction",
      "adaptive_grid_rho_floor", "adaptive_grid_field_floor",
      "adaptive_grid_min_neff", "adaptive_grid_error_reference_field",
      "adaptive_grid_error_absolute_floor_fraction",
      "adaptive_grid_error_catastrophe_absolute_fraction",
      "adaptive_grid_error_exposure_limit"};
  for (std::size_t i = 0;
       i < sizeof(nonnegative_values) / sizeof(nonnegative_values[0]); ++i)
    if (!adaptive_grid_is_finite(nonnegative_values[i]) ||
        nonnegative_values[i] < 0.0)
      adaptive_grid_input_error(
          nonnegative_names[i], "must be finite and non-negative");
  if (!adaptive_grid_is_finite(adaptive_grid_anchor_x_) ||
      !adaptive_grid_is_finite(adaptive_grid_anchor_y_))
    adaptive_grid_input_error(
        "adaptive_grid_anchor_x/y", "must be finite");
  if (!adaptive_grid_is_finite(adaptive_grid_error_target_) ||
      adaptive_grid_error_target_ < 0.0 ||
      adaptive_grid_error_target_ >= 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_error_target",
        "must be 0 (unset) or a finite value in (0,1)");
  if (adaptive_grid_error_norm_policy_ != "component_rms_linf_v1" &&
      adaptive_grid_error_norm_policy_ != "vector_rms_quantile_v1" &&
      adaptive_grid_error_norm_policy_ !=
          "vector_mixed_rms_quantile_v2")
    adaptive_grid_input_error(
        "adaptive_grid_error_norm_policy",
        "must be component_rms_linf_v1, vector_rms_quantile_v1, or "
        "vector_mixed_rms_quantile_v2");
  if (!adaptive_grid_is_finite(adaptive_grid_error_quantile_) ||
      adaptive_grid_error_quantile_ < 0.95 ||
      adaptive_grid_error_quantile_ >= 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_error_quantile",
        "must be a high quantile finite in [0.95,1); 0.999 is the default "
        "p99.9 audit");
  if (!adaptive_grid_is_finite(
          adaptive_grid_error_catastrophe_limit_) ||
      adaptive_grid_error_catastrophe_limit_ <= 0.0 ||
      adaptive_grid_error_catastrophe_limit_ > 2.0)
    adaptive_grid_input_error(
        "adaptive_grid_error_catastrophe_limit",
        "must be finite in (0,2]; two is the exact symmetric-vector bound");
}

// Validate the mixed-vector budget and noise-aware-stop selector.
void SWITCHES::check_adaptive_error_policy() const
{
  const bool mixed_vector_policy =
      adaptive_grid_error_norm_policy_ ==
          "vector_mixed_rms_quantile_v2";
  if (mixed_vector_policy) {
    if (!is_canonical_lowercase_sha256(
            adaptive_grid_error_budget_sha256_))
      adaptive_grid_input_error(
          "adaptive_grid_error_budget_sha256",
          "must be exactly 64 lowercase hexadecimal characters for "
          "vector_mixed_rms_quantile_v2");
    if (!(adaptive_grid_error_reference_field_ > 0.0))
      adaptive_grid_input_error(
          "adaptive_grid_error_reference_field",
          "must be finite and positive for vector_mixed_rms_quantile_v2");
    if (!(adaptive_grid_error_absolute_floor_fraction_ > 0.0) ||
        adaptive_grid_error_absolute_floor_fraction_ > 1.0)
      adaptive_grid_input_error(
          "adaptive_grid_error_absolute_floor_fraction",
          "must be finite in (0,1] for vector_mixed_rms_quantile_v2");
    if (!(adaptive_grid_error_catastrophe_absolute_fraction_ > 0.0) ||
        adaptive_grid_error_catastrophe_absolute_fraction_ > 2.0)
      adaptive_grid_input_error(
          "adaptive_grid_error_catastrophe_absolute_fraction",
          "must be finite in (0,2] for vector_mixed_rms_quantile_v2");
    if (!(adaptive_grid_error_exposure_limit_ > 0.0) ||
        !(adaptive_grid_error_exposure_limit_ < 1.0))
      adaptive_grid_input_error(
          "adaptive_grid_error_exposure_limit",
          "must be finite in (0,1) for vector_mixed_rms_quantile_v2");
  } else if (!adaptive_grid_error_budget_sha256_.empty() ||
             adaptive_grid_error_reference_field_ != 0.0 ||
             adaptive_grid_error_absolute_floor_fraction_ != 0.0 ||
             adaptive_grid_error_catastrophe_absolute_fraction_ != 0.0 ||
             adaptive_grid_error_exposure_limit_ != 0.0) {
    adaptive_grid_input_error(
        "adaptive_grid_error_budget_sha256/reference_field/"
        "absolute_floor_fraction/catastrophe_absolute_fraction/"
        "exposure_limit",
        "must remain empty/zero unless vector_mixed_rms_quantile_v2 is "
        "selected");
  }
  if (adaptive_grid_noise_aware_stop_ != 0 &&
      adaptive_grid_noise_aware_stop_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_noise_aware_stop", "must be 0 or 1");
}

// Validate source tolerances, indicators and update intervals.
void SWITCHES::check_adaptive_source_bounds() const
{
  const double source_error_targets[] = {
      adaptive_grid_source_error_target_,
      adaptive_grid_solver_error_target_,
      adaptive_grid_lumi_error_target_};
  const char* const source_error_names[] = {
      "adaptive_grid_source_error_target",
      "adaptive_grid_solver_error_target",
      "adaptive_grid_lumi_error_target"};
  for (std::size_t index = 0U; index < 3U; ++index)
    if (!adaptive_grid_is_finite(source_error_targets[index]) ||
        source_error_targets[index] < 0.0 ||
        source_error_targets[index] >= 1.0)
      adaptive_grid_input_error(
          source_error_names[index],
          "must be 0 (unset) or a finite value in (0,1)");
  if (!adaptive_grid_is_finite(adaptive_grid_lumi_floor_) ||
      adaptive_grid_lumi_floor_ < 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_lumi_floor",
        "must be finite and non-negative");
  if (!adaptive_grid_is_finite(adaptive_grid_source_width_x_) ||
      !adaptive_grid_is_finite(adaptive_grid_source_width_y_) ||
      adaptive_grid_source_width_x_ < 0.0 ||
      adaptive_grid_source_width_y_ < 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_source_width_x/y",
        "must be finite and non-negative");
  if (!adaptive_grid_is_finite(adaptive_grid_refine_threshold_) ||
      adaptive_grid_refine_threshold_ <= 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_refine_threshold", "must be finite and positive");
  if (!adaptive_grid_is_finite(adaptive_grid_coarsen_threshold_) ||
      adaptive_grid_coarsen_threshold_ < 0.0 ||
      adaptive_grid_coarsen_threshold_ >= adaptive_grid_refine_threshold_)
    adaptive_grid_input_error(
        "adaptive_grid_coarsen_threshold",
        "must be finite, non-negative, and below the refine threshold");
  if (adaptive_grid_monitor_ !=
      "charge_curvature_field_richardson_overlap")
    adaptive_grid_input_error(
        "adaptive_grid_monitor",
        "must be charge_curvature_field_richardson_overlap");
  if (adaptive_grid_indicator_smoothing_passes_ < 0 ||
      adaptive_grid_indicator_smoothing_passes_ > 2)
    adaptive_grid_input_error(
        "adaptive_grid_indicator_smoothing_passes", "must be 0, 1, or 2");
  if (adaptive_grid_indicator_reduction_ != "connected_max")
    adaptive_grid_input_error(
        "adaptive_grid_indicator_reduction", "must be connected_max");
  if (adaptive_grid_shrink_dwell_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_shrink_dwell", "must be non-negative");
  if (adaptive_grid_update_interval_ <= 0)
    adaptive_grid_input_error(
        "adaptive_grid_update_interval", "must be positive");
  if (adaptive_grid_richardson_interval_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_richardson_interval",
        "must be non-negative (0 is the unset sentinel)");
}

// Restrict field operators and error policies to their supported modes.
void SWITCHES::check_adaptive_operator_consistency() const
{
  if (adaptive_grid_richardson_order_source_ != "calibrated_table" &&
      adaptive_grid_richardson_order_source_ != "three_level")
    adaptive_grid_input_error(
        "adaptive_grid_richardson_order_source",
        "must be calibrated_table or three_level");
  if (adaptive_grid_field_operator_ != "legacy_cic12_v1" &&
      adaptive_grid_field_operator_ != "fd4_lagrange4_v1")
    adaptive_grid_input_error(
        "adaptive_grid_field_operator",
        "must be legacy_cic12_v1 or fd4_lagrange4_v1");
  if (adaptive_grid_mode_ < 2 &&
      adaptive_grid_field_operator_ != "legacy_cic12_v1")
    adaptive_grid_input_error(
        "adaptive_grid_field_operator",
        "is only meaningful in adaptive grid modes 2/3");
  if (adaptive_grid_mode_ < 2 &&
      adaptive_grid_error_norm_policy_ != "component_rms_linf_v1")
    adaptive_grid_input_error(
        "adaptive_grid_error_norm_policy",
        "vector quantile policies are only meaningful in adaptive grid "
        "modes 2/3");
  if (adaptive_grid_mode_ < 2 && adaptive_grid_noise_aware_stop_ != 0)
    adaptive_grid_input_error(
        "adaptive_grid_noise_aware_stop",
        "is only meaningful in adaptive grid modes 2/3");
  if (adaptive_grid_mode_ < 2 &&
      adaptive_grid_max_refinement_factor_ != 0U)
    adaptive_grid_input_error(
        "adaptive_grid_max_refinement_factor",
        "is only meaningful in adaptive grid modes 2/3");
}

// Validate patch selectors before the resolution-dependent contracts.
void SWITCHES::check_adaptive_patch_bounds() const
{
  if (adaptive_grid_max_levels_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_max_levels", "must be 1 for the initial one-patch design");
  if (adaptive_grid_max_patches_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_max_patches", "must be 1 for the initial one-patch design");
  if (adaptive_grid_refinement_ratio_ != 2)
    adaptive_grid_input_error(
        "adaptive_grid_refinement_ratio",
        "must be 2; ratio 4 remains reserved until separately qualified");
  if (adaptive_grid_blocking_factor_ < 0 ||
      (adaptive_grid_blocking_factor_ != 0 &&
       !is_power_of_two(adaptive_grid_blocking_factor_)))
    adaptive_grid_input_error(
        "adaptive_grid_blocking_factor",
        "must be 0 (unset) or a positive power of two");
  if (adaptive_grid_error_buffer_cells_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_error_buffer_cells", "must be non-negative");
  if (adaptive_grid_transition_cells_ < 0)
    adaptive_grid_input_error(
        "adaptive_grid_transition_cells", "must be non-negative");
  if (!adaptive_grid_is_finite(adaptive_grid_fine_residual_tolerance_) ||
      adaptive_grid_fine_residual_tolerance_ < 0.0 ||
      adaptive_grid_fine_residual_tolerance_ >= 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_fine_residual_tolerance",
        "must be 0 (unset) or a finite value in (0,1)");
  if (adaptive_grid_terminal_patch_certification_ != 0 &&
      adaptive_grid_terminal_patch_certification_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_terminal_patch_certification", "must be 0 or 1");
  if (adaptive_grid_terminal_patch_certification_ != 0 &&
      adaptive_grid_mode_ != 3)
    adaptive_grid_input_error(
        "adaptive_grid_terminal_patch_certification",
        "is an opt-in Mode-3-only mechanism");
}

// Parse the ratio-two resolution ladder and its optional cap.
std::vector<int> SWITCHES::check_adaptive_spacing_consistency() const
{
  const std::vector<int> spacing_factors =
      validate_adaptive_grid_spacing_classes(
          adaptive_grid_field_spacing_classes_);
  if (spacing_factors.empty())
    adaptive_grid_input_error(
        "adaptive_grid_field_spacing_classes",
        "is required in modes 2/3 (v1 grammar: 1,2,4,...)");
  if (spacing_factors.size() < 3)
    adaptive_grid_input_error(
        "adaptive_grid_field_spacing_classes",
        "must contain at least three classes so the finest fallback has "
        "an asymptotic-consistency gate");
  if (adaptive_grid_max_refinement_factor_ != 0U &&
      (adaptive_grid_max_refinement_factor_ &
       (adaptive_grid_max_refinement_factor_ - 1U)) != 0U)
    adaptive_grid_input_error(
        "adaptive_grid_max_refinement_factor",
        "must be 0 (no user factor cap) or a power of two");
  if (adaptive_grid_max_refinement_factor_ != 0U &&
      adaptive_grid_max_refinement_factor_ <
          static_cast<std::uint64_t>(spacing_factors.back()))
    adaptive_grid_input_error(
        "adaptive_grid_max_refinement_factor",
        "must be at least the final bootstrap spacing factor");
  for (std::size_t index = 1U;
       index < spacing_factors.size(); ++index)
    if (static_cast<std::uint64_t>(spacing_factors[index]) !=
        2U * static_cast<std::uint64_t>(
                 spacing_factors[index - 1U]))
      adaptive_grid_input_error(
          "adaptive_grid_field_spacing_classes",
          "must use consecutive ratio-two classes in modes 2/3 so "
          "three-level contraction and uniform-fine oracles are "
          "well-defined");
  return spacing_factors;
}

// Check source-model-specific widths, reference grids and error controls.
void SWITCHES::check_adaptive_source_consistency() const
{
  if (adaptive_grid_source_model_ != "canonical_cell_average_v1" &&
      adaptive_grid_source_model_ != "reference_cic_v1" &&
      adaptive_grid_source_model_ != "fixed_width_bspline1_v1")
    adaptive_grid_input_error(
        "adaptive_grid_source_model",
        "must explicitly be canonical_cell_average_v1 or "
        "reference_cic_v1 or fixed_width_bspline1_v1 in modes 2/3");
  if (adaptive_grid_source_model_ == "reference_cic_v1")
    {
      if (adaptive_grid_source_reference_nx_ == 0U ||
          adaptive_grid_source_reference_ny_ == 0U)
        adaptive_grid_input_error(
            "adaptive_grid_source_reference_nx/ny",
            "must be explicit positive dimensions for reference_cic_v1");
      require_positive_mode2_tolerance(
          adaptive_grid_source_error_target_,
          "adaptive_grid_source_error_target");
      require_positive_mode2_tolerance(
          adaptive_grid_solver_error_target_,
          "adaptive_grid_solver_error_target");
      require_positive_mode2_tolerance(
          adaptive_grid_lumi_error_target_,
          "adaptive_grid_lumi_error_target");
      if (!(adaptive_grid_lumi_floor_ > 0.0))
        adaptive_grid_input_error(
            "adaptive_grid_lumi_floor",
            "must be finite and positive for reference_cic_v1");
      if (adaptive_grid_anchor_x_ != 0.0 ||
          adaptive_grid_anchor_y_ != 0.0)
        adaptive_grid_input_error(
            "adaptive_grid_anchor_x/y",
            "must be exactly zero for registered reference_cic_v1");
      if (adaptive_grid_richardson_order_source_ != "three_level" ||
          !adaptive_grid_richardson_order_table_.empty())
        adaptive_grid_input_error(
            "adaptive_grid_richardson_order_source/table",
            "reference_cic_v1 requires three_level and an empty table");
      if (adaptive_grid_source_width_x_ != 0.0 ||
          adaptive_grid_source_width_y_ != 0.0)
        adaptive_grid_input_error(
            "adaptive_grid_source_width_x/y",
            "must be zero for reference_cic_v1");
    }
  else if (adaptive_grid_source_model_ == "fixed_width_bspline1_v1")
    {
      if (adaptive_macroparticle_configuration_.mode !=
          feebl::adaptive_particles::Mode::LegacyOff)
        adaptive_grid_input_error(
            "adaptive_grid_source_model/adaptive_macroparticle_mode",
            "fixed_width_bspline1_v1 currently requires "
            "adaptive_macroparticle_mode=0; adaptive source projection "
            "is qualified only for reference_cic_v1");
      if (adaptive_grid_source_reference_nx_ != 0U ||
          adaptive_grid_source_reference_ny_ != 0U)
        adaptive_grid_input_error(
            "adaptive_grid_source_reference_nx/ny",
            "must be zero for fixed_width_bspline1_v1");
      if (!(adaptive_grid_source_width_x_ > 0.0) ||
          !(adaptive_grid_source_width_y_ > 0.0))
        adaptive_grid_input_error(
            "adaptive_grid_source_width_x/y",
            "must be finite positive physical half-widths for "
            "fixed_width_bspline1_v1");
      require_positive_mode2_tolerance(
          adaptive_grid_source_error_target_,
          "adaptive_grid_source_error_target");
      require_positive_mode2_tolerance(
          adaptive_grid_solver_error_target_,
          "adaptive_grid_solver_error_target");
      require_positive_mode2_tolerance(
          adaptive_grid_lumi_error_target_,
          "adaptive_grid_lumi_error_target");
      if (!(adaptive_grid_lumi_floor_ > 0.0))
        adaptive_grid_input_error(
            "adaptive_grid_lumi_floor",
            "must be finite and positive for fixed_width_bspline1_v1");
      if (adaptive_grid_anchor_x_ != 0.0 ||
          adaptive_grid_anchor_y_ != 0.0)
        adaptive_grid_input_error(
            "adaptive_grid_anchor_x/y",
            "must be exactly zero for fixed_width_bspline1_v1");
      if (adaptive_grid_richardson_order_source_ != "three_level" ||
          !adaptive_grid_richardson_order_table_.empty())
        adaptive_grid_input_error(
            "adaptive_grid_richardson_order_source/table",
            "fixed_width_bspline1_v1 requires three_level and an empty "
            "table");
    }
  else if (adaptive_grid_source_reference_nx_ != 0U ||
           adaptive_grid_source_reference_ny_ != 0U ||
           adaptive_grid_source_width_x_ != 0.0 ||
           adaptive_grid_source_width_y_ != 0.0 ||
           adaptive_grid_source_error_target_ != 0.0 ||
           adaptive_grid_solver_error_target_ != 0.0 ||
           adaptive_grid_lumi_error_target_ != 0.0 ||
           adaptive_grid_lumi_floor_ != 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_source_reference_*/source_width_*/source/solver/"
        "lumi targets",
        "are reserved for reference_cic_v1 or fixed_width_bspline1_v1");
}

// Keep precision gates, pilot status and Richardson evidence mutually consistent.
void SWITCHES::check_adaptive_precision_consistency(const std::vector<int>& spacing_factors) const
{
  require_positive_mode2_tolerance(
      adaptive_grid_error_target_, "adaptive_grid_error_target");
  require_positive_mode2_tolerance(
      adaptive_grid_tau_curvature_, "adaptive_grid_tau_curvature");
  require_positive_mode2_tolerance(
      adaptive_grid_tau_field_gradient_,
      "adaptive_grid_tau_field_gradient");
  require_positive_mode2_tolerance(
      adaptive_grid_tau_richardson_, "adaptive_grid_tau_richardson");
  require_positive_mode2_tolerance(
      adaptive_grid_tau_tail_, "adaptive_grid_tau_tail");
  if (adaptive_grid_encounter_tail_fraction_ >= 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_encounter_tail_fraction",
        "must be zero (inherit tau_tail) or finite in (0,1)");
  if (adaptive_grid_coarsen_threshold_ > 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_coarsen_threshold",
        "must not exceed 1 in modes 2/3 because it is the normalized "
        "overlap-selection fraction");
  if (adaptive_grid_rho_floor_ <= 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_rho_floor",
        "is required in modes 2/3 and must be positive");
  if (adaptive_grid_field_floor_ <= 0.0)
    adaptive_grid_input_error(
        "adaptive_grid_field_floor",
        "is required in modes 2/3 and must be positive");
  if (adaptive_grid_richardson_interval_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_richardson_interval",
        "must be 1 in modes 2/3; skipped audits are not implemented");
  if (adaptive_grid_update_interval_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_update_interval",
        "must be 1 in modes 2/3; deferred updates are not implemented");
  if (adaptive_grid_min_neff_ < 1.0)
    adaptive_grid_input_error(
        "adaptive_grid_min_neff",
        "is required in modes 2/3 and must be at least 1");
  if (adaptive_grid_shrink_dwell_ != 1)
    adaptive_grid_input_error(
        "adaptive_grid_shrink_dwell",
        "must be 1 in modes 2/3; stateful hysteresis is not implemented");
  if (adaptive_grid_max_cells_ == 0 || adaptive_grid_max_bytes_ == 0)
    adaptive_grid_input_error(
        "adaptive_grid_max_cells/max_bytes",
        "must both be explicit positive capacities in modes 2/3");
  if (adaptive_grid_fail_on_unmet_error_ == 0)
    {
      if (adaptive_grid_pilot_diagnostics_ != 1)
        adaptive_grid_input_error(
            "adaptive_grid_fail_on_unmet_error",
            "may be 0 only with explicit "
            "adaptive_grid_pilot_diagnostics=1; otherwise modes 2/3 "
            "remain fail-closed");
      if (adaptive_grid_profile_ != "encounter")
        adaptive_grid_input_error(
            "adaptive_grid_profile",
            "must be encounter when "
            "adaptive_grid_pilot_diagnostics=1 so every unmet "
            "encounter remains auditable");
    }
  else if (adaptive_grid_pilot_diagnostics_ != 0)
    adaptive_grid_input_error(
        "adaptive_grid_pilot_diagnostics",
        "requires adaptive_grid_fail_on_unmet_error=0; do not mark a "
        "fail-closed production run as an unqualified pilot");
  if ((adaptive_grid_error_norm_policy_ ==
           "vector_rms_quantile_v1" ||
       adaptive_grid_error_norm_policy_ ==
           "vector_mixed_rms_quantile_v2") &&
      (adaptive_grid_richardson_order_source_ != "three_level" ||
       !adaptive_grid_richardson_order_table_.empty()))
    adaptive_grid_input_error(
        "adaptive_grid_error_norm_policy",
        "vector quantile policies require rolling three_level order "
        "and an empty calibrated-order table");
  if (adaptive_grid_noise_aware_stop_ != 0 &&
      ((adaptive_grid_error_norm_policy_ !=
            "vector_rms_quantile_v1" &&
        adaptive_grid_error_norm_policy_ !=
            "vector_mixed_rms_quantile_v2") ||
       adaptive_grid_pilot_diagnostics_ != 1 ||
       adaptive_grid_fail_on_unmet_error_ != 0 ||
       (adaptive_macroparticle_configuration_.mode !=
            feebl::adaptive_particles::Mode::SplitOnly &&
        adaptive_macroparticle_configuration_.mode !=
            feebl::adaptive_particles::Mode::CoupledSplitMerge)))
    adaptive_grid_input_error(
        "adaptive_grid_noise_aware_stop",
        "requires a vector quantile policy and the explicit "
        "unqualified pilot/root-replay contract (pilot_diagnostics=1, "
        "fail_on_unmet_error=0, adaptive_macroparticle_mode=2 or 3)");
  if (adaptive_grid_profile_ == "off")
    adaptive_grid_input_error(
        "adaptive_grid_profile",
        "must be summary or encounter in experimental modes 2/3");
  if (adaptive_grid_richardson_order_source_ == "calibrated_table")
    validate_adaptive_grid_order_table(
        adaptive_grid_richardson_order_table_, spacing_factors);
  else
    {
      if (!adaptive_grid_richardson_order_table_.empty())
        adaptive_grid_input_error(
            "adaptive_grid_richardson_order_table",
            "must be empty when order source is three_level");
      if (spacing_factors.size() < 3)
        adaptive_grid_input_error(
            "adaptive_grid_field_spacing_classes",
            "must contain at least three classes for three_level order estimation");
    }
  if (!adaptive_grid_schedule_save_.empty() &&
      !adaptive_grid_schedule_load_.empty())
    adaptive_grid_input_error(
        "adaptive_grid_schedule_save/load",
        "cannot both be set in the same run");
}

// Require enough declared capacity for minimum dimensions and guard cells.
void SWITCHES::check_adaptive_capacity_consistency() const
{
  const std::uint64_t minimum_x = adaptive_grid_min_nx_ > 0
      ? static_cast<std::uint64_t>(adaptive_grid_min_nx_) : 4U;
  const std::uint64_t minimum_y = adaptive_grid_min_ny_ > 0
      ? static_cast<std::uint64_t>(adaptive_grid_min_ny_) : 4U;
  if (minimum_x > adaptive_grid_max_cells_ / minimum_y)
    adaptive_grid_input_error(
        "adaptive_grid_max_cells",
        "cannot contain the configured minimum transverse dimensions");
  const std::uint64_t guard_extent =
      2U * static_cast<std::uint64_t>(adaptive_grid_extra_guard_cells_) + 2U;
  if ((adaptive_grid_max_nx_ > 0 &&
       guard_extent > static_cast<std::uint64_t>(adaptive_grid_max_nx_)) ||
      (adaptive_grid_max_ny_ > 0 &&
       guard_extent > static_cast<std::uint64_t>(adaptive_grid_max_ny_)))
    adaptive_grid_input_error(
        "adaptive_grid_extra_guard_cells",
        "leaves no two-cell physical interior within an explicit maximum");
}

// Check the one-patch geometry and terminal-certification capabilities.
void SWITCHES::check_adaptive_patch_consistency(const AdaptiveGridSizeClasses& classes) const
{
  if (adaptive_grid_blocking_factor_ <= 0)
    adaptive_grid_input_error(
        "adaptive_grid_blocking_factor",
        "is required and must be a positive power of two in mode 3");
  if (adaptive_grid_blocking_factor_ %
          adaptive_grid_refinement_ratio_ != 0)
    adaptive_grid_input_error(
        "adaptive_grid_blocking_factor",
        "must be divisible by adaptive_grid_refinement_ratio");
  if (adaptive_grid_error_buffer_cells_ <= 0)
    adaptive_grid_input_error(
        "adaptive_grid_error_buffer_cells",
        "is required and must be positive in mode 3");
  if (adaptive_grid_transition_cells_ <= 0 ||
      (adaptive_grid_transition_cells_ & 1) != 0)
    adaptive_grid_input_error(
        "adaptive_grid_transition_cells",
        "must be positive and even in the one-patch v1 contract");
  require_positive_mode2_tolerance(
      adaptive_grid_fine_residual_tolerance_,
      "adaptive_grid_fine_residual_tolerance");
  if (adaptive_grid_terminal_patch_certification_ != 0)
    {
      if (adaptive_grid_source_model_ !=
          "fixed_width_bspline1_v1")
        adaptive_grid_input_error(
            "adaptive_grid_terminal_patch_certification",
            "requires fixed_width_bspline1_v1; a finite reference "
            "raster cannot provide independent post-terminal "
            "source evidence");
      if (adaptive_grid_field_operator_ != "fd4_lagrange4_v1")
        adaptive_grid_input_error(
            "adaptive_grid_terminal_patch_certification",
            "requires fd4_lagrange4_v1");
      if (adaptive_grid_error_norm_policy_ !=
              "vector_rms_quantile_v1" ||
          adaptive_grid_richardson_order_source_ != "three_level")
        adaptive_grid_input_error(
            "adaptive_grid_terminal_patch_certification",
            "requires vector_rms_quantile_v1 with rolling "
            "three_level order estimation");
      if (adaptive_macroparticle_configuration_.mode !=
          feebl::adaptive_particles::Mode::LegacyOff)
        adaptive_grid_input_error(
            "adaptive_grid_terminal_patch_certification",
            "currently requires adaptive_macroparticle_mode=0; "
            "fresh-root replay remains an external recommendation");
      if (adaptive_grid_profile_ != "encounter")
        adaptive_grid_input_error(
            "adaptive_grid_terminal_patch_certification",
            "requires encounter telemetry while experimental");
    }
  for (int value : classes.x)
    if (value % adaptive_grid_blocking_factor_ != 0)
      adaptive_grid_input_error(
          "adaptive_grid_size_classes_x",
          "entries must be divisible by the mode-3 blocking factor");
  for (int value : classes.y)
    if (value % adaptive_grid_blocking_factor_ != 0)
      adaptive_grid_input_error(
          "adaptive_grid_size_classes_y",
          "entries must be divisible by the mode-3 blocking factor");
}

// Admit scheduler, field-solver and CPU mover combinations.
void SWITCHES::check_wavefront_dispatch_consistency() const
{
  if (wavefront_workspace_max_bytes_ != 0 &&
      collision_schedule_ != "wavefront_parallel")
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_workspace_max_bytes requires "
             "collision_schedule=wavefront_parallel; the requested "
             "workspace budget would otherwise be unused" << std::endl;
      std::exit(2);
    }
  if (wavefront_parallel_physics_ != 0 &&
      wavefront_parallel_physics_ != 1)
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_parallel_physics must be 0 or 1"
          << std::endl;
      std::exit(2);
    }
  if (sparse_cells_ != 0 && sparse_cells_ != 1)
    {
      std::cerr << "FEEBL: ERROR: sparse_cells must be 0 or 1"
                << std::endl;
      std::exit(2);
    }
  if (wavefront_field_solver_ != "fft" &&
      wavefront_field_solver_ != "direct" &&
      wavefront_field_solver_ != "auto")
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_field_solver must be fft, direct, or auto"
          << std::endl;
      std::exit(2);
    }
  if (sparse_direct_interaction_limit_ <= 0)
    {
      std::cerr
          << "FEEBL: ERROR: sparse_direct_interaction_limit must be positive"
          << std::endl;
      std::exit(2);
    }
  if (wavefront_field_solver_ != "fft" &&
      (collision_schedule_ != "wavefront_parallel" || sparse_cells_ != 1 ||
       integration_method != 2 || electron_distribution_rho != 2 ||
       interpolation != 2))
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_field_solver=direct/auto requires "
             "collision_schedule=wavefront_parallel, sparse_cells=1, "
             "integration_method=2, electron_distribution_rho=2, and "
             "interpolation=2. No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (collision_schedule_ != "legacy" &&
      collision_schedule_ != "wavefront_strict" &&
      collision_schedule_ != "wavefront_parallel")
    {
      std::cerr
          << "FEEBL: ERROR: collision_schedule must be legacy, "
             "wavefront_strict, or wavefront_parallel"
          << std::endl;
      std::exit(2);
    }
  if (cpu_primary_move_ != 0 && cpu_primary_move_ != 1)
    {
      std::cerr << "FEEBL: ERROR: cpu_primary_move must be 0 or 1"
                << std::endl;
      std::exit(2);
    }
  if (cpu_primary_move_ == 1 &&
      (interpolation != 2 || do_beamstrahlung_ != 0 ||
       ST_spin_flip_ != 0 || bmt_precession_ != 0 || do_trident != 0 ||
       do_coherent != 0 || do_prod != 0 || extra_grids != 0))
    {
      std::cerr
          << "FEEBL: ERROR: cpu_primary_move=1 requires interpolation=2, "
             "do_eloss=0, ST_spin_flip=0, bmt_precession=0, "
             "do_trident=0, do_coherent=0, do_prod=0, and grids=1. "
             "No fallback was performed."
          << std::endl;
      std::exit(2);
    }
}

// Check the physical-space luminosity grid and its aggregate-only contract.
void SWITCHES::check_spatial_luminosity_consistency() const
{
  if (electron_lumi_spatial_kernel_ != 0 &&
      electron_lumi_spatial_kernel_ != 1)
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_spatial_kernel must be 0 or 1"
          << std::endl;
      std::exit(2);
    }
  if (electron_lumi_spatial_kernel_ == 1)
    {
      if (electron_lumi_spatial_nx_ < 2 ||
          electron_lumi_spatial_ny_ < 2 ||
          !adaptive_grid_is_finite(electron_lumi_spatial_x_min_) ||
          !adaptive_grid_is_finite(electron_lumi_spatial_x_max_) ||
          !adaptive_grid_is_finite(electron_lumi_spatial_y_min_) ||
          !adaptive_grid_is_finite(electron_lumi_spatial_y_max_) ||
          !(electron_lumi_spatial_x_max_ >
            electron_lumi_spatial_x_min_) ||
          !(electron_lumi_spatial_y_max_ >
            electron_lumi_spatial_y_min_) ||
          electron_lumi_spatial_max_pairs_ == 0U ||
          electron_lumi_spatial_max_pairs_ >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()) ||
          !adaptive_grid_is_finite(
              electron_lumi_spatial_closure_tolerance_) ||
          electron_lumi_spatial_closure_tolerance_ <= 0.0 ||
          electron_lumi_spatial_closure_tolerance_ >= 1.0)
        {
          std::cerr
              << "FEEBL: ERROR: electron_lumi_spatial_kernel=1 requires "
                 "explicit nx,ny>=2, finite ordered x/y bounds, a positive "
                 "representable max_pairs, and a finite closure_tolerance "
                 "in (0,1). No fallback was performed."
              << std::endl;
          std::exit(2);
        }
      if (electron_ratio != 1.0F || do_isr != 0 ||
          (do_lumi & 1) != 0 || do_cross != 0 || do_bhabhas != 0 ||
          do_jets != 0 || do_prod != 0 || do_photon_lumi != 0 ||
          do_photons_[0] != 0 || do_photons_[1] != 0 ||
          load_photon != 0 || do_pairs != 0 || do_hadrons != 0 ||
          do_compt != 0 || do_compt_phot != 0 || do_muons != 0 ||
          do_coherent != 0 || do_trident != 0 || do_tertphot != 0 ||
          track_pairs != 0 || track_muons != 0 || store_pairs != 0 ||
          store_muons != 0 || store_hadrons != 0 || load_event != 0 ||
          electron_lumi_precompute_ != 0 ||
          electron_lumi_sampling_ != 0 || photon_lumi_factorized_ != 0 ||
          extra_grids != 0 || gpu_mode_ == 3)
        {
          std::cerr
              << "FEEBL: ERROR: electron_lumi_spatial_kernel=1 is a "
                 "restricted aggregate-only collision path and requires "
                 "electron_ratio=1, do_isr=0, (do_lumi&1)=0, no cross/"
                 "Bhabha/jets/prod, no photon/pair/hadron/Compton/muon/"
                 "coherent/trident collision paths, no loaded events, no "
                 "electron-luminosity precompute/sampling, no factorized "
                 "photon luminosity, grids=1, and no gpu_mode=3 resident "
                 "bypass. No fallback was performed."
              << std::endl;
          std::exit(2);
        }
    }
}

// Validate precomputed luminosity and event-sampling capabilities.
void SWITCHES::check_precompute_luminosity_consistency() const
{
  if (electron_lumi_precompute_ != 0 && electron_lumi_precompute_ != 1 &&
      electron_lumi_precompute_ != 2 && electron_lumi_precompute_ != 3)
    {
      std::cerr << "FEEBL: ERROR: electron_lumi_precompute must be 0, 1, 2, or 3"
                << std::endl;
      std::exit(2);
    }
  if (electron_lumi_precompute_ == 1 &&
      (do_isr != 0 || (do_lumi & 1) != 0 || do_cross != 0 ||
       do_bhabhas != 0 || do_jets != 0 || do_prod != 0 ||
       bmt_precession_ != 0))
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_precompute=1 requires "
             "do_isr=0, (do_lumi&1)=0, do_cross=0, do_bhabhas=0, "
             "do_jets=0, do_prod=0, and bmt_precession=0. "
             "No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (electron_lumi_precompute_ == 2 &&
      (do_isr != 0 || (do_lumi & 1) != 0 || do_cross != 0 ||
       do_bhabhas != 0 || do_jets != 0 || do_prod != 0 ||
       bmt_precession_ != 0 || ST_spin_flip_ != 0 ||
       do_bds_spin_rotation != 0))
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_precompute=2 requires "
             "do_isr=0, (do_lumi&1)=0, do_cross=0, do_bhabhas=0, "
             "do_jets=0, do_prod=0, bmt_precession=0, ST_spin_flip=0, "
             "and do_bds_spin_rotation=0. No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (electron_lumi_sampling_ != 0 && electron_lumi_sampling_ != 1 &&
      electron_lumi_sampling_ != 2 && electron_lumi_sampling_ != 3)
    {
      std::cerr << "FEEBL: ERROR: electron_lumi_sampling must be 0, 1, 2, or 3"
                << std::endl;
      std::exit(2);
    }
  if (electron_lumi_sampling_ != 0 && electron_lumi_precompute_ != 3)
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_sampling="
          << electron_lumi_sampling_ << " requires "
             "electron_lumi_precompute=3. No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (electron_lumi_sampling_ == 3 && wavefront_parallel_physics_ != 1)
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_sampling=3 requires "
             "wavefront_parallel_physics=1 so every encounter owns a "
             "private append-only sample heap. No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (electron_lumi_precompute_ == 3 &&
      (do_isr != 0 || do_lumi != 1 || do_cross != 0 ||
       do_bhabhas != 0 || do_jets != 0 || do_prod != 0 ||
       bmt_precession_ != 0 || ST_spin_flip_ != 0 ||
       do_bds_spin_rotation != 0 || do_coherent != 0 || do_trident != 0 ||
       extra_grids != 0))
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_precompute=3 requires "
             "do_isr=0, do_lumi=1, do_cross=0, do_bhabhas=0, "
             "do_jets=0, do_prod=0, bmt_precession=0, ST_spin_flip=0, "
             "do_bds_spin_rotation=0, do_coherent=0, and do_trident=0. "
             "The main grid must be selected with grids=1. "
             "No fallback was performed."
          << std::endl;
      std::exit(2);
    }
}

// Allow photon factorization only for side-effect-free aggregate features.
void SWITCHES::check_factorized_photon_consistency() const
{
  if (photon_lumi_factorized_ != 0 && photon_lumi_factorized_ != 1)
    {
      std::cerr << "FEEBL: ERROR: photon_lumi_factorized must be 0 or 1"
                << std::endl;
      std::exit(2);
    }
  if (photon_lumi_factorized_ == 1)
    {
      feebl::legacy_photon_lumi::Features features;
      features.do_photon_lumi = do_photon_lumi;
      features.do_lumi = do_lumi;
      features.do_jets = do_jets;
      features.do_compt = do_compt;
      features.do_compt_phot = do_compt_phot;
      features.do_pairs = do_pairs;
      features.do_muons = do_muons;
      features.do_hadrons = do_hadrons;
      features.do_coherent = do_coherent;
      features.do_trident = do_trident;
      features.do_cross = do_cross;
      features.do_bhabhas = do_bhabhas;
      features.do_prod = do_prod;
      features.do_tertphot = do_tertphot;
      std::string reason;
      if (!feebl::legacy_photon_lumi::eligible(features, &reason))
        {
          std::cerr
              << "FEEBL: ERROR: photon_lumi_factorized=1 is valid only for "
                 "the side-effect-free luminosity-aggregate feature family; "
              << reason << ". No fallback was performed." << std::endl;
          std::exit(2);
        }
    }
}

// Prevent worker-private physics from entering shared-state feature paths.
void SWITCHES::check_private_wavefront_consistency() const
{
  const bool spatial_wavefront_collision =
      electron_lumi_spatial_kernel_ == 1 &&
      electron_lumi_precompute_ == 0 && electron_lumi_sampling_ == 0 &&
      do_lumi == 0 && photon_lumi_factorized_ == 0;
  const bool sampled_wavefront_collision =
      electron_lumi_spatial_kernel_ == 0 &&
      electron_lumi_precompute_ == 3 &&
      (electron_lumi_sampling_ == 1 || electron_lumi_sampling_ == 2 ||
       electron_lumi_sampling_ == 3) &&
      do_lumi == 1 && photon_lumi_factorized_ == 1;
  if (wavefront_parallel_physics_ == 1 &&
      (collision_schedule_ != "wavefront_parallel" || sparse_cells_ != 1 ||
       (!spatial_wavefront_collision && !sampled_wavefront_collision)))
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_parallel_physics=1 requires "
             "collision_schedule=wavefront_parallel and sparse_cells=1, "
             "plus either the historical precompute=3 sampled-event "
             "contract or electron_lumi_spatial_kernel=1 with do_lumi=0 "
             "and no precompute/sampling/factorized photon path. "
             "No fallback was performed."
          << std::endl;
      std::exit(2);
    }
  if (wavefront_parallel_physics_ == 1 &&
      (electron_ratio != 1.0F || force_symmetric != 0 ||
       extra_grids != 0 || load_photon != 0 ||
       do_pairs != 0 || do_hadrons != 0 || do_compt != 0 ||
       do_compt_phot != 0 || do_muons != 0 || do_coherent != 0 ||
       do_trident != 0 || do_prod != 0 || do_bhabhas != 0 || do_jets != 0 ||
       do_isr != 0 || do_cross != 0 || do_tertphot != 0 ||
       track_pairs != 0 || track_muons != 0 || ST_spin_flip_ != 0 ||
       bmt_precession_ != 0 || do_bds_spin_rotation != 0 ||
       write_photons != 0 || store_beam != 0 || store_pairs != 0 ||
       store_muons != 0 || store_hadrons != 0 || jet_store != 0 ||
       load_event != 0 || do_dump != 0))
    {
      std::cerr
          << "FEEBL: ERROR: wavefront_parallel_physics=1 does not admit "
             "electron_ratio!=1, force_symmetric, extra grids, loaded "
             "photons, pair/strong-field/"
             "event generators, tracked secondaries, stochastic or legacy "
             "spin paths, or shared output sinks. No fallback was performed."
          << std::endl;
      std::exit(2);
    }
}

// Check secondary generators, tracking and output compatibility.
void SWITCHES::check_secondary_output_consistency() const
{
  if ( do_bhabhas && do_pairs)
    {
      std::cout << " do_pairs= " << do_pairs << " do_bhabhas = " << do_bhabhas << std::endl;
      std::cout << " ERROR : it is not allowed to have do_bhabhas= 1 together with do_pairs = 1 " << std::endl;
      exit(1);
    } 
  if ( do_bhabhas && do_compt)
    {
      std::cout << " do_pairs= " << do_pairs << " do_compt = " << do_compt << std::endl;
      std::cout << " ERROR : it is not allowed to have do_bhabhas= 1 together with do_compt = 1 " << std::endl;
      exit(1);
    } 
  //  if (store_secondaries && !track_secondaries)
  if (store_pairs==1 && !track_pairs)
    {
      //std::cerr << " store_secondaries = " << store_secondaries << " track_secondaries = " << track_secondaries << std::endl;
      //std::cerr << " WARNING : it is not very consistent to store secondaries without tracking them! " << std::endl;
      std::cout << " store_pairs = " << store_pairs << " track_pairs = " << track_pairs << std::endl;
      std::cout << " FEEBL: ERROR : store_pairs without track_pairs produces no usable output. " << std::endl;
      exit(1);
    }
  if (do_muons==0 && (track_muons || store_muons))
    {
      std::cout << " do_muons = " << do_muons << " store_muons = " << store_muons << " track_muons = " << track_muons << std::endl;
      std::cout << " FEEBL: ERROR : track/store muons requires do_muons = 1. " << std::endl;
      exit(1);
    }
  if (store_muons==1 && !track_muons)
    {
      std::cout << " store_muons = " << store_muons << " track_muons = " << track_muons << std::endl;
      std::cout << " FEEBL: ERROR : store_muons without track_muons produces no usable output. " << std::endl;
      exit(1);
    }
  if (track_muons==1 && muon_ecut<10*MUMASS)
    {
      std::cout << " track_muons = " << track_muons << " muon_ecut = "<< muon_ecut <<std::endl;
      std::cout << " WARNING : Tracking for low energy muons not verified functional " << std::endl;

    }

  if (cuts_from_loaded_beam > 0  && load_beam == 0) 
    {
      std::cout << " WARNING : the switch cuts_from_loaded_beam is without effect with load_beam = 0 " << std::endl;
    }
  if(do_tertphot && !(track_pairs || track_muons))
    {
      std::cout << " do_tertphot = " << do_tertphot << " track_pairs = "<< track_pairs << " track_muons = " << track_muons << std::endl;
      std::cout << " WARNING : Tertphots are produced by incoherent particles. They should be tracked." << std::endl;
    }
}


void SWITCHES::readTWOBEAM(const PARAMETERS& param)
{
  //  VALUE value;
  //  load_variable("twobeam",&value);
  //  twobeam=CONTENTS(value);
  twobeam=param.readIValue("twobeam");
}

// void SWITCHES::readCharge_sign_2(const PARAMETERS& param)
// {
//   //  VALUE value;
//   //  load_variable("charge_sign_2",&value);
//   //  charge_sign_2=CONTENTS(value);
//   charge_sign_2=param.readFValue("charge_sign_2");
// }

std::string SWITCHES::adaptive_grid_configuration_flow() const
{
  if (adaptive_grid_mode_ < 2) return std::string();
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "FEEBL_ADAPTIVE_GRID_CONFIG_V1"
      << " contract=mode23-card-v1"
      << " configuration_status=accepted"
      << " requested_mode=" << adaptive_grid_mode_
      << " actual_mode=pending_runtime_preflight"
      << " routing_status=cpu_wavefront_integrated"
      << " qualification_status=experimental"
      << " requested_backend=" << compute_backend_
      << " actual_backend=unresolved"
      << " gpu_mode=" << gpu_mode_
      << " allow_backend_fallback=" << allow_backend_fallback_
      << " scope=" << adaptive_grid_scope_
      << " controller=charge-curvature-field-richardson-overlap-v1"
      << " profile=" << adaptive_grid_profile_
      << " pilot_diagnostics=" << adaptive_grid_pilot_diagnostics_
      << " physics_qualification_claim=0"
      << " convergence_claim=0"
      << " unmet_outputs_authorized="
      << (adaptive_grid_pilot_diagnostics_ != 0 ? 1 : 0)
      << " terminal_unmet_policy="
      << (adaptive_grid_pilot_diagnostics_ != 0
              ? "publish_finest_resource_admissible_uniform_unqualified"
              : "abort_or_bounded_tail_budget")
      << " size_classes_x=" << std::quoted(adaptive_grid_size_classes_x_)
      << " size_classes_y=" << std::quoted(adaptive_grid_size_classes_y_)
      << " min_nx=" << adaptive_grid_min_nx_
      << " min_ny=" << adaptive_grid_min_ny_
      << " max_nx=" << adaptive_grid_max_nx_
      << " max_ny=" << adaptive_grid_max_ny_
      << " extra_guard_cells=" << adaptive_grid_extra_guard_cells_
      << " allow_full_window=" << adaptive_grid_allow_full_window_
      << " max_cells=" << adaptive_grid_max_cells_
      << " max_bytes=" << adaptive_grid_max_bytes_
      << " field_spacing_classes="
      << std::quoted(adaptive_grid_field_spacing_classes_)
      << " field_operator=" << adaptive_grid_field_operator_
      << " max_refinement_factor="
      << adaptive_grid_max_refinement_factor_
      << " anchor_x=" << adaptive_grid_anchor_x_
      << " anchor_y=" << adaptive_grid_anchor_y_
      << " monitor=" << adaptive_grid_monitor_
      << " error_target=" << adaptive_grid_error_target_
      << " error_norm_policy=" << adaptive_grid_error_norm_policy_
      << " error_quantile=" << adaptive_grid_error_quantile_
      << " error_catastrophe_limit="
      << adaptive_grid_error_catastrophe_limit_
      << " error_reference_field="
      << adaptive_grid_error_reference_field_
      << " error_absolute_floor_fraction="
      << adaptive_grid_error_absolute_floor_fraction_
      << " error_catastrophe_absolute_fraction="
      << adaptive_grid_error_catastrophe_absolute_fraction_
      << " error_exposure_limit="
      << adaptive_grid_error_exposure_limit_
      << " error_budget_sha256="
      << (adaptive_grid_error_budget_sha256_.empty()
              ? "not_applicable" : adaptive_grid_error_budget_sha256_)
      << " noise_aware_stop=" << adaptive_grid_noise_aware_stop_
      << " source_model=" << adaptive_grid_source_model_
      << " source_reference_nx=" << adaptive_grid_source_reference_nx_
      << " source_reference_ny=" << adaptive_grid_source_reference_ny_
      << " source_width_x=" << adaptive_grid_source_width_x_
      << " source_width_y=" << adaptive_grid_source_width_y_
      << " source_error_target=" << adaptive_grid_source_error_target_
      << " solver_error_target=" << adaptive_grid_solver_error_target_
      << " lumi_error_target=" << adaptive_grid_lumi_error_target_
      << " lumi_floor=" << adaptive_grid_lumi_floor_
      << " tau_curvature=" << adaptive_grid_tau_curvature_
      << " tau_field_gradient=" << adaptive_grid_tau_field_gradient_
      << " tau_richardson=" << adaptive_grid_tau_richardson_
      << " tau_tail=" << adaptive_grid_tau_tail_
      << " encounter_tail_fraction="
      << adaptive_grid_encounter_tail_fraction_
      << " rho_floor=" << adaptive_grid_rho_floor_
      << " field_floor=" << adaptive_grid_field_floor_
      << " smoothing_passes=" << adaptive_grid_indicator_smoothing_passes_
      << " indicator_reduction=" << adaptive_grid_indicator_reduction_
      << " refine_threshold=" << adaptive_grid_refine_threshold_
      << " coarsen_threshold=" << adaptive_grid_coarsen_threshold_
      << " shrink_dwell=" << adaptive_grid_shrink_dwell_
      << " update_interval=" << adaptive_grid_update_interval_
      << " richardson_interval=" << adaptive_grid_richardson_interval_
      << " richardson_order_source="
      << adaptive_grid_richardson_order_source_
      << " richardson_order_table="
      << std::quoted(adaptive_grid_richardson_order_table_)
      << " min_neff=" << adaptive_grid_min_neff_
      << " fail_on_unmet_error=" << adaptive_grid_fail_on_unmet_error_
      << " schedule_save=" << std::quoted(adaptive_grid_schedule_save_)
      << " schedule_load=" << std::quoted(adaptive_grid_schedule_load_);
  if (adaptive_grid_mode_ == 3)
    out << " max_levels=" << adaptive_grid_max_levels_
        << " max_patches=" << adaptive_grid_max_patches_
        << " refinement_ratio=" << adaptive_grid_refinement_ratio_
        << " blocking_factor=" << adaptive_grid_blocking_factor_
        << " error_buffer_cells=" << adaptive_grid_error_buffer_cells_
        << " transition_cells=" << adaptive_grid_transition_cells_
        << " fine_residual_tolerance="
        << adaptive_grid_fine_residual_tolerance_
        << " terminal_patch_certification="
        << adaptive_grid_terminal_patch_certification_;
  return out.str();
}

std::string SWITCHES::output_flow() const 
{
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << title(std::string("SWITCHES : "));
  out << "charge_sign = " << charge_sign << std::endl;
  if (incoming_map_reslice_ != 0)
    out << "incoming_map_reslice = " << incoming_map_reslice_ << std::endl;
  out << "bmt_precession = " << bmt_precession_ << std::endl;
  out << "ST_spin_flip = " << ST_spin_flip_ << std::endl;
  out << "do_bds_spin_rotation = " << do_bds_spin_rotation << std::endl;
  out << "automatic_grid_sizing = " << automatic_grid_sizing
      << " optimize_grids = " << optimize_grids << std::endl;
  if (adaptive_grid_mode_ != 0)
    {
      if (adaptive_grid_mode_ >= 2)
        out << "adaptive_grid_contract = mode23-card-v1"
            << " adaptive_grid_controller = "
               "charge-curvature-field-richardson-overlap-v1"
            << " adaptive_grid_configuration_status = accepted"
            << " adaptive_grid_actual_mode = pending_runtime_preflight"
            << " adaptive_grid_routing_status = cpu_wavefront_integrated"
            << " adaptive_grid_qualification_status = experimental"
            << std::endl;
      out << "adaptive_grid_mode_requested = " << adaptive_grid_mode_
          << " adaptive_grid_scope = " << adaptive_grid_scope_
          << " adaptive_grid_profile = " << adaptive_grid_profile_
          << " adaptive_grid_pilot_diagnostics = "
          << adaptive_grid_pilot_diagnostics_
          << std::endl;
      if (adaptive_grid_mode_ >= 2)
        out << "adaptive_grid_physics_qualification_claim = 0"
            << " adaptive_grid_convergence_claim = 0"
            << " adaptive_grid_unmet_outputs_authorized = "
            << (adaptive_grid_pilot_diagnostics_ != 0 ? 1 : 0)
            << " adaptive_grid_terminal_unmet_policy = "
            << (adaptive_grid_pilot_diagnostics_ != 0
                    ? "publish_finest_resource_admissible_uniform_unqualified"
                    : "abort_or_bounded_tail_budget")
            << std::endl;
      out << "adaptive_grid_size_classes_x = \""
          << adaptive_grid_size_classes_x_
          << "\" adaptive_grid_size_classes_y = \""
          << adaptive_grid_size_classes_y_ << "\""
          << " adaptive_grid_min_nx = " << adaptive_grid_min_nx_
          << " adaptive_grid_min_ny = " << adaptive_grid_min_ny_
          << " adaptive_grid_max_nx = " << adaptive_grid_max_nx_
          << " adaptive_grid_max_ny = " << adaptive_grid_max_ny_
          << " adaptive_grid_extra_guard_cells = "
          << adaptive_grid_extra_guard_cells_
          << " adaptive_grid_allow_full_window = "
          << adaptive_grid_allow_full_window_
          << std::endl;
      out << "adaptive_grid_max_cells = " << adaptive_grid_max_cells_
          << " adaptive_grid_max_bytes = " << adaptive_grid_max_bytes_
          << " adaptive_grid_schedule_save = \""
          << adaptive_grid_schedule_save_
          << "\" adaptive_grid_schedule_load = \""
          << adaptive_grid_schedule_load_ << "\""
          << std::endl;
      if (adaptive_grid_mode_ >= 2)
        {
          out << "adaptive_grid_field_spacing_classes = \""
              << adaptive_grid_field_spacing_classes_ << "\""
              << " adaptive_grid_field_operator = "
              << adaptive_grid_field_operator_
              << " adaptive_grid_max_refinement_factor = "
              << adaptive_grid_max_refinement_factor_
              << " adaptive_grid_anchor_x = " << adaptive_grid_anchor_x_
              << " adaptive_grid_anchor_y = " << adaptive_grid_anchor_y_
              << " adaptive_grid_monitor = " << adaptive_grid_monitor_
              << " adaptive_grid_error_target = "
              << adaptive_grid_error_target_
              << " adaptive_grid_error_norm_policy = "
              << adaptive_grid_error_norm_policy_
              << " adaptive_grid_error_quantile = "
              << adaptive_grid_error_quantile_
              << " adaptive_grid_error_catastrophe_limit = "
              << adaptive_grid_error_catastrophe_limit_
              << " adaptive_grid_error_reference_field = "
              << adaptive_grid_error_reference_field_
              << " adaptive_grid_error_absolute_floor_fraction = "
              << adaptive_grid_error_absolute_floor_fraction_
              << " adaptive_grid_error_catastrophe_absolute_fraction = "
              << adaptive_grid_error_catastrophe_absolute_fraction_
              << " adaptive_grid_error_exposure_limit = "
              << adaptive_grid_error_exposure_limit_
              << " adaptive_grid_error_budget_sha256 = \""
              << adaptive_grid_error_budget_sha256_ << "\""
              << " adaptive_grid_noise_aware_stop = "
              << adaptive_grid_noise_aware_stop_
              << std::endl;
          out << "adaptive_grid_tau_curvature = "
              << adaptive_grid_tau_curvature_
              << " adaptive_grid_tau_field_gradient = "
              << adaptive_grid_tau_field_gradient_
              << " adaptive_grid_tau_richardson = "
              << adaptive_grid_tau_richardson_
              << " adaptive_grid_tau_tail = " << adaptive_grid_tau_tail_
              << " adaptive_grid_encounter_tail_fraction = "
              << adaptive_grid_encounter_tail_fraction_
              << " adaptive_grid_rho_floor = " << adaptive_grid_rho_floor_
              << " adaptive_grid_field_floor = "
              << adaptive_grid_field_floor_
              << std::endl;
          out << "adaptive_grid_indicator_smoothing_passes = "
              << adaptive_grid_indicator_smoothing_passes_
              << " adaptive_grid_indicator_reduction = "
              << adaptive_grid_indicator_reduction_
              << " adaptive_grid_refine_threshold = "
              << adaptive_grid_refine_threshold_
              << " adaptive_grid_coarsen_threshold = "
              << adaptive_grid_coarsen_threshold_
              << " adaptive_grid_shrink_dwell = "
              << adaptive_grid_shrink_dwell_
              << " adaptive_grid_update_interval = "
              << adaptive_grid_update_interval_
              << " adaptive_grid_richardson_interval = "
              << adaptive_grid_richardson_interval_ << std::endl;
          out << "adaptive_grid_richardson_order_source = "
              << adaptive_grid_richardson_order_source_
              << " adaptive_grid_richardson_order_table = \""
              << adaptive_grid_richardson_order_table_ << "\""
              << " adaptive_grid_min_neff = " << adaptive_grid_min_neff_
              << " adaptive_grid_fail_on_unmet_error = "
              << adaptive_grid_fail_on_unmet_error_ << std::endl;
        }
      if (adaptive_grid_mode_ >= 3)
        out << "adaptive_grid_max_levels = " << adaptive_grid_max_levels_
            << " adaptive_grid_max_patches = " << adaptive_grid_max_patches_
            << " adaptive_grid_refinement_ratio = "
            << adaptive_grid_refinement_ratio_
            << " adaptive_grid_blocking_factor = "
            << adaptive_grid_blocking_factor_
            << " adaptive_grid_error_buffer_cells = "
            << adaptive_grid_error_buffer_cells_
            << " adaptive_grid_transition_cells = "
            << adaptive_grid_transition_cells_
            << " adaptive_grid_fine_residual_tolerance = "
            << adaptive_grid_fine_residual_tolerance_
            << " adaptive_grid_terminal_patch_certification = "
            << adaptive_grid_terminal_patch_certification_ << std::endl;
    }
  if (!do_photon_lumi)
    out << " do_photon_lumi = " << do_photon_lumi << std::endl;
  if (electron_lumi_precompute_ == 1)
    out << " electron_lumi_precompute = 1"
        << " (ordered-invariant-precompute-v1)" << std::endl;
  else if (electron_lumi_precompute_ == 2)
    out << " electron_lumi_precompute = 2"
        << " (ordered-blocked-soa-simd-v1; block=128; "
           "commit=beam1-major-beam2-minor)"
        << std::endl;
  else if (electron_lumi_precompute_ == 3)
    out << " electron_lumi_precompute = 3"
        << " (parallel-blocked-soa-heap-v1; evaluate=static-p1-lanes; "
           "result-commit=beam1-major-beam2-minor; sampling="
        << (electron_lumi_sampling_ == 1
                ? "counter-weighted-reservoir-v2"
                : (electron_lumi_sampling_ == 2
                       ? "counter-keyed-bernoulli-hard-cap-v1"
                       : (electron_lumi_sampling_ == 3
                              ? "counter-geometric-gap-hard-cap-v1"
                              : "legacy-stream-v1")))
        << ")" << std::endl;
  if (electron_lumi_spatial_kernel_ == 1)
    out << " electron_lumi_spatial_kernel = 1"
        << " (fixed-physical-cic-v1; nx="
        << electron_lumi_spatial_nx_
        << "; ny=" << electron_lumi_spatial_ny_
        << "; x_min_nm=" << electron_lumi_spatial_x_min_
        << "; x_max_nm=" << electron_lumi_spatial_x_max_
        << "; y_min_nm=" << electron_lumi_spatial_y_min_
        << "; y_max_nm=" << electron_lumi_spatial_y_max_
        << "; max_pairs=" << electron_lumi_spatial_max_pairs_
        << "; closure_tolerance="
        << electron_lumi_spatial_closure_tolerance_
        << "; side_effect_scope=aggregate-only; runtime=cpu-reference)"
        << std::endl;
  if (photon_lumi_factorized_)
    out << " photon_lumi_factorized = " << photon_lumi_factorized_
        << " (factorized-exact-v2)" << std::endl;
  if (wavefront_parallel_physics_)
    out << " wavefront_parallel_physics = 1"
        << " (encounter-local-full-host-physics-v1; "
           "publication=canonical-barrier)" << std::endl;
  if (cpu_primary_move_)
    out << " cpu_primary_move = " << cpu_primary_move_
        << " (pool-static-primary-v1)" << std::endl;
  if (collision_schedule_ != "legacy")
    {
      const bool mode3_resident =
          actual_collision_schedule_.find("gpu_mode3_") == 0;
      const bool gpu4_parallel_host_physics =
          actual_collision_schedule_ ==
          "gpu_mode4_wave_batched_fields_parallel_host_physics";
      const bool gpu5_adaptive_fields =
          actual_collision_schedule_.find("gpu_mode5_adaptive_fields_") == 0;
      const bool gpu5_parallel_host_physics =
          actual_collision_schedule_ ==
          "gpu_mode5_adaptive_fields_parallel_host_physics";
      const bool cpu_parallel_host_physics =
          actual_collision_schedule_ == "wavefront_parallel_full_host_physics";
      const bool parallel_host_physics =
          gpu4_parallel_host_physics || gpu5_parallel_host_physics ||
          cpu_parallel_host_physics;
      const bool unordered_canonical_publication =
          mode3_resident || parallel_host_physics;
      out << "collision_schedule_requested = " << collision_schedule_
        << " collision_schedule_actual = " << actual_collision_schedule_
        << " scheduler = "
        << (mode3_resident
                ? "encounter-dag-resident-device-all-waves-v1"
                : (gpu5_adaptive_fields
                       ? "encounter-dag-adaptive-class-planmany-fields-v1"
                : (gpu4_parallel_host_physics
                       ? "encounter-dag-wave-batched-planmany-fields-v1"
                       : (cpu_parallel_host_physics
                              ? "encounter-dag-worker-private-host-physics-v1"
                              : "encounter-dag-antidiagonal-v1"))))
        << " diagonals = " << collision_diagonals_
        << " waves = " << collision_waves_
        << " batches = " << collision_batches_
        << " encounters = " << collision_encounters_
        << " maximum_wave_width = " << collision_maximum_wave_width_
        << " parallel_physics_encounters = "
        << collision_parallel_physics_encounters_
        << " prefetched_field_encounters = "
        << collision_prefetched_field_encounters_
        << " parallel_field_encounters = "
        << collision_parallel_field_encounters_
        << " full_parallel_make_step_encounters = "
        << collision_full_parallel_make_step_encounters_
        << " field_workers = " << collision_field_workers_
        << " maximum_nonempty_field_lanes = "
        << collision_maximum_nonempty_field_lanes_
        << " field_setup_ms = " << collision_field_setup_ms_
        << " field_prefetch_ms = " << collision_field_prefetch_ms_
        << " ordered_commit_ms = " << collision_ordered_commit_ms_
        << " field_dense_clear_visits = "
        << collision_field_dense_clear_visits_
        << " field_actual_clear_visits = "
        << collision_field_actual_clear_visits_
        << " field_dense_dot_visits = "
        << collision_field_dense_dot_visits_
        << " field_actual_dot_visits = "
        << collision_field_actual_dot_visits_
        << " field_dispatch_batches = "
        << collision_field_dispatch_batches_
        << " workspace_bytes_each_estimate = "
        << collision_workspace_bytes_each_estimate_
        << " workspace_bytes_reserved_estimate = "
        << collision_workspace_bytes_reserved_estimate_
        << " backend_threads = " << collision_backend_threads_
        << " ordered_commit = "
        << (unordered_canonical_publication ? 0 : 1)
        << " resident_all_waves = " << (mode3_resident ? 1 : 0)
        << " physics_equivalence_scope = "
        << (mode3_resident
                ? "primary-x-y-vx-vy-energy-upsilon+fine-density-lumi"
                : (parallel_host_physics
                       ? "fccz-primary-beamstrahlung-photon-full-ee-luminosity"
                       : "legacy-complete"))
        << " pair-aggregate-luminosity = "
        << (mode3_resident
                ? "unavailable"
                : (parallel_host_physics ? "full-ee" : "legacy"))
        << std::endl;
    }
  if (sparse_cells_)
    out << "sparse_cells = 1"
        << " cell_traversal = active-index-v1"
        << " ordering = ascending-cell/stable-insertion" << std::endl;
  if (wavefront_workspace_max_bytes_ != 0)
    out << "wavefront_workspace_max_bytes = "
        << wavefront_workspace_max_bytes_ << std::endl;
  if (wavefront_field_solver_ != "fft" ||
      sparse_direct_interaction_limit_ != 262144)
    out << "wavefront_field_solver = " << wavefront_field_solver_
        << " sparse_direct_interaction_limit = "
        << sparse_direct_interaction_limit_
        << " source_order = ascending-linear-cell"
        << " kernel = case2-integrated-green-v1" << std::endl;
  out << "integration_method = " << integration_method << " force_symmetric = " << force_symmetric << std::endl;
  out <<  "rndm_load = " << rndm_load << "rndm_save = " << rndm_save << " rndm_seed = " << rndm_seed << std::endl;
  // Preserve the historical default protocol. Accelerated runs record both
  // requested and actual execution state, including any explicit fallback.
  if (compute_backend_ != "serial" || sparse_cells_ != 0 ||
      wavefront_field_solver_ != "fft" ||
      sparse_direct_interaction_limit_ != 262144 ||
      cpu_threads_ != 0 ||
      cpu_variant_ != "auto" || cpu_primary_move_ != 0 ||
      gpu_variant_ != "auto" ||
      gpu_device_ != 0 || gpu_min_particles_ != 262144 ||
      gpu_mode_ != 1 || gpu2_scope_ != "auto" ||
      gpu2_deposition_ != "auto" || gpu2_fft_ != "cufft" ||
      gpu2_streams_ != 0 || gpu2_graphs_ != "auto" ||
      gpu2_fusion_ != "auto" || gpu2_batch_slices_ != 0 ||
      gpu2_pinned_io_ != "auto" || gpu2_host_threads_ != 0 ||
      gpu2_primary_mover_ != "serial" ||
      gpu2_memory_fraction_ != 0.80 || gpu2_profile_ != "summary" ||
      rng_backend_ != "legacy" || reduction_mode_ != "strict" ||
      allow_backend_fallback_ != 0 || engine_ != "legacy" ||
      algorithm_ != "exact" || precision_ != "fp64" ||
      modern_reduction_ != "deterministic" || modern_batch_width_ != 0 ||
      modern_cpu_workers_ != 0 || modern_cpu_runtime_ != "auto" ||
      modern_cpu_affinity_ != "auto" || modern_gpu_streams_ != 0 ||
      modern_gpu_graphs_ != "auto" || modern_trace_ != "off")
    {
      const PARALLEL_BACKEND& backend = parallel_backend();
      const std::string& actual_backend = backend.configured()
          ? backend.actual_backend() : actual_compute_backend_;
      const std::string& actual_cpu_variant = backend.configured()
          ? backend.actual_cpu_variant() : actual_cpu_variant_;
      const int actual_cpu_threads = backend.configured()
          ? backend.actual_cpu_threads() : actual_cpu_threads_;
      const std::string& fallback = backend.configured()
          ? backend.fallback_reason() : backend_fallback_reason_;
      out << "compute_backend_requested = " << compute_backend_
          << " compute_backend_actual = " << actual_backend << std::endl;
      out << "cpu_variant_requested = " << cpu_variant_
          << " cpu_variant_actual = " << actual_cpu_variant
          << " cpu_threads_requested = " << cpu_threads_
          << " cpu_threads_actual = " << actual_cpu_threads
          << " reduction_mode = " << reduction_mode_ << std::endl;
      out << "gpu_variant_requested = " << gpu_variant_
          << " gpu_variant_actual = " << backend.actual_gpu_variant()
          << " gpu_device_requested = " << gpu_device_
          << " gpu_min_particles = " << gpu_min_particles_
          << " rng_backend = " << rng_backend_ << std::endl;
      out << "gpu_mode_requested = " << gpu_mode_
          << " gpu_mode_actual = " << backend.actual_gpu_mode() << std::endl;
      if (engine_ != "legacy" || algorithm_ != "exact" ||
          precision_ != "fp64" || modern_reduction_ != "deterministic" ||
          modern_batch_width_ != 0 || modern_cpu_workers_ != 0 ||
          modern_cpu_runtime_ != "auto" || modern_cpu_affinity_ != "auto" ||
          modern_gpu_streams_ != 0 || modern_gpu_graphs_ != "auto" ||
          modern_trace_ != "off")
        {
          out << "engine_requested = " << engine_
              << " engine_actual = " << backend.actual_engine() << std::endl;
          out << "algorithm = " << algorithm_
              << " precision = " << precision_
              << " modern_reduction = " << modern_reduction_
              << " modern_batch_width = " << modern_batch_width_ << std::endl;
          out << "modern_cpu_workers = " << modern_cpu_workers_
              << " modern_cpu_runtime = " << modern_cpu_runtime_
              << " modern_cpu_affinity = " << modern_cpu_affinity_
              << " modern_gpu_streams = " << modern_gpu_streams_
              << " modern_gpu_graphs = " << modern_gpu_graphs_
              << " modern_trace = " << modern_trace_ << std::endl;
        }
      out << "gpu2_scope = " << gpu2_scope_
          << " gpu2_deposition = " << gpu2_deposition_
          << " gpu2_fft = " << gpu2_fft_
          << " gpu2_streams = " << gpu2_streams_
          << " gpu2_graphs = " << gpu2_graphs_
          << " gpu2_fusion = " << gpu2_fusion_
          << " gpu2_batch_slices = " << gpu2_batch_slices_
          << " gpu2_pinned_io = " << gpu2_pinned_io_
          << " gpu2_host_threads = " << gpu2_host_threads_
          << " gpu2_primary_mover = " << gpu2_primary_mover_
          << " gpu2_memory_fraction = " << gpu2_memory_fraction_
          << " gpu2_profile = " << gpu2_profile_ << std::endl;
      if (!fallback.empty()) out << "backend_fallback = " << fallback << std::endl;
      out << "backend_provenance = " << backend.provenance() << std::endl;
    }
  out << "do_photons.1 = " << do_photons_[0] << " do_photons.2 = " << do_photons_[1] << std::endl;
  out << "write_photons = " << write_photons << std::endl;
  out << "do_comp = " << do_compt << " do_prod = " << do_prod << std::endl;
  out << "electron_ratio = " << electron_ratio << std::endl;
  out << "compt_x_min = " << compt_x_min << " compt_emax = " << compt_emax << " GeV ; compt_scale = " << compt_scale << std::endl;
  out << "do_lumi = " << do_lumi << "num_lumi = " << num_lumi << " lumi_p = " << lumi_p << std::endl;
  out << "do_cross = " << do_cross << " do_isr = "<< do_isr << " do_espread = " << do_espread << std::endl;
  out << "photon_ratio = " << photon_ratio << std::endl;
  out << "do_hadrons = " << do_hadrons << " store_hadrons = " << store_hadrons << " hadron_ratio = " << hadron_ratio << std::endl;
  out << "do_jets = " << do_jets << " store_jets = " << jet_store << std::endl;
  out << "do_pairs = " << do_pairs << " load_events = " << load_event;
  if (imported_spin_mode_ != 0)
    out << " imported_spin_mode = " << imported_spin_mode_;
  out << std::endl;
  out << "do_bhabhas = " << do_bhabhas << std::endl;
  //out << "track_secondaries = " << track_secondaries << " pair_step = " << pair_step << std::endl;
  //out << "store_secondaries = " << store_secondaries << std::endl;
  out << "track_pairs = " << track_pairs << " pair_step = " << pair_step << std::endl;
  out << "store_pairs = " << store_pairs << std::endl;
  out << "bhabha_scal = " << bhabha_scal << " bhabha_ecmload = " << bhabha_ecmload << " GeV " <<  std::endl;
  out << "do_muons = " << do_muons << " track_muons = " << track_muons << " store_muons = " << store_muons << std::endl;
  out << "muon_ratio = " << muon_ratio << " muon_scale = " << muon_scale << " muon_ecut = " << muon_ecut <<std::endl;
  out << "do_coherent = " << do_coherent << std::endl;
  out << "do_trident = " << do_trident << std::endl;
  out << "emin = " << emin << std::endl;
  out << "grids = " << extra_grids+1 << std::endl;
  out << "pair_ecut = " << pair_ecut << " GeV " << std::endl;
  out << "pair_ratio = " << pair_ratio << std::endl;
  out << "bhabha_ratio = " << bhabha_ratio << std::endl;
  out << "pair_q2 = " << pair_q2 << std::endl;
  // printed only when not legacy, so that pair_model=0 output (and the regression references) is unchanged
  if (pair_model != 0)
    out << "pair_model = " << pair_model << std::endl;
  if (pair_model == 2)
    out << "pair_exact_rr = " << pair_exact_rr << std::endl;
  if (photon_polarisation != 0)
    out << "photon_polarisation = " << photon_polarisation << std::endl;
  if (coherent_model != 0)
    out << "coherent_model = " << coherent_model << std::endl;
  out << "beam_pair = " << beam_pair << std::endl;
  out << "jet_ratio = " << jet_ratio << std::endl;
  out << "jet_ptmin = " << jet_pstar << std::endl;
  out << "jet_pythia = " << jet_pythia << std::endl;
  out << "jet_log = " << jet_select << std::endl;
  out << "beam_size = " << geom << " beam_size_scale = " << r_scal << " ext_field = " << ext_field << std::endl;
  // printed only when not legacy (as pair_model)
  if (photon_timing != 0)
    out << "photon_timing = " << photon_timing << std::endl;
  if (photon_angle != 0)
    out << "photon_angle = " << photon_angle << std::endl;
  out << "espread.1 = " << espread1 << " which_espread.1 = " << which_espread1 << " espread.2 = " << espread2 << " which_espread.2 = " << which_espread2 << std::endl;
  out << "f_rep = " << f_rep << " n_b = " << n_b << std::endl;
  out << "do_edm4hep = " << do_edm4hep << std::endl;

  return out.str();
}
