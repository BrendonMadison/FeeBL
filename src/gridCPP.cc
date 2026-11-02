#include <iostream>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include "gridCPP.h"
#include "mathconst.h"
#include "physconst.h"
#include "mathematicalTools.h"
#include "physicalTools.h"
#include "qedbg_impact_parameter.h"
#include "qedbg_photon_angle.h"
#include "qedbg_photon_domain.h"
#include "qedbg_photon_stokes.h"
#include "parallelBackendCPP.h"
#include "legacyPhotonLumiFastCPP.h"
#include "legacyElectronLumiBlockedCPP.h"
#include "electronLumiBernoulliSkipCPP.h"
#ifdef FEEBL_HAVE_OPENMP
#include <omp.h>
#endif

namespace
{
bool electron_lumi_spatial_is_finite(double value)
{
  static_assert(std::numeric_limits<double>::is_iec559,
                "spatial luminosity runtime requires IEEE-754 binary64");
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "spatial luminosity runtime requires 64-bit double");
  std::uint64_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

std::uint64_t electron_lumi_spatial_sequence_hash(
    std::uint64_t geometry_hash,
    const std::vector<std::uint64_t>& plan_hashes)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto add = [&hash](std::uint64_t word) {
    for (unsigned int byte = 0U; byte < 8U; ++byte)
      {
        hash ^= (word >> (8U * byte)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
      }
  };
  add(UINT64_C(0x464545424c535031)); // "FEEBLSP1"
  add(geometry_hash);
  add(static_cast<std::uint64_t>(plan_hashes.size()));
  for (std::size_t index = 0U; index < plan_hashes.size(); ++index)
    add(plan_hashes[index]);
  return hash;
}
}

GENERAL_GRID::GENERAL_GRID()
{
  n_cell_x_=32;
  n_cell_y_=32;
  n_cell_z_=32;
  timestep_ =10;

  min_x_ = 0.0;
  max_x_ = 0.0; 
  min_y_ = 0.0;
  max_y_ = 0.0;
  min_z_ = 0.0;
  max_z_ = 0.0;

  cut_x_ = 0.0;
  cut_y_ = 0.0;
  cut_z_ = 0.0;

  integration_method_ = 0;
  step_ = 0.0;


  rho_x_1_ = 0.0;
  rho_y_1_ = 0.0;
  rho_sum_1_ = 0.0;
  rho_x_2_ = 0.0;
  rho_y_2_ = 0.0;
  rho_sum_2_ = 0.0;

  delta_x_inverse_ = 0.0;
  delta_y_inverse_ = 0.0;

  rho_factor_ = 0.0;
  field_kernel_factor_ = 0.0;

}

void GENERAL_GRID::interpolePotential(float xpart,float ypart, PHI_FLOAT& h_x, PHI_FLOAT& h_y, PHI_FLOAT& phi1_x, PHI_FLOAT& phi2_x, PHI_FLOAT& phi3_x, PHI_FLOAT& phi1_y,PHI_FLOAT& phi2_y, PHI_FLOAT& phi3_y, const PHI_FLOAT *phi) const
{
  int i1, i2;
  float h;
  absoluteCoordinates(xpart, ypart,h_x, h_y);

  // locate in Y
  cellLocalization(h_x, h_y, i1, i2, h);

  phiValuesY(i1, i2,h, phi, phi1_y, phi2_y, phi3_y);

  // locate in X
  cellLocalization(h_y, h_x,i2,i1,h); 


  phiValuesX(i1, i2, h, phi, phi1_x, phi2_x,phi3_x );


  h_x -= floor(h_x);
  h_y -= floor(h_y);
}

// electric field in GV/nm
TRIVECTOR GENERAL_GRID::ElectricFieldCIC(float xpart,float ypart, const PHI_FLOAT *phi) const
{
  //  int k;
  PHI_FLOAT phi1_x,phi2_x,phi3_x,phi1_y,phi2_y,phi3_y,h_x,h_y;
  TRIVECTOR Efield;
  interpolePotential(xpart, ypart, h_x, h_y, phi1_x, phi2_x, phi3_x, phi1_y, phi2_y, phi3_y, phi);

  Efield(0) = (h_x*(phi1_y-phi2_y)+(1.0-h_x)*(phi2_y-phi3_y))*delta_x_inverse_;
  Efield(1) = (h_y*(phi1_x-phi2_x)+(1.0-h_y)*(phi2_x-phi3_x))*delta_y_inverse_;

  Efield(2) = 0.0;
  // the potential seems to have been multiplied by 2 (for taking into 
  // account the magnetic field)
  Efield *= 0.5;
  return Efield;
}

TRIVECTOR GENERAL_GRID::ElectricFieldCICTranslated(
    float xpart, float ypart, const PHI_FLOAT* local_phi,
    int local_n_x, int local_n_y, int origin_x, int origin_y) const
{
  if (local_phi == NULL || local_n_x < 3 || local_n_y < 3 ||
      origin_x < 0 || origin_y < 0 ||
      origin_x > n_cell_x_ - local_n_x ||
      origin_y > n_cell_y_ - local_n_y)
    throw std::invalid_argument(
        "translated CIC field received an invalid canonical window");

  PHI_FLOAT absolute_x = 0.0;
  PHI_FLOAT absolute_y = 0.0;
  int y_base_x = 0;
  int y_base_y = 0;
  int x_base_x = 0;
  int x_base_y = 0;
  float y_weight = 0.0f;
  float x_weight = 0.0f;
  absoluteCoordinates(xpart, ypart, absolute_x, absolute_y);
  cellLocalization(
      absolute_x, absolute_y, y_base_x, y_base_y, y_weight);
  cellLocalization(
      absolute_y, absolute_x, x_base_y, x_base_x, x_weight);
  absolute_x -= floor(absolute_x);
  absolute_y -= floor(absolute_y);
  return ElectricFieldCICTranslatedCanonical(
      local_phi, local_n_x, local_n_y, origin_x, origin_y,
      y_base_x, y_base_y, y_weight,
      x_base_x, x_base_y, x_weight, absolute_x, absolute_y);
}

TRIVECTOR GENERAL_GRID::ElectricFieldCICTranslatedCanonical(
    const PHI_FLOAT* local_phi, int local_n_x, int local_n_y,
    int origin_x, int origin_y,
    int y_base_x, int y_base_y, float y_weight,
    int x_base_x, int x_base_y, float x_weight,
    PHI_FLOAT fractional_x, PHI_FLOAT fractional_y) const
{
  if (local_phi == NULL || local_n_x < 3 || local_n_y < 3 ||
      origin_x < 0 || origin_y < 0 ||
      origin_x > n_cell_x_ - local_n_x ||
      origin_y > n_cell_y_ - local_n_y ||
      !std::isfinite(fractional_x) || !std::isfinite(fractional_y))
    throw std::invalid_argument(
        "canonical translated CIC field received an invalid window");
  const auto local_index = [=](int canonical_x, int canonical_y) {
    const int local_x = canonical_x - origin_x;
    const int local_y = canonical_y - origin_y;
    if (local_x < 0 || local_x >= local_n_x ||
        local_y < 0 || local_y >= local_n_y)
      throw std::out_of_range(
          "translated CIC field stencil is outside its canonical window");
    return static_cast<std::size_t>(local_x) *
               static_cast<std::size_t>(local_n_y) +
           static_cast<std::size_t>(local_y);
  };

  PHI_FLOAT phi1_x = 0.0;
  PHI_FLOAT phi2_x = 0.0;
  PHI_FLOAT phi3_x = 0.0;
  PHI_FLOAT phi1_y = 0.0;
  PHI_FLOAT phi2_y = 0.0;
  PHI_FLOAT phi3_y = 0.0;
  float h_p = 0.0f;

  // Preserve GENERAL_GRID::interpolePotential exactly: Y interpolation
  // first, then X, with float weights and PHI_FLOAT potential arithmetic.
  h_p = 1.0 - y_weight;
  phi1_y = y_weight *
               local_phi[local_index(y_base_x + 1, y_base_y + 1)] +
           h_p * local_phi[local_index(y_base_x + 1, y_base_y)];
  phi2_y = y_weight * local_phi[local_index(y_base_x, y_base_y + 1)] +
           h_p * local_phi[local_index(y_base_x, y_base_y)];
  phi3_y = y_weight *
               local_phi[local_index(y_base_x - 1, y_base_y + 1)] +
           h_p * local_phi[local_index(y_base_x - 1, y_base_y)];

  h_p = 1.0 - x_weight;
  phi1_x = x_weight *
               local_phi[local_index(x_base_x + 1, x_base_y + 1)] +
           h_p * local_phi[local_index(x_base_x, x_base_y + 1)];
  phi2_x = x_weight * local_phi[local_index(x_base_x + 1, x_base_y)] +
           h_p * local_phi[local_index(x_base_x, x_base_y)];
  phi3_x = x_weight *
               local_phi[local_index(x_base_x + 1, x_base_y - 1)] +
           h_p * local_phi[local_index(x_base_x, x_base_y - 1)];

  TRIVECTOR field;
  field(0) = (fractional_x * (phi1_y - phi2_y) +
              (1.0 - fractional_x) * (phi2_y - phi3_y)) *
             delta_x_inverse_;
  field(1) = (fractional_y * (phi1_x - phi2_x) +
              (1.0 - fractional_y) * (phi2_x - phi3_x)) *
             delta_y_inverse_;
  field(2) = 0.0;
  field *= 0.5;
  return field;
}

void GENERAL_GRID::computeFields(int integrationMethod, float charge_sign, PHI_FLOAT *sor_parameter) 
    {
      // (see commentaries in init_grid_phys() )

      //int nn[2];
      switch (integrationMethod)
	{
	case 1:	  
	  champ_.foldfields(slice_of_beam_[0].get_rho(),slice_of_beam_[1].get_rho(),charge_sign);
	  break;
	case 2: 
	  
	  champ_.fold_fft(slice_of_beam_[0].get_rho(),slice_of_beam_[1].get_rho(),charge_sign);
	  break; 
	       
	case 3:
	  champ_.foldfronts(slice_of_beam_[0].get_rho(),slice_of_beam_[1].get_rho() , sor_parameter, charge_sign);
	}
    }


GENERAL_GRID::~GENERAL_GRID()
{
  //  delete [] rho1_;
  //  delete [] rho2_;
}



void  GENERAL_GRID::init_grid_phys (float n_particles1, float n_particles2, float cut_x,float cut_y,float cut_z, float charge_sign,   FFT_SERVER* fourier)
{
  //int i1,i2,j,j0,i;
  // PHI_FLOAT factor,phi0;
   PHI_FLOAT factor;
   //double x0,y0;
   //int nn[2];

  float offsetx, offsety;
  float deltax, deltay, deltaz;
  
  slice_of_beam_[0].update_nb_part_per_macro(n_particles1);
  slice_of_beam_[1].update_nb_part_per_macro(n_particles2);
    /////
  cut_x_ = cut_x;
  cut_y_ = cut_y;
  cut_z_ = cut_z;
  min_x_=-((float)n_cell_x_-2)/((float)n_cell_x_)*cut_x;
  max_x_=((float)n_cell_x_-2)/((float)n_cell_x_)*cut_x;
  min_y_=-((float)n_cell_y_-2)/((float)n_cell_y_)*cut_y;
  max_y_=((float)n_cell_y_-2)/((float)n_cell_y_)*cut_y;
  offsetx=0.5*(float)n_cell_x_;
  offsety=0.5*(float)n_cell_y_;
  deltax=2.0*cut_x/((float)n_cell_x_);
  deltay=2.0*cut_y/((float)n_cell_y_);
  deltaz=2.0*cut_z/((float)n_cell_z_);
  mesh_ = MESH(deltax, deltay, deltaz, offsetx, offsety);
  delta_x_inverse_ = 1.0/deltax;
  delta_y_inverse_ = 1.0/deltay;


  min_z_ = -cut_z;
  max_z_ = cut_z;
  step_=deltaz/(2.0*(float)timestep_);
  //

  factor=-charge_sign*2.0*RE/(deltaz)*EMASSeV;
  rho_factor_ = 2.0*factor;
  factor/=deltax*deltay;
  field_kernel_factor_ = factor;
  champ_.dist_init(factor, deltax, deltay,fourier);
}


GRID::GRID() : GENERAL_GRID()
{
  photon_timing_ = 0;
  photon_birth_[0] = photon_birth_[1] = 0;
  photon_angle_ = 0;
  photon_angle_lf_max_[0] = photon_angle_lf_max_[1] = 0.0;
  photon_angle_seeded_ = false;
  photon_angle_stream_ = 0;
  photon_file_ = NULL;
  hadron_file_ = NULL;
  minijets_ = NULL;
  cross_ = NULL;
  rndm_generator_ = NULL;
  gpu_cic_ = NULL;
  gpu_cic_mode_ = 0;
  gpu_cic_variant_ = feebl::gpu::CicVariant::Atomic;
  gpu_cic_auto_ = false;
  gpu_cic_auto_scanned_ = false;
  gpu_cic_auto_locked_serial_ = false;
  gpu_cic_device_ = 0;
  gpu_cic_min_particles_ = 262144;
  gpu_cic_max_slice_particles_ = 0;
  gpu_cic_timing_ = false;
  gpu_cic_calls_ = 0;
  gpu_cic_fallback_calls_ = 0;
  gpu_cic_fallback_particles_ = 0;
  gpu_cic_packed_calls_ = 0;
  gpu_cic_packed_particles_ = 0;
  gpu_cic_setup_ms_ = 0.0;
  gpu_cic_h2d_ms_ = 0.0;
  gpu_cic_kernel_ms_ = 0.0;
  gpu_cic_d2h_ms_ = 0.0;
  gpu_cic_total_ms_ = 0.0;
  photon_lumi_factorized_seen_ = false;
  photon_lumi_factorized_calls_ = 0;
  photon_lumi_cartesian_calls_ = 0;
  photon_lumi_ge_opportunities_ = 0;
  photon_lumi_eg_opportunities_ = 0;
  photon_lumi_gg_opportunities_ = 0;
  photon_lumi_total_opportunities_ = 0;
  photon_lumi_cartesian_pairs_visited_ = 0;
  photon_lumi_workspace_ = NULL;
  electron_lumi_blocked_workspace_ = NULL;
  electron_lumi_mode3_seen_ = false;
  electron_lumi_mode3_calls_ = 0U;
  electron_lumi_mode3_cells_ = 0U;
  electron_lumi_mode3_pairs_ = 0U;
  electron_lumi_mode3_parallel_pairs_ = 0U;
  electron_lumi_mode3_parallel_sample_candidates_ = 0U;
  electron_lumi_mode3_serial_sample_merge_candidates_ = 0U;
  electron_lumi_mode3_serial_result_commits_ = 0U;
  electron_lumi_mode3_legacy_heap_candidates_ = 0U;
  electron_lumi_mode3_counter_heap_accepts_ = 0U;
  electron_lumi_mode3_sampling_mode_ = 0;
  electron_lumi_mode3_bernoulli_draws_ = 0U;
  electron_lumi_mode3_bernoulli_rejects_ = 0U;
  electron_lumi_mode3_bernoulli_accepted_pairs_ = 0U;
  electron_lumi_mode3_bernoulli_accepted_records_ = 0U;
  electron_lumi_mode3_outer_fused_pairs_ = 0U;
  electron_lumi_mode3_geometric_rows_ = 0U;
  electron_lumi_mode3_geometric_rng_calls_ = 0U;
  electron_lumi_mode3_geometric_successes_ = 0U;
  electron_lumi_mode3_geometric_integer_copies_ = 0U;
  electron_lumi_mode3_geometric_fallback_rows_ = 0U;
  electron_lumi_mode3_geometric_fallback_pairs_ = 0U;
  electron_lumi_mode3_max_workers_ = 1U;
  electron_lumi_mode3_preflight_ms_ = 0.0;
  electron_lumi_mode3_materialize_ms_ = 0.0;
  electron_lumi_mode3_evaluate_ms_ = 0.0;
  electron_lumi_mode3_sample_merge_ms_ = 0.0;
  electron_lumi_mode3_publish_ms_ = 0.0;
  electron_lumi_mode3_outer_fused_ms_ = 0.0;
  electron_lumi_spatial_ids_validated_ = false;
  sparse_cells_enabled_ = false;
  photon_polarisation_ = 0;
  coherent_model_ = 0;
  wavefront_outer_parallel_lane_ = false;
  electron_lumi_stable_ids_enabled_ = false;
  electron_lumi_stable_ids1_ = NULL;
  electron_lumi_stable_ids2_ = NULL;
  sparse_dense_traversal_visits_ = 0U;
  sparse_actual_traversal_visits_ = 0U;
}

GRID::~GRID() 
{
  if (electron_lumi_spatial_receipt_.seen &&
      !wavefront_outer_parallel_lane_)
    {
      const ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT& receipt =
          electron_lumi_spatial_receipt_;
      std::cerr
          << "FEEBL_ELECTRON_LUMI_SPATIAL_RECEIPT_V1"
          << " requested_kernel=1"
          << " actual_kernel=fixed-physical-cic-v1"
          << " implementation=cpu-reference"
          << " qualification=stage-b-restricted"
          << " geometry_hash=" << receipt.geometry_hash
          << " plan_sequence_hash="
          << electron_lumi_spatial_sequence_hash(
                 receipt.geometry_hash, receipt.plan_hashes)
          << " nx=" << receipt.n_x
          << " ny=" << receipt.n_y
          << " x_min_nm=" << receipt.x_min
          << " x_max_nm=" << receipt.x_max
          << " y_min_nm=" << receipt.y_min
          << " y_max_nm=" << receipt.y_max
          << " delta_x_nm=" << receipt.delta_x
          << " delta_y_nm=" << receipt.delta_y
          << " normalization=" << receipt.normalization
          << " max_pairs_per_encounter="
          << receipt.max_pairs_per_encounter
          << " closure_relative_tolerance="
          << receipt.closure_relative_tolerance
          << " encounters=" << receipt.encounters
          << " beam1_input_roots=" << receipt.beam1_input_roots
          << " beam1_accepted_roots=" << receipt.beam1_accepted_roots
          << " beam1_rejected_roots=" << receipt.beam1_rejected_roots
          << " beam2_input_roots=" << receipt.beam2_input_roots
          << " beam2_accepted_roots=" << receipt.beam2_accepted_roots
          << " beam2_rejected_roots=" << receipt.beam2_rejected_roots
          << " beam1_input_population="
          << receipt.beam1_input_population
          << " beam1_accepted_population="
          << receipt.beam1_accepted_population
          << " beam1_rejected_population="
          << receipt.beam1_rejected_population
          << " beam2_input_population="
          << receipt.beam2_input_population
          << " beam2_accepted_population="
          << receipt.beam2_accepted_population
          << " beam2_rejected_population="
          << receipt.beam2_rejected_population
          << " physical_pairs=" << receipt.physical_pairs
          << " pair_weight_sum=" << receipt.pair_weight_sum
          << " density_dot_weight=" << receipt.density_dot_weight
          << " maximum_closure_relative_error="
          << receipt.maximum_closure_relative_error
          << " admission=full-support"
          << " pair_order=beam1-id-major/beam2-id-minor"
          << " collision_kernel_rng_draws=0"
          << " unsupported_side_effects=card-rejected"
          << std::endl;
    }
  if (electron_lumi_mode3_seen_)
    {
      std::cerr
          << "FEEBL electron-lumi summary: requested=parallel-blocked-soa-heap-v1"
          << " calls=" << electron_lumi_mode3_calls_
          << " occupied_cells=" << electron_lumi_mode3_cells_
          << " logical_pairs=" << electron_lumi_mode3_pairs_
          << " parallel_evaluated_pairs="
          << electron_lumi_mode3_parallel_pairs_
          << " parallel_sampled_candidates="
          << electron_lumi_mode3_parallel_sample_candidates_
          << " serial_sample_merge_candidates="
          << electron_lumi_mode3_serial_sample_merge_candidates_
          << " serial_result_commits="
          << electron_lumi_mode3_serial_result_commits_
          << " legacy_heap_candidates="
          << electron_lumi_mode3_legacy_heap_candidates_
          << " counter_heap_accepts="
          << electron_lumi_mode3_counter_heap_accepts_
          << " sampling_mode=" << electron_lumi_mode3_sampling_mode_
          << " bernoulli_draws="
          << electron_lumi_mode3_bernoulli_draws_
          << " bernoulli_rejects="
          << electron_lumi_mode3_bernoulli_rejects_
          << " bernoulli_accepted_pairs="
          << electron_lumi_mode3_bernoulli_accepted_pairs_
          << " bernoulli_accepted_records="
          << electron_lumi_mode3_bernoulli_accepted_records_
          << " outer_fused_pairs="
          << electron_lumi_mode3_outer_fused_pairs_
          << " geometric_rows="
          << electron_lumi_mode3_geometric_rows_
          << " geometric_rng_calls="
          << electron_lumi_mode3_geometric_rng_calls_
          << " geometric_successes="
          << electron_lumi_mode3_geometric_successes_
          << " geometric_integer_copies="
          << electron_lumi_mode3_geometric_integer_copies_
          << " geometric_fallback_rows="
          << electron_lumi_mode3_geometric_fallback_rows_
          << " geometric_fallback_pairs="
          << electron_lumi_mode3_geometric_fallback_pairs_
          << " heap_records_final=" << lumi_heap_ee_.nb_pairs()
          << " max_workers=" << electron_lumi_mode3_max_workers_
          << " preflight_ms=" << electron_lumi_mode3_preflight_ms_
          << " materialize_ms=" << electron_lumi_mode3_materialize_ms_
          << " evaluate_ms=" << electron_lumi_mode3_evaluate_ms_
          << " sample_merge_ms=" << electron_lumi_mode3_sample_merge_ms_
          << " publish_ms=" << electron_lumi_mode3_publish_ms_
          << " outer_fused_ms=" << electron_lumi_mode3_outer_fused_ms_
          << " result_order=beam1-major/beam2-minor"
          << " output_order="
          << (lumi_heap_ee_.deterministic_sampling()
                  ? "stable-pair-identity" : "legacy-shuffle")
          << std::endl;
    }
  if (sparse_cells_enabled_)
    {
      const SPARSE_CELL_STATS stats = sparse_cell_stats();
      std::cerr
          << "FEEBL sparse-cell summary: mode=active-index-v1"
          << " dense_clear_visits=" << stats.dense_clear_visits
          << " actual_clear_visits=" << stats.actual_clear_visits
          << " avoided_clear_visits=" << stats.avoided_clear_visits
          << " dense_traversal_visits=" << stats.dense_traversal_visits
          << " actual_traversal_visits=" << stats.actual_traversal_visits
          << " avoided_traversal_visits=" << stats.avoided_traversal_visits
          << " order=ascending-cell/stable-insertion" << std::endl;
    }
  if (photon_lumi_factorized_seen_)
    {
      std::cerr
          << "FEEBL legacy photon-lumi summary: requested=factorized-exact-v2"
          << " factorized_calls=" << photon_lumi_factorized_calls_
          << " cartesian_calls=" << photon_lumi_cartesian_calls_
          << " logical_ge_pairs=" << photon_lumi_ge_opportunities_
          << " logical_eg_pairs=" << photon_lumi_eg_opportunities_
          << " logical_gg_pairs=" << photon_lumi_gg_opportunities_
          << " cartesian_pairs_visited="
          << photon_lumi_cartesian_pairs_visited_
          << " cartesian_pairs_avoided="
          << (photon_lumi_total_opportunities_ -
              photon_lumi_cartesian_pairs_visited_)
          << " normalization=legacy-sampled-macro-weight"
          << std::endl;
    }
  if ((gpu_cic_mode_ == 1 || gpu_cic_auto_) && gpu_cic_timing_)
    {
      std::cerr << "FEEBL CUDA CIC summary: mode="
                << (gpu_cic_auto_ ? "auto" :
                    feebl::gpu::cic_variant_name(gpu_cic_variant_))
                << " calls=" << gpu_cic_calls_
                << " locked_serial=" << gpu_cic_auto_locked_serial_
                << " max_slice_particles=" << gpu_cic_max_slice_particles_
                << " fallback_calls=" << gpu_cic_fallback_calls_
                << " fallback_particles=" << gpu_cic_fallback_particles_
                << " packed_calls=" << gpu_cic_packed_calls_
                << " packed_particles=" << gpu_cic_packed_particles_
                << " setup_ms=" << gpu_cic_setup_ms_
                << " h2d_ms=" << gpu_cic_h2d_ms_
                << " kernel_ms=" << gpu_cic_kernel_ms_
                << " d2h_ms=" << gpu_cic_d2h_ms_
                << " total_ms=" << gpu_cic_total_ms_ << std::endl;
    }
  delete photon_lumi_workspace_;
  delete electron_lumi_blocked_workspace_;
  delete gpu_cic_;
  if (photon_file_ != NULL) 
    {
      photon_file_->close();
      delete photon_file_;
    }
  if (hadron_file_ != NULL) 
    {
      hadron_file_->close();
      delete hadron_file_;
    }
  delete minijets_;
}

void GRID::enable_sparse_cells(bool enabled)
{
  sparse_cells_enabled_ = enabled;
  sparse_dense_traversal_visits_ = 0U;
  sparse_actual_traversal_visits_ = 0U;

  feebl::sparse_cells::ActiveCellSet* indices[] = {
      &particle_cells1_, &particle_cells2_, &photon_cells1_,
      &photon_cells2_, &extra_photon_cells1_, &extra_photon_cells2_};
  for (std::size_t index = 0; index < sizeof(indices)/sizeof(indices[0]);
       ++index)
    {
      indices[index]->set_enabled(enabled);
      indices[index]->reset_counters();
    }

  slice_of_beam_[0].enableSparseCells(enabled);
  slice_of_beam_[1].enableSparseCells(enabled);

  if (!enabled) return;

  // Normally this API is called before the first encounter. Reconstructing
  // the index also makes a late diagnostic opt-in safe and ensures the next
  // sparse clear cannot leave stale payloads behind.
  for (std::size_t cell = 0; cell < part_pointer1_.size(); ++cell)
    {
      if (!part_pointer1_[cell].empty()) particle_cells1_.touch(cell);
      if (!part_pointer2_[cell].empty()) particle_cells2_.touch(cell);
      if (!grid_photon1_[cell].empty()) photon_cells1_.touch(cell);
      if (!grid_photon2_[cell].empty()) photon_cells2_.touch(cell);
    }
  for (std::size_t cell = 0; cell < extra_photon_pointer1_.size(); ++cell)
    {
      if (!extra_photon_pointer1_[cell].empty())
        extra_photon_cells1_.touch(cell);
      if (!extra_photon_pointer2_[cell].empty())
        extra_photon_cells2_.touch(cell);
    }
  for (std::size_t index = 0; index < sizeof(indices)/sizeof(indices[0]);
       ++index)
    indices[index]->sort_cells();
}

GRID::SPARSE_CELL_STATS GRID::sparse_cell_stats() const
{
  SPARSE_CELL_STATS stats;
  stats.enabled = sparse_cells_enabled_;
  stats.dense_clear_visits =
      particle_cells1_.dense_clear_visits() +
      particle_cells2_.dense_clear_visits() +
      photon_cells1_.dense_clear_visits() +
      photon_cells2_.dense_clear_visits() +
      extra_photon_cells1_.dense_clear_visits() +
      extra_photon_cells2_.dense_clear_visits() +
      slice_of_beam_[0].denseRhoClearVisits() +
      slice_of_beam_[1].denseRhoClearVisits();
  stats.actual_clear_visits =
      particle_cells1_.actual_clear_visits() +
      particle_cells2_.actual_clear_visits() +
      photon_cells1_.actual_clear_visits() +
      photon_cells2_.actual_clear_visits() +
      extra_photon_cells1_.actual_clear_visits() +
      extra_photon_cells2_.actual_clear_visits() +
      slice_of_beam_[0].actualRhoClearVisits() +
      slice_of_beam_[1].actualRhoClearVisits();
  stats.avoided_clear_visits =
      stats.dense_clear_visits - stats.actual_clear_visits;
  stats.dense_traversal_visits = sparse_dense_traversal_visits_;
  stats.actual_traversal_visits = sparse_actual_traversal_visits_;
  stats.avoided_traversal_visits =
      stats.dense_traversal_visits - stats.actual_traversal_visits;
  return stats;
}

bool GRID::enable_gpu_cic(const std::string& variant,
                          int device,
                          std::string& error)
{
  feebl::gpu::CicVariant parsed;
  if (!feebl::gpu::parse_cic_variant(variant, parsed))
    {
      error = "unknown CUDA CIC variant '" + variant +
              "' (expected atomic, shared, or sorted)";
      return false;
    }
  if (gpu_cic_mode_ == 1)
    {
      if (parsed == gpu_cic_variant_) return true;
      error = "CUDA CIC backend is already initialized with a different variant";
      return false;
    }

  gpu_cic_ = new feebl::gpu::CicDepositor;
  if (!gpu_cic_->initialize(device, error))
    {
      delete gpu_cic_;
      gpu_cic_ = NULL;
      return false;
    }
  gpu_cic_variant_ = parsed;
  gpu_cic_mode_ = 1;
  return true;
}

void GRID::configure_gpu_cic()
{
  if (gpu_cic_mode_ != 0) return;

  // Backend choice is exclusively the parsed input-deck contract. Environment
  // overrides used during early prototyping were removed because they could
  // bypass default-serial provenance and race with pool_dual initialization.
  PARALLEL_BACKEND& backend = parallel_backend();
  // Mode 2 deliberately keeps deposition on the historical host path in its
  // first fields-only implementation.  It must never arm the independent
  // mode-1 CIC state machine or claim device deposition in provenance.
  const bool configured_gpu =
      backend.configured() && backend.gpu_mode1_enabled();
  if (!configured_gpu)
    {
      gpu_cic_mode_ = -1;
      return;
    }

  std::string requested_value = backend.actual_gpu_variant();
  const int device = backend.gpu_device();
  gpu_cic_min_particles_ = backend.gpu_min_particles();
  gpu_cic_timing_ = true;

  if (backend.requested_gpu_variant() == "auto" &&
      requested_value == "atomic")
    {
      // compute_backend=gpu is an explicit request. Without explicit fallback
      // authority it must actually execute the measured winner, not quietly
      // threshold back to serial while claiming actual=gpu.
      if (!backend.allow_fallback())
        requested_value = "atomic";
      else
        {
      gpu_cic_auto_ = true;
      gpu_cic_device_ = device;
      gpu_cic_variant_ = feebl::gpu::CicVariant::Atomic;
      // Do not initialize the CUDA runtime here.  Small FeeBL slices should
      // take the serial path without paying driver/context startup costs.
      gpu_cic_mode_ = 2;
      std::cerr << "FEEBL CUDA CIC auto mode armed: atomic threshold="
                << gpu_cic_min_particles_ << " particles device=" << device
                << std::endl;
      return;
        }
    }

  std::string error;
  if (!enable_gpu_cic(requested_value, device, error))
    {
      if (configured_gpu && backend.allow_fallback())
        {
          std::cerr << "FEEBL CUDA CIC unavailable; explicit fallback to serial: "
                    << error << std::endl;
          backend.runtime_fallback_to_serial(
              "CUDA initialization failed; selected serial CIC: " + error);
          gpu_cic_mode_ = -1;
          return;
        }
      std::cerr << "CUDA CIC requested explicitly but unavailable: "
                << error << std::endl;
      std::exit(2);
    }
  std::cerr << "FEEBL CUDA CIC enabled: variant="
            << feebl::gpu::cic_variant_name(gpu_cic_variant_)
            << " device=" << gpu_cic_->device_description() << std::endl;
}




void  GRID::lumi_init(const SWITCHES& switches)
{
  int nmax;
  if(switches.get_do_lumi()){
    if (switches.get_do_lumi()&1) 
      {
	nmax = switches.get_num_lumi();
	if (nmax < 100) nmax = 100;
		const int requested_sampling =
	            switches.get_electron_lumi_precompute() == 3
	                ? switches.get_electron_lumi_sampling() : 0;
		// Sampling mode 3 changes only how Bernoulli successes are found;
		// accepted records retain the append-only mode-2 heap contract.
		const int heap_sampling = requested_sampling == 3
	            ? 2 : requested_sampling;
		lumi_heap_ee_ = LUMI_HEAP_EE(
	            nmax, switches.get_lumi_p(), rndm_generator_, heap_sampling);
      }
    if (switches.get_do_lumi()&2) 
      {
	nmax = switches.get_num_lumi_eg();
	if (nmax < 100) nmax = 100;
	lumi_heap_eg_ = LUMI_HEAP(nmax,switches.get_lumi_p_eg(), rndm_generator_);
	lumi_heap_ge_ = LUMI_HEAP(nmax, switches.get_lumi_p_eg(), rndm_generator_);
      }
    if (switches.get_do_lumi()&4) 
      {
	nmax = switches.get_num_lumi_gg();
	if (nmax < 100) nmax = 100;
	lumi_heap_gg_ = LUMI_HEAP( nmax,switches.get_lumi_p_gg(), rndm_generator_);
      }
  }
}

bool GRID::merge_primary_photon_collision_lane(
    const GRID& lane, std::string* reason)
{
  std::vector<const GRID*> lanes(1U, &lane);
  return merge_primary_photon_collision_lanes(lanes, reason);
}

bool GRID::merge_electron_lumi_spatial_receipt(
    ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT& destination,
    const ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT& source,
    std::string* reason)
{
  if (!source.seen)
    {
      if (reason) reason->clear();
      return true;
    }

  const auto invalid_source = [&source]() {
    const auto counts_close = [](std::uint64_t input,
                                 std::uint64_t accepted,
                                 std::uint64_t rejected) {
      return accepted <= input && rejected <= input - accepted &&
             accepted + rejected == input;
    };
    const long double pair_weight =
        static_cast<long double>(source.pair_weight_sum);
    const long double density_weight =
        static_cast<long double>(source.density_dot_weight);
    const long double closure_scale = std::max(pair_weight, density_weight);
    const long double aggregate_closure =
        closure_scale == 0.0L
            ? 0.0L
            : std::fabs(pair_weight - density_weight) / closure_scale;
    return source.encounters == 0U ||
           source.plan_hashes.size() != source.encounters ||
           source.n_x < 2U || source.n_y < 2U ||
           source.max_pairs_per_encounter == 0U ||
           !counts_close(source.beam1_input_roots,
                         source.beam1_accepted_roots,
                         source.beam1_rejected_roots) ||
           !counts_close(source.beam2_input_roots,
                         source.beam2_accepted_roots,
                         source.beam2_rejected_roots) ||
           source.beam1_rejected_roots != 0U ||
           source.beam2_rejected_roots != 0U ||
           !electron_lumi_spatial_is_finite(source.x_min) ||
           !electron_lumi_spatial_is_finite(source.x_max) ||
           !electron_lumi_spatial_is_finite(source.y_min) ||
           !electron_lumi_spatial_is_finite(source.y_max) ||
           !electron_lumi_spatial_is_finite(source.delta_x) ||
           !electron_lumi_spatial_is_finite(source.delta_y) ||
           !electron_lumi_spatial_is_finite(source.normalization) ||
           !electron_lumi_spatial_is_finite(
               source.closure_relative_tolerance) ||
           !electron_lumi_spatial_is_finite(
               source.beam1_input_population) ||
           !electron_lumi_spatial_is_finite(
               source.beam1_accepted_population) ||
           !electron_lumi_spatial_is_finite(
               source.beam1_rejected_population) ||
           !electron_lumi_spatial_is_finite(
               source.beam2_input_population) ||
           !electron_lumi_spatial_is_finite(
               source.beam2_accepted_population) ||
           !electron_lumi_spatial_is_finite(
               source.beam2_rejected_population) ||
           !electron_lumi_spatial_is_finite(source.pair_weight_sum) ||
           !electron_lumi_spatial_is_finite(source.density_dot_weight) ||
           !electron_lumi_spatial_is_finite(
               source.maximum_closure_relative_error) ||
           !(source.x_max > source.x_min) ||
           !(source.y_max > source.y_min) || source.delta_x <= 0.0 ||
           source.delta_y <= 0.0 || source.normalization <= 0.0 ||
           source.closure_relative_tolerance <= 0.0 ||
           source.closure_relative_tolerance >= 1.0 ||
           source.beam1_input_population < 0.0 ||
           source.beam1_accepted_population < 0.0 ||
           source.beam1_rejected_population != 0.0 ||
           source.beam2_input_population < 0.0 ||
           source.beam2_accepted_population < 0.0 ||
           source.beam2_rejected_population != 0.0 ||
           source.beam1_input_population !=
               source.beam1_accepted_population ||
           source.beam2_input_population !=
               source.beam2_accepted_population ||
           source.pair_weight_sum < 0.0 ||
           source.density_dot_weight < 0.0 ||
           source.maximum_closure_relative_error < 0.0 ||
           source.maximum_closure_relative_error >
               source.closure_relative_tolerance ||
           aggregate_closure > static_cast<long double>(
                                   source.closure_relative_tolerance);
  };
  if (invalid_source())
    {
      if (reason) *reason = "invalid spatial-luminosity worker receipt";
      return false;
    }

  try
    {
      ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT candidate = destination;
      if (!candidate.seen)
        {
          candidate = source;
          destination = candidate;
          if (reason) reason->clear();
          return true;
        }
      if (candidate.geometry_hash != source.geometry_hash ||
          candidate.n_x != source.n_x || candidate.n_y != source.n_y ||
          candidate.x_min != source.x_min ||
          candidate.x_max != source.x_max ||
          candidate.y_min != source.y_min ||
          candidate.y_max != source.y_max ||
          candidate.delta_x != source.delta_x ||
          candidate.delta_y != source.delta_y ||
          candidate.normalization != source.normalization ||
          candidate.max_pairs_per_encounter !=
              source.max_pairs_per_encounter ||
          candidate.closure_relative_tolerance !=
              source.closure_relative_tolerance)
        {
          if (reason)
            *reason = "spatial-luminosity receipt geometry/configuration drift";
          return false;
        }

      const auto add_u64 = [](std::uint64_t& value,
                              std::uint64_t increment) {
        if (increment > std::numeric_limits<std::uint64_t>::max() - value)
          return false;
        value += increment;
        return true;
      };
      const auto add_double = [](double& value, double increment) {
        if (!electron_lumi_spatial_is_finite(value) ||
            !electron_lumi_spatial_is_finite(increment))
          return false;
        const double sum = value + increment;
        if (!electron_lumi_spatial_is_finite(sum)) return false;
        value = sum;
        return true;
      };
      if (!add_u64(candidate.encounters, source.encounters) ||
          !add_u64(candidate.beam1_input_roots,
                   source.beam1_input_roots) ||
          !add_u64(candidate.beam1_accepted_roots,
                   source.beam1_accepted_roots) ||
          !add_u64(candidate.beam1_rejected_roots,
                   source.beam1_rejected_roots) ||
          !add_u64(candidate.beam2_input_roots,
                   source.beam2_input_roots) ||
          !add_u64(candidate.beam2_accepted_roots,
                   source.beam2_accepted_roots) ||
          !add_u64(candidate.beam2_rejected_roots,
                   source.beam2_rejected_roots) ||
          !add_u64(candidate.physical_pairs, source.physical_pairs) ||
          !add_double(candidate.beam1_input_population,
                      source.beam1_input_population) ||
          !add_double(candidate.beam1_accepted_population,
                      source.beam1_accepted_population) ||
          !add_double(candidate.beam1_rejected_population,
                      source.beam1_rejected_population) ||
          !add_double(candidate.beam2_input_population,
                      source.beam2_input_population) ||
          !add_double(candidate.beam2_accepted_population,
                      source.beam2_accepted_population) ||
          !add_double(candidate.beam2_rejected_population,
                      source.beam2_rejected_population) ||
          !add_double(candidate.pair_weight_sum,
                      source.pair_weight_sum) ||
          !add_double(candidate.density_dot_weight,
                      source.density_dot_weight))
        {
          if (reason) *reason = "spatial-luminosity receipt overflow";
          return false;
        }
      candidate.maximum_closure_relative_error = std::max(
          candidate.maximum_closure_relative_error,
          source.maximum_closure_relative_error);
      if (source.plan_hashes.size() >
          candidate.plan_hashes.max_size() - candidate.plan_hashes.size())
        {
          if (reason) *reason = "spatial-luminosity plan receipt overflow";
          return false;
        }
      candidate.plan_hashes.insert(candidate.plan_hashes.end(),
                                   source.plan_hashes.begin(),
                                   source.plan_hashes.end());
      destination = candidate;
    }
  catch (const std::exception& error)
    {
      if (reason)
        *reason = std::string("spatial-luminosity receipt allocation failed: ") +
                  error.what();
      return false;
    }
  if (reason) reason->clear();
  return true;
}

bool GRID::merge_primary_photon_collision_lanes(
    const std::vector<const GRID*>& lanes, std::string* reason)
{
  std::vector<const RESULTS*> lane_results;
  lane_results.reserve(lanes.size());
  for (std::size_t lane = 0; lane < lanes.size(); ++lane)
    {
      if (!lanes[lane])
        {
          if (reason) *reason = "null worker GRID in result publication";
          return false;
        }
      lane_results.push_back(&lanes[lane]->results_);
    }
  RESULTS merged_results = results_;
  if (!merged_results.merge_primary_photon_collision_from_many(lane_results))
    {
      if (reason)
        *reason = "worker RESULTS are non-finite, incompatible, or overflowed";
      return false;
    }

  ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT merged_spatial_receipt =
      electron_lumi_spatial_receipt_;
  for (std::size_t lane = 0; lane < lanes.size(); ++lane)
    {
      std::string spatial_reason;
      if (!merge_electron_lumi_spatial_receipt(
              merged_spatial_receipt,
              lanes[lane]->electron_lumi_spatial_receipt_,
              &spatial_reason))
        {
          if (reason)
            *reason = "worker spatial-luminosity receipt merge failed: " +
                      spatial_reason;
          return false;
        }
    }

  const auto checked_long_sum = [](long int first, long int second,
                                   long int* result) {
    if ((second > 0 && first > std::numeric_limits<long int>::max() - second) ||
        (second < 0 && first < std::numeric_limits<long int>::min() - second))
      return false;
    *result = first + second;
    return true;
  };
  long int beam1_in = 0;
  long int beam1_out = 0;
  long int beam2_in = 0;
  long int beam2_out = 0;
  beam1_in = distribute1_.in;
  beam1_out = distribute1_.out;
  beam2_in = distribute2_.in;
  beam2_out = distribute2_.out;
  for (std::size_t lane = 0; lane < lanes.size(); ++lane)
    {
      long int next_beam1_in = 0;
      long int next_beam1_out = 0;
      long int next_beam2_in = 0;
      long int next_beam2_out = 0;
      if (!checked_long_sum(beam1_in, lanes[lane]->distribute1_.in,
                            &next_beam1_in) ||
          !checked_long_sum(beam1_out, lanes[lane]->distribute1_.out,
                            &next_beam1_out) ||
          !checked_long_sum(beam2_in, lanes[lane]->distribute2_.in,
                            &next_beam2_in) ||
          !checked_long_sum(beam2_out, lanes[lane]->distribute2_.out,
                            &next_beam2_out))
        {
          if (reason) *reason = "worker distribution counters overflowed";
          return false;
        }
      beam1_in = next_beam1_in;
      beam1_out = next_beam1_out;
      beam2_in = next_beam2_in;
      beam2_out = next_beam2_out;
    }

  std::vector<const feebl::design_axis::Spectrum*> design_axis_lanes;
  design_axis_lanes.reserve(lanes.size());
  for (const GRID* lane : lanes)
    design_axis_lanes.push_back(&lane->design_axis_spectrum_);
  if (!design_axis_spectrum_.merge_from_many(design_axis_lanes))
    {
      if (reason) *reason = "worker design-axis spectra are incompatible or overflowed";
      return false;
    }
  results_ = merged_results;
  electron_lumi_spatial_receipt_ = merged_spatial_receipt;
  distribute1_.in = beam1_in;
  distribute1_.out = beam1_out;
  distribute2_.in = beam2_in;
  distribute2_.out = beam2_out;
  if (reason) reason->clear();
  return true;
}

bool GRID::merge_primary_photon_lumi_heaps(
    const std::vector<const GRID*>& lanes, std::string* reason)
{
  std::vector<const LUMI_HEAP_EE*> heaps;
  heaps.reserve(lanes.size());
  for (std::size_t lane = 0; lane < lanes.size(); ++lane)
    {
      if (!lanes[lane])
        {
          if (reason) *reason = "null worker GRID in e-e heap publication";
          return false;
        }
      heaps.push_back(&lanes[lane]->lumi_heap_ee_);
    }
  std::string heap_reason;
  if (!lumi_heap_ee_.merge_deterministic_from_many(heaps, &heap_reason))
    {
      if (reason) *reason = "worker e-e heap merge failed: " + heap_reason;
      return false;
    }
  if (reason) reason->clear();
  return true;
}

void GRID::reset_primary_photon_collision_lane(const SWITCHES& switches)
{
  results_.reset_primary_photon_collision_lane();
  design_axis_spectrum_.reset();
  electron_lumi_spatial_receipt_ =
      ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT();
  // Keep the lane-local deterministic top-K reservoir for the whole
  // crossing.  The exact global top-K is contained in the union of the
  // per-lane top-K sets, so these are published once after the EncounterDAG
  // completes instead of copying a potentially 500k-record heap after every
  // encounter.
  check_distribute(1);
  // Ensure an accidental reconfiguration cannot silently detach the local
  // result object from the immutable card used by subsequent collision calls.
  results_.connect_switches(&switches);
}

void GRID::save_lumi_on_files(SWITCHES& switches, std::string lumi_ee_out, std::string lumi_eg_out, std::string lumi_ge_out, std::string lumi_gg_out)
{
  if(design_axis_spectrum_.enabled()) {
    const double native_weight=results_.pair_spectrum().w_moment().weight();
    if(std::fabs(design_axis_spectrum_.total_weight()-native_weight)>
       1e-12*std::max(1.0,std::fabs(native_weight)))
      throw std::runtime_error("design-axis/all-native pair luminosity weight closure failed");
    std::ofstream output((lumi_ee_out+".design_axis_dLdW.fne").c_str());
    if(!output)throw std::runtime_error("cannot create design-axis spectrum");
    design_axis_spectrum_.write(output,FEEBL_BUILD_SOURCE_HASH,switches.get_rndm_seed());
    if(!output)throw std::runtime_error("design-axis spectrum write failed");
  }
  if(switches.get_do_lumi())
    {
      if (switches.get_do_lumi()&1) 
	{
	  lumi_heap_ee_.saveLumi(lumi_ee_out);
	  if (switches.get_store_native())
	    {
	      // Same records, self-describing. See nativeOutputCPP.h: the legacy
	      // file has no header, no column names and no vz, so it cannot be
	      // read correctly by anyone who did not write this code.
	      NativeRunMeta meta;
	      meta.feebl_version = PACKAGE_VERSION;
	      meta.spin_present  = switches.get_bmt_precession() != 0;
	      meta.rndm_seed     = switches.get_rndm_seed();
	      meta.half_crossing_angle_rad = native_half_crossing_angle_;
	      meta.momenta_carry_crossing_angle = false;   // positions carry the crossing shear
	      lumi_heap_ee_.saveLumiNative(lumi_ee_out + ".fne", meta);

	      // dL/dW, written as its own file rather than added to the
	      // protokoll. A spectrum is tabular, and more to the point the
	      // protokoll is compared byte for byte against v0.5.0 -- a new
	      // line there would break bit-identity for every existing deck.
	      // Same reason the polarization block emits nothing when spin is
	      // off.
	      {
		const pairmoments::SpectrumW& sp = results_.pair_spectrum();
		std::ofstream f((lumi_ee_out + ".dLdW.fne").c_str());
		if (!f) throw std::runtime_error("Could not create native dL/dW output");
		if (f)
		  {
		    f << native_header(meta, "pair_spectrum",
				       "W_low W_high dL_dW weight h1 h1_err h2 h2_err "
				       "h1h2 cov n_eff");
		    unsigned long rows = 0;
		    for (std::size_t i = 0; i < sp.bins(); i++)
		      {
			double dl = 0.0;
			// An empty bin is skipped, not written as zero: no
			// sample and zero luminosity are different claims.
			if (!sp.dL_dW(i, dl)) continue;
			const pairmoments::PairHelicity& h = sp.bin(i);
			f << sp.bin_low(i) << " " << sp.bin_high(i) << " "
			  << dl << " " << sp.bin_weight(i) << " "
			  << h.h1().mean() << " " << h.h1().error() << " "
			  << h.h2().mean() << " " << h.h2().error() << " "
			  << h.mean_product() << " " << h.covariance() << " "
			  << h.h1().n_eff() << std::endl;
			rows++;
		      }
		    f << "#underflow " << sp.underflow() << std::endl
		      << "#overflow " << sp.overflow() << std::endl
		      << "#mean_W " << sp.w_moment().mean()
		      << " +- " << sp.w_moment().error() << std::endl
		      // The UNBINNED rms, accumulated per pair. Anything derived
		      // from the histogram carries the bin width as a floor --
		      // at the default 200 bins over 0..275 GeV that floor is
		      // 1.375/sqrt(12) = 0.397 GeV, which is the same size as
		      // the feature a monochromatization study is looking for.
		      // This number has no such floor.
		      << "#rms_W " << std::sqrt(sp.w_moment().variance())
		      << std::endl
		      << "#n_eff_total " << sp.w_moment().n_eff() << std::endl;
		    f << native_footer("pair_spectrum", rows);
		    f.close();
		    if (!f) throw std::runtime_error("Failed while writing native dL/dW output");
		  }
	      }
	    }
	}
      if (switches.get_do_lumi()&2) 
	{	
	  lumi_heap_eg_.saveLumi(lumi_eg_out);
	  lumi_heap_ge_.saveLumi(lumi_ge_out);
	}
      if (switches.get_do_lumi()&4) 
	{
	  lumi_heap_gg_.saveLumi(lumi_gg_out);
	}
    }
  // Kernel 1 deliberately rejects the e-e event heap, but its deterministic
  // aggregate dL/dW is still a first-class physics result.  Publish that
  // spectrum when native output is requested without manufacturing an empty
  // legacy lumi.ee event file.  This branch is unreachable in Mode 0.
  if (switches.get_electron_lumi_spatial_kernel() == 1 &&
      switches.get_store_native() && !(switches.get_do_lumi() & 1))
    {
      NativeRunMeta meta;
      meta.feebl_version = PACKAGE_VERSION;
      meta.spin_present = switches.get_bmt_precession() != 0;
      meta.rndm_seed = switches.get_rndm_seed();
      meta.half_crossing_angle_rad = native_half_crossing_angle_;
      meta.momenta_carry_crossing_angle = false;
      const pairmoments::SpectrumW& sp = results_.pair_spectrum();
      std::ofstream f((lumi_ee_out + ".dLdW.fne").c_str());
      if (!f)
        throw std::runtime_error(
            "spatial electron luminosity could not create dL/dW output");
      f << native_header(meta, "pair_spectrum",
                         "W_low W_high dL_dW weight h1 h1_err h2 h2_err "
                         "h1h2 cov n_eff");
      unsigned long rows = 0;
      for (std::size_t i = 0; i < sp.bins(); ++i)
        {
          double dl = 0.0;
          if (!sp.dL_dW(i, dl)) continue;
          const pairmoments::PairHelicity& h = sp.bin(i);
          f << sp.bin_low(i) << " " << sp.bin_high(i) << " "
            << dl << " " << sp.bin_weight(i) << " "
            << h.h1().mean() << " " << h.h1().error() << " "
            << h.h2().mean() << " " << h.h2().error() << " "
            << h.mean_product() << " " << h.covariance() << " "
            << h.h1().n_eff() << std::endl;
          ++rows;
        }
      f << "#underflow " << sp.underflow() << std::endl
        << "#overflow " << sp.overflow() << std::endl
        << "#mean_W " << sp.w_moment().mean()
        << " +- " << sp.w_moment().error() << std::endl
        << "#rms_W " << std::sqrt(sp.w_moment().variance()) << std::endl
        << "#n_eff_total " << sp.w_moment().n_eff() << std::endl;
      f << native_footer("pair_spectrum", rows);
      if (!f)
        throw std::runtime_error(
            "spatial electron luminosity failed while writing dL/dW output");
    }
}

void GRID::save_tertphot_on_file(std::string tertphotfile)
{
  FILE_IN_OUT filout;
  filout.open_file(tertphotfile,"w");
  for(unsigned int i =0; i<tertphot_.size();i++)
    {
      filout.save_object_on_persistent_file(&tertphot_[i]);
    }
  filout.close();
}

void GRID::read(const PARAMETERS& param, int automatic)
{
  int n_n;
  float n_f;
  if (automatic != 1) 
    {
      n_n = param.readIValue("n_x");
      if (n_n > 0) n_cell_x_= n_n;
      n_n = param.readIValue("n_y");
      if (n_n > 0) n_cell_y_ = n_n;
      n_n = param.readIValue("n_z");
      if (n_n > 0) n_cell_z_ = n_n;
    }
  n_n = param.readIValue("n_t");
  if (n_n > 0) timestep_ = n_n;
  n_n = param.readIValue("n_m.1");
  //  if (n_n > 0) nb_macroparticles_[0] = n_n;
  if (n_n > 0) slice_of_beam_[0].set_macroparticles(n_n);
  n_n = param.readIValue("n_m.2");
  //  if (n_n > 0) nb_macroparticles_[1] = n_n;
  if (n_n > 0) slice_of_beam_[1].set_macroparticles(n_n);

  // the indexation is unusual, but it is like that in the original guinea-pig C
  n_f = param.readFValue("scale_step.1");
  //  if (n_f > 0.0) scal_step[1] = n_f;
  if (n_f > 0.0) slice_of_beam_[1].set_scal_step(n_f);  
  n_f = param.readFValue("scale_step.2");
  //  if (n_f > 0.0) scal_step[0] = n_f;
  if (n_f > 0.0) slice_of_beam_[0].set_scal_step(n_f);
}


std::string GRID::output_flow() const 
{
  std::ostringstream out;
  out << title(std::string("grid parameters"));
  out << "n_x = " << n_cell_x_ << " n_y = " << n_cell_y_ << std::endl;

  out << "n_z = " << n_cell_z_ << " n_t = " << timestep_ << std::endl;

  out << "n_m.1 = " << slice_of_beam_[0].get_macroparticles()  << " n_m.2 = " << slice_of_beam_[1].get_macroparticles() << std::endl;

  out << "cut_x = " << cut_x_ << " nm ; cut_y = " << cut_y_ << " nm ; cut_z = " << cut_z_*1e-3 << " micrometers " << std::endl;

  out << std::endl;
  out << " ...................................................... " << std::endl;
  out << " relative amount of interacting particles that were outside the grid during one time step : " << std::endl;
  out << "beam 1 : miss = " << distribute1_.delta <<  "  beam 2 : miss = " << distribute2_.delta << std::endl;

  // These counters record macro-particle deposition/stencil visits, not
  // unique particles or physical charge. They do not describe every field
  // source or gather: outer meshes and analytic fallback have separate paths.
  // Preserve the established numeric protocol and emit the explanation on
  // stderr so existing readers can still identify the miss/out fields.
  {
    const float m1 = distribute1_.delta;
    const float m2 = distribute2_.delta;
    const float worst = (m1 > m2) ? m1 : m2;
    if (worst > 0.01f)
      {
        std::cerr << " FEEBL: WARNING : maximum recorded main-grid/source-stencil"
                     " rejection fraction is " << (100.0f * worst) << "%"
                  << std::endl
                  << "   (beam 1 " << (100.0f * m1) << "%, beam 2 "
                  << (100.0f * m2) << "%). These are macro-particle visit counts,"
                  << std::endl
                  << "   not a lost-charge fraction or a calibrated physics error."
                  << std::endl
                  << "   Outer meshes and analytic field fallback may still contribute."
                  << std::endl
                  << "   The sign and size of any luminosity, force or radiation bias"
                     " require a domain-convergence study."
                  << std::endl;
        if (worst > 0.05f)
          std::cerr << "   For precision use, vary cut_x/cut_y and increase n_x/n_y"
                    << std::endl
                    << "   to preserve cell size; then refine the cells independently."
                    << std::endl;
      }
  }
  out << " number of interacting part. that were outside the grid during one time step : " << std::endl;
  out << "out.1=" << distribute1_.tot <<  ";out.2=" << distribute2_.tot << ";" << std::endl; 


 return out.str();
}

void GRID::get_cuts(float& cut_x, float& cut_y, float& cut_z) const
{
  cut_x = cut_x_;
  cut_y = cut_y_;
  cut_z = cut_z_;
}


void GRID::init_grid_comp(int n_cell_x, int n_cell_y, int integration_method,   FFT_SERVER* fourier)
{
  n_cell_x_ = n_cell_x;
  n_cell_y_ = n_cell_y;
  integration_method_ = integration_method;
   champ_ = FIELD(integration_method, n_cell_x_, n_cell_y_, fourier);
   slice_of_beam_[0].resize(n_cell_x_, n_cell_y_);
   slice_of_beam_[1].resize(n_cell_x_, n_cell_y_);

  part_pointer1_ =  std::vector< std::vector<BEAM_PARTICLE_POINTER> >(n_cell_x_*n_cell_y_);
  part_pointer2_ =  std::vector< std::vector<BEAM_PARTICLE_POINTER> >(n_cell_x_*n_cell_y_);


  grid_photon1_ = std::vector< std::vector<BEAM_PHOTON_POINTER> >(n_cell_x_*n_cell_y_);
  grid_photon2_ = std::vector< std::vector<BEAM_PHOTON_POINTER> >(n_cell_x_*n_cell_y_);
  const std::size_t cells = static_cast<std::size_t>(n_cell_x_)*
                            static_cast<std::size_t>(n_cell_y_);
  particle_cells1_.reset(cells);
  particle_cells2_.reset(cells);
  photon_cells1_.reset(cells);
  photon_cells2_.reset(cells);
  sparse_cell_scratch_.clear();
  sparse_cell_scratch_.reserve(cells);
  sparse_traversal_cells_.clear();
  sparse_traversal_cells_.reserve(cells);
}


/*! Routine to calculate the parameters necessary for the iterative method of
   the potential calculation sor2 */

void GRID::init_sor2 (PHI_FLOAT *parameter)
{
  PHI_FLOAT factor;
  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();
  //  float deltaz = mesh_.get_delta_z();

  factor=1.0/(8.0*PI*RE/mesh_.get_delta_z()*1e9/2000.0);
  parameter[0]=factor*(deltay/deltax);
  parameter[1]=parameter[0];
  parameter[2]=factor*(deltax/deltay);
  parameter[3]=parameter[2];
  parameter[4]=-2.0*(parameter[0]+parameter[2]);
  parameter[5]=(deltay*deltay*cos(PI/(float)n_cell_x_)
		+deltax*deltax*cos(PI/(float)n_cell_y_))
                        /(deltax*deltax+deltay*deltay);
}



void GRID::check_distribute(int what)
{
    float tmp;
    if (what==0){
	distribute1_.delta=0.0;
	distribute2_.delta=0.0;
	distribute1_.tot=0;
	distribute2_.tot=0;
    }
    if (what<=1){
	distribute1_.in=0; distribute1_.out=0;
	distribute2_.in=0; distribute2_.out=0;
    }
    if (what==2){
	tmp=(float)distribute1_.out
	    /((float)(distribute1_.in+distribute1_.out));
	if (tmp>distribute1_.delta) distribute1_.delta=tmp;
	tmp=(float)distribute2_.out
	    /((float)(distribute2_.in+distribute2_.out));
	if (tmp>distribute2_.delta) distribute2_.delta=tmp;
	if (distribute1_.out>distribute1_.tot){
	  distribute1_.tot=distribute1_.out;
	}
	if (distribute2_.out>distribute2_.tot){
	  distribute2_.tot=distribute2_.out;
	}
    }
    if (what==3){
      	printf("miss_1=%f;miss_2=%f;\n",distribute1_.delta,distribute2_.delta);
    }
}




/*
  Distributes the beam particles for field calculation
*/

void GRID::distribute_particles(int i_slice1,
				 int i_slice2, 
				 int electron_distribution_rho, 
				 int force_symmetric)
{
  
  slice_of_beam_[0].razRho();
  slice_of_beam_[1].razRho();
  //  razRhos(); 

  PARALLEL_BACKEND& backend = parallel_backend();
  // Configure the one shared CUDA state on the caller thread before any
  // pool_dual workers are released. CPU and GPU backends are mutually
  // exclusive, so worker tasks only observe the stable disabled state.
  configure_gpu_cic();
  if (backend.cpu_enabled() && backend.actual_cpu_variant() == "pool_dual")
    {
      // The two primary beams write disjoint rho arrays and counters. Each
      // task retains the original particle order, so this is the strict,
      // bit-compatible accelerated deposition candidate.
      backend.run_tasks(2U, [&](std::size_t beam) {
        if (beam == 0U)
          {
            if (electron_distribution_rho == 1)
              assignBeamSliceNGP(slice_of_beam_[0], i_slice1, distribute1_);
            else if (electron_distribution_rho == 2)
              assignBeamSliceCIC(slice_of_beam_[0], i_slice1, distribute1_);
          }
        else
          {
            if (electron_distribution_rho == 1)
              assignBeamSliceNGP(slice_of_beam_[1], i_slice2, distribute2_);
            else if (electron_distribution_rho == 2)
              assignBeamSliceCIC(slice_of_beam_[1], i_slice2, distribute2_);
          }
      });
    }
#ifdef FEEBL_HAVE_OPENMP
  else if (backend.gpu_mode2_enabled() &&
           backend.gpu2_plan_committed() &&
           backend.actual_gpu2_options().host_threads == 2 &&
           electron_distribution_rho == 2)
    {
      // The only dependency-safe hybrid host region in the initial GPU2 plan:
      // each section owns one rho array and one DISTRIBUTE counter, and keeps
      // historical particle order inside that beam.  Coherent/trident sources,
      // symmetry, luminosity, RNG and result accumulation all remain serial
      // below this region.
#pragma omp parallel sections num_threads(2)
      {
#pragma omp section
        assignBeamSliceCIC(slice_of_beam_[0], i_slice1, distribute1_);
#pragma omp section
        assignBeamSliceCIC(slice_of_beam_[1], i_slice2, distribute2_);
      }
    }
#endif
  else if (backend.cpu_enabled() && electron_distribution_rho == 2 &&
           backend.actual_cpu_variant() == "pool_private")
    {
      assignBeamSliceCICPrivate(slice_of_beam_[0], i_slice1, distribute1_);
      assignBeamSliceCICPrivate(slice_of_beam_[1], i_slice2, distribute2_);
    }
  else if (backend.cpu_enabled() && electron_distribution_rho == 2 &&
           backend.actual_cpu_variant() == "omp_atomic")
    {
      assignBeamSliceCICAtomic(slice_of_beam_[0], i_slice1, distribute1_);
      assignBeamSliceCICAtomic(slice_of_beam_[1], i_slice2, distribute2_);
    }
  else
    {
      // Serial default, plus the safe NGP fallback for CPU candidates whose
      // optimized kernel currently covers CIC only.
      switch (electron_distribution_rho)
        {
        case 1:
          assignBeamSliceNGP(slice_of_beam_[0], i_slice1, distribute1_);
          assignBeamSliceNGP(slice_of_beam_[1], i_slice2, distribute2_);
          break;
        case 2:
          assignBeamSliceCIC(slice_of_beam_[0], i_slice1, distribute1_);
          assignBeamSliceCIC(slice_of_beam_[1], i_slice2, distribute2_);
          break;
        }
    }

  distribute_coherent_particles(i_slice1, i_slice2, electron_distribution_rho);
  distribute_trident_particles(i_slice1, i_slice2, electron_distribution_rho);  
  if (force_symmetric) 
    {
      slice_of_beam_[0].symmetrizeCharges(n_cell_x_,n_cell_y_);
      slice_of_beam_[1].symmetrizeCharges(n_cell_x_,n_cell_y_);
    }
  slice_of_beam_[0].finalizeActiveRhoCells();
  slice_of_beam_[1].finalizeActiveRhoCells();
}




/*
  Distribute the coherent pair particles for the calculation of the fields
 */

void GRID::distribute_coherent_particles(int i_slice1,
					  int i_slice2,
					  int electron_distribution_rho)
{
  switch (electron_distribution_rho){
    case 1:


      assignCoherentBeamSliceNGP(slice_of_beam_[0], i_slice1, distribute1_);
      assignCoherentBeamSliceNGP(slice_of_beam_[1], i_slice2, distribute2_);
      break;
  case 2:

    assignCoherentBeamSliceCIC(slice_of_beam_[0], i_slice1, distribute1_);
    assignCoherentBeamSliceCIC(slice_of_beam_[1], i_slice2, distribute2_);
    break;
  }
  
}

/*
  Distribute the trident pair particles for the calculation of the fields
 */
void GRID::distribute_trident_particles(int i_slice1,int i_slice2,int electron_distribution_rho)
{
  switch (electron_distribution_rho){
  case 1:
    assignTridentBeamSliceNGP(slice_of_beam_[0], i_slice1, distribute1_);
    assignTridentBeamSliceNGP(slice_of_beam_[1], i_slice2, distribute2_);
    break;
  case 2:
    assignTridentBeamSliceCIC(slice_of_beam_[0], i_slice1, distribute1_);
    assignTridentBeamSliceCIC(slice_of_beam_[1], i_slice2, distribute2_);
    break;
  }
}

//! Distributes the particles for background calculation 

void GRID::distribute_particles_for_background(int i_slice1,
				 int i_slice2,  
				 float electron_ratio)
{
  float ratio,ratio_i_1,ratio_i_2;
  const float eps=1e-5;

  particle_cells1_.clear_touched(part_pointer1_);
  particle_cells2_.clear_touched(part_pointer2_);

  ratio=electron_ratio;
  if (ratio<eps) return;
  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();

  ratio_i_1=1e9/ratio*slice_of_beam_[0].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  ratio_i_2=1e9/ratio*slice_of_beam_[1].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  
  const BEAM* beam1 = slice_of_beam_[0].get_beam();
  const BEAM* beam2 = slice_of_beam_[1].get_beam();
  const std::vector<PRIMARY_PARTICLE_ID>* ids1 = 0;
  const std::vector<PRIMARY_PARTICLE_ID>* ids2 = 0;
  if (electron_lumi_stable_ids_enabled_)
    {
      if (electron_lumi_stable_ids1_ == NULL ||
          electron_lumi_stable_ids2_ == NULL || i_slice1 < 0 || i_slice2 < 0 ||
          static_cast<std::size_t>(i_slice1) >=
              electron_lumi_stable_ids1_->size() ||
          static_cast<std::size_t>(i_slice2) >=
              electron_lumi_stable_ids2_->size())
        throw std::logic_error(
            "electron-luminosity stable-ID sidecar preflight is incomplete");
      ids1 = &(*electron_lumi_stable_ids1_)[static_cast<std::size_t>(i_slice1)];
      ids2 = &(*electron_lumi_stable_ids2_)[static_cast<std::size_t>(i_slice2)];
    }
  distributeScatter1(beam1->getParticleVector(i_slice1), ratio, ratio_i_1,
                     part_pointer1_, particle_cells1_,
                     beam1->xy_adopted()
                         ? beam1->slice_xy(static_cast<std::size_t>(i_slice1))
                         : 0,
                     ids1);
  distributeScatter1(beam2->getParticleVector(i_slice2), ratio, ratio_i_2,
                     part_pointer2_, particle_cells2_,
                     beam2->xy_adopted()
                         ? beam2->slice_xy(static_cast<std::size_t>(i_slice2))
                         : 0,
                     ids2);
  distribute_coherent_particles_for_background(i_slice1, i_slice2, electron_ratio);
  distribute_trident_particles_for_background(i_slice1, i_slice2, electron_ratio);
  particle_cells1_.sort_cells();
  particle_cells2_.sort_cells();
}

//
void GRID::distributeScatter1(
    const std::vector<PARTICLE*>& theParticles, float ratio, float ratio_i,
    std::vector<std::vector<BEAM_PARTICLE_POINTER> >& part_pointer,
    feebl::sparse_cells::ActiveCellSet& active_cells, const float* xy,
    const std::vector<PRIMARY_PARTICLE_ID>* stable_ids)
{
  if (stable_ids != 0 && stable_ids->size() != theParticles.size())
    throw std::logic_error("electron-luminosity stable-ID sidecar shape mismatch");

  float xpart, ypart;
  int i1, i2;
  //  int k,j;
  int j;
  unsigned int k;
  //    const std::vector<PARTICLE*>& theParticles = slice_of_beam_[i_beam-1].get_beam()->getParticleVector(i_slice);
    for (k = 0; k < theParticles.size(); k++)
    {
      if (rndm_generator_->rndm()<ratio)
	{
	  // FEEBL: positions from the beam's contiguous array when it owns them,
	  // avoiding a dereference into a 104-byte object for 8 bytes. The
	  // PARTICLE* is still needed for BEAM_PARTICLE_POINTER, but reading it
	  // out of the vector is sequential and cheap.
	  if (xy) { xpart = xy[2*k]; ypart = xy[2*k+1]; }
	  else      theParticles[k]->XYposition(xpart, ypart);
	  if (particleInGrid(xpart, ypart, i1, i2))
	    {
	      j=i1*n_cell_y_+i2;
	      active_cells.touch(j);
	      const std::uint64_t stable_id = stable_ids == 0
	          ? std::numeric_limits<std::uint64_t>::max()
	          : (*stable_ids)[k];
	      part_pointer[j].push_back(
	          BEAM_PARTICLE_POINTER(theParticles[k], ratio_i, stable_id));
	    }
	}
    }
}

// This method is not used, uses time to calculate.
// Don't delete for the moment

void GRID::distributeScatter2(
    const std::vector<PARTICLE*>& theParticles, float ratio, float ratio_i,
    std::vector<std::vector<BEAM_PARTICLE_POINTER> >& part_pointer,
    feebl::sparse_cells::ActiveCellSet& active_cells)
{

  float xpart, ypart,poids,h_x,h_y;
  int i1, i2;
  int j;
  unsigned int k;

  //  const std::vector<PARTICLE*>& theParticles = slice_of_beam_[i_beam-1].get_beam()->getParticleVector(i_slice);
    for (k = 0; k < theParticles.size(); k++)
      {
	theParticles[k]->XYposition(xpart, ypart);
	if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
	  {
	    if (rndm_generator_->rndm()<ratio)
	      {
		j=i1*n_cell_y_+i2;
		poids = (1.0-h_x)*(1.0-h_y)*ratio_i;

	      active_cells.touch(j);
	      part_pointer[j].push_back(BEAM_PARTICLE_POINTER(theParticles[k], poids));
		j=(i1+1)*n_cell_y_+i2;
		poids =  h_x*(1.0-h_y)*ratio_i;
	      active_cells.touch(j);
	      part_pointer[j].push_back(BEAM_PARTICLE_POINTER(theParticles[k], poids));
		j=i1*n_cell_y_+i2+1;
		poids =  (1.0-h_x)*h_y*ratio_i;
	      active_cells.touch(j);
	      part_pointer[j].push_back(BEAM_PARTICLE_POINTER(theParticles[k], poids));
		j=(i1+1)*n_cell_y_+i2+1;
		poids = h_x*h_y*ratio_i;
	      active_cells.touch(j);
			      part_pointer[j].push_back(BEAM_PARTICLE_POINTER(theParticles[k], poids));
	      }
	  }
      }
}


//  Distributes the particles from coherent pair creation for the background
//  calculation
 

void GRID::distribute_coherent_particles_for_background(int i_slice1,
					  int i_slice2,
					  float electron_ratio)

{

  float ratio,ratio_i_1,ratio_i_2;
  const float eps=1e-5;

  ratio=electron_ratio;
  if (ratio<eps) return;
  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();

  ratio_i_1=1e9/ratio*slice_of_beam_[0].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  ratio_i_2=1e9/ratio*slice_of_beam_[1].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
    distributeScatter1(slice_of_beam_[0].get_beam()->getCoherentVector(i_slice1), ratio, ratio_i_1, part_pointer1_, particle_cells1_);
        distributeScatter1(slice_of_beam_[1].get_beam()->getCoherentVector(i_slice2),ratio, ratio_i_2, part_pointer2_, particle_cells2_);
}

//  Distributes the particles from trident pair creation for the background
//  calculation
void GRID::distribute_trident_particles_for_background(int i_slice1,int i_slice2,float electron_ratio)
{
  float ratio,ratio_i_1,ratio_i_2;
  const float eps=1e-5;

  ratio=electron_ratio;
  if (ratio<eps) return;
  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();

  ratio_i_1=1e9/ratio*slice_of_beam_[0].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  ratio_i_2=1e9/ratio*slice_of_beam_[1].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  distributeScatter1(slice_of_beam_[0].get_beam()->getTridentVector(i_slice1), ratio, ratio_i_1, part_pointer1_, particle_cells1_);
  distributeScatter1(slice_of_beam_[1].get_beam()->getTridentVector(i_slice2), ratio, ratio_i_2, part_pointer2_, particle_cells2_);
}

//! Distributes the virtual photons 

void GRID::distribute_virtual_photons(int i_slice1,
				 int i_slice2,
				 SWITCHES* switches,
				 double s4, double lns4)
{
   const float   emass2=EMASS*EMASS;
  float xmin,r_scal;
  float ratio,ratio_i_1,ratio_i_2;
    int geom;

  r_scal=switches->get_r_scal();
  geom=switches->get_geom();
  ratio=switches->get_electron_ratio();
  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();
  ratio_i_1=1e9/ratio*slice_of_beam_[0].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  ratio_i_2=1e9/ratio*slice_of_beam_[1].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  clear_extra_photons();
  //   extra_photons1_.clear_extra_photons();
  //   extra_photons2_.clear_extra_photons();
  //  switch (switches->get_electron_distribution_scatter())
  //   {
  //    case 1:
  xmin=switches->get_compt_x_min()*emass2/s4;

  // equivalent-photon spectrum: legacy iflag 6 (pair_model 0 and 1; pair_model 1 corrects it by a
  // weight in the pair channels only), or the full-range proposal of the exact kernels (pair_model 2,
  // iflag 11, identical to iflag 6 below Q^2 = s4): the Compton and gamma* hadron channels see the
  // legacy flux in every pair_model
  const int i_equiv = (switches->get_pair_model() == 2) ? 11 : 6;
  electronScatter(extra_photon_pointer1_, extra_photon_cells1_, 1, i_slice1, xmin, s4, lns4, ratio, switches->get_ext_field(), geom, ratio_i_1, r_scal, i_equiv);

  electronScatter(extra_photon_pointer2_, extra_photon_cells2_, 2, i_slice2, xmin, s4, lns4, ratio, switches->get_ext_field(), geom, ratio_i_2, r_scal, i_equiv);
  extra_photon_cells1_.sort_cells();
  extra_photon_cells2_.sort_cells();
  

	  
  //     break;
  // case 2:
  
  //   distributeElectronScatter2(beam1_, i_slice1, ratio, ratio_i_1, part_pointer1_, rndm_generator);
  //   distributeElectronScatter2(beam2_, i_slice2, ratio, ratio_i_2, part_pointer2_, rndm_generator);
  //   break;
  //   }
}

void GRID::electronScatter(
    std::vector<std::vector<EXTRA_PHOTON_POINTER> >& extra_phot_ptr,
    feebl::sparse_cells::ActiveCellSet& active_cells,
    int i_beam, int i_slice, float xmin, double s4, double lns4,
    float ratio, int ext_field, int geom, float ratio_i, float r_scal,
    int i_equiv)
{
  unsigned int k;
  int i1, i2,j;
  int n_phot, i_phot;
  float energy, r_phot;
  float e_phot=0.0,q2=0.0,one_m_x=1.0,x=0.0,y=0.0, radius, theta;  // set by mequiv for iflag 6 and 11
  float xVelocity, yVelocity;
  //PHYSTOOLS phys;

    const std::vector<PARTICLE*>& theParticles = slice_of_beam_[i_beam-1].get_beam()->getParticleVector(i_slice);
    for (k = 0; k < theParticles.size(); k++)
      //  for (i=beam->firstParticleOfSlice(i_slice);i<beam->firstParticleOfSlice(i_slice+1);i++)
    {
      //      energy=beam->energyOfParticle(i);
      energy=theParticles[k]->energy();
      //r_phot=phys.requiv(lns4,xmin,i_equiv)*ratio;
      r_phot=PHYSTOOLS::requiv(lns4,xmin,i_equiv)*ratio;
      n_phot=(int)floor(r_phot);
      r_phot -= n_phot;
      if(rndm_generator_->rndm()<r_phot) n_phot++;
      for (i_phot=0;i_phot<n_phot;i_phot++)
	{

	  //phys.mequiv(s4, lns4,xmin,energy,i_equiv,&e_phot,&q2,&one_m_x, *rndm_generator_);
	  PHYSTOOLS::mequiv(s4, lns4,xmin,energy,i_equiv,&e_phot,&q2,&one_m_x, *rndm_generator_);
	  //#ifdef EXT_FIELD
	  if (ext_field && (pow(q2/(EMASS*EMASS),1.5)*energy*energy < e_phot*e_phot*theParticles[k]->getUps()) )
	    {
	      e_phot=-1.0;
	    }
	  //#endif
	  if (e_phot>0.0)
	    {
	      switch(geom)
		{
		case 0:
		  theParticles[k]->XYposition(x, y);
		  break;
		case 1:
		  radius=HBAR*Cvelocity/sqrt(q2*one_m_x)*r_scal*1e9;
		  radius=std::min(radius,float(1.0e5));

		  theParticles[k]->XYposition(x, y);
		  x += rndm_generator_->rndm_sincos(&theta)*radius;
		  y += theta*radius;
		  break;
		case 2:
		  radius=HBAR*Cvelocity/sqrt(q2*one_m_x)*r_scal*1e9;
		  radius=std::min(radius,float(1e5));
		  theParticles[k]->XYposition(x, y);
		  x += rndm_generator_->gasdev()*radius;
		  y += rndm_generator_->gasdev()*radius;
		  break;
		case 3:
		  {
		    // exact impact-parameter profile (qedbg_impact_parameter.h): b = (1-x) xi/sqrt(D) from the
		    // emitter, D = (1-x) Q^2 = q_T^2 + x^2 m^2, xi from the helicity-conserving or -flip shape
		    // chosen with its share of the exact density at this q_T
		    const double xph = (double)e_phot/energy;
		    const double d = (double)q2*one_m_x;
		    const double eps2 = xph*xph*EMASS*EMASS;
		    const double nonflip = (1.0 + (double)one_m_x*one_m_x)*std::max(d - eps2, 0.0);
		    const double flip = xph*xph*eps2;
		    const int term = (rndm_generator_->rndm()*(nonflip + flip) < flip) ? 1 : 0;
		    const double xi = qedbg::ImpactParameterProfile::instance().xi(term, rndm_generator_->rndm());
		    radius = (float)(one_m_x*xi*HBAR*Cvelocity/std::sqrt(d)*r_scal*1e9);
		    // a photon 10^6 cells from its emitter is off the grid; the clip keeps particleInGrid's
		    // float-to-int conversion in range (the profile itself is not clipped)
		    radius = std::min(radius, 1.0e6f*std::min(mesh_.get_delta_x(), mesh_.get_delta_y()));
		    theParticles[k]->XYposition(x, y);
		    x += rndm_generator_->rndm_sincos(&theta)*radius;
		    y += theta*radius;
		    break;
		  }
		}
	      if (particleInGrid(x, y, i1, i2))
		{
		  j=i1*n_cell_y_+i2;
		  theParticles[k]->velocities(xVelocity, yVelocity);
		  active_cells.touch(j);
		  extra_phot_ptr[j].push_back(EXTRA_PHOTON_POINTER(e_phot, xVelocity, yVelocity,q2,energy,ratio_i));
		  //		  photon->store_vir_photon(e_phot,xVelocity, yVelocity,q2,energy,ratio_i,j);
		}
	      
	    }
	}
    }
}





void GRID::assignBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{

  unsigned int k;
  int i1, i2;
  float xpart, ypart;
  const std::vector<PARTICLE*>& theParticles = sog.get_beam()->getParticleVector(i_slice);    
  for (k = 0; k < theParticles.size(); k++)
    {
      theParticles[k]->XYposition(xpart, ypart);
      if (particleInGrid(xpart, ypart, i1, i2))
	{
	  sog.assignChargeToNGP( i1, i2, 1.0);
	  distribute.in++;
	}
      else   distribute.out++;		
    }
}


void GRID::assignBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{
  // n_macro : nb of particles per macroparticle
  unsigned int k;
  int i1, i2;
  float h_x,h_y;
  float xpart, ypart;
const std::vector<PARTICLE*>& theParticles = sog.get_beam()->getParticleVector(i_slice);

  // A one-time O(number_of_slices) scan proves whether this whole collision
  // can ever reach the measured crossover.  If not, lock serial immediately:
  // no CUDA context and no threshold branch in the remaining hot calls.
  if (gpu_cic_auto_ && !gpu_cic_auto_scanned_)
    {
      gpu_cic_auto_scanned_ = true;
      for (int beam = 0; beam < 2; ++beam)
        {
          const BEAM* candidate_beam = slice_of_beam_[beam].get_beam();
          if (!candidate_beam) continue;
          for (int slice = 0; slice < candidate_beam->number_of_slices(); ++slice)
            gpu_cic_max_slice_particles_ = std::max(
                gpu_cic_max_slice_particles_,
                static_cast<std::size_t>(
                    candidate_beam->numberOfParticlesOfSlice(slice)));
        }
      if (gpu_cic_max_slice_particles_ < gpu_cic_min_particles_)
        {
          gpu_cic_auto_locked_serial_ = true;
          gpu_cic_mode_ = -1;
          parallel_backend().runtime_fallback_to_serial(
              "GPU auto threshold selected serial CIC for every slice");
          std::cerr << "FEEBL CUDA CIC auto locked serial: maximum slice="
                    << gpu_cic_max_slice_particles_ << " is below threshold="
                    << gpu_cic_min_particles_ << std::endl;
        }
    }
  const bool auto_small_slice =
      gpu_cic_auto_ && gpu_cic_mode_ >= 1 &&
      theParticles.size() < gpu_cic_min_particles_;
  if (auto_small_slice)
    {
      ++gpu_cic_fallback_calls_;
      gpu_cic_fallback_particles_ += theParticles.size();
    }
  else if (gpu_cic_mode_ == 2)
    {
      std::string error;
      if (!enable_gpu_cic("atomic", gpu_cic_device_, error))
        {
          // gpu_variant=auto reaches this state only when explicit fallback
          // was authorized during central backend configuration.
          std::cerr << "FEEBL CUDA CIC auto fallback to serial: "
                    << error << std::endl;
          parallel_backend().runtime_fallback_to_serial(
              "CUDA auto initialization failed; selected serial CIC: " + error);
          gpu_cic_mode_ = -1;
        }
      else
        {
          std::cerr << "FEEBL CUDA CIC auto selected GPU at "
                    << theParticles.size() << " particles: device="
                    << gpu_cic_->device_description() << std::endl;
        }
    }
  if (gpu_cic_mode_ == 1 && !auto_small_slice)
    {
      const std::size_t is = static_cast<std::size_t>(i_slice);
      const float* xy = NULL;
      std::vector<float> packed_xy;
      if (sog.get_beam()->xy_adopted() &&
          sog.get_beam()->slice_xy_count(is) == theParticles.size())
        {
          xy = sog.get_beam()->slice_xy(is);
        }
      else
        {
          // Loaded/arbitrary bunch paths are supported even if they have not
          // adopted the contiguous position buffer yet.  Packing is measured
          // in the caller's end-to-end wall time and is not hidden in GPU
          // kernel timings.
          packed_xy.resize(2 * theParticles.size());
          ++gpu_cic_packed_calls_;
          gpu_cic_packed_particles_ += theParticles.size();
          for (std::size_t particle = 0; particle < theParticles.size(); ++particle)
            theParticles[particle]->XYposition(packed_xy[2 * particle],
                                                packed_xy[2 * particle + 1]);
          xy = packed_xy.empty() ? NULL : &packed_xy[0];
        }

      feebl::gpu::CicStats stats;
      std::string error;
      if (!gpu_cic_->deposit(xy, theParticles.size(),
                             static_cast<double>(sog.get_nb_part_per_macro()),
                             n_cell_x_, n_cell_y_,
                             delta_x_inverse_, delta_y_inverse_,
                             mesh_.get_offset_x(), mesh_.get_offset_y(),
                             gpu_cic_variant_, sog.get_rho_mutable(),
                             stats, error))
        {
          std::cerr << "CUDA CIC deposition failed: " << error << std::endl;
          std::exit(2);
        }
      distribute.in += static_cast<long int>(stats.particles_in);
      distribute.out += static_cast<long int>(stats.particles_out);
      parallel_backend().record_gpu_execution(gpu_cic_->device_description());
      ++gpu_cic_calls_;
      gpu_cic_setup_ms_ += stats.setup_ms;
      gpu_cic_h2d_ms_ += stats.h2d_ms;
      gpu_cic_kernel_ms_ += stats.kernel_ms;
      gpu_cic_d2h_ms_ += stats.d2h_ms;
      gpu_cic_total_ms_ += stats.total_ms;
      // Device deposition writes the dense output array directly, bypassing
      // the scalar tracked helpers. Rebuild once so subsequent sparse clear
      // and luminosity traversal remain complete.
      sog.rebuildActiveRhoCells();
      return;
    }

  // Read x,y directly from the beam's contiguous
  // array. Same slice, same index order, same arithmetic -- only the data
  // source changes, so the result is bit-identical. There is no mirror and no
  // sync: this IS the particles' position storage.
  if (sog.get_beam()->xy_adopted())
    {
      const std::size_t is = (std::size_t)i_slice;
      const std::size_t nxy = sog.get_beam()->slice_xy_count(is);
      if (nxy == theParticles.size())
        {
          const float* xy = sog.get_beam()->slice_xy(is);
          for (std::size_t kk = 0; kk < nxy; kk++)
            {
              if (particleInGrid(xy[2*kk], xy[2*kk+1], h_x, h_y, i1, i2))
                { sog.assignChargeToCIC(i1, i2, h_x, h_y, 1.0); distribute.in++; }
              else distribute.out++;
            }
          return;
        }
    }

 for (k = 0; k < theParticles.size(); k++)
	{
	theParticles[k]->XYposition(xpart, ypart);
	  if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
	    {
	      sog.assignChargeToCIC( i1, i2, h_x, h_y, 1.0);
	      distribute.in++;	    
	    }
	  else
	    {
	      //	      std::cout << " particle no "  << k+1 << " slice " << i_slice << " out of grid : x= " << xpart << " y= " << ypart << std::endl;
	      distribute.out++;
	    }
	  
	}
}

void GRID::assignBeamSliceCICAtomic(SLICE_ON_GRID& sog, int i_slice,
                                    DISTRIBUTE& distribute)
{
#ifdef FEEBL_HAVE_OPENMP
  const std::vector<PARTICLE*>& particles =
      sog.get_beam()->getParticleVector(i_slice);
  const bool adopted = sog.get_beam()->xy_adopted() &&
      sog.get_beam()->slice_xy_count(static_cast<std::size_t>(i_slice)) ==
          particles.size();
  const float* xy = adopted
      ? sog.get_beam()->slice_xy(static_cast<std::size_t>(i_slice)) : NULL;
  PHI_FLOAT* rho = sog.get_rho_mutable();
  const PHI_FLOAT charge = sog.get_nb_part_per_macro();
  const int ny = sog.get_nb_cells_y();
  long in_count = 0;
  long out_count = 0;
  const long long count = static_cast<long long>(particles.size());

#pragma omp parallel for schedule(static) \
    num_threads(parallel_backend().actual_cpu_threads()) \
    reduction(+:in_count,out_count)
  for (long long kk = 0; kk < count; ++kk)
    {
      float xpart, ypart, h_x, h_y;
      int i1, i2;
      if (adopted)
        {
          xpart = xy[2*kk];
          ypart = xy[2*kk+1];
        }
      else
        particles[static_cast<std::size_t>(kk)]->XYposition(xpart, ypart);

      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
        {
          int j = i1*ny+i2;
          PHI_FLOAT value = (1.0-h_x)*(1.0-h_y)*charge;
#pragma omp atomic update
          rho[j] += value;
          j = (i1+1)*ny+i2;
          value = h_x*(1.0-h_y)*charge;
#pragma omp atomic update
          rho[j] += value;
          j = i1*ny+i2+1;
          value = (1.0-h_x)*h_y*charge;
#pragma omp atomic update
          rho[j] += value;
          j = (i1+1)*ny+i2+1;
          value = h_x*h_y*charge;
#pragma omp atomic update
          rho[j] += value;
          ++in_count;
        }
      else
        ++out_count;
    }
  distribute.in += in_count;
  distribute.out += out_count;
  sog.rebuildActiveRhoCells();
#else
  // configure() rejects this selection in a non-OpenMP build. Keep a safe
  // guard here as well in case the function is called directly in a test.
  assignBeamSliceCIC(sog, i_slice, distribute);
#endif
}

void GRID::assignBeamSliceCICPrivate(SLICE_ON_GRID& sog, int i_slice,
                                     DISTRIBUTE& distribute)
{
  const std::vector<PARTICLE*>& particles =
      sog.get_beam()->getParticleVector(i_slice);
  const std::size_t count = particles.size();
  if (count == 0) return;

  const std::size_t cells = static_cast<std::size_t>(sog.get_dim_rho());
  // Logical shards depend on the workload, never on cpu_threads. This keeps
  // the summation tree invariant when the physical worker count changes.
  std::size_t shards = (count + 2047U) / 2048U;
  if (shards < 1U) shards = 1U;
  if (shards > 32U) shards = 32U;
  // Bound retained scratch to 64 MiB for one main-grid deposition.
  const std::size_t bytes_per_grid = cells*sizeof(PHI_FLOAT);
  const std::size_t memory_shards = bytes_per_grid == 0 ? 1U
      : std::max<std::size_t>(1U, (64U*1024U*1024U)/bytes_per_grid);
  if (shards > memory_shards) shards = memory_shards;

  const std::size_t scratch_size = shards*cells;
  if (cpu_private_rho_.size() < scratch_size)
    cpu_private_rho_.resize(scratch_size);
  std::vector<long> in_count(shards, 0);
  std::vector<long> out_count(shards, 0);
  const bool adopted = sog.get_beam()->xy_adopted() &&
      sog.get_beam()->slice_xy_count(static_cast<std::size_t>(i_slice)) == count;
  const float* xy = adopted
      ? sog.get_beam()->slice_xy(static_cast<std::size_t>(i_slice)) : NULL;

  parallel_backend().run_tasks(shards, [&](std::size_t shard) {
    PHI_FLOAT* local = &cpu_private_rho_[shard*cells];
    std::fill(local, local+cells, 0.0);
    const std::size_t begin = count*shard/shards;
    const std::size_t end = count*(shard+1U)/shards;
    long local_in = 0;
    long local_out = 0;
    for (std::size_t kk = begin; kk < end; ++kk)
      {
        float xpart, ypart, h_x, h_y;
        int i1, i2;
        if (adopted)
          {
            xpart = xy[2U*kk];
            ypart = xy[2U*kk+1U];
          }
        else
          particles[kk]->XYposition(xpart, ypart);
        if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
          {
            sog.assignChargeToCICBuffer(local, i1, i2, h_x, h_y, 1.0);
            ++local_in;
          }
        else
          ++local_out;
      }
    in_count[shard] = local_in;
    out_count[shard] = local_out;
  });

  PHI_FLOAT* target = sog.get_rho_mutable();
  const std::size_t block_size = 4096U;
  const std::size_t blocks = (cells+block_size-1U)/block_size;
  parallel_backend().run_tasks(blocks, [&](std::size_t block) {
    const std::size_t begin = block*block_size;
    const std::size_t end = std::min(cells, begin+block_size);
    for (std::size_t cell = begin; cell < end; ++cell)
      {
        PHI_FLOAT sum = 0.0;
        for (std::size_t shard = 0; shard < shards; ++shard)
          sum += cpu_private_rho_[shard*cells+cell];
        target[cell] = sum;
      }
  });

  for (std::size_t shard = 0; shard < shards; ++shard)
    {
      distribute.in += in_count[shard];
      distribute.out += out_count[shard];
    }
  sog.rebuildActiveRhoCells();
}


void GRID::assignCoherentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  unsigned int particle;
  const std::vector<PARTICLE*>& coherent = sog.get_beam()->getCoherentVector(i_slice);
  for (particle = 0; particle < coherent.size(); particle++)
    {
      coherent[particle]->XYposition(xpart, ypart);
      if (coherent[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      
      if (particleInGrid(xpart, ypart,i1, i2))
	{
	  
	  sog.assignChargeToNGP( i1, i2,ch);
	  distribute.in++;
	}
      else
	{
	  distribute.out++;
	}
    }
}

void GRID::assignCoherentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  float h_x,h_y;
  unsigned int particle;
  const std::vector<PARTICLE*>& coherent = sog.get_beam()->getCoherentVector(i_slice);
  for (particle = 0; particle < coherent.size(); particle++)
    {
      coherent[particle]->XYposition(xpart, ypart);
      if (coherent[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;      
      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
	{
	  sog.assignChargeToCIC(i1, i2, h_x, h_y, ch);
	  distribute.in++;
	}
      else{
	distribute.out++;
      }
    }
}

void GRID::assignTridentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  unsigned int particle;
  const std::vector<PARTICLE*>& trident = sog.get_beam()->getTridentVector(i_slice);
  for (particle = 0; particle < trident.size(); particle++)
    {
      trident[particle]->XYposition(xpart, ypart);
      if (trident[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      
      if (particleInGrid(xpart, ypart,i1, i2))
	{
	  
	  sog.assignChargeToNGP( i1, i2,ch);
	  distribute.in++;
	}
      else
	{
	  distribute.out++;
	}
    }
}

void GRID::assignTridentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  float h_x,h_y;
  unsigned int particle;
  const std::vector<PARTICLE*>& trident = sog.get_beam()->getTridentVector(i_slice);
  for (particle = 0; particle < trident.size(); particle++)
    {
      trident[particle]->XYposition(xpart, ypart);
      if (trident[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;      
      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2))
	{
	  sog.assignChargeToCIC(i1, i2, h_x, h_y, ch);
	  distribute.in++;
	}
      else{
	distribute.out++;
      }
    }
}

/*! This routine calculates the luminosity from the collision of the two
slices i1 and i2 of grid. The dimensions of the grids have to be the same. */

void GRID::step_lumi(float min_z, PAIR_BEAM& secondaries,int time_counter, SWITCHES& switches, int beamslice1=-1, int beamslice2=-1)
{
  if (switches.get_electron_lumi_spatial_kernel() == 1)
    {
      step_lumi_spatial(switches, beamslice1, beamslice2);
      return;
    }
  if (switches.get_electron_lumi_precompute() == 1)
    {
      step_lumi_precomputed(switches, false, 0.0);
      return;
    }
  if (switches.get_electron_lumi_precompute() == 2 ||
      switches.get_electron_lumi_precompute() == 3)
    {
      step_lumi_blocked(switches, false, 0.0, min_z, time_counter);
      return;
    }

  float sum=0.0;
  int i1,i2,j;
  std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer1,pointer2;

  const PHI_FLOAT* rho1 = slice_of_beam_[0].get_rho();
  const PHI_FLOAT* rho2 = slice_of_beam_[1].get_rho();
  if (!sparse_cells_enabled_)
    {
      for (i1=0;i1 < n_cell_x_;i1++)
	{
	  for (i2=0;i2 < n_cell_y_;i2++)
	    {
	      j=i1*n_cell_y_+i2;
	      sum += rho1[j]*rho2[j];
	      for (pointer1=part_pointer1_[j].begin(); pointer1 != part_pointer1_[j].end(); pointer1++)
		{
		  for( pointer2 = part_pointer2_[j].begin(); pointer2 != part_pointer2_[j].end(); pointer2++)
		    {
		      collide_ee(i1, i2, min_z, *pointer1, *pointer2,switches, secondaries,time_counter, beamslice1, beamslice2 ) ;
		    }
		}
	    }
	}
    }
  else
    {
      feebl::sparse_cells::intersect_sorted(
          slice_of_beam_[0].activeRhoCells(),
          slice_of_beam_[1].activeRhoCells(), &sparse_cell_scratch_);
      feebl::sparse_cells::intersect_sorted(
          particle_cells1_.cells(), particle_cells2_.cells(),
          &electron_lumi_occupied_cells_);
      const std::vector<int>* active_sets[] = {
          &sparse_cell_scratch_, &electron_lumi_occupied_cells_};
      feebl::sparse_cells::sorted_union(active_sets, 2U,
                                        &sparse_traversal_cells_);
      sparse_dense_traversal_visits_ +=
          static_cast<unsigned long long>(n_cell_x_)*n_cell_y_;
      sparse_actual_traversal_visits_ +=
          static_cast<unsigned long long>(sparse_traversal_cells_.size());
      for (std::vector<int>::const_iterator active =
               sparse_traversal_cells_.begin();
           active != sparse_traversal_cells_.end(); ++active)
        {
          j = *active;
          i1 = j/n_cell_y_;
          i2 = j-i1*n_cell_y_;
          sum += rho1[j]*rho2[j];
          for (pointer1=part_pointer1_[j].begin();
               pointer1 != part_pointer1_[j].end(); ++pointer1)
            for (pointer2=part_pointer2_[j].begin();
                 pointer2 != part_pointer2_[j].end(); ++pointer2)
              collide_ee(i1, i2, min_z, *pointer1, *pointer2, switches,
                         secondaries, time_counter, beamslice1, beamslice2);
        }
    }

  results_.add_lumi_fine( sum*1e18/(mesh_.get_delta_x()*mesh_.get_delta_y()*timestep_) );	    
}

void GRID::step_lumi_with_fine_dot(float min_z,
                                   PAIR_BEAM& secondaries,
                                   int time_counter,
                                   SWITCHES& switches,
                                   int beamslice1,
                                   int beamslice2,
                                   double fine_lumi_dot,
                                   const feebl::electron_lumi_sampling::PairIdentity*
                                       sampling_identity_prefix)
{
  if (switches.get_electron_lumi_spatial_kernel() == 1)
    {
      // The supplied dot belongs to the mutable force mesh.  Kernel 1 defines
      // both pair enumeration and lumi_fine on its independent physical CIC
      // raster, so substituting that nominal/adaptive-field dot would violate
      // the very invariance this mode exists to provide.
      step_lumi_spatial(switches, beamslice1, beamslice2);
      return;
    }
  if (switches.get_electron_lumi_precompute() == 1)
    {
      step_lumi_precomputed(switches, true, fine_lumi_dot);
      return;
    }
  if (switches.get_electron_lumi_precompute() == 2 ||
      switches.get_electron_lumi_precompute() == 3)
    {
      step_lumi_blocked(
          switches, true, fine_lumi_dot, min_z, time_counter,
          sampling_identity_prefix);
      return;
    }

  std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer1, pointer2;
  if (!sparse_cells_enabled_)
    {
      for (int i1 = 0; i1 < n_cell_x_; ++i1)
        {
          for (int i2 = 0; i2 < n_cell_y_; ++i2)
            {
              const int cell = i1 * n_cell_y_ + i2;
              for (pointer1 = part_pointer1_[cell].begin();
                   pointer1 != part_pointer1_[cell].end(); ++pointer1)
                {
                  for (pointer2 = part_pointer2_[cell].begin();
                       pointer2 != part_pointer2_[cell].end(); ++pointer2)
                    collide_ee(i1, i2, min_z, *pointer1, *pointer2, switches,
                               secondaries, time_counter,
                               beamslice1, beamslice2);
                }
            }
        }
    }
  else
    {
      feebl::sparse_cells::intersect_sorted(
          particle_cells1_.cells(), particle_cells2_.cells(),
          &electron_lumi_occupied_cells_);
      sparse_dense_traversal_visits_ +=
          static_cast<unsigned long long>(n_cell_x_)*n_cell_y_;
      sparse_actual_traversal_visits_ +=
          static_cast<unsigned long long>(electron_lumi_occupied_cells_.size());
      for (std::vector<int>::const_iterator active =
               electron_lumi_occupied_cells_.begin();
           active != electron_lumi_occupied_cells_.end(); ++active)
        {
          const int cell = *active;
          const int i1 = cell/n_cell_y_;
          const int i2 = cell-i1*n_cell_y_;
          for (pointer1 = part_pointer1_[cell].begin();
               pointer1 != part_pointer1_[cell].end(); ++pointer1)
            for (pointer2 = part_pointer2_[cell].begin();
                 pointer2 != part_pointer2_[cell].end(); ++pointer2)
              collide_ee(i1, i2, min_z, *pointer1, *pointer2, switches,
                         secondaries, time_counter, beamslice1, beamslice2);
        }
    }
  results_.add_lumi_fine(
      fine_lumi_dot * 1e18 /
      (mesh_.get_delta_x() * mesh_.get_delta_y() * timestep_));
}

void GRID::step_lumi_precomputed(const SWITCHES& switches,
                                 bool externally_supplied_fine_dot,
                                 double fine_lumi_dot)
{
  std::size_t maximum_beam1 = 0;
  std::size_t maximum_beam2 = 0;
  const int cells = n_cell_x_ * n_cell_y_;
  const PHI_FLOAT* rho1 = externally_supplied_fine_dot
                              ? 0 : slice_of_beam_[0].get_rho();
  const PHI_FLOAT* rho2 = externally_supplied_fine_dot
                              ? 0 : slice_of_beam_[1].get_rho();
  float fine_lumi_sum = 0.0;
  electron_lumi_occupied_cells_.clear();
  if (!sparse_cells_enabled_)
    {
      for (int cell = 0; cell < cells; ++cell)
        {
          maximum_beam1 = std::max(maximum_beam1, part_pointer1_[cell].size());
          maximum_beam2 = std::max(maximum_beam2, part_pointer2_[cell].size());
          if (!externally_supplied_fine_dot)
            fine_lumi_sum += rho1[cell] * rho2[cell];
          if (!part_pointer1_[cell].empty() && !part_pointer2_[cell].empty())
            electron_lumi_occupied_cells_.push_back(cell);
        }
    }
  else
    {
      feebl::sparse_cells::intersect_sorted(
          particle_cells1_.cells(), particle_cells2_.cells(),
          &electron_lumi_occupied_cells_);
      const std::vector<int>* active_sets[3] = {
          &particle_cells1_.cells(), &particle_cells2_.cells(),
          &sparse_cell_scratch_};
      std::size_t active_set_count = 2U;
      if (!externally_supplied_fine_dot)
        {
          feebl::sparse_cells::intersect_sorted(
              slice_of_beam_[0].activeRhoCells(),
              slice_of_beam_[1].activeRhoCells(), &sparse_cell_scratch_);
          active_set_count = 3U;
        }
      feebl::sparse_cells::sorted_union(active_sets, active_set_count,
                                        &sparse_traversal_cells_);
      sparse_dense_traversal_visits_ +=
          static_cast<unsigned long long>(cells);
      sparse_actual_traversal_visits_ +=
          static_cast<unsigned long long>(sparse_traversal_cells_.size());
      for (std::vector<int>::const_iterator active =
               sparse_traversal_cells_.begin();
           active != sparse_traversal_cells_.end(); ++active)
        {
          const int cell = *active;
          maximum_beam1 = std::max(maximum_beam1, part_pointer1_[cell].size());
          maximum_beam2 = std::max(maximum_beam2, part_pointer2_[cell].size());
          if (!externally_supplied_fine_dot)
            fine_lumi_sum += rho1[cell] * rho2[cell];
        }
    }

  // Reserve before changing RESULTS so allocation failure cannot leave a
  // half-accumulated encounter. Capacity is then reused for every cell.
  electron_lumi_cache1_.reserve(maximum_beam1);
  electron_lumi_cache2_.reserve(maximum_beam2);

  const double electron_mass = static_cast<double>(EMASS);

  for (std::vector<int>::const_iterator occupied =
           electron_lumi_occupied_cells_.begin();
       occupied != electron_lumi_occupied_cells_.end(); ++occupied)
    {
      const int cell = *occupied;

          electron_lumi_cache1_.clear();
          electron_lumi_cache2_.clear();

          for (std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer =
                   part_pointer1_[cell].begin();
               pointer != part_pointer1_[cell].end(); ++pointer)
            {
              ELECTRON_LUMI_CACHED_PARTICLE particle;
              particle.signed_energy = pointer->energy();
              particle.absolute_energy = fabs(particle.signed_energy);
              pointer->velocities(particle.vx, particle.vy);
              particle.vz = sqrt(1. - particle.vx * particle.vx -
                                 particle.vy * particle.vy);
              particle.weight = pointer->weight();
              const double energy = static_cast<double>(particle.signed_energy);
              particle.massive_momentum =
                  (energy > electron_mass)
                      ? std::sqrt((energy - electron_mass) *
                                  (energy + electron_mass))
                      : 0.0;
              particle.charge_class = particle.signed_energy < 0.0 ? 1 : 0;
              electron_lumi_cache1_.push_back(particle);
            }

          for (std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer =
                   part_pointer2_[cell].begin();
               pointer != part_pointer2_[cell].end(); ++pointer)
            {
              ELECTRON_LUMI_CACHED_PARTICLE particle;
              particle.signed_energy = pointer->energy();
              particle.absolute_energy = fabs(particle.signed_energy);
              pointer->velocities(particle.vx, particle.vy);
              particle.vz = sqrt(1. - particle.vx * particle.vx -
                                 particle.vy * particle.vy);
              particle.weight = pointer->weight();
              const double energy = static_cast<double>(particle.signed_energy);
              particle.massive_momentum =
                  (energy > electron_mass)
                      ? std::sqrt((energy - electron_mass) *
                                  (energy + electron_mass))
                      : 0.0;
              particle.charge_class = particle.signed_energy < 0.0 ? 0 : 1;
              electron_lumi_cache2_.push_back(particle);
            }

          // Preserve the legacy beam-1-major/beam-2-minor pair order and the
          // exact RESULTS update sequence. Only pair-invariant particle data
          // have moved out of this Cartesian product.
          for (std::vector<ELECTRON_LUMI_CACHED_PARTICLE>::const_iterator
                   particle1 = electron_lumi_cache1_.begin();
               particle1 != electron_lumi_cache1_.end(); ++particle1)
            {
              for (std::vector<ELECTRON_LUMI_CACHED_PARTICLE>::const_iterator
                       particle2 = electron_lumi_cache2_.begin();
                   particle2 != electron_lumi_cache2_.end(); ++particle2)
                {
                  const float weight = particle1->weight * particle2->weight;
                  const float cosinus =
                      particle1->vx * particle2->vx +
                      particle1->vy * particle2->vy -
                      particle1->vz * particle2->vz;
                  const float ecm = sqrt(
                      2.0 * particle1->absolute_energy *
                      particle2->absolute_energy * (1. - cosinus));

                  results_.add_lumi(particle1->charge_class,
                                    particle2->charge_class, weight);

                  if (particle1->signed_energy * particle2->signed_energy >
                      0.0)
                    {
                      results_.add_lumi_ee(weight);
                      results_.add_step_lumi_ee(weight);

                      const double cosang =
                          static_cast<double>(particle1->vx) *
                              static_cast<double>(particle2->vx) +
                          static_cast<double>(particle1->vy) *
                              static_cast<double>(particle2->vy) +
                          static_cast<double>(particle1->vz) *
                              -static_cast<double>(particle2->vz);
                      const double w2 =
                          2.0 * electron_mass * electron_mass +
                          2.0 *
                              (static_cast<double>(particle1->signed_energy) *
                                   static_cast<double>(particle2->signed_energy) -
                               particle1->massive_momentum *
                                   particle2->massive_momentum * cosang);
                      const double W_pair =
                          (w2 > 0.0) ? std::sqrt(w2) : 0.0;
                      results_.add_pair_W(static_cast<double>(weight), W_pair);

                      if (ecm > switches.get_ecm_min())
                        {
                          results_.add_lumi_ee_high(weight);
                          results_.add_step_lumi_ee_high(weight);
                        }
                      const float help = weight * ecm;
                      results_.add_lumi_ecm(help);
                      results_.add_lumi_ecm2(help * ecm);
                    }
                }
            }
    }

  const double fine_dot = externally_supplied_fine_dot
                              ? fine_lumi_dot
                              : static_cast<double>(fine_lumi_sum);
  results_.add_lumi_fine(
      fine_dot * 1e18 /
      (mesh_.get_delta_x() * mesh_.get_delta_y() * timestep_));
}

void GRID::step_lumi_blocked(const SWITCHES& switches,
                             bool externally_supplied_fine_dot,
                             double fine_lumi_dot,
                             float min_z,
                             int time_counter,
                             const feebl::electron_lumi_sampling::PairIdentity*
                                 sampling_identity_prefix)
{
  namespace blocked = feebl::legacy_electron_lumi_blocked;
  namespace sampling = feebl::electron_lumi_sampling;

  // This routine normally preserves the historical exit-on-contract-failure
  // behavior. In a full-wave private encounter it runs on a persistent pool
  // worker, where std::exit would start global teardown and make the pool join
  // its own thread. Propagate through the pool's exception_ptr boundary
  // instead; principal.cc converts the rethrow to the same process status 2.
  const auto fail_closed_worker_lane = [this]() {
    if (wavefront_outer_parallel_lane_)
      throw std::runtime_error(
          "worker-private electron-luminosity contract failed; see the "
          "preceding diagnostic");
    std::exit(2);
  };

  const bool parallel_mode =
      switches.get_electron_lumi_precompute() == 3;
  const int sampling_mode =
      parallel_mode ? switches.get_electron_lumi_sampling() : 0;
  const bool weighted_sampling = sampling_mode == 1;
  const bool bernoulli_sampling = sampling_mode == 2;
  const bool geometric_sampling = sampling_mode == 3;
  const bool counter_sampling =
      weighted_sampling || bernoulli_sampling || geometric_sampling;
  // A full-wave worker already owns a private GRID, RESULTS object, RNG
  // stream, and crossing heap.  Building a second one-lane reservoir here
  // would evaluate every pair, merge/sort/replay that reservoir, and then
  // traverse the block outputs again.  The fused path below offers directly
  // into this worker's crossing heap and publishes each hot block once.
  const bool outer_fused_counter_sampling =
      wavefront_outer_parallel_lane_ && geometric_sampling;
  if (geometric_sampling && !wavefront_outer_parallel_lane_)
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_sampling=3 requires the private "
             "encounter lanes provided by wavefront_parallel_physics=1. "
             "No fallback was performed."
          << std::endl;
      fail_closed_worker_lane();
    }
  const std::chrono::steady_clock::time_point preflight_started =
      std::chrono::steady_clock::now();

  std::size_t maximum_beam1 = 0;
  std::size_t maximum_beam2 = 0;
  const int cells = n_cell_x_ * n_cell_y_;
  const PHI_FLOAT* rho1 = externally_supplied_fine_dot
                              ? 0 : slice_of_beam_[0].get_rho();
  const PHI_FLOAT* rho2 = externally_supplied_fine_dot
                              ? 0 : slice_of_beam_[1].get_rho();
  float fine_lumi_sum = 0.0;

  // This whole section is preflight: every potentially allocating operation
  // completes before the first RESULTS update.  Capacity then survives all
  // cells, steps, and encounters owned by this GRID.
  electron_lumi_occupied_cells_.clear();
  electron_lumi_occupied_cells_.reserve(static_cast<std::size_t>(cells));
  if (!sparse_cells_enabled_)
    {
      for (int cell = 0; cell < cells; ++cell)
        {
          maximum_beam1 = std::max(maximum_beam1, part_pointer1_[cell].size());
          maximum_beam2 = std::max(maximum_beam2, part_pointer2_[cell].size());
          if (!externally_supplied_fine_dot)
            fine_lumi_sum += rho1[cell] * rho2[cell];
          if (!part_pointer1_[cell].empty() && !part_pointer2_[cell].empty())
            electron_lumi_occupied_cells_.push_back(cell);
        }
    }
  else
    {
      feebl::sparse_cells::intersect_sorted(
          particle_cells1_.cells(), particle_cells2_.cells(),
          &electron_lumi_occupied_cells_);
      const std::vector<int>* active_sets[3] = {
          &particle_cells1_.cells(), &particle_cells2_.cells(),
          &sparse_cell_scratch_};
      std::size_t active_set_count = 2U;
      if (!externally_supplied_fine_dot)
        {
          feebl::sparse_cells::intersect_sorted(
              slice_of_beam_[0].activeRhoCells(),
              slice_of_beam_[1].activeRhoCells(), &sparse_cell_scratch_);
          active_set_count = 3U;
        }
      feebl::sparse_cells::sorted_union(active_sets, active_set_count,
                                        &sparse_traversal_cells_);
      sparse_dense_traversal_visits_ +=
          static_cast<unsigned long long>(cells);
      sparse_actual_traversal_visits_ +=
          static_cast<unsigned long long>(sparse_traversal_cells_.size());
      for (std::vector<int>::const_iterator active =
               sparse_traversal_cells_.begin();
           active != sparse_traversal_cells_.end(); ++active)
        {
          const int cell = *active;
          maximum_beam1 = std::max(maximum_beam1, part_pointer1_[cell].size());
          maximum_beam2 = std::max(maximum_beam2, part_pointer2_[cell].size());
          if (!externally_supplied_fine_dot)
            fine_lumi_sum += rho1[cell] * rho2[cell];
        }
    }

  if (electron_lumi_blocked_workspace_ == NULL)
    electron_lumi_blocked_workspace_ = new blocked::Workspace;
  blocked::Workspace& workspace = *electron_lumi_blocked_workspace_;
  workspace.reserve(maximum_beam1, maximum_beam2);
  if (!workspace.beam1.capacities_cover(maximum_beam1) ||
      !workspace.beam2.capacities_cover(maximum_beam2))
    {
      std::cerr
          << "FEEBL: ERROR: electron_lumi_precompute="
          << switches.get_electron_lumi_precompute()
          << " scratch preflight "
             "did not establish the required capacity. No pair results were "
             "updated."
          << std::endl;
      fail_closed_worker_lane();
    }

  std::size_t workers = 1U;
  std::size_t sampling_capacity = 0U;
  if (parallel_mode)
    {
      workers = wavefront_outer_parallel_lane_
          ? 1U
          : std::min(
                maximum_beam1,
                static_cast<std::size_t>(
                    std::max(1, parallel_backend().actual_cpu_threads())));
      if (workers == 0U) workers = 1U;
      if (counter_sampling)
        sampling_capacity = static_cast<std::size_t>(
            std::max(100, switches.get_num_lumi()));
      if (!outer_fused_counter_sampling)
        {
          try
            {
              workspace.reserve_parallel(
                  workers, maximum_beam2, sampling_capacity, sampling_mode);
            }
          catch (const std::exception& error)
            {
              std::cerr
                  << "FEEBL: ERROR: electron_lumi_precompute=3 parallel scratch "
                     "preflight failed: " << error.what()
                  << ". No fallback was performed and no pair results were updated."
                  << std::endl;
              fail_closed_worker_lane();
            }
          if (!workspace.parallel_capacity_covers(
                  workers, maximum_beam2, sampling_capacity, sampling_mode))
            {
              std::cerr
                  << "FEEBL: ERROR: electron_lumi_precompute=3 parallel scratch "
                     "capacity contract failed. No fallback was performed and no "
                     "pair results were updated."
                  << std::endl;
              fail_closed_worker_lane();
            }
        }
      const int expected_heap_sampling = geometric_sampling ? 2 : sampling_mode;
      if (expected_heap_sampling !=
          lumi_heap_ee_.deterministic_sampling_mode())
        {
          std::cerr
              << "FEEBL: ERROR: electron_lumi_precompute=3 sampler policy "
                 "does not match the initialized e-e heap. No fallback was "
                 "performed and no pair results were updated."
              << std::endl;
          fail_closed_worker_lane();
        }

#ifdef FEEBL_HAVE_OPENMP
      if (workers > 1U)
        {
          if (omp_in_parallel())
            {
              std::cerr
                  << "FEEBL: ERROR: electron_lumi_precompute=3 cannot enter "
                     "from an existing OpenMP region. No fallback was performed."
                  << std::endl;
              fail_closed_worker_lane();
            }
          int observed_workers = 1;
#pragma omp parallel num_threads(static_cast<int>(workers)) shared(observed_workers)
          {
#pragma omp single
            observed_workers = omp_get_num_threads();
          }
          if (observed_workers != static_cast<int>(workers))
            {
              std::cerr
                  << "FEEBL: ERROR: electron_lumi_precompute=3 requested "
                  << workers << " evaluation workers but OpenMP formed "
                  << observed_workers
                  << ". No fallback was performed and no pair results were updated."
                  << std::endl;
              fail_closed_worker_lane();
            }
        }
#else
      if (workers > 1U)
        {
          std::cerr
              << "FEEBL: ERROR: electron_lumi_precompute=3 requested parallel "
                 "evaluation in a build without OpenMP. No fallback was performed."
              << std::endl;
          fail_closed_worker_lane();
        }
#endif

      if (counter_sampling)
        {
          const std::uint64_t missing =
              std::numeric_limits<std::uint64_t>::max();
          for (std::vector<int>::const_iterator occupied =
                   electron_lumi_occupied_cells_.begin();
               occupied != electron_lumi_occupied_cells_.end(); ++occupied)
            {
              const std::vector<BEAM_PARTICLE_POINTER>& first =
                  part_pointer1_[*occupied];
              const std::vector<BEAM_PARTICLE_POINTER>& second =
                  part_pointer2_[*occupied];
              for (std::size_t index = 0; index < first.size(); ++index)
                if (first[index].stable_id() == missing)
                  {
                    std::cerr
                        << "FEEBL: ERROR: electron_lumi_sampling="
                        << sampling_mode << " found a "
                           "missing beam-1 stable identity after sidecar "
                           "preflight. No fallback was performed."
                        << std::endl;
                    fail_closed_worker_lane();
                  }
              for (std::size_t index = 0; index < second.size(); ++index)
                if (second[index].stable_id() == missing)
                  {
                    std::cerr
                        << "FEEBL: ERROR: electron_lumi_sampling="
                        << sampling_mode << " found a "
                           "missing beam-2 stable identity after sidecar "
                           "preflight. No fallback was performed."
                        << std::endl;
                    fail_closed_worker_lane();
                  }
            }
        }

      electron_lumi_mode3_seen_ = true;
      electron_lumi_mode3_sampling_mode_ = sampling_mode;
      ++electron_lumi_mode3_calls_;
      electron_lumi_mode3_cells_ += static_cast<unsigned long long>(
          electron_lumi_occupied_cells_.size());
      electron_lumi_mode3_max_workers_ =
          std::max(electron_lumi_mode3_max_workers_, workers);
      electron_lumi_mode3_preflight_ms_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - preflight_started).count();
    }

  const double electron_mass = static_cast<double>(EMASS);
  for (std::vector<int>::const_iterator occupied =
           electron_lumi_occupied_cells_.begin();
       occupied != electron_lumi_occupied_cells_.end(); ++occupied)
    {
      const int cell = *occupied;
      const std::chrono::steady_clock::time_point materialize_started =
          std::chrono::steady_clock::now();
      workspace.beam1.clear();
      workspace.beam2.clear();

      for (std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer =
               part_pointer1_[cell].begin();
           pointer != part_pointer1_[cell].end(); ++pointer)
        {
          const float signed_energy = pointer->energy();
          const float absolute_energy = fabs(signed_energy);
          float vx = 0.0F;
          float vy = 0.0F;
          pointer->velocities(vx, vy);
          const float vz = sqrt(1. - vx * vx - vy * vy);
          const double energy = static_cast<double>(signed_energy);
          const double massive_momentum =
              (energy > electron_mass)
                  ? std::sqrt((energy - electron_mass) *
                              (energy + electron_mass))
                  : 0.0;
          workspace.beam1.append(
              signed_energy, absolute_energy, vx, vy, vz, pointer->weight(),
              massive_momentum,
              static_cast<std::uint8_t>(signed_energy < 0.0 ? 1 : 0),
              pointer->stable_id());
        }

      for (std::vector<BEAM_PARTICLE_POINTER>::const_iterator pointer =
               part_pointer2_[cell].begin();
           pointer != part_pointer2_[cell].end(); ++pointer)
        {
          const float signed_energy = pointer->energy();
          const float absolute_energy = fabs(signed_energy);
          float vx = 0.0F;
          float vy = 0.0F;
          pointer->velocities(vx, vy);
          const float vz = sqrt(1. - vx * vx - vy * vy);
          const double energy = static_cast<double>(signed_energy);
          const double massive_momentum =
              (energy > electron_mass)
                  ? std::sqrt((energy - electron_mass) *
                              (energy + electron_mass))
                  : 0.0;
          workspace.beam2.append(
              signed_energy, absolute_energy, vx, vy, vz, pointer->weight(),
              massive_momentum,
              static_cast<std::uint8_t>(signed_energy < 0.0 ? 0 : 1),
              pointer->stable_id());
        }

      if (parallel_mode)
        electron_lumi_mode3_materialize_ms_ +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - materialize_started).count();

      // The pure kinematics are evaluated in fixed beam-1 waves.  Every lane
      // owns disjoint BlockKinematics and (for counter sampling) a bounded
      // reservoir.  RESULTS and the legacy heap remain a canonical serial
      // publication boundary, preserving exact legacy arithmetic/RNG order.
      const auto publish_block =
          [&](std::size_t particle1, std::size_t beam2_begin,
              std::size_t count, const blocked::BlockKinematics& output,
              int heap_sampling_mode) {
            const int cellx = cell / n_cell_y_;
            const int celly = cell - cellx * n_cell_y_;
            for (std::size_t local = 0; local < count; ++local)
              {
                const std::size_t particle2 = beam2_begin + local;
                const float weight = output.weight[local];
                const float ecm = output.ecm[local];
                if (heap_sampling_mode < 0)
                  {
                    lumi_heap_ee_.lumi_store_ee(
                        mesh_, cellx, celly, min_z,
                        workspace.beam1.signed_energy[particle1],
                        workspace.beam1.vx[particle1],
                        workspace.beam1.vy[particle1],
                        workspace.beam2.signed_energy[particle2],
                        workspace.beam2.vx[particle2],
                        workspace.beam2.vy[particle2], weight, time_counter);
                    ++electron_lumi_mode3_legacy_heap_candidates_;
                  }
                else if (heap_sampling_mode > 0 && weight != 0.0F)
                  {
                    sampling::PairIdentity identity =
                        sampling_identity_prefix
                            ? *sampling_identity_prefix
                            : sampling::PairIdentity();
                    identity.run_seed = static_cast<std::uint64_t>(
                        switches.get_rndm_seed());
                    if (sampling_identity_prefix)
                      identity.interaction =
                          static_cast<std::uint64_t>(cell);
                    else
                      {
                        identity.substep = static_cast<std::uint64_t>(
                            time_counter);
                        identity.encounter =
                            static_cast<std::uint64_t>(cell);
                      }
                    identity.parent_a =
                        workspace.beam1.stable_id[particle1];
                    identity.parent_b =
                        workspace.beam2.stable_id[particle2];

                    const int records_before = lumi_heap_ee_.nb_pairs();
                    const sampling::OfferStatus stored =
                        heap_sampling_mode == 1
                            ? lumi_heap_ee_.lumi_store_ee_keyed(
                                  mesh_, cellx, celly, min_z,
                                  workspace.beam1.signed_energy[particle1],
                                  workspace.beam1.vx[particle1],
                                  workspace.beam1.vy[particle1],
                                  workspace.beam2.signed_energy[particle2],
                                  workspace.beam2.vx[particle2],
                                  workspace.beam2.vy[particle2], weight,
                                  time_counter, identity)
                            : lumi_heap_ee_.lumi_store_ee_keyed_bernoulli(
                                  mesh_, cellx, celly, min_z,
                                  workspace.beam1.signed_energy[particle1],
                                  workspace.beam1.vx[particle1],
                                  workspace.beam1.vy[particle1],
                                  workspace.beam2.signed_energy[particle2],
                                  workspace.beam2.vx[particle2],
                                  workspace.beam2.vy[particle2], weight,
                                  time_counter, identity);
                    const int records_after = lumi_heap_ee_.nb_pairs();
                    const bool accepted =
                        stored == sampling::OfferStatus::Inserted ||
                        stored == sampling::OfferStatus::Replaced;
                    if (!accepted && stored != sampling::OfferStatus::Rejected)
                      {
                        const std::string error =
                            std::string("fused outer electron-luminosity sampling=") +
                            std::to_string(heap_sampling_mode) + " failed: " +
                            sampling::offer_status_name(stored) + " " +
                            lumi_heap_ee_.deterministic_sampling_error() +
                            ". Increase num_lumi for mode 2 if its hard "
                            "unbiased capacity was exceeded. No fallback was "
                            "performed.";
                        if (geometric_sampling)
                          throw std::runtime_error(error);
                        std::cerr
                            << "FEEBL: ERROR: " << error
                            << std::endl;
                        fail_closed_worker_lane();
                      }
                    if (heap_sampling_mode == 1)
                      {
                        if (accepted)
                          ++electron_lumi_mode3_counter_heap_accepts_;
                      }
                    else
                      {
                        ++electron_lumi_mode3_bernoulli_draws_;
                        if (stored == sampling::OfferStatus::Rejected)
                          ++electron_lumi_mode3_bernoulli_rejects_;
                        else
                          {
                            ++electron_lumi_mode3_bernoulli_accepted_pairs_;
                            if (records_after < records_before)
                              {
                                if (geometric_sampling)
                                  throw std::runtime_error(
                                      "fused outer electron-luminosity "
                                      "Bernoulli sampler shrank its append-only "
                                      "output. No fallback was performed.");
                                std::cerr
                                    << "FEEBL: ERROR: fused outer electron-"
                                       "luminosity Bernoulli sampler shrank "
                                       "its append-only output. No fallback "
                                       "was performed."
                                    << std::endl;
                                fail_closed_worker_lane();
                              }
                            const unsigned long long added =
                                static_cast<unsigned long long>(
                                    records_after - records_before);
                            electron_lumi_mode3_bernoulli_accepted_records_ +=
                                added;
                            electron_lumi_mode3_counter_heap_accepts_ += added;
                          }
                      }
                  }
                results_.add_lumi(
                    workspace.beam1.charge_class[particle1],
                    workspace.beam2.charge_class[particle2], weight);
                if (output.same_sign[local] != 0U)
                  {
                    results_.add_lumi_ee(weight);
                    results_.add_step_lumi_ee(weight);
                    results_.add_pair_W(
                        static_cast<double>(weight),
                        output.invariant_mass[local]);
                    if (design_axis_spectrum_.enabled())
                      design_axis_spectrum_.add(
                          static_cast<double>(weight),
                          std::fabs(static_cast<double>(workspace.beam1.signed_energy[particle1])),
                          workspace.beam1.vx[particle1], workspace.beam1.vy[particle1],
                          std::fabs(static_cast<double>(workspace.beam2.signed_energy[particle2])),
                          workspace.beam2.vx[particle2], workspace.beam2.vy[particle2]);
                    if (output.above_threshold[local] != 0U)
                      {
                        results_.add_lumi_ee_high(weight);
                        results_.add_step_lumi_ee_high(weight);
                      }
                    const float help = weight * ecm;
                    results_.add_lumi_ecm(help);
                    results_.add_lumi_ecm2(help * ecm);
                  }
                if (parallel_mode)
                  ++electron_lumi_mode3_serial_result_commits_;
              }
          };

      if (!parallel_mode)
        {
          for (std::size_t particle1 = 0;
               particle1 < workspace.beam1.size(); ++particle1)
            for (std::size_t beam2_begin = 0;
                 beam2_begin < workspace.beam2.size();
                 beam2_begin += blocked::kBlockSize)
              {
                const std::size_t count = std::min(
                    blocked::kBlockSize,
                    workspace.beam2.size() - beam2_begin);
                if (!blocked::evaluate_block(
                        workspace.beam1, particle1, workspace.beam2,
                        beam2_begin, count, switches.get_ecm_min(),
                        electron_mass, workspace.block))
                  {
                    std::cerr
                        << "FEEBL: ERROR: electron_lumi_precompute=2 internal "
                           "block contract failed. No fallback was performed."
                        << std::endl;
                    fail_closed_worker_lane();
                  }
                publish_block(
                    particle1, beam2_begin, count, workspace.block, 0);
              }
          continue;
        }

      const std::size_t block_count =
          (workspace.beam2.size() + blocked::kBlockSize - 1U) /
          blocked::kBlockSize;

      if (outer_fused_counter_sampling)
        {
          const std::chrono::steady_clock::time_point fused_started =
              std::chrono::steady_clock::now();
          unsigned long long fused_pairs = 0U;

          // V1 assigns geometric opportunity ordinals in ascending stable-ID
          // order.  The normal scatter path already has that order; if a
          // future weighted/reordered beam violates either row assumption,
          // retain exact physics through the mode-2 keyed Bernoulli kernel
          // and expose the fallback in telemetry.
          bool geometric_cell_contract = workspace.beam2.size() != 0U;
          const float beam2_weight = geometric_cell_contract
              ? workspace.beam2.weight[0] : 0.0F;
          for (std::size_t particle2 = 1U;
               geometric_cell_contract &&
                   particle2 < workspace.beam2.size();
               ++particle2)
            if (workspace.beam2.weight[particle2] != beam2_weight ||
                workspace.beam2.stable_id[particle2 - 1U] >=
                    workspace.beam2.stable_id[particle2])
              geometric_cell_contract = false;

          const auto store_geometric_record =
              [&](std::size_t particle1, std::size_t particle2,
                  std::size_t multiplicity) {
                if (multiplicity == 0U) return;
                sampling::PairIdentity identity = sampling_identity_prefix
                    ? *sampling_identity_prefix : sampling::PairIdentity();
                identity.run_seed = static_cast<std::uint64_t>(
                    switches.get_rndm_seed());
                if (sampling_identity_prefix)
                  identity.interaction = static_cast<std::uint64_t>(cell);
                else
                  {
                    identity.substep = static_cast<std::uint64_t>(
                        time_counter);
                    identity.encounter = static_cast<std::uint64_t>(cell);
                  }
                identity.parent_a = workspace.beam1.stable_id[particle1];
                identity.parent_b = workspace.beam2.stable_id[particle2];

                const int records_before = lumi_heap_ee_.nb_pairs();
                const sampling::OfferStatus stored =
                    lumi_heap_ee_.lumi_store_ee_keyed_bernoulli_multiplicity(
                        mesh_, cell / n_cell_y_,
                        cell - (cell / n_cell_y_) * n_cell_y_, min_z,
                        workspace.beam1.signed_energy[particle1],
                        workspace.beam1.vx[particle1],
                        workspace.beam1.vy[particle1],
                        workspace.beam2.signed_energy[particle2],
                        workspace.beam2.vx[particle2],
                        workspace.beam2.vy[particle2], time_counter,
                        identity, multiplicity);
                const int records_after = lumi_heap_ee_.nb_pairs();
                if (stored != sampling::OfferStatus::Inserted ||
                    records_after < records_before ||
                    static_cast<std::size_t>(records_after - records_before) !=
                        multiplicity)
                  {
                    throw std::runtime_error(
                        std::string("electron_lumi_sampling=3 append-only heap ") +
                        "commit failed: " + sampling::offer_status_name(stored) +
                        " " + lumi_heap_ee_.deterministic_sampling_error() +
                        ". Increase num_lumi if its hard unbiased capacity was "
                        "exceeded. No fallback was performed.");
                  }
              };

          for (std::size_t particle1 = 0;
               particle1 < workspace.beam1.size(); ++particle1)
            {
              if (!geometric_cell_contract)
                {
                  ++electron_lumi_mode3_geometric_fallback_rows_;
                  electron_lumi_mode3_geometric_fallback_pairs_ +=
                      static_cast<unsigned long long>(workspace.beam2.size());
                  for (std::size_t block_index = 0;
                       block_index < block_count; ++block_index)
                    {
                      const std::size_t beam2_begin =
                          block_index * blocked::kBlockSize;
                      const std::size_t count = std::min(
                          blocked::kBlockSize,
                          workspace.beam2.size() - beam2_begin);
                      if (!blocked::evaluate_block(
                              workspace.beam1, particle1, workspace.beam2,
                              beam2_begin, count, switches.get_ecm_min(),
                              electron_mass, workspace.block))
                        {
                          throw std::runtime_error(
                              "geometric-fallback electron-luminosity block "
                              "contract failed. No fallback was performed.");
                        }
                      publish_block(particle1, beam2_begin, count,
                                    workspace.block, 2);
                      fused_pairs += static_cast<unsigned long long>(count);
                    }
                  continue;
                }

              sampling::ElectronLumiBernoulliRowPrefixV1 prefix;
              prefix.run_seed = static_cast<std::uint64_t>(
                  switches.get_rndm_seed());
              if (sampling_identity_prefix)
                {
                  prefix.crossing = sampling_identity_prefix->crossing;
                  prefix.diagonal = sampling_identity_prefix->diagonal;
                  prefix.wave = sampling_identity_prefix->wave;
                  prefix.substep = sampling_identity_prefix->substep;
                  prefix.encounter = sampling_identity_prefix->encounter;
                }
              else
                {
                  prefix.substep = static_cast<std::uint64_t>(time_counter);
                  prefix.encounter = static_cast<std::uint64_t>(cell);
                }
              prefix.beam = 1U;
              prefix.row_parent = workspace.beam1.stable_id[particle1];
              prefix.row_ordinal = static_cast<std::uint64_t>(cell);

              const float pair_weight =
                  workspace.beam1.weight[particle1] * beam2_weight;
              // Preserve the legacy binary32 multiplication by lumi_p before
              // promoting to the cursor's audited binary64 inverse transform.
              const double expected_store = static_cast<double>(
                  pair_weight * switches.get_lumi_p());
              if (lumi_heap_ee_.nb_pairs() < 0 ||
                  static_cast<std::size_t>(lumi_heap_ee_.nb_pairs()) >
                      sampling_capacity)
                {
                  throw std::runtime_error(
                      "electron_lumi_sampling=3 heap capacity accounting is "
                      "invalid. No fallback was performed.");
                }
              const std::size_t remaining_capacity =
                  sampling_capacity -
                  static_cast<std::size_t>(lumi_heap_ee_.nb_pairs());
              sampling::ElectronLumiBernoulliSkipLimitsV1 limits;
              limits.max_bernoulli_successes =
                  static_cast<std::uint64_t>(remaining_capacity);
              limits.max_total_copies =
                  static_cast<std::uint64_t>(remaining_capacity);
              sampling::ElectronLumiBernoulliSkipCursorV1 cursor;
              if (!cursor.initialize(
                      prefix, expected_store,
                      static_cast<std::uint64_t>(workspace.beam2.size()),
                      limits))
                {
                  throw std::runtime_error(
                      std::string("electron_lumi_sampling=3 cursor preflight ") +
                      "failed: " +
                      sampling::electron_lumi_bernoulli_skip_error_name_v1(
                          cursor.error()) +
                      ". Increase num_lumi if its hard unbiased capacity was "
                      "exceeded. No fallback was performed.");
                }

              workspace.geometric_success_ordinals.clear();
              for (;;)
                {
                  std::uint64_t opportunity = 0U;
                  const sampling::ElectronLumiBernoulliSkipResultV1 next =
                      cursor.next_success(&opportunity);
                  if (next ==
                      sampling::ElectronLumiBernoulliSkipResultV1::Success)
                    {
                      workspace.geometric_success_ordinals.push_back(
                          opportunity);
                      continue;
                    }
                  if (next ==
                      sampling::ElectronLumiBernoulliSkipResultV1::End)
                    break;
                  throw std::runtime_error(
                      std::string("electron_lumi_sampling=3 cursor enumeration ") +
                      "failed: " +
                      sampling::electron_lumi_bernoulli_skip_error_name_v1(
                          cursor.error()) +
                      ". Increase num_lumi if its hard unbiased capacity was "
                      "exceeded. No fallback was performed.");
                }

              const sampling::ElectronLumiBernoulliSkipTelemetryV1&
                  cursor_stats = cursor.telemetry();
              ++electron_lumi_mode3_geometric_rows_;
              electron_lumi_mode3_geometric_rng_calls_ +=
                  cursor_stats.rng_calls;
              electron_lumi_mode3_geometric_successes_ +=
                  cursor_stats.bernoulli_successes;
              electron_lumi_mode3_geometric_integer_copies_ +=
                  cursor_stats.integer_copies_total;
              electron_lumi_mode3_bernoulli_draws_ +=
                  cursor_stats.row_opportunities;
              const unsigned long long accepted_pairs =
                  cursor_stats.integer_copies_per_opportunity == 0U
                      ? cursor_stats.bernoulli_successes
                      : cursor_stats.row_opportunities;
              electron_lumi_mode3_bernoulli_rejects_ +=
                  cursor_stats.row_opportunities - accepted_pairs;
              electron_lumi_mode3_bernoulli_accepted_pairs_ +=
                  accepted_pairs;
              electron_lumi_mode3_bernoulli_accepted_records_ +=
                  cursor_stats.selected_copies_total;
              electron_lumi_mode3_counter_heap_accepts_ +=
                  cursor_stats.selected_copies_total;

              const std::size_t base_copies = static_cast<std::size_t>(
                  cursor.integer_copies_per_opportunity());
              std::size_t success_cursor = 0U;
              for (std::size_t block_index = 0;
                   block_index < block_count; ++block_index)
              {
                const std::size_t beam2_begin =
                    block_index * blocked::kBlockSize;
                const std::size_t count = std::min(
                    blocked::kBlockSize,
                    workspace.beam2.size() - beam2_begin);
                if (!blocked::evaluate_block(
                        workspace.beam1, particle1, workspace.beam2,
                        beam2_begin, count, switches.get_ecm_min(),
                        electron_mass, workspace.block))
                  {
                    throw std::runtime_error(
                        "fused outer electron-luminosity block contract "
                        "failed. No fallback was performed.");
                  }
                // Aggregates keep the historical beam1-major/beam2-minor
                // arithmetic order.  Sampling publication touches only the
                // preselected O(K) opportunities (or all opportunities when
                // floor(expected_store) is nonzero and output size itself is
                // necessarily O(N)).
                publish_block(particle1, beam2_begin, count,
                              workspace.block, 0);
                const std::size_t block_end = beam2_begin + count;
                if (base_copies == 0U)
                  {
                    while (success_cursor <
                               workspace.geometric_success_ordinals.size() &&
                           workspace.geometric_success_ordinals[
                               success_cursor] < block_end)
                      {
                        const std::size_t particle2 =
                            static_cast<std::size_t>(
                                workspace.geometric_success_ordinals[
                                    success_cursor]);
                        store_geometric_record(particle1, particle2, 1U);
                        ++success_cursor;
                      }
                  }
                else
                  {
                    for (std::size_t particle2 = beam2_begin;
                         particle2 < block_end; ++particle2)
                      {
                        std::size_t multiplicity = base_copies;
                        if (success_cursor <
                                workspace.geometric_success_ordinals.size() &&
                            workspace.geometric_success_ordinals[
                                success_cursor] == particle2)
                          {
                            ++multiplicity;
                            ++success_cursor;
                          }
                        store_geometric_record(
                            particle1, particle2, multiplicity);
                      }
                  }
                fused_pairs += static_cast<unsigned long long>(count);
              }
              if (success_cursor !=
                  workspace.geometric_success_ordinals.size())
                {
                  throw std::runtime_error(
                      "electron_lumi_sampling=3 selected opportunity "
                      "publication was incomplete. No fallback was performed.");
                }
            }
          electron_lumi_mode3_pairs_ += fused_pairs;
          electron_lumi_mode3_parallel_pairs_ += fused_pairs;
          electron_lumi_mode3_parallel_sample_candidates_ += fused_pairs;
          electron_lumi_mode3_outer_fused_pairs_ += fused_pairs;
          electron_lumi_mode3_outer_fused_ms_ +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - fused_started).count();
          continue;
        }

      for (std::size_t wave_begin = 0; wave_begin < workspace.beam1.size();
           wave_begin += workers)
        {
          const std::size_t wave_count =
              std::min(workers, workspace.beam1.size() - wave_begin);
          int failed = 0;
          if (counter_sampling)
            for (std::size_t lane = 0; lane < workers; ++lane)
              {
                if (weighted_sampling)
                  workspace.parallel_samplers[lane].reset(
                      std::min(sampling_capacity, workspace.beam2.size()));
                else
                  workspace.parallel_bernoulli_samplers[lane].reset(
                      sampling_capacity,
                      std::min(sampling_capacity,
                               workspace.beam2.size()));
              }

          const std::chrono::steady_clock::time_point evaluate_started =
              std::chrono::steady_clock::now();
#ifdef FEEBL_HAVE_OPENMP
#pragma omp parallel for schedule(static, 1) \
    num_threads(static_cast<int>(wave_count)) if(wave_count > 1U) \
    reduction(|:failed)
#endif
          for (long long lane_value = 0;
               lane_value < static_cast<long long>(wave_count); ++lane_value)
            {
              const std::size_t lane = static_cast<std::size_t>(lane_value);
              const std::size_t particle1 = wave_begin + lane;
              for (std::size_t block_index = 0; block_index < block_count;
                   ++block_index)
                {
                  const std::size_t beam2_begin =
                      block_index * blocked::kBlockSize;
                  const std::size_t count = std::min(
                      blocked::kBlockSize,
                      workspace.beam2.size() - beam2_begin);
                  blocked::BlockKinematics& output =
                      workspace.parallel_block(lane, block_index);
                  if (!blocked::evaluate_block(
                          workspace.beam1, particle1, workspace.beam2,
                          beam2_begin, count, switches.get_ecm_min(),
                          electron_mass, output))
                    {
                      failed |= 1;
                      continue;
                    }
                  if (counter_sampling)
                    for (std::size_t local = 0; local < count; ++local)
                      {
                        // A zero pair weight contributes nothing to any
                        // luminosity aggregate and has zero A-Res selection
                        // probability.  Preserve fail-closed sampler
                        // semantics for direct callers while skipping this
                        // legitimate collision no-op at the physics adapter.
                        if (output.weight[local] == 0.0F) continue;
                        const std::size_t particle2 = beam2_begin + local;
                        sampling::PairIdentity identity =
                            sampling_identity_prefix
                                ? *sampling_identity_prefix
                                : sampling::PairIdentity();
                        identity.run_seed = static_cast<std::uint64_t>(
                            switches.get_rndm_seed());
                        if (sampling_identity_prefix)
                          identity.interaction =
                              static_cast<std::uint64_t>(cell);
                        else
                          {
                            // Preserve the standalone mode-3 identity
                            // contract outside the physical EncounterDAG.
                            identity.substep = static_cast<std::uint64_t>(
                                time_counter);
                            identity.encounter =
                                static_cast<std::uint64_t>(cell);
                          }
                        identity.parent_a =
                            workspace.beam1.stable_id[particle1];
                        identity.parent_b =
                            workspace.beam2.stable_id[particle2];
                        const std::uint64_t payload =
                            static_cast<std::uint64_t>(lane) *
                                static_cast<std::uint64_t>(
                                    workspace.beam2.size()) +
                            static_cast<std::uint64_t>(particle2);
                        const sampling::OfferResult offered = weighted_sampling
                            ? workspace.parallel_samplers[lane].offer(
                                  identity, output.weight[local], payload)
                            : workspace.parallel_bernoulli_samplers[lane].offer(
                                  identity,
                                  static_cast<double>(
                                      output.weight[local] *
                                      switches.get_lumi_p()),
                                  payload);
                        if (offered.status != sampling::OfferStatus::Inserted &&
                            offered.status != sampling::OfferStatus::Replaced &&
                            offered.status != sampling::OfferStatus::Rejected)
                          {
                            failed |= 1 <<
                                (1 + static_cast<int>(offered.status));
                          }
                      }
                }
            }
          electron_lumi_mode3_evaluate_ms_ +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - evaluate_started).count();

          const unsigned long long wave_pairs =
              static_cast<unsigned long long>(wave_count) *
              static_cast<unsigned long long>(workspace.beam2.size());
          electron_lumi_mode3_pairs_ += wave_pairs;
          electron_lumi_mode3_parallel_pairs_ += wave_pairs;
          if (counter_sampling)
            electron_lumi_mode3_parallel_sample_candidates_ += wave_pairs;
          if (bernoulli_sampling)
            for (std::size_t lane = 0; lane < wave_count; ++lane)
              {
                electron_lumi_mode3_bernoulli_draws_ +=
                    workspace.parallel_bernoulli_samplers[lane]
                        .opportunities();
                electron_lumi_mode3_bernoulli_rejects_ +=
                    workspace.parallel_bernoulli_samplers[lane].rejected();
                electron_lumi_mode3_bernoulli_accepted_pairs_ +=
                    workspace.parallel_bernoulli_samplers[lane]
                        .accepted_pairs();
                electron_lumi_mode3_bernoulli_accepted_records_ +=
                    workspace.parallel_bernoulli_samplers[lane]
                        .accepted_copies();
              }
          if (failed != 0)
            {
              std::cerr
                  << "FEEBL: ERROR: electron_lumi_precompute=3 parallel "
                     "evaluation/sampling contract failed (failure_mask="
                  << failed
                  << "). No fallback was performed."
                  << std::endl;
              fail_closed_worker_lane();
            }

          const std::chrono::steady_clock::time_point merge_started =
              std::chrono::steady_clock::now();
          if (weighted_sampling)
            {
              std::string reason;
              const std::size_t wave_candidates =
                  wave_count * workspace.beam2.size();
              if (!sampling::WeightedReservoir::merge_lanes(
                      workspace.parallel_samplers,
                      std::min(sampling_capacity, wave_candidates),
                      &workspace.merged_sampler, &reason))
                {
                  std::cerr
                      << "FEEBL: ERROR: electron_lumi_sampling=1 deterministic "
                         "lane merge failed: " << reason
                      << ". No fallback was performed." << std::endl;
                  fail_closed_worker_lane();
                }
              const std::vector<sampling::Selection> selected =
                  workspace.merged_sampler.canonical_selection();
              electron_lumi_mode3_serial_sample_merge_candidates_ +=
                  static_cast<unsigned long long>(selected.size());
              const int cellx = cell / n_cell_y_;
              const int celly = cell - cellx * n_cell_y_;
              for (std::vector<sampling::Selection>::const_iterator candidate =
                       selected.begin(); candidate != selected.end(); ++candidate)
                {
                  const std::size_t lane = static_cast<std::size_t>(
                      candidate->payload_token / workspace.beam2.size());
                  const std::size_t particle2 = static_cast<std::size_t>(
                      candidate->payload_token % workspace.beam2.size());
                  if (lane >= wave_count || particle2 >= workspace.beam2.size())
                    {
                      std::cerr
                          << "FEEBL: ERROR: electron_lumi_sampling=1 lane "
                             "payload contract failed. No fallback was performed."
                          << std::endl;
                      fail_closed_worker_lane();
                    }
                  const std::size_t particle1 = wave_begin + lane;
                  const sampling::OfferStatus stored =
                      lumi_heap_ee_.lumi_store_ee_keyed_priority(
                          mesh_, cellx, celly, min_z,
                          workspace.beam1.signed_energy[particle1],
                          workspace.beam1.vx[particle1],
                          workspace.beam1.vy[particle1],
                          workspace.beam2.signed_energy[particle2],
                          workspace.beam2.vx[particle2],
                          workspace.beam2.vy[particle2], time_counter,
                          candidate->identity, candidate->priority);
                  if (stored == sampling::OfferStatus::Inserted ||
                      stored == sampling::OfferStatus::Replaced)
                    ++electron_lumi_mode3_counter_heap_accepts_;
                  else if (stored != sampling::OfferStatus::Rejected)
                    {
                      std::cerr
                          << "FEEBL: ERROR: electron_lumi_sampling=1 global "
                             "reservoir commit failed: "
                          << sampling::offer_status_name(stored) << " "
                          << lumi_heap_ee_.deterministic_sampling_error()
                          << ". No fallback was performed." << std::endl;
                      fail_closed_worker_lane();
                    }
                }
            }
          else if (bernoulli_sampling)
            {
              std::string reason;
              if (!sampling::KeyedBernoulliSampler::merge_lanes(
                      workspace.parallel_bernoulli_samplers,
                      sampling_capacity,
                      &workspace.merged_bernoulli_sampler, &reason))
                {
                  std::cerr
                      << "FEEBL: ERROR: electron_lumi_sampling=2 keyed "
                         "Bernoulli lane merge failed: " << reason
                      << ". Increase num_lumi if the hard unbiased sample "
                         "capacity was exceeded. No fallback was performed."
                      << std::endl;
                  fail_closed_worker_lane();
                }
              const std::vector<sampling::BernoulliSelection> selected =
                  workspace.merged_bernoulli_sampler.canonical_selection();
              electron_lumi_mode3_serial_sample_merge_candidates_ +=
                  static_cast<unsigned long long>(selected.size());
              const int cellx = cell / n_cell_y_;
              const int celly = cell - cellx * n_cell_y_;
              for (std::vector<sampling::BernoulliSelection>::const_iterator
                       candidate = selected.begin();
                   candidate != selected.end(); ++candidate)
                {
                  const std::size_t lane = static_cast<std::size_t>(
                      candidate->payload_token / workspace.beam2.size());
                  const std::size_t particle2 = static_cast<std::size_t>(
                      candidate->payload_token % workspace.beam2.size());
                  if (lane >= wave_count ||
                      particle2 >= workspace.beam2.size() ||
                      candidate->multiplicity == 0U)
                    {
                      std::cerr
                          << "FEEBL: ERROR: electron_lumi_sampling=2 lane "
                             "payload contract failed. No fallback was "
                             "performed."
                          << std::endl;
                      fail_closed_worker_lane();
                    }
                  const std::size_t particle1 = wave_begin + lane;
                  const sampling::OfferStatus stored =
                      lumi_heap_ee_
                          .lumi_store_ee_keyed_bernoulli_multiplicity(
                              mesh_, cellx, celly, min_z,
                              workspace.beam1.signed_energy[particle1],
                              workspace.beam1.vx[particle1],
                              workspace.beam1.vy[particle1],
                              workspace.beam2.signed_energy[particle2],
                              workspace.beam2.vx[particle2],
                              workspace.beam2.vy[particle2], time_counter,
                              candidate->identity,
                              candidate->multiplicity);
                  if (stored == sampling::OfferStatus::Inserted)
                    electron_lumi_mode3_counter_heap_accepts_ +=
                        static_cast<unsigned long long>(
                            candidate->multiplicity);
                  else
                    {
                      std::cerr
                          << "FEEBL: ERROR: electron_lumi_sampling=2 global "
                             "Bernoulli commit failed: "
                          << sampling::offer_status_name(stored) << " "
                          << lumi_heap_ee_.deterministic_sampling_error()
                          << " requested_multiplicity="
                          << candidate->multiplicity
                          << " accepted_records_before_failure="
                          << electron_lumi_mode3_counter_heap_accepts_
                          << " num_lumi=" << switches.get_num_lumi()
                          << ". Increase num_lumi if the hard unbiased sample "
                             "capacity was exceeded. No fallback was "
                             "performed."
                          << std::endl;
                      fail_closed_worker_lane();
                    }
                }
            }
          electron_lumi_mode3_sample_merge_ms_ +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - merge_started).count();

          const std::chrono::steady_clock::time_point publish_started =
              std::chrono::steady_clock::now();
          for (std::size_t lane = 0; lane < wave_count; ++lane)
            {
              const std::size_t particle1 = wave_begin + lane;
              for (std::size_t block_index = 0; block_index < block_count;
                   ++block_index)
                {
                  const std::size_t beam2_begin =
                      block_index * blocked::kBlockSize;
                  const std::size_t count = std::min(
                      blocked::kBlockSize,
                      workspace.beam2.size() - beam2_begin);
                  publish_block(
                      particle1, beam2_begin, count,
                      workspace.parallel_block(lane, block_index),
                      counter_sampling ? 0 : -1);
                }
            }
          electron_lumi_mode3_publish_ms_ +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - publish_started).count();
        }
    }

  const double fine_dot = externally_supplied_fine_dot
                              ? fine_lumi_dot
                              : static_cast<double>(fine_lumi_sum);
  results_.add_lumi_fine(
      fine_dot * 1e18 /
      (mesh_.get_delta_x() * mesh_.get_delta_y() * timestep_));
}

void GRID::step_lumi_spatial(const SWITCHES& switches,
                             int beamslice1,
                             int beamslice2)
{
  using feebl::electron_lumi_spatial::AdmissionAudit;
  using feebl::electron_lumi_spatial::CollisionPair;
  using feebl::electron_lumi_spatial::CollisionPlan;
  using feebl::electron_lumi_spatial::CollisionPlanOptions;
  using feebl::electron_lumi_spatial::PhysicalCicGeometry;
  using feebl::electron_lumi_spatial::PhysicalCicGeometryInput;
  using feebl::electron_lumi_spatial::PhysicalRoot;
  using feebl::electron_lumi_spatial::RootId;

  if (beamslice1 < 0 || beamslice2 < 0 || timestep_ <= 0)
    throw std::logic_error(
        "spatial electron luminosity requires valid current slice indices "
        "and a positive timestep");
  const BEAM* beam1 = slice_of_beam_[0].get_beam();
  const BEAM* beam2 = slice_of_beam_[1].get_beam();
  if (beam1 == NULL || beam2 == NULL)
    throw std::logic_error(
        "spatial electron luminosity entered before beam connection");

  if (!electron_lumi_spatial_ids_validated_)
    {
      beam1->particle_beam().require_valid_particle_id_sidecar();
      beam2->particle_beam().require_valid_particle_id_sidecar();
      electron_lumi_spatial_ids_validated_ = true;
    }
  const std::vector<PARTICLE*>& particles1 =
      beam1->getParticleVector(beamslice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2->getParticleVector(beamslice2);
  const std::vector<PRIMARY_PARTICLE_ID>& ids1 =
      beam1->particle_beam().getParticleIdVectorShapeChecked(beamslice1);
  const std::vector<PRIMARY_PARTICLE_ID>& ids2 =
      beam2->particle_beam().getParticleIdVectorShapeChecked(beamslice2);
  if (particles1.size() != ids1.size() || particles2.size() != ids2.size())
    throw std::logic_error(
        "spatial electron luminosity particle/ID shape changed after preflight");

  PhysicalCicGeometryInput geometry_input;
  geometry_input.n_x = static_cast<std::size_t>(
      switches.get_electron_lumi_spatial_nx());
  geometry_input.n_y = static_cast<std::size_t>(
      switches.get_electron_lumi_spatial_ny());
  geometry_input.x_min = switches.get_electron_lumi_spatial_x_min();
  geometry_input.x_max = switches.get_electron_lumi_spatial_x_max();
  geometry_input.y_min = switches.get_electron_lumi_spatial_y_min();
  geometry_input.y_max = switches.get_electron_lumi_spatial_y_max();
  const PhysicalCicGeometry geometry(geometry_input);

  const double population1 = get_macro_charge(1);
  const double population2 = get_macro_charge(2);
  std::vector<PhysicalRoot> roots1;
  std::vector<PhysicalRoot> roots2;
  roots1.reserve(particles1.size());
  roots2.reserve(particles2.size());
  std::map<RootId, const PARTICLE*> lookup1;
  std::map<RootId, const PARTICLE*> lookup2;

  const auto materialize = [](
      const std::vector<PARTICLE*>& particles,
      const std::vector<PRIMARY_PARTICLE_ID>& ids,
      double population,
      std::vector<PhysicalRoot>& roots,
      std::map<RootId, const PARTICLE*>& lookup) {
    for (std::size_t index = 0U; index < particles.size(); ++index)
      {
        if (particles[index] == NULL)
          throw std::logic_error(
              "spatial electron luminosity encountered a null primary");
        float x = 0.0f;
        float y = 0.0f;
        particles[index]->XYposition(x, y);
        PhysicalRoot root;
        root.id = ids[index];
        root.x = static_cast<double>(x);
        root.y = static_cast<double>(y);
        root.population = population;
        roots.push_back(root);
        if (!lookup.insert(std::make_pair(root.id, particles[index])).second)
          throw std::logic_error(
              "spatial electron luminosity encountered a duplicate root ID");
      }
  };
  materialize(particles1, ids1, population1, roots1, lookup1);
  materialize(particles2, ids2, population2, roots2, lookup2);

  // Audit before strict construction so an out-of-domain failure reports the
  // complete whole-root accounting rather than only an exception category.
  const AdmissionAudit audit1 =
      feebl::electron_lumi_spatial::audit_admission(geometry, roots1);
  const AdmissionAudit audit2 =
      feebl::electron_lumi_spatial::audit_admission(geometry, roots2);
  if (!audit1.rejected.empty() || !audit2.rejected.empty())
    {
      std::cerr
          << "FEEBL_ELECTRON_LUMI_SPATIAL_FAILURE_V1"
          << " reason=out-of-domain-root"
          << " geometry_hash=" << geometry.geometry_hash()
          << " beamslice1=" << beamslice1
          << " beamslice2=" << beamslice2
          << " beam1_input_roots=" << audit1.accounting.input_roots
          << " beam1_accepted_roots=" << audit1.accounting.accepted_roots
          << " beam1_rejected_roots=" << audit1.accounting.rejected_roots
          << " beam1_input_population="
          << audit1.accounting.input_population
          << " beam1_accepted_population="
          << audit1.accounting.accepted_population
          << " beam1_rejected_population="
          << audit1.accounting.rejected_population
          << " beam2_input_roots=" << audit2.accounting.input_roots
          << " beam2_accepted_roots=" << audit2.accounting.accepted_roots
          << " beam2_rejected_roots=" << audit2.accounting.rejected_roots
          << " beam2_input_population="
          << audit2.accounting.input_population
          << " beam2_accepted_population="
          << audit2.accounting.accepted_population
          << " beam2_rejected_population="
          << audit2.accounting.rejected_population
          << " fallback=forbidden"
          << std::endl;
      throw feebl::electron_lumi_spatial::SpatialError(
          feebl::electron_lumi_spatial::SpatialErrorCode::OutOfDomainRoot,
          "spatial electron-luminosity admission rejected one or more "
          "whole physical roots");
    }

  CollisionPlanOptions options;
  options.normalization = 1.0e18 / static_cast<double>(timestep_);
  options.max_pairs = static_cast<std::size_t>(
      switches.get_electron_lumi_spatial_max_pairs());
  options.closure_relative_tolerance =
      switches.get_electron_lumi_spatial_closure_tolerance();
  const CollisionPlan plan =
      CollisionPlan::build(geometry, roots1, roots2, options);
  if (!plan.closure().within_tolerance ||
      plan.beam1_accounting().rejected_roots != 0U ||
      plan.beam2_accounting().rejected_roots != 0U)
    throw std::logic_error(
        "spatial electron luminosity received an unqualified collision plan");

  RESULTS encounter_results;
  encounter_results.connect_switches(&switches);
  encounter_results.enable_gui_diagnostics(results_.gui_diagnostics_enabled());
  encounter_results.configure_gui_peak(results_.gui_peak().definition);
  const pairmoments::SpectrumW& spectrum = results_.pair_spectrum();
  if (spectrum.configured())
    encounter_results.configure_pair_spectrum(
        spectrum.lower_edge(), spectrum.upper_edge(), spectrum.bins());
  for (std::vector<CollisionPair>::const_iterator pair = plan.pairs().begin();
       pair != plan.pairs().end(); ++pair)
    {
      const std::map<RootId, const PARTICLE*>::const_iterator particle1 =
          lookup1.find(pair->beam1_id);
      const std::map<RootId, const PARTICLE*>::const_iterator particle2 =
          lookup2.find(pair->beam2_id);
      if (particle1 == lookup1.end() || particle2 == lookup2.end())
        throw std::logic_error(
            "spatial collision plan referenced an unknown physical root ID");
      collide_ee_spatial(*particle1->second, *particle2->second,
                         pair->weight, switches, encounter_results);
    }
  encounter_results.add_lumi_fine(plan.density_dot_oracle().weight);

  ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT encounter_receipt;
  encounter_receipt.seen = true;
  encounter_receipt.geometry_hash = geometry.geometry_hash();
  encounter_receipt.n_x = geometry.n_x();
  encounter_receipt.n_y = geometry.n_y();
  encounter_receipt.x_min = geometry.x_min();
  encounter_receipt.x_max = geometry.x_max();
  encounter_receipt.y_min = geometry.y_min();
  encounter_receipt.y_max = geometry.y_max();
  encounter_receipt.delta_x = geometry.delta_x();
  encounter_receipt.delta_y = geometry.delta_y();
  encounter_receipt.normalization = options.normalization;
  encounter_receipt.max_pairs_per_encounter =
      static_cast<std::uint64_t>(options.max_pairs);
  encounter_receipt.closure_relative_tolerance =
      options.closure_relative_tolerance;
  encounter_receipt.encounters = 1U;
  encounter_receipt.beam1_input_roots =
      static_cast<std::uint64_t>(plan.beam1_accounting().input_roots);
  encounter_receipt.beam1_accepted_roots =
      static_cast<std::uint64_t>(plan.beam1_accounting().accepted_roots);
  encounter_receipt.beam1_rejected_roots =
      static_cast<std::uint64_t>(plan.beam1_accounting().rejected_roots);
  encounter_receipt.beam2_input_roots =
      static_cast<std::uint64_t>(plan.beam2_accounting().input_roots);
  encounter_receipt.beam2_accepted_roots =
      static_cast<std::uint64_t>(plan.beam2_accounting().accepted_roots);
  encounter_receipt.beam2_rejected_roots =
      static_cast<std::uint64_t>(plan.beam2_accounting().rejected_roots);
  encounter_receipt.physical_pairs =
      static_cast<std::uint64_t>(plan.pairs().size());
  encounter_receipt.beam1_input_population =
      plan.beam1_accounting().input_population;
  encounter_receipt.beam1_accepted_population =
      plan.beam1_accounting().accepted_population;
  encounter_receipt.beam1_rejected_population =
      plan.beam1_accounting().rejected_population;
  encounter_receipt.beam2_input_population =
      plan.beam2_accounting().input_population;
  encounter_receipt.beam2_accepted_population =
      plan.beam2_accounting().accepted_population;
  encounter_receipt.beam2_rejected_population =
      plan.beam2_accounting().rejected_population;
  encounter_receipt.pair_weight_sum = plan.closure().pair_weight_sum;
  encounter_receipt.density_dot_weight =
      plan.closure().density_dot_weight;
  encounter_receipt.maximum_closure_relative_error =
      plan.closure().relative_error;
  encounter_receipt.plan_hashes.push_back(plan.plan_hash());

  ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT merged_receipt =
      electron_lumi_spatial_receipt_;
  std::string receipt_reason;
  if (!merge_electron_lumi_spatial_receipt(
          merged_receipt, encounter_receipt, &receipt_reason))
    throw std::runtime_error(
        "spatial electron-luminosity receipt publication failed: " +
        receipt_reason);
  RESULTS merged_results = results_;
  if (!merged_results.merge_primary_photon_collision_from(encounter_results))
    throw std::overflow_error(
        "spatial electron-luminosity aggregate publication overflowed");
  results_ = merged_results;
  electron_lumi_spatial_receipt_ = merged_receipt;
}

void GRID::collide_ee_spatial(const PARTICLE& particle1,
                              const PARTICLE& particle2,
                              double weight,
                              const SWITCHES& switches,
                              RESULTS& destination) const
{
  const double energy1 = static_cast<double>(particle1.energy());
  const double energy2 = static_cast<double>(particle2.energy());
  double vx1 = 0.0;
  double vy1 = 0.0;
  double vx2 = 0.0;
  double vy2 = 0.0;
  particle1.velocities(vx1, vy1);
  particle2.velocities(vx2, vy2);
  if (!electron_lumi_spatial_is_finite(weight) || weight < 0.0 ||
      !electron_lumi_spatial_is_finite(energy1) ||
      !electron_lumi_spatial_is_finite(energy2) ||
      !electron_lumi_spatial_is_finite(vx1) ||
      !electron_lumi_spatial_is_finite(vy1) ||
      !electron_lumi_spatial_is_finite(vx2) ||
      !electron_lumi_spatial_is_finite(vy2))
    throw std::domain_error(
        "spatial electron luminosity received non-finite pair kinematics");
  const double q1 = vx1 * vx1 + vy1 * vy1;
  const double q2 = vx2 * vx2 + vy2 * vy2;
  if (!electron_lumi_spatial_is_finite(q1) ||
      !electron_lumi_spatial_is_finite(q2) || q1 < 0.0 || q2 < 0.0 ||
      q1 > 1.0 || q2 > 1.0)
    throw std::domain_error(
        "spatial electron luminosity received a non-physical velocity");
  const double vz1 = std::sqrt(1.0 - q1);
  const double vz2 = std::sqrt(1.0 - q2);
  const double cosinus = vx1 * vx2 + vy1 * vy2 - vz1 * vz2;
  double ecm_squared = 2.0 * std::fabs(energy1) * std::fabs(energy2) *
                       (1.0 - cosinus);
  const double ecm_scale =
      std::max(1.0, 2.0 * std::fabs(energy1) * std::fabs(energy2));
  if (ecm_squared < 0.0 &&
      ecm_squared >= -64.0 * std::numeric_limits<double>::epsilon() *
                         ecm_scale)
    ecm_squared = 0.0;
  if (!electron_lumi_spatial_is_finite(ecm_squared) ||
      ecm_squared < 0.0)
    throw std::domain_error(
        "spatial electron luminosity produced an invalid center-of-mass "
        "energy");
  const double ecm = std::sqrt(ecm_squared);

  int j1 = 0;
  int j2 = 1;
  if (energy1 < 0.0) j1 = 1;
  if (energy2 < 0.0) j2 = 0;
  destination.add_lumi(j1, j2, weight);

  if (energy1 * energy2 > 0.0)
    {
      destination.add_lumi_ee(weight);
      destination.add_step_lumi_ee(weight);
      destination.add_gui_collision_moments(weight, particle1, particle2);
      const double W_pair = pairmoments::invariant_mass(
          energy1, vx1, vy1, vz1,
          energy2, vx2, vy2, -vz2,
          static_cast<double>(EMASS));
      if (!electron_lumi_spatial_is_finite(W_pair))
        throw std::domain_error(
            "spatial electron luminosity produced a non-finite invariant "
            "mass");

      bool pair_W_counted = false;
      if (switches.get_bmt_precession())
        {
          const TRIDVECTOR& spin1 = particle1.getSpin();
          const TRIDVECTOR& spin2 = particle2.getSpin();
          const double sx1 = spin1.getComponent(0);
          const double sy1 = spin1.getComponent(1);
          const double sz1 = spin1.getComponent(2);
          const double sx2 = spin2.getComponent(0);
          const double sy2 = spin2.getComponent(1);
          const double sz2 = spin2.getComponent(2);
          if (!electron_lumi_spatial_is_finite(sx1) ||
              !electron_lumi_spatial_is_finite(sy1) ||
              !electron_lumi_spatial_is_finite(sz1) ||
              !electron_lumi_spatial_is_finite(sx2) ||
              !electron_lumi_spatial_is_finite(sy2) ||
              !electron_lumi_spatial_is_finite(sz2))
            throw std::domain_error(
                "spatial electron luminosity received non-finite spin");
          const double lab_parallel = 0.5 * (1.0 + sz1 * sz2);
          const double h1 = sx1 * vx1 + sy1 * vy1 + sz1 * vz1;
          const double h2 = sx2 * vx2 + sy2 * vy2 - sz2 * vz2;
          if (!electron_lumi_spatial_is_finite(lab_parallel) ||
              !electron_lumi_spatial_is_finite(h1) ||
              !electron_lumi_spatial_is_finite(h2))
            throw std::domain_error(
                "spatial electron luminosity produced non-finite spin "
                "observables");
          destination.add_lumi_pp(weight * lab_parallel);
          destination.add_lumi_weighted_helicity(weight, h1, h2, sz1, -sz2);
          destination.add_pair_W(weight, W_pair, h1, h2);
          destination.add_gui_peak(weight,W_pair,energy1,energy2,true,h1,h2);
          pair_W_counted = true;
        }
      if (!pair_W_counted) {
        destination.add_pair_W(weight, W_pair);
        destination.add_gui_peak(weight,W_pair,energy1,energy2);
      }

      if (ecm > static_cast<double>(switches.get_ecm_min()))
        {
          destination.add_lumi_ee_high(weight);
          destination.add_step_lumi_ee_high(weight);
        }
      const double weighted_ecm = weight * ecm;
      const double weighted_ecm2 = weighted_ecm * ecm;
      if (!electron_lumi_spatial_is_finite(weighted_ecm) ||
          !electron_lumi_spatial_is_finite(weighted_ecm2))
        throw std::overflow_error(
            "spatial electron-luminosity E_cm moment overflow");
      destination.add_lumi_ecm(weighted_ecm);
      destination.add_lumi_ecm2(weighted_ecm2);
    }
}

void GRID::collide_ee(int cellx, int celly,float min_z, const BEAM_PARTICLE_POINTER& pointer1, const BEAM_PARTICLE_POINTER& pointer2, SWITCHES& switches,PAIR_BEAM& secondaries, int time_counter, int beamslice1=-1, int beamslice2=-1)
{
  //  JET_FLOAT bhabhan;
  float help,ecm,e1,e2;
  //float ecmratio;
  float weight = pointer1.weight() * pointer2.weight();
  int j1=0,j2=1;
  int bmt_precession = switches.get_bmt_precession();

  e1 =  fabs(pointer1.energy());
  e2 =  fabs(pointer2.energy());
  // The energy spread is NOT applied here any more. It is applied once per
  // particle at beam setup (FEEBL::apply_energy_spread), so e1 and e2 already
  // carry it, along with every other part of the interaction. Applying it here
  // as well would spread twice.

  if (switches.get_do_isr()) isr2(e1,e2,&e1,&e2, *rndm_generator_);
  
  // Modification of the ecm by Strahinja to be safer with CLIC
  float vx1, vy1, vz1, vx2, vy2, vz2;
  float cosinus;
  pointer1.velocities(vx1, vy1);
  pointer2.velocities(vx2, vy2);
  vz1 = sqrt(1. - vx1*vx1 - vy1*vy1);
  vz2 = sqrt(1. - vx2*vx2 - vy2*vy2);
  cosinus = (vx1*vx2 + vy1*vy2 - vz1*vz2); // The minus sign because the particles have opposite vz
  ecm=sqrt(2.0*e1*e2*(1. - cosinus));

  float energy1 = pointer1.energy();
  float energy2 = pointer2.energy();
  if (switches.get_do_lumi()&1)
    {
      float p1Vx, p1Vy, p2Vx, p2Vy;
      pointer1.velocities(p1Vx, p1Vy);
      pointer2.velocities(p2Vx, p2Vy);
      if ( bmt_precession ) 
      //      if (SPIN)
		{
		  lumi_heap_ee_.lumi_store_ee(mesh_, cellx, celly,min_z, energy1, p1Vx, p1Vy, energy2, p2Vx, p2Vy, weight, pointer1.getSpin(),pointer2.getSpin() , time_counter);
		}
      else lumi_heap_ee_.lumi_store_ee(mesh_, cellx, celly,min_z, energy1, p1Vx, p1Vy, energy2, p2Vx, p2Vy, weight,time_counter);
    }

  if (energy1 < 0.0) j1=1;
  if (energy2 < 0.0) j2=0;
  results_.add_lumi(j1, j2, weight);

  if (energy1*energy2 > 0.0)
    {
      if (switches.get_do_cross())
		{
		  cross_->cross_add(e1,e2,weight);
		}
      results_.add_lumi_ee(weight);
      results_.add_step_lumi_ee(weight);
      results_.add_gui_collision_moments(weight, pointer1.diagnostic_particle(),
                                         pointer2.diagnostic_particle());

      // Exact pair invariant mass, electron mass kept.
      //
      // NOT the same quantity as `ecm` above: that uses the massless form
      // sqrt(2 E1 E2 (1 - cos)). The two agree to 1e-14 at collider energies
      // and diverge for asymmetric pairs, which is what a spectrum tail is
      // made of. Both are kept; neither replaces the other.
      //
      // vz2 was formed as a positive magnitude above and the sign folded into
      // `cosinus`, so the direction handed over here has to carry it
      // explicitly: beam 2 travels toward -z.
      const double W_pair = pairmoments::invariant_mass(
            (double)energy1, (double)vx1, (double)vy1,  (double)vz1,
            (double)energy2, (double)vx2, (double)vy2, -(double)vz2,
            (double)EMASS);

      if(design_axis_spectrum_.enabled())
        design_axis_spectrum_.add(static_cast<double>(weight),
          std::fabs(static_cast<double>(energy1)),vx1,vy1,
          std::fabs(static_cast<double>(energy2)),vx2,vy2);

      // Every counted pair contributes to dL/dW; only a pair that actually
      // carries spin contributes to the helicity moments, so an unpolarized
      // run still gets a spectrum without diluting helicities that do not
      // exist.
      bool pair_W_counted = false;

      if ( bmt_precession ) 
	//     if (SPIN)
		{
		  float spin1= pointer1.getSpin()(2);
		  float spin2= pointer2.getSpin()(2);
		  help=0.5*(1.0+spin1*spin2);
		  results_.add_lumi_pp(weight*help);

		  // Luminosity-weighted helicity. Same pairs, same weights as
		  // lumi_ee, so this is exactly
		  //     P_lumi = int L(t) P(t) dt / int L(t) dt
		  // over the particles that actually collide -- which is what a
		  // physics analysis uses, and is NOT the bunch average.
		  //
		  // Helicity, not the lab z projection: each spin is projected on
		  // its OWN direction of motion, and beam 2 travels toward -z.
		  // (add_lumi_pp above still uses the lab Sz product; that is a
		  // separate known limitation, not changed here.)
		  {
		    // p1Vx..p2Vy are scoped to the lumi-heap block above, so read
		    // the velocities from the particles directly.
		    float w1x, w1y, w2x, w2y;
		    pointer1.velocities(w1x, w1y);
		    pointer2.velocities(w2x, w2y);
		    const double q1 = (double)w1x*(double)w1x + (double)w1y*(double)w1y;
		    const double q2 = (double)w2x*(double)w2x + (double)w2y*(double)w2y;
		    if (q1 < 1.0 && q2 < 1.0)
		      {
			const double helicity_vz1 =  std::sqrt(1.0 - q1);
			const double helicity_vz2 = -std::sqrt(1.0 - q2);
			const TRIDVECTOR& s1 = pointer1.getSpin();
			const TRIDVECTOR& s2 = pointer2.getSpin();
			const double h1 = s1.getComponent(0)*(double)w1x
			                + s1.getComponent(1)*(double)w1y
			                + s1.getComponent(2)*helicity_vz1;
			const double h2 = s2.getComponent(0)*(double)w2x
			                + s2.getComponent(1)*(double)w2y
			                + s2.getComponent(2)*helicity_vz2;
			results_.add_lumi_weighted_helicity((double)weight, h1, h2,
                  s1.getComponent(2), -s2.getComponent(2));
			results_.add_pair_W((double)weight, W_pair, h1, h2);
            results_.add_gui_peak((double)weight,W_pair,energy1,energy2,true,h1,h2);
			pair_W_counted = true;
		      }
		  }
		}

      // A pair with no stored spin still belongs in dL/dW. Folding it into
      // the helicity moments as (0,0) would pull those means toward zero
      // with pairs that never carried a helicity, so the luminosity-only
      // overload is used instead.
      if (!pair_W_counted) {
        results_.add_pair_W((double)weight, W_pair);
        results_.add_gui_peak((double)weight,W_pair,energy1,energy2);
      }

      if (ecm > switches.get_ecm_min()) {
	results_.add_lumi_ee_high(weight);
        results_.add_step_lumi_ee_high(weight);
      }
      help= weight*ecm;
      results_.add_lumi_ecm(help);
      results_.add_lumi_ecm2(help*ecm);
    
    }
  
  // here the bhabhas

  if (switches.get_do_bhabhas())
    {
      float part1Vx, part1Vy, part2Vx, part2Vy;
      pointer1.velocities(part1Vx, part1Vy);
      pointer2.velocities(part2Vx, part2Vy);
      // The following line changed by SL to use signed energies (needed in boost_bhabha())
	bhabhas_.make_bhabha(secondaries, part1Vx, part1Vy, part2Vx, part2Vy, e1, e2, ecm, weight, mesh_, cellx, celly,min_z, switches, *rndm_generator_, beamslice1, beamslice2 );

// The following alternative tested whether it matters which beam moves in which direction
//      bhabhas_.make_bhabha(secondaries, part1Vx, part1Vy, part2Vx, part2Vy, -e1, e2, ecm, weight, mesh_, cellx, celly,min_z, switches, *rndm_generator_, beamslice1, beamslice2);
    }

  if (switches.get_do_jets())
    {
      minijets_->mkjll_(secondaries.get_pair_parameters(),pointer1.energy(),pointer2.energy(), weight, switches, *rndm_generator_);

    }
  if (switches.get_do_prod()==1)
    {

      std::cerr << " GRID::collide_ee : do_prod not implemented! " << std::endl;
      exit(1);
    }  
}


void GRID::isr2(float e1,float e2,float *e1p,float *e2p, RNDM& rndm_generator)
{
    double c=2.0*ALPHA_EM/PI,emass2=EMASS*EMASS;
    double x,s,beta,corr,tmp;

    s=4.0*e1*e2;
    beta=c*(log(s/emass2)-1.0);
    do{
	x=pow(1.0-rndm_generator.rndm(),2.0/beta);
	tmp=0.5*beta*pow(x,0.5*beta-1.0)*(1.0+0.375*beta);
	corr=(tmp-0.25*beta*(2.0-x))/tmp;
    }
    while(rndm_generator.rndm()>corr);
    *e1p=e1*(1.0-x);
    do{
	x=pow(1.0-rndm_generator.rndm(),2.0/beta);
	tmp=0.5*beta*pow(x,0.5*beta-1.0)*(1.0+0.375*beta);
	corr=(tmp-0.25*beta*(2.0-x))/tmp;
    }
    while(rndm_generator.rndm()>corr);
    *e2p=e2*(1.0-x);
}


/*! This routine moves the particles one timestep. */

struct GRID::CPU_PRIMARY_MOVE_CONTEXT
{
  GRID* grid;
  const std::vector<GENERAL_GRID*>* grids;
  const std::vector<PARTICLE*>* particles1;
  const std::vector<PARTICLE*>* particles2;
  const PHI_FLOAT* phi1;
  const PHI_FLOAT* phi2;
  float step;
  float scale1;
  float scale2;
  std::size_t count1;
  int extra_grids;
};

void GRID::cpu_primary_move_range(void* opaque,
                                  std::size_t /*lane*/,
                                  std::size_t begin,
                                  std::size_t end)
{
  CPU_PRIMARY_MOVE_CONTEXT& context =
      *static_cast<CPU_PRIMARY_MOVE_CONTEXT*>(opaque);
  const std::size_t beam1_end = std::min(end, context.count1);
  for (std::size_t index = begin; index < beam1_end; ++index)
    {
      PARTICLE* particle = (*context.particles1)[index];
      const TRIVECTOR field = context.grid->EBfieldOnParticle(
          particle, *context.grids, context.phi1, 1, context.extra_grids);
      particle->advanceDueToEBfield(field, context.step, context.scale1);
    }

  const std::size_t beam2_begin = std::max(begin, context.count1);
  for (std::size_t combined = beam2_begin; combined < end; ++combined)
    {
      PARTICLE* particle = (*context.particles2)[combined - context.count1];
      const TRIVECTOR field = context.grid->EBfieldOnParticle(
          particle, *context.grids, context.phi2, 2, context.extra_grids);
      particle->advanceDueToEBfield(field, context.step, context.scale2);
    }
}

void GRID::move_primary_particles_parallel(
    const std::vector<GENERAL_GRID*>& grids,
    int i_slice1,
    int i_slice2,
    int extra_grids)
{
  const std::vector<PARTICLE*>& particles1 =
      slice_of_beam_[0].get_beam()->getParticleVector(i_slice1);
  const std::vector<PARTICLE*>& particles2 =
      slice_of_beam_[1].get_beam()->getParticleVector(i_slice2);
  if (particles2.size() >
      std::numeric_limits<std::size_t>::max() - particles1.size())
    throw std::overflow_error("CPU primary movement particle count overflow");

  CPU_PRIMARY_MOVE_CONTEXT context;
  context.grid = this;
  context.grids = &grids;
  context.particles1 = &particles1;
  context.particles2 = &particles2;
  context.phi1 = champ_.get_phi(2);  // field acting on beam 1
  context.phi2 = champ_.get_phi(1);  // field acting on beam 2
  context.step = step_;
  context.scale1 = slice_of_beam_[0].get_scal_step();
  context.scale2 = slice_of_beam_[1].get_scal_step();
  context.count1 = particles1.size();
  context.extra_grids = extra_grids;

  const std::size_t total = particles1.size() + particles2.size();
  PARALLEL_BACKEND& backend = parallel_backend();
  backend.run_static_ranges(total, &context, &GRID::cpu_primary_move_range);
  backend.record_cpu_primary_move_execution(total);
}

void GRID::move_particles(const std::vector<GENERAL_GRID*>& grids,
                          int i_beam, int i_slice, int interpolation,
                          int do_beamstrahlung, int do_trident, int sokolov,
                          float emin, int /*do_prod*/, int extra_grids,
                          float charge_sign, int bmt_rotate,
                          const float* prefetched_fields,
                          std::size_t prefetched_particle_count)
{
  float vx,vy,xpos,ypos;

  // ultrarelivistic particles 
  //      Bfield *= 1.0/Cvelocity;
  // electric field: GV/nm
  // for the magnetic field, the product c.B
  // also in GV/nm
  //      Efield = EBfield;

  unsigned int k,i;
  const PHI_FLOAT *phi;
  //  BEAM* beam;
  if (i_beam==1) 
    {
      //     beam = slice_of_beam_[0].get_beam();
      //      phi=phi2_;
      phi= champ_.get_phi(2);
    }
  else 
    {
      //     beam = slice_of_beam_[1].get_beam();
      //     phi=phi1_;
      phi= champ_.get_phi(1);
    }
  const std::vector<PARTICLE*>& theParticles = slice_of_beam_[i_beam-1].get_beam()->getParticleVector(i_slice);
  if (prefetched_fields != NULL || prefetched_particle_count != 0)
    {
      if (prefetched_particle_count != theParticles.size() ||
          (prefetched_particle_count != 0 && prefetched_fields == NULL))
        {
          std::cerr << " GRID::move_particles : compact field count does not "
                       "match the active primary slice" << std::endl;
          exit(1);
        }
    }
  switch (interpolation)
    {
    case 1:
      std::cerr << " GRID::move_particles interpolation = 1 is not implemented " << std::endl;
      exit(1);
      break;
    case 2:
      {
	TRIVECTOR EBfield;
	TRIDVECTOR Efield, Bfield;
	float dzOnRadius,oldener;
	////////////////////////////////////////////////
	// in case of do_beamstrahlung, see the implementation in the case of (do_prod>1)&&(i_beam==1) 
	///////////////////////////////////////////////
	PARTICLE* particleIterator;
	for (k = 0; k < theParticles.size(); k++)
	  {
	    particleIterator = theParticles[k];
	    oldener=particleIterator->getEnergy();
	    if (prefetched_fields != NULL)
	      EBfield.setComponents(prefetched_fields[2 * k],
	                            prefetched_fields[2 * k + 1], 0.0f);
	    else
	      EBfield = EBfieldOnParticle(particleIterator, grids, phi,
	                                  i_beam, extra_grids);


	    float vx_kick, vy_kick;   // emitter direction before the kick (photon_timing = 3)
	    particleIterator->velocities(vx_kick, vy_kick);
	    dzOnRadius = particleIterator->advanceDueToEBfield(EBfield, step_,slice_of_beam_[i_beam-1].get_scal_step());
            // Actual current kick, before radiation or trident rescaling.
            results_.add_gui_chi_e(i_beam,particleIterator->getUps());


	    Efield = EBfield;
	    // Reconstruct B = beta_source x E with the SOURCE beam direction.
	    // In BMT, transverse B and beta_particle x E add for opposite-moving
	    // bunches. Using one source sign for both beams would spuriously cancel
	    // the beam-2 precession. The orbit still uses EBfield directly.
	    {
	      // The quasi-static beam solver supplies transverse fields only.
	      // Longitudinal fields are not reconstructed; their omission is a
	      // model approximation, not a bound inferred from gamma alone.
	      // The source beam travels opposite to bsrc, giving these components
	      // of beta_source cross E in the native interaction-field convention.
	      const double bsrc = (i_beam == 1) ? 1.0 : -1.0;
	      Bfield = TRIDVECTOR(bsrc*EBfield(1), -bsrc*EBfield(0), 0.0);
	    }
	    if (bmt_rotate)
	      {
		// The Green function already contains the relative beam charge.
		// rotateBMT's -q prefactor therefore uses -1 with these effective
		// fields, consistently with the primary orbit kick. Multiplying
		// charge_sign again would double-count it. The momentum direction
		// is independently +z for beam 1 and -z for beam 2.
		// See CHARGE_SIGN_DERIVATION.md for the field normalization.
		particleIterator->rotateBMT(Efield, Bfield, -1.0f, step_,
		                            (i_beam == 1) ? 1.0f : -1.0f);
		if (do_beamstrahlung)
		  { 
		    std::vector<float> photonEnergies;
		    // photon_polarisation=1: three emission-frame Stokes parameters per photon (no random numbers)
		    std::vector<float> photonPol;
		    std::vector<float>* pol = photon_polarisation_ ? &photonPol : NULL;
		    results_.updateUpsmax(particleIterator->getUps());
		    if (sokolov) 
		      {
			particleIterator->beamstrahlungSokolov(Efield, Bfield, dzOnRadius, emin, charge_sign,photonEnergies, *rndm_generator_,
						       (i_beam == 1) ? 1.0f : -1.0f, pol);
		      }
		    else 
		      {
			particleIterator->beamstrahlung(Efield, Bfield, dzOnRadius,emin,photonEnergies, *rndm_generator_, pol);
		      }
		    registerPhotons(photonEnergies, *(particleIterator), i_beam, i_slice, vx_kick, vy_kick, oldener, pol, EBfield(0), EBfield(1));
		  }
	      }
	    else
	      {
		if(do_beamstrahlung)
		  {
		    std::vector<float> photonEnergies;
		    std::vector<float> photonPol;
		    std::vector<float>* pol = photon_polarisation_ ? &photonPol : NULL;
		    results_.updateUpsmax(particleIterator->getUps());
		    // 		    particleBeamstrahlung(particleIterator,Efield, Bfield, dzOnRadius, emin, photonEnergies);
		    particleIterator->beamstrahlung(Efield, Bfield, dzOnRadius, emin, photonEnergies, *rndm_generator_, pol);
		    registerPhotons(photonEnergies, *(particleIterator), i_beam, i_slice, vx_kick, vy_kick, oldener, pol, EBfield(0), EBfield(1));
		  }
	      }
	    if(do_trident)
	      {
		std::vector<float> electrons,positrons,virt;
		if(do_beamstrahlung)
		  {
		    particleIterator->setUps(particleIterator->getUps()*particleIterator->getEnergy()/oldener);
		  }
		particleIterator->createTridents(step_*slice_of_beam_[i_beam-1].get_scal_step(),&electrons,&positrons,&virt,*rndm_generator_);
		for(i=0;i<electrons.size();i++)
		  {
		    particleIterator->XYposition(xpos,ypos);
		    particleIterator->velocities(vx,vy);
		    store_trident_particle(*slice_of_beam_[i_beam-1].get_beam(),electrons[i],vx,vy,xpos,ypos,i_slice);
		    store_trident_particle(*slice_of_beam_[i_beam-1].get_beam(),positrons[i],vx,vy,xpos,ypos,i_slice);
		  }
	      }
	  }
	break;
      }
    default:
      std::cerr << " GRID::move_particles : unknown value of interpolation type :  " << interpolation << std::endl;
      exit(1);
    }
}

struct GRID::GPU2_PRIMARY_MOVE_CONTEXT
{
  const std::vector<PARTICLE*>* particles1;
  const std::vector<PARTICLE*>* particles2;
  const float* fields1;
  const float* fields2;
  float step;
  float scale1;
  float scale2;
  std::size_t count1;
};

void GRID::gpu2_primary_move_range(void* opaque,
                                   std::size_t /*lane*/,
                                   std::size_t begin,
                                   std::size_t end)
{
  GPU2_PRIMARY_MOVE_CONTEXT& context =
      *static_cast<GPU2_PRIMARY_MOVE_CONTEXT*>(opaque);
  const std::size_t beam1_end = std::min(end, context.count1);
  for (std::size_t combined = begin; combined < beam1_end; ++combined)
    {
      TRIVECTOR field;
      field.setComponents(context.fields1[2 * combined],
                          context.fields1[2 * combined + 1], 0.0f);
      (*context.particles1)[combined]->advanceDueToEBfield(
          field, context.step, context.scale1);
    }

  const std::size_t beam2_begin = std::max(begin, context.count1);
  for (std::size_t combined = beam2_begin; combined < end; ++combined)
    {
      const std::size_t particle = combined - context.count1;
      TRIVECTOR field;
      field.setComponents(context.fields2[2 * particle],
                          context.fields2[2 * particle + 1], 0.0f);
      (*context.particles2)[particle]->advanceDueToEBfield(
          field, context.step, context.scale2);
    }
}

void GRID::movePrimaryParticlesWithFields(
    const std::vector<GENERAL_GRID*>& grids,
    int i_slice1, int i_slice2, int interpolation,
    int do_beamstrahlung, int sokolov, float emin, int do_prod,
    int extra_grids, float charge_sign, int bmt_rotate,
    const float* fields1, std::size_t field_particle_count1,
    const float* fields2, std::size_t field_particle_count2,
    const std::string& primary_mover, int host_threads,
    int do_trident, int do_coherent)
{
  if (!compactPrimaryFieldContractEligible(
          interpolation, extra_grids, do_prod))
    {
      std::cerr << " GRID::movePrimaryParticlesWithFields : entered outside "
                   "the committed primary GPU2 contract" << std::endl;
      exit(1);
    }
  const std::size_t actual_count1 =
      slice_of_beam_[0].get_beam()->getParticleVector(i_slice1).size();
  const std::size_t actual_count2 =
      slice_of_beam_[1].get_beam()->getParticleVector(i_slice2).size();
  if (field_particle_count1 != actual_count1 ||
      field_particle_count2 != actual_count2 ||
      (actual_count1 != 0 && fields1 == NULL) ||
      (actual_count2 != 0 && fields2 == NULL))
    {
      std::cerr << " GRID::movePrimaryParticlesWithFields : compact field "
                   "buffers do not match the active primary slices"
                << std::endl;
      exit(1);
    }

  if (primary_mover != "serial" && primary_mover != "openmp" &&
      primary_mover != "pool")
    throw std::logic_error(
        "compact primary movement received an uncommitted host mover");
  if (host_threads < 1)
    throw std::logic_error(
        "compact primary movement requires at least one host thread");
  if (actual_count2 >
      std::numeric_limits<std::size_t>::max() - actual_count1)
    throw std::overflow_error(
        "GPU2 compact primary movement particle count overflow");

  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu2_plan_committed() ||
      backend.actual_gpu2_options().scope != "primary" ||
      backend.actual_gpu2_options().primary_mover != primary_mover ||
      backend.actual_gpu2_options().host_threads != host_threads)
    throw std::logic_error(
        "compact primary movement does not match the committed GPU2 plan");
  const std::size_t total = actual_count1 + actual_count2;

  if (primary_mover == "serial")
    {
      if (host_threads != 1)
        throw std::logic_error(
            "serial compact primary movement received extra host threads");
      movePrimaryParticlesWithPrefetchedFields(
          grids, i_slice1, i_slice2, interpolation, do_beamstrahlung,
          sokolov, emin, do_prod, extra_grids, charge_sign, bmt_rotate,
          fields1, field_particle_count1, fields2, field_particle_count2,
          do_trident, do_coherent);
      backend.record_gpu2_primary_mover_execution(
          primary_mover, total, 1U, total == 0 ? 0U : 1U);
      return;
    }

  // Parallel ranges are legal only when the mover consists solely of the
  // independent orbit kick.  This second guard is intentionally redundant
  // with FEEBL preflight so future direct callers cannot enter shared RNG,
  // spin, photon, trident, or result-update code accidentally.
  if (!compactPrimaryParallelMoverContractEligible(
          interpolation, do_beamstrahlung, sokolov, do_prod, extra_grids,
          bmt_rotate, do_trident, do_coherent))
    throw std::logic_error(
        "parallel compact primary movement received shared physics side effects");
  GPU2_PRIMARY_MOVE_CONTEXT context;
  context.particles1 =
      &slice_of_beam_[0].get_beam()->getParticleVector(i_slice1);
  context.particles2 =
      &slice_of_beam_[1].get_beam()->getParticleVector(i_slice2);
  context.fields1 = fields1;
  context.fields2 = fields2;
  context.step = step_;
  context.scale1 = slice_of_beam_[0].get_scal_step();
  context.scale2 = slice_of_beam_[1].get_scal_step();
  context.count1 = actual_count1;

  if (primary_mover == "pool")
    backend.run_gpu2_primary_static_ranges(
        total, &context, &GRID::gpu2_primary_move_range);
  else
    {
#ifdef FEEBL_HAVE_OPENMP
      int observed_team_size = 1;
#pragma omp parallel num_threads(host_threads) shared(context, observed_team_size)
      {
#pragma omp single
        observed_team_size = omp_get_num_threads();
#pragma omp barrier
        // Re-check the exact team before any particle is changed.  A runtime
        // limit introduced after preflight is a hard failure, never a silent
        // smaller-team execution.
        if (observed_team_size == host_threads)
          {
            const std::size_t lane =
                static_cast<std::size_t>(omp_get_thread_num());
            const std::size_t workers =
                static_cast<std::size_t>(observed_team_size);
            const std::size_t quotient = total / workers;
            const std::size_t remainder = total % workers;
            const std::size_t begin =
                lane * quotient + std::min(lane, remainder);
            const std::size_t count = quotient + (lane < remainder);
            if (count != 0)
              gpu2_primary_move_range(&context, lane, begin, begin + count);
          }
      }
      if (observed_team_size != host_threads)
        throw std::runtime_error(
            "OpenMP formed a different GPU2 primary team after preflight; "
            "no particle was moved");
#else
      throw std::logic_error(
          "OpenMP compact primary movement entered a non-OpenMP build");
#endif
    }

  const std::size_t nonempty_ranges = std::min(
      total, static_cast<std::size_t>(host_threads));
  backend.record_gpu2_primary_mover_execution(
      primary_mover, total, static_cast<std::size_t>(host_threads),
      nonempty_ranges);
}

void GRID::gatherPrimaryFieldsForSlices(
    int i_slice1, int i_slice2,
    float* fields1, std::size_t field_particle_count1,
    float* fields2, std::size_t field_particle_count2) const
{
  const std::vector<PARTICLE*>& particles1 =
      slice_of_beam_[0].get_beam()->getParticleVector(i_slice1);
  const std::vector<PARTICLE*>& particles2 =
      slice_of_beam_[1].get_beam()->getParticleVector(i_slice2);
  if (field_particle_count1 != particles1.size() ||
      field_particle_count2 != particles2.size() ||
      (!particles1.empty() && fields1 == NULL) ||
      (!particles2.empty() && fields2 == NULL))
    throw std::invalid_argument(
        "compact field gather buffers do not match the active slices");

  // With extra_grids=0, which is enforced by both production callers, the
  // historical out-of-grid contract is exactly a zero field.  Supplying only
  // this GRID therefore reproduces EBfieldOnParticle without consulting any
  // shared mutable collision state.
  std::vector<GENERAL_GRID*> grids(1U, const_cast<GRID*>(this));
  for (std::size_t particle = 0; particle < particles1.size(); ++particle)
    {
      const TRIVECTOR field = EBfieldOnParticle(
          particles1[particle], grids, get_phi2(), 1, 0);
      fields1[2U * particle] = field(0);
      fields1[2U * particle + 1U] = field(1);
    }
  for (std::size_t particle = 0; particle < particles2.size(); ++particle)
    {
      const TRIVECTOR field = EBfieldOnParticle(
          particles2[particle], grids, get_phi1(), 2, 0);
      fields2[2U * particle] = field(0);
      fields2[2U * particle + 1U] = field(1);
    }
}

void GRID::movePrimaryParticlesWithPrefetchedFields(
    const std::vector<GENERAL_GRID*>& grids,
    int i_slice1, int i_slice2, int interpolation,
    int do_beamstrahlung, int sokolov, float emin, int do_prod,
    int extra_grids, float charge_sign, int bmt_rotate,
    const float* fields1, std::size_t field_particle_count1,
    const float* fields2, std::size_t field_particle_count2,
    int do_trident, int do_coherent)
{
  if (!compactPrimaryFieldContractEligible(
          interpolation, extra_grids, do_prod) ||
      do_trident != 0 || do_coherent != 0)
    throw std::logic_error(
        "prefetched primary movement entered outside its audited contract");
  const std::size_t actual_count1 =
      slice_of_beam_[0].get_beam()->getParticleVector(i_slice1).size();
  const std::size_t actual_count2 =
      slice_of_beam_[1].get_beam()->getParticleVector(i_slice2).size();
  if (field_particle_count1 != actual_count1 ||
      field_particle_count2 != actual_count2 ||
      (actual_count1 != 0 && fields1 == NULL) ||
      (actual_count2 != 0 && fields2 == NULL))
    throw std::invalid_argument(
        "prefetched primary fields do not match the active slices");

  // Only the field provider changes.  The historical physics mover remains
  // authoritative and beam 1 is committed before beam 2 exactly as before.
  move_particles(grids, 1, i_slice1, interpolation, do_beamstrahlung, 0,
                 sokolov, emin, do_prod, extra_grids, charge_sign,
                 bmt_rotate, fields1, field_particle_count1);
  move_particles(grids, 2, i_slice2, interpolation, do_beamstrahlung, 0,
                 sokolov, emin, do_prod, extra_grids, charge_sign,
                 bmt_rotate, fields2, field_particle_count2);
}


void GRID::move_pairs(const std::vector<GENERAL_GRID*>& grids, PAIR_BEAM& pairs, int i_slice, double d_eps_1, double d_eps_2, int extra_grids, float charge_sign_0, RNDM& rndm_generator)
{
  float stepLocal,d;
  int n_pair_steps;
  double mass=pairs.get_pair_parameters().get_mass();

  std::vector<PAIR_PARTICLE>& the_pairs = pairs.get_pairs(i_slice);
  //  std::list<PAIR_PARTICLE>::iterator itr;
  unsigned int k;
  //  for (itr = the_pairs.begin(); itr !=  the_pairs.end(); itr++)
  for (k = 0; k <  the_pairs.size(); k++)
    {      
      //      d=sqrt((rho_sum_1_*d_eps_1 + rho_sum_2_*d_eps_2)/fabs(itr->energy()));
      d=sqrt( (rho_sum_1_*d_eps_1 + rho_sum_2_*d_eps_2)/the_pairs[k].unsigned_energy() );
      n_pair_steps=(int)d+1;
      pairs.add_pair_steps(n_pair_steps);
      stepLocal=step_/(float)n_pair_steps;
      step_pair_1(grids,the_pairs[k],mass,stepLocal,n_pair_steps, extra_grids, charge_sign_0, rndm_generator);
    }
}

void GRID::move_pairs_tertphot(const std::vector<GENERAL_GRID*>& grids, PAIR_BEAM& pairs, int i_slice, double d_eps_1, double d_eps_2, int extra_grids, float charge_sign_0, RNDM& rndm_generator)
{
  float stepLocal,d;
  int n_pair_steps;
  double mass=pairs.get_pair_parameters().get_mass();

  std::vector<PAIR_PARTICLE>& the_pairs = pairs.get_pairs(i_slice);
  //  std::list<PAIR_PARTICLE>::iterator itr;
  unsigned int k;
  //  for (itr = the_pairs.begin(); itr !=  the_pairs.end(); itr++)
  for (k = 0; k <  the_pairs.size(); k++)
    {      
      //      d=sqrt((rho_sum_1_*d_eps_1 + rho_sum_2_*d_eps_2)/fabs(itr->energy()));
      d=sqrt( (rho_sum_1_*d_eps_1 + rho_sum_2_*d_eps_2)/the_pairs[k].unsigned_energy() );
      n_pair_steps=(int)d+1;
      pairs.add_pair_steps(n_pair_steps);
      stepLocal=step_/(float)n_pair_steps;
      step_pair_1_tertphot(grids,the_pairs[k],mass,stepLocal,n_pair_steps, extra_grids, charge_sign_0, rndm_generator);
    }
}

/*
void GRID::deltaVelocityFromFieldCIC(float xpart,float ypart, float energy, PHI_FLOAT *phi, float distance, float& deltavx, float& deltavy)
{
  PHI_FLOAT phi1_x,phi2_x,phi3_x,phi1_y,phi2_y,phi3_y,h_x,h_y;

  interpolePotential(xpart, ypart, h_x, h_y, phi1_x, phi2_x, phi3_x, phi1_y, phi2_y, phi3_y, phi);

  deltavx = (h_x*(phi1_y-phi2_y)+(1.0-h_x)*(phi2_y-phi3_y))*delta_x_inverse_/energy;
  deltavy = (h_y*(phi1_x-phi2_x)+(1.0-h_y)*(phi2_x-phi3_x))*delta_y_inverse_/energy;
  //  std::cout << " :deltaVelocityFromFieldCIC Ex/g= " << deltavx << std::endl;
      deltavx = -deltavx*distance;
      deltavy = -deltavy*distance;
}
*/

TRIVECTOR GRID::electric_field_out_of_main_grid(const std::vector<GENERAL_GRID*>& grids,int beam,PHI_FLOAT x,PHI_FLOAT y, int extra_grids) const
{
  int k;
  PHI_FLOAT tmp,dx,dy;
  const PHI_FLOAT *phi;
  // float temp;
  float ax, ay;
  TRIVECTOR EBfield;
  ax = 0.0;
  ay = 0.0;
  for(k = 0;k <= extra_grids;k++)
    {
      if (grids[k]->coordinatesInGridRange(x,y))
	{
	  if (beam == 1 ) phi = grids[k]->get_phi2();
	  else phi = grids[k]->get_phi1();

	  // electric field in V/m
	  EBfield = grids[k]->ElectricFieldCIC(x, y, phi);
	  return EBfield;
	}
    }
  if (extra_grids<2) 
    {
      EBfield.setComponents(0.0, 0.0, 0.0);
      return EBfield;
    }
  // particle is not in largest grid 
  if (beam==1)
    {
      dx=(x-grids[0]->get_rho_x_2());
      dy=(y-grids[0]->get_rho_y_2());
      tmp=grids[0]->get_rho_factor()*grids[0]->get_rho_sum_2()/(dx*dx+dy*dy);
      ax=dx*tmp;
      ay=dy*tmp;
    }
  else
    {
      dx=(x-grids[0]->get_rho_x_1());
      dy=(y-grids[0]->get_rho_y_1());
      tmp=grids[0]->get_rho_factor()*grids[0]->get_rho_sum_1()/(dx*dx+dy*dy);
      ax=dx*tmp;
      ay=dy*tmp;
    }

      EBfield.setComponents(0.5*ax, 0.5*ay, 0.0);
      return EBfield;
}


// photon_polarisation=1: photonPol holds the photons' Stokes parameters in the emission frame e1 = transverse
// acceleration of the emitter, i.e. along +-(Ex, Ey) (head-on: E and v x B are parallel), e2 = e3 x e1, e3 along
// the emitter's beam (+z for beam 1, -z for beam 2); stored as lab-frame Stokes parameters (qedbg_photon_stokes.h)
void GRID::registerPhotons(const std::vector<float>& photonEnergies, PARTICLE& particle, int i_beam, int i_slice, float vx_kick, float vy_kick, float e_before,
                           const std::vector<float>* photonPol, float Ex, float Ey)
{
  unsigned int k;
  BEAM* beam = slice_of_beam_[i_beam-1].get_beam();
  const bool polarised = (photonPol != NULL && photonPol->size() == 3 * photonEnergies.size());
  // photon_angle: energy of the emitter at each emission (the photons are listed in emission order)
  double e_emitter = e_before;
  //  int  i_beam = beam.label();
  //  std::cout << " GRID::createPhotons : nb photons recuperes " << photonEnergies.size() << std::endl;
  for (k=0; k < photonEnergies.size() ;k++)
    {
      
      //      const PHOTON& foton = 
      if (polarised)
	{
	  float stokes[3];
	  qedbg::stokes_frame_to_lab(&(*photonPol)[3 * k], Ex, Ey, (i_beam == 1) ? +1 : -1, stokes);
	  beam->new_photon(photonEnergies[k], particle, i_slice, stokes);
	}
      else
	beam->new_photon(photonEnergies[k],particle, i_slice);
      // photon_timing >= 1: the emitter has already drifted over this step and the photon is collinear
      // with it, so x - v*step*scal_step is the photon's position at the start of the step -- the time
      // level of the particles, virtual photons and fields this step's collisions use (legacy: binned
      // one step ahead)
      if (photon_timing_)
	{
	  PHOTON& photon = beam->getPhotonVector(i_slice).back();
	  photon.advancePosition(-step_*slice_of_beam_[i_beam-1].get_scal_step());
	  // photon_timing = 3: the step's radiation is spread over its kick, during which the emitter turns
	  // from (vx_kick, vy_kick) to its present direction; the photons leave, on average, along the mean
	  if (photon_timing_ == 3)
	    {
	      float vx, vy;
	      photon.velocities(vx, vy);
	      photon.setVelocities(0.5f*(vx + vx_kick), 0.5f*(vy + vy_kick));
	    }
	}
      if (photon_angle_)
	addOutOfPlaneAngle(beam->getPhotonVector(i_slice).back(), particle, i_beam, photonEnergies[k], e_emitter, e_before, vx_kick, vy_kick);
      e_emitter -= photonEnergies[k];
      results_.add_gui_photon_energy(photonEnergies[k]);

      //     energy -= photonEnergies[k];
      //      const PHOTON& foton = beam->new_photon(photonEnergies[k],particle, i_slice);
//       if (photon_file_ != NULL) 
// 	{
// 	  if (i_beam != 1) 
// 	    {
// 	      PHOTON temp(foton);
// 	      float energyTemp = temp.energy();
// 	      temp.setEnergy(-energyTemp);
// 	      photon_file_->save_object_on_persistent_file(&temp);
// 	    }
// 	  else photon_file_->save_object_on_persistent_file(&foton);	  
// 	}
    }
}


// switch photon_angle and the formation-length limits of mode 2 (called before the movers of every step).
// At the first call with photon_angle != 0 the angle draws get their own stream, seeded from the main
// generator's next draws without advancing it: the main sequence stays that of photon_angle = 0, so that with
// pairs and Compton off all modes share every beam and photon history
void GRID::set_photon_angle(int photon_angle, float lf_max_1, float lf_max_2)
{
  photon_angle_ = photon_angle;
  photon_angle_lf_max_[0] = lf_max_1;
  photon_angle_lf_max_[1] = lf_max_2;
  if (photon_angle_ && !photon_angle_seeded_)
    {
      photon_angle_stream_ = qedbg::seed_from_copy(*rndm_generator_);
      photon_angle_seeded_ = true;
    }
}

// photon_angle >= 1: angle of a new photon out of its emitter's orbit plane, sampled from the LCFA spectrum
// (qedbg_photon_angle.h); the orbit plane is spanned by the emitter's direction and this step's kick
void GRID::addOutOfPlaneAngle(PHOTON& photon, PARTICLE& particle, int i_beam, double omega, double e_emitter, double e_before, float vx_kick, float vy_kick)
{
  // Upsilon of this step (at the energy before the step's photons) scales with the emitter energy
  const double upsilon = particle.getUps()*e_emitter/e_before;
  const double u = omega/e_emitter;
  if (!(upsilon > 0.0) || !(u > 0.0) || !(u < 1.0)) return;
  const double x0 = qedbg::PhotonAngleProfile::xi0(u, upsilon);
  const double gamma = e_emitter/EMASS;
  // photon_angle = 2: only where the LCFA holds, i.e. the formation length (radius of curvature times the
  // angular width, gamma lambda_c width/Upsilon) is shorter than half the opposing bunch length, the scale
  // on which the field seen by the emitter changes
  if (photon_angle_ == 2 &&
      gamma*HBAR*Cvelocity/EMASS*1e9/upsilon*qedbg::PhotonAngleProfile::width(x0) > photon_angle_lf_max_[i_beam-1])
    return;
  // out of the orbit plane: perpendicular to this step's transverse kick
  float vx, vy;
  particle.velocities(vx, vy);
  const double kx = vx - vx_kick, ky = vy - vy_kick, kick = std::sqrt(kx*kx + ky*ky);
  if (!(kick > 0.0)) return;
  const double r1 = qedbg::splitmix64_uniform(photon_angle_stream_);
  const double r2 = qedbg::splitmix64_uniform(photon_angle_stream_);
  const double r3 = qedbg::splitmix64_uniform(photon_angle_stream_);
  const double psi = qedbg::PhotonAngleProfile::instance().sample_t(u, upsilon, r1, r2, r3)/gamma;
  float pvx, pvy;
  photon.velocities(pvx, pvy);
  const double next_vx = pvx - psi*ky/kick;
  const double next_vy = pvy + psi*kx/kick;
  qedbg::validate_photon_angle(psi, next_vx, next_vy);
  photon.setVelocities(next_vx, next_vy);
}

void GRID::beamstrahlungSingleCoherentParticle(PARTICLE* particle, TRIVECTOR EBfield, float dzOnRadius, const std::vector<GENERAL_GRID*>& /*grids*/, int i_beam,  int i_slice, const PHI_FLOAT */*phi*/, float /*distance*/, float emin,int do_prod, int /*extra_grids*/, float /*charge_sign*/, float vx_kick, float vy_kick)
{	
  float upsilon = particle->getUps();

  float energy = 0.0;
  float initialEnergy = fabs(particle->energy());
  results_.updateUpsmax(upsilon);

  std::vector<float> photonEnergies;
  std::vector<float> photonPol;
  std::vector<float>* pol = photon_polarisation_ ? &photonPol : NULL;
  TRIDVECTOR ev1, ev2, ev3; 
  energy = PHYSTOOLS::synrad_no_spin_flip(upsilon, initialEnergy, dzOnRadius,photonEnergies, *rndm_generator_, pol);
  registerPhotons(photonEnergies, *(particle), i_beam, i_slice, vx_kick, vy_kick, initialEnergy, pol, EBfield(0), EBfield(1));
  if (energy < emin)
    {
      std::cout << "PARTICLE:: e_low2: " << energy << std::endl;
      energy = emin;
    }
  if ((do_prod>1)&&( i_beam ==1))
    {
      std::cerr << " GRID:: do_prod not implemented "  << std::endl;
      exit(1);
    }
  if (particle->energy() < 0.0)
    {
      particle->setEnergy( -energy);
    }
  else
    {
      particle->setEnergy(energy);
    }	  
    
}

void GRID::move_coherent_particles(const std::vector<GENERAL_GRID*>& grids,  int i_beam, int i_slice, int interpolation, int do_beamstrahlung, float emin, int do_prod, int extra_grids,float charge_sign)
{
  unsigned int k; 
  //  float initialEnergyForLoss;
  const PHI_FLOAT *phi;

  //  BEAM* beam;

  if (i_beam==1) 
    {
      //     beam = slice_of_beam_[0].get_beam();
      //      phi=phi2_;  
      phi = champ_.get_phi(2);
    }
  else
    { 
      //     beam = slice_of_beam_[1].get_beam();
      //     phi=phi1_;
      phi = champ_.get_phi(1);
    }
  const std::vector<PARTICLE*>& coherent = slice_of_beam_[i_beam-1].get_beam()->getCoherentVector(i_slice);

  switch (interpolation)
    {
    case 1:
      std::cerr << " GRID::move_coherent_particles interpolation = 1 is not implemented " << std::endl;
      exit(1);
      break;
    case 2:
      {
	for (k = 0; k < coherent.size(); k++)
	  {

	    TRIVECTOR EBfield = EBfieldOnParticle(coherent[k], grids, phi, i_beam, extra_grids);
	    float vx_kick, vy_kick;   // emitter direction before the kick (photon_timing = 3)
	    coherent[k]->velocities(vx_kick, vy_kick);
	    float dzOnRadius = coherent[k]->advanceDueToEBfield(EBfield, step_,  slice_of_beam_[i_beam-1].get_scal_step());
	    // (no spin rotation for coherent particles, nor sokolov ternov spin flip)
	    if(do_beamstrahlung)
	      {
		beamstrahlungSingleCoherentParticle(coherent[k], EBfield, dzOnRadius,grids, i_beam, i_slice, phi, step_, emin, do_prod, extra_grids, charge_sign, vx_kick, vy_kick);	
	      }  
	  }      
	break;
      }
    default:
      std::cerr << " GRID::move_coherent_particles : unknown value of interpolation type :  " << interpolation << std::endl;
      exit(1);
    }
}

void GRID::move_trident_particles(const std::vector<GENERAL_GRID*>& grids,  int i_beam, int i_slice, int interpolation, int do_beamstrahlung, float emin, int do_prod, int extra_grids,float charge_sign)
{
  unsigned int k; 
  const PHI_FLOAT *phi;
  if (i_beam==1) 
    {
      phi = champ_.get_phi(2);
    }
  else
    { 
      phi = champ_.get_phi(1);
    }
  const std::vector<PARTICLE*>& trident = slice_of_beam_[i_beam-1].get_beam()->getTridentVector(i_slice);

  switch (interpolation)
    {
    case 1:
      std::cerr << " GRID::move_trident_particles interpolation = 1 is not implemented " << std::endl;
      exit(1);
      break;
    case 2:
      {
	for (k = 0; k < trident.size(); k++)
	  {
	    TRIVECTOR EBfield = EBfieldOnParticle(trident[k], grids, phi, i_beam, extra_grids);
	    float vx_kick, vy_kick;   // emitter direction before the kick (photon_timing = 3)
	    trident[k]->velocities(vx_kick, vy_kick);
	    float dzOnRadius = trident[k]->advanceDueToEBfield(EBfield, step_, slice_of_beam_[i_beam-1].get_scal_step());
	    /* This routine includer transverse momentum rotation due to non-transversality if the fields */
	    //	    float dzOnRadius = trident[k]->advanceTridentDueToEBfield(EBfield, step_, slice_of_beam_[i_beam-1].get_scal_step());
	    // (no spin rotation for coherent particles, nor sokolov ternov spin flip)
	    if(do_beamstrahlung)
	      {
		beamstrahlungSingleCoherentParticle(trident[k], EBfield, dzOnRadius,grids, i_beam, i_slice, phi, step_, emin, do_prod, extra_grids, charge_sign, vx_kick, vy_kick);	
	      }  
	  }      
	break;
      }
    default:
      std::cerr << " GRID::move_trident_particles : unknown value of interpolation type :  " << interpolation << std::endl;
      exit(1);
    }
}


/* Distributes the photons for background calculation */

void GRID::distribute_photons(int slice_1,int slice_2, float photon_ratio, RNDM& rndm_generator)
{
  //int k;
  float ratio,ratio_i_1,ratio_i_2;
  const float eps=1e-5;

  //  std::cout << " GRID::distribute_photons " << std::endl;

  ratio = photon_ratio;
  clear_photon_pointer();
 
  if (ratio<eps) return;

  float deltax = mesh_.get_delta_x();
  float deltay = mesh_.get_delta_y();

  ratio_i_1=1e9/ratio*slice_of_beam_[0].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);
  ratio_i_2=1e9/ratio*slice_of_beam_[1].get_nb_part_per_macro()/sqrt(deltax*deltay*timestep_);

  distributePhotonInBeam(grid_photon1_, photon_cells1_, 1, slice_1,
                         ratio, ratio_i_1, rndm_generator);
  distributePhotonInBeam(grid_photon2_, photon_cells2_, 2, slice_2,
                         ratio, ratio_i_2, rndm_generator);
  photon_cells1_.sort_cells();
  photon_cells2_.sort_cells();

}

//void GRID::distributePhotonInBeam(   std::vector< std::vector<BEAM_PARTICLE_POINTER> >& grid_photon, BEAM* beam, int slice,float ratio, float ratio_i, RNDM& rndm_generator)
void GRID::distributePhotonInBeam(
    std::vector<std::vector<BEAM_PHOTON_POINTER> >& grid_photon,
    feebl::sparse_cells::ActiveCellSet& active_cells,
    int i_beam, int slice, float ratio, float ratio_i,
    RNDM& rndm_generator)
{
  float xphot, yphot;
  int i1, i2, j;
   

  float temp;

  const std::vector<PHOTON>& thePhotons =  slice_of_beam_[i_beam-1].get_beam()->getPhotonVector(slice);
  // photon_timing = 1, 3: a photon born during this step exists, on average, for half of it
  const std::size_t first_newborn = (photon_timing_ == 1 || photon_timing_ == 3) ? photon_birth_[i_beam-1] : thePhotons.size();
  unsigned int k;
  for(k=0; k < thePhotons.size(); k++)
    {
      temp = rndm_generator.rndm();
      if (temp<ratio)
	{
	  thePhotons[k].XYposition(xphot, yphot);

	  if (particleInGrid(xphot, yphot, i1, i2))
	    {
	      j = i1*n_cell_y_ + i2;
	      active_cells.touch(j);
	      grid_photon[j].push_back(BEAM_PHOTON_POINTER(&thePhotons[k], (k < first_newborn) ? ratio_i : 0.5f*ratio_i));

		
	    }
	}
    }
}

/*! This routine calculates the electron-photon, positron-photon and
   photon-photon luminosities */

feebl::legacy_photon_lumi::PhotonSample GRID::load_photon_lumi_sample(
    const void* context, std::size_t index)
{
  const std::vector<BEAM_PHOTON_POINTER>& samples =
      *static_cast<const std::vector<BEAM_PHOTON_POINTER>*>(context);
  const BEAM_PHOTON_POINTER& sample = samples[index];
  const feebl::legacy_photon_lumi::PhotonSample value = {
      sample.energy(), sample.weight()};
  return value;
}

feebl::legacy_photon_lumi::ChargedSample GRID::load_charged_lumi_sample(
    const void* context, std::size_t index)
{
  const std::vector<BEAM_PARTICLE_POINTER>& samples =
      *static_cast<const std::vector<BEAM_PARTICLE_POINTER>*>(context);
  const BEAM_PARTICLE_POINTER& sample = samples[index];
  const feebl::legacy_photon_lumi::ChargedSample value = {
      sample.energy(), sample.weight()};
  return value;
}

void GRID::photon_lumi(float min_z, SWITCHES& switches, PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator)
{

  if (switches.get_photon_lumi_factorized() == 1)
    {
      // Recheck the immutable deck contract at the dispatch boundary.  This
      // is intentionally redundant with SWITCHES::check_consistency: a future
      // mutable switch or alternate caller must not accidentally enter a
      // luminosity-only kernel with hidden per-pair effects enabled.
      feebl::legacy_photon_lumi::Features features;
      features.do_photon_lumi = switches.get_do_photon_lumi();
      features.do_lumi = switches.get_do_lumi();
      features.do_jets = switches.get_do_jets();
      features.do_compt = switches.get_do_compt();
      features.do_compt_phot = switches.get_do_compt_phot();
      features.do_pairs = switches.get_do_pairs();
      features.do_muons = switches.get_do_muons();
      features.do_hadrons = switches.get_do_hadrons();
      features.do_coherent = switches.get_do_coherent();
      features.do_trident = switches.get_do_trident();
      features.do_cross = switches.get_do_cross();
      features.do_bhabhas = switches.get_do_bhabhas();
      features.do_prod = switches.get_do_prod();
      features.do_tertphot = switches.get_do_tertphot();
      std::string reason;
      if (!feebl::legacy_photon_lumi::eligible(features, &reason))
        {
          throw std::runtime_error(
              std::string("factorized photon-luminosity dispatch lost its ") +
              "side-effect-free capability: " + reason +
              ". No fallback was performed.");
        }

      typedef feebl::legacy_photon_lumi::Result FastResult;
      FastResult aggregate;

      try
        {
          if (photon_lumi_workspace_ == NULL)
            photon_lumi_workspace_ =
                new feebl::legacy_photon_lumi::Workspace;

          // The threshold algorithm queries beam 1 and sorts beam 2.  Reserve
          // both largest possible cell occupancies once, before the cell loop,
          // so the persistent GRID-owned workspace never allocates in a cell.
          std::size_t maximum_query_photons = 0u;
          std::size_t maximum_sorted_photons = 0u;
          if (!sparse_cells_enabled_)
            {
              for (std::size_t cell = 0; cell < grid_photon2_.size(); ++cell)
                {
                  maximum_query_photons = std::max(
                      maximum_query_photons, grid_photon1_[cell].size());
                  maximum_sorted_photons = std::max(
                      maximum_sorted_photons, grid_photon2_[cell].size());
                }
            }
          else
            {
              const std::vector<int>* active_sets[] = {
                  &photon_cells1_.cells(), &photon_cells2_.cells(),
                  &particle_cells1_.cells(), &particle_cells2_.cells()};
              feebl::sparse_cells::sorted_union(active_sets, 4U,
                                                &sparse_traversal_cells_);
              for (std::vector<int>::const_iterator active =
                       sparse_traversal_cells_.begin();
                   active != sparse_traversal_cells_.end(); ++active)
                {
                  const std::size_t cell = static_cast<std::size_t>(*active);
                  maximum_query_photons = std::max(
                      maximum_query_photons, grid_photon1_[cell].size());
                  maximum_sorted_photons = std::max(
                      maximum_sorted_photons, grid_photon2_[cell].size());
                }
            }
          photon_lumi_workspace_->reserve(maximum_query_photons,
                                          maximum_sorted_photons);

          if (!sparse_cells_enabled_)
            {
              for (int i1=0; i1<n_cell_x_; ++i1)
                {
                  for (int i2=0; i2<n_cell_y_; ++i2)
                    {
                      const int j=i1*n_cell_y_+i2;
                      const feebl::legacy_photon_lumi::PhotonView photons1(
                          &grid_photon1_[j], grid_photon1_[j].size(),
                          &GRID::load_photon_lumi_sample);
                      const feebl::legacy_photon_lumi::PhotonView photons2(
                          &grid_photon2_[j], grid_photon2_[j].size(),
                          &GRID::load_photon_lumi_sample);
                      const feebl::legacy_photon_lumi::ChargedView particles1(
                          &part_pointer1_[j], part_pointer1_[j].size(),
                          &GRID::load_charged_lumi_sample);
                      const feebl::legacy_photon_lumi::ChargedView particles2(
                          &part_pointer2_[j], part_pointer2_[j].size(),
                          &GRID::load_charged_lumi_sample);

                      feebl::legacy_photon_lumi::accumulate_cell(
                          photons1, particles2, photons2, particles1,
                          switches.get_gg_smin(),
                          feebl::legacy_photon_lumi::Algorithm::FactoredExact,
                          *photon_lumi_workspace_, &aggregate);
                    }
                }
            }
          else
            {
              sparse_dense_traversal_visits_ +=
                  2ULL*static_cast<unsigned long long>(n_cell_x_)*n_cell_y_;
              sparse_actual_traversal_visits_ +=
                  2ULL*static_cast<unsigned long long>(
                      sparse_traversal_cells_.size());
              for (std::vector<int>::const_iterator active =
                       sparse_traversal_cells_.begin();
                   active != sparse_traversal_cells_.end(); ++active)
                {
                  const int j = *active;
                  const feebl::legacy_photon_lumi::PhotonView photons1(
                      &grid_photon1_[j], grid_photon1_[j].size(),
                      &GRID::load_photon_lumi_sample);
                  const feebl::legacy_photon_lumi::PhotonView photons2(
                      &grid_photon2_[j], grid_photon2_[j].size(),
                      &GRID::load_photon_lumi_sample);
                  const feebl::legacy_photon_lumi::ChargedView particles1(
                      &part_pointer1_[j], part_pointer1_[j].size(),
                      &GRID::load_charged_lumi_sample);
                  const feebl::legacy_photon_lumi::ChargedView particles2(
                      &part_pointer2_[j], part_pointer2_[j].size(),
                      &GRID::load_charged_lumi_sample);
                  feebl::legacy_photon_lumi::accumulate_cell(
                      photons1, particles2, photons2, particles1,
                      switches.get_gg_smin(),
                      feebl::legacy_photon_lumi::Algorithm::FactoredExact,
                      *photon_lumi_workspace_, &aggregate);
                }
            }
        }
      catch (const std::exception& error)
        {
          throw std::runtime_error(
              std::string("factorized photon-luminosity validation failed ") +
              "before result publication: " + error.what() +
              ". No fallback was performed.");
        }

      const unsigned long long maximum =
          std::numeric_limits<unsigned long long>::max();
      if (aggregate.ge_pair_opportunities >
              maximum - aggregate.eg_pair_opportunities ||
          aggregate.ge_pair_opportunities +
                  aggregate.eg_pair_opportunities >
              maximum - aggregate.gg_pair_opportunities)
        {
          throw std::overflow_error(
              "factorized photon-luminosity logical pair total overflow");
        }
      const unsigned long long aggregate_total =
          aggregate.ge_pair_opportunities + aggregate.eg_pair_opportunities +
          aggregate.gg_pair_opportunities;
      if (photon_lumi_factorized_calls_ == maximum ||
          photon_lumi_ge_opportunities_ >
              maximum - aggregate.ge_pair_opportunities ||
          photon_lumi_eg_opportunities_ >
              maximum - aggregate.eg_pair_opportunities ||
          photon_lumi_gg_opportunities_ >
              maximum - aggregate.gg_pair_opportunities ||
          photon_lumi_total_opportunities_ > maximum - aggregate_total ||
          photon_lumi_cartesian_pairs_visited_ >
              maximum - aggregate.cartesian_pairs_visited)
        {
          throw std::overflow_error(
              "factorized photon-luminosity provenance counter overflow");
        }

      // Transactional publication: malformed inputs and provenance overflow
      // cannot leave a partially accumulated RESULTS object.  The same
      // sampled macro weights produced by distribute_photons and
      // distribute_particles_for_background are used; those routines and
      // their RNG order are unchanged.
      photon_lumi_factorized_seen_ = true;
      ++photon_lumi_factorized_calls_;
      photon_lumi_ge_opportunities_ += aggregate.ge_pair_opportunities;
      photon_lumi_eg_opportunities_ += aggregate.eg_pair_opportunities;
      photon_lumi_gg_opportunities_ += aggregate.gg_pair_opportunities;
      photon_lumi_total_opportunities_ += aggregate_total;
      photon_lumi_cartesian_pairs_visited_ +=
          aggregate.cartesian_pairs_visited;
      for (int first=0; first<3; ++first)
        for (int second=0; second<3; ++second)
          results_.add_lumi(first, second,
                            aggregate.class_luminosity[3*first+second]);
      results_.add_lumi_ge(aggregate.lumi_ge);
      results_.add_lumi_eg(aggregate.lumi_eg);
      results_.add_lumi_gg(aggregate.lumi_gg);
      results_.add_lumi_gg_high(aggregate.lumi_gg_high);
      return;
    }

  ++photon_lumi_cartesian_calls_;

  int i1,i2,j,n_x,n_y;
  //  std::vector<BEAM_PARTICLE_POINTER>::iterator photon_pointer, photon_pointer1,photon_pointer2;
  std::vector<BEAM_PHOTON_POINTER>::const_iterator photon_pointer; 
  std::vector<BEAM_PHOTON_POINTER>::const_iterator photon_pointer1,photon_pointer2;
  std::vector<BEAM_PARTICLE_POINTER>::const_iterator particle_pointer;
  n_x = n_cell_x_;
  n_y = n_cell_y_;

  const PAIR_PARAMETER& pair_parameter = secondaries.get_pair_parameters();

  for (i1=0;i1<n_x;i1++)
    {
      for (i2=0;i2<n_y;i2++)
	{
	  j=i1*n_y+i2;
	  for (photon_pointer=grid_photon1_[j].begin();photon_pointer != grid_photon1_[j].end(); photon_pointer++)
	    {
	      for (particle_pointer = part_pointer2_[j].begin(); particle_pointer != part_pointer2_[j].end();  particle_pointer++)
		{
		  collide_ge(i1, i2, min_z,*photon_pointer,*particle_pointer, secondaries, switches, pair_parameter, rndm_generator);
		}
	    }
	  
	  for (photon_pointer=grid_photon2_[j].begin(); photon_pointer!= grid_photon2_[j].end(); photon_pointer++)
	    {
	      for (particle_pointer= part_pointer1_[j].begin(); particle_pointer != part_pointer1_[j].end(); particle_pointer++)
		{
		  collide_eg( i1, i2, min_z,*particle_pointer, *photon_pointer, secondaries, switches,pair_parameter,rndm_generator);		  
		}
	    }
	  
	  
	  double returnCrossSectionToAdd[3];
	  double tempor[3];
	  int k;
	  for (k=0; k<3;k++) {
	    tempor[k]=0.0;
	    returnCrossSectionToAdd[k] = 0.0;
	  }
	  
	  for ( photon_pointer1 = grid_photon1_[j].begin(); photon_pointer1 != grid_photon1_[j].end(); photon_pointer1++)
	    {
	      for( photon_pointer2 = grid_photon2_[j].begin(); photon_pointer2 != grid_photon2_[j].end(); photon_pointer2++)
		{
		  collide_gg(i1, i2, min_z,*photon_pointer1, *photon_pointer2, secondaries, muons, switches, rndm_generator, returnCrossSectionToAdd);
		  for (k=0; k<3;k++) tempor[k] += returnCrossSectionToAdd[k];
		}
	      
	    }	  
	  results_. cumulate_hadrons_gg(tempor);
	}
    }
}


void GRID::collide_ge(int cellx, int celly,float min_z, const BEAM_PHOTON_POINTER& photon_pointer, const BEAM_PARTICLE_POINTER& particle_pointer, PAIR_BEAM& secondaries, SWITCHES& switches, const PAIR_PARAMETER& pair_parameter, RNDM& rndm_generator)
{
   float photonEnergy = photon_pointer.energy();
  if (photonEnergy <= 0.0) return; 

  float particleEnergy = particle_pointer.energy(); 
  int j2=1;
  if (particleEnergy <0.0 ) j2=0;

   float weight = photon_pointer.weight()*particle_pointer.weight(); 

  results_.add_lumi(2, j2, weight);

  if (switches.get_do_lumi()&2)
    {
      lumi_heap_ge_.lumi_store(mesh_, cellx, celly,min_z, photonEnergy,particleEnergy, weight);
    }

  results_.add_lumi_ge(weight);

  if (switches.get_do_jets())
    minijets_->mkjbh1_(pair_parameter, photonEnergy,particleEnergy,weight, switches, rndm_generator);
  // previously collide_ge_2
  collide_compton(0, cellx, celly,min_z, photon_pointer, particle_pointer, secondaries, switches,rndm_generator, 1);		    
}

void GRID::collide_eg(int cellx, int celly,float min_z, const BEAM_PARTICLE_POINTER& particle_pointer, const BEAM_PHOTON_POINTER& photon_pointer, PAIR_BEAM& secondaries, SWITCHES& switches, const PAIR_PARAMETER& pair_parameter, RNDM& rndm_generator)
{
  float photonEnergy = photon_pointer.energy();
  if (photonEnergy <= 0.0) return;

  float particleEnergy = particle_pointer.energy();  
  int j1=0;
  if (particleEnergy < 0.0) j1=1;
 
  float weight = photon_pointer.weight()*particle_pointer.weight();

  results_.add_lumi(j1, 2, weight);

  if (switches.get_do_lumi()&2)
    {
      lumi_heap_eg_.lumi_store(mesh_, cellx, celly,min_z,particleEnergy,photonEnergy, weight);
    }

  results_.add_lumi_eg(weight);

  if (switches.get_do_jets())
    minijets_->mkjbh2_(pair_parameter, particleEnergy,photonEnergy,weight, switches, rndm_generator);
  // previously collide_eg_2 
  collide_compton(0,cellx, celly, min_z, photon_pointer, particle_pointer, secondaries, switches, rndm_generator, 2);		    
}

void GRID::collide_gg(int cellx, int celly, float min_z, const BEAM_PHOTON_POINTER& photon_pointer1, const BEAM_PHOTON_POINTER& photon_pointer2, PAIR_BEAM& secondaries ,PAIR_BEAM& muons, SWITCHES& switches, RNDM& rndm_generator, double *returnCrossSectionToAdd)
{
  JET_FLOAT ecm;
  //double beta_x,beta_y;
  // float particleVx1,particleVy1,particleVx2,particleVy2 ;
  float photonEnergy1 = photon_pointer1.energy();
  float photonEnergy2 = photon_pointer2.energy();
  float weight  = photon_pointer1.weight()*photon_pointer2.weight();

  const double s_gg =
      (4.0 * static_cast<double>(photonEnergy1)) *
      static_cast<double>(photonEnergy2);
  ecm=sqrt(s_gg);

  results_.add_lumi(2, 2, weight);
  if (ecm<=0.0) {
    return;
  }

  if (switches.get_do_lumi()&4){

    lumi_heap_gg_.lumi_store(mesh_, cellx, celly,min_z,photonEnergy1, photonEnergy2, weight);
  }
  results_.add_lumi_gg(weight);
  if (feebl::legacy_photon_lumi::gg_above_threshold(
          photonEnergy1, photonEnergy2, switches.get_gg_smin()))
    {
      results_.add_lumi_gg_high(weight);
    }
  if (switches.get_do_jets())
    minijets_->mkjbw_(photonEnergy1,photonEnergy2,weight, switches, rndm_generator );

  collide_gg_XX(0,cellx, celly, min_z,photon_pointer1, photon_pointer2, switches, secondaries, muons, rndm_generator, returnCrossSectionToAdd);
}

//void GRID::collide_gg_XX(int index_of_process, int cellx, int celly,float min_z, const PARTICLE_POINTER& photon1, const PARTICLE_POINTER& photon2, SWITCHES& switches, PAIR_BEAM& secondaries, RNDM& rndm_generator, double returnCrossSectionToAdd[3])
void GRID::collide_gg_XX(int index_of_process, int cellx, int celly,float min_z, const PHOTON_POINTER& photon1, const PHOTON_POINTER& photon2, SWITCHES& switches, PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator, double *returnCrossSectionToAdd)
{
  double beta_x,beta_y;
  float particle1Vx,particle1Vy,particle2Vx,particle2Vy ;
  float phot1_q2, phot1_eorg;
  float phot2_q2, phot2_eorg;
  float weight;
  float energy1 = photon1.energy();
  float energy2 = photon2.energy();
  if (energy1*energy2 <= 0.0) return;

  photon1.velocities(particle1Vx,particle1Vy);
  photon2.velocities(particle2Vx,particle2Vy);
  beta_x=(particle1Vx*energy1 + particle2Vx*energy2)/(energy1+energy2);
  beta_y=(particle1Vy*energy1 + particle2Vy*energy2)/(energy1+energy2);

  phot1_q2 =  photon1.q2();
  phot2_q2 =  photon2.q2();
  phot1_eorg = photon1.eorg();
  phot2_eorg = photon2.eorg();

  weight  = photon1.weight()*photon2.weight();

  // photon_polarisation=1: lab-frame Stokes parameters of the two photons (zero for virtual photons)
  float stokes1[3], stokes2[3];
  const float* st1 = NULL;
  const float* st2 = NULL;
  if (switches.get_photon_polarisation())
    {
      photon1.stokes(stokes1);
      photon2.stokes(stokes2);
      st1 = stokes1;
      st2 = stokes2;
    }

  // pair_model = 2: exact lowest-order kernels whenever a virtual photon takes part (BH, LL)
  const bool exact_pairs = (switches.get_pair_model() == 2) && (index_of_process > 0);
  if (switches.get_do_pairs())
    {
      if (exact_pairs)
	secondaries.make_pair_exact(mesh_, cellx, celly,min_z,index_of_process, energy1,phot1_q2,phot1_eorg,energy2,phot2_q2,phot2_eorg, weight,beta_x,beta_y, false, switches,rndm_generator, st1, st2);
      else
	secondaries. make_pair_bw(mesh_, cellx, celly,min_z,index_of_process, energy1,phot1_q2,phot1_eorg,energy2,phot2_q2,phot2_eorg, weight,beta_x,beta_y, switches,rndm_generator, st1, st2);
    }
  if (switches.get_do_muons())
    {
      if (exact_pairs)
	muons.make_pair_exact(mesh_, cellx, celly,min_z,index_of_process, energy1,phot1_q2,phot1_eorg,energy2,phot2_q2,phot2_eorg, weight,beta_x,beta_y, true, switches,rndm_generator, st1, st2);
      else
	muons.make_muon(mesh_, cellx, celly,min_z, index_of_process, energy1,phot1_q2,phot1_eorg,energy2,phot2_q2,phot2_eorg,weight,beta_x,beta_y, switches, rndm_generator, st1, st2);
    }
  if (switches.get_do_hadrons()) make_hadrons_gg2(min_z, energy1,phot1_q2,energy2,phot2_q2,weight, switches, returnCrossSectionToAdd, rndm_generator);
}

/**********************************************************************/
/* Routines for the creation and storage of hadrons                   */
/**********************************************************************/

void GRID::make_hadrons_gg2(float min_z, float energy1,float q2_1,float energy2,float q2_2,float lumi, SWITCHES& switches, double *returnCrossSectionToAdd, RNDM& rndm_generator)
{
  double s,h;
  double cross_section=0.0;
  int num,k;
  double eps=0.0808,mu=0.4525;

  for (k=0; k<3; k++) returnCrossSectionToAdd[k] = 0.0;

  s=energy1*energy2;
  h=std::max(1.0,pow(s/100.0,0.43));
  if ((q2_1>h)||(q2_2>h)) {
    return;
  }
  s*=4.0;
  if (s<4.0) return;
  switch (switches.get_do_hadrons())
    {
    case 1:
      cross_section=(211.0*pow(s,eps)+297.0*pow(s,-mu))*1e-37*lumi;
      break;
    case 2:
      cross_section=lumi*200e-37*(1.0+6.3e-3*pow(log(s),2.1)+1.96*pow(s,-0.37));
      break;
    case 3:
      cross_section=(211.0*pow(s,eps)+215.0*pow(s,-mu))*1e-37*lumi;
      break;
    case 4:
      cross_section=(-0.99244+0.0989203*log(s+22865.9))*1e-34*lumi;
      break;
    }
  if (hadron_file_ != NULL)
    {
      h=cross_section*switches.get_hadron_ratio();
      num=(int)floor(h);
      h-=num;
      if(h>rndm_generator.rndm()) num++;
      saveOnHadronFile(num, energy1, energy2,min_z, rndm_generator, switches.get_do_edm4hep());
  }
  returnCrossSectionToAdd[0] = cross_section;
  if (s<25.0) return;
  returnCrossSectionToAdd[1] = cross_section;
  if (s<100.0) return;
  returnCrossSectionToAdd[2] = cross_section;
}

/*! This routine calculates the background using the virtual photons with the
   apropriate impact parameter. In the moment it just does pair creation. */

void GRID::photon_lumi_2(float min_z,SWITCHES& switches, PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator)
{
  int i1,i2,j,n_x,n_y;
  std::vector<BEAM_PHOTON_POINTER>::iterator photon_pointer;
  //double beta_x,beta_y,e1,e2;
  n_x = n_cell_x_;
  n_y = n_cell_y_;
  for (i1=0;i1<n_x;i1++)
    {
      for (i2=0;i2<n_y;i2++)
	{
	  j=i1*n_y+i2;

	  double returnCrossSectionToAdd[3];
	  double tempor[3];
	  int k;
	  for (k=0; k<3;k++) {
	    tempor[k]=0.0;
	    returnCrossSectionToAdd[k]=0.0;
	  }


	  std::vector<EXTRA_PHOTON_POINTER>::const_iterator extr_phot;
	  //float extr_phot_e, extr_phot_weight, extr_phot_q2, extr_phot_eorg;
	  for (photon_pointer=grid_photon1_[j].begin(); photon_pointer != grid_photon1_[j].end(); photon_pointer++)
	    {
	      const std::vector<EXTRA_PHOTON_POINTER>& extr_phot_list = extra_photon_pointer2_[j];
	      
	      for (extr_phot = extr_phot_list.begin() ; extr_phot!=extr_phot_list.end(); extr_phot++)
		{

		  collide_gg_XX(1, i1,i2,min_z, *photon_pointer, *extr_phot,switches, secondaries, muons, rndm_generator, returnCrossSectionToAdd);
		  for (k=0; k<3;k++) tempor[k] += returnCrossSectionToAdd[k];
		}
	    }

	  results_. cumulate_hadrons_ge(tempor);
	  for (k=0; k<3;k++) {
	    tempor[k]=0.0;
	    returnCrossSectionToAdd[k]=0.0;
	  }
	  for( photon_pointer=grid_photon2_[j].begin(); photon_pointer!= grid_photon2_[j].end(); photon_pointer++)
	    {
	      const std::vector<EXTRA_PHOTON_POINTER>& extr_phot_list = extra_photon_pointer1_[j];
	      for ( extr_phot = extr_phot_list.begin(); extr_phot != extr_phot_list.end(); extr_phot++)
		{
		  collide_gg_XX(1,i1, i2,min_z,  *extr_phot, *photon_pointer,switches,secondaries, muons, rndm_generator, returnCrossSectionToAdd);
		  for (k=0; k<3;k++) tempor[k] += returnCrossSectionToAdd[k];
		}
	    }
	  results_. cumulate_hadrons_eg(tempor);
	  for (k=0; k<3;k++) {
	    tempor[k]=0.0;
	    returnCrossSectionToAdd[k]=0.0;
	  }

	  const std::vector<EXTRA_PHOTON_POINTER>& extr_phot1_list = extra_photon_pointer1_[j];
	  std::vector<EXTRA_PHOTON_POINTER>::const_iterator extr_phot1;
	  std::vector<EXTRA_PHOTON_POINTER>::const_iterator extr_phot2;
	  for( extr_phot1 = extr_phot1_list.begin(); extr_phot1!=extr_phot1_list.end(); extr_phot1++)
	    {

	      const std::vector<EXTRA_PHOTON_POINTER>& extr_phot2_list = extra_photon_pointer2_[j];

	      for ( extr_phot2 = extr_phot2_list.begin(); extr_phot2!=extr_phot2_list.end(); extr_phot2++)
		{

		  collide_gg_XX(2, i1, i2,min_z, *extr_phot1, *extr_phot2, switches,secondaries, muons, rndm_generator, returnCrossSectionToAdd);
		  for (k=0; k<3;k++) tempor[k] += returnCrossSectionToAdd[k];
		}
	    }
	  results_. cumulate_hadrons_ee(tempor);
	}
    }
}


void GRID::photon_lumi_3(float min_z,SWITCHES& switches, PAIR_BEAM& secondaries,  RNDM& rndm_generator)
{
  int i1,i2,j,n_x,n_y;

  std::vector<BEAM_PARTICLE_POINTER>::iterator particle_pointer;

  std::vector<EXTRA_PHOTON_POINTER>::const_iterator extr_phot1, extr_phot2;

  n_x = n_cell_x_;
  n_y = n_cell_y_;
  for (i1=0;i1<n_x;i1++)
    {
      for (i2=0;i2<n_y;i2++)
	{
	  j=i1*n_y+i2;
	  const std::vector<EXTRA_PHOTON_POINTER>& extr_phot1_list = extra_photon_pointer1_[j];

	  for( extr_phot1 = extr_phot1_list.begin(); extr_phot1!=extr_phot1_list.end(); extr_phot1++)
	    {
	      for ( particle_pointer = part_pointer2_[j].begin(); particle_pointer != part_pointer2_[j].end(); particle_pointer++)
		{
		  collide_compton(1,i1, i2,min_z, *extr_phot1, *particle_pointer, secondaries, switches, rndm_generator, 1);
		}
	    }
	  
	  const std::vector<EXTRA_PHOTON_POINTER>& extr_phot2_list = extra_photon_pointer2_[j];

	  for (extr_phot2 = extr_phot2_list.begin(); extr_phot2!=extr_phot2_list.end(); extr_phot2++)
	    {
	      for (particle_pointer = part_pointer1_[j].begin(); particle_pointer != part_pointer1_[j].end(); particle_pointer++)
		{

		  collide_compton(1,i1, i2,min_z, *extr_phot2, *particle_pointer, secondaries, switches, rndm_generator,2);
		}
	    }
	}
    }
}


void GRID::move_photons2(BEAM& beam,int ibeam,int i_slice, RNDM& rndm_generator)
{
  float ax,ay,wgt;
  PHI_FLOAT phi1_x,phi2_x,phi3_x,phi1_y,phi2_y,phi3_y,h_x,h_y;
  float radius_i;
  const PHI_FLOAT *phi;
  PHI_FLOAT upsilon,upsilon0,tmp;
  PHI_FLOAT x,y;
  int i, nd;


  float scal_step_local;
  //  scal_step_local = scal_step[ibeam-1];
  scal_step_local = slice_of_beam_[ibeam-1].get_scal_step();
  if (ibeam==1){
    //    phi=phi2_;
    phi = champ_.get_phi(2);
    wgt= slice_of_beam_[0].get_nb_part_per_macro();
  }
  else{
    //   phi=phi1_;
    phi = champ_.get_phi(1);
    wgt= slice_of_beam_[1].get_nb_part_per_macro();
  }
  std::vector<PHOTON>& thePhotons =  beam.getPhotonVector(i_slice);
  // photon_timing = 1, 3: a photon born during this step converts, on average, over half of it
  const std::size_t first_newborn = (photon_timing_ == 1 || photon_timing_ == 3) ? photon_birth_[ibeam-1] : thePhotons.size();
  unsigned int k;
  for(k=0; k < thePhotons.size(); k++)
    {
      thePhotons[k].XYposition(x, y);
      if ( coordinatesInGridRange(x, y) )
	{ 
	  interpolePotential(x, y, h_x, h_y, phi1_x, phi2_x, phi3_x, phi1_y, phi2_y, phi3_y, phi);
	  ax = (h_x*(phi1_y-phi2_y)+(1.0-h_x)*(phi2_y-phi3_y))*delta_x_inverse_;
	  ay = (h_y*(phi1_x-phi2_x)+(1.0-h_y)*(phi2_x-phi3_x))*delta_y_inverse_;
	}
      else
	{
	  ax = 0.0;
	  ay = 0.0;
	}
      float factor = step_*scal_step_local;
      //      photItr->move(factor);
      thePhotons[k].advancePosition(factor);

      radius_i = sqrt(ax*ax+ay*ay)*1e9/scal_step_local;
      
      upsilon0=LAMBDA_BAR/EMASS*radius_i;
      //     upsilon=upsilon0*photItr->get_energy()/EMASS;
      upsilon=upsilon0*thePhotons[k].energy()/EMASS;
      if (thePhotons[k].energy()>0.0) results_.add_gui_chi_gamma(upsilon);
      
      if (upsilon>1e-10)
	{
	  tmp=ALPHA_EM*ALPHA_EM/RE*step_*1e-9*scal_step_local*upsilon0*0.23*exp(-8.0/(3.0*upsilon));
	  tmp*=pow(1.0+0.22*upsilon,-1.0/3.0);
	  if (k >= first_newborn) tmp*=0.5;
	  coherent_results_.updateProbmax(tmp);
	  // photon_polarisation=1: the pair rate depends on the photon's linear polarisation tau along the field
	  // (ax, ay) as 1 + tau A(upsilon); the envelope is widened by 1 + |tau|/2 and pick_coherent_energy accepts
	  // the polarised energy split (qedbg_photon_stokes.h)
	  const PHI_FLOAT p_rate = tmp;   // probability of the rate formula, before the envelope factor 1.36
	  float tau = 0.0f;
	  nd=1;   // sub-steps of this step (set below for tmp >= 0.1), used by the survival update
	  if (photon_polarisation_)
	    {
	      float stokes[3];
	      thePhotons[k].getStokes(stokes);
	      tau = (float)qedbg::stokes_along(stokes, ax, ay);
	    }
	  tmp*=1.36;
	  if (tau != 0.0f) tmp *= 1.0 + 0.5*fabs(tau);
	  // tmp >= 0.1: sampled in nd sub-steps below.  The legacy sampling (coherent_model = 0) also gates these
	  // steps by rndm() < tmp, which applies the probability twice for 0.1 <= tmp < 1; coherent_model = 1 drops
	  // that gate.  The random number is drawn in both, so a run without a step with 0.1 <= tmp < 1 is the same
	  // in both.
	  if (rndm_generator.rndm()<tmp || (coherent_model_ == 1 && tmp>=0.1)) {
	    if (tmp<0.1) {
	      if (coherent_generate(beam,i_slice,upsilon,thePhotons[k], rndm_generator, tau)) {
		coherent_results_.updateSumeng(wgt*thePhotons[k].energy());
		thePhotons[k].razEnergy();//photon->energy=0.0;
		tmp=1.0;
		//nyes++;
	      }
	      else {
		tmp=0.0;
		//nno++;
	      }
	    }
	    else {
	      nd=(int)(tmp/0.1+1.0);
	      float ds=tmp/nd;
	      tmp=0.0;
	      for (i=0;i<nd;++i) {
		if (rndm_generator.rndm()<ds) {
		  if (coherent_generate(beam,i_slice,upsilon,thePhotons[k], rndm_generator, tau)) {
		    coherent_results_.updateSumeng(wgt*thePhotons[k].energy());
		    thePhotons[k].razEnergy();//photon->energy=0.0;
		    tmp=1.0;
		    //nyes++;
		    i=nd;
		  }
		}
	      }
	    }
	  }
	  else {
	    tmp=0.0;
	  }
	  /*
	    if (tmp>1.0)
	    {
	    //	      coherent_results.updateSumeng(wgt*photItr->get_energy());
	    coherent_results_.updateSumeng(wgt*thePhotons[k].energy());
	    coherent_generate(beam,i_slice,upsilon,thePhotons[k], rndm_generator);
	    //	      photItr->razEnergy();
	    thePhotons[k].razEnergy();
	    tmp=1.0;
	    }
	    else
	    {
	    if (rndm_generator.rndm()<tmp*1.36) 
	    {
	    coherent_generate(beam,i_slice,upsilon,thePhotons[k], rndm_generator);
	    //		  photItr->razEnergy();
	    thePhotons[k].razEnergy();
	    }
	    }
	    // end test
	  */ 
	  // photon_polarisation=1: a photon that did not convert in this step is more likely to be in the
	  // less-converting polarisation (along the field); its Stokes parameters are updated with the conversion
	  // probability that this sampler realises, p_rate * R(upsilon), in nd sub-steps (dichroism; no birefringence)
	  if (photon_polarisation_ && thePhotons[k].energy() > 0.0)
	    {
	      float stokes[3];
	      thePhotons[k].getStokes(stokes);
	      qedbg::stokes_survival_update(stokes, ax, ay, p_rate*qedbg::coherent_pair_rate_ratio(upsilon),
					    qedbg::coherent_pair_asymmetry(upsilon), nd);
	      thePhotons[k].setStokes(stokes);
	    }
	  tmp*=wgt;
	  //	  coherent_results.updateSums(tmp, photItr->get_energy(), upsilon0);
	  coherent_results_.updateSums(tmp, thePhotons[k].energy(), upsilon0);
	}
      //      photItr++;
    }
}




//void GRID::coherent_generate(BEAM& beam,int i_slice, float upsilon,PHOTON& phot, RNDM& rndm_generator)
int GRID::coherent_generate(BEAM& beam,int i_slice, float upsilon,PHOTON& phot, RNDM& rndm_generator, float tau)
{
  float x,y;
  float vx,vy;
  float energy;
  energy =  phot.energy();
  // number adjustment to be done for total cross section
  if (pick_coherent_energy(upsilon,energy, rndm_generator, tau))
    {
      phot.XYposition(x, y);
      phot.velocities(vx,vy);
      store_coherent_particle(beam,energy,vx,vy,x,y,i_slice);
      store_coherent_particle(beam,-(phot.energy()-energy),vx,vy,x,y,
		     i_slice);
      return 1;
    }
  return 0;
}

bool GRID::pick_coherent_energy(float ups,float& energy, RNDM& rndm_generator, float tau)
{
  float a=0.13513,eta,dxdy,x,h,hp,tmp,tmp2,y,coh;
  y=2.0*rndm_generator.rndm()-1.0;
  eta=8.0/(3.0*ups);
  tmp=fabs(y/(1.0+(1.0-y*y)/(2.0*sqrt(eta))));
  tmp2=1.0-tmp*tmp;
  h=-log(tmp2);
  hp=tmp/tmp2*(1.0+(1.0+y*y)/(2.0*sqrt(eta)))
    /(1.0+(1.0-y*y)/(2.0*sqrt(eta)))
    /(1.0+(1.0-y*y)/(2.0*sqrt(eta)));
  if (y>0.0){
    x=0.5*(1.0+sqrt(h/(eta+h)));
  }
  else{
    x=0.5*(1.0-sqrt(h/(eta+h)));
  }
  dxdy=hp/sqrt(h*(h+eta))*(1.0-h/(h+eta));
  coh=PHYSTOOLS::u(ups);
  tmp=PHYSTOOLS::fcp(ups,x)*a/coh*dxdy;
  if (rndm_generator.rndm()<tmp){
    // photon_polarisation=1: polarised split dW/dx ~ fcp - tau K23, accepted with probability
    // (1 - tau K23/fcp)/(1 + |tau|/2); move_photons2 widened the envelope by 1 + |tau|/2
    if (tau != 0.0f &&
        rndm_generator.rndm() >= qedbg::coherent_pair_split_acceptance(tau, PHYSTOOLS::fcp(ups,x), PHYSTOOLS::fcp_k23(ups,x)))
      return false;
     energy *= x;
    return true;
  } else {
    return false;
  }
}


void GRID::step_pair_1(const std::vector<GENERAL_GRID*>& grids, PAIR_PARTICLE& pair, double mass,float step,int nbSteps, int extra_grids, float charge_sign_0, RNDM& rndm_generator)
{
  float thetamax;
  float ex,ey,bx,by;
  int i;
  // initial half step 
 
  // retrieve E and B fields
  field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);
  thetamax = 2.0*pair.apply_initial_half_step_fields(step, mass, ex,ey, bx, by, rndm_generator);
  /* loop over steps */
  for(i=1;i<nbSteps;i++)
    {
      pair.advancePosition(step);

      // retrieve E and B fields
      field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);    
      thetamax = std::max (thetamax, pair.apply_full_step_fields(step, mass, ex,ey, bx, by, rndm_generator)); 
    }
  /* last half step */
  pair.advancePosition(step);
  field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);
  thetamax = std::max( thetamax, (float)2.0*pair.apply_final_half_step_fields(step, mass, ex,ey, bx, by, thetamax,rndm_generator));
  if (	!pair.last_rescaling_ok() )
    {
      std::cerr << " GRID::step_pair_1() : " << " thetamax " << thetamax << std::endl;
    }	  
}

void GRID::step_pair_1_tertphot(const std::vector<GENERAL_GRID*>& grids, PAIR_PARTICLE& pair, double mass,float step,int nbSteps, int extra_grids, float charge_sign_0, RNDM& rndm_generator)
{
  float thetamax;
  float ex,ey,bx,by;
  int i;
  unsigned int j;
  std::vector<float> photon_e;
  // initial half step 
 
  // retrieve E and B fields
  field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);
  thetamax = 2.0*pair.apply_initial_half_step_fields(step, mass, ex,ey, bx, by, &photon_e, rndm_generator);
  /* loop over steps */
  for(i=1;i<nbSteps;i++)
    {
      pair.advancePosition(step);

      // retrieve E and B fields
      field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);    
      thetamax = std::max (thetamax, pair.apply_full_step_fields(step, mass, ex,ey, bx, by, &photon_e, rndm_generator)); 
    }
  /* last half step */
  pair.advancePosition(step);
  field_pair(pair, grids,ex,ey,bx,by, extra_grids, charge_sign_0);
  thetamax = std::max( thetamax, (float)2.0*pair.apply_final_half_step_fields(step, mass, ex,ey, bx, by, thetamax,rndm_generator));
  for(j=0;j<photon_e.size();j++)
    {
      tertphot_.push_back(TERTPHOTON(photon_e[j],pair));
    }
  if (	!pair.last_rescaling_ok() )
    {
      std::cerr << " GRID::step_pair_1_tertphot() : " << " thetamax " << thetamax << std::endl;
    }	  
}

int GRID::field_pair(const PAIR_PARTICLE& pair, const std::vector<GENERAL_GRID*>& grids,float& ex,float& ey, float& bx,float& by, int extra_grids, float charge_sign_0)
{
  PHI_FLOAT ax_1,ay_1,ax_2,ay_2;
  PHI_FLOAT tmp,dx,dy;
  float Ex1, Ey1,Ex2, Ey2;
  float x,y;
  int i;
  TRIVECTOR Efield1, Efield2;

  pair.XYposition(x, y);
  for(i=0;i<=extra_grids;i++)
    {
      if (grids[i]->coordinatesInGridRange(x,y))
	{
	  Efield1 = grids[i]->ElectricFieldCIC(x, y, grids[i]->get_phi1());
	  Efield2 = grids[i]->ElectricFieldCIC(x, y, grids[i]->get_phi2());

	  Efield1 *= -charge_sign_0;

	  Ex1 = Efield1(0);
	  Ey1 = Efield1(1);
	  Ex2 = Efield2(0);
	  Ey2 = Efield2(1);
	  ex = -(Ex2-Ex1);
	  ey = -(Ey2-Ey1);
	  by =  (Ex2 + Ex1);
	  bx = -(Ey2 + Ey1);
          pair.record_imported_field_gather(i);

	  return i;
	}
    }
/* particle is not in largest grid */
  dx=(x-grids[0]->get_rho_x_1());
  dy=(y-grids[0]->get_rho_y_1());
  tmp=grids[0]->get_rho_factor()*grids[0]->get_rho_sum_1()/(dx*dx+dy*dy);
  tmp*=-charge_sign_0;

/*  tmp=grids[0].rho_factor*grids[0].rho_sum_1/(dx*dx+dy*dy);*/
  ax_1=dx*tmp;
  ay_1=dy*tmp;
  dx=(x-grids[0]->get_rho_x_2());
  dy=(y-grids[0]->get_rho_y_2());
  tmp=grids[0]->get_rho_factor()*grids[0]->get_rho_sum_2()/(dx*dx+dy*dy);
  ax_2=dx*tmp;
  ay_2=dy*tmp;

  ex = -0.5*(ax_2-ax_1);
  ey = -0.5*(ay_2-ay_1);
  by = 0.5*(ax_2+ax_1);
  bx = -0.5*(ay_2+ay_1);
  pair.record_imported_field_gather(-1);
  return -1;
}



EXTRA_GRID::~EXTRA_GRID()
{
}

//same as distribute_particles0 but without counting particles as being on or off the grid 

void EXTRA_GRID::distribute_particles(int i_slice1,
				   int i_slice2,
				   int electron_distribution_rho, 
				   int force_symmetric)
{

  slice_of_beam_[0].razRho();
  slice_of_beam_[1].razRho();

  //  razRhos(); 
  switch (electron_distribution_rho)
    {
    case 1:

//       assignBeamSliceNGP(beam1_, i_slice1, rho1_, nb_part_per_macro_[0]);
//       assignBeamSliceNGP(beam2_, i_slice2, rho2_, nb_part_per_macro_[1]);
      assignBeamSliceNGP(slice_of_beam_[0], i_slice1);
      assignBeamSliceNGP(slice_of_beam_[1], i_slice2);

      break;
    case 2:

//        assignBeamSliceCIC(beam1_, i_slice1, rho1_, nb_part_per_macro_[0]);
//       assignBeamSliceCIC(beam2_, i_slice2, rho2_, nb_part_per_macro_[1]);
      assignBeamSliceCIC(slice_of_beam_[0], i_slice1);
      assignBeamSliceCIC(slice_of_beam_[1], i_slice2);
      break;
    }
  distribute_coherent_particles(i_slice1, i_slice2, electron_distribution_rho);
  distribute_trident_particles(i_slice1, i_slice2, electron_distribution_rho);
  if (force_symmetric)
    {
      slice_of_beam_[0].symmetrizeCharges(n_cell_x_,n_cell_y_);
      slice_of_beam_[1].symmetrizeCharges(n_cell_x_,n_cell_y_);
    }

    // symmetrizeCharges();
}


//  Distribute the coherent pair particles for the calculation of the fields
//  but does not count if they are on the grid or not (this routine is used for
//  the larger grids)
 

void EXTRA_GRID::distribute_coherent_particles(int i_slice1,
					    int i_slice2,
					    int electron_distribution_rho)
{
  switch (electron_distribution_rho)
    {
    case 1:


//       assignCoherentBeamSliceNGP(beam1_, i_slice1, rho1_, nb_part_per_macro_[0]);
//       assignCoherentBeamSliceNGP(beam2_, i_slice2, rho2_, nb_part_per_macro_[1]);
      assignCoherentBeamSliceNGP(slice_of_beam_[0], i_slice1);
      assignCoherentBeamSliceNGP(slice_of_beam_[1], i_slice2);
      break;
    case 2:


      assignCoherentBeamSliceCIC(slice_of_beam_[0], i_slice1);
      assignCoherentBeamSliceCIC(slice_of_beam_[1], i_slice2);
      break;
    }
}

void EXTRA_GRID::distribute_trident_particles(int i_slice1,
					    int i_slice2,
					    int electron_distribution_rho)
{
  switch (electron_distribution_rho)
    {
    case 1:
      assignTridentBeamSliceNGP(slice_of_beam_[0], i_slice1);
      assignTridentBeamSliceNGP(slice_of_beam_[1], i_slice2);
      break;
    case 2:
      assignTridentBeamSliceCIC(slice_of_beam_[0], i_slice1);
      assignTridentBeamSliceCIC(slice_of_beam_[1], i_slice2);
      break;
    }
}

// void EXTRA_GRID::assignCoherentBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro)
// {
//   int i1, i2;
//   float xpart, ypart;  
//   float ch;
//   unsigned int particle;
//   const std::vector<PARTICLE*>& coherent = beam->getCoherentVector(i_slice);
//   for (particle = 0; particle < coherent.size(); particle++)
//     {
//       coherent[particle]->XYposition(xpart, ypart);
//       if (coherent[particle]->energy() < 0.0) ch=-1.0;
//       else ch=1.0;
//       if (particleInGrid(xpart, ypart,i1, i2)) assignChargeToNGP(rho, i1, i2, n_macro*ch);	    
//     }
// }
void EXTRA_GRID::assignCoherentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  unsigned int particle;
  const std::vector<PARTICLE*>& coherent = sog.get_beam()->getCoherentVector(i_slice);
  for (particle = 0; particle < coherent.size(); particle++)
    {
      coherent[particle]->XYposition(xpart, ypart);
      if (coherent[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      if (particleInGrid(xpart, ypart,i1, i2)) sog.assignChargeToNGP(i1, i2,ch);	    
    }
}
void EXTRA_GRID::assignTridentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  unsigned int particle;
  const std::vector<PARTICLE*>& trident = sog.get_beam()->getTridentVector(i_slice);
  for (particle = 0; particle < trident.size(); particle++)
    {
      trident[particle]->XYposition(xpart, ypart);
      if (trident[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      if (particleInGrid(xpart, ypart,i1, i2)) sog.assignChargeToNGP(i1, i2,ch);	    
    }
}
// void EXTRA_GRID::assignBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro)
// {
//   int k, i1, i2;
//   float xpart, ypart;
//     std::vector<PARTICLE*>& theParticles = beam->getParticleVector(i_slice);
//     for (k = 0; k < theParticles.size(); k++)
//    {
// 	theParticles[k]->XYposition(xpart, ypart);
//       if (particleInGrid(xpart, ypart, i1, i2)) assignChargeToNGP(rho, i1, i2, n_macro);
//     }
// }
void EXTRA_GRID::assignBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice)
{
  unsigned int k;
  int i1, i2;
  float xpart, ypart;
    const std::vector<PARTICLE*>& theParticles = sog.get_beam()->getParticleVector(i_slice);
    for (k = 0; k < theParticles.size(); k++)
   {
	theParticles[k]->XYposition(xpart, ypart);
      if (particleInGrid(xpart, ypart, i1, i2)) sog.assignChargeToNGP(i1, i2,1.0);
    }
}
// void EXTRA_GRID::assignBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro)
// {
//   int k, i1, i2;
//   float h_x,h_y;
//   float xpart, ypart;
//     std::vector<PARTICLE*>& theParticles = beam->getParticleVector(i_slice);
//     for (k = 0; k < theParticles.size(); k++)
// 	{
// 	theParticles[k]->XYposition(xpart, ypart);
// 	  if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2)) assignChargeToCIC(rho, i1, i2, h_x, h_y, n_macro);	    	  
// 	}
// }
void EXTRA_GRID::assignBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice)
{
  unsigned int k;
  int i1, i2;
  float h_x,h_y;
  float xpart, ypart;
  const std::vector<PARTICLE*>& theParticles = sog.get_beam()->getParticleVector(i_slice);
  for (k = 0; k < theParticles.size(); k++)
    {
      theParticles[k]->XYposition(xpart, ypart);
      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2)) sog.assignChargeToCIC(i1, i2, h_x, h_y, 1.0);	    	  
    }
}
// void EXTRA_GRID::assignCoherentBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro)
// {
//   int i1, i2;
//   float xpart, ypart;  
//   float ch;
//   float h_x,h_y;
//   unsigned int particle;
//   const std::vector<PARTICLE*>& coherent = beam->getCoherentVector(i_slice);
//   for (particle = 0; particle < coherent.size(); particle++)
//     {
//       coherent[particle]->XYposition(xpart, ypart);
//       if (coherent[particle]->energy() < 0.0) ch=-1.0;
//       else ch=1.0;
//       if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2)) assignChargeToCIC(rho, i1, i2, h_x, h_y, n_macro*ch);	
//     }
// }
void EXTRA_GRID::assignCoherentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  float h_x,h_y;
  unsigned int particle;
  const std::vector<PARTICLE*>& coherent = sog.get_beam()->getCoherentVector(i_slice);
  for (particle = 0; particle < coherent.size(); particle++)
    {
      coherent[particle]->XYposition(xpart, ypart);
      if (coherent[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2)) sog.assignChargeToCIC(i1, i2, h_x, h_y, ch);	
    }
}

void EXTRA_GRID::assignTridentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice)
{
  int i1, i2;
  float xpart, ypart;  
  float ch;
  float h_x,h_y;
  unsigned int particle;
  const std::vector<PARTICLE*>& trident = sog.get_beam()->getTridentVector(i_slice);
  for (particle = 0; particle < trident.size(); particle++)
    {
      trident[particle]->XYposition(xpart, ypart);
      if (trident[particle]->energy() < 0.0) ch=-1.0;
      else ch=1.0;
      if (particleInGrid(xpart, ypart, h_x, h_y, i1, i2)) sog.assignChargeToCIC(i1, i2, h_x, h_y, ch);	
    }
}
