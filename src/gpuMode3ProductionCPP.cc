#include "gpuMode3ProductionCPP.h"

#include "beamCPP.h"
#include "beamParametersCPP.h"
#include "gridCPP.h"
#include "legacyWavefrontSchedulerCPP.h"
#include "modernLegacyAdapterCPP.h"
#include "switchesCPP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace feebl {
namespace gpu {
namespace {

void reject_if(bool condition, const char* feature,
               std::vector<std::string>& rejected) {
  if (condition) rejected.push_back(feature);
}

void validate_production_card_impl(const SWITCHES& s) {
  std::vector<std::string> rejected;
  reject_if(s.get_time_order() != 2, "time_order!=2", rejected);
  reject_if(s.get_collision_schedule() != "wavefront_strict",
            "collision_schedule!=wavefront_strict", rejected);
  reject_if(s.get_integration_method() != 2, "integration_method!=2", rejected);
  reject_if(s.get_electron_distribution_rho() != 2,
            "electron_distribution_rho!=2", rejected);
  reject_if(s.get_interpolation() != 2, "interpolation!=2", rejected);
  reject_if(s.get_extra_grids() != 0, "grids!=1", rejected);
  reject_if(s.get_sparse_cells() != 0, "sparse_cells", rejected);
  // The resident component implements both mechanisms, but neither selector
  // has yet passed a production-card serial/GPU physics qualification.  Keep
  // the v1 production boundary fail-closed until that evidence exists.
  reject_if(s.get_force_symmetric() != 0, "force_symmetric", rejected);
  reject_if(s.get_charge_symmetric() != 0, "charge_symmetric", rejected);
  reject_if(s.get_load_beam() != 0, "load_beam", rejected);
  reject_if(s.get_ext_field() != 0, "ext_field", rejected);

  reject_if(s.get_do_beamstrahlung() != 0, "beamstrahlung/energy-loss", rejected);
  reject_if(s.get_do_photons(1) != 0 || s.get_do_photons(2) != 0,
            "photon emission", rejected);
  reject_if(s.get_load_photon() != 0, "load_photon", rejected);
  reject_if(s.get_write_photons() != 0, "store/write_photons", rejected);
  reject_if(s.get_do_photon_lumi() != 0, "photon luminosity", rejected);

  reject_if(s.get_do_lumi() != 0, "full electron luminosity", rejected);
  reject_if(s.get_store_native() != 0, "store_native collision records", rejected);
  reject_if(s.get_electron_lumi_precompute() != 0,
            "electron_lumi_precompute", rejected);
  reject_if(s.get_photon_lumi_factorized() != 0,
            "photon_lumi_factorized", rejected);

  reject_if(s.get_bmt_precession() != 0, "BMT spin transport", rejected);
  reject_if(s.get_ST_spin_flip() != 0, "Sokolov-Ternov spin flips", rejected);
  reject_if(s.get_do_bds_spin_rotation() != 0, "BDS spin rotation", rejected);

  reject_if(s.get_do_prod() != 0, "pair production", rejected);
  reject_if(s.get_do_coherent() != 0, "coherent secondaries", rejected);
  reject_if(s.get_do_trident() != 0, "trident secondaries", rejected);
  reject_if(s.get_do_pairs() != 0 || s.get_track_pairs() != 0 ||
                s.get_store_pairs() != 0,
            "pair generation/tracking/storage", rejected);
  reject_if(s.get_do_muons() != 0 || s.get_track_muons() != 0 ||
                s.get_store_muons() != 0,
            "muon generation/tracking/storage", rejected);
  reject_if(s.get_do_hadrons() != 0 || s.get_store_hadrons() != 0,
            "hadron generation/storage", rejected);
  reject_if(s.get_do_compt() != 0 || s.get_do_compt_phot() != 0,
            "Compton generation/storage", rejected);
  reject_if(s.get_do_tertphot() != 0, "tertiary photons", rejected);
  reject_if(s.get_do_bhabhas() != 0, "Bhabha events", rejected);
  reject_if(s.get_do_jets() != 0, "jets", rejected);
  reject_if(s.get_do_cross() != 0, "cross-section event sampling", rejected);
  reject_if(s.get_do_isr() != 0, "ISR event sampling", rejected);
  reject_if(s.get_load_event() != 0, "load_event", rejected);
  reject_if(s.get_do_dump() != 0, "collision beam dumps", rejected);
  reject_if(s.get_do_size_log() != 0, "per-diagonal size logging", rejected);

  if (!rejected.empty()) {
    std::ostringstream message;
    message << "gpu_mode=3 primary-resident production subset rejected:";
    for (std::size_t i = 0; i < rejected.size(); ++i)
      message << (i == 0 ? " " : ", ") << rejected[i];
    message << ". This mode currently implements generated primary CIC/FFT/"
               "gather/push/energy/Upsilon only; no fallback was performed.";
    throw std::invalid_argument(message.str());
  }
}

modern::LegacyBeamImportContract make_import_contract(
    const BEAM& beam,
    const BEAM_PARAMETERS& parameters,
    const GENERAL_GRID& grid,
    unsigned int label) {
  const int declared = grid.get_nb_macroparticles(static_cast<int>(label));
  if (declared <= 0)
    throw std::invalid_argument("gpu_mode=3 requires a positive declared n_m for both beams");

  modern::LegacyBeamImportContract contract;
  contract.bunch_population = static_cast<double>(parameters.n_particles());
  contract.macro_weight = grid.get_macro_charge(static_cast<int>(label));
  contract.declared_macro_particle_count =
      static_cast<modern::ParticleIndex>(declared);
  contract.beam_id = label;
  contract.direction = label == 1u ? std::int8_t{1} : std::int8_t{-1};
  contract.species = label == 1u
      ? modern::ParticleSpecies::Electron
      : modern::ParticleSpecies::Positron;
  contract.charge_e = label == 1u ? std::int8_t{-1} : std::int8_t{1};
  contract.maximum_total_particles = static_cast<modern::ParticleIndex>(
      beam.particle_beam().numberOfParticles());
  return contract;
}

void require_valid(const modern::ValidationResult& result,
                   const char* boundary) {
  if (!result) {
    std::ostringstream message;
    message << "gpu_mode=3 " << boundary << " failed";
    if (!result.message.empty()) message << ": " << result.message;
    throw std::runtime_error(message.str());
  }
}

std::uint64_t maximum_wave_width(const modern::EncounterDAG& dag) {
  std::uint64_t maximum = 0;
  for (std::size_t i = 0; i < dag.waves().size(); ++i)
    maximum = std::max(maximum,
        static_cast<std::uint64_t>(dag.waves()[i].encounters.size()));
  return maximum;
}

}  // namespace

void preflight_mode3_card_controls(const SWITCHES& switches) {
  validate_production_card_impl(switches);
}

Mode3ProductionReceipt execute_mode3_primary_crossing(
    BEAM& beam1,
    BEAM& beam2,
    const BEAM_PARAMETERS& parameters1,
    const BEAM_PARAMETERS& parameters2,
    GENERAL_GRID& grid,
    const SWITCHES& switches,
    const std::string& input_card_label,
    int cuda_device,
    double memory_fraction) {
  validate_production_card_impl(switches);
  if (input_card_label.empty())
    throw std::invalid_argument("gpu_mode=3 requires a nonempty input-card label");
  if (grid.get_n_cell_z() <= 0 || grid.get_timestep() <= 0)
    throw std::invalid_argument("gpu_mode=3 requires positive n_z and n_t");
  if (grid.get_fft_spectrum_size() < 0)
    throw std::overflow_error("gpu_mode=3 Green-spectrum length is negative");

  const modern::SliceIndex slices =
      static_cast<modern::SliceIndex>(grid.get_n_cell_z());
  const modern::SubstepIndex substeps =
      static_cast<modern::SubstepIndex>(grid.get_timestep());
  legacy_wavefront::Scheduler scheduler(slices, slices, substeps);

  const modern::LegacyBeamImportContract contract1 =
      make_import_contract(beam1, parameters1, grid, 1u);
  const modern::LegacyBeamImportContract contract2 =
      make_import_contract(beam2, parameters2, grid, 2u);
  modern::BeamSoA primary1(slices, beam1.particle_beam().has_spin());
  modern::BeamSoA primary2(slices, beam2.particle_beam().has_spin());
  primary1.reserve(beam1.particle_beam().numberOfParticles());
  primary2.reserve(beam2.particle_beam().numberOfParticles());
  const modern::LegacyBeamImportSpan span1 =
      modern::append_legacy_particle_beam_slices(
          beam1.particle_beam(), 0, slices, contract1, primary1);
  const modern::LegacyBeamImportSpan span2 =
      modern::append_legacy_particle_beam_slices(
          beam2.particle_beam(), 0, slices, contract2, primary2);

  // Bind the GPU work to the exact legacy source and normalization before
  // allocation/upload.  Values are copied from the live legacy float-backed
  // objects (rather than recomputed from the deck spelling) and are printed
  // in the production receipt, including the small representational residual
  // between bunch_population and macro_weight*n_m.
  require_valid(modern::compare_legacy_particle_beam_import(
                    beam1.particle_beam(), contract1, span1, primary1),
                "beam-1 import/normalization contract");
  require_valid(modern::compare_legacy_particle_beam_import(
                    beam2.particle_beam(), contract2, span2, primary2),
                "beam-2 import/normalization contract");
  require_valid(modern::preflight_legacy_crossing_writeback(
                    primary1, contract1, span1, beam1.particle_beam(),
                    primary2, contract2, span2, beam2.particle_beam()),
                "paired transactional write-back preflight");

  static_assert(sizeof(PHI_FLOAT) == sizeof(double),
                "mode3 Green spectrum requires FP64 PHI_FLOAT");
  Mode3PrimaryCrossingConfig config;
  config.requested_gpu_mode = 3;
  config.failure_policy = Mode3FailurePolicy::AbortNoFallback;
  config.precision = switches.get_precision();
  config.modern_reduction = switches.get_modern_reduction();
  config.resident_options.device = cuda_device;
  config.resident_options.pinned_io = true;
  config.resident_options.memory_fraction = memory_fraction;
  config.resident_options.contract.interpolation = switches.get_interpolation();
  config.resident_options.contract.extra_grids = switches.get_extra_grids();
  config.resident_options.contract.generated_primaries = true;
  config.resident_options.contract.beamstrahlung = false;
  config.resident_options.contract.photon_emission = false;
  config.resident_options.contract.spin_transport = false;
  config.resident_options.contract.sokolov_ternov = false;
  config.resident_options.contract.coherent_secondaries = false;
  config.resident_options.contract.trident_secondaries = false;
  config.resident_options.contract.tracked_pairs = false;
  config.resident_options.contract.tracked_muons = false;
  config.resident_options.contract.full_electron_luminosity = false;

  config.integration_method = switches.get_integration_method();
  config.nx = grid.get_n_cell_x();
  config.ny = grid.get_n_cell_y();
  config.green_spectrum_interleaved =
      reinterpret_cast<const double*>(grid.get_fft_spectrum());
  config.green_spectrum_interleaved_values =
      static_cast<std::size_t>(grid.get_fft_spectrum_size());
  config.green_spectrum_provenance_declared = true;
  config.spectrum_integration_method = switches.get_integration_method();
  config.spectrum_charge_sign = switches.get_charge_sign();
  config.grid.delta_x_inverse = grid.get_delta_x_inverse();
  config.grid.delta_y_inverse = grid.get_delta_y_inverse();
  config.grid.offset_x = grid.get_offset_x();
  config.grid.offset_y = grid.get_offset_y();
  config.grid.physical_min_x = grid.get_min_x();
  config.grid.physical_max_x = grid.get_max_x();
  config.grid.physical_min_y = grid.get_min_y();
  config.grid.physical_max_y = grid.get_max_y();
  config.spectrum_grid = config.grid;
  config.step = grid.get_step();
  config.scale1 = grid.get_scal_step(0);
  config.scale2 = grid.get_scal_step(1);
  config.charge_sign = switches.get_charge_sign();
  config.charge_symmetric_generation = switches.get_charge_symmetric() != 0;
  config.four_way_symmetry = switches.get_force_symmetric() != 0;
  config.input_card_label = input_card_label;
  config.crossing_index = 0;

  Mode3PrimaryCrossingAdapter adapter;
  Mode3PrimaryPreflight preflight;
  std::string error;
  if (!adapter.preflight(primary1, primary2, scheduler.dag(), config,
                         preflight, error))
    throw std::runtime_error("gpu_mode=3 structural preflight failed: " + error);

  Mode3ProductionReceipt receipt;
  receipt.import_contracts_bound_before_cuda = true;
  receipt.paired_writeback_preflight_before_cuda = true;
  receipt.bunch_population1 = contract1.bunch_population;
  receipt.bunch_population2 = contract2.bunch_population;
  receipt.macro_charge1 = contract1.macro_weight;
  receipt.macro_charge2 = contract2.macro_weight;
  receipt.declared_macro_particles1 = contract1.declared_macro_particle_count;
  receipt.declared_macro_particles2 = contract2.declared_macro_particle_count;
  receipt.effective_population1 = contract1.macro_weight *
      static_cast<double>(contract1.declared_macro_particle_count);
  receipt.effective_population2 = contract2.macro_weight *
      static_cast<double>(contract2.declared_macro_particle_count);
  receipt.initialization_rng = "legacy_sequential_generator";
  receipt.collision_rng = "unused_no_stochastic_collision_process";
  if (!adapter.execute(primary1, primary2, scheduler.dag(), config,
                       receipt.crossing, error))
    throw std::runtime_error("gpu_mode=3 resident crossing failed: " + error);

  const double fine_lumi_normalization =
      1.0e18 / (static_cast<double>(grid.get_delta_x()) *
                static_cast<double>(grid.get_delta_y()) *
                static_cast<double>(grid.get_timestep()));
  receipt.normalized_fine_lumi =
      receipt.crossing.resident.fine_lumi_dot_sum * fine_lumi_normalization;
  if (!std::isfinite(receipt.normalized_fine_lumi))
    throw std::runtime_error(
        "gpu_mode=3 normalized fine luminosity is non-finite");

  receipt.writeback = modern::write_back_legacy_crossing_phase_space(
      primary1, contract1, span1, beam1.particle_beam(),
      primary2, contract2, span2, beam2.particle_beam());
  if (!receipt.writeback.crossing_transaction_committed ||
      !receipt.writeback.beam1.upsilon_matches_beam_soa ||
      !receipt.writeback.beam2.upsilon_matches_beam_soa)
    throw std::logic_error(
        "gpu_mode=3 paired write-back receipt is incomplete");

  receipt.diagonals = scheduler.diagonals().size();
  receipt.waves = scheduler.dag().wave_count();
  receipt.encounters = scheduler.dag().encounter_count();
  receipt.maximum_wave_width = maximum_wave_width(scheduler.dag());
  return receipt;
}

}  // namespace gpu
}  // namespace feebl
