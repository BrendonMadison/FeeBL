#include "colliderFieldsCPP.h"
#include "guiSpectrumBinningCPP.h"
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <locale>
#include <map>
#include <new>
#include <stdexcept>
#include "energySpreadCPP.h"
#include "feeblCPP.h"
#include <sstream>
#include <fstream>
#include "incomingMapsCPP.h"
#include "option_args.h"
#include "parallelBackendCPP.h"
#include "gpuMode3ProductionCPP.h"
#include "adaptiveGridScheduleCPP.h"
#include "adaptiveMacroparticleControllerCPP.h"
#include "adaptiveRootReplaySignalCPP.h"
#ifdef FEEBL_HAVE_OPENMP
#include <omp.h>
#endif

namespace {

void gpu2_fatal(const std::string& message)
{
  std::cerr << " FEEBL GPU2 preflight ERROR: " << message << std::endl;
  exit(1);
}

std::uint64_t checked_size_to_u64(std::size_t value, const char* label)
{
  if (sizeof(std::size_t) > sizeof(std::uint64_t) &&
      value > static_cast<std::size_t>(
                  std::numeric_limits<std::uint64_t>::max()))
    throw std::overflow_error(std::string(label) + " exceeds uint64_t");
  return static_cast<std::uint64_t>(value);
}

void checked_add_u64(std::uint64_t& total, std::uint64_t increment,
                     const char* label)
{
  std::uint64_t updated = 0;
  if (!feebl::modern::checked_add(total, increment, updated))
    throw std::overflow_error(std::string(label) + " counter overflow");
  total = updated;
}

void emit_primary_upsilon_digest(const PARTICLE_BEAM& beam)
{
  const std::vector<std::vector<PRIMARY_PARTICLE_ID> >& id_slices =
      beam.getParticleIdSlicesStrict();
  const std::uint64_t fnv_offset = 1469598103934665603ull;
  const std::uint64_t fnv_prime = 1099511628211ull;
  std::uint64_t id_hash = fnv_offset;
  std::uint64_t count = 0;
  long double sum = 0.0L;
  long double sumsq = 0.0L;
  float maximum = 0.0f;
  std::uint64_t nonfinite = 0;
  for (int slice = 0; slice < beam.number_of_slices(); ++slice)
    {
      const std::vector<PARTICLE*>& particles = beam.getParticleVector(slice);
      const std::vector<PRIMARY_PARTICLE_ID>& ids = id_slices[slice];
      if (particles.size() != ids.size())
        throw std::logic_error("primary Upsilon digest found an ID/particle shape mismatch");
      for (std::size_t index = 0; index < particles.size(); ++index)
        {
          const std::uint64_t id = ids[index];
          for (unsigned int byte = 0; byte < 8; ++byte)
            {
              id_hash ^= static_cast<std::uint8_t>(id >> (8U * byte));
              id_hash *= fnv_prime;
            }
          const float upsilon = particles[index]->getUps();
          ++count;
          if (!std::isfinite(upsilon))
            {
              ++nonfinite;
              continue;
            }
          const long double value = static_cast<long double>(upsilon);
          sum += value;
          sumsq += value * value;
          maximum = std::max(maximum, upsilon);
        }
    }
  std::cerr << std::setprecision(17)
            << " FEEBL primary Upsilon digest: beam=" << beam.beam_label()
            << " count=" << count
            << " sum=" << static_cast<double>(sum)
            << " sumsq=" << static_cast<double>(sumsq)
            << " max=" << maximum
            << " nonfinite=" << nonfinite
            << " stable_id_hash=" << id_hash
            << std::endl;
}

void append_u64_little_endian(std::string& output, std::uint64_t value)
{
  for (unsigned int byte = 0; byte < 8U; ++byte)
    output.push_back(static_cast<char>((value >> (8U * byte)) & 0xffU));
}

std::string primary_particle_ordering_sha256(
    const PARTICLE_BEAM& beam1, const PARTICLE_BEAM& beam2)
{
  std::string canonical("FEEBL-primary-ID-ordering-v1\0", 29U);
  const PARTICLE_BEAM* beams[2] = {&beam1, &beam2};
  for (std::size_t beam_index = 0; beam_index < 2U; ++beam_index)
    {
      const std::vector<std::vector<PRIMARY_PARTICLE_ID> >& slices =
          beams[beam_index]->getParticleIdSlicesStrict();
      append_u64_little_endian(canonical,
                               static_cast<std::uint64_t>(beam_index + 1U));
      append_u64_little_endian(canonical,
                               static_cast<std::uint64_t>(slices.size()));
      for (const std::vector<PRIMARY_PARTICLE_ID>& ids : slices)
        {
          append_u64_little_endian(canonical,
                                   static_cast<std::uint64_t>(ids.size()));
          for (PRIMARY_PARTICLE_ID id : ids)
            append_u64_little_endian(canonical,
                                     static_cast<std::uint64_t>(id));
        }
    }
  return feebl::adaptive_grid::sha256_hex(canonical);
}

#ifdef FEEBL_HAVE_GPU5
std::uint64_t primary_slice_order_token(
    const std::vector<PRIMARY_PARTICLE_ID>& ids,
    std::uint64_t beam_label, std::uint64_t slice)
{
  // Stable FNV-1a over source IDs, with explicit beam/slice/count domains.
  // Coordinates are intentionally absent: mutation is carried by the
  // separate monotonic revision while this token binds vector order.
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto add = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8U; ++byte)
      {
        hash ^= (value >> (8U * byte)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
      }
  };
  add(UINT64_C(0x464545424c494431));  // "FEEBLID1"
  add(beam_label);
  add(slice);
  add(static_cast<std::uint64_t>(ids.size()));
  for (PRIMARY_PARTICLE_ID id : ids) add(static_cast<std::uint64_t>(id));
  return hash;
}
#endif

std::string adaptive_schedule_card_sha256(
    const SWITCHES& switches, const GRID& grid,
    const std::string& parameters_label, std::uint64_t lattice_hash)
{
  // This is the resolved, schedule-decision portion of the card.  Execution
  // selectors (CPU/GPU, thread count, batch width and save/load path) are
  // deliberately excluded so one exact geometry schedule can be replayed by
  // either backend.  The raw acc.dat hash is retained separately by the run
  // manifest; exact per-encounter plan hashes reject changed beam geometry.
  std::ostringstream canonical;
  canonical.imbue(std::locale::classic());
  canonical << "FEEBL-adaptive-schedule-card-v1\n"
            << "parameters_label=" << parameters_label << '\n'
            << "mode=" << switches.get_adaptive_grid_mode() << '\n'
            << "scope=" << switches.get_adaptive_grid_scope() << '\n'
            << "controller="
            << (switches.get_adaptive_grid_mode() == 1
                    ? "uniform-window-v1"
                    : (switches.get_adaptive_grid_mode() == 2
                           ? "error-controlled-resolution-v1"
                           : "nested-dirichlet-patch-v1"))
            << '\n'
            << "lattice_hash=" << lattice_hash << '\n'
            << "n_x=" << grid.get_n_cell_x() << '\n'
            << "n_y=" << grid.get_n_cell_y() << '\n'
            << "n_z=" << grid.get_n_cell_z() << '\n'
            << "classes_x="
            << switches.get_adaptive_grid_size_classes_x() << '\n'
            << "classes_y="
            << switches.get_adaptive_grid_size_classes_y() << '\n'
            << "min_nx=" << switches.get_adaptive_grid_min_nx() << '\n'
            << "min_ny=" << switches.get_adaptive_grid_min_ny() << '\n'
            << "max_nx=" << switches.get_adaptive_grid_max_nx() << '\n'
            << "max_ny=" << switches.get_adaptive_grid_max_ny() << '\n'
            << "guard="
            << switches.get_adaptive_grid_extra_guard_cells() << '\n'
            << "allow_full="
            << switches.get_adaptive_grid_allow_full_window() << '\n'
            << "max_cells=" << switches.get_adaptive_grid_max_cells() << '\n'
            << "max_bytes=" << switches.get_adaptive_grid_max_bytes() << '\n';
  if (switches.get_adaptive_grid_mode() >= 2)
    canonical << "spacing_factors="
              << switches.get_adaptive_grid_field_spacing_classes() << '\n'
              << "field_operator="
              << switches.get_adaptive_grid_field_operator() << '\n'
              << "max_refinement_factor="
              << switches.get_adaptive_grid_max_refinement_factor() << '\n'
              << "anchor_x=" << switches.get_adaptive_grid_anchor_x() << '\n'
              << "anchor_y=" << switches.get_adaptive_grid_anchor_y() << '\n'
              << "error_target="
              << switches.get_adaptive_grid_error_target() << '\n'
              << "error_norm_policy="
              << switches.get_adaptive_grid_error_norm_policy() << '\n'
              << "error_budget_sha256="
              << switches.get_adaptive_grid_error_budget_sha256() << '\n'
              << "error_quantile="
              << switches.get_adaptive_grid_error_quantile() << '\n'
              << "error_catastrophe_limit="
              << switches.get_adaptive_grid_error_catastrophe_limit() << '\n'
              << "error_reference_field="
              << switches.get_adaptive_grid_error_reference_field() << '\n'
              << "error_absolute_floor_fraction="
              << switches.get_adaptive_grid_error_absolute_floor_fraction()
              << '\n'
              << "error_catastrophe_absolute_fraction="
              << switches
                     .get_adaptive_grid_error_catastrophe_absolute_fraction()
              << '\n'
              << "error_exposure_limit="
              << switches.get_adaptive_grid_error_exposure_limit() << '\n'
              << "noise_aware_stop="
              << switches.get_adaptive_grid_noise_aware_stop() << '\n'
              << "tau_richardson="
              << switches.get_adaptive_grid_tau_richardson() << '\n'
              << "rho_floor=" << switches.get_adaptive_grid_rho_floor()
              << '\n'
              << "field_floor=" << switches.get_adaptive_grid_field_floor()
              << '\n'
              << "richardson_order_source="
              << switches.get_adaptive_grid_richardson_order_source() << '\n'
              << "richardson_order_table="
              << switches.get_adaptive_grid_richardson_order_table() << '\n'
              << "min_neff=" << switches.get_adaptive_grid_min_neff() << '\n';
  if (switches.get_adaptive_grid_mode() == 3)
    canonical << "refinement_ratio="
              << switches.get_adaptive_grid_refinement_ratio() << '\n'
              << "blocking_factor="
              << switches.get_adaptive_grid_blocking_factor() << '\n'
              << "error_buffer_cells="
              << switches.get_adaptive_grid_error_buffer_cells() << '\n'
              << "transition_cells="
              << switches.get_adaptive_grid_transition_cells() << '\n'
              << "fine_residual_tolerance="
              << switches.get_adaptive_grid_fine_residual_tolerance() << '\n'
              << "terminal_patch_certification="
              << switches.get_adaptive_grid_terminal_patch_certification()
              << '\n';
  if (!canonical)
    throw std::runtime_error(
        "could not serialize adaptive schedule card contract");
  return feebl::adaptive_grid::sha256_hex(canonical.str());
}

}  // namespace

FEEBL::FEEBL(char *name)
    : gpu2_direct_deposition_(false), gpu2_primary_scope_(false),
      gpu2_primary_beam1_in_(0), gpu2_primary_beam1_out_(0),
      gpu2_primary_beam2_in_(0), gpu2_primary_beam2_out_(0),
      gpu4_batch_width_(0), gpu4_maximum_particles1_(0),
      gpu4_maximum_particles2_(0),
#ifdef FEEBL_HAVE_GPU5
      gpu5_batch_capacity_(0),
#endif
      wavefront_field_workers_(0), wavefront_workspace_bytes_each_(0),
      wavefront_workspace_bytes_reserved_(0),
      wavefront_field_setup_ms_(0.0),
      adaptive_root_replay_required_(false),
      adaptive_root_replay_controller_observations_(0U),
      adaptive_root_replay_trace_hash_(0U), cross_(NULL),
      number_of_stored_particles1_(0), number_of_stored_particles2_(0)
{
  time_counter_ = 0;
  std::cout << header();
  init_input_file_names();
  beam_parameters1_.setLabel('1');
  beam_parameters2_.setLabel('2');
  secondaries_.set_name(std::string("pairs"));
  muons_.set_name(std::string("muons"));
  params_.read_accelerator(name); 
}

void FEEBL::emit_adaptive_root_replay_signal(
    bool publication_blocked, bool final_summary_outputs_published) const
{
  if (publication_blocked == final_summary_outputs_published)
    throw std::logic_error(
        "adaptive root replay signal has an inconsistent final-summary "
        "publication disposition");
  const auto require_token = [](const std::string& value,
                                const char* label) {
    if (value.empty() ||
        value.find_first_of("= \t\r\n") != std::string::npos)
      throw std::logic_error(
          std::string("adaptive root replay signal has an invalid ") +
          label + " token");
  };
  require_token(adaptive_root_replay_reason_, "reason");
  require_token(adaptive_root_replay_recommended_action_,
                "recommended-action");

  const int n_m1 = grid_.get_nb_macroparticles(1);
  const int n_m2 = grid_.get_nb_macroparticles(2);
  std::ostringstream signal;
  signal
      << feebl::adaptive_particles::kRootReplaySignalPrefix
      << " schema="
      << feebl::adaptive_particles::kRootReplaySignalSchema
      << " request_schema="
      << feebl::adaptive_particles::kRootReplayRequestSchema
      << " exit_code="
      << (publication_blocked
              ? feebl::adaptive_particles::kRootReplayRequiredExitCode
              : 0)
      << " publication_blocked=" << (publication_blocked ? 1 : 0)
      << " exploratory_outputs_authorized="
      << (publication_blocked ? 0 : 1)
      << " final_physics_outputs_published="
      << (final_summary_outputs_published ? 1 : 0)
      << " publication_scope=final_summary_outputs_only_v1"
      << " pre_gate_diagnostic_artifacts_may_exist=1"
      << " pre_gate_artifacts_physics_qualification_claim=0"
      << " preexisting_artifacts_not_removed=1"
      << " fresh_run_directory_required=1"
      << " final_polarization_sample_complete=1"
      << " bpm_finalization_complete=1"
      << " global_luminosity_publication_complete=1"
      << " complete_collision=1"
      << " physics_qualification_claim=0"
      << " recommended_action="
      << adaptive_root_replay_recommended_action_
      << " reason=" << adaptive_root_replay_reason_
      << " generator_id="
      << feebl::adaptive_particles::kRootReplayGeneratorId
      << " normalization_contract="
      << feebl::adaptive_particles::kRootReplayNormalizationId
      << " rndm_seed=" << switches.get_rndm_seed()
      << " n_m_beam1=" << n_m1
      << " n_m_beam2=" << n_m2
      << " common_n_m=" << (n_m1 == n_m2 ? 1 : 0)
      << " independent_complete_runs=1"
      << " minimum_complete_runs="
      << feebl::adaptive_particles::kRootReplayMinimumCompleteRuns
      << " controller_observations="
      << adaptive_root_replay_controller_observations_
      << " controller_trace_hash="
      << adaptive_root_replay_trace_hash_
      << " roots_added_online=0"
      << " fresh_complete_process_required=1";
  // Build the complete record before the single-threaded write so readers
  // never observe a controller handoff assembled across several log lines.
  std::cerr << signal.str() << std::endl;
}

// Validate backend capabilities before allocating beams or consuming random draws.
// Failure is fatal; this stage does not silently select another algorithm.
void FEEBL::configure_compute_backend()
{
  try
    {
      PARALLEL_BACKEND& backend = parallel_backend();
      GPU2_OPTIONS gpu2;
      gpu2.mode = switches.get_gpu_mode();
      gpu2.scope = switches.get_gpu2_scope();
      gpu2.deposition = switches.get_gpu2_deposition();
      gpu2.fft = switches.get_gpu2_fft();
      gpu2.streams = switches.get_gpu2_streams();
      gpu2.graphs = switches.get_gpu2_graphs();
      gpu2.fusion = switches.get_gpu2_fusion();
      gpu2.batch_slices = switches.get_gpu2_batch_slices();
      gpu2.pinned_io = switches.get_gpu2_pinned_io();
      gpu2.host_threads = switches.get_gpu2_host_threads();
      gpu2.primary_mover = switches.get_gpu2_primary_mover();
      gpu2.memory_fraction = switches.get_gpu2_memory_fraction();
      gpu2.profile = switches.get_gpu2_profile();
      MODERN_RUNTIME_OPTIONS modern;
      modern.engine = switches.get_engine();
      modern.algorithm = switches.get_algorithm();
      modern.precision = switches.get_precision();
      modern.reduction = switches.get_modern_reduction();
      modern.batch_width = switches.get_modern_batch_width();
      modern.cpu_workers = switches.get_modern_cpu_workers();
      modern.cpu_runtime = switches.get_modern_cpu_runtime();
      modern.cpu_affinity = switches.get_modern_cpu_affinity();
      modern.gpu_streams = switches.get_modern_gpu_streams();
      modern.gpu_graphs = switches.get_modern_gpu_graphs();
      modern.trace = switches.get_modern_trace();
      backend.configure(switches.get_compute_backend(),
                        switches.get_cpu_threads(),
                        switches.get_cpu_variant(),
                        switches.get_gpu_variant(),
                        switches.get_gpu_device(),
                        switches.get_gpu_min_particles(),
                        switches.get_rng_backend(),
                        switches.get_reduction_mode(),
                        switches.get_allow_backend_fallback() != 0,
                        gpu2,
                        modern);
      switches.set_compute_provenance(backend.actual_backend(),
                                      backend.actual_cpu_variant(),
                                      backend.actual_cpu_threads(),
                                      backend.fallback_reason());
      if (backend.gpu_mode3_enabled())
        {
          if (get_make_gui_files())
            throw std::runtime_error("--make_gui_files is unsupported for resident gpu_mode=3: no committed intermediate host snapshots");
          feebl::gpu::preflight_mode3_card_controls(switches);
          std::cerr
              << " FEEBL: GPU mode3 card preflight committed before beam "
                 "generation: generated-primary-no-radiation-no-full-lumi-v1, "
                 "fallback=forbidden, initialization_rng=legacy_sequential, "
                 "mode3_collision_rng=unused"
              << std::endl;
        }
      if (switches.get_cpu_primary_move() &&
          backend.requested_gpu2_options().primary_mover != "serial")
        throw std::runtime_error(
            "cpu_primary_move and gpu2_primary_mover are separate, "
            "non-composable selectors; leave gpu2_primary_mover=serial on "
            "CPU primary-move cards");
      if (switches.get_cpu_primary_move())
        {
          if (!backend.cpu_enabled() ||
              (backend.actual_cpu_variant() != "pool_private" &&
               backend.actual_cpu_variant() != "pool_dual"))
            throw std::runtime_error(
                "cpu_primary_move=1 requires an active cpu backend with "
                "cpu_variant=pool_private or pool_dual");
          std::cerr
              << " FEEBL: CPU primary movement preflight committed: "
                 "pool-static-primary-v1, interpolation=2, RNG/results=none, "
                 "threads="
              << backend.actual_cpu_threads() << std::endl;
        }
      const feebl::legacy_wavefront::ScheduleMode collision_schedule =
          feebl::legacy_wavefront::parse_schedule_mode(
              switches.get_collision_schedule());
      const feebl::adaptive_particles::Configuration& adaptive_particles =
          switches.get_adaptive_macroparticle_configuration();
      if (adaptive_particles.mode !=
          feebl::adaptive_particles::Mode::LegacyOff)
        {
          if (switches.get_adaptive_grid_mode() < 2)
            throw std::runtime_error(
                "adaptive macroparticles currently require "
                "adaptive_grid_mode=2 or 3; the fixed-grid and Mode-1 "
                "weighted-source seams are not silently substituted");
          if (backend.requested_backend() != "cpu" ||
              !backend.cpu_enabled() || backend.gpu_mode5_enabled())
            throw std::runtime_error(
                "adaptive macroparticles currently require the explicit "
                "CPU wavefront backend; GPU mode 6 is not yet qualified");
          if (switches.get_adaptive_grid_source_model() !=
              "reference_cic_v1")
            throw std::runtime_error(
                "adaptive macroparticles v1 require "
                "adaptive_grid_source_model=reference_cic_v1 so the exact "
                "root projection can be verified");
          if (adaptive_particles.statistical_cohorts != 1U)
            throw std::runtime_error(
                "projected_current_state presently contains one complete-run "
                "cohort; declaring additional statistical cohorts would "
                "manufacture independent support");
          if (!adaptive_particles.policy_save.empty() ||
              !adaptive_particles.policy_load.empty() ||
              !adaptive_particles.trace_save.empty() ||
              !adaptive_particles.trace_load.empty())
            throw std::runtime_error(
                "adaptive policy/trace persistence is fail-closed until its "
                "versioned replay schema is integrated");
          std::cerr
              << "FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1"
              << " requested_mode="
              << static_cast<int>(adaptive_particles.mode)
              << " actual_mode="
              << static_cast<int>(adaptive_particles.mode)
              << " domain=" << adaptive_particles.domain
              << " source_evolution="
              << adaptive_particles.source_evolution
              << " split=" << adaptive_particles.split_algorithm
              << " merge=" << adaptive_particles.merge_algorithm
              << " roots=new_information:false"
              << " physical_primaries=immutable"
              << " physics_scope=charge_deposition_only"
              << " qualification=software_verification_pending"
              << std::endl;
        }
      if (switches.get_adaptive_grid_mode() >= 1)
        {
          if (switches.get_adaptive_grid_scope() != "encounter")
            throw std::runtime_error(
                "adaptive-grid execution requires "
                "adaptive_grid_scope=encounter; wave scope is diagnostic "
                "only and is not silently substituted");
          if (collision_schedule !=
              feebl::legacy_wavefront::ScheduleMode::Parallel)
            throw std::runtime_error(
                "adaptive-grid execution requires "
                "collision_schedule=wavefront_parallel");
          if (switches.get_wavefront_field_solver() != "fft")
            throw std::runtime_error(
                "adaptive-grid execution requires the exact FFT field solver; "
                "direct/auto is a distinct unqualified operator");
          if (switches.get_adaptive_grid_schedule_save().empty() == false &&
              switches.get_adaptive_grid_schedule_load().empty() == false)
            throw std::runtime_error(
                "adaptive schedule save and replay cannot target the same "
                "run; choose exactly one or neither");
          if (switches.get_adaptive_grid_mode() >= 2)
            {
              if (backend.requested_backend() != "cpu" ||
                  !backend.cpu_enabled())
                throw std::runtime_error(
                    "adaptive_grid_mode=2/3 currently requires an explicitly "
                    "requested CPU wavefront backend; GPU/auto backend drift "
                    "and fallback are forbidden, while GPU mode 5 remains "
                    "the qualified Mode-1 backend");
              if (switches.get_adaptive_grid_anchor_x() != 0.0 ||
                  switches.get_adaptive_grid_anchor_y() != 0.0)
                throw std::runtime_error(
                    "adaptive_grid_mode=2/3 v1 execution requires zero "
                    "field anchors; translated anchor phases are not yet "
                    "qualified and are not silently approximated");
              if (!switches.get_adaptive_grid_schedule_save().empty() ||
                  !switches.get_adaptive_grid_schedule_load().empty())
                throw std::runtime_error(
                    "adaptive_grid_mode=2/3 schedule persistence is disabled "
                    "until the version-2 record includes resolution and "
                    "nested-patch decisions");
            }
        }
      if (backend.gpu_mode4_enabled())
        {
          if (switches.get_wavefront_field_solver() != "fft")
            throw std::runtime_error(
                "gpu_mode=4 requires wavefront_field_solver=fft; the "
                "CPU sparse-direct/auto solver and CUDA PlanMany solver "
                "are separate, non-composable experiments");
          if (collision_schedule !=
              feebl::legacy_wavefront::ScheduleMode::Parallel)
            throw std::runtime_error(
                "gpu_mode=4 requires collision_schedule=wavefront_parallel "
                "to make the wave/bucket field boundary explicit");
          if (switches.get_integration_method() != 2 ||
              switches.get_electron_distribution_rho() != 2 ||
              switches.get_interpolation() != 2)
            throw std::runtime_error(
                "gpu_mode=4 requires FFT fields (integration_method=2), "
                "CIC rho (electron_distribution_rho=2), and CIC "
                "interpolation (interpolation=2)");
          if (switches.get_force_symmetric() != 0 ||
              switches.get_extra_grids() != 0 ||
              switches.get_cpu_primary_move() != 0)
            throw std::runtime_error(
                "gpu_mode=4 currently requires force_symmetric=0, grids=1, "
                "and cpu_primary_move=0");
          if (switches.get_do_prod() || switches.get_do_trident() ||
              switches.get_do_coherent() || switches.get_track_pairs() ||
              switches.get_track_muons() || switches.get_do_pairs() ||
              switches.get_do_hadrons() || switches.get_do_compt() ||
              switches.get_do_muons())
            throw std::runtime_error(
                "gpu_mode=4 supports the exact primary/photon/luminosity "
                "host-physics domain only: pair, muon, hadron, Compton, "
                "coherent, trident, and prod paths must be disabled");
          std::cerr
              << " FEEBL: GPU mode4 card preflight committed: "
                 "encounter-dag-wave-batched-planmany-fields-v1, "
                 "one completion boundary per bounded wave bucket, "
                 "ordered legacy host physics/RNG/beamstrahlung/photon/"
                 "native-luminosity commit, fallback=forbidden, batch_width="
              << backend.actual_gpu2_options().batch_slices << std::endl;
        }
      if (backend.gpu_mode5_enabled())
        {
          if (switches.get_adaptive_grid_mode() != 1)
            throw std::runtime_error(
                "gpu_mode=5 is the adaptive Mode-1 backend and requires "
                "adaptive_grid_mode=1; fixed topology and Modes 2/3 are not "
                "silently substituted");
          if (switches.get_wavefront_field_solver() != "fft" ||
              collision_schedule !=
                  feebl::legacy_wavefront::ScheduleMode::Parallel)
            throw std::runtime_error(
                "gpu_mode=5 requires wavefront_field_solver=fft and "
                "collision_schedule=wavefront_parallel");
          if (switches.get_integration_method() != 2 ||
              switches.get_electron_distribution_rho() != 2 ||
              switches.get_interpolation() != 2)
            throw std::runtime_error(
                "gpu_mode=5 requires integration_method=2, "
                "electron_distribution_rho=2, and interpolation=2; no "
                "alternate deposition/interpolation operator is implemented");
          if (switches.get_force_symmetric() != 0 ||
              switches.get_extra_grids() != 0 ||
              switches.get_cpu_primary_move() != 0)
            throw std::runtime_error(
                "gpu_mode=5 currently requires force_symmetric=0, grids=1, "
                "and cpu_primary_move=0");
          if (switches.get_do_prod() || switches.get_do_trident() ||
              switches.get_do_coherent() || switches.get_track_pairs() ||
              switches.get_track_muons() || switches.get_do_pairs() ||
              switches.get_do_hadrons() || switches.get_do_compt() ||
              switches.get_do_muons())
            throw std::runtime_error(
                "gpu_mode=5 supports only the qualified primary/photon/"
                "luminosity host-physics domain: pair, muon, hadron, "
                "Compton, coherent, trident, and prod paths are rejected");
          std::cerr
              << "FEEBL_GPU5_COMPATIBILITY_V1"
              << " requested_backend=gpu actual_backend=gpu"
              << " requested_gpu_mode=5 actual_gpu_mode=5"
              << " engine=legacy adaptive_grid_mode=1"
              << " adaptive_grid_scope=encounter"
              << " collision_schedule=wavefront_parallel"
              << " integration_method=2 deposition=cic_global_atomic_fp64"
              << " interpolation=cic force_symmetric=0 grids=1"
              << " cpu_primary_move=0 fallback=forbidden"
              << " host_physics=primary_photon_luminosity"
              << " rejected_physics=pairs,muons,hadrons,compton,coherent,"
                 "trident,prod"
              << std::endl;
        }
      if (collision_schedule ==
              feebl::legacy_wavefront::ScheduleMode::Parallel &&
          !backend.gpu_mode4_enabled() && !backend.gpu_mode5_enabled())
        {
          // This selector is a staged algorithm, not concurrent make_step.
          // Same-wave workers own GRID/FIELD/FFT state and read exclusive
          // primary slices; after the barrier the legacy RNG/results mover is
          // committed in canonical order.  Fail closed for every feature that
          // would make the field-prefetch contract incomplete.
          if (!backend.cpu_enabled() ||
              backend.actual_cpu_variant() != "pool_dual" ||
              backend.actual_cpu_threads() < 2 ||
              backend.reduction_mode() != "strict")
            throw std::runtime_error(
                "collision_schedule=wavefront_parallel requires "
                "compute_backend=cpu, cpu_variant=pool_dual, "
                "reduction_mode=strict, and at least two actual CPU threads; "
                "no fallback was performed");
          if (switches.get_integration_method() != 2 ||
              switches.get_electron_distribution_rho() != 2 ||
              switches.get_interpolation() != 2)
            throw std::runtime_error(
                "collision_schedule=wavefront_parallel currently requires "
                "FFT fields (integration_method=2), CIC rho "
                "(electron_distribution_rho=2), and CIC interpolation "
                "(interpolation=2); no fallback was performed");
          if (switches.get_force_symmetric() != 0 ||
              switches.get_extra_grids() != 0 ||
              switches.get_cpu_primary_move() != 0)
            throw std::runtime_error(
                "collision_schedule=wavefront_parallel currently requires "
                "force_symmetric=0, grids=1, and cpu_primary_move=0; "
                "no fallback was performed");
          if (switches.get_do_prod() || switches.get_do_trident() ||
              switches.get_do_coherent() || switches.get_track_pairs() ||
              switches.get_track_muons() || switches.get_do_pairs() ||
              switches.get_do_hadrons() || switches.get_do_compt() ||
              switches.get_do_muons())
            throw std::runtime_error(
                "collision_schedule=wavefront_parallel currently supports "
                "the primary/photon/luminosity FCC-Z domain only: pair, "
                "muon, hadron, Compton, coherent, trident, and prod paths "
                "must be disabled; no fallback was performed");
          std::cerr
              << " FEEBL: collision scheduler preflight committed: "
                 "encounter-dag-staged-fields-v1, worker-private "
                 "GRID/FIELD/FFT, ordered legacy physics commit, "
                 "parallel_full_make_step=0, threads="
              << backend.actual_cpu_threads()
              << " field_solver=" << switches.get_wavefront_field_solver()
              << " sparse_direct_interaction_limit="
              << switches.get_sparse_direct_interaction_limit()
              << std::endl;
        }
      if (collision_schedule ==
              feebl::legacy_wavefront::ScheduleMode::Strict &&
          (switches.get_track_pairs() || switches.get_track_muons()))
        throw std::runtime_error(
            "collision_schedule=wavefront_strict currently supports the "
            "untracked primary collision path only; tracked pair/muon "
            "activation has additional diagonal boundary state. No fallback "
            "was performed");
      // photon_timing and photon_angle are set for each step in FEEBL::make_step; the wavefront-parallel
      // workspaces run their own photon step, which does not
      if ((switches.get_photon_timing() != 0 || switches.get_photon_angle() != 0) &&
          collision_schedule ==
              feebl::legacy_wavefront::ScheduleMode::Parallel)
        throw std::runtime_error(
            "photon_timing != 0 and photon_angle != 0 are implemented in FEEBL::make_step only; "
            "collision_schedule=wavefront_parallel (and GPU modes 4 and 5) "
            "run their own photon step. No fallback was performed");
      if (collision_schedule ==
              feebl::legacy_wavefront::ScheduleMode::Strict &&
          !backend.gpu_mode3_enabled())
        std::cerr
            << " FEEBL: collision scheduler preflight committed: "
               "encounter-dag-antidiagonal-v1, ordered legacy commit, "
               "parallel_physics=0" << std::endl;
      std::cerr << " FEEBL: " << backend.provenance() << std::endl;
    }
  catch (const std::exception& error)
    {
      std::cerr << " FEEBL: ERROR : invalid compute backend configuration: "
                << error.what() << std::endl;
      exit(1);
    }
}

// Bind the checked beam/grid state and configure output-only diagnostics.
// Native card units are converted by the parameter readers, not by the viewer.
void FEEBL::initialize_collision_state()
{
  // the next method needs the switches
  init_output_file_names();

#ifdef TWOBEAM
  std::cerr << " THE TWOBEAM IS NOT PROGRAMMED " << std::endl;
  exit(1);
#endif

  // FEEBL: optimize_grids derives the whole configuration from the beams.
  // It implies automatic xy sizing, and additionally chooses n_z, n_t and the
  // number of nested grids -- none of which upstream's automatic_grid_sizing
  // touches. Without the flag nothing changes: the deck's values are used
  // exactly as given.
  optimized_n_x_ = 0; optimized_n_y_ = 0;
  optimized_n_z_ = 0; optimized_n_t_ = 0; optimized_grids_ = 0;
  if (switches.get_optimize_grids()) optimize_grid_settings();

  grid_.read(params_, switches.get_automatic_grid_sizing());

  if (switches.get_optimize_grids())
    {
      grid_.set_n_cell_x(optimized_n_x_);
      grid_.set_n_cell_y(optimized_n_y_);
      grid_.set_n_cell_z(optimized_n_z_);
      grid_.set_timestep(optimized_n_t_);
    }
  //      std::cout << " feebl : grid parameters "  << " nmacros " << grid_.get_nb_macroparticles(1) << std::endl;

  beam1_.connect_parameters(&beam_parameters1_);
  beam2_.connect_parameters(&beam_parameters2_);
  grid_.connect_beams(&beam1_, &beam2_);
  grid_.set_native_half_crossing_angle(
      0.5 * (std::fabs((double)beam_parameters1_.x_angle())
           + std::fabs((double)beam_parameters2_.x_angle())
           + std::fabs((double)beam_parameters1_.y_angle())
           + std::fabs((double)beam_parameters2_.y_angle())));
  // The diagnostic observes all positive-weight primary collision pairs.
  // Parallel physics uses private spectra and canonical barrier publication;
  // event-heap sampling never selects the pairs entering this histogram.
  if(switches.get_incoming_map_reslice()!=0 && switches.get_incoming_map_reslice()!=1)
    throw std::invalid_argument("incoming_map_reslice must be 0 or 1");
  if(switches.get_incoming_map_reslice() && switches.get_loaded_beam_is_pretransformed())
    throw std::invalid_argument("incoming_map_reslice cannot be requested for an already pretransformed loaded beam");
  if(switches.get_design_axis_spectrum()!=0 && switches.get_design_axis_spectrum()!=1)
    throw std::invalid_argument("design_axis_spectrum must be0 or1");
  if(switches.get_design_axis_spectrum()) {
    const bool legacy_pairs = switches.get_compute_backend()=="serial" &&
       switches.get_collision_schedule()=="legacy" &&
       !switches.get_wavefront_parallel_physics() && !switches.get_cpu_primary_move() &&
       switches.get_gpu2_primary_mover()=="serial" &&
       !switches.get_electron_lumi_precompute() && !switches.get_electron_lumi_sampling();
    const bool parallel_pairs = switches.get_collision_schedule()=="wavefront_parallel" &&
       switches.get_wavefront_parallel_physics()==1 &&
       (switches.get_compute_backend()=="cpu" ||
        (switches.get_compute_backend()=="gpu" && switches.get_gpu_mode()==4)) &&
       switches.get_electron_lumi_precompute()==3 &&
       switches.get_electron_lumi_sampling()>=1 && switches.get_electron_lumi_sampling()<=3 &&
       switches.get_adaptive_grid_mode()==0;
    if((!legacy_pairs && !parallel_pairs) || switches.get_electron_lumi_spatial_kernel() ||
       !switches.get_store_native() || !(switches.get_do_lumi()&1) ||
       beam_parameters1_.y_angle()!=0 || beam_parameters2_.y_angle()!=0 ||
       beam_parameters1_.phi_angle()!=0 || beam_parameters2_.phi_angle()!=0)
      throw std::invalid_argument("design-axis diagnostic requires serial legacy pairs or exact CPU/GPU4 parallel pair publication, store_native/do_lumi, horizontal unrolled geometry");
    grid_.design_axis_spectrum_.configure(beam_parameters1_.ebeam(),beam_parameters2_.ebeam(),
      beam_parameters1_.x_angle(),beam_parameters2_.x_angle(),switches.get_design_axis_peak_bins());
    std::cerr << " FEEBL: output-only design-axis W reconstruction enabled; input energies interpreted as lab energies, local directions rotated onto design axes; dynamics unchanged" << std::endl;
  }
  generator_ = RNDM(switches.get_rndm_seed());
  std::cout << "  the random number seed is : " << switches.get_rndm_seed() << std::endl;
  grid_.connect_random_generator(&generator_);
  // Configure the invariant-mass spectrum before anything collides.
  // Range: 0 to just past the head-on maximum W = 2*sqrt(E1 E2), which
  // for equal beams is E1+E2. The 10% headroom is for the asymmetric
  // and disrupted tail rather than decoration -- anything beyond it is
  // kept as overflow, not discarded.
  //
  // The default spans the whole spectrum in 200 bins, which is right for the
  // beamstrahlung tail and useless for the peak: at ILC250 it is 1.375 GeV per
  // bin, while the core the energy resolution depends on is a fraction of a
  // GeV. The feature is narrower than one bin, and any width read off it is
  // quantisation rather than physics.
  //
  // So the window and resolution are settable. For a peak study put a few
  // thousand bins across a narrow window about nominal W --
  //   spectrum_w_min=240;spectrum_w_max=252;spectrum_w_bins=2400
  // gives 5 MeV bins -- and take the tail from the underflow counter, which is
  // still accumulated. #rms_W in the footer is unbinned either way.
  {
    double lo = switches.get_spectrum_w_min();
    double hi = switches.get_spectrum_w_max();
    int nb = switches.get_spectrum_w_bins();
    if (nb <= 0) nb = 200;
    if (!(hi > lo))
      {
        lo = 0.0;
        hi = 1.1 * ((double)beam_parameters1_.ebeam()
                  + (double)beam_parameters2_.ebeam());
      }
    grid_.get_results().configure_pair_spectrum(lo, hi, (std::size_t)nb);
    // These optional hooks cover serial physics, including field-only GPU
    // solves with the serial mover. Worker-local physics and alternate primary
    // movers expose existing luminosity spectra but do not claim new coverage.
    const bool diagnostic_hooks = get_make_gui_files() &&
        !switches.get_wavefront_parallel_physics() &&
        !switches.get_cpu_primary_move() &&
        switches.get_gpu2_primary_mover() == "serial";
    grid_.get_results().enable_gui_diagnostics(diagnostic_hooks);
    grid_.get_results().configure_gui_peak(feebl::peak_diagnostics::Definition::make(
      beam_parameters1_.ebeam(),beam_parameters2_.ebeam(),
      beam_parameters1_.x_angle(),beam_parameters2_.x_angle(),
      beam_parameters1_.y_angle(),beam_parameters2_.y_angle(),
      switches.get_crab_full_crossing_angle(),hi));
    secondaries_.enable_gui_diagnostics(diagnostic_hooks);
    std::cerr << " FEEBL: dL/dW spectrum " << lo << " to " << hi << " GeV in "
              << nb << " bins (" << (1000.0 * (hi - lo) / (double)nb)
              << " MeV each)" << std::endl;
  }
  // hist_ee_bins and hist_ee_max are REGISTERED but never read: the members
  // they would fill are commented out in switchesCPP.h. Both shipped decks set
  // them, so they look like they configure the ecm histogram and configure
  // nothing. Say so rather than let a deck believe otherwise.
  //
  // A warning, not a refusal: every existing deck carries these lines, and
  // failing on them would reject decks that are otherwise correct.
  {
    const int hb = switches.get_hist_ee_bins_declared();
    const double hm = switches.get_hist_ee_max_declared();
    if (hb != 200 || hm != -1.0)
      std::cerr << " FEEBL: WARNING : hist_ee_bins (" << hb
                << ") and hist_ee_max (" << hm << ") are IGNORED."
                << std::endl
                << "   They are registered as deck keys but never read -- the"
                   " members are commented out in" << std::endl
                << "   switchesCPP.h. The ecm histogram they appear to size"
                   " does not exist. Use" << std::endl
                << "   spectrum_w_min / spectrum_w_max / spectrum_w_bins for"
                   " the dL/dW spectrum." << std::endl;
  }

}

// Publish only after physics finalization and the adaptive qualification gate.
// The requested random-stream checkpoint is checked before success is signaled.
void FEEBL::finalize_collision(const std::string& prot)
{
  // These post-collision beam observables are physics finalization, not file
  // serialization.  Complete them before either disposition of the external
  // fresh-root handoff is allowed to describe the collision as complete.
  grid_.set_bpm_signals();
  if (adaptive_root_replay_required_)
    {
      const bool publication_blocked =
          switches.get_adaptive_macroparticle_configuration().
              fail_on_unmet_error;
      if (publication_blocked)
        {
          emit_adaptive_root_replay_signal(true, false);
          throw feebl::adaptive_particles::RootReplayRequired(
              "adaptive macroparticle end-to-end error target is unmet: " +
              adaptive_root_replay_reason_ +
              "; fresh complete-run roots are required and this invocation "
              "did not publish final summary outputs; pre-gate diagnostic "
              "artifacts, if enabled, are unqualified");
        }
    }
  if (parallel_backend().gpu_mode2_enabled())
    std::cerr << gpu2_fields_summary() << std::endl;
  if (parallel_backend().gpu_mode4_enabled())
    std::cerr << gpu4_wave_summary() << std::endl;
  outputs(prot);
  secondaries_.write_imported_spin("imported_spin.json");
  // The configuration line printed above precedes lazy CUDA initialization
  // and an explicitly authorized runtime fallback. Emit the live state after
  // outputs have captured the same provenance in the result file.
  std::cerr << " FEEBL: final " << parallel_backend().provenance()
            << std::endl;
  if (switches.get_rndm_save()) generator_.rndm_save();
  // An exploratory record is a terminal-success record, not a prediction.
  // Emit it only after every requested final summary and RNG-state output has
  // completed successfully.  A failure in outputs() therefore cannot leave an
  // authoritative-looking exit_code=0 handoff behind.
  if (adaptive_root_replay_required_)
    emit_adaptive_root_replay_signal(false, true);
}

void FEEBL::run( char *par,char *prot)
{
  adaptive_root_replay_required_ = false;
  adaptive_root_replay_reason_.clear();
  adaptive_root_replay_recommended_action_.clear();
  adaptive_root_replay_controller_observations_ = 0U;
  adaptive_root_replay_trace_hash_ = 0U;
  // Accelerator optics must be derived before PARAMETERS expressions are read.
  beam_parameters1_.read(params_);
  beam_parameters2_.read(params_);

  // reading 'PARAMETERS' : switches and grid parameters
  params_.read_parameters(par );
  mode3_input_card_label_ = par ? std::string(par)
                                : std::string("unnamed-parameters-block");
  switches.read(params_);
  configure_compute_backend();
  initialize_collision_state();
  if (!check_parameters()) exit(1);
  simulate();
  finalize_collision(std::string(prot));
}

void FEEBL::save_results_on_files()
{
  grid_.save_lumi_on_files(switches, lumi_ee_file_, lumi_eg_file_, lumi_ge_file_, lumi_gg_file_);
  if(switches.get_do_tertphot())  grid_.save_tertphot_on_file(tertphot_file_);
  if (switches.get_store_beam()) 
    {
      float gridMaxZ = grid_.get_max_z();
      float gridStep = grid_.get_step();
      int gridTimestep = grid_.get_timestep();
      float gridScalStep[2];
      gridScalStep[0] = grid_.get_scal_step(0);
      gridScalStep[1] = grid_.get_scal_step(1);
      beam1_.backstep2(1, gridMaxZ, gridStep,  gridTimestep, gridScalStep);
      beam2_.backstep2(2, gridMaxZ, gridStep,  gridTimestep, gridScalStep);
      number_of_stored_particles1_ = beam1_.store_beam(beam1_file_);
      number_of_stored_particles2_ = beam2_.store_beam(beam2_file_);
      if ( switches.get_do_coherent() )
	{
	  beam1_.store_coherent_beam(coh1_file_);
	  beam2_.store_coherent_beam(coh2_file_);
	}
      if ( switches.get_do_trident() )
	{
	  beam1_.store_trident_beam(tri1_file_);
	  beam2_.store_trident_beam(tri2_file_);
	}
      if ( switches.get_write_photons() )
        {
	  FILE_IN_OUT filout;
	  filout.open_file(photon_file_, "w");
	  for (int h=0; h < grid_.get_n_cell_z(); h++)
	    {	    
	      std::vector<PHOTON>& thePhotons1 =  beam1_.getPhotonVector(h);
	      for(unsigned int k=0;k<thePhotons1.size();k++)
		{
		  if (thePhotons1[k].energy()>0.0){
		    filout.save_object_on_persistent_file( &thePhotons1[k]);
		  }
		}
	      std::vector<PHOTON>& thePhotons2 =  beam2_.getPhotonVector(h);
	      for(unsigned int k=0;k<thePhotons2.size();k++){
		float ene = thePhotons2[k].energy();
		if (thePhotons2[k].energy()>0.0)
		  {
		    thePhotons2[k].setEnergy(-ene);
		    filout.save_object_on_persistent_file(&thePhotons2[k]);
		  }
	      }
	    // photon_vector_.store_photon(photon_file_);
	    }
	  filout.close();
        }
    }
  //   if (switches.get_track_secondaries() || switches.get_store_secondaries()) secondaries_.save_pairs_on_file(secondaries_file_);
  //   if (switches.get_store_secondaries() > 1) secondaries_.save_pairs0_on_file(secondaries0_file_);
  if (switches.get_do_bhabhas())
    {
      grid_.get_bhabhas().save_on_files(switches.get_do_bhabhas(), switches.get_do_edm4hep(), bhabha_prod_, bhphoton_prod_,bhphotons_);
      if (switches.get_track_pairs() || switches.get_store_pairs()) 
	{
	  secondaries_.save_bhabhas_on_file(secondaries_file_);
	}
      if (switches.get_store_pairs() > 1){
	secondaries_.save_bhabhas0_on_file(secondaries0_file_);
      }
    }
  else
    {
      if (switches.get_track_pairs() || switches.get_store_pairs()) 
	{
	  secondaries_.save_pairs_on_file(secondaries_file_);
	}
      if (switches.get_store_pairs() > 1) {
	secondaries_.save_pairs0_on_file(secondaries0_file_);
      }
    }
  if (switches.get_track_muons() || switches.get_store_muons()) 
    {
      muons_.save_pairs_on_file(muons_file_);
    }
  if (switches.get_store_muons() > 1) 
    {
      muons_.save_pairs0_on_file(muons0_file_);
    }
}

void FEEBL::outputs(std::string nameOfProtokoll)
{
  float miss1, miss2;
  long out1, out2;
  if (parallel_backend().gpu_mode3_enabled())
    std::cerr
        << " FEEBL GPU mode3: legacy per-diagonal GRID miss diagnostics are "
           "unavailable; use the resident aggregate deposit/gather in/out "
           "counters in the production summary."
        << std::endl;
  else
    {
      grid_.get_miss(miss1, out1, miss2, out2);
      if (miss1 > 0.1 || miss2 > 0.1)
        {
          std::cerr << " ---------------------------------------------------------- " << std::endl;
          std::cerr << " WARNING : there is a quite important quantity of particles travelling out of grid " << std::endl;
          std::cerr  << "miss.1 = " << miss1 << " out.1 = " << out1 << " miss.2 = " << miss2 << " out.2 = " << out2 << std::endl;
          std::cerr << " maybe the grid dimensions should be rematched " << std::endl;
          std::cerr << " ---------------------------------------------------------- " << std::endl;
        }
    }
  save_results_on_files();
  print_program_outputs(nameOfProtokoll);
  if (get_make_gui_files()) write_gui_manifest();
  //for(int i1=1;i1<=6;i1++) time_.print_timer(i1);
  //time_.print_timer_all(6);
}

bool FEEBL::check_parameters() const
{
  // ---------------------------------------------------------------------
  // FEEBL divergence from GUINEA-PIG++ 1.2.2.
  //
  // xi_x, xi_y and couple_xy are declared and defaulted upstream but are never
  // read by any code path (BEAM_PARAMETERS::couple_xy_ is commented out). A
  // deck that sets them is silently ignored. Warn instead: a parameter that
  // does nothing is a correctness hazard, not a convenience.
  // ---------------------------------------------------------------------
  {
    const char* inert[] = { "xi_x", "xi_y", "couple_xy" };
    for (int i = 0; i < 3; i++)
      {
        double v1 = 0.0, v2 = 0.0;
        std::string n1 = std::string(inert[i]) + ".1";
        std::string n2 = std::string(inert[i]) + ".2";
        v1 = params_.readFValue(n1);
        v2 = params_.readFValue(n2);
        if (v1 != 0.0 || v2 != 0.0)
          {
            std::cerr << " FEEBL::check_parameters: WARNING : '" << inert[i]
                      << "' is accepted by the parser but is NOT IMPLEMENTED,"
                      << " here or in GUINEA-PIG++. It has no effect on the"
                      << " result. Remove it from the deck." << std::endl;
          }
      }
  }

  int k;
  bool check = true;
  // ---------------------------------------------------------------------
  // FEEBL: what a finite-angle run may be quoted for.
  //
  // set_angle_particles applies the crossing angle as a per-slice POSITION
  // shear and never touches vx_/vy_, and ABSTRACT_PARTICLE has no longitudinal
  // direction of its own. So momenta simply do not carry the crossing angle.
  //
  // Luminosity survives this: the slice pairing is the true pairing and the
  // shear reproduces the RELATIVE transverse separation exactly, with the whole
  // residual a common-mode translation that a translation-invariant luminosity
  // is blind to. Everything derived from momenta does not survive -- it is short
  // by the full half crossing angle, which is 378 sigma_x' on the ILC deck, not
  // a small parameter. Laboratory angles require an explicit transformation.
  //
  // Warn rather than refuse: the energy-domain results in the same run are
  // valid, and often they are the point of the run.
  // ---------------------------------------------------------------------
  {
    const double ax = std::fabs((double)beam_parameters1_.x_angle())
                    + std::fabs((double)beam_parameters2_.x_angle());
    const double ay = std::fabs((double)beam_parameters1_.y_angle())
                    + std::fabs((double)beam_parameters2_.y_angle());
    if (ax + ay > 0.0)
      {
        const bool angular = switches.get_do_pairs() || switches.get_do_muons()
          || switches.get_do_photons(1) || switches.get_do_photons(2)
          || switches.get_write_photons() || switches.get_do_lumi()
          || switches.get_store_native();
        if (angular)
          {
            std::ostringstream w;
            w << " FEEBL: WARNING : this run has a crossing angle (half-angle "
              << 0.5 * (ax + ay) << " rad) and requests angular observables."
              << std::endl
              << "   Momenta do NOT carry the crossing angle: every direction in"
                 " this run is in the head-on frame." << std::endl
              << "   Luminosity and energy-domain results are valid. Angular,"
                 " directional and frame-dependent" << std::endl
              << "   observables are NOT -- pair n_pt and cone counts, photon and"
                 " pair directions, lumi.ee.out" << std::endl
              << "   kinematics, and the momentum vectors in the native"
                 " per-particle output." << std::endl
              << "   Transform momenta to the laboratory frame before using laboratory angles."
              << std::endl;
            std::cerr << w.str();
            std::cout << w.str();
          }
      }
  }

  // ---------------------------------------------------------------------
  // FEEBL: a reloaded beam can be transformed twice.
  //
  // beam_displacements_from_data applies the roll, the crossing-angle shear,
  // the offsets and now the incoming crab map -- and it is called
  // UNCONDITIONALLY on both setup paths. A beam stored from a run that already
  // had those applied carries them in its positions, so reloading it applies
  // them a second time: the shear doubles, the crab tilt doubles, and nothing
  // says so. The stored file is not a valid 6D phase space either, since it has
  // <xz>/sigma_z^2 = angle_x with <x_prime> = 0.
  //
  // FEEBL cannot tell from the file, which is a bare particle list with no
  // header. So the deck says: loaded_beam_is_pretransformed=1 means the file
  // ALREADY carries them and they are skipped. The default is 0, which applies
  // them as before -- correct for a genuinely untransformed file, and warned
  // about here because it is silently wrong for any other kind.
  // ---------------------------------------------------------------------
  if (switches.get_load_beam() != 0 && !switches.get_loaded_beam_is_pretransformed())
    {
      const double t =
          std::fabs((double)beam_parameters1_.x_angle())
        + std::fabs((double)beam_parameters2_.x_angle())
        + std::fabs((double)beam_parameters1_.y_angle())
        + std::fabs((double)beam_parameters2_.y_angle())
        + std::fabs((double)beam_parameters1_.phi_angle())
        + std::fabs((double)beam_parameters2_.phi_angle())
        + std::fabs((double)beam_parameters1_.offset_x())
        + std::fabs((double)beam_parameters2_.offset_x())
        + std::fabs((double)beam_parameters1_.offset_y())
        + std::fabs((double)beam_parameters2_.offset_y());
      const bool crab = switches.get_crab_mode(1) || switches.get_crab_mode(2)
                     || switches.get_waist_mode(1) || switches.get_waist_mode(2)
                     || switches.get_ip_dispersion(1) != 0.0
                     || switches.get_ip_dispersion(2) != 0.0;
      if (t > 0.0 || crab)
        {
          std::cerr << " FEEBL: WARNING : load_beam is set together with beam"
                       " transforms." << std::endl
                    << "   The crossing angle, roll, offsets and incoming crab map"
                       " are applied to a loaded beam" << std::endl
                    << "   exactly as to a generated one. If this file came from a"
                       " run that already had them," << std::endl
                    << "   they are now applied TWICE and no other diagnostic will"
                       " say so." << std::endl
                    << "   Set loaded_beam_is_pretransformed=1 to skip them."
                       " This skips all incoming coordinate transforms." << std::endl;
        }
    }

  // ---------------------------------------------------------------------
  // FEEBL: automatic_grid_sizing is not crossing-angle aware.
  //
  // main_grid_automatic_xycuts measures the beam moments AFTER the shear has
  // been applied and then adds the crossing-angle sweep on top, so the sweep is
  // counted twice: cut_x comes out roughly 2.5x too wide and the cell count
  // chosen from it correspondingly too coarse. The two beam-setup paths also
  // order the shear and the sizing oppositely, so they would not even agree
  // with each other. Refuse rather than silently produce a wrong grid.
  // ---------------------------------------------------------------------
  {
    const double a = std::fabs((double)beam_parameters1_.x_angle())
                   + std::fabs((double)beam_parameters2_.x_angle())
                   + std::fabs((double)beam_parameters1_.y_angle())
                   + std::fabs((double)beam_parameters2_.y_angle());
    if (switches.get_automatic_grid_sizing() && a > 0.0)
      {
        std::cerr << " FEEBL::check_parameters: ERROR : automatic_grid_sizing=1"
                     " with a non-zero crossing angle." << std::endl
                  << "   The beam moments are measured after the crossing-angle"
                     " shear and the sweep is then added" << std::endl
                  << "   again, so cut_x comes out about 2.5x too wide and the"
                     " mesh correspondingly too coarse." << std::endl
                  << "   Use optimize_grids=1, or set cut_x and cut_y in the"
                     " deck and check mesh convergence." << std::endl;
        check = false;
      }
  }


  //std::cerr << " checking parameters... " << std::endl;
#ifdef USE_FFT_LOCAL
  int vnx = TOOLS::check_power_of_2(grid_.get_n_cell_x() );
  int vny = TOOLS::check_power_of_2(grid_.get_n_cell_y() );
  if (switches.get_integration_method() == 2)
    {
      if (!vnx || !vny  )
	{
	  std::cerr << " ERROR : with integration_method = 2 (FFT_LOCAL) the cell numbers must be power of 2 " << std::endl;
	  std::cerr << " Consider compiling with FFTW2 or FFTW3 (also for cpu-speed)" << std::endl;
	  std::cerr << " n_x = " << grid_.get_n_cell_x() << " n_y = " << grid_.get_n_cell_y()  << std::endl;
	  check = false;
	}
    }
#endif
  if ( switches.get_do_prod() )
    {
      std::cerr << " ERROR : key_word do_prod not completely implemented " << std::endl;
      // see grid::move_particles for the do_beamstrahlung case
      check = false;
    }
  if (beam_parameters1_.dist_x() != 0 || beam_parameters2_.dist_x() != 0)
    {
      std::cerr << " dist_x other than 0 is not available " << std::endl;
      std::cerr << " dist_x.1 = " << beam_parameters1_.dist_x() << " dist_x.2 = "
	   <<  beam_parameters2_.dist_x() << std::endl;
      check = false;
    }

  bool do_photons = switches.get_do_photons(1)>0 || switches.get_do_photons(2)>0;
  bool load_photon = switches.get_load_photon()>0;
  if (switches.get_do_coherent() && !do_photons && !load_photon)
    {
      std::cerr << " FEEBL::check_parameters:: it is not useful to have do_coherent=1 with no do_photons nor load_photons " << std::endl;
    }
  //if ( switches.get_track_secondaries() > 0 )
  if ( switches.get_track_pairs() > 0 )
    {
      bool do_pair_compt_bha = switches.get_do_pairs()>0 || switches.get_do_compt()>0 || switches.get_do_bhabhas()>0 || switches.get_load_event()>0;
      if (!do_pair_compt_bha) 
	{
	  //std::cerr << " with track_secondaries > 0 there must be  either do_pairs = 1 or do_compt = 1 or do_bhabhas = 1 " << std::endl;
	  std::cerr << " with track_pairs > 0 there must be do_pairs, do_compt, do_bhabhas or load_events enabled " << std::endl;
	  check = false;

	}
    }

  const BEAM_PARAMETERS* bpPtr[2]  = {&beam_parameters1_, &beam_parameters2_};  
  float cutz = params_.readFValue("cut_z");
  if ( !switches.get_load_beam() )
    {
      // program generated beam
      //       if ( cutz <= 0 ) 
      // 	{
      // 	  std::cerr << "  FEEBL::check_parameters: the data of cut_z is mandatory. program stop " << std::endl;
      // 	  check = false;      
      // 	}
      if ( switches.get_cuts_from_loaded_beam() ) 
	{
	  std::cerr << " FEEBL::check_parameters: WARNING  with beams generated by program the keyword cuts_from_loaded_beam is without effect "<< std::endl;
	}

      for (k=0; k < 2; k++)
	{
	  if ( bpPtr[k]->sigma_z() <= 0.0 )
	    {
	      std::cerr << "  FEEBL::check_parameters: the data of sigma_z is mandatory. program stop " << std::endl;
	      check = false;
	    }
	}

      for (k=0; k < 2; k++)
	{
	  int number=0;
	  if (bpPtr[k]->beta_x() > 0.0 ) number++;
	  if (bpPtr[k]->em_x() > 0.0 ) number++;
	  if (bpPtr[k]->sigma_x() > 0.0 ) number++;
	  if (number < 2 )
	    {
	      std::cerr << " FEEBL::check_parameters : ERROR among beta_x, emitt_x, sigma_x only one parameter is given for beam " << k+1 << " . Two are needed " << std::endl;
	      check = false;
	    }
	  number = 0;
	  if (bpPtr[k]->beta_y() > 0.0 ) number++;
	  if (bpPtr[k]->em_y() > 0.0 ) number++;
	  if (bpPtr[k]->sigma_y() > 0.0 ) number++;
	  if (number < 2 )
	    {
	      std::cerr << " FEEBL::check_parameters : ERROR among beta_y, emitt_y, sigma_y only one parameter is given. Two are needed  " << std::endl;
	      check = false;
	    }
	}
    }
  else
    {
      // loaded beam 
      if (switches.get_automatic_grid_sizing() )
	{
	  if ( switches.get_cuts_from_loaded_beam() ) 
	    {
	      std::cerr << " FEEBL::check_parameters: WARNING  with automatic_grid_sizing,  the keyword cuts_from_loaded_beam is without effect "<< std::endl;
	    }
	}
      else
	{
	  if ( params_.readFValue("cut_x") <= 0.0 || params_.readFValue("cut_y") <= 0.0 )
	    {
	      bool test = false;
	      for (k=0; k < 2; k++)
		{
		  int number=0;
		  if (bpPtr[k]->beta_x() > 0.0 ) number++;
		  if (bpPtr[k]->em_x() > 0.0 ) number++;
		  if (bpPtr[k]->sigma_x() > 0.0 ) number++;
		  if (number < 2 )
		    {
		      test = true;
		      check = false;
		    }
		  number = 0;
		  if (bpPtr[k]->beta_y() > 0.0 ) number++;
		  if (bpPtr[k]->em_y() > 0.0 ) number++;
		  if (bpPtr[k]->sigma_y() > 0.0 ) number++;
		  if (number < 2 )
		    {
		      test = true;
		      check = false;
		    }
		}
	      if (test ) 
		{
		  std::cerr << " FEEBL::check_parameters : ERROR cut_x or cut_y is missing. In that case the beam transverse characteristics (beta, sigma, emitt) must be defined. Else use the option automatic_grid_sizing = 1  " << std::endl;
		}
	    }
	  if ( cutz <= 0.0) 
	    {
	      bool test = false;
	      for (k=0; k < 2; k++)
		{
		  if ( bpPtr[k]->sigma_z() <= 0.0 )
		    {
		      test = true;
		      check = false;
		    }
		}
	      if (test) 
		{
		  std::cerr << "  FEEBL::check_parameters: one of sigma_z and cut_z must be given. program stop " << std::endl;
		}
	    }
	}
    }
  if ( !grid_.random_ok() )
    {
      std::cerr << " FEEBL::check_parameters: GRID seems not to be connected to the random generator " << std::endl;
      check = false;
    }
  if (switches.get_ST_spin_flip()  != 0) 
    {
      if (switches.get_bmt_precession() == 0) 
	{
	  std::cerr << " with ST_spin_flip, bmt_precession = 1 is MANDATORY " << std::endl;
	  check = false;
	}
    }

  // scale_step changes the position drift but leaves the momentum kick
  // and BMT precession step unchanged. Its nonunit physical interpretation
  // is unresolved. Refuse it for spin tracking; retain the inherited orbit
  // option with an explicit warning for compatibility.
  {
    const double s1 = (double)grid_.get_scal_step(0);
    const double s2 = (double)grid_.get_scal_step(1);
    const bool scaled = std::fabs(s1 - 1.0) > 1.0e-9 || std::fabs(s2 - 1.0) > 1.0e-9;
    const bool spin_on = switches.get_bmt_precession() || switches.get_ST_spin_flip()
                      || switches.get_do_bds_spin_rotation();
    if (scaled && spin_on)
      {
        std::cerr << " FEEBL: ERROR : scale_step != 1 together with spin is not"
                     " supported." << std::endl
                  << "   scale_step (" << s1 << ", " << s2 << ") lengthens the"
                     " orbit DRIFT but not the" << std::endl
                  << "   momentum kick: advanceDueToEBfield calls"
                     " advanceVelocities with step_ and" << std::endl
                  << "   advancePosition with step_*scale_step. The spin"
                     " precession uses step_, so it" << std::endl
                  << "   agrees with the kick and not with the drift, and which"
                     " of the two a spin" << std::endl
                  << "   should follow is not derivable from the source."
                  << std::endl
                  << "   Rather than report a polarization that depends on an"
                     " unresolved convention," << std::endl
                  << "   FEEBL refuses. Use scale_step = 1 with spin, or run"
                     " without spin." << std::endl;
        check = false;
      }
    else if (scaled)
      {
        std::ostringstream w;
        w << " FEEBL: WARNING : scale_step != 1 (" << s1 << ", " << s2 << ")."
          << std::endl
          << "   It lengthens the orbit drift but NOT the momentum kick"
             " (advanceDueToEBfield" << std::endl
          << "   passes step_ to advanceVelocities and step_*scale_step to"
             " advancePosition)." << std::endl
          << "   That is inherited GUINEA-PIG++ behaviour and is not verified"
             " here. Measured on" << std::endl
          << "   the quick deck: lumi_fine 1.29387e34 at 1.0 against 9.08708e33"
             " at 2.0." << std::endl;
        std::cerr << w.str();
        std::cout << w.str();
      }
  }

  // The e+e- spin path carries the relative source charge exactly once.
  // Same-sign collisions remain disabled until their complete orbit, radiation
  // and spin convention has independent end-to-end validation.
  {
    const double cs = (double)switches.get_charge_sign();
    const bool is_ee   = std::fabs(cs + 1.0) < 1.0e-6;   // e+e-
    const bool is_none = std::fabs(cs)       < 1.0e-6;   // no beam-beam force
    const bool is_same = std::fabs(cs - 1.0) < 1.0e-6;   // e-e-
    if (is_same)
      {
        std::cerr << " FEEBL: ERROR : charge_sign = +1 (e-e-) is not supported"
                     " in this version." << std::endl
                  << "   Same-sign collision spin and radiation conventions"
                     " have not been independently validated." << std::endl
                  << "   Use charge_sign = -1.0 (e+e-) or charge_sign = 0"
                     " (no beam-beam force)." << std::endl;
        check = false;
      }
    else if (!is_ee && !is_none)
      {
        std::cerr << " FEEBL: ERROR : charge_sign = " << cs
                  << " is not a supported value." << std::endl
                  << "   It is the RELATIVE sign of the two beam charges, so"
                     " the only meaningful" << std::endl
                  << "   values are -1.0 (e+e-), 0 (no beam-beam force) and"
                     " +1.0 (e-e-, currently" << std::endl
                  << "   refused, see above). Any other value is applied twice"
                     " -- once in the field" << std::endl
                  << "   normalisation and once in the spin routines -- and"
                     " would scale the" << std::endl
                  << "   precession by its square." << std::endl;
        check = false;
      }
  }

  // THE SILENT POLARIZATION DEFAULT.
  //
  // readData.cc registers polar_x=0, polar_y=0, polar_z=1.0, so enabling
  // bmt_precession without saying anything about polarization produces a beam
  // that is 100% longitudinally polarized -- silently. The beam-parameter
  // listing does print "polar_z = 1", but among many other lines and on every
  // run including those with spin off, where it reads as a placeholder.
  //
  // The DEFAULT VALUE IS DELIBERATELY NOT CHANGED. It is echoed to the
  // protokoll on every run, spin or not, so altering it would change the output
  // of every existing deck and break the bit-identity check against v0.5.0.
  // The trap case is refused instead.
  //
  // The detector is deliberately conservative: it fires only when the deck
  // mentions no polarization key ANYWHERE. A deck that sets polar_ in any block
  // is taken to know what it is doing, so an explicit polar_z=1.0 is accepted
  // without comment. That is deck-level rather than block-level, which is the
  // price of the parser not recording whether a key was assigned or defaulted.
  if (switches.get_bmt_precession() || switches.get_ST_spin_flip()
      || switches.get_do_bds_spin_rotation())
    {
      bool mentions_polar = false;
      std::ifstream deck(get_acc_filename().c_str());
      if (deck)
        {
          std::string line;
          while (std::getline(deck, line))
            if (line.find("polar_") != std::string::npos)
              { mentions_polar = true; break; }
        }
      if (!mentions_polar)
        {
          std::cerr << " FEEBL: ERROR : this run enables spin but the deck never"
                       " mentions polarization." << std::endl
                    << "   polar_x/y/z default to (0, 0, 1), so the beams would"
                       " be 100% longitudinally" << std::endl
                    << "   polarized -- which is a physics choice, not a neutral"
                       " starting point, and it" << std::endl
                    << "   would be made silently." << std::endl
                    << "   State it: put polar_x, polar_y and polar_z in the"
                       " ACCELERATOR block. Use" << std::endl
                    << "   polar_z=0 for an unpolarized beam, or polar_z=1.0 if"
                       " full longitudinal" << std::endl
                    << "   polarization really is what you want." << std::endl;
          check = false;
        }
    }

  if (switches.get_do_edm4hep()) {
    std::cerr << " FEEBL: ERROR : do_edm4hep=1 is not supported." << std::endl
              << "   The EDM4HEP writer was removed in FEEBL 0.6. It could not be"
                 " built or exercised in" << std::endl
              << "   this environment, and review found it writing both beams with"
                 " the same sign of pz" << std::endl
              << "   and reconstructing energy from a truncated float. Rather than"
                 " ship an untestable" << std::endl
              << "   path, use store_native=1: it records the same per-particle"
                 " information in a" << std::endl
              << "   dependency-free, self-describing text format that states its"
                 " own units, frame and" << std::endl
              << "   sign convention (src/nativeOutputCPP.h)." << std::endl
              << "   The switch is still parsed and echoed to the protokoll so"
                 " existing decks and" << std::endl
              << "   reference outputs stay byte-comparable." << std::endl;
    check = false;
  }

  //std::cerr << " end of checking ... " << std::endl;
  return check;
}

void FEEBL::set_simulation()
{
  grid_.check_distribute(0);
  int n_slice=grid_.get_n_cell_z();
  // Spin STORAGE, not spin dynamics. Gating this on bmt_precession alone meant
  // do_bds_spin_rotation=1 by itself allocated plain PARTICLEs, so the BDS
  // rotation had nothing to act on: measured exit 0, no polarization block, no
  // diagnostic. Any switch that needs a spin now allocates one.
  //
  // Precession stays separately gated: GRID::move_particles calls rotateBMT
  // only when its own bmt_rotate parameter is set, which still comes from
  // bmt_precession. So a beam can carry and report a polarization without the
  // collision precessing it.
  const int needs_spin_storage = switches.get_bmt_precession()
                               || switches.get_ST_spin_flip()
                               || switches.get_do_bds_spin_rotation();
  beam1_.make_beam(n_slice, needs_spin_storage, &generator_);
  beam2_.make_beam(n_slice, needs_spin_storage, &generator_);


  if (switches.get_do_size_log())
    {
      beam1_.write_size_init(std::string("beamsize"));
      beam2_.write_size_init(std::string("beamsize"));
    }

  if (switches.get_rndm_load()) 
    {
      generator_.rndm_load();
    }
  if (switches.get_load_event())
    {
      secondaries_.enable_imported_spin(switches.get_imported_spin_mode()==1);
      secondaries_.set_load_file(event_input_file_);
    }


  if (switches.get_do_jets())
    {
      grid_.init_jet(4.0*beam_parameters1_.ebeam()*beam_parameters2_.ebeam(), switches.get_jet_pstar(), switches.get_do_jets(),switches.get_jet_pythia(), switches.get_jet_select(), jet_file_);
    }
  if (switches.get_do_cross())
    {
      if (MCROSS) 
	{
	  if (AVERCROSS) cross_ = new maverCROSS(cross_input_file_);
	  else cross_ = new mCROSS(cross_input_file_);
	}
      else 
	{
	  if (AVERCROSS) cross_ = new averCROSS(cross_input_file_);
	  else cross_ = new CROSS(cross_input_file_);
	}
    }
  else cross_ = NULL;
  grid_.generalInit(switches, photon_file_, hadronfile_, compton_phot_file_,bhabha_input_file_, bhabha_photon_input_file_, cross_);
  //  if(switches.get_do_pairs()||switches.get_do_hadrons()||switches.get_do_compt()
  //       ||switches.get_do_muons())
  //    {
  //     grid_.init_extra_photons();
  //    }
}


// Apply incoming dispersion, crab and crab-waist maps to the beams.
//
// These transformations describe incoming optics. Apply them once per bunch,
// before tracking, without modifying the beam-beam force law.
//
// Every failure below aborts. The whole point of this stage is that a wrong
// convention produces a plausible luminosity rather than a visible error, so
// there is nothing safe to fall back on.
// FEEBL: is the crossing-angle shear resolved by the longitudinal slicing?
//
// PARTICLE_BEAM::set_angle_particles gives every particle in a slice the SAME
// transverse offset, so the sheared bunch is a staircase of step
// angle_x*delta_z. Two different things follow, and they need different
// treatment:
//
//   * WITH a crab map, the tilt is applied per PARTICLE against a shear applied
//     per SLICE, so the two fail to cancel below the slice scale. That is FIRST
//     order and severe: measured 68.4% of head-on at 3.27 sigma_x, 95.9% at
//     0.79. apply_incoming_maps REFUSES above 0.8 sigma_x for that reason.
//
//   * WITHOUT a crab map there is no cancellation to fail. Each slice pair's
//     relative separation is still exact at the slice centres, so the error is
//     ordinary quadrature. Measured on the quick deck at 30 mrad: reducing the
//     smear 4x, from 3.27 to 0.79 sigma_x, moved luminosity by 0.55%. Second
//     order, and not worth refusing over -- especially since benchmark16 s3 sits
//     at 2.43 sigma_x and is a shipped, validated block.
//
// So: warn, with the number, and leave the judgement to whoever reads it.
void FEEBL::warn_on_shear_quantisation(float delta_z_nm) const
{
  if (!(delta_z_nm > 0.0)) return;

  for (int b = 1; b <= 2; b++)
    {
      // A crabbed beam is already covered, and more strictly, by the refusal in
      // apply_incoming_maps. Do not say it twice.
      if (switches.get_crab_mode(b) != 0) continue;

      const BEAM_PARAMETERS& bp = (b == 1) ? beam_parameters1_ : beam_parameters2_;
      const double ax = std::fabs((double)bp.x_angle());
      const double ay = std::fabs((double)bp.y_angle());
      const double sx = (double)bp.sigma_x();
      const double sy = (double)bp.sigma_y();

      double smear = 0.0; char plane = 'x'; double sigma = sx;
      if (sx > 0.0 && ax * (double)delta_z_nm / sx > smear)
        { smear = ax * (double)delta_z_nm / sx; plane = 'x'; sigma = sx; }
      if (sy > 0.0 && ay * (double)delta_z_nm / sy > smear)
        { smear = ay * (double)delta_z_nm / sy; plane = 'y'; sigma = sy; }

      if (smear > 1.0)
        {
          const int nz = grid_.get_n_cell_z();
          std::cerr << " FEEBL: WARNING : beam " << b << " crossing-angle shear is"
                       " quantised at " << smear << " sigma_" << plane
                    << " per slice." << std::endl
                    << "   n_z = " << nz << " gives a " << delta_z_nm
                    << " nm slice; sigma_" << plane << " is " << sigma
                    << " nm. Uncrabbed this is second order" << std::endl
                    << "   (measured 0.55% for a 4x smear reduction on the quick"
                       " deck), so it is a convergence" << std::endl
                    << "   question rather than an error -- but it has not been"
                       " measured for this deck. n_z >= "
                    << (int)std::ceil(nz * smear) << " would remove it."
                    << std::endl;
        }
    }
}

// The beam energy spread, applied once per particle at setup.
//
// Historically this happened inside collide_ee, to the two energies entering
// the luminosity sum, which made it a property of the SPECTRUM rather than of
// the beam. Particles crossed the interaction region monoenergetic, so delta
// was zero for the incoming maps, the deflection and the beamstrahlung alike.
//
// Applying it here instead makes delta a real beam property. Two consequences
// worth stating plainly:
//
//   * It is strictly more physical. Real beams carry their energy spread
//     through the collision, and the spread now feeds the dynamics rather than
//     being pasted onto the answer at the end.
//
//   * It changes results for any deck with do_espread=1, and it must: the
//     spread now affects deflection and beamstrahlung, not just the reported
//     ecm. Every deck in examples/decks leaves do_espread at its default of 0, so
//     no existing result moves -- verified by bitcheck and benchmark16.
void FEEBL::apply_energy_spread()
{
  if (!switches.get_do_espread()) return;

  for (int b = 1; b <= 2; b++)
    {
      const int which = (b == 1) ? switches.get_which_espread1()
                                 : switches.get_which_espread2();
      const float spread = (b == 1) ? switches.get_espread1()
                                    : switches.get_espread2();
      if (which == 0 || spread == 0.0f) continue;

      BEAM& beam = (b == 1) ? beam1_ : beam2_;
      const int nsl = beam.number_of_slices();
      unsigned long int n = 0;
      double s1 = 0.0, s2 = 0.0;

      for (int sl = 0; sl < nsl; sl++)
        {
          std::vector<PARTICLE*>& v = beam.getParticleVector(sl);
          for (std::size_t k = 0; k < v.size(); k++)
            {
              const float f = energy_spread_factor(which, spread, generator_);
              // Beam-particle energy is signed in places (collide_ee takes
              // fabs of it), and a positive factor preserves whatever sign it
              // carries. Unlike PAIR_PARTICLE there is no cached 1/energy to
              // keep in step -- ABSTRACT_PARTICLE stores energy_ alone.
              v[k]->setEnergy(v[k]->energy() * f);
              const double d = (double)f - 1.0;
              s1 += d; s2 += d * d; n++;
            }
        }

      // Report what was actually produced, not what was asked for. Case 1 is
      // flat on [-1.7, +1.7], whose RMS is 0.98 rather than 1, so the delivered
      // sigma is a few percent below the requested spread -- better said out
      // loud than discovered later in a spectrum.
      if (n > 0)
        {
          const double mean = s1 / (double)n;
          const double sd = std::sqrt(std::fmax(s2 / (double)n - mean * mean, 0.0));
          std::cerr << " FEEBL::apply_energy_spread: beam " << b
                    << ": which_espread=" << which
                    << " requested " << spread
                    << ", delivered sigma_delta = " << sd
                    << " over " << n << " particles" << std::endl;
        }
    }
}

// Read the beam-local map and validate the two-beam longitudinal convention.
IncomingMapConfig FEEBL::configure_incoming_map(int b)
{
  IncomingMapConfig c;
  c.crab_mode               = switches.get_crab_mode(b);
  c.crab_plane              = switches.get_crab_plane(b);
  c.crab_frequency_Hz       = switches.get_crab_frequency(b);
  c.crab_phase_rad          = switches.get_crab_phase(b);
  c.crab_target_tilt        = switches.get_crab_target_tilt(b);
  c.crab_harmonic_n         = switches.get_crab_harmonic_n();
  c.include_panofsky_wenzel = switches.get_crab_include_panofsky_wenzel();
  c.waist_mode              = switches.get_waist_mode(b);
  c.waist_fraction          = switches.get_waist_fraction(b);
  c.ip_dispersion_m         = switches.get_ip_dispersion(b);

  if (b == 1)
    {
      const double d1 = switches.get_ip_dispersion(1);
      const double d2 = switches.get_ip_dispersion(2);
      if (d1 != 0.0 && d2 != 0.0 && (d1 > 0.0) == (d2 > 0.0))
        std::cerr << " FEEBL::apply_incoming_maps: WARNING:"
                     " ip_dispersion.1 and ip_dispersion.2 have the"
                  << std::endl
                  << "   SAME sign (" << d1 << ", " << d2 << ")."
                     " Opposite signs are the usual monochromatization choice;"
                  << std::endl
                  << "   for equal dispersions and matched independent Gaussian energy spreads,"
                     " same-sign dispersion selects"
                  << std::endl
                  << "   delta1-delta2 independently of delta1+delta2."
                     " Asymmetric beams require a separate overlap calculation."
                  << std::endl;
    }
  c.full_crossing_angle_rad = switches.get_crab_full_crossing_angle();
  c.symplecticity_tolerance = switches.get_waist_symplecticity_tolerance();
  c.longitudinal_sign       = switches.get_crab_longitudinal_sign(b);

  if (b == 1 && c.crab_mode != CRAB_NONE)
    {
      const int s1 = switches.get_crab_longitudinal_sign(1);
      const int s2 = switches.get_crab_longitudinal_sign(2);
      if (s1 == 0 || s2 == 0 || s1 != -s2)
        {
          std::cerr << " FEEBL::apply_incoming_maps: crab_longitudinal_sign"
                       " must be +1 on one beam and -1" << std::endl
                    << "   on the other; got .1=" << s1 << " .2=" << s2
                    << "." << std::endl
                    << "   The derived tilt is -sign*theta/2 and the"
                       " crossing angle is applied identically to" << std::endl
                    << "   both beams, so equal signs tilt both bunches the"
                       " same way and the crab does not" << std::endl
                    << "   cancel the declared crossing geometry." << std::endl
                    << "   Note this is a PER-BEAM key: writing"
                       " crab_longitudinal_sign=1 sets both."
                    << std::endl
                    << "   Write crab_longitudinal_sign.1=1;"
                       "crab_longitudinal_sign.2=-1 instead." << std::endl;
          exit(1);
        }
    }
  return c;
}

// Require the map and beam angles to describe the same crossing geometry.
void FEEBL::check_incoming_crossing_geometry(
    int b,
    const IncomingMapConfig& c)
{
  if ((c.crab_mode != CRAB_NONE && c.crab_target_tilt == -1.0)
      || c.waist_mode != WAIST_NONE)
    {

      const double ax1 = (double)beam_parameters1_.x_angle();
      const double ax2 = (double)beam_parameters2_.x_angle();
      const double ay1 = (double)beam_parameters1_.y_angle();
      const double ay2 = (double)beam_parameters2_.y_angle();
      if (ax1 * ax2 < 0.0 || ay1 * ay2 < 0.0)
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": the two beams' crossing angles have OPPOSITE"
                       " signs" << std::endl
                    << "   (angle_x " << ax1 << ", " << ax2
                    << " ; angle_y " << ay1 << ", " << ay2 << ")."
                    << std::endl
                    << "   Equal signs are what make a crossing here, since"
                       " zeta_2 = -zeta_1 at a given" << std::endl
                    << "   lab z. Opposite signs tilt both bunches the same"
                       " way -- a different geometry" << std::endl
                    << "   that the derived tilt and crab waist cannot represent" << std::endl
                    << "   because both are built"
                       " from the summed magnitude."
                    << std::endl;
          exit(1);
        }

      if (c.waist_mode != WAIST_NONE && (ax1 < 0.0 || ax2 < 0.0))
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": waist_mode with a NEGATIVE angle_x ("
                    << ax1 << ", " << ax2 << ")." << std::endl
                    << "   g(x) = -fraction*x/tan(theta) is built from the"
                       " declared magnitude and nothing" << std::endl
                    << "   reads the sign, so this deck would get the"
                       " opposite waist shift. Use positive angle_x." << std::endl;
          exit(1);
        }

      const double from_beams =
          std::fabs(ax1) + std::fabs(ax2)
        + std::fabs(ay1) + std::fabs(ay2);
      const double declared = c.full_crossing_angle_rad;
      if (from_beams > 0.0 &&
          std::fabs(declared - from_beams) > 1.0e-6 * (1.0 + from_beams))
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": crab_full_crossing_angle (" << declared
                    << ") disagrees with the beams' own angles ("
                    << from_beams << ")." << std::endl
                    << "   The derived tilt comes from the former, so the"
                       " crab would not cancel the latter and" << std::endl
                    << "   luminosity would be quietly low. Make them agree,"
                       " or set crab_target_tilt" << std::endl
                    << "   explicitly if the difference is deliberate."
                    << std::endl;
          exit(1);
        }
    }
}

// Apply the minimum slice-smear admission guard; convergence is a separate test.
void FEEBL::check_incoming_tilt_resolution(
    int b,
    const INCOMING_MAP_CHAIN& chain,
    const BEAM_PARAMETERS& bp,
    float delta_z_nm)
{
  const double tilt = chain.ip_tilt();
  if (tilt != 0.0 && delta_z_nm > 0.0)
    {
      const double SMEAR_LIMIT = 0.8;
      const double sigma = (chain.tilt_plane() == 1) ? bp.sigma_y() : bp.sigma_x();
      const double smear = std::fabs(tilt) * (double)delta_z_nm;
      if (sigma > 0.0 && smear > SMEAR_LIMIT * sigma)
        {
          const int nz  = grid_.get_n_cell_z();
          const int req = (int)std::ceil(nz * smear / (SMEAR_LIMIT * sigma));
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": n_z = " << nz << " cannot resolve the crab tilt."
                    << std::endl
                    << "   the tilt varies by " << smear << " nm across one "
                    << delta_z_nm << " nm slice, which is "
                    << (smear / sigma) << " sigma_"
                    << ((chain.tilt_plane() == 1) ? 'y' : 'x')
                    << " (" << sigma << " nm), against a limit of "
                    << SMEAR_LIMIT << "." << std::endl
                    << "   resolve this minimum slicing criterion: use n_z >= "
                    << req << ", or reduce the tilt." << std::endl
                    << "   This is an admission floor, not a convergence certificate."
                    << std::endl
                    << "   Check luminosity and spectrum convergence as n_z is increased."
                    << std::endl;
          exit(1);
        }
    }
}

// Convert ordered native particles to canonical metres and normalized momenta.
CANONICAL_BUNCH FEEBL::read_incoming_canonical_bunch(
    int b,
    const IncomingMapConfig& c,
    BEAM& beam,
    const BEAM_PARAMETERS& bp,
    int nsl,
    std::size_t n)
{
  CANONICAL_BUNCH bunch;
  bunch.set_longitudinal(LongitudinalConvention(c.longitudinal_sign, 0.0));
  bunch.set_reference_energy(bp.ebeam());
  bunch.resize(n);

  std::size_t i = 0;
  for (int sl = 0; sl < nsl; sl++)
    {
      std::vector<PARTICLE*>& v = beam.getParticleVector(sl);
      for (std::size_t k = 0; k < v.size(); k++, i++)
        {
          float x, y, vx, vy;
          v[k]->XYposition(x, y);
          v[k]->velocities(vx, vy);
          if (!bunch.from_feebl(i, x, y, v[k]->z(), vx, vy, v[k]->energy()))
            {
              std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                        << ": canonical conversion refused (undeclared "
                           "convention or non-positive reference energy)"
                        << std::endl;
              exit(1);
            }
        }
    }
  return bunch;
}

// Report the actual incoming energy spread and horizontal dispersive size.
void FEEBL::report_incoming_dispersion(
    int b,
    const INCOMING_MAP_CHAIN& chain,
    const BEAM_PARAMETERS& bp,
    const CANONICAL_BUNCH& bunch,
    std::size_t n)
{
  if (chain.ip_dispersion() != 0.0)
    {
      double s1 = 0.0, s2 = 0.0;
      for (std::size_t q = 0; q < n; q++)
        { const double d = bunch.delta(q); s1 += d; s2 += d * d; }
      const double mean = s1 / (double)n;
      const double sd = std::sqrt(std::fmax(s2 / (double)n - mean * mean, 0.0));
      const double disp = std::fabs(chain.ip_dispersion()) * sd;
      const double sx = (double)bp.sigma_x();
      std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                << ": sigma_delta = " << sd
                << ", D*sigma_delta = " << disp * 1.0e6 << " um";
      if (sx > 0.0)
        std::cerr << " = " << (disp * 1.0e9 / sx) << " sigma_x";
      std::cerr << std::endl;
      if (!(sd > 0.0))
        std::cerr << "   WARNING: this bunch is MONOENERGETIC, so the"
                     " dispersion map moves nothing." << std::endl
                  << "   Check the actual incoming energy distribution and"
                     " which_espread before interpreting this map."
                  << std::endl;
    }
}

// Write mapped kinematics in original particle order and measure the direction change.
double FEEBL::write_incoming_canonical_bunch(
    int b,
    BEAM& beam,
    const CANONICAL_BUNCH& bunch,
    int nsl)
{
  double worst_dtheta = 0.0;

  std::size_t i = 0;
  for (int sl = 0; sl < nsl; sl++)
    {
      std::vector<PARTICLE*>& v = beam.getParticleVector(sl);
      for (std::size_t k = 0; k < v.size(); k++, i++)
        {

          float e = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f, vx = 0.0f, vy = 0.0f;
          if (!bunch.to_feebl(i, x, y, z, vx, vy, e))
            {
              std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                        << ": canonical back-conversion refused at particle "
                        << i << std::endl;
              exit(1);
            }

          float vx0 = 0.0f, vy0 = 0.0f;
          v[k]->velocities(vx0, vy0);
          const double dvx = (double)vx - (double)vx0;
          const double dvy = (double)vy - (double)vy0;
          const double dth = std::sqrt(dvx*dvx + dvy*dvy);
          if (dth > worst_dtheta) worst_dtheta = dth;

          v[k]->setParticle(x, y, z, vx, vy, e);
        }
    }
  return worst_dtheta;
}

// Reject unsupported field-region spin maps and report the existing turn-scale guard.
void FEEBL::check_incoming_spin_contract(
    int b,
    const IncomingMapConfig& c,
    const BEAM_PARAMETERS& bp,
    double worst_dtheta)
{
  if (switches.get_bmt_precession() || switches.get_ST_spin_flip()
      || switches.get_do_bds_spin_rotation())
    {
      const bool cavity = (c.crab_mode == CRAB_RF_HARMONICS);
      const bool waist  = (c.waist_mode != WAIST_NONE);
      if (cavity || waist)
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": " << (cavity ? "crab_mode=2 (RF cavity)"
                                       : "waist_mode")
                    << " represents a physical field region," << std::endl
                    << "   and this run has spin enabled. Its spin transport"
                       " requires a field-resolved E/B model."
                    << std::endl
                    << "   The canonical orbit map does not provide that model."
                    << std::endl
                    << "   crab_mode=1 describes an already-crabbed bunch at the IP;"
                    << std::endl
                    << "   for that mode, polar_x/y/z specifies polarization at the IP."
                    << std::endl;
          exit(1);
        }
    }

  if (switches.get_bmt_precession() || switches.get_ST_spin_flip()
      || switches.get_do_bds_spin_rotation())
    {
      const double agamma = PHYSTOOLS::BformFunction(0.0)
                          * (double)bp.gamma_beam();
      const double bound  = (1.0 + agamma) * worst_dtheta;
      std::cout << " incoming maps, beam " << b
                << " : largest momentum turn " << worst_dtheta
                << " rad, a*gamma = " << agamma
                << ", magnetic-deflector rotation scale = " << bound
                << " rad" << std::endl;

      if (bound > 1.0e-2)
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": this map turns the momentum by " << worst_dtheta
                    << " rad, which at a*gamma = " << agamma << std::endl
                    << "   gives a magnetic-deflector rotation scale of " << bound
                    << " rad. This exceeds the incoming-map approximation guard."
                    << std::endl
                    << "   A canonical orbit map alone does not determine spin transport."
                    << std::endl
                    << "   Use crab_mode=1 (the IP tilt), which re-describes"
                       " an already-crabbed bunch" << std::endl
                    << "   and leaves the momentum essentially untouched, and"
                       " give polar_x/y/z as the" << std::endl
                    << "   polarization AT THE IP -- after the upstream"
                       " cavity, which FEEBL does not model." << std::endl;
          exit(1);
        }
      if (bound > 1.0e-4)
        {
          std::ostringstream w;
          w << " FEEBL: WARNING : beam " << b << ": the incoming map turns"
               " the momentum by " << worst_dtheta << " rad." << std::endl
            << "   At a*gamma = " << agamma << " the magnetic-deflector rotation scale is"
               " " << bound << " rad." << std::endl
            << "   This is a diagnostic scale, not a computed spin rotation or bound."
            << std::endl
            << "   Input polarization is defined at the IP; upstream spin transport is external."
            << std::endl;
          std::cerr << w.str();
          std::cout << w.str();
        }
    }
}

void FEEBL::apply_incoming_maps(float delta_z_nm)
{
  for (int b = 1; b <= 2; b++)
    {
      IncomingMapConfig c = configure_incoming_map(b);
      if (!c.active()) continue;

      check_incoming_crossing_geometry(b, c);

      INCOMING_MAP_CHAIN chain;
      if (!chain.configure(c))
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": " << chain.error() << std::endl;
          exit(1);
        }

      double defect = 0.0;
      if (!chain.verify_symplectic(defect))
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": symplectic defect " << defect
                    << " exceeds waist_symplecticity_tolerance "
                    << c.symplecticity_tolerance << std::endl;
          exit(1);
        }

      BEAM& beam = (b == 1) ? beam1_ : beam2_;
      const BEAM_PARAMETERS& bp = (b == 1) ? beam_parameters1_ : beam_parameters2_;

      check_incoming_tilt_resolution(b, chain, bp, delta_z_nm);

      const int nsl = beam.number_of_slices();
      std::size_t n = 0;
      for (int sl = 0; sl < nsl; sl++) n += beam.getParticleVector(sl).size();
      if (n == 0) continue;

      CANONICAL_BUNCH bunch = read_incoming_canonical_bunch(b, c, beam, bp, nsl, n);

      report_incoming_dispersion(b, chain, bp, bunch, n);

      std::string why;
      if (!chain.apply(bunch, why))
        {
          std::cerr << " FEEBL::apply_incoming_maps: beam " << b
                    << ": " << why << std::endl;
          exit(1);
        }

      const double worst_dtheta = write_incoming_canonical_bunch(b, beam, bunch, nsl);

      check_incoming_spin_contract(b, c, bp, worst_dtheta);

      std::cout << " incoming maps applied to beam " << b
                << " : crab_mode=" << c.crab_mode
                << " waist_mode=" << c.waist_mode
                << " sign=" << c.longitudinal_sign
                << " particles=" << n
                << " worst symplectic defect=" << defect << std::endl;
    }
}

void FEEBL::set_beams_and_grids()
{
  std::vector<float> size_x, size_y;
  float size_z;
  int updated_n_cell_x = grid_.get_n_cell_x();
  int updated_n_cell_y = grid_.get_n_cell_y();
  int load_beam = switches.get_load_beam();
  if (load_beam) 
    {
      switch (load_beam) 
	{
	case 1:
	  std::cerr << " FEEBL::set_beams_and_grids(): load_beam = 1 not yet implemented " << std::endl;
	  exit(1);
	case 2:
	  std::cerr << " FEEBL::set_beams_and_grids(): load_beam = 2 not yet implemented " << std::endl;
	  exit(1);
	case 3:
	  {
	    // these pointers will be deleted a the end of loading particles, 
	    // by the beam objects.
	    BEAM_FROM_FILE* bffe = new BEAM_FROM_FILE(get_elfilename());
	    BEAM_FROM_FILE* bffp = new BEAM_FROM_FILE(get_posfilename());
	    xycuts_for_grids(*bffe, *bffp, switches.get_extra_grids()+1, size_x, size_y, updated_n_cell_x, updated_n_cell_y);
	    size_z = zcut_for_grids(*bffe, *bffp);

	    transfer_loaded_particles_in_beam( beam1_, beam_parameters1_, bffe, size_z); 
	    transfer_loaded_particles_in_beam( beam2_, beam_parameters2_, bffp, size_z); 
	    init_grid_phys(beam_parameters1_.n_particles(), beam_parameters2_.n_particles(),size_x, size_y, size_z, updated_n_cell_x, updated_n_cell_y);
	       if (switches.get_loaded_beam_is_pretransformed())
		 {
		   std::cout << " loaded beam declared pre-transformed:"
		                " skipping roll, crossing angle, offsets and"
		                " incoming maps" << std::endl;
		 }
	       else beam_displacements_from_data(grid_.get_delta_z()); 
	  }
	  break;

	default:
	  std::cerr << "load_beam set to incorrect value: " << std::endl;
	  std::cerr << " load_beam= " << load_beam << std::endl;
	  std::cerr << "must be within [0..2] " << std::endl;
	  exit(1);

	} 
    }
  else 
    {


      //      int  n_cell_z= params_.readIValue("n_z");
      int  n_cell_z = grid_.get_n_cell_z();
      if (n_cell_z <= 0) 
	{
	  std::cerr << " FEEBL::set_beams_and_grids() : ERROR : n_cell_z = " << n_cell_z << std::endl;
	  exit(1);
	}
      size_z = cut_z_from_data(switches.get_automatic_grid_sizing());
      float deltaz=2.0*size_z/((float)n_cell_z);
      beam1_.init_particles(grid_.get_nb_macroparticles(1),deltaz, switches.get_charge_symmetric(), switches.get_do_bds_spin_rotation());
      beam2_.init_particles(grid_.get_nb_macroparticles(2),deltaz, switches.get_charge_symmetric(), switches.get_do_bds_spin_rotation());

      // Give both beams' particles contiguous
      // position slots. Value-preserving, so results are unchanged.
      if (!switches.get_incoming_map_reslice())
        {
          beam1_.adopt_xy();
          beam2_.adopt_xy();
        }

      beam_displacements_from_data(deltaz);
      if (switches.get_incoming_map_reslice())
        {
          beam1_.adopt_xy();
          beam2_.adopt_xy();
        }


      xycuts_for_grids( beam1_.particle_beam(), beam2_.particle_beam(),switches.get_extra_grids()+1, size_x, size_y, updated_n_cell_x, updated_n_cell_y);
      init_grid_phys(beam_parameters1_.n_particles(), beam_parameters2_.n_particles(), size_x,size_y,size_z,updated_n_cell_x, updated_n_cell_y);
    }

  if(switches.get_load_photon()) 
    {
      beam1_.load_photons(photon_input_file_, 1, grid_.get_delta_z(), grid_.get_max_z(), grid_.get_n_cell_z());
      beam2_.load_photons(photon_input_file_,2,  grid_.get_delta_z(), grid_.get_max_z(), grid_.get_n_cell_z());
    }

  grid_.enable_sparse_cells(switches.get_sparse_cells() != 0);
  grid_.set_photon_polarisation(switches.get_photon_polarisation());
  grid_.set_coherent_model(switches.get_coherent_model());
  if (switches.get_sparse_cells())
    std::cerr
        << " FEEBL: sparse-cell preflight committed: active-index-v1, "
           "ascending cell order, stable within-cell insertion order"
        << std::endl;

}

void FEEBL::printInitialBeam(const BEAM& beam)
{
  float bidonx, bidony, bidz, bidsigz;
  float bidsigx, bidsigy;
  float bidbetax, bidbetay;
  std::cout << " ******* initial beam characteristics : " << beam.label() << " ************** " << std::endl;
  beam.transverse_sigmas(bidsigx, bidsigy);
  beam.emittances(bidonx, bidony);
  beam.beamZRms(bidz, bidsigz);
  std::cout << " sigmax = " << bidsigx << " sigmay = " << bidsigy << std::endl;
  std::cout << " emittx =" << bidonx << " emitty = " << bidony << std::endl;
  std::cout << " Z0 = " << bidz << " sigZ = " << bidsigz << std::endl;
  bidbetax =  bidsigx* bidsigx*1e-9/(bidonx*EMASS/beam1_.get_ebeam());
  bidbetay =  bidsigy* bidsigy*1e-9/(bidony*EMASS/beam1_.get_ebeam());
  std::cout << " betax = " << bidbetax << " betay = " << bidbetay << std::endl;	
  std::cout << " ------------------------------------------------- " << std::endl;
}


void FEEBL::set_output_data_and_files()
{
  if (switches.get_do_pairs() || switches.get_do_compt() || switches.get_do_bhabhas() || switches.get_load_event())
    {
      secondaries_.resize(grid_.get_n_cell_z());
    }
  if (switches.get_do_muons() )
    {
      muons_.resize(grid_.get_n_cell_z());
    }
}

void FEEBL::initialize_gpu2_fields()
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu_mode2_enabled()) return;
  const GPU2_OPTIONS& requested = backend.requested_gpu2_options();
  // A non-default primary mover is an explicit execution-method request.
  // Even allow_backend_fallback must not turn it into a different algorithm;
  // every preflight failure for such a card is a hard, pre-collision refusal.
  const bool hard_primary_mover_request =
      requested.primary_mover != "serial";
  const auto reject_preflight = [&](const std::string& reason) {
    for (std::size_t index = 0; index < gpu2_field_solvers_.size(); ++index)
      delete gpu2_field_solvers_[index];
    gpu2_field_solvers_.clear();
    gpu2_packed_xy1_.clear();
    gpu2_packed_xy2_.clear();
    gpu2_primary_fields1_.clear();
    gpu2_primary_fields2_.clear();
    gpu2_direct_deposition_ = false;
    gpu2_primary_scope_ = false;
    gpu2_primary_beam1_in_ = 0;
    gpu2_primary_beam1_out_ = 0;
    gpu2_primary_beam2_in_ = 0;
    gpu2_primary_beam2_out_ = 0;
    if (backend.allow_fallback() && !hard_primary_mover_request)
      {
        const std::string exact_reason = "GPU2 preflight: " + reason;
        backend.runtime_fallback_to_serial(exact_reason);
        std::cerr << " FEEBL GPU2 preflight fallback: " << reason
                  << "; actual_backend=serial before collision" << std::endl;
        return;
      }
    gpu2_fatal(reason);
  };

  if (!gpu2_field_solvers_.empty())
    {
      reject_preflight("resident field solvers were initialized more than once");
      return;
    }
  if (switches.get_integration_method() != 2)
    {
      reject_preflight(
          "gpu_mode=2 fields/primary scopes require integration_method=2 "
          "(FFT)");
      return;
    }
  if (gridsPtr_.empty())
    {
      reject_preflight("no initialized main grid was available");
      return;
    }

  if (requested.scope != "auto" && requested.scope != "fields" &&
      requested.scope != "primary")
    {
      reject_preflight("gpu2_scope=" + requested.scope +
                       " is not implemented; this build provides fields and "
                       "the strictly gated primary scope");
      return;
    }
  const bool primary_scope = requested.scope == "primary";
  const bool direct_deposition = primary_scope ||
      requested.deposition == "atomic";
  if (requested.deposition != "auto" &&
      requested.deposition != "atomic")
    {
      reject_preflight("gpu2_deposition=" + requested.deposition +
                       " is not implemented; fields supports auto (host_rho) "
                       "or atomic, and primary supports auto/atomic resolved "
                       "to atomic");
      return;
    }
  if (primary_scope && backend.reduction_mode() != "fast")
    {
      reject_preflight(
          "gpu2_scope=primary requires reduction_mode=fast because its "
          "resident deposition is atomic");
      return;
    }
  if (direct_deposition && backend.reduction_mode() == "deterministic")
    {
      reject_preflight(
          "reduction_mode=deterministic requires sort_reduce, which is not "
          "implemented by the resident fields path; atomic execution refused");
      return;
    }
  if (direct_deposition && switches.get_electron_distribution_rho() != 2)
    {
      reject_preflight(
          "gpu2_deposition=atomic requires electron_distribution_rho=2");
      return;
    }
  if (primary_scope && switches.get_interpolation() != 2)
    {
      reject_preflight(
          "gpu2_scope=primary requires interpolation=2 (CIC)");
      return;
    }
  if (primary_scope &&
      (switches.get_extra_grids() != 0 || gridsPtr_.size() != 1))
    {
      reject_preflight(
          "gpu2_scope=primary requires extra_grids=0 and exactly the main "
          "grid; no compact extra-grid gather is implemented");
      return;
    }
  if (direct_deposition &&
      (switches.get_do_coherent() || switches.get_do_trident()))
    {
      reject_preflight(
          "resident atomic deposition does not support coherent/trident "
          "field-charge sources; use gpu2_scope=fields with auto deposition");
      return;
    }
  // Complete host-phi consumer allowlist after the primary mover is replaced:
  // coherent/trident primary-like particles, coherent photon conversion, and
  // tracked pair/muon transport.  The first gate above excludes the first two
  // consumers; the gate below excludes tracked secondary transport.  If a
  // future stage reads GRID::get_phi*(), it must be added here before primary
  // scope may claim host_phi_materialized=0.
  if (primary_scope &&
      (switches.get_track_pairs() || switches.get_track_muons()))
    {
      reject_preflight(
          "gpu2_scope=primary does not materialize host phi and therefore "
          "requires track_pairs=0 and track_muons=0");
      return;
    }
  if (primary_scope && switches.get_do_prod())
    {
      reject_preflight(
          "gpu2_scope=primary requires do_prod=0");
      return;
    }
  if (requested.streams > 1)
    {
      reject_preflight(
          "gpu2_streams>1 is not implemented by the fields/primary plans");
      return;
    }
  if (requested.graphs == "on")
    {
      reject_preflight(
          "gpu2_graphs=on is not implemented by the fields/primary plans");
      return;
    }
  if (requested.fusion == "on")
    {
      reject_preflight(
          "gpu2_fusion=on is not implemented by the fields/primary plans");
      return;
    }
  if (requested.batch_slices > 1)
    {
      reject_preflight(
          "gpu2_batch_slices>1 is unsafe for the collision dependency "
          "schedule and is not implemented");
      return;
    }
  if (requested.profile != "summary")
    {
      reject_preflight(
          "gpu2_profile=" + requested.profile +
          " is not implemented by the fields/primary plans; use summary");
      return;
    }

  int actual_host_threads = backend.actual_gpu2_options().host_threads;
  if (actual_host_threads < 1) actual_host_threads = 1;
  std::string actual_primary_mover = "serial";
  if (primary_scope)
    {
      actual_primary_mover = requested.primary_mover;
      // `auto` is a transparent dispatch policy, not a hidden benchmark
      // claim: one worker uses the existing serial mover and a larger team
      // uses the persistent pool.  OpenMP remains an explicit candidate so
      // both implementations can be measured on identical cards.
      if (actual_primary_mover == "auto")
        actual_primary_mover = actual_host_threads > 1 ? "pool" : "serial";
      if (actual_primary_mover == "serial" && actual_host_threads != 1)
        {
          reject_preflight(
              "gpu2_primary_mover=serial requires gpu2_host_threads=0/1; "
              "no host threads are ignored silently");
          return;
        }
      const bool parallel_primary = actual_primary_mover != "serial";
      if (parallel_primary &&
          (switches.get_do_beamstrahlung() != 0 ||
           switches.get_ST_spin_flip() != 0 ||
           switches.get_bmt_precession() != 0 ||
           switches.get_do_trident() != 0 ||
           switches.get_do_coherent() != 0 ||
           switches.get_do_prod() != 0 ||
           switches.get_extra_grids() != 0 ||
           switches.get_track_pairs() != 0 ||
           switches.get_track_muons() != 0 ||
           switches.get_interpolation() != 2 ||
           requested.deposition == "shared_tile" ||
           requested.deposition == "sort_reduce"))
        {
          reject_preflight(
              "parallel gpu2 primary movement requires interpolation=2, "
              "do_eloss=0, ST_spin_flip=0, bmt_precession=0, do_trident=0, "
              "do_coherent=0, do_prod=0, extra_grids=0, track_pairs=0, "
              "track_muons=0, and the atomic CIC compact-primary plan");
          return;
        }
    }
  else
    {
      if (requested.primary_mover != "serial")
        {
          reject_preflight(
              "gpu2_primary_mover is valid only with gpu2_scope=primary");
          return;
        }
      if (actual_host_threads > 2)
        {
          reject_preflight(
              "gpu2_host_threads>2 is not implemented for scope=fields; "
              "the only dependency-safe hybrid plan uses two disjoint "
              "primary CIC depositions");
          return;
        }
      if (actual_host_threads == 2 &&
          switches.get_electron_distribution_rho() != 2)
        {
          reject_preflight(
              "gpu2_host_threads=2 requires electron_distribution_rho=2; "
              "only the primary CIC deposition pair is implemented");
          return;
        }
    }
#ifdef FEEBL_HAVE_OPENMP
  if ((!primary_scope && actual_host_threads == 2) ||
      (primary_scope && actual_primary_mover == "openmp"))
    {
      // Make the requested team size a measured preflight fact.  OpenMP may
      // otherwise silently reduce a num_threads region under dynamic teams or
      // OMP_THREAD_LIMIT while provenance continues to advertise the request.
      omp_set_dynamic(0);
      int observed_team_size = 1;
#pragma omp parallel num_threads(actual_host_threads) shared(observed_team_size)
      {
#pragma omp single
        observed_team_size = omp_get_num_threads();
      }
      if (observed_team_size != actual_host_threads)
        {
          std::ostringstream reason;
          reason << "gpu2_host_threads=" << actual_host_threads
                 << " requested, but OpenMP preflight "
                 << "formed a team of " << observed_team_size;
          reject_preflight(reason.str());
          return;
        }
    }
#else
  if (primary_scope && actual_primary_mover == "openmp")
    {
      reject_preflight(
          "gpu2_primary_mover=openmp requires an OpenMP-enabled build");
      return;
    }
#endif
  if (!primary_scope && direct_deposition && actual_host_threads != 1)
    {
      reject_preflight(
          "gpu2_host_threads>1 has no implemented work when "
          "gpu2_scope=fields,gpu2_deposition=atomic moves both primary CIC "
          "deposits to device");
      return;
    }

  feebl::gpu::ResidentFieldOptions solver_options;
  solver_options.device = backend.gpu_device();
  // auto deliberately starts pageable.  Explicit `on` remains a genuine,
  // benchmarkable staging choice rather than a provenance-only switch.
  solver_options.pinned_io = requested.pinned_io == "on";
  solver_options.memory_fraction = requested.memory_fraction;

  std::size_t maximum_particles1 = 0;
  std::size_t maximum_particles2 = 0;
  if (direct_deposition)
    {
      for (int slice = 0; slice < grid_.get_n_cell_z(); ++slice)
        {
          maximum_particles1 = std::max(
              maximum_particles1,
              beam1_.getParticleVector(slice).size());
          maximum_particles2 = std::max(
              maximum_particles2,
              beam2_.getParticleVector(slice).size());
        }
    }

  std::size_t total_resident_bytes = 0;
  std::size_t first_context_resident_bytes = 0;
  std::size_t device_total_bytes = 0;
  long double memory_budget = 0.0L;
  int first_nx = 0;
  int first_ny = 0;
  for (std::size_t index = 0; index < gridsPtr_.size(); ++index)
    {
      GENERAL_GRID& grid = *gridsPtr_[index];
      const int nx = grid.get_n_cell_x();
      const int ny = grid.get_n_cell_y();
      const std::size_t expected_spectrum =
          8U * static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
      if (grid.get_fft_spectrum_size() < 0 ||
          static_cast<std::size_t>(grid.get_fft_spectrum_size()) !=
              expected_spectrum)
        {
          std::ostringstream error;
          error << "grid " << index << " exposes "
                << grid.get_fft_spectrum_size() << " Green-spectrum scalars; "
                << expected_spectrum << " are required for the exact 2nx*2ny "
                << "interleaved FFT layout";
          reject_preflight(error.str());
          return;
        }

      // EXTRA_GRID preserves the main grid's cell dimensions. Once the first
      // exact cuFFT plan/workspace and optional particle reservation have been
      // measured, admit every further equal-shaped context against the global
      // card budget before allocating it.
      if (index > 0)
        {
          if (nx != first_nx || ny != first_ny)
            {
              reject_preflight(
                  "multi-grid GPU2 pre-admission requires identical main and "
                  "extra-grid cell dimensions");
              return;
            }
          if (first_context_resident_bytes >
              static_cast<std::size_t>(-1) - total_resident_bytes ||
              static_cast<long double>(total_resident_bytes) +
                  static_cast<long double>(first_context_resident_bytes) >
                  memory_budget)
            {
              std::ostringstream error;
              error << "aggregate field contexts would require "
                    << (static_cast<long double>(total_resident_bytes) +
                        static_cast<long double>(first_context_resident_bytes))
                    << " bytes before allocation, exceeding "
                    << "gpu2_memory_fraction budget "
                    << static_cast<std::size_t>(memory_budget);
              reject_preflight(error.str());
              return;
            }
        }

      feebl::gpu::ResidentFieldSolver* solver =
          new feebl::gpu::ResidentFieldSolver;
      std::string error;
      if (!solver->initialize(solver_options, nx, ny,
                              grid.get_fft_spectrum(), error))
        {
          delete solver;
          reject_preflight("grid " + std::to_string(index) +
                           " CUDA/cuFFT initialization failed: " + error);
          return;
        }
      gpu2_field_solvers_.push_back(solver);
      if (direct_deposition)
        {
          if (!solver->reserve_particle_capacity(
                  maximum_particles1, maximum_particles2, error))
            {
              reject_preflight(
                  "grid " + std::to_string(index) +
                  " resident particle-capacity reservation failed: " + error);
              return;
            }
        }
      const feebl::gpu::ResidentFieldStats stats =
          solver->stats();
      if (stats.device_total_bytes == 0)
        {
          reject_preflight("CUDA reported zero total device memory");
          return;
        }
      if (index == 0)
        {
          device_total_bytes = stats.device_total_bytes;
          memory_budget =
              static_cast<long double>(requested.memory_fraction) *
              static_cast<long double>(device_total_bytes);
          first_context_resident_bytes = stats.resident_bytes;
          first_nx = nx;
          first_ny = ny;
        }
      else if (stats.device_total_bytes != device_total_bytes ||
               stats.resident_bytes != first_context_resident_bytes)
        {
          reject_preflight(
              "equal-shaped resident solvers returned inconsistent memory "
              "requirements on the same device");
          return;
        }
      if (stats.resident_bytes >
          static_cast<std::size_t>(-1) - total_resident_bytes)
        {
          reject_preflight("aggregate resident-byte accounting overflowed");
          return;
        }
      // resident_bytes already includes the explicit cuFFT workspace.
      total_resident_bytes += stats.resident_bytes;
    }

  if (direct_deposition)
    {
      if (maximum_particles1 > gpu2_packed_xy1_.max_size() / 2 ||
          maximum_particles2 > gpu2_packed_xy2_.max_size() / 2)
        {
          reject_preflight(
              "primary slice size overflows compact host-buffer indexing");
          return;
        }
      try
        {
          if (!beam1_.xy_adopted())
            gpu2_packed_xy1_.resize(2 * maximum_particles1);
          if (!beam2_.xy_adopted())
            gpu2_packed_xy2_.resize(2 * maximum_particles2);
          if (primary_scope)
            {
              gpu2_primary_fields1_.resize(2 * maximum_particles1);
              gpu2_primary_fields2_.resize(2 * maximum_particles2);
            }
        }
      catch (const std::bad_alloc&)
        {
          reject_preflight(
              "could not preallocate compact primary host buffers");
          return;
        }
    }

  GPU2_OPTIONS plan = requested;
  plan.mode = 2;
  plan.scope = primary_scope ? "primary" : "fields";
  plan.deposition = direct_deposition ? "atomic" : "host_rho";
  plan.fft = "cufft";
  plan.streams = 1;
  plan.graphs = "off";
  plan.fusion = "off";
  plan.batch_slices = 1;
  plan.pinned_io = solver_options.pinned_io ? "on" : "off";
  plan.host_threads = actual_host_threads;
  plan.primary_mover = actual_primary_mover;
  try
    {
      backend.record_gpu2_plan(plan);
    }
  catch (const std::exception& error)
    {
      reject_preflight(
          std::string("could not commit the concrete fields plan: ") +
          error.what());
      return;
    }
  gpu2_direct_deposition_ = direct_deposition;
  gpu2_primary_scope_ = primary_scope;

  const std::string primary_update = !primary_scope
      ? "host_phi_mover"
      : (plan.primary_mover == "serial"
             ? "host_serial"
             : (plan.primary_mover == "openmp"
                    ? "host_openmp_static_combined"
                    : "host_pool_static_combined"));
  const std::string primary_side_effects = !primary_scope
      ? "host_phi_mover"
      : (plan.primary_mover == "serial" ? "host_serial"
                                         : "disabled_by_parallel_contract");
  const std::string primary_rng =
      primary_scope && plan.primary_mover != "serial"
          ? "unused_by_parallel_contract" : "legacy_host_ordered";

  std::cerr << " FEEBL GPU2 preflight: requested_scope=" << requested.scope
            << " actual_scope=" << plan.scope
            << " deposition=" << plan.deposition
            << " fft=cufft"
            << " streams=1 graphs=off fusion=off batch_slices=1"
            << " pinned_io=" << plan.pinned_io
            << " host_threads=" << plan.host_threads
            << " primary_mover_requested=" << requested.primary_mover
            << " primary_mover_actual=" << plan.primary_mover
            << " host_boundary="
            << (primary_scope
                    ? "compact_primary_fields"
                    : (direct_deposition
                           ? "compact_metrics_then_phi_materialization"
                           : "phi_materialization"))
            << " primary_update="
            << primary_update
            << " spin_transport="
            << primary_side_effects
            << " beamstrahlung="
            << primary_side_effects
            << " rng=" << primary_rng
            << " host_phi_materialized=" << (primary_scope ? 0 : 1)
            << " solvers=" << gpu2_field_solvers_.size()
            << " resident_bytes=" << total_resident_bytes
            << " memory_budget_bytes="
            << static_cast<std::size_t>(memory_budget) << std::endl;
  std::cerr << " FEEBL GPU2 device: "
            << gpu2_field_solvers_[0]->device_description() << std::endl;
  std::cerr << " FEEBL GPU2 runtime: "
            << gpu2_field_solvers_[0]->runtime_description() << std::endl;
}

double FEEBL::deposit_gpu2_fields(int slice1, int slice2)
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!gpu2_direct_deposition_ ||
      !backend.gpu2_plan_committed())
    gpu2_fatal("resident direct deposition entered without its preflight plan");

  const std::vector<PARTICLE*>& particles1 =
      beam1_.getParticleVector(slice1);
  const std::vector<PARTICLE*>& particles2 =
      beam2_.getParticleVector(slice2);
  const float* xy1 = NULL;
  const float* xy2 = NULL;
  if (beam1_.xy_adopted() &&
      beam1_.slice_xy_count(static_cast<std::size_t>(slice1)) ==
          particles1.size())
    xy1 = beam1_.slice_xy(static_cast<std::size_t>(slice1));
  else
    {
      if (gpu2_packed_xy1_.size() < 2 * particles1.size())
        {
          std::cerr << " FEEBL GPU2 execution ERROR: beam 1 slice exceeds "
                       "preflight host packing capacity" << std::endl;
          exit(1);
        }
      for (std::size_t particle = 0; particle < particles1.size(); ++particle)
        particles1[particle]->XYposition(gpu2_packed_xy1_[2 * particle],
                                         gpu2_packed_xy1_[2 * particle + 1]);
      if (!particles1.empty()) xy1 = &gpu2_packed_xy1_[0];
    }
  if (beam2_.xy_adopted() &&
      beam2_.slice_xy_count(static_cast<std::size_t>(slice2)) ==
          particles2.size())
    xy2 = beam2_.slice_xy(static_cast<std::size_t>(slice2));
  else
    {
      if (gpu2_packed_xy2_.size() < 2 * particles2.size())
        {
          std::cerr << " FEEBL GPU2 execution ERROR: beam 2 slice exceeds "
                       "preflight host packing capacity" << std::endl;
          exit(1);
        }
      for (std::size_t particle = 0; particle < particles2.size(); ++particle)
        particles2[particle]->XYposition(gpu2_packed_xy2_[2 * particle],
                                         gpu2_packed_xy2_[2 * particle + 1]);
      if (!particles2.empty()) xy2 = &gpu2_packed_xy2_[0];
    }

  double main_fine_lumi_dot = 0.0;
  for (std::size_t index = 0; index < gridsPtr_.size(); ++index)
    {
      GENERAL_GRID& grid = *gridsPtr_[index];
      feebl::gpu::ResidentDepositResult result;
      feebl::gpu::ResidentFieldStats statistics;
      std::string error;
      if (!gpu2_field_solvers_[index]->deposit(
              xy1, particles1.size(), grid.get_macro_charge(1),
              xy2, particles2.size(), grid.get_macro_charge(2),
              grid.get_delta_x_inverse(), grid.get_delta_y_inverse(),
              grid.get_offset_x(), grid.get_offset_y(),
              switches.get_force_symmetric() != 0,
              result, statistics, error))
        {
          std::cerr << " FEEBL GPU2 execution ERROR: grid " << index
                    << " resident deposition failed after preflight: "
                    << error << std::endl;
          exit(1);
        }
      if (statistics.last_device_work)
        backend.record_gpu2_execution(
            gpu2_field_solvers_[index]->device_description());
      if (index == 0)
        {
          grid_.add_gpu2_distribution_counts(
              result.beam1_in, result.beam1_out,
              result.beam2_in, result.beam2_out);
          main_fine_lumi_dot = result.fine_lumi_dot;
        }
    }
  return main_fine_lumi_dot;
}

void FEEBL::solve_gpu2_fields(int slice1, int slice2)
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu_mode2_enabled())
    gpu2_fatal("resident field solve entered without an active mode-2 backend");
  if (!backend.gpu2_plan_committed() ||
      gpu2_field_solvers_.size() != gridsPtr_.size())
    gpu2_fatal("resident field solve entered before a complete preflight");

  for (std::size_t index = 0; index < gridsPtr_.size(); ++index)
    {
      GENERAL_GRID& grid = *gridsPtr_[index];
      feebl::gpu::ResidentFieldStats statistics;
      std::string error;
      bool solved = false;
      if (gpu2_primary_scope_)
        {
          if (index != 0 || gridsPtr_.size() != 1)
            gpu2_fatal(
                "primary compact gather reached a non-main grid after "
                "preflight");
          const std::size_t count1 =
              beam1_.getParticleVector(slice1).size();
          const std::size_t count2 =
              beam2_.getParticleVector(slice2).size();
          if (count1 > gpu2_primary_fields1_.size() / 2 ||
              count2 > gpu2_primary_fields2_.size() / 2)
            gpu2_fatal(
                "active slice exceeds preallocated compact field capacity");
          feebl::gpu::ResidentFieldGatherResult gather;
          solved = gpu2_field_solvers_[index]->solve_deposited_and_gather(
              switches.get_charge_sign(),
              grid.get_min_x(), grid.get_max_x(),
              grid.get_min_y(), grid.get_max_y(),
              count1 ? &gpu2_primary_fields1_[0] : NULL, count1,
              count2 ? &gpu2_primary_fields2_[0] : NULL, count2,
              gather, statistics, error);
          if (solved)
            {
              if (gather.beam1_in + gather.beam1_out != count1 ||
                  gather.beam2_in + gather.beam2_out != count2)
                gpu2_fatal(
                    "compact gather returned inconsistent in/out counts");
              gpu2_primary_beam1_in_ += gather.beam1_in;
              gpu2_primary_beam1_out_ += gather.beam1_out;
              gpu2_primary_beam2_in_ += gather.beam2_in;
              gpu2_primary_beam2_out_ += gather.beam2_out;
            }
        }
      else if (gpu2_direct_deposition_)
        solved = gpu2_field_solvers_[index]->solve_deposited(
            switches.get_charge_sign(),
            grid.get_phi1_mutable(), grid.get_phi2_mutable(),
            statistics, error);
      else
        solved = gpu2_field_solvers_[index]->solve(
            grid.get_rho1(), grid.get_rho2(),
            switches.get_charge_sign(),
            grid.get_phi1_mutable(), grid.get_phi2_mutable(), statistics,
            error);
      if (!solved)
        {
          std::cerr << " FEEBL GPU2 execution ERROR: grid " << index
                    << " field solve failed after preflight: " << error
                    << std::endl;
          // Never change algorithms in the middle of a collision.  Such a
          // fallback would combine two numerical trajectories while claiming
          // one backend and invalidate physics/RNG comparisons.
          exit(1);
        }
      if (statistics.last_device_work)
        backend.record_gpu2_execution(
            gpu2_field_solvers_[index]->device_description());
    }
}

std::string FEEBL::gpu2_fields_summary() const
{
  std::uint64_t calls = 0;
  std::uint64_t device_work_calls = 0;
  std::uint64_t deposition_calls = 0;
  std::uint64_t field_solve_calls = 0;
  std::uint64_t field_gather_calls = 0;
  double setup_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  std::uint64_t setup_h2d_bytes = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t h2d_operations = 0;
  std::uint64_t d2h_operations = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t allocations = 0;
  std::uint64_t plans = 0;
  std::size_t resident_bytes = 0;
  std::size_t workspace_bytes = 0;
  for (std::size_t index = 0; index < gpu2_field_solvers_.size(); ++index)
    {
      const feebl::gpu::ResidentFieldStats stats =
          gpu2_field_solvers_[index]->stats();
      calls += stats.calls;
      device_work_calls += stats.device_work_calls;
      deposition_calls += stats.deposition_calls;
      field_solve_calls += stats.field_solve_calls;
      field_gather_calls += stats.field_gather_calls;
      setup_ms += stats.setup_ms;
      h2d_ms += stats.h2d_ms;
      kernel_ms += stats.kernel_ms;
      d2h_ms += stats.d2h_ms;
      total_ms += stats.total_ms;
      setup_h2d_bytes += stats.setup_h2d_bytes;
      h2d_bytes += stats.h2d_bytes;
      d2h_bytes += stats.d2h_bytes;
      h2d_operations += stats.h2d_operations;
      d2h_operations += stats.d2h_operations;
      synchronizations += stats.synchronizations;
      allocations += stats.device_allocations;
      plans += stats.plan_creations;
      resident_bytes += stats.resident_bytes;
      workspace_bytes += stats.cufft_workspace_bytes;
    }
  std::ostringstream out;
  const std::string deposition = gpu2_direct_deposition_
      ? "atomic" : "host_rho";
  const PARALLEL_BACKEND& backend = parallel_backend();
  const std::string primary_mover = gpu2_primary_scope_
      ? backend.actual_gpu2_options().primary_mover : "inactive";
  const std::string primary_update = !gpu2_primary_scope_
      ? "host_phi_mover"
      : (primary_mover == "serial"
             ? "host_serial"
             : (primary_mover == "openmp"
                    ? "host_openmp_static_combined"
                    : "host_pool_static_combined"));
  const std::string primary_side_effects = !gpu2_primary_scope_
      ? "host_phi_mover"
      : (primary_mover == "serial" ? "host_serial"
                                    : "disabled_by_parallel_contract");
  out << " FEEBL GPU2 " << (gpu2_primary_scope_ ? "primary" : "fields")
      << " summary: scope=" << (gpu2_primary_scope_ ? "primary" : "fields")
      << " deposition=" << deposition
      << " fft=cufft host_boundary="
      << (gpu2_primary_scope_
              ? "compact_primary_fields"
              : (gpu2_direct_deposition_
                     ? "compact_metrics_then_phi_materialization"
                     : "phi_materialization"))
      << " primary_update="
      << primary_update
      << " spin_transport="
      << primary_side_effects
      << " beamstrahlung="
      << primary_side_effects
      << " rng="
      << (gpu2_primary_scope_ && primary_mover != "serial"
              ? "unused_by_parallel_contract" : "legacy_host_ordered")
      << " host_phi_materialized=" << (gpu2_primary_scope_ ? 0 : 1)
      << " primary_mover=" << primary_mover
      << " primary_mover_threads="
      << (gpu2_primary_scope_
              ? backend.actual_gpu2_options().host_threads : 0)
      << " primary_mover_calls=" << backend.gpu2_primary_mover_calls()
      << " primary_mover_particles="
      << backend.gpu2_primary_mover_particles()
      << " primary_mover_ranges=" << backend.gpu2_primary_mover_ranges()
      << " primary_mover_observed_threads="
      << backend.gpu2_primary_mover_observed_threads()
      << " solvers=" << gpu2_field_solvers_.size()
      << " calls=" << calls
      << " device_work_calls=" << device_work_calls
      << " deposition_calls=" << deposition_calls
      << " field_solve_calls=" << field_solve_calls
      << " field_gather_calls=" << field_gather_calls
      << " gather_beam1_in=" << gpu2_primary_beam1_in_
      << " gather_beam1_out=" << gpu2_primary_beam1_out_
      << " gather_beam2_in=" << gpu2_primary_beam2_in_
      << " gather_beam2_out=" << gpu2_primary_beam2_out_
      << " setup_ms=" << setup_ms
      << " h2d_ms=" << h2d_ms
      << " kernel_ms=" << kernel_ms
      << " d2h_ms=" << d2h_ms
      << " total_ms=" << total_ms
      << " setup_h2d_bytes=" << setup_h2d_bytes
      << " h2d_bytes=" << h2d_bytes
      << " d2h_bytes=" << d2h_bytes
      << " h2d_operations=" << h2d_operations
      << " d2h_operations=" << d2h_operations
      << " synchronizations=" << synchronizations
      << " device_allocations=" << allocations
      << " plan_creations=" << plans
      << " resident_bytes=" << resident_bytes
      << " cufft_workspace_bytes=" << workspace_bytes;
  return out.str();
}

struct FEEBL::LEGACY_WAVEFRONT_CONTEXT
{
  FEEBL* simulation;
  PHI_FLOAT* sor_parameter;
  const feebl::legacy_wavefront::Scheduler* scheduler;
};

// Defined before the GPU4 driver because that path may dispatch the same
// worker-private host-physics callback after each completed CUDA field batch.
struct FEEBL::WAVEFRONT_PHYSICS_CONTEXT
{
  FEEBL* simulation;
  const feebl::legacy_wavefront::Scheduler* scheduler;
  const feebl::modern::EncounterWave* wave;
  const std::vector<feebl::wavefront_fields::FieldPrefetch>* prefetch;
  std::size_t position_offset;
  std::size_t prefetch_offset;
  int legacy_time_base;
};

void FEEBL::initialize_gpu4_wave_batch()
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu_mode4_enabled()) return;
  if (gpu4_wave_solver_)
    throw std::logic_error("gpu_mode=4 wave solver initialized twice");
  if (gridsPtr_.size() != 1U)
    throw std::logic_error(
        "gpu_mode=4 initialization reached a multi-grid card after preflight");

  gpu4_batch_width_ = static_cast<std::size_t>(
      backend.actual_gpu2_options().batch_slices);
  gpu4_maximum_particles1_ = 0;
  gpu4_maximum_particles2_ = 0;
  for (int slice = 0; slice < beam1_.number_of_slices(); ++slice)
    gpu4_maximum_particles1_ = std::max(
        gpu4_maximum_particles1_, beam1_.getParticleVector(slice).size());
  for (int slice = 0; slice < beam2_.number_of_slices(); ++slice)
    gpu4_maximum_particles2_ = std::max(
        gpu4_maximum_particles2_, beam2_.getParticleVector(slice).size());

  const int nx = grid_.get_n_cell_x();
  const int ny = grid_.get_n_cell_y();
  const std::size_t expected_spectrum =
      8U * static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (grid_.get_fft_spectrum_size() < 0 ||
      static_cast<std::size_t>(grid_.get_fft_spectrum_size()) !=
          expected_spectrum)
    throw std::runtime_error(
        "gpu_mode=4 main grid does not expose the exact 2nx*2ny "
        "interleaved Green spectrum");

  feebl::gpu::WaveBatchOptions options;
  options.device = backend.gpu_device();
  options.batch_width = gpu4_batch_width_;
  options.maximum_particles1 = gpu4_maximum_particles1_;
  options.maximum_particles2 = gpu4_maximum_particles2_;
  options.memory_fraction = backend.actual_gpu2_options().memory_fraction;
  options.pinned_io = true;
  std::unique_ptr<feebl::gpu::WaveBatchedFieldSolver> candidate(
      new feebl::gpu::WaveBatchedFieldSolver());
  std::string error;
  if (!candidate->initialize(options, nx, ny, grid_.get_fft_spectrum(), error))
    throw std::runtime_error(
        "gpu_mode=4 wave-batched CUDA preflight failed: " + error);

  if (gpu4_batch_width_ >
          std::numeric_limits<std::size_t>::max() / 2U ||
      gpu4_maximum_particles1_ >
          std::numeric_limits<std::size_t>::max() /
              (2U * gpu4_batch_width_) ||
      gpu4_maximum_particles2_ >
          std::numeric_limits<std::size_t>::max() /
              (2U * gpu4_batch_width_))
    throw std::overflow_error("gpu_mode=4 host packing capacity overflow");
  try
    {
      gpu4_packed_xy1_.resize(
          2U * gpu4_batch_width_ * gpu4_maximum_particles1_);
      gpu4_packed_xy2_.resize(
          2U * gpu4_batch_width_ * gpu4_maximum_particles2_);
      gpu4_prefetch_.resize(gpu4_batch_width_);
    }
  catch (const std::bad_alloc&)
    {
      throw std::runtime_error(
          "gpu_mode=4 could not reserve fixed host batch buffers");
    }

  backend.record_gpu_mode4_plan(backend.actual_gpu2_options());
  gpu4_wave_solver_ = std::move(candidate);
  const feebl::gpu::WaveBatchStats stats = gpu4_wave_solver_->stats();
  std::cerr
      << " FEEBL GPU mode4 preflight: implementation="
         "encounter-dag-wave-batched-planmany-fields-v1"
      << " batch_width=" << gpu4_batch_width_
      << " maximum_particles1=" << gpu4_maximum_particles1_
      << " maximum_particles2=" << gpu4_maximum_particles2_
      << " fft=cufft_planmany deposition=atomic fusion=on stream=1"
      << " host_boundary=one_per_bounded_wave_bucket"
      << " host_physics=legacy_complete_ordered"
      << " resident_bytes=" << stats.resident_bytes
      << " cufft_workspace_bytes=" << stats.cufft_workspace_bytes
      << " memory_budget_bytes=" << stats.memory_budget_bytes
      << " setup_ms=" << stats.setup_ms
      << " device=\"" << gpu4_wave_solver_->device_description() << "\""
      << std::endl;
}

void FEEBL::beam_interaction_gpu_mode4(PHI_FLOAT* sor_parameter)
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu_mode4_enabled() || !backend.gpu2_plan_committed() ||
      !gpu4_wave_solver_ || !gpu4_wave_solver_->is_available())
    throw std::logic_error(
        "gpu_mode=4 collision entered without its committed CUDA batch plan");

  const int n_slice = grid_.get_n_cell_z();
  const int n_substeps = grid_.get_timestep();
  if (n_slice <= 0 || n_substeps <= 0)
    throw std::logic_error("gpu_mode=4 requires positive n_z and n_t");
  const feebl::legacy_wavefront::Scheduler scheduler(
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SubstepIndex>(n_substeps));
  LEGACY_WAVEFRONT_CONTEXT boundary_context;
  boundary_context.simulation = this;
  boundary_context.sor_parameter = sor_parameter;
  boundary_context.scheduler = &scheduler;
  WAVEFRONT_PHYSICS_CONTEXT physics_context;
  physics_context.simulation = this;
  physics_context.scheduler = &scheduler;
  physics_context.wave = NULL;
  physics_context.prefetch = &gpu4_prefetch_;
  physics_context.position_offset = 0;
  physics_context.prefetch_offset = 0;
  physics_context.legacy_time_base = 0;

  std::uint64_t encounters = 0;
  std::uint64_t batches = 0;
  std::size_t maximum_wave_width = 0;
  double field_prefetch_ms = 0.0;
  double parallel_physics_ms = 0.0;
  double ordered_commit_ms = 0.0;
  std::vector<feebl::gpu::WaveBatchEncounterInput> inputs(
      gpu4_batch_width_);
  std::vector<feebl::gpu::WaveBatchEncounterOutput> outputs(
      gpu4_batch_width_);

  for (const feebl::legacy_wavefront::DiagonalDescriptor& diagonal :
       scheduler.diagonals())
    {
      wavefront_begin_diagonal(&boundary_context, diagonal);
      for (feebl::modern::SubstepIndex substep = 0;
           substep < scheduler.dag().substep_count(); ++substep)
        {
          const feebl::modern::WaveIndex wave_index =
              diagonal.diagonal * scheduler.dag().substep_count() + substep;
          const feebl::modern::EncounterWave& wave =
              scheduler.dag().waves()[static_cast<std::size_t>(wave_index)];
          wavefront_begin_wave(&boundary_context, wave);
          physics_context.wave = &wave;
          maximum_wave_width = std::max(
              maximum_wave_width, wave.encounters.size());

          for (std::size_t begin = 0; begin < wave.encounters.size();
               begin += gpu4_batch_width_)
            {
              const std::size_t count = std::min<std::size_t>(
                  gpu4_batch_width_, wave.encounters.size() - begin);
              for (std::size_t slot = 0; slot < count; ++slot)
                {
                  const feebl::modern::EncounterNode& encounter =
                      scheduler.dag().node(wave.encounters[begin + slot]);
                  const int slice1 = static_cast<int>(encounter.beam1_slice);
                  const int slice2 = static_cast<int>(encounter.beam2_slice);
                  const std::vector<PARTICLE*>& particles1 =
                      beam1_.getParticleVector(slice1);
                  const std::vector<PARTICLE*>& particles2 =
                      beam2_.getParticleVector(slice2);
                  inputs[slot] = feebl::gpu::WaveBatchEncounterInput();
                  inputs[slot].count1 = particles1.size();
                  inputs[slot].count2 = particles2.size();
                  if (beam1_.xy_adopted() &&
                      beam1_.slice_xy_count(
                          static_cast<std::size_t>(slice1)) == particles1.size())
                    inputs[slot].xy1 = beam1_.slice_xy(
                        static_cast<std::size_t>(slice1));
                  else
                    {
                      float* packed = gpu4_packed_xy1_.empty() ? NULL :
                          &gpu4_packed_xy1_[
                              2U * slot * gpu4_maximum_particles1_];
                      for (std::size_t particle = 0;
                           particle < particles1.size(); ++particle)
                        particles1[particle]->XYposition(
                            packed[2 * particle], packed[2 * particle + 1]);
                      inputs[slot].xy1 = packed;
                    }
                  if (beam2_.xy_adopted() &&
                      beam2_.slice_xy_count(
                          static_cast<std::size_t>(slice2)) == particles2.size())
                    inputs[slot].xy2 = beam2_.slice_xy(
                        static_cast<std::size_t>(slice2));
                  else
                    {
                      float* packed = gpu4_packed_xy2_.empty() ? NULL :
                          &gpu4_packed_xy2_[
                              2U * slot * gpu4_maximum_particles2_];
                      for (std::size_t particle = 0;
                           particle < particles2.size(); ++particle)
                        particles2[particle]->XYposition(
                            packed[2 * particle], packed[2 * particle + 1]);
                      inputs[slot].xy2 = packed;
                    }

                  feebl::wavefront_fields::FieldPrefetch& prefetch =
                      gpu4_prefetch_[slot];
                  prefetch.fields1.resize(2U * particles1.size());
                  prefetch.fields2.resize(2U * particles2.size());
                  outputs[slot] = feebl::gpu::WaveBatchEncounterOutput();
                  outputs[slot].fields1 = prefetch.fields1.empty()
                      ? NULL : &prefetch.fields1[0];
                  outputs[slot].capacity1 = particles1.size();
                  outputs[slot].fields2 = prefetch.fields2.empty()
                      ? NULL : &prefetch.fields2[0];
                  outputs[slot].capacity2 = particles2.size();
                }

              const std::chrono::steady_clock::time_point prefetch_start =
                  std::chrono::steady_clock::now();
              std::string error;
              if (!gpu4_wave_solver_->execute(
                      &inputs[0], &outputs[0], count,
                      grid_.get_macro_charge(1), grid_.get_macro_charge(2),
                      grid_.get_delta_x_inverse(),
                      grid_.get_delta_y_inverse(),
                      grid_.get_offset_x(), grid_.get_offset_y(),
                      grid_.get_min_x(), grid_.get_max_x(),
                      grid_.get_min_y(), grid_.get_max_y(), false,
                      switches.get_charge_sign(), error))
                {
                  std::cerr << " FEEBL GPU mode4 execution ERROR after "
                            << batches << " completed batches: " << error
                            << "; mid-collision fallback is forbidden"
                            << std::endl;
                  exit(1);
                }
              field_prefetch_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - prefetch_start)
                      .count();

              // The bucket is the transaction boundary.  Validate every
              // encounter and every returned field before committing the
              // first host particle; a late-slot failure must never leave a
              // partially advanced wave.
              for (std::size_t slot = 0; slot < count; ++slot)
                {
                  const feebl::gpu::WaveBatchEncounterOutput& output =
                      outputs[slot];
                  const feebl::gpu::WaveBatchEncounterInput& input =
                      inputs[slot];
                  const bool deposit1_ok =
                      output.beam1_in <= input.count1 &&
                      output.beam1_out == input.count1 - output.beam1_in;
                  const bool deposit2_ok =
                      output.beam2_in <= input.count2 &&
                      output.beam2_out == input.count2 - output.beam2_in;
                  const bool gather1_ok =
                      output.gather_beam1_in <= input.count1 &&
                      output.gather_beam1_out ==
                          input.count1 - output.gather_beam1_in;
                  const bool gather2_ok =
                      output.gather_beam2_in <= input.count2 &&
                      output.gather_beam2_out ==
                          input.count2 - output.gather_beam2_in;
                  bool finite_fields = std::isfinite(output.fine_lumi_dot);
                  const feebl::wavefront_fields::FieldPrefetch& prefetch =
                      gpu4_prefetch_[slot];
                  for (std::size_t value = 0;
                       finite_fields && value < prefetch.fields1.size();
                       ++value)
                    finite_fields = std::isfinite(prefetch.fields1[value]);
                  for (std::size_t value = 0;
                       finite_fields && value < prefetch.fields2.size();
                       ++value)
                    finite_fields = std::isfinite(prefetch.fields2[value]);
                  if (!deposit1_ok || !deposit2_ok || !gather1_ok ||
                      !gather2_ok || !finite_fields)
                    {
                      std::ostringstream message;
                      message << "gpu_mode=4 bucket validation failed at "
                              << "slot " << slot
                              << "; refusing every host physics commit in "
                              << "this bucket";
                      throw std::runtime_error(message.str());
                    }
                }

              backend.record_gpu_mode4_execution(
                  gpu4_wave_solver_->device_description());
              checked_add_u64(batches, 1, "gpu_mode=4 batch");
              time_.checkpoint();

              const std::chrono::steady_clock::time_point commit_start =
                  std::chrono::steady_clock::now();
              for (std::size_t slot = 0; slot < count; ++slot)
                {
                  feebl::wavefront_fields::FieldPrefetch& prefetch =
                      gpu4_prefetch_[slot];
                  prefetch.beam1_in = outputs[slot].beam1_in;
                  prefetch.beam1_out = outputs[slot].beam1_out;
                  prefetch.beam2_in = outputs[slot].beam2_in;
                  prefetch.beam2_out = outputs[slot].beam2_out;
                  prefetch.fine_lumi_dot = outputs[slot].fine_lumi_dot;
                }
              if (switches.get_wavefront_parallel_physics())
                {
                  for (std::size_t slot_begin = 0; slot_begin < count;
                       slot_begin += wavefront_field_workers_)
                    {
                      const std::size_t host_count = std::min<std::size_t>(
                          wavefront_field_workers_, count - slot_begin);
                      physics_context.position_offset = begin + slot_begin;
                      physics_context.prefetch_offset = slot_begin;
                      physics_context.legacy_time_base = time_counter_;
                      backend.run_gpu2_primary_static_ranges(
                          host_count, &physics_context,
                          &FEEBL::wavefront_process_physics_range);
                      publish_wavefront_physics_batch(
                          begin + slot_begin, host_count);
                      checked_add_u64(
                          encounters,
                          checked_size_to_u64(
                              host_count,
                              "gpu_mode=4 parallel physics encounter count"),
                          "gpu_mode=4 collision encounter");
                    }
                }
              else
                {
                  for (std::size_t slot = 0; slot < count; ++slot)
                    {
                      const feebl::modern::EncounterNode& encounter =
                          scheduler.dag().node(
                              wave.encounters[begin + slot]);
                      make_step(
                          static_cast<int>(encounter.beam1_slice),
                          static_cast<int>(encounter.beam2_slice),
                          sor_parameter, &gpu4_prefetch_[slot]);
                      checked_add_u64(
                          encounters, 1,
                          "gpu_mode=4 collision encounter");
                    }
                }
              const double commit_elapsed_ms =
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - commit_start)
                      .count();
              if (switches.get_wavefront_parallel_physics())
                parallel_physics_ms += commit_elapsed_ms;
              else
                ordered_commit_ms += commit_elapsed_ms;
            }
          wavefront_end_wave(&boundary_context, wave);
        }
      wavefront_end_diagonal(&boundary_context, diagonal);
    }

  if (encounters != scheduler.dag().encounter_count())
    throw std::logic_error(
        "gpu_mode=4 did not exhaust the committed EncounterDAG");
  double lumi_heap_publish_ms = 0.0;
  if (switches.get_wavefront_parallel_physics())
    {
      const std::chrono::steady_clock::time_point publish_start =
          std::chrono::steady_clock::now();
      publish_wavefront_lumi_heaps();
      lumi_heap_publish_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - publish_start).count();
    }
  switches.set_collision_schedule_provenance(
      switches.get_wavefront_parallel_physics()
          ? "gpu_mode4_wave_batched_fields_parallel_host_physics"
          : "gpu_mode4_wave_batched_fields_host_commit",
      checked_size_to_u64(scheduler.diagonals().size(),
                          "gpu_mode=4 diagonal count"),
      scheduler.dag().wave_count(), batches, encounters,
      switches.get_wavefront_parallel_physics() ? encounters : 0,
      checked_size_to_u64(maximum_wave_width,
                          "gpu_mode=4 maximum wave width"));
  switches.set_collision_field_provenance(
      encounters, encounters,
      switches.get_wavefront_parallel_physics() ? encounters : 0,
      checked_size_to_u64(gpu4_batch_width_, "gpu_mode=4 batch width"),
      switches.get_wavefront_parallel_physics()
          ? static_cast<int>(wavefront_field_workers_) : 1,
      gpu4_wave_solver_->stats().setup_ms,
      field_prefetch_ms, ordered_commit_ms,
      0, 0, 0, 0, batches,
      checked_size_to_u64(gpu4_wave_solver_->stats().resident_bytes,
                          "gpu_mode=4 resident bytes"),
      checked_size_to_u64(gpu4_wave_solver_->stats().resident_bytes,
                          "gpu_mode=4 resident bytes"), 1);
  std::cerr
      << "FEEBL legacy wavefront summary: requested=wavefront_parallel"
      << " actual="
      << (switches.get_wavefront_parallel_physics()
              ? "gpu_mode4_wave_batched_fields_parallel_host_physics"
              : "gpu_mode4_wave_batched_fields_host_commit")
      << " scheduler=encounter-dag-wave-batched-planmany-fields-v1"
      << " diagonals=" << scheduler.diagonals().size()
      << " waves=" << scheduler.dag().wave_count()
      << " batches=" << batches
      << " encounters=" << encounters
      << " full_parallel_make_step_encounters="
      << (switches.get_wavefront_parallel_physics() ? encounters : 0)
      << " maximum_wave_width=" << maximum_wave_width
      << " batch_width=" << gpu4_batch_width_
      << " field_prefetch_ms=" << field_prefetch_ms
      << " parallel_physics_ms=" << parallel_physics_ms
      << " ordered_commit_ms=" << ordered_commit_ms
      << " device_synchronization_scope=batch"
      << " per_encounter_synchronizations=0"
      << " ordered_commit="
      << (switches.get_wavefront_parallel_physics() ? 0 : 1)
      << " canonical_barrier_publication="
      << (switches.get_wavefront_parallel_physics() ? 1 : 0)
      << " crossing_lumi_heap_publish_ms=" << lumi_heap_publish_ms
      << " crossing_lumi_heap_publications="
      << (switches.get_wavefront_parallel_physics() ? 1 : 0)
      << std::endl;
}

std::string FEEBL::gpu4_wave_summary() const
{
  if (!gpu4_wave_solver_) return " FEEBL GPU mode4 summary: unavailable";
  const feebl::gpu::WaveBatchStats stats = gpu4_wave_solver_->stats();
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << " FEEBL GPU mode4 summary:"
      << " implementation=encounter-dag-wave-batched-planmany-fields-v1"
      << " batch_width=" << stats.batch_width
      << " batch_calls=" << stats.batch_calls
      << " encounters=" << stats.encounters
      << " padded_fft_slots=" << stats.padded_fft_slots
      << " synchronizations=" << stats.synchronizations
      << " per_encounter_synchronizations=0"
      << " h2d_operations=" << stats.h2d_operations
      << " d2h_operations=" << stats.d2h_operations
      << " h2d_bytes=" << stats.h2d_bytes
      << " d2h_bytes=" << stats.d2h_bytes
      << " kernel_launches=" << stats.kernel_launches
      << " cufft_executes=" << stats.cufft_executes
      << " setup_h2d_operations=" << stats.setup_h2d_operations
      << " setup_h2d_bytes=" << stats.setup_h2d_bytes
      << " setup_synchronizations=" << stats.setup_synchronizations
      << " plan_creations=" << stats.plan_creations
      << " device_allocations=" << stats.device_allocations
      << " pinned_host_allocations=" << stats.pinned_host_allocations
      << " setup_ms=" << stats.setup_ms
      << " host_pack_ms=" << stats.pack_ms
      << " h2d_ms=" << stats.h2d_ms
      << " kernel_ms=" << stats.kernel_ms
      << " d2h_ms=" << stats.d2h_ms
      << " host_unpack_ms=" << stats.unpack_ms
      << " boundary_ms=" << stats.boundary_ms
      << " resident_bytes=" << stats.resident_bytes
      << " cufft_workspace_bytes=" << stats.cufft_workspace_bytes
      << " deposit_beam1_in=" << stats.deposit_beam1_in
      << " deposit_beam1_out=" << stats.deposit_beam1_out
      << " deposit_beam2_in=" << stats.deposit_beam2_in
      << " deposit_beam2_out=" << stats.deposit_beam2_out
      << " gather_beam1_in=" << stats.gather_beam1_in
      << " gather_beam1_out=" << stats.gather_beam1_out
      << " gather_beam2_in=" << stats.gather_beam2_in
      << " gather_beam2_out=" << stats.gather_beam2_out
      << " host_phi_materialized=0"
      << " host_physics=legacy_complete_ordered"
      << " rng=legacy_host_ordered"
      << " device=\"" << gpu4_wave_solver_->device_description() << "\"";
  return out.str();
}

void FEEBL::wavefront_begin_diagonal(
    void* opaque,
    const feebl::legacy_wavefront::DiagonalDescriptor& /* diagonal */)
{
  LEGACY_WAVEFRONT_CONTEXT& context =
      *static_cast<LEGACY_WAVEFRONT_CONTEXT*>(opaque);
  context.simulation->grid_.get_results().set_step_lumi_ee(0.0);
  context.simulation->grid_.get_results().set_step_lumi_ee_high(0.0);
}

void FEEBL::wavefront_begin_wave(
    void* opaque, const feebl::modern::EncounterWave& /* wave */)
{
  LEGACY_WAVEFRONT_CONTEXT& context =
      *static_cast<LEGACY_WAVEFRONT_CONTEXT*>(opaque);
  context.simulation->grid_.check_distribute(1);
}

void FEEBL::wavefront_encounter(
    void* opaque, const feebl::modern::EncounterNode& encounter)
{
  LEGACY_WAVEFRONT_CONTEXT& context =
      *static_cast<LEGACY_WAVEFRONT_CONTEXT*>(opaque);
  context.simulation->make_step(
      static_cast<int>(encounter.beam1_slice),
      static_cast<int>(encounter.beam2_slice), context.sor_parameter);
}

void FEEBL::wavefront_end_wave(
    void* opaque, const feebl::modern::EncounterWave& wave)
{
  LEGACY_WAVEFRONT_CONTEXT& context =
      *static_cast<LEGACY_WAVEFRONT_CONTEXT*>(opaque);
  FEEBL& simulation = *context.simulation;
  simulation.grid_.check_distribute(2);
  if (simulation.switches.get_do_size_log() && !wave.encounters.empty())
    {
      const feebl::modern::EncounterNode& first =
          context.scheduler->dag().node(wave.encounters.front());
      const feebl::modern::EncounterNode& last =
          context.scheduler->dag().node(wave.encounters.back());
      simulation.beam1_.write_size(
          static_cast<int>(first.beam1_slice),
          static_cast<int>(last.beam1_slice));
      simulation.beam2_.write_size(
          static_cast<int>(first.beam1_slice),
          static_cast<int>(last.beam1_slice));
    }
}

void FEEBL::wavefront_end_diagonal(
    void* opaque,
    const feebl::legacy_wavefront::DiagonalDescriptor& diagonal)
{
  LEGACY_WAVEFRONT_CONTEXT& context =
      *static_cast<LEGACY_WAVEFRONT_CONTEXT*>(opaque);
  FEEBL& simulation = *context.simulation;
  simulation.grid_.get_results().add_lumi_ee_step(
      simulation.grid_.get_results().get_step_lumi_ee());
  simulation.grid_.get_results().add_lumi_ee_high_step(
      simulation.grid_.get_results().get_step_lumi_ee_high());

  if (simulation.switches.get_do_dump())
    {
      const int istep = static_cast<int>(diagonal.diagonal);
      if (istep % simulation.switches.get_dump_step() == 0)
        {
          simulation.dump_beams(
              istep, simulation.switches.get_dump_step(),
              simulation.switches.get_dump_particle());
        }
    }

  simulation.write_gui_snapshot(static_cast<int>(diagonal.diagonal));

  if (diagonal.diagonal + 1 ==
      static_cast<feebl::modern::WaveIndex>(
          simulation.grid_.get_n_cell_z()))
    simulation.polarization_.sample(
        POLARIZATION_REPORT::COLLISION,
        simulation.beam1_, simulation.beam2_);
}

void FEEBL::beam_interaction_wavefront_strict(PHI_FLOAT* sor_parameter)
{
  const int n_slice = grid_.get_n_cell_z();
  const int n_substeps = grid_.get_timestep();
  if (n_slice <= 0 || n_substeps <= 0)
    throw std::logic_error(
        "wavefront scheduler requires positive n_z and n_t");

  const feebl::legacy_wavefront::Scheduler scheduler(
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SubstepIndex>(n_substeps));
  LEGACY_WAVEFRONT_CONTEXT context;
  context.simulation = this;
  context.sor_parameter = sor_parameter;
  context.scheduler = &scheduler;
  feebl::legacy_wavefront::StrictCallbacks callbacks;
  callbacks.context = &context;
  callbacks.begin_diagonal = &FEEBL::wavefront_begin_diagonal;
  callbacks.begin_wave = &FEEBL::wavefront_begin_wave;
  callbacks.encounter = &FEEBL::wavefront_encounter;
  callbacks.end_wave = &FEEBL::wavefront_end_wave;
  callbacks.end_diagonal = &FEEBL::wavefront_end_diagonal;

  const feebl::legacy_wavefront::ExecutionReport report =
      scheduler.execute_strict(callbacks);
  switches.set_collision_schedule_provenance(
      feebl::legacy_wavefront::schedule_mode_name(report.actual),
      report.diagonals, report.waves, report.batches, report.encounters,
      report.parallel_physics_encounters, report.maximum_wave_width);
  std::cerr
      << "FEEBL legacy wavefront summary: requested=wavefront_strict"
      << " actual="
      << feebl::legacy_wavefront::schedule_mode_name(report.actual)
      << " scheduler=encounter-dag-antidiagonal-v1"
      << " diagonals=" << report.diagonals
      << " waves=" << report.waves
      << " batches=" << report.batches
      << " encounters=" << report.encounters
      << " maximum_wave_width=" << report.maximum_wave_width
      << " parallel_physics_encounters="
      << report.parallel_physics_encounters
      << " ordered_commit=" << (report.canonical_order ? 1 : 0)
      << " task_local_grid=0"
      << std::endl;
}

struct FEEBL::WAVEFRONT_FIELD_CONTEXT
{
  FEEBL* simulation;
  const feebl::legacy_wavefront::Scheduler* scheduler;
  const feebl::modern::EncounterWave* wave;
  const std::vector<std::size_t>* dispatch_order;
  std::size_t position_offset;
};

void FEEBL::wavefront_plan_field_range(
    void* opaque, std::size_t lane, std::size_t begin, std::size_t end)
{
  WAVEFRONT_FIELD_CONTEXT& context =
      *static_cast<WAVEFRONT_FIELD_CONTEXT*>(opaque);
  FEEBL& simulation = *context.simulation;
  if (!context.wave || lane >= simulation.wavefront_field_workspaces_.size() ||
      context.dispatch_order != NULL || context.position_offset != 0U ||
      end > context.wave->encounters.size())
    throw std::logic_error(
        "adaptive window-plan dispatch received an invalid persistent lane");
  feebl::wavefront_fields::Workspace& workspace =
      *simulation.wavefront_field_workspaces_[lane];
  for (std::size_t position = begin; position < end; ++position)
    {
      if (position >= simulation.wavefront_field_prefetch_.size())
        throw std::out_of_range(
            "adaptive window-plan dispatch exceeded its canonical slots");
      const feebl::modern::EncounterNode& encounter =
          context.scheduler->dag().node(context.wave->encounters[position]);
      feebl::wavefront_fields::FieldPrefetch& prefetch =
          simulation.wavefront_field_prefetch_[position];
      prefetch.window_plan = workspace.make_window_plan(
          static_cast<int>(encounter.beam1_slice),
          static_cast<int>(encounter.beam2_slice));
      const int slice1 = static_cast<int>(encounter.beam1_slice);
      const int slice2 = static_cast<int>(encounter.beam2_slice);
      feebl::wavefront_fields::bind_field_result(
          prefetch, slice1, slice2,
          simulation.beam1_.getParticleVector(slice1),
          simulation.beam2_.getParticleVector(slice2));
    }
}

void FEEBL::wavefront_prepare_field_range(
    void* opaque, std::size_t lane, std::size_t begin, std::size_t end)
{
  WAVEFRONT_FIELD_CONTEXT& context =
      *static_cast<WAVEFRONT_FIELD_CONTEXT*>(opaque);
  FEEBL& simulation = *context.simulation;
  if (!context.wave || lane >= simulation.wavefront_field_workspaces_.size())
    throw std::logic_error(
        "wavefront field dispatch received an invalid persistent lane");
  if (context.position_offset > context.wave->encounters.size() ||
      end > context.wave->encounters.size() - context.position_offset ||
      context.position_offset + end > context.wave->encounters.size())
    throw std::out_of_range(
        "wavefront field dispatch exceeded its committed wave");

  feebl::wavefront_fields::Workspace& workspace =
      *simulation.wavefront_field_workspaces_[lane];
  for (std::size_t position = begin; position < end; ++position)
    {
      const std::size_t dispatch_position =
          context.position_offset + position;
      const std::size_t wave_position = context.dispatch_order
          ? (*context.dispatch_order)[dispatch_position]
          : dispatch_position;
      if (wave_position >= context.wave->encounters.size() ||
          wave_position >= simulation.wavefront_field_prefetch_.size())
        throw std::out_of_range(
            "wavefront adaptive bucket references an invalid canonical slot");
      const feebl::modern::EncounterNode& encounter =
          context.scheduler->dag().node(
              context.wave->encounters[wave_position]);
      workspace.prepare(
          static_cast<int>(encounter.beam1_slice),
          static_cast<int>(encounter.beam2_slice),
          simulation.wavefront_field_prefetch_[wave_position]);
    }
}

void FEEBL::wavefront_process_physics_range(
    void* opaque, std::size_t lane, std::size_t begin, std::size_t end)
{
  WAVEFRONT_PHYSICS_CONTEXT& context =
      *static_cast<WAVEFRONT_PHYSICS_CONTEXT*>(opaque);
  FEEBL& simulation = *context.simulation;
  if (!context.wave || !context.prefetch ||
      lane >= simulation.wavefront_field_workspaces_.size() ||
      end < begin || end - begin > 1U)
    throw std::logic_error(
        "wavefront physics dispatch violated its one-encounter lane lease");
  for (std::size_t item = begin; item < end; ++item)
    {
      const std::size_t position = context.position_offset + item;
      const std::size_t field_position = context.prefetch_offset + item;
      if (position >= context.wave->encounters.size() ||
          field_position >= context.prefetch->size() ||
          position >= simulation.wavefront_physics_lane_for_position_.size())
        throw std::logic_error(
            "wavefront physics dispatch exceeded its committed batch");
      const feebl::modern::EncounterNode& encounter =
          context.scheduler->dag().node(context.wave->encounters[position]);
      feebl::wavefront_fields::PhysicsIdentity identity;
      identity.run_seed = static_cast<std::uint64_t>(
          simulation.switches.get_rndm_seed());
      identity.crossing = 0U;
      identity.diagonal = encounter.diagonal;
      identity.wave = encounter.wave;
      identity.substep = encounter.substep;
      identity.encounter = encounter.id;
      identity.legacy_time_counter = context.legacy_time_base +
          static_cast<int>(item);
      simulation.wavefront_field_workspaces_[lane]
          ->process_primary_photon_collision(
              static_cast<int>(encounter.beam1_slice),
              static_cast<int>(encounter.beam2_slice),
              (*context.prefetch)[field_position], identity);
      simulation.wavefront_physics_lane_for_position_[position] = lane;
    }
}

void FEEBL::publish_wavefront_physics_batch(
    std::size_t begin, std::size_t count)
{
  std::vector<feebl::wavefront_fields::Workspace*> workspaces;
  std::vector<const GRID*> lanes;
  workspaces.reserve(count);
  lanes.reserve(count);
  for (std::size_t offset = 0; offset < count; ++offset)
    {
      const std::size_t position = begin + offset;
      if (position >= wavefront_physics_lane_for_position_.size())
        throw std::logic_error(
            "wavefront physics publication exceeded lane map");
      const std::size_t lane = wavefront_physics_lane_for_position_[position];
      if (lane >= wavefront_field_workspaces_.size())
        throw std::logic_error(
            "wavefront physics publication received an invalid lane");
      feebl::wavefront_fields::Workspace* workspace =
          wavefront_field_workspaces_[lane].get();
      workspaces.push_back(workspace);
      lanes.push_back(&workspace->ready_primary_photon_lane());
    }
  std::string reason;
  if (!grid_.merge_primary_photon_collision_lanes(lanes, &reason))
    throw std::runtime_error(
        "wavefront physics canonical publication failed: " + reason);
  for (std::size_t lane = 0; lane < workspaces.size(); ++lane)
    workspaces[lane]->acknowledge_primary_photon_publication();
  if (count > static_cast<std::size_t>(
                  std::numeric_limits<int>::max() - time_counter_))
    throw std::overflow_error("legacy collision time counter overflow");
  time_counter_ += static_cast<int>(count);
}

void FEEBL::publish_wavefront_lumi_heaps()
{
  // The fixed-physical spatial kernel is aggregate-only and card validation
  // forbids its e-e event heap.  Worker RESULTS/receipts have already been
  // published transactionally at each canonical batch barrier; there is no
  // heap state to merge at the crossing barrier.  The historical sampled
  // wavefront contract always has bit 1 set, so Mode 0 is unchanged.
  if (!(switches.get_do_lumi() & 1)) return;
  std::vector<const GRID*> lanes;
  lanes.reserve(wavefront_field_workspaces_.size());
  for (std::size_t lane = 0;
       lane < wavefront_field_workspaces_.size(); ++lane)
    lanes.push_back(
        &wavefront_field_workspaces_[lane]
             ->completed_primary_photon_lane());
  std::string reason;
  if (!grid_.merge_primary_photon_lumi_heaps(lanes, &reason))
    throw std::runtime_error(
        "wavefront crossing luminosity publication failed: " + reason);
}

void FEEBL::initialize_wavefront_fields()
{
  if (switches.get_collision_schedule() != "wavefront_parallel") return;
  PARALLEL_BACKEND& backend = parallel_backend();
  const bool gpu4_parallel_physics =
      backend.gpu_mode4_enabled() &&
      switches.get_wavefront_parallel_physics();
  const bool gpu5_adaptive = backend.gpu_mode5_enabled();
  if (backend.gpu_mode4_enabled() && !gpu4_parallel_physics) return;
  if (gpu4_parallel_physics)
    {
      if (backend.actual_gpu2_options().primary_mover != "pool" ||
          backend.actual_cpu_threads() < 2)
        throw std::logic_error(
            "gpu_mode=4 parallel host physics entered without its committed "
            "persistent host pool");
    }
  else if (gpu5_adaptive)
    {
      if (backend.actual_gpu2_options().primary_mover != "pool" ||
          backend.actual_cpu_threads() < 2 ||
          switches.get_adaptive_grid_mode() != 1)
        throw std::logic_error(
            "gpu_mode=5 wavefront initialization entered without its "
            "adaptive topology and persistent host-pool contract");
    }
  else if (!backend.cpu_enabled() ||
           backend.actual_cpu_variant() != "pool_dual" ||
           backend.actual_cpu_threads() < 2)
    throw std::logic_error(
        "wavefront field initialization entered without its committed pool");

  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  float cut_x = 0.0f;
  float cut_y = 0.0f;
  float cut_z = 0.0f;
  grid_.get_cuts(cut_x, cut_y, cut_z);
  feebl::wavefront_fields::Configuration configuration;
  configuration.n_x = grid_.get_n_cell_x();
  configuration.n_y = grid_.get_n_cell_y();
  configuration.n_z = grid_.get_n_cell_z();
  configuration.n_t = grid_.get_timestep();
  configuration.cut_x = cut_x;
  configuration.cut_y = cut_y;
  configuration.cut_z = cut_z;
  configuration.charge_sign = switches.get_charge_sign();
  configuration.macro_charge1 = grid_.get_macro_charge(1);
  configuration.macro_charge2 = grid_.get_macro_charge(2);
  configuration.physical_particles1 =
      configuration.macro_charge1 *
      static_cast<double>(beam1_.particle_beam().numberOfParticles());
  configuration.physical_particles2 =
      configuration.macro_charge2 *
      static_cast<double>(beam2_.particle_beam().numberOfParticles());
  const auto maximum_slice_population = [](BEAM& beam) {
    if (beam.number_of_slices() <= 0)
      throw std::runtime_error(
          "wavefront workspace requires a positive beam slice count");
    std::size_t maximum = 0U;
    for (int slice = 0; slice < beam.number_of_slices(); ++slice)
      maximum = std::max(
          maximum, beam.getParticleVector(slice).size());
    return maximum;
  };
  configuration.maximum_slice_particles1 =
      maximum_slice_population(beam1_);
  configuration.maximum_slice_particles2 =
      maximum_slice_population(beam2_);
  configuration.lumi_event_capacity =
      switches.get_wavefront_parallel_physics()
          ? static_cast<std::size_t>(
                std::max(100, switches.get_num_lumi()))
          : 0U;
  configuration.sparse_cells = switches.get_sparse_cells() != 0;
  configuration.field_solver =
      feebl::sparse_direct_fields::parse_mode(
          switches.get_wavefront_field_solver());
  configuration.sparse_direct_interaction_limit =
      static_cast<std::uint64_t>(
          switches.get_sparse_direct_interaction_limit());
  configuration.adaptive_grid_mode = switches.get_adaptive_grid_mode();
  configuration.adaptive_macroparticles =
      switches.get_adaptive_macroparticle_configuration();
  configuration.adaptive_profile = switches.get_adaptive_grid_profile();
  configuration.retain_full_window_descriptor_payload =
      gpu5_adaptive || configuration.adaptive_grid_mode >= 2;
  if (configuration.adaptive_grid_mode >= 1)
    {
      feebl::adaptive_grid::UniformWindowPolicy& policy =
          configuration.adaptive_window_policy;
      const bool implicit_full_class =
          switches.get_adaptive_grid_size_classes_x().empty();
      if (implicit_full_class)
        {
          policy.size_classes_x.push_back(
              static_cast<std::size_t>(configuration.n_x));
          policy.size_classes_y.push_back(
              static_cast<std::size_t>(configuration.n_y));
        }
      else
        {
          policy.size_classes_x = feebl::adaptive_grid::parse_size_classes(
              switches.get_adaptive_grid_size_classes_x());
          policy.size_classes_y = feebl::adaptive_grid::parse_size_classes(
              switches.get_adaptive_grid_size_classes_y());
        }
      policy.min_n_x = switches.get_adaptive_grid_min_nx() == 0
          ? 4U
          : static_cast<std::size_t>(
                switches.get_adaptive_grid_min_nx());
      policy.min_n_y = switches.get_adaptive_grid_min_ny() == 0
          ? 4U
          : static_cast<std::size_t>(
                switches.get_adaptive_grid_min_ny());
      policy.max_n_x = switches.get_adaptive_grid_max_nx() == 0
          ? static_cast<std::size_t>(configuration.n_x)
          : static_cast<std::size_t>(
                switches.get_adaptive_grid_max_nx());
      policy.max_n_y = switches.get_adaptive_grid_max_ny() == 0
          ? static_cast<std::size_t>(configuration.n_y)
          : static_cast<std::size_t>(
                switches.get_adaptive_grid_max_ny());
      policy.extra_guard_cells =
          switches.get_adaptive_grid_extra_guard_cells();
      policy.allow_full_window =
          switches.get_adaptive_grid_allow_full_window() != 0;
      // An explicit allow-full permission makes the canonical topology a
      // named final class when both max dimensions admit it.  This selection
      // is still counted; crop-only qualification sets allow_full_window=0.
      if (!implicit_full_class && policy.allow_full_window &&
          policy.max_n_x == static_cast<std::size_t>(configuration.n_x) &&
          policy.max_n_y == static_cast<std::size_t>(configuration.n_y))
        {
          if (std::find(policy.size_classes_x.begin(),
                        policy.size_classes_x.end(),
                        static_cast<std::size_t>(configuration.n_x)) ==
              policy.size_classes_x.end())
            policy.size_classes_x.push_back(
                static_cast<std::size_t>(configuration.n_x));
          if (std::find(policy.size_classes_y.begin(),
                        policy.size_classes_y.end(),
                        static_cast<std::size_t>(configuration.n_y)) ==
              policy.size_classes_y.end())
            policy.size_classes_y.push_back(
                static_cast<std::size_t>(configuration.n_y));
          std::sort(policy.size_classes_x.begin(),
                    policy.size_classes_x.end());
          std::sort(policy.size_classes_y.begin(),
                    policy.size_classes_y.end());
        }
      policy.fixed_bytes = 1024U * 1024U;
      const std::uint64_t requested_max_bytes =
          switches.get_adaptive_grid_max_bytes();
      const std::uint64_t derived_max_bytes =
          UINT64_C(4) * 1024U * 1024U * 1024U;
      const std::uint64_t effective_max_bytes =
          requested_max_bytes == 0U
              ? derived_max_bytes : requested_max_bytes;
      if (effective_max_bytes > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max()))
        throw std::overflow_error(
            "adaptive_grid_max_bytes does not fit this host size_t");
      policy.max_bytes = static_cast<std::size_t>(effective_max_bytes);
      const std::uint64_t canonical_cells =
          static_cast<std::uint64_t>(configuration.n_x) *
          static_cast<std::uint64_t>(configuration.n_y);
      configuration.adaptive_max_cells =
          switches.get_adaptive_grid_max_cells() == 0U
              ? canonical_cells
              : switches.get_adaptive_grid_max_cells();
      configuration.canonical_geometry_revision = 1U;
      if (configuration.adaptive_grid_mode >= 2)
        {
          configuration.adaptive_spacing_factors =
              feebl::adaptive_grid::parse_refinement_factors(
                  switches.get_adaptive_grid_field_spacing_classes());
          configuration.adaptive_field_operator =
              switches.get_adaptive_grid_field_operator();
          const std::uint64_t requested_max_refinement_factor =
              switches.get_adaptive_grid_max_refinement_factor();
          if (requested_max_refinement_factor >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()))
            throw std::overflow_error(
                "adaptive_grid_max_refinement_factor does not fit this "
                "host size_t");
          configuration.adaptive_max_refinement_factor =
              static_cast<std::size_t>(requested_max_refinement_factor);
          configuration.adaptive_anchor_x =
              switches.get_adaptive_grid_anchor_x();
          configuration.adaptive_anchor_y =
              switches.get_adaptive_grid_anchor_y();
          configuration.adaptive_error_target =
              switches.get_adaptive_grid_error_target();
          configuration.adaptive_error_norm_policy =
              switches.get_adaptive_grid_error_norm_policy();
          configuration.adaptive_error_budget_sha256 =
              switches.get_adaptive_grid_error_budget_sha256();
          configuration.adaptive_error_quantile =
              switches.get_adaptive_grid_error_quantile();
          configuration.adaptive_error_catastrophe_limit =
              switches.get_adaptive_grid_error_catastrophe_limit();
          configuration.adaptive_error_reference_field =
              switches.get_adaptive_grid_error_reference_field();
          configuration.adaptive_error_absolute_floor_fraction =
              switches.get_adaptive_grid_error_absolute_floor_fraction();
          configuration.adaptive_error_catastrophe_absolute_fraction =
              switches
                  .get_adaptive_grid_error_catastrophe_absolute_fraction();
          configuration.adaptive_error_exposure_limit =
              switches.get_adaptive_grid_error_exposure_limit();
          configuration.adaptive_noise_aware_stop =
              switches.get_adaptive_grid_noise_aware_stop() != 0;
          configuration.adaptive_source_model =
              switches.get_adaptive_grid_source_model();
          configuration.adaptive_source_reference_nx =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_source_reference_nx());
          configuration.adaptive_source_reference_ny =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_source_reference_ny());
          configuration.adaptive_source_width_x =
              switches.get_adaptive_grid_source_width_x();
          configuration.adaptive_source_width_y =
              switches.get_adaptive_grid_source_width_y();
          configuration.adaptive_source_error_target =
              switches.get_adaptive_grid_source_error_target();
          configuration.adaptive_solver_error_target =
              switches.get_adaptive_grid_solver_error_target();
          configuration.adaptive_lumi_error_target =
              switches.get_adaptive_grid_lumi_error_target();
          configuration.adaptive_lumi_floor =
              switches.get_adaptive_grid_lumi_floor();
          configuration.adaptive_tau_curvature =
              switches.get_adaptive_grid_tau_curvature();
          configuration.adaptive_tau_field_gradient =
              switches.get_adaptive_grid_tau_field_gradient();
          configuration.adaptive_tau_richardson =
              switches.get_adaptive_grid_tau_richardson();
          configuration.adaptive_tau_tail =
              switches.get_adaptive_grid_tau_tail();
          configuration.adaptive_encounter_tail_fraction =
              switches.get_adaptive_grid_encounter_tail_fraction();
          configuration.adaptive_rho_floor =
              switches.get_adaptive_grid_rho_floor();
          configuration.adaptive_field_floor =
              switches.get_adaptive_grid_field_floor();
          configuration.adaptive_indicator_smoothing_passes =
              static_cast<unsigned int>(
                  switches.get_adaptive_grid_indicator_smoothing_passes());
          configuration.adaptive_refine_threshold =
              switches.get_adaptive_grid_refine_threshold();
          configuration.adaptive_coarsen_threshold =
              switches.get_adaptive_grid_coarsen_threshold();
          configuration.adaptive_richardson_order_source =
              switches.get_adaptive_grid_richardson_order_source();
          configuration.adaptive_richardson_order_table =
              switches.get_adaptive_grid_richardson_order_table();
          configuration.adaptive_min_neff =
              switches.get_adaptive_grid_min_neff();
          configuration.adaptive_fail_on_unmet_error =
              switches.get_adaptive_grid_fail_on_unmet_error() != 0;
          configuration.adaptive_refinement_ratio =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_refinement_ratio());
          configuration.adaptive_blocking_factor =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_blocking_factor());
          configuration.adaptive_error_buffer_cells =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_error_buffer_cells());
          configuration.adaptive_transition_cells =
              static_cast<std::size_t>(
                  switches.get_adaptive_grid_transition_cells());
          configuration.adaptive_fine_residual_tolerance =
              switches.get_adaptive_grid_fine_residual_tolerance();
          configuration.adaptive_terminal_patch_certification =
              switches.get_adaptive_grid_terminal_patch_certification() != 0;
        }
    }

  const std::size_t base_workspace_budget_bytes =
      (switches.get_wavefront_parallel_physics() ||
       configuration.adaptive_grid_mode >= 1)
          ? 4ULL * 1024ULL * 1024ULL * 1024ULL
          : 512ULL * 1024ULL * 1024ULL;
  // Modes 2/3 reserve their retained contexts and transient controller
  // storage separately through adaptive_grid_max_bytes.  The historical
  // four-GiB wavefront ceiling predated those declarations and otherwise
  // rejects every memory-safe long-bunch 1024-grid card before the adaptive
  // cap is even consulted.  Treat the established wavefront allowance and
  // the explicit adaptive reservation as additive, while preserving the
  // independent per-lane adaptive-cap check below.  Overflow fails before
  // any workspace allocation or beam mutation.
  const std::size_t adaptive_workspace_reservation_bytes =
      configuration.adaptive_grid_mode >= 2
          ? configuration.adaptive_window_policy.max_bytes : 0U;
  std::size_t workspace_budget_bytes = base_workspace_budget_bytes;
  if (configuration.adaptive_grid_mode >= 2) {
    if (adaptive_workspace_reservation_bytes >
        std::numeric_limits<std::size_t>::max() -
            base_workspace_budget_bytes)
      throw std::overflow_error(
          "wavefront plus adaptive workspace budget overflows size_t");
    workspace_budget_bytes = base_workspace_budget_bytes +
                             adaptive_workspace_reservation_bytes;
  }
  const std::uint64_t requested_workspace_budget =
      switches.get_wavefront_workspace_max_bytes();
  if (requested_workspace_budget >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("wavefront_workspace_max_bytes exceeds size_t");
  if (requested_workspace_budget != 0U)
    workspace_budget_bytes = static_cast<std::size_t>(requested_workspace_budget);
  wavefront_workspace_bytes_each_ =
      feebl::wavefront_fields::Workspace::estimated_bytes(configuration);
  const std::size_t adaptive_context_bytes_each =
      feebl::wavefront_fields::Workspace::
          estimated_adaptive_context_bytes(configuration);
  const std::size_t adaptive_transient_bytes_each =
      feebl::wavefront_fields::Workspace::
          estimated_adaptive_transient_bytes(configuration);
  if (adaptive_transient_bytes_each >
      std::numeric_limits<std::size_t>::max() -
          adaptive_context_bytes_each)
    throw std::overflow_error(
        "adaptive retained/transient per-lane estimate overflows size_t");
  const std::size_t adaptive_bytes_each =
      adaptive_context_bytes_each + adaptive_transient_bytes_each;
  if (adaptive_bytes_each >
      std::numeric_limits<std::size_t>::max() -
          wavefront_workspace_bytes_each_)
    throw std::overflow_error(
        "combined wavefront/adaptive per-lane estimate overflows size_t");
  const std::size_t design_axis_bytes_each = switches.get_wavefront_parallel_physics()
      ? grid_.design_axis_spectrum_.storage_bytes() : 0U;
  if (design_axis_bytes_each > std::numeric_limits<std::size_t>::max() -
          wavefront_workspace_bytes_each_ - adaptive_bytes_each)
    throw std::overflow_error("design-axis wavefront workspace size overflows size_t");
  const std::size_t combined_bytes_each =
      wavefront_workspace_bytes_each_ + adaptive_bytes_each + design_axis_bytes_each;
  std::size_t memory_lanes = workspace_budget_bytes / combined_bytes_each;
  if (adaptive_bytes_each != 0U)
    memory_lanes = std::min(
        memory_lanes,
        configuration.adaptive_window_policy.max_bytes /
            adaptive_bytes_each);
  wavefront_field_workers_ = std::min<std::size_t>(
      static_cast<std::size_t>(backend.actual_cpu_threads()),
      std::min<std::size_t>(
          static_cast<std::size_t>(grid_.get_n_cell_z()), memory_lanes));
  // An explicit budget must accommodate the requested concurrency. This
  // keeps a production card from silently turning a 16-worker run into a
  // smaller pool when its grid or diagnostic histogram is enlarged.
  if (requested_workspace_budget != 0U && memory_lanes <
      std::min<std::size_t>(static_cast<std::size_t>(backend.actual_cpu_threads()),
                           static_cast<std::size_t>(grid_.get_n_cell_z())))
    throw std::runtime_error(
        "wavefront_workspace_max_bytes cannot accommodate all requested workers");
  if (wavefront_field_workers_ < 2U)
    throw std::runtime_error(
        "collision_schedule=wavefront_parallel cannot reserve two "
        "worker-private GRID/FIELD/FFT/result workspaces inside the bounded "
        "preflight budget; no fallback was performed");
  if (wavefront_field_workers_ >
      std::numeric_limits<std::size_t>::max() / combined_bytes_each)
    throw std::overflow_error(
        "wavefront workspace reservation estimate overflows size_t");
  wavefront_workspace_bytes_reserved_ =
      wavefront_field_workers_ * combined_bytes_each;
  if (configuration.adaptive_grid_mode >= 2)
    {
      const std::size_t total_context_cap =
          configuration.adaptive_window_policy.max_bytes;
      const std::size_t per_lane_adaptive_cap =
          total_context_cap / wavefront_field_workers_;
      if (per_lane_adaptive_cap < adaptive_transient_bytes_each ||
          per_lane_adaptive_cap - adaptive_transient_bytes_each <=
              configuration.adaptive_window_policy.fixed_bytes ||
          per_lane_adaptive_cap - adaptive_transient_bytes_each <
              adaptive_context_bytes_each)
        throw std::runtime_error(
            "adaptive_grid_max_bytes cannot provide the complete declared "
            "uniform/patch context bank plus transient controller high-water "
            "per memory-qualified worker lane");
      configuration.adaptive_window_policy.max_bytes =
          per_lane_adaptive_cap - adaptive_transient_bytes_each;
    }

  wavefront_field_workspaces_.clear();
  wavefront_field_workspaces_.reserve(wavefront_field_workers_);
  // FFTW plan construction is intentionally serial.  Only execution of the
  // completed, disjoint plans is released to the persistent workers.
  for (std::size_t lane = 0; lane < wavefront_field_workers_; ++lane)
    {
      std::unique_ptr<feebl::wavefront_fields::Workspace> workspace(
          new feebl::wavefront_fields::Workspace());
      workspace->initialize(configuration, &beam1_, &beam2_);
      if (switches.get_wavefront_parallel_physics())
        {
          const pairmoments::SpectrumW& spectrum =
              grid_.get_results().pair_spectrum();
          workspace->initialize_physics(
              &switches, &secondaries_, &muons_,
              secondaries_.get_pair_parameters().get_s4(),
              secondaries_.get_pair_parameters().get_lns4(),
              spectrum.lower_edge(), spectrum.upper_edge(), spectrum.bins(),
              static_cast<std::uint64_t>(switches.get_rndm_seed()),
              &grid_.design_axis_spectrum_);
        }
      wavefront_field_workspaces_.push_back(std::move(workspace));
    }
  wavefront_field_prefetch_.resize(
      static_cast<std::size_t>(grid_.get_n_cell_z()));
  wavefront_physics_lane_for_position_.assign(
      static_cast<std::size_t>(grid_.get_n_cell_z()),
      std::numeric_limits<std::size_t>::max());
#ifdef FEEBL_HAVE_GPU5
  if (gpu5_adaptive)
    {
      feebl::adaptive_grid::CanonicalLatticeInput lattice;
      lattice.n_x = static_cast<std::size_t>(configuration.n_x);
      lattice.n_y = static_cast<std::size_t>(configuration.n_y);
      lattice.delta_x = grid_.get_delta_x();
      lattice.delta_y = grid_.get_delta_y();
      lattice.delta_z = grid_.get_delta_z();
      lattice.inverse_delta_x = grid_.get_delta_x_inverse();
      lattice.inverse_delta_y = grid_.get_delta_y_inverse();
      lattice.offset_x = grid_.get_offset_x();
      lattice.offset_y = grid_.get_offset_y();
      lattice.intended_min_x = grid_.get_min_x();
      lattice.intended_max_x = grid_.get_max_x();
      lattice.intended_min_y = grid_.get_min_y();
      lattice.intended_max_y = grid_.get_max_y();
      lattice.charge_sign = configuration.charge_sign;
      lattice.field_kernel_factor = grid_.get_field_kernel_factor();
      lattice.geometry_revision = configuration.canonical_geometry_revision;
      gpu5_lattice_.reset(
          new feebl::adaptive_grid::CanonicalLatticeSnapshot(lattice));
      if (gpu5_lattice_->hash() !=
          wavefront_field_workspaces_[0]->canonical_lattice_hash())
        throw std::logic_error(
            "gpu_mode=5 canonical lattice disagrees with the authoritative "
            "CPU planner lattice");

      const std::vector<std::vector<PRIMARY_PARTICLE_ID> >& ids1 =
          beam1_.particle_beam().getParticleIdSlicesStrict();
      const std::vector<std::vector<PRIMARY_PARTICLE_ID> >& ids2 =
          beam2_.particle_beam().getParticleIdSlicesStrict();
      if (ids1.size() != static_cast<std::size_t>(configuration.n_z) ||
          ids2.size() != static_cast<std::size_t>(configuration.n_z))
        throw std::runtime_error(
            "gpu_mode=5 primary-ID sidecars do not match n_z");
      std::size_t maximum_particles1 = 0U;
      std::size_t maximum_particles2 = 0U;
      gpu5_beam1_order_tokens_.resize(ids1.size());
      gpu5_beam2_order_tokens_.resize(ids2.size());
      for (std::size_t slice = 0; slice < ids1.size(); ++slice)
        {
          maximum_particles1 = std::max(maximum_particles1,
                                        ids1[slice].size());
          maximum_particles2 = std::max(maximum_particles2,
                                        ids2[slice].size());
          gpu5_beam1_order_tokens_[slice] = primary_slice_order_token(
              ids1[slice], 1U, static_cast<std::uint64_t>(slice));
          gpu5_beam2_order_tokens_[slice] = primary_slice_order_token(
              ids2[slice], 2U, static_cast<std::uint64_t>(slice));
        }
      gpu5_beam1_revisions_.assign(ids1.size(), 1U);
      gpu5_beam2_revisions_.assign(ids2.size(), 1U);

      const GPU2_OPTIONS& requested = backend.actual_gpu2_options();
      gpu5_batch_capacity_ = static_cast<std::size_t>(
          requested.batch_slices);
      feebl::adaptive_grid::gpu5::AdaptiveGpuOptions options;
      options.device = backend.gpu_device();
      options.batch_capacity = gpu5_batch_capacity_;
      options.maximum_beam1_descriptors =
          std::max<std::size_t>(1U, maximum_particles1);
      options.maximum_beam2_descriptors =
          std::max<std::size_t>(1U, maximum_particles2);
      options.memory_fraction = requested.memory_fraction;
      const std::uint64_t configured_device_cap =
          switches.get_adaptive_grid_max_bytes();
      if (configured_device_cap != 0U)
        {
          if (configured_device_cap > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()))
            throw std::overflow_error(
                "adaptive_grid_max_bytes does not fit GPU size_t");
          options.maximum_device_bytes = static_cast<std::size_t>(
              configured_device_cap);
        }
      options.pinned_io = true;
      options.deposition =
          feebl::adaptive_grid::gpu5::DepositionVariantId::GlobalAtomic;
      gpu5_wave_solver_.reset(
          new feebl::adaptive_grid::gpu5::AdaptiveWaveBatchExecutor());
      std::string gpu5_error;
      if (!gpu5_wave_solver_->initialize(options, *gpu5_lattice_, gpu5_error))
        throw std::runtime_error(
            "gpu_mode=5 executor preflight failed: " + gpu5_error);

      std::vector<feebl::adaptive_grid::gpu5::WindowClassShape> classes;
      const feebl::adaptive_grid::UniformWindowPolicy& policy =
          configuration.adaptive_window_policy;
      for (std::size_t n_x : policy.size_classes_x)
        for (std::size_t n_y : policy.size_classes_y)
          if (n_x >= policy.min_n_x && n_y >= policy.min_n_y &&
              n_x <= policy.max_n_x && n_y <= policy.max_n_y &&
              n_x <= lattice.n_x && n_y <= lattice.n_y)
            classes.push_back(
                feebl::adaptive_grid::gpu5::WindowClassShape{n_x, n_y});
      std::sort(classes.begin(), classes.end());
      classes.erase(std::unique(classes.begin(), classes.end()),
                    classes.end());
      if (classes.empty())
        throw std::runtime_error(
            "gpu_mode=5 has no reachable adaptive window class");
      if (!gpu5_wave_solver_->prebuild_classes(
              classes.data(), classes.size(), gpu5_error))
        throw std::runtime_error(
            "gpu_mode=5 complete class-bank prebuild failed: " + gpu5_error);
      if (!gpu5_wave_solver_->prebuild_complete())
        throw std::logic_error(
            "gpu_mode=5 executor did not commit complete class prebuild");
      backend.record_gpu_mode5_plan(requested);
      const feebl::adaptive_grid::gpu5::AdaptiveGpuTelemetry telemetry =
          gpu5_wave_solver_->telemetry();
      std::cerr
          << "FEEBL_GPU5_PREBUILD_V1"
          << " classes=" << telemetry.classes_prebuilt
          << " batch_capacity=" << gpu5_batch_capacity_
          << " max_beam1_descriptors="
          << options.maximum_beam1_descriptors
          << " max_beam2_descriptors="
          << options.maximum_beam2_descriptors
          << " lattice_hash=" << gpu5_lattice_->hash()
          << " plan_creations=" << telemetry.plan_creations
          << " green_creations=" << telemetry.green_creations
          << " tracked_device_bytes=" << telemetry.tracked_device_bytes
          << " peak_tracked_device_bytes="
          << telemetry.peak_tracked_device_bytes
          << " measured_device_high_water_bytes="
          << telemetry.measured_device_high_water_bytes
          << " memory_budget_bytes=" << telemetry.memory_budget_bytes
          << " deposition="
          << feebl::adaptive_grid::gpu5::deposition_variant_name(
                 telemetry.actual_deposition)
          << " cold_miss_policy=forbidden"
          << " device=\"" << gpu5_wave_solver_->device_description()
          << "\"" << std::endl;
    }
#else
  if (gpu5_adaptive)
    throw std::logic_error(
        "gpu_mode=5 selected in a build without FEEBL_HAVE_GPU5");
#endif
  wavefront_field_setup_ms_ =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
  std::cerr
      << " FEEBL: wavefront field workspace preflight: active_workers="
      << wavefront_field_workers_
      << " backend_threads=" << backend.actual_cpu_threads()
      << " bytes_each_estimate=" << wavefront_workspace_bytes_each_
      << " adaptive_context_bytes_each_estimate="
      << adaptive_context_bytes_each
      << " adaptive_transient_bytes_each_estimate="
      << adaptive_transient_bytes_each
      << " adaptive_total_bytes_each_estimate="
      << adaptive_bytes_each
      << " design_axis_bytes_each_estimate=" << design_axis_bytes_each
      << " bytes_reserved_estimate="
      << wavefront_workspace_bytes_reserved_
      << " base_budget_bytes=" << base_workspace_budget_bytes
      << " adaptive_reservation_budget_bytes="
      << adaptive_workspace_reservation_bytes
      << " budget_bytes=" << workspace_budget_bytes
      << " requested_budget_bytes=" << requested_workspace_budget
      << " field_solver="
      << (gpu5_adaptive ? "gpu5-adaptive-class-planmany"
          : (gpu4_parallel_physics ? "gpu-prefetched"
                                : switches.get_wavefront_field_solver())
          )
      << " parallel_physics="
      << switches.get_wavefront_parallel_physics()
      << " sparse_direct_interaction_limit="
      << switches.get_sparse_direct_interaction_limit()
      << " adaptive_grid_requested="
      << configuration.adaptive_grid_mode
      << " adaptive_grid_actual="
      << configuration.adaptive_grid_mode
      << " adaptive_scope=" << switches.get_adaptive_grid_scope()
      << " adaptive_controller="
      << (configuration.adaptive_grid_mode == 1
              ? "uniform-window-v1"
              : (configuration.adaptive_grid_mode == 2
                     ? "error-controlled-resolution-v1"
                     : (configuration.adaptive_grid_mode == 3
                            ? "nested-dirichlet-patch-v1" : "fixed")))
      << " adaptive_class_x=\""
      << switches.get_adaptive_grid_size_classes_x() << "\""
      << " adaptive_class_y=\""
      << switches.get_adaptive_grid_size_classes_y() << "\""
      << " adaptive_max_cells=" << configuration.adaptive_max_cells
      << " adaptive_max_bytes="
      << switches.get_adaptive_grid_max_bytes()
      << " adaptive_context_bytes_per_lane_cap="
      << configuration.adaptive_window_policy.max_bytes
      << " adaptive_full_window_allowed="
      << switches.get_adaptive_grid_allow_full_window()
      << " adaptive_spacing_factors=\""
      << switches.get_adaptive_grid_field_spacing_classes() << "\""
      << " adaptive_field_operator="
      << configuration.adaptive_field_operator
      << " adaptive_max_refinement_factor="
      << configuration.adaptive_max_refinement_factor
      << " adaptive_error_target="
      << configuration.adaptive_error_target
      << " adaptive_error_budget_sha256="
      << (configuration.adaptive_error_budget_sha256.empty()
              ? "not_applicable"
              : configuration.adaptive_error_budget_sha256)
      << " adaptive_richardson_order_source="
      << configuration.adaptive_richardson_order_source
      << " adaptive_richardson_interval_requested="
      << switches.get_adaptive_grid_richardson_interval()
      << " adaptive_richardson_interval_actual="
      << (configuration.adaptive_grid_mode >= 2 ? 1 : 0)
      << " adaptive_update_interval_requested="
      << switches.get_adaptive_grid_update_interval()
      << " adaptive_update_interval_actual="
      << (configuration.adaptive_grid_mode >= 2 ? 1 : 0)
      << " adaptive_controller_history="
      << (configuration.adaptive_grid_mode >= 2 ? "stateless" : "none")
      << " adaptive_refinement_ratio="
      << configuration.adaptive_refinement_ratio
      << " adaptive_fallbacks=0"
      << std::endl;
}

void FEEBL::beam_interaction_wavefront_parallel(PHI_FLOAT* sor_parameter)
{
  const int n_slice = grid_.get_n_cell_z();
  const int n_substeps = grid_.get_timestep();
  if (n_slice <= 0 || n_substeps <= 0)
    throw std::logic_error(
        "parallel wavefront scheduler requires positive n_z and n_t");
  PARALLEL_BACKEND& backend = parallel_backend();
  if (wavefront_field_workspaces_.size() != wavefront_field_workers_ ||
      wavefront_field_workers_ < 2U)
    throw std::logic_error(
        "parallel wavefront scheduler entered before field initialization");

  const feebl::legacy_wavefront::Scheduler scheduler(
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SliceIndex>(n_slice),
      static_cast<feebl::modern::SubstepIndex>(n_substeps));
  LEGACY_WAVEFRONT_CONTEXT boundary_context;
  boundary_context.simulation = this;
  boundary_context.sor_parameter = sor_parameter;
  boundary_context.scheduler = &scheduler;
  WAVEFRONT_FIELD_CONTEXT field_context;
  field_context.simulation = this;
  field_context.scheduler = &scheduler;
  field_context.wave = NULL;
  field_context.dispatch_order = NULL;
  field_context.position_offset = 0;
  WAVEFRONT_PHYSICS_CONTEXT physics_context;
  physics_context.simulation = this;
  physics_context.scheduler = &scheduler;
  physics_context.wave = NULL;
  physics_context.prefetch = &wavefront_field_prefetch_;
  physics_context.position_offset = 0;
  physics_context.prefetch_offset = 0;
  physics_context.legacy_time_base = 0;

  std::uint64_t encounters = 0;
  std::uint64_t field_dispatch_batches = 0;
  std::uint64_t parallel_field_encounters = 0;
  std::size_t maximum_wave_width = 0;
  std::size_t maximum_nonempty_lanes = 0;
  double field_prefetch_ms = 0.0;
  double parallel_physics_ms = 0.0;
  double ordered_commit_ms = 0.0;
  double adaptive_planning_ms = 0.0;
  double adaptive_bucketing_ms = 0.0;
  double adaptive_context_setup_ms = 0.0;
  std::uint64_t adaptive_particle_encounters = 0U;
  std::uint64_t adaptive_particle_commits = 0U;
  std::uint64_t adaptive_particle_root_visits1 = 0U;
  std::uint64_t adaptive_particle_root_visits2 = 0U;
  std::uint64_t adaptive_particle_premerge_visits1 = 0U;
  std::uint64_t adaptive_particle_premerge_visits2 = 0U;
  std::uint64_t adaptive_particle_representative_visits1 = 0U;
  std::uint64_t adaptive_particle_representative_visits2 = 0U;
  std::uint64_t adaptive_particle_many_root_visits1 = 0U;
  std::uint64_t adaptive_particle_many_root_visits2 = 0U;
  std::size_t adaptive_particle_peak_representatives1 = 0U;
  std::size_t adaptive_particle_peak_representatives2 = 0U;
  double adaptive_particle_min_clone_support1 =
      std::numeric_limits<double>::infinity();
  double adaptive_particle_min_clone_support2 =
      std::numeric_limits<double>::infinity();
  double adaptive_particle_max_weight_ess1 = 0.0;
  double adaptive_particle_max_weight_ess2 = 0.0;
  double adaptive_particle_max_projection_residual = 0.0;
  std::uint64_t adaptive_particle_trace_hash =
      UINT64_C(1469598103934665603);
  std::unique_ptr<feebl::adaptive_particles::CoupledController>
      adaptive_particle_controller;
  if (switches.get_adaptive_macroparticle_mode() != 0)
    adaptive_particle_controller.reset(
        new feebl::adaptive_particles::CoupledController(
            switches.get_adaptive_macroparticle_configuration()));
  std::uint64_t adaptive_particle_controller_observations = 0U;
  std::uint64_t adaptive_particle_controller_sampling_unknown = 0U;
  std::uint64_t adaptive_particle_controller_actions[7] = {};
  std::uint64_t adaptive_particle_controller_recommendations[7] = {};
  bool adaptive_particle_controller_unmet = false;
  bool adaptive_particle_controller_replay_required = false;
  std::string adaptive_particle_controller_first_unmet_reason;
  feebl::adaptive_particles::ControllerAction
      adaptive_particle_controller_replay_recommendation =
          feebl::adaptive_particles::ControllerAction::Hold;
  const auto mix_adaptive_particle_trace =
      [&adaptive_particle_trace_hash](std::uint64_t value) {
        for (unsigned int byte = 0U; byte < 8U; ++byte) {
          adaptive_particle_trace_hash ^=
              (value >> (8U * byte)) & UINT64_C(0xff);
          adaptive_particle_trace_hash *= UINT64_C(1099511628211);
        }
      };
  std::map<std::pair<std::size_t, std::size_t>, std::uint64_t>
      adaptive_class_encounters;
  std::unique_ptr<feebl::adaptive_grid::AdaptiveGridSchedule>
      adaptive_schedule;
  std::unique_ptr<feebl::adaptive_grid::AdaptiveGridSchedule>
      replay_schedule;
  std::size_t adaptive_schedule_cursor = 0U;
  std::string adaptive_schedule_hash =
      switches.get_adaptive_grid_mode() >= 2
          ? "disabled-mode23-requires-v2" : std::string();
  std::string adaptive_card_hash;
  std::uint64_t adaptive_lattice_hash = 0U;
  if (switches.get_adaptive_grid_mode() >= 1)
    {
      adaptive_lattice_hash =
          wavefront_field_workspaces_[0]->canonical_lattice_hash();
      if (adaptive_lattice_hash == 0U)
        throw std::logic_error(
            "adaptive schedule preflight found no canonical lattice hash");
      adaptive_card_hash = adaptive_schedule_card_sha256(
          switches, grid_, mode3_input_card_label_, adaptive_lattice_hash);
    }
  if (switches.get_adaptive_grid_mode() == 1)
    {
      const feebl::adaptive_grid::AdaptiveGridScheduleHeader header(
          adaptive_card_hash, adaptive_lattice_hash, FEEBL_BUILD_SOURCE_HASH,
          std::string(PACKAGE_VERSION) + "-adaptive-schedule-v1",
          "uniform-window-v1",
          primary_particle_ordering_sha256(
              beam1_.particle_beam(), beam2_.particle_beam()));
      adaptive_schedule.reset(
          new feebl::adaptive_grid::AdaptiveGridSchedule(header));
      if (!switches.get_adaptive_grid_schedule_load().empty())
        {
          replay_schedule.reset(
              new feebl::adaptive_grid::AdaptiveGridSchedule(
                  feebl::adaptive_grid::AdaptiveGridSchedule::load(
                      switches.get_adaptive_grid_schedule_load())));
          if (replay_schedule->header() != header)
            throw std::runtime_error(
                "adaptive schedule replay rejected before collision: "
                "card/lattice/source/build/controller/particle-ordering "
                "header mismatch");
        }
    }

  for (const feebl::legacy_wavefront::DiagonalDescriptor& diagonal :
       scheduler.diagonals())
    {
      wavefront_begin_diagonal(&boundary_context, diagonal);
      for (feebl::modern::SubstepIndex substep = 0;
           substep < scheduler.dag().substep_count(); ++substep)
        {
          const feebl::modern::WaveIndex wave_index =
              diagonal.diagonal * scheduler.dag().substep_count() + substep;
          const feebl::modern::EncounterWave& wave =
              scheduler.dag().waves()[static_cast<std::size_t>(wave_index)];
          wavefront_begin_wave(&boundary_context, wave);
          field_context.wave = &wave;
          physics_context.wave = &wave;
          // Freeze pre-kick diagnostics now, but do not let the controller
          // observe or decide until this wave's canonical publication barrier
          // has completed below.
          std::vector<feebl::adaptive_particles::ControllerObservation>
              pending_adaptive_particle_observations;

          if (switches.get_adaptive_grid_mode() >= 1)
            {
              // Freeze every pre-kick descriptor in canonical encounter
              // order, then group only the RNG-free field work.  Physics and
              // publication below remain in canonical order.
              const std::chrono::steady_clock::time_point planning_start =
                  std::chrono::steady_clock::now();
              std::vector<std::size_t> field_order(wave.encounters.size());
              std::map<std::pair<std::size_t, std::size_t>, std::size_t>
                  class_estimates;
              field_context.dispatch_order = NULL;
              field_context.position_offset = 0U;
              backend.run_static_ranges_limited(
                  wave.encounters.size(), wavefront_field_workers_,
                  &field_context, &FEEBL::wavefront_plan_field_range);
              for (std::size_t position = 0;
                   position < wave.encounters.size(); ++position)
                {
                  field_order[position] = position;
                  const feebl::modern::EncounterNode& encounter =
                      scheduler.dag().node(wave.encounters[position]);
                  feebl::wavefront_fields::FieldPrefetch& prefetch =
                      wavefront_field_prefetch_[position];
                  const feebl::adaptive_grid::UniformWindowPlan& plan =
                      *prefetch.window_plan;
                  const bool empty_plan = plan.empty();
                  const std::uint64_t local_n_x = empty_plan
                      ? 0U : checked_size_to_u64(
                          plan.local_n_x(), "schedule local n_x");
                  const std::uint64_t local_n_y = empty_plan
                      ? 0U : checked_size_to_u64(
                          plan.local_n_y(), "schedule local n_y");
                  if (adaptive_schedule)
                    {
                      const feebl::adaptive_grid::AdaptiveGridEncounterRecord
                          schedule_record(
                          feebl::adaptive_grid::AdaptiveGridEncounterKey{
                              0U,
                              static_cast<std::uint64_t>(
                                  diagonal.diagonal),
                              static_cast<std::uint64_t>(wave_index),
                              static_cast<std::uint64_t>(substep),
                              checked_size_to_u64(
                                  adaptive_schedule_cursor,
                                  "adaptive schedule encounter index")},
                          static_cast<std::int64_t>(encounter.beam1_slice),
                          static_cast<std::int64_t>(encounter.beam2_slice),
                          empty_plan ? 0 : plan.origin_x(),
                          empty_plan ? 0 : plan.origin_y(),
                          local_n_x, local_n_y,
                          empty_plan ? 0U : checked_size_to_u64(
                              plan.padded_n_x(), "schedule padded n_x"),
                          empty_plan ? 0U : checked_size_to_u64(
                              plan.padded_n_y(), "schedule padded n_y"),
                          plan.plan_hash(),
                          empty_plan
                              ? std::string("empty")
                              : std::to_string(local_n_x) + "x" +
                                    std::to_string(local_n_y),
                          checked_size_to_u64(
                              plan.accepted_source_count(),
                              "schedule accepted source count"),
                          checked_size_to_u64(
                              plan.accepted_target_count(),
                              "schedule accepted target count"),
                              !empty_plan && plan.full_window());
                      adaptive_schedule->append(schedule_record);
                      if (replay_schedule)
                        {
                          if (adaptive_schedule_cursor >=
                                  replay_schedule->records().size() ||
                              replay_schedule->records()[
                                  adaptive_schedule_cursor] != schedule_record)
                            throw std::runtime_error(
                                "adaptive schedule replay rejected before "
                                "field or particle mutation at canonical "
                                "encounter " +
                                std::to_string(adaptive_schedule_cursor));
                        }
                      ++adaptive_schedule_cursor;
                    }
                  if (!prefetch.window_plan->empty())
                    {
                      const std::pair<std::size_t, std::size_t> shape(
                          prefetch.window_plan->local_n_x(),
                          prefetch.window_plan->local_n_y());
                      if (!prefetch.window_plan->full_window())
                        class_estimates.emplace(
                            shape, prefetch.window_plan->estimated_bytes());
                      checked_add_u64(
                          adaptive_class_encounters[shape], 1,
                          "adaptive class encounter");
                    }
                }
              adaptive_planning_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - planning_start)
                      .count();

              const std::chrono::steady_clock::time_point setup_start =
                  std::chrono::steady_clock::now();
              const std::uint64_t requested_cap =
                  switches.get_adaptive_grid_max_bytes();
              const std::size_t total_cap = static_cast<std::size_t>(
                  requested_cap == 0U
                      ? UINT64_C(4) * 1024U * 1024U * 1024U
                      : requested_cap);
              // Mode 1 uses the raw window geometry and benefits from eager
              // shape prebuilds.  Modes 2/3 add factor-dependent guards and
              // spacing, so these raw factor-one contexts would be unused;
              // their bounded per-factor cache is populated on first use.
              if (!backend.gpu_mode5_enabled() &&
                  switches.get_adaptive_grid_mode() == 1)
                for (const auto& item : class_estimates)
                  {
                  std::size_t current_bytes = 0U;
                  std::size_t missing = 0U;
                  bool wisdom_available = false;
                  for (const auto& workspace : wavefront_field_workspaces_)
                    {
                      if (workspace->adaptive_context_bytes() >
                          std::numeric_limits<std::size_t>::max() -
                              current_bytes)
                        throw std::overflow_error(
                            "adaptive context-pool byte total overflow");
                      current_bytes += workspace->adaptive_context_bytes();
                      if (workspace->has_adaptive_class(
                              item.first.first, item.first.second))
                        wisdom_available = true;
                      else
                        ++missing;
                    }
                  if (missing != 0U &&
                      item.second >
                          (total_cap - std::min(total_cap, current_bytes)) /
                              missing)
                    throw std::runtime_error(
                        "adaptive class prebuild nx=" +
                        std::to_string(item.first.first) + " ny=" +
                        std::to_string(item.first.second) +
                        " estimated_bytes_each=" +
                        std::to_string(item.second) + " missing_lanes=" +
                        std::to_string(missing) + " current_context_bytes=" +
                        std::to_string(current_bytes) + " cap_bytes=" +
                        std::to_string(total_cap) + " would exceed "
                        "adaptive_grid_max_bytes before allocation; no "
                        "encounter in the failing wave and no fallback was "
                        "executed");
                  for (auto& workspace : wavefront_field_workspaces_)
                    if (!workspace->has_adaptive_class(
                            item.first.first, item.first.second))
                      {
                        workspace->prebuild_adaptive_class(
                            item.first.first, item.first.second,
                            wisdom_available);
                        wisdom_available = true;
                      }
                  }
              adaptive_context_setup_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - setup_start).count();

              const std::chrono::steady_clock::time_point bucket_start =
                  std::chrono::steady_clock::now();
              std::stable_sort(
                  field_order.begin(), field_order.end(),
                  [this](std::size_t first, std::size_t second) {
                    const auto& first_plan =
                        wavefront_field_prefetch_[first].window_plan;
                    const auto& second_plan =
                        wavefront_field_prefetch_[second].window_plan;
                    const std::pair<std::size_t, std::size_t> first_shape =
                        first_plan->empty()
                            ? std::pair<std::size_t, std::size_t>(0U, 0U)
                            : std::make_pair(first_plan->local_n_x(),
                                             first_plan->local_n_y());
                    const std::pair<std::size_t, std::size_t> second_shape =
                        second_plan->empty()
                            ? std::pair<std::size_t, std::size_t>(0U, 0U)
                            : std::make_pair(second_plan->local_n_x(),
                                             second_plan->local_n_y());
                    return first_shape < second_shape;
                  });
              adaptive_bucketing_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - bucket_start).count();
#ifdef FEEBL_HAVE_GPU5
              if (backend.gpu_mode5_enabled())
                {
                  if (!gpu5_wave_solver_ ||
                      !gpu5_wave_solver_->prebuild_complete() ||
                      gpu5_batch_capacity_ == 0U)
                    throw std::logic_error(
                        "gpu_mode=5 collision entered without its committed "
                        "class bank and batch capacity");
                  std::vector<
                      feebl::adaptive_grid::gpu5::AdaptiveEncounterResult>
                      gpu_results(wave.encounters.size());
                  std::size_t class_begin = 0U;
                  while (class_begin < field_order.size())
                    {
                      const auto& first_plan = wavefront_field_prefetch_[
                          field_order[class_begin]].window_plan;
                      const std::pair<std::size_t, std::size_t> shape =
                          first_plan->empty()
                              ? std::make_pair(std::size_t(0), std::size_t(0))
                              : std::make_pair(first_plan->local_n_x(),
                                               first_plan->local_n_y());
                      std::size_t class_end = class_begin + 1U;
                      while (class_end < field_order.size())
                        {
                          const auto& plan = wavefront_field_prefetch_[
                              field_order[class_end]].window_plan;
                          const std::pair<std::size_t, std::size_t> next_shape =
                              plan->empty()
                                  ? std::make_pair(std::size_t(0),
                                                   std::size_t(0))
                                  : std::make_pair(plan->local_n_x(),
                                                   plan->local_n_y());
                          if (next_shape != shape) break;
                          ++class_end;
                        }
                      for (std::size_t begin = class_begin;
                           begin < class_end; begin += gpu5_batch_capacity_)
                        {
                          const std::size_t count = std::min<std::size_t>(
                              gpu5_batch_capacity_, class_end - begin);
                          std::vector<feebl::adaptive_grid::gpu5::
                              AdaptiveEncounterSubmission> submissions(count);
                          for (std::size_t slot = 0; slot < count; ++slot)
                            {
                              const std::size_t position =
                                  field_order[begin + slot];
                              const feebl::modern::EncounterNode& encounter =
                                  scheduler.dag().node(
                                      wave.encounters[position]);
                              const std::size_t slice1 =
                                  static_cast<std::size_t>(
                                      encounter.beam1_slice);
                              const std::size_t slice2 =
                                  static_cast<std::size_t>(
                                      encounter.beam2_slice);
                              if (slice1 >= gpu5_beam1_revisions_.size() ||
                                  slice2 >= gpu5_beam2_revisions_.size())
                                throw std::out_of_range(
                                    "gpu_mode=5 encounter slice exceeds its "
                                    "revision/order-token bank");
                              auto& submission = submissions[slot];
                              submission.plan = wavefront_field_prefetch_[
                                  position].window_plan.get();
                              submission.binding.canonical_result_slot =
                                  position;
                              submission.binding.encounter_id =
                                  static_cast<std::uint64_t>(encounter.id);
                              submission.binding.beam1_slice =
                                  static_cast<std::int64_t>(slice1);
                              submission.binding.beam2_slice =
                                  static_cast<std::int64_t>(slice2);
                              submission.binding.beam1_revision =
                                  gpu5_beam1_revisions_[slice1];
                              submission.binding.beam2_revision =
                                  gpu5_beam2_revisions_[slice2];
                              submission.binding.beam1_order_token =
                                  gpu5_beam1_order_tokens_[slice1];
                              submission.binding.beam2_order_token =
                                  gpu5_beam2_order_tokens_[slice2];
                            }
                          const std::chrono::steady_clock::time_point
                              field_start =
                                  std::chrono::steady_clock::now();
                          std::string gpu5_error;
                          if (!gpu5_wave_solver_->execute_batch(
                                  submissions.data(), submissions.size(),
                                  grid_.get_macro_charge(1),
                                  grid_.get_macro_charge(2),
                                  gpu_results.data(), gpu_results.size(),
                                  gpu5_error))
                            throw std::runtime_error(
                                "gpu_mode=5 adaptive class batch failed "
                                "without fallback: " + gpu5_error);
                          for (std::size_t slot = 0; slot < count; ++slot)
                            {
                              const std::size_t position =
                                  submissions[slot].binding.
                                      canonical_result_slot;
                              auto& result = gpu_results[position];
                              if (!gpu5_wave_solver_->validate_result_binding(
                                      result, submissions[slot],
                                      grid_.get_macro_charge(1),
                                      grid_.get_macro_charge(2), gpu5_error))
                                throw std::runtime_error(
                                    "gpu_mode=5 result rejected before host "
                                    "physics mutation: " + gpu5_error);
                              const feebl::modern::EncounterNode& encounter =
                                  scheduler.dag().node(
                                      wave.encounters[position]);
                              const int slice1 = static_cast<int>(
                                  encounter.beam1_slice);
                              const int slice2 = static_cast<int>(
                                  encounter.beam2_slice);
                              auto& output =
                                  wavefront_field_prefetch_[position];
                              output.fields1.swap(result.fields1);
                              output.fields2.swap(result.fields2);
                              output.fine_lumi_dot = result.fine_lumi_dot;
                              output.beam1_in = result.beam1_in;
                              output.beam1_out = result.beam1_out;
                              output.beam2_in = result.beam2_in;
                              output.beam2_out = result.beam2_out;
                              feebl::wavefront_fields::bind_field_result(
                                  output, slice1, slice2,
                                  beam1_.getParticleVector(slice1),
                                  beam2_.getParticleVector(slice2));
                            }
                          backend.record_gpu_mode5_execution(
                              gpu5_wave_solver_->device_description());
                          field_prefetch_ms +=
                              std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() -
                                  field_start).count();
                          checked_add_u64(field_dispatch_batches, 1,
                                          "gpu_mode=5 class batch");
                        }
                      class_begin = class_end;
                    }
                }
              else
#endif
                {
                  field_context.dispatch_order = &field_order;
                  for (std::size_t begin = 0; begin < field_order.size();
                       begin += wavefront_field_workers_)
                    {
                      field_context.position_offset = begin;
                      const std::size_t count = std::min<std::size_t>(
                          wavefront_field_workers_,
                          field_order.size() - begin);
                      const std::chrono::steady_clock::time_point field_start =
                          std::chrono::steady_clock::now();
                      backend.run_static_ranges(
                          count, &field_context,
                          &FEEBL::wavefront_prepare_field_range);
                      field_prefetch_ms +=
                          std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - field_start)
                              .count();
                      checked_add_u64(field_dispatch_batches, 1,
                                      "adaptive field dispatch batch");
                    }
                  field_context.dispatch_order = NULL;
                }

              if (switches.get_adaptive_macroparticle_mode() != 0)
                for (std::size_t position = 0U;
                     position < wave.encounters.size(); ++position)
                  {
                    const feebl::wavefront_fields::FieldPrefetch& receipt =
                        wavefront_field_prefetch_[position];
                    ++adaptive_particle_encounters;
                    if (receipt.adaptive_macroparticle_committed)
                      ++adaptive_particle_commits;
                    checked_add_u64(
                        adaptive_particle_root_visits1,
                        receipt.adaptive_macroparticle_roots1,
                        "adaptive particle beam-1 root visits");
                    checked_add_u64(
                        adaptive_particle_root_visits2,
                        receipt.adaptive_macroparticle_roots2,
                        "adaptive particle beam-2 root visits");
                    checked_add_u64(
                        adaptive_particle_premerge_visits1,
                        receipt.adaptive_macroparticle_premerge1,
                        "adaptive particle beam-1 premerge visits");
                    checked_add_u64(
                        adaptive_particle_premerge_visits2,
                        receipt.adaptive_macroparticle_premerge2,
                        "adaptive particle beam-2 premerge visits");
                    checked_add_u64(
                        adaptive_particle_representative_visits1,
                        receipt.adaptive_macroparticle_representatives1,
                        "adaptive particle beam-1 representative visits");
                    checked_add_u64(
                        adaptive_particle_representative_visits2,
                        receipt.adaptive_macroparticle_representatives2,
                        "adaptive particle beam-2 representative visits");
                    checked_add_u64(
                        adaptive_particle_many_root_visits1,
                        receipt.adaptive_macroparticle_many_root1,
                        "adaptive particle beam-1 many-root visits");
                    checked_add_u64(
                        adaptive_particle_many_root_visits2,
                        receipt.adaptive_macroparticle_many_root2,
                        "adaptive particle beam-2 many-root visits");
                    adaptive_particle_peak_representatives1 = std::max(
                        adaptive_particle_peak_representatives1,
                        receipt.adaptive_macroparticle_representatives1);
                    adaptive_particle_peak_representatives2 = std::max(
                        adaptive_particle_peak_representatives2,
                        receipt.adaptive_macroparticle_representatives2);
                    if (receipt.adaptive_macroparticle_clone_support1 > 0.0)
                      adaptive_particle_min_clone_support1 = std::min(
                          adaptive_particle_min_clone_support1,
                          receipt.adaptive_macroparticle_clone_support1);
                    if (receipt.adaptive_macroparticle_clone_support2 > 0.0)
                      adaptive_particle_min_clone_support2 = std::min(
                          adaptive_particle_min_clone_support2,
                          receipt.adaptive_macroparticle_clone_support2);
                    adaptive_particle_max_weight_ess1 = std::max(
                        adaptive_particle_max_weight_ess1,
                        receipt.adaptive_macroparticle_weight_ess1);
                    adaptive_particle_max_weight_ess2 = std::max(
                        adaptive_particle_max_weight_ess2,
                        receipt.adaptive_macroparticle_weight_ess2);
                    adaptive_particle_max_projection_residual = std::max(
                        adaptive_particle_max_projection_residual,
                        receipt.adaptive_macroparticle_projection_residual);
                    mix_adaptive_particle_trace(
                        receipt.adaptive_macroparticle_hash1);
                    mix_adaptive_particle_trace(
                        receipt.adaptive_macroparticle_hash2);
                    mix_adaptive_particle_trace(
                        receipt.adaptive_macroparticle_revision1);
                    mix_adaptive_particle_trace(
                        receipt.adaptive_macroparticle_revision2);
                    feebl::adaptive_particles::ControllerObservation
                        observation;
                    observation.epoch = adaptive_particle_encounters;
                    observation.error.mesh =
                        receipt.estimated_field_error;
                    // One self-consistent bunch realization cannot estimate
                    // its own initial-source sampling uncertainty.  Root and
                    // clone support are controller diagnostics, not
                    // independent qualification replicates.
                    observation.error.source_sampling = 0.0;
                    observation.error.source_sampling_known = false;
                    observation.error.resampling_bias =
                        receipt.adaptive_macroparticle_projection_residual;
                    observation.support.distinct_roots =
                        static_cast<double>(std::min(
                            receipt.adaptive_macroparticle_roots1,
                            receipt.adaptive_macroparticle_roots2));
                    observation.support.clone_support_proxy = std::min(
                        receipt.adaptive_macroparticle_clone_support1,
                        receipt.adaptive_macroparticle_clone_support2);
                    observation.support.protected_tail_support =
                        receipt.minimum_effective_particles_per_cell;
                    observation.support.independent_complete_runs = 1U;
                    observation.support.ancestry_exact = true;
                    observation.mesh_refinement_available =
                        receipt.error_target_met;
                    observation.initial_state_replay_available = false;
                    observation.conservative_merge_feasible =
                        receipt.adaptive_macroparticle_premerge1 >
                            receipt.adaptive_macroparticle_representatives1 ||
                        receipt.adaptive_macroparticle_premerge2 >
                            receipt.adaptive_macroparticle_representatives2;
                    pending_adaptive_particle_observations.push_back(
                        observation);
                  }

              if (switches.get_adaptive_grid_mode() >= 2 &&
                  switches.get_adaptive_grid_profile() == "encounter")
                for (std::size_t position = 0U;
                     position < wave.encounters.size(); ++position)
                  {
                    const feebl::modern::EncounterNode& encounter =
                        scheduler.dag().node(wave.encounters[position]);
                    const feebl::wavefront_fields::FieldPrefetch& receipt =
                        wavefront_field_prefetch_[position];
                    std::cerr
                        << "FEEBL_ADAPTIVE_ENCOUNTER_V1"
                        << " diagonal=" << diagonal.diagonal
                        << " wave=" << wave_index
                        << " substep=" << substep
                        << " encounter=" << encounter.id
                        << " beam1_slice=" << encounter.beam1_slice
                        << " beam2_slice=" << encounter.beam2_slice
                        << " selected_spacing_factor="
                        << receipt.selected_spacing_factor
                        << " selected_grid_nx="
                        << receipt.selected_grid_n_x
                        << " selected_grid_ny="
                        << receipt.selected_grid_n_y
                        << " selected_delta_x="
                        << receipt.selected_delta_x
                        << " selected_delta_y="
                        << receipt.selected_delta_y
                        << " estimated_field_error="
                        << receipt.estimated_field_error
                        << " error_norm_policy="
                        << switches.get_adaptive_grid_error_norm_policy()
                        << " difference_rms="
                        << receipt.field_difference_rms
                        << " difference_linf="
                        << receipt.field_difference_linf
                        << " component_linf_acceptance_gate="
                        << (switches.get_adaptive_grid_error_norm_policy() ==
                                    "component_rms_linf_v1"
                                ? 1 : 0)
                        << " vector_difference_rms="
                        << receipt.vector_difference_rms
                        << " vector_difference_quantile="
                        << receipt.vector_difference_quantile
                        << " vector_difference_maximum="
                        << receipt.vector_difference_maximum
                        << " vector_mixed_difference_maximum="
                        << receipt.vector_mixed_difference_maximum
                        << " vector_absolute_difference_rms="
                        << receipt.vector_absolute_difference_rms
                        << " vector_absolute_difference_quantile="
                        << receipt.vector_absolute_difference_quantile
                        << " vector_absolute_difference_maximum="
                        << receipt.vector_absolute_difference_maximum
                        << " vector_catastrophe_targets="
                        << receipt.vector_catastrophe_targets
                        << " vector_catastrophe_exposure="
                        << receipt.vector_catastrophe_exposure
                        << " vector_quantile_probability="
                        << receipt.vector_quantile_probability
                        << " vector_accepted_targets="
                        << receipt.vector_accepted_targets
                        << " vector_above_target="
                        << receipt.vector_above_target
                        << " vector_above_target_exposure="
                        << receipt.vector_above_target_exposure
                        << " vector_maximum_beam_above_target_exposure="
                        << receipt
                               .vector_maximum_beam_above_target_exposure
                        << " vector_catastrophe_guard_passed="
                        << (receipt.vector_catastrophe_guard_passed ? 1 : 0)
                        << " noise_aware_stop_triggered="
                        << (receipt.noise_aware_stop_triggered ? 1 : 0)
                        << " noise_aware_stop_factor="
                        << receipt.noise_aware_stop_factor
                        << " noise_aware_stop_quantile_root_limited="
                        << (receipt.noise_aware_stop_quantile_root_limited
                                ? 1 : 0)
                        << " noise_aware_stop_noncontracting="
                        << (receipt.noise_aware_stop_noncontracting ? 1 : 0)
                        << " vector_distinct_roots="
                        << receipt.vector_distinct_roots
                        << " vector_minimum_quantile_roots="
                        << receipt.vector_minimum_quantile_roots
                        << " tail_field_sensitivity_metric="
                        << "symmetric_vector_triplet_v1"
                        << " tail_field_sensitivity="
                        << receipt.tail_field_sensitivity
                        << " tail_field_sensitivity_previous_pair="
                        << receipt.tail_field_sensitivity_previous_pair
                        << " tail_field_sensitivity_latest_pair="
                        << receipt.tail_field_sensitivity_latest_pair
                        << " tail_uniform_lumi_measure="
                        << receipt.tail_uniform_lumi_measure
                        << " curvature_score="
                        << receipt.charge_curvature_score
                        << " field_gradient_score="
                        << receipt.field_gradient_score
                        << " neff_p10="
                        << receipt.effective_particles_per_cell
                        << " neff_critical_min="
                        << receipt.minimum_effective_particles_per_cell
                        << " source_reference_factor="
                        << receipt.source_reference_factor
                        << " source_model_tag=" << receipt.source_model_tag
                        << " source_beam1_in=" << receipt.beam1_in
                        << " source_beam1_out=" << receipt.beam1_out
                        << " source_beam2_in=" << receipt.beam2_in
                        << " source_beam2_out=" << receipt.beam2_out
                        << " source_density_hash="
                        << receipt.source_density_hash
                        << " source_sampling_hash="
                        << receipt.source_sampling_hash
                        << " source_sampling_hierarchy_hash="
                        << receipt.source_sampling_hierarchy_hash
                        << " adaptive_macroparticle_mode="
                        << receipt.adaptive_macroparticle_mode
                        << " adaptive_macroparticle_roots1="
                        << receipt.adaptive_macroparticle_roots1
                        << " adaptive_macroparticle_roots2="
                        << receipt.adaptive_macroparticle_roots2
                        << " adaptive_macroparticle_premerge1="
                        << receipt.adaptive_macroparticle_premerge1
                        << " adaptive_macroparticle_premerge2="
                        << receipt.adaptive_macroparticle_premerge2
                        << " adaptive_macroparticle_representatives1="
                        << receipt.adaptive_macroparticle_representatives1
                        << " adaptive_macroparticle_representatives2="
                        << receipt.adaptive_macroparticle_representatives2
                        << " adaptive_macroparticle_many_root1="
                        << receipt.adaptive_macroparticle_many_root1
                        << " adaptive_macroparticle_many_root2="
                        << receipt.adaptive_macroparticle_many_root2
                        << " adaptive_macroparticle_weight_ess1="
                        << receipt.adaptive_macroparticle_weight_ess1
                        << " adaptive_macroparticle_weight_ess2="
                        << receipt.adaptive_macroparticle_weight_ess2
                        << " adaptive_macroparticle_clone_support1="
                        << receipt.adaptive_macroparticle_clone_support1
                        << " adaptive_macroparticle_clone_support2="
                        << receipt.adaptive_macroparticle_clone_support2
                        << " adaptive_macroparticle_projection_residual="
                        << receipt.adaptive_macroparticle_projection_residual
                        << " adaptive_macroparticle_hash1="
                        << receipt.adaptive_macroparticle_hash1
                        << " adaptive_macroparticle_hash2="
                        << receipt.adaptive_macroparticle_hash2
                        << " adaptive_macroparticle_revision1="
                        << receipt.adaptive_macroparticle_revision1
                        << " adaptive_macroparticle_revision2="
                        << receipt.adaptive_macroparticle_revision2
                        << " adaptive_macroparticle_committed="
                        << (receipt.adaptive_macroparticle_committed ? 1 : 0)
                        << " source_budget_runtime_verified="
                        << (receipt.source_budget_runtime_verified ? 1 : 0)
                        << " source_build_ms=" << receipt.source_build_ms
                        << " source_restrict_ms="
                        << receipt.source_restrict_ms
                        << " source_prolong_ms="
                        << receipt.source_prolong_ms
                        << " source_max_charge_closure="
                        << receipt.source_maximum_charge_closure
                        << " source_max_projection_moment_shift="
                        << receipt.source_maximum_projection_moment_shift
                        << " source_lumi_oracle_error="
                        << receipt.source_lumi_oracle_error
                        << " source_max_lumi_oracle_error="
                        << receipt.source_maximum_lumi_oracle_error
                        << " error_target_met="
                        << (receipt.error_target_met ? 1 : 0)
                        << " noise_limited="
                        << (receipt.noise_limited ? 1 : 0)
                        << " encounter_tail_budget_eligible="
                        << (receipt.encounter_tail_budget_eligible ? 1 : 0)
                        << " encounter_tail_budget_used="
                        << (receipt.encounter_tail_budget_used ? 1 : 0)
                        << " encounter_population_weight="
                        << receipt.encounter_population_weight
                        << " encounter_tail_selected_fraction="
                        << receipt.encounter_tail_selected_fraction
                        << " nested_patch_requested="
                        << (receipt.nested_patch_requested ? 1 : 0)
                        << " nested_patch_used="
                        << (receipt.nested_patch_used ? 1 : 0)
                        << " nested_patch_promoted_uniform="
                        << (receipt.nested_patch_promoted_to_uniform ? 1 : 0)
                        << " nested_patch_rejection_mask="
                        << receipt.nested_patch_rejection_mask
                        << " nested_coarse_cells="
                        << receipt.nested_patch_coarse_cells
                        << " nested_fine_cells="
                        << receipt.nested_patch_fine_cells
                        << " nested_coarse_delta_x="
                        << receipt.nested_patch_coarse_delta_x
                        << " nested_coarse_delta_y="
                        << receipt.nested_patch_coarse_delta_y
                        << " nested_fine_delta_x="
                        << receipt.nested_patch_fine_delta_x
                        << " nested_fine_delta_y="
                        << receipt.nested_patch_fine_delta_y
                        << " nested_neff_p10="
                        << receipt.nested_patch_effective_particles_per_cell
                        << " nested_neff_critical_min="
                        << receipt.
                            nested_patch_minimum_effective_particles_per_cell
                        << " nested_residual="
                        << receipt.nested_patch_residual
                        << " nested_field_error_rms="
                        << receipt.nested_patch_field_error_rms
                        << " nested_field_error_linf="
                        << receipt.nested_patch_field_error_linf
                        << " nested_lumi_error="
                        << receipt.nested_patch_lumi_error
                        << " nested_tail_fraction="
                        << receipt.nested_patch_tail_fraction
                        << " nested_straddling_sources="
                        << receipt.nested_patch_straddling_sources
                        << " nested_iterations="
                        << receipt.nested_patch_iterations
                        << " terminal_patch_attempted="
                        << (receipt.terminal_patch_certification_attempted
                                ? 1 : 0)
                        << " terminal_patch_accepted="
                        << (receipt.terminal_patch_certification_accepted
                                ? 1 : 0)
                        << " terminal_patch_used="
                        << (receipt.terminal_patch_used ? 1 : 0)
                        << " terminal_patch_uniform_fallback="
                        << (receipt.terminal_patch_uniform_fallback ? 1 : 0)
                        << " terminal_patch_external_replay_recommended="
                        << (receipt.terminal_patch_external_replay_recommended
                                ? 1 : 0)
                        << " terminal_patch_rejection_mask="
                        << receipt.terminal_patch_rejection_mask
                        << " terminal_patch_factor_r="
                        << receipt.terminal_patch_factor_r
                        << " terminal_patch_factor_r2="
                        << receipt.terminal_patch_factor_r2
                        << " terminal_patch_cells_r="
                        << receipt.terminal_patch_cells_r
                        << " terminal_patch_cells_r2="
                        << receipt.terminal_patch_cells_r2
                        << " terminal_patch_geometry_hash_r="
                        << receipt.terminal_patch_geometry_hash_r
                        << " terminal_patch_geometry_hash_r2="
                        << receipt.terminal_patch_geometry_hash_r2
                        << " terminal_patch_source_hash_r="
                        << receipt.terminal_patch_source_hash_r
                        << " terminal_patch_source_hash_r2="
                        << receipt.terminal_patch_source_hash_r2
                        << " terminal_patch_local_target_hash1="
                        << receipt.terminal_patch_local_target_hash1
                        << " terminal_patch_local_target_hash2="
                        << receipt.terminal_patch_local_target_hash2
                        << " terminal_patch_outside_target_hash1="
                        << receipt.terminal_patch_outside_target_hash1
                        << " terminal_patch_outside_target_hash2="
                        << receipt.terminal_patch_outside_target_hash2
                        << " terminal_patch_outside_unresolved_exposure="
                        << receipt.terminal_patch_outside_unresolved_exposure
                        << " terminal_patch_outside_field_contribution="
                        << receipt.terminal_patch_outside_field_error_contribution
                        << " terminal_patch_outside_field_maximum="
                        << receipt.terminal_patch_outside_field_maximum
                        << " terminal_patch_outside_luminosity_fraction="
                        << receipt.terminal_patch_outside_luminosity_fraction
                        << " terminal_patch_outside_luminosity_earlier="
                        << receipt.terminal_patch_outside_luminosity_earlier
                        << " terminal_patch_outside_luminosity_previous="
                        << receipt.terminal_patch_outside_luminosity_previous
                        << " terminal_patch_outside_luminosity_parent="
                        << receipt.terminal_patch_outside_luminosity_parent
                        << " terminal_patch_observed_field_order="
                        << receipt.terminal_patch_observed_field_order
                        << " terminal_patch_observed_outside_field_order="
                        << receipt.terminal_patch_observed_outside_field_order
                        << " terminal_patch_composite_field_rms="
                        << receipt.terminal_patch_composite_field_rms
                        << " terminal_patch_composite_field_quantile="
                        << receipt.terminal_patch_composite_field_quantile
                        << " terminal_patch_composite_field_maximum="
                        << receipt.terminal_patch_composite_field_maximum
                        << " terminal_patch_composite_field_bound="
                        << receipt.terminal_patch_composite_field_bound
                        << " terminal_patch_observed_luminosity_order="
                        << receipt.terminal_patch_observed_luminosity_order
                        << " terminal_patch_observed_outside_luminosity_order="
                        << receipt.
                            terminal_patch_observed_outside_luminosity_order
                        << " terminal_patch_outside_luminosity_error="
                        << receipt.terminal_patch_outside_luminosity_error
                        << " terminal_patch_composite_luminosity_bound="
                        << receipt.terminal_patch_composite_luminosity_bound
                        << " terminal_patch_boundary_flux_rms="
                        << receipt.terminal_patch_boundary_flux_rms
                        << " terminal_patch_boundary_flux_quantile="
                        << receipt.terminal_patch_boundary_flux_quantile
                        << " terminal_patch_boundary_flux_maximum="
                        << receipt.terminal_patch_boundary_flux_maximum
                        << std::endl;
                  }

              if (switches.get_wavefront_parallel_physics())
                for (std::size_t begin = 0;
                     begin < wave.encounters.size();
                     begin += wavefront_field_workers_)
                  {
                    const std::size_t count = std::min<std::size_t>(
                        wavefront_field_workers_,
                        wave.encounters.size() - begin);
                    const std::chrono::steady_clock::time_point physics_start =
                        std::chrono::steady_clock::now();
                    physics_context.position_offset = begin;
                    physics_context.prefetch_offset = begin;
                    physics_context.legacy_time_base = time_counter_;
                    backend.run_static_ranges(
                        count, &physics_context,
                        &FEEBL::wavefront_process_physics_range);
                    publish_wavefront_physics_batch(begin, count);
                    parallel_physics_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - physics_start)
                            .count();
                    checked_add_u64(
                        encounters,
                        checked_size_to_u64(
                            count, "parallel physics encounter count"),
                        "collision encounter");
                  }
            }
          else
            for (std::size_t begin = 0; begin < wave.encounters.size();
                 begin += wavefront_field_workers_)
              {
                field_context.position_offset = begin;
                const std::size_t count = std::min<std::size_t>(
                    wavefront_field_workers_, wave.encounters.size() - begin);
                const std::chrono::steady_clock::time_point field_start =
                    std::chrono::steady_clock::now();
                backend.run_static_ranges(
                    count, &field_context,
                    &FEEBL::wavefront_prepare_field_range);
                field_prefetch_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - field_start).count();
                checked_add_u64(field_dispatch_batches, 1,
                                "field dispatch batch");
                if (switches.get_wavefront_parallel_physics())
                  {
                    const std::chrono::steady_clock::time_point physics_start =
                        std::chrono::steady_clock::now();
                    physics_context.position_offset = begin;
                    physics_context.prefetch_offset = begin;
                    physics_context.legacy_time_base = time_counter_;
                    backend.run_static_ranges(
                        count, &physics_context,
                        &FEEBL::wavefront_process_physics_range);
                    publish_wavefront_physics_batch(begin, count);
                    parallel_physics_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - physics_start)
                            .count();
                    checked_add_u64(
                        encounters,
                        checked_size_to_u64(
                            count, "parallel physics encounter count"),
                        "collision encounter");
                  }
              }
          // Parallel CPU time has its own wall-clock provenance and must not
          // leak into the next legacy timer bucket.
          time_.checkpoint();

          if (!switches.get_wavefront_parallel_physics())
            {
              const std::chrono::steady_clock::time_point commit_start =
                  std::chrono::steady_clock::now();
              for (std::size_t position = 0;
                   position < wave.encounters.size(); ++position)
                {
                  const feebl::modern::EncounterNode& encounter =
                      scheduler.dag().node(wave.encounters[position]);
                  make_step(
                      static_cast<int>(encounter.beam1_slice),
                      static_cast<int>(encounter.beam2_slice), sor_parameter,
                      &wavefront_field_prefetch_[position]);
                  checked_add_u64(encounters, 1, "collision encounter");
                }
              ordered_commit_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - commit_start).count();
            }
#ifdef FEEBL_HAVE_GPU5
          if (backend.gpu_mode5_enabled())
            for (std::size_t position = 0;
                 position < wave.encounters.size(); ++position)
              {
                const feebl::modern::EncounterNode& encounter =
                    scheduler.dag().node(wave.encounters[position]);
                const std::size_t slice1 = static_cast<std::size_t>(
                    encounter.beam1_slice);
                const std::size_t slice2 = static_cast<std::size_t>(
                    encounter.beam2_slice);
                if (slice1 >= gpu5_beam1_revisions_.size() ||
                    slice2 >= gpu5_beam2_revisions_.size() ||
                    gpu5_beam1_revisions_[slice1] ==
                        std::numeric_limits<std::uint64_t>::max() ||
                    gpu5_beam2_revisions_[slice2] ==
                        std::numeric_limits<std::uint64_t>::max())
                  throw std::overflow_error(
                      "gpu_mode=5 slice mutation revision overflow");
                ++gpu5_beam1_revisions_[slice1];
                ++gpu5_beam2_revisions_[slice2];
              }
#endif
          wavefront_end_wave(&boundary_context, wave);

          // Only now have every encounter kick, stochastic update, ordered
          // result merge, slice revision, and end-wave boundary check for
          // this wave completed.  A decision made here can affect only a
          // later transaction or a fresh-process replay; it cannot change an
          // encounter whose source/field result is already in flight.
          for (feebl::adaptive_particles::ControllerObservation&
                   observation : pending_adaptive_particle_observations)
            {
              observation.canonical_publication_complete = true;
              if (!adaptive_particle_controller)
                throw std::logic_error(
                    "adaptive particle observation reached the wave barrier "
                    "without a controller");
              const feebl::adaptive_particles::ControllerDecision decision =
                  adaptive_particle_controller->observe(observation);
              ++adaptive_particle_controller_observations;
              if (!observation.error.source_sampling_known)
                ++adaptive_particle_controller_sampling_unknown;
              const std::size_t action_index =
                  static_cast<std::size_t>(decision.action);
              const std::size_t recommendation_index =
                  static_cast<std::size_t>(decision.recommended_action);
              if (action_index >= 7U || recommendation_index >= 7U)
                throw std::logic_error(
                    "adaptive particle controller returned an invalid "
                    "action");
              ++adaptive_particle_controller_actions[action_index];
              ++adaptive_particle_controller_recommendations[
                  recommendation_index];
              mix_adaptive_particle_trace(decision.decision_hash);
              if (decision.replay_required)
                {
                  adaptive_particle_controller_replay_required = true;
                  // Preserve the strongest action over the complete collision.
                  // A replay-only recommendation in an early wave must not
                  // mask a later observation that also requires mesh work.
                  adaptive_particle_controller_replay_recommendation =
                      feebl::adaptive_particles::
                          combine_replay_recommendation(
                              adaptive_particle_controller_replay_recommendation,
                              decision.recommended_action);
                }
              if (decision.action ==
                  feebl::adaptive_particles::ControllerAction::
                      UnmetFailClosed)
                {
                  adaptive_particle_controller_unmet = true;
                  if (adaptive_particle_controller_first_unmet_reason.empty())
                    adaptive_particle_controller_first_unmet_reason =
                        decision.reason;
                }
              if (switches.get_adaptive_macroparticle_configuration().
                      profile == "detail")
                std::cerr
                    << "FEEBL_ADAPTIVE_MACROPARTICLE_DECISION_V1"
                    << " epoch=" << observation.epoch
                    << " publication_barrier=complete"
                    << " action="
                    << feebl::adaptive_particles::controller_action_name(
                           decision.action)
                    << " recommended="
                    << feebl::adaptive_particles::controller_action_name(
                           decision.recommended_action)
                    << " mesh_ratio=" << decision.mesh_ratio
                    << " source_sampling_ratio="
                    << decision.source_sampling_ratio
                    << " source_sampling_known=0"
                    << " distinct_roots="
                    << observation.support.distinct_roots
                    << " clone_support_proxy="
                    << observation.support.clone_support_proxy
                    << " protected_tail_support="
                    << observation.support.protected_tail_support
                    << " independent_complete_runs=1"
                    << " total_bound_met="
                    << (decision.total_error_bound_met ? 1 : 0)
                    << " decision_hash=" << decision.decision_hash
                    << " reason=" << decision.reason
                    << std::endl;
            }

          maximum_wave_width = std::max(
              maximum_wave_width, wave.encounters.size());
          maximum_nonempty_lanes = std::max(
              maximum_nonempty_lanes,
              std::min<std::size_t>(
                  wave.encounters.size(),
                  wavefront_field_workers_));
          if (wave.encounters.size() > 1U)
            checked_add_u64(
                parallel_field_encounters,
                checked_size_to_u64(wave.encounters.size(),
                                    "parallel field encounter count"),
                "parallel field encounter");
        }
      wavefront_end_diagonal(&boundary_context, diagonal);
    }

  if (encounters != scheduler.dag().encounter_count())
    throw std::logic_error(
        "parallel wavefront scheduler did not exhaust its EncounterDAG");
  if (adaptive_schedule)
    {
      if (adaptive_schedule_cursor !=
          static_cast<std::size_t>(scheduler.dag().encounter_count()))
        throw std::logic_error(
            "adaptive schedule did not record every canonical encounter");
      if (replay_schedule)
        {
          if (adaptive_schedule_cursor != replay_schedule->records().size())
            throw std::runtime_error(
                "adaptive schedule replay rejected: saved schedule has "
                "surplus or missing canonical encounters");
          replay_schedule->validate_exact(*adaptive_schedule);
        }
      adaptive_schedule_hash = adaptive_schedule->stable_hash();
      if (!switches.get_adaptive_grid_schedule_save().empty())
        adaptive_schedule->save_atomic(
            switches.get_adaptive_grid_schedule_save());
      std::cerr
          << "FEEBL adaptive schedule: format="
          << feebl::adaptive_grid::kAdaptiveGridScheduleFormatVersion
          << " records=" << adaptive_schedule_cursor
          << " hash=" << adaptive_schedule_hash
          << " action="
          << (replay_schedule
                  ? "replay-validated"
                  : (switches.get_adaptive_grid_schedule_save().empty()
                         ? "computed"
                         : "saved"))
          << " path=\""
          << (replay_schedule
                  ? switches.get_adaptive_grid_schedule_load()
                  : switches.get_adaptive_grid_schedule_save())
          << "\"" << std::endl;
    }
  double lumi_heap_publish_ms = 0.0;
  if (switches.get_wavefront_parallel_physics())
    {
      const std::chrono::steady_clock::time_point publish_start =
          std::chrono::steady_clock::now();
      publish_wavefront_lumi_heaps();
      lumi_heap_publish_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - publish_start).count();
    }
  const std::uint64_t batches = checked_size_to_u64(
      scheduler.make_batches().size(), "wavefront batch count");
  std::uint64_t dense_clear_visits = 0;
  std::uint64_t actual_clear_visits = 0;
  std::uint64_t dense_dot_visits = 0;
  std::uint64_t actual_dot_visits = 0;
  std::uint64_t direct_encounters = 0;
  std::uint64_t fft_encounters = 0;
  std::uint64_t direct_active_sources = 0;
  std::uint64_t direct_target_nodes = 0;
  std::uint64_t estimated_interactions = 0;
  std::uint64_t direct_interactions = 0;
  std::uint64_t maximum_estimated_interactions = 0;
  double direct_ms = 0.0;
  double fft_ms = 0.0;
  std::uint64_t adaptive_encounters = 0;
  std::uint64_t adaptive_full_windows = 0;
  std::uint64_t adaptive_required_cells = 0;
  std::uint64_t adaptive_local_cells = 0;
  std::uint64_t adaptive_padded_cells = 0;
  std::uint64_t adaptive_global_cells_avoided = 0;
  std::uint64_t adaptive_plan_builds = 0;
  std::uint64_t adaptive_context_builds = 0;
  std::uint64_t adaptive_context_hits = 0;
  std::uint64_t adaptive_uniform_context_misses = 0;
  std::uint64_t adaptive_patch_context_misses = 0;
  std::uint64_t adaptive_context_evictions = 0;
  std::size_t adaptive_context_bytes = 0U;
  double adaptive_bounds_worker_ms = 0.0;
  double adaptive_context_build_worker_ms = 0.0;
  double adaptive_deposit_worker_ms = 0.0;
  double adaptive_solve_worker_ms = 0.0;
  double adaptive_gather_worker_ms = 0.0;
  std::uint64_t mode2_encounters = 0;
  std::uint64_t mode2_field_solves = 0;
  std::uint64_t mode2_refined_encounters = 0;
  std::uint64_t mode2_unmet_error_encounters = 0;
  std::uint64_t mode2_noise_aware_stop_encounters = 0;
  std::uint64_t vector_catastrophe_failure_encounters = 0;
  double maximum_vector_difference = 0.0;
  double maximum_beam_above_target_exposure = 0.0;
  std::uint64_t mode2_tail_budget_eligible_encounters = 0;
  std::uint64_t mode2_tail_budget_used_encounters = 0;
  std::size_t encounter_tail_budget_selected_pairs = 0U;
  double encounter_tail_budget_selected_fraction = 0.0;
  // Legacy extrapolated Richardson sum is diagnostic only.  Tail
  // qualification uses the bounded symmetric sensitivity and the direct
  // uniform-luminosity exposure fraction below.
  double encounter_tail_weighted_field_error = 0.0;
  long double encounter_tail_weighted_field_sensitivity = 0.0L;
  double encounter_tail_weighted_source_lumi_error = 0.0;
  long double encounter_all_uniform_lumi_measure = 0.0L;
  long double encounter_tail_uniform_lumi_measure = 0.0L;
  bool encounter_tail_budget_receipt_seen = false;
  std::uint64_t mode3_patch_requests = 0;
  std::uint64_t mode3_patch_encounters = 0;
  std::uint64_t mode3_uniform_promotions = 0;
  std::uint64_t mode3_terminal_uniform_encounters = 0;
  std::uint64_t mode3_terminal_patch_attempts = 0;
  std::uint64_t mode3_terminal_patch_accepts = 0;
  std::uint64_t mode3_terminal_patch_used = 0;
  std::uint64_t mode3_terminal_patch_rejects = 0;
  std::uint64_t mode3_terminal_patch_uniform_fallbacks = 0;
  std::uint64_t mode3_terminal_patch_resource_rejects = 0;
  std::uint64_t mode3_terminal_patch_nonresource_rejects = 0;
  std::uint32_t mode3_terminal_patch_rejection_mask_or = 0U;
  std::uint64_t mode3_no_patch_encounters = 0;
  std::uint64_t mode3_empty_encounters = 0;
  std::uint64_t mode3_patch_coarse_cells = 0;
  std::uint64_t mode3_patch_fine_cells = 0;
  std::uint64_t mode3_attempted_patch_coarse_cells = 0;
  std::uint64_t mode3_attempted_patch_fine_cells = 0;
  std::uint64_t mode3_harmonic_iterations = 0;
  std::uint32_t mode3_rejection_mask_or = 0U;
  double mode3_controller_worker_ms = 0.0;
  double mode3_harmonic_worker_ms = 0.0;
  double maximum_estimated_field_error = 0.0;
  double maximum_patch_residual = 0.0;
  double maximum_patch_field_error = 0.0;
  double maximum_examined_patch_lumi_oracle_error = 0.0;
  double maximum_accepted_patch_lumi_oracle_error = 0.0;
  std::uint64_t patch_lumi_oracle_evaluations = 0U;
  std::uint64_t accepted_patch_lumi_oracle_evaluations = 0U;
  double maximum_source_lumi_oracle_error = 0.0;
  double maximum_selected_source_lumi_oracle_error = 0.0;
  double maximum_source_charge_closure = 0.0;
  double maximum_source_projection_moment_shift = 0.0;
  std::uint64_t source_hierarchy_encounters = 0U;
  std::uint64_t source_density_hash_nonzero_encounters = 0U;
  std::uint64_t source_sampling_hash_nonzero_encounters = 0U;
  std::uint64_t source_sampling_hierarchy_hash_nonzero_encounters = 0U;
  std::uint64_t source_beam1_in_visits = 0U;
  std::uint64_t source_beam1_out_visits = 0U;
  std::uint64_t source_beam2_in_visits = 0U;
  std::uint64_t source_beam2_out_visits = 0U;
  double source_build_worker_ms = 0.0;
  double source_restrict_worker_ms = 0.0;
  double source_prolong_worker_ms = 0.0;
  std::vector<std::size_t> mode2_spacing_factors =
      switches.get_adaptive_grid_mode() >= 2
          ? feebl::adaptive_grid::parse_refinement_factors(
                switches.get_adaptive_grid_field_spacing_classes())
          : std::vector<std::size_t>();
  std::vector<std::uint64_t> mode2_spacing_selections(
      mode2_spacing_factors.size(), 0U);
  for (std::size_t lane = 0;
       lane < wavefront_field_workspaces_.size(); ++lane)
    {
      checked_add_u64(
          dense_clear_visits,
          wavefront_field_workspaces_[lane]->dense_clear_visits(),
          "dense-clear visit");
      checked_add_u64(
          actual_clear_visits,
          wavefront_field_workspaces_[lane]->actual_clear_visits(),
          "actual-clear visit");
      checked_add_u64(
          dense_dot_visits,
          wavefront_field_workspaces_[lane]->dense_dot_visits(),
          "dense-dot visit");
      checked_add_u64(
          actual_dot_visits,
          wavefront_field_workspaces_[lane]->actual_dot_visits(),
          "actual-dot visit");
      checked_add_u64(
          direct_encounters,
          wavefront_field_workspaces_[lane]->direct_encounters(),
          "sparse-direct encounter");
      checked_add_u64(
          fft_encounters,
          wavefront_field_workspaces_[lane]->fft_encounters(),
          "FFT encounter");
      checked_add_u64(
          direct_active_sources,
          wavefront_field_workspaces_[lane]->direct_active_sources(),
          "sparse-direct active source");
      checked_add_u64(
          direct_target_nodes,
          wavefront_field_workspaces_[lane]->direct_target_nodes(),
          "sparse-direct target node");
      checked_add_u64(
          estimated_interactions,
          wavefront_field_workspaces_[lane]->estimated_interactions(),
          "sparse-direct estimated interaction");
      checked_add_u64(
          direct_interactions,
          wavefront_field_workspaces_[lane]->direct_interactions(),
          "sparse-direct executed interaction");
      maximum_estimated_interactions = std::max(
          maximum_estimated_interactions,
          wavefront_field_workspaces_[lane]->maximum_estimated_interactions());
      direct_ms += wavefront_field_workspaces_[lane]->direct_ms();
      fft_ms += wavefront_field_workspaces_[lane]->fft_ms();
      checked_add_u64(
          adaptive_encounters,
          wavefront_field_workspaces_[lane]->adaptive_encounters(),
          "adaptive encounter");
      checked_add_u64(
          adaptive_full_windows,
          wavefront_field_workspaces_[lane]
              ->adaptive_full_window_encounters(),
          "adaptive full-window encounter");
      checked_add_u64(
          adaptive_required_cells,
          wavefront_field_workspaces_[lane]->adaptive_required_cells(),
          "adaptive required cells");
      checked_add_u64(
          adaptive_local_cells,
          wavefront_field_workspaces_[lane]->adaptive_local_cells(),
          "adaptive local cells");
      checked_add_u64(
          adaptive_padded_cells,
          wavefront_field_workspaces_[lane]->adaptive_padded_cells(),
          "adaptive padded cells");
      checked_add_u64(
          adaptive_global_cells_avoided,
          wavefront_field_workspaces_[lane]
              ->adaptive_global_cells_avoided(),
          "adaptive global cells avoided");
      checked_add_u64(
          adaptive_plan_builds,
          wavefront_field_workspaces_[lane]->adaptive_plan_builds(),
          "adaptive plan builds");
      checked_add_u64(
          adaptive_context_builds,
          wavefront_field_workspaces_[lane]->adaptive_context_builds(),
          "adaptive context builds");
      checked_add_u64(
          adaptive_context_hits,
          wavefront_field_workspaces_[lane]->adaptive_context_hits(),
          "adaptive context hits");
      checked_add_u64(
          adaptive_uniform_context_misses,
          wavefront_field_workspaces_[lane]
              ->adaptive_uniform_context_misses(),
          "adaptive uniform context miss");
      checked_add_u64(
          adaptive_patch_context_misses,
          wavefront_field_workspaces_[lane]
              ->adaptive_patch_context_misses(),
          "adaptive patch context miss");
      checked_add_u64(
          adaptive_context_evictions,
          wavefront_field_workspaces_[lane]->adaptive_context_evictions(),
          "adaptive context eviction");
      if (wavefront_field_workspaces_[lane]->adaptive_context_bytes() >
          std::numeric_limits<std::size_t>::max() - adaptive_context_bytes)
        throw std::overflow_error("adaptive context byte total overflow");
      adaptive_context_bytes +=
          wavefront_field_workspaces_[lane]->adaptive_context_bytes();
      adaptive_bounds_worker_ms +=
          wavefront_field_workspaces_[lane]->adaptive_bounds_ms();
      adaptive_context_build_worker_ms +=
          wavefront_field_workspaces_[lane]->adaptive_context_build_ms();
      adaptive_deposit_worker_ms +=
          wavefront_field_workspaces_[lane]->adaptive_deposit_ms();
      adaptive_solve_worker_ms +=
          wavefront_field_workspaces_[lane]->adaptive_solve_ms();
      adaptive_gather_worker_ms +=
          wavefront_field_workspaces_[lane]->adaptive_gather_ms();
      checked_add_u64(
          mode2_encounters,
          wavefront_field_workspaces_[lane]->mode2_encounters(),
          "Mode-2 encounter");
      checked_add_u64(
          mode2_field_solves,
          wavefront_field_workspaces_[lane]->mode2_field_solves(),
          "Mode-2 field solve");
      checked_add_u64(
          mode2_refined_encounters,
          wavefront_field_workspaces_[lane]->mode2_refined_encounters(),
          "Mode-2 refined encounter");
      checked_add_u64(
          mode2_unmet_error_encounters,
          wavefront_field_workspaces_[lane]->mode2_unmet_error_encounters(),
          "Mode-2 unmet-error encounter");
      checked_add_u64(
          mode2_noise_aware_stop_encounters,
          wavefront_field_workspaces_[lane]
              ->mode2_noise_aware_stop_encounters(),
          "Mode-2 noise-aware stop encounter");
      checked_add_u64(
          vector_catastrophe_failure_encounters,
          wavefront_field_workspaces_[lane]
              ->vector_catastrophe_failure_encounters(),
          "adaptive vector-catastrophe failure encounter");
      maximum_vector_difference = std::max(
          maximum_vector_difference,
          wavefront_field_workspaces_[lane]
              ->maximum_vector_difference());
      maximum_beam_above_target_exposure = std::max(
          maximum_beam_above_target_exposure,
          wavefront_field_workspaces_[lane]
              ->maximum_beam_above_target_exposure());
      checked_add_u64(
          mode2_tail_budget_eligible_encounters,
          wavefront_field_workspaces_[lane]
              ->mode2_tail_budget_eligible_encounters(),
          "Mode-2 encounter-tail eligible encounter");
      checked_add_u64(
          mode2_tail_budget_used_encounters,
          wavefront_field_workspaces_[lane]
              ->mode2_tail_budget_used_encounters(),
          "Mode-2 encounter-tail used encounter");
      const std::size_t lane_tail_pairs =
          wavefront_field_workspaces_[lane]
              ->encounter_tail_budget_selected_pairs();
      const double lane_tail_fraction =
          wavefront_field_workspaces_[lane]
              ->encounter_tail_budget_selected_fraction();
      if (!encounter_tail_budget_receipt_seen) {
        encounter_tail_budget_selected_pairs = lane_tail_pairs;
        encounter_tail_budget_selected_fraction = lane_tail_fraction;
        encounter_tail_budget_receipt_seen = true;
      } else if (lane_tail_pairs != encounter_tail_budget_selected_pairs ||
                 lane_tail_fraction !=
                     encounter_tail_budget_selected_fraction) {
        throw std::logic_error(
            "Mode-2 encounter-tail budget disagrees across worker lanes");
      }
      encounter_tail_weighted_field_error +=
          wavefront_field_workspaces_[lane]
              ->encounter_tail_weighted_field_error();
      encounter_tail_weighted_field_sensitivity +=
          wavefront_field_workspaces_[lane]
              ->encounter_tail_weighted_field_sensitivity();
      encounter_tail_weighted_source_lumi_error +=
          wavefront_field_workspaces_[lane]
              ->encounter_tail_weighted_source_lumi_error();
      encounter_all_uniform_lumi_measure +=
          wavefront_field_workspaces_[lane]
              ->encounter_all_uniform_lumi_measure();
      encounter_tail_uniform_lumi_measure +=
          wavefront_field_workspaces_[lane]
              ->encounter_tail_uniform_lumi_measure();
      if (!std::isfinite(encounter_tail_weighted_field_error) ||
          !std::isfinite(encounter_tail_weighted_field_sensitivity) ||
          !std::isfinite(encounter_tail_weighted_source_lumi_error) ||
          !std::isfinite(encounter_all_uniform_lumi_measure) ||
          !std::isfinite(encounter_tail_uniform_lumi_measure))
        throw std::overflow_error(
            "Mode-2 encounter-tail aggregate error overflowed");
      checked_add_u64(
          mode3_patch_requests,
          wavefront_field_workspaces_[lane]->mode3_patch_requests(),
          "Mode-3 patch request");
      checked_add_u64(
          mode3_patch_encounters,
          wavefront_field_workspaces_[lane]->mode3_patch_encounters(),
          "Mode-3 patch encounter");
      checked_add_u64(
          mode3_uniform_promotions,
          wavefront_field_workspaces_[lane]->mode3_uniform_promotions(),
          "Mode-3 uniform promotion");
      checked_add_u64(
          mode3_terminal_uniform_encounters,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_uniform_encounters(),
          "Mode-3 terminal uniform encounter");
      checked_add_u64(
          mode3_terminal_patch_attempts,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_attempts(),
          "Mode-3 terminal patch attempt");
      checked_add_u64(
          mode3_terminal_patch_accepts,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_accepts(),
          "Mode-3 terminal patch acceptance");
      checked_add_u64(
          mode3_terminal_patch_used,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_used(),
          "Mode-3 terminal patch use");
      checked_add_u64(
          mode3_terminal_patch_rejects,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_rejects(),
          "Mode-3 terminal patch rejection");
      checked_add_u64(
          mode3_terminal_patch_uniform_fallbacks,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_uniform_fallbacks(),
          "Mode-3 terminal patch uniform fallback");
      checked_add_u64(
          mode3_terminal_patch_resource_rejects,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_resource_rejects(),
          "Mode-3 terminal patch resource rejection");
      checked_add_u64(
          mode3_terminal_patch_nonresource_rejects,
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_nonresource_rejects(),
          "Mode-3 terminal patch non-resource rejection");
      mode3_terminal_patch_rejection_mask_or |=
          wavefront_field_workspaces_[lane]
              ->mode3_terminal_patch_rejection_mask_or();
      checked_add_u64(
          mode3_no_patch_encounters,
          wavefront_field_workspaces_[lane]->mode3_no_patch_encounters(),
          "Mode-3 no-patch encounter");
      checked_add_u64(
          mode3_empty_encounters,
          wavefront_field_workspaces_[lane]->mode3_empty_encounters(),
          "Mode-3 empty encounter");
      checked_add_u64(
          mode3_patch_coarse_cells,
          wavefront_field_workspaces_[lane]->mode3_patch_coarse_cells(),
          "Mode-3 coarse patch cell");
      checked_add_u64(
          mode3_patch_fine_cells,
          wavefront_field_workspaces_[lane]->mode3_patch_fine_cells(),
          "Mode-3 fine patch cell");
      checked_add_u64(
          mode3_attempted_patch_coarse_cells,
          wavefront_field_workspaces_[lane]
              ->mode3_attempted_patch_coarse_cells(),
          "Mode-3 attempted coarse patch cell");
      checked_add_u64(
          mode3_attempted_patch_fine_cells,
          wavefront_field_workspaces_[lane]
              ->mode3_attempted_patch_fine_cells(),
          "Mode-3 attempted fine patch cell");
      checked_add_u64(
          mode3_harmonic_iterations,
          wavefront_field_workspaces_[lane]->mode3_harmonic_iterations(),
          "Mode-3 harmonic iteration");
      mode3_rejection_mask_or |=
          wavefront_field_workspaces_[lane]->mode3_rejection_mask_or();
      mode3_controller_worker_ms +=
          wavefront_field_workspaces_[lane]->mode3_controller_ms();
      mode3_harmonic_worker_ms +=
          wavefront_field_workspaces_[lane]->mode3_harmonic_ms();
      maximum_estimated_field_error = std::max(
          maximum_estimated_field_error,
          wavefront_field_workspaces_[lane]
              ->maximum_estimated_field_error());
      maximum_patch_residual = std::max(
          maximum_patch_residual,
          wavefront_field_workspaces_[lane]->maximum_patch_residual());
      maximum_patch_field_error = std::max(
          maximum_patch_field_error,
          wavefront_field_workspaces_[lane]->maximum_patch_field_error());
      maximum_examined_patch_lumi_oracle_error = std::max(
          maximum_examined_patch_lumi_oracle_error,
          wavefront_field_workspaces_[lane]
              ->maximum_examined_patch_lumi_oracle_error());
      maximum_accepted_patch_lumi_oracle_error = std::max(
          maximum_accepted_patch_lumi_oracle_error,
          wavefront_field_workspaces_[lane]
              ->maximum_accepted_patch_lumi_oracle_error());
      checked_add_u64(
          patch_lumi_oracle_evaluations,
          wavefront_field_workspaces_[lane]
              ->patch_lumi_oracle_evaluations(),
          "Mode-3 patch luminosity-oracle evaluation");
      checked_add_u64(
          accepted_patch_lumi_oracle_evaluations,
          wavefront_field_workspaces_[lane]
              ->accepted_patch_lumi_oracle_evaluations(),
          "Mode-3 accepted patch luminosity-oracle evaluation");
      maximum_source_lumi_oracle_error = std::max(
          maximum_source_lumi_oracle_error,
          wavefront_field_workspaces_[lane]
              ->maximum_source_lumi_oracle_error());
      maximum_selected_source_lumi_oracle_error = std::max(
          maximum_selected_source_lumi_oracle_error,
          wavefront_field_workspaces_[lane]
              ->maximum_selected_source_lumi_oracle_error());
      maximum_source_charge_closure = std::max(
          maximum_source_charge_closure,
          wavefront_field_workspaces_[lane]
              ->maximum_source_charge_closure());
      maximum_source_projection_moment_shift = std::max(
          maximum_source_projection_moment_shift,
          wavefront_field_workspaces_[lane]
              ->maximum_source_projection_moment_shift());
      checked_add_u64(
          source_hierarchy_encounters,
          wavefront_field_workspaces_[lane]->source_hierarchy_encounters(),
          "reference source hierarchy encounter");
      checked_add_u64(
          source_density_hash_nonzero_encounters,
          wavefront_field_workspaces_[lane]
              ->source_density_hash_nonzero_encounters(),
          "reference source density hash encounter");
      checked_add_u64(
          source_sampling_hash_nonzero_encounters,
          wavefront_field_workspaces_[lane]
              ->source_sampling_hash_nonzero_encounters(),
          "reference source sampling hash encounter");
      checked_add_u64(
          source_sampling_hierarchy_hash_nonzero_encounters,
          wavefront_field_workspaces_[lane]
              ->source_sampling_hierarchy_hash_nonzero_encounters(),
          "reference source sampling hierarchy hash encounter");
      checked_add_u64(
          source_beam1_in_visits,
          wavefront_field_workspaces_[lane]->source_beam1_in_visits(),
          "source beam-1 admitted visit");
      checked_add_u64(
          source_beam1_out_visits,
          wavefront_field_workspaces_[lane]->source_beam1_out_visits(),
          "source beam-1 rejected visit");
      checked_add_u64(
          source_beam2_in_visits,
          wavefront_field_workspaces_[lane]->source_beam2_in_visits(),
          "source beam-2 admitted visit");
      checked_add_u64(
          source_beam2_out_visits,
          wavefront_field_workspaces_[lane]->source_beam2_out_visits(),
          "source beam-2 rejected visit");
      source_build_worker_ms +=
          wavefront_field_workspaces_[lane]->source_build_ms();
      source_restrict_worker_ms +=
          wavefront_field_workspaces_[lane]->source_restrict_ms();
      source_prolong_worker_ms +=
          wavefront_field_workspaces_[lane]->source_prolong_ms();
      const std::vector<std::uint64_t>& lane_selections =
          wavefront_field_workspaces_[lane]
              ->mode2_spacing_selection_counts();
      while (mode2_spacing_factors.size() < lane_selections.size()) {
        if (mode2_spacing_factors.empty() ||
            mode2_spacing_factors.back() >
                std::numeric_limits<std::size_t>::max() / 2U)
          throw std::overflow_error(
              "Mode-2 dynamic spacing telemetry factor overflow");
        mode2_spacing_factors.push_back(
            2U * mode2_spacing_factors.back());
      }
      if (mode2_spacing_selections.size() < lane_selections.size())
        mode2_spacing_selections.resize(lane_selections.size(), 0U);
      for (std::size_t index = 0U;
           index < lane_selections.size(); ++index)
        checked_add_u64(mode2_spacing_selections[index],
                        lane_selections[index],
                        "Mode-2 spacing selection");
    }
  std::uint64_t adaptive_context_cold_misses =
      adaptive_uniform_context_misses;
  checked_add_u64(adaptive_context_cold_misses,
                  adaptive_patch_context_misses,
                  "adaptive context cold miss");
  const double encounter_tail_tolerance =
      switches.get_adaptive_grid_tau_tail();
  const double encounter_tail_eligibility_fraction =
      switches.get_adaptive_grid_encounter_tail_fraction() > 0.0
          ? switches.get_adaptive_grid_encounter_tail_fraction()
          : encounter_tail_tolerance;
  const double encounter_tail_roundoff =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, encounter_tail_tolerance);
  if (encounter_tail_weighted_field_sensitivity < 0.0L ||
      encounter_all_uniform_lumi_measure < 0.0L ||
      encounter_tail_uniform_lumi_measure < 0.0L ||
      encounter_tail_uniform_lumi_measure >
          encounter_all_uniform_lumi_measure +
              static_cast<long double>(encounter_tail_roundoff))
    throw std::logic_error(
        "Mode-2/3 cumulative tail sensitivity/luminosity accounting is "
        "inconsistent");
  long double encounter_tail_lumi_exposure_fraction = 0.0L;
  if (encounter_all_uniform_lumi_measure > 0.0L) {
    encounter_tail_lumi_exposure_fraction =
        encounter_tail_uniform_lumi_measure /
        encounter_all_uniform_lumi_measure;
  } else if (encounter_tail_uniform_lumi_measure != 0.0L) {
    throw std::logic_error(
        "Mode-2/3 tail luminosity numerator is nonzero with zero total "
        "luminosity");
  }
  if (!std::isfinite(encounter_tail_lumi_exposure_fraction) ||
      encounter_tail_lumi_exposure_fraction < 0.0L ||
      encounter_tail_lumi_exposure_fraction >
          1.0L + static_cast<long double>(encounter_tail_roundoff))
    throw std::logic_error(
        "Mode-2/3 tail luminosity exposure fraction is invalid");
  const bool encounter_tail_budget_met =
      encounter_tail_budget_selected_fraction <=
          encounter_tail_eligibility_fraction + encounter_tail_roundoff &&
      encounter_tail_weighted_field_sensitivity <=
          static_cast<long double>(encounter_tail_tolerance +
                                   encounter_tail_roundoff) &&
      encounter_tail_weighted_source_lumi_error <=
          encounter_tail_tolerance + encounter_tail_roundoff &&
      encounter_tail_lumi_exposure_fraction <=
          static_cast<long double>(encounter_tail_tolerance +
                                   encounter_tail_roundoff);
  if (switches.get_adaptive_grid_mode() >= 2 &&
      switches.get_adaptive_grid_fail_on_unmet_error() != 0 &&
      !encounter_tail_budget_met) {
    std::ostringstream message;
    message << "Mode-2/3 cumulative encounter-tail budget failed"
            << " (declared=" << encounter_tail_tolerance
            << " eligibility_fraction="
            << encounter_tail_eligibility_fraction
            << " selected_fraction="
            << encounter_tail_budget_selected_fraction
            << " weighted_richardson_error_diagnostic="
            << encounter_tail_weighted_field_error
            << " weighted_field_sensitivity="
            << encounter_tail_weighted_field_sensitivity
            << " weighted_source_lumi_error="
            << encounter_tail_weighted_source_lumi_error
            << " luminosity_exposure_fraction="
            << encounter_tail_lumi_exposure_fraction << ')';
    throw std::runtime_error(message.str());
  }
  const std::string actual_wavefront_schedule =
      backend.gpu_mode5_enabled()
          ? (switches.get_wavefront_parallel_physics()
                 ? "gpu_mode5_adaptive_fields_parallel_host_physics"
                 : "gpu_mode5_adaptive_fields_ordered_host_physics")
          : (switches.get_wavefront_parallel_physics()
                 ? "wavefront_parallel_full_host_physics"
                 : "wavefront_parallel_staged_fields");
  const char* actual_wavefront_scheduler = backend.gpu_mode5_enabled()
      ? "encounter-dag-adaptive-class-planmany-fields-v1"
      : (switches.get_wavefront_parallel_physics()
             ? "encounter-dag-worker-private-host-physics-v1"
             : "encounter-dag-staged-fields-v1");
  switches.set_collision_schedule_provenance(
      actual_wavefront_schedule,
      checked_size_to_u64(scheduler.diagonals().size(),
                          "wavefront diagonal count"),
      scheduler.dag().wave_count(), batches, encounters,
      switches.get_wavefront_parallel_physics() ? encounters : 0,
      checked_size_to_u64(maximum_wave_width, "maximum wave width"));
  switches.set_collision_field_provenance(
      encounters, parallel_field_encounters,
      switches.get_wavefront_parallel_physics() ? encounters : 0,
      checked_size_to_u64(maximum_nonempty_lanes,
                          "maximum nonempty field lanes"),
      static_cast<int>(wavefront_field_workers_),
      wavefront_field_setup_ms_, field_prefetch_ms, ordered_commit_ms,
      dense_clear_visits, actual_clear_visits,
      dense_dot_visits, actual_dot_visits,
      field_dispatch_batches,
      checked_size_to_u64(wavefront_workspace_bytes_each_,
                          "workspace byte estimate"),
      checked_size_to_u64(wavefront_workspace_bytes_reserved_,
                          "workspace reservation estimate"),
      backend.actual_cpu_threads());
  std::cerr
      << "FEEBL legacy wavefront summary: requested=wavefront_parallel"
      << " actual=" << actual_wavefront_schedule
      << " scheduler=" << actual_wavefront_scheduler
      << " diagonals=" << scheduler.diagonals().size()
      << " waves=" << scheduler.dag().wave_count()
      << " batches=" << batches
      << " encounters=" << encounters
      << " maximum_wave_width=" << maximum_wave_width
      << " prefetched_field_encounters=" << encounters
      << " parallel_field_encounters=" << parallel_field_encounters
      << " full_parallel_make_step_encounters="
      << (switches.get_wavefront_parallel_physics() ? encounters : 0)
      << " field_workers=" << wavefront_field_workers_
      << " backend_threads=" << backend.actual_cpu_threads()
      << " field_dispatch_batches=" << field_dispatch_batches
      << " workspace_bytes_each_estimate="
      << wavefront_workspace_bytes_each_
      << " workspace_bytes_reserved_estimate="
      << wavefront_workspace_bytes_reserved_
      << " maximum_nonempty_field_lanes=" << maximum_nonempty_lanes
      << " field_setup_ms=" << wavefront_field_setup_ms_
      << " field_prefetch_ms=" << field_prefetch_ms
      << " parallel_physics_ms=" << parallel_physics_ms
      << " ordered_commit_ms=" << ordered_commit_ms
      << " field_dense_clear_visits=" << dense_clear_visits
      << " field_actual_clear_visits=" << actual_clear_visits
      << " field_dense_dot_visits=" << dense_dot_visits
      << " field_actual_dot_visits=" << actual_dot_visits
      << " ordered_commit="
      << (switches.get_wavefront_parallel_physics() ? 0 : 1)
      << " canonical_barrier_publication="
      << (switches.get_wavefront_parallel_physics() ? 1 : 0)
      << " crossing_lumi_heap_publish_ms=" << lumi_heap_publish_ms
      << " crossing_lumi_heap_publications="
      << (switches.get_wavefront_parallel_physics() ? 1 : 0)
      << std::endl;
  if (switches.get_adaptive_macroparticle_mode() != 0)
    {
      const feebl::adaptive_particles::Configuration& adaptive =
          switches.get_adaptive_macroparticle_configuration();
      const auto count_root_families = [](const PARTICLE_BEAM& beam) {
        const std::vector<std::vector<PRIMARY_PARTICLE_ID> >& slices =
            beam.getParticleIdSlicesStrict();
        std::set<std::pair<unsigned int, std::uint64_t> > families;
        for (const std::vector<PRIMARY_PARTICLE_ID>& ids : slices)
          for (PRIMARY_PARTICLE_ID id : ids)
            {
              PRIMARY_PARTICLE_ID_FIELDS decoded;
              if (!decode_primary_particle_id(id, decoded))
                throw std::logic_error(
                    "adaptive summary encountered an invalid root ID");
              families.insert(std::make_pair(
                  static_cast<unsigned int>(decoded.provenance),
                  decoded.source_ordinal));
            }
        return families.size();
      };
      const std::size_t root_families1 =
          count_root_families(beam1_.particle_beam());
      const std::size_t root_families2 =
          count_root_families(beam2_.particle_beam());
      if (!std::isfinite(adaptive_particle_min_clone_support1))
        adaptive_particle_min_clone_support1 = 0.0;
      if (!std::isfinite(adaptive_particle_min_clone_support2))
        adaptive_particle_min_clone_support2 = 0.0;
      const std::uint64_t merge_reductions1 =
          adaptive_particle_premerge_visits1 >=
                  adaptive_particle_representative_visits1
              ? adaptive_particle_premerge_visits1 -
                    adaptive_particle_representative_visits1
              : 0U;
      const std::uint64_t merge_reductions2 =
          adaptive_particle_premerge_visits2 >=
                  adaptive_particle_representative_visits2
              ? adaptive_particle_premerge_visits2 -
                    adaptive_particle_representative_visits2
              : 0U;
      std::cerr
          << "FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1"
          << " requested_mode=" << static_cast<int>(adaptive.mode)
          << " actual_mode=" << static_cast<int>(adaptive.mode)
          << " backend=cpu"
          << " domain=" << adaptive.domain
          << " source_evolution=" << adaptive.source_evolution
          << " decision_scope=" << adaptive.decision_scope
          << " commit_scope=" << adaptive.commit_scope
          << " strategy=" << adaptive.strategy
          << " split=" << adaptive.split_algorithm
          << " merge=" << adaptive.merge_algorithm
          << " root_source=" << adaptive.root_source
          << " root_seed=" << adaptive.root_seed
          << " scramble_id=" << adaptive.scramble_id
          << " error_target=" << adaptive.error_target
          << " resample_bias_target="
          << adaptive.resample_bias_target
          << " min_root_support=" << adaptive.minimum_root_support
          << " min_clone_support=" << adaptive.minimum_clone_support
          << " statistical_cohorts_declared="
          << adaptive.statistical_cohorts
          << " independent_complete_runs=1"
          << " refine_threshold=" << adaptive.refine_threshold
          << " coarsen_threshold=" << adaptive.coarsen_threshold
          << " update_interval=" << adaptive.update_interval
          << " cooldown=" << adaptive.cooldown
          << " dwell=" << adaptive.dwell
          << " max_change_factor=" << adaptive.maximum_change_factor
          << " max_generation=" << adaptive.maximum_generation
          << " max_count=" << adaptive.maximum_count
          << " max_bytes=" << adaptive.maximum_bytes
          << " max_weight_ratio=" << adaptive.maximum_weight_ratio
          << " determinism=" << adaptive.determinism
          << " profile=" << adaptive.profile
          << " fail_on_unmet_error="
          << (adaptive.fail_on_unmet_error ? 1 : 0)
          << " declared_n_m1=" << grid_.get_nb_macroparticles(1)
          << " declared_n_m2=" << grid_.get_nb_macroparticles(2)
          << " live_distinct_roots1="
          << beam1_.particle_beam().numberOfParticles()
          << " live_distinct_roots2="
          << beam2_.particle_beam().numberOfParticles()
          << " root_families1=" << root_families1
          << " root_families2=" << root_families2
          << " encounter_source_builds="
          << adaptive_particle_encounters
          << " commits=" << adaptive_particle_commits
          << " root_visits1=" << adaptive_particle_root_visits1
          << " root_visits2=" << adaptive_particle_root_visits2
          << " premerge_representative_visits1="
          << adaptive_particle_premerge_visits1
          << " premerge_representative_visits2="
          << adaptive_particle_premerge_visits2
          << " representative_visits1="
          << adaptive_particle_representative_visits1
          << " representative_visits2="
          << adaptive_particle_representative_visits2
          << " peak_representatives1="
          << adaptive_particle_peak_representatives1
          << " peak_representatives2="
          << adaptive_particle_peak_representatives2
          << " final_live_representatives1=0"
          << " final_live_representatives2=0"
          << " merge_reductions1=" << merge_reductions1
          << " merge_reductions2=" << merge_reductions2
          << " many_root_representative_visits1="
          << adaptive_particle_many_root_visits1
          << " many_root_representative_visits2="
          << adaptive_particle_many_root_visits2
          << " maximum_weight_ess1="
          << adaptive_particle_max_weight_ess1
          << " maximum_weight_ess2="
          << adaptive_particle_max_weight_ess2
          << " minimum_clone_support_proxy1="
          << adaptive_particle_min_clone_support1
          << " minimum_clone_support_proxy2="
          << adaptive_particle_min_clone_support2
          << " maximum_projection_relative_l1="
          << adaptive_particle_max_projection_residual
          << " controller=coupled_error_budget_v1"
          << " controller_observations="
          << adaptive_particle_controller_observations
          << " controller_source_sampling_known=0"
          << " controller_source_sampling_unknown_observations="
          << adaptive_particle_controller_sampling_unknown
          << " controller_action_hold="
          << adaptive_particle_controller_actions[0]
          << " controller_action_refine="
          << adaptive_particle_controller_actions[1]
          << " controller_action_replay="
          << adaptive_particle_controller_actions[2]
          << " controller_action_joint="
          << adaptive_particle_controller_actions[3]
          << " controller_action_merge="
          << adaptive_particle_controller_actions[4]
          << " controller_action_diagnostics="
          << adaptive_particle_controller_actions[5]
          << " controller_action_unmet="
          << adaptive_particle_controller_actions[6]
          << " controller_recommends_refine="
          << adaptive_particle_controller_recommendations[1]
          << " controller_recommends_replay="
          << adaptive_particle_controller_recommendations[2]
          << " controller_recommends_joint="
          << adaptive_particle_controller_recommendations[3]
          << " controller_recommends_merge="
          << adaptive_particle_controller_recommendations[4]
          << " controller_replay_required="
          << (adaptive_particle_controller_replay_required ? 1 : 0)
          << " controller_total_bound=unavailable_single_run"
          << " trace_hash=" << adaptive_particle_trace_hash
          << " roots_added_online=0"
          << " source_sampling_error_reduced_by_split=0"
          << " physical_primary_population_mutations=0"
          << " source_enters_lumi_ee=0"
          << " source_enters_stochastic_physics=0"
          << " qualification=software_verified_not_physics_qualified"
          << std::endl;
      if (adaptive.mode !=
              feebl::adaptive_particles::Mode::Diagnostics &&
          adaptive_particle_controller_replay_required)
        {
          if (adaptive_particle_controller_first_unmet_reason.empty() ||
              adaptive_particle_controller_replay_recommendation ==
                  feebl::adaptive_particles::ControllerAction::Hold)
            throw std::logic_error(
                "adaptive particle replay request is missing its canonical "
                "controller reason or recommendation");
          adaptive_root_replay_required_ = true;
          adaptive_root_replay_reason_ =
              adaptive_particle_controller_first_unmet_reason;
          adaptive_root_replay_recommended_action_ =
              feebl::adaptive_particles::controller_action_name(
                  adaptive_particle_controller_replay_recommendation);
          adaptive_root_replay_controller_observations_ =
              adaptive_particle_controller_observations;
          adaptive_root_replay_trace_hash_ = adaptive_particle_trace_hash;
        }
      if (adaptive.mode !=
              feebl::adaptive_particles::Mode::Diagnostics &&
          adaptive.fail_on_unmet_error &&
          adaptive_particle_controller_unmet &&
          !adaptive_particle_controller_replay_required)
        throw std::runtime_error(
            "adaptive macroparticle end-to-end error target is unmet: " +
            adaptive_particle_controller_first_unmet_reason +
            "; no physics files were published (use independent complete "
            "runs/root replay, or set fail_on_unmet_error=0 only for an "
            "explicitly unqualified mechanism run)");
    }
  if (mode2_noise_aware_stop_encounters > mode2_unmet_error_encounters)
    throw std::logic_error(
        "adaptive noise-stop count exceeds unmet-error count");
  if (switches.get_adaptive_grid_mode() >= 1 &&
      !backend.gpu_mode5_enabled())
    {
      const double full_fraction = adaptive_encounters == 0U
          ? 0.0
          : static_cast<double>(adaptive_full_windows) /
                static_cast<double>(adaptive_encounters);
      std::cerr
          << "FEEBL adaptive grid summary: requested="
          << switches.get_adaptive_grid_mode()
          << " actual=" << switches.get_adaptive_grid_mode()
          << " backend=cpu topology="
          << (switches.get_adaptive_grid_mode() == 1
                  ? "exact-aligned-window-fft-v1"
                  : (switches.get_adaptive_grid_mode() == 2
                         ? "uniform-variable-spacing-fft-v1"
                         : "coarse-plus-nested-dirichlet-patch-v1"))
          << " controller="
          << (switches.get_adaptive_grid_mode() == 1
                  ? "uniform-window-v1"
                  : (switches.get_adaptive_grid_mode() == 2
                          ? "same-particle-richardson-v1"
                          : "richardson-connected-overlap-patch-v1"))
          << " field_operator="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_field_operator()
                  : "not_applicable")
          << " error_norm_policy="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_error_norm_policy()
                  : "not_applicable")
          << " error_quantile="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_error_quantile()
                  : 0.0)
          << " error_reference_field="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_error_reference_field()
                  : 0.0)
          << " error_absolute_floor_fraction="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches
                        .get_adaptive_grid_error_absolute_floor_fraction()
                  : 0.0)
          << " error_catastrophe_absolute_fraction="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches
                        .get_adaptive_grid_error_catastrophe_absolute_fraction()
                  : 0.0)
          << " error_exposure_limit="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_error_exposure_limit()
                  : 0.0)
          << " error_budget_sha256="
          << (switches.get_adaptive_grid_error_budget_sha256().empty()
                  ? "not_applicable"
                  : switches.get_adaptive_grid_error_budget_sha256())
          << " error_catastrophe_limit="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_error_catastrophe_limit()
                  : 0.0)
          << " noise_aware_stop_enabled="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_noise_aware_stop()
                  : 0)
          << " pilot_diagnostics="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_pilot_diagnostics()
                  : 0)
          << " physics_qualification_claim=0"
          << " convergence_claim=0"
          << " pilot_output_status="
          << (switches.get_adaptive_grid_pilot_diagnostics() != 0
                  ? "unqualified_diagnostics"
                  : "not_pilot")
          << " terminal_unmet_policy="
          << (switches.get_adaptive_grid_pilot_diagnostics() != 0
                  ? "publish_finest_resource_admissible_uniform_unqualified"
                  : "abort_or_bounded_tail_budget")
          << " capacity_max_cells="
          << switches.get_adaptive_grid_max_cells()
          << " capacity_max_bytes="
          << switches.get_adaptive_grid_max_bytes()
          << " capacity_max_refinement_factor="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_max_refinement_factor()
                  : 0U)
          << " max_refinement_factor="
          << (switches.get_adaptive_grid_mode() >= 2
                  ? switches.get_adaptive_grid_max_refinement_factor()
                  : 0U)
          << " scope=encounter"
          << " encounters=" << adaptive_encounters
          << " plans=" << adaptive_plan_builds
          << " required_unique_node_visits=" << adaptive_required_cells
          << " local_cell_visits=" << adaptive_local_cells
          << " padded_fft_cell_visits=" << adaptive_padded_cells
          << " canonical_cells_avoided="
          << adaptive_global_cells_avoided
          << " full_window_selections=" << adaptive_full_windows
          << " full_window_fraction=" << full_fraction
          << " context_builds=" << adaptive_context_builds
          << " context_hits=" << adaptive_context_hits
          << " uniform_context_misses="
          << adaptive_uniform_context_misses
          << " patch_context_misses=" << adaptive_patch_context_misses
          << " context_evictions=" << adaptive_context_evictions
          << " context_bytes_estimate=" << adaptive_context_bytes
          << " bounds_wall_ms=" << adaptive_planning_ms
          << " bounds_worker_ms=" << adaptive_bounds_worker_ms
          << " bucketing_ms=" << adaptive_bucketing_ms
          << " context_setup_wall_ms=" << adaptive_context_setup_ms
          << " context_build_worker_ms="
          << adaptive_context_build_worker_ms
          << " deposit_worker_ms=" << adaptive_deposit_worker_ms
          << " solve_worker_ms=" << adaptive_solve_worker_ms
          << " gather_worker_ms=" << adaptive_gather_worker_ms
          << " mode2_encounters=" << mode2_encounters
          << " mode2_field_solves=" << mode2_field_solves
          << " mode2_refined_encounters=" << mode2_refined_encounters
          << " mode2_unmet_error_encounters="
          << mode2_unmet_error_encounters
          << " mode2_noise_aware_stop_encounters="
          << mode2_noise_aware_stop_encounters
          << " vector_catastrophe_failure_encounters="
          << vector_catastrophe_failure_encounters
          << " maximum_vector_difference_maximum="
          << maximum_vector_difference
          << " maximum_beam_above_target_exposure="
          << maximum_beam_above_target_exposure
          << " mode2_unresolved_non_noise_encounters="
          << (mode2_unmet_error_encounters -
              mode2_noise_aware_stop_encounters)
          << " terminal_resource_ladder_unmet_encounters="
          << (mode2_unmet_error_encounters -
              mode2_noise_aware_stop_encounters)
          << " terminal_resource_ladder_unmet_semantics="
          << "legacy_unresolved_non_noise_v1"
          << " mode2_tail_budget_eligible_encounters="
          << mode2_tail_budget_eligible_encounters
          << " mode2_tail_budget_used_encounters="
          << mode2_tail_budget_used_encounters
          << " encounter_tail_budget_selected_pairs="
          << encounter_tail_budget_selected_pairs
          << " encounter_tail_eligibility_fraction="
          << encounter_tail_eligibility_fraction
          << " encounter_tail_budget_selected_fraction="
          << encounter_tail_budget_selected_fraction
          << " encounter_tail_field_metric="
          << "symmetric_vector_triplet_v1"
          << " encounter_tail_weighted_richardson_error_diagnostic="
          << encounter_tail_weighted_field_error
          << " encounter_tail_weighted_field_sensitivity="
          << encounter_tail_weighted_field_sensitivity
          << " encounter_tail_weighted_source_lumi_error="
          << encounter_tail_weighted_source_lumi_error
          << " encounter_all_uniform_lumi_measure="
          << encounter_all_uniform_lumi_measure
          << " encounter_tail_uniform_lumi_measure="
          << encounter_tail_uniform_lumi_measure
          << " encounter_tail_lumi_exposure_fraction="
          << encounter_tail_lumi_exposure_fraction
          << " encounter_tail_budget_met="
          << (encounter_tail_budget_met ? 1 : 0)
          << " maximum_estimated_field_error="
          << maximum_estimated_field_error
          << " mode3_patch_requests=" << mode3_patch_requests
          << " mode3_patch_encounters=" << mode3_patch_encounters
          << " mode3_uniform_promotions=" << mode3_uniform_promotions
          << " mode3_terminal_uniform_encounters="
          << mode3_terminal_uniform_encounters
          << " mode3_terminal_patch_attempts="
          << mode3_terminal_patch_attempts
          << " mode3_terminal_patch_certification_enabled="
          << switches.get_adaptive_grid_terminal_patch_certification()
          << " mode3_terminal_patch_accepts="
          << mode3_terminal_patch_accepts
          << " mode3_terminal_patch_used="
          << mode3_terminal_patch_used
          << " mode3_terminal_patch_rejects="
          << mode3_terminal_patch_rejects
          << " mode3_terminal_patch_uniform_fallbacks="
          << mode3_terminal_patch_uniform_fallbacks
          << " mode3_terminal_patch_rejects_with_resource="
          << mode3_terminal_patch_resource_rejects
          << " mode3_terminal_patch_rejects_with_nonresource="
          << mode3_terminal_patch_nonresource_rejects
          << " mode3_terminal_patch_rejection_mask_or="
          << mode3_terminal_patch_rejection_mask_or
          << " mode3_no_patch_encounters=" << mode3_no_patch_encounters
          << " mode3_empty_encounters=" << mode3_empty_encounters
          << " mode3_patch_coarse_cells=" << mode3_patch_coarse_cells
          << " mode3_patch_fine_cells=" << mode3_patch_fine_cells
          << " mode3_attempted_patch_coarse_cells="
          << mode3_attempted_patch_coarse_cells
          << " mode3_attempted_patch_fine_cells="
          << mode3_attempted_patch_fine_cells
          << " mode3_harmonic_iterations="
          << mode3_harmonic_iterations
          << " mode3_rejection_mask_or=" << mode3_rejection_mask_or
          << " mode3_controller_inclusive_worker_ms="
          << mode3_controller_worker_ms
          << " mode3_harmonic_worker_ms=" << mode3_harmonic_worker_ms
          << " maximum_patch_residual=" << maximum_patch_residual
          << " maximum_patch_field_error=" << maximum_patch_field_error
          << " patch_lumi_oracle_evaluations="
          << patch_lumi_oracle_evaluations
          << " accepted_patch_lumi_oracle_evaluations="
          << accepted_patch_lumi_oracle_evaluations
          << " maximum_accepted_patch_lumi_oracle_error="
          << maximum_accepted_patch_lumi_oracle_error
          << " maximum_examined_patch_lumi_oracle_error="
          << maximum_examined_patch_lumi_oracle_error
          << " source_model=" << switches.get_adaptive_grid_source_model()
          << " source_reference_factor="
          << (switches.get_adaptive_grid_mode() >= 2 &&
                      switches.get_adaptive_grid_source_model() ==
                          "reference_cic_v1"
                  ? switches.get_adaptive_grid_source_reference_nx() /
                        static_cast<std::uint64_t>(grid_.get_n_cell_x())
                  : switches.get_adaptive_grid_mode() >= 2 &&
                            switches.get_adaptive_grid_source_model() ==
                                "fixed_width_bspline1_v1"
                        ? 0U
                        : 1U)
          << " source_budget_runtime_verified=0"
          << " source_hierarchy_encounters="
          << source_hierarchy_encounters
          << " source_density_hash_nonzero_encounters="
          << source_density_hash_nonzero_encounters
          << " source_sampling_hash_nonzero_encounters="
          << source_sampling_hash_nonzero_encounters
          << " source_sampling_hierarchy_hash_nonzero_encounters="
          << source_sampling_hierarchy_hash_nonzero_encounters
          << " source_beam1_in_visits=" << source_beam1_in_visits
          << " source_beam1_out_visits=" << source_beam1_out_visits
          << " source_beam2_in_visits=" << source_beam2_in_visits
          << " source_beam2_out_visits=" << source_beam2_out_visits
          << " source_build_worker_ms=" << source_build_worker_ms
          << " source_restrict_worker_ms=" << source_restrict_worker_ms
          << " source_prolong_worker_ms=" << source_prolong_worker_ms
          << " maximum_source_charge_closure="
          << maximum_source_charge_closure
          << " maximum_source_projection_moment_shift="
          << maximum_source_projection_moment_shift
          << " maximum_selected_source_lumi_oracle_error="
          << maximum_selected_source_lumi_oracle_error
          << " maximum_examined_source_lumi_oracle_error="
          << maximum_source_lumi_oracle_error
          << " fallback_count=" << mode3_uniform_promotions
          << " cold_miss_count=" << adaptive_context_cold_misses
          << " card_hash=" << adaptive_card_hash
          << " schedule_hash=" << adaptive_schedule_hash
          << std::endl;
      if (switches.get_adaptive_grid_mode() >= 2) {
        const double catastrophe_limit =
            switches.get_adaptive_grid_error_catastrophe_limit();
        const feebl::wavefront_fields::runtime_safety::
            AdaptiveGridCertificateDecision certificate_decision =
                feebl::wavefront_fields::runtime_safety::
                    evaluate_adaptive_grid_certificate(
                        mode2_encounters,
                        mode2_unmet_error_encounters,
                        mode2_noise_aware_stop_encounters,
                        vector_catastrophe_failure_encounters,
                        maximum_vector_difference,
                        catastrophe_limit,
                        mode3_terminal_patch_attempts,
                        mode3_terminal_patch_accepts,
                        mode3_terminal_patch_used,
                        mode3_terminal_patch_rejects,
                        mode3_terminal_patch_uniform_fallbacks,
                        mode3_terminal_patch_resource_rejects,
                        mode3_terminal_patch_nonresource_rejects,
                        mode3_terminal_patch_rejection_mask_or,
                        adaptive_root_replay_required_);
        const std::uint64_t unresolved_non_noise_encounters =
            certificate_decision.unresolved_non_noise_encounters;
        // Preserve the V1 field for existing readers.  Its historical value
        // means unresolved_non_noise, not a proven memory/cell/factor cause.
        const std::uint64_t resource_unmet_encounters =
            certificate_decision.resource_unmet_encounters;
        const std::uint64_t certified_encounters =
            certificate_decision.certified_encounters;
        const bool external_replay_recommended =
            certificate_decision.external_replay_recommended;
        const bool all_encounters_certified =
            certificate_decision.all_encounters_certified;
        std::cerr
            << "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1"
            << " schema=feebl.adaptive-grid-certificate.v1"
            << " requested_mode=" << switches.get_adaptive_grid_mode()
            << " actual_mode=" << switches.get_adaptive_grid_mode()
            << " error_norm_policy="
            << switches.get_adaptive_grid_error_norm_policy()
            << " source_model="
            << switches.get_adaptive_grid_source_model()
            << " error_budget_sha256="
            << (switches.get_adaptive_grid_error_budget_sha256().empty()
                    ? "not_applicable"
                    : switches.get_adaptive_grid_error_budget_sha256())
            << " card_hash=" << adaptive_card_hash
            << " error_target="
            << switches.get_adaptive_grid_error_target()
            << " error_quantile="
            << switches.get_adaptive_grid_error_quantile()
            << " error_reference_field="
            << switches.get_adaptive_grid_error_reference_field()
            << " error_absolute_floor_fraction="
            << switches.get_adaptive_grid_error_absolute_floor_fraction()
            << " error_catastrophe_absolute_fraction="
            << switches
                   .get_adaptive_grid_error_catastrophe_absolute_fraction()
            << " error_exposure_limit="
            << switches.get_adaptive_grid_error_exposure_limit()
            << " maximum_beam_above_target_exposure="
            << maximum_beam_above_target_exposure
            << " pilot_diagnostics="
            << switches.get_adaptive_grid_pilot_diagnostics()
            << " fail_on_unmet_error="
            << switches.get_adaptive_grid_fail_on_unmet_error()
            << " encounters=" << mode2_encounters
            << " certified_encounters=" << certified_encounters
            << " unmet_encounters=" << mode2_unmet_error_encounters
            << " noise_stop_encounters="
            << mode2_noise_aware_stop_encounters
            << " resource_unmet_encounters="
            << resource_unmet_encounters
            << " vector_catastrophe_failure_encounters="
            << vector_catastrophe_failure_encounters
            << " maximum_vector_difference="
            << maximum_vector_difference
            << " catastrophe_limit=" << catastrophe_limit
            << " terminal_patch_enabled="
            << switches.get_adaptive_grid_terminal_patch_certification()
            << " terminal_patch_attempts="
            << mode3_terminal_patch_attempts
            << " terminal_patch_accepts="
            << mode3_terminal_patch_accepts
            << " terminal_patch_rejects="
            << mode3_terminal_patch_rejects
            << " terminal_patch_rejection_mask_or="
            << mode3_terminal_patch_rejection_mask_or
            << " external_replay_recommended="
            << (external_replay_recommended ? 1 : 0)
            << " all_encounters_certified="
            << (all_encounters_certified ? 1 : 0)
            << " disposition="
            << (all_encounters_certified
                    ? "accepted_all_encounters"
                    : "unqualified_diagnostics")
            << std::endl;
        std::cerr
            << "FEEBL_ADAPTIVE_GRID_UNRESOLVED_V1"
            << " schema=feebl.adaptive-grid-unresolved.v1"
            << " unresolved_non_noise_encounters="
            << unresolved_non_noise_encounters
            << " legacy_resource_unmet_alias="
            << resource_unmet_encounters
            << " terminal_patch_attempts="
            << mode3_terminal_patch_attempts
            << " terminal_patch_accepts="
            << mode3_terminal_patch_accepts
            << " terminal_patch_used="
            << mode3_terminal_patch_used
            << " terminal_patch_rejects="
            << mode3_terminal_patch_rejects
            << " terminal_patch_uniform_fallbacks="
            << mode3_terminal_patch_uniform_fallbacks
            << " terminal_patch_rejects_with_resource="
            << mode3_terminal_patch_resource_rejects
            << " terminal_patch_rejects_with_nonresource="
            << mode3_terminal_patch_nonresource_rejects
            << " terminal_patch_rejection_mask_or="
            << mode3_terminal_patch_rejection_mask_or
            << std::endl;
      }
      for (const auto& item : adaptive_class_encounters)
        std::cerr
            << "FEEBL adaptive class: nx=" << item.first.first
            << " ny=" << item.first.second
            << " encounters=" << item.second << std::endl;
      for (std::size_t index = 0U;
           index < mode2_spacing_factors.size(); ++index)
        std::cerr
            << "FEEBL adaptive spacing class: factor="
            << mode2_spacing_factors[index]
            << " selected_encounters="
            << mode2_spacing_selections[index] << std::endl;
    }
#ifdef FEEBL_HAVE_GPU5
  if (switches.get_adaptive_grid_mode() == 1 &&
      backend.gpu_mode5_enabled())
    {
      if (!gpu5_wave_solver_)
        throw std::logic_error(
            "gpu_mode=5 telemetry requested without an executor");
      const feebl::adaptive_grid::gpu5::AdaptiveGpuTelemetry stats =
          gpu5_wave_solver_->telemetry();
      const double full_fraction = stats.encounters == 0U
          ? 0.0
          : static_cast<double>(stats.full_window_encounters) /
                static_cast<double>(stats.encounters);
      std::cerr
          << "FEEBL_GPU5_TELEMETRY_V1"
          << " requested_topology=1 actual_topology=1"
          << " requested_backend=gpu actual_backend=gpu"
          << " requested_gpu_mode=5 actual_gpu_mode=5"
          << " controller=uniform-window-v1 scope=encounter"
          << " schedule_hash=" << adaptive_schedule_hash
          << " prebuild_complete=" << (stats.prebuild_complete ? 1 : 0)
          << " classes_prebuilt=" << stats.classes_prebuilt
          << " plan_cache_hits=" << stats.plan_cache_hits
          << " plan_cache_misses=" << stats.plan_cache_misses
          << " green_cache_hits=" << stats.green_cache_hits
          << " green_cache_misses=" << stats.green_cache_misses
          << " plan_creations=" << stats.plan_creations
          << " green_creations=" << stats.green_creations
          << " batch_calls=" << stats.batch_calls
          << " encounters=" << stats.encounters
          << " empty_encounters=" << stats.empty_encounters
          << " full_window_selections="
          << stats.full_window_encounters
          << " full_window_fraction=" << full_fraction
          << " required_unique_node_visits="
          << stats.required_node_visits
          << " local_cell_visits=" << stats.selected_local_cell_visits
          << " padded_fft_cell_visits="
          << stats.padded_fft_cell_visits
          << " actual_planmany_cell_visits="
          << stats.actual_planmany_cell_visits
          << " canonical_cells_avoided=" << stats.canonical_cells_avoided
          << " padded_fft_slots=" << stats.padded_fft_slots
          << " deposit_beam1_in=" << stats.deposit_beam1_in
          << " deposit_beam1_out=" << stats.deposit_beam1_out
          << " deposit_beam2_in=" << stats.deposit_beam2_in
          << " deposit_beam2_out=" << stats.deposit_beam2_out
          << " gather_beam1_in=" << stats.gather_beam1_in
          << " gather_beam1_out=" << stats.gather_beam1_out
          << " gather_beam2_in=" << stats.gather_beam2_in
          << " gather_beam2_out=" << stats.gather_beam2_out
          << " h2d_operations=" << stats.h2d_operations
          << " d2h_operations=" << stats.d2h_operations
          << " h2d_bytes=" << stats.h2d_bytes
          << " d2h_bytes=" << stats.d2h_bytes
          << " kernel_launches=" << stats.kernel_launches
          << " cufft_executes=" << stats.cufft_executes
          << " synchronizations=" << stats.synchronizations
          << " memory_budget_bytes=" << stats.memory_budget_bytes
          << " tracked_device_bytes=" << stats.tracked_device_bytes
          << " peak_tracked_device_bytes="
          << stats.peak_tracked_device_bytes
          << " measured_device_high_water_bytes="
          << stats.measured_device_high_water_bytes
          << " pinned_host_bytes=" << stats.pinned_host_bytes
          << " plan_workspace_bytes=" << stats.plan_workspace_bytes
          << " setup_ms=" << stats.setup_ms
          << " plan_build_ms=" << stats.plan_build_ms
          << " green_build_ms=" << stats.green_build_ms
          << " bounds_wall_ms=" << adaptive_planning_ms
          << " bucketing_ms=" << adaptive_bucketing_ms
          << " host_pack_ms=" << stats.host_pack_ms
          << " h2d_ms=" << stats.h2d_ms
          << " kernel_ms=" << stats.kernel_ms
          << " d2h_ms=" << stats.d2h_ms
          << " host_publish_ms=" << stats.host_publish_ms
          << " containment_failures=" << stats.containment_failures
          << " binding_failures=" << stats.binding_failures
          << " fallback_count=0 cold_miss_count=0"
          << " context_poisoned=" << (stats.context_poisoned ? 1 : 0)
          << " deposition="
          << feebl::adaptive_grid::gpu5::deposition_variant_name(
                 stats.actual_deposition)
          << std::endl;
      for (const auto& item : adaptive_class_encounters)
        std::cerr
            << "FEEBL adaptive class: backend=gpu gpu_mode=5 nx="
            << item.first.first << " ny=" << item.first.second
            << " encounters=" << item.second << std::endl;
    }
#endif
  if (switches.get_wavefront_field_solver() != "fft")
    std::cerr
        << "FEEBL sparse-direct field summary: requested="
        << switches.get_wavefront_field_solver()
        << " crossover_interactions="
        << switches.get_sparse_direct_interaction_limit()
        << " direct_encounters=" << direct_encounters
        << " fft_encounters=" << fft_encounters
        << " active_sources=" << direct_active_sources
        << " target_nodes=" << direct_target_nodes
        << " estimated_interactions=" << estimated_interactions
        << " direct_interactions=" << direct_interactions
        << " maximum_estimated_interactions="
        << maximum_estimated_interactions
        << " direct_worker_ms=" << direct_ms
        << " fft_worker_ms=" << fft_ms
        << " kernel=case2-integrated-green-v1"
        << " source_order=ascending-linear-cell"
        << " fallback=explicit-auto-crossover-only"
        << std::endl;
}

void FEEBL::beam_interaction_gpu_mode3()
{
  PARALLEL_BACKEND& backend = parallel_backend();
  if (!backend.gpu_mode3_enabled())
    throw std::logic_error(
        "mode-3 production crossing called without an active gpu_mode=3 backend");

  const feebl::gpu::Mode3ProductionReceipt receipt =
      feebl::gpu::execute_mode3_primary_crossing(
          beam1_, beam2_, beam_parameters1_, beam_parameters2_, grid_,
          switches, mode3_input_card_label_, backend.gpu_device(),
          backend.actual_gpu2_options().memory_fraction);

  const feebl::gpu::ResidentPrimaryStats& stats = receipt.crossing.resident;
  // GRID::step_lumi accumulates this charge-density observable even when
  // do_lumi=0.  The resident pipeline returns the sum of rho1*rho2 over every
  // encounter.  Its legacy normalization was finite-validated before the
  // paired beam write-back; publish it only now, after that transaction.
  grid_.get_results().add_lumi_fine(receipt.normalized_fine_lumi);
  grid_.get_results().set_primary_only_mode3(true);

  backend.record_gpu_mode3_execution(receipt.crossing.device_description);
  switches.set_collision_schedule_provenance(
      "gpu_mode3_primary_resident_all_waves",
      receipt.diagonals, receipt.waves, 1, receipt.encounters, 0,
      receipt.maximum_wave_width);

  const std::streamsize previous_cerr_precision = std::cerr.precision();
  std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10)
      << " FEEBL GPU mode3 production summary:"
      << " implementation=" << receipt.crossing.implementation_id
      << " card=" << receipt.crossing.input_card_label
      << " precision=" << receipt.crossing.precision
      << " modern_reduction=" << receipt.crossing.modern_reduction
      << " initialization_rng=" << receipt.initialization_rng
      << " mode3_collision_rng=" << receipt.collision_rng
      << " import_contracts_bound_before_cuda="
      << receipt.import_contracts_bound_before_cuda
      << " paired_writeback_preflight_before_cuda="
      << receipt.paired_writeback_preflight_before_cuda
      << " paired_writeback_committed="
      << receipt.writeback.crossing_transaction_committed
      << " upsilon_committed_beam1="
      << receipt.writeback.beam1.upsilon_matches_beam_soa
      << " upsilon_committed_beam2="
      << receipt.writeback.beam2.upsilon_matches_beam_soa
      << " bunch_population1_live=" << receipt.bunch_population1
      << " bunch_population2_live=" << receipt.bunch_population2
      << " declared_n_m1=" << receipt.declared_macro_particles1
      << " declared_n_m2=" << receipt.declared_macro_particles2
      << " macro_charge1_live=" << receipt.macro_charge1
      << " macro_charge2_live=" << receipt.macro_charge2
      << " effective_population1=" << receipt.effective_population1
      << " effective_population2=" << receipt.effective_population2
      << " population_residual1="
      << receipt.bunch_population1 - receipt.effective_population1
      << " population_residual2="
      << receipt.bunch_population2 - receipt.effective_population2
      << " diagonals=" << receipt.diagonals
      << " waves=" << receipt.waves
      << " encounters=" << receipt.encounters
      << " uploads=" << stats.upload_calls
      << " resident_batches=" << stats.batch_calls
      << " downloads=" << stats.download_calls
      << " h2d_operations=" << stats.h2d_operations
      << " d2h_operations=" << stats.d2h_operations
      << " h2d_bytes=" << stats.h2d_bytes
      << " d2h_bytes=" << stats.d2h_bytes
      << " setup_h2d_operations=" << stats.setup_h2d_operations
      << " setup_h2d_bytes=" << stats.setup_h2d_bytes
      << " setup_synchronizations=" << stats.setup_synchronizations
      << " synchronizations=" << stats.synchronizations
      << " intermediate_particle_h2d_bytes="
      << stats.intermediate_particle_h2d_bytes
      << " intermediate_particle_d2h_bytes="
      << stats.intermediate_particle_d2h_bytes
      << " host_field_materializations=" << stats.host_field_materializations
      << " deposit_beam1_in_total=" << stats.deposit_beam1_in
      << " deposit_beam1_out_total=" << stats.deposit_beam1_out
      << " deposit_beam2_in_total=" << stats.deposit_beam2_in
      << " deposit_beam2_out_total=" << stats.deposit_beam2_out
      << " gather_beam1_in_total=" << stats.gather_beam1_in
      << " gather_beam1_out_total=" << stats.gather_beam1_out
      << " gather_beam2_in_total=" << stats.gather_beam2_in
      << " gather_beam2_out_total=" << stats.gather_beam2_out
      << " legacy_grid_miss_diagnostics=unavailable"
      << " raw_fine_lumi_dot_diagnostic=" << stats.fine_lumi_dot_sum
      << " normalized_fine_lumi=" << receipt.normalized_fine_lumi
      << " fine_charge_density_luminosity_implemented=1"
      << " full_pair_event_luminosity_implemented=0"
      << " fallback_used=" << receipt.crossing.fallback_used
      << " device=\"" << receipt.crossing.device_description << "\""
      << std::endl;
  std::cerr.precision(previous_cerr_precision);
}

void FEEBL::simulate()
{
  //int i1;
  int gridTimestep;
  float gridScalStep[2], gridMaxZ, gridStep;  
  PHI_FLOAT sor_parameter[6];
  set_simulation();
  set_beams_and_grids();
  set_output_data_and_files();
  grid_.init_sor2(sor_parameter);
  gridMaxZ = grid_.get_max_z();
  gridStep = grid_.get_step();
  gridTimestep = grid_.get_timestep();
  gridScalStep[0] = grid_.get_scal_step(0);
  gridScalStep[1] = grid_.get_scal_step(1);

  beam1_.backstep(1, gridMaxZ, gridStep, gridTimestep, gridScalStep);
  beam2_.backstep(2, gridMaxZ, gridStep, gridTimestep, gridScalStep);

  if (parallel_backend().gpu_mode3_enabled())
    {
      // The qualified subset rejects spin, but keep the ordinary reporting
      // lifecycle explicit and truthful: these samples are disabled rather
      // than fabricated from an unported transport.
      polarization_.set_enabled(false);
      polarization_.sample(POLARIZATION_REPORT::INITIAL, beam1_, beam2_);
      beam_interaction_gpu_mode3();
      polarization_.sample(POLARIZATION_REPORT::FINAL, beam1_, beam2_);
      if (switches.get_store_beam())
        {
          emit_primary_upsilon_digest(beam1_.particle_beam());
          emit_primary_upsilon_digest(beam2_.particle_beam());
        }
      return;
    }

  // Worker-local full collision workspaces need the immutable pair kinematic
  // constants during preflight.  Establish them before any accelerated plan;
  // this is deterministic setup and consumes no random numbers.
  secondaries_.set_pair_parameters(
      beam1_, beam2_, 0, switches.get_pair_ecut(), switches.get_pair_step(),
      grid_.get_step(), grid_.get_timestep());
  muons_.set_pair_parameters(
      beam1_, beam2_, 1, switches.get_muon_ecut(), switches.get_pair_step(),
      grid_.get_step(), grid_.get_timestep());

  // All maps, rotations, offsets and backsteps are now complete, and every
  // grid has its final FFT spectrum.  Commit accelerated plans before the
  // collision mutates either beam.
  initialize_gpu2_fields();
  initialize_gpu4_wave_batch();
  initialize_wavefront_fields();

  // Polarization is only meaningful if spin is being transported at all, and
  // spin exists only when bmt_precession is on.
  polarization_.set_enabled(beam1_.has_spin() && beam2_.has_spin());
  polarization_.sample(POLARIZATION_REPORT::INITIAL, beam1_, beam2_);

  switch (switches.get_time_order())
    {
    case 1:
      std::cerr << " SIMULATE time_order = 1 to be done " << std::endl;
      exit(1);
      //      break;
    case 2:
      if (switches.get_track_pairs() && switches.get_track_muons())
		{
		  beam_interaction_with_trackpair_muon(secondaries_, muons_, sor_parameter );
		}
      else if (switches.get_track_pairs()) 
		{
		  beam_interaction_with_trackpair(secondaries_, sor_parameter );
		}
      else if (switches.get_track_muons()) 
		{
		  beam_interaction_with_trackpair(muons_, sor_parameter );
		}
	  else  beam_interaction( sor_parameter );
      break;
    }

  polarization_.sample(POLARIZATION_REPORT::FINAL, beam1_, beam2_);
  if (switches.get_store_beam())
    {
      emit_primary_upsilon_digest(beam1_.particle_beam());
      emit_primary_upsilon_digest(beam2_.particle_beam());
    }
}

std::string FEEBL::output_flow() const 
{
  std::ostringstream out;
  //double esum1,esum2;
  out << title(std::string("other information "));
  if (!generator_.loaded_checkpoint_checksum().empty())
    out << " RNG checkpoint loaded: FEEBL_RNG_V2; payload_fnv1a64="
        << generator_.loaded_checkpoint_checksum()
        << "; checkpoint state supersedes rndm_seed" << std::endl;
  //   esum1 = beam1_.meanEnergy();
  //   esum2 = beam2_.meanEnergy();
  //   out << " de1 = " <<  esum1 << " de2 = " <<  esum2 << std::endl;
  //if (switches.get_store_secondaries()) 
  if (switches.get_store_pairs() > 1)
    {
      out << " initial pairs are saved on file : " << secondaries0_file_ << std::endl;
    }
  //if (switches.get_track_secondaries()) 
  if (switches.get_track_pairs() || switches.get_store_pairs())
   {
     out << " pairs are saved on file : " << secondaries_file_ << std::endl;
   }
  if (switches.get_store_muons()) 
    {
      out << " initial muons are saved on file : " << muons0_file_ << std::endl;
    }
  //if (switches.get_track_secondaries()) 
  if (switches.get_track_muons()) 
   {
     out << " muons are saved on file : " << muons_file_ << std::endl;
   }
  if (switches.get_do_bhabhas())
    {
      int nbhabha_ini, nbhabha_photon_ini;
      grid_.get_bhabhas().numbers_of_loaded(nbhabha_ini, nbhabha_photon_ini); 
      out << " " << nbhabha_ini << " bhabhas have been loaded " << nbhabha_photon_ini << " bhabha_photons have been loaded " <<std::endl; 
      out << " read bhabha samples are saved on file : " << bhabha_prod_ << std::endl;
    }
  if (switches.get_do_jets())
    {
      out << " jets are saved on file : " << jet_file_ << std::endl;
    }
    if (switches.get_store_beam()) 
    {
      out << number_of_stored_particles1_  << " particles are saved on file : " << beam1_file_ << std::endl;
      out << number_of_stored_particles2_  << " particles are saved on file : " << beam2_file_ << std::endl;
      out << " coherent particles of beam1 are saved on file : " << coh1_file_ << std::endl;
      out << " coherent particles of beam2 are saved on file : " << coh2_file_ << std::endl;
      out << " trident particles of beam1 are saved on file : " << tri1_file_ << std::endl;
      out << " trident particles of beam2 are saved on file : " << tri2_file_ << std::endl;
    }
   return out.str();
}

void FEEBL::make_step(
    int i1, int i2, PHI_FLOAT* sor_parameter,
    const feebl::wavefront_fields::FieldPrefetch* prefetched_fields)
{
  if (prefetched_fields)
    feebl::wavefront_fields::validate_field_result_binding(
        *prefetched_fields, i1, i2,
        beam1_.getParticleVector(i1), beam2_.getParticleVector(i2));
  int i_grid;
  int i_offset;

  float min_z = 0.5*(i2-i1-1);

  const PAIR_PARAMETER& ppar = secondaries_.get_pair_parameters();
  const PAIR_PARAMETER& mpar = muons_.get_pair_parameters();
  if(ppar.get_s4()!=mpar.get_s4()){
    std::cout << "s4 must be the same for muons and pairs" << std::endl;
  }
  double gpu2_fine_lumi_dot = 0.0;
  const bool audit_mesh=get_make_gui_files() && !prefetched_fields &&
      gui_mesh_coverage_reason().empty();
  const auto deposition_before=grid_.current_deposition_counts();
  feebl::collider_fields::SourceScan deposited_scan1,deposited_scan2;
  feebl::collider_fields::MeshEncounterCoverage mesh_encounter;

  if (prefetched_fields)
    {
      // Workers have produced only immutable rho-derived data.  Rebuild the
      // authoritative background cell lists here so all event/RNG traversal
      // remains in legacy order.
      grid_.all_distribute_background_only(
          i1, i2, switches, ppar.get_s4(), ppar.get_lns4());
      grid_.add_external_distribution_counts(
          prefetched_fields->beam1_in, prefetched_fields->beam1_out,
          prefetched_fields->beam2_in, prefetched_fields->beam2_out);
    }
  else if (gpu2_direct_deposition_)
    {
      // Preserve the host collision-cell and virtual-photon RNG paths.  The
      // RNG-free primary charge deposition now precedes step_lumi on device;
      // its compact metric is the only rho-derived value needed on host.
      grid_.all_distribute_background_only(
          i1, i2, switches, ppar.get_s4(), ppar.get_lns4());
      gpu2_fine_lumi_dot = deposit_gpu2_fields(i1, i2);
    }
  else
    {
      grid_.all_distribute(i1, i2, switches,ppar.get_s4(), ppar.get_lns4());
				
      for (i_grid=1;i_grid<=switches.get_extra_grids();i_grid++)
        gridsPtr_[i_grid]->distribute_particles(
            i1, i2, switches.get_electron_distribution_rho(),
            switches.get_force_symmetric());
    }
  
  if(audit_mesh) {
    deposited_scan1=feebl::collider_fields::scan_mesh_sources(grid_,beam1_.getParticleVector(i1));
    deposited_scan2=feebl::collider_fields::scan_mesh_sources(grid_,beam2_.getParticleVector(i2));
    const auto after=grid_.current_deposition_counts();
    const std::size_t cells=std::size_t(grid_.get_n_cell_x())*grid_.get_n_cell_y();
    mesh_encounter.beam[0]=feebl::collider_fields::bind_native_deposition(
        deposited_scan1,deposition_before[0],deposition_before[1],after[0],after[1],
        grid_.get_rho1(),cells,grid_.get_macro_charge(1));
    mesh_encounter.beam[1]=feebl::collider_fields::bind_native_deposition(
        deposited_scan2,deposition_before[2],deposition_before[3],after[2],after[3],
        grid_.get_rho2(),cells,grid_.get_macro_charge(2));
    mesh_encounter.id=time_counter_;mesh_encounter.diagonal=i1+i2;
    mesh_encounter.beam1_slice=i1;mesh_encounter.beam2_slice=i2;
  }

  time_.add_timer(1);
  if (prefetched_fields)
    grid_.step_lumi_with_fine_dot(
        min_z, secondaries_, time_counter_, switches, i1, i2,
        prefetched_fields->fine_lumi_dot);
  else if (gpu2_direct_deposition_)
    grid_.step_lumi_with_fine_dot(
        min_z, secondaries_, time_counter_, switches, i1, i2,
        gpu2_fine_lumi_dot);
  else
    grid_.step_lumi(min_z, secondaries_, time_counter_, switches, i1, i2 ) ;
  time_counter_++;

  time_.add_timer(2); 

  grid_.update_slice_charge(i1, i2);

  if (prefetched_fields)
    {
      // phi belongs to a worker-private scratch grid and is intentionally not
      // materialized here; compact fields are the only stage boundary.
    }
  else if (parallel_backend().gpu_mode2_enabled())
    solve_gpu2_fields(i1, i2);
  else
    {
      grid_.computeFields(switches.get_integration_method(), switches.get_charge_sign(),sor_parameter);
      for (i_grid=1;i_grid<=switches.get_extra_grids();i_grid++)
        {
          gridsPtr_[i_grid]->computeFields(switches.get_integration_method(), switches.get_charge_sign(),sor_parameter);
        }
    }
  
  if(audit_mesh) {
    // Bind coverage to coordinates immediately before the real serial mover.
    // The exact XY vector comparison proves no luminosity/background stage
    // changed the already-deposited primary states before their field gather.
    feebl::collider_fields::verify_pre_kick_sources(deposited_scan1,
        feebl::collider_fields::scan_mesh_sources(grid_,beam1_.getParticleVector(i1)));
    feebl::collider_fields::verify_pre_kick_sources(deposited_scan2,
        feebl::collider_fields::scan_mesh_sources(grid_,beam2_.getParticleVector(i2)));
    const int nz=grid_.get_n_cell_z();
    if(gui_mesh_slice_visits_.empty())gui_mesh_slice_visits_.assign(std::size_t(nz)*nz,0);
    mesh_encounter.substep_index=gui_mesh_slice_visits_.at(std::size_t(i1)*nz+i2)++;
    if(mesh_encounter.substep_index>=grid_.get_timestep() || gui_mesh_encounters_.size()>=32768)
      throw std::logic_error("native mesh coverage encounter bound exceeded");
    gui_mesh_encounters_.push_back(mesh_encounter);
  }

  // Sample only an actual committed host potential, before any particle kick.
  capture_gui_fields(i1,i2,!prefetched_fields &&
      !parallel_backend().gpu_mode2_enabled());

  time_.add_timer(3);
  //int nbeam = 1;

  //   grid_.moveAllParticles( gridsPtr_, beam1_, nbeam, i1, switches.get_interpolation(), switches.get_do_beamstrahlung(),switches.get_ST_spin_flip(), switches.get_emin(), switches.get_do_prod(),switches.get_extra_grids(), switches.get_charge_sign(),switches.get_bmt_precession());

  // photons emitted while the particles move are the newborn photons of this step (photon_timing)
  grid_.mark_photon_birth(i1, i2, switches.get_photon_timing());
  // photon_angle: the field seen by an emitter changes over half the opposing bunch length
  grid_.set_photon_angle(switches.get_photon_angle(), 0.5f*beam_parameters2_.sigma_z(), 0.5f*beam_parameters1_.sigma_z());
  if (prefetched_fields)
    {
      grid_.movePrimaryParticlesWithPrefetchedFields(
          gridsPtr_, i1, i2, switches.get_interpolation(),
          switches.get_do_beamstrahlung(), switches.get_ST_spin_flip(),
          switches.get_emin(), switches.get_do_prod(),
          switches.get_extra_grids(), switches.get_charge_sign(),
          switches.get_bmt_precession(),
          prefetched_fields->fields1.empty()
              ? NULL : &prefetched_fields->fields1[0],
          prefetched_fields->fields1.size() / 2U,
          prefetched_fields->fields2.empty()
              ? NULL : &prefetched_fields->fields2[0],
          prefetched_fields->fields2.size() / 2U,
          switches.get_do_trident(), switches.get_do_coherent());
    }
  else if (gpu2_primary_scope_)
    {
      const std::size_t count1 = beam1_.getParticleVector(i1).size();
      const std::size_t count2 = beam2_.getParticleVector(i2).size();
      grid_.movePrimaryParticlesWithFields(
          gridsPtr_, i1, i2, switches.get_interpolation(),
          switches.get_do_beamstrahlung(), switches.get_ST_spin_flip(),
          switches.get_emin(), switches.get_do_prod(),
          switches.get_extra_grids(), switches.get_charge_sign(),
          switches.get_bmt_precession(),
          count1 ? &gpu2_primary_fields1_[0] : NULL, count1,
          count2 ? &gpu2_primary_fields2_[0] : NULL, count2,
          parallel_backend().actual_gpu2_options().primary_mover,
          parallel_backend().actual_gpu2_options().host_threads,
          switches.get_do_trident(), switches.get_do_coherent());
    }
  else
    grid_.moveAllParticles(
        gridsPtr_, i1, i2, switches.get_interpolation(),
        switches.get_do_beamstrahlung(), switches.get_ST_spin_flip(),
        switches.get_emin(), switches.get_do_prod(),
        switches.get_extra_grids(), switches.get_charge_sign(),
        switches.get_bmt_precession(), switches.get_do_trident(),
        switches.get_do_coherent(), switches.get_cpu_primary_move());
  
  // nbeam = 2;
  //  std::cout << " advance the second beam.... " << std::endl;
  //  grid_.moveAllParticles( gridsPtr_, beam2_, nbeam, i2, switches.get_interpolation(), switches.get_do_beamstrahlung(),switches.get_ST_spin_flip(),switches.get_emin(), switches.get_do_prod(),switches.get_extra_grids(), switches.get_charge_sign(), switches.get_bmt_precession());
  
  time_.add_timer(4);
 
  if ( switches.get_do_photons(1) || switches.get_do_photons(2) || switches.get_load_photon())
    {
      // FEEBL: photon_lumi collides every photon against every particle and
      // every other photon -- 2.1 billion calls on ILC 250, ~23% of wall
      // time -- to build lumi_eg / lumi_ge / lumi_gg. Those ARE written to
      // the output, so skipping them is not free and must be opt-in:
      // do_photon_lumi defaults to 1, i.e. upstream behaviour, bit-identical.
      //
      // The photon deposition itself still runs whenever a downstream
      // consumer needs it -- photon_lumi_2 (pairs/hadrons/compt/muons),
      // photon_lumi_3 (compt) and the coherent path all read it.
      const bool feebl_need_photon_grid =
          switches.get_do_photon_lumi() || switches.get_do_pairs()
       || switches.get_do_hadrons()     || switches.get_do_compt()
       || switches.get_do_muons()       || switches.get_do_coherent();
      // distribute_photons draws from generator_, so skipping it shifts the RNG
      // stream and perturbs everything downstream (measured: +0.076% on ILC).
      // It is only ~4% of runtime, so always run it: that keeps do_photon_lumi=0
      // bit-identical while still skipping photon_lumi, which is the ~19%.
      (void)feebl_need_photon_grid;
      grid_.distribute_photons(i1,i2, switches.get_photon_ratio(),generator_ );
      if (switches.get_do_photon_lumi())
        { grid_.photon_lumi(min_z,switches,secondaries_, muons_, generator_); }

      if(switches.get_do_pairs()||switches.get_do_hadrons()||switches.get_do_compt() || switches.get_do_muons())
		{
		  // here the pairs are generated
		  grid_.photon_lumi_2(min_z,switches, secondaries_, muons_, generator_);
		}
      if (switches.get_do_compt()) 
		{
		  grid_.photon_lumi_3(min_z,switches, secondaries_, generator_);
		}
      if (switches.get_do_coherent())
		{
		  grid_.move_photons2(beam1_,1,i1, generator_);
		  grid_.move_photons2(beam2_,2,i2,  generator_);
		}
      else
		{
		  grid_.move_photons(beam1_,1,i1);
		  grid_.move_photons(beam2_,2,i2);
		}
      
    } // end if (store photons)
  else if (switches.get_pair_model() >= 1 && (switches.get_do_pairs() || switches.get_do_muons()))
    {
      // The virtual photons are distributed in all_distribute whenever these processes are on, but the
      // legacy code only collides them inside the real-photon block above, so a run without real
      // photons silently produced no Landau-Lifshitz (virtual x virtual) pairs.  The corrected models
      // collide them here; the real-photon lists are empty, so only the virtual x virtual loop acts.
      grid_.photon_lumi_2(min_z,switches, secondaries_, muons_, generator_);
    }
  time_.add_timer(5);
  i_offset=i1+i2-grid_.get_n_cell_z();
  //if(switches.get_track_secondaries())
  if(switches.get_track_pairs())
    {
      double d_eps_1, d_eps_2;

      secondaries_.get_pair_parameters().get_d_eps(d_eps_1, d_eps_2);
      if(switches.get_do_tertphot())
	{
	  if (i_offset<0)
	    {
	      grid_.move_pairs_tertphot(gridsPtr_, secondaries_, i1, d_eps_1, d_eps_2, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	  else
	    {
	      grid_.move_pairs_tertphot(gridsPtr_,secondaries_, i1-i_offset-1, d_eps_1, d_eps_2, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	}
      else
	{
	  if (i_offset<0)
	    {
	      grid_.move_pairs(gridsPtr_, secondaries_, i1, d_eps_1, d_eps_2, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	  else
	    {
	      grid_.move_pairs(gridsPtr_,secondaries_, i1-i_offset-1, d_eps_1, d_eps_2, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	}
    }
  if(switches.get_track_muons())
    {
      double d_eps_1_mu, d_eps_2_mu;

      muons_.get_pair_parameters().get_d_eps(d_eps_1_mu, d_eps_2_mu);
      if(switches.get_do_tertphot())
	{
	  if (i_offset<0)
	    {
	      grid_.move_pairs_tertphot(gridsPtr_, muons_, i1, d_eps_1_mu, d_eps_2_mu, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	  else
	    {
	      grid_.move_pairs_tertphot(gridsPtr_, muons_, i1-i_offset-1, d_eps_1_mu, d_eps_2_mu, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	}
      else
	{
	  if (i_offset<0)
	    {
	      grid_.move_pairs(gridsPtr_, muons_, i1, d_eps_1_mu, d_eps_2_mu, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }
	  else
	    {
	      grid_.move_pairs(gridsPtr_, muons_, i1-i_offset-1, d_eps_1_mu, d_eps_2_mu, switches.get_extra_grids(), switches.get_charge_sign_0(), generator_);
	    }	  
	}
    }
  // std::cout << " for the extra photons " << std::endl;
  if (switches.get_do_pairs() || 
      switches.get_do_hadrons() ||
      switches.get_do_compt() ||
      switches.get_do_muons())
    {
      grid_.clear_extra_photons();
    }
  time_.add_timer(6);
}



void FEEBL::iteration_on_overlaping_slices(int firstSliceOfBeam1, int lastSliceOfBeam2,PHI_FLOAT* sor_parameter)
{
  int i0;
  grid_.get_results().set_step_lumi_ee (0.0);
  grid_.get_results().set_step_lumi_ee_high(0.0);
  for (i0=0;i0<grid_.get_timestep();i0++)
    {
      make_time_step_on_slices(firstSliceOfBeam1, lastSliceOfBeam2, sor_parameter);
    }
   grid_.get_results().add_lumi_ee_step(grid_.get_results().get_step_lumi_ee());
   grid_.get_results().add_lumi_ee_high_step(grid_.get_results().get_step_lumi_ee_high());	
  if (switches.get_do_dump())
    {
      int istep;
      istep = firstSliceOfBeam1 + lastSliceOfBeam2;
      if (istep%switches.get_dump_step() ==0)
		{
		  dump_beams(istep,switches.get_dump_step(), switches.get_dump_particle());
		}
    } 
  write_gui_snapshot(firstSliceOfBeam1 + lastSliceOfBeam2);
}

void FEEBL::iteration_on_overlaping_slices_with_trackpair(PAIR_BEAM& pair_beam_ref, int firstSliceOfBeam1, int lastSliceOfBeam2, PHI_FLOAT* sor_parameter)
{
  int i0;
  unsigned int numberToDistribute = lastSliceOfBeam2-firstSliceOfBeam1+1;
  grid_.get_results().set_step_lumi_ee (0.0);
  grid_.get_results().set_step_lumi_ee_high(0.0);
  if (switches.get_load_event()) 
    {
      //pair_beam_ref.load_events(time_counter_, switches.get_pair_ratio(), switches.get_track_secondaries(), generator_);
      pair_beam_ref.load_events(time_counter_, switches.get_pair_ratio(), switches.get_track_pairs(), generator_);
    }
  for (i0=0;i0<grid_.get_timestep();i0++)
    {
      pair_beam_ref.distribute_pairs(grid_.get_delta_z(),numberToDistribute);
      pair_beam_ref.move_unactive_pairs(grid_.get_step());
      make_time_step_on_slices(firstSliceOfBeam1, lastSliceOfBeam2, sor_parameter);
      // once the current pairs have been moved the have to be redistributed
      // and before that, desactived
      pair_beam_ref.desactive_current_pairs(numberToDistribute);
    }
  grid_.get_results().add_lumi_ee_step(grid_.get_results().get_step_lumi_ee());
  grid_.get_results().add_lumi_ee_high_step(grid_.get_results().get_step_lumi_ee_high());
  if (switches.get_do_dump())
    {
      int istep;
      istep = firstSliceOfBeam1 + lastSliceOfBeam2;
      if (istep%switches.get_dump_step() ==0)
	dump_beams(istep,switches.get_dump_step(), switches.get_dump_particle());
    } 
  write_gui_snapshot(firstSliceOfBeam1 + lastSliceOfBeam2);
}

void FEEBL::iteration_on_overlaping_slices_with_trackpair_muon(PAIR_BEAM& pair_beam_ref, PAIR_BEAM& muon_beam_ref, int firstSliceOfBeam1, int lastSliceOfBeam2, PHI_FLOAT* sor_parameter)
{
  int i0;
  unsigned int numberToDistribute = lastSliceOfBeam2-firstSliceOfBeam1+1;
  grid_.get_results().set_step_lumi_ee (0.0);
  grid_.get_results().set_step_lumi_ee_high(0.0);
  if (switches.get_load_event()) 
    {
      //pair_beam_ref.load_events(time_counter_, switches.get_pair_ratio(), switches.get_track_secondaries(), generator_);
      pair_beam_ref.load_events(time_counter_, switches.get_pair_ratio(), switches.get_track_pairs(), generator_);
    }
  for (i0=0;i0<grid_.get_timestep();i0++)
    {
      pair_beam_ref.distribute_pairs(grid_.get_delta_z(),numberToDistribute);
      pair_beam_ref.move_unactive_pairs(grid_.get_step());
      muon_beam_ref.distribute_pairs(grid_.get_delta_z(),numberToDistribute);
      muon_beam_ref.move_unactive_pairs(grid_.get_step());

      make_time_step_on_slices(firstSliceOfBeam1, lastSliceOfBeam2, sor_parameter);
      // once the current pairs have been moved the have to be redistributed
      // and before that, desactived
      pair_beam_ref.desactive_current_pairs(numberToDistribute);
      muon_beam_ref.desactive_current_pairs(numberToDistribute);
    }
   grid_.get_results().add_lumi_ee_step(grid_.get_results().get_step_lumi_ee());
   grid_.get_results().add_lumi_ee_high_step(grid_.get_results().get_step_lumi_ee_high());
  if (switches.get_do_dump())
    {
      int istep;
      istep = firstSliceOfBeam1 + lastSliceOfBeam2;
      if (istep%switches.get_dump_step() ==0)
	dump_beams(istep,switches.get_dump_step(), switches.get_dump_particle());
    } 
  write_gui_snapshot(firstSliceOfBeam1 + lastSliceOfBeam2);
}

void FEEBL::print_program_outputs(std::string nameOfProtokoll)
{

  FILE_IN_OUT filout;

  filout.open_file(nameOfProtokoll, "w");
  filout.set_header(header());
  filout.save_object_on_output_listing(&beam_parameters1_);
  filout.save_object_on_output_listing(&beam_parameters2_);
  filout.save_object_on_output_listing(&switches);
  filout.save_object_on_output_listing(&grid_);
  filout.save_object_on_output_listing(&beam1_);
  filout.save_object_on_output_listing(&beam2_);
  if (switches.get_do_coherent())
    {
      filout.save_object_on_output_listing(&grid_.get_coherent_results());
    }
  if (switches.get_do_trident())
    {
      filout.save_object_on_output_listing(&grid_.get_trident_results());
    }
  if (switches.get_do_pairs() || switches.get_do_bhabhas() || switches.get_do_compt() || switches.get_load_event())
    {
      //     filout.save_object_on_output_file(&secondaries_);
      filout.save_object_on_output_listing(secondaries_.get_results());
    }
  if (switches.get_do_muons())
    {
      filout.save_object_on_output_listing(muons_.get_results());
    }
  if (switches.get_do_jets())
    {
      filout.save_object_on_output_listing(grid_.get_minijets());
    }
  if (switches.get_do_cross())
    {
      filout.save_object_on_output_listing(cross_);
    }
  if (switches.get_do_compt())
    {
      filout.save_object_on_output_listing(&grid_.get_compton());
    }
  //  filout.save_object_on_output_file(& muon_results);
  filout.save_object_on_output_listing(&grid_.get_results());
  polarization_.set_luminosity_weighted(
      grid_.get_results().get_lumi_weighted_helicity1(),
      grid_.get_results().get_lumi_weighted_helicity2(),
      grid_.get_results().get_lumi_weight_sum());
  filout.save_object_on_output_listing(&polarization_);
  filout.save_object_on_output_listing(this);
  filout.close();
}

void FEEBL::xycuts_for_grids(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 , int nbgrids, std::vector<float>& size_x, std::vector<float>& size_y, int& updated_n_cell_x, int& updated_n_cell_y ) const
{
  int counter;
  float tmp;
   //float cutx, cuty;
  if (nbgrids > 7) 
    {
      std::cerr << " FEEBL::cuts_from_sigmas : ERROR the number of grids is limited to 6 , nbgrids = " << nbgrids << std::endl;
      exit(1);
    }
  size_x.clear();
  size_y.clear();
  size_x.resize(nbgrids);
  size_y.resize(nbgrids);
  counter = 0;
  if (switches.get_automatic_grid_sizing() ) 
    {
      // automatic generation of cuts
      main_grid_automatic_xycuts(bff1,bff2, size_x[0], size_y[0], updated_n_cell_x, updated_n_cell_y);
    }
  else 
    {
      // cuts from user's data : 
      main_grid_xycuts_from_user(bff1,bff2, size_x[0], size_y[0]);
    }
  counter++;
  if (counter >= nbgrids) return;
  size_x[counter]=2.0*size_x[0];
  size_y[counter]=2.0*size_y[0];
  counter++;
  if (counter >= nbgrids) return;
  size_x[counter]=size_x[1];
  tmp=pow((double)(size_x[0]/size_y[0]),(double)0.333);
  size_y[counter]=size_y[1]*tmp;
  counter++;
  if (counter >= nbgrids) return;

  size_x[counter]=size_x[1];
  size_y[counter]=size_y[1]*tmp*tmp;
  counter++;
  if (counter >= nbgrids) return;

  size_x[counter]=size_x[1];
  size_y[counter]=size_x[1];
  counter++;
  if (counter >= nbgrids) return;

  size_x[counter]=size_x[1]*2.0;
  size_y[counter]=size_x[1]*2.0;
  counter++;
  if (counter >= nbgrids) return;

  size_x[counter]=size_x[1]*6.0;
  size_y[counter]=size_x[1]*6.0;
}

float FEEBL::zcut_for_grids(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ) const
{
  //float  cutz, 
  float size_z;
  if (switches.get_automatic_grid_sizing() ) 
    {
      // automatic generation of cuts
      size_z = default_zcut_from_beams(bff1,bff2);
    }
  else 
    {
      // cuts from user's data : 
      size_z = main_grid_zcut_from_user(bff1,bff2);
    }
  return size_z;
}


void FEEBL::main_grid_xycuts_from_loaded_beam(const BEAM_FROM_FILE& bff1,const BEAM_FROM_FILE& bff2, float& size_x, float& size_y ) const
{
  float cutx, cuty;
      float dumx, dumy;
      float sigx, sigy, sigx2, sigy2; 
      cutx = params_.readFValue("cut_x_factor");
      cuty = params_.readFValue("cut_y_factor");
      bff1.beamXyRms(dumx, dumy, sigx, sigy);
      bff2.beamXyRms(dumx, dumy, sigx2, sigy2);
      if (sigx < sigx2) sigx = sigx2;
      if (sigy < sigy2) sigy = sigy2;
      if(cutx>0.0) size_x = cutx*sigx;
      else size_x = 3.0*sigx;
      if(cuty>0.0) size_y = cuty*sigy;
      else size_y = 3.0*sigy;  
}
float FEEBL::main_grid_zcut_from_loaded_beam(const BEAM_FROM_FILE& bff1,const BEAM_FROM_FILE& bff2) const
{
  float  cutz;
  float dumz, size_z;
      float sigz, sigz2; 
      cutz = params_.readFValue("cut_z_factor");
      bff1.beamZRms(dumz, sigz);
      bff2.beamZRms( dumz,sigz2);
      if (sigz < sigz2) sigz = sigz2;
      if(cutz>0.0) size_z = cutz*sigz;
      else size_z = default_zcut(sigz);
      return size_z;
}




// ---------------------------------------------------------------------------
// FEEBL: optimize_grids
//
// Derive the grid and stepping configuration from the coupled beam scales.
// These empirical defaults were calibrated on the 16-point luminosity test.
// They provide starting settings, not an observable-specific convergence
// certificate; the comments record the scale behind each choice.
//
// The controlling physics is:
//   D_y      vertical disruption; how hard the opposing bunch focuses this one
//   Phi      Piwinski angle, sigma_z*tan(theta)/sigma_x; how much the crossing
//            angle shortens the luminous region
// ---------------------------------------------------------------------------
void FEEBL::optimize_grid_settings()
{
  const float RE_LOCAL  = 2.8179403262e-15f;   // classical electron radius, m
  const float EMASS_GEV = 0.51099895069e-3f;

  // sigma_* are stored in nm, beta_* in mm once read() has finished.
  float sx = 0.5f*(beam_parameters1_.sigma_x() + beam_parameters2_.sigma_x())*1e-9f;
  float sy = 0.5f*(beam_parameters1_.sigma_y() + beam_parameters2_.sigma_y())*1e-9f;
  float sz = 0.5f*(beam_parameters1_.sigma_z() + beam_parameters2_.sigma_z())*1e-9f;
  float by = 0.5f*(beam_parameters1_.beta_y()  + beam_parameters2_.beta_y())*1e-3f;
  float en = 0.5f*(beam_parameters1_.ebeam()   + beam_parameters2_.ebeam());
  float np = 0.5f*(beam_parameters1_.n_particles() + beam_parameters2_.n_particles());
  float ax = 0.5f*(fabs(beam_parameters1_.x_angle()) + fabs(beam_parameters2_.x_angle()));

  if (sx <= 0.0f || sy <= 0.0f || sz <= 0.0f || en <= 0.0f)
    {
      std::cerr << " FEEBL::optimize_grid_settings: ERROR : optimize_grids=1 needs"
                << " sigma_x, sigma_y, sigma_z and energy to be determinable"
                << " from the ACCELERATOR block." << std::endl;
      exit(1);
    }

  float gamma = en/EMASS_GEV;
  float Dy  = 2.0f*RE_LOCAL*np*sz/(gamma*sy*(sx+sy));
  float Dx  = 2.0f*RE_LOCAL*np*sz/(gamma*sx*(sx+sy));
  float Phi = (ax > 0.0f) ? sz*tan(ax)/sx : 0.0f;

  // ---- cut_x -------------------------------------------------------------
  // With a crossing angle the bunch centroid sweeps transversely by z*sin(ax),
  // so the box must span +-3 sigma_z of that sweep as well as the beam itself.
  // Upstream omits the sweep entirely.
  float sweep = (ax > 0.0f) ? 3.0f*sz*sin(ax) : 0.0f;
  float cutx  = sweep + 6.0f*sx;

  // ---- cut_y -------------------------------------------------------------
  // Two regimes, both calibrated against the hand-tuned benchmark decks:
  //   crossing-angle machines (Phi > 1): what leaves the box is the hourglass
  //     tail, which carries charge but almost no luminosity, so 8 sigma_y is
  //     enough and large grid escape is harmless (measured: FCC-ee Z runs at
  //     58% escape and still agrees to 0.5% with a box 4x wider).
  //   head-on machines: the vertical pinch itself has to fit, which scales
  //     with the disruption. 10*sqrt(1+D_y) reproduces the hand-tuned choices
  //     (LEP D_y=0.11 -> 12 sigma_y; ILC D_y=34.5 -> 59 sigma_y).
  float cuty;
  if (Phi > 1.0f) cuty = 8.0f*sy;
  else
    {
      // Hand tuning used 12 sigma_y at D_y = 0.1 and 60 sigma_y from
      // D_y = 24 upward; a sqrt law under-sizes the middle. Interpolate
      // linearly and saturate, which reproduces both anchors.
      float f = 12.0f + 48.0f*(Dy/25.0f);
      if (f < 12.0f) f = 12.0f;
      if (f > 60.0f) f = 60.0f;
      cuty = f*sy;
    }

  // ---- cells -------------------------------------------------------------
  // Target delta <= 0.125 sigma in both transverse directions. Since
  // delta = 2*cut/n, that means n = 16*cut/sigma -- NOT 8*cut/sigma, which
  // yields delta = 0.25 sigma and, after rounding down to a power of two, as
  // much as 0.375 sigma.
  //
  // This matters more than it looks. The luminosity is an overlap integral of
  // two peaked Gaussians; sampling the core on a coarse mesh biases it upward.
  // Measured on the ILC 500/550/1000 group: delta_x = 0.375 sigma_x gave
  // +6.0%, while the hand-tuned 0.047 sigma_x agreed to 0.1%. And in the other
  // direction, a box wide enough to hold the tails but resolved only to
  // delta_y = 3.4 sigma_y came out 46% LOW. Both failures are silent.
  //
  // Round UP to the next power of two, never down: rounding down is what
  // turned a 0.25 sigma target into 0.375 sigma.
  // The cap is not arbitrary. When a crossing angle is present cut_x is
  // dominated by the SWEEP (FCC-ee Z: 84 sigma_x, of which 78 is sweep), but
  // the luminosity is produced near x = 0 at each z -- the swept region is
  // traversed, not integrated over. Resolving the whole sweep to 0.125 sigma
  // would need 2048 cells in x and make the optimizer SLOWER than the
  // hand-tuned deck, which is the opposite of the point. 512 is where the hand
  // tuning converged for exactly this reason.
  // delta = 2*cut/n, so n = 8*cut/sigma gives delta = 0.25 sigma.
  //
  // 0.25 sigma is not a guess. An n_y ladder on ILC 500, everything else held
  // at the hand-tuned values, gave:
  //     delta_y/sigma_y   0.938   0.469   0.234   0.117
  //     L (1e38)          1.169   1.429   1.509   1.528
  // i.e. still climbing 5.6% between 0.469 and 0.234 and only 1.2% between
  // 0.234 and 0.117. Anything coarser than ~0.25 sigma is materially low; the
  // hand-tuned decks sat at 0.469 sigma_y and were 6.5% under-converged as a
  // result. Round UP, never down.
  int nx = 32; while ((float)nx < 8.0f*cutx/sx && nx < 1024) nx *= 2;
  int ny = 32; while ((float)ny < 8.0f*cuty/sy && ny < 1024) ny *= 2;
  if (nx > 1024) nx = 1024;
  if (ny > 1024) ny = 1024;

  // ---- n_z ---------------------------------------------------------------
  // Longitudinal slicing has to resolve the pinch. ILC 250 (D_y = 34.5) at
  // fixed step: n_z 32 -> 64 moved L by -1.2%, 64 -> 128 by -0.12%.
  // Measured: 64 slices is converged from D_y = 0.1 (LEP) all the way to
  // D_y = 773 (FCC-ee Z); the ILC ladder moved -1.2% from 32 to 64 and only
  // -0.12% from 64 to 128. An earlier logarithmic rule picked 128 at high D_y,
  // which cost 2x and moved FCC-ee Z by +2.4% for no gain in accuracy.
  int nz = (Dy < 1.0f) ? 32 : 64;

  // ---- n_t ---------------------------------------------------------------
  // The kick is integrated with step deltaz/(2*n_t), deltaz = 2*cut_z/n_z, so
  // only the product n_z*n_t matters; measured convergence at ~320. At low
  // disruption it is irrelevant (LEP moved 0.01% from n_t=1 to n_t=10), so
  // scale the requirement by how strong the pinch actually is.
  float strength = Dy/10.0f; if (strength > 1.0f) strength = 1.0f;
  int nt = (int)ceil((320.0f/(float)nz)*strength);
  if (nt < 1)  nt = 1;
  if (nt > 20) nt = 20;

  // ---- grids -------------------------------------------------------------
  // Nested grids only serve particles outside the main grid. Measured
  // bit-identical at zero escape, and 0.07% different at 58% escape, for 6.3x
  // the cost.
  int ngrids = 1;

  // ---- publish -----------------------------------------------------------
  // Write into the parameter store so the ordinary (non-automatic) path
  // consumes them, and so the values reported below are the values used.
  params_.setDoubleValue("cut_x", cutx*1e9);          // nm
  params_.setDoubleValue("cut_y", cuty*1e9);          // nm
  params_.setDoubleValue("cut_z", 3.0f*sz*1e6);       // micrometers
  params_.setDoubleValue("grids", (double)ngrids);

  optimized_n_x_ = nx; optimized_n_y_ = ny;
  optimized_n_z_ = nz; optimized_n_t_ = nt; optimized_grids_ = ngrids;

  std::cout << std::endl;
  std::cout << " ---------------- FEEBL optimize_grids ----------------" << std::endl;
  std::cout << " derived from the beams:" << std::endl;
  std::cout << "   sigma_x = " << sx*1e9 << " nm      sigma_y = " << sy*1e9 << " nm" << std::endl;
  std::cout << "   sigma_z = " << sz*1e3 << " mm      beta_y* = " << by*1e3 << " mm" << std::endl;
  std::cout << "   D_x     = " << Dx << "   D_y = " << Dy << std::endl;
  std::cout << "   Piwinski= " << Phi;
  if (ax > 0.0f) std::cout << "   (half crossing angle " << ax*1e3 << " mrad)";
  std::cout << std::endl;
  std::cout << " chosen:" << std::endl;
  std::cout << "   cut_x = " << cutx*1e9 << " nm  = " << cutx/sx << " sigma_x";
  if (sweep > 0.0f) std::cout << "   (includes " << sweep*1e9 << " nm crossing-angle sweep)";
  std::cout << std::endl;
  std::cout << "   cut_y = " << cuty*1e9 << " nm  = " << cuty/sy << " sigma_y"
            << ((Phi > 1.0f) ? "   (crossing-angle regime)" : "   (head-on regime, from D_y)")
            << std::endl;
  std::cout << "   n_x   = " << nx << "   delta_x = " << (2.0f*cutx/nx)/sx << " sigma_x" << std::endl;
  std::cout << "   n_y   = " << ny << "   delta_y = " << (2.0f*cuty/ny)/sy << " sigma_y" << std::endl;
  std::cout << "   n_z   = " << nz << "   n_t = " << nt
            << "   (n_z*n_t = " << nz*nt << ")" << std::endl;
  std::cout << "   grids = " << ngrids << std::endl;
  std::cout << std::endl;
  std::cout << " To reproduce this run WITHOUT the flag, delete optimize_grids" << std::endl;
  std::cout << " and paste the following into $PARAMETERS:" << std::endl << std::endl;
  std::cout << "   n_x=" << nx << ";n_y=" << ny << ";n_z=" << nz << ";n_t=" << nt << ";" << std::endl;
  std::cout << "   cut_x=" << cutx*1e9 << ";cut_y=" << cuty*1e9
            << ";cut_z=" << 3.0f*sz*1e6 << ";grids=" << ngrids << ";" << std::endl;
  std::cout << "   (cut_x, cut_y in nm; cut_z in micrometers)" << std::endl << std::endl;
  std::cout << " NOTE: heuristics calibrated on the FEEBL benchmark, not a" << std::endl;
  std::cout << " convergence proof for YOUR configuration. If the answer matters," << std::endl;
  std::cout << " vary n_z, n_t and n_y and confirm the result is flat." << std::endl;
  std::cout << " ------------------------------------------------------" << std::endl;
  std::cout << std::endl;
}

void FEEBL::main_grid_automatic_xycuts(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ,float& size_x, float& size_y, int& new_n_cell_x, int& new_n_cell_y) const
{
  float x01, y01, sigmax1, sigmay1;
  float x02, y02, sigmax2, sigmay2;
  float dx, dy, xmoy, ymoy, sigmax, sigmay;
  float theta0, gam1, gam2, gamma;
  float npart1, npart2, nbpart;
  float Dx, Dy;
  float betax, betay;
  float offset;
  float emittx1, emitty1, emittx2, emitty2;
  float z01, sigmaz1;
  float z02, sigmaz2;
  float sigmaz;

  bff1.beamXyRms(x01, y01, sigmax1, sigmay1);
  bff2.beamXyRms(x02,y02, sigmax2, sigmay2);
  bff1.emittances(emittx1, emitty1);
  bff2.emittances(emittx2, emitty2);
  gam1 = bff1.gamma();
  gam2 = bff2.gamma();

  // emittx = 0.5*(emittx1+emittx2);
  // emitty = 0.5*(emitty1+emitty2);

  npart1  = beam_parameters1_.n_particles();
  npart2  = beam_parameters2_.n_particles();

  xmoy = 0.5*( x01 + x02 );
  ymoy = 0.5*( y01 + y02 );
  dx = 0.5*fabs(x02 - x01);
  dy = 0.5*fabs(y02 - y01);
  //
  sigmax = (sigmax2 > sigmax1) ? sigmax2 : sigmax1; 
  sigmay = (sigmay2 > sigmay1) ? sigmay2 : sigmay1;

  gamma = 0.5*(gam1 + gam2);
  //  std::cout << " gamma = " << gamma << std::endl;
  nbpart = 0.5*(npart1 + npart2);


  // the sigmas are in nanometres
  theta0 = 2.*nbpart*RE/(gamma*(sigmax+sigmay)*1.0e-9);

  bff1.beamZRms(z01, sigmaz1);
  bff2.beamZRms(z02, sigmaz2);

  sigmaz = (sigmaz2 > sigmaz1) ? sigmaz2 : sigmaz1;


  Dy = theta0*sigmaz/sigmay;
  Dx = theta0*sigmaz/sigmax;

  betay = 0.5*(beam_parameters1_.beta_y()+beam_parameters2_.beta_y());
  betax = 0.5*(beam_parameters1_.beta_x()+beam_parameters2_.beta_x());

  test_size_due_to_cdm(gamma, sigmax, sigmay, sigmaz, betax, betay);


  offset = size_due_to_cdm_deflection(theta0, sigmaz, sigmax, betax, Dx,dx);
  //  std::cout << " offset in x " << offset << std::endl;
  // FEEBL: with a crossing angle the bunch centroid sweeps transversely by
  // z*sin(angle_x) as it crosses, so the grid must span +-3 sigma_z of that
  // sweep or most of the interaction leaves the box. Upstream omitted this
  // term entirely; for FCC-ee at 30 mrad it is 652 um against a 6 sigma_x of
  // 50 um, i.e. the grid was 13x too narrow.
  {
    float ax = 0.5f*(fabs(beam_parameters1_.x_angle()) + fabs(beam_parameters2_.x_angle()));
    if (ax > 0.0f) size_x_sweep_ = 3.0f*sigmaz*sin(ax); else size_x_sweep_ = 0.0f;
  }
  size_x = fabs(xmoy) + dx + offset + 6.0*sigmax + size_x_sweep_;
  //  std::cout << " size_x = " << size_x << std::endl;
  offset = size_due_to_cdm_deflection(theta0, sigmaz, sigmay, betay, Dy,dy);
  //  std::cout << " offset in y " << offset << std::endl;
  size_y = fabs(ymoy) + dy + offset + 10.0*sigmay;
  //  std::cout << " size_y = " << size_y << std::endl;
  new_n_cell_x = adjust_nb_cells_from_cut(size_x, sigmax);
  new_n_cell_y = adjust_nb_cells_from_cut(size_y, sigmay);
}
float FEEBL::default_zcut_from_beams(const ABSTRACT_PARTICLE_BEAM& bff1,const ABSTRACT_PARTICLE_BEAM& bff2 ) const
{
  float z01, sigmaz1;
  float z02, sigmaz2;
  float sigmaz;
  bff1.beamZRms(z01, sigmaz1);
  bff2.beamZRms(z02, sigmaz2);

  sigmaz = (sigmaz2 > sigmaz1) ? sigmaz2 : sigmaz1;
  return default_zcut(sigmaz);
}



namespace {
std::string gui_json_string(const std::string& text)
{
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text)
    {
      if (c == '"' || c == '\\') out << '\\' << c;
      else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned>(c)
                           << std::dec;
      else out << c;
    }
  out << '"';
  return out.str();
}
}

void FEEBL::write_gui_snapshot(int diagonal)
{
  if (!get_make_gui_files()) return;
  const int last = 2 * grid_.get_n_cell_z() - 2;
  const int stride = std::max(1, (last + 62) / 63);
  if (diagonal != last && diagonal % stride != 0) return;
  if (!gui_diagonals_.empty() && diagonal <= gui_diagonals_.back())
    throw std::runtime_error("GUI snapshot diagonal is not increasing");
  if (gui_diagonals_.size() >= 64)
    throw std::runtime_error("GUI snapshot frame bound exceeded");
  // Deterministic thinning over the entire bunch. No RNG, deposits, or
  // physics arrays are modified; dump_beam copies each selected particle.
  // Keep the donor sampled branch: its stride=1 reconstruction has a different
  // slice boundary convention and has not been reconciled with this adapter.
  const int s1 = std::max(2, (beam1_.numberOfParticles() + 4095) / 4096);
  const int s2 = std::max(2, (beam2_.numberOfParticles() + 4095) / 4096);
  const std::string suffix = std::to_string(gui_diagonals_.size());
  beam1_.dump_beam("gui_b1." + suffix, diagonal, s1, grid_.get_timestep(),
                  grid_.get_step(), grid_.get_max_z());
  beam2_.dump_beam("gui_b2." + suffix, diagonal, s2, grid_.get_timestep(),
                  grid_.get_step(), grid_.get_max_z());
  const int interval_start = gui_diagonals_.empty() ? 0 : gui_diagonals_.back()+1;
  gui_diagnostic_frames_.push_back(gui_diagnostics_payload(
      interval_start, diagonal, gui_diagonals_.empty() ? NULL : &gui_previous_results_));
  gui_previous_results_ = grid_.get_results();
  grid_.get_results().reset_gui_interval();
  secondaries_.reset_gui_interval();
  const std::string field_frame=gui_field_pending_diagonal_==diagonal?
      gui_field_pending_:
      "{\"available\":false,\"reason\":\"No admitted native host field sample for this frame/backend\",\"encounter\":null,\"maps\":null}";
  gui_field_frames_.push_back(field_frame.substr(0,field_frame.size()-1)+
      ",\"longitudinal\":"+gui_longitudinal_payload(diagonal)+
      ",\"snapshot\":"+gui_snapshot_fields_payload(diagonal)+"}");
  gui_secondary_frames_.push_back(gui_secondary_frame_payload(diagonal));
  gui_imported_frames_.push_back(gui_imported_current());
  gui_diagonals_.push_back(diagonal);
  gui_strides1_.push_back(s1); gui_strides2_.push_back(s2);
}


std::string FEEBL::gui_secondary_frame_payload(int diagonal) const
{
  std::string reason;
  if (!switches.get_track_pairs()) reason="Secondary tracking is disabled for this run";
  else if (switches.get_collision_schedule()!="legacy" ||
           parallel_backend().actual_backend()!="serial" || switches.get_wavefront_parallel_physics())
    reason="Secondary frame snapshots require the committed serial legacy tracker";
  else if (secondaries_.n_particles()<0 ||
           static_cast<std::size_t>(secondaries_.n_particles())>feebl::secondary_frames::max_records)
    reason="Secondary frame snapshots exceed the bounded one-million-record population";
  std::ostringstream out;out << std::setprecision(17);
  if (!reason.empty()) {
    out << "{\"id\":" << gui_diagonals_.size() << ",\"diagonal\":" << diagonal
        << ",\"available\":false,\"reason\":" << gui_json_string(reason)
        << ",\"nrecords\":0,\"joint\":null,\"joint_folded\":null,\"marginals\":null,\"selection\":null,\"position_z\":null}";
  } else {
    const auto snapshot=feebl::secondary_frames::PositionSnapshot::capture(
        secondaries_.completed_tracking_snapshot(),static_cast<std::size_t>(secondaries_.n_particles()));
    snapshot.write_frame(out,int(gui_diagonals_.size()),diagonal);
  }
  return out.str();
}

void FEEBL::write_gui_secondary_frames() const
{
  if (gui_secondary_frames_.size()!=gui_diagonals_.size() || gui_secondary_frames_.size()>64)
    throw std::logic_error("secondary snapshot frame IDs do not match GUI frames");
  std::ostringstream out;out << std::setprecision(17);
  out << "{\"schema_version\":\"feebl.collider.secondary-frames/1\","
      << "\"phase\":\"post-diagonal-tracker-state\",\"coordinate_frame\":\"native_pair_tracking\","
      << "\"weighting\":\"one per current stored tracked secondary record; no luminosity or pair-ratio reweighting\","
      << "\"position_z_definition\":";
  feebl::secondary_frames::write_position_z_definition(out);
  out << ",\"definitions\":{"
      << "\"selection\":\"All admitted tracked secondary particles so far, including escaped particles, exactly once from authoritative reserve after active-particle deactivation\","
      << "\"timing\":\"Current state after the final substep of each saved collision diagonal; not interval production, and not a sum over frames\","
      << "\"normalization\":\"One count per stored tracker record after native pair_ratio and pair_ecut admission; no physical or luminosity weighting\","
      << "\"coordinates\":\"theta=atan2(hypot(vx,vy),vz); folded theta=atan2(hypot(vx,vy),abs(vz)); pT=abs(E)*hypot(vx,vy). Native axes, not reconciled detector arms\","
      << "\"precision\":\"Direct native particle state; final pairs.dat uses six significant digits, so its derived bin assignments and moments can differ by serialization rounding\"},"
      << "\"evaluated_flags\":{\"do_pairs\":" << switches.get_do_pairs()
      << ",\"load_events\":" << switches.get_load_event() << ",\"track_pairs\":" << switches.get_track_pairs()
      << ",\"store_pairs\":" << switches.get_store_pairs() << ",\"pair_ratio\":" << switches.get_pair_ratio()
      << ",\"pair_ecut\":" << switches.get_pair_ecut() << ",\"beam_pair\":" << switches.get_beam_pair()
      << ",\"do_comp\":" << switches.get_do_compt() << ",\"do_prod\":" << switches.get_do_prod()
      << ",\"do_bhabhas\":" << switches.get_do_bhabhas() << "},\"axes\":";
  feebl::secondary_frames::write_axes(out);out << ",\"frames\":[";
  for(std::size_t i=0;i<gui_secondary_frames_.size();++i){if(i)out << ',';out << gui_secondary_frames_[i];}
  out << "]}\n";const std::string bytes=out.str();
  if(bytes.size()>feebl::secondary_frames::max_bytes)
    throw std::length_error("secondary frame artifact exceeds 16 MiB bound");
  std::ofstream file("gui_secondary_frames.json");
  file << bytes;if(!file)throw std::runtime_error("Cannot complete native secondary frame export");
}


std::string FEEBL::gui_imported_unavailable_reason() const
{
  if(!switches.get_load_event())return "No imported event population was enabled";
  if(!switches.get_track_pairs())return "Imported event tracking is disabled";
  if(switches.get_do_pairs() || switches.get_beam_pair() || switches.get_do_compt() ||
     switches.get_do_prod() || switches.get_do_bhabhas())
    return "Imported identity export requires a pure imported secondary population";
  if(switches.get_collision_schedule()!="legacy" || parallel_backend().actual_backend()!="serial" || switches.get_wavefront_parallel_physics() ||
     switches.get_cpu_primary_move() || switches.get_gpu2_primary_mover()!="serial")
    return "Imported identity export requires the committed serial legacy tracker with enabled diagnostic hooks";
  if(secondaries_.imported_registry().overflow())return "Imported identity diagnostic exceeds 8192 tracked records";
  return std::string();
}

std::string FEEBL::gui_imported_current() const
{
  if(!gui_imported_unavailable_reason().empty())return "null";
  std::ostringstream out;out << std::setprecision(17);
  secondaries_.imported_registry().write_current(out,secondaries_.completed_tracking_snapshot());
  return out.str();
}

void FEEBL::write_gui_imported_trajectories() const
{
  if(gui_imported_frames_.size()!=gui_diagonals_.size() || gui_diagonals_.size()>64)
    throw std::logic_error("imported identity frames do not match GUI frame IDs");
  const std::string reason=gui_imported_unavailable_reason();
  std::ostringstream out;out << std::setprecision(17);
  out << "{\"schema_version\":\"feebl.collider.imported-trajectories/1\",\"available\":" << (reason.empty()?"true":"false")
      << ",\"reason\":" << (reason.empty()?"null":gui_json_string(reason))
      << ",\"phase\":\"post-diagonal-tracker-state\",\"coordinate_frame\":\"native_pair_tracking\","
      << "\"identity_definition\":\"Native admitted label bound at ingestion to one-based event.ini record ordinal; no charge or output-row inference\","
      << "\"precision\":\"Initial and current values are the actual native float particle state promoted to double for JSON\","
      << "\"field_gather_definition\":\"Cumulative actual field_pair calls: main mesh, outer mesh, analytic monopole fallback; escaped particles can move without a field call\","
      << "\"source_file\":\"event.ini\",\"evaluated_flags\":{\"load_events\":" << switches.get_load_event()
      << ",\"track_pairs\":" << switches.get_track_pairs() << ",\"do_pairs\":" << switches.get_do_pairs()
      << ",\"beam_pair\":" << switches.get_beam_pair() << ",\"do_comp\":" << switches.get_do_compt()
      << ",\"do_prod\":" << switches.get_do_prod() << ",\"do_bhabhas\":" << switches.get_do_bhabhas()
      << ",\"pair_ratio\":" << switches.get_pair_ratio() << ",\"pair_ecut\":" << switches.get_pair_ecut()
      << ",\"charge_sign\":" << switches.get_charge_sign() << "},"
      << "\"initial_columns\":[\"label\",\"source_record\",\"scheduled_time_counter\",\"admitted_time_counter\",\"signed_energy_GeV\",\"x_nm\",\"y_nm\",\"z_nm\",\"vx\",\"vy\",\"vz\"],"
      << "\"current_columns\":[\"label\",\"vx\",\"vy\",\"vz\",\"main_mesh_gathers\",\"outer_mesh_gathers\",\"monopole_gathers\"],\"initial_records\":";
  if(reason.empty())secondaries_.imported_registry().write_initial(out);else out << "null";
  out << ",\"frames\":[";
  for(std::size_t i=0;i<gui_diagonals_.size();++i){if(i)out << ',';
    out << "{\"id\":" << i << ",\"diagonal\":" << gui_diagonals_[i] << ",\"records\":"
        << (reason.empty()?gui_imported_frames_[i]:"null") << '}';}
  out << "],\"final\":{\"records\":" << gui_imported_current() << "}}\n";
  const auto bytes=out.str();if(bytes.size()>feebl::imported_trajectories::max_bytes)
    throw std::length_error("imported trajectory artifact exceeds 32 MiB bound");
  std::ofstream file("gui_imported_trajectories.json");file << bytes;
  if(!file)throw std::runtime_error("Cannot complete imported trajectory export");
}

void FEEBL::write_gui_manifest() const
{
  write_gui_diagnostics();
  write_gui_fields();
  write_gui_mesh_coverage();
  write_gui_secondary_frames();
  write_gui_imported_trajectories();
  std::ofstream f("gui_manifest.raw.json");
  if (!f) throw std::runtime_error("Cannot write GUI raw manifest");
  f << std::setprecision(17);
  f << "{\n  \"schema_version\": \"feebl.collider.raw/1\",\n";
  f << "  \"acc_file\": " << gui_json_string(get_acc_filename()) << ",\n";
  f << "  \"backend_requested\": " << gui_json_string(switches.get_compute_backend()) << ",\n";
  f << "  \"backend_actual_receipt\": " << gui_json_string(parallel_backend().provenance()) << ",\n";
  const std::string checkpoint = generator_.loaded_checkpoint_checksum();
  f << "  \"rng_checkpoint\": {\"loaded\":" << (checkpoint.empty() ? "false" : "true")
    << ",\"format\":" << (checkpoint.empty() ? "null" : "\"FEEBL_RNG_V2\"")
    << ",\"payload_checksum_fnv1a64\":" << (checkpoint.empty() ? "null" : gui_json_string(checkpoint))
    << ",\"precedence\":" << (checkpoint.empty() ? "\"seed_only\"" : "\"checkpoint_overrides_seed\"")
    << "},\n";
  f << "  \"gpu_mode\": " << switches.get_gpu_mode() << ",\n";
  f << "  \"n_cell_z\": " << grid_.get_n_cell_z() << ",\n";
  f << "  \"mesh_coverage_file\": \"gui_mesh_coverage.json\",\n";
  f << "  \"secondary_frames_file\": \"gui_secondary_frames.json\",\n";
  f << "  \"imported_trajectories_file\": \"gui_imported_trajectories.json\",\n";
  f << "  \"energy_GeV\": [" << beam_parameters1_.ebeam() << "," << beam_parameters2_.ebeam() << "],\n";
  f << "  \"native_beam_angles_x_rad\": [" << beam_parameters1_.x_angle() << "," << beam_parameters2_.x_angle() << "],\n";
  f << "  \"native_beam_angles_y_rad\": [" << beam_parameters1_.y_angle() << "," << beam_parameters2_.y_angle() << "],\n";
  f << "  \"espread\": [" << switches.get_espread1() << "," << switches.get_espread2() << "],\n";
  f << "  \"ip_dispersion_m\": [" << switches.get_ip_dispersion(1) << "," << switches.get_ip_dispersion(2) << "],\n";
  f << "  \"crab_mode\": [" << switches.get_crab_mode(1) << "," << switches.get_crab_mode(2) << "],\n";
  f << "  \"waist_mode\": [" << switches.get_waist_mode(1) << "," << switches.get_waist_mode(2) << "],\n";
  const bool field_geometry_available=grid_.get_n_cell_x()>=4 && grid_.get_n_cell_y()>=4;
  f << "  \"field_overlay_availability\":{\"computed\":"
    << (gui_field_any_available_?"true":"false") << ",\"file\":"
    << (field_geometry_available?"\"gui_fields.json\"":"null") << ",\"reason\":";
  if (gui_field_any_available_) f << "null";
  else f << gui_json_string(field_geometry_available?
      "No admitted native field maps for this solver/backend; inspect field frame reasons":
      "Native grid axes below four; optional field file omitted");
  f << "},\n";
  f << "  \"frames\": [\n";
  for (std::size_t i = 0; i < gui_diagonals_.size(); ++i)
    {
      if (i) f << ",\n";
      // Elapsed physical time since diagonal 0, not an invented IP time.
      const double t = gui_diagonals_[i] * static_cast<double>(grid_.get_timestep()) * grid_.get_step() * 1e-9 / 299792458.0;
      f << "    {\"id\":" << i << ",\"diagonal\":" << gui_diagonals_[i]
        << ",\"elapsed_s\":" << t << ",\"beam1\":\"gui_b1." << i
        << "\",\"beam2\":\"gui_b2." << i << "\",\"sample_stride\":["
        << gui_strides1_[i] << "," << gui_strides2_[i] << "]}";
    }
  f << "\n  ]\n}\n";
  if (!f) throw std::runtime_error("Incomplete GUI raw manifest write");
  std::cout << "FeeBL: bounded collider snapshots written; run tools/finalize_gui_dataset.py to authenticate the dataset" << std::endl;
}


namespace {
void gui_write_number(std::ostream& out,double value)
{
  if (!std::isfinite(value))
    throw std::domain_error("non-finite native collider diagnostic export");
  out << value;
}
void gui_write_beam_moments(std::ostream& out,
                            const feebl::diagnostics::BeamMoments& moments)
{
  if (!(moments.weight()>0.0)) { out << "null"; return; }
  if (!moments.finite()) throw std::domain_error("non-finite beam moments");
  static const char* keys[5]={"energy_GeV","x_nm","y_nm","vx","vy"};
  out << "{\"weight_sum\":"; gui_write_number(out,moments.weight());
  for (std::size_t i=0;i<5;++i) {
    out << ",\"" << keys[i] << "\":{\"mean\":";
    gui_write_number(out,moments.values[i].mean());
    out << ",\"rms\":"; gui_write_number(out,moments.values[i].rms());
    out << '}';
  }
  out << '}';
}
void gui_write_two_beam_moments(std::ostream& out,
    const feebl::diagnostics::CollisionMoments& moments,const char* weighting)
{
  out << "{\"weighting\":" << gui_json_string(weighting) << ",\"beam1\":";
  gui_write_beam_moments(out,moments.beam1);
  out << ",\"beam2\":"; gui_write_beam_moments(out,moments.beam2); out << '}';
}
feebl::diagnostics::BeamMoments gui_all_primary_moments(const BEAM& beam)
{
  feebl::diagnostics::BeamMoments result;
  for (int slice=0;slice<beam.number_of_slices();++slice)
    for (const PARTICLE* p:beam.getParticleVector(slice)) {
      PHI_FLOAT x,y; double vx,vy;
      p->XYposition(x,y); p->velocities(vx,vy);
      result.add(1.0,p->energy(),x,y,vx,vy);
    }
  return result;
}
void gui_write_histogram(std::ostream& out,
    const feebl::diagnostics::Histogram& h,const char* quantity,
    const char* unit,const char* weight_unit,const char* weighting)
{
  if (!h.finite()) throw std::domain_error("non-finite native histogram");
  out << "{\"quantity\":" << gui_json_string(quantity)
      << ",\"unit\":" << gui_json_string(unit)
      << ",\"weight_unit\":" << gui_json_string(weight_unit)
      << ",\"weighting\":" << gui_json_string(weighting)
      << ",\"normalization\":\"bin-integrated\",\"samples\":" << h.samples
      << ",\"bins\":[";
  double total=h.underflow+h.overflow;
  for (std::size_t i=0;i<h.bins;++i) {
    if (i) out << ',';
    if (h.weights[i]<0.0) throw std::domain_error("negative histogram interval");
    out << "{\"low\":" << h.low(i) << ",\"high\":" << h.low(i+1)
        << ",\"weight\":" << h.weights[i] << '}';
    total+=h.weights[i];
  }
  out << "],\"underflow\":" << h.underflow << ",\"overflow\":" << h.overflow
      << ",\"total_weight\":"; gui_write_number(out,total); out << '}';
}
void gui_write_luminosity(std::ostream& out,const pairmoments::SpectrumW& h,
                          const pairmoments::SpectrumW* previous,double nominal_w)
{
  if (!h.configured()) { out << "null"; return; }
  if (!h.finite() || (previous && !previous->finite()))
    throw std::domain_error("non-finite native luminosity histogram");
  if (previous && (previous->bins()!=h.bins() ||
      previous->lower_edge()!=h.lower_edge() || previous->upper_edge()!=h.upper_edge()))
    throw std::logic_error("native luminosity histogram changed geometry");
  // Keep native peak resolution where the bounded viewer budget permits it;
  // only adjacent tail bins are grouped first. The full native file is intact.
  const std::vector<feebl::gui_spectrum::BinRange> ranges =
      feebl::gui_spectrum::partition(h.bins(),h.lower_edge(),h.upper_edge(),
                                    0.96*nominal_w,1.02*nominal_w);
  out << "{\"quantity\":\"W\",\"unit\":\"GeV\","
      << "\"weight_unit\":\"m^-2 per BX\","
      << "\"weighting\":\"native collision pair luminosity\","
      << "\"normalization\":\"bin-integrated\",\"bins\":[";
  const double under=h.underflow()-(previous?previous->underflow():0.0);
  const double over=h.overflow()-(previous?previous->overflow():0.0);
  double total=under+over;
  for (std::size_t range=0;range<ranges.size();++range) {
    const std::size_t first=ranges[range].first;
    const std::size_t end=ranges[range].end;
    double weight=0.0;
    for (std::size_t i=first;i<end;++i)
      weight+=h.bin_weight(i)-(previous?previous->bin_weight(i):0.0);
    if (weight<0.0) throw std::domain_error("negative luminosity interval");
    if (first) out << ',';
    out << "{\"low\":" << h.bin_low(first) << ",\"high\":" << h.bin_high(end-1)
        << ",\"weight\":" << weight << '}';
    total+=weight;
  }
  out << "],\"underflow\":" << under << ",\"overflow\":" << over
      << ",\"total_weight\":"; gui_write_number(out,total); out << '}';
}
}

std::string FEEBL::gui_diagnostics_payload(int first,int last,
    const RESULTS* previous,bool integrated) const
{
  const RESULTS& current=grid_.get_results();
  const bool hooks=current.gui_diagnostics_enabled();
  std::ostringstream out; out << std::setprecision(17);
  out << "{\"diagonal_start\":" << first << ",\"diagonal_end\":" << last
      << ",\"luminosity_spectrum\":";
  const double nominal_w=2.0*std::sqrt((double)beam_parameters1_.ebeam()*
                                     (double)beam_parameters2_.ebeam());
  gui_write_luminosity(out,current.pair_spectrum(),
                       previous?&previous->pair_spectrum():NULL,nominal_w);
  const double sw=(!integrated && hooks)?current.gui_interval_sw():
      current.get_lumi_weight_sum()-(previous?previous->get_lumi_weight_sum():0.0);
  const double sh1=(!integrated && hooks)?current.gui_interval_h1():
      current.get_lumi_helicity_sum1()-(previous?previous->get_lumi_helicity_sum1():0.0);
  const double sh2=(!integrated && hooks)?current.gui_interval_h2():
      current.get_lumi_helicity_sum2()-(previous?previous->get_lumi_helicity_sum2():0.0);
  out << ",\"polarization\":";
  if (sw>0.0) {
    out << "{\"weighting\":\"native collision pair luminosity\",\"weight_sum\":" << sw
        << ",\"beam1\":";
    gui_write_number(out,sh1/sw);
    out << ",\"beam2\":";
    gui_write_number(out,sh2/sw);
    const bool interval=!integrated && hooks;
    const auto write_correlation=[&](const char* key,bool axis) {
      const pairmoments::PairHelicity& a=axis?current.axis_correlation(interval):
          current.helicity_correlation(interval);
      const pairmoments::PairHelicity* b=(!interval && previous)?
          (axis?&previous->axis_correlation():&previous->helicity_correlation()):nullptr;
      const double bw=b?b->weight():0.0;
      const double weight=a.weight()-bw;
      out << ",\"" << key << "\":";
      if(!(weight>0.0)){out << "null";return;}
      const double product=(a.weight()*a.mean_product()-(b?bw*b->mean_product():0.0))/weight;
      out << "{\"weight_sum\":";gui_write_number(out,weight);
      out << ",\"mean_product\":";gui_write_number(out,product);
      out << ",\"beam1_mean\":";
      gui_write_number(out,(a.weight()*a.h1().mean()-(b?bw*b->h1().mean():0.0))/weight);
      out << ",\"beam2_mean\":";
      gui_write_number(out,(a.weight()*a.h2().mean()-(b?bw*b->h2().mean():0.0))/weight);
      out << '}';
    };
    write_correlation("local_helicity",false);
    write_correlation("design_axis_projection",true);
    out << '}';
  } else out << "null";
  const feebl::diagnostics::CollisionMoments& collision=integrated?
      current.gui_collision_moments():current.gui_interval_collision_moments();
  // Coverage follows the actual enumerator, not a subtraction of very large
  // cumulative weights that could erase a tiny late-frame contribution.
  const bool moment_coverage=hooks &&
      (switches.get_electron_lumi_precompute()==0 ||
       switches.get_electron_lumi_spatial_kernel()==1) &&
      collision.beam1.weight()>0.0 && collision.beam2.weight()>0.0;
  out << ",\"beam_moments\":";
  if (moment_coverage)
    gui_write_two_beam_moments(out,collision,"native collision pair luminosity");
  else out << "null";
  out << ",\"peak\":";
  const bool peak_coverage=hooks &&
      (switches.get_electron_lumi_precompute()==0 ||
       switches.get_electron_lumi_spatial_kernel()==1);
  (integrated?current.gui_peak():current.gui_interval_peak()).write(out,peak_coverage);
  feebl::diagnostics::CollisionMoments bunch;
  bunch.beam1=gui_all_primary_moments(beam1_); bunch.beam2=gui_all_primary_moments(beam2_);
  out << ",\"bunch_moments\":";
  gui_write_two_beam_moments(out,bunch,"all stored native primary macroparticles at diagonal barrier; no inactive-slice drift synchronization");
  const BunchPolarization p1=longitudinal_polarization(beam1_);
  const BunchPolarization p2=longitudinal_polarization(beam2_);
  out << ",\"bunch_polarization\":[";
  if(p1.valid) gui_write_number(out,p1.p); else out << "null";
  out << ','; if(p2.valid) gui_write_number(out,p2.p); else out << "null";
  out << "],\"distributions\":{\"chi_e\":";
  if (hooks) gui_write_histogram(out,integrated?current.gui_chi_e():current.gui_interval_chi_e(),
      "native primary kick chi_e","1","macroparticle-kick evaluations",
      "one per actual primary field kick; repeated across substeps; before radiation");
  else out << "null";
  const bool species_supported=feebl::diagnostics::canonical_primary_pdg(
      1,switches.get_charge_sign())==11;
  out << ",\"chi_e_electron\":";
  if(hooks && species_supported)
    gui_write_histogram(out,integrated?current.gui_chi_e_beam(1):current.gui_interval_chi_e_beam(1),
      "native primary electron kick chi_e","1","macroparticle-kick evaluations",
      "one per actual primary field kick; repeated across substeps; before radiation");
  else out << "null";
  out << ",\"chi_e_positron\":";
  if(hooks && species_supported)
    gui_write_histogram(out,integrated?current.gui_chi_e_beam(2):current.gui_interval_chi_e_beam(2),
      "native primary positron kick chi_e","1","macroparticle-kick evaluations",
      "one per actual primary field kick; repeated across substeps; before radiation");
  else out << "null";
  out << ",\"chi_gamma\":";
  if (hooks && switches.get_do_coherent())
    gui_write_histogram(out,integrated?current.gui_chi_gamma():current.gui_interval_chi_gamma(),
      "native coherent-path photon chi_gamma","1","macro-photon field evaluations",
      "one per positive-energy coherent-model photon field evaluation; repeated across substeps");
  else out << "null";
  out << ",\"photon_energy\":";
  if (hooks && switches.get_do_beamstrahlung())
    gui_write_histogram(out,integrated?current.gui_photon_energy():current.gui_interval_photon_energy(),
      "registered beamstrahlung photon energy","GeV","native emitted macro-photon records",
      "one per registerPhotons record, including primary/coherent emission; excludes tertiary photons");
  else out << "null";
  out << ",\"pair_energy\":";
  const feebl::diagnostics::Histogram& pairs=integrated?
      secondaries_.get_results()->gui_pair_energy():secondaries_.get_results()->gui_interval_pair_energy();
  if (hooks && switches.get_do_pairs())
    gui_write_histogram(out,pairs,
      "incoherent pair daughter energy","GeV","weighted particles per BX",
      "IPC generator wgt per daughter before storage energy cut and thinning; excludes Compton/coherent/trident");
  else out << "null";
  out << "}}";
  return out.str();
}

void FEEBL::write_gui_diagnostics() const
{
  if (gui_diagonals_.empty() || gui_diagnostic_frames_.size()!=gui_diagonals_.size())
    throw std::logic_error("native collider diagnostics have no matching frames");
  std::ofstream out("gui_diagnostics.json");
  if (!out) throw std::runtime_error("Cannot write native GUI diagnostics");
  out << std::setprecision(17);
  out << "{\"schema_version\":\"feebl.collider.diagnostics/1\","
      << "\"definitions\":{\"rms\":\"centered population standard deviation; no independent-seed uncertainty\","
      << "\"vx_vy\":\"native dimensionless transverse direction components, approximately radians at small angle\","
      << "\"polarization\":\"mean spin projection onto own native direction, beam1 +z and beam2 -z; spin magnitude retained\","
      << "\"time\":\"luminosity and emissions accumulated over inclusive diagonal interval; bunch moments use stored endpoint states, not drift-synchronized display copies\","
      << "\"coordinates\":\"native simulated frame; no design crossing-angle or detector-plane transformation\"},"
      << "\"nominal_polarization\":{\"convention\":\"effective configured native spin vectors for generated beams; loaded spins may differ; reference helicity uses design +/-z\","
      << "\"enabled\":" << ((beam1_.has_spin()&&beam2_.has_spin())?"true":"false");
  const TRIDVECTOR polar1=beam_parameters1_.get_polar(),polar2=beam_parameters2_.get_polar();
  out << ",\"beam1\":[" << polar1(0) << ',' << polar1(1) << ',' << polar1(2)
      << "],\"beam2\":[" << polar2(0) << ',' << polar2(1) << ',' << polar2(2)
      << "],\"reference_helicity\":[" << polar1(2) << ',' << -polar2(2) << "]},"
      << "\"primary_species\":{\"supported\":"
      << (feebl::diagnostics::canonical_primary_pdg(1,switches.get_charge_sign())==11?"true":"false")
      << ",\"charge_sign\":" << switches.get_charge_sign() << ",\"beam1_pdg\":";
  const bool species_supported=feebl::diagnostics::canonical_primary_pdg(1,switches.get_charge_sign())==11;
  if(species_supported)out << "11";else out << "null";
  out << ",\"beam2_pdg\":";if(species_supported)out << "-11";else out << "null";
  out << ",\"basis\":\"canonical e+e- primary beam assignment; not per-particle species measurement\"},"
      << "\"unavailable_reasons\":{"
      << "\"beam_moments\":\"Requires complete native collision pair moment coverage and nonzero luminosity; optimized precompute modes are not instrumented\","
      << "\"polarization\":\"Requires stored spin and positive collision luminosity weight\","
      << "\"chi_e\":\"Requires instrumented serial primary mover; no stale snapshot upsilon substitutes\","
      << "\"chi_e_electron\":\"Requires instrumented serial primary mover and exact charge_sign=-1 canonical e+e- assignment; other charge modes have no species inference\","
      << "\"chi_e_positron\":\"Requires instrumented serial primary mover and exact charge_sign=-1 canonical e+e- assignment; other charge modes have no species inference\","
      << "\"chi_gamma\":\"Computed only by native coherent photon path; ordinary photon propagation does not evaluate chi_gamma\","
      << "\"photon_energy\":\"Requires radiation enabled and instrumented serial physics; native emission records are not physical yield weights\","
      << "\"pair_energy\":\"Requires incoherent-pair generation and instrumented serial physics; excludes other pair mechanisms\"},"
      << "\"frames\":[";
  for (std::size_t i=0;i<gui_diagnostic_frames_.size();++i) {
    if (i) out << ',';
    out << "{\"id\":" << i << ',' << gui_diagnostic_frames_[i].substr(1);
  }
  out << "],\"total\":" << gui_diagnostics_payload(0,gui_diagonals_.back(),NULL,true) << "}\n";
  if (!out) throw std::runtime_error("Incomplete native GUI diagnostics write");
}


namespace {
void gui_write_field_array(std::ostream& out,const std::vector<double>& values) {
  out << '[';
  for(std::size_t i=0;i<values.size();++i) {if(i)out << ',';out << values[i];}
  out << ']';
}
std::string gui_field_strip(int i1,int i2,double dz,int substep,int counter,
    const std::string& reason,const feebl::collider_fields::CrossSections* sample) {
  const auto z=feebl::collider_fields::encounter_z(i1,i2,dz);
  std::ostringstream out;out << std::setprecision(17);
  out << "{\"beam1_slice\":" << i1 << ",\"beam2_slice\":" << i2
      << ",\"substep_index\":";
  if(substep<0)out << "null";else out << substep;
  out << ",\"encounter_counter\":";
  if(counter<0)out << "null";else out << counter;
  out << ",\"nominal_z_nm\":" << z.nominal_nm
      << ",\"nominal_z_bounds_nm\":[" << z.nominal_low_nm << ',' << z.nominal_high_nm
      << "],\"display_z_nm\":{\"beam1\":" << z.beam1_nm << ",\"beam2\":" << z.beam2_nm
      << "},\"display_z_bounds_nm\":{\"beam1\":[" << z.beam1_nm-z.half_width_nm
      << ',' << z.beam1_nm+z.half_width_nm << "],\"beam2\":["
      << z.beam2_nm-z.half_width_nm << ',' << z.beam2_nm+z.half_width_nm
      << "]},\"available\":" << (sample?"true":"false") << ",\"reason\":";
  if(sample)out << "null";else out << gui_json_string(reason);
  out << ",\"maps\":";
  if(!sample)out << "null";
  else {
    out << "{\"beam1\":{\"zx\":";gui_write_field_array(out,sample->beam1_zx);
    out << ",\"zy\":";gui_write_field_array(out,sample->beam1_zy);
    out << "},\"beam2\":{\"zx\":";gui_write_field_array(out,sample->beam2_zx);
    out << ",\"zy\":";gui_write_field_array(out,sample->beam2_zy);out << "}}";
  }
  out << '}';return out.str();
}
}

std::string FEEBL::gui_longitudinal_payload(int diagonal) const
{
  const int nz=grid_.get_n_cell_z();
  const int first=std::max(0,diagonal-nz+1),last=std::min(diagonal,nz-1);
  const bool admitted=feebl::collider_fields::longitudinal_axis_cap(nz)>0;
  bool available=false;
  if(admitted && gui_field_diagonal_==diagonal)
    for(bool v:gui_longitudinal_available_)available=available||v;
  std::ostringstream out;out << std::setprecision(17);
  out << "{\"available\":" << (available?"true":"false") << ",\"reason\":";
  if(available)out << "null";
  else out << gui_json_string(admitted?"No admitted final-substep encounter cross-sections for this frame/backend":
      "Longitudinal field export supports native n_z from 1 through 128");
  out << ",\"expected_encounters\":" << last-first+1 << ",\"strips\":[";
  if(admitted)for(int i=first;i<=last;++i) {
    if(i!=first)out << ',';
    if(gui_field_diagonal_==diagonal && std::size_t(i)<gui_longitudinal_strips_.size() &&
        !gui_longitudinal_strips_[i].empty())out << gui_longitudinal_strips_[i];
    else out << gui_field_strip(i,diagonal-i,grid_.get_delta_z(),-1,-1,
        "Encounter cross-section not captured by the admitted native host path",nullptr);
  }
  out << "]}";return out.str();
}

std::string FEEBL::gui_snapshot_fields_payload(int diagonal)
{
  const int nz=grid_.get_n_cell_z();
  const std::size_t cap=feebl::collider_fields::snapshot_axis_cap(nz);
  std::string reason;
  if(!cap)reason="Full-bunch field snapshots support native n_z from 1 through 128";
  else if(switches.get_collision_schedule()!="legacy" ||
      parallel_backend().actual_backend()!="serial" || switches.get_wavefront_parallel_physics())
    reason="Full-bunch field snapshots currently require the committed serial legacy schedule";
  else if(switches.get_adaptive_grid_mode()!=0)
    reason="Full-bunch field snapshots require fixed native geometry";
  else if(switches.get_integration_method()!=2 || switches.get_interpolation()!=2 ||
      switches.get_electron_distribution_rho()!=2)
    reason="Full-bunch field snapshots currently require native FFT solve and CIC deposit/gather";
  else if(switches.get_charge_sign()!=-1.0f || grid_.get_scal_step(0)!=1.0f ||
      grid_.get_scal_step(1)!=1.0f)
    reason="Full-bunch field snapshots require charge_sign=-1 and both scale_step values=1";
  else if(switches.get_ip_dispersion(1)!=0.0 || switches.get_ip_dispersion(2)!=0.0)
    reason="Full-bunch field snapshots cannot infer slice bands after IP dispersion changes stored z";
  else if(switches.get_do_coherent() || switches.get_do_trident())
    reason="Full-bunch field snapshots currently cover primary sources only; coherent/trident sources are not admitted";
  else if(switches.get_force_symmetric())
    reason="Full-bunch field snapshots do not replace sources by forced charge symmetry";
  else if(!feebl::collider_fields::SnapshotGrid::supported_geometry(grid_))
    reason="Full-bunch field snapshots require native axes >=4 and at most 1048576 transverse cells";
  else if(beam1_.number_of_slices()!=nz || beam2_.number_of_slices()!=nz)
    reason="Full-bunch field snapshots require source slicing matching the native grid";
  std::ostringstream out;out << std::setprecision(17);
  out << "{\"available\":" << (reason.empty()?"true":"false") << ",\"reason\":";
  if(reason.empty())out << "null";else out << gui_json_string(reason);
  out << ",\"diagonal\":" << diagonal
      << ",\"phase\":\"post-diagonal/display-synchronized\",\"expected_slices\":" << nz
      << ",\"sources\":";
  if(!reason.empty()) {out << "null}";return out.str();}
  if(!gui_snapshot_grid_)
    gui_snapshot_grid_.reset(new feebl::collider_fields::SnapshotGrid(grid_));
  // Both beams' same-index source slices can share one independent FFT call:
  // their potentials remain separate, with no target interaction or kick.
  std::vector<std::string> source1,source2;source1.reserve(nz);source2.reserve(nz);
  feebl::collider_fields::MeshSnapshotCoverage mesh_snapshot;
  mesh_snapshot.id=int(gui_diagonals_.size());mesh_snapshot.diagonal=diagonal;

  for(int slice=0;slice<nz;++slice) {
    const auto c1=gui_snapshot_grid_->deposit(1,beam1_.getParticleVector(slice),
        grid_.get_macro_charge(1),slice,diagonal,beam1_.sign_label());
    const auto c2=gui_snapshot_grid_->deposit(2,beam2_.getParticleVector(slice),
        grid_.get_macro_charge(2),slice,diagonal,beam2_.sign_label());
    mesh_snapshot.beam[0].add(feebl::collider_fields::snapshot_coverage(c1));
    mesh_snapshot.beam[1].add(feebl::collider_fields::snapshot_coverage(c2));
    const auto lines=gui_snapshot_grid_->solve(cap);
    for(int beam=1;beam<=2;++beam) {
      const auto z=feebl::collider_fields::snapshot_band(slice,diagonal,nz,
          grid_.get_delta_z(),beam==1?beam1_.sign_label():beam2_.sign_label());
      const auto& counts=beam==1?c1:c2;
      std::ostringstream strip;strip << std::setprecision(17);
      strip << "{\"slice\":" << slice << ",\"display_z_nm\":" << z.centre
          << ",\"display_z_bounds_nm\":[" << z.low << ',' << z.high
          << "],\"source_particles\":" << counts.source_particles
          << ",\"deposited_particles\":" << counts.deposited_particles
          << ",\"outside_particles\":" << counts.outside_particles << ",\"maps\":{\"zx\":";
      gui_write_field_array(strip,beam==1?lines.beam1_zx:lines.beam2_zx);
      strip << ",\"zy\":";gui_write_field_array(strip,beam==1?lines.beam1_zy:lines.beam2_zy);
      strip << "}}";(beam==1?source1:source2).push_back(strip.str());
    }
  }
  out << '{';
  for(int beam=1;beam<=2;++beam) {
    if(beam==2)out << ',';
    out << "\"beam" << beam << "\":{\"strips\":[";
    const auto& strips=beam==1?source1:source2;
    for(std::size_t i=0;i<strips.size();++i) {if(i)out << ',';out << strips[i];}
    out << "]}";
  }
  if(gui_mesh_coverage_reason().empty())gui_mesh_snapshot_frames_.push_back(mesh_snapshot);
  out << "}}";return out.str();
}

void FEEBL::capture_gui_fields(int beam1_slice,int beam2_slice,bool host_phi_valid)
{
  if (!get_make_gui_files() || switches.get_collision_schedule()!="legacy" ||
      parallel_backend().actual_backend()!="serial" ||
      switches.get_wavefront_parallel_physics() ||
      grid_.get_n_cell_x()<4 || grid_.get_n_cell_y()<4) return;
  const int diagonal=beam1_slice+beam2_slice;
  const int final_diagonal=2*grid_.get_n_cell_z()-2;
  const int stride=std::max(1,(final_diagonal+62)/63);
  if (diagonal!=final_diagonal && diagonal%stride!=0) return;
  const bool central=beam1_slice==diagonal/2 && beam2_slice==diagonal-diagonal/2;
  const std::size_t axis_cap=feebl::collider_fields::longitudinal_axis_cap(grid_.get_n_cell_z());
  if (gui_field_diagonal_!=diagonal) {
    gui_field_diagonal_=diagonal;gui_field_visits_=0;
    const std::size_t n=axis_cap?grid_.get_n_cell_z():0;
    gui_longitudinal_visits_.assign(n,0);
    gui_longitudinal_strips_.assign(n,std::string());
    gui_longitudinal_available_.assign(n,false);
  }
  if(!axis_cap && !central)return;
  int visits=0;
  if(axis_cap)visits=++gui_longitudinal_visits_.at(beam1_slice);
  else visits=++gui_field_visits_;
  if (visits<grid_.get_timestep()) return;
  if (visits!=grid_.get_timestep())
    throw std::logic_error("field encounter visited too many substeps");
  std::string unavailable;
  if (!host_phi_valid || parallel_backend().gpu_mode2_enabled())
    unavailable="Actual solver host potentials are unavailable for this backend";
  else if (switches.get_integration_method()!=2)
    unavailable="Field overlay currently admits the native FFT field solve only";
  else if (switches.get_interpolation()!=2)
    unavailable="Field overlay currently admits the native CIC gather only";
  else if (switches.get_charge_sign()!=-1.0f ||
           grid_.get_scal_step(0)!=1.0f || grid_.get_scal_step(1)!=1.0f)
    unavailable="Field overlay convention requires charge_sign=-1 and both scale_step values=1";
  if(axis_cap) {
    std::string longitudinal_unavailable=unavailable;
    if(longitudinal_unavailable.empty() &&
        (switches.get_ip_dispersion(1)!=0.0 || switches.get_ip_dispersion(2)!=0.0))
      longitudinal_unavailable="Longitudinal display mapping is unavailable after an IP dispersion map changes stored z without reslicing";
    if(longitudinal_unavailable.empty()) {
      const auto sample=feebl::collider_fields::sample_cross_sections(grid_,axis_cap);
      gui_longitudinal_strips_[beam1_slice]=gui_field_strip(beam1_slice,beam2_slice,
          grid_.get_delta_z(),visits-1,time_counter_-1,std::string(),&sample);
      gui_longitudinal_available_[beam1_slice]=true;
    } else gui_longitudinal_strips_[beam1_slice]=gui_field_strip(beam1_slice,beam2_slice,
          grid_.get_delta_z(),visits-1,time_counter_-1,longitudinal_unavailable,nullptr);
  }
  if(!central)return;
  std::ostringstream out; out << std::setprecision(17);
  if (!unavailable.empty()) {
    out << "{\"available\":false,\"reason\":" << gui_json_string(unavailable)
        << ",\"encounter\":null,\"maps\":null}";
  } else {
    const feebl::collider_fields::PlaneSample sample=
        feebl::collider_fields::sample_main_grid(grid_);
    gui_field_any_available_=true;
    out << "{\"available\":true,\"reason\":null,\"encounter\":{"
        << "\"beam1_slice\":" << beam1_slice << ",\"beam2_slice\":" << beam2_slice
        << ",\"substep_index\":" << visits-1
        << ",\"encounter_counter\":" << time_counter_-1
        << ",\"nominal_z_nm\":" << 0.5*(beam2_slice-beam1_slice)*grid_.get_delta_z()
        << ",\"phase\":\"post-solve/pre-kick\"},\"maps\":{\"beam1\":";
    gui_write_field_array(out,sample.beam1);out << ",\"beam2\":";
    gui_write_field_array(out,sample.beam2);out << "}}";
  }
  gui_field_pending_diagonal_=diagonal;
  gui_field_pending_=out.str();
}


std::string FEEBL::gui_mesh_coverage_reason() const
{
  const int nz=grid_.get_n_cell_z(),nt=grid_.get_timestep();
  if(nz<1 || nz>128 || nt<1 || std::uint64_t(nz)*nz*nt>32768)
    return "Coverage supports 1..128 slices and at most 32768 actual encounters";
  if(switches.get_collision_schedule()!="legacy" || parallel_backend().actual_backend()!="serial" ||
      switches.get_wavefront_parallel_physics() || switches.get_cpu_primary_move() ||
      switches.get_adaptive_grid_mode()!=0 || parallel_backend().gpu_mode2_enabled())
    return "Coverage requires fixed-grid serial legacy deposition and primary mover";
  if(switches.get_integration_method()!=2 || switches.get_interpolation()!=2 ||
      switches.get_electron_distribution_rho()!=2)
    return "Coverage requires native FFT solve and CIC deposit/gather";
  if(switches.get_charge_sign()!=-1.0f || switches.get_force_symmetric() ||
      switches.get_do_coherent() || switches.get_do_trident() || switches.get_do_prod()!=0)
    return "Coverage requires unmodified primary e+e- source deposition without coherent/trident/production additions";
  if(grid_.get_scal_step(0)!=1 || grid_.get_scal_step(1)!=1 ||
      switches.get_ip_dispersion(1)!=0 || switches.get_ip_dispersion(2)!=0 ||
      !feebl::collider_fields::SnapshotGrid::supported_geometry(grid_))
    return "Coverage requires admitted full-snapshot geometry and display mapping";
  return {};
}

void FEEBL::write_gui_mesh_coverage() const
{
  const std::string reason=gui_mesh_coverage_reason();
  const int nz=grid_.get_n_cell_z(),nt=grid_.get_timestep();
  if(reason.empty()) {
    if(gui_mesh_encounters_.size()!=std::size_t(nz)*nz*nt ||
        gui_mesh_snapshot_frames_.size()!=gui_diagonals_.size())
      throw std::logic_error("incomplete native production/snapshot mesh coverage");
    for(int visits:gui_mesh_slice_visits_)
      if(visits!=nt)throw std::logic_error("native mesh coverage omitted a slice substep");
  }
  std::ofstream out("gui_mesh_coverage.json");
  if(!out)throw std::runtime_error("Cannot write native mesh coverage");
  out << std::setprecision(17)
      << "{\"schema_version\":\"feebl.collider.mesh-coverage/1\",\"available\":"
      << (reason.empty()?"true":"false") << ",\"reason\":";
  if(reason.empty())out << "null";else out << gui_json_string(reason);
  out << ",\"definitions\":{\"production\":\"actual native main-grid deposition counter differences, cross-checked against current primary sources and live rho charge\","
      << "\"gather\":\"native safe main-grid gather predicate immediately before serial primary mover; exact source XY order and values unchanged since deposition\","
      << "\"snapshot\":\"all primary source copies drift-synchronized to the displayed frame; no thinning\","
      << "\"scope\":\"primary e+e- beam sources; produced incoherent pairs are tracked test particles and are not deposited field sources\","
      << "\"charge_roundoff_model\":\"u32+(1+u32)*gamma_(8N+32), u32=epsilon(float)/2, gamma_k=k*epsilon(PHI_FLOAT)/2/(1-k*epsilon(PHI_FLOAT)/2); N=deposited sources; zero sources require zero charge\"},\"geometry\":{\"n_x\":" << grid_.get_n_cell_x()
      << ",\"n_y\":" << grid_.get_n_cell_y() << ",\"n_z\":" << nz << ",\"n_t\":" << nt
      << ",\"delta_x_nm\":" << grid_.get_delta_x() << ",\"delta_y_nm\":" << grid_.get_delta_y()
      << ",\"delta_z_nm\":" << grid_.get_delta_z() << ",\"main_gather_bounds_nm\":["
      << grid_.get_min_x() << ',' << grid_.get_max_x() << ',' << grid_.get_min_y() << ',' << grid_.get_max_y()
      << "]},\"encounters\":[";
  std::array<feebl::collider_fields::CoverageStats,2> totals;
  for(std::size_t i=0;i<gui_mesh_encounters_.size();++i) {
    const auto& e=gui_mesh_encounters_[i];if(i)out << ',';
    out << "{\"id\":" << e.id << ",\"diagonal\":" << e.diagonal << ",\"beam1_slice\":" << e.beam1_slice
        << ",\"beam2_slice\":" << e.beam2_slice << ",\"substep_index\":" << e.substep_index
        << ",\"source_coordinates_unchanged_before_kick\":true";
    for(int beam=0;beam<2;++beam) {
      const auto& c=e.beam[beam];totals[beam].add(c.stats);
      out << ",\"beam" << beam+1 << "\":{";
      feebl::collider_fields::write_coverage_members(out,c.stats);
      out << ",\"macro_charge\":" << c.macro_charge << ",\"native_rho_charge\":" << c.native_rho_charge
          << ",\"expected_rho_charge\":" << c.expected_rho_charge
          << ",\"charge_relative_error\":" << c.charge_relative_error
          << ",\"charge_relative_error_bound\":" << c.charge_relative_error_bound << '}';
    }
    out << '}';
  }
  out << "],\"summary\":{\"encounters\":" << gui_mesh_encounters_.size();
  for(int beam=0;beam<2;++beam) {out << ",\"beam" << beam+1 << "\":{";
    feebl::collider_fields::write_coverage_members(out,totals[beam]);out << '}';}
  out << "},\"snapshot_frames\":[";
  for(std::size_t i=0;i<gui_mesh_snapshot_frames_.size();++i) {
    const auto& f=gui_mesh_snapshot_frames_[i];if(i)out << ',';
    out << "{\"id\":" << f.id << ",\"diagonal\":" << f.diagonal;
    for(int beam=0;beam<2;++beam) {out << ",\"beam" << beam+1 << "\":{";
      feebl::collider_fields::write_coverage_members(out,f.beam[beam]);out << '}';}
    out << '}';
  }
  out << "]}\n";
  if(!out || out.tellp()>std::streamoff(32*1024*1024))
    throw std::runtime_error("Native mesh coverage exceeds its 32 MiB byte bound");
}

void FEEBL::write_gui_fields() const
{
  if (grid_.get_n_cell_x()<4 || grid_.get_n_cell_y()<4) {
    std::cout << "FeeBL: field overlay unavailable: native grid axes must be at least 4; gui_fields.json omitted" << std::endl;
    return;
  }
  if (gui_field_frames_.size()!=gui_diagonals_.size() || gui_diagonals_.empty())
    throw std::logic_error("field overlay frame identity mismatch");
  const feebl::collider_fields::PlaneSample g=
      feebl::collider_fields::main_grid_geometry(grid_);
  std::ofstream out("gui_fields.json");
  if (!out) throw std::runtime_error("Cannot write native field overlay");
  out << std::setprecision(17);
  out << "{\"schema_version\":\"feebl.collider.fields/1\","
      << "\"plane\":{\"axes\":[\"x\",\"y\"],\"coordinate_unit\":\"nm\","
      << "\"frame\":\"native simulated frame\"},\"geometry\":{"
      << "\"nx\":" << g.nx << ",\"ny\":" << g.ny
      << ",\"x_min_nm\":" << g.x_min_nm << ",\"x_max_nm\":" << g.x_max_nm
      << ",\"y_min_nm\":" << g.y_min_nm << ",\"y_max_nm\":" << g.y_max_nm
      << ",\"native_domain_nm\":[" << grid_.get_min_x() << ',' << grid_.get_max_x()
      << ',' << grid_.get_min_y() << ',' << grid_.get_max_y() << ']'
      << ",\"storage_order\":\"row-major-y-x\","
      << "\"sampling\":\"uniform CIC gather points; bounds are sample centres\"},"
      << "\"sources\":[{\"id\":\"beam1\",\"kind\":\"beam_beam\","
      << "\"label\":\"Source beam 1 acting on beam 2\","
      << "\"quantity\":\"native transverse electric-equivalent field magnitude\","
      << "\"unit\":\"GV/nm\",\"source_beam\":1,\"target_beam\":2},"
      << "{\"id\":\"beam2\",\"kind\":\"beam_beam\","
      << "\"label\":\"Source beam 2 acting on beam 1\","
      << "\"quantity\":\"native transverse electric-equivalent field magnitude\","
      << "\"unit\":\"GV/nm\",\"source_beam\":2,\"target_beam\":1}],"
      << "\"conventions\":{\"selection\":\"central active slice encounter, final substep, pre-kick\","
      << "\"scope\":\"single source slice; not whole-bunch or diagonal-averaged field\","
      << "\"timing\":\"elapsed_s is the displayed end-diagonal timestamp; field timing is the exact encounter ordinal and pre-kick phase; no physical field sample time is inferred\","
      << "\"z\":\"nominal_z_nm is native encounter-cell centre; no longitudinal field reconstruction\","
      << "\"normalization\":\"ElectricFieldCIC includes one-half of differentiated native potential; primary kick is delta_v=-2*E_CIC*step/E_particle; native kernel embeds charge_sign\","
      << "\"charge_sign\":" << switches.get_charge_sign()
      << ",\"scale_step\":[" << grid_.get_scal_step(0) << ',' << grid_.get_scal_step(1)
      << "],\"native_grid\":[" << grid_.get_n_cell_x() << ',' << grid_.get_n_cell_y()
      << "],\"n_z\":" << grid_.get_n_cell_z() << ",\"n_t\":" << grid_.get_timestep()
      << "},\"longitudinal\":{\"schema_version\":\"feebl.collider.longitudinal/1\","
      << "\"coordinate_unit\":\"nm\",\"frame\":\"legacy_collider_dump\","
      << "\"n_z\":" << grid_.get_n_cell_z() << ",\"delta_z_nm\":" << grid_.get_delta_z()
      << ",\"selection\":\"all active source-slice encounters, final substep, pre-kick\","
      << "\"timing\":\"field samples are post-solve/pre-kick on the final substep; particles are displayed after the diagonal barrier\","
      << "\"z_mapping\":\"source-specific nominal slice bands under the legacy particle-dump z convention; raw encounter z is retained separately\","
      << "\"planes\":{";
  const std::size_t longitudinal_cap=feebl::collider_fields::longitudinal_axis_cap(grid_.get_n_cell_z());
  const auto lines=feebl::collider_fields::main_grid_geometry(grid_,longitudinal_cap?longitudinal_cap:2);
  const auto snapshot_cap=feebl::collider_fields::snapshot_axis_cap(grid_.get_n_cell_z());
  const auto snapshot_lines=feebl::collider_fields::main_grid_geometry(grid_,snapshot_cap?snapshot_cap:2);
  out << "\"zx\":{\"axes\":[\"z\",\"x\"],\"fixed_axis\":\"y\",\"fixed_coordinate_nm\":0,"
      << "\"samples\":" << lines.nx << ",\"min_nm\":" << lines.x_min_nm << ",\"max_nm\":" << lines.x_max_nm
      << "},\"zy\":{\"axes\":[\"z\",\"y\"],\"fixed_axis\":\"x\",\"fixed_coordinate_nm\":0,"
      << "\"samples\":" << lines.ny << ",\"min_nm\":" << lines.y_min_nm << ",\"max_nm\":" << lines.y_max_nm
      << "}}},\"snapshot\":{\"schema_version\":\"feebl.collider.snapshot-fields/1\","
      << "\"coordinate_unit\":\"nm\",\"frame\":\"legacy_collider_dump\",\"n_z\":" << grid_.get_n_cell_z()
      << ",\"delta_z_nm\":" << grid_.get_delta_z()
      << ",\"selection\":\"all primary source slices at every saved diagonal\","
      << "\"timing\":\"post-diagonal; all source copies drift-synchronized with the sampled particle display\","
      << "\"model\":\"quasi-static native transverse slice-field reconstruction; primary sources only; not a full 3D electromagnetic field\","
      << "\"normalization\":\"same native macro charge, CIC deposition, Green kernel and half-gradient gather; no sum of source magnitudes\","
      << "\"domain\":\"native main transverse grid; excluded source particles counted per slice; no outside-grid field extrapolation\","
      << "\"planes\":{\"zx\":{\"axes\":[\"z\",\"x\"],\"fixed_axis\":\"y\",\"fixed_coordinate_nm\":0,"
      << "\"samples\":" << snapshot_lines.nx << ",\"min_nm\":" << snapshot_lines.x_min_nm << ",\"max_nm\":" << snapshot_lines.x_max_nm
      << "},\"zy\":{\"axes\":[\"z\",\"y\"],\"fixed_axis\":\"x\",\"fixed_coordinate_nm\":0,"
      << "\"samples\":" << snapshot_lines.ny << ",\"min_nm\":" << snapshot_lines.y_min_nm << ",\"max_nm\":" << snapshot_lines.y_max_nm
      << "}}},\"frames\":[";
  for (std::size_t i=0;i<gui_diagonals_.size();++i) {
    if(i) out << ',';
    const double elapsed=gui_diagonals_[i]*static_cast<double>(grid_.get_timestep())*
        grid_.get_step()*1e-9/299792458.0;
    out << "{\"id\":" << i << ",\"diagonal\":" << gui_diagonals_[i]
        << ",\"elapsed_s\":" << elapsed << ',' << gui_field_frames_[i].substr(1);
  }
  out << "]}\n";
  if (!out) throw std::runtime_error("Incomplete native field overlay write");
  if(out.tellp()>std::streamoff(64*1024*1024))
    throw std::runtime_error("Native field overlay exceeded the 64 MiB file bound");
}
