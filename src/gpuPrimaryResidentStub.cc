#include "gpuPrimaryResident.h"

namespace feebl {
namespace gpu {

ResidentPrimaryContract::ResidentPrimaryContract()
    : interpolation(2),
      extra_grids(0),
      generated_primaries(true),
      beamstrahlung(false),
      photon_emission(false),
      spin_transport(false),
      sokolov_ternov(false),
      coherent_secondaries(false),
      trident_secondaries(false),
      tracked_pairs(false),
      tracked_muons(false),
      full_electron_luminosity(false)
{
}

bool resident_primary_contract_supported(
    const ResidentPrimaryContract& contract,
    std::string& reason)
{
  reason.clear();
  if (contract.interpolation != 2)
    reason = "resident primary pipeline requires CIC interpolation=2";
  else if (contract.extra_grids != 0)
    reason = "resident primary pipeline requires extra_grids=0";
  else if (!contract.generated_primaries)
    reason = "resident primary pipeline currently requires generated primaries";
  else if (contract.beamstrahlung)
    reason = "resident primary beamstrahlung is not implemented";
  else if (contract.photon_emission)
    reason = "resident primary photon emission is not implemented";
  else if (contract.spin_transport)
    reason = "resident primary spin transport is not implemented";
  else if (contract.sokolov_ternov)
    reason = "resident primary Sokolov-Ternov transport is not implemented";
  else if (contract.coherent_secondaries)
    reason = "resident primary coherent secondaries are not implemented";
  else if (contract.trident_secondaries)
    reason = "resident primary trident secondaries are not implemented";
  else if (contract.tracked_pairs)
    reason = "resident primary pair tracking is not implemented";
  else if (contract.tracked_muons)
    reason = "resident primary muon tracking is not implemented";
  else if (contract.full_electron_luminosity)
    reason = "resident primary full electron-luminosity events and spectra are not implemented; only the raw fine-grid dot product is available";
  return reason.empty();
}

ResidentPrimaryOptions::ResidentPrimaryOptions()
    : device(0), pinned_io(false), memory_fraction(0.80), contract()
{
}

ResidentPrimaryEncounter::ResidentPrimaryEncounter()
    : wave_index(0),
      beam1_offset(0),
      beam1_count(0),
      beam2_offset(0),
      beam2_count(0),
      charge1(0.0),
      charge2(0.0),
      step(0.0f),
      scale1(1.0f),
      scale2(1.0f),
      charge_sign(-1.0f),
      four_way_symmetry(false)
{
}

ResidentPrimaryGrid::ResidentPrimaryGrid()
    : delta_x_inverse(0.0f),
      delta_y_inverse(0.0f),
      offset_x(0.0f),
      offset_y(0.0f),
      physical_min_x(0.0f),
      physical_max_x(0.0f),
      physical_min_y(0.0f),
      physical_max_y(0.0f)
{
}

ResidentPrimaryStats::ResidentPrimaryStats()
    : upload_calls(0),
      batch_calls(0),
      download_calls(0),
      waves(0),
      encounters(0),
      particle_updates(0),
      deposition_calls(0),
      field_solve_calls(0),
      gather_push_calls(0),
      deposit_beam1_in(0),
      deposit_beam1_out(0),
      deposit_beam2_in(0),
      deposit_beam2_out(0),
      gather_beam1_in(0),
      gather_beam1_out(0),
      gather_beam2_in(0),
      gather_beam2_out(0),
      h2d_bytes(0),
      d2h_bytes(0),
      h2d_operations(0),
      d2h_operations(0),
      synchronizations(0),
      setup_h2d_bytes(0),
      setup_h2d_operations(0),
      setup_synchronizations(0),
      intermediate_particle_h2d_bytes(0),
      intermediate_particle_d2h_bytes(0),
      host_field_materializations(0),
      nonfinite_state_events(0),
      state_epoch(0),
      particle_capacity1(0),
      particle_capacity2(0),
      particle_count1(0),
      particle_count2(0),
      resident_bytes(0),
      device_free_bytes_at_init(0),
      device_total_bytes(0),
      memory_budget_bytes(0),
      upload_ms(0.0),
      kernel_ms(0.0),
      download_ms(0.0),
      fine_lumi_dot_sum(0.0),
      state_resident(false),
      context_poisoned(false)
{
}

struct ResidentPrimaryPipeline::Impl
{
  ResidentPrimaryStats statistics;
};

ResidentPrimaryPipeline::ResidentPrimaryPipeline() : impl_(new Impl)
{
}

ResidentPrimaryPipeline::~ResidentPrimaryPipeline()
{
  delete impl_;
}

bool ResidentPrimaryPipeline::initialize(
    const ResidentPrimaryOptions& options,
    int,
    int,
    const double*,
    std::size_t,
    const ResidentPrimaryGrid&,
    float,
    std::size_t,
    std::size_t,
    std::string& error)
{
  if (!resident_primary_contract_supported(options.contract, error))
    return false;
  error = "resident primary pipeline is unavailable: FeeBL was built without CUDA/cuFFT support";
  return false;
}

bool ResidentPrimaryPipeline::is_available() const
{
  return false;
}

bool ResidentPrimaryPipeline::state_is_resident() const
{
  return false;
}

std::string ResidentPrimaryPipeline::device_description() const
{
  return "CUDA unavailable in this FeeBL build";
}

std::string ResidentPrimaryPipeline::runtime_description() const
{
  return "CUDA/cuFFT unavailable in this FeeBL build";
}

ResidentPrimaryStats ResidentPrimaryPipeline::stats() const
{
  return impl_->statistics;
}

bool ResidentPrimaryPipeline::upload(
    const float*, const float*, const float*, const float*, std::size_t,
    const float*, const float*, const float*, const float*, std::size_t,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  statistics = impl_->statistics;
  error = "resident primary pipeline is unavailable: FeeBL was built without CUDA/cuFFT support";
  return false;
}

bool ResidentPrimaryPipeline::run(
    const ResidentPrimaryGrid&,
    const ResidentPrimaryEncounter*,
    std::size_t,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  statistics = impl_->statistics;
  error = "resident primary pipeline is unavailable: FeeBL was built without CUDA/cuFFT support";
  return false;
}

bool ResidentPrimaryPipeline::download(
    float*, float*, float*, float*, std::size_t,
    float*, float*, float*, float*, std::size_t,
    ResidentPrimaryStats& statistics,
    std::string& error)
{
  statistics = impl_->statistics;
  error = "resident primary pipeline is unavailable: FeeBL was built without CUDA/cuFFT support";
  return false;
}

}  // namespace gpu
}  // namespace feebl
