#ifndef FEEBL_GPU_PRIMARY_RESIDENT_SEEN
#define FEEBL_GPU_PRIMARY_RESIDENT_SEEN

#include <cstddef>
#include <cstdint>
#include <string>

namespace feebl {
namespace gpu {

// The first fully device-resident primary-particle contract is deliberately
// narrow.  Keeping these switches explicit makes production integration fail
// closed instead of silently running unsupported physics on an incomplete
// accelerator path.  The implemented subset is generated primary leptons,
// CIC interpolation, the main grid, orbit transport, and the raw fine-grid
// luminosity dot product.  Full pair/event luminosity is not represented by
// that dot product and is therefore a separate rejected capability.
struct ResidentPrimaryContract
{
  int interpolation;
  int extra_grids;
  bool generated_primaries;
  bool beamstrahlung;
  bool photon_emission;
  bool spin_transport;
  bool sokolov_ternov;
  bool coherent_secondaries;
  bool trident_secondaries;
  bool tracked_pairs;
  bool tracked_muons;
  bool full_electron_luminosity;

  ResidentPrimaryContract();
};

bool resident_primary_contract_supported(
    const ResidentPrimaryContract& contract,
    std::string& reason);

struct ResidentPrimaryOptions
{
  int device;
  bool pinned_io;
  double memory_fraction;
  ResidentPrimaryContract contract;

  ResidentPrimaryOptions();
};

// One pair of longitudinal slices in a collision wave.  Particle state is
// stored as two whole-bunch arrays; offsets/counts select the disjoint slices
// active in this encounter.  Encounters bearing the same wave_index must use
// disjoint ranges in both beams.  The implementation validates that invariant
// before enqueueing any work.
struct ResidentPrimaryEncounter
{
  std::uint32_t wave_index;
  std::size_t beam1_offset;
  std::size_t beam1_count;
  std::size_t beam2_offset;
  std::size_t beam2_count;
  double charge1;
  double charge2;
  float step;
  float scale1;
  float scale2;
  float charge_sign;
  bool four_way_symmetry;

  ResidentPrimaryEncounter();
};

struct ResidentPrimaryGrid
{
  float delta_x_inverse;
  float delta_y_inverse;
  float offset_x;
  float offset_y;
  float physical_min_x;
  float physical_max_x;
  float physical_min_y;
  float physical_max_y;

  ResidentPrimaryGrid();
};

// Cumulative, independently queryable proof of the actual host/device
// boundary.  Operational H2D/D2H counters cover explicit state upload,
// compact batch metrics, and explicit snapshots; the one-time Green-spectrum
// upload is reported separately as setup traffic.  A successful batch has
// exactly one explicit stream synchronization, no particle-state transfer,
// and no host field materialization regardless of encounter count.  Compact
// metrics are always copied through pinned memory.  If pinned_io=false, state
// upload/snapshot use pageable caller memory and the CUDA runtime may perform
// additional implicit staging/synchronization not represented by this count;
// production mode 3 therefore requires pinned_io=true.
struct ResidentPrimaryStats
{
  std::uint64_t upload_calls;
  std::uint64_t batch_calls;
  std::uint64_t download_calls;
  std::uint64_t waves;
  std::uint64_t encounters;
  std::uint64_t particle_updates;
  std::uint64_t deposition_calls;
  std::uint64_t field_solve_calls;
  std::uint64_t gather_push_calls;
  std::uint64_t deposit_beam1_in;
  std::uint64_t deposit_beam1_out;
  std::uint64_t deposit_beam2_in;
  std::uint64_t deposit_beam2_out;
  std::uint64_t gather_beam1_in;
  std::uint64_t gather_beam1_out;
  std::uint64_t gather_beam2_in;
  std::uint64_t gather_beam2_out;
  std::uint64_t h2d_bytes;
  std::uint64_t d2h_bytes;
  std::uint64_t h2d_operations;
  std::uint64_t d2h_operations;
  std::uint64_t synchronizations;
  std::uint64_t setup_h2d_bytes;
  std::uint64_t setup_h2d_operations;
  std::uint64_t setup_synchronizations;
  std::uint64_t intermediate_particle_h2d_bytes;
  std::uint64_t intermediate_particle_d2h_bytes;
  std::uint64_t host_field_materializations;
  std::uint64_t nonfinite_state_events;
  std::uint64_t state_epoch;
  std::size_t particle_capacity1;
  std::size_t particle_capacity2;
  std::size_t particle_count1;
  std::size_t particle_count2;
  std::size_t resident_bytes;
  std::size_t device_free_bytes_at_init;
  std::size_t device_total_bytes;
  std::size_t memory_budget_bytes;
  double upload_ms;
  double kernel_ms;
  double download_ms;
  double fine_lumi_dot_sum;
  bool state_resident;
  bool context_poisoned;

  ResidentPrimaryStats();
};

class ResidentPrimaryPipeline
{
 public:
  ResidentPrimaryPipeline();
  ~ResidentPrimaryPipeline();

  ResidentPrimaryPipeline(const ResidentPrimaryPipeline&) = delete;
  ResidentPrimaryPipeline& operator=(const ResidentPrimaryPipeline&) = delete;

  // dist_interleaved is FFT(kernel)/N in the same padded 2*nx by 2*ny,
  // interleaved FP64-complex representation consumed by ResidentFieldSolver.
  bool initialize(const ResidentPrimaryOptions& options,
                  int nx,
                  int ny,
                  const double* dist_interleaved,
                  std::size_t dist_interleaved_values,
                  const ResidentPrimaryGrid& spectrum_grid,
                  float spectrum_charge_sign,
                  std::size_t maximum_particles1,
                  std::size_t maximum_particles2,
                  std::string& error);

  bool is_available() const;
  bool state_is_resident() const;
  std::string device_description() const;
  std::string runtime_description() const;
  ResidentPrimaryStats stats() const;

  // xy and velocity_xy are interleaved [x0,y0,x1,y1,...].  energy and
  // upsilon are one value per particle.  Upsilon is mutable even when
  // radiation is disabled because PARTICLE::advanceDueToEBfield recomputes it
  // after every kick.  These are the only whole-state H2D transfers.
  bool upload(const float* xy1,
              const float* velocity_xy1,
              const float* energy1,
              const float* upsilon1,
              std::size_t particle_count1,
              const float* xy2,
              const float* velocity_xy2,
              const float* energy2,
              const float* upsilon2,
              std::size_t particle_count2,
              ResidentPrimaryStats& statistics,
              std::string& error);

  // Validate the entire schedule before work, enqueue every encounter in
  // wave order on one nonblocking stream, and synchronize only once after the
  // final compact metrics copy.  State remains resident on success.
  bool run(const ResidentPrimaryGrid& grid,
           const ResidentPrimaryEncounter* encounters,
           std::size_t encounter_count,
           ResidentPrimaryStats& statistics,
           std::string& error);

  // Explicit snapshot; it does not evict state.  Output capacities are in
  // particles, not floats.  These are the only whole-state D2H transfers.
  bool download(float* xy1,
                float* velocity_xy1,
                float* energy1,
                float* upsilon1,
                std::size_t output_capacity1,
                float* xy2,
                float* velocity_xy2,
                float* energy2,
                float* upsilon2,
                std::size_t output_capacity2,
                ResidentPrimaryStats& statistics,
                std::string& error);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace gpu
}  // namespace feebl

#endif
