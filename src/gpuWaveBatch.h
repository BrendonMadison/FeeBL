#ifndef FEEBL_GPU_WAVE_BATCH_SEEN
#define FEEBL_GPU_WAVE_BATCH_SEEN

#include <cstddef>
#include <cstdint>
#include <string>

namespace feebl {
namespace gpu {

// One host-side encounter in an EncounterDAG wave.  Positions are packed as
// x0,y0,x1,y1,... and remain owned by the caller for execute()'s duration.
struct WaveBatchEncounterInput {
  const float* xy1;
  std::size_t count1;
  const float* xy2;
  std::size_t count2;

  WaveBatchEncounterInput()
      : xy1(0), count1(0), xy2(0), count2(0) {}
};

// Compact host boundary for one encounter.  The field arrays are
// x0,y0,x1,y1,... and must have capacity for the corresponding input count.
struct WaveBatchEncounterOutput {
  float* fields1;
  std::size_t capacity1;
  float* fields2;
  std::size_t capacity2;
  std::size_t beam1_in;
  std::size_t beam1_out;
  std::size_t beam2_in;
  std::size_t beam2_out;
  std::size_t gather_beam1_in;
  std::size_t gather_beam1_out;
  std::size_t gather_beam2_in;
  std::size_t gather_beam2_out;
  double fine_lumi_dot;

  WaveBatchEncounterOutput();
};

struct WaveBatchOptions {
  int device;
  std::size_t batch_width;
  std::size_t maximum_particles1;
  std::size_t maximum_particles2;
  double memory_fraction;
  bool pinned_io;

  WaveBatchOptions();
};

// All counters describe observed work.  setup_* is kept separate from the
// collision path.  A synchronization is one batch completion boundary, never
// an encounter-level fence.  kernel_launches excludes the two cuFFT executes
// recorded separately in cufft_executes.
struct WaveBatchStats {
  std::uint64_t batch_calls;
  std::uint64_t encounters;
  std::uint64_t padded_fft_slots;
  std::uint64_t h2d_operations;
  std::uint64_t d2h_operations;
  std::uint64_t h2d_bytes;
  std::uint64_t d2h_bytes;
  std::uint64_t synchronizations;
  std::uint64_t kernel_launches;
  std::uint64_t cufft_executes;
  std::uint64_t setup_h2d_operations;
  std::uint64_t setup_h2d_bytes;
  std::uint64_t setup_synchronizations;
  std::uint64_t plan_creations;
  std::uint64_t device_allocations;
  std::uint64_t pinned_host_allocations;
  std::uint64_t deposit_beam1_in;
  std::uint64_t deposit_beam1_out;
  std::uint64_t deposit_beam2_in;
  std::uint64_t deposit_beam2_out;
  std::uint64_t gather_beam1_in;
  std::uint64_t gather_beam1_out;
  std::uint64_t gather_beam2_in;
  std::uint64_t gather_beam2_out;
  double setup_ms;
  double pack_ms;
  double h2d_ms;
  double kernel_ms;
  double d2h_ms;
  double unpack_ms;
  double boundary_ms;
  std::size_t resident_bytes;
  std::size_t cufft_workspace_bytes;
  std::size_t device_free_bytes_at_init;
  std::size_t device_total_bytes;
  std::size_t memory_budget_bytes;
  std::size_t batch_width;
  std::size_t maximum_particles1;
  std::size_t maximum_particles2;
  int nx;
  int ny;
  bool pinned_io;
  bool context_poisoned;

  WaveBatchStats();
};

// Fixed-capacity, reusable PlanMany implementation for gpu_mode=4.  Every
// execute() call packs up to batch_width independent encounters, performs a
// single asynchronous H2D/kernel/D2H submission on one stream, and waits once
// at the batch boundary.  No API can expose partially completed encounters.
class WaveBatchedFieldSolver {
 public:
  WaveBatchedFieldSolver();
  ~WaveBatchedFieldSolver();

  WaveBatchedFieldSolver(const WaveBatchedFieldSolver&) = delete;
  WaveBatchedFieldSolver& operator=(const WaveBatchedFieldSolver&) = delete;

  bool initialize(const WaveBatchOptions& options,
                  int nx,
                  int ny,
                  const double* dist_interleaved,
                  std::string& error);

  bool execute(const WaveBatchEncounterInput* inputs,
               WaveBatchEncounterOutput* outputs,
               std::size_t encounter_count,
               double charge1,
               double charge2,
               float delta_x_inverse,
               float delta_y_inverse,
               float offset_x,
               float offset_y,
               float physical_min_x,
               float physical_max_x,
               float physical_min_y,
               float physical_max_y,
               bool four_way_symmetry,
               float charge_sign,
               std::string& error);

  bool is_available() const;
  std::string device_description() const;
  std::string runtime_description() const;
  WaveBatchStats stats() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace gpu
}  // namespace feebl

#endif  // FEEBL_GPU_WAVE_BATCH_SEEN
