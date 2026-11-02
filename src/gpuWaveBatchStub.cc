#include "gpuWaveBatch.h"

namespace feebl {
namespace gpu {

WaveBatchEncounterOutput::WaveBatchEncounterOutput()
    : fields1(0), capacity1(0), fields2(0), capacity2(0),
      beam1_in(0), beam1_out(0), beam2_in(0), beam2_out(0),
      gather_beam1_in(0), gather_beam1_out(0),
      gather_beam2_in(0), gather_beam2_out(0), fine_lumi_dot(0.0) {}

WaveBatchOptions::WaveBatchOptions()
    : device(0), batch_width(0), maximum_particles1(0),
      maximum_particles2(0), memory_fraction(0.80), pinned_io(true) {}

WaveBatchStats::WaveBatchStats()
    : batch_calls(0), encounters(0), padded_fft_slots(0),
      h2d_operations(0), d2h_operations(0), h2d_bytes(0), d2h_bytes(0),
      synchronizations(0), kernel_launches(0), cufft_executes(0),
      setup_h2d_operations(0), setup_h2d_bytes(0),
      setup_synchronizations(0), plan_creations(0), device_allocations(0),
      pinned_host_allocations(0), deposit_beam1_in(0),
      deposit_beam1_out(0), deposit_beam2_in(0), deposit_beam2_out(0),
      gather_beam1_in(0), gather_beam1_out(0), gather_beam2_in(0),
      gather_beam2_out(0), setup_ms(0.0), pack_ms(0.0), h2d_ms(0.0),
      kernel_ms(0.0), d2h_ms(0.0), unpack_ms(0.0), boundary_ms(0.0),
      resident_bytes(0), cufft_workspace_bytes(0),
      device_free_bytes_at_init(0), device_total_bytes(0),
      memory_budget_bytes(0), batch_width(0), maximum_particles1(0),
      maximum_particles2(0), nx(0), ny(0), pinned_io(false),
      context_poisoned(false) {}

struct WaveBatchedFieldSolver::Impl {
  WaveBatchStats statistics;
};

WaveBatchedFieldSolver::WaveBatchedFieldSolver() : impl_(new Impl) {}
WaveBatchedFieldSolver::~WaveBatchedFieldSolver() { delete impl_; }

bool WaveBatchedFieldSolver::initialize(const WaveBatchOptions&,
                                        int,
                                        int,
                                        const double*,
                                        std::string& error) {
  error = "gpu_mode=4 wave-batched fields are unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool WaveBatchedFieldSolver::execute(
    const WaveBatchEncounterInput*, WaveBatchEncounterOutput*, std::size_t,
    double, double, float, float, float, float, float, float, float, float,
    bool, float, std::string& error) {
  error = "gpu_mode=4 wave-batched fields are unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool WaveBatchedFieldSolver::is_available() const { return false; }
std::string WaveBatchedFieldSolver::device_description() const {
  return "CUDA unavailable";
}
std::string WaveBatchedFieldSolver::runtime_description() const {
  return "CUDA/cuFFT unavailable";
}
WaveBatchStats WaveBatchedFieldSolver::stats() const {
  return impl_->statistics;
}

}  // namespace gpu
}  // namespace feebl
