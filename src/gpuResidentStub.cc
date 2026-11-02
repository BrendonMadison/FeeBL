#include "gpuResident.h"

namespace feebl {
namespace gpu {

ResidentFieldOptions::ResidentFieldOptions()
    : device(0), pinned_io(false), memory_fraction(0.80)
{
}

ResidentFieldStats::ResidentFieldStats()
    : calls(0),
      device_work_calls(0),
      deposition_calls(0),
      field_solve_calls(0),
      field_gather_calls(0),
      failed_device_calls(0),
      failure_drains(0),
      failure_drain_failures(0),
      setup_ms(0.0),
      h2d_ms(0.0),
      kernel_ms(0.0),
      d2h_ms(0.0),
      total_ms(0.0),
      last_h2d_ms(0.0),
      last_kernel_ms(0.0),
      last_d2h_ms(0.0),
      last_total_ms(0.0),
      last_device_work(false),
      setup_h2d_bytes(0),
      h2d_bytes(0),
      d2h_bytes(0),
      last_h2d_bytes(0),
      last_d2h_bytes(0),
      setup_h2d_operations(0),
      h2d_operations(0),
      d2h_operations(0),
      last_h2d_operations(0),
      last_d2h_operations(0),
      setup_synchronizations(0),
      synchronizations(0),
      last_synchronizations(0),
      device_allocations(0),
      pinned_host_allocations(0),
      plan_creations(0),
      resident_bytes(0),
      field_required_bytes(0),
      cufft_workspace_bytes(0),
      device_free_bytes_at_init(0),
      device_total_bytes(0),
      memory_budget_bytes(0),
      particle_capacity1(0),
      particle_capacity2(0),
      nx(0),
      ny(0),
      padded_nx(0),
      padded_ny(0),
      cuda_runtime_version(-1),
      cuda_driver_version(-1),
      cufft_runtime_version(-1),
      pinned_io(false),
      context_poisoned(false),
      memory_fraction(0.0)
{
}

ResidentDepositResult::ResidentDepositResult()
    : beam1_in(0),
      beam1_out(0),
      beam2_in(0),
      beam2_out(0),
      fine_lumi_dot(0.0)
{
}

ResidentFieldGatherResult::ResidentFieldGatherResult()
    : beam1_in(0), beam1_out(0), beam2_in(0), beam2_out(0)
{
}

struct ResidentFieldSolver::Impl
{
  ResidentFieldStats statistics;
};

ResidentFieldSolver::ResidentFieldSolver() : impl_(new Impl)
{
}

ResidentFieldSolver::~ResidentFieldSolver()
{
  delete impl_;
}

bool ResidentFieldSolver::initialize(const ResidentFieldOptions&,
                                     int,
                                     int,
                                     const double*,
                                     std::string& error)
{
  error = "GPU2 resident field solver is unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::initialize(int device,
                                     int nx,
                                     int ny,
                                     const double* dist_interleaved,
                                     bool pinned_io,
                                     std::string& error)
{
  ResidentFieldOptions options;
  options.device = device;
  options.pinned_io = pinned_io;
  return initialize(options, nx, ny, dist_interleaved, error);
}

bool ResidentFieldSolver::is_available() const
{
  return false;
}

std::string ResidentFieldSolver::device_description() const
{
  return "CUDA unavailable in this FeeBL build";
}

std::string ResidentFieldSolver::runtime_description() const
{
  return "CUDA/cuFFT unavailable in this FeeBL build";
}

ResidentFieldStats ResidentFieldSolver::stats() const
{
  return impl_->statistics;
}

bool ResidentFieldSolver::reserve_particle_capacity(std::size_t,
                                                    std::size_t,
                                                    std::string& error)
{
  error = "GPU2 resident field solver is unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::solve(const double*,
                                const double*,
                                float,
                                double*,
                                double*,
                                ResidentFieldStats& statistics,
                                std::string& error)
{
  statistics = impl_->statistics;
  error = "GPU2 resident field solver is unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::deposit_and_solve(const float*,
                                            std::size_t,
                                            double,
                                            const float*,
                                            std::size_t,
                                            double,
                                            float,
                                            float,
                                            float,
                                            float,
                                            bool,
                                            float,
                                            double*,
                                            double*,
                                            ResidentDepositResult& result,
                                            ResidentFieldStats& statistics,
                                            std::string& error)
{
  result = ResidentDepositResult();
  statistics = impl_->statistics;
  error = "GPU2 resident CIC/field solver is unavailable: FeeBL was built "
          "without CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::deposit(const float*,
                                  std::size_t,
                                  double,
                                  const float*,
                                  std::size_t,
                                  double,
                                  float,
                                  float,
                                  float,
                                  float,
                                  bool,
                                  ResidentDepositResult& result,
                                  ResidentFieldStats& statistics,
                                  std::string& error)
{
  result = ResidentDepositResult();
  statistics = impl_->statistics;
  error = "GPU2 resident CIC is unavailable: FeeBL was built without "
          "CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::solve_deposited(float,
                                          double*,
                                          double*,
                                          ResidentFieldStats& statistics,
                                          std::string& error)
{
  statistics = impl_->statistics;
  error = "GPU2 resident deposited-field solve is unavailable: FeeBL was "
          "built without CUDA/cuFFT support";
  return false;
}

bool ResidentFieldSolver::solve_deposited_and_gather(
    float,
    float,
    float,
    float,
    float,
    float*,
    std::size_t,
    float*,
    std::size_t,
    ResidentFieldGatherResult& result,
    ResidentFieldStats& statistics,
    std::string& error)
{
  result = ResidentFieldGatherResult();
  statistics = impl_->statistics;
  error = "GPU2 resident compact field gather is unavailable: FeeBL was "
          "built without CUDA/cuFFT support";
  return false;
}

}  // namespace gpu
}  // namespace feebl
