#include "parallelBackendCPP.h"
#include "config.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef FEEBL_HAVE_OPENMP
#include <omp.h>
#endif

#ifndef FEEBL_BUILD_SOURCE_HASH
#define FEEBL_BUILD_SOURCE_HASH "unknown"
#endif

#ifndef FEEBL_CUDA_TOOLKIT_VERSION
#define FEEBL_CUDA_TOOLKIT_VERSION "none"
#endif

namespace
{

std::string lower_ascii(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

unsigned int available_threads()
{
  const unsigned int logical = std::thread::hardware_concurrency();
#ifdef __linux__
  // Prefer physical cores for automatic latency mode.  Explicit cpu_threads
  // can still request SMT.  The sysfs topology query naturally respects the
  // hardware exposed to a VM; affinity itself is recorded separately.
  std::set<std::string> physical_cores;
  std::vector<unsigned int> candidate_cpus;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
    {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &allowed))
          candidate_cpus.push_back(static_cast<unsigned int>(cpu));
    }
  if (candidate_cpus.empty())
    for (unsigned int cpu = 0; cpu < (logical == 0 ? 1U : logical); ++cpu)
      candidate_cpus.push_back(cpu);

  for (std::size_t index = 0; index < candidate_cpus.size(); ++index)
    {
      const unsigned int cpu = candidate_cpus[index];
      std::ostringstream core_path, package_path;
      core_path << "/sys/devices/system/cpu/cpu" << cpu << "/topology/core_id";
      package_path << "/sys/devices/system/cpu/cpu" << cpu
                   << "/topology/physical_package_id";
      std::ifstream core_file(core_path.str().c_str());
      std::ifstream package_file(package_path.str().c_str());
      std::string core, package;
      if (core_file >> core && package_file >> package)
        physical_cores.insert(package + ":" + core);
    }
  if (!physical_cores.empty())
    return static_cast<unsigned int>(physical_cores.size());
#endif
  return logical == 0 ? 1U : logical;
}

std::string affinity_description()
{
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    {
      std::ostringstream out;
      bool first = true;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask))
          {
            if (!first) out << ',';
            out << cpu;
            first = false;
          }
      return first ? "empty" : out.str();
    }
#endif
  return "unavailable";
}

} // namespace

class CPU_THREAD_POOL
{
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::function<void(std::size_t)> job_;
  PARALLEL_BACKEND::STATIC_RANGE_TASK range_job_;
  void* range_context_;
  std::exception_ptr error_;
  std::size_t generation_;
  std::size_t next_task_;
  std::size_t task_count_;
  std::size_t workers_pending_;
  std::size_t range_item_count_;
  std::size_t range_active_workers_;
  bool static_ranges_;
  bool job_active_;
  bool stopping_;

  CPU_THREAD_POOL(const CPU_THREAD_POOL&);
  CPU_THREAD_POOL& operator=(const CPU_THREAD_POOL&);

  void worker_loop(std::size_t worker_index)
  {
    std::size_t observed_generation = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;)
      {
        work_ready_.wait(lock, [this, observed_generation]() {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) return;
        observed_generation = generation_;

        if (static_ranges_)
          {
            const std::size_t workers = range_active_workers_;
            if (worker_index < workers)
              {
                const std::size_t quotient = range_item_count_ / workers;
                const std::size_t remainder = range_item_count_ % workers;
                const std::size_t begin = worker_index * quotient +
                    std::min(worker_index, remainder);
                const std::size_t count = quotient +
                    (worker_index < remainder);
                lock.unlock();
                try
                  {
                    if (count != 0)
                      range_job_(range_context_, worker_index, begin,
                                 begin + count);
                  }
                catch (...)
                  {
                    std::lock_guard<std::mutex> error_lock(mutex_);
                    if (!error_) error_ = std::current_exception();
                  }
                lock.lock();
              }
          }
        else
          {
            for (;;)
              {
                if (next_task_ >= task_count_) break;
                const std::size_t index = next_task_++;
                lock.unlock();
                try
                  {
                    job_(index);
                  }
                catch (...)
                  {
                    std::lock_guard<std::mutex> error_lock(mutex_);
                    if (!error_) error_ = std::current_exception();
                  }
                lock.lock();
              }
          }

        if (--workers_pending_ == 0)
          {
            job_active_ = false;
            work_done_.notify_one();
          }
      }
  }

 public:
  explicit CPU_THREAD_POOL(std::size_t thread_count)
      : range_job_(NULL), range_context_(NULL), generation_(0), next_task_(0),
        task_count_(0), workers_pending_(0), range_item_count_(0),
        range_active_workers_(0), static_ranges_(false), job_active_(false),
        stopping_(false)
  {
    if (thread_count == 0) thread_count = 1;
    workers_.reserve(thread_count);
    try
      {
        for (std::size_t i = 0; i < thread_count; ++i)
          workers_.push_back(
              std::thread(&CPU_THREAD_POOL::worker_loop, this, i));
      }
    catch (...)
      {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          stopping_ = true;
          ++generation_;
          work_ready_.notify_all();
        }
        for (std::size_t i = 0; i < workers_.size(); ++i)
          if (workers_[i].joinable()) workers_[i].join();
        throw;
      }
  }

  ~CPU_THREAD_POOL()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
      work_ready_.notify_all();
    }
    for (std::size_t i = 0; i < workers_.size(); ++i)
      workers_[i].join();
  }

  void run(std::size_t task_count,
           const std::function<void(std::size_t)>& task)
  {
    if (task_count == 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    if (job_active_)
      throw std::logic_error("nested CPU thread-pool dispatch is not supported");

    job_ = task;
    range_job_ = NULL;
    range_context_ = NULL;
    error_ = std::exception_ptr();
    next_task_ = 0;
    task_count_ = task_count;
    range_item_count_ = 0;
    static_ranges_ = false;
    workers_pending_ = workers_.size();
    job_active_ = true;
    ++generation_;
    work_ready_.notify_all();
    work_done_.wait(lock, [this]() { return !job_active_; });
    const std::exception_ptr error = error_;
    job_ = std::function<void(std::size_t)>();
    lock.unlock();
    if (error) std::rethrow_exception(error);
  }

  void run_static_ranges(std::size_t item_count,
                         void* context,
                         PARALLEL_BACKEND::STATIC_RANGE_TASK task)
  {
    run_static_ranges_limited(item_count, workers_.size(), context, task);
  }

  void run_static_ranges_limited(
      std::size_t item_count,
      std::size_t active_workers,
      void* context,
      PARALLEL_BACKEND::STATIC_RANGE_TASK task)
  {
    if (!task)
      throw std::invalid_argument("static-range task must not be null");
    if (active_workers == 0 || active_workers > workers_.size())
      throw std::invalid_argument(
          "static-range active worker count is outside the persistent pool");
    if (item_count == 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    if (job_active_)
      throw std::logic_error("nested CPU thread-pool dispatch is not supported");

    job_ = std::function<void(std::size_t)>();
    range_job_ = task;
    range_context_ = context;
    error_ = std::exception_ptr();
    next_task_ = 0;
    task_count_ = 0;
    range_item_count_ = item_count;
    range_active_workers_ = active_workers;
    static_ranges_ = true;
    workers_pending_ = workers_.size();
    job_active_ = true;
    ++generation_;
    work_ready_.notify_all();
    work_done_.wait(lock, [this]() { return !job_active_; });
    const std::exception_ptr error = error_;
    range_job_ = NULL;
    range_context_ = NULL;
    range_item_count_ = 0;
    range_active_workers_ = 0;
    static_ranges_ = false;
    lock.unlock();
    if (error) std::rethrow_exception(error);
  }
};

PARALLEL_BACKEND::PARALLEL_BACKEND()
    : requested_backend_("serial"), requested_cpu_variant_("auto"),
      requested_gpu_variant_("auto"), actual_engine_("legacy"),
      modern_provenance_active_(false), rng_backend_("legacy"),
      reduction_mode_("strict"), actual_backend_("serial"),
      actual_cpu_variant_("serial"), actual_gpu_variant_("serial"),
      requested_cpu_threads_(0),
      actual_cpu_threads_(1), gpu_device_(0), gpu_min_particles_(262144),
      gpu_calls_(0), gpu2_calls_(0), gpu3_calls_(0), gpu4_calls_(0),
      gpu5_calls_(0),
      cpu_primary_move_calls_(0),
      cpu_primary_move_particles_(0), gpu2_primary_mover_calls_(0),
      gpu2_primary_mover_particles_(0), gpu2_primary_mover_ranges_(0),
      gpu2_primary_mover_observed_threads_(0),
      gpu2_plan_committed_(false),
      allow_fallback_(false), configured_(false)
{
  // Zero means that no GPU execution mode is active.  The requested default
  // remains mode 1 for backward-compatible explicit GPU cards.
  actual_gpu2_.mode = 0;
}

PARALLEL_BACKEND::~PARALLEL_BACKEND() = default;

bool PARALLEL_BACKEND::openmp_available() const
{
#ifdef FEEBL_HAVE_OPENMP
  return true;
#else
  return false;
#endif
}

bool PARALLEL_BACKEND::gpu_available() const
{
#ifdef FEEBL_HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool PARALLEL_BACKEND::gpu_mode2_available() const
{
#ifdef FEEBL_HAVE_GPU2
  return true;
#else
  return false;
#endif
}

bool PARALLEL_BACKEND::gpu_mode3_available() const
{
  // Mode 3 is implemented by the same CUDA/cuFFT resident library as mode 2,
  // but has a separate whole-crossing contract and production selector.
#ifdef FEEBL_HAVE_GPU2
  return true;
#else
  return false;
#endif
}

bool PARALLEL_BACKEND::gpu_mode4_available() const
{
  // The wave-batched legacy-physics mode uses the CUDA/cuFFT resident
  // library, but has its own fail-closed selector and execution receipt.
#ifdef FEEBL_HAVE_GPU2
  return true;
#else
  return false;
#endif
}

bool PARALLEL_BACKEND::gpu_mode5_available() const
{
  // Adaptive class-batched fields have a distinct CUDA target and build
  // capability.  Do not infer it from mode 2/4 availability.
#ifdef FEEBL_HAVE_GPU5
  return true;
#else
  return false;
#endif
}

void PARALLEL_BACKEND::configure(const std::string& compute_backend,
                                 int cpu_threads,
                                 const std::string& cpu_variant,
                                 const std::string& gpu_variant,
                                 int gpu_device,
                                 int gpu_min_particles,
                                 const std::string& rng_backend,
                                 const std::string& reduction_mode,
                                 bool allow_backend_fallback)
{
  configure(compute_backend, cpu_threads, cpu_variant, gpu_variant,
            gpu_device, gpu_min_particles, rng_backend, reduction_mode,
            allow_backend_fallback, GPU2_OPTIONS(), MODERN_RUNTIME_OPTIONS());
}

void PARALLEL_BACKEND::configure(const std::string& compute_backend,
                                 int cpu_threads,
                                 const std::string& cpu_variant,
                                 const std::string& gpu_variant,
                                 int gpu_device,
                                 int gpu_min_particles,
                                 const std::string& rng_backend,
                                 const std::string& reduction_mode,
                                 bool allow_backend_fallback,
                                 const GPU2_OPTIONS& gpu2_options)
{
  configure(compute_backend, cpu_threads, cpu_variant, gpu_variant,
            gpu_device, gpu_min_particles, rng_backend, reduction_mode,
            allow_backend_fallback, gpu2_options, MODERN_RUNTIME_OPTIONS());
}

void PARALLEL_BACKEND::configure(const std::string& compute_backend,
                                 int cpu_threads,
                                 const std::string& cpu_variant,
                                 const std::string& gpu_variant,
                                 int gpu_device,
                                 int gpu_min_particles,
                                 const std::string& rng_backend,
                                 const std::string& reduction_mode,
                                 bool allow_backend_fallback,
                                 const GPU2_OPTIONS& gpu2_options,
                                 const MODERN_RUNTIME_OPTIONS& modern_options)
{
  if (configured_)
    throw std::logic_error("parallel backend configured more than once");
  if (cpu_threads < 0)
    throw std::invalid_argument("cpu_threads must be zero or positive");
  if (gpu_device < 0)
    throw std::invalid_argument("gpu_device must be zero or positive");
  if (gpu_min_particles < 0)
    throw std::invalid_argument("gpu_min_particles must be zero or positive");

  requested_backend_ = lower_ascii(compute_backend);
  requested_cpu_variant_ = lower_ascii(cpu_variant);
  requested_gpu_variant_ = lower_ascii(gpu_variant);
  requested_gpu2_ = gpu2_options;
  requested_gpu2_.scope = lower_ascii(requested_gpu2_.scope);
  requested_gpu2_.deposition = lower_ascii(requested_gpu2_.deposition);
  requested_gpu2_.fft = lower_ascii(requested_gpu2_.fft);
  requested_gpu2_.graphs = lower_ascii(requested_gpu2_.graphs);
  requested_gpu2_.fusion = lower_ascii(requested_gpu2_.fusion);
  requested_gpu2_.pinned_io = lower_ascii(requested_gpu2_.pinned_io);
  requested_gpu2_.primary_mover =
      lower_ascii(requested_gpu2_.primary_mover);
  requested_gpu2_.profile = lower_ascii(requested_gpu2_.profile);
  requested_modern_ = modern_options;
  requested_modern_.engine = lower_ascii(requested_modern_.engine);
  requested_modern_.algorithm = lower_ascii(requested_modern_.algorithm);
  requested_modern_.precision = lower_ascii(requested_modern_.precision);
  requested_modern_.reduction = lower_ascii(requested_modern_.reduction);
  requested_modern_.cpu_runtime = lower_ascii(requested_modern_.cpu_runtime);
  requested_modern_.cpu_affinity = lower_ascii(requested_modern_.cpu_affinity);
  requested_modern_.gpu_graphs = lower_ascii(requested_modern_.gpu_graphs);
  requested_modern_.trace = lower_ascii(requested_modern_.trace);
  const MODERN_RUNTIME_OPTIONS modern_defaults;
  modern_provenance_active_ =
      requested_modern_.engine != modern_defaults.engine ||
      requested_modern_.algorithm != modern_defaults.algorithm ||
      requested_modern_.precision != modern_defaults.precision ||
      requested_modern_.reduction != modern_defaults.reduction ||
      requested_modern_.batch_width != modern_defaults.batch_width ||
      requested_modern_.cpu_workers != modern_defaults.cpu_workers ||
      requested_modern_.cpu_runtime != modern_defaults.cpu_runtime ||
      requested_modern_.cpu_affinity != modern_defaults.cpu_affinity ||
      requested_modern_.gpu_streams != modern_defaults.gpu_streams ||
      requested_modern_.gpu_graphs != modern_defaults.gpu_graphs ||
      requested_modern_.trace != modern_defaults.trace;
  actual_engine_ = requested_modern_.engine == "legacy"
      ? "legacy" : "unavailable";
  rng_backend_ = lower_ascii(rng_backend);
  reduction_mode_ = lower_ascii(reduction_mode);
  requested_cpu_threads_ = cpu_threads;
  gpu_device_ = gpu_device;
  gpu_min_particles_ = static_cast<std::size_t>(gpu_min_particles);
  allow_fallback_ = allow_backend_fallback;
  fallback_reason_.clear();
  gpu_calls_ = 0;
  gpu2_calls_ = 0;
  gpu3_calls_ = 0;
  gpu4_calls_ = 0;
  gpu5_calls_ = 0;
  cpu_primary_move_calls_ = 0;
  cpu_primary_move_particles_ = 0;
  gpu2_primary_mover_calls_ = 0;
  gpu2_primary_mover_particles_ = 0;
  gpu2_primary_mover_ranges_ = 0;
  gpu2_primary_mover_observed_threads_ = 0;
  gpu2_plan_committed_ = false;
  gpu_device_description_.clear();
  actual_gpu2_ = GPU2_OPTIONS();
  actual_gpu2_.mode = 0;
  const auto append_fallback_reason =
      [this](const std::string& reason)
      {
        if (!fallback_reason_.empty()) fallback_reason_ += "; ";
        fallback_reason_ += reason;
      };

  if (requested_backend_ != "serial" && requested_backend_ != "cpu" &&
      requested_backend_ != "gpu" && requested_backend_ != "auto")
    throw std::invalid_argument("compute_backend must be serial, cpu, gpu, or auto");
  if (requested_modern_.engine != "legacy" &&
      requested_modern_.engine != "modern_ref_cpu" &&
      requested_modern_.engine != "modern_cpu" &&
      requested_modern_.engine != "modern_gpu")
    throw std::invalid_argument(
        "engine must be legacy, modern_ref_cpu, modern_cpu, or modern_gpu");
  if (requested_modern_.algorithm != "exact" &&
      requested_modern_.algorithm != "factored_exact" &&
      requested_modern_.algorithm != "sampled_validated" &&
      requested_modern_.algorithm != "adaptive_validated")
    throw std::invalid_argument(
        "algorithm must be exact, factored_exact, sampled_validated, or "
        "adaptive_validated");
  if (requested_modern_.precision != "fp64" &&
      requested_modern_.precision != "mixed_qualified")
    throw std::invalid_argument("precision must be fp64 or mixed_qualified");
  if (requested_modern_.reduction != "deterministic" &&
      requested_modern_.reduction != "fast_qualified")
    throw std::invalid_argument(
        "modern_reduction must be deterministic or fast_qualified");
  if (requested_modern_.batch_width < 0)
    throw std::invalid_argument("modern_batch_width must be zero or positive");
  if (requested_modern_.cpu_workers < 0)
    throw std::invalid_argument("modern_cpu_workers must be zero or positive");
  if (requested_modern_.cpu_runtime != "auto" &&
      requested_modern_.cpu_runtime != "pool" &&
      requested_modern_.cpu_runtime != "openmp")
    throw std::invalid_argument(
        "modern_cpu_runtime must be auto, pool, or openmp");
  if (requested_modern_.cpu_affinity != "auto" &&
      requested_modern_.cpu_affinity != "none" &&
      requested_modern_.cpu_affinity != "compact" &&
      requested_modern_.cpu_affinity != "scatter")
    throw std::invalid_argument(
        "modern_cpu_affinity must be auto, none, compact, or scatter");
  if (requested_modern_.gpu_streams < 0)
    throw std::invalid_argument("modern_gpu_streams must be zero or positive");
  if (requested_modern_.gpu_graphs != "auto" &&
      requested_modern_.gpu_graphs != "off" &&
      requested_modern_.gpu_graphs != "on")
    throw std::invalid_argument("modern_gpu_graphs must be auto, off, or on");
  if (requested_modern_.trace != "off" &&
      requested_modern_.trace != "summary" &&
      requested_modern_.trace != "full")
    throw std::invalid_argument("modern_trace must be off, summary, or full");
  if (requested_cpu_variant_ != "auto" &&
      requested_cpu_variant_ != "pool_private" &&
      requested_cpu_variant_ != "pool_dual" &&
      requested_cpu_variant_ != "omp_atomic")
    throw std::invalid_argument(
        "cpu_variant must be auto, pool_private, pool_dual, or omp_atomic");
  if (requested_gpu_variant_ == "tiled") requested_gpu_variant_ = "shared";
  if (requested_gpu_variant_ == "binning") requested_gpu_variant_ = "sorted";
  if (requested_gpu_variant_ != "auto" &&
      requested_gpu_variant_ != "atomic" &&
      requested_gpu_variant_ != "shared" &&
      requested_gpu_variant_ != "sorted")
    throw std::invalid_argument(
        "gpu_variant must be auto, atomic, shared (or tiled), or sorted");
  if (requested_gpu2_.mode != 1 && requested_gpu2_.mode != 2 &&
      requested_gpu2_.mode != 3 && requested_gpu2_.mode != 4 &&
      requested_gpu2_.mode != 5)
    throw std::invalid_argument(
        "gpu_mode must be 1, 2, production-limited modern mode 3, or "
        "wave-batched legacy-physics mode 4, or adaptive-grid mode 5");
  if (requested_gpu2_.scope != "auto" &&
      requested_gpu2_.scope != "fields" &&
      requested_gpu2_.scope != "primary" &&
      requested_gpu2_.scope != "full")
    throw std::invalid_argument(
        "gpu2_scope must be auto, fields, primary, or full");
  if (requested_gpu2_.deposition != "auto" &&
      requested_gpu2_.deposition != "atomic" &&
      requested_gpu2_.deposition != "shared_tile" &&
      requested_gpu2_.deposition != "sort_reduce")
    throw std::invalid_argument(
        "gpu2_deposition must be auto, atomic, shared_tile, or sort_reduce");
  if (requested_gpu2_.fft != "cufft")
    throw std::invalid_argument("gpu2_fft currently supports only cufft");
  if (requested_gpu2_.streams < 0)
    throw std::invalid_argument("gpu2_streams must be zero or positive");
  if (requested_gpu2_.graphs != "auto" && requested_gpu2_.graphs != "off" &&
      requested_gpu2_.graphs != "on")
    throw std::invalid_argument("gpu2_graphs must be auto, off, or on");
  if (requested_gpu2_.fusion != "auto" && requested_gpu2_.fusion != "off" &&
      requested_gpu2_.fusion != "on")
    throw std::invalid_argument("gpu2_fusion must be auto, off, or on");
  if (requested_gpu2_.batch_slices < 0)
    throw std::invalid_argument("gpu2_batch_slices must be zero or positive");
  if (requested_gpu2_.pinned_io != "auto" &&
      requested_gpu2_.pinned_io != "off" &&
      requested_gpu2_.pinned_io != "on")
    throw std::invalid_argument("gpu2_pinned_io must be auto, off, or on");
  if (requested_gpu2_.host_threads < 0)
    throw std::invalid_argument("gpu2_host_threads must be zero or positive");
  if (requested_gpu2_.primary_mover != "serial" &&
      requested_gpu2_.primary_mover != "openmp" &&
      requested_gpu2_.primary_mover != "pool" &&
      requested_gpu2_.primary_mover != "auto")
    throw std::invalid_argument(
        "gpu2_primary_mover must be serial, openmp, pool, or auto");
  if (!(requested_gpu2_.memory_fraction > 0.0 &&
        requested_gpu2_.memory_fraction <= 1.0))
    throw std::invalid_argument(
        "gpu2_memory_fraction must be greater than zero and at most one");
  if (requested_gpu2_.profile != "off" &&
      requested_gpu2_.profile != "summary" &&
      requested_gpu2_.profile != "nvtx")
    throw std::invalid_argument("gpu2_profile must be off, summary, or nvtx");
  if (rng_backend_ != "legacy" && rng_backend_ != "counter")
    throw std::invalid_argument("rng_backend must be legacy or counter");
  if (reduction_mode_ != "strict" && reduction_mode_ != "deterministic" &&
      reduction_mode_ != "fast")
    throw std::invalid_argument(
        "reduction_mode must be strict, deterministic, or fast");

  const bool modern_engine = requested_modern_.engine != "legacy";
  if (!modern_engine && modern_provenance_active_)
    throw std::invalid_argument(
        "algorithm, precision, and modern_* controls require a nonlegacy "
        "engine; engine=legacy accepts only their historical defaults");
  if (!modern_engine && requested_gpu2_.mode == 3)
    throw std::invalid_argument(
        "gpu_mode=3 requires engine=modern_gpu");
  if (modern_engine &&
      (requested_gpu2_.mode == 4 || requested_gpu2_.mode == 5))
    throw std::invalid_argument(
        "gpu_mode=4 and gpu_mode=5 are legacy-physics compositional modes "
        "and require engine=legacy");

  if (modern_engine)
    {
      // The modern engine selector owns collision execution.  The legacy
      // compute_backend remains "serial" because it still initializes the
      // beams, geometry, grid and outputs before the narrow mode-3 crossing.
      if (requested_backend_ != "serial")
        throw std::invalid_argument(
            "nonlegacy engine selection requires compute_backend=serial; "
            "compute_backend controls only the legacy physics engine");

      if (requested_modern_.engine == "modern_gpu")
        {
          // This first production mode is intentionally narrow and truthful:
          // legacy initialization consumes the historical RNG, while the
          // qualified no-radiation/no-luminosity collision consumes no RNG.
          // Requiring counter here would falsely claim that beam generation
          // had been ported to counter streams.
          if (rng_backend_ != "legacy")
            throw std::invalid_argument(
                "engine=modern_gpu gpu_mode=3 requires rng_backend=legacy: "
                "initialization remains legacy and the qualified collision "
                "uses no random numbers");
          if (allow_fallback_)
            throw std::invalid_argument(
                "engine=modern_gpu gpu_mode=3 is abort-only; "
                "allow_backend_fallback must be 0");
          if (requested_gpu2_.mode != 3)
            throw std::invalid_argument(
                "engine=modern_gpu requires gpu_mode=3");
          if (requested_modern_.algorithm != "exact" ||
              requested_modern_.precision != "mixed_qualified" ||
              requested_modern_.reduction != "fast_qualified")
            throw std::invalid_argument(
                "engine=modern_gpu gpu_mode=3 requires algorithm=exact, "
                "precision=mixed_qualified, and "
                "modern_reduction=fast_qualified; particles are FP32 and "
                "FP64 atomic CIC accumulation is unordered");
          if (requested_modern_.batch_width != 0 ||
              requested_modern_.cpu_workers != 0 ||
              requested_modern_.cpu_runtime != "auto" ||
              requested_modern_.cpu_affinity != "auto")
            throw std::invalid_argument(
                "gpu_mode=3 submits the complete EncounterDAG; "
                "modern_batch_width must be 0 and modern CPU controls must "
                "remain at their defaults");
          if (requested_modern_.gpu_streams != 1 ||
              requested_modern_.gpu_graphs != "off" ||
              requested_modern_.trace != "summary")
            throw std::invalid_argument(
                "gpu_mode=3 currently requires modern_gpu_streams=1, "
                "modern_gpu_graphs=off, and modern_trace=summary");
          if (requested_gpu_variant_ != "auto" ||
              gpu_min_particles_ != 262144)
            throw std::invalid_argument(
                "gpu_variant and gpu_min_particles are mode-1 controls and "
                "must remain at their defaults with gpu_mode=3");
          if (requested_gpu2_.scope != "primary" ||
              requested_gpu2_.deposition != "atomic" ||
              requested_gpu2_.fft != "cufft" ||
              requested_gpu2_.streams != 1 ||
              requested_gpu2_.graphs != "off" ||
              requested_gpu2_.fusion != "off" ||
              requested_gpu2_.batch_slices != 0 ||
              requested_gpu2_.pinned_io != "on" ||
              requested_gpu2_.host_threads != 1 ||
              requested_gpu2_.primary_mover != "serial" ||
              requested_gpu2_.profile != "summary")
            throw std::invalid_argument(
                "gpu_mode=3 requires gpu2_scope=primary, "
                "gpu2_deposition=atomic, gpu2_fft=cufft, gpu2_streams=1, "
                "gpu2_graphs=off, gpu2_fusion=off, gpu2_batch_slices=0, "
                "gpu2_pinned_io=on, gpu2_host_threads=1, "
                "gpu2_primary_mover=serial, and gpu2_profile=summary");
          if (!gpu_mode3_available())
            throw std::runtime_error(
                "engine=modern_gpu gpu_mode=3 requested, but this build has "
                "no resident CUDA primary pipeline; rebuild with "
                "-DFEEBL_ENABLE_CUDA=ON");

          actual_engine_ = "modern_gpu";
          actual_backend_ = "gpu";
          actual_cpu_variant_ = "serial";
          actual_gpu_variant_ = "resident_primary_whole_crossing";
          actual_cpu_threads_ = 1;
          actual_gpu2_ = requested_gpu2_;
          configured_ = true;
          return;
        }

      if (rng_backend_ != "counter")
        throw std::invalid_argument(
            "engine=" + requested_modern_.engine +
            " requires rng_backend=counter for schedule-independent streams");

      if (requested_modern_.engine == "modern_ref_cpu" &&
          (requested_modern_.algorithm != "exact" ||
           requested_modern_.precision != "fp64" ||
           requested_modern_.reduction != "deterministic"))
        throw std::invalid_argument(
            "engine=modern_ref_cpu requires algorithm=exact, precision=fp64, "
            "and modern_reduction=deterministic");

      if (requested_modern_.gpu_streams != 0 ||
          requested_modern_.gpu_graphs != "auto")
        throw std::invalid_argument(
            "modern_gpu_streams and modern_gpu_graphs require "
            "engine=modern_gpu");
      if (requested_gpu2_.mode == 3)
        throw std::invalid_argument(
            "gpu_mode=3 requires engine=modern_gpu");

      // Record requested=.../actual=unavailable before refusing.  This makes
      // unit-level provenance useful without ever claiming modern execution.
      throw std::runtime_error(
          "engine=" + requested_modern_.engine +
          " is parsed and provenance-ready, but no modern physics engine is "
          "wired; refusing before simulation (actual_engine=unavailable)");
    }

  if (rng_backend_ == "counter")
    throw std::runtime_error(
        "rng_backend=counter is a validated CPU/CUDA reference engine, but "
        "is not yet wired through every stochastic FeeBL process; refusing "
        "to mix RNG semantics silently");

  if (requested_gpu2_.mode == 4)
    {
      const bool serial_host_plan =
          requested_gpu2_.primary_mover == "serial" &&
          requested_gpu2_.host_threads == 1;
      const bool pool_host_plan =
          requested_gpu2_.primary_mover == "pool" &&
          requested_gpu2_.host_threads >= 2;
      if (requested_backend_ != "gpu")
        throw std::invalid_argument(
            "gpu_mode=4 requires compute_backend=gpu");
      if (allow_fallback_)
        throw std::invalid_argument(
            "gpu_mode=4 is abort-only; allow_backend_fallback must be 0");
      if (reduction_mode_ != "fast")
        throw std::invalid_argument(
            "gpu_mode=4 currently requires reduction_mode=fast because "
            "FP64 atomic CIC changes accumulation order");
      if (requested_gpu_variant_ != "auto" || gpu_min_particles_ != 262144)
        throw std::invalid_argument(
            "gpu_variant and gpu_min_particles are mode-1 controls and must "
            "remain at defaults with gpu_mode=4");
      if (requested_gpu2_.scope != "primary" ||
          requested_gpu2_.deposition != "atomic" ||
          requested_gpu2_.fft != "cufft" ||
          requested_gpu2_.streams != 1 ||
          requested_gpu2_.graphs != "off" ||
          requested_gpu2_.fusion != "on" ||
          requested_gpu2_.batch_slices < 1 ||
          requested_gpu2_.batch_slices > 256 ||
          requested_gpu2_.pinned_io != "on" ||
          (!serial_host_plan && !pool_host_plan) ||
          requested_gpu2_.profile != "summary")
        throw std::invalid_argument(
            "gpu_mode=4 requires gpu2_scope=primary, "
            "gpu2_deposition=atomic, gpu2_fft=cufft, gpu2_streams=1, "
            "gpu2_graphs=off, gpu2_fusion=on, gpu2_batch_slices=1..256, "
            "gpu2_pinned_io=on, and either gpu2_primary_mover=serial "
            "with gpu2_host_threads=1 or gpu2_primary_mover=pool with "
            "gpu2_host_threads>=2; gpu2_profile=summary");
      if (!gpu_mode4_available())
        throw std::runtime_error(
            "gpu_mode=4 requested, but this build has no wave-batched "
            "CUDA/cuFFT pipeline; rebuild with -DFEEBL_ENABLE_CUDA=ON");
      actual_engine_ = "legacy";
      actual_backend_ = "gpu";
      actual_cpu_variant_ = pool_host_plan
          ? "gpu4_pool_static_ranges" : "serial_ordered_host_physics";
      actual_gpu_variant_ = "wave_batched_planmany_fields";
      actual_cpu_threads_ = requested_gpu2_.host_threads;
      actual_gpu2_ = requested_gpu2_;
      configured_ = true;
      return;
    }

  if (requested_gpu2_.mode == 5)
    {
      if (requested_backend_ != "gpu")
        throw std::invalid_argument(
            "gpu_mode=5 requires compute_backend=gpu");
      if (allow_fallback_)
        throw std::invalid_argument(
            "gpu_mode=5 is abort-only; allow_backend_fallback must be 0");
      if (reduction_mode_ != "fast")
        throw std::invalid_argument(
            "gpu_mode=5 currently requires reduction_mode=fast because "
            "the qualified global-atomic FP64 CIC deposition is unordered");
      if (requested_gpu_variant_ != "auto" || gpu_min_particles_ != 262144)
        throw std::invalid_argument(
            "gpu_variant and gpu_min_particles are mode-1 controls and must "
            "remain at defaults with gpu_mode=5");
      if (requested_gpu2_.scope != "primary" ||
          requested_gpu2_.deposition != "atomic" ||
          requested_gpu2_.fft != "cufft" ||
          requested_gpu2_.streams != 1 ||
          requested_gpu2_.graphs != "off" ||
          requested_gpu2_.fusion != "on" ||
          requested_gpu2_.batch_slices < 1 ||
          requested_gpu2_.batch_slices > 256 ||
          requested_gpu2_.pinned_io != "on" ||
          requested_gpu2_.host_threads < 2 ||
          requested_gpu2_.primary_mover != "pool" ||
          requested_gpu2_.profile != "summary")
        throw std::invalid_argument(
            "gpu_mode=5 requires gpu2_scope=primary, "
            "gpu2_deposition=atomic, gpu2_fft=cufft, gpu2_streams=1, "
            "gpu2_graphs=off, gpu2_fusion=on, gpu2_batch_slices=1..256, "
            "gpu2_pinned_io=on, gpu2_host_threads>=2, "
            "gpu2_primary_mover=pool, and gpu2_profile=summary; "
            "shared_tile and sort_reduce are explicit unimplemented "
            "variants, not fallbacks");
      if (!gpu_mode5_available())
        throw std::runtime_error(
            "gpu_mode=5 requested, but this build has no adaptive "
            "CUDA/cuFFT class-batched executor; rebuild with "
            "-DFEEBL_ENABLE_CUDA=ON");
      actual_engine_ = "legacy";
      actual_backend_ = "gpu";
      actual_cpu_variant_ = "gpu5_pool_static_ranges_pending";
      actual_gpu_variant_ = "adaptive_class_planmany_fields";
      actual_cpu_threads_ = requested_gpu2_.host_threads;
      actual_gpu2_ = requested_gpu2_;
      configured_ = true;
      return;
    }

  const GPU2_OPTIONS gpu2_defaults;
  const bool nondefault_gpu2_controls =
      requested_gpu2_.scope != gpu2_defaults.scope ||
      requested_gpu2_.deposition != gpu2_defaults.deposition ||
      requested_gpu2_.fft != gpu2_defaults.fft ||
      requested_gpu2_.streams != gpu2_defaults.streams ||
      requested_gpu2_.graphs != gpu2_defaults.graphs ||
      requested_gpu2_.fusion != gpu2_defaults.fusion ||
      requested_gpu2_.batch_slices != gpu2_defaults.batch_slices ||
      requested_gpu2_.pinned_io != gpu2_defaults.pinned_io ||
      requested_gpu2_.host_threads != gpu2_defaults.host_threads ||
      requested_gpu2_.primary_mover != gpu2_defaults.primary_mover ||
      requested_gpu2_.memory_fraction != gpu2_defaults.memory_fraction ||
      requested_gpu2_.profile != gpu2_defaults.profile;
  if (requested_gpu2_.mode == 1 && nondefault_gpu2_controls)
    {
      if (!allow_fallback_)
        throw std::invalid_argument(
            "gpu2_* controls require compute_backend=gpu and gpu_mode=2");
      append_fallback_reason(
          "gpu2_* controls ignored because gpu_mode=1 was selected");
    }
  if (requested_gpu2_.mode == 2 && requested_backend_ != "gpu")
    {
      if (!allow_fallback_)
        throw std::invalid_argument(
            "gpu_mode=2 requires compute_backend=gpu");
      append_fallback_reason(
          "gpu_mode=2 ignored because compute_backend was not gpu");
    }
  if (requested_gpu2_.mode == 2 &&
      (requested_gpu_variant_ != "auto" || gpu_min_particles_ != 262144))
    {
      if (!allow_fallback_)
        throw std::invalid_argument(
            "gpu_variant and gpu_min_particles are GPU mode 1 controls; "
            "leave them at their defaults with gpu_mode=2");
      append_fallback_reason(
          "non-default GPU mode 1 controls ignored by GPU mode 2");
    }

  actual_cpu_threads_ = cpu_threads == 0
      ? static_cast<int>(available_threads()) : cpu_threads;
  if (actual_cpu_threads_ < 1) actual_cpu_threads_ = 1;

  if (requested_backend_ == "serial")
    {
      actual_backend_ = "serial";
      actual_cpu_variant_ = "serial";
      actual_gpu_variant_ = "serial";
      actual_cpu_threads_ = 1;
      actual_gpu2_.mode = 0;
      configured_ = true;
      return;
    }

  if (requested_backend_ == "gpu")
    {
      const bool requested_gpu_available = requested_gpu2_.mode == 2
          ? gpu_mode2_available() : gpu_available();
      if (requested_gpu_available)
        {
          actual_backend_ = "gpu";
          actual_gpu2_ = requested_gpu2_;
          if (requested_gpu2_.mode == 1)
            {
              actual_cpu_variant_ = "serial";
              actual_gpu_variant_ = requested_gpu_variant_ == "auto"
                  ? (reduction_mode_ == "deterministic" ? "sorted" : "atomic")
                  : requested_gpu_variant_;
              actual_cpu_threads_ = 1;
            }
          else
            {
              actual_gpu_variant_ = "resident";
              actual_gpu2_.host_threads = requested_gpu2_.host_threads == 0
                  ? 1 : requested_gpu2_.host_threads;
              actual_cpu_threads_ = actual_gpu2_.host_threads;
              if (requested_gpu2_.scope == "primary")
                {
                  if (requested_gpu2_.primary_mover == "openmp")
                    {
                      if (!openmp_available())
                        throw std::runtime_error(
                            "gpu2_primary_mover=openmp requires an "
                            "OpenMP-enabled build; no fallback was performed");
                      actual_cpu_variant_ = "gpu2_openmp_primary";
                    }
                  else if (requested_gpu2_.primary_mover == "pool")
                    actual_cpu_variant_ = "gpu2_pool_primary";
                  else if (requested_gpu2_.primary_mover == "auto")
                    actual_cpu_variant_ = "gpu2_primary_pending";
                  else
                    actual_cpu_variant_ = "serial";
                }
              else
                {
                  actual_cpu_variant_ = actual_cpu_threads_ > 1
                      ? "gpu2_openmp_host" : "serial";
                  if (actual_cpu_threads_ > 1 && !openmp_available())
                    {
                      if (!allow_fallback_)
                        throw std::runtime_error(
                            "gpu2_host_threads>1 requires an OpenMP-enabled build");
                      actual_cpu_threads_ = 1;
                      actual_gpu2_.host_threads = 1;
                      actual_cpu_variant_ = "serial";
                      append_fallback_reason(
                          "OpenMP unavailable; selected one GPU2 host thread");
                    }
                }
            }
          if (reduction_mode_ == "strict")
            {
              if (!allow_fallback_)
                throw std::runtime_error(
                    "reduction_mode=strict requires serial or cpu_variant=pool_dual; "
                    "CUDA CIC changes floating-point accumulation order");
              actual_backend_ = "serial";
              actual_gpu_variant_ = "serial";
              actual_gpu2_.mode = 0;
              append_fallback_reason(
                  "strict reduction rejected CUDA CIC; selected serial oracle");
            }
          else if (requested_gpu2_.mode == 1 &&
                   reduction_mode_ == "deterministic" &&
                   actual_gpu_variant_ != "sorted")
            {
              if (!allow_fallback_)
                throw std::runtime_error(
                    "CUDA deterministic reduction requires gpu_variant=sorted");
              actual_gpu_variant_ = "sorted";
              append_fallback_reason(
                  "deterministic reduction selected the sorted CUDA kernel");
            }
          else if (requested_gpu2_.mode == 2 &&
                   reduction_mode_ == "deterministic" &&
                   actual_gpu2_.deposition != "auto" &&
                   actual_gpu2_.deposition != "sort_reduce")
            {
              if (!allow_fallback_)
                throw std::runtime_error(
                    "GPU2 deterministic reduction requires "
                    "gpu2_deposition=auto or sort_reduce");
              actual_gpu2_.deposition = "sort_reduce";
              append_fallback_reason(
                  "deterministic reduction selected GPU2 sort_reduce deposition");
            }
          configured_ = true;
          return;
        }
      if (!allow_fallback_)
        {
          if (requested_gpu2_.mode == 2)
            throw std::runtime_error(
                "gpu_mode=2 requested, but this build has no resident GPU2 "
                "backend; build and link the resident pipeline with "
                "FEEBL_HAVE_GPU2");
          throw std::runtime_error(
              "compute_backend=gpu requested, but this build has no CUDA backend; "
              "rebuild with -DFEEBL_ENABLE_CUDA=ON");
        }
      append_fallback_reason(requested_gpu2_.mode == 2
          ? "resident GPU2 backend unavailable; selected CPU backend"
          : "CUDA backend unavailable; selected CPU backend");
    }

  if (requested_backend_ == "auto")
    {
      actual_backend_ = "serial";
      actual_cpu_variant_ = "serial";
      actual_gpu_variant_ = "serial";
      actual_cpu_threads_ = 1;
      actual_gpu2_.mode = 0;
      append_fallback_reason(
          "no accelerated backend passed the preregistered representative "
          "end-to-end 10% speedup gate; retained the serial oracle");
      configured_ = true;
      return;
    }

  actual_backend_ = "cpu";
  actual_gpu_variant_ = "serial";
  actual_gpu2_.mode = 0;
  actual_cpu_variant_ = requested_cpu_variant_;
  if (actual_cpu_variant_ == "auto")
    actual_cpu_variant_ = reduction_mode_ == "strict" ? "pool_dual" : "pool_private";

  if (reduction_mode_ == "strict" && actual_cpu_variant_ != "pool_dual")
    {
      if (!allow_fallback_ && requested_backend_ != "auto")
        throw std::runtime_error(
            "strict reduction requires cpu_variant=pool_dual; "
            "pool_private changes summation grouping and omp_atomic is unordered");
      append_fallback_reason(
          "strict reduction selected the serial-order pool_dual kernel");
      actual_cpu_variant_ = "pool_dual";
    }

  if (actual_cpu_variant_ == "omp_atomic" && !openmp_available())
    {
      if (!allow_fallback_ && requested_backend_ != "auto")
        throw std::runtime_error(
            "cpu_variant=omp_atomic requested, but OpenMP was not enabled at build time");
      append_fallback_reason("OpenMP unavailable; selected pool_private");
      actual_cpu_variant_ = "pool_private";
    }

  if (actual_cpu_variant_ == "omp_atomic" && reduction_mode_ != "fast")
    {
      if (!allow_fallback_)
        throw std::runtime_error(
            "cpu_variant=omp_atomic requires reduction_mode=fast");
      append_fallback_reason(
          "ordered reduction requested; selected pool_private instead of omp_atomic");
      actual_cpu_variant_ = "pool_private";
    }

#ifdef FEEBL_HAVE_OPENMP
  if (actual_cpu_variant_ == "omp_atomic")
    {
      omp_set_dynamic(0);
      omp_set_num_threads(actual_cpu_threads_);
    }
#endif

  if (actual_cpu_variant_ == "pool_private" || actual_cpu_variant_ == "pool_dual")
    pool_.reset(new CPU_THREAD_POOL(static_cast<std::size_t>(actual_cpu_threads_)));
  configured_ = true;
}

void PARALLEL_BACKEND::run_tasks(
    std::size_t task_count,
    const std::function<void(std::size_t)>& task)
{
  if (!pool_)
    {
      for (std::size_t i = 0; i < task_count; ++i) task(i);
      return;
    }
  pool_->run(task_count, task);
}

void PARALLEL_BACKEND::run_static_ranges(
    std::size_t item_count,
    void* context,
    STATIC_RANGE_TASK task)
{
  if (!task)
    throw std::invalid_argument("static-range task must not be null");
  if (gpu_mode5_enabled())
    {
      if (!gpu2_plan_committed_ || !gpu2_primary_pool_ ||
          actual_gpu2_.primary_mover != "pool")
        throw std::logic_error(
            "gpu_mode=5 host static-range dispatch requires its committed "
            "persistent pool plan");
      gpu2_primary_pool_->run_static_ranges(item_count, context, task);
      return;
    }
  if (!pool_)
    {
      if (item_count != 0) task(context, 0, 0, item_count);
      return;
    }
  pool_->run_static_ranges(item_count, context, task);
}

void PARALLEL_BACKEND::run_static_ranges_limited(
    std::size_t item_count,
    std::size_t active_lanes,
    void* context,
    STATIC_RANGE_TASK task)
{
  if (!task)
    throw std::invalid_argument("static-range task must not be null");
  if (active_lanes == 0)
    throw std::invalid_argument(
        "static-range active lane count must be positive");
  if (gpu_mode5_enabled())
    {
      if (!gpu2_plan_committed_ || !gpu2_primary_pool_ ||
          actual_gpu2_.primary_mover != "pool")
        throw std::logic_error(
            "gpu_mode=5 host static-range dispatch requires its committed "
            "persistent pool plan");
      gpu2_primary_pool_->run_static_ranges_limited(
          item_count, active_lanes, context, task);
      return;
    }
  if (!pool_)
    {
      if (active_lanes != 1)
        throw std::invalid_argument(
            "serial static-range dispatch supports exactly one active lane");
      if (item_count != 0) task(context, 0, 0, item_count);
      return;
    }
  pool_->run_static_ranges_limited(
      item_count, active_lanes, context, task);
}

void PARALLEL_BACKEND::run_gpu2_primary_static_ranges(
    std::size_t item_count,
    void* context,
    STATIC_RANGE_TASK task)
{
  if (!task)
    throw std::invalid_argument("GPU2 primary static-range task must not be null");
  if (!gpu2_plan_committed_ ||
      (!gpu_mode2_enabled() && !gpu_mode4_enabled() &&
       !gpu_mode5_enabled()) ||
      actual_gpu2_.scope != "primary" ||
      actual_gpu2_.primary_mover != "pool" || !gpu2_primary_pool_)
    throw std::logic_error(
        "GPU2/mode-4/mode-5 primary pool dispatch requires a committed "
        "pool plan");
  gpu2_primary_pool_->run_static_ranges(item_count, context, task);
}

void PARALLEL_BACKEND::record_cpu_primary_move_execution(
    std::size_t particles)
{
  if (!cpu_enabled() ||
      (actual_cpu_variant_ != "pool_private" &&
       actual_cpu_variant_ != "pool_dual"))
    throw std::logic_error(
        "CPU primary movement requires an active persistent pool backend");
  ++cpu_primary_move_calls_;
  cpu_primary_move_particles_ += particles;
}

void PARALLEL_BACKEND::record_gpu2_primary_mover_execution(
    const std::string& mover,
    std::size_t particles,
    std::size_t observed_threads,
    std::size_t nonempty_ranges)
{
  const std::string normalized = lower_ascii(mover);
  if (!gpu2_plan_committed_ ||
      (!gpu_mode2_enabled() && !gpu_mode4_enabled() &&
       !gpu_mode5_enabled()) ||
      actual_gpu2_.scope != "primary" ||
      normalized != actual_gpu2_.primary_mover)
    throw std::logic_error(
        "GPU2/mode-4/mode-5 primary mover execution does not match its "
        "committed plan");
  if (particles != 0 && nonempty_ranges == 0)
    throw std::logic_error(
        "GPU2 primary mover recorded particles without an execution range");
  if (observed_threads !=
      static_cast<std::size_t>(actual_gpu2_.host_threads))
    throw std::logic_error(
        "GPU2 primary mover observed a different team than its committed plan");
  if (nonempty_ranges > static_cast<std::size_t>(actual_gpu2_.host_threads))
    throw std::logic_error(
        "GPU2 primary mover recorded more ranges than committed workers");
  ++gpu2_primary_mover_calls_;
  gpu2_primary_mover_particles_ += particles;
  gpu2_primary_mover_ranges_ += nonempty_ranges;
  gpu2_primary_mover_observed_threads_ = observed_threads;
}

void PARALLEL_BACKEND::runtime_fallback_to_serial(const std::string& reason)
{
  if (!allow_fallback_)
    throw std::logic_error("runtime backend fallback was not authorized");
  actual_backend_ = "serial";
  actual_cpu_variant_ = "serial";
  actual_gpu_variant_ = "serial";
  actual_cpu_threads_ = 1;
  actual_gpu2_.mode = 0;
  gpu2_plan_committed_ = false;
  pool_.reset();
  gpu2_primary_pool_.reset();
  if (!reason.empty())
    {
      if (!fallback_reason_.empty()) fallback_reason_ += "; ";
      fallback_reason_ += reason;
    }
}

void PARALLEL_BACKEND::record_gpu_execution(
    const std::string& device_description)
{
  if (!gpu_mode1_enabled())
    throw std::logic_error("GPU mode 1 execution recorded for a non-mode-1 backend");
  ++gpu_calls_;
  if (gpu_device_description_.empty())
    gpu_device_description_ = device_description;
}

void PARALLEL_BACKEND::record_gpu2_plan(const GPU2_OPTIONS& actual_options)
{
  if (!gpu_mode2_enabled())
    throw std::logic_error("GPU2 plan recorded for a non-mode-2 backend");
  GPU2_OPTIONS plan = actual_options;
  plan.scope = lower_ascii(plan.scope);
  plan.deposition = lower_ascii(plan.deposition);
  plan.fft = lower_ascii(plan.fft);
  plan.graphs = lower_ascii(plan.graphs);
  plan.fusion = lower_ascii(plan.fusion);
  plan.pinned_io = lower_ascii(plan.pinned_io);
  plan.primary_mover = lower_ascii(plan.primary_mover);
  plan.profile = lower_ascii(plan.profile);
  if (plan.mode != 2 ||
      (plan.scope != "fields" && plan.scope != "primary" &&
       plan.scope != "full") ||
      (plan.deposition != "atomic" && plan.deposition != "shared_tile" &&
       plan.deposition != "sort_reduce" &&
       plan.deposition != "host_rho") ||
      plan.fft != "cufft" || plan.streams < 1 ||
      (plan.graphs != "off" && plan.graphs != "on") ||
      (plan.fusion != "off" && plan.fusion != "on") ||
      plan.batch_slices < 1 ||
      (plan.pinned_io != "off" && plan.pinned_io != "on") ||
      plan.host_threads < 1 ||
      (plan.primary_mover != "serial" &&
       plan.primary_mover != "openmp" &&
       plan.primary_mover != "pool") ||
      !(plan.memory_fraction > 0.0 && plan.memory_fraction <= 1.0) ||
      (plan.profile != "off" && plan.profile != "summary" &&
       plan.profile != "nvtx"))
    throw std::invalid_argument(
        "GPU2 preflight plan must contain concrete, valid choices");
  // host_rho is an actual-plan-only marker for the first fields ablation.  It
  // is intentionally absent from input validation and cannot be used to make
  // a wider primary/full plan look resident when deposition remains on host.
  if (plan.deposition == "host_rho" && plan.scope != "fields")
    throw std::invalid_argument(
        "GPU2 host_rho deposition is valid only for scope=fields");
  if (plan.scope != "primary" && plan.primary_mover != "serial")
    throw std::invalid_argument(
        "GPU2 primary mover must remain serial outside scope=primary");
  if (plan.scope == "primary" && plan.primary_mover == "serial" &&
      plan.host_threads != 1)
    throw std::invalid_argument(
        "GPU2 serial primary mover requires exactly one host thread");
  if (plan.scope == "primary" && plan.primary_mover == "openmp" &&
      !openmp_available())
    throw std::invalid_argument(
        "GPU2 OpenMP primary mover requires an OpenMP-enabled build");
  if (reduction_mode_ == "deterministic" && plan.deposition == "atomic")
    throw std::invalid_argument(
        "GPU2 deterministic execution cannot commit atomic deposition; "
        "sort_reduce is required");
  if (plan.memory_fraction > requested_gpu2_.memory_fraction)
    throw std::invalid_argument(
        "GPU2 plan exceeds gpu2_memory_fraction reservation limit");
  // Construct the persistent pool before publishing the plan.  Allocation or
  // thread-creation failure therefore leaves the previous uncommitted state
  // intact and can be handled by the caller's preflight failure policy.
  std::unique_ptr<CPU_THREAD_POOL> candidate_primary_pool;
  if (plan.scope == "primary" && plan.primary_mover == "pool")
    candidate_primary_pool.reset(
        new CPU_THREAD_POOL(static_cast<std::size_t>(plan.host_threads)));

  actual_gpu2_ = plan;
  actual_cpu_threads_ = plan.host_threads;
  if (plan.scope == "primary" && plan.primary_mover == "openmp")
    actual_cpu_variant_ = "gpu2_openmp_primary";
  else if (plan.scope == "primary" && plan.primary_mover == "pool")
    actual_cpu_variant_ = "gpu2_pool_primary";
  else
    actual_cpu_variant_ = plan.host_threads > 1
        ? "gpu2_openmp_host" : "serial";
  gpu2_primary_pool_ = std::move(candidate_primary_pool);
  gpu2_plan_committed_ = true;
}

void PARALLEL_BACKEND::record_gpu2_execution(
    const std::string& device_description)
{
  if (!gpu_mode2_enabled() || !gpu2_plan_committed_)
    throw std::logic_error(
        "GPU2 execution requires an active backend and committed preflight plan");
  ++gpu_calls_;
  ++gpu2_calls_;
  if (gpu_device_description_.empty())
    gpu_device_description_ = device_description;
}

void PARALLEL_BACKEND::record_gpu_mode3_execution(
    const std::string& device_description)
{
  if (!gpu_mode3_enabled() || !configured_)
    throw std::logic_error(
        "GPU mode 3 execution requires an active modern_gpu backend");
  if (allow_fallback_)
    throw std::logic_error(
        "GPU mode 3 execution cannot be recorded with fallback enabled");
  if (device_description.empty())
    throw std::invalid_argument(
        "GPU mode 3 execution requires a nonempty device description");
  if (gpu3_calls_ != 0)
    throw std::logic_error(
        "GPU mode 3 whole-crossing execution may be recorded only once");
  if (gpu_calls_ == std::numeric_limits<std::size_t>::max() ||
      gpu3_calls_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("GPU mode 3 execution counter overflow");
  ++gpu_calls_;
  ++gpu3_calls_;
  if (gpu_device_description_.empty())
    gpu_device_description_ = device_description;
}

void PARALLEL_BACKEND::record_gpu_mode4_plan(
    const GPU2_OPTIONS& actual_options)
{
  if (!gpu_mode4_enabled() || !configured_)
    throw std::logic_error(
        "GPU mode 4 plan requires an active legacy compositional backend");
  if (gpu2_plan_committed_)
    throw std::logic_error("GPU mode 4 plan was already committed");
  GPU2_OPTIONS plan = actual_options;
  plan.scope = lower_ascii(plan.scope);
  plan.deposition = lower_ascii(plan.deposition);
  plan.fft = lower_ascii(plan.fft);
  plan.graphs = lower_ascii(plan.graphs);
  plan.fusion = lower_ascii(plan.fusion);
  plan.pinned_io = lower_ascii(plan.pinned_io);
  plan.primary_mover = lower_ascii(plan.primary_mover);
  plan.profile = lower_ascii(plan.profile);
  const bool serial_host_plan =
      plan.primary_mover == "serial" && plan.host_threads == 1;
  const bool pool_host_plan =
      plan.primary_mover == "pool" && plan.host_threads >= 2;
  if (plan.mode != 4 || plan.scope != "primary" ||
      plan.deposition != "atomic" || plan.fft != "cufft" ||
      plan.streams != 1 || plan.graphs != "off" ||
      plan.fusion != "on" || plan.batch_slices < 1 ||
      plan.batch_slices > 256 || plan.pinned_io != "on" ||
      (!serial_host_plan && !pool_host_plan) ||
      plan.profile != "summary" ||
      !(plan.memory_fraction > 0.0 && plan.memory_fraction <= 1.0) ||
      plan.memory_fraction > requested_gpu2_.memory_fraction)
    throw std::invalid_argument(
        "GPU mode 4 preflight plan is not the qualified wave-batched plan");
  if (plan.host_threads != requested_gpu2_.host_threads ||
      plan.primary_mover != requested_gpu2_.primary_mover)
    throw std::invalid_argument(
        "GPU mode 4 preflight host plan does not match the requested "
        "gpu2_host_threads/gpu2_primary_mover selectors");

  // Construct the custom pool before publishing the plan.  It is independent
  // of OpenMP and remains alive for the backend lifetime.  If thread creation
  // fails, the mode-4 plan remains uncommitted and cannot execute or fall back.
  std::unique_ptr<CPU_THREAD_POOL> candidate_primary_pool;
  if (pool_host_plan)
    candidate_primary_pool.reset(
        new CPU_THREAD_POOL(static_cast<std::size_t>(plan.host_threads)));

  actual_gpu2_ = plan;
  actual_cpu_threads_ = plan.host_threads;
  actual_cpu_variant_ = pool_host_plan
      ? "gpu4_pool_static_ranges" : "serial_ordered_host_physics";
  gpu2_primary_pool_ = std::move(candidate_primary_pool);
  gpu2_plan_committed_ = true;
}

void PARALLEL_BACKEND::record_gpu_mode4_execution(
    const std::string& device_description)
{
  if (!gpu_mode4_enabled() || !gpu2_plan_committed_ || !configured_)
    throw std::logic_error(
        "GPU mode 4 execution requires its committed batch plan");
  if (allow_fallback_)
    throw std::logic_error("GPU mode 4 execution cannot use fallback");
  if (device_description.empty())
    throw std::invalid_argument(
        "GPU mode 4 execution requires a device description");
  if (gpu_calls_ == std::numeric_limits<std::size_t>::max() ||
      gpu4_calls_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("GPU mode 4 execution counter overflow");
  ++gpu_calls_;
  ++gpu4_calls_;
  if (gpu_device_description_.empty())
    gpu_device_description_ = device_description;
}

void PARALLEL_BACKEND::record_gpu_mode5_plan(
    const GPU2_OPTIONS& actual_options)
{
  if (!gpu_mode5_enabled() || !configured_)
    throw std::logic_error(
        "GPU mode 5 plan requires an active legacy adaptive backend");
  if (gpu2_plan_committed_)
    throw std::logic_error("GPU mode 5 plan was already committed");
  GPU2_OPTIONS plan = actual_options;
  plan.scope = lower_ascii(plan.scope);
  plan.deposition = lower_ascii(plan.deposition);
  plan.fft = lower_ascii(plan.fft);
  plan.graphs = lower_ascii(plan.graphs);
  plan.fusion = lower_ascii(plan.fusion);
  plan.pinned_io = lower_ascii(plan.pinned_io);
  plan.primary_mover = lower_ascii(plan.primary_mover);
  plan.profile = lower_ascii(plan.profile);
  if (plan.mode != 5 || plan.scope != "primary" ||
      plan.deposition != "atomic" || plan.fft != "cufft" ||
      plan.streams != 1 || plan.graphs != "off" ||
      plan.fusion != "on" || plan.batch_slices < 1 ||
      plan.batch_slices > 256 || plan.pinned_io != "on" ||
      plan.host_threads < 2 || plan.primary_mover != "pool" ||
      plan.profile != "summary" ||
      !(plan.memory_fraction > 0.0 && plan.memory_fraction <= 1.0) ||
      plan.memory_fraction > requested_gpu2_.memory_fraction)
    throw std::invalid_argument(
        "GPU mode 5 preflight plan is not the qualified adaptive "
        "class-batched plan");
  if (plan.host_threads != requested_gpu2_.host_threads ||
      plan.batch_slices != requested_gpu2_.batch_slices)
    throw std::invalid_argument(
        "GPU mode 5 preflight plan changed a requested host/batch bound");

  // Publish the plan only after its persistent pool has been constructed.
  std::unique_ptr<CPU_THREAD_POOL> candidate_primary_pool(
      new CPU_THREAD_POOL(static_cast<std::size_t>(plan.host_threads)));
  actual_gpu2_ = plan;
  actual_cpu_threads_ = plan.host_threads;
  actual_cpu_variant_ = "gpu5_pool_static_ranges";
  actual_gpu_variant_ = "adaptive_class_planmany_fields";
  gpu2_primary_pool_ = std::move(candidate_primary_pool);
  gpu2_plan_committed_ = true;
}

void PARALLEL_BACKEND::record_gpu_mode5_execution(
    const std::string& device_description)
{
  if (!gpu_mode5_enabled() || !gpu2_plan_committed_ || !configured_)
    throw std::logic_error(
        "GPU mode 5 execution requires its committed adaptive batch plan");
  if (allow_fallback_)
    throw std::logic_error("GPU mode 5 execution cannot use fallback");
  if (device_description.empty())
    throw std::invalid_argument(
        "GPU mode 5 execution requires a device description");
  if (gpu_calls_ == std::numeric_limits<std::size_t>::max() ||
      gpu5_calls_ == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("GPU mode 5 execution counter overflow");
  ++gpu_calls_;
  ++gpu5_calls_;
  if (gpu_device_description_.empty())
    gpu_device_description_ = device_description;
}

std::string PARALLEL_BACKEND::provenance() const
{
  const bool mode3_active = gpu_mode3_enabled();
  const bool mode4_active = gpu_mode4_enabled();
  const bool mode5_active = gpu_mode5_enabled();
  const bool resident_controls_active =
      actual_gpu2_.mode == 2 || mode3_active || mode4_active || mode5_active;
  const bool primary_plan =
      gpu2_plan_committed_ && actual_gpu2_.scope == "primary";
  const bool mode4_pool_plan =
      mode4_active && primary_plan && actual_gpu2_.primary_mover == "pool";
  const bool mode5_pool_plan =
      mode5_active && primary_plan && actual_gpu2_.primary_mover == "pool";
  std::string primary_mover = "inactive";
  std::string primary_update = "inactive";
  std::string primary_side_effects = "inactive";
  std::string primary_rng = "inactive";
  if (mode3_active)
    {
      primary_mover = "device_gather_push";
      primary_update = "resident_device_gather_push_whole_crossing";
      primary_side_effects = "unsupported_rejected";
      primary_rng = "unused_no_stochastic_collision";
    }
  else if (mode5_active)
    {
      primary_mover = mode5_pool_plan ? "pool" : "invalid";
      primary_update = "host_pool_after_adaptive_class_batch";
      primary_side_effects = "qualified_host_primary_photon_luminosity";
      primary_rng = "qualified_host_wavefront_contract";
    }
  else if (mode4_active)
    {
      primary_mover = mode4_pool_plan ? "pool" : "host_serial";
      primary_update = mode4_pool_plan
          ? "host_pool_static_ranges_after_wave_batch"
          : "host_serial_ordered_after_wave_batch";
      primary_side_effects = mode4_pool_plan
          ? "caller_owned_lane_contract" : "host_serial_ordered";
      primary_rng = mode4_pool_plan
          ? "caller_owned_lane_contract" : "legacy_host_ordered";
    }
  else if (primary_plan)
    {
      primary_mover = actual_gpu2_.primary_mover;
      primary_update = primary_mover == "serial"
          ? "host_serial"
          : (primary_mover == "openmp"
                 ? "host_openmp_static_combined"
                 : "host_pool_static_combined");
      primary_side_effects = primary_mover == "serial"
          ? "host_serial" : "disabled_by_parallel_contract";
      primary_rng = primary_mover == "serial"
          ? "legacy_host_ordered" : "unused_by_parallel_contract";
    }
  std::ostringstream out;
  out << "compute_backend requested=" << requested_backend_
      << " actual=" << actual_backend_
      << " cpu_variant requested=" << requested_cpu_variant_
      << " actual=" << actual_cpu_variant_
      << " cpu_threads requested=" << requested_cpu_threads_
      << " actual=" << actual_cpu_threads_
      << " reduction_mode=" << reduction_mode_
      << " rng_backend=" << rng_backend_
      << " gpu_variant requested=" << requested_gpu_variant_
      << " actual=" << actual_gpu_variant_
      << " gpu_device=" << gpu_device_
      << " gpu_min_particles=" << gpu_min_particles_
      << " gpu_calls=" << gpu_calls_
      << " gpu_mode requested=" << requested_gpu2_.mode
      << " actual=" << actual_gpu2_.mode
      << " gpu2_scope requested=" << requested_gpu2_.scope
      << " actual=" << (resident_controls_active ? actual_gpu2_.scope : "inactive")
      << " gpu2_deposition requested=" << requested_gpu2_.deposition
      << " actual=" << (resident_controls_active ? actual_gpu2_.deposition : "inactive")
      << " gpu2_fft requested=" << requested_gpu2_.fft
      << " actual=" << (resident_controls_active ? actual_gpu2_.fft : "inactive")
      << " gpu2_streams requested=" << requested_gpu2_.streams
      << " actual=" << (resident_controls_active ? actual_gpu2_.streams : 0)
      << " gpu2_graphs requested=" << requested_gpu2_.graphs
      << " actual=" << (resident_controls_active ? actual_gpu2_.graphs : "inactive")
      << " gpu2_fusion requested=" << requested_gpu2_.fusion
      << " actual=" << (resident_controls_active ? actual_gpu2_.fusion : "inactive")
      << " gpu2_batch_slices requested=" << requested_gpu2_.batch_slices
      << " actual=" << (resident_controls_active ? actual_gpu2_.batch_slices : 0)
      << " gpu2_pinned_io requested=" << requested_gpu2_.pinned_io
      << " actual=" << (resident_controls_active ? actual_gpu2_.pinned_io : "inactive")
      << " gpu2_host_threads requested=" << requested_gpu2_.host_threads
      << " actual=" << (resident_controls_active ? actual_gpu2_.host_threads : 0)
      << " gpu2_primary_mover requested=" << requested_gpu2_.primary_mover
      << " actual=" << primary_mover
      << " gpu2_memory_fraction=" << requested_gpu2_.memory_fraction
      << " gpu2_profile requested=" << requested_gpu2_.profile
      << " actual=" << (resident_controls_active ? actual_gpu2_.profile : "inactive")
      << " gpu2_plan=" << (gpu2_plan_committed_ ? "committed" :
                            (mode3_active ? "mode3_crossing_contract" :
                             "uncommitted"))
      << " gpu2_calls=" << gpu2_calls_
      << " gpu3_calls=" << gpu3_calls_
      << " gpu4_calls=" << gpu4_calls_
      << " gpu5_calls=" << gpu5_calls_
      << " gpu2_host_boundary="
      << (mode3_active
              ? "one_pinned_upload_one_pinned_download"
              : (mode5_active
              ? "one_completion_boundary_per_adaptive_class_batch"
              : (mode4_active
              ? "one_completion_boundary_per_wave_bucket"
              : (gpu2_plan_committed_ && actual_gpu2_.scope == "primary"
              ? "compact_primary_fields"
              : (gpu2_plan_committed_ && actual_gpu2_.scope == "fields"
                     ? (actual_gpu2_.deposition == "host_rho"
                            ? "phi_materialization"
                            : (actual_gpu2_.deposition == "atomic"
                                   ? "compact_metrics_then_phi_materialization"
                                   : "inactive"))
                     : "inactive")))))
      << " gpu2_primary_update="
      << primary_update
      << " gpu2_spin_transport="
      << primary_side_effects
      << " gpu2_beamstrahlung="
      << primary_side_effects
      << " gpu2_rng="
      << primary_rng
      << " gpu2_host_phi_materialized="
      << ((mode3_active || mode4_active || mode5_active) ? 0 :
          (gpu2_plan_committed_ && actual_gpu2_.scope == "primary" ? 0 :
          (gpu2_plan_committed_ && actual_gpu2_.scope == "fields" ? 1 : -1)))
      << " gpu2_primary_mover_calls=" << gpu2_primary_mover_calls_
      << " gpu2_primary_mover_particles=" << gpu2_primary_mover_particles_
      << " gpu2_primary_mover_ranges=" << gpu2_primary_mover_ranges_
      << " gpu2_primary_mover_observed_threads="
      << gpu2_primary_mover_observed_threads_
      << " initialization_rng="
      << (mode3_active ? "legacy_sequential" : "as_rng_backend")
      << " mode3_collision_rng="
      << (mode3_active ? "unused" : "inactive")
      << " gpu_device_description=\""
      << (gpu_device_description_.empty() ? "not-initialized"
                                          : gpu_device_description_)
      << '\"'
      << " openmp=" << (openmp_available() ? "enabled" : "disabled")
      << " cuda=" << (gpu_available() ? "enabled" : "disabled")
      << " cuda_toolkit=" << FEEBL_CUDA_TOOLKIT_VERSION
      << " affinity=" << affinity_description()
      << " cxx=" << __cplusplus
      << " compiler=\"" << __VERSION__ << "\""
      << " source=" << FEEBL_BUILD_SOURCE_HASH;
  if (cpu_primary_move_calls_ != 0)
    out << " cpu_primary_move=pool-static-primary-v1"
        << " cpu_primary_move_calls=" << cpu_primary_move_calls_
        << " cpu_primary_move_particles=" << cpu_primary_move_particles_;
  if (modern_provenance_active_)
    {
      out << " engine requested=" << requested_modern_.engine
          << " actual=" << actual_engine_
          << " algorithm requested=" << requested_modern_.algorithm
          << " actual=" << (mode3_active
                                ? requested_modern_.algorithm : "inactive")
          << " precision requested=" << requested_modern_.precision
          << " actual=" << (mode3_active
                                ? requested_modern_.precision : "inactive")
          << " modern_reduction requested=" << requested_modern_.reduction
          << " actual=" << (mode3_active
                                ? requested_modern_.reduction : "inactive")
          << " modern_batch_width requested=" << requested_modern_.batch_width
          << " actual=" << (mode3_active
                                ? requested_modern_.batch_width : 0)
          << " modern_cpu_workers requested=" << requested_modern_.cpu_workers
          << " actual=0"
          << " modern_cpu_runtime requested=" << requested_modern_.cpu_runtime
          << " actual=inactive"
          << " modern_cpu_affinity requested=" << requested_modern_.cpu_affinity
          << " actual=inactive"
          << " modern_gpu_streams requested=" << requested_modern_.gpu_streams
          << " actual=" << (mode3_active
                                ? requested_modern_.gpu_streams : 0)
          << " modern_gpu_graphs requested=" << requested_modern_.gpu_graphs
          << " actual=" << (mode3_active
                                ? requested_modern_.gpu_graphs : "inactive")
          << " modern_trace requested=" << requested_modern_.trace
          << " actual=" << (mode3_active
                                ? requested_modern_.trace : "inactive")
          << " mode3_policy="
          << (mode3_active
                  ? "exact_schedule/mixed_phase_fp32_fields_fp64/"
                    "fast_atomic_cic"
                  : "inactive");
    }
  if (!fallback_reason_.empty()) out << " fallback=\"" << fallback_reason_ << '"';
  return out.str();
}

PARALLEL_BACKEND& parallel_backend()
{
  static PARALLEL_BACKEND backend;
  return backend;
}
