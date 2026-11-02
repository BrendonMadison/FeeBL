#ifndef PARALLEL_BACKEND_CPP_SEEN
#define PARALLEL_BACKEND_CPP_SEEN

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

class CPU_THREAD_POOL;

/** Input-card controls for the resident CUDA pipeline.
 *
 * These defaults are deliberately inert while gpu_mode=1.  Keeping them in a
 * value object avoids growing the historical configure interface every time a
 * resident-pipeline tuning control is added.
 */
struct GPU2_OPTIONS
{
  int mode;
  std::string scope;
  std::string deposition;
  std::string fft;
  int streams;
  std::string graphs;
  std::string fusion;
  int batch_slices;
  std::string pinned_io;
  int host_threads;
  std::string primary_mover;
  double memory_fraction;
  std::string profile;

  GPU2_OPTIONS()
      : mode(1), scope("auto"), deposition("auto"), fft("cufft"),
        streams(0), graphs("auto"), fusion("auto"), batch_slices(0),
        pinned_io("auto"), host_threads(0), primary_mover("serial"),
        memory_fraction(0.80), profile("summary")
  {}
};

/** Parsed controls for the future modern simulation engines.
 *
 * The value object is intentionally inert today.  Keeping the modern controls
 * separate from compute_backend/reduction_mode prevents an unwired engine
 * request from being mistaken for one of the existing legacy accelerators.
 * PARALLEL_BACKEND validates and records the request, then refuses every
 * non-legacy engine before simulation until an engine is actually connected.
 */
struct MODERN_RUNTIME_OPTIONS
{
  std::string engine;
  std::string algorithm;
  std::string precision;
  std::string reduction;
  int batch_width;
  int cpu_workers;
  std::string cpu_runtime;
  std::string cpu_affinity;
  int gpu_streams;
  std::string gpu_graphs;
  std::string trace;

  MODERN_RUNTIME_OPTIONS()
      : engine("legacy"), algorithm("exact"), precision("fp64"),
        reduction("deterministic"), batch_width(0), cpu_workers(0),
        cpu_runtime("auto"), cpu_affinity("auto"), gpu_streams(0),
        gpu_graphs("auto"), trace("off")
  {}
};

/** Runtime selection and provenance for compute backends.
 *
 * The historical FeeBL path remains the default. CPU acceleration is entered
 * only after an explicit compute_backend="cpu" (or "auto") request. The
 * worker pool is created once after the deck is read and reused thereafter.
 */
class PARALLEL_BACKEND
{
  std::unique_ptr<CPU_THREAD_POOL> pool_;
  // gpu_mode=2 and the opt-in gpu_mode=4 host-pool plan are compositional
  // CUDA+host plans.  Their static-range pool is deliberately separate from
  // the CPU backend pool so compute_backend=gpu can never masquerade as, or
  // accidentally dispatch through, a CPU backend.
  std::unique_ptr<CPU_THREAD_POOL> gpu2_primary_pool_;
  std::string requested_backend_;
  std::string requested_cpu_variant_;
  std::string requested_gpu_variant_;
  GPU2_OPTIONS requested_gpu2_;
  GPU2_OPTIONS actual_gpu2_;
  MODERN_RUNTIME_OPTIONS requested_modern_;
  std::string actual_engine_;
  bool modern_provenance_active_;
  std::string rng_backend_;
  std::string reduction_mode_;
  std::string actual_backend_;
  std::string actual_cpu_variant_;
  std::string actual_gpu_variant_;
  std::string fallback_reason_;
  int requested_cpu_threads_;
  int actual_cpu_threads_;
  int gpu_device_;
  std::size_t gpu_min_particles_;
  std::size_t gpu_calls_;
  std::size_t gpu2_calls_;
  std::size_t gpu3_calls_;
  std::size_t gpu4_calls_;
  std::size_t gpu5_calls_;
  std::size_t cpu_primary_move_calls_;
  std::size_t cpu_primary_move_particles_;
  std::size_t gpu2_primary_mover_calls_;
  std::size_t gpu2_primary_mover_particles_;
  std::size_t gpu2_primary_mover_ranges_;
  std::size_t gpu2_primary_mover_observed_threads_;
  std::string gpu_device_description_;
  bool gpu2_plan_committed_;
  bool allow_fallback_;
  bool configured_;

  PARALLEL_BACKEND(const PARALLEL_BACKEND&);
  PARALLEL_BACKEND& operator=(const PARALLEL_BACKEND&);

 public:
  PARALLEL_BACKEND();
  ~PARALLEL_BACKEND();

  void configure(const std::string& compute_backend,
                 int cpu_threads,
                 const std::string& cpu_variant,
                 const std::string& gpu_variant,
                 int gpu_device,
                 int gpu_min_particles,
                 const std::string& rng_backend,
                 const std::string& reduction_mode,
                 bool allow_backend_fallback);

  void configure(const std::string& compute_backend,
                 int cpu_threads,
                 const std::string& cpu_variant,
                 const std::string& gpu_variant,
                 int gpu_device,
                 int gpu_min_particles,
                 const std::string& rng_backend,
                 const std::string& reduction_mode,
                 bool allow_backend_fallback,
                 const GPU2_OPTIONS& gpu2_options);

  void configure(const std::string& compute_backend,
                 int cpu_threads,
                 const std::string& cpu_variant,
                 const std::string& gpu_variant,
                 int gpu_device,
                 int gpu_min_particles,
                 const std::string& rng_backend,
                 const std::string& reduction_mode,
                 bool allow_backend_fallback,
                 const GPU2_OPTIONS& gpu2_options,
                 const MODERN_RUNTIME_OPTIONS& modern_options);

  // Compatibility overload used by the focused CPU unit tests and embedders.
  void configure(const std::string& compute_backend,
                 int cpu_threads,
                 const std::string& cpu_variant,
                 const std::string& reduction_mode,
                 bool allow_backend_fallback)
  {
    configure(compute_backend, cpu_threads, cpu_variant,
              "auto", 0, 262144, "legacy", reduction_mode,
              allow_backend_fallback);
  }

  bool cpu_enabled() const { return actual_backend_ == "cpu"; }
  bool gpu_enabled() const { return actual_backend_ == "gpu"; }
  bool gpu_mode1_enabled() const
  { return actual_backend_ == "gpu" && actual_gpu2_.mode == 1; }
  bool gpu_mode2_enabled() const
  { return actual_backend_ == "gpu" && actual_gpu2_.mode == 2; }
  bool gpu_mode3_enabled() const
  { return actual_backend_ == "gpu" && actual_gpu2_.mode == 3 &&
           actual_engine_ == "modern_gpu"; }
  bool gpu_mode4_enabled() const
  { return actual_backend_ == "gpu" && actual_gpu2_.mode == 4 &&
           actual_engine_ == "legacy"; }
  bool gpu_mode5_enabled() const
  { return actual_backend_ == "gpu" && actual_gpu2_.mode == 5 &&
           actual_engine_ == "legacy"; }
  bool serial_enabled() const { return actual_backend_ == "serial"; }
  bool openmp_available() const;
  bool gpu_available() const;
  bool gpu_mode2_available() const;
  bool gpu_mode3_available() const;
  bool gpu_mode4_available() const;
  bool gpu_mode5_available() const;
  bool configured() const { return configured_; }

  const std::string& requested_backend() const { return requested_backend_; }
  const std::string& requested_cpu_variant() const { return requested_cpu_variant_; }
  const std::string& requested_gpu_variant() const { return requested_gpu_variant_; }
  const std::string& rng_backend() const { return rng_backend_; }
  const std::string& reduction_mode() const { return reduction_mode_; }
  const std::string& actual_backend() const { return actual_backend_; }
  const std::string& actual_cpu_variant() const { return actual_cpu_variant_; }
  const std::string& actual_gpu_variant() const { return actual_gpu_variant_; }
  const MODERN_RUNTIME_OPTIONS& requested_modern_options() const
  { return requested_modern_; }
  const std::string& requested_engine() const
  { return requested_modern_.engine; }
  const std::string& actual_engine() const { return actual_engine_; }
  bool modern_provenance_active() const { return modern_provenance_active_; }
  const std::string& fallback_reason() const { return fallback_reason_; }
  int requested_cpu_threads() const { return requested_cpu_threads_; }
  int actual_cpu_threads() const { return actual_cpu_threads_; }
  int gpu_device() const { return gpu_device_; }
  std::size_t gpu_min_particles() const { return gpu_min_particles_; }
  int requested_gpu_mode() const { return requested_gpu2_.mode; }
  int actual_gpu_mode() const { return actual_gpu2_.mode; }
  const GPU2_OPTIONS& requested_gpu2_options() const { return requested_gpu2_; }
  const GPU2_OPTIONS& actual_gpu2_options() const { return actual_gpu2_; }
  bool gpu2_plan_committed() const { return gpu2_plan_committed_; }
  std::size_t gpu2_calls() const { return gpu2_calls_; }
  std::size_t gpu3_calls() const { return gpu3_calls_; }
  std::size_t gpu4_calls() const { return gpu4_calls_; }
  std::size_t gpu5_calls() const { return gpu5_calls_; }
  bool allow_fallback() const { return allow_fallback_; }

  /** Execute numbered, independent tasks on the persistent worker team. */
  void run_tasks(std::size_t task_count,
                 const std::function<void(std::size_t)>& task);

  /** Execute one stable contiguous range on every persistent worker.
   *
   * The callback/context form is deliberate: unlike std::function it needs
   * no per-dispatch heap object.  Its arguments are context, stable lane ID,
   * begin, and end.  A lane is permanently bound to one worker thread for the
   * pool lifetime and is in [0, actual_cpu_threads()).  Empty ranges are
   * skipped, item order inside every range is increasing, and the boundaries
   * depend only on item_count and the configured worker count.  The serial
   * fallback invokes lane 0.  Nested dispatch is rejected.
   */
  typedef void (*STATIC_RANGE_TASK)(void*, std::size_t, std::size_t,
                                    std::size_t);
  void run_static_ranges(std::size_t item_count,
                         void* context,
                         STATIC_RANGE_TASK task);

  /** Execute one stable contiguous range on at most active_lanes workers.
   *
   * This is the fail-closed dispatch for callers whose preflight admitted
   * fewer lane-private workspaces than the persistent pool owns.  Lane IDs
   * remain stable and are restricted to [0, active_lanes).  No temporary
   * worker team is created.
   */
  void run_static_ranges_limited(std::size_t item_count,
                                 std::size_t active_lanes,
                                 void* context,
                                 STATIC_RANGE_TASK task);

  /** Dispatch stable contiguous ranges on a committed GPU2/mode-4 primary
   * pool.  This entry point refuses serial/OpenMP and uncommitted plans and
   * never falls back to a different host runtime.
   */
  void run_gpu2_primary_static_ranges(std::size_t item_count,
                                      void* context,
                                      STATIC_RANGE_TASK task);

  /** Record observed production execution, not merely a requested flag. */
  void record_cpu_primary_move_execution(std::size_t particles);
  std::size_t cpu_primary_move_calls() const
  { return cpu_primary_move_calls_; }
  std::size_t cpu_primary_move_particles() const
  { return cpu_primary_move_particles_; }

  /** Record observed compact-primary or mode-4 host-pool execution. */
  void record_gpu2_primary_mover_execution(const std::string& mover,
                                            std::size_t particles,
                                            std::size_t observed_threads,
                                            std::size_t nonempty_ranges);
  std::size_t gpu2_primary_mover_calls() const
  { return gpu2_primary_mover_calls_; }
  std::size_t gpu2_primary_mover_particles() const
  { return gpu2_primary_mover_particles_; }
  std::size_t gpu2_primary_mover_ranges() const
  { return gpu2_primary_mover_ranges_; }
  std::size_t gpu2_primary_mover_observed_threads() const
  { return gpu2_primary_mover_observed_threads_; }

  /** Record an expressly authorized late device failure/threshold fallback. */
  void runtime_fallback_to_serial(const std::string& reason);
  void record_gpu_execution(const std::string& device_description);
  /** Commit the concrete, immutable choices made by GPU2 preflight. */
  void record_gpu2_plan(const GPU2_OPTIONS& actual_options);
  /** Record observed resident execution; a requested mode alone is not a pass. */
  void record_gpu2_execution(const std::string& device_description);
  /** Record one successfully committed whole-crossing mode-3 transaction. */
  void record_gpu_mode3_execution(const std::string& device_description);
  /** Commit the immutable wave-batched fields plan selected for mode 4. */
  void record_gpu_mode4_plan(const GPU2_OPTIONS& actual_options);
  /** Record one completed mode-4 wave/bucket host boundary. */
  void record_gpu_mode4_execution(const std::string& device_description);
  /** Commit the immutable adaptive class-bank/host-pool plan for mode 5. */
  void record_gpu_mode5_plan(const GPU2_OPTIONS& actual_options);
  /** Record one completed adaptive class batch after canonical publication. */
  void record_gpu_mode5_execution(const std::string& device_description);

  std::string provenance() const;
};

PARALLEL_BACKEND& parallel_backend();

#endif
