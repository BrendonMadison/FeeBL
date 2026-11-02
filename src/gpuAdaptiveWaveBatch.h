#ifndef FEEBL_GPU_ADAPTIVE_WAVE_BATCH_SEEN
#define FEEBL_GPU_ADAPTIVE_WAVE_BATCH_SEEN

#include "uniformWindowPlanCPP.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace feebl {
namespace adaptive_grid {
namespace gpu5 {

// Versioned identifiers are part of persisted provenance.  Values 2 and 3 are
// deliberately enumerable so input/provenance code never has to guess what a
// future implementation means; this executor currently admits only value 1.
enum class DepositionVariantId : std::uint32_t {
  GlobalAtomic = 1,
  SharedTile = 2,
  SegmentedReduction = 3
};

enum class ComplexPrecisionId : std::uint32_t {
  Float64Complex = 1
};

enum class TransformTypeId : std::uint32_t {
  ComplexToComplex2D = 1
};

enum class GreenSolverId : std::uint32_t {
  Case2FreeSpaceDoubledDomain = 1
};

enum class GreenNormalizationId : std::uint32_t {
  PredividedByFourNxNy = 1
};

enum class StoredChargeRepresentationId : std::uint32_t {
  IntegratedNodalMacroparticleCount = 1
};

// Bump this value whenever a kernel layout, cache identity, or result-binding
// contract changes.  A caller may replace it with a stronger build/source hash
// but may not pass zero.
constexpr std::uint64_t kExecutorAbiHashV1 = UINT64_C(0x8f92d7c54b31ae01);

struct DeviceIdentity {
  int ordinal = -1;
  std::array<std::uint8_t, 16> uuid{{0}};
  int compute_major = 0;
  int compute_minor = 0;
  int cuda_runtime_version = 0;
  int cuda_driver_version = 0;
  int cufft_version = 0;

  bool operator==(const DeviceIdentity& other) const noexcept;
  bool operator<(const DeviceIdentity& other) const noexcept;
};

// Exact identity of one reusable solve plan.  The dimensions use the same
// x-major/y-minor row-major order as gpuWaveBatch and FIELD's padded storage.
// Encounter origins, particle counts, and mutation revisions never belong in
// this key.
struct FftPlanCacheKey {
  DeviceIdentity device;
  ComplexPrecisionId precision = ComplexPrecisionId::Float64Complex;
  TransformTypeId transform_type = TransformTypeId::ComplexToComplex2D;
  std::size_t logical_n_x = 0;
  std::size_t logical_n_y = 0;
  std::size_t padded_n_x = 0;
  std::size_t padded_n_y = 0;
  std::size_t batch_capacity = 0;
  std::uint32_t rank = 2;
  std::array<std::size_t, 2> dimensions{{0, 0}};
  std::array<std::size_t, 2> input_embed{{0, 0}};
  std::array<std::size_t, 2> output_embed{{0, 0}};
  std::size_t input_stride = 1;
  std::size_t output_stride = 1;
  std::size_t input_distance = 0;
  std::size_t output_distance = 0;
  std::int32_t forward_direction = -1;
  std::int32_t inverse_direction = 1;
  bool in_place = true;
  bool explicit_workspace = true;
  bool one_nonblocking_stream = true;
  std::uint64_t build_abi_hash = kExecutorAbiHashV1;

  bool operator==(const FftPlanCacheKey& other) const noexcept;
  bool operator<(const FftPlanCacheKey& other) const noexcept;
  std::uint64_t stable_hash() const noexcept;
};

// Exact identity of one resident Green spectrum.  The historical kernel
// factor already contains -charge_sign and all physical constants; the raw
// charge-sign bits are retained separately for FIELD's exact-zero gate and
// provenance.  Batch capacity and encounter origin are intentionally absent.
struct GreenSpectrumCacheKey {
  DeviceIdentity device;
  GreenSolverId solver = GreenSolverId::Case2FreeSpaceDoubledDomain;
  GreenNormalizationId normalization =
      GreenNormalizationId::PredividedByFourNxNy;
  StoredChargeRepresentationId stored_charge =
      StoredChargeRepresentationId::IntegratedNodalMacroparticleCount;
  ComplexPrecisionId precision = ComplexPrecisionId::Float64Complex;
  std::size_t logical_n_x = 0;
  std::size_t logical_n_y = 0;
  std::size_t padded_n_x = 0;
  std::size_t padded_n_y = 0;
  std::uint32_t delta_x_bits = 0;
  std::uint32_t delta_y_bits = 0;
  std::uint32_t delta_z_bits = 0;
  std::uint32_t charge_sign_bits = 0;
  std::uint64_t field_kernel_factor_bits = 0;
  // Green construction uses one in-place FP64 cuFFT with explicit workspace.
  std::uint32_t spectrum_generator_id = 1;
  int spectrum_generator_version = 0;
  bool explicit_workspace = true;
  std::uint64_t build_abi_hash = kExecutorAbiHashV1;

  bool operator==(const GreenSpectrumCacheKey& other) const noexcept;
  bool operator<(const GreenSpectrumCacheKey& other) const noexcept;
  std::uint64_t stable_hash() const noexcept;
};

struct WindowClassShape {
  std::size_t local_n_x = 0;
  std::size_t local_n_y = 0;

  bool operator==(const WindowClassShape& other) const noexcept {
    return local_n_x == other.local_n_x && local_n_y == other.local_n_y;
  }
  bool operator<(const WindowClassShape& other) const noexcept {
    return local_n_x < other.local_n_x ||
           (local_n_x == other.local_n_x && local_n_y < other.local_n_y);
  }
};

struct AdaptiveGpuOptions {
  int device = 0;
  std::size_t batch_capacity = 1;
  std::size_t maximum_beam1_descriptors = 1;
  std::size_t maximum_beam2_descriptors = 1;
  double memory_fraction = 0.80;
  // Zero means that memory_fraction is the only configured device cap.
  std::size_t maximum_device_bytes = 0;
  bool pinned_io = true;
  DepositionVariantId deposition = DepositionVariantId::GlobalAtomic;
  std::uint64_t build_abi_hash = kExecutorAbiHashV1;
};

// Snapshot identity supplied by the wave scheduler.  Revisions and order
// tokens must be sampled before field work and supplied again when a result is
// consumed.  No coordinate hash is required in the hot path.
struct AdaptiveEncounterBinding {
  std::size_t canonical_result_slot = 0;
  std::uint64_t encounter_id = 0;
  std::int64_t beam1_slice = -1;
  std::int64_t beam2_slice = -1;
  std::uint64_t beam1_revision = 0;
  std::uint64_t beam2_revision = 0;
  std::uint64_t beam1_order_token = 0;
  std::uint64_t beam2_order_token = 0;
};

struct AdaptiveEncounterSubmission {
  const UniformWindowPlan* plan = nullptr;
  AdaptiveEncounterBinding binding;
};

struct AdaptiveResultBinding {
  AdaptiveEncounterBinding encounter;
  std::uint64_t lattice_hash = 0;
  std::uint64_t geometry_revision = 0;
  std::uint64_t window_plan_hash = 0;
  CanonicalIndex origin_x = 0;
  CanonicalIndex origin_y = 0;
  std::size_t local_n_x = 0;
  std::size_t local_n_y = 0;
  std::size_t padded_n_x = 0;
  std::size_t padded_n_y = 0;
  std::size_t beam1_particle_count = 0;
  std::size_t beam2_particle_count = 0;
  std::uint64_t beam1_macro_charge_bits = 0;
  std::uint64_t beam2_macro_charge_bits = 0;
  std::uint64_t fft_plan_key_hash = 0;
  std::uint64_t green_spectrum_key_hash = 0;
  std::uint64_t binding_hash = 0;
};

struct AdaptiveEncounterResult {
  bool valid = false;
  AdaptiveResultBinding binding;
  std::vector<float> fields1;
  std::vector<float> fields2;
  std::size_t beam1_in = 0;
  std::size_t beam1_out = 0;
  std::size_t beam2_in = 0;
  std::size_t beam2_out = 0;
  std::size_t gather_beam1_in = 0;
  std::size_t gather_beam1_out = 0;
  std::size_t gather_beam2_in = 0;
  std::size_t gather_beam2_out = 0;
  double fine_lumi_dot = 0.0;

  void reset() noexcept;
  void swap(AdaptiveEncounterResult& other) noexcept;
};

struct AdaptiveGpuTelemetry {
  std::uint64_t prebuild_calls = 0;
  std::uint64_t classes_prebuilt = 0;
  std::uint64_t plan_cache_hits = 0;
  std::uint64_t plan_cache_misses = 0;
  std::uint64_t green_cache_hits = 0;
  std::uint64_t green_cache_misses = 0;
  std::uint64_t plan_creations = 0;
  std::uint64_t green_creations = 0;
  std::uint64_t batch_calls = 0;
  std::uint64_t encounters = 0;
  std::uint64_t empty_encounters = 0;
  std::uint64_t full_window_encounters = 0;
  std::uint64_t required_node_visits = 0;
  std::uint64_t selected_local_cell_visits = 0;
  std::uint64_t padded_fft_cell_visits = 0;
  std::uint64_t actual_planmany_cell_visits = 0;
  std::uint64_t canonical_cells_avoided = 0;
  std::uint64_t padded_fft_slots = 0;
  std::uint64_t h2d_operations = 0;
  std::uint64_t d2h_operations = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t kernel_launches = 0;
  std::uint64_t cufft_executes = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t deposit_beam1_in = 0;
  std::uint64_t deposit_beam1_out = 0;
  std::uint64_t deposit_beam2_in = 0;
  std::uint64_t deposit_beam2_out = 0;
  std::uint64_t gather_beam1_in = 0;
  std::uint64_t gather_beam1_out = 0;
  std::uint64_t gather_beam2_in = 0;
  std::uint64_t gather_beam2_out = 0;
  std::uint64_t containment_failures = 0;
  std::uint64_t binding_failures = 0;
  std::size_t memory_budget_bytes = 0;
  std::size_t tracked_device_bytes = 0;
  std::size_t peak_tracked_device_bytes = 0;
  std::size_t measured_device_high_water_bytes = 0;
  std::size_t pinned_host_bytes = 0;
  std::size_t plan_workspace_bytes = 0;
  double setup_ms = 0.0;
  double plan_build_ms = 0.0;
  double green_build_ms = 0.0;
  double host_pack_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double host_publish_ms = 0.0;
  DepositionVariantId requested_deposition =
      DepositionVariantId::GlobalAtomic;
  DepositionVariantId actual_deposition =
      DepositionVariantId::GlobalAtomic;
  bool prebuild_complete = false;
  bool context_poisoned = false;
};

const char* deposition_variant_name(DepositionVariantId variant) noexcept;

// One executor is bound to one immutable canonical lattice and one CUDA
// device.  All allowed window classes must be prebuilt before execute_batch;
// an unbuilt class is an error, never a fixed-grid/mode-4/CPU fallback.
class AdaptiveWaveBatchExecutor {
 public:
  AdaptiveWaveBatchExecutor();
  ~AdaptiveWaveBatchExecutor();

  AdaptiveWaveBatchExecutor(const AdaptiveWaveBatchExecutor&) = delete;
  AdaptiveWaveBatchExecutor& operator=(const AdaptiveWaveBatchExecutor&) =
      delete;

  bool initialize(const AdaptiveGpuOptions& options,
                  const CanonicalLatticeSnapshot& lattice,
                  std::string& error);

  bool prebuild_classes(const WindowClassShape* classes,
                        std::size_t class_count,
                        std::string& error);

  bool execute_batch(const AdaptiveEncounterSubmission* submissions,
                     std::size_t encounter_count,
                     double beam1_macro_charge,
                     double beam2_macro_charge,
                     AdaptiveEncounterResult* canonical_results,
                     std::size_t canonical_result_capacity,
                     std::string& error);

  // Re-evaluate the complete result binding immediately before consumption.
  // The expected submission should contain current slice revisions/count/order
  // tokens and the same immutable window plan; expected charges are compared
  // by exact IEEE-754 bits against the values used by execute_batch.
  bool validate_result_binding(const AdaptiveEncounterResult& result,
                               const AdaptiveEncounterSubmission& expected,
                               double expected_beam1_macro_charge,
                               double expected_beam2_macro_charge,
                               std::string& error) const;

  bool fft_plan_key(const WindowClassShape& shape,
                    FftPlanCacheKey& key,
                    std::string& error) const;
  bool green_spectrum_key(const WindowClassShape& shape,
                          GreenSpectrumCacheKey& key,
                          std::string& error) const;

  bool initialized() const noexcept;
  bool prebuild_complete() const noexcept;
  bool poisoned() const noexcept;
  DeviceIdentity device_identity() const;
  AdaptiveGpuTelemetry telemetry() const;
  std::string device_description() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace gpu5
}  // namespace adaptive_grid
}  // namespace feebl

#endif  // FEEBL_GPU_ADAPTIVE_WAVE_BATCH_SEEN
