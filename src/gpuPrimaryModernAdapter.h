#ifndef FEEBL_GPU_PRIMARY_MODERN_ADAPTER_SEEN
#define FEEBL_GPU_PRIMARY_MODERN_ADAPTER_SEEN

#include "beamSoACPP.h"
#include "encounterDAGCPP.h"
#include "gpuPrimaryResident.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace feebl {
namespace gpu {

// Mode 3 is an explicit, abort-only production contract.  AllowFallback is
// represented only so preflight can reject an unsafe selector configuration;
// this adapter never contains a serial implementation or invokes one after a
// CUDA failure.
enum class Mode3FailurePolicy : std::uint8_t {
  AbortNoFallback = 0,
  AllowFallback = 1
};

// All values here are immutable crossing-level inputs.  The Green spectrum is
// the already transformed and normalized FIELD::dist_init spectrum in
// interleaved FP64-complex layout.  spectrum_charge_sign records the sign used
// to construct it; the CUDA field solver cannot infer or repair a mismatch.
struct Mode3PrimaryCrossingConfig {
  int requested_gpu_mode = 3;
  Mode3FailurePolicy failure_policy = Mode3FailurePolicy::AbortNoFallback;
  ResidentPrimaryOptions resident_options;
  // Required card-level qualification declarations.  The present path uses
  // FP32 legacy particle phase space with FP64 grids/FFT (mixed) and unordered
  // FP64 atomic CIC accumulation (fast rather than deterministic reduction).
  std::string precision;
  std::string modern_reduction;

  int integration_method = 2;
  int nx = 0;
  int ny = 0;
  const double* green_spectrum_interleaved = nullptr;
  std::size_t green_spectrum_interleaved_values = 0;
  // Mandatory provenance copied from the FIELD/Grid that constructed the
  // spectrum.  Mode 3 rejects a spectrum whose mesh/sign differs from run().
  bool green_spectrum_provenance_declared = false;
  int spectrum_integration_method = 2;
  float spectrum_charge_sign = -1.0f;
  ResidentPrimaryGrid spectrum_grid;
  ResidentPrimaryGrid grid;

  float step = 0.0f;
  float scale1 = 1.0f;
  float scale2 = 1.0f;
  float charge_sign = -1.0f;
  // Particle generation symmetry (`charge_symmetric`) controls stable-ID
  // provenance.  It is independent of density symmetrization
  // (`force_symmetric`), represented by four_way_symmetry below.
  bool charge_symmetric_generation = false;
  bool four_way_symmetry = false;

  std::string input_card_label;
  std::uint64_t crossing_index = 0;
};

// Structural preflight performs no allocation and launches no CUDA work.  It
// describes the exact mapping that execute() will use.  Device/memory
// preflight remains a separate, mandatory stage inside pipeline.initialize().
struct Mode3PrimaryPreflight {
  bool passed = false;
  std::size_t particle_count1 = 0;
  std::size_t particle_count2 = 0;
  std::uint64_t slice_count1 = 0;
  std::uint64_t slice_count2 = 0;
  std::uint64_t substeps = 0;
  std::uint64_t waves = 0;
  std::uint64_t encounters = 0;
  std::size_t expected_green_spectrum_values = 0;
  double uniform_macro_weight1 = 0.0;
  double uniform_macro_weight2 = 0.0;
  std::uint64_t green_spectrum_fingerprint = 0;
  std::uint64_t schedule_fingerprint = 0;
  std::uint64_t invariant_layout_fingerprint = 0;
};

// Auditable receipt for one whole crossing.  The counters are copied from the
// resident component, then checked before BeamSoA is mutated.  "Mapped" means
// x/y/vx/vy/energy/Upsilon were resident and mutable; z, identity, slice CSR,
// weights, spin/species/direction/flags stay host-authoritative invariants.
struct Mode3PrimaryCrossingReport {
  std::string implementation_id;
  std::string input_card_label;
  std::string device_description;
  std::string runtime_description;
  std::string precision;
  std::string modern_reduction;
  std::uint64_t crossing_index = 0;
  int requested_gpu_mode = 0;
  int selected_gpu_mode = 0;
  bool fallback_permitted = false;
  bool fallback_used = false;
  bool structural_preflight_passed = false;
  bool device_preflight_passed = false;
  bool state_uploaded_once = false;
  bool all_waves_submitted_once = false;
  bool state_downloaded_once = false;
  bool outputs_committed = false;
  bool slice_offsets_and_order_preserved = false;
  bool host_invariants_preserved = false;
  std::uint32_t mapped_mutable_components = 0;
  std::uint32_t host_invariant_components = 0;
  std::uint64_t green_spectrum_fingerprint = 0;
  std::uint64_t schedule_fingerprint = 0;
  std::uint64_t invariant_layout_fingerprint_before = 0;
  std::uint64_t invariant_layout_fingerprint_after = 0;
  Mode3PrimaryPreflight preflight;
  ResidentPrimaryStats resident;
};

// One adapter instance is one crossing.  Reuse is rejected so its receipt can
// prove exactly one upload/run/download lifecycle without counter ambiguity.
class Mode3PrimaryCrossingAdapter {
 public:
  Mode3PrimaryCrossingAdapter();
  ~Mode3PrimaryCrossingAdapter();

  Mode3PrimaryCrossingAdapter(const Mode3PrimaryCrossingAdapter&) = delete;
  Mode3PrimaryCrossingAdapter& operator=(
      const Mode3PrimaryCrossingAdapter&) = delete;

  bool preflight(const modern::BeamSoA& beam1,
                 const modern::BeamSoA& beam2,
                 const modern::EncounterDAG& encounters,
                 const Mode3PrimaryCrossingConfig& config,
                 Mode3PrimaryPreflight& result,
                 std::string& error) const;

  // Transactional with respect to BeamSoA: no component changes until CUDA
  // initialization, upload, all waves, final download, output validation, and
  // residency-counter verification have all succeeded.
  bool execute(modern::BeamSoA& beam1,
               modern::BeamSoA& beam2,
               const modern::EncounterDAG& encounters,
               const Mode3PrimaryCrossingConfig& config,
               Mode3PrimaryCrossingReport& report,
               std::string& error);

 private:
  ResidentPrimaryPipeline pipeline_;
  bool execution_attempted_;
};

const char* mode3_primary_implementation_id() noexcept;

}  // namespace gpu
}  // namespace feebl

#endif  // FEEBL_GPU_PRIMARY_MODERN_ADAPTER_SEEN
