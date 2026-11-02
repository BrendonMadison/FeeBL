#ifndef FEEBL_GPU_MODE3_PRODUCTION_CPP_SEEN
#define FEEBL_GPU_MODE3_PRODUCTION_CPP_SEEN

#include "gpuPrimaryModernAdapter.h"
#include "modernLegacyWritebackCPP.h"

#include <cstdint>
#include <string>

class BEAM;
class BEAM_PARAMETERS;
class GENERAL_GRID;
class SWITCHES;

namespace feebl {
namespace gpu {

// Card-only gate used immediately after backend selection, before beam
// generation or any CUDA interaction.  The execution entry point repeats it
// defensively after the final grid has been constructed.
void preflight_mode3_card_controls(const SWITCHES& switches);

// Receipt returned only after both legacy beams have been committed.  It is
// deliberately crossing-scoped: there is no production API that can publish
// a one-beam GPU result.
struct Mode3ProductionReceipt {
  Mode3PrimaryCrossingReport crossing;
  modern::LegacyCrossingWritebackReceipt writeback;
  std::uint64_t diagonals = 0;
  std::uint64_t waves = 0;
  std::uint64_t encounters = 0;
  std::uint64_t maximum_wave_width = 0;
  bool import_contracts_bound_before_cuda = false;
  bool paired_writeback_preflight_before_cuda = false;
  // Computed and finite-validated after CUDA, before legacy mutation.  The
  // caller publishes it to RESULTS only after the paired write-back receipt
  // proves both beams committed.
  double normalized_fine_lumi = 0.0;
  double bunch_population1 = 0.0;
  double bunch_population2 = 0.0;
  double macro_charge1 = 0.0;
  double macro_charge2 = 0.0;
  double effective_population1 = 0.0;
  double effective_population2 = 0.0;
  std::uint64_t declared_macro_particles1 = 0;
  std::uint64_t declared_macro_particles2 = 0;
  std::string initialization_rng;
  std::string collision_rng;
};

// Execute the currently qualified production subset.  Every unsupported
// switch is rejected before CUDA allocation/upload.  Generated particles are
// imported with the exact bunch population / declared n_m / macro-charge
// normalization used by FeeBL, all encounter waves execute resident, and the
// two legacy beams are then updated transactionally.
Mode3ProductionReceipt execute_mode3_primary_crossing(
    BEAM& beam1,
    BEAM& beam2,
    const BEAM_PARAMETERS& parameters1,
    const BEAM_PARAMETERS& parameters2,
    GENERAL_GRID& grid,
    const SWITCHES& switches,
    const std::string& input_card_label,
    int cuda_device,
    double memory_fraction);

}  // namespace gpu
}  // namespace feebl

#endif  // FEEBL_GPU_MODE3_PRODUCTION_CPP_SEEN
