#include "gpuPrimaryModernAdapter.h"

#include "primaryParticleIdCPP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace feebl {
namespace gpu {

namespace {

const char kImplementationId[] =
    "feebl.gpu.primary-resident.mode3.v1";
const std::uint64_t kFnvOffset = 1469598103934665603ull;
const std::uint64_t kFnvPrime = 1099511628211ull;

struct PreparedCrossing {
  Mode3PrimaryPreflight receipt;
  std::vector<ResidentPrimaryEncounter> schedule;
};

void hash_byte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= static_cast<std::uint64_t>(value);
  hash *= kFnvPrime;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) {
  // Specify byte order explicitly so provenance does not depend on host
  // endianness.  This is an audit fingerprint, not a cryptographic digest.
  for (unsigned int byte = 0; byte < 8; ++byte)
    hash_byte(hash, static_cast<std::uint8_t>(value >> (8u * byte)));
}

void hash_double(std::uint64_t& hash, double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected FP64 size");
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

void hash_float(std::uint64_t& hash, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected FP32 size");
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

bool checked_size_product(std::size_t left,
                          std::size_t right,
                          std::size_t& result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

bool checked_size_sum(std::size_t left,
                      std::size_t right,
                      std::size_t& result) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    return false;
  result = left + right;
  return true;
}

bool size_to_u64(std::size_t value, std::uint64_t& result) {
  if (value > static_cast<std::size_t>(
                  std::numeric_limits<std::uint64_t>::max()))
    return false;
  result = static_cast<std::uint64_t>(value);
  return true;
}

bool exact_legacy_float(double value, float& narrowed) {
  narrowed = static_cast<float>(value);
  return std::isfinite(value) && std::isfinite(narrowed) &&
         static_cast<double>(narrowed) == value;
}

bool primary_species(feebl::modern::ParticleSpecies species) {
  return species == modern::ParticleSpecies::Electron ||
         species == modern::ParticleSpecies::Positron;
}

bool same_grid(const ResidentPrimaryGrid& first,
               const ResidentPrimaryGrid& second) {
  return first.delta_x_inverse == second.delta_x_inverse &&
         first.delta_y_inverse == second.delta_y_inverse &&
         first.offset_x == second.offset_x &&
         first.offset_y == second.offset_y &&
         first.physical_min_x == second.physical_min_x &&
         first.physical_max_x == second.physical_max_x &&
         first.physical_min_y == second.physical_min_y &&
         first.physical_max_y == second.physical_max_y;
}

std::uint64_t invariant_beam_fingerprint(const modern::BeamSoA& beam,
                                         std::uint64_t beam_marker) {
  std::uint64_t hash = kFnvOffset;
  hash_u64(hash, beam_marker);
  hash_u64(hash, beam.slice_count());
  hash_u64(hash, beam.size());
  hash_u64(hash, static_cast<std::uint64_t>(beam.spin_enabled()));
  hash_u64(hash, static_cast<std::uint64_t>(beam.metadata().units));
  hash_double(hash, beam.metadata().bunch_population);
  for (modern::ArrayOffset offset : beam.slice_offsets()) hash_u64(hash, offset);
  for (std::size_t index = 0; index < beam.stable_ids().size(); ++index) {
    hash_u64(hash, beam.stable_ids()[index]);
    hash_double(hash, beam.z()[index]);
    hash_double(hash, beam.macro_weight()[index]);
    if (beam.spin_enabled()) {
      hash_double(hash, beam.sx()[index]);
      hash_double(hash, beam.sy()[index]);
      hash_double(hash, beam.sz()[index]);
    }
    hash_u64(hash, static_cast<std::uint64_t>(beam.species()[index]));
    hash_u64(hash, static_cast<std::uint8_t>(beam.charge_e()[index]));
    hash_u64(hash, static_cast<std::uint8_t>(beam.direction()[index]));
    hash_u64(hash, beam.beam_id()[index]);
    hash_u64(hash, beam.flags()[index]);
  }
  return hash;
}

std::uint64_t invariant_layout_fingerprint(const modern::BeamSoA& beam1,
                                           const modern::BeamSoA& beam2) {
  const std::uint64_t first = invariant_beam_fingerprint(beam1, 1u);
  const std::uint64_t second = invariant_beam_fingerprint(beam2, 2u);
  std::uint64_t combined = kFnvOffset;
  hash_u64(combined, first);
  hash_u64(combined, second);
  return combined;
}

bool validate_generated_beam(const modern::BeamSoA& beam,
                             unsigned int expected_beam_label,
                             bool charge_symmetric_generation,
                             double& uniform_weight,
                             std::string& error) {
  const modern::ValidationResult structural = beam.validate();
  if (!structural) {
    std::ostringstream message;
    message << "beam " << expected_beam_label
            << " BeamSoA validation failed at " << structural.index
            << ": " << structural.message;
    error = message.str();
    return false;
  }
  if (beam.spin_enabled()) {
    error = "gpu_mode=3 resident primary domain does not implement spin transport";
    return false;
  }
  if (beam.empty()) {
    error = "gpu_mode=3 resident primary domain requires both beams to contain primaries";
    return false;
  }
  if (beam.metadata().units != modern::UnitConvention::LegacyFeeBLNmGeV ||
      !std::isfinite(beam.metadata().bunch_population) ||
      beam.metadata().bunch_population <= 0.0) {
    error = "gpu_mode=3 requires positive-population FeeBL nm/GeV BeamSoA metadata";
    return false;
  }

  uniform_weight = beam.macro_weight().front();
  float legacy_weight = 0.0f;
  if (uniform_weight <= 0.0 ||
      !exact_legacy_float(uniform_weight, legacy_weight)) {
    error = "gpu_mode=3 requires a positive legacy-float-exact macro-particle weight";
    return false;
  }

  const std::int8_t expected_direction =
      expected_beam_label == 1u ? std::int8_t{1} : std::int8_t{-1};
  for (std::size_t index = 0; index < beam.stable_ids().size(); ++index) {
    PRIMARY_PARTICLE_ID_FIELDS identity;
    if (!decode_primary_particle_id(beam.stable_ids()[index], identity) ||
        identity.beam_label != expected_beam_label) {
      std::ostringstream message;
      message << "gpu_mode=3 requires a valid source-stable generated ID for beam "
              << expected_beam_label << " at particle " << index;
      error = message.str();
      return false;
    }
    const PRIMARY_PARTICLE_PROVENANCE required_provenance =
        charge_symmetric_generation
            ? PRIMARY_PARTICLE_PROVENANCE::GENERATED_SYMMETRIC
            : PRIMARY_PARTICLE_PROVENANCE::GENERATED_ORDINARY;
    if (identity.provenance != required_provenance) {
      std::ostringstream message;
      message << "gpu_mode=3 generated-primary provenance disagrees with charge_symmetric generation at beam "
              << expected_beam_label << " particle " << index;
      error = message.str();
      return false;
    }
    if (beam.beam_id()[index] != expected_beam_label ||
        beam.direction()[index] != expected_direction ||
        !primary_species(beam.species()[index]) ||
        beam.flags()[index] != 0u) {
      std::ostringstream message;
      message << "gpu_mode=3 primary metadata is unsupported at beam "
              << expected_beam_label << " particle " << index;
      error = message.str();
      return false;
    }
    if (beam.macro_weight()[index] != uniform_weight) {
      std::ostringstream message;
      message << "gpu_mode=3 CIC currently requires uniform macro weights; beam "
              << expected_beam_label << " differs at particle " << index;
      error = message.str();
      return false;
    }

    float narrowed = 0.0f;
    const bool phase_space_exact =
        exact_legacy_float(beam.x()[index], narrowed) &&
        exact_legacy_float(beam.y()[index], narrowed) &&
        exact_legacy_float(beam.vx()[index], narrowed) &&
        exact_legacy_float(beam.vy()[index], narrowed) &&
        exact_legacy_float(beam.energy()[index], narrowed) &&
        beam.energy()[index] > 0.0 &&
        exact_legacy_float(beam.upsilon()[index], narrowed) &&
        beam.upsilon()[index] >= 0.0;
    if (!phase_space_exact) {
      std::ostringstream message;
      message << "gpu_mode=3 requires x/y/vx/vy/energy/Upsilon to round-trip exactly through legacy float at beam "
              << expected_beam_label << " particle " << index;
      error = message.str();
      return false;
    }
  }
  return true;
}

bool validate_grid_and_spectrum(const Mode3PrimaryCrossingConfig& config,
                                std::size_t& expected_values,
                                std::uint64_t& spectrum_fingerprint,
                                std::string& error) {
  if (config.integration_method != 2) {
    error = "gpu_mode=3 resident primary domain requires FFT integration_method=2";
    return false;
  }
  if (!config.green_spectrum_provenance_declared ||
      config.spectrum_integration_method != config.integration_method) {
    error = "gpu_mode=3 requires declared Green-spectrum mesh/sign/integration provenance";
    return false;
  }
  if (config.nx < 3 || config.ny < 3 ||
      config.nx > std::numeric_limits<int>::max() / 2 ||
      config.ny > std::numeric_limits<int>::max() / 2) {
    error = "gpu_mode=3 requires representable field dimensions nx,ny >= 3";
    return false;
  }
  if (config.four_way_symmetry &&
      ((config.nx & 1) != 0 || (config.ny & 1) != 0)) {
    error = "gpu_mode=3 four-way density symmetry requires even nx and ny";
    return false;
  }

  std::size_t padded_cells = 0;
  if (!checked_size_product(static_cast<std::size_t>(2 * config.nx),
                            static_cast<std::size_t>(2 * config.ny),
                            padded_cells) ||
      !checked_size_product(padded_cells, static_cast<std::size_t>(2),
                            expected_values)) {
    error = "gpu_mode=3 Green spectrum dimensions overflow size_t";
    return false;
  }
  if (!config.green_spectrum_interleaved ||
      config.green_spectrum_interleaved_values != expected_values) {
    std::ostringstream message;
    message << "gpu_mode=3 requires exactly " << expected_values
            << " interleaved Green-spectrum values";
    error = message.str();
    return false;
  }
  spectrum_fingerprint = kFnvOffset;
  for (std::size_t index = 0; index < expected_values; ++index) {
    const double value = config.green_spectrum_interleaved[index];
    if (!std::isfinite(value)) {
      std::ostringstream message;
      message << "gpu_mode=3 Green spectrum contains a non-finite value at "
              << index;
      error = message.str();
      return false;
    }
    hash_double(spectrum_fingerprint, value);
  }

  if (!std::isfinite(config.charge_sign) ||
      !(config.charge_sign == -1.0f || config.charge_sign == 0.0f)) {
    error = "gpu_mode=3 currently supports charge_sign=-1 (e+e-) or 0 (zero field), not like-sign beams";
    return false;
  }
  if (config.spectrum_charge_sign != config.charge_sign) {
    error = "gpu_mode=3 Green-spectrum charge-sign provenance does not match charge_sign";
    return false;
  }
  if (!same_grid(config.spectrum_grid, config.grid)) {
    error = "gpu_mode=3 Green-spectrum mesh provenance does not match the run mesh";
    return false;
  }
  if (!std::isfinite(config.step) || config.step <= 0.0f ||
      !std::isfinite(config.scale1) || config.scale1 <= 0.0f ||
      !std::isfinite(config.scale2) || config.scale2 <= 0.0f) {
    error = "gpu_mode=3 requires finite positive step and per-beam scale_step values";
    return false;
  }
  if (!std::isfinite(config.step * config.scale1) ||
      !std::isfinite(config.step * config.scale2)) {
    error = "gpu_mode=3 step*scale_step overflows legacy float arithmetic";
    return false;
  }
  const float dz1 = static_cast<float>(
      static_cast<double>(config.step) * 1.0e-9 *
      static_cast<double>(config.scale1));
  const float dz2 = static_cast<float>(
      static_cast<double>(config.step) * 1.0e-9 *
      static_cast<double>(config.scale2));
  if (!std::isfinite(dz1) || dz1 <= 0.0f ||
      !std::isfinite(dz2) || dz2 <= 0.0f) {
    error = "gpu_mode=3 step*1e-9*scale_step is not a positive representable legacy dz";
    return false;
  }
  const ResidentPrimaryGrid& grid = config.grid;
  if (!std::isfinite(grid.delta_x_inverse) ||
      !std::isfinite(grid.delta_y_inverse) ||
      !std::isfinite(grid.offset_x) || !std::isfinite(grid.offset_y) ||
      !std::isfinite(grid.physical_min_x) ||
      !std::isfinite(grid.physical_max_x) ||
      !std::isfinite(grid.physical_min_y) ||
      !std::isfinite(grid.physical_max_y) ||
      grid.delta_x_inverse <= 0.0f || grid.delta_y_inverse <= 0.0f ||
      !(grid.physical_min_x < grid.physical_max_x) ||
      !(grid.physical_min_y < grid.physical_max_y)) {
    error = "gpu_mode=3 requires finite positive mesh spacing and ordered physical bounds";
    return false;
  }
  return true;
}

bool validate_initial_grid_transforms(const modern::BeamSoA& beam,
                                      const ResidentPrimaryGrid& grid,
                                      unsigned int beam_label,
                                      std::string& error) {
  for (std::size_t index = 0; index < beam.x().size(); ++index) {
    const float x = static_cast<float>(beam.x()[index]);
    const float y = static_cast<float>(beam.y()[index]);
    const float cic_x =
        x * grid.delta_x_inverse + (grid.offset_x - 0.5f);
    const float cic_y =
        y * grid.delta_y_inverse + (grid.offset_y - 0.5f);
    const float field_x = x * grid.delta_x_inverse + grid.offset_x;
    const float field_y = y * grid.delta_y_inverse + grid.offset_y;
    if (!std::isfinite(cic_x) || !std::isfinite(cic_y) ||
        !std::isfinite(field_x) || !std::isfinite(field_y)) {
      std::ostringstream message;
      message << "gpu_mode=3 initial coordinate-to-grid transform overflows at beam "
              << beam_label << " particle " << index;
      error = message.str();
      return false;
    }
  }
  return true;
}

bool prepare_crossing(const modern::BeamSoA& beam1,
                      const modern::BeamSoA& beam2,
                      const modern::EncounterDAG& dag,
                      const Mode3PrimaryCrossingConfig& config,
                      PreparedCrossing& prepared,
                      std::string& error) {
  error.clear();
  prepared = PreparedCrossing();
  if (config.requested_gpu_mode != 3) {
    error = "resident primary production adapter is selectable only as gpu_mode=3";
    return false;
  }
  if (config.failure_policy != Mode3FailurePolicy::AbortNoFallback) {
    error = "gpu_mode=3 is abort-only: CPU/GPU fallback after selection is forbidden";
    return false;
  }
  if (config.precision != "mixed_qualified") {
    error = "gpu_mode=3 requires precision=\"mixed_qualified\": particle phase space is FP32 while CIC/FFT fields are FP64";
    return false;
  }
  if (config.modern_reduction != "fast_qualified") {
    error = "gpu_mode=3 requires modern_reduction=\"fast_qualified\": FP64 atomic CIC accumulation is unordered";
    return false;
  }
  if (config.input_card_label.empty()) {
    error = "gpu_mode=3 requires a nonempty input-card provenance label";
    return false;
  }
  std::string capability_reason;
  if (!resident_primary_contract_supported(config.resident_options.contract,
                                           capability_reason)) {
    error = std::string("gpu_mode=3 unsupported physics contract: ") +
            capability_reason;
    return false;
  }
  if (config.resident_options.device < 0 ||
      !std::isfinite(config.resident_options.memory_fraction) ||
      config.resident_options.memory_fraction <= 0.0 ||
      config.resident_options.memory_fraction > 1.0) {
    error = "gpu_mode=3 requires a nonnegative CUDA device and 0<memory_fraction<=1";
    return false;
  }
  if (!config.resident_options.pinned_io) {
    error = "gpu_mode=3 production residency accounting requires pinned_io=1";
    return false;
  }

  std::size_t expected_spectrum_values = 0;
  std::uint64_t spectrum_fingerprint = 0;
  if (!validate_grid_and_spectrum(config, expected_spectrum_values,
                                  spectrum_fingerprint, error))
    return false;

  const modern::ValidationResult dag_check = dag.validate();
  if (!dag_check) {
    std::ostringstream message;
    message << "gpu_mode=3 EncounterDAG validation failed at "
            << dag_check.index << ": " << dag_check.message;
    error = message.str();
    return false;
  }
  if (!dag.is_legacy_antidiagonal()) {
    error = "gpu_mode=3 currently supports only the canonical legacy anti-diagonal EncounterDAG";
    return false;
  }
  if (dag.encounter_count() == 0 || dag.wave_count() == 0) {
    error = "gpu_mode=3 requires a nonempty collision EncounterDAG";
    return false;
  }
  if (dag.beam1_slice_count() != beam1.slice_count() ||
      dag.beam2_slice_count() != beam2.slice_count()) {
    error = "gpu_mode=3 EncounterDAG slice counts do not match the two BeamSoA layouts";
    return false;
  }
  if (!modern::fits_size_t(beam1.size()) ||
      !modern::fits_size_t(beam2.size()) ||
      beam1.size() > std::numeric_limits<std::size_t>::max() / 2u ||
      beam2.size() > std::numeric_limits<std::size_t>::max() / 2u ||
      !modern::fits_size_t(dag.encounter_count())) {
    error = "gpu_mode=3 BeamSoA or EncounterDAG extent is not representable by CUDA host arrays";
    return false;
  }
  if (dag.wave_count() > std::numeric_limits<std::uint32_t>::max()) {
    error = "gpu_mode=3 EncounterDAG wave index exceeds resident uint32 representation";
    return false;
  }

  double weight1 = 0.0;
  double weight2 = 0.0;
  if (!validate_generated_beam(beam1, 1u,
                               config.charge_symmetric_generation,
                               weight1, error) ||
      !validate_generated_beam(beam2, 2u,
                               config.charge_symmetric_generation,
                               weight2, error))
    return false;
  if (!validate_initial_grid_transforms(beam1, config.grid, 1u, error) ||
      !validate_initial_grid_transforms(beam2, config.grid, 2u, error))
    return false;
  if (beam1.charge_e().front() * beam2.charge_e().front() != -1) {
    error = "gpu_mode=3 currently requires opposite-charge e+e- primary beams";
    return false;
  }
  for (std::int8_t charge : beam1.charge_e())
    if (charge != beam1.charge_e().front()) {
      error = "gpu_mode=3 requires one uniform primary species/charge per beam";
      return false;
    }
  for (std::int8_t charge : beam2.charge_e())
    if (charge != beam2.charge_e().front()) {
      error = "gpu_mode=3 requires one uniform primary species/charge per beam";
      return false;
    }

  try {
    prepared.schedule.reserve(
        static_cast<std::size_t>(dag.encounter_count()));
    std::uint64_t schedule_hash = kFnvOffset;
    hash_u64(schedule_hash, dag.beam1_slice_count());
    hash_u64(schedule_hash, dag.beam2_slice_count());
    hash_u64(schedule_hash, dag.substep_count());
    hash_double(schedule_hash, weight1);
    hash_double(schedule_hash, weight2);
    hash_float(schedule_hash, config.step);
    hash_float(schedule_hash, config.scale1);
    hash_float(schedule_hash, config.scale2);
    hash_float(schedule_hash, config.charge_sign);
    hash_u64(schedule_hash, static_cast<std::uint64_t>(
                                config.charge_symmetric_generation));
    hash_u64(schedule_hash, static_cast<std::uint64_t>(
                                config.four_way_symmetry));

    for (const modern::EncounterWave& wave : dag.waves()) {
      if (wave.index > std::numeric_limits<std::uint32_t>::max()) {
        error = "gpu_mode=3 EncounterDAG wave index exceeds resident uint32 representation";
        return false;
      }
      for (modern::EncounterId id : wave.encounters) {
        const modern::EncounterNode& node = dag.node(id);
        const std::pair<modern::ArrayOffset, modern::ArrayOffset> range1 =
            beam1.slice_range(node.beam1_slice);
        const std::pair<modern::ArrayOffset, modern::ArrayOffset> range2 =
            beam2.slice_range(node.beam2_slice);
        if (!modern::fits_size_t(range1.first) ||
            !modern::fits_size_t(range1.second) ||
            !modern::fits_size_t(range2.first) ||
            !modern::fits_size_t(range2.second)) {
          error = "gpu_mode=3 slice offset is not representable by resident size_t";
          return false;
        }
        ResidentPrimaryEncounter mapped;
        mapped.wave_index = static_cast<std::uint32_t>(wave.index);
        mapped.beam1_offset = static_cast<std::size_t>(range1.first);
        mapped.beam1_count =
            static_cast<std::size_t>(range1.second - range1.first);
        mapped.beam2_offset = static_cast<std::size_t>(range2.first);
        mapped.beam2_count =
            static_cast<std::size_t>(range2.second - range2.first);
        mapped.charge1 = weight1;
        mapped.charge2 = weight2;
        mapped.step = config.step;
        mapped.scale1 = config.scale1;
        mapped.scale2 = config.scale2;
        mapped.charge_sign = config.charge_sign;
        mapped.four_way_symmetry = config.four_way_symmetry;
        prepared.schedule.push_back(mapped);

        hash_u64(schedule_hash, id);
        hash_u64(schedule_hash, node.beam1_slice);
        hash_u64(schedule_hash, node.beam2_slice);
        hash_u64(schedule_hash, node.substep);
        hash_u64(schedule_hash, wave.index);
        hash_u64(schedule_hash, range1.first);
        hash_u64(schedule_hash, range1.second);
        hash_u64(schedule_hash, range2.first);
        hash_u64(schedule_hash, range2.second);
      }
    }
    if (prepared.schedule.size() != dag.encounter_count()) {
      error = "gpu_mode=3 EncounterDAG mapping did not preserve every encounter";
      return false;
    }

    prepared.receipt.passed = true;
    prepared.receipt.particle_count1 =
        static_cast<std::size_t>(beam1.size());
    prepared.receipt.particle_count2 =
        static_cast<std::size_t>(beam2.size());
    prepared.receipt.slice_count1 = beam1.slice_count();
    prepared.receipt.slice_count2 = beam2.slice_count();
    prepared.receipt.substeps = dag.substep_count();
    prepared.receipt.waves = dag.wave_count();
    prepared.receipt.encounters = dag.encounter_count();
    prepared.receipt.expected_green_spectrum_values =
        expected_spectrum_values;
    prepared.receipt.uniform_macro_weight1 = weight1;
    prepared.receipt.uniform_macro_weight2 = weight2;
    prepared.receipt.green_spectrum_fingerprint = spectrum_fingerprint;
    prepared.receipt.schedule_fingerprint = schedule_hash;
    prepared.receipt.invariant_layout_fingerprint =
        invariant_layout_fingerprint(beam1, beam2);
  } catch (const std::exception& exception) {
    error = std::string("gpu_mode=3 structural preflight allocation failed: ") +
            exception.what();
    return false;
  }
  return true;
}

void initialize_report(const Mode3PrimaryCrossingConfig& config,
                       Mode3PrimaryCrossingReport& report) {
  report = Mode3PrimaryCrossingReport();
  report.implementation_id = kImplementationId;
  report.input_card_label = config.input_card_label;
  report.precision = config.precision;
  report.modern_reduction = config.modern_reduction;
  report.crossing_index = config.crossing_index;
  report.requested_gpu_mode = config.requested_gpu_mode;
  report.selected_gpu_mode = config.requested_gpu_mode == 3 ? 3 : 0;
  report.fallback_permitted =
      config.failure_policy == Mode3FailurePolicy::AllowFallback;
  report.fallback_used = false;
  report.mapped_mutable_components = 6;
  report.host_invariant_components = 10;
}

void pack_beam(const modern::BeamSoA& beam,
               std::vector<float>& xy,
               std::vector<float>& velocity,
               std::vector<float>& energy,
               std::vector<float>& upsilon) {
  const std::size_t count = static_cast<std::size_t>(beam.size());
  xy.resize(2 * count);
  velocity.resize(2 * count);
  energy.resize(count);
  upsilon.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    xy[2 * index] = static_cast<float>(beam.x()[index]);
    xy[2 * index + 1] = static_cast<float>(beam.y()[index]);
    velocity[2 * index] = static_cast<float>(beam.vx()[index]);
    velocity[2 * index + 1] = static_cast<float>(beam.vy()[index]);
    energy[index] = static_cast<float>(beam.energy()[index]);
    upsilon[index] = static_cast<float>(beam.upsilon()[index]);
  }
}

bool validate_download(const std::vector<float>& xy,
                       const std::vector<float>& velocity,
                       const std::vector<float>& energy,
                       const std::vector<float>& upsilon,
                       const modern::BeamSoA& original,
                       unsigned int beam_label,
                       std::string& error) {
  const std::size_t count = static_cast<std::size_t>(original.size());
  if (xy.size() != 2 * count || velocity.size() != 2 * count ||
      energy.size() != count || upsilon.size() != count) {
    error = "gpu_mode=3 downloaded state shape changed unexpectedly";
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(xy[2 * index]) ||
        !std::isfinite(xy[2 * index + 1]) ||
        !std::isfinite(velocity[2 * index]) ||
        !std::isfinite(velocity[2 * index + 1]) ||
        !std::isfinite(energy[index]) || energy[index] <= 0.0f ||
        !std::isfinite(upsilon[index]) || upsilon[index] < 0.0f) {
      std::ostringstream message;
      message << "gpu_mode=3 CUDA output is non-finite or non-positive at beam "
              << beam_label << " particle " << index;
      error = message.str();
      return false;
    }
    if (static_cast<double>(energy[index]) != original.energy()[index]) {
      std::ostringstream message;
      message << "gpu_mode=3 no-radiation contract changed energy at beam "
              << beam_label << " particle " << index;
      error = message.str();
      return false;
    }
  }
  return true;
}

bool validate_residency_receipt(const ResidentPrimaryStats& stats,
                                const Mode3PrimaryPreflight& preflight,
                                std::string& error) {
  std::size_t particle_count = 0;
  std::size_t state_values = 0;
  std::size_t state_bytes_size = 0;
  std::uint64_t state_bytes = 0;
  if (!checked_size_sum(preflight.particle_count1,
                        preflight.particle_count2, particle_count) ||
      !checked_size_product(particle_count, static_cast<std::size_t>(6),
                            state_values) ||
      !checked_size_product(state_values, sizeof(float), state_bytes_size) ||
      !size_to_u64(state_bytes_size, state_bytes)) {
    error = "gpu_mode=3 residency receipt byte accounting overflows";
    return false;
  }
  if (stats.upload_calls != 1 || stats.batch_calls != 1 ||
      stats.download_calls != 1 || stats.state_epoch != 2 ||
      stats.synchronizations != 3 ||
      stats.waves != preflight.waves ||
      stats.encounters != preflight.encounters ||
      stats.intermediate_particle_h2d_bytes != 0 ||
      stats.intermediate_particle_d2h_bytes != 0 ||
      stats.host_field_materializations != 0 ||
      stats.nonfinite_state_events != 0 ||
      stats.h2d_bytes != state_bytes || stats.d2h_bytes < state_bytes ||
      !stats.state_resident || stats.context_poisoned) {
    std::ostringstream message;
    message << "gpu_mode=3 residency receipt invariant failed"
            << " (upload=" << stats.upload_calls
            << ", batch=" << stats.batch_calls
            << ", download=" << stats.download_calls
            << ", sync=" << stats.synchronizations
            << ", waves=" << stats.waves
            << ", encounters=" << stats.encounters
            << ", intermediate_h2d="
            << stats.intermediate_particle_h2d_bytes
            << ", intermediate_d2h="
            << stats.intermediate_particle_d2h_bytes
            << ", host_fields=" << stats.host_field_materializations << ')';
    error = message.str();
    return false;
  }
  return true;
}

void commit_beam(modern::BeamSoA& beam,
                 const std::vector<float>& xy,
                 const std::vector<float>& velocity,
                 const std::vector<float>& energy,
                 const std::vector<float>& upsilon) {
  const std::size_t count = static_cast<std::size_t>(beam.size());
  for (std::size_t index = 0; index < count; ++index) {
    beam.x_data()[index] = static_cast<double>(xy[2 * index]);
    beam.y_data()[index] = static_cast<double>(xy[2 * index + 1]);
    beam.vx_data()[index] = static_cast<double>(velocity[2 * index]);
    beam.vy_data()[index] = static_cast<double>(velocity[2 * index + 1]);
    beam.energy_data()[index] = static_cast<double>(energy[index]);
    beam.upsilon_data()[index] = static_cast<double>(upsilon[index]);
  }
}

std::string no_fallback_error(const char* stage, const std::string& cause) {
  std::ostringstream message;
  message << "gpu_mode=3 " << stage << " failed: " << cause
          << "; crossing outputs were not committed and fallback is forbidden";
  return message.str();
}

}  // namespace

const char* mode3_primary_implementation_id() noexcept {
  return kImplementationId;
}

Mode3PrimaryCrossingAdapter::Mode3PrimaryCrossingAdapter()
    : pipeline_(), execution_attempted_(false) {}

Mode3PrimaryCrossingAdapter::~Mode3PrimaryCrossingAdapter() = default;

bool Mode3PrimaryCrossingAdapter::preflight(
    const modern::BeamSoA& beam1,
    const modern::BeamSoA& beam2,
    const modern::EncounterDAG& encounters,
    const Mode3PrimaryCrossingConfig& config,
    Mode3PrimaryPreflight& result,
    std::string& error) const {
  result = Mode3PrimaryPreflight();
  PreparedCrossing prepared;
  if (!prepare_crossing(beam1, beam2, encounters, config, prepared, error))
    return false;
  result = prepared.receipt;
  return true;
}

bool Mode3PrimaryCrossingAdapter::execute(
    modern::BeamSoA& beam1,
    modern::BeamSoA& beam2,
    const modern::EncounterDAG& encounters,
    const Mode3PrimaryCrossingConfig& config,
    Mode3PrimaryCrossingReport& report,
    std::string& error) {
  initialize_report(config, report);
  error.clear();
  if (execution_attempted_) {
    error = "gpu_mode=3 adapter is one crossing/one attempt; reuse and fallback are forbidden";
    return false;
  }
  execution_attempted_ = true;

  PreparedCrossing prepared;
  if (!prepare_crossing(beam1, beam2, encounters, config, prepared, error)) {
    error = no_fallback_error("structural preflight", error);
    return false;
  }
  report.preflight = prepared.receipt;
  report.structural_preflight_passed = true;
  report.green_spectrum_fingerprint =
      prepared.receipt.green_spectrum_fingerprint;
  report.schedule_fingerprint = prepared.receipt.schedule_fingerprint;
  report.invariant_layout_fingerprint_before =
      prepared.receipt.invariant_layout_fingerprint;

  std::vector<float> xy1;
  std::vector<float> velocity1;
  std::vector<float> energy1;
  std::vector<float> upsilon1;
  std::vector<float> xy2;
  std::vector<float> velocity2;
  std::vector<float> energy2;
  std::vector<float> upsilon2;
  try {
    pack_beam(beam1, xy1, velocity1, energy1, upsilon1);
    pack_beam(beam2, xy2, velocity2, energy2, upsilon2);
  } catch (const std::exception& exception) {
    error = no_fallback_error(
        "whole-beam staging",
        std::string("host allocation failed: ") + exception.what());
    return false;
  }

  std::string pipeline_error;
  if (!pipeline_.initialize(
          config.resident_options, config.nx, config.ny,
          config.green_spectrum_interleaved,
          config.green_spectrum_interleaved_values,
          config.spectrum_grid, config.spectrum_charge_sign,
          prepared.receipt.particle_count1,
          prepared.receipt.particle_count2, pipeline_error)) {
    report.resident = pipeline_.stats();
    error = no_fallback_error("CUDA device/memory preflight", pipeline_error);
    return false;
  }
  report.device_preflight_passed = true;
  report.device_description = pipeline_.device_description();
  report.runtime_description = pipeline_.runtime_description();

  ResidentPrimaryStats stats;
  if (!pipeline_.upload(
          xy1.data(), velocity1.data(), energy1.data(), upsilon1.data(),
          prepared.receipt.particle_count1,
          xy2.data(), velocity2.data(), energy2.data(), upsilon2.data(),
          prepared.receipt.particle_count2, stats, pipeline_error)) {
    report.resident = stats;
    error = no_fallback_error("whole-beam upload", pipeline_error);
    return false;
  }
  report.state_uploaded_once = stats.upload_calls == 1;

  if (!pipeline_.run(config.grid, prepared.schedule.data(),
                     prepared.schedule.size(), stats, pipeline_error)) {
    report.resident = stats;
    error = no_fallback_error("all-wave resident batch", pipeline_error);
    return false;
  }
  report.all_waves_submitted_once = stats.batch_calls == 1;

  if (!pipeline_.download(
          xy1.data(), velocity1.data(), energy1.data(), upsilon1.data(),
          prepared.receipt.particle_count1,
          xy2.data(), velocity2.data(), energy2.data(), upsilon2.data(),
          prepared.receipt.particle_count2, stats, pipeline_error)) {
    report.resident = stats;
    error = no_fallback_error("final whole-beam snapshot", pipeline_error);
    return false;
  }
  report.state_downloaded_once = stats.download_calls == 1;
  report.resident = stats;

  std::string receipt_error;
  if (!validate_download(xy1, velocity1, energy1, upsilon1, beam1, 1u,
                         receipt_error) ||
      !validate_download(xy2, velocity2, energy2, upsilon2, beam2, 2u,
                         receipt_error) ||
      !validate_residency_receipt(stats, prepared.receipt, receipt_error)) {
    error = no_fallback_error("post-CUDA receipt validation", receipt_error);
    return false;
  }
  const std::uint64_t invariant_before_commit =
      invariant_layout_fingerprint(beam1, beam2);
  if (invariant_before_commit !=
      prepared.receipt.invariant_layout_fingerprint) {
    error = no_fallback_error(
        "transaction boundary",
        "BeamSoA identity/layout/invariant components changed while CUDA was running");
    return false;
  }

  // Non-allocating, non-throwing component writes after every validation gate.
  commit_beam(beam1, xy1, velocity1, energy1, upsilon1);
  commit_beam(beam2, xy2, velocity2, energy2, upsilon2);
  report.outputs_committed = true;

  const modern::ValidationResult beam1_after = beam1.validate();
  const modern::ValidationResult beam2_after = beam2.validate();
  report.invariant_layout_fingerprint_after =
      invariant_layout_fingerprint(beam1, beam2);
  report.slice_offsets_and_order_preserved =
      report.invariant_layout_fingerprint_after ==
      prepared.receipt.invariant_layout_fingerprint;
  report.host_invariants_preserved =
      report.slice_offsets_and_order_preserved;
  if (!beam1_after || !beam2_after || !report.host_invariants_preserved) {
    // This branch can only diagnose an internal programming error: all
    // committed components were already finite and BeamSoA shape is immutable.
    // Do not suggest a fallback because CUDA already mutated the crossing.
    error = "gpu_mode=3 internal post-commit BeamSoA invariant failure; fallback is forbidden";
    return false;
  }
  return true;
}

}  // namespace gpu
}  // namespace feebl
