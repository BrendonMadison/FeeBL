#include "cellCSRCPP.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

namespace feebl {
namespace modern {

UniformGrid2D::UniformGrid2D(Scalar xmin, Scalar xmax, Scalar ymin,
                             Scalar ymax, CellKey nx, CellKey ny)
    : xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      nx_(nx),
      ny_(ny),
      active_nx_(nx),
      active_ny_(ny),
      cell_count_(0),
      mapping_mode_(GridMappingMode::CompactHalfOpen) {
  if (!std::isfinite(xmin_) || !std::isfinite(xmax_) ||
      !std::isfinite(ymin_) || !std::isfinite(ymax_) ||
      !(xmin_ < xmax_) || !(ymin_ < ymax_))
    throw std::invalid_argument("UniformGrid2D requires finite increasing bounds");
  if (nx_ == 0 || ny_ == 0)
    throw std::invalid_argument("UniformGrid2D dimensions must be non-zero");
  if (!checked_multiply(nx_, ny_, cell_count_))
    throw std::overflow_error("UniformGrid2D cell-count multiplication overflow");
  if (cell_count_ == kOutsideCell)
    throw std::overflow_error("UniformGrid2D cell key collides with outside sentinel");
}

UniformGrid2D::UniformGrid2D(GridMappingMode mode, Scalar cut_x, Scalar cut_y,
                             CellKey storage_nx, CellKey storage_ny)
    : xmin_(-cut_x),
      xmax_(cut_x),
      ymin_(-cut_y),
      ymax_(cut_y),
      nx_(storage_nx),
      ny_(storage_ny),
      active_nx_(storage_nx > 0 ? storage_nx - 1 : 0),
      active_ny_(storage_ny > 0 ? storage_ny - 1 : 0),
      cell_count_(0),
      mapping_mode_(mode) {
  if (mode != GridMappingMode::LegacyNGP &&
      mode != GridMappingMode::LegacyCIC)
    throw std::invalid_argument("legacy grid factory received non-legacy mode");
  if (!std::isfinite(cut_x) || !std::isfinite(cut_y) ||
      !(cut_x > 0.0) || !(cut_y > 0.0))
    throw std::invalid_argument("legacy grid cuts must be finite and positive");
  if (nx_ < 2 || ny_ < 2 || nx_ > static_cast<CellKey>(INT_MAX) ||
      ny_ > static_cast<CellKey>(INT_MAX))
    throw std::invalid_argument(
        "legacy grid storage dimensions must be in [2, INT_MAX]");
  if (!checked_multiply(nx_, ny_, cell_count_))
    throw std::overflow_error("legacy grid storage-count multiplication overflow");
  if (cell_count_ == kOutsideCell)
    throw std::overflow_error("legacy grid cell key collides with outside sentinel");

  // Match GENERAL_GRID::init_grid_phys line-for-line in float storage.  The
  // explicit casts document the legacy narrowing points.
  const float cut_x_f = static_cast<float>(cut_x);
  const float cut_y_f = static_cast<float>(cut_y);
  const float nx_f = static_cast<float>(nx_);
  const float ny_f = static_cast<float>(ny_);
  const float delta_x = static_cast<float>(2.0 * cut_x_f / nx_f);
  const float delta_y = static_cast<float>(2.0 * cut_y_f / ny_f);
  legacy_inverse_dx_ = static_cast<float>(1.0 / delta_x);
  legacy_inverse_dy_ = static_cast<float>(1.0 / delta_y);
  legacy_offset_x_ = static_cast<float>(0.5 * nx_f);
  legacy_offset_y_ = static_cast<float>(0.5 * ny_f);
  if (!std::isfinite(legacy_inverse_dx_) ||
      !std::isfinite(legacy_inverse_dy_) ||
      !std::isfinite(legacy_offset_x_) ||
      !std::isfinite(legacy_offset_y_))
    throw std::overflow_error("legacy grid float geometry is not representable");
}

UniformGrid2D UniformGrid2D::legacy_ngp(Scalar cut_x, Scalar cut_y,
                                        CellKey storage_nx,
                                        CellKey storage_ny) {
  return UniformGrid2D(GridMappingMode::LegacyNGP, cut_x, cut_y,
                       storage_nx, storage_ny);
}

UniformGrid2D UniformGrid2D::legacy_cic(Scalar cut_x, Scalar cut_y,
                                        CellKey storage_nx,
                                        CellKey storage_ny) {
  return UniformGrid2D(GridMappingMode::LegacyCIC, cut_x, cut_y,
                       storage_nx, storage_ny);
}

CellMapping UniformGrid2D::map(Scalar x, Scalar y) const noexcept {
  CellMapping result;
  if (mapping_mode_ == GridMappingMode::LegacyNGP ||
      mapping_mode_ == GridMappingMode::LegacyCIC) {
    const float xpart = static_cast<float>(x);
    const float ypart = static_cast<float>(y);
    if (!std::isfinite(xpart) || !std::isfinite(ypart)) return result;

    float hx = static_cast<float>(xpart * legacy_inverse_dx_ +
                                  legacy_offset_x_);
    float hy = static_cast<float>(ypart * legacy_inverse_dy_ +
                                  legacy_offset_y_);
    if (mapping_mode_ == GridMappingMode::LegacyCIC) {
      // The old expression subtracts a double literal and stores to float.
      hx = static_cast<float>(static_cast<double>(hx) - 0.5);
      hy = static_cast<float>(static_cast<double>(hy) - 0.5);
    }
    if (!std::isfinite(hx) || !std::isfinite(hy)) return result;
    const double floor_x = std::floor(static_cast<double>(hx));
    const double floor_y = std::floor(static_cast<double>(hy));
    if (floor_x < static_cast<double>(INT_MIN) ||
        floor_x > static_cast<double>(INT_MAX) ||
        floor_y < static_cast<double>(INT_MIN) ||
        floor_y > static_cast<double>(INT_MAX))
      return result;
    const int i = static_cast<int>(floor_x);
    const int j = static_cast<int>(floor_y);
    if (i < 0 || j < 0 || i >= static_cast<int>(active_nx_) ||
        j >= static_cast<int>(active_ny_))
      return result;

    result.inside = true;
    result.ix = static_cast<CellKey>(i);
    result.iy = static_cast<CellKey>(j);
    result.key = result.ix * ny_ + result.iy;
    if (mapping_mode_ == GridMappingMode::LegacyCIC) {
      result.weight_x = static_cast<Scalar>(
          static_cast<float>(hx - static_cast<float>(i)));
      result.weight_y = static_cast<Scalar>(
          static_cast<float>(hy - static_cast<float>(j)));
    }
    return result;
  }

  if (!std::isfinite(x) || !std::isfinite(y) || x < xmin_ || !(x < xmax_) ||
      y < ymin_ || !(y < ymax_))
    return result;

  const long double fx =
      (static_cast<long double>(x) - static_cast<long double>(xmin_)) /
      (static_cast<long double>(xmax_) - static_cast<long double>(xmin_));
  const long double fy =
      (static_cast<long double>(y) - static_cast<long double>(ymin_)) /
      (static_cast<long double>(ymax_) - static_cast<long double>(ymin_));
  CellKey ix = static_cast<CellKey>(fx * static_cast<long double>(nx_));
  CellKey iy = static_cast<CellKey>(fy * static_cast<long double>(ny_));

  // x/y passed the half-open range check.  Clamp only a possible final-bit
  // rounding of the scaled coordinate, never an actually out-of-grid point.
  if (ix >= nx_) ix = nx_ - 1;
  if (iy >= ny_) iy = ny_ - 1;
  result.inside = true;
  result.ix = ix;
  result.iy = iy;
  result.key = iy * nx_ + ix;  // constructor proved this cannot overflow
  return result;
}

CellCSR::CellCSR(const BeamSoA& beam, const UniformGrid2D& grid,
                 CellOrder order)
    : grid_(grid), order_(order) {
  const ValidationResult beam_validation = beam.validate();
  if (!beam_validation)
    throw std::invalid_argument("CellCSR received invalid BeamSoA: " +
                                beam_validation.message);
  if (order_ != CellOrder::DenseInput &&
      order_ != CellOrder::ByStableParticleId)
    throw std::invalid_argument("CellCSR received unknown ordering mode");

  std::uint64_t offset_count = 0;
  if (!checked_add(grid_.cell_count(), 1, offset_count) ||
      !fits_size_t(offset_count) ||
      offset_count > cell_offsets_.max_size())
    throw std::length_error("CellCSR offset array is not representable");
  if (!fits_size_t(beam.size()) ||
      beam.size() > particle_indices_.max_size() ||
      beam.size() > particle_cell_keys_.max_size() ||
      beam.size() > outside_particles_.max_size())
    throw std::length_error("CellCSR particle arrays are not representable");

  const std::size_t n_cells = static_cast<std::size_t>(grid_.cell_count());
  const std::size_t n_particles = static_cast<std::size_t>(beam.size());
  std::vector<ArrayOffset> counts(n_cells, 0);
  particle_cell_keys_.resize(n_particles, kOutsideCell);

  // Pass 1: deterministic mapping and exact occupancy counts.
  ArrayOffset outside_count = 0;
  for (std::size_t p = 0; p < n_particles; ++p) {
    const CellMapping mapping = grid_.map(beam.x()[p], beam.y()[p]);
    particle_cell_keys_[p] = mapping.key;
    if (!mapping.inside) {
      if (outside_count == std::numeric_limits<ArrayOffset>::max())
        throw std::overflow_error("CellCSR outside count overflow");
      ++outside_count;
      continue;
    }
    ArrayOffset& count = counts[static_cast<std::size_t>(mapping.key)];
    if (count == std::numeric_limits<ArrayOffset>::max())
      throw std::overflow_error("CellCSR occupancy count overflow");
    ++count;
  }

  // Prefix scan with checked addition.
  cell_offsets_.resize(static_cast<std::size_t>(offset_count), 0);
  ArrayOffset running = 0;
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    ArrayOffset next = 0;
    if (!checked_add(running, counts[cell], next))
      throw std::overflow_error("CellCSR prefix-sum overflow");
    cell_offsets_[cell + 1] = next;
    running = next;
  }
  if (!fits_size_t(running))
    throw std::length_error("CellCSR in-grid population is not representable");

  particle_indices_.resize(static_cast<std::size_t>(running));
  outside_particles_.reserve(static_cast<std::size_t>(outside_count));
  std::vector<ArrayOffset> cursor(cell_offsets_.begin(),
                                  cell_offsets_.end() - 1);

  // Pass 2: scanning p in dense order gives stable within-cell insertion.
  for (std::size_t p = 0; p < n_particles; ++p) {
    const CellKey key = particle_cell_keys_[p];
    if (key == kOutsideCell) {
      outside_particles_.push_back(static_cast<ParticleIndex>(p));
      continue;
    }
    ArrayOffset& position = cursor[static_cast<std::size_t>(key)];
    particle_indices_[static_cast<std::size_t>(position++)] =
        static_cast<ParticleIndex>(p);
  }

  if (order_ == CellOrder::ByStableParticleId) {
    const auto by_stable_id = [&beam](ParticleIndex a, ParticleIndex b) {
      const StableParticleId ida =
          beam.stable_ids()[static_cast<std::size_t>(a)];
      const StableParticleId idb =
          beam.stable_ids()[static_cast<std::size_t>(b)];
      return ida < idb || (ida == idb && a < b);
    };
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      const std::size_t begin =
          static_cast<std::size_t>(cell_offsets_[cell]);
      const std::size_t end =
          static_cast<std::size_t>(cell_offsets_[cell + 1]);
      std::sort(particle_indices_.begin() + begin,
                particle_indices_.begin() + end, by_stable_id);
    }
    std::sort(outside_particles_.begin(), outside_particles_.end(),
              by_stable_id);
  }
}

CellRange CellCSR::cell_range(CellKey cell) const {
  if (cell >= grid_.cell_count())
    throw std::out_of_range("CellCSR cell key out of range");
  const std::size_t index = static_cast<std::size_t>(cell);
  return CellRange{cell_offsets_[index], cell_offsets_[index + 1]};
}

ValidationResult CellCSR::validate(const BeamSoA& beam) const {
  const ValidationResult beam_validation = beam.validate();
  if (!beam_validation)
    return ValidationResult::failure("underlying BeamSoA is invalid: " +
                                     beam_validation.message,
                                     beam_validation.index);
  if (order_ != CellOrder::DenseInput &&
      order_ != CellOrder::ByStableParticleId)
    return ValidationResult::failure("unknown CellCSR ordering mode");

  std::uint64_t expected_offsets = 0;
  if (!checked_add(grid_.cell_count(), 1, expected_offsets) ||
      !fits_size_t(expected_offsets) ||
      cell_offsets_.size() != static_cast<std::size_t>(expected_offsets))
    return ValidationResult::failure("CellCSR offset-array shape is invalid");
  if (particle_cell_keys_.size() != static_cast<std::size_t>(beam.size()))
    return ValidationResult::failure("CellCSR particle-to-cell map has wrong length");
  if (cell_offsets_.empty() || cell_offsets_.front() != 0)
    return ValidationResult::failure("CellCSR offsets must begin at zero");

  ArrayOffset previous = 0;
  for (std::size_t cell = 0; cell < cell_offsets_.size(); ++cell) {
    if (cell_offsets_[cell] < previous ||
        cell_offsets_[cell] > particle_indices_.size())
      return ValidationResult::failure("CellCSR offsets are not monotonic and bounded",
                                       static_cast<std::uint64_t>(cell));
    previous = cell_offsets_[cell];
  }
  if (cell_offsets_.back() != particle_indices_.size())
    return ValidationResult::failure("CellCSR final offset disagrees with membership length");

  const std::size_t n = static_cast<std::size_t>(beam.size());
  std::vector<std::uint8_t> seen(n, 0);
  for (CellKey cell = 0; cell < grid_.cell_count(); ++cell) {
    const CellRange range = cell_range(cell);
    ParticleIndex preceding = 0;
    StableParticleId preceding_id = 0;
    bool have_preceding = false;
    for (ArrayOffset pos = range.begin; pos < range.end; ++pos) {
      const ParticleIndex p =
          particle_indices_[static_cast<std::size_t>(pos)];
      if (p >= beam.size())
        return ValidationResult::failure("CellCSR particle index is out of range", pos);
      if (particle_cell_keys_[static_cast<std::size_t>(p)] != cell)
        return ValidationResult::failure("CellCSR membership disagrees with cached cell key", p);
      const CellMapping current =
          grid_.map(beam.x()[static_cast<std::size_t>(p)],
                    beam.y()[static_cast<std::size_t>(p)]);
      if (!current.inside || current.key != cell)
        return ValidationResult::failure("CellCSR membership is stale relative to coordinates", p);
      if (seen[static_cast<std::size_t>(p)]++)
        return ValidationResult::failure("particle occurs more than once in CellCSR", p);

      if (have_preceding) {
        if (order_ == CellOrder::DenseInput && !(preceding < p))
          return ValidationResult::failure("dense-order CellCSR segment is unstable", cell);
        const StableParticleId id =
            beam.stable_ids()[static_cast<std::size_t>(p)];
        if (order_ == CellOrder::ByStableParticleId && !(preceding_id < id))
          return ValidationResult::failure("stable-ID CellCSR segment is unordered", cell);
      }
      preceding = p;
      preceding_id = beam.stable_ids()[static_cast<std::size_t>(p)];
      have_preceding = true;
    }
  }

  ParticleIndex preceding_outside = 0;
  StableParticleId preceding_outside_id = 0;
  bool have_outside = false;
  for (ParticleIndex p : outside_particles_) {
    if (p >= beam.size())
      return ValidationResult::failure("outside particle index is out of range", p);
    if (particle_cell_keys_[static_cast<std::size_t>(p)] != kOutsideCell ||
        grid_.map(beam.x()[static_cast<std::size_t>(p)],
                  beam.y()[static_cast<std::size_t>(p)]).inside)
      return ValidationResult::failure("outside membership disagrees with mapping", p);
    if (seen[static_cast<std::size_t>(p)]++)
      return ValidationResult::failure("particle occurs in multiple CSR populations", p);
    if (have_outside) {
      if (order_ == CellOrder::DenseInput && !(preceding_outside < p))
        return ValidationResult::failure("outside dense order is unstable", p);
      const StableParticleId id =
          beam.stable_ids()[static_cast<std::size_t>(p)];
      if (order_ == CellOrder::ByStableParticleId &&
          !(preceding_outside_id < id))
        return ValidationResult::failure("outside stable-ID order is unordered", p);
    }
    preceding_outside = p;
    preceding_outside_id = beam.stable_ids()[static_cast<std::size_t>(p)];
    have_outside = true;
  }

  for (std::size_t p = 0; p < n; ++p) {
    const CellMapping current = grid_.map(beam.x()[p], beam.y()[p]);
    if (particle_cell_keys_[p] != current.key)
      return ValidationResult::failure("cached particle cell key is stale",
                                       static_cast<std::uint64_t>(p));
    if (seen[p] != 1)
      return ValidationResult::failure("particle is absent from CSR membership",
                                       static_cast<std::uint64_t>(p));
  }
  return ValidationResult::success();
}

}  // namespace modern
}  // namespace feebl
