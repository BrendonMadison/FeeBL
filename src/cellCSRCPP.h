#ifndef FEEBL_CELL_CSR_CPP_H
#define FEEBL_CELL_CSR_CPP_H

#include "beamSoACPP.h"
#include "modernTypesCPP.h"

#include <utility>
#include <vector>

namespace feebl {
namespace modern {

struct CellMapping {
  bool inside = false;
  CellKey key = kOutsideCell;
  CellKey ix = 0;
  CellKey iy = 0;
  Scalar weight_x = 0.0;
  Scalar weight_y = 0.0;
};

enum class GridMappingMode : std::uint8_t {
  // Compact general-purpose bins, with no guard row/column.
  CompactHalfOpen = 1,
  // Bit-semantic reproduction of GENERAL_GRID::particleInGrid(x,y,i,j).
  LegacyNGP = 2,
  // Bit-semantic reproduction of the -0.5-shifted CIC overload, including
  // interpolation fractions and the terminal guard row/column.
  LegacyCIC = 3
};

// A uniform 2-D mapping specification.  The ordinary constructor creates a
// compact half-open mesh: xmin <= x < xmax, ymin <= y < ymax, key=iy*nx+ix.
// It is intentionally *not* claimed to reproduce GENERAL_GRID.
//
// The legacy factories reproduce FeeBL's float arithmetic and formulas:
//   NGP: floor(float(x)*inv_dx + offset_x)
//   CIC: floor(float(float(x)*inv_dx + offset_x - 0.5))
// with 0 <= i < n_x-1, 0 <= j < n_y-1 and storage key=i*n_y+j.  n_x/n_y are
// full legacy storage strides, so terminal guard cells remain addressable in
// offsets even though no particle is based in them.
class UniformGrid2D {
 public:
  UniformGrid2D(Scalar xmin, Scalar xmax, Scalar ymin, Scalar ymax,
                CellKey nx, CellKey ny);

  static UniformGrid2D legacy_ngp(Scalar cut_x, Scalar cut_y,
                                  CellKey storage_nx, CellKey storage_ny);
  static UniformGrid2D legacy_cic(Scalar cut_x, Scalar cut_y,
                                  CellKey storage_nx, CellKey storage_ny);

  Scalar xmin() const noexcept { return xmin_; }
  Scalar xmax() const noexcept { return xmax_; }
  Scalar ymin() const noexcept { return ymin_; }
  Scalar ymax() const noexcept { return ymax_; }
  CellKey nx() const noexcept { return nx_; }
  CellKey ny() const noexcept { return ny_; }
  CellKey active_nx() const noexcept { return active_nx_; }
  CellKey active_ny() const noexcept { return active_ny_; }
  CellKey cell_count() const noexcept { return cell_count_; }
  GridMappingMode mapping_mode() const noexcept { return mapping_mode_; }

  CellMapping map(Scalar x, Scalar y) const noexcept;

 private:
  UniformGrid2D(GridMappingMode mode, Scalar cut_x, Scalar cut_y,
                CellKey storage_nx, CellKey storage_ny);

  Scalar xmin_;
  Scalar xmax_;
  Scalar ymin_;
  Scalar ymax_;
  CellKey nx_;
  CellKey ny_;
  CellKey active_nx_;
  CellKey active_ny_;
  CellKey cell_count_;
  GridMappingMode mapping_mode_;
  float legacy_inverse_dx_ = 0.0f;
  float legacy_inverse_dy_ = 0.0f;
  float legacy_offset_x_ = 0.0f;
  float legacy_offset_y_ = 0.0f;
};

enum class CellOrder : std::uint8_t {
  // Stable with respect to the dense input order; this is the low-overhead
  // deterministic CPU two-pass path.
  DenseInput = 1,
  // Canonical validation path, independent of dense storage permutation.
  ByStableParticleId = 2
};

struct CellRange {
  ArrayOffset begin = 0;
  ArrayOffset end = 0;

  ArrayOffset size() const noexcept { return end - begin; }
  bool empty() const noexcept { return begin == end; }
};

// Compact segmented cell membership.  Construction performs map/count then
// prefix-scan/fill, with checked arithmetic throughout.  The fill scans dense
// particles in increasing index order, giving a stable within-cell order; the
// optional canonical mode then orders each segment by stable particle ID.
class CellCSR {
 public:
  CellCSR(const BeamSoA& beam, const UniformGrid2D& grid,
          CellOrder order = CellOrder::ByStableParticleId);

  const UniformGrid2D& grid() const noexcept { return grid_; }
  CellOrder order() const noexcept { return order_; }
  const std::vector<ArrayOffset>& cell_offsets() const noexcept {
    return cell_offsets_;
  }
  const std::vector<ParticleIndex>& particle_indices() const noexcept {
    return particle_indices_;
  }
  const std::vector<CellKey>& particle_cell_keys() const noexcept {
    return particle_cell_keys_;
  }
  const std::vector<ParticleIndex>& outside_particles() const noexcept {
    return outside_particles_;
  }

  CellRange cell_range(CellKey cell) const;
  ValidationResult validate(const BeamSoA& beam) const;

 private:
  UniformGrid2D grid_;
  CellOrder order_;
  std::vector<ArrayOffset> cell_offsets_;
  std::vector<ParticleIndex> particle_indices_;
  std::vector<CellKey> particle_cell_keys_;
  std::vector<ParticleIndex> outside_particles_;
};

}  // namespace modern
}  // namespace feebl

#endif  // FEEBL_CELL_CSR_CPP_H
