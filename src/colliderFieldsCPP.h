#ifndef FEEBL_COLLIDER_FIELDS_CPP_H
#define FEEBL_COLLIDER_FIELDS_CPP_H

#include "gridCPP.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace feebl { namespace collider_fields {
// A scalar view of actual solved potentials gathered with the native CIC
// stencil. Coordinates are nm; values are native electric-equivalent GV/nm.
// This is not a charge reconstruction, field solve, or sum of source norms.
struct PlaneSample {
  std::size_t nx=0,ny=0;
  double x_min_nm=0,x_max_nm=0,y_min_nm=0,y_max_nm=0;
  std::vector<double> beam1,beam2; // y-major rows, x changes fastest
};
inline bool finite_number(double value) {
  // Preserve validation under legacy -ffast-math compilation.
  static_assert(sizeof(double)==sizeof(std::uint64_t),"requires binary64");
  std::uint64_t bits=0; std::memcpy(&bits,&value,sizeof(bits));
  return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
inline PlaneSample main_grid_geometry(const GENERAL_GRID& grid,
                                     std::size_t maximum_axis=64) {
  if (maximum_axis<2 || maximum_axis>64 ||
      grid.get_n_cell_x()<4 || grid.get_n_cell_y()<4)
    throw std::invalid_argument("native field map needs grid axes >=4 and cap 2..64");
  PlaneSample result;
  result.nx=std::min(maximum_axis,static_cast<std::size_t>(grid.get_n_cell_x()-2));
  result.ny=std::min(maximum_axis,static_cast<std::size_t>(grid.get_n_cell_y()-2));
  result.x_min_nm=double(grid.get_min_x())+0.5*double(grid.get_delta_x());
  result.x_max_nm=double(grid.get_max_x())-0.5*double(grid.get_delta_x());
  result.y_min_nm=double(grid.get_min_y())+0.5*double(grid.get_delta_y());
  result.y_max_nm=double(grid.get_max_y())-0.5*double(grid.get_delta_y());
  if (!finite_number(result.x_min_nm) || !finite_number(result.x_max_nm) ||
      !finite_number(result.y_min_nm) || !finite_number(result.y_max_nm) ||
      !(result.x_max_nm>result.x_min_nm) || !(result.y_max_nm>result.y_min_nm))
    throw std::invalid_argument("invalid native field sampling geometry");
  return result;
}
inline PlaneSample sample_main_grid(const GENERAL_GRID& grid,
    const PHI_FLOAT* phi1,const PHI_FLOAT* phi2,std::size_t maximum_axis=64) {
  if (!phi1 || !phi2) throw std::invalid_argument("missing native field potentials");
  PlaneSample result=main_grid_geometry(grid,maximum_axis);
  // Reject invalid solver arrays before interpolation. This reads the existing
  // host solution; it neither repairs nor replaces a solver value.
  const std::size_t native_size=static_cast<std::size_t>(grid.get_n_cell_x())*
      static_cast<std::size_t>(grid.get_n_cell_y());
  for (std::size_t i=0;i<native_size;++i)
    if (!finite_number(phi1[i]) || !finite_number(phi2[i]))
      throw std::domain_error("non-finite native field potential");
  result.beam1.reserve(result.nx*result.ny);
  result.beam2.reserve(result.nx*result.ny);
  for (std::size_t iy=0;iy<result.ny;++iy) {
    const float y=static_cast<float>(result.y_min_nm+
        (result.y_max_nm-result.y_min_nm)*double(iy)/double(result.ny-1));
    for (std::size_t ix=0;ix<result.nx;++ix) {
      const float x=static_cast<float>(result.x_min_nm+
          (result.x_max_nm-result.x_min_nm)*double(ix)/double(result.nx-1));
      if (!grid.coordinatesInGridRange(x,y))
        throw std::domain_error("native field sample outside safe CIC stencil");
      const TRIVECTOR e1=grid.ElectricFieldCIC(x,y,phi1);
      const TRIVECTOR e2=grid.ElectricFieldCIC(x,y,phi2);
      const double a=std::hypot(double(e1(0)),double(e1(1)));
      const double b=std::hypot(double(e2(0)),double(e2(1)));
      if (!finite_number(a) || !finite_number(b))
        throw std::domain_error("non-finite native field magnitude");
      result.beam1.push_back(a); result.beam2.push_back(b);
    }
  }
  return result;
}
inline PlaneSample sample_main_grid(const GENERAL_GRID& grid,
                                   std::size_t maximum_axis=64) {
  return sample_main_grid(grid,grid.get_phi1(),grid.get_phi2(),maximum_axis);
}

// Source-specific longitudinal display bands, retaining the solver's raw
// encounter cell. The legacy dump changes the sign and shifts each beam.
struct EncounterZ {
  double nominal_nm, nominal_low_nm, nominal_high_nm;
  double beam1_nm, beam2_nm, half_width_nm;
};
inline EncounterZ encounter_z(int i1,int i2,double dz) {
  if(i1<0 || i2<0 || !finite_number(dz) || !(dz>0.0))
    throw std::invalid_argument("invalid longitudinal encounter geometry");
  EncounterZ z;
  z.nominal_nm=0.5*(double(i2)-double(i1))*dz;
  z.half_width_nm=0.5*dz;
  z.nominal_low_nm=z.nominal_nm-z.half_width_nm;
  z.nominal_high_nm=z.nominal_nm+z.half_width_nm;
  z.beam1_nm=-z.nominal_nm+dz;
  z.beam2_nm=-z.nominal_nm-dz;
  return z;
}
inline std::size_t longitudinal_axis_cap(int n_z) {
  if(n_z<1 || n_z>128) return 0; // explicit unavailable extension, not thinning
  const int last=2*n_z-2, stride=std::max(1,(last+62)/63);
  std::size_t encounters=0;
  for(int d=0;d<=last;++d) {
    if(d!=last && d%stride!=0) continue;
    encounters+=std::min(d,n_z-1)-std::max(0,d-n_z+1)+1;
  }
  return std::min(std::size_t(64),std::min(std::size_t(4096/n_z),
      std::size_t(524288)/(4*encounters)));
}
struct CrossSections {
  PlaneSample geometry;
  std::vector<double> beam1_zx,beam1_zy,beam2_zx,beam2_zy;
};
inline CrossSections sample_cross_sections(const GENERAL_GRID& grid,
    const PHI_FLOAT* phi1,const PHI_FLOAT* phi2,std::size_t maximum_axis=64) {
  if(!phi1 || !phi2) throw std::invalid_argument("missing native field potentials");
  CrossSections result;
  result.geometry=main_grid_geometry(grid,maximum_axis);
  const std::size_t native_size=static_cast<std::size_t>(grid.get_n_cell_x())*
      static_cast<std::size_t>(grid.get_n_cell_y());
  for(std::size_t i=0;i<native_size;++i)
    if(!finite_number(phi1[i]) || !finite_number(phi2[i]))
      throw std::domain_error("non-finite native field potential");
  // These are independent gathers at exactly y=0 or x=0, not extracted rows
  // from an even-sized XY raster (whose centres need not include the axis).
  for(int plane=0;plane<2;++plane) {
    const std::size_t n=plane==0?result.geometry.nx:result.geometry.ny;
    const double lo=plane==0?result.geometry.x_min_nm:result.geometry.y_min_nm;
    const double hi=plane==0?result.geometry.x_max_nm:result.geometry.y_max_nm;
    std::vector<double>& a=plane==0?result.beam1_zx:result.beam1_zy;
    std::vector<double>& b=plane==0?result.beam2_zx:result.beam2_zy;
    a.reserve(n);b.reserve(n);
    for(std::size_t i=0;i<n;++i) {
      const float transverse=static_cast<float>(lo+(hi-lo)*double(i)/double(n-1));
      const float x=plane==0?transverse:0.0f, y=plane==0?0.0f:transverse;
      if(!grid.coordinatesInGridRange(x,y))
        throw std::domain_error("longitudinal line outside safe CIC stencil");
      const TRIVECTOR e1=grid.ElectricFieldCIC(x,y,phi1);
      const TRIVECTOR e2=grid.ElectricFieldCIC(x,y,phi2);
      const double v1=std::hypot(double(e1(0)),double(e1(1)));
      const double v2=std::hypot(double(e2(0)),double(e2(1)));
      if(!finite_number(v1)||!finite_number(v2))
        throw std::domain_error("non-finite native field cross-section");
      a.push_back(v1);b.push_back(v2);
    }
  }
  return result;
}
inline CrossSections sample_cross_sections(const GENERAL_GRID& grid,
                                           std::size_t maximum_axis=64) {
  return sample_cross_sections(grid,grid.get_phi1(),grid.get_phi2(),maximum_axis);
}
} }
#endif
