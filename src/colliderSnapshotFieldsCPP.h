#ifndef FEEBL_COLLIDER_SNAPSHOT_FIELDS_CPP_H
#define FEEBL_COLLIDER_SNAPSHOT_FIELDS_CPP_H

#include "colliderFieldsCPP.h"
#include <memory>

namespace feebl { namespace collider_fields {

// This is the every_particle>1 branch used by the GUI's native particle dump,
// applied to an owning copy of EVERY source particle. It is not an integrator:
// inactive slices are linearly drifted to the displayed diagonal endpoint.
inline void synchronize_display_copy(PARTICLE& copy,int slice,int diagonal,
    int n_z,int n_t,float step,float max_z,int sign) {
  if(n_z<1 || slice<0 || slice>=n_z || diagonal<0 || diagonal>2*n_z-2 ||
      n_t<1 || !finite_number(step) || !(step>0) ||
      !finite_number(max_z) || (sign!=1 && sign!=-1))
    throw std::invalid_argument("invalid snapshot display synchronization");
  const float dz0=n_t*step; // preserve the native dump's float operation order
  if(diagonal<n_z && slice>diagonal)
    copy.set_z_velocity_corrected_for_dump(slice,diagonal,0,dz0,max_z,sign);
  else if(diagonal>=n_z && slice<=diagonal-n_z)
    copy.set_z_velocity_corrected_for_dump(slice,diagonal,n_z,dz0,max_z,sign);
  else copy.set_z_for_dump(diagonal,dz0,max_z,sign);
}

struct SnapshotBand { double centre,low,high; };
inline SnapshotBand snapshot_band(int slice,int diagonal,int n_z,double dz,int sign) {
  if(n_z<1 || slice<0 || slice>=n_z || diagonal<0 || diagonal>2*n_z-2 ||
      !finite_number(dz) || !(dz>0) || (sign!=1 && sign!=-1))
    throw std::invalid_argument("invalid full-slice snapshot band");
  const double centre=sign*(double(slice)+1.0-0.5*diagonal)*dz;
  return {centre,centre-0.5*dz,centre+0.5*dz};
}
inline std::size_t snapshot_axis_cap(int n_z) {
  return n_z>=1 && n_z<=128?std::min(std::size_t(64),std::size_t(4096/n_z)):0;
}
struct SnapshotDeposit {
  std::size_t source_particles=0,deposited_particles=0,outside_particles=0;
  std::size_t gather_inside_particles=0,gather_outside_particles=0;
  bool has_bounds=false;
  double x_min=0,x_max=0,y_min=0,y_max=0;
};

// No beam pointer, RNG, result accumulator, or live phi/rho is attached here.
// This scratch grid owns its FIELD and FFT plans. Only the source geometry and
// already normalized native Green-kernel factor are read from the live grid.
// GENERAL_GRID's separate default FIELD remains unused; constructing our own
// FIELD directly also avoids copying its not-yet-initialized potential arrays.
class SnapshotGrid final : public GENERAL_GRID {
  FFT_SERVER fourier_;
  FIELD field_;
  std::vector<PHI_FLOAT> rho1_,rho2_;
  void distribute_coherent_particles(int,int,int) override {}
  void distribute_particles(int,int,int,int) override {}
  static int checked_nx(const GENERAL_GRID& native) {
    if(!supported_geometry(native))throw std::invalid_argument("unsupported snapshot geometry");
    return native.get_n_cell_x();
  }
 public:
  static bool supported_geometry(const GENERAL_GRID& native) {
    const int nx=native.get_n_cell_x(),ny=native.get_n_cell_y();
    return nx>=4 && ny>=4 && nx<=1048576/ny &&
        snapshot_axis_cap(native.get_n_cell_z())>0;
  }
  explicit SnapshotGrid(const GENERAL_GRID& native)
      :field_(2,checked_nx(native),native.get_n_cell_y(),&fourier_) {
    if(!supported_geometry(native))
      throw std::invalid_argument("snapshot grid exceeds the bounded native geometry");
    n_cell_x_=native.get_n_cell_x();n_cell_y_=native.get_n_cell_y();
    n_cell_z_=native.get_n_cell_z();timestep_=native.get_timestep();
    min_x_=native.get_min_x();max_x_=native.get_max_x();
    min_y_=native.get_min_y();max_y_=native.get_max_y();
    min_z_=native.get_min_z();max_z_=native.get_max_z();step_=native.get_step();
    delta_x_inverse_=native.get_delta_x_inverse();
    delta_y_inverse_=native.get_delta_y_inverse();
    mesh_=MESH(native.get_delta_x(),native.get_delta_y(),native.get_delta_z(),
        0.5f*n_cell_x_,0.5f*n_cell_y_);
    field_kernel_factor_=native.get_field_kernel_factor();
    if(!finite_number(field_kernel_factor_))
      throw std::invalid_argument("snapshot native field kernel is not finite");
    field_.dist_init(field_kernel_factor_,native.get_delta_x(),
        native.get_delta_y(),&fourier_);
    const std::size_t cells=std::size_t(n_cell_x_)*n_cell_y_;
    rho1_.resize(cells);rho2_.resize(cells);
  }
  SnapshotGrid(const SnapshotGrid&)=delete;
  SnapshotGrid& operator=(const SnapshotGrid&)=delete;

  SnapshotDeposit deposit(int beam,const std::vector<PARTICLE*>& particles,
      PHI_FLOAT macro_charge,int slice,int diagonal,int sign) {
    if((beam!=1 && beam!=2) || !finite_number(macro_charge) || !(macro_charge>0))
      throw std::invalid_argument("invalid snapshot source or macro charge");
    auto& rho=beam==1?rho1_:rho2_;
    std::fill(rho.begin(),rho.end(),PHI_FLOAT(0));
    SnapshotDeposit counts;counts.source_particles=particles.size();
    for(const PARTICLE* original:particles) {
      if(!original)throw std::logic_error("null snapshot source particle");
      // ABSTRACT_PARTICLE's copy constructor detaches adopted XY storage.
      // Spin does not enter electrostatic charge deposition, so base copying
      // suffices for both native PARTICLE and PARTICLE_WITH_SPIN primaries.
      PARTICLE copy(*original);
      synchronize_display_copy(copy,slice,diagonal,n_cell_z_,timestep_,step_,max_z_,sign);
      float x,y,hx,hy;int ix,iy;copy.XYposition(x,y);
      if(!finite_number(x) || !finite_number(y) || !finite_number(copy.z()))
        throw std::domain_error("non-finite synchronized snapshot source");
      if(!counts.has_bounds) {counts.x_min=counts.x_max=x;counts.y_min=counts.y_max=y;counts.has_bounds=true;}
      else {counts.x_min=std::min(counts.x_min,double(x));counts.x_max=std::max(counts.x_max,double(x));
        counts.y_min=std::min(counts.y_min,double(y));counts.y_max=std::max(counts.y_max,double(y));}
      if(coordinatesInGridRange(x,y))++counts.gather_inside_particles;
      else ++counts.gather_outside_particles;
      if(particleInGrid(x,y,hx,hy,ix,iy)) {
        // Same native CIC helper, operands, and source iteration order as the
        // serial primary path. Charge normalization is supplied by live grid.
        assignChargeToCIC(rho.data(),ix,iy,hx,hy,macro_charge);
        ++counts.deposited_particles;
      } else ++counts.outside_particles;
    }
    return counts;
  }
  CrossSections solve(std::size_t maximum_axis) {
    field_.fold_fft(rho1_.data(),rho2_.data(),-1.0f);
    return sample_cross_sections(*this,field_.get_phi(1),field_.get_phi(2),maximum_axis);
  }
  const std::vector<PHI_FLOAT>& rho(int beam) const {
    if(beam!=1 && beam!=2)throw std::invalid_argument("invalid snapshot rho source");
    return beam==1?rho1_:rho2_;
  }
  const PHI_FLOAT* phi(int beam) const {return field_.get_phi(beam);}
};
} }
#endif
