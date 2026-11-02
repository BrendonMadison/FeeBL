#ifndef FEEBL_COLLIDER_MESH_COVERAGE_CPP_H
#define FEEBL_COLLIDER_MESH_COVERAGE_CPP_H
#include "colliderSnapshotFieldsCPP.h"
#include <array>
#include <ostream>
#include <sstream>
#include <iomanip>

namespace feebl { namespace collider_fields {
struct CoverageStats {
  std::uint64_t source_particles=0,deposited_particles=0,outside_particles=0;
  std::uint64_t gather_inside_particles=0,gather_outside_particles=0;
  bool has_bounds=false;
  double x_min=0,x_max=0,y_min=0,y_max=0;
  void point(float x,float y) {
    if(!finite_number(x)||!finite_number(y))throw std::domain_error("non-finite mesh coverage source");
    if(!has_bounds) {x_min=x_max=x;y_min=y_max=y;has_bounds=true;}
    else {x_min=std::min(x_min,double(x));x_max=std::max(x_max,double(x));
      y_min=std::min(y_min,double(y));y_max=std::max(y_max,double(y));}
  }
  void add(const CoverageStats& b) {
    source_particles+=b.source_particles;deposited_particles+=b.deposited_particles;
    outside_particles+=b.outside_particles;gather_inside_particles+=b.gather_inside_particles;
    gather_outside_particles+=b.gather_outside_particles;
    if(b.has_bounds) {point(b.x_min,b.y_min);point(b.x_max,b.y_max);}
  }
};
struct SourceScan {CoverageStats stats;std::vector<float> xy;};
inline SourceScan scan_mesh_sources(GENERAL_GRID& grid,const std::vector<PARTICLE*>& particles) {
  SourceScan scan;scan.stats.source_particles=particles.size();scan.xy.reserve(2*particles.size());
  for(const auto* p:particles) {
    if(!p)throw std::logic_error("null mesh coverage source");
    float x,y,hx,hy;int ix,iy;p->XYposition(x,y);scan.stats.point(x,y);
    scan.xy.push_back(x);scan.xy.push_back(y);
    if(grid.particleInGrid(x,y,hx,hy,ix,iy))++scan.stats.deposited_particles;
    else ++scan.stats.outside_particles;
    if(grid.coordinatesInGridRange(x,y))++scan.stats.gather_inside_particles;
    else ++scan.stats.gather_outside_particles;
  }
  return scan;
}
struct SourceCoverage {
  CoverageStats stats;
  double macro_charge=0,native_rho_charge=0,expected_rho_charge=0,charge_relative_error=0;
  double charge_relative_error_bound=0;
};
// The native fourth CIC corner is h_x*h_y*charge. Both h operands are
// float, so that product rounds in binary32 before promotion to PHI_FLOAT;
// the other three corners contain double 1.0. Their four weights therefore
// need not sum to exactly one. Do not "repair" native weights for a diagnostic.
// One such product has <=u32 relative error and hx*hy<=1, giving <=u32
// relative charge error. A conservative gamma_(8N+32) bounds the remaining
// positive PHI_FLOAT products/additions and reduction: at most 4N deposited
// corner additions, 4N nonzero-cell reduction additions, plus fixed weight
// arithmetic. Combining the two error factors yields u32+(1+u32)*gamma.
inline double native_cic_charge_roundoff_bound(std::uint64_t deposited) {
  if(!deposited)return 0.0;
  const double u32=0.5*std::numeric_limits<float>::epsilon();
  const double uphi=0.5*std::numeric_limits<PHI_FLOAT>::epsilon();
  const double ku=(8.0*double(deposited)+32.0)*uphi;
  if(!(ku<1.0))throw std::overflow_error("native CIC charge roundoff bound is unbounded");
  const double gamma=ku/(1.0-ku);
  return u32+(1.0+u32)*gamma;
}
inline SourceCoverage bind_native_deposition(const SourceScan& scan,
    long before_in,long before_out,long after_in,long after_out,
    const PHI_FLOAT* rho,std::size_t cells,double macro_charge) {
  if(!rho || !finite_number(macro_charge) || !(macro_charge>0) ||
      before_in<0 || before_out<0 || after_in<before_in || after_out<before_out)
    throw std::invalid_argument("invalid native deposition coverage binding");
  if(std::uint64_t(after_in-before_in)!=scan.stats.deposited_particles ||
      std::uint64_t(after_out-before_out)!=scan.stats.outside_particles)
    throw std::logic_error("coverage scan does not match actual native deposition counters");
  long double charge=0;
  for(std::size_t i=0;i<cells;++i) {
    if(!finite_number(rho[i]) || rho[i]<0)
      throw std::domain_error("invalid primary-only native charge array");
    charge+=rho[i];
  }
  SourceCoverage result;result.stats=scan.stats;result.macro_charge=macro_charge;
  result.native_rho_charge=double(charge);
  result.expected_rho_charge=macro_charge*scan.stats.deposited_particles;
  result.charge_relative_error=result.expected_rho_charge>0?
      std::abs(result.native_rho_charge-result.expected_rho_charge)/result.expected_rho_charge:
      std::abs(result.native_rho_charge);
  result.charge_relative_error_bound=native_cic_charge_roundoff_bound(scan.stats.deposited_particles);
  if(!finite_number(result.charge_relative_error) ||
      result.charge_relative_error>result.charge_relative_error_bound) {
    std::ostringstream message;message << std::setprecision(17)
        << "native rho charge closure failed: deposited=" << scan.stats.deposited_particles
        << " macro_charge=" << macro_charge << " actual=" << result.native_rho_charge
        << " nominal=" << result.expected_rho_charge << " relative_error=" << result.charge_relative_error
        << " native_roundoff_bound=" << result.charge_relative_error_bound;
    throw std::logic_error(message.str());
  }
  return result;
}
inline void verify_pre_kick_sources(const SourceScan& deposited,const SourceScan& before_kick) {
  if(deposited.xy!=before_kick.xy ||
      deposited.stats.source_particles!=before_kick.stats.source_particles ||
      deposited.stats.gather_inside_particles!=before_kick.stats.gather_inside_particles ||
      deposited.stats.gather_outside_particles!=before_kick.stats.gather_outside_particles)
    throw std::logic_error("primary source coordinates changed between deposition and pre-kick audit");
}
inline CoverageStats snapshot_coverage(const SnapshotDeposit& deposit) {
  CoverageStats stats;stats.source_particles=deposit.source_particles;
  stats.deposited_particles=deposit.deposited_particles;stats.outside_particles=deposit.outside_particles;
  stats.gather_inside_particles=deposit.gather_inside_particles;
  stats.gather_outside_particles=deposit.gather_outside_particles;
  if(deposit.has_bounds) {stats.point(deposit.x_min,deposit.y_min);stats.point(deposit.x_max,deposit.y_max);}
  return stats;
}
struct MeshEncounterCoverage {
  int id=0,diagonal=0,beam1_slice=0,beam2_slice=0,substep_index=0;
  std::array<SourceCoverage,2> beam;
};
struct MeshSnapshotCoverage {int id=0,diagonal=0;std::array<CoverageStats,2> beam;};
inline void write_coverage_members(std::ostream& out,const CoverageStats& c) {
  out << "\"source_particles\":" << c.source_particles
      << ",\"deposited_particles\":" << c.deposited_particles
      << ",\"outside_particles\":" << c.outside_particles
      << ",\"gather_inside_particles\":" << c.gather_inside_particles
      << ",\"gather_outside_particles\":" << c.gather_outside_particles
      << ",\"source_bounds_nm\":";
  if(c.has_bounds)out << '[' << c.x_min << ',' << c.x_max << ',' << c.y_min << ',' << c.y_max << ']';
  else out << "null";
}
} }
#endif
