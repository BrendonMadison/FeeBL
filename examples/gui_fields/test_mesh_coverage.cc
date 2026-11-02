// Native full-bunch diagnostic: production CIC/FFT parity, independent
// endpoint drift arithmetic, all slice bands, and live-state noninterference.
#include "colliderMeshCoverageCPP.h"
#include <sstream>
#include <array>
#include <cstdlib>
#include <iostream>
#include <numeric>

void mesh_coverage_production_cic(std::size_t,float,float,int,std::vector<PHI_FLOAT>&,double&);
namespace {
unsigned checks=0;
void check(bool ok,const char* reason) {
  ++checks;if(!ok){std::cerr << "mesh coverage test: " << reason << '\n';std::exit(1);}
}
void near(double a,double b,const char* reason,double tol=2e-6) {
  check(std::abs(a-b)<=tol*std::max(1.0,std::abs(b)),reason);
}
template<class F>void rejects(F action) {
  bool did=false;try{action();}catch(const std::exception&){did=true;}
  check(did,"invalid input was accepted");
}
std::array<float,6> state(const PARTICLE& p) {
  float x,y,vx,vy;p.XYposition(x,y);p.velocities(vx,vy);
  return {{x,y,p.z(),vx,vy,p.energy()}};
}
class NativeOracle final:public GENERAL_GRID {
  FFT_SERVER fourier_;
  void distribute_coherent_particles(int,int,int) override {}
  void distribute_particles(int,int,int,int) override {}
 public:
  NativeOracle() {
    n_cell_x_=16;n_cell_y_=12;n_cell_z_=8;timestep_=2;
    champ_=FIELD(2,n_cell_x_,n_cell_y_,&fourier_);
    slice_of_beam_[0].resize(n_cell_x_,n_cell_y_);
    slice_of_beam_[1].resize(n_cell_x_,n_cell_y_);
    set_nb_macroparticles(1,4);set_nb_macroparticles(2,8);
    // Exact production initialization establishes the authoritative kernel and
    // macro charges 3.25 and 6.5 (independently different source weights).
    init_grid_phys(13.0f,52.0f,16.0f,18.0f,40.0f,-1.0f,&fourier_);
  }
  void deposit_ready(int beam,const std::vector<PARTICLE>& copies) {
    auto& source=slice_of_beam_[beam-1];source.razRho();
    for(const auto& p:copies) {
      float x,y,hx,hy;int ix,iy;p.XYposition(x,y);
      if(particleInGrid(x,y,hx,hy,ix,iy))
        source.assignChargeToCIC(ix,iy,hx,hy,1.0f);
    }
  }
  std::vector<PHI_FLOAT> rho(int beam)const {
    const auto* values=beam==1?get_rho1():get_rho2();
    return {values,values+get_n_cell_x()*get_n_cell_y()};
  }
  std::vector<PHI_FLOAT> phi(int beam)const {
    const auto* values=beam==1?get_phi1():get_phi2();
    return {values,values+get_n_cell_x()*get_n_cell_y()};
  }
  void solve(){computeFields(2,-1.0f,nullptr);}
};

void test_bound_actual_deposit_and_pre_kick() {
  NativeOracle native;
  PARTICLE a,b,c;a.setParticle(-2.25f,0.3f,-35,0.04f,-0.03f,125);
  b.setParticle(-14.5f,0,-35,0.0f,0.0f,125); // deposit admitted, gather not safe
  c.setParticle(1e6f,0,-35,0,0,125);
  float adopted[2];a.set_position_slot(adopted);
  const auto before_a=state(a),before_b=state(b),before_c=state(c);
  std::vector<PARTICLE*> particles{&a,&b,&c};
  native.deposit_ready(1,{a,b,c});native.deposit_ready(2,{});native.solve();
  const auto rho=native.rho(1),phi=native.phi(1);
  const auto scan=feebl::collider_fields::scan_mesh_sources(native,particles);
  check(scan.stats.source_particles==3 && scan.stats.deposited_particles==2 && scan.stats.outside_particles==1,
      "source-deposit coverage counts");
  check(scan.stats.gather_inside_particles==1 && scan.stats.gather_outside_particles==2,
      "deposit and safe gather domains must be distinguished");
  check(scan.stats.has_bounds && scan.stats.x_min==-14.5 && scan.stats.x_max==1e6,
      "source extrema must include excluded particles");
  const auto bound=feebl::collider_fields::bind_native_deposition(scan,10,4,12,5,
      native.get_rho1(),rho.size(),native.get_macro_charge(1));
  near(bound.native_rho_charge,6.5,"native charge sum from actual deposited rho");
  check(bound.expected_rho_charge==6.5 && bound.charge_relative_error<1e-14,"charge normalization closure");
  feebl::collider_fields::verify_pre_kick_sources(scan,
      feebl::collider_fields::scan_mesh_sources(native,particles));
  check(native.rho(1)==rho && native.phi(1)==phi && state(a)==before_a &&
      state(b)==before_b && state(c)==before_c && adopted[0]==before_a[0],
      "readonly coverage mutated native arrays or particles");
  rejects([&]{feebl::collider_fields::bind_native_deposition(scan,10,4,13,5,native.get_rho1(),rho.size(),3.25);});
  rejects([&]{feebl::collider_fields::bind_native_deposition(scan,10,4,12,6,native.get_rho1(),rho.size(),3.25);});
  rejects([&]{feebl::collider_fields::bind_native_deposition(scan,10,4,9,5,native.get_rho1(),rho.size(),3.25);});
  auto bad_rho=rho;bad_rho[0]+=0.01;
  rejects([&]{feebl::collider_fields::bind_native_deposition(scan,10,4,12,5,bad_rho.data(),bad_rho.size(),3.25);});
  rejects([&]{feebl::collider_fields::bind_native_deposition(scan,10,4,12,5,native.get_rho1(),rho.size(),6.5);});
  a.apply_position_offset(0.001f,0);
  rejects([&]{feebl::collider_fields::verify_pre_kick_sources(scan,
      feebl::collider_fields::scan_mesh_sources(native,particles));});
  a.apply_position_offset(-0.001f,0);
  std::vector<PARTICLE*> reordered{&b,&a,&c};
  rejects([&]{feebl::collider_fields::verify_pre_kick_sources(scan,
      feebl::collider_fields::scan_mesh_sources(native,reordered));});
  std::ostringstream json;feebl::collider_fields::write_coverage_members(json,bound.stats);
  check(json.str().find("\"outside_particles\":1")!=std::string::npos,"coverage serialization uses actual missed count");
}

void test_empty_and_snapshot_aggregate() {
  NativeOracle native;
  native.deposit_ready(1,{});native.deposit_ready(2,{});native.solve();
  const auto empty=feebl::collider_fields::scan_mesh_sources(native,{});
  const auto bound=feebl::collider_fields::bind_native_deposition(empty,3,1,3,1,
      native.get_rho1(),std::size_t(native.get_n_cell_x())*native.get_n_cell_y(),native.get_macro_charge(1));
  check(!bound.stats.has_bounds && bound.stats.source_particles==0 && bound.charge_relative_error==0,
      "empty source coverage must be explicitly empty");
  std::ostringstream json;feebl::collider_fields::write_coverage_members(json,bound.stats);
  check(json.str().find("\"source_bounds_nm\":null")!=std::string::npos,"empty extrema must be null");
  PARTICLE a,b;a.setParticle(0,0,-35,0.5f,0.5f,125);b.setParticle(1e6f,-1e6f,-35,0,0,125);
  feebl::collider_fields::SnapshotGrid scratch(native);
  const auto first=scratch.deposit(1,{&a,&b},native.get_macro_charge(1),0,14,1);
  const auto coverage=feebl::collider_fields::snapshot_coverage(first);
  check(coverage.source_particles==2 && coverage.deposited_particles+coverage.outside_particles==2,
      "snapshot source coverage sum");
  // Inactive slice0 at d14: native display drift distance=(14+1-0-8)*5=35.
  near(coverage.x_min,17.5,"snapshot extrema must use synchronized copied x");
  near(coverage.y_max,17.5,"snapshot extrema must use synchronized copied y");
  check(coverage.x_max==1e6 && coverage.y_min==-1e6,"snapshot extrema include all excluded copies");
  check(coverage.gather_inside_particles+coverage.gather_outside_particles==2,"snapshot gather counts");
  feebl::collider_fields::CoverageStats total;total.add(coverage);total.add(coverage);
  check(total.source_particles==4 && total.outside_particles==2*coverage.outside_particles &&
      total.x_min==coverage.x_min && total.x_max==coverage.x_max,"snapshot frame aggregation");
  check(state(a)[0]==0 && state(a)[1]==0,"snapshot coverage changed live source");
}

void test_native_float_corner_and_realistic_macrocharge() {
  const float hx=0.8873465f,hy=0.9372821f;
  const double exact_product=double(hx)*double(hy);
  const float rounded_product=hx*hy;
  const double corner_bias=double(rounded_product)-exact_product;
  check(std::abs(corner_bias)>1e-10,"fixture must expose the old invalid charge tolerance");
  double max_strict_error=0,max_production_error=0;
  for(int macros:{9976,10000})for(std::size_t count:{std::size_t(1),std::size_t(7),
      std::size_t(161),std::size_t(9976)}) {
    SLICE_ON_GRID strict;strict.resize(4,4);strict.set_macroparticles(macros);
    strict.update_nb_part_per_macro(2.0e10f);
    for(std::size_t i=0;i<count;++i)strict.assignChargeToCIC(1,1,hx,hy,1.0f);
    std::vector<PHI_FLOAT> fast;double fast_macro=0;
    mesh_coverage_production_cic(count,hx,hy,macros,fast,fast_macro);
    check(fast_macro==strict.get_nb_part_per_macro(),"realistic macro charge differs across FP contracts");
    feebl::collider_fields::SourceScan scan;
    scan.stats.source_particles=scan.stats.deposited_particles=count;
    const auto a=feebl::collider_fields::bind_native_deposition(scan,0,0,count,0,
        strict.get_rho(),16,strict.get_nb_part_per_macro());
    const auto b=feebl::collider_fields::bind_native_deposition(scan,0,0,count,0,
        fast.data(),fast.size(),fast_macro);
    max_strict_error=std::max(max_strict_error,a.charge_relative_error);
    max_production_error=std::max(max_production_error,b.charge_relative_error);
    check(a.charge_relative_error>1e-10 && b.charge_relative_error>1e-10,
        "realistic fixture did not reproduce old checker failure");
    check(a.charge_relative_error<=a.charge_relative_error_bound &&
        b.charge_relative_error<=b.charge_relative_error_bound,"derived native roundoff bound");
    // Independent signed bias oracle, with only double accumulation error.
    const double signed_bias=(a.native_rho_charge-a.expected_rho_charge)/a.expected_rho_charge;
    check(std::abs(signed_bias-corner_bias)<1e-11,"nominal charge deviation not explained by native float corner");
    auto broken=fast;broken[5]+=b.expected_rho_charge*1e-5;
    bool rejected=false;
    try {feebl::collider_fields::bind_native_deposition(scan,0,0,count,0,broken.data(),broken.size(),fast_macro);}
    catch(const std::logic_error& error) {
      const std::string message=error.what();
      rejected=message.find("actual=")!=std::string::npos &&
          message.find("nominal=")!=std::string::npos &&
          message.find("native_roundoff_bound=")!=std::string::npos;
    }
    check(rejected,"material charge corruption must fail with quantitative evidence");
  }
  check(feebl::collider_fields::native_cic_charge_roundoff_bound(0)==0,"empty source bound must be exact zero");
  std::cout << std::setprecision(17) << "native float-corner charge bias=" << corner_bias
      << " strict_max_relative_error=" << max_strict_error
      << " production_max_relative_error=" << max_production_error
      << " bound_at_9976=" << feebl::collider_fields::native_cic_charge_roundoff_bound(9976) << '\n';
}
}
int main() {
  test_bound_actual_deposit_and_pre_kick();test_empty_and_snapshot_aggregate();
  test_native_float_corner_and_realistic_macrocharge();
  std::cout << "PASS: " << checks << " mesh coverage checks; actual counter binding, native rho charge, source/gather separation, exact pre-kick binding, all-copy extrema and noninterference\n";
}
