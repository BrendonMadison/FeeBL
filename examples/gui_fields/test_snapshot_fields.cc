// Native full-bunch diagnostic: production CIC/FFT parity, independent
// endpoint drift arithmetic, all slice bands, and live-state noninterference.
#include "colliderSnapshotFieldsCPP.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace {
unsigned checks=0;
void check(bool ok,const char* reason) {
  ++checks;if(!ok){std::cerr << "snapshot field test: " << reason << '\n';std::exit(1);}
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

void test_display_sync_and_full_bands() {
  constexpr int nz=8,nt=2;constexpr float step=2.5f,dz=10,maxz=40,dt=5;
  for(int d=0;d<=2*nz-2;++d)for(int sign:{1,-1}) {
    double lowest=1e99,highest=-1e99;
    for(int slice=0;slice<nz;++slice) {
      const auto band=feebl::collider_fields::snapshot_band(slice,d,nz,dz,sign);
      lowest=std::min(lowest,band.low);highest=std::max(highest,band.high);
      for(int boundary=0;boundary<3;++boundary) {
        PARTICLE original;original.setParticle(1.25f,-2.5f,
            -maxz+(slice+0.5f*boundary)*dz,0.125f,-0.25f,125.0f);
        float adopted[2];original.set_position_slot(adopted);
        const auto before=state(original);PARTICLE copy(original);
        feebl::collider_fields::synchronize_display_copy(copy,slice,d,nz,nt,step,maxz,sign);
        const auto after=state(copy);
        // Independent physical display elapsed drift outside the active slice
        // range, including d=nz-1 -> nz transition and both late endpoints.
        float drift=0;
        if(d<slice)drift=(d+1-slice)*dt;
        else if(d>=slice+nz)drift=(d+1-slice-nz)*dt;
        near(after[0],before[0]+before[3]*drift,"copy x at display time");
        near(after[1],before[1]+before[4]*drift,"copy y at display time");
        near(after[2],sign*((slice+0.5*boundary)*dz+(1-d)*dt),"copy z at display time");
        check(state(original)==before && adopted[0]==before[0] && adopted[1]==before[1],
            "copy drift changed original/adopted XY");
        if(boundary==1)near(after[2],band.centre,"full-source band centre vs native dump");
        else near(after[2],((boundary==0)==(sign==1))?band.low:band.high,
            "full-source band boundary vs native dump");
      }
    }
    near(highest-lowest,nz*dz,"source longitudinal width must remain fixed");
  }
  for(int nz2=1;nz2<=128;++nz2) {
    const std::size_t cap=feebl::collider_fields::snapshot_axis_cap(nz2);
    const int last=2*nz2-2,stride=std::max(1,(last+62)/63);
    std::size_t frames=0;for(int d=0;d<=last;++d)if(d==last||d%stride==0)++frames;
    check(cap>=2&&cap<=64,"snapshot sample axis range");
    check(4*cap*frames*nz2<=1048576,"snapshot scalar bound");
  }
  check(feebl::collider_fields::snapshot_axis_cap(129)==0,"unsupported slice count");
  rejects([]{feebl::collider_fields::snapshot_band(8,0,8,10,1);});
  rejects([]{feebl::collider_fields::snapshot_band(0,0,8,10,0);});
}

void test_private_native_parity_and_noninterference() {
  NativeOracle live,oracle;
  PARTICLE a,b,c,e;a.setParticle(-2.25f,0.3f,-35,0.04f,-0.03f,125);
  b.setParticle(3.125f,-1.75f,-35,-0.02f,0.05f,125);
  c.setParticle(1e6f,0,-35,0,0,125);e.setParticle(-0.5f,4.125f,-35,0.06f,-0.04f,125);
  float adopted[2];a.set_position_slot(adopted);
  std::vector<PARTICLE*> p1{&a,&b,&c},p2{&e,&b};
  std::vector<PARTICLE> originals{a,b,c,e};
  live.deposit_ready(1,{a,e});live.deposit_ready(2,{b});live.solve();
  const auto live_rho1=live.rho(1),live_rho2=live.rho(2),live_phi1=live.phi(1),live_phi2=live.phi(2);
  feebl::collider_fields::SnapshotGrid scratch(live);
  check(scratch.get_delta_x()==live.get_delta_x() && scratch.get_delta_y()==live.get_delta_y(),
      "private geometry differs from native grid");
  near(scratch.get_field_kernel_factor(),live.get_field_kernel_factor(),"private Green normalization",0);
  for(int d:{0,3,7,8,11,14})for(int slice:{0,3,7}) {
    std::vector<PARTICLE> copies1,copies2;
    for(int beam:{1,2}) {
      const auto& ptrs=beam==1?p1:p2;auto& copies=beam==1?copies1:copies2;
      for(const auto* p:ptrs) {
        copies.push_back(*p);auto& copy=copies.back();
        // Oracle uses the production dump entrypoints, independently of the
        // reconstruction helper under test.
        const float dt=live.get_timestep()*live.get_step();
        if(d<8 && slice>d)copy.set_z_velocity_corrected_for_dump(slice,d,0,dt,live.get_max_z(),beam==1?1:-1);
        else if(d>=8 && slice<=d-8)copy.set_z_velocity_corrected_for_dump(slice,d,8,dt,live.get_max_z(),beam==1?1:-1);
        else copy.set_z_for_dump(d,dt,live.get_max_z(),beam==1?1:-1);
      }
    }
    const auto counts1=scratch.deposit(1,p1,live.get_macro_charge(1),slice,d,1);
    const auto counts2=scratch.deposit(2,p2,live.get_macro_charge(2),slice,d,-1);
    check(counts1.source_particles==3 && counts1.deposited_particles==2 && counts1.outside_particles==1,
        "source1 deposit/outside coverage");
    check(counts2.source_particles==2 && counts2.deposited_particles==2 && counts2.outside_particles==0,
        "source2 deposit/outside coverage");
    oracle.deposit_ready(1,copies1);oracle.deposit_ready(2,copies2);oracle.solve();
    check(scratch.rho(1)==oracle.rho(1) && scratch.rho(2)==oracle.rho(2),
        "private rho is not exactly native CIC deposition");
    near(std::accumulate(scratch.rho(1).begin(),scratch.rho(1).end(),0.0),2*live.get_macro_charge(1),
        "source1 deposited macrocharge conservation");
    near(std::accumulate(scratch.rho(2).begin(),scratch.rho(2).end(),0.0),2*live.get_macro_charge(2),
        "source2 deposited macrocharge conservation");
    const auto sample=scratch.solve(8);
    const auto expected=feebl::collider_fields::sample_cross_sections(oracle,8);
    for(int beam:{1,2}) {
      const auto phi=oracle.phi(beam);
      for(std::size_t i=0;i<phi.size();++i)
        check(scratch.phi(beam)[i]==phi[i],"private potential not bit-exact native FFT");
      check(*std::max_element(phi.begin(),phi.end())!=0 || *std::min_element(phi.begin(),phi.end())!=0,"native potential parity oracle is vacuous");
    }
    for(const auto& pair:{std::make_pair(&sample.beam1_zx,&expected.beam1_zx),
        std::make_pair(&sample.beam1_zy,&expected.beam1_zy),
        std::make_pair(&sample.beam2_zx,&expected.beam2_zx),
        std::make_pair(&sample.beam2_zy,&expected.beam2_zy)})
      for(std::size_t i=0;i<pair.first->size();++i)
        check((*pair.first)[i]==(*pair.second)[i],"private cross-section not bit-exact native field");
    check(live.rho(1)==live_rho1 && live.rho(2)==live_rho2 && live.phi(1)==live_phi1 && live.phi(2)==live_phi2,
        "diagnostic changed the authoritative rho/phi");
    for(std::size_t i=0;i<originals.size();++i)
      check(state(*std::vector<PARTICLE*>{&a,&b,&c,&e}[i])==state(originals[i]),"source particle changed");
  }
  // Native packed-complex FFT can leave roundoff in an individually empty
  // source. Do not erase native values; measure its size relative to the
  // nonempty source and prove no material source mixing.
  scratch.deposit(1,{},live.get_macro_charge(1),0,0,1);
  scratch.deposit(2,p2,live.get_macro_charge(2),0,0,-1);
  const auto single=scratch.solve(8);
  double empty_max=0,filled_max=0;
  for(double v:single.beam1_zx)empty_max=std::max(empty_max,v);
  for(double v:single.beam1_zy)empty_max=std::max(empty_max,v);
  for(double v:single.beam2_zx)filled_max=std::max(filled_max,v);
  for(double v:single.beam2_zy)filled_max=std::max(filled_max,v);
  check(filled_max>0 && empty_max/filled_max<1e-12,"empty-source packed FFT leakage is material");
  std::cout << "native empty-source leakage / nonempty maximum=" << empty_max/filled_max << '\n';
  const auto zero1=scratch.deposit(1,{},live.get_macro_charge(1),0,0,1);
  scratch.deposit(2,{},live.get_macro_charge(2),0,0,-1);
  const auto empty=scratch.solve(8);
  check(zero1.source_particles==0 && zero1.deposited_particles==0 && zero1.outside_particles==0,"empty source counts");
  for(const auto* line:{&empty.beam1_zx,&empty.beam1_zy,&empty.beam2_zx,&empty.beam2_zy})
    for(double v:*line)check(v==0,"empty source must produce computed zero field");
  rejects([&]{scratch.deposit(1,p1,-1,0,0,1);});
  rejects([&]{scratch.deposit(3,p1,1,0,0,1);});
}
}
int main() {
  test_display_sync_and_full_bands();test_private_native_parity_and_noninterference();
  std::cout << "PASS: " << checks << " snapshot field checks; native CIC/FFT parity, all-slice geometry, display synchronization, bounded samples, and live-state noninterference\n";
}
