#include "particleBeamCPP.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

namespace {
int failures=0, checks=0;
void check(bool ok,const char* text) { ++checks; if (!ok) { ++failures; std::cerr<<"FAIL "<<text<<'\n'; } }
template<class E,class F> bool raises(F f) { try {f();} catch(const E&) {return true;} catch(...) {} return false; }
struct State { PARTICLE* pointer; PRIMARY_PARTICLE_ID id; int slice; std::array<float,6> p; std::array<double,3> spin; };
std::vector<State> states(const PARTICLE_BEAM& b) {
  std::vector<State> out;
  for(int s=0;s<b.number_of_slices();++s) {
    const auto& particles=b.getParticleVector(s);
    const auto& ids=b.getParticleIdVectorStrict(s);
    for(std::size_t k=0;k<particles.size();++k) {
      State x; x.pointer=particles[k]; x.id=ids[k]; x.slice=s;
      particles[k]->XYposition(x.p[0],x.p[1]); x.p[2]=particles[k]->z();
      particles[k]->velocities(x.p[3],x.p[4]); x.p[5]=particles[k]->energy();
      for(int j=0;j<3;++j) x.spin[j]=particles[k]->getSpin()(j);
      out.push_back(x);
    }
  }
  return out;
}
void initialize(PARTICLE_BEAM& b) {
  b.init_particles(16,100,10,1,0,0,1e-6f,1e-6f,10,125,0,0);
}
void equal_lists(const std::vector<State>& a,const std::vector<State>& b,const char* why) {
  bool ok=a.size()==b.size();
  for(std::size_t i=0;ok && i<a.size();++i)
    ok=a[i].pointer==b[i].pointer && a[i].id==b[i].id && a[i].slice==b[i].slice &&
       std::memcmp(a[i].p.data(),b[i].p.data(),6*sizeof(float))==0 && a[i].spin==b[i].spin;
  check(ok,why);
}
void test_beam(unsigned label) {
  TRIDVECTOR polar; polar.setComponents(.2,-.3,.8);
  RNDM random(61028), twin_random(61028);
  PARTICLE_BEAM beam(4,1,polar,&random,label), twin(4,1,polar,&twin_random,label);
  initialize(beam); initialize(twin);
  const auto initial=states(beam);
  const auto noop=beam.reslice_after_incoming_map(10);
  check(noop.particles==16 && noop.moved==0,"unchanged coordinates have no migration");
  equal_lists(initial,states(beam),"no-op preserves pointer/identity/order/state/spin exactly");
  check(random.rndm()==twin_random.rndm(),"reslicing does not consume random numbers");
  const float targets[]={-20,-10,-0.0f,0,10,std::nextafter(20.0f,0.0f),
    std::nextafter(-20.0f,0.0f),std::nextafter(-10.0f,-20.0f),std::nextafter(10.0f,0.0f),
    -15,15,-5,5,-19,19,1};
  for(std::size_t i=0;i<initial.size();++i) initial[i].pointer->setZ(targets[i]);
  const auto mapped=states(beam);
  const unsigned long count=beam.numberOfParticles();
  const auto receipt=beam.reslice_after_incoming_map(10);
  const auto after=states(beam);
  check(receipt.particles==count && beam.numberOfParticles()==count,"population bookkeeping conserved");
  check(receipt.moved>0 && receipt.maximum_slice_shift>0 && receipt.maximum_abs_z_nm==20,"migration receipt reports actual movement");
  std::map<PRIMARY_PARTICLE_ID,State> indexed;
  for(const auto& p:mapped) indexed.emplace(p.id,p);
  for(const auto& p:after) {
    const auto& original=indexed.at(p.id);
    check(p.pointer==original.pointer && std::memcmp(p.p.data(),original.p.data(),6*sizeof(float))==0 && p.spin==original.spin,
          "all six coordinates, spin and physical identity conserved");
    check(p.slice==static_cast<int>(std::floor(double(p.p[2])/10+2)),"boundary maps to half-open physical bin");
    PRIMARY_PARTICLE_ID_FIELDS decoded;
    check(decode_primary_particle_id(p.id,decoded) && decoded.beam_label==label,"beam label preserved");
  }
  for(int s=0;s<4;++s) {
    std::vector<PRIMARY_PARTICLE_ID> expected,actual;
    for(const auto& p:mapped) if(static_cast<int>(std::floor(double(p.p[2])/10+2))==s) expected.push_back(p.id);
    for(const auto& p:after) if(p.slice==s) actual.push_back(p.id);
    check(expected==actual,"rebucketing preserves stable original traversal order within destination");
  }
  const auto repeat=beam.reslice_after_incoming_map(10);
  check(repeat.moved==0,"reslicing is idempotent");
  equal_lists(after,states(beam),"repeat is byte-identical in state and order");

  PARTICLE* probe=after.front().pointer;
  const float saved=probe->z();
  for(float bad:{20.0f,std::nextafter(-20.0f,-21.0f),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    probe->setZ(bad); const auto before=states(beam);
    check(raises<std::out_of_range>([&](){beam.reslice_after_incoming_map(10);}),"outside/open-upper/nonfinite z fails closed");
    equal_lists(before,states(beam),"failed preflight changes no container, pointer or value");
    probe->setZ(saved);
  }
  check(raises<std::invalid_argument>([&](){beam.reslice_after_incoming_map(0);}),"zero spacing refused");
  check(raises<std::invalid_argument>([&](){beam.reslice_after_incoming_map(-1);}),"negative spacing refused");
  check(raises<std::invalid_argument>([&](){beam.reslice_after_incoming_map(std::numeric_limits<float>::infinity());}),"nonfinite spacing refused");
  const auto pre_adopt=states(beam);
  beam.adopt_xy();
  equal_lists(pre_adopt,states(beam),"post-reslice SoA adoption preserves state and pointers");
  check(raises<std::logic_error>([&](){beam.reslice_after_incoming_map(10);}),"adopted position storage prevents unsafe rebucketing");
  for(int s=0;s<4;++s) {
    const auto& p=beam.getParticleVector(s); const float* xy=beam.slice_xy(s);
    check(beam.slice_xy_count(s)==p.size(),"SoA shape matches resliced ownership");
    for(std::size_t k=0;k<p.size();++k) {float x,y;p[k]->XYposition(x,y);check(xy[2*k]==x&&xy[2*k+1]==y,"SoA positions reference correct resliced particle");}
  }
}
}
int main() {
  test_beam(1); test_beam(2);
  std::cout<<"incoming-map reslice: "<<(failures?"FAIL":"PASS")<<" checks="<<checks<<" failures="<<failures<<'\n';
  return failures?1:0;
}
