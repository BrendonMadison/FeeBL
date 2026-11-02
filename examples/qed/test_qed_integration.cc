// Bounded tests of the QED integration boundaries and sampling failure paths.
// No beam collision, field mesh, or stochastic convergence scan is run here.
#include "qedbg_pair_kernels.h"
#include "qedbg_photon_polarisation.h"
#include "pairCrossSectionCPP.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Action> void requires_failure(Action action, const char* message) {
  bool failed=false;
  try { action(); } catch (const std::exception&) { failed=true; }
  require(failed,message);
}
void close(double a,double b,double relative,const char* message) {
  require(std::abs(a-b)<=relative*std::max(std::abs(a),std::abs(b)),message);
}
}

int run_tests() {
  using namespace qedbg;
  require(kconst::me==EMASS && kconst::mmu==MUMASS,
          "creation and transport must use the same rest masses");
  require(feebl::pair_production::breit_wheeler_m2(EMASS*EMASS)==0,
          "the default cross section must vanish at threshold");

  PhotonProposal prop{1e-9, 250000.0, .00232461, EMASS, false};
  const double x=.01, q2=1e-8;
  const double below=std::nextafter(prop.xmin,0.0);
  require(prop.density(below,q2)==0.0,"alternative channel must respect xmin");
  require(prop.density_from_sample(below,q2)>0,
          "rounding must not delete the generating channel");
  const double qmin=EMASS*EMASS*x*x/(1-x);
  require(prop.density(x,std::nextafter(qmin,0.0))==0.0,
          "alternative channel must respect Q2 minimum");
  require(prop.density_from_sample(x,std::nextafter(qmin,0.0))>0,
          "generating Q2 must survive a storage-boundary discrepancy");
  close(prop.density(x,q2),prop.density_from_sample(x,q2),1e-15,
        "interior generating density must be unchanged");

  // Move only the support edge by one double ULP around this known sample.
  // Its originating channel must remain present in both exact kernels.
  KernelEvent event;
  const double u[5]={.21,.42,.63,.34,.19};
  prop.xmin=std::nextafter(x,0.0);
  const double bh_inside=bh_kernel(250.,false,+1,x,q2,1.,false,prop,u,event);
  const double ll_inside=ll_kernel(250.,false,x,q2,250.,true,x,q2,false,prop,u,event);
  prop.xmin=std::nextafter(x,1.0);
  const double bh_outside=bh_kernel(250.,false,+1,x,q2,1.,false,prop,u,event);
  const double ll_outside=ll_kernel(250.,false,x,q2,250.,true,x,q2,false,prop,u,event);
  require(bh_inside>0 && ll_inside>0,"kernel boundary fixture must be physical");
  std::cout << "boundary BH " << bh_inside << " -> " << bh_outside << "; LL " << ll_inside << " -> " << ll_outside << std::endl;
  close(bh_inside,bh_outside,1e-12,"BH generating density vanished at boundary");
  close(ll_inside,ll_outside,1e-12,"LL generating density vanished at boundary");

  require(checked_repetitions(0.,"test",1)==1,"zero estimate needs one evaluation");
  require(checked_repetitions(2.5,"test",1)==3,"repetition count changed");
  for (double value : {-1.,std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(),1e25}) {
    requires_failure([&] { checked_repetitions(value,"test",1); },
                     "invalid repetitions did not fail");
    requires_failure([&] { check_chunk_weight(value,"test",1); },
                     "invalid chunks did not fail");
  }
  requires_failure([&] { checked_repetitions(kMaximumPairWork,"test",1); },
                   "repetition bound must precede int conversion");
  check_chunk_weight(kMaximumPairWork,"test",1);

  double c,s,cp,sp;
  unsigned i=0;
  auto reject_unpolarised=[&]() { return (++i%2) ? .5 : .99; };
  requires_failure([&] { bw_sample_angle(.5,reject_unpolarised,c,s,3); },
                   "unpolarised rejection loop did not stop");
  i=0;
  const double draws[4]={.5,0.,0.,.999};
  auto reject_polarised=[&]() { return draws[i++%4]; };
  const float linear[3]={0,0,1};
  requires_failure([&] { bw_sample_angle_polarised(.999,linear,linear,
                         reject_polarised,c,s,cp,sp,3); },
                   "polarised rejection loop did not stop");
  auto accept=[]() { return .5; };
  const float unpolarised[3]={0,0,0};
  bw_sample_angle_polarised(.5,unpolarised,unpolarised,accept,c,s,cp,sp,3);
  close(c*c+s*s,1.,1e-14,"accepted polar angle is not normalized");
  close(cp*cp+sp*sp,1.,1e-14,"accepted azimuth is not normalized");
  std::cout << "PASS: QED support, mass, workload and rejection boundaries\n";
  return 0;
}
int main() {
  try { return run_tests(); }
  catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
