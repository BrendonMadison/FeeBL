// Exercise the production geometry partition and actual native SpectrumW
// accumulation. No beam generation, FFT planning, RNG or simulation is run.
#include "guiSpectrumBinningCPP.h"
#include "pairMomentsCPP.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {
unsigned long checks = 0;
void require(bool value,const char* reason) {
  ++checks;
  if(!value) { std::cerr << "GUI spectrum binning: " << reason << '\n';std::exit(1); }
}
template<class F> void rejects(F action) {
  bool rejected=false;try {action();} catch(const std::exception&) {rejected=true;}
  require(rejected,"invalid geometry accepted");
}
void coverage(const std::vector<feebl::gui_spectrum::BinRange>& ranges,std::size_t n) {
  require(!ranges.empty()&&ranges.size()<=2048,"output bin bound");
  std::size_t next=0;
  for(const auto& r:ranges) {
    require(r.first==next&&r.end>r.first&&r.end<=n,"partition gap, overlap or empty group");
    next=r.end;
  }
  require(next==n,"last native bin omitted");
}
std::vector<double> grouped(const pairmoments::SpectrumW& h,
    const pairmoments::SpectrumW* previous,
    const std::vector<feebl::gui_spectrum::BinRange>& ranges) {
  std::vector<double> result;
  for(const auto& r:ranges) {
    double weight=0;
    for(std::size_t i=r.first;i<r.end;++i)
      weight+=h.bin_weight(i)-(previous?previous->bin_weight(i):0.0);
    result.push_back(weight);
  }
  return result;
}
}
int main() {
  using feebl::gui_spectrum::partition;
  // Every smaller spectrum is identical, even with an irrelevant peak window.
  for(std::size_t n:{1U,2U,17U,1024U,2048U}) {
    const auto p=partition(n,0,255,1000,1010);coverage(p,n);
    require(p.size()==n,"small histogram changed resolution");
    for(std::size_t i=0;i<n;++i)
      require(p[i].first==i&&p[i].end==i+1,"identity partition changed native index");
  }
  const auto ilc=partition(10200,0,255,240,255);coverage(ilc,10200);
  require(ilc.size()==1560,"ILC expected 960 coarse plus 600 fine bins");
  for(std::size_t i=0;i<960;++i)
    require(ilc[i].first==10*i&&ilc[i].end==10*(i+1),"ILC tail is not 250 MeV");
  for(std::size_t i=960;i<ilc.size();++i)
    require(ilc[i].first==9600+i-960&&ilc[i].end==ilc[i].first+1,"ILC peak is not 25 MeV");
  // Windows absent, partially clipped, full, zero-width, narrow and straddling
  // native edges. Nondivisible n exercises both independently rounded tails.
  const double cuts[][2]={{-20,-1},{300,400},{0,255},{-20,400},{-20,10},
    {250,400},{127,127},{127.013,127.014},{100.001,150.003},{240,255}};
  for(std::size_t n:{2049U,2051U,4097U,10001U,10200U,65537U,1000003U}) {
    for(const auto& cut:cuts) {
      const auto p=partition(n,0,255,cut[0],cut[1]);coverage(p,n);
      const auto q=partition(n,0,255,cut[0],cut[1]);
      require(p.size()==q.size(),"partition changed across frames");
      for(std::size_t i=0;i<p.size();++i)
        require(p[i].first==q[i].first&&p[i].end==q[i].end,"nondeterministic partition");
    }
  }
  const auto full=partition(10200,0,255,0,255);
  require(full.size()==2040,"oversized all-peak histogram not grouped to budget");
  for(const auto& r:full)require(r.end-r.first==5,"all-peak group size");
  const auto absent=partition(10200,0,255,300,400);
  require(absent.size()==1020,"outside peak did not use bounded tail grouping");
  // A cut through a native bin retains that entire bin; no manufactured
  // fraction or split index is introduced, including around both cut edges.
  const auto narrow=partition(10200,0,255,240.001,240.026);
  bool first=false,second=false;
  for(const auto& r:narrow) {
    first=first||(r.first==9600&&r.end==9601);
    second=second||(r.first==9601&&r.end==9602);
  }
  require(first&&second,"cut-straddling peak bins were not preserved whole");

  // Actual SpectrumW with exact integer weights: grouping conserves every
  // native bin, luminosity interval and flow. Geometry is shared by frames.
  pairmoments::SpectrumW fine,coarse;fine.configure(0,255,10200);coarse.configure(0,255,200);
  std::vector<double> intervals(ilc.size(),0.0);
  double sum_under=0,sum_over=0,sum_records=0;
  for(unsigned int frame=0;frame<5;++frame) {
    const pairmoments::SpectrumW previous=fine;
    for(std::size_t i=0;i<10200;++i) {
      const double w=double((i%17+1)*(frame+1));
      const double W=(double(i)+0.5)*0.025;
      fine.add(w,W);coarse.add(w,W);sum_records+=w;
    }
    fine.add(frame+1,-1);coarse.add(frame+1,-1);sum_records+=frame+1;
    fine.add(frame+2,256);coarse.add(frame+2,256);sum_records+=frame+2;
    const auto interval=grouped(fine,&previous,ilc);
    for(std::size_t j=0;j<interval.size();++j) {
      require(interval[j]>=0,"negative interval bin");intervals[j]+=interval[j];
    }
    sum_under+=fine.underflow()-previous.underflow();
    sum_over+=fine.overflow()-previous.overflow();
  }
  const auto total=grouped(fine,nullptr,ilc);
  double all=0;
  for(std::size_t i=0;i<total.size();++i) {
    require(total[i]==intervals[i],"frame-to-total grouped-bin closure");all+=total[i];
  }
  require(sum_under==fine.underflow()&&sum_over==fine.overflow(),"flow interval closure");
  require(all+sum_under+sum_over==sum_records,"all native weights not conserved");
  require(fine.w_moment().mean()==coarse.w_moment().mean()&&
          fine.w_moment().variance()==coarse.w_moment().variance()&&
          fine.w_moment().weight()==coarse.w_moment().weight(),
          "unbinned native moments depend on configured bin resolution");
  rejects([&]{partition(0,0,255,240,255);});
  rejects([&]{partition(10200,255,0,240,255);});
  rejects([&]{partition(10200,0,255,255,240);});
  rejects([&]{partition(10200,0,std::numeric_limits<double>::infinity(),240,255);});
  rejects([&]{partition(10200,0,255,std::numeric_limits<double>::quiet_NaN(),255);});
  std::cout << "PASS GUI native spectrum partition: " << checks << " checks\n";
}
