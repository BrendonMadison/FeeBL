#include "designAxisSpectrumCPP.h"
#include "pairMomentsCPP.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
using namespace feebl::design_axis;
static void check(bool yes,const char* name){if(!yes){std::cerr<<name<<'\n';std::exit(1);}}
static bool close(double a,double b,double r=2e-13){return std::fabs(a-b)<=r*std::max(1.0,std::max(std::fabs(a),std::fabs(b)));}
// Independently construct lab four-vectors and square their sum in long double.
static long double oracle(double E1,Direction n1,double E2,Direction n2,double a1,double a2){
  const long double m=electron_mass_GeV;
  const long double p1=sqrtl((E1-m)*(E1+m)),p2=sqrtl((E2-m)*(E2+m));
  const long double x=p1*(cosl(a1)*n1.x+sinl(a1)*n1.z)+p2*(cosl(a2)*n2.x-sinl(a2)*n2.z);
  const long double y=p1*n1.y+p2*n2.y;
  const long double z=p1*(-sinl(a1)*n1.x+cosl(a1)*n1.z)+p2*(sinl(a2)*n2.x+cosl(a2)*n2.z);
  return sqrtl((E1+E2)*(E1+E2)-x*x-y*y-z*z);
}
// Histogram resolution must change binning only, never event weights or moments.
static void test_high_resolution() {
  Spectrum coarse, fine;
  coarse.configure(62.5,62.5,0,0,50000);
  fine.configure(62.5,62.5,0,0,1000000);
  check(fine.bins().size()==1002560,"one-million-bin capacity with full tails");
  check(std::fabs((fine.bins()[2048].high-fine.bins()[2048].low)-1e-5)<3e-14,
        "10 keV native peak bins at 125 GeV");
  for(int i=0;i<200;++i) {
    // Stay away from shared boundaries; equal energies make W exactly 2E.
    const double energy=62.5+(i-100)*0.000013+0.0000015;
    const double weight=1+(i%7);
    coarse.add(weight,energy,0,0,energy,0,0);
    fine.add(weight,energy,0,0,energy,0,0);
  }
  for(double energy:{10.,59.,66.,68.}) {
    coarse.add(3,energy,0,0,energy,0,0);
    fine.add(3,energy,0,0,energy,0,0);
  }
  check(coarse.total_weight()==fine.total_weight() &&
        coarse.total_weight2()==fine.total_weight2() &&
        coarse.contributions()==fine.contributions() &&
        coarse.mean()==fine.mean() && coarse.rms()==fine.rms(),
        "histogram-only change preserves eventwise moments exactly");
  for(std::size_t index=0;index<coarse.bins().size();++index) {
    const auto& expected=coarse.bins()[index];
    const std::size_t first=index<2048 ? index :
        (index<52048 ? 2048+(index-2048)*20 : 1002048+(index-52048));
    const std::size_t size=(index>=2048 && index<52048) ? 20 : 1;
    double weight=0,weight2=0; std::uint64_t count=0;
    for(std::size_t child=first;child<first+size;++child) {
      weight+=fine.bins()[child].weight;
      weight2+=fine.bins()[child].weight2;
      count+=fine.bins()[child].contributions;
    }
    check(weight==expected.weight && weight2==expected.weight2 && count==expected.contributions,
          "fine-bin grouping reproduces old bins including both tails");
  }
  for(std::size_t index=1;index<fine.bins().size();++index)
    check(fine.bins()[index].low==fine.bins()[index-1].high,
          "one-million-bin edges remain contiguous");

  Spectrum separated;
  separated.configure(62.5,62.5,0,0,800000);
  separated.add(1,62.500003,0,0,62.500003,0,0);
  separated.add(1,62.500015,0,0,62.500015,0,0);
  std::size_t occupied=0;
  for(const auto& bin:separated.bins()) occupied+=(bin.contributions!=0);
  check(occupied==2,"12.5 keV bins separate peaks 24 keV apart");
  for(int invalid:{99,1000001}) {
    bool rejected=false;
    try {Spectrum other;other.configure(62.5,62.5,0,0,invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"histogram allocation bounds enforced");
  }
  std::cout<<"PASS fine histogram: 10/12.5 keV resolution, tails, counts, sumw2, exact moments, grouped closure and bounds\n";
}

int main(){
  test_high_resolution();
  const double m=electron_mass_GeV;
  auto n1=direction(.002,-.001,1),n2=direction(-.001,.003,-1);
  check(close(mass(62.5,n1,62.5,n2,0,0),pairmoments::invariant_mass(62.5,n1.x,n1.y,n1.z,62.5,n2.x,n2.y,n2.z,m)),"zero-angle identity");
  for(double a:{0.,.007,.015}){
    const double expected=std::sqrt(4*m*m+4*(62.5*62.5-m*m)*std::cos(a)*std::cos(a));
    check(close(mass(62.5,{0,0,1},62.5,{0,0,-1},a,a),expected),"massive symmetric nominal cos");
    check(close(mass(187.5,n1,125.*125./750.,n2,a,a),double(oracle(187.5,n1,125.*125./750.,n2,a,a))),"asymmetric perturbed four-vector invariant");
    auto r=rotate_y(n1,a);check(close(r.x*r.x+r.y*r.y+r.z*r.z,1),"rotation norm");
    auto r1=rotate_y({0,0,1},a),r2=rotate_y({0,0,-1},-a);
    check(r1.x>=0&&r2.x>=0&&close(r1.x,r2.x),"both beams positive crossing px");
  }
  check(close(mass(2*m,n1,3*m,n2,.015,.015),double(oracle(2*m,n1,3*m,n2,.015,.015)),1e-15),"non-negligible rest mass");
  Spectrum s;s.configure(62.5,62.5,.007,.007,10000);
  double weighted=0,weighted2=0;unsigned count=0;
  for(int i=0;i<21;++i){double E=62.5+(i-10)*.0001,w=i+1;s.add(w,E,0,0,E,0,0);weighted+=w;weighted2+=w*w;++count;}
  double sum=0,sum2=0;std::uint64_t nc=0;
  for(std::size_t i=0;i<s.bins().size();++i){const auto& b=s.bins()[i];sum+=b.weight;sum2+=b.weight2;nc+=b.contributions;if(i)check(b.low==s.bins()[i-1].high,"contiguous edge geometry");}
  check(close(sum,weighted)&&close(sum2,weighted2)&&nc==count&&s.contributions()==count,"all-bin moment/weight closure");
  double expected_mean=0,expected_v=0;
  for(int i=0;i<21;++i){double E=62.5+(i-10)*.0001,w=i+1;expected_mean+=w*double(oracle(E,{0,0,1},E,{0,0,-1},.007,.007))/weighted;}
  for(int i=0;i<21;++i){double E=62.5+(i-10)*.0001,w=i+1;double d=double(oracle(E,{0,0,1},E,{0,0,-1},.007,.007))-expected_mean;expected_v+=w*d*d/weighted;}
  check(close(s.mean(),expected_mean)&&std::fabs(s.rms()-std::sqrt(expected_v))<1e-11,"centered sub-MeV width");
  std::ostringstream out;s.write(out,"test",7);check(out.str().find("#all_bins_written yes")!=std::string::npos&&out.str().find("#END design_axis_pair_spectrum records 12560")!=std::string::npos,"complete file contract");
  bool rejected=false;try{direction(2,0,1);}catch(...){rejected=true;}check(rejected,"invalid direction rejected");
  std::cout<<"PASS design-axis zero-angle, nominal cos, asymmetric/perturbed/massive 4vectors, rotation norm/sign, bin and moment closure, sub-MeV width, complete zeros, invalid input\n";
  std::cout<<std::setprecision(17)<<"ILC125_design_W="<<mass(62.5,{0,0,1},62.5,{0,0,-1},.007,.007)<<" FCC125_design_W="<<mass(62.5,{0,0,1},62.5,{0,0,-1},.015,.015)<<'\n';
}
