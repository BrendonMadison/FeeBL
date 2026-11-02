#include "colliderPeakDiagnosticsCPP.h"
#include "resultsCPP.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace {
unsigned checks=0;
void check(bool condition,const char* message){++checks;if(!condition){std::cerr<<message<<'\n';std::exit(1);}}
void near(double a,double b,double tolerance,const char* message){check(std::abs(a-b)<=tolerance,message);}
template<class F>void rejects(F call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}check(caught,"invalid data must be rejected");}
std::string json(const feebl::peak_diagnostics::Accumulator& a,bool coverage=true){std::ostringstream out;out<<std::setprecision(17);a.write(out,coverage);return out.str();}
void configure(RESULTS& result,const feebl::peak_diagnostics::Definition& definition){result.enable_gui_diagnostics(true);result.configure_gui_peak(definition);result.configure_pair_spectrum(0,300,600);}
}
int main(int argc,char** argv){
  using namespace feebl::peak_diagnostics;
  const Definition head=Definition::make(125,125,0,0,0,0,0,255);
  check(head.available,"head-on reference available");near(head.nominal,250,1e-12,"head-on design invariant mass");
  const Definition d=Definition::make(125,125,double(.007f),double(.007f),0,0,.014,255);
  check(d.available,"matching double design angle available");
  near(d.nominal,249.99387502501045,3e-14,"reference agrees with published Python nominal formula");
  check(d.full_angle==.014 && d.minimum==.98*d.nominal,"use exact declared double angle and inclusive threshold");
  check(!Definition::make(125,125,.007,-.007,0,0,.014,255).available,"opposite signs unavailable");
  check(!Definition::make(125,125,.007,.007,.001,.001,.016,255).available,"mixed crossing planes unavailable");
  check(!Definition::make(125,125,.007,.007,0,0,.015,255).available,"mismatched angle unavailable");
  check(!Definition::make(125,125,.007,.007,0,0,0,255).available,"unrecorded crossing reference unavailable");
  check(!Definition::make(0,125,0,0,0,0,0,255).available,"unphysical energy unavailable");
  check(!Definition::make(125,125,0,0,0,0,std::numeric_limits<double>::quiet_NaN(),255).available,"NaN reference unavailable");
  const Definition asymmetric=Definition::make(50,200,0,0,0,0,0,210);
  check(asymmetric.available && asymmetric.nominal<200,"massive asymmetric reference retains electron mass");
  Accumulator a(d);
  a.add(8,std::nextafter(d.minimum,0.0),99,100,true,.1,.2);
  a.add(0,d.minimum,80,90,true,.1,.2);
  check(a.contributions==0 && a.weight==0,"below-threshold and zero-weight evaluations excluded");
  a.add(2,d.minimum,-120,125,true,.8,-.3);
  a.add(3,250,124,126,true,.6,-.1);
  a.add(5,260,130,131,true,.4,.1);
  check(a.contributions==3 && a.spin_contributions==3 && a.weight==10 && a.spin_weight==10,"one common selected population");
  check(a.bins[0]==2 && a.overflow==5,"inclusive cutoff and upper flow are exact");
  near(std::accumulate(a.bins.begin(),a.bins.end(),0.0)+a.overflow,a.weight,0,"selected spectrum closes");
  near(a.energy1.mean(a.weight,d.energy1),126.2,1e-13,"absolute weighted energy includes overflow");
  near(a.energy2.mean(a.weight,d.energy2),128.3,1e-13,"beam2 shares identical selected weights");
  near(a.h1_sum/a.weight,.54,1e-15,"selected beam1 helicity weighted mean");
  near(a.h2_sum/a.weight,-.04,1e-15,"selected beam2 helicity weighted mean");
  near(a.W.mean(a.weight,d.nominal),(2*d.minimum+3*250+5*260)/10,1e-13,"exact unbinned W mean");
  check(json(a).find("\"polarization\":null")==std::string::npos,"complete spin coverage available");
  Accumulator partial=a;partial.add(1,250,125,125);
  check(json(partial).find("\"polarization\":null")!=std::string::npos,"partial spin coverage never dilutes polarization");
  check(partial.weight==11 && partial.spin_weight==10,"partial spin receipts exposed");
  Accumulator zero(d);const auto empty=json(zero);
  check(empty.find("\"available\":true")!=std::string::npos && empty.find("\"beam_energy_GeV\":null")!=std::string::npos,"computed-empty distinct from unavailable");
  check(json(a,false).find("\"luminosity_spectrum\":null")!=std::string::npos,"unsupported mode omits plausible histograms");
  check(json(Accumulator()).find("\"selection\":null")!=std::string::npos,"ambiguous reference not guessed");
  rejects([&]{zero.add(-1,250,125,125);});
  rejects([&]{zero.add(1,std::numeric_limits<double>::infinity(),125,125);});
  rejects([&]{zero.add(1,250,125,125,true,std::numeric_limits<double>::quiet_NaN(),0);});

  // Independent long-double oracle over correlated energies and helicities,
  // with exact cut, deep tail, overflow, negative stored energies and zero w.
  Accumulator all(d),left(d),right(d);
  long double sw=0,se1=0,se2=0,sm=0,sh1=0,sh2=0;
  std::uint64_t count=0;
  for(int i=0;i<2000;++i){
    const double e1=125+.7*std::sin(i*.013),e2=125+.9*std::cos(i*.017);
    const double mass=(i%29==0?240:(i%31==0?258:pairmoments::invariant_mass(e1,0,0,1,e2,0,0,-1,EMASS)));
    const double weight=(i%17==0?0:1.0+(i%13)*.125),h1=.8-.01*(e1-125),h2=-.3+.02*(e2-125);
    all.add(weight,mass,-e1,e2,true,h1,h2);
    (i<1000?left:right).add(weight,mass,-e1,e2,true,h1,h2);
    if(weight>0 && mass>=d.minimum){++count;sw+=weight;se1+=weight*(long double)e1;se2+=weight*(long double)e2;sm+=weight*(long double)mass;sh1+=weight*(long double)h1;sh2+=weight*(long double)h2;}
  }
  check(all.contributions==count && all.spin_contributions==count,"oracle exact population mask");
  near(all.weight,double(sw),0,"oracle exact common weight");
  near(all.energy1.mean(all.weight,d.energy1),double(se1/sw),3e-13,"oracle E1 mean");
  near(all.energy2.mean(all.weight,d.energy2),double(se2/sw),3e-13,"oracle E2 mean");
  near(all.W.mean(all.weight,d.nominal),double(sm/sw),3e-13,"oracle W mean");
  near(all.h1_sum/all.weight,double(sh1/sw),3e-15,"oracle P1 mean");
  near(all.h2_sum/all.weight,double(sh2/sw),3e-15,"oracle P2 mean");
  check(left.merge(right),"matching reduction succeeds");
  check(left.contributions==all.contributions && left.weight==all.weight,"reduction population closure");
  near(left.energy1.mean(left.weight,d.energy1),all.energy1.mean(all.weight,d.energy1),3e-13,"reduction E1 closure");
  for(std::size_t i=0;i<histogram_bins;++i)near(left.bins[i],all.bins[i],0,"reduction histogram bin closure");
  Accumulator narrow(d);narrow.add(1,250,125-1e-6,125+1e-6);narrow.add(1,250,125+1e-6,125-1e-6);
  near(narrow.energy1.rms(2),double(125+1e-6)-125,1e-15,"shifted moments resolve narrow RMS without cancellation");

  RESULTS actual;configure(actual,d);
  actual.add_lumi_ee(1e30);actual.add_step_lumi_ee(1e30);actual.add_pair_W(1e30,250,.8,-.3);actual.add_lumi_weighted_helicity(1e30,.8,-.3);
  const RESULTS original=actual;
  actual.add_gui_peak(1e30,250,125,125,true,.8,-.3);
  check(actual.get_step_lumi_ee()==original.get_step_lumi_ee() && actual.get_lumi_weight_sum()==original.get_lumi_weight_sum() && actual.get_lumi_helicity_sum1()==original.get_lumi_helicity_sum1() && actual.get_lumi_helicity_sum2()==original.get_lumi_helicity_sum2(),"optional observables do not mutate luminosity/helicity state");
  for(std::size_t i=0;i<actual.pair_spectrum().bins();++i)
    check(actual.pair_spectrum().bin_weight(i)==original.pair_spectrum().bin_weight(i),"optional observables do not mutate native spectrum");
  actual.reset_gui_interval();actual.add_gui_peak(1e-30,250,124,126,true,.4,.2);
  check(actual.gui_interval_peak().weight==1e-30 && actual.gui_interval_peak().contributions==1,"tiny late interval is not cumulative subtraction");
  near(actual.gui_interval_peak().energy1.mean(1e-30,d.energy1),124,0,"tiny interval energy remains defined");
  check(actual.gui_peak().contributions==2,"interval reset preserves total population");
  RESULTS lane;configure(lane,d);lane.add_gui_peak(3,251,123,128,true,.7,.1);
  check(actual.merge_primary_photon_collision_from(lane),"RESULTS reduction carries conditional moments");
  check(actual.gui_interval_peak().contributions==2 && actual.gui_peak().contributions==3,"RESULTS total and interval reductions");
  RESULTS bad;configure(bad,head);bad.add_gui_peak(2,250,125,125);
  const std::string before=json(actual.gui_peak());
  check(!actual.merge_primary_photon_collision_from(bad),"different nominal selection cannot merge");
  check(before==json(actual.gui_peak()),"failed RESULTS reduction is transactional");
  actual.reset_primary_photon_collision_lane();
  check(actual.gui_peak().contributions==0 && actual.gui_peak().definition.same(d),"lane reset preserves selection only");
  actual.enable_gui_diagnostics(false);actual.add_gui_peak(2,250,125,125);
  check(actual.gui_peak().contributions==0,"disabled diagnostics do not accumulate");
  if(argc==2){std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"populated\":";a.write(out,true);out<<",\"empty\":";zero.write(out,true);out<<",\"partial_spin\":";partial.write(out,true);out<<",\"unavailable\":";Accumulator().write(out,false);out<<"}\n";check(bool(out),"fixture written");}
  std::cout << "PASS " << checks << " native peak diagnostic checks\n";
}
