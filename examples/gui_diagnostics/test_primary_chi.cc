// Real native RESULTS add/reset/merge paths; no particle motion or RNG.
#include "resultsCPP.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
namespace {
unsigned long checks=0;
void check(bool value,const char* message) {
  ++checks;if(!value){std::cerr << "primary chi diagnostic: " << message << '\n';std::exit(1);}
}
template<class F> void rejects(F action) {
  bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
  check(rejected,"invalid identity/input was not rejected");
}
void same(const feebl::diagnostics::Histogram& a,const feebl::diagnostics::Histogram& b) {
  check(a.log_low==b.log_low&&a.log_high==b.log_high,"histogram geometry changed");
  check(a.weights==b.weights,"histogram bin weights changed");
  check(a.underflow==b.underflow&&a.overflow==b.overflow&&a.samples==b.samples,
        "histogram flows or sample count changed");
}
void closure(const RESULTS& r,bool interval) {
  const auto& c=interval?r.gui_interval_chi_e():r.gui_chi_e();
  const auto& a=interval?r.gui_interval_chi_e_beam(1):r.gui_chi_e_beam(1);
  const auto& b=interval?r.gui_interval_chi_e_beam(2):r.gui_chi_e_beam(2);
  check(c.samples==a.samples+b.samples,"species sample closure");
  check(c.underflow==a.underflow+b.underflow,"species underflow closure");
  check(c.overflow==a.overflow+b.overflow,"species overflow closure");
  for(std::size_t i=0;i<c.bins;++i) {
    check(c.weights[i]==a.weights[i]+b.weights[i],"exact combined-bin closure");
    check(c.low(i)==a.low(i)&&a.low(i)==b.low(i),"split bin edge equality");
  }
}
void seed(RESULTS& r) {
  r.enable_gui_diagnostics(true);
  r.add_gui_chi_e(1,0.001);r.add_gui_chi_e(2,0.1);
  r.add_gui_chi_e(1,0.0);r.add_gui_chi_e(2,2000.0);
}
}
int main() {
  using feebl::diagnostics::canonical_primary_pdg;
  // Species convention is explicit in gpuMode3ProductionCPP.cc's import
  // contract. Relative charge or positive stored energy alone is insufficient.
  check(canonical_primary_pdg(1,-1.0)==11,"canonical electron identity");
  check(canonical_primary_pdg(2,-1.0)==-11,"canonical positron identity");
  for(double charge:{0.0,1.0,-0.9999999,-2.0,
      std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    check(canonical_primary_pdg(1,charge)==0&&canonical_primary_pdg(2,charge)==0,
          "unsupported charge mode assigned a species");
  }
  rejects([]{canonical_primary_pdg(0,-1.0);});
  rejects([]{canonical_primary_pdg(3,-1.0);});

  RESULTS r;seed(r);
  feebl::diagnostics::Histogram legacy(-10,3);
  for(double value:{0.001,0.1,0.0,2000.0})legacy.add(value);
  same(r.gui_chi_e(),legacy);same(r.gui_interval_chi_e(),legacy);
  check(r.gui_chi_e_beam(1).weights!=r.gui_chi_e_beam(2).weights,
        "different beam inputs were cloned into identical histograms");
  check(r.gui_chi_e_beam(1).samples==2&&r.gui_chi_e_beam(2).samples==2,"beam record counts");
  check(r.gui_chi_e_beam(1).underflow==1&&r.gui_chi_e_beam(1).overflow==0,
        "electron flow identity");
  check(r.gui_chi_e_beam(2).underflow==0&&r.gui_chi_e_beam(2).overflow==1,
        "positron flow identity");
  closure(r,false);closure(r,true);
  const RESULTS before=r;
  rejects([&]{r.add_gui_chi_e(0,1.0);});rejects([&]{r.add_gui_chi_e(3,1.0);});
  rejects([&]{r.add_gui_chi_e(1,-1.0);});
  rejects([&]{r.add_gui_chi_e(2,std::numeric_limits<double>::infinity());});
  rejects([&]{r.gui_chi_e_beam(0);});rejects([&]{r.gui_interval_chi_e_beam(3);});
  same(r.gui_chi_e(),before.gui_chi_e());
  same(r.gui_chi_e_beam(1),before.gui_chi_e_beam(1));
  same(r.gui_interval_chi_e_beam(2),before.gui_interval_chi_e_beam(2));

  // Frame interval reset preserves the accumulated total and each identity.
  r.reset_gui_interval();
  check(r.gui_interval_chi_e().samples==0&&r.gui_interval_chi_e_beam(1).samples==0&&
        r.gui_interval_chi_e_beam(2).samples==0,"interval species reset");
  same(r.gui_chi_e(),before.gui_chi_e());closure(r,true);
  r.add_gui_chi_e(2,5.0);r.add_gui_chi_e(2,6.0);r.add_gui_chi_e(1,0.02);
  check(r.gui_interval_chi_e_beam(1).samples==1&&r.gui_interval_chi_e_beam(2).samples==2,
        "new interval carries wrong beam counts");
  closure(r,false);closure(r,true);
  auto reconstructed=before.gui_chi_e();reconstructed.merge(r.gui_interval_chi_e());
  same(reconstructed,r.gui_chi_e());
  for(int beam=1;beam<=2;++beam) {
    auto joined=before.gui_chi_e_beam(beam);joined.merge(r.gui_interval_chi_e_beam(beam));
    same(joined,r.gui_chi_e_beam(beam));
  }

  // Production transactional worker reduction and lane reset preserve splits.
  RESULTS lane1,lane2,destination;seed(lane1);seed(lane2);
  lane1.add_gui_chi_e(1,0.4);lane2.add_gui_chi_e(2,0.8);
  destination.enable_gui_diagnostics(true);
  const std::vector<const RESULTS*> lanes{&lane1,&lane2};
  check(destination.merge_primary_photon_collision_from_many(lanes),"native RESULTS multi-lane merge failed");
  check(destination.gui_chi_e_beam(1).samples==5&&destination.gui_chi_e_beam(2).samples==5,
        "merged species sample counts");
  for(int beam=1;beam<=2;++beam) {
    auto expected=lane1.gui_chi_e_beam(beam);expected.merge(lane2.gui_chi_e_beam(beam));
    same(destination.gui_chi_e_beam(beam),expected);
  }
  closure(destination,false);closure(destination,true);
  lane1.reset_primary_photon_collision_lane();
  check(lane1.gui_diagnostics_enabled(),"lane reset lost diagnostics enablement");
  check(lane1.gui_chi_e().samples==0&&lane1.gui_chi_e_beam(1).samples==0&&
        lane1.gui_chi_e_beam(2).samples==0,"lane reset retained old species counts");
  closure(lane1,false);closure(lane1,true);
  RESULTS disabled;disabled.add_gui_chi_e(1,0.5);disabled.add_gui_chi_e(2,0.9);
  check(disabled.gui_chi_e().samples==0,"disabled diagnostics modified histograms");
  std::cout << "PASS native primary chi species split: " << checks << " checks\n";
}
