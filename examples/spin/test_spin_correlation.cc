#include "resultsCPP.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(double value,double expected){check(std::abs(value-expected)<2e-14,"spin correlation mismatch");}
void configure(RESULTS& a){a.enable_gui_diagnostics(true);a.configure_pair_spectrum(0,510,2);}
int main(){
  RESULTS all,left,right;configure(all);configure(left);configure(right);
  all.add_lumi_weighted_helicity(1,.8,.9,.75,.85);
  all.add_lumi_weighted_helicity(3,.2,-.1,.25,-.2);
  left.add_lumi_weighted_helicity(1,.8,.9,.75,.85);
  right.add_lumi_weighted_helicity(3,.2,-.1,.25,-.2);
  near(all.helicity_correlation().mean_product(),.165);
  near(all.axis_correlation().mean_product(),.121875);
  near(all.helicity_correlation().h1().mean(),.35);
  near(all.helicity_correlation().h2().mean(),.15);
  check(std::abs(all.helicity_correlation().mean_product()-.35*.15)>.1,
        "must retain correlation, not multiply beam means");
  check(left.merge_primary_photon_collision_from(right),"merge failed");
  near(left.helicity_correlation().mean_product(),all.helicity_correlation().mean_product());
  near(left.axis_correlation().mean_product(),all.axis_correlation().mean_product());
  near(left.helicity_correlation(true).weight(),4);
  near(left.axis_correlation(true).weight(),4);
  left.reset_gui_interval();
  check(left.helicity_correlation(true).empty() && left.axis_correlation(true).empty(),"interval not reset");
  near(left.helicity_correlation().mean_product(),.165);
  left.reset_primary_photon_collision_lane();
  check(left.helicity_correlation().empty() && left.axis_correlation().empty(),"lane not reset");
  RESULTS unknown;configure(unknown);unknown.add_lumi_weighted_helicity(1,.8,.9);
  near(unknown.helicity_correlation().mean_product(),.72);
  check(unknown.axis_correlation().empty(),"missing axis projection must remain missing");
  RESULTS invalid;configure(invalid);
  invalid.add_lumi_weighted_helicity(1,1,1,std::numeric_limits<double>::quiet_NaN(),1);
  const double before=all.helicity_correlation().weight();
  check(!all.merge_primary_photon_collision_from(invalid),"nonfinite correlation merged");
  near(all.helicity_correlation().weight(),before);
  // Equal fully longitudinal beams: the published correlation definition
  // returns their common polarization, without a factor of two.
  RESULTS equal;equal.add_lumi_weighted_helicity(7,.9976,.9976,.9976,.9976);
  near(100*(1-std::sqrt(equal.axis_correlation().mean_product())),.24);
  std::cout<<"PASS weighted correlation, axis distinction, merge, reset, missing data, transactional rejection and symmetric limit\n";
}
