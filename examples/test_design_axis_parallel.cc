#include "designAxisSpectrumCPP.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using feebl::design_axis::Spectrum;

static void require(bool condition, const char* message) {
  if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

static bool close(double a, double b, double relative=2e-13) {
  return std::fabs(a-b) <= relative*std::max(1.,std::max(std::fabs(a),std::fabs(b)));
}

static std::string snapshot(const Spectrum& spectrum) {
  std::ostringstream stream;
  spectrum.write(stream,"unit-test",19);
  return stream.str();
}

static void equivalent(const Spectrum& a, const Spectrum& b) {
  require(a.bins().size()==b.bins().size(),"matching bin count");
  require(a.contributions()==b.contributions(),"all collision pairs retained");
  require(close(a.total_weight(),b.total_weight()),"total luminosity preserved");
  require(close(a.total_weight2(),b.total_weight2()),"squared weights preserved");
  require(close(a.mean(),b.mean()),"design-axis mean preserved");
  require(std::fabs(a.rms()-b.rms())<2e-11,"design-axis RMS preserved");
  for (std::size_t i=0;i<a.bins().size();++i) {
    const auto& x=a.bins()[i]; const auto& y=b.bins()[i];
    require(x.low==y.low && x.high==y.high,"bin boundaries preserved exactly");
    require(x.contributions==y.contributions,"per-bin contribution count preserved");
    require(close(x.weight,y.weight) && close(x.weight2,y.weight2),
            "per-bin luminosity and squared weight preserved");
  }
}

// Each lane represents a disjoint slice encounter. Varying energies and
// directions exercise the same per-pair reconstruction as the production
// block publisher; the merged observable must not depend on worker grouping.
static void merged_encounters(double energy1, double energy2, double angle,
                              int fine_bins) {
  Spectrum serial, merged, lane1, lane2, lane3;
  serial.configure(energy1,energy2,angle,angle,fine_bins);
  merged.configure_like(serial);
  lane1.configure_like(serial); lane2.configure_like(serial); lane3.configure_like(serial);
  Spectrum* lanes[]={&lane1,&lane2,&lane3};
  const auto initial_storage=lane1.storage_bytes();
  for (int encounter=0;encounter<5;++encounter) {
    for (auto lane:lanes) lane->reset();
    for (int index=0;index<357;++index) {
      const double scale=1.+((index%13)-6)*1e-6;
      const double e1=energy1*scale, e2=energy2/scale;
      const double weight=(index%17+1)*.73;
      const double x1=((index%7)-3)*1e-5, y1=((index%5)-2)*2e-5;
      const double x2=((index%11)-5)*1e-5, y2=((index%3)-1)*2e-5;
      serial.add(weight,e1,x1,y1,e2,x2,y2);
      lanes[index%3]->add(weight,e1,x1,y1,e2,x2,y2);
    }
    // Include both histogram tails and overflow in the merge contract.
    for (double multiplier:{.3,1.07,1.6}) {
      serial.add(3,energy1*multiplier,0,0,energy2*multiplier,0,0);
      lane3.add(3,energy1*multiplier,0,0,energy2*multiplier,0,0);
    }
    require(merged.merge_from_many({&lane1,&lane2,&lane3}),"valid lane merge");
    equivalent(serial,merged);
    require(lane1.storage_bytes()==initial_storage,"lane reset reuses allocation");
  }
  lane1.reset();
  require(lane1.contributions()==0 && lane1.total_weight()==0 &&
          lane1.total_weight2()==0 && lane1.mean()==0 && lane1.rms()==0,
          "encounter reset clears all moments");
  for (const auto& bin:lane1.bins())
    require(bin.weight==0 && bin.weight2==0 && bin.contributions==0,
            "encounter reset clears every populated bin");
  std::cout << "PASS lane reduction: " << energy1 << '+' << energy2
            << " GeV, angle=" << angle << ", fine bins=" << fine_bins << '\n';
}

static void transactional_rejection() {
  Spectrum destination, valid, mismatch, disabled;
  destination.configure(62.5,62.5,.007,.007,100);
  valid.configure_like(destination);
  mismatch.configure(62.5,62.5,0,0,100);
  destination.add(2,62.5,0,0,62.5,0,0);
  valid.add(3,62.5,0,0,62.5,0,0);
  const auto before=snapshot(destination);
  require(!destination.merge_from_many({&valid,&mismatch}),"reject mismatched axes");
  require(snapshot(destination)==before,"late incompatible lane cannot partly publish");
  require(!destination.merge_from_many({&valid,&disabled}),"reject disabled lane");
  require(!destination.merge_from_many({&valid,nullptr}),"reject null lane");
  require(!destination.merge_from_many({&destination}),"reject self merge");
  require(snapshot(destination)==before,"all rejected merges preserve destination");

  Spectrum huge, overflow;
  huge.configure(62.5,62.5,0,0,100); overflow.configure_like(huge);
  huge.add(1e154,62.5,0,0,62.5,0,0);
  overflow.add(1e154,62.5,0,0,62.5,0,0);
  const auto overflow_before=snapshot(overflow);
  require(!overflow.merge_from_many({&huge}),"reject squared-weight overflow");
  require(snapshot(overflow)==overflow_before,"overflow cannot partly publish");
  Spectrum disabled2;
  require(disabled.merge_from_many({&disabled2}),"disabled diagnostic remains disabled");
  std::cout << "PASS transaction: configuration/null/self/overflow rejection without mutation\n";
}

int main() {
  merged_encounters(62.5015312813,62.5015312813,.007,800000);
  merged_encounters(187.5,125.*125./750.,0,10000);
  transactional_rejection();
}
