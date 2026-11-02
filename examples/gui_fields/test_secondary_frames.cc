#include "colliderSecondaryFramesCPP.h"
#include "pairsCPP.h"
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace {
unsigned checks=0;
void check(bool value,const char* message) {
  ++checks;if(!value){std::cerr << "secondary snapshot test: " << message << '\n';std::exit(1);}
}
void near(double actual,double expected,double tolerance,const char* message) {
  check(std::abs(actual-expected)<=tolerance,message);
}
template<class F>void rejects(F action) {
  bool caught=false;try{action();}catch(const std::exception&){caught=true;}
  check(caught,"expected invalid snapshot to be rejected");
}
std::size_t sum(const std::vector<std::size_t>& values) {
  return std::accumulate(values.begin(),values.end(),std::size_t(0));
}
void closure(const feebl::secondary_frames::Snapshot& s) {
  for(const auto* j:{&s.joint,&s.joint_folded}) {
    check(j->samples==s.nrecords,"joint sample count");
    check(sum(j->counts)==j->in_range,"joint row-major count sum");
    check(j->in_range+j->nonpositive+j->underflow+j->overflow==s.nrecords,"joint excluded/flow closure");
    check(j->theta_zero+j->pt_zero-j->both_zero==j->nonpositive,"zero union closure");
  }
  for(const auto* m:{&s.theta,&s.theta_folded,&s.pt})
    check(sum(m->counts)+m->underflow+m->overflow==s.nrecords,"marginal count closure");
  std::size_t processes=0;for(const auto& kv:s.processes)processes+=kv.second;
  check(processes==s.nrecords && s.positive+s.negative==s.nrecords,"selection count closure");
}
std::string states(const std::list<PAIR_PARTICLE>& particles) {
  std::ostringstream out;for(const auto& p:particles)out << p.persistent_flow() << '\n';return out.str();
}
}

int main(int argc,char** argv) {
  using namespace feebl::secondary_frames;
  const Axes& a=axes();
  for(const auto* v:{&a.theta,&a.folded,&a.pt,&a.marginal_theta,&a.marginal_folded,&a.marginal_pt})
    for(std::size_t i=1;i<v->size();++i)check((*v)[i]>(*v)[i-1] && feebl::secondary_frames::finite((*v)[i]),"finite increasing fixed edges");
  check(a.theta.size()==129 && a.pt.size()==129 && a.marginal_pt.size()==161,"fixed axis dimensions");
  check(bin(a.theta,pi)==127 && bin(a.folded,pi*0.5)==127,"exact top edge included");
  check(bin(a.pt,1e6)==127 && bin(a.pt,2e6)==128 && bin(a.pt,1e-13)==-1,"logarithmic flow bounds");
  std::vector<PAIR_PARTICLE> known;
  known.emplace_back(1,0,0,0,0,.3f,.4f,std::sqrt(.75f),2);
  known.emplace_back(1,1,0,0,0,.3f,.4f,-std::sqrt(.75f),-2);
  auto s=Snapshot::capture(known,2);closure(s);
  near(double(s.pt.moments.mean),std::hypot(double(.3f),double(.4f))*2,1e-15,"native p/E transverse momentum");
  near(double(s.theta.moments.mean),pi*0.5,1e-15,"forward/backward global angle mean");
  near(double(s.theta_folded.moments.mean),std::atan2(std::hypot(double(.3f),double(.4f)),double(std::sqrt(.75f))),1e-15,"folded angle measured directly");
  check(std::count_if(s.joint.counts.begin(),s.joint.counts.end(),[](std::size_t n){return n>0;})==2,"two global angular bins");
  check(std::count_if(s.joint_folded.counts.begin(),s.joint_folded.counts.end(),[](std::size_t n){return n>0;})==1,"folded branches share one bin");
  const int ix=bin(a.theta,std::atan2(std::hypot(double(.3f),double(.4f)),double(std::sqrt(.75f))));
  const int iy=bin(a.pt,std::hypot(double(.3f),double(.4f))*2);
  check(s.joint.counts[std::size_t(iy)*128+ix]==1,"pT-major theta-minor storage order");
  known.emplace_back(3,2,0,0,0,0,0,1,2);
  known.emplace_back(4,2,0,0,0,0,0,-1,-2);
  known.emplace_back(5,2,0,0,0,1e-20f,0,1,2);
  known.emplace_back(6,2,0,0,0,.1f,0,.99f,1e8f);
  s=Snapshot::capture(known,6);closure(s);
  check(s.joint.nonpositive==2 && s.joint.theta_zero==1 && s.joint.pt_zero==2,"zero angles/momenta are retained explicitly");
  check(s.joint_folded.both_zero==2,"backward exact axis folds to zero");
  check(s.joint.underflow==1 && s.joint.overflow==1,"nonzero extreme momenta retained as flows");
  check(s.pt.overflow==1,"marginal overflow retained");
  Marginal outside;outside.add(2e6,a.marginal_pt);std::ostringstream outside_json;outside.write(outside_json,a.marginal_pt);
  check(outside_json.str().find("\"mode\":null")!=std::string::npos,"all-overflow marginal has no invented mode");
  std::vector<PAIR_PARTICLE> empty;auto zero=Snapshot::capture(empty,0);closure(zero);
  std::ostringstream json;json << std::setprecision(17);zero.write_frame(json,0,0);
  check(json.str().find("\"available\":true")!=std::string::npos,"computed empty is available");
  check(json.str().find("\"statistics\":null")!=std::string::npos,"computed empty has null moments");
  rejects([&]{Snapshot::capture(known,5);});
  rejects([&]{Snapshot::capture(empty,max_records+1);});
  std::vector<PAIR_PARTICLE> invalid;
  invalid.emplace_back(1,0,0,0,0,0,0,0,2);rejects([&]{Snapshot::capture(invalid,1);});
  invalid.clear();invalid.emplace_back(1,0,0,0,0,std::numeric_limits<float>::infinity(),0,1,2);
  rejects([&]{Snapshot::capture(invalid,1);});

  // Exercise actual tracker ownership: one active record and one escaped
  // reserve record. Deactivation leaves stale active copies deliberately.
  PAIR_BEAM tracker;tracker.resize(1);RNDM rng(6712);MESH mesh(1,1,1,0,0);
  tracker.new_pair(mesh,0,0,0,0,2,.6f,.8f,std::sqrt(3.0-EMASS*EMASS),1,1,0,rng);
  tracker.new_pair(mesh,0,0,1000,1,-2,.6f,.8f,-std::sqrt(3.0-EMASS*EMASS),1,1,0,rng);
  auto first=Snapshot::capture(tracker.completed_tracking_snapshot(),tracker.n_particles());closure(first);
  check(first.nrecords==2,"initial admitted tracked population");
  tracker.distribute_pairs(10,1);
  check(tracker.numberOfParticles(0)==1,"one actual active record");
  rejects([&]{tracker.completed_tracking_snapshot();});
  tracker.get_pairs(0)[0].advanceVelocities(.1f,0,1);
  tracker.desactive_current_pairs(1);
  check(tracker.numberOfParticles(0)==1,"stale active copy persists after deactivation");
  const std::string before=states(tracker.completed_tracking_snapshot());
  RNDM control=rng;
  auto completed=Snapshot::capture(tracker.completed_tracking_snapshot(),tracker.n_particles());closure(completed);
  check(completed.nrecords==2,"reserve-only traversal avoids stale active duplicate");
  check(completed.pt.moments.mean!=first.pt.moments.mean,"current snapshot follows actual tracked momentum update");
  check(before==states(tracker.completed_tracking_snapshot()),"snapshot does not mutate native particles or order");
  for(int i=0;i<32;++i)check(rng.rndm_pairs()==control.rndm_pairs(),"snapshot does not consume RNG");
  tracker.distribute_pairs(10,1);tracker.desactive_current_pairs(1);
  auto next=Snapshot::capture(tracker.completed_tracking_snapshot(),tracker.n_particles());closure(next);
  check(next.nrecords==2,"repeated frames do not accumulate duplicate record counts");
  tracker.new_pair(mesh,0,0,1000,2,2,.2f,.1f,std::sqrt(3.95-EMASS*EMASS),1,1,0,rng);
  auto later=Snapshot::capture(tracker.completed_tracking_snapshot(),tracker.n_particles());closure(later);
  check(later.nrecords==3,"later snapshot includes newly admitted records once");
  check(first.nrecords==2,"previous immutable snapshot is unchanged");
  std::ostringstream final;final << std::setprecision(17);later.write_frame(final,2,4);
  check(final.str().size()<max_bytes/64,"one native frame respects total byte budget");

  // Position and direction are deliberately opposed. These records also mix
  // production process IDs, charge signs, exact +/-zero, excluded log values,
  // and native imported records to exercise the same generic tracker path.
  std::vector<PAIR_PARTICLE> positioned;
  positioned.emplace_back(1,0,0,0,5,0,0,-1,2);
  positioned.emplace_back(2,1,0,0,-3,.3f,.4f,1,-2);
  positioned.emplace_back(3,2,0,0,0,0,0,1,3);
  positioned.emplace_back(4,3,0,0,.25f,.1f,0,.99f,-4);
  positioned.emplace_back(5,4,0,0,-.25f,1e-20f,0,-.99f,5);
  positioned.emplace_back(6,5,0,0,-0.0f,.1f,0,-.99f,-5);
  positioned.emplace_back(7,-99,0,0,4,.2f,.1f,-.97f,6);
  const auto split=PositionSnapshot::capture(positioned,positioned.size());
  closure(split.all);closure(split.positive);closure(split.negative);
  check(split.all.nrecords==7 && split.positive.nrecords==3 && split.negative.nrecords==2 && split.zero_records==2,"position-z partition counts, with both signed zeros included only in All");
  std::vector<PAIR_PARTICLE> expected_positive{positioned[0],positioned[3],positioned[6]};
  std::vector<PAIR_PARTICLE> expected_negative{positioned[1],positioned[4]};
  std::vector<PAIR_PARTICLE> expected_zero{positioned[2],positioned[5]};
  const Snapshot zp=Snapshot::capture(expected_positive,3),zn=Snapshot::capture(expected_negative,2),zz=Snapshot::capture(expected_zero,2);
  const Snapshot old_inclusive=Snapshot::capture(positioned,positioned.size());
  const auto serialized=[](const Snapshot& v){std::ostringstream text;text << std::setprecision(17);v.write_frame(text,0,0);return text.str();};
  check(serialized(split.all)==serialized(old_inclusive),"inclusive serialization and statistical arithmetic are byte-identical to the old scan");
  check(serialized(split.positive)==serialized(zp) && serialized(split.negative)==serialized(zn),"subsets match independent position-index oracles, including exact marginal moments");
  check(split.positive.theta.moments.maximum==pi,"positive z may carry negative pz; direction is not the selection");
  check(split.positive.processes.at(-99)==1,"imported tracked particles follow the same stored-position convention");
  for(std::size_t i=0;i<joint_bins*joint_bins;++i) {
    check(split.positive.joint.counts[i]+split.negative.joint.counts[i]+zz.joint.counts[i]==split.all.joint.counts[i],"per-cell global-angle partition closure including z0");
    check(split.positive.joint_folded.counts[i]+split.negative.joint_folded.counts[i]+zz.joint_folded.counts[i]==split.all.joint_folded.counts[i],"per-cell folded-angle partition closure including z0");
  }
  for(const auto tuple:{std::array<const Joint*,4>{{&split.all.joint,&split.positive.joint,&split.negative.joint,&zz.joint}},std::array<const Joint*,4>{{&split.all.joint_folded,&split.positive.joint_folded,&split.negative.joint_folded,&zz.joint_folded}}}) {
    for(auto member:{&Joint::samples,&Joint::in_range,&Joint::theta_zero,&Joint::pt_zero,&Joint::both_zero,&Joint::nonpositive,&Joint::underflow,&Joint::overflow})
      check(tuple[0]->*member==tuple[1]->*member+tuple[2]->*member+tuple[3]->*member,"joint flow and counter partition closure");
  }
  for(std::size_t i=0;i<marginal_bins;++i) {
    check(split.positive.theta.counts[i]+split.negative.theta.counts[i]+zz.theta.counts[i]==split.all.theta.counts[i],"global marginal position closure");
    check(split.positive.theta_folded.counts[i]+split.negative.theta_folded.counts[i]+zz.theta_folded.counts[i]==split.all.theta_folded.counts[i],"folded marginal position closure");
    check(split.positive.pt.counts[i]+split.negative.pt.counts[i]+zz.pt.counts[i]==split.all.pt.counts[i],"pT marginal position closure");
  }
  std::ostringstream position_json;position_json << std::setprecision(17);split.write_frame(position_json,0,0);
  check(position_json.str().find("\"counts_sparse\":[")!=std::string::npos,"subsets use sparse row-major joints");
  check(position_json.str().find("\"position_z\":{\"zero_records\":2")!=std::string::npos,"z0 count exported explicitly");
  check(position_json.str().size()<max_bytes/64,"bounded sparse native fixture fits frame budget");
  std::ostringstream subset;split.positive.write_population(subset,true);
  check(subset.str().find("admitted_tracker_count")==std::string::npos && subset.str().find("authoritative_reserve_count")==std::string::npos,"subset does not claim full tracker ownership receipts");
  positioned[0].setZ(-5);
  const auto migrated=PositionSnapshot::capture(positioned,positioned.size());
  check(migrated.positive.nrecords==2 && migrated.negative.nrecords==3 && migrated.zero_records==2,"next frame follows current position after a plane crossing");
  check(split.positive.nrecords==3 && split.negative.nrecords==2,"previous frame remains immutable");
  const auto empty_split=PositionSnapshot::capture(empty,0);
  check(empty_split.all.nrecords==0 && empty_split.positive.nrecords==0 && empty_split.negative.nrecords==0 && empty_split.zero_records==0,"empty interval has valid empty subsets");
  rejects([&]{PositionSnapshot::capture(positioned,positioned.size()-1);});
  positioned[0].setZ(std::numeric_limits<float>::infinity());
  rejects([&]{PositionSnapshot::capture(positioned,positioned.size());});
  const std::string native_before=states(tracker.completed_tracking_snapshot());
  RNDM split_rng_control=rng;
  const auto tracked_split=PositionSnapshot::capture(tracker.completed_tracking_snapshot(),tracker.n_particles());
  check(tracked_split.all.nrecords==3 && tracked_split.positive.nrecords+tracked_split.negative.nrecords+tracked_split.zero_records==3,"authoritative reserve partition has no stale active duplicates");
  check(native_before==states(tracker.completed_tracking_snapshot()),"partition scan preserves every native particle and reserve order");
  for(int i=0;i<32;++i)check(rng.rndm_pairs()==split_rng_control.rndm_pairs(),"position partitions never consume RNG");
  if(argc==2){std::ofstream fixture(argv[1]);fixture << std::setprecision(17) << "{\"position_z_definition\":";write_position_z_definition(fixture);fixture << ",\"axes\":";write_axes(fixture);fixture << ",\"frames\":[";split.write_frame(fixture,0,0);fixture << ',';migrated.write_frame(fixture,1,1);fixture << ',';empty_split.write_frame(fixture,2,2);fixture << "]}\n";check(bool(fixture),"native sparse partition fixture saved");}
  std::cout << "PASS " << checks << " secondary frame checks; reserve lifecycle, z-position partition closure, current momenta, particle/RNG noninterference\n";

}
