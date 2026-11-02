#include "pairsCPP.h"
#include "colliderImportedTrajectoriesCPP.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>
namespace {
unsigned checks=0;
void check(bool ok,const char* message){++checks;if(!ok){std::cerr << message << '\n';std::exit(1);}}
template<class F>void rejects(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes,"expected identity rejection");}
std::string current(const PAIR_BEAM& p){std::ostringstream out;out << std::setprecision(17);p.imported_registry().write_current(out,p.completed_tracking_snapshot());return out.str();}

void write_text(const std::string& path,const std::string& text) {
  std::ofstream out(path);out << text;check(bool(out),"fixture write");
}

std::string read_text(const std::string& path) {
  std::ifstream input(path);check(bool(input),"fixture read");
  return std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
}

void same_random_stream(RNDM& actual,RNDM& expected,const char* label) {
  for (int draw=0;draw<32;++draw) check(actual.rndm_pairs()==expected.rndm_pairs(),label);
}

// Both generated-particle entry points must reject invalid four-vectors
// before thinning or random cell-position draws mutate the simulation stream.
void check_generated_admission() {
  const float nan=std::numeric_limits<float>::quiet_NaN();
  const float infinity=std::numeric_limits<float>::infinity();
  const float invalid[][4]={
    {0,0,0,0}, {nan,0,0,1}, {infinity,0,0,1},
    {1,nan,0,1}, {1,0,infinity,1}, {-1,0,0,nan}};
  const MESH mesh(1,1,1,0,0);
  for (int indexed:{0,1}) for (const auto& row:invalid) {
    PAIR_BEAM secondaries;secondaries.resize(1);
    RNDM random(3178),expected=random;
    rejects([&] {
      if (indexed)
        secondaries.new_pair(17u,mesh,0,0,0,0,row[0],row[1],row[2],row[3],
                             0.3f,1,0,random,0,0);
      else
        secondaries.new_pair(mesh,0,0,0,0,row[0],row[1],row[2],row[3],
                             0.3f,1,0,random);
    });
    check(secondaries.n_particles()==0,"invalid generated state creates no particle");
    same_random_stream(random,expected,"invalid generated state consumes no random draws");
  }
  // Rest is a valid generated state, with charge carried independently of
  // momentum. Neither overload may discard it as an invalid zero direction.
  for (int indexed:{0,1}) for (int charge:{-1,1}) {
    PAIR_BEAM secondaries;secondaries.resize(1);
    RNDM random(3178);
    const float energy=charge*static_cast<float>(EMASS);
    if (indexed)
      secondaries.new_pair(17u,mesh,0,0,0,0,energy,0,0,0,1,1,0,random,0,0);
    else
      secondaries.new_pair(mesh,0,0,0,0,energy,0,0,0,1,1,0,random);
    check(secondaries.n_particles()==1,"rest state survives generated-particle admission");
    const auto& particle=secondaries.completed_tracking_snapshot().front();
    float vx,vy;particle.velocities(vx,vy);
    check(vx==0 && vy==0 && particle.Zvelocity()==0,
          "rest admission preserves a finite zero momentum");
  }
}

// Exercise the real event loader and its native sidecar, including copies
// between active/reserve ownership. Spins are supplied, never sampled here.
void check_imported_spin(const std::string& prefix) {
  const std::string spin_path=prefix+"_spin.ini", legacy_path=prefix+"_legacy.ini";
  const std::string output_path=prefix+"_spin.json";
  const std::string rows[]={"5 125 0 0 0 .6 0 .8", "5 -125 0 0 0 -.6 0 -.8",
                            "10 125 0 0 1000 0 1 0"};
  const std::string spins[]={" .6 0 .8", " .6 0 .8", " 0 .3 0"};
  std::string with_spin, without_spin;
  for (int row=0;row<3;++row) {
    with_spin+=rows[row]+spins[row]+"\n";
    without_spin+=rows[row]+"\n";
  }
  write_text(spin_path,with_spin);write_text(legacy_path,without_spin);
  PAIR_BEAM polarized,legacy;
  polarized.resize(1);legacy.resize(1);
  polarized.enable_gui_diagnostics(true);legacy.enable_gui_diagnostics(true);
  polarized.enable_imported_spin(true);
  polarized.set_load_file(spin_path);legacy.set_load_file(legacy_path);
  RNDM spin_rng(81237),legacy_rng(81237);
  for (int time:{0,5,12,20}) {
    polarized.load_events(time,1,1,spin_rng);
    legacy.load_events(time,1,1,legacy_rng);
    check(polarized.n_particles()==legacy.n_particles(),"spin columns preserve deferred event admission");
  }
  same_random_stream(spin_rng,legacy_rng,"spin parsing consumes no additional admission RNG");
  check(polarized.n_particles()==3,"all eleven-column records admitted");
  int row=0;
  for (const auto& particle:polarized.completed_tracking_snapshot()) {
    check(particle.imported_spin_enabled(),"imported particle retains enabled spin state");
    check(particle.get_label()==row+1,"spin import retains source identity");
    const double expected_helicity[]={1,-1,.3};
    check(std::abs(particle.imported_helicity()-expected_helicity[row])<1e-14,
          "helicity uses normalized actual momentum, including positron direction");
    ++row;
  }
  polarized.write_imported_spin(output_path.c_str());
  const std::string initial_json=read_text(output_path);
  check(initial_json.find("\"source_record\":2")!=std::string::npos &&
        initial_json.find("\"pdg_id\":-11")!=std::string::npos,
        "sidecar preserves positron source and species identity");
  const auto spin_position=initial_json.find("\"spin\":[");
  check(spin_position!=std::string::npos,"sidecar contains spin vector");
  std::istringstream serialized_spin(initial_json.substr(spin_position+8));
  double sx=0,sy=0,sz=0;char comma1=0,comma2=0;
  check(bool(serialized_spin>>sx>>comma1>>sy>>comma2>>sz) && comma1==',' && comma2==',' &&
        sx==.6 && sy==0 && sz==.8,"sidecar spin values round trip at double precision");
  check(initial_json.find("\"radiative_spin_flips\":false")!=std::string::npos,
        "sidecar states radiation-free transport scope");

  polarized.distribute_pairs(10,1);
  check(polarized.numberOfParticles(0)==2,"two polarized particles enter active tracking");
  RNDM untouched_rng=spin_rng;
  for (auto& particle:polarized.get_pairs(0)) {
    particle.apply_initial_half_step_fields(1,EMASS,1e-8f,0,0,2e-8f,spin_rng);
    particle.apply_full_step_fields(1,EMASS,1e-8f,0,0,2e-8f,spin_rng);
    particle.apply_final_half_step_fields(1,EMASS,1e-8f,0,0,2e-8f,0,spin_rng);
    check(std::isfinite(particle.imported_helicity()),"tracked helicity remains finite");
    check(particle.imported_spin_steps()==3,"all physical half/full field intervals recorded");
  }
  same_random_stream(spin_rng,untouched_rng,"radiation-free BMT transport consumes no RNG");
  polarized.desactive_current_pairs(1);
  polarized.write_imported_spin(output_path.c_str());
  const std::string final_json=read_text(output_path);
  polarized.write_imported_spin(output_path.c_str());
  check(final_json==read_text(output_path),"spin serialization is observational and repeatable");
  check(final_json!=initial_json,"sidecar records actual transported spin state");
  const auto& reordered=polarized.completed_tracking_snapshot();
  check(reordered.front().get_label()==3 && reordered.front().getSpin()(1)==.3,
        "spin survives reserve reordering and partial polarization is preserved");
  check(final_json.find("\"label\":1")<final_json.find("\"label\":3"),
        "sidecar sorts by stable identity rather than final reserve order");

  // Malformed spin data must fail before thinning draws or particle creation.
  const std::string invalid[]={
    rows[0], rows[0]+" 0", rows[0]+" 0 0", rows[0]+" 0 0 1 extra",
    rows[0]+" 0 0 1.1", rows[0]+" nan 0 0",
    "5 0.0001 0 0 0 0 0 1 0 0 1", "5 125 0 0 0 0 0 0 0 0 1",
    "5 125 0 0 0 0 0 1.1 0 0 1"};
  for (std::size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);++i) {
    const std::string bad_path=prefix+"_bad_"+std::to_string(i)+".ini";
    write_text(bad_path,invalid[i]+"\n");
    {
      PAIR_BEAM bad;bad.resize(1);bad.enable_imported_spin(true);bad.set_load_file(bad_path);
      RNDM rng(5531),control=rng;
      rejects([&]{bad.load_events(5,1,1,rng);});
      check(bad.n_particles()==0,"invalid spin input admits no particle");
      same_random_stream(rng,control,"invalid spin input consumes no RNG");
    }
    std::remove(bad_path.c_str());
  }
  {
    PAIR_BEAM disabled;disabled.resize(1);disabled.set_load_file(spin_path);RNDM rng(9);
    rejects([&]{disabled.load_events(5,1,1,rng);});
    check(disabled.n_particles()==0,"spin columns require explicit spin mode");
  }
  // The two live readers still own their streams; POSIX permits unlinking
  // these private temporary fixtures after all checks have completed.
  std::remove(spin_path.c_str());std::remove(legacy_path.c_str());std::remove(output_path.c_str());
}
}
int main(int argc,char** argv){
  const std::string path="/tmp/feebl_imported_identity_"+std::to_string(getpid())+".ini";
  {std::ofstream f(path);f << "5 125 0 0 0 .01745 0 .99984\n5 -125 0 0 0 -.01745 0 -.99984\n10 125 0 0 1000 .01 .02 .999\n";}
  PAIR_BEAM tracker;tracker.resize(1);tracker.enable_gui_diagnostics(true);tracker.set_load_file(path);RNDM rng(6712);
  tracker.load_events(0,1,1,rng);check(tracker.n_particles()==0,"future event must remain deferred");
  check(current(tracker)=="[]","empty pre-injection snapshot");
  tracker.load_events(5,1,1,rng);check(tracker.n_particles()==2,"whole event admitted exactly once");
  const auto& initial=tracker.imported_registry().initial();
  check(initial.size()==2 && initial[0].source_record==1 && initial[1].source_record==2,"source row identity captured at ingestion");
  check(initial[0].label!=initial[1].label && initial[0].scheduled==5 && initial[0].admitted==5,"stable unique labels and injection counter");
  tracker.load_events(7,1,1,rng);check(tracker.n_particles()==2,"second deferred row not admitted early");
  tracker.load_events(12,1,1,rng);check(tracker.n_particles()==3,"deferred final row admitted at actual call time");
  check(tracker.imported_registry().initial()[2].source_record==3 && tracker.imported_registry().initial()[2].admitted==12,"deferred row preserves original ordinal and actual admission time");
  RNDM control=rng;
  for(int time=13;time<45;++time){tracker.load_events(time,1,1,rng);check(tracker.n_particles()==3,"EOF must not reimport last deferred row");}
  for(int i=0;i<32;++i)check(rng.rndm_pairs()==control.rndm_pairs(),"no RNG calls after EOF");
  tracker.distribute_pairs(10,1);rejects([&]{current(tracker);});
  check(tracker.numberOfParticles(0)==2,"two imported particles enter actual active pool");
  tracker.get_pairs(0)[0].record_imported_field_gather(0);
  tracker.get_pairs(0)[0].record_imported_field_gather(2);
  tracker.get_pairs(0)[0].record_imported_field_gather(-1);
  tracker.get_pairs(0)[0].advanceVelocities(.01f,0,1);
  tracker.desactive_current_pairs(1);
  const std::string state=current(tracker);check(state==current(tracker),"snapshot is observational and deterministic");
  check(state.find(",1,1,1]")!=std::string::npos,"actual gather categories survive active/reserve copies");
  check(tracker.completed_tracking_snapshot().front().get_label()==3,"real reserve order differs from input order");
  check(state.find("[[1,")==0,"export joins and sorts stable labels instead of reserve order");
  std::list<PAIR_PARTICLE> duplicate=tracker.completed_tracking_snapshot();duplicate.push_back(duplicate.front());
  rejects([&]{std::ostringstream out;tracker.imported_registry().write_current(out,duplicate);});
  PAIR_PARTICLE electron(1,-99,0,0,0,0,0,1,125),positron(2,-99,0,0,0,0,0,1,-125);
  electron.apply_electric_field_on_pair(1,2,0);positron.apply_electric_field_on_pair(1,2,0);
  float ex,ey,px,py;electron.velocities(ex,ey);positron.velocities(px,py);
  check(ex>0 && px<0 && ex==-px,"+energy and -energy reverse actual electric kick");
  PAIR_PARTICLE em(1,-99,0,0,0,0,0,1,125),pm(2,-99,0,0,0,0,0,1,-125);
  em.apply_magnetic_field_on_pair(.25,1,0,2);pm.apply_magnetic_field_on_pair(.25,1,0,2);
  em.velocities(ex,ey);pm.velocities(px,py);
  check(ex<0 && px>0 && ex==-px,"native +energy charge has v cross B sign and opposite charge reverses it");
  // Under charge_sign=-1 the native field_pair output for an isolated beam2
  // is ex=-E2, by=E2. A +z positive-energy tracker therefore gets -2E2,
  // matching the native primary electron's force, not the positron's.
  PAIR_PARTICLE untouched(9,-99,0,0,0,.1f,.2f,.9f,125);untouched.record_imported_field_gather(-1);
  check(untouched.imported_field_gathers(2)==0,"ordinary particles do not enable imported diagnostic counters");
  feebl::imported_trajectories::Registry bounded;
  for(std::size_t i=0;i<=feebl::imported_trajectories::max_records;++i)bounded.add(untouched,i+1,0,0);
  check(bounded.overflow() && bounded.initial().empty(),"diagnostic bound does not limit physical import");
  if(argc==2){std::ofstream out(argv[1]);out << std::setprecision(17) << "{\"initial_records\":";tracker.imported_registry().write_initial(out);out << ",\"current\":" << state << "}\n";check(bool(out),"unit fixture write");}
  check_imported_spin(path);
  check_generated_admission();
  std::remove(path.c_str());std::cout << "PASS " << checks << " imported identity, delayed-input, spin, ownership, force-sign and gather checks\n";
}
