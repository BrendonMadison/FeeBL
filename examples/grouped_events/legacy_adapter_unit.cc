#include "bhabhaSamplesCPP.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
void check(bool good,const char* why){if(!good)throw std::runtime_error(why);}
int main(int argc,char** argv)try {
  if(argc!=2)throw std::runtime_error("test needs fixture directory");
  const std::string prefix=std::string(argv[1])+"/";
  {std::ofstream f(prefix+"legacy-good.ini");f<<"42 1 2 3 4 -1 -2 -3 -4 0\n43 2 3 4 5 -2 -3 -4 5 1\n";}
  BHABHASAMPLES samples;check(samples.load_bhabha(prefix+"legacy-good.ini")==2,"legacy count");
  const auto& grouped=samples.grouped_event(0);check(grouped.id=="legacy-bhabha:42","event label");check(grouped.particles.size()==2,"grouped size");check(grouped.particles[1].input.energy==4,"nonnegative stored energy");check(grouped.particles[0].pdg==0&&grouped.particles[1].pdg==0,"legacy species explicitly unresolved");
  float px1,py1,pz1,en1,px2,py2,pz2,en2;int photons;unsigned int id;
  check(samples.pick_next_bhabha(125,-125,2,250,px1,py1,pz1,en1,px2,py2,pz2,en2,photons,id),"picker");
  check(id==42&&photons==0&&px1==2&&py1==4&&pz1==6&&en1==8&&px2==-2&&py2==-4&&pz2==-6&&en2==-8,"legacy signed/scaled exact outputs");
  float ecm,m1,m2,vx1,vy1,vz1,vx2,vy2,vz2;
  samples.get_parameters_for_output(0,id,ecm,m1,en1,vx1,vy1,vz1,m2,en2,vx2,vy2,vz2,photons);
  check(id==42&&ecm==250&&m1==125&&m2==125&&en1==4&&en2==-4&&vx1==.25f&&vy1==.5f&&vz1==.75f&&vx2==-.25f,"legacy output convention");
  {std::ofstream f(prefix+"legacy-bad.ini");f<<"42 1 2 3 4 -1 -2 -3 -4 0\n43 2 3\n";}
  BHABHASAMPLES bad;bool refused=false;try{bad.load_bhabha(prefix+"legacy-bad.ini");}catch(const std::exception&){refused=true;}check(refused,"truncated input refusal");
  bool no_partial=false;try{bad.grouped_event(0);}catch(const std::out_of_range&){no_partial=true;}check(no_partial,"transactional reader");
  {std::ofstream f(prefix+"legacy-duplicate.ini");f<<"42 1 2 3 4 -1 -2 -3 -4 0\n42 1 2 3 4 -1 -2 -3 -4 0\n";}
  BHABHASAMPLES duplicate;refused=false;try{duplicate.load_bhabha(prefix+"legacy-duplicate.ini");}catch(const std::exception&){refused=true;}check(refused,"duplicate ID refusal");
  std::cout<<"PASS legacy adapter exact values, grouping, signs, scaling and transactional malformed/duplicate refusal\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
