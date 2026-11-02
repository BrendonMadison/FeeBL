#include "groupedEventsCPP.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace feebl::events;
namespace {
int checks=0;
void check(bool good,const std::string& why) {++checks;if(!good)throw std::runtime_error(why);}
void near(double got,double want,double atol,const std::string& why) {check(std::abs(got-want)<=atol,why);}
template<class F> void refused(F f,const std::string& why) {bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,why);}
Particle particle(int pdg,std::size_t row,double px,double py,double pz) {
  Particle p;p.pdg=pdg;p.id="source:0:"+std::to_string(row);p.original_row=row;p.mass_GeV=species(pdg).mass_GeV;
  p.input={px,py,pz,std::hypot(std::hypot(px,py),std::hypot(pz,p.mass_GeV))};p.output=p.input;return p;
}
Event event(std::size_t n) {
  Event e;e.id="source:0";e.process_id=7;e.weight=-.25;e.scale=91.2;e.alpha_qed=.007;e.alpha_qcd=.118;
  for(std::size_t i=0;i<n;++i)e.particles.push_back(particle(i%3==2?22:(i%2? -11:11),i+1,.1,.2,i%2?-2:3));
  return e;
}
LheDocument document(std::size_t n) {
  LheDocument d;d.prefix="<LesHouchesEvents version=\"3.0\">\n<header><generator>synthetic transport qualification fixture</generator></header>\n<init>\n11 -11 500 31.3 0 0 0 0 -4 1\n1.0 0.01 1.0 7\n</init>\n";d.suffix="</LesHouchesEvents>\n";d.events.push_back(event(n));
  d.events[0].extensions="<rwgt><wgt id=\"scale_up\">-0.3</wgt></rwgt>\n";return d;
}
struct Constant:FieldProvider {
  Field field;std::string version()const override{return "manufactured_constant_v1";}
  Field at(const Vec3&,double)const override{return field;}
};
struct Varying:FieldProvider {
  mutable double earliest=1e20,latest=-1e20;
  std::string version()const override{return "manufactured_time_linear_v1";}
  Field at(const Vec3&,double t)const override {earliest=std::min(earliest,t);latest=std::max(latest,t);Field f;f.electric_GV_per_mm[0]=1e-4*t;return f;}
};
}
int main(int argc,char** argv)try {
  Constant zero;Vertex v;v.position_mm={{1,2,3}};v.time_ns=.2;v.method="explicit_lab";
  TransportPolicy policy;policy.end_time_ns=.3;policy.max_step_ns=.002;
  for(std::size_t n:{1,2,3,17}) {
    auto d=document(n);std::ostringstream text;write_lhe(text,d);std::istringstream in(text.str());auto back=read_lhe(in,"fixture_sha");
    check(back.events[0].particles.size()==n,"multiplicity roundtrip");check(back.events[0].weight==-.25,"signed generator weight");
    check(back.events[0].extensions==d.events[0].extensions,"extension bytes");check(back.prefix==d.prefix,"asymmetric init preserved");
    auto tr=transport(back.events[0],v,zero,policy);check(tr.size()==n,"variable final multiplicity");
    for(const auto& t:tr) {near(t.output.px,t.input.px,0,"zero-field momentum");near(t.output_vertex.position_mm[2],3+299.792458*.1*t.input.pz/t.input.energy,1e-11,"zero-field drift");near(t.electric_work_GeV,0,0,"zero-field work");}
    std::reverse(back.events[0].particles.begin(),back.events[0].particles.end());std::ostringstream sorted;write_lhe(sorted,back);std::istringstream sorted_in(sorted.str());auto reread=read_lhe(sorted_in,"fixture_sha");
    check(reread.events[0].particles[0].id=="fixture_sha:0:1","sorting stable particle identity");
  }
  auto e=event(1);auto input=e.particles[0].input;Constant electric;electric.field.electric_GV_per_mm[0]=.001;
  auto tr=transport(e,v,electric,policy);near(tr[0].output.px,input.px-299.792458*.001*.1,1e-13,"uniform E impulse");near(tr[0].output.energy-input.energy,tr[0].electric_work_GeV,1e-12,"uniform E work");
  auto positron=event(1);positron.particles[0]=particle(-11,1,.1,.2,3);auto pos=transport(positron,v,electric,policy);near(pos[0].output.px,input.px+299.792458*.001*.1,1e-13,"charge reversal");
  auto muon=event(1);muon.particles[0]=particle(13,1,.1,.2,3);auto mu=transport(muon,v,electric,policy);near(mu[0].output.px,tr[0].output.px,1e-14,"mass-independent electric impulse");check(mu[0].output_vertex.position_mm[2]<tr[0].output_vertex.position_mm[2],"mass-dependent velocity");
  Varying varying;auto timed=event(1);auto dynamic=transport(timed,v,varying,policy);near(dynamic[0].output.px,input.px-299.792458*1e-4*(.3*.3-.2*.2)/2,1e-13,"evolving-time field impulse");near(varying.earliest,.2,1e-15,"correct creation time");near(varying.latest,.3,1e-15,"correct final exposure time");
  Constant magnetic;magnetic.field.cB_GV_per_mm[2]=.02;auto b=event(1);auto small=policy;small.max_step_ns=.001;auto bt=transport(b,v,magnetic,small);double angle=299.792458*.02*.1/input.energy;
  near(bt[0].output.px,input.px*std::cos(angle)-input.py*std::sin(angle),1e-12,"uniform B analytic orbit px");near(bt[0].output.py,input.px*std::sin(angle)+input.py*std::cos(angle),1e-12,"uniform B analytic orbit py");near(bt[0].output.energy,input.energy,1e-13,"magnetic energy");near(bt[0].electric_work_GeV,0,0,"magnetic work");
  auto coarse=policy;coarse.max_step_ns=.1;auto medium=policy;medium.max_step_ns=.05;auto eb=event(1);auto c=transport(eb,v,magnetic,coarse);auto m=transport(eb,v,magnetic,medium);check(std::abs(m[0].output.px-bt[0].output.px)<std::abs(c[0].output.px-bt[0].output.px)/10,"RK4 step convergence");
  auto neutral=event(1);neutral.particles[0]=particle(22,1,0,0,2);auto nt=transport(neutral,v,electric,policy);near(nt[0].output.energy,2,0,"neutral exact energy");near(nt[0].output.px,0,0,"neutral no kick");
  auto cancellation=event(1);cancellation.particles[0]=particle(11,1,0,0,1000);
  Constant crossed;crossed.field.cB_GV_per_mm[1]=.001;crossed.field.electric_GV_per_mm[0]=.001*cancellation.particles[0].input.pz/cancellation.particles[0].input.energy;
  auto ct=transport(cancellation,v,crossed,policy);near(ct[0].output.px,0,1e-15,"ultrarelativistic crossed-field cancellation");
  auto mixed=event(3);mixed.particles[0].status=-1;mixed.particles[1].mothers={{1,1}};auto mt=transport(mixed,v,electric,policy);check(mt.size()==2,"only final states transported");near(mixed.particles[0].output.px,mixed.particles[0].input.px,0,"incoming unchanged");
  auto invalid=event(2);invalid.particles[1].pdg=211;auto before=invalid.particles[0].output;refused([&]{transport(invalid,v,electric,policy);},"unknown species refused");near(invalid.particles[0].output.px,before.px,0,"failed event atomicity");
  auto wrongmass=event(1);wrongmass.particles[0].mass_GeV=1;refused([&]{transport(wrongmass,v,electric,policy);},"inconsistent mass refused");
  auto offshell=event(1);offshell.particles[0].input.energy=1;refused([&]{transport(offshell,v,electric,policy);},"off-shell refused");
  auto radiation=policy;radiation.radiation="beamstrahlung";refused([&]{transport(e,v,electric,radiation);},"unqualified radiation refused");
  auto decay=policy;decay.decay="enabled";refused([&]{transport(e,v,electric,decay);},"decay policy refused");
  auto budget=policy;budget.max_steps=1;refused([&]{transport(e,v,electric,budget);},"bounded steps");
  auto origin=v;origin.method="";refused([&]{transport(e,origin,electric,policy);},"invented origin refused");
  std::ostringstream valid;write_lhe(valid,document(3));auto bytes=valid.str();
  refused([&]{std::istringstream s(bytes.substr(0,bytes.find("</event>")));read_lhe(s,"x");},"truncated event refused");
  refused([&]{std::istringstream s(bytes.substr(0,bytes.find("</LesHouchesEvents>")));read_lhe(s,"x");},"truncated root refused");
  auto bad=bytes;bad.replace(bad.find("3 7 -0.25"),1,"2");refused([&]{std::istringstream s(bad);read_lhe(s,"x");},"NUP undercount refused");
  auto xml=bytes;xml.replace(xml.find("</rwgt>"),7,"</wrong>");refused([&]{std::istringstream s(xml);read_lhe(s,"x");},"malformed extension XML refused");
  auto extra=bytes;extra.insert(extra.find("</event>"),"# comment\n11 1 0 0 0 0 0 0 1 1 0 0 9\n");refused([&]{std::istringstream s(extra);read_lhe(s,"x");},"extra row after comments refused");
  // Independent-review regressions: unbounded recursive regex, substring
  // extraction from comments, and XML entities must not bypass validation.
  const auto parse=[](const std::string& value){std::istringstream input(value);return read_lhe(input,"review");};
  auto long_attribute=bytes;
  long_attribute.replace(long_attribute.find("<generator>"),11,"<generator note=\""+std::string(50000,'x')+"\">");
  auto long_doc=parse(long_attribute);std::ostringstream long_roundtrip;write_lhe(long_roundtrip,long_doc);
  check(long_doc.prefix==long_attribute.substr(0,long_attribute.find("<event>"))&&parse(long_roundtrip.str()).prefix==long_doc.prefix,"50k quoted header attribute exact preservation");
  auto long_event_attribute=bytes;
  long_event_attribute.replace(long_event_attribute.find("<event>"),7,"<event note='"+std::string(50000,'x')+"'>");
  auto long_event_doc=parse(long_event_attribute);std::ostringstream long_event_roundtrip;write_lhe(long_event_roundtrip,long_event_doc);
  check(long_event_doc.events[0].opening_tag=="<event note='"+std::string(50000,'x')+"'>"&&parse(long_event_roundtrip.str()).events[0].opening_tag==long_event_doc.events[0].opening_tag,"50k quoted event attribute exact preservation");
  auto unclosed_long=long_attribute;unclosed_long.erase(unclosed_long.find(std::string(50000,'x'))+50000,1);
  refused([&]{parse(unclosed_long);},"long unterminated quoted attribute refused");
  const auto init_begin=bytes.find("<init>"),init_end=bytes.find("</init>")+7;
  const auto init_block=bytes.substr(init_begin,init_end-init_begin);
  auto comment_init=bytes;comment_init.replace(init_begin,init_end-init_begin,"<!--"+init_block+"-->");
  refused([&]{parse(comment_init);},"comment cannot supply actual init");
  auto decoy_init=bytes;decoy_init.insert(init_begin,"<!--<init>not numeric &unknown;</init>-->\n");
  check(parse(decoy_init).events.size()==1,"commented init and raw comment entity ignored with actual init");
  auto nested_init=bytes;nested_init.replace(init_begin,init_end-init_begin,"<wrapper>"+init_block+"</wrapper>");
  refused([&]{parse(nested_init);},"nested init cannot supply direct-root init");
  auto duplicated_init=bytes;duplicated_init.insert(init_end,init_block);
  refused([&]{parse(duplicated_init);},"duplicate actual init refused");
  auto self_closing_init=bytes;self_closing_init.replace(init_begin,init_end-init_begin,"<init/>");
  refused([&]{parse(self_closing_init);},"self-closing init refused");
  for(const std::string entity:{"&unknown;","&amp","&#;","&#x;","&#-1;","&#x110000;","&#1114112;","&#xD800;","&#0;","&#11;","&#xFFFE;","&#xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;","&#12z;"}) {
    auto header_entity=bytes;header_entity.insert(header_entity.find("</generator>"),entity);
    refused([&]{parse(header_entity);},"invalid header entity refused: "+entity);
    auto extension_entity=bytes;extension_entity.insert(extension_entity.find("</wgt>"),entity);
    refused([&]{parse(extension_entity);},"invalid event extension entity refused: "+entity);
    auto attribute_entity=bytes;attribute_entity.replace(attribute_entity.find("<generator>"),11,"<generator note=\""+entity+"\">");
    refused([&]{parse(attribute_entity);},"invalid quoted attribute entity refused: "+entity);
  }
  auto valid_entities=bytes;
  const std::string entities="&amp;&lt;&gt;&apos;&quot;&#9;&#10;&#13;&#32;&#65;&#x41;&#x10FFFF;";
  valid_entities.replace(valid_entities.find("<generator>"),11,"<generator note=\""+entities+"\">");
  valid_entities.insert(valid_entities.find("</generator>"),entities);
  valid_entities.insert(valid_entities.find("</wgt>"),entities);
  auto entity_doc=parse(valid_entities);std::ostringstream entity_roundtrip;write_lhe(entity_roundtrip,entity_doc);
  auto entity_reread=parse(entity_roundtrip.str());
  check(entity_doc.prefix==valid_entities.substr(0,valid_entities.find("<event>"))&&entity_reread.prefix==entity_doc.prefix&&entity_reread.events[0].extensions==entity_doc.events[0].extensions,"predefined and legal numeric XML entities preserved exactly");
  for(const std::string weight:{"-2147483648","2147483647","2147483648","0","-5","5"}) {
    auto invalid_weight=bytes;invalid_weight.replace(invalid_weight.find("0 0 0 0 -4 1"),12,"0 0 0 0 "+weight+" 1");
    refused([&]{parse(invalid_weight);},"invalid IDWTUP including INT_MIN refused: "+weight);
  }
  for(const std::string weight:{"-4","-3","-2","-1","1","2","3","4"}) {
    auto valid_weight=bytes;valid_weight.replace(valid_weight.find("0 0 0 0 -4 1"),12,"0 0 0 0 "+weight+" 1");
    check(parse(valid_weight).events.size()==1,"valid IDWTUP accepted: "+weight);
  }
  auto dupe=event(2);dupe.particles[1].id=dupe.particles[0].id;refused([&]{validate_event(dupe);},"duplicate identity refused");
  std::ostringstream side;write_sidecar(side,std::string(64,'a'),dynamic);check(side.str().find("\"live_bunch_integration\":false")!=std::string::npos,"sidecar capability honesty");
  if(argc==2) {
    auto d=document(3);std::ofstream original(std::string(argv[1])+"/synthetic.original.lhe");write_lhe(original,d);
    auto tracks=transport(d.events[0],v,electric,policy);std::ofstream transported(std::string(argv[1])+"/synthetic.transported.lhe");write_lhe(transported,d,true);
    std::ofstream json(std::string(argv[1])+"/synthetic.sidecar.unbound.json");write_sidecar(json,std::string(64,'0'),tracks);
  }
  std::cout<<"PASS grouped-event qualification checks="<<checks<<" live_bunch_integration=false real_generator_interop=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
