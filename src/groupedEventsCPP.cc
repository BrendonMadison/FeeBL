#include "groupedEventsCPP.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace feebl { namespace events {
namespace {
constexpr double c_mm_ns=299.792458;
constexpr double pi=3.1415926535897932384626433832795;
void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error("grouped events: "+why);
}
bool finite(double x) { return std::isfinite(x); }
bool finite(const FourMomentum& p) {
  return finite(p.px)&&finite(p.py)&&finite(p.pz)&&finite(p.energy);
}
std::string trim(std::string s) {
  const auto a=s.find_first_not_of(" \t\r\n");
  return a==std::string::npos ? "" : s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
template<class... T> void row(const std::string& s,T&... values) {
  std::string data=s.substr(0,s.find('#'));
  std::istringstream in(data);
  require(bool((in>>...>>values)),"invalid or truncated numeric row");
  std::string extra;
  require(!(in>>extra),"unexpected extra numeric fields");
}
std::string next_data(std::istringstream& in) {
  std::string line;
  while(std::getline(in,line)) {
    auto t=trim(line);
    if (!t.empty()&&t[0]!='#') return t;
  }
  throw std::runtime_error("grouped events: missing numeric row");
}
std::string event_data(std::istringstream& in) {
  std::string line;
  while(std::getline(in,line)) {
    auto t=trim(line);
    if(t.empty())continue;
    // This profile preserves metadata after NUP rows. Refuse metadata in
    // numerical rows instead of silently deleting comments or annotations.
    require(t.find('#')==std::string::npos,"comments within numerical event rows unsupported");
    return t;
  }
  throw std::runtime_error("grouped events: missing event numeric row");
}
// The XML envelope is scanned iteratively. In particular, an arbitrarily long
// quoted attribute must not consume the C++ call stack through std::regex.
struct EventBlock { std::size_t begin=0, data_begin=0, data_end=0, end=0; };
struct Markup {
  std::vector<EventBlock> events;
  EventBlock init;
  bool has_init=false;
};
bool xml_space(char c) { return c==' '||c=='\t'||c=='\r'||c=='\n'; }
bool name_start(char c) {
  return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'||c==':';
}
bool name_character(char c) {
  return name_start(c)||(c>='0'&&c<='9')||c=='-'||c=='.';
}
bool xml_character(unsigned long value) {
  return value==9||value==10||value==13||
         (value>=0x20&&value<=0xd7ff)||
         (value>=0xe000&&value<=0xfffd)||
         (value>=0x10000&&value<=0x10ffff);
}
void validate_entities(const std::string& text,std::size_t begin,std::size_t end) {
  for(std::size_t i=begin;i<end;++i) {
    const unsigned char c=text[i];
    require(c>=0x20||c==9||c==10||c==13,"illegal XML control character");
    if(c!='&')continue;
    const auto semicolon=text.find(';',i+1);
    require(semicolon!=std::string::npos&&semicolon<end,"unterminated XML entity");
    const auto body=text.substr(i+1,semicolon-i-1);
    if(body=="amp"||body=="lt"||body=="gt"||body=="apos"||body=="quot") {
      i=semicolon;continue;
    }
    require(!body.empty()&&body[0]=='#',"undefined XML entity");
    std::size_t digit=1;unsigned long base=10,value=0;
    if(digit<body.size()&&body[digit]=='x') {base=16;++digit;}
    require(digit<body.size(),"empty XML numeric entity");
    for(;digit<body.size();++digit) {
      const char d=body[digit];unsigned long n=16;
      if(d>='0'&&d<='9')n=static_cast<unsigned long>(d-'0');
      else if(d>='a'&&d<='f')n=static_cast<unsigned long>(d-'a'+10);
      else if(d>='A'&&d<='F')n=static_cast<unsigned long>(d-'A'+10);
      require(n<base&&value<=(0x10ffff-n)/base,"invalid/out-of-range XML numeric entity");
      value=value*base+n;
    }
    require(xml_character(value),"illegal XML numeric character");
    i=semicolon;
  }
}
struct XmlTag {
  std::string name;
  std::vector<std::pair<std::string,std::string>> attributes;
  bool closing=false,self_closing=false;
};
XmlTag parse_tag(const std::string& text,std::size_t begin,std::size_t end) {
  XmlTag tag;std::size_t at=begin;
  require(at<end,"empty XML tag");
  if(text[at]=='/') {tag.closing=true;++at;}
  require(at<end&&name_start(text[at]),"invalid XML element name");
  const auto name_begin=at++;
  while(at<end&&name_character(text[at]))++at;
  tag.name=text.substr(name_begin,at-name_begin);
  std::set<std::string> seen;
  while(at<end) {
    const auto before_space=at;
    while(at<end&&xml_space(text[at]))++at;
    if(at==end)break;
    if(!tag.closing&&text[at]=='/'&&at+1==end) {
      tag.self_closing=true;break;
    }
    require(!tag.closing&&at>before_space&&name_start(text[at]),"invalid XML attribute syntax");
    const auto attr_begin=at++;
    while(at<end&&name_character(text[at]))++at;
    auto name=text.substr(attr_begin,at-attr_begin);
    require(seen.insert(name).second,"duplicate XML attribute");
    while(at<end&&xml_space(text[at]))++at;
    require(at<end&&text[at]=='=',"missing XML attribute equals");++at;
    while(at<end&&xml_space(text[at]))++at;
    require(at<end&&(text[at]=='\''||text[at]=='"'),"unquoted XML attribute");
    const char quote=text[at++];const auto value_begin=at;
    while(at<end&&text[at]!=quote) {
      require(text[at]!='<',"unescaped angle in XML attribute");++at;
    }
    require(at<end,"unterminated XML attribute");
    validate_entities(text,value_begin,at);
    tag.attributes.emplace_back(name,text.substr(value_begin,at-value_begin));++at;
  }
  return tag;
}
Markup validate_markup(const std::string& doc) {
  std::vector<std::string> stack;Markup result;
  std::size_t at=0;bool seen_root=false,closed_root=false,seen_declaration=false;
  while(at<doc.size()) {
    if(doc[at]!='<') {
      const auto tag=doc.find('<',at);
      const auto end=tag==std::string::npos?doc.size():tag;
      validate_entities(doc,at,end);
      if(stack.empty())
        for(auto i=at;i<end;++i)require(xml_space(doc[i]),"text outside XML root");
      require(doc.substr(at,end-at).find("]]>")==std::string::npos,"CDATA terminator in XML text");
      at=end;continue;
    }
    if(doc.compare(at,4,"<!--")==0) {
      const auto end=doc.find("-->",at+4);
      require(end!=std::string::npos,"truncated XML comment");
      require(doc.substr(at+4,end-at-4).find("--")==std::string::npos,"invalid XML comment");
      at=end+3;continue;
    }
    if(doc.compare(at,5,"<?xml")==0) {
      require(at==0&&!seen_root&&!seen_declaration,"misplaced XML declaration");
      const auto end=doc.find("?>",at+5);
      require(end!=std::string::npos,"truncated XML declaration");
      const auto declaration=parse_tag(doc,at+2,end);
      require(declaration.name=="xml"&&!declaration.closing&&!declaration.self_closing&&
              !declaration.attributes.empty()&&declaration.attributes[0]==std::make_pair(std::string("version"),std::string("1.0")),"unsupported XML declaration");
      for(std::size_t i=1;i<declaration.attributes.size();++i) {
        const auto& attribute=declaration.attributes[i];
        require((attribute.first=="encoding"&&(attribute.second=="UTF-8"||attribute.second=="utf-8"))||
                (attribute.first=="standalone"&&(attribute.second=="yes"||attribute.second=="no")),"unsupported XML declaration attribute");
      }
      seen_declaration=true;at=end+2;continue;
    }
    require(doc.compare(at,2,"<!")!=0&&doc.compare(at,2,"<?")!=0,"XML declaration outside supported profile");
    std::size_t end=at+1;char quote=0;
    for(;end<doc.size();++end) {
      const char c=doc[end];
      if(quote) {if(c==quote)quote=0;}
      else if(c=='\''||c=='"')quote=c;
      else if(c=='>')break;
    }
    require(end<doc.size()&&!quote,"truncated XML tag");
    const auto tag=parse_tag(doc,at+1,end);
    const auto& key=tag.name;
    if(tag.closing) {
      require(!stack.empty()&&stack.back()==key,"mismatched XML closing tag");
      if(key=="event") {result.events.back().data_end=at;result.events.back().end=end+1;}
      if(key=="init") {result.init.data_end=at;result.init.end=end+1;}
      stack.pop_back();if(stack.empty())closed_root=true;
    } else {
      require(!closed_root,"multiple XML roots");
      if(stack.empty()) {
        require(!seen_root&&key=="LesHouchesEvents"&&!tag.self_closing,"LHE root required");
        bool supported_version=false;
        for(const auto& attribute:tag.attributes)if(attribute.first=="version") {
          const auto& value=attribute.second;
          supported_version=value.size()>=3&&value[0]=='3'&&value[1]=='.';
          for(std::size_t i=2;i<value.size();++i)
            supported_version=supported_version&&value[i]>='0'&&value[i]<='9';
        }
        require(supported_version,"LHEF 3.x root required");seen_root=true;
      }
      if(key=="init") {
        require(stack.size()==1&&stack.back()=="LesHouchesEvents"&&!tag.self_closing&&
                !result.has_init&&result.events.empty(),"exactly one direct-root init before events required");
        result.has_init=true;result.init={at,end+1,0,0};
      }
      if(key=="event") {
        require(stack.size()==1&&stack.back()=="LesHouchesEvents"&&!tag.self_closing&&result.init.end>0,"event requires preceding direct-root init");
        result.events.push_back({at,end+1,0,0});
      }
      if(!tag.self_closing)stack.push_back(key);
    }
    at=end+1;
  }
  require(stack.empty()&&closed_root&&result.has_init&&result.init.end>0,"complete root and actual direct-root init required");
  return result;
}
void validate_extension_payload(const std::string& text) {
  int depth=0;std::size_t at=0;
  while(at<text.size()) {
    if(text[at]!='<') {
      const auto end=text.find('<',at);
      if(depth==0) {
        std::istringstream plain(text.substr(at,end==std::string::npos?end:end-at));
        std::string line;
        while(std::getline(plain,line)) {
          auto t=trim(line);
          require(t.empty()||t[0]=='#',"extra numeric/text rows outside extension elements");
        }
      }
      if(end==std::string::npos)break;
      at=end;continue;
    }
    if(text.compare(at,4,"<!--")==0) {
      auto end=text.find("-->",at+4);require(end!=std::string::npos,"truncated extension comment");at=end+3;continue;
    }
    std::size_t end=at+1;char quote=0;
    for(;end<text.size();++end) {
      char c=text[end];if(quote){if(c==quote)quote=0;}
      else if(c=='\''||c=='"')quote=c;else if(c=='>')break;
    }
    require(end<text.size(),"truncated extension element");
    auto tag=trim(text.substr(at+1,end-at-1));
    if(tag[0]=='/')--depth;else if(tag.back()!='/')++depth;
    require(depth>=0,"extension nesting underflow");at=end+1;
  }
  require(depth==0,"unclosed extension element");
}
double norm(const FourMomentum& p) { return std::hypot(std::hypot(p.px,p.py),p.pz); }
double theta(const FourMomentum& p) { return std::atan2(std::hypot(p.px,p.py),p.pz); }
double phi(const FourMomentum& p) { return std::atan2(p.py,p.px); }
double wrapped(double x) { return x-2*pi*std::floor((x+pi)/(2*pi)); }
std::string escaped(const std::string& in) {
  std::ostringstream out;
  for(unsigned char c:in) {
    if(c=='"'||c=='\\') out<<'\\'<<c;
    else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }
  return out.str();
}
void vector_json(std::ostream& o,const Vec3& v) { o<<'['<<v[0]<<','<<v[1]<<','<<v[2]<<']'; }
void momentum_json(std::ostream& o,const FourMomentum& p) {
  o<<'['<<p.px<<','<<p.py<<','<<p.pz<<','<<p.energy<<']';
}
using State=std::array<double,7>; // x,y,z,px,py,pz,integrated electric work
State derivative(const State& y,double t,double mass,double charge,const FieldProvider& provider) {
  const double energy=std::hypot(std::hypot(y[3],y[4]),std::hypot(y[5],mass));
  require(energy>0&&finite(energy),"zero or invalid trajectory energy");
  Vec3 beta{{y[3]/energy,y[4]/energy,y[5]/energy}};
  State d{{c_mm_ns*beta[0],c_mm_ns*beta[1],c_mm_ns*beta[2],0,0,0,0}};
  if(charge==0) return d;
  const auto f=provider.at(Vec3{{y[0],y[1],y[2]}},t);
  for(int i=0;i<3;++i) require(finite(f.electric_GV_per_mm[i])&&finite(f.cB_GV_per_mm[i]),"nonfinite field");
  for(int i=0;i<3;++i) {
    const int j=(i+1)%3,k=(i+2)%3;
    d[3+i]=charge*c_mm_ns*(f.electric_GV_per_mm[i]+beta[j]*f.cB_GV_per_mm[k]-beta[k]*f.cB_GV_per_mm[j]);
    d[6]+=charge*c_mm_ns*f.electric_GV_per_mm[i]*beta[i];
  }
  return d;
}
State plus(const State& a,const State& b,double scale) {
  State r;
  for(std::size_t i=0;i<r.size();++i) r[i]=a[i]+scale*b[i];
  return r;
}
void validate_init(const std::string& prefix) {
  // Prefix ends immediately before the first event. Complete only its XML
  // envelope and extract the actual direct-root init span from the scanner.
  const auto envelope=prefix+"</LesHouchesEvents>";
  const auto markup=validate_markup(envelope);
  require(markup.events.empty(),"event records cannot occur in the document prefix");
  const auto& init=markup.init;
  std::istringstream in(envelope.substr(init.data_begin,init.data_end-init.data_begin));
  int beam1,beam2,pdfg1,pdfg2,pdfs1,pdfs2,weight,nproc;
  double e1,e2;
  row(next_data(in),beam1,beam2,e1,e2,pdfg1,pdfg2,pdfs1,pdfs2,weight,nproc);
  require(finite(e1)&&finite(e2)&&e1>=0&&e2>=0&&beam1!=0&&beam2!=0&&nproc>=1&&nproc<=100000,"invalid init beam/process metadata");
  require(weight>=-4&&weight<=4&&weight!=0,"unsupported IDWTUP");
  for(int i=0;i<nproc;++i) {
    double xsec,error,xmax; int process;
    row(next_data(in),xsec,error,xmax,process);
    require(finite(xsec)&&finite(error)&&finite(xmax)&&error>=0,"invalid process table");
  }
  std::string extra,extensions;
  while(std::getline(in,extra)) {
    if(extensions.empty()&&(trim(extra).empty()||trim(extra)[0]=='#')) continue;
    extensions+=extra+'\n';
  }
  // Process table count is exact; remaining XML metadata is kept verbatim in
  // prefix, including initrwgt, generator and unknown well-formed extensions.
  // LHEF permits optional generator data after the required init table.
  // Preserve it byte-for-byte in prefix, including opaque text following XML.
  // A second numeric process row before the extensions is still rejected.
  require(trim(extensions).empty()||trim(extensions)[0]=='<',"extra process rows or unsupported init text");
}
} // namespace

void validate_event(const Event& e) {
  require(!e.id.empty(),"missing stable event ID");
  require(finite(e.weight)&&finite(e.scale)&&finite(e.alpha_qed)&&finite(e.alpha_qcd),"nonfinite event metadata");
  require(!e.particles.empty(),"empty event");
  std::set<std::string> ids; std::set<std::size_t> rows;
  std::size_t finals=0;
  for(const auto& p:e.particles) {
    require(!p.id.empty()&&ids.insert(p.id).second,"duplicate/missing particle ID");
    require(p.original_row>=1&&p.original_row<=e.particles.size()&&rows.insert(p.original_row).second,"invalid original row identity");
    require(p.pdg!=0&&p.status!=0&&finite(p.input)&&finite(p.output),"invalid particle data");
    require(p.input.energy>=0&&p.output.energy>=0&&finite(p.mass_GeV)&&p.mass_GeV>=0&&finite(p.lifetime_mm)&&p.lifetime_mm>=0&&finite(p.spin),"invalid mass/energy/lifetime/spin");
    for(auto m:p.mothers) require(m>=0&&std::size_t(m)<=e.particles.size(),"mother index outside event");
    for(auto c:p.colors) require(c>=0,"negative color tag");
    if(p.status==1) ++finals;
  }
  require(finals>=1,"event has no final-state particles");
}

LheDocument read_lhe(std::istream& input,const std::string& source) {
  require(!source.empty(),"source namespace required");
  std::ostringstream bytes; bytes<<input.rdbuf();
  require(!input.bad(),"input read failed");
  const std::string doc=bytes.str();
  const auto markup=validate_markup(doc);
  const auto& blocks=markup.events;
  require(!blocks.empty(),"no event blocks");
  LheDocument out; out.prefix=doc.substr(0,blocks.front().begin); validate_init(out.prefix);
  for(std::size_t index=0;index<blocks.size();++index) {
    const auto& block=blocks[index];
    std::istringstream in(doc.substr(block.data_begin,block.data_end-block.data_begin));
    Event e; e.file_event_index=out.events.size(); e.id=source+":"+std::to_string(e.file_event_index);
    e.opening_tag=doc.substr(block.begin,block.data_begin-block.begin);
    long long n;
    row(event_data(in),n,e.process_id,e.weight,e.scale,e.alpha_qed,e.alpha_qcd);
    require(n>=1&&n<=1000000,"NUP outside bounded profile");
    for(long long j=0;j<n;++j) {
      Particle p; p.original_row=std::size_t(j+1); p.id=e.id+":"+std::to_string(p.original_row);
      row(event_data(in),p.pdg,p.status,p.mothers[0],p.mothers[1],p.colors[0],p.colors[1],p.input.px,p.input.py,p.input.pz,p.input.energy,p.mass_GeV,p.lifetime_mm,p.spin);
      p.output=p.input; e.particles.push_back(p);
    }
    std::string line;
    while(std::getline(in,line)) e.extensions+=line+"\n";
    const auto ext=trim(e.extensions);
    require(ext.empty()||ext[0]=='<'||ext[0]=='#',"extra particle rows after NUP");
    // Unknown extensions are opaque and preserved; no physics interpretation.
    validate_extension_payload(e.extensions);
    validate_event(e); out.events.push_back(e);
    if(index+1<blocks.size())
      require(trim(doc.substr(block.end,blocks[index+1].begin-block.end)).empty(),"inter-event metadata unsupported");
  }
  out.suffix=doc.substr(blocks.back().end);
  require(trim(out.suffix)=="</LesHouchesEvents>","invalid/truncated document suffix");
  return out;
}

void write_lhe(std::ostream& out,const LheDocument& doc,bool transported) {
  // Sorting in memory must not corrupt mother references: emit original rows.
  std::set<std::string> ids;
  for(const auto& e:doc.events) { validate_event(e); require(ids.insert(e.id).second,"duplicate event ID"); }
  validate_init(doc.prefix);
  out<<doc.prefix<<std::setprecision(17);
  for(const auto& e:doc.events) {
    out<<e.opening_tag<<'\n'<<e.particles.size()<<' '<<e.process_id<<' '<<e.weight<<' '<<e.scale<<' '<<e.alpha_qed<<' '<<e.alpha_qcd<<'\n';
    std::vector<const Particle*> sorted;
    for(const auto& p:e.particles) sorted.push_back(&p);
    std::sort(sorted.begin(),sorted.end(),[](const Particle* a,const Particle* b){return a->original_row<b->original_row;});
    for(const auto* p:sorted) {
      const auto& q=(transported&&p->status==1)?p->output:p->input;
      out<<p->pdg<<' '<<p->status<<' '<<p->mothers[0]<<' '<<p->mothers[1]<<' '<<p->colors[0]<<' '<<p->colors[1]<<' '<<q.px<<' '<<q.py<<' '<<q.pz<<' '<<q.energy<<' '<<p->mass_GeV<<' '<<p->lifetime_mm<<' '<<p->spin<<'\n';
    }
    out<<e.extensions;
    if(transported) out<<"# FeeBL transported final states: external fields exchange four-momentum; see sidecar.\n";
    out<<"</event>\n";
  }
  out<<doc.suffix; require(bool(out),"output write failed");
}

Species species(int pdg) {
  switch(pdg) {
    case 11:return {-1,0.00051099895}; case -11:return {1,0.00051099895};
    case 13:return {-1,0.1056583755}; case -13:return {1,0.1056583755};
    case 22:case 12:case -12:case 14:case -14:case 16:case -16:return {0,0};
    default:throw std::runtime_error("grouped events: unsupported PDG "+std::to_string(pdg));
  }
}

std::vector<Track> transport(Event& event,const Vertex& vertex,const FieldProvider& field,const TransportPolicy& policy) {
  validate_event(event);
  require(vertex.method=="explicit_lab"||vertex.method=="luminosity_sample", "explicit physical vertex convention required");
  require(finite(vertex.time_ns)&&finite(policy.end_time_ns)&&policy.end_time_ns>=vertex.time_ns,"invalid time interval");
  for(double x:vertex.position_mm) require(finite(x),"nonfinite vertex");
  require(finite(policy.max_step_ns)&&policy.max_step_ns>0&&policy.max_steps>0,"positive bounded step required");
  require(policy.radiation=="none"&&policy.decay=="none","radiation/decay policy unsupported");
  require(!field.version().empty(),"field provenance required");
  const double duration=policy.end_time_ns-vertex.time_ns;
  const double requested=std::ceil(duration/policy.max_step_ns);
  require(finite(requested)&&requested<=double(policy.max_steps),"step budget exceeded");
  const auto steps=static_cast<std::size_t>(requested);
  const double h=steps?duration/double(steps):0;
  std::vector<Track> tracks;
  for(const auto& p:event.particles) {
    if(p.status!=1) continue;
    const auto s=species(p.pdg);
    require(std::abs(p.mass_GeV-s.mass_GeV)<=std::max(1e-12,s.mass_GeV*1e-5),"mass inconsistent with supported PDG");
    require(p.lifetime_mm==0,"nonzero decay lifetime requires unsupported policy");
    const double on_shell=std::hypot(norm(p.input),p.mass_GeV);
    require(p.input.energy>0&&std::abs(p.input.energy-on_shell)<=1e-9*std::max(1.,on_shell),"off-shell final state outside transport profile");
    Track tr; tr.event_id=event.id;tr.particle_id=p.id;tr.original_row=p.original_row;tr.pdg=p.pdg;
    tr.file_event_index=event.file_event_index;tr.process_id=event.process_id;tr.original_event_weight=event.weight;
    tr.input=p.input;tr.input_vertex=vertex;tr.field_version=field.version();tr.charge_e=s.charge_e;tr.mass_GeV=p.mass_GeV;tr.steps=steps;
    State y{{vertex.position_mm[0],vertex.position_mm[1],vertex.position_mm[2],p.input.px,p.input.py,p.input.pz,0}};
    for(std::size_t i=0;i<steps;++i) {
      const double t=vertex.time_ns+double(i)*h;
      auto k1=derivative(y,t,p.mass_GeV,s.charge_e,field);
      auto k2=derivative(plus(y,k1,h*.5),t+h*.5,p.mass_GeV,s.charge_e,field);
      auto k3=derivative(plus(y,k2,h*.5),t+h*.5,p.mass_GeV,s.charge_e,field);
      auto k4=derivative(plus(y,k3,h),t+h,p.mass_GeV,s.charge_e,field);
      for(std::size_t j=0;j<y.size();++j) { y[j]+=h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;require(finite(y[j]),"nonfinite trajectory"); }
    }
    tr.output={y[3],y[4],y[5],std::hypot(std::hypot(y[3],y[4]),std::hypot(y[5],p.mass_GeV))};
    // Neutrals preserve generator momentum/energy bit-for-bit.
    if(s.charge_e==0||steps==0) tr.output=p.input;
    tr.output_vertex=vertex;tr.output_vertex.position_mm={{y[0],y[1],y[2]}};tr.output_vertex.time_ns=policy.end_time_ns;
    tr.electric_work_GeV=y[6];tr.delta_theta=theta(tr.output)-theta(tr.input);tr.delta_phi=wrapped(phi(tr.output)-phi(tr.input));
    tr.outgoing_pz_sign=(tr.output.pz>0)-(tr.output.pz<0);
    tr.delta_theta_out=tr.outgoing_pz_sign<0?-tr.delta_theta:tr.delta_theta;
    tracks.push_back(tr);
  }
  for(auto& p:event.particles) if(p.status==1) {
    const auto it=std::find_if(tracks.begin(),tracks.end(),[&](const Track& t){return t.particle_id==p.id;});
    p.output=it->output;
  }
  return tracks;
}

void write_sidecar(std::ostream& out,const std::string& sha,const std::vector<Track>& tracks) {
  require(sha.size()==64&&std::all_of(sha.begin(),sha.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}),"sidecar requires lowercase SHA-256");
  std::set<std::pair<std::string,std::string>> ids;
  for(const auto& t:tracks) require(ids.insert({t.event_id,t.particle_id}).second,"duplicate sidecar particle identity");
  out<<std::setprecision(17)<<"{\"schema_version\":\"feebl.event_transport/1\",\"event_file_sha256\":\""<<sha<<"\",\"frame\":\"lab\",\"units\":{\"position\":\"mm\",\"time\":\"ns\",\"momentum\":\"GeV\",\"angle\":\"rad\",\"E\":\"GV/mm\",\"cB\":\"GV/mm\"},\"stepper\":\"fixed_rk4_qualification_v1\",\"live_bunch_integration\":false,\"particles\":[";
  bool first=true;
  for(const auto& t:tracks) {
    if(!first) out<<',';
    first=false;
    out<<"{\"event_id\":\""<<escaped(t.event_id)<<"\",\"file_event_index\":"<<t.file_event_index<<",\"process_id\":"<<t.process_id<<",\"original_event_weight\":"<<t.original_event_weight<<",\"particle_id\":\""<<escaped(t.particle_id)<<"\",\"original_row\":"<<t.original_row<<",\"PDG\":"<<t.pdg<<",\"status\":1,\"charge_e\":"<<t.charge_e<<",\"mass_GeV\":"<<t.mass_GeV<<",\"input_four_vector\":";momentum_json(out,t.input);
    out<<",\"output_four_vector\":";momentum_json(out,t.output);
    out<<",\"input_position\":";vector_json(out,t.input_vertex.position_mm);
    out<<",\"output_position\":";vector_json(out,t.output_vertex.position_mm);
    out<<",\"input_time\":"<<t.input_vertex.time_ns<<",\"output_time\":"<<t.output_vertex.time_ns<<",\"vertex_method\":\""<<escaped(t.input_vertex.method)<<"\",\"theta_lab_in\":"<<theta(t.input)<<",\"theta_lab_out\":"<<theta(t.output)<<",\"phi_in\":"<<phi(t.input)<<",\"phi_out\":"<<phi(t.output)<<",\"delta_theta\":"<<t.delta_theta<<",\"wrapped_delta_phi\":"<<t.delta_phi<<",\"delta_theta_out\":"<<t.delta_theta_out<<",\"delta_energy\":"<<t.output.energy-t.input.energy<<",\"electric_work\":"<<t.electric_work_GeV<<",\"outgoing_pz_sign\":"<<t.outgoing_pz_sign<<",\"observation_surface\":\""<<t.surface<<"\",\"stop_reason\":\""<<t.stop_reason<<"\",\"field_version\":\""<<escaped(t.field_version)<<"\",\"steps\":"<<t.steps<<'}';
  }
  out<<"]}\n";require(bool(out),"sidecar write failed");
}
} }
