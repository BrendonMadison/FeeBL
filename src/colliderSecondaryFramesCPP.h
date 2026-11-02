#ifndef FEEBL_COLLIDER_SECONDARY_FRAMES_CPP_H
#define FEEBL_COLLIDER_SECONDARY_FRAMES_CPP_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace feebl { namespace secondary_frames {
constexpr std::size_t joint_bins=128, marginal_bins=160, max_records=1000000;
constexpr std::size_t max_bytes=16*1024*1024;
constexpr double pi=3.141592653589793238462643383279502884;
inline bool finite(double v) {
  std::uint64_t bits;static_assert(sizeof(bits)==sizeof(v),"IEEE binary64 required");
  std::memcpy(&bits,&v,sizeof(bits));
  return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
inline std::vector<double> linear_edges(double hi,std::size_t n) {
  std::vector<double> out(n+1);
  for(std::size_t i=0;i<=n;++i)out[i]=hi*double(i)/double(n);
  out.front()=0;out.back()=hi;return out;
}
inline std::vector<double> log_edges(double lo,double hi,std::size_t n) {
  std::vector<double> out(n+1);const double a=std::log10(lo),b=std::log10(hi);
  for(std::size_t i=0;i<=n;++i)out[i]=std::pow(10.,a+(b-a)*double(i)/double(n));
  out.front()=lo;out.back()=hi;return out;
}
struct Axes {
  std::vector<double> theta=log_edges(1e-12,pi,joint_bins);
  std::vector<double> folded=log_edges(1e-12,0.5*pi,joint_bins);
  std::vector<double> pt=log_edges(1e-12,1e6,joint_bins);
  std::vector<double> marginal_theta=linear_edges(pi,marginal_bins);
  std::vector<double> marginal_folded=linear_edges(0.5*pi,marginal_bins);
  std::vector<double> marginal_pt;
  Axes():marginal_pt(log_edges(1e-12,1e6,marginal_bins-1)) {
    marginal_pt.insert(marginal_pt.begin(),0.0);
  }
};
inline const Axes& axes() {static const Axes value;return value;}
inline int bin(const std::vector<double>& edges,double value) {
  if(value<edges.front())return -1;
  if(value>edges.back())return int(edges.size()-1);
  return std::min(int(edges.size()-2),int(std::upper_bound(edges.begin(),edges.end(),value)-edges.begin())-1);
}
template<class T> inline void write_array(std::ostream& out,const std::vector<T>& values) {
  out << '[';for(std::size_t i=0;i<values.size();++i){if(i)out << ',';out << values[i];}out << ']';
}
inline void write_axes(std::ostream& out) {
  const Axes& a=axes();
  out << "{\"theta_edges_rad\":";write_array(out,a.theta);
  out << ",\"theta_folded_edges_rad\":";write_array(out,a.folded);
  out << ",\"pT_edges_GeV\":";write_array(out,a.pt);
  out << ",\"marginal_theta_edges_rad\":";write_array(out,a.marginal_theta);
  out << ",\"marginal_theta_folded_edges_rad\":";write_array(out,a.marginal_folded);
  out << ",\"marginal_pT_edges_GeV\":";write_array(out,a.marginal_pt);out << '}';
}
inline void write_position_z_definition(std::ostream& out) {
  out << "{\"coordinate\":\"stored native z position\",\"unit\":\"nm\","
      << "\"positive\":\"z > 0\",\"negative\":\"z < 0\","
      << "\"zero\":\"z == 0; included only in All\",\"direction_not_used\":true}";
}
struct Moments {
  std::size_t count=0;long double mean=0,m2=0;double minimum=0,maximum=0;
  void add(double value) {
    if(!finite(value))throw std::domain_error("non-finite secondary snapshot observable");
    if(!count)minimum=maximum=value;
    else {minimum=std::min(minimum,value);maximum=std::max(maximum,value);}
    ++count;const long double d=static_cast<long double>(value)-mean;
    mean+=d/count;m2+=d*(static_cast<long double>(value)-mean);
  }
};
struct Marginal {
  std::vector<std::size_t> counts=std::vector<std::size_t>(marginal_bins,0);
  std::size_t underflow=0,overflow=0;Moments moments;
  void add(double value,const std::vector<double>& edges) {
    moments.add(value);const int i=bin(edges,value);
    if(i<0)++underflow;else if(i==int(counts.size()))++overflow;else ++counts[i];
  }
  void write(std::ostream& out,const std::vector<double>& edges) const {
    out << "{\"counts\":";write_array(out,counts);
    out << ",\"samples\":" << moments.count << ",\"underflow\":" << underflow
        << ",\"overflow\":" << overflow << ",\"total_weight\":" << moments.count << ",\"statistics\":";
    if(!moments.count)out << "null";
    else {
      std::size_t mode=0;
      for(std::size_t i=1;i<counts.size();++i)
        if(double(counts[i])/(edges[i+1]-edges[i])>double(counts[mode])/(edges[mode+1]-edges[mode]))mode=i;
      const double mean=double(moments.mean),variance=std::max(0.0,double(moments.m2/moments.count));
      if(!finite(mean)||!finite(variance))throw std::domain_error("secondary snapshot moment overflow");
      out << "{\"mean\":" << mean << ",\"variance\":" << variance << ",\"min\":" << moments.minimum
          << ",\"max\":" << moments.maximum << ",\"mode\":";
      if(moments.count==underflow+overflow)
        out << "null,\"mode_method\":\"unavailable: no in-range marginal records\"}";
      else out << (edges[mode]+edges[mode+1])*0.5
               << ",\"mode_method\":\"centre of highest count-density histogram bin; approximate\"}";
    }out << '}';
  }
};
struct Joint {
  std::vector<std::size_t> counts=std::vector<std::size_t>(joint_bins*joint_bins,0);
  std::size_t samples=0,in_range=0,theta_zero=0,pt_zero=0,both_zero=0,nonpositive=0,underflow=0,overflow=0;
  void add(double theta,double pt,const std::vector<double>& x,const std::vector<double>& y) {
    ++samples;if(theta<=0)++theta_zero;if(pt<=0)++pt_zero;
    if(theta<=0&&pt<=0)++both_zero;
    if(theta<=0||pt<=0){++nonpositive;return;}
    const int ix=bin(x,theta),iy=bin(y,pt);
    if(ix<0||iy<0)++underflow;
    else if(ix==int(joint_bins)||iy==int(joint_bins))++overflow;
    else {++counts[std::size_t(iy)*joint_bins+ix];++in_range;}
  }
  void write(std::ostream& out,bool sparse=false) const {
    out << (sparse?"{\"counts_sparse\":":"{\"counts\":");
    if(sparse) {
      out << '[';bool first=true;
      for(std::size_t i=0;i<counts.size();++i)if(counts[i]) {
        if(!first)out << ',';
        first=false;
        out << '[' << i << ',' << counts[i] << ']';
      }
      out << ']';
    } else write_array(out,counts);
    out << ",\"samples\":" << samples << ",\"in_range_records\":" << in_range
        << ",\"excluded\":{\"theta_nonpositive\":" << theta_zero << ",\"pT_nonpositive\":" << pt_zero
        << ",\"both_nonpositive\":" << both_zero << ",\"nonpositive_records\":" << nonpositive
        << ",\"underflow_records\":" << underflow << ",\"overflow_records\":" << overflow << "}}";
  }
};
struct Snapshot {
  std::size_t nrecords=0,positive=0,negative=0;std::map<int,std::size_t> processes;
  Joint joint,joint_folded;Marginal theta,theta_folded,pt;
  template<class Particle> void add(const Particle& particle) {
    if(nrecords>=max_records)throw std::length_error("secondary snapshot particle bound exceeded");
    float vx,vy;particle.velocities(vx,vy);
    const double e=particle.signed_energy(),vz=particle.Zvelocity();
    if(!finite(e)||!finite(vx)||!finite(vy)||!finite(vz)||e==0)
      throw std::domain_error("invalid native secondary snapshot kinematics");
    const double transverse=std::hypot(double(vx),double(vy));
    if(transverse==0&&vz==0)throw std::domain_error("undefined secondary snapshot direction");
    const double t=std::atan2(transverse,vz),f=std::atan2(transverse,std::fabs(vz)),p=std::fabs(e)*transverse;
    if(!finite(t)||!finite(f)||!finite(p))throw std::domain_error("non-finite derived secondary snapshot kinematics");
    const Axes& a=axes();joint.add(t,p,a.theta,a.pt);joint_folded.add(f,p,a.folded,a.pt);
    theta.add(t,a.marginal_theta);theta_folded.add(f,a.marginal_folded);pt.add(p,a.marginal_pt);
    ++nrecords;++processes[particle.get_process()];if(e>0)++positive;else ++negative;
  }
  template<class Range> static Snapshot capture(const Range& particles,std::size_t admitted) {
    if(admitted>max_records)throw std::length_error("secondary snapshot particle bound exceeded");
    if(particles.size()!=admitted)throw std::logic_error("secondary snapshot requires completed tracking ownership");
    Snapshot result;for(const auto& particle:particles)result.add(particle);return result;
  }
  // Shared axes live in the file header. Only the two added position subsets
  // use sparse joints; the inclusive serialization retains its original shape.
  void write_population(std::ostream& out,bool subset=false) const {
    out << "\"nrecords\":" << nrecords << ",\"joint\":";
    joint.write(out,subset);out << ",\"joint_folded\":";joint_folded.write(out,subset);
    out << ",\"marginals\":{\"theta\":";theta.write(out,axes().marginal_theta);
    out << ",\"theta_folded\":";theta_folded.write(out,axes().marginal_folded);
    out << ",\"pT\":";pt.write(out,axes().marginal_pt);
    out << "},\"selection\":{\"process_counts\":{";
    bool first=true;for(const auto& item:processes){if(!first)out << ',';first=false;out << '"' << item.first << "\":" << item.second;}
    out << "},\"signed_energy_counts\":{\"positive_energy\":" << positive << ",\"negative_energy\":" << negative << '}';
    if(!subset)out << ",\"admitted_tracker_count\":" << nrecords << ",\"authoritative_reserve_count\":" << nrecords;
    out << '}';
  }
  void write_frame(std::ostream& out,int id,int diagonal,bool close_frame=true) const {
    out << "{\"id\":" << id << ",\"diagonal\":" << diagonal
        << ",\"available\":true,\"reason\":null,";
    write_population(out);
    if(close_frame)out << '}';
  }
};
struct PositionSnapshot {
  Snapshot all,positive,negative;
  std::size_t zero_records=0;
  template<class Range> static PositionSnapshot capture(const Range& particles,std::size_t admitted) {
    if(admitted>max_records)throw std::length_error("secondary snapshot particle bound exceeded");
    if(particles.size()!=admitted)throw std::logic_error("secondary snapshot requires completed tracking ownership");
    PositionSnapshot result;
    for(const auto& particle:particles) {
      const double z=particle.z();
      if(!finite(z))throw std::domain_error("non-finite stored secondary z position");
      // Preserve the inclusive traversal and arithmetic exactly. Classification
      // reads the current stored position, never longitudinal velocity, charge,
      // source beam, detector arm, or a final state from a different time.
      result.all.add(particle);
      if(z>0)result.positive.add(particle);
      else if(z<0)result.negative.add(particle);
      else ++result.zero_records;
    }
    return result;
  }
  void write_frame(std::ostream& out,int id,int diagonal) const {
    if(positive.nrecords+negative.nrecords+zero_records!=all.nrecords)
      throw std::logic_error("secondary position partition count does not close");
    all.write_frame(out,id,diagonal,false);
    out << ",\"position_z\":{\"zero_records\":" << zero_records << ",\"positive\":{";
    positive.write_population(out,true);
    out << "},\"negative\":{";negative.write_population(out,true);out << "}}}";
  }
};
} }
#endif
