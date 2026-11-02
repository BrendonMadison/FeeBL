#ifndef FEEBL_COLLIDER_IMPORTED_TRAJECTORIES_CPP_H
#define FEEBL_COLLIDER_IMPORTED_TRAJECTORIES_CPP_H
#include "particlesCPP.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace feebl { namespace imported_trajectories {
constexpr std::size_t max_records=8192;
constexpr std::size_t max_bytes=32u*1024u*1024u;
struct Initial {
  int label,scheduled,admitted;
  std::size_t source_record;
  float energy,x,y,z,vx,vy,vz;
};
// Diagnostic identity is attached exactly when an input row is admitted. Native
// particle labels survive copying/reordering through active and reserve pools.
// Exhausting this diagnostic bound never changes the physical population.
class Registry {
  std::vector<Initial> initial_;
  bool overflow_=false;
public:
  void add(const PAIR_PARTICLE& p,std::size_t row,int scheduled,int admitted) {
    if(overflow_)return;
    if(initial_.size()==max_records){overflow_=true;initial_.clear();return;}
    float x,y,vx,vy;p.XYposition(x,y);p.velocities(vx,vy);
    initial_.push_back({p.get_label(),scheduled,admitted,row,p.signed_energy(),x,y,p.z(),vx,vy,p.Zvelocity()});
  }
  bool overflow() const{return overflow_;}
  const std::vector<Initial>& initial() const{return initial_;}
  void write_initial(std::ostream& out) const {
    out << '[';bool first=true;
    for(const auto& p:initial_){if(!first)out << ',';first=false;
      out << '[' << p.label << ',' << p.source_record << ',' << p.scheduled << ',' << p.admitted
          << ',' << p.energy << ',' << p.x << ',' << p.y << ',' << p.z
          << ',' << p.vx << ',' << p.vy << ',' << p.vz << ']';}
    out << ']';
  }
  template<class Range>void write_current(std::ostream& out,const Range& particles) const {
    if(overflow_)throw std::logic_error("imported identity registry exceeded diagnostic bound");
    std::map<int,std::array<double,6>> current;
    for(const auto& p:particles){float vx,vy;p.velocities(vx,vy);const float vz=p.Zvelocity();
      if(!std::isfinite(vx)||!std::isfinite(vy)||!std::isfinite(vz)||
         (vx==0 && vy==0 && vz==0))throw std::logic_error("invalid tracked imported velocity");
      if(!current.emplace(p.get_label(),std::array<double,6>{{vx,vy,vz,
         double(p.imported_field_gathers(0)),double(p.imported_field_gathers(1)),double(p.imported_field_gathers(2))}}).second)
        throw std::logic_error("duplicate imported tracker label");
    }
    if(current.size()!=initial_.size())throw std::logic_error("imported tracker identity population does not close");
    for(const auto& p:initial_)if(current.count(p.label)!=1)
      throw std::logic_error("imported tracker identity missing from current reserve");
    out << '[';bool first=true;for(const auto& p:current){if(!first)out << ',';first=false;
      out << '[' << p.first;for(double value:p.second)out << ',' << value;out << ']';}
    out << ']';
  }
};
} }
#endif
