#ifndef FEEBL_DESIGN_AXIS_SPECTRUM_CPP_H
#define FEEBL_DESIGN_AXIS_SPECTRUM_CPP_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <vector>

// Opt-in read-only reconstruction. Energies are interpreted as lab energies,
// and native directions as perturbations about each design beam axis. This is
// NOT a Lorentz transformation of the dynamics or an upstream optics model.
namespace feebl { namespace design_axis {
constexpr double electron_mass_GeV=0.00051099895;
struct Direction { double x,y,z; };
inline Direction rotate_y(Direction n,double a) {
  if(a==0) return n;
  const double c=std::cos(a),s=std::sin(a);
  return Direction{c*n.x+s*n.z,n.y,-s*n.x+c*n.z};
}
inline Direction direction(double x,double y,int sign) {
  const double q=x*x+y*y;
  if(!std::isfinite(q)||q>1||q<0||(sign!=1&&sign!=-1))
    throw std::domain_error("design-axis invalid native direction");
  return Direction{x,y,sign*std::sqrt(1-q)};
}
inline double mass(double e1,Direction n1,double e2,Direction n2,
                   double a1,double a2) {
  const double m=electron_mass_GeV;
  if(!std::isfinite(e1)||!std::isfinite(e2)||e1<m||e2<m||
     !std::isfinite(a1)||!std::isfinite(a2))
    throw std::domain_error("design-axis invalid energies or angles");
  n1=rotate_y(n1,a1);n2=rotate_y(n2,-a2);
  const double p1=std::sqrt((e1-m)*(e1+m));
  const double p2=std::sqrt((e2-m)*(e2+m));
  const double dot=n1.x*n2.x+n1.y*n2.y+n1.z*n2.z;
  const double w2=2*m*m+2*(e1*e2-p1*p2*dot);
  if(!std::isfinite(w2)||w2<0)
    throw std::domain_error("design-axis nonfinite/negative invariant mass");
  return std::sqrt(w2);
}
struct Bin {
  double low=0,high=0,weight=0,weight2=0;
  std::uint64_t contributions=0;
};
class Spectrum {
  bool enabled_=false;
  double e1_=0,e2_=0,a1_=0,a2_=0,nominal_=0,pl_=0,ph_=0,upper_=0;
  double under_=0,over_=0,sw_=0,sw2_=0,sd_=0,sd2_=0;
  std::uint64_t count_=0;
  std::size_t fine_=0;
  std::vector<Bin> bins_;
  std::vector<std::size_t> touched_;
  void segment(double lo,double hi,std::size_t n) {
    for(std::size_t i=0;i<n;++i){
      Bin b;b.low=lo+(hi-lo)*double(i)/double(n);
      b.high=(i+1==n)?hi:lo+(hi-lo)*double(i+1)/double(n);
      bins_.push_back(b);
    }
  }
 public:
  bool enabled()const{return enabled_;}
  double total_weight()const{return sw_;}
  double total_weight2()const{return sw2_;}
  std::uint64_t contributions()const{return count_;}
  double mean()const{return sw_>0?nominal_+sd_/sw_:0;}
  double rms()const{return sw_>0?std::sqrt(std::max(0.0,sd2_/sw_-(sd_/sw_)*(sd_/sw_))):0;}
  const std::vector<Bin>& bins()const{return bins_;}
  // Worker lanes retain bin geometry and clear only bins touched by the
  // previous encounter. This avoids an O(fine bins) clear per slice pair.
  void reset() {
    for (const auto i : touched_) {
      bins_[i].weight=0; bins_[i].weight2=0; bins_[i].contributions=0;
    }
    touched_.clear();
    under_=over_=sw_=sw2_=sd_=sd2_=0; count_=0;
  }
  void configure_like(const Spectrum& source) {
    if (source.enabled_) configure(source.e1_,source.e2_,source.a1_,source.a2_,
                                  static_cast<int>(source.fine_));
  }
  std::size_t storage_bytes()const {
    return bins_.capacity()*sizeof(Bin)+touched_.capacity()*sizeof(std::size_t);
  }
  // Stage only populated bins, then publish once every source passes its
  // configuration, finite-value and counter-overflow checks. The caller gives
  // lanes in canonical encounter order; no worker writes a shared histogram.
  bool merge_from_many(const std::vector<const Spectrum*>& sources) {
    std::map<std::size_t,Bin> changed;
    double under=under_,over=over_,sw=sw_,sw2=sw2_,sd=sd_,sd2=sd2_;
    std::uint64_t count=count_;
    const auto add_count=[](std::uint64_t& a,std::uint64_t b) {
      if (b>std::numeric_limits<std::uint64_t>::max()-a) return false;
      a+=b; return true;
    };
    const auto add_value=[](double& a,double b) {
      if (!std::isfinite(b)||!std::isfinite(a+b)) return false;
      a+=b; return true;
    };
    for (const auto source : sources) {
      if (!source || source==this || source->enabled_!=enabled_) return false;
      if (!enabled_) continue;
      if (e1_!=source->e1_ || e2_!=source->e2_ || a1_!=source->a1_ ||
          a2_!=source->a2_ || fine_!=source->fine_ ||
          bins_.size()!=source->bins_.size()) return false;
      if (!add_value(under,source->under_) || !add_value(over,source->over_) ||
          !add_value(sw,source->sw_) || !add_value(sw2,source->sw2_) ||
          !add_value(sd,source->sd_) || !add_value(sd2,source->sd2_) ||
          !add_count(count,source->count_)) return false;
      for (const auto i : source->touched_) {
        if (i>=bins_.size()) return false;
        auto item=changed.emplace(i,bins_[i]).first;
        Bin& target=item->second;
        const Bin& incoming=source->bins_[i];
        if (!add_value(target.weight,incoming.weight) ||
            !add_value(target.weight2,incoming.weight2) ||
            !add_count(target.contributions,incoming.contributions)) return false;
      }
    }
    for (const auto& item : changed) {
      if (bins_[item.first].contributions==0) touched_.push_back(item.first);
      bins_[item.first]=item.second;
    }
    under_=under; over_=over; sw_=sw; sw2_=sw2; sd_=sd; sd2_=sd2; count_=count;
    return true;
  }
  void configure(double e1,double e2,double a1,double a2,int fine) {
    if(enabled_||fine<100||fine>1000000||a1<0||a2<0||a1+a2>=.2)
      throw std::invalid_argument("design-axis spectrum configuration outside admitted bounds");
    nominal_=mass(e1,Direction{0,0,1},e2,Direction{0,0,-1},a1,a2);
    e1_=e1;e2_=e2;a1_=a1;a2_=a2;fine_=std::size_t(fine);
    pl_=.96*nominal_;ph_=1.04*nominal_;
    upper_=std::max(1.1*(e1+e2),1.1*nominal_);
    // Reserve the complete output histogram once. At 125 GeV, one million
    // central bins resolve 10 keV while retaining the unchanged full tails.
    bins_.reserve(2048+fine_+512);
    touched_.reserve(2048+fine_+512);
    segment(0,pl_,2048);segment(pl_,ph_,fine_);segment(ph_,upper_,512);
    enabled_=true;
  }
  void add(double weight,double e1,double x1,double y1,double e2,double x2,double y2) {
    if(!enabled_)return;
    if(!std::isfinite(weight)||weight<0)
      throw std::domain_error("design-axis invalid luminosity weight");
    if(weight==0)return;
    const double w=mass(e1,direction(x1,y1,1),e2,direction(x2,y2,-1),a1_,a2_);
    const double d=w-nominal_;
    sw_+=weight;sw2_+=weight*weight;sd_+=weight*d;sd2_+=weight*d*d;++count_;
    if(!std::isfinite(sw_)||!std::isfinite(sw2_)||!std::isfinite(sd_)||!std::isfinite(sd2_))
      throw std::overflow_error("design-axis moment overflow");
    if(w<0){under_+=weight;return;}
    if(w>=upper_){over_+=weight;return;}
    std::size_t i;
    if(w<pl_)i=std::min(std::size_t(2047),std::size_t(w/pl_*2048));
    else if(w<ph_)i=2048+std::min(fine_-1,std::size_t((w-pl_)/(ph_-pl_)*fine_));
    else i=2048+fine_+std::min(std::size_t(511),std::size_t((w-ph_)/(upper_-ph_)*512));
    if(bins_[i].contributions==0)touched_.push_back(i);
    bins_[i].weight+=weight;bins_[i].weight2+=weight*weight;++bins_[i].contributions;
  }
  void write(std::ostream& out,const char* source_hash,int seed)const {
    if(!enabled_)throw std::logic_error("disabled design-axis spectrum write");
    const std::streamsize precision=out.precision();out<<std::setprecision(17);
    out<<"#FEEBL-DESIGN-AXIS 1.0\n"
       <<"#source_revision "<<source_hash<<"\n#rndm_seed "<<seed
       <<"\n#convention input_energies_as_lab;native_directions_about_design_beam_axes;Ry(+a1),Ry(-a2)\n"
       <<"#scope output_only_kinematic_reconstruction;not_Lorentz_transported_dynamics\n"
       <<"#ISR pre_optional_ISR\n#energy_unit GeV\n#weight_unit m^-2_per_BX\n"
       <<"#weight2_unit m^-4_per_BX2\n#contributions_unit correlated_collision_pair_evaluations\n"
       <<"#sampling all_positive_weight_ee_collision_pairs;no_event_heap_sampling\n"
       <<"#energy1_GeV "<<e1_<<"\n#energy2_GeV "<<e2_
       <<"\n#angle1_x_rad "<<a1_<<"\n#angle2_x_rad "<<a2_
       <<"\n#electron_mass_GeV "<<electron_mass_GeV<<"\n#nominal_W_GeV "<<nominal_
       <<"\n#peak_bins "<<fine_<<"\n#all_bins_written yes\n#bins "<<bins_.size()
       <<"\n#underflow "<<under_<<"\n#overflow "<<over_
       <<"\n#total_weight "<<sw_<<"\n#total_weight2 "<<sw2_
       <<"\n#mean_W "<<mean()<<"\n#rms_W "<<rms()
       <<"\n#centered_reference_W "<<nominal_<<"\n#sum_weight_delta_W "<<sd_
       <<"\n#sum_weight_delta_W2 "<<sd2_<<"\n#contributions "<<count_
       <<"\n#COLUMNS W_low W_high weight weight2 contributions\n";
    for(const auto& b:bins_)out<<b.low<<' '<<b.high<<' '<<b.weight<<' '<<b.weight2<<' '<<b.contributions<<'\n';
    out<<"#END design_axis_pair_spectrum records "<<bins_.size()<<'\n';
    out.precision(precision);
  }
};
}}
#endif
