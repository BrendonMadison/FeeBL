#ifndef FEEBL_COLLIDER_PEAK_DIAGNOSTICS_CPP_H
#define FEEBL_COLLIDER_PEAK_DIAGNOSTICS_CPP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>

// Optional read-only observables. No particle, field, random stream, native
// spectrum, or physics decision is modified by this accumulator.
namespace feebl { namespace peak_diagnostics {
inline bool finite(double value) {
  std::uint64_t bits; std::memcpy(&bits,&value,sizeof bits);
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
constexpr double reference_mass_GeV=0.00051099895;
constexpr double threshold_fraction=0.98;
constexpr std::size_t histogram_bins=512;

struct Definition {
  bool available=false;
  std::string reason="Peak diagnostics were not configured";
  std::string angle_source;
  double energy1=0,energy2=0,full_angle=0,nominal=0,minimum=0,upper=0;
  static Definition make(double e1,double e2,double ax1,double ax2,
                         double ay1,double ay2,double declared_angle,
                         double spectrum_upper) {
    Definition d;
    for(double x:{e1,e2,ax1,ax2,ay1,ay2,declared_angle,spectrum_upper})
      if(!finite(x)){d.reason="Non-finite nominal reference input";return d;}
    if(e1<=reference_mass_GeV || e2<=reference_mass_GeV) {
      d.reason="Nominal reference energies must exceed the electron rest mass";return d;
    }
    const double receipt=std::abs(ax1)+std::abs(ax2)+std::abs(ay1)+std::abs(ay2);
    const bool both_planes=(ax1!=0 || ax2!=0) && (ay1!=0 || ay2!=0);
    if(both_planes || ax1*ax2<0 || ay1*ay2<0) {
      d.reason="Ambiguous design crossing reference: mixed planes or opposite beam-angle signs";return d;
    }
    if(receipt==0 && declared_angle==0) {
      d.full_angle=0;d.angle_source="evaluated head-on beam angles";
    } else if(receipt>0 && declared_angle>0 && declared_angle<3.14159265358979323846 &&
              std::abs(declared_angle-receipt)<=1e-6*(1+receipt)) {
      d.full_angle=declared_angle;
      d.angle_source="evaluated crab_full_crossing_angle matched to beam-angle receipt";
    } else {
      d.reason="No unambiguous evaluated design crossing angle matching the beam-angle receipt";return d;
    }
    d.energy1=e1;d.energy2=e2;
    const double p1=std::sqrt((e1-reference_mass_GeV)*(e1+reference_mass_GeV));
    const double p2=std::sqrt((e2-reference_mass_GeV)*(e2+reference_mass_GeV));
    d.nominal=std::sqrt(2*reference_mass_GeV*reference_mass_GeV+
                        2*(e1*e2+p1*p2*std::cos(d.full_angle)));
    d.minimum=threshold_fraction*d.nominal;
    d.upper=std::max(spectrum_upper,1.02*d.nominal);
    if(!finite(d.nominal) || !finite(d.upper) || !(d.upper>d.minimum)) {
      d.reason="Invalid design nominal invariant mass or peak histogram geometry";return d;
    }
    d.available=true;d.reason.clear();return d;
  }
  bool same(const Definition& o) const {
    return available==o.available && reason==o.reason && angle_source==o.angle_source &&
      energy1==o.energy1 && energy2==o.energy2 && full_angle==o.full_angle &&
      nominal==o.nominal && minimum==o.minimum && upper==o.upper;
  }
  void write(std::ostream& out) const {
    if(!available){out << "null";return;}
    out << "{\"threshold_fraction\":0.98,\"nominal_W_GeV\":" << nominal
        << ",\"minimum_W_GeV\":" << minimum
        << ",\"upper_cut_GeV\":null,\"comparison\":\"W >= minimum_W_GeV\","
        << "\"nominal_reference\":{\"energy_GeV\":[" << energy1 << ',' << energy2
        << "],\"full_crossing_angle_rad\":" << full_angle
        << ",\"electron_mass_GeV\":" << reference_mass_GeV
        << ",\"angle_source\":\"" << angle_source
        << "\",\"definition\":\"sqrt(2*m_e^2 + 2*(E1*E2 + p1*p2*cos(full_crossing_angle)))\","
        << "\"native_W\":\"Existing massive invariant mass of native simulated incoming directions; no design crossing rotation; beam energies before optional ISR sampling\"}}";
  }
};

// Moments about the nominal reference avoid catastrophic cancellation when a
// narrow energy core sits near 125 GeV. This is a population RMS, not an error.
struct ShiftedMoment {
  double sum=0,square=0;
  void add(double weight,double value,double reference) {
    const double delta=value-reference;sum+=weight*delta;square+=weight*delta*delta;
  }
  void merge(const ShiftedMoment& other){sum+=other.sum;square+=other.square;}
  bool valid() const{return finite(sum)&&finite(square);}
  double mean(double weight,double reference)const{return reference+sum/weight;}
  double rms(double weight)const{return std::sqrt(std::max(0.0,square/weight-(sum/weight)*(sum/weight)));}
  void write(std::ostream& out,double weight,double reference)const {
    if(!(weight>0)){out << "null";return;}
    out << "{\"mean\":" << mean(weight,reference) << ",\"rms\":" << rms(weight) << '}';
  }
};

struct Accumulator {
  Definition definition;
  double weight=0,spin_weight=0,h1_sum=0,h2_sum=0,overflow=0;
  std::uint64_t contributions=0,spin_contributions=0;
  ShiftedMoment energy1,energy2,W;
  std::array<double,histogram_bins> bins{};
  Accumulator()=default;
  explicit Accumulator(const Definition& d):definition(d){}
  void add(double w,double mass,double e1,double e2,bool spin=false,double h1=0,double h2=0) {
    if(!definition.available)return;
    if(!finite(w)||w<0||!finite(mass)||mass<0||!finite(e1)||!finite(e2) ||
       (spin&&(!finite(h1)||!finite(h2))))
      throw std::domain_error("Invalid native peak diagnostic contribution");
    if(w==0 || mass<definition.minimum)return;
    ++contributions;weight+=w;
    energy1.add(w,std::abs(e1),definition.energy1);
    energy2.add(w,std::abs(e2),definition.energy2);
    W.add(w,mass,definition.nominal);
    if(spin){++spin_contributions;spin_weight+=w;h1_sum+=w*h1;h2_sum+=w*h2;}
    if(mass>=definition.upper)overflow+=w;
    else {
      const std::size_t index=std::min(histogram_bins-1,std::size_t(
        (mass-definition.minimum)/(definition.upper-definition.minimum)*histogram_bins));
      bins[index]+=w;
    }
  }
  bool valid()const {
    if(!finite(weight)||!finite(spin_weight)||!finite(h1_sum)||!finite(h2_sum)||
       !finite(overflow)||!energy1.valid()||!energy2.valid()||!W.valid())return false;
    for(double value:bins)if(!finite(value))return false;
    return true;
  }
  bool merge(const Accumulator& other) {
    if(!definition.same(other.definition))return false;
    weight+=other.weight;spin_weight+=other.spin_weight;
    h1_sum+=other.h1_sum;h2_sum+=other.h2_sum;overflow+=other.overflow;
    contributions+=other.contributions;spin_contributions+=other.spin_contributions;
    energy1.merge(other.energy1);energy2.merge(other.energy2);W.merge(other.W);
    for(std::size_t i=0;i<histogram_bins;++i)bins[i]+=other.bins[i];
    return valid();
  }
  void write(std::ostream& out,bool coverage)const {
    if(!valid())throw std::domain_error("Non-finite native peak diagnostic aggregate");
    const bool available=definition.available&&coverage;
    out << "{\"available\":" << (available?"true":"false") << ",\"reason\":";
    if(available)out << "null";
    else out << '"' << (definition.available?
      "Actual native collision contributions are not instrumented for this execution mode":definition.reason) << '"';
    out << ",\"selection\":";definition.write(out);
    out << ",\"weight_sum\":" << (available?weight:0)
        << ",\"contributions\":" << (available?contributions:0)
        << ",\"spin_weight_sum\":" << (available?spin_weight:0)
        << ",\"spin_contributions\":" << (available?spin_contributions:0)
        << ",\"beam_energy_GeV\":";
    if(available&&weight>0){out << "{\"beam1\":";energy1.write(out,weight,definition.energy1);
      out << ",\"beam2\":";energy2.write(out,weight,definition.energy2);out << '}';}
    else out << "null";
    out << ",\"W_GeV\":";
    if(available)W.write(out,weight,definition.nominal);else out << "null";
    const bool polarized=available&&weight>0&&spin_contributions==contributions;
    out << ",\"polarization\":";
    if(polarized)out << "{\"weighting\":\"native collision pair luminosity\",\"weight_sum\":"
      << weight << ",\"beam1\":" << h1_sum/weight << ",\"beam2\":" << h2_sum/weight << '}';
    else out << "null";
    out << ",\"polarization_unavailable_reason\":";
    if(polarized)out << "null";
    else out << '"' << (!available?"Peak collision contribution coverage is unavailable":
      (weight==0?"No positive luminosity weight passes the peak selection":
      "Valid stored helicities do not cover every selected luminosity contribution")) << '"';
    out << ",\"luminosity_spectrum\":";
    if(!available){out << "null}";return;}
    out << "{\"quantity\":\"W\",\"unit\":\"GeV\",\"weight_unit\":\"m^-2 per BX\","
        << "\"weighting\":\"native collision pair luminosity\",\"normalization\":\"bin-integrated\","
        << "\"samples\":" << contributions << ",\"bins\":[";
    for(std::size_t i=0;i<histogram_bins;++i){
      if(i)out << ',';
      const double low=definition.minimum+(definition.upper-definition.minimum)*double(i)/histogram_bins;
      const double high=i+1==histogram_bins?definition.upper:
        definition.minimum+(definition.upper-definition.minimum)*double(i+1)/histogram_bins;
      out << "{\"low\":" << low << ",\"high\":" << high << ",\"weight\":" << bins[i] << '}';
    }
    out << "],\"underflow\":0,\"overflow\":" << overflow << ",\"total_weight\":" << weight << "}}";
  }
};
} }
#endif
