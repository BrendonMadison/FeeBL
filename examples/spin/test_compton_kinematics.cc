// Independent checks of head-on Compton scattering in GeV and metres.
// The reference uses the electron rest-frame Klein--Nishina angular law
// and a Lorentz boost, rather than the production light-cone construction.
#include "comptonKinematicsCPP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
using Vector = std::array<double, 3>;
using Incident = feebl::compton::HeadOnIncident;
using State = feebl::compton::FinalState;
constexpr double epsilon = std::numeric_limits<double>::epsilon();
int failures = 0;

void check(bool valid, const char* label, double error = 0.0) {
  std::printf("%s: %s (%.12g)\n", label, valid ? "PASS" : "FAIL", error);
  failures += !valid;
}

long double dot(const Vector& first, const Vector& second) {
  long double result = 0.0L;
  for (int axis = 0; axis < 3; ++axis)
    result += static_cast<long double>(first[axis])*second[axis];
  return result;
}

double magnitude(const Vector& vector) {
  return std::sqrt(static_cast<double>(dot(vector, vector)));
}

// d sigma / d r divided by 2 pi r_e^2, with r=(1-cos Theta*)/2.
// This is the rest-frame angular Klein--Nishina law, including its energy
// Jacobian implicitly through epsilon=1/(1+x*r).
long double angular_density(long double x, long double r) {
  const long double ratio = 1.0L/(1.0L+x*r);
  const long double sine2 = 4.0L*r*(1.0L-r);
  return ratio*ratio*(ratio+1.0L/ratio-sine2);
}

// Sixteen-point Gauss--Legendre quadrature. Geometrically spaced panels
// resolve the forward peak of width 1/x without using the production CDF.
long double angular_integral(long double x, long double upper) {
  constexpr long double nodes[] = {
    .095012509837637440185L, .281603550779258913230L,
    .458016777657227386342L, .617876244402643748447L,
    .755404408355003033895L, .865631202387831743880L,
    .944575023073232576078L, .989400934991649932596L};
  constexpr long double weights[] = {
    .189450610455068496285L, .182603415044923588867L,
    .169156519395002538189L, .149595988816576732081L,
    .124628971255533872052L, .095158511682492784810L,
    .062253523938647892863L, .027152459411754094852L};
  if (upper == 0.0L) return 0.0L;
  long double lower = 0.0L;
  long double end = std::min(upper, 1.0L/std::max(1.0L, x));
  long double total = 0.0L;
  while (lower < upper) {
    const long double midpoint = (lower+end)/2.0L;
    const long double half_width = (end-lower)/2.0L;
    for (int node = 0; node < 8; ++node) {
      const long double offset = half_width*nodes[node];
      total += half_width*weights[node]*(angular_density(x, midpoint-offset)
                                        +angular_density(x, midpoint+offset));
    }
    lower = end;
    end = std::min(upper, 2.0L*end);
  }
  return total;
}

struct RestFrameReference {
  long double photon_energy;
  long double photon_parallel;
  long double electron_energy;
  long double electron_parallel;
  long double transverse_momentum;
};

// Scatter the photon through Theta* in the electron rest frame and actively
// boost both final four-vectors along the original electron direction.
RestFrameReference rest_frame_reference(const Incident& in, double t) {
  const long double mass = in.mass_GeV;
  const long double energy = in.energy_GeV;
  const long double photon = in.photon_GeV;
  const long double momentum = std::sqrt((energy-mass)*(energy+mass));
  const long double gamma = energy/mass;
  const long double beta = momentum/energy;
  const long double incident_rest = gamma*photon*(1.0L+beta);
  const long double x = 2.0L*incident_rest/mass;
  // At the exact production endpoint the angle is analytically pi.
  const long double r = t == std::log1p(in.x)
      ? 1.0L : std::expm1(static_cast<long double>(t))/x;
  const long double cosine = 1.0L-2.0L*r;
  const long double sine = 2.0L*std::sqrt(std::max(0.0L, r*(1.0L-r)));
  const long double outgoing_rest = incident_rest/(1.0L+x*r);
  const long double photon_z_rest = -outgoing_rest*cosine;
  const long double electron_energy_rest = mass+incident_rest-outgoing_rest;
  const long double electron_z_rest = -incident_rest-photon_z_rest;
  return {
    gamma*(outgoing_rest+beta*photon_z_rest),
    gamma*(photon_z_rest+beta*outgoing_rest),
    gamma*(electron_energy_rest+beta*electron_z_rest),
    gamma*(electron_z_rest+beta*electron_energy_rest),
    outgoing_rest*sine};
}

// Scale shell errors in energy, avoiding an E^2-p^2 subtraction that loses
// the mass at high gamma. Conservation retains the original world axes.
double conservation_error(const Incident& in, const State& out) {
  const double scale = in.energy_GeV+in.photon_GeV;
  for (int axis = 0; axis < 3; ++axis)
    if (!std::isfinite(out.electron_momentum_GeV[axis])
        || !std::isfinite(out.photon_momentum_GeV[axis]))
      return std::numeric_limits<double>::infinity();
  if (!std::isfinite(out.electron_angle_rad) || !std::isfinite(out.photon_angle_rad))
    return std::numeric_limits<double>::infinity();
  double error = std::abs(out.electron_energy_GeV+out.photon_energy_GeV-scale);
  for (int axis = 0; axis < 3; ++axis) {
    const double initial = (in.momentum_GeV-in.photon_GeV)*in.axis[axis];
    error = std::max(error, std::abs(out.electron_momentum_GeV[axis]
                                  +out.photon_momentum_GeV[axis]-initial));
  }
  error = std::max(error, std::abs(out.photon_energy_GeV
                                  -magnitude(out.photon_momentum_GeV)));
  error = std::max(error, std::abs(out.electron_energy_GeV
      -std::hypot(in.mass_GeV, magnitude(out.electron_momentum_GeV))));
  if (!std::isfinite(error) || !std::isfinite(out.photon_energy_GeV)
      || !std::isfinite(out.electron_energy_GeV)
      || out.photon_energy_GeV < 0.0
      || out.electron_energy_GeV < in.mass_GeV*(1.0-32.0*epsilon))
    return std::numeric_limits<double>::infinity();
  return error/scale;
}

void check_kinematics() {
  const double mass = EMASS;
  const double energies[] = {mass, std::nextafter(mass, 1.0),
                             1.001*mass, 2.0*mass, 0.1, 125.0};
  const double photon_fractions[] = {1e-9, 1e-3, 0.1, 1.0, 3.0};
  const std::array<double, 2> slopes[] = {{{0.0, 0.0}}, {{0.2, -0.3}},
                                         {{0.95, 0.1}}};
  const double fractions[] = {0.0, 1e-9, 0.17, 0.5, 0.93, 1.0-1e-9, 1.0};
  double worst_conservation = 0.0;
  double worst_reference = 0.0;
  double worst_angle = 0.0;
  double worst_pt_error = 0.0;
  std::array<double, 10> worst_pt_case{};
  for (double energy : energies) for (double fraction : photon_fractions) {
    for (int direction : {-1, 1}) for (const auto& slope : slopes) {
      const auto in = feebl::compton::incident(energy, energy*fraction,
                                              slope[0], slope[1], direction);
      const double scale = energy+in.photon_GeV;
      for (double position : fractions) {
        const double t = position*std::log1p(in.x);
        const auto out = feebl::compton::final_state(in, t, 0.371);
        const auto reference = rest_frame_reference(in, t);
        worst_conservation = std::max(worst_conservation, conservation_error(in, out));
        const long double electron_parallel = dot(out.electron_momentum_GeV, in.axis);
        const long double photon_parallel = dot(out.photon_momentum_GeV, in.axis);
        // The reference boost may cancel at high gamma; its rounding budget
        // is explicitly scaled by gamma^2 times long-double machine epsilon.
        const double gamma = energy/mass;
        const double reference_budget = 128.0*epsilon
            +64.0*std::numeric_limits<long double>::epsilon()*gamma*gamma;
        const long double errors[] = {
          out.electron_energy_GeV-reference.electron_energy,
          out.photon_energy_GeV-reference.photon_energy,
          electron_parallel-reference.electron_parallel,
          photon_parallel-reference.photon_parallel};
        for (long double error : errors)
          worst_reference = std::max(worst_reference,
              static_cast<double>(std::abs(error))/(scale*reference_budget));
        Vector transverse = out.electron_momentum_GeV;
        for (int axis = 0; axis < 3; ++axis)
          transverse[axis] -= static_cast<double>(electron_parallel)*in.axis[axis];
        const double pt = magnitude(transverse);
        // Near the backward-photon endpoint, pT is a square root of the
        // distance to the rounded log1p(x) boundary. Its forward error is
        // ill-conditioned there. Compare the smooth pT^2 invariant with
        // the same rounding budget; keep energy and longitudinal tests.
        const double pt_error = static_cast<double>(std::abs(
            static_cast<long double>(pt)*pt
            -reference.transverse_momentum*reference.transverse_momentum))
                /(scale*scale*reference_budget);
        if (pt_error > worst_pt_error) {
          worst_pt_error = pt_error;
          worst_pt_case = {{energy, in.photon_GeV, in.x, t, position,
              static_cast<double>(direction), slope[0], slope[1], pt,
              static_cast<double>(reference.transverse_momentum)}};
        }
        worst_reference = std::max(worst_reference,
            pt_error);
        // Direction is undefined at exactly zero recoil momentum; compare
        // angles only where a finite transverse projection is resolved.
        if (magnitude(out.electron_momentum_GeV) > 1e-12*scale) {
          const double angle = std::atan2(pt, static_cast<double>(electron_parallel));
          worst_angle = std::max(worst_angle, std::abs(angle-out.electron_angle_rad));
        }
      }
    }
  }
  check(worst_conservation < 128.0*epsilon,
        "Four-momentum and mass shells across energy, direction and axis", worst_conservation);
  check(worst_reference < 1.0,
        "Final states agree with independent rest-frame Lorentz boost", worst_reference);
  if (worst_pt_error >= 1.0) {
    std::printf("Worst transverse oracle case: E=%.17g k=%.17g x=%.17g "
        "t=%.17g position=%.17g dir=%.17g vx=%.17g vy=%.17g "
        "pt=%.17g reference_pt=%.17g\n", worst_pt_case[0], worst_pt_case[1],
        worst_pt_case[2], worst_pt_case[3], worst_pt_case[4], worst_pt_case[5],
        worst_pt_case[6], worst_pt_case[7], worst_pt_case[8], worst_pt_case[9]);
  }
  check(worst_angle < 2e-6, "Recoil angle includes backward and oblique tracks", worst_angle);
}

void check_cross_section_and_sampling() {
  const double invariants[] = {1e-18, 1e-12, 1e-6, 0.01, 0.1,
                               1.0, 10.0, 1e3, 1e6, 1e12};
  const double quantiles[] = {0.0, 1e-9, 0.01, 0.1, 0.5, 0.9, 0.99,
                              1.0-1e-9, 1.0};
  double worst_integral = 0.0;
  double worst_cdf = 0.0;
  bool monotone = true;
  for (double x : invariants) {
    const long double angular_total = angular_integral(x, 1.0L);
    const long double reference = 2.0L*PI*RE*RE*angular_total;
    worst_integral = std::max(worst_integral, static_cast<double>(
        std::abs(feebl::compton::cross_section_m2(x)/reference-1.0L)));
    double previous = -1.0;
    for (double uniform : quantiles) {
      const double t = feebl::compton::sample_log_recoil(x, uniform);
      monotone = monotone && t >= previous && t >= 0.0 && t <= std::log1p(x);
      previous = t;
      const long double r = uniform == 1.0 ? 1.0L
          : std::expm1(static_cast<long double>(t))/x;
      const long double cdf = angular_integral(x, r)/angular_total;
      worst_cdf = std::max(worst_cdf, static_cast<double>(std::abs(cdf-uniform)));
      // Compare the public partial integral too; inverse-CDF self-consistency
      // alone would miss a shared mistake in the integral and its inversion.
      const long double partial = feebl::compton::integral_log_recoil(x, t)/x;
      worst_integral = std::max(worst_integral,
          static_cast<double>(std::abs(partial-angular_integral(x, r))/angular_total));
    }
  }
  check(monotone, "Inverse CDF is ordered and includes both exact endpoints");
  check(worst_integral < 3e-12,
        "Total and partial cross sections match independent KN angular quadrature",
        worst_integral);
  check(worst_cdf < 3e-12, "Inverse CDF matches independent angular probabilities",
        worst_cdf);

  const double thomson = (8.0/3.0)*PI*RE*RE;
  const double soft_error = std::abs(feebl::compton::cross_section_m2(1e-12)/thomson-1.0);
  check(soft_error < 2e-11, "Soft-photon total approaches the Thomson cross section", soft_error);
  const double hard_x = 1e12;
  const double hard_limit = 2.0*PI*RE*RE*(std::log(hard_x)+0.5)/hard_x;
  const double hard_error = std::abs(feebl::compton::cross_section_m2(hard_x)/hard_limit-1.0);
  check(hard_error < 1e-9, "Hard-photon total approaches the KN asymptotic limit", hard_error);
}

void check_special_states() {
  const double mass = EMASS;
  const auto beam1=feebl::compton::incident_for_native_branch(125,0.01,0.2,-0.3,2).kinematics;
  const auto beam2=feebl::compton::incident_for_native_branch(125,0.01,0.2,-0.3,1).kinematics;
  check(beam1.direction==1 && beam2.direction==-1 &&
        beam1.axis[0]==beam2.axis[0] && beam1.axis[1]==beam2.axis[1] &&
        beam1.axis[2]==-beam2.axis[2],
        "Native ge branch targets beam2 (-z); eg branch targets beam1 (+z)");
  for (int branch:{1,2}) {
    const auto normal=feebl::compton::incident_for_native_branch(125,0.01,0.2,-0.3,branch);
    const auto conjugate=feebl::compton::incident_for_native_branch(-125,0.01,0.2,-0.3,branch);
    const double t=0.7*std::log1p(normal.kinematics.x);
    const auto first=feebl::compton::final_state(normal.kinematics,t,0.3);
    const auto second=feebl::compton::final_state(conjugate.kinematics,t,0.3);
    double difference=std::abs(first.electron_energy_GeV-second.electron_energy_GeV);
    difference=std::max(difference,std::abs(first.photon_energy_GeV-second.photon_energy_GeV));
    for (int axis=0;axis<3;++axis) {
      difference=std::max(difference,std::abs(first.electron_momentum_GeV[axis]-second.electron_momentum_GeV[axis]));
      difference=std::max(difference,std::abs(first.photon_momentum_GeV[axis]-second.photon_momentum_GeV[axis]));
    }
    check(normal.charge_sign==-conjugate.charge_sign &&
          normal.charge_sign==normal.kinematics.direction &&
          normal.kinematics.axis==conjugate.kinematics.axis &&
          normal.kinematics.x==conjugate.kinematics.x &&
          difference<8.0*epsilon*(normal.kinematics.energy_GeV+normal.kinematics.photon_GeV),
          "Charge conjugation changes species only, preserving Compton probability and recoil",difference);
  }
  const auto in = feebl::compton::incident(125.0, 0.01, 0.0, 0.0, 1);
  const double t = std::log1p(0.5*in.x);
  const double y = -std::expm1(-t);
  const double old_x = 4.0*in.energy_GeV*in.photon_GeV/(mass*mass);
  const double old_energy = (1.0-y)*in.energy_GeV;
  const double old_pt2 = mass*mass*y*(old_x-(old_x+1.0)*y);
  const double old_pz2 = old_energy*old_energy-old_pt2-mass*mass;
  const auto fixed = feebl::compton::final_state(in, t, 0.0);
  check(old_pz2 < 0.0 && conservation_error(in, fixed) < 128.0*epsilon,
        "A concrete legacy imaginary-pz recoil becomes a physical state", old_pz2);
  check(fixed.electron_momentum_GeV[2] < 0.0 && fixed.electron_angle_rad > PI/2.0,
        "Hard recoil can reverse without changing the incident species direction");

  const auto rest_in = feebl::compton::incident(mass, 0.2*mass, 0.0, 0.0, -1);
  const auto unscattered = feebl::compton::final_state(rest_in, 0.0, 1.2);
  check(unscattered.electron_energy_GeV == mass
        && magnitude(unscattered.electron_momentum_GeV) == 0.0,
        "An initially stationary unscattered electron remains at rest");

  const double energy = 2.0*mass;
  const double momentum = std::sqrt((energy-mass)*(energy+mass));
  const double photon = 0.5*(mass-mass*mass/(energy+momentum));
  const auto stopping = feebl::compton::incident(energy, photon, 0.2, 0.3, 1);
  const auto stopped = feebl::compton::final_state(stopping, std::log1p(stopping.x), 0.9);
  // At exact rest, |p| has square-root sensitivity to endpoint rounding.
  // Kinetic energy and p^2 are smooth and retain the original epsilon bound.
  const double rest_error = std::max(conservation_error(stopping, stopped),
      std::max(std::abs(stopped.electron_energy_GeV-mass)/mass,
        static_cast<double>(dot(stopped.electron_momentum_GeV,
                                stopped.electron_momentum_GeV))/(mass*mass)));
  check(rest_error < 64.0*epsilon,
        "A physically stopped recoil is retained without a direction division", rest_error);
  if (rest_error >= 64.0*epsilon) {
    std::printf("Stopped recoil diagnostic: E=%.17g k=%.17g x=%.17g "
        "tmax=%.17g Eout=%.17g m=%.17g p2_over_m2=%.17g\n",
        energy, photon, stopping.x,
        std::log1p(stopping.x), stopped.electron_energy_GeV, mass,
        static_cast<double>(dot(stopped.electron_momentum_GeV,
                                stopped.electron_momentum_GeV))/(mass*mass));
  }

  // Equal azimuths for the two beam signs must preserve transverse moments.
  // Averaging uniform azimuths tests isotropy without relying on a particular
  // transverse-basis phase convention.
  double worst_transverse_mean = 0.0;
  for (int direction : {-1, 1}) {
    const auto tilted = feebl::compton::incident(0.1, 0.01, 0.2, -0.3, direction);
    Vector mean = {{0.0, 0.0, 0.0}};
    for (int azimuth = 0; azimuth < 16; ++azimuth) {
      const auto out = feebl::compton::final_state(
          tilted, 0.5*std::log1p(tilted.x), 2.0*PI*azimuth/16.0);
      const double parallel = static_cast<double>(dot(out.electron_momentum_GeV, tilted.axis));
      for (int axis = 0; axis < 3; ++axis)
        mean[axis] += (out.electron_momentum_GeV[axis]-parallel*tilted.axis[axis])/16.0;
    }
    worst_transverse_mean = std::max(worst_transverse_mean, magnitude(mean)/0.1);
  }
  check(worst_transverse_mean < 64.0*epsilon,
        "Uniform azimuths are isotropic for both incident beam directions", worst_transverse_mean);
}

template <typename Operation>
bool rejects(Operation operation) {
  try { operation(); }
  catch (const std::domain_error&) { return true; }
  return false;
}

void check_invalid_input() {
  using namespace feebl::compton;
  const auto valid = incident(1.0, 0.01, 0.0, 0.0, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const bool rejected = rejects([&] { incident(0.5*EMASS, 0.01, 0.0, 0.0, 1); })
      && rejects([&] { incident_for_native_branch(1.0,0.01,0.0,0.0,-1); })
      && rejects([&] { incident(1.0, 0.0, 0.0, 0.0, 1); })
      && rejects([&] { incident(1.0, 0.01, 1.0, 0.0, 1); })
      && rejects([&] { incident(1.0, 0.01, 0.0, 0.0, 0); })
      && rejects([&] { incident(nan, 0.01, 0.0, 0.0, 1); })
      && rejects([&] { cross_section_m2(infinity); })
      && rejects([&] { sample_log_recoil(0.0, 0.5); })
      && rejects([&] { sample_log_recoil(1.0, -0.1); })
      && rejects([&] { sample_log_recoil(1.0, 1.1); })
      && rejects([&] { final_state(valid, -0.1, 0.0); })
      && rejects([&] { final_state(valid, std::nextafter(std::log1p(valid.x), infinity), 0.0); })
      && rejects([&] { final_state(valid, 0.1, nan); });
  check(rejected, "Invalid states and genuinely out-of-support draws fail explicitly");
}
}

int main() {
  try {
    check_kinematics();
    check_cross_section_and_sampling();
    check_special_states();
    check_invalid_input();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Unexpected Compton test exception: %s\n", error.what());
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
