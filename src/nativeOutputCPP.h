#ifndef NATIVEOUTPUTCPP_SEEN
#define NATIVEOUTPUTCPP_SEEN

#include <string>
#include <sstream>

// Self-describing native text output. Named columns, units and beam-direction
// conventions supplement the unlabelled legacy output without an I/O library.
// The frame declaration distinguishes crossing-angle position shears from
// momentum rotations, so downstream angular observables use the correct frame.

struct NativeRunMeta
{
  std::string feebl_version;
  std::string accelerator;         // deck names, for provenance
  std::string parameters;
  double      half_crossing_angle_rad;
  bool        momenta_carry_crossing_angle;
  bool        spin_present;
  long        rndm_seed;

  NativeRunMeta()
    : feebl_version("unknown"), accelerator(""), parameters(""),
      half_crossing_angle_rad(0.0), momenta_carry_crossing_angle(false),
      spin_present(false), rndm_seed(0) {}
};

// One header per file. `block` names the record type, `columns` is a
// space-separated list matching the row layout exactly.
inline std::string native_header(const NativeRunMeta& m,
                                 const char* block, const char* columns)
{
  std::ostringstream o;
  o << "#FEEBL-NATIVE 1.0\n"
    << "#feebl_version " << m.feebl_version << "\n"
    << "#accelerator " << m.accelerator << "\n"
    << "#parameters " << m.parameters << "\n"
    << "#rndm_seed " << m.rndm_seed << "\n"
    << "#\n"
    << "# units\n"
    << "#length_unit nm\n"
    << "#z_unit um\n"
    << "#energy_unit GeV\n"
    << "#angle_unit rad\n"
    << "#\n"
    << "# frame and conventions -- read these before using the numbers\n"
    // Not unconditionally "lab": when a crossing angle is present and the
    // momenta do not carry it, the directions in this file ARE the head-on
    // frame, and the #WARNING block below says so. Declaring "lab" here as
    // well would let a consumer that trusts the key and skips the prose read
    // head-on directions as detector directions -- the precise failure this
    // format exists to prevent.
    << "#frame "
    << ((!m.momenta_carry_crossing_angle && m.half_crossing_angle_rad != 0.0)
        ? "head_on" : "lab") << "\n"
    << "#beam1_direction +z\n"
    << "#beam2_direction -z\n"
    << "#transverse_momenta_are slopes_vx_vy\n"
    << "#vz_sign_convention beam1_positive_beam2_negative\n"
    << "#half_crossing_angle_rad " << m.half_crossing_angle_rad << "\n"
    << "#momenta_carry_crossing_angle "
    << (m.momenta_carry_crossing_angle ? "yes" : "no") << "\n";
  if (!m.momenta_carry_crossing_angle && m.half_crossing_angle_rad != 0.0)
    {
      o << "#WARNING momenta are in a HEAD-ON frame. The crossing angle is applied\n"
        << "#WARNING as a per-slice position shear only, so every direction here is\n"
        << "#WARNING short by the half crossing angle. Energy-domain quantities are\n"
        << "#WARNING valid; laboratory angular observables require a frame transformation.\n";
    }
  o << "#spin_present " << (m.spin_present ? "yes" : "no") << "\n";
  if (m.spin_present)
    o << "#spin_convention polarization_vector_lab_frame\n"
      << "#spin_modulus_is degree_of_polarization\n"
      << "#spin_is_unit_vector no\n"
      << "#helicity_is s_dot_phat_using_vz_sign\n";
  if (std::string(block) == "pair_spectrum")
    o << "#weight_unit m^-2_per_crossing\n"
      << "#dL_dW_unit m^-2_per_crossing_per_GeV\n"
      << "#helicity_error_model independent_encounter_SEM_proxy\n"
      << "#helicity_error_formula sqrt(weighted_population_variance/n_eff)\n"
      << "#n_eff_definition (sum_w)^2/sum_w_squared\n"
      << "#helicity_error_excludes shared_macroparticle_correlations grid_timestep_model_errors\n"
      << "#helicity_error_zero_if n_eff_le_1_or_zero_observed_variance\n"
      << "#helicity_error_zero_is_not a_calibrated_zero_uncertainty\n"
      << "#precision_uncertainty requires_independent_run_or_ancestry_cluster_covariance\n";
  o << "#\n"
    << "#BLOCK " << block << "\n"
    << "#COLUMNS " << columns << "\n";
  return o.str();
}

inline std::string native_footer(const char* block, unsigned long nrec)
{
  std::ostringstream o;
  o << "#END " << block << " records " << nrec << "\n";
  return o.str();
}

#endif
