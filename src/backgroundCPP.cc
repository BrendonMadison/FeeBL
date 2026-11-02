#include "backgroundCPP.h"
#include "comptonKinematicsCPP.h"
#include <cmath>
#include <limits>
#include <stdexcept>

// The existing equivalent-photon flux supplies quasi-real photons. Final
// states use exact real-photon, head-on kinematics about the incident primary
// direction. Neither arbitrary photon angles nor off-shell recoil are modeled.
void COMPT::compt_do(const MESH& mesh, int cellx, int celly, float min_z,
                    PAIR_BEAM& secondaries, int index_of_process,
                    float epart, float ephot, float q2, float vx, float vy,
                    float wgt, int dir, SWITCHES& switches, RNDM& rndm_generator)
{
  if (!std::isfinite(q2) || q2 < 0.0f || !std::isfinite(wgt) || wgt < 0.0f)
    throw std::domain_error("Compton virtuality and luminosity weight must be finite and nonnegative");
  if (q2 > EMASS*EMASS) return;
  if (index_of_process < 0 || index_of_process > 1)
    throw std::domain_error("Compton generation requires a real or equivalent-photon process");
  lsum_ += wgt;
  ++ncall_;
  if (!std::isfinite(epart) || epart == 0.0f)
    throw std::domain_error("Compton source requires finite nonzero signed energy");
  if (epart < 0.0f) {
    ++opposite_charge_source_encounters_;
    opposite_charge_source_weight_ += wgt;
  }
  if (ephot == 0.0f) return; // An empty photon-energy bin has no collision.
  const auto source = feebl::compton::incident_for_native_branch(epart,ephot,vx,vy,dir);
  const auto& incident=source.kinematics;
  const double invariant = incident.x*EMASS*EMASS;
  if (q2 > invariant) return;
  if (invariant < switches.get_compt_x_min()*EMASS*EMASS*4.0) return;
  const double expected = feebl::compton::cross_section_m2(incident.x)*
                          wgt*switches.get_compt_scale();
  if (!std::isfinite(expected) || expected < 0.0 ||
      expected >= static_cast<double>(std::numeric_limits<int>::max()-1))
    throw std::domain_error("Compton event multiplicity is outside the finite integer domain");
  sum_ += expected;
  const int count = static_cast<int>(std::floor(expected))+1;
  const double weight = expected/count;

  for (int event=0; event<count; ++event) {
    const double t = feebl::compton::sample_log_recoil(
        incident.x,rndm_generator.rndm());
    if (weight <= 1e-5) continue;
    auto final = feebl::compton::final_state(incident,t,0.0);
    if (final.electron_energy_GeV < switches.get_compt_emax() &&
        final.electron_energy_GeV > switches.get_pair_ecut()) {
      sum2_ += weight;
      const double phi=2.0*PI*rndm_generator.rndm();
      final=feebl::compton::final_state(incident,t,phi);
      // Native signed energy identifies the species, independently of the
      // final momentum direction: hard recoils can travel backwards.
      const double signed_energy = source.charge_sign*final.electron_energy_GeV;
      const auto& momentum=final.electron_momentum_GeV;
      if (compt_results_.store_compt(index_of_process,signed_energy,
                                    momentum[0],momentum[1],momentum[2],
                                    weight,rndm_generator)) {
        secondaries.new_pair(mesh,cellx,celly,min_z,index_of_process,
                             signed_energy,momentum[0],momentum[1],momentum[2],
                             switches.get_pair_ratio(),switches.get_track_pairs(),
                             switches.get_store_pairs(),rndm_generator);
        if (compton_phot_file_ != NULL)
          compton_phot_file_->save_compton_photon(
              final.photon_energy_GeV,final.photon_momentum_GeV[0],
              final.photon_momentum_GeV[1]);
      }
    }
    if (final.photon_angle_rad > 1e-3) {
      sum3_ += weight;
      sume3_ += final.photon_energy_GeV*weight;
    }
    if (final.electron_angle_rad > 1e-3) {
      sum4_ += weight;
      sume4_ += final.electron_energy_GeV*weight;
    }
  }
}
