#include "pairsCPP.h"
#include "pairCrossSectionCPP.h"
#include <cmath>
#include <iomanip>
#include <list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "physconst.h"
#include "qedbg_pair_kernels.h"
#include "qedbg_photon_polarisation.h"
#include "qedbg_sampling_guard.h"

namespace {
// Validate generated records before thinning or random position assignment.
// A failed kernel must not turn into a NaN trajectory several steps later.
void validate_generated_secondary(int process, double mass, double energy,
                                  double px, double py, double pz) {
  const double on_shell_energy = std::hypot(mass, std::hypot(px, py, pz));
  // Some inherited generators have already stored a float four-vector. Permit
  // its rounding budget, but reject a physical mass-shell error before thinning
  // consumes random numbers. Canonical transport then uses its momentum once.
  const double tolerance = 8.0*std::numeric_limits<float>::epsilon()*
                           std::max(std::abs(energy), on_shell_energy);
  if (qedbg::finite_nonnegative(std::abs(energy)) && energy != 0.0 &&
      qedbg::finite_nonnegative(std::abs(px)) &&
      qedbg::finite_nonnegative(std::abs(py)) &&
      qedbg::finite_nonnegative(std::abs(pz)) &&
      qedbg::finite_nonnegative(mass) && mass > 0.0 &&
      std::abs(energy) >= static_cast<float>(mass) &&
      std::abs(std::abs(energy)-on_shell_energy) <= tolerance) return;
  std::ostringstream message;
  message << std::setprecision(17) << "Invalid generated secondary: process="
          << process << " signed_energy_GeV=" << energy
          << " momentum_GeV=(" << px << ',' << py << ',' << pz << ')'
          << " rest_mass_GeV=" << mass << " on_shell_energy_GeV=" << on_shell_energy;
  throw std::domain_error(message.str());
}
}


void PAIR_BEAM::distribute_pairs(float delta_z,unsigned int n)
{
  float startLocal,delta_i;
  unsigned int k;
  startLocal = 0.5*delta_z*n;
  delta_i=1.0/delta_z;
  if ( n > active_pairs_.size())
    {
      std::cout << " PAIR_BEAM::distribute_pairs vector too small " << std::endl;
      exit(1);
    }
  for(k=0;k<n;k++) 
    {
      active_pairs_[k].clear();
    }
  std::list<PAIR_PARTICLE>::iterator itr = reserve_.begin();
  int m;
  while(itr != reserve_.end())
    {
      m=(int)floor((startLocal-itr->z())*delta_i);
      if ((m>=0)&&(m< (int)n))
	{
	  active_pairs_[m].push_back(*itr);
	  reserve_.erase(itr++);
	}
      else
	{
	  itr++;
	}
    }
}

void PAIR_BEAM::move_unactive_pairs(float step)
{
  std::list<PAIR_PARTICLE>::iterator itr = reserve_.begin();
  while(itr != reserve_.end())
    {
      itr->advancePosition(step);
      itr++;
    }
}


void PAIR_BEAM::new_pair(const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, double energy,double px,double py,double pz, float ratio, int tracking, int saving, RNDM& rndm_generator )
{
  validate_generated_secondary(index_of_process,pair_parameter_.get_mass(),energy,px,py,pz);
  // test if particle energy is above the required minumum 
  if (fabs(energy) < pair_parameter_.get_ecut() )   
    {
      return;
    }    

  // reduce the number of stored particles if requested (to speed up tracking) 
  if (rndm_generator.rndm_pairs() > ratio)
    {
      return;
    }

  // store particles for tracking?
  if (!tracking  && !saving  ) return;

  float xx,yy,zz,vxx,vyy,vzz;

  mesh.pair_guess_position_in_cell(cellx, celly, min_z, xx, yy, zz,rndm_generator);

  vxx=px/fabs(energy);
  vyy=py/fabs(energy);
  vzz=pz/fabs(energy);

  count_pairs_++;
  PAIR_PARTICLE pair_temp = PAIR_PARTICLE(count_pairs_, index_of_process,xx,yy,zz,vxx,vyy,vzz, energy);
  pair_temp.initialize_transport_momentum(pair_parameter_.get_mass(),px,py,pz);
  // if (store_pairs>1) store particles at production time for comparison
  if (saving > 1)
    {
      pairs0_.push_back(pair_temp);
    }

  if (tracking)
    {
      reserve_.push_back(pair_temp);
    }
}

void PAIR_BEAM::new_pair(const unsigned int evtIndex, const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, double energy,double px,double py,double pz, float ratio, int tracking, int saving, RNDM& rndm_generator, int beamslice1=-1, int beamslice2=-1 )
{
  validate_generated_secondary(index_of_process,pair_parameter_.get_mass(),energy,px,py,pz);
  // test if particle energy is above the required minumum
  if (fabs(energy) < pair_parameter_.get_ecut() )
    {
      return;
    }

  // reduce the number of stored particles if requested (to speed up tracking)
  if (rndm_generator.rndm_pairs() > ratio) 
    {
      return;
    }    
  
  // store particles for tracking? 
  if (!tracking  && !saving  ) return;
  
  float xx,yy,zz,vxx,vyy,vzz;
  
  mesh.pair_guess_position_in_cell(cellx, celly, min_z, xx, yy, zz,rndm_generator);
  

  vxx=px/fabs(energy);
  vyy=py/fabs(energy);
  vzz=pz/fabs(energy);
  
  count_pairs_++;
  PAIR_PARTICLE pair_temp = PAIR_PARTICLE(evtIndex, index_of_process,xx,yy,zz,vxx,vyy,vzz, energy);
  pair_temp.initialize_transport_momentum(pair_parameter_.get_mass(),px,py,pz);

  // for retrieval of timing
  pair_temp.set_slices12( beamslice1 , beamslice2 );

  // if (store_pairs>1) store particles at production time for comparison 
  if (saving > 1)
    {
      pairs0_.push_back(pair_temp);
    }
 
  if (tracking) 
    {
      reserve_.push_back(pair_temp);
    }
//std::cout << " manu :  end of in pairsCPP:new_pair with evtIndex " << std::endl;

}

// pair_model=1: weight of a virtual photon (eorg > 0) drawn from the legacy spectrum (iflag 6) that
// turns it into the corrected equivalent photon: CODATA alpha over the legacy constant times the
// finite-electron-mass factor of the transverse flux (Budnev et al., Phys. Rep. 15 (1975) 181),
//   A = 1 - (1-x) Q2min/(P(x) Q^2),  Q2min = m^2 x^2/(1-x),  P(x) = (1+(1-x)^2)/2,  0 <= A <= 1.
// Applied in the pair and muon channels only; the photons themselves stay those of the legacy model,
// which the Compton and gamma* hadron channels also use.  Real photons (eorg < 0) have weight 1.
static double epa_corrected_weight(float eph, float q2, float eorg)
{
  if (!(eorg > 0.0f)) return 1.0;
  const double x = (double)eph / (double)eorg, omx = 1.0 - x;
  if (!(x > 0.0 && omx > 0.0 && q2 > 0.0f)) return 0.0;
  const double qmin = x * x * EMASS * EMASS / omx, splitP = 0.5 * (1.0 + omx * omx);
  const double A = 1.0 - omx * qmin / (splitP * (double)q2);
  return (A > 0.0) ? A * (QEDBG_ALPHA_OVER_PI / .00232461) : 0.0;
}

// incoherent pair creation : Breit-Wheeler processes
void PAIR_BEAM::make_pair_bw(const MESH& mesh, int cellx, int celly,float min_z,int index_of_process, float eph1,float q2_1,float eorg1, float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1, const float* stokes2)
{
  const double emass_2=EMASS*EMASS,one=1.0;
  double phi;
  double ptot, gam2i, c;
  double pz1,pz2,e1,e2,px1,px2,py1,py2;
  int i;
  double sigma;
  double sigmap = 0.0;
  int niter;
  double pt;
  double ecm2;

  const int pair_model = switches.get_pair_model();
  ecm2 = double(eph1) * eph2;
  if (ecm2 <= emass_2) return;
  gam2i = emass_2 / ecm2;
  // FeeBL already corrects the Born normalisation in its default model.
  // Preserve that correction when adding the EPA and exact-LO options.
  sigma = feebl::pair_production::breit_wheeler_m2(ecm2)*flum;
  if (pair_model == 1)
    sigma *= epa_corrected_weight(eph1, q2_1, eorg1) *
             epa_corrected_weight(eph2, q2_2, eorg2);
  if (pair_model >= 1 && stokes1 && stokes2)
    {
      if (index_of_process == 0) pairs_results_.add_unpolarised(0, 2.0 * sigma);
      sigma *= qedbg::bw_polarisation_factor(gam2i, stokes1, stokes2);
    }

  niter = qedbg::checked_repetitions(sigma, "Breit-Wheeler yield", index_of_process);
  sigma /= niter;
  for (i = 1; i <= niter; ++i)
    {
      ptot = sqrt(ecm2 - emass_2);
      // photon_polarisation=1: the azimuth is sampled together with the polar angle (polarised density)
      bool pol_phi = false;
      double cphi = 0.0, sphi = 0.0;
      if (pair_model == 0)
	{
	  PHYSTOOLS::mkit(gam2i, c, rndm_generator);
	  pt = sqrt(one - c * c) * ptot;
	}
      else if (stokes1 && stokes2)
	{
	  double sth;
	  qedbg::bw_sample_angle_polarised(gam2i, stokes1, stokes2, [&rndm_generator]() { return (double)rndm_generator.rndm_pairs(); }, c, sth, cphi, sphi);
	  pt = sth * ptot;
	  pol_phi = true;
	}
      else
	{
	  // exact angular sampling (mkit's Newton inversion floors theta* near 1.4 mrad for gam2i <~ 1e-5)
	  double sth;
	  qedbg::bw_sample_angle(gam2i, [&rndm_generator]() { return (double)rndm_generator.rndm_pairs(); }, c, sth);
	  pt = sth * ptot;
	}
      pz1 = c * ptot;
      pz2=-pz1;
      e1 = sqrt(ecm2);
      e2=e1;
      PHYSTOOLS::lorent_pair(eph1,eph2,e1,pz1);
      switch(switches.get_pair_q2())
	{
	case 0:
	  if ((q2_1>emass_2)||(q2_2>emass_2))
	    {
	      sigmap=0.0;
	    }
	  else
	    {
	      sigmap=sigma;
	    }
	  break;
	case 1:
	  if ((q2_1>(pt*pt+emass_2))||(q2_2>(pt*pt+emass_2)))
	    {
	      sigmap=0.0;
	    }
	  else
	    {
	      sigmap=sigma;
	    }
	  break;
	case 2:
	  if ((q2_1>ecm2)||(q2_2>ecm2))
	    {
	      sigmap=0.0;
	    }
	  else
	    {
	      sigmap=sigma;
	    }
	  break;
	}
      if (pol_phi)
	{
	  // l- (particle 1) at the sampled azimuth in the lab transverse frame
	  px1=cphi*pt;
	  py1=sphi*pt;
	}
      else
	{
      phi=2.0*PI*rndm_generator.rndm_pairs();
      px1=sin(phi)*pt;
      py1=cos(phi)*pt;
	}
      px2=-px1;
      py2=-py1;
   
      PHYSTOOLS::lorent(e1,px1,beta_x);
   
      PHYSTOOLS::lorent(e1,py1,beta_y);
      
      PHYSTOOLS::lorent_pair(eph1,eph2, e2, pz2);
      PHYSTOOLS::lorent(e2,px2,beta_x);
      PHYSTOOLS::lorent(e2,py2,beta_y);
      e2= -e2;
    
      if (pair_model == 0)
	{
      book_keeping(mesh, index_of_process, e1,px1,py1,pz1, e2, px2,py2,pz2, sigmap,cellx, celly,min_z, switches, rndm_generator );

      if (switches.get_beam_pair())
	{
	  if (eorg1>0.0)
	    {
	      book_keeping_p(mesh, index_of_process, eorg1-eph1,sigmap,cellx, celly,min_z,switches, rndm_generator );
	    }
	  if (eorg2>0.0)
	    {
	      book_keeping_p(mesh,index_of_process, -(eorg2-eph2),sigmap,cellx, celly,min_z,switches,rndm_generator );
	    }
	}
	}
      else
	{
	  // as book_keeping(); with beam_pair=1 the scattered beam leptons (energy eorg - eph along their
	  // beam, as in the legacy model) are stored with their pair (same accept draw) and with the
	  // charge of their beam, and are not booked in the pair counters (as in pair_model=2)
	  const bool lucky = !(sigmap < rndm_generator.rndm());
	  pairs_results_.store_full_pair(index_of_process, e1,px1,py1,pz1, e2,px2,py2,pz2, sigmap, lucky);
	  if (lucky)
	    {
	      new_pair(mesh, cellx, celly, min_z, index_of_process, e1,px1,py1,pz1, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	      new_pair(mesh, cellx, celly, min_z, index_of_process, e2,px2,py2,pz2, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	      if (switches.get_beam_pair())
		{
		  if (eorg1 > 0.0)
		    new_pair(mesh, cellx, celly, min_z, index_of_process, eorg1-eph1, 0.0, 0.0, eorg1-eph1, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
		  if (eorg2 > 0.0)
		    new_pair(mesh, cellx, celly, min_z, index_of_process, (switches.get_charge_sign() > 0.0f) ? eorg2-eph2 : -(eorg2-eph2), 0.0, 0.0, -(eorg2-eph2), switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
		}
	    }
	}
    }
} /* pair_bw */


void  PAIR_BEAM::make_muon(const MESH& mesh, int cellx, int celly,float min_z, int index_of_process, float eph1,float q2_1,float eorg1,float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1, const float* stokes2)
{
  const double mumass_2=MUMASS*MUMASS,one=1.0;
  double phi;
  double ptot,gam2i,c,e1,e2;
  int i;
  double sigma;
  double sigmap = 0.0;
  int niter;
  double pt,px1,py1,pz1,px2,py2,pz2;
  double ecm2;
  const double emass2=EMASS*EMASS;
  const int pair_model = switches.get_pair_model();
  const double q2cut0 = (pair_model >= 1) ? mumass_2 : emass2;
  ecm2 = double(eph1) * eph2;
  if (ecm2 <= mumass_2) return;
  gam2i = mumass_2 / ecm2;
  sigma = feebl::pair_production::breit_wheeler_m2(ecm2,MUMASS)*flum*
          switches.get_muon_scale();
  if (pair_model == 1)
    sigma *= epa_corrected_weight(eph1, q2_1, eorg1) *
             epa_corrected_weight(eph2, q2_2, eorg2);
  if (pair_model >= 1 && stokes1 && stokes2)
    {
      if (index_of_process == 0)
        pairs_results_.add_unpolarised(0, 2.0 * sigma / switches.get_muon_scale());
      sigma *= qedbg::bw_polarisation_factor(gam2i, stokes1, stokes2);
    }
  niter = qedbg::checked_repetitions(sigma, "Breit-Wheeler yield", index_of_process);
  sigma /= niter;
  for (i = 1; i <= niter; ++i)
    {
      ptot = sqrt(ecm2 - mumass_2);
      bool pol_phi = false;
      double cphi = 0.0, sphi = 0.0;
      if (pair_model == 0)
	{
	  PHYSTOOLS::mkit(gam2i, c, rndm_generator);
	  pt = sqrt(one - c * c) * ptot;
	}
      else if (stokes1 && stokes2)
	{
	  double sth;
	  qedbg::bw_sample_angle_polarised(gam2i, stokes1, stokes2, [&rndm_generator]() { return (double)rndm_generator.rndm_pairs(); }, c, sth, cphi, sphi);
	  pt = sth * ptot;
	  pol_phi = true;
	}
      else
	{
	  double sth;
	  qedbg::bw_sample_angle(gam2i, [&rndm_generator]() { return (double)rndm_generator.rndm_pairs(); }, c, sth);
	  pt = sth * ptot;
	}
      pz1 = c * ptot;
      pz2=-pz1;
      e1 = sqrt(ecm2);
      e2=e1;
      PHYSTOOLS::lorent_pair(eph1,eph2,e1,pz1);
      
      switch(switches.get_pair_q2()){
      case 0:
	// pair_model>=1: the cut scale is the produced lepton's mass (legacy: electron mass)
	if ((q2_1>q2cut0)||(q2_2>q2cut0)){
	  sigmap=0.0;
	}
	else{
	  sigmap=sigma;
	}
	break;
      case 1:
	if ((q2_1>(pt*pt+mumass_2))||(q2_2>(pt*pt+mumass_2))){
	  sigmap=0.0;
	}
	else{
	  sigmap=sigma;
	    }
	break;
      case 2:
	if ((q2_1>ecm2)||(q2_2>ecm2)){
	  sigmap=0.0;
	}
	else{
	  sigmap=sigma;
	}
	break;
      }
      if (pol_phi)
	{
	  px1=cphi*pt;
	  py1=sphi*pt;
	}
      else
	{
      phi=2.0*PI*rndm_generator.rndm_pairs();
      px1=sin(phi)*pt;
      py1=cos(phi)*pt;
	}
      px2=-px1;
      py2=-py1;
      PHYSTOOLS::lorent(e1,px1,beta_x);
      PHYSTOOLS::lorent(e1,py1,beta_y);

      PHYSTOOLS::lorent_pair(eph1,eph2, e2, pz2);
      PHYSTOOLS::lorent(e2,px2,beta_x);
      PHYSTOOLS::lorent(e2,py2,beta_y);
      e2= -e2;

      book_keeping_muon(mesh,index_of_process, e1,px1,py1,pz1, e2, px2,py2,pz2, sigmap,cellx, celly,min_z, switches, rndm_generator );

    }
} /* make_muon */

// pair_model = 2: exact lowest-order QED (src/qedbg_pair_kernels.h).  The virtual photons were drawn
// from PHYSTOOLS::mequiv(iflag=11); the kernel builds exact four-momenta from (x = eph/eorg, Q^2) and
// random azimuth/decay angles and returns K such that the expected number of pairs is flum*K.
// Identical leptons, all tree diagrams, lepton masses, photon virtuality and transverse momentum are
// exact; the event is stored with weight flum*K split into chunks <= 1 (unbiased).  Photon pairs whose
// on-shell yield estimate reaches 1 (muons with a large muon_scale) are evaluated nrep times with
// independent kinematics, like the legacy niter loop, instead of storing identical copies.
void PAIR_BEAM::make_pair_exact(const MESH& mesh, int cellx, int celly,float min_z,int index_of_process, float eph1,float q2_1,float eorg1, float eph2,float q2_2,float eorg2, float flum,float beta_x,float beta_y, bool muonPair, SWITCHES& switches,RNDM& rndm_generator, const float* stokes1, const float* stokes2)
{
  const bool virt1 = eorg1 > 0.0, virt2 = eorg2 > 0.0;
  if (!virt1 && !virt2) return;
  if (eph1 <= 0.0 || eph2 <= 0.0) return;
  const double s4 = pair_parameter_.get_s4();
  qedbg::PhotonProposal prop;
  prop.xmin = switches.get_compt_x_min()*EMASS*EMASS/s4;   // GRID::distribute_virtual_photons
  prop.q2max = 4.0*s4;                                     // mequiv iflag 11
  prop.alphaPi = .00232461;                                // requiv iflag 11 (iflag-6 constant)
  prop.emass = EMASS;
  prop.massFactor = false;
  // beam 1 carries electrons; beam 2 carries positrons unless both beams have the same charge
  const bool anti1 = false, anti2 = !(switches.get_charge_sign() > 0.0f);
  const double scale = muonPair ? switches.get_muon_scale() : 1.0;
  // Russian roulette on a cheap on-shell estimate (unbiased): the exact kernel is evaluated with
  // probability p = min(1, flum*sigma_BW(4 E1 E2)/pair_exact_rr) and its weight is divided by p.
  // Every photon pair keeps at least probability 0.05.  pair_exact_rr = 0 evaluates every photon pair.
  double prr = 1.0;
  const double rr = switches.get_pair_exact_rr();
  const double c_est = flum * feebl::pair_production::breit_wheeler_m2(
      double(eph1)*eph2, muonPair ? MUMASS : EMASS) * scale;
  // Check before conversion to int or any loop. This is a workload bound,
  // not a cut on the physical distribution: exceeding it fails the run.
  const int nrep = qedbg::checked_repetitions(c_est, "exact-pair proposal yield", index_of_process);
  if (rr > 0.0)
    {
      // the floor bounds the weight amplification (1/p <= 20) for photon pairs whose on-shell estimate
      // badly underestimates the exact kernel (non-multiperipheral graphs, W^2 > 4 E1 E2 corners);
      // a smaller floor is unbiased too but heavy tailed, i.e. low at practical statistics
      prr = std::max((c_est > 0.0) ? std::min(1.0, c_est / rr) : 0.0, 0.05);
      if (rndm_generator.rndm_pairs() >= prr) return;
    }
  // nrep depends only on the photon pair, before the kernel's random draws.
  // Dividing by nrep and the roulette probability preserves its expectation.
  for (int irep = 0; irep < nrep; ++irep)
    {
  qedbg::KernelEvent ev;
  double K = 0.0, K_unpol = 0.0;
  if (virt1 && virt2)
    {
      const double u[5] = {rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs()};
      K = qedbg::ll_kernel(eorg1, anti1, (double)eph1/eorg1, q2_1, eorg2, anti2, (double)eph2/eorg2, q2_2, muonPair, prop, u, ev);
    }
  else
    {
      const double u[4] = {rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs(), rndm_generator.rndm_pairs()};
      // photon_polarisation=1: the real photon's lab-frame Stokes parameters (NULL: unpolarised)
      if (virt1)
	K = qedbg::bh_kernel(eorg1, anti1, +1, (double)eph1/eorg1, q2_1, eph2, muonPair, prop, u, ev, stokes2, &K_unpol);
      else
	K = qedbg::bh_kernel(eorg2, anti2, -1, (double)eph2/eorg2, q2_2, eph1, muonPair, prop, u, ev, stokes1, &K_unpol);
    }
  double c = K * flum * scale / (prr * nrep);
  qedbg::check_chunk_weight(c, "exact-pair sampled yield", index_of_process);
  pairs_results_.record_exact_weight(index_of_process, c);
  // Book the comparison before the polarised zero-weight exit. An event
  // suppressed by its Stokes state may still have an unpolarised expectation.
  if (K_unpol > 0.0)
    pairs_results_.add_unpolarised(index_of_process, 2.0*K_unpol*flum/(prr*nrep));
  if (!(c > 0.0)) continue;
  // transverse motion of the colliding photons (as in make_pair_bw)
  for (int k = 0; k < ev.n; ++k)
    {
      PHYSTOOLS::lorent(ev.p[k].E, ev.p[k].px, beta_x);
      PHYSTOOLS::lorent(ev.p[k].E, ev.p[k].py, beta_y);
    }
  int il = -1, ia = -1;
  for (int k = 0; k < ev.n; ++k)
    if (ev.p[k].produced) { if (ev.p[k].charge < 0) il = k; else ia = k; }
  if (il < 0 || ia < 0)
    throw std::domain_error("Positive exact-pair weight has no produced lepton pair");
  const qedbg::KernelParticle& L = ev.p[il];
  const qedbg::KernelParticle& A = ev.p[ia];
  while (c > 0.0)
    {
      const double w = (c > 1.0) ? 1.0 : c;
      c -= w;
      const int records_before = count_pairs_;
      if (muonPair)
        {
          const bool lucky = !(w < rndm_generator.rndm());
          pairs_results_.store_full_pair(index_of_process, L.E,L.px,L.py,L.pz,
              -A.E,A.px,A.py,A.pz,w/switches.get_muon_scale(),lucky);
          if (lucky)
            {
              pairs_results_.record_exact_accepted_chunk(index_of_process);
              new_pair(mesh,cellx,celly,min_z,index_of_process,
                  L.E,L.px,L.py,L.pz,
                  switches.get_muon_ratio(),switches.get_track_muons(),switches.get_store_muons(),rndm_generator);
              new_pair(mesh,cellx,celly,min_z,index_of_process,
                  -A.E,A.px,A.py,A.pz,
                  switches.get_muon_ratio(),switches.get_track_muons(),switches.get_store_muons(),rndm_generator);
            }
        }
      else
	{
	  // as book_keeping(), but with beam_pair=1 the scattered beam leptons are stored together with
	  // their pair (same accept draw) and are not booked in the pair counters
	  const bool lucky = !(w < rndm_generator.rndm());
	  pairs_results_.store_full_pair(index_of_process, L.E, L.px, L.py, L.pz, -A.E, A.px, A.py, A.pz, w, lucky);
	  if (lucky)
	    {
              pairs_results_.record_exact_accepted_chunk(index_of_process);
	      new_pair(mesh, cellx, celly, min_z, index_of_process, L.E, L.px, L.py, L.pz, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	      new_pair(mesh, cellx, celly, min_z, index_of_process, (-A.E), A.px, A.py, A.pz, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	      if (switches.get_beam_pair())
		for (int k = 0; k < ev.n; ++k)
		  if (!ev.p[k].produced)
		    new_pair(mesh, cellx, celly, min_z, index_of_process, ((ev.p[k].charge < 0) ? ev.p[k].E : -ev.p[k].E), ev.p[k].px, ev.p[k].py, ev.p[k].pz, switches.get_pair_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator);
	    }
	}
      // Count admitted records after the physical cut and storage thinning.
      // Copies of one kernel event remain correlated and are not extra trials.
      for (int j=records_before; j<count_pairs_; ++j)
        pairs_results_.record_exact_stored_record(index_of_process);
    }
    }
} /* make_pair_exact */



void PAIR_BEAM::admit_event_line(const std::string& line,std::size_t source_row,
                                int admitted,float ratio,int tracking,RNDM& rng)
{
  std::istringstream input(line);
  int scheduled;float energy,x,y,z,vx,vy,vz;
  if (!(input>>scheduled>>energy>>x>>y>>z>>vx>>vy>>vz))
    throw std::runtime_error("Malformed event.ini record "+std::to_string(source_row));
  const float values[]={energy,x,y,z,vx,vy,vz};
  for (const float value:values)
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite event.ini record "+std::to_string(source_row));
  TRIDVECTOR spin;
  if (imported_spin_enabled_ && !(input>>spin(0)>>spin(1)>>spin(2)))
    throw std::runtime_error("imported_spin_mode=1 requires sx sy sz in event.ini record "+std::to_string(source_row));
  input>>std::ws;
  if (!input.eof()) throw std::runtime_error("Extra columns in event.ini record "+std::to_string(source_row));
  if (imported_spin_enabled_) {
    const double speed2=double(vx)*vx+double(vy)*vy+double(vz)*vz;
    if (!(std::abs(energy)>EMASS) || !(speed2>0.0) || speed2>1.0+1e-6 ||
        !std::isfinite(spin.norm2()) || spin.norm2()>1.0+1e-10)
      throw std::runtime_error("Unphysical imported spin or momentum in event.ini record "+std::to_string(source_row));
  }
  new_event(energy,x,y,z,vx,vy,vz,ratio,tracking,rng,source_row,scheduled,admitted,
            imported_spin_enabled_ ? &spin : nullptr);
}

void PAIR_BEAM::load_events(int time_counter,float ratio,int tracking,RNDM& rng)
{
  if (!file_of_events_) return;
  if (!event_to_store_.empty()) {
    int scheduled;std::istringstream input(event_to_store_);
    if (!(input>>scheduled)) throw std::runtime_error("Malformed deferred event.ini time counter");
    if (scheduled>time_counter) return;
    admit_event_line(event_to_store_,pending_source_row_,time_counter,ratio,tracking,rng);
    event_to_store_.clear();
  }
  std::string line;
  while (file_of_events_->read_line(line)) {
    ++imported_source_row_;
    int scheduled;std::istringstream input(line);
    if (!(input>>scheduled))
      throw std::runtime_error("Malformed event.ini time counter at record "+std::to_string(imported_source_row_));
    if (scheduled>time_counter) {
      event_to_store_=line;pending_source_row_=imported_source_row_;break;
    }
    admit_event_line(line,imported_source_row_,time_counter,ratio,tracking,rng);
  }
}

std::string PAIR_BEAM::output_flow() const 
{
  // unused method : concerns only particular pairs (see 
  // PAIR_BEAM::compute_pairs_calls)
  double e1,e2;
  long int n1,n2;
  std::ostringstream out;
  out << title(std::string("pair beams"));
  compute_pairs_calls(n1, e1, n2, e2); 
  out << "pairs_ncal.1 = " << n1 << " energy (GeV) : pairs_cal.1 = " << e1 << std::endl;
  out << "pairs_ncal.2 = " << n2 << " energy (GeV) : pairs_cal.2 = " << e2 << std::endl;
  out << "mstp: " << count_pairs_ << " " << pair_track_.call() << " " <<  (float)pair_track_.step()/ std::max( (float)pair_track_.call(),float(1.0)) << std::endl;
  return out.str();
}

void PAIR_BEAM::compute_pairs_calls(long int& n1, double& e1, long int& n2, double& e2) const
{
  std::list<PAIR_PARTICLE>::const_iterator point;
  float vx, vy,vz, energy, abs_energy;
  float z,r,b;
  float pt,pz,r0,phi;
 
  z=2300.0;
  r=18.0;
  b=3.0;
  e1=0.0;
  e2=0.0;
  n1=0;
  n2=0;
  point= reserve_.begin();
  while(point != reserve_.end())
    {
      //      point->velocities(vx,vy,vz);
      point->velocities(vx,vy);
      vz = point->Zvelocity();
      abs_energy = point->unsigned_energy();
      energy = point->signed_energy();
	pt=sqrt(vx*vx+vy*vy)*abs_energy;
	pz=vz*energy;
	r0=3.3333e3*pt/b;
	phi=z/r0*pt/std::abs(pz);
	if (sqrt(2.0*(1.0-cos(phi)))*r0>r){
	  if (vz>0.0) {
	    e1+=abs_energy;
	    n1++;
	  }
	  else{
	    e2+=abs_energy;
	    n2++;
	  }
	}
	point++;
    }
}

void  PAIR_PARAMETER::init(BEAM& beam1, BEAM& beam2, int massflag, float pair_ecut, float pair_step, float step, int timestep)
{

  float sigma_x1, sigma_y1;
  float sigma_x2, sigma_y2;


  beam1.transverse_sigmas(sigma_x1, sigma_y1);
  beam2.transverse_sigmas(sigma_x2, sigma_y2);

  ecut=pair_ecut;

  s4 =  beam1.get_ebeam()  *  beam2.get_ebeam();
  lns4=log(s4/(EMASS*EMASS));
  d_eps_1_ = 2.0*RE*1e9* step *EMASS* pair_step / (( sigma_x1+ sigma_y1 )* sigma_y1 * timestep);
  d_eps_2_ = 2.0*RE*1e9*step*EMASS*pair_step / ((sigma_x2+sigma_y2)*sigma_y2*timestep);
  if(massflag==0){
    mass_=EMASS;
  }
  else if(massflag==1){
    mass_=MUMASS;
  }
}

void PAIR_PARAMETER::jet_equiv (float xmin,float e,int iflag,float& eph,float& q2,float& wgt, RNDM& rndm_generator) const
{
  const float emass2=EMASS*EMASS,eps=1e-30;
  float help,q2max,q2min,lnx,x,qxmin;
  wgt=1.0;
  switch (iflag)
    {
    case 1:
      q2max=1.0;
      eph=(1.0-xmin)*rndm_generator.rndm_equiv()+xmin;
      help=1.0-eph;
      if (help<=eps){
	  eph=0.0;
	  wgt=0.0;
	  q2=0.0;
	  return;
      }
      qxmin=eph*eph/help;
      q2min=emass2*qxmin;
      wgt=-0.5*(1.0+(help*help)) / (eph*log(xmin))
	   *log(q2max/q2min)
	   /lns4*(1.0-xmin);
      wgt=std::max(float(0.0),wgt);
      q2= q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
      eph *= e;
      return;
    case 2:
      eph=(1.0-xmin)*rndm_generator.rndm_equiv()+xmin;
      help=1.0 - eph;
      q2min=emass2;
      wgt=-0.5*(1.0+(help*help)) / (eph*log(xmin))*(1.0-xmin);
      wgt=std::max(float(0.0),wgt);
      q2= q2min*pow(e*e/q2min,rndm_generator.rndm_equiv());
      eph *= e;
      return;
    case 3:
      q2max=1.0;
      eph = (1.0-xmin)*rndm_generator.rndm_equiv()+xmin;
      help=1.0 - eph;
      if (help<=eps){
	  eph=0.0;
	  wgt=0.0;
	  q2=0.0;
	  return;
      }
      qxmin=eph*eph/help;
      q2min=emass2*qxmin;
      wgt=-0.5*(1.0+(help*help)) / (eph*log(xmin))
	   *log(q2max/q2min)
	   /lns4*(1.0-xmin);
      wgt=std::max(float(0.0),wgt);
      q2= q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
      eph *= e;
      return;
    case 4:
      eph=(1.0-xmin)*rndm_generator.rndm_equiv()+xmin;
      help=1.0 - eph;
      if (help<=0.0){
	  eph=0.0;
	  wgt=0.0;
	  q2=0.0;
	  return;
      }
      qxmin=eph*eph/help;
      q2min=emass2*qxmin;
      wgt=-0.5*(1.0+(help*help)) / (eph*log(xmin))
	   *(lns4-log(qxmin))
	   /lns4*(1.0-xmin);
      wgt=std::max(float(0.0),wgt);
      q2= q2min*pow(e*e/q2min,rndm_generator.rndm_equiv());
      eph*=e;
      return;
    case 5:
	if(rndm_generator.rndm_equiv()<0.5){
	    lnx=-sqrt(rndm_generator.rndm_equiv())
		*lns4;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lns4;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=s4;
	}
	if((1.0+(1.0-x)*(1.0-x))*0.5<rndm_generator.rndm_equiv()){
	    eph=0.0;
	    q2=0.0;
	}
	else{
	    eph=e*x;
	    q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	}
	if (q2*(1.0-x)<x*x*emass2) eph=0.0;
	return;
    }
} 

