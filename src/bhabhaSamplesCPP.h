#ifndef BHABHASAMPLES_SEEN
#define BHABHASAMPLES_SEEN

#include "particlesCPP.h"
#include "groupedEventsCPP.h"
#include <stdexcept>
#include "pairsCPP.h"
#include "mathematicalEntities.h"

#include "abstractParticle.h"
#include "fileInputOutput.h"



#include <iostream>
#include <string>
#include <vector>

class BHABHA_PHOTON_SAMPLES : public ABSTRACT_BHABHA_PHOTON_SAMPLES
{

  std::vector<QUADRIVECTOR> bhabha_photons_;
  std::vector<int> number_bhabha_;
  int label_;
  unsigned int next_;
  public :
    
    BHABHA_PHOTON_SAMPLES() : label_(-1), next_(0) {;}
  
  
  virtual ~BHABHA_PHOTON_SAMPLES() {;}
  
  virtual inline int get_label() const  { return label_;}
  
  
  virtual inline unsigned int nb_samples() const 
    {
      if ( next_ == 0 ) return bhabha_photons_.size();
      else return next_;
    }
  
  
  virtual inline void get_parameters_for_output(unsigned int number, int& number_bhabha, float& en,float& vx,float& vy, float& vz) const
    {
      bhabha_photons_[number].trivector(vx, vy, vz);
      en = bhabha_photons_[number].energy();
      number_bhabha = number_bhabha_[number];
    }
  
  virtual inline void add_bhabha_photon(int nbhabha, float px, float py, float pz, float en)
    {
      QUADRIVECTOR bhab_phot = QUADRIVECTOR(px,py,pz,en);
      number_bhabha_.push_back(nbhabha);
      bhabha_photons_.push_back(bhab_phot);
    }
  
  
  inline void set_label(int label)  {label_ = label;}
  
/*  inline void create_bhabha_photon(int number_bhabha, float px, float py, float pz, float en)
    {
      QUADRIVECTOR bhab_phot = QUADRIVECTOR(px,py,pz,en);
      bhabha_photons_.push_back(bhab_phot);
      number_bhabha_.push_back(number_bhabha);
    }
*/
/* This function shouldn't exist - no changing the number of bhabha after loading
    inline void set_number_bhabha(int index, int num)
    {
      number_bhabha_[index] = num;
    }
*/
  inline  int load(std::string bhabhaPhotonFIleIni)
    {
      FILE_IN_OUT filin;
      filin.open_file(bhabhaPhotonFIleIni,"r");
      filin.read_bhabhaPhotonsamples(this);
      filin.close();
      return bhabha_photons_.size();
    }
  
  bool pick_next(float ecmratio, float& en,float& px,float& py,float& pz, int& found)  ; 
  
/*  inline  void set(int index, float en,float px,float py,float pz, int number_bhabha)
    {
      bhabha_photons_[index].set(px, py, pz, en);
      number_bhabha_[index] = number_bhabha;
    }
*/
  inline void save_on_file(std::string nameOfOutputFile) const 
    {
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      filout.save_bhabhaPhotonSamples(this);
      filout.close();
    }


};

class BHABHASAMPLES : public ABSTRACT_BHABHASAMPLES
{
  
  typedef struct
  {
    // The fixed legacy reader is an adapter to the grouped event model.
    feebl::events::Event event;
    std::vector<float> legacy_signed_energy;
    float mother1=0,mother2=0,eCM=0;
    unsigned int evtIndex;
    int nbphot;
  } BHABHA_INI;
  
  std::vector<BHABHA_INI> bhabha_;
  unsigned long next_;
  unsigned long prod_info_;
  
  public :
    
    BHABHASAMPLES() : next_(0), prod_info_(0) {;}
  
  virtual  ~BHABHASAMPLES() {;}
  
  
  virtual inline void get_parameters_for_output(unsigned int number, unsigned int& evtIdx, float& eCM, float& mother1_en,float&e1,float&vx1,float& vy1, float&vz1, float& mother2_en, float& e2, float& vx2, float&vy2, float&vz2, int& nbphot) const
    {
      float px, py, pz;
      
      if (number >= next_)
		{
		  std::cerr << " WARNING : BHABHASAMPLES::get_parameters_for_output: number of bhabha_prod to save is out of range number = " << number << " next_= " << next_  << std::endl;
		  return;
		}
      evtIdx = bhabha_[number].evtIndex;
      eCM = bhabha_[number].eCM;
      mother1_en = bhabha_[number].mother1;
      mother2_en = bhabha_[number].mother2;
      
      e1 = bhabha_[number].legacy_signed_energy.at(0);
      
      const auto& first = bhabha_[number].event.particles.at(0).input;
      px=first.px; py=first.py; pz=first.pz;
      vx1=px/std::abs(e1);
      vy1=py/std::abs(e1);
      vz1=pz/std::abs(e1);
      
      e2=bhabha_[number].legacy_signed_energy.at(1);
      const auto& second = bhabha_[number].event.particles.at(1).input;
      px=second.px; py=second.py; pz=second.pz;
      vx2=px/std::abs(e2);
      vy2=py/std::abs(e2);
      vz2=pz/std::abs(e2); 
      nbphot = bhabha_[number].nbphot;
    }
 
  virtual inline unsigned int nb_samples() const {return next_;}
  

  bool pick_next_bhabha(float e1, float e2, float ecmratio, float eCM, float& px1,float& py1,float& pz1, float& en1,float& px2,float& py2,float& pz2,float& en2, int& nbphot, unsigned int& number_bhabha);
  
  
  inline void save_on_file(std::string nameOfOutputFile) const 
    {
      FILE_IN_OUT filout;
      filout.open_file(nameOfOutputFile, "w");
      filout.save_bhabhasamples(this);
      filout.close();
    }

 
  virtual inline void add_bhabha(unsigned int evtIdx, float px1, float py1, float pz1, float e1, float px2, float py2, float pz2, float e2, int nbphot)
    {
      BHABHA_INI bhab;
      bhab.event.id = "legacy-bhabha:"+std::to_string(evtIdx);
      bhab.event.file_event_index = bhabha_.size();
      // Public grouped energies are nonnegative. Preserve the inherited
      // signed-energy convention only in this private compatibility adapter.
      feebl::events::Particle first, second;
      first.id=bhab.event.id+":1"; first.original_row=1; first.pdg=0;
      second.id=bhab.event.id+":2"; second.original_row=2; second.pdg=0;
      first.input={px1,py1,pz1,std::abs(e1)}; first.output=first.input;
      second.input={px2,py2,pz2,std::abs(e2)}; second.output=second.input;
      // The historical converter/charge ordering is unauthenticated: PDG 0
      // explicitly marks unknown species. LHE export and the new transport
      // reject these records until a source-specific mapping is qualified.
      // Legacy cards may use massless generator kinematics. Their records
      // are not silently admitted to the massive new transport profile.
      bhab.event.particles={first,second};
      bhab.legacy_signed_energy={e1,e2};
      bhab.nbphot = nbphot;
      bhab.evtIndex = evtIdx;
      bhabha_.push_back(bhab);
    }
  
  inline  int load_bhabha(std::string bhabhaFIleIni)
    {
      FILE_IN_OUT filin;
      filin.open_file(bhabhaFIleIni,"r");
      filin.read_bhabhasamples(this);
      filin.close();
      return bhabha_.size();
    }

  // Read-only identity/data seam for migration and diagnostics. The inherited
  // picker below still deliberately admits only its two-body adapter.
  const feebl::events::Event& grouped_event(std::size_t index) const
    { return bhabha_.at(index).event; }

};

class BHABHA
{
  BHABHASAMPLES bhabhaReserve_; 
  BHABHA_PHOTON_SAMPLES bhabhaPhotonReserve_;
  BHABHA_PHOTON_SAMPLES boostedBhabhaPhotons_;
  //  PAIR_BEAM bhabhas_;
  int nbhabha_ini_;
  int nbhabha_photon_ini_;
  
  
  /********************************************************/
  /*!boost bhabha from CM frame of e+e- (P1P2) to lab frame*/
  /********************************************************/
  void bhabha_rotation(float theta, float phi, float& px, float& py, float& pz)
    {
      double pxin=px;
      double pyin=py;
      double pzin=pz;
      double costh = cos(theta);
      double sinth = sin(theta);
      double cosphi = cos(phi);
      double sinphi = sin(phi);
      px = costh*cosphi*pxin-sinphi*pyin+sinth*cosphi*pzin;
      py = costh*sinphi*pxin+cosphi*pyin+sinth*sinphi*pzin;
      pz = -sinth*pxin+costh*pzin;
    }
  
  int fourboost(float &en, float &px, float &py, float &pz, double beta_x, double beta_y, double beta_z)
  {
		double ein = en;
		double pxin = px;
		double pyin = py;
		double pzin = pz;
		double check_invar_mass = ein*ein - pxin*pxin - pyin*pyin - pzin*pzin;

		double betasq = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

		if(1. - betasq < 1.e-20)
		{
			printf("Boost too far in fourboost. beta = (%f, %f, %f)\n", beta_x, beta_y, beta_z);
			px = py = pz = en = 0;
			return -1;
		}

		double gamma = 1./sqrt(1.- betasq);
		double gm1 = gamma - 1.;
		double Bxx = gm1*beta_x*beta_x/betasq;
		double Bxy = gm1*beta_x*beta_y/betasq;
		double Bxz = gm1*beta_x*beta_z/betasq;
		double Byy = gm1*beta_y*beta_y/betasq;
		double Byz = gm1*beta_y*beta_z/betasq;
		double Bzz = gm1*beta_z*beta_z/betasq;

		double enout = gamma*(ein - beta_x*pxin - beta_y*pyin - beta_z*pzin);
		double pxout = -gamma*beta_x*ein + (Bxx+1.)*pxin + Bxy*pyin + Bxz*pzin;
		double pyout = -gamma*beta_y*ein + Bxy*pxin + (Byy+1.)*pyin + Byz*pzin;
		double pzout = -gamma*beta_z*ein + Bxz*pxin + Byz*pyin + (Bzz+1.)*pzin;
		en = enout; px = pxout; py = pyout; pz = pzout;

		double end_invar_mass = enout*enout - pxout*pxout - pyout*pyout - pzout*pzout;

		if(std::abs(check_invar_mass) > 1. || std::abs(end_invar_mass) > 1. || std::abs(check_invar_mass - end_invar_mass)>1.)
		{
			printf("Start invariant mass: %f\n", check_invar_mass);
			printf("End invariant mass: %f\n", end_invar_mass);
			return -2;
		}

		return 0;
  }

/*
  void lorent_bhabha(float e1,float e2,float pz1,float pz2,float& e,float& pz)
    {
      float beta, eold, gam;
      
      beta = -(pz1 + pz2) / (e1 + e2);
      gam = 1.0 / sqrt(1.0 - beta * beta);
      eold = e;
      e = gam * (e - beta * pz);
      pz = gam * (pz - beta * eold);
    }
  
  void lorent_bhabha_back(float& e,float& pl,float beta)
    {
      float gamma,eold;
      gamma=1.0/sqrt(1.0-beta*beta);
      eold=e;
      e=gamma*(eold + beta * pl);
      pl=gamma*(pl + beta * eold);
    }
  
  void frame_change_part_of_bhabha(float partVx, float partVy, float e, float& px, float& py,float& theta, float& phi)
    {
      float cosphi,sinphi;
      px=e*partVx;
      py=e*partVy;
      
      theta=asin(sqrt(partVx*partVx+partVy*partVy));
      if(std::abs(theta)<0.00001)
		{
		  phi=0.;
		}
      else
		{
		  cosphi=partVx/sin(theta);
		  sinphi=partVy/sin(theta);
		  if(sinphi>0.) phi=acos(cosphi);
		  if(sinphi<0.) phi=2*PI-acos(cosphi);
		}
    }
  
  void lorent_bhabha_transformation(float e1, float e2, float pz1, float pz2,float beta_x, float beta_y, float theta, float phi, float& pxin,float& pyin,float& pzin,float& ein)
    {
      
      lorent_bhabha(e1,e2,pz1,pz2, ein,pzin);
      lorent_bhabha_back( ein,pxin,beta_x);
      lorent_bhabha_back( ein,pyin,beta_y);
      bhabha_rotation(theta,phi,pxin,pyin,pzin);
    }
*/
  void boost_bhabha(float part1Vx, float part1Vy, float part2Vx, float part2Vy,float e1, float e2, float& px1in,float& py1in,float& pz1in,float& e1in,float& px2in,float& py2in,float& pz2in,float& e2in, int nphot, float ecmratio,  int do_bhabha, int number_bhabha);
  
 public:
  
  BHABHA() : nbhabha_ini_(0),nbhabha_photon_ini_(0) {;}
  ~BHABHA() {;}
  
  inline void load_samples(int do_bhabhas, std::string bhabha_samples, std::string bhabha_photon_samples)
    {
	  std::cout << "Loading Bhabha samples\n";
      nbhabha_ini_ = bhabhaReserve_.load_bhabha(bhabha_samples);
      if (do_bhabhas > 1) 
		{
    	  std::cout << "Loading Bhabha photons\n";
		  nbhabha_photon_ini_ =  bhabhaPhotonReserve_.load(bhabha_photon_samples);
		}
    }
  
    inline void make_bhabha(PAIR_BEAM& bhabhas, float part1Vx, float part1Vy, float part2Vx, float part2Vy, float e1, float e2,float ecm, float weight, MESH& mesh, int cellx, int celly,float min_z, const SWITCHES& switches, RNDM& rndm_generator, int beamslice1, int beamslice2)
    {
      float px1, py1, pz1, en1, px2, py2, pz2, en2;
      int nbphot;
      unsigned int evtIndex;
      float ecmratio = ecm/switches.get_bhabha_ecmload();
      double bhabhan = switches.get_bhabha_scal()*weight*std::pow(float(ecmratio*ecmratio),float(-.9891));
      if (rndm_generator.rndm()< bhabhan)
        {
          if (bhabhaReserve_.pick_next_bhabha(e1, e2, ecmratio, ecm, px1, py1, pz1, en1, px2, py2, pz2, en2, nbphot, evtIndex) )
            {
              boost_bhabha(part1Vx, part1Vy, part2Vx, part2Vy, e1, e2, px1, py1, pz1, en1, px2, py2, pz2, en2, nbphot, ecmratio, switches.get_do_bhabhas(), evtIndex);
              bhabhas.new_pair(evtIndex, mesh, cellx, celly, min_z, -1, en1, px1, py1,pz1, switches.get_bhabha_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator, beamslice1, beamslice2 );
              bhabhas.new_pair(evtIndex, mesh, cellx, celly, min_z, -1, en2, px2, py2,pz2, switches.get_bhabha_ratio(), switches.get_track_pairs(), switches.get_store_pairs(), rndm_generator, beamslice1, beamslice2  );

            }
        }
    }
  

  inline void numbers_of_loaded(int& nbhabha_ini, int& nbhabha_photon_ini) const
    {
      nbhabha_ini = nbhabha_ini_;
      nbhabha_photon_ini = nbhabha_photon_ini_;
    }
  inline void save_on_files(int do_bhabhas, int /* unused: was the EDM4HEP flag */, std::string bhabha_prod, std::string bhphoton_prod, std::string bhphotons) const
    {
	  std::cout << "Saving Bhabha samples in " << bhabha_prod << std::endl;
      bhabhaReserve_.save_on_file(bhabha_prod);
      //     bhabhaSamples_.save_on_file_pour_C(std::string("bhabhaIniPourC"));
      if (do_bhabhas == 2)
		{
    	  std::cout << "Saving Bhabha photon samples in " << bhphoton_prod << std::endl;
		  bhabhaPhotonReserve_.save_on_file(bhphoton_prod);
    	  std::cout << "Saving boosted Bhabha photons in " << bhphotons << std::endl;
		  boostedBhabhaPhotons_.save_on_file(bhphotons);
		}
    }
};

#endif
