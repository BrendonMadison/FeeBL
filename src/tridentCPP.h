//#include "particlesCPP.h"
#include "physicalTools.h"
//#include "gridCPP.h"
#include "rndmCPP.h"
#include <iostream>
#include <vector>
#include <fstream>

class TRIDENT
{
 private:
  int i_equiv; //10: virtual photon spectrum up to m*m with spin correction 2: same without spin correction
  double s4,lns4;
  float one_m_x,r_phot,xmin;
  float help,e_phot,q2;
  bool virtExist,flag;

 public:
  TRIDENT();
  ~TRIDENT() {;}
  
  bool makeVirtualPhoton(float* Emother,float* e_phot,float* q2, RNDM& rndm_generator_);
  void convertVirtualPhotons(float* Emother,std::vector<float> energies, std::vector<float>* tridents , float ups, double dz,RNDM& rndm_generator_);

  bool pick_trident_energy(float kappa,float& energy, RNDM& rndm_generator);

  void createTridents(float* Emother,float ups,double dz,std::vector<float>* electrons,std::vector<float>* positrons,std::vector<float>* virt,RNDM& rndm_generator);

  void createTridents(float* Emother,float ups,double dz,std::vector<float>* electrons, std::vector<float>* positrons, RNDM& rndm_generator);

  /* Field parameter as seen by the photon */
  inline float kappa(float ups,float e_phot_,float Emother)
    {
      return ups*e_phot_/Emother;
    }

};

