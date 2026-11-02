#include "tridentCPP.h"

#include <iostream>

namespace {
  const double CVELOCITY = 2.99792458e+17; /* nm/s */
  const double COMPTON = HBAR*CVELOCITY/EMASS;
}

/* Contact Jakob Esberg, Barbara Dalena or Daniel Schulte */

TRIDENT::TRIDENT() : help(0.0),e_phot(0.0),q2(0.0)
{
  i_equiv=2; //10: virtual photon spectrum up to m*m with spin correction 2: same without spin correction
  s4=0;
  lns4=0;
  one_m_x=0;
  xmin=0.00001;
  r_phot=PHYSTOOLS::requiv(0,xmin,i_equiv); 
  virtExist=false;
  flag=false;
}

bool TRIDENT::makeVirtualPhoton(float* Emother,float* e_phot_,float* q2_,RNDM& rndm_generator)
{
  int n_phot;
  float initialEnergy =fabs(*Emother);
  n_phot=(int)floor(r_phot);
  r_phot -= n_phot;
  if(rndm_generator.rndm()<r_phot) 
    {
      PHYSTOOLS::mequiv(s4, lns4,xmin,initialEnergy,i_equiv,e_phot_,q2_,&one_m_x, rndm_generator);
      if(*e_phot_>0.0) return true;
      else return false;
    }
  else return false;
} //jakob


/* Decides whether to create pairs from the virtual photons */
/* Modifies the energy of the particle Emother */
void TRIDENT::convertVirtualPhotons(float* Emother,std::vector<float> energies, std::vector<float>* tridents , float ups, double dz,RNDM& rndm_generator_)
{
  double p;
  float kap;
  if(energies.size()>1) std::cout << "TRIDENT::createTridents. Error, tridents not able to cope with multiple photons" << std::endl;
  for(unsigned int i=0;i<energies.size();i++)
    {
      kap=kappa(ups,energies[i],fabs(*Emother));    
      p=PHYSTOOLS::u(kap)*ALPHA_EM*EMASS*dz/(energies[i]*COMPTON);
      if(p>0.5)
	{
	  std::cout << "WARNING, TRIDENT::convertVirtualPhotons. Probability exceeding 0.5" << std::endl;
	}
      //      else{
      if(p<0.01)
	{
	  if(p>rndm_generator_.rndm())
	    {
	      tridents->push_back(energies[i]);
	      *Emother+=(*Emother<0.0) ? energies[i]: -energies[i];
	    }
	}
      else if(1-exp(-p)>rndm_generator_.rndm())
	{
	  tridents->push_back(energies[i]);
	  *Emother+=(*Emother<0.0) ? energies[i]: -energies[i];
	}
    }
} //jakob

bool TRIDENT::pick_trident_energy(float kappa,float& energy, RNDM& rndm_generator)
{
  double a=0.13513,eta,dxdy,x,h,hp,tmp,tmp2,y,coh;
  y=2.0*rndm_generator.rndm()-1.0;
  eta=8.0/(3.0*kappa);
  tmp=fabs(y/(1.0+(1.0-y*y)/(2.0*sqrt(eta))));
  tmp2=1.0-tmp*tmp;
  h=-log(tmp2);
  hp=tmp/tmp2*(1.0+(1.0+y*y)/(2.0*sqrt(eta)))
    /(1.0+(1.0-y*y)/(2.0*sqrt(eta)))
    /(1.0+(1.0-y*y)/(2.0*sqrt(eta)));
  if (y>0.0){
    x=0.5*(1.0+sqrt(h/(eta+h)));
  }
  else{
    x=0.5*(1.0-sqrt(h/(eta+h)));
  }
  dxdy=hp/sqrt(h*(h+eta))*(1.0-h/(h+eta));
  coh=PHYSTOOLS::u(kappa);
  tmp=PHYSTOOLS::fcp(kappa,x)*a/coh*dxdy;
  if (rndm_generator.rndm()<tmp){
     energy *= x;
    return true;
  } else {
    return false;
  }
}

void TRIDENT::createTridents(float* Emother,float ups,double dz,std::vector<float>* electrons, std::vector<float>* positrons,std::vector<float>* virt, RNDM& rndm_generator) //dz is in nm
{  
  std::vector<float> energies;
  float kap;
  e_phot=0;
  q2=0;
  virtExist=makeVirtualPhoton(Emother,&e_phot,&q2,rndm_generator);   
  if(virtExist)
    {
      energies.push_back(e_phot); 
    }
  convertVirtualPhotons(Emother,energies,electrons,ups,dz,rndm_generator);
  if (energies.size()>1) std::cout << "TRIDENT::createTridents. Error, tridents not able to cope with multiple photons"<<std::endl;
  if (electrons->size()>1) std::cout << "TRIDENT::createTridents. Error, tridents not able to cope with multiple electrons"<<std::endl;
  for(unsigned int i=0;i<electrons->size();i++)
    {
      help=(*electrons)[i];
      flag=false;
      kap=kappa(ups,help,fabs(*Emother));
      while(!flag)
	{
	  flag=pick_trident_energy(kap,(*electrons)[i],rndm_generator);
	}
      positrons->push_back((*electrons)[i]-help);
      virt->push_back(q2);
    }
  //  bottom: ; //to be removed
}

void TRIDENT::createTridents(float* Emother,float ups,double dz,std::vector<float>* electrons, std::vector<float>* positrons, RNDM& rndm_generator) //dz is in nm
{  
  std::vector<float> energies;
  float kap;
  e_phot=0;
  q2=0;
  virtExist=makeVirtualPhoton(Emother,&e_phot,&q2,rndm_generator);   
  if(virtExist)
    {
      energies.push_back(e_phot); 
    }
  convertVirtualPhotons(Emother,energies,electrons,ups,dz,rndm_generator);
  if (electrons->size()>1) std::cout << "TRIDENT::createTridents. Error, tridents not able to cope with multiple electrons"<<std::endl;
  for(unsigned int i=0;i<electrons->size();i++)
    {
      help=(*electrons)[i];
      flag=false;
      kap=kappa(ups,help,fabs(*Emother));
      while(!flag)
	{
	  flag=pick_trident_energy(kap,(*electrons)[i],rndm_generator);
	}
      positrons->push_back((*electrons)[i]-help);
      //      virt->push_back(q2);
    }
}
