#ifndef PHYSICALTOOLS_SEEN
#define  PHYSICALTOOLS_SEEN
#include <iostream>
#include <vector>
#include "besselCPP.h"
#include "typeDefs.h"
#include "rndmCPP.h"
#include "physconst.h"
#include "mathematicalEntities.h"
#include "mathematicalTools.h"

// CODATA 2018 fine-structure constant over pi.  pair_model=1 weights the legacy equivalent photons
// (constant .00232461, alpha = 1/136.93) by QEDBG_ALPHA_OVER_PI/.00232461 in the pair channels.
#ifndef QEDBG_ALPHA_OVER_PI
#define QEDBG_ALPHA_OVER_PI (7.2973525693e-3/3.14159265358979323846)
#endif

class PHYSTOOLS
{

  static BESSEL bessel_;

  public : 

    /* This routine takes an electron with energy e and longitudinal momentum
       pz in the center of mass frame of the two photons e1 and e2 and boosts the
       momentum to the laborytory frame. */
    
    inline static void lorent_pair(float e1,float e2,double& e,double& pz)
    {
      double beta, eold, gam;
      
      beta = -((double)e1 - (double)e2) / ((double)e1 + (double)e2);
      gam = (double)1.0 / sqrt((double)1.0 - beta * beta);
      eold = e;
      e = gam * (e - beta * pz);
      pz = gam * (pz - beta * eold);
    }
  
  inline static void lorent_pair(float e1,float e2, float& e, float& pz)
    {
      double ed = (double)e;
      double pzd = (double)pz;
      lorent_pair(e1, e2, ed, pzd);
      e = (float)ed;
      pz = (float)pzd;
    }
  
  inline static  void lorent(double& e,double& pl,double beta)
    {
      /* +beta because want to transform back */
      double gamma,eold;
      gamma=1.0/sqrt(1.0-beta*beta);
      eold = e;
      e = gamma*(eold + beta * pl);
      pl = gamma*(pl + beta * eold);
    }
  
  inline static  void lorent(float& e, float& pl, float beta)
    {
      double ed = (double)e;
      double pld = (double)pl;
      lorent(ed, pld, (double)beta);
      e = (float)ed;
      pl = (float)pld;
    }
  
  
  /* Differential probability for producing a pair * sqrt(3)*PI */
  
  inline static float fcp(float ups,float x)
    {
      float eta;
      eta=2.0/(3.0*ups*x*(1.0-x));
      return bessel_.ki13(eta)+(x/(1.0-x)+(1.0-x)/x)*bessel_.k23(eta);
    }
  
  /* the K_{2/3} part of fcp: the linear-polarisation term of the pair spectrum is -tau * fcp_k23
     (photon_polarisation=1, qedbg::coherent_pair_split_acceptance) */
  inline static float fcp_k23(float ups,float x)
    {
      float eta;
      eta=2.0/(3.0*ups*x*(1.0-x));
      return bessel_.k23(eta);
    }

  /* total probability of producing a pair */
  
  inline static float u(float ups)
    {
      return 0.23*ups*exp(-8.0/(3.0*ups))*pow(1.0+0.22*ups,-1.0/3.0);
    }
  
  
  inline static  JET_FLOAT pair_ang_d(JET_FLOAT x, double scdgam2i)
    {
      JET_FLOAT t,u;
      t=-(1.0-x);
      u=-(1.0+x);
      return t/u+u/t-2.0*scdgam2i*(1.0/t+1.0/u)-scdgam2i*scdgam2i*(1.0/t+1.0/u)*(1.0/t+1.0/u);
    }
  
  inline static  JET_FLOAT pair_ang_i(JET_FLOAT x, double scdgam2i)
    {
      return 2.0*((1.0+scdgam2i*(1.0-0.5*scdgam2i))*log((1.0+x)/(1.0-x))- (1.0+scdgam2i*scdgam2i/(1.0-x*x))*x);
    }
  
  inline static JET_FLOAT hcd_q1q2_q1q2(JET_FLOAT x)
    {
      JET_FLOAT t,u;
      t=0.5*(1.0-x);
      u=0.5*(1.0+x);
      return 4.0/9.0*(1.0+u*u)/(t*t);
    }
  
  inline static JET_FLOAT hci_q1q2_q1q2(JET_FLOAT x)
    {
      return 4.0/9.0*(x+8.0/(1.0-x)+4.0*log(1.0-x));
    }
  
  inline static JET_FLOAT hcd_qqb(JET_FLOAT x)
    {
      JET_FLOAT t,u;
      t=1.0-x;
      u=1.0+x;
      return (t*t+u*u)/(t*u);
    }
  
  inline static JET_FLOAT hci_qqb(JET_FLOAT x)
    {
    return 2.0*(log((1.0+x)/(1.0-x))-x);
    }
  
  /* sigma=t/s+s/t */
  
  inline static JET_FLOAT hcd_qph(JET_FLOAT x)
    {
      JET_FLOAT t,s;
      t=1.0-x;
      s=2.0;
      return (s*s+t*t)/(s*t);
    }
  
  inline static JET_FLOAT hci_qph(JET_FLOAT x)
    {
      return -2.0*log(1.0-x)+0.5*x*(1.0-0.5*x);
    }
  
  
  inline static JET_FLOAT hcd_q1q1_q1q1(JET_FLOAT x) 
    {
      JET_FLOAT t,u;
      t=0.5*(1.0-x);
      u=0.5*(1.0+x);
      return 4.0/9.0*((1.0+u*u)/(t*t)+(1.0+t*t)/(u*u))-8.0/(27.0*u*t);
    }
  
  inline static JET_FLOAT hci_q1q1_q1q1(JET_FLOAT x)
    {
      return 8.0/9.0*(x*(1.0+8.0/(1.0-x*x))+8.0/3.0*log((1.0-x)/(1.0+x)));
    }
  
  inline static JET_FLOAT hcd_q1q1b_q2q2b(JET_FLOAT x) 
    {
      JET_FLOAT t,u;
      t=0.5*(1.0-x);
      u=0.5*(1.0+x);
      return 4.0/9.0*(t*t+u*u);
    }
  
  inline static  JET_FLOAT hci_q1q1b_q2q2b(JET_FLOAT x)
    {
      return 1.0/27.0*x*(3.0+x*x);
    }
  
  inline static JET_FLOAT hcd_q1q1b_q1q1b(JET_FLOAT x) 
    {
      JET_FLOAT t,u;
      t=-0.5*(1.0-x);
      u=-0.5*(1.0+x);
      return 4.0/9.0*((1.0+u*u)/(t*t)+(t*t+u*u))-8.0*u*u/(27.0*t);
    }
  
  inline static JET_FLOAT hci_q1q1b_q1q1b(JET_FLOAT x)
    {
      return 2.0/9.0*(x*(1.0-x*(1.0-x))+16.0/(1.0-x)+16.0/3.0*log(1.0-x));
    }
  
  inline static JET_FLOAT hcd_q1q1b_gg(JET_FLOAT x) 
    {
      JET_FLOAT t,u;
      t=0.5*(1.0-x);
      u=0.5*(1.0+x);
      return 32.0/27.0*(u/t+t/u)-8.0/3.0*(t*t+u*u);
    }
  
  inline static JET_FLOAT  hci_q1q1b_gg(JET_FLOAT x)
    {
      return 4.0/9.0*(x*(-25.0/3.0-x*x)+16.0/3.0*log((1.0+x)/(1.0-x)));
    }
  
  
  inline static JET_FLOAT hcd_gg_q1q1b(JET_FLOAT x) 
    {
      JET_FLOAT t,u;
      t=0.5*(1.0-x);
      u=0.5*(1.0+x);
      return 1.0/6.0*(u/t+t/u)-3.0/8.0*(t*t+u*u);
    }
  
  inline static JET_FLOAT hci_gg_q1q1b(JET_FLOAT x)
    {
      return -25.0/48.0*x+log((1.0+x)/(1.0-x))/3.0-x*x*x/16.0;
    }
  
  inline static  JET_FLOAT hcd_qg_gq(JET_FLOAT x) 
    {
    JET_FLOAT t,u;
    t=-0.5*(1.0-x);
    u=-0.5*(1.0+x);
    return -4.0/9.0*(1.0+u*u)/u+(u*u+1.0)/(t*t);
    }
  
  inline static  JET_FLOAT hci_qg_gq(JET_FLOAT x)
    {
      /*    return 8.0/9.0*log(0.5*(1.0+x))+11.0/9.0*x+x*x/9.0+8.0/(1.0-x)
	    +4.0*log(1.0-x);*/
      return 8.0/9.0*log(1.0+x)+11.0/9.0*x+x*x/9.0+8.0/(1.0-x)
	+4.0*log(1.0-x);
    }
  
  inline static  JET_FLOAT hcd_gg_gg(JET_FLOAT x)
    {
      JET_FLOAT t,u;
      t=-0.5*(1.0-x);
      u=-0.5*(1.0+x);
      return 9.0/2.0*(3.0-u*t-u/(t*t)-t/(u*u));
    }
  
  inline static  JET_FLOAT hci_gg_gg(JET_FLOAT x) 
    {
      return 99.0/8.0*x+3.0/8.0*x*x*x-9.0*log((1.0+x)/(1.0-x))+36.0*x/(1.0-x*x);
    }
  
  
  static void mkit(double gam2i,double& c, RNDM& rndm_generator);
  
  inline static float synrad_p0(float eng,float dzOnRadius) 
    {
      // 4**(1/3)*(1/137)/(Gamma(4/3)*emass) 
      // cf p21 daniel's thesis and ref 21
      return PCONST*eng*dzOnRadius;
    }
  
  // integral K(5/3,u) du  over x < u < infinity
  inline static void  synradKi53andK23(double x, double& Ki53, double& K23);
  
  inline static double synradK13(double x) 
    {
      if (x <= 165.) return TOOLS::K13(x);
      else return 0.0;
    }
  
  static double fradu0(double ups);
  static double fradsp(double ups);
  
  static float synrad_spin_flip (float upsilonSingleP, float eng,const TRIDVECTOR& e2, const TRIDVECTOR& e3,TRIDVECTOR& polar, float dzOnRadius,std::vector<float>& photon, RNDM& rndm_generator);
  // photonStokes (optional, photon_polarisation=1): three entries per entry of 'photon', the photon's Stokes
  // parameters (xi1, xi2, xi3) in its emission frame e1 = transverse acceleration, e2 = e3 x e1, e3 = emitter
  // velocity (qedbg::synchrotron_stokes in qedbg_photon_stokes.h; emitter spin included in the spin-flip path)
  static float synrad_spin_flip (float upsilonSingleP,float eng, const TRIDVECTOR& e1, const TRIDVECTOR& e2, const TRIDVECTOR& e3, TRIDVECTOR& polar, float dzOnRadius, std::vector<float>&  photon, RNDM& rndm_generator, std::vector<float>* photonStokes = NULL);
  static float synrad_no_spin_flip (float upsilonSingleP,float eng, float dzOnRadius,std::vector<float>& photon, RNDM& rndm_generator, std::vector<float>* photonStokes = NULL);
  
  // photonStokes (optional): float[3], emission-frame Stokes parameters of the emitted photon
  static int synrad_0_no_spin_flip (float upsilonSingleP,float eng,float dzOnRadius,float* x, RNDM& rndm_generator, float* photonStokes = NULL);
  static int synrad_0_spin_flip (float upsilonSingleP,float eng,const TRIDVECTOR& e2, const TRIDVECTOR& e3, TRIDVECTOR& polar, float dzOnRadius,float* x, RNDM& rndm_generator);
  static int synrad_0_spin_flip (float upsilonSingleP,float eng, const TRIDVECTOR& e1, const TRIDVECTOR& e2, const TRIDVECTOR& e3, TRIDVECTOR& polar, TRIDVECTOR& stokes, float dzOnRadius,float* photonEnergy, RNDM& rndm_generator, float* photonStokes = NULL);
  
  
  static double BformFunction(double ups);
  
  static void referenceSpin(double vxd, double vyd, TRIDVECTOR& e1, TRIDVECTOR& e2, TRIDVECTOR& e3, TRIDVECTOR Efield, TRIDVECTOR Bfield, float charge_sign, double betaz = 1.0);
  
  static TRIDVECTOR transverse_Lorentz_force(double vxd, double vyd, TRIDVECTOR E, TRIDVECTOR B, double betaz = 1.0);
  
/**********************************************/
/*! Routines for the generation of secondaries */
/**********************************************/

/*! This routine gives the number of equivalent photons for an electron
   with energy e above an energy fraction xmin according to spectrum number
   iflag */

inline static float requiv(double lns4, float xmin,int iflag)
{
  float help,lnxmin;

  if (xmin>=1.0) return 0.0;
  switch (iflag)
    {
    case 1: 
      help=log(xmin);
      return help * help * .00232460830350086;/* 1/137 / pi */
    case 2:
      help=log(xmin);
      return help * help * .00232460830350086;
    case 3:
      return log(xmin) * -.00464921660700172 * 0.5*lns4;
    case 4:
      return log(xmin) * -.003951834115951462 * 0.5*lns4;
    case 5:
      help=lns4;
      return 2.0*.00232461*help*help;
    case 6:
      help= lns4;
      lnxmin=-log(xmin);
      return .00232461*lnxmin*(lnxmin+help);
    case 7:
      help= lns4;
      lnxmin=-log(xmin);
      return .00232461*lnxmin*(lnxmin+help);
    case 8:
      help= lns4;
      lnxmin=-log(xmin);
      return .00232461*lnxmin*(lnxmin+help);
    case 9:
      help= lns4;
      lnxmin=-log(xmin);
      return .00232461*lnxmin*(lnxmin+help);
    case 10:
      help=log(xmin);
      return (help * help + (-7.0/6.0-1/3*xmin*xmin*xmin+0.5*xmin*xmin+xmin-log(xmin)) )* .00232460830350086;
    case 11:
      // pair_model=2 proposal: iflag 6 extended to Q^2 <= 4*s4 (~s), same constant, so that below s4
      // (where Compton and gamma* hadrons take their photons) the density is that of iflag 6
      lnxmin=-log(xmin);
      return .00232461*lnxmin*(lnxmin+lns4+log(4.0));
    }
  return 0.0;
} /* requiv */

/*! New version of equiv */

inline static void mequiv (double s4, double lns4, float xmin,float e,int iflag,float *eph,float *q2,float *one_m_x, RNDM& rndm_generator)
{
  const float emass2=EMASS*EMASS,eps=1e-5;
  float help,q2max,q2min,lnx,x,lnxmin,z;
  switch (iflag)
    {
    case 1:
      *eph=e*powf(xmin,sqrt(1.0-rndm_generator.rndm_equiv()));
      *q2=0.0;
      *one_m_x=0.0;
      return;
    case 2:
      x=pow(xmin,sqrt(rndm_generator.rndm_equiv()));
      help=1.0-x;
      *eph = e*x;
      if (rndm_generator.rndm_equiv()>(help*help+1.0)*0.5) *eph=0.0;
      *q2=0.0;
      *one_m_x=help;
      return;
    case 3:
      x=pow(xmin,rndm_generator.rndm_equiv());
      help=1.0-x;
      *eph=e*x;
      if (rndm_generator.rndm_equiv()>(help*help+1.0)*0.5) *eph=0.0;
      *q2=emass2*pow(e*e/emass2,rndm_generator.rndm_equiv());
      *one_m_x=help;
      return;
    case 4:
      *eph=pow(xmin,rndm_generator.rndm_equiv());
      help=1.0-*eph;
      *eph*=e;
      if (rndm_generator.rndm_equiv()>(help*help+1.0)*0.5) *eph=0.0;
      *q2=emass2*pow(e*e/emass2,rndm_generator.rndm_equiv());
      *one_m_x=help;
      return;
    case 5:
	if(rndm_generator.rndm_equiv()<0.5){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lns4;
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
	    *eph=0.0;
	    *q2=0.0;
	}
	else{
	    *eph=e*x;
	    *q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	}
/*	if (*q2*(1.0-x)<x*x*emass2) *eph=0.0;*/
/*	if (*q2<x*x*emass2/(1.0-x)*exp(1.0/(1.0+0.5*x*x/(1.0-x)))) *eph=0.0;*/
/*	if (rndm_generator.rndm_equiv()>(log(*q2*(1.0-x)/(x*x*emass2))
			  -2.0*(1.0-x)/(1.0+(1.0-x)*(1.0-x)))
	    /log(*q2*(1.0-x)/(x*x*emass2)))
	    *eph=0.0;*/
	*one_m_x=1.0-x;
	return;
    case 6:
        lnxmin=-log(xmin);
	if(rndm_generator.rndm_equiv()<lnxmin/(lnxmin+lns4)){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lnxmin;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lnxmin;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=s4;
	}
	if((1.0+(1.0-x)*(1.0-x))*0.5<rndm_generator.rndm_equiv()){
	    *eph=0.0;
	    *q2=0.0;
	}
	else{
	    *eph=e*x;
	    *q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	}
	if (*q2*(1.0-x)<x*x*emass2) *eph=0.0;
	*one_m_x=1.0-x;
	return;
    case 7:
        lnxmin=-log(xmin);
	if(rndm_generator.rndm_equiv()<lnxmin/(lnxmin+lns4)){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lnxmin;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lnxmin;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=s4;
	}
	if((1.0+(1.0-x)*(1.0-x))*0.5<rndm_generator.rndm_equiv()){
	    *eph=0.0;
	    *q2=0.0;
	}
	else{
	    *eph=e*x;
	    *q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	}
	*one_m_x=1.0-x;
	return;
    case 8:
        lnxmin=-log(xmin);
	if(rndm_generator.rndm_equiv()<lnxmin/(lnxmin+lns4)){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lnxmin;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lnxmin;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=s4;
	}
        *eph=e*x;
        *q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	*one_m_x=1.0-x;
	return;
    case 9:
        lnxmin=-log(xmin);
	if(rndm_generator.rndm_equiv()*(lnxmin+lns4)<lnxmin){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lnxmin;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lnxmin;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=s4;
	}
        *eph=e*x;
        z=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	*q2=z-x*x*emass2;
	if (2.0*(1.0-x)* *q2+x*x*z<rndm_generator.rndm_equiv()*2.0*z){
	  *q2=0.0;
	  *eph=0.0;
	}
	if (*q2*e*e<x*x*emass2*emass2) {
	  *q2=0.0;
	  *eph=0.0;
	}
	else{
	  if (1.0-x>eps){
	    *q2/=(1.0-x);
	  }
	  else{
	    *eph=0.0;
	    *q2=0.0;
	  }
	}
	*one_m_x=1.0-x;
	return;
    case 11:
      {
	// iflag 11 (pair_model=2): proposal for the exact lowest-order kernels; as iflag 6 but
	//   Q^2 extends to 4*s4 (~s), i.e. the full kinematic range of the scattered lepton.
	//   requiv(..., 11) gives the matching rate with the iflag-6 constant.
	const double q2top = 4.0*s4;
	const double lnq2top = log(q2top/emass2);
	lnxmin=-log(xmin);
	if(rndm_generator.rndm_equiv()<lnxmin/(lnxmin+lnq2top)){
	    lnx=-sqrt(rndm_generator.rndm_equiv())*lnxmin;
	    x=exp(lnx);
	    q2min=x*x*emass2;
	    q2max=emass2;
	}
	else{
	    lnx=-rndm_generator.rndm_equiv()*lnxmin;
	    x=exp(lnx);
	    q2min=emass2;
	    q2max=q2top;
	}
	if((1.0+(1.0-x)*(1.0-x))*0.5<rndm_generator.rndm_equiv()){
	    *eph=0.0;
	    *q2=0.0;
	}
	else{
	    *eph=e*x;
	    *q2=q2min*pow(q2max/q2min,rndm_generator.rndm_equiv());
	}
	if (*q2*(1.0-x)<x*x*emass2) *eph=0.0;
	*one_m_x=1.0-x;
	return;
      }
    case 10:

      double spin= .00232460830350086*(-7.0/6.0-1/3*xmin*xmin*xmin+0.5*xmin*xmin+xmin-log(xmin)); //total spin part of spectrum
      double eps_ = 0.00001; // how close it should be to the random number
      if (spin/(spin+requiv(lns4,xmin,iflag))>rndm_generator.rndm_equiv())
	{
	  double yk=rndm_generator.rndm_equiv()*spin/.00232460830350086;
	  double y1=yk;
	  double x0;
	  double x1=exp(-y1);;
	  help=0;
	  while(fabs( (-7.0/6.0-1.0/3.0*x1*x1*x1+0.5*x1*x1+x1-log(x1) -yk))>eps_ )
	    {	  
	      x0=exp(-y1);
	      y1= yk- (-7.0/6.0-1.0/3.0*x0*x0*x0+0.5*x0*x0+x0-log(x0))+help;
	      x1=exp(-y1);
	      help=y1;
	    }
	  if(x1<xmin){*eph=0;}
	  else{*eph = e*x1;}
	  *q2=0.0;
	  *one_m_x=0;
	}
      else
	{
	  x=pow(xmin,sqrt(rndm_generator.rndm_equiv()));
	  help=1.0-x;
	  *eph = e*x;
	  if (rndm_generator.rndm_equiv()>(help*help+1.0)*0.5) *eph=0.0;
	  *q2=0.0;
	  *one_m_x=0;
	}
      return;
    }
 } /* mequiv */

};



class QUADRIVECTOR
{
  protected :

  float v1_,v2_,v3_;
  float v4_;


 inline void multByFloat(const float& fac)
   {
     v1_ *= fac; 
     v2_ *= fac; 
     v3_ *= fac; 
     v4_ *= fac; 
   }


  public : 

 QUADRIVECTOR() : v1_(0.0),v2_(0.0),v3_(0.0),v4_(0.0)  {;}

QUADRIVECTOR(float vx,float vy,float vz, float energy) 
    {
      set(vx, vy, vz, energy);
    }

inline void set (float vx,float vy,float vz, float energy)
  {
    v1_     = vx;
    v2_     = vy;
    v3_     = vz;
    v4_ = energy;         
  }

inline void trivector(float& vx,float& vy,float& vz) const
  {
    vx     =  v1_;
    vy     =  v2_;
    vz     =  v3_;
  }

inline void quadrivector(float& vx,float& vy,float& vz, float& energy) const
  {
    vx     =  v1_;
    vy     =  v2_;
    vz     =  v3_;
    energy =  v4_;
  }

 inline float energy() const {return v4_;}

 inline QUADRIVECTOR& operator *= (const float& fac)
   {
     multByFloat(fac);
     return *this;
   }

inline void boost_cecile(float e1, float e2, float beta_x, float beta_y)
  {
    PHYSTOOLS::lorent_pair(e1,e2,v4_, v3_);
    PHYSTOOLS::lorent(v4_ ,v1_ ,beta_x);
    PHYSTOOLS::lorent(v4_ ,v2_ ,beta_y);
  }
 
 inline void momentumToVelocity() 
   {
     if (v4_ == 0.0) 
       {
	 std::cout << " mathematicalEntities::momentumToVelocity : null energy! " << std::endl;
	 return;
       }
     float factor = 1.0/fabs(v4_);
     v1_ *= factor;
     v2_ *= factor;
     v3_ *= factor;
   }

};



#endif
