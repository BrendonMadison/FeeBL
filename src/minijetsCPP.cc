#include "minijetsCPP.h" 
#include "pairCrossSectionCPP.h"
#include <sstream>

void ABSTRACT_MINIJETS::init_jet_file(float s,float ptmin, std::string jetfileName)
{
  if ( !jetfileName.empty() ) 
    {
      jetfile_ = new FILE_IN_OUT();
      jetfile_->open_file(jetfileName, "w");
      jetfile_->set_jet_header(ptmin,sqrt(s));
    }
  else 
    {
      jetfile_ = NULL;
      std::cerr << " MINIJETS::init_jet_file : ERROR : a jetfile must be defined " << std::endl;
    }
}

void ABSTRACT_MINIJETS::update_statistics_arrays(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT pt,JET_FLOAT h)
{
  float c1,c2;
  int i_pt  = 0;
  int i_c = 0;
  c1=fabs(pz1)/sqrt(pt*pt+pz1*pz1);
  c2=fabs(pz2)/sqrt(pt*pt+pz2*pz2);


  //locating the jet in the predefined arrays
  // increment current arrays with jet contribution


  int n_c = c_.size();

  while (pt_[i_pt] < pt) i_pt++;
  while (c_[i_c] < c1) i_c++;


  v_[i_c+n_c*i_pt] += h;


 // v2_ not used for now
  //  v2_[i_c+n_c*i_pt] += h*h;


  i_c = 0;
  while (c_[i_c] < c2 ) i_c++;

  v_[i_c+n_c*i_pt] += h;

 // v2_ not used for now
  //  v2_[i_c+n_c*i_pt] += h*h;
}



std::string ABSTRACT_MINIJETS::output_flow() const 
{
  std::ostringstream out;
  // int i_c,i_pt;
  //int n_pt = pt_.size();
  //int n_c = c_.size();
  out << jet_results_.output_flow();
  // do not delete the following statements
  //   out << title(std::string("jets storage data"));
  //   out << "jet_storage=(";
  //   for (i_pt=0;i_pt< n_pt;i_pt++)
  //     {
  //       for (i_c=0;i_c<n_c;i_c++)
  // 	{
  // 	  out << v_[i_c+n_c*i_pt];
  // 	}
  //       out << std::endl;
  //     }
  //   out << ")" << std::endl;
  return out.str();
}


MINIJETS::~MINIJETS() {;}

MINIJETS::MINIJETS(float s,float ptmin,int iparam,int jet_select, std::string jetfileName) : ABSTRACT_MINIJETS(s, ptmin, iparam, jet_select, jetfileName)
{
  jet_select_x_ = jet_select;
  initNewton(s, ptmin);
}

void MINIJETS::initNewton(float s, float ptmin)
{
    JET_FLOAT xmin, xmax;
    int n=1024;
    xmax=sqrt(1.0-ptmin*ptmin/(0.25*s));
    xmin=-xmax;
    make_the_newtons(xmin, xmax,n);
}

void MINIJETS::make_the_newtons(double xmin,double xmax,int n)
{
  newton_[0].make_newton(&PHYSTOOLS::hci_q1q2_q1q2,&PHYSTOOLS::hcd_q1q2_q1q2,xmin,xmax,n);
  newton_[1].make_newton(&PHYSTOOLS::hci_q1q1_q1q1,&PHYSTOOLS::hcd_q1q1_q1q1,xmin,xmax,n);
  newton_[2].make_newton(&PHYSTOOLS::hci_q1q1b_q2q2b,&PHYSTOOLS::hcd_q1q1b_q2q2b,xmin,xmax,n);

  newton_[3].make_newton(&PHYSTOOLS::hci_q1q1b_q1q1b,&PHYSTOOLS::hcd_q1q1b_q1q1b,xmin,xmax,n);

  newton_[4].make_newton(&PHYSTOOLS::hci_q1q1b_gg,&PHYSTOOLS::hcd_q1q1b_gg,xmin,xmax,n);
  newton_[5].make_newton(&PHYSTOOLS::hci_gg_q1q1b,&PHYSTOOLS::hcd_gg_q1q1b,xmin,xmax,n);
  newton_[6].make_newton(&PHYSTOOLS::hci_qg_gq,&PHYSTOOLS::hcd_qg_gq,xmin,xmax,n);
  newton_[7].make_newton(&PHYSTOOLS::hci_gg_gg,&PHYSTOOLS::hcd_gg_gg,xmin,xmax,n);
  newton_[8].make_newton(&PHYSTOOLS::hci_qqb,&PHYSTOOLS::hcd_qqb,xmin,xmax,n);
  newton_[9].make_newton(&PHYSTOOLS::hci_qph,&PHYSTOOLS::hcd_qph,xmin,xmax,n);
}
// This routine produces the minijets from gamma gamma collision (?)
void MINIJETS::mkjll_(const PAIR_PARAMETER& pair_parameter,float e1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
  float gam2i;
  float rphot;
  //float q2_1,q2_2,wgt1,wgt2;
  double pstar2 = jet_parameter_.get_pstar2();
  if (e1*e2 <= pstar2) return;
  int d_spectrum = jet_parameter_.get_d_spectrum();
  int r_spectrum = jet_parameter_.get_r_spectrum();
  //float jet_ratio = switches.get_jet_ratio();
  gam2i = pstar2/(e1*e2);
  float jetrqvd =  pair_parameter.jet_requiv(gam2i, d_spectrum);
  float jetrqvr =  pair_parameter.jet_requiv(gam2i, r_spectrum);
  rphot = jetrqvd*jetrqvd;
  mkj_no_pythia12(pair_parameter,d_spectrum, d_spectrum, rphot, gam2i,e1, e2, flum, switches, rndm_generator, &MINIJETS::make_jet_0);

   rphot = jetrqvd*jetrqvr;
   mkj_no_pythia12(pair_parameter, r_spectrum, d_spectrum, rphot, gam2i,e1, e2, flum, switches, rndm_generator, &MINIJETS::make_jet_1a);


   mkj_no_pythia12(pair_parameter,d_spectrum, r_spectrum, rphot, gam2i,e1, e2, flum, switches, rndm_generator, &MINIJETS::make_jet_1b);

   rphot = jetrqvr*jetrqvr;
   mkj_no_pythia12(pair_parameter, r_spectrum, r_spectrum, rphot, gam2i,e1, e2, flum, switches, rndm_generator, &MINIJETS::make_jet_2);
}

void MINIJETS::mkjbh1_(const PAIR_PARAMETER& pair_parameter, float eph1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
   float gam2i;
   float rphotr, rphotd;
   int d_spectrum, r_spectrum;
  // This routine produces the minijets from gamma e collision 
  if (eph1*e2 <= jet_parameter_.get_pstar2()) return;
   
  gam2i = jet_parameter_.get_pstar2() / (eph1 * e2);
  d_spectrum = jet_parameter_.get_d_spectrum();
  r_spectrum = jet_parameter_.get_r_spectrum();
  rphotd = pair_parameter.jet_requiv(gam2i,d_spectrum);
  rphotr = pair_parameter.jet_requiv(gam2i,r_spectrum);
  mkj_no_pythia1(pair_parameter,d_spectrum, rphotd,  gam2i,eph1,e2,flum, switches, rndm_generator, &MINIJETS::make_jet_0);
  mkj_no_pythia1(pair_parameter,d_spectrum, rphotd, gam2i,eph1,e2,flum, switches, rndm_generator, &MINIJETS::make_jet_1a);
  mkj_no_pythia1(pair_parameter,r_spectrum, rphotr,  gam2i,eph1,e2,flum, switches, rndm_generator, &MINIJETS::make_jet_1b);
  mkj_no_pythia1(pair_parameter,r_spectrum, rphotr, gam2i,eph1,e2,flum, switches,rndm_generator, &MINIJETS::make_jet_2);
}

/* This routine produces the minijets from e gamma collision */

void MINIJETS::mkjbh2_(const PAIR_PARAMETER& pair_parameter, float e1,float eph2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
  float gam2i;
  float rphotr, rphotd;
  //int k; 
  int d_spectrum, r_spectrum;
  if (e1 * eph2 <= jet_parameter_.get_pstar2()) return;
    
  gam2i = jet_parameter_.get_pstar2() / (e1 * eph2);
  d_spectrum = jet_parameter_.get_d_spectrum();
  r_spectrum = jet_parameter_.get_r_spectrum();
  rphotd = pair_parameter.jet_requiv(gam2i, d_spectrum);
  rphotr = pair_parameter.jet_requiv(gam2i,r_spectrum);

  mkj_no_pythia2(pair_parameter,d_spectrum, rphotd, gam2i, eph2, e1, flum, switches, rndm_generator, &MINIJETS::make_jet_0);  
  mkj_no_pythia2(pair_parameter,r_spectrum, rphotr, gam2i, eph2, e1, flum, switches, rndm_generator, &MINIJETS::make_jet_1a);  
  mkj_no_pythia2(pair_parameter,d_spectrum, rphotd, gam2i, eph2, e1, flum, switches, rndm_generator, &MINIJETS::make_jet_1b);  
  mkj_no_pythia2(pair_parameter,r_spectrum, rphotr, gam2i, eph2, e1, flum, switches, rndm_generator, &MINIJETS::make_jet_2);  
}

// This routine produces the minijets from e+e- collision 

void MINIJETS::mkjbw_(float eph1,float eph2,float flum, SWITCHES& switches, RNDM& rndm_generator)
{
  make_jet_0(eph1,0.0,eph2,0.0,flum,switches, rndm_generator);
  make_jet_1a(eph1,0.0,eph2,0.0,flum, switches, rndm_generator);
  make_jet_1b(eph1,0.0,eph2,0.0,flum,switches, rndm_generator);
  make_jet_2(eph1,0.0,eph2,0.0,flum, switches,  rndm_generator);    
}


void MINIJETS::mkj_no_pythia1(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph1,float  ener2, float flum, SWITCHES& switches, RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&))
{
  // minijets from gamma e collision 
  int k;
  float eph2, q2, wgt;
  int niter = update_niter(rphot,rndm_generator);
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener2,spectrum,eph2, q2, wgt, rndm_generator);
      (this->*make_jet_n)(eph1,0.0,eph2,q2,flum*wgt, switches, rndm_generator); 
    }
}
void MINIJETS::mkj_no_pythia2(const PAIR_PARAMETER& pair_parameter, int spectrum,float rphot, float gam2i,float eph2,float  ener1, float flum, SWITCHES& switches, RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&))
{

  // minijets from e gamma collision
  int k;
  float eph1, q2, wgt;
  int niter = update_niter(rphot,rndm_generator);
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener1,spectrum,eph1, q2, wgt, rndm_generator);
      (this->*make_jet_n)(eph1,q2,eph2,0.0,flum*wgt, switches, rndm_generator); 
    }
}

void MINIJETS::mkj_no_pythia12(const PAIR_PARAMETER& pair_parameter,int spectrum1, int spectrum2, float rphot, float gam2i,float ener1,float  ener2, float flum, SWITCHES& switches,  RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&))
{
  int k;
  float eph1, q2_1, wgt1;
  float eph2, q2_2, wgt2;
  int niter = update_niter(rphot,rndm_generator);
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener1,spectrum1,eph1,q2_1,wgt1, rndm_generator);
      pair_parameter.jet_equiv(gam2i,ener2,spectrum2,eph2,q2_2,wgt2, rndm_generator);
      (this->*make_jet_n)(eph1,q2_1,eph2,q2_2,flum*wgt1*wgt2, switches, rndm_generator); 
    }
}




void MINIJETS::make_jet_0(float eph1,float q2_1,float eph2,float q2_2,float flum,SWITCHES& switches, RNDM& rndm_generator)
{
    JET_FLOAT c,h;
    int k, n;
    JET_FLOAT c0, e1, e2,q2, s4;
    int nf;
    JET_FLOAT pt;
    JET_FLOAT pz1, pz2, ecm2,eph1p,eph2p;
    JET_FLOAT gam2i,sigma_h,one=1.0,ptot;
    int niter;
/* This routine produces the minijets for two colliding photons */
/* iflag gives the calling routine */
/*   0 : two real photons */
/*   1 : first is virtual Q2 calculated */
/*   2 : second is virtual Q2 calculated */
/*   3 : both are virtual Q2 calculated */

    s4 = eph1 * eph2;
    q2 = s4;
    if (s4 < jet_parameter_.get_pstar2()) return;

    eph1p=eph1;
    eph2p=eph2;

    // no_pythia
    eph1=-1.0;
    eph2=-1.0;
    //  

    c0 = sqrt(1. - jet_parameter_.get_pstar2() / s4);
#ifdef CHARM_MASS
    nf=3;
#else
    nf = 4;
    if (q2 < jet_parameter_.get_q2_1()) nf = 3;
    if (q2 > jet_parameter_.get_q2_2()) nf = 5;
#endif
    // thesis, p. 59-60 (see values of constants in physconst.h file)
    h = ECONST0[nf - 3] * FACT_OF_h0 / s4 * (log((c0 + 1.) / (1. - 
	    c0)) - c0) * flum;
/*scd*/
    h *= switches.get_jet_ratio();
    n = (int)floor(h) + 1;
    h /= (float) n;
    ecm2 = sqrt(s4);
   for (k = 1; k <= n; k++)
      {
	c = mkcos(c0, rndm_generator);
	pt = sqrt((1. - c * c) * s4);
	e1 = ecm2;
	e2 = ecm2;
	pz1 = c * ecm2;
	pz2 = -c * ecm2;
	lorent_jet(eph1p,eph2p,e1,pz1);
	lorent_jet(eph1p,eph2p,e2,pz2);
/* scd-1*/
/* correction for the transverse momentum */
	q2=pt*pt;
	if((jet_parameter_.get_d_spectrum()==3)||((q2>q2_1)&&(q2>q2_2))){
	    store_jet(pz1,pz2,eph1,eph2,pt,h,1,switches, rndm_generator);
	    jet_results_.increment_sigma(0,h/switches.get_jet_ratio());
	}
      }
#ifndef CHARM_MASS
    return;
#else
    if (s4<=CHARM_MASS*CHARM_MASS) return;
    gam2i=CHARM_MASS*CHARM_MASS/s4;
    // Three colours times the fourth power of the charm electric charge.
    sigma_h=feebl::pair_production::breit_wheeler_m2(s4,CHARM_MASS)*48.0/81.0*flum;
    sigma_h*=switches.get_jet_ratio();
    niter = (int)floor(sigma_h) + 1;
    sigma_h /= niter;
    for (k = 1; k <= niter; ++k){
      PHYSTOOLS::mkit(gam2i, c, rndm_generator);
	ptot = sqrt(s4-CHARM_MASS*CHARM_MASS);
	pt = sqrt(one-c*c)*ptot;
/*
 * Use transverse mass cutoff. Might be better in some cases to use only pt.
 */
	if (pt*pt+CHARM_MASS*CHARM_MASS>jet_parameter_.get_pstar2())
	  {
	    e1 = ecm2;
	    e2 = ecm2;
	    pz1 = c * ecm2;
	    pz2 = -c * ecm2;
	    lorent_jet(eph1p,eph2p,e1,pz1);
	    lorent_jet(eph1p,eph2p,e2,pz2);
	    q2=pt*pt;
	    if((jet_parameter_.get_d_spectrum()==3)||((q2>q2_1)&&(q2>q2_2))){
		store_jet(pz1,pz2,eph1,eph2,pt,sigma_h,1,switches, rndm_generator);
		jet_results_.increment_sigma(0,sigma_h/switches.get_jet_ratio());
	    }
	}
    }
#endif
    return;
}


void MINIJETS::make_jet_1a(float eph1,float q2_1,float eph2,float q2_2,float flum,SWITCHES& switches, RNDM& rndm_generator)
{
    float d__1;

    /* Local variables */
    float grho;
    JET_FLOAT qrho, c, h;
    int i, n;
    float tttt, xxxx2;
   JET_FLOAT c0,e1,e2,s4;
    int nf;
    JET_FLOAT s04, pt,q2_d;
    JET_FLOAT pz1, pz2;
    JET_FLOAT eph1p,eph2p;
    JET_FLOAT q2,part1[7],part2[7],alphas;
    JET_FLOAT lnx = 0.0;
    const JET_FLOAT charge_q2_1=1.0/9.0,charge_q2_2=4.0/9.0;


/* This routine produces the minijets for two colliding photons */
/* iflag gives the calling routine */
/*   0 : two real photons */
/*   1 : first is virtual Q2 calculated */
/*   2 : second is virtual Q2 calculated */
/*   3 : both are virtual Q2 calculated */


/* eph1 resolved */

    s04 = eph1 * eph2;
    if (s04 <= jet_parameter_.get_pstar2()) return;
    if(jet_select_x_)
      {
	lnx=log(s04/jet_parameter_.get_pstar2());
	xxxx2=exp(-rndm_generator.rndm_jet()*lnx);
    }
    else{
	xxxx2 = rndm_generator.rndm_jet();
    }

    eph1p=eph1*xxxx2;
    eph2p=eph2;

    //    pythia_
    eph1-=eph1p;
    eph2=-1.0;
    //     
    s4 = eph1p*eph2p;
    if (s4 < jet_parameter_.get_pstar2()) return;

    q2=s4;

    nf = 4;
    if (q2 < jet_parameter_.get_q2_1())
      {
	nf = 3;
	if (q2<1.0) q2=1.0;
      }
    else
      {
	if (q2 > jet_parameter_.get_q2_2())
	  {
	    nf = 5;
	    if (q2>1e4) q2=1e4;
	  }
      }

    hadrons(xxxx2,xxxx2,q2,nf,part1,part2,alphas);

    qrho=2.0*(charge_q2_1*(part1[1]+part1[3]+part1[5])
	      +charge_q2_2*(part1[2]+part1[4]+part1[6]));
    grho=part1[0];
    if(jet_select_x_)  tttt=lnx*xxxx2;
    else tttt = 1.0;
    c0 = sqrt(1. - jet_parameter_.get_pstar2() / s4);
    // thesis, p. 59-60 (see values of constants in physconst.h file)
    h = grho * FACT_OF_h1 * ECONST1[nf - 3] * alphas / s4 * (log((c0 + 
	    1.) / (1. - c0)) - c0) * flum *tttt;

    h *= switches.get_jet_ratio();
    n = (int)floor(h) + 1;
    h /= (float) n;
    for (i = 1; i <= n; ++i)
      {
	c = mkcos(c0, rndm_generator);
	e1 = sqrt(s4);
	pz1 = e1 * c;
	e2 = e1;
	pz2 = -pz1;
	pt = sqrt((1. - c * c) * e1 * e2);
	lorent_jet(eph1p,eph2p,e1,pz1);
	lorent_jet(eph1p,eph2p,e2,pz2);
/* scd-1 */
	if (jet_parameter_.get_d_spectrum() == 5)  q2_d=pt*pt;
	else  q2_d=q2;

	if((q2_1<q2)&&(q2_2<q2_d))
	  {
	    store_jet(pz1,pz2,eph1,eph2,pt,h,2,switches,rndm_generator);
	    jet_results_.increment_sigma( 1, h/switches.get_jet_ratio());
	  }
      }
    d__1 = -c0;
    h = qrho * FACT_OF_h2 * alphas / s4 * (-log(1. - c0) + c0 * (
	    2. - c0) / 8. - (-log(1. - d__1) + d__1 * (2. - d__1) / 8.)) *
	    flum * tttt;
/*scd*/
    h *= switches.get_jet_ratio();
    n = (int)floor(h) + 1;
    h /= (float) n;
    for (i = 1; i <= n; ++i)
      {
	c  = mkcosb(c0, rndm_generator);
	e1 = sqrt(s4);
	pz1 = e1 * c;
	e2 = e1;
	pz2 = -pz1;
	pt = sqrt((1. - c * c) * e1 * e2);
	lorent_jet(eph1p,eph2p, e1, pz1);
	lorent_jet(eph1p,eph2p, e2, pz2);

	if (jet_parameter_.get_d_spectrum() == 5)  q2_d=pt*pt;	
	else  q2_d=q2;
	
	if((q2_1<q2)&&(q2_2<q2_d))
	  {
	    store_jet(pz1,pz2,eph1,eph2,pt,h,3,switches, rndm_generator);
	    jet_results_.increment_sigma( 1, h/switches.get_jet_ratio());
	}
      }
    //    return 0;
}

void MINIJETS::make_jet_1b(float eph1,float q2_1,float eph2,float q2_2,float flum, SWITCHES& switches,  RNDM& rndm_generator)
{
    /* System generated locals */
    float d__1, d__2;
    /* Local variables */
    float grho;
    JET_FLOAT qrho, c, h;
    int i, n;
    float tttt, xxxx2;
    JET_FLOAT c0, e1, e2,  s4;
    int nf;
    JET_FLOAT s04, pt,q2_d;
    JET_FLOAT pz1, pz2;
    JET_FLOAT eph1p,eph2p;
    JET_FLOAT q2,part1[7],part2[7],alphas;
    JET_FLOAT lnx = 0.0;
    const JET_FLOAT charge_q2_1=1.0/9.0,charge_q2_2=4.0/9.0;


/* This routine produces the minijets for two colliding photons */
/* iflag gives the calling routine */
/*   0 : two real photons */
/*   1 : first is virtual Q2 calculated */
/*   2 : second is virtual Q2 calculated */
/*   3 : both are virtual Q2 calculated */

/* eph2 resolved */

    s04 = eph1 * eph2;
    if (s04 <= jet_parameter_.get_pstar2()) return;
    if(jet_select_x_)
      {
	lnx=log(s04/jet_select_x_);
	xxxx2=exp(-rndm_generator.rndm_jet()*lnx);
      }
    else
      {
	xxxx2 = rndm_generator.rndm_jet();
      }

    eph1p=eph1;
    eph2p=eph2*xxxx2;

    s4 = eph1p*eph2p;
    if (s4 < jet_parameter_.get_pstar2()) return;

    //   pythia_
    eph1 = -1.0;
    eph2 -= eph2p;
    //    
/* Computing MAX */
    d__1 = 1., d__2 = std::min(s4,9999.);
    q2 = std::max(d__1,d__2);
    
    nf = 4;
    if (q2 < jet_parameter_.get_q2_1())
      {
	nf = 3;
	if (q2<1.0) q2=1.0;
      }
    else
      {
	if (q2 > jet_parameter_.get_q2_2())
	  {
	    nf = 5;
	    if(q2>1e4) q2=1e4;
	  }
    }
    hadrons(xxxx2,xxxx2,q2,nf,part1,part2,alphas);

    qrho=2.0*(charge_q2_1*(part2[1]+part2[3]+part2[5])
	      +charge_q2_2*(part2[2]+part2[4]+part2[6]));
    grho=part2[0];
    if(jet_select_x_)  tttt=lnx*xxxx2;
    else  tttt = 1.0;
    c0 = sqrt(1. - jet_parameter_.get_pstar2() / s4);
    // thesis, p. 59-60 (see values of constants in physconst.h file)
    h = grho * FACT_OF_h1 * ECONST1[nf - 3] * alphas / s4 * (log((c0 + 
	    1.) / (1. - c0)) - c0) * flum * tttt;

    h *= switches.get_jet_ratio();
    n = (int)floor(h) + 1;
    h /= (float) n;
    for (i = 1; i <= n; ++i) 
      {
	c = mkcos(c0, rndm_generator);
	e1 = sqrt(s4);
	pz1 = e1 * c;
	e2 = e1;
	pz2 = -pz1;
	pt = sqrt((1. - c * c) * e1 * e2);
	lorent_jet(eph1p,eph2p,e1,pz1);
	lorent_jet(eph1p,eph2p,e2,pz2);
/* scd-1 */
	if (jet_parameter_.get_d_spectrum() == 5)  q2_d=pt*pt;	
	else q2_d=q2;
	
	if((q2_1 < q2_d)&&( q2_2 < q2))
	  {
	    store_jet(pz1,pz2,eph1,eph2,pt,h,2,switches, rndm_generator);
	    jet_results_.increment_sigma(1, h/switches.get_jet_ratio());
	  }
      }
    d__1 = -c0;
    h = qrho * FACT_OF_h2 * alphas / s4 * (-log(1. - c0) + c0 * (
	    2. - c0) / 8. - (-log(1. - d__1) + d__1 * (2. - d__1) / 8.)) *
	    flum * tttt;
/*scd*/
    h *= switches.get_jet_ratio();
    n = (int)floor(h) + 1;
    h /= (float) n;
    for (i = 1; i <= n; ++i) 
      {
	c = mkcosb(c0, rndm_generator);
	e1 = sqrt(s4);
	pz1 = e1 * c;
	e2 = e1;
	pz2 = -pz1;
	pt = sqrt((1. - c * c) * e1 * e2);
	lorent_jet(eph1p,eph2p,e1,pz1);
	lorent_jet(eph1p,eph2p,e2,pz2);
/* scd-1 */
	if (jet_parameter_.get_d_spectrum() == 5)  q2_d=pt*pt;	  
	else q2_d=q2;
	
	if((q2_1<q2_d)&&(q2_2<q2))
	  {
	    store_jet(pz1,pz2,eph1,eph2,pt,h,3,switches, rndm_generator);
	    jet_results_.increment_sigma(1, h/switches.get_jet_ratio());
	}
      }
    //    return 0;
} 

JET_FLOAT MINIJETS::hadcross(JET_FLOAT eph1,JET_FLOAT q2_1,JET_FLOAT eph2,JET_FLOAT q2_2, JET_FLOAT lumi, SWITCHES& switches, RNDM& rndm_generator)
{
    int nf,i;
    JET_FLOAT x1,x2,q2,e1,e2;
    JET_FLOAT lnx = 0.0;
    JET_FLOAT q1q1,q1q2,gg,gq,qg,qsum1,qsum2,cmax,part1[7],part2[7],alphas;
    JET_FLOAT s[8],fact,e0,pz1,pz2,pt,tmp;
   const JET_FLOAT sigma0=0.4*3.9e5,eps=1e-300;

    if(jet_select_x_)
      {
	lnx=eph1*eph2;
	if (lnx < jet_parameter_.get_pstar2()) return 0.0;
	lnx=log(jet_parameter_.get_pstar2()/lnx);
	x1=exp(lnx*rndm_generator.rndm());
	x2=exp(lnx*rndm_generator.rndm());
      }
    else
      {
	x1=rndm_generator.rndm();
	x2=rndm_generator.rndm();
    }
    e1=eph1*x1;
    e2=eph2*x2;
	eph1-=e1;
	eph2-=e2;
    q2=e1*e2;
    if(q2 <= jet_parameter_.get_pstar2()) return 0.0;
    nf=4;
    if (q2 < jet_parameter_.get_q2_1()) nf=3;
    if (q2 > jet_parameter_.get_q2_2()) nf=5;
    hadrons(x1,x2,q2,nf,part1,part2,alphas);

    gg=part1[0]*part2[0];

    gq=0.0;
    qg=0.0;
    for (i=1;i<=nf;i++) 
      {
	gq+=part2[i];
	qg+=part1[i];
      }
    gq*=part1[0];
    qg*=part2[0];
    q1q1=0.0;
    qsum1=0.0;
    qsum2=0.0;
    for (i=1;i<=nf;i++) 
      {
	q1q1+=part1[i]*part2[i];
	qsum1+=part1[i];
	qsum2+=part2[i];
      }
    gq*=2.0;
    qg*=2.0;
    q1q1*=2.0;
    qsum1*=2.0;
    qsum2*=2.0;
    q1q2=qsum1*qsum2-2.0*q1q1;

    fact=sigma0*alphas*alphas/q2*lumi*1e-37*switches.get_jet_ratio();
    if (fact<eps) return 0.0;
    if(jet_select_x_)
      {
	tmp=x1*x2*lnx*lnx;
	fact*=tmp;
      }
    if((q2<q2_1)||(q2<q2_2)) return 0.0;
    gg*=fact;
    gq*=fact;
    qg*=fact;
    q1q1*=fact;
    q1q2*=fact;
    
    e0=sqrt(q2);
    cmax=sqrt(1.0-jet_parameter_.get_pstar2()/q2);

    prepareToStoreHadCross(0, s, q1q2, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt, s[0],4, switches, rndm_generator);

    prepareToStoreHadCross(1, s, 0.5*q1q1, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt, s[1],5, switches,  rndm_generator);

    prepareToStoreHadCross(2, s, (nf-1)*q1q1, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt, s[2],6, switches, rndm_generator);

    prepareToStoreHadCross(3, s, q1q1, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt, s[3],7, switches, rndm_generator);

    prepareToStoreHadCross(4, s, 0.5*q1q1, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt,s[4],8, switches, rndm_generator);

    prepareToStoreHadCross(5, s, nf*gg, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
   store_jet(pz1,pz2,eph1,eph2,pt,s[5],9, switches, rndm_generator);


    prepareToStoreHadCross(6, s, gq+qg, cmax, e0,e1,e2, pz1, pz2, pt, rndm_generator);
   store_jet(pz1,pz2,eph1,eph2,pt,s[6]*qg/(qg+gq),10, switches, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt,s[6]*gq/(qg+gq),-10, switches, rndm_generator);


    prepareToStoreHadCross(7, s, 0.5*gg, cmax, e0, e1,e2, pz1, pz2, pt, rndm_generator);
    store_jet(pz1,pz2,eph1,eph2,pt, s[7],11, switches, rndm_generator);

    tmp=(s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7])/switches.get_jet_ratio();
    if(tmp>1.0){
      std::cerr << "MINIJETS:: " << x1 << " " << x2 << " " << lnx << " " << tmp  << std::endl;
    }
    return tmp;
}


void MINIJETS::lorent_jet(JET_FLOAT e1,JET_FLOAT e2,JET_FLOAT& e,JET_FLOAT& pz)
{
    double beta, eold, gam;

    beta = -((double)e1 - (double)e2) / ((double)e1 + (double)e2);
    gam = (double)1.0 / sqrt((double)1.0 - beta * beta);
    eold = e;
    e = gam * (e - beta * pz);
    pz = gam * (pz - beta * eold);
}


void MINIJETS::hadrons_grv(JET_FLOAT x1,JET_FLOAT x2,JET_FLOAT q2,int flavours, JET_FLOAT *parton1,JET_FLOAT *parton2,JET_FLOAT& alphas)
{
    parton1[0] = ALPHA_EM*grvpar_1_.grvgl(x1,q2);
    parton1[1] = ALPHA_EM*grvpar_1_.grvdl(x1,q2);
    parton1[2] = ALPHA_EM*grvpar_1_.grvul(x1,q2);
    parton1[3] = ALPHA_EM*grvpar_1_.grvsl(x1,q2);
    parton1[4] = ALPHA_EM*grvpar_1_.grvcl(x1,q2);
    parton1[5] = ALPHA_EM*grvpar_1_.grvbl(x1,q2);
    parton1[6] = 0.0;
    parton2[0] = ALPHA_EM*grvpar_1_.grvgl(x2,q2);
    parton2[1] = ALPHA_EM*grvpar_1_.grvdl(x2,q2);
    parton2[2] = ALPHA_EM*grvpar_1_.grvul(x2,q2);
    parton2[3] = ALPHA_EM*grvpar_1_.grvsl(x2,q2);
    parton2[4] = ALPHA_EM*grvpar_1_.grvcl(x2,q2);
    parton2[5] = ALPHA_EM*grvpar_1_.grvbl(x2,q2);
    parton2[6] = 0.0;
    switch(flavours){
    case 3:
    alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda3_2()));
    break;
    case 4:
    alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda4_2()));
    break;
    case 5:
    alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda5_2()));
    break;
    }
}


void MINIJETS::hadrons_dg(JET_FLOAT x1,JET_FLOAT x2,JET_FLOAT q2,int flavours, JET_FLOAT *parton1,JET_FLOAT *parton2,JET_FLOAT& alphas)
{

JET_FLOAT alpha=ALPHA_EM;
  JET_FLOAT tau,ans,bns,cns,dns,ens,as,bs,cs,ds,es,ag,bg,cg,qns,qs,q13,q23,gg;
  JET_FLOAT xp,xh1,xh2,xh3;


  tau=log(q2*LAMBDA2_i);
/*  *alphas=TWELVE_PI/((33.0-2.0*flavours)*tau);*/
  tau=log(tau);
  switch(flavours-2)
    {
    case 1:
      alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda3_2()));
      ans= helper_function(tau, ANS1);
      bns= helper_function(tau, BNS1);
      cns= helper_function(tau, CNS1);
      dns= helper_function(tau, DNS1);
      ens= helper_function(tau, ENS1);
      as=  helper_function(tau, AS1);
      bs=  helper_function(tau, BS1);
      cs=  helper_function(tau, CS1);
      ds=  helper_function(tau, DS1);
      es=  helper_function(tau, ES1);
      ag=  helper_function(tau, AG1);
      bg=  helper_function(tau, BG1);
      cg=  helper_function(tau, CG1);

      xp=1.0-x1;
      if (xp<=0.0){
	parton1[0]=0.0;
	parton1[1]=0.0;
	parton1[2]=0.0;
	parton1[3]=0.0;
	parton1[4]=0.0;
	parton1[5]=0.0;
	parton1[6]=0.0;
      }
      else{
	xh1=x1*x1+xp*xp;
	xh2=log(xp);
	qns=xh1/(ans-bns*xh2)+cns*pow(x1,dns-1.0)*pow(xp,ens);
	qs=ESENS1*xh1/(as-bs*xh2)+cs*pow(x1,ds-1.0)*pow(xp,es);
	gg=alpha*(ag*pow(x1,bg-1.0)*pow(xp,cg));
	q13=alpha*SIXTH*(qs-4.5*qns);
	q23=alpha*SIXTH*(qs+9.0*qns);
	parton1[0]=gg;
	parton1[1]=q13;
	parton1[2]=q23;
	parton1[3]=q13;
	parton1[4]=0.0;
	parton1[5]=0.0;
	parton1[6]=0.0;
      }
      xp=1.0-x2;
      if (xp<=0.0){
	parton2[0]=0.0;
	parton2[1]=0.0;
	parton2[2]=0.0;
	parton2[3]=0.0;
	parton2[4]=0.0;
	parton2[5]=0.0;
	parton2[6]=0.0;
      }
      else{
	xh1=x2*x2+xp*xp;
	xh2=log(xp);
	qns=xh1/(ans-bns*xh2)+cns*pow(x2,dns-1.0)*pow(xp,ens);
	qs=ESENS1*xh1/(as-bs*xh2)+cs*pow(x2,ds-1.0)*pow(xp,es);
	gg=alpha*(ag*pow(x2,bg-1.0)*pow(xp,cg));
	q13=alpha*SIXTH*(qs-4.5*qns);
	q23=alpha*SIXTH*(qs+9.0*qns);
	parton2[0]=gg;
	parton2[1]=q13;
	parton2[2]=q23;
	parton2[3]=q13;
	parton2[4]=0.0;
	parton2[5]=0.0;
	parton2[6]=0.0;
      }
      break;
    case 2:
      alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda4_2()));

      ans= helper_function(tau, ANS2);
      bns= helper_function(tau, BNS2);
      cns= helper_function(tau, CNS2);
      dns= helper_function(tau, DNS2);
      ens= helper_function(tau, ENS2);
      as=  helper_function(tau, AS2);
      bs=  helper_function(tau, BS2);
      cs=  helper_function(tau, CS2);
      ds=  helper_function(tau, DS2);
      es=  helper_function(tau, ES2);
      ag=  helper_function(tau, AG2);
      bg=  helper_function(tau, BG2);
      cg=  helper_function(tau, CG2);

      xp=1.0-x1;
      if (xp<=0.0){
	parton1[0]=0.0;
	parton1[1]=0.0;
	parton1[2]=0.0;
	parton1[3]=0.0;
	parton1[4]=0.0;
	parton1[5]=0.0;
	parton1[6]=0.0;
      }
      else{
	xh1=x1*x1+xp*xp;
	xh2=log(xp);
	qns=xh1/(ans-bns*xh2)+cns*pow(x1,dns-1.0)*pow(xp,ens);
	qs=ESENS2*xh1/(as-bs*xh2)+cs*pow(x1,ds-1.0)*pow(xp,es);
	gg=alpha*ag*pow(x1,bg-1.0)*pow(xp,cg);
	q13=alpha*EIGTH*(qs-6.0*qns);
	q23=alpha*EIGTH*(qs+6.0*qns);
	parton1[0]=gg;
	parton1[1]=q13;
	parton1[2]=q23;
	parton1[3]=q13;
	parton1[4]=q23;
	parton1[5]=0.0;
	parton1[6]=0.0;
      }
      xp=1.0-x2;
      if (xp<=0.0){
	parton2[0]=0.0;
	parton2[1]=0.0;
	parton2[2]=0.0;
	parton2[3]=0.0;
	parton2[4]=0.0;
	parton2[5]=0.0;
	parton2[6]=0.0;
      }
      else{
	xh1=x2*x2+xp*xp;
	xh2=log(xp);
	qns=xh1/(ans-bns*xh2)+cns*pow(x2,dns-1.0)*pow(xp,ens);
	qs=ESENS2*xh1/(as-bs*xh2)+cs*pow(x2,ds-1.0)*pow(xp,es);
	gg=alpha*ag*pow(x2,bg-1.0)*pow(xp,cg);
	q13=alpha*EIGTH*(qs-6.0*qns);
	q23=alpha*EIGTH*(qs+6.0*qns);
	parton2[0]=gg;
	parton2[1]=q13;
	parton2[2]=q23;
	parton2[3]=q13;
	parton2[4]=q23;
	parton2[5]=0.0;
	parton2[6]=0.0;
      }
      break;
    case 3:
      alphas=TWELVE_PI/((33.0-2.0*flavours)*log(q2/jet_parameter_.get_lambda5_2()));

      ans= helper_function(tau, ANS3);
      bns= helper_function(tau, BNS3);
      cns= helper_function(tau, CNS3);
      dns= helper_function(tau, DNS3);
      ens= helper_function(tau, ENS3);
      as=  helper_function(tau, AS3);
      bs=  helper_function(tau, BS3);
      cs=  helper_function(tau, CS3);
      ds=  helper_function(tau, DS3);
      es=  helper_function(tau, ES3);
      ag=  helper_function(tau, AG3);
      bg=  helper_function(tau, BG3);
      cg=  helper_function(tau, CG3);

      xp=1.0-x1;
      if (xp<=0.0){
	parton1[0]=0.0;
	parton1[1]=0.0;
	parton1[2]=0.0;
	parton1[3]=0.0;
	parton1[4]=0.0;
	parton1[5]=0.0;
	parton1[6]=0.0;
      }
      else{
	xh1=x1*x1+xp*xp;
	xh2=log(xp);
	xh3=log(x1);
	qns=xh1/(ans-bns*xh2)+cns*exp(xh3*(dns-1.0))*exp(xh2*ens);
	qs=ESENS3*xh1/(as-bs*xh2)+cs*exp(xh3*(ds-1.0))*exp(xh2*es);
	gg=alpha*ag*exp(xh3*(bg-1.0))*exp(xh2*cg);
	q13=alpha*TENTH*(qs-5.0*qns);
	q23=alpha*TENTH*(qs+7.5*qns);
	parton1[0]=gg;
	parton1[1]=q13;
	parton1[2]=q23;
	parton1[3]=q13;
	parton1[4]=q23;
	parton1[5]=q13;
	parton1[6]=0.0;
      }
      xp=1.0-x2;
      if (xp<=0.0){
	parton2[0]=0.0;
	parton2[1]=0.0;
	parton2[2]=0.0;
	parton2[3]=0.0;
	parton2[4]=0.0;
	parton2[5]=0.0;
	parton2[6]=0.0;
      }
      else{
	xh1=x2*x2+xp*xp;
	xh2=log(xp);
	xh3=log(x2);
	qns=xh1/(ans-bns*xh2)+cns*exp(xh3*(dns-1.0))*exp(xh2*ens);
	qs=ESENS3*xh1/(as-bs*xh2)+cs*exp(xh3*(ds-1.0))*exp(xh2*es);
	gg=alpha*ag*exp(xh3*(bg-1.0))*exp(xh2*cg);
	q13=alpha*TENTH*(qs-5.0*qns);
	q23=alpha*TENTH*(qs+7.5*qns);
	parton2[0]=gg;
	parton2[1]=q13;
	parton2[2]=q23;
	parton2[3]=q13;
	parton2[4]=q23;
	parton2[5]=q13;
	parton2[6]=0.0;
      }
      break;
    }
}


void MINIJETS::store_jet(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT eph1,JET_FLOAT eph2, JET_FLOAT pt,JET_FLOAT h,int event, SWITCHES& switches, RNDM& rndm_generator)
{
  int i, num;
  update_statistics_arrays(pz1,pz2,pt,h);
  if (!switches.get_jet_store()) return;
  num=(int)floor(h);
  h-=num;
  if(h>rndm_generator.rndm()) num++;  
  for (i=0;i<num;i++){
    jetfile_->save_jet(eph1,eph2,pz1,pz2,pt,event); 
  }
   
}







void MINIJETS_PYTHIA::initPythia()
{
  jet_spline0_.spline_init( std::string("pythia0.ini") );
  jet_spline1_.spline_init( std::string("pythia1.ini") );
  jet_spline2_.spline_init( std::string("pythia2.ini") );
}


void MINIJETS_PYTHIA::mkjll_(const PAIR_PARAMETER& pair_parameter,float e1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
  float gam2i;
  float rphot;
  //float eph1,eph2,q2_1,q2_2,wgt1,wgt2;
  double pstar2 = jet_parameter_.get_pstar2();
  if (e1*e2 <= pstar2) return;
  int d_spectrum = jet_parameter_.get_d_spectrum();
  int r_spectrum = jet_parameter_.get_r_spectrum();
  //float jet_ratio = switches.get_jet_ratio();
  gam2i = pstar2/(e1*e2);
  float jetrqvd =  pair_parameter.jet_requiv(gam2i, d_spectrum);
  float jetrqvr =  pair_parameter.jet_requiv(gam2i, r_spectrum);
  rphot = jetrqvd*jetrqvd;


  mkj_pythia12(pair_parameter, d_spectrum, d_spectrum, rphot, gam2i, e1, e2, flum, switches.get_jet_ratio(), jet_spline0_, 0, 0, rndm_generator);
  rphot = jetrqvd*jetrqvr;
  mkj_pythia12(pair_parameter,r_spectrum, d_spectrum, rphot, gam2i, e1, e2, flum, switches.get_jet_ratio(), jet_spline1_, 1, 1,  rndm_generator);
  mkj_pythia12(pair_parameter,d_spectrum, r_spectrum, rphot, gam2i, e1, e2, flum, switches.get_jet_ratio(), jet_spline1_, 1, -1, rndm_generator);
  rphot = jetrqvr*jetrqvr;
  mkj_pythia12(pair_parameter, r_spectrum, r_spectrum,rphot, gam2i, e1, e2, flum, switches.get_jet_ratio(), jet_spline2_, 2, 2, rndm_generator);

}

void MINIJETS_PYTHIA::mkjbh1_(const PAIR_PARAMETER& pair_parameter, float eph1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
   float gam2i;
  float rphot;
  // This routine produces the minijets from gamma e collision 

  if (eph1*e2 <= jet_parameter_.get_pstar2()) return;
   
  gam2i = jet_parameter_.get_pstar2() / (eph1 * e2);
  int d_spectrum = jet_parameter_.get_d_spectrum();
  int r_spectrum = jet_parameter_.get_r_spectrum();
  
  rphot = pair_parameter.jet_requiv(gam2i,d_spectrum);
  mkj_pythia1(pair_parameter, d_spectrum, rphot,  gam2i,eph1, e2, flum, switches.get_jet_ratio(), jet_spline0_, 0, 0, rndm_generator);

  mkj_pythia1(pair_parameter,d_spectrum, rphot,  gam2i,eph1, e2, flum, switches.get_jet_ratio(), jet_spline1_, 1, 0, rndm_generator);

  rphot = pair_parameter.jet_requiv(gam2i,r_spectrum);

  mkj_pythia1(pair_parameter, r_spectrum, rphot, gam2i,eph1, e2, flum, switches.get_jet_ratio(), jet_spline1_, 1, -1, rndm_generator);

  mkj_pythia1(pair_parameter, r_spectrum,rphot,  gam2i,eph1, e2, flum, switches.get_jet_ratio(), jet_spline2_, 2, 2, rndm_generator);
}

void MINIJETS_PYTHIA::mkjbh2_(const PAIR_PARAMETER& pair_parameter, float e1,float eph2, float flum, SWITCHES& switches, RNDM& rndm_generator)
{
  float gam2i;
  // static float eph1,q2,wgt;

  if (e1 * eph2 <= jet_parameter_.get_pstar2()) return;
    
  gam2i = jet_parameter_.get_pstar2() / (e1 * eph2);
  int d_spectrum = jet_parameter_.get_d_spectrum();
  int r_spectrum = jet_parameter_.get_r_spectrum();
float rphotd = pair_parameter.jet_requiv(gam2i, d_spectrum);
float  rphotr = pair_parameter.jet_requiv(gam2i,r_spectrum);

 mkj_pythia2(pair_parameter,d_spectrum, rphotd, gam2i, eph2,  e1, flum, switches.get_jet_ratio(), jet_spline0_, 0, 0, rndm_generator); 
 mkj_pythia2(pair_parameter,r_spectrum, rphotr, gam2i, eph2,  e1, flum, switches.get_jet_ratio(), jet_spline1_, 1, 0, rndm_generator); 
     mkj_pythia2(pair_parameter,d_spectrum, rphotd, gam2i, eph2,  e1, flum, switches.get_jet_ratio(), jet_spline1_, 1, -1, rndm_generator); 
     mkj_pythia2(pair_parameter,r_spectrum, rphotr, gam2i, eph2,  e1, flum, switches.get_jet_ratio(), jet_spline2_, 2, 2, rndm_generator); 
}




// This routine produces the minijets from e+e- collision 

void MINIJETS_PYTHIA::mkjbw_(float eph1,float eph2,float flum, SWITCHES& switches, RNDM& rndm_generator)
{
      float jet_ratio = switches.get_jet_ratio();
      make_jet(0, eph1,0.0,eph2,0.0,flum, jet_ratio, jet_spline0_, 0, rndm_generator);
      make_jet(1, eph1,0.0,eph2,0.0,flum,jet_ratio, jet_spline1_, 0, rndm_generator);
      make_jet(1, eph1,0.0,eph2,0.0,flum,jet_ratio, jet_spline1_,  -1, rndm_generator);
      make_jet(2,eph1,0.0,eph2,0.0,flum, jet_ratio, jet_spline2_, 2, rndm_generator);
}


void MINIJETS_PYTHIA::mkj_pythia1(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph1,float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator)
{
  // minijets from gamma e collision 
  int k;
  float eph2, q2, wgt;
  int niter = update_niter(rphot,rndm_generator);
//    niter = (int) rphot;
//  if ( rndm_generator.rndm_jet() <= (rphot-niter) ) ++niter;
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener2,spectrum,eph2, q2, wgt, rndm_generator);
      make_jet(index_of_sigma,eph1,0.0,eph2,q2,flum*wgt,ratio,jet_spline, optionStoreJet, rndm_generator);
    }
}

void MINIJETS_PYTHIA::mkj_pythia2(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph2,float  ener1, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator)
{
  // minijets from e gamma collision
  int k;
  float eph1, q2, wgt;
  int niter = update_niter(rphot,rndm_generator);
//   niter = (int) rphot;
//  if ( rndm_generator.rndm_jet() <= (rphot-niter) ) ++niter;
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener1,spectrum,eph1, q2, wgt, rndm_generator);
      make_jet(index_of_sigma,eph1,q2, eph2,0.0,flum*wgt,ratio,jet_spline, optionStoreJet, rndm_generator);
    }
}


void MINIJETS_PYTHIA::mkj_pythia12(const PAIR_PARAMETER& pair_parameter, int spectrum1, int spectrum2,float rphot, float gam2i,float ener1, float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator)
{
  int k;
  float eph1, q2_1, wgt1;
  float eph2, q2_2, wgt2;
  int niter = update_niter(rphot,rndm_generator);
// niter = (int)floor(rphot);
//  if ( rndm_generator.rndm_jet() <= (rphot-niter) ) ++niter;
  for (k = 0; k < niter; k++) 
    {
      pair_parameter.jet_equiv(gam2i,ener1,spectrum1,eph1,q2_1,wgt1, rndm_generator);
      pair_parameter.jet_equiv(gam2i,ener2,spectrum2,eph2,q2_2,wgt2, rndm_generator);
      make_jet(index_of_sigma,eph1,q2_1,eph2,q2_2,flum*wgt1*wgt2,ratio,jet_spline, optionStoreJet, rndm_generator);
    }
}

bool MINIJETS_PYTHIA::deltaSigma(float e1,float e2,float q2_1,float q2_2, float flum, const SPLINE& jet_spline, float& delta) const
  {
    float s,ecm;
    s=e1*e2;
    if ((q2_1>s)||(q2_2>s)) return false;
    ecm=sqrt(4.0*s);
    delta =  jet_spline.spline_int(ecm)*flum;
    return true;
  }


void MINIJETS_PYTHIA::store_jet(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT eph1,JET_FLOAT eph2, JET_FLOAT pt,JET_FLOAT h,int event, SWITCHES& switches, RNDM& rndm_generator)
{
  int i, num;
  update_statistics_arrays(pz1,pz2,pt,h);
  if (!switches.get_jet_store()) return;
  num=(int)floor(h);
  h-=num;
  if(h>rndm_generator.rndm()) num++;   
  for (i=0;i<num;i++) jetfile_->save_jet(eph1,eph2,event);
}

void MINIJETS_PYTHIA::storeJet( float e1,float e2,float sigma, float jet_ratio, RNDM& rndm_generator, int optionStoreJet) const 
   {
    int n;
    sigma *= jet_ratio;
    n = (int)floor(sigma);
    sigma -= (float)n;
    // why do this call?
    std::cout << " MINIJETS::storeJet : look if the call to rndm_generator is useful??? " << std::endl;
    if (rndm_generator.rndm_jet()<sigma) n++;
    jetfile_->save_jet(e1,e2,optionStoreJet);
   }
