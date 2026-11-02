#ifndef MINIJETS_SEEN
#define MINIJETS_SEEN
#include <vector>
#include <cmath>
#include "typeDefs.h"
#include "fileInputOutput.h"
#include "splineCPP.h"
#include "resultsCPP.h"
#include "switchesCPP.h"
#include "jetParameterCPP.h"
#include "mathconst.h"
#include "physconst.h"
#include "mathematicalTools.h"
#include "physicalTools.h"
#include "abstractParticle.h"
#include "grvCPP.h"
#include "pairsCPP.h"

/**************************************************************************/
/*                                                                        */
/* Subroutines to efficiently satisfy probability distributions           */
/*                                                                        */
/**************************************************************************/

class NEWTON
{
  double (*f_)(double),(*df_)(double);
  double ymin_,dy_i_;
  int n_;
  std::vector<double> x_;

 public : 

 NEWTON():f_(NULL),df_(NULL),ymin_(0.0),dy_i_(0.0),n_(0) {;}

  ~NEWTON() {;}

  void make_newton(double (*f)(double),double (*df)(double),double xmin,double xmax,int n)
  {
    double dy;
    int i;
    x_ =  std::vector<double>(n);
    n_=n;
    ymin_=(*f)(xmin);
    dy=((*f)(xmax)-(*f)(xmin))/(double)(n-1);


    dy_i_ = 1.0/dy;
    f_ = f;
    df_ = df;
    for (i=1;i<n-1;i++) {
      TOOLS::equal_newton(f,df,xmin,xmax,ymin_+dy*i,x_[i]);

    }
    x_[0]   = xmin;
    x_[n-1] = xmax;
  }

  void get_angle_sigma(double c0,double& c, double& sigma, RNDM& rndm_generator) const
  {
    double sigma0,y,xmin,xmax,tmp;
    int i;
    sigma0=(*f_)(-c0);
    sigma=(*f_)(c0)-sigma0;


    y=sigma*rndm_generator.rndm()+sigma0;
    tmp=(y-ymin_)*dy_i_;
    i=(int)floor(tmp);
    tmp-=i;
    xmin=x_[i];
    xmax=x_[i+1];
    c=xmin+(xmax-xmin)*tmp;
    TOOLS::equal_newton(f_,df_,xmin,xmax,y,c);
  }

  inline double get_angle(double c0, RNDM& rndm_generator) const
  {
    double c, sigma;
    get_angle_sigma(c0, c, sigma,rndm_generator);
    return c;
  }

};



class ABSTRACT_MINIJETS : public ABSTRACT_IO_CLASS
{

 protected : 


  JET_PARAMETER jet_parameter_;

  JET_RESULTS jet_results_;


  FILE_IN_OUT* jetfile_;


  std::vector<double> v_; 

  // v2_ not used for now
  //std::vector<double> v2_;

  std::vector<JET_FLOAT> pt_;
  std::vector<JET_FLOAT> c_;


  ABSTRACT_MINIJETS() : jetfile_(NULL)  {;}

  ABSTRACT_MINIJETS(float s,float ptmin,int iparam, int /*jet_select*/, std::string jetfileName) : jetfile_(NULL)
    {
      set();
      init_jet_file(s,ptmin,jetfileName);
      jet_parameter_.init(s, ptmin, iparam);
    }


  void set() 
  {
    int n_pt = 11;
    pt_.resize(n_pt);
    pt_[0]=1.6;
    pt_[1]=2.0;
    pt_[2]=2.5;
    pt_[3]=3.2;
    pt_[4]=5.0;
    pt_[5]=8.0;
    pt_[6]=15.0;
    pt_[7]=25.0;
    pt_[8]=40.0;
    pt_[9]=70.0;
    pt_[10]=10000.0;

    int n_c = 1;  
    c_.resize(n_c);
    c_[0]=1.01;


    v_.resize(n_pt*n_c, 0.0);
    // v2_.resize(n_pt*n_c, 0.0);
  }

  inline int update_niter(float rphot, RNDM& rndm_generator)
  {
    int niter;
    niter = (int) rphot;
    if ( rndm_generator.rndm_jet() <= (rphot-niter) ) ++niter;
    return niter;
  }

  void init_jet_file(float s,float ptmin, std::string jetfileName);
  void update_statistics_arrays(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT pt,JET_FLOAT h);

  virtual  void store_jet(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT eph1,JET_FLOAT eph2, JET_FLOAT pt,JET_FLOAT h,int event, SWITCHES& switches, RNDM& rndm_generator) = 0;

 private:
  /// assignment and copy constructor not implemented nor used
  ABSTRACT_MINIJETS& operator=(const ABSTRACT_MINIJETS&);
  ABSTRACT_MINIJETS(ABSTRACT_MINIJETS&);

 public :


  virtual  ~ABSTRACT_MINIJETS()
    {
      if (jetfile_ != NULL) 
	{
	  jetfile_->close();
	  delete jetfile_;
	}
    }



  // This routine produces the minijets from gamma gamma collision (?)
  virtual void mkjll_(const PAIR_PARAMETER& pair_parameter,float e1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator) = 0;

  // This routine produces the minijets from gamma e collision 
  virtual   void mkjbh1_(const PAIR_PARAMETER& pair_parameter, float eph1,float e2,float flum, SWITCHES& switches, RNDM& rndm_generator) =0;

  //This routine produces the minijets from e gamma collision 
  virtual  void mkjbh2_(const PAIR_PARAMETER& pair_parameter, float e1,float eph2, float flum, SWITCHES& switches, RNDM& rndm_generator) =0;

  // This routine produces the minijets from e+e- collision 
  virtual void mkjbw_(float eph1,float eph2,float flum, SWITCHES& switches, RNDM& rndm_generator) =0;


  virtual std::string output_flow() const ;

};


class MINIJETS : public ABSTRACT_MINIJETS
{

 protected : 




  virtual   void store_jet(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT eph1,JET_FLOAT eph2, JET_FLOAT pt,JET_FLOAT h,int event, SWITCHES& switches, RNDM& rndm_generator);

 private : 

  /*
    typedef struct NEWTON
    {
    double (*f_)(double),(*df_)(double);
    double ymin_,dy_i_;
    int n_;
    std::vector<double> x_;
    };
  */
  NEWTON newton_[11];

  GRVPAR grvpar_1_;

  int jet_select_x_;



  inline JET_FLOAT mkcos(JET_FLOAT c0, RNDM& rndm_generator)
  {
    return newton_[8].get_angle(c0,rndm_generator);
  }

  inline JET_FLOAT mkcosb(JET_FLOAT c0, RNDM& rndm_generator)
  {
    return newton_[9].get_angle(c0,rndm_generator);
  }

  void mkj_pythia1(PAIR_PARAMETER& pair_parameter,float rphot, float gam2i,float eph1,float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);
  void mkj_pythia2(const PAIR_PARAMETER& pair_parameter, float rphot, float gam2i,float eph2,float  ener1, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);
  void mkj_pythia12(PAIR_PARAMETER& pair_parameter, float rphot, float gam2i,float ener1, float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);

  void mkj_no_pythia1(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph1,float  ener2, float flum, SWITCHES& switches, RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&));

  void mkj_no_pythia2(const PAIR_PARAMETER& pair_parameter, int spectrum,float rphot, float gam2i,float eph2,float  ener1, float flum, SWITCHES& switches, RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&));


  void mkj_no_pythia12(const PAIR_PARAMETER& pair_parameter,int spectrum1, int spectrum2, float rphot, float gam2i,float ener1,float  ener2, float flum, SWITCHES& switches, RNDM& rndm_generator, void (MINIJETS::*make_jet_n)(float,float ,float,float , float,SWITCHES&, RNDM&));

  void make_jet_0(float eph1,float q2_1,float eph2,float q2_2, float flum,SWITCHES& switches, RNDM& rndm_generator);

  void make_jet_1a(float eph1,float q2_1,float eph2,float q2_2,float flum,SWITCHES& switches, RNDM& rndm_generator);

  void make_jet_1b(float eph1,float q2_1,float eph2,float q2_2, float flum,SWITCHES& switches, RNDM& rndm_generator);


  void make_jet_2(float eph1,float q2_1,float eph2,float q2_2,float flum, SWITCHES& switches,  RNDM& rndm_generator)
  {
    jet_results_.increment_sigma(2,hadcross(eph1,q2_1,eph2,q2_2,flum, switches, rndm_generator));
  } /* make_jet_2 */


  JET_FLOAT hadcross(JET_FLOAT eph1,JET_FLOAT q2_1,JET_FLOAT eph2,JET_FLOAT q2_2, JET_FLOAT lumi, SWITCHES& switches, RNDM& rndm_generator);


  inline void hadrons(JET_FLOAT x1,JET_FLOAT x2,JET_FLOAT q2,int flavours,
		      JET_FLOAT *parton1,JET_FLOAT *parton2,JET_FLOAT& alphas)
  {
    switch(jet_parameter_.get_iparam())
      {
      case 1:
	hadrons_dg(x1,x2,q2,flavours,parton1,parton2,alphas);
	break;
      case 2:
	hadrons_grv(x1,x2,q2,flavours,parton1,parton2,alphas);
	break;
      }
  }

  inline double helper_function(double tau, const double tab[4]) const
  {
    return tab[0]*exp(tau*tab[1])+tab[2]*exp(tau*-tab[3]);
  }
  void hadrons_grv(JET_FLOAT x1,JET_FLOAT x2,JET_FLOAT q2,int flavours, JET_FLOAT *parton1,JET_FLOAT *parton2,JET_FLOAT& alphas);


  void hadrons_dg(JET_FLOAT x1,JET_FLOAT x2,JET_FLOAT q2,int flavours, JET_FLOAT *parton1,JET_FLOAT *parton2,JET_FLOAT& alphas);

  inline void prepareToStoreHadCross(int number, JET_FLOAT* s, JET_FLOAT factor, JET_FLOAT cmax, JET_FLOAT e0, JET_FLOAT e1,JET_FLOAT  e2, JET_FLOAT& pz1, JET_FLOAT& pz2,JET_FLOAT& pt, RNDM& rndm_generator)
  {
    JET_FLOAT ehad1,ehad2;
    JET_FLOAT xkekseksa;
    JET_FLOAT temp;
    newton_[number].get_angle_sigma(cmax,xkekseksa,temp, rndm_generator);
    // get_angle( number,cmax,xkekseksa,temp, rndm_generator);
    s[number]=factor*temp;
    ehad1=e0;
    ehad2=e0;
    pz1=e0*xkekseksa;
    pz2=-pz1;
    pt=e0*sqrt(1.0-xkekseksa*xkekseksa);
    lorent_jet(e1,e2,ehad1,pz1);
    lorent_jet(e1,e2,ehad2,pz2);
  }

  void lorent_jet(JET_FLOAT e1,JET_FLOAT e2,JET_FLOAT& e,JET_FLOAT& pz);

  void initNewton(float s, float ptmin);

 private:
  /// no default constructor (not implemented)
  MINIJETS();

 public:

  virtual  ~MINIJETS();

  MINIJETS(float s,float ptmin,int iparam,int jet_select, std::string jetfileName);

  void make_the_newtons(double xmin,double xmax,int n);

  virtual void mkjll_(const PAIR_PARAMETER& pair_parameter,float e1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator);

  virtual   void mkjbh1_(const PAIR_PARAMETER& pair_parameter, float eph1,float e2,float flum, SWITCHES& switches, RNDM& rndm_generator);

  void mkjbh2_(const PAIR_PARAMETER& pair_parameter, float e1,float eph2, float flum, SWITCHES& switches,RNDM& rndm_generator);
  virtual void mkjbw_(float eph1,float eph2,float flum, SWITCHES& switches, RNDM& rndm_generator);

};

class MINIJETS_PYTHIA : public  ABSTRACT_MINIJETS
{

 protected : 
  virtual  void store_jet(JET_FLOAT pz1,JET_FLOAT pz2,JET_FLOAT eph1,JET_FLOAT eph2, JET_FLOAT pt,JET_FLOAT h,int event, SWITCHES& switches, RNDM& rndm_generator);
 private : 

  SPLINE jet_spline0_,jet_spline1_,jet_spline2_;

  void initPythia();

 MINIJETS_PYTHIA()  : ABSTRACT_MINIJETS() {;}

 public : 

  virtual  ~MINIJETS_PYTHIA()  {;}

 MINIJETS_PYTHIA(float s,float ptmin,int iparam, int jet_select, std::string jetfileName) : ABSTRACT_MINIJETS(s, ptmin, iparam, jet_select, jetfileName)
    {
      initPythia();
    }
  void mkjll_(const PAIR_PARAMETER& pair_parameter,float e1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator);

  // This routine produces the minijets from gamma e collision 
  void mkjbh1_(const PAIR_PARAMETER& pair_parameter, float eph1,float e2, float flum, SWITCHES& switches, RNDM& rndm_generator);

  //This routine produces the minijets from e gamma collision 
  virtual  void mkjbh2_(const PAIR_PARAMETER& pair_parameter, float e1,float eph2, float flum, SWITCHES& switches, RNDM& rndm_generator);

  // This routine produces the minijets from e+e- collision 
  void mkjbw_(float eph1,float eph2,float flum, SWITCHES& switches, RNDM& rndm_generator);

  inline void make_jet(int index, float e1,float e2,float q2_1,float q2_2,float flum,float jet_ratio, const SPLINE& jet_spline, int optionStoreJet, RNDM& rndm_generator)
  {
    float sigma;
    if (!deltaSigma(e1, e2, q2_1, q2_2, flum,jet_spline, sigma)) return;
    jet_results_.increment_sigma(index,sigma);
    storeJet( e1, e2, sigma, jet_ratio,rndm_generator, optionStoreJet);
  }


  void mkj_pythia1(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph1,float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);


  void mkj_pythia2(const PAIR_PARAMETER& pair_parameter, int spectrum, float rphot, float gam2i,float eph2,float  ener1, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);


  void mkj_pythia12(const PAIR_PARAMETER& pair_parameter, int spectrum1, int spectrum2,float rphot, float gam2i,float ener1, float  ener2, float flum,  float ratio,const SPLINE& jet_spline, int index_of_sigma, int optionStoreJet, RNDM& rndm_generator);

  bool deltaSigma(float e1,float e2,float q2_1,float q2_2, float flum, const SPLINE& jet_spline, float& delta) const;


  void storeJet( float e1,float e2,float sigma, float jet_ratio, RNDM& rndm_generator, int optionStoreJet) const;

};

#endif
