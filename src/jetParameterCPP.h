#ifndef JETPARAMETER_SEEN
#define JETPARAMETER_SEEN

#include <cstdio>
#include "typeDefs.h"
#include "define.h"

class JET_PARAMETER
{
  //double ebeam_;
  double ptmin_,lns4_,pstar2_,q2_1_,q2_2_;
  double lambda3_2_,lambda4_2_,lambda5_2_;
  //   int select_x_;
  int d_spectrum_, r_spectrum_,iparam_;


 public:

 JET_PARAMETER(): ptmin_(0.0),lns4_(0.0),pstar2_(0.0),q2_1_(0.0),
		  q2_2_(0.0),lambda3_2_(0.0),lambda4_2_(0.0),lambda5_2_(0.0),
		  d_spectrum_(0),r_spectrum_(0),iparam_(0){;}

  void init(float s,float ptmin,int iparam);

  inline double get_pstar2() const {return pstar2_;};
  inline int get_d_spectrum() const {return d_spectrum_;};
  inline int get_r_spectrum() const {return r_spectrum_;};
  inline double get_q2_1() const {return q2_1_;}
  inline double get_q2_2() const {return q2_2_;}
  // inline int get_select_x() const {return select_x_;}
  inline double get_lambda3_2() const {return lambda3_2_;}
  inline double get_lambda4_2() const {return lambda4_2_;}
  inline double get_lambda5_2() const {return lambda5_2_;}
  inline int get_iparam() const {return iparam_;}
};


#endif
