#ifndef GRV_SEEN
#define GRV_SEEN

#include <cmath>

/* Common Block Declarations */

class GRVPAR 
{
    double alp_, bet_, a_, b_, ba_, bb_, c_, d_, e_, ep_, sp_;



/* Table of constant values */

inline double fl(double x,double s)
{
  /*   double sqrt(), log(), exp(double); */
/* ****************************************************************** */
/* * Functional form of the distribution function of light partons. * */
/* * S = log( log(Q**2/.232^2) / log(.25/.232^2) ) (LO),            * */
/* * S = log( log(Q**2/.248^2) / log(.3/.248^2) )  (HO).            * */
/* ****************************************************************** */
  
  return (std::pow(x,a_) * (ba_ + bb_ * std::sqrt(x) + c_ * std::pow(x,b_))
	  + std::pow(s,alp_) * 
	  std::exp(-e_ + std::sqrt(-ep_ * std::pow(s,bet_) *
				   std::log(x)))) * std::pow(1.0-x,d_) / x;
} 

inline double fh(double x,double s)
{
  // System generated locals 

  // Builtin functions 

  /* ************************************************* */
  /* * Same as FL, but for the heavy s, c, b quarks. * */
  /* ************************************************* */
  if (s - sp_ < 0.0) {
    return 0.0;
  }
  return ((s - sp_) * std::pow(x,a_) * 
	  (ba_ + bb_ * std::sqrt(x) + c_ * std::pow(x,b_)) + 
	  std::pow(s-sp_,alp_) * 
	  std::exp(-e_ + std::sqrt(-ep_ * std::pow(s,bet_) * std::log(x)))) *
    std::pow(1.0-x,d_) / x;
}

double grvuh(double x,double q2);

double grvdh(double x,double q2);

double grvgh(double x,double q2);
 /* grvgh */

double grvsh(double x,double q2);

double grvch(double x,double q2);

double grvbh(double x,double q2);

double grvuh0(double x,double q2);

double grvdh0(double x,double q2);

double grvgh0(double x,double q2);

double grvsh0(double x,double q2);

double grvch0(double x,double q2);

double grvbh0(double x,double q2);


public:

double grvul(double x,double q2);

double grvdl(double x,double q2);

double grvgl(double x,double q2);

double grvsl(double x,double q2);

double grvcl(double x,double q2);

double grvbl(double x,double q2);


};


#endif
