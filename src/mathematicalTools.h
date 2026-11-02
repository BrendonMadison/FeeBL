#ifndef MATHEMATICALTOOLS_SEEN
#define  MATHEMATICALTOOLS_SEEN
#include <iostream>
#include "besselCPP.h"
#include "typeDefs.h"
#include "rndmCPP.h"
#include "physconst.h"

class TOOLS 
{

  public : 
    
    inline static int check_power_of_2(int aTester )
    {
      int ntest=0;
      int nh = aTester;
      while (nh%2 == 0) 
	{
	  nh = nh >> 1;
	  ntest++;
	}
      if (nh>1)
	{
	  ntest = 0;
	}
      return ntest;
    }
  
  inline static int nearest_power_of_2(float x)
    {
      int value = (int)x;
      int index = 2;
      while (index < value) index *= 2;
      int index2 = index/2; 
      return (value - index2) < index - value ? index2 : index;
    }
  
  inline static void equal_newton(double (*fint)(double),double (*fdiff)(double),
				  double xmin,double xmax,double y,double& x)
    {
      //    double eps=1e-6,tiny=1e-20;
      double eps=1e-6;
      //    double ytry,xtry,dfdx;
      double ytry,xtry;
      int i=0;
      xtry = x;
      ytry=(*fint)(xtry);
      while (fabs(ytry-y)>(fabs(ytry)+fabs(y))*eps
	     && (xmax-xmin)>eps) {
	i++;
	xtry-=(ytry-y)/(*fdiff)(xtry);
	if ((xtry>=xmax)||(xtry<=xmin)) {
	  xtry=0.5*(xmax+xmin);
	}
	ytry=(*fint)(xtry);
	if(ytry<y) {
	  xmin=xtry;
	}
	else {
	  xmax=xtry;
	}
      }
      x = xtry;
    }
  
  
  inline static void equal_newton(double (*fint)(double, double),double (*fdiff)(double, double), double xmin,double xmax,double y, double scdgam2i, double& x)
    {
      //    double eps=1e-6,tiny=1e-20;
      double eps=1e-6;
      //    double ytry,xtry,dfdx;
      double ytry,xtry;
      int i=0;
      xtry = x;
      ytry=(*fint)(xtry, scdgam2i);
      while (fabs(ytry-y)>(fabs(ytry)+fabs(y))*eps
	     && (xmax-xmin)>eps) {
	i++;
	xtry-=(ytry-y)/(*fdiff)(xtry, scdgam2i);
	if ((xtry>=xmax)||(xtry<=xmin)) {
	  xtry=0.5*(xmax+xmin);
	}
	ytry=(*fint)(xtry, scdgam2i);
	if(ytry<y) {
	  xmin=xtry;
	}
	else {
	  xmax=xtry;
	}
      }
      x = xtry;
    }  
  
  // MODIFIED BESSEL FUNCTION K(1/3,X) (from Yokoya, corresponds to L=1 in CAIN)
  static double K13(double x);
  static double Ki13(double x);
  
  static void rotmat(double angle, double axis[3], double r[3][3]);
  
};

#endif
