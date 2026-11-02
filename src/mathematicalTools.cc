#include "mathematicalTools.h"

#include <iostream>

void TOOLS::rotmat(double angle, double axis[3], double r[3][3])
  // from CAIN
{
  double a0, a1, a2, a3, si;
  a0 = cos(angle/2.);
  si = sin(angle/2.);
  a1 = si*axis[0];
  a2 = si*axis[1];
  a3 = si*axis[2];
  r[0][0] = 1. - 2.0*(a2*a2 + a3*a3);
  r[1][0] = 2.0*(a1*a2 - a3*a0);
  r[2][0] = 2.0*(a1*a3 + a2*a0);
  r[0][1] = 2.0*(a1*a2 + a3*a0);
  r[1][1] = 1. - 2.0*(a3*a3 + a1*a1);
  r[2][1] = 2.0*(a2*a3 - a1*a0);
  r[0][2] = 2.0*(a1*a3 - a2*a0);
  r[1][2] = 2.0*(a2*a3 + a1*a0);
  r[2][2] = 1. - 2.0*(a1*a1 + a2*a2);
}

// MODIFIED BESSEL FUNCTION K(1/3,X) (from Yokoya, corresponds to L=1 in CAIN)
double TOOLS::K13(double x) 
{
  const double A[4] = { 1.687570834 , -1.611154585 , 0.611515182 , -0.256982212 };
  const double B[4] = { 1.253273433 ,  0.671444784 , 0.604094190 ,  0.010909437 };
  const double x0 = 0.546;
  double resu = 0.0;
  double x13, x2,y;
  if ( x <= 0.0 ) 
    {
      std::cerr << " mathematicalTools::K13 : invalid arguments : x = " << x << std::endl;
      return 0.0;
    }
  if ( x <= x0 ) 
    {
      x13 = pow( x, 0.3333333333);
      x2 = x*x;
      resu = (A[0]+A[2]*x2)/x13+(A[1]+A[3]*x2)*x13;
    }
  else
    {
      y = 1.0/x;
      resu = sqrt(y) * ( B[0] + B[1] * y ) / ( 1.0 + y * ( B[2] + y * B[3] ) );
      if ( x >= 130. ) resu = 0.0;
      else resu *= exp(-x);
    }
  return resu;
}

// Integral of  MODIFIED BESSEL FUNCTION K(1/3,X) (from Yokoya, corresponds 
// to L=1 in CAIN)
// integral K(1/3,u).du  over x < u < infinity
double TOOLS::Ki13(double x) 
{
  double x2, x23, y;
  const double A[5] = {1.8136145, -2.5278090, 1.1979670, -0.20490413, 0.058671692 };
  const double B[4] = {1.2531864, 1.6215470, 1.8548547, 0.28146211 };
  const double x0 = 1.2777;
  double resu = 0.0;
  if ( x < 0.0 ) 
    {
      std::cerr << " mathematicalTools::Ki13 : invalid arguments : x = " << x << std::endl;
      return 0.0;
    }
  if ( x < x0 ) 
    {
      if (x == 0.0 ) resu = A[0];
      else
	{
	  x23 = pow( x, 0.666666667);
	  x2 = x*x;
	  resu = A[0] + x23 * (A[1] + A[3] * x2 + x23 * (A[2] + A[4] * x2));
	}
    }
  else
    {
      y = 1.0/x;
      resu = sqrt(y) * ( B[0] + B[1] * y ) / ( 1.0 + y * ( B[2] + y * B[3] ) );
      if ( x >= 130. ) resu = 0.0;
      else resu *= exp(-x);
    }
  return resu;
}
