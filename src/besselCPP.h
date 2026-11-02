#ifndef BESSEL_SEEN
#define BESSEL_SEEN

#include "splineCPP.h"

class BESSEL
{

SPLINE bessel_sp1,bessel_sp2;


 public: 

 BESSEL();

double k23(double x);

double ki13(double x);

};
#endif
