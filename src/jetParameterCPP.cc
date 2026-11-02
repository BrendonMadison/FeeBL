#include "jetParameterCPP.h"
#include "physconst.h"
#include <cmath>

// this method is part of the old init_jet
void JET_PARAMETER::init(float s,float ptmin,int iparam)
{
    d_spectrum_=5;
    r_spectrum_=2;
    pstar2_=ptmin*ptmin;
    ptmin_ = ptmin;
    lns4_=log(0.25*s/(EMASS*EMASS));
    q2_1_=20.0;
    q2_2_=200.0;
    // set lambda4 
    switch(iparam)
      {
      case 1:
	lambda4_2_=0.4*0.4;
	break;
      case 2:
	lambda4_2_=0.2*0.2;
	break;
      }
    iparam_=iparam;
    lambda3_2_=exp((2.0*log(q2_1_)
				 +25.0*log(lambda4_2_))/27.0);
    lambda5_2_=exp((-2.0*log(q2_2_)
				 +25.0*log(lambda4_2_))/23.0);
    //     if (!jet_pythia) select_x_ = jet_select;
}
