#ifndef PHYSCONST_SEEN
#define PHYSCONST_SEEN
#define ALPHA_EM (1.0/137.035989)
#define EMASS 0.51099906e-3 /* GeV */
#define EMASSeV 0.51099906e+6 /* eV */
#define EMASSMeV 0.51099906 /* MeV */
#define MUMASS 105.658389e-3 /* GeV */
#define MUMASSeV 105.658389e+6 /* eV */
#define ANOMEL 1.15965221e-3 

#define Cvelocity 299792458.0 /* m/s */
#define CmeterPns 0.299792458 /* m/ns */
#define HBARjs 1.05457266e-34 /* J s */
#define HBAR 6.582122e-25 /* GeV s */
#define HBAReVs 6.582122e-16 /* GeV s */
#define RE 2.81794092e-15 /* m */

// longueur d'onde compton divisee par 2.pi
#define LAMBDA_BAR 3.86159323e-13  /* m */
#define LAMBDA_BARnm 0.386159323  /* nm */

#define GeV2J 1.60217733e-10
#define CHARM_MASS 1.6

const double EMASS2 = EMASS*EMASS;

// constantes de synrad_0
  // 4**(1/3)*(1/137)/(Gamma(4/3)*emass) 
  const double PCONST = 25.4e0;


// CONST0 = 9*2**(-1/3)*Gamma(2/3)
const double CONST0 = 9.67287708690521;

// CONST1=(Fine structure constant)*CONST0/[Sqrt(3)*Pi]
const double CONST1 = 1.297210720417891e-02;

// CONST3 =5/(2*sqrt(3))*(Fine structure constant)
const double CONST3 = 0.01053281803;


  const double A1 =9.0267435,A2=14.0612755,A3=0.2636751,            
      B1=12.9947460,B2=44.4076697,B3=38.3037583,B4=1.7430767;                        

// constantes for the G funtions of the beamstrahlung calculations (synrad_0)                      
  const double FA1=-0.8432885317,FA2=0.1835132767,FA3=-0.05279496590,
      FA4=0.01564893164,GA0=0.4999456517,GA1=-0.5853467515,GA2=0.3657833336,
      GA3=-0.06950552838,GA4=0.01918038595,FB0=2.066603927,FB1=-0.5718025331,
      FB2=0.04243170587,FC0=-0.9691386396,FC1=5.651947051,FC2=-0.6903991322,
      GB0=1.8852203645,GB1=-0.5176616313,GB2=0.03812218492,GC0=-0.4915880600,
      GC1=6.1800441958,GC2=-0.6524469236,FD0=1.0174394594,FD1=0.5831679349,
      FE0=0.9949036186,GD0=0.2847316689,GD1=0.5830684600,GE0=0.3915531539;
  const double YA1=0.53520052,YA2=0.30528148,YA3=0.14175015,YA4=0.41841680,
      YB0=0.011920005,YB1=0.20653610,YB2=-0.32809490,YC0=0.33141692e-2,
      YC1=0.19270658,YC2=0.88765651,YD0=148.32871,YD1=674.99117,YE0=-692.23986,
      YE1=-225.45670;

// constant for computing the G-function of the beamstrahlung (synrad_0)
// a quantity is defined xcrit = hbar*omegac/E 
// ( omegac : critic pulsation of the synchrotron radiation; E energy of the 
// electron, in joules)
// xcrit is written as : CCRIT*(Egev)^2/rho
// (rho : bending radius ; Egev is E, but expressed in GeV)
// so that CCRIT takes the value : 
// 3/2* 10^-9 hbar*c/ ( e*(E0gev)^3)
// from the value of omegac (thesis p. 16), with : E0gev : rest energy of
// the electron, in GeV : e : electron charge
// so : CCRIT = 3/2 * c * (hbar/e) 10^-9 / (E0gev)^3
// HBAR  (see above) = hbar/e = hbar expressed in GeV.s
const double CCRIT= 1.5*Cvelocity*HBAReVs/(EMASSMeV*EMASSMeV*EMASSMeV);
//  const double CCRIT=2.22e-6;

// minijets (thesis, p.59)
// (make_jets methods...)

const double FACT_OF_h0 = 6.428172329600001e-36;
const double FACT_OF_h1 = 4.403298045776e-34;
const double FACT_OF_h2 = 5.871064061034666e-34;

const float ECONST0[3] = {0.666666,1.2592593,1.2962963};

const float ECONST1[3] = { .666666,1.111111,1.22222222 };

  const double ANS1[4]={2.285,-0.01526,1.330e3,4.219};
  const double BNS1[4]={6.073,-0.8132,-41.31,3.165};
  const double CNS1[4]={-0.4202,0.01778,0.9216,0.18};
  const double DNS1[4]={-0.08083,0.6346,1.208,0.203};
  const double ENS1[4]={0.05526,1.136,0.9512,0.01163};
  const double AS1[4]={16.69,-0.7916,1.099e3,4.428};
  const double BS1[4]={0.176,0.04794,1.047,0.025};
  const double CS1[4]={-0.0208,0.3386e-2,4.853,0.8404};
  const double DS1[4]={-0.01685,1.353,1.426,1.239};
  const double ES1[4]={-0.1986,1.1,1.136,-0.2779};
  const double AG1[4]={-0.207,0.6158,1.074,0.0};
  const double BG1[4]={-0.1987,0.6257,8.352,5.024};
  const double CG1[4]={5.119,-0.2752,-6.993,2.298};

  const double ANS2[4]={-0.3711,1.061,4.758,-0.01503};
  const double BNS2[4]={-0.1717,0.7815,1.535,0.7067e-2};
  const double CNS2[4]={0.08766,0.02197,0.1096,0.204};
  const double DNS2[4]={-0.8915,0.2857,2.973,0.1185};
  const double ENS2[4]={-0.1816,0.5866,2.421,0.4059};
  const double AS2[4]={-0.1207,1.071,1.977,-0.8625e-2};
  const double BS2[4]={25.0,-1.648,-0.01563,6.438};
  const double CS2[4]={-0.0123,1.162,0.4824,-0.011};
  const double DS2[4]={-0.09194,0.7912,0.6397,2.327};
  const double ES2[4]={0.02015,0.9869,-0.07036,0.01694};
  const double AG2[4]={0.8926e-2,0.6594,0.4766,0.01975};
  const double BG2[4]={0.05085,0.2774,-0.3906,-0.3212};
  const double CG2[4]={-0.2313,0.1382,6.542,0.5162};

  const double ANS3[4]={15.8,-0.9464,-0.5,-0.2118};
  const double BNS3[4]={2.742,-0.7332,0.7148,3.287};
  const double CNS3[4]={0.02917,0.04657,0.1785,0.04811};
  const double DNS3[4]={-0.0342,0.7196,0.7338,0.08139};
  const double ENS3[4]={-0.02302,0.9229,0.5873,-0.79e-4};
  const double AS3[4]={6.734,-1.008,-0.08594,0.07625};
  const double BS3[4]={59.88,-2.983,4.48,0.9686};
  const double CS3[4]={-0.3226e-2,0.8432,0.3616,0.1383e-2};
  const double DS3[4]={-0.03321,0.9475,-0.3198,0.02132};
  const double ES3[4]={0.1059,0.6954,-0.6663,0.3683};
  const double AG3[4]={0.03197,1.018,0.2461,0.02707};
  const double BG3[4]={-0.618e-2,0.9476,-0.6094,-0.01067};
  const double CG3[4]={-0.1216,0.9047,2.653,0.2003e-2};


  const double ESENS1=9.0,ESENS2=10.0,ESENS3=9.16666666667;
  const double LAMBDA2_i=1.0/(0.4*0.4);


#endif
