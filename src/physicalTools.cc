#include "physicalTools.h"
#include "qedbg_photon_stokes.h"
#include <cmath>

#include <iostream>
#include <stdexcept>

BESSEL PHYSTOOLS::bessel_ = BESSEL();

void PHYSTOOLS::mkit(double gam2i,double& c, RNDM& rndm_generator)
{
   JET_FLOAT x,sigma0,beta,y;
    beta=sqrt((double)1.0-gam2i);
    //   scdgam2i_ = gam2i;
    sigma0= pair_ang_i(beta, gam2i);
    y=(2.0*rndm_generator.rndm_pairs()-1.0)*sigma0;
    if (y<=0.0)
      {
	x=-0.5*beta;
	TOOLS::equal_newton(&PHYSTOOLS::pair_ang_i,&PHYSTOOLS::pair_ang_d,-beta,0.0,y,gam2i, x);

      }
    else
      {
	x=0.5*beta;
	TOOLS::equal_newton(&PHYSTOOLS::pair_ang_i,&PHYSTOOLS::pair_ang_d,0.0,beta,y,gam2i, x);
      }
    c= x/beta;
}

float PHYSTOOLS::synrad_spin_flip (float upsilonSingleP,float eng, const TRIDVECTOR& e1, const TRIDVECTOR& e2, const TRIDVECTOR& e3, TRIDVECTOR& polar, float dzOnRadius, std::vector<float>&  photon, RNDM& rndm_generator, std::vector<float>* photonStokes) 
{
  // At fixed local field the quantum parameter is proportional to energy.
  // Recoil changes E between internal radiation substeps, while dz/radius
  // below changes inversely with E.  Keep both quantities consistent with
  // chi = gamma * |F_perp| / E_S (CAIN Sec. 5.8.1).
  const float entry_energy = eng;
  int n,i,j=0;
  float tmp;
  float photener = 0.0;
  // corresponds to A const p.21 in DS's thesis
  tmp=synrad_p0(eng,dzOnRadius);
  n=(int)(tmp*10.0)+1;

  dzOnRadius /= (double)n;
  photon.clear();
  if (photonStokes) photonStokes->clear();
  float xi[3] = {0.0f, 0.0f, 0.0f};
  TRIDVECTOR stokes;
  for (i=0;i<n;i++)
    {
      //      int temp = synrad_0(eng,e2, e3, polar, sokolov, radius_i,dz,photon+j, rndm_generator);
      int temp;

      temp = synrad_0_spin_flip( upsilonSingleP*(eng/entry_energy),eng,e1, e2, e3, polar,stokes,  dzOnRadius,&photener, rndm_generator, photonStokes ? xi : NULL);
      if (temp) 
	{
	  photon.push_back(photener);
	  if (photonStokes) photonStokes->insert(photonStokes->end(), xi, xi + 3);
	  dzOnRadius *= eng/(eng-photon[j]);
	  if (photon[j]<=0.0) 
	    {
	      std::cerr << "warning PHYSTOOLS::synrad " << photon[j] << " " << eng << " " << j << " " << n << std::endl;
	    }
	  
	  eng -= photon[j];
	  j++;
	  // Bound pathological radiation histories before continuing the particle.
	  // The vector is dynamic; this is a physics-domain guard, not its capacity.
	  if (j>=1000)
	    {
	      throw std::overflow_error(
	          "PHYSTOOLS::synrad produced at least 1000 photons for one "
	          "particle");
	    }
	}
      
    }
  return eng;   
}

// return final energy of the radiating particle
float PHYSTOOLS::synrad_no_spin_flip (float upsilonSingleP,float eng, float dzOnRadius,std::vector<float>&  photon, RNDM& rndm_generator, std::vector<float>* photonStokes) 
{
  // At fixed local field the quantum parameter is proportional to energy.
  // Recoil changes E between internal radiation substeps, while dz/radius
  // below changes inversely with E.  Keep both quantities consistent with
  // chi = gamma * |F_perp| / E_S (CAIN Sec. 5.8.1).
  const float entry_energy = eng;
  int n,i,j=0;
  float tmp;
  float photener = 0.0;
  // corresponds to A const p.21 in DS's thesis
  tmp=synrad_p0(eng,dzOnRadius);
  n=(int)(tmp*10.0)+1;

  dzOnRadius /= (double)n;
  photon.clear();
  if (photonStokes) photonStokes->clear();
  float xi[3] = {0.0f, 0.0f, 0.0f};
  for (i=0;i<n;i++)
    {
      int temp;

      temp = synrad_0_no_spin_flip( upsilonSingleP*(eng/entry_energy),eng, dzOnRadius,&photener, rndm_generator, photonStokes ? xi : NULL);
	

      if (temp) 
	{
	  photon.push_back(photener);
	  if (photonStokes) photonStokes->insert(photonStokes->end(), xi, xi + 3);
	  dzOnRadius *= eng/(eng-photon[j]);
	  if (photon[j]<=0.0) 
	    {
	      std::cerr << "warning PHYSTOOLS::synrad " << photon[j] << " " << eng << " " << j << " " << n << std::endl;
	    }
	  
	  eng -= photon[j];
	  j++;
	  // Bound pathological radiation histories before continuing the particle.
	  // The vector is dynamic; this is a physics-domain guard, not its capacity.
	  if (j>=1000)
	    {
	      throw std::overflow_error(
	          "PHYSTOOLS::synrad_no_spin_flip produced at least 1000 photons "
	          "for one particle");
	    }
	}
    }
  return eng;   
}

// Synchrotron emission for an electron or positron using its local spin basis.
// adopt the variables change of CAIN (Yokoya, user's guide 235.1 p. 111))
// eng : energy in GeV
// dz : metres 
// with spin flip
// return the photon energy in 'photonEnergy'
// update the vector 'polar' :final polarization vector of the particle
// Optionally return the emitted photon's Stokes parameters in photonStokes.
int PHYSTOOLS::synrad_0_spin_flip (float upsilonSingleP,float eng, const TRIDVECTOR& e1, const TRIDVECTOR& e2, const TRIDVECTOR& e3, TRIDVECTOR& polar, TRIDVECTOR& /*stokes*/, float dzOnRadius,float* photonEnergy, RNDM& rndm_generator, float* photonStokes)
{
  int k;
  double x, s2, s3;
  double fu0, fusp;
  double p0,p1,v1,v3,g;
  double fK13, fKi13, fKi53, fK23;
  if (eng<=0.0){
    std::cerr << "Initial particle energy below zero : " << eng << std::endl;
    return 1;
  }
  double upsilon = (double)upsilonSingleP;

  // upsilon_bar = h_bar. omegac / E
  double upsilon_bar = 1.5*upsilon;
  //  double upsilon_bar = CCRIT*eng*eng*radius_i;
  //  double upsilon = 0.6666666667 * upsilon_bar;
  double gamma = eng/EMASS;
  double factor =  pow ( (1.0 + 0.5 * upsilon_bar ), 0.33333333333);
  p0 = CONST1 * dzOnRadius * gamma  / factor ; 

  if (rndm_generator.rndm_synrad()>p0){
    s2 = polar(0)*e2(0) + polar(1)*e2(1) + polar(2)*e2(2);
    fu0 = 1.0 - CONST3 * gamma*dzOnRadius * fradu0(upsilon);
    fusp = CONST3 * gamma* dzOnRadius * fradsp(upsilon);
    double c0 = fu0 + fusp * s2;
    for (k=0; k<3; k++){
      polar(k) = ( polar(k) * fu0 + fusp * e2(k) ) / c0;
    }
    double sum = polar.norm2();
    if (sum > 1.0){
      sum = 1.0 / sqrt(sum);
      polar *= sum;
    }
    *photonEnergy = 0.0;
    return 0;
  }
  else{
    p1=rndm_generator.rndm_synrad();
    while((v1=rndm_generator.rndm_synrad())==0.0) ; /* v1!= 0.0 */
    v3 = v1*v1*v1;
    double xden = 1.0 - v3 + 0.5 * upsilon_bar * ( 1.0 + v3 * v3 );
    x = upsilon_bar * v3 / xden;
    double x1 = 1.0 - x;
    double z = x/(upsilon_bar * x1);
    synradKi53andK23(z, fKi53, fK23);
    double F00 = fKi53 + (x*x/x1) * fK23;
    double F00star;
    s2 = polar(0)*e2(0) + polar(1)*e2(1) + polar(2)*e2(2);
    fK13 = synradK13(z);
    F00star = F00 - s2 * x * fK13; 
    double dxdy = 3.0 * v1 * v1 * (  upsilon_bar + x * (1.0 - upsilon_bar * v3 ) ) /xden;
    g  = F00star * dxdy * factor /(CONST0  * upsilon );
    //  stokes.clear();
    if( p1 < g){
      s3 = polar(0)*e3(0) + polar(1)*e3(1) + polar(2)*e3(2);
      // Photon Stokes parameters are not exported by this sampler.
      fKi13 = TOOLS::Ki13(z);
      if (photonStokes)
	{
	  // photon_polarisation=1: the photon's Stokes parameters in the frame (e1, e2, e3) from the spin before
	  // the emission (qedbg::synchrotron_stokes: Chen et al. 2022, eq. (17); its rate F* is F00star above)
	  const double s1 = polar(0)*e1(0) + polar(1)*e1(1) + polar(2)*e1(2);
	  qedbg::synchrotron_stokes(x, fKi53, fK23, fK13, fKi13, s1, s2, s3, photonStokes);
	}
      for (k=0; k < 3; k++)
	{
	  polar(k) = ( F00*polar(k) - x/(1.0-x) * fK13 * e2(k) - x*x/(1.0-x) * ( s3*e3(k)* fKi13 + (polar(k) - s3 * e3(k)  ) * fK23 ) ) / F00star;
	}
      //           std::cout << " norma of polar " << polar.norm() << std::endl;
      //	    polar.print();
      *photonEnergy = eng * x;
      return 1;
    }
    else{
	fu0 = 1.0 - CONST3 * gamma*dzOnRadius * fradu0(upsilon);
	fusp = CONST3 * gamma* dzOnRadius * fradsp(upsilon);
	double c0 = fu0 + fusp * s2;
	for (k=0; k<3; k++)
	  {
	    polar(k) = ( polar(k) * fu0 + fusp * e2(k) ) / c0;
	  }
	double sum = polar.norm2();
	if (sum > 1.0)
	  {
	    sum = 1.0 / sqrt(sum);
	    polar *= sum;
	  }
	*photonEnergy = 0.0;
	return 0;
    }
  }
}

// adopt the variables change of CAIN (Yokoya, user's guide 235.1 p. 111))
// eng : energy in GeV
// dz : metres 
// without spin flip
int PHYSTOOLS::synrad_0_no_spin_flip (float upsilonSingleP, float eng, float dzOnRadius,float* photonEnergy, RNDM& rndm_generator, float* photonStokes)
{
  double x;
  double p0,p1,v1,v3,g;
  double fKi53, fK23;//fK13, fKi13, 
  //j=0;
  if (eng<=0.0)
    {
      std::cerr << "Initial particle energy below zero : " << eng << std::endl;
      return 1;
    }

  double upsilon = (double)upsilonSingleP;
  // upsilon_bar = h_bar. omegac / E
  double upsilon_bar = 1.5*upsilon;
  //  double upsilon_bar = CCRIT*eng*eng*radius_i;

  //  double upsilon = 0.6666666667 * upsilon_bar;
  double gamma = eng/EMASS;
  double factor =  pow ( (1.0 + 0.5 * upsilon_bar ), 0.33333333333);
  p0 = CONST1 * dzOnRadius * gamma / factor ; 
  //  std::cout << " p0 = " << p0 << " previously " << synrad_p0(eng,radius_i,dz) << std::endl;
  if (rndm_generator.rndm_synrad()>p0) return 0;
  p1=rndm_generator.rndm_synrad();
  while((v1=rndm_generator.rndm_synrad())==0.0) ; /* v1!= 0.0 */
  v3 = v1*v1*v1;
  double xden = 1.0 - v3 + 0.5 * upsilon_bar * ( 1.0 + v3 * v3 );
  x = upsilon_bar * v3 / xden;
  double x1 = 1.0 - x;
  double z = x/(upsilon_bar * x1);
  synradKi53andK23(z, fKi53, fK23);
  double F00 = fKi53 + (x*x/x1) * fK23;

  double dxdy = 3.0 * v1 * v1 * (  upsilon_bar + x * (1.0 - upsilon_bar * v3 ) ) /xden;
  g  = F00 * dxdy * factor /(CONST0  * upsilon );
  if ( p1 < g) 
    {
      *photonEnergy = eng * x;
      // photon_polarisation=1: unpolarised emitter, linear polarisation K23/F along e1 (qedbg_photon_stokes.h)
      if (photonStokes) qedbg::synchrotron_stokes(x, fKi53, fK23, 0.0, 0.0, 0.0, 0.0, 0.0, photonStokes);
      return 1;
    }
  else 
    {
      *photonEnergy = 0.0;
      return 0;
    }
}


double PHYSTOOLS::BformFunction(double ups)
{
  // function giving the coefficient for interpolating the anomalous 
  // magnetic moment, as a function of upsilon following Baier

  //   a = alpha/(2*pi)*F(U) 
  //             2             x*dx                   x      t**3 
  //      F(U)= --- integral -------- * integral sin[---(t + ----)]*dt
  //             U           (1+x)**3                 U       3     
  //      F(0)=1

  double mu;
  double ups0 = 0.6125;
  double acoef[8] = { 12.190054896 , 24.159295870, -37.341016656 , -190.56332408 , -267.06921477, -80.540475512, -95.539356489, 246.29207314};

  double bcoef[8] = { 0.51911469393, 0.75556983816, -0.98346938317, 0.31707943752, -1.5451047715, 0.67601308567, -0.061924565451, -0.23548134968 };

  double u2, u3,u4,logups;

  if (ups <= 0.0) mu = 1.0;
  else
    {
      if ( ups <= ups0) 
	{
	  logups = log(ups);
	  mu = 1 + ups*ups*(acoef[0]*logups+ acoef[1] + 
				ups*(acoef[2]*logups + acoef[3] + 
				     ups*(acoef[4]*logups + acoef[5] +
					  ups*(acoef[6]*logups + 
					       acoef[7]))));
	}
      else
	{
	  u2 = 1./(ups*ups);
	  u3 = pow(u2, 0.3333333333333333);
	  u4 = u3*u3;
	  logups = log(ups);
	  mu = bcoef[0]*u3 + bcoef[1]*u4 + (bcoef[2]*logups + bcoef[3])*u2 + 
	    (bcoef[4]*u3 + bcoef[5]*u4 + (bcoef[6]*logups + bcoef[7])*u2)*u2;
	}
    }
  mu *= ANOMEL;
  return mu;
}

// integral K(5/3,u).du  over x < u < infinity
void  PHYSTOOLS::synradKi53andK23(double x, double& Ki53, double& K23)
 {
   if (x <= 1.54 )
     {
       double x1 = pow(x, 0.6666666667);
       double x2 = x1*x1;
       double x1inv = 1.0/x1;
       Ki53 = ( ( 0.03363782042 * x1 - 0.1134842702 ) * x2 +
                0.3944669710 ) * x2  -
	 1.812672515 + 2.1495282415344901 * x1inv;
       
       K23 = ( ( ( 0.04122878139 * x1 - 0.1494040962 ) * x2 +
		 0.7862616059 ) * x1 - 1.258219373 ) * x1 + 1.074647298 * x1inv;

     }
   else 
     if ( x <= 4.48 )
       {
	 Ki53 = ( ( 0.09120815010 * x  -1.229105693) * x + 4.442223505 ) /
	   ( ( ( x - 0.6903991322) * x + 5.651947051 ) * x - 0.9691386396 );
	 K23 = ( ( 0.08194471311 * x - 1.112728296 ) * x + 4.052334415 ) /
           ( ( ( x - 0.6524469236 ) * x + 6.1800441958 ) * x - 0.4915880600 );
       }
     else
       if ( x <= 165.0 )
	 {
	   double c = exp(-x) / sqrt(x);
	   Ki53 = c * ( 2.187014852 + 1.253535946 * x ) / ( 0.9949036186 + x );
	   K23  = c * ( 0.6120387636 + 1.253322122 * x ) / ( 0.3915531539 + x );
	 }
       else
	 {
	   Ki53 = 0.0;
	   K23  = 0.0;
	 }
 }



//  FRADU0 = (total radiation rate in quantum theory)/(that in classical)
//  as a function of Upsilon.
//   Accuracy:  Max.relative error < 5.05D-6  (for any Upsilon>=0)
// (from yokoya)
double PHYSTOOLS::fradu0(double ups)
{
  double x;
  const double A[6] = {8.0503959320, 10.9756518968, 1.4194297054, 8.9730711310, 15.8341489137, 4.1313700056 };
  const double B[6] = {1.0118692717, 2.9309973207, 1.6930111582, 2.8972660432, 2.2315495296, 1.7387035105 };
  const double ups0 = 1.3261;
  double resu;
  if (ups <= ups0) resu = ( ( ( A[2] * ups +A[1] ) * ups +A[0] ) * ups +1.0 ) / 
		     ( ( ( A[5] * ups +A[4] ) * ups +A[3] ) * ups +1.0);
  else
    {
      x = pow(1.0/ups, 0.3333333333);
      resu = ( ( B[2] * x +B[1] ) * x +B[0] ) * x / ( ( ( B[5] * x + B[4] ) * x + B[3] ) * x + 1.0);
    }
  return resu;
}

//   Integral of the spin-dependent term of radiation,
//   normalized by the classical total rate of radiation.
//                  2
//   FRADSP(Y)= ------ * Integral x*dx*K(1/3,(2/3/Y)*x/(1-x)) 
//               5*Pi*Y                      (over 0<x<1)
//   Relative error:  < 0.936E-5 for any Y.
// (from yokoya)
double PHYSTOOLS::fradsp(double ups)
{
  double x;
  const double A[7] = { .299997194, 2.62120367, .386895664, 15.6610660, 57.6471880, 26.4479482, .543635258 };
  const double B[6] = {9.91434629e-02, 4.92000917e-04, 1.45454865, 1.78219471e-03, 3.90751942e-01, 1.88294532e-01 };
  const double ups0 = 1.06237;
  double resu = 0.0;
  if (ups <= ups0 )
    {
      resu = ups * ( A[0] + ups * ( A[1] + ups * A[2] ) ) / 
	( 1.0 + ups * ( A[3] + ups * ( A[4] + ups  * ( A[5] + ups * A[6] ) ) ) );
    }
  else
    {
      x = pow(1.0/ups, 0.3333333333);
      resu = x*x * B[0] / ( 1.0 + x * ( B[1] + x * ( B[2] + x * ( B[3] + x* ( B[4] + x * B[5] ) ) ) ) );
    }
  return resu;
}


// electromagnetic fields in GV/nm
void PHYSTOOLS::referenceSpin(double vxd, double vyd, TRIDVECTOR& e1, TRIDVECTOR& e2, TRIDVECTOR& e3, TRIDVECTOR Efield, TRIDVECTOR Bfield, float /*charge_sign*/, double betaz)
{
  const double transverse2 = vxd*vxd + vyd*vyd;
  if (!(transverse2 < 1.0))
    throw std::domain_error("spin basis requires a subluminal transverse direction");
  const double vz = std::copysign(std::sqrt(1.0-transverse2), betaz);
  e3.setComponents(vxd, vyd, vz);
  e1 = transverse_Lorentz_force(vxd, vyd, Efield, Bfield, vz);
  // CAIN 2.1e, Sec. 5.8.3, Eqs. (99)-(104), defines e1 along the
  // transverse ACCELERATION and e2 = e3 x e1.  These solver fields already
  // include the relative bunch charge.  The orbit contract in
  // PARTICLE::advanceDueToEBfield is dv = -2*E_stored*ds/E, so the force in
  // this convention is -(E_stored + beta x B_stored).  Applying charge_sign
  // again would double-count the source charge, and omitting this minus sign
  // reverses the radiative transverse-polarization asymmetry.
  e1 *= -1.0;

  // A particle can sit where the transverse Lorentz force is exactly zero -- on
  // the axis of a symmetric bunch, or in a run with the field switched off.
  // e1 was then normalised from a zero vector, giving NaN in the whole spin
  // basis and, through synrad_spin_flip, NaN spins that no comparison caught.
  //
  // The basis is genuinely undefined there, but any orthonormal pair
  // perpendicular to e3 is as good as any other, because with no force there is
  // no preferred transverse direction and nothing physical depends on the
  // choice. Pick one deterministically instead of producing NaN.
  if (!(e1.norm() > 0.0))
    {
      const double ax = std::fabs(e3(0)), ay = std::fabs(e3(1)), az = std::fabs(e3(2));
      TRIDVECTOR seed;
      if (ax <= ay && ax <= az)      seed.setComponents(1.0, 0.0, 0.0);
      else if (ay <= az)             seed.setComponents(0.0, 1.0, 0.0);
      else                           seed.setComponents(0.0, 0.0, 1.0);
      e1.setComponents(e3(1)*seed(2) - e3(2)*seed(1),
                       e3(2)*seed(0) - e3(0)*seed(2),
                       e3(0)*seed(1) - e3(1)*seed(0));
    }
  e1.renormalize();
  e2.setComponents( e3(1)*e1(2) - e3(2)*e1(1), e3(2)*e1(0) - e3(0)*e1(2), e3(0)*e1(1) - e3(1)*e1(0) );
}

// transverse Lorentz force in eV/m, with respect to a particle trajectory
// data E and B, in eV/m
TRIDVECTOR PHYSTOOLS::transverse_Lorentz_force(double vxd, double vyd, TRIDVECTOR E, TRIDVECTOR B, double betaz)
{
  TRIDVECTOR  Ftrans;
  double xcomp, ycomp, zcomp;
  // F = E + beta x B, transverse to the momentum. beta_z used to be hard-coded
  // to +1 here, as it was in rotateBMT: right for beam 1, wrong for beam 2.
  //
  // This matters more than it looks. With B now built from the SOURCE's
  // direction, beam 2 gets B = (-Ey, Ex, 0); combining that with a hard-coded
  // beta_z = +1 gives beta x B = -E and hence F = E - E = 0 exactly. The spin
  // basis would then be built from a zero vector -- degenerate, and only saved
  // from being NaN by the guard in referenceSpin. With the true beta_z = -1 it
  // gives beta x B = +E and F = 2E, the counter-propagating doubling that is
  // physically right and that the orbit already applies to both beams.
  double ElongProjection = vxd*E(0) + vyd*E(1) + betaz*E(2);
  xcomp = E(0) - vxd*ElongProjection + vyd*B(2) - betaz*B(1);
  ycomp = E(1) - vyd*ElongProjection + betaz*B(0) - vxd*B(2);
  zcomp = E(2) - betaz*ElongProjection + vxd*B(1) - vyd*B(0);
  Ftrans.setComponents(xcomp, ycomp, zcomp);
  return Ftrans;
}
