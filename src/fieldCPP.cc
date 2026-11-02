#include "fieldCPP.h"

/**********************************************/
/* Routines for the calculation of the fields */
/**********************************************/




/*! This routine calculates the potentials with the help of the fast fourier
   transform */

void FIELD::fold_fft(const PHI_FLOAT *rho1,const PHI_FLOAT *rho2,const PHI_FLOAT *dist,PHI_FLOAT *phi1, PHI_FLOAT *phi2,int n_x,int n_y, float charge_sign, ABSTRACT_FOURIER* fourier_forward, ABSTRACT_FOURIER* fourier_backward)
{
  // fourier_forward and fourier_backward are predefined fourier operators
  // the first one operates on data file : fourier_forward->data_vector() 
  // to be filled (see temp, below)
  // the same thing for  fourier_backward
  // these are complex fourier 
  // the charges of  beam1 will be put in the real part
  // the charges of  beam2 will be put in the imginary part

  // (see commentaries in GRID::init_grid_phys)
  // here one makes a convolution between a vector of charge (rho1 or rho2)
  // (in fact number of charges in cells) and a Green's kernel (dist)

  int i1,i2,i,j,j0,m,m2;
  //  int nn[2];
  float eps=1e-5;
#ifdef SHORTCUT_FFT
  int flag1=0,flag2=0;
#endif
  /* return no field */
  
  if (std::abs(charge_sign)<eps)
    {
      for (i=0;i<n_x*n_y;i++)
	{
	  phi1[i]=0.0;
	  phi2[i]=0.0;
	}
      return;
    }
  // nn[0]=2*n_y;
  // nn[1]=2*n_x;

  double* temp = fourier_forward->data_vector();
  double* temp_back =  fourier_backward->data_vector();
  for (i1=0;i1<n_x;i1++)
    {

      for (i2=0;i2<n_y;i2++)
	{
	  j=2*(i1*2*n_y+i2);
	  j0=i1*n_y+i2;
	  temp[j]=rho1[j0];
	  temp[j+1]=rho2[j0];
	  j += n_y + n_y;
 	  temp[j]=0.0;
 	  temp[j+1]=0.0;
#ifdef SHORTCUT_FFT
	  if (rho1[j0]!=0.0) 
	    {
	      flag1=1;
	    }
	  if (rho2[j0]!=0.0)
	    {
	      flag2=1;
	    }
#endif
	}

//       for (i2=0;i2<n_y;i2++)
// 	{
// 	  j=2*((i1*2+1)*n_y+i2);
// 	  temp[j]=0.0;
// 	  temp[j+1]=0.0;
// 	}
    }

  for (i1=n_x;i1<2*n_x;i1++)
    {
      for (i2=0;i2<2*n_y;i2++)
	{
	  j=2*(i1*2*n_y+i2);
	  temp[j]=0.0;
	  temp[j+1]=0.0;
	}
    }
#ifdef SHORTCUT_FFT
  if (flag1 && flag2) 
    {
#endif
      i1=2;
      fourier_forward->make();

      for (i=0;i<n_x*n_y*4;i++)
	{
	  j=2*i;
	  temp_back[j]=temp[j]*dist[j]-temp[j+1]*dist[j+1];
	  temp_back[j+1]=temp[j+1]*dist[j]+temp[j]*dist[j+1];
	}

      fourier_backward->make();

#ifdef SHORTCUT_FFT
      //   printf("there are charges in slice\n");

    }
  else
    {
      std::cerr << " No charge in slice flag1 = " << flag1 << " flag2= " << flag2 << std::endl;
    }
#endif
  m=0;
  m2=0;
  for (i1=0;i1<n_x;i1++)
    {
      for (i2=0;i2<n_y;i2++)
	{
	  phi1[m]=temp_back[m2++];
	  phi2[m]=temp_back[m2++];
	  m++;
	}
      m2+=2*n_y;
    }
}

/*! This routine calculates the potentials on the outer boundary of the grid
   by direct calculation. */

void FIELD::foldfronts (const PHI_FLOAT *rho,const PHI_FLOAT *dist,PHI_FLOAT *phi,int n_x,int n_y, float charge_sign)
{
  double s;
  //  int i1,i2,i3,i4,j;
  int i1,i2,i3,i4;
  float eps=1e-5;

  if (std::abs(charge_sign)<eps){
      for (i1=0;i1<n_x*n_y;i1++){
	  phi[i1]=0.0;
      }
      return;
  }

  i2=0;
  for (i1=0;i1<n_x;i1++)
    {
      s=0.0;
      for (i3=0;i3<n_x;i3++)
	{
	  for (i4=0;i4<n_y;i4++)
	    {
	      s+=rho[i3*n_y+i4]
		*dist[std::abs(i1-i3)*n_y+std::abs(i2-i4)];
	    }
	}
      phi[i1*n_y+i2]=s;
    }
  i2=n_y-1;
  for (i1=0;i1<n_x;i1++)
    {
      s=0.0;
      for (i3=0;i3<n_x;i3++)
	{
	  for (i4=0;i4<n_y;i4++)
	    {
	      s+=rho[i3*n_y+i4]
		*dist[std::abs(i1-i3)*n_y+std::abs(i2-i4)];
	    }
	}
      phi[i1*n_y+i2]=s;
    }
  i1=0;
  for (i2=1;i2<n_y-1;i2++)
    {
      s=0.0;
      for (i3=0;i3<n_x;i3++)
	{
	  for (i4=0;i4<n_y;i4++)
	    {
	      s+=rho[i3*n_y+i4]
		*dist[std::abs(i1-i3)*n_y+std::abs(i2-i4)];
	    }
	}
      phi[i1*n_y+i2]=s;
    }
  i1=n_x-1;
  for (i2=1;i2<n_y-1;i2++)
    {
      s=0.0;
      for (i3=0;i3<n_x;i3++)
	{
	  for (i4=0;i4<n_y;i4++)
	    {
	      s+=rho[i3*n_y+i4]
		*dist[std::abs(i1-i3)*n_y+std::abs(i2-i4)];
	    }
	}
      phi[i1*n_y+i2]=s;
    }
}

/*! Routine to calculate the potentials with an iterative method */

void FIELD::sor2 (const PHI_FLOAT *rho,PHI_FLOAT *phi,int n_x,int n_y,PHI_FLOAT *parameter, float charge_sign)
{
  const int MAXIT=1000;
  const PHI_FLOAT eps=1e-5;
  double anormf=0.0,anorm;
  PHI_FLOAT omega,resid,e_inv;
  int i1,i2,n,j;
  PHI_FLOAT a,b,c,d,e,rjac;
  float small=1e-5;

  if (std::abs(charge_sign)<small){
      return;
  }

  a=parameter[0];
  b=parameter[1];
  c=parameter[2];
  d=parameter[3];
  e=parameter[4];
  rjac=parameter[5];
  for (i1=1;i1<n_x-1;i1++)
    {
      for (i2=1;i2<n_y-1;i2++)
	{
	  anormf += std::abs((double)rho[i1*n_y+i2]);
	}
    }
  omega=1.0;
  e_inv=1.0/e;
  for (n=0;n<MAXIT;n++)
    {
      anorm=0.0;
      for (i1=1;i1<n_x-1;i1++)
	{
	  for (i2=1;i2<n_y-1;i2++)
	    {
	      if (((i1+i2) % 2)==(n % 2))
		{
		  j=i1*n_y+i2;
		  resid=a*phi[j+n_y]+b*phi[j-n_y]+c*phi[j+1]
		    +d*phi[j-1]+e*phi[j]-rho[j];
		  anorm += std::abs((double)resid);
		  phi[j] -= omega*resid*e_inv;
		}
	    }
	}
      if (n==0)
	{
	  omega=1.0/(1.0-0.5*rjac*rjac);
	}
      else
	{
	  omega=1.0/(1.0-0.25*rjac*rjac*omega);
	}
      if ((n>0)&&(anorm<eps*anormf))
	{
	  return;
	}
    }
  printf ("Warning: too many iterations in function sor2\n");
}


/*! This routine is a subroutine for init_grid. 

calculate the expression $\Phi$i0,j0 of the thesis p. 20 

*/

double FIELD::f_potential_2(double x0,double y0,double dx,double dy)
{
  double x,y,sum;
  x=x0+dx;
  y=y0+dy;
  sum = x*y*(log(x*x+y*y)-3.0)
      +x*x*atan(y/x)+y*y*atan(x/y);
  x=x0+dx;
  y=y0-dy;
  sum -= x*y*(log(x*x+y*y)-3.0)
      +x*x*atan(y/x)+y*y*atan(x/y);
  x=x0-dx;
  y=y0+dy;
  sum -= x*y*(log(x*x+y*y)-3.0)
      +x*x*atan(y/x)+y*y*atan(x/y);
  x=x0-dx;
  y=y0-dy;
  sum += x*y*(log(x*x+y*y)-3.0)
      +x*x*atan(y/x)+y*y*atan(x/y);
  return sum;
}

/*! This routine folds the charge distribution and the greensfunction of
a grid to get the potentials. */

void FIELD::foldfields (const PHI_FLOAT *rho,const PHI_FLOAT *dist,PHI_FLOAT *phi,int n_x,int n_y, float charge_sign)
{
  double s;
  //  int i1,i2,i3,i4,j;
  int i1,i2,i3,i4;
  //int field_typ=1;
  float eps=1e-5;

  if (std::abs(charge_sign)<eps){
      for (i1=0;i1<n_x*n_y;i1++){
	  phi[i1]=0.0;
      }
      return;
  }
  
  for (i1=0;i1<n_x;i1++)
    {
      for (i2=0;i2<n_y;i2++)
	{
	  s=0.0;
	  for (i3=0;i3<n_x;i3++)
	    {
	      for (i4=0;i4<n_y;i4++)
		{
		  s+=rho[i3*n_y+i4]
		      *dist[std::abs(i1-i3)*n_y+std::abs(i2-i4)];
		}
	    }
	  phi[i1*n_y+i2]=s;
	}
    }
}


  // the Poisson's will be solved with the help of a 2D-Green's Kernel
  // for the eq. Delta(Phi) = rho/epsilon0
  // the 2d Green's Kernel is : G(x,y) = ln(x^2 + y^2)/(4.pi.epsilon0)
  // the solution for the potential reads : 
  //  Phi(x,y) = SIGMAij(Qij/delz)Gij
  // Qij : charge in the cell indexed by (i,j) so that Qij/(delx.dely.delz)
  // approximate the charge density rho. 
  // Qij = nij.e (nij : number of part. in the cell ij ; e : electron charge)
  // Gij = an estimation of the Green's kernel
  // the function f_potential returns :
  // 4.pi.epsilon0.Integral of (ln(x^2 + y^2)dx.dy
  // so that an suited approximation of Gij is : 
  // Gij ~ (1/(4.pi.epsilon0)(1/(delx.dely))f_potential(i,j)
  // now, phi reads : 

  // Phi(x,y) ~ SIGMAij(Qij/delz)(1/(4.pi.epsilon0))(1/(delx.dely))f_potential(i,j)
  // or : Phi(x,y) ~ (e/(4.pi.epsilon0.delx.dely.delz))SIGMAij nij(1/(delx.dely))f_potential(i,j)
  // (Qij = nij.e)
  // 
  // the factor below is : 
  // factor = 2.*(electron radius)(electron mass in eV)/(delx*dely*delz)
  // i.e. :  e/(2.pi.epsilon0.delx*dely*delz)
  // so that Phi could be written : 
  // Phi(x,y) = (1/2).factor.SIGMAij(nij.f_potential(i,j))
  // here is prepared the kernel  : factor.f_potential
  // its Fourier transform is calculated

  // The inherited derivation by GLM leaves the factor 1/2 convention open.
  // Check the combined kernel and force normalization against the analytic
  // field tests before changing this factor; the potential uses GV.

void FIELD::dist_init(PHI_FLOAT factor, float deltax, float deltay,   FFT_SERVER* fourier)
{
  int k,i1, i2,j, j0;
  int nn[2];
  PHI_FLOAT phi0;
  double x0,y0;
  switch (integration_method_)
    {
    case 1:
    case 3:
      phi0 = factor*
	(f_potential(0.5*deltax,0.5*deltay)
	 - f_potential(-0.5*deltax,0.5*deltay)
	 - f_potential(0.5*deltax,-0.5*deltay)
	 + f_potential(-0.5*deltax,-0.5*deltay));
      for (i1=0;i1 < nb_cells_x_;i1++)
	{
	  for (i2=0;i2 < nb_cells_y_;i2++)
	    {
	      j=i1*nb_cells_y_+i2;
	      x0=i1;
	      y0=i2;
	      dist_[j]=factor*
		(f_potential((x0+0.5)*deltax,(y0+0.5)*deltay)
		 - f_potential((x0-0.5)*deltax,(y0+0.5)*deltay)
		 -  f_potential((x0+0.5)*deltax,(y0-0.5)*deltay)
		 + f_potential((x0-0.5)*deltax,(y0-0.5)*deltay))
		- phi0;
	    }
	}
      break;
    case 2:


      phi0 = factor;

      phi0 *= f_potential_2(0.0,0.0,0.5*deltax,0.5*deltay);

      nn[0]=2*nb_cells_y_;
      nn[1]=2*nb_cells_x_;

      ABSTRACT_FOURIER* fourier_transform = fourier->new_fft(std::string("for2"),nn);
      double* temp = fourier_transform->data_vector(); 
      for (i1=0;i1<dist_size_;i1++) temp[i1]=0.0;

      for (i1=0;i1<nb_cells_x_;i1++)
	{
	  for (i2=0;i2<nb_cells_y_;i2++)
	    {
	      j=i1*nb_cells_y_+i2;
	      j0=2*(i1*2*nb_cells_y_+i2);
	      x0=(double)i1;
	      y0=(double)i2;


	      temp[j0]=factor*
		f_potential_2(x0*deltax,y0*deltay,
				     0.5*deltax,0.5*deltay) -phi0;
	      // Keep the authoritative real-space quadrant used to construct the
	      // FFT spectrum.  Sparse-direct fields consume this exact value; they
	      // do not rederive the kernel or its historical normalization.
	      if (real_space_kernel_ != NULL)
		real_space_kernel_[j] = temp[j0];
	      if (i2>=1)
		{
		   	   
		  temp[2*((i1+1)*2*nb_cells_y_-i2)]=temp[j0];
		  if (i1>=1)
		    {


		      temp[2*(2*nb_cells_y_*(2*nb_cells_x_-i1)+i2)]=temp[j0];


		      temp[2*(2*nb_cells_y_*(2*nb_cells_x_-i1)+2*nb_cells_y_-i2)]=temp[j0];
		    }
		}
	      else
		{
		  if (i1>=1)
		    {
		      temp[2*(2*nb_cells_y_*(2*nb_cells_x_-i1)+i2)]=temp[j0];
		    }
		}
	    }
	}

      fourier_transform->make();

      for (k=0; k < dist_size_ ; k++) 
	{
	  dist_[k] =  temp[k]/(double)(4*nb_cells_x_*nb_cells_y_);
	}
 
      break;
    }
}
