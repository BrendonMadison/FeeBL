#include "splineCPP.h"
#include <cstdio>
#include "fileInputOutput.h"

SPLINE::~SPLINE()
{
  delete [] tab_;
}

void SPLINE::spline_init(std::string pythiaFile)
{
  int logx, logy, nentries;
  std::vector<double> x, y;
  FILE_IN_OUT filin;
  filin.open_file(pythiaFile,"r");
  nentries = filin.read_pythia_file(logx,logy, x, y);
  filin.close();
  spline_init(x, logx, y, logy,nentries);
}



void SPLINE::compute_tab()
{
  int i;
  double u[SPLINE_NMAX],sig,p;
  u[0]=0.0;
  for (i=1;i < n_-1 ;i++)
    {
      sig=(tab_[i].x-tab_[i-1].x)
	/(tab_[i+1].x-tab_[i-1].x);
      p=1.0/(sig*tab_[i-1].y2+2.0);
      tab_[i].y2=(sig-1.0)*p;
      u[i]=(6.0*((tab_[i+1].y-tab_[i].y)
		 /(tab_[i+1].x-tab_[i].x)
		 -(tab_[i].y-tab_[i-1].y)
		 /(tab_[i].x-tab_[i-1].x))
	    /(tab_[i+1].x-tab_[i-1].x)-sig*u[i-1])*p;
    }
  tab_[n_-1].y2=0.0;
  for (i=n_-2;i>=0;i--)
    {
      tab_[i].y2=tab_[i].y2*tab_[i+1].y2+u[i];
    }
}

void SPLINE::spline_init(const double* x,int xscald,const double* y,int yscald,int nd)
{
     int i;
//     double u[SPLINE_NMAX],sig,p;

    if (nd>SPLINE_NMAX)
      {
	std::cerr << " SPLINE::spline_init Error: to many values in spline_init\n" << std::endl;
	exit(1);
      }
    //
    n_=nd;
    xscal_ = xscald;
    yscal_ = yscald;
    tab_ = new spline_tab_entry[n_];
    tab_[0].y2=0.0;
    for (i=0;i < n_ ;i++)
      {
	switch (xscal_)
	  {
	  case 0:
	    tab_[i].x=x[i];
	    break;
	  case 1:
	    tab_[i].x=log(x[i]);
	    break;
	  }
	switch (yscal_)
	  {
	  case 0:
	    tab_[i].y=y[i];
	    break;
	  case 1:
	    tab_[i].y=log(y[i]);
	    break;
	  }	    
      }
   compute_tab();
}

void SPLINE::spline_init(const std::vector<double>& x,int xscald, const std::vector<double>& y,int yscald,int nd)
{
     int i;
//     double u[SPLINE_NMAX],sig,p;

     if (nd != (int) x.size() || nd != (int) y.size() )
       {
	 std::cerr << " SPLINE::spline_init ERROR : x and y vectoes have not the same size " << std::endl;
	 exit(1);
       }
    if (nd>SPLINE_NMAX)
      {
	std::cerr << " SPLINE::spline_init Error: to many values in spline_init\n" << std::endl;
	exit(1);
      }
    //
    n_=nd;
    xscal_ = xscald;
    yscal_ = yscald;
    tab_ = new spline_tab_entry[n_];
    tab_[0].y2=0.0;
    for (i=0;i < n_ ;i++)
      {
	switch (xscal_)
	  {
	  case 0:
	    tab_[i].x=x[i];
	    break;
	  case 1:
	    tab_[i].x=log(x[i]);
	    break;
	  }
	switch (yscal_)
	  {
	  case 0:
	    tab_[i].y=y[i];
	    break;
	  case 1:
	    tab_[i].y=log(y[i]);
	    break;
	  }	    
      }
    compute_tab();
}

double SPLINE::spline_int(double x) const
{
    int kmin,kmax,kpoint;
    double a,b,w;

    kmin=0;
    kmax = n_-1;
    switch(xscal_){
    case 0:	
	break;
    case 1:
	x=log(x);
	break;
    }
    if (x>tab_[kmax].x){
	if (yscal_){
	    return exp(tab_[kmax].y);
	}
	else{
	    return tab_[kmax].y;
	}
    }
    if (x<tab_[0].x){
      if (yscal_) {
	return exp(tab_[0].y);
      }
      else {
	return tab_[0].y;
      }
    }
    while (kmax-kmin>1){
	kpoint=(kmax+kmin)/2;
	if (tab_[kpoint].x>x){
	    kmax=kpoint;
	}
	else{
	    kmin=kpoint;
	}
    }
    w=tab_[kmax].x-tab_[kmin].x;
    a=(tab_[kmax].x-x)/w;
    b=(x-tab_[kmin].x)/w;
    x=a*tab_[kmin].y+b*tab_[kmax].y+
	(a*(a*a-1.0)*tab_[kmin].y2
	 +b*(b*b-1.0)*tab_[kmax].y2)*w*w/6.0;
    switch (yscal_){
    case 0:
	return x;
    case 1:
	return exp(x);
    }
    return x;
}

MSPLINE::~MSPLINE()
{
  delete [] x_;
  delete [] y_;
  delete [] y2_;
}

void MSPLINE::mspline_init(const double* xd,int xscald,const double* yd,int yscald,int nd,int nvald)
{
    int i,j;
    double u[SPLINE_NMAX],sig,p;

    if (nd>SPLINE_NMAX) {
	fprintf(stderr,"Error: to many values in mspline_init\n");
	exit(1);
    }
    n_=nd;
    nval_=nvald;
    xscal_=xscald;
    yscal_=yscald;
    x_=new double[n_];
    y_= new double[n_*nval_];
    y2_=new double[n_*nval_];
    for (j=0;j<nval_;j++){
	y2_[j]=0.0;
    }
    for (i=0;i<n_;i++){
	switch (xscal_){
	case 0:
	    x_[i]=xd[i];
	    break;
	case 1:
	    x_[i]=log(xd[i]);
	    break;
	}
	for (j=0;j<nval_;j++){
	    switch (yscal_){
	    case 0:
		y_[i*nval_+j]=yd[i*nval_+j];
		break;
	    case 1:
		y_[i*nval_+j]=log(yd[i*nval_+j]);
		break;
	    }
	}	    
    }
    for (j=0;j<nval_;j++){
	u[0]=0.0;
	for (i=1;i<n_-1;i++){
	    sig=(x_[i]-x_[i-1])
		/(x_[i+1]-x_[i-1]);
	    p=1.0/(sig*y2_[(i-1)*nval_+j]+2.0);
	    y2_[i*nval_+j]=(sig-1.0)*p;
	    u[i]=(6.0*((y_[(i+1)*nval_+j]-y_[i*nval_+j])
		       /(x_[i+1]-x_[i])
		       -(y_[i*nval_+j]-y_[(i-1)*nval_+j])
		       /(x_[i]-x_[i-1]))
		  /(x_[i+1]-x_[i-1])-sig*u[i-1])*p;
	}
	y2_[(n_-1)*nval_+j]=0.0;
	for (i=n_-2;i>=0;i--){
	    y2_[i*nval_+j]=
		y2_[i*nval_+j]*y2_[(i+1)*nval_+j]+u[i];
	}
    }
}

void MSPLINE::mspline_int(double xd,double yd[])
{
    int kmin,kmax,kpoint,j,nval;
    double a,b,w,tmp;

    nval=nval_;
    kmin=0;
    kmax=n_-1;
    switch(xscal_){
    case 0:	
	break;
    case 1:
	xd=log(xd);
	break;
    }
    if (xd>x_[kmax]){
	if (yscal_){
	    for (j=0;j<nval;j++) {yd[j]=exp(y_[kmax*nval+j]);}
	}
	else{
	    for (j=0;j<nval;j++) {yd[j]=y_[kmax*nval+j];}
	}
	return;
    }
    if (xd<x_[0]){
	for (j=0;j<nval;j++) {yd[j]=0.0;}
	return;
    }
    while (kmax-kmin>1){
	kpoint=(kmax+kmin)/2;
	if (x_[kpoint]>xd){
	    kmax=kpoint;
	}
	else{
	    kmin=kpoint;
	}
    }
    w=x_[kmax]-x_[kmin];
    a=(x_[kmax]-xd)/w;
    b=(xd-x_[kmin])/w;
    for (j=0;j<nval;j++){
	tmp=a*y_[kmin*nval+j]+b*y_[kmax*nval+j]+
	    (a*(a*a-1.0)*y2_[kmin*nval+j]
	     +b*(b*b-1.0)*y2_[kmax*nval+j])*w*w/6.0;
	switch (yscal_){
	case 0:
	    yd[j]=tmp;
	    break;
	case 1:
	    yd[j]=exp(tmp);
	    break;
	}
    }
}


