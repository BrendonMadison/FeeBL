#ifndef SPLINE_SEEN
#define SPLINE_SEEN

#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#define SPLINE_NMAX 1000

class SPLINE

{

typedef struct{double x,y,y2;}
spline_tab_entry;

int n_,xscal_,yscal_;
spline_tab_entry *tab_;

 void compute_tab();

 public :

 SPLINE() : n_(0),xscal_(0),yscal_(0),tab_(NULL) {;}

 SPLINE(const SPLINE& spl) : tab_(NULL)
  {
    recop(spl);
  }
 SPLINE& operator = (const SPLINE& spl) { 
   if (this == &spl) return *this; // protect against self-assignment
   // delete old memory:
   delete [] tab_;
   // assign new memory:
   recop(spl); 
   return *this;
 } 

 void spline_init(std::string pythiaFile);
 void spline_init(const std::vector<double>& x,int xscald, const std::vector<double>&,int yscald,int nd);
void spline_init(const double* x,int xscald,const double* y,int yscald,int n);
double spline_int(double x) const;

 inline void recop(const SPLINE& spl)
 {
   int k;
   n_ = spl.n_;
   xscal_ = spl.xscal_;
   yscal_ = spl.yscal_;
   tab_ = new spline_tab_entry[n_];
   for (k = 0; k < n_; k++)  tab_[k] = spl.tab_[k];     
 }

 ~SPLINE();


};


class MSPLINE

{
  int n_,xscal_,yscal_,nval_;
  double* x_;
  double* y_;
  double* y2_;
  
 public : 

 MSPLINE():n_(0),xscal_(0),yscal_(0),nval_(0),x_(NULL),y_(NULL),y2_(NULL)  {;}

  void mspline_init(const double* x,int xscal,const double* y,int yscal,int n,int nval);


 MSPLINE(const MSPLINE& spl) : x_(NULL), y_(NULL), y2_(NULL)
  {
    recop(spl);
  }

 ~MSPLINE();

 MSPLINE& operator = (const MSPLINE& spl) {
   if (this == &spl) return *this; // protect against self-assignment
   // delete old memory:
   delete [] x_;
   delete [] y_;
   delete [] y2_;
   // assign new memory:
   recop(spl); 
   return *this;
 }

 inline void recop(const MSPLINE& spl)
 {
   int k;
   n_ = spl.n_;
   nval_ = spl.nval_;
   xscal_ = spl.xscal_;
   yscal_ = spl.yscal_;
    x_=new double[n_];
    y_= new double[n_*nval_];
    y2_=new double[n_*nval_];

    for (k = 0; k < n_; k++)  x_[k] = spl.x_[k];
    for (k = 0; k < n_*nval_; k++)
      {
	y_[k] = spl.y_[k];
	y2_[k] = spl.y2_[k];
      }
         
 }

void mspline_int(double x,double y[]);

};

#endif
