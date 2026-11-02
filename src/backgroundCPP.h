#ifndef BACKGROUND_SEEN
#define BACKGROUND_SEEN

#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include "typeDefs.h"
#include "fileInputOutput.h"
#include "meshCPP.h"
#include "pairsCPP.h"
#include "splineCPP.h"
#include "resultsCPP.h"
#include "switchesCPP.h"
#include "jetParameterCPP.h"
#include "mathconst.h"
#include "physconst.h"
#include "abstractParticle.h"
#include "minijetsCPP.h"
#include "mathematicalTools.h"

class COMPT : public ABSTRACT_IO_CLASS
{
  
  COMPT_RESULTS compt_results_;
  double sum_,sum2_,sum3_,sume3_,sum4_,sume4_;
  double lsum_;
  int ncall_;
  std::uint64_t opposite_charge_source_encounters_ = 0;
  double opposite_charge_source_weight_ = 0.0;
  
  FILE_IN_OUT* compton_phot_file_;
  
  
 public:
  COMPT() : lsum_(0.0), ncall_(0), compton_phot_file_(NULL)
    {
      sum_=0.;
      sum2_=0.;
      sum3_ = 0.;
      sume3_ = 0.;
      sum4_=0.;
      sume4_=0.;
    }
  
  ~COMPT() 
    {
      delete compton_phot_file_; 
    }
  
  inline void connect_compt_phot_file(std::string name)
    {
      compton_phot_file_ = new FILE_IN_OUT();
      compton_phot_file_->open_file(name, "w");
    }
  
  void compt_do(const MESH& mesh, int cellx, int celly,float min_z, PAIR_BEAM& secondaries, int index_of_process,float epart,float ephot,float q2,float vx,float vy,float wgt, int dir,SWITCHES& switches, RNDM& rndm_generator);
    
  virtual inline std::string  output_flow() const 
    {
      std::ostringstream out;
      out << title(std::string("compton parameters"));
      out << "compt_sum = " << sum_ << std::endl;
      out << "compt_sum2 = " << sum2_ << std::endl;
      out << "compt_sum3 = " << sum3_ << " compt_sume3 = " << sume3_ << std::endl;
      out << "compt_sum4 = " << sum4_ << " compt_sume4 = " << sume4_ << std::endl;
      if (opposite_charge_source_encounters_ != 0) {
        out << "compt_opposite_charge_source_encounters = "
            << opposite_charge_source_encounters_ << std::endl;
        out << "compt_opposite_charge_source_weight = "
            << opposite_charge_source_weight_ << std::endl;
      }
      out << compt_results_.output_flow();
      return out.str();
    }

  
  inline void compt_write()
    {
      std::cout << "compt_sum= " << sum_ << std::endl;
      std::cout << "compt_sum2= " << sum2_ << std::endl;
      std::cout << "compt_sum3= " << sum3_ << " compt_sume3= " << sume3_ << std::endl;
  std::cout << "compt_sum4= " << sum4_ << " compt_sume4= " << sume4_ << std::endl;
    }

};

//&&&&&&&&&&&&&&&&&&&&&&&
class CROSS_DATA : public ABSTRACT_CROSS_DATA
{
  
  int number_of_energies_;
  int number_of_cross_sections_per_energy_;
  std::vector<float> energies_;
  std::vector< std::vector<float> > cross_sections_;
  
  public :
    
    CROSS_DATA() {;}
  CROSS_DATA(std::string crossIniFile)
    {
      load_cross(crossIniFile);
    }
  
  virtual  ~CROSS_DATA() {;}
  
  
  inline const std::vector<float>& energies() { return energies_;}
  inline const std::vector< std::vector<float> >& cross_sections() { return cross_sections_;}
  
  virtual inline void resize(int n, int nval)
    {
      number_of_energies_ = n;
      energies_.clear();
      energies_.reserve(number_of_energies_);
      cross_sections_.reserve(number_of_energies_);
      number_of_cross_sections_per_energy_ = nval;
    }

  virtual inline void add_data(float ener, const float* data)
    {
      
      int k;
      energies_.push_back(ener);
      cross_sections_.push_back(std::vector<float>(number_of_cross_sections_per_energy_));
      for (k=0; k< number_of_cross_sections_per_energy_; k++) 
	{
	  cross_sections_.back()[k] = data[k];
	}
    }
  
  inline int load_cross(std::string crossFileIni)
    {
      FILE_IN_OUT filin;
      filin.open_file(crossFileIni,"r");
      filin.read_cross(this);
      filin.close();
      return energies_.size();
    }
  
  
  inline void print()
    {
      unsigned int k,j;
      std::cout << " *******  CROSS_DATA *************** " << std::endl;
      std::cout << " number of energies : " << number_of_energies_ << std::endl;
      std::cout << " number of values per energy  : " << number_of_cross_sections_per_energy_ << std::endl;
      for (k=0; k < energies_.size(); k++)
	{
	  std::cout << " en= " << energies_[k] << " values: ";
	  for (j=0;j < cross_sections_[k].size(); j++)
	    {
	      std::cout << " " << cross_sections_[k][j];
	    } 
	  std::cout << std::endl;
	} 
    }
};

//&&&&&&&&&&&&&&&&&&&&&&&&
class GENERAL_CROSS : public ABSTRACT_IO_CLASS
{
  
 protected : 
  int nb_ener_;
  long int cross_call_;
  
 public :
   
 GENERAL_CROSS() : nb_ener_(0), cross_call_(0) {;}
  virtual  ~GENERAL_CROSS() {;}
  virtual  void cross_add(float e1,float e2,float flum) = 0;
  
};


class mCROSS  : public GENERAL_CROSS
{
  
 protected:
  
  MSPLINE mspline_;
  std::vector<double> sum_, sum2_;
  int ncross_per_ener_;
  
  mCROSS() 
    {  
      std::cout << " constructor mcross " << std::endl;
      ncross_per_ener_ = 0;
    }
  
  inline  void cumulate_sum(float flum, double* store)
    {
      int j;
      double tmp;
      flum *= 1e-37;
      for (j=0;j < ncross_per_ener_;j++)
	{
	  tmp=store[j]*flum;
	  sum_[j] += tmp;
	  sum2_[j] += tmp*tmp;
	}
    }
  
  
 public: 
  ~mCROSS() {;}
  
  mCROSS(std::string crossIniFile)
    {
      int k,j;
      double* xx;
      double* yy;
      int logx = 0;
      int logy = 0;
      CROSS_DATA cr_data(crossIniFile);
      const std::vector<float>& energ = cr_data.energies();
      const std::vector< std::vector<float> >& cross_val = cr_data.cross_sections();
      ncross_per_ener_ = cross_val[0].size();
      nb_ener_ = (int) energ.size();
      if ( nb_ener_ <= 0)
	{
	  std::cerr << " mCROSS:: WARNING no cross data " << std::endl;
	  return;
	}
      xx = new double[nb_ener_];
      yy = new double[ nb_ener_*ncross_per_ener_];
      for (k=0; k< nb_ener_; k++)
	{
	  xx[k] = energ[k];
	  if ( ncross_per_ener_ != (int) cross_val[k].size())
	    {
	      std::cerr << " mCROSS : incoherent data size, energy = " << energ[k] << " nval= " << 
		cross_val[k].size() << " supposed to be : " << ncross_per_ener_ << std::endl;
	    }
	  for (j=0; j< ncross_per_ener_; j++)
	    {
	      yy[k*ncross_per_ener_+j]=cross_val[k][j];
	    }
	}
      
      mspline_.mspline_init(xx,logx,yy,logy,nb_ener_,ncross_per_ener_);
      sum_.resize(ncross_per_ener_);
      sum2_.resize(ncross_per_ener_);
      for (j=0;j<ncross_per_ener_;j++)
	{
	  sum_[j]=0.0;
	  sum2_[j]=0.0;
	}
      delete[] xx;
      delete[] yy;
    } 
  
  virtual void cross_add(float e1,float e2,float flum)
    {
      double ecm;
      double* store = new double[ncross_per_ener_];
      ecm=2.0*sqrt(e1*e2);
      mspline_.mspline_int(ecm,store);
      cumulate_sum(flum, store);
      cross_call_++;
      delete [] store;
    }
  
  virtual std::string output_flow() const 
    {
      std::ostringstream out;
      int k;
      out << title(std::string("m-cross parameters"));
      out << " cross={ ";
      for (k=0; k < ncross_per_ener_; k++) out << "  " << sum_[k];
      out << " } " << std::endl;;
      out << " errors of the cross section calculation : cross_var= " << std::endl;
      out << " { ";
      for (k=0; k < ncross_per_ener_; k++)
	{
	  if (cross_call_ <= 1) out << " 0.0 " ;
	  else 
	    {
	      out <<  "  " << sqrt(std::max(0.0,(sum2_[k]/(double)cross_call_ - sum_[k]*sum_[k]/((double)cross_call_*(double)cross_call_))*(double)cross_call_));
	    }
	}
      out << "} " << std::endl;
      out << " ncross_ncall= " << cross_call_ << std::endl;
      return out.str();
    }
};

class maverCROSS : public mCROSS
{
  
  MSPLINE mspline_aver_;
  std::vector<double> cross_aver_;
  
  public :
    
  maverCROSS() {;}
  
  maverCROSS(std::string crossIniFile) : mCROSS(crossIniFile)
    {
      int k;
      mspline_aver_ = mspline_;
      cross_aver_.resize(ncross_per_ener_);
      for (k=0;k<ncross_per_ener_;k++)
	{
	  cross_aver_[k] = 0.0;
	}
    }
  
  ~maverCROSS() {;}
  
  
  virtual  void cross_add(float e1,float e2,float flum)
    {
      double* store = new double [ncross_per_ener_];
      int j;
      double ecm, tmp;
      
      ecm=2.0*sqrt(e1*e2);
      
      mspline_.mspline_int(ecm,store);
      cumulate_sum(flum, store);
      
      flum *= 1e-37;
      mspline_aver_.mspline_int(ecm,store);
      for (j=0;j< ncross_per_ener_ ;j++)
	{
	tmp=store[j]*flum;
	cross_aver_[j] += tmp;
      }
      delete [] store;
      cross_call_++;
    }
  
  virtual std::string output_flow() const 
    {
      std::ostringstream out;
      int j;
      out << mCROSS::output_flow() << std::endl;
      
      out << "cross_aver={ ";
      for (j=0;j<ncross_per_ener_;j++)
	{
	  out << " " << cross_aver_[j]/sum_[j];
	}
      out << " }" <<  std::endl;
      return out.str();
    }
};

class CROSS  : public GENERAL_CROSS
{

 protected : 
  SPLINE spline_;
  double sum_,sum2_;
  
 public: 
  CROSS() {sum_ = 0.0; sum2_ = 0.0;}
  
 CROSS(std::string crossIniFile) : sum_(0.0),sum2_(0.0)       
   {
     int k;
     double* xx;
     double* yy;
     int logx = 0;
     int logy = 0;
     CROSS_DATA cr_data(crossIniFile);
     const std::vector<float>& energ = cr_data.energies();
     const std::vector< std::vector<float> >& cross_val = cr_data.cross_sections();
     nb_ener_ = energ.size();
     if ( nb_ener_ <= 0)
       {
	 std::cerr << " mCROSS:: WARNING no cross data " << std::endl;
	 return;
       }
     int ncross_per_ener = cross_val[0].size();
     xx = new double[nb_ener_];
     yy = new double[nb_ener_];
     for (k=0; k< nb_ener_; k++)
       {
	 xx[k] = energ[k];
	 if ( ncross_per_ener != (int) cross_val[k].size())
	   {
	     std::cerr << " mCROSS : incoherent data size, energy = " << energ[k] << " nval= " << 
	       cross_val[k].size() << " supposed to be : " << ncross_per_ener << std::endl;
	   }
	 yy[k]=cross_val[k][0];
	 
       }
     
     spline_.spline_init(xx,logx,yy,logy,nb_ener_);
     delete[] xx;
     delete[] yy;
   }
 
~CROSS() {;}
 

 virtual  void cross_add(float e1,float e2,float flum)
   {
     double ecm, tmp;
     ecm=2.0*sqrt(e1*e2);
     flum *= 1e-37;
     
     tmp = spline_.spline_int(ecm)*flum;
     sum_ += tmp;
     sum2_ += tmp*tmp;

     cross_call_++;
   }
 
 virtual std::string output_flow() const 
   {
     std::ostringstream out;
     out << title(std::string(" cross parameters ")); 
     out << "cross= " << sum_ << std::endl;
     if(cross_call_ <=1 )   out << "cross_var= 0.0 " << std::endl;
     else
       {
	 out << "cross_var= " << sqrt(std::max(0.0,(sum2_/(double)cross_call_ - sum_*sum_/((double)cross_call_*cross_call_))*(double)cross_call_)) << std::endl;
       }
     out << "cross_ncall=" << cross_call_ << std::endl;
     return out.str(); 
   }
};

class averCROSS : public CROSS
{
  
  SPLINE spline_aver_;
  double cross_aver_;
  
  public :
    
  averCROSS() {;}
  
  averCROSS(std::string crossIniFile):CROSS(crossIniFile)
   {
     spline_aver_ = spline_;
     cross_aver_ = 0.0;      
   }
  
  ~averCROSS() {;}
  

  virtual  void cross_add(float e1,float e2,float flum)
  {
    double ecm, tmp;
    ecm=2.0*sqrt(e1*e2);
    flum *= 1e-37;
    
    tmp = spline_.spline_int(ecm)*flum;
    sum_ += tmp;
    sum2_ += tmp*tmp;

    tmp = spline_aver_.spline_int(ecm)*flum;
    cross_aver_ += tmp;
    cross_call_++;
  }

 virtual std::string output_flow() const 
   {
     std::ostringstream out;
     //int j;
     out << CROSS::output_flow() << std::endl;

     out << "cross_aver= " << cross_aver_/sum_ << std::endl;
   return out.str();
   } 
};

#endif
