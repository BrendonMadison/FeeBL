#ifndef BEAMPARAMETERS_SEEN
#define BEAMPARAMETERS_SEEN


#include <cstdio>
#include <sstream>
#include <string>
#include "abstractParticle.h"
#include "parametersCPP.h"
#include "mathematicalEntities.h"

class BEAM_PARAMETERS : public ABSTRACT_IO_CLASS
{

  float ebeam_,n_particles_;
  float gamma_beam_;
  
  // emittances in m.rad
  float em_x_,em_y_;
  
  float sigma_x_,sigma_y_,sigma_z_;
  int dist_x_, dist_z_;
  
  // in mm
  float beta_x_,beta_y_;
  
  // initial polarization direction 
  // if this vector is not normed (the norme must be <= 1)
  // polarization represents a mixed state
  TRIDVECTOR polar_;
  
  float offset_x_,offset_y_,offset_z_;
  float waist_x_,waist_y_;
  // float couple_xy_;
  float phi_angle_;
  float x_angle_,y_angle_;
  float L_0_,L_;
  //  int   what_distribution_;
  int bunches_per_train_;
  float frep_;
  int trav_focus_;
  char extension_[3];

  void set_polar(double compx, double compy, double compz);

  void create_name(std::string& name, std::string param);

  inline std::string param_with_extension(std::string param) const
    {
      return param.append(extension_);
    }
  
  
  bool acc_test(float& emitt,float& beta,float& sigma);
  
 public:
  
  BEAM_PARAMETERS();
  
  virtual std::string output_flow() const;


  inline int label() const 
    {
      std::istringstream stream1;
      stream1.str(std::string(&extension_[1]));
      int lab;
      stream1 >> lab;
      return lab;
    }  
  
  void read(const PARAMETERS& param);
  
  void setLabel(char label);
  
  inline float ebeam() const {return ebeam_;}
  inline float gamma_beam() const {return gamma_beam_;}
  
  // betas in mm
  inline float beta_x() const { return beta_x_;}
  inline float beta_y() const { return beta_y_;}
  
  inline float n_particles()  const  {return   n_particles_;}     
  // emittances in m.rad
  inline float em_x() const {return em_x_;};
  inline float em_y() const {return em_y_;};
  
  inline float sigma_x() const { return sigma_x_;}
  inline float sigma_y() const { return sigma_y_;}
  
  inline float sigma_z() const 
    {
      return sigma_z_;
    }
  
  inline int dist_x() const {return dist_x_;}
  inline int dist_z() const {return dist_z_;}
  inline float offset_x() const {return offset_x_;}
  inline float offset_y() const {return offset_y_;}
  inline float offset_z() const {return offset_z_;}
  inline float waist_x() const {return waist_x_;}
  inline float waist_y() const {return waist_y_;}
  inline float phi_angle() const {return phi_angle_;}
  inline float x_angle() const {return x_angle_;}
  inline float y_angle() const {return y_angle_;}
  inline float L_0() const {return L_0_;}
  inline float L() const {return L_;}
  inline int bunches_per_train() const {return bunches_per_train_;}
  inline float frep() const {return frep_;}
  inline int trav_focus() const  {return trav_focus_;}
  inline TRIDVECTOR get_polar() const { return polar_;}
  // inline float get_polar_rate() const {return polar_rate_;}
};

#endif
