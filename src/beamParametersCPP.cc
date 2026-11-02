#include "physconst.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include "beamParametersCPP.h"

BEAM_PARAMETERS::BEAM_PARAMETERS():ebeam_(0.0),n_particles_(0.0),gamma_beam_(0.0),waist_x_(0.0),waist_y_(0.0),L_0_(0.0),L_(0.0),bunches_per_train_(0),frep_(0.0)
{
  sigma_x_=0.0;
  sigma_y_=0.0;
  sigma_z_=0.0;
  em_x_=0.0;
  em_y_=0.0;
  beta_x_=0.0;
  beta_y_=0.0;
  offset_x_=0.0;
  offset_y_=0.0;
  offset_z_=0.0;
  phi_angle_=0.0;
  x_angle_=0.0;
  y_angle_=0.0;
  dist_x_=0;
  //  dist_y=0;
  dist_z_=0;
  trav_focus_=0;
  // initialise extension
  setLabel('0');
}

bool BEAM_PARAMETERS::acc_test(float& emitt,float& beta,float& sigma)
{

  // if beta has been given :
  //   if emitt has been given too, compute sigma from (emitt, beta)
  //   if emitt has not been given, compute emitt from (sigma, beta)
  //                                but, if sigma has not been given ??
  // if beta has not been given :
  //   compute beta from (sigma, emitt)

  // 2 of the three variables beta, emitt, sigma have to be given. Else, 
  // the problem must have load_beam =1, with automatic grid sizing.

  bool test = true;

  if (beta>0.0)
    {
      if (emitt>0.0)
	{
          sigma=sqrt(emitt *EMASS/ebeam_ * beta)*1e9;
        }
      else
	{
	  if ( sigma > 0.0 )
	    {
	      emitt=sigma * sigma *1e-18 / beta * ebeam_/EMASS;
	    }
	  else test = false;
	}
    }
  else
    {
      if (sigma > 0.0 && emitt > 0.0) 
	{
	  beta=(sigma * sigma *1e-18)/(emitt*EMASS/ebeam_);
	}
      else test = false;
    }
  return test;
}


void BEAM_PARAMETERS::setLabel(char label)
{
  extension_[0]= '.';
  extension_[1]= label;
  extension_[2]= 0;
}


void BEAM_PARAMETERS::create_name(std::string& name, std::string param)
{
  name.assign(param);
  name.append(extension_);
}


void BEAM_PARAMETERS::read(const PARAMETERS& param)
{
  std::string name;
  float compx, compy, compz;

  create_name(name, "energy");
  ebeam_ = param.readFValue(name);
  gamma_beam_ = ebeam_/EMASS;
  create_name(name, "particles");
  n_particles_ = param.readFValue(name)*1e10;

  create_name(name, "emitt_x");
  em_x_ = param.readFValue(name)*1e-6;
  create_name(name, "emitt_y");
  em_y_ = param.readFValue(name)*1e-6;
  
  create_name(name, "beta_x");
  beta_x_ = param.readFValue(name)*1e-3;
  create_name(name, "beta_y");
  beta_y_ = param.readFValue(name)*1e-3;

  create_name(name, "sigma_x");
  sigma_x_ = param.readFValue(name);

  create_name(name, "sigma_y");
  sigma_y_ = param.readFValue(name);

  create_name(name, "sigma_z");
  sigma_z_ = param.readFValue(name)*1e3;

  create_name(name, "dist_z");
  dist_z_ = param.readIValue(name); 

  create_name(name, "dist_x");
  dist_x_ = param.readIValue(name); 

  //  create_name(name, "dist_y");
  //  dist_y = readIValue(name); 

  create_name(name, "trav_focus");
  trav_focus_ = param.readIValue(name); 

  create_name(name, "offset_x");
  offset_x_ = param.readFValue(name);

  create_name(name, "offset_y");
  offset_y_ = param.readFValue(name);

  create_name(name, "offset_z");
  offset_z_ = param.readFValue(name)*1e3;

  create_name(name, "waist_x");
  waist_x_=param.readFValue(name)*1e3;

  create_name(name, "waist_y");
  waist_y_=param.readFValue(name)*1e3;

  create_name(name, "angle_phi");
  phi_angle_=param.readFValue(name);

  create_name(name, "angle_x");
  x_angle_=param.readFValue(name);

  create_name(name, "angle_y");
  y_angle_=param.readFValue(name);

  bunches_per_train_=param.readIValue("n_b");

  frep_=param.readFValue("f_rep");

  if ( !acc_test(em_x_, beta_x_, sigma_x_) ) sigma_x_ = 0.0;

  // emmittance in mm.mrad
  create_name(name, "emitt_x");
  param.setDoubleValue(name, em_x_*1e6);

  beta_x_ *= 1e3;

  create_name(name, "beta_x");
  param.setDoubleValue(name, beta_x_);
  create_name(name, "sigma_x");
  param.setDoubleValue(name, sigma_x_);
  if ( !acc_test(em_y_, beta_y_, sigma_y_) ) sigma_y_ = 0.0;

  // emmittance in mm.mrad
  create_name(name, "emitt_y");
  param.setDoubleValue(name, em_y_*1e6);

  beta_y_ *= 1e3;
  create_name(name, "beta_y");
  param.setDoubleValue(name, beta_y_);
  create_name(name, "sigma_y");
  param.setDoubleValue(name, sigma_y_);
  // polarization 
  create_name(name, "polar_x");
  compx = param.readFValue(name);
  create_name(name, "polar_y");
  compy = param.readFValue(name);
  create_name(name, "polar_z");
  compz = param.readFValue(name);
  set_polar((double)compx, (double)compy, (double)compz);
}

void BEAM_PARAMETERS::set_polar(double compx, double compy, double compz)
{
  polar_.setComponents(compx, compy, compz);
  if (polar_.norm() >1.0) polar_.renormalize();
}

std::string BEAM_PARAMETERS::output_flow() const
{
  std::ostringstream out;
  std::string start(" beam parameter ");
  start.append(std::string(&extension_[1]));
  out << title(start);
  out << " energy : " << ebeam_ << " GeV ; particles : " << n_particles_ << std::endl;
  out << "  sigma_x  : "  << sigma_x_ << " nm ;  sigma_y : " << sigma_y_ << " nm ; sigma_z : " << sigma_z_*1e-3 << " micrometers " <<  std::endl;
  out << " emitt_x : " << em_x_*1.e6 <<  " emitt_y : " << em_y_*1.e6 << "  (mm.mrad) " << std::endl;
  out << " beta_x : " << beta_x_*1e3 <<  " beta_y : " << beta_y_*1e3 << " (micrometers) " << std::endl;
  out << " offset_x : "  << offset_x_ << " nm ; offset_y : " << offset_y_ << " nm ; offset_z : " << offset_z_*1.e-3 << " micrometers " << std::endl;
  out << " waist_x : " << waist_x_*1.e-3 <<  " waist_y : " << waist_y_*1.e-3 << " (micrometers) " << std::endl;
  out << " angle_x : "  << x_angle_ << " angle_y : " << y_angle_ << " angle_phi : " << phi_angle_ << " (radians) " << std::endl;
  out << " type of distribution charge :  dist_x : " << dist_x_ <<  " dist_z : " << dist_z_ << std::endl;
  out << " initial polarization (ONLY FOR bmt_precession = 1 and internally generated beam) : polar_x = " << polar_.getComponent(0) << " polar_y = " << polar_.getComponent(1) << " polar_z = " << polar_.getComponent(2) << std::endl;
  return out.str();
}


