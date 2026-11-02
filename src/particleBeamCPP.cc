#include <iostream>
#include <algorithm>
#include "particleBeamCPP.h"

// emittances in mm.mrad
void BEAM_FROM_FILE::emittances(float& emittx, float& emitty) const
{
  unsigned int k;
  int n_particles = particles_.size();
  float xpos, ypos;
  float x2, y2;
  float vx, vy;
  float beta_x2,beta_y2;
  float x_beta_x, y_beta_y;
  emittx = 0.0;
  emitty = 0.0;
  if (n_particles<=0) return;
  
  x2 = 0.0;
  y2 = 0.0;
  beta_x2 = 0.0;
  beta_y2 = 0.0;
  x_beta_x = 0.0;
  y_beta_y = 0.0;
  for (k=0; k < particles_.size(); k++)
    {
	  particles_[k].XYposition(xpos, ypos);
	  particles_[k].velocities(vx,vy);
	  // the positions are in nm, the "speed" is in radians.
	  // in fact, only betax, betay (betaz is supposed 
	  // practically equal to 1)
	  // the energy is in GeV

	  // x, y in mm

	  xpos *= 1.0e-6;
	  ypos *= 1.0e-6;

	  // beta in mrad
	  vx *= 1.0e3;
	  vy *= 1.0e3;

	  // for now x2 et y2 in mm^2
	  x2 += xpos*xpos;
	  y2 += ypos*ypos;
	  beta_x2 += vx*vx;
	  beta_y2 += vy*vy;

	  // mm.mrad
	  x_beta_x += xpos*vx;
	  y_beta_y += ypos*vy;
	
    }

  x2 /= n_particles;
  y2 /= n_particles;
  beta_x2 /= n_particles;
  beta_y2 /= n_particles;
  x_beta_x /= n_particles;
  y_beta_y /= n_particles;
  // in mm.mrad
  emittx = sqrt(x2*beta_x2 - x_beta_x*x_beta_x)*gamma_;
  emitty = sqrt(y2*beta_y2 - y_beta_y*y_beta_y)*gamma_;
}

void BEAM_FROM_FILE::beamXyRms(float& xmean, float& ymean,float& sigmaxRms, float& sigmayRms) const
{
  int j;
  float xpos, ypos;
  float x0, y0;
  float sigmax, sigmay;

  x0 = 0.;
  y0 =0.;
  sigmax = 0.;
  sigmay = 0.;
  int n_particles = (int) particles_.size();
  if (n_particles==0) return;

  for (j=0; j < n_particles; j++)
    {
      particles_[j].XYposition(xpos, ypos);
      x0 += xpos;
      y0 += ypos;
      sigmax += xpos*xpos;
      sigmay += ypos*ypos;
    }
  x0 /= (float)n_particles;
  y0 /= (float)n_particles;
  sigmax /= (float)n_particles;
  sigmay /= (float)n_particles;
  sigmaxRms = sqrt(sigmax-x0*x0);
  sigmayRms = sqrt(sigmay-y0*y0);
  xmean = x0;
  ymean = y0;
}

void BEAM_FROM_FILE::beamZRms(float& zmean, float& sigmazRms) const
{
  unsigned int j;
  float zpos;
  float z0;
  float sigmaz;

  z0 = 0.;
  sigmaz = 0.;
  int n_particles = particles_.size();
  if (n_particles==0) return;

  for (j=0; j < particles_.size(); j++)
    {
      zpos = particles_[j].z();
      z0 += zpos;
      sigmaz += zpos*zpos;
    }
  z0 /= (float)n_particles;
  sigmaz /= (float)n_particles;
  sigmazRms = sqrt(sigmaz-z0*z0);
  zmean = z0;
}



void PARTICLE_BEAM::init_particles(unsigned long int nbpart, float sigma_x,float sigma_y, float sigma_z,int dist_x,int dist_z,float emx, float emy,float delta_z,float energy, int charge_symmetric, int do_bds_spin_rotation)
{
  //int k;
  float sigma_x_prime,sigma_y_prime;
  //  polarization_ = polar;

  initial_number_of_particles_ = nbpart;
 
  // Legacy input units are preserved in this emittance-to-divergence
  // conversion: EMASSeV supplies the matching electron mass scale.

  sigma_x_prime=emx*EMASSeV/(energy*sigma_x);
  sigma_y_prime=emy*EMASSeV/(energy*sigma_y);
  if(charge_symmetric)
    {
      fill_symmetric_beam(dist_x, dist_z,delta_z, sigma_x, sigma_y,sigma_z, sigma_x_prime,sigma_y_prime, energy, do_bds_spin_rotation);
    }
  else
    {
      fill_beam(dist_x, dist_z,delta_z, sigma_x, sigma_y,sigma_z, sigma_x_prime,sigma_y_prime, energy, do_bds_spin_rotation);
    }
  compute_gamma();
  sigmax_ = sigma_x;
  sigmay_ = sigma_y;
  sigmaz_ = sigma_z;
}


void PARTICLE_BEAM::fill_beam(int dist_x, int dist_z, float delta_z, float sigma_x,float sigma_y, float sigma_z, float sigma_x_prime,float sigma_y_prime, float energy, int do_bds_spin_rotation)
{
  switch(dist_x)
    {
    case 0 :
      assign_xyz_normal_distribution(dist_z, delta_z, sigma_x,sigma_y,sigma_z, sigma_x_prime,sigma_y_prime, energy, do_bds_spin_rotation);
      break;
    default:
      std::cerr << " PARTICLE_BEAM::fill_beam :: unknown x distribution dist_x = " << dist_x << std::endl;
      exit(1); 
    }
}

void PARTICLE_BEAM::fill_symmetric_beam(int dist_x, int dist_z, float delta_z, float sigma_x,float sigma_y, float sigma_z, float sigma_x_prime,float sigma_y_prime, float energy, int do_bds_spin_rotation)
{
  switch(dist_x)
    {
    case 0 :
      assign_symmetric_xyz_normal_distribution(dist_z, delta_z, sigma_x,sigma_y,sigma_z, sigma_x_prime,sigma_y_prime, energy, do_bds_spin_rotation);
      break;
    default:
      std::cerr << " PARTICLE_BEAM::fill_symmetric_beam :: unknown x distribution dist_x = " << dist_x << std::endl;
      exit(1); 
    }

}


void PARTICLE_BEAM::assign_xyz_normal_distribution(int dist_z, float delta_z, float sigma_x,float sigma_y, float sigma_z, float sigma_x_prime,float sigma_y_prime, float energy, int do_bds_spin_rotation)
{
  unsigned int k;
  float ztemp; //float xtemp, ytemp, ztemp, vxtemp, vytemp;
  number_of_particles_dispatched_in_slices_ = 0;
  int nSlices = (int) particle_.size();
  switch(dist_z)
    {     
      // normal distribution in z 
    case 0:      
      for (k=0;k< initial_number_of_particles_;k++)
	{
	  ztemp=rndm_generator_->gasdev()*sigma_z;
	  dispatch_random_particle_in_slices(
	      ztemp, delta_z, sigma_x, sigma_y, sigma_z, sigma_x_prime,
	      sigma_y_prime, energy, nSlices, do_bds_spin_rotation, k);
	}
      break;
      // constant distribution in z 
    case 1: 
      {
	float bunchlength=SQRT3*sigma_z;
	for (k=0;k< initial_number_of_particles_;k++)
	  {
	    ztemp =(2.0*rndm_generator_->rndm()-1.0)*bunchlength;
	    dispatch_random_particle_in_slices(
	        ztemp, delta_z, sigma_x, sigma_y, sigma_z, sigma_x_prime,
	        sigma_y_prime, energy, nSlices, do_bds_spin_rotation, k);
	  }
      }
      break; 
    default:
      std::cerr << " PARTICLE_BEAM::assign_xyz_normal_distribution :: unknown z distribution dist_z = " << dist_z << std::endl;
      exit(1); 
    }
}

void PARTICLE_BEAM::assign_symmetric_xyz_normal_distribution(int dist_z, float delta_z, float sigma_x,float sigma_y,float sigma_z, float sigma_x_prime,float sigma_y_prime, float energy, int do_bds_spin_rotation)
{
  unsigned long int k;
  float ztemp;//float xtemp, ytemp,ztemp, vxtemp, vytemp;
  number_of_particles_dispatched_in_slices_ = 0;
  unsigned int nSlices = particle_.size();
  switch(dist_z)
    {      
      // normal distribution in z 
    case 0:      
      for (k=0;k< initial_number_of_particles_/4;k++)
	{
	  ztemp=rndm_generator_->gasdev()*sigma_z;
	  dispatch_symmetric_random_particle_in_slices(
	      ztemp, delta_z, sigma_x, sigma_y, sigma_z, sigma_x_prime,
	      sigma_y_prime, energy, nSlices, do_bds_spin_rotation, k);
	}
      break;
      // constant distribution in z 
    case 1: 
      {
	float bunchlength=SQRT3*sigma_z;
	for (k=0;k< initial_number_of_particles_;k++)
	  {
	    ztemp =(2.0*rndm_generator_->rndm()-1.0)*bunchlength;
	    dispatch_symmetric_random_particle_in_slices(
	        ztemp, delta_z, sigma_x, sigma_y, sigma_z, sigma_x_prime,
	        sigma_y_prime, energy, nSlices, do_bds_spin_rotation, k);
	  }
      }
      break;
    default:
      std::cerr << " PARTICLE_BEAM::assign_symmetric_xyz_normal_distribution :: unknown z distribution dist_z = " << dist_z << std::endl;
      exit(1); 
    }
}

// after loading, the object pointed by bff is empty and 
// the pointer bff is invalidated
unsigned int PARTICLE_BEAM::load_particles(BEAM_FROM_FILE*& bff, float /*emin*/, float zmin, float deltaz, float sigx, float sigy, float sigz)
{
  int k;
  float ztest;
  int nSlices = particle_.size();
  unsigned int count=0;
  //TRIDVECTOR polar;
  number_of_particles_dispatched_in_slices_ = 0;
  
  while( bff->not_empty() )
    {
      const PARTICLE_INTERFACE& particle = bff->pick_last();
      const std::uint64_t source_record_ordinal =
          bff->pick_last_source_record_ordinal();
      ztest = particle.z();
      
      k=(int)floor((ztest-zmin)/deltaz)+1;
      if((k>0) && (k <= nSlices))
	{
	  set_new_particle_in_slice(
	      k-1, particle,
	      make_loaded_primary_particle_id(beam_label_,
	                                      source_record_ordinal));
	  count++;
	}
      bff->erase_last_particle();
    } 
  delete bff;
  bff = NULL;
  compute_gamma();
  sigmax_ = sigx;
  sigmay_ = sigy;
  sigmaz_ = sigz;
  return count;
}
void PARTICLE_BEAM::meanPositionOfSlice(int slice, float& x,float& y) const
{
  int i,n_particles;
  float xpart, ypart;
  double sum_x=0.0,sum_y=0.0;
  n_particles= particle_[slice].size();
  if (n_particles==0)
    {
      std::cerr << "PARTICLE_BEAM::warning:: rms_distribution : no particles in slice " << slice << std::endl;
      x=0.0;
      y=0.0;
      return;
    }
  for (i=0;i< n_particles;i++)
    {

      particle_[slice][i]->XYposition(xpart, ypart);
     sum_x += xpart;
      sum_y += ypart;
    }
  sum_x /= (double) n_particles;
  sum_y /= (double) n_particles;
  x=sum_x;
  y=sum_y;
}

// emittances in mm.mrad
void PARTICLE_BEAM::emittances(float& emittx, float& emitty) const
{
  unsigned int k,j;
  int n_particles = number_of_particles_dispatched_in_slices_;
  float xpos, ypos;
  float x2, y2;
  float vx, vy;//, gamma;
  float beta_x2,beta_y2;
  float x_beta_x, y_beta_y;
  emittx = 0.0;
  emitty = 0.0;
  if (n_particles==0) return;
  
  //  gamma = 0.0;
  x2 = 0.0;
  y2 = 0.0;
  beta_x2 = 0.0;
  beta_y2 = 0.0;
  x_beta_x = 0.0;
  y_beta_y = 0.0;
  for (j=0; j < particle_.size(); j++)
    {
      for (k=0; k < particle_[j].size(); k++)
	{
	  particle_[j][k]->XYposition(xpos, ypos);
	  particle_[j][k]->velocities(vx,vy);
	  //	  energy = particle_[j][k].energy();
	  // the positions are in nm, the "speed" is in radians.
	  // in fact, only betax, betay (betaz is supposed 
	  // practically equal to 1)
	  // the energy is in GeV

	  // x, y in mm
	  xpos *= 1.0e-6;
	  ypos *= 1.0e-6;

	  // beta in mrad
	  vx *= 1.0e3;
	  vy *= 1.0e3;

	  //	  gamma += energy;

	  // for now x2 and y2 in mm^2
	  x2 += xpos*xpos;
	  y2 += ypos*ypos;
	  beta_x2 += vx*vx;
	  beta_y2 += vy*vy;

	  // mm.mrad
	  x_beta_x += xpos*vx;
	  y_beta_y += ypos*vy;
	}
    }

  x2 /= n_particles;
  y2 /= n_particles;
  beta_x2 /= n_particles;
  beta_y2 /= n_particles;
  x_beta_x /= n_particles;
  y_beta_y /= n_particles;
  //  gamma /= EMASS*n_particles;

  // in mm.mrad
  emittx = sqrt(x2*beta_x2 - x_beta_x*x_beta_x)*gamma_;
  emitty = sqrt(y2*beta_y2 - y_beta_y*y_beta_y)*gamma_;
}

void PARTICLE_BEAM::beamXyRms(float& xmean, float& ymean,  float& sigmaxRms, float& sigmayRms) const
{
  unsigned int k,j;
  float xpos, ypos;
  float x0, y0;
  float sigmax, sigmay;
  x0 = 0.;
  y0 =0.;
  sigmax = 0.;
  sigmay = 0.;
  int n_particles = number_of_particles_dispatched_in_slices_;
  if (n_particles==0) return;

  for (j=0; j < particle_.size(); j++)
    {
      for (k=0; k < particle_[j].size(); k++)
	{
	  particle_[j][k]->XYposition(xpos, ypos);
	  x0 += xpos;
	  y0 += ypos;
	  sigmax += xpos*xpos;
	  sigmay += ypos*ypos;
	}
    }
  x0 /= (float)n_particles;
  y0 /= (float)n_particles;
  sigmax /= (float)n_particles;
  sigmay /= (float)n_particles;
  sigmaxRms = sqrt(sigmax-x0*x0);
  sigmayRms = sqrt(sigmay-y0*y0);
  xmean = x0;
  ymean = y0;
}
void PARTICLE_BEAM::beamZRms(float& zmean,  float& sigmazRms) const
{
  unsigned int k,j;
  float  zpos;
  float z0;
  float  sigmaz;
  z0 = 0.;
  sigmaz = 0.;
  int n_particles = number_of_particles_dispatched_in_slices_;
  if (n_particles==0) return;

  for (j=0; j < particle_.size(); j++)
    {
      for (k=0; k < particle_[j].size(); k++)
	{
	  zpos = particle_[j][k]->z();
	  z0 += zpos;
	  sigmaz += zpos*zpos;
	}
    }
  z0 /= (float)n_particles;
  sigmaz /= (float)n_particles;
  sigmazRms = sqrt(sigmaz-z0*z0);
  zmean = z0;
}

void PARTICLE_BEAM::transverseRms(int slice,double& xmin,double& xmax,double& xmean,double&  ymin,double& ymax,double& ymean,double&  sigmaxRms,double& sigmayRms) const 
{
  unsigned int k;
  float xpos, ypos;
  double x0, y0, xmn, ymn, xmx, ymx;
  double sigmax, sigmay;
  double dx, dy;
  double n;
  x0=0.0; y0=0.0;
  xmn=1e30; xmx=-1e30;
  ymn=1e30; ymx=-1e30;
  sigmax=0.0; sigmay=0.0;
  for (k=0; k< particle_[slice].size(); k++)
    {
      particle_[slice][k]->XYposition(xpos, ypos);
      dx = (double)xpos;
      dy = (double)ypos;
      xmn = std::min(xmn, dx);
      xmx = std::max(xmx, dx);
      x0 += dx;
      ymn = std::min(ymn,dy);
      ymx = std::max(ymx,dy);
      y0 += dy;
      sigmax += dx*dx;
      sigmay += dy*dy;
    }
  n = (double)particle_[slice].size();
  n = std::max(1.0,n);
  x0 /= n;
  y0 /= n;
  sigmax /= n;
  sigmay /= n;
  sigmaxRms=sqrt(std::max(0.0,sigmax-x0*x0));
  sigmayRms=sqrt(std::max(0.0,sigmay-y0*y0));
  xmean = x0;
  ymean = y0;
  xmin = xmn;
  xmax = xmx;
  ymin = ymn;
  ymax = ymx;
}

// ********************************************************************
// Routines to manipulate the initial particle distribution           
// *********************************************************************

void PARTICLE_BEAM::rotate_particles(float angle)
{
  unsigned int j,k;
  float c,s;//,x,y;
  //float vx, vy;
  c=cos(angle);
  s=sin(angle);


  for (j=0; j < particle_.size(); j++)
    {
      for (k=0; k < particle_[j].size(); k++)
	{
	  particle_[j][k]->rotate(c , s);
	}
    }
}

void PARTICLE_BEAM::set_angle_particles(float x_angle, float y_angle,float delta_z)
{
    float x_step,y_step,x_offset,y_offset;
    unsigned int i,j;
    int n_slice = (int) particle_.size();
    x_step=x_angle*delta_z;
    y_step=y_angle*delta_z;
    x_offset=-0.5*(n_slice-1)*x_step;
    y_offset=-0.5*(n_slice-1)*y_step;
    for (i=0;i<particle_.size();i++)
      {
	for (j=0;j< particle_[i].size();j++)
	  {
	    particle_[i][j]->apply_position_offset(x_offset, y_offset);
	}
	x_offset += x_step;
	y_offset += y_step;
    }
}

void  PARTICLE_BEAM::set_particles_offset(int slice, float offset_x,float offset_y,float waist_x, float waist_y)
{
  unsigned int k;
  float vx, vy;
  for (k = 0; k < particle_[slice].size(); k++)
    {
      particle_[slice][k]->apply_position_offset(offset_x, offset_y);
      particle_[slice][k]->velocities(vx, vy);
      particle_[slice][k]->apply_position_offset(vx*waist_x, vy*waist_y);
    }
}

// (end of the block above)

void PARTICLE_BEAM::backstep (int beam,int trav_focus, float max_z, float step,  int timestep, float scal_step[2])
{
// note: the particles are tracked only if they reached the head of the other bunch, thus the backstepping distance has to be halfed     
  if (trav_focus) 
    {
      backParticlesBefore(max_z);
    }
  else 
    {
      if (ZPOS) 
	{
	  backSlicesWithZ(max_z);
	} 
      else 
	{
	  backSlices(beam, step,  timestep, scal_step);
	}
    }  
}

void PARTICLE_BEAM::backstep2 (int nbeam, float max_z, float step,  int timestep, float scal_step[2])
{
  
// note: the particles are tracked only if they reached the head of the other bunch, thus the backstepping distance has to be halfed     

  //#ifdef ZPOS
  if (ZPOS)
    {
      backSlicesWithZ2(max_z);
    }
  else
    {
      //#else
      backSlices2(nbeam, step,  timestep, scal_step);
    }
  //#endif
      
}

int PARTICLE_BEAM::store_beam(std::string name) const
{
  
  int number = 0;// it was unsigned long
  //
  FILE_IN_OUT filout;
  filout.open_file(name, "w");
  int h;
  int k;
  int j;
  std::vector<unsigned long int> order;
  number = numberOfParticles();
  rndm_generator_->getShuffledIntegerSequence(number, order);

   for (h = 0; h < number; h++) 
     {
       index_slice( order[h] -1 , k,j);
       filout.save_object_on_persistent_file( particle_[j][k]);
     }

//   for (j = 0; j < particle_.size(); j++)
//     {
//       for (k= 0; k < particle_[j].size(); k++) 
// 	{
// 	  filout.save_object_on_persistent_file( particle_[j][k]); 
// 	  number++;
// 	}
//     }
  filout.close();
  return (int)number;
}


void PARTICLE_BEAM::store_coherent_beam(std::string name) const
{
  //unsigned long int number;
  unsigned long int i,n;
  FILE_IN_OUT filout;
  filout.open_file(name, "w");
  n = coherent_.size();
  unsigned long int k;
  for (i=0; i<n ;i++)
    {
      for (k=0; k < coherent_[i].size(); k++) 
	{
	  filout.save_object_on_persistent_file(coherent_[i][k]); 
	}       
    }
    filout.close();
}


void PARTICLE_BEAM::store_trident_beam(std::string name) const
{
  //unsigned long int number;
  unsigned long int i,n;
  FILE_IN_OUT filout;
  filout.open_file(name, "w");
  n = trident_.size();
  unsigned long int k;
  for (i=0; i<n ;i++)
    {
      for (k=0; k < trident_[i].size(); k++) 
	{
	  filout.save_object_on_persistent_file(trident_[i][k]); 
	}       
    }
    filout.close();
}


void PARTICLE_BEAM::dump_beam(std::string name, int istep, int every_particle, int timestep, float step, float max_z, int  sign_label)
{
  FILE_IN_OUT file;
  file.open_file(name,"w");
  int nopart_in_slice, slice;
  int j,k;
  float dz0;
  dz0 = timestep*step;
  int coupure;
  int jini, jincrement;
  int n_slice = number_of_slices();
  int nopartIni, nb_part_in_slice;
  //
  if (every_particle>1)    jincrement = every_particle;		
  else  jincrement = 1;
  if( (istep < n_slice) || (istep == n_slice && every_particle <= 1) )
    {
      if (istep < n_slice)
	if ( every_particle>1 ) coupure = istep;
	else coupure = istep-1;
      else coupure = istep-1;
      nopartIni =0;
      for (k=0;k<=coupure;k++) 
	{
	  if (every_particle>1) 
	    jini = (  (nopartIni+every_particle-1)/every_particle )*every_particle;
	  else  jini = nopartIni;
	  nb_part_in_slice = numberOfParticlesOfSlice(k);
	  for ( j= jini; j< nopartIni + nb_part_in_slice ; j += jincrement )
	    {
	      index_slice(j, nopart_in_slice, slice);


	      PARTICLE* part = newPart(*particle_[slice][nopart_in_slice]);
	      part->set_z_for_dump(istep,dz0, max_z, sign_label ); 
	      file.save_object_on_persistent_file( part); 
	      delete part;
	    }
	  nopartIni += nb_part_in_slice;
	}
      for (k=coupure+1;k< n_slice;k++) 
	{
	  if (every_particle>1) 
	    jini = (  (nopartIni+every_particle-1)/every_particle )*every_particle;
	  else  jini = nopartIni;
	  nb_part_in_slice = numberOfParticlesOfSlice(k);
	  for (j= jini;j< nopartIni + nb_part_in_slice;j += jincrement)
	    {
	      index_slice(j, nopart_in_slice, slice);
	      PARTICLE* part = newPart(*particle_[slice][nopart_in_slice]);
	      part->set_z_velocity_corrected_for_dump(slice, istep,0, dz0, max_z,sign_label );
	      file.save_object_on_persistent_file( part); 
	      delete part;
	    }
	  nopartIni += nb_part_in_slice;
	}
    }
  else
    {
      coupure = istep - n_slice;
      nopartIni =0;
      for (k=0; k <= coupure;k++)
	{
	  if (every_particle>1) 
	    jini = (  (nopartIni+every_particle-1)/every_particle )*every_particle;
	  else  jini = nopartIni;
	  nb_part_in_slice = numberOfParticlesOfSlice(k);
	  for (j=jini;j< nopartIni + nb_part_in_slice ;j+=jincrement)
	    {
	      index_slice(j, nopart_in_slice, slice);
	      PARTICLE* part = newPart(*particle_[slice][nopart_in_slice]);
	      part->set_z_velocity_corrected_for_dump(slice, istep, n_slice, dz0, max_z,sign_label );


	      file.save_object_on_persistent_file( part); 
	      delete part;
	    }
	  nopartIni += nb_part_in_slice;
	}
      for (k = coupure+1; k < n_slice; k++) 
	{
	  if (every_particle>1) 
	    jini = (  (nopartIni+every_particle-1)/every_particle )*every_particle;
	  else  jini = nopartIni;
	  nb_part_in_slice = numberOfParticlesOfSlice(k);
	  for (j= jini ;j < nopartIni + nb_part_in_slice;j+= jincrement)
	    {
	      index_slice(j, nopart_in_slice, slice);

	      PARTICLE* part = newPart(*particle_[slice][nopart_in_slice]);
	      part->set_z_for_dump(istep,dz0, max_z, sign_label );
	      file.save_object_on_persistent_file( part); 
	      delete part;
	    }
	  nopartIni += nb_part_in_slice;
	}
    }
  file.close();
}     
  

void PARTICLE_BEAM::ang_dis(unsigned int n_bin, std::vector< std::vector<float> >& bin ) const
 {
   //FILE *datei;
   //   unsigned int n_bin=200;
   //   float bin[3][200];
   int j1,j2,j3;
   unsigned int i,k;
   float vx, vy;
   //   return;

   for (k=0; k < bin.size() ; k++)
     {
       bin[k].clear();
     }
   bin.clear();
   bin.resize(3);
   for (k=0; k < bin.size() ; k++)
     {
       bin[k].resize(n_bin);
     }

   for (i=0;i<n_bin;i++)
     {
       bin[0][i]=0.0;
       bin[1][i]=0.0;
       bin[2][i]=0.0;
     }
   for (k = 0; k < particle_.size(); k++)
     {
       for (i=0;i< particle_[k].size();i++)
	 {
	   particle_[k][i]->velocities(vx, vy);

	   j1=(int)floor(vx*100000.0+100.0);
	   j2=(int)floor(vy*100000.0+100.0);
	   j3=(int)floor(sqrt(vx*vx + vy*vy)*100000.0+100.0);

	   if ((j1>=0)&&(j1< (int)n_bin))
	     {
	       bin[0][j1] += 1.0;
	     }
	   if ((j2>=0)&&(j2< (int)n_bin))
	     {
	       bin[1][j2] += 1.0;
	     }
	   if ((j3>=0)&&(j3< (int)n_bin))
	     {
	       bin[2][j3] += 1.0;
	     }
	 }
     }
 }


std::string PARTICLE_BEAM::output_flow() const
{
  std::ostringstream out;
  int i,nslice,j, n_tot=0;
  unsigned int k;
  double esum,esum_tot=0.0;
  out <<  " ---------- " << std::string("particle beam") << " : " << std::endl;

  nslice = coherent_.size();
  for( j=0; j <nslice; j++){
    i=0;
    esum=0.0;
    const   std::vector<PARTICLE*>& point = coherent_[j];
    for (k= 0; k < point.size() ; k++)
      {
	i++;
	esum += point[k]->energy();
      }
    if (i) out << "slice " << j << " contains " << i << "  coherent particles with " << esum << " GeV " << std::endl;
    esum_tot += esum;
    n_tot += i;
  }
  out << " total number of coherent particles is " << n_tot << " with " << esum_tot << " GeV " << std::endl;

  out << " number of tracked macroparticles : " << number_of_particles_dispatched_in_slices_ << std::endl;

  return out.str();
}


double PARTICLE_BEAM::meanLostEnergy(float ebeam) const
  {
    unsigned int k,j;
    if (number_of_particles_dispatched_in_slices_ == 0) return 0.0;
    double esum = number_of_particles_dispatched_in_slices_*ebeam;

    for (j=0; j < particle_.size(); j++)
      {
	for (k=0; k < particle_[j].size() ;k++)
	  {
	    esum -= particle_[j][k]->energy();
	  }
      }
    esum /= (double)number_of_particles_dispatched_in_slices_;
    return esum;
  }


////////// class PHOTON_BEAM  //////////////////////////////



PHOTON_BEAM::PHOTON_BEAM(int n_sliceDATA) // AL: this seems useless : end_(NULL) 
{
  n_slice_=n_sliceDATA;
  slice_photon_vector_ = std::vector< std::vector<PHOTON> > (n_slice_);
  photon_count_by_slice_ = std::vector<PHOTON_COUNTER>(n_slice_);
}

PHOTON_BEAM::~PHOTON_BEAM()
{

}
// This routine stores a photon into a photon beam. 
void PHOTON_BEAM::load_photons(std::string filename, int type_of_beam, float delta_z, float max_z, int n_cell_z)
{
  float ener,vx,vy,x,y,z,hel;//, dummy;
  float xt, yt;
  int slice;
  FILE_IN_OUT filin;
  filin.open_file(filename, "r");
  PARTICLE_INTERFACE particle;
  while(filin.read_particle(particle))
    {
      particle.get_parameters(ener,x ,y , z,vx ,vy);
      hel = particle.get_helicity();
      z *= 1.0e3;
      slice=(int)floor(z/delta_z+0.5*(float)n_cell_z);
      xt= x-0.5*vx*(max_z+z);
      yt = y-0.5*vy*(max_z+z);
      
      if(ener < 0.0)
	{
	  ener = -ener;
	  if(type_of_beam == 2)
	    {
	      new_loaded_photon(slice, ener,hel, xt, yt, vx, vy);
	    } 
	}
      else 
	{
	  if(type_of_beam == 1)
	    {
	      new_loaded_photon(slice, ener,hel, xt, yt, vx, vy);
	    } 
	}   
    }
  filin.close();
}


void PHOTON_BEAM::dump_photons(std::string name,int istep,int every_particle, int timestep, float step, float max_z, int  sign_label)
{
  FILE_IN_OUT file;
  file.open_file(name,"w");
  int j,k;
  float dz0;
  dz0 = timestep*step;
  if( istep < n_slice_ )
    {
      for (k=0; k<= istep; k++)
	{
	  for (j=0; j < sizeOfSlice(k); j++)
	    {
	      if(getPhotonVector(k)[j].no()%every_particle==0)
		{
		  PHOTON* phot = new PHOTON(getPhotonVector(k)[j]);
		  phot->set_z_for_dump(istep,dz0, max_z, sign_label ); 
		  file.save_object_on_persistent_file( phot); 
		  delete phot;
		}
	    }
	}
      for (k=istep+1;k < n_slice_;k++)
	{
	  for (j=0; j < sizeOfSlice(k); j++)
	    {
	      if(getPhotonVector(k)[j].no()%every_particle==0)
		{
		  PHOTON* phot = new PHOTON(getPhotonVector(k)[j]);
		  phot->set_z_velocity_corrected_for_dump(k,istep,0,dz0,max_z,sign_label);
		  file.save_object_on_persistent_file( phot); 
		  delete phot;
		}
	    }
	}
    }
  else
    {
      for (k = 0; k <= istep - n_slice_; k++) 
	{
	  for (j=0; j < sizeOfSlice(k); j++)
	    {
	      if(getPhotonVector(k)[j].no()%every_particle==0)
		{
		  PHOTON* phot = new PHOTON(getPhotonVector(k)[j]);
		  phot->set_z_velocity_corrected_for_dump(k,istep,n_slice_,dz0,max_z,sign_label);
		  file.save_object_on_persistent_file( phot); 
		  delete phot;
		}
	    }
	}
      for (k = istep-n_slice_ + 1; k < n_slice_; k++) 
	{
	  for (j=0; j < sizeOfSlice(k); j++)
	    {
	      if(getPhotonVector(k)[j].no()%every_particle==0)
		{
		  PHOTON* phot = new PHOTON(getPhotonVector(k)[j]);
		  phot->set_z_for_dump(istep,dz0,max_z,sign_label); 
		  file.save_object_on_persistent_file( phot); 
		  delete phot;
		}
	    }
	}
    }
  file.close();
}

// int PHOTON_BEAM::store_photon(std::string name) const
// {

//   int number = 0; //unsigned long int number = 0;
//   double en_sum= 0;
//   int n_tot = 0;
//   //
//   FILE_IN_OUT filout;
//   filout.open_file(name, "w");
//   int h;
//   int k;
//   //int j;
//   std::vector<unsigned long int> order;
//   //photon_info(en_sum,number);
//   //rndm_generator_->getShuffledIntegerSequence(number, order);
//   //   for (h = 0; h < number; h++)
//   //     {
//   //       //  index_slice( order[h] -1 , k,j);
//   //       filout.save_object_on_persistent_file( getPhotonVector(k)[j]);
//   //     }
//   for (h=0; h < n_slice_; h++)
//     {
//       photon_info_slice(h,en_sum,number,n_tot);
//       for (k=0;k<n_tot;k++)
//         {
//           if(getPhotonVector(h)[k].energy() > 0.0 )
//             {

//               filout.save_object_on_persistent_file( &getPhotonVector(h)[k]);
//             }
//         }
//     }
//   filout.close();
//   return (int)number;
// }


PARTICLE_BEAM::IncomingMapResliceReceipt
PARTICLE_BEAM::reslice_after_incoming_map(float delta_z_nm)
{
  if (xy_adopted_)
    throw std::logic_error("incoming-map reslicing must precede XY adoption");
  if (!std::isfinite(delta_z_nm) || !(delta_z_nm > 0.0f) || particle_.empty())
    throw std::invalid_argument("incoming-map reslicing requires finite positive slice spacing and a nonempty slice lattice");
  require_valid_particle_id_sidecar();
  for (std::size_t s=0; s<coherent_.size(); ++s)
    if (!coherent_[s].empty())
      throw std::logic_error("incoming-map reslicing is forbidden after coherent production");
  for (std::size_t s=0; s<trident_.size(); ++s)
    if (!trident_[s].empty())
      throw std::logic_error("incoming-map reslicing is forbidden after trident production");

  IncomingMapResliceReceipt receipt;
  const std::size_t slices=particle_.size();
  const double half=0.5*static_cast<double>(slices)*delta_z_nm;
  std::vector<std::size_t> destination, counts(slices,0);
  destination.reserve(number_of_particles_dispatched_in_slices_);
  std::size_t outside=0, nonfinite=0;
  for (std::size_t s=0; s<slices; ++s)
    for (std::size_t k=0; k<particle_[s].size(); ++k)
      {
        if (particle_[s][k]==NULL)
          throw std::logic_error("incoming-map reslicing encountered a null primary");
        const double z=particle_[s][k]->z();
        ++receipt.particles;
        if (!std::isfinite(z)) { ++nonfinite; continue; }
        receipt.maximum_abs_z_nm=std::max(receipt.maximum_abs_z_nm,std::fabs(z));
        if (z < -half || !(z < half)) { ++outside; continue; }
        const double coordinate=z/static_cast<double>(delta_z_nm)+0.5*slices;
        const double floored=std::floor(coordinate);
        // Guard roundoff at the open upper boundary before integer conversion.
        if (!(floored>=0.0) || !(floored<static_cast<double>(slices)))
          { ++outside; continue; }
        const std::size_t target=static_cast<std::size_t>(floored);
        destination.push_back(target);
        ++counts[target];
        if (target!=s) ++receipt.moved;
        const std::size_t shift=target>s ? target-s : s-target;
        receipt.maximum_slice_shift=std::max(receipt.maximum_slice_shift,shift);
      }
  if (outside || nonfinite)
    {
      std::ostringstream why;
      why << "incoming-map reslicing refused: beam=" << beam_label_
          << " particles=" << receipt.particles << " outside=" << outside
          << " nonfinite=" << nonfinite << " domain_nm=[" << -half
          << ',' << half << ") max_abs_z_nm=" << receipt.maximum_abs_z_nm
          << "; enlarge cut_z and choose n_z explicitly; no particles were dropped";
      throw std::out_of_range(why.str());
    }
  if (receipt.particles!=number_of_particles_dispatched_in_slices_)
    throw std::logic_error("incoming-map reslicing particle count disagrees with population bookkeeping");
  if (!receipt.moved) return receipt; // preserve exact original insertion order

  std::vector<std::vector<PARTICLE*> > reordered(slices);
  std::vector<std::vector<PRIMARY_PARTICLE_ID> > reordered_ids(slices);
  for (std::size_t s=0; s<slices; ++s)
    { reordered[s].reserve(counts[s]); reordered_ids[s].reserve(counts[s]); }
  std::size_t index=0;
  for (std::size_t s=0; s<slices; ++s)
    for (std::size_t k=0; k<particle_[s].size(); ++k,++index)
      {
        const std::size_t target=destination[index];
        reordered[target].push_back(particle_[s][k]);
        reordered_ids[target].push_back(particle_ids_[s][k]);
      }
  // Both temporary containers are complete before the noexcept commit.
  // Only non-owning pointer lists move: no particle is cloned or destroyed.
  particle_.swap(reordered);
  particle_ids_.swap(reordered_ids);
  return receipt;
}
