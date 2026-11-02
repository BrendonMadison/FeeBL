#ifndef PARTICLES_SEEN
#define PARTICLES_SEEN
#include <iostream>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "define.h"
#include "typeDefs.h"
#include "abstractIOclass.h"
#include "abstractParticle.h"
#include "physicalTools.h"
#include "mathematicalEntities.h"
#include "tridentCPP.h"
#include "secondaryMomentumCPP.h"

class PARTICLE : public ABSTRACT_PARTICLE, public ABSTRACT_IO_CLASS
{

  protected :

    float ups_;
    TRIDENT trident_;
  

  PARTICLE (float xi, float yi, float zi, float vxi,float vyi, float energyi) : ABSTRACT_PARTICLE (xi, yi, zi,vxi, vyi,energyi)  {;}
  
     
 public:

  PARTICLE() : ABSTRACT_PARTICLE(),ABSTRACT_IO_CLASS(),ups_(0.0) {;}

  // return the quantity dz/radius
  inline float advanceDueToEBfield(TRIVECTOR EBfield, float distance, float scal_step_local)
    {
      float deltaVx, deltaVy;
      // divide by energy in GeV.
      // the total field (electric+magnetic) is equivalent to 
      // 2 X electric field

      advanceVelocities(-2.0*EBfield(0), -2.0*EBfield(1), distance, deltaVx, deltaVy);
      advancePosition(distance*scal_step_local); 
      float dz = distance*1e-9*scal_step_local;
      float dzOnRadius = sqrt(deltaVx*deltaVx+deltaVy*deltaVy);
      float radius_i = dzOnRadius/dz;
      float upsilon=LAMBDA_BAR/(EMASS*EMASS)*energy_*energy_*radius_i;
      ups_ = upsilon;
      return dzOnRadius;
    }

  inline float advanceTridentDueToEBfield(TRIVECTOR EBfield, float distance, float scal_step_local)
    {
      float deltaVx, deltaVy;
      // divide by energy in GeV.
      // the total field (electric+magnetic) is equivalent to 
      // 2 X electric field
      
      advanceVelocities(-2.0*EBfield(0), -2.0*EBfield(1), distance, deltaVx, deltaVy);
      advancePosition(distance*scal_step_local); 
      float dz = distance*1e-9*scal_step_local;
      float dzOnRadius = sqrt(deltaVx*deltaVx+deltaVy*deltaVy);
      float radius_i = dzOnRadius/dz;
      float upsilon=LAMBDA_BAR/(EMASS*EMASS)*energy_*energy_*radius_i;
      ups_ = upsilon;
      return dzOnRadius;
    }

  inline void apply_position_offset(float dx, float dy)
    {
      pos_[0] += dx;
      pos_[1] += dy;
    }
  
  
  // virtual: PARTICLE_WITH_SPIN must carry spin_ through the same rotation.
  // A non-virtual rotate() called through the PARTICLE* that PARTICLE_BEAM
  // stores would silently dispatch here and leave the spin behind.
  virtual inline void rotate(float cosa, float sina)
    {
      float x = pos_[0];
      float y = pos_[1];
      pos_[0] = cosa*x + sina*y;
      pos_[1] = -sina*x + cosa*y;
      float vx = vx_;
      float vy = vy_;
      vx_ = cosa*vx + sina*vy;
      vy_ = -sina*vx + cosa*vy;
    }


  inline void adjustEnergyToMinimum(float emin)
    {
      if (energy_ < emin)
	{
	  std::cout << "PARTICLE:: e_low2: " << energy_ << std::endl;
	  energy_ = emin;
	}
    };


  virtual inline void setParticle(float x,float y,float z,float vx,float vy,float energy)
    {
      ABSTRACT_PARTICLE::set(energy, x, y,z,vx,vy);
    }
  
  virtual inline void setParticle(const PARTICLE_INTERFACE& part)
    {
      ABSTRACT_PARTICLE::set(part);
    }


  virtual inline void setParticle(float x,float y,float z,float vx,float vy,float energy, const TRIDVECTOR* /*dummy*/)
    {
      setParticle(x,y,z,vx,vy,energy);
    }
  inline float getEnergy() const {return energy_;}
  inline void setUps(float upsilon) { ups_ = upsilon;}
  inline float getUps() const { return ups_;}
  
  
  virtual inline const TRIDVECTOR& getSpin() const
    {
      std::cerr << " PARTICLE::get_spin : a PARTICLE has no spin, use PARTICLE_WITH_SPIN " << std::endl;
      exit(1);
/*       const TRIDVECTOR tv(0.0,0.0,0.0); */
/*       return tv; */
    }


  virtual inline void rotateBMT(TRIDVECTOR /*Efield*/, TRIDVECTOR /*Bfield*/, float /*charge_sign*/, float /*dt*/, float /*betaz*/ = 1.0f) {;}
  
  // photonPol (optional): three emission-frame Stokes parameters per photon, in step with photonEnergies
  inline void beamstrahlung(TRIDVECTOR /*Efield*/, TRIDVECTOR /*Bfield*/, float dzOnRadius,  float emin, std::vector<float>& photonEnergies, RNDM& rndm_generator, std::vector<float>* photonPol = NULL)
    {
      energy_ = PHYSTOOLS::synrad_no_spin_flip(ups_, energy_, dzOnRadius ,photonEnergies, rndm_generator, photonPol);
      adjustEnergyToMinimum(emin);
    }
    
  inline void createTridents(double dz,std::vector<float>* electrons,std::vector<float>* positrons,std::vector<float>* virt,RNDM& rndm_generator)
    {
      trident_.createTridents(&energy_,ups_,dz,electrons,positrons,virt,rndm_generator);
    }

  inline void createTridents(double dz,std::vector<float>* electrons,std::vector<float>* positrons,RNDM& rndm_generator)
    {
      trident_.createTridents(&energy_,ups_,dz,electrons,positrons,rndm_generator);
    }
  
  virtual inline void beamstrahlungSokolov(TRIDVECTOR /*Efield*/, TRIDVECTOR /*Bfield*/, float /*dzOnRadius*/,  float /*emin*/, float /*charge_sign*/, std::vector<float>& /*photonEnergies*/, RNDM& /*rndm_generator*/, float /*betaz*/ = 1.0f, std::vector<float>* /*photonPol*/ = NULL)
    {
      std::cerr << " illegal call of PARTICLE::beamstrahlungSokolov " << std::endl;
      exit(1);
    }
  

  virtual std::string  output_flow() const
    {
      std::cerr << "  output_flow not programmed for class PARTICLE " << std::endl;
      return std::string(" ");
    }

  virtual std::string persistent_flow() const
    {
      std::ostringstream out;
      out <<  energy_ << " " << pos_[0]*1e-3 << " " << pos_[1]*1e-3 << " " << zpos_*1e-3 << " " << vx_*1e6 << " " << vy_*1e6;
      return out.str();
    }
  

  virtual  inline void init_from_input_file(float energy, float xpos, float ypos, float zpos, float vx, float vy) 
    {
      float facPosition = 1.0e3;
      float facVelocity = 1.0e-6;
      set(energy, xpos, ypos,zpos,vx,vy);
      // transform to nm and rad
      transform_to_internal_units(facPosition, facVelocity);
    }

};


class PARTICLE_WITH_SPIN : public PARTICLE
{
  protected :
  TRIDVECTOR spin_;

  PARTICLE_WITH_SPIN (float xi, float yi, float zi, float vxi,float vyi, float energyi, TRIDVECTOR sp) : PARTICLE (xi, yi, zi, vxi, vyi, energyi) 
    {
      spin_ = sp;
    }
  
  
 public:
  
  PARTICLE_WITH_SPIN() : PARTICLE() {;}
  
 
  virtual inline const TRIDVECTOR& getSpin() const
    {
      return spin_;
    }
  
  
  virtual inline void setParticle(float x,float y,float z,float vx,float vy,float energy, const TRIDVECTOR* polar)
    {
      ABSTRACT_PARTICLE::set(energy, x, y,z,vx,vy);
      spin_ = *polar;
    }
  
  
  virtual inline void setParticle(const PARTICLE_INTERFACE& part)
    {
      ABSTRACT_PARTICLE::set(part);
      spin_ = part.getSpin();
    }
  
  
  inline void setSpin(TRIDVECTOR sp)
    {
      //      std::cout << " appel setSpin " << std::endl;
      spin_ = sp;
    }


  virtual  inline void init_from_input_file(float /*energy*/, float /*xpos*/, float /*ypos*/, float /*zpos*/, float /*vx*/, float /*vy*/) 
    {
      std::cerr << " reading from input file not programmed for PARTICLE_WITH_SPIN " << std::endl;
      exit(1);
      
    }

  virtual void rotateBMT(TRIDVECTOR Efield, TRIDVECTOR Bfield, float charge_sign, float dt, float betaz = 1.0f);

  // A transverse roll (angle_phi) is a rotation about z, and the spin must turn
  // with it. PARTICLE::rotate moves position and velocity only; leaving spin_
  // behind expresses the spin in a different azimuthal frame from the velocity
  // and the fields that rotateBMT subsequently combines with it, so the
  // precession -- and through it the longitudinal polarization -- comes out
  // wrong.
  //
  // s_z is invariant under a rotation about z, which is exactly why this never
  // surfaced as a direct error in P: the damage is entirely in the precession
  // that follows, not in the reported projection itself.
  //
  // spin_ is double. setComponents() takes floats and would narrow it, so the
  // components are assigned through operator(), which returns double&.
  virtual inline void rotate(float cosa, float sina)
    {
      PARTICLE::rotate(cosa, sina);
      const double sx = spin_(0);
      const double sy = spin_(1);
      spin_(0) =  cosa*sx + sina*sy;
      spin_(1) = -sina*sx + cosa*sy;
    }
  
  
  virtual inline void beamstrahlungSokolov(TRIDVECTOR Efield, TRIDVECTOR Bfield, float dzOnRadius,  float emin, float charge_sign, std::vector<float>& photonEnergies, RNDM& rndm_generator, float betaz = 1.0f, std::vector<float>* photonPol = NULL)
    {
      TRIDVECTOR ev1, ev2, ev3; 
      PHYSTOOLS::referenceSpin((double)vx_, (double)vy_, ev1, ev2, ev3, Efield, Bfield, charge_sign, (double)betaz);
      energy_ = PHYSTOOLS::synrad_spin_flip(ups_,energy_, ev1, ev2, ev3, spin_,dzOnRadius,photonEnergies, rndm_generator, photonPol);
      adjustEnergyToMinimum(emin);
    }
  
  
  virtual std::string persistent_flow() const
    {
      std::ostringstream out;
      out <<  PARTICLE::persistent_flow() << " " << spin_(0) << " " << spin_(1) << " " << spin_(2);
      return out.str();
    }
  
  
};

class PHOTON : public ABSTRACT_PARTICLE, public ABSTRACT_IO_CLASS
{
 public:
  typedef std::uint64_t id_type;

 private:
  TRIDVECTOR stokes_;
  float helicity_;
  
  id_type no_;
  
  
 public:
  
  PHOTON() 
    {
      helicity_=0.;
      no_=0;
    }

  PHOTON(  float xi, float yi,  float zi, float vxi,float vyi, float energyi, float helicityi, id_type no) : ABSTRACT_PARTICLE (xi, yi, zi, vxi, vyi,energyi)
    {
      helicity_ = helicityi;
      no_ = no;
    }
    
  PHOTON(float energy, const PARTICLE& particle, float helicityi, id_type no)
    {
      
      float x,y,vx,vy;
      particle.XYposition(x,y);
      particle.velocities(vx,vy);
      pos_[0] = x;
      pos_[1] = y;
      zpos_ = particle.z();
      vx_ = vx;
      vy_ = vy;
      energy_ = energy;
      helicity_ = helicityi;
      no_ = no;
    }
  
  virtual std::string  output_flow() const
    {
      std::cerr << "  output_flow not programmed for class PHOTON " << std::endl;
      return std::string(" ");
    }

  virtual std::string persistent_flow() const
    {
      std::ostringstream out;
      /*   out <<  energy_ << " " << pos_[0]*1e-3 << " " << pos_[1]*1e-3 << " " << zpos_*1e-3 << " " << vx_*1e6 << " " << vy_*1e6 << " " << helicity_ << " " << no_; */
      // from original, to be modified perhaps
      out <<  energy_ << " " <<  vx_ << " " << vy_<< " ";
      
      return out.str();
    }

  virtual  inline void init_from_input_file(float energy, float xpos, float ypos, float zpos, float vx, float vy) 
    {
      set(energy, xpos, ypos,zpos,vx,vy);
      std::cerr << " reading from input file to be checked for photons " << std::endl;
      exit(1);
    } 
  
  virtual  inline void init_from_input_file(float energy, float xpos, float ypos, float zpos, float vx, float vy, float hel) 
    {
      
      PHOTON::init_from_input_file(energy, xpos, ypos, zpos, vx, vy);
      helicity_ = hel;
      //  no_ = 0;
      std::cerr << " reading from input file to be checked for photons " << std::endl;
      exit(1);
    } 
  
  virtual inline const TRIDVECTOR& getSpin() const 
    { 
      std::cerr << " PHOTON::getSpin : not programmed " << std::endl;
      exit(1);
/*       const TRIDVECTOR tv(0.0,0.0,0.0); */
/*       return tv; */
    }
  
  
  inline id_type no() const {return no_;}

  // photon_polarisation=1: lab-frame Stokes parameters (s1, s2, s3) in the convention of
  // qedbg_photon_polarisation.h; zero (unpolarised) unless set at emission
  inline void setStokes(const float s[3]) { stokes_.setComponents(s[0], s[1], s[2]); }
  inline void getStokes(float s[3]) const { s[0] = (float)stokes_(0); s[1] = (float)stokes_(1); s[2] = (float)stokes_(2); }
  
};

inline bool operator < ( const PARTICLE& p1, const PARTICLE& p2)
{
  return (p1.z() < p2.z());
}


class PAIR_PARTICLE :  public ABSTRACT_PARTICLE, public ABSTRACT_IO_CLASS
{
  feebl::secondary_transport::Momentum transport_;
  double last_electric_scale_=1.0;
  
  float velocityz_;
  
  int process_;
  int label_;

  int icharge_;
  bool imported_spin_enabled_=false;
  TRIDVECTOR imported_spin_;
  TRIDVECTOR imported_electric_, imported_magnetic_;
  double imported_max_chi_=0.0;
  unsigned long long imported_spin_steps_=0;
  bool imported_field_audit_=false;
  mutable unsigned long long imported_field_gathers_[3]={0,0,0};
  bool last_rescaling_ok_;

  // MANU, to retrieve the time :
   int slice1_ ;
   int slice2_ ;

 public:
  
  PAIR_PARTICLE() : ABSTRACT_PARTICLE ()
    {
      velocityz_ =0.0;
      icharge_ = 0;
      process_ = -99;
      label_ = -99;
      last_rescaling_ok_ = true;

      // -- MANU :
	slice1_ = -1 ;
	slice2_ =-1 ;
      // ----
    }
  
  PAIR_PARTICLE(int label, int index_of_process, float x,float y,float z, float vx,float vy,float vz, float energy) : ABSTRACT_PARTICLE (x,y,z,vx,vy,energy)
    {
      velocityz_ =vz;
      if (energy > 0.0) 
	{
	  icharge_ = 0;
	}
      else 
	{
	  icharge_ = 1;
	  energy_ = -energy;
	}
      process_ = index_of_process;
      label_ = label;
      last_rescaling_ok_ = true;

      slice1_ = -1;
      slice2_ = -1;

    }
  virtual ~PAIR_PARTICLE () {;}

  // Generated momenta are authoritative; initialize before storage/copying.
  // Imported records instead use their total energy and direction. Both
  // initializers preserve the signed-energy charge convention of the ctor.
  void initialize_transport_momentum(double mass,double px,double py,double pz);
  void initialize_transport_from_energy(double mass);
  double transport_energy() const;
  double transport_momentum(int axis) const;
  double transport_kinetic_energy() const;
  void apply_radiation_recoil(double photon_energy_GeV);

  // Keep supported direct mutations synchronized with canonical momentum.
  // Mutating through an ABSTRACT_PARTICLE base reference is unsupported and
  // is detected at the next transport entry, rather than silently ignored.
  void setEnergy(float energy);
  void setVelocities(float vx,float vy);
  void advanceVelocities(float fx,float fy,float step);
  void advanceVelocities(float fx,float fy,float step,float& dvx,float& dvy);
  void razEnergy();
  
  virtual const TRIDVECTOR& getSpin() const
  {
    if (!imported_spin_enabled_)
      throw std::logic_error("Secondary spin was not initialized");
    return imported_spin_;
  }

  // Only explicitly polarized imported electron/positron records use this
  // state. Radiation is disabled for them until a secondary spin-flip model
  // with a full three-dimensional emission basis has been validated.
  void initialize_imported_spin(const TRIDVECTOR& spin);
  bool imported_spin_enabled() const { return imported_spin_enabled_; }
  double imported_helicity() const;
  double imported_max_chi() const { return imported_max_chi_; }
  unsigned long long imported_spin_steps() const { return imported_spin_steps_; }
  int force_charge_sign() const { return icharge_ ? -1 : 1; }
    
  virtual std::string  output_flow() const
    {
      std::cerr << "  output_flow not programmed for class PAIR_PARTICLE " << std::endl;
      return std::string(" ");
    }
  
  
  virtual std::string persistent_flow() const
  {
    std::ostringstream out;
    float ener;
    if ( icharge_) ener = -energy_;
    else ener = energy_;

    out <<  ener << " " <<  vx_ << " " << vy_ << " " << velocityz_ << " " << pos_[0] << " " << pos_[1] << " " << zpos_ << " " << process_ << " " <<  slice1_ <<" " << slice2_  ;
    return out.str();
  }

  inline int get_label() const {return label_;}
  // Opt-in diagnostic counters follow imported particle copies. They never
  // enter force, stepping, admission, or radiation calculations.
  inline void enable_imported_field_audit() { imported_field_audit_=true; }
  inline void record_imported_field_gather(int grid) const {
    if(imported_field_audit_)++imported_field_gathers_[grid<0?2:(grid==0?0:1)];
  }
  inline unsigned long long imported_field_gathers(int category) const {return imported_field_gathers_[category];}
  inline int get_process() const {return process_;}
  
  inline bool last_rescaling_ok() const {return  last_rescaling_ok_;}
  inline float unsigned_energy() const { return energy_;}
  
  inline float signed_energy() const
    {
      float ener;
      if ( icharge_) ener = -energy_;
      else ener = energy_;
      return ener;
    }

  inline void set_slices12( int i1, int i2 ){ 
	slice1_ = i1 ;
	slice2_ = i2;
	return ;
  }

  inline void get_slices12( int &i1,  int &i2 ) {
	i1 = slice1_ ;
	i2 = slice2_ ;
	return;
  }



  inline float Zvelocity() const 
    {
      return  velocityz_;
    }
  
  inline float velocity_q() const 
    {
      return vx_*vx_ + vy_*vy_ + velocityz_*velocityz_;
    }
  

  
  inline void advancePosition(float step)
    {
      ABSTRACT_PARTICLE::advancePosition(step);
      zpos_ += velocityz_*step;
    }

  void apply_electric_field_on_pair(float step_2, float ex, float ey); 
  
  float  scale_pair(float vold2, double mass);
  
  inline void scale_velocities_sync(double scal, double& vx0, double& vy0, double& vz0) const
    {
#ifdef PAIR_SYN
  vx0 *= scal;
  vy0 *= scal;
  vz0 *= scal;
#endif
    }


  void synchrotron_radiation(float step, double mass, float step_fraction, double vx0, double vy0, double vz0, std::vector<float>* photon_e, RNDM& rndm_generator);

  void synchrotron_radiation(float step, double mass, float step_fraction, double vx0, double vy0, double vz0, RNDM& rndm_generator);
 
  float apply_magnetic_field_on_pair(float fac_theta, float step_q, float bx, float by);

  float apply_final_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, float thetamx, RNDM& rndm_generator);

  float apply_initial_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, std::vector<float>* photon_e, RNDM& rndm_generator);

  float apply_initial_half_step_fields(float step, double mass, float ex,float ey, float bx, float by, RNDM& rndm_generator);
 
  float apply_full_step_fields(float step, double mass, float ex,float ey, float bx, float by, std::vector<float>* photon_e, RNDM& rndm_generator); 

  float apply_full_step_fields(float step, double mass, float ex,float ey, float bx, float by, RNDM& rndm_generator); 

  void update_energy(double mass);

 private:
  void ensure_transport(double mass);
  void validate_transport_mirrors() const;
  void sync_transport_mirrors();
  void prepare_imported_spin_fields(float ex,float ey,float bx,float by);
  void rotate_imported_spin(double ds_nm, const TRIDVECTOR& electric,
                            const TRIDVECTOR& magnetic, double mass);
  double electric_kick(float ds_nm,double mass,float ex,float ey);


};

class  PHOTON_DATA
{
  int i_,n_,j_;
  float scal_,scal_i_,scal2_,scal2_i_,emin_;
  
  
 public:
  
  PHOTON_DATA()
    {
      n_       = 1;
      j_       = 1;
      scal_    = 1.0*n_;
      scal_i_  = 1.0/scal_;
      emin_    = 50.0;
      scal2_   = 1.e3;
      scal2_i_ = 1.0;
    }
  
  
  inline void updateXX() 
    { 
      if (!(--j_)) j_ = n_; 
      i_ = j_; 
    }

  inline void nTOi() 
    {
      i_ = n_; 
    }

  inline float get_scal2() const 
    {
      std::cout << "PHOTON_DATA : scal2 " << std::endl; 
      return scal2_;
    }

};

class PHOTON_COUNTER
{
  double esum_;
  int number_;
  
 public:
  
  PHOTON_COUNTER() : esum_(0.0), number_(0) {}

  // This counter is deliberately non-atomic.  PHOTON_BEAM owns one counter
  // per longitudinal slice; a same-wave worker has exclusive ownership of
  // that slice.  Crossing-level totals are merged only after the wave barrier
  // and always in increasing slice order, avoiding both arrival-order IDs and
  // non-reproducible floating-point atomics.
  inline void addPhoton(float energy)
    {
      esum_ += static_cast<double>(energy);
      number_++;
    };

  inline void merge(const PHOTON_COUNTER& other)
    {
      esum_ += other.esum_;
      number_ += other.number_;
    }
  

  inline double getSum() const
    {
      return esum_;
    }
  inline int getNumber() const
    {
      return number_;
    }
};

// Photons from incoherent pairs
class TERTPHOTON : public ABSTRACT_PARTICLE, public ABSTRACT_IO_CLASS
{
  float vz_;
 
 public:
  
  TERTPHOTON() 
    {
      vz_=0.;
    }

  TERTPHOTON(  float xi, float yi,  float zi, float vxi,float vyi, float vzi, float energyi) : ABSTRACT_PARTICLE (xi, yi, zi, vxi, vyi,energyi)
    {
      vz_=vzi;
    }
    
  TERTPHOTON(float energy, const PAIR_PARTICLE& particle)
    {
      float x,y,vx,vy,norm;
      particle.XYposition(x,y);
      particle.velocities(vx,vy);
      pos_[0] = x;
      pos_[1] = y;
      zpos_ = particle.z();
      vx_ = vx;
      vy_ = vy;
      vz_=particle.Zvelocity();
      norm=sqrt(vx_*vx_+vy_*vy_+vz_*vz_);
      vx_/=norm;
      vy_/=norm;
      vz_/=norm;
      energy_ = energy;
    }
  
  virtual std::string  output_flow() const
    {
      std::cerr << "  output_flow not programmed for class TERTPHOTON " << std::endl;
      return std::string(" ");
    }

  virtual std::string persistent_flow() const
    {
      std::ostringstream out;
      /*   out <<  energy_ << " " << pos_[0]*1e-3 << " " << pos_[1]*1e-3 << " " << zpos_*1e-3 << " " << vx_*1e6 << " " << vy_*1e6 << " " << helicity_ << " " << no_; */
      // from original, to be modified perhaps
      out <<  energy_ << " " <<  vx_ << " " << vy_<< " " << vz_ << " "<<  pos_[0] << " " << pos_[1]<< " " << zpos_ << " ";
      
      return out.str();
    }
  
  virtual inline const TRIDVECTOR& getSpin() const 
    { 
      std::cerr << " PHOTON::getSpin : not programmed " << std::endl;
      exit(1);
/*       const TRIDVECTOR tv(0.0,0.0,0.0); */
/*       return tv; */
    }
};

#endif
