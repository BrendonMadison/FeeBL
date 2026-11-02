#ifndef ABSTRACTPARTICLE_SEEN
#define ABSTRACTPARTICLE_SEEN
#include <iostream>
#include <sstream>
#include <string>

#include "typeDefs.h"
#include "abstractIOclass.h"
#include "mathematicalEntities.h"

class ABSTRACT_BEAM
{

 protected:
 public: 
  ABSTRACT_BEAM() {;}
  virtual ~ABSTRACT_BEAM() {;}
  virtual int sizeOfPhotonSlice(int slice) const = 0;
  virtual int number_of_slices() const =0;
  virtual int numberOfParticlesOfSlice(int slice) const = 0;
  
};


class ABSTRACT_PARTICLE
{
  
 protected: 
  
  // unit of pos_[0], pos_[1] : nm 
  // Positions live behind a redirectable pointer so
  // the hot kernels can iterate a contiguous float array instead of chasing a
  // pointer into a 104-byte polymorphic object for 8 bytes of data.
  //
  // Measured in situ (perf, ILC 250): assignBeamSliceCIC 13.75 s -> 1.21 s when
  // it reads the SoA instead of the object graph. Note that figure came from a
  // build whose sync warmed the cache first, so treat 4-11x as the range.
  //
  // own_xy_ is the particle's OWN storage and pos_ points at it by default, so
  // a standalone or copy-constructed PARTICLE is entirely self-contained.
  // PARTICLE_BEAM re-points pos_ at its SoA slot for beam particles. Copies
  // never alias: the copy constructor resets pos_ to its own array.
  float own_xy_[2];
  float* pos_;
  
  float zpos_;
  
  // unit of velocity : radian (?)
  float vx_,vy_;

  // energy in GeV
  float energy_;

  inline void set(float energyi, float xi, float yi,float zi,float vxi,float vyi)
    {
      energy_ = energyi;
      pos_[0]= xi;
      pos_[1] = yi;
      zpos_ = zi;
      vx_ = vxi;
      vy_ = vyi;
    }
  
  inline void set(const ABSTRACT_PARTICLE& part)
    {
      set (part.energy_, part.pos_[0], part.pos_[1], part.zpos_, part.vx_, part.vy_);
    }
  
  
  // internal units seems to be nm for x,y,z
  inline void transform_to_internal_units(float facPosition, float facVelocity)
    {
      pos_[0] *= facPosition;
      pos_[1] *= facPosition;
      zpos_ *= facPosition;
      
      vx_ *= facVelocity;
      vy_ *= facVelocity;
    }
  
  
  ABSTRACT_PARTICLE (float xi, float yi,float zi, float vxi,float vyi,float energyi)
    {
      pos_ = own_xy_;   // FEEBL: self-hosted until a beam re-points it
      set(energyi, xi, yi,zi, vxi, vyi);
    }

  
  public : 
    
    ABSTRACT_PARTICLE () 
    {
      pos_ = own_xy_;   // FEEBL: self-hosted until a beam re-points it
      set(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); 
    }
  
  // FEEBL: a copy owns its own position storage. Propagating pos_ would alias
  // two particles onto one slot, which corrupts silently instead of crashing.
  ABSTRACT_PARTICLE (const ABSTRACT_PARTICLE& p)
    : zpos_(p.zpos_), vx_(p.vx_), vy_(p.vy_), energy_(p.energy_)
    {
      pos_ = own_xy_;
      own_xy_[0] = p.pos_[0];
      own_xy_[1] = p.pos_[1];
    }

  ABSTRACT_PARTICLE& operator = (const ABSTRACT_PARTICLE& p)
    {
      if (this != &p)
        {
          pos_[0] = p.pos_[0];
          pos_[1] = p.pos_[1];
          zpos_ = p.zpos_; vx_ = p.vx_; vy_ = p.vy_; energy_ = p.energy_;
        }
      return *this;
    }

  // FEEBL: hand this particle a slot in a beam's contiguous xy array. The two
  // current values move with it, so the redirect is value-preserving.
  inline void set_position_slot(float* slot)
    {
      slot[0] = pos_[0];
      slot[1] = pos_[1];
      pos_ = slot;
    }

  virtual ~ABSTRACT_PARTICLE () {;}
  
  inline float z() const {return zpos_;}
  inline void setZ( float zi) { zpos_ = zi;};
  
  virtual const TRIDVECTOR& getSpin() const = 0;
  
  
  inline void razEnergy() {energy_ = 0.0;}
  
  
  inline void setVelocities(float vxin, float vyin)
    {
      vx_ = vxin;
      vy_ = vyin;
    };
  
  inline void velocities(float& vxout, float& vyout) const 
    {
      vxout = vx_;
      vyout = vy_;
    };
  
  inline void velocities(double& vxout, double& vyout) const 
    {
      vxout = (double)vx_;
      vyout = (double)vy_;
    };
  
  inline void XYposition(PHI_FLOAT& xpos, PHI_FLOAT& ypos) const
    {
      xpos = (PHI_FLOAT)pos_[0];
      ypos = (PHI_FLOAT)pos_[1];
    };
  
  
  inline void XYposition(float& xpos, float& ypos) const
    {
      xpos = pos_[0];
      ypos = pos_[1];
    };
  
  inline float energy() const {return energy_;};
  
  inline void setEnergy(float en) {energy_ = en;}; 
  
  inline void advancePosition(float distance)
    {
      pos_[0] += vx_*distance;
      pos_[1] += vy_*distance;
    };
  
  inline void advanceVelocities(float Fx, float Fy, float step)
  {
    vx_ += step*Fx/energy_;
    vy_ += step*Fy/energy_;
  }
  
  inline void advanceVelocities(float Fx, float Fy, float step, float& deltaVx, float& deltaVy)
    {
      deltaVx = step*Fx/energy_;
      deltaVy = step*Fy/energy_;
      vx_ += deltaVx;
      vy_ += deltaVy;
    }

  
  inline void set_z_for_dump(int istep,float dz0, float max_z, int sign_label ) 
    {
      float z,dz;
      float sign = (float)sign_label;
      
      z = zpos_;
      dz = max_z-istep*dz0;
      //  std::cout << " set_z_for_dump istep= " << istep <<  " dz= " << dz << std::endl;
      z += dz;
      zpos_ = sign*(z + dz0);
    }
  
  inline void set_z_velocity_corrected_for_dump(int slice, int istep,int complement, float dz0, float max_z, int sign_label ) 
    {
      float dz;
      //      float vx, vy unused;
      set_z_for_dump(istep, dz0, max_z, sign_label);
      dz=(slice-istep + complement -1)*dz0;
      advancePosition(-dz);
    }
};



class ABSTRACT_BHABHA_PHOTON_SAMPLES
{
 public :

   ABSTRACT_BHABHA_PHOTON_SAMPLES() {;}
 virtual  ~ABSTRACT_BHABHA_PHOTON_SAMPLES() {;}
 virtual  int get_label() const = 0;
 virtual  unsigned int nb_samples() const = 0;
 
 virtual  void get_parameters_for_output(unsigned int number, int& number_bhabha, float& en,float& vx,float& vy, float&vz) const = 0;
 
 virtual  void add_bhabha_photon(int nbhabha, float px, float py, float pz, float en) = 0;
// virtual  void create_bhabha_photon(int nbhabha, float px, float py, float pz, float en) = 0;
 
 
};

class ABSTRACT_BHABHASAMPLES
{
  public : 
    ABSTRACT_BHABHASAMPLES() {;}
  virtual ~ABSTRACT_BHABHASAMPLES() {;}
  virtual unsigned int nb_samples() const = 0;
  
  virtual  void get_parameters_for_output(unsigned int number, unsigned int& evtIdx, float& eCM, float& mother1_en,float&e1,float&vx1,float& vy1, float&vz1, float& mother2_en, float& e2, float& vx2, float&vy2, float&vz2, int& nbphot) const = 0;
  
  virtual  void add_bhabha(unsigned int evtIdx, float px1, float py1, float pz1, float e1, float px2, float py2, float pz2, float e2, int nbphot) = 0;
  
};

class ABSTRACT_CROSS_DATA
{
  
  public :
    
    ABSTRACT_CROSS_DATA() {;}
  virtual ~ABSTRACT_CROSS_DATA() {;}
  virtual void resize(int n, int nval) = 0;
  virtual void add_data(float ener, const float* data) =0;
  
};


class ABSTRACT_LUMI_HEAP : public ABSTRACT_IO_CLASS
{

 public:
  ABSTRACT_LUMI_HEAP() {;}
  virtual ~ABSTRACT_LUMI_HEAP() {;}

  virtual  int nb_pairs() const = 0;
  virtual  void get_parameters_for_output(unsigned int number, float& e1,float& e2,float& x,float& y,float& z) const = 0;
  virtual  void get_parameters_for_output(unsigned int number, float& e1,float& e2,float& x,float& y,float& z, float& vx1,float& vy1,float& vx2,float& vy2, float& sx1, float& sy1, float& sz1, float& sx2, float& sy2, float& sz2,int& t)  const = 0;
  virtual std::string output_flow() const 
    {
      std::ostringstream out;
      out << " ABSTRACT_LUMI_HEAP:: no data for output file in abstract class " << std::endl;
      return out.str();
    }
};

class PARTICLE_INTERFACE : public ABSTRACT_PARTICLE
{
  friend class BEAM_FROM_FILE;
  protected : 
    
    
    TRIDVECTOR polar_;
  
  
  
 public:
  
  PARTICLE_INTERFACE() {;}
  virtual ~PARTICLE_INTERFACE() {;}
  
  inline bool good() const
    {
    bool test = true;
    // if the energy is 0 or negative the reading of the particle on the
    // file has probably failed. This test is made because the program
    // has to divide by energy.
    if ( energy_ < 1.9e-12 ) test = false;
    return test;
  }

 inline void init_from_input_file(float energy, float xpos, float ypos, float zpos, float vx, float vy, float polx, float poly, float polz)
   {
     set(energy, xpos, ypos,zpos,vx,vy);
     polar_.setComponents((double)polx, (double)poly, (double)polz);
   }



 inline void get_parameters(float& energy, float& xpos, float& ypos, float& zpos, float& vx, float& vy) const
 {
   xpos = pos_[0];
   ypos = pos_[1];
   zpos = zpos_;
   vx = vx_;
   vy = vy_;
   energy = energy_;
 }



 inline float get_helicity() const { return polar_(0);}
 
 virtual inline const TRIDVECTOR& getSpin() const 
   { 
     return polar_;
   }
 
};


#endif
