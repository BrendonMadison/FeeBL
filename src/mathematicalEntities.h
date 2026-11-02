#ifndef MATHEMATICALENTITIES_SEEN
#define MATHEMATICALENTITIES_SEEN

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "mathematicalTools.h"
//#include "physicalTools.h"
#include "abstractIOclass.h"
//#include "abstractParticle.h"

class TRIVECTOR
{
  float vec_[3];
  
  public :
    
    TRIVECTOR()
    {
      int k;
      for (k=0; k<3; k++) vec_[k] = 0.0;
    }
  
  TRIVECTOR( const TRIVECTOR& triv)
    {
      setComponents( triv.vec_[0],triv.vec_[1], triv.vec_[2]);
    }
  TRIVECTOR(float x,float y,float z)
    {
      vec_[0] = x;
      vec_[1] = y;
      vec_[2] = z;
    }

  inline float& operator() (int index) { return vec_[index]; }
  inline const float& operator() (int index) const { return vec_[index]; }
  
  inline TRIVECTOR& operator = (const TRIVECTOR& triv)
    {
      setComponents( triv.vec_[0],triv.vec_[1], triv.vec_[2]);
      return *this;
    }
  
  inline TRIVECTOR& operator = (float value)
    {
      setComponents( value,value, value);
      return *this;
    }
  
  inline TRIVECTOR& operator *= (const float& factor)
    {
      int k;
      for (k=0; k < 3; k++) vec_[k] *= factor;
      return *this;
    }
  
  inline float getComponent(int index ) const {return vec_[index];}
  
  inline void setComponent(int index, float value )  {vec_[index] = value;}
  
  inline void getComponents(float& x,float& y,float& z) const
    {
      x = vec_[0];
      y = vec_[1];
      z = vec_[2];
    }
  
  inline void setComponents(float x,float y,float z)
    {
      vec_[0] = x;
      vec_[1] = y;
      vec_[2] = z;
    }
  
  
  inline float norm2() const
    {
      return vec_[0]*vec_[0] + vec_[1]*vec_[1] + vec_[2]*vec_[2];
    }
  inline float norm() const
    {
      return std::sqrt(std::abs(norm2()));
    }
  
  inline void renormalize()
    {
      int k;
      float normeInv = 1.0/norm();
      for (k=0; k< 3 ; k++) vec_[k] *= normeInv;
    }
  
  inline void opposite()
    {
      int k;
      for (k=0; k< 3 ; k++) vec_[k] = -vec_[k];
    }
  
  inline void print()
    {
      std::cout << " x comp. = " << vec_[0] << " y comp. = " << vec_[1] << " z comp. = " << vec_[2] << std::endl;
    }
};

class TRIDVECTOR
{
  double vec_[3];

  public :
    
    TRIDVECTOR() 
    {
      int k;
      for (k=0; k<3; k++) vec_[k] = 0.0;
    }
  
  TRIDVECTOR( const TRIDVECTOR& triv) 
    {
      setComponents( triv.vec_[0],triv.vec_[1], triv.vec_[2]); 
    }
  TRIDVECTOR( const TRIVECTOR& triv) 
    {
      setComponents( (double)triv(0),(double)triv(1), (double)triv(2)); 
    }
  
  
  TRIDVECTOR(double x,double y,double z)
    {
      vec_[0] = x;
      vec_[1] = y;
      vec_[2] = z;
    }

  inline double& operator() (int index) { return vec_[index]; }
  inline const double& operator() (int index) const { return vec_[index]; }
  
  inline TRIDVECTOR& operator = (const TRIDVECTOR& triv) 
    {
      setComponents( triv.vec_[0],triv.vec_[1], triv.vec_[2]); 
      return *this;
    }
  inline TRIDVECTOR& operator = (const TRIVECTOR& triv) 
    {
      setComponents( (double)triv(0),(double)triv(1), (double)triv(2)); 
      return *this;
    }

  inline TRIDVECTOR& operator = (double value) 
    {
      setComponents( value,value, value); 
      return *this;
    }
  
  inline TRIDVECTOR& operator *= (const double& factor)
    {
      int k;
      for (k=0; k < 3; k++) vec_[k] *= factor;
      return *this;
    }
  
  inline double getComponent(int index ) const {return vec_[index];}
  
  inline void setComponent(int index, double value )  {vec_[index] = value;}
  
  inline void getComponents(double& x,double& y,double& z) const
   {
     x = vec_[0];
     y = vec_[1];
     z = vec_[2];
   }

  inline void setComponents(double x, double y, double z) 
    {
      vec_[0] = x;
      vec_[1] = y;
      vec_[2] = z;
    }
  
  inline void clear()
    {
      vec_[0] = 0.0;
      vec_[1] = 0.0;
      vec_[2] = 0.0;
    }

  inline double norm2() const
    {
      return vec_[0]*vec_[0] + vec_[1]*vec_[1] + vec_[2]*vec_[2];
    }
  inline double norm() const
    {
      return std::sqrt(std::abs(norm2()));
    }
  
  inline void renormalize()
    {
      int k;
      const double n = norm();
      // A zero vector has no direction. Dividing by zero here gave 0 * inf =
      // NaN in all three components, which then propagated silently -- NaN
      // fails every ordinary comparison downstream, so nothing caught it.
      // Leaving the zero vector alone is not meaningful either, but it is
      // inspectable, whereas NaN is not. Callers that can hit this must decide
      // what the degenerate direction should be; see PHYSTOOLS::referenceSpin.
      if (!(n > 0.0)) return;
      const double normeInv = 1.0/n;
      for (k=0; k< 3 ; k++) vec_[k] *= normeInv;
    }

  inline void opposite()
    {
      int k;
      for (k=0; k< 3 ; k++) vec_[k] = -vec_[k];
    }
  
  inline void print()
    {
      std::cout << " x comp. = " << vec_[0] << " y comp. = " << vec_[1] << " z comp. = " << vec_[2] << std::endl;
    }
};


class named_int_vector : public ABSTRACT_IO_CLASS
{
  std::vector<int> vector_;
  std::string name_;

 public :

  named_int_vector() {;}

  inline void put_name(std::string name) {name_ = name;}
  inline void clear()
  {
    name_ = std::string (" ");
    vector_.clear();
  }
  inline void  add_component(int comp) 
  {
    vector_.push_back(comp);
  }
  ~named_int_vector() {;}

  virtual std::string output_flow() const 
  {
    std::ostringstream out;
    std::string start = std::string("vector").append(name_);
    out << title(start);
    out << " vector " << name_ << std::endl;
    int k;
    for (k=0; k < (int)vector_.size(); k++)
      {
	out << name_ << " k = " << k << " :" << vector_[k] << std::endl;
      }
    return out.str();  
  }
  
  
};


#endif
