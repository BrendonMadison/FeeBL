#ifndef EXTRAPHOTON_SEEN
#define EXTRAPHOTON_SEEN

class  EXTRA
{
  float energy_,vx_,vy_,q2_, eorg_;
  //  float weight_;

 public:

  EXTRA() {;}

  EXTRA(float energy,float vx,float vy,float q2,float eorg) 
    {
      energy_ = energy;
      vx_     = vx;
      vy_      = vy;
      q2_      = q2;
      eorg_    = eorg;
      //         weight_  = weight;
    }

  ~EXTRA() {;}
  inline void get_parameters(float& energy,float& vx,float& vy,float& q2,float& eorg) const
  {
      energy = energy_;
      vx     = vx_;
      vy      = vy_;
      q2      = q2_;
      eorg    = eorg_;
      //         weight  = weight_;
  }
    inline float energy() const {return energy_;}
    //       inline float weight() const {return weight_;}
    inline float q2() const {return q2_;}
    inline float eorg() const { return eorg_;}
    inline  void velocities(float& vx, float& vy) const
    {
      vx = vx_;
      vy = vy_;
    }
};

#endif
