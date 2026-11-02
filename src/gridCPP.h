#ifndef GRID_SEEN
#define GRID_SEEN
#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <limits>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>
#include "typeDefs.h"
#include "meshCPP.h"
#include "beamCPP.h"
#include "extraPhotonCPP.h"
#include "switchesCPP.h"
#include "resultsCPP.h"
#include "designAxisSpectrumCPP.h"
#include "pairsCPP.h"
#include "lumiCPP.h"
#include "jetParameterCPP.h"
#include "backgroundCPP.h"
#include "rndmCPP.h"
#include "bhabhaSamplesCPP.h"
#include "abstractParticle.h"
#include "fourierCPP.h"
#include "fieldCPP.h"
#include "gpuCic.h"
#include "gpuResident.h"
#include "electronLumiSpatialCPP.h"
#include "parametersCPP.h"
#include "mathematicalEntities.h"

namespace feebl {
namespace sparse_cells {

// Generation-stamped index of cells touched during one encounter.  The cell
// payload remains in the historical vector-of-vectors, so pointer insertion
// order and every physics object are unchanged.  Only clearing and traversal
// learn which cells can contain work.
class ActiveCellSet
{
 public:
  ActiveCellSet()
      : generation_(1U), sorted_(true), enabled_(false), epochs_(0U),
        dense_clear_visits_(0U), actual_clear_visits_(0U) {}

  explicit ActiveCellSet(std::size_t cell_count)
      : generation_(1U), sorted_(true), enabled_(false), epochs_(0U),
        dense_clear_visits_(0U), actual_clear_visits_(0U)
  {
    reset(cell_count);
  }

  void reset(std::size_t cell_count)
  {
    cells_.clear();
    cells_.reserve(cell_count);
    stamps_.assign(cell_count, 0U);
    generation_ = 1U;
    sorted_ = true;
    epochs_ = 0U;
    dense_clear_visits_ = 0U;
    actual_clear_visits_ = 0U;
  }

  void set_enabled(bool enabled)
  {
    if (enabled_ == enabled) return;
    enabled_ = enabled;
    clear_index();
  }

  void clear_index()
  {
    cells_.clear();
    std::fill(stamps_.begin(), stamps_.end(), 0U);
    generation_ = 1U;
    sorted_ = true;
  }

  bool enabled() const { return enabled_; }

  void touch(int cell)
  {
    if (!enabled_) return;
    assert(cell >= 0);
    const std::size_t index = static_cast<std::size_t>(cell);
    assert(index < stamps_.size());
    if (stamps_[index] == generation_) return;
    stamps_[index] = generation_;
    cells_.push_back(cell);
    sorted_ = cells_.size() < 2U ||
              (sorted_ && cells_[cells_.size()-2U] < cell);
  }

  void sort_cells()
  {
    if (!sorted_)
      {
        std::sort(cells_.begin(), cells_.end());
        sorted_ = true;
      }
  }

  const std::vector<int>& cells() const
  {
    // Physics loops rely on the historical ascending linear-cell order.
    assert(sorted_);
    return cells_;
  }

  template <typename Value>
  void clear_touched(std::vector<std::vector<Value> >& bins)
  {
    assert(bins.size() == stamps_.size());
    dense_clear_visits_ += static_cast<unsigned long long>(bins.size());
    if (enabled_)
      {
        actual_clear_visits_ +=
            static_cast<unsigned long long>(cells_.size());
        for (std::vector<int>::const_iterator cell = cells_.begin();
             cell != cells_.end(); ++cell)
          bins[static_cast<std::size_t>(*cell)].clear();
      }
    else
      {
        actual_clear_visits_ += static_cast<unsigned long long>(bins.size());
        for (std::size_t cell = 0; cell < bins.size(); ++cell)
          bins[cell].clear();
      }
    begin_epoch();
  }

  template <typename Value>
  void clear_values(Value* values)
  {
    assert(values != NULL || cells_.empty());
    dense_clear_visits_ += static_cast<unsigned long long>(stamps_.size());
    if (enabled_)
      {
        actual_clear_visits_ +=
            static_cast<unsigned long long>(cells_.size());
        for (std::vector<int>::const_iterator cell = cells_.begin();
             cell != cells_.end(); ++cell)
          values[*cell] = static_cast<Value>(0.0);
      }
    else
      {
        actual_clear_visits_ +=
            static_cast<unsigned long long>(stamps_.size());
        if (!stamps_.empty())
          std::fill(values, values+stamps_.size(), static_cast<Value>(0.0));
      }
    begin_epoch();
  }

  void begin_epoch()
  {
    ++epochs_;
    cells_.clear();
    sorted_ = true;
    ++generation_;
    if (generation_ == 0U)
      {
        std::fill(stamps_.begin(), stamps_.end(), 0U);
        generation_ = 1U;
      }
  }

  unsigned long long epochs() const { return epochs_; }
  unsigned long long dense_clear_visits() const
    { return dense_clear_visits_; }
  unsigned long long actual_clear_visits() const
    { return actual_clear_visits_; }
  unsigned long long avoided_clear_visits() const
    { return dense_clear_visits_ - actual_clear_visits_; }
  void reset_counters()
  {
    epochs_ = 0U;
    dense_clear_visits_ = 0U;
    actual_clear_visits_ = 0U;
  }

 private:
  std::vector<int> cells_;
  std::vector<unsigned int> stamps_;
  unsigned int generation_;
  bool sorted_;
  bool enabled_;
  unsigned long long epochs_;
  unsigned long long dense_clear_visits_;
  unsigned long long actual_clear_visits_;
};

inline void intersect_sorted(const std::vector<int>& first,
                             const std::vector<int>& second,
                             std::vector<int>* result)
{
  assert(result != NULL);
  result->clear();
  result->reserve(std::min(first.size(), second.size()));
  std::vector<int>::const_iterator a = first.begin();
  std::vector<int>::const_iterator b = second.begin();
  while (a != first.end() && b != second.end())
    {
      if (*a < *b) ++a;
      else if (*b < *a) ++b;
      else
        {
          result->push_back(*a);
          ++a;
          ++b;
        }
    }
}

inline void sorted_union(const std::vector<int>* const* sets,
                         std::size_t set_count,
                         std::vector<int>* result)
{
  assert(result != NULL);
  result->clear();
  std::size_t required = 0U;
  for (std::size_t set = 0; set < set_count; ++set)
    required += sets[set]->size();
  result->reserve(required);
  for (std::size_t set = 0; set < set_count; ++set)
    result->insert(result->end(), sets[set]->begin(), sets[set]->end());
  std::sort(result->begin(), result->end());
  result->erase(std::unique(result->begin(), result->end()), result->end());
}

}  // namespace sparse_cells

namespace legacy_photon_lumi {
struct PhotonSample;
struct ChargedSample;
class Workspace;
}  // namespace legacy_photon_lumi
namespace legacy_electron_lumi_blocked {
struct Workspace;
}  // namespace legacy_electron_lumi_blocked
}  // namespace feebl

//! virtual class 
/*! 
 attributes of GENERAL_GRID are arrays for field computations.


 attributes of derived class GRID are arrays giving for each cell pointers to particles and photons which are affected to this cell. (as a result of 'distribute')

 inheriting class EXTRA_GRID has the same attributes as GENERAL_GRID
*/ 

class SLICE_ON_GRID
{

  float nb_part_per_macro_;
  float scal_step_;
  int nb_macroparticles_;
  BEAM* beam_;
  PHI_FLOAT *rho_;
  int dim_rho_;
  int nb_cells_y_;
  feebl::sparse_cells::ActiveCellSet active_rho_cells_;

  // not implemented
  SLICE_ON_GRID(SLICE_ON_GRID& s);

  inline void set_rho(int dim)
    {
      if (rho_ != NULL) delete [] rho_;
      if (dim > 0) 
	{
	  rho_ = new PHI_FLOAT[dim];
	  std::fill(rho_, rho_+dim, static_cast<PHI_FLOAT>(0.0));
	}
      else
	{
	  rho_ = NULL;
	}
      active_rho_cells_.reset(static_cast<std::size_t>(dim > 0 ? dim : 0));
    }

  inline  void set_size(int nx, int ny)
    {
      nb_cells_y_ = ny;
      dim_rho_ = nx*ny;
      set_rho(dim_rho_);
    }
  
  
  inline void setZero()
    {
      nb_part_per_macro_ = 0.0;
      scal_step_ = 1.0;
      nb_macroparticles_ = 0;
      beam_ = NULL;
      rho_ = NULL;
      dim_rho_ = 0;
      nb_cells_y_ = 0;
    }
  
  public :
    
    SLICE_ON_GRID() 
    {
      setZero();
    }
  
  SLICE_ON_GRID(int nx, int ny)
    {
      setZero();
      set_size(nx,ny);
    } 
  
  SLICE_ON_GRID(const SLICE_ON_GRID& s) 
    {
      setZero();
      beam_ = s.beam_;
      nb_part_per_macro_ =  s.nb_part_per_macro_;
      scal_step_ = s.scal_step_ ;
      nb_macroparticles_ = s.nb_macroparticles_;
      nb_cells_y_ = s.nb_cells_y_;
      dim_rho_ = s.dim_rho_;
      set_rho( dim_rho_);
      /*       std::cout << " SLICE_ON_GRID : copy constructor " << std::endl; */
      /*       std::cout << " old address " << s.rho_ << " new address " << rho_ << std::endl; */
      /*       std::cout << " nb_macroparticles_ " <<  nb_macroparticles_ << std::endl; */
      /*       std::cout << " nb_part_per_macro_ " << nb_part_per_macro_ << " scal_step_ " << scal_step_ << " nb_cells_y_ " << nb_cells_y_ << " dim_rho_ " << dim_rho_ << std::endl; */
      /*       std::cout << " *********************************************** " << std::endl; */
    }
  
  ~SLICE_ON_GRID()
    {
      delete [] rho_;
    }
  
  inline SLICE_ON_GRID& operator = (const SLICE_ON_GRID& s)
    {
      if (this == &s) return *this; // protect against self-assignment
      int k;
      beam_ = s.beam_;
      nb_part_per_macro_ =  s.nb_part_per_macro_;
      scal_step_ = s.scal_step_ ;
      nb_macroparticles_ = s.nb_macroparticles_;
      nb_cells_y_ = s.nb_cells_y_;
      dim_rho_ = s.dim_rho_;
      // delete old memory:
      delete[] rho_;
      rho_ = NULL;
      // assign new memory:
      set_rho( dim_rho_);
      for (k=0; k < dim_rho_ ; k++) rho_[k] = s.rho_[k];
      rebuildActiveRhoCells();
      return *this;
    }
  
  inline void connect_beam(BEAM* b) { beam_ = b;}
  
  
  inline void resize(int nx, int ny)
  {
    nb_cells_y_ = ny;
    dim_rho_ = nx*ny;
    set_rho( dim_rho_);
  }

  inline void set_macroparticles(int mac)
    {
      nb_macroparticles_ = mac;
    }
  
  inline int get_macroparticles() const 
    {
      return nb_macroparticles_;
    }
  
  inline const BEAM* get_beam() const 
    {
      return beam_;
    }
  
  inline BEAM* get_beam() 
    {
      return beam_;
    }
  
  inline void set_scal_step(float s)
    {
      scal_step_ = s;
    }
  
  inline float get_scal_step() const 
    {
      return scal_step_;
    }
  
  inline void update_nb_part_per_macro(float nb_macro_part)
    {
      if (nb_macroparticles_ > 0)
	{
	  nb_part_per_macro_ = nb_macro_part/(float)nb_macroparticles_;
	}
      else nb_part_per_macro_ = 0.0;
    }
  
  inline float get_nb_part_per_macro() const 
    {
      return nb_part_per_macro_;
    }
  
  inline const PHI_FLOAT* get_rho() const { return rho_;}
  inline PHI_FLOAT* get_rho_mutable() { return rho_;}
  inline int get_dim_rho() const { return dim_rho_;}
  inline int get_nb_cells_y() const { return nb_cells_y_;}
  inline const std::vector<int>& activeRhoCells() const
    { return active_rho_cells_.cells(); }
  inline void finalizeActiveRhoCells() { active_rho_cells_.sort_cells(); }
  inline void enableSparseCells(bool enabled)
    {
      active_rho_cells_.set_enabled(enabled);
      active_rho_cells_.reset_counters();
      if (enabled) rebuildActiveRhoCells();
    }
  inline unsigned long long denseRhoClearVisits() const
    { return active_rho_cells_.dense_clear_visits(); }
  inline unsigned long long actualRhoClearVisits() const
    { return active_rho_cells_.actual_clear_visits(); }

  inline void rebuildActiveRhoCells()
    {
      if (!active_rho_cells_.enabled()) return;
      active_rho_cells_.clear_index();
      for (int cell=0; cell<dim_rho_; ++cell)
        if (rho_[cell] != static_cast<PHI_FLOAT>(0.0))
          active_rho_cells_.touch(cell);
      active_rho_cells_.sort_cells();
    }
  
  inline void assignChargeToNGP(int i1,int i2, float chgSgn) 
    {  
      const int cell=i1*nb_cells_y_+i2;
      active_rho_cells_.touch(cell);
      rho_[cell] += nb_part_per_macro_*chgSgn;
    }
  
  inline void assignChargeToCIC(int i1,int i2, float h_x, float h_y,  float chgSgn) 
    { 
      int j;
      PHI_FLOAT charge = nb_part_per_macro_*chgSgn;
      j=i1*nb_cells_y_+i2;
      active_rho_cells_.touch(j);
      rho_[j] += (1.0-h_x)*(1.0-h_y)*charge;
      j=(i1+1)*nb_cells_y_+i2;
      active_rho_cells_.touch(j);
      rho_[j] += h_x*(1.0-h_y)*charge;
      j=i1*nb_cells_y_+i2+1;
      active_rho_cells_.touch(j);
      rho_[j] += (1.0-h_x)*h_y*charge;
      j=(i1+1)*nb_cells_y_+i2+1;
      active_rho_cells_.touch(j);
      rho_[j] += h_x*h_y*charge;
    };

  /** Deposit into caller-owned scratch with exactly the serial CIC formula. */
  inline void assignChargeToCICBuffer(PHI_FLOAT* target, int i1, int i2,
                                      float h_x, float h_y, float chgSgn) const
    {
      const PHI_FLOAT charge = nb_part_per_macro_*chgSgn;
      int j=i1*nb_cells_y_+i2;
      target[j] += (1.0-h_x)*(1.0-h_y)*charge;
      j=(i1+1)*nb_cells_y_+i2;
      target[j] += h_x*(1.0-h_y)*charge;
      j=i1*nb_cells_y_+i2+1;
      target[j] += (1.0-h_x)*h_y*charge;
      j=(i1+1)*nb_cells_y_+i2+1;
      target[j] += h_x*h_y*charge;
    }
  
  inline void razRho()
    {
      active_rho_cells_.clear_values(rho_);
    }
  
  inline void symmetrizeCharges(int n_cell_x, int n_cell_y)
    {
      PHI_FLOAT tmp;
      int i1, i2;
      int j1, j2, j3, j4;
      for (i1=0;i1<n_cell_x/2;i1++){
	for (i2=0;i2<n_cell_y/2;i2++)
	  {
	    j1=i1*n_cell_y + i2;
	    j2=(n_cell_x - 1-i1)*n_cell_y +i2;
	    j3=i1*n_cell_y +(n_cell_y -1-i2);
	    j4=(n_cell_x-1-i1)*n_cell_y +(n_cell_y -1-i2);
	    active_rho_cells_.touch(j1);
	    active_rho_cells_.touch(j2);
	    active_rho_cells_.touch(j3);
	    active_rho_cells_.touch(j4);
	    tmp=0.25*(rho_[j1]+rho_[j2]+rho_[j3]+rho_[j4]);
	    rho_[j1]=tmp;
	    rho_[j2]=tmp;
	    rho_[j3]=tmp;
	    rho_[j4]=tmp;
	    
	    /* 	      tmp=0.25*(rho2_[j1]+rho2_[j2]+rho2_[j3]+rho2_[j4]); */
	    /* 	      rho2_[j1]=tmp; */
	    /* 	      rho2_[j2]=tmp; */
	    /* 	      rho2_[j3]=tmp; */
	    /* 	      rho2_[j4]=tmp; */
	  }
      }
    };
};

class GENERAL_GRID
{
  
 protected: 
  
  /*! number of cells in three coordinates */
  int n_cell_x_,n_cell_y_,n_cell_z_; 
  int timestep_;
  MESH mesh_;
  float min_x_,max_x_,min_y_,max_y_,min_z_,max_z_;
  float cut_x_,cut_y_,cut_z_;
  //  float nb_part_per_macro_[2];
  float step_;
  //    float scal_step[2];
  float rho_x_1_,rho_y_1_,rho_sum_1_,rho_x_2_,rho_y_2_,rho_sum_2_;

  float delta_x_inverse_,delta_y_inverse_;
  
  PHI_FLOAT rho_factor_;
  // Exact post-normalization factor supplied to FIELD::dist_init.  Adaptive
  // field views must reuse this stored value; re-deriving it from cuts and
  // spacings can change the last bits and defeat crop/global equivalence.
  PHI_FLOAT field_kernel_factor_;
  
  // int nb_macroparticles_[2];

  int integration_method_;

  //  PHI_FLOAT *dist_;

  FIELD champ_;

  SLICE_ON_GRID slice_of_beam_[2];

  //  BEAM* beam1_;
  // PHI_FLOAT *rho1_;

  //  BEAM* beam2_;
  //  PHI_FLOAT *rho2_;

  GENERAL_GRID();

  virtual void distribute_coherent_particles(int i_slice1,
					  int i_slice2,
				    int electron_distribution_rho) = 0;

  public :

  virtual ~GENERAL_GRID();

  // Set by FEEBL so the native output header can state the crossing angle the
  // run was configured with; the grid itself has no notion of one.
  double native_half_crossing_angle_;
  feebl::design_axis::Spectrum design_axis_spectrum_;
  inline void set_native_half_crossing_angle(double a)
    { native_half_crossing_angle_ = a; }

  inline void connect_beams(BEAM* b1, BEAM* b2)
    {
      slice_of_beam_[0].connect_beam(b1);
      slice_of_beam_[1].connect_beam(b2);
      /*   beam1_ = b1; */
      /*   beam2_ = b2; */
    } 
  
  inline int get_n_cell_x() const {return n_cell_x_;};
  inline int get_n_cell_y() const {return n_cell_y_;};
  inline int get_n_cell_z() const {return n_cell_z_;};
  
  // no_beam = 1,2
  inline int get_nb_macroparticles(int nobeam) const 
    {
      return  slice_of_beam_[nobeam-1].get_macroparticles();
      // return nb_macroparticles_[nobeam-1];
    }
  
  inline int get_timestep() const {return timestep_;};
  // FEEBL: optimize_grids sets these after GRID::read.
  inline void set_n_cell_x(int n) {n_cell_x_ = n;};
  inline void set_n_cell_y(int n) {n_cell_y_ = n;};
  inline void set_n_cell_z(int n) {n_cell_z_ = n;};
  inline void set_timestep(int n) {timestep_ = n;};
  
  inline float get_min_x() const {return min_x_;};
  inline float get_max_x() const {return max_x_;};
  inline float get_min_y() const {return min_y_;};
  inline float get_max_y() const {return max_y_;};
  inline float get_min_z() const {return min_z_;};
  inline float get_max_z() const {return max_z_;};
  inline float get_delta_x() const {return mesh_.get_delta_x();};
  inline float get_delta_y() const {return mesh_.get_delta_y();};
  inline float get_delta_z() const {return mesh_.get_delta_z();};
  inline float get_scal_step(int index) const
    {
      return slice_of_beam_[index].get_scal_step();
      //return scal_step[index];
    }
  //  inline float get_cut_z() const {return cut_z_;};
  inline float get_step() const {return step_;};
  inline float get_offset_x() const {return mesh_.get_offset_x();};
  inline float get_offset_y() const {return mesh_.get_offset_y();};
  
  inline const MESH* get_mesh() const {return &mesh_;}
  
  inline float get_rho_x_1() const {return rho_x_1_;};
  inline float get_rho_y_1() const {return rho_y_1_;};
  inline float get_rho_sum_1() const {return rho_sum_1_;};
  inline float get_rho_sum_2() const {return rho_sum_2_;};
  inline float get_rho_x_2() const {return rho_x_2_;};
  inline float get_rho_y_2() const {return rho_y_2_;};
  
  
  inline PHI_FLOAT get_rho_factor() const {return rho_factor_;};
  inline PHI_FLOAT get_field_kernel_factor() const
    { return field_kernel_factor_; }
  
  inline const PHI_FLOAT* get_phi1() const {return champ_.get_phi(1);};
  inline const PHI_FLOAT* get_phi2() const {return champ_.get_phi(2);};

  // Narrow field-solver integration surface.  These accessors expose the
  // existing arrays without transferring ownership or changing GENERAL_GRID's
  // implicit copy behavior (used by EXTRA_GRID).
  inline const PHI_FLOAT* get_rho1() const
    { return slice_of_beam_[0].get_rho(); }
  inline const PHI_FLOAT* get_rho2() const
    { return slice_of_beam_[1].get_rho(); }
  inline PHI_FLOAT* get_rho1_mutable()
    { return slice_of_beam_[0].get_rho_mutable(); }
  inline PHI_FLOAT* get_rho2_mutable()
    { return slice_of_beam_[1].get_rho_mutable(); }
  inline PHI_FLOAT* get_phi1_mutable()
    { return champ_.get_phi_mutable(1); }
  inline PHI_FLOAT* get_phi2_mutable()
    { return champ_.get_phi_mutable(2); }
  inline const PHI_FLOAT* get_fft_spectrum() const
    { return champ_.get_fft_spectrum(); }
  inline const PHI_FLOAT* get_real_space_field_kernel() const
    { return champ_.get_real_space_kernel(); }
  inline void enable_real_space_field_kernel(bool enabled)
    { champ_.enable_real_space_kernel(enabled); }
  inline int get_fft_spectrum_size() const
    { return champ_.get_fft_spectrum_size(); }
  inline float get_delta_x_inverse() const { return delta_x_inverse_; }
  inline float get_delta_y_inverse() const { return delta_y_inverse_; }
  inline double get_macro_charge(int beam) const
    {
      if (beam == 1) return slice_of_beam_[0].get_nb_part_per_macro();
      if (beam == 2) return slice_of_beam_[1].get_nb_part_per_macro();
      std::cerr << " GENERAL_GRID::get_macro_charge(): ERROR beam=" << beam
                << std::endl;
      exit(1);
    }

  //  void get_cuts(float& x,float& y,float&z) const {x = cut_x_; y = cut_y_; z = cut_z_;};

  // nobeam = 1,2
  inline void set_nb_macroparticles(int nobeam, int npart)
    { 
      slice_of_beam_[nobeam-1].set_macroparticles(npart);
      //nb_macroparticles_[nobeam-1] = npart;
    }
  
  /* inline void update_nb_part_per_macro(int nobeam, float n_particles) */
  /* { */
  /*   int index = nobeam -1; */
  /*   if (nb_macroparticles_[index] > 0) */
  /*     { */
  /*       nb_part_per_macro_[index] = n_particles/(float)nb_macroparticles_[index]; */
  /*     } */
  /* } */
  /* inline void razRhos() */
  /* { */
  /*   int i1, i2, j; */
  /*   for (i1=0;i1<n_cell_x_;i1++) */
  /*     { */
  /*       for (i2=0;i2<n_cell_y_;i2++) */
  /* 	{ */
  /* 	  j=i1*n_cell_y_+i2; */
  /* 	  rho1_[j]=0.0; */
  /* 	  rho2_[j]=0.0; */
  /* 	} */
  /*     } */
  /* }; */
  
  inline void absoluteCoordinates(float x, float y, PHI_FLOAT& h_x, PHI_FLOAT& h_y) const
    {
      h_x=x*delta_x_inverse_ + mesh_.get_offset_x();
      h_y=y*delta_y_inverse_ + mesh_.get_offset_y();
    }
  
  
  // given coordinates x,y : return the cell coordinates i1,i2 of the cell
  // containing (x,y) ; ai is the biggest integer that is lower than x
  // bi is the biggest integer lower than y-1/2; weight = y -1/2 -i2
  inline void cellLocalization(float x, float y, int& ai, int& bi, float& weight) const
    {
      ai=(int)floor(x);
      weight = y-0.5;
      bi=(int)floor(weight);
      weight -= bi;
    };
  
  inline void cellLocalization(PHI_FLOAT x, PHI_FLOAT y, int& ai, int& bi, PHI_FLOAT& weight) const
    {
      ai=(int)floor(x);
      weight = y-0.5;
      bi=(int)floor(weight);
      weight -= bi;
    };
  
  inline int slice_from_z(float z)
    {
      return (int)floor(z/get_delta_z()+0.5*(float)n_cell_z_);
    }
  
  inline void phiValuesX(int i1, int i2,register float h, const PHI_FLOAT * phi, PHI_FLOAT& phi1_x, PHI_FLOAT& phi2_x, PHI_FLOAT& phi3_x) const
    {
      int j;
      register float h_p;
      // 
      //
      h_p=1.0-h;
      j=i1*n_cell_y_+i2;
      phi1_x=h*phi[j+n_cell_y_+1]+h_p*phi[j+1];
      phi2_x=h*phi[j+n_cell_y_] + h_p*phi[j];
      phi3_x=h*phi[j+n_cell_y_-1] + h_p*phi[j-1];
    }

  inline void phiValuesY(int i1, int i2,register float h, const PHI_FLOAT * phi, PHI_FLOAT& phi1_y, PHI_FLOAT& phi2_y, PHI_FLOAT& phi3_y) const
    {
      int j;
      register float h_p;
      
      h_p=1.0-h;
      j=i1*n_cell_y_+i2;
      phi1_y=h*phi[j+n_cell_y_+1] + h_p*phi[j+n_cell_y_];
      phi2_y=h*phi[j+1] + h_p*phi[j];
      phi3_y=h*phi[j-n_cell_y_+1] + h_p*phi[j-n_cell_y_];
    }
  
  inline bool particleInGrid(float xpart, float ypart, float& h_x,float& h_y, int& i1, int& i2)
    { 
      h_x=(xpart*delta_x_inverse_ + mesh_.get_offset_x()-0.5);
      i1=(int)floor(h_x);
      h_x -= (float)i1;
      h_y=(ypart*delta_y_inverse_ +  mesh_.get_offset_y()-0.5);
      i2=(int)floor(h_y);
      h_y -= (float)i2;
      if ((i1>=0)&&(i1<n_cell_x_-1)&&(i2>=0)&&(i2<n_cell_y_-1)) return true;
      else return false;
    }
  
  inline bool particleInGrid(float xpart, float ypart, int& i1, int& i2)
    { 
      i1=(int)floor(xpart*delta_x_inverse_+ mesh_.get_offset_x());
      i2=(int)floor(ypart*delta_y_inverse_+ mesh_.get_offset_y());
      if ((i1>=0)&&(i1<n_cell_x_-1)&&(i2>=0)&&(i2<n_cell_y_-1)) return true;
      else return false;
    }
  
  inline void assignChargeToNGP(PHI_FLOAT* rho, int i1,int i2, PHI_FLOAT charge) {  rho[i1*n_cell_y_+i2] += charge;};
  
  static inline void assignChargeToCICWithStride(
      PHI_FLOAT* rho, int stride_y, int i1, int i2, float h_x,
      float h_y, PHI_FLOAT charge)
    {
      int j;
      j=i1*stride_y+i2;
      rho[j] += (1.0-h_x)*(1.0-h_y)*charge;
      j=(i1+1)*stride_y+i2;
      rho[j] += h_x*(1.0-h_y)*charge;
      j=i1*stride_y+i2+1;
      rho[j] += (1.0-h_x)*h_y*charge;
      j=(i1+1)*stride_y+i2+1;
      rho[j] += h_x*h_y*charge;
    };

  inline void assignChargeToCIC(PHI_FLOAT* rho, int i1,int i2, float h_x, float h_y, PHI_FLOAT charge)
    {
      assignChargeToCICWithStride(
          rho, n_cell_y_, i1, i2, h_x, h_y, charge);
    };

  /** Deposit an already-admitted canonical CIC cloud into an aligned view. */
  inline void assignChargeToCICTranslated(
      PHI_FLOAT* rho, int local_n_x, int local_n_y,
      int origin_x, int origin_y, int canonical_i1, int canonical_i2,
      float h_x, float h_y, PHI_FLOAT charge) const
    {
      const int local_i1 = canonical_i1 - origin_x;
      const int local_i2 = canonical_i2 - origin_y;
      if (rho == NULL || local_n_x < 2 || local_n_y < 2 ||
          local_i1 < 0 || local_i1 >= local_n_x - 1 ||
          local_i2 < 0 || local_i2 >= local_n_y - 1)
        throw std::out_of_range(
            "translated CIC source cloud is outside its canonical window");
      // Keep the established operand order in one shared implementation.
      assignChargeToCICWithStride(
          rho, local_n_y, local_i1, local_i2, h_x, h_y, charge);
    };
  
  /* inline void symmetrizeCharges() */
  /* { */
  /*   PHI_FLOAT tmp; */
  /*   int i1, i2; */
  /*   int j1, j2, j3, j4; */
  /*       for (i1=0;i1<n_cell_x_/2;i1++){ */
  /* 	  for (i2=0;i2<n_cell_y_/2;i2++) */
  /* 	    { */
  /* 	      j1=i1*n_cell_y_+i2; */
  /* 	      j2=(n_cell_x_-1-i1)*n_cell_y_+i2; */
  /* 	      j3=i1*n_cell_y_+(n_cell_y_-1-i2); */
  /* 	      j4=(n_cell_x_-1-i1)*n_cell_y_+(n_cell_y_-1-i2); */
  /* 	      tmp=0.25*(rho1_[j1]+rho1_[j2]+rho1_[j3]+rho1_[j4]); */
  /* 	      rho1_[j1]=tmp; */
  /* 	      rho1_[j2]=tmp; */
  /* 	      rho1_[j3]=tmp; */
  /* 	      rho1_[j4]=tmp; */
  /* 	      tmp=0.25*(rho2_[j1]+rho2_[j2]+rho2_[j3]+rho2_[j4]); */
  /* 	      rho2_[j1]=tmp; */
  /* 	      rho2_[j2]=tmp; */
  /* 	      rho2_[j3]=tmp; */
  /* 	      rho2_[j4]=tmp; */
  /* 	  } */
  /*       } */
  /* }; */
  
  inline bool coordinatesInGridRange(float xpart, float ypart) const
    {
      // The historical physical-only predicate admitted a one-ULP upper
      // boundary layer where float localization could round to nx-1/ny-1 and
      // ElectricFieldCIC's +1 stencil then read beyond phi.  Preserve strict
      // physical membership while requiring the localized stencil to be
      // valid.  The compact resident-GPU gather uses this same intended
      // domain contract.
      return feebl::gpu::resident_field_point_in_intended_domain(
          xpart, ypart, min_x_, max_x_, min_y_, max_y_,
          delta_x_inverse_, delta_y_inverse_, mesh_.get_offset_x(),
          mesh_.get_offset_y(), n_cell_x_, n_cell_y_);
    };
  
  void interpolePotential(float xpart,float ypart, PHI_FLOAT& h_x, PHI_FLOAT& h_y, PHI_FLOAT& phi1_x, PHI_FLOAT& phi2_x, PHI_FLOAT& phi3_x, PHI_FLOAT& phi1_y,PHI_FLOAT& phi2_y, PHI_FLOAT& phi3_y, const PHI_FLOAT *phi) const;
  
  TRIVECTOR ElectricFieldCIC(float xpart,float ypart,const PHI_FLOAT *phi) const;

  /**
   * Evaluate the historical twelve-node CIC field stencil from an
   * integer-aligned canonical sub-window.
   *
   * Canonical floating mapping/localization is performed first with the same
   * operation order as ElectricFieldCIC; only the decided integer node
   * indices are translated by (origin_x, origin_y).  This is the field-only
   * seam used by adaptive_grid_mode=1 and deliberately does not define a new
   * physical grid.
   */
  TRIVECTOR ElectricFieldCICTranslated(
      float xpart, float ypart, const PHI_FLOAT* local_phi,
      int local_n_x, int local_n_y, int origin_x, int origin_y) const;

  /**
   * Evaluate a translated field from immutable canonical localization
   * decisions.  This is the shared arithmetic core for planned CPU/GPU
   * field results; no affine coordinate map or floor is repeated here.
   */
  TRIVECTOR ElectricFieldCICTranslatedCanonical(
      const PHI_FLOAT* local_phi, int local_n_x, int local_n_y,
      int origin_x, int origin_y,
      int y_base_x, int y_base_y, float y_weight,
      int x_base_x, int x_base_y, float x_weight,
      PHI_FLOAT fractional_x, PHI_FLOAT fractional_y) const;
  
  void computeFields(int integrationMethod, float charge_sign, PHI_FLOAT *sor_parameter); 
  
  virtual void distribute_particles(int i_slice1,
				    int i_slice2,
				    int electron_distribution_rho, 
				    int force_symmetric) = 0;
  
  
  void  init_grid_phys (float n_particles1, float n_particles2,float cut_x,float cut_y,float cut_z, float charge_sign,   FFT_SERVER* fourier);
  
};



class GRID : public GENERAL_GRID , public ABSTRACT_IO_CLASS
{

  ///////////// private struct ///////////////////////////////////////  
  struct DISTRIBUTE
  {
    long int in, out;
    float delta;
    long int tot;
    DISTRIBUTE():in(0),out(0),delta(0.0),tot(0){}
  };

  struct ELECTRON_LUMI_CACHED_PARTICLE
  {
    float signed_energy;
    float absolute_energy;
    float vx;
    float vy;
    float vz;
    float weight;
    double massive_momentum;
    int charge_class;
  };

  struct CPU_PRIMARY_MOVE_CONTEXT;
  /////////////////////////////////////////////////////////////////////
  
  ///////////// private class ///////////////////////////////////////
  class PARTICLE_POINTER 
  {
    
  protected :
    float weight_;
    
  public :

    PARTICLE_POINTER() {;}
    PARTICLE_POINTER(float weight) {weight_ = weight;}


    virtual  ~PARTICLE_POINTER() {;}
    inline float weight() const {return weight_;}; 
    virtual   void velocities(float& vx, float& vy) const = 0;
    virtual   const TRIDVECTOR& getSpin() const = 0;
    virtual    float energy() const = 0;
    //   virtual  float q2() const = 0;
    //   virtual  float eorg() const = 0;
  };

  
  class BEAM_PARTICLE_POINTER : public PARTICLE_POINTER
  {
  
  private:
    
    const ABSTRACT_PARTICLE* partic_;
    std::uint64_t stable_id_;
  
  public:
  
    BEAM_PARTICLE_POINTER()
        : partic_(NULL), stable_id_(std::numeric_limits<std::uint64_t>::max())
      {;}
  
    BEAM_PARTICLE_POINTER(
        const ABSTRACT_PARTICLE* partic, float weight,
        std::uint64_t stable_id = std::numeric_limits<std::uint64_t>::max())
        : PARTICLE_POINTER(weight), partic_(partic), stable_id_(stable_id)
    {
      //	weight_ = weight;
    }
    
    ~BEAM_PARTICLE_POINTER() {;}
  
    virtual  inline  void velocities(float& vx, float& vy) const {partic_->velocities(vx, vy);}
  
    virtual inline const TRIDVECTOR& getSpin() const 
    { 
      return  partic_->getSpin();
    }
 
    virtual   inline   float energy() const { return partic_->energy();}
    inline std::uint64_t stable_id() const { return stable_id_; }
    inline const ABSTRACT_PARTICLE& diagnostic_particle() const { return *partic_; }
    
    //  virtual inline float q2() const { return 0.0;};
    //   virtual inline float eorg() const { return 0.0;};
    
    
    /*  inline float get_spin(int nc) const   */
    /*  {  */
    /*    //  std::cout << " BEAM_PARTICLE_POINTER:: get_spin() " << std::endl; */
    /*    return partic_->get_spin()(nc-1); */
    /*  } */
    
  };
  //////////////////
  
  class PHOTON_POINTER : public PARTICLE_POINTER
  {
    
  protected :
    
  public : 
    
    PHOTON_POINTER() {;}
    PHOTON_POINTER(float weight) : PARTICLE_POINTER(weight) {;}
    virtual ~PHOTON_POINTER() {;}
 
    virtual  float q2() const = 0;
    virtual  float eorg() const = 0;
    // lab-frame Stokes parameters (qedbg_photon_polarisation.h); virtual photons are unpolarised
    virtual void stokes(float s[3]) const { s[0] = s[1] = s[2] = 0.0f; }

  };

  class BEAM_PHOTON_POINTER : public PHOTON_POINTER
  {
    
  private:
    
    const PHOTON* partic_;
    
  public:
    
    BEAM_PHOTON_POINTER() {;}
    
    BEAM_PHOTON_POINTER(const PHOTON* partic, float weight) : PHOTON_POINTER(weight)
    {
      partic_ =partic;
    }
    
    virtual ~BEAM_PHOTON_POINTER() {;}
    
    virtual inline void velocities(float& vx, float& vy) const {partic_->velocities(vx, vy);}
    
    virtual inline const TRIDVECTOR& getSpin() const 
    { 
      std::cerr << " a photon has no spin " << std::endl;
      exit(1);
      /*       const TRIDVECTOR tv(0.0,0.0,0.0); */
      /*       return tv; */
    }
    
    virtual inline float energy() const { return partic_->energy();}
  
    virtual inline float q2() const { return 0.0;};
    virtual inline float eorg() const { return -1.0;};
    virtual inline void stokes(float s[3]) const { partic_->getStokes(s); }
    
    inline float spin(int nc) const  { return partic_->getSpin()(nc-1);}
    
  };
  //////////////////
  
  class EXTRA_PHOTON_POINTER : public PHOTON_POINTER
  {
    EXTRA* extra_phot_;
    
  public :
    
    EXTRA_PHOTON_POINTER() : extra_phot_(NULL) {;}
    ~EXTRA_PHOTON_POINTER()
      {
	//  std::cout << " destructor EXTRA_PHOTON_POINTER() pointer = "<< extra_phot_ << endl;
	delete extra_phot_;
      }
    
    EXTRA_PHOTON_POINTER& operator=(const EXTRA_PHOTON_POINTER& pt)
      {
	if (this == &pt) return *this; // Gracefully handle self assignment
	
	// call base class assignment operator
	PHOTON_POINTER::operator=(pt);
	
	float energy,vx,vy, q2, eorg;
	pt.extra_phot_->get_parameters(energy,vx,vy,q2,eorg);
	EXTRA* tmp = new EXTRA(energy, vx, vy, q2, eorg);
	delete extra_phot_;
	extra_phot_ = tmp;
	
	return *this;
      }

    EXTRA_PHOTON_POINTER(const EXTRA_PHOTON_POINTER& pt) : PHOTON_POINTER(pt)
    {
      float energy,vx,vy, q2, eorg;
      pt.extra_phot_->get_parameters(energy,vx,vy,q2,eorg);
      extra_phot_ = new EXTRA(energy, vx, vy,q2,eorg);
    }
    
    EXTRA_PHOTON_POINTER(float energy,float vx,float vy,float q2,float eorg,float weight) : PHOTON_POINTER(weight)
    {
      extra_phot_ = new EXTRA(energy, vx, vy,q2,eorg);
    }
    virtual inline float energy() const {return extra_phot_->energy();}
    virtual inline float q2() const {return extra_phot_->q2();}
    virtual inline float eorg() const {return extra_phot_->eorg();}
    virtual inline void velocities(float& vx, float& vy) const { extra_phot_->velocities(vx, vy);}
    virtual inline const TRIDVECTOR& getSpin() const 
    { 
      std::cerr << " no spin for extra photon " << std::endl;
      exit(1);
      /*       const TRIDVECTOR tv(0.0,0.0,0.0); */
      /*       return tv; */
    }
    
  };
  //////////////////////////////////////////////////////////////////////
  
  
  //! for each cell, a list of pointers to particles assigned to this celle (beam1)
  std::vector< std::vector<BEAM_PARTICLE_POINTER> > part_pointer1_;
  //! for each cell, a list of pointers to particles assigned to this celle (beam2)
  std::vector< std::vector<BEAM_PARTICLE_POINTER> > part_pointer2_;
  
  //! for each cell, a list of pointers to photons assigned to this celle (beam1)
  std::vector< std::vector<BEAM_PHOTON_POINTER> > grid_photon1_;
  //! for each cell, a list of pointers to photons assigned to this celle (beam2)
  std::vector< std::vector<BEAM_PHOTON_POINTER> > grid_photon2_;
  
  //! for each cell, a list of pointers to extra_photons assigned to this celle (beam2)
  std::vector< std::vector<EXTRA_PHOTON_POINTER> > extra_photon_pointer1_;
  //! for each cell, a list of pointers to extra_photons assigned to this celle (beam2)
  std::vector< std::vector<EXTRA_PHOTON_POINTER> > extra_photon_pointer2_;

  // Sparse encounter-local indices for the historical cell buckets.  The
  // bucket representation itself is retained to keep object lifetimes and
  // per-cell insertion order identical to the legacy implementation.
  feebl::sparse_cells::ActiveCellSet particle_cells1_;
  feebl::sparse_cells::ActiveCellSet particle_cells2_;
  feebl::sparse_cells::ActiveCellSet photon_cells1_;
  feebl::sparse_cells::ActiveCellSet photon_cells2_;
  feebl::sparse_cells::ActiveCellSet extra_photon_cells1_;
  feebl::sparse_cells::ActiveCellSet extra_photon_cells2_;
  std::vector<int> sparse_cell_scratch_;
  std::vector<int> sparse_traversal_cells_;

  bool sparse_cells_enabled_;
  // photon_polarisation switch: beamstrahlung photons get lab-frame Stokes parameters at emission
  int photon_polarisation_;
  // coherent_model switch: 1 = corrected sub-stepping of large coherent conversion probabilities (move_photons2)
  int coherent_model_;
  bool wavefront_outer_parallel_lane_;
  bool electron_lumi_stable_ids_enabled_;
  const std::vector<std::vector<PRIMARY_PARTICLE_ID> >*
      electron_lumi_stable_ids1_;
  const std::vector<std::vector<PRIMARY_PARTICLE_ID> >*
      electron_lumi_stable_ids2_;
  unsigned long long sparse_dense_traversal_visits_;
  unsigned long long sparse_actual_traversal_visits_;
  
  // A std::vector of the photons generated by incoherent pairs. They do not iteract with the beam.
  std::vector<TERTPHOTON> tertphot_;
  
  // switch photon_timing (0 legacy; 1, 2, 3: newborn beamstrahlung photons start at their emitter's
  // start-of-step position, 1 and 3 also halve their weight in the step of their birth, 3 also gives them
  // the emitter's mean direction over the step's kick) and, for the current
  // step, the index of the first photon born in it in the photon lists of slice i1 of beam 1 and i2 of beam 2
  int photon_timing_;
  std::size_t photon_birth_[2];
  // switch photon_angle (0 collinear emission; 1, 2: angle out of the emitter's orbit plane from the LCFA
  // spectrum, 2 only for photons whose formation length is below photon_angle_lf_max_ [nm], half the bunch
  // length of the beam opposing the emitter of beam 1 resp. 2)
  int photon_angle_;
  float photon_angle_lf_max_[2];
  // state of the angle draws' own random stream, seeded at the first set_photon_angle with photon_angle != 0
  bool photon_angle_seeded_;
  std::uint64_t photon_angle_stream_;

  RNDM* rndm_generator_;
  
  
  RESULTS results_;
  COHERENT_RESULTS coherent_results_;
  // PHOTON_RESULTS photon_results_;
  TRIDENT_RESULTS trident_results_;
  
  
  COMPT compt_;
  
  BHABHA bhabhas_;
  
  
  FILE_IN_OUT* photon_file_;
  FILE_IN_OUT* hadron_file_;
  
  LUMI_HEAP_EE lumi_heap_ee_;
  LUMI_HEAP lumi_heap_eg_;
  LUMI_HEAP lumi_heap_ge_;
  LUMI_HEAP lumi_heap_gg_;
  
  
  DISTRIBUTE distribute1_;
  DISTRIBUTE distribute2_;

  // Reused by pool_private. The fixed logical shards make the reduction
  // independent of which worker happens to execute a shard.
  std::vector<PHI_FLOAT> cpu_private_rho_;

  // Lazy, GRID-owned storage for electron_lumi_precompute=1. Empty vectors
  // keep the default legacy path allocation-free; capacity survives all
  // encounters in an opted-in run.
  std::vector<ELECTRON_LUMI_CACHED_PARTICLE> electron_lumi_cache1_;
  std::vector<ELECTRON_LUMI_CACHED_PARTICLE> electron_lumi_cache2_;
  std::vector<int> electron_lumi_occupied_cells_;

  // Lazy, GRID-owned 64-byte-aligned SoA and fixed block output storage for
  // electron_lumi_precompute=2.  Keeping it opaque leaves the historical grid
  // header independent of the bounded vector-kernel implementation.
  feebl::legacy_electron_lumi_blocked::Workspace*
      electron_lumi_blocked_workspace_;
  bool electron_lumi_mode3_seen_;
  unsigned long long electron_lumi_mode3_calls_;
  unsigned long long electron_lumi_mode3_cells_;
  unsigned long long electron_lumi_mode3_pairs_;
  unsigned long long electron_lumi_mode3_parallel_pairs_;
  unsigned long long electron_lumi_mode3_parallel_sample_candidates_;
  unsigned long long electron_lumi_mode3_serial_sample_merge_candidates_;
  unsigned long long electron_lumi_mode3_serial_result_commits_;
  unsigned long long electron_lumi_mode3_legacy_heap_candidates_;
  unsigned long long electron_lumi_mode3_counter_heap_accepts_;
  int electron_lumi_mode3_sampling_mode_;
  unsigned long long electron_lumi_mode3_bernoulli_draws_;
  unsigned long long electron_lumi_mode3_bernoulli_rejects_;
  unsigned long long electron_lumi_mode3_bernoulli_accepted_pairs_;
  unsigned long long electron_lumi_mode3_bernoulli_accepted_records_;
  unsigned long long electron_lumi_mode3_outer_fused_pairs_;
  unsigned long long electron_lumi_mode3_geometric_rows_;
  unsigned long long electron_lumi_mode3_geometric_rng_calls_;
  unsigned long long electron_lumi_mode3_geometric_successes_;
  unsigned long long electron_lumi_mode3_geometric_integer_copies_;
  unsigned long long electron_lumi_mode3_geometric_fallback_rows_;
  unsigned long long electron_lumi_mode3_geometric_fallback_pairs_;
  std::size_t electron_lumi_mode3_max_workers_;
  double electron_lumi_mode3_preflight_ms_;
  double electron_lumi_mode3_materialize_ms_;
  double electron_lumi_mode3_evaluate_ms_;
  double electron_lumi_mode3_sample_merge_ms_;
  double electron_lumi_mode3_publish_ms_;
  double electron_lumi_mode3_outer_fused_ms_;

  struct ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT
  {
    bool seen = false;
    std::uint64_t geometry_hash = 0U;
    std::size_t n_x = 0U;
    std::size_t n_y = 0U;
    double x_min = 0.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 0.0;
    double delta_x = 0.0;
    double delta_y = 0.0;
    double normalization = 0.0;
    std::uint64_t max_pairs_per_encounter = 0U;
    double closure_relative_tolerance = 0.0;
    std::uint64_t encounters = 0U;
    std::uint64_t beam1_input_roots = 0U;
    std::uint64_t beam1_accepted_roots = 0U;
    std::uint64_t beam1_rejected_roots = 0U;
    std::uint64_t beam2_input_roots = 0U;
    std::uint64_t beam2_accepted_roots = 0U;
    std::uint64_t beam2_rejected_roots = 0U;
    std::uint64_t physical_pairs = 0U;
    double beam1_input_population = 0.0;
    double beam1_accepted_population = 0.0;
    double beam1_rejected_population = 0.0;
    double beam2_input_population = 0.0;
    double beam2_accepted_population = 0.0;
    double beam2_rejected_population = 0.0;
    double pair_weight_sum = 0.0;
    double density_dot_weight = 0.0;
    double maximum_closure_relative_error = 0.0;
    std::vector<std::uint64_t> plan_hashes;
  };
  ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT electron_lumi_spatial_receipt_;
  bool electron_lumi_spatial_ids_validated_;

  static bool merge_electron_lumi_spatial_receipt(
      ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT& destination,
      const ELECTRON_LUMI_SPATIAL_RUNTIME_RECEIPT& source,
      std::string* reason);
  
  ABSTRACT_MINIJETS* minijets_;
  
  GENERAL_CROSS* cross_;

  // Optional CUDA CIC acceleration.  The pointer keeps CUDA types out of this
  // header and the implementation library provides a CPU-only fail-fast stub.
  feebl::gpu::CicDepositor* gpu_cic_;
  int gpu_cic_mode_;  // 0: inspect deck, -1: serial, 1: CUDA, 2: lazy auto
  feebl::gpu::CicVariant gpu_cic_variant_;
  bool gpu_cic_auto_;
  bool gpu_cic_auto_scanned_;
  bool gpu_cic_auto_locked_serial_;
  int gpu_cic_device_;
  std::size_t gpu_cic_min_particles_;
  std::size_t gpu_cic_max_slice_particles_;
  bool gpu_cic_timing_;
  unsigned long long gpu_cic_calls_;
  unsigned long long gpu_cic_fallback_calls_;
  unsigned long long gpu_cic_fallback_particles_;
  unsigned long long gpu_cic_packed_calls_;
  unsigned long long gpu_cic_packed_particles_;
  double gpu_cic_setup_ms_;
  double gpu_cic_h2d_ms_;
  double gpu_cic_kernel_ms_;
  double gpu_cic_d2h_ms_;
  double gpu_cic_total_ms_;

  // Provenance for the explicitly selected legacy photon-luminosity
  // factorization.  These remain completely silent on the default path.
  bool photon_lumi_factorized_seen_;
  unsigned long long photon_lumi_factorized_calls_;
  unsigned long long photon_lumi_cartesian_calls_;
  unsigned long long photon_lumi_ge_opportunities_;
  unsigned long long photon_lumi_eg_opportunities_;
  unsigned long long photon_lumi_gg_opportunities_;
  unsigned long long photon_lumi_total_opportunities_;
  unsigned long long photon_lumi_cartesian_pairs_visited_;
  feebl::legacy_photon_lumi::Workspace* photon_lumi_workspace_;

  static feebl::legacy_photon_lumi::PhotonSample
  load_photon_lumi_sample(const void* context, std::size_t index);
  static feebl::legacy_photon_lumi::ChargedSample
  load_charged_lumi_sample(const void* context, std::size_t index);

  void configure_gpu_cic();
  
  /* inline void connect_secondaries(PAIR_BEAM* secondaries) */
  /* { */
  /*   secondaries_ = secondaries; */
  /* } */
  
  void  lumi_init( const SWITCHES& switches);
  
  //void deltaVelocityFromFieldCIC(float xpart,float ypart, float energy,   PHI_FLOAT *phi, float distance,float& ax, float& ay);
  
  // spread_energy moved to energySpreadCPP.h as energy_spread_factor() and is
  // now applied at beam setup rather than at collision time.

  void step_lumi_precomputed(const SWITCHES& switches,
                             bool externally_supplied_fine_dot,
                             double fine_lumi_dot);
  void step_lumi_blocked(const SWITCHES& switches,
                         bool externally_supplied_fine_dot,
                         double fine_lumi_dot,
                         float min_z,
                         int time_counter,
                         const feebl::electron_lumi_sampling::PairIdentity*
                             sampling_identity_prefix = NULL);
  void step_lumi_spatial(const SWITCHES& switches,
                         int beamslice1,
                         int beamslice2);
  void collide_ee_spatial(const PARTICLE& particle1,
                          const PARTICLE& particle2,
                          double weight,
                          const SWITCHES& switches,
                          RESULTS& destination) const;
  
void collide_ee(int cellx, int celly,float min_z, const BEAM_PARTICLE_POINTER& pointer1, const BEAM_PARTICLE_POINTER& pointer2, SWITCHES& switches,PAIR_BEAM& secondaries,int time_counter, int beamslice1, int beamslice2) ;
  
  void collide_ge(int cellx, int celly,float min_z, const BEAM_PHOTON_POINTER& photon_pointer, const BEAM_PARTICLE_POINTER& particle_pointer, PAIR_BEAM& secondaries, SWITCHES& switches, const PAIR_PARAMETER& pair_parameter, RNDM& rndm_generator);
  
  void collide_eg(int cellx, int celly,float min_z,const BEAM_PARTICLE_POINTER& particle_pointer, const BEAM_PHOTON_POINTER& photon_pointer, PAIR_BEAM& secondaries, SWITCHES& switches, const PAIR_PARAMETER& pair_parameter, RNDM& rndm_generator);

  void collide_gg(int cellx, int celly, float min_z, const BEAM_PHOTON_POINTER& photon_pointer1, const BEAM_PHOTON_POINTER& photon_pointer2, PAIR_BEAM& secondaries, PAIR_BEAM& muons, SWITCHES& switches, RNDM& rndm_generator, double *returnCrossSectionToAdd);

  // with dir = 1 and photon_q2=0., it is the old collide_ge_2
  // with dir = 2 and photon_q2=0., it is the old collide_eg_2
  // with dir = 1 and photon_q2 != 0., it is the old collide_ge_3
  // with dir = 2 and photon_q2!= 0., it is the old collide_eg_3

  // argument  index_of_process is old 'scdn1+scdn2'
  // 0 : Breit-Wheeler
  // 1 : Bethe-Heitler
  // 2 : Landau-Lifschitz

  inline void collide_compton(int index_of_process, int cellx, int celly,float min_z, const PHOTON_POINTER& extr_phot, const BEAM_PARTICLE_POINTER& particle,PAIR_BEAM& secondaries,SWITCHES& switches, RNDM& rndm_generator, int dir)
  {
    if (switches.get_do_compt())
      {
	float particleEnergy = particle.energy();
	float particleVx,particleVy;
	particle.velocities(particleVx, particleVy);

	float photonEnergy = extr_phot.energy();
	float photon_q2 =  extr_phot.q2(); 

	float weight = extr_phot.weight()*particle.weight();

	compt_.compt_do(mesh_, cellx, celly,min_z, secondaries,index_of_process,particleEnergy,photonEnergy,photon_q2,particleVx, particleVy,weight,dir,switches, rndm_generator);
      }
  }

  void collide_gg_XX(int index_of_process, int cellx, int celly,float min_z, const PHOTON_POINTER& photon1, const PHOTON_POINTER& photon2, SWITCHES& switches, PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator, double *returnCrossSectionToAdd);

  void make_hadrons_gg2(float min_z, float energy1,float q2_1,float energy2,float q2_2,float lumi, SWITCHES& switches,double *returnCrossSectionToAdd, RNDM& rndm_generator);

  void isr2(float e1,float e2,float *e1p,float *e2p, RNDM& rndm_generator);

  //void interpolePotential(float xpart,float ypart, PHI_FLOAT& h_x, PHI_FLOAT& h_y, PHI_FLOAT& phi1_x, PHI_FLOAT& phi2_x, PHI_FLOAT& phi3_x, PHI_FLOAT& phi1_y,PHI_FLOAT& phi2_y, PHI_FLOAT& phi3_y, PHI_FLOAT *phi);

  inline void store_coherent_particle(BEAM& beam,float energy,float vx,float vy, float x,float y,int slice)
  {
    beam.newCoherent(slice, x,y,vx,vy,energy);
    coherent_results_.addCountEnergy(energy);
  }

  inline void store_trident_particle(BEAM& beam,float energy,float vx,float vy, float x,float y,int slice)
  {
    beam.newTrident(slice, x,y,vx,vy,energy);
    //  std::cout << "GRID::create trident : vel "<< energy << " "  << x << " " << y << " " << vx << " " << vy << std::endl;	    
    trident_results_.addCountEnergy(energy);
  }

  int coherent_generate(BEAM& beam,int i_slice, float upsilon,PHOTON& phot, RNDM& rndm_generator, float tau = 0.0f);

  // tau (photon_polarisation=1): the photon's linear Stokes parameter along the field (0: unpolarised, unchanged)
  bool pick_coherent_energy(float ups,float& energy, RNDM& rndm_generator, float tau = 0.0f);

  void apply_electric_field_on_pair(float step_2, float ex, float ey, float& vx,float& vy,float e_inv) const
  {
    // apply electric field in one step : step_2
    // e_inv : inverse of energy

    vx += step_2*ex*e_inv;
    vy += step_2*ey*e_inv;
  }

  /*
    void scale_pair(float vold2, float& vx, float& vy, float& vz, float& vx0, float& vy0, float& vz0,float& eng, float& e_inv, float& e_inv2 ) const
    {
    #ifdef SCALE_ENERGY
    float scal;
    scal=sqrt((vold2*eng*eng + EMASS2)/((vx*vx+vy*vy+vz*vz)*eng*eng + EMASS2));
    vx*=scal;
    vy*=scal;
    vz*=scal;
    #ifdef PAIR_SYN
    vx0 *= scal;
    vy0 *= scal;
    vz0 *= scal;
    #endif
    eng/=scal;
    e_inv=1.0/eng;
    e_inv2=e_inv*e_inv;
    #endif
    }
  */



  int field_pair(const PAIR_PARTICLE& pair, const std::vector<GENERAL_GRID*>& grids,float& ex,float& ey, float& bx,float& by, int extra_grids, float charge_sign_0);

  //void step_pair_1X(const std::vector<GENERAL_GRID*>& grids, PAIR_PARTICLE& pair,float step,int n, int extra_grids, float charge_sign_0, RNDM& rndm_generator);
  void step_pair_1(const std::vector<GENERAL_GRID*>& grids, PAIR_PARTICLE& pair,double mass,float step,int n, int extra_grids, float charge_sign_0, RNDM& rndm_generator);
  void step_pair_1_tertphot(const std::vector<GENERAL_GRID*>& grids, PAIR_PARTICLE& pair,double mass,float step,int n, int extra_grids, float charge_sign_0, RNDM& rndm_generator);
  // float  doElossParticle(PARTICLE& particle, int sokolov, TRIDVECTOR Efiekd, TRIDVECTOR Bfield, BEAM& beam, float initialEnergyForLoss, int i_slice,float radius_i, float dz, float emin, int do_prod, float charg_sign);

  //void computePhotonEnergies(std::vector<float>& photonEnergies, PARTICLE& particle, float  upsilon, int sokolov, TRIDVECTOR Efield, TRIDVECTOR Bfield,  float energy, float radius_i,  float dz, float charge_sign);

  // photon_timing/photon_angle use the step's kick (vx_kick, vy_kick) and the emitter energy before the
  // step's photons (e_before); photon_polarisation=1: photonPol holds three emission-frame Stokes
  // parameters per photon (NULL: unpolarised), e1 along the transverse field (Ex, Ey) acting on the
  // emitter, e3 along its beam
  void registerPhotons(const std::vector<float>& photonEnergies, PARTICLE& particle, int i_beam, int i_slice,
                       float vx_kick, float vy_kick, float e_before,
                       const std::vector<float>* photonPol = NULL, float Ex = 0.0f, float Ey = 0.0f);
  void addOutOfPlaneAngle(PHOTON& photon, PARTICLE& particle, int i_beam, double omega, double e_emitter, double e_before, float vx_kick, float vy_kick);
  // void referenceSpin(PARTICLE& part, TRIDVECTOR& e1, TRIDVECTOR& e2, TRIDVECTOR& e3, TRIDVECTOR Efield, TRIDVECTOR Bfield, float charge_sign);
  // void distributeElectronScatter1(int i_beam, int i_slice,float ratio, float ratio_i, std::vector< std::vector<BEAM_PARTICLE_POINTER> >& part_pointer);
  void distributeScatter1(const std::vector<PARTICLE*>& theParticles,
                          float ratio, float ratio_i,
                          std::vector<std::vector<BEAM_PARTICLE_POINTER> >& part_pointer,
                          feebl::sparse_cells::ActiveCellSet& active_cells,
                          const float* xy = 0,
                          const std::vector<PRIMARY_PARTICLE_ID>* stable_ids = 0);
  //void distributeElectronScatter2(int i_beam, int i_slice,float ratio, float ratio_i, std::vector< std::vector<BEAM_PARTICLE_POINTER> >& part_pointer);
  void distributeScatter2(
      const std::vector<PARTICLE*>& theParticles, float ratio, float ratio_i,
      std::vector<std::vector<BEAM_PARTICLE_POINTER> >& part_pointer,
      feebl::sparse_cells::ActiveCellSet& active_cells);

  void electronScatter(
      std::vector<std::vector<EXTRA_PHOTON_POINTER> >& extra_phot_ptr,
      feebl::sparse_cells::ActiveCellSet& active_cells,
      int i_beam, int i_slice, float xmin, double s4, double lns4,
      float ratio, int ext_field, int geom, float ratio_i, float r_scal,
      int i_equiv = 6);

  //void distributePhotonInBeam(   std::vector< std::vector<BEAM_PARTICLE_POINTER> >& grid_photon, BEAM* beam, int slice,float ratio, float ratio_i, RNDM& rndm_generator);
  void distributePhotonInBeam(
      std::vector<std::vector<BEAM_PHOTON_POINTER> >& grid_photon,
      feebl::sparse_cells::ActiveCellSet& active_cells,
      int i_beam, int slice, float ratio, float ratio_i,
      RNDM& rndm_generator);

  //! Distributes the virtual photons 
  void distribute_virtual_photons(int i_slice1,
				  int i_slice2,
				  SWITCHES* switches,
				  double s4, double lns4);

  virtual void distribute_coherent_particles(int i_slice1,
					     int i_slice2,
					     int electron_distribution_rho);

  virtual void distribute_trident_particles(int i_slice1,
					    int i_slice2,
					    int electron_distribution_rho);
  //! Distributes the particles for background calculation 
  //void distributeCoherentScatterInBeam1(int i_beam, int i_slice, float ratio, float ratio_i, std::vector< std::vector<BEAM_PARTICLE_POINTER> >& part_pointer);
  // void distributeCoherentScatterInBeam2(int i_beam, int i_slice, float ratio, float ratio_i, std::vector< std::vector<BEAM_PARTICLE_POINTER> >& part_pointer);
  void distribute_particles_for_background(int i_slice1,
					   int i_slice2,  
					   float electron_ratio);

  //!  Distributes the particles from coherent pair creation for the background calculation

  void distribute_coherent_particles_for_background(int i_slice1,
						    int i_slice2, 
						    float electron_ratio);

  void distribute_trident_particles_for_background(int i_slice1,
						   int i_slice2, 
						   float electron_ratio);

  TRIVECTOR electric_field_out_of_main_grid(const std::vector<GENERAL_GRID*>& grids,int beam,PHI_FLOAT x,PHI_FLOAT y, int extra_grids) const;

  //int field_coherent(const std::vector<GENERAL_GRID*>& grids,int beam,PHI_FLOAT x,PHI_FLOAT y,float ener, float distance, float& deltaVx,float& deltaVy, int extra_grids);
  //void assignBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro, DISTRIBUTE& distribute);
  void assignBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);

  //void assignBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro, DISTRIBUTE& distribute);
  void assignBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);
  void assignBeamSliceCICAtomic(SLICE_ON_GRID& sog, int i_slice,
                                DISTRIBUTE& distribute);
  void assignBeamSliceCICPrivate(SLICE_ON_GRID& sog, int i_slice,
                                 DISTRIBUTE& distribute);

  //void assignCoherentBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro, DISTRIBUTE& distribute);
  void assignCoherentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);

  //void assignCoherentBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro, DISTRIBUTE& distribute);
  void assignCoherentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);

  void assignTridentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);

  void assignTridentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice, DISTRIBUTE& distribute);

  void move_particles(const std::vector<GENERAL_GRID*>& grids, int i_beam,
                      int i_slice, int interpolation, int do_beamstrahlung,
                      int do_trident, int sokolov, float emin, int do_prod,
                      int extra_grids, float charge_sign, int bmt_rotate,
                      const float* prefetched_fields,
                      std::size_t prefetched_particle_count);

  static void cpu_primary_move_range(void* context,
                                     std::size_t lane,
                                     std::size_t begin,
                                     std::size_t end);
  void move_primary_particles_parallel(
      const std::vector<GENERAL_GRID*>& grids,
      int i_slice1,
      int i_slice2,
      int extra_grids);

  struct GPU2_PRIMARY_MOVE_CONTEXT;
  static void gpu2_primary_move_range(void* context,
                                      std::size_t lane,
                                      std::size_t begin,
                                      std::size_t end);

  void move_coherent_particles(const std::vector<GENERAL_GRID*>& grids, int i_beam, int i_slice, int interpolation, int do_beamstrahlung,float emin, int do_prod,int extra_grids, float charge_sign);

  void move_trident_particles(const std::vector<GENERAL_GRID*>& grids, int i_beam, int i_slice, int interpolation, int do_beamstrahlung,float emin, int do_prod,int extra_grids, float charge_sign);

  inline TRIVECTOR EBfieldOnParticle(const PARTICLE* particle, const std::vector<GENERAL_GRID*>& grids, const PHI_FLOAT *phi, int i_beam, int extra_grids) const
  {
    float xpart, ypart;
    TRIVECTOR EBfield;
    particle->XYposition(xpart, ypart);
    if ( coordinatesInGridRange(xpart, ypart) )
      {
	// electric field in GV/nm
	EBfield = ElectricFieldCIC(xpart, ypart, phi);
      }
    else
      {
	EBfield = electric_field_out_of_main_grid(grids,i_beam, (PHI_FLOAT)xpart, (PHI_FLOAT)ypart, extra_grids);
      }
    return EBfield;
  }

  //float moveSingleParticle(PARTICLE* particle, TRIVECTOR EBfield, float dzOnRadius,bool rotateSpin, const std::vector<GENERAL_GRID*>& grids, BEAM& beam, int i_beam, int i_slice, PHI_FLOAT *phi, float distance, int do_beamstrahlung, int sokolov, float emin,int do_prod, int extra_grids, float charge_sign);

  //void particleBeamstrahlung(PARTICLE* particle, TRIDVECTOR Efield, TRIDVECTOR Bfield, float dzOnRadius,  float emin, std::vector<float>& photonEnergies);
  //void particleBeamstrahlungSokolov(PARTICLE_WITH_SPIN* particle, TRIDVECTOR Efield, TRIDVECTOR Bfield, float dzOnRadius, float emin, float charge_sign, std::vector<float>& photonEnergies);

  void beamstrahlungSingleCoherentParticle(PARTICLE* particle, TRIVECTOR EBfield, float dzOnRadius, const std::vector<GENERAL_GRID*>& grids, int i_beam, int i_slice, const PHI_FLOAT *phi, float /*distance*/, float emin,int do_prod, int extra_grids, float charge_sign, float vx_kick, float vy_kick);

  //void advanceCoherentParticlesNGP();
  //void advanceCoherentParticlesCIC(PARTICLE& particle, const std::vector<GENERAL_GRID*>& grids, PHI_FLOAT *phi, int i_beam, float emin, int do_prod, int extra_grids, float scal_step, RNDM& rndm_generator);
  //float requiv(double lns4, float xmin,int iflag);
  //void mequiv (double s4, double lns4, float xmin,float e,int iflag,float *eph,float *q2,float *one_m_x, RNDM& rndm_generator);

  inline   void saveOnHadronFile(int num, float energy1, float energy2, float min_z, RNDM& rndm_generator, int /* unused: was the EDM4HEP flag */ )
  {
    int i;
    for (i=0;i<num;i++)
      {
	float temp_z = 1e-3*(min_z+rndm_generator.rndm())*mesh_.get_delta_z();

	hadron_file_->save_hadron(energy1,energy2,temp_z);
      }
  }

  inline void clear_photon_pointer()
  {
    photon_cells1_.clear_touched(grid_photon1_);
    photon_cells2_.clear_touched(grid_photon2_);
  }

 public :

  struct SPARSE_CELL_STATS
  {
    bool enabled;
    unsigned long long dense_clear_visits;
    unsigned long long actual_clear_visits;
    unsigned long long avoided_clear_visits;
    unsigned long long dense_traversal_visits;
    unsigned long long actual_traversal_visits;
    unsigned long long avoided_traversal_visits;
  };

  GRID();
  ~GRID();

  // Explicit opt-in. The default remains the legacy dense clear/traversal
  // path until the central input-card layer selects this mode.
  void enable_sparse_cells(bool enabled);
  inline bool sparse_cells_enabled() const { return sparse_cells_enabled_; }
  inline void set_photon_polarisation(int value) { photon_polarisation_ = value; }
  inline void set_coherent_model(int value) { coherent_model_ = value; }
  SPARSE_CELL_STATS sparse_cell_stats() const;

  // Narrow integration hook used by the central readData/SWITCHES backend.
  bool enable_gpu_cic(const std::string& variant,
                      int device,
                      std::string& error);

  inline void connect_random_generator( RNDM* rndm_generator) 
  {
    rndm_generator_ = rndm_generator;
  }

  inline bool random_ok() const
  {
    return rndm_generator_ != NULL;
  }

  void set_photon_angle(int photon_angle, float lf_max_1, float lf_max_2);

  // called before the particles of slices i_slice1 (beam 1) and i_slice2 (beam 2) are moved
  inline void mark_photon_birth(int i_slice1, int i_slice2, int photon_timing)
  {
    photon_timing_ = photon_timing;
    photon_birth_[0] = slice_of_beam_[0].get_beam()->getPhotonVector(i_slice1).size();
    photon_birth_[1] = slice_of_beam_[1].get_beam()->getPhotonVector(i_slice2).size();
  }

  inline RESULTS& get_results() { return results_;}
  inline const RESULTS& get_results() const { return results_;}

  // Worker-local full encounter mode.  The outer EncounterDAG already owns
  // all useful CPU parallelism, so the blocked e-e kernel must stay serial
  // inside this GRID to avoid nested oversubscription.  Publication remains a
  // deterministic keyed-reservoir/result merge at the wave barrier.
  inline void set_wavefront_outer_parallel_lane(bool enabled)
    { wavefront_outer_parallel_lane_ = enabled; }
  bool merge_primary_photon_collision_lane(const GRID& lane,
                                            std::string* reason);
  bool merge_primary_photon_collision_lanes(
      const std::vector<const GRID*>& lanes,
      std::string* reason);
  bool merge_primary_photon_lumi_heaps(
      const std::vector<const GRID*>& lanes,
      std::string* reason);
  void reset_primary_photon_collision_lane(const SWITCHES& switches);

  inline const COHERENT_RESULTS& get_coherent_results() const { return coherent_results_;}

  // inline const PHOTON_RESULTS& get_photon_results() const { return photon_results_;}

  inline const TRIDENT_RESULTS& get_trident_results() const { return trident_results_;}

  inline const COMPT& get_compton() const {return compt_;}

  /* inline  void get_distributes(DISTRIBUTE& distr1, DISTRIBUTE& distr2) const  */
  /* {  */
  /*   distr1 = distribute1_; */
  /*   distr2 = distribute2_; */
  /* } */

  inline void get_miss(float& miss1, long& out1, float& miss2, long& out2) const 
  {
    miss1 = distribute1_.delta;
    out1 = distribute1_.tot;
    miss2 = distribute2_.delta;
    out2 = distribute2_.tot;
  }

  inline const ABSTRACT_MINIJETS* get_minijets() const {return minijets_;}

  inline const BHABHA& get_bhabhas() const { return bhabhas_;}

  void read(const PARAMETERS& param, int automatic);

  inline void generalInit(const SWITCHES& switches, std::string photon_file,   std::string hadron_file, std::string c_phot_name, std::string bhabha_samples_file, std::string bhabha_photon_samples_file, GENERAL_CROSS* cross)
  {
    results_.connect_switches(&switches);
    if (switches.get_do_coherent()) coherent_results_.init();
    if (switches.get_do_trident()) trident_results_.init();
   
    lumi_init(switches);
    if ( !photon_file.empty() ) 
      {  
	photon_file_ = new FILE_IN_OUT();
	photon_file_->open_file(photon_file, "w");
      }
    else photon_file_ = NULL;
   
   
    if ( !hadron_file.empty() ) 
      {  
	hadron_file_ = new FILE_IN_OUT();
	hadron_file_->open_file(hadron_file, "w");
      }
    else hadron_file_ = NULL;
    cross_ = cross;
    if ( !c_phot_name.empty() ) 
      {
	compt_.connect_compt_phot_file(c_phot_name); 
      }
    if (switches.get_do_bhabhas()) bhabhas_.load_samples(switches.get_do_bhabhas(), bhabha_samples_file, bhabha_photon_samples_file);
  }

  void get_cuts(float& cx,float& cy,float& cz) const;
  //void init_grid_comp (int n_cell_x, int n_cell_y, int integration_method);
  void init_grid_comp (int n_cell_x, int n_cell_y, int integration_method,   FFT_SERVER* fourier = NULL);

  /*! Routine to calculate the parameters necessary for the iterative method of
    the potential calculation sor2 */
  void init_sor2 (PHI_FLOAT *parameter);

  inline void init_extra_photons()
  {
    int totalSize = n_cell_x_*n_cell_y_;
    //  extra_photons_cell1_ = std::vector< std::list<EXTRA> >(totalSize);
    //   extra_photons_cell2_ = std::vector< std::list<EXTRA> >(totalSize);
    extra_photon_pointer1_ = std::vector< std::vector<EXTRA_PHOTON_POINTER> >(totalSize);
    extra_photon_pointer2_ = std::vector< std::vector<EXTRA_PHOTON_POINTER> >(totalSize);
    extra_photon_cells1_.reset(static_cast<std::size_t>(totalSize));
    extra_photon_cells2_.reset(static_cast<std::size_t>(totalSize));
    clear_extra_photons();
    /*      extra_photons1_.init_extra_photons(totalSize); */
    /*      extra_photons2_.init_extra_photons(totalSize); */
    /*      extra_photons1_.clear_extra_photons(); */
    /*      extra_photons2_.clear_extra_photons(); */
  }

  inline void init_jet(float s,float ptmin,int iparam,int jet_pythia, int jet_select, std::string jetfileName)
  {
    if (!jet_pythia)
      {
	minijets_ = new MINIJETS(s, ptmin, iparam, jet_select,jetfileName);
      }
    else
      {
	minijets_ = new MINIJETS_PYTHIA(s, ptmin, iparam, jet_select,jetfileName);
      }
  };

  inline void all_distribute(int i1, int i2, SWITCHES& switches,double s4, double lns4)
  {
    electron_lumi_stable_ids_enabled_ =
        switches.get_electron_lumi_precompute() == 3 &&
        switches.get_electron_lumi_sampling() != 0;
    if (electron_lumi_stable_ids_enabled_ &&
        electron_lumi_stable_ids1_ == NULL)
      {
        electron_lumi_stable_ids1_ =
            &slice_of_beam_[0].get_beam()->particle_beam()
                 .getParticleIdSlicesStrict();
        electron_lumi_stable_ids2_ =
            &slice_of_beam_[1].get_beam()->particle_beam()
                 .getParticleIdSlicesStrict();
      }
    // distribute the particle (electrons...) charges over the grid
    distribute_particles(i1, i2, switches.get_electron_distribution_rho(), switches.get_force_symmetric());
  
    // Distributes the particles for background calculation 
    distribute_particles_for_background(i1, i2, switches.get_electron_ratio());
  
    if (switches.get_do_pairs()||switches.get_do_hadrons()||switches.get_do_compt() || switches.get_do_muons())
      {
	distribute_virtual_photons(i1, i2, &switches,s4, lns4);
      }
  }

  // GPU2 direct deposition owns rho on the device, but the collision cell
  // lists and virtual-photon RNG sequence remain authoritative on the host.
  // This is exactly all_distribute() with only its first, RNG-free charge
  // deposition operation omitted.
  inline void all_distribute_background_only(int i1, int i2,
                                              SWITCHES& switches,
                                              double s4, double lns4)
  {
    electron_lumi_stable_ids_enabled_ =
        switches.get_electron_lumi_precompute() == 3 &&
        switches.get_electron_lumi_sampling() != 0;
    if (electron_lumi_stable_ids_enabled_ &&
        electron_lumi_stable_ids1_ == NULL)
      {
        electron_lumi_stable_ids1_ =
            &slice_of_beam_[0].get_beam()->particle_beam()
                 .getParticleIdSlicesStrict();
        electron_lumi_stable_ids2_ =
            &slice_of_beam_[1].get_beam()->particle_beam()
                 .getParticleIdSlicesStrict();
      }
    distribute_particles_for_background(
        i1, i2, switches.get_electron_ratio());
    if (switches.get_do_pairs() || switches.get_do_hadrons() ||
        switches.get_do_compt() || switches.get_do_muons())
      distribute_virtual_photons(i1, i2, &switches, s4, lns4);
  }

  // Read-only current-substep counters; GUI coverage differences these around
  // the actual native deposition, never substituting an inferred scan count.
  inline std::array<long int,4> current_deposition_counts() const {
    return {{distribute1_.in,distribute1_.out,distribute2_.in,distribute2_.out}};
  }

  // Preserve check_distribute()'s historical aggregate diagnostics when the
  // primary in/out counts originate from an audited external deposition
  // stage (resident GPU2 or wavefront worker-private CPU workspaces).
  inline void add_external_distribution_counts(std::size_t beam1_in,
                                                std::size_t beam1_out,
                                                std::size_t beam2_in,
                                                std::size_t beam2_out)
  {
    distribute1_.in += static_cast<long int>(beam1_in);
    distribute1_.out += static_cast<long int>(beam1_out);
    distribute2_.in += static_cast<long int>(beam2_in);
    distribute2_.out += static_cast<long int>(beam2_out);
  }

  inline void add_gpu2_distribution_counts(std::size_t beam1_in,
                                            std::size_t beam1_out,
                                            std::size_t beam2_in,
                                            std::size_t beam2_out)
  {
    add_external_distribution_counts(
        beam1_in, beam1_out, beam2_in, beam2_out);
  }

  virtual void distribute_particles(int i_slice1,
				    int i_slice2, 
				    int electron_distribution_rho, 
				    int force_symmetric);

  void distribute_photons(int slice_1,int slice_2, float photon_ratio, RNDM& rndm_generator);

  void check_distribute(int what);

  void save_lumi_on_files(SWITCHES& switches, std::string lumi_ee_out, std::string lumi_eg_out, std::string lumi_ge_out, std::string lumi_gg_out);

  void save_tertphot_on_file(std::string tertphot);

  inline void clear_extra_photons()
  {
    extra_photon_cells1_.clear_touched(extra_photon_pointer1_);
    extra_photon_cells2_.clear_touched(extra_photon_pointer2_);
  }

  //! compute luminosity at each step
  /*!
    for each grid cell, call collide_ee(), computing all collisions between a particle from beam1 and a particle from beam2. The summed luminosity is added in 'results' (lumi_fine)
  */

  void step_lumi(float min_z,PAIR_BEAM& secondaries, int time_counter, SWITCHES& switches, int beamslice1, int beamslice2) ;

  // Same cell/pair traversal and RNG ordering as step_lumi(), with only the
  // rho dot product supplied by a resident device reduction.
  void step_lumi_with_fine_dot(float min_z, PAIR_BEAM& secondaries,
                               int time_counter, SWITCHES& switches,
                               int beamslice1, int beamslice2,
                               double fine_lumi_dot,
                               const feebl::electron_lumi_sampling::PairIdentity*
                                   sampling_identity_prefix = NULL);

  inline void update_slice_charge(int slice1, int slice2)
  {
    /*  beam1_->meanPositionOfSlice(slice1, rho_x_1_, rho_y_1_); */
    /*  beam2_->meanPositionOfSlice(slice2, rho_x_2_, rho_y_2_); */
    slice_of_beam_[0].get_beam()->meanPositionOfSlice(slice1, rho_x_1_, rho_y_1_);
    slice_of_beam_[1].get_beam()->meanPositionOfSlice(slice2, rho_x_2_, rho_y_2_);
    float  rho_sum = slice_of_beam_[0].get_beam()->numberOfParticlesOfSlice(slice1);
    rho_sum_1_ = rho_sum*slice_of_beam_[0].get_nb_part_per_macro(); 
    rho_sum = slice_of_beam_[1].get_beam()->numberOfParticlesOfSlice(slice2); 
    rho_sum_2_ = rho_sum*slice_of_beam_[1].get_nb_part_per_macro();    
  }

  inline void moveAllParticles(const std::vector<GENERAL_GRID*>& grids,  int i_slice1, int i_slice2 , int interpolation,int do_beamstrahlung, int sokolov, float emin, int do_prod, int extra_grids, float charge_sign,int bmt_rotate,int do_trident, int do_coherent, int cpu_primary_move)
  {
    if (cpu_primary_move)
      {
        // SWITCHES/FEEBL rejects unsupported cards before simulation.  Keep a
        // second guard at the kernel boundary so a future direct caller cannot
        // accidentally enter a shared-RNG or shared-result region.
        if (interpolation != 2 || do_beamstrahlung != 0 || sokolov != 0 ||
            do_prod != 0 || extra_grids != 0 || bmt_rotate != 0 ||
            do_trident != 0 || do_coherent != 0)
          throw std::logic_error(
              "CPU primary movement received an unsupported physics scope");
        move_primary_particles_parallel(grids, i_slice1, i_slice2,
                                         extra_grids);
        return;
      }
    move_particles(grids, 1, i_slice1,interpolation, do_beamstrahlung,do_trident,sokolov,  emin, do_prod, extra_grids, charge_sign,bmt_rotate, NULL, 0);
    move_coherent_particles(grids, 1, i_slice1, interpolation, do_beamstrahlung, emin, do_prod, extra_grids, charge_sign);
  
    //  std::cout << " advance the second beam.... " << std::endl;
    move_particles(grids, 2, i_slice2,interpolation, do_beamstrahlung,do_trident,sokolov,  emin, do_prod, extra_grids, charge_sign,bmt_rotate, NULL, 0);
    move_coherent_particles(grids, 2, i_slice2, interpolation, do_beamstrahlung, emin, do_prod, extra_grids, charge_sign);
    if(do_trident)
      {
	move_trident_particles(grids, 1, i_slice1, interpolation, do_beamstrahlung, emin, do_prod, extra_grids, charge_sign);
	move_trident_particles(grids, 2, i_slice2, interpolation, do_beamstrahlung, emin, do_prod, extra_grids, charge_sign);
      }
  };

  // gpu2_scope=primary entry point.  The resident solver returns the
  // opposing (Ex,Ey) in each slice's historical particle order.  This method
  // deliberately calls the same primary mover implementation as the host-phi
  // path and keeps beam 1 before beam 2, preserving every host RNG draw and
  // all BMT/ST/beamstrahlung side effects.  Coherent/trident and extra-grid
  // consumers are excluded by FEEBL preflight before this can be entered.
  static bool compactPrimaryFieldContractEligible(
      int interpolation, int extra_grids, int do_prod)
  {
    return interpolation == 2 && extra_grids == 0 && do_prod == 0;
  }

  static bool compactPrimaryParallelMoverContractEligible(
      int interpolation, int do_beamstrahlung, int sokolov, int do_prod,
      int extra_grids, int bmt_rotate, int do_trident, int do_coherent)
  {
    return interpolation == 2 && do_beamstrahlung == 0 && sokolov == 0 &&
           do_prod == 0 && extra_grids == 0 && bmt_rotate == 0 &&
           do_trident == 0 && do_coherent == 0;
  }

  // Gather the opposing compact (Ex,Ey) fields for two primary slices from
  // this GRID's already-solved phi arrays.  Particle order and the
  // out-of-main-grid behavior are identical to move_particles().
  void gatherPrimaryFieldsForSlices(
      int i_slice1, int i_slice2,
      float* fields1, std::size_t field_particle_count1,
      float* fields2, std::size_t field_particle_count2) const;

  // Physics-owning serial mover for an audited compact field provider.  This
  // deliberately retains historical beam order and therefore all RNG,
  // beamstrahlung, ST, BMT, photon, and RESULTS side effects.
  void movePrimaryParticlesWithPrefetchedFields(
      const std::vector<GENERAL_GRID*>& grids,
      int i_slice1, int i_slice2, int interpolation,
      int do_beamstrahlung, int sokolov, float emin, int do_prod,
      int extra_grids, float charge_sign, int bmt_rotate,
      const float* fields1, std::size_t field_particle_count1,
      const float* fields2, std::size_t field_particle_count2,
      int do_trident, int do_coherent);

  void movePrimaryParticlesWithFields(
      const std::vector<GENERAL_GRID*>& grids,
      int i_slice1, int i_slice2, int interpolation,
      int do_beamstrahlung, int sokolov, float emin, int do_prod,
      int extra_grids, float charge_sign, int bmt_rotate,
      const float* fields1, std::size_t field_particle_count1,
      const float* fields2, std::size_t field_particle_count2,
      const std::string& primary_mover, int host_threads,
      int do_trident, int do_coherent);

  inline void move_photons(BEAM& beam,int i_beam,int i_slice)
  {
    float delta;
    delta=step_*    slice_of_beam_[i_beam-1].get_scal_step();
    beam.move_photon_beam(i_slice, delta);
  }

  void move_photons2(BEAM& beam,int ibeam,int i_slice,RNDM& rndm_generator);

  void move_pairs(const std::vector<GENERAL_GRID*>& grids, PAIR_BEAM& pairs, int i_slice, double d_eps_1, double d_eps_2, int extra_grids, float charge_sign_0, RNDM& rndm_generator);

  void move_pairs_tertphot(const std::vector<GENERAL_GRID*>& grids, PAIR_BEAM& pairs, int i_slice, double d_eps_1, double d_eps_2, int extra_grids, float charge_sign_0, RNDM& rndm_generator);

  void photon_lumi(float min_z,SWITCHES& switches, PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator);

  void photon_lumi_2(float min_z,SWITCHES& switches,PAIR_BEAM& secondaries, PAIR_BEAM& muons, RNDM& rndm_generator);

  void photon_lumi_3(float min_z,SWITCHES& switches, PAIR_BEAM& secondaries,RNDM& rndm_generator);

  inline void set_bpm_signals() 
  {
    results_.bpm_signal(*slice_of_beam_[0].get_beam());
    results_.bpm_signal(*slice_of_beam_[1].get_beam());
    results_.bpm_signal_coherent(*slice_of_beam_[0].get_beam());
    results_.bpm_signal_coherent(*slice_of_beam_[1].get_beam());

  }

  virtual std::string output_flow() const;

};

class EXTRA_GRID : public GENERAL_GRID
{

 private :

 EXTRA_GRID() : GENERAL_GRID() {;};
  
  virtual void distribute_coherent_particles(int i_slice1,
					     int i_slice2,
					     int electron_distribution_rho);
  
  virtual void distribute_trident_particles(int i_slice1,
					    int i_slice2,
					    int electron_distribution_rho);
  
  //void assignCoherentBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro);
  void assignCoherentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice);
  
  // void assignBeamSliceNGP(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro);
  void assignBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice);
  
  // void assignBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro);
  void assignBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice);
  
  // void assignCoherentBeamSliceCIC(BEAM *beam, int i_slice, PHI_FLOAT *rho, float n_macro);
  void assignCoherentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice);
  
  void assignTridentBeamSliceNGP(SLICE_ON_GRID& sog, int i_slice);
  void assignTridentBeamSliceCIC(SLICE_ON_GRID& sog, int i_slice);
  
 public : 
 EXTRA_GRID(const GENERAL_GRID& grid) : GENERAL_GRID(grid) {;}
  ~EXTRA_GRID();
  
  // old name : void distribute_particles0_n(BEAM *beam1, int i_slice1,
  virtual void distribute_particles(int i_slice1,
				    int i_slice2,
				    int electron_distribution_rho, 
				    int force_symmetric);
};
#endif
