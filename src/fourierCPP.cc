#include "fourierCPP.h"

#include <limits>
#include <new>

// FEEBL: FFTW planning mode.
//
// Upstream leaves DO_MEASURE commented out in define.h, noting that
// FFTW_MEASURE "can give a speedup by a factor of a few" on some systems and
// "no difference or a slow down" on others, and that WISDOM would be the
// better long-term fix. That caution is well founded, and measurement here
// reproduces both halves of it. (define.h is not in this file's include chain,
// so setting the macro there would have had no effect regardless.)
//
// Measured on i9-10900KF / FFTW 3.3.10, production build, strictly serial,
// 3 repetitions, median, output BIT-IDENTICAL in every case:
//
//     FCC-ee Z  (1024x64  = 65536 pts)   136.12 s ->  99.02 s   1.375x
//     ILC 250   (  64x512 = 32768 pts)   123.39 s -> 121.59 s   1.015x
//     LEP1      (  64x128 =  8192 pts)     5.89 s ->   6.34 s   0.929x
//
// The cost is a fixed ~0.45 s of planning, so the trade depends entirely on
// transform size and run length: a large win where the FFT dominates, a real
// 7% loss on a short small-grid run. A blanket default would therefore ship a
// regression to short-run users, which is presumably why upstream left it off.
//
// So choose per transform: use FFTW_MEASURE only when the grid is big enough
// for the better plan to repay the planning cost. The threshold sits between
// the LEP1 and ILC grid sizes above; it is calibrated on those three points,
// not derived, and is worth re-checking on other hardware.
//
// Overrides: define FEEBL_FFTW_ESTIMATE to force ESTIMATE everywhere
// (upstream behaviour), or DO_MEASURE to force MEASURE everywhere.
//
// Plans are built once in FIELD::dist_init and executed ~20480 times per run.
// Run-to-run deterministic here: three unloaded runs and one under 8-way CPU
// load produced identical output, so FFTW's plan choice is stable against
// machine load on this system.
#ifndef FEEBL_MEASURE_MIN_POINTS
#define FEEBL_MEASURE_MIN_POINTS 16384
#endif

static inline unsigned feebl_fftw_plan_flag(int n_points,
                                            bool wisdom_only = false,
                                            bool force_estimate = false)
{
  if (force_estimate) return FFTW_ESTIMATE;
  unsigned flag;
#if defined(FEEBL_FFTW_ESTIMATE)
  (void)n_points;
  flag = FFTW_ESTIMATE;
#elif defined(DO_MEASURE)
  (void)n_points;
  flag = FFTW_MEASURE;
#else
  flag = (n_points >= FEEBL_MEASURE_MIN_POINTS)
      ? FFTW_MEASURE : FFTW_ESTIMATE;
#endif
  // The authoritative main GRID is planned first. Worker-private plans must
  // reuse that exact measured algorithm: allowing every lane to benchmark
  // independently can select algebraically equivalent FFT decompositions and
  // introduce deterministic one-ULP trajectory differences. ESTIMATE is
  // already algorithmically fixed and need not consult wisdom.
  if (wisdom_only && flag == FFTW_MEASURE)
    flag |= FFTW_WISDOM_ONLY;
  return flag;
}
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int checked_fft_point_count(const int nn[2]) {
  if (!nn || nn[0] <= 0 || nn[1] <= 0 ||
      nn[0] > std::numeric_limits<int>::max() / nn[1])
    throw std::invalid_argument("invalid or overflowing FFT dimensions");
  return nn[0] * nn[1];
}

}  // namespace

#ifdef USE_FFTW2

FOUR_FFTW2_ONE::FOUR_FFTW2_ONE(std::string prep, int nn[2]) : in_(NULL), out_(NULL) 
{
  plan_ = NULL;
  size_of_data_ = checked_fft_point_count(nn);
  in_ = reinterpret_cast<fftw_complex*>(
      new double[static_cast<std::size_t>(size_of_data_) * 2U]);
  fftw_direction direction;
  if (prep == std::string("for2")) 
    {
      direction = FFTW_FORWARD;
    }
  else if (prep == std::string("back2")) 
    {
      direction = FFTW_BACKWARD;
    }
  else
    {
      std::cerr << " FOUR_FFTW2_ONE:: ERROR unknown type of fft preparation = " << prep << std::endl;
      exit(1);
    }
  {
    std::lock_guard<std::mutex> lock(feebl::fftw_planning::process_mutex());
    plan_ = fftw2d_create_plan_specific(
        nn[1], nn[0], direction, FFTW_ESTIMATE | FFTW_IN_PLACE,
        in_, 1, out_, 1);
  }
  if (plan_ == NULL)
    {
      delete [] reinterpret_cast<double*>(in_);
      in_ = NULL;
      out_ = NULL;
      throw std::runtime_error("FFTW2 failed to construct a field plan");
    }
}
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
FOUR_FFTW2_MANY::FOUR_FFTW2_MANY(std::string prep, int nn[2]) 
{
  plan_a_ = NULL;
  plan_b_ = NULL;
  size_of_data_ = checked_fft_point_count(nn);
  in_ = reinterpret_cast<fftw_complex*>(
      new double[static_cast<std::size_t>(size_of_data_) * 2U]);
  out_ = in_;
      	
  {
    std::lock_guard<std::mutex> lock(feebl::fftw_planning::process_mutex());
    if (prep == std::string("for3"))
      {
        howmanya_ = nn[0]/2;
        howmanyb_ = nn[1];

        istridea_ = nn[0];
        istrideb_ = 1;

        idista_ = 1;
        idistb_ = nn[0];

        plan_a_ = fftw_create_plan_specific(nn[1],FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE,in_,nn[0],out_,1);
        plan_b_ = fftw_create_plan_specific(nn[0],FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE,in_,1,out_,1);
      }
    else if (prep ==  std::string("back3"))
      {
        howmanya_ = nn[1];
        howmanyb_ = nn[0]/2;

        istridea_ = 1;
        istrideb_ = nn[0];

        idista_ = nn[0];
        idistb_ = 1;

        plan_a_ = fftw_create_plan_specific(nn[0],FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE,in_,1,out_,1);
        plan_b_ = fftw_create_plan_specific(nn[1],FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE,in_,nn[0],out_,1);
      }
    else
      {
        std::cerr << " FOUR_FFTW2_MANY:: ERROR unknown type of fft preparation = " << prep << std::endl;
        exit(1);
      }
    if (plan_a_ == NULL || plan_b_ == NULL)
      {
        if (plan_a_ != NULL) fftw_destroy_plan(plan_a_);
        if (plan_b_ != NULL) fftw_destroy_plan(plan_b_);
        plan_a_ = NULL;
        plan_b_ = NULL;
      }
  }
  if (plan_a_ == NULL || plan_b_ == NULL)
    {
      delete [] reinterpret_cast<double*>(in_);
      in_ = NULL;
      out_ = NULL;
      throw std::runtime_error("FFTW2 failed to construct field plans");
    }
}
#endif

#ifdef USE_FFTW3

FOUR_FFTW3_ONE::FOUR_FFTW3_ONE(std::string prep, int nn[2]) 
{
  plan_ = NULL;
  
  size_of_data_ = checked_fft_point_count(nn);
  if (static_cast<std::size_t>(size_of_data_) >
      std::numeric_limits<std::size_t>::max() / sizeof(fftw_complex))
    throw std::bad_alloc();
  in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size_of_data_);
  if (in_ == NULL) throw std::bad_alloc();
  out_ = in_;
  
   
  {
    std::lock_guard<std::mutex> lock(feebl::fftw_planning::process_mutex());
    if (prep == std::string("for2"))
      {
        plan_ = fftw_plan_dft_2d(nn[1],nn[0],in_, out_, FFTW_FORWARD, FFTW_ESTIMATE);
      }
    else if (prep == std::string("back2"))
      {
        plan_ = fftw_plan_dft_2d(nn[1],nn[0],in_, out_, FFTW_BACKWARD,FFTW_ESTIMATE);
      }
    else
      {
        std::cerr << " FOUR_FFTW3:: ERROR unknown type of fft preparation = " << prep << std::endl;
        exit(1);
      }
  }
  if (plan_ == NULL)
    {
      fftw_free(in_);
      in_ = NULL;
      out_ = NULL;
      throw std::runtime_error("FFTW failed to construct a field plan");
    }
}

FOUR_FFTW3_MANY::FOUR_FFTW3_MANY(
    std::string prep, int nn[2], bool wisdom_only, bool force_estimate)
{
  plan_a_ = NULL;
  plan_b_ = NULL;
  
  size_of_data_ = checked_fft_point_count(nn);
  if (static_cast<std::size_t>(size_of_data_) >
      std::numeric_limits<std::size_t>::max() / sizeof(fftw_complex))
    throw std::bad_alloc();
  in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size_of_data_);
  if (in_ == NULL) throw std::bad_alloc();
  out_ = in_;

  {
    std::lock_guard<std::mutex> lock(feebl::fftw_planning::process_mutex());
    if (prep == std::string("for3"))
      {
        plan_a_ = fftw_plan_many_dft( 1, &nn[1],nn[0]/2,in_, NULL,nn[0],1, out_, NULL,nn[0],1,
				      FFTW_FORWARD,feebl_fftw_plan_flag(size_of_data_, wisdom_only, force_estimate));
        plan_b_ = fftw_plan_many_dft( 1, &nn[0],nn[1],in_, NULL,1,nn[0], out_, NULL,1,nn[0],
				      FFTW_FORWARD,feebl_fftw_plan_flag(size_of_data_, wisdom_only, force_estimate));
      }
    else if (prep ==  std::string("back3"))
      {
        plan_a_ = fftw_plan_many_dft( 1, &nn[0],nn[1],in_, NULL,1, nn[0], out_, NULL,1,nn[0],
				      FFTW_BACKWARD,feebl_fftw_plan_flag(size_of_data_, wisdom_only, force_estimate));
        plan_b_ =  fftw_plan_many_dft( 1, &nn[1],nn[0]/2,in_, NULL,nn[0],1, out_, NULL,nn[0],1,
				       FFTW_BACKWARD,feebl_fftw_plan_flag(size_of_data_, wisdom_only, force_estimate));
      }
    else
      {
        std::cerr << " FOUR_FFTW3_MANY:: ERROR unknown type of fft preparation = " << prep << std::endl;
        exit(1);
      }
    if (plan_a_ == NULL || plan_b_ == NULL)
      {
        if (plan_a_ != NULL) fftw_destroy_plan(plan_a_);
        if (plan_b_ != NULL) fftw_destroy_plan(plan_b_);
        plan_a_ = NULL;
        plan_b_ = NULL;
      }
  }
  if (plan_a_ == NULL || plan_b_ == NULL)
    {
      fftw_free(in_);
      in_ = NULL;
      out_ = NULL;
      throw std::runtime_error(
          wisdom_only
              ? "authoritative FFTW wisdom unavailable for worker plan"
              : "FFTW failed to construct a field plan");
    }
}


#endif


#ifdef USE_FFT_LOCAL


LOCAL_FOURIER::LOCAL_FOURIER(std::string prep,int nn[2]) 
    {
      std::size_t k;
      size_of_data_ = checked_fft_point_count(nn);
      in_ = new double[static_cast<std::size_t>(size_of_data_) * 2U];
      for (k = 0U; k < static_cast<std::size_t>(size_of_data_) * 2U; ++k)
        in_[k] = 0.0;
      if (prep == std::string("for2") || prep == std::string("for3")) direction_ = 1;
      else 
	if (prep == std::string("back2") || prep == std::string("back3")) direction_ = -1;
	else 
	  {
	    std::cerr << " LOCAL_FOURIER:: ERROR unknown type of fft preparation = " << prep << std::endl;
	    exit(1);
	  }

      nn_[0] = nn[0];
      nn_[1] = nn[1];
    }


void LOCAL_FOURIER::fourtrans (double* data,int nn[],int isign)
{
  /* System generated locals */
  int i__1, i__2, i__3, i__4, i__5, i__6;
  double d__1;
  
  /* Builtin functions 
    double sin(); */
  
  /* Local variables */
  int idim, ibit, nrem, ntot, i2rev, i3rev, n;
  double theta, tempi, tempr;
  int nprev, i2;
  double wtemp;
  int i1, i3, k1, k2;
  double wi, wr;
  int ip1, ip2, ip3;
  double wpi, wpr;
  int ifp1, ifp2;
  int ndim=2;
  /* Parameter adjustments */
  --nn;
  --data;
  
  /* Function Body */
  ntot = 1;
  i__1 = ndim;
  for (idim = 1; idim <= i__1; ++idim)
    {
      ntot *= nn[idim];
    }
  nprev = 1;
  i__1 = ndim;
  for (idim = 1; idim <= i__1; ++idim) {
    n = nn[idim];
    nrem = ntot / (n * nprev);
    ip1 = nprev << 1;
    ip2 = ip1 * n;
    ip3 = ip2 * nrem;
    i2rev = 1;
    i__2 = ip2;
    i__3 = ip1;
    for (i2 = 1; i__3 < 0 ? i2 >= i__2 : i2 <= i__2; i2 += i__3) {
      if (i2 < i2rev) {
	i__4 = i2 + ip1 - 2;
	for (i1 = i2; i1 <= i__4; i1 += 2) {
	  i__5 = ip3;
	  i__6 = ip2;
	  for (i3 = i1; i__6 < 0 ? i3 >= i__5 : i3 <= i__5; i3 += 
	       i__6) {
	    i3rev = i2rev + i3 - i2;
	    tempr = data[i3];
	    tempi = data[i3 + 1];
			data[i3] = data[i3rev];
	    data[i3 + 1] = data[i3rev + 1];
	    data[i3rev] = tempr;
	    data[i3rev + 1] = tempi;
	  }
	}
      }
      ibit = ip2 / 2;
    L1:
      if (ibit >= ip1 && i2rev > ibit) {
	i2rev -= ibit;
	ibit /= 2;
	goto L1;
      }
      i2rev += ibit;
    }
    ifp1 = ip1;
  L2:
    if (ifp1 < ip2) {
      ifp2 = ifp1 << 1;
      theta = isign * 6.28318530717959 / (ifp2 / ip1);
      /* Computing 2nd power */
      d__1 = sin(theta * .5);
      wpr = d__1 * d__1 * -2.;
      wpi = sin(theta);
      wr = 1.;
      wi = 0.;
      i__3 = ifp1;
      i__2 = ip1;
      for (i3 = 1; i__2 < 0 ? i3 >= i__3 : i3 <= i__3; i3 += i__2) {
	i__4 = i3 + ip1 - 2;
	for (i1 = i3; i1 <= i__4; i1 += 2) {
	  i__6 = ip3;
		    i__5 = ifp2;
	  for (i2 = i1; i__5 < 0 ? i2 >= i__6 : i2 <= i__6; i2 += 
	       i__5) {
	    k1 = i2;
	    k2 = k1 + ifp1;
	    tempr = wr * data[k2] - wi * data[k2 + 1];
	    tempi = wr * data[k2 + 1] + wi * data[k2];
	    data[k2] = data[k1] - tempr;
	    data[k2 + 1] = data[k1 + 1] - tempi;
	    data[k1] += tempr;
	    data[k1 + 1] += tempi;
	  }
	}
	wtemp = wr;
	wr = wr * wpr - wi * wpi + wr;
	wi = wi * wpr + wtemp * wpi + wi;
      }
      ifp1 = ifp2;
      goto L2;
    }
    nprev = n * nprev;
  }
  return;
}
#endif
