// qedbg_real.h -- extended-precision scalar used for kinematics in the qedBackgrounds kernels.
// Default: long double (x87 80-bit, 64-bit mantissa on x86-64 Linux/GCC) -- no extra dependency.
// Define QEDBG_USE_QUADMATH to use __float128 with libquadmath (offline oracle tools).
// All kinematics are formed in light-cone components, so the relevant small quantities
// (photon virtualities, propagator denominators) never arise from cancellations between
// beam-energy-sized numbers; long double is sufficient for momentum fractions x >~ 1e-12.
#pragma once
#include <cfloat>
#include <cmath>

#ifdef QEDBG_USE_QUADMATH
#include <quadmath.h>
namespace qedbg {
constexpr bool kExtendedLongDouble = true;
typedef __float128 Q;
inline Q q_sqrt(Q x) { return sqrtq(x); }
inline Q q_log(Q x) { return logq(x); }
inline Q q_exp(Q x) { return expq(x); }
inline Q q_expm1(Q x) { return expm1q(x); }
inline Q q_atanh(Q x) { return atanhq(x); }
inline Q q_tanh(Q x) { return tanhq(x); }
inline Q q_sinh(Q x) { return sinhq(x); }
inline Q q_cosh(Q x) { return coshq(x); }
inline Q q_sin(Q x) { return sinq(x); }
inline Q q_cos(Q x) { return cosq(x); }
inline Q q_fabs(Q x) { return fabsq(x); }
inline Q q_atan2(Q y, Q x) { return atan2q(y, x); }
inline Q q_pi() { return M_PIq; }
}  // namespace qedbg
#else
namespace qedbg {
// The exact kernels need an extended-precision long double (x87 80-bit: 64-bit mantissa).  Where long
// double is the IEEE double (e.g. AArch64 macOS) this header still compiles, so builds of the host codes
// never break; the host codes refuse pair_model=2 at input time when this flag is false.
constexpr bool kExtendedLongDouble = (LDBL_MANT_DIG >= 64);
typedef long double Q;
inline Q q_sqrt(Q x) { return std::sqrt(x); }
inline Q q_log(Q x) { return std::log(x); }
inline Q q_exp(Q x) { return std::exp(x); }
inline Q q_expm1(Q x) { return std::expm1(x); }
inline Q q_atanh(Q x) { return std::atanh(x); }
inline Q q_tanh(Q x) { return std::tanh(x); }
inline Q q_sinh(Q x) { return std::sinh(x); }
inline Q q_cosh(Q x) { return std::cosh(x); }
inline Q q_sin(Q x) { return std::sin(x); }
inline Q q_cos(Q x) { return std::cos(x); }
inline Q q_fabs(Q x) { return std::fabs(x); }
inline Q q_atan2(Q y, Q x) { return std::atan2(y, x); }
inline Q q_pi() { return 3.141592653589793238462643383279502884L; }
}  // namespace qedbg
#endif
