#ifndef FEEBL_FFTW_PLANNER_MUTEX_CPP_H
#define FEEBL_FFTW_PLANNER_MUTEX_CPP_H

#include <mutex>

namespace feebl {
namespace fftw_planning {

// FFTW documents plan creation and destruction as non-thread-safe.  FeeBL
// creates worker-private field plans and Mode-3 DST plans from different
// translation units, so every FFTW planning API must share this one lock.
inline std::mutex& process_mutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace fftw_planning
}  // namespace feebl

#endif
