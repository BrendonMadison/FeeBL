#ifndef TIMER_SEEN
#define TIMER_SEEN

#include <ctime>

class TIMER
{
  clock_t timer_store[100];
  clock_t old_clock,zero_clock;

 public:

TIMER();

void clear_timer(int no);

void add_timer(int i);

// Start a new attribution interval without charging elapsed work to a legacy
// timer bucket. Used when a separately reported parallel stage completes.
inline void checkpoint() { old_clock = clock(); }

void print_timer(int i);

/* prints all the timers to stdout */

void print_timer_all(int n);

};
#endif
