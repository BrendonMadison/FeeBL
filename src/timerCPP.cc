#include "timerCPP.h"
#include <iostream>

TIMER::TIMER()
{
  int i;
  old_clock=clock();
  zero_clock=old_clock;
  for(i=0;i<100;i++) timer_store[i]=old_clock;
}

void TIMER::clear_timer(int no)
{
  timer_store[no]=zero_clock;
}

void TIMER::add_timer(int i)
{
  clock_t help,temp_clock;
  temp_clock=clock();
  help=temp_clock-old_clock;
  timer_store[i]+=help;
  old_clock=temp_clock;
}

void TIMER::print_timer(int i)
{
  std::cout << "timer no " << i << " :  " << (float)(timer_store[i]-zero_clock)/CLOCKS_PER_SEC << " s " << std::endl;
}

/* prints all the timers to stdout */

void TIMER::print_timer_all(int n)
{
  int i;
  int sum=0;
  for(i=0;i<=n;i++) sum+=(int)(timer_store[i]-zero_clock);
  std::cout << "timer sum : " << (float)sum/CLOCKS_PER_SEC << std::endl;
}
