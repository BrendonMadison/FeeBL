#ifndef MEMORY_SEEN
#define MEMORY_SEEN

struct memory_pointer
{
  void *memory;
  struct memory_pointer  *next;
};
typedef struct memory_pointer MEMORY_POINTER;

/** 
 * Storage and routines for the memory requirements 
 *
 * A linked list of MEMORY POINTERS is created
 */

class MEMORY_ACCOUNT
{
 public:
  MEMORY_ACCOUNT();
  ~MEMORY_ACCOUNT();
  void* get_memory(int);
  long get_amount()const;
  long get_calls()const;
  
 private:
  /** current allocated amount and number of calls*/
  long amount,calls;
  MEMORY_POINTER *first;
};

#endif
