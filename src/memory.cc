#include "memory.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>

/**********************************************************************/
/* memory_account routines                                            */
/**********************************************************************/

MEMORY_ACCOUNT::MEMORY_ACCOUNT():amount(0),calls(0),first(NULL)
{
}

void* MEMORY_ACCOUNT::get_memory(int size)
{
  MEMORY_POINTER *temp;
  //  printf(" get_memory : requested ed size= %d\n", size);
  //  printf(" get_memory : account size= %d\n", amount);
  calls++;
  amount += size;
  temp=(MEMORY_POINTER*)malloc(sizeof(MEMORY_POINTER));
  if (temp==NULL){
      fprintf(stderr,"not enough memory\n");
      fprintf(stderr,"%d bytes required\n",size);
      fflush(stderr);
      exit(10);
  }
  temp->next=first;
  temp->memory=malloc(size);
  if(temp->memory==NULL){
      fprintf(stderr,"not enough memory\n");
      fprintf(stderr,"%d bytes required\n",size);
      fflush(stderr);
      exit(10);
  }
  first=temp;
  return temp->memory;
}

MEMORY_ACCOUNT::~MEMORY_ACCOUNT()
{
  MEMORY_POINTER *temp=first;
  while(temp!=NULL)
    {
      first=temp->next;
      free(temp->memory);
      free(temp);
      temp=first;
    }
  amount=0;
  calls=0;
}

long MEMORY_ACCOUNT::get_amount()const
{
  return amount;
}

long MEMORY_ACCOUNT::get_calls()const
{
  return calls;
}
