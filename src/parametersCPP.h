#ifndef PARAMETERS_SEEN
#define PARAMETERS_SEEN

#include <iostream>
#include <cstdlib>

#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "readData.h"

#include "option_args.h"

class PARAMETERS
{

  MEMORY_ACCOUNT m_account_;
  DATEI datei_;


  inline int readIntegerValue(std::string name) const
  {
    VALUE value;
    get_named_variable( const_cast<char*>(name.c_str()), &value);
    const double number = CONTENTS(value);
    if (!std::isfinite(number) || number < std::numeric_limits<int>::min() ||
        number > std::numeric_limits<int>::max() || std::trunc(number) != number)
      throw std::invalid_argument("Integer input outside supported range: " + name);
    return static_cast<int>(number);
  }
  inline float readFloatValue(std::string name) const
  {
    VALUE value;
    get_named_variable( const_cast<char*>(name.c_str()), &value);
    const double number = CONTENTS(value);
    if (!std::isfinite(number) || std::fabs(number) > std::numeric_limits<float>::max())
      throw std::invalid_argument("Floating input outside supported range: " + name);
    return static_cast<float>(number);
  }

  inline double readDoubleValue(std::string name) const
  {
    VALUE value;
    get_named_variable( const_cast<char*>(name.c_str()), &value);
    const double number = CONTENTS(value);
    if (!std::isfinite(number))
      throw std::invalid_argument("Non-finite input: " + name);
    return number;
  }

  inline std::string readStringValue(std::string name) const
  {
    VALUE value;
    get_named_variable(const_cast<char*>(name.c_str()), &value);
    if (value.type != T_STRING || value.value.s == NULL)
      {
        std::cerr << "PARAMETERS: " << name << " is not a string" << std::endl;
        exit(1);
      }
    return std::string(value.value.s);
  }


  inline int fileOpen(DATEI *datei,std::string name, std::string type) const
  {
    return  file_open(datei,const_cast<char*>(name.c_str()),const_cast<char*>(type.c_str()));
  }


  inline int fileFindWord(DATEI *datei, std::string word) const
  {
    return  file_find_word(datei,   const_cast<char*>(word.c_str()));
  }

  inline int fileNextWord(DATEI *datei, std::string word) const
  {
    return  file_next_word(datei, const_cast<char*>(word.c_str()));
  }

  inline int fileReadBraces(DATEI *datei,std::string begin, std::string end, char* buff,int n_max) const
  {
    return file_read_braces(datei,const_cast<char*>(begin.c_str()),const_cast<char*>(end.c_str()),buff,n_max);
    //  return file_read_braces(datei,begin, end,buff,n_max);
  }

 public: 

  inline void read_accelerator(char *name) 
  {
    int n=max_buffer2;
    char buffer[max_buffer2];
  
    def_acc(&m_account_);

    do
      {
	do
	  {
	    if (fileFindWord(&datei_,"$ACCELERATOR::")==0) 
	      {
		// No beam state may be constructed without its named input block.
		std::cerr << " FEEBL: ERROR : no $ACCELERATOR:: block named '"
			  << name << "' in the input deck." << std::endl
			  << "   Nothing can be simulated without it, so this is"
			     " fatal rather than a warning." << std::endl;
		exit(1);
	      }
	  }
	while(fileNextWord(&datei_,name)==0);
      }
    while(!fileReadBraces(&datei_,"{","}",buffer,n)) ;
    input_values(buffer, &m_account_);
  }

  inline void read_parameters(char *par ) 
  {
    int n=max_buffer2;
    char buffer[max_buffer2];
  
    do
      {
	do
	  {
	    if (fileFindWord(&datei_,"$PARAMETERS::")==0) 
	      {
		// A missing numerical configuration is fatal before simulation.
		std::cerr << " FEEBL: ERROR : no $PARAMETERS:: block named '"
			  << par << "' in the input deck." << std::endl
			  << "   Nothing can be simulated without it, so this is"
			     " fatal rather than a warning." << std::endl;
		exit(1);
	      }
	  }
	while(file_next_word(&datei_,par)==0);
      }
    while(!fileReadBraces(&datei_,"{","}",buffer,n)) ;
    def_param(&m_account_);
    input_values(buffer, &m_account_);
  }



  PARAMETERS() 
    {
      int testFile =0;
      testFile = fileOpen(&datei_,get_acc_filename(),"r");
  
      if (!testFile) 
	{
	  std::cerr << " PARAMETERS:: error in opening file acc.dat " << std::endl;
	  exit(1);
	}
      // The versioned adaptive-grid contract adds 44 explicit card keys.
      // Keep a bounded heap, but size it above the declared schema rather
      // than failing during parser initialization before any card is read.
      init_named_variable(320, &m_account_);
    }

  ~PARAMETERS() {  file_close(&datei_);}


  void setDoubleValue(std::string name, double d) const
  {
    VALUE value;
    double_to_value(d,&value);
    set_named_variable( const_cast<char*>(name.c_str()), value);  
  }


  inline double readDValue(std::string name) const
  {
    return readDoubleValue(name);
  }

  inline float readFValue(std::string name) const
  {
    return readFloatValue(name);
  }

  inline int readIValue(std::string name) const
  {
    return readIntegerValue(name);
  }

  inline std::string readSValue(std::string name) const
  {
    return readStringValue(name);
  }


};

#endif
