#ifndef ABSTRACTIOCLASS_SEEN
#define ABSTRACTIOCLASS_SEEN

#include <sstream>
#include <string>

class  ABSTRACT_IO_CLASS
{

 protected :
  inline std::string title(std::string str) const 
   
{
  std::ostringstream out;
  
  out << "                                         " << std::endl;
  out << " *********************************************************************** " << std::endl;
  out << " ------------- " << str << " ------------ " << std::endl;
  out << "             " << std::endl;
  return out.str();
}

 public :
   
 ABSTRACT_IO_CLASS() {;}
 virtual  ~ABSTRACT_IO_CLASS() {;}
 
 virtual std::string output_flow() const =0;
 virtual std::string persistent_flow() const {return "";}

};


#endif
