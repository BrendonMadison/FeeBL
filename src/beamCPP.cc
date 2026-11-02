#include "beamCPP.h"

#include <sstream>
#include <string>

void BEAM::write_size(int first_slice, int last_slice)
 {
   int k;
   std::ostringstream ostr;
   double xrms,yrms;
   double xmin,xmax,xmean,ymin,ymax,ymean;
   for (k=first_slice; k<= last_slice; k++)
     { 
       particle_beam_.transverseRms(k, xmin, xmax, xmean, ymin, ymax, ymean, xrms, yrms);
       ostr << " "<<  k << " "<< xrms << " "<<  yrms <<  " "<< xmin << " "<<  xmax << " "<< xmean << " "<< ymin << " "<< ymax << " "<<  ymean << std::endl;
     }
   ostr << std::endl;
   std::string out = ostr.str();
   size_log_file_->write_line(out);
 }






