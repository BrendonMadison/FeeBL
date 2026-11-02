// This translation unit intentionally uses the production fast-math contract,
// unlike the strict audit test driver. It executes the actual native CIC helper.
#include "gridCPP.h"
#include <vector>
void mesh_coverage_production_cic(std::size_t count,float hx,float hy,
    int macroparticles,std::vector<PHI_FLOAT>& values,double& macro_charge) {
  SLICE_ON_GRID source;source.resize(4,4);source.set_macroparticles(macroparticles);
  source.update_nb_part_per_macro(2.0e10f);
  for(std::size_t i=0;i<count;++i)source.assignChargeToCIC(1,1,hx,hy,1.0f);
  macro_charge=source.get_nb_part_per_macro();
  values.assign(source.get_rho(),source.get_rho()+16);
}
