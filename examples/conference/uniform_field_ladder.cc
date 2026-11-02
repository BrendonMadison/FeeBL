// Diagnostic isolation of the retained standalone oracle. This is not a
// production FFT/operator test and cannot qualify FeeBL machine observables.
#define main retained_stage4_main
#include "../adaptive_grid/test_adaptive_composite_physics_validation.cc"
#undef main

int main() {
  try {
    std::cout << "n,h,solve_gather_relative_l2,exact_potential_gather_relative_l2,solve_only_field_relative_l2,iterations,residual_l2\n";
    const double locations[8][2]={{-2.1,-.45},{-1,.2},{-.25,-1.4},{.8,-.45},{1.5,.7},{2.4,-1.1},{.35,1.8},{-1.8,1.55}};
    for (std::size_t n: {20U,40U,80U,160U,320U}) {
      const double h=16.0/n;
      UniformFineOracle grid(n,n,-8.,-8.,h,h,2U);
      auto boundary=uniform_dirichlet(grid,gaussian_phi);
      auto solved=grid.solve(gaussian_uniform_source(grid),boundary);
      require(solved.second.converged,"uniform solve failed");
      std::vector<double> exact_potential;
      for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j)
        if(!grid.fixed(i,j)) exact_potential.push_back(gaussian_phi(grid.center_x(i),grid.center_y(j)));
      std::vector<double> numerical, exact_gather, oracle;
      for(std::size_t i=0;i<8;++i) {
        const auto probe=particle(3000U+i,locations[i][0],locations[i][1],0.,.35,.35);
        const auto a=grid.gather_tent(probe,solved.first,boundary);
        const auto b=grid.gather_tent(probe,exact_potential,boundary);
        const auto c=gaussian_tent_convolution(probe);
        numerical.insert(numerical.end(),{a.electric_x,a.electric_y});
        exact_gather.insert(exact_gather.end(),{b.electric_x,b.electric_y});
        oracle.insert(oracle.end(),{c.electric_x,c.electric_y});
      }
      std::cout<<std::setprecision(17)<<n<<','<<h<<','
        <<error_norms(numerical,oracle,1e-7).relative_l2<<','
        <<error_norms(exact_gather,oracle,1e-7).relative_l2<<','
        <<error_norms(numerical,exact_gather,1e-7).relative_l2<<','
        <<solved.second.iterations<<','<<solved.second.residual_l2<<std::endl;
    }
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
