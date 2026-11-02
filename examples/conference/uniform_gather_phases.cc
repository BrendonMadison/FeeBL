#define main retained_stage4_main
#include "../adaptive_grid/test_adaptive_composite_physics_validation.cc"
#undef main
int main(){
  try{
    std::cout<<"n,phase_x,phase_y,exact_potential_gather_relative_l2\n";
    const double xy[8][2]={{-2.1,-.45},{-1,.2},{-.25,-1.4},{.8,-.45},{1.5,.7},{2.4,-1.1},{.35,1.8},{-1.8,1.55}};
    for(std::size_t n:{80U,160U,320U}){
      double h=16.0/n;UniformFineOracle g(n,n,-8,-8,h,h,2U);
      auto b=uniform_dirichlet(g,gaussian_phi);std::vector<double> p;
      for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)if(!g.fixed(i,j))p.push_back(gaussian_phi(g.center_x(i),g.center_y(j)));
      for(int phase=0;phase<17;++phase){
        double px=phase/17.0,py=((7*phase)%17)/17.0;std::vector<double>a,r;
        for(std::size_t i=0;i<8;++i){auto q=particle(3000+i,xy[i][0]+px*h,xy[i][1]+py*h,0.,.35,.35);auto u=g.gather_tent(q,p,b);auto v=gaussian_tent_convolution(q);a.insert(a.end(),{u.electric_x,u.electric_y});r.insert(r.end(),{v.electric_x,v.electric_y});}
        std::cout<<std::setprecision(17)<<n<<','<<px<<','<<py<<','<<error_norms(a,r,1e-7).relative_l2<<std::endl;
      }
    }return 0;
  }catch(const std::exception&e){std::cerr<<e.what();return 1;}
}
