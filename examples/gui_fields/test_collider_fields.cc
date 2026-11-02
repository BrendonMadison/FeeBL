// Independent manufactured-potential regression for the actual GUI field
// sampler. No beam generation, FFT planning, RNG, or simulation is run.
#include "colliderFieldsCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << "collider field sampler failure: " << message << '\n';
    std::exit(1);
  }
}

void near(double actual, double expected, const char* message)
{
  if (std::fabs(actual - expected) >
      3e-6 * std::max(1.0, std::fabs(expected))) {
    std::cerr << "actual=" << actual << " expected=" << expected << '\n';
    require(false, message);
  }
}

template<class Callable>
void rejects(Callable action, const char* message)
{
  bool rejected = false;
  try { action(); }
  catch (const std::exception&) { rejected = true; }
  require(rejected, message);
}

class Geometry final : public GENERAL_GRID
{
 public:
  const int nx, ny;
  const float dx, dy;
  Geometry(int x_cells, int y_cells, float x_spacing, float y_spacing)
      : nx(x_cells), ny(y_cells), dx(x_spacing), dy(y_spacing)
  {
    n_cell_x_ = nx; n_cell_y_ = ny; n_cell_z_ = 8;
    delta_x_inverse_ = 1.0f / dx;
    delta_y_inverse_ = 1.0f / dy;
    mesh_ = MESH(dx, dy, 10.0f, 0.5f * nx, 0.5f * ny);
    cut_x_ = 0.5f * nx * dx;
    cut_y_ = 0.5f * ny * dy;
    min_x_ = -(0.5f * nx - 1.0f) * dx;
    max_x_ = -min_x_;
    min_y_ = -(0.5f * ny - 1.0f) * dy;
    max_y_ = -min_y_;
  }
  void distribute_coherent_particles(int, int, int) override {}
  void distribute_particles(int, int, int, int) override {}
  double node_x(int ix) const { return (ix + 0.5 - 0.5 * nx) * dx; }
  double node_y(int iy) const { return (iy + 0.5 - 0.5 * ny) * dy; }
};

template<class Potential>
std::vector<PHI_FLOAT> potential(const Geometry& grid, Potential phi)
{
  std::vector<PHI_FLOAT> result(static_cast<std::size_t>(grid.nx) * grid.ny);
  // Native potential layout is x-major/y-fast, unlike exported raster rows.
  for (int ix = 0; ix < grid.nx; ++ix)
    for (int iy = 0; iy < grid.ny; ++iy)
      result[static_cast<std::size_t>(ix) * grid.ny + iy] =
          phi(grid.node_x(ix), grid.node_y(iy));
  return result;
}

double coordinate(double lo, double hi, std::size_t i, std::size_t n)
{
  return lo + (hi - lo) * static_cast<double>(i) /
                  static_cast<double>(n - 1);
}

void test_affine_units_and_readonly()
{
  const Geometry grid(14, 10, 2.0f, 4.0f);
  auto phi1 = potential(grid, [](double x, double y) { return 6*x + 8*y + 17; });
  auto phi2 = potential(grid, [](double x, double y) { return -10*x + 24*y - 29; });
  const auto before1 = phi1, before2 = phi2;
  const auto sampled = feebl::collider_fields::sample_main_grid(
      grid, phi1.data(), phi2.data(), 64);
  require(sampled.nx == 12 && sampled.ny == 8, "native interior dimensions");
  require(sampled.beam1.size() == sampled.nx * sampled.ny &&
          sampled.beam2.size() == sampled.nx * sampled.ny, "raster dimensions");
  near(sampled.x_min_nm, -11.0, "physical x minimum in nm");
  near(sampled.x_max_nm, 11.0, "physical x maximum in nm");
  near(sampled.y_min_nm, -14.0, "physical y minimum in nm");
  near(sampled.y_max_nm, 14.0, "physical y maximum in nm");
  // E_CIC = (1/2) grad(phi): affine gradients independently imply 5 and 13.
  for (double value : sampled.beam1) near(value, 5.0, "beam1 half-gradient magnitude");
  for (double value : sampled.beam2) near(value, 13.0, "beam2 independent source magnitude");
  const TRIVECTOR components = grid.ElectricFieldCIC(0.0f, 0.0f, phi2.data());
  near(components(0), -5.0, "native signed x component convention");
  near(components(1), 12.0, "native signed y component convention");
  require(phi1 == before1 && phi2 == before2, "sampling must not mutate potentials");
  const auto repeated = feebl::collider_fields::sample_main_grid(
      grid, phi1.data(), phi2.data(), 64);
  require(repeated.beam1 == sampled.beam1 && repeated.beam2 == sampled.beam2,
          "repeated const sampling must be identical");
}

void test_quadratic_orientation_and_stencil()
{
  const Geometry grid(14, 10, 2.0f, 4.0f);
  const auto phi1 = potential(grid, [](double x, double y) {
    return 0.5*x*x + 0.125*y*y + 0.25*x*y + 20*x + 30*y;
  });
  const auto phi2 = potential(grid, [](double x, double y) {
    return 0.25*x*x + 0.375*y*y - 0.125*x*y - 10*x + 18*y;
  });
  const auto sampled = feebl::collider_fields::sample_main_grid(
      grid, phi1.data(), phi2.data(), 6);
  require(sampled.nx == 6 && sampled.ny == 6, "requested sampling cap");
  for (std::size_t iy = 0; iy < sampled.ny; ++iy)
    for (std::size_t ix = 0; ix < sampled.nx; ++ix) {
      const double x = coordinate(sampled.x_min_nm, sampled.x_max_nm, ix, sampled.nx);
      const double y = coordinate(sampled.y_min_nm, sampled.y_max_nm, iy, sampled.ny);
      require(grid.coordinatesInGridRange(static_cast<float>(x), static_cast<float>(y)),
              "each exported coordinate must have a valid native gather stencil");
      // Independent analytic half-gradients; no copied interpolation formula.
      const double e1x = 0.5*x + 0.125*y + 10.0;
      const double e1y = 0.125*y + 0.125*x + 15.0;
      const double e2x = 0.25*x - 0.0625*y - 5.0;
      const double e2y = 0.375*y - 0.0625*x + 9.0;
      const std::size_t at = iy * sampled.nx + ix;
      near(sampled.beam1[at], std::hypot(e1x, e1y), "beam1 raster y-major/x-fast mapping");
      near(sampled.beam2[at], std::hypot(e2x, e2y), "beam2 raster y-major/x-fast mapping");
    }
}

void test_zero_cap_and_rejections()
{
  const Geometry grid(80, 70, 2.0f, 4.0f);
  const auto zero = potential(grid, [](double, double) { return 0.0; });
  const auto sampled = feebl::collider_fields::sample_main_grid(
      grid, zero.data(), zero.data());
  require(sampled.nx == 64 && sampled.ny == 64, "default hard axis bound");
  require(std::all_of(sampled.beam1.begin(), sampled.beam1.end(),
                     [](double value) { return value == 0.0; }) &&
          std::all_of(sampled.beam2.begin(), sampled.beam2.end(),
                     [](double value) { return value == 0.0; }), "zero field remains exact zero");
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, zero.data(), zero.data(), 0); },
          "zero axis cap must be rejected");
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, zero.data(), zero.data(), 1); },
          "singleton axis cap must be rejected");
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, zero.data(), zero.data(), 65); },
          "axis cap larger than hard bound must be rejected");
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, nullptr, zero.data()); },
          "null source potential must be rejected");
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, zero.data(), nullptr); },
          "null second source potential must be rejected");
  auto bad = zero;
  std::fill(bad.begin(), bad.end(), std::numeric_limits<PHI_FLOAT>::quiet_NaN());
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, bad.data(), zero.data()); },
          "nonfinite sampled fields must be rejected");
  std::fill(bad.begin(), bad.end(), std::numeric_limits<PHI_FLOAT>::infinity());
  rejects([&] { feebl::collider_fields::sample_main_grid(grid, zero.data(), bad.data()); },
          "infinite potential must be rejected");
  const Geometry too_small(3, 8, 2.0f, 4.0f);
  const auto small_phi = potential(too_small, [](double, double) { return 0.0; });
  rejects([&] { feebl::collider_fields::sample_main_grid(too_small, small_phi.data(), small_phi.data()); },
          "grid too small for safe interior raster must be rejected");
}

void test_longitudinal_cross_sections()
{
  const Geometry grid(14,10,2.0f,4.0f);
  const auto phi1=potential(grid,[](double x,double y) {
    return 0.5*x*x+0.125*y*y+0.25*x*y+20*x+30*y;
  });
  const auto phi2=potential(grid,[](double x,double y) {
    return 0.25*x*x+0.375*y*y-0.125*x*y-10*x+18*y;
  });
  const auto before1=phi1,before2=phi2;
  const auto lines=feebl::collider_fields::sample_cross_sections(grid,phi1.data(),phi2.data(),6);
  require(lines.beam1_zx.size()==6&&lines.beam1_zy.size()==6&&
          lines.beam2_zx.size()==6&&lines.beam2_zy.size()==6,"cross-section sample counts");
  for(std::size_t i=0;i<6;++i) {
    const double x=coordinate(lines.geometry.x_min_nm,lines.geometry.x_max_nm,i,6);
    const double y=coordinate(lines.geometry.y_min_nm,lines.geometry.y_max_nm,i,6);
    near(lines.beam1_zx[i],std::hypot(0.5*x+10,0.125*x+15),"zx actual y=0 gather");
    near(lines.beam1_zy[i],std::hypot(0.125*y+10,0.125*y+15),"zy actual x=0 gather");
    near(lines.beam2_zx[i],std::hypot(0.25*x-5,-0.0625*x+9),"zx source2 independent gather");
    near(lines.beam2_zy[i],std::hypot(-0.0625*y-5,0.375*y+9),"zy source2 independent gather");
  }
  require(phi1==before1&&phi2==before2,"cross-section sampling mutated potentials");
  const auto zero=potential(grid,[](double,double){return 0.0;});
  const auto zeros=feebl::collider_fields::sample_cross_sections(grid,zero.data(),zero.data(),4);
  for(double v:zeros.beam1_zx)require(v==0,"computed zero cross-section changed");
  for(double v:zeros.beam2_zy)require(v==0,"computed zero second source changed");
  rejects([&]{feebl::collider_fields::sample_cross_sections(grid,nullptr,phi2.data());},"null line potential accepted");
  auto bad=phi1;bad[0]=std::numeric_limits<PHI_FLOAT>::infinity();
  rejects([&]{feebl::collider_fields::sample_cross_sections(grid,bad.data(),phi2.data());},"nonfinite line potential accepted");
  rejects([&]{feebl::collider_fields::sample_cross_sections(grid,phi1.data(),phi2.data(),1);},"singleton line cap accepted");
}

// Use the actual particle storage and native dump transform, including both
// beam signs and the stored slice's two boundaries; do not merely compare two
// copies of the export formula. No particle generator or RNG is needed.
class DumpCoordinateParticle final : public ABSTRACT_PARTICLE {
  TRIDVECTOR spin_;
 public:
  const TRIDVECTOR& getSpin() const override {return spin_;}
};
void test_longitudinal_native_dump_coordinates_and_budget()
{
  const int nz=8;
  const float dz=10.0f,max_z=40.0f;
  for(int d=0;d<=2*nz-2;++d) {
    for(int i1=std::max(0,d-nz+1);i1<=std::min(d,nz-1);++i1) {
      const int i2=d-i1;
      const auto z=feebl::collider_fields::encounter_z(i1,i2,dz);
      near(z.nominal_high_nm-z.nominal_low_nm,dz,"native collision-cell width");
      for(int beam=1;beam<=2;++beam) {
        const int slice=beam==1?i1:i2,sign=beam==1?1:-1;
        const double centre=beam==1?z.beam1_nm:z.beam2_nm;
        double displayed[3];
        for(int position=0;position<3;++position) {
          DumpCoordinateParticle p;
          // The untransformed slice spans [-max_z+i*dz,-max_z+(i+1)*dz].
          p.setZ(-max_z+(slice+0.5f*position)*dz);
          p.set_z_for_dump(d,dz/2,max_z,sign);
          displayed[position]=p.z();
        }
        near(displayed[1],centre,"source band centre vs actual particle dump");
        near(std::min(displayed[0],displayed[2]),centre-z.half_width_nm,"source lower band vs stored boundary dump");
        near(std::max(displayed[0],displayed[2]),centre+z.half_width_nm,"source upper band vs stored boundary dump");
      }
    }
  }
  for(int n=1;n<=128;++n) {
    const std::size_t cap=feebl::collider_fields::longitudinal_axis_cap(n);
    require(cap>=2&&cap<=64,"longitudinal cap range");
    const int last=2*n-2,stride=std::max(1,(last+62)/63);
    std::size_t encounters=0,frames=0;
    for(int d=0;d<=last;++d)if(d==last||d%stride==0) {
      ++frames;encounters+=std::min(d,n-1)-std::max(0,d-n+1)+1;
    }
    require(4*cap*encounters<=524288,"longitudinal scalar budget");
    require(4*cap*encounters+2*64*64*frames<=1048576,"combined field scalar budget");
  }
  require(feebl::collider_fields::longitudinal_axis_cap(64)==64,"64-slice run retains64 line samples");
  require(feebl::collider_fields::longitudinal_axis_cap(128)==32,"128-slice line cap");
  require(feebl::collider_fields::longitudinal_axis_cap(129)==0,"unsupported slice count must be explicit");
  rejects([&]{feebl::collider_fields::encounter_z(-1,0,dz);},"invalid encounter slice accepted");
  rejects([&]{feebl::collider_fields::encounter_z(0,0,0);},"zero encounter width accepted");
}

} // namespace

int main()
{
  test_affine_units_and_readonly();
  test_quadratic_orientation_and_stencil();
  test_zero_cap_and_rejections();
  test_longitudinal_cross_sections();
  test_longitudinal_native_dump_coordinates_and_budget();
  std::cout << "PASS: actual native field sampler; XY/longitudinal analytic gathers, stored-particle dump coordinates, units, read-only and budgets\n";
  return 0;
}
