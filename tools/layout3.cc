// Which particle layout wins, and does the answer change with thread count?
//
// Three candidates:
//   A  vector<PARTICLE*> -> 104-byte polymorphic objects   (current)
//   B  SoA float   x[], y[], vx[], vy[], energy[]          (GPU branch wants this)
//   C  SoA double  x[], y[], px[], py[], zeta[], delta[], w[]  (crab report wants this)
//
// Two kernels, because they touch different amounts of state:
//   deposit : reads x,y only          -> A 112 B, B 8 B,  C 16 B per particle
//   push    : reads/writes x,y,vx,vy,E -> A 112 B, B 20 B, C 40 B per particle
//
// The single-thread result is not the interesting one. FEEBL measured at 7% of
// peak DRAM bandwidth single-threaded, so bytes barely matter there; but 7% x 10
// threads is ~70%, which does approach saturation. If doubles cost anything it
// should appear as worse SCALING, not worse single-thread time. That is the
// hypothesis this tests.
//
// One persistent parallel region, not one per call: the point here is sustained
// throughput per layout, not OpenMP launch overhead (already measured at 44% of
// runtime when the region is entered per slice).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "particlesCPP.h"

struct TP : public PARTICLE {
  TP() {}
  TP(float x, float y) { set(250.0f, x, y, 0.0f, 0.0f, 0.0f); }
  // PARTICLE has no public position/velocity setters; expose them here so the
  // AoS push benchmark writes back the same fields the SoA versions do.
  inline void put(float x, float y, float vx, float vy) {
    xpos_ = x; ypos_ = y; vx_ = vx; vy_ = vy;
  }
};

static const size_t NPART = 2000000;   // scaled-up case: the GPU motivation
static const int    NREP  = 12;

static double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
  int nx = (argc > 1) ? atoi(argv[1]) : 64;
  int ny = (argc > 2) ? atoi(argv[2]) : 512;
  int maxth = (argc > 3) ? atoi(argv[3]) : 10;

  std::mt19937 rng(4242);
  std::normal_distribution<float> gx(0.0f, nx * 0.12f), gy(0.0f, ny * 0.12f);

  std::vector<float>  fx(NPART), fy(NPART), fvx(NPART), fvy(NPART), fen(NPART);
  std::vector<double> dx_(NPART), dy_(NPART), dpx(NPART), dpy(NPART),
                      dze(NPART), dde(NPART), dw(NPART);
  std::vector<TP*> ptrs(NPART);

  for (size_t i = 0; i < NPART; i++) {
    float X = gx(rng), Y = gy(rng);
    fx[i]=X; fy[i]=Y; fvx[i]=1e-6f; fvy[i]=1e-6f; fen[i]=250.0f;
    dx_[i]=X; dy_[i]=Y; dpx[i]=1e-6; dpy[i]=1e-6; dze[i]=0.0; dde[i]=0.0; dw[i]=1.0;
    ptrs[i] = new TP(X, Y);
  }

  const size_t ncell = (size_t)(nx + 2) * (ny + 2);
  printf("%zu particles, grid %dx%d\n", NPART, nx, ny);
  printf("  A ptr+object  %6.1f MB   B SoA float %6.1f MB   C SoA double(6D+w) %6.1f MB\n\n",
         (double)(NPART*(sizeof(TP)+8))/1048576.0,
         (double)(NPART*20)/1048576.0,
         (double)(NPART*56)/1048576.0);

  printf("  %-22s", "kernel / layout");
  for (int t = 1; t <= maxth; t *= 2) printf("%9dt", t);
  if (maxth == 10) printf("%9dt", 10);
  printf("     scaling\n");

  volatile double sink = 0.0;

#define BENCH(NAME, BODY)                                                     \
  { printf("  %-22s", NAME); double t1 = 0;                                   \
    for (int nth = 1; nth <= maxth; ) {                                       \
      std::vector<double> grid(ncell * nth, 0.0);                             \
      double t0 = now();                                                      \
      for (int r = 0; r < NREP; r++) {                                        \
        _Pragma("omp parallel num_threads(nth)")                              \
        {                                                                     \
          int tid = 0;                                                        \
          _Pragma("omp critical") { }                                         \
          tid = omp_get_thread_num();                                         \
          double* g = &grid[(size_t)tid * ncell];                             \
          _Pragma("omp for schedule(static)")                                 \
          for (long i = 0; i < (long)NPART; i++) { BODY }                     \
        }                                                                     \
      }                                                                       \
      double el = now() - t0;                                                 \
      if (nth == 1) t1 = el;                                                  \
      printf("%8.1fns", el * 1e9 / ((double)NPART * NREP));                   \
      sink += grid[0];                                                        \
      if (nth == maxth) { printf("   %5.1fx", t1 / el); break; }              \
      nth = (nth * 2 > maxth) ? maxth : nth * 2;                              \
    } printf("\n"); }

  // ---- deposition: reads x,y only -------------------------------------
  puts("\n-- deposit (reads x,y) --");
  BENCH("A ptr -> object",
        float X, Y; ptrs[i]->XYposition(X, Y);
        int i1 = (int)(X + nx*0.5f), i2 = (int)(Y + ny*0.5f);
        if (i1>=0 && i1<nx-1 && i2>=0 && i2<ny-1) g[(size_t)i1*ny+i2] += 1.0;)

  BENCH("B SoA float",
        float X = fx[i], Y = fy[i];
        int i1 = (int)(X + nx*0.5f), i2 = (int)(Y + ny*0.5f);
        if (i1>=0 && i1<nx-1 && i2>=0 && i2<ny-1) g[(size_t)i1*ny+i2] += 1.0;)

  BENCH("C SoA double",
        double X = dx_[i], Y = dy_[i];
        int i1 = (int)(X + nx*0.5), i2 = (int)(Y + ny*0.5);
        if (i1>=0 && i1<nx-1 && i2>=0 && i2<ny-1) g[(size_t)i1*ny+i2] += 1.0;)

  // ---- push: reads and writes x,y,vx,vy,E -------------------------------
  puts("\n-- push (reads+writes x,y,vx,vy,E) --");
  BENCH("A ptr -> object",
        float X, Y, VX, VY; ptrs[i]->XYposition(X, Y); ptrs[i]->velocities(VX, VY);
        float E = ptrs[i]->energy();
        VX += 1e-9f*E; VY += 1e-9f*E; X += VX; Y += VY;
        ptrs[i]->put(X, Y, VX, VY);
        g[0] += 0.0;)

  BENCH("B SoA float",
        float X=fx[i], Y=fy[i], VX=fvx[i], VY=fvy[i], E=fen[i];
        VX += 1e-9f*E; VY += 1e-9f*E; X += VX; Y += VY;
        fx[i]=X; fy[i]=Y; fvx[i]=VX; fvy[i]=VY;
        g[0] += 0.0;)

  BENCH("C SoA double",
        double X=dx_[i], Y=dy_[i], PX=dpx[i], PY=dpy[i], D=dde[i];
        PX += 1e-9*D + 1e-9; PY += 1e-9*D + 1e-9; X += PX; Y += PY;
        dx_[i]=X; dy_[i]=Y; dpx[i]=PX; dpy[i]=PY;
        g[0] += 0.0;)

  if (sink == 1234.5) printf(" ");
  return 0;
}
