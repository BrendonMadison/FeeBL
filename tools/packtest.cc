// Design question: within the hot tier, should co-accessed fields be separate
// arrays (pure SoA) or packed together (AoSoA-lite)?
//
// The deposit reads x AND y for the same particle. Pure SoA touches two
// independent streams; packing them as a float2 makes it one 8-byte contiguous
// load -- one cache line instead of two, and on GPU one coalesced transaction
// instead of two.
//
// Also tested: what a per-particle weight costs the deposit, since the crab
// report requires positive weights for arbitrary bunches while today's charge
// is per-slice and uniform.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

struct XY { float x, y; };
struct XYW { float x, y, w; };

static const size_t NPART = 2000000;
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
  std::normal_distribution<float> gx(0.0f, nx*0.12f), gy(0.0f, ny*0.12f);

  std::vector<float> sx(NPART), sy(NPART), sw(NPART);
  std::vector<XY>    pxy(NPART);
  std::vector<XYW>   pxyw(NPART);
  for (size_t i = 0; i < NPART; i++) {
    float X = gx(rng), Y = gy(rng);
    sx[i]=X; sy[i]=Y; sw[i]=1.0f;
    pxy[i].x=X; pxy[i].y=Y;
    pxyw[i].x=X; pxyw[i].y=Y; pxyw[i].w=1.0f;
  }

  const size_t ncell = (size_t)(nx+2)*(ny+2);
  printf("%zu particles, grid %dx%d\n\n", NPART, nx, ny);
  printf("  %-30s", "hot-tier packing");
  for (int t = 1; t <= maxth; t *= 2) printf("%9dt", t);
  if (maxth == 10) printf("%9dt", 10);
  printf("\n");

  volatile double sink = 0.0;

#define BENCH(NAME, ...)                                                      \
  { printf("  %-30s", NAME);                                                  \
    for (int nth = 1; nth <= maxth; ) {                                       \
      std::vector<double> grid(ncell * nth, 0.0);                             \
      double t0 = now();                                                      \
      for (int r = 0; r < NREP; r++) {                                        \
        _Pragma("omp parallel num_threads(nth)")                              \
        { int tid = omp_get_thread_num();                                     \
          double* g = &grid[(size_t)tid * ncell];                             \
          _Pragma("omp for schedule(static)")                                 \
          for (long i = 0; i < (long)NPART; i++) { __VA_ARGS__ } }            \
      }                                                                       \
      double el = now() - t0;                                                 \
      printf("%8.2fns", el*1e9/((double)NPART*NREP));                         \
      sink += grid[0];                                                        \
      if (nth == maxth) break;                                                \
      nth = (nth*2 > maxth) ? maxth : nth*2;                                  \
    } printf("\n"); }

  puts("\n-- deposit, uniform weight (today's case) --");
  BENCH("separate x[], y[]  (pure SoA)",
        float X=sx[i], Y=sy[i];
        int i1=(int)(X+nx*0.5f), i2=(int)(Y+ny*0.5f);
        if(i1>=0&&i1<nx-1&&i2>=0&&i2<ny-1) g[(size_t)i1*ny+i2] += 1.0;)

  BENCH("packed xy[] (float2)",
        float X=pxy[i].x, Y=pxy[i].y;
        int i1=(int)(X+nx*0.5f), i2=(int)(Y+ny*0.5f);
        if(i1>=0&&i1<nx-1&&i2>=0&&i2<ny-1) g[(size_t)i1*ny+i2] += 1.0;)

  puts("\n-- deposit, per-particle weight (arbitrary bunches) --");
  BENCH("separate x[], y[], w[]",
        float X=sx[i], Y=sy[i], W=sw[i];
        int i1=(int)(X+nx*0.5f), i2=(int)(Y+ny*0.5f);
        if(i1>=0&&i1<nx-1&&i2>=0&&i2<ny-1) g[(size_t)i1*ny+i2] += W;)

  BENCH("packed xyw[] (float3)",
        float X=pxyw[i].x, Y=pxyw[i].y, W=pxyw[i].w;
        int i1=(int)(X+nx*0.5f), i2=(int)(Y+ny*0.5f);
        if(i1>=0&&i1<nx-1&&i2>=0&&i2<ny-1) g[(size_t)i1*ny+i2] += W;)

  if (sink == 1234.5) printf(" ");
  return 0;
}
