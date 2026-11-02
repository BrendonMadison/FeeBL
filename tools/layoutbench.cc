// Isolate the assignBeamSliceCIC access pattern and measure what the storage
// layout actually costs. Hardware counters are unavailable on this host (no
// perf, no valgrind, no sudo), so this microbenchmark is the substitute: it
// reproduces the real kernel's memory behaviour using the real PARTICLE class
// and varies ONLY the layout.
//
// The real kernel, from gridCPP.cc:
//     theParticles[k]->XYposition(xpart, ypart);        // deref 104-byte heap object
//     if (particleInGrid(...)) sog.assignChargeToCIC(i1,i2,h_x,h_y,1.0);
// and assignChargeToCIC performs four read-modify-writes into rho_, two of
// which sit n_y doubles away from the other two.
//
// Layouts:
//   A  vector<PARTICLE*>, individually new'd, SHUFFLED    (current, fragmented)
//   B  vector<PARTICLE*>, individually new'd, alloc order (current, best case)
//   C  vector<PARTICLE*> into one contiguous arena        (Tier 1: pool)
//   D  SoA float x[], y[]                                 (Tier 2: SoA)
// All four perform identical arithmetic and identical grid writes.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "particlesCPP.h"

// PARTICLE's constructor and set() are protected. Derive to construct them.
// No new data members, so sizeof and layout are identical to PARTICLE.
struct TP : public PARTICLE {
  TP() {}
  TP(float x, float y) { set(250.0f, x, y, 0.0f, 0.0f, 0.0f); }
};

typedef double PHI;
static const int    NREP  = 40;
static const size_t NPART = 200000;   // n_m in the production decks

struct Grid {
  int nx, ny;
  std::vector<PHI> rho;
  Grid(int x, int y) : nx(x), ny(y), rho((size_t)(x + 2) * (y + 2), 0.0) {}
  inline void cic(int i1, int i2, float hx, float hy, PHI q) {
    int j = i1 * ny + i2;
    rho[j]      += (1.0 - hx) * (1.0 - hy) * q;
    rho[j + ny] += hx * (1.0 - hy) * q;
    rho[j + 1]  += (1.0 - hx) * hy * q;
    rho[j + ny + 1] += hx * hy * q;
  }
};

static inline bool inGrid(float x, float y, float& hx, float& hy,
                          int& i1, int& i2, int nx, int ny, float dx, float dy)
{
  float fx = (x / dx) + nx * 0.5f, fy = (y / dy) + ny * 0.5f;
  i1 = (int)fx; i2 = (int)fy;
  if (i1 < 0 || i1 >= nx - 1 || i2 < 0 || i2 >= ny - 1) return false;
  hx = fx - i1; hy = fy - i2;
  return true;
}

static double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
  int nx = (argc > 1) ? atoi(argv[1]) : 128;
  int ny = (argc > 2) ? atoi(argv[2]) : 256;
  float dx = 1.0f, dy = 1.0f;

  std::mt19937 rng(12345);
  std::normal_distribution<float> gx(0.0f, nx * 0.12f), gy(0.0f, ny * 0.12f);
  std::vector<float> px(NPART), py(NPART);
  for (size_t i = 0; i < NPART; i++) { px[i] = gx(rng); py[i] = gy(rng); }

  // A/B: individually heap-allocated, exactly as particleBeamCPP.h does
  std::vector<PARTICLE*> ptrs(NPART);
  for (size_t i = 0; i < NPART; i++) ptrs[i] = new TP(px[i], py[i]);
  std::vector<PARTICLE*> shuffled = ptrs;
  std::shuffle(shuffled.begin(), shuffled.end(), rng);

  // C: one contiguous arena
  TP* arena = (TP*)::operator new(sizeof(TP) * NPART);
  std::vector<PARTICLE*> apt(NPART);
  for (size_t i = 0; i < NPART; i++) {
    new (&arena[i]) TP(px[i], py[i]);
    apt[i] = &arena[i];
  }

  printf("grid %dx%d, %zu particles, %d reps\n", nx, ny, NPART, NREP);
  printf("  sizeof(PARTICLE)=%zu  rho=%.1f MB  particles=%.1f MB\n\n",
         sizeof(PARTICLE),
         (double)((nx + 2) * (ny + 2) * sizeof(PHI)) / 1048576.0,
         (double)(NPART * sizeof(TP)) / 1048576.0);

  double t; float hx, hy; int i1, i2; volatile double sink = 0;

#define RUN(NAME, ACCESS)                                                \
  { Grid g(nx, ny); t = now();                                           \
    for (int r = 0; r < NREP; r++)                                       \
      for (size_t k = 0; k < NPART; k++) {                               \
        float X, Y; ACCESS;                                              \
        if (inGrid(X, Y, hx, hy, i1, i2, nx, ny, dx, dy))                \
          g.cic(i1, i2, hx, hy, 1.0);                                    \
      }                                                                  \
    double el = now() - t; sink += g.rho[0];                             \
    printf("  %-36s %7.3f s   %6.2f ns/particle\n", NAME, el,            \
           el * 1e9 / ((double)NPART * NREP)); }

  RUN("A  ptr vector, shuffled heap",    shuffled[k]->XYposition(X, Y))
  RUN("B  ptr vector, alloc order",      ptrs[k]->XYposition(X, Y))
  RUN("C  ptr vector, contiguous arena", apt[k]->XYposition(X, Y))
  RUN("D  SoA float arrays",             X = px[k]; Y = py[k])

  if (sink == 12345.0) printf(" ");
  return 0;
}
