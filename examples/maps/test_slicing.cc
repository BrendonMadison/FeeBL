// Does the crossing-angle shear sit on the slice it thinks it does?
//
// The crossing shear uses the slice centre; the crab tilt uses each
// particle coordinate. Check whether the difference is a finite-slice
// approximation or an offset in the longitudinal assignment convention.
//
// This test settles it. Two things have to line up:
//
//   ASSIGNMENT   particleBeamCPP.h dispatch_random_particle_in_slices:
//                    j = floor(z/dz + 0.5*n + 1),  slice = j-1
//                gridCPP.h:418:
//                    slice = floor(z/dz + 0.5*n)
//                These must be the SAME function. The +1 and the -1 cancel, but
//                they cancel in float arithmetic, which is worth asserting
//                rather than reasoning about.
//
//   GEOMETRY     the centre of slice s under that assignment must be
//                    dz*(s - (n-1)/2)
//                which is exactly the offset set_angle_particles applies.
//
// If both hold, the shear sits on the true slice centre, the residual is
// +-dz/2 with ZERO mean, and it vanishes linearly in dz. That refutes the
// half-slice hypothesis and leaves the wobble to be explained by something
// else. The test exists so that a future edit to either formula cannot
// re-open the question silently.
#include <cstdio>
#include <cmath>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

// the two formulas, in the production types
static int slice_beam(float z, float dz, int n)
{ return (int)std::floor(z / dz + 0.5f * n + 1) - 1; }

static int slice_grid(float z, float dz, int n)
{ return (int)std::floor(z / dz + 0.5f * (float)n); }

int main()
{
  const int NS[] = { 16, 32, 53, 64, 80, 160 };
  const float SIGZ = 300000.0f;             // 300 um in nm, the ILC deck
  const float MAXZ = 3.0f * SIGZ;

  // 1. The two assignments are the same function.
  {
    long tested = 0, differ = 0;
    for (int k = 0; k < 6; k++)
      {
        const int n = NS[k];
        const float dz = 2.0f * MAXZ / (float)n;
        for (long i = 0; i <= 200000; i++)
          {
            const float z = -MAXZ + (2.0f * MAXZ) * (float)i / 200000.0f;
            if (slice_beam(z, dz, n) != slice_grid(z, dz, n)) differ++;
            tested++;
          }
      }
    printf("    (%ld positions across six n_z)\n", tested);
    check(tested > 1000000, "the scan is not vacuous");
    check(differ == 0, "beam and grid slice assignment agree everywhere");
  }

  // 2. The slice centre is what set_angle_particles assumes, and the residual
  //    is bounded by half a slice.
  {
    long over = 0, tested = 0;
    for (int k = 0; k < 6; k++)
      {
        const int n = NS[k];
        const float dz = 2.0f * MAXZ / (float)n;
        for (long i = 0; i <= 200000; i++)
          {
            const float z = -MAXZ + (2.0f * MAXZ) * (float)i / 200000.0f;
            const int s = slice_grid(z, dz, n);
            if (s < 0 || s >= n) continue;               // dropped by the guard
            const double zc = (double)dz * ((double)s - 0.5 * (n - 1));
            if (std::fabs(zc - (double)z) > 0.5 * dz * 1.001) over++;
            tested++;
          }
      }
    check(tested > 1000000, "the centre scan is not vacuous");
    check(over == 0, "shear offset is the true slice centre, residual <= dz/2");
  }

  // 3. THE REFUTATION, stated the way it is actually falsifiable.
  //
  //    A half-slice bias means the mean residual is dz/2, i.e. a FRACTION of
  //    0.5 that no refinement removes. So the fraction is the quantity to
  //    bound, not the absolute number.
  //
  //    The absolute mean does not go to zero here, and that is this scan own
  //    artifact rather than the code: sampling z uniformly at 200000 steps
  //    over +-3 sigma_z puts a pitch of 9 nm on the axis, and the residual
  //    floors at half of it, 4.5 nm, for every n_z. Chasing that number would
  //    be measuring the test. The fraction carries the physics, and it comes
  //    out four orders below the half-slice value.
  {
    double worst = 0.0;
    printf("    %8s %14s %14s %12s\n",
           "n_z", "mean residual", "frac of dz", "half-slice");
    for (int k = 0; k < 6; k++)
      {
        const int n = NS[k];
        const float dz = 2.0f * MAXZ / (float)n;
        double sum = 0.0; long cnt = 0;
        for (long i = 0; i <= 200000; i++)
          {
            const float z = -MAXZ + (2.0f * MAXZ) * (float)i / 200000.0f;
            const int sl = slice_grid(z, dz, n);
            if (sl < 0 || sl >= n) continue;
            sum += (double)dz * ((double)sl - 0.5 * (n - 1)) - (double)z;
            cnt++;
          }
        const double mean = (cnt > 0) ? sum / cnt : 0.0;
        const double frac = mean / dz;
        printf("    %8d %14.3e %14.3e %12.1f\n", n, mean, frac, 0.5);
        if (std::fabs(frac) > worst) worst = std::fabs(frac);
      }
    check(worst < 0.01,
          "mean residual is <1% of a slice, not the 50% a bias gives");
    check(worst > 0.0, "and the measurement is live, not identically zero");
  }

  printf("\n  %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
