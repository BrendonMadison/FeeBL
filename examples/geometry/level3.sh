#!/bin/bash
# With disruption off, compare luminosity with the analytic rigid-bunch value.
#
# Rigid beams are obtained with charge_sign=0, which zeroes the field. That was
# cross-checked independently against the low-intensity limit: scaling the bunch
# population down by 1e-4 and back up by N^2 gives 7.17158e33 against
# charge_sign=0's 7.16844e33, agreeing to 0.04%.
#
# The analytic reference is geometric overlap times the hourglass factor:
#   L = N1 N2 f / (4 pi sigma_x sigma_y) * R_geom * H
# with R_geom = 1/sqrt(1+Phi^2) and Phi the Piwinski angle. Head-on gives
# R_geom = 1 exactly, so this deck tests the hourglass arm.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/quick.dat

sed 's/charge_sign=-1.0/charge_sign=0/g' "$D" > acc.dat
"$B" c_nt1 p_c_nt1 o.dat >/dev/null 2>&1

val () { grep -m1 "$1 =" o.dat | sed 's/.*= *//;s/ .*//'; }
SX=$(grep -m1 -o "sigma_x  *: *[0-9.e+-]*" o.dat | sed "s/.*: *//")
SY=$(grep -m1 -o "sigma_y *: *[0-9.e+-]*" o.dat | sed "s/.*: *//")
LF=$(val lumi_fine)
# sigma_z and beta_y are printed on the beam lines in micrometers
SZ=$(grep -m1 -o "sigma_z : [0-9.e+-]*" o.dat | sed 's/.*: //')
BY=$(grep -m1 -o "beta_y : [0-9.e+-]*" o.dat | sed 's/.*: //')
NP=$(grep -m1 -o "particles : [0-9.e+-]*" o.dat | sed 's/.*: //')

echo "  from the run:"
printf "    sigma_x %s nm   sigma_y %s nm   sigma_z %s um   beta_y %s um   N %s e10\n" \
  "$SX" "$SY" "$SZ" "$BY" "$NP"

cat > cmp.cc <<'EOF'
#include "collisionGeometryCPP.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv)
{
  const double sx = atof(argv[1]) * 1e-9;      // nm -> m
  const double sy = atof(argv[2]) * 1e-9;
  const double sz = atof(argv[3]) * 1e-6;      // um -> m
  const double by = atof(argv[4]) * 1e-6;
  const double N  = atof(argv[5]);   // already absolute in the output
  const double Lf = atof(argv[6]);
  RigidLuminosity r = rigid_luminosity(N, N, 1.0, sx, sy, sz, by, 0.0);
  printf("\n  analytic rigid reference:\n");
  printf("    geometric (head-on)   %.5e m^-2\n", r.l_headon);
  printf("    Piwinski angle        %.4f   R_geom %.6f\n", r.piwinski, r.r_geom);
  printf("    hourglass H           %.5f\n", r.hourglass);
  printf("    predicted L           %.5e m^-2\n", r.l_total);
  printf("    FEEBL rigid lumi_fine %.5e m^-2\n", Lf);
  const double dev = (Lf - r.l_total) / r.l_total;
  printf("\n    deviation             %+.3f%%\n", 100.0 * dev);
  // The analytic form is itself an approximation (Gaussian, thin-lens
  // hourglass), so agreement to a few percent is the meaningful result here,
  // not agreement to the simulation's statistical noise.
  const bool ok = (dev > -0.05 && dev < 0.05);
  printf("    %s\n", ok ? "PASS: rigid FEEBL reproduces the analytic value within 5%"
                        : "FAIL: rigid FEEBL and the analytic value disagree");
  return ok ? 0 : 1;
}
EOF
g++ -std=c++17 -O2 -I"$ROOT/src" cmp.cc -o cmp
./cmp "$SX" "$SY" "$SZ" "$BY" "$NP" "$LF"
