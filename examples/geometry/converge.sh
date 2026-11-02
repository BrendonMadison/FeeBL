#!/bin/bash
# Separate discretization error from the rigid-overlap approximation by
# refining the transverse mesh and longitudinal integration.
#
# The quick deck runs n_y=256 with cut_y=459.55 nm, i.e. delta_y/sigma_y = 0.469
# -- exactly the under-converged value from this project's earlier n_y ladder,
# where refining 0.469 -> 0.234 RAISED the luminosity by about 5%. It also runs
# n_z*n_t = 16 against the ~320 product established as converged.
#
# Under-resolution lowers the computed luminosity, which is the right sign. If
# that is the explanation, the deviation must shrink toward zero as the grid is
# refined. If it plateaus near -2%, the residual is something else and the
# formula needs re-examining.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/quick.dat

# analytic reference, computed once (independent of the grid)
cat > ref.cc <<'EOF'
#include "collisionGeometryCPP.h"
#include <cstdio>
int main(){
  const double sx=515.48e-9, sy=7.65916e-9, sz=300e-6, by=410e-6, N=2e10;
  RigidLuminosity r = rigid_luminosity(N,N,1.0,sx,sy,sz,by,0.0);
  printf("%.10e %.10f %.10e\n", r.l_headon, r.hourglass, r.l_total);
  return 0;
}
EOF
. "$ROOT/examples/lib/harness.sh"
g++ -std=c++17 -O2 -I"$ROOT/src" ref.cc -o ref
read LH HG LT < <(./ref)
printf "  analytic: geometric %.5e  hourglass %.5f  predicted %.5e\n\n" "$LH" "$HG" "$LT"

printf "  %-28s %14s %10s %9s\n" "grid" "rigid lumi_fine" "dy/sigma_y" "deviation"
run () {   # ny  nz  nt
  sed "s/charge_sign=-1.0/charge_sign=0/g; s/n_y=256/n_y=$1/g; s/n_z=16/n_z=$2/g; s/n_t=1;/n_t=$3;/g" \
      "$D" > acc.dat
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  feebl_run "$B" c_nt1 p_c_nt1 o.dat
  local L dy
  L=$(need_num o.dat lumi_fine)
  DEV=$(awk -v L="$L" -v T="$LT" 'BEGIN{printf "%.6f", 100*(L-T)/T}')
  dy=$(awk -v n="$1" 'BEGIN{printf "%.3f", 2*459.55/n/7.65916}')
  awk -v L="$L" -v T="$LT" -v g="n_y=$1 n_z=$2 n_t=$3" -v dy="$dy" \
    'BEGIN{printf "  %-28s %14s %10s %+8.3f%%\n", g, L, dy, 100*(L-T)/T}'
}

fails=0
run 256  16 1; D1=$DEV     # the deck as shipped
run 512  16 1; D2=$DEV     # refine y only
run 1024 16 1; D3=$DEV
run 1024 64 5; D4=$DEV     # and the longitudinal product, 16 -> 320

echo
echo "  If the deviation shrinks monotonically toward zero, the -2.15% was the"
echo "  deck being under-resolved, not an error in the analytic reference."
echo

# ASSERT the stated conclusion instead of inviting the reader to eyeball it.
chk () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
         [ "$1" = 1 ] || fails=$((fails+1)); }

chk "$(awk -v a="$D1" -v b="$D2" -v c="$D3" -v d="$D4" 'BEGIN{
        a=(a<0?-a:a); b=(b<0?-b:b); c=(c<0?-c:c); d=(d<0?-d:d)
        print (b<=a && c<=b && d<=c)?1:0}')" \
    "|deviation| shrinks monotonically under refinement"
chk "$(awk -v d="$D4" 'BEGIN{d=(d<0?-d:d); print (d<1.0)?1:0}')" \
    "the finest grid lands within 1% of the analytic value"
