#!/bin/bash
# Stage E, third arm: ROTATION COVARIANCE.
#
# Rotating both beams and the grid together must preserve luminosity.
# This is a symmetry of the physics, so any
# violation beyond discretization is a defect in the deposition, the field
# solver or the FFT.
#
# The grid is axis-aligned and rectangular, so it cannot be rotated by a general
# angle. Three tests follow from that, in increasing generality:
#
#   1. EXACT DISCRETE SYMMETRY. On a SQUARE grid (n_x=n_y, cut_x=cut_y) a
#      rotation by a multiple of pi/2 maps grid cells exactly onto grid cells.
#      Luminosity must be invariant to rounding, not merely to discretization.
#
#   2. TRANSPOSED GRID. The real ILC flat beam rotated by pi/2 with cut_x and
#      cut_y swapped. Exercises the shipped anisotropic geometry.
#
#   3. GENERAL ANGLE. A rotation by pi/4 is NOT a symmetry of a square grid, so
#      CIC deposition leaves a residual. It must converge away under refinement;
#      if it plateaus, it is a defect. Measured: +1.67% -> +0.74% -> +0.15% as
#      delta halves, i.e. first order in delta. Run with --converge.
#
# CALIBRATION NOTES (why the configurations are what they are):
#   * A ROUND beam is a WEAK test: it is rotationally symmetric, so rotating it
#     is physically a no-op and the measurement barely moves (-0.116% even at
#     grid aspect ratio 3.9). It cannot detect much. Rejected.
#   * The real ILC beam has aspect ratio 67, unresolvable on a square grid at
#     n=256 (delta_y/sigma_y = 3.2). Used only for test 2, where the grid is
#     transposed along with it.
#   * So tests 1 and 3 use a MODERATE aspect ratio of 4: beta_y := beta_x and
#     emitt_y := emitt_x/16 gives sigma_y = sigma_x/4 = 128.9 nm, resolved at
#     delta/sigma_y = 0.19 on the square n=256 grid, and genuinely anisotropic
#     so that rotation is a real constraint.
#
# All runs keep the beam-beam fields ON. Covariance of the field solver is the
# property under test; disabling it would test nothing.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_rotcov.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
go () { "$B" c_nt1 p_c_nt1 o.dat >/dev/null 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
        grep -m1 "lumi_fine =" o.dat | sed 's/.*= *//;s/ .*//'; }

# Aspect-4 beam on a square grid. Note every substitution is IN PLACE against
# the value the deck actually assigns: prepending a key does nothing, because a
# later assignment in the same block wins.
aspect4 () {   # n  phi1  phi2
  sed "s/beta_y=0.41/beta_y=13.0/g; s/emitt_y=0.035/emitt_y=0.3125/g; \
       s/cut_y=459.55/cut_y=3092.9/g; \
       s/n_x=256/n_x=$1/g; s/n_y=256/n_y=$1/g; \
       s/angle_phi.1=0.00000000/angle_phi.1=$2/g; \
       s/angle_phi.2=0.00000000/angle_phi.2=$3/g" "$D" > acc.dat
}

cmp_pct () {  # value ref tol label
  awk -v v="$1" -v r="$2" -v t="$3" -v l="$4" 'BEGIN{
    d = 100*(v-r)/r; a = (d<0)?-d:d
    printf "  %-46s %+9.4f%%  (tol %.2f%%)  %s\n", l, d, t, (a<=t)?"PASS":"FAIL"
    exit (a<=t)?0:1 }' || fails=$((fails+1))
}

echo "=== 1. exact discrete symmetry: square grid, multiples of pi/2 ==="
aspect4 256 0.0 0.0; REF=$(go)
for P in 1.57079633 3.14159265 4.71238898; do
  aspect4 256 $P $P; L=$(go)
  cmp_pct "$L" "$REF" 0.02 "phi = $P, both beams + square grid"
done

echo
echo "=== 2. transposed grid: real ILC flat beam, pi/2, cut_x <-> cut_y ==="
sed "s/XXnoopXX/x/" "$D" > acc.dat; FR=$(go)
sed "s/cut_x=3092.9/cut_x=__C__/g; s/cut_y=459.55/cut_y=3092.9/g; \
     s/cut_x=__C__/cut_x=459.55/g; \
     s/angle_phi.1=0.00000000/angle_phi.1=1.57079633/g; \
     s/angle_phi.2=0.00000000/angle_phi.2=1.57079633/g" "$D" > acc.dat
FT=$(go)
cmp_pct "$FT" "$FR" 0.05 "flat beam rotated pi/2 + grid transposed"

echo
echo "=== 3. falsifiers: these MUST move far outside tolerance ==="
big () {  # value ref label   -- passes only if the deviation is LARGE
  awk -v v="$1" -v r="$2" -v l="$3" 'BEGIN{
    d = 100*(v-r)/r; a = (d<0)?-d:d
    printf "  %-46s %+9.3f%%  (need >1%%)   %s\n", l, d, (a>1.0)?"PASS":"FAIL"
    exit (a>1.0)?0:1 }' || fails=$((fails+1))
}
aspect4 256 0.78539816 0.0; L=$(go)
big "$L" "$REF" "only beam 1 rotated pi/4 (aspect-4)"
aspect4 256 1.57079633 0.0; L=$(go)
big "$L" "$REF" "only beam 1 rotated pi/2 (aspect-4)"
sed "s/angle_phi.1=0.00000000/angle_phi.1=1.57079633/g; \
     s/angle_phi.2=0.00000000/angle_phi.2=1.57079633/g" "$D" > acc.dat
L=$(go)
big "$L" "$FR" "flat beam rotated pi/2, grid NOT transposed"
sed "s/cut_x=3092.9/cut_x=__C__/g; s/cut_y=459.55/cut_y=3092.9/g; \
     s/cut_x=__C__/cut_x=459.55/g" "$D" > acc.dat
L=$(go)
big "$L" "$FR" "grid transposed, beams NOT rotated"

if [ "$1" = "--converge" ]; then
  echo
  echo "=== 4. general angle pi/4: must CONVERGE, not plateau ==="
  printf "  %6s %9s %16s %16s %11s\n" "n" "d/sig_y" "phi=0" "phi=pi/4" "deviation"
  prev=""
  for N in 256 512 1024; do
    aspect4 $N 0.0 0.0;               A=$(go)
    aspect4 $N 0.78539816 0.78539816; C=$(go)
    dev=$(awk -v a="$A" -v b="$C" 'BEGIN{d=100*(b-a)/a; printf "%.4f", (d<0)?-d:d}')
    awk -v n="$N" -v a="$A" -v b="$C" -v d="$dev" 'BEGIN{
      printf "  %6d %9.3f %16s %16s %+10.4f%%\n", n, (2*3092.9/n)/128.87, a, b, 100*(b-a)/a}'
    if [ -n "$prev" ]; then
      awk -v p="$prev" -v c="$dev" 'BEGIN{exit (c<p)?0:1}' || \
        { echo "  FAIL: deviation did not shrink on refinement ($prev -> $dev)"; fails=$((fails+1)); }
    fi
    prev=$dev
  done
  awk -v d="$prev" 'BEGIN{
    printf "  %-46s %9.4f%%  (tol 0.30%%)  %s\n", "converged deviation at n=1024", d, (d<=0.30)?"PASS":"FAIL"
    exit (d<=0.30)?0:1 }' || fails=$((fails+1))
fi

echo
if [ "$fails" -eq 0 ]; then echo "  rotation covariance holds"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
