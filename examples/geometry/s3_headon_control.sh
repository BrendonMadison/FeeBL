#!/bin/bash
# benchmark16 s3 moved +2.55% going from n_z=64 to n_z=128. Is that the
# crossing-angle shear staircase, or just ordinary longitudinal convergence?
#
# Doubling n_z refines TWO things at once: the per-slice shear quantisation
# (2.43 -> 1.21 sigma_x) and the longitudinal integration itself. Attributing
# the shift to the shear without a control would be the same "changed two
# variables at once" mistake that wrecked an earlier probe in this project.
#
# Control: run the SAME n_z ladder with the crossing angle set to zero.
#   * head-on also shifts ~2.55%  -> ordinary convergence, the shear is innocent,
#     and s3's golden is under-resolved for a reason unrelated to the angle.
#   * head-on is flat             -> the 2.55% IS the shear staircase, and the
#     guard for uncrabbed finite-angle runs should refuse, not warn.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/benchmark16.dat

run () {   # n_z  angle  label
  sed "s/{n_x=1024;n_y=512;n_z=64;/{n_x=1024;n_y=512;n_z=$1;/g; \
       s/angle_x.1=0.01500000;angle_x.2=0.01500000/angle_x.1=$2;angle_x.2=$2/g" "$D" > acc.dat
  local gotz gota
  gotz=$(sed -n '/PARAMETERS:: p_s3/,/}/p' acc.dat | grep -o 'n_z=[0-9]*' | head -1 | cut -d= -f2)
  gota=$(sed -n '/ACCELERATOR:: s3/,/}/p' acc.dat | grep -o 'angle_x.1=[0-9.]*' | head -1 | cut -d= -f2)
  if [ "$gotz" != "$1" ]; then echo "  ABORT: n_z is $gotz, wanted $1"; exit 1; fi
  case "$gota" in "$2"|"$2"*) ;; *) echo "  ABORT: angle_x is $gota, wanted $2"; exit 1;; esac
  "$B" s3 p_s3 o.dat >out.txt 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
  local L; L=$(grep -m1 "lumi_ee =" o.dat | sed 's/.*= *//;s/ .*//')
  [ -z "$L" ] && L=$(grep -m1 "lumi_fine =" o.dat | sed 's/.*= *//;s/ .*//')
  echo "$L"
}

echo "  benchmark16 s3, FCC-ee Z -- head-on CONTROL for the n_z ladder"
A=$(run 64  0.00000000)
echo "  head-on  n_z=64    L = $A"
Bv=$(run 128 0.00000000)
echo "  head-on  n_z=128   L = $Bv"
awk -v a="$A" -v b="$Bv" 'BEGIN{printf "\n  head-on shift 64 -> 128 : %+.2f%%\n", 100*(b-a)/a}'
echo "  finite-angle shift (measured earlier) : +2.55%"
echo
echo "  Comparable shifts => ordinary convergence. A flat control => the shear."
