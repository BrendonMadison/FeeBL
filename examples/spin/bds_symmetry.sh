#!/bin/bash
# The two BDS spin-rotation dispatchers must agree, and neither may emit NaN.
#
# FEEBL generates the beam through fill_beam or fill_symmetric_beam depending on
# `charge_symmetric`, and each carries its own copy of the optional BDS spin
# rotation. They were not the same rotation.
#
#   nonsymmetric:  theta = asin(sqrt(vx^2 + vy^2))
#   symmetric:     theta = asin(vx / cos(atan(vx/vy)))
#
# cos(atan(vx/vy)) = |vy|/sqrt(vx^2+vy^2), so the second evaluates to the first
# scaled by vx/|vy| -- equal only when |vx| = |vy|, three times too large at
# vx = 3*vy, and a hundred times too large at vx = 100*vy. The atan arguments
# were transposed; cos(atan(vy/vx)) reproduces asin(|v|) exactly.
#
# Both then divided the quaternion by sqrt(vx^2+vy^2), which is 0/0 for an
# exactly on-axis particle.
#
# Measured before the fix, charge_symmetric=1 with do_bds_spin_rotation=1:
#     (P.1, P.2) = (+0.9711, +nan)
# and after:
#     (P.1, P.2) = (+0.9999, -0.9999)
# matching the nonsymmetric path. So the defect produced a 2.9% polarization
# error on one beam and destroyed the other outright.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
. "$ROOT/examples/lib/harness.sh"
W="${TMPDIR:-/tmp}/feebl_bdssym.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-54s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

# Returns "P1 P2" from the initial-state polarization line.
run () {   # charge_symmetric
  sed -e "0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;do_bds_spin_rotation=1;charge_symmetric=$1;do_lumi=0/}" \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=1.0;polar_z.2=1.0;/g" \
      "$D" > acc.dat
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  feebl_run "$B" c_nt1 p_c_nt1 o.dat
  local line
  line=$(grep "at initial" o.dat | head -1)
  [ -n "$line" ] || { echo "  FATAL: no polarization line; the run measured nothing" >&2; exit 92; }
  echo "$line" | sed 's/.*= (//; s/).*//' \
               | awk -F' , ' '{p1=$1; p2=$2; gsub(/ .*/,"",p1); gsub(/ .*/,"",p2); print p1, p2}'
}

echo "=== the two beam generators must apply the SAME BDS rotation ==="
read -r A1 A2 <<< "$(run 0)"
read -r S1 S2 <<< "$(run 1)"
printf "  charge_symmetric=0 (fill_beam)           : P.1=%s  P.2=%s\n" "$A1" "$A2"
printf "  charge_symmetric=1 (fill_symmetric_beam) : P.1=%s  P.2=%s\n" "$S1" "$S2"
echo

# NaN first: it is the loudest failure and the one that was actually present.
finite () { awk -v x="$1" 'BEGIN{ exit (x ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) ? 0 : 1 }'; }
for v in "$A1" "$A2" "$S1" "$S2"; do
  if ! finite "$v"; then
    ok 0 "every reported polarization is finite (got '$v')"
    echo "  $fails failure(s)"; exit 1
  fi
done
ok 1 "every reported polarization is finite (no NaN)"

near () { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; print (d<=t)?1:0}'; }

# The two generators draw different particles, so require agreement at the
# 1e-3 level rather than equality. A vx/|vy| scaling of the rotation angle
# cannot hide inside that: it moved P.1 by 2.9%.
ok "$(near "$S1" "$A1" 0.005)" "beam 1 agrees between the two generators"
ok "$(near "$S2" "$A2" 0.005)" "beam 2 agrees between the two generators"

# Non-vacuity: a BDS rotation that did nothing at all would also agree.
ok "$(awk -v a="$A1" 'BEGIN{print (a<1.0)?1:0}')" \
   "and the rotation actually depolarizes (P.1 < 1)"

# Beam 2 travels -z, so its helicity must be opposite in sign.
ok "$(awk -v a="$A1" -v b="$A2" 'BEGIN{print (a*b<0)?1:0}')" \
   "the two beams have opposite helicity, as -z requires"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
