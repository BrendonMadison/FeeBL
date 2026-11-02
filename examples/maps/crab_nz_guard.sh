#!/bin/bash
# The crab n_z guard.
#
# FEEBL applies the crossing angle as a per-SLICE position shear while the crab
# tilt is applied per PARTICLE, so the two cancel only down to the slice scale.
# Under-slicing therefore produces a luminosity that is tens of percent low and
# looks entirely plausible -- and optimize_grids cannot catch it, because it
# sizes the grid from the UN-crabbed crossing-angle sweep.
#
# Calibration -- crab recovery against a matched head-on baseline at 30 mrad:
#     n_z      16     32     53     64     80    160
#     smear  3.27   1.64   0.99   0.82   0.65   0.33   (sigma_x)
#     n_t=1  68.4%  88.3%  92.3%  99.1%  95.5%     --
#     p=320     --     --     --  100.0%  95.4%  98.8%  (n_t = 320/n_z)
#
# The criterion bites hard at large smear -- 3.27 sigma costs a third of the
# luminosity -- and that is the footgun the refusal exists to stop. Below about
# 1 sigma it stops being predictive: recovery is 92-100% with structure that
# tracks n_z and NOT the longitudinal product (n_z=80 gives 95.5% at n_t=1 and
# 95.4% at n_t=4). The guard is a floor, not a guarantee of accuracy.
#
# Refusals exit before tracking, so they are cheap. Acceptance means a full run,
# so the accepting cases here use a small tilt deliberately.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_nzguard.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
S="crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;"

# n_z  full_angle  half_angle  tilt-or-empty  -> sets rc, req
run () {
  local extra="crab_mode=1;crab_full_crossing_angle=$2;$S"
  [ -n "$4" ] && extra="crab_mode=1;crab_target_tilt=$4;crab_full_crossing_angle=$2;$S"
  sed "s/n_z=16/n_z=$1/g; \
       s/angle_x.1=0.00000000/angle_x.1=$3/g; s/angle_x.2=0.00000000/angle_x.2=$3/g; \
       s|^do_lumi=0;|$extra do_lumi=0;|" "$D" > acc.dat
  set +e; "$B" c_nt1 p_c_nt1 o.dat >out.txt 2>&1; rc=$?; set -e
  # an accepting run prints no such line; without the guard, set -e would
  # kill the script on the failing grep
  req=$(grep -o "use n_z >= [0-9]*" out.txt | head -1 | grep -o "[0-9]*$" || true)
}

refuses () {   # n_z angle half label expected_req
  run "$1" "$2" "$3" ""
  local ok=1
  [ "$rc" = "1" ] || ok=0
  [ -n "$5" ] && [ "$req" != "$5" ] && ok=0
  printf "  %-52s %s\n" "$4" "$([ $ok = 1 ] && echo PASS || echo FAIL)"
  [ $ok = 1 ] || { fails=$((fails+1)); sed 's/^/      /' out.txt | head -4; }
}

accepts () {   # n_z angle half tilt label
  run "$1" "$2" "$3" "$4"
  local ok=1; [ "$rc" = "0" ] || ok=0
  printf "  %-52s %s\n" "$5" "$([ $ok = 1 ] && echo PASS || echo FAIL)"
  [ $ok = 1 ] || { fails=$((fails+1)); sed 's/^/      /' out.txt | head -4; }
}

echo "=== fires when the slicing cannot resolve the tilt ==="
refuses 16 0.030 0.015 "n_z=16 at 30 mrad refused, asks for 66" 66
refuses 32 0.030 0.015 "n_z=32 at 30 mrad refused, asks for 66" 66
refuses 65 0.030 0.015 "n_z=65 refused: exact lower edge of the boundary" 66

echo
echo "=== the requirement scales with the tilt, not with n_z alone ==="
refuses 16 0.015 0.0075 "half the crossing angle asks for half the n_z (33)" 33
refuses 16 0.060 0.030 "double the crossing angle asks for double (131)" 131

echo
echo "=== does not fire when the tilt is small enough to resolve ==="
accepts 16 0.030 0.015 0.0002 "small explicit tilt runs at n_z=16"

echo
echo "=== never fires with no crab configured ==="
sed "s/QQ/q/" "$D" > acc.dat
set +e; "$B" c_nt1 p_c_nt1 o.dat >out.txt 2>&1; rc=$?; set -e
printf "  %-52s %s\n" "crab off, n_z=16 runs" "$([ "$rc" = 0 ] && echo PASS || echo FAIL)"
[ "$rc" = 0 ] || fails=$((fails+1))
if grep -q "cannot resolve" out.txt; then
  printf "  %-52s %s\n" "and prints no crab diagnostics" "FAIL"; fails=$((fails+1))
else
  printf "  %-52s %s\n" "and prints no crab diagnostics" "PASS"
fi

echo
if [ "$fails" -eq 0 ]; then echo "  the crab n_z guard holds"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
