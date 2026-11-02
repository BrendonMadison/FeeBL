#!/bin/bash
# Stage D gate: does crab crossing actually RESTORE luminosity?
#
# This is what separates a wired-up map chain from a working one. A hook can
# compile, stay bit-identical when off, refuse bad configurations and still be
# physically inert -- which is exactly what the RF cavity map is when applied at
# the IP, because a momentum kick only becomes a position tilt after transport
# FEEBL does not have.
#
# THE SIGN IS PER BEAM AND IS NOT OPTIONAL. FEEBL applies the crossing angle as
# a per-slice position shear with an identical formula for both beams
# (PARTICLE_BEAM::set_angle_particles), so both bunches acquire the SAME
# lab-frame tilt. A tilt applied in each beam's own zeta gives OPPOSITE lab
# tilts, since zeta_2 = -zeta_1 at the same lab z. Cancelling both therefore
# needs opposite per-beam tilts, t_b = -sign_b * theta/2, which is what the
# derived default produces.
#
# n_z here is 66, the smallest that satisfies the crab tilt guard at 30 mrad
# (see crab_nz_guard.sh), and recovery lands at 95.9%. Run with --converged for
# n_z=64, n_t=5, which measures 100.0% -- but note that is one point, not a
# rule: at the same longitudinal product n_z=80 gives 95.4% and n_z=160 gives
# 98.8%. The remaining n_z dependence is unresolved. The gate therefore
# asserts a floor, not exact recovery.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_crablumi.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

NZ=66; NT=1; MIN_RECOVERY=85
if [ "$1" = "--converged" ]; then NZ=64; NT=5; MIN_RECOVERY=99; fi

fails=0
go () { "$B" c_nt1 p_c_nt1 o.dat >out.txt 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
        grep -m1 "lumi_fine =" o.dat | sed 's/.*= *//;s/ .*//'; }
S="crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;"
GRID="s/n_z=16/n_z=$NZ/g; s/n_t=1;/n_t=$NT;/g;"
XING="s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g;"

echo "  grid: n_z=$NZ n_t=$NT (product $((NZ*NT)); ~320 is longitudinally converged)"
echo

sed "$GRID" "$D" > acc.dat;         HEADON=$(go)
sed "$GRID $XING" "$D" > acc.dat;   NOCRAB=$(go)
sed "$GRID $XING s|^do_lumi=0;|crab_mode=1;crab_full_crossing_angle=0.030;$S do_lumi=0;|" "$D" > acc.dat
CRAB=$(go)
sed "$GRID $XING s|^do_lumi=0;|crab_mode=1;crab_target_tilt.1=0.015;crab_target_tilt.2=-0.015;crab_full_crossing_angle=0.030;$S do_lumi=0;|" "$D" > acc.dat
REVERSED=$(go)
sed "$GRID $XING s|^do_lumi=0;|crab_mode=1;crab_target_tilt.1=-0.015;crab_target_tilt.2=-0.015;crab_full_crossing_angle=0.030;$S do_lumi=0;|" "$D" > acc.dat
SAMESIGN=$(go)

printf "  %-38s %16s %9s\n" "configuration" "lumi_fine" "of head-on"
row () { awk -v l="$1" -v h="$HEADON" -v n="$2" 'BEGIN{printf "  %-38s %16s %8.1f%%\n", n, l, 100*l/h}'; }
row "$HEADON"   "head-on, no crossing angle"
row "$NOCRAB"   "30 mrad crossing, no crab"
row "$CRAB"     "30 mrad crossing, crab (derived tilt)"
row "$REVERSED" "  ... per-beam signs REVERSED"
row "$SAMESIGN" "  ... same sign on both beams"
echo

awk -v a="$CRAB" -v b="$NOCRAB" 'BEGIN{
  r=a/b; printf "  %-52s x%6.2f  %s\n", "crab recovers at least 5x over the un-crabbed crossing", r, (r>=5.0)?"PASS":"FAIL"
  exit (r>=5.0)?0:1}' || fails=$((fails+1))
awk -v c="$CRAB" -v h="$HEADON" -v m="$MIN_RECOVERY" 'BEGIN{
  r=100*c/h; printf "  %-52s %6.1f%%  %s\n", "and reaches at least " m "% of head-on", r, (r>=m)?"PASS":"FAIL"
  exit (r>=m)?0:1}' || fails=$((fails+1))
# Falsifiers. Measured against the CORRECT choice, not against no-crab: with the
# same local tilt on both beams the lab tilts come out opposite, so one beam is
# cancelled while the other is doubled, and whether that beats no-crab at all
# depends on the grid -- 0.75x at n_z=16 but 1.20x at n_z=66. What IS robust is
# that both wrong choices sit far below the correct one.
f () { awk -v a="$1" -v b="$2" -v l="$3" 'BEGIN{
  r=a/b; printf "  %-52s %7.3f  %s\n", l, r, (r<0.5)?"PASS":"FAIL"; exit (r<0.5)?0:1}' || fails=$((fails+1)); }
f "$REVERSED" "$CRAB" "reversed per-beam signs: fraction of correct crab"
f "$SAMESIGN" "$CRAB" "same tilt on both beams: fraction of correct crab"

echo
if [ "$fails" -eq 0 ]; then echo "  crab crossing restores luminosity"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
