#!/bin/bash
# Enforce the frame and resolution limits of finite-angle simulations.
#
#  * Momenta never carry the crossing angle: set_angle_particles shears
#    positions only and ABSTRACT_PARTICLE has no longitudinal direction. So any
#    angular observable is computed in the head-on frame. That is a WARNING
#    rather than a refusal, because the energy-domain results in the same run
#    are valid and are usually the point of the run.
#  * automatic_grid_sizing measures beam moments AFTER the shear and then adds
#    the sweep again, so cut_x comes out ~2.5x too wide. That is a REFUSAL.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_faguards.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }
XING="s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g;"
go () { set +e; "$B" c_nt1 p_c_nt1 o.dat >out.txt 2>&1; rc=$?; set -e; }

echo "=== the angular-observable warning ==="
sed "$XING" "$D" > acc.dat; go
[ "$rc" = 0 ] && ok 1 "a finite-angle run still runs" || ok 0 "a finite-angle run still runs"
grep -q "Momenta do NOT carry the crossing angle" out.txt \
  && ok 1 "and warns that momenta are in the head-on frame" \
  || ok 0 "and warns that momenta are in the head-on frame"
grep -q "half-angle 0.015" out.txt \
  && ok 1 "reporting the actual half-angle" || ok 0 "reporting the actual half-angle"

# head-on must stay silent, or every existing run grows noise
sed "s/QQ/q/" "$D" > acc.dat; go
grep -q "Momenta do NOT carry" out.txt \
  && ok 0 "head-on runs stay silent" || ok 1 "head-on runs stay silent"

# and a finite-angle run with NO angular observable requested stays silent
sed "$XING s/do_photons=1;/do_photons=0;/; s/^do_lumi=0;/do_lumi=0;/" "$D" > acc.dat; go
grep -q "Momenta do NOT carry" out.txt \
  && ok 0 "finite angle with no angular observable stays silent" \
  || ok 1 "finite angle with no angular observable stays silent"

echo
echo "=== the automatic_grid_sizing refusal ==="
sed "$XING s/^load_beam=0;/automatic_grid_sizing=1;load_beam=0;/" "$D" > acc.dat; go
[ "$rc" = 1 ] && grep -q "automatic_grid_sizing=1 with a non-zero crossing angle" out.txt \
  && ok 1 "refused with a crossing angle" || ok 0 "refused with a crossing angle"

sed "s/^load_beam=0;/automatic_grid_sizing=1;load_beam=0;/" "$D" > acc.dat; go
[ "$rc" = 0 ] && ok 1 "still allowed head-on" || ok 0 "still allowed head-on"

echo
echo "=== the shear-quantisation warning (uncrabbed) ==="
# Uncrabbed, the per-slice shear staircase is a convergence question rather than
# a cancellation failure -- measured 0.55% for a 4x smear reduction on this deck
# -- so it warns and lets the run proceed.
sed "$XING" "$D" > acc.dat; go
[ "$rc" = 0 ] && ok 1 "an under-sliced uncrabbed run still proceeds"                || ok 0 "an under-sliced uncrabbed run still proceeds"
grep -q "shear is quantised at" out.txt   && ok 1 "and warns, with the smear in sigma"   || ok 0 "and warns, with the smear in sigma"
grep -q "n_z >= " out.txt   && ok 1 "naming an n_z that would remove it" || ok 0 "naming an n_z that would remove it"

sed "s/QQ/q/" "$D" > acc.dat; go
grep -q "shear is quantised at" out.txt   && ok 0 "head-on stays silent" || ok 1 "head-on stays silent"

# A crabbed beam is already covered, more strictly, by the refusal in
# apply_incoming_maps; it must not be warned about twice.
sed "$XING s/n_z=16/n_z=66/g; s|^do_lumi=0;|crab_mode=1;crab_full_crossing_angle=0.030;crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1; do_lumi=0;|" "$D" > acc.dat
go
[ "$rc" = 0 ] && ok 1 "a crabbed run within its guard proceeds"                || ok 0 "a crabbed run within its guard proceeds"
grep -q "shear is quantised at" out.txt   && ok 0 "and is not warned about twice" || ok 1 "and is not warned about twice"

echo
if [ "$fails" -eq 0 ]; then echo "  finite-angle scope guards hold"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
