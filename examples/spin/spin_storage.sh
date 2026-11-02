#!/bin/bash
# Spin storage is not spin dynamics, and the collision state is sampled on every
# path.
#
# TWO DEFECTS, both silent.
#
# 1. Storage was allocated only when bmt_precession was set, so a run asking
#    only for do_bds_spin_rotation got plain PARTICLE objects: the BDS rotation
#    had nothing to act on, no polarization was reported, and the run exited 0
#    with no diagnostic. Measured before the fix: "NO polarization block".
#
# 2. The COLLISION state was sampled only in FEEBL::beam_interaction. With
#    track_pairs or track_muons the run takes a different path, so the report
#    printed "(not sampled)" without saying why. Measured before the fix:
#    "collision (NOT SAMPLED)" with track_pairs enabled.
#
# Precession remains separately gated on bmt_precession -- GRID::move_particles
# calls rotateBMT only when its own parameter is set -- so a beam can carry and
# report a polarization without the collision precessing it. The third case
# below asserts exactly that separation.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
. "$ROOT/examples/lib/harness.sh"
W="${TMPDIR:-/tmp}/feebl_spinstore.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

run () {   # extra switches
  sed -e "0,/do_lumi=0/{s/do_lumi=0/$1;do_lumi=0/}" \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=0.8;polar_z.2=0.8;/g" \
      "$D" > acc.dat
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  feebl_run "$B" c_nt1 p_c_nt1 o.dat
}

has_report ()  { grep -q "at initial" o.dat; }
coll_sampled () { grep "at collision" o.dat 2>/dev/null | grep -qv "not sampled"; }

echo "=== storage follows the need for a spin, not bmt_precession ==="
run "do_bds_spin_rotation=1"
ok "$(has_report && echo 1 || echo 0)" \
   "do_bds_spin_rotation alone reports a polarization"

echo
echo "=== but precession stays gated on bmt_precession ==="
# With BDS on and BMT off the spin is set at generation and must NOT then be
# precessed by the collision, so initial and final must agree exactly.
INI=$(grep "at initial" o.dat | sed 's/.*= (//; s/ .*//')
FIN=$(grep "at final"   o.dat | sed 's/.*= (//; s/ .*//')
printf "      initial %s   final %s\n" "$INI" "$FIN"
ok "$(awk -v a="$INI" -v b="$FIN" 'BEGIN{d=a-b; if(d<0)d=-d; print (d<1e-9)?1:0}')" \
   "with BMT off the collision does not change the spin"

echo
echo "=== the collision state is sampled on every interaction path ==="
run "bmt_precession=1"
ok "$(coll_sampled && echo 1 || echo 0)" "plain collision path samples it"

run "bmt_precession=1;do_pairs=1;track_pairs=1;store_pairs=1"
ok "$(coll_sampled && echo 1 || echo 0)" "the track_pairs path samples it"

run "bmt_precession=1;do_pairs=1;track_pairs=1;store_pairs=1;do_muons=1;track_muons=1;store_muons=1"
ok "$(coll_sampled && echo 1 || echo 0)" "the track_pairs+muon path samples it"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
