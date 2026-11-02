#!/bin/bash
# The incoming map chain on the LOADED-BEAM path.
#
# FEEBL::apply_incoming_maps sits at the head of beam_displacements_from_data,
# which both beam sources call -- generated bunches and load_beam=3. That the
# loaded path was covered was an untested claim until this ran, and the two
# paths differ in a way that matters: on the loaded path init_grid_phys has
# already run, so delta_z comes from the grid rather than being computed
# locally, which is also what the crab n_z guard reads.
#
# Round trip: store_beam=1 writes beam1.dat/beam2.dat, fed back with -e/-p.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_crabload.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
S="crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;"
ok () { printf "  %-54s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

# 1. produce the beam files
sed "s/^load_beam=0;/load_beam=0;store_beam=1;/" "$D" > acc.dat
set +e; "$B" c_nt1 p_c_nt1 o.dat >out.txt 2>&1; set -e
[ -s beam1.dat ] && [ -s beam2.dat ] && ok 1 "store_beam=1 writes both beam files" \
  || ok 0 "store_beam=1 writes both beam files"

load () {   # extra-parameters -> sets rc
  sed "s/^load_beam=0;/load_beam=3;/; s|^do_lumi=0;|$1 do_lumi=0;|" "$D" > acc.dat
  set +e; "$B" -e beam1.dat -p beam2.dat c_nt1 p_c_nt1 o.dat >out.txt 2>&1; rc=$?; set -e
}

# 2. reload with no crab: must run, and say nothing about maps
load ""
[ "$rc" = 0 ] && ok 1 "load_beam=3 runs with no crab" || ok 0 "load_beam=3 runs with no crab"
grep -q "incoming maps applied" out.txt && ok 0 "and applies no maps" || ok 1 "and applies no maps"

# 3. reload with a crab small enough to resolve: the hook must fire, both beams
load "crab_mode=1;crab_target_tilt=0.0002;crab_full_crossing_angle=0.030;$S"
[ "$rc" = 0 ] && ok 1 "load_beam=3 runs with crab configured" \
               || ok 0 "load_beam=3 runs with crab configured"
n=$(grep -c "incoming maps applied" out.txt || true)
[ "$n" = 2 ] && ok 1 "the hook fires on the loaded path, for both beams" \
              || ok 0 "the hook fires on the loaded path, for both beams"
grep -q "incoming maps applied to beam 1 .* sign=1" out.txt && ok 1 "beam 1 carries sign +1" \
  || ok 0 "beam 1 carries sign +1"
grep -q "incoming maps applied to beam 2 .* sign=-1" out.txt && ok 1 "beam 2 carries sign -1" \
  || ok 0 "beam 2 carries sign -1"

# 4. the n_z guard must fire here too -- delta_z comes from a different source
load "crab_mode=1;crab_full_crossing_angle=0.030;$S"
[ "$rc" = 1 ] && grep -q "cannot resolve the crab tilt" out.txt \
  && ok 1 "the n_z guard fires on the loaded path" \
  || ok 0 "the n_z guard fires on the loaded path"

# 5. and the C5 refusal
load "crab_mode=1;crab_target_tilt=0.0002;crab_full_crossing_angle=0.030;"
[ "$rc" = 1 ] && grep -q "crab_longitudinal_sign" out.txt \
  && ok 1 "the C5 refusal fires on the loaded path" \
  || ok 0 "the C5 refusal fires on the loaded path"

# 6. the reload double-transform hazard
#
# beam_displacements_from_data applies the roll, the crossing angle, the offsets
# and the crab map, and it is called unconditionally on BOTH setup paths. A beam
# stored from a run that already had them carries them in its positions, so a
# reload applies them twice and nothing else says so. FEEBL cannot tell from the
# file -- it is a bare particle list with no header -- so the deck must declare.
load "crab_mode=1;crab_target_tilt=0.0002;crab_full_crossing_angle=0.030;$S"
grep -q "applied TWICE" out.txt && ok 1 "reloading with transforms warns about doubling"                                || ok 0 "reloading with transforms warns about doubling"

load "crab_mode=1;crab_target_tilt=0.0002;crab_full_crossing_angle=0.030;loaded_beam_is_pretransformed=1;$S"
[ "$rc" = 0 ] && ok 1 "declaring the file pre-transformed still runs"               || ok 0 "declaring the file pre-transformed still runs"
grep -q "skipping roll, crossing angle" out.txt   && ok 1 "and the transforms are skipped, not reapplied"   || ok 0 "and the transforms are skipped, not reapplied"
grep -q "applied TWICE" out.txt && ok 0 "and it does not also warn"                                || ok 1 "and it does not also warn"

load ""
grep -q "applied TWICE" out.txt && ok 0 "a plain reload with no transforms is quiet"                                || ok 1 "a plain reload with no transforms is quiet"

echo
if [ "$fails" -eq 0 ]; then echo "  the map chain covers the loaded-beam path"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
