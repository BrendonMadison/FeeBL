#!/bin/bash
# Negative checks: a run that cannot do what it was asked must FAIL, loudly and
# with a non-zero status.
#
# Batch systems must receive a nonzero exit status for unsupported modes,
# invalid configurations and missing runtime dependencies.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_negative.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0

# run the deck produced by a sed expression; sets rc and leaves out.txt
run () {
  sed "$1" "$D" > acc.dat
  set +e
  "$B" c_nt1 p_c_nt1 o.dat > out.txt 2>&1
  rc=$?
  set -e
}

# a refusal must be non-zero AND say why
refuses () {   # sed  needle  label
  run "$1"
  local ok=1
  [ "$rc" -ne 0 ] || ok=0
  grep -qi "$2" out.txt || ok=0
  printf "  %-56s %s\n" "$3" "$([ $ok = 1 ] && echo PASS || echo "FAIL (exit $rc)")"
  if [ $ok = 0 ]; then fails=$((fails+1)); sed 's/^/      /' out.txt | tail -3; fi
}

# THE CONVERSE MATTERS AS MUCH. A refusal test that only ever checks refusals
# would pass against a binary that refuses everything, so assert that the way
# out actually works.
accepts () {   # sed  label
  run "$1"
  if [ "$rc" -eq 0 ]; then
    printf "  %-56s PASS\n" "$2"
  else
    printf "  %-56s FAIL (exit %s)\n" "$2" "$rc"
    fails=$((fails+1)); tail -3 out.txt | sed "s/^/      /"
  fi
}

echo "=== unimplemented modes must fail, not report success ==="
refuses "s/^load_beam=0;/load_beam=1;/" "not yet implemented" "load_beam=1 exits non-zero"
refuses "s/^load_beam=0;/load_beam=2;/" "not yet implemented" "load_beam=2 exits non-zero"
echo
echo
echo "=== refusals that used to exit(0) must now exit non-zero ==="
# Exercise configuration errors that occur after parsing.
refuses "s/store_pairs=0/store_pairs=1/" "store_pairs" \
        "store_pairs without track_pairs exits non-zero"
refuses "s/do_muons=0/do_muons=0;track_muons=1/" "muons" \
        "track_muons without do_muons exits non-zero"
refuses "s/dist_z=0;/dist_z=0;dist_x=9;/" "dist_x other than 0" \
        "an unknown dist_x exits non-zero"

echo
echo "=== scale_step scales the drift but not the kick ==="
# PARTICLE::advanceDueToEBfield calls advanceVelocities with step_ and
# advancePosition with step_*scale_step, so the momentum kick and the drift use
# different lengths. rotateBMT receives step_, agreeing with the kick and not
# with the drift. A nonunit scale is refused for spin tracking because its
# physical interpretation is unresolved; orbit-only use emits a warning.
refuses "s/grids=1;/grids=1;scale_step=2.0;/g;0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;polar_z.1=0.8;do_lumi=0/}" \
        "scale_step != 1 together with spin" \
        "scale_step != 1 with spin is refused"
accepts "s/grids=1;/grids=1;scale_step=2.0;/g" \
        "and is allowed, with a warning, without spin"
accepts "s/grids=1;/grids=1;scale_step=1.0;/g" \
        "scale_step = 1 is unaffected"

echo "=== charge_sign: only the supported values run ==="
# charge_sign is the RELATIVE sign of the two beam charges (switchesCPP.h:77),
# not a species charge. It reaches the physics twice -- once through the field
# normalisation at gridCPP.cc:149 and once in the spin routines -- so for e-e-
# the orbit force reverses while the spin precession does not. Measured: e+e-
# pinches to 1.29387e34 and e-e- defocuses to 1.79356e33 against a force-free
# 7.16844e33, yet the spin rotation is bit-identical between the two.
#
# e-e- is therefore refused for now rather than producing a self-inconsistent
# run, and any value outside {-1, 0, +1} is refused because it would be applied
# twice and scale the precession by its square. See CHARGE_SIGN_DERIVATION.md.
refuses "s/charge_sign=-1.0/charge_sign=1.0/g" "not supported in this version" \
        "e-e- is refused, and says it is planned"
refuses "s/charge_sign=-1.0/charge_sign=2.5/g" "not a supported value" \
        "an arbitrary charge_sign is refused"
refuses "s/charge_sign=-1.0/charge_sign=-3.0/g" "not a supported value" \
        "a negative non-unit charge_sign is refused"

# The converse: the two supported values must still run. A guard that refused
# everything would pass the three checks above.
accepts "s/XXnoopXX/x/"                              "e+e- (charge_sign=-1) still runs"
accepts "s/charge_sign=-1.0/charge_sign=0/g"         "charge_sign=0 (no force) still runs"

echo "=== a spin run must state its polarization, not inherit (0,0,1) ==="
# readData.cc registers polar_x=0, polar_y=0, polar_z=1.0. Enabling
# bmt_precession without saying anything therefore gave a 100% longitudinally
# polarized beam, silently -- a physics choice made by omission.
#
# The default value itself is unchanged, because it is echoed to the protokoll
# on every run including spin-off runs, and changing it would break the
# bit-identity check against v0.5.0. The trap case is refused instead.
SPIN_ON="0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;do_lumi=0/}"
POLAR="s/waist_y=0;/waist_y=0;polar_z.1=1.0;polar_z.2=1.0;/g"
refuses "$SPIN_ON" "never mentions polarization" \
        "spin enabled with no polar_x/y/z is refused"


accepts "$SPIN_ON; $POLAR" "and stating polar_z explicitly is accepted"
accepts "s/XXnoopXX/x/"   "a deck with spin OFF is untouched by the guard"

echo
echo "=== configuration refusals still fire, and still fail ==="
S="crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;"
refuses "s|^do_lumi=0;|crab_mode=1;crab_frequency=400e6;crab_full_crossing_angle=0.030; do_lumi=0;|" \
        "longitudinal" "crab without a declared longitudinal sign (C5)"
refuses "s|^do_lumi=0;|crab_mode=2;crab_full_crossing_angle=0.030;${S} do_lumi=0;|" \
        "frequency" "crab_mode=2 without a frequency"
refuses "s|^do_lumi=0;|crab_mode=1;crab_frequency=400e6;crab_full_crossing_angle=0.030;crab_include_panofsky_wenzel=0;${S} do_lumi=0;|" \
        "canonical" "a deliberately non-canonical map"
refuses "s|^do_lumi=0;|waist_mode=1;${S} do_lumi=0;|" \
        "crossing angle" "crab waist with no crossing angle"
refuses "s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g; s|^do_lumi=0;|crab_mode=1;crab_full_crossing_angle=0.030;${S} do_lumi=0;|" \
        "cannot resolve" "a crab tilt the slicing cannot resolve"
refuses "s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g; s/^load_beam=0;/automatic_grid_sizing=1;load_beam=0;/" \
        "automatic_grid_sizing" "automatic_grid_sizing with a crossing angle"

refuses "s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g; s/n_z=16/n_z=66/g; s|^do_lumi=0;|crab_mode=1;crab_full_crossing_angle=0.020;${S} do_lumi=0;|"         "disagrees with the beams" "crab_full_crossing_angle that contradicts the beam angles"
refuses "s|^do_lumi=0;|crab_mode=3;crab_frequency=400e6;crab_full_crossing_angle=0.030;${S} do_lumi=0;|"         "crab_mode must be" "an out-of-range crab_mode"
refuses "s|^do_lumi=0;|crab_mode=1;crab_plane=2;crab_frequency=400e6;crab_full_crossing_angle=0.030;${S} do_lumi=0;|"         "crab_plane must be" "an out-of-range crab_plane"

echo
echo "=== and a healthy deck still succeeds ==="
run "s/QQnothingQQ/x/"
printf "  %-56s %s\n" "the unmodified quick deck exits zero" \
       "$([ "$rc" = 0 ] && echo PASS || echo "FAIL (exit $rc)")"
[ "$rc" = 0 ] || fails=$((fails+1))

echo
echo "=== the harness itself must not turn a dead binary into a pass ==="
# With libfftw3 undiscoverable, all three quick-suite cases
# failed to start, no result files were written, and the summary loop -- which
# skipped missing results -- printed 0/3 while the wrapper reported exit 0.
# Feed run.sh a binary that always dies and check it now says so.
STUB="$W/always_fails"
printf '#!/bin/sh
exit 127
' > "$STUB"; chmod +x "$STUB"
set +e
HOUT=$(bash "$ROOT/examples/run.sh" quick -b "$STUB" 2>&1); hrc=$?
set -e
hok=1
[ "$hrc" -ne 0 ] || hok=0
echo "$HOUT" | grep -qE "NO RESULT|0/3|FAIL" || hok=0
echo "$HOUT" | grep -q "3/3 passed" && hok=0
printf "  %-56s %s
" "a binary that cannot run is reported as failure"        "$([ $hok = 1 ] && echo PASS || echo "FAIL (exit $hrc)")"
if [ $hok = 0 ]; then fails=$((fails+1)); echo "$HOUT" | tail -5 | sed 's/^/      /'; fi

echo
if [ "$fails" -eq 0 ]; then echo "  every refusal reports failure"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
