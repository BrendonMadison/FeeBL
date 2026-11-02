#!/bin/bash
# An incoming map must declare what it physically IS, and the spin action
# follows from that -- not from how big the kick happens to be.
#
#   CRAB_LINEAR_IP_TILT  re-describes an already-crabbed bunch AT the IP. No
#                        field is traversed, so the spin the beam carries is
#                        already the at-IP spin and must NOT be rotated. That is
#                        consistent with polar_x/y/z being defined at the IP.
#
#   CRAB_RF_HARMONICS    a real deflecting cavity. Traversing it precesses the
#                        spin by (1 + a*gamma) times the orbit angle for a
#                        magnetic deflector -- a*gamma is 284 at 125 GeV -- and
#                        by a different coefficient for the electric part. FEEBL
#                        is not told the E/B split, so the coefficient is not
#                        derivable, and applying the orbit half alone would be an
#                        orbital map presented as beamline spin transport.
#
#   waist_mode           likewise a field region.
#
# So the two field-region modes are refused when spin is enabled. Both remain
# usable without spin, which is what the crab-waist and cavity-nonlinearity
# studies need.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_mapspin.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
S="crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;"
X="s/angle_x.1=0.00000000/angle_x.1=0.015/g; s/angle_x.2=0.00000000/angle_x.2=0.015/g;"
G="s/n_z=16/n_z=66/g;"

run () {   # map-switches  spin(0|1)
  local spin=""
  [ "$2" = 1 ] && spin="bmt_precession=1;"
  sed -e "$G $X" \
      -e "s|^do_lumi=0;|$1${S}${spin}do_lumi=0;|" \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=0.8;polar_z.2=0.8;/g" \
      "$D" > acc.dat
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  set +e
  "$B" c_nt1 p_c_nt1 o.dat > out.txt 2>&1
  RC=$?
  set -e
}

refuse () {   # label  map  needle
  run "$2" 1
  local ok=1
  [ "$RC" -ne 0 ] || ok=0
  grep -q "$3" out.txt || ok=0
  printf "  %-52s %s\n" "$1" "$([ $ok = 1 ] && echo PASS || echo "FAIL (exit $RC)")"
  [ $ok = 1 ] || { fails=$((fails+1)); tail -3 out.txt | sed 's/^/      /'; }
}

allow () {   # label  map  spin
  run "$2" "$3"
  local ok=1; [ "$RC" -eq 0 ] || ok=0
  printf "  %-52s %s\n" "$1" "$([ $ok = 1 ] && echo PASS || echo "FAIL (exit $RC)")"
  [ $ok = 1 ] || { fails=$((fails+1)); tail -3 out.txt | sed 's/^/      /'; }
}

CRAB1="crab_mode=1;crab_full_crossing_angle=0.030;"
CRAB2="crab_mode=2;crab_frequency=1.3e9;crab_harmonic_n=3;crab_full_crossing_angle=0.030;"
WAIST="crab_mode=1;crab_full_crossing_angle=0.030;waist_mode=1;waist_fraction=1.0;"

echo "=== field-region maps are refused when spin is enabled ==="
refuse "crab_mode=2 (RF cavity) with spin" "$CRAB2" "physical field region"
refuse "waist_mode with spin"              "$WAIST" "physical field region"

echo
echo "=== the IP tilt is a re-description, so it is allowed with spin ==="
allow "crab_mode=1 with spin" "$CRAB1" 1

echo
echo "=== and every map still runs without spin ==="
# A policy that refused these outright would pass the two checks above while
# breaking the crab-waist and cavity-nonlinearity studies they exist for.
allow "crab_mode=1, no spin" "$CRAB1" 0
allow "crab_mode=2, no spin" "$CRAB2" 0
allow "waist_mode, no spin"  "$WAIST" 0

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
