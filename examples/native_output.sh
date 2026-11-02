#!/bin/bash
# FEEBL native event output: self-describing, dependency-free.
#
# Check named columns, units, signed longitudinal directions and frame
# declarations in the native text records.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_native.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

run () {   # store_native  half_angle  extra
  # polar_z stated explicitly. This used to be left to the parser
  # default of (0,0,1), so the test silently depended on the beams
  # being 100% longitudinally polarized -- exactly the assumption a
  # spin run must not make by accident. FEEBL now refuses a spin run
  # whose deck never mentions polarization.
  sed -e '0,/n_m=50000/{s/n_m=50000/n_m=5000/}' \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=1.0;polar_z.2=1.0;/g" \
      -e "0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;ST_spin_flip=0;store_native=$1;do_lumi=1/}" \
      -e '0,/num_lumi=100000/{s/num_lumi=100000/num_lumi=50/}' \
      -e "s/angle_x.1=0.00000000/angle_x.1=$2/; s/angle_x.2=0.00000000/angle_x.2=$2/" \
      "$D" > acc.dat
  rm -f lumi.ee.out lumi.ee.out.fne
  "$B" c_nt1 p_c_nt1 o.dat > stdout.log 2>&1 || true
}

echo "=== the switch controls it, and defaults to off ==="
run 0 0.0
[ ! -f lumi.ee.out.fne ] && ok 1 "store_native=0 writes no native file" \
                         || ok 0 "store_native=0 writes no native file"
[ -s lumi.ee.out ] && ok 1 "and the legacy file is untouched" \
                   || ok 0 "and the legacy file is untouched"

run 1 0.0
[ -s lumi.ee.out.fne ] && ok 1 "store_native=1 writes one" \
                       || ok 0 "store_native=1 writes one"

echo
echo "=== the header states every convention the numbers depend on ==="
for key in "#FEEBL-NATIVE" "#feebl_version" "#length_unit" "#energy_unit" \
           "#frame lab" "#beam1_direction" "#beam2_direction" \
           "#vz_sign_convention" "#momenta_carry_crossing_angle" \
           "#spin_present" "#BLOCK" "#COLUMNS"; do
  grep -q -- "$key" lumi.ee.out.fne && ok 1 "declares $key" || ok 0 "declares $key"
done

echo
echo "=== columns are named, and match the rows ==="
NC=$(grep '^#COLUMNS' lumi.ee.out.fne | sed 's/^#COLUMNS //' | wc -w)
NF=$(grep -v '^#' lumi.ee.out.fne | head -1 | wc -w)
[ "$NC" = "$NF" ] && ok 1 "column count matches the record width ($NC)" \
                  || ok 0 "column count matches the record width (declared $NC, rows $NF)"

echo
echo "=== vz is present, signed per beam, and consistent with vx,vy ==="
# columns: 1 e1  2 e2  3 x  4 y  5 z  6 t  7 vx1 8 vy1 9 vz1  10 vx2 11 vy2 12 vz2
BAD=$(grep -v '^#' lumi.ee.out.fne | awk '
  { v1 = sqrt(1 - $7*$7 - $8*$8); v2 = -sqrt(1 - $10*$10 - $11*$11);
    if (($9 - v1)^2 > 1e-12 || ($12 - v2)^2 > 1e-12) bad++;
    if ($9 <= 0) s1++; if ($12 >= 0) s2++ }
  END { printf "%d %d %d", bad+0, s1+0, s2+0 }')
set -- $BAD
[ "$1" = 0 ] && ok 1 "vz equals sqrt(1-vx^2-vy^2) on every record" \
             || ok 0 "vz equals sqrt(1-vx^2-vy^2) on every record ($1 bad)"
[ "$2" = 0 ] && ok 1 "beam 1 vz is positive throughout" \
             || ok 0 "beam 1 vz is positive throughout ($2 bad)"
[ "$3" = 0 ] && ok 1 "beam 2 vz is negative throughout" \
             || ok 0 "beam 2 vz is negative throughout ($3 bad)"

echo
echo "=== the record count in the footer is the truth ==="
DECL=$(grep '^#END' lumi.ee.out.fne | awk '{print $NF}')
ACT=$(grep -vc '^#' lumi.ee.out.fne)
[ "$DECL" = "$ACT" ] && ok 1 "footer count matches the rows ($ACT)" \
                     || ok 0 "footer count matches the rows (says $DECL, has $ACT)"

echo
echo "=== a finite-angle run warns that its directions are head-on ==="
run 1 0.015
grep -q "WARNING momenta are in a HEAD-ON frame" lumi.ee.out.fne \
  && ok 1 "the head-on-frame warning is present at 15 mrad" \
  || ok 0 "the head-on-frame warning is present at 15 mrad"
grep -q "#half_crossing_angle_rad 0.015" lumi.ee.out.fne \
  && ok 1 "and the angle itself is recorded" || ok 0 "and the angle itself is recorded"
# The frame KEY must agree with the warning PROSE. These used to disagree:
# the header said "#frame lab" while warning that the momenta were head-on.
grep -q "^#frame head_on" lumi.ee.out.fne \
  && ok 1 "and #frame says head_on, agreeing with the warning" \
  || ok 0 "and #frame says head_on, agreeing with the warning"

run 1 0.0
grep -q "WARNING momenta are in a HEAD-ON frame" lumi.ee.out.fne \
  && ok 0 "a head-on run carries no such warning" \
  || ok 1 "a head-on run carries no such warning"
grep -q "^#frame lab" lumi.ee.out.fne \
  && ok 1 "and a zero-angle run really is the lab frame" \
  || ok 0 "and a zero-angle run really is the lab frame"

echo
if [ "$fails" -eq 0 ]; then echo "  the native output is self-describing and correct"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
