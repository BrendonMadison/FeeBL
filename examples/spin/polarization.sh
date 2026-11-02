#!/bin/bash
# Longitudinal bunch polarization, checked against cases with an exact answer.
#
# P is HELICITY, <s.p_hat>: spin projected on each particle's own direction of
# motion. Beam 2 travels toward -z, so a LAB spin along +z gives it NEGATIVE
# helicity. That sign is the whole reason this test exists -- projecting both
# beams on +z would produce a plausible number with beam 2 inverted, which is
# the failure mode this project keeps finding.
#
# NOTE: polar_z defaults to 1.0 (readData.cc:1002), so a deck that enables BMT
# without naming a polarization gets FULLY longitudinally polarised beams, and
# setting only polar_x gives (1,0,1) renormalised to (0.707,0,0.707). Every case
# below therefore sets all three components explicitly.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_pol.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

# Spin transport OFF for the pure-geometry cases: with no precession the initial
# polarization must come back EXACTLY, so any deviation is a bug in the
# projection rather than physics. BMT is on because spin is only stored then.
run () {   # polar-assignments  [extra]
  sed -e '0,/n_m=50000/{s/n_m=50000/n_m=5000/}' \
      -e "0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;ST_spin_flip=0;do_lumi=0/}" \
      -e "s/angle_x.1=0.00000000/$1;angle_x.1=0.00000000/" \
      "$D" > acc.dat
  "$B" c_nt1 p_c_nt1 o.dat > stdout.log 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
}

# field of the "initial" line: 1 -> P.1, 2 -> P.2, lab1, lab2
grab () {   # which
  local line
  line=$(grep "at initial" o.dat | head -1)
  case "$1" in
    1)    echo "$line" | sed 's/.*= (\([^ ]*\).*/\1/' ;;
    2)    echo "$line" | sed 's/.*, \([-+][0-9.]*\) +-.*/\1/' ;;
    lab1) echo "$line" | awk -F'zhat>: ' '{print $2}' | awk -F' , ' '{print $1}' ;;
    lab2) echo "$line" | awk -F'zhat>: ' '{print $2}' | awk -F' , ' '{print $2}' | tr -d ']' ;;
  esac
}

echo "=== a beam fully aligned with its own motion has helicity +1 ==="
run "polar_x.1=0;polar_y.1=0;polar_z.1=1.0;polar_x.2=0;polar_y.2=0;polar_z.2=0.0"
ok "$([ "$(grab 1)" = "+1.0000" ] && echo 1 || echo 0)" \
   "beam 1, lab spin +z  -> P.1 = +1.0000  (got $(grab 1))"

echo
echo "=== beam 2 travels toward -z, so the SAME lab spin flips its helicity ==="
run "polar_x.1=0;polar_y.1=0;polar_z.1=0.0;polar_x.2=0;polar_y.2=0;polar_z.2=1.0"
ok "$([ "$(grab 2)" = "-1.0000" ] && echo 1 || echo 0)" \
   "beam 2, lab spin +z  -> P.2 = -1.0000  (got $(grab 2))"
ok "$([ "$(grab lab2)" = "+1.0000" ] && echo 1 || echo 0)" \
   "and its lab projection is still +1.0000  (got $(grab lab2))"

run "polar_x.1=0;polar_y.1=0;polar_z.1=0.0;polar_x.2=0;polar_y.2=0;polar_z.2=-1.0"
ok "$([ "$(grab 2)" = "+1.0000" ] && echo 1 || echo 0)" \
   "beam 2, lab spin -z  -> P.2 = +1.0000  (got $(grab 2))"

echo
echo "=== a purely transverse spin has zero helicity ==="
run "polar_x.1=1.0;polar_y.1=0;polar_z.1=0;polar_x.2=1.0;polar_y.2=0;polar_z.2=0"
P1=$(grab 1); P2=$(grab 2)
ok "$(awk -v a="$P1" 'BEGIN{print (a<0.001 && a>-0.001)?1:0}')" \
   "beam 1, lab spin +x  -> P.1 = 0        (got $P1)"
ok "$(awk -v a="$P2" 'BEGIN{print (a<0.001 && a>-0.001)?1:0}')" \
   "beam 2, lab spin +x  -> P.2 = 0        (got $P2)"

echo
echo "=== a partial polarization is reported as given ==="
run "polar_x.1=0;polar_y.1=0;polar_z.1=0.8;polar_x.2=0;polar_y.2=0;polar_z.2=-0.3"
ok "$([ "$(grab 1)" = "+0.8000" ] && echo 1 || echo 0)" \
   "P.1 = +0.8000                          (got $(grab 1))"
ok "$([ "$(grab 2)" = "+0.3000" ] && echo 1 || echo 0)" \
   "P.2 = +0.3000 from lab -0.3            (got $(grab 2))"

echo
echo "=== all three states are sampled, and reported ==="
for st in initial collision final; do
  grep -q "at $st" o.dat && ok 1 "the $st state is present" || ok 0 "the $st state is present"
done

echo
echo "=== with no spin stored the block is absent, not zero-filled ==="
# It emits nothing rather than "not computed": two extra lines in every
# spin-less run would break the bit-identity check against the v0.5.0 reference,
# which is a standing invariant here.
sed "s/QQnothingQQ/x/" "$D" > acc.dat
"$B" c_nt1 p_c_nt1 o.dat > stdout.log 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
if grep -q "longitudinal bunch polarization" o.dat; then
  ok 0 "no polarization block is emitted"
else
  ok 1 "no polarization block is emitted"
fi
if grep -q "at initial" o.dat; then
  ok 0 "and no polarization numbers are printed"
else
  ok 1 "and no polarization numbers are printed"
fi

echo
if [ "$fails" -eq 0 ]; then echo "  longitudinal polarization is reported correctly"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
