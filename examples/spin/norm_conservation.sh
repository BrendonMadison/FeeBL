#!/bin/bash
# Deterministic spin precession must not change the length of the spin vector.
#
# Thomas-BMT precession preserves |s|. A BMT-only drift therefore measures
# numerical integration error. The stochastic spin-flip arm is reported
# separately because photon emission can change the spin state.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_spin.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
TOL=${FEEBL_SPIN_TOL:-1e-5}

# Use 5000 macroparticles and store 500 luminosity pairs with their spins.
mk () {   # ST_spin_flip
  # polar_z stated explicitly. This used to be left to the parser
  # default of (0,0,1), so the test silently depended on the beams
  # being 100% longitudinally polarized -- exactly the assumption a
  # spin run must not make by accident. FEEBL now refuses a spin run
  # whose deck never mentions polarization.
  sed -e '0,/n_m=50000/{s/n_m=50000/n_m=5000/}' \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=1.0;polar_z.2=1.0;/g" \
      -e "0,/do_lumi=0/{s/do_lumi=0/do_lumi=1;bmt_precession=1;ST_spin_flip=$1/}" \
      -e '0,/num_lumi=100000/{s/num_lumi=100000/num_lumi=500/}' \
      "$D" > acc.dat
}

norms () {   # -> "records min1 mean1 max1 min2 mean2 max2"
  awk 'NF >= 16 {
      n1=sqrt($11*$11+$12*$12+$13*$13); n2=sqrt($14*$14+$15*$15+$16*$16);
      if (c==0 || n1<a) a=n1; if (c==0 || n1>b) b=n1;
      if (c==0 || n2<d) d=n2; if (c==0 || n2>e) e=n2;
      s1+=n1; s2+=n2; c++ }
    END { if (c) printf "%d %.9g %.9g %.9g %.9g %.9g %.9g", c, a, s1/c, b, d, s2/c, e;
          else printf "0 0 0 0 0 0 0" }' lumi.ee.out
}

echo "=== BMT only: deterministic, so |s| must be conserved ==="
rm -rf bmt; mkdir bmt; cd bmt
mk 0
set +e; "$B" c_nt1 p_c_nt1 out.dat > stdout.log 2>&1; rc=$?; set -e
if [ "$rc" -ne 0 ] || [ ! -f lumi.ee.out ]; then
  echo "  the BMT run did not produce lumi.ee.out (exit $rc)"; tail -3 stdout.log | sed 's/^/      /'
  exit 1
fi
read -r N A1 M1 B1 A2 M2 B2 <<EOF
$(norms)
EOF
printf "  records %s\n" "$N"
printf "  beam 1  |s| min %.9g  mean %.9g  max %.9g\n" "$A1" "$M1" "$B1"
printf "  beam 2  |s| min %.9g  mean %.9g  max %.9g\n" "$A2" "$M2" "$B2"

assert_norm () {   # value label
  awk -v v="$1" -v t="$TOL" -v l="$2" 'BEGIN{
    d = v - 1; if (d < 0) d = -d
    printf "  %-46s %+.3e  %s\n", l, v-1, (d<=t)?"PASS":"FAIL"
    exit (d<=t)?0:1 }' || fails=$((fails+1))
}
echo
assert_norm "$A1" "beam 1 minimum |s| - 1"
assert_norm "$B1" "beam 1 maximum |s| - 1"
assert_norm "$A2" "beam 2 minimum |s| - 1"
assert_norm "$B2" "beam 2 maximum |s| - 1"
awk -v n="$N" 'BEGIN{ printf "  %-46s %s\n", "and the sample is not empty", (n>0)?"PASS":"FAIL"; exit (n>0)?0:1 }' \
  || fails=$((fails+1))
cd ..

echo
echo "=== BMT + spin flip: reported only, not a physics gate ==="
rm -rf st; mkdir st; cd st
mk 1
set +e; "$B" c_nt1 p_c_nt1 out.dat > stdout.log 2>&1; rc=$?; set -e
if [ "$rc" -eq 0 ] && [ -f lumi.ee.out ]; then
  read -r N A1 M1 B1 A2 M2 B2 <<EOF
$(norms)
EOF
  printf "  records %s\n" "$N"
  printf "  beam 1  |s| min %.9g  mean %.9g  max %.9g\n" "$A1" "$M1" "$B1"
  printf "  beam 2  |s| min %.9g  mean %.9g  max %.9g\n" "$A2" "$M2" "$B2"
  echo "  (stochastic emission may legitimately move these; no assertion made)"
else
  echo "  the spin-flip run failed (exit $rc) -- reported, not asserted"
fi
cd ..

echo
if [ "$fails" -eq 0 ]; then echo "  deterministic precession conserves the spin norm"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
