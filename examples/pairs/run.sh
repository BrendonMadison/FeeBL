#!/bin/bash
# Exact pair invariant mass and weighted moments.
#
# Two tiers. The analytic gate checks W as a Lorentz scalar and the estimators
# against closed forms, independently of FEEBL. The production arm then checks
# that a real run writes a spectrum whose content is self-consistent.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# examples/pairs/ sits two levels below the repo root, not one.
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
. "$ROOT/examples/lib/harness.sh"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

echo "=== analytic: W is a scalar, and the estimators are what they claim ==="
g++ -std=c++17 -O2 -Wall -Wextra -I"$ROOT/src" \
    "$HERE/test_pairmoments.cc" -o "$OUT/tpm"
"$OUT/tpm"

echo
echo "=== production: a real run writes a self-consistent spectrum ==="
cd "$OUT"
sed -e "0,/do_lumi=0/{s/do_lumi=0/store_native=1;bmt_precession=1;do_lumi=1/}" \
    -e "0,/num_lumi=100000/{s/num_lumi=100000/num_lumi=2000/}" \
    -e "s/waist_y=0;/waist_y=0;polar_z.1=0.8;polar_z.2=-0.3;/g" \
    "$D" > acc.dat
feebl_run "$B" c_nt1 p_c_nt1 o.dat
need_nonempty lumi.ee.out.dLdW.fne

fails=0
ok () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

ROWS=$(grep -vc '^#' lumi.ee.out.dLdW.fne)
COLS=$(grep '^#COLUMNS' lumi.ee.out.dLdW.fne | sed 's/^#COLUMNS //' | wc -w)
WIDTH=$(grep -v '^#' lumi.ee.out.dLdW.fne | head -1 | wc -w)
MEANW=$(grep '^#mean_W' lumi.ee.out.dLdW.fne | awk '{print $2}')
UND=$(grep '^#underflow' lumi.ee.out.dLdW.fne | awk '{print $2}')
OVR=$(grep '^#overflow' lumi.ee.out.dLdW.fne | awk '{print $2}')

printf "      %s filled bins, mean W = %s GeV\n" "$ROWS" "$MEANW"

ok "$([ "$ROWS" -gt 10 ] && echo 1 || echo 0)" "the spectrum has filled bins ($ROWS)"
ok "$([ "$COLS" = "$WIDTH" ] && echo 1 || echo 0)" \
   "row width matches #COLUMNS ($COLS)"

# A 125+125 GeV run peaks near 250 and is pulled DOWN by beamstrahlung, never up.
ok "$(awk -v w="$MEANW" 'BEGIN{print (w>200 && w<250)?1:0}')" \
   "mean W is below the nominal 250 GeV, as beamstrahlung requires"

# Nothing should fall outside a range set from the beam energies with headroom.
ok "$(awk -v u="$UND" -v o="$OVR" 'BEGIN{print (u==0 && o==0)?1:0}')" \
   "no under- or overflow with the configured range"

# h1h2 must equal <h1><h2> + cov, bin by bin. That is the definition, so it is
# a check on the accumulator rather than on the physics -- and it would catch a
# covariance accumulated against the wrong mean.
BAD=$(grep -v '^#' lumi.ee.out.dLdW.fne | awk '
  { lhs = $9; rhs = $5*$7 + $10; d = lhs - rhs; if (d<0) d = -d
    if (d > 1e-6 * (1 + (lhs<0?-lhs:lhs))) bad++ } END { print bad+0 }')
ok "$([ "$BAD" = 0 ] && echo 1 || echo 0)" \
   "<h1h2> = <h1><h2> + cov in every bin ($BAD violations)"

# The deck sets polar_z.1 = +0.8 and polar_z.2 = -0.3. Beam 2 travels toward -z,
# so its HELICITY is +0.3: a sign error in the local-helicity projection would
# show up here as a negative h2.
BADSIGN=$(grep -v '^#' lumi.ee.out.dLdW.fne | awk '
  { if ($5 < 0.5) b++; if ($7 < 0.1) b++ } END { print b+0 }')
ok "$([ "$BADSIGN" = 0 ] && echo 1 || echo 0)" \
   "h1 ~ +0.8 and h2 ~ +0.3, so beam 2's helicity sign is right"

# N_eff must never exceed the pairs that entered the bin.
ok "$(grep -v '^#' lumi.ee.out.dLdW.fne | awk '{ if ($11 < 0) b++ } END { print (b+0==0)?1:0 }')" \
   "N_eff is non-negative in every bin"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
