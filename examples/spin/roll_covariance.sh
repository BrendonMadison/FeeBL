#!/bin/bash
# The transverse roll must carry the spin -- END TO END.
#
# examples/spin/test_roll.cc gates the mechanism at the unit level. This gates the
# consequence: run a full collision with a partly TRANSVERSE initial spin, once
# upright and once rolled by pi/2, and check that the spin came round with the
# beam.
#
# Why a transverse component is essential: a roll is a rotation about z, so s_z
# is invariant under it. A purely longitudinal spin (the usual polar_z=0.8 deck)
# cannot detect this bug at all -- which is exactly why it survived. The signal
# only exists in s_x, s_y.
#
# The covariance statement is exact and independent of how much the bunch
# depolarizes on the way through, because it relates the two runs rather than
# either run to a prediction:
#
#     <s_x>_rolled  =  <s_y>_upright
#     <s_y>_rolled  = -<s_x>_upright
#     <s_z>_rolled  =  <s_z>_upright
#
# for a roll of pi/2 in PARTICLE::rotate's sense (x' = cos.x + sin.y).
#
# Configuration notes:
#   * SQUARE grid (cut_y := cut_x) so a pi/2 roll maps mesh cells exactly onto
#     mesh cells and the field solver is not being asked to do anything the
#     discretisation cannot represent.
#   * do_photons=0 so the run is deterministic: the only spin dynamics is BMT
#     precession, and the comparison is not fighting the RNG.
#   * store_beam=1 dumps the post-collision beam, whose last three columns are
#     spin (PARTICLE_WITH_SPIN::persistent_flow).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_rollcov.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
ok () { printf "  %-52s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

mk () {   # phi
  sed "s/cut_y=459.55/cut_y=3092.9/g; \
       s/angle_phi.1=0.00000000/angle_phi.1=$1/g; \
       s/angle_phi.2=0.00000000/angle_phi.2=$1/g; \
       s/waist_y=0;/waist_y=0;polar_x.1=0.6;polar_y.1=0.0;polar_z.1=0.8;polar_x.2=0.6;polar_y.2=0.0;polar_z.2=0.8;/g; \
       s/do_photons=1/do_photons=0;bmt_precession=1;store_beam=1/g" "$D" > acc.dat
}

# columns: energy x y z vx vy sx sy sz
mean () { awk '{sx+=$7; sy+=$8; sz+=$9; n++}
               END{if(n==0){print "EMPTY"; exit 1}
                   printf "%.6f %.6f %.6f\n", sx/n, sy/n, sz/n}' "$1"; }

run () {  # phi outfile
  mk "$1"
  "$B" c_nt1 p_c_nt1 o.dat >/dev/null 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }
  [ -s beam1.dat ] || { echo "  beam1.dat absent or empty -- the run did not"; \
                        echo "  produce a beam dump, so this test checked NOTHING."; exit 1; }
  cp beam1.dat "$2"
}

echo "=== the roll carries the spin, through a full collision ==="
run 0.0        b_ref.dat
run 1.57079633 b_rot.dat

read -r RX RY RZ <<< "$(mean b_ref.dat)"
read -r TX TY TZ <<< "$(mean b_rot.dat)"
printf "  upright  <s> = (%+.4f, %+.4f, %+.4f)\n" "$RX" "$RY" "$RZ"
printf "  rolled   <s> = (%+.4f, %+.4f, %+.4f)\n" "$TX" "$TY" "$TZ"

near () { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; print (d<=t)?1:0}'; }
far  () { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; print (d>t)?1:0}'; }

TOL=0.02

# NON-VACUITY FIRST. If the deck did not actually give the beam a transverse
# spin there is nothing here to rotate, and every covariance check below would
# pass trivially. Refuse to report success in that case.
ok "$(far "$RX" 0 0.3)" "the upright run really has a transverse spin"

# THE FALSIFIER. This is the bug's exact signature: with a non-virtual
# PARTICLE::rotate the spin is untouched and <s>_rolled == <s>_upright.
ok "$(far "$TX" "$RX" 0.3)" "the roll really moved the spin (not a no-op)"

# COVARIANCE.
ok "$(near "$TX" "$RY"  "$TOL")" "<s_x> rolled  ==  <s_y> upright"
# negate through awk: "-$RX" would become "--0.6" for a negative RX
NRX=$(awk -v x="$RX" 'BEGIN{printf "%.6f", -x}')
ok "$(near "$TY" "$NRX" "$TOL")" "<s_y> rolled  == -<s_x> upright"
ok "$(near "$TZ" "$RZ"  "$TOL")" "<s_z> is invariant under the roll"

# And the length is untouched, so nothing was rescaled into agreement.
NR_=$(awk -v x="$RX" -v y="$RY" -v z="$RZ" 'BEGIN{printf "%.6f", sqrt(x*x+y*y+z*z)}')
NT_=$(awk -v x="$TX" -v y="$TY" -v z="$TZ" 'BEGIN{printf "%.6f", sqrt(x*x+y*y+z*z)}')
ok "$(near "$NT_" "$NR_" 0.005)" "|<s>| is unchanged by the roll"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
