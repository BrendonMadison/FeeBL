#!/bin/bash
# The two beams are counter-propagating, so their spins must precess alike.
#
# BMT precession is  omega ~ ga1*B_trans - ce1*gamma*(beta x E),  and at 125 GeV
# ga1 and ce1*gamma are equal to six significant figures -- they differ by
# exactly 1/(1+gamma) = 4.09e-6. So whether the magnetic and electric terms ADD
# or CANCEL is decided entirely by the relative sign of B and beta x E, and the
# answer swings the precession by five orders of magnitude.
#
# Counter-propagating beams must ADD. The near-cancellation is the 1/gamma
# suppression that belongs to a CO-moving source, and getting it for one beam
# only means the field reconstruction used one source direction for both.
#
# The sharp test: give the two beams DIFFERENT spin lengths and identical
# everything else. A small rotation tilts s by theta, so the helicity change is
# Delta(s.p_hat) = -s_z*(1-cos theta) ~ -s_z*theta^2/2. Both beams see the same
# field, hence the same theta, so the two depolarizations must be in the ratio
# of their spin lengths -- and nothing else.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_sym.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

fails=0
SZ1=0.8
SZ2=0.3
EXPECT=$(awk -v a=$SZ2 -v b=$SZ1 'BEGIN{printf "%.4f", a/b}')

# Spin flip OFF: deterministic precession only, so this is pure field geometry
# with no stochastic contribution to argue about.
sed -e "0,/do_lumi=0/{s/do_lumi=0/bmt_precession=1;ST_spin_flip=0;do_lumi=0/}" \
    -e "s/angle_x.1=0.00000000/polar_x.1=0;polar_y.1=0;polar_z.1=$SZ1;polar_x.2=0;polar_y.2=0;polar_z.2=$SZ2;angle_x.1=0.00000000/" \
    "$D" > acc.dat
"$B" c_nt1 p_c_nt1 o.dat > stdout.log 2>&1 || { echo "FeeBL simulation failed; refusing stale output." >&2; exit 1; }

# Anchor on the DATA line: an explanatory sentence in the same report also
# contains this word, and matching it fed text to awk.
LINE=$(grep "depolarization (final - initial)" o.dat | head -1)
D1=$(echo "$LINE" | sed 's/.*beam 1 \([-+0-9.e]*\).*/\1/')
D2=$(echo "$LINE" | sed 's/.*beam 2 \([-+0-9.e]*\).*/\1/')
printf "  spin lengths      : beam 1 %s , beam 2 %s\n" "$SZ1" "$SZ2"
printf "  depolarization    : beam 1 %s , beam 2 %s\n" "$D1" "$D2"

RATIO=$(awk -v a="$D2" -v b="$D1" 'BEGIN{ if (b==0) print 0; else { r=a/b; print (r<0)?-r:r } }')
printf "  |beam2 / beam1|   : %s   (expected %s from the spin lengths)\n" "$RATIO" "$EXPECT"

awk -v r="$RATIO" -v e="$EXPECT" 'BEGIN{
  d = r - e; if (d < 0) d = -d
  ok = (e > 0) && (d/e < 0.25)
  printf "  %-52s %s\n", "the two beams precess alike, scaled by |s|", ok?"PASS":"FAIL"
  exit ok?0:1 }' || fails=$((fails+1))

# and the sign: beam 1 loses helicity, beam 2's helicity is negative so it moves
# the other way, both shrinking in magnitude
awk -v a="$D1" -v b="$D2" 'BEGIN{
  ok = (a < 0) && (b > 0)
  printf "  %-52s %s\n", "and in opposite directions, both toward zero", ok?"PASS":"FAIL"
  exit ok?0:1 }' || fails=$((fails+1))

echo
if [ "$fails" -eq 0 ]; then echo "  the beams are treated symmetrically"; else
  echo "  $fails FAILURE(S) -- one beam is getting the co-moving field geometry"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
