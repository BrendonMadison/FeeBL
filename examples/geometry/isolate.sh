#!/bin/bash
# Which knob collapsed the decoupling probe: sigma_z, or cut_z?
#
# l3_decouple.sh scaled sigma_z and cut_z together and saw luminosity fall by
# three orders of magnitude, which is unphysical -- head-on rigid luminosity
# should depend on sigma_z only through the hourglass factor. So one of the two
# substitutions perturbed something uncontrolled.
#
# Hold cut_z at the deck value so the bunch stays fully covered, and vary
# sigma_z alone. If luminosity stays flat, the collapse came from cut_z and the
# probe was simply mis-scaled, not from any sigma_z dependence in the physics.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/quick.dat

. "$ROOT/examples/lib/harness.sh"

# probe() sets PROBE_L. This script used to run FEEBL as `|| true` and print
# whatever grep returned, so it passed with a failing stub.
PROBE_L=""

probe () {   # sigma_z_um  cut_z_mm
  sed "s/charge_sign=-1.0/charge_sign=0/g; s/sigma_z=300/sigma_z=$1/g; s/cut_z=3.0/cut_z=$2/g" "$D" > acc.dat
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  feebl_run "$B" c_nt1 p_c_nt1 o.dat
  local L
  L=$(need_num o.dat lumi_fine)
  PROBE_L=$L
  printf "  sigma_z=%-5s um  cut_z=%-5s mm   L = %s\n" "$1" "$2" "$L"
}

fails=0
echo "  cut_z HELD at the deck value, sigma_z varied alone:"
probe 300 3.0; H300=$PROBE_L
probe 30  3.0; H30=$PROBE_L
probe 3   3.0; H3=$PROBE_L
echo
echo "  sigma_z and cut_z scaled together (the original, failed probe):"
probe 30  0.3;  S30=$PROBE_L
probe 3   0.03; S3=$PROBE_L
echo

# ASSERTIONS. The whole point of the probe is that holding cut_z keeps the
# bunch covered, so luminosity moves only through the hourglass factor, while
# scaling both collapses it. Assert exactly that, rather than printing it.
chk () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
         [ "$1" = 1 ] || fails=$((fails+1)); }

chk "$(awk -v a="$H300" -v b="$H3" 'BEGIN{r=b/a; print (r>0.2 && r<5.0)?1:0}')" \
    "cut_z held: 100x in sigma_z stays within a factor 5"
chk "$(awk -v a="$S3" -v b="$H3" 'BEGIN{print (a < 0.2*b)?1:0}')" \
    "cut_z scaled too: luminosity collapses, as the failed probe did"
chk "$(awk -v a="$H300" -v b="$H30" -v c="$H3" 'BEGIN{print (a>0 && b>0 && c>0)?1:0}')" \
    "every held-cut_z run produced a positive luminosity"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
