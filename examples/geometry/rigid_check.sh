#!/bin/bash
# Rigid-beam mechanism check.
#
# Two mistakes already made here, both worth not repeating:
#   1. inserting charge_sign=0 at the head of the block, when the deck assigns
#      charge_sign=-1.0 further down and the later value wins -- so the test
#      changed nothing and reported "no difference";
#   2. grepping "lumi_fine=" when the output writes "lumi_fine = value".
# Substitute in place, and use the extraction that is known to work.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/quick.dat

. "$ROOT/examples/lib/harness.sh"

# run() sets RUN_LF so the assertions at the bottom have something to compare.
# This script used to run FEEBL as `|| true` and print whatever grep returned,
# so it reported success with the binary replaced by a failing stub.
RUN_LF=""

run () {   # $1 label, $2 sed expr ("" = baseline)
  if [ -z "$2" ]; then cp "$D" acc.dat; else sed "$2" "$D" > acc.dat; fi
  [ -s acc.dat ] || { echo "  FATAL: deck substitution left acc.dat empty" >&2; exit 93; }
  feebl_run "$B" c_nt1 p_c_nt1 o.dat
  local lf le
  lf=$(need_num o.dat lumi_fine)
  le=$(need_num o.dat lumi_ee)
  RUN_LF=$lf
  printf "  %-22s lumi_fine = %-14s lumi_ee = %s\n" "$1" "$lf" "$le"
}

echo "  deck contains:"
grep -o "charge_sign=[^;]*" "$D" | head -1 | sed 's/^/    /'
grep -o "particles=[^;]*" "$D" | head -1 | sed 's/^/    /'
echo

fails=0
run "baseline"          "";                                   L_BASE=$RUN_LF
run "charge_sign = 0"   "s/charge_sign=-1.0/charge_sign=0/g";  L_RIGID=$RUN_LF
run "particles x 1e-4"  "s/particles=2.0000/particles=0.0002/g"; L_SMALL=$RUN_LF

echo
echo "  charge_sign=0 should zero the field. Reducing the bunch population"
echo "  approaches the same rigid limit from the other side: luminosity falls"
echo "  as N^2 but the ENHANCEMENT over geometric overlap goes to 1."
echo

# ASSERTIONS. Without these the script measured three numbers and asserted
# nothing, which is why a dead binary could pass it.
chk () { printf "  %-52s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
         [ "$1" = 1 ] || fails=$((fails+1)); }

chk "$(awk -v a="$L_BASE" -v b="$L_RIGID" 'BEGIN{d=(a-b)/b; if(d<0)d=-d; print (d>0.01)?1:0}')" \
    "disabling the field changes luminosity (pinch is real)"
chk "$(awk -v a="$L_BASE" -v b="$L_RIGID" 'BEGIN{print (a>b)?1:0}')" \
    "and the pinch ENHANCES it, so baseline > rigid"
chk "$(awk -v a="$L_SMALL" 'BEGIN{print (a>0)?1:0}')" \
    "the low-population run still produces a positive luminosity"
chk "$(awk -v a="$L_SMALL" -v b="$L_BASE" 'BEGIN{print (a<b)?1:0}')" \
    "and it is far smaller, as N^2 requires"

echo
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
