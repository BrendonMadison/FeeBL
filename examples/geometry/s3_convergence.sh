#!/bin/bash
# Is benchmark16 s3 (FCC-ee Z, the block that validates published finite-angle
# luminosity) converged in n_z?
#
# The crossing angle is applied as a per-slice position shear, so the sheared
# bunch's transverse profile is a staircase whose step is angle_x*delta_z. On s3:
#   sigma_z 14.5 mm, cut_z = 3 sigma_z, n_z = 64  ->  delta_z = 1359 um
#   step = 0.015 * 1359 um = 20.4 um,  sigma_x = 8.40 um  ->  2.43 sigma_x
#
# n_t does NOT help: delta_z = 2*cut_z/n_z has no n_t in it. So although
# increasing n_z*n_t refines force integration, the position shear is
# resolved by n_z alone.
#
# The first version of this probe anchored its sed on "^n_x=1024", but the deck
# line begins with a brace -- "{n_x=1024;..." -- so the substitution never
# matched and BOTH runs used n_z=64, returning identical numbers that looked
# like perfect convergence. The deck is now verified after every edit, and the
# probe refuses to run if the value did not change.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
B=${FEEBL_TEST_BINARY:-$ROOT/src/feebl}
D=$ROOT/examples/decks/benchmark16.dat
. "$ROOT/examples/lib/harness.sh"
RUN_L=""

run () {   # n_z
  sed "s/{n_x=1024;n_y=512;n_z=64;/{n_x=1024;n_y=512;n_z=$1;/g" "$D" > acc.dat
  # prove the edit took: the p_s3 block must now carry the requested n_z
  local got
  got=$(sed -n '/PARAMETERS:: p_s3/,/}/p' acc.dat | grep -o 'n_z=[0-9]*' | head -1 | cut -d= -f2)
  if [ "$got" != "$1" ]; then
    echo "  ABORT: deck still has n_z=$got, wanted $1 -- the probe would be vacuous"
    exit 1
  fi
  feebl_run "$B" s3 p_s3 o.dat
  local L
  if grep -q "lumi_ee =" o.dat; then L=$(need_num o.dat lumi_ee)
  else                               L=$(need_num o.dat lumi_fine); fi
  RUN_L=$L
  awk -v n="$1" -v l="$L" 'BEGIN{
    step = 0.015 * (2*3*14500.0/n)
    printf "  n_z=%-5s step %6.1f um = %.2f sigma_x    L = %s\n", n, step, step/8.40, l}'
}

echo "  benchmark16 s3, FCC-ee Z"
run 64;  L64=$RUN_L
run 128; L128=$RUN_L
echo
echo "  Identical values now mean convergence rather than a failed substitution."
echo
fails=0
chk () { printf "  %-56s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
         [ "$1" = 1 ] || fails=$((fails+1)); }
chk "$(awk -v a="$L64" -v b="$L128" 'BEGIN{print (a>0 && b>0)?1:0}')" \
    "both refinements produced a positive luminosity"
echo
echo "  n_z 64 -> 128 shift: $(awk -v a="$L64" -v b="$L128" 'BEGIN{printf "%+.3f%%", 100*(b-a)/a}')"
echo "  (recorded, not asserted: s3 is documented as not converged)"
echo "  $fails failure(s)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
