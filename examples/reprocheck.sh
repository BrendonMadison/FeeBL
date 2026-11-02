#!/bin/bash
# Reproducibility check: do the suites run against the tree they live in?
#
# Copy the source tree and run its suites from the relocated directory.
# Corrupting the original executable makes accidental path reuse visible.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
WORK=$(mktemp -d)
ALT="$WORK/feebl-elsewhere"

cp -a "$SRC" "$ALT"
echo "  tree under test : $SRC"
echo "  copied to       : $ALT"

# Make the original binary unusable for the duration. If any script still
# reaches for the original tree, it now fails loudly instead of quietly
# producing the right answer from the wrong place.
#
# The original is MOVED ASIDE, not overwritten. The previous version copied it
# away and then wrote a stub over the real path, so a hard kill between those
# two steps left the developer's working tree holding a stub. A rename never
# destroys the file.
SAVE="$WORK/feebl.orig"
mv "$SRC/src/feebl" "$SAVE"
printf '#!/bin/sh\necho "LEAKED: a test reached the ORIGINAL tree" >&2\nexit 99\n' > "$SRC/src/feebl"
chmod +x "$SRC/src/feebl"
restore () {
  rm -f "$SRC/src/feebl"
  if [ -f "$SAVE" ]; then mv "$SAVE" "$SRC/src/feebl"; chmod +x "$SRC/src/feebl"; fi
  rm -rf "$WORK"
}
trap restore EXIT

fails=0
run () {   # script  label
  local out rc
  set +e
  out=$(cd "$ALT" && bash "$1" 2>&1); rc=$?
  set -e
  if echo "$out" | grep -q "LEAKED"; then
    printf "  %-44s LEAKED to the original tree\n" "$2"; fails=$((fails+1))
  elif [ "$rc" -ne 0 ]; then
    printf "  %-44s FAILED (exit %s)\n" "$2" "$rc"; fails=$((fails+1))
    echo "$out" | tail -4 | sed 's/^/      /'
  else
    printf "  %-44s PASS\n" "$2"
  fi
}

echo
echo "=== suites run from the copy, original binary sabotaged ==="
run examples/conventions/run.sh   "conventions"
run examples/canonical/run.sh     "canonical + HDF5 reader"
run examples/negative_checks.sh   "negative checks"
run examples/maps/run.sh          "maps, crab and the guards"

echo
echo "=== and bitcheck refuses rather than passing when the reference is absent ==="
set +e
out=$(cd "$ALT" && FEEBL_REF=/nonexistent bash examples/bitcheck.sh 2>&1); rc=$?
set -e
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "did NOT run"; then
  printf "  %-44s PASS\n" "missing reference reports failure"
else
  printf "  %-44s FAIL (exit %s)\n" "missing reference reports failure" "$rc"; fails=$((fails+1))
fi

echo
if [ "$fails" -eq 0 ]; then echo "  the suites are location-independent"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
