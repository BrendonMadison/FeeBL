#!/bin/bash
# Fast bit-identity check against the v0.5.0 tag. Quick deck only (~15 s), so
# this can run after every increment; the three-regime and speed runs are for
# the end of a batch.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
REF=${FEEBL_REF:-}

if [ ! -x "$REF" ]; then
  echo "  cannot run: no v0.5.0 reference binary at $REF"
  echo "  set FEEBL_REF to one, or build the v0.5.0 tag."
  echo "  this check did NOT run -- reporting failure rather than success."
  exit 1
fi
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
cp $ROOT/examples/decks/quick.dat acc.dat || exit 1
fail=0
for t in c_nt1 c_nt2 c_nt5; do
  "$REF" "$t" "p_$t" "r_$t.dat" >/dev/null 2>&1
  "$BIN" "$t" "p_$t" "n_$t.dat" >/dev/null 2>&1
  if [ ! -s "r_$t.dat" ] || [ ! -s "n_$t.dat" ]; then
    printf "  %-8s NO OUTPUT -- run failed\n" "$t"; fail=1; continue
  fi
  # the version banner always differs: develop is 0.6.0-dev, the tag is 0.5.0
  grep -v "Beam Luminosity" "r_$t.dat" > a.txt
  grep -v "Beam Luminosity" "n_$t.dat" > b.txt
  if cmp -s a.txt b.txt; then
    printf "  %-8s BIT-IDENTICAL (%s lines)\n" "$t" "$(wc -l < a.txt)"
  else
    printf "  %-8s DIFFERS (%s lines)\n" "$t" "$(diff a.txt b.txt | grep -c '^[<>]')"
    diff a.txt b.txt | head -6 | sed 's/^/      /'
    fail=1
  fi
done
exit $fail
