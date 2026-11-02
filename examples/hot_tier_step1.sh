#!/bin/bash
# Stage B step 1 gate:
#   (a) verify_hot_tier=1 reports ZERO mismatches on both beams -- the float
#       SoA conversion reproduces the object graph exactly, not approximately;
#   (b) the DEFAULT path is still bit-identical to tag v0.5.0, since nothing
#       reads the hot tier yet and the check writes only to stderr.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REF=${FEEBL_REF:-}

if [ ! -x "$REF" ]; then
  echo "  cannot run: no v0.5.0 reference binary at $REF"
  echo "  set FEEBL_REF to one, or build the v0.5.0 tag."
  echo "  this check did NOT run -- reporting failure rather than success."
  exit 1
fi
NEW=$ROOT/src/feebl
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cd "$W"

echo "=== (a) conversion exactness, verify_hot_tier=1 ==="
python3 - <<'PY'
import io
s = io.open("$ROOT/examples/decks/quick.dat", encoding="utf-8").read()
r = []; i = 0
while True:
    j = s.find("$PARAMETERS::", i)
    if j < 0:
        r.append(s[i:]); break
    k = s.find("{", j); r.append(s[i:k+1]); r.append("verify_hot_tier=1;"); i = k + 1
io.open("acc.dat", "w", encoding="utf-8", newline="\n").write("".join(r))
PY
"$NEW" c_nt1 p_c_nt1 hot.dat 2>&1 >/dev/null | grep -i "hot tier" | sed 's/^/  /'

echo
echo "=== (b) default path bit-identical to v0.5.0, banner excluded ==="
cp $ROOT/examples/decks/quick.dat acc.dat
for t in c_nt1 c_nt2 c_nt5; do
  "$REF" "$t" "p_$t" "r_$t.dat" >/dev/null 2>&1
  "$NEW" "$t" "p_$t" "n_$t.dat" >/dev/null 2>&1
  # develop deliberately bumps the version, so the banner line always
  # differs. Compare everything else.
  grep -v "Fast e+e- Beam Luminosity" "r_$t.dat" > "r_$t.cmp"
  grep -v "Fast e+e- Beam Luminosity" "n_$t.dat" > "n_$t.cmp"
  if diff -q "r_$t.cmp" "n_$t.cmp" >/dev/null 2>&1; then
    printf "  %-10s BIT-IDENTICAL\n" "$t"
  else
    printf "  %-10s *** DIFFERS (%s lines) ***\n" "$t" \
      "$(diff "r_$t.cmp" "n_$t.cmp" | grep -c '^[<>]')"
  fi
done
