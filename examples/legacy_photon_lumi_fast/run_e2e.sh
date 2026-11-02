#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
deck="$root/examples/legacy_photon_lumi_fast/tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-photon-fast-e2e.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/cartesian" "$work/factorized"

rejected_status=0
(
  cd "$work"
  "$binary" --acc_file="$deck" tiny rejected_heap rejected.out \
    >rejected.stdout 2>rejected.stderr
) || rejected_status=$?
if [[ "$rejected_status" -ne 2 ]]; then
  echo "photon_lumi_factorized=1 rejection exited $rejected_status, expected 2" >&2
  exit 1
fi
grep -q "No fallback was performed" "$work/rejected.stderr"
grep -q "gamma-e luminosity heap is enabled" "$work/rejected.stderr"

(
  cd "$work/cartesian"
  "$binary" --acc_file="$deck" tiny cartesian result.out \
    >stdout.txt 2>stderr.txt
)
(
  cd "$work/factorized"
  "$binary" --acc_file="$deck" tiny factorized result.out \
    >stdout.txt 2>stderr.txt
)

# photon_lumi contains no RNG draw in the admitted feature family.  An exact
# final state match proves that photon generation/sampling and all preceding
# random ordering remained untouched.
cmp "$work/cartesian/rndm.save" "$work/factorized/rndm.save"

python3 - "$work/cartesian/result.out" "$work/factorized/result.out" \
           "$work/cartesian/stderr.txt" "$work/factorized/stderr.txt" <<'PY'
import math
import pathlib
import re
import sys

cart_path, fast_path, cart_err_path, fast_err_path = map(pathlib.Path, sys.argv[1:])
aggregate = re.compile(
    r"^(lumi_(?:eg|ge|gg|gg_high)|lumi\[[0-2]\]\[[0-2]\])\s*=\s*"
    r"([-+0-9.eE]+)", re.MULTILINE)

def values(path):
    found = {name: float(value) for name, value in aggregate.findall(path.read_text())}
    expected = {"lumi_eg", "lumi_ge", "lumi_gg", "lumi_gg_high"}
    expected.update(f"lumi[{i}][{j}]" for i in range(3) for j in range(3))
    missing = expected.difference(found)
    if missing:
        raise SystemExit(f"{path}: missing aggregates {sorted(missing)}")
    return found

cart = values(cart_path)
fast = values(fast_path)
relative_gate = 2.0e-6
absolute_gate = 2.0e-6
for name in sorted(cart):
    a, b = cart[name], fast[name]
    tolerance = max(absolute_gate, relative_gate * max(abs(a), abs(b)))
    if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance:
        raise SystemExit(
            f"aggregate mismatch {name}: cartesian={a:.17g} factorized={b:.17g} "
            f"tolerance={tolerance:.17g}")

cart_err = cart_err_path.read_text()
fast_err = fast_err_path.read_text()
if "FEEBL legacy photon-lumi summary" in cart_err:
    raise SystemExit("default Cartesian run emitted factorized provenance")
match = re.search(
    r"FEEBL legacy photon-lumi summary:\s*"
    r"requested=factorized-exact-v2.*factorized_calls=(\d+).*"
    r"cartesian_calls=(\d+).*cartesian_pairs_visited=(\d+).*"
    r"cartesian_pairs_avoided=(\d+)", fast_err)
if not match:
    raise SystemExit("factorized run did not emit complete provenance counters")
calls, cart_calls, visited, avoided = map(int, match.groups())
if calls <= 0 or cart_calls != 0 or visited != 0 or avoided <= 0:
    raise SystemExit(
        f"invalid provenance calls={calls} cart_calls={cart_calls} "
        f"visited={visited} avoided={avoided}")
print(
    f"PASS: {len(cart)} aggregates within 2e-6 relative / 2e-6 absolute; "
    f"RNG state identical; "
    f"factorized_calls={calls}; Cartesian visits avoided={avoided}")
PY
