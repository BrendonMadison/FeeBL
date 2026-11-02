#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/candidate-feebl [frozen-legacy-feebl]" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
baseline=""
if [[ $# -eq 2 ]]; then
  baseline="$(readlink -f "$2")"
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
deck="$root/examples/legacy_electron_lumi_production/tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-electron-precompute-e2e.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/cartesian" "$work/precomputed" "$work/blocked"

reject () {
  local profile="$1" expected="$2" status
  set +e
  (
    cd "$work"
    "$binary" --acc_file="$deck" tiny "$profile" "$profile.out" \
      >"$profile.stdout" 2>"$profile.stderr"
  )
  status=$?
  set -e
  if [[ $status -ne 2 ]]; then
    echo "$profile: expected fail-closed exit 2, observed $status" >&2
    exit 1
  fi
  grep -q "$expected" "$work/$profile.stderr"
  printf 'PASS: %-32s exit=2 diagnostic=%s\n' "$profile" "$expected"
}

reject rejected_isr "No fallback was performed"
reject rejected_heap "(do_lumi&1)=0"
reject rejected_cross "do_cross=0"
reject rejected_bhabhas "do_bhabhas=0"
reject rejected_jets "do_jets=0"
reject rejected_prod "do_prod=0"
reject rejected_bmt "bmt_precession=0"
reject rejected_invalid "electron_lumi_precompute must be 0, 1, 2, or 3"
reject blocked_rejected_isr "electron_lumi_precompute=2 requires"
reject blocked_rejected_heap "(do_lumi&1)=0"
reject blocked_rejected_cross "do_cross=0"
reject blocked_rejected_bhabhas "do_bhabhas=0"
reject blocked_rejected_jets "do_jets=0"
reject blocked_rejected_prod "do_prod=0"
reject blocked_rejected_bmt "bmt_precession=0"
reject blocked_rejected_st "ST_spin_flip=0"
reject blocked_rejected_bds_spin "do_bds_spin_rotation=0"

(
  cd "$work/cartesian"
  "$binary" --acc_file="$deck" tiny cartesian result.out \
    >stdout.txt 2>stderr.txt
)
(
  cd "$work/precomputed"
  "$binary" --acc_file="$deck" tiny precomputed result.out \
    >stdout.txt 2>stderr.txt
)
(
  cd "$work/blocked"
  "$binary" --acc_file="$deck" tiny blocked result.out \
    >stdout.txt 2>stderr.txt
)

# The admitted kernel has no RNG operation. Exact state identity proves that
# the surrounding particle generation and all preceding random ordering stayed
# untouched.
cmp "$work/cartesian/rndm.save" "$work/precomputed/rndm.save"
cmp "$work/cartesian/rndm.save" "$work/blocked/rndm.save"

python3 - "$work/cartesian/result.out" "$work/precomputed/result.out" \
  "$work/blocked/result.out" <<'PY'
import pathlib
import re
import sys

cartesian, precomputed, blocked = map(pathlib.Path, sys.argv[1:])
cart = cartesian.read_bytes()
pre = precomputed.read_bytes()
blk = blocked.read_bytes()
marker1 = re.compile(br"^ electron_lumi_precompute = 1 .*\n", re.MULTILINE)
marker2 = re.compile(br"^ electron_lumi_precompute = 2 .*\n", re.MULTILINE)
if marker1.search(cart) or marker2.search(cart):
    raise SystemExit("default Cartesian output unexpectedly contains fast-path provenance")
for label, payload, marker, other in (
        ("precomputed", pre, marker1, marker2),
        ("blocked", blk, marker2, marker1)):
    matches = marker.findall(payload)
    if len(matches) != 1:
        raise SystemExit(
            f"{label}: expected one fast-path provenance line, found {len(matches)}")
    if other.search(payload):
        raise SystemExit(f"{label}: contains the wrong fast-path provenance")
    stripped = marker.sub(b"", payload)
    if cart == stripped:
        continue
    for line, (left, right) in enumerate(
            zip(cart.splitlines(), stripped.splitlines()), start=1):
        if left != right:
            raise SystemExit(
                f"{label}: exact production differential failed at line {line}:\n"
                f"Cartesian: {left!r}\n{label}: {right!r}")
    raise SystemExit(
        f"{label}: exact production differential failed: byte lengths "
        f"{len(cart)} != {len(stripped)}")
print("PASS: modes 1 and 2 exactly match Cartesian output after removing provenance")
PY

if [[ -n "$baseline" ]]; then
  mkdir -p "$work/frozen"
  (
    cd "$work/frozen"
    "$baseline" --acc_file="$deck" tiny cartesian result.out \
      >stdout.txt 2>stderr.txt
  )
  cmp "$work/cartesian/rndm.save" "$work/frozen/rndm.save"
  python3 - "$work/cartesian/result.out" "$work/frozen/result.out" <<'PY'
import pathlib
import re
import sys

def legacy_payload(path):
    lines = pathlib.Path(path).read_bytes().splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines)
                  if re.match(br"^ \*{10,}", line)), None)
    if start is None:
        raise SystemExit(f"{path}: legacy banner boundary not found")
    return b"".join(line for line in lines[start:]
                    if not re.match(br"^ *\*", line))

candidate = legacy_payload(sys.argv[1])
frozen = legacy_payload(sys.argv[2])
if candidate != frozen:
    raise SystemExit("default-off candidate differs from frozen legacy payload")
print("PASS: default-off candidate payload and RNG state match the frozen binary exactly")
PY
fi
