#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/feebl [evidence-directory]" >&2
  exit 2
fi
command -v valgrind >/dev/null || {
  echo "Valgrind is required for the Helgrind race gate" >&2
  exit 2
}

binary="$(readlink -f "$1")"
evidence="${2:-}"
if [[ -n "$evidence" ]]; then
  if [[ -e "$evidence" ]]; then
    echo "evidence directory already exists: $evidence" >&2
    exit 2
  fi
  evidence="$(readlink -m "$evidence")"
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/cpu_primary_move/tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-cpu-primary-helgrind.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

sed 's|cpu_primary_move=0;|compute_backend="cpu";cpu_threads=4;cpu_variant="pool_dual";reduction_mode="strict";cpu_primary_move=1;|' \
  "$template" >"$work/acc.dat"
(
  cd "$work"
  valgrind --tool=helgrind --error-exitcode=97 --fair-sched=yes \
    --log-file=helgrind.log \
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
)
grep -q 'ERROR SUMMARY: 0 errors from 0 contexts' "$work/helgrind.log"
grep -Eq \
  'FEEBL: final .*cpu_primary_move=pool-static-primary-v1.*cpu_primary_move_calls=[1-9][0-9]*' \
  "$work/stderr.txt"
if [[ -n "$evidence" ]]; then
  mkdir -p "$evidence"
  cp "$work/acc.dat" "$work/helgrind.log" "$work/stdout.txt" \
    "$work/stderr.txt" "$work/result.out" "$evidence/"
  sha256sum "$binary" "$template" >"$evidence/input-sha256.txt"
  {
    printf 'binary=%s\n' "$binary"
    printf 'source_root=%s\n' "$root"
    git -C "$root" rev-parse HEAD
    git -C "$root" status --short --branch
    uname -a
    valgrind --version
  } >"$evidence/provenance.txt"
fi
echo "PASS: Helgrind found no race in the four-worker production movement path"
