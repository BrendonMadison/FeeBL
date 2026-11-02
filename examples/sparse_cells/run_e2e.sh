#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/legacy_wavefront_scheduler/stateful_tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-sparse-e2e.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

make_case()
{
  local name="$1" sparse="$2"
  mkdir -p "$work/$name"
  sed "s/collision_schedule=\"legacy\";/sparse_cells=$sparse;collision_schedule=\"legacy\";/" \
    "$template" >"$work/$name/acc.dat"
}

run_case()
{
  local name="$1" sparse="$2"
  make_case "$name" "$sparse"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  )
}

run_case dense 0
run_case sparse 1

# The selector and backend provenance are the only intentional protocol
# additions. Every scientific result line and all stateful output artifacts
# must remain byte-for-byte identical.
provenance_pattern=' cpu_primary_move = |compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance|collision_schedule_requested|sparse_cells = '
grep -Ev "$provenance_pattern" \
  "$work/dense/result.out" >"$work/dense/result.normalized"
grep -Ev "$provenance_pattern" \
  "$work/sparse/result.out" >"$work/sparse/result.normalized"
cmp "$work/dense/result.normalized" "$work/sparse/result.normalized"

mapfile -t dense_outputs < <(
  cd "$work/dense"
  find . -maxdepth 1 -type f \
    ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
    ! -name result.out ! -name result.normalized -printf '%f\n' | sort
)
mapfile -t sparse_outputs < <(
  cd "$work/sparse"
  find . -maxdepth 1 -type f \
    ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
    ! -name result.out ! -name result.normalized -printf '%f\n' | sort
)
if [[ "${dense_outputs[*]}" != "${sparse_outputs[*]}" ]]; then
  echo "dense and sparse output file sets differ" >&2
  exit 1
fi
for output in "${dense_outputs[@]}"; do
  cmp "$work/dense/$output" "$work/sparse/$output"
done

if grep -q 'FEEBL sparse-cell summary:' "$work/dense/stderr.txt"; then
  echo "default dense mode emitted sparse execution provenance" >&2
  exit 1
fi
grep -Eq \
  'FEEBL sparse-cell summary:.*avoided_clear_visits=[1-9][0-9]*.*avoided_traversal_visits=[1-9][0-9]*.*order=ascending-cell/stable-insertion' \
  "$work/sparse/stderr.txt"

echo "PASS: sparse cells preserve stateful FeeBL output byte-for-byte"
echo "PASS: active-cell provenance proves avoided clear and traversal work"
