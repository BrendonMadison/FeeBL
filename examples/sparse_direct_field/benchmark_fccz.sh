#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 /path/to/feebl /path/to/source-card direct|auto|fft /path/to/evidence" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
source_card="$(readlink -f "$2")"
solver="$3"
evidence="$4"
case "$solver" in
  direct|auto|fft) ;;
  *) echo "invalid field solver: $solver" >&2; exit 2 ;;
esac
mkdir -p "$evidence"
run_dir="$(mktemp -d "/tmp/feebl-sparse-direct-${solver}.XXXXXX")"
monotonic_runner="$(cd "$(dirname "$0")/../field_speed_experiments" && pwd)/run_monotonic.py"

limit=262144
sed "s/sparse_cells=1;/sparse_cells=1;wavefront_field_solver=\"$solver\";sparse_direct_interaction_limit=$limit;/" \
  "$source_card" >"$run_dir/acc.dat"
(
  cd "$run_dir"
  "$monotonic_runner" --json monotonic-time.json \
    --resource resource-time.txt --stdout stdout.txt --stderr stderr.txt -- \
    taskset -c 0,2,4,6,8,10,12,14,16,18 \
    "$binary" fccz p_fccz result.out
)
cp -a "$run_dir"/. "$evidence"/
printf '%s\n' "$run_dir" >"$evidence/linux-run-directory.txt"
grep -vE \
  '^collision_schedule_requested|^sparse_cells =|^wavefront_field_solver =|compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
  "$evidence/result.out" >"$evidence/result.normalized"
(
  cd "$evidence"
  sha256sum result.normalized lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne \
    >output-sha256.txt
  sha256sum "$binary" >binary-sha256.txt
)
