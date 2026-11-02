#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/legacy_wavefront_scheduler/stateful_tiny.dat"
compare="$root/examples/gpu/compare_feebl_outputs.py"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-sparse-direct-e2e.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

make_case()
{
  local name="$1" solver="$2" limit="$3" charge="${4:--1.0}"
  mkdir -p "$work/$name"
  sed \
    -e "s/charge_sign=-1\.0;/charge_sign=$charge;/" \
    -e "s/collision_schedule=\"legacy\";/compute_backend=\"cpu\";cpu_threads=4;cpu_variant=\"pool_dual\";reduction_mode=\"strict\";sparse_cells=1;wavefront_field_solver=\"$solver\";sparse_direct_interaction_limit=$limit;collision_schedule=\"wavefront_parallel\";/" \
    "$template" >"$work/$name/acc.dat"
}

run_case()
{
  local name="$1" solver="$2" limit="$3" charge="${4:--1.0}"
  make_case "$name" "$solver" "$limit" "$charge"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  )
}

normalize_result()
{
  grep -vE \
    '^collision_schedule_requested|^sparse_cells =|^wavefront_field_solver =|compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
    "$1"
}

run_case fft fft 262144
run_case direct direct 262144
run_case direct-repeat direct 262144
run_case auto-direct auto 100000000
run_case auto-fft auto 1

for name in fft direct direct-repeat auto-direct auto-fft; do
  normalize_result "$work/$name/result.out" >"$work/$name/result.normalized"
done

# Direct convolution and FFT have different floating summation trees.  The
# accepted physics envelope is deliberately far tighter than FeeBL's 1--3%
# application accuracy: every published result scalar must agree to 5e-5.
python3 "$compare" "$work/fft/result.normalized" \
  "$work/direct/result.normalized" --rtol 5e-5 --atol 1e-10

# Auto selection is literal at both sides of the registered crossover.
cmp "$work/direct/result.normalized" "$work/auto-direct/result.normalized"
cmp "$work/fft/result.normalized" "$work/auto-fft/result.normalized"
grep -Eq 'sparse-direct field summary:.*direct_encounters=64.*fft_encounters=0.*direct_interactions=[1-9][0-9]*' \
  "$work/direct/stderr.txt"
grep -Eq 'sparse-direct field summary:.*direct_encounters=64.*fft_encounters=0' \
  "$work/auto-direct/stderr.txt"
grep -Eq 'sparse-direct field summary:.*direct_encounters=0.*fft_encounters=64' \
  "$work/auto-fft/stderr.txt"

# A forced-direct repeat must reproduce every physics artifact bit for bit.
cmp "$work/direct/result.normalized" "$work/direct-repeat/result.normalized"
mapfile -t direct_outputs < <(
  cd "$work/direct"
  find . -maxdepth 1 -type f \
    ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
    ! -name result.out ! -name result.normalized -printf '%f\n' | sort
)
for output in "${direct_outputs[@]}"; do
  cmp "$work/direct/$output" "$work/direct-repeat/$output"
done

# Preserve FIELD::fold_fft's historical exact zero shortcut.
run_case zero-fft fft 262144 0.0
run_case zero-direct direct 262144 0.0
normalize_result "$work/zero-fft/result.out" >"$work/zero-fft/result.normalized"
normalize_result "$work/zero-direct/result.out" >"$work/zero-direct/result.normalized"
cmp "$work/zero-fft/result.normalized" "$work/zero-direct/result.normalized"

# Unsupported and internally inconsistent cards fail; none silently falls
# back to the legacy FFT implementation.
mkdir -p "$work/rejected-selector"
sed 's/collision_schedule="legacy";/wavefront_field_solver="approximate";collision_schedule="legacy";/' \
  "$template" >"$work/rejected-selector/acc.dat"
status=0
(cd "$work/rejected-selector" &&
 "$binary" tiny exact result.out >stdout.txt 2>stderr.txt) || status=$?
[[ $status -ne 0 ]]
grep -q 'wavefront_field_solver must be fft, direct, or auto' \
  "$work/rejected-selector/stderr.txt"

mkdir -p "$work/rejected-scope"
sed 's/collision_schedule="legacy";/wavefront_field_solver="direct";collision_schedule="legacy";/' \
  "$template" >"$work/rejected-scope/acc.dat"
status=0
(cd "$work/rejected-scope" &&
 "$binary" tiny exact result.out >stdout.txt 2>stderr.txt) || status=$?
[[ $status -ne 0 ]]
grep -q 'No fallback was performed' "$work/rejected-scope/stderr.txt"

echo "sparse-direct field end-to-end tests passed"
