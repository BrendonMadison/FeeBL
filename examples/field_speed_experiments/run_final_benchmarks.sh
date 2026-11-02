#!/usr/bin/env bash
set -euo pipefail

usage()
{
  cat >&2 <<'EOF'
usage:
  run_final_benchmarks.sh cpu CPU_BINARY CPU_CARD EVIDENCE_DIR [REPEATS]
  run_final_benchmarks.sh gpu CUDA_BINARY GPU_CARD EVIDENCE_DIR [REPEATS] [BATCH]

The CPU matrix alternates fft, auto, and forced-direct runs after one auto
warm-up.  The GPU matrix runs one warm-up followed by repeated exact mode-4
runs at the requested PlanMany capacity (default 64).  Timed runs use fixed
physical-core affinity and retain cards, logs, resource receipts, outputs,
and SHA-256 digests under EVIDENCE_DIR.
EOF
  exit 2
}

[[ $# -ge 4 ]] || usage
mode="$1"
binary="$(readlink -f "$2")"
source_card="$(readlink -f "$3")"
evidence="$(readlink -m "$4")"
repeats="${5:-3}"

[[ -x "$binary" ]] || { echo "not an executable: $binary" >&2; exit 2; }
[[ -f "$source_card" ]] || { echo "not a card: $source_card" >&2; exit 2; }
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
monotonic_runner="$repo/examples/field_speed_experiments/run_monotonic.py"
mkdir -p "$evidence"

normalise_result()
{
  grep -vE \
    '^collision_schedule_requested|^sparse_cells =|^wavefront_field_solver =|compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
    "$1/result.out" >"$1/result.normalized"
}

hash_outputs()
{
  (
    cd "$1"
    sha256sum result.normalized lumi.ee.out lumi.ee.out.fne \
      lumi.ee.out.dLdW.fne >output-sha256.txt
    sha256sum "$binary" >binary-sha256.txt
    sha256sum acc.dat >card-sha256.txt
  )
}

run_gpu_once()
{
  local label="$1"
  local batch="$2"
  local destination="$evidence/$label"
  local run_dir
  run_dir="$(mktemp -d "/tmp/feebl-gpu-mode4-b${batch}.XXXXXX")"
  mkdir -p "$destination"
  sed -E "s/gpu2_batch_slices=[0-9]+;/gpu2_batch_slices=${batch};/" \
    "$source_card" >"$run_dir/acc.dat"
  (
    cd "$run_dir"
    "$monotonic_runner" --json monotonic-time.json \
      --resource resource-time.txt --stdout stdout.txt --stderr stderr.txt -- \
      taskset -c 0 "$binary" fccz p_fccz result.out
  )
  cp -a "$run_dir"/. "$destination"/
  printf '%s\n' "$run_dir" >"$destination/linux-run-directory.txt"
  normalise_result "$destination"
  hash_outputs "$destination"
}

case "$mode" in
  cpu)
    helper="$repo/examples/sparse_direct_field/benchmark_fccz.sh"
    "$helper" "$binary" "$source_card" auto "$evidence/warmup-auto"
    for ((i=1; i<=repeats; ++i)); do
      "$helper" "$binary" "$source_card" fft "$evidence/fft-$i"
      "$helper" "$binary" "$source_card" auto "$evidence/auto-$i"
      "$helper" "$binary" "$source_card" direct "$evidence/direct-$i"
    done
    ;;
  gpu)
    batch="${6:-64}"
    [[ "$batch" =~ ^[1-9][0-9]*$ ]] && ((batch <= 256)) || {
      echo "BATCH must be in [1,256]" >&2
      exit 2
    }
    run_gpu_once "warmup-batch-$batch" "$batch"
    for ((i=1; i<=repeats; ++i)); do
      run_gpu_once "batch-${batch}-$i" "$batch"
    done
    ;;
  *) usage ;;
esac
