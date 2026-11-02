#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 /absolute/path/to/feebl card.dat output-dir [workers ...]" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
card="$(readlink -f "$2")"
output="$(readlink -m "$3")"
shift 3
workers=("$@")
if [[ ${#workers[@]} -eq 0 ]]; then workers=(2 4 8); fi
if [[ -e "$output" ]]; then
  echo "refusing existing output directory: $output" >&2
  exit 2
fi
mkdir -p "$output"

cpu_list="${FEEBL_BENCH_CPU_LIST:-0-$(( $(nproc) - 1 ))}"
common='reduction_mode="strict";allow_backend_fallback=0;electron_lumi_precompute=0;photon_lumi_factorized=1;sparse_cells=1;'

inject()
{
  local destination="$1" fragment="$2"
  # n_x begins the selected PARAMETERS body in the supplied FCC-Z deck.
  # Refuse ambiguous cards instead of modifying an unintended block.
  if [[ $(grep -c '{n_x=' "$card") -ne 1 ]]; then
    echo "expected exactly one PARAMETERS block beginning with {n_x=" >&2
    exit 2
  fi
  sed "0,/{n_x=/s//{${fragment}n_x=/" "$card" >"$destination"
}

normalize_result()
{
  grep -vE \
    'collision_schedule_requested|^sparse_cells =|compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
    "$1"
}

run_case()
{
  local name="$1" fragment="$2"
  mkdir -p "$output/$name"
  inject "$output/$name/acc.dat" "$fragment"
  (
    cd "$output/$name"
    /usr/bin/time -f '%e' -o wall_seconds.txt \
      taskset -c "$cpu_list" "$binary" fccz p_fccz result.out \
      >stdout.txt 2>stderr.txt
    normalize_result result.out >result.normalized
  )
}

run_case strict \
  "compute_backend=\"serial\";${common}collision_schedule=\"wavefront_strict\";"

for threads in "${workers[@]}"; do
  if [[ "$threads" -lt 2 ]]; then
    echo "parallel benchmark workers must be >=2" >&2
    exit 2
  fi
  run_case "parallel-w${threads}" \
    "compute_backend=\"cpu\";cpu_threads=${threads};cpu_variant=\"pool_dual\";${common}collision_schedule=\"wavefront_parallel\";"
done

reference="$output/strict"
mapfile -t reference_outputs < <(
  cd "$reference"
  find . -maxdepth 1 -type f \
    ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
    ! -name wall_seconds.txt ! -name result.out \
    ! -name result.normalized -printf '%f\n' | sort
)
for threads in "${workers[@]}"; do
  candidate="$output/parallel-w${threads}"
  cmp "$reference/result.normalized" "$candidate/result.normalized"
  for produced in "${reference_outputs[@]}"; do
    cmp "$reference/$produced" "$candidate/$produced"
  done
  grep -Eq \
    'actual=wavefront_parallel_staged_fields.*diagonals=319.*waves=319.*encounters=25600.*maximum_wave_width=160.*prefetched_field_encounters=25600.*parallel_field_encounters=25598.*full_parallel_make_step_encounters=0' \
    "$candidate/stderr.txt"
  grep -q "field_workers=${threads}" "$candidate/stderr.txt"
done

{
  printf 'case\tworkers\twall_seconds\tspeedup_vs_strict\n'
  strict_seconds="$(<"$reference/wall_seconds.txt")"
  printf 'strict\t1\t%s\t1.000000\n' "$strict_seconds"
  for threads in "${workers[@]}"; do
    seconds="$(<"$output/parallel-w${threads}/wall_seconds.txt")"
    speedup="$(awk -v oracle="$strict_seconds" -v candidate="$seconds" \
      'BEGIN { printf "%.6f", oracle/candidate }')"
    printf 'parallel\t%s\t%s\t%s\n' "$threads" "$seconds" "$speedup"
  done
} >"$output/timings.tsv"

{
  printf 'binary=%s\n' "$binary"
  printf 'card=%s\n' "$card"
  printf 'cpu_list=%s\n' "$cpu_list"
  sha256sum "$binary" "$card"
  uname -a
} >"$output/provenance.txt"

cat "$output/timings.tsv"
echo "PASS: all normalized physics and generated data files are byte-identical"
