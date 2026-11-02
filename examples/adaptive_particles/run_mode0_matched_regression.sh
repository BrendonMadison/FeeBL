#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BASELINE_FEEBL FINAL_FEEBL CARD RESULT_DIRECTORY" >&2
  exit 2
fi

baseline=$(realpath -- "$1")
final=$(realpath -- "$2")
card=$(realpath -- "$3")
result=$4

if [[ ! -x "$baseline" || ! -x "$final" || ! -f "$card" ]]; then
  echo "baseline, final executable, or card is unavailable" >&2
  exit 2
fi
if [[ -e "$result" ]]; then
  echo "refusing to overwrite existing result directory: $result" >&2
  exit 2
fi
mkdir -p -- "$result/baseline" "$result/final"
result=$(cd -- "$result" && pwd -P)
cp -- "$card" "$result/baseline/acc.dat"
cp -- "$card" "$result/final/acc.dat"

run_one() {
  local executable=$1 directory=$2
  (
    cd -- "$directory"
    /usr/bin/time -v "$executable" fccz p_fccz out.dat \
      >log.txt 2>time.txt
  )
  [[ -s "$directory/out.dat" ]] || {
    echo "matched Mode-0 run did not publish out.dat: $directory" >&2
    exit 1
  }
}

# Deliberately sequential: timing overlap would invalidate even the diagnostic
# wall-time record and could perturb the strict wavefront resource path.
run_one "$baseline" "$result/baseline"
run_one "$final" "$result/final"

normalize_output() {
  sed -E \
    -e 's/field_setup_ms = [^ ]+/field_setup_ms = <timing>/' \
    -e 's/field_prefetch_ms = [^ ]+/field_prefetch_ms = <timing>/' \
    -e 's/ordered_commit_ms = [^ ]+/ordered_commit_ms = <timing>/' \
    "$1" > "$2"
}
normalize_output "$result/baseline/out.dat" \
                 "$result/baseline/out.normalized.dat"
normalize_output "$result/final/out.dat" \
                 "$result/final/out.normalized.dat"

cmp -- "$result/baseline/acc.dat" "$result/final/acc.dat"
cmp -- "$result/baseline/log.txt" "$result/final/log.txt"
cmp -- "$result/baseline/out.normalized.dat" \
       "$result/final/out.normalized.dat"

{
  printf 'schema=FEEBL_ADAPTIVE_MACROPARTICLE_MODE0_MATCH_V1\n'
  printf 'status=PASS\n'
  printf 'comparison=physics_and_configuration_exact_after_timing_mask\n'
  printf 'masked_fields=field_setup_ms,field_prefetch_ms,ordered_commit_ms\n'
  printf 'baseline_executable=%s\n' "$baseline"
  printf 'final_executable=%s\n' "$final"
  printf 'card=%s\n' "$card"
  printf 'utc_finished=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$result/receipt.txt"

(
  cd -- "$result"
  find baseline final -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$result/SHA256SUMS.txt"

echo "Mode-0 matched regression PASS results=$result"
