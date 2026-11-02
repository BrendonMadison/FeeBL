#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_exe=${1:-python3}

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PYTHON_EXECUTABLE NEW_OUTPUT_DIRECTORY [PARTICLES] [REPEATS]" >&2
  exit 2
fi

requested_output=$2
particles=${3:-320}
repeats=${4:-5}
output_parent=$(dirname -- "$requested_output")
output_name=$(basename -- "$requested_output")
mkdir -p -- "$output_parent"
output_parent=$(cd -- "$output_parent" && pwd)
output_dir="$output_parent/$output_name"

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing evidence directory: $output_dir" >&2
  exit 2
fi
if [[ ! "$particles" =~ ^[0-9]+$ ]] || (( particles < 65 )); then
  echo "PARTICLES must be an integer >= 65" >&2
  exit 2
fi
if [[ ! "$repeats" =~ ^[0-9]+$ ]] || (( repeats < 1 )); then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi

log_dir=$(mktemp -d "$output_parent/.adaptive-particle-comparison-logs.XXXXXX")
failure_note() {
  if [[ -d "$log_dir" ]]; then
    echo "comparison failed; partial logs retained at $log_dir" >&2
  fi
}
trap failure_note EXIT

export PYTHONHASHSEED=0
export TZ=UTC
export LC_ALL=C

"$python_exe" "$script_dir/test_compare_algorithms.py" 2>&1 \
  | tee "$log_dir/tests.log"
"$python_exe" "$script_dir/compare_algorithms.py" \
  --output "$output_dir" \
  --particles "$particles" \
  --repeats "$repeats" 2>&1 \
  | tee "$log_dir/comparison.log"

mv -- "$log_dir/tests.log" "$output_dir/tests.log"
mv -- "$log_dir/comparison.log" "$output_dir/comparison.log"
rmdir -- "$log_dir"

{
  echo "schema=feebl.adaptive_particles.prototype_comparison_run.v1"
  echo "standalone_not_production_integration=true"
  echo "python_executable=$python_exe"
  "$python_exe" --version
  echo "PYTHONHASHSEED=$PYTHONHASHSEED"
  echo "TZ=$TZ"
  echo "LC_ALL=$LC_ALL"
  echo "particles=$particles"
  echo "repeats=$repeats"
  uname -a 2>/dev/null || true
} > "$output_dir/environment.txt"

(
  cd -- "$script_dir"
  sha256sum \
    compare_algorithms.py \
    test_compare_algorithms.py \
    run_comparison.sh
) > "$output_dir/source_hashes.sha256"

(
  cd -- "$output_dir"
  sha256sum \
    comparison.json \
    comparison.csv \
    failure_cases.csv \
    tests.log \
    comparison.log \
    environment.txt
) > "$output_dir/evidence_hashes.sha256"

(
  cd -- "$script_dir"
  sha256sum -c "$output_dir/source_hashes.sha256"
)
(
  cd -- "$output_dir"
  sha256sum -c evidence_hashes.sha256
)

trap - EXIT
echo "PASS: standalone comparison evidence retained at $output_dir"
