#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 FEEBL_EXECUTABLE [RESULT_DIRECTORY]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../.." && pwd -P)
executable=$(realpath -- "$1")
fixture="$source_root/examples/adaptive_grid/tiny_mode1_validation.dat"
[[ -x "$executable" ]] || { echo "not executable: $executable" >&2; exit 2; }
[[ -f "$fixture" ]] || { echo "missing fixture: $fixture" >&2; exit 2; }

if [[ $# -eq 2 ]]; then
  result_root=$2
  mkdir -p -- "$result_root"
  result_root=$(cd -- "$result_root" && pwd -P)
else
  result_root=$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode1-process.XXXXXXXX")
fi

pass_count=0

assert_no_physics_artifact() {
  local directory=$1
  local artifact
  for artifact in out.dat lumi.ee.out lumi.ee.out.fne \
                  lumi.ee.out.dLdW.fne; do
    if [[ -s "$directory/$artifact" ]]; then
      echo "fail-closed case produced nonempty $artifact in $directory" >&2
      exit 1
    fi
  done
}

make_card() {
  local destination=$1
  local block=$2
  while IFS= read -r line; do
    if [[ "$line" == "# ADAPTIVE_GRID_OPTIONS" ]]; then
      printf '%s\n' "$block"
    else
      printf '%s\n' "$line"
    fi
  done < "$fixture" > "$destination"
}

run_ok() {
  local name=$1
  local block=$2
  local directory="$result_root/$name"
  mkdir -p -- "$directory"
  make_card "$directory/acc.dat" "$block"
  echo "RUN  $name"
  (
    cd -- "$directory"
    "$executable" fccz p_fccz out.dat > log.txt 2>&1
  )
  ((pass_count += 1))
}

run_fail() {
  local name=$1
  local block=$2
  local expected=$3
  local directory="$result_root/$name"
  mkdir -p -- "$directory"
  make_card "$directory/acc.dat" "$block"
  echo "XFAIL $name (expected refusal)"
  if (
    cd -- "$directory"
    "$executable" fccz p_fccz out.dat > log.txt 2>&1
  ); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  if ! grep -Eiq -- "$expected" "$directory/log.txt"; then
    echo "$name failed for the wrong reason; expected /$expected/" >&2
    tail -80 "$directory/log.txt" >&2
    exit 1
  fi
  assert_no_physics_artifact "$directory"
  ((pass_count += 1))
}

compare_physics() {
  local reference=$1
  local candidate=$2
  local label=$3
  local artifacts=(lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne)
  for artifact in "${artifacts[@]}"; do
    [[ -f "$reference/$artifact" && -f "$candidate/$artifact" ]] || {
      echo "$label missing physics artifact $artifact" >&2
      exit 1
    }
    cmp -- "$reference/$artifact" "$candidate/$artifact" || {
      echo "$label is not byte-identical for $artifact" >&2
      exit 1
    }
  done
  ((pass_count += 1))
}

mode1_common='adaptive_grid_mode=1;adaptive_grid_scope="encounter";
adaptive_grid_min_nx=8;adaptive_grid_min_ny=8;
adaptive_grid_max_nx=32;adaptive_grid_max_ny=32;
adaptive_grid_extra_guard_cells=0;adaptive_grid_allow_full_window=1;
adaptive_grid_max_cells=1024;adaptive_grid_max_bytes=67108864;
adaptive_grid_profile="summary";'

# 1. No adaptive option at all: this is the parser/default and Mode-0 seam.
run_ok parser_defaults ''
grep -q 'adaptive_grid_requested=0 adaptive_grid_actual=0' \
  "$result_root/parser_defaults/log.txt"
if grep -q 'FEEBL adaptive grid summary:' \
    "$result_root/parser_defaults/log.txt"; then
  echo "Mode 0 unexpectedly entered adaptive execution" >&2
  exit 1
fi
((pass_count += 1))

# 2. Canonical-sized Mode 1 exercises the explicit full-window topology.
run_ok mode1_full "$mode1_common
adaptive_grid_size_classes_x=\"32\";
adaptive_grid_size_classes_y=\"32\";
adaptive_grid_schedule_save=\"schedule.ag1\";"
grep -Eq 'adaptive_grid_actual=1|adaptive_grid_actual = 1' \
  "$result_root/mode1_full/log.txt" "$result_root/mode1_full/out.dat"
grep -Eq 'full_window_selections=[1-9][0-9]*' \
  "$result_root/mode1_full/log.txt"
((pass_count += 1))

# 3. A multi-class crop, then a replay of its immutable schedule.
run_ok mode1_crop "$mode1_common
adaptive_grid_size_classes_x=\"8,12,16,24,32\";
adaptive_grid_size_classes_y=\"8,12,16,24,32\";
adaptive_grid_schedule_save=\"schedule.ag1\";"
grep -Eq 'canonical_cells_avoided=[1-9][0-9]*' \
  "$result_root/mode1_crop/log.txt"
grep -q 'fallback_count=0' "$result_root/mode1_crop/log.txt"
grep -q 'cold_miss_count=0' "$result_root/mode1_crop/log.txt"
((pass_count += 1))

compare_physics "$result_root/parser_defaults" "$result_root/mode1_full" \
  'Mode-0/full-window'
compare_physics "$result_root/mode1_full" "$result_root/mode1_crop" \
  'full-window/crop'

mkdir -p -- "$result_root/mode1_replay"
cp -- "$result_root/mode1_crop/schedule.ag1" \
  "$result_root/mode1_replay/schedule.ag1"
make_card "$result_root/mode1_replay/acc.dat" "$mode1_common
adaptive_grid_size_classes_x=\"8,12,16,24,32\";
adaptive_grid_size_classes_y=\"8,12,16,24,32\";
adaptive_grid_schedule_load=\"schedule.ag1\";"
echo 'RUN  mode1_replay'
(
  cd -- "$result_root/mode1_replay"
  "$executable" fccz p_fccz out.dat > log.txt 2>&1
)
grep -q 'action=replay-validated' "$result_root/mode1_replay/log.txt"
compare_physics "$result_root/mode1_crop" "$result_root/mode1_replay" \
  'schedule save/replay'
((pass_count += 1))

# 4. Parser and cross-parameter failures must be nonzero and diagnostic.
run_fail invalid_mode 'adaptive_grid_mode=9;' \
  'adaptive_grid_mode.*must be 0, 1, 2, or 3'
run_fail invalid_class_order "$mode1_common
adaptive_grid_size_classes_x=\"12,8\";
adaptive_grid_size_classes_y=\"8,12\";" \
  'adaptive_grid_size_classes_x.*strictly increasing'
run_fail invalid_minmax 'adaptive_grid_mode=1;
adaptive_grid_min_nx=24;adaptive_grid_max_nx=16;' \
  'adaptive_grid_min_nx.*must not exceed'
run_fail invalid_scope "$mode1_common
adaptive_grid_scope=\"wave\";
adaptive_grid_size_classes_x=\"8,16,32\";
adaptive_grid_size_classes_y=\"8,16,32\";" \
  'scope=encounter|wave scope is diagnostic'
run_fail mode2_fail_closed 'adaptive_grid_mode=2;' \
  'adaptive_grid_field_spacing_classes.*required'
run_fail mode3_fail_closed 'adaptive_grid_mode=3;' \
  'adaptive_grid_field_spacing_classes.*required'
run_fail conflicting_schedule_paths "$mode1_common
adaptive_grid_size_classes_x=\"8,16,32\";
adaptive_grid_size_classes_y=\"8,16,32\";
adaptive_grid_schedule_save=\"a.ag1\";
adaptive_grid_schedule_load=\"b.ag1\";" \
  'schedule save and replay cannot target the same run'
run_fail impossible_memory_cap "$mode1_common
adaptive_grid_max_bytes=1;
adaptive_grid_size_classes_x=\"8,16,32\";
adaptive_grid_size_classes_y=\"8,16,32\";" \
  'memory budget|configured budget|exceeds.*bytes|MemoryBudget'

# 5. A valid saved file must reject both content corruption and a controller
# fingerprint mismatch before any collision result is accepted.
mkdir -p -- "$result_root/corrupt_schedule"
cp -- "$result_root/mode1_crop/schedule.ag1" \
  "$result_root/corrupt_schedule/schedule.ag1"
sed -i '4s/\t0$/\t1/' "$result_root/corrupt_schedule/schedule.ag1"
make_card "$result_root/corrupt_schedule/acc.dat" "$mode1_common
adaptive_grid_size_classes_x=\"8,12,16,24,32\";
adaptive_grid_size_classes_y=\"8,12,16,24,32\";
adaptive_grid_schedule_load=\"schedule.ag1\";"
echo 'XFAIL corrupt_schedule (expected refusal)'
if (
  cd -- "$result_root/corrupt_schedule"
  "$executable" fccz p_fccz out.dat > log.txt 2>&1
); then
  echo 'corrupted schedule unexpectedly succeeded' >&2
  exit 1
fi
grep -Eiq 'hash|corrupt|mismatch|invalid' \
  "$result_root/corrupt_schedule/log.txt"
assert_no_physics_artifact "$result_root/corrupt_schedule"
((pass_count += 1))

mkdir -p -- "$result_root/incompatible_schedule"
cp -- "$result_root/mode1_crop/schedule.ag1" \
  "$result_root/incompatible_schedule/schedule.ag1"
make_card "$result_root/incompatible_schedule/acc.dat" "$mode1_common
adaptive_grid_extra_guard_cells=1;
adaptive_grid_size_classes_x=\"8,12,16,24,32\";
adaptive_grid_size_classes_y=\"8,12,16,24,32\";
adaptive_grid_schedule_load=\"schedule.ag1\";"
echo 'XFAIL incompatible_schedule (expected refusal)'
if (
  cd -- "$result_root/incompatible_schedule"
  "$executable" fccz p_fccz out.dat > log.txt 2>&1
); then
  echo 'incompatible schedule unexpectedly succeeded' >&2
  exit 1
fi
grep -Eiq 'fingerprint|incompatible|mismatch|header' \
  "$result_root/incompatible_schedule/log.txt"
assert_no_physics_artifact "$result_root/incompatible_schedule"
((pass_count += 1))

echo "mode-1 sequential process validation PASS checks=$pass_count results=$result_root"
