#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: $0 FEEBL_EXECUTABLE [RESULT_DIRECTORY]" >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../.." && pwd -P)
executable=$(realpath -- "$1")
fixture="$source_root/examples/adaptive_grid/tiny_mode23_validation.dat"
[[ -x "$executable" && -f "$fixture" ]] || exit 2

if [[ $# -eq 2 ]]; then
  result_root=$2
  mkdir -p -- "$result_root"
  result_root=$(cd -- "$result_root" && pwd -P)
else
  result_root=$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode23-mechanism.XXXXXXXX")
fi

make_fixed_card() {
  local cells=$1 destination=$2
  sed \
    -e 's/# ADAPTIVE_GRID_OPTIONS/adaptive_grid_mode=0;adaptive_grid_profile="off";/' \
    -e "s/n_x=64;n_y=64/n_x=$cells;n_y=$cells/" \
    "$fixture" > "$destination"
}

run_case() {
  local name=$1 card=$2 directory="$result_root/$1"
  mkdir -p -- "$directory"
  rm -f -- "$directory/acc.dat" "$directory/log.txt" \
    "$directory/out.dat" "$directory/lumi.ee.out" \
    "$directory/lumi.ee.out.fne" "$directory/lumi.ee.out.dLdW.fne"
  cp -- "$card" "$directory/acc.dat"
  echo "RUN  $name"
  (cd -- "$directory" && "$executable" fccz p_fccz out.dat >log.txt 2>&1)
  [[ -s "$directory/out.dat" ]] || {
    echo "$name did not publish out.dat" >&2
    exit 1
  }
}

for cells in 64 128 256; do
  directory="$result_root/fixed$cells"
  mkdir -p -- "$directory"
  make_fixed_card "$cells" "$directory/card.dat"
  run_case "fixed$cells" "$directory/card.dat"
done
run_case mode2 "$source_root/examples/adaptive_grid/mode2_error_controlled_cpu.dat"
run_case mode3 "$source_root/examples/adaptive_grid/mode3_nested_patch_cpu.dat"

# Exercise the terminal-uniform state independently of the patch/promotion
# fixture.  The coherent source hierarchy legitimately changes which level
# satisfies the field estimator, so coverage is split across deterministic
# cards instead of depending on one accidental mix of encounter states.
terminal_directory="$result_root/mode3_terminal"
mkdir -p -- "$terminal_directory"
sed 's/adaptive_grid_tau_curvature=0.99;adaptive_grid_tau_field_gradient=0.015;/adaptive_grid_tau_curvature=0.99;adaptive_grid_tau_field_gradient=0.01;/' \
  "$source_root/examples/adaptive_grid/mode3_nested_patch_cpu.dat" \
  | sed 's/adaptive_grid_max_refinement_factor=0;/adaptive_grid_max_refinement_factor=4;/' \
  > "$terminal_directory/card.dat"
run_case mode3_terminal "$terminal_directory/card.dat"
grep -Eq 'mode3_terminal_uniform_encounters=[1-9][0-9]*' \
  "$result_root/mode3_terminal/log.txt"
grep -Eq 'mode3_patch_requests=0.*mode3_uniform_promotions=0' \
  "$result_root/mode3_terminal/log.txt"

python3 "$script_dir/compare_modes23_physics.py" mechanism \
  "$result_root/fixed64/out.dat" \
  "$result_root/fixed128/out.dat" \
  "$result_root/fixed256/out.dat" \
  "$result_root/mode2/out.dat" \
  "$result_root/mode3/out.dat" \
  "$result_root/mode2/log.txt" \
  "$result_root/mode3/log.txt" \
  "$result_root/mechanism_validation.json"
