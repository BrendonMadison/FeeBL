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
  result_root=$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode23-pilot.XXXXXXXX")
fi

checks=0
make_card() {
  local destination=$1 block=$2
  while IFS= read -r line; do
    if [[ "$line" == "# ADAPTIVE_GRID_OPTIONS" ]]; then
      printf '%s\n' "$block"
    elif [[ "$line" == 'n_t=1;n_m='* ]]; then
      printf 'n_t=1;n_m=8192;\n'
    else
      printf '%s\n' "$line"
    fi
  done < "$fixture" > "$destination"
}

clear_artifacts() {
  local directory=$1
  rm -f -- "$directory/acc.dat" "$directory/log.txt" \
    "$directory/out.dat" "$directory/lumi.ee.out" \
    "$directory/lumi.ee.out.fne" "$directory/lumi.ee.out.dLdW.fne"
}

assert_no_physics_artifact() {
  local directory=$1 artifact
  for artifact in out.dat lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
    [[ ! -s "$directory/$artifact" ]] || {
      echo "invalid pilot card produced $artifact" >&2
      exit 1
    }
  done
}

run_invalid() {
  local name=$1 block=$2 expected=$3 directory="$result_root/$1"
  mkdir -p -- "$directory"
  clear_artifacts "$directory"
  make_card "$directory/acc.dat" "$block"
  if (cd -- "$directory" &&
      "$executable" fccz p_fccz out.dat >log.txt 2>&1); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  grep -Eiq -- "$expected" "$directory/log.txt" || {
    echo "$name failed for the wrong reason; expected /$expected/" >&2
    tail -60 "$directory/log.txt" >&2
    exit 1
  }
  if grep -q 'FEEBL_ADAPTIVE_GRID_CONFIG_V1' "$directory/log.txt"; then
    echo "$name emitted an accepted receipt for an invalid pilot card" >&2
    exit 1
  fi
  assert_no_physics_artifact "$directory"
  ((checks += 1))
}

run_pilot() {
  local name=$1 block=$2 mode=$3 directory="$result_root/$1" summary
  mkdir -p -- "$directory"
  clear_artifacts "$directory"
  make_card "$directory/acc.dat" "$block"
  (cd -- "$directory" &&
    "$executable" fccz p_fccz out.dat >log.txt 2>&1)

  grep -Eq 'FEEBL_ADAPTIVE_GRID_CONFIG_V1 .*configuration_status=accepted.*pilot_diagnostics=1.*physics_qualification_claim=0.*convergence_claim=0.*unmet_outputs_authorized=1.*terminal_unmet_policy=publish_finest_resource_admissible_uniform_unqualified' \
    "$directory/log.txt"
  grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*error_target_met=0' \
    "$directory/log.txt"
  summary=$(grep 'FEEBL adaptive grid summary:' "$directory/log.txt" | tail -1)
  [[ "$summary" =~ requested=$mode[[:space:]]+actual=$mode ]] || {
    echo "$name did not execute requested mode $mode: $summary" >&2
    exit 1
  }
  [[ "$summary" =~ pilot_diagnostics=1 ]] || exit 1
  [[ "$summary" =~ physics_qualification_claim=0 ]] || exit 1
  [[ "$summary" =~ convergence_claim=0 ]] || exit 1
  [[ "$summary" =~ pilot_output_status=unqualified_diagnostics ]] || exit 1
  [[ "$summary" =~ terminal_unmet_policy=publish_finest_resource_admissible_uniform_unqualified ]] || exit 1
  [[ "$summary" =~ capacity_max_cells=1048576 ]] || exit 1
  [[ "$summary" =~ capacity_max_bytes=2147483648 ]] || exit 1
  [[ "$summary" =~ capacity_max_refinement_factor=4 ]] || exit 1
  [[ "$summary" =~ mode2_unmet_error_encounters=([1-9][0-9]*) ]] || {
    echo "$name did not report an unmet terminal ladder: $summary" >&2
    exit 1
  }
  local unmet=${BASH_REMATCH[1]}
  [[ "$summary" =~ terminal_resource_ladder_unmet_encounters=$unmet([[:space:]]|$) ]] || {
    echo "$name terminal-unmet telemetry disagrees: $summary" >&2
    exit 1
  }
  [[ -s "$directory/out.dat" ]] || {
    echo "$name did not publish its explicitly unqualified artifact" >&2
    exit 1
  }
  grep -q 'adaptive_grid_physics_qualification_claim = 0 adaptive_grid_convergence_claim = 0 adaptive_grid_unmet_outputs_authorized = 1' \
    "$directory/out.dat"
  if grep -Eq '(^|[[:space:]])(physics_qualification_claim|convergence_claim)=1([[:space:]]|$)' \
      "$directory/log.txt"; then
    echo "$name falsely emitted a positive qualification/convergence claim" >&2
    exit 1
  fi
  ((checks += 14))
}

mode2='adaptive_grid_mode=2;adaptive_grid_scope="encounter";
adaptive_grid_size_classes_x="8,16,32,64";adaptive_grid_size_classes_y="8,16,32,64";
adaptive_grid_min_nx=8;adaptive_grid_min_ny=8;
adaptive_grid_max_nx=64;adaptive_grid_max_ny=64;
adaptive_grid_extra_guard_cells=1;adaptive_grid_allow_full_window=1;
adaptive_grid_max_cells=1048576;adaptive_grid_max_bytes=2147483648;
adaptive_grid_field_spacing_classes="1,2,4";
adaptive_grid_field_operator="legacy_cic12_v1";
adaptive_grid_max_refinement_factor=4;
adaptive_grid_source_model="canonical_cell_average_v1";
adaptive_grid_anchor_x=0;adaptive_grid_anchor_y=0;
adaptive_grid_monitor="charge_curvature_field_richardson_overlap";
adaptive_grid_error_target=1e-15;
adaptive_grid_tau_curvature=1e-15;adaptive_grid_tau_field_gradient=1e-15;
adaptive_grid_tau_richardson=1e-15;adaptive_grid_tau_tail=1e-15;
adaptive_grid_rho_floor=1e-30;adaptive_grid_field_floor=1e-30;
adaptive_grid_indicator_smoothing_passes=1;
adaptive_grid_indicator_reduction="connected_max";
adaptive_grid_refine_threshold=10.0;adaptive_grid_coarsen_threshold=0.35;
adaptive_grid_shrink_dwell=1;adaptive_grid_update_interval=1;
adaptive_grid_richardson_interval=1;
adaptive_grid_richardson_order_source="three_level";
adaptive_grid_richardson_order_table="";
adaptive_grid_min_neff=1;adaptive_grid_fail_on_unmet_error=0;
adaptive_grid_pilot_diagnostics=1;adaptive_grid_profile="encounter";'

mode3="$mode2
adaptive_grid_mode=3;adaptive_grid_max_levels=1;adaptive_grid_max_patches=1;
adaptive_grid_refinement_ratio=2;adaptive_grid_blocking_factor=8;
adaptive_grid_error_buffer_cells=2;adaptive_grid_transition_cells=4;
adaptive_grid_fine_residual_tolerance=0.003;"

run_pilot valid_unqualified_mode2 "$mode2" 2
run_pilot valid_unqualified_mode3 "$mode3" 3
grep -Eq 'mode3_patch_requests=0' \
  "$result_root/valid_unqualified_mode3/log.txt"
grep -Eq 'mode3_terminal_uniform_encounters=[1-9][0-9]*' \
  "$result_root/valid_unqualified_mode3/log.txt"
((checks += 2))

run_invalid missing_explicit_pilot \
  "${mode2/adaptive_grid_pilot_diagnostics=1;/adaptive_grid_pilot_diagnostics=0;}" \
  'fail_on_unmet_error.*only with explicit.*pilot_diagnostics=1'

run_invalid pilot_requires_encounter_profile \
  "${mode2/adaptive_grid_profile=\"encounter\"/adaptive_grid_profile=\"summary\"}" \
  'adaptive_grid_profile.*must be encounter'

run_invalid pilot_requires_nonaborting_policy \
  "${mode2/adaptive_grid_fail_on_unmet_error=0/adaptive_grid_fail_on_unmet_error=1}" \
  'adaptive_grid_pilot_diagnostics.*requires.*fail_on_unmet_error=0'

run_invalid invalid_pilot_boolean \
  "${mode2/adaptive_grid_pilot_diagnostics=1/adaptive_grid_pilot_diagnostics=2}" \
  'adaptive_grid_pilot_diagnostics.*must be 0 or 1'

printf 'Mode-2/3 unqualified pilot diagnostics validation passed (%d checks).\n' \
  "$checks"
