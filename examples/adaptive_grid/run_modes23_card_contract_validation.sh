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
  result_root=$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode23-card.XXXXXXXX")
fi

checks=0
make_card() {
  local destination=$1 block=$2 macro_particles=${3:-}
  while IFS= read -r line; do
    if [[ "$line" == "# ADAPTIVE_GRID_OPTIONS" ]]; then
      printf '%s\n' "$block"
    elif [[ -n "$macro_particles" && "$line" == 'n_t=1;n_m='* ]]; then
      printf 'n_t=1;n_m=%s;\n' "$macro_particles"
    else
      printf '%s\n' "$line"
    fi
  done < "$fixture" > "$destination"
}

assert_no_physics_artifact() {
  local directory=$1 artifact
  for artifact in out.dat lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
    [[ ! -s "$directory/$artifact" ]] || {
      echo "configuration-only case produced $artifact" >&2
      exit 1
    }
  done
}

run_invalid() {
  local name=$1 block=$2 expected=$3 directory="$result_root/$1"
  mkdir -p -- "$directory"
  rm -f -- "$directory/acc.dat" "$directory/log.txt" \
    "$directory/out.dat" "$directory/lumi.ee.out" \
    "$directory/lumi.ee.out.fne" "$directory/lumi.ee.out.dLdW.fne"
  make_card "$directory/acc.dat" "$block"
  if (cd -- "$directory" && "$executable" fccz p_fccz out.dat >log.txt 2>&1); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  grep -Eiq -- "$expected" "$directory/log.txt" || {
    echo "$name failed for the wrong reason; expected /$expected/" >&2
    tail -60 "$directory/log.txt" >&2
    exit 1
  }
  if grep -q 'FEEBL_ADAPTIVE_GRID_CONFIG_V1' "$directory/log.txt"; then
    echo "$name emitted an accepted receipt for an invalid card" >&2
    exit 1
  fi
  assert_no_physics_artifact "$directory"
  ((checks += 1))
}

run_runtime_invalid() {
  local name=$1 block=$2 expected=$3 directory="$result_root/$1"
  mkdir -p -- "$directory"
  rm -f -- "$directory/acc.dat" "$directory/log.txt" \
    "$directory/out.dat" "$directory/lumi.ee.out" \
    "$directory/lumi.ee.out.fne" "$directory/lumi.ee.out.dLdW.fne"
  make_card "$directory/acc.dat" "$block"
  if (cd -- "$directory" && "$executable" fccz p_fccz out.dat >log.txt 2>&1); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  grep -q 'FEEBL_ADAPTIVE_GRID_CONFIG_V1.*configuration_status=accepted' \
    "$directory/log.txt"
  grep -Eiq -- "$expected" "$directory/log.txt" || {
    echo "$name failed for the wrong runtime reason; expected /$expected/" >&2
    tail -60 "$directory/log.txt" >&2
    exit 1
  }
  assert_no_physics_artifact "$directory"
  ((checks += 1))
}

run_accepted() {
  local name=$1 block=$2 macro_particles=${3:-} directory="$result_root/$1"
  mkdir -p -- "$directory"
  rm -f -- "$directory/acc.dat" "$directory/log.txt" \
    "$directory/out.dat" "$directory/lumi.ee.out" \
    "$directory/lumi.ee.out.fne" "$directory/lumi.ee.out.dLdW.fne"
  make_card "$directory/acc.dat" "$block" "$macro_particles"
  (cd -- "$directory" && "$executable" fccz p_fccz out.dat >log.txt 2>&1)
  grep -q 'FEEBL_ADAPTIVE_GRID_CONFIG_V1.*configuration_status=accepted' "$directory/log.txt"
  grep -q 'routing_status=cpu_wavefront_integrated.*qualification_status=experimental' "$directory/log.txt"
  [[ -s "$directory/out.dat" ]] || {
    echo "$name did not publish its expected deterministic operator artifact" >&2
    exit 1
  }
  ((checks += 1))
}

assert_zero_tail_budget_receipts() {
  local logfile=$1
  awk '
    /^FEEBL_ADAPTIVE_ENCOUNTER_V1 / {
      ++encounters
      eligible = used = population = selected = "missing"
      for (field = 1; field <= NF; ++field) {
        split($field, value, "=")
        if (value[1] == "encounter_tail_budget_eligible") eligible = value[2]
        if (value[1] == "encounter_tail_budget_used") used = value[2]
        if (value[1] == "encounter_population_weight") population = value[2]
        if (value[1] == "encounter_tail_selected_fraction") selected = value[2]
      }
      if (eligible != "0" || used != "0" ||
          population == "missing" || !(population > 0.0) ||
          selected == "missing" || !(selected == 0.0))
        bad = 1
    }
    END { exit !(encounters > 0 && !bad) }
  ' "$logfile" || {
    echo "unexpected encounter-tail telemetry in $logfile" >&2
    exit 1
  }
}

mode2='adaptive_grid_mode=2;adaptive_grid_scope="encounter";
adaptive_grid_size_classes_x="8,16,32,64";adaptive_grid_size_classes_y="8,16,32,64";
adaptive_grid_min_nx=8;adaptive_grid_min_ny=8;
adaptive_grid_max_nx=64;adaptive_grid_max_ny=64;
adaptive_grid_extra_guard_cells=1;adaptive_grid_allow_full_window=1;
adaptive_grid_max_cells=1048576;adaptive_grid_max_bytes=2147483648;
adaptive_grid_field_spacing_classes="1,2,4";
adaptive_grid_source_model="canonical_cell_average_v1";
adaptive_grid_anchor_x=0;adaptive_grid_anchor_y=0;
adaptive_grid_monitor="charge_curvature_field_richardson_overlap";
adaptive_grid_error_target=0.9;
adaptive_grid_tau_curvature=0.99;adaptive_grid_tau_field_gradient=0.015;
adaptive_grid_tau_richardson=0.9;adaptive_grid_tau_tail=0.001;
adaptive_grid_rho_floor=1e-30;adaptive_grid_field_floor=1e-30;
adaptive_grid_indicator_smoothing_passes=1;
adaptive_grid_indicator_reduction="connected_max";
adaptive_grid_refine_threshold=10.0;adaptive_grid_coarsen_threshold=0.35;
adaptive_grid_shrink_dwell=1;adaptive_grid_update_interval=1;
adaptive_grid_richardson_interval=1;
adaptive_grid_richardson_order_source="three_level";
adaptive_grid_richardson_order_table="";
adaptive_grid_min_neff=1;adaptive_grid_fail_on_unmet_error=1;
adaptive_grid_profile="encounter";'

mode3="$mode2
adaptive_grid_mode=3;adaptive_grid_max_levels=1;adaptive_grid_max_patches=1;
adaptive_grid_refinement_ratio=2;adaptive_grid_blocking_factor=8;
adaptive_grid_error_buffer_cells=2;adaptive_grid_transition_cells=4;
adaptive_grid_fine_residual_tolerance=0.003;"

run_accepted valid_mode2 "$mode2"
grep -Eq 'field_spacing_classes="1,2,4".*error_target=0\.9[0-9]*([[:space:]]|$)' \
  "$result_root/valid_mode2/log.txt"
grep -q 'field_operator=legacy_cic12_v1.*max_refinement_factor=0' \
  "$result_root/valid_mode2/log.txt"
grep -q 'error_norm_policy=component_rms_linf_v1.*error_quantile=0.999.*error_catastrophe_limit=0.5.*noise_aware_stop=0' \
  "$result_root/valid_mode2/log.txt"
# The declared list is only a bootstrap.  A zero optional cap must publish at
# least one resource-admissible dynamic class beyond factor four even though
# this loose fixture converges before solving it.
grep -q 'FEEBL adaptive spacing class: factor=8 selected_encounters=0' \
  "$result_root/valid_mode2/log.txt"
grep -q 'coarsen_threshold=.*shrink_dwell=1.*update_interval=1.*richardson_interval=1' \
  "$result_root/valid_mode2/log.txt"
grep -q 'profile=encounter' "$result_root/valid_mode2/log.txt"
((checks += 6))

vector_mode2="$mode2
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";
adaptive_grid_error_quantile=0.999;
adaptive_grid_error_catastrophe_limit=2.0;
adaptive_grid_fail_on_unmet_error=0;
adaptive_grid_pilot_diagnostics=1;"
run_accepted valid_vector_policy_mode2 "$vector_mode2"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*error_norm_policy=vector_rms_quantile_v1.*component_linf_acceptance_gate=0.*vector_difference_rms=[0-9.eE+-]+.*vector_difference_quantile=[0-9.eE+-]+.*vector_difference_maximum=[0-9.eE+-]+.*vector_quantile_probability=0.999.*vector_accepted_targets=[1-9][0-9]*.*vector_above_target=[0-9]+.*vector_above_target_exposure=[0-9.eE+-]+.*vector_catastrophe_guard_passed=[01]' \
  "$result_root/valid_vector_policy_mode2/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*error_norm_policy=vector_rms_quantile_v1.*mode2_unmet_error_encounters=[0-9]+.*mode2_noise_aware_stop_encounters=0' \
  "$result_root/valid_vector_policy_mode2/log.txt"
((checks += 2))

# Mixed V2 controls are meaningful only when prospectively bound to an
# immutable kick/error-budget document. Two otherwise identical runs with
# different canonical digests must carry different adaptive schedule/run
# identities; merely printing the digest is not sufficient.
budget_sha_a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
budget_sha_b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
mixed_mode2="$mode2
adaptive_grid_error_norm_policy=\"vector_mixed_rms_quantile_v2\";
adaptive_grid_error_budget_sha256=\"$budget_sha_a\";
adaptive_grid_error_reference_field=12.5;
adaptive_grid_error_absolute_floor_fraction=0.001;
adaptive_grid_error_catastrophe_limit=2.0;
adaptive_grid_error_catastrophe_absolute_fraction=0.01;
adaptive_grid_error_exposure_limit=0.5;
adaptive_grid_fail_on_unmet_error=0;
adaptive_grid_pilot_diagnostics=1;"
run_accepted valid_mixed_policy_budget_a "$mixed_mode2"
grep -Eq "FEEBL_ADAPTIVE_GRID_CONFIG_V1 .*error_norm_policy=vector_mixed_rms_quantile_v2.*error_budget_sha256=$budget_sha_a" \
  "$result_root/valid_mixed_policy_budget_a/log.txt"
grep -Eq "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1 .*error_budget_sha256=$budget_sha_a" \
  "$result_root/valid_mixed_policy_budget_a/log.txt"

mixed_mode2_b="${mixed_mode2/$budget_sha_a/$budget_sha_b}"
run_accepted valid_mixed_policy_budget_b "$mixed_mode2_b"
grep -Eq "FEEBL_ADAPTIVE_GRID_CONFIG_V1 .*error_norm_policy=vector_mixed_rms_quantile_v2.*error_budget_sha256=$budget_sha_b" \
  "$result_root/valid_mixed_policy_budget_b/log.txt"
card_hash_a=$(sed -n 's/.* card_hash=\([0-9a-f]\{64\}\).*/\1/p' \
  "$result_root/valid_mixed_policy_budget_a/log.txt" | tail -1)
card_hash_b=$(sed -n 's/.* card_hash=\([0-9a-f]\{64\}\).*/\1/p' \
  "$result_root/valid_mixed_policy_budget_b/log.txt" | tail -1)
[[ ${#card_hash_a} -eq 64 && ${#card_hash_b} -eq 64 && \
   "$card_hash_a" != "$card_hash_b" ]] || {
  echo "mixed-V2 budget digest did not change the adaptive run identity" >&2
  exit 1
}
((checks += 4))

mode2_fd4="$mode2
adaptive_grid_field_operator=\"fd4_lagrange4_v1\";
adaptive_grid_max_refinement_factor=8;"
run_accepted valid_mode2_fd4 "$mode2_fd4"
grep -q 'field_operator=fd4_lagrange4_v1.*max_refinement_factor=8' \
  "$result_root/valid_mode2_fd4/log.txt"
grep -q 'requested=2 actual=2.*field_operator=fd4_lagrange4_v1' \
  "$result_root/valid_mode2_fd4/log.txt"
((checks += 2))

reference2="$mode2
adaptive_grid_field_spacing_classes=\"1,2,4,8\";
adaptive_grid_source_model=\"reference_cic_v1\";
adaptive_grid_source_reference_nx=128;
adaptive_grid_source_reference_ny=128;
adaptive_grid_source_error_target=0.9;
adaptive_grid_solver_error_target=0.9;
adaptive_grid_lumi_error_target=0.9;
adaptive_grid_lumi_floor=1e-300;"
run_accepted valid_reference_cic_mode2 "$reference2"
grep -Eq 'source_model=reference_cic_v1.*source_reference_nx=128.*lumi_floor=1e-300' \
  "$result_root/valid_reference_cic_mode2/log.txt"
grep -Eq 'source_reference_factor=2.*source_density_hash=[1-9][0-9]*.*source_sampling_hash=[1-9][0-9]*.*source_sampling_hierarchy_hash=[1-9][0-9]*.*source_budget_runtime_verified=0' \
  "$result_root/valid_reference_cic_mode2/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*source_beam1_in=[0-9]+.*source_beam1_out=[0-9]+.*source_beam2_in=[0-9]+.*source_beam2_out=[0-9]+' \
  "$result_root/valid_reference_cic_mode2/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*source_beam1_in_visits=[0-9]+.*source_beam1_out_visits=[0-9]+.*source_beam2_in_visits=[0-9]+.*source_beam2_out_visits=[0-9]+' \
  "$result_root/valid_reference_cic_mode2/log.txt"
# In this source-factor-two fixture, factor 1 has only the transition pair
# (1,2), so no same-regime three-level Richardson certificate exists.  It
# must refine instead of publishing an uncertified source-projection result.
grep -q 'FEEBL adaptive spacing class: factor=1 selected_encounters=0' \
  "$result_root/valid_reference_cic_mode2/log.txt"
((checks += 5))

# The 64x64 fixture spans [-36000,36000] nm x [-300,300] nm.  These
# physical half-widths are exactly one half of a finest-grid cell in each
# plane and, unlike a grid-relative shape, remain unchanged on every spacing
# class.  Reference dimensions are explicitly zero because this source has
# no hidden absolute reference raster.
bspline2="$mode2
adaptive_grid_source_model=\"fixed_width_bspline1_v1\";
adaptive_grid_source_reference_nx=0;
adaptive_grid_source_reference_ny=0;
adaptive_grid_source_width_x=562.5;
adaptive_grid_source_width_y=4.6875;
adaptive_grid_source_error_target=0.99;
adaptive_grid_solver_error_target=0.99;
adaptive_grid_lumi_error_target=0.99;
adaptive_grid_lumi_floor=1e-300;
adaptive_grid_encounter_tail_fraction=0.01;"
run_accepted valid_fixed_width_bspline1_mode2 "$bspline2" 262144
grep -Eq 'source_model=fixed_width_bspline1_v1.*source_reference_nx=0.*source_reference_ny=0.*source_width_x=562\.5.*source_width_y=4\.6875.*encounter_tail_fraction=0\.01[0-9]*([[:space:]]|$)' \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
grep -Eq 'source_reference_factor=0.*source_model_tag=4779251917745697024.*source_density_hash=[1-9][0-9]*.*source_sampling_hash=[1-9][0-9]*.*source_sampling_hierarchy_hash=[1-9][0-9]*.*source_budget_runtime_verified=0' \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*source_beam1_in=[0-9]+.*source_beam1_out=[0-9]+.*source_beam2_in=[0-9]+.*source_beam2_out=[0-9]+.*source_max_charge_closure=[0-9.eE+-]+.*source_max_projection_moment_shift=[0-9.eE+-]+' \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*source_model=fixed_width_bspline1_v1.*source_reference_factor=0.*source_density_hash_nonzero_encounters=[1-9][0-9]*.*source_sampling_hash_nonzero_encounters=[1-9][0-9]*.*source_sampling_hierarchy_hash_nonzero_encounters=[1-9][0-9]*.*source_beam1_in_visits=[0-9]+.*source_beam1_out_visits=[0-9]+.*source_beam2_in_visits=[0-9]+.*source_beam2_out_visits=[0-9]+' \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
assert_zero_tail_budget_receipts \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*mode2_tail_budget_eligible_encounters=0.*mode2_tail_budget_used_encounters=0.*encounter_tail_budget_selected_pairs=0.*encounter_tail_budget_selected_fraction=0([[:space:]]|$).*encounter_tail_field_metric=symmetric_vector_triplet_v1.*encounter_tail_weighted_richardson_error_diagnostic=0([[:space:]]|$).*encounter_tail_weighted_field_sensitivity=0([[:space:]]|$).*encounter_tail_weighted_source_lumi_error=0([[:space:]]|$).*encounter_tail_lumi_exposure_fraction=0([[:space:]]|$).*encounter_tail_budget_met=1([[:space:]]|$)' \
  "$result_root/valid_fixed_width_bspline1_mode2/log.txt"
((checks += 6))

# Four declared levels exercise the rolling three-level audit.  The fixture
# converges after factors 1/2/4, so factor 8 must remain unsolved and
# unselected; solving all four would regress to eager fixed-ladder work.
mode2_lazy="$mode2
adaptive_grid_field_spacing_classes=\"1,2,4,8\";
adaptive_grid_error_target=0.9;adaptive_grid_tau_richardson=0.9;
adaptive_grid_tau_field_gradient=0.99;"
run_accepted valid_mode2_lazy "$mode2_lazy"
lazy_summary=$(grep 'FEEBL adaptive grid summary:' \
  "$result_root/valid_mode2_lazy/log.txt" | tail -1)
lazy_encounters=$(sed -n 's/.* mode2_encounters=\([0-9][0-9]*\).*/\1/p' \
  <<<"$lazy_summary")
lazy_solves=$(sed -n 's/.* mode2_field_solves=\([0-9][0-9]*\).*/\1/p' \
  <<<"$lazy_summary")
[[ -n "$lazy_encounters" && -n "$lazy_solves" && \
   "$lazy_solves" -eq $((3 * lazy_encounters)) ]] || {
  echo "Mode 2 did not stop lazily after its three-level audit: $lazy_summary" >&2
  exit 1
}
grep -q 'FEEBL adaptive spacing class: factor=8 selected_encounters=0' \
  "$result_root/valid_mode2_lazy/log.txt"
((checks += 2))

run_accepted valid_mode3 "$mode3"
grep -q 'max_levels=1.*max_patches=1.*refinement_ratio=2.*blocking_factor=8' \
  "$result_root/valid_mode3/log.txt"
grep -q 'error_buffer_cells=2.*transition_cells=4.*fine_residual_tolerance=0.003' \
  "$result_root/valid_mode3/log.txt"
grep -q 'profile=encounter' "$result_root/valid_mode3/log.txt"
grep -q 'terminal_patch_certification=0' \
  "$result_root/valid_mode3/log.txt"
((checks += 4))

reference3="$mode3
adaptive_grid_field_spacing_classes=\"1,2,4,8\";
adaptive_grid_source_model=\"reference_cic_v1\";
adaptive_grid_source_reference_nx=128;
adaptive_grid_source_reference_ny=128;
adaptive_grid_source_error_target=0.9;
adaptive_grid_solver_error_target=0.9;
adaptive_grid_lumi_error_target=0.9;
adaptive_grid_lumi_floor=1e-300;"
run_accepted valid_reference_cic_mode3 "$reference3"
grep -Eq 'mode3_patch_requests=[1-9][0-9]*' \
  "$result_root/valid_reference_cic_mode3/log.txt"
grep -Eq 'source_reference_factor=2.*source_density_hash=[1-9][0-9]*' \
  "$result_root/valid_reference_cic_mode3/log.txt"
grep -Eq 'patch_lumi_oracle_evaluations=[1-9][0-9]*.*accepted_patch_lumi_oracle_evaluations=[1-9][0-9]*.*maximum_accepted_patch_lumi_oracle_error=[0-9.eE+-]+.*maximum_examined_patch_lumi_oracle_error=[0-9.eE+-]+' \
  "$result_root/valid_reference_cic_mode3/log.txt"
((checks += 3))

bspline3="$bspline2
adaptive_grid_mode=3;adaptive_grid_max_levels=1;adaptive_grid_max_patches=1;
adaptive_grid_refinement_ratio=2;adaptive_grid_blocking_factor=8;
adaptive_grid_error_buffer_cells=2;adaptive_grid_transition_cells=4;
adaptive_grid_fine_residual_tolerance=0.003;"
run_accepted valid_fixed_width_bspline1_mode3 "$bspline3" 262144
grep -Eq 'source_model=fixed_width_bspline1_v1.*source_reference_nx=0.*source_reference_ny=0.*source_width_x=562\.5.*source_width_y=4\.6875.*encounter_tail_fraction=0\.01[0-9]*([[:space:]]|$)' \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
grep -Eq 'source_reference_factor=0.*source_model_tag=4779251917745697024.*source_density_hash=[1-9][0-9]*.*source_sampling_hash=[1-9][0-9]*.*source_sampling_hierarchy_hash=[1-9][0-9]*' \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*source_beam1_in=[0-9]+.*source_beam1_out=[0-9]+.*source_beam2_in=[0-9]+.*source_beam2_out=[0-9]+' \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
grep -Eq 'patch_lumi_oracle_evaluations=[1-9][0-9]*.*accepted_patch_lumi_oracle_evaluations=[1-9][0-9]*.*maximum_accepted_patch_lumi_oracle_error=[0-9.eE+-]+.*maximum_examined_patch_lumi_oracle_error=[0-9.eE+-]+' \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
assert_zero_tail_budget_receipts \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*mode2_tail_budget_eligible_encounters=0.*mode2_tail_budget_used_encounters=0.*encounter_tail_budget_selected_pairs=0.*encounter_tail_budget_selected_fraction=0([[:space:]]|$).*encounter_tail_field_metric=symmetric_vector_triplet_v1.*encounter_tail_weighted_richardson_error_diagnostic=0([[:space:]]|$).*encounter_tail_weighted_field_sensitivity=0([[:space:]]|$).*encounter_tail_weighted_source_lumi_error=0([[:space:]]|$).*encounter_tail_lumi_exposure_fraction=0([[:space:]]|$).*encounter_tail_budget_met=1([[:space:]]|$)' \
  "$result_root/valid_fixed_width_bspline1_mode3/log.txt"
((checks += 6))

# Tighten only the patch/uniform luminosity oracle far below roundoff.  The
# registered patch source remains exact; the deliberately impossible receipt
# threshold must therefore exercise bit 0x80 and safe uniform promotion.
reference3_lumi_oracle="$reference3
adaptive_grid_lumi_error_target=1e-30;"
run_accepted valid_reference_cic_mode3_lumi_oracle "$reference3_lumi_oracle"
grep -Eq 'nested_patch_promoted_uniform=1.*nested_patch_rejection_mask=(12[89]|1[3-9][0-9]|2[0-5][0-9]).*nested_lumi_error=[1-9][0-9.eE+-]*' \
  "$result_root/valid_reference_cic_mode3_lumi_oracle/log.txt"
grep -Eq 'patch_lumi_oracle_evaluations=[1-9][0-9]*.*maximum_examined_patch_lumi_oracle_error=[1-9][0-9.eE+-]*' \
  "$result_root/valid_reference_cic_mode3_lumi_oracle/log.txt"
((checks += 2))

# Coherent-source prolongation changes the field-error ordering enough that a
# single statistical fixture should not be expected to hit patch, promotion,
# and terminal-uniform states simultaneously.  Retain explicit terminal-path
# coverage with a deterministic gradient gate that forces the already-declared
# factor-4 class while preserving every field/Richardson/noise acceptance gate.
# The explicit factor-4 cap is local to this negative-path fixture: production
# cards use cap zero and continue extending the ladder while resources permit.
mode3_terminal="$mode3
adaptive_grid_tau_field_gradient=0.01;
adaptive_grid_max_refinement_factor=4;"
run_accepted valid_mode3_terminal "$mode3_terminal"
grep -Eq 'mode3_terminal_uniform_encounters=[1-9][0-9]*' \
  "$result_root/valid_mode3_terminal/log.txt"
grep -Eq 'mode3_patch_requests=0.*mode3_uniform_promotions=0' \
  "$result_root/valid_mode3_terminal/log.txt"
((checks += 2))

# The experimental terminal route must be explicit and must retain an
# unqualified-pilot outcome when any part of its composite proof rejects.
# This fixture forces the factor-four uniform terminal and verifies that the
# explicit factor-four cap rejects the proposed factor-eight/factor-sixteen
# patches before a patch field allocation or solve.  The attempt and uniform
# fallback must remain fully visible in telemetry.
terminal_patch3="$bspline3
adaptive_grid_field_operator=\"fd4_lagrange4_v1\";
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";
adaptive_grid_error_quantile=0.999;
adaptive_grid_error_catastrophe_limit=2.0;
adaptive_grid_tau_field_gradient=0.01;
adaptive_grid_max_refinement_factor=4;
adaptive_grid_fail_on_unmet_error=0;
adaptive_grid_pilot_diagnostics=1;
adaptive_grid_terminal_patch_certification=1;"
run_accepted valid_mode3_terminal_patch_smoke "$terminal_patch3" 262144
grep -Eq 'FEEBL_ADAPTIVE_GRID_CONFIG_V1 .*terminal_patch_certification=1' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*terminal_patch_attempted=1.*terminal_patch_accepted=[01].*terminal_patch_rejection_mask=[0-9]+.*terminal_patch_geometry_hash_r=[0-9]+.*terminal_patch_geometry_hash_r2=[0-9]+' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ENCOUNTER_V1 .*nested_coarse_cells=0.*nested_fine_cells=0.*terminal_patch_attempted=1.*terminal_patch_accepted=0.*terminal_patch_used=0.*terminal_patch_uniform_fallback=1' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
awk '
  /^FEEBL_ADAPTIVE_ENCOUNTER_V1 / {
    attempted = used = fallback = mask = factor_r = factor_r2 = -1
    coarse_cells = fine_cells = -1
    for (field = 1; field <= NF; ++field) {
      split($field, value, "=")
      if (value[1] == "terminal_patch_attempted") attempted = value[2]
      if (value[1] == "terminal_patch_used") used = value[2]
      if (value[1] == "terminal_patch_uniform_fallback") fallback = value[2]
      if (value[1] == "terminal_patch_rejection_mask") mask = value[2]
      if (value[1] == "terminal_patch_factor_r") factor_r = value[2]
      if (value[1] == "terminal_patch_factor_r2") factor_r2 = value[2]
      if (value[1] == "nested_coarse_cells") coarse_cells = value[2]
      if (value[1] == "nested_fine_cells") fine_cells = value[2]
    }
    if (attempted == 1 && used == 0 && fallback == 1 &&
        factor_r == 8 && factor_r2 == 16 &&
        int(mask / 4096) % 2 == 1 &&
        coarse_cells == 0 && fine_cells == 0)
      capped_without_solve = 1
  }
  END { exit !capped_without_solve }
' "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*mode3_terminal_patch_attempts=[1-9][0-9]*.*mode3_terminal_patch_(accepts|rejects)=[1-9][0-9]*.*mode3_terminal_patch_rejection_mask_or=[0-9]+' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
grep -Eq 'FEEBL adaptive grid summary: .*mode3_terminal_patch_attempts=[1-9][0-9]*.*mode3_terminal_patch_accepts=0.*mode3_terminal_patch_used=0.*mode3_terminal_patch_rejects=[1-9][0-9]*.*mode3_terminal_patch_uniform_fallbacks=[1-9][0-9]*.*mode3_terminal_patch_rejects_with_resource=[1-9][0-9]*.*mode3_attempted_patch_coarse_cells=0.*mode3_attempted_patch_fine_cells=0' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
grep -Eq '^FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1 schema=feebl\.adaptive-grid-certificate\.v1 requested_mode=3 actual_mode=3 error_norm_policy=vector_rms_quantile_v1 source_model=fixed_width_bspline1_v1 error_budget_sha256=not_applicable .*terminal_patch_enabled=1 terminal_patch_attempts=[1-9][0-9]* terminal_patch_accepts=[0-9]+ terminal_patch_rejects=[0-9]+ terminal_patch_rejection_mask_or=[0-9]+ external_replay_recommended=[01] all_encounters_certified=[01] disposition=(accepted_all_encounters|unqualified_diagnostics)$' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
[[ $(grep -c '^FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1 ' \
       "$result_root/valid_mode3_terminal_patch_smoke/log.txt") -eq 1 ]]
grep -Eq '^FEEBL_ADAPTIVE_GRID_UNRESOLVED_V1 schema=feebl\.adaptive-grid-unresolved\.v1 unresolved_non_noise_encounters=[0-9]+ legacy_resource_unmet_alias=[0-9]+ terminal_patch_attempts=[1-9][0-9]* terminal_patch_accepts=0 terminal_patch_used=0 terminal_patch_rejects=[1-9][0-9]* terminal_patch_uniform_fallbacks=[1-9][0-9]* terminal_patch_rejects_with_resource=[1-9][0-9]* terminal_patch_rejects_with_nonresource=[0-9]+ terminal_patch_rejection_mask_or=[1-9][0-9]*$' \
  "$result_root/valid_mode3_terminal_patch_smoke/log.txt"
((checks += 9))

# This fixture preserves the production field/oracle gates and changes only
# the harmonic stopping tolerance.  Its nontrivial deterministic patch has a
# residual well above 1e-18, so it must retain attempted-work telemetry, skip
# the field oracle, and publish the already-qualified uniform-fine solution.
mode3_residual="$mode3
adaptive_grid_tau_field_gradient=0.02;
adaptive_grid_fine_residual_tolerance=1e-18;"
run_accepted valid_mode3_residual "$mode3_residual"
grep -Eq 'nested_patch_promoted_uniform=1.*nested_patch_rejection_mask=32.*nested_coarse_cells=[1-9][0-9]*.*nested_residual=[1-9][0-9.eE+-]*.*nested_field_error_rms=0.*nested_field_error_linf=0.*nested_iterations=[1-9][0-9]*' \
  "$result_root/valid_mode3_residual/log.txt"
grep -Eq 'mode3_attempted_patch_coarse_cells=[1-9][0-9]*.*mode3_harmonic_iterations=[1-9][0-9]*.*mode3_rejection_mask_or=[0-9]+' \
  "$result_root/valid_mode3_residual/log.txt"
((checks += 2))

python3 "$script_dir/compare_modes23_physics.py" validate-logs \
  "$result_root/valid_mode2/log.txt" \
  "$result_root/valid_mode3/log.txt" \
  "$result_root/mechanism_telemetry_validation.json" \
  --mode2-lazy-log "$result_root/valid_mode2_lazy/log.txt" \
  --mode3-residual-log "$result_root/valid_mode3_residual/log.txt" \
  --reference-mode2-log "$result_root/valid_reference_cic_mode2/log.txt" \
  --reference-mode3-log "$result_root/valid_reference_cic_mode3/log.txt" \
  --reference-mode3-lumi-oracle-log \
    "$result_root/valid_reference_cic_mode3_lumi_oracle/log.txt"
python3 "$script_dir/compare_modes23_physics.py" \
  rejection-receipt-self-test
((checks += 2))

run_invalid missing_mode2_contract 'adaptive_grid_mode=2;' \
  'adaptive_grid_field_spacing_classes.*required'
run_invalid invalid_field_operator "$mode2
adaptive_grid_field_operator=\"bicubic_guess\";" \
  'adaptive_grid_field_operator.*legacy_cic12_v1.*fd4_lagrange4_v1'
run_invalid invalid_error_norm_policy "$mode2
adaptive_grid_error_norm_policy=\"mean_only_guess\";" \
  'adaptive_grid_error_norm_policy.*component_rms_linf_v1.*vector_rms_quantile_v1'
run_invalid low_error_quantile "$mode2
adaptive_grid_error_quantile=0.9;" \
  'adaptive_grid_error_quantile.*high quantile'
run_invalid excessive_catastrophe_limit "$mode2
adaptive_grid_error_catastrophe_limit=2.01;" \
  'adaptive_grid_error_catastrophe_limit.*\(0,2]'
run_invalid mixed_budget_missing "$mode2
adaptive_grid_error_norm_policy=\"vector_mixed_rms_quantile_v2\";
adaptive_grid_error_reference_field=12.5;
adaptive_grid_error_absolute_floor_fraction=0.001;
adaptive_grid_error_catastrophe_absolute_fraction=0.01;
adaptive_grid_error_exposure_limit=0.5;" \
  'adaptive_grid_error_budget_sha256.*64 lowercase hexadecimal'
run_invalid mixed_budget_short "$mode2
adaptive_grid_error_norm_policy=\"vector_mixed_rms_quantile_v2\";
adaptive_grid_error_budget_sha256=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";
adaptive_grid_error_reference_field=12.5;
adaptive_grid_error_absolute_floor_fraction=0.001;
adaptive_grid_error_catastrophe_absolute_fraction=0.01;
adaptive_grid_error_exposure_limit=0.5;" \
  'adaptive_grid_error_budget_sha256.*64 lowercase hexadecimal'
run_invalid mixed_budget_uppercase "$mode2
adaptive_grid_error_norm_policy=\"vector_mixed_rms_quantile_v2\";
adaptive_grid_error_budget_sha256=\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\";
adaptive_grid_error_reference_field=12.5;
adaptive_grid_error_absolute_floor_fraction=0.001;
adaptive_grid_error_catastrophe_absolute_fraction=0.01;
adaptive_grid_error_exposure_limit=0.5;" \
  'adaptive_grid_error_budget_sha256.*64 lowercase hexadecimal'
run_invalid nonmixed_budget_illegal "$mode2
adaptive_grid_error_budget_sha256=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";" \
  'adaptive_grid_error_budget_sha256/reference_field.*remain empty/zero'
run_invalid vector_policy_calibrated_order "$mode2
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";
adaptive_grid_richardson_order_source=\"calibrated_table\";
adaptive_grid_richardson_order_table=\"1:1,2:1,4:1\";" \
  'vector quantile policies require rolling three_level'
run_invalid noise_stop_without_replay_controller "$mode2
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";
adaptive_grid_noise_aware_stop=1;
adaptive_grid_pilot_diagnostics=1;
adaptive_grid_fail_on_unmet_error=0;" \
  'noise_aware_stop.*replay contract'
run_invalid non_power_refinement_cap "$mode2
adaptive_grid_max_refinement_factor=12;" \
  'adaptive_grid_max_refinement_factor.*power of two'
run_invalid refinement_cap_below_bootstrap "$mode2
adaptive_grid_max_refinement_factor=2;" \
  'adaptive_grid_max_refinement_factor.*final bootstrap'
run_invalid invalid_spacing_factor "$mode2
adaptive_grid_field_spacing_classes=\"1,3,4\";" \
  'field_spacing_classes.*powers of two'
run_invalid skipped_spacing_factor "$mode2
adaptive_grid_field_spacing_classes=\"1,4,8\";
adaptive_grid_richardson_order_table=\"1:1,4:1,8:1\";" \
  'field_spacing_classes.*consecutive ratio-two'
run_invalid invalid_hysteresis "$mode2
adaptive_grid_coarsen_threshold=10.0;" \
  'coarsen_threshold.*below the refine threshold'
run_invalid invalid_overlap_fraction "$mode2
adaptive_grid_coarsen_threshold=1.01;" \
  'coarsen_threshold.*(must not exceed|at most) 1'
run_invalid missing_component_tolerance "$mode2
adaptive_grid_tau_tail=0;" \
  'tau_tail.*required.*\(0,1\)'
run_invalid incomplete_order_table "$mode2
adaptive_grid_richardson_order_source=\"calibrated_table\";
adaptive_grid_richardson_order_table=\"1:1,2:1\";" \
  'order_table.*exactly one entry'
run_invalid insufficient_three_level "$mode2
adaptive_grid_field_spacing_classes=\"1,2\";
adaptive_grid_richardson_order_source=\"three_level\";
adaptive_grid_richardson_order_table=\"\";" \
  'field_spacing_classes.*at least three classes'
run_invalid zero_capacity "$mode2
adaptive_grid_max_cells=0;" \
  'max_cells/max_bytes.*explicit positive capacities'
run_invalid invalid_minimum_dimension "$mode2
adaptive_grid_min_nx=2;adaptive_grid_min_ny=2;" \
  'adaptive_grid_min_nx.*at least 4'
run_invalid impossible_guard "$mode2
adaptive_grid_extra_guard_cells=32;" \
  'extra_guard_cells.*no two-cell physical interior'
run_invalid mode3_missing_block "$mode3
adaptive_grid_blocking_factor=0;" \
  'blocking_factor.*required'
run_invalid mode3_invalid_levels "$mode3
adaptive_grid_max_levels=2;" \
  'max_levels.*must be 1'
run_invalid mode3_invalid_ratio "$mode3
adaptive_grid_refinement_ratio=3;" \
  'refinement_ratio.*must be 2'
run_invalid mode3_odd_transition "$mode3
adaptive_grid_transition_cells=3;" \
  'transition_cells.*positive and even'
run_invalid mode3_zero_transition "$mode3
adaptive_grid_transition_cells=0;" \
  'transition_cells.*positive'
run_invalid mode3_missing_residual "$mode3
adaptive_grid_fine_residual_tolerance=0;" \
  'fine_residual_tolerance.*required'
run_invalid terminal_patch_mode2 "$bspline2
adaptive_grid_terminal_patch_certification=1;" \
  'terminal_patch_certification.*Mode-3-only'
run_invalid terminal_patch_raster_source "$reference3
adaptive_grid_terminal_patch_certification=1;
adaptive_grid_field_operator=\"fd4_lagrange4_v1\";
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";" \
  'terminal_patch_certification.*requires fixed_width_bspline1_v1'
run_invalid terminal_patch_legacy_operator "$bspline3
adaptive_grid_terminal_patch_certification=1;
adaptive_grid_error_norm_policy=\"vector_rms_quantile_v1\";" \
  'terminal_patch_certification.*requires fd4_lagrange4_v1'
run_invalid diagnostics_disabled "$mode2
adaptive_grid_profile=\"off\";" \
  'adaptive_grid_profile.*summary or encounter'
run_invalid reference_missing_lumi_floor "$reference2
adaptive_grid_lumi_floor=0;" \
  'adaptive_grid_lumi_floor.*positive'
run_invalid bspline_missing_widths "$mode2
adaptive_grid_source_model=\"fixed_width_bspline1_v1\";
adaptive_grid_source_error_target=0.9;
adaptive_grid_solver_error_target=0.9;
adaptive_grid_lumi_error_target=0.9;
adaptive_grid_lumi_floor=1e-300;" \
  'adaptive_grid_source_width_x/y.*positive physical half-widths'
run_invalid bspline_zero_width "$bspline2
adaptive_grid_source_width_x=0;" \
  'adaptive_grid_source_width_x/y.*positive physical half-widths'
run_invalid bspline_negative_width "$bspline2
adaptive_grid_source_width_y=-4.6875;" \
  'adaptive_grid_source_width_x/y.*non-negative'
run_invalid bspline_negative_encounter_tail_fraction "$bspline2
adaptive_grid_encounter_tail_fraction=-0.01;" \
  'adaptive_grid_encounter_tail_fraction.*non-negative'
run_invalid bspline_unit_encounter_tail_fraction "$bspline2
adaptive_grid_encounter_tail_fraction=1.0;" \
  'adaptive_grid_encounter_tail_fraction.*(finite in|\(0,1\))'
run_invalid bspline_nonzero_reference_dimensions "$bspline2
adaptive_grid_source_reference_nx=64;" \
  'adaptive_grid_source_reference_nx/ny.*zero'
run_invalid reference_nonzero_source_width "$reference2
adaptive_grid_source_width_x=562.5;" \
  'adaptive_grid_source_width_x/y.*zero'
run_invalid canonical_nonzero_source_width "$mode2
adaptive_grid_source_width_y=4.6875;" \
  'adaptive_grid_source_reference_.*/source_width_.*/source/solver/lumi targets.*reserved'
run_runtime_invalid reference_signed_index_overflow "$reference2
adaptive_grid_source_reference_nx=65536;
adaptive_grid_source_reference_ny=65536;" \
  'signed-int linear index'
run_runtime_invalid reference_alignment_capacity "$reference2
adaptive_grid_size_classes_x=\"8,16,32\";
adaptive_grid_size_classes_y=\"8,16,32\";
adaptive_grid_max_nx=32;adaptive_grid_max_ny=32;
adaptive_grid_max_cells=85000;" \
  'class bank cannot fit adaptive_grid_max_cells'
run_runtime_invalid translated_anchor_unqualified "$mode2
adaptive_grid_anchor_x=1;" \
  'requires zero field anchors'
run_runtime_invalid mode23_schedule_v1_refused "$mode2
adaptive_grid_schedule_save=\"mode2.ag1\";" \
  'schedule persistence is disabled'
run_runtime_invalid mode23_backend_drift "$mode2
compute_backend=\"gpu\";allow_backend_fallback=1;" \
  'requires an explicitly requested CPU wavefront backend'

echo "mode-2/3 card contract validation PASS checks=$checks results=$result_root"
