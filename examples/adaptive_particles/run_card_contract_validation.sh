#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FEEBL_EXECUTABLE SCRATCH_PARENT" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../.." && pwd -P)
executable=$(realpath -- "$1")
scratch_parent=$2

if [[ ! -x "$executable" ]]; then
  echo "FeeBL executable is missing or not executable: $executable" >&2
  exit 2
fi
mkdir -p -- "$scratch_parent"
scratch_parent=$(cd -- "$scratch_parent" && pwd -P)
result_root=$(mktemp -d "$scratch_parent/feebl-adaptive-particle-card.XXXXXXXX")

diagnostic_card="$source_root/examples/adaptive_particles/diagnostics_cpu_grid2.dat"
split_card="$source_root/examples/adaptive_particles/mode2_split_cpu_grid2.dat"
merge_card="$source_root/examples/adaptive_particles/mode3_split_merge_cpu_grid3.dat"
gpu6_negative_card="$source_root/examples/adaptive_particles/gpu6_unsupported_fail_closed.dat"
grid1_card="$source_root/examples/adaptive_grid/mode1_cpu_exact_window.dat"
for required in "$diagnostic_card" "$split_card" "$merge_card" \
                "$gpu6_negative_card" "$grid1_card"; do
  if [[ ! -f "$required" ]]; then
    echo "required in-tree card is missing: $required" >&2
    exit 2
  fi
done

checks=0
printf 'schema=FEEBL_ADAPTIVE_MACROPARTICLE_CARD_VALIDATION_V1\n' \
  > "$result_root/manifest.txt"
printf 'utc_started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$result_root/manifest.txt"
printf 'source_root=%s\nexecutable=%s\n' "$source_root" "$executable" \
  >> "$result_root/manifest.txt"
if command -v sha256sum >/dev/null 2>&1; then
  printf 'executable_sha256=%s\n' \
    "$(sha256sum -- "$executable" | awk '{print $1}')" \
    >> "$result_root/manifest.txt"
fi

run_success() {
  local name=$1 card=$2 directory="$result_root/$1"
  mkdir -- "$directory"
  cp -- "$card" "$directory/acc.dat"
  echo "RUN accepted $name"
  (cd -- "$directory" &&
    "$executable" fccz p_fccz out.dat >log.txt 2>&1)
  if [[ ! -s "$directory/out.dat" ]]; then
    echo "$name did not create a nonempty out.dat" >&2
    exit 1
  fi
  printf '%s\taccepted\n' "$name" >> "$result_root/cases.tsv"
  ((checks += 1))
}

run_rejected() {
  local name=$1 card=$2 expected=$3 directory="$result_root/$1"
  mkdir -- "$directory"
  cp -- "$card" "$directory/acc.dat"
  echo "RUN rejected $name"
  if (cd -- "$directory" &&
      "$executable" fccz p_fccz out.dat >log.txt 2>&1); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  if ! grep -Eiq -- "$expected" "$directory/log.txt"; then
    echo "$name failed for the wrong reason; expected /$expected/" >&2
    tail -80 "$directory/log.txt" >&2
    exit 1
  fi
  printf '%s\trejected\t%s\n' "$name" "$expected" \
    >> "$result_root/cases.tsv"
  ((checks += 1))
}

run_replay_required() {
  local name=$1 card=$2 directory="$result_root/$1" status=0
  mkdir -- "$directory"
  cp -- "$card" "$directory/acc.dat"
  echo "RUN replay-required $name"
  (cd -- "$directory" &&
    "$executable" fccz p_fccz out.dat >log.txt 2>&1) || status=$?
  if [[ $status -ne 75 ]]; then
    echo "$name returned $status instead of replay-required status 75" >&2
    tail -80 "$directory/log.txt" >&2
    exit 1
  fi
  if [[ -e "$directory/out.dat" ]]; then
    echo "$name published out.dat across the unmet-error gate" >&2
    exit 1
  fi
  if [[ $(grep -c '^FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2 ' \
                   "$directory/log.txt") -ne 1 ]]; then
    echo "$name did not emit exactly one replay-required record" >&2
    exit 1
  fi
  if ! grep -Eq \
      'FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2.*exit_code=75.*publication_blocked=1.*exploratory_outputs_authorized=0.*final_physics_outputs_published=0.*publication_scope=final_summary_outputs_only_v1.*pre_gate_diagnostic_artifacts_may_exist=1.*pre_gate_artifacts_physics_qualification_claim=0.*fresh_run_directory_required=1.*final_polarization_sample_complete=1.*bpm_finalization_complete=1.*global_luminosity_publication_complete=1.*roots_added_online=0.*fresh_complete_process_required=1' \
      "$directory/log.txt"; then
    echo "$name did not emit the canonical replay-required record" >&2
    tail -80 "$directory/log.txt" >&2
    exit 1
  fi
  printf '%s\treplay-required\texit=75\n' "$name" \
    >> "$result_root/cases.tsv"
  ((checks += 1))
}

assert_terminal_exploratory_signal() {
  local name=$1 directory="$result_root/$1" signal_line final_line
  if [[ $(grep -c '^FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2 ' \
                   "$directory/log.txt") -ne 1 ]]; then
    echo "$name did not emit exactly one exploratory replay record" >&2
    exit 1
  fi
  signal_line=$(grep -n '^FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2 ' \
                       "$directory/log.txt" | cut -d: -f1)
  final_line=$(grep -n '^ FEEBL: final ' "$directory/log.txt" | tail -1 | \
                     cut -d: -f1)
  if [[ -z "$final_line" || $signal_line -le $final_line ]]; then
    echo "$name emitted exploratory success before final output completion" >&2
    exit 1
  fi
}

materialize() {
  local name=$1 source=$2
  shift 2
  local directory="$result_root/materialized"
  mkdir -p -- "$directory"
  sed "$@" "$source" > "$directory/$name.dat"
  printf '%s\n' "$directory/$name.dat"
}

inject_particle_block() {
  local name=$1 destination="$result_root/materialized/$1.dat"
  # Read the explicit v1 particle block from the diagnostic card and insert
  # it immediately before the first ordinary grid dimension in a separately
  # valid adaptive-grid Mode-1 card. This lets the process reach the intended
  # cross-capability preflight instead of failing an unrelated Mode-2 key.
  awk '
    FNR == NR {
      if ($0 ~ /^adaptive_macroparticle_/) block = block $0 ORS
      next
    }
    !inserted && $0 ~ /^n_x=/ {
      printf "%s", block
      inserted = 1
    }
    { print }
  ' "$diagnostic_card" "$grid1_card" > "$destination"
  printf '%s\n' "$destination"
}

# Backward-compatibility cases. The omitted case removes every new key. The
# explicit Mode-0 case leaves the complete canonical block in place. Neither
# may emit an active-source preflight or run summary.
omitted=$(materialize omitted "$diagnostic_card" \
  -e '/^adaptive_macroparticle_/d')
explicit_mode0=$(materialize explicit_mode0 "$diagnostic_card" \
  -e 's/^adaptive_macroparticle_mode=1;/adaptive_macroparticle_mode=0;/')

run_success omitted "$omitted"
if grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_' "$result_root/omitted/log.txt"; then
  echo "omitted defaults unexpectedly activated adaptive particles" >&2
  exit 1
fi
run_success explicit_mode0 "$explicit_mode0"
if grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_' \
     "$result_root/explicit_mode0/log.txt"; then
  echo "explicit Mode 0 unexpectedly activated adaptive particles" >&2
  exit 1
fi
((checks += 2))

# The three currently admitted CPU routes.
run_success diagnostics_cpu_grid2 "$diagnostic_card"
grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1.*requested_mode=1.*actual_mode=1' \
  "$result_root/diagnostics_cpu_grid2/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1.*requested_mode=1.*actual_mode=1.*encounter_source_builds=[1-9][0-9]*.*commits=0.*representative_visits1=[1-9][0-9]*.*representative_visits2=[1-9][0-9]*' \
  "$result_root/diagnostics_cpu_grid2/log.txt"

run_success split_cpu_grid2 "$split_card"
grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1.*requested_mode=2.*actual_mode=2' \
  "$result_root/split_cpu_grid2/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1.*requested_mode=2.*actual_mode=2.*encounter_source_builds=[1-9][0-9]*.*commits=[1-9][0-9]*.*representative_visits1=[1-9][0-9]*.*representative_visits2=[1-9][0-9]*' \
  "$result_root/split_cpu_grid2/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2.*exit_code=0.*publication_blocked=0.*exploratory_outputs_authorized=1.*final_physics_outputs_published=1.*publication_scope=final_summary_outputs_only_v1.*pre_gate_diagnostic_artifacts_may_exist=1.*pre_gate_artifacts_physics_qualification_claim=0.*fresh_run_directory_required=1.*final_polarization_sample_complete=1.*bpm_finalization_complete=1.*global_luminosity_publication_complete=1.*roots_added_online=0.*fresh_complete_process_required=1' \
  "$result_root/split_cpu_grid2/log.txt"
assert_terminal_exploratory_signal split_cpu_grid2

run_success split_merge_cpu_grid3 "$merge_card"
grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1.*requested_mode=3.*actual_mode=3' \
  "$result_root/split_merge_cpu_grid3/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1.*requested_mode=3.*actual_mode=3.*encounter_source_builds=[1-9][0-9]*.*commits=[1-9][0-9]*.*representative_visits1=[1-9][0-9]*.*representative_visits2=[1-9][0-9]*' \
  "$result_root/split_merge_cpu_grid3/log.txt"
grep -Eq 'FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2.*exit_code=0.*publication_blocked=0.*exploratory_outputs_authorized=1.*final_physics_outputs_published=1.*publication_scope=final_summary_outputs_only_v1.*pre_gate_diagnostic_artifacts_may_exist=1.*pre_gate_artifacts_physics_qualification_claim=0.*fresh_run_directory_required=1.*final_polarization_sample_complete=1.*bpm_finalization_complete=1.*global_luminosity_publication_complete=1.*roots_added_online=0.*fresh_complete_process_required=1' \
  "$result_root/split_merge_cpu_grid3/log.txt"
assert_terminal_exploratory_signal split_merge_cpu_grid3
((checks += 8))

# Parser and semantic validation failures.
invalid_mode=$(materialize invalid_mode "$diagnostic_card" \
  -e 's/^adaptive_macroparticle_mode=1;/adaptive_macroparticle_mode=4;/')
run_rejected invalid_mode "$invalid_mode" \
  'adaptive_macroparticle_mode.*0, 1, 2, or 3'

full_physics=$(materialize full_physics "$diagnostic_card" \
  -e 's/adaptive_macroparticle_domain="field_source"/adaptive_macroparticle_domain="full_physics"/')
run_rejected full_physics "$full_physics" \
  'full_physics.*not physics-qualified|adaptive full_physics domain'

persistent=$(materialize persistent "$diagnostic_card" \
  -e 's/adaptive_macroparticle_source_evolution="projected_current_state"/adaptive_macroparticle_source_evolution="persistent_replayed_roots"/')
run_rejected persistent_replayed_roots "$persistent" \
  'persistent_replayed_roots.*external replay generator'

invalid_scope=$(materialize invalid_scope "$diagnostic_card" \
  -e 's/adaptive_macroparticle_decision_scope="encounter"/adaptive_macroparticle_decision_scope="slice"/')
run_rejected invalid_decision_scope "$invalid_scope" \
  'decisions must be encounter proposals with wave commits'

invalid_hysteresis=$(materialize invalid_hysteresis "$diagnostic_card" \
  -e 's/adaptive_macroparticle_coarsen_threshold=0.35/adaptive_macroparticle_coarsen_threshold=1.01/')
run_rejected invalid_hysteresis "$invalid_hysteresis" \
  'invalid adaptive macroparticle error/controller limits'

# Milestone capability failures that must occur before adaptive source work.
run_rejected gpu6_unimplemented "$gpu6_negative_card" \
  'gpu_mode must be.*adaptive-grid mode 5'
if [[ -e "$result_root/gpu6_unimplemented/out.dat" ]]; then
  echo "unsupported GPU Mode 6 published out.dat" >&2
  exit 1
fi
((checks += 1))

wrong_grid=$(inject_particle_block wrong_grid)
run_rejected grid_mode1_unimplemented "$wrong_grid" \
  'require adaptive_grid_mode=2 or 3|fixed-grid and Mode-1'

wrong_source=$(materialize wrong_source "$diagnostic_card" \
  -e 's/adaptive_grid_source_model="reference_cic_v1"/adaptive_grid_source_model="canonical_cell_average_v1"/' \
  -e 's/adaptive_grid_source_reference_nx=128/adaptive_grid_source_reference_nx=0/' \
  -e 's/adaptive_grid_source_reference_ny=128/adaptive_grid_source_reference_ny=0/' \
  -e 's/adaptive_grid_source_error_target=0.9/adaptive_grid_source_error_target=0/' \
  -e 's/adaptive_grid_solver_error_target=0.9/adaptive_grid_solver_error_target=0/' \
  -e 's/adaptive_grid_lumi_error_target=0.9/adaptive_grid_lumi_error_target=0/' \
  -e 's/adaptive_grid_lumi_floor=1e-300/adaptive_grid_lumi_floor=0/')
run_rejected non_reference_source "$wrong_source" \
  'require adaptive_grid_source_model=reference_cic_v1'

fixed_width_source=$(materialize fixed_width_source "$diagnostic_card" \
  -e 's/adaptive_grid_source_model="reference_cic_v1"/adaptive_grid_source_model="fixed_width_bspline1_v1"/' \
  -e 's/adaptive_grid_source_reference_nx=128/adaptive_grid_source_reference_nx=0/' \
  -e 's/adaptive_grid_source_reference_ny=128;/adaptive_grid_source_reference_ny=0;adaptive_grid_source_width_x=562.5;adaptive_grid_source_width_y=4.6875;/')
run_rejected fixed_width_source_with_adaptive_macroparticles \
  "$fixed_width_source" \
  'fixed_width_bspline1_v1.*adaptive_macroparticle_mode=0'

two_cohorts=$(materialize two_cohorts "$diagnostic_card" \
  -e 's/adaptive_macroparticle_statistical_cohorts=1/adaptive_macroparticle_statistical_cohorts=2/')
run_rejected manufactured_cohorts "$two_cohorts" \
  'one complete-run cohort|manufacture.*support'

policy_path=$(materialize policy_path "$diagnostic_card" \
  -e 's/adaptive_macroparticle_policy_save=""/adaptive_macroparticle_policy_save="policy.json"/')
run_rejected policy_persistence_unimplemented "$policy_path" \
  'policy/trace persistence.*fail-closed|replay schema'

wrong_root_source=$(materialize wrong_root_source "$diagnostic_card" \
  -e 's/adaptive_macroparticle_root_source="physical_primary_roots_v1"/adaptive_macroparticle_root_source="pilot_replay_rqmc_v1"/')
run_rejected rqmc_root_source_unimplemented "$wrong_root_source" \
  'adaptive_macroparticle_root_source|root[-_ ]source|pilot_replay_rqmc_v1|physical_primary_roots_v1'

wrong_root_seed=$(materialize wrong_root_seed "$diagnostic_card" \
  -e 's/adaptive_macroparticle_root_seed=0/adaptive_macroparticle_root_seed=1/')
run_rejected adaptive_root_rng_unimplemented "$wrong_root_seed" \
  'root_seed|root seed|root RNG'

wrong_scramble=$(materialize wrong_scramble "$diagnostic_card" \
  -e 's/adaptive_macroparticle_scramble_id=0/adaptive_macroparticle_scramble_id=1/')
run_rejected adaptive_scramble_unimplemented "$wrong_scramble" \
  'scramble_id|scramble ID|RQMC'

wrong_interval=$(materialize wrong_interval "$diagnostic_card" \
  -e 's/adaptive_macroparticle_update_interval=1/adaptive_macroparticle_update_interval=2/')
run_rejected controller_interval_unimplemented "$wrong_interval" \
  'update_interval|controller schedule'

wrong_cooldown=$(materialize wrong_cooldown "$diagnostic_card" \
  -e 's/adaptive_macroparticle_cooldown=0/adaptive_macroparticle_cooldown=1/')
run_rejected controller_cooldown_unimplemented "$wrong_cooldown" \
  'cooldown|controller schedule'

wrong_dwell=$(materialize wrong_dwell "$diagnostic_card" \
  -e 's/adaptive_macroparticle_dwell=2/adaptive_macroparticle_dwell=3/')
run_rejected controller_dwell_unimplemented "$wrong_dwell" \
  'dwell|controller schedule'

wrong_merge=$(materialize wrong_merge "$diagnostic_card" \
  -e 's/adaptive_macroparticle_merge="exact_cell_projection_v1"/adaptive_macroparticle_merge="mpcr_v1"/')
run_rejected general_mpcr_unimplemented "$wrong_merge" \
  'unqualified adaptive macroparticle merge algorithm|merge (algorithm|identifier)'

# Process-level transaction limit. This card passes parsing/preflight, then
# rejects the first nontrivial projected source rather than depositing a
# truncated representation.
count_limited=$(materialize count_limited "$split_card" \
  -e 's/adaptive_macroparticle_max_count=0/adaptive_macroparticle_max_count=1/')
run_rejected source_count_limit "$count_limited" \
  'exceeds adaptive_macroparticle_max_count'
grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1' \
  "$result_root/source_count_limit/log.txt"
((checks += 1))

# A single complete realization cannot estimate eta_N.  The production
# default must therefore finish the collision, emit the process-level fresh-
# root replay handoff, and exit 75 before final physics publication.  The
# supplied active cards opt into exploratory output with zero and emit the
# same signal with an exit-code/publication disposition of zero.
unavailable_total_bound=$(materialize unavailable_total_bound "$split_card" \
  -e 's/adaptive_macroparticle_fail_on_unmet_error=0/adaptive_macroparticle_fail_on_unmet_error=1/')
run_replay_required unavailable_total_bound "$unavailable_total_bound"
grep -q 'FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1.*controller_total_bound=unavailable_single_run' \
  "$result_root/unavailable_total_bound/log.txt"
grep -q 'FEEBL: ADAPTIVE ROOT REPLAY REQUIRED:.*fresh complete-run roots are required' \
  "$result_root/unavailable_total_bound/log.txt"
((checks += 2))

printf 'utc_finished=%s\nchecks=%s\nstatus=PASS\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checks" \
  >> "$result_root/manifest.txt"
echo "adaptive macroparticle card/process validation PASS checks=$checks results=$result_root"
