#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/legacy_wavefront_scheduler/stateful_tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-wavefront-e2e.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

make_case()
{
  local name="$1" selector="$2" threads="${3:-0}"
  mkdir -p "$work/$name"
  if [[ "$selector" == wavefront_parallel ]]; then
    sed "s/collision_schedule=\"legacy\";/compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"pool_dual\";reduction_mode=\"strict\";collision_schedule=\"$selector\";/" \
      "$template" >"$work/$name/acc.dat"
  else
    sed "s/collision_schedule=\"legacy\";/collision_schedule=\"$selector\";/" \
      "$template" >"$work/$name/acc.dat"
  fi
}

run_case()
{
  local name="$1" selector="$2" threads="${3:-0}"
  make_case "$name" "$selector" "$threads"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  )
}

run_case legacy legacy
run_case strict wavefront_strict
run_case parallel2 wavefront_parallel 2
run_case parallel4 wavefront_parallel 4
run_case parallel4_repeat wavefront_parallel 4
make_case parallel4_sparse wavefront_parallel 4
sed -i 's/collision_schedule=/sparse_cells=1;collision_schedule=/' \
  "$work/parallel4_sparse/acc.dat"
(
  cd "$work/parallel4_sparse"
  "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
)
make_case parallel4_sparse_repeat wavefront_parallel 4
sed -i 's/collision_schedule=/sparse_cells=1;collision_schedule=/' \
  "$work/parallel4_sparse_repeat/acc.dat"
(
  cd "$work/parallel4_sparse_repeat"
  "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
)

normalize_result()
{
  grep -vE \
    '^collision_schedule_requested|^sparse_cells =|compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
    "$1"
}

# Requested/actual schedule provenance is the only intentional result-file
# difference.  The stateful beamstrahlung/photon/luminosity calculation, RNG
# state, native records, selected event heap, and explicit do_dump outputs must
# remain byte-for-byte identical.
for name in legacy strict parallel2 parallel4 parallel4_repeat \
            parallel4_sparse parallel4_sparse_repeat; do
  normalize_result "$work/$name/result.out" >"$work/$name/result.normalized"
done
for candidate in strict parallel2 parallel4 parallel4_repeat \
                 parallel4_sparse parallel4_sparse_repeat; do
  cmp "$work/legacy/result.normalized" "$work/$candidate/result.normalized"
done

mapfile -t legacy_outputs < <(
  cd "$work/legacy"
  find . -maxdepth 1 -type f \
    ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
    ! -name result.out ! -name result.normalized -printf '%f\n' | sort
)
for candidate in strict parallel2 parallel4 parallel4_repeat \
                 parallel4_sparse parallel4_sparse_repeat; do
  mapfile -t candidate_outputs < <(
    cd "$work/$candidate"
    find . -maxdepth 1 -type f \
      ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
      ! -name result.out ! -name result.normalized -printf '%f\n' | sort
  )
  if [[ "${legacy_outputs[*]}" != "${candidate_outputs[*]}" ]]; then
    echo "$candidate and legacy output file sets differ" >&2
    printf 'legacy: %s\n%s: %s\n' \
      "${legacy_outputs[*]}" "$candidate" "${candidate_outputs[*]}" >&2
    exit 1
  fi
  for output in "${legacy_outputs[@]}"; do
    cmp "$work/legacy/$output" "$work/$candidate/$output"
  done
done

find "$work/strict" -maxdepth 1 -type f \
  \( -name 'b1.*' -o -name 'b2.*' -o -name 'bp1.*' -o -name 'bp2.*' \) \
  -print -quit | grep -q .
grep -Eq \
  'FEEBL legacy wavefront summary:.*diagonals=15.*waves=15.*batches=15.*encounters=64.*parallel_physics_encounters=0.*ordered_commit=1' \
  "$work/strict/stderr.txt"
for candidate in parallel2 parallel4 parallel4_repeat \
                 parallel4_sparse parallel4_sparse_repeat; do
  grep -Eq \
    'FEEBL legacy wavefront summary:.*actual=wavefront_parallel_staged_fields.*diagonals=15.*waves=15.*batches=15.*encounters=64.*maximum_wave_width=8.*prefetched_field_encounters=64.*parallel_field_encounters=62.*full_parallel_make_step_encounters=0.*ordered_commit=1' \
    "$work/$candidate/stderr.txt"
done
grep -q 'field_workers=2' "$work/parallel2/stderr.txt"
grep -q 'field_workers=4' "$work/parallel4/stderr.txt"
grep -Eq \
  'field_dense_clear_visits=[1-9][0-9]*.*field_actual_clear_visits=[1-9][0-9]*.*field_dense_dot_visits=[1-9][0-9]*.*field_actual_dot_visits=[1-9][0-9]*' \
  "$work/parallel4_sparse/stderr.txt"
if grep -q 'legacy wavefront summary' "$work/legacy/stderr.txt"; then
  echo "legacy default emitted wavefront provenance" >&2
  exit 1
fi

make_variant_case()
{
  local name="$1" nt="$2" sparse="$3" selector="$4" threads="$5" spin="$6"
  local schedule="collision_schedule=\"$selector\";"
  if [[ "$selector" == wavefront_parallel ]]; then
    schedule="compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"pool_dual\";reduction_mode=\"strict\";collision_schedule=\"wavefront_parallel\";"
  fi
  mkdir -p "$work/$name"
  sed \
    -e "s/n_t=1;/n_t=$nt;/" \
    -e "s/collision_schedule=\"legacy\";/sparse_cells=$sparse;$schedule/" \
    -e "$([[ "$spin" == 1 ]] && printf '%s' 's/waist_y=0;/waist_y=0;polar_z.1=0.8;polar_z.2=-0.6;/')" \
    -e "$([[ "$spin" == 1 ]] && printf '%s' 's/ST_spin_flip=0;bmt_precession=0;/ST_spin_flip=1;bmt_precession=1;/')" \
    "$template" >"$work/$name/acc.dat"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
    normalize_result result.out >result.normalized
  )
}

compare_case_directories()
{
  local reference="$1" candidate="$2"
  cmp "$work/$reference/result.normalized" \
      "$work/$candidate/result.normalized"
  mapfile -t reference_files < <(
    cd "$work/$reference"
    find . -maxdepth 1 -type f \
      ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
      ! -name result.out ! -name result.normalized -printf '%f\n' | sort
  )
  mapfile -t candidate_files < <(
    cd "$work/$candidate"
    find . -maxdepth 1 -type f \
      ! -name acc.dat ! -name stdout.txt ! -name stderr.txt \
      ! -name result.out ! -name result.normalized -printf '%f\n' | sort
  )
  [[ "${reference_files[*]}" == "${candidate_files[*]}" ]]
  for produced in "${reference_files[@]}"; do
    cmp "$work/$reference/$produced" "$work/$candidate/$produced"
  done
}

# Multiple integration substeps exercise the full diagonal/substep DAG, not
# just the n_t=1 special case. Both dense and sparse field work must remain
# exact across worker counts and repeats.
for nt in 2 3; do
  for sparse in 0 1; do
    oracle="nt${nt}-s${sparse}-legacy"
    make_variant_case "$oracle" "$nt" "$sparse" legacy 0 0
    for candidate in p2 p4 p4-repeat; do
      threads=4
      [[ "$candidate" == p2 ]] && threads=2
      name="nt${nt}-s${sparse}-${candidate}"
      make_variant_case "$name" "$nt" "$sparse" \
        wavefront_parallel "$threads" 0
      compare_case_directories "$oracle" "$name"
      expected_waves=$((15 * nt))
      expected_encounters=$((64 * nt))
      expected_parallel=$((62 * nt))
      grep -Eq \
        "waves=${expected_waves}.*encounters=${expected_encounters}.*parallel_field_encounters=${expected_parallel}.*full_parallel_make_step_encounters=0" \
        "$work/$name/stderr.txt"
    done
  done
done

# BMT precession plus stochastic Sokolov-Ternov spin flips share the legacy
# RNG/result path. They remain in the ordered commit and are explicitly
# qualified here with nontrivial input polarization, n_t=2, and sparse fields.
make_variant_case spin-legacy 2 1 legacy 0 1
make_variant_case spin-p4 2 1 wavefront_parallel 4 1
make_variant_case spin-p4-repeat 2 1 wavefront_parallel 4 1
compare_case_directories spin-legacy spin-p4
compare_case_directories spin-legacy spin-p4-repeat
grep -q 'longitudinal bunch polarization' "$work/spin-p4/result.out"

make_case rejected_parallel wavefront_parallel 1
parallel_status=0
(
  cd "$work/rejected_parallel"
  "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
) || parallel_status=$?
if [[ $parallel_status -eq 0 ]]; then
  echo "wavefront_parallel ran without a concurrent persistent team" >&2
  exit 1
fi
grep -q 'requires compute_backend=cpu' \
  "$work/rejected_parallel/stderr.txt"
grep -qi 'no fallback was performed' "$work/rejected_parallel/stderr.txt"

reject_parallel_scope()
{
  local name="$1" edit="$2" expected="$3" status=0
  make_case "$name" wavefront_parallel 4
  sed -i "$edit" "$work/$name/acc.dat"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  ) || status=$?
  if [[ $status -eq 0 ]]; then
    echo "$name: unsupported parallel scope ran instead of failing" >&2
    exit 1
  fi
  grep -q "$expected" "$work/$name/stderr.txt"
  grep -qi 'no fallback was performed' "$work/$name/stderr.txt"
}

reject_parallel_scope rejected_symmetric \
  's/force_symmetric=0/force_symmetric=1/' 'force_symmetric=0'
reject_parallel_scope rejected_extra_grid \
  's/grids=1/grids=2/' 'grids=1'
reject_parallel_scope rejected_ngp \
  's/force_symmetric=0/force_symmetric=0;electron_distribution_rho=1/' \
  'electron_distribution_rho=2'
reject_parallel_scope rejected_cpu_mover \
  's/do_eloss=1/do_eloss=0/;s/collision_schedule=/cpu_primary_move=1;collision_schedule=/' \
  'cpu_primary_move=0'

make_case invalid invalid_selector
invalid_status=0
(
  cd "$work/invalid"
  "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
) || invalid_status=$?
if [[ $invalid_status -ne 2 ]]; then
  echo "invalid selector exited $invalid_status, expected 2" >&2
  exit 1
fi
grep -q 'collision_schedule must be legacy, wavefront_strict, or wavefront_parallel' \
  "$work/invalid/stderr.txt"

echo "PASS: strict and 2/4-worker staged fields are byte-exact for stateful FCC-Z-like physics"
echo "PASS: repeats, RNG/native/event/dump outputs match; 15 waves and 64 encounters reported"
echo "PASS: n_t=2/3 dense+sparse and BMT/ST spin/RNG variants are byte-exact"
echo "PASS: 62 encounters execute field work in concurrent waves; unsupported selectors fail closed"
