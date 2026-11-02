#!/usr/bin/env bash
# Tiny whole-program gate for the fail-closed wave-batched GPU field mode.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/gpu/mode4_fccz_exact.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode4-e2e.XXXXXX")"
cleanup()
{
  case "$work" in
    /tmp/feebl-mode4-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing unsafe cleanup target: $work" >&2 ;;
  esac
}
trap cleanup EXIT

[[ -x "$binary" ]] || { echo "FeeBL binary is not executable: $binary" >&2; exit 2; }
mkdir -p "$work/run"
sed \
  -e 's/gpu2_batch_slices=64/gpu2_batch_slices=4/' \
  -e 's/n_x=128;n_y=256;n_z=160/n_x=32;n_y=32;n_z=4/' \
  -e 's/n_t=1;n_m=25000/n_t=1;n_m=1000/' \
  -e 's/num_lumi=50000/num_lumi=2000/' \
  -e 's/spectrum_w_bins=4000/spectrum_w_bins=100/' \
  -e 's/hist_ee_bins=1010/hist_ee_bins=100/' \
  "$template" >"$work/run/acc.dat"

(
  cd "$work/run"
  "$binary" fccz p_fccz result.out >stdout.txt 2>stderr.txt
)

for artifact in result.out lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  [[ -s "$work/run/$artifact" ]] || {
    echo "gpu_mode=4 did not produce $artifact" >&2
    exit 1
  }
done

grep -Eq \
  'FEEBL legacy wavefront summary:.*waves=7 batches=7 encounters=16 .*maximum_wave_width=4 batch_width=4 .*field_prefetch_ms=[0-9.eE+-]+ parallel_physics_ms=0 ordered_commit_ms=[0-9.eE+-]+ .*per_encounter_synchronizations=0 ordered_commit=1' \
  "$work/run/stderr.txt" || {
    echo "gpu_mode=4 tiny run did not prove its wave/bucket schedule" >&2
    cat "$work/run/stderr.txt" >&2
    exit 1
  }
grep -Eq \
  'FEEBL GPU mode4 summary:.*batch_calls=7 encounters=16 padded_fft_slots=12 synchronizations=7 per_encounter_synchronizations=0 h2d_operations=21 d2h_operations=21 .*kernel_launches=56 cufft_executes=14 .*plan_creations=1 .*host_physics=legacy_complete_ordered rng=legacy_host_ordered' \
  "$work/run/stderr.txt" || {
    echo "gpu_mode=4 tiny run violated its structural telemetry oracle" >&2
    cat "$work/run/stderr.txt" >&2
    exit 1
  }
grep -Eq \
  'FEEBL: final compute_backend requested=gpu actual=gpu .*gpu_variant requested=auto actual=wave_batched_planmany_fields .*gpu_mode requested=4 actual=4 .*gpu2_plan=committed .*gpu4_calls=7 .*gpu2_host_boundary=one_completion_boundary_per_wave_bucket .*gpu2_rng=legacy_host_ordered' \
  "$work/run/stderr.txt" || {
    echo "gpu_mode=4 final provenance is incomplete" >&2
    cat "$work/run/stderr.txt" >&2
    exit 1
  }
grep -Eq \
  'collision_schedule_requested = wavefront_parallel collision_schedule_actual = gpu_mode4_wave_batched_fields_host_commit .*batches = 7 encounters = 16 .*ordered_commit = 1 .*physics_equivalence_scope = legacy-complete' \
  "$work/run/result.out" || {
    echo "gpu_mode=4 result provenance is incomplete" >&2
    exit 1
  }
if grep -Eq 'fallback_used=1|actual=serial_fallback|backend_fallback=' \
     "$work/run/stderr.txt" "$work/run/result.out"; then
  echo "gpu_mode=4 silently fell back" >&2
  exit 1
fi

# Independent whole-program oracle: derive a host-FFT card that changes only
# the backend contract, run the same seeded physics, and compare every numeric
# result field and every luminosity-table bin.  This guards against a common
# error that a batched CUDA implementation can share with the scalar CUDA
# reference used by the low-level unit test.
mkdir -p "$work/cpu-reference"
sed \
  -e 's/compute_backend="gpu";gpu_mode=4;gpu_device=0;/compute_backend="cpu";cpu_threads=2;cpu_variant="pool_dual";gpu_mode=1;/' \
  -e '/^gpu2_scope=/d' \
  -e '/^gpu2_streams=/d' \
  -e '/^gpu2_batch_slices=/d' \
  -e '/^gpu2_primary_mover=/d' \
  -e 's/^gpu2_profile=.*$/reduction_mode="strict";allow_backend_fallback=0;/' \
  "$work/run/acc.dat" >"$work/cpu-reference/acc.dat"
(
  cd "$work/cpu-reference"
  "$binary" fccz p_fccz result.out >stdout.txt 2>stderr.txt
)
normalise='^collision_schedule_requested|^sparse_cells =|^wavefront_field_solver =| cpu_primary_move = |compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance'
grep -vE "$normalise" "$work/cpu-reference/result.out" \
  >"$work/cpu-reference/result.normalized"
grep -vE "$normalise" "$work/run/result.out" >"$work/run/result.normalized"
python3 "$root/examples/gpu/compare_feebl_outputs.py" \
  "$work/cpu-reference/result.normalized" "$work/run/result.normalized" \
  --rtol 5e-5 --atol 1e-12
for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  python3 "$root/examples/gpu/compare_feebl_outputs.py" \
    "$work/cpu-reference/$artifact" "$work/run/$artifact" \
    --rtol 0.03 --atol 1e-12
done

# A second identical mode-4 execution must reproduce all physics artifacts
# byte for byte.  Timing and telemetry streams are intentionally excluded.
mkdir -p "$work/repeat"
cp "$work/run/acc.dat" "$work/repeat/acc.dat"
(
  cd "$work/repeat"
  "$binary" fccz p_fccz result.out >stdout.txt 2>stderr.txt
)
grep -vE "$normalise" "$work/repeat/result.out" \
  >"$work/repeat/result.normalized"
for artifact in result.normalized lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/run/$artifact" "$work/repeat/$artifact"
done

expect_rejected()
{
  local name="$1"
  local from="$2"
  local to="$3"
  local expected="$4"
  local second_from="${5:-}"
  local second_to="${6:-}"
  local directory="$work/rejected-$name"
  local status=0
  mkdir -p "$directory"
  if [[ -n "$second_from" ]]; then
    sed -e "s/$from/$to/" -e "s/$second_from/$second_to/" \
      "$work/run/acc.dat" >"$directory/acc.dat"
  else
    sed "s/$from/$to/" "$work/run/acc.dat" >"$directory/acc.dat"
  fi
  (
    cd "$directory"
    "$binary" fccz p_fccz result.out >stdout.txt 2>stderr.txt
  ) || status=$?
  if [[ $status -eq 0 ]] || ! grep -q "$expected" "$directory/stderr.txt"; then
    echo "gpu_mode=4 did not reject unsupported case: $name" >&2
    cat "$directory/stderr.txt" >&2
    exit 1
  fi
}

# Prove that mode 4 cannot silently ignore a CPU solver selector, fall back,
# or enter an unqualified legacy-physics domain.
expect_rejected direct \
  'wavefront_field_solver="fft";' 'wavefront_field_solver="direct";' \
  'gpu_mode=4 requires wavefront_field_solver=fft'
expect_rejected fallback \
  'allow_backend_fallback=0;' 'allow_backend_fallback=1;' \
  'gpu_mode=4 is abort-only; allow_backend_fallback must be 0'
expect_rejected symmetric \
  'force_symmetric=0;' 'force_symmetric=1;' \
  'gpu_mode=4 currently requires force_symmetric=0, grids=1'
expect_rejected extra-grid \
  'grids=1;' 'grids=2;' \
  'gpu_mode=4 currently requires force_symmetric=0, grids=1'
expect_rejected hadrons \
  'do_hadrons=0;' 'do_hadrons=1;' \
  'gpu_mode=4 supports the exact primary/photon/luminosity' \
  'photon_lumi_factorized=1;' 'photon_lumi_factorized=0;'

echo "PASS: tiny gpu_mode=4 ran 16 encounters in seven bounded wave batches"
echo "PASS: host photons/luminosity/native path completed with no fallback"
echo "PASS: mode4 agrees numerically with the independent host-FFT oracle"
echo "PASS: repeated mode4 physics artifacts are byte-identical"
echo "PASS: mode4 rejected fallback, direct fields, grids, symmetry, and hadrons"
