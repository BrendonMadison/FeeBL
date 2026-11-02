#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/gpu/mode3_reduced_fccz.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-mode3-e2e.XXXXXX")"
cleanup()
{
  case "$work" in
    /tmp/feebl-mode3-e2e.*) rm -rf -- "$work" ;;
    *) echo "refusing unsafe cleanup target: $work" >&2 ;;
  esac
}
trap cleanup EXIT

run_ok()
{
  local name=$1 block=$2
  mkdir -p "$work/$name"
  cp "$template" "$work/$name/acc.dat"
  (
    cd "$work/$name"
    "$binary" fccz_mode3 "$block" result.out >stdout.txt 2>stderr.txt
  )
  for artifact in beam1.dat beam2.dat rndm.save result.out; do
    [[ -s "$work/$name/$artifact" ]] || {
      echo "$name did not produce $artifact" >&2
      exit 1
    }
  done
}

run_ok serial serial_reduced
run_ok mode3 mode3_reduced
run_ok mode3_repeat mode3_reduced

# Same mesh, slice shape, and macroparticle count as the supplied FCC-Z request.
# This is a residency/schedule/normalization qualification; the small card
# above remains the serial numerical oracle.
mkdir -p "$work/mode3-production-shape"
sed \
  -e 's/n_x=32;n_y=32;n_z=4;n_t=2;n_m=4096;/n_x=128;n_y=256;n_z=160;n_t=1;n_m=25000;/' \
  -e 's/load_beam=0;store_beam=1;/load_beam=0;store_beam=0;/' \
  "$template" >"$work/mode3-production-shape/acc.dat"
(
  cd "$work/mode3-production-shape"
  "$binary" fccz_mode3 mode3_reduced result.out >stdout.txt 2>stderr.txt
)
grep -Eq \
  'FEEBL GPU mode3 production summary:.*diagonals=319 waves=319 encounters=25600 uploads=1 resident_batches=1 downloads=1 h2d_operations=8 d2h_operations=9 .*synchronizations=3 .*intermediate_particle_h2d_bytes=0 intermediate_particle_d2h_bytes=0 host_field_materializations=0' \
  "$work/mode3-production-shape/stderr.txt" || {
    echo "production-shape FCC-Z mode3 run did not prove full residency/schedule" >&2
    cat "$work/mode3-production-shape/stderr.txt" >&2
    exit 1
  }
# Pin the supplied FCC-Z card's authoritative legacy-float normalization.  The
# input `particles=24.3` is converted to physical particles with the historical
# factor of 1e10 before it is stored as a float; mode 3 must bind those live
# values rather than reconstructing ideal decimal inputs.
grep -Eq \
  'bunch_population1_live=243000000512 bunch_population2_live=243000000512 declared_n_m1=25000 declared_n_m2=25000 macro_charge1_live=9720000 macro_charge2_live=9720000 effective_population1=243000000000 effective_population2=243000000000 population_residual1=512 population_residual2=512' \
  "$work/mode3-production-shape/stderr.txt" || {
    echo "mode3 did not preserve the supplied FCC-Z live population normalization" >&2
    cat "$work/mode3-production-shape/stderr.txt" >&2
    exit 1
  }

for beam in beam1.dat beam2.dat; do
  python3 "$root/examples/gpu/compare_mode3_beams.py" \
    "$work/serial/$beam" "$work/mode3/$beam" --rtol 2e-5 --atol 2e-5
  cmp "$work/mode3/$beam" "$work/mode3_repeat/$beam"
done
cmp "$work/mode3/rndm.save" "$work/mode3_repeat/rndm.save"
# The legacy collide_ee aggregate path consumes rndm7 even with do_lumi=0;
# mode3 truthfully declares collision_rng=unused because that path is not
# ported.  Therefore serial-vs-mode3 final RNG equality is neither expected nor
# claimed.  Exact mode3 repeats above prove a deterministic checkpoint; the
# no-collision-RNG claim comes from the guarded non-stochastic implementation
# contract and its explicit production provenance, not from repeat equality.
python3 "$root/examples/gpu/compare_mode3_upsilon.py" \
  "$work/serial/stderr.txt" "$work/mode3/stderr.txt" \
  --rtol 2e-5 --atol 1e-12
python3 "$root/examples/gpu/compare_mode3_upsilon.py" \
  "$work/mode3/stderr.txt" "$work/mode3_repeat/stderr.txt" \
  --rtol 0 --atol 0
python3 "$root/examples/gpu/compare_mode3_lumi.py" \
  "$work/serial/result.out" "$work/mode3/result.out" --rtol 2e-5
python3 "$root/examples/gpu/compare_mode3_lumi.py" \
  "$work/mode3/result.out" "$work/mode3_repeat/result.out" --rtol 0

summary='FEEBL GPU mode3 production summary:'
grep -Fq "$summary" "$work/mode3/stderr.txt"
grep -Eq \
  'precision=mixed_qualified modern_reduction=fast_qualified initialization_rng=legacy_sequential_generator mode3_collision_rng=unused_no_stochastic_collision_process .*import_contracts_bound_before_cuda=1 paired_writeback_preflight_before_cuda=1 paired_writeback_committed=1 upsilon_committed_beam1=1 upsilon_committed_beam2=1 .*waves=14 encounters=32 uploads=1 resident_batches=1 downloads=1 h2d_operations=8 d2h_operations=9 h2d_bytes=[1-9][0-9]* d2h_bytes=[1-9][0-9]* setup_h2d_operations=1 setup_h2d_bytes=[1-9][0-9]* setup_synchronizations=1 synchronizations=3 .*intermediate_particle_h2d_bytes=0 intermediate_particle_d2h_bytes=0 host_field_materializations=0 .*fine_charge_density_luminosity_implemented=1 full_pair_event_luminosity_implemented=0 fallback_used=0' \
  "$work/mode3/stderr.txt" || {
    echo "mode3 summary does not prove the qualified one-upload/all-waves/one-download lifecycle" >&2
    cat "$work/mode3/stderr.txt" >&2
    exit 1
  }
grep -Eq \
  'physics_equivalence_scope = primary-x-y-vx-vy-energy-upsilon\+fine-density-lumi pair-aggregate-luminosity = unavailable' \
  "$work/mode3/result.out" || {
    echo "mode3 result did not mark unimplemented pair-aggregate luminosity" >&2
    exit 1
  }
grep -q 'mode3_result_scope = primary-x-y-vx-vy-energy-upsilon+fine-density-lumi' \
  "$work/mode3/result.out"
grep -q 'ordered_commit = 0' "$work/mode3/result.out"
if grep -q 'ordered legacy commit' "$work/mode3/stderr.txt"; then
  echo "mode3 emitted contradictory legacy ordered-commit provenance" >&2
  exit 1
fi
grep -q 'lumi_ee = unavailable (pair enumeration not implemented in gpu_mode=3)' \
  "$work/mode3/result.out"
if grep -Eq '^lumi_ee = 0([.]0*)? ' "$work/mode3/result.out"; then
  echo "mode3 emitted a misleading numeric pair-aggregate luminosity" >&2
  exit 1
fi
grep -Eq \
  'FEEBL: final .*engine requested=modern_gpu .*algorithm requested=exact .*precision requested=mixed_qualified .*modern_reduction requested=fast_qualified .*modern_gpu_streams requested=1 .*modern_gpu_graphs requested=off .*modern_trace requested=summary' \
  "$work/mode3/stderr.txt" || {
    echo "final mode3 selector provenance is incomplete" >&2
    cat "$work/mode3/stderr.txt" >&2
    exit 1
  }
if grep -Eq 'fallback="|fallback_used=1' "$work/mode3/stderr.txt"; then
  echo "mode3 used or recorded a fallback" >&2
  exit 1
fi

# The resident component has symmetry machinery, but the production v1 card
# boundary does not advertise it until card-level serial/GPU physics tests are
# available.  Both selectors must therefore fail before generation or CUDA.
for selector in force_symmetric charge_symmetric; do
  reject_dir="$work/reject-$selector"
  mkdir -p "$reject_dir"
  sed "s/${selector}=0;/${selector}=1;/" "$template" >"$reject_dir/acc.dat"
  status=0
  (
    cd "$reject_dir"
    "$binary" fccz_mode3 mode3_reduced result.out >stdout.txt 2>stderr.txt
  ) || status=$?
  if [[ $status -eq 0 ]] ||
     ! grep -q "gpu_mode=3 primary-resident production subset rejected:.*${selector}" \
       "$reject_dir/stderr.txt"; then
    echo "mode3 did not fail closed for unqualified $selector" >&2
    cat "$reject_dir/stderr.txt" >&2
    exit 1
  fi
  if grep -q "$summary" "$reject_dir/stderr.txt"; then
    echo "unqualified $selector reached resident execution" >&2
    exit 1
  fi
  for artifact in result.out beam1.dat beam2.dat lumi.ee.out lumi.ee.native; do
    if [[ -e "$reject_dir/$artifact" ]]; then
      echo "unqualified $selector created artifact $artifact" >&2
      exit 1
    fi
  done
done

# Wrong numerical qualification must fail before beam generation/CUDA.
mkdir -p "$work/reject-precision"
sed 's/precision="mixed_qualified"/precision="fp64"/' \
  "$template" >"$work/reject-precision/acc.dat"
status=0
(
  cd "$work/reject-precision"
  "$binary" fccz_mode3 mode3_reduced result.out >stdout.txt 2>stderr.txt
) || status=$?
if [[ $status -eq 0 ]] ||
   ! grep -q 'precision=mixed_qualified' \
     "$work/reject-precision/stderr.txt"; then
  echo "mode3 wrong-precision selector did not fail closed" >&2
  cat "$work/reject-precision/stderr.txt" >&2
  exit 1
fi

# The exact supplied FCC-Z physics card is deliberately unsupported until
# beamstrahlung/photon/full luminosity are ported.  The card-only gate must
# reject it before creating any physics/output artifact.
mkdir -p "$work/reject-original-fccz"
cp "$template" "$work/reject-original-fccz/acc.dat"
status=0
(
  cd "$work/reject-original-fccz"
  "$binary" fccz_mode3 mode3_original_fccz_rejected result.out \
    >stdout.txt 2>stderr.txt
) || status=$?
if [[ $status -eq 0 ]] ||
   ! grep -q 'gpu_mode=3 primary-resident production subset rejected' \
     "$work/reject-original-fccz/stderr.txt"; then
  echo "original FCC-Z radiation/luminosity card did not fail closed" >&2
  cat "$work/reject-original-fccz/stderr.txt" >&2
  exit 1
fi
if grep -q "$summary" "$work/reject-original-fccz/stderr.txt"; then
  echo "unsupported original FCC-Z card reached resident execution" >&2
  exit 1
fi
for artifact in result.out beam1.dat beam2.dat lumi.ee.out lumi.eg.out \
                lumi.ge.out lumi.gg.out lumi.ee.native; do
  if [[ -e "$work/reject-original-fccz/$artifact" ]]; then
    echo "unsupported original FCC-Z card created artifact $artifact" >&2
    exit 1
  fi
done

echo "PASS: production gpu_mode=3 ran the reduced FCC-Z-derived card"
echo "PASS: supplied 128x256x160 production shape ran all 25,600 encounters resident"
echo "PASS: serial primary state agrees within 2e-5 and repeats are exact"
echo "PASS: serial Upsilon moments agree within 2e-5; IDs/counts and repeats are exact"
echo "PASS: legacy-normalized fine charge-density luminosity agrees within 2e-5"
echo "PASS: mode3 RNG checkpoint repeats exactly; legacy pair-aggregate rndm7 is out of scope"
echo "PASS: unqualified force_symmetric and charge_symmetric cards failed before artifacts"
echo "PASS: exact supplied FCC-Z radiation/luminosity card failed before artifacts"
