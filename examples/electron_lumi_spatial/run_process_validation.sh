#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: $0 FEEBL_EXECUTABLE SCRATCH_PARENT" >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fixture="$script_dir/tiny_spatial_runtime.dat"
executable=$(realpath -- "$1")
scratch_parent=$2
[[ -x "$executable" && -f "$fixture" ]] || exit 2
mkdir -p -- "$scratch_parent"
scratch_parent=$(cd -- "$scratch_parent" && pwd -P)
result_root=$(mktemp -d "$scratch_parent/feebl-spatial-lumi.XXXXXXXX")

checks=0
materialize() {
  local name=$1
  shift
  mkdir -p -- "$result_root/cards"
  if [[ $# -eq 0 ]]; then
    cp -- "$fixture" "$result_root/cards/$name.dat"
  else
    sed "$@" "$fixture" > "$result_root/cards/$name.dat"
  fi
  printf '%s\n' "$result_root/cards/$name.dat"
}

run_success() {
  local name=$1 card=$2 directory="$result_root/$1"
  mkdir -- "$directory"
  cp -- "$card" "$directory/acc.dat"
  (cd -- "$directory" &&
    "$executable" tiny spatial out.dat >log.txt 2>&1)
  [[ -s "$directory/out.dat" ]] || {
    echo "$name did not publish out.dat" >&2
    exit 1
  }
  ((checks += 1))
}

run_rejected() {
  local name=$1 card=$2 expected=$3 directory="$result_root/$1"
  mkdir -- "$directory"
  cp -- "$card" "$directory/acc.dat"
  if (cd -- "$directory" &&
      "$executable" tiny spatial out.dat >log.txt 2>&1); then
    echo "$name unexpectedly succeeded" >&2
    exit 1
  fi
  grep -Eiq -- "$expected" "$directory/log.txt" || {
    echo "$name failed for the wrong reason; expected /$expected/" >&2
    tail -80 "$directory/log.txt" >&2
    exit 1
  }
  [[ ! -s "$directory/out.dat" ]] || {
    echo "$name published out.dat across a fail-closed gate" >&2
    exit 1
  }
  ((checks += 1))
}

# Mode 0 remains observationally inert whether the new block is omitted or
# explicitly dormant.  Full out.dat identity is stronger than comparing only
# the luminosity lines.
omitted=$(materialize omitted -e '/^electron_lumi_spatial_/d')
explicit_zero=$(materialize explicit_zero \
  -e 's/electron_lumi_spatial_kernel=1/electron_lumi_spatial_kernel=0/')
run_success omitted "$omitted"
run_success explicit_zero "$explicit_zero"
cmp -- "$result_root/omitted/out.dat" "$result_root/explicit_zero/out.dat"
cmp -- "$result_root/omitted/rndm.save" \
       "$result_root/explicit_zero/rndm.save"
if grep -q 'FEEBL_ELECTRON_LUMI_SPATIAL' \
     "$result_root/omitted/log.txt" "$result_root/explicit_zero/log.txt"; then
  echo "Mode 0 emitted an active spatial-luminosity receipt" >&2
  exit 1
fi
((checks += 2))

# Parser and capability matrix.  Every unsupported collision side effect is
# refused before simulation, never silently routed through the legacy pair
# kernel.
invalid_mode=$(materialize invalid_mode \
  -e 's/electron_lumi_spatial_kernel=1/electron_lumi_spatial_kernel=2/')
run_rejected invalid_mode "$invalid_mode" \
  'electron_lumi_spatial_kernel must be 0 or 1'

bad_geometry=$(materialize bad_geometry \
  -e 's/electron_lumi_spatial_nx=129/electron_lumi_spatial_nx=1/')
run_rejected bad_geometry "$bad_geometry" \
  'requires explicit nx,ny>=2'

bad_cap=$(materialize bad_cap \
  -e 's/electron_lumi_spatial_max_pairs=1000000/electron_lumi_spatial_max_pairs=0/')
run_rejected bad_cap "$bad_cap" 'positive representable max_pairs'

bad_ratio=$(materialize bad_ratio \
  -e 's/electron_ratio=1.0/electron_ratio=0.5/')
run_rejected bad_ratio "$bad_ratio" 'restricted aggregate-only collision path'

bad_isr=$(materialize bad_isr \
  -e 's/do_eloss=0;/do_eloss=0;do_isr=1;/')
run_rejected bad_isr "$bad_isr" 'restricted aggregate-only collision path'

bad_heap=$(materialize bad_heap -e 's/do_lumi=0/do_lumi=1/')
run_rejected bad_heap "$bad_heap" 'restricted aggregate-only collision path'

bad_photons=$(materialize bad_photons \
  -e 's/do_photons=0/do_photons=1/')
run_rejected bad_photons "$bad_photons" \
  'restricted aggregate-only collision path'

bad_photon_lumi=$(materialize bad_photon_lumi \
  -e 's/do_photon_lumi=0/do_photon_lumi=1/')
run_rejected bad_photon_lumi "$bad_photon_lumi" \
  'restricted aggregate-only collision path'

bad_precompute=$(materialize bad_precompute \
  -e 's/electron_lumi_precompute=0/electron_lumi_precompute=1/')
run_rejected bad_precompute "$bad_precompute" \
  'restricted aggregate-only collision path'

bad_extra_grid=$(materialize bad_extra_grid -e 's/grids=1/grids=2/')
run_rejected bad_extra_grid "$bad_extra_grid" \
  'restricted aggregate-only collision path'

bad_loaded_event=$(materialize bad_loaded_event \
  -e 's/load_events=0/load_events=1/')
run_rejected bad_loaded_event "$bad_loaded_event" \
  'restricted aggregate-only collision path'

# Runtime failures are also fail closed.  The narrow window must report whole
# rejected-root accounting; the tiny pair cap must stop construction before
# any aggregate is published.
outside=$(materialize outside \
  -e 's/electron_lumi_spatial_x_min=-10000.0/electron_lumi_spatial_x_min=-1.0/' \
  -e 's/electron_lumi_spatial_x_max=10000.0/electron_lumi_spatial_x_max=1.0/' \
  -e 's/electron_lumi_spatial_y_min=-2000.0/electron_lumi_spatial_y_min=-1.0/' \
  -e 's/electron_lumi_spatial_y_max=2000.0/electron_lumi_spatial_y_max=1.0/')
run_rejected outside "$outside" \
  'FEEBL_ELECTRON_LUMI_SPATIAL_FAILURE_V1.*out-of-domain-root'
grep -Eq 'beam1_rejected_roots=[1-9][0-9]*|beam2_rejected_roots=[1-9][0-9]*' \
  "$result_root/outside/log.txt"
((checks += 1))

pair_cap=$(materialize pair_cap \
  -e 's/electron_lumi_spatial_max_pairs=1000000/electron_lumi_spatial_max_pairs=1/')
run_rejected pair_cap "$pair_cap" \
  'max_pairs|pair resource limit|resource limit|pair count exceeds configured maximum'

# Frozen physical roots: changing only the field/force mesh must leave the
# total, high-energy, E_cm/spin aggregates, plan sequence, and dL/dW byte
# identical.  Runs are deliberately sequential.
grid64=$(materialize grid64)
grid128=$(materialize grid128 -e 's/n_x=64;n_y=64/n_x=128;n_y=128/')
grid256=$(materialize grid256 -e 's/n_x=64;n_y=64/n_x=256;n_y=256/')
run_success grid64 "$grid64"
run_success grid128 "$grid128"
run_success grid256 "$grid256"

for name in grid64 grid128 grid256; do
  directory="$result_root/$name"
  [[ -s "$directory/lumi.ee.out.dLdW.fne" ]] || {
    echo "$name did not publish aggregate dL/dW" >&2
    exit 1
  }
  [[ ! -e "$directory/lumi.ee.out" ]] || {
    echo "$name manufactured a forbidden e-e event heap file" >&2
    exit 1
  }
  grep -E '^(lumi_fine|lumi_ee|lumi_ee_high|lumi_pp|E_cm =|lumi\[)' \
    "$directory/out.dat" > "$directory/physics.txt"
  grep 'FEEBL_ELECTRON_LUMI_SPATIAL_RECEIPT_V1' \
    "$directory/log.txt" > "$directory/receipt.txt"
  grep -Eq 'beam1_rejected_roots=0.*beam2_rejected_roots=0.*physical_pairs=[1-9][0-9]*.*maximum_closure_relative_error=0([[:space:]]|$)' \
    "$directory/receipt.txt"
done
cmp -- "$result_root/grid64/physics.txt" "$result_root/grid128/physics.txt"
cmp -- "$result_root/grid64/physics.txt" "$result_root/grid256/physics.txt"
cmp -- "$result_root/grid64/receipt.txt" "$result_root/grid128/receipt.txt"
cmp -- "$result_root/grid64/receipt.txt" "$result_root/grid256/receipt.txt"
cmp -- "$result_root/grid64/lumi.ee.out.dLdW.fne" \
       "$result_root/grid128/lumi.ee.out.dLdW.fne"
cmp -- "$result_root/grid64/lumi.ee.out.dLdW.fne" \
       "$result_root/grid256/lumi.ee.out.dLdW.fne"
cmp -- "$result_root/omitted/rndm.save" "$result_root/grid64/rndm.save"
cmp -- "$result_root/grid64/rndm.save" "$result_root/grid128/rndm.save"
cmp -- "$result_root/grid64/rndm.save" "$result_root/grid256/rndm.save"
((checks += 11))

# Exercise the worker-private GRID route as well as the direct main-grid
# route.  Receipt and result publication happen at the canonical batch
# barrier; the forbidden e-e heap has no crossing-level publication.
worker=$(materialize worker \
  -e 's/{n_x=64;/{compute_backend="cpu";cpu_threads=2;cpu_variant="pool_dual";reduction_mode="strict";allow_backend_fallback=0;sparse_cells=1;collision_schedule="wavefront_parallel";wavefront_field_solver="fft";wavefront_parallel_physics=1;n_x=64;/')
run_success worker "$worker"
grep -E '^(lumi_fine|lumi_ee|lumi_ee_high|lumi_pp|E_cm =|lumi\[)' \
  "$result_root/worker/out.dat" > "$result_root/worker/physics.txt"
grep 'FEEBL_ELECTRON_LUMI_SPATIAL_RECEIPT_V1' \
  "$result_root/worker/log.txt" > "$result_root/worker/receipt.txt"
cmp -- "$result_root/grid64/physics.txt" "$result_root/worker/physics.txt"
cmp -- "$result_root/grid64/receipt.txt" "$result_root/worker/receipt.txt"
cmp -- "$result_root/grid64/lumi.ee.out.dLdW.fne" \
       "$result_root/worker/lumi.ee.out.dLdW.fne"
[[ ! -e "$result_root/worker/lumi.ee.out" ]]
((checks += 5))

# The restricted kernel still supports deterministic spin aggregates.  Beam 2
# travels toward -z, so polar_z=-0.3 corresponds to local helicity +0.3.
spin=$(materialize spin \
  -e 's/do_eloss=0;/do_eloss=0;bmt_precession=1;/' \
  -e 's/waist_y=0;/waist_y=0;polar_x.1=0;polar_y.1=0;polar_z.1=0.8;polar_x.2=0;polar_y.2=0;polar_z.2=-0.3;/')
run_success spin "$spin"
grep -Fq '(P.1 , P.2) luminosity-weighted = (+0.8000 , +0.3000)' \
  "$result_root/spin/out.dat"
grep -Eq '^lumi_pp = [1-9][0-9.eE+-]* ' "$result_root/spin/out.dat"
[[ -s "$result_root/spin/lumi.ee.out.dLdW.fne" ]]
((checks += 3))

printf 'FEEBL_ELECTRON_LUMI_SPATIAL_PROCESS_VALIDATION_V1 checks=%d status=PASS result_root=%s\n' \
  "$checks" "$result_root"
