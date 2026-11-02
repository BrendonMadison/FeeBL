#!/bin/bash
# Whole-program qualification for the deliberately narrow gpu2_scope=primary
# host boundary.  This test requires a real CUDA build/device; CPU-only CI
# exercises the backend/provenance state machine and CUDA-free API stubs.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
SOURCE_DECK=$ROOT/examples/decks/quick.dat
[ -x "$BIN" ] || { echo "no executable FeeBL binary: $BIN"; exit 2; }

WORK=$(mktemp -d /tmp/feebl-gpu2-primary.XXXXXXXX) || exit 2
case "$WORK" in
  /tmp/feebl-gpu2-primary.*) ;;
  *) echo "refusing unsafe temporary directory: $WORK"; exit 2 ;;
esac
cleanup()
{
  if [ "${KEEP_WORK:-0}" != 1 ]; then
    case "$WORK" in
      /tmp/feebl-gpu2-primary.*) [ ! -d "$WORK" ] || rm -rf -- "$WORK" ;;
      *) echo "refusing unsafe cleanup target: $WORK" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

PRIMARY='compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";gpu_device=0;reduction_mode="fast";'
FIELDS='compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu2_deposition="atomic";gpu_device=0;reduction_mode="fast";'

make_deck()
{
  local config=$1 out=$2
  mkdir -p "$out"
  awk -v config="$config" '
    {
      sub(/^\{n_x=256;n_y=256;n_z=16;/,
          "{n_x=32;n_y=32;n_z=2;")
      sub(/^n_t=1;n_m=50000;/, "n_t=1;n_m=2000;")
      sub(/^hist_ee_bins=1010;/, config "hist_ee_bins=1010;")
      print
    }
  ' "$SOURCE_DECK" > "$out/acc.dat"
  # Checkpoint the final legacy RNG state.  A compact field provider is not
  # permitted to reorder the serial beam-1 then beam-2 stochastic physics.
  sed -i '0,/rndm_save=0/{s/rndm_save=0/rndm_save=1/}' "$out/acc.dat"
}

normalize()
{
  grep -vE 'compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' "$1"
}

run_ok()
{
  local dir=$1
  (cd "$dir" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
    echo "FeeBL run failed: $dir"
    cat "$dir/stderr"
    exit 1
  }
  [ -s "$dir/rndm.save" ] || {
    echo "missing RNG checkpoint: $dir/rndm.save"
    exit 1
  }
}

compare_pair()
{
  local fields=$1 primary=$2 label=$3
  python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
    <(normalize "$fields/result.out") <(normalize "$primary/result.out") \
    --rtol 5e-6 || {
      echo "$label output exceeded the preregistered numerical bound"
      exit 1
    }
  cmp -s "$fields/rndm.save" "$primary/rndm.save" || {
    echo "$label changed the final legacy RNG state"
    exit 1
  }
}

compare_mover_exact()
{
  local oracle=$1 candidate=$2 label=$3
  cmp -s <(normalize "$oracle/result.out") \
         <(normalize "$candidate/result.out") || {
    echo "$label changed normalized FeeBL output"
    diff -u <(normalize "$oracle/result.out") \
            <(normalize "$candidate/result.out") || true
    exit 1
  }
  for artifact in beam1.dat beam2.dat rndm.save; do
    [ -s "$oracle/$artifact" ] && [ -s "$candidate/$artifact" ] || {
      echo "$label did not retain ordered artifact $artifact"
      exit 1
    }
    cmp -s "$oracle/$artifact" "$candidate/$artifact" || {
      echo "$label changed ordered artifact $artifact"
      exit 1
    }
  done
}

# The reference and candidate use the same resident atomic deposition/cuFFT.
# Their only difference is full phi D2H + host CIC versus device CIC + compact
# Ex/Ey D2H.  Enable BMT and Sokolov-Ternov beamstrahlung so the shared mover,
# spin path, photon registration, and RNG order are all non-vacuous.
REF=$WORK/fields-spin
CAND=$WORK/primary-spin
make_deck "${FIELDS}bmt_precession=1;ST_spin_flip=1;" "$REF"
make_deck "${PRIMARY}bmt_precession=1;ST_spin_flip=1;" "$CAND"
for deck in "$REF/acc.dat" "$CAND/acc.dat"; do
  sed -i '0,/angle_x\.1=0\.00000000/{s/angle_x\.1=0\.00000000/polar_x.1=0;polar_y.1=0;polar_z.1=0.8;polar_x.2=0;polar_y.2=0;polar_z.2=-0.8;angle_x.1=0.00000000/}' "$deck"
done
run_ok "$REF"
run_ok "$CAND"

grep -Eq 'FEEBL GPU2 preflight: .*actual_scope=primary deposition=atomic .*host_boundary=compact_primary_fields primary_update=host_serial spin_transport=host_serial beamstrahlung=host_serial rng=legacy_host_ordered host_phi_materialized=0' "$CAND/stderr" || {
  echo "primary preflight provenance is incomplete"
  cat "$CAND/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 primary summary: scope=primary deposition=atomic fft=cufft host_boundary=compact_primary_fields primary_update=host_serial spin_transport=host_serial beamstrahlung=host_serial rng=legacy_host_ordered host_phi_materialized=0 .*device_work_calls=[1-9][0-9]* .*field_solve_calls=[1-9][0-9]* field_gather_calls=[1-9][0-9]*' "$CAND/stderr" || {
  echo "primary summary does not prove compact field execution"
  cat "$CAND/stderr"
  exit 1
}
grep -Eq 'FEEBL: final .*gpu2_scope requested=primary actual=primary .*gpu2_deposition requested=auto actual=atomic .*gpu2_plan=committed gpu2_calls=[1-9][0-9]* .*gpu2_host_boundary=compact_primary_fields gpu2_primary_update=host_serial gpu2_spin_transport=host_serial gpu2_beamstrahlung=host_serial gpu2_rng=legacy_host_ordered gpu2_host_phi_materialized=0' "$CAND/stderr" || {
  echo "final provenance does not prove the committed primary plan"
  cat "$CAND/stderr"
  exit 1
}
if grep -Eq 'fallback=|backend_fallback' "$CAND/stderr" "$CAND/result.out"; then
  echo "unexpected fallback was recorded for primary execution"
  exit 1
fi
compare_pair "$REF" "$CAND" "spin/ST primary"

# The parallel mover contract deliberately removes every primary-push RNG,
# spin, photon, trident and shared-result side effect.  Compare all host
# runtimes against the default serial compact-primary path exactly, including
# ordered beam records and the final legacy RNG checkpoint.  The worker ladder
# is powers of two up to the smaller of the machine and the bounded test cap.
SAFE_PRIMARY="${PRIMARY}do_eloss=0;ST_spin_flip=0;bmt_precession=0;do_coherent=0;do_trident=0;do_prod=0;track_pairs=0;track_muons=0;store_beam=1;"
DEFAULT_MOVER=$WORK/mover-default-serial
EXPLICIT_SERIAL=$WORK/mover-explicit-serial
make_deck "$SAFE_PRIMARY" "$DEFAULT_MOVER"
make_deck "${SAFE_PRIMARY}gpu2_primary_mover=\"serial\";gpu2_host_threads=1;" \
          "$EXPLICIT_SERIAL"
run_ok "$DEFAULT_MOVER"
run_ok "$EXPLICIT_SERIAL"
compare_mover_exact "$DEFAULT_MOVER" "$EXPLICIT_SERIAL" \
                    "omitted/default serial primary mover"

available_workers=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
case "$available_workers" in
  ''|*[!0-9]*) available_workers=1 ;;
esac
max_workers=${FEEBL_GPU2_PRIMARY_MAX_WORKERS:-8}
case "$max_workers" in
  ''|*[!0-9]*|0) echo "invalid FEEBL_GPU2_PRIMARY_MAX_WORKERS"; exit 2 ;;
esac
if [ "$max_workers" -gt "$available_workers" ]; then
  max_workers=$available_workers
fi

openmp_available=0
if grep -q 'openmp=enabled' "$DEFAULT_MOVER/stderr"; then
  openmp_available=1
fi
for mover in pool openmp; do
  if [ "$mover" = openmp ] && [ "$openmp_available" -ne 1 ]; then
    echo "SKIP: OpenMP primary mover matrix (build reports openmp=disabled)"
    continue
  fi
  workers=1
  while [ "$workers" -le "$max_workers" ]; do
    candidate=$WORK/mover-$mover-$workers
    make_deck \
      "${SAFE_PRIMARY}gpu2_primary_mover=\"$mover\";gpu2_host_threads=$workers;" \
      "$candidate"
    run_ok "$candidate"
    compare_mover_exact "$DEFAULT_MOVER" "$candidate" \
                        "$mover/$workers compact primary mover"
    grep -Eq "primary_mover_requested=$mover primary_mover_actual=$mover" \
      "$candidate/stderr" || {
      echo "$mover/$workers preflight did not report requested/actual mover"
      cat "$candidate/stderr"
      exit 1
    }
    grep -Eq "primary_mover=$mover primary_mover_threads=$workers primary_mover_calls=[1-9][0-9]* primary_mover_particles=[1-9][0-9]* primary_mover_ranges=[1-9][0-9]* primary_mover_observed_threads=$workers" \
      "$candidate/stderr" || {
      echo "$mover/$workers summary did not prove host execution"
      cat "$candidate/stderr"
      exit 1
    }
    grep -Eq "gpu2_primary_mover requested=$mover actual=$mover .*gpu2_primary_mover_calls=[1-9][0-9]* .*gpu2_primary_mover_particles=[1-9][0-9]* .*gpu2_primary_mover_ranges=[1-9][0-9]* .*gpu2_primary_mover_observed_threads=$workers" \
      "$candidate/stderr" || {
      echo "$mover/$workers final provenance counters are incomplete"
      cat "$candidate/stderr"
      exit 1
    }
    workers=$((workers * 2))
  done
done

# Auto is provenance-visible and deterministic: one worker resolves serial;
# a larger team resolves the persistent pool.  It is a policy convenience,
# not evidence that either candidate wins a benchmark.
for workers in 1 "$max_workers"; do
  if [ "$workers" -eq 1 ]; then expected=serial; else expected=pool; fi
  candidate=$WORK/mover-auto-$workers
  make_deck \
    "${SAFE_PRIMARY}gpu2_primary_mover=\"auto\";gpu2_host_threads=$workers;" \
    "$candidate"
  run_ok "$candidate"
  compare_mover_exact "$DEFAULT_MOVER" "$candidate" \
                      "auto/$workers compact primary mover"
  grep -Eq "primary_mover_requested=auto primary_mover_actual=$expected" \
    "$candidate/stderr" || {
    echo "auto/$workers resolution was not reported exactly"
    cat "$candidate/stderr"
    exit 1
  }
  [ "$max_workers" -ne 1 ] || break
done

# Registering the historically fixed rho switch must not alter the frozen
# default.  Compare an omitted key with an explicit value of 2, including the
# final legacy RNG checkpoint.
RHO_DEFAULT=$WORK/rho-default
RHO_EXPLICIT=$WORK/rho-explicit
make_deck "$FIELDS" "$RHO_DEFAULT"
make_deck "${FIELDS}electron_distribution_rho=2;" "$RHO_EXPLICIT"
run_ok "$RHO_DEFAULT"
run_ok "$RHO_EXPLICIT"
compare_pair "$RHO_DEFAULT" "$RHO_EXPLICIT" "rho parser default"

# Charge-sign transport is part of both the orbit and spin contract.  Exercise
# the supported e+e- sign and the historical |charge_sign|<1e-5 zero-field
# gate.  Same-sign (+1) beams are deliberately rejected by the global physics
# validator until their orbit and spin charge paths agree; that fail-closed
# production contract is asserted with the other rejection gates below.
for charge in physical zero; do
  F=$WORK/fields-$charge
  P=$WORK/primary-$charge
  make_deck "$FIELDS" "$F"
  make_deck "$PRIMARY" "$P"
  if [ "$charge" = physical ]; then value=-1.0; else value=0.0; fi
  sed -i "s/charge_sign=-1\.0;/charge_sign=$value;/" "$F/acc.dat" "$P/acc.dat"
  run_ok "$F"
  run_ok "$P"
  compare_pair "$F" "$P" "$charge-charge primary"
done

# Deliberately shrink the physical field domain so the production integration
# observes out-of-grid primaries.  Compact gather must return zeros for them,
# matching the no-extra-grid host mover exactly and without an OOB stencil.
OUT_F=$WORK/fields-out-of-grid
OUT_P=$WORK/primary-out-of-grid
make_deck "$FIELDS" "$OUT_F"
make_deck "$PRIMARY" "$OUT_P"
for deck in "$OUT_F/acc.dat" "$OUT_P/acc.dat"; do
  sed -i 's/cut_x=3092\.9;cut_y=459\.55;/cut_x=1.0;cut_y=1.0;/' "$deck"
done
run_ok "$OUT_F"
run_ok "$OUT_P"
grep -Eq 'FEEBL GPU2 primary summary: .*gather_beam1_out=[1-9][0-9]* .*gather_beam2_out=[1-9][0-9]*' "$OUT_P/stderr" || {
  echo "out-of-grid production case did not exercise compact zero fields"
  cat "$OUT_P/stderr"
  exit 1
}
compare_pair "$OUT_F" "$OUT_P" "out-of-grid primary"

# Every primary eligibility gate is fail-closed.  These cases must terminate
# during global parameter validation or GPU2 preflight, before any committed
# primary summary can be emitted.  The standalone resident unit separately
# covers zero-particle slices, null zero-count buffers, canaries, boundary
# nextafter values, and both charge signs at the API level.
expect_reject()
{
  local name=$1 config=$2 mutation=$3 expected=$4
  local dir=$WORK/reject-$name
  make_deck "$config" "$dir"
  if [ -n "$mutation" ]; then
    sed -i "$mutation" "$dir/acc.dat"
  fi
  set +e
  (cd "$dir" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr)
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] || ! grep -Fq "$expected" "$dir/stderr"; then
    echo "primary rejection gate failed: $name (rc=$rc, expected: $expected)"
    cat "$dir/stderr"
    exit 1
  fi
  if grep -q 'FEEBL GPU2 primary summary' "$dir/stderr"; then
    echo "rejected primary case reached collision execution: $name"
    exit 1
  fi
}

expect_reject integration "${PRIMARY}integration_method=1;" '' \
  'fields/primary scopes require integration_method=2'
expect_reject rho "${PRIMARY}electron_distribution_rho=1;" '' \
  'gpu2_deposition=atomic requires electron_distribution_rho=2'
expect_reject same-sign "$PRIMARY" 's/charge_sign=-1\.0;/charge_sign=1.0;/' \
  'charge_sign = +1 (e-e-) is not supported in this version'
expect_reject extra-grid "$PRIMARY" 's/grids=1;/grids=2;/' \
  'gpu2_scope=primary requires extra_grids=0 and exactly the main grid'
expect_reject deposition \
  'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";gpu2_deposition="shared_tile";gpu_device=0;reduction_mode="fast";' '' \
  'gpu2_deposition=shared_tile is not implemented'
expect_reject reduction \
  'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";gpu_device=0;reduction_mode="deterministic";' '' \
  'gpu2_scope=primary requires reduction_mode=fast'
expect_reject streams "${PRIMARY}gpu2_streams=2;" '' \
  'gpu2_streams>1 is not implemented'
expect_reject graphs "${PRIMARY}gpu2_graphs=\"on\";" '' \
  'gpu2_graphs=on is not implemented'
expect_reject fusion "${PRIMARY}gpu2_fusion=\"on\";" '' \
  'gpu2_fusion=on is not implemented'
expect_reject batch "${PRIMARY}gpu2_batch_slices=2;" '' \
  'gpu2_batch_slices>1 is unsafe'
expect_reject host-threads "${PRIMARY}gpu2_host_threads=2;" '' \
  'gpu2_primary_mover=serial requires gpu2_host_threads=0/1'
expect_reject mover-fields \
  "${FIELDS}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;" '' \
  'gpu2_primary_mover is valid only with gpu2_scope=primary'
expect_reject mover-eloss \
  "${PRIMARY}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;do_eloss=1;" '' \
  'parallel gpu2 primary movement requires interpolation=2, do_eloss=0'
expect_reject mover-st \
  "${PRIMARY}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;ST_spin_flip=1;bmt_precession=1;" \
  '0,/angle_x\.1=0\.00000000/{s/angle_x\.1=0\.00000000/polar_x.1=0;polar_y.1=0;polar_z.1=0;polar_x.2=0;polar_y.2=0;polar_z.2=0;angle_x.1=0.00000000/}' \
  'parallel gpu2 primary movement requires interpolation=2, do_eloss=0'
expect_reject mover-bmt \
  "${PRIMARY}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;bmt_precession=1;" \
  '0,/angle_x\.1=0\.00000000/{s/angle_x\.1=0\.00000000/polar_x.1=0;polar_y.1=0;polar_z.1=0;polar_x.2=0;polar_y.2=0;polar_z.2=0;angle_x.1=0.00000000/}' \
  'parallel gpu2 primary movement requires interpolation=2, do_eloss=0'
expect_reject mover-no-fallback \
  "${PRIMARY}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;do_eloss=1;allow_backend_fallback=1;" '' \
  'parallel gpu2 primary movement requires interpolation=2, do_eloss=0'

PARALLEL_PRIMARY="${PRIMARY}gpu2_primary_mover=\"pool\";gpu2_host_threads=2;do_eloss=0;ST_spin_flip=0;bmt_precession=0;do_coherent=0;do_trident=0;do_prod=0;track_pairs=0;track_muons=0;"
expect_reject mover-interpolation \
  "${PARALLEL_PRIMARY}integration_method=1;" '' \
  'fields/primary scopes require integration_method=2'
expect_reject mover-rho \
  "${PARALLEL_PRIMARY}electron_distribution_rho=1;" '' \
  'gpu2_deposition=atomic requires electron_distribution_rho=2'
expect_reject mover-extra-grid "$PARALLEL_PRIMARY" 's/grids=1;/grids=2;/' \
  'gpu2_scope=primary requires extra_grids=0 and exactly the main grid'
expect_reject mover-shared-tile \
  "${PARALLEL_PRIMARY}gpu2_deposition=\"shared_tile\";" '' \
  'gpu2_deposition=shared_tile is not implemented'
expect_reject mover-sort-reduce \
  "${PARALLEL_PRIMARY}gpu2_deposition=\"sort_reduce\";" '' \
  'gpu2_deposition=sort_reduce is not implemented'
expect_reject mover-coherent "$PARALLEL_PRIMARY" \
  's/do_coherent=0;/do_coherent=1;/' \
  'does not support coherent/trident field-charge sources'
expect_reject mover-trident "$PARALLEL_PRIMARY" \
  's/do_trident=0;/do_trident=1;/' \
  'does not support coherent/trident field-charge sources'
expect_reject mover-track-pairs \
  "${PARALLEL_PRIMARY}do_pairs=1;track_pairs=1;" '' \
  'requires track_pairs=0 and track_muons=0'
expect_reject mover-track-muons "${PARALLEL_PRIMARY}track_muons=1;" \
  's/do_muons=0;/do_muons=1;/' \
  'requires track_pairs=0 and track_muons=0'
expect_reject mover-prod "${PARALLEL_PRIMARY}do_prod=1;" '' \
  'key_word do_prod not completely implemented'
expect_reject profile "${PRIMARY}gpu2_profile=\"off\";" '' \
  'gpu2_profile=off is not implemented'
expect_reject coherent "$PRIMARY" 's/do_coherent=0;/do_coherent=1;/' \
  'does not support coherent/trident field-charge sources'
expect_reject trident "$PRIMARY" 's/do_trident=0;/do_trident=1;/' \
  'does not support coherent/trident field-charge sources'
expect_reject track-pairs "${PRIMARY}do_pairs=1;track_pairs=1;" '' \
  'requires track_pairs=0 and track_muons=0'
expect_reject track-muons "${PRIMARY}track_muons=1;" \
  's/do_muons=0;/do_muons=1;/' \
  'requires track_pairs=0 and track_muons=0'
expect_reject do-prod "${PRIMARY}do_prod=1;" '' \
  'key_word do_prod not completely implemented'

echo "PASS: GPU2 primary compact boundary, host physics/RNG order, fields regression, and all rejection gates"
