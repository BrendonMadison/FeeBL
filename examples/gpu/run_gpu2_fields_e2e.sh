#!/bin/bash
# Focused whole-program gate for the deliberately narrow first GPU2 plan.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
SOURCE_DECK=$ROOT/examples/decks/quick.dat
[ -x "$BIN" ] || { echo "no executable FeeBL binary: $BIN"; exit 2; }

WORK=$(mktemp -d /tmp/feebl-gpu2-fields.XXXXXXXX) || exit 2
case "$WORK" in
  /tmp/feebl-gpu2-fields.*) ;;
  *) echo "refusing unsafe temporary directory: $WORK"; exit 2 ;;
esac
cleanup()
{
  if [ "${KEEP_WORK:-0}" != 1 ]; then
    case "$WORK" in
      /tmp/feebl-gpu2-fields.*) [ ! -d "$WORK" ] || rm -rf -- "$WORK" ;;
      *) echo "refusing unsafe cleanup target: $WORK" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

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
}

normalize()
{
  grep -vE 'compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' "$1"
}

SERIAL=$WORK/serial
make_deck '' "$SERIAL"
(cd "$SERIAL" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "serial oracle failed"
  cat "$SERIAL/stderr"
  exit 1
}

GPU2=$WORK/gpu2
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;reduction_mode="fast";' "$GPU2"
(cd "$GPU2" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "GPU2 fields execution failed"
  cat "$GPU2/stderr"
  exit 1
}

grep -Eq 'FEEBL GPU2 preflight: .*actual_scope=fields deposition=host_rho .*host_boundary=phi_materialization' "$GPU2/stderr" || {
  echo "missing honest GPU2 preflight provenance"
  cat "$GPU2/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 fields summary: .*calls=[1-9][0-9]* device_work_calls=[1-9][0-9]*' "$GPU2/stderr" || {
  echo "GPU2 solver did not report an observed field call"
  cat "$GPU2/stderr"
  exit 1
}
grep -Eq 'FEEBL: final .*gpu_mode requested=2 actual=2 .*gpu2_deposition requested=auto actual=host_rho .*gpu2_plan=committed gpu2_calls=[1-9][0-9]* .*gpu2_host_boundary=phi_materialization' "$GPU2/stderr" || {
  echo "final provenance does not prove actual GPU2 execution"
  cat "$GPU2/stderr"
  exit 1
}
if grep -Eq 'fallback=|backend_fallback' "$GPU2/stderr" "$GPU2/result.out"; then
  echo "unexpected fallback was recorded for the GPU2 execution"
  exit 1
fi

python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
  <(normalize "$SERIAL/result.out") <(normalize "$GPU2/result.out") \
  --rtol 5e-6 || {
    echo "GPU2 fields output exceeded the preregistered numerical bound"
    exit 1
  }

# The historical zero-charge shortcut performs no collision-loop CUDA work.
# A committed resident plan is still truthful, but observed-execution counters
# must remain zero rather than turning an API no-op into promotion evidence.
ZERO_SERIAL=$WORK/zero-charge-serial
make_deck '' "$ZERO_SERIAL"
sed -i 's/charge_sign=-1\.0;/charge_sign=0.0;/' "$ZERO_SERIAL/acc.dat"
(cd "$ZERO_SERIAL" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "zero-charge serial oracle failed"
  cat "$ZERO_SERIAL/stderr"
  exit 1
}
ZERO_GPU2=$WORK/zero-charge-gpu2
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;reduction_mode="deterministic";' "$ZERO_GPU2"
sed -i 's/charge_sign=-1\.0;/charge_sign=0.0;/' "$ZERO_GPU2/acc.dat"
(cd "$ZERO_GPU2" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "zero-charge GPU2 run failed"
  cat "$ZERO_GPU2/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 fields summary: .*calls=[1-9][0-9]* device_work_calls=0 ' "$ZERO_GPU2/stderr" || {
  echo "zero-charge GPU2 summary falsely records device work"
  cat "$ZERO_GPU2/stderr"
  exit 1
}
grep -Eq 'FEEBL: final .*gpu2_plan=committed gpu2_calls=0 ' "$ZERO_GPU2/stderr" || {
  echo "zero-charge final provenance falsely records GPU2 execution"
  cat "$ZERO_GPU2/stderr"
  exit 1
}
cmp -s <(normalize "$ZERO_SERIAL/result.out") \
       <(normalize "$ZERO_GPU2/result.out") || {
  echo "zero-charge GPU2 shortcut differs from the serial oracle"
  exit 1
}

# Explicit atomic is the restricted resident-deposition candidate.  Unlike
# auto/host_rho it transfers slice xy, returns only compact counts+dot before
# luminosity, and materializes phi only after update_slice_charge.
DIRECT=$WORK/direct-atomic
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu2_deposition="atomic";gpu_device=0;reduction_mode="fast";' "$DIRECT"
(cd "$DIRECT" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "GPU2 direct atomic fields execution failed"
  cat "$DIRECT/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 preflight: .*deposition=atomic .*host_boundary=compact_metrics_then_phi_materialization' "$DIRECT/stderr" || {
  echo "direct atomic plan lacks its exact host-boundary provenance"
  cat "$DIRECT/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 fields summary: .*deposition=atomic .*calls=[1-9][0-9]* device_work_calls=[1-9][0-9]* deposition_calls=[1-9][0-9]* field_solve_calls=[1-9][0-9]*' "$DIRECT/stderr" || {
  echo "direct atomic deposition/solve was not observed"
  cat "$DIRECT/stderr"
  exit 1
}
grep -Eq 'FEEBL: final .*gpu2_deposition requested=atomic actual=atomic .*gpu2_calls=[1-9][0-9]* .*gpu2_host_boundary=compact_metrics_then_phi_materialization' "$DIRECT/stderr" || {
  echo "direct atomic final provenance is incomplete"
  cat "$DIRECT/stderr"
  exit 1
}
python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
  <(normalize "$SERIAL/result.out") <(normalize "$DIRECT/result.out") \
  --rtol 5e-6 || {
    echo "direct atomic output exceeded the preregistered numerical bound"
    exit 1
  }

# Exercise the real main+extra-grid ownership path.  Each grid has a distinct
# Green spectrum/context; direct atomic must also deposit both grids while only
# the main grid contributes the compact fine-luminosity metric.
EXTRA_SERIAL=$WORK/extra-grid-serial
make_deck '' "$EXTRA_SERIAL"
sed -i 's/grids=1;/grids=2;/' "$EXTRA_SERIAL/acc.dat"
(cd "$EXTRA_SERIAL" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "extra-grid serial oracle failed"
  cat "$EXTRA_SERIAL/stderr"
  exit 1
}
EXTRA_HOST=$WORK/extra-grid-host-rho
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;reduction_mode="deterministic";' "$EXTRA_HOST"
sed -i 's/grids=1;/grids=2;/' "$EXTRA_HOST/acc.dat"
(cd "$EXTRA_HOST" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "extra-grid host-rho GPU2 run failed"
  cat "$EXTRA_HOST/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 preflight: .*solvers=2 ' "$EXTRA_HOST/stderr" &&
grep -Eq 'FEEBL GPU2 fields summary: .*solvers=2 .*device_work_calls=[1-9][0-9]* .*plan_creations=2 ' "$EXTRA_HOST/stderr" || {
  echo "extra-grid host-rho run did not retain two resident contexts"
  cat "$EXTRA_HOST/stderr"
  exit 1
}
python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
  <(normalize "$EXTRA_SERIAL/result.out") <(normalize "$EXTRA_HOST/result.out") \
  --rtol 5e-6 || {
    echo "extra-grid host-rho output exceeded the numerical bound"
    exit 1
  }

EXTRA_ATOMIC=$WORK/extra-grid-atomic
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu2_deposition="atomic";gpu_device=0;reduction_mode="fast";' "$EXTRA_ATOMIC"
sed -i 's/grids=1;/grids=2;/' "$EXTRA_ATOMIC/acc.dat"
(cd "$EXTRA_ATOMIC" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "extra-grid atomic GPU2 run failed"
  cat "$EXTRA_ATOMIC/stderr"
  exit 1
}
grep -Eq 'FEEBL GPU2 fields summary: .*deposition=atomic .*solvers=2 .*device_work_calls=[1-9][0-9]* .*plan_creations=2 ' "$EXTRA_ATOMIC/stderr" || {
  echo "extra-grid atomic run did not execute two resident contexts"
  cat "$EXTRA_ATOMIC/stderr"
  exit 1
}
python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
  <(normalize "$EXTRA_SERIAL/result.out") <(normalize "$EXTRA_ATOMIC/result.out") \
  --rtol 5e-6 || {
    echo "extra-grid atomic output exceeded the numerical bound"
    exit 1
  }

# Same-device deterministic reproducibility is a separate gate from the
# fast-mode differential bound above.  It also proves that preflight creates
# one persistent plan, rather than silently replanning inside the collision.
for repeat in 1 2 3; do
  deterministic=$WORK/deterministic-$repeat
  make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;reduction_mode="deterministic";' "$deterministic"
  (cd "$deterministic" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
    echo "deterministic GPU2 repeat $repeat failed"
    cat "$deterministic/stderr"
    exit 1
  }
  grep -Eq 'FEEBL GPU2 fields summary: .*calls=[1-9][0-9]* device_work_calls=[1-9][0-9]* .*plan_creations=1' "$deterministic/stderr" || {
    echo "deterministic repeat $repeat did not retain exactly one plan"
    cat "$deterministic/stderr"
    exit 1
  }
  grep -Eq 'FEEBL: final .*gpu2_plan=committed gpu2_calls=[1-9][0-9]*' "$deterministic/stderr" || {
    echo "deterministic repeat $repeat lacks observed-execution provenance"
    exit 1
  }
done
cmp -s <(normalize "$WORK/deterministic-1/result.out") \
       <(normalize "$WORK/deterministic-2/result.out") &&
cmp -s <(normalize "$WORK/deterministic-1/result.out") \
       <(normalize "$WORK/deterministic-3/result.out") || {
  echo "same-device deterministic GPU2 repeats are not byte-identical"
  exit 1
}

# In a combined CUDA+OpenMP build, compare the only authorized two-worker
# section against the deterministic one-worker run.  CUDA-only builds report
# the capability honestly and skip this candidate.
if grep -q 'openmp=enabled' "$WORK/deterministic-1/stderr"; then
  HYBRID=$WORK/hybrid-two-host-threads
  make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;gpu2_host_threads=2;reduction_mode="deterministic";' "$HYBRID"
  (cd "$HYBRID" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
    echo "two-thread GPU2 hybrid candidate failed"
    cat "$HYBRID/stderr"
    exit 1
  }
  grep -Eq 'cpu_variant requested=auto actual=gpu2_openmp_host .*gpu2_host_threads requested=2 actual=2' "$HYBRID/stderr" || {
    echo "hybrid run did not record the actual two-worker host plan"
    cat "$HYBRID/stderr"
    exit 1
  }
  cmp -s <(normalize "$WORK/deterministic-1/result.out") \
         <(normalize "$HYBRID/result.out") || {
    echo "two-thread primary CIC sections differ from one-thread physics"
    exit 1
  }

  LIMITED_TEAM=$WORK/authorized-limited-openmp-team
  make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;gpu2_host_threads=2;reduction_mode="deterministic";allow_backend_fallback=1;' "$LIMITED_TEAM"
  (cd "$LIMITED_TEAM" && OMP_THREAD_LIMIT=1 "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
    echo "authorized limited-team fallback did not complete serially"
    cat "$LIMITED_TEAM/stderr"
    exit 1
  }
  grep -Fq 'GPU2 preflight: gpu2_host_threads=2 requested, but OpenMP preflight formed a team of 1' "$LIMITED_TEAM/stderr" || {
    echo "OpenMP team-size preflight fallback lacks its exact reason"
    cat "$LIMITED_TEAM/stderr"
    exit 1
  }
  grep -Eq 'FEEBL: final compute_backend requested=gpu actual=serial .*gpu2_calls=0 ' "$LIMITED_TEAM/stderr" || {
    echo "limited OpenMP team falsely claims hybrid/GPU2 execution"
    cat "$LIMITED_TEAM/stderr"
    exit 1
  }
  cmp -s <(normalize "$SERIAL/result.out") \
         <(normalize "$LIMITED_TEAM/result.out") || {
    echo "limited-team preflight fallback differs from the serial oracle"
    exit 1
  }
else
  echo "SKIP: OpenMP-disabled CUDA build has no two-worker hybrid candidate"
fi

# Unsupported controls may fall back only when the card grants that authority,
# and the fallback must happen before collision so the result is one coherent
# serial trajectory.
AUTHORIZED=$WORK/authorized-preflight-fallback
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="full";gpu_device=0;reduction_mode="fast";allow_backend_fallback=1;' "$AUTHORIZED"
(cd "$AUTHORIZED" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "authorized GPU2 preflight fallback did not complete serially"
  cat "$AUTHORIZED/stderr"
  exit 1
}
grep -Fq 'FEEBL GPU2 preflight fallback: gpu2_scope=full is not implemented; this build provides fields and the strictly gated primary scope; actual_backend=serial before collision' "$AUTHORIZED/stderr" || {
  echo "authorized fallback lacks its exact pre-collision reason"
  cat "$AUTHORIZED/stderr"
  exit 1
}
grep -Eq 'FEEBL: final compute_backend requested=gpu actual=serial .*gpu_mode requested=2 actual=0 .*gpu2_calls=0 .*GPU2 preflight: gpu2_scope=full is not implemented' "$AUTHORIZED/stderr" || {
  echo "authorized fallback final provenance is not actual serial"
  cat "$AUTHORIZED/stderr"
  exit 1
}
cmp -s <(normalize "$SERIAL/result.out") \
       <(normalize "$AUTHORIZED/result.out") || {
  echo "authorized preflight fallback does not match the serial oracle"
  exit 1
}

# Resource and device initialization failures obey the same opt-in preflight
# rule.  This fraction is deliberately smaller than one 32x32 field context;
# every partially created CUDA resource must be released before serial starts.
MEMORY_FALLBACK=$WORK/authorized-memory-fallback
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";gpu_device=0;gpu2_memory_fraction=0.000001;reduction_mode="fast";allow_backend_fallback=1;' "$MEMORY_FALLBACK"
(cd "$MEMORY_FALLBACK" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr) || {
  echo "authorized GPU2 memory fallback did not complete serially"
  cat "$MEMORY_FALLBACK/stderr"
  exit 1
}
grep -Fq 'FEEBL GPU2 preflight fallback: grid 0 CUDA/cuFFT initialization failed: GPU2 field context requires' "$MEMORY_FALLBACK/stderr" || {
  echo "memory-limit fallback lacks its exact failure class"
  cat "$MEMORY_FALLBACK/stderr"
  exit 1
}
grep -Eq 'FEEBL: final compute_backend requested=gpu actual=serial .*gpu2_calls=0 .*fallback="GPU2 preflight: grid 0 CUDA/cuFFT initialization failed: GPU2 field context requires' "$MEMORY_FALLBACK/stderr" || {
  echo "memory-limit fallback final provenance is not actual serial"
  cat "$MEMORY_FALLBACK/stderr"
  exit 1
}
cmp -s <(normalize "$SERIAL/result.out") \
       <(normalize "$MEMORY_FALLBACK/result.out") || {
  echo "authorized memory fallback does not match the serial oracle"
  exit 1
}

UNSUPPORTED=$WORK/unsupported-full
make_deck 'compute_backend="gpu";gpu_mode=2;gpu2_scope="full";gpu_device=0;reduction_mode="fast";' "$UNSUPPORTED"
set +e
(cd "$UNSUPPORTED" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr)
rc=$?
set -e
if [ "$rc" -eq 0 ] ||
   ! grep -q 'gpu2_scope=full is not implemented' "$UNSUPPORTED/stderr"; then
  echo "unsupported full scope did not fail during preflight"
  cat "$UNSUPPORTED/stderr"
  exit 1
fi
if grep -q 'FEEBL GPU2 fields summary' "$UNSUPPORTED/stderr"; then
  echo "unsupported scope reached collision execution"
  exit 1
fi

echo "PASS: whole-program GPU2 fields execution, provenance, physics bound, and scope refusal"
