#!/usr/bin/env bash
# Build, test, benchmark, and render the implemented FeeBL backend candidates.
# Every build and result lives below a new output directory; the source deck is
# never edited and an existing output directory is never overwritten.
set -euo pipefail

FEEBL_REPRO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FEEBL_REPRO_MODE=cpu
FEEBL_REPRO_OUTPUT=""
FEEBL_REPRO_QUICK=0
FEEBL_REPRO_RUN_BENCHMARKS=1

feebl_repro_source_git_root()
{
  local source_root git_root
  source_root=$(cd "$FEEBL_REPRO_ROOT" 2>/dev/null && pwd -P) || source_root=
  git_root=$(git -C "$FEEBL_REPRO_ROOT" rev-parse --show-toplevel 2>/dev/null) || git_root=
  if [ -n "$git_root" ]; then
    git_root=$(cd "$git_root" 2>/dev/null && pwd -P) || git_root=
  fi
  if [ -n "$source_root" ] && [ "$git_root" = "$source_root" ]; then
    printf '%s\n' "$git_root"
    return 0
  fi
  return 1
}

feebl_repro_source_revision()
{
  local git_root revision
  if git_root=$(feebl_repro_source_git_root); then
    git -C "$git_root" rev-parse --verify HEAD
  elif [ -r "$FEEBL_REPRO_ROOT/SOURCE_REVISION" ]; then
    revision=$(sed -n '1p' "$FEEBL_REPRO_ROOT/SOURCE_REVISION")
    if printf '%s\n' "$revision" | LC_ALL=C grep -Eq '^([0-9a-fA-F]{40}(-dirty)?|sha256:[0-9a-fA-F]{64})$'; then
      printf '%s\n' "$revision"
    else
      echo unknown
    fi
  else
    echo unknown
  fi
}

feebl_repro_usage()
{
  cat <<'USAGE'
usage: tools/reproduce_parallel_backends.sh [options]

  --mode cpu|gpu|all     backends to reproduce (default: cpu)
  --output DIR           new artifact directory (default: timestamped in /tmp)
  --quick                one small benchmark repetition; never a release result
  --no-benchmarks        build and run correctness tests only
  -h, --help             show this help

Useful environment controls:
  FEEBL_REPRO_BUILD_JOBS       parallel compiler jobs (default: nproc)
  FEEBL_REPRO_THREADS          CPU scaling list, e.g. 1,2,4,8
  FEEBL_REPRO_REPETITIONS      measured repetitions (default: 7)
  FEEBL_REPRO_WARMUPS          warm-up rounds (default: 1)
  FEEBL_REPRO_CASES            whitespace-separated quick-deck cases
  FEEBL_REPRO_AFFINITY         taskset list passed to the benchmark harness
  FEEBL_REPRO_MAX_LOAD1        optional idle-load gate
  FEEBL_REPRO_MAX_OTHER_FEEBL  pre-existing FeeBL limit (default: 0)
  FEEBL_CUDA_ARCHITECTURES     CMake CUDA architectures (default: native)
  FEEBL_CUDA_HOST_COMPILER     optional nvcc host compiler
  FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER  OFF by default
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode) FEEBL_REPRO_MODE=${2:-}; shift 2 ;;
    --output) FEEBL_REPRO_OUTPUT=${2:-}; shift 2 ;;
    --quick) FEEBL_REPRO_QUICK=1; shift ;;
    --no-benchmarks) FEEBL_REPRO_RUN_BENCHMARKS=0; shift ;;
    -h|--help) feebl_repro_usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; feebl_repro_usage >&2; exit 2 ;;
  esac
done

case "$FEEBL_REPRO_MODE" in
  cpu|gpu|all) ;;
  *) echo "--mode must be cpu, gpu, or all" >&2; exit 2 ;;
esac

if [ -z "$FEEBL_REPRO_OUTPUT" ]; then
  FEEBL_REPRO_OUTPUT="${TMPDIR:-/tmp}/feebl-reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$FEEBL_REPRO_OUTPUT" != /* ]]; then
  FEEBL_REPRO_OUTPUT="$PWD/$FEEBL_REPRO_OUTPUT"
fi
if [ -e "$FEEBL_REPRO_OUTPUT" ]; then
  echo "refusing to overwrite existing output: $FEEBL_REPRO_OUTPUT" >&2
  exit 2
fi
mkdir -p "$FEEBL_REPRO_OUTPUT"
FEEBL_REPRO_OUTPUT=$(cd "$FEEBL_REPRO_OUTPUT" && pwd -P)
FEEBL_REPRO_BUILD_ROOT="$FEEBL_REPRO_OUTPUT/build"
FEEBL_REPRO_RESULTS_ROOT="$FEEBL_REPRO_OUTPUT/results"

# Keep a complete human log and an expanded command trace. The log is written
# directly rather than through a background tee so EXIT-trap finalization can
# close and hash it without a pipe deadlock.
exec 7>&1 8>&2
exec > "$FEEBL_REPRO_OUTPUT/run.log" 2>&1
exec 9>"$FEEBL_REPRO_OUTPUT/commands.xtrace"
export BASH_XTRACEFD=9
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

feebl_repro_finalize()
{
  local campaign_rc=$1 final_rc=$1 ledger_rc=0 manifest_rc=0
  trap - EXIT
  set +e
  set +x
  echo "FeeBL reproduction campaign completion: exit_code=$campaign_rc"
  exec 9>&-
  exec 1>&7 2>&8
  {
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "campaign_exit_code=$campaign_rc"
    echo "run_log_closed_before_hash=1"
    echo "source_revision=$(feebl_repro_source_revision)"
    echo "mode=$FEEBL_REPRO_MODE"
  } > "$FEEBL_REPRO_OUTPUT/completion.txt"
  ledger_rc=$?
  if [ "$final_rc" -eq 0 ] && [ "$ledger_rc" -ne 0 ]; then
    final_rc=3
  fi
  (
    set -euo pipefail
    cd "$FEEBL_REPRO_OUTPUT" || exit 1
    if [ -d build ]; then
      find build -type f -path '*/src/feebl' -print0 \
        | sort -z | xargs -0 -r sha256sum > binary-checksums.sha256
    else
      : > binary-checksums.sha256
    fi
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  )
  manifest_rc=$?
  if [ "$final_rc" -eq 0 ] && [ "$manifest_rc" -ne 0 ]; then
    final_rc=3
  fi
  if [ "$final_rc" -eq 0 ]; then
    echo "FeeBL reproduction artifacts: $FEEBL_REPRO_OUTPUT"
  else
    echo "FeeBL reproduction failed (campaign=$campaign_rc, ledger=$ledger_rc, integrity=$manifest_rc); partial evidence: $FEEBL_REPRO_OUTPUT" >&2
  fi
  exit "$final_rc"
}
trap 'feebl_repro_finalize "$?"' EXIT

FEEBL_REPRO_JOBS=${FEEBL_REPRO_BUILD_JOBS:-$(nproc)}
FEEBL_REPRO_PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null \
  | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l)
if [ "$FEEBL_REPRO_PHYSICAL_CORES" -lt 1 ]; then
  FEEBL_REPRO_PHYSICAL_CORES=1
fi
FEEBL_REPRO_REPS=${FEEBL_REPRO_REPETITIONS:-7}
FEEBL_REPRO_WARMS=${FEEBL_REPRO_WARMUPS:-1}
FEEBL_REPRO_CASE_LIST=${FEEBL_REPRO_CASES:-"c_nt1 c_nt2 c_nt5"}
FEEBL_REPRO_THREAD_LIST=${FEEBL_REPRO_THREADS:-"1,2,4,$FEEBL_REPRO_PHYSICAL_CORES"}
if [ "$FEEBL_REPRO_QUICK" -eq 1 ]; then
  FEEBL_REPRO_REPS=1
  FEEBL_REPRO_WARMS=0
  FEEBL_REPRO_CASE_LIST=${FEEBL_REPRO_CASES:-"c_nt1"}
  FEEBL_REPRO_THREAD_LIST=${FEEBL_REPRO_THREADS:-"2"}
fi
case "$FEEBL_REPRO_REPS:$FEEBL_REPRO_WARMS" in
  *[!0-9:]*) echo "repetition and warm-up counts must be integers" >&2; exit 2 ;;
esac
if [ "$FEEBL_REPRO_REPS" -lt 1 ]; then
  echo "at least one measured repetition is required" >&2
  exit 2
fi

mkdir -p "$FEEBL_REPRO_BUILD_ROOT" "$FEEBL_REPRO_RESULTS_ROOT"

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_root=$FEEBL_REPRO_ROOT"
  echo "source_revision=$(feebl_repro_source_revision)"
  echo "source_status_begin"
  if FEEBL_REPRO_GIT_ROOT=$(feebl_repro_source_git_root); then
    git -C "$FEEBL_REPRO_GIT_ROOT" status --short 2>/dev/null || true
  else
    echo "not-a-source-repository"
  fi
  echo "source_status_end"
  echo "mode=$FEEBL_REPRO_MODE"
  echo "quick=$FEEBL_REPRO_QUICK"
  echo "benchmarks=$FEEBL_REPRO_RUN_BENCHMARKS"
  echo "repetitions=$FEEBL_REPRO_REPS"
  echo "warmups=$FEEBL_REPRO_WARMS"
  echo "cases=$FEEBL_REPRO_CASE_LIST"
  echo "threads=$FEEBL_REPRO_THREAD_LIST"
  echo "affinity=${FEEBL_REPRO_AFFINITY:-not-set}"
  echo "max_load1=${FEEBL_REPRO_MAX_LOAD1:-not-set}"
  echo "max_other_feebl=${FEEBL_REPRO_MAX_OTHER_FEEBL:-0}"
  uname -a
  lscpu
  grep -E 'MemTotal|MemAvailable' /proc/meminfo || true
  cmake --version
  "${CXX:-c++}" --version
  pkg-config --modversion fftw3 2>/dev/null || echo "fftw3-pkg-config=unavailable"
  python3 --version
  if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version \
      --format=csv,noheader
  fi
} > "$FEEBL_REPRO_OUTPUT/environment.txt"

FEEBL_REPRO_CPU_SUMMARY=""
FEEBL_REPRO_GPU_RAW=""

if [ "$FEEBL_REPRO_MODE" = cpu ] || [ "$FEEBL_REPRO_MODE" = all ]; then
  FEEBL_REPRO_SERIAL_BUILD="$FEEBL_REPRO_BUILD_ROOT/serial-release"
  FEEBL_REPRO_OPENMP_BUILD="$FEEBL_REPRO_BUILD_ROOT/openmp-release"
  cmake -S "$FEEBL_REPRO_ROOT" -B "$FEEBL_REPRO_SERIAL_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DFFTW3=ON \
    -DFEEBL_ENABLE_OPENMP=OFF -DFEEBL_ENABLE_CUDA=OFF
  cmake --build "$FEEBL_REPRO_SERIAL_BUILD" -j "$FEEBL_REPRO_JOBS"
  cmake -S "$FEEBL_REPRO_ROOT" -B "$FEEBL_REPRO_OPENMP_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DFFTW3=ON \
    -DFEEBL_ENABLE_OPENMP=ON -DFEEBL_ENABLE_CUDA=OFF
  cmake --build "$FEEBL_REPRO_OPENMP_BUILD" -j "$FEEBL_REPRO_JOBS"

  ctest --test-dir "$FEEBL_REPRO_SERIAL_BUILD" \
    -R '^feebl-(cpu-backend|backend-interface)' --output-on-failure
  ctest --test-dir "$FEEBL_REPRO_OPENMP_BUILD" \
    -R '^feebl-(cpu-backend|backend-interface)' --output-on-failure
  bash "$FEEBL_REPRO_ROOT/examples/cpu_backends.sh" \
    "$FEEBL_REPRO_SERIAL_BUILD/src/feebl"
  bash "$FEEBL_REPRO_ROOT/examples/cpu_backends.sh" \
    "$FEEBL_REPRO_OPENMP_BUILD/src/feebl"
  bash "$FEEBL_REPRO_ROOT/examples/run.sh" quick \
    -b "$FEEBL_REPRO_SERIAL_BUILD/src/feebl"

  if [ "$FEEBL_REPRO_RUN_BENCHMARKS" -eq 1 ]; then
    FEEBL_REPRO_CPU_RESULTS="$FEEBL_REPRO_RESULTS_ROOT/cpu"
    FEEBL_REPRO_BENCH_ARGS=(
      --binary "$FEEBL_REPRO_OPENMP_BUILD/src/feebl"
      --source-root "$FEEBL_REPRO_ROOT"
      --deck "$FEEBL_REPRO_ROOT/examples/decks/quick.dat"
      --variant serial
      --baseline serial
      --warmups "$FEEBL_REPRO_WARMS"
      --repetitions "$FEEBL_REPRO_REPS"
      --max-other-feebl "${FEEBL_REPRO_MAX_OTHER_FEEBL:-0}"
      --output "$FEEBL_REPRO_CPU_RESULTS"
    )
    for FEEBL_REPRO_CASE in $FEEBL_REPRO_CASE_LIST; do
      FEEBL_REPRO_BENCH_ARGS+=(--case "$FEEBL_REPRO_CASE")
    done
    if [ -n "${FEEBL_REPRO_AFFINITY:-}" ]; then
      FEEBL_REPRO_BENCH_ARGS+=(--affinity "$FEEBL_REPRO_AFFINITY")
    fi
    if [ -n "${FEEBL_REPRO_MAX_LOAD1:-}" ]; then
      FEEBL_REPRO_BENCH_ARGS+=(--max-load1 "$FEEBL_REPRO_MAX_LOAD1")
    fi

    declare -A FEEBL_REPRO_SEEN_THREADS=()
    IFS=',' read -r -a FEEBL_REPRO_THREAD_VALUES <<< "$FEEBL_REPRO_THREAD_LIST"
    for FEEBL_REPRO_THREAD_COUNT in "${FEEBL_REPRO_THREAD_VALUES[@]}"; do
      case "$FEEBL_REPRO_THREAD_COUNT" in
        ''|*[!0-9]*|0) echo "invalid CPU thread count: $FEEBL_REPRO_THREAD_COUNT" >&2; exit 2 ;;
      esac
      if [ -n "${FEEBL_REPRO_SEEN_THREADS[$FEEBL_REPRO_THREAD_COUNT]:-}" ]; then
        continue
      fi
      FEEBL_REPRO_SEEN_THREADS[$FEEBL_REPRO_THREAD_COUNT]=1
      for FEEBL_REPRO_CPU_VARIANT in pool_dual pool_private omp_atomic; do
        FEEBL_REPRO_VARIANT_NAME="${FEEBL_REPRO_CPU_VARIANT}_${FEEBL_REPRO_THREAD_COUNT}"
        case "$FEEBL_REPRO_CPU_VARIANT" in
          pool_dual) FEEBL_REPRO_REDUCTION=strict ;;
          pool_private) FEEBL_REPRO_REDUCTION=deterministic ;;
          omp_atomic) FEEBL_REPRO_REDUCTION=fast ;;
        esac
        FEEBL_REPRO_BENCH_ARGS+=(--variant "$FEEBL_REPRO_VARIANT_NAME")
        FEEBL_REPRO_BENCH_ARGS+=(--deck-fragment
          "$FEEBL_REPRO_VARIANT_NAME=compute_backend=\"cpu\";cpu_threads=$FEEBL_REPRO_THREAD_COUNT;cpu_variant=\"$FEEBL_REPRO_CPU_VARIANT\";reduction_mode=\"$FEEBL_REPRO_REDUCTION\";")
      done
    done
    python3 "$FEEBL_REPRO_ROOT/tools/benchmark_backends.py" \
      "${FEEBL_REPRO_BENCH_ARGS[@]}"
    FEEBL_REPRO_CPU_SUMMARY="$FEEBL_REPRO_CPU_RESULTS/summary.csv"
  fi
fi

if [ "$FEEBL_REPRO_MODE" = gpu ] || [ "$FEEBL_REPRO_MODE" = all ]; then
  command -v nvcc >/dev/null 2>&1 || {
    echo "GPU mode requested but nvcc is unavailable" >&2; exit 2;
  }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "GPU mode requested but nvidia-smi is unavailable" >&2; exit 2;
  }
  nvidia-smi -L
  FEEBL_REPRO_CUDA_BUILD="$FEEBL_REPRO_BUILD_ROOT/cuda-release"
  FEEBL_REPRO_CUDA_ARGS=(
    -DCMAKE_BUILD_TYPE=Release -DFFTW3=ON
    -DFEEBL_ENABLE_OPENMP=OFF -DFEEBL_ENABLE_CUDA=ON
    "-DFEEBL_CUDA_ARCHITECTURES=${FEEBL_CUDA_ARCHITECTURES:-native}"
    "-DFEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER=${FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER:-OFF}"
  )
  if [ -n "${FEEBL_CUDA_HOST_COMPILER:-}" ]; then
    FEEBL_REPRO_CUDA_ARGS+=("-DCMAKE_CUDA_HOST_COMPILER=$FEEBL_CUDA_HOST_COMPILER")
  fi
  cmake -S "$FEEBL_REPRO_ROOT" -B "$FEEBL_REPRO_CUDA_BUILD" \
    "${FEEBL_REPRO_CUDA_ARGS[@]}"
  cmake --build "$FEEBL_REPRO_CUDA_BUILD" -j "$FEEBL_REPRO_JOBS"
  ctest --test-dir "$FEEBL_REPRO_CUDA_BUILD" \
    -R '^feebl-(gpu-cic-correctness|backend-interface)$' --output-on-failure

  FEEBL_REPRO_GPU_RESULTS="$FEEBL_REPRO_RESULTS_ROOT/gpu-cic"
  bash "$FEEBL_REPRO_ROOT/examples/gpu/run_feebl_gpu_integration.sh" \
    "$FEEBL_REPRO_CUDA_BUILD/src/feebl" \
    "$FEEBL_REPRO_RESULTS_ROOT/gpu-integration"
  if [ "$FEEBL_REPRO_RUN_BENCHMARKS" -eq 1 ]; then
    FEEBL_GPU_BUILD_DIR="$FEEBL_REPRO_CUDA_BUILD" \
    FEEBL_GPU_RESULTS_DIR="$FEEBL_REPRO_GPU_RESULTS" \
    FEEBL_GPU_REPETITIONS="$FEEBL_REPRO_REPS" \
    FEEBL_CUDA_ARCHITECTURES="${FEEBL_CUDA_ARCHITECTURES:-native}" \
    FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER="${FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER:-OFF}" \
      bash "$FEEBL_REPRO_ROOT/examples/gpu/run_gpu_benchmarks.sh"
    FEEBL_REPRO_GPU_RAW="$FEEBL_REPRO_GPU_RESULTS/gpu_cic_benchmark.csv"
  fi
fi

FEEBL_REPRO_RENDER_ARGS=(--output "$FEEBL_REPRO_OUTPUT/rendered")
if [ -n "$FEEBL_REPRO_CPU_SUMMARY" ]; then
  FEEBL_REPRO_RENDER_ARGS+=(--cpu-summary "$FEEBL_REPRO_CPU_SUMMARY")
fi
if [ -n "$FEEBL_REPRO_GPU_RAW" ]; then
  FEEBL_REPRO_RENDER_ARGS+=(--gpu-raw "$FEEBL_REPRO_GPU_RAW")
fi
python3 "$FEEBL_REPRO_ROOT/tools/render_parallel_results.py" \
  "${FEEBL_REPRO_RENDER_ARGS[@]}"

set +x
exit 0
