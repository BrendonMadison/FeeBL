#!/usr/bin/env bash
# Reproduce FeeBL serial, OpenMP, CUDA mode-1, and CUDA mode-2 evidence.
# Outputs are append-free: a campaign refuses an existing destination.
set -uo pipefail

FEEBL_GPU2_REPRO_ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
FEEBL_GPU2_REPRO_OUTPUT=""
FEEBL_GPU2_REPRO_QUICK=0
FEEBL_GPU2_REPRO_BENCHMARKS=0
FEEBL_GPU2_REPRO_DEVICE=${FEEBL_GPU2_DEVICE:-0}
FEEBL_GPU2_REPRO_ARCH=${FEEBL_CUDA_ARCHITECTURES:-native}

usage()
{
  cat <<'USAGE'
usage: tools/reproduce_gpu2.sh [options]

  --output DIR             new artifact directory (default: timestamped /tmp)
  --quick                  reduced smoke campaign; explicitly non-decision
  --benchmarks             opt in to timing (disabled by default)
  --device ID              CUDA device (default: FEEBL_GPU2_DEVICE or 0)
  --cuda-architectures A   CMake CUDA architecture list (default: native)
  -h, --help               show this help

Environment controls:
  FEEBL_GPU2_BUILD_JOBS          compiler jobs (default: nproc)
  FEEBL_CUDA_HOST_COMPILER      optional nvcc host compiler
  FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER  OFF by default
  FEEBL_GPU2_BENCH_REPETITIONS  measured timing rounds (default: 15)
  FEEBL_GPU2_BENCH_WARMUPS      untimed rounds (default: 2)
  FEEBL_GPU2_HOST_THREADS       hybrid host workers (current qualified value: 2)
  FEEBL_GPU2_MAX_PARTICLES      mode-1 microbenchmark ceiling (default: 4194304)

Exit status: 0 means every requested gate passed, 1 means at least one gate
failed, 3 means no gate failed but required evidence was blocked/unavailable.
The steps.tsv ledger is authoritative.  This focused campaign is never a full
promotion decision: it does not execute the powered stochastic, complete
physics-domain, sanitizer, and native-Linux qualification matrix.  A --quick
run is additionally ineligible as timing evidence.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) FEEBL_GPU2_REPRO_OUTPUT=${2:-}; shift 2 ;;
    --quick) FEEBL_GPU2_REPRO_QUICK=1; shift ;;
    --benchmarks) FEEBL_GPU2_REPRO_BENCHMARKS=1; shift ;;
    --device) FEEBL_GPU2_REPRO_DEVICE=${2:-}; shift 2 ;;
    --cuda-architectures) FEEBL_GPU2_REPRO_ARCH=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$FEEBL_GPU2_REPRO_DEVICE" in
  ''|*[!0-9]*) echo "--device must be a non-negative integer" >&2; exit 2 ;;
esac
if [ -z "$FEEBL_GPU2_REPRO_ARCH" ]; then
  echo "--cuda-architectures may not be empty" >&2
  exit 2
fi

if [ -z "$FEEBL_GPU2_REPRO_OUTPUT" ]; then
  FEEBL_GPU2_REPRO_OUTPUT="${TMPDIR:-/tmp}/feebl-gpu2-reproduction-$(date -u +%Y%m%dT%H%M%SZ)-$$"
elif [[ "$FEEBL_GPU2_REPRO_OUTPUT" != /* ]]; then
  FEEBL_GPU2_REPRO_OUTPUT="$PWD/$FEEBL_GPU2_REPRO_OUTPUT"
fi
if [ -e "$FEEBL_GPU2_REPRO_OUTPUT" ]; then
  echo "refusing to overwrite existing output: $FEEBL_GPU2_REPRO_OUTPUT" >&2
  exit 2
fi
mkdir -p "$FEEBL_GPU2_REPRO_OUTPUT"
FEEBL_GPU2_REPRO_OUTPUT=$(cd "$FEEBL_GPU2_REPRO_OUTPUT" && pwd -P)

FEEBL_GPU2_REPRO_BUILD_ROOT="$FEEBL_GPU2_REPRO_OUTPUT/build"
FEEBL_GPU2_REPRO_RESULTS_ROOT="$FEEBL_GPU2_REPRO_OUTPUT/results"
FEEBL_GPU2_REPRO_LOG_ROOT="$FEEBL_GPU2_REPRO_OUTPUT/logs"
FEEBL_GPU2_REPRO_STEPS="$FEEBL_GPU2_REPRO_OUTPUT/steps.tsv"
FEEBL_GPU2_REPRO_COMMANDS="$FEEBL_GPU2_REPRO_OUTPUT/commands.sh"
FEEBL_GPU2_REPRO_RUN_LOG="$FEEBL_GPU2_REPRO_OUTPUT/run.log"
mkdir -p "$FEEBL_GPU2_REPRO_BUILD_ROOT" "$FEEBL_GPU2_REPRO_RESULTS_ROOT" \
  "$FEEBL_GPU2_REPRO_LOG_ROOT" "$FEEBL_GPU2_REPRO_OUTPUT/cards"
cat > "$FEEBL_GPU2_REPRO_OUTPUT/PROMOTION_NON_DECISION.txt" <<'EOF'
This focused campaign records build, component, integration, and optional
timing evidence.  It does not execute the powered stochastic-equivalence,
complete physics-domain, sanitizer, native-Linux Compute Sanitizer, and
fresh-archive qualification matrix.  Its decision_eligible value is therefore
always zero; a PASS is not a production/default-backend promotion decision.
EOF
printf 'sequence\tcategory\tname\tstatus\texit_code\tduration_seconds\tlog\tdetail\n' \
  > "$FEEBL_GPU2_REPRO_STEPS"
printf '#!/usr/bin/env bash\n# Exact commands issued by reproduce_gpu2.sh; paths are campaign-specific.\n' \
  > "$FEEBL_GPU2_REPRO_COMMANDS"
: > "$FEEBL_GPU2_REPRO_RUN_LOG"

FEEBL_GPU2_REPRO_SEQUENCE=0
FEEBL_GPU2_REPRO_HAS_FAIL=0
FEEBL_GPU2_REPRO_HAS_BLOCKED=0
FEEBL_GPU2_REPRO_FINALIZED=0
declare -A FEEBL_GPU2_REPRO_BUILD_OK=()
declare -A FEEBL_GPU2_REPRO_LIST_LOG=()

say()
{
  printf '%s\n' "$*"
  printf '%s\n' "$*" >> "$FEEBL_GPU2_REPRO_RUN_LOG"
}

clean_field()
{
  printf '%s' "$1" | tr '\t\r\n' '   '
}

record_status()
{
  local category=$1 name=$2 status=$3 rc=$4 duration=$5 log=$6 detail=$7
  FEEBL_GPU2_REPRO_SEQUENCE=$((FEEBL_GPU2_REPRO_SEQUENCE + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$FEEBL_GPU2_REPRO_SEQUENCE" "$(clean_field "$category")" \
    "$(clean_field "$name")" "$status" "$rc" "$duration" \
    "$(clean_field "$log")" "$(clean_field "$detail")" \
    >> "$FEEBL_GPU2_REPRO_STEPS"
  case "$status" in
    FAIL) FEEBL_GPU2_REPRO_HAS_FAIL=1 ;;
    BLOCKED) FEEBL_GPU2_REPRO_HAS_BLOCKED=1 ;;
  esac
  say "[$status] $category/$name: $detail"
}

record_blocked()
{
  record_status "$1" "$2" BLOCKED 3 0 "" "$3"
}

record_skipped()
{
  record_status "$1" "$2" SKIPPED 0 0 "" "$3"
}

run_step()
{
  local category=$1 name=$2
  shift 2
  local safe_name log start end rc duration quoted arg
  safe_name=$(printf '%s-%s' "$category" "$name" | tr -c 'A-Za-z0-9_.-' '_')
  log="$FEEBL_GPU2_REPRO_LOG_ROOT/$(printf '%03d' $((FEEBL_GPU2_REPRO_SEQUENCE + 1)))-$safe_name.log"
  quoted=""
  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    quoted+=" $arg"
  done
  {
    printf '\n# %s/%s\n' "$category" "$name"
    printf 'cd %q\n' "$FEEBL_GPU2_REPRO_ROOT"
    printf '%s\n' "${quoted# }"
  } >> "$FEEBL_GPU2_REPRO_COMMANDS"
  say "[RUN] $category/$name:${quoted}"
  start=$(date +%s)
  (cd "$FEEBL_GPU2_REPRO_ROOT" && "$@") > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  duration=$((end - start))
  {
    printf '\n===== %s/%s (exit=%s) =====\n' "$category" "$name" "$rc"
    cat "$log"
  } >> "$FEEBL_GPU2_REPRO_RUN_LOG"
  if [ "$rc" -eq 0 ]; then
    record_status "$category" "$name" PASS 0 "$duration" "$log" "command completed"
    return 0
  fi
  record_status "$category" "$name" FAIL "$rc" "$duration" "$log" "command failed"
  return "$rc"
}

source_revision()
{
  local root
  root=$(git -C "$FEEBL_GPU2_REPRO_ROOT" rev-parse --show-toplevel 2>/dev/null) || root=""
  if [ -n "$root" ] && [ "$(cd "$root" 2>/dev/null && pwd -P)" = "$FEEBL_GPU2_REPRO_ROOT" ]; then
    git -C "$root" rev-parse --verify HEAD 2>/dev/null || echo unknown
  elif [ -r "$FEEBL_GPU2_REPRO_ROOT/SOURCE_REVISION" ]; then
    sed -n '1p' "$FEEBL_GPU2_REPRO_ROOT/SOURCE_REVISION"
  else
    echo unknown
  fi
}

write_environment()
{
  local output=$1
  {
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_root=$FEEBL_GPU2_REPRO_ROOT"
    echo "source_revision=$(source_revision)"
    echo "quick=$FEEBL_GPU2_REPRO_QUICK"
    echo "decision_eligible=0"
    echo "evidence_scope=focused_correctness_and_optional_timing_only"
    echo "benchmarks_requested=$FEEBL_GPU2_REPRO_BENCHMARKS"
    echo "cuda_device_requested=$FEEBL_GPU2_REPRO_DEVICE"
    echo "cuda_architectures=$FEEBL_GPU2_REPRO_ARCH"
    echo "gpu2_host_threads_requested=$FEEBL_GPU2_REPRO_HOST_THREADS"
    echo "command=uname -a"; uname -a 2>&1 || true
    echo "command=lscpu"; lscpu 2>&1 || true
    echo "command=grep MemTotal|MemAvailable /proc/meminfo"
    grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>&1 || true
    echo "command=cmake --version"; cmake --version 2>&1 || true
    echo "command=${CXX:-c++} --version"; "${CXX:-c++}" --version 2>&1 || true
    echo "command=python3 --version"; python3 --version 2>&1 || true
    echo "command=pkg-config --modversion fftw3"; pkg-config --modversion fftw3 2>&1 || true
    echo "command=nvcc --version"; nvcc --version 2>&1 || true
    echo "command=nvidia-smi -L"; nvidia-smi -L 2>&1 || true
    echo "command=nvidia-smi query"
    nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.total,driver_version,pstate,temperature.gpu,power.limit \
      --format=csv,noheader 2>&1 || true
    echo "command=omp environment"
    env | LC_ALL=C sort | grep -E '^(OMP|CUDA|NVIDIA|CMAKE|CC|CXX)_' || true
  } > "$output"
}

configure_build()
{
  local name=$1 openmp=$2 cuda=$3
  local build="$FEEBL_GPU2_REPRO_BUILD_ROOT/$name"
  local args=(cmake -S "$FEEBL_GPU2_REPRO_ROOT" -B "$build"
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DFFTW3=ON
    "-DFEEBL_ENABLE_OPENMP=$openmp" "-DFEEBL_ENABLE_CUDA=$cuda")
  if [ "$cuda" = ON ]; then
    args+=("-DFEEBL_CUDA_ARCHITECTURES=$FEEBL_GPU2_REPRO_ARCH"
      "-DFEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER=${FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER:-OFF}")
    if [ -n "${FEEBL_CUDA_HOST_COMPILER:-}" ]; then
      args+=("-DCMAKE_CUDA_HOST_COMPILER=$FEEBL_CUDA_HOST_COMPILER")
    fi
  fi
  if ! run_step build "$name-configure" "${args[@]}"; then
    FEEBL_GPU2_REPRO_BUILD_OK[$name]=0
    record_skipped build "$name-build" "configuration failed"
    return 1
  fi
  if run_step build "$name-build" cmake --build "$build" -j "$FEEBL_GPU2_REPRO_JOBS"; then
    FEEBL_GPU2_REPRO_BUILD_OK[$name]=1
    return 0
  fi
  FEEBL_GPU2_REPRO_BUILD_OK[$name]=0
  return 1
}

list_ctests()
{
  local name=$1 build="$FEEBL_GPU2_REPRO_BUILD_ROOT/$1"
  if [ "${FEEBL_GPU2_REPRO_BUILD_OK[$name]:-0}" -ne 1 ]; then
    record_skipped test "$name-list" "build unavailable"
    return 1
  fi
  if run_step test "$name-list" ctest --test-dir "$build" -N; then
    FEEBL_GPU2_REPRO_LIST_LOG[$name]="$FEEBL_GPU2_REPRO_LOG_ROOT/$(printf '%03d' "$FEEBL_GPU2_REPRO_SEQUENCE")-test-$name-list.log"
    return 0
  fi
  return 1
}

ctest_names()
{
  local name=$1 log=${FEEBL_GPU2_REPRO_LIST_LOG[$1]:-}
  [ -r "$log" ] || return 1
  sed -n -E 's/^[[:space:]]*Test[[:space:]]+#[0-9]+:[[:space:]]+//p' "$log"
}

run_ctest_group()
{
  local name=$1 group=$2 regex=$3 requirement=$4
  local build="$FEEBL_GPU2_REPRO_BUILD_ROOT/$name" matches
  matches=$(ctest_names "$name" 2>/dev/null | grep -E "$regex" || true)
  if [ -z "$matches" ]; then
    if [ "$requirement" = required ]; then
      record_blocked test "$name-$group" "no registered CTest matched: $regex"
    else
      record_skipped test "$name-$group" "no registered CTest matched: $regex"
    fi
    return 3
  fi
  case "$name" in
    cuda-*)
      run_step test "$name-$group" env \
        "CUDA_VISIBLE_DEVICES=$FEEBL_GPU2_REPRO_DEVICE" \
        ctest --test-dir "$build" -R "$regex" --output-on-failure
      ;;
    *)
      run_step test "$name-$group" \
        ctest --test-dir "$build" -R "$regex" --output-on-failure
      ;;
  esac
}

find_gpu2_integration_script()
{
  local candidate
  for candidate in \
    "$FEEBL_GPU2_REPRO_ROOT/examples/gpu/run_feebl_gpu2_integration.sh" \
    "$FEEBL_GPU2_REPRO_ROOT/examples/gpu/run_gpu2_fields_e2e.sh" \
    "$FEEBL_GPU2_REPRO_ROOT/examples/gpu/run_gpu2_integration.sh" \
    "$FEEBL_GPU2_REPRO_ROOT/examples/gpu/gpu2_integration.sh"; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

finalize()
{
  local incoming_rc=${1:-0} status exit_code
  [ "$FEEBL_GPU2_REPRO_FINALIZED" -eq 0 ] || return
  FEEBL_GPU2_REPRO_FINALIZED=1
  trap - EXIT INT TERM
  set +u
  if [ "$incoming_rc" -ne 0 ] && [ "$FEEBL_GPU2_REPRO_HAS_FAIL" -eq 0 ]; then
    record_status campaign unexpected-exit FAIL "$incoming_rc" 0 "" "campaign terminated unexpectedly"
  fi

  if [ "$FEEBL_GPU2_REPRO_HAS_FAIL" -eq 1 ]; then
    status=FAIL; exit_code=1
  elif [ "$FEEBL_GPU2_REPRO_HAS_BLOCKED" -eq 1 ]; then
    status=BLOCKED; exit_code=3
  else
    status=PASS; exit_code=0
  fi

  if command -v git >/dev/null 2>&1 &&
     git -C "$FEEBL_GPU2_REPRO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$FEEBL_GPU2_REPRO_ROOT" status --short > "$FEEBL_GPU2_REPRO_OUTPUT/source-status.txt" 2>&1 || true
    git -C "$FEEBL_GPU2_REPRO_ROOT" diff --binary > "$FEEBL_GPU2_REPRO_OUTPUT/source-diff.patch" 2>&1 || true
    (
      cd "$FEEBL_GPU2_REPRO_ROOT" || exit
      git ls-files -co --exclude-standard -z 2>/dev/null | LC_ALL=C sort -z |
        while IFS= read -r -d '' FEEBL_GPU2_REPRO_SOURCE_FILE; do
          FEEBL_GPU2_REPRO_SOURCE_ABS="$FEEBL_GPU2_REPRO_ROOT/$FEEBL_GPU2_REPRO_SOURCE_FILE"
          case "$FEEBL_GPU2_REPRO_SOURCE_ABS" in
            "$FEEBL_GPU2_REPRO_OUTPUT"|"$FEEBL_GPU2_REPRO_OUTPUT"/*) continue ;;
          esac
          sha256sum -- "$FEEBL_GPU2_REPRO_SOURCE_FILE"
        done
    ) > "$FEEBL_GPU2_REPRO_OUTPUT/source-checksums.sha256" 2>&1 || true
  else
    echo "git source provenance unavailable" > "$FEEBL_GPU2_REPRO_OUTPUT/source-status.txt"
    : > "$FEEBL_GPU2_REPRO_OUTPUT/source-diff.patch"
    (
      cd "$FEEBL_GPU2_REPRO_ROOT" || exit
      find . -type f ! -path './.git/*' ! -path '*/__pycache__/*' \
        ! -name '*.pyc' -print0 | LC_ALL=C sort -z |
        while IFS= read -r -d '' FEEBL_GPU2_REPRO_SOURCE_FILE; do
          FEEBL_GPU2_REPRO_SOURCE_FILE=${FEEBL_GPU2_REPRO_SOURCE_FILE#./}
          FEEBL_GPU2_REPRO_SOURCE_ABS="$FEEBL_GPU2_REPRO_ROOT/$FEEBL_GPU2_REPRO_SOURCE_FILE"
          case "$FEEBL_GPU2_REPRO_SOURCE_ABS" in
            "$FEEBL_GPU2_REPRO_OUTPUT"|"$FEEBL_GPU2_REPRO_OUTPUT"/*) continue ;;
          esac
          sha256sum -- "$FEEBL_GPU2_REPRO_SOURCE_FILE"
        done
    ) > "$FEEBL_GPU2_REPRO_OUTPUT/source-checksums.sha256" 2>&1 || true
  fi
  say "FeeBL GPU2 reproduction status: $status"
  say "Artifacts: $FEEBL_GPU2_REPRO_OUTPUT"
  (
    cd "$FEEBL_GPU2_REPRO_OUTPUT" || exit
    find build -type f \( -name 'feebl' -o -name 'feebl_*' \) -perm -u+x -print0 2>/dev/null |
      LC_ALL=C sort -z | xargs -0 -r sha256sum
  ) > "$FEEBL_GPU2_REPRO_OUTPUT/binary-checksums.sha256" 2>&1 || true
  (
    cd "$FEEBL_GPU2_REPRO_OUTPUT" || exit
    find cards results -type f \( -name '*.dat' -o -name '*.card' -o -name 'acc.dat' \) -print0 2>/dev/null |
      LC_ALL=C sort -z | xargs -0 -r sha256sum
  ) > "$FEEBL_GPU2_REPRO_OUTPUT/card-checksums.sha256" 2>&1 || true

  {
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "campaign_status=$status"
    echo "campaign_exit_code=$exit_code"
    echo "quick=$FEEBL_GPU2_REPRO_QUICK"
    echo "decision_eligible=0"
    echo "evidence_scope=focused_correctness_and_optional_timing_only"
    echo "benchmarks_requested=$FEEBL_GPU2_REPRO_BENCHMARKS"
    echo "source_revision=$(source_revision)"
    echo "steps_ledger=steps.tsv"
  } > "$FEEBL_GPU2_REPRO_OUTPUT/completion.txt"
  (
    cd "$FEEBL_GPU2_REPRO_OUTPUT" || exit
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z |
      xargs -0 -r sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  ) || {
    echo "artifact integrity manifest failed" >&2
    [ "$exit_code" -ne 0 ] || exit_code=1
  }
  exit "$exit_code"
}
trap 'finalize "$?"' EXIT
trap 'exit 130' INT TERM

FEEBL_GPU2_REPRO_JOBS=${FEEBL_GPU2_BUILD_JOBS:-$(nproc 2>/dev/null || echo 1)}
FEEBL_GPU2_REPRO_PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null |
  awk -F, '!/^#/ {print $1 "," $2}' | LC_ALL=C sort -u | wc -l)
[ "$FEEBL_GPU2_REPRO_PHYSICAL_CORES" -ge 1 ] || FEEBL_GPU2_REPRO_PHYSICAL_CORES=1
FEEBL_GPU2_REPRO_HOST_THREADS=${FEEBL_GPU2_HOST_THREADS:-2}
FEEBL_GPU2_REPRO_REPETITIONS=${FEEBL_GPU2_BENCH_REPETITIONS:-15}
FEEBL_GPU2_REPRO_WARMUPS=${FEEBL_GPU2_BENCH_WARMUPS:-2}
if [ "$FEEBL_GPU2_REPRO_QUICK" -eq 1 ]; then
  FEEBL_GPU2_REPRO_REPETITIONS=1
  FEEBL_GPU2_REPRO_WARMUPS=0
  cat > "$FEEBL_GPU2_REPRO_OUTPUT/QUICK_NON_DECISION.txt" <<'EOF'
This campaign used --quick.  Its reduced repetitions and coverage are useful
only as a smoke test and MUST NOT support a speed, physics, or promotion claim.
EOF
fi
case "$FEEBL_GPU2_REPRO_JOBS:$FEEBL_GPU2_REPRO_HOST_THREADS:$FEEBL_GPU2_REPRO_REPETITIONS:$FEEBL_GPU2_REPRO_WARMUPS" in
  *[!0-9:]*) echo "jobs, threads, repetitions, and warmups must be integers" >&2; exit 2 ;;
esac
if [ "$FEEBL_GPU2_REPRO_JOBS" -lt 1 ] || [ "$FEEBL_GPU2_REPRO_HOST_THREADS" -lt 1 ] ||
   [ "$FEEBL_GPU2_REPRO_REPETITIONS" -lt 1 ]; then
  echo "jobs, host threads, and repetitions must be positive" >&2
  exit 2
fi
if [ "$FEEBL_GPU2_REPRO_HOST_THREADS" -ne 2 ]; then
  echo "the current GPU2 hybrid candidate requires FEEBL_GPU2_HOST_THREADS=2" >&2
  exit 2
fi

write_environment "$FEEBL_GPU2_REPRO_OUTPUT/environment.txt"
if [ -d "$FEEBL_GPU2_REPRO_ROOT/examples/gpu2" ]; then
  run_step evidence copy-example-cards cp -a \
    "$FEEBL_GPU2_REPRO_ROOT/examples/gpu2/." "$FEEBL_GPU2_REPRO_OUTPUT/cards/" || true
else
  record_blocked evidence example-cards "examples/gpu2 is absent"
fi

FEEBL_GPU2_REPRO_CORE_TOOLS_OK=1
for FEEBL_GPU2_REPRO_TOOL in cmake ctest python3 "${CXX:-c++}"; do
  if command -v "$FEEBL_GPU2_REPRO_TOOL" >/dev/null 2>&1; then
    record_status prerequisite "$FEEBL_GPU2_REPRO_TOOL" PASS 0 0 "" "tool available: $(command -v "$FEEBL_GPU2_REPRO_TOOL")"
  else
    record_blocked prerequisite "$FEEBL_GPU2_REPRO_TOOL" "required build/test tool unavailable"
    FEEBL_GPU2_REPRO_CORE_TOOLS_OK=0
  fi
done

if [ "$FEEBL_GPU2_REPRO_CORE_TOOLS_OK" -eq 1 ]; then
  configure_build serial-release OFF OFF || true
  configure_build openmp-release ON OFF || true
else
  for FEEBL_GPU2_REPRO_BUILD in serial-release openmp-release; do
    FEEBL_GPU2_REPRO_BUILD_OK[$FEEBL_GPU2_REPRO_BUILD]=0
    record_blocked build "$FEEBL_GPU2_REPRO_BUILD" "core build tools unavailable"
  done
fi

FEEBL_GPU2_REPRO_CUDA_TOOLKIT=0
FEEBL_GPU2_REPRO_CUDA_RUNTIME=0
if command -v nvcc >/dev/null 2>&1; then
  FEEBL_GPU2_REPRO_CUDA_TOOLKIT=1
  record_status prerequisite nvcc PASS 0 0 "" "CUDA compiler available: $(command -v nvcc)"
else
  record_blocked prerequisite nvcc "CUDA compiler unavailable; CUDA builds blocked"
fi
if command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
     grep -Fx "$FEEBL_GPU2_REPRO_DEVICE" >/dev/null; then
  FEEBL_GPU2_REPRO_CUDA_RUNTIME=1
  record_status prerequisite cuda-device PASS 0 0 "" \
    "nvidia-smi reports requested physical device $FEEBL_GPU2_REPRO_DEVICE"
else
  record_blocked prerequisite cuda-device \
    "requested physical device $FEEBL_GPU2_REPRO_DEVICE is absent from the nvidia-smi index list"
fi

if [ "$FEEBL_GPU2_REPRO_CORE_TOOLS_OK" -eq 1 ] && [ "$FEEBL_GPU2_REPRO_CUDA_TOOLKIT" -eq 1 ]; then
  configure_build cuda-mode1-release OFF ON || true
  configure_build cuda-mode2-1t-release OFF ON || true
  configure_build cuda-openmp-release ON ON || true
else
  for FEEBL_GPU2_REPRO_BUILD in cuda-mode1-release cuda-mode2-1t-release cuda-openmp-release; do
    FEEBL_GPU2_REPRO_BUILD_OK[$FEEBL_GPU2_REPRO_BUILD]=0
    record_blocked build "$FEEBL_GPU2_REPRO_BUILD" "CUDA or core build tools unavailable"
  done
fi

for FEEBL_GPU2_REPRO_BUILD in serial-release openmp-release cuda-mode1-release cuda-mode2-1t-release cuda-openmp-release; do
  list_ctests "$FEEBL_GPU2_REPRO_BUILD" || true
done

run_ctest_group serial-release cpu-focus \
  '^feebl-(cpu-backend-unit|cpu-backends-end-to-end|backend-interface|gpu2-backend-unit)$' required || true
run_ctest_group openmp-release cpu-focus \
  '^feebl-(cpu-backend-unit|cpu-backends-end-to-end|backend-interface|gpu2-backend-unit)$' required || true
for FEEBL_GPU2_REPRO_BUILD in cuda-mode1-release cuda-mode2-1t-release cuda-openmp-release; do
  run_ctest_group "$FEEBL_GPU2_REPRO_BUILD" host-interface \
    '^feebl-(backend-interface|gpu2-backend-unit)$' required || true
done

if [ "$FEEBL_GPU2_REPRO_CUDA_RUNTIME" -eq 1 ]; then
  run_ctest_group cuda-mode1-release mode1-component \
    '^feebl-gpu-cic-correctness$' required || true
  for FEEBL_GPU2_REPRO_BUILD in cuda-mode2-1t-release cuda-openmp-release; do
    run_ctest_group "$FEEBL_GPU2_REPRO_BUILD" gpu2-component \
      '^feebl-(gpu2-(resident|fft|cic).*|gpu-resident.*)$' required || true
    run_ctest_group "$FEEBL_GPU2_REPRO_BUILD" gpu2-integration \
      '^feebl-(gpu2.*(end-to-end|integration).*|gpu-resident.*(end-to-end|integration).*)$' required || true
  done
else
  record_blocked test cuda-mode1-device "CUDA runtime/device unavailable"
  record_blocked test cuda-mode2-1t-device "CUDA runtime/device unavailable"
  record_blocked test cuda-openmp-device "CUDA runtime/device unavailable"
fi

if [ "$FEEBL_GPU2_REPRO_CUDA_RUNTIME" -eq 1 ] &&
   [ "${FEEBL_GPU2_REPRO_BUILD_OK[cuda-mode1-release]:-0}" -eq 1 ]; then
  run_step integration mode1-production-deck \
    env "CUDA_VISIBLE_DEVICES=$FEEBL_GPU2_REPRO_DEVICE" \
      bash "$FEEBL_GPU2_REPRO_ROOT/examples/gpu/run_feebl_gpu_integration.sh" \
      "$FEEBL_GPU2_REPRO_BUILD_ROOT/cuda-mode1-release/src/feebl" \
      "$FEEBL_GPU2_REPRO_RESULTS_ROOT/mode1-integration" || true
else
  record_blocked integration mode1-production-deck "mode-1 CUDA binary/device unavailable"
fi

FEEBL_GPU2_REPRO_INTEGRATION_SCRIPT=$(find_gpu2_integration_script || true)
if [ -n "$FEEBL_GPU2_REPRO_INTEGRATION_SCRIPT" ] && [ "$FEEBL_GPU2_REPRO_CUDA_RUNTIME" -eq 1 ]; then
  for FEEBL_GPU2_REPRO_PAIR in \
    "cuda-mode2-1t-release:gpu2-one-thread" \
    "cuda-openmp-release:gpu2-cuda-openmp-hybrid"; do
    FEEBL_GPU2_REPRO_BUILD=${FEEBL_GPU2_REPRO_PAIR%%:*}
    FEEBL_GPU2_REPRO_LABEL=${FEEBL_GPU2_REPRO_PAIR#*:}
    if [ "${FEEBL_GPU2_REPRO_BUILD_OK[$FEEBL_GPU2_REPRO_BUILD]:-0}" -eq 1 ]; then
      run_step integration "$FEEBL_GPU2_REPRO_LABEL" env \
        "CUDA_VISIBLE_DEVICES=$FEEBL_GPU2_REPRO_DEVICE" \
        "FEEBL_GPU2_RESULTS_DIR=$FEEBL_GPU2_REPRO_RESULTS_ROOT/$FEEBL_GPU2_REPRO_LABEL" \
        bash "$FEEBL_GPU2_REPRO_INTEGRATION_SCRIPT" \
          "$FEEBL_GPU2_REPRO_BUILD_ROOT/$FEEBL_GPU2_REPRO_BUILD/src/feebl" \
          "$FEEBL_GPU2_REPRO_RESULTS_ROOT/$FEEBL_GPU2_REPRO_LABEL" || true
    else
      record_blocked integration "$FEEBL_GPU2_REPRO_LABEL" "required build unavailable"
    fi
  done
else
  if [ -z "$FEEBL_GPU2_REPRO_INTEGRATION_SCRIPT" ]; then
    record_blocked integration gpu2-production-deck "no GPU2 integration script is present"
  else
    record_blocked integration gpu2-production-deck "CUDA runtime/device unavailable"
  fi
fi

if [ "$FEEBL_GPU2_REPRO_BENCHMARKS" -eq 1 ]; then
  if [ "$FEEBL_GPU2_REPRO_CUDA_RUNTIME" -eq 1 ] &&
     [ "${FEEBL_GPU2_REPRO_BUILD_OK[cuda-mode1-release]:-0}" -eq 1 ] &&
     [ -x "$FEEBL_GPU2_REPRO_BUILD_ROOT/cuda-mode1-release/src/feebl_gpu_cic_test" ]; then
    run_step benchmark mode1-cic \
      "$FEEBL_GPU2_REPRO_BUILD_ROOT/cuda-mode1-release/src/feebl_gpu_cic_test" \
      --device "$FEEBL_GPU2_REPRO_DEVICE" \
      --benchmark "$FEEBL_GPU2_REPRO_RESULTS_ROOT/mode1-cic-benchmark.csv" \
      --repetitions "$FEEBL_GPU2_REPRO_REPETITIONS" \
      --max-particles "${FEEBL_GPU2_MAX_PARTICLES:-4194304}" || true
  else
    record_blocked benchmark mode1-cic "mode-1 benchmark executable/device unavailable"
  fi
  if [ "$FEEBL_GPU2_REPRO_CUDA_RUNTIME" -eq 1 ] &&
     [ "${FEEBL_GPU2_REPRO_BUILD_OK[cuda-openmp-release]:-0}" -eq 1 ]; then
    FEEBL_GPU2_REPRO_E2E_ARGS=(python3 "$FEEBL_GPU2_REPRO_ROOT/tools/benchmark_backends.py"
      --binary "$FEEBL_GPU2_REPRO_BUILD_ROOT/cuda-openmp-release/src/feebl"
      --source-root "$FEEBL_GPU2_REPRO_ROOT"
      --deck "$FEEBL_GPU2_REPRO_ROOT/examples/decks/quick.dat"
      --case c_nt1 --variant serial --variant cpu_dual \
      --variant cpu_private --variant mode1 --variant gpu2_1t \
      --variant gpu2_atomic --variant gpu2_hybrid
      --baseline serial --warmups "$FEEBL_GPU2_REPRO_WARMUPS"
      --repetitions "$FEEBL_GPU2_REPRO_REPETITIONS"
      --output "$FEEBL_GPU2_REPRO_RESULTS_ROOT/gpu2-e2e-benchmark"
      --gpu-device "$FEEBL_GPU2_REPRO_DEVICE" --max-other-feebl 0
      --observable 'lumi_fine=^lumi_fine\s*=\s*([-+0-9.eE]+)'
      --observable 'lumi_ee=^lumi_ee\s*=\s*([-+0-9.eE]+)\s+m'
      --observable 'lumi_ee_high=^lumi_ee_high\s*=\s*([-+0-9.eE]+)'
      --observable 'upsmax=^upsmax=\s*([-+0-9.eE]+)'
      --deck-fragment 'serial=compute_backend="serial";'
      --deck-fragment 'cpu_dual=compute_backend="cpu";cpu_threads=2;cpu_variant="pool_dual";reduction_mode="strict";allow_backend_fallback=0;'
      --deck-fragment "cpu_private=compute_backend=\"cpu\";cpu_threads=$FEEBL_GPU2_REPRO_PHYSICAL_CORES;cpu_variant=\"pool_private\";reduction_mode=\"deterministic\";allow_backend_fallback=0;"
      --deck-fragment "mode1=compute_backend=\"gpu\";gpu_mode=1;gpu_variant=\"atomic\";gpu_device=$FEEBL_GPU2_REPRO_DEVICE;gpu_min_particles=0;reduction_mode=\"fast\";allow_backend_fallback=0;"
      --deck-fragment "gpu2_1t=compute_backend=\"gpu\";gpu_mode=2;gpu2_scope=\"fields\";gpu2_deposition=\"auto\";gpu2_fft=\"cufft\";gpu2_streams=1;gpu2_graphs=\"off\";gpu2_fusion=\"off\";gpu2_batch_slices=1;gpu2_pinned_io=\"on\";gpu2_host_threads=1;gpu2_memory_fraction=0.80;gpu2_profile=\"summary\";gpu_device=$FEEBL_GPU2_REPRO_DEVICE;reduction_mode=\"deterministic\";allow_backend_fallback=0;"
      --deck-fragment "gpu2_atomic=compute_backend=\"gpu\";gpu_mode=2;gpu2_scope=\"fields\";gpu2_deposition=\"atomic\";gpu2_fft=\"cufft\";gpu2_streams=1;gpu2_graphs=\"off\";gpu2_fusion=\"off\";gpu2_batch_slices=1;gpu2_pinned_io=\"on\";gpu2_host_threads=1;gpu2_memory_fraction=0.80;gpu2_profile=\"summary\";gpu_device=$FEEBL_GPU2_REPRO_DEVICE;reduction_mode=\"fast\";allow_backend_fallback=0;"
      --deck-fragment "gpu2_hybrid=compute_backend=\"gpu\";gpu_mode=2;gpu2_scope=\"fields\";gpu2_deposition=\"auto\";gpu2_fft=\"cufft\";gpu2_streams=1;gpu2_graphs=\"off\";gpu2_fusion=\"off\";gpu2_batch_slices=1;gpu2_pinned_io=\"on\";gpu2_host_threads=$FEEBL_GPU2_REPRO_HOST_THREADS;gpu2_memory_fraction=0.80;gpu2_profile=\"summary\";gpu_device=$FEEBL_GPU2_REPRO_DEVICE;reduction_mode=\"deterministic\";allow_backend_fallback=0;")
    if run_step benchmark gpu2-end-to-end "${FEEBL_GPU2_REPRO_E2E_ARGS[@]}"; then
      run_step benchmark gpu2-end-to-end-validate python3 -c '
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
required_observables = {"lumi_fine", "lumi_ee", "lumi_ee_high", "upsmax"}
expected_variants = {"serial", "cpu_dual", "cpu_private", "mode1",
                     "gpu2_1t", "gpu2_atomic", "gpu2_hybrid"}
if not rows:
    raise SystemExit("benchmark produced no sample records")
if {row.get("variant") for row in rows} != expected_variants:
    raise SystemExit("benchmark variant set is incomplete or unexpected")
bad_rc = [row for row in rows if row.get("returncode") != 0]
if bad_rc:
    raise SystemExit(f"{len(bad_rc)} benchmark samples exited nonzero")
missing = [row for row in rows
           if not required_observables.issubset(row.get("observables", {}))]
if missing:
    raise SystemExit(f"{len(missing)} samples omitted preregistered observables")
serial_by_round = {
    (row["case"], row["repetition"], row["warmup"]): row
    for row in rows if row["variant"] == "serial"
}
# Tolerances are frozen by algorithm class, before inspecting this campaign:
# strict preserves the oracle; deterministic field solvers get a tight FFT
# allowance; explicitly fast/unordered reductions use the established 5e-6
# whole-program bound.  The absolute term protects exact/near-zero fixtures.
tolerances = {
    "cpu_dual": (0.0, 0.0),
    "gpu2_1t": (1.0e-12, 1.0e-10),
    "gpu2_hybrid": (1.0e-12, 1.0e-10),
    "cpu_private": (1.0e-12, 5.0e-6),
    "mode1": (1.0e-12, 5.0e-6),
    "gpu2_atomic": (1.0e-12, 5.0e-6),
}
for row in rows:
    if row["variant"] != "serial":
        key = (row["case"], row["repetition"], row["warmup"])
        reference = serial_by_round.get(key)
        if reference is None:
            raise SystemExit(f"sample {key} lacks its paired serial oracle")
        atol, rtol = tolerances[row["variant"]]
        for observable in sorted(required_observables):
            expected = float(reference["observables"][observable])
            observed = float(row["observables"][observable])
            limit = atol + rtol * abs(expected)
            if abs(observed - expected) > limit:
                raise SystemExit(
                    f"{row['variant']} round {key} observable {observable} "
                    f"differs: observed={observed:.17g} expected={expected:.17g} "
                    f"limit={limit:.17g}")
        if row["variant"] == "cpu_dual" and (
                row.get("normalized_sha256") != reference.get("normalized_sha256")):
            raise SystemExit("strict cpu_dual sample is not normalized-byte-identical to serial")
    stderr = (pathlib.Path(row["run_directory"]) / "stderr.log").read_text(
        errors="replace")
    if "backend_fallback" in stderr or "fallback=" in stderr:
        raise SystemExit(f"{row['variant']} timing sample used a fallback")
    if row["variant"] == "mode1" and not re.search(
            r"^\s*FEEBL: final (?=[^\n]*gpu_mode requested=1 actual=1)"
            r"(?=[^\n]*gpu_calls=[1-9][0-9]*)[^\n]*$", stderr,
            re.MULTILINE):
        raise SystemExit("mode1 sample lacks proof of actual CUDA execution")
    if row["variant"] == "gpu2_1t" and not re.search(
            r"FEEBL GPU2 fields summary:.*device_work_calls=[1-9][0-9]*", stderr):
        raise SystemExit("GPU2 sample lacks proof of actual resident-field execution")
    if row["variant"] == "gpu2_atomic" and not re.search(
            r"FEEBL GPU2 fields summary:.*deposition=atomic.*deposition_calls=[1-9][0-9]*", stderr):
        raise SystemExit("GPU2 atomic sample lacks proof of direct resident deposition")
    if row["variant"] == "gpu2_hybrid" and not re.search(
            r"cpu_variant requested=auto actual=gpu2_openmp_host.*gpu2_host_threads requested=2 actual=2", stderr):
        raise SystemExit("GPU2 hybrid sample lacks proof of two-worker host execution")
    if row["variant"].startswith("cpu_") and not re.search(
            r"compute_backend requested=cpu actual=cpu", stderr):
        raise SystemExit("CPU candidate sample lacks proof of actual CPU execution")
' "$FEEBL_GPU2_REPRO_RESULTS_ROOT/gpu2-e2e-benchmark/samples.jsonl" || true
    fi
  else
    record_blocked benchmark gpu2-end-to-end "combined CUDA+OpenMP binary/device unavailable"
  fi
else
  record_skipped benchmark all "benchmarks are opt-in; rerun with --benchmarks"
fi

if command -v python3 >/dev/null 2>&1; then
  run_step evidence summarize-gpu2-statistics \
    python3 "$FEEBL_GPU2_REPRO_ROOT/tools/summarize_gpu2_stats.py" \
      "$FEEBL_GPU2_REPRO_LOG_ROOT" "$FEEBL_GPU2_REPRO_RESULTS_ROOT" \
      --csv "$FEEBL_GPU2_REPRO_OUTPUT/gpu2-statistics.csv" \
      --json "$FEEBL_GPU2_REPRO_OUTPUT/gpu2-statistics.json" || true
else
  record_blocked evidence summarize-gpu2-statistics "python3 unavailable"
fi

finalize 0
