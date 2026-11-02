#!/bin/bash
# Diagnostic interleaved end-to-end benchmark for FeeBL CPU candidates.
# Release evidence must use benchmark_backends.py or the reproduction driver,
# which retain per-run decks, outputs, logs, hashes, and idle-gate records.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/src/feebl
DECK=$ROOT/examples/decks/quick.dat
ACC=c_nt1
PAR=p_c_nt1
THREADS=1,2,4
REPEATS=7
VARIANTS=serial,pool_dual,pool_private,omp_atomic
OUT=cpu_backend_bench.csv

usage()
{
  echo "usage: $0 [--binary PATH] [--deck PATH] [--accelerator NAME]"
  echo "          [--parameters NAME] [--threads 1,2,4] [--repeats N]"
  echo "          [--variants LIST] [--output FILE]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --binary) BIN=$2; shift 2;;
    --deck) DECK=$2; shift 2;;
    --accelerator) ACC=$2; shift 2;;
    --parameters) PAR=$2; shift 2;;
    --threads) THREADS=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --variants) VARIANTS=$2; shift 2;;
    --output) OUT=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1"; usage; exit 2;;
  esac
done

absolute_path()
{
  local path=$1 directory basename
  directory=$(dirname "$path")
  basename=$(basename "$path")
  (cd "$directory" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$basename")
}

source_git_root()
{
  local source_root git_root
  source_root=$(cd "$ROOT" 2>/dev/null && pwd -P) || source_root=
  git_root=$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null) || git_root=
  if [ -n "$git_root" ]; then
    git_root=$(cd "$git_root" 2>/dev/null && pwd -P) || git_root=
  fi
  if [ -n "$source_root" ] && [ "$git_root" = "$source_root" ]; then
    printf '%s\n' "$git_root"
    return 0
  fi
  return 1
}

source_revision()
{
  local git_root revision
  if git_root=$(source_git_root); then
    revision=$(git -C "$git_root" rev-parse --verify HEAD)
    if [ -n "$(git -C "$git_root" status --porcelain --untracked-files=normal --ignore-submodules)" ]; then
      revision="$revision-dirty"
    fi
    printf '%s\n' "$revision"
  elif [ -r "$ROOT/SOURCE_REVISION" ]; then
    revision=$(sed -n '1p' "$ROOT/SOURCE_REVISION")
    if printf '%s\n' "$revision" | LC_ALL=C grep -Eq '^([0-9a-fA-F]{40}(-dirty)?|sha256:[0-9a-fA-F]{64})$'; then
      printf '%s\n' "$revision"
    else
      echo unknown
    fi
  else
    echo unknown
  fi
}

[ -x "$BIN" ] || { echo "no executable binary: $BIN"; exit 2; }
[ -f "$DECK" ] || { echo "no input deck: $DECK"; exit 2; }
case "$REPEATS" in *[!0-9]*|'') echo "repeats must be a positive integer"; exit 2;; esac
[ "$REPEATS" -gt 0 ] || { echo "repeats must be positive"; exit 2; }

# Each simulation runs from a temporary directory. Resolve user-supplied
# relative paths before the first chdir so a missing binary cannot turn into a
# misleading sub-millisecond "benchmark" with exit code 127.
BIN=$(absolute_path "$BIN") || { echo "cannot resolve binary path: $BIN"; exit 2; }
DECK=$(absolute_path "$DECK") || { echo "cannot resolve deck path: $DECK"; exit 2; }
mkdir -p "$(dirname "$OUT")"
OUT=$(absolute_path "$OUT") || { echo "cannot resolve output path: $OUT"; exit 2; }

WORK=$(mktemp -d)
trap 'test "${KEEP_WORK:-0}" = 1 || rm -rf "$WORK"' EXIT
MANIFEST=${OUT%.csv}.manifest.txt
mkdir -p "$(dirname "$MANIFEST")"
FAILURES=0

{
  echo "evidence_class=diagnostic_non_archival"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "binary=$BIN"
  sha256sum "$BIN" "$DECK"
  sha256sum "$0"
  uname -a
  lscpu
  printf 'loadavg_before='
  cat /proc/loadavg
  echo "source_revision=$(source_revision)"
  echo "source_status_begin"
  if FEEBL_CPU_GIT_ROOT=$(source_git_root); then
    git -C "$FEEBL_CPU_GIT_ROOT" status --short
  else
    echo "not-a-source-repository"
  fi
  echo "source_status_end"
  "$BIN" --version
  echo "threads=$THREADS"
  echo "variants=$VARIANTS"
  echo "repeats=$REPEATS"
  echo "affinity=${FEEBL_CPUSET:-not-pinned}"
  echo "case_order=cyclic_rotation_by_repetition"
} > "$MANIFEST"

echo 'variant,threads,repetition,elapsed_seconds,exit_code,lumi_0_1,actual_backend,actual_variant' > "$OUT"

make_deck()
{
  local config=$1 output=$2
  awk -v target="$PAR" -v config="$config" '
    /^\$PARAMETERS::/ { active=($2==target) }
    active && /}/ { sub(/}/, config "}"); active=0 }
    { print }
  ' "$DECK" > "$output"
}

run_one()
{
  local variant=$1 threads=$2 repetition=$3 config reduction run start end elapsed rc got actual actual_variant
  reduction=fast
  case "$variant" in
    serial) config='compute_backend="serial";'; threads=1;;
    pool_dual) reduction=strict; config="compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"pool_dual\";reduction_mode=\"$reduction\";";;
    pool_private) reduction=deterministic; config="compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"pool_private\";reduction_mode=\"$reduction\";";;
    omp_atomic) config="compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"omp_atomic\";reduction_mode=\"fast\";";;
    *) echo "unknown variant in list: $variant"; return 2;;
  esac

  run=$WORK/${repetition}_${variant}_${threads}
  mkdir -p "$run"
  make_deck "$config" "$run/acc.dat"
  start=$(date +%s%N)
  if [ -n "${FEEBL_CPUSET:-}" ] && command -v taskset >/dev/null 2>&1; then
    (cd "$run" && taskset -c "$FEEBL_CPUSET" "$BIN" "$ACC" "$PAR" result.out >stdout 2>stderr)
  else
    (cd "$run" && "$BIN" "$ACC" "$PAR" result.out >stdout 2>stderr)
  fi
  rc=$?
  if [ "$rc" -ne 0 ]; then FAILURES=$((FAILURES+1)); fi
  end=$(date +%s%N)
  elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.9f",(b-a)/1e9}')
  got=$(grep -m1 'lumi\[0\]\[1\]' "$run/result.out" 2>/dev/null | sed 's/.*= *//')
  actual=$(sed -n 's/.* actual=\([^ ]*\) cpu_variant.*/\1/p' "$run/stderr" | head -1)
  actual_variant=$(sed -n 's/.*cpu_variant requested=[^ ]* actual=\([^ ]*\).*/\1/p' "$run/stderr" | head -1)
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$variant" "$threads" "$repetition" \
    "$elapsed" "$rc" "${got:-NA}" "${actual:-NA}" "${actual_variant:-NA}" >> "$OUT"
}

variant_list=${VARIANTS//,/ }
thread_list=${THREADS//,/ }

# Build the case list once and rotate its starting point on every repetition.
# This deterministic Latin-cycle-like ordering avoids always granting serial
# the same cache, turbo, and thermal position while remaining reproducible.
case_variants=()
case_threads=()
for variant in $variant_list; do
  if [ "$variant" = serial ]; then
    case_variants+=(serial)
    case_threads+=(1)
  else
    for threads in $thread_list; do
      case_variants+=("$variant")
      case_threads+=("$threads")
    done
  fi
done
case_count=${#case_variants[@]}
[ "$case_count" -gt 0 ] || { echo "no benchmark cases selected"; exit 2; }

rep=1
while [ "$rep" -le "$REPEATS" ]; do
  offset=$(((rep-1)%case_count))
  step=0
  while [ "$step" -lt "$case_count" ]; do
    index=$(((offset+step)%case_count))
    run_one "${case_variants[$index]}" "${case_threads[$index]}" "$rep"
    step=$((step+1))
  done
  rep=$((rep+1))
done

{
  printf 'loadavg_after='
  cat /proc/loadavg
} >> "$MANIFEST"

echo "raw results: $OUT"
echo "manifest:    $MANIFEST"
echo "Keep cold starts, failures, and regressions in the analysis; do not select"
echo "a winner from kernel timing or successful cases alone."
if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES benchmark case(s) failed; inspect exit_code and stderr" >&2
  exit 1
fi
