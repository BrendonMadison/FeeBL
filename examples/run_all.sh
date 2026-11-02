#!/bin/bash
# Final verification on a frozen tree, in one invocation.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
# Refuse to run inside another run. The meta test copies the tree and
# executes what it finds; without this, run_all -> dead_binary -> run_all
# recurses, and each level rewrites the results file of the one above it.
if [ -n "${FEEBL_RUN_ALL_ACTIVE:-}" ]; then
  echo "examples/run_all.sh: already running (FEEBL_RUN_ALL_ACTIVE); refusing to recurse." >&2
  exit 0
fi
export FEEBL_RUN_ALL_ACTIVE=1
RUN_ALL_FAILURES=0

# This historical aggregate includes differential tests against an external
# reference. CTest is the self-contained alternative for a source release.
if [ -z "${FEEBL_REF:-}" ] || [ ! -x "$FEEBL_REF" ]; then
  echo "Set FEEBL_REF to an executable reference build, or run the CTest suite." >&2
  exit 2
fi

source_revision () {
  local source_root git_root revision dirty_suffix
  source_root=$(cd "$SRC" 2>/dev/null && pwd -P) || source_root=
  git_root=$(git -C "$SRC" rev-parse --show-toplevel 2>/dev/null) || git_root=
  if [ -n "$git_root" ]; then
    git_root=$(cd "$git_root" 2>/dev/null && pwd -P) || git_root=
  fi
  if [ -n "$source_root" ] && [ "$git_root" = "$source_root" ]; then
    revision=$(git -C "$SRC" rev-parse --short HEAD)
    dirty_suffix=
    if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=normal --ignore-submodules)" ]; then
      dirty_suffix=-dirty
    fi
    printf '%s%s\n' "$revision" "$dirty_suffix"
  elif [ -r "$SRC/SOURCE_REVISION" ]; then
    revision=$(sed -n '1p' "$SRC/SOURCE_REVISION")
    if printf '%s\n' "$revision" | LC_ALL=C grep -Eq '^([0-9a-fA-F]{40}(-dirty)?|sha256:[0-9a-fA-F]{64})$'; then
      case "$revision" in
        *-dirty) printf '%.12s-dirty\n' "$revision" ;;
        *) printf '%.12s\n' "$revision" ;;
      esac
    else
      echo unknown
    fi
  else
    echo unknown
  fi
}

# Results go INSIDE the tree being tested, not into $HOME. An absolute path
# escapes the meta test's sandbox copy and overwrites the real record.
RES=${FEEBL_RESULTS:-$SRC/TEST_RESULTS.txt}
cd "$SRC"

run_suite () {
  echo; echo "############ $1 ############"; shift
  set +e; "$@" 2>&1; local rc=$?; set -e
  echo "[exit status: $rc]"
  if [ "$rc" -ne 0 ]; then
    RUN_ALL_FAILURES=$((RUN_ALL_FAILURES + 1))
  fi
}

{
  echo "FEEBL 0.6.0-dev -- test results"
  echo "revision : $(source_revision)"
  echo "toolchain: $(g++ --version | head -1)"
  echo "host     : $(uname -srm)"
  echo
  echo "The runner does not intentionally modify source files or binaries."

  run_suite "META: can the suites fail at all?"              bash ./examples/meta/dead_binary.sh --quick
  run_suite "version identity"                              bash ./examples/version_identity.sh
  run_suite "bitcheck: bit-identical against v0.5.0"        bash ./examples/bitcheck.sh
  run_suite "quick suite vs goldens"                        bash ./examples/run.sh quick
  run_suite "exact differential vs the v0.5.0 binary"       bash ./examples/run.sh quick --exact "$FEEBL_REF"
  run_suite "negative checks: failures must report failure" bash ./examples/negative_checks.sh
  run_suite "reproducibility: location-independent suites"  bash ./examples/reprocheck.sh
  run_suite "native event output"                           bash ./examples/native_output.sh
  run_suite "native output: read back and validated"        bash ./examples/native_roundtrip.sh
  run_suite "conventions (Stage A)"                         bash ./examples/conventions/run.sh
  run_suite "canonical tier and HDF5 reader (Stage C)"      bash ./examples/canonical/run.sh
  run_suite "incoming maps, crab and the guards (Stage D)"  bash ./examples/maps/run.sh
  run_suite "geometry, Level 3, rotation, scope (Stage E)"  bash ./examples/geometry/run.sh
  run_suite "counter RNG V1/V2 and CPU/CUDA parity"          bash ./examples/rng/run.sh
  run_suite "spin: BMT, norm, symmetry, polarization, P_lumi" bash ./examples/spin/run.sh
  run_suite "pairs: exact W, weighted moments, dL/dW"        bash ./examples/pairs/run.sh

  echo
  echo "############ benchmark16 ############"
  echo "NOT RUN by this aggregate driver; no benchmark16 pass is claimed here."
  echo "Run and archive it separately with: bash examples/run.sh benchmark16 -j 16"
} > "$RES" 2>&1

echo "  lines            : $(wc -l < "$RES")"
echo "  suites recorded  : $(grep -c 'exit status' "$RES")"
echo "  non-zero exits   : $(grep -c 'exit status: [^0]' "$RES")"
grep 'exit status: [^0]' "$RES" | sed 's/^/    /' || true
echo "  FAIL/DIFFERS     : $(grep -cE 'FAIL|DIFFERS' "$RES")"
grep -nE 'FAIL|DIFFERS' "$RES" | head -5 | sed 's/^/    /' || true
if [ "$RUN_ALL_FAILURES" -ne 0 ]; then
  echo "examples/run_all.sh: $RUN_ALL_FAILURES suite(s) exited non-zero" >&2
  exit 1
fi
