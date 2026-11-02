#!/bin/bash
# Reproducible fresh-process stress wrapper for the resident CUDA component.
# Each invocation retains separate stdout, stderr, and numeric return code.
set -u

BIN=${1:?resident test executable required}
OUT=${2:?output directory required}
REPEATS=${3:-5}

[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }
case "$REPEATS" in
  ''|*[!0-9]*) echo "repeat count must be a positive integer" >&2; exit 2 ;;
esac
[ "$REPEATS" -gt 0 ] || { echo "repeat count must be positive" >&2; exit 2; }
mkdir -p "$OUT" || exit 2

overall=0
for mode in launch-blocking normal; do
  iteration=1
  while [ "$iteration" -le "$REPEATS" ]; do
    prefix="$OUT/$mode-$iteration"
    if [ "$mode" = launch-blocking ]; then
      CUDA_LAUNCH_BLOCKING=1 "$BIN" >"$prefix.stdout" 2>"$prefix.stderr"
    else
      env -u CUDA_LAUNCH_BLOCKING "$BIN" >"$prefix.stdout" 2>"$prefix.stderr"
    fi
    rc=$?
    printf '%s\n' "$rc" >"$prefix.rc"
    printf '%s run %d rc=%d\n' "$mode" "$iteration" "$rc"
    [ "$rc" -eq 0 ] || overall=1
    iteration=$((iteration + 1))
  done
done

exit "$overall"
