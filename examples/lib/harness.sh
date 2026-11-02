#!/bin/bash
# Shared test helpers: require successful simulation and nonempty numerical
# output before comparing results. Source this file before running a fixture:
#
#     . "$ROOT/examples/lib/harness.sh"
#     feebl_run "$B" c_nt1 p_c_nt1 o.dat
#     L=$(need_num o.dat lumi_fine)
#
# Both helpers terminate on failure so missing output cannot count as agreement.

# feebl_run BIN [ARGS...]
#   Runs FEEBL. Aborts unless it exits 0 AND wrote a non-empty output file.
#   The output file is taken to be the last argument, which is FEEBL's
#   convention (`feebl <accelerator> <parameters> <output>`).
feebl_run () {
  local bin="$1"; shift
  local out=""
  for a in "$@"; do out="$a"; done      # last argument
  [ -n "$out" ] || { echo "  FATAL: feebl_run needs an output argument" >&2; exit 90; }

  rm -f "$out"
  local rc=0
  "$bin" "$@" > feebl_stdout.txt 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FATAL: $bin exited $rc -- the simulation did not run." >&2
    echo "  ---- first lines of its output ----" >&2
    sed -n '1,12p' feebl_stdout.txt >&2
    exit 90
  fi
  if [ ! -s "$out" ]; then
    echo "  FATAL: $bin exited 0 but wrote no (or empty) $out." >&2
    sed -n '1,12p' feebl_stdout.txt >&2
    exit 91
  fi
}

# need_num FILE KEY
#   Echoes the value of a "KEY = value" line. Aborts if the key is absent or the
#   value does not parse as a number. Assign it -- do NOT use it inline in
#   printf, because a command substitution used as an argument discards the exit
#   status and the abort would be silently swallowed:
#
#       L=$(need_num o.dat lumi_fine)          # correct
#       printf '%s' "$(need_num o.dat lumi_fine)"   # WRONG, status discarded
need_num () {
  local file="$1" key="$2" v
  if [ ! -s "$file" ]; then
    echo "  FATAL: $file is missing or empty; cannot read '$key'." >&2
    exit 92
  fi
  v=$(grep -m1 -- "$key =" "$file" 2>/dev/null | sed 's/.*= *//;s/ .*//')
  if [ -z "$v" ]; then
    echo "  FATAL: key '$key' absent from $file -- nothing was measured." >&2
    exit 92
  fi
  if ! awk -v x="$v" 'BEGIN{ exit (x ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) ? 0 : 1 }'; then
    echo "  FATAL: key '$key' in $file is not numeric (got '$v')." >&2
    exit 92
  fi
  printf '%s' "$v"
}

# need_nonempty FILE  -- abort unless FILE exists and has content.
need_nonempty () {
  [ -s "$1" ] || { echo "  FATAL: expected output $1 is missing or empty." >&2; exit 91; }
}
