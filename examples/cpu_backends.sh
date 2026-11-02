#!/bin/bash
# Focused end-to-end correctness gate for the intra-simulation CPU backends.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
DECK=$ROOT/examples/decks/quick.dat
[ -x "$BIN" ] || { echo "no executable FeeBL binary: $BIN"; exit 2; }

WORK=$(mktemp -d)
trap 'test "${KEEP_WORK:-0}" = 1 || rm -rf "$WORK"' EXIT
failures=0
tests=0

pass()
{
  tests=$((tests+1))
  printf "  %-66s PASS\n" "$1"
}

fail()
{
  tests=$((tests+1)); failures=$((failures+1))
  printf "  %-66s FAIL\n" "$1"
}

make_deck()
{
  local config=$1 out=$2
  mkdir -p "$out"
  awk -v config="$config" '
    { sub(/^hist_ee_bins=1010;/, config "hist_ee_bins=1010;"); print }
  ' "$DECK" > "$out/acc.dat"
}

run_case()
{
  local name=$1 config=$2
  LAST_DIR=$WORK/$name
  make_deck "$config" "$LAST_DIR"
  (cd "$LAST_DIR" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr)
  LAST_RC=$?
}

normalize()
{
  # Strip only the deliberately variable execution-provenance block. Physics,
  # counters and every other output line remain byte-compared.
  grep -vE 'compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' "$1"
}

lumi()
{
  grep -m1 'lumi\[0\]\[1\]' "$1" | sed 's/.*= *//'
}

equivalent_lumi()
{
  awk -v a="$1" -v b="$2" 'BEGIN {
    d=a-b; if (d<0) d=-d; scale=(b<0?-b:b); if (scale<1) scale=1;
    exit !(d/scale <= 1e-12)
  }'
}

echo "=== serial default and persistent pool candidates ==="
run_case serial ''
SERIAL=$LAST_DIR
[ "$LAST_RC" -eq 0 ] && pass "serial default runs" || fail "serial default runs"
if grep -q 'compute_backend_requested' "$SERIAL/result.out"; then
  fail "default protocol remains free of new provenance lines"
else
  pass "default protocol remains free of new provenance lines"
fi

run_case pool_dual 'compute_backend="cpu";cpu_threads=4;cpu_variant="pool_dual";reduction_mode="strict";'
DUAL=$LAST_DIR
[ "$LAST_RC" -eq 0 ] && pass "pool_dual runs" || fail "pool_dual runs"
if cmp -s <(normalize "$SERIAL/result.out") <(normalize "$DUAL/result.out"); then
  pass "pool_dual is byte-identical after removing only backend provenance"
else
  fail "pool_dual is byte-identical after removing only backend provenance"
fi

run_case pool_private_2 'compute_backend="cpu";cpu_threads=2;cpu_variant="pool_private";reduction_mode="deterministic";'
PRIVATE2=$LAST_DIR
[ "$LAST_RC" -eq 0 ] && pass "pool_private with two workers runs" || fail "pool_private with two workers runs"
run_case pool_private_4 'compute_backend="cpu";cpu_threads=4;cpu_variant="pool_private";reduction_mode="deterministic";'
PRIVATE4=$LAST_DIR
[ "$LAST_RC" -eq 0 ] && pass "pool_private with four workers runs" || fail "pool_private with four workers runs"
if cmp -s <(normalize "$PRIVATE2/result.out") <(normalize "$PRIVATE4/result.out"); then
  pass "pool_private result is independent of physical worker count"
else
  fail "pool_private result is independent of physical worker count"
fi
if equivalent_lumi "$(lumi "$PRIVATE4/result.out")" "$(lumi "$SERIAL/result.out")"; then
  pass "pool_private luminosity agrees with serial to 1e-12 relative"
else
  fail "pool_private luminosity agrees with serial to 1e-12 relative"
fi

echo
echo "=== explicit refusal and optional OpenMP candidate ==="
run_case strict_private 'compute_backend="cpu";cpu_threads=4;cpu_variant="pool_private";reduction_mode="strict";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'strict reduction requires' "$LAST_DIR/stderr"; then
  pass "strict mode refuses a regrouped private reduction"
else
  fail "strict mode refuses a regrouped private reduction"
fi

run_case bad_variant 'compute_backend="cpu";cpu_threads=4;cpu_variant="not_a_backend";reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'cpu_variant must be' "$LAST_DIR/stderr"; then
  pass "unknown CPU variants fail nonzero with a diagnostic"
else
  fail "unknown CPU variants fail nonzero with a diagnostic"
fi

if grep -q 'openmp=enabled' "$SERIAL/stderr"; then
  run_case omp_ordered_rejected 'compute_backend="cpu";cpu_threads=4;cpu_variant="omp_atomic";reduction_mode="deterministic";'
  if [ "$LAST_RC" -ne 0 ] && grep -q 'requires reduction_mode=fast' "$LAST_DIR/stderr"; then
    pass "OpenMP atomics reject a deterministic reduction claim"
  else
    fail "OpenMP atomics reject a deterministic reduction claim"
  fi

  run_case omp_atomic 'compute_backend="cpu";cpu_threads=4;cpu_variant="omp_atomic";reduction_mode="fast";'
  if [ "$LAST_RC" -eq 0 ] && equivalent_lumi "$(lumi "$LAST_DIR/result.out")" "$(lumi "$SERIAL/result.out")"; then
    pass "OpenMP atomic candidate runs and matches serial luminosity"
  else
    fail "OpenMP atomic candidate runs and matches serial luminosity"
  fi
else
  run_case omp_unavailable 'compute_backend="cpu";cpu_threads=4;cpu_variant="omp_atomic";reduction_mode="fast";'
  if [ "$LAST_RC" -ne 0 ] && grep -q 'OpenMP was not enabled' "$LAST_DIR/stderr"; then
    pass "a requested unavailable OpenMP backend fails rather than falling back"
  else
    fail "a requested unavailable OpenMP backend fails rather than falling back"
  fi
fi

echo
echo "$((tests-failures))/$tests CPU backend checks passed"
if [ "${KEEP_WORK:-0}" = 1 ]; then echo "work retained at $WORK"; fi
exit "$failures"
