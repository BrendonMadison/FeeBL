#!/bin/bash
# End-to-end gate for the deck-level backend interface and fail-fast policy.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
DECK=$ROOT/examples/decks/quick.dat
[ -x "$BIN" ] || { echo "no executable FeeBL binary: $BIN"; exit 2; }

WORK=$(mktemp -d)
trap 'test "${KEEP_WORK:-0}" = 1 || rm -rf "$WORK"' EXIT
failures=0
tests=0

pass() { tests=$((tests+1)); printf "  %-68s PASS\n" "$1"; }
fail() { tests=$((tests+1)); failures=$((failures+1)); printf "  %-68s FAIL\n" "$1"; }

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
  grep -vE 'compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' "$1"
}

echo "=== deck interface and fail-fast semantics ==="
run_case serial ''
SERIAL=$LAST_DIR
[ "$LAST_RC" -eq 0 ] && pass "serial oracle runs" || fail "serial oracle runs"

if [ "$LAST_RC" -eq 0 ] &&
   grep -Eq 'FEEBL: final compute_backend requested=serial actual=serial .*gpu_calls=0' "$SERIAL/stderr"; then
  pass "final stderr contains live serial provenance"
else
  fail "final stderr contains live serial provenance"
fi
if ! grep -Eq 'engine requested=|engine_requested|modern_(reduction|batch|cpu|gpu|trace)' \
     "$SERIAL/stderr" "$SERIAL/result.out"; then
  pass "legacy defaults add no modern provenance output"
else
  fail "legacy defaults add no modern provenance output"
fi

LAST_DIR=$WORK/parser_perturbed
make_deck '' "$LAST_DIR"
(cd "$LAST_DIR" && MALLOC_PERTURB_=165 "$BIN" \
  c_nt1 p_c_nt1 result.out >stdout 2>stderr)
LAST_RC=$?
if [ "$LAST_RC" -eq 0 ] && cmp -s "$SERIAL/result.out" "$LAST_DIR/result.out"; then
  pass "parser variable names remain valid with nonzero allocator fill"
else
  fail "parser variable names remain valid with nonzero allocator fill"
fi

LAST_DIR=$WORK/legacy_env_ignored
make_deck '' "$LAST_DIR"
(cd "$LAST_DIR" && FEEBL_GPU_CIC=atomic "$BIN" \
  c_nt1 p_c_nt1 result.out >stdout 2>stderr)
LAST_RC=$?
if [ "$LAST_RC" -eq 0 ] &&
   ! grep -q 'CUDA CIC enabled' "$LAST_DIR/stderr" &&
   cmp -s "$SERIAL/result.out" "$LAST_DIR/result.out"; then
  pass "legacy environment override cannot bypass a serial deck"
else
  fail "legacy environment override cannot bypass a serial deck"
fi

run_case production_auto 'compute_backend="auto";'
if [ "$LAST_RC" -eq 0 ] &&
   grep -Eq 'FEEBL: final compute_backend requested=auto actual=serial .*gpu_calls=0' "$LAST_DIR/stderr" &&
   grep -q '10% speedup gate' "$LAST_DIR/stderr" &&
   cmp -s <(normalize "$SERIAL/result.out") <(normalize "$LAST_DIR/result.out"); then
  pass "production auto retains serial when no candidate passes its speed gate"
else
  fail "production auto retains serial when no candidate passes its speed gate"
fi

run_case counter 'rng_backend="counter";rndm_load=0;rndm_save=0;'
if [ "$LAST_RC" -ne 0 ] && grep -q 'not yet wired through every stochastic' "$LAST_DIR/stderr"; then
  pass "incomplete counter RNG integration is rejected explicitly"
else
  fail "incomplete counter RNG integration is rejected explicitly"
fi

run_case modern_bad_engine 'engine="warp_drive";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'engine must be legacy, modern_ref_cpu, modern_cpu, or modern_gpu' "$LAST_DIR/stderr"; then
  pass "modern engine parser rejects unknown engine names"
else
  fail "modern engine parser rejects unknown engine names"
fi

run_case modern_bad_algorithm 'algorithm="approximate";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'algorithm must be exact' "$LAST_DIR/stderr"; then
  pass "modern algorithm parser rejects unqualified approximations"
else
  fail "modern algorithm parser rejects unqualified approximations"
fi

run_case modern_control_on_legacy 'modern_batch_width=64;'
if [ "$LAST_RC" -ne 0 ] && grep -q 'engine=legacy accepts only' "$LAST_DIR/stderr"; then
  pass "modern tuning controls cannot be silently consumed by legacy"
else
  fail "modern tuning controls cannot be silently consumed by legacy"
fi

run_case modern_requires_counter 'engine="modern_cpu";modern_cpu_workers=4;'
if [ "$LAST_RC" -ne 0 ] && grep -q 'engine=modern_cpu requires rng_backend=counter' "$LAST_DIR/stderr"; then
  pass "every modern CPU request requires counter RNG"
else
  fail "every modern CPU request requires counter RNG"
fi

run_case modern_cpu_unwired 'engine="modern_cpu";algorithm="factored_exact";precision="mixed_qualified";modern_reduction="fast_qualified";modern_batch_width=128;modern_cpu_workers=4;modern_cpu_runtime="pool";modern_cpu_affinity="compact";modern_trace="summary";rng_backend="counter";'
if [ "$LAST_RC" -ne 0 ] &&
   grep -q 'engine=modern_cpu is parsed and provenance-ready' "$LAST_DIR/stderr" &&
   grep -q 'actual_engine=unavailable' "$LAST_DIR/stderr"; then
  pass "complete modern CPU card parses, then fails before simulation"
else
  fail "complete modern CPU card parses, then fails before simulation"
fi

run_case legacy_mode3 'gpu_mode=3;'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu_mode=3 requires engine=modern_gpu' "$LAST_DIR/stderr"; then
  pass "production GPU mode 3 cannot select a legacy backend"
else
  fail "production GPU mode 3 cannot select a legacy backend"
fi

run_case modern_gpu_false_policy 'engine="modern_gpu";gpu_mode=3;modern_gpu_streams=1;modern_gpu_graphs="off";rng_backend="legacy";'
if [ "$LAST_RC" -ne 0 ] &&
   grep -q 'requires algorithm=exact, precision=mixed_qualified' "$LAST_DIR/stderr"; then
  pass "mode 3 rejects a false FP64/deterministic policy claim"
else
  fail "mode 3 rejects a false FP64/deterministic policy claim"
fi

if ! grep -q 'cuda=enabled' "$SERIAL/stderr"; then
  run_case modern_gpu_unavailable 'engine="modern_gpu";gpu_mode=3;algorithm="exact";precision="mixed_qualified";modern_reduction="fast_qualified";modern_gpu_streams=1;modern_gpu_graphs="off";modern_trace="summary";gpu2_scope="primary";gpu2_deposition="atomic";gpu2_streams=1;gpu2_graphs="off";gpu2_fusion="off";gpu2_batch_slices=0;gpu2_pinned_io="on";gpu2_host_threads=1;gpu2_primary_mover="serial";gpu2_profile="summary";rng_backend="legacy";allow_backend_fallback=0;'
  if [ "$LAST_RC" -ne 0 ] &&
     grep -q 'no resident CUDA primary pipeline' "$LAST_DIR/stderr"; then
    pass "valid mode-3 selector fails closed in a non-CUDA build"
  else
    fail "valid mode-3 selector fails closed in a non-CUDA build"
  fi
else
  pass "valid CUDA mode-3 selector is covered by its dedicated production E2E"
fi

run_case gpu2_bad_mode 'compute_backend="gpu";gpu_mode=6;reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu_mode must be 1, 2, production-limited modern mode 3, or wave-batched legacy-physics mode 4, or adaptive-grid mode 5' "$LAST_DIR/stderr"; then
  pass "GPU mode parser rejects unsupported mode numbers"
else
  fail "GPU mode parser rejects unsupported mode numbers"
fi

run_case gpu4_bad_contract 'compute_backend="gpu";gpu_mode=4;gpu2_scope="primary";gpu2_deposition="atomic";gpu2_streams=1;gpu2_graphs="off";gpu2_fusion="off";gpu2_batch_slices=32;gpu2_pinned_io="on";gpu2_host_threads=1;gpu2_primary_mover="serial";gpu2_profile="summary";reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu_mode=4 requires .*gpu2_fusion=on' "$LAST_DIR/stderr"; then
  pass "GPU mode 4 rejects an incomplete batching contract"
else
  fail "GPU mode 4 rejects an incomplete batching contract"
fi

run_case gpu4_fallback_rejected 'compute_backend="gpu";gpu_mode=4;gpu2_scope="primary";gpu2_deposition="atomic";gpu2_fft="cufft";gpu2_streams=1;gpu2_graphs="off";gpu2_fusion="on";gpu2_batch_slices=32;gpu2_pinned_io="on";gpu2_host_threads=1;gpu2_primary_mover="serial";gpu2_profile="summary";reduction_mode="fast";allow_backend_fallback=1;'
if [ "$LAST_RC" -ne 0 ] &&
   grep -q 'gpu_mode=4 is abort-only; allow_backend_fallback must be 0' \
     "$LAST_DIR/stderr"; then
  pass "GPU mode 4 rejects fallback instead of changing algorithms"
else
  fail "GPU mode 4 rejects fallback instead of changing algorithms"
fi

if ! grep -q 'cuda=enabled' "$SERIAL/stderr"; then
  run_case gpu4_unavailable 'compute_backend="gpu";gpu_mode=4;gpu2_scope="primary";gpu2_deposition="atomic";gpu2_fft="cufft";gpu2_streams=1;gpu2_graphs="off";gpu2_fusion="on";gpu2_batch_slices=32;gpu2_pinned_io="on";gpu2_host_threads=1;gpu2_primary_mover="serial";gpu2_profile="summary";reduction_mode="fast";allow_backend_fallback=0;'
  if [ "$LAST_RC" -ne 0 ] &&
     grep -q 'no wave-batched CUDA/cuFFT pipeline' "$LAST_DIR/stderr"; then
    pass "valid GPU mode 4 fails closed in a non-CUDA build"
  else
    fail "valid GPU mode 4 fails closed in a non-CUDA build"
  fi
else
  pass "valid CUDA mode 4 is covered by its dedicated numerical E2E"
fi

run_case gpu2_bad_scope 'compute_backend="gpu";gpu_mode=2;gpu2_scope="typo";reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu2_scope must be' "$LAST_DIR/stderr"; then
  pass "GPU2 parser rejects an unknown resident scope"
else
  fail "GPU2 parser rejects an unknown resident scope"
fi

run_case gpu2_bad_primary_mover 'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";gpu2_primary_mover="threads";reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu2_primary_mover must be serial, openmp, pool, or auto' "$LAST_DIR/stderr"; then
  pass "GPU2 parser rejects an unknown primary host mover"
else
  fail "GPU2 parser rejects an unknown primary host mover"
fi

run_case gpu2_control_on_mode1 'compute_backend="gpu";gpu_mode=1;gpu2_streams=2;reduction_mode="fast";'
if [ "$LAST_RC" -ne 0 ] && grep -q 'gpu2_.* controls require' "$LAST_DIR/stderr"; then
  pass "GPU2-only controls cannot be silently consumed by mode 1"
else
  fail "GPU2-only controls cannot be silently consumed by mode 1"
fi

if grep -q 'cuda=enabled' "$SERIAL/stderr"; then
  run_case gpu_strict_rejected 'compute_backend="gpu";gpu_variant="atomic";gpu_device=0;reduction_mode="strict";'
  if [ "$LAST_RC" -ne 0 ] && grep -q 'CUDA CIC changes floating-point accumulation order' "$LAST_DIR/stderr"; then
    pass "strict reduction rejects order-changing CUDA accumulation"
  else
    fail "strict reduction rejects order-changing CUDA accumulation"
  fi

  run_case gpu_atomic_deterministic_rejected 'compute_backend="gpu";gpu_variant="atomic";gpu_device=0;reduction_mode="deterministic";'
  if [ "$LAST_RC" -ne 0 ] && grep -q 'requires gpu_variant=sorted' "$LAST_DIR/stderr"; then
    pass "deterministic CUDA reduction rejects atomic deposition"
  else
    fail "deterministic CUDA reduction rejects atomic deposition"
  fi

  run_case gpu_auto 'compute_backend="gpu";gpu_variant="auto";gpu_device=0;reduction_mode="fast";'
  GPU_AUTO=$LAST_DIR
  if [ "$LAST_RC" -eq 0 ] &&
     grep -q 'CUDA CIC enabled: variant=atomic' "$GPU_AUTO/stderr" &&
     grep -Eq 'calls=[1-9][0-9]*' "$GPU_AUTO/stderr" &&
     grep -Eq 'FEEBL: final compute_backend requested=gpu actual=gpu .*gpu_calls=[1-9][0-9]*' \
       "$GPU_AUTO/stderr" &&
     grep -q 'compute_backend_actual = gpu' "$GPU_AUTO/result.out"; then
    pass "explicit GPU auto executes the measured atomic winner"
  else
    fail "explicit GPU auto executes the measured atomic winner"
  fi

  run_case gpu_auto_deterministic 'compute_backend="gpu";gpu_variant="auto";gpu_device=0;reduction_mode="deterministic";'
  if [ "$LAST_RC" -eq 0 ] &&
     grep -q 'CUDA CIC enabled: variant=sorted' "$LAST_DIR/stderr" &&
     grep -Eq 'FEEBL: final compute_backend requested=gpu actual=gpu .*gpu_variant requested=auto actual=sorted.*gpu_calls=[1-9][0-9]*' \
       "$LAST_DIR/stderr" &&
     grep -q 'compute_backend_actual = gpu' "$LAST_DIR/result.out" &&
     grep -q 'gpu_variant_requested = auto gpu_variant_actual = sorted' \
       "$LAST_DIR/result.out"; then
    pass "GPU auto honors deterministic reduction by selecting sorted"
  else
    fail "GPU auto honors deterministic reduction by selecting sorted"
  fi

  run_case gpu_guard 'compute_backend="gpu";gpu_variant="auto";gpu_device=0;gpu_min_particles=262144;reduction_mode="fast";allow_backend_fallback=1;'
  if [ "$LAST_RC" -eq 0 ] &&
     grep -q 'CUDA CIC auto locked serial' "$LAST_DIR/stderr" &&
     grep -Eq 'FEEBL: final compute_backend requested=gpu actual=serial .*gpu_calls=0 .*GPU auto threshold selected serial' \
       "$LAST_DIR/stderr" &&
     grep -q 'compute_backend_actual = serial' "$LAST_DIR/result.out" &&
     grep -q 'GPU auto threshold selected serial' "$LAST_DIR/result.out" &&
     cmp -s <(normalize "$SERIAL/result.out") <(normalize "$LAST_DIR/result.out"); then
    pass "authorized GPU threshold fallback records actual serial execution"
  else
    fail "authorized GPU threshold fallback records actual serial execution"
  fi

  run_case gpu_atomic 'compute_backend="gpu";gpu_variant="atomic";gpu_device=0;reduction_mode="fast";'
  if [ "$LAST_RC" -eq 0 ] &&
     grep -q 'CUDA CIC enabled: variant=atomic' "$LAST_DIR/stderr" &&
     grep -Eq 'calls=[1-9][0-9]*' "$LAST_DIR/stderr" &&
     python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" \
       <(normalize "$SERIAL/result.out") <(normalize "$LAST_DIR/result.out") \
       --rtol 5e-6 >/dev/null; then
    pass "deck-selected CUDA atomic path satisfies the preregistered bound"
  else
    fail "deck-selected CUDA atomic path satisfies the preregistered bound"
  fi
else
  run_case gpu_unavailable 'compute_backend="gpu";gpu_variant="atomic";'
  if [ "$LAST_RC" -ne 0 ] && grep -q 'rebuild with -DFEEBL_ENABLE_CUDA=ON' "$LAST_DIR/stderr"; then
    pass "CPU-only build refuses an unavailable explicit GPU"
  else
    fail "CPU-only build refuses an unavailable explicit GPU"
  fi

  run_case gpu_fallback 'compute_backend="gpu";gpu_variant="atomic";allow_backend_fallback=1;cpu_threads=2;'
  if [ "$LAST_RC" -eq 0 ] && grep -q 'actual=cpu' "$LAST_DIR/stderr" &&
     grep -q 'CUDA backend unavailable' "$LAST_DIR/stderr"; then
    pass "requested fallback is explicit and records the actual CPU backend"
  else
    fail "requested fallback is explicit and records the actual CPU backend"
  fi
fi

echo
echo "$((tests-failures))/$tests backend-interface checks passed"
if [ "${KEEP_WORK:-0}" = 1 ]; then echo "work retained at $WORK"; fi
exit "$failures"
