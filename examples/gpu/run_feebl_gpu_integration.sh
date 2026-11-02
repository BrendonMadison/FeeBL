#!/usr/bin/env bash
# Whole-program serial/CUDA differential tests through the production deck
# interface. These are intentionally not default CTest cases: generated and
# loaded-bunch coverage takes roughly two minutes on the reference host.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${1:-"$ROOT/build-cuda/src/feebl"}
RESULTS=${2:-"$ROOT/gpu-integration-results"}

test -x "$BIN" || { echo "no FeeBL binary: $BIN" >&2; exit 2; }
test ! -e "$RESULTS" || { echo "refusing to reuse results: $RESULTS" >&2; exit 2; }
mkdir -p "$RESULTS"

make_deck()
{
  local source=$1 fragment=$2 output=$3
  awk -v config="$fragment" '
    { sub(/^hist_ee_bins=1010;/, config "hist_ee_bins=1010;"); print }
  ' "$source" > "$output"
}

run_generated()
{
  local label=$1 fragment=$2
  local run="$RESULTS/generated-$label"
  mkdir -p "$run"
  make_deck "$ROOT/examples/decks/quick.dat" "$fragment" "$run/acc.dat"
  (cd "$run" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr)
}

normalize()
{
  grep -vE 'compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' "$1"
}

run_generated serial ''
run_generated atomic 'compute_backend="gpu";gpu_variant="atomic";reduction_mode="fast";'
run_generated shared 'compute_backend="gpu";gpu_variant="shared";reduction_mode="fast";'
run_generated sorted 'compute_backend="gpu";gpu_variant="sorted";reduction_mode="deterministic";'
run_generated sorted-repeat 'compute_backend="gpu";gpu_variant="sorted";reduction_mode="deterministic";'

for variant in atomic shared sorted; do
  grep -q "CUDA CIC enabled: variant=$variant" "$RESULTS/generated-$variant/stderr"
  grep -Eq 'calls=[1-9][0-9]*' "$RESULTS/generated-$variant/stderr"
  grep -q 'compute_backend_actual = gpu' "$RESULTS/generated-$variant/result.out"
done

# The local deterministic contract is intentionally scoped to the same exact
# binary, CUDA toolkit, device and input. It does not promise cross-device
# Thrust reduction identity.
cmp "$RESULTS/generated-sorted/result.out" \
    "$RESULTS/generated-sorted-repeat/result.out"

# The three GPU algorithms should produce the same physics result for this
# fixed workload. Provenance differs by design and is removed only here.
cmp <(normalize "$RESULTS/generated-atomic/result.out") \
    <(normalize "$RESULTS/generated-shared/result.out")
cmp <(normalize "$RESULTS/generated-atomic/result.out") \
    <(normalize "$RESULTS/generated-sorted/result.out")
python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" --rtol 5e-6 \
  <(normalize "$RESULTS/generated-serial/result.out") \
  <(normalize "$RESULTS/generated-atomic/result.out")

# Threshold fallback is permitted only when the deck explicitly authorizes it.
run_generated auto-guard 'compute_backend="gpu";gpu_variant="auto";gpu_min_particles=262144;reduction_mode="fast";allow_backend_fallback=1;'
cmp <(normalize "$RESULTS/generated-serial/result.out") \
    <(normalize "$RESULTS/generated-auto-guard/result.out")
grep -q 'auto locked serial' "$RESULTS/generated-auto-guard/stderr"
grep -q 'compute_backend_actual = serial' \
  "$RESULTS/generated-auto-guard/result.out"
grep -q 'gpu_calls=0' "$RESULTS/generated-auto-guard/result.out"

# Exercise the loaded/arbitrary-bunch packing path. First create canonical
# legacy beam files, then feed the same files to serial and explicit CUDA runs.
GENERATOR="$RESULTS/loaded-generator"
mkdir -p "$GENERATOR"
make_deck "$ROOT/examples/decks/quick.dat" 'store_beam=1;' "$GENERATOR/acc.dat"
(cd "$GENERATOR" && "$BIN" c_nt1 p_c_nt1 result.out >stdout 2>stderr)
test -s "$GENERATOR/beam1.dat" && test -s "$GENERATOR/beam2.dat"

run_loaded()
{
  local label=$1 fragment=$2
  local run="$RESULTS/loaded-$label"
  mkdir -p "$run"
  make_deck "$ROOT/examples/decks/quick.dat" \
    "load_beam=3;$fragment" "$run/acc.dat"
  (cd "$run" && "$BIN" \
    -e "$GENERATOR/beam1.dat" -p "$GENERATOR/beam2.dat" \
    c_nt1 p_c_nt1 result.out >stdout 2>stderr)
}

run_loaded serial ''
run_loaded atomic 'compute_backend="gpu";gpu_variant="atomic";reduction_mode="fast";'
grep -q 'CUDA CIC enabled: variant=atomic' "$RESULTS/loaded-atomic/stderr"
grep -Eq 'calls=[1-9][0-9]*' "$RESULTS/loaded-atomic/stderr"
grep -Eq 'packed_calls=[1-9][0-9]*' "$RESULTS/loaded-atomic/stderr"
python3 "$ROOT/examples/gpu/compare_feebl_outputs.py" --rtol 5e-6 \
  <(normalize "$RESULTS/loaded-serial/result.out") \
  <(normalize "$RESULTS/loaded-atomic/result.out")

echo "PASS: generated and loaded-bunch whole-FeeBL CUDA differential checks"
