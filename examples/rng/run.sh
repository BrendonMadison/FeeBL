#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
work=${FEEBL_RNG_TEST_WORK:-"${TMPDIR:-/tmp}/feebl-counter-rng-test"}
mkdir -p "$work"

${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$root/src" "$root/examples/rng/test_counter_rng.cc" \
  -o "$work/test_counter_rng"
"$work/test_counter_rng"

${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  -fno-fast-math -I"$root/src" "$root/examples/rng/test_counter_rng_v2.cc" \
  -o "$work/test_counter_rng_v2"
"$work/test_counter_rng_v2"

if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  nvcc -std=c++17 -O2 -I"$root/src" \
    "$root/examples/rng/test_counter_rng_cuda.cu" \
    -o "$work/test_counter_rng_cuda"
  "$work/test_counter_rng_cuda"
else
  echo "counter RNG CUDA parity: SKIP (CUDA compiler or device unavailable)"
fi
