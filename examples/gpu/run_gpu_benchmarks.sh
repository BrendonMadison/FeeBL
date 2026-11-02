#!/usr/bin/env bash
# Rebuild and regenerate the CUDA CIC correctness and raw timing CSVs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=${FEEBL_GPU_BUILD_DIR:-"$ROOT/build-gpu-repro"}
RESULTS=${FEEBL_GPU_RESULTS_DIR:-"$ROOT/gpu-benchmark-results"}
ARCH=${FEEBL_CUDA_ARCHITECTURES:-native}
ALLOW_HOST=${FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER:-OFF}
DEVICE=${FEEBL_GPU_DEVICE:-0}
REPETITIONS=${FEEBL_GPU_REPETITIONS:-7}
MAX_PARTICLES=${FEEBL_GPU_MAX_PARTICLES:-4194304}

source_revision()
{
  local source_root git_root revision
  source_root=$(cd "$ROOT" 2>/dev/null && pwd -P) || source_root=
  git_root=$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null) || git_root=
  if [ -n "$git_root" ]; then
    git_root=$(cd "$git_root" 2>/dev/null && pwd -P) || git_root=
  fi
  if [ -n "$source_root" ] && [ "$git_root" = "$source_root" ]; then
    git -C "$ROOT" rev-parse --verify HEAD
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

mkdir -p "$RESULTS"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFEEBL_ENABLE_CUDA=ON \
  -DFEEBL_CUDA_ARCHITECTURES="$ARCH" \
  -DFEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER="$ALLOW_HOST"
cmake --build "$BUILD" -j
ctest --test-dir "$BUILD" -R feebl-gpu-cic-correctness --output-on-failure

"$BUILD/src/feebl_gpu_cic_test" \
  --device "$DEVICE" \
  --correctness \
  --correctness-csv "$RESULTS/gpu_cic_correctness.csv" \
  --benchmark "$RESULTS/gpu_cic_benchmark.csv" \
  --repetitions "$REPETITIONS" \
  --max-particles "$MAX_PARTICLES"

{
  date --iso-8601=seconds
  uname -a
  source_revision
  cmake --version | head -1
  "${CXX:-c++}" --version | head -1
  nvcc --version
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv,noheader
} >"$RESULTS/environment.txt"

echo "results: $RESULTS"
