#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/feebl-modern-collision.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

compiler="${CXX:-g++}"
optimization=(-O2)
runtime=()
if [[ "${FEEBL_COLLISION_SANITIZE:-0}" == "1" ]]; then
  optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  runtime=(env ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1)
fi
"$compiler" \
  -std=c++17 "${optimization[@]}" \
  -Wall -Wextra -Wpedantic -Werror -fno-fast-math \
  -I"$repo_root/src" \
  "$repo_root/src/beamSoACPP.cc" \
  "$repo_root/src/cellCSRCPP.cc" \
  "$repo_root/src/modernCollisionCPP.cc" \
  "$repo_root/examples/modern_collision/test_modern_collision.cc" \
  -o "$build_dir/test_modern_collision"

"${runtime[@]}" "$build_dir/test_modern_collision"
