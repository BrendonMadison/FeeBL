#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

cxx="${CXX:-c++}"
common=(
  -std=c++17 -Wall -Wextra -Wpedantic -Werror
  -fno-fast-math -ffp-contract=off
  -I"$root_dir/src"
  "$root_dir/examples/electron_lumi_bernoulli_skip/test_electron_lumi_bernoulli_skip.cc"
  "$root_dir/src/electronLumiBernoulliSkipCPP.cc"
)

"$cxx" -O3 "${common[@]}" -o "$work_dir/bernoulli-skip-optimized"
"$work_dir/bernoulli-skip-optimized"

"$cxx" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
  "${common[@]}" -o "$work_dir/bernoulli-skip-sanitized"
ASAN_OPTIONS=detect_leaks=1 "$work_dir/bernoulli-skip-sanitized"
