#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
work="${FEEBL_ELECTRON_LUMI_SAMPLER_TEST_WORK:-${TMPDIR:-/tmp}/feebl-electron-lumi-sampler-test}"
mkdir -p "$work"
read -r -a cxx_command <<<"${CXX:-c++}"
common=(
  -std=c++17 -fno-fast-math -ffp-contract=off
  -Wall -Wextra -Werror -I"$root/src"
  "$root/examples/electron_lumi_sampler/test_electron_lumi_sampler.cc"
  "$root/src/electronLumiSamplerCPP.cc"
)

"${cxx_command[@]}" -O3 -DNDEBUG "${common[@]}" \
  -o "$work/test_electron_lumi_sampler_optimized"
"$work/test_electron_lumi_sampler_optimized"

"${cxx_command[@]}" -O1 -g -D_GLIBCXX_ASSERTIONS \
  -fsanitize=address,undefined,float-cast-overflow -fno-omit-frame-pointer \
  "${common[@]}" -o "$work/test_electron_lumi_sampler_sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  "$work/test_electron_lumi_sampler_sanitized"
