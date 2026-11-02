#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/feebl-legacy-ee-precompute.XXXXXX")
trap 'rm -rf -- "${build_dir}"' EXIT

read -r -a cxx_command <<<"${CXX:-g++}"
common=(
  -std=c++17 -fno-fast-math -ffp-contract=off
  -Wall -Wextra -Wpedantic -Werror -pthread
  -I"${repo_root}/src"
  "${repo_root}/src/legacyElectronLumiPrecomputeCPP.cc"
  "${script_dir}/test_legacy_electron_lumi_precompute.cc"
)

"${cxx_command[@]}" -O2 -g -DNDEBUG "${common[@]}" \
  -o "${build_dir}/test-optimized"
"${build_dir}/test-optimized"

"${cxx_command[@]}" -O1 -g -D_GLIBCXX_ASSERTIONS \
  -fsanitize=address,undefined,float-cast-overflow -fno-omit-frame-pointer \
  "${common[@]}" -o "${build_dir}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  "${build_dir}/test-sanitized"
