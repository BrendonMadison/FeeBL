#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/feebl-sparse-cells.XXXXXX")
trap 'rm -rf -- "${build_dir}"' EXIT

read -r -a cxx_command <<<"${CXX:-g++}"
printf '%s\n' '#define HAVE_STPCPY 1' > "${build_dir}/config.h"
common=(
  -std=c++17 -fno-fast-math -ffp-contract=off
  -Wall -Wextra -Wpedantic -Werror -Wno-register -Wno-unused-parameter -pthread
  -I"${build_dir}" -I"${repo_root}/src"
  "${script_dir}/test_sparse_cells.cc"
)

"${cxx_command[@]}" -O2 -g "${common[@]}" \
  -o "${build_dir}/test-optimized"
"${build_dir}/test-optimized"

"${cxx_command[@]}" -O1 -g -D_GLIBCXX_ASSERTIONS \
  -fsanitize=address,undefined,float-cast-overflow -fno-omit-frame-pointer \
  "${common[@]}" -o "${build_dir}/test-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  "${build_dir}/test-sanitized"
