#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/feebl-modern-photon-soa.XXXXXX")
trap 'rm -rf -- "${build_dir}"' EXIT

cxx=${CXX:-g++}
optimization=(-O2 -g)
runtime=()
if [[ "${FEEBL_PHOTON_SOA_SANITIZE:-0}" == "1" ]]; then
  optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  runtime=(env ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1)
fi

"${cxx}" -std=c++17 "${optimization[@]}" -fno-fast-math \
  -Wall -Wextra -Wpedantic -Werror \
  -I"${repo_root}/src" \
  "${script_dir}/test_modern_photon_soa.cc" \
  "${repo_root}/src/modernPhotonSoACPP.cc" \
  -o "${build_dir}/test_modern_photon_soa"

"${runtime[@]}" "${build_dir}/test_modern_photon_soa"
