#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${TMPDIR:-/tmp}/feebl-electron-lumi-sampling2-test"
mkdir -p "${build_dir}"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I"${repo_root}/src" \
  "${repo_root}/src/electronLumiSamplerCPP.cc" \
  "${repo_root}/examples/electron_lumi_sampling2/test_fast_bernoulli.cc" \
  -o "${build_dir}/test_fast_bernoulli"

"${build_dir}/test_fast_bernoulli"
