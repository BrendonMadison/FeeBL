#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/feebl-modern-cpu-executor.XXXXXX")
trap 'rm -rf -- "${build_dir}"' EXIT

cxx=${CXX:-g++}
optimization=(-O2 -g)
runtime=()
if [[ "${FEEBL_EXECUTOR_SANITIZE:-0}" == "1" &&
      "${FEEBL_EXECUTOR_TSAN:-0}" == "1" ]]; then
  echo "ASan/UBSan and TSan executor modes are mutually exclusive" >&2
  exit 2
elif [[ "${FEEBL_EXECUTOR_SANITIZE:-0}" == "1" ]]; then
  optimization=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
  runtime=(env ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1)
elif [[ "${FEEBL_EXECUTOR_TSAN:-0}" == "1" ]]; then
  optimization=(-O1 -g -fsanitize=thread -fno-omit-frame-pointer)
  runtime=(env TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1)
fi
common=(
  -std=c++17 "${optimization[@]}"
  -Wall -Wextra -Wpedantic -Werror -pthread
  -I"${repo_root}/src"
  "${repo_root}/src/encounterDAGCPP.cc"
  "${repo_root}/src/modernCpuExecutorCPP.cc"
  "${script_dir}/test_modern_cpu_executor.cc"
)

"${cxx}" "${common[@]}" -o "${build_dir}/test-serial"
"${runtime[@]}" "${build_dir}/test-serial" > "${build_dir}/serial.out"

if "${cxx}" -fopenmp "${common[@]}" -o "${build_dir}/test-openmp"; then
  OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=10 OMP_NUM_THREADS=10 \
    "${runtime[@]}" "${build_dir}/test-openmp" > "${build_dir}/openmp.out"
  diff -u "${build_dir}/serial.out" "${build_dir}/openmp.out"
elif [[ "${FEEBL_EXECUTOR_ALLOW_NO_OPENMP:-0}" != "1" ]]; then
  echo "OpenMP executor test build failed; set FEEBL_EXECUTOR_ALLOW_NO_OPENMP=1 only on a documented CPU-only toolchain" >&2
  exit 1
fi

cat "${build_dir}/serial.out"
