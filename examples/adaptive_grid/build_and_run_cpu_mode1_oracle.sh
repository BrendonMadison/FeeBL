#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 BUILD_DIRECTORY [OUTPUT_DIRECTORY]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../.." && pwd -P)
build_dir=$(cd -- "$1" && pwd -P)
output_dir=${2:-"$build_dir/adaptive-grid-validation"}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)

[[ -f "$build_dir/src/config.h" ]] || {
  echo "missing configured header: $build_dir/src/config.h" >&2
  exit 2
}

cxx=${CXX:-c++}
common=(
  -std=c++17 -O2 -Wno-register -fno-fast-math -ffp-contract=off
  -ffunction-sections -fdata-sections -DCOMPUTER64b -DUSE_FFTW3
  -I "$build_dir/src" -I "$source_root/src"
)

compile() {
  local source=$1
  local object=$2
  shift 2
  "$cxx" "${common[@]}" "$@" -c "$source" -o "$object"
}

# Compile the test with warning-clean requirements. Production translation
# units keep the repository's historical warning policy, including register.
compile "$source_root/examples/adaptive_grid/test_mode1_field_equivalence.cc" \
  "$output_dir/test_mode1_field_equivalence.o" \
  -Wall -Wextra -Wshadow -Werror
compile "$source_root/src/uniformWindowPlanCPP.cc" \
  "$output_dir/uniformWindowPlanCPP.o"
compile "$source_root/src/adaptiveResolutionControllerCPP.cc" \
  "$output_dir/adaptiveResolutionControllerCPP.o"
compile "$source_root/src/fieldCPP.cc" "$output_dir/fieldCPP.o"
compile "$source_root/src/fourierCPP.cc" "$output_dir/fourierCPP.o"
compile "$source_root/src/gridCPP.cc" "$output_dir/gridCPP.o"
# GRID's production CIC dispatch references its CPU-only GPU adapter and
# backend state even when this oracle selects the serial deposition path.
compile "$source_root/src/gpuCicStub.cc" "$output_dir/gpuCicStub.o"
compile "$source_root/src/parallelBackendCPP.cc" "$output_dir/parallelBackendCPP.o"
compile "$source_root/src/tridentCPP.cc" "$output_dir/tridentCPP.o"
compile "$source_root/src/wavefrontFieldWorkspaceCPP.cc" \
  "$output_dir/wavefrontFieldWorkspaceCPP.o"
compile "$source_root/src/wavefrontFieldValidationCPP.cc" \
  "$output_dir/wavefrontFieldValidationCPP.o"

"$cxx" -Wl,--gc-sections \
  "$output_dir/test_mode1_field_equivalence.o" \
  "$output_dir/uniformWindowPlanCPP.o" \
  "$output_dir/adaptiveResolutionControllerCPP.o" \
  "$output_dir/fieldCPP.o" \
  "$output_dir/fourierCPP.o" \
  "$output_dir/gridCPP.o" \
  "$output_dir/gpuCicStub.o" \
  "$output_dir/parallelBackendCPP.o" \
  "$output_dir/tridentCPP.o" \
  "$output_dir/wavefrontFieldWorkspaceCPP.o" \
  "$output_dir/wavefrontFieldValidationCPP.o" \
  -lfftw3 -lm -pthread -o "$output_dir/test_mode1_field_equivalence"

"$output_dir/test_mode1_field_equivalence"
