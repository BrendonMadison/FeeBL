#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${TMPDIR:-/tmp}/feebl-legacy-photon-lumi-fast-${USER:-unknown}"
mkdir -p "$build"

common=(
  -std=c++11 -fno-fast-math -Wall -Wextra -Wpedantic -Werror
  -I"$root/src"
  "$root/examples/legacy_photon_lumi_fast/test_legacy_photon_lumi_fast.cc"
  "$root/src/legacyPhotonLumiFastCPP.cc"
)

"${CXX:-c++}" -O3 "${common[@]}" -o "$build/test_legacy_photon_lumi_fast"
"$build/test_legacy_photon_lumi_fast"

# Compile the shared threshold definition under the same GNU fast-math
# contract as GRID::collide_gg.  The main oracle above is deliberately built
# without fast-math, so together the two probes catch cross-TU predicate drift.
"${CXX:-c++}" -std=c++11 -O3 -ffast-math \
  -Wall -Wextra -Wpedantic -Werror -I"$root/src" \
  "$root/examples/legacy_photon_lumi_fast/test_production_threshold.cc" \
  -o "$build/test_production_threshold"
"$build/test_production_threshold"

if [[ "${FEEBL_LEGACY_PHOTON_LUMI_SANITIZE:-0}" == "1" ]]; then
  "${CXX:-c++}" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${common[@]}" \
    -o "$build/test_legacy_photon_lumi_fast_sanitize"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "$build/test_legacy_photon_lumi_fast_sanitize"
fi
