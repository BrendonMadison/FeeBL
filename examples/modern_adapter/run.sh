#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

extra_flags=()
if [[ "${FEEBL_ADAPTER_SANITIZE:-0}" == "1" ]]; then
  extra_flags+=(
    -O1 -g
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
  )
else
  extra_flags+=(-O2)
fi

"${CXX:-g++}" -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -Wno-error=unused-result -DCOMPUTER64b -I"$root/src" \
  "${extra_flags[@]}" \
  "$here/test_modern_legacy_adapter.cc" \
  "$root/src/modernLegacyAdapterCPP.cc" \
  "$root/src/beamSoACPP.cc" \
  "$root/src/particleBeamCPP.cc" \
  "$root/src/particlesCPP.cc" \
  "$root/src/tridentCPP.cc" \
  "$root/src/physicalTools.cc" \
  "$root/src/rndmCPP.cc" \
  "$root/src/besselCPP.cc" \
  "$root/src/splineCPP.cc" \
  "$root/src/mathematicalTools.cc" \
  -o "$work/test_modern_legacy_adapter"

if [[ "${FEEBL_ADAPTER_SANITIZE:-0}" == "1" ]]; then
  ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
    "$work/test_modern_legacy_adapter"
else
  "$work/test_modern_legacy_adapter"
fi
