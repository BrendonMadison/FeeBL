#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${TMPDIR:-/tmp}/feebl-modern-photon-lumi-${USER:-unknown}"
mkdir -p "$build"

"${CXX:-c++}" -std=c++17 -O2 -fno-fast-math -Wall -Wextra -Wpedantic \
  -Werror -I"$root/src" \
  "$root/examples/modern_photon_lumi/test_modern_photon_lumi.cc" \
  "$root/src/modernPhotonLumiCPP.cc" \
  "$root/src/beamSoACPP.cc" "$root/src/cellCSRCPP.cc" \
  -o "$build/test_modern_photon_lumi"

"$build/test_modern_photon_lumi"
