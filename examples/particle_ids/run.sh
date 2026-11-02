#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wno-error=unused-result -DCOMPUTER64b -I"$root/src" \
  "$here/test_primary_particle_ids.cc" \
  "$root/src/particleBeamCPP.cc" \
  "$root/src/particlesCPP.cc" \
  "$root/src/tridentCPP.cc" \
  "$root/src/physicalTools.cc" \
  "$root/src/rndmCPP.cc" \
  "$root/src/besselCPP.cc" \
  "$root/src/splineCPP.cc" \
  "$root/src/mathematicalTools.cc" \
  -o "$work/test_primary_particle_ids"

"$work/test_primary_particle_ids" "$here/loaded_primary_ids.dat"
