#!/bin/bash
# Test the standalone maps, coordinate conversions and incoming map chain.
# Crab crossing and crab waist describe incoming optics and leave the
# beam-beam force law unchanged.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC="$HERE"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

echo "  compiler: $(g++ --version | head -1)"
echo
echo "=== upstream reference module (unmodified) ==="
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I"$SRC" -I"$ROOT/src" \
    "$SRC/reference_maps.cpp" "$SRC/test_reference_maps.cpp" -o "$OUT/test_maps"
"$OUT/test_maps"

echo
echo "=== C2: what the angle-vs-canonical-momentum conversion is worth ==="
g++ -std=c++17 -O2 -Wall -I"$SRC" -I"$ROOT/src" \
    "$HERE/c2_conversion.cc" "$SRC/reference_maps.cpp" -o "$OUT/c2"
"$OUT/c2"

echo
echo "=== Stage D: the chain wired to the canonical tier ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" \
    "$HERE/test_chain.cc" "$ROOT/src/incomingMapsCPP.cc" "$ROOT/src/gpgeomMaps.cc" \
    -o "$OUT/test_chain"
"$OUT/test_chain"

echo
echo "=== crab waist: the g(x) that symplecticity cannot check ==="
g++ -std=c++17 -O2 -Wall -Wextra -I"$ROOT/src" "$HERE/test_gvalue.cc" "$ROOT/src/gpgeomMaps.cc" -o "$OUT/test_gvalue"
"$OUT/test_gvalue"

echo
echo "=== IP dispersion: symplectic, invertible, and it narrows dL/dW ==="
g++ -std=c++17 -O2 -Wall -Wextra -I"$ROOT/src" \
    "$HERE/test_dispersion.cc" "$ROOT/src/gpgeomMaps.cc" \
    -o "$OUT/test_disp"
"$OUT/test_disp"

echo
echo "=== C5: the two-beam longitudinal sign, explicit vs derived ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" \
    "$HERE/test_two_beam_sign.cc" "$ROOT/src/incomingMapsCPP.cc" "$ROOT/src/gpgeomMaps.cc" \
    -o "$OUT/test_sign"
"$OUT/test_sign"

echo
echo "=== does crab crossing actually restore luminosity? ==="
bash "$HERE/crab_luminosity.sh"

echo
echo "=== does the crossing-angle shear sit on the true slice centre? ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" "$HERE/test_slicing.cc" -o "$OUT/tsl"
"$OUT/tsl"

echo
echo "=== a map must declare what it physically is ==="
bash "$HERE/map_spin_policy.sh"

echo
echo "=== the crab n_z guard ==="
bash "$HERE/crab_nz_guard.sh"

echo
echo "=== the map chain on the loaded-beam path ==="
bash "$HERE/crab_loaded.sh"

echo
echo "=== configuration validation ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" \
    "$HERE/test_config_validation.cc" "$ROOT/src/incomingMapsCPP.cc" "$ROOT/src/gpgeomMaps.cc" \
    -o "$OUT/test_cfg"
"$OUT/test_cfg"
