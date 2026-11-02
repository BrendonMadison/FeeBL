#!/bin/bash
# Beam frames and rigid-bunch overlap checks. With the beam-beam field
# switched off, FeeBL's luminosity must reproduce the analytic
# Gaussian value. Rigid beams come from charge_sign=0, cross-checked
# independently against the low-intensity limit (agreement 0.04%).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

echo "=== geometry and the analytic reference ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" "$HERE/test_geometry.cc" -o "$OUT/tg"
"$OUT/tg"

echo
echo "=== rigid-beam mechanism check ==="
bash "$HERE/rigid_check.sh"

echo
echo "=== Level 3: rigid FEEBL against the analytic value ==="
bash "$HERE/level3.sh"

echo
echo "=== hourglass verified against a closed form and a finer rule ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" "$HERE/test_hourglass.cc" -o "$OUT/hg"
"$OUT/hg"

echo
echo "=== rotation covariance ==="
bash "$HERE/rotation.sh"

echo
echo "=== head-on -> lab: the transformation OUT of the frame ==="
g++ -std=c++17 -O2 -Wall -I"$ROOT/src" "$HERE/test_headon.cc" -o "$OUT/th"
"$OUT/th"

echo
echo "=== what a finite-angle run may be quoted for ==="
bash "$HERE/finite_angle_guards.sh"
