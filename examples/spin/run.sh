#!/bin/bash
# Spin suite.
#
# Run analytic rotation checks and collision-level spin-norm diagnostics.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

# The analytic gate links against the production objects. Build them if a
# previous `make clean` removed them, so the suite is self-sufficient.
if [ -n "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  OBJ="CMAKE_PRODUCTION_OBJECTS"
elif ! ls "$ROOT"/src/*.o >/dev/null 2>&1; then
  echo "  (building objects for the analytic gate)"
  ( cd "$ROOT/src" && make feebl >/dev/null 2>&1 ) || true
fi

echo "=== analytic: BMT precession is an exact rotation at the right angle ==="
if [ -z "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  CXX_CMD=${CXX:-g++}
  OBJ=$(ls "$ROOT"/src/*.o 2>/dev/null | grep -v 'principal\.o' | tr '\n' ' ')
  FFTW_INC=$(sed -n 's/^FFTW_CFLAGS *= *//p' "$ROOT/src/Makefile" | head -1)
  FFTW_LIB=$(sed -n 's/^FFTW_LDFLAGS *= *//p' "$ROOT/src/Makefile" | head -1)
fi
if [ -z "$OBJ" ]; then
  echo "  cannot build: no objects in $ROOT/src and the build did not produce any"
  echo "  this check did NOT run -- reporting failure rather than success."
  exit 1
fi
# shellcheck disable=SC2086
if [ -n "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  python3 "$HERE/compile_spin_probe.py" --build "$FEEBL_TEST_BUILD_DIR" --root "$ROOT" --source "$HERE/test_bmt.cc" --output "$OUT/test_bmt"
else
"$CXX_CMD" -std=c++17 -O2 -Wall -DUSE_NON_CONST $FFTW_INC -I"$ROOT/src" \
    "$HERE/test_bmt.cc" $OBJ $FFTW_LIB -o "$OUT/test_bmt"
fi
"$OUT/test_bmt"

echo
echo "=== BMT direction geometry and radiative acceleration basis ==="
if [ -n "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  python3 "$HERE/compile_spin_probe.py" --build "$FEEBL_TEST_BUILD_DIR" --root "$ROOT" --source "$HERE/test_spin_physics.cc" --output "$OUT/test_spin_physics"
else
  "$CXX_CMD" -std=c++17 -O2 -Wall -DUSE_NON_CONST $FFTW_INC -I"$ROOT/src" "$HERE/test_spin_physics.cc" $OBJ $FFTW_LIB -o "$OUT/test_spin_physics"
fi
"$OUT/test_spin_physics"

echo
echo "=== the transverse roll (angle_phi) carries the spin ==="
# shellcheck disable=SC2086
if [ -n "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  python3 "$HERE/compile_spin_probe.py" --build "$FEEBL_TEST_BUILD_DIR" --root "$ROOT" --source "$HERE/test_roll.cc" --output "$OUT/test_roll"
else
"$CXX_CMD" -std=c++17 -O2 -Wall -DUSE_NON_CONST $FFTW_INC -I"$ROOT/src"     "$HERE/test_roll.cc" $OBJ $FFTW_LIB -o "$OUT/test_roll"
fi
"$OUT/test_roll"

echo
echo "=== the precession must follow the relative beam charge ==="
# shellcheck disable=SC2086
if [ -n "${FEEBL_TEST_BUILD_DIR:-}" ]; then
  python3 "$HERE/compile_spin_probe.py" --build "$FEEBL_TEST_BUILD_DIR" --root "$ROOT" --source "$HERE/test_chargesign.cc" --output "$OUT/test_cs"
else
"$CXX_CMD" -std=c++17 -O2 -Wall -DUSE_NON_CONST $FFTW_INC -I"$ROOT/src" \
    "$HERE/test_chargesign.cc" $OBJ $FFTW_LIB -o "$OUT/test_cs"
fi
"$OUT/test_cs"

echo
echo "=== spin norm through a real collision ==="
bash "$HERE/norm_conservation.sh"

echo
echo "=== longitudinal bunch polarization ==="
bash "$HERE/polarization.sh"

echo
echo "=== the two beams must precess alike ==="
bash "$HERE/beam_symmetry.sh"

echo
echo "=== the two BDS beam generators must agree ==="
bash "$HERE/bds_symmetry.sh"

echo
echo "=== spin storage, and collision sampling on every path ==="
bash "$HERE/spin_storage.sh"

echo
echo "=== the roll carries the spin through a full collision ==="
bash "$HERE/roll_covariance.sh"

echo
echo "=== luminosity-weighted polarization ==="
bash "$HERE/luminosity_weighted.sh"
