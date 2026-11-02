#!/bin/bash
# Canonical conversions and the unsupported-reader contract always run.
# With HDF5 development tools, generate a local fixture and test real reads.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

echo "=== canonical tier (no dependencies) ==="
CXX=${CXX:-c++}
"$CXX" -std=c++17 -O2 -Wall -I"$ROOT/src" \
    "$HERE/test_canonical.cc" -o "$OUT/tc"
"$OUT/tc"

echo
echo "=== reader without HDF5: explicit rejection ==="
"$CXX" -std=c++17 -O2 -Wall -I"$ROOT/src" \
    "$HERE/test_reader.cc" "$ROOT/src/bunchReaderCPP.cc" -o "$OUT/tr_no_hdf5"
"$OUT/tr_no_hdf5"

echo "=== HDF5 reader: generated fixture ==="
h5_compiler=()
h5_flags=()
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists hdf5; then
  h5_compiler=("$CXX")
  read -r -a h5_flags <<< "$(pkg-config --cflags --libs hdf5)"
elif command -v h5c++ >/dev/null 2>&1; then
  h5_compiler=(h5c++)
fi
if [ "${#h5_compiler[@]}" -gt 0 ]; then
  # Both programs use the same discovered library. No Python HDF5 package or
  # file from a separate documentation checkout is required.
  "${h5_compiler[@]}" -std=c++17 -O2 -Wall "$HERE/make_fixture.cc" \
      "${h5_flags[@]}" -o "$OUT/make_fixture"
  "$OUT/make_fixture" "$OUT/arbitrary_bunch_example.h5"
  "${h5_compiler[@]}" -std=c++17 -O2 -Wall -DFEEBL_HDF5 -I"$ROOT/src" \
      "$HERE/test_reader.cc" "$ROOT/src/bunchReaderCPP.cc" \
      "${h5_flags[@]}" -o "$OUT/tr"
  "$OUT/tr" "$OUT/arbitrary_bunch_example.h5"
else
  echo "  NOT RUN: positive HDF5 import tests require pkg-config hdf5 or h5c++."
  echo "  The canonical and unsupported-reader checks above were executed."
fi
