#!/bin/bash
# Portable entry point for the bundled numeric-output comparator.
# Usage: compare_numeric.sh ACTUAL EXPECTED [RELTOL]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "${PYTHON:-python3}" "$HERE/compare_numeric.py" "$@"
