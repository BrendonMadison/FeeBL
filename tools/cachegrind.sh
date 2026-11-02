#!/bin/bash
# Which kernel generates the cache misses? Whole-program counters said ILC 250
# misses 19.1% of cache references, but not where. Tier 1 of the plan assumes
# assignBeamSliceCIC's four-point scatter is the source; if it is really the
# FFT, the threading priority is wrong.
#
# cachegrind simulates the cache and attributes per function. ~50x slower,
# hence LEP1 (6 s) rather than ILC (130 s).
#
# Notes:
#  --cache-sim=yes is REQUIRED: valgrind 3.22 defaults to instruction counting
#    only, and silently reports "Events recorded: Ir" otherwise.
#  LLC modelled at 16 MB, the nearest valid size below the real 20 MB
#    (cachegrind requires a power-of-two set count). Slightly pessimistic.
set -e
W=$HOME/cg; rm -rf $W; mkdir -p $W; cd $W
cp $HOME/auto16.dat acc.dat

valgrind --tool=cachegrind --cache-sim=yes \
  --D1=32768,8,64 --LL=16777216,16,64 \
  --cachegrind-out-file=cg.out \
  $HOME/feebl/src/feebl s0 p_s0 cg.dat > vg.log 2>&1

echo "=== events recorded (must include D1mr/DLmr, not just Ir) ==="
cg_annotate cg.out 2>/dev/null | grep -E "^Events (recorded|shown)" | sed 's/^/  /'

echo
echo "=== whole-program summary ==="
cg_annotate cg.out 2>/dev/null | sed -n '/-- Summary/,/-- File:function/p' \
  | grep -vE "^--|^$|Summary|File:function" | head -6

echo
echo "=== per-function, sorted by LL data misses ==="
cg_annotate --sort=DLmr,DLmw --threshold=2 cg.out 2>/dev/null \
  | sed -n '/-- File:function/,$p' | grep -E "GRID::|FIELD::|PHYSTOOLS::|fftw|PARTICLE" \
  | head -14 | cut -c1-150
