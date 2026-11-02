#!/bin/bash
# Run every monochromatization example. Each writes into its own directory so
# the outputs do not overwrite one another.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${FEEBL:-$HERE/../../src/feebl}"
[ -x "$BIN" ] || { echo "no FEEBL binary at $BIN (set \$FEEBL)"; exit 1; }

for deck in "$HERE"/*.dat; do
  tag=$(basename "$deck" .dat)
  case "$tag" in ilc250_*) acc=ilc250; par=p_ilc250 ;;
                 ilcz_*)   acc=ilcz;   par=p_ilcz ;;
                 fccz_*)   acc=fccz;   par=p_fccz ;;
                 *) continue ;; esac
  mkdir -p "$HERE/out/$tag"; cd "$HERE/out/$tag"
  cp "$deck" acc.dat
  echo "=== $tag ==="
  "$BIN" "$acc" "$par" out.dat < /dev/null > log.txt 2>&1 || {
    echo "  FAILED, see $PWD/log.txt"; continue; }
  grep -E "^lumi_ee = " out.dat | head -1 | sed 's/^/  /'
  grep -E "^#rms_W|^#mean_W" lumi.ee.out.dLdW.fne 2>/dev/null | sed 's/^/  /'
done
