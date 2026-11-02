#!/bin/bash
# Native format: write it, read it back, and check the reader has teeth.
#
# examples/native_output.sh checks the header DECLARES its conventions. This
# checks the records OBEY them, that an independent reader can consume the
# file, and that the reader rejects damaged input. Together they are what
# "self-describing" has to mean if it is going to replace EDM4HEP.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
D="$ROOT/examples/decks/quick.dat"
W="${TMPDIR:-/tmp}/feebl_nativert.$$"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
trap 'cd /; rm -rf "$W"' EXIT

run () {   # half_crossing_angle
  sed -e '0,/do_lumi=0/{s/do_lumi=0/store_native=1;bmt_precession=1;do_lumi=1/}' \
      -e '0,/num_lumi=100000/{s/num_lumi=100000/num_lumi=200/}' \
      -e "s/waist_y=0;/waist_y=0;polar_z.1=0.8;polar_z.2=-0.3;/g" \
      -e "s/angle_x.1=0.00000000/angle_x.1=$1/; s/angle_x.2=0.00000000/angle_x.2=$1/" \
      "$D" > acc.dat
  rm -f lumi.ee.out.fne
  "$B" c_nt1 p_c_nt1 o.dat > stdout.log 2>&1 || true
  if [ ! -s lumi.ee.out.fne ]; then
    echo "  no native file produced for angle=$1 -- this test checked NOTHING."
    sed -n '1,20p' stdout.log
    exit 1
  fi
}

echo "### finite crossing angle (15 mrad): the head-on case ###"
run 0.015
python3 "$HERE/native_roundtrip.py" lumi.ee.out.fne 0.015 || exit 1

echo
echo "### zero crossing angle: the file really is the lab frame ###"
run 0.0
python3 "$HERE/native_roundtrip.py" lumi.ee.out.fne 0.0 || exit 1
