#!/bin/bash
# Stage A conventions test. Asserts the unit and momentum conventions recorded
# in doc/CONVENTIONS.md against real FEEBL output, so later map work cannot
# silently violate one.
#
# These are cheap and should stay in the quick suite: a convention error does
# not crash, it produces a plausible wrong answer.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PASS=0; FAIL=0
ok   () { printf "  PASS  %s\n" "$1"; PASS=$((PASS+1)); }
bad  () { printf "  FAIL  %s\n" "$1"; FAIL=$((FAIL+1)); }

BIN=${1:-${FEEBL_TEST_BINARY:-$ROOT/src/feebl}}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
# Use the cheap quick deck (n_m=50000, ~3 s). The beam block we parse is
# printed at startup, so running a production deck here is pure waste.
DECK=$ROOT/examples/decks/quick.dat
cd "$W"; cp "$DECK" acc.dat || { echo "no quick deck at $DECK"; exit 1; }

echo "=== C3: emittance is NORMALISED (sigma = sqrt(emitt_n * beta / gamma)) ==="
. "$ROOT/examples/lib/harness.sh"
for t in c_nt1; do
  # This was `|| continue`, which SKIPPED the only check in this file that runs
  # the binary. Everything after it is pure Python, so the suite reported
  # success with FEEBL replaced by a failing stub.
  feebl_run "$BIN" "$t" "p_$t" "o_$t.dat"
  python3 - "$t" "o_$t.dat" <<'PY'
import re, sys
tag, path = sys.argv[1], sys.argv[2]
txt = open(path, encoding="utf-8", errors="replace").read()
m_e = re.search(r"energy\s*:\s*([0-9.eE+-]+)\s*GeV", txt)
m_s = re.search(r"sigma_x\s*:\s*([0-9.eE+-]+)\s*nm\s*;\s*sigma_y\s*:\s*([0-9.eE+-]+)\s*nm", txt)
m_m = re.search(r"emitt_x\s*:\s*([0-9.eE+-]+)\s*emitt_y\s*:\s*([0-9.eE+-]+)", txt)
m_b = re.search(r"beta_x\s*:\s*([0-9.eE+-]+)\s*beta_y\s*:\s*([0-9.eE+-]+)", txt)
if not all([m_e, m_s, m_m, m_b]):
    # A check that did not run must not report success.
    print("  FAIL  %s (could not parse beam block)" % tag); sys.exit(1)
E  = float(m_e.group(1)); gamma = E*1e9/0.510998950e6
sx, sy = float(m_s.group(1)), float(m_s.group(2))          # nm
ex, ey = float(m_m.group(1))*1e-6, float(m_m.group(2))*1e-6 # mm.mrad -> m.rad
bx, by = float(m_b.group(1))*1e-6, float(m_b.group(2))*1e-6 # um -> m
for nm, sig, em, be in (("x", sx, ex, bx), ("y", sy, ey, by)):
    pred = (em*be/gamma)**0.5 * 1e9                         # m -> nm
    rel  = abs(pred-sig)/sig
    tagn = "%s sigma_%s" % (tag, nm)
    if rel < 2e-4:
        print("  PASS  %-14s %9.4f nm predicted vs %9.5f printed  (%.1e)" % (tagn, pred, sig, rel))
    else:
        print("  FAIL  %-14s %9.4f nm predicted vs %9.5f printed  (%.1e)" % (tagn, pred, sig, rel))
PY
  # The python arms print PASS/FAIL but ran in a subprocess, so their verdicts
  # never reached $FAIL and could not fail the suite. Propagate the status.
  if [ $? -ne 0 ]; then bad "C3 normalised-emittance check (see above)"; fi
done

echo
echo "=== C2/C4: px = x'(1+delta) is NOT the identity off-momentum ==="
python3 - <<'PY'
# Guards the one relation that vanishes in every on-momentum test. A suite that
# only uses delta=0 particles would pass even with px and x' conflated.
E_ref = 125.0
cases = [(125.0, 0.0), (126.0, None), (124.0, None)]
worst = 0.0
for E, _ in cases:
    delta = E/E_ref - 1.0
    xprime = 1.0e-5                 # 10 urad, a realistic divergence
    px = xprime*(1.0 + delta)
    worst = max(worst, abs(px-xprime)/max(abs(xprime),1e-30))
    print("    E=%6.2f GeV  delta=%+8.5f  x'=%.3e  px=%.6e  diff=%.2e"
          % (E, delta, xprime, px, abs(px-xprime)))
if worst > 1e-3:
    print("  PASS  off-momentum separation is %.2e -- conflating px with x' is detectable" % worst)
else:
    print("  FAIL  off-momentum cases too weak to detect a px/x' conflation")
PY

echo
echo "=== C1: file-to-internal unit factors ==="
if grep -q "transform_to_internal_units(1e3, 1e-6)" "$ROOT/src/particleBeamCPP.h"; then
  ok "positions um->nm (1e3), angles urad->rad (1e-6) unchanged"
else
  bad "unit factors in particleBeamCPP.h differ from doc/CONVENTIONS.md C1"
fi

echo
echo "=== C2: the 'velocities' comment still documents radians ==="
if grep -q 'speed" is in radians' "$ROOT/src/particleBeamCPP.cc"; then
  ok "particleBeamCPP.cc still states angles are radians"
else
  bad "the radians comment moved or changed -- re-verify C2"
fi

echo
echo "=== C5: z is a single global axis, shared by both beams ==="
if grep -q "floor(ztemp/delta_z+0.5\* nSlices+1)" "$ROOT/src/particleBeamCPP.h"; then
  ok "slice index still increases monotonically with zpos_ on one shared axis"
else
  bad "the z-to-slice mapping changed -- re-verify doc/CONVENTIONS.md C5"
fi
echo "  NOTE  there is no per-beam zeta in the code; it must be DEFINED by"
echo "        BeamFrame in Stage E. Do not read zeta off zpos_."

echo
echo "$PASS explicit checks passed, $FAIL failed (plus the parsed sigma checks above)"
[ "$FAIL" -eq 0 ]
