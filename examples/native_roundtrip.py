"""Round-trip gate for the FEEBL native format: write it, read it, check it.

Two halves, and the second is the one that gives the first any meaning.

POSITIVE: the file FEEBL just wrote parses, and its CONTENT honours every
convention its own header declares -- not merely that the keys are present,
which is all the header test could check.

NEGATIVE: the reader rejects damaged files. A validator that accepts everything
would pass the positive half trivially, so each corruption below is applied to
the real file and must raise.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
import feebl_native as fn

fails = 0


def ok(cond, what):
    global fails
    print("  %-56s %s" % (what, "PASS" if cond else "FAIL"))
    if not cond:
        fails += 1


def raises(exc, fnc, what):
    try:
        fnc()
    except exc:
        ok(True, what)
        return
    except Exception as e:                      # wrong exception type is a fail
        ok(False, "%s (raised %s instead)" % (what, type(e).__name__))
        return
    ok(False, "%s (accepted it)" % what)


path = sys.argv[1]
phi_expected = float(sys.argv[2])

print("=== the file FEEBL wrote reads back ===")
f = fn.read(path)
ok(True, "the reader accepts it")
ok(f.block == "lumi_ee", "block is lumi_ee")
ok(len(f.rows) > 0, "it holds at least one record (%d)" % len(f.rows))
ok(f.declared_records == len(f.rows),
   "footer count equals the rows read (%d)" % len(f.rows))

EXPECT = ("e1 e2 x y z t vx1 vy1 vz1 vx2 vy2 vz2 "
          "sx1 sy1 sz1 sx2 sy2 sz2 index").split()
ok(f.columns == EXPECT, "columns are exactly the documented layout")

print()
print("=== the CONTENT honours the conventions the header declares ===")
ok(abs(f.half_crossing_angle - phi_expected) < 1e-9,
   "declared half crossing angle is %g" % phi_expected)
ok(f.frame == ("head_on" if phi_expected else "lab"),
   "frame is %s" % ("head_on" if phi_expected else "lab"))
ok(f.momenta_carry_crossing_angle is False,
   "and it admits the momenta do not carry the angle")

bad_sign = bad_unit = bad_e = bad_spin = 0
mod = {1: [], 2: []}
for r in f.rows:
    if not (r["vz1"] > 0.0):
        bad_sign += 1
    if not (r["vz2"] < 0.0):
        bad_sign += 1
    for b in (1, 2):
        n = math.sqrt(r["vx%d" % b] ** 2 + r["vy%d" % b] ** 2 + r["vz%d" % b] ** 2)
        if abs(n - 1.0) > 1e-6:
            bad_unit += 1
        if not (r["e%d" % b] > 0.0):
            bad_e += 1
        m = math.sqrt(r["sx%d" % b] ** 2 + r["sy%d" % b] ** 2 + r["sz%d" % b] ** 2)
        mod[b].append(m)
        if m > 1.0 + 1e-6:
            bad_spin += 1

ok(bad_sign == 0, "vz_sign_convention holds on every record")
ok(bad_unit == 0, "every direction is a unit vector")
ok(bad_e == 0, "every energy is positive")
if f.spin_present:
    # The header declares the modulus IS the degree of polarization, not 1.
    # That was measured, and it was wrong before: FEEBL initialises every
    # particle to the beam polar vector, so polar_z=0.8 stores |s|=0.8. The
    # header used to say "unit_vector_lab_frame", which would have had a
    # consumer rescale every spin by 1/P and report full polarization.
    ok(f.meta.get("spin_is_unit_vector") == "no",
       "the header does NOT claim spin is a unit vector")
    ok(bad_spin == 0, "no stored polarization exceeds 1")
    for b, want in ((1, 0.8), (2, 0.3)):
        lo, hi = min(mod[b]), max(mod[b])
        ok(abs(lo - want) < 2e-3 and abs(hi - want) < 2e-3,
           "beam %d |s| is the deck's polarization %.1f (%.4f..%.4f)"
           % (b, want, lo, hi))
    ok(max(mod[1]) - min(mod[1]) < 1e-3,
       "and precession preserves the modulus across records")
else:
    ok(True, "spin_present=no, so no spin to check")

print()
print("=== the lab-frame correction ===")
if phi_expected:
    d1h = f.headon_direction(f.rows[0], 1)
    d1l = f.lab_direction(f.rows[0], 1)
    d2h = f.headon_direction(f.rows[0], 2)
    d2l = f.lab_direction(f.rows[0], 2)
    s = math.sin(phi_expected)
    ok(abs((d1l[0] - d1h[0]) - s) < 1e-4,
       "beam 1 x-direction shifts by +sin(phi)")
    ok(abs((d2l[0] - d2h[0]) - s) < 1e-4,
       "beam 2 x-direction shifts by +sin(phi) TOO (not -)")
    ok((d1l[0] - d1h[0]) * (d2l[0] - d2h[0]) > 0.0,
       "both beams shift the SAME way, as the lab geometry requires")
    E, px, py, pz = 12.0, 0.3, -0.2, -11.0
    E2, px2, py2, pz2 = f.to_lab(E, px, py, pz)
    m2a = E * E - px * px - py * py - pz * pz
    m2b = E2 * E2 - px2 * px2 - py2 * py2 - pz2 * pz2
    ok(abs(m2b - m2a) < 1e-8 * (1 + abs(m2a)), "to_lab preserves invariant mass")
else:
    ok(f.to_lab(9.0, 1.0, 2.0, 3.0) == (9.0, 1.0, 2.0, 3.0),
       "at zero crossing angle to_lab is exactly the identity")

print()
print("=== the reader REJECTS damaged files (else the above proves nothing) ===")
raw = open(path).read().split("\n")

def write_tmp(lines, name):
    p = path + "." + name
    open(p, "w").write("\n".join(lines))
    return p

# a data row removed, footer left claiming the old count
trunc = [l for l in raw]
for i in range(len(trunc) - 1, -1, -1):
    if trunc[i] and not trunc[i].startswith("#"):
        del trunc[i]
        break
p = write_tmp(trunc, "trunc")
raises(fn.NativeFormatError, lambda: fn.read(p), "a truncated file is rejected")

# a required convention key removed
nokey = [l for l in raw if not l.startswith("#vz_sign_convention")]
p = write_tmp(nokey, "nokey")
raises(fn.NativeFormatError, lambda: fn.read(p),
       "a header missing a convention is rejected")

# a row with the wrong number of fields
badrow = list(raw)
for i, l in enumerate(badrow):
    if l and not l.startswith("#"):
        badrow[i] = l + " 999"
        break
p = write_tmp(badrow, "badrow")
raises(fn.NativeFormatError, lambda: fn.read(p),
       "a row wider than #COLUMNS is rejected")

# not our format at all
p = write_tmp(["something else entirely"] + raw[1:], "nomagic")
raises(fn.NativeFormatError, lambda: fn.read(p),
       "a file without the magic line is rejected")

# and the frame guard
if phi_expected:
    raises(fn.FrameError, lambda: fn.read(path, allow_head_on=False),
           "head-on directions are refused unless asked for explicitly")
else:
    ok(True, "frame guard not applicable to a lab-frame file")

print()
print("  %d failure(s)" % fails)
sys.exit(1 if fails else 0)
