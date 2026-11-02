"""Reader for the FEEBL native event format (#FEEBL-NATIVE 1.0).

This is the consumer-facing half of dropping EDM4HEP. Writing a self-describing
file is only half the job: what EDM4HEP actually gave people was the ability to
READ the output without knowing how the generator stores things. So this module
is part of the format, not a test utility.

Three things it refuses to do, because each is a documented way FEEBL output has
been misread before:

  * It will not guess units. If the header does not declare them, it raises.
  * It will not let the footer disagree with the file. A truncated file is an
    error, not a short list of records.
  * It will not hand you a lab-frame angle when the file is in the head-on
    frame. FEEBL applies a crossing angle as a position shear only, so every
    direction in such a file is short by the half crossing angle. Ask for lab
    directions and you either get the boost applied (to_lab) or an exception --
    never a silently wrong number.

Usage:
    from feebl_native import read
    f = read("lumi.ee.out.fne")
    print(f.frame, f.half_crossing_angle, len(f.rows))
    vx, vy, vz = f.lab_direction(f.rows[0], beam=1)
"""

import math

MAGIC = "#FEEBL-NATIVE"
SUPPORTED = ("1.0",)

# Declared by the writer; a file missing any of these is not self-describing
# and we decline to interpret it rather than assuming the current defaults.
REQUIRED = (
    "length_unit",
    "z_unit",
    "energy_unit",
    "angle_unit",
    "frame",
    "beam1_direction",
    "beam2_direction",
    "transverse_momenta_are",
    "vz_sign_convention",
    "momenta_carry_crossing_angle",
    "spin_present",
)


class NativeFormatError(Exception):
    pass


class FrameError(Exception):
    """Raised when a lab-frame quantity is requested from a head-on file."""


class NativeFile(object):
    def __init__(self):
        self.meta = {}
        self.block = None
        self.columns = []
        self.rows = []  # list of dict: column name -> float
        self.warnings = []
        self.declared_records = None

    @property
    def frame(self):
        return self.meta.get("frame")

    @property
    def half_crossing_angle(self):
        return float(self.meta.get("half_crossing_angle_rad", 0.0))

    @property
    def momenta_carry_crossing_angle(self):
        return self.meta.get("momenta_carry_crossing_angle") == "yes"

    @property
    def spin_present(self):
        return self.meta.get("spin_present") == "yes"

    def to_lab(self, E, px, py, pz):
        """Boost a four-momentum from the head-on frame to the lab.

        A boost, NOT a rotation. In the lab both beams carry the same sign of
        transverse momentum, so px -> px + beta*E (E > 0 always) is right and
        px -> px + phi*pz is not: the latter mirrors everything backward-going,
        including beam 2 and every backward secondary.
        """
        phi = self.half_crossing_angle
        if phi == 0.0:
            return (E, px, py, pz)
        beta = math.sin(phi)
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        return (gamma * (E + beta * px), gamma * (px + beta * E), py, pz)

    def lab_direction(self, row, beam):
        """Unit direction of one particle of a lumi_ee record, in the LAB."""
        if beam not in (1, 2):
            raise ValueError("beam must be 1 or 2")
        for k in ("vx%d" % beam, "vy%d" % beam, "vz%d" % beam, "e%d" % beam):
            if k not in self.columns:
                raise NativeFormatError("column %r absent; this is not a " "lumi_ee record" % k)
        e = row["e%d" % beam]
        vx = row["vx%d" % beam]
        vy = row["vy%d" % beam]
        vz = row["vz%d" % beam]
        E, px, py, pz = self.to_lab(e, e * vx, e * vy, e * vz)
        n = math.sqrt(px * px + py * py + pz * pz)
        if not n > 0.0:
            raise NativeFormatError("zero momentum in record")
        return (px / n, py / n, pz / n)

    def headon_direction(self, row, beam):
        """Unit direction as STORED, i.e. in the frame the file declares."""
        return (row["vx%d" % beam], row["vy%d" % beam], row["vz%d" % beam])


def read(path, allow_head_on=True):
    f = NativeFile()
    nrows = 0
    with open(path, "r") as fh:
        first = fh.readline()
        if not first.startswith(MAGIC):
            raise NativeFormatError("%s: not a FEEBL native file (no %s)" % (path, MAGIC))
        version = first.strip().split()[-1]
        if version not in SUPPORTED:
            raise NativeFormatError(
                "%s: format version %s is not supported %r" % (path, version, SUPPORTED)
            )
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:]
                if not body or body.startswith(" "):
                    continue
                key, _, val = body.partition(" ")
                if key == "WARNING":
                    f.warnings.append(val)
                elif key == "BLOCK":
                    f.block = val.strip()
                elif key == "COLUMNS":
                    f.columns = val.split()
                elif key == "END":
                    parts = val.split()
                    if len(parts) >= 3 and parts[1] == "records":
                        f.declared_records = int(parts[2])
                else:
                    f.meta[key] = val.strip()
                continue
            fields = line.split()
            if not f.columns:
                raise NativeFormatError("%s: data before #COLUMNS" % path)
            if len(fields) != len(f.columns):
                raise NativeFormatError(
                    "%s: row %d has %d fields, #COLUMNS declares %d"
                    % (path, nrows + 1, len(fields), len(f.columns))
                )
            vals = []
            for x in fields:
                v = float(x)
                if not math.isfinite(v):
                    raise NativeFormatError(
                        "%s: non-finite value %r in row %d" % (path, x, nrows + 1)
                    )
                vals.append(v)
            f.rows.append(dict(zip(f.columns, vals)))
            nrows += 1

    missing = [k for k in REQUIRED if k not in f.meta]
    if missing:
        raise NativeFormatError(
            "%s: header does not declare %s; refusing to " "infer them" % (path, ", ".join(missing))
        )
    if f.declared_records is None:
        raise NativeFormatError("%s: no #END footer -- file may be truncated" % path)
    if f.declared_records != nrows:
        raise NativeFormatError(
            "%s: footer declares %d records, file holds %d" % (path, f.declared_records, nrows)
        )
    if f.frame not in ("lab", "head_on"):
        raise NativeFormatError("%s: unknown frame %r" % (path, f.frame))
    if f.frame == "head_on" and not allow_head_on:
        raise FrameError(
            "%s: directions are in the head-on frame; pass "
            "allow_head_on=True and use to_lab()/lab_direction()" % path
        )
    return f
