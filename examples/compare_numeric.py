#!/usr/bin/env python3
"""Compare FeeBL text output with the inherited Numdiff relative-error rule.

Line order and nonnumeric whitespace-delimited fields must match. Finite real
fields pass if |a-b| <= relative_tolerance*min(|a|,|b|); absolute tolerance is
zero. Attached assignments such as de1=432.135; remain literal fields, as in
the previous Numdiff invocation without custom delimiters. No lines, banners,
settings, or results are excluded. This is a native-output comparator, not a
general replacement for Numdiff's complex-number and locale options.
"""
import argparse
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
import re
import sys


REAL = re.compile(r"[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
NONFINITE = re.compile(r"[+-]?(?:inf(?:inity)?|s?nan)(?:\([^)]*\))?\Z", re.I)


def fields(line):
    """Use the original space/tab field separation without dropping lines."""
    return re.findall(r"[^ \t]+", line)


def field_equal(actual, expected, relative_tolerance):
    """Return equality under the finite-real numeric or literal text rule."""
    if NONFINITE.fullmatch(actual) or NONFINITE.fullmatch(expected):
        return False
    if actual == expected:
        return True
    if not REAL.fullmatch(actual) or not REAL.fullmatch(expected):
        return False
    # Decimal arithmetic keeps a value exactly on the tolerance boundary from
    # moving across it because a binary float cannot represent the input.
    with localcontext() as context:
        context.prec = max(80, len(actual) + len(expected) + len(str(relative_tolerance)) + 16)
        left, right = Decimal(actual.replace(",", "")), Decimal(expected.replace(",", ""))
        return abs(left-right) <= relative_tolerance * min(abs(left), abs(right))


def compare_text(actual, expected, relative_tolerance):
    """Return every line/field mismatch; caller limits only printed diagnostics."""
    left, right = actual.splitlines(), expected.splitlines()
    differences = []
    if len(left) != len(right):
        differences.append(f"line count: actual {len(left)}, expected {len(right)}")
    for line_number, (a_line, b_line) in enumerate(zip(left, right), 1):
        a_fields, b_fields = fields(a_line), fields(b_line)
        if len(a_fields) != len(b_fields):
            differences.append(f"line {line_number}: field count {len(a_fields)} != {len(b_fields)}")
        for field_number, (a, b) in enumerate(zip(a_fields, b_fields), 1):
            if not field_equal(a, b, relative_tolerance):
                differences.append(f"line {line_number}, field {field_number}: {a!r} != {b!r}")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("relative_tolerance", nargs="?", default="1e-6")
    args = parser.parse_args()
    try:
        tolerance = Decimal(args.relative_tolerance)
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError("relative tolerance must be finite and nonnegative")
    except (InvalidOperation, ValueError) as error:
        parser.error(str(error))
    try:
        actual, expected = (p.read_text(encoding="utf-8") for p in (args.actual, args.expected))
        if not actual.strip() or not expected.strip():
            raise ValueError("both comparison files must be nonempty")
        differences = compare_text(actual, expected, tolerance)
    except (OSError, UnicodeError, ValueError, InvalidOperation) as error:
        print(f"compare: {error}", file=sys.stderr)
        return 3
    print(f"compare: finite-real relative tolerance {tolerance}; zero absolute tolerance")
    if differences:
        for difference in differences[:30]:
            print(difference)
        print(f"FAIL: {len(differences)} differing line/field checks")
        return 1
    print("PASS: all lines and fields agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
