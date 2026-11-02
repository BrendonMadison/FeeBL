#!/usr/bin/env python3
"""Compare the unconditional legacy fine charge-density luminosity."""

import argparse
import math
import re
from pathlib import Path


PATTERN = re.compile(r"^lumi_fine\s*=\s*([^\s]+)", re.MULTILINE)


def read_value(path: Path):
    match = PATTERN.search(path.read_text())
    if not match:
        raise SystemExit(f"{path}: lumi_fine not found")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise SystemExit(f"{path}: lumi_fine is nonfinite")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    args = parser.parse_args()
    expected = read_value(args.reference)
    observed = read_value(args.candidate)
    bound = args.rtol * max(1.0, abs(expected), abs(observed))
    error = abs(observed - expected)
    if error > bound:
        raise SystemExit(
            f"lumi_fine mismatch expected={expected:.17g} "
            f"observed={observed:.17g} error={error:.6g} bound={bound:.6g}")
    print(
        f"PASS: lumi_fine relative error="
        f"{error / max(1.0, abs(expected)):.6g} <= {args.rtol:g}")


if __name__ == "__main__":
    main()
