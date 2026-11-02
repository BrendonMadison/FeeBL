#!/usr/bin/env python3
"""Compare production serial/mode3 Upsilon moments and stable identity."""

import argparse
import math
import re
from pathlib import Path


PATTERN = re.compile(
    r"FEEBL primary Upsilon digest: beam=(?P<beam>[12]) "
    r"count=(?P<count>\d+) sum=(?P<sum>\S+) sumsq=(?P<sumsq>\S+) "
    r"max=(?P<max>\S+) nonfinite=(?P<nonfinite>\d+) "
    r"stable_id_hash=(?P<hash>\d+)")


def parse(path: Path):
    result = {}
    for match in PATTERN.finditer(path.read_text()):
        beam = int(match.group("beam"))
        result[beam] = {
            "count": int(match.group("count")),
            "sum": float(match.group("sum")),
            "sumsq": float(match.group("sumsq")),
            "max": float(match.group("max")),
            "nonfinite": int(match.group("nonfinite")),
            "hash": int(match.group("hash")),
        }
    if set(result) != {1, 2}:
        raise SystemExit(f"{path}: expected exactly one digest for each beam")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()
    reference = parse(args.reference)
    candidate = parse(args.candidate)
    for beam in (1, 2):
        left = reference[beam]
        right = candidate[beam]
        for field in ("count", "nonfinite", "hash"):
            if left[field] != right[field]:
                raise SystemExit(
                    f"beam {beam} {field} differs: {left[field]} != {right[field]}")
        if left["nonfinite"] != 0:
            raise SystemExit(f"beam {beam} contains nonfinite Upsilon")
        for field in ("sum", "sumsq", "max"):
            expected = left[field]
            observed = right[field]
            bound = args.atol + args.rtol * max(abs(expected), abs(observed))
            if not (math.isfinite(expected) and math.isfinite(observed)) or \
                    abs(observed - expected) > bound:
                raise SystemExit(
                    f"beam {beam} Upsilon {field} differs: expected={expected:.17g} "
                    f"observed={observed:.17g} bound={bound:.6g}")
    print("PASS: production Upsilon count/identity exact and moments within bound")


if __name__ == "__main__":
    main()
