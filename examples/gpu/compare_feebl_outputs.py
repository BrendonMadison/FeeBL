#!/usr/bin/env python3
"""Compare two FeeBL text outputs with a declared numeric tolerance."""

import argparse
import math
import re
import sys


NUMBER = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--rtol", type=float, default=5.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    with open(args.reference, encoding="utf-8") as stream:
        reference = stream.readlines()
    with open(args.candidate, encoding="utf-8") as stream:
        candidate = stream.readlines()
    if len(reference) != len(candidate):
        print(f"line-count mismatch: {len(reference)} != {len(candidate)}")
        return 1

    maximum_absolute = 0.0
    maximum_relative = 0.0
    compared = 0
    for line_number, (left, right) in enumerate(zip(reference, candidate), 1):
        left_values = [float(value) for value in NUMBER.findall(left)]
        right_values = [float(value) for value in NUMBER.findall(right)]
        left_shape = NUMBER.sub("<number>", left)
        right_shape = NUMBER.sub("<number>", right)
        if left_shape != right_shape or len(left_values) != len(right_values):
            print(f"non-numeric structure differs on line {line_number}")
            print(f"reference: {left.rstrip()}")
            print(f"candidate: {right.rstrip()}")
            return 1
        for index, (expected, actual) in enumerate(
            zip(left_values, right_values), 1
        ):
            absolute = abs(expected - actual)
            scale = max(abs(expected), abs(actual))
            relative = absolute / scale if scale else 0.0
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            compared += 1
            if not math.isclose(
                expected, actual, rel_tol=args.rtol, abs_tol=args.atol
            ):
                print(
                    f"line {line_number} value {index}: {actual} != "
                    f"{expected} (abs={absolute:.6g}, rel={relative:.6g})"
                )
                return 1
    print(
        f"PASS: {compared} numeric fields, max_abs={maximum_absolute:.6g}, "
        f"max_rel={maximum_relative:.6g}, rtol={args.rtol:g}, "
        f"atol={args.atol:g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
