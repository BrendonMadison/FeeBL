#!/usr/bin/env python3
"""Compare stored FeeBL primary ensembles independent of legacy RNG row shuffles."""

import argparse
import math
from pathlib import Path


def read_rows(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(token) for token in line.split()]
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_number}: nonnumeric beam row: {exc}")
        if len(row) != 6 or not all(math.isfinite(value) for value in row):
            raise SystemExit(f"{path}:{line_number}: expected six finite values")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: empty beam file")
    # PARTICLE_BEAM::store_beam deliberately shuffles records with FeeBL's
    # legacy RNG.  The serial collide_ee path advances that RNG even when
    # do_lumi=0, while mode3 truthfully does not execute or consume RNG for the
    # unported pair-aggregate path.  The resulting text-row order therefore
    # differs even though transactional write-back preserves the stable-ID
    # slice order (proved separately at the adapter boundary and by the
    # Upsilon/ID digest).  Gaussian energies are source-stable and continuous,
    # so lexicographic numeric order gives a deterministic ensemble comparison.
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    reference = read_rows(args.reference)
    candidate = read_rows(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit(
            f"row count differs: {len(reference)} != {len(candidate)}")

    worst = 0.0
    worst_location = (0, 0)
    for row_index, (expected, observed) in enumerate(zip(reference, candidate)):
        for column, (left, right) in enumerate(zip(expected, observed)):
            error = abs(right - left)
            bound = args.atol + args.rtol * max(abs(left), abs(right))
            ratio = error / bound if bound else (0.0 if error == 0.0 else math.inf)
            if ratio > worst:
                worst = ratio
                worst_location = (row_index, column)
            if error > bound:
                raise SystemExit(
                    f"state mismatch row={row_index} column={column}: "
                    f"expected={left:.17g} observed={right:.17g} "
                    f"abs_error={error:.6g} bound={bound:.6g}")
    print(
        "PASS: stored primary state within rtol="
        f"{args.rtol:g}, atol={args.atol:g}; "
        f"worst_bound_fraction={worst:.6g} at row={worst_location[0]} "
        f"column={worst_location[1]}")


if __name__ == "__main__":
    main()
