#!/usr/bin/env python3
"""Reduce HALHF secondary populations to common log-angle/log-pT support cells.

This is a topology comparison, not a yield comparison with raster pixels.
theta is measured relative to the nearest beam axis. Hemisphere selection uses
the stored final z coordinate, independently of particle species and momentum.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_statistics import BENCHMARKS, read_rows, column


def reduce_population(path, references):
    """Count every native row on the published support grid; do not subsample."""
    data = np.loadtxt(path, ndmin=2)
    energy, vx, vy, vz, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 6]
    theta = np.arctan2(np.hypot(vx, vy), np.abs(vz))
    # Native secondary velocities are beta=p/E, not unit momentum directions.
    pt = np.abs(energy) * np.hypot(vx, vy)
    arrays = {}
    totals = {}
    for hemisphere, sign in (("forward", 1), ("backward", -1)):
        rows = references[hemisphere]
        xedges = np.unique(
            np.r_[column(rows, "log10_theta_low"), column(rows, "log10_theta_high")]
        )
        yedges = np.unique(
            np.r_[column(rows, "log10_pT_low"), column(rows, "log10_pT_high")]
        )
        for species, energy_sign in (("electron", 1), ("positron", -1)):
            selected = (
                (z * sign > 0) & (energy * energy_sign > 0) & (theta > 0) & (pt > 0)
            )
            counts = np.histogram2d(
                np.log10(theta[selected]), np.log10(pt[selected]), bins=(xedges, yedges)
            )[0]
            key = hemisphere + "_" + species
            arrays[key] = counts
            arrays[hemisphere + "_xedges"] = xedges
            arrays[hemisphere + "_yedges"] = yedges
            totals[key] = {
                "selected_rows": int(selected.sum()),
                "within_reference_support": int(counts.sum()),
            }
    return (
        arrays,
        {
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "native_rows": len(data),
            "selection": totals,
            "momentum_convention": "Native vx,vy,vz are beta=p/E; pT=abs(E)*hypot(vx,vy), with E in GeV.",
            "population_scope": "All admitted rows in this sample; process codes alone cannot isolate pair channels when Compton recoils are enabled.",
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--incoherent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = {
        side: read_rows(BENCHMARKS / "reference" / f"halhf_pairs_250_{side}_ink.csv")
        for side in ("forward", "backward")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    metadata, arrays = {}, {}
    for label, path in (("mixed", args.mixed), ("incoherent", args.incoherent)):
        values, metadata[label] = reduce_population(path, references)
        arrays.update({label + "_" + key: value for key, value in values.items()})
    np.savez_compressed(args.output / "halhf_pair_support.npz", **arrays)
    (args.output / "halhf_pair_support.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
