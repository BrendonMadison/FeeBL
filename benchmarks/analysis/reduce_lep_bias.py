#!/usr/bin/env python3
"""Compare like-for-like leading-order OPAL boundary estimators.

The published radiation/acceptance result is a separate observable. Conditional
probe SEMs include the paired inner/outer covariance, but do not estimate field
sampling uncertainty or tracking discretization error.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from comparison_statistics import INPUTS, read_rows


def boundary_bias(inner_urad, outer_urad):
    """Return the leading-order relative luminosity bias in percent."""
    inner_rad, outer_rad = 0.0313, 0.0516
    return (
        -200
        / (inner_rad ** -2 - outer_rad ** -2)
        * (
            np.asarray(inner_urad) * 1e-6 / inner_rad ** 3
            - np.asarray(outer_urad) * 1e-6 / outer_rad ** 3
        )
    )


def calculate():
    """Match the reference's two printed mean angles to the same estimator."""
    reference = float(boundary_bias(12.81, 11.19))
    records = []
    for name in ("lep_probe_records.csv", "lep_transverse_refined_probe_records.csv"):
        rows = [row for row in read_rows(INPUTS / name) if row["role"] == "electron"]
        inner = {
            int(r["event_id"]): float(r["focus_urad"])
            for r in rows
            if r["cohort"] == "inner"
        }
        outer = {
            int(r["event_id"]) - 1024: float(r["focus_urad"])
            for r in rows
            if r["cohort"] == "outer"
        }
        if inner.keys() != outer.keys():
            raise ValueError("LEP probe pairing is incomplete")
        bias = boundary_bias(
            [inner[k] for k in sorted(inner)], [outer[k] for k in sorted(inner)]
        )
        mean, sem = float(bias.mean()), float(bias.std(ddof=1) / np.sqrt(len(bias)))
        records.append(
            {
                "source": name,
                "probe_pairs": len(bias),
                "mean_bias_percent": mean,
                "conditional_probe_sem_percent": sem,
                "same_estimator_reference_percent": reference,
                "relative_difference_percent": 100 * (mean / reference - 1),
                "published_radiation_acceptance_bias_percent": -0.1059,
                "qualification": "Leading boundary estimator compared separately from the radiation/acceptance result",
            }
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(calculate(), stream, indent=2)
        stream.write("\n")
