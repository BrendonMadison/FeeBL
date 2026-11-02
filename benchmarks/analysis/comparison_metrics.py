#!/usr/bin/env python3
"""Evaluate descriptive conditional histogram distances without p-value claims."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_statistics import column, read_rows, write_rows
from paper_tables import tabular


def distances(low, high, first, second):
    """Integrate |CDF difference| exactly for piecewise-constant densities."""
    low, high, first, second = map(np.asarray, (low, high, first, second))
    if np.any(high <= low) or np.any(low[1:] < high[:-1] - 1e-9):
        raise ValueError("Comparison support must contain ordered disjoint bins")
    if not np.isclose(first.sum(), 1) or not np.isclose(second.sum(), 1):
        raise ValueError("Conditional histogram masses must sum to one")
    cumulative, maximum, wasserstein, previous = 0.0, 0.0, 0.0, float(low[0])
    for a, b, difference in zip(low, high, first - second):
        wasserstein += max(0.0, a - previous) * abs(cumulative)
        endpoint = cumulative + difference
        if cumulative * endpoint >= 0:
            integral = (abs(cumulative) + abs(endpoint)) / 2
        else:
            integral = (cumulative ** 2 + endpoint ** 2) / (
                2 * (abs(cumulative) + abs(endpoint))
            )
        wasserstein += (b - a) * integral
        maximum = max(maximum, abs(cumulative), abs(endpoint))
        cumulative, previous = endpoint, b
    return {
        "conditional_cdf_distance": float(maximum),
        "wasserstein1_GeV": float(wasserstein),
    }


def generate(comparisons, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result, inputs = [], {}
    for path in sorted(Path(comparisons).glob("*_pull.csv")):
        if not path.stem.startswith(("ilc_1000", "halhf_")):
            continue
        rows = read_rows(path)
        metrics = distances(
            column(rows, "bin_low"),
            column(rows, "bin_high"),
            column(rows, "simulation_mass"),
            column(rows, "reference_mass"),
        )
        pulls = column(rows, "pull")
        result.append(
            {
                "panel": path.stem,
                **metrics,
                "pull_rms_descriptive": float(np.sqrt(np.mean(pulls ** 2))),
                "maximum_absolute_pull": float(max(abs(pulls))),
                "bins": len(rows),
            }
        )
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_rows(output / "spectrum_comparison_metrics.csv", result)
    tabular(
        output / "spectrum_comparison_metrics.tex",
        ["Panel", r"$D_{\rm CDF}$", r"$W_1$ [GeV]", "Pull RMS"],
        [
            [
                r["panel"].replace("_pull", "").replace("_", " ").upper(),
                f"{r['conditional_cdf_distance']:.5f}",
                f"{r['wasserstein1_GeV']:.5f}",
                f"{r['pull_rms_descriptive']:.3f}",
            ]
            for r in result
        ],
    )
    (output / "spectrum_comparison_metrics_provenance.json").write_text(
        json.dumps(
            {
                "producer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "input_comparison_sha256": inputs,
                "normalization": "Both distributions condition on the same retained bin support; missing intervals remain gaps.",
                "wasserstein_rule": "Exact absolute linear-CDF integral within each constant-density bin and constant-CDF integral across support gaps.",
                "qualification": "Descriptive shape distances; no global significance or uncertainty from reference authors is inferred.",
            },
            indent=2,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.comparisons, args.output), indent=2))
