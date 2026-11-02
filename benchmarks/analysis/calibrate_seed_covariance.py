#!/usr/bin/env python3
"""Check ensemble uncertainty on held-out collisions and force-free controls.

Eight complete collisions estimate sampling covariance; four untouched
collisions test it. Student-t, rather than unit-normal, reference distributions
account for estimating a variance from eight observations. Correlated bins do
not supply independent degrees of freedom, and the report does not infer a
global chi-square. No observed discrepancy rescales an uncertainty estimate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from comparison_statistics import (
    PANELS,
    BENCHMARKS,
    column,
    read_rows,
    reference_estimate,
    sample_covariance,
)


def projection_check(values, target, train_count):
    """Test a fixed linear observable using independent training/test means."""
    training, held_out = values[:train_count], values[train_count:]
    variance = np.var(training, ddof=1)
    prediction_se = np.sqrt(variance * (1 / len(training) + 1 / len(held_out)))
    difference = float(held_out.mean() - training.mean())
    statistic = difference / prediction_se if prediction_se > 0 else None
    mean = float(values.mean())
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    analytic_statistic = (
        (mean - target) / standard_error if standard_error > 0 else None
    )
    return {
        "mean": mean,
        "standard_error_of_mean": standard_error,
        "fixed_reference": float(target),
        "held_out_minus_training": difference,
        "held_out_difference_standard_error": float(prediction_se),
        "held_out_t": statistic,
        "held_out_degrees_of_freedom": len(training) - 1,
        "held_out_two_sided_p": float(2 * t.sf(abs(statistic), len(training) - 1))
        if statistic is not None
        else None,
        "reference_t": analytic_statistic,
        "reference_degrees_of_freedom": len(values) - 1,
        "reference_two_sided_p": float(
            2 * t.sf(abs(analytic_statistic), len(values) - 1)
        )
        if analytic_statistic is not None
        else None,
    }


def calibrate(ensemble, train_count=8):
    """Retain full covariance and summarize prespecified, low-dimensional checks."""
    samples = np.load(ensemble)
    result = {
        "sampling_unit": "independent complete native collision",
        "training_count": train_count,
        "variance_rule": "unbiased seed covariance; covariance of mean is S/n",
        "selection_rule": "first eight seeds train; remaining seeds held out; no tuning",
        "qualification": "Small-ensemble check, not numerical convergence or reference-error calibration",
        "student_reference_scope": "The t7 reference assumes approximately Gaussian complete-run estimates; sparse or strongly skewed bins need more seeds and remain descriptive.",
        "panels": [],
    }
    for name in sorted(key for key in samples.files if "__" not in key):
        values = samples[name]
        if len(values) < train_count + 4:
            raise ValueError("At least four held-out collisions are required")
        rows = read_rows(PANELS / (name + ".csv"))
        training, held_out = values[:train_count], values[train_count:]
        mean, covariance = sample_covariance(values)
        train_variance = np.var(training, axis=0, ddof=1)
        selected = train_variance > 0
        differences = held_out.mean(axis=0) - training.mean(axis=0)
        variance = train_variance * (1 / train_count + 1 / len(held_out))
        standardized = differences[selected] / np.sqrt(variance[selected])
        critical = t.ppf(0.975, train_count - 1)
        entry = {
            "panel": name,
            "seed_count": len(values),
            "held_out_count": len(held_out),
            "covariance_rank": int(np.linalg.matrix_rank(values - mean)),
            "nonzero_training_variance_bins": int(selected.sum()),
            "excluded_zero_variance_bins": np.flatnonzero(~selected).tolist(),
            "held_out_t_rms_descriptive": float(np.sqrt(np.mean(standardized ** 2))),
            "expected_t_rms": float(np.sqrt((train_count - 1) / (train_count - 3))),
            "fraction_inside_pointwise_95percent_t_interval": float(
                np.mean(abs(standardized) <= critical)
            ),
            "fraction_interpretation": "descriptive: bins are correlated; no binomial coverage error assigned",
            "maximum_absolute_held_out_t": float(np.max(abs(standardized))),
            "covariance_trace_of_mean": float(np.trace(covariance)),
        }
        historical_rows = read_rows(
            BENCHMARKS / "example_output/historical_comparisons" / (name + ".csv")
        )
        historical = np.sqrt(column(historical_rows, "simulation_variance"))
        empirical_single_run = np.sqrt(np.diag(covariance) * len(values))
        good = (historical > 0) & (empirical_single_run > 0)
        if np.any(good):
            ratios = historical[good] / empirical_single_run[good]
            entry[
                "historical_proxy_over_single_seed_empirical_sigma_quantiles"
            ] = np.quantile(ratios, [0.1, 0.5, 0.9]).tolist()
        if name.startswith("monochromatization_"):
            reference, _, _ = reference_estimate(name, rows)
            width = column(rows, "bin_high") - column(rows, "bin_low")
            center = (column(rows, "bin_high") + column(rows, "bin_low")) / 2
            # Integrals use the same piecewise-constant histogram representation
            # for both prediction and measurement. Centering avoids cancellation.
            projectors = {
                "mean_offset_from_250_GeV": width * (center - 250),
                "second_moment_about_250_GeV2": width * (center - 250) ** 2,
                "mass_above_249_8_GeV": np.maximum(
                    0,
                    column(rows, "bin_high")
                    - np.maximum(column(rows, "bin_low"), 249.8),
                ),
            }
            entry["fixed_observables"] = {
                label: projection_check(
                    values @ weights, reference @ weights, train_count
                )
                for label, weights in projectors.items()
            }
        result["panels"].append(entry)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-count", default=8, type=int)
    args = parser.parse_args()
    if args.train_count < 4:
        parser.error("At least four training seeds are required")
    with args.output.open("x") as stream:
        json.dump(
            calibrate(args.ensemble, args.train_count),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
