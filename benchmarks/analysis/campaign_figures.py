#!/usr/bin/env python3
"""Plot completed-family comparisons and descriptive seed-calibration checks."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_statistics import column, read_rows
from regenerate_paper import (
    binned_panel,
    comparison_axes,
    collider_spectrum_figure,
    draw_pulls,
    error_series,
    save,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_figure(document, output):
    """Compare descriptive bin summaries with their finite-ensemble reference."""
    panels = document["panels"]
    names = {
        "monochromatization_d0_pull": r"Geometry: $D_x=0$",
        "monochromatization_opposite_pull": "Geometry: opposite dispersion",
        "monochromatization_same_pull": "Geometry: equal dispersion",
    }
    labels = [
        names.get(
            row["panel"], row["panel"].replace("_pull", "").replace("_", " ").upper()
        )
        for row in panels
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, max(3, 1.5 + 0.28 * len(panels))),
        sharey=True,
        layout="constrained",
    )
    y = np.arange(len(panels))
    axes[0].plot(
        [row["held_out_t_rms_descriptive"] for row in panels], y, "o", color="#0072B2"
    )
    axes[0].axvline(np.sqrt(7 / 5), color=".25", ls="--", label=r"$t_7$ expectation")
    axes[0].set(
        xlabel="Held-out statistic RMS",
        xlim=(
            0,
            max(1.45, 1.1 * max(row["held_out_t_rms_descriptive"] for row in panels)),
        ),
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(
        [row["fraction_inside_pointwise_95percent_t_interval"] for row in panels],
        y,
        "s",
        color="#D55E00",
    )
    axes[1].axvline(0.95, color=".25", ls="--")
    axes[1].set(
        xlabel="Fraction inside pointwise 95% interval",
        xlim=(
            min(
                0.85,
                min(
                    row["fraction_inside_pointwise_95percent_t_interval"]
                    for row in panels
                )
                - 0.03,
            ),
            1.01,
        ),
    )
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
    figure.suptitle(
        "Eight training and four held-out collisions; correlated-bin summaries",
        fontsize=9,
    )
    save(figure, output, "seed_covariance_calibration")


def geometry_figures(comparisons, calibration, output):
    """Show three force-free Gaussian controls and their fixed-observable pulls."""
    cases = ("d0", "opposite", "same")
    titles = (r"$D_x=0$", "Opposite dispersion", "Equal dispersion")
    data = [
        read_rows(comparisons / f"monochromatization_{case}_pull.csv") for case in cases
    ]
    limits = (
        min(column(rows, "bin_low").min() for rows in data),
        max(column(rows, "bin_high").max() for rows in data),
    )
    figure, axes = plt.subplots(
        2,
        3,
        sharex="col",
        figsize=(6.7, 4.6),
        gridspec_kw={"height_ratios": [2.5, 1.4]},
        layout="constrained",
    )
    for index, (case, title) in enumerate(zip(cases, titles)):
        rows = data[index]
        x = (column(rows, "bin_low") + column(rows, "bin_high")) / 2
        top, bottom = axes[:, index]
        top.plot(
            x,
            column(rows, "reference_gaussian_density_per_GeV"),
            color=".2",
            lw=1,
            label="Analytic Gaussian",
        )
        error_series(
            top,
            x,
            column(rows, "simulation_density_per_GeV"),
            np.sqrt(column(rows, "simulation_variance")),
            "FeeBL: 12-seed mean",
            "#0072B2",
            "o",
        )
        draw_pulls(bottom, rows, x)
        top.set(title=title, ylabel=r"Density [GeV$^{-1}$]" if index == 0 else "")
        bottom.set(xlabel=r"$W$ [GeV]", ylabel="Pull" if index == 0 else "")
        bottom.set_xlim(*limits)
        if index == 0:
            top.legend(fontsize=7, loc="lower left")
    save(figure, output, "geometry_seed_controls")
    fixed = {
        row["panel"]: row["fixed_observables"]
        for row in calibration["panels"]
        if "fixed_observables" in row
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(6.7, 2.9), sharey=True, layout="constrained"
    )
    observables = (
        ("mean_offset_from_250_GeV", r"Mean $W$"),
        ("second_moment_about_250_GeV2", r"$\langle(W-250\,\mathrm{GeV})^2\rangle$"),
        ("mass_above_249_8_GeV", r"Fraction $W\geq249.8$ GeV"),
    )
    for axis, (observable, title) in zip(axes, observables):
        values = [
            fixed[f"monochromatization_{case}_pull"][observable]["reference_t"]
            for case in cases
        ]
        axis.axvspan(-1, 1, color=".9")
        axis.axvline(0, color=".25", lw=0.7)
        axis.errorbar(
            values, range(3), xerr=np.ones(3), fmt="o", capsize=3, color="#0072B2"
        )
        axis.set(title=title, xlabel="Fixed-observable pull", xlim=(-3.1, 3.1))
        axis.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(range(3), titles)
    axes[0].invert_yaxis()
    save(figure, output, "geometry_fixed_observables")


def generate(compact, output):
    """Generate only supported families; preserve explicit coverage omissions."""
    compact, output = Path(compact), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    calibration = json.loads((compact / "calibration.json").read_text())
    comparisons = compact / "comparisons"
    calibration_figure(calibration, output)
    if all(
        (comparisons / f"monochromatization_{case}_pull.csv").is_file()
        for case in ("d0", "opposite", "same")
    ):
        geometry_figures(comparisons, calibration, output)
    for prefix, title in (
        ("ilc_1000", "ILC 1 TeV"),
        ("halhf_250", "HALHF 250 GeV"),
        ("halhf_380", "HALHF 380 GeV"),
        ("halhf_550", "HALHF 550 GeV"),
    ):
        if not all(
            (comparisons / f"{prefix}_{scope}_pull.csv").is_file()
            for scope in ("full", "peak")
        ):
            continue
        collider_spectrum_figure(
            comparisons, output, prefix, title, prefix + "_seed_comparison"
        )
    inputs = [
        compact / "calibration.json",
        *sorted(comparisons.glob("*.csv")),
        *sorted(comparisons.glob("*.npz")),
        comparisons / "comparison_coverage.json",
    ]
    receipt = {
        "schema": "feebl.campaign-figures/1",
        "producer_sha256": digest(__file__),
        "plot_helpers_sha256": digest(Path(__file__).with_name("regenerate_paper.py")),
        "inputs": {
            str(path.relative_to(compact)).replace("\\", "/"): digest(path)
            for path in inputs
        },
        "coverage": json.loads((comparisons / "comparison_coverage.json").read_text()),
        "outputs": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "qualification": "Only completed ensembles are shown; covariance calibration is descriptive, not a convergence certificate.",
    }
    (output / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.compact, args.output)
