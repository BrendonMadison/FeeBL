#!/usr/bin/env python3
"""Replot preserved comparisons and horizontal-dispersion scans.

No native solver is executed. Per-bin pulls are recomputed using both stored
variance contributions and their covariance. Their error bars are +/-1 with
the estimated denominator held fixed. Bin correlations and uncertainty-model
sensitivity prevent interpreting these plots as calibrated significance tests.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).resolve().parents[1] / "example_output"
COLORS = {"ilc": "#0072B2", "halhf": "#D55E00", "fcc": "#009E73"}


def read_rows(path: Path) -> list[dict]:
    """Read named CSV columns without changing normalization or units."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows: list[dict], name: str) -> np.ndarray:
    """Convert a finite numerical column; reject malformed comparison data."""
    values = np.array([float(row[name]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite values in {name}")
    return values


def plot_comparison(path: Path, output: Path) -> dict:
    """Write an upper comparison and lower pull panel; return pull diagnostics."""
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty comparison: {path.name}")
    fields = rows[0]
    if "simulation_mass" in fields:
        simulation_key, reference_key = "simulation_mass", "reference_mass"
        ylabel = "Probability in retained bin"
    elif "simulation_density_per_GeV" in fields:
        simulation_key = "simulation_density_per_GeV"
        reference_key = "reference_gaussian_density_per_GeV"
        ylabel = r"Probability density [GeV$^{-1}$]"
    elif "simulation_mean_urad" in fields:
        simulation_key, reference_key = "simulation_mean_urad", "reference_mean_urad"
        ylabel = r"Mean focusing [$\mu$rad]"
    else:
        simulation_key, reference_key = "simulation_count", "reference_count"
        ylabel = "Individual leptons per bunch crossing"
    simulation, reference = column(rows, simulation_key), column(rows, reference_key)
    simulation_variance = column(rows, "simulation_variance")
    reference_variance = column(rows, "reference_variance")
    covariance = column(rows, "simulation_reference_covariance")
    variance = simulation_variance + reference_variance - 2 * covariance
    if (
        np.any(variance <= 0)
        or np.any(simulation_variance < 0)
        or np.any(reference_variance < 0)
    ):
        raise ValueError(f"Invalid uncertainty model: {path.name}")
    pull = (simulation - reference) / np.sqrt(variance)
    if not np.allclose(pull, column(rows, "pull"), rtol=2e-12, atol=2e-12):
        raise ValueError(
            f"Stored pulls disagree with their declared definition: {path.name}"
        )

    if "phi_rad" in fields:
        x, xlabel = column(rows, "phi_rad"), r"Initial azimuth $\phi$ [rad]"
    elif "bin_low" in fields:
        x = (column(rows, "bin_low") + column(rows, "bin_high")) / 2
        if "lep_focusing" in path.stem:
            xlabel = r"Focusing [$\mu$rad]"
        else:
            xlabel = r"Collision energy $W$ [GeV]"
    elif "W_GeV" in fields:
        x, xlabel = column(rows, "W_GeV"), r"Collision energy $W$ [GeV]"
    else:
        x, xlabel = np.arange(len(rows)), "Pair-production channel"

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.5, 6.3),
        gridspec_kw={"height_ratios": [3, 1.4]},
        layout="constrained",
    )
    top.errorbar(
        x,
        reference,
        yerr=np.sqrt(reference_variance),
        fmt="s",
        ms=2.8,
        capsize=2,
        elinewidth=0.7,
        color="#333333",
        label="Reference",
    )
    top.errorbar(
        x,
        simulation,
        yerr=np.sqrt(simulation_variance),
        fmt="o",
        ms=2.4,
        capsize=2,
        elinewidth=0.6,
        color=COLORS["ilc"],
        label="FeeBL",
    )
    title = path.stem.removesuffix("_pull").replace("_", " ")
    title = title.replace("ilc", "ILC").replace("halhf", "HALHF").replace("lep", "LEP")
    title = title.replace(" gp", " — GUINEA-PIG").replace(" cain", " — CAIN")
    top.set(ylabel=ylabel, title=title)
    top.legend(loc="best", frameon=False)
    if "simulation_count" in fields:
        top.set_yscale("log")
        bottom.set_xticks(x, [row["process"].replace("-", "\n") for row in rows])
    bottom.axhspan(-2, 2, color="0.93")
    bottom.axhspan(-1, 1, color="0.85")
    bottom.axhline(0, color="0.3", lw=0.7)
    bottom.errorbar(
        x,
        pull,
        yerr=np.ones_like(pull),
        fmt="o",
        ms=2.4,
        capsize=2,
        elinewidth=0.7,
        color=COLORS["ilc"],
    )
    bottom.set(xlabel=xlabel, ylabel="Pull")
    bottom.set_ylim(
        min(-3.2, float(pull.min()) - 1.2), max(3.2, float(pull.max()) + 1.2)
    )
    for ax in (top, bottom):
        ax.grid(alpha=0.15)
    fig.savefig(output / f"{path.stem}.png", dpi=180)
    plt.close(fig)
    return {
        "panel": path.stem,
        "bins": len(rows),
        "maximum_absolute_pull": float(np.max(np.abs(pull))),
        "median_absolute_pull": float(np.median(np.abs(pull))),
        "pull_rms": float(np.sqrt(np.mean(pull ** 2))),
        "bins_above_abs_two": int(np.sum(np.abs(pull) > 2)),
        "interpretation": "Descriptive estimated-uncertainty diagnostic",
    }


def plot_scans(output: Path) -> None:
    """Draw native luminosity/observed-width scans and separate analytic curves.

    Native single-seed points have no ensemble error bar. FCC-ee width points
    are excluded because their strong representation sensitivity prevents a
    physical width interpretation. This exclusion is recorded in the caption.
    """
    rows = read_rows(DATA / "scans/high_dispersion.csv")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 7), layout="constrained")
    fig.suptitle(
        "Exploratory pre-ISR collisions; bars show independent-seed SEM", fontsize=11
    )
    for machine in ("ilc", "halhf"):
        subset = sorted(
            [r for r in rows if r["machine"] == machine and r["role"] == "scan"],
            key=lambda r: float(r["dispersion_mm"]),
        )
        x = column(subset, "dispersion_mm")
        for ax, observable in zip(
            axes, ("luminosity_cm2_per_second", "observed_peak_FWHM_MeV")
        ):
            mean = column(subset, observable + "_mean")
            ax.plot(x, mean, "o-", color=COLORS[machine], label=machine.upper())
            for row in subset:
                if row[observable + "_SEM"]:
                    ax.errorbar(
                        float(row["dispersion_mm"]),
                        float(row[observable + "_mean"]),
                        yerr=float(row[observable + "_SEM"]),
                        capsize=3,
                        color=COLORS[machine],
                    )
    fcc = json.loads((DATA / "scans/fcc_crabwaist.json").read_text(encoding="utf-8"))
    fcc_rows = [r for r in fcc["rows"] if r["role"] == "single_seed_pilot"]
    axes[0].plot(
        [r["dispersion_mm"] for r in fcc_rows],
        [r["luminosity_cm2_s"] for r in fcc_rows],
        "s-",
        color=COLORS["fcc"],
        label="FCC-ee crab waist",
    )
    axes[0].set(ylabel=r"Luminosity [cm$^{-2}$ s$^{-1}$ per IP]", yscale="log")
    axes[1].set(
        ylabel="Observed peak FWHM [MeV]",
        yscale="log",
        xlabel=r"Horizontal dispersion magnitude $|D_x|$ [mm]",
    )
    axes[1].axhspan(1, 4, color="0.88", label="1–4 MeV target")
    axes[1].text(
        0.98,
        0.96,
        "FCC-ee width requires further resolution studies",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(output / "native_horizontal_dispersion.png", dpi=180)
    plt.close(fig)

    rows = read_rows(DATA / "scans/gaussian_overlap_baselines.csv")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 7), layout="constrained")
    fig.suptitle(
        "First-order Gaussian overlap; fixed incoming moments; analytic forecast",
        fontsize=11,
    )
    for machine in ("ilc", "halhf", "fcc"):
        subset = [r for r in rows if r["machine"] == machine]
        if not subset:
            continue
        x = column(subset, "dispersion_mm")
        valid = np.array(
            [r["first_order_diagnostic_within_5percent"] == "True" for r in subset]
        )
        for ax, observable in zip(
            axes, ("luminosity_cm2_s", "linear_gaussian_FWHM_MeV")
        ):
            values = column(subset, observable)
            ax.plot(x, values, color=COLORS[machine], alpha=0.25)
            ax.plot(
                x,
                np.where(valid, values, np.nan),
                color=COLORS[machine],
                label={
                    "ilc": "ILC baseline",
                    "halhf": "HALHF baseline",
                    "fcc": "FCC-ee baseline",
                }[machine],
            )
    axes[0].set(
        ylabel=r"Gaussian-overlap luminosity [cm$^{-2}$ s$^{-1}$]", yscale="log"
    )
    axes[1].set(
        ylabel="Linear Gaussian FWHM [MeV]",
        yscale="log",
        xscale="log",
        xlabel=r"Horizontal dispersion magnitude $|D_x|$ [mm]",
    )
    axes[1].text(
        0.02,
        0.03,
        "Faint: nonlinear RMS check differs by >5%\n"
        "ILC: 14 mrad + ideal tilt; HALHF: head-on; FCC-ee: 30 mrad, RF tilt off",
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
    )
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(output / "gaussian_overlap_forecasts.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New plot directory."
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory.")
    args.output.mkdir(parents=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.right": False, "axes.spines.top": False}
    )
    summary = [
        plot_comparison(path, args.output)
        for path in sorted((DATA / "comparisons").glob("*.csv"))
    ]
    plot_scans(args.output)
    with (args.output / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(
        f"Recomputed {sum(row['bins'] for row in summary)} pulls in {len(summary)} panels."
    )


if __name__ == "__main__":
    main()
