#!/usr/bin/env python3
"""Regenerate the paper's numerical figures and tables from packaged data.

The comparison directory must be produced by comparison_statistics.py. It
contains both variance contributions and independently simulated seed means.
This command runs no collision simulation or fit. Historical numerical studies
retain their explicit source identities and qualification limits.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from comparison_statistics import read_rows, column, write_rows, INPUTS, BENCHMARKS
from paper_tables import build_tables
from reduce_study_sources import regenerate as regenerate_studies
from project_gaussian_overlap import generate as generate_gaussian_overlap
from sampler_oracle import evaluate as evaluate_sampler
from comparison_metrics import generate as generate_comparison_metrics
from display_rebin import load_display
from polarization_figure import generate as generate_polarization_figure
from qed_publication import DEFAULT_REDUCTION, export_products, copy_figures

DATA = BENCHMARKS / "example_output"
COLORS = {"ilc": "#0072B2", "halhf": "#D55E00", "fcc": "#009E73"}
MARKERS = {"ilc": "o", "halhf": "s", "fcc": "^"}
NAMES = {"ilc": "ILC/LCF", "halhf": "HALHF", "fcc": "FCC-ee"}


def save(fig, output, name):
    """Save a final-size vector figure and a matching inspection PNG."""
    fig.savefig(
        output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(output / f"{name}.png", dpi=160)
    plt.close(fig)


def comparison_axes():
    """Create two comparison panels with aligned, uncropped pull axes."""
    figure, axes = plt.subplots(
        2,
        2,
        sharex="col",
        figsize=(6.7, 4.7),
        gridspec_kw={"height_ratios": [2.5, 1.35]},
        layout="constrained",
    )
    return figure, axes[0], axes[1]


def error_series(axis, x, values, sigma, label, color, marker, log=False):
    """Draw capped errors; downward arrows denote log intervals reaching zero.

    Such intervals retain their full numerical errors in the data and pulls.
    Their nonpositive lower endpoint cannot be represented on a logarithmic
    axis; an arrow replaces that endpoint instead of an invented lower cap.
    """
    positive = values > 0 if log else np.ones(len(values), dtype=bool)
    normal = positive & ((values - sigma > 0) if log else positive)
    axis.errorbar(
        x[normal],
        values[normal],
        yerr=sigma[normal],
        fmt=marker,
        ms=2.1,
        capsize=1.6,
        elinewidth=0.55,
        color=color,
        label=label,
    )
    limited = positive & ~normal
    if np.any(limited):
        # Show the actual positive upper endpoint separately; Matplotlib's
        # upper-limit arrow represents only the nonrepresentable lower side.
        axis.vlines(
            x[limited],
            values[limited],
            values[limited] + sigma[limited],
            color=color,
            lw=0.55,
        )
        axis.plot(
            x[limited],
            values[limited] + sigma[limited],
            ls="",
            marker="_",
            ms=3.2,
            mew=0.55,
            color=color,
        )
        axis.errorbar(
            x[limited],
            values[limited],
            yerr=np.vstack((0.6 * values[limited], sigma[limited])),
            uplims=True,
            fmt=marker,
            ms=2.1,
            capsize=1.6,
            elinewidth=0.55,
            color=color,
        )
    if log:
        axis.set_yscale("log")


def draw_pulls(axis, rows, x, color=COLORS["ilc"], marker="o", label=None):
    """Draw all finite pulls with unit capped errors and hollow empty bins."""
    values = column(rows, "pull")
    pairs = (
        ("simulation_mass", "reference_mass"),
        ("simulation_count", "reference_count"),
        ("simulation_mean_urad", "reference_mean_urad"),
        ("simulation_density_per_GeV", "reference_gaussian_density_per_GeV"),
    )
    simulation_key, reference_key = next(pair for pair in pairs if pair[0] in rows[0])
    variance = column(rows, "simulation_variance") + column(rows, "reference_variance")
    variance -= 2 * np.array(
        [float(row.get("simulation_reference_covariance", 0)) for row in rows]
    )
    expected = (column(rows, simulation_key) - column(rows, reference_key)) / np.sqrt(
        variance
    )
    if not np.allclose(values, expected, rtol=1e-7, atol=1e-8):
        raise ValueError("Stored pulls disagree with the two variance contributions")
    empty = np.array(
        [
            str(row.get("empty_bin_regularized", "False")).lower() == "true"
            for row in rows
        ]
    )
    axis.axhspan(-2, 2, color=".94", zorder=0)
    axis.axhspan(-1, 1, color=".86", zorder=0)
    axis.axhline(0, color=".35", lw=0.6)
    for selected, face in ((~empty, color), (empty, "white")):
        axis.errorbar(
            x[selected],
            values[selected],
            yerr=np.ones(selected.sum()),
            fmt=marker,
            ms=2,
            mfc=face,
            mec=color,
            color=color,
            capsize=1.5,
            elinewidth=0.5,
            label=label if face != "white" else None,
        )
    margin = max(3.2, np.max(np.abs(values)) + 1.25)
    axis.set_ylim(-margin, margin)
    axis.set_ylabel("Pull")
    axis.grid(alpha=0.12)


def binned_panel(top, bottom, rows, title, log=False):
    """Plot conditional densities while computing pulls in original mass units."""
    low, high = column(rows, "bin_low"), column(rows, "bin_high")
    center, width = (low + high) / 2, high - low
    simulation, reference = (
        column(rows, "simulation_mass"),
        column(rows, "reference_mass"),
    )
    for key, values, color, label, marker in (
        ("reference", reference, ".2", "Published curve", "s"),
        ("simulation", simulation, COLORS["ilc"], "FeeBL", "o"),
    ):
        error_series(
            top,
            center,
            values / width,
            np.sqrt(column(rows, key + "_variance")) / width,
            label,
            color,
            marker,
            log,
        )
    top.set_title(title)
    if log:
        positive_centers = np.r_[simulation / width, reference / width]
        positive_centers = positive_centers[positive_centers > 0]
        upper = max(
            np.max((simulation + np.sqrt(column(rows, "simulation_variance"))) / width),
            np.max((reference + np.sqrt(column(rows, "reference_variance"))) / width),
        )
        top.set_ylim(0.3 * positive_centers.min(), 1.5 * upper)
    top.set_ylabel(r"Conditional density [GeV$^{-1}$]")
    top.legend(frameon=True, fontsize=8)
    top.grid(alpha=0.12)
    draw_pulls(bottom, rows, center)
    bottom.set_xlabel(r"Collision energy $W$ [GeV]")
    rms = np.sqrt(np.mean(column(rows, "pull") ** 2))
    top.text(
        0.03,
        0.05,
        f"Pull RMS = {rms:.2f}",
        transform=top.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def collider_spectrum_figure(comparisons, output, prefix, title, filename):
    """Render matched display-bin counts with complete covariance transfer."""
    figure, upper, lower = comparison_axes()
    for index, scope in enumerate(("full", "peak")):
        limits = None
        if scope == "peak":
            limits = {"ilc_1000": (990.0, 1010.0), "halhf_250": (242.0, 252.0)}.get(
                prefix
            )
        name = f"{prefix}_{scope}_pull"
        if prefix == "ilc_1000" and scope == "full":
            source_rows = read_rows(comparisons / (name + ".csv"))
            limits = (550.0, max(float(row["bin_high"]) for row in source_rows))
        if prefix in ("ilc_1000", "halhf_250"):
            tail_only = prefix == "ilc_1000" and scope == "full"
            rows = load_display(
                comparisons, name, output / "display_data", count=40, limits=limits,
                exclude_first_bin=tail_only,
                display_upper_limit=990.0 if tail_only else None,
            )
        else:
            rows = read_rows(comparisons / (name + ".csv"))
        historical = "Dirichlet proxy" in rows[0].get("simulation_method", "")
        display_scope = "tail" if prefix == "ilc_1000" and scope == "full" else scope
        panel_title = f"{title}: {display_scope}" + (" (historical)" if historical else "")
        binned_panel(upper[index], lower[index], rows, panel_title, scope == "full")
        if limits is not None:
            lower[index].set_xlim(*limits)
            if prefix == "ilc_1000":
                lower[index].set_xlim(min(float(row["bin_low"]) for row in rows),
                                      max(float(row["bin_high"]) for row in rows))
                upper[index].legend(loc="upper left", frameon=True, fontsize=8)
                metadata_path = output / "display_data" / (name + "_display.json")
                metadata = json.loads(metadata_path.read_text())
                metadata["view_limits_role"] = "Requested rebin/crop window; the displayed axis ends at retained support."
                metadata["display_axis_limits_GeV"] = list(map(float, lower[index].get_xlim()))
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    save(figure, output, filename)


def collider_figures(comparisons, output, include_focusing_pairs=True):
    """Draw the LEP, ILC and HALHF comparisons, including both uncertainty terms."""
    for prefix, filename, title in (
        ("ilc_1000", "ilc_spectrum", "ILC 1 TeV"),
        ("halhf_250", "halhf_spectra_250", "HALHF 250 GeV"),
        ("halhf_380", "halhf_spectra_380", "HALHF 380 GeV"),
        ("halhf_550", "halhf_spectra_550", "HALHF 550 GeV"),
    ):
        collider_spectrum_figure(comparisons, output, prefix, title, filename)
    if not include_focusing_pairs:
        return
    figure, upper, lower = comparison_axes()
    for index, suffix in enumerate(("vs_phi", "numerical_vs_phi")):
        rows = read_rows(comparisons / f"lep_final_focusing_{suffix}_pull.csv")
        x = column(rows, "phi_rad")
        if index == 0:
            error_series(
                upper[0],
                x,
                column(rows, "simulation_mean_urad"),
                np.sqrt(column(rows, "simulation_variance")),
                "FeeBL",
                COLORS["ilc"],
                "o",
            )
        color, marker = (".2", "s") if index == 0 else (COLORS["halhf"], "^")
        error_series(
            upper[0],
            x,
            column(rows, "reference_mean_urad"),
            np.sqrt(column(rows, "reference_variance")),
            "Published GP" if index == 0 else "Published field integral",
            color,
            marker,
        )
        draw_pulls(lower[0], rows, x, color, marker)
    upper[0].set(
        title="LEP 1994: angular dependence", ylabel=r"Mean focusing [$\mu$rad]"
    )
    upper[0].legend(fontsize=8, loc="lower right")
    lower[0].set_xlabel(r"Initial azimuth $\phi$ [rad]")
    rows = read_rows(comparisons / "lep_focusing_distribution_pull.csv")
    low, high = column(rows, "bin_low"), column(rows, "bin_high")
    x, width = (low + high) / 2, high - low
    for key, color, marker, label in (
        ("reference", ".2", "s", "Published curve"),
        ("simulation", COLORS["ilc"], "o", "FeeBL"),
    ):
        error_series(
            upper[1],
            x,
            column(rows, key + "_mass") / width,
            np.sqrt(column(rows, key + "_variance")) / width,
            label,
            color,
            marker,
        )
    upper[1].set(
        title="LEP 1994: focusing distribution", ylabel=r"Density [$\mu$rad$^{-1}$]"
    )
    upper[1].legend(fontsize=8)
    draw_pulls(lower[1], rows, x)
    lower[1].set_xlabel(r"Focusing $\theta_i-\theta_f$ [$\mu$rad]")
    save(figure, output, "lep_focusing")
    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6.7, 4.9),
        layout="constrained",
        gridspec_kw={"height_ratios": [2.5, 1.4]},
    )
    x = np.arange(3)
    for index, code in enumerate(("gp", "cain")):
        rows = read_rows(comparisons / f"ilc_pair_yields_{code}_pull.csv")
        label, color, marker = (
            ("GUINEA-PIG", ".2", "s") if index == 0 else ("CAIN", COLORS["halhf"], "^")
        )
        error_series(
            axes[0],
            x + (index - 1) * 0.15,
            column(rows, "reference_count"),
            np.sqrt(column(rows, "reference_variance")),
            label,
            color,
            marker,
            True,
        )
        draw_pulls(axes[1], rows, x + (index - 0.5) * 0.08, color, marker, label)
    error_series(
        axes[0],
        x + 0.15,
        column(rows, "simulation_count"),
        np.sqrt(column(rows, "simulation_variance")),
        "FeeBL",
        COLORS["ilc"],
        "o",
        True,
    )
    axes[0].set(
        ylabel=r"Individual $e^-+e^+$ per crossing",
        title="ILC 500 GeV; 5 MeV production threshold",
    )
    axes[0].legend(ncols=3, fontsize=8)
    axes[1].set_xticks(x, ["Breit–Wheeler", "Bethe–Heitler", "Landau–Lifshitz"])
    axes[1].legend(ncols=2, fontsize=8)
    save(figure, output, "ilc_pair_yields")


def sampler_figure(output, document):
    """Use independently seeded native trials and an exact-Bessel target."""
    rows = document["runs"]
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Sampler settings reused a seed")
    subsets = []
    for spin in (-1, 0, 1):
        for field in ("pull_p", "pull_x"):
            values = np.array([row[field] for row in rows if row["s2"] == spin])
            subsets.append(
                {
                    "S2_equals_Sy": spin,
                    "observable": field,
                    "independent_streams": len(values),
                    "mean_pull": values.mean(),
                    "empirical_SEM_of_mean_pull": values.std(ddof=1)
                    / np.sqrt(len(values)),
                    "unit_variance_SEM": 1 / np.sqrt(len(values)),
                    "pull_RMS": np.sqrt(np.mean(values ** 2)),
                    "interpretation": "Descriptive mean across fixed chi/exposure settings; S2 is the transverse radiation-basis spin, equal to Sy in this fixture, not beam helicity.",
                }
            )
    write_rows(output / "sampler_spin_subsets.csv", subsets)
    figure, axes = plt.subplots(
        1, 2, sharey=True, figsize=(6.7, 4.1), layout="constrained"
    )
    for axis, field, title in zip(
        axes,
        ("pull_p", "pull_x"),
        ("Photon-emission probability", "Mean radiated-energy fraction"),
    ):
        axis.axhspan(-2, 2, color=".94")
        axis.axhspan(-1, 1, color=".86")
        axis.axhline(0, lw=0.6, color=".4")
        for spin_index, (spin, marker, color) in enumerate(
            ((-1, "v", COLORS["ilc"]), (0, "o", ".15"), (1, "^", COLORS["halhf"]))
        ):
            for exposure_index, exposure in enumerate((0.005, 0.02)):
                selected = [
                    row
                    for row in rows
                    if row["s2"] == spin and row["exposure"] == exposure
                ]
                x = np.array([r["chi"] for r in selected])
                replicate = np.tile(np.arange(4), len(selected) // 4)
                x *= np.exp(
                    (spin_index * 8 + exposure_index * 4 + replicate - 11.5) * 0.018
                )
                values = np.array([row[field] for row in selected])
                axis.errorbar(
                    x,
                    values,
                    yerr=np.ones(len(values)),
                    fmt=marker,
                    ms=2.3,
                    mfc=color if exposure_index else "white",
                    color=color,
                    capsize=1.1,
                    elinewidth=0.45,
                )
        axis.set(
            xscale="log",
            xlabel=r"Quantum parameter $\chi$",
            title=title,
            ylim=(-4.7, 4.7),
        )
        axis.set_xticks([0.01, 0.05, 0.2, 1, 3], ["0.01", "0.05", "0.2", "1", "3"])
        rms = np.sqrt(np.mean([row[field] ** 2 for row in rows]))
        axis.text(
            0.03,
            0.97,
            f"RMS = {rms:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    axes[0].set_ylabel("Pull")
    handles = [
        Line2D([], [], marker=m, ls="", color=c, label=rf"$S_2={s:+d}$")
        for s, m, c in (
            (-1, "v", COLORS["ilc"]),
            (0, "o", ".15"),
            (1, "^", COLORS["halhf"]),
        )
    ]
    handles += [
        Line2D([], [], marker="o", ls="", color=".3", mfc=face, label=label)
        for face, label in (("white", "Exposure 0.005"), (".3", "Exposure 0.020"))
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=8)
    figure.suptitle(
        r"Local transverse spin $S_2$ ($S_y$ in this fixed-field test)", fontsize=9
    )
    save(figure, output, "sampler_pulls")


def confirmation_figures(output, scan_data):
    """Show baseline and selected spectra with seed SEMs in a single panel."""
    for machine in ("ilc", "halhf"):
        rows = read_rows(
            scan_data / f"{machine}_selected_vs_baseline_peak_spectrum.csv"
        )
        low, high = column(rows, "low_GeV"), column(rows, "high_GeV")
        x = (low + high) / 2
        selected = (x >= 124.4) & (x <= 125.4)
        baseline, candidate = (
            column(rows, "reference_mean"),
            column(rows, "candidate_mean"),
        )
        va, vb = (
            column(rows, "reference_mean_variance"),
            column(rows, "candidate_mean_variance"),
        )
        figure, axis = plt.subplots(figsize=(6.7, 3.5), layout="constrained")
        for values, variance, color, marker, label in (
            (baseline, va, ".2", "s", "Reference optics"),
            (candidate, vb, COLORS[machine], "o", "Quarter beta; opposite dispersion"),
        ):
            error_series(
                axis,
                x[selected],
                values[selected],
                np.sqrt(variance[selected]),
                label,
                color,
                marker,
            )
        axis.set(
            title=f"{NAMES[machine]}: selected horizontal-dispersion optics",
            ylabel=r"Density [GeV$^{-1}$]",
            xlabel=r"Collision energy $W$ [GeV]",
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.12)
        save(figure, output, f"higgs_{machine}_selected_vs_baseline_peak_spectrum")


def scan_figures(output, scan_data, model_rows):
    """Separate observed widths, numerical resolution and analytical projections."""
    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.7, 6.1), layout="constrained"
    )
    rows = read_rows(scan_data / "high_dispersion.csv")
    for machine in ("ilc", "halhf"):
        selected = sorted(
            [
                row
                for row in rows
                if row["machine"] == machine and row["role"] == "scan"
            ],
            key=lambda row: float(row["dispersion_mm"]),
        )
        x = column(selected, "dispersion_mm")
        for axis, observable in zip(
            axes, ("luminosity_cm2_per_second", "observed_peak_FWHM_MeV")
        ):
            axis.plot(
                x,
                column(selected, observable + "_mean"),
                color=COLORS[machine],
                marker=MARKERS[machine],
                ms=4,
                label=NAMES[machine],
            )
            for row in selected:
                if row[observable + "_SEM"]:
                    axis.errorbar(
                        float(row["dispersion_mm"]),
                        float(row[observable + "_mean"]),
                        yerr=float(row[observable + "_SEM"]),
                        color=COLORS[machine],
                        capsize=3,
                    )
                else:
                    axis.plot(
                        float(row["dispersion_mm"]),
                        float(row[observable + "_mean"]),
                        marker=MARKERS[machine],
                        mfc="white",
                        mec=COLORS[machine],
                    )
                if observable == "observed_peak_FWHM_MeV":
                    # This full representation range is not a statistical error.
                    low = float(row["resolution_low_MeV"])
                    high = float(row["resolution_high_MeV"])
                    value = float(row[observable + "_mean"])
                    axis.errorbar(float(row["dispersion_mm"]), value,
                        yerr=[[max(0, value - low)], [max(0, high - value)]],
                        fmt="none", color=COLORS[machine], alpha=0.55,
                        capsize=6, elinewidth=1.2)
                    if json.loads(row["flags"]):
                        axis.plot(float(row["dispersion_mm"]), value,
                            marker="x", color=".2", ms=7)
    fcc = json.loads((DATA / "scans/fcc_crabwaist.json").read_text())
    selected = [row for row in fcc["rows"] if row["role"] == "single_seed_pilot"]
    axes[0].plot(
        [row["dispersion_mm"] for row in selected],
        [row["luminosity_cm2_s"] for row in selected],
        marker=MARKERS["fcc"],
        mfc="white",
        color=COLORS["fcc"],
        label="FCC-ee crab waist",
    )
    fcc_widths = {
        row["case"]: row["resolution"]
        for row in json.loads((scan_data / "fcc_resolution_summary.json").read_text())["cases"]
    }
    for row in selected:
        resolved = fcc_widths[row["name"]]
        value = resolved["fwhm_MeV"]
        axes[1].errorbar(row["dispersion_mm"], value,
            yerr=[[value - resolved["resolution_low_MeV"]],
                  [resolved["resolution_high_MeV"] - value]],
            marker=MARKERS["fcc"], mfc="white", color=COLORS["fcc"],
            capsize=6, label="FCC-ee crab waist" if row["dispersion_mm"] == 0 else None)
        if resolved["flags"]:
            axes[1].plot(row["dispersion_mm"], value, marker="x", color=".2", ms=7)
    # Show the actual matched-seed mesh comparison, not a correction to the mean.
    for row in rows:
        if row["role"] != "numerical_control":
            continue
        fine = float(row["observed_peak_FWHM_MeV_mean"])
        coarse = fine - float(row["matched_Nx_observed_peak_FWHM_MeV_difference"])
        x = float(row["dispersion_mm"])
        axes[1].plot([x, x], [fine, coarse], color=".3", lw=2)
        axes[1].plot(x, fine, marker="D", mfc="white", mec=".3", ms=5,
            label="Matched horizontal-mesh refinement" if row["machine"] == "ilc" else None)
    axes[0].set(
        yscale="log",
        ylabel=r"Luminosity [cm$^{-2}$ s$^{-1}$ per IP]",
        title="Native 125 GeV collisions, before ISR",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        yscale="log",
        ylabel="Observed peak FWHM [MeV]",
        xlabel=r"Horizontal dispersion $|D_x|$ [mm]",
    )
    axes[1].axhspan(1, 4, color=".9", label="1–4 MeV target")
    axes[1].axhline(4.1, color=".35", ls=":", label="Higgs width 4.1 MeV")
    axes[1].text(0.98, 0.96,
        "Wide caps: bin-resolution range, not seed SEM\nDiamonds: paired 2048 to 4096 horizontal-cell check",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    axes[1].legend(fontsize=8, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.15)
    save(figure, output, "three_collider_dispersion_summary")
    rows = model_rows
    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.7, 6.4), layout="constrained"
    )
    for machine in COLORS:
        selected = [row for row in rows if row["machine"] == machine]
        x = column(selected, "dispersion_mm")
        good = np.array(
            [
                str(row["first_order_diagnostic_within_5percent"]) == "True"
                for row in selected
            ]
        )
        for index, observable in enumerate(
            ("luminosity_cm2_s", "linear_gaussian_FWHM_MeV")
        ):
            y = column(selected, observable)
            axes[index].plot(
                x,
                y,
                color=COLORS[machine],
                alpha=0.3 if index else 1,
                marker=MARKERS[machine],
                markevery=45,
                ms=3,
            )
            if index:
                axes[index].plot(
                    x,
                    np.where(good, y, np.nan),
                    color=COLORS[machine],
                    marker=MARKERS[machine],
                    markevery=45,
                    ms=3,
                )
    labels = [
        "ILC: 14 mrad, ideal tilt",
        "HALHF: head-on",
        "FCC-ee: 30 mrad, no RF tilt",
    ]
    handles = [
        Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=3, label=label)
        for m, label in zip(COLORS, labels)
    ]
    axes[0].legend(handles=handles, fontsize=8)
    axes[0].set(
        ylabel=r"Rigid luminosity [cm$^{-2}$ s$^{-1}$]",
        title="Fixed-input Gaussian projections",
    )
    axes[1].set(
        ylabel="Linear Gaussian FWHM [MeV]",
        xlabel=r"Horizontal dispersion $|D_x|$ [mm]",
    )
    for axis in axes:
        axis.set(xscale="log", yscale="log")
        axis.axvline(100, color=".4", ls=":")
        axis.grid(alpha=0.15)
    axes[1].axhspan(1, 4, color=".9")
    axes[1].axhline(4.1, color=".4", lw=0.7)
    figure.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
    figure.text(
        0.08,
        0.012,
        "Omitted: radiation, disruption, RF errors, transported optics and waist/hourglass evolution.\n"
        "FCC-ee L(100 mm)/L(0): rigid model 0.753; native 0.483. Matched controls are required.\n"
        "Faint width segments: nonlinear RMS check differs by >5%; beyond 100 mm is projection only.",
        fontsize=8,
    )
    save(figure, output, "common_finite_angle_extrapolations")


def fcc_fragmentation(output):
    """Rebin native spectra in fixed groups without fitting or smoothing."""
    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.7, 5.8), layout="constrained"
    )
    for axis, name, dispersion in zip(
        axes, ("fcc_cw_d000_pilot_s1", "fcc_cw_d100_pilot_s1"), (0, 100)
    ):
        text = gzip.decompress(
            (INPUTS / "native" / (name + ".fne.gz")).read_bytes()
        ).decode()
        data = np.loadtxt(io.StringIO(text))
        low, high, weights = data[:, 0], data[:, 1], data[:, 2]
        total = weights.sum()
        fine = np.flatnonzero((high - low) < 0.00021)
        selected = (low >= 124.8) & (high <= 125.2)
        center = (low + high) / 2
        density = weights / (high - low) / total / 1000
        axis.step(
            1000 * (center[selected] - 125),
            density[selected],
            where="mid",
            color=".5",
            lw=0.6,
            label="Raw 0.2 MeV bins",
        )
        starts = np.arange(fine[0], fine[-1] + 1, 4)
        stops = np.minimum(starts + 4, fine[-1] + 1)
        gx = (low[starts] + high[stops - 1]) / 2
        grouped = np.array([weights[a:b].sum() for a, b in zip(starts, stops)])
        gd = grouped / (high[stops - 1] - low[starts]) / total / 1000
        good = (gx >= 124.8) & (gx <= 125.2)
        axis.step(
            1000 * (gx[good] - 125),
            gd[good],
            where="mid",
            lw=1,
            color=COLORS["ilc"],
            label="4-bin groups, fixed phase",
        )
        axis.axhline(
            density[fine].max() / 2,
            ls="--",
            color=COLORS["halhf"],
            lw=0.8,
            label="Raw maximum / 2",
        )
        axis.set(
            title=f"FCC-ee: |D| = {dispersion} mm; one seed",
            ylabel=r"Density [MeV$^{-1}$]",
            xlim=(-200, 200),
        )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel(r"$W-125$ GeV [MeV]")
    save(figure, output, "fcc_peak_fragmentation")


def halhf_pair_populations(output):
    """Compare occupied angular supports, without equating plot ink to yields."""
    samples = np.load(INPUTS / "halhf_pair_support.npz")
    figure, axes = plt.subplots(
        3, 2, sharex="col", sharey="col", figsize=(6.7, 7.5), layout="constrained"
    )
    species = (("electron", COLORS["halhf"], "^"), ("positron", COLORS["ilc"], "s"))
    for index, direction in enumerate(("forward", "backward")):
        rows = read_rows(
            BENCHMARKS / "reference" / f"halhf_pairs_250_{direction}_ink.csv"
        )
        x = (column(rows, "log10_theta_low") + column(rows, "log10_theta_high")) / 2
        y = (column(rows, "log10_pT_low") + column(rows, "log10_pT_high")) / 2
        for name, color, marker in species:
            key = (
                "electron_red_ink_fraction"
                if name == "electron"
                else "positron_blue_ink_fraction"
            )
            occupied = column(rows, key) > 0
            axes[0, index].scatter(
                x[occupied],
                y[occupied],
                marker=marker,
                s=5,
                color=color,
                alpha=0.65,
                linewidths=0,
            )
        for row_index, population in enumerate(("mixed", "incoherent"), start=1):
            xedges = samples[f"{population}_{direction}_xedges"]
            yedges = samples[f"{population}_{direction}_yedges"]
            gx, gy = np.meshgrid(
                (xedges[:-1] + xedges[1:]) / 2,
                (yedges[:-1] + yedges[1:]) / 2,
                indexing="ij",
            )
            for name, color, marker in species:
                occupied = samples[f"{population}_{direction}_{name}"] > 0
                axes[row_index, index].scatter(
                    gx[occupied],
                    gy[occupied],
                    marker=marker,
                    s=5,
                    color=color,
                    alpha=0.65,
                    linewidths=0,
                )
        for row_index, label in enumerate(
            (
                "Published occupied pixels",
                "Historical FeeBL: mixed backgrounds",
                "Historical FeeBL: pair-only control",
            )
        ):
            axis = axes[row_index, index]
            if row_index == 0:
                axis.set_title(label + "; " + direction)
            else:
                axis.set_title(
                    label + (r"; $z>0$" if direction == "forward" else r"; $z<0$")
                )
            axis.grid(alpha=0.12)
            axis.set_xlim(
                column(rows, "log10_theta_low").min(),
                column(rows, "log10_theta_high").max(),
            )
            axis.set_ylim(
                column(rows, "log10_pT_low").min(), column(rows, "log10_pT_high").max()
            )
            if index == 0:
                axis.set_ylabel(r"$\log_{10}(p_T/\mathrm{GeV})$")
        axes[-1, index].set_xlabel(
            r"$\log_{10}(\theta/\mathrm{rad})$; nearest beam direction"
        )
    handles = [
        Line2D([], [], ls="", marker=marker, color=color, ms=3, label=name)
        for name, color, marker in species
    ]
    axes[2, 0].legend(handles=handles, loc="lower left", fontsize=8)
    figure.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    figure.text(
        0.09,
        0.01,
        "Historical transport: mixed sample includes Compton recoils; repaired-kernel result pending.\n"
        "Topology only; no common yield normalization. Final-position z selection; theta folds to nearest beam.",
        fontsize=8,
    )
    save(figure, output, "halhf_pair_populations")


def imported_spin_figure(output, numerical_data):
    """Recompute matched-event spin statistics directly from native gzip records."""
    producer = BENCHMARKS.parent / "tools/analyze_imported_spin.py"
    specification = importlib.util.spec_from_file_location(
        "analyze_imported_spin", producer
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    files = {
        "field_on": "field_on.json.gz",
        "field_off": "field_off.json.gz",
        "refined": "time_refined.json.gz",
    }
    payloads, provenance = {}, {}
    for label, filename in files.items():
        payloads[label], provenance[label] = module.read_input(
            DATA / "imported_spin" / filename
        )
    result, arrays = module.analyze(payloads)
    result["inputs"] = provenance
    result["producer_sha256"] = hashlib.sha256(producer.read_bytes()).hexdigest()
    module.plot_results(
        result, arrays, output, figure_width=6.7, stem="imported_spin_transport"
    )
    (numerical_data / "imported_spin_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qed-reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument(
        "--include-imported-spin",
        action="store_true",
        help="Also regenerate the separately retained imported-particle spin study",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    qed_tables = args.output / "qed_products/tables"
    export_products("tables", args.qed_reduction, qed_tables)
    figures = args.output / "figures"
    figures.mkdir()
    scan_data = args.output / "numerical_data"
    regenerate_studies(scan_data)
    model_rows = generate_gaussian_overlap()
    write_rows(scan_data / "gaussian_overlap_baselines.csv", model_rows)
    sampler = evaluate_sampler(INPUTS / "sampler_records")
    (scan_data / "sampler_oracle.json").write_text(json.dumps(sampler, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    collider_figures(args.comparisons, figures, include_focusing_pairs=False)
    sampler_figure(figures, sampler)
    generate_polarization_figure(figures)
    confirmation_figures(figures, scan_data)
    scan_figures(figures, scan_data, model_rows)
    fcc_fragmentation(figures)
    qed_figures = args.output / "qed_products/figures"
    export_products("figures", args.qed_reduction, qed_figures)
    qed_provenance = copy_figures(qed_figures, figures, args.qed_reduction)
    imported_spin = (
        imported_spin_figure(figures, scan_data) if args.include_imported_spin else None
    )
    build_tables(
        args.output / "tables",
        scan_data,
        imported_spin,
        include_imported_spin=args.include_imported_spin,
        qed_reduction=args.qed_reduction,
        qed_tables=qed_tables,
    )
    metric_directory = scan_data / "spectrum_comparison_metrics"
    generate_comparison_metrics(args.comparisons, metric_directory)
    for path in metric_directory.iterdir():
        shutil.copyfile(path, args.output / "tables" / path.name)
    traceability_path = args.output / "tables/table_traceability.json"
    traceability = json.loads(traceability_path.read_text())
    traceability["spectrum_comparison_metrics.tex"] = {
        "producer": "benchmarks/analysis/comparison_metrics.py",
        "source": "supplied comparisons/*_pull.csv; exact conditional piecewise-density CDF distances",
    }
    traceability_path.write_text(json.dumps(traceability, indent=2) + "\n")
    receipt = {
        "producer": Path(__file__).name,
        "figures": [
            {
                "file": path.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(figures.glob("*.pdf"))
        ],
        "comparison_sources": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(args.comparisons.glob("*"))
            if path.is_file()
        ],
        "display_reductions": [
            {
                "file": path.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted((figures / "display_data").glob("*"))
            if path.is_file()
        ],
        "tables": [
            {
                "file": path.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted((args.output / "tables").iterdir())
            if path.is_file()
        ],
        "replay_scope": "Verified focusing/pair ensembles; historical HALHF luminosity spectra remain separate; retained response fits, Bessel quadrature and Gaussian overlap are reproduced without simulation",
        "qed_ensemble": {**qed_provenance, "table_receipt_sha256": hashlib.sha256((qed_tables/"table_receipt.json").read_bytes()).hexdigest()},
        "output_sha256": {path.relative_to(args.output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(args.output.rglob("*")) if path.is_file()},
        "imported_spin_included": args.include_imported_spin,
        "imported_spin_native_inputs": imported_spin["inputs"]
        if imported_spin is not None
        else [],
        "minimum_nominal_font_points": 6.5,
        "figure_width_inches": 6.7,
        "statistical_pull_bars": "+/-1 with endcaps; bands +/-1 and +/-2 are visual guides",
    }
    (args.output / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
