#!/usr/bin/env python3
"""Render the three monochromatization figures from compact numerical inputs.

This is a standalone renderer: it needs NumPy, Matplotlib and figure_data.json,
but neither FeeBL, the original repository nor the native collision files.
No statistical or physical model is fitted during rendering.
"""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"ilc": "#0072B2", "halhf": "#D55E00", "fcc": "#009E73"}
NAMES = {"ilc": "ILC/LCF", "halhf": "HALHF", "fcc": "FCC-ee crab waist"}
PEAK_NAMES = {"ilc": "ILC", "halhf": "HALHF"}
METADATA = {"CreationDate": None, "ModDate": None}


def save_figure(fig, output, name):
    """Keep vector PDFs and matching raster previews together."""
    fig.savefig(output / (name + ".pdf"), metadata=METADATA)
    fig.savefig(output / (name + ".png"), dpi=180)
    plt.close(fig)


def render_confirmation(data, output):
    """Show broad and narrow shapes at unit height without shifting their energies."""
    for panel in data["confirmation"]:
        machine = panel["machine"]
        width = panel["narrow_width_MeV"]
        zoom_half_MeV = max(1.0, 3 * width)
        nominal = panel["nominal_W_GeV"]
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.0), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.15, 1]})
        for curve in panel["curves"]:
            x = np.asarray(curve["x_GeV"])
            edges = np.asarray(curve["edges_GeV"])
            y = np.asarray(curve["y"])
            if not np.isclose(y.max(), 1.0, rtol=0, atol=1e-12):
                raise ValueError("Display curve does not have unit maximum")
            error = None if curve["error"] is None else np.asarray(curve["error"])
            n = curve["seeds"]
            if curve["role"] == "narrow":
                label = f"High dispersion: {panel['dispersion_mm']:.1f} mm ({n} " + ("seed)" if n == 1 else "seeds)")
            else:
                label = f"{curve['label']} ({n} seeds)"
            for index, axis in enumerate(axes):
                xx = x if index == 0 else (x - nominal) * 1000
                xe = edges if index == 0 else (edges - nominal) * 1000
                # Draw complete bin edges. Cropping by bin centers would erase
                # a broad bin that spans the entire narrow zoom window.
                axis.stairs(y, xe, color=curve["color"], linestyle=curve["linestyle"],
                            linewidth=1.05, baseline=None, label=label)
                low, high = (124.4, 125.4) if index == 0 else (-zoom_half_MeV, zoom_half_MeV)
                indices = np.flatnonzero((xx >= low) & (xx <= high))
                if error is not None and len(indices):
                    sample = indices[::max(1, int(len(indices) / 30))]
                    axis.errorbar(xx[sample], y[sample], yerr=error[sample], fmt="none",
                                  color=curve["color"], capsize=1.7, elinewidth=.6, alpha=.7)
        axes[0].set(xlabel=r"Collision energy $W$ [GeV]", ylabel="Spectrum / maximum bin height",
                    xlim=(124.4, 125.4))
        axes[1].set(xlabel=r"$W-125\,\mathrm{GeV}$ [MeV]", xlim=(-zoom_half_MeV, zoom_half_MeV))
        for axis in axes:
            axis.set_ylim(-.03, 1.11)
            axis.grid(alpha=.14)
            axis.tick_params(labelsize=9)
        axes[1].text(.97, .96, f"Narrow FWHM: {width:.3g} MeV", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=8.5,
                     bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
        if panel["narrow_resolution_flags"]:
            axes[1].text(.97, .86, "Histogram-resolution advisory", transform=axes[1].transAxes,
                         ha="right", va="top", fontsize=7.8)
        fig.suptitle(rf"{PEAK_NAMES[machine]} at $W_0 = {nominal:g}\,\mathrm{{GeV}}$", fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", fontsize=8.5)
        save_figure(fig, output, f"higgs_{machine}_selected_vs_baseline_peak_spectrum")


def render_scan(data, output):
    """Connect scan points and show seed errors and histogram-grouping intervals."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.6), layout="constrained")
    for machine in ("ilc", "halhf", "fcc"):
        points = sorted(data["scan"]["machines"][machine], key=lambda point: point["dispersion_mm"])
        marker = {"ilc": "o", "halhf": "s", "fcc": "^"}[machine]
        for index, axis in enumerate(axes):
            axis.plot([p["dispersion_mm"] for p in points], [p["values"][index] for p in points],
                      color=COLORS[machine], linewidth=1.1, label=NAMES[machine])
            for point in points:
                x, y = point["dispersion_mm"], point["values"][index]
                axis.errorbar(x, y, yerr=point["errors"][index], fmt=marker, color=COLORS[machine],
                              mfc=COLORS[machine], capsize=3, ms=2.4)
                if index == 1:
                    lower = max(0, y-point["resolution_low_MeV"])
                    upper = max(0, point["resolution_high_MeV"]-y)
                    axis.errorbar(x, y, yerr=[[lower], [upper]], fmt="none", color=COLORS[machine],
                                  alpha=.45, capsize=6)
                    if point["flags"]:
                        axis.plot(x, y, "x", color=".15", ms=5)
    axes[1].axhspan(1, 4, color=".92", label="1-4 MeV")
    axes[1].axhline(4.1, color=".3", ls=":", lw=1.1, label="Higgs width: 4.1 MeV")
    axes[0].set(ylabel=r"Luminosity [cm$^{-2}$ s$^{-1}$ per IP]", title="125 GeV collisions, before ISR")
    axes[1].set(ylabel="Observed peak FWHM [MeV]", xlabel=r"Horizontal dispersion $|D_x|$ [mm]")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=.14)
        axis.legend(fontsize=8)
        axis.tick_params(labelsize=9)
    save_figure(fig, output, "three_collider_dispersion_summary")


def render(data, output):
    """Render every available confirmation panel and the common scan figure."""
    if data["schema"] != "feebl.narrow-peak-figure-data/1":
        raise ValueError("Unsupported figure-data schema")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    render_confirmation(data, output)
    render_scan(data, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    render(data, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

