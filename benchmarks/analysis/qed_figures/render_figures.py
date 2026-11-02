"""Render the independent-collision comparisons with the paper's plot style."""
import importlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from reduce_figures import digest, read_csv, read_json, save_json


def data(rows, name):
    return np.array([float(r[name]) for r in rows])


def draw_comparison(plot, top, bottom, rows, x, factor, ref_label,
                    ref_color=".2", ref_marker="s", draw_simulation=True,
                    simulation_label="FeeBL"):
    if draw_simulation:
        plot.error_series(top, x, data(rows, "simulation")*factor,
            np.sqrt(data(rows, "simulation_variance"))*factor,
            simulation_label, "#0072B2", "o")
    plot.error_series(top, x, data(rows, "reference")*factor,
        np.sqrt(data(rows, "reference_variance"))*factor,
        ref_label, ref_color, ref_marker)
    native_rows = [dict(r, simulation_mass=r["simulation"], reference_mass=r["reference"])
                   for r in rows]
    plot.draw_pulls(bottom, native_rows, x, ref_color, ref_marker, ref_label)


def render_lep(reduction, plot, output=None):
    output = output or reduction
    fig, top, bottom = plot.comparison_axes()
    for index, suffix in enumerate(("vs_phi", "numerical_vs_phi")):
        rows = read_csv(reduction/f"lep_final_focusing_{suffix}_pull.csv")
        draw_comparison(plot, top[0], bottom[0], rows, data(rows, "phi_rad"), 1,
            "Published GP" if index == 0 else "Published field integral",
            ".2" if index == 0 else "#D55E00", "s" if index == 0 else "^",
            draw_simulation=index == 0, simulation_label="FeeBL: 10 fields")
    rows = read_csv(reduction/"lep_focusing_distribution_pull.csv")
    low, high = data(rows, "bin_low"), data(rows, "bin_high")
    draw_comparison(plot, top[1], bottom[1], rows, (low+high)/2, 1/(high-low),
        "Published curve", simulation_label="FeeBL: 10 fields")
    top[0].set(title="LEP: initial azimuth", ylabel=r"Mean focusing [$\mu$rad]")
    top[1].set(title="LEP: focusing distribution", ylabel=r"Conditional density [$\mu$rad$^{-1}$]")
    bottom[0].set_xlabel(r"Initial azimuth $\phi$ [rad]")
    bottom[1].set_xlabel(r"Focusing $\theta_i-\theta_f$ [$\mu$rad]")
    for axis in top:
        axis.legend(fontsize=7, loc="best")
        axis.grid(alpha=.15)
    fig.suptitle("Independent-field mean and SEM; 9 variance degrees of freedom", fontsize=9)
    plot.save(fig, output, "figure2_lep_focusing")


def render_pairs(reduction, plot, output=None):
    output = output or reduction
    fig, top, bottom = plot.comparison_axes()
    x = np.arange(3)
    for index, model in enumerate((1, 2)):
        for ref_index, code in enumerate(("gp", "cain")):
            rows = read_csv(reduction/f"ilc_model{model}_{code}_pull.csv")
            color, marker = (".2", "s") if code == "gp" else ("#D55E00", "^")
            label = "GUINEA-PIG" if code == "gp" else "CAIN"
            plot.error_series(top[index], x+(ref_index-1)*.16, data(rows, "reference"),
                np.sqrt(data(rows, "reference_variance")), label, color, marker, log=True)
            native_rows = [dict(r, simulation_count=r["simulation"], reference_count=r["reference"])
                           for r in rows]
            plot.draw_pulls(bottom[index], native_rows, x+(ref_index-.5)*.08, color, marker, label)
        plot.error_series(top[index], x+.16, data(rows, "simulation"),
            np.sqrt(data(rows, "simulation_variance")), "FeeBL: 12 seeds", "#0072B2", "o", log=True)
        top[index].set(title=f"ILC 500 GeV: pair model {model}", ylabel=r"Effective daughter cross section [mb]")
        top[index].set_ylim(.7, 60)
        top[index].legend(fontsize=6.5, loc="best")
        top[index].grid(alpha=.15)
        bottom[index].set_xticks(x, ["BW", "BH", "LL"])
        bottom[index].legend(fontsize=6.5, loc="best")
    fig.suptitle("5 MeV daughter production cut; reference paired-count uncertainty scenario", fontsize=8.5)
    plot.save(fig, output, "figure3_ilc_pair_yields")


def render_angular(reduction, plot, output=None):
    """Show the reference support and occupancy frequency of the independent seeds."""
    output = output or reduction
    arrays = np.load(reduction/"halhf_angular_ensemble.npz", allow_pickle=False)
    fig, axes = plt.subplots(3, 2, figsize=(6.7, 7.3), sharex="col", sharey="col", layout="constrained")
    species_styles = (("electron", "#0072B2", "^"), ("positron", "#D55E00", "s"))
    for column, side in enumerate(("forward", "backward")):
        x, y = arrays[side+"_xedges"], arrays[side+"_yedges"]
        xx, yy = np.meshgrid((x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2, indexing="ij")
        for row in range(3):
            axis = axes[row, column]
            for species, color, marker in species_styles:
                fraction = (arrays[f"{side}_{species}_reference_support"].astype(float) if row == 0 else
                            arrays[f"model{row}_{side}_{species}_occupied_seed_fraction"])
                selected = fraction > 0
                alpha = np.full(int(selected.sum()), .7) if row == 0 else fraction[selected]
                axis.scatter(xx[selected], yy[selected], marker=marker, s=5, color=color,
                             alpha=alpha, linewidths=0)
            title = "Published occupied pixels" if row == 0 else f"FeeBL model {row}: 12 seeds"
            axis.set_title(title+(r"; $z>0$" if column == 0 else r"; $z<0$"), fontsize=9)
            axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
            axis.grid(alpha=.12)
            if column == 0:
                axis.set_ylabel(r"$\log_{10}(p_T/\mathrm{GeV})$")
        axes[-1, column].set_xlabel(r"$\log_{10}(\theta/\mathrm{rad})$; nearest beam")
    handles = [Line2D([], [], ls="", marker=marker, color=color, ms=3, label=name)
               for name, color, marker in species_styles]
    axes[-1, 0].legend(handles=handles, loc="lower left", fontsize=8)
    # Constrained-layout rectangles use left, bottom, width and height.
    # Reserve the footer inside the canvas rather than shifting the top row out.
    fig.get_layout_engine().set(rect=(0, .065, 1, .935))
    fig.text(.08, .012, "Pure incoherent daughters; final-position z selection; no final-energy recut.\n"
             "FeeBL opacity: fraction of seeds occupying a cell. Support only; image pixels are not yields.", fontsize=8)
    plot.save(fig, output, "figure6_halhf_angular_support")


def render(reduction, source, destination=None):
    sys.path.insert(0, str(Path(source)/"benchmarks/analysis"))
    plot = importlib.import_module("regenerate_paper")
    output = Path(destination) if destination is not None else Path(reduction)
    if destination is not None:
        output.mkdir(parents=True, exist_ok=False)
    render_lep(Path(reduction), plot, output)
    render_pairs(Path(reduction), plot, output)
    render_angular(Path(reduction), plot, output)


def verify_reduction(reduction, source):
    """Require unchanged numerical products and source helpers before replotting."""
    receipt_path = reduction/"analysis_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "REDUCED_NOT_PHYSICS_QUALIFIED":
        raise ValueError("A completed reduction receipt is required")
    expected_files = receipt.get("output_sha256", {})
    if not expected_files:
        raise ValueError("Reduction does not identify its numerical outputs")
    for name, expected in expected_files.items():
        if Path(name).name != name or digest(reduction/name) != expected:
            raise ValueError("Reduction artifact changed: "+name)
    for name, expected in receipt["reused_analysis_sha256"].items():
        if Path(name).name != name or digest(source/"benchmarks/analysis"/name) != expected:
            raise ValueError("Reused plotting/analysis source changed: "+name)
    for name, expected in receipt["producer_sha256"].items():
        if Path(name).name != name or digest(Path(__file__).parent/name) != expected:
            raise ValueError("Figure producer source changed: "+name)
    if receipt["source_native_identity"] != (source/"SOURCE_REVISION").read_text().strip():
        raise ValueError("Figure source identity differs from the admitted reduction")
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_reduction(args.reduction, args.source)
    render(args.reduction, args.source, args.output)
    save_json(args.output/"rerender_receipt.json", dict(
        status="RENDERED_FROM_VERIFIED_REDUCTION",
        analysis_receipt_sha256=digest(args.reduction/"analysis_receipt.json"),
        input_artifact_sha256=receipt["output_sha256"],
        reused_analysis_sha256=receipt["reused_analysis_sha256"],
        producer_sha256=receipt["producer_sha256"],
        output_sha256={p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
