"""Render central pair-production values without altering the numerical study."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]
DEFAULT_REDUCTION = SOURCE / "benchmarks/example_output/ll_diagnosis/reduction"


def digest(path):
    """Identify the exact preserved input or generated display file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_summary(reduction):
    """Require the original numerical summary and its complete producer identity."""
    receipt = json.loads((reduction / "analysis_receipt.json").read_text())
    if receipt["status"] != "COMPLETE_VERIFIED_REDUCTION":
        raise ValueError("A complete verified reduction is required")
    for name, expected in receipt["producer_sha256"].items():
        path = SOURCE / "benchmarks/analysis/ll_diagnosis" / name
        if path.name != name or digest(path) != expected:
            raise ValueError("Numerical producer changed: " + name)
    if digest(reduction / "control_summary.json") != receipt["output_sha256"]["control_summary.json"]:
        raise ValueError("Numerical summary changed")
    summary = json.loads((reduction / "control_summary.json").read_text())
    if summary["independent_collisions_per_arm"] != 4:
        raise ValueError("Unexpected collision count")
    return summary


def render(reduction, output):
    """Display the original 27 ratios; retain their SEMs in the display receipt."""
    summary = verified_summary(reduction)
    annotations = json.loads((HERE / "reference_annotations.json").read_text())
    arms = annotations["arms"]
    channels = ("BW", "BH", "LL")
    lookup = {(row["arm"], row["channel"]): row for row in summary["records"]}
    expected = {(arm["arm"], channel) for arm in arms for channel in channels}
    if len(summary["records"]) != 27 or set(lookup) != expected:
        raise ValueError("The complete nine-by-three control set is required")
    output.mkdir(parents=True, exist_ok=False)
    styles = {0: ("#555555", "s"), 1: ("#0072B2", "o"), 2: ("#D55E00", "D")}
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(6.7, 5.3), sharey=True, layout="constrained")
        for panel, channel in zip(axes, channels):
            for index, arm in enumerate(arms):
                row = lookup[arm["arm"], channel]
                color, marker = styles[arm["model"]]
                panel.plot(row["cross_section_mb"], index, marker=marker, ms=4,
                           color=color, linestyle="none")
            panel.set_title(channel)
            panel.set_xlabel("Effective daughter\ncross section [mb]")
            panel.grid(axis="x", alpha=.20)
            panel.set_axisbelow(True)
            panel.margins(x=.15)
            panel.set_ylim(8.6, -.65)
            for split in (2.5, 4.5, 5.5, 7.5):
                panel.axhline(split, color=".85", lw=.5, zorder=0)
        labels = [arm["label"] + (r"$^{" + arm["superscript"] + "}$" if arm["superscript"] else "")
                  for arm in arms]
        axes[0].set_yticks(np.arange(len(arms)), labels)
        handles = [Line2D([], [], color=styles[model][0], marker=styles[model][1],
                         ls="", ms=4, label=label) for model, label in (
            (0, "Legacy EPA (Born normalization corrected)"),
            (1, "Finite-mass EPA"), (2, "Full QED amplitude"))]
        handles.append(Line2D([], [], ls="", label=r"$^{*}$ GUINEA-PIG; $^{\dagger}$ CAIN: documented option"))
        figure.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
        figure.suptitle("Central values; four collision runs per configuration\n"
                         "Unpolarized photons; 5 MeV daughter production cut", fontsize=8.5)
        # Artist coordinates must reproduce the saved values without scaling or resampling.
        for panel, channel in zip(axes, channels):
            points = [line for line in panel.lines if line.get_marker() in ("s", "o", "D")]
            np.testing.assert_array_equal([line.get_xdata()[0] for line in points],
                [lookup[arm["arm"], channel]["cross_section_mb"] for arm in arms])
            if panel.containers:
                raise ValueError("This central-value display must not contain error-bar artists")
        figure.savefig(output / "pair_process_controls.pdf", metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(output / "pair_process_controls.png", dpi=180)
        plt.close(figure)
    records = [{key: row[key] for key in ("arm", "channel", "cross_section_mb", "sem_mb")}
               for row in summary["records"]]
    receipt = dict(schema="feebl.ll-control-display/1", status="VERIFIED_CENTRAL_VALUE_DISPLAY",
        input_reduction={name: digest(reduction / name) for name in ("analysis_receipt.json", "control_summary.json")},
        producer_sha256={name: digest(HERE / name) for name in ("render.py", "reference_annotations.json")},
        output_sha256={name: digest(output / name) for name in ("pair_process_controls.pdf", "pair_process_controls.png")},
        records=records, four_collision_runs_per_configuration=4,
        statistical_error_bars_drawn=False, particle_records_discarded=0,
        scope="Presentation only. All central values, SEMs, covariances and native records remain unchanged. Symbols describe antecedents of named options, not identical full configurations.")
    (output / "display_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("Rendered all 27 original central values; no records discarded and no uncertainty estimates altered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.reduction, args.output)
