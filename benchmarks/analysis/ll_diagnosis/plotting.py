"""Publication figure for the measured nine-arm pair-production controls."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from definitions import ARMS, CHANNELS


def plot_controls(summary, output, preview_label=""):
    """Show all arms without implying that changes form an additive sequence."""
    output = Path(output)
    lookup = {(row["arm"], row["channel"]): row for row in summary["records"]}
    styles = {0: ("#555555", "s"), 1: ("#0072B2", "o"), 2: ("#D55E00", "D")}
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(6.7, 5.3), sharey=True, layout="constrained")
        for panel, channel in zip(axes, CHANNELS):
            for index, (arm, label, model, _, _) in enumerate(ARMS):
                row = lookup[arm, channel]
                color, marker = styles[model]
                panel.errorbar(row["cross_section_mb"], index, xerr=row["sem_mb"],
                    fmt=marker, ms=4, color=color, elinewidth=.8, capsize=2.5, capthick=.8)
            panel.set_title(channel)
            panel.set_xlabel("Effective daughter\ncross section [mb]")
            panel.grid(axis="x", alpha=.20)
            panel.set_axisbelow(True)
            panel.margins(x=.15)
            panel.set_ylim(8.6, -.65)
            for split in (2.5, 4.5, 5.5, 7.5):
                panel.axhline(split, color=".85", lw=.5, zorder=0)
        axes[0].set_yticks(np.arange(len(ARMS)), [label for _, label, *_ in ARMS])
        handles = [Line2D([], [], color=styles[model][0], marker=styles[model][1],
                   ls="", ms=4, label=label) for model, label in
                   ((0, "Legacy EPA (Born normalization corrected)"), (1, "Finite-mass EPA"), (2, "Full QED amplitude"))]
        figure.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=8)
        figure.suptitle((preview_label+"\n" if preview_label else "")+
                         "Four matched collisions per arm; sampling SEM, 3 variance degrees of freedom\n"
                         "Unpolarized photons; 5 MeV daughter production cut", fontsize=8.5)
        figure.savefig(output/"pair_process_controls.pdf", metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(output/"pair_process_controls.png", dpi=180)
        plt.close(figure)
