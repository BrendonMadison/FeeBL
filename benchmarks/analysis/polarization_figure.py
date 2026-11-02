"""Plot the measured ILC scenarios without inventing uncovered collider points."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparison_statistics import BENCHMARKS, read_rows, write_rows


def generate(output):
    """Separate three-seed per-beam helicity losses from one-seed axis products."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = BENCHMARKS / "example_output"
    cut_path, scenario_path = (
        data / "binned_energy_cuts.csv",
        data / "intercode_comparison.csv",
    )
    cuts, scenarios = read_rows(cut_path), read_rows(scenario_path)
    runs = {f"particles_40k_seed{seed}_st1" for seed in (1, 2, 3)}
    selections = (
        ("all", "Inclusive"),
        ("W_ge_0p98_W0", r"$W\geq0.98W_0$"),
        ("W_ge_0p99_W0", r"$W\geq0.99W_0$"),
    )
    summaries = []
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 3.8), layout="constrained")
    for beam, marker, color, offset in (
        (1, "o", "#0072B2", -0.07),
        (2, "s", "#D55E00", 0.07),
    ):
        means, sems = [], []
        for selection, label in selections:
            rows = [
                row
                for row in cuts
                if row["run"] in runs and row["selection"] == selection
            ]
            if len(rows) != 3:
                raise ValueError(
                    "Each helicity selection requires three complete seeds"
                )
            values = np.array([float(row[f"beam{beam}_loss_pp"]) for row in rows])
            mean, sem = values.mean(), values.std(ddof=1) / np.sqrt(3)
            means.append(mean)
            sems.append(sem)
            summaries.append(
                {
                    "quantity": "per_beam_local_helicity_loss_pp",
                    "scenario": "ILC500 beta_x21mm",
                    "beam": beam,
                    "selection": selection,
                    "mean_pp": mean,
                    "seed_sem_pp": sem,
                    "seeds": 3,
                    "definition": "100*(1-<h_i>_L), initially h_i=1; independent complete-run SEM",
                }
            )
        axes[0].errorbar(
            np.arange(3) + offset,
            means,
            yerr=sems,
            fmt=marker + "-",
            ms=4,
            capsize=3,
            lw=0.8,
            color=color,
            label="Electron" if beam == 1 else "Positron",
        )
    axes[0].set_xticks(np.arange(3), [label for _, label in selections], rotation=18)
    axes[0].set(
        title="ILC 500 GeV: per-beam helicity", ylabel="Luminosity-weighted loss [%]"
    )
    axes[0].legend(fontsize=8)
    for field, offset, color, marker, label in (
        ("GPpp_pp", -0.18, ".25", "s", "GUINEA-PIG++"),
        ("CAIN_pp", 0.0, "#009E73", "^", "CAIN"),
        ("local_axis_loss_pp", 0.18, "#0072B2", "o", "FeeBL"),
    ):
        axes[1].plot(
            np.arange(len(scenarios)) + offset,
            [float(row[field]) for row in scenarios],
            ls="",
            marker=marker,
            color=color,
            ms=4,
            mfc="white",
            label=label,
        )
    for row in scenarios:
        summaries.append(
            {
                "quantity": "correlated_fixed_axis_loss_pp",
                "scenario": "ILC500 " + row["case"],
                "beam": "combined",
                "selection": "inclusive",
                "mean_pp": float(row["local_axis_loss_pp"]),
                "seed_sem_pp": "",
                "seeds": 1,
                "definition": "100*(1-sqrt(-<s_z1*s_z2>_L)), initially s_z1=+1,s_z2=-1; no seed SEM for singleton",
            }
        )
    axes[1].set_xticks(
        np.arange(len(scenarios)),
        [row["case"].replace("_", " ").title() for row in scenarios],
        rotation=18,
    )
    axes[1].set(
        title="ILC 500 GeV: fixed-axis product",
        ylabel="Correlated fixed-axis loss [%]",
    )
    axes[1].legend(fontsize=7.5)
    axes[1].text(
        0.03,
        0.06,
        "Single-seed FeeBL points;\nreference errors not supplied",
        fontsize=7.5,
        transform=axes[1].transAxes,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
        axis.set_ylim(bottom=0)
    figure.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    figure.text(
        0.09,
        0.02,
        "Historical ILC results only. HALHF depolarization coverage pending.",
        fontsize=8,
    )
    figure.savefig(
        output / "depolarization_scenarios.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(output / "depolarization_scenarios.png", dpi=160)
    plt.close(figure)
    write_rows(output / "depolarization_scenarios.csv", summaries)
    receipt = {
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (cut_path, scenario_path)
        },
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": {
            "ILC": "Historical recorded scenarios",
            "HALHF": "No qualified per-beam ensemble in retained dataset",
        },
        "uncertainty": "Left: three-seed SEM, numerical and binned-output rounding errors excluded. Right: no fabricated seed or reference errors. Values from distinct polarization definitions are on separate axes.",
    }
    (output / "depolarization_scenarios.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
