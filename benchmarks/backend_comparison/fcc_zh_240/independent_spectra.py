"""Compare disjoint CPU and GPU seed ensembles from the saved backend study.

Odd seeds select four CPU collisions; even seeds select four GPU collisions.
The selection is fixed by seed parity, not by observed agreement. This
separate display does not replace the full paired-closure or timing results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from paired_spectra import (checked_file, mean_covariance, read_spectrum,
                            remap_mass, validate_inventory)


CPU_SEEDS = (72942001, 72942003, 72942005, 72942007)
GPU_SEEDS = (72942002, 72942004, 72942006, 72942008)


def select_independent_rows(document):
    """Require the complete saved study before selecting two disjoint ensembles."""
    validate_inventory(document)
    if set(document["seeds"]) != set(CPU_SEEDS + GPU_SEEDS):
        raise ValueError("Independent comparison requires the eight declared seeds")
    selected = {"cpu": [], "gpu": []}
    for backend, seeds in (("cpu", CPU_SEEDS), ("gpu", GPU_SEEDS)):
        selected[backend] = [row for row in document["runs"]
                             if row["variant"] == backend and row["seed"] in seeds]
        selected[backend].sort(key=lambda row: row["seed"])
        if tuple(row["seed"] for row in selected[backend]) != seeds:
            raise ValueError("Incomplete independent seed selection")
    if {row["seed"] for row in selected["cpu"]} & {row["seed"] for row in selected["gpu"]}:
        raise ValueError("Independent arms cannot share a seed")
    return selected


def independent_comparison(cpu, gpu):
    """Combine covariance of independent means without paired subtraction.

    Each row is one complete collision and each column is a fixed display
    bin. The output density units are inherited from the input arrays.
    """
    cpu, gpu = np.asarray(cpu, dtype=float), np.asarray(gpu, dtype=float)
    if cpu.ndim != 2 or gpu.ndim != 2 or cpu.shape[1] != gpu.shape[1]:
        raise ValueError("Independent arms require matching bin dimensions")
    cpu_mean, cpu_cov = mean_covariance(cpu)
    gpu_mean, gpu_cov = mean_covariance(gpu)
    difference, covariance = gpu_mean - cpu_mean, cpu_cov + gpu_cov
    cpu_var, gpu_var = np.diag(cpu_cov), np.diag(gpu_cov)
    variance = np.maximum(0, np.diag(covariance))
    uncertainty = np.sqrt(variance)
    pulls, states, welch_df = [], [], []
    for index, sigma in enumerate(uncertainty):
        if sigma > 0:
            pulls.append(float(difference[index] / sigma))
            states.append("estimated_independent_means")
            denominator = cpu_var[index] ** 2 / (len(cpu)-1) + gpu_var[index] ** 2 / (len(gpu)-1)
            welch_df.append(float(variance[index] ** 2 / denominator))
        elif difference[index] == 0:
            pulls.append(0.0)
            states.append("zero_variance_identity_convention")
            welch_df.append(0.0)
        else:
            pulls.append("Infinity" if difference[index] > 0 else "-Infinity")
            states.append("nonzero_difference_without_observed_scatter")
            welch_df.append(0.0)
    return dict(cpu_replicas=len(cpu), gpu_replicas=len(gpu),
                cpu_mean=cpu_mean.tolist(), gpu_mean=gpu_mean.tolist(),
                cpu_covariance_of_mean=cpu_cov.tolist(),
                gpu_covariance_of_mean=gpu_cov.tolist(),
                difference_gpu_minus_cpu=difference.tolist(),
                difference_covariance_of_mean=covariance.tolist(),
                combined_sem=uncertainty.tolist(), pull=pulls, pull_state=states,
                pull_error=[1.0] * len(pulls), marginal_welch_degrees_of_freedom=welch_df,
                covariance_rank=int(np.linalg.matrix_rank(covariance)),
                method="Sample covariance of each four-collision mean, added with zero cross-covariance because the seed sets are disjoint.")


def plot_comparison(panels, output):
    """Display the selected energy window with CPU-minus-GPU unit-error pulls.

    The saved reduction retains both windows and its GPU-minus-CPU convention.
    Negation here changes only the displayed subtraction order, not its error.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(6.7, 5.1), sharex="col", squeeze=False,
                               gridspec_kw={"height_ratios": [3, 1]})
    for column, panel in enumerate(("peak",)):
        result = panels[panel]
        display_pulls = -np.asarray(result["pull"], dtype=float)
        edges = np.asarray(result["edges_GeV"])
        centers = (edges[:-1] + edges[1:]) / 2
        for backend, color, marker, label in (
                ("cpu", "#0072B2", "o", "CPU: four independent seeds"),
                ("gpu", "#D55E00", "s", "GPU: four different seeds")):
            errors = np.sqrt(np.maximum(0, np.diag(result[backend + "_covariance_of_mean"])))
            axes[0, column].errorbar(centers, result[backend + "_mean"], yerr=errors,
                color=color, marker=marker, markersize=5 if backend == "cpu" else 2.5,
                markerfacecolor="none" if backend == "cpu" else color,
                linewidth=.9, capsize=2, label=label)
        for state, color, size in (("estimated_independent_means", "black", 3),
                                  ("zero_variance_identity_convention", "0.6", 2)):
            indices = [i for i, value in enumerate(result["pull_state"]) if value == state]
            if indices:
                axes[1, column].errorbar(centers[indices],
                    display_pulls[indices], yerr=np.ones(len(indices)),
                    fmt="o", color=color, markersize=size, capsize=2)
        divergent = [i for i, state in enumerate(result["pull_state"])
                     if state == "nonzero_difference_without_observed_scatter"]
        if divergent:
            finite = [abs(float(value)) for value in result["pull"] if np.isfinite(float(value))]
            bound = max(4., max(finite, default=0) + 2.)
            for index in divergent:
                sign = 1 if display_pulls[index] > 0 else -1
                axes[1, column].annotate("", xy=(centers[index], sign * bound),
                    xytext=(centers[index], sign * (bound-1)),
                    arrowprops={"arrowstyle": "->", "color": "#D55E00"})
            axes[1, column].set_ylim(-bound-.5, bound+.5)
        axes[1, column].axhline(0, color="0.5", linewidth=.6)
        axes[1, column].set(xlim=(edges[0], edges[-1]), xlabel=r"$W$ [GeV]",
                            ylabel=r"Pull (CPU$-$GPU)")
        axes[0, column].set_ylabel(r"$dL/dW$ [$10^{34}$ m$^{-2}$ GeV$^{-1}$ per crossing]")
        axes[0, column].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if column == 0:
            axes[0, column].legend(fontsize=7, loc="upper left")
    figure.suptitle("FCC-ee, 240 GeV: disjoint CPU and GPU seed ensembles")
    figure.tight_layout()
    figure.savefig(output / "gpu_cpu_independent.pdf", bbox_inches="tight")
    figure.savefig(output / "gpu_cpu_independent.png", bbox_inches="tight", dpi=160)
    plt.close(figure)


def reduce(manifest_path, output):
    """Authenticate native records and write the fixed independent display."""
    manifest_path, output = Path(manifest_path), Path(output)
    document = json.loads(manifest_path.read_text())
    selected = select_independent_rows(document)
    values = {backend: {panel: [] for panel in document["panels"]} for backend in selected}
    for backend, rows in selected.items():
        for row in rows:
            spectrum = checked_file(manifest_path.parent, row["spectrum"])
            receipt_path = checked_file(manifest_path.parent, row["receipt"])
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "EXECUTED" or receipt["returncode"] != 0:
                raise ValueError("Unsuccessful run cannot enter independent comparison")
            for key in ("seed", "source_identity", "binary_sha256", "physics_input_sha256"):
                if receipt[key] != row[key]:
                    raise ValueError("Receipt identity mismatch: " + key)
            if receipt["outputs"][spectrum.name]["sha256"] != row["spectrum"]["sha256"]:
                raise ValueError("Spectrum is not bound to its execution receipt")
            native, _ = read_spectrum(spectrum.read_text(), row["seed"])
            for panel, definition in document["panels"].items():
                edges = np.asarray(definition["edges_GeV"])
                values[backend][panel].append(remap_mass(native, edges) / np.diff(edges) / 1e34)
    panels = {}
    for panel, definition in document["panels"].items():
        panels[panel] = independent_comparison(values["cpu"][panel], values["gpu"][panel])
        panels[panel]["edges_GeV"] = definition["edges_GeV"]
    summary = dict(schema="feebl.independent-backend-display/1",
        selection={"cpu_seeds": CPU_SEEDS, "gpu_seeds": GPU_SEEDS,
                   "rule": "Odd seeds for CPU, even seeds for GPU; fixed without selecting on outcomes."},
        original_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panels=panels,
        scope="Independent four-versus-four statistical display; the original eight paired closures and complete timing comparison remain separate.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_comparison(panels, output)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in output.iterdir() if path.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reduce(args.manifest, args.output)
