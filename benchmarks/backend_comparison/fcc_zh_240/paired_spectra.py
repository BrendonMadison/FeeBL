"""Reduce matched CPU/GPU collisions without treating encounters as trials.

This offline analysis reads saved outputs and does not launch simulations.
The manifest fixes the seeds, comparisons and display bins before execution.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


COLUMNS = "W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(root, entry):
    """Resolve and authenticate one relative artifact without escaping root."""
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or escaping analysis input")
    if digest(path) != entry["sha256"]:
        raise ValueError("Analysis input checksum differs: " + entry["path"])
    return path


def read_spectrum(text, expected_seed):
    """Read the ordinary native pair spectrum, retaining sparse-bin semantics."""
    lines = text.splitlines()
    if not lines or lines[0] != "#FEEBL-NATIVE 1.0":
        raise ValueError("Expected the ordinary native spectrum format")
    if lines.count("#BLOCK pair_spectrum") != 1:
        raise ValueError("Exactly one pair-spectrum block is required")
    if lines.count("#COLUMNS " + COLUMNS) != 1:
        raise ValueError("Native spectrum columns changed")
    seeds = [int(line.split()[1]) for line in lines if line.startswith("#rndm_seed ")]
    if seeds != [expected_seed]:
        raise ValueError("Native seed does not match its declared replica")
    rows, active, footer = [], False, None
    metadata = {}
    for line in lines:
        if line == "#BLOCK pair_spectrum":
            active = True
        elif line.startswith("#END "):
            if footer is not None or not active:
                raise ValueError("Unexpected native footer")
            footer = re.fullmatch(r"#END pair_spectrum records (\d+)", line)
            if footer is None:
                raise ValueError("Malformed native footer")
            active = False
        elif line.startswith(("#underflow ", "#overflow ", "#mean_W ", "#rms_W ")):
            key, value = line[1:].split()[:2]
            if key in metadata:
                raise ValueError("Repeated native metadata")
            metadata[key] = float(value)
        elif line.strip() and not line.startswith("#"):
            if not active:
                raise ValueError("Data outside native block")
            values = [float(value) for value in line.split()]
            if len(values) != 11 or not np.isfinite(values).all():
                raise ValueError("Malformed or nonfinite native row")
            rows.append([values[0], values[1], values[3]])
    if footer is None or int(footer[1]) != len(rows) or not rows:
        raise ValueError("Incomplete or empty native spectrum")
    bins = np.asarray(rows)
    if np.any(bins[:, 1] <= bins[:, 0]) or np.any(bins[:, 2] < 0):
        raise ValueError("Invalid spectrum edges or weights")
    if np.any(bins[1:, 0] < bins[:-1, 1]):
        raise ValueError("Overlapping, duplicate or unordered native bins")
    if set(metadata) != {"underflow", "overflow", "mean_W", "rms_W"}:
        raise ValueError("Missing native flow or unbinned-moment metadata")
    if not np.isfinite(list(metadata.values())).all():
        raise ValueError("Nonfinite spectrum metadata")
    if min(metadata["underflow"], metadata["overflow"], metadata["rms_W"]) < 0:
        raise ValueError("Negative flow weight or RMS")
    return bins, metadata


def remap_mass(native, edges):
    """Conservatively rebin mass; omitted native bins have observed mass zero.

    Fractional intersections use a piecewise-constant native-bin density.
    This does not establish that unsampled bins have zero population variance.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("Finite fixed display edges are required")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Display edges must increase")
    overlap = np.maximum(
        0,
        np.minimum(edges[1:, None], native[:, 1])
        - np.maximum(edges[:-1, None], native[:, 0]),
    )
    matrix = overlap / (native[:, 1] - native[:, 0])
    return matrix @ native[:, 2]


def mean_covariance(values):
    """Mean and empirical covariance of the mean across independent seeds."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent collision vectors required")
    shifted = values - values[0]
    centered = shifted - shifted.mean(axis=0)
    return (
        values[0] + shifted.mean(axis=0),
        centered.T @ centered / (len(values) * (len(values) - 1)),
    )


def paired_comparison(cpu, gpu):
    """Retain the covariance of shared-seed noise exactly once."""
    cpu, gpu = np.asarray(cpu, dtype=float), np.asarray(gpu, dtype=float)
    if cpu.shape != gpu.shape:
        raise ValueError("CPU/GPU replica vectors must have equal shape")
    cpu_mean, cpu_cov = mean_covariance(cpu)
    gpu_mean, gpu_cov = mean_covariance(gpu)
    difference, covariance = mean_covariance(gpu - cpu)
    cpu_center = cpu - cpu_mean
    gpu_center = gpu - gpu_mean
    cross = cpu_center.T @ gpu_center / (len(cpu) * (len(cpu) - 1))
    # Direct difference covariance is numerically stable when common noise
    # nearly cancels. The expanded form is retained as a diagnostic only.
    expanded = cpu_cov + gpu_cov - cross - cross.T
    scale = max(float(np.max(np.abs(cpu_cov + gpu_cov))), 1e-300)
    if np.max(np.abs(covariance - expanded)) > 2e-12 * scale:
        raise ValueError("Paired covariance identity did not close")
    sigma = np.sqrt(np.maximum(0, np.diag(covariance)))
    states, pulls = [], []
    for index, error in enumerate(sigma):
        if error > 0:
            states.append("estimated")
            pulls.append(float(difference[index] / error))
        elif np.all(cpu[:, index] == gpu[:, index]):
            states.append("zero_denominator_identity_convention")
            pulls.append(0.0)
        else:
            states.append("nonzero_difference_without_seed_scatter")
            # JSON has no numeric infinity. Preserve its sign explicitly and
            # draw a limit arrow, rather than manufacturing an error floor.
            pulls.append("Infinity" if difference[index] > 0 else "-Infinity")
    return {
        "replicas": len(cpu),
        "degrees_of_freedom": len(cpu) - 1,
        "cpu_mean": cpu_mean.tolist(),
        "gpu_mean": gpu_mean.tolist(),
        "cpu_covariance_of_mean": cpu_cov.tolist(),
        "gpu_covariance_of_mean": gpu_cov.tolist(),
        "cpu_gpu_cross_covariance_of_mean": cross.tolist(),
        "difference_gpu_minus_cpu": difference.tolist(),
        "difference_covariance_of_mean": covariance.tolist(),
        "combined_sem": sigma.tolist(),
        "pull": pulls,
        "pull_state": states,
        "pull_error": [1.0 for state in states],
        "covariance_identity_max_absolute_error": float(
            np.max(np.abs(covariance - expanded))
        ),
        "covariance_rank": int(np.linalg.matrix_rank(covariance)),
    }


def validate_inventory(document):
    """Reject missing, duplicate or substituted replicas before reduction."""
    if document["schema"] != "feebl.gpu-cpu-stress-analysis/1":
        raise ValueError("Unsupported stress analysis manifest")
    seeds, variants = document["seeds"], document["variants"]
    if len(seeds) < 4 or len(set(seeds)) != len(seeds):
        raise ValueError("At least four distinct predeclared seeds required")
    if len(set(variants)) != len(variants) or len(variants) < 2:
        raise ValueError("Distinct backend variants required")
    rows = document["runs"]
    identities = [(row["variant"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate variant/seed pair")
    if set(identities) != {(v, s) for v in variants for s in seeds}:
        raise ValueError("Incomplete or undeclared seed/backend ensemble")
    for seed in seeds:
        pair = [row for row in rows if row["seed"] == seed]
        if len({row["physics_input_sha256"] for row in pair}) != 1:
            raise ValueError("CPU/GPU physical input differs for a paired seed")
    if len({row["source_identity"] for row in rows}) != 1:
        raise ValueError("Stress comparison requires one native source identity")
    for variant in variants:
        if len({r["binary_sha256"] for r in rows if r["variant"] == variant}) != 1:
            raise ValueError("A backend ensemble mixes native binaries")
    for entry in document["comparisons"]:
        if entry["cpu"] not in variants or entry["gpu"] not in variants:
            raise ValueError("Comparison refers to an undeclared variant")
    if len({row["id"] for row in document["comparisons"]}) != len(
        document["comparisons"]
    ):
        raise ValueError("Duplicate comparison label")


def timing_record(row, receipt, stderr):
    """Keep wall, CPU consumption, device intervals and allocation separate."""
    if receipt.get("status") != "EXECUTED" or receipt.get("returncode") != 0:
        raise ValueError("Unsuccessful native execution cannot enter comparison")
    for key in ("seed", "binary_sha256"):
        if receipt[key] != row[key]:
            raise ValueError("Execution identity differs from analysis manifest")
    resources = receipt["resources"]
    wall = float(receipt["elapsed_seconds"])
    user = float(resources["user_cpu_seconds"])
    system = float(resources["system_cpu_seconds"])
    if (
        not np.isfinite([wall, user, system]).all()
        or wall <= 0
        or min(user, system) < 0
    ):
        raise ValueError("Invalid process resource accounting")
    gpu_lines = [
        line
        for line in stderr.splitlines()
        if re.match(r"\s*FEEBL (GPU2 (fields|primary)|GPU mode4) summary:", line)
    ]
    if len(gpu_lines) > 1:
        raise ValueError("Ambiguous GPU timing summaries")
    gpu = dict(re.findall(r"(\w+)=([^\s]+)", gpu_lines[0])) if gpu_lines else {}
    if row["gpu_count"]:
        if row["gpu_count"] != 1 or not row.get("gpu_uuid") or not gpu:
            raise ValueError(
                "Each supported GPU collision needs one identified device and a native summary"
            )
        work = gpu.get("device_work_calls", gpu.get("batch_calls", 0))
        if int(work) <= 0:
            raise ValueError("GPU run did not report device work")
        if "GPU mode4" in gpu_lines[0] and (
            gpu.get("implementation") != "encounter-dag-wave-batched-planmany-fields-v1"
            or int(gpu.get("kernel_launches", 0)) <= 0
            or int(gpu.get("cufft_executes", 0)) <= 0
        ):
            raise ValueError("GPU4 instrumented work contract failed")
    elif gpu:
        raise ValueError("CPU run unexpectedly reported GPU work")
    result = {
        "variant": row["variant"],
        "seed": row["seed"],
        "wall_seconds": wall,
        "user_cpu_seconds": user,
        "system_cpu_seconds": system,
        "cpu_core_seconds": user + system,
        "cpu_threads_assigned": row["cpu_threads"],
        "cpu_slot_seconds_reserved_for_process": wall * row["cpu_threads"],
        "gpu_uuid": row.get("gpu_uuid"),
        "gpu_device_seconds_reserved_for_process": wall * row["gpu_count"],
    }
    for native, label in [
        ("kernel_ms", "cuda_event_kernel_seconds"),
        ("h2d_ms", "cuda_event_h2d_seconds"),
        ("d2h_ms", "cuda_event_d2h_seconds"),
        ("setup_ms", "gpu_setup_host_elapsed_seconds"),
        ("total_ms", "gpu_solver_host_elapsed_seconds"),
        ("boundary_ms", "gpu_batch_boundary_host_elapsed_seconds"),
        ("host_pack_ms", "gpu_pack_host_elapsed_seconds"),
        ("host_unpack_ms", "gpu_unpack_host_elapsed_seconds"),
    ]:
        value = float(gpu[native]) / 1000 if native in gpu else None
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("Invalid GPU timing value")
        result[label] = value
    return result


def plot_comparison(document, comparisons, destination):
    """Render original spectra and paired pulls with capped unit error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2, 2, figsize=(8.2, 5.8), sharex="col", gridspec_kw={"height_ratios": [3, 1]}
    )
    for column, panel in enumerate(("full", "peak")):
        result = comparisons[panel]
        edges = np.asarray(document["panels"][panel]["edges_GeV"])
        centers = (edges[:-1] + edges[1:]) / 2
        for key, color, marker in [("cpu", "#0072B2", "o"), ("gpu", "#D55E00", "s")]:
            error = np.sqrt(np.maximum(0, np.diag(result[key + "_covariance_of_mean"])))
            axes[0, column].errorbar(
                centers,
                result[key + "_mean"],
                yerr=error,
                color=color,
                marker=marker,
                markersize=5 if key == "cpu" else 2.5,
                markerfacecolor="none" if key == "cpu" else color,
                linewidth=0.9,
                capsize=2,
                label="CPU (16 threads)" if key == "cpu" else "GPU + 16 host threads",
            )
        indices = [
            i for i, state in enumerate(result["pull_state"]) if state == "estimated"
        ]
        if indices:
            axes[1, column].errorbar(
                centers[indices],
                np.asarray(result["pull"], dtype=float)[indices],
                yerr=np.ones(len(indices)),
                fmt="o",
                color="black",
                markersize=3,
                capsize=2,
            )
        equal = [
            i
            for i, state in enumerate(result["pull_state"])
            if state == "zero_denominator_identity_convention"
        ]
        if equal:
            axes[1, column].errorbar(
                centers[equal],
                np.zeros(len(equal)),
                yerr=np.ones(len(equal)), fmt="o", markersize=2, capsize=2,
                color="0.6",
                label="Identical pairs: zero convention",
            )
        unresolved = [i for i, state in enumerate(result["pull_state"])
                      if state == "nonzero_difference_without_seed_scatter"]
        if unresolved:
            finite = [abs(float(value)) for value in result["pull"]
                      if np.isfinite(float(value))]
            bound = max(4.0, max(finite, default=0) + 2.0)
            for index in unresolved:
                sign = 1 if result["pull"][index] == "Infinity" else -1
                axes[1, column].annotate("", xy=(centers[index], sign * bound),
                    xytext=(centers[index], sign * (bound - 1)),
                    arrowprops={"arrowstyle": "->", "color": "#D55E00"})
            axes[1, column].set_ylim(-bound - 0.5, bound + 0.5)
            axes[1, column].text(
                0.02,
                0.94,
                f"{len(unresolved)} divergent quotients: zero estimated SEM",
                va="top",
                transform=axes[1, column].transAxes,
                fontsize=7,
            )
        axes[1, column].axhline(0, color="0.5", linewidth=0.6)
        axes[1, column].set_xlabel(r"$W$ [GeV]")
        axes[1, column].set_ylabel("Paired pull")
        axes[0, column].set_title("Full spectrum" if panel == "full" else "Peak")
        axes[0, column].set_ylabel(
            r"$dL/dW$ [$10^{34}$ m$^{-2}$ GeV$^{-1}$ per crossing]"
        )
        if column == 0:
            axes[0, column].legend(fontsize=7, loc="upper left")
        axes[0, column].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axes[1, column].set_xlim(edges[0], edges[-1])
    figure.suptitle(
        f"{document['scenario_label']}; {len(document['seeds'])} independent matched seeds"
    )
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def timing_comparisons(document, records):
    """Descriptive matched wall-time ratios; shared timing blocks are not iid."""
    indexed = {(row["variant"], row["seed"]): row for row in records}
    result = {}
    for comparison in document["comparisons"]:
        ratios = [
            indexed[comparison["cpu"], seed]["wall_seconds"]
            / indexed[comparison["gpu"], seed]["wall_seconds"]
            for seed in document["seeds"]
        ]
        result[comparison["id"]] = {
            "cpu_to_gpu_wall_ratio_by_seed": dict(
                zip(map(str, document["seeds"]), ratios)
            ),
            "median_wall_ratio": float(np.median(ratios)),
            "minimum_wall_ratio": min(ratios),
            "maximum_wall_ratio": max(ratios),
            "interpretation": "Descriptive per-process wall-time ratio under declared contention; no independent timing confidence interval.",
        }
    return result


def reduce(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    document = json.loads(manifest_path.read_text())
    validate_inventory(document)
    root = manifest_path.parent
    values, timings, flows = {}, [], []
    for row in document["runs"]:
        path = checked_file(root, row["spectrum"])
        receipt_path = checked_file(root, row["receipt"])
        stderr_path = checked_file(root, row["stderr"])
        receipt = json.loads(receipt_path.read_text())
        for name, artifact in [
            (path.name, row["spectrum"]),
            (stderr_path.name, row["stderr"]),
        ]:
            if receipt["outputs"][name]["sha256"] != artifact["sha256"]:
                raise ValueError("Output is not bound to its execution receipt")
        native, meta = read_spectrum(path.read_text(), row["seed"])
        timings.append(timing_record(row, receipt, stderr_path.read_text()))
        entry = {"variant": row["variant"], "seed": row["seed"], **meta}
        for panel, definition in document["panels"].items():
            edges = np.asarray(definition["edges_GeV"])
            mass = remap_mass(native, edges)
            values[row["variant"], row["seed"], panel] = mass / np.diff(edges) / 1e34
            entry[panel + "_retained_mass"] = float(mass.sum())
            entry[panel + "_outside_mass"] = float(native[:, 2].sum() - mass.sum())
        flows.append(entry)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for item in document["comparisons"]:
        panels = {}
        for panel, definition in document["panels"].items():
            cpu = [values[item["cpu"], seed, panel] for seed in document["seeds"]]
            gpu = [values[item["gpu"], seed, panel] for seed in document["seeds"]]
            result = paired_comparison(cpu, gpu)
            result.update(
                cpu_variant=item["cpu"],
                gpu_variant=item["gpu"],
                edges_GeV=definition["edges_GeV"],
            )
            panels[panel] = result
        results[item["id"]] = panels
        plot_comparison(document, panels, output / item["id"])
    summary = {
        "schema": "feebl.gpu-cpu-stress-comparison/1",
        "status": "REDUCED_REQUIRES_SCIENTIFIC_REVIEW",
        "manifest_sha256": digest(manifest_path),
        "producer_sha256": digest(__file__),
        "comparisons": results,
        "timings": timings,
        "flows": flows,
        "timing_comparisons": timing_comparisons(document, timings),
        "notes": [
            "Covariance uses independent complete collisions and paired differences.",
            "Identical pairs use pull=0 and unit display error by explicit zero-denominator convention; no positive variance or degrees of freedom is inferred.",
            "Nonzero differences with zero estimated SEM are signed Infinity strings in JSON and limit arrows in plots, not a finite significance claim.",
            "No full-bin chi-square or convergence/equivalence claim is inferred.",
            "Device reservation, CUDA event intervals and wall time are different quantities.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    receipt = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "reduction_sha256.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reduce(arguments.manifest, arguments.output)
