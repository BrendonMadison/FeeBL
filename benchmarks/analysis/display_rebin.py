"""Remap spectrum masses and complete covariance onto supported display bins."""

import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_statistics import column, read_rows, write_rows


def display_edges(low, high, count=40, limits=None):
    """Allocate exactly count bins across connected measured support components.

    Components retain their gaps. Within each component, the requested display
    bins target uniform widths, snapping to source edges when enough edges
    exist. A view window clips support without extrapolation.
    """
    components = []
    for left, right in zip(low, high):
        if limits is not None:
            left, right = max(left, limits[0]), min(right, limits[1])
        if right <= left:
            continue
        if components and left <= components[-1][1] + 1e-9:
            components[-1][1] = max(right, components[-1][1])
        else:
            components.append([left, right])
    if not components or count < len(components):
        raise ValueError(
            "Requested display bins cannot preserve the supported components"
        )
    widths = np.diff(np.asarray(components), axis=1)[:, 0]
    quotas = (count - len(components)) * widths / widths.sum()
    bins = 1 + np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - np.floor(quotas)), kind="stable")[
        : count - bins.sum()
    ]:
        bins[index] += 1
    edges = [
        np.linspace(left, right, n + 1) for (left, right), n in zip(components, bins)
    ]
    for index, ((left, right), edge) in enumerate(zip(components, edges)):
        candidates = np.unique(np.r_[low, high])
        candidates = candidates[
            (candidates > left + 1e-9) & (candidates < right - 1e-9)
        ]
        if len(candidates) >= len(edge) - 2 and len(edge) > 2:
            nearest = candidates[np.argmin(abs(edge[1:-1, None] - candidates), axis=1)]
            if np.all(np.diff(nearest) > 0):
                edges[index] = np.r_[left, nearest, right]
    return (
        np.concatenate([edge[:-1] for edge in edges]),
        np.concatenate([edge[1:] for edge in edges]),
    )


def remap(rows, covariance, count=40, limits=None, exclude_first_bin=False,
          display_upper_limit=None):
    """Transfer masses and covariance onto selected display bins.

    Optional display cuts act after defining the original coarse edges, so
    removing an endpoint bin does not shift every remaining boundary. A
    partially retained upper bin is integrated from the fine source bins.
    """
    low, high = column(rows, "bin_low"), column(rows, "bin_high")
    target_low, target_high = display_edges(low, high, count, limits)
    if exclude_first_bin:
        target_low, target_high = target_low[1:], target_high[1:]
    if display_upper_limit is not None:
        selected = target_low < display_upper_limit
        target_low = target_low[selected]
        target_high = np.minimum(target_high[selected], display_upper_limit)
    if not len(target_low):
        raise ValueError("Display cuts exclude every bin")
    overlap = np.maximum(
        0, np.minimum(target_high[:, None], high) - np.maximum(target_low[:, None], low)
    )
    matrix = overlap / (high - low)
    measured = matrix @ column(rows, "simulation_mass")
    reference = matrix @ column(rows, "reference_mass")
    transferred = {
        key: matrix @ covariance[key] @ matrix.T
        for key in ("simulation", "reference", "simulation_reference")
    }
    variance = np.diag(
        transferred["simulation"]
        + transferred["reference"]
        - transferred["simulation_reference"]
        - transferred["simulation_reference"].T
    )
    if np.any(variance <= 0):
        raise ValueError("A rebinned comparison has nonpositive uncertainty")
    pulls = (measured - reference) / np.sqrt(variance)
    regularized = np.array(
        [str(row.get("empty_bin_regularized", False)).lower() == "true" for row in rows]
    )
    contains_regularizer = np.any((matrix > 0) & regularized[None, :], axis=1)
    result = []
    for index, (left, right) in enumerate(zip(target_low, target_high)):
        row = {
            "bin_low": left,
            "bin_high": right,
            "simulation_mass": measured[index],
            "reference_mass": reference[index],
            "simulation_variance": transferred["simulation"][index, index],
            "reference_variance": transferred["reference"][index, index],
            "simulation_reference_covariance": transferred["simulation_reference"][
                index, index
            ],
            "pull": pulls[index],
            "pull_standard_uncertainty": 1.0,
            "empty_bin_regularized": bool(
                measured[index] == 0 and contains_regularizer[index]
            ),
            "contains_regularized_variance": bool(contains_regularizer[index]),
            "simulation_method": rows[0].get(
                "simulation_method", "retained source covariance"
            ),
        }
        result.append(row)
    return result, transferred, matrix


def load_display(directory, name, output, count=40, limits=None,
                 exclude_first_bin=False, display_upper_limit=None):
    """Require a full covariance sidecar and save auditable display reductions."""
    directory, output = Path(directory), Path(output)
    csv_path = directory / (name + ".csv")
    covariance_path = directory / (name + "_covariance.npz")
    rows = read_rows(csv_path)
    with np.load(covariance_path) as archive:
        covariance = {key: archive[key] for key in archive.files}
    for key in ("simulation", "reference"):
        if not np.allclose(
            np.diag(covariance[key]),
            column(rows, key + "_variance"),
            rtol=1e-8,
            atol=1e-18,
        ):
            raise ValueError("Covariance sidecar does not match the source table")
    reduced, transferred, matrix = remap(
        rows, covariance, count, limits, exclude_first_bin, display_upper_limit
    )
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / (name + ".csv"), reduced)
    np.savez_compressed(
        output / (name + "_covariance.npz"), **transferred, remap=matrix
    )
    receipt = {
        "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_covariance_sha256": hashlib.sha256(
            covariance_path.read_bytes()
        ).hexdigest(),
        "source_bins": len(rows),
        "display_bins": len(reduced),
        "view_limits_GeV": limits,
        "exclude_first_coarse_bin": exclude_first_bin,
        "display_upper_limit_GeV": display_upper_limit,
        "method": "Linear bin-overlap transfer A; C_display=A C_source A^T; pulls recomputed from transferred central values and covariance.",
        "normalization": "Original panel normalization retained; display clipping does not renormalize.",
        "support": "Connected input support only; gaps preserved; no reference extrapolation.",
        "source_simulation_method": rows[0].get("simulation_method"),
        "display_pull_rms": float(np.sqrt(np.mean(column(reduced, "pull") ** 2))),
        "source_mass_sums": {
            key: float(column(rows, key + "_mass").sum())
            for key in ("simulation", "reference")
        },
        "display_retained_mass_sums": {
            key: float(column(reduced, key + "_mass").sum())
            for key in ("simulation", "reference")
        },
        "source_bins_without_display_overlap": int(np.count_nonzero(matrix.sum(axis=0) == 0)),
        "source_bins_with_fractional_display_overlap": int(
            np.count_nonzero((matrix.sum(axis=0) > 1e-12) & (matrix.sum(axis=0) < 1 - 1e-12))
        ),
    }
    (output / (name + "_display.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return reduced
