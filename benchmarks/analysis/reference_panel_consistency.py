#!/usr/bin/env python3
"""Compare full and peak reference panels on the same coarse-bin support.

Both panels are conditionally normalized after transfer to the original full
panel's bins. Clipped edge bins require the published step to be constant
within its bin. A second, interior-only comparison avoids that assumption at
the outer support boundary and reports how much peak-panel mass it discards.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_statistics import (
    PANELS,
    column,
    read_rows,
    reference_estimate,
    sample_covariance,
    write_rows,
)
from paper_tables import tabular
import reference_uncertainty as reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap_matrix(low, high, target_low, target_high):
    """Map integrated step-bin masses using exact interval intersections."""
    overlap = np.maximum(
        0, np.minimum(target_high[:, None], high) - np.maximum(target_low[:, None], low)
    )
    return overlap / (high - low)


def conditioned(values, covariance, matrix):
    """Transfer mass and covariance before conditioning on retained support."""
    mass = matrix @ values
    return (
        mass / mass.sum(),
        reference.normalize_covariance(mass, matrix @ covariance @ matrix.T),
    )


def common_bins(full_low, full_high, peak_low, peak_high, interior):
    """Intersect supports without filling a censored peak-panel interval."""
    components = []
    for a, b in sorted(zip(peak_low, peak_high)):
        if components and a <= components[-1][1] + 1e-10:
            components[-1][1] = max(components[-1][1], b)
        else:
            components.append([a, b])
    intervals = []
    for a, b in zip(full_low, full_high):
        if interior:
            if any(a >= p and b <= q for p, q in components):
                intervals.append((a, b))
        else:
            intervals.extend(
                (max(a, p), min(b, q))
                for p, q in components
                if min(b, q) - max(a, p) > 1e-10
            )
    return np.asarray(intervals).T


def run(output, ensemble=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native = np.load(ensemble, allow_pickle=False) if ensemble is not None else None
    summaries, detail, sources = [], [], {}
    for collider in ("ilc_1000", "halhf_250", "halhf_380", "halhf_550"):
        full_name, peak_name = (
            collider + "_" + scope + "_pull" for scope in ("full", "peak")
        )
        full_rows, peak_rows = (
            read_rows(PANELS / (name + ".csv")) for name in (full_name, peak_name)
        )
        full, full_cov, _ = reference_estimate(full_name, full_rows)
        peak, peak_cov, _ = reference_estimate(peak_name, peak_rows)
        flo, fhi = column(full_rows, "bin_low"), column(full_rows, "bin_high")
        plo, phi = column(peak_rows, "bin_low"), column(peak_rows, "bin_high")
        lower, upper = max(flo.min(), plo.min()), min(fhi.max(), phi.max())
        for name in (full_name, peak_name):
            source = reference.curve_panel(name)
            sources[name] = {
                "file": source["path"].name,
                "sha256": digest(source["path"]),
                "primary_url": source["source_url"],
                "locator": source["source_locator"],
                "comparison_support_sha256": digest(PANELS / (name + ".csv")),
            }
        for mode in ("clipped_common_support", "complete_coarse_bins_only"):
            low, high = common_bins(
                flo, fhi, plo, phi, mode == "complete_coarse_bins_only"
            )
            fmatrix, pmatrix = (
                overlap_matrix(a, b, low, high) for a, b in ((flo, fhi), (plo, phi))
            )
            covered = (pmatrix @ (phi - plo)) / (high - low)
            if np.any(abs(covered - 1) > 1e-4):
                raise ValueError("Do not bridge an unmeasured peak-panel gap")
            fv, fc = conditioned(full, full_cov, fmatrix)
            pv, pc = conditioned(peak, peak_cov, pmatrix)
            variance = np.diag(fc + pc)
            scenario_pull = (fv - pv) / np.sqrt(variance)
            row = {
                "collider": collider,
                "support": mode,
                "coarse_bins": len(low),
                "minimum_energy_GeV": float(low.min()),
                "maximum_energy_GeV": float(high.max()),
                "retained_full_panel_mass_fraction": float((fmatrix @ full).sum()),
                "retained_peak_panel_mass_fraction": float((pmatrix @ peak).sum()),
                "reference_conditional_cdf_distance": float(
                    np.max(abs(np.cumsum(fv - pv)))
                ),
                "reference_total_variation_distance": float(np.sum(abs(fv - pv)) / 2),
                "reference_zero_cross_covariance_pull_rms": float(
                    np.sqrt(np.mean(scenario_pull ** 2))
                ),
                "reference_zero_cross_covariance_max_abs_pull": float(
                    max(abs(scenario_pull))
                ),
                "native_seed_count": 0,
                "native_paired_cdf_distance": "",
                "native_paired_total_variation_distance": "",
                "native_max_absolute_bin_difference": "",
                "native_max_paired_bin_sem": "",
            }
            if native is not None and full_name in native and peak_name in native:
                if not np.array_equal(
                    native[full_name + "__seeds"], native[peak_name + "__seeds"]
                ):
                    raise ValueError(
                        "Native full and peak panels must use the same seeds"
                    )
                nf, npk = native[full_name] @ fmatrix.T, native[peak_name] @ pmatrix.T
                nf /= nf.sum(axis=1)[:, None]
                npk /= npk.sum(axis=1)[:, None]
                difference, paired_covariance = sample_covariance(nf - npk)
                row.update(
                    native_seed_count=len(nf),
                    native_paired_cdf_distance=float(max(abs(np.cumsum(difference)))),
                    native_paired_total_variation_distance=float(
                        np.sum(abs(difference)) / 2
                    ),
                    native_max_absolute_bin_difference=float(max(abs(difference))),
                    native_max_paired_bin_sem=float(
                        np.sqrt(np.diag(paired_covariance)).max()
                    ),
                )
            summaries.append(row)
            for i, (a, b) in enumerate(zip(low, high)):
                detail.append(
                    {
                        "collider": collider,
                        "support": mode,
                        "bin_low_GeV": a,
                        "bin_high_GeV": b,
                        "full_panel_conditional_mass": fv[i],
                        "peak_panel_conditional_mass": pv[i],
                        "full_panel_variance": fc[i, i],
                        "peak_panel_variance": pc[i, i],
                        "zero_cross_covariance_scenario_pull": scenario_pull[i],
                        "pull_display_error": 1.0,
                    }
                )
    if native is not None:
        native.close()
    write_rows(output / "panel_consistency_summary.csv", summaries)
    write_rows(output / "panel_consistency_bins.csv", detail)
    tabular(
        output / "panel_consistency.tex",
        [
            "Collider",
            "Support",
            "Bins",
            r"Peak mass retained",
            r"$D_{\rm CDF}$",
            "TV distance",
        ],
        [
            [
                r["collider"].replace("_", " ").upper(),
                "Overlap" if r["support"].startswith("clipped") else "Interior",
                str(r["coarse_bins"]),
                f"{r['retained_peak_panel_mass_fraction']:.3f}",
                f"{r['reference_conditional_cdf_distance']:.4f}",
                f"{r['reference_total_variation_distance']:.4f}",
            ]
            for r in summaries
        ],
    )
    provenance = {
        "schema": "feebl.reference-panel-consistency/1",
        "producer_sha256": digest(__file__),
        "sources": sources,
        "native_ensemble_sha256": digest(ensemble) if ensemble is not None else None,
        "native_ensemble_metadata_sha256": digest(Path(ensemble).with_suffix(".json"))
        if ensemble is not None
        else None,
        "definition": "Integrate published step densities onto complete or clipped full-panel bins and normalize both on the identical retained union.",
        "limitations": [
            "Full and peak plots may be views of the same source sample; their cross-covariance is unknown. The zero-cross-covariance standardized difference is an explicit scenario, not a significance.",
            "Clipped coarse edge bins assume a constant density within the full-panel step. Interior-only support avoids this edge assumption but may remove most of the peak; retained peak mass is reported.",
            "The native paired control measures transfer/conditioning differences between two retained histogram representations of the same twelve collision runs. It is not a new native simulation or an independent physics comparison.",
            "No inconsistent reference bin is deleted; both support definitions and every transferred bin are exported.",
        ],
    }
    (output / "panel_consistency_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.ensemble), indent=2))
