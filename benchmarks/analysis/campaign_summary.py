#!/usr/bin/env python3
"""Write manuscript tables from the independent-seed and domain campaign.

This reduction uses compact numerical records, never starts a simulation, and
does not assign a physics pass from a successful variance check. Three paired
seeds compare transverse domains at fixed cell spacing. The native miss
counter measures rejected main-grid/source-stencil visits, not lost charge.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparison_statistics import read_rows, write_rows
from paper_tables import mean_sem, tabular


def digest(path):
    """Hash a compact input record for the regeneration receipt."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel_label(name):
    """Use short labels while preserving the complete identifier in CSV."""
    labels = {
        "monochromatization_d0_pull": r"Geometry, $D_x=0$",
        "monochromatization_opposite_pull": "Geometry, opposite dispersion",
        "monochromatization_same_pull": "Geometry, equal dispersion",
        "ilc_pair_yields_cain_pull": "ILC pair yields (CAIN comparison)",
        "ilc_pair_yields_gp_pull": "ILC pair yields (GP comparison)",
    }
    if name in labels:
        return labels[name]
    machine, energy, region, _ = name.split("_")
    return f"{machine.upper()} {energy} GeV, {region}"


def summarize_calibration(path, output):
    """Export descriptive held-out checks and fixed analytic observables."""
    calibration = json.loads(path.read_text())
    rows, table, projections = [], [], []
    for panel in calibration["panels"]:
        row = {
            key: panel[key]
            for key in (
                "panel",
                "seed_count",
                "held_out_count",
                "covariance_rank",
                "nonzero_training_variance_bins",
                "held_out_t_rms_descriptive",
                "expected_t_rms",
                "fraction_inside_pointwise_95percent_t_interval",
                "maximum_absolute_held_out_t",
            )
        }
        row["excluded_zero_variance_bin_count"] = len(
            panel["excluded_zero_variance_bins"]
        )
        quantiles = panel.get(
            "historical_proxy_over_single_seed_empirical_sigma_quantiles", [None] * 3
        )
        for label, value in zip(("p10", "median", "p90"), quantiles):
            row["historical_to_empirical_single_seed_sigma_" + label] = value
        rows.append(row)
        table.append(
            [
                panel_label(row["panel"]),
                row["covariance_rank"],
                f"{row['held_out_t_rms_descriptive']:.2f}",
                f"{100 * row['fraction_inside_pointwise_95percent_t_interval']:.1f}",
                row["excluded_zero_variance_bin_count"],
            ]
        )
        for observable, values in panel.get("fixed_observables", {}).items():
            projections.append(
                {"panel": panel["panel"], "observable": observable, **values}
            )
    write_rows(output / "seed_covariance_summary.csv", rows)
    tabular(
        output / "seed_covariance.tex",
        ["Panel", "Rank", r"Held-out $t$ RMS", r"Coverage [\%]", "Zero bins"],
        table,
    )
    if projections:
        write_rows(output / "geometry_observables.csv", projections)
        projection_table = []
        for row in projections:
            label = {
                "mean_offset_from_250_GeV": r"$\langle W-250\rangle$ [GeV]",
                "second_moment_about_250_GeV2": r"$\langle(W-250)^2\rangle$ [GeV$^2$]",
                "mass_above_249_8_GeV": r"$\Pr(W>249.8\,\mathrm{GeV})$",
            }[row["observable"]]
            projection_table.append(
                [
                    panel_label(row["panel"]),
                    label,
                    mean_sem(row["mean"], row["standard_error_of_mean"]),
                    f"{row['fixed_reference']:.6g}",
                    f"{row['reference_t']:.2f}",
                    f"{row['held_out_t']:.2f}",
                ]
            )
        tabular(
            output / "geometry_observables.tex",
            [
                "Control",
                "Observable",
                "Mean and SEM",
                "Analytic",
                r"$t_{11}$",
                r"Held-out $t_7$",
            ],
            projection_table,
        )
    return rows


def moments(values):
    """Return the mean and independent-seed standard error."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite independent seed values are required")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize_domains(baseline_path, domain_path, output):
    """Compare each expanded domain with precisely matched baseline seeds."""
    baseline = {(row["case"], row["seed"]): row for row in read_rows(baseline_path)}
    expanded = read_rows(domain_path)
    groups = sorted(
        {(row["baseline_case"], int(row["domain_factor"])) for row in expanded}
    )
    rows, table = [], []
    metrics = ("lumi_ee", "lumi_ee_high", "lumi_gg", "n_phot1", "n_phot2", "de1", "de2")
    for case, factor in groups:
        selected = sorted(
            (
                row
                for row in expanded
                if row["baseline_case"] == case and int(row["domain_factor"]) == factor
            ),
            key=lambda row: int(row["seed"]),
        )
        if len(selected) != 3 or len({row["seed"] for row in selected}) != 3:
            raise ValueError(
                "Each domain control requires three distinct matched seeds"
            )
        controls = [baseline[(case, row["seed"])] for row in selected]
        binary_hashes = {row["binary_sha256"] for row in selected + controls}
        if len(binary_hashes) != 1:
            raise ValueError("Domain comparisons must use the same native binary")
        row = {
            "baseline_case": case,
            "domain_factor": factor,
            "seed_count": len(selected),
            "binary_sha256": binary_hashes.pop(),
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in selected])
            reference = np.asarray([float(item[metric]) for item in controls])
            row[metric + "_mean"], row[metric + "_sem"] = moments(values)
            row[metric + "_baseline_mean"], row[metric + "_baseline_sem"] = moments(
                reference
            )
            (
                row[metric + "_paired_difference_mean"],
                row[metric + "_paired_difference_sem"],
            ) = moments(values - reference)
        reference_luminosity = np.asarray([float(item["lumi_ee"]) for item in controls])
        if np.any(reference_luminosity <= 0):
            raise ValueError(
                "A luminosity-retention ratio needs positive baseline luminosity"
            )
        ratios = (
            np.asarray([float(item["lumi_ee"]) for item in selected])
            / reference_luminosity
        )
        row["paired_luminosity_ratio"], row["paired_luminosity_ratio_sem"] = moments(
            ratios
        )
        for beam in (1, 2):
            key = f"outside_fraction_beam{beam}"
            for label, source in (("domain", selected), ("baseline", controls)):
                values = [float(item[key]) for item in source]
                row[f"{label}_maximum_recorded_rejection_fraction_beam{beam}"] = max(
                    values
                )
        rows.append(row)
        table.append(
            [
                case.replace("halhf", "HALHF "),
                factor,
                mean_sem(
                    row["paired_luminosity_ratio"], row["paired_luminosity_ratio_sem"]
                ),
                f"{100 * row['domain_maximum_recorded_rejection_fraction_beam1']:.3g}",
                f"{100 * row['domain_maximum_recorded_rejection_fraction_beam2']:.3g}",
            ]
        )
    write_rows(output / "domain_refinement_summary.csv", rows)
    tabular(
        output / "domain_refinement.tex",
        [
            "Case",
            "Extent factor",
            r"$\mathcal L/\mathcal L_0$",
            r"Reject 1 [\%]",
            r"Reject 2 [\%]",
        ],
        table,
    )
    return rows


def summarize(compact, output, calibration_only=False):
    """Create new derived tables with hashes and interpretation limits."""
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        compact / name
        for name in (
            ("calibration.json",)
            if calibration_only
            else ("calibration.json", "baseline_scalars.csv", "domain_scalars.csv")
        )
    ]
    calibration = summarize_calibration(paths[0], output)
    domains = [] if calibration_only else summarize_domains(paths[1], paths[2], output)
    receipt = {
        "schema": "feebl.campaign-summary/1",
        "inputs": {path.name: digest(path) for path in paths},
        "producer_sha256": digest(__file__),
        "calibration_panels": len(calibration),
        "domain_groups": len(domains),
        "domain_scope": "Not included: calibration-only reduction"
        if calibration_only
        else "Three-seed matched domain comparisons",
        "interpretation": [
            "Held-out bins are correlated; their coverage fraction has no binomial error here.",
            "Eight training and four held-out runs give a t7 benchmark with expected RMS sqrt(7/5).",
            "The two pair-reference panels reuse the same simulation sample and are not independent checks.",
            "Analytic-control offsets test numerical bias separately from held-out variance consistency.",
            "Domain ratios use three matched seeds and the same binary; changed random branching can weaken pairing.",
            "Domain and cell counts grow together. These tests preserve transverse cell spacing.",
            "Rejection entries are maxima over seeds of native maximum per-step visit fractions, not lost charge.",
            "Small-ensemble sampling errors do not include numerical discretization or model uncertainty.",
        ],
    }
    (output / "campaign_summary.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.compact, args.output, args.calibration_only), indent=2
        )
    )
