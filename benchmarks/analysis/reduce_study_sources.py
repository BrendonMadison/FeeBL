#!/usr/bin/env python3
"""Reconstruct study statistics from retained native scalar spectra.

No simulation or response fit runs here. The response-fit widths are retained
qualified fit outputs; their bounded fitting implementation is in study_fits/.
Luminosity, resonance overlaps, density means/covariances, paired ratios and
empirical FWHM are recomputed from the native bins.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from comparison_statistics import INPUTS, BENCHMARKS, read_rows, column, write_rows
from empirical_fwhm import representations
from peak_resolution import analyze as resolution_width, POLICY as RESOLUTION_POLICY
from study_fits.frozen_spectrum_analysis import load_design_spectrum, resonance_score


def summarize(values, kind, name, metric, unit):
    """Use complete seeded collisions as independent observations."""
    values = np.asarray(values)
    count = len(values)
    variance = float(np.var(values - values[0], ddof=1))
    mean = float(values.mean())
    sem = math.sqrt(variance / count)
    critical = float(t.ppf(0.975, count - 1))
    return dict(
        kind=kind,
        name=name,
        metric=metric,
        unit=unit,
        complete=True,
        n=count,
        status="sample_summary",
        mean=mean,
        sample_variance=variance,
        sample_sd=math.sqrt(variance),
        sem=sem,
        degrees_of_freedom=count - 1,
        student_t_critical=critical,
        confidence_level=0.95,
        t95_low=mean - critical * sem,
        t95_high=mean + critical * sem,
    )


def spectrum_rows(spectra, template):
    """Rebin paired spectra to fixed published display edges, conserving mass."""
    low, high = column(template, "low_GeV"), column(template, "high_GeV")
    all_values = []
    for cohort in spectra:
        values = []
        for spectrum in cohort:
            boundaries = np.r_[spectrum["low"], spectrum["high"][-1]]
            cumulative = np.r_[0.0, np.cumsum(spectrum["weight"])]
            mass = np.interp(high, boundaries, cumulative) - np.interp(
                low, boundaries, cumulative
            )
            density = mass / spectrum["numeric"]["total_weight"] / (high - low)
            values.append(density)
        all_values.append(np.array(values))
    baseline, selected = all_values
    a, b = baseline - baseline[0], selected - selected[0]
    count = len(a)
    ma, mb = baseline[0] + a.mean(axis=0), selected[0] + b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1) / count, b.var(axis=0, ddof=1) / count
    covariance = ((a - a.mean(axis=0)) * (b - b.mean(axis=0))).sum(axis=0) / (
        count * (count - 1)
    )
    return [
        dict(
            low_GeV=l,
            high_GeV=h,
            reference_mean=x,
            candidate_mean=y,
            reference_mean_variance=vx,
            candidate_mean_variance=vy,
            mean_covariance=cov,
            difference_mean_variance=max(0.0, vx + vy - 2 * cov),
        )
        for l, h, x, y, vx, vy, cov in zip(low, high, ma, mb, va, vb, covariance)
    ]


def confirmation(output):
    """Rebuild the two paired three-seed studies and Student-t intervals."""
    native = INPUTS / "study_native"
    fits = {
        record["case"]: record
        for record in json.loads((INPUTS / "confirmation_fit_records.json").read_text())
    }
    records = []
    for machine in ("ilc", "halhf"):
        spectra, metrics = [], []
        for scenario in ("base", "selected"):
            cohort, measured, seeds = [], [], []
            for index in range(1, 4):
                case = f"{machine}_confirm_{scenario}_s{index}"
                spectrum = load_design_spectrum(native / (case + ".fne.gz"))
                fit = fits[case]
                if not fit["qualified"]:
                    raise ValueError("Unqualified fit cannot enter the width table")
                luminosity = spectrum["numeric"]["total_weight"]
                resonance = resonance_score(
                    spectrum["low"], spectrum["high"], spectrum["weight"]
                )
                measured.append(
                    {
                        "luminosity_per_crossing": luminosity,
                        "luminosity_per_second": luminosity
                        / 1e4
                        * fit["parameters"]["repetition_Hz"],
                        "core_sigma": 1000 * fit["sigma_W_GeV"],
                        "resonance_overlap": resonance["peak_normalized_overlap"],
                        "resonance_luminosity": resonance[
                            "effective_luminosity_m2_per_BX"
                        ],
                    }
                )
                cohort.append(spectrum)
                seeds.append(int(spectrum["metadata"]["rndm_seed"]))
            if len(set(seeds)) != 3:
                raise ValueError("Repeated seed within confirmation group")
            spectra.append(cohort)
            metrics.append(
                {key: np.array([r[key] for r in measured]) for key in measured[0]}
            )
            for metric, unit in [
                ("luminosity_per_second", "cm^-2 s^-1 per IP"),
                ("core_sigma", "MeV"),
                ("resonance_overlap", "dimensionless"),
            ]:
                label = machine + ("_baseline" if scenario == "base" else "_selected")
                records.append(
                    summarize(metrics[-1][metric], "group", label, metric, unit)
                )
        if [h["metadata"]["rndm_seed"] for h in spectra[0]] != [
            h["metadata"]["rndm_seed"] for h in spectra[1]
        ]:
            raise ValueError("Paired seed identities disagree")
        for metric in (
            "luminosity_per_crossing",
            "core_sigma",
            "resonance_overlap",
            "resonance_luminosity",
        ):
            records.append(
                summarize(
                    metrics[1][metric] / metrics[0][metric],
                    "paired_candidate_over_baseline",
                    machine + "_selected_vs_baseline",
                    metric,
                    "dimensionless",
                )
            )
        name = f"{machine}_selected_vs_baseline_peak_spectrum.csv"
        template = read_rows(BENCHMARKS / "example_output/scans" / name)
        write_rows(output / name, spectrum_rows(spectra, template))
    write_rows(output / "confirmation_metrics.csv", records)


def high_dispersion(output):
    """Reconstruct fixed-resolution widths and retain the original raw diagnostics."""
    native = INPUTS / "study_native"
    manifest = json.loads((native / "manifest.json").read_text())
    records = []
    for row in manifest:
        if row["family"] != "high_dispersion":
            continue
        spectrum = load_design_spectrum(native / row["compressed_file"])
        stop = 2048 + int(spectrum["metadata"]["peak_bins"])
        measured = representations(
            spectrum["low"], spectrum["high"], spectrum["weight"], 2048, stop
        )
        resolved = resolution_width(spectrum, row["group"])
        records.append(
            {
                **row,
                "seed": int(spectrum["metadata"]["rndm_seed"]),
                "luminosity_m2_per_BX": spectrum["numeric"]["total_weight"],
                "observed_FWHM_MeV": resolved["fwhm_MeV"],
                "legacy_raw_FWHM_MeV": measured[0]["fwhm_MeV"],
                "legacy_maximum_grouping_relative_change": max(
                    abs(r["fwhm_relative_change_from_raw"]) for r in measured
                ),
                "maximum_grouping_relative_change": resolved["maximum_resolution_relative_change"],
                "resolution": resolved,
                "representations": measured,
            }
        )
    (output / "empirical_fwhm_records.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    groups = []
    for label in sorted({r["group"] for r in records}):
        selected = [r for r in records if r["group"] == label]
        values = np.array([r["observed_FWHM_MeV"] for r in selected])
        groups.append(
            {
                "group": label,
                "n": len(values),
                "observed_peak_FWHM_MeV_mean": float(values.mean()),
                "observed_peak_FWHM_MeV_SEM": float(
                    values.std(ddof=1) / np.sqrt(len(values))
                )
                if len(values) > 1
                else "",
                "maximum_per_seed_representation_change": max(
                    r["maximum_grouping_relative_change"] for r in selected
                ),
            }
        )
    write_rows(output / "empirical_fwhm_summary.csv", groups)
    # Keep source-defined optics and grid labels, but recompute every plotted
    # luminosity/width mean, seed SEM, ratio and grouping sensitivity.
    source_rows = read_rows(BENCHMARKS / "example_output/scans/high_dispersion.csv")
    for row in source_rows:
        selected = sorted(
            [r for r in records if r["group"] == row["group"]], key=lambda r: r["seed"]
        )
        if len(selected) != int(row["n"]):
            raise ValueError("Incomplete native seed set for " + row["group"])
        machine = row["machine"]
        rate = 6560 if machine == "ilc" else 16000
        baseline = {r["seed"]: r for r in records if r["group"] == machine + "_d0_scan"}
        for metric, key, factor in [
            ("luminosity_m2_per_BX", "luminosity_m2_per_BX", 1),
            ("luminosity_cm2_per_second", "luminosity_m2_per_BX", rate / 1e4),
            ("observed_peak_FWHM_MeV", "observed_FWHM_MeV", 1),
        ]:
            values = np.array([r[key] * factor for r in selected])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_SEM"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else ""
            )
            row[metric + "_sample_SD"] = (
                float(values.std(ddof=1)) if len(values) > 1 else ""
            )
        ratios = np.array(
            [
                r["luminosity_m2_per_BX"] / baseline[r["seed"]]["luminosity_m2_per_BX"]
                for r in selected
            ]
        )
        row["luminosity_retention_mean"] = float(ratios.mean())
        row["luminosity_retention_SEM"] = (
            float(ratios.std(ddof=1) / np.sqrt(len(ratios))) if len(ratios) > 1 else ""
        )
        row["maximum_per_seed_representation_change"] = max(
            r["maximum_grouping_relative_change"] for r in selected
        )
        row["legacy_flags"] = row.get("flags", "[]")
        row["flags"] = json.dumps([
            dict(case=r["case"], flags=r["resolution"]["flags"])
            for r in selected if r["resolution"]["flags"]
        ])
        row["analysis_bin_width_MeV"] = selected[0]["resolution"]["primary_bin_width_MeV"]
        primary_group = selected[0]["resolution"]["primary_group"]
        means = [np.mean([r["resolution"]["phases"][index]["fwhm_MeV"] for r in selected])
                 for index in range(3 * primary_group)]
        row["bin_alignment_low_MeV"] = float(min(means[:primary_group]))
        row["bin_alignment_high_MeV"] = float(max(means[:primary_group]))
        row["resolution_low_MeV"] = float(min(means))
        row["resolution_high_MeV"] = float(max(means))
    write_rows(output / "high_dispersion.csv", source_rows)
    fcc_native = INPUTS / "monochrom_resolution_native"
    fcc_manifest = json.loads((fcc_native / "manifest.json").read_text())
    fcc = []
    for row in fcc_manifest["cases"]:
        path = fcc_native / row["compressed_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["compressed_sha256"]:
            raise ValueError("Changed FCC native spectrum: " + row["case"])
        spectrum = load_design_spectrum(path)
        fcc.append(dict(**row, resolution=resolution_width(spectrum, row["case"])))
    (output / "fcc_resolution_summary.json").write_text(json.dumps({
        "policy": RESOLUTION_POLICY, "cases": fcc,
        "scope": "Single-seed observed widths; all alignment ranges are numerical diagnostics, not statistical errors"
    }, indent=2) + "\n")
    receipt = {
        "status": "OBSERVED_WIDTHS_REGENERATED_FROM_NATIVE_BINS",
        "policy": RESOLUTION_POLICY,
        "input_manifest_sha256": {
            "analysis_inputs/study_native/manifest.json": hashlib.sha256((native / "manifest.json").read_bytes()).hexdigest(),
            "analysis_inputs/monochrom_resolution_native/manifest.json": hashlib.sha256((fcc_native / "manifest.json").read_bytes()).hexdigest(),
        },
        "producer_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("reduce_study_sources.py", "peak_resolution.py", "empirical_fwhm.py")},
        "input_spectrum_sha256": {
            path.relative_to(INPUTS).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([native / row["compressed_file"] for row in manifest if row["family"] == "high_dispersion"]
                               + [fcc_native / row["compressed_file"] for row in fcc_manifest["cases"]])
        },
        "output_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("empirical_fwhm_records.json", "empirical_fwhm_summary.csv", "high_dispersion.csv", "fcc_resolution_summary.json")},
    }
    (output / "peak_resolution_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def regenerate(output):
    """Regenerate every supported scalar study statistic into a new directory."""
    output.mkdir(parents=True, exist_ok=False)
    confirmation(output)
    high_dispersion(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regenerate(args.output)
