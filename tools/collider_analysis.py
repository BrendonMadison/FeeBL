#!/usr/bin/env python3
"""Shared native e+e- spectrum adapter for GUI and paper. No new physics estimator."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "feebl.collider.analysis/1"
COLUMNS = "W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff".split()
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_BINS = 20000
ELECTRON_MASS_GEV = 0.00051099895


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bounded(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("missing, symlinked or oversized source: " + str(path))
    return path.read_text(encoding="utf-8")


def local_name(name):
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or name in (".", "..")
    ):
        raise ValueError("unsafe local artifact name")
    return name


def nominal_w(energies, full_angle):
    if len(energies) != 2 or any(not math.isfinite(v) or v <= ELECTRON_MASS_GEV for v in energies):
        raise ValueError("two finite beam energies above the electron mass are required")
    if not math.isfinite(full_angle) or not 0 <= full_angle < math.pi:
        raise ValueError("full crossing-angle magnitude must be in [0, pi)")
    e1, e2 = energies
    p1 = math.sqrt((e1 - ELECTRON_MASS_GEV) * (e1 + ELECTRON_MASS_GEV))
    p2 = math.sqrt((e2 - ELECTRON_MASS_GEV) * (e2 + ELECTRON_MASS_GEV))
    return math.sqrt(2 * ELECTRON_MASS_GEV**2 + 2 * (e1 * e2 + p1 * p2 * math.cos(full_angle)))


def parse_spectrum(text):
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("spectrum byte bound")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "#FEEBL-NATIVE 1.0":
        raise ValueError("not an admitted FEEBL native spectrum")
    meta = {}
    rows = []
    footer = None
    ended = False
    seen_single = set()
    critical = {
        "BLOCK",
        "COLUMNS",
        "underflow",
        "overflow",
        "mean_W",
        "rms_W",
        "n_eff_total",
        "energy_unit",
        "half_crossing_angle_rad",
        "frame",
        "momenta_carry_crossing_angle",
    }
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if ended:
            raise ValueError("data after spectrum footer")
        if line.startswith("#END "):
            match = re.fullmatch(r"#END pair_spectrum records (\d+)", line)
            if not match:
                raise ValueError("invalid spectrum footer")
            footer = int(match.group(1))
            ended = True
            continue
        if line.startswith("# "):
            continue  # Native prose headings are not metadata.
        if line.startswith("#"):
            parts = line[1:].split(None, 1)
            if len(parts) == 2:
                key, value = parts
                if key in critical:
                    if key in seen_single:
                        raise ValueError("duplicate native metadata: " + key)
                    seen_single.add(key)
                if key != "WARNING":
                    meta[key] = value
            continue
        if meta.get("BLOCK") != "pair_spectrum" or meta.get("COLUMNS", "").split() != COLUMNS:
            raise ValueError("unexpected native block or columns")
        parts = line.split()
        if len(parts) != 11:
            raise ValueError("truncated spectrum row")
        row = [float(x) for x in parts]
        if not all(math.isfinite(x) for x in row):
            raise ValueError("nonfinite spectrum row")
        lo, hi, dl, weight, *_ = row
        if not 0 <= lo < hi or dl <= 0 or weight <= 0 or row[5] < 0 or row[7] < 0 or row[10] < 0:
            raise ValueError("invalid spectrum bounds, weights or error fields")
        if rows and lo < rows[-1][1] - 1e-12:
            raise ValueError("overlapping or unordered spectrum bins")
        rows.append(row)
        if len(rows) > MAX_BINS:
            raise ValueError("spectrum bin bound")
    if meta.get("BLOCK") != "pair_spectrum" or meta.get("COLUMNS", "").split() != COLUMNS:
        raise ValueError("missing native block or column declaration")
    if footer is None or footer != len(rows):
        raise ValueError("missing/mismatched spectrum footer")
    if meta.get("energy_unit") != "GeV":
        raise ValueError("incompatible energy units")
    for key in ("underflow", "overflow", "rms_W", "n_eff_total"):
        if key not in meta:
            raise ValueError("missing spectrum statistic: " + key)
        value = float(meta[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid " + key)
        meta[key] = value
    match = re.fullmatch(r"(\S+) \+- (\S+)", meta.get("mean_W", ""))
    if not match:
        raise ValueError("missing/malformed mean_W")
    mean, err = map(float, match.groups())
    if not all(math.isfinite(x) and x >= 0 for x in (mean, err)):
        raise ValueError("invalid mean_W")
    meta["mean_W"] = mean
    meta["mean_W_error"] = err
    half = float(meta.get("half_crossing_angle_rad", "nan"))
    if not math.isfinite(half):
        raise ValueError("missing native crossing-angle receipt")
    meta["half_crossing_angle_rad"] = half
    return meta, rows


def parse_summary(text):
    values = [
        float(v) for v in re.findall(r"^lumi_ee\s*=\s*([+\-0-9.eE]+)\s+m\*\*\(-2\)", text, re.M)
    ]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] < 0:
        raise ValueError("expected one explicit per-crossing lumi_ee in m^-2")
    return values[0]


def analyze_native(
    spectrum_text,
    summary_text,
    energies,
    full_crossing_angle_rad,
    crossing_frequency_hz=None,
    normalization_scope="per_ip",
):
    if normalization_scope not in ("per_ip", "machine_total_unallocated"):
        raise ValueError("normalization_scope must be per_ip or machine_total_unallocated")
    wnom = nominal_w(energies, full_crossing_angle_rad)
    if crossing_frequency_hz is not None and (
        not math.isfinite(crossing_frequency_hz) or crossing_frequency_hz <= 0
    ):
        raise ValueError("explicit crossing frequency must be finite and positive")
    meta, rows = parse_spectrum(spectrum_text)
    total_summary = parse_summary(summary_text) if summary_text is not None else None
    total_bins = math.fsum(row[3] for row in rows)
    under, over = meta["underflow"], meta["overflow"]
    total = math.fsum([total_bins, under, over])
    factor = crossing_frequency_hz * 1e-4 if crossing_frequency_hz is not None else None
    finite_angle = full_crossing_angle_rad != 0 or meta["half_crossing_angle_rad"] != 0
    limitations = [
        "No seed-ensemble or reference uncertainty is inferred from the native weighted-record errors.",
        "Missing bin records mean no sampled records; no zero-density bins or error bars are invented.",
        "Shape normalization retains underflow and overflow in the denominator; display crops never renormalize.",
        "No ISR is added by this adapter; upstream physics switches remain the source of the modeled spectrum.",
        "Gaussian-core fit is unavailable; native full RMS is not a Gaussian-core width.",
    ]
    if finite_angle:
        limitations.insert(
            0,
            "FINITE CROSSING: native directions omit the design crossing rotation; native W is a head-on-direction approximation. Bins are preserved, not shifted to design W.",
        )
    angle_match = math.isclose(
        abs(2 * meta["half_crossing_angle_rad"]),
        full_crossing_angle_rad,
        rel_tol=1e-5,
        abs_tol=1e-10,
    )
    if not angle_match:
        limitations.insert(
            0,
            "Supplied design full angle differs from the native header; investigate the geometry receipt.",
        )
    bins = []
    for row in rows:
        lo, hi, dl, weight, *_ = row
        bins.append(
            dict(
                W_low_GeV=lo,
                W_high_GeV=hi,
                dL_dW_m2_per_crossing_GeV=dl,
                weight_m2_per_crossing=weight,
                dL_dW_cm2_s_GeV=dl * factor if factor is not None else None,
                shape_density_per_GeV=dl / total if total > 0 else None,
                uncertainty=None,
                native_columns=row,
            )
        )
    gaps = [
        dict(low_GeV=rows[i - 1][1], high_GeV=rows[i][0], status="NO_SAMPLED_RECORD")
        for i in range(1, len(rows))
        if rows[i][0] > rows[i - 1][1] + 1e-10
    ]
    integral_intervals = {}
    for fraction in (0.98, 0.99):
        cut = fraction * wnom
        lower = math.fsum(r[3] for r in rows if r[0] >= cut)
        upper = math.fsum(r[3] for r in rows if r[1] > cut)
        if rows and cut <= rows[-1][1]:
            lower += over
        upper += over
        if not rows or cut < rows[0][0]:
            upper += under
        integral_intervals[str(fraction)] = dict(
            threshold_GeV=cut,
            weight_m2_per_crossing_bounds=[lower, upper],
            fraction_of_total_bounds=[lower / total, upper / total] if total > 0 else None,
            definition="Bounds from whole/cut-straddling native bins and unresolved flow locations, including high side; not a statistical confidence interval",
        )
    relative_delta = (
        (total - total_summary) / total_summary if total_summary and total_summary > 0 else None
    )
    normalization_status = (
        "UNAVAILABLE_SUMMARY"
        if total_summary is None
        else (
            "MATCH_WITHIN_NATIVE_TEXT_ROUNDING"
            if math.isclose(total, total_summary, rel_tol=5e-5, abs_tol=1e-30)
            else "MISMATCH_REQUIRES_INVESTIGATION"
        )
    )
    return dict(
        schema_version=SCHEMA,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observable="e+e- luminosity spectrum",
        validation_status="UNVALIDATED_PHYSICS",
        definition=dict(
            native_W="Massive invariant mass of simulated incoming directions; no design crossing rotation in native header",
            design_W="sqrt(2*m_e^2 + 2*(E1*E2 + p1*p2*cos(full_crossing_angle)))",
            nominal_W_GeV=wnom,
            energy_GeV=list(energies),
            full_crossing_angle_rad=full_crossing_angle_rad,
            angle_receipt_matches_design=angle_match,
            finite_angle_approximation=finite_angle,
            native_frame=meta.get("frame"),
            native_momenta_carry_crossing_angle=meta.get("momenta_carry_crossing_angle"),
        ),
        normalization=dict(
            crossing_frequency_hz=crossing_frequency_hz,
            conversion_m2_to_cm2_s=factor,
            allocation_scope=normalization_scope,
            scope=(
                "per IP; explicitly supplied per-IP collision frequency"
                if normalization_scope == "per_ip"
                else "machine total with IP allocation unknown; never quote as per-IP luminosity"
            ),
            bin_sum_m2_per_crossing=total_bins,
            underflow_m2_per_crossing=under,
            overflow_m2_per_crossing=over,
            total_spectrum_m2_per_crossing=total,
            total_summary_m2_per_crossing=total_summary,
            total_L_cm2_s=(
                total_summary * factor if total_summary is not None and factor is not None else None
            ),
            comparison_status=normalization_status,
            relative_spectrum_minus_summary=relative_delta,
        ),
        native_statistics=dict(
            mean_W_GeV=meta["mean_W"],
            weighted_record_mean_error_GeV=meta["mean_W_error"],
            full_RMS_W_GeV=meta["rms_W"],
            weighted_record_n_eff=meta["n_eff_total"],
            gaussian_core_fit=None,
        ),
        high_energy_integral_bounds=integral_intervals,
        missing_record_gaps=gaps,
        bins=bins,
        plot=dict(
            full=dict(x_min_GeV=0, y_scale="log"),
            peak=dict(x_min_GeV=0.98 * wnom, y_scale="linear"),
            x_max_GeV=max([wnom * 1.005] + [r[1] for r in rows]),
            crop_renormalization=False,
        ),
        limitations=limitations,
        native_header=meta,
    )


def write_analysis(run, analysis):
    target = run / "collider_analysis.json"
    target.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    with (run / "ee_spectrum.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "W_low_GeV",
                "W_high_GeV",
                "dL_dW_m2_per_crossing_GeV",
                "weight_m2_per_crossing",
                "dL_dW_cm2_s_GeV",
                "shape_density_per_GeV",
            ]
        )
        for b in analysis["bins"]:
            w.writerow(
                [
                    b[k] if b[k] is not None else ""
                    for k in (
                        "W_low_GeV",
                        "W_high_GeV",
                        "dL_dW_m2_per_crossing_GeV",
                        "weight_m2_per_crossing",
                        "dL_dW_cm2_s_GeV",
                        "shape_density_per_GeV",
                    )
                ]
            )
    return analysis


def analyze_run(
    run_dir,
    energy_GeV,
    full_crossing_angle_rad,
    crossing_frequency_hz=None,
    spectrum_name="lumi.ee.out.dLdW.fne",
    summary_name="out.dat",
    normalization_scope="per_ip",
):
    run = Path(run_dir)
    spectrum = run / local_name(spectrum_name)
    summary = run / local_name(summary_name)
    analysis = analyze_native(
        read_bounded(spectrum),
        read_bounded(summary) if summary.exists() else None,
        energy_GeV,
        full_crossing_angle_rad,
        crossing_frequency_hz,
        normalization_scope,
    )
    analysis["sources"] = dict(
        spectrum=dict(path=spectrum.name, sha256=sha256(spectrum), bytes=spectrum.stat().st_size)
    )
    if summary.exists():
        analysis["sources"]["summary"] = dict(
            path=summary.name, sha256=sha256(summary), bytes=summary.stat().st_size
        )
    analysis["adapter_sha256"] = sha256(Path(__file__))
    return write_analysis(run, analysis)


def plot_analysis(path, output_dir=None):
    """Publication figure source is exactly the same JSON read by the browser."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    a = json.loads(path.read_text())
    out = Path(output_dir) if output_dir else path.parent
    out.mkdir(parents=True, exist_ok=True)
    per_second = a["normalization"]["crossing_frequency_hz"] is not None
    key = "dL_dW_cm2_s_GeV" if per_second else "dL_dW_m2_per_crossing_GeV"
    label = (
        r"dL/dW [cm$^{-2}$ s$^{-1}$ GeV$^{-1}$]"
        if per_second
        else r"dL/dW [m$^{-2}$ crossing$^{-1}$ GeV$^{-1}$]"
    )
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, mode in zip(axs, ("full", "peak")):
        for b in a["bins"]:
            ax.plot(
                [b["W_low_GeV"], b["W_high_GeV"]], [b[key], b[key]], color="#0072b2", linewidth=1.5
            )
        ax.set_xlim(a["plot"][mode]["x_min_GeV"], a["plot"]["x_max_GeV"])
        ax.set_yscale(a["plot"][mode]["y_scale"])
        ax.set_xlabel("Native W [GeV]")
        ax.set_ylabel(label)
        if mode == "peak":
            ax.set_ylim(bottom=0)
        ax.axvline(a["definition"]["nominal_W_GeV"], color="#d55e00", linestyle="--", linewidth=0.8)
        ax.set_title("Full spectrum" if mode == "full" else "Peak: W ≥ 0.98 Wnom")
        ax.grid(alpha=0.2)
    caption = "Unvalidated native spectrum; gaps = no sampled records; no uncertainty bands."
    caption += (
        " Per IP."
        if a["normalization"].get("allocation_scope", "per_ip") == "per_ip"
        else " Machine total; IP allocation unknown."
    )
    if a["definition"]["finite_angle_approximation"]:
        caption += " Finite-angle head-on-direction approximation."
    fig.suptitle(caption, fontsize=9)
    fig.savefig(out / "ee_spectrum.png", dpi=180)
    fig.savefig(out / "ee_spectrum.pdf")
    plt.close(fig)
    (out / "ee_spectrum_caption.txt").write_text(
        caption
        + "\n"
        + json.dumps(a["normalization"], indent=2)
        + "\nAnalysis SHA256: "
        + sha256(path)
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--energy", type=float, nargs=2)
    p.add_argument("--full-crossing-angle-rad", type=float)
    p.add_argument("--crossing-frequency-hz", type=float)
    p.add_argument("--spectrum", default="lumi.ee.out.dLdW.fne")
    p.add_argument("--summary", default="out.dat")
    p.add_argument(
        "--normalization-scope", choices=("per_ip", "machine_total_unallocated"), default="per_ip"
    )
    p.add_argument("--plots", action="store_true")
    p.add_argument("--plot-analysis", type=Path)
    p.add_argument("--plot-output", type=Path)
    a = p.parse_args()
    if a.plot_analysis:
        plot_analysis(a.plot_analysis, a.plot_output)
        return
    if a.run is None or a.energy is None or a.full_crossing_angle_rad is None:
        p.error("run, --energy and --full-crossing-angle-rad are required")
    result = analyze_run(
        a.run,
        a.energy,
        a.full_crossing_angle_rad,
        a.crossing_frequency_hz,
        a.spectrum,
        a.summary,
        a.normalization_scope,
    )
    if a.plots:
        plot_analysis(a.run / "collider_analysis.json", a.plot_output)
    print(
        json.dumps(
            dict(
                status="PARSED_NOT_PHYSICS_VALIDATED",
                bins=len(result["bins"]),
                normalization=result["normalization"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
