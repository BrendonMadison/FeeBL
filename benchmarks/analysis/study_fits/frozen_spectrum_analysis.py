"""Analyze the opt-in design-axis PRE-ISR luminosity spectrum.

This module reads the scalar native companion, never the large frame JSON or a
rescaled legacy spectrum. Full RMS is the native unbinned centered moment.
Quantiles use a piecewise-uniform density within each native bin and report
resolution information. The beta/delta Gaussian fit integrates every native
bin, including all zero bins. A bounded runtime does not turn a failed or
unresolved fit into an admitted width.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
SPECTRUM_NAME = "lumi.ee.out.design_axis_dLdW.fne"
VENDOR = HERE / "vendor/collider_beta_peak_fit.py"
VENDOR_SHA256 = "e5da6c066d8199967ec94d50af7935de3f92f93b193540892e25c8e6f55ae441"
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_NATIVE_BINS = 250_000
MAX_FIT_BINS = 20_000
SCHEMA = "feebl.monochromatization.study-analysis/2"
NESTED_FITTER = HERE / "nested_beta_peak_fit.py"


def file_record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_design_spectrum(path):
    """Read complete nonuniform bins and authoritative unbinned moments."""
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink(),
        "Missing or linked design-axis scalar spectrum",
    )
    require(
        0 < path.stat().st_size <= MAX_SOURCE_BYTES,
        "Scalar spectrum exceeds the bounded source size",
    )
    header, rows, columns = {}, [], None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        require(
            f.readline().strip() == "#FEEBL-DESIGN-AXIS 1.0",
            "Not the design-axis companion schema",
        )
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                parts = text[1:].strip().split(None, 1)
                if not parts:
                    continue
                key, value = parts[0], parts[1] if len(parts) > 1 else ""
                if key == "COLUMNS":
                    require(columns is None, "Duplicate column declaration")
                    columns = value.split()
                else:
                    require(
                        key not in header or header[key] == value,
                        "Conflicting native header: " + key,
                    )
                    header[key] = value
                continue
            require(
                columns == ["W_low", "W_high", "weight", "weight2", "contributions"],
                "Unexpected design spectrum columns",
            )
            values = [float(x) for x in text.split()]
            require(
                len(values) == 5 and all(math.isfinite(v) for v in values),
                "Malformed native spectrum row",
            )
            rows.append(values)
            require(
                len(rows) <= MAX_NATIVE_BINS,
                "Native spectrum exceeds the bounded bin count",
            )
    require(rows, "Empty design-axis spectrum")
    data = np.asarray(rows, dtype=float)
    low, high, weight, weight2, counts = data.T
    require(np.all(low >= 0) and np.all(high > low), "Invalid bin edges")
    require(
        np.all(weight >= 0) and np.all(weight2 >= 0) and np.all(counts >= 0),
        "Negative native accumulation",
    )
    require(np.all(counts == np.floor(counts)), "Nonintegral native contribution count")
    require(
        np.allclose(low[1:], high[:-1], rtol=1e-12, atol=1e-12),
        "Missing or overlapping bins; empty native bins must be present",
    )
    numeric = {}
    for key in [
        "energy1_GeV",
        "energy2_GeV",
        "angle1_x_rad",
        "angle2_x_rad",
        "nominal_W_GeV",
        "underflow",
        "overflow",
        "total_weight",
        "total_weight2",
        "mean_W",
        "rms_W",
        "contributions",
    ]:
        require(key in header, "Missing native metadata: " + key)
        numeric[key] = float(header[key])
        require(math.isfinite(numeric[key]), "Nonfinite native metadata: " + key)
    require(header.get("weight_unit") == "m^-2_per_BX", "Unrecognized luminosity unit")
    require(header.get("energy_unit") == "GeV", "Unrecognized energy unit")
    require(
        header.get("weight2_unit") == "m^-4_per_BX2", "Unrecognized squared-weight unit"
    )
    require(
        header.get("ISR") == "pre_optional_ISR",
        "The requested observable must be PRE-ISR",
    )
    require(
        header.get("all_bins_written") == "yes",
        "Native companion does not declare complete bins",
    )
    require(
        header.get("contributions_unit") == "correlated_collision_pair_evaluations",
        "Unexpected native contribution semantics",
    )
    require(
        header.get("END") == f"design_axis_pair_spectrum records {len(rows)}",
        "Incomplete native spectrum termination",
    )
    require(
        int(header.get("bins", -1)) == len(rows), "Declared native bin count mismatch"
    )
    require(
        numeric["total_weight"] > 0 and numeric["rms_W"] >= 0,
        "Nonpositive luminosity or invalid native RMS",
    )
    require(
        all(
            numeric[k] >= 0
            for k in ["underflow", "overflow", "total_weight2", "contributions"]
        ),
        "Negative total or flow",
    )
    total = math.fsum(weight.tolist())
    closure = (total + numeric["underflow"] + numeric["overflow"]) / numeric[
        "total_weight"
    ] - 1
    require(abs(closure) <= 5e-6, "Native histogram/flow luminosity does not close")
    require(
        numeric["contributions"] == math.floor(numeric["contributions"]),
        "Nonintegral total contribution count",
    )
    require(
        float(counts.sum()) <= numeric["contributions"],
        "Binned contributions exceed total count",
    )
    require(
        np.all((weight > 0) == (counts > 0)),
        "Bin weights and positive-contribution counts disagree",
    )
    require(
        np.all(weight * weight <= counts * weight2 * (1 + 1e-10)),
        "Native bin squared-weight identity failed",
    )
    squared_sum = math.fsum(weight2.tolist())
    require(
        squared_sum <= numeric["total_weight2"] * (1 + 5e-6),
        "Binned squared weights exceed native total",
    )
    if numeric["underflow"] + numeric["overflow"] == 0:
        require(
            float(counts.sum()) == numeric["contributions"],
            "Native contribution counts do not close",
        )
        require(
            math.isclose(
                squared_sum, numeric["total_weight2"], rel_tol=5e-6, abs_tol=0
            ),
            "Native squared-weight accumulation does not close",
        )
    for key in ["centered_reference_W", "sum_weight_delta_W", "sum_weight_delta_W2"]:
        require(key in header, "Missing native centered moment: " + key)
        numeric[key] = float(header[key])
        require(math.isfinite(numeric[key]), "Nonfinite native centered moment: " + key)
    delta = numeric["sum_weight_delta_W"] / numeric["total_weight"]
    reconstructed_mean = numeric["centered_reference_W"] + delta
    reconstructed_rms = math.sqrt(
        max(
            0.0,
            numeric["sum_weight_delta_W2"] / numeric["total_weight"] - delta * delta,
        )
    )
    require(
        math.isclose(
            reconstructed_mean, numeric["mean_W"], rel_tol=1e-12, abs_tol=1e-12
        ),
        "Native centered mean does not close",
    )
    require(
        math.isclose(reconstructed_rms, numeric["rms_W"], rel_tol=1e-12, abs_tol=1e-12),
        "Native centered RMS does not close",
    )
    return dict(
        low=low,
        high=high,
        weight=weight,
        weight2=weight2,
        counts=counts,
        metadata=header,
        numeric=numeric,
        closure_relative=closure,
        histogram_weight=total,
        source=file_record(path),
    )


def _quantile_function(low, high, probability):
    occupied = probability > 0
    lo, hi, p = low[occupied], high[occupied], probability[occupied]
    cdf = np.concatenate(([0.0], np.cumsum(p)))
    cdf[-1] = 1.0

    def inverse(q, side="left"):
        q = np.asarray(q, dtype=float)
        indices = np.minimum(np.searchsorted(cdf, q, side=side) - 1, len(p) - 1)
        indices = np.maximum(indices, 0)
        fraction = np.clip((q - cdf[indices]) / p[indices], 0, 1)
        value = lo[indices] + fraction * (hi[indices] - lo[indices])
        return value

    return inverse, cdf, lo, hi


def quantile_widths(low, high, weight):
    total = math.fsum(np.asarray(weight).tolist())
    require(total > 0, "No luminosity in recorded bins")
    p = np.asarray(weight, dtype=float) / total
    inverse, cdf, lo, hi = _quantile_function(low, high, p)
    output = {
        "estimator": "native-bin piecewise-uniform luminosity density; no event-count interpretation",
        "median_W_GeV": float(inverse(0.5)),
        "central": {},
        "shortest": {},
    }
    for mass in [0.68, 0.90]:
        q1, q2 = (1 - mass) / 2, (1 + mass) / 2
        left, right = map(float, inverse([q1, q2]))
        use = (high >= left) & (low <= right)
        width = right - left
        output["central"][str(mass)] = dict(
            probability=mass,
            low_W_GeV=left,
            high_W_GeV=right,
            width_GeV=width,
            gaussian_equivalent_sigma_GeV=width / (2 * float(ndtri(q2))),
            largest_native_bin_width_GeV=float(np.max((high - low)[use])),
        )
        # The quantile function is piecewise linear. A minimum interval occurs
        # where either endpoint lies at a CDF knot, including support endpoints.
        q = np.unique(
            np.clip(np.concatenate(([0.0, 1 - mass], cdf, cdf - mass)), 0, 1 - mass)
        )
        lefts, rights = inverse(q, side="right"), inverse(q + mass, side="left")
        index = int(np.argmin(rights - lefts))
        output["shortest"][str(mass)] = dict(
            probability=mass,
            low_W_GeV=float(lefts[index]),
            high_W_GeV=float(rights[index]),
            width_GeV=float(rights[index] - lefts[index]),
            lower_cumulative_probability=float(q[index]),
        )
    return output


def fractions_above(low, high, weight, nominal):
    total = math.fsum(np.asarray(weight).tolist())
    results = {}
    for fraction in [0.98, 0.99, 0.999]:
        threshold = nominal * fraction
        full = float(np.sum(weight[low >= threshold]))
        crossing = (low < threshold) & (high > threshold)
        unknown = float(np.sum(weight[crossing]))
        estimate = full + float(
            np.sum(
                weight[crossing]
                * (high[crossing] - threshold)
                / (high[crossing] - low[crossing])
            )
        )
        results[str(fraction)] = dict(
            threshold_W_GeV=threshold,
            fraction=estimate / total,
            within_bin_bounds=[full / total, (full + unknown) / total],
            estimator="piecewise-uniform native bin density; bounds allow arbitrary within-bin positions",
        )
    return results


def resonance_score(low, high, weight, mass=125.0, width=0.0041):
    """Peak-normalized narrow Breit-Wigner in W; PRE-ISR diagnostic only."""
    require(mass > 0 and width > 0, "Invalid resonance parameters")
    a = width / 2
    average = (
        a * (np.arctan((high - mass) / a) - np.arctan((low - mass) / a)) / (high - low)
    )
    kernel_lo = a * a / ((low - mass) ** 2 + a * a)
    kernel_hi = a * a / ((high - mass) ** 2 + a * a)
    minimum = np.minimum(kernel_lo, kernel_hi)
    maximum = np.maximum(kernel_lo, kernel_hi)
    maximum[(low <= mass) & (high >= mass)] = 1.0
    total = math.fsum(np.asarray(weight).tolist())
    score = float(np.dot(weight, average) / total)
    return dict(
        available=True,
        mass_GeV=mass,
        width_GeV=width,
        peak_normalized_overlap=score,
        effective_luminosity_m2_per_BX=score * total,
        within_bin_overlap_bounds=[
            float(np.dot(weight, minimum) / total),
            float(np.dot(weight, maximum) / total),
        ],
        kernel="(Gamma/2)^2 / ((W-M)^2 + (Gamma/2)^2)",
        stage="PRE-ISR; narrow-resonance approximation in W",
        interpretation="Luminosity-spectrum overlap only; no ISR convolution, hard-process normalization, branching fraction, detector response, or Higgs-event forecast",
        estimator="exact kernel integral against piecewise-uniform density in every native bin",
    )


class FitBudgetExceeded(RuntimeError):
    pass


def _chunked_beta(
    base, low, high, *args, chunk_bins=1024, check=lambda: None, **kwargs
):
    """Evaluate unchanged native-bin integrals with bounded working arrays."""
    check()
    low, high = np.asarray(low), np.asarray(high)
    if low.ndim != 1 or len(low) <= chunk_bins:
        return base(low, high, *args, **kwargs)
    result = np.empty(low.shape, dtype=float)
    for start in range(0, len(low), chunk_bins):
        check()
        result[start : start + chunk_bins] = base(
            low[start : start + chunk_bins],
            high[start : start + chunk_bins],
            *args,
            **kwargs,
        )
    return result


def fit_core(
    low,
    high,
    weight,
    nominal,
    *,
    reference_sigma=None,
    max_seconds=180.0,
    chunk_bins=1024,
):
    """Reuse full-bin estimator with bounded quadrature memory/time."""
    require(
        file_record(VENDOR)["sha256"] == VENDOR_SHA256,
        "Vendored beta fitter hash mismatch",
    )
    if len(low) > MAX_FIT_BINS:
        return dict(
            available=False,
            qualified=False,
            status="analysis_resource_limit",
            reason="This scalar fit retains the original 20000-bin bound; choose the declared nonuniform native mesh rather than omit bins.",
            native_bins=len(low),
            limit=MAX_FIT_BINS,
        )
    require(max_seconds > 0 and 1 <= chunk_bins <= 8192, "Invalid fit resource bound")
    spec = importlib.util.spec_from_file_location(
        "study_beta_" + uuid.uuid4().hex, VENDOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_beta, base_gaussian = module.beta_bins, module.gaussian_bins
    started = time.monotonic()

    def check():
        if time.monotonic() - started > max_seconds:
            raise FitBudgetExceeded(
                "Core-fit runtime budget exceeded; no qualified width is assigned"
            )

    def gaussian(*args, **kwargs):
        check()
        return base_gaussian(*args, **kwargs)

    def beta(lo, hi, *args, **kwargs):
        return _chunked_beta(
            base_beta, lo, hi, *args, chunk_bins=chunk_bins, check=check, **kwargs
        )

    module.beta_bins, module.gaussian_bins = beta, gaussian
    histogram = dict(
        underflow=0.0,
        overflow=0.0,
        total_weight=math.fsum(weight.tolist()),
        bins=[
            dict(low=float(a), high=float(b), weight=float(w))
            for a, b, w in zip(low, high, weight)
        ],
    )
    try:
        nested_spec = importlib.util.spec_from_file_location(
            "study_nested_" + uuid.uuid4().hex, NESTED_FITTER
        )
        nested = importlib.util.module_from_spec(nested_spec)
        nested_spec.loader.exec_module(nested)
        result = nested.fit_histogram(
            module, histogram, float(nominal), reference_sigma
        )
    except FitBudgetExceeded as error:
        result = copy.deepcopy(getattr(module, "study_nested_progress", {}))
        result.update(
            available=False,
            reason=str(error),
            candidate=None,
            candidate_status="runtime_limit",
        )
    except (
        ValueError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as error:
        result = copy.deepcopy(getattr(module, "study_nested_progress", {}))
        result.update(
            available=False,
            reason="Study fit numerical failure: " + str(error),
            candidate=None,
            candidate_status="numerical_failure",
        )
    # Numerical curves are reconstructible from the parameters and native bins;
    # omit the large curve list from scalar study summaries.
    if isinstance(result.get("candidate"), dict):
        result["candidate"].pop("curve", None)
    result.update(
        qualified=bool(result["available"]),
        native_bins=len(low),
        elapsed_seconds=time.monotonic() - started,
        resource_limits=dict(
            max_seconds=max_seconds,
            quadrature_chunk_bins=chunk_bins,
            max_fit_bins=MAX_FIT_BINS,
        ),
        retained_empty_bins=int(np.count_nonzero(weight == 0)),
        implementation=file_record(VENDOR),
        study_nested_implementation=file_record(NESTED_FITTER),
        interpretation="Descriptive beta/delta Gaussian response width; not full RMS, a significance test, or a matched-control narrowing factor",
    )
    return result


def analyze_run(
    native_directory,
    nominalW=125.0,
    *,
    fit=True,
    fit_budget_seconds=180.0,
    reference_sigma=None,
    resonance=True,
    resonance_mass=125.0,
    resonance_width=0.0041,
    repetition_Hz=None,
):
    """Return luminosity, native full RMS, robust widths and qualified core fit.

    Only the design-axis PRE-ISR companion is accepted. Its kinematics correct
    the reported invariant energy under the stated design-axis reconstruction;
    this adapter does not validate or replace the legacy collision dynamics.
    """
    directory = Path(native_directory).resolve()
    require(directory.is_dir(), "Native output directory does not exist")
    h = load_design_spectrum(directory / SPECTRUM_NAME)
    n, low, high, weight = h["numeric"], h["low"], h["high"], h["weight"]
    require(
        math.isfinite(nominalW) and nominalW > 0, "Invalid nominal collision energy"
    )
    require(
        math.isclose(n["nominal_W_GeV"], nominalW, rel_tol=1e-7, abs_tol=1e-6),
        "Requested nominal energy disagrees with native design-axis metadata",
    )
    flow = n["underflow"] + n["overflow"]
    source_files = [
        h["source"],
        file_record(Path(__file__)),
        file_record(VENDOR),
        file_record(NESTED_FITTER),
    ]
    for filename in ["out.dat", "input.card", "receipt.json", "run_receipt.json"]:
        if (directory / filename).is_file():
            source_files.append(file_record(directory / filename))
    result = dict(
        schema=SCHEMA,
        native_directory=directory.name,
        nominal_W_GeV=nominalW,
        energy_definition="design-axis reconstructed invariant W, PRE-ISR",
        collision_dynamics="The design-axis observable does not replace the underlying head-on-frame collision-dynamics approximation.",
        native_metadata=h["metadata"],
        luminosity=dict(
            per_crossing_m2=n["total_weight"], per_crossing_cm2=n["total_weight"] / 1e4
        ),
        full_spectrum=dict(
            available=True,
            qualified=True,
            mean_W_GeV=n["mean_W"],
            sigma_W_GeV=n["rms_W"],
            variance_W_GeV2=n["rms_W"] ** 2,
            estimator="native unbinned centered moments over all accepted PRE-ISR collision contributions; including recorded flows",
            qualification="Availability and arithmetic provenance; not a claim of statistical or discretization convergence",
        ),
        histogram=dict(
            bins=len(low),
            occupied_bins=int(np.count_nonzero(weight)),
            empty_bins=int(np.count_nonzero(weight == 0)),
            min_W_GeV=float(low[0]),
            max_W_GeV=float(high[-1]),
            min_bin_width_GeV=float(np.min(high - low)),
            max_bin_width_GeV=float(np.max(high - low)),
            underflow_weight=n["underflow"],
            overflow_weight=n["overflow"],
            flow_fraction=flow / n["total_weight"],
            weight_closure_relative=h["closure_relative"],
        ),
        source_files=source_files,
    )
    if repetition_Hz is not None:
        require(
            math.isfinite(repetition_Hz) and repetition_Hz > 0,
            "Invalid crossing frequency",
        )
        result["luminosity"].update(
            crossings_per_second=repetition_Hz,
            per_second_cm2=n["total_weight"] / 1e4 * repetition_Hz,
        )
    robust = quantile_widths(low, high, weight)
    robust.update(
        qualified=flow == 0,
        normalization="recorded in-range luminosity",
        flow_qualification=None
        if flow == 0
        else "Flows outside the histogram are excluded; these are conditional widths",
    )
    result["quantile_widths"] = robust
    result["high_energy_fractions"] = dict(
        qualified=flow == 0,
        normalization="recorded in-range luminosity",
        values=fractions_above(low, high, weight, nominalW),
    )
    if not fit:
        core = dict(
            available=False,
            qualified=False,
            status="not_requested",
            reason="Core fit not requested for this analysis",
        )
    elif flow > 0:
        core = dict(
            available=False,
            qualified=False,
            status="unresolved_flow",
            reason="Positive luminosity flow lacks coordinates for a full-spectrum fit",
        )
    else:
        core = fit_core(
            low,
            high,
            weight,
            nominalW,
            reference_sigma=reference_sigma,
            max_seconds=fit_budget_seconds,
        )
    result["core_beta_gaussian_fit"] = core
    result["resonance_overlap"] = (
        resonance_score(low, high, weight, resonance_mass, resonance_width)
        if resonance
        else {"available": False, "reason": "Not requested"}
    )
    if resonance:
        result["resonance_overlap"].update(
            qualified=flow == 0,
            normalization="recorded in-range luminosity",
            flow_qualification=None
            if flow == 0
            else "Flows excluded; the overlap is conditional and not a full-spectrum score",
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native_directory")
    p.add_argument("--nominal-W", type=float, default=125.0)
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--fit-budget-seconds", type=float, default=180.0)
    p.add_argument("--reference-sigma", type=float)
    p.add_argument("--no-resonance", action="store_true")
    p.add_argument("--resonance-mass", type=float, default=125.0)
    p.add_argument("--resonance-width", type=float, default=0.0041)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = analyze_run(
        args.native_directory,
        args.nominal_W,
        fit=not args.no_fit,
        fit_budget_seconds=args.fit_budget_seconds,
        reference_sigma=args.reference_sigma,
        resonance=not args.no_resonance,
        resonance_mass=args.resonance_mass,
        resonance_width=args.resonance_width,
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
