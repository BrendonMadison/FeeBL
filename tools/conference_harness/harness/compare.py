"""Seed-ensemble equivalence: no inference from individual correlated macroparticles."""

from __future__ import annotations

import math
from pathlib import Path
import random

from .common import (
    Invalid,
    authentic_artifacts,
    canonical,
    digest,
    finite,
    integer,
    keys,
    publication_scope,
    read_json,
    require,
    sha256,
)


def authenticate_bounds(contract, metric_names):
    """Authenticate scoped human-reviewed bound reports, not their scientific truth."""
    evidence = contract["uncertainty_evidence"]
    bounds = contract["uncertainty_bounds"]
    require(
        isinstance(evidence, dict) and set(evidence) == set(bounds),
        "Every uncertainty component needs its own authenticated evidence",
    )
    receipts = {}
    for component, declared in bounds.items():
        pointer = evidence[component]
        keys(pointer, ("path", "sha256"))
        path = Path(pointer["path"])
        require(
            path.is_absolute() and path.is_file() and sha256(path) == pointer["sha256"],
            f"Missing/hash-mismatching bound evidence: {component}",
        )
        report = read_json(path)
        keys(
            report,
            (
                "schema",
                "component",
                "bound",
                "unit",
                "metric_names",
                "method",
                "reviewer",
                "status",
                "zero_justification",
            ),
        )
        require(
            report["schema"] == "feebl.bound_evidence.v1" and report["component"] == component,
            "Bound evidence schema/component mismatch",
        )
        require(
            report["unit"] == "dimensionless" and finite(report["bound"]) == declared,
            "Bound report value/units differ from declared contract",
        )
        require(
            isinstance(report["metric_names"], list)
            and len(set(report["metric_names"])) == len(report["metric_names"])
            and set(report["metric_names"]) == set(metric_names),
            "Bound evidence does not cover all and only the registered metrics",
        )
        require(
            report["status"] == "QUALIFIED"
            and isinstance(report["method"], str)
            and report["method"].strip()
            and isinstance(report["reviewer"], str)
            and report["reviewer"].strip(),
            "Bound has not been methodically qualified and reviewed",
        )
        if declared == 0:
            require(
                isinstance(report["zero_justification"], str)
                and len(report["zero_justification"].strip()) >= 20,
                "Zero uncertainty is unknown unless explicitly and specifically justified",
            )
        else:
            require(
                report["zero_justification"] is None,
                "Positive bounds must not carry a contradictory zero justification",
            )
        receipts[component] = {
            "path": str(path),
            "sha256": pointer["sha256"],
            "bound": declared,
            "method": report["method"],
            "reviewer": report["reviewer"],
        }
    return receipts


def histogram(data, authenticate=True):
    keys(data, ("schema", "classification", "observable", "provenance", "seeds"))
    require(data["schema"] == "feebl.histogram_ensemble.v1", "Unknown histogram schema")
    require(
        data["classification"] in {"SIMULATION", "REFERENCE", "EXPERIMENT", "SYNTHETIC_ONLY"},
        "Unknown classification",
    )
    obs = data["observable"]
    keys(
        obs,
        (
            "name",
            "definition",
            "axes",
            "value_unit",
            "normalization",
            "frame",
            "beam_convention",
            "selection",
            "nominal_ecm_GeV",
            "sample_kind",
            "hemisphere_definition",
            "angle_convention",
            "observation",
            "cuts",
        ),
        ("peak_window_GeV",),
    )
    for name in (
        "name",
        "definition",
        "value_unit",
        "frame",
        "beam_convention",
        "selection",
        "sample_kind",
        "hemisphere_definition",
        "angle_convention",
    ):
        require(
            isinstance(obs[name], str) and obs[name].strip(), f"Missing observable metadata: {name}"
        )
    require(
        obs["sample_kind"]
        in {"ee_luminosity", "produced_background_pairs", "transport_final_states"},
        "Unrecognized sample kind; incoming luminosity pairs are not produced background pairs",
    )
    require(
        obs["normalization"] in {"absolute_per_crossing", "absolute_luminosity", "probability"},
        "Undefined normalization",
    )
    require(finite(obs["nominal_ecm_GeV"]) > 0, "Nominal invariant ECM must be positive")
    require(
        isinstance(obs["observation"], dict) and obs["observation"],
        "Observation plane/time/field metadata required (explicit not_applicable permitted)",
    )
    require(
        isinstance(obs["cuts"], dict) and obs["cuts"],
        "Explicit cuts, including a declared none, are required",
    )
    axes = obs["axes"]
    require(
        isinstance(axes, list) and 1 <= len(axes) <= 2,
        "Only fixed-lattice 1D or joint 2D histograms are supported",
    )
    sizes = []
    for axis in axes:
        keys(axis, ("name", "unit", "edges", "transform", "log_base", "reference_unit"))
        require(axis["name"] and axis["unit"], "Axis name/unit required")
        edges = axis["edges"]
        require(isinstance(edges, list) and len(edges) >= 2, "Axis has no bins")
        values = [finite(x, "edge") for x in edges]
        require(
            all(a < b for a, b in zip(values, values[1:])),
            "Edges must be finite and strictly increasing",
        )
        require(axis["transform"] in {"linear", "log"}, "Unknown axis transform")
        if axis["transform"] == "log":
            require(
                finite(axis["log_base"]) > 1
                and isinstance(axis["reference_unit"], str)
                and axis["reference_unit"],
                "Logarithms require base and dimensionful reference unit",
            )
            require(axis["unit"] == "dimensionless", "Logarithm axes must be dimensionless")
        else:
            require(
                axis["log_base"] is None and axis["reference_unit"] is None,
                "Linear axes cannot have implicit logarithm metadata",
            )
        sizes.append(len(edges) - 1)
    if "peak_window_GeV" in obs:
        window = obs["peak_window_GeV"]
        require(
            window == {"lower": 0.98 * obs["nominal_ecm_GeV"], "upper": None},
            "Peak window is fixed at 0.98 times nominal invariant ECM upwards, not a fitted or per-beam window",
        )
    require(isinstance(data["seeds"], list) and data["seeds"], "Empty seed ensemble")
    nbin = math.prod(sizes)
    seen = set()
    ess = []
    signed = False
    for record in data["seeds"]:
        keys(record, ("seed", "independent", "n_records", "sumw", "sumw2", "sum_abs_w"))
        seed = integer(record["seed"], "seed", 0)
        require(seed not in seen, "Duplicate ensemble seed")
        seen.add(seed)
        require(
            record["independent"] is True,
            "Independent-seed assertions are mandatory; macroparticles are not independent replicates",
        )
        integer(record["n_records"], "n_records", 1)
        for field in ("sumw", "sumw2", "sum_abs_w"):
            require(
                isinstance(record[field], list) and len(record[field]) == nbin,
                f"{field} length does not match configured bin lattice",
            )
            for value in record[field]:
                finite(value, field)
        for w, w2, aw in zip(record["sumw"], record["sumw2"], record["sum_abs_w"]):
            require(w2 >= 0 and aw >= abs(w), "Invalid sumw2/sum_abs_w")
            require(
                (w2 > 0 and aw > 0) or (w == 0 and w2 == 0 and aw == 0),
                "Nonzero count with zero second moment",
            )
            require(
                w2 <= aw * aw * (1 + 1e-12), "Second moment exceeds the squared absolute-weight sum"
            )
            signed |= aw > abs(w) * (1 + 1e-12) or w < 0
        total = sum(record["sumw"])
        second = sum(record["sumw2"])
        require(total > 0 and second > 0, "Zero/nonpositive total weight or variance information")
        require(
            sum(record["sum_abs_w"]) ** 2 <= record["n_records"] * second * (1 + 1e-9),
            "n_records inconsistent with the absolute-weight effective sample size",
        )
        ess.append(
            {
                "seed": seed,
                "signed_effective_sample_size": total * total / second,
                "absolute_effective_sample_size": sum(record["sum_abs_w"]) ** 2 / second,
            }
        )
    if authenticate:
        authentic_artifacts(data["provenance"])
    return {"sizes": sizes, "signed": signed, "ess": ess}


def quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return (
        values[lower] * (upper - index) + values[upper] * (index - lower)
        if upper != lower
        else values[lower]
    )


def _mean(vectors, indices=None):
    indices = list(range(len(vectors))) if indices is None else indices
    return [sum(vectors[k][j] for k in indices) / len(indices) for j in range(len(vectors[0]))]


def _shape(vector):
    total = sum(vector)
    require(
        total > 0 and all(x >= 0 for x in vector),
        "CDF/TV metrics require positive-weight probability distributions",
    )
    return [x / total for x in vector]


def _cdf(vector, sizes):
    if len(sizes) == 1:
        total = 0.0
        result = []
        for value in vector:
            total += value
            result.append(total)
        return result
    nx, ny = sizes  # row-major x outermost, y innermost
    result = [0.0] * len(vector)
    for x in range(nx):
        for y in range(ny):
            k = x * ny + y
            result[k] = (
                vector[k]
                + (result[k - ny] if x else 0)
                + (result[k - 1] if y else 0)
                - (result[k - ny - 1] if x and y else 0)
            )
    return result


def compare(candidate, reference, contract):
    """Return BLOCKED on incomplete provenance; never promote a noisy plot to PASS."""
    result = {
        "schema": "feebl.comparison_result.v1",
        "status": "BLOCKED",
        "scope": publication_scope(),
    }
    try:
        result.update(
            {
                "candidate_sha256": digest(candidate),
                "reference_sha256": digest(reference),
                "contract_sha256": digest(contract),
            }
        )
        ci = histogram(candidate)
        ri = histogram(reference)
        require(
            candidate["classification"] != "SYNTHETIC_ONLY"
            and reference["classification"] != "SYNTHETIC_ONLY",
            "Synthetic inputs cannot establish physical equivalence",
        )
        require(
            reference["classification"] in {"REFERENCE", "EXPERIMENT"},
            "Reference classification must be REFERENCE or EXPERIMENT",
        )
        require(
            canonical(candidate["observable"]) == canonical(reference["observable"]),
            "Observable, units, bin lattice, normalization or coordinate conventions differ",
        )
        keys(
            contract,
            (
                "schema",
                "registration",
                "observable_name",
                "relative_tolerance",
                "confidence_level",
                "bootstrap_replicates",
                "bootstrap_seed",
                "min_independent_seeds",
                "ensemble_design",
                "uncertainty_bounds",
                "uncertainty_evidence",
                "metrics",
                "distribution_tolerance",
            ),
        )
        require(contract["schema"] == "feebl.comparison_contract.v1", "Unknown comparison contract")
        keys(
            contract["registration"],
            ("campaign_id", "registered_utc", "locked_before_results", "reviewer"),
        )
        require(
            contract["registration"]["locked_before_results"] is True
            and all(
                contract["registration"][x] for x in ("campaign_id", "registered_utc", "reviewer")
            ),
            "Missing preregistration assertion; a hash does not prove historical preregistration",
        )
        require(
            contract["observable_name"] == candidate["observable"]["name"],
            "Contract names a different observable",
        )
        tolerance = finite(contract["relative_tolerance"])
        dtol = finite(contract["distribution_tolerance"])
        require(
            0 < tolerance <= 0.01 and 0 < dtol <= 0.01,
            "This publication gate requires at most 1% tolerances",
        )
        confidence = finite(contract["confidence_level"])
        require(0.95 <= confidence < 1, "At least 95% family confidence required")
        reps = integer(contract["bootstrap_replicates"], "bootstrap_replicates", 1000)
        require(reps <= 100000, "Excessive bootstrap request")
        rng = random.Random(integer(contract["bootstrap_seed"], "bootstrap_seed", 0))
        minimum = integer(contract["min_independent_seeds"], "min_independent_seeds", 8)
        require(
            min(len(candidate["seeds"]), len(reference["seeds"])) >= minimum,
            "Insufficient independent seeds; no particle-level bootstrap substitution",
        )
        bounds = contract["uncertainty_bounds"]
        keys(
            bounds,
            (
                "candidate_numerical",
                "reference_numerical",
                "reference_model",
                "reference_extraction",
            ),
        )
        require(
            all(0 <= finite(value) < 1 for value in bounds.values()),
            "Invalid non-statistical uncertainty bounds",
        )
        floor = sum(bounds.values())
        result["uncertainty_bounds"] = bounds
        result["conservative_added_relative_bound"] = floor
        require(
            floor < min(tolerance, dtol),
            "Declared numerical/reference uncertainty floor exhausts the precision budget",
        )
        require(
            not ci["signed"] and not ri["signed"],
            "Signed weights detected: ESS reported, but positive-measure CDF/TV equivalence is undefined; register an explicitly signed observable instead",
        )
        seeds_c = {x["seed"] for x in candidate["seeds"]}
        seeds_r = {x["seed"] for x in reference["seeds"]}
        paired = contract["ensemble_design"] == "paired_common_random_numbers"
        require(paired or contract["ensemble_design"] == "independent", "Unknown ensemble design")
        require(
            seeds_c == seeds_r if paired else seeds_c.isdisjoint(seeds_r),
            "Seed sets do not match the declared ensemble design",
        )
        c = [x["sumw"][:] for x in sorted(candidate["seeds"], key=lambda x: x["seed"])]
        r = [x["sumw"][:] for x in sorted(reference["seeds"], key=lambda x: x["seed"])]
        if candidate["observable"]["normalization"] == "probability":
            c, r = [_shape(v) for v in c], [_shape(v) for v in r]
        metrics = contract["metrics"]
        require(isinstance(metrics, list) and metrics, "Preregistered integral bands are required")
        names = set()
        for metric in metrics:
            keys(metric, ("name", "kind", "bins"))
            require(
                metric["kind"] == "integral" and metric["name"] not in names,
                "Duplicate/unsupported metric",
            )
            names.add(metric["name"])
            bins = metric["bins"]
            require(
                isinstance(bins, list) and bins and len(set(bins)) == len(bins),
                "Missing/duplicate metric bins",
            )
            require(
                all(type(x) is int and 0 <= x < len(c[0]) for x in bins),
                "Metric bin outside lattice",
            )
            if metric["name"] == "peak_top_2_percent":
                obs = candidate["observable"]
                require(
                    len(obs["axes"]) == 1
                    and obs["axes"][0]["unit"] == "GeV"
                    and "peak_window_GeV" in obs,
                    "Peak metric requires an energy lattice and fixed peak metadata",
                )
                edges = obs["axes"][0]["edges"]
                threshold = 0.98 * obs["nominal_ecm_GeV"]
                require(
                    threshold in edges,
                    "Peak threshold is not a configured bin edge; no silent partial-bin interpolation",
                )
                require(
                    bins == list(range(edges.index(threshold), len(edges) - 1)),
                    "Peak metric must include every configured bin at or above 0.98 nominal ECM",
                )
        result["uncertainty_evidence"] = authenticate_bounds(
            contract, [x["name"] for x in metrics] + ["total_variation", "max_joint_cdf_distance"]
        )
        nmetric = len(metrics) + 2
        tail = (1 - confidence) / (2 * nmetric)  # Bonferroni family interval
        require(
            reps * tail >= 5, "Too few bootstrap replicas for requested family tail probability"
        )

        def values(cv, rv):
            out = []
            for metric in metrics:
                denom = sum(rv[j] for j in metric["bins"])
                require(
                    denom > 0,
                    f"Reference band {metric['name']} is empty; do not use empty-bin relative errors",
                )
                out.append(sum(cv[j] for j in metric["bins"]) / denom - 1)
            cp, rp = _shape(cv), _shape(rv)
            out.extend(
                (
                    0.5 * sum(abs(x - y) for x, y in zip(cp, rp)),
                    max(abs(x - y) for x, y in zip(_cdf(cp, ci["sizes"]), _cdf(rp, ri["sizes"]))),
                )
            )
            return out

        observed = values(_mean(c), _mean(r))
        samples = [[] for _ in range(nmetric)]
        for _ in range(reps):
            ic = [rng.randrange(len(c)) for _ in c]
            ir = ic if paired else [rng.randrange(len(r)) for _ in r]
            for target, value in zip(samples, values(_mean(c, ic), _mean(r, ir))):
                target.append(value)
        result["metrics"] = []
        passes = []
        for index, (name, point, sample) in enumerate(
            zip(
                [x["name"] for x in metrics] + ["total_variation", "max_joint_cdf_distance"],
                observed,
                samples,
            )
        ):
            low, high = quantile(sample, tail), quantile(sample, 1 - tail)
            distribution = index >= len(metrics)
            accepted = (
                high + floor <= dtol
                if distribution
                else low - floor >= -tolerance and high + floor <= tolerance
            )
            passes.append(accepted)
            result["metrics"].append(
                {
                    "name": name,
                    "estimate": point,
                    "bootstrap_interval": [low, high],
                    "expanded_interval": [
                        max(0, low - floor) if distribution else low - floor,
                        high + floor,
                    ],
                    "pass": accepted,
                }
            )
        result.update(
            {
                "status": "NUMERICAL_EQUIVALENCE_PASS" if all(passes) else "FAIL_OR_INCONCLUSIVE",
                "confidence_family": confidence,
                "bootstrap_replicates": reps,
                "ensemble_design": contract["ensemble_design"],
                "effective_sample_sizes": {"candidate": ci["ess"], "reference": ri["ess"]},
                "signed_weights": False,
                "preregistration_is_a_reviewed_assertion_not_a_timestamp_authentication": True,
            }
        )
    except (Invalid, KeyError, TypeError, ValueError, OSError) as error:
        result["reason"] = str(error)
    return result
