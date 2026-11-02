"""Descriptive, bin-integrated Gaussian fit of a luminosity peak.

The squared density residual is a shape discrepancy, not a statistical chi-square.
Native luminosity weights are not event counts and no variance estimate is inferred.
"""

import math
import copy
import hashlib
from pathlib import Path

SCHEMA = "feebl.collider.gaussian-peak-fit/1"
ESTIMATOR = "bin-integrated Gaussian density least-squares shape fit; profiled amplitude"
RULES = dict(
    minimum_occupied_bins=6,
    minimum_sigma_over_bin_width=1.5,
    maximum_relative_l2_residual=0.25,
    minimum_positive_bins_each_side=2,
    maximum_sigma_over_fit_span=0.25,
    minimum_parameter_boundary_fraction=1e-4,
)
VERIFICATION_TOLERANCE = dict(
    mean_absolute_fraction_of_nominal=1e-9,
    sigma_relative=2e-6,
    area_relative=2e-6,
    residual_absolute=2e-7,
    dimensionless_quality_relative=3e-6,
    dimensionless_quality_absolute=2e-7,
    optimizer_counts="recorded provenance; not a cross-platform equality condition",
)
SQRT2 = math.sqrt(2.0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def gaussian_bin_probability(lo, hi, mu, sigma):
    a = (lo - mu) / (sigma * SQRT2)
    b = (hi - mu) / (sigma * SQRT2)
    if a >= 0:
        return 0.5 * (math.erfc(a) - math.erfc(b))
    if b <= 0:
        return 0.5 * (math.erfc(-b) - math.erfc(-a))
    return 0.5 * (math.erf(b) - math.erf(a))


def profile(bins, mu, sigma):
    """Profile area against normalized luminosity density, integrating each bin."""
    sw = math.fsum(w for _, _, w in bins)
    y = [w / sw for _, _, w in bins]
    q = [gaussian_bin_probability(a, b, mu, sigma) for a, b, _ in bins]
    yy = math.fsum(p * p / (b - a) for p, (a, b, _) in zip(y, bins))
    qq = math.fsum(p * p / (b - a) for p, (a, b, _) in zip(q, bins))
    pq = math.fsum(p * r / (b - a) for p, r, (a, b, _) in zip(y, q, bins))
    if qq <= 0:
        return math.inf, 0.0
    area = pq / qq
    residual = math.fsum((p - area * r) ** 2 / (b - a) for p, r, (a, b, _) in zip(y, q, bins)) / yy
    return residual, area


def minimize_shape(bins):
    """Deterministic bounded multi-start Nelder-Mead; two dimensionless parameters."""
    lo = bins[0][0]
    hi = bins[-1][1]
    span = hi - lo
    width = min(b - a for a, b, _ in bins)
    sigma_lo = width / 4
    sigma_hi = span / 2
    logratio = math.log(sigma_hi / sigma_lo)
    sw = math.fsum(w for _, _, w in bins)
    mode = max(bins, key=lambda b: b[2] / (b[1] - b[0]))
    mode = (mode[0] + mode[1]) / 2
    mean = lo + math.fsum(((a + b) / 2 - lo) * w / sw for a, b, w in bins)
    evaluations = 0

    def unpack(x):
        return lo + x[0] * span, sigma_lo * math.exp(x[1] * logratio)

    def evaluate(x):
        nonlocal evaluations
        x = tuple(min(1.0, max(0.0, v)) for v in x)
        evaluations += 1
        mu, sigma = unpack(x)
        loss, area = profile(bins, mu, sigma)
        return (loss, x, area)

    seeds = []
    for mu in (mode, mean):
        for k in range(13):
            seeds.append(evaluate(((mu - lo) / span, k / 12)))
    seeds.sort(key=lambda p: (p[0], p[1]))
    solutions = []
    for seed in seeds[:3]:
        x = seed[1]
        simplex = [
            seed,
            evaluate((x[0] + (0.02 if x[0] < 0.96 else -0.02), x[1])),
            evaluate((x[0], x[1] + (0.03 if x[1] < 0.94 else -0.03))),
        ]
        converged = False
        for iteration in range(240):
            simplex.sort(key=lambda p: (p[0], p[1]))
            best, middle, worst = simplex
            distance = max(abs(p[1][k] - best[1][k]) for p in simplex for k in (0, 1))
            if distance < 1e-8 and worst[0] - best[0] < 1e-11:
                converged = True
                break
            center = tuple((best[1][k] + middle[1][k]) / 2 for k in (0, 1))
            reflected = evaluate(tuple(2 * center[k] - worst[1][k] for k in (0, 1)))
            if reflected[0] < best[0]:
                expanded = evaluate(
                    tuple(center[k] + 2 * (reflected[1][k] - center[k]) for k in (0, 1))
                )
                simplex[2] = min((reflected, expanded), key=lambda p: (p[0], p[1]))
            elif reflected[0] < middle[0]:
                simplex[2] = reflected
            else:
                outside = reflected[0] < worst[0]
                target = reflected if outside else worst
                contracted = evaluate(
                    tuple(center[k] + 0.5 * (target[1][k] - center[k]) for k in (0, 1))
                )
                if contracted[0] < target[0]:
                    simplex[2] = contracted
                else:
                    simplex = [
                        best,
                        *[
                            evaluate(
                                tuple(best[1][k] + 0.5 * (p[1][k] - best[1][k]) for k in (0, 1))
                            )
                            for p in (middle, worst)
                        ],
                    ]
        simplex.sort(key=lambda p: (p[0], p[1]))
        solutions.append((simplex[0], converged, iteration + 1))
    best, converged, iterations = min(solutions, key=lambda r: (r[0][0], r[0][1]))
    mu, sigma = unpack(best[1])
    return dict(
        mean=mu,
        sigma=sigma,
        relative_l2_residual=math.sqrt(best[0]),
        area=best[2],
        converged=converged,
        iterations=iterations,
        evaluations=evaluations,
        parameter_position=list(best[1]),
        sigma_bounds=[sigma_lo, sigma_hi],
    )


def fit_histogram(histogram, nominal, reference=None, lower_fraction=0.98, identity=None):
    require(finite(nominal) and nominal > 0, "invalid peak nominal energy")
    require(finite(lower_fraction) and 0 < lower_fraction < 1, "invalid peak threshold fraction")
    requested = lower_fraction * nominal
    result = dict(
        available=False,
        reason=None,
        estimator=ESTIMATOR,
        mean_W_GeV=None,
        sigma_W_GeV=None,
        gaussian_area_native=None,
        gaussian_area_over_selected_luminosity=None,
        input_reference_ratio=None,
        nominal_W_GeV=nominal,
        requested_lower_W_GeV=requested,
        fit_lower_W_GeV=None,
        fit_upper_W_GeV=None,
        selected_bin_start=None,
        selected_bin_stop=None,
        selected_luminosity_weight=0.0,
        occupied_bins=0,
        selected_bins=0,
        relative_l2_residual=None,
        quality=None,
        solver=None,
        candidate=None,
        candidate_display_allowed=False,
        candidate_status="unavailable",
    )
    if identity:
        result.update(identity)

    def fail(reason):
        result["reason"] = reason
        if not result["candidate_display_allowed"]:
            result["candidate"] = None
        return result

    if histogram is None:
        return fail("Native peak luminosity histogram unavailable.")
    bins = histogram.get("bins")
    require(isinstance(bins, list) and 0 < len(bins) <= 20000, "invalid peak histogram bin bound")
    previous = None
    for b in bins:
        a, z, w = [b.get(k) for k in ("low", "high", "weight")]
        require(
            all(finite(v) for v in (a, z, w)) and 0 <= a < z and w >= 0,
            "invalid peak histogram bin",
        )
        if previous is not None:
            require(
                math.isclose(a, previous, rel_tol=1e-12, abs_tol=1e-14 * nominal),
                "gapped peak histogram",
            )
        previous = z
    require(
        all(
            finite(histogram.get(k)) and histogram[k] >= 0
            for k in ("underflow", "overflow", "total_weight")
        ),
        "invalid peak histogram flow",
    )
    require(
        math.isclose(
            math.fsum(b["weight"] for b in bins) + histogram["underflow"] + histogram["overflow"],
            histogram["total_weight"],
            rel_tol=1e-8,
            abs_tol=0,
        ),
        "peak luminosity closure",
    )
    selected = [
        i
        for i, b in enumerate(bins)
        if b["low"] >= requested or math.isclose(b["low"], requested, rel_tol=1e-12)
    ]
    if not selected:
        return fail("No whole native bins in the requested peak window.")
    start = selected[0]
    stop = selected[-1] + 1
    fitbins = [(b["low"] / nominal, b["high"] / nominal, b["weight"]) for b in bins[start:stop]]
    weight = math.fsum(b[2] for b in fitbins)
    occupied = sum(b[2] > 0 for b in fitbins)
    result.update(
        selected_bin_start=start,
        selected_bin_stop=stop,
        selected_bins=len(fitbins),
        occupied_bins=occupied,
        selected_luminosity_weight=weight,
        fit_lower_W_GeV=bins[start]["low"],
        fit_upper_W_GeV=bins[stop - 1]["high"],
    )
    if histogram["overflow"] > 0:
        return fail("Positive upper-flow luminosity lacks exported peak coordinates.")
    if weight <= 0 or occupied < RULES["minimum_occupied_bins"]:
        return fail("Peak fit needs positive luminosity in at least six native bins.")
    if len(fitbins) < 8:
        return fail("Peak fit window has fewer than eight native bins.")
    fitted = minimize_shape(fitbins)
    mu = fitted["mean"]
    sigma = fitted["sigma"]
    span = fitbins[-1][1] - fitbins[0][0]
    width = max(b - a for a, b, _ in fitbins if b >= mu - 2 * sigma and a <= mu + 2 * sigma)
    left = sum(w > 0 and (a + b) / 2 < mu and (a + b) / 2 >= mu - 2 * sigma for a, b, w in fitbins)
    right = sum(
        w > 0 and (a + b) / 2 >= mu and (a + b) / 2 <= mu + 2 * sigma for a, b, w in fitbins
    )
    quality = dict(
        sigma_over_bin_width=sigma / width,
        sigma_over_fit_span=sigma / span,
        positive_bins_left=left,
        positive_bins_right=right,
        bracketed_one_sigma=mu - sigma > fitbins[0][0] and mu + sigma < fitbins[-1][1],
        interior_parameters=all(
            RULES["minimum_parameter_boundary_fraction"]
            < v
            < 1 - RULES["minimum_parameter_boundary_fraction"]
            for v in fitted["parameter_position"]
        ),
    )
    result.update(
        relative_l2_residual=fitted["relative_l2_residual"],
        quality=quality,
        candidate=dict(
            mean_W_GeV=mu * nominal,
            sigma_W_GeV=sigma * nominal,
            gaussian_area_native=fitted["area"] * weight,
            gaussian_area_over_selected_luminosity=fitted["area"],
            input_reference_ratio=(
                reference / (sigma * nominal) if finite(reference) and reference > 0 else None
            ),
        ),
        solver={
            k: fitted[k]
            for k in (
                "converged",
                "iterations",
                "evaluations",
                "parameter_position",
                "sigma_bounds",
            )
        },
    )
    if not fitted["converged"]:
        return fail("Deterministic peak-shape optimizer did not converge.")
    if not quality["interior_parameters"]:
        return fail("Peak fit reached a parameter search boundary.")
    if quality["sigma_over_bin_width"] < RULES["minimum_sigma_over_bin_width"]:
        return fail("Gaussian width is not resolved by native peak bins.")
    if quality["sigma_over_fit_span"] > RULES["maximum_sigma_over_fit_span"]:
        return fail("Gaussian width is too broad for the declared peak fit window.")
    if (
        not quality["bracketed_one_sigma"]
        or min(left, right) < RULES["minimum_positive_bins_each_side"]
    ):
        return fail("The native peak is not sampled on both sides of the fitted center.")
    result.update(candidate_display_allowed=True, candidate_status="resolved_shape_mismatch")
    if result["relative_l2_residual"] > RULES["maximum_relative_l2_residual"]:
        return fail(
            "Peak shape is not adequately represented by one Gaussian under the declared residual rule."
        )
    result.update(
        available=True,
        reason=None,
        mean_W_GeV=mu * nominal,
        sigma_W_GeV=sigma * nominal,
        candidate_status="admitted",
        gaussian_area_native=fitted["area"] * weight,
        gaussian_area_over_selected_luminosity=fitted["area"],
        input_reference_ratio=(
            reference / (sigma * nominal) if finite(reference) and reference > 0 else None
        ),
    )
    return result


def build(diagnostics, reference, source_reference):
    """Keep integrated and interval fits distinct; never average fitted widths."""
    from collider_peak_diagnostics import validate_all

    validate_all(diagnostics)

    def make(record, is_total=False):
        identity = {k: record[k] for k in ("diagonal_start", "diagonal_end")}
        if not is_total:
            identity["id"] = record["id"]
        peak = record["peak"]
        s = peak["selection"]
        nominal = s["nominal_W_GeV"] if s else None
        fit = fit_histogram(
            peak["luminosity_spectrum"] if peak["available"] else None,
            nominal or 1.0,
            reference,
            identity=identity,
        )
        if nominal is None:
            fit.update(nominal_W_GeV=None, requested_lower_W_GeV=None)
        if not peak["available"]:
            fit["reason"] = "Native peak unavailable: " + peak["reason"]
        return fit

    return dict(
        schema_version=SCHEMA,
        estimator=ESTIMATOR,
        source_reference=copy.deepcopy(source_reference),
        selection=copy.deepcopy(diagnostics["total"]["peak"]["selection"]),
        rules=copy.deepcopy(RULES),
        verification_tolerance=copy.deepcopy(VERIFICATION_TOLERANCE),
        objective="sum((p_i-A*q_i)^2 / bin_width_i) / sum(p_i^2 / bin_width_i); p_i=L_i/sum(L), q_i=Gaussian bin probability; amplitude A profiled analytically",
        normalization="gaussian_area_native is the fitted full-line Gaussian area in native m^-2 per BX; fitted bin luminosity=area*(Phi(high)-Phi(low))",
        scope="Fixed native peak window to its exported upper edge; all high-side bins included; no upper-flow coordinate inference; integrated fit is not an average of frame fits",
        statistical_interpretation="Descriptive shape discrepancy; no Poisson-count assumption, p-value, fit-parameter statistical error, or independent-seed uncertainty",
        reference_sigma_W_GeV=reference,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        references=[
            dict(
                title="ROOT histogram fitting: bin integrals and weighted errors",
                url="https://root.cern/manual/fitting/",
            ),
            dict(
                title="ROOT TH1: weighted histogram likelihood requires sum of squared weights",
                url="https://root.cern.ch/doc/master/classTH1.html",
            ),
        ],
        total=make(diagnostics["total"], True),
        frames=[make(r) for r in diagnostics["frames"]],
    )


def _validate_legacy_gaussian(data, reference, diagnostics=None):
    require(
        isinstance(data, dict)
        and data.get("schema_version") == SCHEMA
        and data.get("estimator") == ESTIMATOR
        and data.get("rules") == RULES
        and data.get("verification_tolerance") == VERIFICATION_TOLERANCE
        and data.get("reference_sigma_W_GeV") == reference,
        "unknown peak-fit contract/rules",
    )
    require(
        isinstance(data.get("frames"), list) and len(data["frames"]) <= 64, "peak-fit frame bound"
    )
    for i, r in enumerate([data["total"], *data["frames"]]):
        require(
            type(r.get("available")) is bool and r.get("estimator") == ESTIMATOR,
            "invalid peak fit availability",
        )
        if i:
            require(r.get("id") == i - 1, "invalid peak fit ID")
        require(
            type(r.get("diagonal_start")) is int
            and type(r.get("diagonal_end")) is int
            and 0 <= r["diagonal_start"] <= r["diagonal_end"],
            "invalid peak fit interval",
        )
        c = r.get("candidate")
        allow = r.get("candidate_display_allowed")
        require(
            type(allow) is bool
            and r.get("candidate_status") in ("unavailable", "resolved_shape_mismatch", "admitted"),
            "unknown candidate state",
        )
        require(
            allow == (r["candidate_status"] != "unavailable")
            and r["available"] == (r["candidate_status"] == "admitted"),
            "inconsistent candidate admission",
        )
        if c is not None:
            require(
                all(
                    finite(c.get(k)) and c[k] > 0
                    for k in (
                        "mean_W_GeV",
                        "sigma_W_GeV",
                        "gaussian_area_native",
                        "gaussian_area_over_selected_luminosity",
                    )
                ),
                "invalid fitted candidate",
            )
            require(
                close(
                    c["gaussian_area_native"],
                    c["gaussian_area_over_selected_luminosity"] * r["selected_luminosity_weight"],
                ),
                "Gaussian area normalization mismatch",
            )
            expected_ratio = (
                reference / c["sigma_W_GeV"] if finite(reference) and reference > 0 else None
            )
            require(
                c.get("input_reference_ratio") == expected_ratio,
                "candidate input-reference ratio mismatch",
            )
        require((c is not None) == allow, "unresolved candidate must be absent")
        if allow:
            q = r.get("quality")
            solver = r.get("solver")
            require(
                c is not None
                and q
                and solver
                and solver["converged"]
                and q["interior_parameters"]
                and q["bracketed_one_sigma"]
                and q["sigma_over_bin_width"] >= RULES["minimum_sigma_over_bin_width"]
                and q["sigma_over_fit_span"] <= RULES["maximum_sigma_over_fit_span"]
                and min(q["positive_bins_left"], q["positive_bins_right"])
                >= RULES["minimum_positive_bins_each_side"],
                "unresolved candidate is displayed",
            )
            require(
                finite(r.get("relative_l2_residual")) and r["relative_l2_residual"] >= 0,
                "invalid descriptive residual",
            )
            require(
                r["available"]
                == (r["relative_l2_residual"] <= RULES["maximum_relative_l2_residual"]),
                "peak fit residual admission mismatch",
            )
        if r["available"]:
            require(
                r.get("reason") is None and all(r.get(k) == c[k] for k in c),
                "admitted fit/candidate mismatch",
            )
            ratio = reference / r["sigma_W_GeV"] if finite(reference) and reference > 0 else None
            require(
                r.get("input_reference_ratio") == ratio, "peak width input-reference ratio mismatch"
            )
        else:
            require(
                isinstance(r.get("reason"), str)
                and r["reason"]
                and all(
                    r.get(k) is None
                    for k in (
                        "mean_W_GeV",
                        "sigma_W_GeV",
                        "gaussian_area_native",
                        "gaussian_area_over_selected_luminosity",
                        "input_reference_ratio",
                    )
                ),
                "unavailable fit carries admitted width/ratio",
            )
    if diagnostics is not None:
        expected = build(diagnostics, reference, data["source_reference"])
        require(
            data["selection"] == expected["selection"]
            and equivalent_records(data["total"], expected["total"])
            and len(data["frames"]) == len(expected["frames"])
            and all(equivalent_records(a, b) for a, b in zip(data["frames"], expected["frames"])),
            "peak fits differ from deterministic analysis of authenticated native bins",
        )
    return data


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-280)


def equivalent_records(actual, expected):
    """Accept libm/optimizer rounding, never altered native weights or identities."""
    nominal = expected.get("nominal_W_GeV") or 1.0

    def compare(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(compare(a[k], b[k], path + (k,)) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(compare(x, y, path) for x, y in zip(a, b))
        key = path[-1] if path else ""
        if path and path[0] == "solver" and key in ("iterations", "evaluations"):
            return type(a) is int and 0 < a <= 2000
        if finite(a) and finite(b):
            if key == "mean_W_GeV":
                return math.isclose(a, b, rel_tol=0, abs_tol=1e-9 * nominal)
            if key in ("sigma_W_GeV", "input_reference_ratio"):
                return math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-14)
            if key in ("gaussian_area_native", "gaussian_area_over_selected_luminosity"):
                return math.isclose(a, b, rel_tol=2e-6, abs_tol=0)
            if key == "relative_l2_residual":
                return math.isclose(a, b, rel_tol=0, abs_tol=2e-7)
            if path and path[0] in ("quality", "solver") and type(a) is float:
                return math.isclose(a, b, rel_tol=3e-6, abs_tol=2e-7)
        return type(a) is type(b) and a == b

    return compare(actual, expected)




def validate(data, reference, diagnostics=None):
    """Read legacy Gaussian datasets and the explicit Appendix A.1 model."""
    if (
        isinstance(data, dict)
        and data.get("schema_version") == "feebl.collider.beta-gaussian-peak-fit/1"
    ):
        from collider_beta_peak_fit import validate as validate_beta

        return validate_beta(data, reference, diagnostics)
    return _validate_legacy_gaussian(data, reference, diagnostics)
