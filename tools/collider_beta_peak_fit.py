"""Wilson arXiv:2308.10414 Appendix A.1 beta/delta mixture convolved with Gaussian.

Fits bin-integrated native luminosity as a descriptive density shape. The
Gaussian response width is not the full-spectrum RMS or a causal narrowing gain.
NumPy/SciPy are required only for this new schema; legacy Gaussian datasets remain
readable through collider_peak_fit without importing this optional engine.
"""

import copy
import functools
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import betaln, ndtr, roots_legendre

SCHEMA = "feebl.collider.beta-gaussian-peak-fit/1"
ESTIMATOR = "bin-integrated beta-delta Gaussian convolution density least-squares shape fit; nonnegative component amplitudes profiled"
MODEL = (
    "Wilson Appendix A.1 beta distribution plus delta at x=1, convolved with a Gaussian response"
)
RULES = dict(
    minimum_occupied_bins=12,
    minimum_sigma_over_bin_width=1.5,
    maximum_relative_l2_residual=0.25,
    minimum_positive_bins_each_side=2,
    maximum_sigma_fraction_of_nominal=0.01,
    minimum_parameter_boundary_fraction=1e-4,
    alpha_bounds=[0.05, 500.0],
    beta_bounds=[0.02, 20.0],
    endpoint_fraction_bounds=[0.98, 1.02],
    minimum_component_fraction=1e-6,
    maximum_near_optimal_sigma_spread=0.02,
    near_optimal_objective_absolute=1e-6,
    maximum_quadrature_relative_l2_error=2e-6,
    maximum_quadrature_sigma_relative_change=0.002,
    maximum_jacobian_condition_number=1e10,
    sigma_profile_probe_fraction=0.10,
    minimum_sigma_profile_objective_increase=1e-4,
    quadrature_fit_order=48,
    quadrature_refit_order=96,
    quadrature_check_order=192,
    gaussian_truncation_sigma=8.0,
    maximum_function_evaluations_per_start=240,
)
TOLERANCE = dict(
    endpoint_absolute_fraction_of_nominal=2e-8,
    sigma_relative=2e-5,
    shape_parameter_relative=2e-4,
    area_relative=2e-5,
    residual_absolute=2e-6,
    curve_probability_absolute=2e-7,
    curve_probability_relative=2e-4,
    optimizer_counts="recorded provenance; not a cross-platform equality condition",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@functools.lru_cache(maxsize=8)
def legendre(order):
    return roots_legendre(order)


def gaussian_bins(low, high, mean, sigma):
    a = (np.asarray(low) - mean) / sigma
    b = (np.asarray(high) - mean) / sigma
    # Complementary CDF subtraction retains small high-side probabilities.
    return np.maximum(0.0, np.where(a >= 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a)))


def beta_bins(low, high, endpoint, sigma, alpha, beta, order=96):
    """Integrate Eq(A.1) over bins, removing both beta endpoint singularities.

    On x<=1/2 substitute t=x**alpha; on x>=1/2 use t=(1-x)**beta.
    The Gaussian bin probability is integrated, not the density at a bin center.
    Restricting latent x to each bin expanded by8sigma omits <=erfc(8/sqrt2)
    total Gaussian probability, independent of the normalized beta parameters.
    """
    require(
        endpoint > 0 and sigma > 0 and alpha > 0 and beta > 0, "invalid beta-Gaussian parameters"
    )
    a = np.asarray(low, dtype=float) / endpoint
    b = np.asarray(high, dtype=float) / endpoint
    s = sigma / endpoint
    nodes, weights = legendre(order)
    lower = np.maximum(0.0, a - RULES["gaussian_truncation_sigma"] * s)
    upper = np.minimum(1.0, b + RULES["gaussian_truncation_sigma"] * s)
    result = np.zeros(a.shape)
    for right in (False, True):
        lo = np.maximum(lower, 0.5) if right else lower
        hi = upper if right else np.minimum(upper, 0.5)
        use = hi > lo
        lo = lo[use]
        hi = hi[use]
        if not len(lo):
            continue
        power = beta if right else alpha
        tlo = (1 - hi) ** power if right else lo**power
        thi = (1 - lo) ** power if right else hi**power
        t = (tlo[:, None] + thi[:, None]) / 2 + (thi - tlo)[:, None] / 2 * nodes
        # A transformed interval can underflow only where its beta probability
        # is far below binary64 resolution. Endpoint t=0 maps exactly to x=0/1;
        # its transformed density is finite and its zero interval contributes0.
        with np.errstate(divide="ignore"):
            x = -np.expm1(np.log(t) / power) if right else np.exp(np.log(t) / power)
        logdensity = (
            ((alpha - 1) * np.log(x) if right else (beta - 1) * np.log1p(-x))
            - math.log(power)
            - betaln(alpha, beta)
        )
        q = gaussian_bins(a[use, None], b[use, None], x, s)
        result[use] += np.sum(q * np.exp(logdensity) * weights, axis=1) * (thi - tlo) / 2
    require(
        np.isfinite(result).all() and (result >= 0).all(), "nonfinite beta convolution quadrature"
    )
    return result


def component_profile(y, width, g, t=None):
    """Exact nonnegative least squares for one/two component amplitudes."""
    norm = math.sqrt(float(np.sum(y * y / width)))
    target = y / np.sqrt(width) / norm
    q = np.asarray([g] if t is None else [g, t]).T / np.sqrt(width[:, None]) / norm
    if t is None:
        co = np.array([max(0.0, float(q[:, 0] @ target / (q[:, 0] @ q[:, 0])))])
    else:
        co = np.linalg.lstsq(q, target, rcond=None)[0]
        if min(co) < 0:
            choices = [
                np.array([max(0.0, float(q[:, 0] @ target / (q[:, 0] @ q[:, 0]))), 0.0]),
                np.array([0.0, max(0.0, float(q[:, 1] @ target / (q[:, 1] @ q[:, 1])))]),
            ]
            co = min(choices, key=lambda c: float(np.sum((q @ c - target) ** 2)))
    return q @ co - target, co


def bounds(nominal, width):
    return np.array(
        [0.98 * nominal, math.log(width / 4), math.log(0.05), math.log(0.02)]
    ), np.array([1.02 * nominal, math.log(0.01 * nominal), math.log(500.0), math.log(20.0)])


def unpack(parameters):
    return float(parameters[0]), *map(lambda v: float(math.exp(v)), parameters[1:])


def optimize(low, high, y, nominal, reference=None):
    width = high - low
    bnds = bounds(nominal, float(min(width)))
    scale = np.array([0.002 * nominal, 1.0, 1.0, 1.0])
    calls = 0

    def evaluation(p, order=48, full=False):
        nonlocal calls
        calls += 1
        e, s, a, b = unpack(p)
        g = gaussian_bins(low, high, e, s)
        t = beta_bins(low, high, e, s, a, b, order)
        residual, co = component_profile(y, width, g, t)
        return (residual, co, g, t) if full else residual

    sigma = reference if finite(reference) and reference > 0 else nominal * 0.001
    sigma = min(math.exp(bnds[1][1]) * 0.8, max(math.exp(bnds[0][1]) * 2, sigma))
    starts = [
        (nominal, sigma, a, b) for a, b in [(30.0, 0.3), (5.0, 0.3), (100.0, 0.5), (1.0, 0.1)]
    ]
    solutions = []
    for e, s, a, b in starts:
        p = np.array([e, math.log(s), math.log(a), math.log(b)])
        r = least_squares(
            evaluation,
            p,
            bounds=bnds,
            x_scale=scale,
            max_nfev=RULES["maximum_function_evaluations_per_start"],
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        solutions.append(r)
    best = min(solutions, key=lambda r: (float(np.sum(r.fun * r.fun)), tuple(r.x)))
    initial_sigma = unpack(best.x)[1]
    refined = least_squares(
        lambda p: evaluation(p, 96),
        best.x,
        bounds=bnds,
        x_scale=scale,
        max_nfev=RULES["maximum_function_evaluations_per_start"],
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    p = refined.x
    residual, co, g, t = evaluation(p, 192, True)
    _, _, _, t96 = evaluation(p, 96, True)
    e, s, a, b = unpack(p)
    area = float(sum(co))
    fraction = float(co[0] / area)
    q = fraction * g + (1 - fraction) * t
    q96 = fraction * g + (1 - fraction) * t96
    quad = float(np.linalg.norm(q - q96) / max(np.linalg.norm(q), 1e-300))
    loss = float(residual @ residual)
    close = [
        unpack(r.x)[1]
        for r in solutions
        if float(r.fun @ r.fun)
        <= float(best.fun @ best.fun) + RULES["near_optimal_objective_absolute"]
    ]
    spread = (max(close) - min(close)) / s if close else math.inf
    sv = np.linalg.svd(refined.jac * scale, compute_uv=False)
    condition = float(sv[0] / sv[-1]) if sv[-1] > 0 else None
    probes = []
    # Profile nuisance parameters at fixed +/-10% sigma; these are descriptive
    # resolution checks, not confidence intervals or statistical uncertainties.
    for sign in (-1, 1):
        fixed = math.log(s * (1 + sign * RULES["sigma_profile_probe_fraction"]))

        def probe_eval(v):
            return evaluation(np.array([v[0], fixed, v[1], v[2]]), 96)

        selected = [0, 2, 3]
        r = least_squares(
            probe_eval,
            p[selected],
            bounds=(bnds[0][selected], bnds[1][selected]),
            x_scale=scale[selected],
            max_nfev=120,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        rr = evaluation(np.array([r.x[0], fixed, r.x[1], r.x[2]]), 192)
        probes.append(
            dict(
                sigma_W_GeV=math.exp(fixed),
                objective_increase=float(rr @ rr) - loss,
                converged=bool(r.success),
            )
        )
    position = ((p - bnds[0]) / (bnds[1] - bnds[0])).tolist()
    return dict(
        endpoint=e,
        sigma=s,
        alpha=a,
        beta=b,
        fraction=fraction,
        area=area,
        relative_l2_residual=math.sqrt(loss),
        tail=t.tolist(),
        converged=bool(refined.success),
        evaluations=calls,
        starts=len(starts),
        parameter_position=position,
        parameter_bounds=[bnds[0].tolist(), bnds[1].tolist()],
        sigma_bounds=[math.exp(bnds[0][1]), math.exp(bnds[1][1])],
        near_optimal_sigma_spread=spread,
        quadrature_relative_l2_error=quad,
        quadrature_sigma_relative_change=abs(s / initial_sigma - 1),
        jacobian_condition_number=condition,
        sigma_profile_probes=probes,
        model_regime="beta_delta_gaussian",
    )


def gaussian_limit(low, high, y, nominal, seed):
    width = high - low
    b = bounds(nominal, float(min(width)))

    def objective(p):
        return component_profile(y, width, gaussian_bins(low, high, p[0], math.exp(p[1])))[0]

    r = least_squares(
        objective,
        [seed["endpoint"], math.log(seed["sigma"])],
        bounds=(b[0][:2], b[1][:2]),
        x_scale=[0.002 * nominal, 1.0],
        max_nfev=120,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    e, s = float(r.x[0]), math.exp(r.x[1])
    residual, co = component_profile(y, width, gaussian_bins(low, high, e, s))
    loss = float(residual @ residual)
    if math.sqrt(loss) > seed["relative_l2_residual"] + 1e-6:
        return None
    probes = []
    for sign in (-1, 1):
        sig = s * (1 + sign * RULES["sigma_profile_probe_fraction"])

        def obj(p):
            return component_profile(y, width, gaussian_bins(low, high, p[0], sig))[0]

        p = least_squares(
            obj, [e], bounds=([b[0][0]], [b[1][0]]), max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10
        )
        rr = obj(p.x)
        probes.append(
            dict(
                sigma_W_GeV=sig, objective_increase=float(rr @ rr) - loss, converged=bool(p.success)
            )
        )
    return dict(
        seed,
        endpoint=e,
        sigma=s,
        alpha=None,
        beta=None,
        fraction=1.0,
        area=float(co[0]),
        tail=None,
        relative_l2_residual=math.sqrt(loss),
        converged=bool(r.success),
        model_regime="gaussian_delta_limit",
        parameter_position=((r.x - b[0][:2]) / (b[1][:2] - b[0][:2])).tolist(),
        parameter_bounds=[b[0][:2].tolist(), b[1][:2].tolist()],
        quadrature_relative_l2_error=0.0,
        quadrature_sigma_relative_change=0.0,
        jacobian_condition_number=float(np.linalg.cond(r.jac * np.array([0.002 * nominal, 1.0]))),
        sigma_profile_probes=probes,
        near_optimal_sigma_spread=0.0,
    )


def _fit_histogram_uncached(histogram, nominal, reference=None, identity=None):
    result = dict(
        available=False,
        reason=None,
        estimator=ESTIMATOR,
        source_histogram="luminosity_spectrum",
        endpoint_W_GeV=None,
        sigma_W_GeV=None,
        sigma_fractional=None,
        alpha=None,
        beta=None,
        peak_fraction=None,
        model_area_native=None,
        model_area_over_selected_luminosity=None,
        input_reference_ratio=None,
        nominal_W_GeV=nominal,
        requested_lower_W_GeV=None,
        fit_lower_W_GeV=None,
        fit_upper_W_GeV=None,
        selected_bin_start=0,
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
        model_regime=None,
    )
    if identity:
        result.update(identity)

    def fail(reason):
        result["reason"] = reason
        if not result["candidate_display_allowed"]:
            result["candidate"] = None
        return result

    if histogram is None or nominal is None:
        return fail("Authenticated full luminosity histogram or nominal reference unavailable.")
    require(finite(nominal) and nominal > 0, "invalid beta-fit nominal energy")
    bins = histogram.get("bins")
    require(
        isinstance(bins, list) and 1 <= len(bins) <= 20000, "invalid beta-fit histogram bin bound"
    )
    previous = None
    for b in bins:
        require(
            all(finite(b.get(k)) for k in ("low", "high", "weight"))
            and 0 <= b["low"] < b["high"]
            and b["weight"] >= 0,
            "invalid beta-fit histogram bin",
        )
        if previous is not None:
            require(
                math.isclose(b["low"], previous, rel_tol=1e-12, abs_tol=nominal * 1e-14),
                "gapped beta-fit histogram",
            )
        previous = b["high"]
    require(
        all(
            finite(histogram.get(k)) and histogram[k] >= 0
            for k in ("underflow", "overflow", "total_weight")
        ),
        "invalid beta-fit histogram flows",
    )
    weight = math.fsum(b["weight"] for b in bins)
    require(
        math.isclose(
            weight + histogram["underflow"] + histogram["overflow"],
            histogram["total_weight"],
            rel_tol=1e-8,
            abs_tol=0,
        ),
        "beta-fit luminosity closure",
    )
    occupied = sum(b["weight"] > 0 for b in bins)
    result.update(
        selected_bin_stop=len(bins),
        selected_bins=len(bins),
        occupied_bins=occupied,
        selected_luminosity_weight=weight,
        fit_lower_W_GeV=bins[0]["low"],
        fit_upper_W_GeV=bins[-1]["high"],
    )
    if histogram["underflow"] > 0 or histogram["overflow"] > 0:
        return fail(
            "Positive luminosity flow lacks exported coordinates for the full-spectrum fit."
        )
    if weight <= 0 or occupied < RULES["minimum_occupied_bins"]:
        return fail("Beta-Gaussian fit needs positive luminosity in at least twelve native bins.")
    low = np.array([b["low"] for b in bins])
    high = np.array([b["high"] for b in bins])
    y = np.array([b["weight"] / weight for b in bins])
    try:
        fitted = optimize(low, high, y, nominal, reference)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return fail(
            "Numerical optimizer could not resolve the beta-Gaussian shape within declared bounds."
        )
    if fitted["fraction"] >= 1 - RULES["minimum_component_fraction"]:
        limit = gaussian_limit(low, high, y, nominal, fitted)
        if limit is not None:
            fitted = limit
    e, s = fitted["endpoint"], fitted["sigma"]
    centers = (low + high) / 2
    near = (high >= e - 2 * s) & (low <= e + 2 * s)
    width = float(max((high - low)[near])) if np.any(near) else math.inf
    left = int(np.sum((y > 0) & (centers < e) & (centers >= e - 2 * s)))
    right = int(np.sum((y > 0) & (centers >= e) & (centers <= e + 2 * s)))
    q = dict(
        sigma_over_bin_width=s / width,
        sigma_fraction_of_nominal=s / nominal,
        positive_bins_left=left,
        positive_bins_right=right,
        bracketed_one_sigma=bool(e - s > low[0] and e + s < high[-1]),
        interior_parameters=all(
            RULES["minimum_parameter_boundary_fraction"]
            < v
            < 1 - RULES["minimum_parameter_boundary_fraction"]
            for v in fitted["parameter_position"]
        ),
        component_identifiable=fitted["model_regime"] == "gaussian_delta_limit"
        or RULES["minimum_component_fraction"]
        < fitted["fraction"]
        < 1 - RULES["minimum_component_fraction"],
        **{
            k: fitted[k]
            for k in (
                "near_optimal_sigma_spread",
                "quadrature_relative_l2_error",
                "quadrature_sigma_relative_change",
                "jacobian_condition_number",
                "sigma_profile_probes",
            )
        },
    )
    candidate = dict(
        model_regime=fitted["model_regime"],
        endpoint_W_GeV=e,
        sigma_W_GeV=s,
        sigma_fractional=s / e,
        alpha=fitted["alpha"],
        beta=fitted["beta"],
        peak_fraction=fitted["fraction"],
        model_area_native=fitted["area"] * weight,
        model_area_over_selected_luminosity=fitted["area"],
        input_reference_ratio=reference / s if finite(reference) and reference > 0 else None,
        curve=dict(
            source_bin_start=0, source_bin_stop=len(bins), tail_bin_probabilities=fitted["tail"]
        ),
    )
    result.update(
        relative_l2_residual=fitted["relative_l2_residual"],
        quality=q,
        candidate=candidate,
        model_regime=fitted["model_regime"],
        solver={
            k: fitted[k]
            for k in (
                "converged",
                "evaluations",
                "starts",
                "parameter_position",
                "parameter_bounds",
                "sigma_bounds",
            )
        },
    )
    if not fitted["converged"]:
        return fail("Beta-Gaussian optimizer did not converge.")
    if not q["interior_parameters"]:
        return fail("Beta-Gaussian parameters reached a declared search boundary.")
    if not q["component_identifiable"]:
        return fail("Gaussian core component is not separately resolved from the beta tail.")
    if q["sigma_over_bin_width"] < RULES["minimum_sigma_over_bin_width"]:
        return fail("Gaussian response width is not resolved by the native bins.")
    if not q["bracketed_one_sigma"] or min(left, right) < RULES["minimum_positive_bins_each_side"]:
        return fail("Gaussian response core is not sampled on both sides of its endpoint.")
    if q["near_optimal_sigma_spread"] > RULES["maximum_near_optimal_sigma_spread"]:
        return fail("Near-optimal fits disagree on the Gaussian response width.")
    if (
        q["quadrature_relative_l2_error"] > RULES["maximum_quadrature_relative_l2_error"]
        or q["quadrature_sigma_relative_change"] > RULES["maximum_quadrature_sigma_relative_change"]
    ):
        return fail("Gaussian response width or curve is not stable under quadrature refinement.")
    if (
        q["jacobian_condition_number"] is None
        or q["jacobian_condition_number"] > RULES["maximum_jacobian_condition_number"]
    ):
        return fail("Beta-Gaussian shape parameters are numerically degenerate.")
    if any(
        not p["converged"]
        or p["objective_increase"] < RULES["minimum_sigma_profile_objective_increase"]
        for p in q["sigma_profile_probes"]
    ):
        return fail("The Gaussian response width is weakly identified by the exported spectrum.")
    result.update(candidate_display_allowed=True, candidate_status="resolved_shape_mismatch")
    if result["relative_l2_residual"] > RULES["maximum_relative_l2_residual"]:
        return fail("Beta-Gaussian shape mismatch exceeds the declared descriptive residual rule.")
    result.update(
        available=True,
        reason=None,
        candidate_status="admitted",
        **{k: v for k, v in candidate.items() if k not in ("curve", "model_regime")},
    )
    return result


_FIT_CACHE = OrderedDict()


def fit_histogram(histogram, nominal, reference=None, identity=None):
    """Cache source-derived results within this process, never supplied fit data."""
    key = hashlib.sha256(
        json.dumps(
            [histogram, nominal, reference], sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = _fit_histogram_uncached(histogram, nominal, reference)
        if len(_FIT_CACHE) > 96:
            _FIT_CACHE.popitem(last=False)
    else:
        _FIT_CACHE.move_to_end(key)
    result = copy.deepcopy(_FIT_CACHE[key])
    if identity:
        result.update(identity)
    return result


def build(diagnostics, reference, source_reference):
    from collider_peak_diagnostics import validate_all

    validate_all(diagnostics)

    def make(record, total=False):
        identity = {k: record[k] for k in ("diagonal_start", "diagonal_end")}
        if not total:
            identity["id"] = record["id"]
        peak = record["peak"]
        selection = peak["selection"]
        nominal = selection["nominal_W_GeV"] if selection else None
        r = fit_histogram(
            record.get("luminosity_spectrum") if peak["available"] else None,
            nominal,
            reference,
            identity,
        )
        if r["candidate_display_allowed"]:
            c = r["candidate"]
            bins = peak["luminosity_spectrum"]["bins"]
            tail = (
                None
                if c["model_regime"] == "gaussian_delta_limit"
                else beta_bins(
                    np.array([b["low"] for b in bins]),
                    np.array([b["high"] for b in bins]),
                    c["endpoint_W_GeV"],
                    c["sigma_W_GeV"],
                    c["alpha"],
                    c["beta"],
                    192,
                ).tolist()
            )
            c["peak_curve"] = dict(
                source_bin_start=0, source_bin_stop=len(bins), tail_bin_probabilities=tail
            )
        if not peak["available"]:
            r["reason"] = "Native reference unavailable: " + peak["reason"]
        return r

    return dict(
        schema_version=SCHEMA,
        estimator=ESTIMATOR,
        model=MODEL,
        source_reference=copy.deepcopy(source_reference),
        selection=copy.deepcopy(diagnostics["total"]["peak"]["selection"]),
        rules=copy.deepcopy(RULES),
        verification_tolerance=copy.deepcopy(TOLERANCE),
        fit_source=dict(
            histogram="luminosity_spectrum",
            selection="all exported native bins; no rebinning; flow must be zero",
        ),
        formula="p(E)=f_peak*N(E;E0,sigma_W)+(1-f_peak)*integral_0^1 Beta(x;alpha,beta)*N(E;E0*x,sigma_W) dx; sigma_fractional=sigma_W/E0",
        objective="sum((p_i-A*q_i)^2/bin_width_i)/sum(p_i^2/bin_width_i); p_i=L_i/sum(L); Gaussian and beta-convolved component amplitudes profiled by nonnegative least squares",
        normalization="model_area_native is the fitted full-real-line mixture area in native m^-2 per BX; fitted bin luminosity=area*mixture_bin_probability",
        scope="Full informative luminosity spectrum determines beta tail; native W>=0.98 nominal diagnostic averages remain separate and unchanged; total fit is not an average of frame fits",
        statistical_interpretation="Descriptive weighted shape fit; no Poisson-count assumption, p-value, parameter statistical uncertainty, or causal matched-control gain",
        quadrature=dict(
            method="Gauss-Legendre after x^alpha / (1-x)^beta endpoint substitutions; exact Gaussian CDF bin integrals",
            fit_order=48,
            refit_order=96,
            check_order=192,
            gaussian_window_sigma=8.0,
            omitted_gaussian_probability_bound=math.erfc(8 / math.sqrt(2)),
        ),
        reference_sigma_W_GeV=reference,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(numpy=np.__version__, scipy=scipy.__version__),
        references=[
            dict(
                title="G.W.Wilson, Appendix A.1 equations(A.1)-(A.3), arXiv:2308.10414",
                url="https://arxiv.org/pdf/2308.10414",
            )
        ],
        total=make(diagnostics["total"], True),
        frames=[make(r) for r in diagnostics["frames"]],
    )


def resolved_quality(q, solver):
    return (
        isinstance(q, dict)
        and isinstance(solver, dict)
        and solver.get("converged") is True
        and q.get("interior_parameters") is True
        and q.get("component_identifiable") is True
        and q.get("bracketed_one_sigma") is True
        and finite(q.get("sigma_over_bin_width"))
        and q["sigma_over_bin_width"] >= RULES["minimum_sigma_over_bin_width"]
        and finite(q.get("sigma_fraction_of_nominal"))
        and 0 < q["sigma_fraction_of_nominal"] <= RULES["maximum_sigma_fraction_of_nominal"]
        and all(
            type(q.get(k)) is int and q[k] >= RULES["minimum_positive_bins_each_side"]
            for k in ("positive_bins_left", "positive_bins_right")
        )
        and all(
            finite(q.get(k)) and 0 <= q[k] <= RULES[rule]
            for k, rule in (
                ("near_optimal_sigma_spread", "maximum_near_optimal_sigma_spread"),
                ("quadrature_relative_l2_error", "maximum_quadrature_relative_l2_error"),
                ("quadrature_sigma_relative_change", "maximum_quadrature_sigma_relative_change"),
                ("jacobian_condition_number", "maximum_jacobian_condition_number"),
            )
        )
        and isinstance(q.get("sigma_profile_probes"), list)
        and len(q["sigma_profile_probes"]) == 2
        and all(
            p.get("converged") is True
            and finite(p.get("objective_increase"))
            and p["objective_increase"] >= RULES["minimum_sigma_profile_objective_increase"]
            for p in q["sigma_profile_probes"]
        )
    )


def validate(data, reference, diagnostics=None):
    require(
        isinstance(data, dict)
        and data.get("schema_version") == SCHEMA
        and data.get("estimator") == ESTIMATOR
        and data.get("model") == MODEL
        and data.get("rules") == RULES
        and data.get("verification_tolerance") == TOLERANCE
        and data.get("reference_sigma_W_GeV") == reference,
        "unknown beta-Gaussian fit contract/rules",
    )
    require(
        data.get("fit_source")
        == dict(
            histogram="luminosity_spectrum",
            selection="all exported native bins; no rebinning; flow must be zero",
        ),
        "unknown beta-Gaussian source histogram",
    )
    require(
        isinstance(data.get("frames"), list) and len(data["frames"]) <= 64, "beta-fit frame bound"
    )
    fields = (
        "endpoint_W_GeV",
        "sigma_W_GeV",
        "sigma_fractional",
        "alpha",
        "beta",
        "peak_fraction",
        "model_area_native",
        "model_area_over_selected_luminosity",
        "input_reference_ratio",
    )
    records = [data["total"], *data["frames"]]
    for i, r in enumerate(records):
        require(
            type(r.get("available")) is bool
            and r.get("estimator") == ESTIMATOR
            and r.get("source_histogram") == "luminosity_spectrum",
            "invalid beta fit availability/source",
        )
        if i:
            require(r.get("id") == i - 1, "invalid beta fit ID")
        require(
            type(r.get("diagonal_start")) is int
            and type(r.get("diagonal_end")) is int
            and 0 <= r["diagonal_start"] <= r["diagonal_end"],
            "invalid beta fit interval",
        )
        c = r.get("candidate")
        allow = r.get("candidate_display_allowed")
        require(
            type(allow) is bool
            and r.get("candidate_status") in ("unavailable", "resolved_shape_mismatch", "admitted")
            and allow == (r["candidate_status"] != "unavailable")
            and r["available"] == (r["candidate_status"] == "admitted")
            and (c is not None) == allow,
            "inconsistent beta fit candidate admission",
        )
        if allow:
            require(
                resolved_quality(r.get("quality"), r.get("solver")),
                "unresolved beta-Gaussian width is displayed",
            )
            require(
                finite(r.get("relative_l2_residual"))
                and r["relative_l2_residual"] >= 0
                and r["available"]
                == (r["relative_l2_residual"] <= RULES["maximum_relative_l2_residual"]),
                "beta-fit residual admission mismatch",
            )
            require(
                c.get("model_regime") == r.get("model_regime")
                and r["model_regime"] in ("beta_delta_gaussian", "gaussian_delta_limit"),
                "unknown beta-fit regime",
            )
            require(
                all(
                    finite(c.get(k)) and c[k] > 0
                    for k in (
                        "endpoint_W_GeV",
                        "sigma_W_GeV",
                        "sigma_fractional",
                        "model_area_native",
                        "model_area_over_selected_luminosity",
                    )
                ),
                "invalid beta fit core width/normalization",
            )
            require(
                math.isclose(
                    c["sigma_fractional"], c["sigma_W_GeV"] / c["endpoint_W_GeV"], rel_tol=1e-12
                )
                and math.isclose(
                    c["model_area_native"],
                    c["model_area_over_selected_luminosity"] * r["selected_luminosity_weight"],
                    rel_tol=1e-12,
                ),
                "beta fit units/area mismatch",
            )
            ratio = reference / c["sigma_W_GeV"] if finite(reference) and reference > 0 else None
            require(
                c.get("input_reference_ratio") == ratio,
                "beta Gaussian core input-reference ratio mismatch",
            )
            curve = c.get("curve")
            require(
                isinstance(curve, dict)
                and set(curve) == {"source_bin_start", "source_bin_stop", "tail_bin_probabilities"}
                and curve["source_bin_start"] == r["selected_bin_start"] == 0
                and curve["source_bin_stop"] == r["selected_bin_stop"] == r["selected_bins"],
                "beta curve source geometry mismatch",
            )
            if c["model_regime"] == "gaussian_delta_limit":
                require(
                    c.get("alpha") is None
                    and c.get("beta") is None
                    and c.get("peak_fraction") == 1.0
                    and curve["tail_bin_probabilities"] is None,
                    "invalid pure-delta Gaussian limit",
                )
            else:
                require(
                    all(
                        finite(c.get(k))
                        and RULES[k + "_bounds"][0] < c[k] < RULES[k + "_bounds"][1]
                        for k in ("alpha", "beta")
                    )
                    and finite(c.get("peak_fraction"))
                    and RULES["minimum_component_fraction"]
                    < c["peak_fraction"]
                    < 1 - RULES["minimum_component_fraction"],
                    "unresolved beta mixture parameters",
                )
                t = curve["tail_bin_probabilities"]
                require(
                    isinstance(t, list)
                    and len(t) == r["selected_bins"]
                    and all(finite(v) and 0 <= v <= 1 for v in t)
                    and math.fsum(t) <= 1 + 1e-5,
                    "invalid beta tail bin probabilities",
                )
            if diagnostics is not None:
                h = (diagnostics["total"] if i == 0 else diagnostics["frames"][i - 1])[
                    "luminosity_spectrum"
                ]
                bins = h["bins"]
                low = np.array([b["low"] for b in bins])
                high = np.array([b["high"] for b in bins])
                weights = np.array([b["weight"] for b in bins])
                weights /= weights.sum()
                require(
                    len(bins) == r["selected_bins"]
                    and r["selected_luminosity_weight"] == math.fsum(b["weight"] for b in bins)
                    and r["fit_lower_W_GeV"] == bins[0]["low"]
                    and r["fit_upper_W_GeV"] == bins[-1]["high"],
                    "beta fit source bin binding mismatch",
                )
                g = gaussian_bins(low, high, c["endpoint_W_GeV"], c["sigma_W_GeV"])
                t = (
                    None
                    if c["model_regime"] == "gaussian_delta_limit"
                    else np.array(c["curve"]["tail_bin_probabilities"])
                )
                residual, co = component_profile(weights, high - low, g, t)
                require(
                    math.isclose(
                        float(sum(co)), c["model_area_over_selected_luminosity"], rel_tol=2e-6
                    )
                    and math.isclose(
                        float(co[0] / sum(co)), c["peak_fraction"], rel_tol=2e-6, abs_tol=1e-8
                    )
                    and math.isclose(
                        float(np.linalg.norm(residual)),
                        r["relative_l2_residual"],
                        rel_tol=2e-6,
                        abs_tol=1e-8,
                    ),
                    "beta curve profile/residual mismatch",
                )
                peak = (diagnostics["total"] if i == 0 else diagnostics["frames"][i - 1])["peak"][
                    "luminosity_spectrum"
                ]
                pc = c.get("peak_curve")
                require(
                    isinstance(pc, dict)
                    and set(pc) == {"source_bin_start", "source_bin_stop", "tail_bin_probabilities"}
                    and pc["source_bin_start"] == 0
                    and pc["source_bin_stop"] == len(peak["bins"]),
                    "beta peak overlay source geometry mismatch",
                )
                if c["model_regime"] == "gaussian_delta_limit":
                    require(
                        pc["tail_bin_probabilities"] is None,
                        "Gaussian limit contains a beta peak curve",
                    )
                else:
                    actual = pc["tail_bin_probabilities"]
                    pb = peak["bins"]
                    require(
                        isinstance(actual, list)
                        and len(actual) == len(pb)
                        and all(finite(v) and 0 <= v <= 1 for v in actual),
                        "invalid beta peak overlay probability",
                    )
                    expected = beta_bins(
                        np.array([b["low"] for b in pb]),
                        np.array([b["high"] for b in pb]),
                        c["endpoint_W_GeV"],
                        c["sigma_W_GeV"],
                        c["alpha"],
                        c["beta"],
                        192,
                    )
                    require(
                        np.allclose(actual, expected, rtol=2e-4, atol=2e-7),
                        "beta peak curve differs from model parameters",
                    )
        if r["available"]:
            require(
                r.get("reason") is None and all(r.get(k) == c[k] for k in fields),
                "admitted beta fit/candidate mismatch",
            )
        else:
            require(
                isinstance(r.get("reason"), str)
                and r["reason"]
                and all(r.get(k) is None for k in fields),
                "unavailable beta fit carries admitted width",
            )
    if diagnostics is not None:
        expected = build(diagnostics, reference, data["source_reference"])
        require(
            data["selection"] == expected["selection"]
            and equivalent(data["total"], expected["total"])
            and len(data["frames"]) == len(expected["frames"])
            and all(equivalent(a, b) for a, b in zip(data["frames"], expected["frames"])),
            "beta-Gaussian fits differ from deterministic analysis of authenticated native bins",
        )
    return data


def equivalent(actual, expected):
    nominal = expected.get("nominal_W_GeV") or 1.0

    def same(a, b, path=()):
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(same(a[k], b[k], path + (k,)) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(same(x, y, path) for x, y in zip(a, b))
        key = path[-1] if path else ""
        if key == "evaluations" and path[0] == "solver":
            return type(a) is int and 0 < a <= 20000
        if finite(a) and finite(b):
            if key == "endpoint_W_GeV":
                return math.isclose(a, b, rel_tol=0, abs_tol=2e-8 * nominal)
            if key in ("sigma_W_GeV", "sigma_fractional", "input_reference_ratio"):
                return math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-12)
            if key in ("alpha", "beta", "peak_fraction"):
                return math.isclose(a, b, rel_tol=2e-4, abs_tol=1e-8)
            if key in ("model_area_native", "model_area_over_selected_luminosity"):
                return math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-280)
            if key == "relative_l2_residual":
                return math.isclose(a, b, rel_tol=0, abs_tol=2e-6)
            if key == "tail_bin_probabilities":
                return math.isclose(a, b, rel_tol=2e-4, abs_tol=2e-7)
            if path and path[0] in ("quality", "solver") and type(a) is float:
                return math.isclose(a, b, rel_tol=5e-4, abs_tol=2e-6)
        return type(a) is type(b) and a == b

    return same(actual, expected)
