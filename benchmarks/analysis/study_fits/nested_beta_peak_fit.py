"""Study-only nested limits of the beta/delta Gaussian spectrum fit.

All bins, density objective, quadrature and numerical quality requirements are
retained. The zero-delta limit has its own reduced fit; a Gaussian limit must
also survive full-family profiles and expanded-tail-bound diagnostics.
"""
import copy, math
import numpy as np
from scipy.optimize import least_squares

SCHEMA = "feebl.monochromatization.nested-beta-fit/2"
POLICY = dict(
    objective="squared normalized density L2",
    maximum_objective_excess=1e-6,
    maximum_relative_width_difference_and_spread=0.02,
    expanded_alpha_upper=2000.0,
    expanded_beta_lower=0.005,
    expanded_full_start_sigma_factors=[0.5, 2.0],
    full_family_profile_sigma_factors=[0.9, 1.1],
    interpretation="Bounded descriptive response-width qualification; not global identification or statistical coverage",
)


def summary(raw):
    return {k: v for k, v in raw.items() if k not in ("tail",)}


def reduced_fit(engine, low, high, y, nominal, seed, regime):
    gaussian = regime == "gaussian_delta_limit"
    width = high - low
    bounds = engine.bounds(nominal, float(min(width)))
    if gaussian:
        bounds = (bounds[0][:2], bounds[1][:2])
    scale = np.array(
        [0.002 * nominal, 1.0] if gaussian else [0.002 * nominal, 1.0, 1.0, 1.0]
    )
    calls = 0

    def evaluation(p, order=48, full=False):
        nonlocal calls
        calls += 1
        if gaussian:
            e, s = float(p[0]), math.exp(p[1])
            curve = engine.gaussian_bins(low, high, e, s)
        else:
            e, s, a, b = engine.unpack(p)
            curve = engine.beta_bins(low, high, e, s, a, b, order)
        residual, co = engine.component_profile(y, width, curve)
        return (residual, co, curve) if full else residual

    if gaussian:
        starts = [
            np.array([seed["endpoint"], math.log(seed["sigma"] * factor)])
            for factor in [0.5, 1.0, 2.0]
        ]
    else:
        starts = [
            np.array(
                [
                    seed["endpoint"],
                    math.log(seed["sigma"] * factor),
                    math.log(a),
                    math.log(b),
                ]
            )
            for factor, a, b in [
                (1.0, seed["alpha"], seed["beta"]),
                (0.5, 5.0, 0.3),
                (2.0, 100.0, 0.5),
                (1.0, 1.0, 0.1),
            ]
        ]
    starts = [np.clip(p, bounds[0] + 1e-8, bounds[1] - 1e-8) for p in starts]
    solutions = [
        least_squares(
            evaluation,
            p,
            bounds=bounds,
            x_scale=scale,
            max_nfev=240,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        for p in starts
    ]
    best = min(solutions, key=lambda r: (float(r.fun @ r.fun), tuple(r.x)))
    initial_sigma = math.exp(best.x[1])
    refined = least_squares(
        lambda p: evaluation(p, 96),
        best.x,
        bounds=bounds,
        x_scale=scale,
        max_nfev=240,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    residual, co, curve = evaluation(refined.x, 192, True)
    _, _, curve96 = evaluation(refined.x, 96, True)
    loss = float(residual @ residual)
    e, s = float(refined.x[0]), math.exp(refined.x[1])
    close = [
        math.exp(r.x[1])
        for r in solutions
        if float(r.fun @ r.fun)
        <= float(best.fun @ best.fun) + POLICY["maximum_objective_excess"]
    ]
    singular = np.linalg.svd(refined.jac * scale, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    probes = []
    selected = [0] if gaussian else [0, 2, 3]
    for factor in POLICY["full_family_profile_sigma_factors"]:
        fixed = math.log(s * factor)

        def profile(v):
            p = np.empty(len(scale))
            p[1] = fixed
            p[selected] = v
            return evaluation(p, 96)

        r = least_squares(
            profile,
            refined.x[selected],
            bounds=(bounds[0][selected], bounds[1][selected]),
            x_scale=scale[selected],
            max_nfev=120,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        p = np.empty(len(scale))
        p[1] = fixed
        p[selected] = r.x
        rr = evaluation(p, 192)
        probes.append(
            dict(
                sigma_W_GeV=s * factor,
                objective_increase=float(rr @ rr) - loss,
                converged=bool(r.success),
            )
        )
    return dict(
        endpoint=e,
        sigma=s,
        alpha=None if gaussian else math.exp(refined.x[2]),
        beta=None if gaussian else math.exp(refined.x[3]),
        fraction=1.0 if gaussian else 0.0,
        area=float(co[0]),
        relative_l2_residual=math.sqrt(loss),
        tail=None if gaussian else curve.tolist(),
        converged=bool(refined.success),
        evaluations=calls,
        starts=len(starts),
        parameter_position=((refined.x - bounds[0]) / (bounds[1] - bounds[0])).tolist(),
        parameter_bounds=[bounds[0].tolist(), bounds[1].tolist()],
        sigma_bounds=[math.exp(bounds[0][1]), math.exp(bounds[1][1])],
        near_optimal_sigma_spread=(max(close) - min(close)) / s,
        quadrature_relative_l2_error=float(
            np.linalg.norm(curve - curve96) / max(np.linalg.norm(curve), 1e-300)
        ),
        quadrature_sigma_relative_change=abs(s / initial_sigma - 1),
        jacobian_condition_number=condition,
        sigma_profile_probes=probes,
        model_regime=regime,
        start_solutions=[
            dict(
                initial_sigma_GeV=math.exp(p[1]),
                sigma_GeV=math.exp(r.x[1]),
                objective=float(r.fun @ r.fun),
                converged=bool(r.success),
            )
            for p, r in zip(starts, solutions)
        ],
    )


def quality(engine, raw, low, high, y, nominal):
    e, s = raw["endpoint"], raw["sigma"]
    centres = (low + high) / 2
    near = (high >= e - 2 * s) & (low <= e + 2 * s)
    width = float(max((high - low)[near])) if np.any(near) else math.inf
    rules = engine.RULES
    return dict(
        sigma_over_bin_width=s / width,
        sigma_fraction_of_nominal=s / nominal,
        positive_bins_left=int(
            np.sum((y > 0) & (centres < e) & (centres >= e - 2 * s))
        ),
        positive_bins_right=int(
            np.sum((y > 0) & (centres >= e) & (centres <= e + 2 * s))
        ),
        bracketed_one_sigma=bool(e - s > low[0] and e + s < high[-1]),
        interior_parameters=all(
            rules["minimum_parameter_boundary_fraction"]
            < v
            < 1 - rules["minimum_parameter_boundary_fraction"]
            for v in raw["parameter_position"]
        ),
        component_identifiable=raw["area"] > 0,
        **{
            key: raw[key]
            for key in [
                "near_optimal_sigma_spread",
                "quadrature_relative_l2_error",
                "quadrature_sigma_relative_change",
                "jacobian_condition_number",
                "sigma_profile_probes",
            ]
        }
    )


def reasons(engine, raw, q):
    r = engine.RULES
    fail = []
    if not raw["converged"]:
        fail.append("reduced optimizer did not converge")
    if not q["interior_parameters"]:
        fail.append("reduced parameters reached a declared boundary")
    if not q["component_identifiable"]:
        fail.append("reduced component normalization is unresolved")
    if q["sigma_over_bin_width"] < r["minimum_sigma_over_bin_width"]:
        fail.append("response width is not bin-resolved")
    if (
        not q["bracketed_one_sigma"]
        or min(q["positive_bins_left"], q["positive_bins_right"])
        < r["minimum_positive_bins_each_side"]
    ):
        fail.append("response is not sampled on both sides")
    if q["near_optimal_sigma_spread"] > r["maximum_near_optimal_sigma_spread"]:
        fail.append("reduced starts disagree on response width")
    if (
        q["quadrature_relative_l2_error"] > r["maximum_quadrature_relative_l2_error"]
        or q["quadrature_sigma_relative_change"]
        > r["maximum_quadrature_sigma_relative_change"]
    ):
        fail.append("reduced quadrature is unstable")
    if (
        q["jacobian_condition_number"] is None
        or q["jacobian_condition_number"] > r["maximum_jacobian_condition_number"]
    ):
        fail.append("reduced Jacobian is ill-conditioned")
    if any(
        not p["converged"]
        or p["objective_increase"] < r["minimum_sigma_profile_objective_increase"]
        for p in q["sigma_profile_probes"]
    ):
        fail.append("reduced sigma profile is weak or unconverged")
    if raw["relative_l2_residual"] > r["maximum_relative_l2_residual"]:
        fail.append("reduced shape mismatch exceeds the retained limit")
    return fail


def comparison_failures(engine, reduced, alternatives):
    failures = []
    best = min(raw["relative_l2_residual"] ** 2 for raw in alternatives)
    if reduced["relative_l2_residual"] ** 2 > best + POLICY["maximum_objective_excess"]:
        failures.append("nested model exceeds the squared-objective tolerance")
    for raw in alternatives:
        if not raw["converged"]:
            failures.append("full-model alternative did not converge")
        spread = (
            raw["near_optimal_sigma_spread"] + raw["quadrature_sigma_relative_change"]
        )
        if (
            abs(reduced["sigma"] / raw["sigma"] - 1) + spread
            > POLICY["maximum_relative_width_difference_and_spread"]
        ):
            failures.append(
                "response width is unstable against full-model alternatives"
            )
        if (
            raw["quadrature_relative_l2_error"]
            > engine.RULES["maximum_quadrature_relative_l2_error"]
        ):
            failures.append("full-model alternative quadrature is unstable")
        if (
            raw["quadrature_sigma_relative_change"]
            > engine.RULES["maximum_quadrature_sigma_relative_change"]
        ):
            failures.append(
                "full-model alternative quadrature width shift exceeds the retained limit"
            )
        if any(
            not p["converged"]
            or p["objective_increase"]
            < engine.RULES["minimum_sigma_profile_objective_increase"]
            for p in raw["sigma_profile_probes"]
        ):
            failures.append("full-model sigma profile is weak or unconverged")
    return failures


def full_family_profiles(engine, low, high, y, nominal, sigma, alternatives):
    """Profile all full-family nuisance coordinates at the Gaussian sigma probes."""
    width = high - low
    scale = np.array([0.002 * nominal, 1.0, 1.0])
    selected = [0, 2, 3]
    best = min(raw["relative_l2_residual"] ** 2 for raw in alternatives)
    profiles = []
    for factor in POLICY["full_family_profile_sigma_factors"]:
        fixed = sigma * factor
        solutions = []
        for raw in alternatives:
            bounds = tuple(np.array(x)[selected] for x in raw["parameter_bounds"])
            start = np.array(
                [raw["endpoint"], math.log(raw["alpha"]), math.log(raw["beta"])]
            )

            def evaluation(p, order=96):
                g = engine.gaussian_bins(low, high, p[0], fixed)
                t = engine.beta_bins(
                    low, high, p[0], fixed, math.exp(p[1]), math.exp(p[2]), order
                )
                return engine.component_profile(y, width, g, t)[0]

            r = least_squares(
                evaluation,
                start,
                bounds=bounds,
                x_scale=scale,
                max_nfev=160,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            residual = evaluation(r.x, 192)
            solutions.append(
                dict(objective=float(residual @ residual), converged=bool(r.success))
            )
        profiles.append(
            dict(
                sigma_W_GeV=fixed,
                objective_increase=min(s["objective"] for s in solutions) - best,
                converged=all(s["converged"] for s in solutions),
                alternatives=solutions,
            )
        )
    return profiles


def admit(legacy, raw, q, weight, reference):
    result = copy.deepcopy(legacy)
    e, s = raw["endpoint"], raw["sigma"]
    candidate = dict(
        model_regime=raw["model_regime"],
        endpoint_W_GeV=e,
        sigma_W_GeV=s,
        sigma_fractional=s / e,
        alpha=raw["alpha"],
        beta=raw["beta"],
        peak_fraction=raw["fraction"],
        model_area_native=raw["area"] * weight,
        model_area_over_selected_luminosity=raw["area"],
        input_reference_ratio=reference / s if reference and reference > 0 else None,
    )
    result.update(
        available=True,
        qualified=True,
        reason=None,
        candidate=candidate,
        candidate_status="admitted",
        candidate_display_allowed=True,
        model_regime=raw["model_regime"],
        relative_l2_residual=raw["relative_l2_residual"],
        quality=q,
        solver={
            key: raw[key]
            for key in [
                "converged",
                "evaluations",
                "starts",
                "parameter_position",
                "parameter_bounds",
                "sigma_bounds",
            ]
        },
    )
    result.update(
        {key: value for key, value in candidate.items() if key != "model_regime"}
    )
    return result


def fit_histogram(engine, histogram, nominal, reference=None):
    captured = {}
    original = engine.optimize

    def capture(*args, **kwargs):
        raw = original(*args, **kwargs)
        captured["full"] = raw
        return raw

    engine.optimize = capture
    try:
        legacy = engine._uncached_fit(histogram, nominal, reference)
    finally:
        engine.optimize = original
    result = copy.deepcopy(legacy)
    audit = dict(
        schema=SCHEMA,
        policy=copy.deepcopy(POLICY),
        legacy_qualification={
            key: legacy.get(key)
            for key in [
                "available",
                "reason",
                "model_regime",
                "sigma_W_GeV",
                "relative_l2_residual",
            ]
        },
        regimes=[],
    )
    result["study_nested_qualification"] = audit
    if "full" not in captured:
        return result
    full = captured["full"]
    audit["standard_full_model"] = summary(full)
    if legacy["available"] and legacy["model_regime"] == "beta_delta_gaussian":
        return result
    result.update(
        available=False, qualified=False, candidate_status="study_limit_unqualified"
    )
    if legacy["available"]:
        audit["legacy_candidate"] = {
            k: v for k, v in (legacy.get("candidate") or {}).items() if k != "curve"
        }
        result[
            "reason"
        ] = "The legacy Gaussian limit requires the additional study full-family qualification."
        result.update(candidate=None, candidate_display_allowed=False)
        for key in [
            "endpoint_W_GeV",
            "sigma_W_GeV",
            "sigma_fractional",
            "alpha",
            "beta",
            "peak_fraction",
            "model_area_native",
            "model_area_over_selected_luminosity",
            "input_reference_ratio",
        ]:
            result[key] = None
    engine.study_nested_progress = result
    bins = histogram["bins"]
    low = np.array([b["low"] for b in bins])
    high = np.array([b["high"] for b in bins])
    weight = math.fsum(b["weight"] for b in bins)
    y = np.array([b["weight"] / weight for b in bins])
    if full["fraction"] <= engine.RULES["minimum_component_fraction"]:
        pure = reduced_fit(engine, low, high, y, nominal, full, "beta_gaussian_limit")
        q = quality(engine, pure, low, high, y, nominal)
        fail = reasons(engine, pure, q) + comparison_failures(engine, pure, [full])
        audit["regimes"].append(
            dict(model=summary(pure), quality=q, rejection_reasons=fail)
        )
        if not fail:
            result = admit(legacy, pure, q, weight, reference)
            result["study_nested_qualification"] = audit
            return result
    gaussian = reduced_fit(engine, low, high, y, nominal, full, "gaussian_delta_limit")
    q = quality(engine, gaussian, low, high, y, nominal)
    fail = reasons(engine, gaussian, q) + comparison_failures(engine, gaussian, [full])
    gaussian_audit = dict(model=summary(gaussian), quality=q, rejection_reasons=fail)
    audit["regimes"].append(gaussian_audit)
    if fail:
        return result
    # Wider tail bounds are diagnostic only: no wider-bound beta solution is
    # admitted. They test whether the purported Gaussian width is stable.
    original_bounds = engine.bounds

    def expanded_bounds(*args):
        lo, hi = original_bounds(*args)
        lo = lo.copy()
        hi = hi.copy()
        lo[3] = math.log(POLICY["expanded_beta_lower"])
        hi[2] = math.log(POLICY["expanded_alpha_upper"])
        return lo, hi

    alternatives = [full]
    engine.bounds = expanded_bounds
    try:
        for factor in POLICY["expanded_full_start_sigma_factors"]:
            alternatives.append(
                original(low, high, y, nominal, gaussian["sigma"] * factor)
            )
    finally:
        engine.bounds = original_bounds
    audit["expanded_full_models"] = [summary(raw) for raw in alternatives[1:]]
    fail += comparison_failures(engine, gaussian, alternatives)
    profiles = full_family_profiles(
        engine, low, high, y, nominal, gaussian["sigma"], alternatives
    )
    audit["gaussian_full_family_sigma_profiles"] = profiles
    if any(
        not p["converged"]
        or p["objective_increase"]
        < engine.RULES["minimum_sigma_profile_objective_increase"]
        for p in profiles
    ):
        fail.append("full-family sigma profile is weak or unconverged")
    gaussian_audit["rejection_reasons"] = fail
    if not fail:
        result = admit(legacy, gaussian, q, weight, reference)
        result["study_nested_qualification"] = audit
    return result
