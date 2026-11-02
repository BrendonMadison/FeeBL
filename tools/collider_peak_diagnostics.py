"""Validate exact native high-W luminosity selection and conditional moments."""

import math

DEFINITION = "sqrt(2*m_e^2 + 2*(E1*E2 + p1*p2*cos(full_crossing_angle)))"
NATIVE_W = "Existing massive invariant mass of native simulated incoming directions; no design crossing rotation; beam energies before optional ISR sampling"
ANGLE_SOURCES = (
    "evaluated head-on beam angles",
    "evaluated crab_full_crossing_angle matched to beam-angle receipt",
)
MASS = 0.00051099895


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def count(v):
    return type(v) is int and 0 <= v <= 9007199254740991


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-9, abs_tol=1e-280)


def selection(s):
    require(
        isinstance(s, dict)
        and s.get("threshold_fraction") == 0.98
        and s.get("upper_cut_GeV") is None
        and s.get("comparison") == "W >= minimum_W_GeV",
        "unknown peak selection",
    )
    r = s.get("nominal_reference")
    require(isinstance(r, dict), "missing peak nominal reference")
    e = r.get("energy_GeV")
    angle = r.get("full_crossing_angle_rad")
    require(
        isinstance(e, list)
        and len(e) == 2
        and all(finite(v) and v > MASS for v in e)
        and finite(angle)
        and 0 <= angle < math.pi,
        "invalid peak reference inputs",
    )
    require(
        r.get("electron_mass_GeV") == MASS
        and r.get("definition") == DEFINITION
        and r.get("native_W") == NATIVE_W
        and r.get("angle_source") in ANGLE_SOURCES,
        "unknown peak nominal convention",
    )
    require(
        (angle == 0) == (r["angle_source"] == ANGLE_SOURCES[0]),
        "peak nominal angle source mismatch",
    )
    p = [math.sqrt((v - MASS) * (v + MASS)) for v in e]
    nominal = math.sqrt(2 * MASS * MASS + 2 * (e[0] * e[1] + p[0] * p[1] * math.cos(angle)))
    require(
        finite(s.get("nominal_W_GeV"))
        and math.isclose(s["nominal_W_GeV"], nominal, rel_tol=1e-12)
        and finite(s.get("minimum_W_GeV"))
        and math.isclose(s["minimum_W_GeV"], 0.98 * nominal, rel_tol=1e-12),
        "peak nominal/cut formula mismatch",
    )
    return s


def record(p):
    from collider_diagnostics import histogram, polarization

    require(
        isinstance(p, dict) and type(p.get("available")) is bool, "invalid native peak availability"
    )
    if p.get("selection") is not None:
        selection(p["selection"])
    require(
        all(finite(p.get(k)) and p[k] >= 0 for k in ("weight_sum", "spin_weight_sum"))
        and all(count(p.get(k)) for k in ("contributions", "spin_contributions")),
        "invalid peak counters",
    )
    require(
        p["spin_contributions"] <= p["contributions"]
        and p["spin_weight_sum"] <= p["weight_sum"] * (1 + 1e-12),
        "peak spin coverage exceeds population",
    )
    require(
        (p["spin_contributions"] == 0) == (p["spin_weight_sum"] == 0),
        "peak positive-weight spin count mismatch",
    )
    if not p["available"]:
        require(
            isinstance(p.get("reason"), str)
            and p["reason"]
            and all(
                p[k] == 0
                for k in ("weight_sum", "spin_weight_sum", "contributions", "spin_contributions")
            )
            and all(
                p.get(k) is None
                for k in ("beam_energy_GeV", "W_GeV", "polarization", "luminosity_spectrum")
            ),
            "unavailable peak carries data",
        )
        return
    require(
        p.get("reason") is None and p.get("selection") is not None,
        "available peak missing reference",
    )
    h = p.get("luminosity_spectrum")
    histogram(h)
    require(
        isinstance(h, dict)
        and h.get("quantity") == "W"
        and h.get("unit") == "GeV"
        and h.get("weight_unit") == "m^-2 per BX"
        and h.get("weighting") == "native collision pair luminosity"
        and len(h["bins"]) == 512
        and h.get("samples") == p["contributions"]
        and h["underflow"] == 0
        and close(h["total_weight"], p["weight_sum"]),
        "peak histogram definition/count mismatch",
    )
    low = p["selection"]["minimum_W_GeV"]
    high = h["bins"][-1]["high"]
    require(
        high >= 1.02 * p["selection"]["nominal_W_GeV"] * (1 - 1e-12),
        "peak histogram omits native high side",
    )
    require(
        all(
            math.isclose(b["low"], low + (high - low) * i / 512, rel_tol=1e-12)
            and math.isclose(b["high"], low + (high - low) * (i + 1) / 512, rel_tol=1e-12)
            for i, b in enumerate(h["bins"])
        ),
        "native peak histogram geometry mismatch",
    )
    if p["weight_sum"] == 0:
        require(
            p["contributions"] == 0 and p.get("beam_energy_GeV") is None and p.get("W_GeV") is None,
            "empty peak has invented moments",
        )
    else:
        require(
            p["contributions"] > 0 and isinstance(p.get("beam_energy_GeV"), dict),
            "weighted peak requires moments",
        )
        values = [p["beam_energy_GeV"].get(k) for k in ("beam1", "beam2")] + [p.get("W_GeV")]
        require(
            all(
                isinstance(v, dict)
                and finite(v.get("mean"))
                and v["mean"] > 0
                and finite(v.get("rms"))
                and v["rms"] >= 0
                for v in values
            ),
            "invalid conditional peak moments",
        )
        require(p["W_GeV"]["mean"] >= low * (1 - 1e-12), "peak W mean below selection")
    polarized = p["weight_sum"] > 0 and p["spin_contributions"] == p["contributions"]
    if polarized:
        polarization(p.get("polarization"))
        require(
            p.get("polarization") is not None
            and p["polarization"]["weighting"] == "native collision pair luminosity"
            and close(p["spin_weight_sum"], p["weight_sum"])
            and close(p["polarization"]["weight_sum"], p["weight_sum"])
            and p.get("polarization_unavailable_reason") is None,
            "peak polarization coverage mismatch",
        )
    else:
        require(
            p.get("polarization") is None
            and isinstance(p.get("polarization_unavailable_reason"), str)
            and p["polarization_unavailable_reason"],
            "unavailable peak polarization requires reason",
        )


def validate_all(d):
    records = [d["total"], *d["frames"]]
    if not any("peak" in r for r in records):
        return
    require(all("peak" in r for r in records), "partial native peak export")
    for r in records:
        record(r["peak"])
        # Old frame spectra are differences of cumulative bins; tiny late
        # intervals may be rounded away there. New peak intervals accumulate
        # directly, so the subset check is only meaningful on integrated totals.
        if r is d["total"] and r["peak"]["available"] and r.get("luminosity_spectrum") is not None:
            require(
                r["peak"]["weight_sum"] <= r["luminosity_spectrum"]["total_weight"] * (1 + 2e-9),
                "peak selected luminosity exceeds full luminosity",
            )
    total = d["total"]["peak"]
    fs = [r["peak"] for r in d["frames"]]
    require(
        all(
            p["selection"] == total["selection"] and p["available"] == total["available"]
            for p in fs
        ),
        "inconsistent native peak selection/coverage",
    )
    if not total["available"]:
        return
    for k in ("weight_sum", "spin_weight_sum"):
        require(close(math.fsum(p[k] for p in fs), total[k]), "peak frame/total weight closure")
    for k in ("contributions", "spin_contributions"):
        require(sum(p[k] for p in fs) == total[k], "peak frame/total count closure")
    th = total["luminosity_spectrum"]
    for i, b in enumerate(th["bins"]):
        require(
            all(
                (
                    p["luminosity_spectrum"]["bins"][i]["low"],
                    p["luminosity_spectrum"]["bins"][i]["high"],
                )
                == (b["low"], b["high"])
                for p in fs
            )
            and close(
                math.fsum(p["luminosity_spectrum"]["bins"][i]["weight"] for p in fs), b["weight"]
            ),
            "peak bin frame/total closure",
        )
    require(
        close(math.fsum(p["luminosity_spectrum"]["overflow"] for p in fs), th["overflow"]),
        "peak overflow frame/total closure",
    )
    w = total["weight_sum"]
    if w:
        positive = [p for p in fs if p["weight_sum"] > 0]
        for key in ("beam1", "beam2", "W"):
            get = lambda p: p["W_GeV"] if key == "W" else p["beam_energy_GeV"][key]
            t = get(total)
            mean = math.fsum(p["weight_sum"] * get(p)["mean"] for p in positive) / w
            variance = (
                math.fsum(
                    p["weight_sum"] * (get(p)["rms"] ** 2 + (get(p)["mean"] - t["mean"]) ** 2)
                    for p in positive
                )
                / w
            )
            require(math.isclose(mean, t["mean"], rel_tol=2e-9), "peak mean frame/total closure")
            require(
                math.isclose(
                    variance, t["rms"] ** 2, rel_tol=2e-8, abs_tol=2e-10 * max(1.0, variance)
                ),
                "peak RMS frame/total closure",
            )
        if total["polarization"] is not None:
            require(
                all(p["polarization"] is not None for p in positive),
                "peak spin coverage frame/total mismatch",
            )
            for key in ("beam1", "beam2"):
                mean = math.fsum(p["weight_sum"] * p["polarization"][key] for p in positive) / w
                require(
                    math.isclose(mean, total["polarization"][key], rel_tol=2e-9, abs_tol=2e-10),
                    "peak polarization frame/total closure",
                )


def bind_reference(d, manifest, analysis):
    validate_all(d)
    if "peak" not in d["total"]:
        return
    s = d["total"]["peak"]["selection"]
    if s is None:
        return
    require(
        isinstance(analysis, dict) and isinstance(analysis.get("definition"), dict),
        "native peak requires authenticated design analysis",
    )
    r = s["nominal_reference"]
    a = analysis["definition"]
    require(
        r["energy_GeV"] == manifest.get("energy_GeV") == a.get("energy_GeV"),
        "peak nominal energies disagree with native manifest/design analysis",
    )
    require(
        a.get("angle_receipt_matches_design") is True and a.get("design_W") == DEFINITION,
        "peak design analysis has no matching native angle receipt",
    )
    ax = manifest.get("native_beam_angles_x_rad")
    ay = manifest.get("native_beam_angles_y_rad")
    require(
        all(isinstance(v, list) and len(v) == 2 and all(finite(x) for x in v) for v in (ax, ay)),
        "peak reference missing native beam angles",
    )
    angle = r["full_crossing_angle_rad"]
    if angle == 0:
        require(all(v == 0 for v in ax + ay), "head-on peak reference disagrees with native angles")
    else:
        active = ax if any(ax) and not any(ay) else ay if any(ay) and not any(ax) else None
        require(
            active is not None
            and active[0] * active[1] >= 0
            and abs(angle - sum(map(abs, active))) <= 1e-6 * (1 + sum(map(abs, active))),
            "peak crossing reference disagrees with native angle receipt",
        )
    require(
        math.isclose(
            r["full_crossing_angle_rad"],
            a.get("full_crossing_angle_rad", math.nan),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(s["nominal_W_GeV"], a.get("nominal_W_GeV", math.nan), rel_tol=1e-12),
        "peak nominal reference disagrees with externally declared design angle",
    )
