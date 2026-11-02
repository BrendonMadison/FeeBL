"""Validate bounded native frame observables, including integrated/frame closure."""

import math

SCHEMA = "feebl.collider.diagnostics/1"
DISTRIBUTIONS = ("chi_e", "chi_gamma", "photon_energy", "pair_energy")
SPECIES_DISTRIBUTIONS = ("chi_e_electron", "chi_e_positron")
SPECIES_BASIS = "canonical e+e- primary beam assignment; not per-particle species measurement"
CHI_WEIGHTING = "one per actual primary field kick; repeated across substeps; before radiation"
MOMENTS = ("energy_GeV", "x_nm", "y_nm", "vx", "vy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-9, abs_tol=1e-280)


def histogram(h):
    if h is None:
        return
    require(isinstance(h, dict), "histogram object required")
    for name in ("quantity", "unit", "weight_unit", "weighting"):
        require(
            isinstance(h.get(name), str) and 0 < len(h[name]) <= 512,
            "histogram definition required",
        )
    require(h.get("normalization") == "bin-integrated", "unknown histogram normalization")
    bins = h.get("bins")
    require(isinstance(bins, list) and 1 <= len(bins) <= 2048, "diagnostic histogram bin bound")
    previous = None
    for b in bins:
        require(
            isinstance(b, dict) and all(finite(b.get(k)) for k in ("low", "high", "weight")),
            "nonfinite diagnostic bin",
        )
        require(b["high"] > b["low"] and b["weight"] >= 0, "invalid diagnostic bin")
        require(previous is None or b["low"] >= previous, "overlapping diagnostic bins")
        previous = b["high"]
    require(
        all(finite(h.get(k)) and h[k] >= 0 for k in ("underflow", "overflow", "total_weight")),
        "invalid histogram flows",
    )
    require(
        close(
            math.fsum([b["weight"] for b in bins] + [h["underflow"], h["overflow"]]),
            h["total_weight"],
        ),
        "histogram weight closure failed",
    )


def moments(m):
    if m is None:
        return
    require(
        isinstance(m, dict) and isinstance(m.get("weighting"), str) and m["weighting"],
        "moments weighting required",
    )
    for beam in ("beam1", "beam2"):
        b = m.get(beam)
        require(
            isinstance(b, dict) and finite(b.get("weight_sum")) and b["weight_sum"] > 0,
            "invalid beam moment weight",
        )
        for key in MOMENTS:
            pair = b.get(key)
            require(
                isinstance(pair, dict)
                and finite(pair.get("mean"))
                and finite(pair.get("rms"))
                and pair["rms"] >= 0,
                "invalid beam moment",
            )


def polarization(p):
    if p is None:
        return
    require(
        isinstance(p, dict) and isinstance(p.get("weighting"), str) and p["weighting"],
        "polarization weighting required",
    )
    require(finite(p.get("weight_sum")) and p["weight_sum"] > 0, "invalid polarization weight")
    require(
        all(finite(p.get(k)) and abs(p[k]) <= 1 + 1e-7 for k in ("beam1", "beam2")),
        "invalid polarization",
    )


def species_metadata(d):
    records = d["frames"] + [d.get("total")]
    if "primary_species" not in d:
        require(
            all(
                isinstance(r, dict)
                and not any(k in (r.get("distributions") or {}) for k in SPECIES_DISTRIBUTIONS)
                for r in records
            ),
            "species histograms require mapping metadata",
        )
        return None
    m = d["primary_species"]
    require(
        isinstance(m, dict)
        and isinstance(m.get("supported"), bool)
        and finite(m.get("charge_sign"))
        and m.get("basis") == SPECIES_BASIS,
        "invalid primary species mapping",
    )
    require(
        m["supported"] == (m["charge_sign"] == -1), "unsupported primary charge/species mapping"
    )
    require(
        (
            (m.get("beam1_pdg") == 11 and m.get("beam2_pdg") == -11)
            if m["supported"]
            else (m.get("beam1_pdg") is None and m.get("beam2_pdg") is None)
        ),
        "invalid primary species PDG assignment",
    )
    return m


def species_record(r, m):
    distributions = r["distributions"]
    combined = distributions["chi_e"]
    electron = distributions["chi_e_electron"]
    positron = distributions["chi_e_positron"]
    available = m["supported"] and combined is not None
    require(
        (electron is not None) == available and (positron is not None) == available,
        "inconsistent primary species histogram availability",
    )
    if not available:
        return
    for h, quantity in [
        (combined, "native primary kick chi_e"),
        (electron, "native primary electron kick chi_e"),
        (positron, "native primary positron kick chi_e"),
    ]:
        require(
            h["quantity"] == quantity
            and h["unit"] == "1"
            and h["weight_unit"] == "macroparticle-kick evaluations"
            and h["weighting"] == CHI_WEIGHTING,
            "invalid primary chi definition",
        )
        require(
            all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and v == int(v)
                and 0 <= v <= 9007199254740991
                for v in [h["underflow"], h["overflow"], h["total_weight"]]
                + [b["weight"] for b in h["bins"]]
            ),
            "primary chi requires integer kick counts",
        )
        require(
            sum(b["weight"] for b in h["bins"]) + h["underflow"] + h["overflow"]
            == h["total_weight"],
            "primary chi exact histogram closure failed",
        )
        require(
            len(h["bins"]) == len(combined["bins"])
            and all(
                (b["low"], b["high"]) == (t["low"], t["high"])
                for b, t in zip(h["bins"], combined["bins"])
            ),
            "species chi histogram edges mismatch",
        )
    require(
        all(
            a["weight"] + b["weight"] == t["weight"]
            for a, b, t in zip(electron["bins"], positron["bins"], combined["bins"])
        ),
        "species/combined chi bin closure failed",
    )
    require(
        all(
            electron[k] + positron[k] == combined[k]
            for k in ("underflow", "overflow", "total_weight", "samples")
        ),
        "species/combined chi count closure failed",
    )


def validate_record(r, distribution_keys=DISTRIBUTIONS):
    require(isinstance(r, dict), "diagnostic record required")
    histogram(r.get("luminosity_spectrum"))
    polarization(r.get("polarization"))
    moments(r.get("beam_moments"))
    if "bunch_moments" in r:
        moments(r["bunch_moments"])
    if "bunch_polarization" in r:
        b = r["bunch_polarization"]
        require(
            isinstance(b, list)
            and len(b) == 2
            and all(v is None or (finite(v) and abs(v) <= 1 + 1e-7) for v in b),
            "invalid bunch polarization",
        )
    d = r.get("distributions")
    require(
        isinstance(d, dict) and all(k in d for k in distribution_keys),
        "distribution availability must be explicit",
    )
    for key in distribution_keys:
        histogram(d[key])
        h = d[key]
        if h is not None:
            require(
                isinstance(h.get("samples"), int)
                and not isinstance(h["samples"], bool)
                and 0 <= h["samples"] <= 9007199254740991,
                "invalid histogram sample count",
            )
            if key != "pair_energy":
                require(
                    h["samples"] == h["total_weight"], "unit-record histogram sample closure failed"
                )


def validate(d, manifest_frames):
    require(isinstance(d, dict) and d.get("schema_version") == SCHEMA, "unknown diagnostics schema")
    frames = d.get("frames")
    require(
        isinstance(frames, list) and len(frames) == len(manifest_frames) and 1 <= len(frames) <= 64,
        "diagnostics frame count mismatch",
    )
    species = species_metadata(d)
    distribution_keys = DISTRIBUTIONS + (SPECIES_DISTRIBUTIONS if species is not None else ())
    nominal = d.get("nominal_polarization")
    require(isinstance(d.get("unavailable_reasons"), dict), "availability reasons required")
    if nominal is not None:
        require(
            isinstance(nominal, dict) and isinstance(nominal.get("convention"), str),
            "nominal polarization convention required",
        )
        for key, length in (("beam1", 3), ("beam2", 3), ("reference_helicity", 2)):
            require(
                isinstance(nominal.get(key), list)
                and len(nominal[key]) == length
                and all(finite(v) and abs(v) <= 1 + 1e-7 for v in nominal[key]),
                "invalid nominal polarization",
            )
        require(
            math.isclose(nominal["reference_helicity"][0], nominal["beam1"][2], abs_tol=1e-9)
            and math.isclose(nominal["reference_helicity"][1], -nominal["beam2"][2], abs_tol=1e-9),
            "reference helicity convention mismatch",
        )
    else:
        require(
            d["unavailable_reasons"].get("nominal_polarization"),
            "missing nominal polarization requires reason",
        )
    previous = -1
    for i, (r, m) in enumerate(zip(frames, manifest_frames)):
        require(
            r.get("id") == i
            and r.get("diagonal_start") == previous + 1
            and r.get("diagonal_end") == m.get("diagonal"),
            "diagnostics interval mismatch",
        )
        require(
            isinstance(r["diagonal_end"], int) and r["diagonal_end"] >= r["diagonal_start"],
            "invalid diagonal interval",
        )
        previous = r["diagonal_end"]
        validate_record(r, distribution_keys)
        if species is not None:
            species_record(r, species)
    total = d.get("total")
    validate_record(total, distribution_keys)
    if species is not None:
        species_record(total, species)
    require(
        total.get("diagonal_start") == 0 and total.get("diagonal_end") == previous,
        "total diagnostic interval mismatch",
    )
    for key in ("luminosity_spectrum", *distribution_keys):
        get = (
            (lambda r: r.get(key))
            if key == "luminosity_spectrum"
            else (lambda r: r["distributions"][key])
        )
        th = get(total)
        fh = [get(f) for f in frames]
        require(all((h is None) == (th is None) for h in fh), "inconsistent histogram availability")
        if th is None:
            if key != "luminosity_spectrum":
                require(
                    isinstance(d["unavailable_reasons"].get(key), str)
                    and d["unavailable_reasons"][key],
                    "unavailable observable requires reason",
                )
            continue
        for h in fh:
            require(
                len(h["bins"]) == len(th["bins"])
                and all(
                    (b["low"], b["high"]) == (t["low"], t["high"])
                    for b, t in zip(h["bins"], th["bins"])
                ),
                "frame histogram edges mismatch",
            )
            require(
                all(h[k] == th[k] for k in ("quantity", "unit", "weight_unit", "weighting")),
                "frame histogram definition mismatch",
            )
        for i, b in enumerate(th["bins"]):
            require(
                close(math.fsum(h["bins"][i]["weight"] for h in fh), b["weight"]),
                "frame/total bin closure failed",
            )
        for k in ("underflow", "overflow", "total_weight"):
            require(close(math.fsum(h[k] for h in fh), th[k]), "frame/total flow closure failed")
        if key != "luminosity_spectrum":
            require(
                sum(h["samples"] for h in fh) == th["samples"],
                "frame/total sample count closure failed",
            )
        if (
            species is not None
            and species["supported"]
            and key in ("chi_e", *SPECIES_DISTRIBUTIONS)
        ):
            require(
                all(
                    sum(h["bins"][i]["weight"] for h in fh) == b["weight"]
                    for i, b in enumerate(th["bins"])
                )
                and all(
                    sum(h[k] for h in fh) == th[k]
                    for k in ("underflow", "overflow", "total_weight", "samples")
                ),
                "species chi exact frame/total count closure failed",
            )
    ps = [f["polarization"] for f in frames if f.get("polarization") is not None]
    tp = total.get("polarization")
    if tp is not None:
        require(
            ps and close(math.fsum(p["weight_sum"] for p in ps), tp["weight_sum"]),
            "polarization weight closure failed",
        )
        if total.get("luminosity_spectrum") is not None:
            require(
                close(tp["weight_sum"], total["luminosity_spectrum"]["total_weight"]),
                "total polarization/luminosity weight mismatch",
            )
        for k in ("beam1", "beam2"):
            require(
                math.isclose(
                    math.fsum(p[k] * p["weight_sum"] for p in ps) / tp["weight_sum"],
                    tp[k],
                    rel_tol=2e-9,
                    abs_tol=2e-10,
                ),
                "polarization weighted closure failed",
            )
    else:
        require(not ps, "missing total polarization")
    ms = [f["beam_moments"] for f in frames if f.get("beam_moments") is not None]
    tm = total.get("beam_moments")
    if tm is not None:
        require(ms, "missing frame luminosity-weighted moments")
        for beam in ("beam1", "beam2"):
            w = tm[beam]["weight_sum"]
            require(
                close(math.fsum(m[beam]["weight_sum"] for m in ms), w),
                "beam moment weight closure failed",
            )
            if total.get("luminosity_spectrum") is not None:
                require(
                    close(w, total["luminosity_spectrum"]["total_weight"]),
                    "total moment/luminosity weight mismatch",
                )
            for key in MOMENTS:
                mean = tm[beam][key]["mean"]
                combined = math.fsum(m[beam]["weight_sum"] * m[beam][key]["mean"] for m in ms) / w
                require(
                    math.isclose(combined, mean, rel_tol=2e-8, abs_tol=1e-12),
                    "beam moment mean closure failed",
                )
                variance = (
                    math.fsum(
                        m[beam]["weight_sum"]
                        * (m[beam][key]["rms"] ** 2 + (m[beam][key]["mean"] - mean) ** 2)
                        for m in ms
                    )
                    / w
                )
                rms = tm[beam][key]["rms"]
                scale = max(abs(mean), abs(combined), math.sqrt(variance), rms, 1e-140)
                # Means are serialized as doubles. The tolerance includes their
                # rounding propagated through the centered variance, not mean^2
                # times a loose relative tolerance that hides real narrow spreads.
                mean_roundoff = 128 * math.ulp(1.0) * scale
                variance_tolerance = (
                    2e-8 * max(variance, rms**2)
                    + 4 * mean_roundoff * max(math.sqrt(variance), rms)
                    + mean_roundoff**2
                )
                require(
                    abs(variance - rms**2) <= variance_tolerance,
                    "beam moment RMS closure failed",
                )
    else:
        require(not ms, "missing total luminosity-weighted moments")
    from collider_peak_diagnostics import validate_all

    validate_all(d)
    return dict(
        frames=len(frames), frame_total_closure="PASS", physics_validation="NOT_ESTABLISHED"
    )
