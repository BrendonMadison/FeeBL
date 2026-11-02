#!/usr/bin/env python3
"""Declared Gaussian optics reference and observed/input width ratios.

These are different quantities. Neither measures a causal monochromatization
gain without an authenticated matched collision control. Optional peak fits are
descriptive Gaussian shape fits, independently qualified from full-spectrum RMS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "feebl.collider.monochromatization/1"
MAX_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KEYS = (
    "energy_GeV",
    "espread",
    "ip_dispersion_m",
    "native_beam_angles_x_rad",
    "native_beam_angles_y_rad",
    "crab_mode",
    "waist_mode",
)
TOTAL_ESTIMATOR = "native unbinned centered full-spectrum RMS; printed native precision"
FRAME_ESTIMATOR = "luminosity-weighted bin-center population RMS; not a core fit"
FRAME_RULE = (
    "Positive luminosity, zero under/overflow, at least two positive-weight bins, "
    "and estimated RMS at least the widest positive-weight bin width."
)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def name(value):
    require(
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
        and value not in (".", ".."),
        "unsafe monochromatization source name",
    )
    return value


def read(root, filename):
    p = Path(root) / name(filename)
    require(
        p.is_file()
        and not p.is_symlink()
        and p.resolve().parent == Path(root).resolve()
        and 0 < p.stat().st_size <= MAX_BYTES,
        "missing/unsafe/oversized source: " + filename,
    )
    raw = p.read_bytes()
    return raw.decode("utf-8"), dict(
        path=filename, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )


def pair(value, label, positive=False):
    require(
        isinstance(value, list) and len(value) == 2 and all(finite(v) for v in value),
        "invalid " + label,
    )
    if positive:
        require(all(v > 0 for v in value), "nonpositive " + label)
    return value


def one(text, expression, label):
    found = re.findall(expression, text)
    require(len(found) == 1, "missing/duplicate evaluated native " + label)
    value = float(found[0])
    require(math.isfinite(value), "nonfinite evaluated native " + label)
    return value


def parse_inputs(text, manifest):
    headers = list(re.finditer(r"-+\s*beam parameter ([12])\s*-+", text))
    require(
        len(headers) == 2 and [m[1] for m in headers] == ["1", "2"],
        "native beam parameter blocks absent/ambiguous",
    )
    snapshot = {k: pair(manifest.get(k), "native manifest " + k)[:] for k in KEYS}
    beams = []
    for index, h in enumerate(headers):
        end = headers[index + 1].start() if index == 0 else text.find("SWITCHES", h.end())
        require(end > h.end(), "native switch block absent")
        block = text[h.end() : end]
        beams.append(
            dict(
                energy=one(block, r"\benergy\s*:\s*(" + NUM + r")\s+GeV", "energy"),
                sigma=one(block, r"\bsigma_x\s*:\s*(" + NUM + r")\s+nm", "sigma_x"),
                dist=one(block, r"\bdist_x\s*:\s*(" + NUM + r")(?![.\w+*/-])", "dist_x"),
                ax=one(block, r"\bangle_x\s*:\s*(" + NUM + r")(?![.\w+*/-])", "angle_x"),
                ay=one(block, r"\bangle_y\s*:\s*(" + NUM + r")(?![.\w+*/-])", "angle_y"),
            )
        )
    switch = text[text.find("SWITCHES") : text.find("grid parameters")]
    enabled = one(switch, r"\bdo_espread\s*=\s*(" + NUM + r")(?![.\w+*/-])", "do_espread")
    require(enabled in (0, 1), "unsupported energy-spread enable flag")
    requested, which, effective = [], [], []
    for i in (1, 2):
        s = one(
            switch,
            r"(?<![\w.])espread\." + str(i) + r"\s*=\s*(" + NUM + r")(?![.\w+*/-])",
            "espread",
        )
        w = one(
            switch,
            r"\bwhich_espread\." + str(i) + r"\s*=\s*(" + NUM + r")(?![.\w+*/-])",
            "which_espread",
        )
        require(s >= 0 and w in (0, 1, 2, 3), "unsupported native spread distribution")
        requested.append(s)
        which.append(int(w))
        effective.append(
            0.0 if not enabled or w == 0 else s * (1.7 / math.sqrt(3) if w == 1 else 1)
        )
    inputs = dict(
        energy_GeV=[b["energy"] for b in beams],
        sigma_beta_x_m=[b["sigma"] * 1e-9 for b in beams],
        sigma_delta=effective,
        requested_espread=requested,
        ip_dispersion_m=snapshot["ip_dispersion_m"],
        do_espread=int(enabled),
        which_espread=which,
        dist_x=[b["dist"] for b in beams],
        native_beam_angles_x_rad=[b["ax"] for b in beams],
        native_beam_angles_y_rad=[b["ay"] for b in beams],
        crab_mode=snapshot["crab_mode"],
        waist_mode=snapshot["waist_mode"],
    )
    for key in ("energy_GeV", "native_beam_angles_x_rad", "native_beam_angles_y_rad"):
        require(
            all(
                math.isclose(a, b, rel_tol=5e-6, abs_tol=5e-12)
                for a, b in zip(inputs[key], snapshot[key])
            ),
            "native listing/manifest mismatch: " + key,
        )
    require(
        all(
            math.isclose(a, b, rel_tol=5e-6, abs_tol=5e-12)
            for a, b in zip(requested, snapshot["espread"])
        ),
        "native spread listing/manifest mismatch",
    )
    return inputs, snapshot


def unavailable(reason, **extra):
    return dict(available=False, value=None, reason=reason, **extra)


def ideal_reference(inputs):
    for k in ("energy_GeV", "sigma_beta_x_m"):
        pair(inputs[k], k, positive=True)
    for k in (
        "sigma_delta",
        "ip_dispersion_m",
        "requested_espread",
        "which_espread",
        "dist_x",
        "native_beam_angles_x_rad",
        "native_beam_angles_y_rad",
        "crab_mode",
        "waist_mode",
    ):
        pair(inputs[k], k)
    s1, s2 = inputs["sigma_delta"]
    d1, d2 = inputs["ip_dispersion_m"]
    b1, b2 = inputs["sigma_beta_x_m"]
    require(s1 >= 0 and s2 >= 0, "negative effective spread")
    require(
        type(inputs["do_espread"]) is int and inputs["do_espread"] in (0, 1),
        "invalid energy-spread flag",
    )
    for supplied, requested, which in zip(
        inputs["sigma_delta"], inputs["requested_espread"], inputs["which_espread"]
    ):
        require(
            requested >= 0 and type(which) is int and which in (0, 1, 2, 3), "invalid spread model"
        )
        expected = (
            0.0
            if not inputs["do_espread"] or which == 0
            else requested * (1.7 / math.sqrt(3) if which == 1 else 1)
        )
        require(supplied == expected, "effective spread does not match native distribution")
    if d1 == d2 == 0 or s1 == s2 == 0:
        return dict(available=True, value=1.0, reason=None)
    if any(s > 0 and w != 3 for s, w in zip(inputs["sigma_delta"], inputs["which_espread"])) or any(
        inputs["dist_x"]
    ):
        return unavailable(
            "Nonzero-dispersion reference requires independent Gaussian energy and transverse distributions."
        )
    if any(
        inputs[k][i] != 0
        for k in ("native_beam_angles_x_rad", "native_beam_angles_y_rad", "crab_mode", "waist_mode")
        for i in (0, 1)
    ):
        return unavailable(
            "Nonzero-dispersion reference is admitted only for head-on geometry without crab or waist maps."
        )
    v = s1 * s1 + s2 * s2
    b = b1 * b1 + b2 * b2
    denominator = b + (d1 * s1) ** 2 + (d2 * s2) ** 2
    # Positive algebraic form of 1^T(S^-1+gg^T/B)^-1 1 avoids cancellation.
    variance = (v * b + (s1 * s2 * (d1 + d2)) ** 2) / denominator
    value = math.sqrt(v / variance) if variance > 0 else math.inf
    if not finite(value):
        return unavailable("Analytic factor is outside finite numerical range.")
    return dict(available=True, value=value, reason=None)


def width_ratio(reference, sigma, estimator, reason=None, **extra):
    payload = dict(sigma_W_GeV=sigma, estimator=estimator, **extra)
    if reason:
        return unavailable(reason, **payload)
    if not finite(reference) or reference <= 0:
        return unavailable("A positive configured input-width reference is required.", **payload)
    if not finite(sigma) or sigma <= 0:
        return unavailable("A positive resolved observed width is required.", **payload)
    return dict(available=True, value=reference / sigma, reason=None, **payload)


def frame_width(record, reference):
    extra = {k: record[k] for k in ("id", "diagonal_start", "diagonal_end")}
    h = record.get("luminosity_spectrum")
    if h is None:
        return width_ratio(
            reference,
            None,
            FRAME_ESTIMATOR,
            "Native frame luminosity spectrum unavailable.",
            **extra,
        )
    require(
        isinstance(h.get("bins"), list) and len(h["bins"]) <= 20000, "invalid frame spectrum bins"
    )
    flow = [h.get(k) for k in ("underflow", "overflow", "total_weight")]
    require(all(finite(v) and v >= 0 for v in flow), "invalid frame luminosity weights")
    occupied = []
    previous = -math.inf
    for b in h["bins"]:
        lo, hi, w = [b.get(k) for k in ("low", "high", "weight")]
        require(
            all(finite(v) for v in (lo, hi, w)) and 0 <= lo < hi and w >= 0 and lo >= previous,
            "invalid frame spectrum geometry/weight",
        )
        previous = hi
        if w > 0:
            occupied.append((0.5 * (lo + hi), hi - lo, w))
    sw = math.fsum(b[2] for b in occupied)
    require(
        math.isclose(sw + flow[0] + flow[1], flow[2], rel_tol=1e-8, abs_tol=0),
        "frame spectrum weight closure",
    )
    if flow[0] or flow[1]:
        return width_ratio(
            reference,
            None,
            FRAME_ESTIMATOR,
            "Frame flow weights have no exported energy coordinates.",
            **extra,
        )
    if sw <= 0 or len(occupied) < 2:
        return width_ratio(
            reference,
            None,
            FRAME_ESTIMATOR,
            "Frame needs positive luminosity in at least two distinct bins.",
            **extra,
        )
    # Shift before summing to avoid loss of variance near a narrow high-energy peak.
    origin = occupied[0][0]
    mu = math.fsum((x - origin) * w / sw for x, _, w in occupied)
    sigma = math.sqrt(math.fsum(w / sw * ((x - origin) - mu) ** 2 for x, _, w in occupied))
    if sigma < max(b[1] for b in occupied):
        return width_ratio(
            reference,
            sigma,
            FRAME_ESTIMATOR,
            "Frame RMS is smaller than the widest occupied bin; unresolved width.",
            **extra,
        )
    return width_ratio(reference, sigma, FRAME_ESTIMATOR, **extra)


def validate(data, diagnostics=None, analysis=None, manifest=None):
    require(
        isinstance(data, dict) and data.get("schema_version") == SCHEMA,
        "unknown monochromatization schema",
    )
    if diagnostics is not None and "peak" in diagnostics.get("total", {}):
        require(
            "peak_fit" in data,
            "native peak export requires its bound fit result or unavailable reason",
        )

    def walk(value):
        if isinstance(value, float):
            require(math.isfinite(value), "nonfinite monochromatization value")
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(data)
    ideal = ideal_reference(data["inputs"])
    for k in ("available", "value", "reason"):
        require(data["ideal"][k] == ideal[k], "ideal formula/admission mismatch")
    native = data["native_manifest_parameters"]
    for key in KEYS:
        pair(native.get(key), "native manifest " + key)
        supplied = data["inputs"]["requested_espread" if key == "espread" else key]
        if key in ("ip_dispersion_m", "crab_mode", "waist_mode"):
            require(supplied == native[key], "native manifest parameter mismatch: " + key)
        else:
            require(
                all(
                    math.isclose(a, b, rel_tol=5e-6, abs_tol=5e-12)
                    for a, b in zip(supplied, native[key])
                ),
                "evaluated/native input mismatch: " + key,
            )
    reference = math.sqrt(math.prod(data["inputs"]["energy_GeV"])) * math.hypot(
        *data["inputs"]["sigma_delta"]
    )
    require(
        data["measured"]["reference_sigma_W_GeV"] == reference, "input width reference mismatch"
    )
    records = [data["measured"]["total"], *data["measured"]["frames"], data["matched_control"]]
    require(len(data["measured"]["frames"]) <= 64, "frame cardinality")
    for i, r in enumerate(records):
        require(type(r.get("available")) is bool, "missing availability")
        if r["available"]:
            require(
                finite(r.get("value")) and r["value"] >= 0 and r.get("reason") is None,
                "invalid available factor",
            )
            require(
                finite(r.get("sigma_W_GeV"))
                and r["sigma_W_GeV"] > 0
                and math.isclose(r["value"], reference / r["sigma_W_GeV"], rel_tol=1e-12),
                "measured width ratio mismatch",
            )
        else:
            require(
                r.get("value") is None and isinstance(r.get("reason"), str) and bool(r["reason"]),
                "unavailable factor requires null and reason",
            )
    require(data["measured"]["total"]["estimator"] == TOTAL_ESTIMATOR, "unknown total estimator")
    for i, r in enumerate(data["measured"]["frames"]):
        require(
            r["id"] == i
            and type(r["diagonal_start"]) is int
            and type(r["diagonal_end"]) is int
            and 0 <= r["diagonal_start"] <= r["diagonal_end"],
            "invalid frame identity",
        )
        require(r["estimator"] == FRAME_ESTIMATOR, "unknown frame estimator")
        require(
            r["diagonal_start"]
            == (0 if i == 0 else data["measured"]["frames"][i - 1]["diagonal_end"] + 1),
            "noncontiguous diagnostic intervals",
        )
    require(
        data["matched_control"]["available"] is False, "matched-control support is not implemented"
    )
    for ref in data["source_artifacts"].values():
        name(ref["path"])
        require(
            type(ref["bytes"]) is int
            and 0 < ref["bytes"] <= MAX_BYTES
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "invalid source artifact",
        )
    require("native_summary" in data["source_artifacts"], "missing evaluated-optics provenance")
    if "peak_fit" in data:
        from collider_peak_fit import validate as validate_fit

        require(
            data["peak_fit"]["source_reference"] == data["source_artifacts"].get("diagnostics"),
            "peak fit native source role mismatch",
        )
        validate_fit(data["peak_fit"], reference, diagnostics)
        if diagnostics is not None and manifest is not None:
            from collider_peak_diagnostics import bind_reference

            bind_reference(diagnostics, manifest, analysis)
    require(
        len(json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        <= MAX_OUTPUT_BYTES,
        "monochromatization output byte bound",
    )
    return data


def analyze_run(root, manifest, summary_name="out.dat"):
    text, summary_ref = read(root, summary_name)
    inputs, snapshot = parse_inputs(text, manifest)
    result = dict(
        schema_version=SCHEMA,
        inputs=inputs,
        native_manifest_parameters=snapshot,
        ideal=dict(
            ideal_reference(inputs),
            label="Ideal Gaussian head-on monochromatization factor",
            unit="1",
            scope="static evaluated-optics reference; not a frame or luminosity-weighted measurement",
        ),
        definition=dict(
            model="independent Gaussian input energies and horizontal betatron coordinates; head-on spatial overlap; first-order invariant-mass width",
            formula="V=s1^2+s2^2; B=b1^2+b2^2; Vcoll=(V*B+s1^2*s2^2*(D1+D2)^2)/(B+D1^2*s1^2+D2^2*s2^2); ideal=sqrt(V/Vcoll)",
            meaning="Ideal 1 means no dispersion-selection narrowing. Measured width ratio >1 means narrower than the nominal input reference; <1 means broader.",
            frame_resolution_rule=FRAME_RULE,
            limitations=[
                "Evaluated out.dat parameters retain native printed rounding; sigma_x is interpreted as the configured pre-dispersion betatron size, not a measurement of loaded particles.",
                "The ideal factor omits crossing, hourglass, collective fields, beamstrahlung, angular spread and equilibrium changes; zero dispersion returns reference1 even for a non-head-on actual run.",
                "Input-width reference is an independent head-on first-order approximation. The observed spectrum includes native collision effects; this ratio does not isolate dispersion gain.",
                "Full RMS is tail-sensitive and is not a Gaussian core fit. Frame bin-center estimates omit within-bin variance and require the stated resolution rule.",
                "Total is formed from the integrated spectrum; it is not an average of frame ratios. No independent-seed uncertainty is inferred.",
            ],
        ),
        source_artifacts=dict(native_summary=summary_ref),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        references=[
            dict(
                title="Valdivia Garcia and Zimmermann, The challenge of monochromatization (2021)",
                url="https://link.springer.com/article/10.1140/epjp/s13360-021-02151-y",
            ),
            dict(
                title="Monochromatization optics for FCC-ee, IPAC2022 Eq. (3)",
                url="https://proceedings.jacow.org/ipac2022/papers/wepopt017.pdf",
            ),
        ],
    )
    reference = math.sqrt(math.prod(inputs["energy_GeV"])) * math.hypot(*inputs["sigma_delta"])
    measured = dict(
        label="Measured width ratio to input reference",
        definition="sigma_W,input / observed full-spectrum RMS",
        reference_sigma_W_GeV=reference,
        reference_scope="independent head-on input energies; first-order approximation; not a matched collision control",
        total=width_ratio(
            reference, None, TOTAL_ESTIMATOR, "Authenticated integrated native RMS unavailable."
        ),
        frames=[],
    )
    observed_sources = {}
    for key, default in [
        ("analysis", "collider_analysis.json"),
        ("diagnostics", "gui_diagnostics.json"),
    ]:
        filename = manifest.get(key, {}).get("path", default)
        if not (Path(root) / name(filename)).exists():
            continue
        raw, ref = read(root, filename)
        declared = manifest.get(key)
        if declared:
            require(
                all(declared.get(k) == ref[k] for k in ("path", "sha256", "bytes")),
                "source authentication mismatch: " + key,
            )
        source = json.loads(raw)
        result["source_artifacts"][key] = ref
        observed_sources[key] = source
        if key == "analysis":
            require(
                source.get("schema_version") == "feebl.collider.analysis/1",
                "unknown analysis source",
            )
            stat = source.get("native_statistics") or {}
            sigma = stat.get("full_RMS_W_GeV")
            require(sigma is None or finite(sigma) and sigma >= 0, "invalid native total RMS")
            measured["total"] = width_ratio(reference, sigma, TOTAL_ESTIMATOR)
        else:
            require(
                source.get("schema_version") == "feebl.collider.diagnostics/1",
                "unknown diagnostics source",
            )
            frames = source.get("frames")
            require(
                isinstance(frames, list) and len(frames) == len(manifest["frames"]) <= 64,
                "diagnostic frame cardinality",
            )
            for i, (r, m) in enumerate(zip(frames, manifest["frames"])):
                require(
                    r["id"] == m["id"] == i and r["diagonal_end"] == m["diagonal"],
                    "diagnostic frame identity mismatch",
                )
                measured["frames"].append(frame_width(r, reference))
    result["measured"] = measured
    result["matched_control"] = unavailable(
        "No authenticated matched reference run and frozen width estimator supplied."
    )
    diagnostic_source = observed_sources.get("diagnostics")
    if diagnostic_source and "peak" in diagnostic_source.get("total", {}):
        from collider_peak_diagnostics import bind_reference
        from collider_beta_peak_fit import build

        bind_reference(diagnostic_source, manifest, observed_sources.get("analysis"))
        result["peak_fit"] = build(
            diagnostic_source, reference, result["source_artifacts"]["diagnostics"]
        )
    return validate(result)


def write_analysis(root, data):
    validate(data)
    path = Path(root) / "collider_monochromatization.json"
    require(not path.is_symlink(), "refuse symlinked monochromatization output")
    serialized = json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    require(
        len(serialized.encode("utf-8")) <= MAX_OUTPUT_BYTES, "monochromatization output byte bound"
    )
    path.write_text(serialized, encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    man = json.loads((args.run / "gui_manifest.json").read_text())
    write_analysis(args.run, analyze_run(args.run, man))
