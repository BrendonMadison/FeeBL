"""Validate and expand bounded native current-state secondary histograms.

Snapshots are not cumulative histograms: each observes the authoritative reserve
after tracker deactivation, including previously escaped records exactly once.
"""

import copy
import json
import math

SCHEMA = "feebl.collider.secondary-frames/1"
PHASE = "post-diagonal-tracker-state"
WEIGHTING = (
    "one per current stored tracked secondary record; no luminosity or pair-ratio reweighting"
)
MAX_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 64
MAX_RECORDS = 1_000_000
KEYS = ("theta", "theta_folded", "pT")
QUANTITIES = dict(
    theta="native polar angle from +z",
    theta_folded="native polar angle from nearer +z or -z axis",
    pT="transverse momentum",
)
AXIS_KEYS = dict(theta="theta_edges_rad", theta_folded="theta_folded_edges_rad", pT="pT_edges_GeV")
MARGINAL_KEYS = dict(
    theta="marginal_theta_edges_rad",
    theta_folded="marginal_theta_folded_edges_rad",
    pT="marginal_pT_edges_GeV",
)
EXCLUSIONS = {
    "theta_nonpositive",
    "pT_nonpositive",
    "both_nonpositive",
    "nonpositive_records",
    "underflow_records",
    "overflow_records",
}
MODE = "centre of highest count-density histogram bin; approximate"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def count(v):
    return type(v) is int and 0 <= v <= MAX_RECORDS


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def axes_valid(axes):
    require(
        isinstance(axes, dict)
        and set(axes) == set(AXIS_KEYS.values()) | set(MARGINAL_KEYS.values()),
        "unknown native snapshot axes",
    )
    for key, edges in axes.items():
        n = 161 if key.startswith("marginal_") else 129
        require(
            isinstance(edges, list)
            and len(edges) == n
            and all(finite(v) for v in edges)
            and all(a < b for a, b in zip(edges, edges[1:])),
            "invalid native snapshot axis",
        )
        require(
            edges[0] >= 0 if key.startswith("marginal_") else edges[0] > 0,
            "invalid native snapshot log/physical axis",
        )
    for key, end in [("theta", math.pi), ("theta_folded", math.pi / 2), ("pT", 1e6)]:
        e = axes[AXIS_KEYS[key]]
        require(
            math.isclose(e[0], 1e-12, rel_tol=1e-12) and math.isclose(e[-1], end, rel_tol=1e-12),
            "native snapshot axis endpoints mismatch",
        )
        expected = [10 ** (-12 + (math.log10(end) + 12) * i / 128) for i in range(129)]
        require(
            all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(e, expected)),
            "native snapshot axis is not fixed logarithmic geometry",
        )
        m = axes[MARGINAL_KEYS[key]]
        expected = (
            [0] + [10 ** (-12 + 18 * i / 159) for i in range(160)]
            if key == "pT"
            else [end * i / 160 for i in range(161)]
        )
        require(
            all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-30) for a, b in zip(m, expected)),
            "native snapshot marginal geometry mismatch",
        )


def joint_valid(j, n):
    require(
        isinstance(j, dict)
        and isinstance(j.get("counts"), list)
        and len(j["counts"]) == 128 * 128
        and all(count(v) for v in j["counts"]),
        "invalid native snapshot joint counts",
    )
    ex = j.get("excluded")
    require(
        isinstance(ex, dict) and set(ex) == EXCLUSIONS and all(count(v) for v in ex.values()),
        "invalid native snapshot exclusions",
    )
    require(
        ex["both_nonpositive"] <= min(ex["theta_nonpositive"], ex["pT_nonpositive"])
        and ex["nonpositive_records"]
        == ex["theta_nonpositive"] + ex["pT_nonpositive"] - ex["both_nonpositive"],
        "native snapshot exclusion union mismatch",
    )
    total = sum(j["counts"])
    require(
        count(j.get("samples"))
        and count(j.get("in_range_records"))
        and j["samples"] == n
        and j["in_range_records"] == total
        and total + ex["nonpositive_records"] + ex["underflow_records"] + ex["overflow_records"]
        == n,
        "native snapshot joint count closure",
    )


def marginal_valid(h, n, edges, key):
    require(
        isinstance(h, dict)
        and isinstance(h.get("counts"), list)
        and len(h["counts"]) == 160
        and all(count(v) for v in h["counts"]),
        "invalid native snapshot marginal counts",
    )
    require(
        all(count(h.get(k)) for k in ("samples", "underflow", "overflow", "total_weight"))
        and h["samples"] == h["total_weight"] == n
        and sum(h["counts"]) + h["underflow"] + h["overflow"] == n,
        "native snapshot marginal count closure",
    )
    s = h.get("statistics")
    if n == 0:
        require(s is None, "empty computed snapshot needs null statistics")
        return
    require(
        isinstance(s, dict)
        and all(finite(s.get(k)) for k in ("mean", "variance", "min", "max"))
        and s["variance"] >= 0
        and 0 <= s["min"] <= s["mean"] <= s["max"],
        "invalid current-record statistics",
    )
    if key != "pT":
        require(
            s["max"] <= math.pi * (0.5 if key == "theta_folded" else 1) + 1e-12,
            "snapshot angle outside physical range",
        )
    if sum(h["counts"]) == 0:
        require(
            s.get("mode") is None
            and s.get("mode_method") == "unavailable: no in-range marginal records",
            "all-flow snapshot cannot infer a marginal mode",
        )
        return
    require(finite(s.get("mode")), "nonfinite snapshot marginal mode")
    require(s.get("mode_method") == MODE, "unknown snapshot mode estimator")
    density = [w / (b - a) for w, a, b in zip(h["counts"], edges, edges[1:])]
    maximum = max(density)
    require(
        any(
            d == maximum and math.isclose(s["mode"], (a + b) / 2, rel_tol=1e-12, abs_tol=1e-30)
            for d, a, b in zip(density, edges, edges[1:])
        ),
        "snapshot density mode mismatch",
    )


def validate_raw(raw, manifest=None, flags=None):
    from collider_secondary_positions import definition_valid, validate as validate_positions

    require(
        isinstance(raw, dict)
        and raw.get("schema_version") == SCHEMA
        and raw.get("phase") == PHASE
        and raw.get("coordinate_frame") == "native_pair_tracking"
        and raw.get("weighting") == WEIGHTING,
        "unknown native secondary snapshot contract",
    )
    require(
        isinstance(raw.get("definitions"), dict)
        and set(raw["definitions"])
        == {"selection", "timing", "normalization", "coordinates", "precision"}
        and all(isinstance(v, str) and 0 < len(v) <= 2048 for v in raw["definitions"].values()),
        "missing native snapshot definitions",
    )
    if flags is not None:
        actual = raw.get("evaluated_flags")
        require(
            isinstance(actual, dict) and set(actual) == set(flags),
            "snapshot evaluated flag keys differ from native summary",
        )
        for k, expected in flags.items():
            observed = actual[k]
            if k in ("pair_ratio", "pair_ecut"):
                require(
                    finite(observed)
                    and math.isclose(observed, expected, rel_tol=5e-6, abs_tol=5e-12),
                    "snapshot evaluated floating flags differ from printed native summary",
                )
            elif k == "do_bhabhas" and expected is None:
                require(observed == 0 and flags["do_pairs"], "unknown non-pair native frame layout")
            else:
                require(
                    type(observed) is int and observed == expected,
                    "snapshot evaluated integer flags differ from native summary",
                )
    axes = raw.get("axes")
    axes_valid(axes)
    has_positions = "position_z_definition" in raw
    if has_positions:
        definition_valid(raw["position_z_definition"])
    frames = raw.get("frames")
    require(
        isinstance(frames, list) and 1 <= len(frames) <= MAX_FRAMES, "native snapshot frame bound"
    )
    if manifest is not None:
        require(len(frames) == len(manifest["frames"]), "snapshot/display frame count mismatch")
    previous = -1
    for i, f in enumerate(frames):
        require(
            type(f.get("id")) is int
            and f["id"] == i
            and type(f.get("diagonal")) is int
            and f["diagonal"] > previous,
            "invalid native snapshot identity",
        )
        previous = f["diagonal"]
        if manifest is not None:
            require(
                manifest["frames"][i]["id"] == i
                and manifest["frames"][i]["diagonal"] == f["diagonal"],
                "snapshot/display diagonal mismatch",
            )
        require(
            type(f.get("available")) is bool and count(f.get("nrecords")),
            "invalid snapshot availability/count",
        )
        n = f["nrecords"]
        if not f["available"]:
            require(
                isinstance(f.get("reason"), str)
                and f["reason"]
                and n == 0
                and all(
                    f.get(k) is None for k in ("joint", "joint_folded", "marginals", "selection")
                ),
                "unavailable native snapshot carries data",
            )
            require(
                f.get("position_z") is None, "unavailable native snapshot carries stored-z data"
            )
            if has_positions:
                require(
                    "position_z" in f,
                    "unavailable native snapshot must explicitly omit stored-z data with null",
                )
            continue
        require(f.get("reason") is None, "computed snapshot carries unavailable reason")
        joint_valid(f.get("joint"), n)
        joint_valid(f.get("joint_folded"), n)
        require(
            isinstance(f.get("marginals"), dict) and set(f["marginals"]) == set(KEYS),
            "unknown native snapshot marginals",
        )
        for key in KEYS:
            marginal_valid(f["marginals"][key], n, axes[MARGINAL_KEYS[key]], key)
        s = f.get("selection")
        require(
            isinstance(s, dict)
            and count(s.get("admitted_tracker_count"))
            and count(s.get("authoritative_reserve_count"))
            and s["admitted_tracker_count"] == s["authoritative_reserve_count"] == n,
            "snapshot authoritative population count mismatch",
        )
        for key in ("process_counts", "signed_energy_counts"):
            require(
                isinstance(s.get(key), dict)
                and all(count(v) for v in s[key].values())
                and sum(s[key].values()) == n,
                "snapshot process/sign count closure",
            )
        require(
            set(s["signed_energy_counts"]) == {"positive_energy", "negative_energy"}
            and len(s["process_counts"]) <= 128,
            "unknown snapshot sign/process keys",
        )
        require(
            all(str(int(k)) == k and -999 <= int(k) <= 999 for k in s["process_counts"]),
            "invalid snapshot process identifier",
        )
        if has_positions:
            validate_positions(f.get("position_z"), f, axes)
        else:
            require(f.get("position_z") is None, "snapshot stored-z data lacks definition")
    return raw


def expand(raw, manifest, flags):
    import collider_secondary_kinematics as adapter

    validate_raw(raw, manifest, flags)
    key, label = adapter._classification(flags)
    frames = []
    for native, display in zip(raw["frames"], manifest["frames"]):
        pops = {
            k: adapter._unavailable(
                title, "This population is not separately identified in the native snapshot"
            )
            for k, title in [
                ("pairs", "Produced pairs"),
                ("deflected", "Deflected particles"),
                ("combined", "Combined tracked population"),
            ]
        }
        n = native["nrecords"]
        available = native["available"]
        reason = native["reason"]
        if available:
            require(
                flags["track_pairs"] and key is not None,
                "computed snapshot in unadmitted tracking mode",
            )

            def joint(which, angle):
                return dict(
                    native[which],
                    theta_edges_rad=raw["axes"][angle],
                    pT_edges_GeV=raw["axes"]["pT_edges_GeV"],
                    storage_order="row-major-pT-theta",
                )

            histograms = {}
            for k in KEYS:
                h = native["marginals"][k]
                edges = raw["axes"][MARGINAL_KEYS[k]]
                histograms[k] = dict(
                    quantity=QUANTITIES[k],
                    unit="GeV/c" if k == "pT" else "rad",
                    weight_unit="stored particle records",
                    weighting=WEIGHTING,
                    normalization="bin-integrated",
                    bins=[
                        dict(low=a, high=b, weight=w)
                        for a, b, w in zip(edges, edges[1:], h["counts"])
                    ],
                    **{
                        p: h[p]
                        for p in ("samples", "underflow", "overflow", "total_weight", "statistics")
                    },
                )
            selection = dict(
                native["selection"],
                pair_ratio=flags["pair_ratio"],
                pair_ecut_GeV=flags["pair_ecut"],
                origin_separation=False,
            )
            pops[key] = dict(
                label=label,
                available=True,
                reason=None,
                nrecords=n,
                joint=joint("joint", "theta_edges_rad"),
                joint_folded=joint("joint_folded", "theta_folded_edges_rad"),
                distributions=histograms,
                selection=selection,
                normalization=dict(
                    kind="stored-record counts",
                    weight_unit="stored particle records",
                    weighting=WEIGHTING,
                    physical_weight_available=False,
                    completeness_verified=False,
                ),
            )
            if "position_z_definition" in raw:
                pops[key]["position_z"] = copy.deepcopy(native["position_z"])
        elif key is not None:
            pops[key]["reason"] = reason
        frames.append(
            dict(
                id=native["id"],
                diagonal=native["diagonal"],
                elapsed_s=display["elapsed_s"],
                phase=PHASE,
                available=available,
                reason=reason,
                populations=pops,
                tracker_count_verification=dict(
                    available=available,
                    admitted_tracker_count=n if available else None,
                    authoritative_reserve_count=n if available else None,
                    actual_records=n,
                    reason=reason,
                ),
            )
        )
    return frames


def validate_extension(data, manifest=None, raw=None):
    """Check additive frame payloads; final top-level populations remain separate."""
    refs = data["source_artifacts"]
    reference = refs.get("frame_snapshots")
    frames = data.get("frames", [])
    if reference is None:
        require(
            not data["frame_availability"] and frames == [],
            "frames require an authenticated native snapshot source",
        )
        return
    require(
        reference["path"] == "gui_secondary_frames.json" and 0 < reference["bytes"] <= MAX_BYTES,
        "invalid native snapshot source role/bound",
    )
    require(
        isinstance(frames, list) and 1 <= len(frames) <= MAX_FRAMES, "expanded snapshot frame bound"
    )
    if manifest is not None:
        require(
            manifest.get("secondary_frames_file") == reference["path"],
            "native manifest snapshot declaration mismatch",
        )
        require(len(frames) == len(manifest["frames"]), "expanded snapshot/display count mismatch")
    definition = data.get("frame_definition")
    require(
        isinstance(definition, dict)
        and definition.get("schema_version") == SCHEMA
        and definition.get("phase") == PHASE
        and definition.get("weighting") == WEIGHTING
        and definition.get("coordinate_frame") == "native_pair_tracking",
        "invalid frame model definition",
    )
    has_positions = "position_z_definition" in definition
    if has_positions:
        from collider_secondary_positions import definition_valid

        definition_valid(definition["position_z_definition"])
        require(
            data.get("position_z_definition") == definition["position_z_definition"],
            "frame stored-z definition differs from global definition",
        )
    require(
        type(data["frame_availability"]) is bool
        and data["frame_availability"] == any(f.get("available") is True for f in frames),
        "frame availability summary mismatch",
    )
    if raw is not None:
        expected = expand(raw, manifest, data["evaluated_flags"])
        require(
            frames == expected and definition["definitions"] == raw["definitions"],
            "expanded frame data differs from authenticated native histograms",
        )
        require(
            definition.get("position_z_definition") == raw.get("position_z_definition"),
            "expanded frame stored-z capability differs from native source",
        )
    # Reconstruct compact records from the expanded schema for independent
    # closure/geometry checks even when the raw source is not supplied.
    native = dict(
        schema_version=SCHEMA,
        phase=PHASE,
        coordinate_frame="native_pair_tracking",
        weighting=WEIGHTING,
        definitions=definition.get("definitions"),
        evaluated_flags=data["evaluated_flags"],
        axes=definition.get("axes"),
        frames=[],
    )
    if has_positions:
        native["position_z_definition"] = definition["position_z_definition"]
    import collider_secondary_kinematics as adapter

    key, label = adapter._classification(data["evaluated_flags"])
    previous_time = -math.inf
    for i, f in enumerate(frames):
        require(
            f.get("phase") == PHASE
            and finite(f.get("elapsed_s"))
            and f["elapsed_s"] > previous_time,
            "invalid snapshot timing",
        )
        previous_time = f["elapsed_s"]
        if manifest is not None:
            require(
                f["elapsed_s"] == manifest["frames"][i]["elapsed_s"],
                "snapshot elapsed time differs from native manifest",
            )
        pops = f.get("populations")
        require(
            isinstance(pops, dict) and set(pops) == {"pairs", "deflected", "combined"},
            "invalid snapshot populations",
        )
        available = [(k, p) for k, p in pops.items() if p.get("available") is True]
        require(
            len(available) == (1 if f.get("available") is True else 0),
            "snapshot population availability mismatch",
        )
        compact = dict(
            id=f.get("id"),
            diagonal=f.get("diagonal"),
            available=f.get("available"),
            reason=f.get("reason"),
            nrecords=0,
            joint=None,
            joint_folded=None,
            marginals=None,
            selection=None,
        )
        if has_positions:
            compact["position_z"] = None
        for k, p in pops.items():
            require(
                isinstance(p.get("label"), str) and 0 < len(p["label"]) <= 1024,
                "invalid snapshot population label",
            )
            require(type(p.get("available")) is bool, "snapshot population needs availability")
            if not p["available"]:
                require(
                    p.get("nrecords") == 0
                    and isinstance(p.get("reason"), str)
                    and p["reason"]
                    and all(
                        p.get(k2) is None
                        for k2 in ("joint", "joint_folded", "selection", "normalization")
                    )
                    and p.get("distributions") == dict(theta=None, pT=None, theta_folded=None),
                    "unavailable snapshot population carries data",
                )
                require(
                    p.get("position_z") is None,
                    "unavailable snapshot population carries stored-z data",
                )
        if available:
            popkey, p = available[0]
            n = p.get("nrecords")
            require(
                popkey == key
                and p["label"] == label
                and data["evaluated_flags"]["track_pairs"]
                and p.get("reason") is None
                and count(n),
                "unadmitted snapshot population",
            )
            compact["nrecords"] = n
            for jointkey, angle in [("joint", "theta"), ("joint_folded", "theta_folded")]:
                j = p[jointkey]
                require(
                    j.get("storage_order") == "row-major-pT-theta"
                    and j.get("theta_edges_rad") == native["axes"][AXIS_KEYS[angle]]
                    and j.get("pT_edges_GeV") == native["axes"]["pT_edges_GeV"],
                    "snapshot joint axes disagree with model",
                )
                compact[jointkey] = {
                    k: j[k] for k in ("counts", "samples", "in_range_records", "excluded")
                }
            compact["marginals"] = {}
            for k in KEYS:
                h = p["distributions"][k]
                edges = native["axes"][MARGINAL_KEYS[k]]
                require(
                    h.get("quantity") == QUANTITIES[k]
                    and h.get("unit") == ("GeV/c" if k == "pT" else "rad")
                    and h.get("weight_unit") == "stored particle records"
                    and h.get("weighting") == WEIGHTING
                    and h.get("normalization") == "bin-integrated",
                    "invalid snapshot histogram weighting/units",
                )
                require(
                    len(h["bins"]) == 160
                    and all(
                        b["low"] == a and b["high"] == z
                        for b, a, z in zip(h["bins"], edges, edges[1:])
                    ),
                    "snapshot marginal axis mismatch",
                )
                compact["marginals"][k] = dict(
                    counts=[b["weight"] for b in h["bins"]],
                    **{
                        k2: h[k2]
                        for k2 in ("samples", "underflow", "overflow", "total_weight", "statistics")
                    },
                )
            s = p["selection"]
            flags = data["evaluated_flags"]
            require(
                s.get("origin_separation") is False
                and s.get("pair_ratio") == flags["pair_ratio"]
                and s.get("pair_ecut_GeV") == flags["pair_ecut"],
                "snapshot selection mismatch",
            )
            compact["selection"] = {
                k: s[k]
                for k in (
                    "process_counts",
                    "signed_energy_counts",
                    "admitted_tracker_count",
                    "authoritative_reserve_count",
                )
            }
            if has_positions:
                compact["position_z"] = p.get("position_z")
            else:
                require(p.get("position_z") is None, "frame stored-z data lacks native capability")
            norm = p.get("normalization")
            require(
                norm
                == dict(
                    kind="stored-record counts",
                    weight_unit="stored particle records",
                    weighting=WEIGHTING,
                    physical_weight_available=False,
                    completeness_verified=False,
                ),
                "snapshot is not a final count-completeness claim",
            )
        verification = f.get("tracker_count_verification")
        n = compact["nrecords"]
        a = f.get("available")
        require(
            verification
            == dict(
                available=a,
                admitted_tracker_count=n if a else None,
                authoritative_reserve_count=n if a else None,
                actual_records=n,
                reason=f.get("reason"),
            ),
            "snapshot count verification mismatch",
        )
        native["frames"].append(compact)
    validate_raw(native, manifest)


def attach(data, raw, source_reference, manifest):
    data["source_artifacts"]["frame_snapshots"] = source_reference
    data["frame_definition"] = dict(
        schema_version=SCHEMA,
        phase=PHASE,
        coordinate_frame="native_pair_tracking",
        weighting=WEIGHTING,
        definitions=copy.deepcopy(raw["definitions"]),
        axes=copy.deepcopy(raw["axes"]),
    )
    if "position_z_definition" in raw:
        data["position_z_definition"] = copy.deepcopy(raw["position_z_definition"])
        data["frame_definition"]["position_z_definition"] = copy.deepcopy(
            raw["position_z_definition"]
        )
    data["frames"] = expand(raw, manifest, data["evaluated_flags"])
    data["frame_availability"] = any(f["available"] for f in data["frames"])
    data["definitions"][
        "time"
    ] = "Top-level populations are final stored output; optional frames are native current-state snapshots and are never summed into the final population."
    validate_extension(data, manifest, raw)
    return data
