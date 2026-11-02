"""Bounded validation of native 2D field samples; no physics qualification."""

import math
import re

SCHEMA = "feebl.collider.fields/1"
MAX_VALUES = 64 * 64 * 64 * 8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def integer(x, lo, hi):
    return isinstance(x, int) and not isinstance(x, bool) and lo <= x <= hi


def text(x):
    return isinstance(x, str) and 0 < len(x) <= 2048


MAX_LONGITUDINAL_VALUES = 524288
LONG_SELECTION = "all active source-slice encounters, final substep, pre-kick"
LONG_TIMING = "field samples are post-solve/pre-kick on the final substep; particles are displayed after the diagonal barrier"
LONG_Z_MAPPING = "source-specific nominal slice bands under the legacy particle-dump z convention; raw encounter z is retained separately"


def near(a, b):
    return finite(a) and finite(b) and abs(a - b) <= 1e-10 * max(1, abs(b))


def validate_longitudinal(data):
    metadata = data.get("longitudinal")
    if metadata is None:
        require(
            "longitudinal" not in data and all("longitudinal" not in f for f in data["frames"]),
            "longitudinal samples without geometry",
        )
        return None
    require(
        isinstance(metadata, dict)
        and metadata.get("schema_version") == "feebl.collider.longitudinal/1",
        "unknown longitudinal field schema",
    )
    require(
        metadata.get("coordinate_unit") == "nm" and metadata.get("frame") == "legacy_collider_dump",
        "longitudinal coordinate convention",
    )
    n = data["conventions"]["n_z"]
    dz = metadata.get("delta_z_nm")
    require(
        integer(metadata.get("n_z"), 1, 1000000) and metadata["n_z"] == n and finite(dz) and dz > 0,
        "longitudinal slice geometry",
    )
    for key, expected in [
        ("selection", LONG_SELECTION),
        ("timing", LONG_TIMING),
        ("z_mapping", LONG_Z_MAPPING),
    ]:
        require(metadata.get(key) == expected, "longitudinal " + key + " convention")
    sources = {v["id"]: v for v in data["sources"]}
    require(
        set(sources) == {"beam1", "beam2"}
        and all(
            sources["beam" + str(i)]["kind"] == "beam_beam"
            and sources["beam" + str(i)]["source_beam"] == i
            for i in (1, 2)
        ),
        "longitudinal native source identities",
    )
    planes = metadata.get("planes")
    g = data["geometry"]
    require(isinstance(planes, dict) and set(planes) == {"zx", "zy"}, "longitudinal planes")
    cap = min(64, 4096 // n) if n <= 128 else 2
    for key, axis, fixed in [("zx", "x", "y"), ("zy", "y", "x")]:
        q = planes[key]
        require(
            isinstance(q, dict)
            and q.get("axes") == ["z", axis]
            and q.get("fixed_axis") == fixed
            and finite(q.get("fixed_coordinate_nm"))
            and q["fixed_coordinate_nm"] == 0,
            "longitudinal cross-section plane",
        )
        require(
            integer(q.get("samples"), 2, cap)
            and near(q.get("min_nm"), g[axis + "_min_nm"])
            and near(q.get("max_nm"), g[axis + "_max_nm"]),
            "longitudinal sample geometry",
        )
    maxima = {key: {view: 0.0 for view in planes} for key in sources}
    expected_total = (
        sum(min(f["diagonal"], n - 1) - max(0, f["diagonal"] - n + 1) + 1 for f in data["frames"])
        if n <= 128
        else 0
    )
    require(
        2 * sum(q["samples"] for q in planes.values()) * expected_total <= MAX_LONGITUDINAL_VALUES,
        "longitudinal sample budget",
    )
    value_count = 0
    available_frames = 0
    available_strips = 0
    for f in data["frames"]:
        record = f.get("longitudinal")
        d = f["diagonal"]
        first = max(0, d - n + 1)
        last = min(d, n - 1)
        expected = last - first + 1
        require(
            isinstance(record, dict)
            and isinstance(record.get("available"), bool)
            and integer(record.get("expected_encounters"), 1, n)
            and record["expected_encounters"] == expected,
            "longitudinal frame coverage",
        )
        strips = record.get("strips")
        require(isinstance(strips, list), "longitudinal strips")
        if n > 128:
            require(
                not record["available"] and text(record.get("reason")) and not strips,
                "unsupported longitudinal slice count requires no strips",
            )
            continue
        require(len(strips) == expected, "longitudinal active encounter coverage")
        any_available = False
        ordinals = set()
        for offset, strip in enumerate(strips):
            i1 = first + offset
            i2 = d - i1
            z = (i2 - i1) * dz / 2
            require(
                isinstance(strip, dict)
                and integer(strip.get("beam1_slice"), 0, n - 1)
                and integer(strip.get("beam2_slice"), 0, n - 1)
                and strip["beam1_slice"] == i1
                and strip["beam2_slice"] == i2,
                "longitudinal encounter order",
            )
            require(near(strip.get("nominal_z_nm"), z), "longitudinal nominal z")
            bounds = strip.get("nominal_z_bounds_nm")
            require(
                isinstance(bounds, list)
                and len(bounds) == 2
                and near(bounds[0], z - dz / 2)
                and near(bounds[1], z + dz / 2),
                "longitudinal nominal z bounds",
            )
            centres = strip.get("display_z_nm")
            bands = strip.get("display_z_bounds_nm")
            require(
                isinstance(centres, dict)
                and isinstance(bands, dict)
                and set(centres) == set(sources)
                and set(bands) == set(sources),
                "longitudinal source-specific z geometry",
            )
            for source, sign in [("beam1", 1), ("beam2", -1)]:
                centre = -z + sign * dz
                band = bands[source]
                require(
                    near(centres[source], centre)
                    and isinstance(band, list)
                    and len(band) == 2
                    and near(band[0], centre - dz / 2)
                    and near(band[1], centre + dz / 2),
                    "longitudinal display z mapping",
                )
            require(isinstance(strip.get("available"), bool), "longitudinal strip availability")
            if not strip["available"]:
                require(
                    text(strip.get("reason")) and strip.get("maps") is None,
                    "unavailable longitudinal strip carries data",
                )
                q, counter = strip.get("substep_index"), strip.get("encounter_counter")
                if q is not None or counter is not None:
                    require(
                        integer(q, 0, data["conventions"]["n_t"] - 1)
                        and q == data["conventions"]["n_t"] - 1
                        and integer(counter, 0, 9007199254740991)
                        and counter not in ordinals,
                        "unavailable longitudinal capture receipt",
                    )
                    ordinals.add(counter)
                continue
            any_available = True
            available_strips += 1
            require(
                strip.get("reason") is None
                and integer(strip.get("substep_index"), 0, data["conventions"]["n_t"] - 1)
                and strip["substep_index"] == data["conventions"]["n_t"] - 1,
                "longitudinal evaluation phase",
            )
            counter = strip.get("encounter_counter")
            require(
                integer(counter, 0, 9007199254740991) and counter not in ordinals,
                "longitudinal encounter ordinal",
            )
            ordinals.add(counter)
            require(
                data["conventions"]["charge_sign"] == -1
                and data["conventions"]["scale_step"] == [1, 1],
                "unadmitted longitudinal field normalization",
            )
            require(
                g["x_min_nm"] <= 0 <= g["x_max_nm"] and g["y_min_nm"] <= 0 <= g["y_max_nm"],
                "longitudinal cross-section outside sample domain",
            )
            maps = strip.get("maps")
            require(
                isinstance(maps, dict) and set(maps) == set(sources), "longitudinal source maps"
            )
            for source, views in maps.items():
                require(
                    isinstance(views, dict) and set(views) == set(planes), "longitudinal map planes"
                )
                for view, values in views.items():
                    require(
                        isinstance(values, list)
                        and len(values) == planes[view]["samples"]
                        and all(finite(v) and v >= 0 for v in values),
                        "longitudinal values",
                    )
                    maxima[source][view] = max(maxima[source][view], max(values))
                    value_count += len(values)
        require(
            record["available"] == any_available
            and (record.get("reason") is None if any_available else text(record.get("reason"))),
            "longitudinal frame availability mismatch",
        )
        available_frames += int(any_available)
    require(value_count <= MAX_LONGITUDINAL_VALUES, "longitudinal value bound")
    return dict(
        available_frames=available_frames,
        available_strips=available_strips,
        scalar_count=value_count,
        source_maxima=maxima,
    )


MAX_SNAPSHOT_VALUES = 1048576
SNAPSHOT_SELECTION = "all primary source slices at every saved diagonal"
SNAPSHOT_TIMING = (
    "post-diagonal; all source copies drift-synchronized with the sampled particle display"
)
SNAPSHOT_MODEL = "quasi-static native transverse slice-field reconstruction; primary sources only; not a full 3D electromagnetic field"


def validate_snapshot(data):
    if "snapshot" not in data:
        require(
            all("snapshot" not in f for f in data["frames"]), "snapshot samples without geometry"
        )
        return None
    m = data["snapshot"]
    g = data["geometry"]
    c = data["conventions"]
    n = c["n_z"]
    dz = m.get("delta_z_nm") if isinstance(m, dict) else None
    require(
        isinstance(m, dict)
        and m.get("schema_version") == "feebl.collider.snapshot-fields/1"
        and m.get("coordinate_unit") == "nm"
        and m.get("frame") == "legacy_collider_dump",
        "unsupported snapshot field schema/coordinates",
    )
    require(
        integer(m.get("n_z"), 1, 1000000) and m["n_z"] == n and finite(dz) and dz > 0,
        "invalid snapshot slice geometry",
    )
    require(
        m.get("selection") == SNAPSHOT_SELECTION
        and m.get("timing") == SNAPSHOT_TIMING
        and m.get("model") == SNAPSHOT_MODEL,
        "unsupported snapshot field conventions",
    )
    sources = {v["id"]: v for v in data["sources"]}
    require(
        set(sources) == {"beam1", "beam2"}
        and all(
            sources["beam" + str(i)]["kind"] == "beam_beam"
            and sources["beam" + str(i)]["source_beam"] == i
            for i in (1, 2)
        ),
        "snapshot native source identities",
    )
    if data.get("longitudinal"):
        require(
            near(dz, data["longitudinal"]["delta_z_nm"]),
            "snapshot/encounter slice spacing mismatch",
        )
    planes = m.get("planes")
    require(isinstance(planes, dict) and set(planes) == {"zx", "zy"}, "snapshot planes")
    cap = min(64, 4096 // n) if n <= 128 else 2
    for view, axis, fixed in [("zx", "x", "y"), ("zy", "y", "x")]:
        q = planes[view]
        require(
            isinstance(q, dict)
            and q.get("axes") == ["z", axis]
            and q.get("fixed_axis") == fixed
            and q.get("fixed_coordinate_nm") == 0
            and finite(q.get("fixed_coordinate_nm")),
            "snapshot cross-section plane",
        )
        require(
            integer(q.get("samples"), 2, cap)
            and near(q.get("min_nm"), g[axis + "_min_nm"])
            and near(q.get("max_nm"), g[axis + "_max_nm"]),
            "snapshot sample geometry",
        )
    if n <= 128:
        require(
            len(data["frames"]) * 2 * n * sum(q["samples"] for q in planes.values())
            <= MAX_SNAPSHOT_VALUES,
            "snapshot sample budget",
        )
    maxima = {id: {view: 0.0 for view in planes} for id in sources}
    value_count = 0
    available_frames = 0
    available_strips = 0
    for f in data["frames"]:
        r = f.get("snapshot")
        d = f["diagonal"]
        require(
            isinstance(r, dict)
            and isinstance(r.get("available"), bool)
            and integer(r.get("diagonal"), 0, 2 * n - 2)
            and r["diagonal"] == d
            and r.get("phase") == "post-diagonal/display-synchronized"
            and integer(r.get("expected_slices"), 1, 1000000)
            and r["expected_slices"] == n,
            "snapshot frame/phase mismatch",
        )
        if not r["available"]:
            require(
                text(r.get("reason")) and r.get("sources") is None,
                "unavailable snapshot carries sources",
            )
            continue
        require(
            n <= 128
            and r.get("reason") is None
            and isinstance(r.get("sources"), dict)
            and set(r["sources"]) == set(sources),
            "snapshot availability/sources",
        )
        require(
            c["charge_sign"] == -1 and c["scale_step"] == [1, 1],
            "unadmitted snapshot normalization",
        )
        require(
            g["x_min_nm"] <= 0 <= g["x_max_nm"] and g["y_min_nm"] <= 0 <= g["y_max_nm"],
            "snapshot cross-section outside sample domain",
        )
        available_frames += 1
        for id, sign in [("beam1", 1), ("beam2", -1)]:
            source = r["sources"][id]
            require(
                isinstance(source, dict)
                and isinstance(source.get("strips"), list)
                and len(source["strips"]) == n,
                "incomplete snapshot source coverage",
            )
            for i, s in enumerate(source["strips"]):
                centre = sign * (i + 1 - d / 2) * dz
                bounds = s.get("display_z_bounds_nm") if isinstance(s, dict) else None
                require(
                    isinstance(s, dict)
                    and integer(s.get("slice"), 0, n - 1)
                    and s["slice"] == i
                    and near(s.get("display_z_nm"), centre)
                    and isinstance(bounds, list)
                    and len(bounds) == 2
                    and near(bounds[0], centre - dz / 2)
                    and near(bounds[1], centre + dz / 2),
                    "snapshot source-slice display geometry mismatch",
                )
                require(
                    all(
                        integer(s.get(key), 0, 9007199254740991)
                        for key in ("source_particles", "deposited_particles", "outside_particles")
                    )
                    and s["deposited_particles"] + s["outside_particles"] == s["source_particles"],
                    "snapshot source/deposition count closure",
                )
                maps = s.get("maps")
                require(isinstance(maps, dict) and set(maps) == set(planes), "snapshot map planes")
                for view, values in maps.items():
                    require(
                        isinstance(values, list)
                        and len(values) == planes[view]["samples"]
                        and all(finite(v) and v >= 0 for v in values),
                        "snapshot values",
                    )
                    maxima[id][view] = max(maxima[id][view], max(values))
                    value_count += len(values)
                available_strips += 1
    require(value_count <= MAX_SNAPSHOT_VALUES, "snapshot scalar count bound")
    return dict(
        available_frames=available_frames,
        available_strips=available_strips,
        scalar_count=value_count,
        source_maxima=maxima,
    )


def validate(data, manifest_frames):
    require(isinstance(data, dict) and data.get("schema_version") == SCHEMA, "unknown field schema")
    plane = data.get("plane")
    require(
        isinstance(plane, dict)
        and plane.get("axes") == ["x", "y"]
        and plane.get("coordinate_unit") == "nm"
        and plane.get("frame") == "native simulated frame",
        "unsupported field plane",
    )
    g = data.get("geometry")
    require(
        isinstance(g, dict) and integer(g.get("nx"), 2, 64) and integer(g.get("ny"), 2, 64),
        "field geometry size bound",
    )
    require(
        g.get("storage_order") == "row-major-y-x"
        and g.get("sampling") == "uniform CIC gather points; bounds are sample centres",
        "field storage convention",
    )
    for axis in ("x", "y"):
        lo, hi = g.get(axis + "_min_nm"), g.get(axis + "_max_nm")
        require(
            finite(lo) and finite(hi) and hi > lo and math.isfinite(hi - lo),
            "invalid field geometry extent",
        )
    domain = g.get("native_domain_nm")
    require(
        isinstance(domain, list)
        and len(domain) == 4
        and all(finite(v) for v in domain)
        and domain[0] < domain[1]
        and domain[2] < domain[3],
        "invalid native field domain",
    )
    require(
        domain[0] <= g["x_min_nm"] < g["x_max_nm"] <= domain[1]
        and domain[2] <= g["y_min_nm"] < g["y_max_nm"] <= domain[3],
        "field samples outside native domain",
    )
    sources = data.get("sources")
    require(isinstance(sources, list) and 1 <= len(sources) <= 8, "field source bound")
    ids = set()
    for s in sources:
        require(
            isinstance(s, dict)
            and isinstance(s.get("id"), str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", s["id"]),
            "invalid field source ID",
        )
        require(s["id"] not in ids, "duplicate field source")
        ids.add(s["id"])
        require(s.get("kind") in ("beam_beam", "external"), "unsupported field source kind")
        require(
            text(s.get("label")) and text(s.get("quantity")) and s.get("unit") in ("GV/nm", "V/m"),
            "field source quantity/unit",
        )
        if s["kind"] == "beam_beam":
            require(
                integer(s.get("source_beam"), 1, 2)
                and integer(s.get("target_beam"), 1, 2)
                and s["target_beam"] == 3 - s["source_beam"],
                "field beam identity",
            )
            require(
                s["quantity"] == "native transverse electric-equivalent field magnitude"
                and s["unit"] == "GV/nm",
                "native field definition",
            )
        else:
            require(
                s.get("source_beam") is None and s.get("target_beam") is None,
                "external field beam identity",
            )
    c = data.get("conventions")
    require(
        isinstance(c, dict)
        and c.get("selection") == "central active slice encounter, final substep, pre-kick"
        and c.get("scope") == "single source slice; not whole-bunch or diagonal-averaged field",
        "field sampling convention",
    )
    require(finite(c.get("charge_sign")) and c["charge_sign"] in (-1, 0, 1), "field charge sign")
    require(
        isinstance(c.get("scale_step"), list)
        and len(c["scale_step"]) == 2
        and all(finite(v) and v > 0 for v in c["scale_step"]),
        "field scale step",
    )
    require(
        integer(c.get("n_z"), 1, 1000000) and integer(c.get("n_t"), 1, 1000000),
        "field slice/time counts",
    )
    frames = data.get("frames")
    require(
        isinstance(frames, list) and len(frames) == len(manifest_frames) and 1 <= len(frames) <= 64,
        "field frame count mismatch",
    )
    require(len(frames) * len(ids) * g["nx"] * g["ny"] <= MAX_VALUES, "field dataset value bound")
    available = 0
    previous = -1
    maxima = {i: 0.0 for i in ids}
    for i, (f, m) in enumerate(zip(frames, manifest_frames)):
        require(
            isinstance(f, dict)
            and integer(f.get("id"), 0, 63)
            and f["id"] == i
            and integer(f.get("diagonal"), 0, 2 * c["n_z"] - 2)
            and f["diagonal"] == m.get("diagonal")
            and f["diagonal"] > previous,
            "field frame/diagonal mismatch",
        )
        previous = f["diagonal"]
        require(
            finite(f.get("elapsed_s"))
            and f["elapsed_s"] >= 0
            and finite(m.get("elapsed_s"))
            and math.isclose(f["elapsed_s"], m["elapsed_s"], rel_tol=1e-12, abs_tol=1e-30),
            "field display timestamp mismatch",
        )
        require(isinstance(f.get("available"), bool), "field availability required")
        if not f["available"]:
            require(
                text(f.get("reason")) and f.get("maps") is None and f.get("encounter") is None,
                "unavailable field requires reason and null data",
            )
            continue
        available += 1
        require(f.get("reason") is None, "available field cannot have unavailable reason")
        require(
            c["charge_sign"] == -1 and c["scale_step"] == [1, 1], "unadmitted field normalization"
        )
        e = f.get("encounter")
        diag = f["diagonal"]
        require(
            isinstance(e, dict)
            and integer(e.get("beam1_slice"), 0, c["n_z"] - 1)
            and integer(e.get("beam2_slice"), 0, c["n_z"] - 1)
            and e["beam1_slice"] == diag // 2
            and e["beam2_slice"] == diag - diag // 2,
            "central field encounter mismatch",
        )
        require(
            integer(e.get("substep_index"), 0, c["n_t"] - 1)
            and e["substep_index"] == c["n_t"] - 1
            and integer(e.get("encounter_counter"), 0, 9007199254740991)
            and finite(e.get("nominal_z_nm"))
            and e.get("phase") == "post-solve/pre-kick",
            "field encounter phase",
        )
        maps = f.get("maps")
        require(isinstance(maps, dict) and set(maps) == ids, "field map source mismatch")
        for key, values in maps.items():
            require(
                isinstance(values, list) and len(values) == g["nx"] * g["ny"],
                "field map dimension mismatch",
            )
            require(
                all(finite(v) and v >= 0 for v in values), "nonfinite or negative field magnitude"
            )
            maxima[key] = max(maxima[key], max(values))
    longitudinal = validate_longitudinal(data)
    snapshot = validate_snapshot(data)
    require(
        len(frames) * len(ids) * g["nx"] * g["ny"]
        + (longitudinal["scalar_count"] if longitudinal else 0)
        + (snapshot["scalar_count"] if snapshot else 0)
        <= MAX_VALUES,
        "combined field dataset value bound",
    )
    return dict(
        frames=len(frames),
        available_frames=available,
        source_maxima=maxima,
        longitudinal=longitudinal,
        snapshot=snapshot,
        physics_validation="NOT_ESTABLISHED",
    )
