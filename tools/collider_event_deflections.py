"""Audited imported-event angular deflections; no identity inference or simulation."""

import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from collider_secondary_kinematics import evaluated_flags, _read_bytes

SCHEMA = "feebl.collider.event-deflections/1"
RAW_SCHEMA = "feebl.collider.imported-trajectories/1"
IDENTITY_SCHEMA = "feebl.collider.manufactured-events/1"
MAX_RECORDS = 8192
MAX_FRAMES = 64
MAX_BYTES = 16 * 1024 * 1024
BINS = 160
INITIAL_COLUMNS = [
    "label",
    "source_record",
    "scheduled_time_counter",
    "admitted_time_counter",
    "signed_energy_GeV",
    "x_nm",
    "y_nm",
    "z_nm",
    "vx",
    "vy",
    "vz",
]
CURRENT_COLUMNS = [
    "label",
    "vx",
    "vy",
    "vz",
    "main_mesh_gathers",
    "outer_mesh_gathers",
    "monopole_gathers",
]
DEFINITIONS = dict(
    theta="atan2(hypot(vx,vy),vz)",
    phi="atan2(vy,vx)",
    delta_theta="theta_current - theta_initial",
    delta_phi="wrapped current phi minus initial phi in [-180,180) degrees",
    initial_reference="actual native float state at admission",
    weighting="one count per admitted tracked record; no luminosity or event weighting",
)
MODE = "centre of highest count-density histogram bin; approximate"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(v):
    return type(v) in (float, int) and math.isfinite(v)


def integer(v, lo=0, hi=2**53 - 1):
    return type(v) is int and lo <= v <= hi


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(root, name):
    data, ref = _read_bytes(root, name)
    return (
        json.loads(
            data,
            object_pairs_hook=unique_object,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
        ),
        ref,
    )


def float32(v):
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        raise ValueError("input exceeds native float range")


def parse_input(data):
    records = []
    for line in data.decode("utf-8", errors="strict").splitlines():
        require(0 < len(line) <= 2048, "blank or oversized imported row")
        words = line.split()
        require(len(words) == 8, "native imported input needs exactly eight columns")
        row = [float(x) for x in words]
        require(all(finite(x) for x in row), "nonfinite imported input")
        require(row[0].is_integer() and 0 <= row[0] <= 2**31 - 1, "invalid injection counter")
        require(
            row[1] != 0 and 0 < sum(x * x for x in row[5:8]) <= 1 + 4e-6,
            "invalid initial energy/direction",
        )
        require(not records or row[0] >= records[-1][0], "input injection counters must be sorted")
        records.append([int(row[0])] + [float32(v) for v in row[1:]])
        require(len(records) <= MAX_RECORDS, "imported input record bound")
    require(records, "empty manufactured event input")
    return records


def identity_map(identity, source_ref, n):
    require(
        identity.get("schema_version") == IDENTITY_SCHEMA
        and identity.get("coordinate_frame") == "native_pair_tracking",
        "unsupported event identity schema/frame",
    )
    require(
        identity.get("source_artifact") == source_ref, "identity source hash/length/name mismatch"
    )
    events = identity.get("events")
    require(
        isinstance(events, list) and 0 < len(events) <= MAX_RECORDS,
        "bounded event identity list required",
    )
    byrow = {}
    event_ids = set()
    slots = {}
    template = None
    for event in events:
        eid = event.get("event_id")
        require(integer(eid) and eid not in event_ids, "duplicate/invalid explicit event ID")
        event_ids.add(eid)
        particles = event.get("particles")
        require(
            isinstance(particles, list) and 0 < len(particles) <= 16,
            "bounded event particle identities required",
        )
        present = set()
        for p in particles:
            row = p.get("source_record")
            slot = p.get("event_slot")
            pdg = p.get("pdg_id")
            role = p.get("role")
            require(
                integer(row, 1, n) and row not in byrow, "duplicate/missing source row identity"
            )
            require(
                isinstance(slot, str)
                and re.fullmatch(r"[a-z][a-z0-9_]{0,47}", slot) is not None
                and slot not in present,
                "invalid or duplicate explicit event slot",
            )
            require(
                pdg in (11, -11) and type(pdg) is int,
                "native imported tracker supports audited electron/positron slots only",
            )
            require(
                isinstance(role, str) and 0 < len(role) <= 64, "explicit particle role required"
            )
            descriptor = dict(event_slot=slot, pdg_id=pdg, label=role)
            require(
                slot not in slots or slots[slot] == descriptor,
                "event slot identity differs between events",
            )
            slots[slot] = descriptor
            present.add(slot)
            byrow[row] = dict(event_id=eid, **descriptor)
        if template is None:
            template = present
        require(present == template, "each event must declare the same explicit slots")
    require(
        set(byrow) == set(range(1, n + 1)),
        "event identity does not bind every source row exactly once",
    )
    return byrow, slots, event_ids


def angles(v):
    require(
        len(v) == 3 and all(finite(x) for x in v) and 0 < sum(x * x for x in v) <= 1 + 8e-6,
        "invalid tracked velocity",
    )
    transverse = math.hypot(v[0], v[1])
    return math.degrees(math.atan2(transverse, v[2])), (
        None if transverse == 0 else math.degrees(math.atan2(v[1], v[0]))
    )


def delta(initial, current):
    ti, pi = angles(initial)
    tc, pc = angles(current)
    return tc - ti, None if pi is None or pc is None else (pc - pi + 180) % 360 - 180


def validate_raw(raw, identity, records, source_ref, manifest=None, flags=None):
    require(
        raw.get("schema_version") == RAW_SCHEMA
        and raw.get("coordinate_frame") == "native_pair_tracking"
        and raw.get("phase") == "post-diagonal-tracker-state",
        "unsupported imported trajectory schema/coordinates/timing",
    )
    require(
        raw.get("initial_columns") == INITIAL_COLUMNS
        and raw.get("current_columns") == CURRENT_COLUMNS
        and raw.get("source_file") == "event.ini",
        "unknown native imported record columns",
    )
    require(type(raw.get("available")) is bool, "native trajectory availability missing")
    if not raw["available"]:
        require(
            isinstance(raw.get("reason"), str)
            and raw["reason"]
            and raw.get("initial_records") is None,
            "invalid unavailable trajectory record",
        )
        return None
    require(raw.get("reason") is None, "available trajectories have unavailable reason")
    f = raw.get("evaluated_flags")
    require(isinstance(f, dict), "evaluated native flags missing")
    require(
        f.get("load_events") == 1
        and f.get("track_pairs") == 1
        and all(
            f.get(k) == 0 for k in ("do_pairs", "beam_pair", "do_comp", "do_prod", "do_bhabhas")
        ),
        "mixed or unsupported native tracker population",
    )
    require(
        finite(f.get("pair_ratio"))
        and 0 <= f["pair_ratio"] <= 1
        and finite(f.get("pair_ecut"))
        and f["pair_ecut"] >= 0
        and f.get("charge_sign") in (-1, 0),
        "unsupported admission/charge convention",
    )
    if flags:
        for k, value in f.items():
            require(
                k in flags and math.isclose(value, flags[k], rel_tol=5e-6, abs_tol=5e-12),
                "native and summary evaluated flags differ: " + k,
            )
    byrow, slots, event_ids = identity_map(identity, source_ref, len(records))
    initial = raw.get("initial_records")
    require(
        isinstance(initial, list) and len(initial) <= len(records),
        "invalid admitted initial records",
    )
    bylabel = {}
    seenrows = set()
    for r in initial:
        require(
            isinstance(r, list) and len(r) == 11 and all(finite(x) for x in r),
            "invalid native initial identity record",
        )
        label, row, scheduled, admitted = r[:4]
        require(
            integer(label, 1, 2**31 - 1)
            and label not in bylabel
            and integer(row, 1, len(records))
            and row not in seenrows,
            "duplicate native label/source identity",
        )
        require(
            integer(scheduled) and integer(admitted) and admitted >= scheduled,
            "invalid native admission counters",
        )
        inp = records[row - 1]
        require(
            scheduled == inp[0] and r[4:] == inp[1:],
            "actual native initial floats differ from bound input row",
        )
        require(
            (r[4] > 0) == (byrow[row]["pdg_id"] == 11),
            "explicit PDG conflicts with audited native Lorentz-charge sign",
        )
        angles(r[8:11])
        bylabel[label] = dict(initial=r, **byrow[row])
        seenrows.add(row)
    frames = raw.get("frames")
    require(
        isinstance(frames, list) and 0 < len(frames) <= MAX_FRAMES, "native imported frame bound"
    )
    if manifest:
        require(
            manifest.get("imported_trajectories_file") == "gui_imported_trajectories.json",
            "manifest does not declare native imported export",
        )
        require(len(manifest.get("frames", [])) == len(frames), "native/GUI frame count mismatch")
        require(
            all(
                v == 0
                for k in ("native_beam_angles_x_rad", "native_beam_angles_y_rad")
                for v in manifest.get(k, [None])
            ),
            "deflection diagnostic requires head-on native beam axes",
        )
    previous = {}
    for index, frame in enumerate(frames):
        require(
            frame.get("id") == index
            and integer(frame.get("diagonal"))
            and (index == 0 or frame["diagonal"] > frames[index - 1]["diagonal"]),
            "native imported frame order invalid",
        )
        if manifest:
            require(
                frame["diagonal"] == manifest["frames"][index]["diagonal"]
                and frame["id"] == manifest["frames"][index]["id"],
                "native/GUI frame identity mismatch",
            )
        rows = frame.get("records")
        current = check_current(rows, bylabel)
        require(
            set(previous) <= set(current), "admitted tracked identity disappeared between frames"
        )
        for label, old in previous.items():
            require(
                all(a <= b for a, b in zip(old[4:], current[label][4:])),
                "cumulative native field gather counter regressed",
            )
        previous = current
    final = check_current(raw.get("final", {}).get("records"), bylabel)
    require(set(final) == set(bylabel), "final current/initial identity population mismatch")
    require(final == previous, "final native state differs from the final saved diagonal")
    return bylabel, slots, event_ids


def check_current(rows, bylabel):
    require(isinstance(rows, list) and len(rows) <= MAX_RECORDS, "bounded current records required")
    result = {}
    for row in rows:
        require(
            isinstance(row, list)
            and len(row) == 7
            and integer(row[0], 1, 2**31 - 1)
            and row[0] in bylabel
            and row[0] not in result,
            "unknown/duplicate current native identity",
        )
        angles(row[1:4])
        require(all(integer(v) for v in row[4:]), "invalid actual field-gather counts")
        result[row[0]] = row
    return result


def histogram(values, edges, identifier, excluded=0):
    counts = [0] * BINS
    under = over = 0
    for v in values:
        if v < edges[0]:
            under += 1
        elif v > edges[-1]:
            over += 1
        else:
            counts[min(BINS - 1, bisect.bisect_right(edges, v) - 1)] += 1
    n = len(values)
    stats = None
    if n:
        mean = math.fsum(values) / n
        density = [c / (b - a) for c, a, b in zip(counts, edges, edges[1:])]
        i = density.index(max(density))
        stats = dict(
            mean=mean,
            variance=math.fsum((v - mean) ** 2 for v in values) / n,
            min=min(values),
            max=max(values),
            mode=(edges[i] + edges[i + 1]) / 2,
            mode_method=MODE,
        )
    return dict(
        quantity=identifier,
        unit="deg",
        weight_unit="stored particle records",
        weighting=DEFINITIONS["weighting"],
        normalization="bin-integrated",
        bins=[dict(low=a, high=b, weight=c) for a, b, c in zip(edges, edges[1:], counts)],
        samples=n,
        underflow=under,
        overflow=over,
        total_weight=n,
        statistics=stats,
        excluded_undefined_azimuth=excluded,
    )


def make_data(raw, identity, records, source_ref, manifest=None, flags=None):
    joined = validate_raw(raw, identity, records, source_ref, manifest, flags)
    base = dict(
        schema_version=SCHEMA,
        available=joined is not None,
        reason=raw.get("reason"),
        coordinate_frame="native_pair_tracking",
        phase="post-diagonal-tracker-state",
        definitions=copy.deepcopy(DEFINITIONS),
        plots=[],
        total=None,
        frames=[],
    )
    if joined is None:
        return base
    bylabel, slots, event_ids = joined
    plots = []
    for slot, meta in slots.items():
        for observable, suffix in [("delta_theta", "Δθ"), ("delta_phi", "Δφ")]:
            plots.append(
                dict(
                    id=slot + "_" + observable,
                    event_slot=slot,
                    pdg_id=meta["pdg_id"],
                    label=meta["label"] + " " + suffix,
                    observable=observable,
                    unit="deg",
                )
            )
    scopes = [frame["records"] for frame in raw["frames"]] + [raw["final"]["records"]]
    values = []
    extents = {p["id"]: 0.0 for p in plots}
    for rows in scopes:
        vv = {p["id"]: [] for p in plots}
        excluded = {p["id"]: 0 for p in plots}
        admitted_events = {}
        for row in rows:
            label = row[0]
            item = bylabel[label]
            slot = item["event_slot"]
            admitted_events.setdefault(item["event_id"], 0)
            admitted_events[item["event_id"]] += 1
            dt, dp = delta(item["initial"][8:11], row[1:4])
            for observable, v in [("delta_theta", dt), ("delta_phi", dp)]:
                key = slot + "_" + observable
                if v is None:
                    excluded[key] += 1
                else:
                    vv[key].append(v)
                    extents[key] = max(extents[key], abs(v))
        counts = dict(
            expected=len(event_ids),
            complete=sum(n == len(slots) for n in admitted_events.values()),
            incomplete=sum(n != len(slots) for n in admitted_events.values()),
            not_yet_admitted=len(event_ids) - len(admitted_events),
        )
        gathers = {
            k: sum(row[i] for row in rows)
            for k, i in [("main_mesh", 4), ("outer_mesh", 5), ("monopole", 6)]
        }
        values.append((vv, excluded, counts, gathers, len(rows)))
    for p in plots:
        half = max(1e-9, min(180.0, extents[p["id"]] * 1.1))
        p["edges"] = [half * (2 * i / BINS - 1) for i in range(BINS + 1)]

    def scope(value):
        vv, excluded, counts, gathers, n = value
        return dict(
            available=True,
            nrecords=n,
            event_counts=counts,
            field_gathers=gathers,
            histograms={
                p["id"]: histogram(vv[p["id"]], p["edges"], p["id"], excluded[p["id"]])
                for p in plots
            },
        )

    base["plots"] = plots
    base["total"] = scope(values[-1])
    for i, (frame, value) in enumerate(zip(raw["frames"], values)):
        elapsed = manifest["frames"][i]["elapsed_s"] if manifest else None
        base["frames"].append(
            dict(id=frame["id"], diagonal=frame["diagonal"], elapsed_s=elapsed, **scope(value))
        )
    return base


def build(root, manifest=None):
    root = Path(root)
    sources = {}
    if not (root / "event_identity.json").exists():
        return dict(
            schema_version=SCHEMA,
            available=False,
            reason="No explicit imported-event identity manifest is available",
            coordinate_frame="native_pair_tracking",
            phase="post-diagonal-tracker-state",
            definitions=copy.deepcopy(DEFINITIONS),
            plots=[],
            total=None,
            frames=[],
            source_artifacts={},
        )
    identity, sources["identity"] = read_json(root, "event_identity.json")
    raw, sources["trajectories"] = read_json(root, "gui_imported_trajectories.json")
    input_data, sources["input"] = _read_bytes(root, "event.ini")
    records = parse_input(input_data)
    summary, sources["summary"] = _read_bytes(root, "out.dat")
    summary = summary.decode("utf-8", errors="strict")
    flags = evaluated_flags(summary)
    hits = re.findall(
        r"(?<![\w.])charge_sign\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=\s|;|$)",
        summary,
    )
    require(len(hits) == 1, "missing/duplicate evaluated charge_sign")
    flags["charge_sign"] = float(hits[0])
    result = make_data(raw, identity, records, sources["input"], manifest, flags)
    result["source_artifacts"] = sources
    validate(result)
    return result


def validate(data):
    require(
        isinstance(data, dict)
        and data.get("schema_version") == SCHEMA
        and data.get("definitions") == DEFINITIONS
        and data.get("coordinate_frame") == "native_pair_tracking"
        and data.get("phase") == "post-diagonal-tracker-state",
        "invalid deflection schema/definitions",
    )
    require(
        type(data.get("available")) is bool and isinstance(data.get("source_artifacts"), dict),
        "invalid deflection availability/sources",
    )
    for ref in data["source_artifacts"].values():
        require(
            isinstance(ref, dict)
            and isinstance(ref.get("path"), str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", ref["path"])
            and integer(ref.get("bytes"), 0, 32 * 1024 * 1024)
            and isinstance(ref.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "invalid bound source artifact",
        )
    if not data["available"]:
        require(
            isinstance(data.get("reason"), str)
            and data["reason"]
            and data.get("plots") == []
            and data.get("frames") == []
            and data.get("total") is None,
            "invalid unavailable deflection result",
        )
        return True
    require(
        data.get("reason") is None
        and set(data["source_artifacts"]) == {"identity", "trajectories", "input", "summary"},
        "available result must bind all sources",
    )
    plots = data.get("plots")
    require(
        isinstance(plots, list) and 0 < len(plots) <= 32 and len(plots) % 2 == 0,
        "invalid plot descriptors",
    )
    ids = set()
    slots = {}
    for p in plots:
        require(
            p.get("observable") in ("delta_theta", "delta_phi")
            and p.get("unit") == "deg"
            and p.get("pdg_id") in (-11, 11)
            and p.get("id") == p.get("event_slot", "") + "_" + p["observable"]
            and p["id"] not in ids,
            "invalid plot identity",
        )
        ids.add(p["id"])
        slots.setdefault(p["event_slot"], set()).add(p["observable"])
        edges = p.get("edges")
        require(
            isinstance(edges, list)
            and len(edges) == BINS + 1
            and all(finite(v) for v in edges)
            and all(a < b for a, b in zip(edges, edges[1:]))
            and edges[0] < 0 < edges[-1]
            and edges[0] == -edges[-1],
            "invalid fixed signed angle bins",
        )
    require(
        all(v == {"delta_theta", "delta_phi"} for v in slots.values()),
        "each slot needs both angular deflections",
    )
    frames = data.get("frames")
    require(isinstance(frames, list) and 0 < len(frames) <= MAX_FRAMES, "deflection frame bound")
    for i, f in enumerate(frames):
        require(
            f.get("id") == i
            and integer(f.get("diagonal"))
            and (i == 0 or f["diagonal"] > frames[i - 1]["diagonal"])
            and (f.get("elapsed_s") is None or finite(f["elapsed_s"]) and f["elapsed_s"] >= 0),
            "invalid deflection frame identity/time",
        )
    for scope in frames + [data.get("total")]:
        require(
            isinstance(scope, dict)
            and scope.get("available") is True
            and integer(scope.get("nrecords"), 0, MAX_RECORDS),
            "invalid deflection scope",
        )
        ec = scope.get("event_counts")
        require(
            isinstance(ec, dict)
            and set(ec) == {"expected", "complete", "incomplete", "not_yet_admitted"}
            and all(integer(v, 0, MAX_RECORDS) for v in ec.values())
            and ec["expected"] == ec["complete"] + ec["incomplete"] + ec["not_yet_admitted"],
            "event-count closure",
        )
        require(
            isinstance(scope.get("field_gathers"), dict)
            and set(scope["field_gathers"]) == {"main_mesh", "outer_mesh", "monopole"}
            and all(integer(v) for v in scope["field_gathers"].values()),
            "invalid native gather receipt",
        )
        require(
            isinstance(scope.get("histograms"), dict) and set(scope["histograms"]) == ids,
            "deflection histogram set mismatch",
        )
        slotcounts = {}
        for p in plots:
            h = scope["histograms"][p["id"]]
            bins = h.get("bins")
            n = h.get("samples")
            excluded = h.get("excluded_undefined_azimuth")
            require(
                h.get("quantity") == p["id"]
                and h.get("unit") == "deg"
                and h.get("weight_unit") == "stored particle records"
                and h.get("weighting") == DEFINITIONS["weighting"]
                and h.get("normalization") == "bin-integrated",
                "deflection histogram semantics",
            )
            require(
                isinstance(bins, list)
                and len(bins) == BINS
                and all(
                    b.get("low") == p["edges"][i]
                    and b.get("high") == p["edges"][i + 1]
                    and integer(b.get("weight"), 0, MAX_RECORDS)
                    for i, b in enumerate(bins)
                ),
                "histogram axis/count mismatch",
            )
            require(
                integer(n, 0, MAX_RECORDS)
                and integer(excluded, 0, MAX_RECORDS)
                and integer(h.get("underflow"), 0, MAX_RECORDS)
                and integer(h.get("overflow"), 0, MAX_RECORDS)
                and h.get("total_weight") == n
                and sum(b["weight"] for b in bins) + h["underflow"] + h["overflow"] == n,
                "deflection histogram closure",
            )
            require(
                p["observable"] != "delta_theta" or excluded == 0,
                "polar angle cannot have undefined azimuth exclusions",
            )
            slot = p["event_slot"]
            require(
                slot not in slotcounts or slotcounts[slot] == n + excluded,
                "slot marginal sample mismatch",
            )
            slotcounts[slot] = n + excluded
            st = h.get("statistics")
            if n == 0:
                require(st is None, "empty histogram must have null statistics")
            else:
                require(
                    isinstance(st, dict)
                    and all(finite(st.get(k)) for k in ("mean", "variance", "min", "max", "mode"))
                    and st["variance"] >= 0
                    and -180 <= st["min"] <= st["mean"] <= st["max"] <= 180
                    and st.get("mode_method") == MODE,
                    "invalid exact deflection moments",
                )
                density = [b["weight"] / (b["high"] - b["low"]) for b in bins]
                mx = max(density)
                require(
                    any(
                        d == mx
                        and math.isclose(
                            st["mode"], (b["low"] + b["high"]) / 2, rel_tol=1e-12, abs_tol=1e-30
                        )
                        for b, d in zip(bins, density)
                    ),
                    "invalid approximate deflection mode",
                )
        require(
            sum(slotcounts.values()) == scope["nrecords"], "selected plot populations do not close"
        )
    return True


def write_analysis(root, data):
    validate(data)
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    require(len(payload) <= MAX_BYTES, "deflection derived artifact exceeds 16 MiB bound")
    p = Path(root) / "collider_event_deflections.json"
    p.write_bytes(payload)
    return dict(path=p.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
