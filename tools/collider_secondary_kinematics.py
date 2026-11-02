#!/usr/bin/env python3
"""Bounded, final-only angle/pT adapter for native tracked secondary records.

This reads existing outputs. It neither regenerates particles nor estimates
luminosity, detector acceptance, missing origins, or unrecorded frame evolution.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "feebl.collider.secondary-kinematics/1"
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 1_000_000
JOINT_BINS = 128
MARGINAL_BINS = 160
ELECTRON_MASS_GEV = 0.00051099895
WEIGHTING = "one per final stored tracked secondary record; no luminosity or pair-ratio reweighting"
FLAGS = (
    "do_pairs",
    "load_events",
    "track_pairs",
    "store_pairs",
    "pair_ratio",
    "pair_ecut",
    "beam_pair",
    "do_comp",
    "do_prod",
)
INTEGER_FLAGS = set(FLAGS) - {"pair_ratio", "pair_ecut"}
INTEGER_FLAGS.add("do_bhabhas")


def _name(name):
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or name in (".", "..")
    ):
        raise ValueError("unsafe artifact name")
    return name


def _read_bytes(root, name):
    path = Path(root) / _name(name)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("missing, symlinked or oversized source: " + name)
    data = path.read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("source byte bound")
    return data, dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _read(root, name):
    data, ref = _read_bytes(root, name)
    return data.decode("utf-8", errors="strict"), ref


def evaluated_flags(text):
    result = {}
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    for key in FLAGS + ("do_bhabhas",):
        hits = re.findall(
            r"(?<![\w.])" + re.escape(key) + r"\s*=\s*" + number + r"(?=\s|;|$)", text
        )
        if key == "do_bhabhas" and not hits:
            # Old summaries omit this flag. Native preflight prohibits
            # do_pairs && do_bhabhas, so old admitted pair runs remain usable.
            result[key] = None
            continue
        if len(hits) != 1:
            raise ValueError("missing or duplicate evaluated flag: " + key)
        value = float(hits[0])
        if (
            not math.isfinite(value)
            or value < 0
            or (key in INTEGER_FLAGS and not value.is_integer())
        ):
            raise ValueError("invalid evaluated flag: " + key)
        result[key] = int(value) if key in INTEGER_FLAGS else value
    if not 0 <= result["pair_ratio"] <= 1:
        raise ValueError("pair_ratio outside [0,1]")
    if result["do_bhabhas"] and (result["do_pairs"] or result["do_comp"]):
        raise ValueError("contradictory evaluated Bhabha/pair/Compton settings")
    return result


def _native_count(text):
    hits = re.findall(r"(?<![\w.])n_pairs\s*=\s*([0-9]+)(?=\s|;|$)", text)
    if not hits or len(set(hits)) != 1:
        raise ValueError("missing or contradictory native n_pairs count")
    return int(hits[0])


def parse_records(text, flags, positions=None):
    """Parse records; optionally append (stored z, process, energy-sign) receipts."""
    theta, pt, folded = [], [], []
    processes = {}
    signs = dict(positive_energy=0, negative_energy=0)
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > 2048:
            raise ValueError("secondary row length bound")
        fields = line.split()
        bhabha = bool(flags.get("do_bhabhas"))
        if len(fields) != (11 if bhabha else 10):
            raise ValueError("unexpected native secondary column count at row " + str(number))
        if bhabha:
            index = float(fields[0])
            if not math.isfinite(index) or not index.is_integer() or not 0 <= index <= 2**31 - 1:
                raise ValueError("invalid Bhabha event index")
            fields = fields[1:]
        row = [float(v) for v in fields]
        if not all(math.isfinite(v) for v in row):
            raise ValueError("nonfinite secondary record")
        energy, vx, vy, vz = row[:4]
        # Native serialization has six significant digits. Unit beta can round
        # above one by ~2e-6; this checks format-scale plausibility, not mass-shell
        # qualification. Do not renormalize the written components.
        norm2 = vx * vx + vy * vy + vz * vz
        if energy == 0 or not 0 < norm2 <= 1 + 4e-6:
            raise ValueError("zero energy, undefined direction or invalid native velocity norm")
        if abs(energy) < ELECTRON_MASS_GEV * (1 - 2e-6):
            raise ValueError("secondary energy below electron rest mass")
        if any(not v.is_integer() or not -1 <= v <= 1_000_000 for v in row[8:10]):
            raise ValueError("invalid native source-slice index")
        process = row[7]
        if not process.is_integer() or not -999 <= process <= 999:
            raise ValueError("invalid native process index")
        if (
            flags["do_pairs"]
            and not (flags["load_events"] or flags["do_comp"] or flags["do_prod"])
            and process not in (0, 1, 2)
        ):
            raise ValueError("unexpected process in unmixed native pair output")
        transverse = math.hypot(vx, vy)
        angle = math.atan2(transverse, vz)
        momentum = abs(energy) * transverse
        if not math.isfinite(momentum):
            raise ValueError("nonfinite derived transverse momentum")
        theta.append(angle)
        pt.append(momentum)
        folded.append(math.atan2(transverse, abs(vz)))
        key = str(int(process))
        processes[key] = processes.get(key, 0) + 1
        signs["positive_energy" if energy > 0 else "negative_energy"] += 1
        if positions is not None:
            positions.append((row[6], key, "positive_energy" if energy > 0 else "negative_energy"))
        if len(theta) > MAX_RECORDS:
            raise ValueError("secondary record bound")
    return theta, pt, processes, signs, folded


def _log_edges(values, bins):
    positive = [v for v in values if v > 0]
    if not positive:
        return [10 ** (-6 + 6 * i / bins) for i in range(bins + 1)]
    low = math.floor(math.log10(min(positive)))
    high = math.ceil(math.log10(max(positive)))
    if high <= low:
        high = low + 1
    if low < -300 or high > 300:
        raise ValueError("logarithmic axis range exceeds finite display bound")
    return [10 ** (low + (high - low) * i / bins) for i in range(bins + 1)]


def _bin(edges, value):
    if value < edges[0]:
        return -1
    if value > edges[-1]:
        return len(edges) - 1
    return min(len(edges) - 2, bisect.bisect_right(edges, value) - 1)


def histogram(values, edges, quantity, unit):
    bins = [dict(low=edges[i], high=edges[i + 1], weight=0) for i in range(len(edges) - 1)]
    under, over = 0, 0
    for value in values:
        index = _bin(edges, value)
        if index < 0:
            under += 1
        elif index == len(bins):
            over += 1
        else:
            bins[index]["weight"] += 1
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / n
    # Unequal-width bins require density to define a bin-based mode.
    mode_bin = max(bins, key=lambda b: b["weight"] / (b["high"] - b["low"]))
    return dict(
        quantity=quantity,
        unit=unit,
        weight_unit="stored particle records",
        weighting=WEIGHTING,
        normalization="bin-integrated",
        bins=bins,
        samples=n,
        underflow=under,
        overflow=over,
        total_weight=n,
        statistics=dict(
            mean=mean,
            variance=variance,
            min=min(values),
            max=max(values),
            mode=(mode_bin["low"] + mode_bin["high"]) / 2,
            mode_method="centre of highest count-density histogram bin; approximate",
        ),
    )


def joint_histogram(theta, pt):
    x, y = _log_edges(theta, JOINT_BINS), _log_edges(pt, JOINT_BINS)
    counts = [0] * (JOINT_BINS * JOINT_BINS)
    excluded = dict(
        theta_nonpositive=0,
        pT_nonpositive=0,
        both_nonpositive=0,
        nonpositive_records=0,
        underflow_records=0,
        overflow_records=0,
    )
    for a, p in zip(theta, pt):
        if a <= 0:
            excluded["theta_nonpositive"] += 1
        if p <= 0:
            excluded["pT_nonpositive"] += 1
        if a <= 0 and p <= 0:
            excluded["both_nonpositive"] += 1
        if a <= 0 or p <= 0:
            excluded["nonpositive_records"] += 1
            continue
        ix, iy = _bin(x, a), _bin(y, p)
        if ix < 0 or iy < 0:
            excluded["underflow_records"] += 1
        elif ix == JOINT_BINS or iy == JOINT_BINS:
            excluded["overflow_records"] += 1
        else:
            counts[iy * JOINT_BINS + ix] += 1
    return dict(
        theta_edges_rad=x,
        pT_edges_GeV=y,
        storage_order="row-major-pT-theta",
        counts=counts,
        samples=len(theta),
        in_range_records=sum(counts),
        excluded=excluded,
    )


def _unavailable(label, reason):
    return dict(
        label=label,
        available=False,
        reason=reason,
        nrecords=0,
        joint=None,
        joint_folded=None,
        distributions=dict(theta=None, pT=None, theta_folded=None),
        selection=None,
        normalization=None,
    )


def _classification(flags):
    pairs = bool(flags["do_pairs"])
    extra = bool(flags["load_events"] or flags["do_comp"] or flags["do_prod"])
    if not pairs and flags.get("do_bhabhas") is None:
        return None, None
    if flags.get("do_bhabhas"):
        return "deflected", "Bhabha/imported particles after beam-field tracking"
    if pairs and (flags["beam_pair"] or extra):
        return (
            "combined",
            "Pairs + deflected primary/secondary particles (inseparable native output)",
        )
    if pairs:
        return "pairs", "Produced pairs after beam-field tracking"
    if extra:
        return "deflected", "Imported/scattered particles after beam-field tracking"
    return None, None


def _analyze_final_run(root, summary_name="out.dat"):
    text, summary_ref = _read(root, summary_name)
    flags = evaluated_flags(text)
    populations = {
        key: _unavailable(
            label, "This population is not separately identified in the native output"
        )
        for key, label in (
            ("pairs", "Produced pairs"),
            ("deflected", "Deflected particles"),
            ("combined", "Combined tracked population"),
        )
    }
    result = dict(
        schema_version=SCHEMA,
        phase="final",
        coordinate_frame="native_pair_tracking",
        frame_availability=False,
        evaluated_flags=flags,
        source_artifacts=dict(summary=summary_ref, pairs=None, imported_input=None),
        definitions=dict(
            theta="atan2(hypot(vx,vy),vz), relative to native +z; not a reconciled detector arm",
            theta_folded="atan2(hypot(vx,vy),abs(vz)), relative to the nearer native +z or -z axis",
            pT="abs(signed_energy_GeV)*hypot(vx,vy), with stored vx,vy=p/E; GeV/c",
            population="final stored tracked secondary particles; one record is one particle, not one pair",
            time="final tracking output only; no per-frame tracking history is exported",
            joint="logarithmic axes; nonpositive theta/pT retained in explicit exclusion counts",
            signed_energy="native charge encoding retained as sign counts; no detector-arm or species inference",
            imported_input="No imported-event input was enabled in the evaluated run settings",
            limitations="native serialization precision; no row-level physical weights or separable recoil-origin tag",
        ),
        record_count_verification=dict(
            available=False,
            expected_records=None,
            actual_records=0,
            reason="Native expected stored-record count is not yet established",
        ),
        populations=populations,
    )
    if flags["load_events"]:
        imported = Path(root) / "event.ini"
        if imported.exists() or imported.is_symlink():
            _, input_ref = _read_bytes(root, "event.ini")
            result["source_artifacts"]["imported_input"] = input_ref
            result["definitions"][
                "imported_input"
            ] = "Raw native event.ini bytes are authenticated; input payload was not parsed or physics-qualified by this adapter"
        else:
            result["definitions"][
                "imported_input"
            ] = "Native event.ini was not retained; the imported input provenance is unavailable"
            result["definitions"][
                "limitations"
            ] += "; imported input bytes are absent and cannot be independently authenticated"
    key, label = _classification(flags)
    if not flags["track_pairs"]:
        for pop in populations.values():
            pop["reason"] = "Final secondary tracking was disabled in evaluated run settings"
        return validate(result)
    if key is None:
        for pop in populations.values():
            pop["reason"] = (
                "Evaluated do_bhabhas flag is missing; non-pair output layout cannot be inferred safely"
                if flags.get("do_bhabhas") is None
                else "No admitted pair/imported/scattered population is identified by evaluated run settings"
            )
        return validate(result)
    path = Path(root) / "pairs.dat"
    if not path.exists():
        populations[key]["reason"] = "Final native pairs.dat was not exported for this run"
        return validate(result)
    records, source_ref = _read(root, "pairs.dat")
    result["source_artifacts"]["pairs"] = source_ref
    positions = []
    theta, pt, processes, signs, folded = parse_records(records, flags, positions)
    n = len(theta)
    # n_pairs counts successful stochastic creation BEFORE pair_ratio/ecut.
    # The rest-mass-limited generated pairs have no storage loss in this case.
    checkable = (
        key == "pairs" and flags["pair_ratio"] == 1 and flags["pair_ecut"] <= ELECTRON_MASS_GEV
    )
    if checkable:
        expected = _native_count(text)
        if expected != n:
            raise ValueError(
                "native n_pairs versus final stored pair count mismatch (possible truncation)"
            )
        result["record_count_verification"] = dict(
            available=True,
            expected_records=expected,
            actual_records=n,
            reason="Unmixed generated pairs; pair_ratio=1 and pair_ecut no greater than rest mass; matches native n_pairs",
        )
    else:
        result["record_count_verification"] = dict(
            available=False,
            expected_records=None,
            actual_records=n,
            reason="Mixed sources or storage cuts/thinning prevent inference of final row count from native n_pairs",
        )
    if not n:
        populations[key]["reason"] = "No final stored tracked records are present"
        return validate(result)
    populations[key] = dict(
        label=label,
        available=True,
        reason=None,
        nrecords=n,
        joint=joint_histogram(theta, pt),
        joint_folded=joint_histogram(folded, pt),
        distributions=dict(
            theta=histogram(
                theta,
                [math.pi * i / MARGINAL_BINS for i in range(MARGINAL_BINS + 1)],
                "native polar angle from +z",
                "rad",
            ),
            theta_folded=histogram(
                folded,
                [0.5 * math.pi * i / MARGINAL_BINS for i in range(MARGINAL_BINS + 1)],
                "native polar angle from nearer +z or -z axis",
                "rad",
            ),
            pT=histogram(
                pt, [0.0] + _log_edges(pt, MARGINAL_BINS - 1), "transverse momentum", "GeV/c"
            ),
        ),
        selection=dict(
            process_counts=processes,
            signed_energy_counts=signs,
            pair_ratio=flags["pair_ratio"],
            pair_ecut_GeV=flags["pair_ecut"],
            origin_separation=False,
        ),
        normalization=dict(
            kind="stored-record counts",
            weight_unit="stored particle records",
            weighting=WEIGHTING,
            physical_weight_available=False,
            completeness_verified=checkable,
        ),
    )
    from collider_secondary_positions import DEFINITION, build

    result["position_z_definition"] = dict(DEFINITION)
    populations[key]["position_z"] = build(theta, pt, folded, positions, populations[key])
    return validate(result)


def _count(value):
    return type(value) is int and 0 <= value <= MAX_RECORDS


def validate(data, manifest=None, frame_snapshots=None):
    if (
        data.get("schema_version") != SCHEMA
        or data.get("phase") != "final"
        or type(data.get("frame_availability")) is not bool
    ):
        raise ValueError("invalid secondary kinematics schema/phase")
    if data.get("coordinate_frame") != "native_pair_tracking":
        raise ValueError("invalid secondary coordinates")
    flags = data.get("evaluated_flags", {})
    if set(flags) != set(FLAGS) | {"do_bhabhas"}:
        raise ValueError("missing evaluated secondary flags")
    for key, value in flags.items():
        if key == "do_bhabhas" and value is None:
            continue
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (key in INTEGER_FLAGS and type(value) is not int)
        ):
            raise ValueError("invalid evaluated secondary flag")
    if flags["pair_ratio"] > 1 or (flags["do_bhabhas"] and (flags["do_pairs"] or flags["do_comp"])):
        raise ValueError("contradictory evaluated secondary flags")
    refs = data.get("source_artifacts", {})
    if not {"summary", "pairs"}.issubset(refs) or set(refs) - {
        "summary",
        "pairs",
        "imported_input",
        "frame_snapshots",
    }:
        raise ValueError("invalid secondary source-reference keys")
    for key in ("summary", "pairs", "imported_input", "frame_snapshots"):
        ref = refs.get(key)
        if ref is None and key != "summary":
            continue
        if not isinstance(ref, dict) or not (
            type(ref.get("bytes")) is int and 0 <= ref["bytes"] <= MAX_SOURCE_BYTES
        ):
            raise ValueError("invalid secondary source reference")
        _name(ref["path"])
        if not re.fullmatch(r"[a-f0-9]{64}", ref.get("sha256", "")):
            raise ValueError("invalid secondary source digest")
    if refs.get("imported_input") is not None and (
        not flags["load_events"] or refs["imported_input"]["path"] != "event.ini"
    ):
        raise ValueError("imported input reference disagrees with evaluated mode")
    populations = data.get("populations", {})
    if set(populations) != {"pairs", "deflected", "combined"}:
        raise ValueError("invalid secondary population keys")
    available = 0
    for population_key, pop in populations.items():
        if not isinstance(pop.get("label"), str) or not 0 < len(pop["label"]) <= 1024:
            raise ValueError("invalid secondary population label")
        if type(pop.get("available")) is not bool or not _count(pop.get("nrecords")):
            raise ValueError("invalid secondary population availability/count")
        if not pop["available"]:
            if (
                not isinstance(pop.get("reason"), str)
                or not pop["reason"]
                or pop.get("joint") is not None
                or pop.get("joint_folded") is not None
                or pop.get("nrecords") != 0
            ):
                raise ValueError("invalid unavailable secondary population")
            if (
                pop.get("distributions") != dict(theta=None, pT=None, theta_folded=None)
                or pop.get("selection") is not None
                or pop.get("normalization") is not None
            ):
                raise ValueError("unavailable secondary population contains data")
            continue
        available += 1
        n = pop["nrecords"]
        if not n or pop.get("reason") is not None:
            raise ValueError("invalid available secondary population")
        if (
            not flags["track_pairs"]
            or _classification(flags)[0] != population_key
            or refs.get("pairs") is None
        ):
            raise ValueError("population disagrees with evaluated native mode")
        for joint in (pop["joint"], pop["joint_folded"]):
            axes = [joint["theta_edges_rad"], joint["pT_edges_GeV"]]
            for edges in axes:
                if (
                    not 2 <= len(edges) <= JOINT_BINS + 1
                    or any(not math.isfinite(v) or v <= 0 for v in edges)
                    or any(a >= b for a, b in zip(edges, edges[1:]))
                ):
                    raise ValueError("invalid joint logarithmic axis")
            cells = joint["counts"]
            if (
                joint.get("storage_order") != "row-major-pT-theta"
                or len(cells) != (len(axes[0]) - 1) * (len(axes[1]) - 1)
                or not all(_count(v) for v in cells)
            ):
                raise ValueError("invalid joint storage/counts")
            ex = joint["excluded"]
            if set(ex) != {
                "theta_nonpositive",
                "pT_nonpositive",
                "both_nonpositive",
                "nonpositive_records",
                "underflow_records",
                "overflow_records",
            } or not all(_count(v) for v in ex.values()):
                raise ValueError("invalid joint exclusions")
            if (
                ex["both_nonpositive"] > min(ex["theta_nonpositive"], ex["pT_nonpositive"])
                or ex["nonpositive_records"]
                != ex["theta_nonpositive"] + ex["pT_nonpositive"] - ex["both_nonpositive"]
            ):
                raise ValueError("joint nonpositive count union does not close")
            if (
                joint.get("samples") != n
                or joint.get("in_range_records") != sum(cells)
                or sum(cells)
                + ex["nonpositive_records"]
                + ex["underflow_records"]
                + ex["overflow_records"]
                != n
            ):
                raise ValueError("joint record count does not close")
        expected_units = {
            "theta": ("native polar angle from +z", "rad"),
            "theta_folded": ("native polar angle from nearer +z or -z axis", "rad"),
            "pT": ("transverse momentum", "GeV/c"),
        }
        for key in ("theta", "pT", "theta_folded"):
            h = pop["distributions"][key]
            if (h.get("quantity"), h.get("unit")) != expected_units[key]:
                raise ValueError("invalid secondary marginal quantity/units")
            if (
                h.get("samples") != n
                or h.get("total_weight") != n
                or h.get("weight_unit") != "stored particle records"
                or h.get("weighting") != WEIGHTING
                or h.get("normalization") != "bin-integrated"
            ):
                raise ValueError("invalid secondary marginal normalization")
            if not 1 <= len(h["bins"]) <= 2048:
                raise ValueError("secondary marginal bin bound")
            last = -math.inf
            for b in h["bins"]:
                if (
                    not math.isfinite(b["low"])
                    or not math.isfinite(b["high"])
                    or b["low"] < last
                    or b["high"] <= b["low"]
                    or not _count(b["weight"])
                ):
                    raise ValueError("invalid secondary marginal bin")
                last = b["high"]
            if (
                not all(_count(h.get(k)) for k in ("underflow", "overflow"))
                or sum(b["weight"] for b in h["bins"]) + h["underflow"] + h["overflow"] != n
            ):
                raise ValueError("secondary marginal count closure")
            stats = h["statistics"]
            if (
                any(not math.isfinite(stats[k]) for k in ("mean", "variance", "min", "max", "mode"))
                or stats["variance"] < 0
                or not stats["min"] <= stats["mean"] <= stats["max"]
            ):
                raise ValueError("invalid secondary marginal statistics")
        selection = pop["selection"]
        if (
            any(not _count(v) for v in selection["process_counts"].values())
            or any(not _count(v) for v in selection["signed_energy_counts"].values())
            or sum(selection["process_counts"].values()) != n
            or sum(selection["signed_energy_counts"].values()) != n
        ):
            raise ValueError("secondary process/sign count closure")
        if (
            selection["pair_ratio"] != flags["pair_ratio"]
            or selection["pair_ecut_GeV"] != flags["pair_ecut"]
            or selection["origin_separation"] is not False
        ):
            raise ValueError("secondary selection disagrees with evaluated settings")
        norm = pop["normalization"]
        if (
            norm.get("kind") != "stored-record counts"
            or norm.get("weight_unit") != "stored particle records"
            or norm.get("weighting") != WEIGHTING
            or norm.get("physical_weight_available") is not False
        ):
            raise ValueError("invalid secondary population normalization")
        if norm.get("completeness_verified") is not data["record_count_verification"].get(
            "available"
        ):
            raise ValueError("inconsistent secondary completeness claim")
    if available > 1:
        raise ValueError("legacy records cannot be split into simultaneous origin populations")
    check = data.get("record_count_verification", {})
    if (
        type(check.get("available")) is not bool
        or not _count(check.get("actual_records"))
        or not isinstance(check.get("reason"), str)
    ):
        raise ValueError("invalid secondary record-count verification")
    if available and check["actual_records"] != sum(p["nrecords"] for p in populations.values()):
        raise ValueError("secondary verified/read count mismatch")
    if check["available"]:
        if (
            _classification(flags)[0] != "pairs"
            or flags["pair_ratio"] != 1
            or flags["pair_ecut"] > ELECTRON_MASS_GEV
            or check.get("expected_records") != check["actual_records"]
        ):
            raise ValueError("unsupported secondary completeness claim")
    elif check.get("expected_records") is not None:
        raise ValueError("unverified secondary count has an expected count")
    from collider_secondary_positions import validate_populations

    validate_populations(data)
    from collider_secondary_frames import validate_extension

    validate_extension(data, manifest, frame_snapshots)
    if len(json.dumps(data, allow_nan=False)) > MAX_OUTPUT_BYTES:
        raise ValueError("secondary output byte bound")
    return data


def analyze_run(root, summary_name="out.dat", manifest=None):
    """Final stored output and independent current-state snapshots, when exported."""
    result = _analyze_final_run(root, summary_name)
    filename = "gui_secondary_frames.json"
    path = Path(root) / filename
    declared = manifest.get("secondary_frames_file") if manifest is not None else None
    if declared is not None and declared != filename:
        raise ValueError("unknown native secondary snapshot declaration")
    if not path.exists() and not path.is_symlink():
        if declared is not None:
            raise ValueError("declared native secondary snapshots are missing")
        return result
    if manifest is None:
        # Standalone analysis can obtain display identities from a retained
        # native/finalized manifest. Finalization supplies its manifest directly.
        candidate = next(
            (
                n
                for n in ("gui_manifest.json", "gui_manifest.raw.json")
                if (Path(root) / n).is_file()
            ),
            None,
        )
        if candidate is None:
            raise ValueError("native secondary snapshots require a display manifest")
        metadata, _ = _read(root, candidate)
        manifest = json.loads(metadata)
    if manifest.get("secondary_frames_file") != filename:
        raise ValueError("native secondary snapshot file is not declared by the manifest")
    from collider_secondary_frames import MAX_BYTES, attach

    raw_text, reference = _read(root, filename)
    if reference["bytes"] > MAX_BYTES:
        raise ValueError("native secondary snapshot byte bound")
    raw = json.loads(raw_text)
    attach(result, raw, reference, manifest)
    return validate(result, manifest, raw)


def write_analysis(root, data):
    validate(data)
    path = Path(root) / "collider_secondary_kinematics.json"
    if path.is_symlink():
        raise ValueError("refuse symlinked output")
    path.write_text(
        json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--summary-name", default="out.dat")
    args = parser.parse_args()
    print(write_analysis(args.root, analyze_run(args.root, args.summary_name)))
