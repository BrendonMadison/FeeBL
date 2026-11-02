#!/usr/bin/env python3
"""Authenticate bounded native collider display data. This is not physics validation."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

MAX_BYTES = 128 * 1024 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ROWS = 4096
SCHEMA = "feebl.collider/2"


def digest(path):
    """Return the SHA-256 of a file without loading the full file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_path(root, name):
    """Resolve one regular file directly inside root; reject links and traversal."""
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or name in (".", "..")
    ):
        raise ValueError("unsafe artifact path")
    p = root / name
    if p.is_symlink() or not p.is_file() or p.resolve().parent != root.resolve():
        raise ValueError("missing, linked or escaping artifact: " + name)
    return p


def describe_frame(root, name):
    """Validate a sampled beam frame and return its hash, columns, and bounds.

    Native display columns are E [GeV], x/y/z [um], vx/vy [urad], followed
    by optional spin components. This sample has no luminosity weights.
    """
    p = safe_path(root, name)
    if not 0 < p.stat().st_size <= MAX_FRAME_BYTES:
        raise ValueError("frame size bound")
    rows = []
    columns = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = [float(x) for x in line.split()]
        if len(row) not in (6, 9) or any(not math.isfinite(x) for x in row):
            raise ValueError("invalid particle row")
        if columns is not None and columns != len(row):
            raise ValueError("inconsistent particle columns")
        columns = len(row)
        rows.append(row)
        if len(rows) > MAX_ROWS:
            raise ValueError("particle row bound")
    if not rows:
        raise ValueError("empty frame")
    return dict(
        path=name,
        sha256=digest(p),
        bytes=p.stat().st_size,
        rows=len(rows),
        columns=columns,
        bounds=[[min(r[i] for r in rows), max(r[i] for r in rows)] for i in range(6)],
    )


def verify(root, man):
    """Validate artifact integrity and frame/total consistency, raising ValueError.

    Successful validation establishes the data contract, not physical accuracy.
    """
    if man.get("schema_version") != SCHEMA:
        raise ValueError("unsupported schema")
    if man.get("units") != dict(energy="GeV", position="um", angle="urad", time="s"):
        raise ValueError("incompatible units")
    if man.get("coordinates", {}).get("frame") != "legacy_collider_dump":
        raise ValueError("unknown coordinate frame")
    if "mesh_coverage_file" in man:
        ref = man.get("mesh_coverage")
        if not isinstance(ref, dict) or ref.get("path") != man["mesh_coverage_file"]:
            raise ValueError("declared native mesh coverage must be authenticated")
        safe_path(root, man["mesh_coverage_file"])
    if "secondary_frames_file" in man:
        if (
            man["secondary_frames_file"] != "gui_secondary_frames.json"
            or "secondary_kinematics" not in man
        ):
            raise ValueError(
                "declared native secondary snapshots must be authenticated by derived analysis"
            )
        safe_path(root, man["secondary_frames_file"])
    if "imported_trajectories_file" in man:
        if (
            man["imported_trajectories_file"] != "gui_imported_trajectories.json"
            or "event_deflections" not in man
        ):
            raise ValueError("declared imported trajectories must be authenticated")
        safe_path(root, man["imported_trajectories_file"])
    frames = man.get("frames", [])
    if not 1 <= len(frames) <= 64:
        raise ValueError("frame cardinality bound")
    seen, total, prev = set(), 0, -1.0
    for i, frame in enumerate(frames):
        t = frame["elapsed_s"]
        if frame["id"] != i or not math.isfinite(t) or t <= prev:
            raise ValueError("nonmonotone time or frame ID")
        prev = t
        for key in ("beam1", "beam2"):
            art = frame[key]
            if art["path"] in seen:
                raise ValueError("duplicate artifact")
            seen.add(art["path"])
            actual = describe_frame(root, art["path"])
            if art != actual:
                raise ValueError("artifact mismatch: " + art["path"])
            total += art["bytes"]
    if total > MAX_BYTES:
        raise ValueError("dataset bound")
    for rec in man.get("receipts", {}).values():
        if isinstance(rec, dict) and "path" in rec:
            p = safe_path(root, rec["path"])
            if digest(p) != rec["sha256"]:
                raise ValueError("receipt hash mismatch")
    extras = [
        (man[k], 64 * 1024 * 1024 if k == "fields" else 32 * 1024 * 1024, False)
        for k in (
            "analysis",
            "analysis_csv",
            "diagnostics",
            "fields",
            "mesh_coverage",
            "monochromatization",
            "secondary_kinematics",
            "event_deflections",
        )
        if k in man
    ]
    extras += [(r, 32 * 1024 * 1024, False) for r in man.get("analysis_sources", [])]
    extras += [(r, 32 * 1024 * 1024, True) for r in man.get("diagnostic_analysis_sources", [])]
    authenticated = {}
    for rec, limit, allow_empty in extras:
        p = safe_path(root, rec["path"])
        if rec["path"] in seen:
            raise ValueError("analysis artifact aliases a particle frame")
        min_bytes = 0 if allow_empty else 1
        if (
            p.stat().st_size != rec["bytes"]
            or not min_bytes <= rec["bytes"] <= limit
            or digest(p) != rec["sha256"]
        ):
            raise ValueError("analysis artifact mismatch")
        if rec["path"] in authenticated:
            if authenticated[rec["path"]] != rec:
                raise ValueError("conflicting analysis source references")
            continue
        authenticated[rec["path"]] = rec
        total += rec["bytes"]
    if total > MAX_BYTES:
        raise ValueError("dataset with analysis exceeds total size bound")
    for key, module_name in (
        ("monochromatization", "collider_monochromatization"),
        ("secondary_kinematics", "collider_secondary_kinematics"),
        ("event_deflections", "collider_event_deflections"),
    ):
        if key in man:
            import importlib

            module = importlib.import_module(module_name)
            data = json.loads(safe_path(root, man[key]["path"]).read_text())
            if key == "event_deflections":
                module.validate(data)
                declared = man.get("imported_trajectories_file")
                raw_ref = authenticated.get(declared) if declared else None
                if declared and raw_ref is None:
                    raise ValueError(
                        "native imported trajectory source is not bound by the manifest"
                    )
                for ref in data.get("source_artifacts", {}).values():
                    if ref is not None and authenticated.get(ref["path"]) != ref:
                        raise ValueError("imported event source is not authenticated")
                # Rebuild from authenticated initial identities and native states;
                # validation must not merely trust supplied angular histograms.
                if module.build(root, man) != data:
                    raise ValueError("event deflection analysis does not match native trajectories")
                if data.get("available") and (
                    raw_ref is None or data["source_artifacts"]["trajectories"] != raw_ref
                ):
                    raise ValueError(
                        "available deflections must bind the declared native trajectory source"
                    )
            elif key == "secondary_kinematics":
                # Validate source roles and bounds before reading the referenced
                # raw snapshot payload. Authentication precedes interpretation.
                module.validate(data, manifest=man)
                snapshot_ref = data.get("source_artifacts", {}).get("frame_snapshots")
                declared = man.get("secondary_frames_file")
                if declared is not None and (
                    snapshot_ref is None or snapshot_ref.get("path") != declared
                ):
                    raise ValueError("declared native secondary snapshot source is not bound")
                if (
                    snapshot_ref is not None
                    and authenticated.get(snapshot_ref["path"]) != snapshot_ref
                ):
                    raise ValueError(
                        "native secondary snapshot source is not bound by the manifest"
                    )
                raw = (
                    json.loads(safe_path(root, snapshot_ref["path"]).read_text())
                    if snapshot_ref is not None
                    else None
                )
                module.validate(data, manifest=man, frame_snapshots=raw)
            else:
                context = {
                    role: (
                        json.loads(safe_path(root, man[role]["path"]).read_text())
                        if role in man
                        else None
                    )
                    for role in ("analysis", "diagnostics")
                }
                module.validate(data, manifest=man, **context)
            if key == "monochromatization":
                if any(
                    data["native_manifest_parameters"].get(k) != man.get(k) for k in module.KEYS
                ):
                    raise ValueError("monochromatization parameters do not match native manifest")
                for role in ("analysis", "diagnostics"):
                    if role in man and data["source_artifacts"].get(role) != man[role]:
                        raise ValueError("monochromatization source role mismatch: " + role)
            bound_sources = {r["path"]: r for r in man.get("diagnostic_analysis_sources", [])}
            for ref in data.get("source_artifacts", {}).values():
                if ref is not None and bound_sources.get(ref["path"]) != ref:
                    raise ValueError("diagnostic analysis source is not bound by the manifest")
    if "diagnostics" in man:
        from collider_diagnostics import validate

        validate(json.loads(safe_path(root, man["diagnostics"]["path"]).read_text()), frames)
    if "fields" in man:
        from collider_fields import validate as validate_fields

        validate_fields(json.loads(safe_path(root, man["fields"]["path"]).read_text()), frames)
    if "mesh_coverage" in man:
        coverage = json.loads(safe_path(root, man["mesh_coverage"]["path"]).read_text())
        if (
            coverage.get("schema_version") != "feebl.collider.mesh-coverage/1"
            or type(coverage.get("available")) is not bool
        ):
            raise ValueError("unsupported mesh coverage contract")
        if coverage["available"]:
            saved = coverage.get("snapshot_frames", [])
            if len(saved) != len(frames) or any(
                s["id"] != f["id"] or s["diagonal"] != f["diagonal"] for s, f in zip(saved, frames)
            ):
                raise ValueError("mesh coverage snapshot identity mismatch")
            if len(coverage.get("encounters", [])) != coverage.get("summary", {}).get("encounters"):
                raise ValueError("mesh coverage encounter cardinality mismatch")
    return dict(
        status="PASS_FOR_STATED_TARGET",
        scope="dataset bytes, finite values, IDs, times, units, optional native frame-total closure and field-map contract only",
        frames=len(frames),
        bytes=total,
        physics_validation="NOT_ESTABLISHED",
    )


def bind_diagnostic_analysis(root, man, summary_name="out.dat"):
    """Add derived diagnostics without changing the recorded simulation provenance."""
    import importlib

    sources = {}
    adapters = {}
    for key, module_name in (
        ("monochromatization", "collider_monochromatization"),
        ("secondary_kinematics", "collider_secondary_kinematics"),
        ("event_deflections", "collider_event_deflections"),
    ):
        module = importlib.import_module(module_name)
        data = (
            module.build(root, man)
            if key == "event_deflections"
            else (
                module.analyze_run(root, man, summary_name=summary_name)
                if key == "monochromatization"
                else module.analyze_run(root, summary_name=summary_name, manifest=man)
            )
        )
        module.validate(data)
        module.write_analysis(root, data)
        path = safe_path(root, module_name + ".json")
        man[key] = dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size)
        for ref in data.get("source_artifacts", {}).values():
            if ref is None:
                continue
            if ref["path"] in sources and sources[ref["path"]] != ref:
                raise ValueError("inconsistent diagnostic source reference")
            sources[ref["path"]] = ref
        adapters[module_name] = digest(Path(module.__file__))
        if key == "secondary_kinematics" and data.get("source_artifacts", {}).get(
            "frame_snapshots"
        ):
            frames_module = importlib.import_module("collider_secondary_frames")
            adapters["collider_secondary_frames"] = digest(Path(frames_module.__file__))
        if key == "secondary_kinematics" and data.get("position_z_definition"):
            positions_module = importlib.import_module("collider_secondary_positions")
            adapters["collider_secondary_positions"] = digest(Path(positions_module.__file__))
        if key == "monochromatization" and "peak_fit" in data:
            for helper in ("collider_peak_fit", "collider_peak_diagnostics"):
                adapters[helper] = digest(Path(importlib.import_module(helper).__file__))
        if (
            key == "monochromatization"
            and data.get("peak_fit", {}).get("schema_version")
            == "feebl.collider.beta-gaussian-peak-fit/1"
        ):
            adapters["collider_beta_peak_fit"] = digest(
                Path(importlib.import_module("collider_beta_peak_fit").__file__)
            )
    declared = man.get("imported_trajectories_file")
    if declared is not None:
        p = safe_path(root, declared)
        raw_ref = dict(path=declared, bytes=p.stat().st_size, sha256=digest(p))
        if declared in sources and sources[declared] != raw_ref:
            raise ValueError("conflicting imported trajectory source")
        sources[declared] = raw_ref
    man["diagnostic_analysis_sources"] = list(sources.values())
    man["diagnostic_analysis_provenance"] = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        adapter_sha256=adapters,
        scope="Derived analysis of authenticated native outputs; original simulation source and binary preserved",
    )
    return man


def augment_diagnostic_analysis(root, summary_name="out.dat"):
    """Re-analyse a copied finalized dataset; native sources and snapshots are untouched."""
    path = safe_path(root, "gui_manifest.json")
    man = json.loads(path.read_text())
    verify(root, man)
    bind_diagnostic_analysis(root, man, summary_name)
    result = verify(root, man)
    path.write_text(json.dumps(man, indent=2, allow_nan=False) + "\n")
    return result


def bind_completion_receipt(root, man):
    """Bind the final backend state; initialization counters are not completion."""
    log = safe_path(root, "log.txt")
    if log.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("runtime log size bound")
    lines = [
        line.strip()
        for line in log.read_text(encoding="utf-8").splitlines()
        if "FEEBL: final compute_backend requested=" in line
    ]
    if len(lines) != 1:
        raise ValueError("exactly one final runtime backend receipt required")
    receipt = lines[0]
    match = re.search(r"compute_backend requested=(\w+) actual=(\w+)", receipt)
    if not match or match[1] != match[2]:
        raise ValueError("backend mismatch or fallback in completion receipt")
    if match[1] == "gpu" and (
        'gpu_device_description="not-initialized"' in receipt
        or not re.search(r"gpu_calls=([1-9]\d*)", receipt)
    ):
        raise ValueError("GPU completion lacks actual device/calls")
    man["backend_initialization_receipt"] = man["backend_actual_receipt"]
    man["backend_actual_receipt"] = receipt
    man["receipts"]["runtime_completion_log"] = dict(path="log.txt", sha256=digest(log))


def source_identity(source):
    """Return source metadata and a patch for a checkout or exported source tree.

    Git discovery must identify this exact directory; an unrelated parent
    repository does not identify the simulation source. A source export uses
    SOURCE_REVISION and hashes its build, kernel, and analysis files. The
    resulting hashes describe the files supplied at finalization, not proof
    that a particular binary was built from them; its hash is recorded too.
    """
    source = Path(source).resolve(strict=True)
    try:
        git_root = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        git_root = None
    if (
        git_root is not None
        and git_root.returncode == 0
        and Path(git_root.stdout.strip()).resolve() == source
    ):
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
        untracked = subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
        hashes = {
            name: digest(source / name)
            for name in untracked
            if (source / name).is_file() and not (source / name).is_symlink()
        }
        return (
            dict(
                commit=commit,
                dirty=bool(patch or untracked),
                untracked_sha256=hashes,
                identity_method="git",
            ),
            patch,
        )

    revision_file = source / "SOURCE_REVISION"
    if revision_file.is_symlink() or not revision_file.is_file():
        raise ValueError("source export requires a regular SOURCE_REVISION file")
    revision = revision_file.read_text(encoding="utf-8").strip()
    git_revision = re.fullmatch(r"[0-9a-fA-F]{40}(?:-dirty)?", revision)
    export_revision = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", revision)
    if not git_revision and not export_revision:
        raise ValueError("SOURCE_REVISION requires a Git revision or sha256:<64 hex> export identity")
    files = [p for p in source.iterdir() if p.is_file() and p.suffix != ".pdf"]
    for directory in ("src", "cmake", "tools", "examples"):
        files.extend(
            p
            for p in (source / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    if any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in files):
        raise ValueError("source export contains linked or escaping files")
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in sorted(files)}
    return (
        dict(
            commit=revision.removesuffix("-dirty") if git_revision else None,
            dirty=revision.endswith("-dirty"),
            source_revision=revision,
            source_export_sha256=export_revision.group(1).lower() if export_revision else None,
            untracked_sha256={},
            identity_method="source_export",
            source_files_sha256=hashes,
        ),
        b"",
    )


def resolve_collision_normalization(raw, summary, angle, frequency):
    """Resolve the horizontal angle [rad] and crossings/s from evaluated output.

    Explicit CLI overrides take precedence and are recorded. The native beam
    convention uses the sum of horizontal angles; nonhorizontal geometries need
    an explicit convention instead of an inferred scalar magnitude.
    """
    provenance = {}
    if angle is None:
        ax, ay = raw.get("native_beam_angles_x_rad"), raw.get("native_beam_angles_y_rad")
        valid = lambda v: isinstance(v, list) and len(v) == 2 and all(
            isinstance(x, (float, int)) and math.isfinite(x) for x in v
        )
        if valid(ax) and valid(ay) and all(x == 0 for x in ay):
            angle = abs(math.fsum(ax))
            provenance["full_crossing_angle_rad"] = {
                "source": "evaluated native horizontal beam angles",
                "rule": "abs(angle_x.1 + angle_x.2); native opposite-zeta convention",
                "inputs": ax,
            }
    else:
        provenance["full_crossing_angle_rad"] = {"source": "explicit CLI override"}
    if frequency is None:
        number = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
        values = [re.findall(r"\b" + key + r"\s*=\s*" + number, summary)
                  for key in ("f_rep", "n_b")]
        if all(len(v) == 1 for v in values):
            rate, bunches = (float(v[0]) for v in values)
            frequency = rate * bunches
            provenance["crossing_frequency_hz"] = {
                "source": "evaluated native summary",
                "rule": "f_rep * n_b per interaction point",
                "f_rep_Hz": rate, "n_b": bunches,
            }
    else:
        provenance["crossing_frequency_hz"] = {"source": "explicit CLI override"}
    for key, value in (("full_crossing_angle_rad", angle), ("crossing_frequency_hz", frequency)):
        if value is not None:
            if not math.isfinite(value) or value < 0 or (key == "crossing_frequency_hz" and value == 0):
                raise ValueError("invalid " + key)
            provenance[key]["value"] = value
    return angle, frequency, provenance


def finalize(
    root,
    binary,
    card,
    source,
    full_crossing_angle_rad=None,
    crossing_frequency_hz=None,
    summary_name="out.dat",
    normalization_scope="per_ip",
):
    """Bind native outputs to their input card, executable and source identity.

    The full angle is in radians and the crossing frequency in Hz. Diagnostic
    sidecars are written inside root; raw native outputs remain unchanged.
    Malformed, incomplete or inconsistent datasets raise an exception.
    """
    raw_path = safe_path(root, "gui_manifest.raw.json")
    if raw_path.stat().st_size > 1024 * 1024:
        raise ValueError("raw manifest bound")
    raw = json.loads(raw_path.read_text())
    if raw.pop("schema_version") != "feebl.collider.raw/1":
        raise ValueError("unsupported producer")
    for name, origin in (("input.card", card),):
        dest = root / name
        if dest.exists() and digest(dest) != digest(origin):
            raise ValueError("refuse to replace existing receipt")
        if not dest.exists():
            shutil.copyfile(origin, dest)
    summary_path = safe_path(root, summary_name) if (root / summary_name).exists() else None
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path else ""
    full_crossing_angle_rad, crossing_frequency_hz, normalization_inputs = (
        resolve_collision_normalization(raw, summary_text, full_crossing_angle_rad, crossing_frequency_hz)
    )
    identity, patch = source_identity(source)
    patch_path = root / "source.patch"
    patch_path.write_bytes(patch)
    man = dict(
        raw,
        schema_version=SCHEMA,
        run_id=root.name,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        acc_file="input.card",
        units=dict(energy="GeV", position="um", angle="urad", time="s"),
        coordinates=dict(
            frame="legacy_collider_dump",
            origin="native dump reconstruction",
            time_origin="first committed diagonal",
            beam_z_dump_sign=[1, -1],
            beam_centroid_motion_sign=[-1, 1],
            qualification="Native legacy dump convention; canonical lab-axis reconciliation is unresolved.",
        ),
        coverage=dict(
            pre_collision_initial=False,
            first_frame_phase="after_diagonal_0",
            final_committed_diagonal=True,
            limitation="No independently reconciled pre-collision initial dump; sampled branch uses stride >= 2.",
        ),
        normalization_inputs=normalization_inputs,
        normalization=dict(
            kind="display_sample",
            weights="not exported; never use for luminosity",
            luminosity="unavailable in this snapshot contract",
        ),
        source=identity,
        receipts=dict(
            binary=dict(sha256=digest(binary)),
            card=dict(path="input.card", sha256=digest(card)),
            source_patch=dict(path="source.patch", sha256=digest(patch_path)),
        ),
        validation=dict(
            status="UNVALIDATED",
            scope="collider snapshots only",
            uncertainties=None,
            unresolved=[
                "1% physics equivalence not established",
                "lab-axis reconciliation",
                "display thinning omits particles and cannot estimate tails or weighted observables",
            ],
        ),
        availability=dict(
            ee_spectrum="UNAVAILABLE: authenticated estimator adapter pending",
            pair_positive_arm="UNAVAILABLE: produced-pair adapter pending",
            pair_negative_arm="UNAVAILABLE: produced-pair adapter pending",
            external_events="UNAVAILABLE: event sidecar adapter pending",
        ),
    )
    bind_completion_receipt(root, man)
    spectrum = root / "lumi.ee.out.dLdW.fne"
    if spectrum.exists() and full_crossing_angle_rad is not None:
        from collider_analysis import analyze_run

        analysis = analyze_run(
            root,
            raw["energy_GeV"],
            full_crossing_angle_rad,
            crossing_frequency_hz,
            summary_name=summary_name,
            normalization_scope=normalization_scope,
        )

        def artifact(name):
            p = safe_path(root, name)
            return dict(path=name, sha256=digest(p), bytes=p.stat().st_size)

        man["analysis"] = artifact("collider_analysis.json")
        man["analysis_csv"] = artifact("ee_spectrum.csv")
        man["analysis_sources"] = list(analysis["sources"].values())
        man["availability"][
            "ee_spectrum"
        ] = "AVAILABLE_NATIVE_UNVALIDATED: full/peak, original bin weights and flow weights"
        man["normalization"]["luminosity"] = analysis["normalization"]
    elif spectrum.exists():
        raise ValueError("Cannot derive a horizontal crossing angle; supply --full-crossing-angle-rad with its convention")
    man["frames"] = [
        dict(f, beam1=describe_frame(root, f["beam1"]), beam2=describe_frame(root, f["beam2"]))
        for f in raw["frames"]
    ]
    diagnostics = root / "gui_diagnostics.json"
    if diagnostics.exists():
        from collider_diagnostics import validate

        data = json.loads(safe_path(root, diagnostics.name).read_text())
        validate(data, man["frames"])
        man["diagnostics"] = dict(
            path=diagnostics.name, sha256=digest(diagnostics), bytes=diagnostics.stat().st_size
        )
        man["availability"][
            "frame_diagnostics"
        ] = "AVAILABLE_NATIVE_UNVALIDATED: native frame intervals and integrated observables"
    fields = root / "gui_fields.json"
    if fields.exists():
        from collider_fields import validate as validate_fields

        if not 0 < fields.stat().st_size <= 64 * 1024 * 1024:
            raise ValueError("field artifact size bound")
        data = json.loads(safe_path(root, fields.name).read_text())
        field_status = validate_fields(data, man["frames"])
        man["fields"] = dict(path=fields.name, sha256=digest(fields), bytes=fields.stat().st_size)
        field_views = []
        if field_status["available_frames"]:
            field_views.append("central source-slice x-y")
        if (field_status.get("longitudinal") or {}).get("available_frames"):
            field_views.append("source-specific z-x/z-y cross-sections; gaps transparent")
        if (field_status.get("snapshot") or {}).get("available_frames"):
            field_views.append(
                "full primary-source snapshots synchronized with the particle display; z-x/z-y"
            )
        man["availability"]["field_maps"] = (
            "AVAILABLE_NATIVE_UNVALIDATED: CIC electric-equivalent field maps; "
            + "; ".join(field_views)
            if field_views
            else "UNAVAILABLE: see native field sidecar reasons"
        )
    coverage_name = raw.get("mesh_coverage_file")
    if coverage_name is not None:
        coverage = safe_path(root, coverage_name)
        if not 0 < coverage.stat().st_size <= 32 * 1024 * 1024:
            raise ValueError("mesh coverage artifact size bound")
        man["mesh_coverage"] = dict(
            path=coverage.name, sha256=digest(coverage), bytes=coverage.stat().st_size
        )
        man["availability"][
            "mesh_coverage"
        ] = "AUTHENTICATED_NATIVE_RECORDS: per-encounter primary deposition and gather coverage, plus saved snapshot source coverage; no physics qualification"
    if (root / summary_name).is_file():
        bind_diagnostic_analysis(root, man, summary_name)
    result = verify(root, man)
    path = root / "gui_manifest.json"
    path.write_text(json.dumps(man, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--card", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--augment-diagnostics",
        action="store_true",
        help="Add derived diagnostics to an existing finalized dataset; preserve simulation provenance",
    )
    p.add_argument("--full-crossing-angle-rad", type=float, help="Override evaluated horizontal angle [rad]")
    p.add_argument("--crossing-frequency-hz", type=float, help="Override evaluated f_rep * n_b [crossings/s/IP]")
    p.add_argument("--summary", default="out.dat")
    p.add_argument(
        "--normalization-scope", choices=("per_ip", "machine_total_unallocated"), default="per_ip"
    )
    a = p.parse_args()
    if a.verify and a.augment_diagnostics:
        p.error("--verify and --augment-diagnostics are mutually exclusive")
    if a.augment_diagnostics:
        result = augment_diagnostic_analysis(a.run, a.summary)
    elif a.verify:
        result = verify(a.run, json.loads((a.run / "gui_manifest.json").read_text()))
    else:
        if not all((a.binary, a.card, a.source)):
            p.error("--binary, --card and --source are required")
        result = finalize(
            a.run,
            a.binary,
            a.card,
            a.source,
            a.full_crossing_angle_rad,
            a.crossing_frequency_hz,
            a.summary,
            a.normalization_scope,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
