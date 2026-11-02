#!/usr/bin/env python3
"""Run the P1/P2 fail-closed, reproducibility, physics, and timing matrix.

The runner is selector-agnostic by design.  A JSON manifest names complete
baseline/candidate cards, required provenance receipts, unsupported-selector
cards (expressed as final assignments appended to the selected PARAMETERS
block), and stage timing patterns.  This lets a new GPU sparse-field mode and
a same-wave concurrency mode evolve without weakening the qualification gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

from lumi_spectrum_equivalence import Thresholds, compare_ensemble, read_spectrum


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
REQUIRED_ARTIFACTS = (
    "result.out",
    "lumi.ee.out",
    "lumi.ee.out.fne",
    "lumi.ee.out.dLdW.fne",
)
REJECTION_SCIENCE_ARTIFACTS = (
    "lumi.ee.out",
    "lumi.ee.out.fne",
    "lumi.ee.out.dLdW.fne",
    "beam1.dat",
    "beam2.dat",
)
DEFAULT_NORMALIZE = (
    r"^collision_schedule_requested",
    r"^sparse_cells =",
    r"^wavefront_field_solver =",
    r"compute_backend_requested",
    r"cpu_variant_requested",
    r"gpu_variant_requested",
    r"gpu_mode_requested",
    r"gpu2_",
    r"backend_fallback",
    r"backend_provenance",
)
DEFAULT_FORBIDDEN_SUCCESS = (
    r"fallback_used\s*=\s*1",
    r"actual\s*=\s*serial_fallback",
    r"actual\s*=\s*cpu_fallback",
)


class QualificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def inject_parameters(source: str, parameters: str, seed: int,
                      fragment: str = "") -> str:
    if re.search(r"\brndm_(?:seed|load|save)\s*=", fragment, re.IGNORECASE):
        raise ValueError("selector fragment may not override qualification RNG controls")
    pattern = re.compile(
        rf"(?ms)(^\$PARAMETERS::[ \t]*{re.escape(parameters)}[ \t]*\r?\n[ \t]*\{{)"
        rf"(.*?)(\}})"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one $PARAMETERS:: {parameters} block; found {len(matches)}"
        )
    addition = fragment.strip()
    if addition and not addition.endswith(";"):
        addition += ";"
    addition += f"rndm_seed={seed};rndm_load=0;rndm_save=0;"
    match = matches[0]
    replacement = match.group(1) + match.group(2) + addition + match.group(3)
    return source[:match.start()] + replacement + source[match.end():]


def parse_named_block(source: str, kind: str, name: str) -> dict[str, str]:
    pattern = re.compile(
        rf"(?ms)^\${re.escape(kind)}::[ \t]*{re.escape(name)}[ \t]*\r?\n"
        rf"[ \t]*\{{(.*?)\}}"
    )
    matches = pattern.findall(source)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ${kind}:: {name} block; found {len(matches)}")
    assignments: dict[str, str] = {}
    for raw in matches[0].split(";"):
        statement = raw.strip()
        if not statement:
            continue
        key, separator, value = statement.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key):
            raise ValueError(f"cannot parse assignment in ${kind}:: {name}: {statement!r}")
        assignments[key] = re.sub(r"\s+", "", value)
    return assignments


def compare_card_physics(baseline: str, candidate: str, accelerator: str,
                         parameters: str, ignored_keys: Iterable[str]) -> dict[str, Any]:
    ignored = {"rndm_seed", "rndm_load", "rndm_save", *ignored_keys}
    changed: list[str] = []
    canonical: dict[str, dict[str, dict[str, str]]] = {}
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        canonical[label] = {
            "accelerator": parse_named_block(source, "ACCELERATOR", accelerator),
            "parameters": parse_named_block(source, "PARAMETERS", parameters),
        }
        for block in canonical[label].values():
            for key in ignored:
                block.pop(key, None)
    for block_name in ("accelerator", "parameters"):
        left = canonical["baseline"][block_name]
        right = canonical["candidate"][block_name]
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                changed.append(f"{block_name}.{key}")
    rendered = {
        label: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for label, value in canonical.items()
    }
    return {
        "equivalent": not changed,
        "ignored_assignment_keys": sorted(ignored),
        "canonical_sha256": rendered,
        "unexpected_differences": changed,
    }


def require_regexes(text: str, patterns: Iterable[str], context: str) -> None:
    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE) is None:
            raise QualificationError(f"{context}: missing required receipt /{pattern}/")


def forbid_regexes(text: str, patterns: Iterable[str], context: str) -> None:
    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE) is not None:
            raise QualificationError(f"{context}: forbidden fallback /{pattern}/ was emitted")


def parse_stages(text: str, patterns: dict[str, str], context: str) -> dict[str, float]:
    stages: dict[str, float] = {}
    for name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if not matches:
            raise QualificationError(f"{context}: missing stage timing {name} /{pattern}/")
        match = matches[-1]
        if match.lastindex != 1:
            raise QualificationError(
                f"{context}: stage regex {name!r} must contain exactly one capture group"
            )
        value = float(match.group(1))
        if not math.isfinite(value) or value < 0.0:
            raise QualificationError(f"{context}: invalid stage timing {name}={value}")
        stages[name] = value
    return stages


def normalize_result(source: Path, destination: Path,
                     patterns: Sequence[str]) -> None:
    compiled = [re.compile(pattern) for pattern in patterns]
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines(True)
    destination.write_text(
        "".join(line for line in lines if not any(item.search(line) for item in compiled)),
        encoding="utf-8",
    )


def compare_numeric_text(reference: Path, candidate: Path,
                         rtol: float, atol: float) -> dict[str, Any]:
    number = re.compile(r"(?<![A-Za-z_])" + NUMBER)
    left = reference.read_text(encoding="utf-8").splitlines(keepends=True)
    right = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(left) != len(right):
        return {"pass": False, "reason": "line-count mismatch",
                "reference_lines": len(left), "candidate_lines": len(right)}
    compared = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failures: list[dict[str, Any]] = []
    for line_number, (left_line, right_line) in enumerate(zip(left, right), 1):
        left_values = [float(item) for item in number.findall(left_line)]
        right_values = [float(item) for item in number.findall(right_line)]
        if (number.sub("<number>", left_line) != number.sub("<number>", right_line)
                or len(left_values) != len(right_values)):
            return {"pass": False, "reason": "non-numeric structure mismatch",
                    "line": line_number}
        for value_index, (expected, actual) in enumerate(zip(left_values, right_values), 1):
            absolute = abs(expected - actual)
            scale = max(abs(expected), abs(actual))
            relative = absolute / scale if scale else 0.0
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            compared += 1
            if not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
                if len(failures) < 10:
                    failures.append({
                        "line": line_number, "value": value_index,
                        "reference": expected, "candidate": actual,
                        "absolute": absolute, "relative": relative,
                    })
    return {
        "pass": not failures,
        "numeric_fields": compared,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "first_failures": failures,
        "rtol": rtol,
        "atol": atol,
    }


def run_one(binary: Path, command_prefix: Sequence[str], accelerator: str,
            parameters: str, source_card: str, seed: int, run_dir: Path,
            label: str, variant: dict[str, Any], timeout: float,
            stage_patterns: dict[str, str], fragment: str = "",
            extra_environment: dict[str, str] | None = None,
            require_success: bool = True) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    card = run_dir / "acc.dat"
    card.write_text(
        inject_parameters(source_card, parameters, seed, fragment), encoding="utf-8"
    )
    command = [*command_prefix, str(binary), accelerator, parameters, "result.out"]
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "OMP_DYNAMIC": "FALSE"})
    if extra_environment:
        environment.update({str(key): str(value)
                            for key, value in extra_environment.items()})
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command, cwd=run_dir, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        (run_dir / "timeout.txt").write_text(str(error) + "\n", encoding="utf-8")
        raise QualificationError(f"{label} exceeded {timeout:g} seconds") from error
    stopped = time.perf_counter_ns()
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result_text = ""
    if (run_dir / "result.out").is_file():
        result_text = (run_dir / "result.out").read_text(
            encoding="utf-8", errors="replace"
        )
    evidence_text = completed.stdout + "\n" + completed.stderr + "\n" + result_text
    receipt = {
        "label": label,
        "seed": seed,
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": (stopped - started) / 1.0e9,
        "clock": "python.time.perf_counter_ns",
        "clock_monotonic": True,
        "user_cpu_seconds": after_usage.ru_utime - before_usage.ru_utime,
        "system_cpu_seconds": after_usage.ru_stime - before_usage.ru_stime,
        "max_rss_kib_after_child": after_usage.ru_maxrss,
        "card_sha256": sha256(card),
    }
    if require_success:
        if completed.returncode != 0:
            raise QualificationError(
                f"{label} failed with rc={completed.returncode}; see {run_dir / 'stderr.txt'}"
            )
        for name in variant.get("required_artifacts", REQUIRED_ARTIFACTS):
            artifact = run_dir / name
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise QualificationError(f"{label}: missing nonempty artifact {name}")
        require_regexes(
            evidence_text,
            [*variant.get("receipt_regex", ()), *variant.get("success_only_regex", ())],
            label,
        )
        forbid_regexes(
            evidence_text,
            [*DEFAULT_FORBIDDEN_SUCCESS, *variant.get("forbidden_regex", ())],
            label,
        )
        receipt["stages"] = parse_stages(evidence_text, stage_patterns, label)
    receipt["artifact_sha256"] = {
        name: sha256(path)
        for name in variant.get("required_artifacts", REQUIRED_ARTIFACTS)
        if (path := run_dir / name).is_file()
    }
    (run_dir / "run-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"receipt": receipt, "evidence_text": evidence_text, "directory": run_dir}


def validate_manifest(manifest: dict[str, Any], smoke: bool) -> None:
    if manifest.get("schema") != "feebl.priority12.qualification.v1":
        raise ValueError("manifest schema must be feebl.priority12.qualification.v1")
    for name in ("accelerator", "parameters", "baseline", "candidate"):
        if name not in manifest:
            raise ValueError(f"manifest is missing {name}")
    for name in ("accelerator", "parameters"):
        if not isinstance(manifest[name], str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", manifest[name]
        ):
            raise ValueError(f"manifest {name} is not a safe block name")
    for label in ("baseline", "candidate"):
        variant = manifest[label]
        if not isinstance(variant.get("card"), str):
            raise ValueError(f"{label}.card is required")
    selector_keys = manifest.get("selector_assignment_keys", ())
    if any(not isinstance(key, str)
           or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key)
           for key in selector_keys):
        raise ValueError("selector_assignment_keys contains an invalid assignment name")
    if not manifest["candidate"].get("receipt_regex"):
        raise ValueError(
            "candidate.receipt_regex must prove requested/actual selector and concurrency"
        )
    if not manifest["candidate"].get("success_only_regex"):
        raise ValueError(
            "candidate.success_only_regex must identify a completed/committed fast path"
        )
    if not manifest.get("rejections"):
        raise ValueError("at least one fail-closed rejection case is required")
    names = [item.get("name") for item in manifest["rejections"]]
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
           for name in names) or len(names) != len(set(names)):
        raise ValueError("rejection names must be unique safe path components")
    for case in manifest["rejections"]:
        if not case.get("stderr_regex"):
            raise ValueError(f"rejection {case['name']} needs stderr_regex")
        if case.get("base", "candidate") not in ("baseline", "candidate"):
            raise ValueError(f"rejection {case['name']} has invalid base")
    regex_fields = [
        *manifest["baseline"].get("receipt_regex", ()),
        *manifest["baseline"].get("forbidden_regex", ()),
        *manifest["candidate"].get("receipt_regex", ()),
        *manifest["candidate"].get("success_only_regex", ()),
        *manifest["candidate"].get("forbidden_regex", ()),
        *(case["stderr_regex"] for case in manifest["rejections"]),
        *(pattern for case in manifest["rejections"]
          for pattern in case.get("forbidden_regex", ())),
    ]
    try:
        for pattern in regex_fields:
            re.compile(pattern)
    except (TypeError, re.error) as error:
        raise ValueError(f"manifest contains an invalid regular expression: {error}") from error
    stage_regex = manifest.get("stage_regex", {})
    for label in ("baseline", "candidate"):
        patterns = stage_regex.get(label, {})
        if not patterns:
            raise ValueError(f"stage_regex.{label} must contain at least one stage timer")
        for name, pattern in patterns.items():
            try:
                compiled = re.compile(pattern)
            except (TypeError, re.error) as error:
                raise ValueError(f"invalid {label} stage regex {name}: {error}") from error
            if compiled.groups != 1:
                raise ValueError(f"stage regex {label}.{name} must have one capture group")
    for label, names_for_closure in manifest.get("nonoverlap_stage_names", {}).items():
        if label not in ("baseline", "candidate"):
            raise ValueError(f"nonoverlap_stage_names has unknown variant {label}")
        unknown = set(names_for_closure) - set(stage_regex[label])
        if unknown:
            raise ValueError(
                f"nonoverlap_stage_names.{label} names unknown stages: {sorted(unknown)}"
            )
    deterministic = manifest.get("deterministic", {})
    repeats = int(deterministic.get("timing_repeats", 7))
    if repeats < (1 if smoke else 7):
        raise ValueError("timing_repeats must be >=7 for qualification (>=1 for smoke)")
    warmups = int(deterministic.get("warmups_per_variant", 1))
    if warmups < (0 if smoke else 1):
        raise ValueError("qualification requires at least one warm-up per variant")
    statistical = manifest.get("statistical", {})
    seed_count = int(statistical.get("seed_count", 0))
    if not smoke and seed_count < 8:
        raise ValueError("statistical qualification requires at least eight paired seeds")
    if int(deterministic.get("seed", 2)) < 0 or int(statistical.get("first_seed", 101)) < 0:
        raise ValueError("qualification seeds must be non-negative")
    threshold_values = {
        name: statistical[name]
        for name in Thresholds.__dataclass_fields__ if name in statistical
    }
    thresholds = Thresholds(**threshold_values)
    if any(value < 0.0 for value in thresholds.__dict__.values()) \
            or thresholds.coarse_width == 0.0:
        raise ValueError("statistical thresholds must be non-negative and width positive")
    if not 0.0 <= thresholds.minimum_bin_pass_fraction <= 1.0:
        raise ValueError("minimum_bin_pass_fraction must be in [0,1]")


def variant_source(manifest_path: Path, variant: dict[str, Any]) -> tuple[Path, str]:
    path = resolve_path(manifest_path, variant["card"])
    return path, path.read_text(encoding="utf-8")


def verify_rejections(binary: Path, manifest_path: Path, manifest: dict[str, Any],
                      sources: dict[str, str], output: Path, timeout: float,
                      command_prefix: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_success = manifest["candidate"].get("success_only_regex", ())
    for case in manifest["rejections"]:
        base = case.get("base", "candidate")
        run = run_one(
            binary, command_prefix, manifest["accelerator"], manifest["parameters"],
            sources[base], int(case.get("seed", 2)),
            output / "rejections" / case["name"], f"rejection:{case['name']}",
            manifest[base], timeout, {}, fragment=case.get("fragment", ""),
            extra_environment=case.get("environment", {}), require_success=False,
        )
        receipt = run["receipt"]
        if receipt["returncode"] == 0:
            raise QualificationError(
                f"rejection {case['name']} silently succeeded (possible fallback)"
            )
        stderr = (run["directory"] / "stderr.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        require_regexes(stderr, [case["stderr_regex"]], f"rejection:{case['name']}")
        forbidden_artifacts = case.get(
            "forbidden_artifacts", REJECTION_SCIENCE_ARTIFACTS
        )
        produced = [
            name for name in forbidden_artifacts
            if (run["directory"] / name).is_file()
            and (run["directory"] / name).stat().st_size > 0
        ]
        if produced:
            raise QualificationError(
                f"rejection {case['name']} produced science artifacts: {produced}"
            )
        forbid_regexes(
            run["evidence_text"],
            [*candidate_success, *case.get("forbidden_regex", ())],
            f"rejection:{case['name']}",
        )
        records.append({
            "name": case["name"], "returncode": receipt["returncode"],
            "expected_diagnostic": case["stderr_regex"],
            "produced_science_artifacts": produced,
            "pass": True,
        })
    return records


def artifact_hash_matrix(runs: Sequence[dict[str, Any]], names: Sequence[str],
                         normalize_patterns: Sequence[str]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for run in runs:
        directory: Path = run["directory"]
        normalize_result(directory / "result.out", directory / "result.normalized",
                         normalize_patterns)
    expanded = tuple("result.normalized" if item == "result.out" else item
                     for item in names)
    for name in expanded:
        hashes = [sha256(run["directory"] / name) for run in runs]
        matrix[name] = {"sha256": hashes, "byte_identical": len(set(hashes)) == 1}
    return matrix


def timing_summary(runs: Sequence[dict[str, Any]], nonoverlap: Sequence[str]) -> dict[str, Any]:
    wall = [float(run["receipt"]["wall_seconds"]) for run in runs]
    stage_names = sorted(set().union(
        *(run["receipt"].get("stages", {}).keys() for run in runs)
    ))
    stages = {
        name: [float(run["receipt"]["stages"][name]) for run in runs]
        for name in stage_names
    }
    closure = []
    for run in runs:
        stage_seconds = math.fsum(
            float(run["receipt"].get("stages", {}).get(name, 0.0)) / 1000.0
            for name in nonoverlap
        )
        wall_seconds = float(run["receipt"]["wall_seconds"])
        closure.append({
            "wall_seconds": wall_seconds,
            "declared_nonoverlap_stage_seconds": stage_seconds,
            "consistent": stage_seconds <= wall_seconds + max(0.050, 0.02 * wall_seconds),
        })
    return {
        "samples_seconds": wall,
        "median_seconds": statistics.median(wall),
        "minimum_seconds": min(wall),
        "maximum_seconds": max(wall),
        "stage_samples_ms": stages,
        "stage_medians_ms": {
            name: statistics.median(values) for name, values in stages.items()
        },
        "nonoverlap_stage_closure": closure,
    }


def run_matrix(binary: Path, manifest_path: Path, manifest: dict[str, Any],
               sources: dict[str, str], output: Path, timeout: float,
               command_prefix: Sequence[str], smoke: bool) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    deterministic = manifest.get("deterministic", {})
    repeats = 1 if smoke else int(deterministic.get("timing_repeats", 7))
    seed = int(deterministic.get("seed", 2))
    warmups = 0 if smoke else int(deterministic.get("warmups_per_variant", 1))
    stage_patterns = manifest.get("stage_regex", {})
    for label in ("baseline", "candidate"):
        for index in range(warmups):
            run_one(
                binary, command_prefix, manifest["accelerator"], manifest["parameters"],
                sources[label], seed, output / "warmup" / f"{label}-{index + 1}",
                f"warmup:{label}:{index + 1}", manifest[label], timeout,
                stage_patterns.get(label, {}),
            )
    runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    for index in range(repeats):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for label in order:
            runs[label].append(run_one(
                binary, command_prefix, manifest["accelerator"], manifest["parameters"],
                sources[label], seed,
                output / "timing" / f"pair-{index + 1:02d}" / label,
                f"timing:{label}:{index + 1}", manifest[label], timeout,
                stage_patterns.get(label, {}),
            ))

    normalize_patterns = tuple(manifest.get("result_normalize_regex", DEFAULT_NORMALIZE))
    artifacts = tuple(manifest.get("deterministic_artifacts", REQUIRED_ARTIFACTS))
    hashes = {
        label: artifact_hash_matrix(items, artifacts, normalize_patterns)
        for label, items in runs.items()
    }
    require_repeat = bool(deterministic.get("require_repeat_byte_identity", True))
    repeat_ok = all(
        details["byte_identical"]
        for label in hashes.values() for details in label.values()
    )
    if require_repeat and not repeat_ok:
        raise QualificationError("within-selector deterministic artifact hashes differ")

    reference_dir = runs["baseline"][0]["directory"]
    candidate_dir = runs["candidate"][0]["directory"]
    numeric = compare_numeric_text(
        reference_dir / "result.normalized",
        candidate_dir / "result.normalized",
        float(deterministic.get("result_rtol", 5.0e-5)),
        float(deterministic.get("result_atol", 1.0e-12)),
    )
    if not numeric["pass"]:
        raise QualificationError("baseline/candidate normalized results exceed tolerance")
    cross_exact_names = tuple(deterministic.get("cross_mode_exact_artifacts", ()))
    cross_exact = {
        name: {
            "reference_sha256": sha256(reference_dir / name),
            "candidate_sha256": sha256(candidate_dir / name),
            "byte_identical": sha256(reference_dir / name) == sha256(candidate_dir / name),
        }
        for name in cross_exact_names
    }
    if any(not item["byte_identical"] for item in cross_exact.values()):
        raise QualificationError("declared cross-selector exact artifacts differ")

    nonoverlap = manifest.get("nonoverlap_stage_names", {})
    timings = {
        label: timing_summary(items, nonoverlap.get(label, ()))
        for label, items in runs.items()
    }
    if any(
        not item["consistent"]
        for label in timings.values()
        for item in label["nonoverlap_stage_closure"]
    ):
        raise QualificationError("declared non-overlapping stage times exceed wall time")
    timings["candidate_speedup_vs_baseline"] = (
        timings["baseline"]["median_seconds"]
        / timings["candidate"]["median_seconds"]
    )
    return {
        "within_selector_hashes": hashes,
        "within_selector_byte_identity": repeat_ok,
        "cross_selector_exact": cross_exact,
        "normalized_result_comparison": numeric,
        "timing": timings,
    }, runs


def run_statistical(binary: Path, manifest: dict[str, Any], sources: dict[str, str],
                    output: Path, timeout: float, command_prefix: Sequence[str],
                    smoke: bool) -> dict[str, Any] | None:
    configuration = manifest.get("statistical", {})
    count = 0 if smoke else int(configuration.get("seed_count", 0))
    if count == 0:
        return None
    first_seed = int(configuration.get("first_seed", 101))
    stage_patterns = manifest.get("stage_regex", {})
    reference = []
    candidate = []
    for offset in range(count):
        seed = first_seed + offset
        order = ("baseline", "candidate") if offset % 2 == 0 else ("candidate", "baseline")
        directories: dict[str, Path] = {}
        for label in order:
            run = run_one(
                binary, command_prefix, manifest["accelerator"], manifest["parameters"],
                sources[label], seed,
                output / "statistical" / f"seed-{seed:04d}" / label,
                f"statistical:{label}:{seed}", manifest[label], timeout,
                stage_patterns.get(label, {}),
            )
            directories[label] = run["directory"]
        reference.append(read_spectrum(directories["baseline"] / "lumi.ee.out.dLdW.fne"))
        candidate.append(read_spectrum(directories["candidate"] / "lumi.ee.out.dLdW.fne"))
    values = {
        name: configuration[name]
        for name in Thresholds.__dataclass_fields__ if name in configuration
    }
    result = compare_ensemble(reference, candidate, Thresholds(**values))
    (output / "statistical" / "lumi-spectrum-equivalence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        raise QualificationError("paired-seed luminosity-spectrum equivalence failed")
    return result


def git_revision(binary: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(binary.parent), "rev-parse", "HEAD"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="one untuned pair, no warm-up/statistical ensemble; result is SMOKE_PASS",
    )
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if not os.access(binary, os.X_OK):
        parser.error(f"binary is not executable: {binary}")
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        validate_manifest(manifest, args.smoke)
    except ValueError as error:
        parser.error(str(error))
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be an absent or empty directory; evidence is never reused")
    output.mkdir(parents=True, exist_ok=True)
    command_prefix = tuple(str(item) for item in manifest.get("command_prefix", ()))
    cards: dict[str, Path] = {}
    sources: dict[str, str] = {}
    for label in ("baseline", "candidate"):
        cards[label], sources[label] = variant_source(manifest_path, manifest[label])

    try:
        card_equivalence = compare_card_physics(
            sources["baseline"], sources["candidate"], manifest["accelerator"],
            manifest["parameters"], manifest.get("selector_assignment_keys", ()),
        )
        if not card_equivalence["equivalent"]:
            raise QualificationError(
                "baseline/candidate physics cards differ outside selector_assignment_keys: "
                + ", ".join(card_equivalence["unexpected_differences"])
            )
        rejections = verify_rejections(
            binary, manifest_path, manifest, sources, output,
            args.timeout_seconds, command_prefix,
        )
        deterministic, _ = run_matrix(
            binary, manifest_path, manifest, sources, output,
            args.timeout_seconds, command_prefix, args.smoke,
        )
        statistical = run_statistical(
            binary, manifest, sources, output, args.timeout_seconds,
            command_prefix, args.smoke,
        )
    except (QualificationError, ValueError) as error:
        failure = {"schema": "feebl.priority12.qualification-result.v1",
                   "status": "FAIL", "reason": str(error)}
        (output / "qualification-result.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    result = {
        "schema": "feebl.priority12.qualification-result.v1",
        "status": "SMOKE_PASS" if args.smoke else "PASS",
        "scope": "smoke" if args.smoke else "qualification",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "binary_git_revision_if_discoverable": git_revision(binary),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "cards": {label: {"path": str(path), "sha256": sha256(path)}
                  for label, path in cards.items()},
        "card_physics_equivalence": card_equivalence,
        "platform": platform.platform(),
        "python": sys.version,
        "fail_closed_rejections": rejections,
        "deterministic_and_timing": deterministic,
        "statistical_luminosity_spectrum": statistical,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output / "qualification-result.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
