#!/usr/bin/env python3
"""Independent validation and promotion for the photon-luminosity benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "feebl.legacy_photon_lumi_fast.e2e.v4"
CLASSIFICATION = "exploratory_end_to_end_feature_path"
EXPECTED_AGGREGATES = tuple(
    ["lumi_eg", "lumi_ge", "lumi_gg", "lumi_gg_high"]
    + [f"lumi[{row}][{column}]" for row in range(3) for column in range(3)]
)
AGGREGATE_RE = re.compile(
    r"^(lumi_(?:eg|ge|gg|gg_high)|lumi\[[0-2]\]\[[0-2]\])\s*=\s*"
    r"([-+0-9.eE]+)",
    re.MULTILINE,
)
SUMMARY_RE = re.compile(
    r"FEEBL legacy photon-lumi summary:\s*"
    r"requested=factorized-exact-v2.*?factorized_calls=(\d+).*?"
    r"cartesian_calls=(\d+).*?cartesian_pairs_visited=(\d+).*?"
    r"cartesian_pairs_avoided=(\d+)",
    re.DOTALL,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CSV_FIELDS = (
    "schema", "valid", "classification", "claim_eligible", "case",
    "macroparticles_per_beam", "repetitions", "pair", "repetition",
    "order_index", "variant", "seconds", "timing_sha256", "result_sha256", "rng_sha256",
    "stdout_sha256", "stderr_sha256", "physics_equivalent",
    "rng_identical", "provenance_valid", "cartesian_median_seconds",
    "factorized_median_seconds", "ratio_of_medians_speedup",
    "median_paired_speedup", "accuracy_gate_relative",
    "accuracy_gate_absolute",
)


class ValidationError(RuntimeError):
    """Raised when a staging or promoted result violates the evidence schema."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finite_positive(value: Any, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name} is not numeric")
    converted = float(value)
    require(math.isfinite(converted) and converted > 0.0,
            f"{name} is not finite and positive")
    return converted


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{name} is not an integer")
    require(value >= minimum, f"{name} is below {minimum}")
    return value


def close_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)


def explicit_build_text(value: Any, name: str) -> str:
    require(isinstance(value, str), f"{name} provenance is absent")
    normalized = value.strip()
    rejected = {"unknown", "unspecified", "none", "n/a"}
    require(bool(normalized) and normalized.lower() not in rejected and
            not normalized.lower().startswith("unknown"),
            f"{name} provenance must be explicit and not 'unknown'")
    return normalized


def safe_relative(root: Path, value: Any, description: str) -> Path:
    require(isinstance(value, str) and value, f"{description} path is absent")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"{description} path escapes the evidence directory")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    require(resolved == resolved_root or resolved_root in resolved.parents,
            f"{description} resolved outside the evidence directory")
    return resolved


def parse_aggregates(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="strict")
    matches = AGGREGATE_RE.findall(text)
    require(len(matches) == len(EXPECTED_AGGREGATES),
            f"{path}: expected exactly 13 unique photon aggregate rows")
    found = {name: float(value) for name, value in matches}
    require(set(found) == set(EXPECTED_AGGREGATES),
            f"{path}: expected exactly 13 photon aggregates")
    require(all(math.isfinite(value) for value in found.values()),
            f"{path}: aggregate is non-finite")
    return found


def validate_artifact(record: Any, root: Path, description: str,
                      verify_files: bool) -> Path:
    require(isinstance(record, dict), f"{description} record is absent")
    path = safe_relative(root, record.get("path"), description)
    digest = record.get("sha256")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{description} SHA-256 is malformed")
    if verify_files:
        require(path.is_file(), f"{description} file is absent")
        require(sha256_file(path) == digest, f"{description} hash mismatch")
    return path


def validate_provenance(document: dict[str, Any], root: Path,
                        verify_files: bool) -> None:
    provenance = document.get("provenance")
    require(isinstance(provenance, dict), "provenance is absent")
    revision = provenance.get("git_revision")
    require(isinstance(revision, str) and
            re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None,
            "git revision is not a full object ID")
    require(isinstance(provenance.get("git_dirty"), bool),
            "git dirty flag is not Boolean")
    status = provenance.get("git_status_porcelain")
    require(isinstance(status, list) and all(isinstance(line, str) for line in status),
            "git status listing is absent")
    require(bool(status) == provenance["git_dirty"],
            "git dirty flag disagrees with status listing")
    require(sha256_text("\n".join(status)) == provenance.get("git_status_sha256"),
            "git status digest is inconsistent")
    compiler = explicit_build_text(provenance.get("compiler"), "compiler")
    compile_flags = explicit_build_text(
        provenance.get("compile_flags"), "compile flags"
    )
    require(isinstance(provenance.get("command_template"), list) and
            provenance["command_template"], "command template is absent")
    environment = provenance.get("performance_environment")
    require(isinstance(environment, dict) and
            environment.get("OMP_NUM_THREADS") == "1" and
            environment.get("OMP_DYNAMIC") == "FALSE" and
            environment.get("OMP_PROC_BIND") == "close",
            "single-thread OpenMP benchmark environment is not recorded")
    host = provenance.get("host")
    require(isinstance(host, dict), "host metadata is absent")
    require(integer(host.get("pinned_cpu"), "pinned CPU") >= 0,
            "pinned CPU is invalid")
    require(isinstance(host.get("allowed_cpu_list"), list) and
            host["pinned_cpu"] in host["allowed_cpu_list"],
            "pinned CPU is outside recorded affinity")
    require(environment.get("OMP_PLACES") == f"{{{host['pinned_cpu']}}}",
            "OpenMP place does not match the pinned benchmark CPU")
    artifacts = provenance.get("artifacts")
    require(isinstance(artifacts, dict), "artifact provenance map is absent")
    required = {
        "benchmark_binary", "deck_template", "runner", "validator",
        "validator_mutation_tests", "fast_source", "fast_header",
        "grid_source", "switch_source", "ldd_output", "lscpu_output",
        "build_manifest",
    }
    require(required.issubset(artifacts), "required provenance artifacts are absent")
    artifact_paths: dict[str, Path] = {}
    for name, record in artifacts.items():
        artifact_paths[name] = validate_artifact(
            record, root, f"provenance artifact {name}", verify_files
        )
    if provenance["git_dirty"]:
        require("working_tree_diff" in artifacts,
                "dirty-tree provenance omitted the retained working-tree diff")
    else:
        require("working_tree_diff" not in artifacts,
                "clean-tree provenance unexpectedly contains a dirty diff")
    libraries = provenance.get("linked_libraries")
    require(isinstance(libraries, list) and libraries,
            "linked-library provenance is absent")
    original_paths: set[str] = set()
    retained_paths: set[Path] = set()
    for index, library in enumerate(libraries):
        require(isinstance(library, dict), f"linked library {index} is invalid")
        original = library.get("original_host_path")
        require(isinstance(original, str) and Path(original).is_absolute(),
                f"linked library {index} original host path is not absolute")
        resolved_host = library.get("resolved_host_path")
        require(isinstance(resolved_host, str) and
                Path(resolved_host).is_absolute(),
                f"linked library {index} resolved host path is not absolute")
        require(original not in original_paths,
                f"linked library {index} original path is duplicated")
        original_paths.add(original)
        retained = validate_artifact(
            library.get("retained"), root,
            f"retained linked library {index}", verify_files,
        )
        require(retained not in retained_paths,
                f"linked library {index} retained path is duplicated")
        retained_paths.add(retained)
    if verify_files:
        if provenance["git_dirty"]:
            require(artifact_paths["working_tree_diff"].stat().st_size > 0,
                    "retained working-tree diff is empty")
        try:
            build_manifest = json.loads(
                artifact_paths["build_manifest"].read_text(
                    encoding="utf-8", errors="strict"
                )
            )
        except json.JSONDecodeError as exc:
            raise ValidationError("retained build manifest is invalid JSON") from exc
        expected_manifest_keys = {
            "schema", "compiler", "compile_flags", "binary_sha256",
            "git_revision", "git_dirty", "git_status_sha256",
            "working_tree_diff_sha256", "retained_artifact_sha256",
            "retained_library_sha256",
        }
        require(isinstance(build_manifest, dict) and
                set(build_manifest) == expected_manifest_keys,
                "retained build manifest fields are invalid")
        require(build_manifest["schema"] ==
                "feebl.legacy_photon_lumi_fast.build.v1",
                "retained build manifest schema is invalid")
        require(build_manifest["compiler"] == compiler and
                build_manifest["compile_flags"] == compile_flags,
                "retained build manifest disagrees with compiler provenance")
        require(build_manifest["binary_sha256"] ==
                artifacts["benchmark_binary"]["sha256"],
                "retained build manifest binary hash is inconsistent")
        require(build_manifest["git_revision"] == revision and
                build_manifest["git_dirty"] == provenance["git_dirty"] and
                build_manifest["git_status_sha256"] ==
                provenance["git_status_sha256"],
                "retained build manifest disagrees with git provenance")
        expected_diff = (artifacts["working_tree_diff"]["sha256"]
                         if provenance["git_dirty"] else None)
        require(build_manifest["working_tree_diff_sha256"] == expected_diff,
                "retained build manifest dirty-diff hash is inconsistent")
        expected_artifacts = {
            name: record["sha256"] for name, record in artifacts.items()
            if name != "build_manifest"
        }
        require(build_manifest["retained_artifact_sha256"] ==
                dict(sorted(expected_artifacts.items())),
                "retained build manifest artifact hashes are incomplete")
        expected_libraries = [
            row["retained"]["sha256"] for row in libraries
        ]
        require(build_manifest["retained_library_sha256"] == expected_libraries,
                "retained build manifest library hashes are inconsistent")
        ldd_text = artifact_paths["ldd_output"].read_text(encoding="utf-8")
        reported: set[str] = set()
        for line in ldd_text.splitlines():
            match = re.search(r"=>\s+(/\S+)", line)
            if match is None:
                match = re.match(r"\s*(/\S+)\s+\(", line)
            if match is not None:
                reported.add(match.group(1))
        require(reported and reported == original_paths,
                "retained linked libraries differ from the captured ldd output")


def physics_close(cartesian: dict[str, float], factorized: dict[str, float],
                  relative_gate: float, absolute_gate: float) -> bool:
    for name in EXPECTED_AGGREGATES:
        left, right = cartesian[name], factorized[name]
        tolerance = max(
            absolute_gate,
            relative_gate * max(abs(left), abs(right)),
        )
        if abs(left - right) > tolerance:
            return False
    return True


def sample_paths(sample: dict[str, Any], root: Path,
                 verify_files: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("stdout", "stderr", "result", "rng", "timing"):
        record = sample.get("artifacts", {}).get(name)
        paths[name] = validate_artifact(
            record, root, f"sample {sample.get('case')} {sample.get('pair')} {name}",
            verify_files,
        )
    return paths


def parse_timing(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: raw timing record is invalid JSON") from exc
    require(isinstance(record, dict) and set(record) == {
        "clock", "start_ns", "end_ns", "elapsed_ns",
    }, f"{path}: raw timing record fields are invalid")
    require(record["clock"] == "time.perf_counter_ns",
            f"{path}: raw timing clock is not monotonic perf_counter_ns")
    start = integer(record["start_ns"], f"{path} timing start")
    end = integer(record["end_ns"], f"{path} timing end")
    elapsed = integer(record["elapsed_ns"], f"{path} timing elapsed", 1)
    require(end > start and elapsed == end - start,
            f"{path}: raw timing interval is inconsistent")
    return record


def validate_raw_timing(paths: dict[str, Path], seconds: float,
                        description: str, verify_files: bool) -> None:
    if not verify_files:
        return
    timing = parse_timing(paths["timing"])
    recomputed = timing["elapsed_ns"] * 1.0e-9
    require(close_number(seconds, recomputed),
            f"{description} seconds differ from raw monotonic timing")


def validate_sample_command(sample: dict[str, Any], variant: str,
                            description: str) -> None:
    command = sample.get("command")
    require(isinstance(command, list) and len(command) >= 2 and
            all(isinstance(argument, str) for argument in command),
            f"{description} command is absent")
    require(command[-2:] == [variant, "result.out"],
            f"{description} command does not select its recorded variant")


def validate_variant_provenance(sample: dict[str, Any], variant: str,
                                stderr: str, description: str,
                                verify_files: bool) -> None:
    if variant == "cartesian":
        require("provenance" not in sample,
                f"{description} Cartesian run has factorized provenance")
        if verify_files:
            require("FEEBL legacy photon-lumi summary" not in stderr,
                    f"{description} Cartesian run emitted fast-path provenance")
        return
    require(isinstance(sample.get("provenance"), dict),
            f"{description} stored fast-path provenance is absent")
    if not verify_files:
        return
    match = SUMMARY_RE.search(stderr)
    require(match is not None, f"{description} fast-path provenance is absent")
    calls, cart_calls, visited, avoided = map(int, match.groups())
    require(calls > 0 and cart_calls == 0 and visited == 0 and avoided > 0,
            f"{description} fast-path provenance counters are invalid")
    require(sample["provenance"] == {
        "factorized_calls": calls,
        "cartesian_calls": cart_calls,
        "cartesian_pairs_visited": visited,
        "cartesian_pairs_avoided": avoided,
    }, f"{description} stored fast-path provenance differs from stderr")


def register_unique_paths(paths: dict[str, Path], seen: set[Path],
                          description: str) -> None:
    for name, path in paths.items():
        require(path not in seen,
                f"{description} reuses a prior raw {name} artifact")
        seen.add(path)


def computed_rows(document: dict[str, Any]) -> list[dict[str, str]]:
    boolean = lambda value: "true" if value else "false"
    rows: list[dict[str, str]] = []
    for case in document["cases"]:
        summary = case["summary"]
        for sample in sorted(case["samples"],
                             key=lambda item: (item["pair"], item["order_index"])):
            rows.append({
                "schema": document["schema"],
                "valid": boolean(document["valid"]),
                "classification": document["classification"],
                "claim_eligible": boolean(document["eligible_for_project_level_2x_claim"]),
                "case": case["name"],
                "macroparticles_per_beam": str(case["macroparticles_per_beam"]),
                "repetitions": str(document["repetitions"]),
                "pair": str(sample["pair"]),
                "repetition": str(sample["repetition"]),
                "order_index": str(sample["order_index"]),
                "variant": sample["variant"],
                "seconds": repr(float(sample["seconds"])),
                "timing_sha256": sample["artifacts"]["timing"]["sha256"],
                "result_sha256": sample["artifacts"]["result"]["sha256"],
                "rng_sha256": sample["artifacts"]["rng"]["sha256"],
                "stdout_sha256": sample["artifacts"]["stdout"]["sha256"],
                "stderr_sha256": sample["artifacts"]["stderr"]["sha256"],
                "physics_equivalent": boolean(sample["physics_equivalent"]),
                "rng_identical": boolean(sample["rng_identical"]),
                "provenance_valid": "true",
                "cartesian_median_seconds": repr(float(summary["cartesian_median_seconds"])),
                "factorized_median_seconds": repr(float(summary["factorized_median_seconds"])),
                "ratio_of_medians_speedup": repr(float(summary["ratio_of_medians_speedup"])),
                "median_paired_speedup": repr(float(summary["median_paired_speedup"])),
                "accuracy_gate_relative": repr(float(document["accuracy_gate_relative"])),
                "accuracy_gate_absolute": repr(float(document["accuracy_gate_absolute"])),
            })
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == CSV_FIELDS,
                "CSV header differs from schema")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(root: Path,
                      prospective_files: dict[str, str] | None = None) -> None:
    prospective = prospective_files or {}
    manifest = root / "SHA256SUMS"
    require(manifest.is_file(), "checksum manifest is absent")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "checksum manifest has a malformed row")
        digest, name = match.groups()
        require(name not in listed, f"duplicate manifest entry {name}")
        listed[name] = digest
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    expected.update(prospective)
    require(set(listed) == expected, "checksum manifest file set is incomplete")
    for name, digest in listed.items():
        if name in prospective:
            require(prospective[name] == digest,
                    f"checksum manifest mismatch for prospective {name}")
        else:
            source = root / name
            require(source.is_file(),
                    f"checksum manifest source is absent for {name}")
            require(sha256_file(source) == digest,
                    f"checksum manifest mismatch for {name}")


def validate_document(document: dict[str, Any], root: Path,
                      expected_valid: bool, verify_files: bool = True) -> None:
    require(document.get("schema") == SCHEMA, "unexpected JSON schema")
    require(document.get("valid") is expected_valid, "unexpected valid state")
    require(document.get("validation_status") ==
            ("validated" if expected_valid else "unvalidated_staging"),
            "unexpected validation status")
    require(document.get("classification") == CLASSIFICATION,
            "benchmark classification is invalid")
    require(document.get("eligible_for_project_level_2x_claim") is False,
            "feature benchmark was incorrectly made project-claim eligible")
    require("not a whole-program 2x claim" in document.get("disclaimer", ""),
            "project-claim disclaimer is absent")
    authenticity = document.get("evidence_authenticity")
    require(isinstance(authenticity, dict) and
            authenticity.get("external_cryptographic_witness") is False and
            "do not provide independent cryptographic witness authenticity" in
            authenticity.get("statement", ""),
            "unkeyed-evidence authenticity limitation is absent")
    repetitions = integer(document.get("repetitions"), "repetitions", 5)
    warmups = integer(document.get("warmups_per_variant_case"), "warmups", 1)
    require(warmups >= 1, "at least one warmup is required")
    relative_gate = finite_positive(
        document.get("accuracy_gate_relative"), "relative accuracy gate"
    )
    absolute_gate = finite_positive(
        document.get("accuracy_gate_absolute"), "absolute accuracy gate"
    )
    require(relative_gate <= 2.0e-6,
            "relative accuracy gate is wider than the audited 2e-6 gate")
    require(absolute_gate <= 2.0e-6,
            "absolute accuracy gate is wider than the audited 2e-6 gate")
    validate_provenance(document, root, verify_files)
    cases = document.get("cases")
    require(isinstance(cases, list) and cases, "benchmark cases are absent")
    case_names: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "case is not an object")
        name = case.get("name")
        require(isinstance(name, str) and name and name not in case_names,
                "case name is absent or duplicated")
        case_names.add(name)
        integer(case.get("macroparticles_per_beam"), "macroparticles", 4000)
        deck = validate_artifact(case.get("deck"), root, f"case {name} deck", verify_files)
        if verify_files:
            deck_text = deck.read_text(encoding="utf-8")
            require(f"n_m={case['macroparticles_per_beam']}" in deck_text,
                    f"case {name} deck does not contain requested macroparticle count")
        warmup_samples = case.get("warmups")
        require(isinstance(warmup_samples, list) and
                len(warmup_samples) == 2 * warmups,
                f"case {name} warmup count is invalid")
        artifact_paths_seen: set[Path] = set()
        warmup_groups: dict[int, dict[str, dict[str, Any]]] = {}
        for sample in warmup_samples:
            description = f"case {name} warmup"
            require(sample.get("case") == name,
                    f"{description} case mismatch")
            pair = integer(sample.get("pair"), f"{description} pair")
            require(0 <= pair < warmups,
                    f"{description} pair index is invalid")
            require(sample.get("repetition") == pair,
                    f"{description} repetition does not equal pair")
            order = integer(sample.get("order_index"), f"{description} order")
            require(order in (0, 1), f"{description} order index is invalid")
            variant = sample.get("variant")
            require(variant in ("cartesian", "factorized"),
                    f"{description} variant is invalid")
            require(sample.get("warmup") is True,
                    f"{description} is not marked as a warmup")
            require(sample.get("exit_code") == 0, f"case {name} warmup failed")
            seconds = finite_positive(
                sample.get("seconds"), f"case {name} warmup time"
            )
            validate_sample_command(sample, variant, description)
            paths = sample_paths(sample, root, verify_files)
            register_unique_paths(paths, artifact_paths_seen, description)
            validate_raw_timing(paths, seconds, description, verify_files)
            aggregates = (parse_aggregates(paths["result"])
                          if verify_files else sample.get("aggregates"))
            require(isinstance(sample.get("aggregates"), dict) and
                    aggregates == sample["aggregates"],
                    f"{description} stored aggregates differ from raw result")
            stderr = (paths["stderr"].read_text(encoding="utf-8")
                      if verify_files else "")
            validate_variant_provenance(
                sample, variant, stderr, description, verify_files
            )
            require(variant not in warmup_groups.setdefault(pair, {}),
                    f"{description} pair {pair} duplicates {variant}")
            warmup_groups[pair][variant] = sample
        for pair in range(warmups):
            group = warmup_groups.get(pair, {})
            require(set(group) == {"cartesian", "factorized"},
                    f"case {name} warmup pair {pair} is incomplete")
            cart, fast = group["cartesian"], group["factorized"]
            require(cart["order_index"] == 0 and fast["order_index"] == 1,
                    f"case {name} warmup pair {pair} order is invalid")
            require(physics_close(cart["aggregates"], fast["aggregates"],
                                  relative_gate, absolute_gate),
                    f"case {name} warmup pair {pair} fails physics gate")
            require(cart["artifacts"]["rng"]["sha256"] ==
                    fast["artifacts"]["rng"]["sha256"],
                    f"case {name} warmup pair {pair} changes RNG state")
        samples = case.get("samples")
        require(isinstance(samples, list) and len(samples) == 2 * repetitions,
                f"case {name} sample count is invalid")
        groups: dict[int, dict[str, dict[str, Any]]] = {}
        variant_times = {"cartesian": [], "factorized": []}
        for sample in samples:
            require(sample.get("case") == name, f"case {name} sample case mismatch")
            pair = integer(sample.get("pair"), f"case {name} pair")
            require(0 <= pair < repetitions, f"case {name} pair index is invalid")
            require(sample.get("repetition") == pair,
                    f"case {name} repetition does not equal pair")
            order = integer(sample.get("order_index"), f"case {name} order")
            require(order in (0, 1), f"case {name} order index is invalid")
            variant = sample.get("variant")
            require(variant in ("cartesian", "factorized"),
                    f"case {name} variant is invalid")
            require(sample.get("warmup") is False,
                    f"case {name} timed sample is marked as a warmup")
            require(sample.get("exit_code") == 0, f"case {name} sample failed")
            seconds = finite_positive(sample.get("seconds"), f"case {name} time")
            validate_sample_command(sample, variant, f"case {name} sample")
            paths = sample_paths(sample, root, verify_files)
            register_unique_paths(
                paths, artifact_paths_seen, f"case {name} sample"
            )
            validate_raw_timing(
                paths, seconds, f"case {name} sample", verify_files
            )
            aggregates = parse_aggregates(paths["result"]) if verify_files else sample.get("aggregates")
            require(isinstance(sample.get("aggregates"), dict),
                    f"case {name} stored aggregates are absent")
            require(aggregates == sample["aggregates"],
                    f"case {name} stored aggregates differ from raw result")
            stderr = paths["stderr"].read_text(encoding="utf-8") if verify_files else ""
            validate_variant_provenance(
                sample, variant, stderr, f"case {name} sample", verify_files
            )
            require(variant not in groups.setdefault(pair, {}),
                    f"case {name} pair {pair} duplicates {variant}")
            groups[pair][variant] = sample
            variant_times[variant].append(seconds)
        paired_speedups: list[float] = []
        for pair in range(repetitions):
            group = groups.get(pair, {})
            require(set(group) == {"cartesian", "factorized"},
                    f"case {name} pair {pair} is incomplete")
            cart, fast = group["cartesian"], group["factorized"]
            require({cart["order_index"], fast["order_index"]} == {0, 1},
                    f"case {name} pair {pair} order is invalid")
            expected_first = "cartesian" if pair % 2 == 0 else "factorized"
            first = cart if cart["order_index"] == 0 else fast
            require(first["variant"] == expected_first,
                    f"case {name} pair {pair} does not alternate AB/BA")
            equivalent = physics_close(
                cart["aggregates"], fast["aggregates"],
                relative_gate, absolute_gate,
            )
            rng_identical = cart["artifacts"]["rng"]["sha256"] == fast["artifacts"]["rng"]["sha256"]
            require(equivalent, f"case {name} pair {pair} fails physics gate")
            require(rng_identical, f"case {name} pair {pair} changes RNG state")
            require(cart.get("physics_equivalent") is True and
                    fast.get("physics_equivalent") is True,
                    f"case {name} pair {pair} physics status is not true")
            require(cart.get("rng_identical") is True and fast.get("rng_identical") is True,
                    f"case {name} pair {pair} RNG status is not true")
            paired_speedups.append(cart["seconds"] / fast["seconds"])
        cart_median = statistics.median(variant_times["cartesian"])
        fast_median = statistics.median(variant_times["factorized"])
        calculated = {
            "cartesian_median_seconds": cart_median,
            "factorized_median_seconds": fast_median,
            "ratio_of_medians_speedup": cart_median / fast_median,
            "median_paired_speedup": statistics.median(paired_speedups),
            "cartesian_min_seconds": min(variant_times["cartesian"]),
            "cartesian_max_seconds": max(variant_times["cartesian"]),
            "factorized_min_seconds": min(variant_times["factorized"]),
            "factorized_max_seconds": max(variant_times["factorized"]),
        }
        summary = case.get("summary")
        require(isinstance(summary, dict) and set(summary) == set(calculated),
                f"case {name} summary fields are invalid")
        for key, value in calculated.items():
            require(close_number(finite_positive(summary.get(key), f"case {name} {key}"), value),
                    f"case {name} {key} was not independently recomputed")
    if expected_valid:
        validation = document.get("validation")
        require(isinstance(validation, dict), "validation attestation is absent")
        retained_validator = document["provenance"]["artifacts"]["validator"]
        require(validation.get("validator") == retained_validator.get("path"),
                "validation attestation does not name the retained validator")
        digest = validation.get("validator_sha256")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                "validator attestation hash is malformed")
        require(digest == retained_validator.get("sha256"),
                "validator attestation hash does not match retained provenance")
        if verify_files:
            validate_manifest(root)
    else:
        require("validation" not in document,
                "staging document already contains a validation attestation")


def validate_pair(json_path: Path, csv_path: Path, expected_valid: bool,
                  verify_files: bool = True) -> tuple[dict[str, Any], list[dict[str, str]]]:
    document = json.loads(json_path.read_text(encoding="utf-8"))
    require(document.get("evidence_root") == ".",
            "evidence root must be relocatable and equal to '.'")
    root = json_path.resolve().parent
    validate_document(document, root, expected_valid, verify_files)
    rows = read_csv(csv_path)
    expected_rows = computed_rows(document)
    require(rows == expected_rows, "CSV differs from independently generated rows")
    return document, rows


def write_manifest(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    temporary = root / "SHA256SUMS.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, root / "SHA256SUMS")


def atomic_write_bytes(destination: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def promote(staging_json: Path, staging_csv: Path,
            final_json: Path, final_csv: Path) -> None:
    document, _ = validate_pair(staging_json, staging_csv, False, True)
    root = staging_json.resolve().parent
    require(staging_csv.parent.resolve() == root.resolve() and
            final_json.parent.resolve() == root.resolve() and
            final_csv.parent.resolve() == root.resolve(),
            "staging and promoted JSON/CSV must live at evidence root")
    manifest = root / "SHA256SUMS"
    require(not final_json.exists() and not final_csv.exists() and
            not manifest.exists(),
            "refusing promotion over existing final evidence")
    promoted = copy.deepcopy(document)
    promoted["valid"] = True
    promoted["validation_status"] = "validated"
    retained_validator = promoted["provenance"]["artifacts"]["validator"]
    promoted["validation"] = {
        "validator": retained_validator["path"],
        "validator_sha256": retained_validator["sha256"],
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [
            "raw physics aggregates independently reparsed",
            "2e-6 relative and 2e-6 absolute physics gates applied to every paired run",
            "RNG state SHA-256 identical in every paired run",
            "factorized provenance requires zero Cartesian visits",
            "AB/BA order, raw timings, medians, and paired speedups recomputed",
            "source, binary, build manifest, dirty diff when applicable, harness, host, affinity, and linked libraries hashed",
            "raw monotonic timing intervals independently reparsed",
            "CSV regenerated from validated JSON",
            "unkeyed evidence explicitly disclaims witness authenticity",
        ],
    }
    candidate_root = Path(tempfile.mkdtemp(
        prefix=f".{root.name}.prepublish-", dir=root.parent
    ))
    candidate_removed = False
    try:
        for path in root.rglob("*"):
            require(not path.is_symlink(),
                    "evidence tree contains an unsupported symbolic link")
        shutil.copytree(root, candidate_root, dirs_exist_ok=True)
        candidate_staging_json = candidate_root / staging_json.name
        candidate_staging_csv = candidate_root / staging_csv.name
        candidate_staging_json.unlink()
        candidate_staging_csv.unlink()
        candidate_final_json = candidate_root / final_json.name
        candidate_final_csv = candidate_root / final_csv.name
        candidate_final_json.write_text(
            json.dumps(promoted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(candidate_final_csv, computed_rows(promoted))
        write_manifest(candidate_root)
        validate_pair(
            candidate_final_json, candidate_final_csv, True, True
        )
        final_json_bytes = candidate_final_json.read_bytes()
        final_csv_bytes = candidate_final_csv.read_bytes()
        manifest_bytes = (candidate_root / "SHA256SUMS").read_bytes()
        shutil.rmtree(candidate_root)
        candidate_removed = True

        # Publish dependencies first.  The validated JSON is the final atomic
        # marker and is deliberately absent throughout every earlier step.
        atomic_write_bytes(final_csv, final_csv_bytes)
        atomic_write_bytes(manifest, manifest_bytes)
        staging_json.unlink()
        staging_csv.unlink()
        validate_manifest(root, {
            final_json.name: hashlib.sha256(final_json_bytes).hexdigest(),
        })
        require(read_csv(final_csv) == computed_rows(promoted),
                "published CSV changed after prepublication validation")
        atomic_write_bytes(final_json, final_json_bytes)
    except Exception:
        final_json.unlink(missing_ok=True)
        raise
    finally:
        if not candidate_removed:
            shutil.rmtree(candidate_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--json", type=Path, required=True)
    validate.add_argument("--csv", type=Path, required=True)
    validate.add_argument("--expected-valid", choices=("true", "false"), required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--staging-json", type=Path, required=True)
    promote_parser.add_argument("--staging-csv", type=Path, required=True)
    promote_parser.add_argument("--final-json", type=Path, required=True)
    promote_parser.add_argument("--final-csv", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            validate_pair(args.json, args.csv, args.expected_valid == "true", True)
        else:
            promote(args.staging_json, args.staging_csv,
                    args.final_json, args.final_csv)
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        raise SystemExit(f"photon benchmark validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
