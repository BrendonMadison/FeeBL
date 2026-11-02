#!/usr/bin/env python3
"""Support and independent validation for the executor microbenchmark.

Only this validator may promote an unvalidated staging pair to a valid JSON/CSV
result pair. It is standard-library-only so validation needs no FeeBL runtime.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "feebl.modern_cpu_executor.microbenchmark.v2"
CLASSIFICATION = "exploratory_microkernel"
WORKER_COUNTS = (1, 2, 4, 8, 10)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEX64_RE = re.compile(r"^0x[0-9a-f]{16}$")
CSV_FIELDS = (
    "schema",
    "valid",
    "classification",
    "claim_eligible",
    "preset",
    "shape",
    "beam1_slices",
    "beam2_slices",
    "substeps",
    "encounters",
    "waves",
    "intensity",
    "iterations",
    "variant",
    "execution",
    "workers",
    "schedule",
    "timing_scope",
    "warmups",
    "repetition",
    "order_index",
    "seconds",
    "checksum",
    "oracle_checksum",
    "deterministic_checksum_match",
    "report_success",
    "used_serial_fallback",
    "attempted_encounters",
    "completed_encounters",
    "published_encounters",
    "requested_workers",
    "actual_workers",
    "parallel_regions",
    "preflight_bytes",
    "construction_seconds",
    "median_seconds",
    "p95_seconds",
    "min_seconds",
    "max_seconds",
    "exploratory_speedup_vs_serial_median",
)


class ValidationError(RuntimeError):
    """The result pair does not satisfy the microbenchmark schema."""


def fail(message: str) -> "None":
    raise SystemExit(f"modern CPU executor benchmark: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def run_text(argv: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(
            argv, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot run {' '.join(argv)!r}: {exc}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def expand_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for item in spec.strip().split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            fail(f"cannot parse CPU-list token {item!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            fail(f"descending CPU-list range {item!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        fail("the process has an empty CPU affinity")
    return cpus


def read_first(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def parse_assignments(values: Iterable[str], description: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            fail(f"{description} entry {value!r} is not NAME=VALUE")
        name, assigned = value.split("=", 1)
        if not name or name in result:
            fail(f"invalid or duplicate {description} name {name!r}")
        result[name] = assigned
    return result


def find_openmp_runtime(binary: Path) -> Path:
    candidates: list[Path] = []
    for line in run_text(["ldd", str(binary)]).splitlines():
        match = re.search(r"\b(lib(?:gomp|omp)\.so[^\s]*)\s+=>\s+(\S+)", line)
        if match and match.group(2) != "not":
            candidates.append(Path(match.group(2)))
    if len(candidates) != 1 or not candidates[0].is_file():
        fail("could not identify exactly one linked libgomp/libomp runtime")
    return candidates[0]


def command_affinity(args: argparse.Namespace) -> None:
    if platform.system() != "Linux":
        fail("physical-core affinity discovery currently requires Linux")
    status_text = read_first("/proc/self/status")
    if status_text is None:
        fail("cannot read /proc/self/status")
    match = re.search(r"^Cpus_allowed_list:\s*(\S+)\s*$", status_text, re.MULTILINE)
    if not match:
        fail("/proc/self/status has no Cpus_allowed_list")
    allowed_text = match.group(1)
    allowed = expand_cpu_list(allowed_text)

    records: list[dict[str, int]] = []
    seen_cores: set[tuple[int, int]] = set()
    for line in run_text(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"]).splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 4:
            fail(f"unexpected lscpu row {line!r}")
        cpu, core, socket = (int(fields[index]) for index in range(3))
        online = fields[3].strip().upper()
        if online not in {"Y", "YES", "1"} or cpu not in allowed:
            continue
        key = (socket, core)
        if key in seen_cores:
            continue
        seen_cores.add(key)
        records.append({"cpu": cpu, "core": core, "socket": socket})
    records.sort(key=lambda record: (record["socket"], record["core"], record["cpu"]))
    if not records:
        fail("no allowed online physical core was discovered")
    capacity = min(len(records), args.requested_workers)
    selected = records[:capacity]
    cpu_list = ",".join(str(record["cpu"]) for record in selected)
    omp_places = ",".join("{" + str(record["cpu"]) + "}" for record in selected)
    worker_counts = [count for count in WORKER_COUNTS if count <= capacity]

    lscpu_human = run_text(["lscpu"])
    model_match = re.search(r"^Model name:\s*(.+?)\s*$", lscpu_human, re.MULTILINE)
    hash_specs = parse_assignments(args.hash_file, "hash-file")
    artifacts = {name: artifact_record(Path(path)) for name, path in hash_specs.items()}
    binary = Path(args.binary)
    artifacts["benchmark_binary"] = artifact_record(binary)
    runtime = find_openmp_runtime(binary)
    artifacts["openmp_runtime"] = artifact_record(runtime)
    if runtime.name.startswith("libgomp"):
        artifacts["libgomp"] = dict(artifacts["openmp_runtime"])

    git_root = Path(args.git_root).resolve(strict=True)
    git_revision = run_text(["git", "rev-parse", "HEAD"], cwd=str(git_root))
    git_status = run_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=str(git_root),
    )
    omp_environment = parse_assignments(args.openmp_env, "OpenMP environment")
    omp_environment["OMP_PLACES"] = omp_places
    omp_environment["OMP_NUM_THREADS"] = str(capacity)
    metadata = {
        "schema": SCHEMA,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": model_match.group(1) if model_match else "unavailable",
        "allowed_cpu_list_before_pin": allowed_text,
        "available_allowed_physical_cores": len(records),
        "requested_worker_capacity": args.requested_workers,
        "selected_worker_capacity": capacity,
        "benchmark_worker_counts": worker_counts,
        "selected_physical_cpus": selected,
        "selected_cpu_list": cpu_list,
        "omp_places": omp_places,
        "openmp_environment": omp_environment,
        "physical_core_selection": "lowest allowed CPU for each unique (socket,core)",
        "governor": read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "wsl_interop": "WSL_INTEROP" in os.environ,
        "provenance": {
            "git_revision": git_revision,
            "git_dirty": bool(git_status),
            "git_status_porcelain": git_status.splitlines(),
            "git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
            "compiler_invocation": args.compiler,
            "artifacts": artifacts,
        },
    }
    Path(args.metadata).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(cpu_list)
    print(omp_places)
    print(capacity)


def expected_variant_specs(capacity: int) -> list[tuple[str, str, int]]:
    return [("serial_ref", "SerialReference", 1)] + [
        (f"openmp_{workers}", "OpenMP", workers) for workers in WORKER_COUNTS if workers <= capacity
    ]


def finite_positive(value: Any, description: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} is not numeric",
    )
    converted = float(value)
    require(
        math.isfinite(converted) and converted > 0.0, f"{description} is not finite and positive"
    )
    return converted


def finite_nonnegative(value: Any, description: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} is not numeric",
    )
    converted = float(value)
    require(
        math.isfinite(converted) and converted >= 0.0,
        f"{description} is not finite and non-negative",
    )
    return converted


def integer(value: Any, description: str, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool), f"{description} is not an integer"
    )
    require(value >= minimum, f"{description} is below {minimum}")
    return value


def nearest_rank(values: list[float], probability: float) -> float:
    require(bool(values), "cannot summarize an empty sample set")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)]


def validate_provenance(host: dict[str, Any], verify_files: bool) -> None:
    provenance = host.get("provenance")
    require(isinstance(provenance, dict), "host provenance is absent")
    require(isinstance(provenance.get("git_dirty"), bool), "git_dirty is not Boolean")
    lines = provenance.get("git_status_porcelain")
    require(
        isinstance(lines, list) and all(isinstance(line, str) for line in lines),
        "git status listing is absent",
    )
    status_text = "\n".join(lines)
    require(
        hashlib.sha256(status_text.encode("utf-8")).hexdigest()
        == provenance.get("git_status_sha256"),
        "git status digest is inconsistent",
    )
    require(bool(lines) == provenance["git_dirty"], "git dirty flag is inconsistent with status")
    revision = provenance.get("git_revision")
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision),
        "git revision is not a full object ID",
    )
    artifacts = provenance.get("artifacts")
    require(isinstance(artifacts, dict), "provenance artifact map is absent")
    require(
        {"benchmark_binary", "openmp_runtime"}.issubset(artifacts),
        "binary/OpenMP runtime hashes are absent",
    )
    require(any(name.startswith("source_") for name in artifacts), "source hashes are absent")
    require(any(name.startswith("harness_") for name in artifacts), "harness hashes are absent")
    runtime_path = str(artifacts["openmp_runtime"].get("path", ""))
    if Path(runtime_path).name.startswith("libgomp"):
        require("libgomp" in artifacts, "linked libgomp hash is absent")
    for name, record in artifacts.items():
        require(isinstance(record, dict), f"artifact {name} is not an object")
        path = record.get("path")
        digest = record.get("sha256")
        require(isinstance(path, str) and path, f"artifact {name} path is absent")
        require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"artifact {name} SHA-256 is malformed",
        )
        if verify_files:
            require(Path(path).is_file(), f"artifact {name} no longer exists at validation")
            require(
                sha256_file(Path(path)) == digest,
                f"artifact {name} SHA-256 does not match its file",
            )


def as_csv_row(
    document: dict[str, Any],
    case: dict[str, Any],
    aggregate: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, str]:
    boolean = lambda value: "true" if value else "false"
    return {
        "schema": document["schema"],
        "valid": boolean(document["valid"]),
        "classification": document["classification"],
        "claim_eligible": boolean(document["eligible_for_project_level_2x_claim"]),
        "preset": document["preset"],
        "shape": case["shape"],
        "beam1_slices": str(case["beam1_slices"]),
        "beam2_slices": str(case["beam2_slices"]),
        "substeps": str(case["substeps"]),
        "encounters": str(case["encounters"]),
        "waves": str(case["waves"]),
        "intensity": case["intensity"],
        "iterations": str(case["integer_mix_iterations"]),
        "variant": aggregate["variant"],
        "execution": aggregate["execution"],
        "workers": str(aggregate["workers"]),
        "schedule": aggregate["schedule"],
        "timing_scope": document["timing_scope_id"],
        "warmups": str(document["warmups_per_variant_case"]),
        "repetition": str(sample["repetition"]),
        "order_index": str(sample["order_index"]),
        "seconds": repr(float(sample["seconds"])),
        "checksum": sample["checksum"],
        "oracle_checksum": case["oracle_checksum"],
        "deterministic_checksum_match": boolean(
            aggregate["deterministic_checksum_match_same_implementation_oracle"]
        ),
        "report_success": boolean(sample["report_success"]),
        "used_serial_fallback": boolean(sample["used_serial_fallback"]),
        "attempted_encounters": str(sample["attempted_encounters"]),
        "completed_encounters": str(sample["completed_encounters"]),
        "published_encounters": str(sample["published_encounters"]),
        "requested_workers": str(sample["requested_workers"]),
        "actual_workers": str(sample["actual_workers"]),
        "parallel_regions": str(sample["parallel_regions"]),
        "preflight_bytes": str(aggregate["preflight_bytes"]),
        "construction_seconds": repr(float(aggregate["construction_seconds"])),
        "median_seconds": repr(float(aggregate["median_seconds"])),
        "p95_seconds": repr(float(aggregate["p95_seconds"])),
        "min_seconds": repr(float(aggregate["min_seconds"])),
        "max_seconds": repr(float(aggregate["max_seconds"])),
        "exploratory_speedup_vs_serial_median": repr(
            float(aggregate["exploratory_speedup_vs_serial_median"])
        ),
    }


def compare_csv_row(actual: dict[str, str], expected: dict[str, str], key: str) -> None:
    require(set(actual) == set(CSV_FIELDS), f"CSV fields differ from schema at {key}")
    float_fields = {
        "seconds",
        "construction_seconds",
        "median_seconds",
        "p95_seconds",
        "min_seconds",
        "max_seconds",
        "exploratory_speedup_vs_serial_median",
    }
    for field in CSV_FIELDS:
        if field in float_fields:
            try:
                actual_value, expected_value = float(actual[field]), float(expected[field])
            except ValueError as exc:
                raise ValidationError(f"CSV {field} is not numeric at {key}") from exc
            require(math.isfinite(actual_value), f"CSV {field} is non-finite at {key}")
            require(actual_value == expected_value, f"CSV {field} differs from JSON at {key}")
        else:
            require(actual[field] == expected[field], f"CSV {field} differs from JSON at {key}")


def validate_pair(
    document: dict[str, Any],
    rows: list[dict[str, str]],
    repetitions: int,
    expected_valid: bool,
    verify_provenance_files: bool = True,
) -> None:
    require(document.get("schema") == SCHEMA, "unexpected JSON schema")
    require(document.get("valid") is expected_valid, "JSON valid state is unexpected")
    require(
        document.get("validation_status")
        == ("validated" if expected_valid else "unvalidated_staging"),
        "JSON validation status is unexpected",
    )
    if expected_valid:
        validation = document.get("validation")
        require(isinstance(validation, dict), "validation attestation is absent")
        require(
            validation.get("validator") == "modern_cpu_executor_benchmark_support.py",
            "validation attestation names an unexpected validator",
        )
        require(
            isinstance(validation.get("validator_sha256"), str)
            and SHA256_RE.fullmatch(validation["validator_sha256"]) is not None,
            "validator attestation SHA-256 is malformed",
        )
    else:
        require(
            "validation" not in document,
            "unvalidated staging document contains a validation attestation",
        )
    require(
        document.get("classification") == CLASSIFICATION,
        "classification is not exploratory_microkernel",
    )
    require(
        document.get("eligible_for_project_level_2x_claim") is False,
        "microbenchmark was incorrectly marked claim-eligible",
    )
    require(
        "same-implementation" in str(document.get("oracle_scope", "")),
        "same-implementation oracle limitation is absent",
    )
    require(
        document.get("timing_scope_id") == "executor_execute_only",
        "timing scope identifier is invalid",
    )
    integer(document.get("warmups_per_variant_case"), "warmups", 1)
    require(
        integer(document.get("repetitions_per_variant_case"), "JSON repetitions", 3) == repetitions,
        "JSON repetitions differ from validator request",
    )

    host, binding = document.get("host_metadata"), document.get("runtime_binding")
    require(
        isinstance(host, dict) and isinstance(binding, dict),
        "host/runtime binding metadata is absent",
    )
    require(host.get("schema") == SCHEMA, "host metadata schema is inconsistent")
    capacity = integer(host.get("selected_worker_capacity"), "worker capacity", 1)
    selected = host.get("selected_physical_cpus")
    require(
        isinstance(selected, list) and len(selected) == capacity,
        "selected physical CPU list has wrong length",
    )
    selected_cpus = {integer(record.get("cpu"), "selected CPU") for record in selected}
    require(len(selected_cpus) == capacity, "selected physical CPUs are not unique")
    expected_counts = [count for count in WORKER_COUNTS if count <= capacity]
    require(
        host.get("benchmark_worker_counts") == expected_counts,
        "benchmark worker counts do not match capacity",
    )
    omp_environment = host.get("openmp_environment")
    require(isinstance(omp_environment, dict), "OpenMP environment metadata is absent")
    for name in (
        "OMP_DYNAMIC",
        "OMP_THREAD_LIMIT",
        "OMP_NUM_THREADS",
        "OMP_MAX_ACTIVE_LEVELS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_WAIT_POLICY",
        "OMP_SCHEDULE",
    ):
        require(name in omp_environment, f"effective {name} metadata is absent")
    worker_cpus, worker_places = (
        binding.get("worker_cpus_at_probe"),
        binding.get("worker_places_at_probe"),
    )
    require(
        isinstance(worker_cpus, list)
        and len(worker_cpus) == capacity
        and set(worker_cpus) == selected_cpus,
        "runtime workers do not exactly match selected physical CPUs",
    )
    require(
        isinstance(worker_places, list)
        and len(worker_places) == capacity
        and len(set(worker_places)) == capacity,
        "runtime OpenMP places are not unique",
    )
    require(
        binding.get("omp_num_places") == capacity,
        "runtime OpenMP place count differs from capacity",
    )
    require(
        binding.get("omp_proc_bind") not in {"false", "unknown", None},
        "OpenMP processor binding is inactive",
    )
    validate_provenance(host, verify_provenance_files)
    provenance = host["provenance"]
    require(
        document.get("git_revision") == provenance["git_revision"],
        "top-level and provenance git revisions differ",
    )
    require(
        document.get("compiler_command") == provenance.get("compiler_invocation"),
        "top-level and provenance compiler invocations differ",
    )
    require(
        isinstance(document.get("compile_flags"), str) and bool(document["compile_flags"]),
        "compile flags are absent",
    )
    require(integer(document.get("openmp_macro"), "OpenMP macro", 1) > 0, "OpenMP macro is absent")
    if expected_valid:
        validator_artifact = host["provenance"]["artifacts"].get("harness_validator")
        require(
            isinstance(validator_artifact, dict)
            and validator_artifact.get("sha256") == document["validation"]["validator_sha256"],
            "validator attestation differs from harness provenance",
        )

    cases = document.get("cases")
    require(isinstance(cases, list) and bool(cases), "JSON contains no cases")
    expected_specs = expected_variant_specs(capacity)
    expected_names = {spec[0] for spec in expected_specs}
    expected_csv: dict[tuple[str, str, str, int], dict[str, str]] = {}
    seen_cases: set[tuple[str, str]] = set()
    for case_index, case in enumerate(cases):
        require(isinstance(case, dict), f"case {case_index} is not an object")
        case_key = (str(case.get("shape")), str(case.get("intensity")))
        require(case_key not in seen_cases, f"duplicate case {case_key}")
        seen_cases.add(case_key)
        encounters = integer(case.get("encounters"), f"encounters {case_key}", 1)
        for field in (
            "waves",
            "beam1_slices",
            "beam2_slices",
            "substeps",
            "integer_mix_iterations",
        ):
            integer(case.get(field), f"{field} {case_key}", 1)
        oracle = case.get("oracle_checksum")
        require(
            isinstance(oracle, str) and HEX64_RE.fullmatch(oracle) is not None,
            f"oracle checksum is malformed for {case_key}",
        )
        aggregates = case.get("aggregates")
        require(isinstance(aggregates, list), f"aggregates absent for {case_key}")
        names = [aggregate.get("variant") for aggregate in aggregates]
        require(len(names) == len(set(names)), f"duplicate variants for {case_key}")
        require(set(names) == expected_names, f"variant set is incomplete for {case_key}")
        by_name = {aggregate["variant"]: aggregate for aggregate in aggregates}
        order_by_repetition = {repetition: set() for repetition in range(repetitions)}
        for name, execution, workers in expected_specs:
            aggregate = by_name[name]
            require(
                aggregate.get("execution") == execution,
                f"execution kind is wrong for {case_key}/{name}",
            )
            require(
                aggregate.get("workers") == workers, f"worker count is wrong for {case_key}/{name}"
            )
            require(
                aggregate.get("schedule") == "Static", f"schedule is wrong for {case_key}/{name}"
            )
            integer(aggregate.get("preflight_bytes"), f"preflight bytes {case_key}/{name}", 1)
            finite_nonnegative(
                aggregate.get("construction_seconds"), f"construction time {case_key}/{name}"
            )
            checksum = aggregate.get("checksum")
            require(
                isinstance(checksum, str) and HEX64_RE.fullmatch(checksum) is not None,
                f"aggregate checksum is malformed for {case_key}/{name}",
            )
            require(
                checksum == oracle,
                f"aggregate checksum differs from same-implementation oracle for {case_key}/{name}",
            )
            require(
                aggregate.get("deterministic_checksum_match_same_implementation_oracle") is True,
                f"deterministic checksum self-consistency failed for {case_key}/{name}",
            )
            samples = aggregate.get("samples")
            require(
                isinstance(samples, list) and len(samples) == repetitions,
                f"sample count is wrong for {case_key}/{name}",
            )
            repetition_ids: set[int] = set()
            times: list[float] = []
            for sample in samples:
                repetition = integer(sample.get("repetition"), f"repetition {case_key}/{name}")
                order_index = integer(sample.get("order_index"), f"order index {case_key}/{name}")
                require(
                    repetition < repetitions and order_index < len(expected_specs),
                    f"sample identity is out of range for {case_key}/{name}",
                )
                require(
                    repetition not in repetition_ids, f"duplicate repetition for {case_key}/{name}"
                )
                repetition_ids.add(repetition)
                order_by_repetition[repetition].add(order_index)
                times.append(
                    finite_positive(
                        sample.get("seconds"), f"sample time {case_key}/{name}/{repetition}"
                    )
                )
                require(
                    sample.get("checksum") == checksum,
                    f"sample checksum differs for {case_key}/{name}/{repetition}",
                )
                require(
                    sample.get("report_success") is True
                    and sample.get("used_serial_fallback") is False,
                    f"execution report failed for {case_key}/{name}/{repetition}",
                )
                require(
                    sample.get("attempted_encounters") == encounters
                    and sample.get("completed_encounters") == encounters
                    and sample.get("published_encounters") == 0,
                    f"encounter report is wrong for {case_key}/{name}/{repetition}",
                )
                require(
                    sample.get("requested_workers") == workers
                    and sample.get("actual_workers") == workers,
                    f"worker report differs for {case_key}/{name}/{repetition}",
                )
                require(
                    sample.get("parallel_regions") == (1 if execution == "OpenMP" else 0),
                    f"parallel-region report differs for {case_key}/{name}/{repetition}",
                )
                key = (case_key[0], case_key[1], name, repetition)
                require(key not in expected_csv, f"duplicate logical sample {key}")
                expected_csv[key] = as_csv_row(document, case, aggregate, sample)
            require(
                repetition_ids == set(range(repetitions)),
                f"repetitions are not exhaustive for {case_key}/{name}",
            )
            calculated = {
                "median_seconds": statistics.median(times),
                "p95_seconds": nearest_rank(times, 0.95),
                "min_seconds": min(times),
                "max_seconds": max(times),
            }
            for field, value in calculated.items():
                recorded = finite_positive(aggregate.get(field), f"{field} {case_key}/{name}")
                require(
                    recorded == value, f"{field} was not recomputed correctly for {case_key}/{name}"
                )
        expected_orders = set(range(len(expected_specs)))
        for repetition, order_indexes in order_by_repetition.items():
            require(
                order_indexes == expected_orders,
                f"interleaving is not a permutation for {case_key}/{repetition}",
            )
        serial_median = float(by_name["serial_ref"]["median_seconds"])
        for name in expected_names:
            aggregate = by_name[name]
            expected_speedup = serial_median / float(aggregate["median_seconds"])
            recorded = finite_positive(
                aggregate.get("exploratory_speedup_vs_serial_median"), f"speedup {case_key}/{name}"
            )
            require(
                recorded == expected_speedup,
                f"speedup was not recomputed correctly for {case_key}/{name}",
            )

    require(
        len(rows) == len(expected_csv), f"CSV has {len(rows)} samples; JSON has {len(expected_csv)}"
    )
    seen_rows: set[tuple[str, str, str, int]] = set()
    for row_index, row in enumerate(rows):
        require(set(row) == set(CSV_FIELDS), f"CSV row {row_index} is malformed")
        try:
            key = (row["shape"], row["intensity"], row["variant"], int(row["repetition"]))
        except (KeyError, ValueError) as exc:
            raise ValidationError(f"CSV row {row_index} has an invalid key") from exc
        require(
            key in expected_csv and key not in seen_rows,
            f"CSV row {row_index} is absent from or duplicated against JSON",
        )
        seen_rows.add(key)
        compare_csv_row(row, expected_csv[key], str(key))
    require(seen_rows == set(expected_csv), "CSV is not an exact JSON projection")


def read_pair(json_path: Path, csv_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON output: {exc}") from exc
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(
                tuple(reader.fieldnames or ()) == CSV_FIELDS,
                "CSV header does not exactly match schema",
            )
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"cannot read CSV output: {exc}") from exc
    return document, rows


def validate_paths(
    json_path: Path,
    csv_path: Path,
    repetitions: int,
    expected_valid: bool,
    verify_provenance_files: bool = True,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    require(json_path.is_file() and csv_path.is_file(), "both JSON and CSV result files must exist")
    document, rows = read_pair(json_path, csv_path)
    validate_pair(document, rows, repetitions, expected_valid, verify_provenance_files)
    return document, rows


def write_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def promote_pair(
    document: dict[str, Any],
    rows: list[dict[str, str]],
    repetitions: int,
    json_path: Path,
    csv_path: Path,
) -> None:
    require(
        json_path.parent == csv_path.parent and json_path.parent.is_dir(),
        "promotion paths must share an existing directory",
    )
    require(
        not json_path.exists() and not csv_path.exists(), "refusing to overwrite promoted output"
    )
    promoted = copy.deepcopy(document)
    promoted["valid"] = True
    promoted["validation_status"] = "validated"
    promoted["validation"] = {
        "validator": "modern_cpu_executor_benchmark_support.py",
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": (
            "finite positive samples; report/checksum invariants; variant uniqueness; "
            "standard median; nearest-rank p95; min/max; speedups; exact CSV "
            "projection; provenance file hashes"
        ),
    }
    promoted_rows = copy.deepcopy(rows)
    for row in promoted_rows:
        row["valid"] = "true"
    validate_pair(promoted, promoted_rows, repetitions, True, True)
    json_temp, csv_temp = (
        json_path.with_name(json_path.name + ".tmp"),
        csv_path.with_name(csv_path.name + ".tmp"),
    )
    require(
        not json_temp.exists() and not csv_temp.exists(),
        "refusing to overwrite promotion temporary files",
    )
    try:
        write_json(json_temp, promoted)
        write_csv(csv_temp, promoted_rows)
        validate_paths(json_temp, csv_temp, repetitions, True, True)
        csv_temp.replace(csv_path)
        try:
            json_temp.replace(json_path)  # valid=true JSON is the final marker.
        except BaseException:
            csv_path.unlink(missing_ok=True)
            raise
    finally:
        json_temp.unlink(missing_ok=True)
        csv_temp.unlink(missing_ok=True)


def command_validate(args: argparse.Namespace) -> None:
    expected_valid = args.expected_valid == "true"
    try:
        document, rows = validate_paths(
            Path(args.json),
            Path(args.csv),
            args.repetitions,
            expected_valid,
            not args.skip_provenance_file_check,
        )
        if args.promote_json or args.promote_csv:
            require(not expected_valid, "only unvalidated staging files may be promoted")
            require(
                bool(args.promote_json) and bool(args.promote_csv),
                "both promoted paths are required",
            )
            promote_pair(
                document, rows, args.repetitions, Path(args.promote_json), Path(args.promote_csv)
            )
    except ValidationError as exc:
        fail(str(exc))
    print(
        f"validated {len(document['cases'])} cases and {len(rows)} samples; "
        f"valid={str(expected_valid).lower()}"
    )
    if args.promote_json:
        print(f"promoted validated JSON: {args.promote_json}")
        print(f"promoted validated CSV:  {args.promote_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    affinity = subparsers.add_parser("affinity")
    affinity.add_argument("--requested-workers", type=int, required=True)
    affinity.add_argument("--metadata", required=True)
    affinity.add_argument("--binary", required=True)
    affinity.add_argument("--git-root", required=True)
    affinity.add_argument("--compiler", required=True)
    affinity.add_argument("--hash-file", action="append", default=[])
    affinity.add_argument("--openmp-env", action="append", default=[])
    affinity.set_defaults(function=command_affinity)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--json", required=True)
    validate.add_argument("--csv", required=True)
    validate.add_argument("--repetitions", type=int, required=True)
    validate.add_argument("--expected-valid", choices=("true", "false"), required=True)
    validate.add_argument("--promote-json")
    validate.add_argument("--promote-csv")
    validate.add_argument("--skip-provenance-file-check", action="store_true")
    validate.set_defaults(function=command_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    numeric = getattr(args, "requested_workers", getattr(args, "repetitions", 1))
    if numeric < 1:
        fail("numeric arguments must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
