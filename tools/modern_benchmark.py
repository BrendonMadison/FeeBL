#!/usr/bin/env python3
"""Strict paired benchmark and run-manifest harness for FeeBL modern engines.

The harness does not implement or enable a physics engine.  It materializes an
independent card for each fresh CLI process, preserves every invalid sample,
and refuses to turn timing alone into a qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - FeeBL production hosts are POSIX.
    resource = None  # type: ignore[assignment]


RUN_MANIFEST_SCHEMA = "feebl-modern-run-manifest-v1"
CAMPAIGN_SCHEMA = "feebl-modern-benchmark-v1"
REQUIRED_CAPABILITIES = (
    "beam_soa",
    "cell_csr",
    "stable_ids",
    "counter_rng",
    "encounter_dag",
)
REQUIRED_WORK_COUNTERS = (
    "encounters",
    "primary_particles",
    "physics_events",
)


@dataclass(frozen=True)
class Variant:
    label: str
    fragment: str
    candidate: bool


@dataclass(frozen=True)
class ManifestLimits:
    max_host_stages: int | None
    max_transfers: int | None
    max_allocations: int | None
    max_synchronizations: int | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fragment(fragment: str) -> str:
    if "\x00" in fragment:
        raise ValueError("card fragment contains a NUL byte")
    fragment = fragment.strip()
    if fragment and not fragment.endswith(";"):
        fragment += ";"
    return fragment


def materialize_card(source: bytes, parameter_block: str, fragment: str) -> bytes:
    """Insert *fragment* in exactly one parameter block without mutating source."""
    fragment = normalize_fragment(fragment)
    if not fragment:
        return source
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source card is not UTF-8") from error
    newline = "\r\n" if "\r\n" in text else "\n"
    pattern = re.compile(
        r"(?m)^([ \t]*\$PARAMETERS::[ \t]*"
        + re.escape(parameter_block)
        + r"[ \t]*(?:\r?\n[ \t]*)?\{)"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one $PARAMETERS:: {parameter_block} block; " f"found {len(matches)}"
        )
    rendered = pattern.sub(
        lambda match: match.group(1) + newline + fragment,
        text,
        count=1,
    )
    return rendered.encode("utf-8")


def run_text(command: Sequence[str], timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def dynamic_dependencies(executable: Path) -> list[Path]:
    if not sys.platform.startswith("linux") or shutil.which("ldd") is None:
        return []
    text = run_text(["ldd", str(executable)])
    paths: set[Path] = set()
    for line in text.splitlines():
        match = re.search(r"=>\s+(/[^ ]+)", line)
        if match is None:
            match = re.match(r"\s*(/[^ ]+)", line)
        if match:
            path = Path(match.group(1))
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths, key=str)


def shebang_executable(path: Path) -> Path | None:
    try:
        first = path.open("rb").readline(4096).decode("utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not first.startswith("#!"):
        return None
    words = shlex.split(first[2:].strip())
    if not words:
        return None
    if Path(words[0]).name == "env" and len(words) > 1:
        resolved = shutil.which(words[1])
    else:
        resolved = words[0]
    if not resolved:
        return None
    candidate = Path(resolved)
    return candidate.resolve() if candidate.is_file() else None


def build_hash_inventory(
    binary: Path, card: Path, dependencies: Sequence[Path], affinity: str | None
) -> tuple[list[dict[str, Any]], str]:
    records: dict[str, dict[str, Any]] = {}

    def add(role: str, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"{role} is missing or is not a file: {resolved}")
        key = str(resolved)
        record = records.setdefault(
            key,
            {
                "path": key,
                "roles": [],
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            },
        )
        if role not in record["roles"]:
            record["roles"].append(role)

    add("benchmark_harness", Path(__file__))
    add("python_executable", Path(sys.executable))
    add("simulation_executable", binary)
    add("source_card", card)
    interpreter = shebang_executable(binary)
    if interpreter is not None:
        add("simulation_interpreter", interpreter)
    for dependency in dependencies:
        add("explicit_dependency", dependency)
    dependency_roots = [binary, Path(sys.executable)]
    if interpreter is not None:
        dependency_roots.append(interpreter)
    for root in dependency_roots:
        for dependency in dynamic_dependencies(root):
            add("dynamic_dependency", dependency)
    if affinity:
        taskset = shutil.which("taskset")
        if not taskset:
            raise ValueError("--affinity requires the taskset executable")
        add("affinity_launcher", Path(taskset))
    ordered = sorted(records.values(), key=lambda item: item["path"])
    for record in ordered:
        record["roles"].sort()
    digest = sha256_bytes(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return ordered, digest


def host_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def gpu_identity() -> dict[str, Any]:
    query = "index,uuid,name,driver_version,pci.bus_id,compute_cap," "memory.total"
    output = run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    return {
        "query": query,
        "raw": output,
        "sha256": sha256_bytes(output.encode("utf-8")),
    }


def parse_environment(assignments: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment assignment: {assignment!r}")
        result[name] = value
    return result


def execution_environment(overrides: dict[str, str]) -> dict[str, str]:
    allowed = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OMP_PLACES",
        "OMP_PROC_BIND",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update(overrides)
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    return environment


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_candidate_manifest(
    manifest: Any,
    *,
    expected_engine: str,
    executable_sha256: str,
    card_sha256: str,
    inventory_sha256: str,
    output_sha256: str | None,
    required_capabilities: Sequence[str],
    limits: ManifestLimits,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["candidate run manifest must be a JSON object"]
    required = {
        "schema",
        "requested_engine",
        "actual_engine",
        "executable_sha256",
        "input_card_sha256",
        "input_inventory_sha256",
        "output_sha256",
        "capabilities",
        "work_counters",
        "fallback",
        "fallback_reason",
        "host_stage_count",
        "transfer_count",
        "transfer_bytes",
        "allocation_count",
        "synchronization_count",
        "completion_marker",
        "nan_detected",
    }
    missing = sorted(required - set(manifest))
    if missing:
        errors.append("candidate manifest missing fields: " + ", ".join(missing))
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        errors.append(f"candidate manifest schema must be {RUN_MANIFEST_SCHEMA}")
    if manifest.get("requested_engine") != expected_engine:
        errors.append("candidate requested engine mismatch")
    if manifest.get("actual_engine") != expected_engine:
        errors.append("candidate actual engine/backend mismatch")
    expected_hashes = {
        "executable_sha256": executable_sha256,
        "input_card_sha256": card_sha256,
        "input_inventory_sha256": inventory_sha256,
        "output_sha256": output_sha256,
    }
    for name, expected in expected_hashes.items():
        if manifest.get(name) != expected:
            errors.append(f"candidate manifest {name} mismatch")
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        errors.append("candidate capabilities must be an object")
    else:
        for name, value in capabilities.items():
            if not isinstance(value, bool):
                errors.append(f"candidate capability {name} must be boolean")
        for name in required_capabilities:
            if capabilities.get(name) is not True:
                errors.append(f"required candidate capability is not true: {name}")
    counters = manifest.get("work_counters")
    if not isinstance(counters, dict):
        errors.append("candidate work_counters must be an object")
    else:
        for name, value in counters.items():
            if not _integer(value) or value < 0:
                errors.append(f"candidate work counter {name} must be non-negative integer")
        for name in REQUIRED_WORK_COUNTERS:
            if not _integer(counters.get(name)) or counters.get(name, 0) <= 0:
                errors.append(f"candidate work counter must be positive: {name}")
    if manifest.get("fallback") is not False:
        errors.append("candidate reported fallback")
    if manifest.get("fallback_reason") not in ("", None):
        errors.append("candidate fallback_reason is not empty")
    numeric_markers = (
        "host_stage_count",
        "transfer_count",
        "transfer_bytes",
        "allocation_count",
        "synchronization_count",
    )
    for name in numeric_markers:
        value = manifest.get(name)
        if not _integer(value) or value < 0:
            errors.append(f"candidate {name} must be a non-negative integer")
    thresholds = (
        ("host_stage_count", limits.max_host_stages),
        ("transfer_count", limits.max_transfers),
        ("allocation_count", limits.max_allocations),
        ("synchronization_count", limits.max_synchronizations),
    )
    for name, maximum in thresholds:
        value = manifest.get(name)
        if maximum is not None and _integer(value) and value > maximum:
            errors.append(f"candidate {name}={value} exceeds limit {maximum}")
    if manifest.get("completion_marker") != "complete":
        errors.append("candidate completion marker is not complete")
    if manifest.get("nan_detected") is not False:
        errors.append("candidate reported a NaN marker")
    return errors


def proc_tree_pids(root: int) -> set[int]:
    pending = [root]
    found: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(word) for word in children.read_text().split())
        except (OSError, ValueError):
            pass
    return found


def proc_tree_rss_kib(root: int) -> int:
    total = 0
    for pid in proc_tree_pids(root):
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            continue
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
        if match:
            total += int(match.group(1))
    return total


def monitor_rss(
    pid: int, stop: threading.Event, interval: float, destination: dict[str, int]
) -> None:
    peak = 0
    while not stop.is_set():
        peak = max(peak, proc_tree_rss_kib(pid))
        stop.wait(interval)
    destination["peak_rss_kib"] = max(peak, proc_tree_rss_kib(pid))


def child_affinity(pid: int) -> str | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    match = re.search(r"^Cpus_allowed_list:\s*(.+)$", status, re.MULTILINE)
    return match.group(1).strip() if match else None


def usage_snapshot() -> tuple[float, float]:
    if resource is None:
        return 0.0, 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime, usage.ru_stime


def terminate_process(process: subprocess.Popen[Any]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover
            process.terminate()
        process.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except OSError:
            pass
        process.wait()


def execute_sample(
    *,
    binary: Path,
    rendered_card: bytes,
    variant: Variant,
    accelerator: str,
    parameters: str,
    result_name: str,
    expected_outputs: Sequence[str],
    run_directory: Path,
    block: int,
    warmup: bool,
    order_in_block: int,
    timeout: float,
    rss_interval: float,
    affinity: str | None,
    base_environment: dict[str, str],
    inventory_sha256: str,
    binary_sha256: str,
    expected_candidate_engine: str,
    required_capabilities: Sequence[str],
    limits: ManifestLimits,
) -> dict[str, Any]:
    run_directory.mkdir(parents=True, exist_ok=False)
    card_path = run_directory / "acc.dat"
    card_path.write_bytes(rendered_card)
    card_sha256 = sha256_bytes(rendered_card)
    for relative in expected_outputs:
        (run_directory / relative).parent.mkdir(parents=True, exist_ok=True)
    stdout_path = run_directory / "stdout.log"
    stderr_path = run_directory / "stderr.log"
    candidate_manifest_path = run_directory / "candidate_run_manifest.json"
    command = [str(binary), accelerator, parameters, result_name]
    if affinity:
        command = [str(Path(shutil.which("taskset") or "taskset")), "-c", affinity, *command]
    environment = dict(base_environment)
    environment["FEEBL_INPUT_INVENTORY_SHA256"] = inventory_sha256
    environment["FEEBL_MATERIALIZED_CARD_SHA256"] = card_sha256
    if variant.candidate:
        environment["FEEBL_RUN_MANIFEST"] = str(candidate_manifest_path)
        environment["FEEBL_EXPECTED_ENGINE"] = expected_candidate_engine
    else:
        environment.pop("FEEBL_RUN_MANIFEST", None)
        environment.pop("FEEBL_EXPECTED_ENGINE", None)

    errors: list[str] = []
    timed_out = False
    returncode: int | None = None
    peak_rss = 0
    affinity_observed: str | None = None
    before_user, before_system = usage_snapshot()
    start_ns = time.perf_counter_ns()
    process: subprocess.Popen[Any] | None = None
    stop = threading.Event()
    monitor: threading.Thread | None = None
    monitoring: dict[str, int] = {}
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=run_directory,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            affinity_observed = child_affinity(process.pid)
            monitor = threading.Thread(
                target=monitor_rss,
                args=(process.pid, stop, rss_interval, monitoring),
                daemon=True,
            )
            monitor.start()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process(process)
                returncode = process.returncode
    except OSError as error:
        errors.append(f"process launch failed: {error}")
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=max(1.0, 2.0 * rss_interval))
        peak_rss = int(monitoring.get("peak_rss_kib", 0))
    wall_seconds = (time.perf_counter_ns() - start_ns) / 1.0e9
    after_user, after_system = usage_snapshot()
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    if timed_out:
        errors.append(f"process timed out after {timeout:g} seconds")
    if returncode != 0:
        errors.append(f"process return code is {returncode}")

    infrastructure = {
        "acc.dat",
        "stdout.log",
        "stderr.log",
        "candidate_run_manifest.json",
    }
    actual_files = sorted(
        str(path.relative_to(run_directory))
        for path in run_directory.rglob("*")
        if (path.is_file() or path.is_symlink())
        and str(path.relative_to(run_directory)) not in infrastructure
    )
    expected_set = set(expected_outputs)
    missing_outputs = sorted(expected_set - set(actual_files))
    extra_outputs = sorted(set(actual_files) - expected_set)
    if missing_outputs:
        errors.append("missing output files: " + ", ".join(missing_outputs))
    if extra_outputs:
        errors.append("extra output files: " + ", ".join(extra_outputs))
    output_hashes = {
        name: sha256_file(run_directory / name)
        for name in expected_outputs
        if (run_directory / name).is_file()
    }

    manifest_data: Any = None
    manifest_hash: str | None = None
    if candidate_manifest_path.exists():
        manifest_hash = sha256_file(candidate_manifest_path)
        try:
            manifest_data = json.loads(candidate_manifest_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"candidate run manifest is not valid JSON: {error}")
    if variant.candidate:
        if not candidate_manifest_path.exists():
            errors.append("candidate run manifest is missing")
        elif manifest_data is not None:
            errors.extend(
                validate_candidate_manifest(
                    manifest_data,
                    expected_engine=expected_candidate_engine,
                    executable_sha256=binary_sha256,
                    card_sha256=card_sha256,
                    inventory_sha256=inventory_sha256,
                    output_sha256=output_hashes.get(result_name),
                    required_capabilities=required_capabilities,
                    limits=limits,
                )
            )

    return {
        "schema": "feebl-modern-benchmark-sample-v1",
        "variant": variant.label,
        "candidate": variant.candidate,
        "block": block,
        "warmup": warmup,
        "order_in_block": order_in_block,
        "command": command,
        "command_display": shlex.join(command),
        "returncode": returncode,
        "signal": signal_number,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "user_seconds": max(0.0, after_user - before_user),
        "system_seconds": max(0.0, after_system - before_system),
        "peak_rss_kib": peak_rss,
        "child_cpu_affinity": affinity_observed,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "input_card_sha256": card_sha256,
        "expected_outputs": list(expected_outputs),
        "observed_output_files": actual_files,
        "missing_outputs": missing_outputs,
        "extra_outputs": extra_outputs,
        "output_sha256": output_hashes,
        "candidate_manifest_present": candidate_manifest_path.exists(),
        "candidate_manifest_sha256": manifest_hash,
        "candidate_manifest": manifest_data,
        "run_directory": str(run_directory),
        "errors": errors,
        "valid": not errors,
    }


def persist_sample(sample: dict[str, Any], jsonl: Path) -> None:
    sample["valid"] = not sample["errors"]
    run_directory = Path(sample["run_directory"])
    (run_directory / "sample.json").write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with jsonl.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, sort_keys=True) + "\n")


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")
    index = min(len(sorted_values) - 1, max(0, int(math.floor(probability * len(sorted_values)))))
    return sorted_values[index]


def block_bootstrap_interval(
    log_speedups: Sequence[float], *, seed: int, draws: int
) -> tuple[float, float]:
    if not log_speedups:
        raise ValueError("block bootstrap requires paired blocks")
    rng = random.Random(seed)
    count = len(log_speedups)
    estimates = sorted(
        math.exp(statistics.median([log_speedups[rng.randrange(count)] for _ in range(count)]))
        for _ in range(draws)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def analyze(
    samples: Sequence[dict[str, Any]], *, baseline: str, candidate: str, seed: int, draws: int
) -> dict[str, Any]:
    by_block: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        if sample["warmup"]:
            continue
        by_block.setdefault(sample["block"], {})[sample["variant"]] = sample
    log_speedups: list[float] = []
    valid_blocks: list[int] = []
    for block, variants in sorted(by_block.items()):
        reference = variants.get(baseline)
        contender = variants.get(candidate)
        if not reference or not contender or not reference["valid"] or not contender["valid"]:
            continue
        if reference["wall_seconds"] <= 0.0 or contender["wall_seconds"] <= 0.0:
            continue
        valid_blocks.append(block)
        log_speedups.append(
            math.log(reference["wall_seconds"]) - math.log(contender["wall_seconds"])
        )
    if log_speedups:
        estimate = math.exp(statistics.median(log_speedups))
        low, high = block_bootstrap_interval(log_speedups, seed=seed, draws=draws)
    else:
        estimate = low = high = None
    return {
        "schema": "feebl-modern-benchmark-analysis-v1",
        "label": "EXPLORATORY_TIMING_ONLY",
        "qualification_pass": None,
        "simultaneous_physics_and_performance_pass": None,
        "physics_validation_performed": False,
        "warning": (
            "Timing evidence is exploratory. This harness does not fabricate "
            "a simultaneous physics-accuracy and performance pass."
        ),
        "estimator": "exp(median(log(T_baseline/T_candidate)))",
        "bootstrap": "deterministic paired-block bootstrap",
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "valid_paired_blocks": valid_blocks,
        "paired_samples": len(log_speedups),
        "paired_log_median_speedup": estimate,
        "paired_log_median_speedup_ci95_low": low,
        "paired_log_median_speedup_ci95_high": high,
        "invalid_sample_count": sum(not sample["valid"] for sample in samples),
    }


def safe_name(value: str, description: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{description} must match [A-Za-z0-9_.-]+")
    return value


def safe_relative(value: str, description: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{description} must be a safe relative path")
    return str(path)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--baseline-fragment", default="")
    parser.add_argument("--candidate-fragment", required=True)
    parser.add_argument("--baseline-label", default="legacy")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--expected-candidate-engine", required=True)
    parser.add_argument("--result-name", default="result.dat")
    parser.add_argument("--expected-output", action="append", default=[])
    parser.add_argument("--dependency", action="append", type=Path, default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--rss-interval", type=float, default=0.02)
    parser.add_argument("--affinity")
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--require-capability", action="append", default=[])
    parser.add_argument("--max-host-stages", type=int)
    parser.add_argument("--max-transfers", type=int)
    parser.add_argument("--max-allocations", type=int)
    parser.add_argument("--max-synchronizations", type=int)
    parser.add_argument("--allow-paired-output-difference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one measured paired block and no warmups",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    try:
        baseline_label = safe_name(args.baseline_label, "baseline label")
        candidate_label = safe_name(args.candidate_label, "candidate label")
        expected_engine = safe_name(args.expected_candidate_engine, "expected candidate engine")
        accelerator = safe_name(args.accelerator, "accelerator")
        parameters = safe_name(args.parameters, "parameters")
        result_name = safe_relative(args.result_name, "result name")
        expected_outputs = [
            result_name,
            *(safe_relative(item, "expected output") for item in args.expected_output),
        ]
        if len(expected_outputs) != len(set(expected_outputs)):
            raise ValueError("expected output names must be unique")
        if baseline_label == candidate_label:
            raise ValueError("baseline and candidate labels must differ")
        if args.repetitions < 1 or args.warmups < 0:
            raise ValueError("repetitions must be >=1 and warmups must be >=0")
        if args.timeout <= 0.0 or args.rss_interval <= 0.0:
            raise ValueError("timeout and RSS interval must be positive")
        if args.bootstrap_draws < 100:
            raise ValueError("bootstrap draws must be at least 100")
        for name in (
            "max_host_stages",
            "max_transfers",
            "max_allocations",
            "max_synchronizations",
        ):
            value = getattr(args, name)
            if value is not None and value < 0:
                raise ValueError(name.replace("_", " ") + " must be non-negative")
        overrides = parse_environment(args.env)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    binary = args.binary.resolve()
    source_card = args.card.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"binary is missing or not executable: {binary}")
    if not source_card.is_file():
        raise SystemExit(f"source card is missing: {source_card}")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")

    source_card_before = source_card.read_bytes()
    source_card_sha256 = sha256_bytes(source_card_before)
    variants = (
        Variant(baseline_label, normalize_fragment(args.baseline_fragment), False),
        Variant(candidate_label, normalize_fragment(args.candidate_fragment), True),
    )
    try:
        rendered_cards = {
            variant.label: materialize_card(source_card_before, parameters, variant.fragment)
            for variant in variants
        }
        inventory, inventory_sha256 = build_hash_inventory(
            binary, source_card, args.dependency, args.affinity
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    required_capabilities = sorted(
        set(
            [
                *REQUIRED_CAPABILITIES,
                *args.require_capability,
            ]
        )
    )
    repetitions = 1 if args.smoke else args.repetitions
    warmups = 0 if args.smoke else args.warmups
    plan: list[dict[str, Any]] = []
    total_blocks = warmups + repetitions
    for sequence in range(total_blocks):
        order = variants if sequence % 2 == 0 else tuple(reversed(variants))
        plan.append(
            {
                "sequence": sequence,
                "block": sequence if sequence < warmups else sequence - warmups,
                "warmup": sequence < warmups,
                "order": [variant.label for variant in order],
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv if argv is None else [str(Path(__file__)), *argv],
        "status": "dry_run" if args.dry_run else "running",
        "smoke": args.smoke,
        "binary": str(binary),
        "source_card": str(source_card),
        "source_card_sha256": source_card_sha256,
        "materialized_card_sha256": {
            label: sha256_bytes(card) for label, card in rendered_cards.items()
        },
        "fragments_sha256": {
            variant.label: sha256_bytes(variant.fragment.encode("utf-8")) for variant in variants
        },
        "hash_inventory": inventory,
        "hash_inventory_sha256": inventory_sha256,
        "host_cpu_affinity": host_affinity(),
        "requested_affinity": args.affinity,
        "gpu_identity": gpu_identity(),
        "execution_environment": execution_environment(overrides),
        "expected_candidate_engine": expected_engine,
        "required_capabilities": required_capabilities,
        "expected_outputs": expected_outputs,
        "order_policy": "cyclic AB/BA paired blocks",
        "plan": plan,
        "analysis_label": "EXPLORATORY_TIMING_ONLY",
    }
    campaign_path = output / "campaign_manifest.json"
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        analysis = analyze(
            [],
            baseline=baseline_label,
            candidate=candidate_label,
            seed=args.bootstrap_seed,
            draws=args.bootstrap_draws,
        )
        (output / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    base_environment = execution_environment(overrides)
    limits = ManifestLimits(
        args.max_host_stages,
        args.max_transfers,
        args.max_allocations,
        args.max_synchronizations,
    )
    binary_sha256 = sha256_file(binary)
    samples: list[dict[str, Any]] = []
    jsonl = output / "samples.jsonl"
    for block_plan in plan:
        block_samples: list[dict[str, Any]] = []
        for order_in_block, label in enumerate(block_plan["order"]):
            variant = next(item for item in variants if item.label == label)
            kind = "warmup" if block_plan["warmup"] else "run"
            run_directory = (
                output
                / "runs"
                / (f"{kind}-{block_plan['block']:03d}-{order_in_block}-" f"{variant.label}")
            )
            sample = execute_sample(
                binary=binary,
                rendered_card=rendered_cards[variant.label],
                variant=variant,
                accelerator=accelerator,
                parameters=parameters,
                result_name=result_name,
                expected_outputs=expected_outputs,
                run_directory=run_directory,
                block=block_plan["block"],
                warmup=block_plan["warmup"],
                order_in_block=order_in_block,
                timeout=args.timeout,
                rss_interval=args.rss_interval,
                affinity=args.affinity,
                base_environment=base_environment,
                inventory_sha256=inventory_sha256,
                binary_sha256=binary_sha256,
                expected_candidate_engine=expected_engine,
                required_capabilities=required_capabilities,
                limits=limits,
            )
            block_samples.append(sample)

        reference = next(item for item in block_samples if not item["candidate"])
        contender = next(item for item in block_samples if item["candidate"])
        if (
            not args.allow_paired_output_difference
            and reference["output_sha256"] != contender["output_sha256"]
        ):
            contender["errors"].append("paired output hash mismatch")
        for sample in block_samples:
            persist_sample(sample, jsonl)
            samples.append(sample)
            state = "VALID" if sample["valid"] else "INVALID"
            print(
                f"block={sample['block']:03d} order={sample['order_in_block']} "
                f"{sample['variant']:16s} {sample['wall_seconds']:9.4f}s {state}",
                flush=True,
            )

    analysis = analyze(
        samples,
        baseline=baseline_label,
        candidate=candidate_label,
        seed=args.bootstrap_seed,
        draws=args.bootstrap_draws,
    )
    analysis_path = output / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_unchanged = sha256_file(source_card) == source_card_sha256
    invalid_samples = [sample for sample in samples if not sample["valid"]]
    campaign["source_card_unchanged"] = source_unchanged
    campaign["sample_count"] = len(samples)
    campaign["invalid_sample_count"] = len(invalid_samples)
    campaign["valid_paired_blocks"] = analysis["paired_samples"]
    campaign["samples_jsonl_sha256"] = sha256_file(jsonl)
    campaign["analysis_sha256"] = sha256_file(analysis_path)
    campaign["status"] = (
        "complete"
        if source_unchanged and not invalid_samples and analysis["paired_samples"] == repetitions
        else "failed"
    )
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if campaign["status"] != "complete":
        print(
            f"campaign failed: invalid_samples={len(invalid_samples)} "
            f"valid_pairs={analysis['paired_samples']}/{repetitions}; "
            f"partial evidence retained at {output}",
            file=sys.stderr,
        )
        return 1
    print(f"exploratory timing evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
