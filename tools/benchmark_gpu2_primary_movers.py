#!/usr/bin/env python3
"""Preregistered paired benchmark for compositional GPU2 primary movers.

The runner deliberately has no adaptive search.  It executes the frozen case
matrix in balanced blocks, retains every process artifact, and refuses timing
unless the source tree, GPU, binary hashes, physical-core affinity, provenance,
normalized output, and RNG checkpoint gates all pass.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PREREGISTERED_THREADS = (1, 2, 4, 8, 10)
PREREGISTERED_BLOCKS = 9
PREREGISTERED_WARMUPS = 2
PREREGISTERED_PARTICLES = 100_000
PREREGISTERED_GRID = (32, 32, 8)
PREREGISTERED_BOOTSTRAP_SEED = 20260829
PREREGISTERED_BOOTSTRAP_DRAWS = 20_000
PREREGISTERED_MAX_LOAD1 = 2.0
PREREGISTERED_HISTORICAL_SHA256 = "b05c34b43f008b7342c2673067ba46179bcce6d9570ec812627ea6f65fd1659e"

BASE_PHYSICS = (
    "integration_method=2;electron_distribution_rho=2;"
    "do_eloss=0;do_photons=0;do_lumi=0;"
    "ST_spin_flip=0;bmt_precession=0;"
    "do_coherent=0;do_trident=0;do_prod=0;"
    "do_pairs=0;track_pairs=0;do_muons=0;track_muons=0;"
    "grids=1;store_beam=0;rndm_load=0;rndm_save=1;"
)
# The frozen historical executable predates registration of this input-card
# spelling; its legacy implementation is nevertheless hard-wired to the same
# CIC rho method.  Amendment 001 removes only the redundant parser key from
# that comparator deck after the first campaign stopped before a successful
# warmup or any measured sample.
HISTORICAL_BASE_PHYSICS = BASE_PHYSICS.replace("electron_distribution_rho=2;", "")

NORMALIZE_RE = re.compile(
    r"compute_backend_requested|cpu_variant_requested|gpu_variant_requested|"
    r"gpu_mode_requested|gpu2_|backend_fallback|backend_provenance"
)


@dataclass(frozen=True)
class Case:
    name: str
    binary_kind: str
    scope: str
    mover: str
    threads: int
    fragment: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    except OSError as error:
        return f"unavailable: {error}\n"


def physical_cpus() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    chosen: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            identity = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError:
            identity = ("logical", str(cpu))
        if identity not in seen:
            seen.add(identity)
            chosen.append(cpu)
    return chosen


def load_average() -> tuple[float, float, float]:
    values = Path("/proc/loadavg").read_text().split()[:3]
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def active_gpu_processes() -> str:
    return command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )


def gpu_is_idle(report: str) -> bool:
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    return not [line for line in lines if "No running processes" not in line]


def make_cases() -> list[Case]:
    cases = [
        Case("historical_serial", "historical", "serial", "serial", 1, HISTORICAL_BASE_PHYSICS),
        Case(
            "gpu_fields_t1",
            "current",
            "fields",
            "serial",
            1,
            BASE_PHYSICS + 'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";'
            'gpu2_deposition="atomic";gpu2_host_threads=1;'
            'gpu_device=0;reduction_mode="fast";',
        ),
        Case(
            "gpu_primary_serial_t1",
            "current",
            "primary",
            "serial",
            1,
            BASE_PHYSICS + 'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";'
            'gpu2_deposition="atomic";gpu2_primary_mover="serial";'
            'gpu2_host_threads=1;gpu_device=0;reduction_mode="fast";',
        ),
    ]
    for mover in ("openmp", "pool"):
        for threads in PREREGISTERED_THREADS:
            cases.append(
                Case(
                    f"gpu_primary_{mover}_t{threads}",
                    "current",
                    "primary",
                    mover,
                    threads,
                    BASE_PHYSICS + 'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";'
                    'gpu2_deposition="atomic";'
                    f'gpu2_primary_mover="{mover}";'
                    f"gpu2_host_threads={threads};"
                    'gpu_device=0;reduction_mode="fast";',
                )
            )
    return cases


def make_deck(source: Path, destination: Path, fragment: str) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "{n_x=256;n_y=256;n_z=16;",
        f"{{n_x={PREREGISTERED_GRID[0]};n_y={PREREGISTERED_GRID[1]};"
        f"n_z={PREREGISTERED_GRID[2]};",
    )
    text = text.replace("n_t=1;n_m=50000;", f"n_t=1;n_m={PREREGISTERED_PARTICLES};")
    text = text.replace("hist_ee_bins=1010;", fragment + "hist_ee_bins=1010;")
    destination.write_text(text, encoding="utf-8")


def normalize_result(path: Path) -> str:
    return "".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines(True)
        if not NORMALIZE_RE.search(line)
    )


def parse_gpu_summary(stderr: str, case: Case, source_head: str) -> dict[str, str]:
    final = [line for line in stderr.splitlines() if line.startswith(" FEEBL: final ")]
    if len(final) != 1:
        raise RuntimeError(f"{case.name}: expected one final provenance line")
    if f"source={source_head}" not in final[0] or "source=" + source_head + "-dirty" in final[0]:
        raise RuntimeError(f"{case.name}: binary source provenance is not clean HEAD")
    if "fallback=" in stderr or "backend_fallback" in stderr:
        raise RuntimeError(f"{case.name}: fallback is a failed sample")
    label = "primary" if case.scope == "primary" else "fields"
    summaries = [line for line in stderr.splitlines() if f"FEEBL GPU2 {label} summary:" in line]
    if len(summaries) != 1:
        raise RuntimeError(f"{case.name}: expected one GPU2 {label} summary")
    fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", summaries[0]))
    if fields.get("scope") != case.scope:
        raise RuntimeError(f"{case.name}: actual scope mismatch")
    if int(fields.get("device_work_calls", "0")) <= 0:
        raise RuntimeError(f"{case.name}: no device work observed")
    if case.scope == "primary":
        expected = {
            "host_boundary": "compact_primary_fields",
            "primary_mover": case.mover,
            "primary_mover_threads": str(case.threads),
            "primary_mover_observed_threads": str(case.threads),
        }
        for key, value in expected.items():
            if fields.get(key) != value:
                raise RuntimeError(f"{case.name}: {key}={fields.get(key)!r}, expected {value!r}")
        for key in ("primary_mover_calls", "primary_mover_particles", "primary_mover_ranges"):
            if int(fields.get(key, "0")) <= 0:
                raise RuntimeError(f"{case.name}: {key} did not prove execution")
    return fields


def run_one(
    case: Case,
    binary: Path,
    deck: Path,
    run_dir: Path,
    cpus: Sequence[int],
    phase: str,
    block: int,
    order_index: int,
    source_head: str,
    timeout_seconds: int,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(deck, run_dir / "acc.dat")
    affinity = cpus[: case.threads]
    cpu_list = ",".join(str(cpu) for cpu in affinity)
    command = ["taskset", "-c", cpu_list, str(binary), "c_nt1", "p_c_nt1", "result.out"]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(case.threads),
            "OMP_PROC_BIND": "TRUE",
            "OMP_PLACES": "cores",
        }
    )
    before = load_average()
    started_ns = time.monotonic_ns()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed_ns = time.monotonic_ns() - started_ns
    after = load_average()
    (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    key: environment[key]
                    for key in ("OMP_DYNAMIC", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{case.name} {phase} block {block} exited {completed.returncode}")
    required = (run_dir / "result.out", run_dir / "rndm.save")
    if not all(path.is_file() and path.stat().st_size for path in required):
        raise RuntimeError(f"{case.name}: missing result or RNG checkpoint")
    normalized = normalize_result(run_dir / "result.out")
    (run_dir / "result.normalized.out").write_text(normalized, encoding="utf-8")
    summary: dict[str, str] = {}
    if case.binary_kind == "current":
        summary = parse_gpu_summary(completed.stderr, case, source_head)
    return {
        "phase": phase,
        "block": block,
        "order_index": order_index,
        "case": case.name,
        "scope": case.scope,
        "mover": case.mover,
        "threads": case.threads,
        "cpu_affinity": cpu_list,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_ns / 1.0e9,
        "returncode": completed.returncode,
        "load1_before": before[0],
        "load1_after": after[0],
        "result_sha256": sha256(run_dir / "result.out"),
        "normalized_result_sha256": sha256(run_dir / "result.normalized.out"),
        "rng_sha256": sha256(run_dir / "rndm.save"),
        "stderr_sha256": sha256(run_dir / "stderr.txt"),
        "gpu_summary": summary,
        "run_directory": str(run_dir),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def paired_bootstrap(
    reference: Sequence[float], candidate: Sequence[float], seed: int
) -> tuple[float, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired bootstrap requires equal nonempty vectors")
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(PREREGISTERED_BOOTSTRAP_DRAWS):
        indices = [generator.randrange(len(reference)) for _ in reference]
        ref = statistics.median(reference[index] for index in indices)
        cand = statistics.median(candidate[index] for index in indices)
        values.append(ref / cand)
    return percentile(values, 0.025), percentile(values, 0.975)


def balanced_order(cases: Sequence[Case], block: int) -> list[Case]:
    offset = block % len(cases)
    order = list(cases[offset:]) + list(cases[:offset])
    if block % 2:
        order.reverse()
    return order


def summarize(rows: Sequence[dict[str, object]], cases: Sequence[Case]) -> dict[str, object]:
    measured = [row for row in rows if row["phase"] == "measured"]
    by_case: dict[str, list[float]] = {
        case.name: [float(row["elapsed_seconds"]) for row in measured if row["case"] == case.name]
        for case in cases
    }
    for name, values in by_case.items():
        if len(values) != PREREGISTERED_BLOCKS:
            raise RuntimeError(f"{name}: expected {PREREGISTERED_BLOCKS} samples")
    references = {
        "historical_serial": by_case["historical_serial"],
        "gpu_fields_t1": by_case["gpu_fields_t1"],
        "gpu_primary_serial_t1": by_case["gpu_primary_serial_t1"],
    }
    results: dict[str, object] = {}
    for case in cases:
        times = by_case[case.name]
        median = statistics.median(times)
        mad = statistics.median(abs(value - median) for value in times)
        comparisons: dict[str, object] = {}
        for reference_name, reference in references.items():
            low, high = paired_bootstrap(
                reference,
                times,
                PREREGISTERED_BOOTSTRAP_SEED + sum(map(ord, case.name + reference_name)),
            )
            ratio = statistics.median(reference) / median
            comparisons[reference_name] = {
                "speedup_definition": f"{reference_name}_seconds/{case.name}_seconds",
                "ratio_of_medians": ratio,
                "paired_ratios": [left / right for left, right in zip(reference, times)],
                "paired_bootstrap_95_ci": [low, high],
                "two_x_gate": ratio >= 2.0 and low >= 2.0,
                "positive_speedup_gate": low > 1.0,
            }
        results[case.name] = {
            "median_seconds": median,
            "mad_seconds": mad,
            "samples_seconds": times,
            "comparisons": comparisons,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--historical-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-confirmed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    if not args.idle_confirmed:
        parser.error("--idle-confirmed is mandatory for the sealed campaign")
    root = Path(__file__).resolve().parents[1]
    source_head = command_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    source_status = command_output(["git", "-C", str(root), "status", "--porcelain=v1"])
    if source_status:
        raise RuntimeError("source worktree must be clean before timing")
    current_binary = args.binary.resolve()
    historical_binary = args.historical_binary.resolve()
    for binary in (current_binary, historical_binary):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"not an executable: {binary}")
    historical_hash = sha256(historical_binary)
    if historical_hash != PREREGISTERED_HISTORICAL_SHA256:
        raise RuntimeError("historical binary hash does not match preregistration")
    cpus = physical_cpus()
    if len(cpus) < max(PREREGISTERED_THREADS):
        raise RuntimeError("fewer than ten physical cores are available")
    initial_load = load_average()
    if initial_load[0] > PREREGISTERED_MAX_LOAD1:
        raise RuntimeError(
            f"initial load1 {initial_load[0]:.3f} exceeds " f"{PREREGISTERED_MAX_LOAD1:.3f}"
        )
    active_before = active_gpu_processes()
    if not gpu_is_idle(active_before):
        raise RuntimeError("GPU has another compute process")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "active_gpu_before.txt").write_text(active_before, encoding="utf-8")
    cases = make_cases()
    decks = output / "decks"
    decks.mkdir()
    source_deck = root / "examples/decks/quick.dat"
    deck_paths: dict[str, Path] = {}
    for case in cases:
        path = decks / f"{case.name}.dat"
        make_deck(source_deck, path, case.fragment)
        deck_paths[case.name] = path

    environment = {
        "schema": "feebl-gpu2-primary-mover-preregistered-v1",
        "classification": "feature_bounded_benchmark",
        "claim_eligible": False,
        "claim_limit": (
            "one simplified mover-heavy workload with radiation, spin, "
            "luminosity and tracked-secondary side effects disabled"
        ),
        "source_head": source_head,
        "source_status_before_output_creation": source_status,
        "current_binary": str(current_binary),
        "current_binary_sha256": sha256(current_binary),
        "historical_binary": str(historical_binary),
        "historical_binary_sha256": historical_hash,
        "source_deck_sha256": sha256(source_deck),
        "deck_sha256": {name: sha256(path) for name, path in deck_paths.items()},
        "physical_cpu_order": cpus,
        "used_physical_cpus": cpus[: max(PREREGISTERED_THREADS)],
        "threads": PREREGISTERED_THREADS,
        "blocks": PREREGISTERED_BLOCKS,
        "warmups_per_case": PREREGISTERED_WARMUPS,
        "particles_per_beam": PREREGISTERED_PARTICLES,
        "grid": PREREGISTERED_GRID,
        "initial_load": initial_load,
        "max_initial_load1": PREREGISTERED_MAX_LOAD1,
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(["nvidia-smi", "-q"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "cmake": command_output(["cmake", "--version"]),
        "ldd_current": command_output(["ldd", str(current_binary)]),
        "ldd_historical": command_output(["ldd", str(historical_binary)]),
        "processes_before": command_output(
            ["ps", "-eo", "pid,ppid,psr,pcpu,pmem,stat,comm,args", "--sort=-pcpu"]
        ),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    canonical_normalized: str | None = None
    canonical_rng: str | None = None
    try:
        phases = [("warmup", block) for block in range(PREREGISTERED_WARMUPS)]
        phases += [("measured", block) for block in range(PREREGISTERED_BLOCKS)]
        for phase, block in phases:
            order = balanced_order(cases, block)
            block_rows: list[dict[str, object]] = []
            for order_index, case in enumerate(order):
                binary = historical_binary if case.binary_kind == "historical" else current_binary
                run_dir = (
                    output / phase / f"block-{block:02d}" / (f"order-{order_index:02d}-{case.name}")
                )
                row = run_one(
                    case,
                    binary,
                    deck_paths[case.name],
                    run_dir,
                    cpus,
                    phase,
                    block,
                    order_index,
                    source_head,
                    args.timeout_seconds,
                )
                rows.append(row)
                block_rows.append(row)
            baseline = next(row for row in block_rows if row["case"] == "gpu_primary_serial_t1")
            normalized_hash = str(baseline["normalized_result_sha256"])
            rng_hash = str(baseline["rng_sha256"])
            if canonical_normalized is None:
                canonical_normalized = normalized_hash
                canonical_rng = rng_hash
            if normalized_hash != canonical_normalized or rng_hash != canonical_rng:
                raise RuntimeError("serial baseline changed across blocks")
            baseline_dir = Path(str(baseline["run_directory"]))
            baseline_text = (baseline_dir / "result.normalized.out").read_text()
            for row in block_rows:
                if (
                    row["normalized_result_sha256"] != normalized_hash
                    or row["rng_sha256"] != rng_hash
                ):
                    candidate_dir = Path(str(row["run_directory"]))
                    candidate_text = (candidate_dir / "result.normalized.out").read_text()
                    diff = "".join(
                        difflib.unified_diff(
                            baseline_text.splitlines(True),
                            candidate_text.splitlines(True),
                            fromfile="gpu_primary_serial_t1",
                            tofile=str(row["case"]),
                        )
                    )
                    (candidate_dir / "comparison_failure.diff").write_text(diff, encoding="utf-8")
                    raise RuntimeError(f"{row['case']}: normalized output or RNG was not exact")
            (output / f"gpu_after_{phase}_block_{block:02d}.txt").write_text(
                active_gpu_processes(), encoding="utf-8"
            )
    except Exception as error:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "utc_failed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "completed_runs": len(rows),
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    results = summarize(rows, cases)
    summary = {
        "schema": "feebl-gpu2-primary-mover-preregistered-v1",
        "classification": "feature_bounded_benchmark",
        "claim_eligible": False,
        "broad_2x_to_10x_claim": False,
        "accuracy_gate": (
            "byte-identical normalized result.out and rndm.save for every "
            "warmup and measured block; orbit-state identity was separately "
            "qualified by the registered E2E test"
        ),
        "accuracy_gate_passed": True,
        "measured_blocks": PREREGISTERED_BLOCKS,
        "all_cases": [case.name for case in cases],
        "results": results,
        "maximum_load1_before": max(float(row["load1_before"]) for row in rows),
        "maximum_load1_after": max(float(row["load1_after"]) for row in rows),
        "normalized_result_sha256": canonical_normalized,
        "rng_sha256": canonical_rng,
        "utc_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "raw_timings.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "phase",
            "block",
            "order_index",
            "case",
            "scope",
            "mover",
            "threads",
            "cpu_affinity",
            "elapsed_ns",
            "elapsed_seconds",
            "load1_before",
            "load1_after",
            "normalized_result_sha256",
            "rng_sha256",
            "run_directory",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    (output / "processes_after.txt").write_text(
        command_output(["ps", "-eo", "pid,ppid,psr,pcpu,pmem,stat,comm,args", "--sort=-pcpu"]),
        encoding="utf-8",
    )
    (output / "active_gpu_after.txt").write_text(active_gpu_processes(), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "median_seconds": value["median_seconds"],
                    "vs_historical": value["comparisons"]["historical_serial"],
                    "vs_gpu_fields": value["comparisons"]["gpu_fields_t1"],
                    "vs_gpu_primary_serial": value["comparisons"]["gpu_primary_serial_t1"],
                }
                for name, value in results.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
