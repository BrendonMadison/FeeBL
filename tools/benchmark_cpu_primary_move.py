#!/usr/bin/env python3
"""Retained paired end-to-end benchmark for cpu_primary_move=1.

The paired control uses the same persistent pool, thread count, strict
deposition path, affinity, deck, and binary with only cpu_primary_move=0.
Consequently control/candidate isolates the production movement selector.
A separately interleaved one-core serial oracle is also retained so the
combined CPU backend contribution can be reported without conflating it with
the selector-only ratio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_result_sha256(path: pathlib.Path) -> str:
    provenance = re.compile(
        rb" cpu_primary_move = |compute_backend_requested|"
        rb"cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|"
        rb"gpu2_|backend_fallback|backend_provenance"
    )
    payload = b"".join(
        line for line in path.read_bytes().splitlines(keepends=True) if not provenance.search(line)
    )
    return hashlib.sha256(payload).hexdigest()


def read_text(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}\n"


def load_average() -> Tuple[float, float, float]:
    first = pathlib.Path("/proc/loadavg").read_text().split()[:3]
    return tuple(float(value) for value in first)  # type: ignore[return-value]


def physical_cpus() -> List[int]:
    allowed = sorted(os.sched_getaffinity(0))
    selected: List[int] = []
    seen = set()
    for cpu in allowed:
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
            identity = (package, core)
        except OSError:
            identity = ("logical", str(cpu))
        if identity not in seen:
            seen.add(identity)
            selected.append(cpu)
    return selected


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ratio(
    control: Sequence[float], candidate: Sequence[float], seed: int, draws: int = 10000
) -> Tuple[float, float]:
    if len(control) != len(candidate) or not control:
        raise ValueError("paired bootstrap requires equal nonempty samples")
    generator = random.Random(seed)
    ratios = []
    for _ in range(draws):
        indices = [generator.randrange(len(control)) for _ in control]
        left = statistics.median(control[index] for index in indices)
        right = statistics.median(candidate[index] for index in indices)
        ratios.append(left / right)
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def parse_threads(value: str) -> List[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item < 1 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "threads must be a comma-separated list of unique positive integers"
        )
    return result


def render(template: str, replacement: str) -> str:
    marker = "cpu_primary_move=0;"
    if template.count(marker) != 1:
        raise RuntimeError("benchmark template must contain exactly one selector marker")
    return template.replace(marker, replacement)


def run_one(
    binary: pathlib.Path,
    template: str,
    run_dir: pathlib.Path,
    case: str,
    threads: int,
    cpus: Sequence[int],
    replacement: str,
    repeat: int,
    order: int,
) -> Dict[str, object]:
    run_dir.mkdir(parents=True)
    (run_dir / "acc.dat").write_text(render(template, replacement))
    cpu_list = ",".join(str(cpu) for cpu in cpus[:threads])
    command = ["taskset", "-c", cpu_list, str(binary), "tiny", "bench", "result.out"]
    environment = os.environ.copy()
    environment.update({"OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "1"})
    before_load = load_average()
    start_ns = time.monotonic_ns()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    end_ns = time.monotonic_ns()
    after_load = load_average()
    (run_dir / "stdout.txt").write_text(completed.stdout)
    (run_dir / "stderr.txt").write_text(completed.stderr)
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case} repeat {repeat} failed with {completed.returncode}:\n"
            f"{completed.stderr[-2000:]}"
        )
    if case.endswith("primary"):
        if "cpu_primary_move=pool-static-primary-v1" not in completed.stderr:
            raise RuntimeError(f"{case} did not record observed production execution")
    elif "cpu_primary_move=pool-static-primary-v1" in completed.stderr:
        raise RuntimeError(f"control case {case} unexpectedly executed primary path")
    return {
        "case": case,
        "threads": threads,
        "repeat": repeat,
        "order": order,
        "cpus": cpu_list,
        "elapsed_ns": end_ns - start_ns,
        "elapsed_seconds": (end_ns - start_ns) / 1.0e9,
        "load1_before": before_load[0],
        "load5_before": before_load[1],
        "load15_before": before_load[2],
        "load1_after": after_load[0],
        "load5_after": after_load[1],
        "load15_after": after_load[2],
        "returncode": completed.returncode,
        "run_directory": str(run_dir.relative_to(run_dir.parents[2])),
        "result_sha256": sha256(run_dir / "result.out"),
        "normalized_result_sha256": normalized_result_sha256(run_dir / "result.out"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--threads", type=parse_threads, default=[1, 2, 4, 8, 10])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--idle-confirmed",
        action="store_true",
        help=(
            "attest that no unrelated CPU-intensive campaign is running; "
            "required for archival-valid timings"
        ),
    )
    parser.add_argument(
        "--max-initial-load1",
        type=float,
        default=2.0,
        help="maximum pre-warmup load1 accepted by the archival idle gate",
    )
    parser.add_argument(
        "--deck",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / "examples/cpu_primary_move/benchmark.dat",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 3 or arguments.warmups < 0 or arguments.max_initial_load1 < 0.0:
        parser.error(
            "repeats must be at least 3, warmups nonnegative, and the load " "threshold nonnegative"
        )
    binary = arguments.binary.resolve()
    deck = arguments.deck.resolve()
    output = arguments.output.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"binary is not executable: {binary}")
    cpus = physical_cpus()
    if max(arguments.threads) > len(cpus):
        parser.error(
            f"requested {max(arguments.threads)} physical CPUs but only {len(cpus)} "
            f"are available in affinity {sorted(os.sched_getaffinity(0))}"
        )
    if output.exists():
        parser.error(f"output already exists: {output}")
    initial_load = load_average()
    output.mkdir(parents=True)
    (output / "runs").mkdir()
    template = deck.read_text()
    shutil.copy2(deck, output / "benchmark.dat")

    root = pathlib.Path(__file__).resolve().parents[1]
    provenance = output / "provenance"
    provenance.mkdir()
    (provenance / "lscpu.txt").write_text(
        read_text(["lscpu", "-e=cpu,node,socket,core,online,maxmhz"])
    )
    (provenance / "uname.txt").write_text(read_text(["uname", "-a"]))
    git_status = read_text(["git", "-C", str(root), "status", "--short", "--branch"])
    git_porcelain = read_text(["git", "-C", str(root), "status", "--porcelain=v1"])
    source_clean = not git_porcelain.strip()
    (provenance / "git_status.txt").write_text(git_status)
    (provenance / "git_status_porcelain.txt").write_text(git_porcelain)
    (provenance / "git_head.txt").write_text(
        read_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    )
    (provenance / "binary_ldd.txt").write_text(read_text(["ldd", str(binary)]))
    (provenance / "processes_before.txt").write_text(
        read_text(["ps", "-eo", "pid,ppid,psr,pcpu,pmem,stat,comm,args", "--sort=-pcpu"])
    )
    manifest = {
        "schema": "feebl.cpu-primary-move.e2e.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "deck": str(deck),
        "deck_sha256": sha256(deck),
        "threads": arguments.threads,
        "repeats": arguments.repeats,
        "warmups": arguments.warmups,
        "command_line": sys.argv,
        "operator_idle_confirmed": arguments.idle_confirmed,
        "initial_load": initial_load,
        "max_initial_load1": arguments.max_initial_load1,
        "source_clean": source_clean,
        "physical_cpu_representatives": cpus,
        "affinity": sorted(os.sched_getaffinity(0)),
        "valid": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Warm the binary/FFTW plans without retaining the timing as evidence.
    warm_threads = max(arguments.threads)
    for warmup in range(arguments.warmups):
        replacement = (
            f'compute_backend="cpu";cpu_threads={warm_threads};'
            'cpu_variant="pool_dual";reduction_mode="strict";'
            "cpu_primary_move=1;"
        )
        run_one(
            binary,
            template,
            output / "runs" / f"warmup_{warmup:02d}",
            "warmup_primary",
            warm_threads,
            cpus,
            replacement,
            -1,
            warmup,
        )

    rows: List[Dict[str, object]] = []
    for repeat in range(arguments.repeats):
        # Serial is interleaved once per cycle and always pinned to the same
        # first physical core used by all candidates.
        rows.append(
            run_one(
                binary,
                template,
                output / "runs" / f"r{repeat:02d}_serial",
                "serial",
                1,
                cpus,
                "cpu_primary_move=0;",
                repeat,
                0,
            )
        )
        order_index = 1
        thread_order = arguments.threads if repeat % 2 == 0 else list(reversed(arguments.threads))
        for threads in thread_order:
            common = (
                f'compute_backend="cpu";cpu_threads={threads};'
                'cpu_variant="pool_dual";reduction_mode="strict";'
            )
            cases = [
                (f"cpu{threads}_control", common + "cpu_primary_move=0;"),
                (f"cpu{threads}_primary", common + "cpu_primary_move=1;"),
            ]
            if (repeat + threads) % 2:
                cases.reverse()
            for case, replacement in cases:
                rows.append(
                    run_one(
                        binary,
                        template,
                        output / "runs" / f"r{repeat:02d}_{case}",
                        case,
                        threads,
                        cpus,
                        replacement,
                        repeat,
                        order_index,
                    )
                )
                order_index += 1

    columns = list(rows[0])
    physics_hashes = {str(row["normalized_result_sha256"]) for row in rows}
    if len(physics_hashes) != 1:
        raise RuntimeError(
            "benchmark accuracy gate failed: normalized result hashes differ: "
            + ", ".join(sorted(physics_hashes))
        )
    with (output / "raw_timings.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    serial_times = [float(row["elapsed_seconds"]) for row in rows if row["case"] == "serial"]
    summary: Dict[str, object] = {
        "serial_median_seconds": statistics.median(serial_times),
        "thread_results": {},
    }
    for threads in arguments.threads:
        control = [
            float(row["elapsed_seconds"]) for row in rows if row["case"] == f"cpu{threads}_control"
        ]
        candidate = [
            float(row["elapsed_seconds"]) for row in rows if row["case"] == f"cpu{threads}_primary"
        ]
        low, high = bootstrap_ratio(control, candidate, 91000 + threads)
        summary["thread_results"][str(threads)] = {  # type: ignore[index]
            "control_median_seconds": statistics.median(control),
            "candidate_median_seconds": statistics.median(candidate),
            "selector_speedup_ratio_of_medians": statistics.median(control)
            / statistics.median(candidate),
            "selector_speedup_bootstrap_95_low": low,
            "selector_speedup_bootstrap_95_high": high,
            "serial_oracle_speedup_ratio_of_medians": statistics.median(serial_times)
            / statistics.median(candidate),
            "paired_ratios": [left / right for left, right in zip(control, candidate)],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    maximum_load_before = max(float(row["load1_before"]) for row in rows)
    maximum_load_after = max(float(row["load1_after"]) for row in rows)
    initial_idle_gate = initial_load[0] <= arguments.max_initial_load1
    validity_failures = []
    if not arguments.idle_confirmed:
        validity_failures.append("operator did not attest an idle host")
    if not initial_idle_gate:
        validity_failures.append(
            f"initial load1 {initial_load[0]:.3f} exceeded " f"{arguments.max_initial_load1:.3f}"
        )
    if not source_clean:
        validity_failures.append("source worktree was not clean")
    valid = not validity_failures
    manifest.update(
        {
            "maximum_observed_load1_before": maximum_load_before,
            "maximum_observed_load1_after": maximum_load_after,
            "initial_idle_gate": (f"pre-warmup load1 <= {arguments.max_initial_load1:.3f}"),
            "initial_idle_gate_passed": initial_idle_gate,
            "valid": valid,
            "accuracy_gate": "exact normalized result SHA-256 identity",
            "normalized_result_sha256": next(iter(physics_hashes)),
            "validity_failures": validity_failures,
            "validity_note": (
                "all archival gates passed"
                if valid
                else "diagnostic only: " + "; ".join(validity_failures)
            ),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (provenance / "processes_after.txt").write_text(
        read_text(["ps", "-eo", "pid,ppid,psr,pcpu,pmem,stat,comm,args", "--sort=-pcpu"])
    )

    lines = [
        "# CPU primary-movement paired benchmark",
        "",
        f"Validity: **{manifest['validity_note']}**.",
        "",
        "Archival validity requires an explicit operator idle attestation, a",
        "clean source worktree, the pre-warmup load threshold, and exact",
        "normalized-result identity across every timed run. Loads observed",
        "after warmup are retained but are not an idle gate because the",
        "benchmark's own workers necessarily raise them.",
        "",
        "The selector speedup compares identical strict persistent-pool runs",
        "with only `cpu_primary_move` changed from 0 to 1. The serial-oracle",
        "column compares the historical one-core backend with the candidate.",
        "",
        "| threads | control s | primary s | selector speedup | 95% bootstrap | serial/primary |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threads in arguments.threads:
        item = summary["thread_results"][str(threads)]  # type: ignore[index]
        lines.append(
            f"| {threads} | {item['control_median_seconds']:.6f} | "
            f"{item['candidate_median_seconds']:.6f} | "
            f"{item['selector_speedup_ratio_of_medians']:.3f}x | "
            f"[{item['selector_speedup_bootstrap_95_low']:.3f}, "
            f"{item['selector_speedup_bootstrap_95_high']:.3f}] | "
            f"{item['serial_oracle_speedup_ratio_of_medians']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "All raw timings, run decks, stdout, stderr, result files, affinity,",
            "binary hash, dynamic-library list, source state, and host topology are",
            "retained beside this summary.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    if not valid:
        print("WARNING: host idle gate failed; timings are diagnostic only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
