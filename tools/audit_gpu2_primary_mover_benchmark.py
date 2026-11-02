#!/usr/bin/env python3
"""Independent standard-library audit of retained GPU2 mover evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DRAW_COUNT = 20_000
SEED = 20260829
REFERENCE_NAMES = ("historical_serial", "gpu_fields_t1", "gpu_primary_serial_t1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap(left: list[float], right: list[float], seed: int) -> list[float]:
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(DRAW_COUNT):
        indices = [generator.randrange(len(left)) for _ in left]
        ratios.append(
            statistics.median(left[index] for index in indices)
            / statistics.median(right[index] for index in indices)
        )
    return [percentile(ratios, 0.025), percentile(ratios, 0.975)]


def close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-12 * max(1.0, abs(left), abs(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    summary = json.loads((evidence / "summary.json").read_text())
    environment = json.loads((evidence / "environment.json").read_text())
    with (evidence / "raw_timings.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    failures: list[str] = []
    cases = list(summary["all_cases"])
    if len(rows) != 143:
        failures.append(f"expected 143 rows, found {len(rows)}")
    if Counter(row["phase"] for row in rows) != {"warmup": 26, "measured": 117}:
        failures.append("phase counts differ from 26 warmups + 117 measurements")
    for case in cases:
        counts = Counter(row["phase"] for row in rows if row["case"] == case)
        if counts != {"warmup": 2, "measured": 9}:
            failures.append(f"{case}: wrong warmup/measured count {dict(counts)}")

    normalized_hashes: set[str] = set()
    rng_hashes: set[str] = set()
    phase_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        block = int(row["block"])
        order = int(row["order_index"])
        run_dir = (
            evidence / row["phase"] / f"block-{block:02d}" / f"order-{order:02d}-{row['case']}"
        )
        required = (
            run_dir / "acc.dat",
            run_dir / "command.json",
            run_dir / "stdout.txt",
            run_dir / "stderr.txt",
            run_dir / "result.out",
            run_dir / "result.normalized.out",
            run_dir / "rndm.save",
        )
        if not all(path.is_file() for path in required):
            failures.append(f"{row['case']} block {block}: missing run artifact")
            continue
        normalized = sha256(run_dir / "result.normalized.out")
        rng = sha256(run_dir / "rndm.save")
        if normalized != row["normalized_result_sha256"]:
            failures.append(f"{row['case']} block {block}: normalized hash drift")
        if rng != row["rng_sha256"]:
            failures.append(f"{row['case']} block {block}: RNG hash drift")
        normalized_hashes.add(normalized)
        rng_hashes.add(rng)
        stderr = (run_dir / "stderr.txt").read_text(encoding="utf-8")
        if "fallback=" in stderr or "backend_fallback" in stderr:
            failures.append(f"{row['case']} block {block}: fallback token")
        if row["case"] != "historical_serial":
            if f"source={environment['source_head']}" not in stderr or "-dirty" in stderr:
                failures.append(f"{row['case']} block {block}: source provenance drift")
            summary_lines = [
                line
                for line in stderr.splitlines()
                if "FEEBL GPU2 " in line and " summary:" in line
            ]
            if len(summary_lines) != 1:
                failures.append(f"{row['case']} block {block}: GPU summary count")
            else:
                fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", summary_lines[0]))
                for key in ("setup_ms", "kernel_ms", "d2h_ms", "total_ms"):
                    phase_values[row["case"]][key].append(float(fields[key]))
                if int(fields.get("device_work_calls", "0")) <= 0:
                    failures.append(f"{row['case']} block {block}: zero device work")
                if row["scope"] == "primary":
                    expected = {
                        "scope": "primary",
                        "primary_mover": row["mover"],
                        "primary_mover_threads": row["threads"],
                        "primary_mover_observed_threads": row["threads"],
                    }
                    for key, value in expected.items():
                        if fields.get(key) != value:
                            failures.append(f"{row['case']} block {block}: {key} provenance")

    if len(normalized_hashes) != 1 or summary["normalized_result_sha256"] not in normalized_hashes:
        failures.append("normalized results are not one exact hash")
    if len(rng_hashes) != 1 or summary["rng_sha256"] not in rng_hashes:
        failures.append("RNG checkpoints are not one exact hash")
    if not summary["accuracy_gate_passed"]:
        failures.append("runner did not pass accuracy gate")
    if environment["source_status_before_output_creation"] != "":
        failures.append("source was not clean before output creation")

    measured: dict[str, list[float]] = {}
    for case in cases:
        selected = sorted(
            (row for row in rows if row["phase"] == "measured" and row["case"] == case),
            key=lambda row: int(row["block"]),
        )
        measured[case] = [float(row["elapsed_seconds"]) for row in selected]
    recomputed: dict[str, object] = {}
    for case in cases:
        values = measured[case]
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        recorded = summary["results"][case]
        if not close(median, float(recorded["median_seconds"])):
            failures.append(f"{case}: median mismatch")
        if not close(mad, float(recorded["mad_seconds"])):
            failures.append(f"{case}: MAD mismatch")
        comparisons: dict[str, object] = {}
        for reference in REFERENCE_NAMES:
            ratio = statistics.median(measured[reference]) / median
            interval = bootstrap(
                measured[reference],
                values,
                SEED + sum(map(ord, case + reference)),
            )
            recorded_comparison = recorded["comparisons"][reference]
            if not close(ratio, float(recorded_comparison["ratio_of_medians"])):
                failures.append(f"{case}/{reference}: ratio mismatch")
            for actual, expected in zip(interval, recorded_comparison["paired_bootstrap_95_ci"]):
                if not close(actual, float(expected)):
                    failures.append(f"{case}/{reference}: interval mismatch")
            comparisons[reference] = {"ratio_of_medians": ratio, "paired_bootstrap_95_ci": interval}
        recomputed[case] = {
            "median_seconds": median,
            "mad_seconds": mad,
            "comparisons": comparisons,
        }

    median_phases = {
        case: {key: statistics.median(values) for key, values in metrics.items()}
        for case, metrics in phase_values.items()
    }
    audit = {
        "schema": "feebl-gpu2-primary-mover-independent-audit-v1",
        "pass": not failures,
        "failures": failures,
        "raw_rows": len(rows),
        "warmup_rows": sum(row["phase"] == "warmup" for row in rows),
        "measured_rows": sum(row["phase"] == "measured" for row in rows),
        "cases": len(cases),
        "exact_normalized_result_hashes": sorted(normalized_hashes),
        "exact_rng_hashes": sorted(rng_hashes),
        "source_head": environment["source_head"],
        "current_binary_sha256": environment["current_binary_sha256"],
        "historical_binary_sha256": environment["historical_binary_sha256"],
        "recomputed_results": recomputed,
        "median_gpu_phase_ms": median_phases,
    }
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
