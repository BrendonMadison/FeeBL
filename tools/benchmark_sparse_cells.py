#!/usr/bin/env python3
"""Reproducible dense/sparse FeeBL A/B benchmark for one input card."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


COMMON = (
    'compute_backend="serial";reduction_mode="strict";'
    "allow_backend_fallback=0;electron_lumi_precompute=0;"
    "photon_lumi_factorized=1;"
)
CASES = {
    "dense": COMMON + 'sparse_cells=0;collision_schedule="legacy";',
    "sparse": COMMON + 'sparse_cells=1;collision_schedule="legacy";',
    "sparse_strict": (COMMON + 'sparse_cells=1;collision_schedule="wavefront_strict";'),
}
REQUIRED_OUTPUTS = (
    "result.out",
    "lumi.ee.out",
    "lumi.ee.out.fne",
    "lumi.ee.out.dLdW.fne",
)
PROVENANCE = re.compile(
    r" cpu_primary_move = |compute_backend_requested|cpu_variant_requested|"
    r"gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|"
    r"backend_provenance|collision_schedule_requested|sparse_cells = "
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inject(source: str, fragment: str, block_name: str) -> str:
    pattern = re.compile(
        rf"(?ms)(^\$PARAMETERS::[ \t]*{re.escape(block_name)}[ \t]*\r?\n" r"[ \t]*\{)(.*?)(\})"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one ${block_name} parameter block; " f"found {len(matches)}"
        )
    match = matches[0]
    replacement = match.group(1) + fragment + match.group(2) + match.group(3)
    return source[: match.start()] + replacement + source[match.end() :]


def physical_cpu() -> int:
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("process has no allowed CPU")
    return allowed[0]


def normalized_result(path: Path) -> bytes:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines(True)
    return "".join(line for line in lines if not PROVENANCE.search(line)).encode()


def run_one(
    binary: Path,
    source: str,
    accelerator: str,
    parameters: str,
    output: Path,
    case: str,
    ordinal: int,
    cpu: int,
    timeout: float,
) -> dict[str, object]:
    run_dir = output / f"{case}-{ordinal:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    card = run_dir / "acc.dat"
    card.write_text(inject(source, CASES[case], parameters), encoding="utf-8")
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        "resource-time.txt",
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        accelerator,
        parameters,
        "result.out",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "1",
            "OMP_PROC_BIND": "TRUE",
            "OMP_PLACES": "cores",
        }
    )
    started = time.monotonic_ns()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed_ns = time.monotonic_ns() - started
    (run_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{case} run {ordinal} failed ({completed.returncode}):\n" + completed.stderr[-4000:]
        )
    for name in REQUIRED_OUTPUTS:
        path = run_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"{case} did not produce nonempty {name}")
    if case != "dense" and "FEEBL sparse-cell summary:" not in completed.stderr:
        raise RuntimeError(f"{case} did not prove sparse execution")
    if case == "sparse_strict" and "FEEBL legacy wavefront summary:" not in completed.stderr:
        raise RuntimeError("strict case did not prove EncounterDAG execution")
    produced = {
        path.name: sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
        and path.name not in {"acc.dat", "stdout.txt", "stderr.txt", "resource-time.txt"}
    }
    return {
        "case": case,
        "ordinal": ordinal,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_ns / 1.0e9,
        "run_directory": str(run_dir),
        "result_sha256": sha256(run_dir / "result.out"),
        "normalized_result_sha256": hashlib.sha256(
            normalized_result(run_dir / "result.out")
        ).hexdigest(),
        "output_sha256": produced,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", default="fccz")
    parser.add_argument("--parameters", default="p_fccz")
    parser.add_argument("--dense-runs", type=int, default=1)
    parser.add_argument("--sparse-runs", type=int, default=3)
    parser.add_argument("--strict-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    card = args.card.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing existing output directory: {output}")
    counts = {
        "dense": args.dense_runs,
        "sparse": args.sparse_runs,
        "sparse_strict": args.strict_runs,
    }
    if any(value < 1 for value in counts.values()):
        parser.error("all run counts must be positive")
    output.mkdir(parents=True)
    cpu = physical_cpu()
    source = card.read_text(encoding="utf-8")
    manifest = {
        "schema": "feebl.sparse-cell-ab.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "card": str(card),
        "card_sha256": sha256(card),
        "accelerator": args.accelerator,
        "parameters": args.parameters,
        "cpu": cpu,
        "counts": counts,
        "cases": CASES,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records: list[dict[str, object]] = []
    maximum = max(counts.values())
    for ordinal in range(maximum):
        for case in ("dense", "sparse", "sparse_strict"):
            if ordinal >= counts[case]:
                continue
            record = run_one(
                binary,
                source,
                args.accelerator,
                args.parameters,
                output,
                case,
                ordinal,
                cpu,
                args.timeout,
            )
            records.append(record)
            print(f"{case} {ordinal}: {record['elapsed_seconds']:.6f} s", flush=True)

    reference = next(record for record in records if record["case"] == "dense")
    reference_files = reference["output_sha256"]
    verification: dict[str, object] = {}
    for case in CASES:
        case_records = [record for record in records if record["case"] == case]
        normalized = {record["normalized_result_sha256"] for record in case_records}
        if normalized != {reference["normalized_result_sha256"]}:
            raise RuntimeError(f"{case} normalized result differs from dense")
        for record in case_records:
            files = record["output_sha256"]
            if set(files) != set(reference_files):
                raise RuntimeError(f"{case} output inventory differs from dense")
            for name in files:
                if name == "result.out":
                    continue
                if files[name] != reference_files[name]:
                    raise RuntimeError(f"{case} {name} differs from dense")
        verification[case] = {
            "status": "PASS",
            "runs": len(case_records),
            "normalized_result_sha256": next(iter(normalized)),
            "non_result_outputs_byte_identical": True,
        }

    timings = {
        case: [float(record["elapsed_seconds"]) for record in records if record["case"] == case]
        for case in CASES
    }
    medians = {case: statistics.median(values) for case, values in timings.items()}
    summary = {
        "schema": "feebl.sparse-cell-ab-summary.v1",
        "status": "PASS",
        "timings_seconds": timings,
        "median_seconds": medians,
        "speedup_dense_over_sparse": medians["dense"] / medians["sparse"],
        "speedup_dense_over_sparse_strict": (medians["dense"] / medians["sparse_strict"]),
        "verification": verification,
        "records": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
