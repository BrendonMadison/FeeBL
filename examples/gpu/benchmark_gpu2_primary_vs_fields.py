#!/usr/bin/env python3
"""Paired, provenance-retaining GPU2 primary-vs-fields performance screen.

This is intentionally classified as an exploratory screen: it compares the
host-boundary ablation only, uses fewer repetitions than the preregistered
promotion campaign by default, and has no serial/CPU baselines or powered
stochastic-equivalence matrix.  It must never emit a promotion claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


FIELDS = (
    'compute_backend="gpu";gpu_mode=2;gpu2_scope="fields";'
    'gpu2_deposition="atomic";gpu_device=0;reduction_mode="fast";'
)
PRIMARY = (
    'compute_backend="gpu";gpu_mode=2;gpu2_scope="primary";'
    'gpu2_deposition="auto";gpu_device=0;reduction_mode="fast";'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return result.stdout


def make_deck(source: Path, output: Path, fragment: str,
              particles: int) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace("n_t=1;n_m=50000;",
                        f"n_t=1;n_m={particles};")
    text = text.replace("hist_ee_bins=1010;",
                        fragment + "hist_ee_bins=1010;")
    text = text.replace("rndm_save=0;", "rndm_save=1;")
    output.write_text(text, encoding="utf-8")


def parse_summary(stderr: str, variant: str) -> dict[str, float | int | str]:
    prefix = f"FEEBL GPU2 {variant} summary:"
    lines = [line.strip() for line in stderr.splitlines() if prefix in line]
    if len(lines) != 1:
        raise RuntimeError(f"expected one {prefix!r} line, got {len(lines)}")
    fields: dict[str, float | int | str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", lines[0]):
        try:
            fields[key] = int(value)
        except ValueError:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
    expected_scope = "primary" if variant == "primary" else "fields"
    if fields.get("scope") != expected_scope:
        raise RuntimeError(f"wrong actual scope in summary: {lines[0]}")
    if int(fields.get("device_work_calls", 0)) <= 0:
        raise RuntimeError(f"no observed device work: {lines[0]}")
    if variant == "primary":
        exact = {
            "host_boundary": "compact_primary_fields",
            "primary_update": "host_serial",
            "spin_transport": "host_serial",
            "beamstrahlung": "host_serial",
            "rng": "legacy_host_ordered",
            "host_phi_materialized": 0,
        }
        for key, value in exact.items():
            if fields.get(key) != value:
                raise RuntimeError(
                    f"primary provenance mismatch {key}={fields.get(key)!r}")
    return fields


def run_one(binary: Path, deck: Path, run_dir: Path, cpu: str,
            variant: str) -> dict[str, object]:
    run_dir.mkdir(parents=True)
    shutil.copy2(deck, run_dir / "acc.dat")
    command = ["taskset", "--cpu-list", cpu, str(binary),
               "c_nt1", "p_c_nt1", "result.out"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=run_dir, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    elapsed = time.perf_counter() - started
    (run_dir / "stdout").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr").write_text(result.stderr, encoding="utf-8")
    (run_dir / "elapsed_seconds.txt").write_text(
        f"{elapsed:.9f}\n", encoding="ascii")
    if result.returncode != 0:
        raise RuntimeError(
            f"{variant} exited {result.returncode}; see {run_dir / 'stderr'}")
    if not (run_dir / "result.out").is_file() or not (run_dir / "rndm.save").is_file():
        raise RuntimeError(f"{variant} omitted result or RNG checkpoint")
    summary = parse_summary(result.stderr, variant)
    return {
        "variant": variant,
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "command": command,
        "result_sha256": sha256(run_dir / "result.out"),
        "rng_sha256": sha256(run_dir / "rndm.save"),
        "stdout_sha256": sha256(run_dir / "stdout"),
        "stderr_sha256": sha256(run_dir / "stderr"),
        "summary": summary,
        "run_directory": str(run_dir),
    }


def compare_physics(root: Path, fields: dict[str, object],
                    primary: dict[str, object]) -> None:
    fields_dir = Path(str(fields["run_directory"]))
    primary_dir = Path(str(primary["run_directory"]))
    provenance = re.compile(
        r"compute_backend_requested|cpu_variant_requested|gpu_variant_requested|"
        r"gpu_mode_requested|gpu2_|backend_fallback|backend_provenance")
    normalized_paths = []
    for label, run_dir in (("fields", fields_dir),
                           ("primary", primary_dir)):
        normalized = primary_dir / f"normalized-{label}.result.out"
        lines = (run_dir / "result.out").read_text(encoding="utf-8").splitlines(True)
        normalized.write_text("".join(line for line in lines
                                      if not provenance.search(line)),
                              encoding="utf-8")
        normalized_paths.append(normalized)
    compare = subprocess.run(
        [sys.executable, str(root / "examples/gpu/compare_feebl_outputs.py"),
         str(normalized_paths[0]), str(normalized_paths[1]),
         "--rtol", "5e-6"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    (primary_dir / "physics_comparison.txt").write_text(
        compare.stdout, encoding="utf-8")
    if compare.returncode != 0:
        raise RuntimeError(
            f"paired physics comparison failed; see "
            f"{primary_dir / 'physics_comparison.txt'}")
    if fields["rng_sha256"] != primary["rng_sha256"]:
        raise RuntimeError("primary changed the paired legacy RNG checkpoint")


def bootstrap_ci(values: list[float], seed: int = 20260826,
                 draws: int = 20000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    generator = random.Random(seed)
    medians = []
    for _ in range(draws):
        medians.append(statistics.median(
            values[generator.randrange(len(values))] for _ in values))
    medians.sort()
    return [medians[int(0.025 * (draws - 1))],
            medians[int(0.975 * (draws - 1))]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", default="0")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--particles", type=int, default=50000)
    parser.add_argument("--allow-active-gpu", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.warmups < 1 or args.particles < 1:
        parser.error("pairs, warmups, and particles must be positive")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"not an executable: {binary}")

    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    active_gpu = command_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"])
    (output / "active_gpu_before.txt").write_text(active_gpu,
                                                   encoding="utf-8")
    meaningful = [line for line in active_gpu.splitlines()
                  if line.strip() and "No running processes" not in line]
    if meaningful and not args.allow_active_gpu:
        raise RuntimeError(
            "GPU already has a compute process; refusing a contended screen")

    decks = output / "decks"
    decks.mkdir()
    source_deck = root / "examples/decks/quick.dat"
    fields_deck = decks / "fields.dat"
    primary_deck = decks / "primary.dat"
    make_deck(source_deck, fields_deck, FIELDS, args.particles)
    make_deck(source_deck, primary_deck, PRIMARY, args.particles)
    if sha256(fields_deck) == sha256(primary_deck):
        raise RuntimeError("candidate decks unexpectedly have the same hash")

    environment = {
        "classification": "exploratory_screen",
        "claim_eligible": False,
        "claim_ineligibility": [
            "primary-vs-fields ablation only; no serial/CPU baselines",
            "default repetition count is below preregistered promotion count",
            "no powered stochastic equivalence or broad physics matrix",
        ],
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_head": command_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "source_status": command_output(
            ["git", "-C", str(root), "status", "--porcelain=v1"]),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "fields_deck_sha256": sha256(fields_deck),
        "primary_deck_sha256": sha256(primary_deck),
        "cpu_affinity": args.cpu,
        "pairs": args.pairs,
        "warmups_per_variant": args.warmups,
        "particles_per_beam": args.particles,
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(["nvidia-smi", "-q"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "ldd": command_output(["ldd", str(binary)]),
        "uptime": command_output(["uptime"]),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    runs: list[dict[str, object]] = []
    for warmup in range(args.warmups):
        for variant, deck in (("fields", fields_deck),
                              ("primary", primary_deck)):
            run_one(binary, deck,
                    output / "warmups" / f"{warmup:02d}-{variant}",
                    args.cpu, variant)

    for pair in range(args.pairs):
        order = (("fields", fields_deck), ("primary", primary_deck))
        if pair % 2:
            order = tuple(reversed(order))
        pair_runs: dict[str, dict[str, object]] = {}
        for order_index, (variant, deck) in enumerate(order):
            record = run_one(
                binary, deck,
                output / "runs" / f"pair-{pair:02d}-{order_index}-{variant}",
                args.cpu, variant)
            record["pair"] = pair
            record["order_index"] = order_index
            runs.append(record)
            pair_runs[variant] = record
        compare_physics(root, pair_runs["fields"], pair_runs["primary"])

    fields_times = [float(run["elapsed_seconds"]) for run in runs
                    if run["variant"] == "fields"]
    primary_times = [float(run["elapsed_seconds"]) for run in runs
                     if run["variant"] == "primary"]
    paired = []
    for pair in range(args.pairs):
        by_variant = {str(run["variant"]): float(run["elapsed_seconds"])
                      for run in runs if run["pair"] == pair}
        paired.append(by_variant["fields"] / by_variant["primary"])
    summary = {
        "schema": "feebl-gpu2-primary-screen-v1",
        "classification": "exploratory_screen",
        "claim_eligible": False,
        "speedup_definition": "fields_seconds / primary_seconds",
        "fields_median_seconds": statistics.median(fields_times),
        "primary_median_seconds": statistics.median(primary_times),
        "ratio_of_medians_speedup": (
            statistics.median(fields_times) /
            statistics.median(primary_times)),
        "paired_speedups": paired,
        "median_paired_speedup": statistics.median(paired),
        "paired_median_bootstrap_95_ci": bootstrap_ci(paired),
        "physics_rtol": 5e-6,
        "rng_checkpoint": "byte-identical in every measured pair",
        "runs": runs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair", "order_index", "variant", "elapsed_seconds",
                         "result_sha256", "rng_sha256", "run_directory"])
        for run in runs:
            writer.writerow([run["pair"], run["order_index"], run["variant"],
                             f"{float(run['elapsed_seconds']):.9f}",
                             run["result_sha256"], run["rng_sha256"],
                             run["run_directory"]])
    print(json.dumps({key: summary[key] for key in (
        "classification", "claim_eligible", "fields_median_seconds",
        "primary_median_seconds", "ratio_of_medians_speedup",
        "median_paired_speedup", "paired_median_bootstrap_95_ci")},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
