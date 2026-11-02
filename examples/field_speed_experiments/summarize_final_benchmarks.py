#!/usr/bin/env python3
"""Summarize and validate the final CPU/GPU field-speed benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable


ARTIFACTS = (
    "result.normalized",
    "lumi.ee.out",
    "lumi.ee.out.fne",
    "lumi.ee.out.dLdW.fne",
)
NUMBER = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
ELAPSED = re.compile(
    r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)"
)
SOURCE = re.compile(r"\bsource=([0-9a-f]+(?:-dirty)?)\b")
FIELD_SETUP = re.compile(r"\bfield_setup_ms=([0-9.eE+-]+)")
SETUP = re.compile(r"\bsetup_ms=([0-9.eE+-]+)")
FIELD_PREFETCH = re.compile(r"\bfield_prefetch_ms=([0-9.eE+-]+)")
ORDERED_COMMIT = re.compile(r"\bordered_commit_ms=([0-9.eE+-]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_seconds(path: Path) -> float:
    monotonic = path.with_name("monotonic-time.json")
    if monotonic.is_file():
        payload = json.loads(monotonic.read_text(encoding="utf-8"))
        if (
            payload.get("clock") != "python.time.perf_counter_ns"
            or payload.get("clock_monotonic") is not True
            or payload.get("exit_code") != 0
        ):
            raise RuntimeError(f"invalid monotonic timing receipt: {monotonic}")
        return float(payload["wall_seconds"])
    text = path.read_text(encoding="utf-8")
    match = ELAPSED.search(text)
    if not match:
        raise RuntimeError(f"missing elapsed time in {path}")
    pieces = [float(piece) for piece in match.group(1).split(":")]
    if len(pieces) == 2:
        return pieces[0] * 60.0 + pieces[1]
    if len(pieces) == 3:
        return pieces[0] * 3600.0 + pieces[1] * 60.0 + pieces[2]
    raise RuntimeError(f"unrecognized elapsed time in {path}")


def phase_total_seconds(path: Path) -> float:
    text = (path / "stderr.txt").read_text(encoding="utf-8")
    field_setup = FIELD_SETUP.search(text)
    setup_matches = SETUP.findall(text)
    prefetch = FIELD_PREFETCH.search(text)
    commit = ORDERED_COMMIT.search(text)
    if not prefetch or not commit:
        raise RuntimeError(f"missing phase telemetry in {path / 'stderr.txt'}")
    if field_setup:
        setup_ms = float(field_setup.group(1))
    elif setup_matches:
        setup_ms = float(setup_matches[-1])
    else:
        raise RuntimeError(f"missing setup telemetry in {path / 'stderr.txt'}")
    return (setup_ms + float(prefetch.group(1)) + float(commit.group(1))) / 1000.0


def run_directories(root: Path, prefix: str) -> list[Path]:
    result = sorted(
        (path for path in root.glob(f"{prefix}-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not result:
        raise RuntimeError(f"no {prefix} runs under {root}")
    return result


def timing_summary(paths: Iterable[Path]) -> dict[str, object]:
    paths = list(paths)
    values = [elapsed_seconds(path / "resource-time.txt") for path in paths]
    phase_totals = [phase_total_seconds(path) for path in paths]
    monotonic = [(path / "monotonic-time.json").is_file() for path in paths]
    phase_consistent = [
        wall + 0.050 >= phases for wall, phases in zip(values, phase_totals)
    ]
    return {
        "samples_seconds": values,
        "phase_totals_seconds": phase_totals,
        "monotonic_receipts": monotonic,
        "phase_consistent": phase_consistent,
        "count": len(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def artifact_hashes(paths: Iterable[Path]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ARTIFACTS:
        hashes = [sha256(path / name) for path in paths]
        result[name] = {
            "byte_identical": len(set(hashes)) == 1,
            "sha256": hashes,
        }
    return result


def repeat_comparisons(paths: list[Path]) -> dict[str, list[dict[str, object]]]:
    reference = paths[0]
    return {
        name: [compare_text(reference / name, path / name) for path in paths[1:]]
        for name in ARTIFACTS
    }


def compare_text(reference: Path, candidate: Path) -> dict[str, object]:
    left = reference.read_text(encoding="utf-8").splitlines(keepends=True)
    right = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(left) != len(right):
        raise RuntimeError(
            f"line-count mismatch for {reference.name}: {len(left)} != {len(right)}"
        )
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst: dict[str, object] | None = None
    compared = 0
    for line_number, (left_line, right_line) in enumerate(zip(left, right), 1):
        left_values = [float(value) for value in NUMBER.findall(left_line)]
        right_values = [float(value) for value in NUMBER.findall(right_line)]
        if (
            NUMBER.sub("<number>", left_line) != NUMBER.sub("<number>", right_line)
            or len(left_values) != len(right_values)
        ):
            raise RuntimeError(
                f"non-numeric structure mismatch in {reference.name}, "
                f"line {line_number}"
            )
        for value_index, (expected, actual) in enumerate(
            zip(left_values, right_values), 1
        ):
            absolute = abs(expected - actual)
            scale = max(abs(expected), abs(actual))
            relative = absolute / scale if scale else 0.0
            maximum_absolute = max(maximum_absolute, absolute)
            if relative > maximum_relative:
                maximum_relative = relative
                worst = {
                    "line": line_number,
                    "value_index": value_index,
                    "reference": expected,
                    "candidate": actual,
                    "absolute": absolute,
                    "relative": relative,
                }
            compared += 1
    return {
        "numeric_fields": compared,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "worst_relative_field": worst,
    }


def source_labels(paths: Iterable[Path]) -> list[str]:
    labels: list[str] = []
    for path in paths:
        match = SOURCE.search((path / "result.out").read_text(encoding="utf-8"))
        if not match:
            raise RuntimeError(f"missing source provenance in {path / 'result.out'}")
        labels.append(match.group(1))
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_root", type=Path)
    parser.add_argument("gpu_root", type=Path)
    parser.add_argument("--historical-seconds", type=float, default=147.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    cpu_root = args.cpu_root.resolve(strict=True)
    gpu_root = args.gpu_root.resolve(strict=True)
    cpu = {
        mode: run_directories(cpu_root, mode)
        for mode in ("fft", "auto", "direct")
    }
    gpu = run_directories(gpu_root, "batch-64")

    timings = {mode: timing_summary(paths) for mode, paths in cpu.items()}
    timings["gpu_mode4_batch64"] = timing_summary(gpu)
    fft_median = float(timings["fft"]["median_seconds"])
    for mode in ("auto", "direct", "gpu_mode4_batch64"):
        median = float(timings[mode]["median_seconds"])
        timings[mode]["speedup_vs_cpu_fft"] = fft_median / median
        timings[mode]["speedup_vs_historical_147s"] = (
            args.historical_seconds / median
        )

    hashes = {mode: artifact_hashes(paths) for mode, paths in cpu.items()}
    hashes["gpu_mode4_batch64"] = artifact_hashes(gpu)
    repeat_numeric = {
        mode: repeat_comparisons(paths) for mode, paths in cpu.items()
    }
    repeat_numeric["gpu_mode4_batch64"] = repeat_comparisons(gpu)
    cpu_reference = cpu["fft"][0]
    comparisons = {
        f"cpu_fft_vs_gpu_{name}": compare_text(
            cpu_reference / name, gpu[0] / name
        )
        for name in ARTIFACTS
    }
    comparisons["cpu_cross_solver_sha256"] = {
        name: {
            mode: sha256(cpu[mode][0] / name)
            for mode in ("fft", "auto", "direct")
        }
        for name in ARTIFACTS
    }

    labels = source_labels(
        [path for paths in cpu.values() for path in paths] + gpu
    )
    clean_source = len(set(labels)) == 1 and all(
        not label.endswith("-dirty") for label in labels
    )
    result_limit = 5.0e-5
    luminosity_hard_limit = 3.0e-2
    result_ok = (
        comparisons["cpu_fft_vs_gpu_result.normalized"]
        ["maximum_relative_difference"] <= result_limit
    )
    luminosity_ok = all(
        comparisons[f"cpu_fft_vs_gpu_{name}"]["maximum_relative_difference"]
        <= luminosity_hard_limit
        for name in ARTIFACTS[1:]
    )
    luminosity_preferred = all(
        comparisons[f"cpu_fft_vs_gpu_{name}"]["maximum_relative_difference"]
        <= 1.0e-2
        for name in ARTIFACTS[1:]
    )
    repeat_numeric_ok = all(
        comparison["maximum_relative_difference"]
        <= (result_limit if name == "result.normalized" else luminosity_hard_limit)
        for mode in repeat_numeric.values()
        for name, comparisons_for_file in mode.items()
        for comparison in comparisons_for_file
    )
    repeat_byte_identical = all(
        details["byte_identical"]
        for mode in hashes.values()
        for details in mode.values()
    )
    cpu_solver_ok = all(
        len(set(details.values())) == 1
        for details in comparisons["cpu_cross_solver_sha256"].values()
    )
    timing_receipts_ok = all(
        all(details["monotonic_receipts"]) and all(details["phase_consistent"])
        for details in timings.values()
    )

    payload = {
        "schema": "feebl.field-speed-final.v1",
        "status": "PASS"
        if clean_source
        and result_ok
        and luminosity_ok
        and repeat_numeric_ok
        and cpu_solver_ok
        and timing_receipts_ok
        else "FAIL",
        "source": {
            "labels": sorted(set(labels)),
            "single_clean_commit": clean_source,
        },
        "timings": timings,
        "within_mode_byte_determinism": hashes,
        "within_mode_numeric_reproducibility": repeat_numeric,
        "comparisons": comparisons,
        "acceptance": {
            "normalized_result_relative_limit": result_limit,
            "luminosity_binwise_relative_hard_limit": luminosity_hard_limit,
            "normalized_result_pass": result_ok,
            "luminosity_pass": luminosity_ok,
            "luminosity_preferred_one_percent_pass": luminosity_preferred,
            "within_mode_numeric_reproducibility_pass": repeat_numeric_ok,
            "within_mode_byte_determinism_observed": repeat_byte_identical,
            "cpu_cross_solver_byte_identity_pass": cpu_solver_ok,
            "monotonic_phase_consistency_pass": timing_receipts_ok,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
