#!/usr/bin/env python3
"""Sealed six-cell FeeBL luminosity optimization timing matrix.

This runner is intentionally specific to the preregistered reduced
benchmark16/s9/p_s9/n_m=50000 exploratory campaign.  It refuses selector,
sample-count, CPU, or schedule overrides.  Every FeeBL invocation is a fresh
CPU-0-pinned process and every artifact is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PHOTON_SUPPORT_DIR = REPO_ROOT / "examples" / "legacy_photon_lumi_fast"
sys.path.insert(0, str(PHOTON_SUPPORT_DIR))
import photon_lumi_benchmark_support as photon_support  # noqa: E402


EXPECTED_BINARY_SHA256 = (
    "fde8656602fd35baa28604bc67ada21d769835eec5bde406269f1a5a5bc9d734"
)
EXPECTED_DECK_SHA256 = (
    "9a1de68dae835271efcb7b4d1f45099f726a976d3b2295c3274c58107d0ec77e"
)
PINNED_CPU = 0
MACROPARTICLES = 50000
WARMUP_CYCLES = 1
TIMED_BLOCKS = 9
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260826
PHOTON_RELATIVE_GATE = 2.0e-6
PHOTON_ABSOLUTE_GATE = 2.0e-6

VARIANTS = {
    "e0p0": {"electron_lumi_precompute": 0, "photon_lumi_factorized": 0},
    "e1p0": {"electron_lumi_precompute": 1, "photon_lumi_factorized": 0},
    "e2p0": {"electron_lumi_precompute": 2, "photon_lumi_factorized": 0},
    "e0p1": {"electron_lumi_precompute": 0, "photon_lumi_factorized": 1},
    "e1p1": {"electron_lumi_precompute": 1, "photon_lumi_factorized": 1},
    "e2p1": {"electron_lumi_precompute": 2, "photon_lumi_factorized": 1},
}

# Six Williams/cyclic AB/BA rows, followed by the three preregistered even-row
# extensions.  This literal is checked against PREREGISTRATION.json.
SCHEDULE = [
    ["e0p0", "e1p0", "e2p1", "e2p0", "e1p1", "e0p1"],
    ["e1p0", "e2p0", "e0p0", "e0p1", "e2p1", "e1p1"],
    ["e2p0", "e0p1", "e1p0", "e1p1", "e0p0", "e2p1"],
    ["e0p1", "e1p1", "e2p0", "e2p1", "e1p0", "e0p0"],
    ["e1p1", "e2p1", "e0p1", "e0p0", "e2p0", "e1p0"],
    ["e2p1", "e0p0", "e1p1", "e1p0", "e0p1", "e2p0"],
    ["e0p0", "e1p0", "e2p1", "e2p0", "e1p1", "e0p1"],
    ["e2p0", "e0p1", "e1p0", "e1p1", "e0p0", "e2p1"],
    ["e1p1", "e2p1", "e0p1", "e0p0", "e2p0", "e1p0"],
]
WARMUP_ORDER = ["e0p0", "e1p0", "e2p0", "e0p1", "e1p1", "e2p1"]

ELECTRON_PROVENANCE_RE = re.compile(
    br"^ electron_lumi_precompute = [12] .*\n", re.MULTILINE
)
PHOTON_PROVENANCE_RE = re.compile(
    br"^ photon_lumi_factorized = 1 .*\n", re.MULTILINE
)
PHOTON_SUMMARY_RE = photon_support.SUMMARY_RE


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout.rstrip()


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return run_text(["git", "-C", str(REPO_ROOT), *arguments])

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = status.splitlines() if status else []
    return {
        "commit": git("rev-parse", "HEAD"),
        "status_porcelain": lines,
        "status_sha256": sha256_bytes("\n".join(lines).encode()),
        "tracked_diff_sha256": sha256_bytes(
            git("diff", "--binary", "HEAD").encode()
        ),
    }


def profile_bounds(raw: bytes) -> tuple[int, int]:
    header = re.compile(
        br"(^\$PARAMETERS::[ \t]+p_s9[ \t]*\r?\n[ \t]*\{)", re.MULTILINE
    )
    matches = list(header.finditer(raw))
    if len(matches) != 1:
        raise RuntimeError(f"expected one p_s9 block, found {len(matches)}")
    start = matches[0].end()
    next_block = raw.find(b"\n$", start)
    end = len(raw) if next_block < 0 else next_block
    return start, end


def derive_deck(raw: bytes, electron: int, photon: int) -> bytes:
    start, end = profile_bounds(raw)
    block = raw[start:end]
    frozen_selector = b"electron_lumi_precompute=0;photon_lumi_factorized=0;"
    if block.count(frozen_selector) != 1:
        raise RuntimeError("frozen p_s9 selector prefix changed")
    for frozen_override in (
        b"n_x=64;n_y=64;n_z=8;",
        b"n_t=1;n_m=50000;",
        b"rndm_seed=424242;rndm_load=0;rndm_save=1;",
    ):
        if block.count(frozen_override) != 1:
            raise RuntimeError(
                f"frozen reduced-card override changed: {frozen_override!r}"
            )
    selectors = (
        f"electron_lumi_precompute={electron};"
        f"photon_lumi_factorized={photon};"
    ).encode()
    return raw[:start] + block.replace(frozen_selector, selectors, 1) + raw[end:]


def work_signature(deck: bytes) -> str:
    normalized, replacements = re.subn(
        br"electron_lumi_precompute=[012];photon_lumi_factorized=[01];",
        b"electron_lumi_precompute=<E>;photon_lumi_factorized=<P>;",
        deck,
    )
    if replacements != 1:
        raise RuntimeError("derived card selector normalization failed")
    return sha256_bytes(normalized)


def electron_payload(result: pathlib.Path) -> bytes:
    """Return byte-exact electron luminosity aggregates and step history."""
    lines = result.read_bytes().splitlines(keepends=True)
    selected: list[bytes] = []
    in_steps = False
    step_count = 0
    for line in lines:
        if (line.startswith(b"lumi_ee = ") or
                line.startswith(b"lumi_ee_high = ") or
                line.startswith(b"E_cm = ") or
                line.startswith(b"lumi_ee=") or
                line.startswith(b"lumi_ee_high=")):
            selected.append(line)
        if line.startswith(b"step ") and b"lumi total" in line:
            selected.append(line)
            in_steps = True
            continue
        if in_steps:
            if re.fullmatch(br"[0-9]+[ \t]+[-+0-9.eE]+[ \t]+[-+0-9.eE]+\r?\n?", line):
                selected.append(line)
                step_count += 1
            else:
                in_steps = False
    if len(selected) < 6 or step_count < 1:
        raise RuntimeError(
            f"electron payload parse failed: {len(selected)} lines, "
            f"{step_count} step rows"
        )
    return b"".join(selected)


def normalized_result(result: pathlib.Path) -> bytes:
    payload = result.read_bytes()
    payload = ELECTRON_PROVENANCE_RE.sub(b"", payload)
    payload = PHOTON_PROVENANCE_RE.sub(b"", payload)
    return payload


def read_number(path: pathlib.Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def system_snapshot() -> dict[str, Any]:
    loadavg = pathlib.Path("/proc/loadavg").read_text().strip().split()
    temperatures: list[dict[str, Any]] = []
    for pattern in (
        "/sys/class/thermal/thermal_zone*/temp",
        "/sys/class/hwmon/hwmon*/temp*_input",
    ):
        for path in sorted(pathlib.Path("/").glob(pattern.lstrip("/"))):
            value = read_number(path)
            if value is not None:
                temperatures.append({"path": str(path), "millidegrees_c": value})
    frequencies: list[dict[str, Any]] = []
    for path in sorted(pathlib.Path("/sys/devices/system/cpu").glob(
            "cpu*/cpufreq/scaling_cur_freq")):
        value = read_number(path)
        if value is not None:
            frequencies.append({"path": str(path), "khz": value})
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "load1": float(loadavg[0]),
        "load5": float(loadavg[1]),
        "load15": float(loadavg[2]),
        "runnable_entities": loadavg[3],
        "temperatures": temperatures,
        "temperature_status": "available" if temperatures else "unavailable",
        "cpu_frequencies": frequencies,
        "frequency_status": "available" if frequencies else "unavailable",
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def photon_provenance(stderr: pathlib.Path, enabled: bool) -> dict[str, int] | None:
    text = stderr.read_text(encoding="utf-8", errors="replace")
    match = PHOTON_SUMMARY_RE.search(text)
    if not enabled:
        if match is not None:
            raise RuntimeError("photon selector 0 unexpectedly emitted fast provenance")
        return None
    if match is None:
        raise RuntimeError("photon selector 1 omitted factorized provenance")
    calls, cartesian, visited, avoided = map(int, match.groups())
    if not (calls > 0 and cartesian == 0 and visited == 0 and avoided > 0):
        raise RuntimeError("factorized provenance counters violate contract")
    return {
        "factorized_calls": calls,
        "cartesian_calls": cartesian,
        "cartesian_pairs_visited": visited,
        "cartesian_pairs_avoided": avoided,
    }


def assert_selector_provenance(result: pathlib.Path, variant: str) -> None:
    payload = result.read_bytes()
    electron = VARIANTS[variant]["electron_lumi_precompute"]
    photon = VARIANTS[variant]["photon_lumi_factorized"]
    electron_rows = ELECTRON_PROVENANCE_RE.findall(payload)
    photon_rows = PHOTON_PROVENANCE_RE.findall(payload)
    if electron == 0 and electron_rows:
        raise RuntimeError(f"{variant}: default electron selector emitted provenance")
    if electron in (1, 2):
        expected = f" electron_lumi_precompute = {electron} ".encode()
        if len(electron_rows) != 1 or expected not in electron_rows[0]:
            raise RuntimeError(f"{variant}: electron selector provenance mismatch")
    if photon == 0 and photon_rows:
        raise RuntimeError(f"{variant}: default photon selector emitted provenance")
    if photon == 1 and len(photon_rows) != 1:
        raise RuntimeError(f"{variant}: photon selector provenance mismatch")


def run_once(binary: pathlib.Path, deck: pathlib.Path, variant: str,
             run_dir: pathlib.Path, block: int, order_index: int,
             warmup: bool, evidence_root: pathlib.Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    before_path = run_dir / "system-before.json"
    after_path = run_dir / "system-after.json"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    write_json(before_path, system_snapshot())
    command = [
        "taskset", "--cpu-list", str(PINNED_CPU), str(binary),
        f"--acc_file={deck}", "s9", "p_s9", "result.out",
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "{0}",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    started = time.perf_counter_ns()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command, cwd=run_dir, env=environment,
            stdout=stdout, stderr=stderr, check=False,
        )
    ended = time.perf_counter_ns()
    write_json(after_path, system_snapshot())
    timing_path = run_dir / "timing.json"
    write_json(timing_path, {
        "clock": "time.perf_counter_ns",
        "start_ns": started,
        "end_ns": ended,
        "elapsed_ns": ended - started,
        "seconds": (ended - started) * 1.0e-9,
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"{run_dir.name}: FeeBL exited {completed.returncode}; see retained stderr"
        )
    result_path = run_dir / "result.out"
    rng_path = run_dir / "rndm.save"
    if not result_path.is_file() or not rng_path.is_file():
        raise RuntimeError(f"{run_dir.name}: missing result.out or rndm.save")
    assert_selector_provenance(result_path, variant)
    fast = photon_provenance(
        stderr_path, VARIANTS[variant]["photon_lumi_factorized"] == 1
    )
    record = {
        "variant": variant,
        "selectors": VARIANTS[variant],
        "block": block,
        "order_index": order_index,
        "warmup": warmup,
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_ns": ended - started,
        "seconds": (ended - started) * 1.0e-9,
        "normalized_result_sha256": sha256_bytes(normalized_result(result_path)),
        "electron_payload_sha256": sha256_bytes(electron_payload(result_path)),
        "photon_aggregates": photon_support.parse_aggregates(result_path),
        "factorized_provenance": fast,
        "artifacts": {
            name: artifact(path, evidence_root) for name, path in {
                "stdout": stdout_path,
                "stderr": stderr_path,
                "result": result_path,
                "rng": rng_path,
                "timing": timing_path,
                "system_before": before_path,
                "system_after": after_path,
            }.items()
        },
    }
    print(
        f"{'warmup' if warmup else f'block {block:02d}'} "
        f"slot {order_index} {variant}: {record['seconds']:.6f} s",
        flush=True,
    )
    return record


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def comparison(samples: dict[str, list[float]], baseline: str,
               candidate: str) -> dict[str, Any]:
    first = samples[baseline]
    second = samples[candidate]
    if len(first) != TIMED_BLOCKS or len(second) != TIMED_BLOCKS:
        raise RuntimeError("comparison sample count violates preregistration")
    point = statistics.median(first) / statistics.median(second)
    paired = [left / right for left, right in zip(first, second)]
    seed_material = f"{BOOTSTRAP_SEED}:{baseline}:{candidate}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(TIMED_BLOCKS) for _ in range(TIMED_BLOCKS)]
        bootstrap.append(
            statistics.median(first[index] for index in indices) /
            statistics.median(second[index] for index in indices)
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "interpretation": "values above one favor candidate",
        "ratio_of_medians_speedup": point,
        "median_paired_speedup": statistics.median(paired),
        "paired_speedups": paired,
        "paired_bootstrap_ratio_of_medians_95pct": {
            "lower": percentile(bootstrap, 0.025),
            "upper": percentile(bootstrap, 0.975),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": seed,
            "resampling_unit": "complete timed block",
            "method": "percentile interval with linear quantile interpolation",
        },
    }


def validate_outputs(records: list[dict[str, Any]], root: pathlib.Path) -> dict[str, Any]:
    if len(records) != (WARMUP_CYCLES + TIMED_BLOCKS) * len(VARIANTS):
        raise RuntimeError("run count violates preregistration")
    by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for record in records:
        by_variant[record["variant"]].append(record)

    # Every repetition of a variant must be byte-identical, including warmup.
    repeat_exact = {}
    for variant, rows in by_variant.items():
        result_hashes = {row["artifacts"]["result"]["sha256"] for row in rows}
        rng_hashes = {row["artifacts"]["rng"]["sha256"] for row in rows}
        if len(result_hashes) != 1 or len(rng_hashes) != 1:
            raise RuntimeError(f"{variant}: repeated result or RNG bytes changed")
        repeat_exact[variant] = {
            "result_sha256": next(iter(result_hashes)),
            "rng_sha256": next(iter(rng_hashes)),
        }

    # All selectors must preserve the final RNG stream exactly.
    all_rng = {value["rng_sha256"] for value in repeat_exact.values()}
    if len(all_rng) != 1:
        raise RuntimeError("selector matrix changed final RNG state")

    # Electron modes must be full-output exact within each photon algorithm.
    same_photon_exact = {}
    for photon in (0, 1):
        names = [f"e{electron}p{photon}" for electron in (0, 1, 2)]
        hashes = {
            by_variant[name][0]["normalized_result_sha256"] for name in names
        }
        if len(hashes) != 1:
            raise RuntimeError(f"photon={photon}: electron modes differ after provenance")
        same_photon_exact[str(photon)] = next(iter(hashes))

    # Electron observables are exact across all six selector combinations.
    electron_hashes = {
        rows[0]["electron_payload_sha256"] for rows in by_variant.values()
    }
    if len(electron_hashes) != 1:
        raise RuntimeError("electron payload differs across selector matrix")

    # Photon factorization is checked against its audited 2e-6 hybrid gate.
    photon_checks = []
    for electron in (0, 1, 2):
        cart = by_variant[f"e{electron}p0"][0]["photon_aggregates"]
        fast = by_variant[f"e{electron}p1"][0]["photon_aggregates"]
        if not photon_support.physics_close(
                cart, fast, PHOTON_RELATIVE_GATE, PHOTON_ABSOLUTE_GATE):
            raise RuntimeError(f"electron mode {electron}: photon physics gate failed")
        photon_checks.append({
            "electron_lumi_precompute": electron,
            "relative_gate": PHOTON_RELATIVE_GATE,
            "absolute_gate": PHOTON_ABSOLUTE_GATE,
            "status": "pass",
        })

    fast_counters = {
        json.dumps(row["factorized_provenance"], sort_keys=True)
        for name, rows in by_variant.items() if name.endswith("p1")
        for row in rows
    }
    if len(fast_counters) != 1:
        raise RuntimeError("factorized logical-work counters changed across runs")
    return {
        "repeat_result_and_rng_byte_identity": repeat_exact,
        "all_variants_rng_sha256": next(iter(all_rng)),
        "same_photon_provenance_stripped_full_output_sha256": same_photon_exact,
        "all_variants_electron_payload_sha256": next(iter(electron_hashes)),
        "photon_factorization_physics_checks": photon_checks,
        "factorized_logical_work": json.loads(next(iter(fast_counters))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--deck", required=True, type=pathlib.Path)
    parser.add_argument("--preregistration", required=True, type=pathlib.Path)
    parser.add_argument("--amendment", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    for command in ("git", "g++", "cmake", "ldd", "lscpu", "taskset", "uname"):
        if shutil.which(command) is None:
            parser.error(f"required command unavailable: {command}")
    binary = args.binary.resolve(strict=True)
    deck = args.deck.resolve(strict=True)
    preregistration = args.preregistration.resolve(strict=True)
    amendment_path = args.amendment.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if sha256_file(binary) != EXPECTED_BINARY_SHA256:
        raise SystemExit("binary SHA256 differs from preregistration")
    if sha256_file(deck) != EXPECTED_DECK_SHA256:
        raise SystemExit("source-deck SHA256 differs from preregistration")
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if prereg.get("schedule") != SCHEDULE or prereg.get("variants") != VARIANTS:
        raise SystemExit("runner literals disagree with PREREGISTRATION.json")
    if (amendment.get("schedule") != SCHEDULE or
            amendment.get("variants") != VARIANTS or
            amendment.get("exact_reduced_deck_sha256") != EXPECTED_DECK_SHA256 or
            amendment.get("prior_attempt_status") !=
            "aborted-before-first-timed-sample"):
        raise SystemExit("runner literals disagree with AMENDMENT-01.json")
    if sorted(os.sched_getaffinity(0)).count(PINNED_CPU) != 1:
        raise SystemExit("CPU0 is not in the allowed affinity mask")

    output.mkdir(parents=True)
    try:
        os.sched_setaffinity(0, {PINNED_CPU})
        provenance = output / "provenance"
        raw_root = output / "raw"
        inputs = output / "inputs"
        provenance.mkdir()
        raw_root.mkdir()
        inputs.mkdir()

        copied_binary = provenance / "feebl"
        shutil.copy2(binary, copied_binary)
        copied_binary.chmod(copied_binary.stat().st_mode | 0o111)
        copied_deck = inputs / "p_s9_50k_default.dat"
        shutil.copy2(deck, copied_deck)
        copied_runner = provenance / "benchmark_matrix_e2e.py"
        shutil.copy2(pathlib.Path(__file__), copied_runner)
        copied_prereg = provenance / "PREREGISTRATION.json"
        shutil.copy2(preregistration, copied_prereg)
        copied_amendment = provenance / "AMENDMENT-01.json"
        shutil.copy2(amendment_path, copied_amendment)

        derived_paths: dict[str, pathlib.Path] = {}
        derived_records: dict[str, Any] = {}
        signatures = set()
        source_bytes = deck.read_bytes()
        for variant, selectors in VARIANTS.items():
            payload = derive_deck(
                source_bytes, selectors["electron_lumi_precompute"],
                selectors["photon_lumi_factorized"],
            )
            path = inputs / f"{variant}.dat"
            path.write_bytes(payload)
            signature = work_signature(payload)
            signatures.add(signature)
            derived_paths[variant] = path
            derived_records[variant] = {
                **artifact(path, output),
                "selectors": selectors,
                "work_signature_sha256": signature,
            }
        if len(signatures) != 1:
            raise RuntimeError("derived cards differ beyond the two selectors")

        toolchain = {
            "gxx_version": run_text(["g++", "--version"]),
            "cmake_version": run_text(["cmake", "--version"]),
            "python": sys.version,
            "platform": platform.platform(),
            "uname": run_text(["uname", "-a"]),
            "lscpu": run_text(["lscpu"]),
            "ldd": run_text(["ldd", str(copied_binary)]),
        }
        write_json(provenance / "toolchain.json", toolchain)
        linked_libraries = []
        for line in toolchain["ldd"].splitlines():
            match = re.search(r"=>\s+(/\S+)", line)
            if match is None:
                match = re.match(r"\s*(/\S+)\s+\(", line)
            if match and pathlib.Path(match.group(1)).is_file():
                library = pathlib.Path(match.group(1)).resolve()
                linked_libraries.append({
                    "path": str(library),
                    "sha256": sha256_file(library),
                    "bytes": library.stat().st_size,
                })
        preflight = system_snapshot()
        running_feebl = subprocess.run(
            ["pgrep", "-x", "feebl"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False, text=True,
        ).stdout.split()
        if running_feebl:
            raise RuntimeError(f"preflight found running FeeBL PIDs: {running_feebl}")
        if preflight["load1"] > prereg["idle_preflight"]["maximum_load1"]:
            raise RuntimeError(
                f"preflight load1 {preflight['load1']} exceeds preregistered limit"
            )
        write_json(provenance / "system-preflight.json", preflight)

        records: list[dict[str, Any]] = []
        for cycle in range(WARMUP_CYCLES):
            for order_index, variant in enumerate(WARMUP_ORDER):
                records.append(run_once(
                    copied_binary, derived_paths[variant], variant,
                    raw_root / f"warmup-{cycle:02d}-{order_index}-{variant}",
                    cycle, order_index, True, output,
                ))
        for block, order in enumerate(SCHEDULE):
            for order_index, variant in enumerate(order):
                records.append(run_once(
                    copied_binary, derived_paths[variant], variant,
                    raw_root / f"block-{block:02d}-{order_index}-{variant}",
                    block, order_index, False, output,
                ))

        accuracy = validate_outputs(records, output)
        seconds: dict[str, list[float]] = {name: [] for name in VARIANTS}
        for block in range(TIMED_BLOCKS):
            rows = [row for row in records if not row["warmup"] and row["block"] == block]
            mapping = {row["variant"]: row for row in rows}
            if set(mapping) != set(VARIANTS):
                raise RuntimeError(f"block {block} does not contain the fixed matrix")
            for variant in VARIANTS:
                seconds[variant].append(mapping[variant]["seconds"])

        comparisons = {}
        for baseline, candidate in (
            ("e0p0", "e1p0"), ("e0p0", "e2p0"), ("e1p0", "e2p0"),
            ("e0p1", "e1p1"), ("e0p1", "e2p1"), ("e1p1", "e2p1"),
            ("e0p0", "e0p1"), ("e1p0", "e1p1"), ("e2p0", "e2p1"),
            ("e0p0", "e2p1"),
        ):
            key = f"{candidate}_vs_{baseline}"
            comparisons[key] = comparison(seconds, baseline, candidate)

        postflight = system_snapshot()
        write_json(provenance / "system-postflight.json", postflight)
        document = {
            "schema": "feebl-electron-photon-lumi-fixed-matrix-v1",
            "valid": True,
            "classification": "exploratory-training-card-feature-path",
            "eligible_for_broad_or_project_level_speedup_claim": False,
            "disclaimer": (
                "Reduced benchmark16 p_s9 n_m=50000 training-card screen only; "
                "it is not a broad FeeBL performance claim."
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "timing_scope": "fresh process launch through normal FeeBL exit",
            "matrix_frozen_before_first_timing": True,
            "cpu": PINNED_CPU,
            "warmup_cycles": WARMUP_CYCLES,
            "timed_blocks": TIMED_BLOCKS,
            "schedule": SCHEDULE,
            "variants": VARIANTS,
            "seconds_by_variant_in_block_order": seconds,
            "summary_by_variant": {
                variant: {
                    "median_seconds": statistics.median(values),
                    "minimum_seconds": min(values),
                    "maximum_seconds": max(values),
                    "mean_seconds": statistics.mean(values),
                } for variant, values in seconds.items()
            },
            "comparisons": comparisons,
            "accuracy_and_work": accuracy,
            "input_work_signature_sha256": next(iter(signatures)),
            "derived_cards": derived_records,
            "provenance": {
                "binary": artifact(copied_binary, output),
                "source_deck": artifact(copied_deck, output),
                "runner": artifact(copied_runner, output),
                "preregistration": artifact(copied_prereg, output),
                "amendment": artifact(copied_amendment, output),
                "toolchain": artifact(provenance / "toolchain.json", output),
                "system_preflight": artifact(
                    provenance / "system-preflight.json", output
                ),
                "system_postflight": artifact(
                    provenance / "system-postflight.json", output
                ),
                "linked_libraries": linked_libraries,
                "git": git_metadata(),
                "environment": {
                    "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
                    "OMP_PROC_BIND": "close", "OMP_PLACES": "{0}",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                },
            },
            "samples": records,
        }
        write_json(output / "results.json", document)
        print(json.dumps({
            "valid": True,
            "output": str(output),
            "summary_by_variant": document["summary_by_variant"],
            "requested_comparisons": {
                key: value for key, value in comparisons.items()
                if key in {
                    "e2p0_vs_e0p0", "e2p0_vs_e1p0",
                    "e2p1_vs_e0p1", "e2p1_vs_e1p1", "e2p1_vs_e0p0",
                }
            },
        }, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            "schema": "feebl-electron-photon-lumi-fixed-matrix-failure-v1",
            "utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "retained_partial_output": True,
        }
        write_json(output / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    main()
