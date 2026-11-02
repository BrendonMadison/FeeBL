#!/usr/bin/env python3
"""Run a retained, independently validated photon-luminosity benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
import photon_lumi_benchmark_support as support  # noqa: E402


def run_text(argv: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n"
            f"{completed.stderr}"
        )
    return completed.stdout.rstrip()


def run_bytes(argv: list[str], cwd: Path | None = None,
              accepted: tuple[int, ...] = (0,)) -> bytes:
    completed = subprocess.run(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in accepted:
        error = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n"
            f"{error}"
        )
    return completed.stdout


def artifact(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": support.sha256_file(resolved),
    }


def copy_artifact(source: Path, destination: Path, root: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return artifact(destination, root)


def git_metadata() -> dict[str, Any]:
    revision = run_text(["git", "rev-parse", "HEAD"], REPO_ROOT)
    status_text = run_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        REPO_ROOT,
    )
    status_lines = status_text.splitlines() if status_text else []
    return {
        "git_revision": revision,
        "git_dirty": bool(status_lines),
        "git_status_porcelain": status_lines,
        "git_status_sha256": hashlib.sha256(
            "\n".join(status_lines).encode("utf-8")
        ).hexdigest(),
    }


def dirty_worktree_patch() -> bytes:
    """Capture tracked and untracked repository state before evidence exists."""
    chunks = [run_bytes(
        ["git", "diff", "--binary", "--full-index", "HEAD", "--"],
        REPO_ROOT,
    )]
    untracked = run_bytes(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        REPO_ROOT,
    )
    for encoded in untracked.split(b"\0"):
        if not encoded:
            continue
        relative = os.fsdecode(encoded)
        chunks.append(run_bytes(
            ["git", "diff", "--binary", "--full-index", "--no-index",
             "--", "/dev/null", relative],
            REPO_ROOT, accepted=(0, 1),
        ))
    patch = b"".join(chunks)
    if not patch:
        raise RuntimeError(
            "git reports a dirty tree but no tracked/untracked patch was captured"
        )
    return patch


def require_build_descriptor(parser: argparse.ArgumentParser,
                             value: str, option: str) -> str:
    normalized = value.strip()
    rejected = {"unknown", "unspecified", "none", "n/a"}
    if not normalized or normalized.lower() in rejected or \
            normalized.lower().startswith("unknown"):
        parser.error(f"{option} must be explicit and must not be 'unknown'")
    return normalized


def allowed_physical_cpu() -> tuple[int, list[int], list[dict[str, int]]]:
    allowed = sorted(os.sched_getaffinity(0))
    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    text = run_text(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"])
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 4 or fields[3].strip().upper() not in {"Y", "YES", "1"}:
            continue
        cpu, core, socket = map(int, fields[:3])
        if cpu not in allowed or (socket, core) in seen:
            continue
        seen.add((socket, core))
        rows.append({"cpu": cpu, "core": core, "socket": socket})
    rows.sort(key=lambda row: (row["socket"], row["core"], row["cpu"]))
    if not rows:
        raise RuntimeError("no allowed online physical CPU was discovered")
    return rows[0]["cpu"], allowed, rows


def linked_libraries(ldd_text: str) -> list[tuple[str, Path]]:
    paths: dict[str, Path] = {}
    for line in ldd_text.splitlines():
        match = re.search(r"=>\s+(/\S+)", line)
        if not match:
            match = re.match(r"\s*(/\S+)\s+\(", line)
        if match:
            original = match.group(1)
            candidate = Path(original)
            if not candidate.is_file():
                raise RuntimeError(
                    f"ldd reported an unavailable shared library: {original}"
                )
            paths[original] = candidate.resolve()
    if not paths:
        raise RuntimeError("ldd did not expose any hashable shared libraries")
    return sorted(paths.items())


def parse_fast_provenance(stderr_path: Path) -> dict[str, int]:
    match = support.SUMMARY_RE.search(stderr_path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("factorized run did not emit complete provenance")
    calls, cart_calls, visited, avoided = map(int, match.groups())
    if not (calls > 0 and cart_calls == 0 and visited == 0 and avoided > 0):
        raise RuntimeError("factorized run emitted invalid provenance counters")
    return {
        "factorized_calls": calls,
        "cartesian_calls": cart_calls,
        "cartesian_pairs_visited": visited,
        "cartesian_pairs_avoided": avoided,
    }


def run_once(binary: Path, deck: Path, case_name: str, variant: str,
             run_dir: Path, pinned_cpu: int, pair: int,
             order_index: int, warmup: bool, root: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    command = [
        "taskset", "--cpu-list", str(pinned_cpu), str(binary),
        f"--acc_file={deck}", "tiny", variant, "result.out",
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": f"{{{pinned_cpu}}}",
    })
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command, cwd=run_dir, env=environment,
            stdout=stdout, stderr=stderr, check=False,
        )
        ended = time.perf_counter_ns()
    elapsed_ns = ended - started
    if elapsed_ns <= 0:
        raise RuntimeError(f"{case_name}/{variant} has nonpositive elapsed time")
    elapsed = elapsed_ns * 1.0e-9
    timing_path = run_dir / "timing.json"
    timing_path.write_text(json.dumps({
        "clock": "time.perf_counter_ns",
        "start_ns": started,
        "end_ns": ended,
        "elapsed_ns": elapsed_ns,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path = run_dir / "result.out"
    rng_path = run_dir / "rndm.save"
    if completed.returncode != 0:
        error = stderr_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"{case_name}/{variant} exited {completed.returncode}:\n{error}"
        )
    if not result_path.is_file() or not rng_path.is_file():
        raise RuntimeError(f"{case_name}/{variant} omitted result.out or rndm.save")
    record: dict[str, Any] = {
        "case": case_name,
        "pair": pair,
        "repetition": pair,
        "order_index": order_index,
        "variant": variant,
        "warmup": warmup,
        "seconds": elapsed,
        "exit_code": completed.returncode,
        "command": command,
        "aggregates": support.parse_aggregates(result_path),
        "artifacts": {
            "stdout": artifact(stdout_path, root),
            "stderr": artifact(stderr_path, root),
            "result": artifact(result_path, root),
            "rng": artifact(rng_path, root),
            "timing": artifact(timing_path, root),
        },
    }
    if variant == "factorized":
        record["provenance"] = parse_fast_provenance(stderr_path)
    elif "FEEBL legacy photon-lumi summary" in stderr_path.read_text(encoding="utf-8"):
        raise RuntimeError("Cartesian run unexpectedly emitted fast-path provenance")
    return record


def validate_pair_in_runner(cartesian: dict[str, Any],
                            factorized: dict[str, Any], relative_gate: float,
                            absolute_gate: float) -> None:
    physics = support.physics_close(
        cartesian["aggregates"], factorized["aggregates"],
        relative_gate, absolute_gate,
    )
    rng = (cartesian["artifacts"]["rng"]["sha256"] ==
           factorized["artifacts"]["rng"]["sha256"])
    if not physics:
        raise RuntimeError(
            "paired run failed the 2e-6 relative / 2e-6 absolute "
            "aggregate physics gate"
        )
    if not rng:
        raise RuntimeError("paired run changed final RNG state")
    for sample in (cartesian, factorized):
        sample["physics_equivalent"] = physics
        sample["rng_identical"] = rng


def materialize_deck(template: str, macroparticles: int, path: Path) -> None:
    marker = "n_m=4000"
    if template.count(marker) != 2:
        raise RuntimeError("benchmark deck template no longer has two n_m=4000 cases")
    path.write_text(template.replace(marker, f"n_m={macroparticles}"),
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument(
        "--macroparticles", type=int, nargs="+", default=[4000, 20000, 50000],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compile-flags", required=True)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("repetitions must be at least 5")
    if args.warmups < 1:
        parser.error("warmups must be positive")
    if any(value < 4000 for value in args.macroparticles):
        parser.error("each macroparticle count must be at least 4000")
    if len(set(args.macroparticles)) != len(args.macroparticles):
        parser.error("macroparticle counts must be unique")
    args.compiler = require_build_descriptor(
        parser, args.compiler, "--compiler"
    )
    args.compile_flags = require_build_descriptor(
        parser, args.compile_flags, "--compile-flags"
    )
    for command in ("git", "lscpu", "ldd", "taskset"):
        if shutil.which(command) is None:
            raise SystemExit(f"required command is unavailable: {command}")
    binary = args.binary.resolve(strict=True)
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {binary}")

    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "test_photon_lumi_benchmark_validator.py")],
        check=True,
    )
    git = git_metadata()
    working_tree_patch = dirty_worktree_patch() if git["git_dirty"] else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or
              REPO_ROOT / "benchmark_results" /
              f"legacy_photon_lumi_fast_{timestamp}").resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    provenance_dir = output / "provenance"
    provenance_dir.mkdir()

    copied_binary = provenance_dir / "feebl"
    shutil.copy2(binary, copied_binary)
    copied_binary.chmod(copied_binary.stat().st_mode | 0o111)
    ldd_text = run_text(["ldd", str(copied_binary)])
    lscpu_text = run_text(["lscpu"])
    (provenance_dir / "ldd.txt").write_text(ldd_text + "\n", encoding="utf-8")
    (provenance_dir / "lscpu.txt").write_text(lscpu_text + "\n", encoding="utf-8")
    working_tree_patch_path = provenance_dir / "working_tree.patch"
    if working_tree_patch is not None:
        working_tree_patch_path.write_bytes(working_tree_patch)
    pinned_cpu, allowed_cpus, physical_cpus = allowed_physical_cpu()
    retained_libraries: list[dict[str, Any]] = []
    for index, (original, source) in enumerate(linked_libraries(ldd_text)):
        destination = (
            provenance_dir / "libraries" /
            f"{index:03d}-{source.name}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        retained_libraries.append({
            "original_host_path": original,
            "resolved_host_path": str(source),
            "retained": artifact(destination, output),
        })

    sources = {
        "benchmark_binary": copied_binary,
        "deck_template": SCRIPT_DIR / "tiny.dat",
        "runner": Path(__file__),
        "validator": SCRIPT_DIR / "photon_lumi_benchmark_support.py",
        "validator_mutation_tests": SCRIPT_DIR / "test_photon_lumi_benchmark_validator.py",
        "fast_source": REPO_ROOT / "src" / "legacyPhotonLumiFastCPP.cc",
        "fast_header": REPO_ROOT / "src" / "legacyPhotonLumiFastCPP.h",
        "grid_source": REPO_ROOT / "src" / "gridCPP.cc",
        "switch_source": REPO_ROOT / "src" / "switchesCPP.cc",
        "ldd_output": provenance_dir / "ldd.txt",
        "lscpu_output": provenance_dir / "lscpu.txt",
    }
    artifacts: dict[str, dict[str, str]] = {}
    for name, source in sources.items():
        if name in {"benchmark_binary", "ldd_output", "lscpu_output"}:
            artifacts[name] = artifact(source, output)
        else:
            suffix = source.suffix or ".txt"
            destination = provenance_dir / "inputs" / f"{name}{suffix}"
            artifacts[name] = copy_artifact(source, destination, output)
    if working_tree_patch is not None:
        artifacts["working_tree_diff"] = artifact(
            working_tree_patch_path, output
        )

    build_manifest_path = provenance_dir / "build_manifest.json"
    build_manifest = {
        "schema": "feebl.legacy_photon_lumi_fast.build.v1",
        "compiler": args.compiler,
        "compile_flags": args.compile_flags,
        "binary_sha256": artifacts["benchmark_binary"]["sha256"],
        "git_revision": git["git_revision"],
        "git_dirty": git["git_dirty"],
        "git_status_sha256": git["git_status_sha256"],
        "working_tree_diff_sha256": (
            artifacts["working_tree_diff"]["sha256"]
            if working_tree_patch is not None else None
        ),
        "retained_artifact_sha256": {
            name: record["sha256"] for name, record in sorted(artifacts.items())
        },
        "retained_library_sha256": [
            row["retained"]["sha256"] for row in retained_libraries
        ],
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["build_manifest"] = artifact(build_manifest_path, output)

    governor = None
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor_path.is_file():
        governor = governor_path.read_text(encoding="utf-8").strip()
    environment = {
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": f"{{{pinned_cpu}}}",
        "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY", "<unset>"),
        "GOMP_CPU_AFFINITY": os.environ.get("GOMP_CPU_AFFINITY", "<unset>"),
        "KMP_AFFINITY": os.environ.get("KMP_AFFINITY", "<unset>"),
    }
    provenance = {
        **git,
        "compiler": args.compiler,
        "compile_flags": args.compile_flags,
        "command_template": [
            "taskset", "--cpu-list", str(pinned_cpu), "provenance/feebl",
            "--acc_file=<materialized-deck>", "tiny", "<variant>", "result.out",
        ],
        "performance_environment": environment,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "allowed_cpu_list": allowed_cpus,
            "physical_cpu_topology": physical_cpus,
            "pinned_cpu": pinned_cpu,
            "cpu_model": next(
                (line.split(":", 1)[1].strip() for line in lscpu_text.splitlines()
                 if line.startswith("Model name:")),
                "unavailable",
            ),
            "scaling_governor": governor,
            "wsl_interop": "WSL_INTEROP" in os.environ,
        },
        "artifacts": artifacts,
        "linked_libraries": retained_libraries,
    }

    template = (SCRIPT_DIR / "tiny.dat").read_text(encoding="utf-8")
    cases: list[dict[str, Any]] = []
    relative_gate = 2.0e-6
    absolute_gate = 2.0e-6
    for macroparticles in args.macroparticles:
        name = f"n{macroparticles}"
        case_root = output / "raw" / name
        case_root.mkdir(parents=True)
        deck = case_root / "benchmark.dat"
        materialize_deck(template, macroparticles, deck)
        case: dict[str, Any] = {
            "name": name,
            "macroparticles_per_beam": macroparticles,
            "deck": artifact(deck, output),
            "warmups": [],
            "samples": [],
        }
        for warmup in range(args.warmups):
            for order, variant in enumerate(("cartesian", "factorized")):
                sample = run_once(
                    copied_binary, deck, name, variant,
                    case_root / f"warmup-{warmup:02d}-{order}-{variant}",
                    pinned_cpu, warmup, order, True, output,
                )
                case["warmups"].append(sample)
        for pair in range(args.repetitions):
            order = ("cartesian", "factorized") if pair % 2 == 0 else (
                "factorized", "cartesian"
            )
            pair_samples: dict[str, dict[str, Any]] = {}
            for order_index, variant in enumerate(order):
                sample = run_once(
                    copied_binary, deck, name, variant,
                    case_root / f"pair-{pair:02d}-{order_index}-{variant}",
                    pinned_cpu, pair, order_index, False, output,
                )
                pair_samples[variant] = sample
                case["samples"].append(sample)
            validate_pair_in_runner(
                pair_samples["cartesian"], pair_samples["factorized"],
                relative_gate, absolute_gate,
            )
        cartesian_times = [
            sample["seconds"] for sample in case["samples"]
            if sample["variant"] == "cartesian"
        ]
        factorized_times = [
            sample["seconds"] for sample in case["samples"]
            if sample["variant"] == "factorized"
        ]
        paired = []
        for pair in range(args.repetitions):
            selected = {sample["variant"]: sample for sample in case["samples"]
                        if sample["pair"] == pair}
            paired.append(selected["cartesian"]["seconds"] /
                          selected["factorized"]["seconds"])
        cart_median = statistics.median(cartesian_times)
        fast_median = statistics.median(factorized_times)
        case["summary"] = {
            "cartesian_median_seconds": cart_median,
            "factorized_median_seconds": fast_median,
            "ratio_of_medians_speedup": cart_median / fast_median,
            "median_paired_speedup": statistics.median(paired),
            "cartesian_min_seconds": min(cartesian_times),
            "cartesian_max_seconds": max(cartesian_times),
            "factorized_min_seconds": min(factorized_times),
            "factorized_max_seconds": max(factorized_times),
        }
        cases.append(case)

    document = {
        "schema": support.SCHEMA,
        "valid": False,
        "validation_status": "unvalidated_staging",
        "classification": support.CLASSIFICATION,
        "eligible_for_project_level_2x_claim": False,
        "disclaimer": (
            "This is an exploratory end-to-end benchmark of one opt-in feature "
            "path, not a whole-program 2x claim."
        ),
        "evidence_authenticity": {
            "external_cryptographic_witness": False,
            "statement": (
                "Unkeyed local SHA-256 integrity checks do not provide "
                "independent cryptographic witness authenticity."
            ),
        },
        "evidence_root": ".",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_scope": (
            "process launch through normal FeeBL exit, including initialization, "
            "collision, output, and RNG-state save"
        ),
        "accuracy_gate_relative": relative_gate,
        "accuracy_gate_absolute": absolute_gate,
        "warmups_per_variant_case": args.warmups,
        "repetitions": args.repetitions,
        "provenance": provenance,
        "cases": cases,
    }
    staging_json = output / "photon_lumi_benchmark.staging.json"
    staging_csv = output / "photon_lumi_benchmark.staging.csv"
    final_json = output / "photon_lumi_benchmark.json"
    final_csv = output / "photon_lumi_benchmark.csv"
    staging_json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    support.write_csv(staging_csv, support.computed_rows(document))
    support.promote(staging_json, staging_csv, final_json, final_csv)
    print(json.dumps({
        "valid": True,
        "classification": support.CLASSIFICATION,
        "eligible_for_project_level_2x_claim": False,
        "evidence_directory": str(output),
        "cases": [
            {"name": case["name"], **case["summary"]} for case in cases
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, support.ValidationError) as exc:
        raise SystemExit(f"photon benchmark failed: {exc}") from exc
