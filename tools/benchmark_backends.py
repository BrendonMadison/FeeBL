#!/usr/bin/env python3
"""Reproducible, interleaved FeeBL backend benchmark driver.

The driver deliberately launches exactly one simulation at a time.  It records
raw logs, result hashes, process memory, optional GPU telemetry, and a machine
manifest.  Variant selection can use input-deck fragments and environment
variables so the same harness works with production and experimental backend
interfaces without modifying the source deck.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_NORMALIZE = [
    r"^\s*\*.*(?:time|version|compiled|backend|thread|device).*$",
    r"^\s*(?:elapsed|wall|cpu)\s+time\b.*$",
    r"^\s*FEEBL backend provenance\b.*$",
    r"^\s*(?:compute_backend_requested|cpu_variant_requested|"
    r"gpu_variant_requested|gpu_mode_requested|gpu2_[A-Za-z0-9_]+|backend_fallback|"
    r"backend_provenance)\b.*$",
]


@dataclass(frozen=True)
class Variant:
    name: str
    environment: dict[str, str]
    deck_fragment: str = ""


@dataclass
class Sample:
    case: str
    variant: str
    repetition: int
    warmup: bool
    order_in_round: int
    load1_before: float
    load1_after: float
    other_feebl_processes_before: int
    returncode: int
    wall_seconds: float
    user_seconds: float
    system_seconds: float
    peak_rss_kib: int
    peak_gpu_memory_mib: float | None
    mean_gpu_utilization_percent: float | None
    output_size: int
    output_sha256: str | None
    normalized_sha256: str | None
    input_deck_sha256: str
    run_directory: str
    observables: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_variant(text: str) -> Variant:
    name, separator, assignments = text.partition(":")
    name = name.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"invalid variant name: {name!r}")
    environment: dict[str, str] = {}
    if separator and assignments:
        for assignment in assignments.split(","):
            key, equals, value = assignment.partition("=")
            if not equals or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise argparse.ArgumentTypeError(
                    f"invalid environment assignment in {text!r}: {assignment!r}"
                )
            environment[key] = value
    return Variant(name=name, environment=environment)


def parse_deck_fragment(text: str) -> tuple[str, str]:
    """Parse NAME=FEEBL_PARAMETER_ASSIGNMENTS.

    This is deliberately a separate option from environment variants.  FeeBL's
    CPU backend is selected through its input deck, not through an environment
    alias, and silently benchmarking an ignored environment variable would
    compare the serial executable with itself.
    """
    name, equals, fragment = text.partition("=")
    name = name.strip()
    fragment = fragment.strip()
    if not equals or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not fragment:
        raise argparse.ArgumentTypeError("deck fragment must be NAME=PARAMETER_ASSIGNMENTS")
    if "\x00" in fragment:
        raise argparse.ArgumentTypeError("deck fragment contains a NUL byte")
    if not fragment.endswith(";"):
        fragment += ";"
    return name, fragment


def materialize_deck(source: bytes, parameter_block: str, fragment: str) -> bytes:
    """Return an independently materialized deck without mutating *source*.

    The fragment is inserted only into the parameter block used by the sample.
    Serial variants with an empty fragment retain the exact source bytes and
    therefore the exact source SHA-256.
    """
    if not fragment:
        return source
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("input deck is not UTF-8; cannot insert a fragment") from error
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


def run_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def source_git_root(source_root: Path) -> Path | None:
    """Return a Git root only when the source directory is that exact root."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root.resolve()), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            git_root = Path(completed.stdout.strip()).resolve()
            if git_root == source_root.resolve():
                return git_root
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def source_revision(source_root: Path) -> str:
    git_root = source_git_root(source_root)
    if git_root is not None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(git_root), "rev-parse", "--verify", "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    fallback = source_root.resolve() / "SOURCE_REVISION"
    try:
        first_line = fallback.read_text(encoding="utf-8").splitlines()[0].strip()
        if re.fullmatch(r"(?:[0-9a-fA-F]{40}(?:-dirty)?|sha256:[0-9a-fA-F]{64})", first_line):
            return first_line
        return "unknown"
    except (OSError, IndexError, UnicodeError):
        return "unknown"


def environment_manifest(
    binary: Path, deck: Path, source_root: Path, variants: Sequence[Variant], cases: Sequence[str]
) -> dict[str, object]:
    source = binary.resolve()
    manifest: dict[str, object] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "binary": str(source),
        "binary_sha256": sha256_file(source),
        "deck": str(deck.resolve()),
        "deck_sha256": sha256_file(deck.resolve()),
        "source_root": str(source_root.resolve()),
        "cases": list(cases),
        "variants": [asdict(variant) for variant in variants],
        "uname": run_text(["uname", "-a"]),
        "lscpu": run_text(["lscpu"]),
        "nvidia_smi": run_text(["nvidia-smi", "-q"]),
        "gcc": run_text(["g++", "--version"]),
        "cmake": run_text(["cmake", "--version"]),
    }
    revision = source_revision(source_root)
    manifest["source_revision"] = revision
    # An archive's content digest is a source identity, never a Git commit.
    is_content_digest = revision.startswith("sha256:")
    manifest["source_git_commit"] = None if is_content_digest else revision
    manifest["source_export_sha256"] = revision[7:].lower() if is_content_digest else None
    git_root = source_git_root(source_root)
    manifest["source_git_status"] = (
        run_text(["git", "-C", str(git_root), "status", "--short"])
        if git_root is not None
        else "not-a-source-repository"
    )
    return manifest


def normalize_output(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    lines = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in patterns):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def parse_observables(text: str, expressions: dict[str, re.Pattern[str]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, expression in expressions.items():
        match = expression.search(text)
        if match:
            try:
                values[name] = float(match.group(1))
            except (IndexError, ValueError):
                pass
    return values


def proc_tree_rss_kib(pid: int) -> int:
    total = 0
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(errors="replace")
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
            if match:
                total += int(match.group(1))
            children = Path(f"/proc/{current}/task/{current}/children")
            if children.exists():
                pending.extend(int(item) for item in children.read_text().split())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return total


def count_processes_named(name: str) -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == name:
                count += 1
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return count


def wait_for_idle(
    max_load1: float | None,
    max_other_feebl: int | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    if max_load1 is None and max_other_feebl is None:
        return
    start = time.monotonic()
    while True:
        load1 = os.getloadavg()[0]
        feebl_count = count_processes_named("feebl")
        load_ok = max_load1 is None or load1 <= max_load1
        process_ok = max_other_feebl is None or feebl_count <= max_other_feebl
        if load_ok and process_ok:
            return
        if time.monotonic() - start >= timeout_seconds:
            raise TimeoutError(
                f"idle gate timed out: load1={load1:.2f}, " f"other feebl processes={feebl_count}"
            )
        print(f"idle gate: load1={load1:.2f}, other feebl={feebl_count}; waiting", flush=True)
        time.sleep(poll_seconds)


def gpu_sample(device: int) -> tuple[float, float] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        first = completed.stdout.splitlines()[0]
        memory, utilization = (float(value.strip()) for value in first.split(",")[:2])
        return memory, utilization
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def monitor_process(
    pid: int,
    stop: threading.Event,
    interval: float,
    sample_gpu: bool,
    gpu_device: int,
    destination: dict[str, object],
) -> None:
    peak_rss = 0
    gpu_memory: list[float] = []
    gpu_utilization: list[float] = []
    while not stop.wait(interval):
        peak_rss = max(peak_rss, proc_tree_rss_kib(pid))
        if sample_gpu:
            sample = gpu_sample(gpu_device)
            if sample is not None:
                gpu_memory.append(sample[0])
                gpu_utilization.append(sample[1])
    peak_rss = max(peak_rss, proc_tree_rss_kib(pid))
    destination["peak_rss_kib"] = peak_rss
    destination["peak_gpu_memory_mib"] = max(gpu_memory) if gpu_memory else None
    destination["mean_gpu_utilization_percent"] = (
        statistics.fmean(gpu_utilization) if gpu_utilization else None
    )


def execute_sample(
    *,
    binary: Path,
    deck: Path,
    case: str,
    variant: Variant,
    repetition: int,
    warmup: bool,
    order_in_round: int,
    output_root: Path,
    affinity: str | None,
    timeout: float,
    telemetry_interval: float,
    gpu_telemetry: bool,
    gpu_device: int,
    max_load1: float | None,
    max_other_feebl: int | None,
    idle_poll: float,
    idle_timeout: float,
    normalize_patterns: Sequence[re.Pattern[str]],
    observable_patterns: dict[str, re.Pattern[str]],
) -> Sample:
    label = f"{'warmup' if warmup else 'run'}-{repetition:03d}-{order_in_round:02d}-{variant.name}-{case}"
    run_directory = output_root / "runs" / label
    run_directory.mkdir(parents=True, exist_ok=False)
    source_deck = deck.read_bytes()
    rendered_deck = materialize_deck(source_deck, f"p_{case}", variant.deck_fragment)
    input_deck = run_directory / "acc.dat"
    input_deck.write_bytes(rendered_deck)
    input_deck_sha256 = hashlib.sha256(rendered_deck).hexdigest()
    result = run_directory / "result.dat"

    command = [str(binary), case, f"p_{case}", result.name]
    if affinity:
        command = ["taskset", "-c", affinity, *command]
    environment = os.environ.copy()
    environment.update(variant.environment)
    environment["LC_ALL"] = "C"

    wait_for_idle(max_load1, max_other_feebl, idle_poll, idle_timeout)
    load1_before = os.getloadavg()[0]
    other_feebl = count_processes_named("feebl")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        cwd=run_directory,
        env=environment,
        stdout=(run_directory / "stdout.log").open("wb"),
        stderr=(run_directory / "stderr.log").open("wb"),
        start_new_session=True,
    )
    monitoring: dict[str, object] = {}
    stop = threading.Event()
    thread = threading.Thread(
        target=monitor_process,
        args=(process.pid, stop, telemetry_interval, gpu_telemetry, gpu_device, monitoring),
        daemon=True,
    )
    thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
        returncode = 124 if returncode == 0 else returncode
    finally:
        stop.set()
        thread.join(timeout=max(2.0, telemetry_interval * 2))
    wall = (time.perf_counter_ns() - start) / 1.0e9
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    output_text = result.read_text(errors="replace") if result.exists() else ""
    normalized = normalize_output(output_text, normalize_patterns) if output_text else ""
    sample = Sample(
        case=case,
        variant=variant.name,
        repetition=repetition,
        warmup=warmup,
        order_in_round=order_in_round,
        returncode=returncode,
        load1_before=load1_before,
        load1_after=os.getloadavg()[0],
        other_feebl_processes_before=other_feebl,
        wall_seconds=wall,
        user_seconds=after.ru_utime - before.ru_utime,
        system_seconds=after.ru_stime - before.ru_stime,
        peak_rss_kib=int(monitoring.get("peak_rss_kib", 0)),
        peak_gpu_memory_mib=monitoring.get("peak_gpu_memory_mib"),
        mean_gpu_utilization_percent=monitoring.get("mean_gpu_utilization_percent"),
        output_size=result.stat().st_size if result.exists() else 0,
        output_sha256=sha256_file(result) if result.exists() else None,
        normalized_sha256=(hashlib.sha256(normalized.encode()).hexdigest() if normalized else None),
        input_deck_sha256=input_deck_sha256,
        run_directory=str(run_directory),
        observables=parse_observables(output_text, observable_patterns),
    )
    (run_directory / "sample.json").write_text(json.dumps(asdict(sample), indent=2) + "\n")
    if returncode != 0 or not result.exists() or result.stat().st_size == 0:
        raise RuntimeError(f"{label} failed: exit={returncode}, output_exists={result.exists()}")
    return sample


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def bootstrap_interval(
    values: Sequence[float], statistic, seed: int, draws: int = 10000
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws)
    )
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]


def summarize(samples: Sequence[Sample], baseline: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        if not sample.warmup:
            groups.setdefault((sample.case, sample.variant), []).append(sample)
    rows: list[dict[str, object]] = []
    for (case, variant), group in sorted(groups.items()):
        values = [sample.wall_seconds for sample in group]
        median = statistics.median(values)
        low, high = bootstrap_interval(values, statistics.median, seed=20260825)
        base_group = groups.get((case, baseline), [])
        base_values = [sample.wall_seconds for sample in base_group]
        base_median = statistics.median(base_values) if base_values else math.nan
        # Samples are deliberately interleaved by repetition. Preserve that
        # pairing in the speedup estimator and confidence interval so transient
        # machine load affects numerator and denominator together.
        base_by_repetition = {sample.repetition: sample.wall_seconds for sample in base_group}
        paired_speedups = [
            base_by_repetition[sample.repetition] / sample.wall_seconds
            for sample in group
            if sample.repetition in base_by_repetition
        ]
        speedup = statistics.median(paired_speedups) if paired_speedups else math.nan
        if paired_speedups:
            speed_low, speed_high = bootstrap_interval(
                paired_speedups, statistics.median, seed=20260825
            )
        else:
            speed_low = speed_high = math.nan
        rows.append(
            {
                "case": case,
                "variant": variant,
                "samples": len(values),
                "paired_samples": len(paired_speedups),
                "median_seconds": median,
                "mad_seconds": median_absolute_deviation(values),
                "median_ci95_low": low,
                "median_ci95_high": high,
                "ratio_of_medians_vs_baseline": (base_median / median if base_values else math.nan),
                "speedup_vs_baseline": speedup,
                "speedup_ci95_low": speed_low,
                "speedup_ci95_high": speed_high,
            }
        )
    return rows


def write_summary(rows: Sequence[dict[str, object]], destination: Path) -> None:
    if not rows:
        return
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_observable(text: str) -> tuple[str, re.Pattern[str]]:
    name, equals, expression = text.partition("=")
    if not equals or not name:
        raise argparse.ArgumentTypeError("observable must be NAME=REGEX_WITH_CAPTURE")
    return name, re.compile(expression, re.MULTILINE)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--deck", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="cases", required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument(
        "--deck-fragment",
        action="append",
        type=parse_deck_fragment,
        default=[],
        metavar="NAME=ASSIGNMENTS",
        help=(
            "insert assignments into the selected p_CASE block for NAME; "
            "the source deck is never modified"
        ),
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source tree used for Git provenance (default: script parent)",
    )
    parser.add_argument("--affinity", help="taskset CPU list, for example 0-9")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--gpu-telemetry", action="store_true")
    parser.add_argument(
        "--gpu-device", type=int, default=0, help="physical nvidia-smi device index for telemetry"
    )
    parser.add_argument(
        "--max-load1", type=float, help="wait until one-minute load is at or below this value"
    )
    parser.add_argument(
        "--max-other-feebl",
        type=int,
        help="wait until at most this many pre-existing feebl processes run",
    )
    parser.add_argument("--idle-poll", type=float, default=30.0)
    parser.add_argument("--idle-timeout", type=float, default=21600.0)
    parser.add_argument("--normalize-regex", action="append", default=[])
    parser.add_argument("--observable", action="append", type=parse_observable, default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.repetitions < 1 or args.warmups < 0:
        raise SystemExit("repetitions must be >=1 and warmups >=0")
    if args.gpu_device < 0:
        raise SystemExit("gpu device must be non-negative")
    names = [variant.name for variant in args.variant]
    if len(names) != len(set(names)):
        raise SystemExit("variant names must be unique")
    if args.baseline not in names:
        raise SystemExit("baseline must name one of the variants")
    fragments: dict[str, str] = {}
    for name, fragment in args.deck_fragment:
        if name in fragments:
            raise SystemExit(f"duplicate deck fragment for variant {name!r}")
        fragments[name] = fragment
    unknown_fragments = sorted(set(fragments) - set(names))
    if unknown_fragments:
        raise SystemExit(
            "deck fragment names do not match declared variants: " + ", ".join(unknown_fragments)
        )
    variants = [
        replace(variant, deck_fragment=fragments.get(variant.name, "")) for variant in args.variant
    ]
    binary = args.binary.resolve()
    deck = args.deck.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"binary is missing or not executable: {binary}")
    if not deck.is_file():
        raise SystemExit(f"deck is missing: {deck}")
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root is missing: {source_root}")

    # Validate every case/variant deck before creating the output directory, so
    # a typo cannot leave a partial benchmark that looks complete.
    base_deck = deck.read_bytes()
    deck_hashes: dict[str, dict[str, str]] = {}
    for case in args.cases:
        deck_hashes[case] = {}
        for variant in variants:
            rendered = materialize_deck(base_deck, f"p_{case}", variant.deck_fragment)
            deck_hashes[case][variant.name] = hashlib.sha256(rendered).hexdigest()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = environment_manifest(binary, deck, source_root, variants, args.cases)
    manifest["materialized_deck_sha256"] = deck_hashes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    normalize_patterns = [
        re.compile(pattern) for pattern in [*DEFAULT_NORMALIZE, *args.normalize_regex]
    ]
    observables = dict(args.observable)
    if not observables:
        observables = {
            "lumi_01": re.compile(r"lumi\[0\]\[1\]\s*=\s*([-+0-9.eE]+)"),
            "lumi_fine": re.compile(r"lumi_fine\s*=\s*([-+0-9.eE]+)"),
            "upsmax": re.compile(r"upsmax\s*=\s*([-+0-9.eE]+)"),
        }

    samples: list[Sample] = []
    jsonl = output / "samples.jsonl"
    total_rounds = args.warmups + args.repetitions
    for case_index, case in enumerate(args.cases):
        for round_index in range(total_rounds):
            warmup = round_index < args.warmups
            repetition = round_index if warmup else round_index - args.warmups
            rotation = (round_index + case_index) % len(variants)
            ordered = variants[rotation:] + variants[:rotation]
            for order_in_round, variant in enumerate(ordered):
                sample = execute_sample(
                    binary=binary,
                    deck=deck,
                    case=case,
                    variant=variant,
                    repetition=repetition,
                    warmup=warmup,
                    order_in_round=order_in_round,
                    output_root=output,
                    affinity=args.affinity,
                    timeout=args.timeout,
                    telemetry_interval=args.telemetry_interval,
                    gpu_telemetry=args.gpu_telemetry,
                    gpu_device=args.gpu_device,
                    max_load1=args.max_load1,
                    max_other_feebl=args.max_other_feebl,
                    idle_poll=args.idle_poll,
                    idle_timeout=args.idle_timeout,
                    normalize_patterns=normalize_patterns,
                    observable_patterns=observables,
                )
                samples.append(sample)
                with jsonl.open("a") as stream:
                    stream.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
                print(
                    f"{case:18s} {variant.name:16s} {sample.wall_seconds:10.4f}s "
                    f"rss={sample.peak_rss_kib / 1024:8.1f}MiB "
                    f"hash={sample.normalized_sha256 or '-'}",
                    flush=True,
                )

    rows = summarize(samples, args.baseline)
    write_summary(rows, output / "summary.csv")
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
