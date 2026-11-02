#!/usr/bin/env python3
"""Run paired FeeBL seeds and perform preregistered equivalence checks.

This is an accuracy harness, not a speed benchmark. Each seed gets independent
serial/candidate directories and a deck copied from the requested source. Run
order alternates by seed. The tool preserves stdout, stderr, protocol outputs,
decks, hashes, scalar CSV, and a JSON analysis containing paired 90/95% CIs,
TOST decisions, KS statistics, and Wasserstein distances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import scipy
    from scipy import stats
except ImportError:  # pragma: no cover - documented fallback
    scipy = None
    stats = None


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
SCALAR_PATTERNS = {
    "lumi_fine": rf"(?m)^\s*lumi_fine\s*=\s*({NUMBER})",
    "lumi_ee": rf"(?m)^\s*lumi_ee\s*=\s*({NUMBER})",
    "lumi_ee_high": rf"(?m)^\s*lumi_ee_high\s*=\s*({NUMBER})",
    "lumi_eg": rf"(?m)^\s*lumi_eg\s*=\s*({NUMBER})",
    "lumi_ge": rf"(?m)^\s*lumi_ge\s*=\s*({NUMBER})",
    "lumi_gg": rf"(?m)^\s*lumi_gg\s*=\s*({NUMBER})",
    "upsmax": rf"(?m)^\s*upsmax\s*=\s*({NUMBER})",
}

REQUIRED_METRICS = {
    "lumi_fine",
    "lumi_ee",
    "lumi_ee_high",
    "lumi_eg",
    "lumi_ge",
    "lumi_gg",
    "upsmax",
    "beam1_energy_loss",
    "beam2_energy_loss",
    "beam1_miss",
    "beam2_miss",
    *(f"lumi_{row}_{column}" for row in range(3) for column in range(3)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inject_fragment(source: str, fragment: str, seed: int, parameters: str) -> str:
    """Append settings to exactly one named parameter block.

    Settings are appended so the forced seed/save controls win over values in
    the source deck. Candidate fragments are forbidden from spelling those
    controls, preventing an apparently paired comparison from changing them.
    """
    if re.search(r"\brndm_(?:seed|load|save)\s*=", fragment, re.IGNORECASE):
        raise ValueError("candidate fragment may not override paired RNG controls")
    pattern = re.compile(
        rf"(?ms)(^\$PARAMETERS::[ \t]*{re.escape(parameters)}[ \t]*\r?\n[ \t]*\{{)"
        rf"(.*?)"
        rf"(\}})"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one $PARAMETERS:: {parameters} block; found {len(matches)}"
        )
    normalized_fragment = fragment.strip()
    if normalized_fragment and not normalized_fragment.endswith(";"):
        normalized_fragment += ";"
    enforced = f"{normalized_fragment}rndm_seed={seed};rndm_load=0;rndm_save=0;"
    match = matches[0]
    replacement = match.group(1) + match.group(2) + enforced + match.group(3)
    return source[: match.start()] + replacement + source[match.end() :]


def parse_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, pattern in SCALAR_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            metrics[name] = float(match.group(1))

    for row, col, value in re.findall(rf"(?m)^\s*lumi\[(\d+)\]\[(\d+)\]\s*=\s*({NUMBER})", text):
        metrics[f"lumi_{row}_{col}"] = float(value)

    losses = re.findall(
        rf"average energy loss of the beam particles \(GeV\)\s*:\s*de\s*=\s*({NUMBER})",
        text,
    )
    for index, value in enumerate(losses[:2], 1):
        metrics[f"beam{index}_energy_loss"] = float(value)

    miss = re.search(
        rf"beam 1\s*:\s*miss\s*=\s*({NUMBER})\s+beam 2\s*:\s*miss\s*=\s*({NUMBER})",
        text,
    )
    if miss:
        metrics["beam1_miss"] = float(miss.group(1))
        metrics["beam2_miss"] = float(miss.group(2))
    return metrics


def run_one(
    binary: Path,
    source_deck: str,
    output_root: Path,
    label: str,
    fragment: str,
    seed: int,
    accelerator: str,
    parameters: str,
    expected_actual_backend: str | None,
    timeout_seconds: float,
) -> Dict[str, object]:
    run_dir = output_root / f"seed-{seed:04d}" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    deck = run_dir / "acc.dat"
    deck.write_text(inject_fragment(source_deck, fragment, seed, parameters), encoding="utf-8")
    command = [str(binary), accelerator, parameters, "result.out"]
    try:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        (run_dir / "timeout.txt").write_text(str(error) + "\n", encoding="utf-8")
        raise RuntimeError(f"{label} seed {seed} exceeded {timeout_seconds:g} seconds") from error
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    result = run_dir / "result.out"
    record: Dict[str, object] = {
        "seed": seed,
        "backend": label,
        "returncode": completed.returncode,
        "command": command,
        "deck_sha256": sha256(deck),
        "result_sha256": sha256(result) if result.exists() else "",
    }
    if completed.returncode != 0 or not result.exists():
        raise RuntimeError(
            f"{label} seed {seed} failed with rc={completed.returncode}; "
            f"see {run_dir / 'stderr.log'}"
        )
    result_text = result.read_text(encoding="utf-8", errors="replace")
    if expected_actual_backend is not None:
        backend_match = re.search(
            r"compute_backend_requested\s*=\s*(\w+)\s+" r"compute_backend_actual\s*=\s*(\w+)",
            result_text,
        )
        if not backend_match:
            raise RuntimeError(f"{label} seed {seed} did not emit backend provenance")
        if backend_match.group(2) != expected_actual_backend:
            raise RuntimeError(
                f"{label} seed {seed} selected {backend_match.group(2)!r}, "
                f"expected {expected_actual_backend!r}"
            )
        record["requested_backend"] = backend_match.group(1)
        record["actual_backend"] = backend_match.group(2)
    parsed = parse_metrics(result_text)
    missing = REQUIRED_METRICS - set(parsed)
    if missing:
        raise RuntimeError(
            f"{label} seed {seed} is missing required observables: " + ", ".join(sorted(missing))
        )
    nonfinite = sorted(name for name, value in parsed.items() if not math.isfinite(value))
    if nonfinite:
        raise RuntimeError(
            f"{label} seed {seed} has non-finite observables: " + ", ".join(nonfinite)
        )
    record.update(parsed)
    return record


def quantile(values: Sequence[float], probability: float) -> float:
    if stats is None:  # guarded in main; retained for direct-function callers
        raise RuntimeError("SciPy is required for Student-t confidence intervals")
    return float(stats.t.ppf(probability, len(values) - 1))


def distribution_checks(
    reference: Sequence[float], candidate: Sequence[float]
) -> Tuple[float, float, float]:
    if stats is not None:
        ks = stats.ks_2samp(reference, candidate, method="auto")
        wasserstein = float(stats.wasserstein_distance(reference, candidate))
        return float(ks.statistic), float(ks.pvalue), wasserstein
    ordered_reference = sorted(reference)
    ordered_candidate = sorted(candidate)
    wasserstein = statistics.fmean(abs(a - b) for a, b in zip(ordered_reference, ordered_candidate))
    return float("nan"), float("nan"), wasserstein


def analyze_metric(
    name: str,
    reference: Sequence[float],
    candidate: Sequence[float],
    relative_margin: float,
    absolute_margin: float,
) -> Dict[str, object]:
    differences = [b - a for a, b in zip(reference, candidate)]
    n = len(differences)
    mean_reference = statistics.fmean(reference)
    mean_candidate = statistics.fmean(candidate)
    mean_difference = statistics.fmean(differences)
    sd_difference = statistics.stdev(differences) if n > 1 else 0.0
    standard_error = sd_difference / math.sqrt(n) if n else float("nan")
    margin = max(absolute_margin, relative_margin * abs(mean_reference))
    if standard_error == 0.0:
        ci90 = [mean_difference, mean_difference]
        ci95 = [mean_difference, mean_difference]
    else:
        ci90_half = quantile(differences, 0.95) * standard_error
        ci95_half = quantile(differences, 0.975) * standard_error
        ci90 = [mean_difference - ci90_half, mean_difference + ci90_half]
        ci95 = [mean_difference - ci95_half, mean_difference + ci95_half]
    equivalent = ci90[0] > -margin and ci90[1] < margin
    ks_stat, ks_pvalue, wasserstein = distribution_checks(reference, candidate)
    return {
        "metric": name,
        "n": n,
        "reference_mean": mean_reference,
        "candidate_mean": mean_candidate,
        "paired_mean_difference": mean_difference,
        "paired_effect_relative": (
            mean_difference / mean_reference if mean_reference else mean_difference
        ),
        "paired_sd": sd_difference,
        "ci90_difference": ci90,
        "ci95_difference": ci95,
        "equivalence_margin_absolute": margin,
        "tost_equivalent": equivalent,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "wasserstein": wasserstein,
    }


def write_scalar_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-fragment", required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--candidate-expected-backend", required=True)
    parser.add_argument("--accelerator", default="c_nt1")
    parser.add_argument("--parameters", default="p_c_nt1")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--relative-margin", type=float, default=5.0e-6)
    parser.add_argument("--absolute-margin", type=float, default=1.0e-12)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if stats is None:
        parser.error("SciPy is required; use the pinned analysis environment")
    binary = args.binary.resolve(strict=True)
    deck = args.deck.resolve(strict=True)
    if args.seeds < 30:
        parser.error("statistical qualification requires at least 30 paired seeds")
    if args.first_seed < 0:
        parser.error("--first-seed must be non-negative")
    if args.relative_margin <= 0.0 or args.absolute_margin < 0.0:
        parser.error("equivalence margins must be positive/non-negative")
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    if args.candidate_label == "serial" or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", args.candidate_label
    ):
        parser.error("candidate label must be a safe name other than 'serial'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.candidate_expected_backend):
        parser.error("candidate expected backend must be a safe name")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be absent or empty (raw evidence is never reused)")
    args.output.mkdir(parents=True, exist_ok=True)
    source_deck = deck.read_text(encoding="utf-8")

    records: List[Dict[str, object]] = []
    for offset in range(args.seeds):
        seed = args.first_seed + offset
        order = [
            ("serial", ""),
            (args.candidate_label, args.candidate_fragment),
        ]
        if offset % 2:
            order.reverse()
        for label, fragment in order:
            records.append(
                run_one(
                    binary,
                    source_deck,
                    args.output,
                    label,
                    fragment,
                    seed,
                    args.accelerator,
                    args.parameters,
                    None if label == "serial" else args.candidate_expected_backend,
                    args.timeout_seconds,
                )
            )
        print(f"completed paired seed {seed}", flush=True)

    write_scalar_csv(args.output / "scalar_observables.csv", records)
    by_seed = {(int(row["seed"]), str(row["backend"])): row for row in records}
    metadata = {
        "seed",
        "backend",
        "returncode",
        "command",
        "deck_sha256",
        "result_sha256",
        "requested_backend",
        "actual_backend",
    }
    metric_sets = [set(row) - metadata for row in records]
    if any(item != metric_sets[0] for item in metric_sets[1:]):
        raise RuntimeError("parsed observable sets differ across paired runs")
    metric_names = sorted(metric_sets[0])
    analyses = []
    for metric in metric_names:
        reference = [
            float(by_seed[(args.first_seed + index, "serial")][metric])
            for index in range(args.seeds)
        ]
        candidate = [
            float(by_seed[(args.first_seed + index, args.candidate_label)][metric])
            for index in range(args.seeds)
        ]
        analyses.append(
            analyze_metric(
                metric,
                reference,
                candidate,
                args.relative_margin,
                args.absolute_margin,
            )
        )

    manifest = {
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source_deck": str(deck),
        "source_deck_sha256": sha256(deck),
        "candidate_fragment": args.candidate_fragment,
        "candidate_label": args.candidate_label,
        "candidate_expected_backend": args.candidate_expected_backend,
        "accelerator": args.accelerator,
        "parameters": args.parameters,
        "first_seed": args.first_seed,
        "seeds": args.seeds,
        "timeout_seconds": args.timeout_seconds,
        "relative_margin_preregistered": args.relative_margin,
        "absolute_margin_preregistered": args.absolute_margin,
        "python": sys.version,
        "platform": platform.platform(),
        "scipy": getattr(scipy, "__version__", "unavailable"),
        "all_metrics_tost_equivalent": all(bool(item["tost_equivalent"]) for item in analyses),
        "metrics": analyses,
    }
    (args.output / "equivalence.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "metrics": len(analyses),
                "all_metrics_tost_equivalent": manifest["all_metrics_tost_equivalent"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if manifest["all_metrics_tost_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
