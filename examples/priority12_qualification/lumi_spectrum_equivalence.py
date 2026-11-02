#!/usr/bin/env python3
"""Uncertainty-aware equivalence checks for FeeBL dL/dW spectra.

The comparison deliberately uses the integrated ``weight`` column of the
native pair-spectrum file.  Comparing ``dL_dW`` bin by bin with a relative
epsilon is unstable in empty tails and is not a statistical qualification.
Instead, this tool checks luminosity normalization, moments, distribution
distances, and paired-seed confidence intervals after a declared coarse
graining.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Sequence


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
META_VALUE = re.compile(rf"^#(underflow|overflow)\s+({NUMBER})\s*$")


@dataclass(frozen=True)
class Spectrum:
    path: Path
    low: tuple[float, ...]
    high: tuple[float, ...]
    weights: tuple[float, ...]
    underflow: float
    overflow: float

    @property
    def in_range_total(self) -> float:
        return math.fsum(self.weights)

    @property
    def total(self) -> float:
        return self.in_range_total + self.underflow + self.overflow

    @property
    def probabilities(self) -> tuple[float, ...]:
        total = self.in_range_total
        return tuple(value / total for value in self.weights)

    @property
    def centers(self) -> tuple[float, ...]:
        return tuple(0.5 * (a + b) for a, b in zip(self.low, self.high))

    @property
    def mean(self) -> float:
        return math.fsum(
            center * probability
            for center, probability in zip(self.centers, self.probabilities)
        )

    @property
    def rms(self) -> float:
        mean = self.mean
        return math.sqrt(math.fsum(
            probability * (center - mean) ** 2
            for center, probability in zip(self.centers, self.probabilities)
        ))


@dataclass(frozen=True)
class Thresholds:
    total_relative: float = 0.01
    mean_shift_sigma: float = 0.02
    rms_relative: float = 0.01
    ks: float = 0.005
    wasserstein_sigma: float = 0.01
    coarse_tv: float = 0.01
    leakage_absolute: float = 0.001
    coarse_width: float = 0.010
    coarse_bin_relative: float = 0.03
    coarse_bin_absolute: float = 5.0e-5
    minimum_bin_probability: float = 1.0e-3
    minimum_bin_pass_fraction: float = 0.95


def read_spectrum(path: Path) -> Spectrum:
    low: list[float] = []
    high: list[float] = []
    weights: list[float] = []
    leakage = {"underflow": 0.0, "overflow": 0.0}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = META_VALUE.match(line)
            if match:
                leakage[match.group(1)] = float(match.group(2))
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: expected at least four columns")
        try:
            lo, hi, weight = float(fields[0]), float(fields[1]), float(fields[3])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric spectrum row") from error
        if not all(math.isfinite(value) for value in (lo, hi, weight)):
            raise ValueError(f"{path}:{line_number}: non-finite spectrum value")
        if hi <= lo or weight < 0.0:
            raise ValueError(f"{path}:{line_number}: invalid bin or negative weight")
        if low and lo < high[-1] - 1.0e-12:
            raise ValueError(f"{path}:{line_number}: bins overlap or are unordered")
        low.append(lo)
        high.append(hi)
        weights.append(weight)
    if not weights or math.fsum(weights) <= 0.0:
        raise ValueError(f"{path}: spectrum has no positive in-range weight")
    for name, value in leakage.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}: invalid {name}")
    return Spectrum(
        path, tuple(low), tuple(high), tuple(weights),
        leakage["underflow"], leakage["overflow"],
    )


def require_same_bins(reference: Spectrum, candidate: Spectrum) -> None:
    if len(reference.weights) != len(candidate.weights):
        raise ValueError(
            f"bin-count mismatch: {reference.path} has {len(reference.weights)}, "
            f"{candidate.path} has {len(candidate.weights)}"
        )
    for index, (rlo, rhi, clo, chi) in enumerate(zip(
        reference.low, reference.high, candidate.low, candidate.high
    )):
        if not (math.isclose(rlo, clo, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(rhi, chi, rel_tol=0.0, abs_tol=1.0e-12)):
            raise ValueError(f"bin-edge mismatch at zero-based bin {index}")


def aggregate(spectra: Sequence[Spectrum], label: str) -> Spectrum:
    if not spectra:
        raise ValueError(f"no {label} spectra supplied")
    first = spectra[0]
    for spectrum in spectra[1:]:
        require_same_bins(first, spectrum)
    weights = tuple(math.fsum(row) for row in zip(*(item.weights for item in spectra)))
    return Spectrum(
        Path(f"<{label}-aggregate>"), first.low, first.high, weights,
        math.fsum(item.underflow for item in spectra),
        math.fsum(item.overflow for item in spectra),
    )


def coarse_probabilities(spectrum: Spectrum, width: float) -> dict[int, float]:
    if width <= 0.0:
        raise ValueError("coarse width must be positive")
    origin = spectrum.low[0]
    buckets: dict[int, float] = {}
    for center, probability in zip(spectrum.centers, spectrum.probabilities):
        bucket = math.floor((center - origin) / width + 1.0e-12)
        buckets[bucket] = buckets.get(bucket, 0.0) + probability
    return buckets


def distribution_metrics(reference: Spectrum, candidate: Spectrum,
                         thresholds: Thresholds) -> dict[str, float]:
    require_same_bins(reference, candidate)
    if reference.rms <= 0.0:
        raise ValueError(f"{reference.path}: reference spectrum has zero RMS width")
    ref_p = reference.probabilities
    cand_p = candidate.probabilities
    ref_cdf = 0.0
    cand_cdf = 0.0
    ks = 0.0
    wasserstein = 0.0
    for lo, hi, left, right in zip(reference.low, reference.high, ref_p, cand_p):
        ref_cdf += left
        cand_cdf += right
        delta = abs(ref_cdf - cand_cdf)
        ks = max(ks, delta)
        wasserstein += delta * (hi - lo)
    ref_coarse = coarse_probabilities(reference, thresholds.coarse_width)
    cand_coarse = coarse_probabilities(candidate, thresholds.coarse_width)
    buckets = set(ref_coarse) | set(cand_coarse)
    tv = 0.5 * math.fsum(
        abs(ref_coarse.get(key, 0.0) - cand_coarse.get(key, 0.0))
        for key in buckets
    )
    ref_leakage = (reference.underflow + reference.overflow) / reference.total
    cand_leakage = (candidate.underflow + candidate.overflow) / candidate.total
    return {
        "total_relative_effect": candidate.total / reference.total - 1.0,
        "mean_shift_reference_sigma": (
            (candidate.mean - reference.mean) / reference.rms
        ),
        "rms_relative_effect": candidate.rms / reference.rms - 1.0,
        "ks_distance": ks,
        "wasserstein_reference_sigma": wasserstein / reference.rms,
        "coarse_total_variation": tv,
        "leakage_fraction_effect": cand_leakage - ref_leakage,
        "reference_total": reference.total,
        "candidate_total": candidate.total,
        "reference_mean": reference.mean,
        "candidate_mean": candidate.mean,
        "reference_rms": reference.rms,
        "candidate_rms": candidate.rms,
    }


# Two-sided 90% confidence interval: t_(0.95, df).
T90 = (
    6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833,
    1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734,
    1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703,
    1.701, 1.699, 1.697,
)


def confidence_interval_90(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot form a confidence interval from no values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    error = statistics.stdev(values) / math.sqrt(len(values))
    critical = T90[len(values) - 2] if len(values) <= 31 else 1.645
    return mean - critical * error, mean + critical * error


def interval_inside(interval: Sequence[float], margin: float) -> bool:
    return interval[0] >= -margin and interval[1] <= margin


def compare_ensemble(reference: Sequence[Spectrum], candidate: Sequence[Spectrum],
                     thresholds: Thresholds = Thresholds()) -> dict[str, object]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("reference and candidate must contain the same nonzero pair count")
    for left, right in zip(reference, candidate):
        require_same_bins(left, right)

    aggregate_reference = aggregate(reference, "reference")
    aggregate_candidate = aggregate(candidate, "candidate")
    aggregate_metrics = distribution_metrics(
        aggregate_reference, aggregate_candidate, thresholds
    )
    paired_effects = {
        "total_relative": [right.total / left.total - 1.0
                           for left, right in zip(reference, candidate)],
        "mean_shift_sigma": [(right.mean - left.mean) / left.rms
                             for left, right in zip(reference, candidate)],
        "rms_relative": [right.rms / left.rms - 1.0
                         for left, right in zip(reference, candidate)],
    }
    paired_limits = {
        "total_relative": thresholds.total_relative,
        "mean_shift_sigma": thresholds.mean_shift_sigma,
        "rms_relative": thresholds.rms_relative,
    }
    paired_intervals = {
        name: {
            "mean": statistics.fmean(values),
            "ci90": list(confidence_interval_90(values)),
            "margin": paired_limits[name],
            "equivalent": interval_inside(
                confidence_interval_90(values), paired_limits[name]
            ),
        }
        for name, values in paired_effects.items()
    }

    coarse_reference = [coarse_probabilities(item, thresholds.coarse_width)
                        for item in reference]
    coarse_candidate = [coarse_probabilities(item, thresholds.coarse_width)
                        for item in candidate]
    active_buckets = sorted(set().union(*(item.keys() for item in coarse_reference)))
    bin_checks: list[dict[str, object]] = []
    for bucket in active_buckets:
        ref_values = [item.get(bucket, 0.0) for item in coarse_reference]
        cand_values = [item.get(bucket, 0.0) for item in coarse_candidate]
        reference_mean = statistics.fmean(ref_values)
        if reference_mean < thresholds.minimum_bin_probability:
            continue
        differences = [right - left for left, right in zip(ref_values, cand_values)]
        interval = confidence_interval_90(differences)
        margin = max(
            thresholds.coarse_bin_absolute,
            thresholds.coarse_bin_relative * reference_mean,
        )
        standard_error = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0
        )
        bin_checks.append({
            "bucket": bucket,
            "reference_probability": reference_mean,
            "paired_mean_difference": statistics.fmean(differences),
            "ci90": list(interval),
            "equivalence_margin": margin,
            "equivalent": interval_inside(interval, margin),
            "standardized_pull": (
                statistics.fmean(differences) / standard_error
                if standard_error > 0.0 else (0.0 if not any(differences) else None)
            ),
        })
    bin_pass_fraction = (
        sum(bool(item["equivalent"]) for item in bin_checks) / len(bin_checks)
        if bin_checks else 1.0
    )

    aggregate_limits = {
        "total_relative_effect": thresholds.total_relative,
        "mean_shift_reference_sigma": thresholds.mean_shift_sigma,
        "rms_relative_effect": thresholds.rms_relative,
        "ks_distance": thresholds.ks,
        "wasserstein_reference_sigma": thresholds.wasserstein_sigma,
        "coarse_total_variation": thresholds.coarse_tv,
        "leakage_fraction_effect": thresholds.leakage_absolute,
    }
    aggregate_pass = {
        name: abs(float(aggregate_metrics[name])) <= limit
        for name, limit in aggregate_limits.items()
    }
    status = (
        all(aggregate_pass.values())
        and all(bool(item["equivalent"]) for item in paired_intervals.values())
        and bin_pass_fraction >= thresholds.minimum_bin_pass_fraction
    )
    return {
        "schema": "feebl.lumi-spectrum-equivalence.v1",
        "status": "PASS" if status else "FAIL",
        "pair_count": len(reference),
        "thresholds": thresholds.__dict__,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_pass": aggregate_pass,
        "paired_effect_intervals": paired_intervals,
        "coarse_bin_equivalence": {
            "tested_bins": len(bin_checks),
            "passing_bins": sum(bool(item["equivalent"]) for item in bin_checks),
            "pass_fraction": bin_pass_fraction,
            "required_fraction": thresholds.minimum_bin_pass_fraction,
            "bins": bin_checks,
        },
    }


def paths(values: Iterable[Sequence[str]]) -> tuple[list[Path], list[Path]]:
    reference: list[Path] = []
    candidate: list[Path] = []
    for left, right in values:
        reference.append(Path(left).resolve(strict=True))
        candidate.append(Path(right).resolve(strict=True))
    return reference, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair", nargs=2, action="append", required=True,
        metavar=("REFERENCE", "CANDIDATE"),
        help="paired lumi.ee.out.dLdW.fne files; repeat once per seed",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--total-relative", type=float, default=0.01)
    parser.add_argument("--mean-shift-sigma", type=float, default=0.02)
    parser.add_argument("--rms-relative", type=float, default=0.01)
    parser.add_argument("--ks", type=float, default=0.005)
    parser.add_argument("--wasserstein-sigma", type=float, default=0.01)
    parser.add_argument("--coarse-tv", type=float, default=0.01)
    parser.add_argument("--leakage-absolute", type=float, default=0.001)
    parser.add_argument("--coarse-width", type=float, default=0.010)
    parser.add_argument("--coarse-bin-relative", type=float, default=0.03)
    parser.add_argument("--coarse-bin-absolute", type=float, default=5.0e-5)
    parser.add_argument("--minimum-bin-probability", type=float, default=1.0e-3)
    parser.add_argument("--minimum-bin-pass-fraction", type=float, default=0.95)
    args = parser.parse_args()
    thresholds = Thresholds(**{
        name: getattr(args, name)
        for name in Thresholds.__dataclass_fields__
    })
    if any(value < 0.0 for value in thresholds.__dict__.values()):
        parser.error("all thresholds must be non-negative")
    if thresholds.coarse_width == 0.0:
        parser.error("--coarse-width must be positive")
    if not 0.0 <= thresholds.minimum_bin_pass_fraction <= 1.0:
        parser.error("--minimum-bin-pass-fraction must be in [0,1]")
    reference_paths, candidate_paths = paths(args.pair)
    result = compare_ensemble(
        [read_spectrum(path) for path in reference_paths],
        [read_spectrum(path) for path in candidate_paths],
        thresholds,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
