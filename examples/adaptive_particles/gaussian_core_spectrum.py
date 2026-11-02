#!/usr/bin/env python3
"""Extract a beamstrahlung-resistant Gaussian core from a FeeBL spectrum.

The primary estimator is a deterministic quadratic fit to the logarithm of
the *upper* flank of the luminosity-spectrum peak.  Beamstrahlung moves
luminosity to lower invariant mass, so the upper flank is substantially less
sensitive to the radiative tail than a moment over the complete spectrum.

Only the Python standard library is used.  The numerical and quality-control
constants below are part of the versioned method; the command line cannot tune
them after a spectrum has been examined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCHEMA = "feebl.gaussian-core-spectrum.v1"
METHOD_ID = "upper-flank-log-quadratic-v5"

SMOOTHING_KERNEL = (1.0, 4.0, 6.0, 4.0, 1.0)
SMOOTHING_DENOMINATOR = 16.0
MAIN_LOWER_SIGMA = 0.25
MAIN_UPPER_SIGMA = 2.50
STABILITY_UPPER_SIGMAS = (2.00, 3.00)
MINIMUM_DENSITY_FRACTION = 0.01
MINIMUM_FIT_BINS = 20
MAXIMUM_ITERATIONS = 10
MU_CONVERGENCE_GEV = 1.0e-5
SIGMA_RELATIVE_CONVERGENCE = 1.0e-4
MINIMUM_LOG_R_SQUARED = 0.995
MAXIMUM_ABS_LOG_RESIDUAL = 0.15
MAXIMUM_UNDER_OVERFLOW_FRACTION = 1.0e-3
MAXIMUM_WINDOW_MU_SIGMA_FRACTION = 5.0e-2
MAXIMUM_WINDOW_SIGMA_RELATIVE_SHIFT = 3.0e-2
MAXIMUM_REBIN_MU_SHIFT_GEV = 5.0e-4
MAXIMUM_REBIN_SIGMA_RELATIVE_SHIFT = 5.0e-3
MODE_VERTEX_TOLERANCE_SIGMA = 0.25
MINIMUM_UPPER_BOUND_SIGMAS = 3.0
BIN_UNIFORMITY_RELATIVE_TOLERANCE = 1.0e-9
BIN_CONTIGUITY_ABSOLUTE_TOLERANCE_GEV = 1.0e-10
MAXIMUM_CONFIGURED_BINS = 10_000_000
LATTICE_EDGE_RELATIVE_TOLERANCE = 1.0e-6
LATTICE_EDGE_BIN_FRACTION_TOLERANCE = 0.2
# Native pair-spectrum rows print both dL/dW and the integrated bin weight at
# limited decimal precision.  Two independently rounded columns can differ by
# several parts in 10^6 even when they originate from the same accumulator.
# These tolerances are therefore frozen format-authentication tolerances, not
# adjustable physics cuts.
ROW_WEIGHT_RELATIVE_TOLERANCE = 2.0e-5
TOTAL_LUMINOSITY_RELATIVE_TOLERANCE = 2.0e-5

NUMBER_PATTERN = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
UNSIGNED_NUMBER_PATTERN = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
NUMBER = re.compile(rf"^{NUMBER_PATTERN}$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class SpectrumFitError(RuntimeError):
    """A malformed spectrum or a failed frozen fit-quality gate."""


@dataclass(frozen=True)
class SpectrumBin:
    low: float
    high: float
    density: float
    weight: float

    @property
    def center(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def luminosity(self) -> float:
        # The native weight column is the conserved integrated quantity.  The
        # independently printed density is authenticated against it on input.
        return self.weight


@dataclass(frozen=True)
class SpectrumExpectation:
    """Controller-owned identity and histogram-lattice contract.

    FeeBL intentionally omits empty histogram rows.  The configured range and
    bin count are therefore required to recover leading/trailing zeros and the
    parity used by the adjacent-pair rebin stability check.
    """

    rndm_seed: int
    w_min_gev: float
    w_max_gev: float
    configured_bins: int
    expected_total_luminosity: float
    run_id: str | None = None

    def validated(self) -> "SpectrumExpectation":
        if (
            isinstance(self.rndm_seed, bool)
            or not isinstance(self.rndm_seed, int)
            or not (1 <= self.rndm_seed <= 2_147_483_647)
        ):
            raise SpectrumFitError("expected rndm_seed must be an integer in [1,2147483647]")
        if not math.isfinite(self.w_min_gev) or not math.isfinite(self.w_max_gev):
            raise SpectrumFitError("expected spectrum range must be finite")
        if not self.w_max_gev > self.w_min_gev:
            raise SpectrumFitError("expected spectrum_w_max must exceed spectrum_w_min")
        if (
            isinstance(self.configured_bins, bool)
            or not isinstance(self.configured_bins, int)
            or not (MINIMUM_FIT_BINS + 6 <= self.configured_bins <= MAXIMUM_CONFIGURED_BINS)
        ):
            raise SpectrumFitError(
                f"expected spectrum_w_bins must be in [{MINIMUM_FIT_BINS + 6},"
                f"{MAXIMUM_CONFIGURED_BINS}]"
            )
        if self.run_id is not None and SAFE_RUN_ID.fullmatch(self.run_id) is None:
            raise SpectrumFitError("expected run id is not a safe identifier")
        if not math.isfinite(self.expected_total_luminosity) or self.expected_total_luminosity <= 0:
            raise SpectrumFitError("expected total luminosity must be finite and positive")
        return self


@dataclass(frozen=True)
class Spectrum:
    bins: tuple[SpectrumBin, ...]
    underflow: float
    overflow: float
    footer_mean_w: float | None
    footer_rms_w: float | None
    footer_n_eff_total: float | None
    input_sha256: str
    input_name: str
    native_format_version: str
    feebl_version: str
    rndm_seed: int
    declared_record_count: int
    lattice_phase_proven: bool
    configured_w_min_gev: float | None
    configured_w_max_gev: float | None
    configured_bin_count: int | None
    maximum_lattice_edge_residual_gev: float | None
    maximum_row_weight_relative_residual: float
    expected_total_luminosity: float | None
    total_luminosity_relative_residual: float | None
    expected_run_id: str | None

    @property
    def bin_width(self) -> float:
        return self.bins[0].width

    @property
    def in_range_luminosity(self) -> float:
        return math.fsum(item.luminosity for item in self.bins)

    @property
    def total_luminosity(self) -> float:
        return self.in_range_luminosity + self.underflow + self.overflow

    @property
    def under_overflow_fraction(self) -> float:
        return (self.underflow + self.overflow) / self.total_luminosity


@dataclass(frozen=True)
class RawFit:
    mu_gev: float
    sigma_core_gev: float
    sigma_bin_gev: float
    peak_density: float
    core_area: float
    iteration_count: int
    fit_bin_count: int
    lower_sigma: float
    upper_sigma: float
    log_r_squared: float
    log_residual_rms: float
    maximum_abs_log_residual: float
    mu_covariance_se_gev: float
    sigma_covariance_se_gev: float
    modal_center_gev: float
    window_low_gev: float
    window_high_gev: float


def _finite_number(token: str, where: str) -> float:
    if NUMBER.fullmatch(token) is None:
        raise SpectrumFitError(f"{where}: invalid numeric token {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise SpectrumFitError(f"{where}: non-finite value")
    return value


def _one_prefixed_line(
    lines: Sequence[str], prefix: str, pattern: re.Pattern[str], where: str
) -> tuple[re.Match[str], int]:
    candidates = [(index, line) for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(candidates) != 1:
        raise SpectrumFitError(f"{where}: expected exactly one {prefix!r} line")
    index, line = candidates[0]
    match = pattern.fullmatch(line)
    if match is None:
        raise SpectrumFitError(f"{where}: malformed line {line!r}")
    return match, index


def _one_footer(lines: Sequence[str], key: str) -> tuple[float, int]:
    if key == "mean_W":
        # nativeOutputCPP writes the mean and its standard error together.
        # Requiring the uncertainty also prevents a truncated ``+-`` suffix
        # from being mistaken for a valid scalar footer.
        pattern = re.compile(
            rf"#mean_W[ \t]+({NUMBER_PATTERN})" rf"[ \t]+\+\-[ \t]+({UNSIGNED_NUMBER_PATTERN})"
        )
    elif key in {"underflow", "overflow", "rms_W", "n_eff_total"}:
        # These four native footers are scalar records.  In particular, do
        # not accept a writer-incompatible uncertainty or any other suffix.
        pattern = re.compile(rf"#{re.escape(key)}[ \t]+({NUMBER_PATTERN})")
    else:  # Internal programming error rather than an input-format error.
        raise ValueError(f"unknown pair_spectrum footer {key!r}")
    match, index = _one_prefixed_line(
        lines,
        f"#{key}",
        pattern,
        f"#{key} footer",
    )
    return _finite_number(match.group(1), f"#{key}"), index


def _unbound_dense_bins(raw_bins: Sequence[SpectrumBin]) -> tuple[tuple[SpectrumBin, ...], float]:
    first_width = raw_bins[0].width
    maximum_residual = 0.0
    bins: list[SpectrumBin] = []
    for index, item in enumerate(raw_bins):
        width_residual = abs(item.width - first_width)
        maximum_residual = max(maximum_residual, width_residual)
        if width_residual > BIN_UNIFORMITY_RELATIVE_TOLERANCE * first_width:
            raise SpectrumFitError(f"bin {index}: nonuniform bin width")
        if bins:
            gap = item.low - bins[-1].high
            maximum_residual = max(maximum_residual, abs(gap))
            if gap < -BIN_CONTIGUITY_ABSOLUTE_TOLERANCE_GEV:
                raise SpectrumFitError(f"bin {index}: bins overlap or are out of order")
            if abs(gap) > BIN_CONTIGUITY_ABSOLUTE_TOLERANCE_GEV:
                raise SpectrumFitError(
                    "sparse spectrum has unprovable lattice phase; provide the exact "
                    "configured spectrum range and bin count"
                )
        bins.append(item)
    return tuple(bins), maximum_residual


def _bound_lattice_bins(
    raw_bins: Sequence[SpectrumBin], expectation: SpectrumExpectation
) -> tuple[tuple[SpectrumBin, ...], float]:
    width = (expectation.w_max_gev - expectation.w_min_gev) / expectation.configured_bins
    scale = max(abs(expectation.w_min_gev), abs(expectation.w_max_gev), 1.0)
    tolerance = max(
        BIN_CONTIGUITY_ABSOLUTE_TOLERANCE_GEV,
        min(
            LATTICE_EDGE_BIN_FRACTION_TOLERANCE * width,
            LATTICE_EDGE_RELATIVE_TOLERANCE * scale,
        ),
    )
    if tolerance >= 0.5 * width:
        raise SpectrumFitError("configured lattice is too fine to recover unambiguously")
    occupied: dict[int, SpectrumBin] = {}
    previous_index = -1
    maximum_residual = 0.0
    for record_index, item in enumerate(raw_bins):
        lattice_coordinate = (item.low - expectation.w_min_gev) / width
        lattice_index = round(lattice_coordinate)
        if not (0 <= lattice_index < expectation.configured_bins):
            raise SpectrumFitError(
                f"record {record_index}: bin lies outside configured spectrum range"
            )
        expected_low = expectation.w_min_gev + lattice_index * width
        expected_high = expectation.w_min_gev + (lattice_index + 1) * width
        low_residual = abs(item.low - expected_low)
        high_residual = abs(item.high - expected_high)
        maximum_residual = max(maximum_residual, low_residual, high_residual)
        if low_residual > tolerance or high_residual > tolerance:
            raise SpectrumFitError(
                f"record {record_index}: bin edges are not aligned to the configured lattice"
            )
        if lattice_index <= previous_index:
            raise SpectrumFitError(
                f"record {record_index}: duplicate or out-of-order configured lattice index"
            )
        previous_index = lattice_index
        occupied[lattice_index] = SpectrumBin(
            expected_low, expected_high, item.density, item.weight
        )
    bins = tuple(
        occupied.get(
            index,
            SpectrumBin(
                expectation.w_min_gev + index * width,
                expectation.w_min_gev + (index + 1) * width,
                0.0,
                0.0,
            ),
        )
        for index in range(expectation.configured_bins)
    )
    return bins, maximum_residual


def read_spectrum(
    path: Path,
    expectation: SpectrumExpectation | None = None,
    *,
    allow_unbound_diagnostic: bool = False,
) -> Spectrum:
    """Read and strictly validate a FeeBL ``pair_spectrum`` text file."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SpectrumFitError(f"cannot read input: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpectrumFitError("input is not valid UTF-8") from exc
    # FeeBL normally writes LF, while copied fixtures can acquire CRLF on
    # Windows.  Normalize that one conventional representation only; a lone
    # carriage return remains malformed input.
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        raise SpectrumFitError("input contains a lone carriage return")
    lines = text.splitlines()
    if not lines or lines[0] != "#FEEBL-NATIVE 1.0":
        raise SpectrumFitError("first line must be exactly #FEEBL-NATIVE 1.0")
    if sum(line.startswith("#FEEBL-NATIVE") for line in lines) != 1:
        raise SpectrumFitError("native-format header must occur exactly once")
    version_match, version_index = _one_prefixed_line(
        lines,
        "#feebl_version",
        re.compile(r"#feebl_version[ \t]+([A-Za-z0-9][A-Za-z0-9.+_-]*)"),
        "FeeBL version",
    )
    seed_match, seed_index = _one_prefixed_line(
        lines,
        "#rndm_seed",
        re.compile(r"#rndm_seed[ \t]+([0-9]+)"),
        "random-seed identity",
    )
    energy_match, energy_index = _one_prefixed_line(
        lines,
        "#energy_unit",
        re.compile(r"#energy_unit[ \t]+(GeV)"),
        "energy unit",
    )
    del energy_match
    native_seed = int(seed_match.group(1))

    block_match, block_index = _one_prefixed_line(
        lines,
        "#BLOCK",
        re.compile(r"#BLOCK[ \t]+pair_spectrum"),
        "pair_spectrum block",
    )
    del block_match
    columns_candidates = [
        (index, line) for index, line in enumerate(lines) if line.startswith("#COLUMNS")
    ]
    expected = "W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
    if columns_candidates != [(block_index + 1, f"#COLUMNS {expected}")]:
        raise SpectrumFitError(
            "pair_spectrum requires one exact #COLUMNS line immediately inside the block"
        )
    end_match, end_index = _one_prefixed_line(
        lines,
        "#END pair_spectrum",
        re.compile(r"#END pair_spectrum[ \t]+records[ \t]+([0-9]+)"),
        "pair_spectrum end marker",
    )
    declared_record_count = int(end_match.group(1))
    if not (0 < version_index < block_index and 0 < seed_index < block_index):
        raise SpectrumFitError("FeeBL version and random-seed identity must precede the block")
    if not (0 < energy_index < block_index < end_index):
        raise SpectrumFitError("energy unit must precede a complete pair_spectrum block")
    if any(line.strip() for line in lines[end_index + 1 :]):
        raise SpectrumFitError("nonblank content follows the pair_spectrum end marker")

    footer_values: dict[str, float] = {}
    footer_indices: list[int] = []
    for key in ("underflow", "overflow", "mean_W", "rms_W", "n_eff_total"):
        value, index = _one_footer(lines, key)
        footer_values[key] = value
        footer_indices.append(index)
    if footer_indices != list(range(end_index - 5, end_index)):
        raise SpectrumFitError("pair_spectrum footers must be unique, ordered, and contiguous")
    data_start = block_index + 2
    data_end = footer_indices[0]
    if data_end <= data_start:
        raise SpectrumFitError("pair_spectrum block contains no records")

    raw_bins: list[SpectrumBin] = []
    maximum_row_weight_relative_residual = 0.0
    for zero_index in range(data_start, data_end):
        raw_line = lines[zero_index]
        line_number = zero_index + 1
        line = raw_line.strip()
        if not line or line.startswith("#"):
            raise SpectrumFitError(
                f"line {line_number}: unexpected non-record inside pair_spectrum"
            )
        fields = line.split()
        if len(fields) != 11:
            raise SpectrumFitError(
                f"line {line_number}: expected 11 pair_spectrum columns, found {len(fields)}"
            )
        low = _finite_number(fields[0], f"line {line_number} W_low")
        high = _finite_number(fields[1], f"line {line_number} W_high")
        density = _finite_number(fields[2], f"line {line_number} dL_dW")
        weight = _finite_number(fields[3], f"line {line_number} weight")
        for column, token in enumerate(fields[4:], start=5):
            _finite_number(token, f"line {line_number} column {column}")
        if high <= low:
            raise SpectrumFitError(f"line {line_number}: nonpositive bin width")
        if density < 0:
            raise SpectrumFitError(f"line {line_number}: negative luminosity density")
        if weight < 0:
            raise SpectrumFitError(f"line {line_number}: negative luminosity weight")
        reconstructed_weight = density * (high - low)
        row_scale = max(abs(weight), abs(reconstructed_weight), sys.float_info.min)
        row_relative_residual = abs(weight - reconstructed_weight) / row_scale
        maximum_row_weight_relative_residual = max(
            maximum_row_weight_relative_residual, row_relative_residual
        )
        if row_relative_residual > ROW_WEIGHT_RELATIVE_TOLERANCE:
            raise SpectrumFitError(
                f"line {line_number}: weight disagrees with dL_dW times bin width"
            )
        raw_bins.append(SpectrumBin(low, high, density, weight))

    if len(raw_bins) != declared_record_count:
        raise SpectrumFitError(
            "pair_spectrum footer record count does not equal the number of stored records"
        )
    if len(raw_bins) < MINIMUM_FIT_BINS + 6:
        raise SpectrumFitError("too few spectrum bins for the frozen core fit")

    if expectation is not None and allow_unbound_diagnostic:
        raise SpectrumFitError(
            "bound expectations and unbound diagnostic mode are mutually exclusive"
        )
    if expectation is None and not allow_unbound_diagnostic:
        raise SpectrumFitError(
            "qualification-bound histogram expectations are required; alternatively request "
            "explicit unbound diagnostic mode"
        )
    if expectation is not None:
        expectation = expectation.validated()
        if native_seed != expectation.rndm_seed:
            raise SpectrumFitError(
                f"native rndm_seed {native_seed} does not match expected {expectation.rndm_seed}"
            )
        bins, maximum_edge_residual = _bound_lattice_bins(raw_bins, expectation)
        lattice_phase_proven = True
    else:
        bins, maximum_edge_residual = _unbound_dense_bins(raw_bins)
        lattice_phase_proven = False

    underflow = footer_values["underflow"]
    overflow = footer_values["overflow"]
    if underflow < 0 or overflow < 0:
        raise SpectrumFitError("underflow and overflow must be nonnegative")
    provisional_total = math.fsum(item.luminosity for item in bins) + underflow + overflow
    expected_total = None if expectation is None else expectation.expected_total_luminosity
    total_relative_residual = None
    if expected_total is not None:
        total_scale = max(abs(provisional_total), abs(expected_total), sys.float_info.min)
        total_relative_residual = abs(provisional_total - expected_total) / total_scale
        if total_relative_residual > TOTAL_LUMINOSITY_RELATIVE_TOLERANCE:
            raise SpectrumFitError(
                "spectrum total luminosity disagrees with executed out.dat lumi_ee"
            )
    result = Spectrum(
        bins=bins,
        underflow=underflow,
        overflow=overflow,
        footer_mean_w=footer_values["mean_W"],
        footer_rms_w=footer_values["rms_W"],
        footer_n_eff_total=footer_values["n_eff_total"],
        input_sha256=hashlib.sha256(payload).hexdigest(),
        input_name=path.name,
        native_format_version="1.0",
        feebl_version=version_match.group(1),
        rndm_seed=native_seed,
        declared_record_count=declared_record_count,
        lattice_phase_proven=lattice_phase_proven,
        configured_w_min_gev=(None if expectation is None else expectation.w_min_gev),
        configured_w_max_gev=(None if expectation is None else expectation.w_max_gev),
        configured_bin_count=(None if expectation is None else expectation.configured_bins),
        maximum_lattice_edge_residual_gev=maximum_edge_residual,
        maximum_row_weight_relative_residual=maximum_row_weight_relative_residual,
        expected_total_luminosity=expected_total,
        total_luminosity_relative_residual=total_relative_residual,
        expected_run_id=(None if expectation is None else expectation.run_id),
    )
    if result.in_range_luminosity <= 0 or result.total_luminosity <= 0:
        raise SpectrumFitError("spectrum has zero total luminosity")
    if result.under_overflow_fraction > MAXIMUM_UNDER_OVERFLOW_FRACTION:
        raise SpectrumFitError(
            "underflow plus overflow exceeds the frozen 0.1% luminosity-fraction limit"
        )
    if result.footer_rms_w is not None and result.footer_rms_w < 0:
        raise SpectrumFitError("#rms_W must be nonnegative")
    if result.footer_n_eff_total is not None and result.footer_n_eff_total < 0:
        raise SpectrumFitError("#n_eff_total must be nonnegative")
    return result


def _smoothed_densities(bins: Sequence[SpectrumBin]) -> list[float]:
    result = [item.density for item in bins]
    for index in range(2, len(bins) - 2):
        result[index] = (
            math.fsum(
                coefficient * bins[index + offset - 2].density
                for offset, coefficient in enumerate(SMOOTHING_KERNEL)
            )
            / SMOOTHING_DENOMINATOR
        )
    return result


def _modal_index(bins: Sequence[SpectrumBin]) -> int:
    smoothed = _smoothed_densities(bins)
    maximum = max(smoothed[2:-2])
    # A high-energy tie is deterministic and is conservative against a
    # beamstrahlung-induced low-energy shoulder.
    candidates = [i for i in range(2, len(bins) - 2) if smoothed[i] == maximum]
    if not candidates:
        raise SpectrumFitError("cannot locate an interior spectrum mode")
    return candidates[-1]


def _initial_mu_sigma(bins: Sequence[SpectrumBin], modal_index: int) -> tuple[float, float]:
    centers = [bins[i].center for i in (modal_index - 1, modal_index, modal_index + 1)]
    values = [bins[i].density for i in (modal_index - 1, modal_index, modal_index + 1)]
    if min(values) <= 0:
        raise SpectrumFitError("nonpositive density prevents modal interpolation")
    spacing = bins[modal_index].width
    left, middle, right = (math.log(item) for item in values)
    slope = (right - left) / (2.0 * spacing)
    curvature = (right + left - 2.0 * middle) / (2.0 * spacing * spacing)
    shift = -slope / (2.0 * curvature) if curvature < 0 else 0.0
    if not math.isfinite(shift) or abs(shift) > spacing:
        shift = 0.0
    mu = centers[1] + shift

    half = 0.5 * bins[modal_index].density
    crossing: float | None = None
    for index in range(modal_index, len(bins) - 1):
        y0 = bins[index].density
        y1 = bins[index + 1].density
        if y0 >= half and 0 < y1 <= half:
            log_half = math.log(half)
            fraction = (log_half - math.log(y0)) / (math.log(y1) - math.log(y0))
            crossing = bins[index].center + fraction * (bins[index + 1].center - bins[index].center)
            break
    if crossing is None or crossing <= mu:
        raise SpectrumFitError("upper flank never reaches the half-maximum")
    sigma = (crossing - mu) / math.sqrt(2.0 * math.log(2.0))
    if not math.isfinite(sigma) or sigma <= bins[modal_index].width:
        raise SpectrumFitError("invalid half-maximum core-width initialization")
    return mu, sigma


def _solve_three(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    augmented = [list(matrix[row]) + [vector[row]] for row in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-30:
            raise SpectrumFitError("singular quadratic normal matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for item in range(column, 4):
            augmented[column][item] /= divisor
        for row in range(3):
            if row == column:
                continue
            multiplier = augmented[row][column]
            for item in range(column, 4):
                augmented[row][item] -= multiplier * augmented[column][item]
    return [augmented[row][3] for row in range(3)]


def _inverse_three(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    columns = [
        _solve_three(matrix, [1.0 if row == column else 0.0 for row in range(3)])
        for column in range(3)
    ]
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def _quadratic_fit(
    selected: Sequence[SpectrumBin], reference_mu: float
) -> tuple[list[float], list[list[float]], list[float], float, float, float]:
    xs = [item.center - reference_mu for item in selected]
    ys = [math.log(item.density) for item in selected]
    sums = [math.fsum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[0], sums[1], sums[2]],
        [sums[1], sums[2], sums[3]],
        [sums[2], sums[3], sums[4]],
    ]
    vector = [
        math.fsum(ys),
        math.fsum(x * y for x, y in zip(xs, ys)),
        math.fsum(x * x * y for x, y in zip(xs, ys)),
    ]
    coefficients = _solve_three(matrix, vector)
    residuals = [
        y - (coefficients[0] + coefficients[1] * x + coefficients[2] * x * x)
        for x, y in zip(xs, ys)
    ]
    sse = math.fsum(item * item for item in residuals)
    mean_y = math.fsum(ys) / len(ys)
    sst = math.fsum((item - mean_y) ** 2 for item in ys)
    if sst <= 0:
        raise SpectrumFitError("upper-flank logarithms have zero variance")
    r_squared = 1.0 - sse / sst
    residual_rms = math.sqrt(sse / len(residuals))
    maximum_residual = max(abs(item) for item in residuals)
    covariance = _inverse_three(matrix)
    residual_variance = sse / (len(selected) - 3)
    covariance = [
        [residual_variance * covariance[row][column] for column in range(3)] for row in range(3)
    ]
    return coefficients, covariance, residuals, r_squared, residual_rms, maximum_residual


def _selected_upper_flank(
    bins: Sequence[SpectrumBin],
    mu: float,
    sigma: float,
    upper_sigma: float,
    modal_density: float,
) -> tuple[list[SpectrumBin], tuple[int, ...]]:
    lower = mu + MAIN_LOWER_SIGMA * sigma
    upper = mu + upper_sigma * sigma
    threshold = MINIMUM_DENSITY_FRACTION * modal_density
    indices = tuple(
        index
        for index, item in enumerate(bins)
        if lower <= item.center <= upper and item.density >= threshold
    )
    return [bins[index] for index in indices], indices


def _mapped_core_parameters(
    spectrum: Spectrum,
    selected: Sequence[SpectrumBin],
    reference_mu: float,
) -> tuple[list[float], list[list[float]], float, float, float, float, float]:
    coefficients, covariance, _, r_squared, residual_rms, maximum_residual = _quadratic_fit(
        selected, reference_mu
    )
    _, slope, curvature = coefficients
    if curvature >= 0:
        raise SpectrumFitError("upper-flank log fit is not concave")
    next_mu = reference_mu - slope / (2.0 * curvature)
    sigma_bin_squared = -1.0 / (2.0 * curvature)
    sigma_core_squared = sigma_bin_squared - spectrum.bin_width**2 / 12.0
    if sigma_core_squared <= 0:
        raise SpectrumFitError("bin-width correction leaves nonpositive core variance")
    next_sigma = math.sqrt(sigma_core_squared)
    if not math.isfinite(next_mu) or not math.isfinite(next_sigma):
        raise SpectrumFitError("non-finite fitted core parameters")
    return (
        coefficients,
        covariance,
        r_squared,
        residual_rms,
        maximum_residual,
        next_mu,
        next_sigma,
    )


def _fit_once(
    spectrum: Spectrum,
    upper_sigma: float,
    enforce_quality: bool = True,
) -> RawFit:
    bins = spectrum.bins
    modal_index = _modal_index(bins)
    modal_center = bins[modal_index].center
    modal_density = bins[modal_index].density
    if modal_density <= 0:
        raise SpectrumFitError("spectrum mode has nonpositive density")
    mu, sigma = _initial_mu_sigma(bins, modal_index)
    converged = False
    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        selected, selected_indices = _selected_upper_flank(
            bins, mu, sigma, upper_sigma, modal_density
        )
        if len(selected) < MINIMUM_FIT_BINS:
            raise SpectrumFitError(
                f"upper-flank fit has {len(selected)} bins; at least {MINIMUM_FIT_BINS} are required"
            )
        (
            _,
            _,
            _,
            _,
            _,
            next_mu,
            next_sigma,
        ) = _mapped_core_parameters(spectrum, selected, mu)
        _, next_selected_indices = _selected_upper_flank(
            bins, next_mu, next_sigma, upper_sigma, modal_density
        )
        mu_change = abs(next_mu - mu)
        sigma_change = abs(next_sigma - sigma) / next_sigma
        mu, sigma = next_mu, next_sigma
        if (
            mu_change < MU_CONVERGENCE_GEV
            and sigma_change < SIGMA_RELATIVE_CONVERGENCE
            and next_selected_indices == selected_indices
        ):
            converged = True
            break
    if not converged:
        raise SpectrumFitError("upper-flank fit did not converge in 10 iterations")

    # Recompute in the converged window so diagnostics and covariance refer to
    # the published parameters.  The recomputed map is itself required to be
    # a fixed point, including exact selected-bin identity.  This prevents a
    # final changed-window fit from escaping the convergence test above.
    selected, selected_indices = _selected_upper_flank(bins, mu, sigma, upper_sigma, modal_density)
    if len(selected) < MINIMUM_FIT_BINS:
        raise SpectrumFitError("converged upper-flank window contains too few bins")
    (
        coefficients,
        covariance,
        r_squared,
        residual_rms,
        maximum_residual,
        fitted_mu,
        fitted_sigma,
    ) = _mapped_core_parameters(spectrum, selected, mu)
    _, fitted_indices = _selected_upper_flank(
        bins, fitted_mu, fitted_sigma, upper_sigma, modal_density
    )
    intercept, slope, curvature = coefficients
    sigma_bin_squared = -1.0 / (2.0 * curvature)
    if (
        abs(fitted_mu - mu) >= MU_CONVERGENCE_GEV
        or abs(fitted_sigma - sigma) / fitted_sigma >= SIGMA_RELATIVE_CONVERGENCE
        or fitted_indices != selected_indices
    ):
        raise SpectrumFitError("post-convergence publish fit is not a stable fixed point")

    checked_selected, checked_indices = _selected_upper_flank(
        bins, fitted_mu, fitted_sigma, upper_sigma, modal_density
    )
    (
        _,
        _,
        _,
        _,
        _,
        checked_mu,
        checked_sigma,
    ) = _mapped_core_parameters(spectrum, checked_selected, fitted_mu)
    _, checked_next_indices = _selected_upper_flank(
        bins, checked_mu, checked_sigma, upper_sigma, modal_density
    )
    if (
        checked_indices != selected_indices
        or checked_next_indices != checked_indices
        or abs(checked_mu - fitted_mu) >= MU_CONVERGENCE_GEV
        or abs(checked_sigma - fitted_sigma) / checked_sigma >= SIGMA_RELATIVE_CONVERGENCE
    ):
        raise SpectrumFitError("independent post-convergence fixed-point recheck failed")
    log_peak = intercept - slope * slope / (4.0 * curvature)
    if log_peak > math.log(sys.float_info.max):
        raise SpectrumFitError("fitted peak density overflows binary64")
    peak_density = math.exp(log_peak)
    core_area = math.sqrt(2.0 * math.pi) * peak_density * fitted_sigma

    inverse_curvature_squared = curvature * curvature
    grad_mu = (0.0, -1.0 / (2.0 * curvature), slope / (2.0 * inverse_curvature_squared))
    grad_sigma = (0.0, 0.0, 1.0 / (4.0 * fitted_sigma * inverse_curvature_squared))

    def propagated(gradient: Sequence[float]) -> float:
        variance = math.fsum(
            gradient[row] * covariance[row][column] * gradient[column]
            for row in range(3)
            for column in range(3)
        )
        return math.sqrt(max(0.0, variance))

    if enforce_quality:
        if r_squared < MINIMUM_LOG_R_SQUARED:
            raise SpectrumFitError(
                f"poor Gaussian-core fit: log-space R^2={r_squared:.8g} < {MINIMUM_LOG_R_SQUARED}"
            )
        if maximum_residual > MAXIMUM_ABS_LOG_RESIDUAL:
            raise SpectrumFitError(
                "poor Gaussian-core fit: maximum absolute log residual "
                f"{maximum_residual:.8g} > {MAXIMUM_ABS_LOG_RESIDUAL}"
            )
        if abs(fitted_mu - modal_center) > MODE_VERTEX_TOLERANCE_SIGMA * fitted_sigma:
            raise SpectrumFitError("fitted Gaussian vertex is inconsistent with the observed mode")
        if bins[-1].high < fitted_mu + MINIMUM_UPPER_BOUND_SIGMAS * fitted_sigma:
            raise SpectrumFitError(
                "spectrum upper boundary is less than three fitted sigma above the core"
            )

    return RawFit(
        mu_gev=fitted_mu,
        sigma_core_gev=fitted_sigma,
        sigma_bin_gev=math.sqrt(sigma_bin_squared),
        peak_density=peak_density,
        core_area=core_area,
        iteration_count=iteration,
        fit_bin_count=len(selected),
        lower_sigma=MAIN_LOWER_SIGMA,
        upper_sigma=upper_sigma,
        log_r_squared=r_squared,
        log_residual_rms=residual_rms,
        maximum_abs_log_residual=maximum_residual,
        mu_covariance_se_gev=propagated(grad_mu),
        sigma_covariance_se_gev=propagated(grad_sigma),
        modal_center_gev=modal_center,
        window_low_gev=selected[0].low,
        window_high_gev=selected[-1].high,
    )


def _paired_rebin(spectrum: Spectrum) -> Spectrum:
    source = list(spectrum.bins)
    if len(source) % 2:
        # FeeBL omits empty histogram records.  When the last stored nonzero
        # record leaves an odd reconstructed prefix, append the next known
        # lattice bin with exactly zero density.  It is far above the selected
        # flank and changes neither the primary fit nor its luminosity.
        low = source[-1].high
        source.append(SpectrumBin(low, low + spectrum.bin_width, 0.0, 0.0))
    bins: list[SpectrumBin] = []
    for index in range(0, len(source), 2):
        left = source[index]
        right = source[index + 1]
        width = right.high - left.low
        weight = left.luminosity + right.luminosity
        density = weight / width
        bins.append(SpectrumBin(left.low, right.high, density, weight))
    return Spectrum(
        bins=tuple(bins),
        underflow=spectrum.underflow,
        overflow=spectrum.overflow,
        footer_mean_w=spectrum.footer_mean_w,
        footer_rms_w=spectrum.footer_rms_w,
        footer_n_eff_total=spectrum.footer_n_eff_total,
        input_sha256=spectrum.input_sha256,
        input_name=spectrum.input_name,
        native_format_version=spectrum.native_format_version,
        feebl_version=spectrum.feebl_version,
        rndm_seed=spectrum.rndm_seed,
        declared_record_count=spectrum.declared_record_count,
        lattice_phase_proven=spectrum.lattice_phase_proven,
        configured_w_min_gev=spectrum.configured_w_min_gev,
        configured_w_max_gev=spectrum.configured_w_max_gev,
        configured_bin_count=(
            None
            if spectrum.configured_bin_count is None
            else (spectrum.configured_bin_count + spectrum.configured_bin_count % 2) // 2
        ),
        maximum_lattice_edge_residual_gev=spectrum.maximum_lattice_edge_residual_gev,
        maximum_row_weight_relative_residual=spectrum.maximum_row_weight_relative_residual,
        expected_total_luminosity=spectrum.expected_total_luminosity,
        total_luminosity_relative_residual=spectrum.total_luminosity_relative_residual,
        expected_run_id=spectrum.expected_run_id,
    )


def _fixed_window_parameters(
    spectrum: Spectrum, reference_mu: float, reference_sigma: float, upper_sigma: float
) -> tuple[float, float]:
    """Refit an alternate window whose coordinates come from the primary fit.

    Holding the coordinates fixed prevents the sensitivity test from mixing
    non-Gaussian shape dependence with feedback from independently moving
    window boundaries.
    """
    modal_density = spectrum.bins[_modal_index(spectrum.bins)].density
    lower = reference_mu + MAIN_LOWER_SIGMA * reference_sigma
    upper = reference_mu + upper_sigma * reference_sigma
    threshold = MINIMUM_DENSITY_FRACTION * modal_density
    selected = [
        item
        for item in spectrum.bins
        if lower <= item.center <= upper and item.density >= threshold
    ]
    if len(selected) < MINIMUM_FIT_BINS:
        raise SpectrumFitError("alternate fixed fit window contains too few bins")
    coefficients, _, _, _, _, _ = _quadratic_fit(selected, reference_mu)
    _, slope, curvature = coefficients
    if curvature >= 0:
        raise SpectrumFitError("alternate fixed-window log fit is not concave")
    mu = reference_mu - slope / (2.0 * curvature)
    variance = -1.0 / (2.0 * curvature) - spectrum.bin_width**2 / 12.0
    if variance <= 0 or not math.isfinite(mu):
        raise SpectrumFitError("alternate fixed-window fit has invalid core parameters")
    return mu, math.sqrt(variance)


def fit_spectrum(spectrum: Spectrum) -> dict[str, object]:
    """Apply the frozen primary fit and all window/binning stability gates."""
    main = _fit_once(spectrum, MAIN_UPPER_SIGMA)
    alternatives = [
        _fixed_window_parameters(spectrum, main.mu_gev, main.sigma_core_gev, upper)
        for upper in STABILITY_UPPER_SIGMAS
    ]
    # Evaluate pair rebinning in the primary fit's fixed physical window.
    # Letting the coarser rebinned lattice move the window can create a
    # harmless two-cycle when a boundary alternates across one paired bin;
    # that would conflate binning sensitivity with window-selection feedback.
    # The published primary fit above still has the full parameter-and-bin-set
    # fixed-point requirement and independent recheck.
    rebinned = (
        _fixed_window_parameters(
            _paired_rebin(spectrum),
            main.mu_gev,
            main.sigma_core_gev,
            MAIN_UPPER_SIGMA,
        )
        if spectrum.lattice_phase_proven
        else None
    )

    window_mu_shift = max(abs(mu - main.mu_gev) for mu, _ in alternatives)
    window_mu_sigma_fraction = window_mu_shift / main.sigma_core_gev
    window_sigma_shift = max(
        abs(sigma - main.sigma_core_gev) / main.sigma_core_gev for _, sigma in alternatives
    )
    rebin_mu_shift = None if rebinned is None else abs(rebinned[0] - main.mu_gev)
    rebin_sigma_shift = (
        None if rebinned is None else abs(rebinned[1] - main.sigma_core_gev) / main.sigma_core_gev
    )
    if window_mu_sigma_fraction > MAXIMUM_WINDOW_MU_SIGMA_FRACTION:
        raise SpectrumFitError(
            "fit-window core-mean shift "
            f"{window_mu_sigma_fraction:.8g} core sigma exceeds 0.05 core sigma"
        )
    if window_sigma_shift > MAXIMUM_WINDOW_SIGMA_RELATIVE_SHIFT:
        raise SpectrumFitError(f"fit-window core-width shift {window_sigma_shift:.8g} exceeds 0.03")
    if rebin_mu_shift is not None and rebin_mu_shift > MAXIMUM_REBIN_MU_SHIFT_GEV:
        raise SpectrumFitError(
            f"pair-rebin core-mean shift {rebin_mu_shift:.8g} GeV exceeds 0.0005 GeV"
        )
    if rebin_sigma_shift is not None and rebin_sigma_shift > MAXIMUM_REBIN_SIGMA_RELATIVE_SHIFT:
        raise SpectrumFitError(f"pair-rebin core-width shift {rebin_sigma_shift:.8g} exceeds 0.005")

    mu_systematic = max(
        window_mu_shift,
        0.0 if rebin_mu_shift is None else rebin_mu_shift,
    )
    sigma_systematic = max(
        max(abs(sigma - main.sigma_core_gev) for _, sigma in alternatives),
        0.0 if rebinned is None else abs(rebinned[1] - main.sigma_core_gev),
    )
    mu_fit_uncertainty = math.hypot(main.mu_covariance_se_gev, mu_systematic)
    sigma_fit_uncertainty = math.hypot(main.sigma_covariance_se_gev, sigma_systematic)

    def numeric(value: float | None) -> str | None:
        return None if value is None else format(value, ".17g")

    return {
        "schema": SCHEMA,
        "schema_version": "1",
        "status": (
            "accepted_core_fit_not_physics_qualification"
            if spectrum.lattice_phase_proven
            else "accepted_unbound_diagnostic_core_fit_not_physics_qualification"
        ),
        "input": {
            "file_name": spectrum.input_name,
            "sha256": spectrum.input_sha256,
        },
        "qualification_binding": {
            "status": (
                "qualification_input_bound"
                if spectrum.lattice_phase_proven
                else "unbound_diagnostic_only"
            ),
            "native_format": f"FEEBL-NATIVE {spectrum.native_format_version}",
            "feebl_version": spectrum.feebl_version,
            "rndm_seed": str(spectrum.rndm_seed),
            "run_id": spectrum.expected_run_id,
            "spectrum_w_min_GeV": numeric(spectrum.configured_w_min_gev),
            "spectrum_w_max_GeV": numeric(spectrum.configured_w_max_gev),
            "spectrum_w_bins": (
                None
                if spectrum.configured_bin_count is None
                else str(spectrum.configured_bin_count)
            ),
            "lattice_phase_proven": spectrum.lattice_phase_proven,
            "stored_record_count": str(spectrum.declared_record_count),
            "maximum_lattice_edge_residual_GeV": numeric(
                spectrum.maximum_lattice_edge_residual_gev
            ),
            "expected_total_luminosity": numeric(spectrum.expected_total_luminosity),
            "total_luminosity_relative_residual": numeric(
                spectrum.total_luminosity_relative_residual
            ),
        },
        "method": {
            "id": METHOD_ID,
            "beamstrahlung_policy": "fit_unsmoothed_upper_flank_only",
            "smoothing_kernel_for_mode_only": "1,4,6,4,1_over_16",
            "main_window_sigma": ["0.25", "2.5"],
            "stability_upper_window_sigma": ["2", "3"],
            "minimum_density_fraction_of_mode": "0.01",
            "bin_variance_correction": "subtract_delta_w_squared_over_12",
            "maximum_iterations": str(MAXIMUM_ITERATIONS),
            "mu_convergence_GeV": "0.00001",
            "sigma_relative_convergence": "0.0001",
            "fixed_point_policy": ("parameter_and_selected_bin_stability_plus_independent_recheck"),
            "pair_rebin_window_policy": "fixed_primary_physical_window",
        },
        "spectrum": {
            "bin_count": str(len(spectrum.bins)),
            "stored_record_count": str(spectrum.declared_record_count),
            "bin_width_GeV": numeric(spectrum.bin_width),
            "in_range_luminosity": numeric(spectrum.in_range_luminosity),
            "underflow_luminosity": numeric(spectrum.underflow),
            "overflow_luminosity": numeric(spectrum.overflow),
            "under_overflow_fraction": numeric(spectrum.under_overflow_fraction),
            "global_mean_W_GeV_diagnostic_only": numeric(spectrum.footer_mean_w),
            "global_rms_W_GeV_diagnostic_only": numeric(spectrum.footer_rms_w),
            "n_eff_total_diagnostic": numeric(spectrum.footer_n_eff_total),
            "maximum_row_weight_relative_residual": numeric(
                spectrum.maximum_row_weight_relative_residual
            ),
        },
        "core_fit": {
            "mu_core_GeV": numeric(main.mu_gev),
            "sigma_core_GeV": numeric(main.sigma_core_gev),
            "sigma_bin_convolved_GeV": numeric(main.sigma_bin_gev),
            "peak_dL_dW": numeric(main.peak_density),
            "gaussian_core_area": numeric(main.core_area),
            "mu_fit_uncertainty_GeV_diagnostic": numeric(mu_fit_uncertainty),
            "sigma_fit_uncertainty_GeV_diagnostic": numeric(sigma_fit_uncertainty),
            "mu_covariance_se_GeV_diagnostic": numeric(main.mu_covariance_se_gev),
            "sigma_covariance_se_GeV_diagnostic": numeric(main.sigma_covariance_se_gev),
            "fit_bin_count": str(main.fit_bin_count),
            "iteration_count": str(main.iteration_count),
            "modal_bin_center_GeV": numeric(main.modal_center_gev),
            "fit_window_low_GeV": numeric(main.window_low_gev),
            "fit_window_high_GeV": numeric(main.window_high_gev),
            "log_r_squared": numeric(main.log_r_squared),
            "log_residual_rms": numeric(main.log_residual_rms),
            "maximum_abs_log_residual": numeric(main.maximum_abs_log_residual),
        },
        "stability": {
            "maximum_window_mu_shift_GeV": numeric(window_mu_shift),
            "maximum_window_mu_shift_sigma_fraction": numeric(window_mu_sigma_fraction),
            "maximum_window_sigma_relative_shift": numeric(window_sigma_shift),
            "pair_rebin_mu_shift_GeV": numeric(rebin_mu_shift),
            "pair_rebin_sigma_relative_shift": numeric(rebin_sigma_shift),
            "window_mu_limit_sigma_fraction": "0.05",
            "window_mu_limit_GeV_for_this_fit": numeric(
                MAXIMUM_WINDOW_MU_SIGMA_FRACTION * main.sigma_core_gev
            ),
            "window_sigma_relative_limit": "0.03",
            "pair_rebin_mu_limit_GeV": "0.0005",
            "pair_rebin_sigma_relative_limit": "0.005",
            "pair_rebin_status": (
                "accepted_bound_lattice_phase"
                if rebinned is not None
                else "not_performed_unbound_lattice_phase"
            ),
        },
        "quality": {
            "minimum_log_r_squared": "0.995",
            "maximum_abs_log_residual": "0.15",
            "maximum_under_overflow_fraction": "0.001",
            "row_weight_relative_tolerance": "0.00002",
            "total_luminosity_relative_tolerance": "0.00002",
            "minimum_upper_boundary_sigma": "3",
            "qualification_uncertainty": (
                "paired_independent_complete_run_seed_interval_required; "
                "per_fit_uncertainties_are_diagnostic_only"
            ),
            "qualification_input_binding": (
                "required_and_satisfied"
                if spectrum.lattice_phase_proven
                else "absent_unbound_diagnostic_only"
            ),
        },
    }


CSV_COLUMNS = (
    "schema",
    "input_file_name",
    "input_sha256",
    "binding_status",
    "run_id",
    "rndm_seed",
    "spectrum_w_min_GeV",
    "spectrum_w_max_GeV",
    "spectrum_w_bins",
    "lattice_phase_proven",
    "stored_record_count",
    "expected_total_luminosity",
    "total_luminosity_relative_residual",
    "maximum_row_weight_relative_residual",
    "status",
    "mu_core_GeV",
    "sigma_core_GeV",
    "gaussian_core_area",
    "peak_dL_dW",
    "mu_fit_uncertainty_GeV_diagnostic",
    "sigma_fit_uncertainty_GeV_diagnostic",
    "log_r_squared",
    "maximum_abs_log_residual",
    "maximum_window_mu_shift_GeV",
    "maximum_window_mu_shift_sigma_fraction",
    "maximum_window_sigma_relative_shift",
    "pair_rebin_mu_shift_GeV",
    "pair_rebin_sigma_relative_shift",
    "global_mean_W_GeV_diagnostic_only",
    "global_rms_W_GeV_diagnostic_only",
    "under_overflow_fraction",
)


def csv_row(result: dict[str, object]) -> dict[str, str]:
    input_data = result["input"]
    binding = result["qualification_binding"]
    fit = result["core_fit"]
    stability = result["stability"]
    spectrum = result["spectrum"]
    assert isinstance(input_data, dict)
    assert isinstance(binding, dict)
    assert isinstance(fit, dict)
    assert isinstance(stability, dict)
    assert isinstance(spectrum, dict)
    return {
        "schema": str(result["schema"]),
        "input_file_name": str(input_data["file_name"]),
        "input_sha256": str(input_data["sha256"]),
        "binding_status": str(binding["status"]),
        "run_id": "" if binding["run_id"] is None else str(binding["run_id"]),
        "rndm_seed": str(binding["rndm_seed"]),
        "spectrum_w_min_GeV": (
            "" if binding["spectrum_w_min_GeV"] is None else str(binding["spectrum_w_min_GeV"])
        ),
        "spectrum_w_max_GeV": (
            "" if binding["spectrum_w_max_GeV"] is None else str(binding["spectrum_w_max_GeV"])
        ),
        "spectrum_w_bins": (
            "" if binding["spectrum_w_bins"] is None else str(binding["spectrum_w_bins"])
        ),
        "lattice_phase_proven": ("true" if binding["lattice_phase_proven"] is True else "false"),
        "stored_record_count": str(binding["stored_record_count"]),
        "expected_total_luminosity": (
            ""
            if binding["expected_total_luminosity"] is None
            else str(binding["expected_total_luminosity"])
        ),
        "total_luminosity_relative_residual": (
            ""
            if binding["total_luminosity_relative_residual"] is None
            else str(binding["total_luminosity_relative_residual"])
        ),
        "maximum_row_weight_relative_residual": str(
            spectrum["maximum_row_weight_relative_residual"]
        ),
        "status": str(result["status"]),
        "mu_core_GeV": str(fit["mu_core_GeV"]),
        "sigma_core_GeV": str(fit["sigma_core_GeV"]),
        "gaussian_core_area": str(fit["gaussian_core_area"]),
        "peak_dL_dW": str(fit["peak_dL_dW"]),
        "mu_fit_uncertainty_GeV_diagnostic": str(fit["mu_fit_uncertainty_GeV_diagnostic"]),
        "sigma_fit_uncertainty_GeV_diagnostic": str(fit["sigma_fit_uncertainty_GeV_diagnostic"]),
        "log_r_squared": str(fit["log_r_squared"]),
        "maximum_abs_log_residual": str(fit["maximum_abs_log_residual"]),
        "maximum_window_mu_shift_GeV": str(stability["maximum_window_mu_shift_GeV"]),
        "maximum_window_mu_shift_sigma_fraction": str(
            stability["maximum_window_mu_shift_sigma_fraction"]
        ),
        "maximum_window_sigma_relative_shift": str(
            stability["maximum_window_sigma_relative_shift"]
        ),
        "pair_rebin_mu_shift_GeV": (
            ""
            if stability["pair_rebin_mu_shift_GeV"] is None
            else str(stability["pair_rebin_mu_shift_GeV"])
        ),
        "pair_rebin_sigma_relative_shift": (
            ""
            if stability["pair_rebin_sigma_relative_shift"] is None
            else str(stability["pair_rebin_sigma_relative_shift"])
        ),
        "global_mean_W_GeV_diagnostic_only": str(spectrum["global_mean_W_GeV_diagnostic_only"]),
        "global_rms_W_GeV_diagnostic_only": str(spectrum["global_rms_W_GeV_diagnostic_only"]),
        "under_overflow_fraction": str(spectrum["under_overflow_fraction"]),
    }


def _exclusive_output(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
    except OSError as exc:
        raise SpectrumFitError(f"cannot create output {path}: {exc}") from exc


def write_outputs(result: dict[str, object], json_path: Path, csv_path: Path) -> None:
    if json_path.resolve() == csv_path.resolve():
        raise SpectrumFitError("JSON and CSV outputs must be different files")
    json_text = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(csv_row(result))
    if json_path.exists() or csv_path.exists():
        raise SpectrumFitError("refusing to overwrite an existing JSON or CSV output")
    _exclusive_output(json_path, json_text)
    try:
        _exclusive_output(csv_path, buffer.getvalue())
    except Exception:
        try:
            json_path.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the frozen upper-flank Gaussian core from a FeeBL dL/dW spectrum."
    )
    parser.add_argument("spectrum", type=Path, help="FeeBL lumi.ee.out.dLdW.fne file")
    parser.add_argument("--json", required=True, type=Path, dest="json_path")
    parser.add_argument("--csv", required=True, type=Path, dest="csv_path")
    parser.add_argument(
        "--expected-seed",
        help="exact rndm_seed from the executed acc.dat and native header",
    )
    parser.add_argument(
        "--expected-w-min",
        help="exact spectrum_w_min literal from the executed acc.dat, in GeV",
    )
    parser.add_argument(
        "--expected-w-max",
        help="exact spectrum_w_max literal from the executed acc.dat, in GeV",
    )
    parser.add_argument(
        "--expected-bins",
        help="exact spectrum_w_bins integer from the executed acc.dat",
    )
    parser.add_argument(
        "--expected-total-luminosity",
        help="executed out.dat lumi_ee value in the native luminosity units",
    )
    parser.add_argument(
        "--expected-run-id",
        help="optional safe controller-owned run identity to bind into both outputs",
    )
    parser.add_argument(
        "--allow-unbound-diagnostic",
        action="store_true",
        help=(
            "permit a dense contiguous spectrum without configured-lattice expectations; "
            "the result is explicitly non-qualifying and skips pair-rebin validation"
        ),
    )
    return parser


def _expectation_from_args(args: argparse.Namespace) -> SpectrumExpectation | None:
    raw = (
        args.expected_seed,
        args.expected_w_min,
        args.expected_w_max,
        args.expected_bins,
        args.expected_total_luminosity,
    )
    provided = sum(item is not None for item in raw)
    if provided not in (0, 5):
        raise SpectrumFitError(
            "--expected-seed, --expected-w-min, --expected-w-max, --expected-bins, "
            "and --expected-total-luminosity "
            "must be supplied together"
        )
    if provided == 0:
        if args.expected_run_id is not None:
            raise SpectrumFitError("--expected-run-id requires bound histogram expectations")
        if not args.allow_unbound_diagnostic:
            raise SpectrumFitError(
                "qualification-bound expectations are required unless "
                "--allow-unbound-diagnostic is explicit"
            )
        return None
    if args.allow_unbound_diagnostic:
        raise SpectrumFitError(
            "--allow-unbound-diagnostic cannot be combined with bound expectations"
        )
    assert all(item is not None for item in raw)
    if re.fullmatch(r"[0-9]+", args.expected_seed) is None:
        raise SpectrumFitError("--expected-seed must be an exact unsigned integer")
    if re.fullmatch(r"[0-9]+", args.expected_bins) is None:
        raise SpectrumFitError("--expected-bins must be an exact unsigned integer")
    return SpectrumExpectation(
        rndm_seed=int(args.expected_seed),
        w_min_gev=_finite_number(args.expected_w_min, "--expected-w-min"),
        w_max_gev=_finite_number(args.expected_w_max, "--expected-w-max"),
        configured_bins=int(args.expected_bins),
        expected_total_luminosity=_finite_number(
            args.expected_total_luminosity, "--expected-total-luminosity"
        ),
        run_id=args.expected_run_id,
    ).validated()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expectation = _expectation_from_args(args)
        result = fit_spectrum(
            read_spectrum(
                args.spectrum,
                expectation,
                allow_unbound_diagnostic=args.allow_unbound_diagnostic,
            )
        )
        write_outputs(result, args.json_path, args.csv_path)
    except SpectrumFitError as exc:
        print(f"gaussian_core_spectrum: ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "gaussian_core_spectrum: accepted "
        f"mu={result['core_fit']['mu_core_GeV']} GeV "
        f"sigma={result['core_fit']['sigma_core_GeV']} GeV",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
