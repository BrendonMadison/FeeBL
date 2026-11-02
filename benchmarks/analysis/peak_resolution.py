"""Observed peak widths with explicit, scale-appropriate histogram resolution.

Broad peaks use 6.4 MeV bins. The scan's predefined Gaussian-width targets
of 4, 2 and 1 MeV use 0.8, 0.4 and 0.2 MeV bins, respectively. Every alignment
at the primary and doubled resolution is retained. These are observed FWHM
values, not fitted Gaussian response parameters. Alignment ranges describe
numerical sensitivity; only independent complete collisions provide seed SEM.
"""

import math

import numpy as np
from scipy.special import ndtr

from empirical_fwhm import measure

POLICY = {
    "schema": "feebl.observed-peak-resolution/2",
    "primary_groups": {"broad": 32, "sigma4": 4, "sigma2": 2, "sigma1": 1},
    "native_bin_width_MeV": 0.2,
    "comparison": "All phases at the primary and twice-primary group sizes",
    "maximum_resolution_relative_change": 0.05,
    "single_component_required": True,
    "choice": "Fixed per predefined optics target, not optimized per simulated histogram",
    "uncertainty": "Alignment/resolution envelopes are not confidence intervals or seed SEM",
    "physical_scope": "Histogram resolution only; particle, mesh, slice and time convergence remain separate",
}


def group_spectrum(spectrum, group_size, phase):
    """Conserve full native mass/counts while combining only the fine segment."""
    low, high, weight = (spectrum[key] for key in ("low", "high", "weight"))
    first = 2048
    last = first + int(spectrum["metadata"]["peak_bins"])
    if group_size < 1 or not 0 <= phase < group_size:
        raise ValueError("Invalid histogram grouping")
    boundaries = list(range(first))
    if phase:
        boundaries.append(first)
    boundaries.extend(range(min(first + phase, last), last, group_size))
    boundaries.extend(range(last, len(low)))
    start = np.asarray(boundaries)
    stop = np.r_[start[1:], len(low)]
    mass = np.add.reduceat(weight, start)
    if not math.isclose(math.fsum(mass), math.fsum(weight), rel_tol=5e-15):
        raise ValueError("Grouping changed native luminosity")
    return dict(low=low[start], high=high[stop - 1], weight=mass,
        source_start=start, source_stop=stop,
        fine=(start >= first) & (stop <= last),
        representation=dict(group_size=int(group_size), phase=int(phase)))


def resolution_class(group):
    """Map the predeclared scan family to its fixed analysis resolution."""
    for name in ("sigma4", "sigma2", "sigma1"):
        if "_sig" + name[-1] + "_" in group:
            return name
    if "_d0_" in group or group.startswith("fcc_"):
        return "broad"
    raise ValueError("Unknown predefined resolution class: " + group)


def phase_measurements(spectrum, primary_group):
    """Return compact half-height measurements at every required alignment."""
    records = []
    for size in (primary_group, 2 * primary_group):
        for phase in range(size):
            value = measure(group_spectrum(spectrum, size, phase))
            records.append(dict(group_size=size, phase=phase,
                fwhm_MeV=value["fwhm_MeV"],
                components=len(value["fine_above_half_components"]),
                flags=value["flags"], available=value["available"]))
    return records


def gaussian_response(spectrum, primary_group, fwhm_MeV):
    """Calibrate bin response at four sub-bin offsets; do not correct data."""
    low, high = spectrum["low"], spectrum["high"]
    sigma = fwhm_MeV / (1000 * math.sqrt(8 * math.log(2)))
    nominal = spectrum["numeric"]["nominal_W_GeV"]
    native_width = high[2048] - low[2048]
    errors = []
    primary_errors = []
    for offset in (-0.375, -0.125, 0.125, 0.375):
        mean = nominal + offset * native_width
        a, b = (low - mean) / sigma, (high - mean) / sigma
        mass = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
        control = dict(spectrum, weight=mass)
        for row in phase_measurements(control, primary_group):
            error = abs(row["fwhm_MeV"] / fwhm_MeV - 1)
            errors.append(error)
            if row["group_size"] == primary_group:
                primary_errors.append(error)
    return dict(truth_fwhm_MeV=fwhm_MeV, sub_bin_offsets=4,
        primary_maximum_absolute_relative_bias=max(primary_errors),
        all_representations_maximum_absolute_relative_bias=max(errors),
        interpretation="Smooth Gaussian numerical response only; no model-based correction to the observed peak")


def analyze(spectrum, group):
    """Measure a saved spectrum, retaining instability instead of masking it."""
    regime = resolution_class(group)
    primary_group = POLICY["primary_groups"][regime]
    phases = phase_measurements(spectrum, primary_group)
    primary = phases[0]
    flags = []
    if any(not row["available"] for row in phases):
        flags.append("MISSING_HALF_HEIGHT_CROSSING")
    if any(row["components"] != 1 for row in phases):
        flags.append("UNRESOLVED_HALF_HEIGHT_COMPONENTS")
    support_flags = {"PEAK_AT_FINE_SEGMENT_EDGE",
                     "HALF_HEIGHT_CROSSING_USES_COARSE_OR_FINE_EDGE_BIN"}
    if any(support_flags.intersection(row["flags"]) for row in phases):
        flags.append("UNRESOLVED_FINE_SEGMENT_SUPPORT")
    if not primary["available"]:
        raise ValueError("Primary observed width is undefined for " + group)
    widths = [row["fwhm_MeV"] for row in phases if row["available"]]
    value = primary["fwhm_MeV"]
    change = max(abs(width / value - 1) for width in widths)
    if change > POLICY["maximum_resolution_relative_change"]:
        flags.append("RESOLUTION_CHANGE_EXCEEDS_FIVE_PERCENT")
    fine_width = spectrum["high"][2048] - spectrum["low"][2048]
    if not math.isclose(1000 * fine_width, 0.2, rel_tol=1e-5):
        raise ValueError("This frozen scan policy requires native 0.2 MeV bins")
    aligned = [row["fwhm_MeV"] for row in phases
               if row["group_size"] == primary_group and row["available"]]
    return dict(regime=regime, primary_group=primary_group, primary_phase=0,
        primary_bin_width_MeV=1000 * fine_width * primary_group,
        fwhm_MeV=value, alignment_low_MeV=min(aligned), alignment_high_MeV=max(aligned),
        resolution_low_MeV=min(widths), resolution_high_MeV=max(widths),
        maximum_resolution_relative_change=change, phases=phases,
        gaussian_control=gaussian_response(spectrum, primary_group, value),
        flags=flags, status="RESOLUTION_ADVISORY" if flags else "BINNING_STABLE_AT_FIVE_PERCENT_TARGET")
