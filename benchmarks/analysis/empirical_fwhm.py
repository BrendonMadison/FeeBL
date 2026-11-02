"""Observed histogram-peak FWHM, with no parametric fit or smoothing."""
from __future__ import annotations
import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "feebl.empirical-native-peak-fwhm/1"
POLICY = dict(
    observable="Full observed peak FWHM of piecewise-linear native-bin density at bin centers",
    peak_search="Global maximum density in the fixed native fine segment, nominally0.96–1.04W0; no data-selected search window",
    density="Bin luminosity weight divided by actual bin width",
    crossings="Nearest left/right half-height crossings enclosing the selected strict-above-half connected component; search complete native support",
    interpolation="Linear interpolation between neighboring bin centers; exact threshold samples select their center",
    peak_ties="Exact equal-height ties: select lowest-energy maximum bin; report every tied bin and contiguous maximum plateau extent",
    halfheight_plateau="Choose the boundary nearest the selected peak; export all exact half-height plateaus and flag them",
    primary=dict(group_size=1, phase=0),
    representations=[dict(group_size=k, phase=p) for k in (1, 2, 4) for p in range(k)],
    grouping="Group only the fine segment; retain partial leading/trailing groups with actual widths and every coarse bin; conserve total native mass",
    uncertainty="No Poisson interval or fit-derived confidence interval is assigned. Independent native seeds provide scatter later. Crossing brackets and grouping changes are numerical diagnostics, not confidence intervals.",
    relation_to_sigma="This includes the full observed peak shape; it is not2.355 times a fitted Gaussian response sigma and never replaces a rejected sigma fit",
    flags="Retain multiple components, exact ties/plateaus, missing crossings, crossings outside fine segment, and isolated excursions; no smoothing correction or noise subtraction",
    grouping_sensitivity_advisory_relative=0.05,
    maximum_bins=250000,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_record(path):
    path = Path(path)
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    return list(
        zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1))
    )


def group_histogram(low, high, weight, fine_start, fine_stop, group_size=1, phase=0):
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    weight = np.asarray(weight, float)
    require(
        len(low) == len(high) == len(weight) and 0 < len(low) <= POLICY["maximum_bins"],
        "Invalid bounded histogram size",
    )
    require(
        np.isfinite(low).all()
        and np.isfinite(high).all()
        and np.isfinite(weight).all(),
        "Nonfinite histogram value",
    )
    require(
        np.all(high > low) and np.all(weight >= 0),
        "Invalid histogram widths or weights",
    )
    require(
        np.allclose(high[:-1], low[1:], rtol=1e-12, atol=1e-12),
        "Histogram has gaps or overlaps",
    )
    require(0 <= fine_start < fine_stop <= len(low), "Invalid fixed fine segment")
    require(
        group_size in (1, 2, 4) and 0 <= phase < group_size,
        "Representation not predeclared",
    )
    starts = list(range(fine_start))
    if phase:
        starts.append(fine_start)
    starts.extend(range(min(fine_start + phase, fine_stop), fine_stop, group_size))
    starts.extend(range(fine_stop, len(low)))
    start = np.array(starts, dtype=int)
    stop = np.r_[start[1:], len(low)]
    mass = np.add.reduceat(weight, start)
    total = math.fsum(weight)
    require(
        math.isclose(math.fsum(mass), total, rel_tol=5e-15, abs_tol=0),
        "Grouping changed total native mass",
    )
    grouped = dict(
        low=low[start],
        high=high[stop - 1],
        weight=mass,
        source_start=start,
        source_stop=stop,
        fine=(start >= fine_start) & (stop <= fine_stop),
        native_fine_start=int(fine_start),
        native_fine_stop=int(fine_stop),
    )
    grouped["representation"] = dict(
        group_size=group_size,
        phase=phase,
        native_bins=len(low),
        represented_bins=len(start),
        fine_source_bins=[int(fine_start), int(fine_stop)],
        total_weight=total,
        mass_closure_relative=math.fsum(mass) / total - 1 if total > 0 else 0,
        partial_fine_groups=[
            dict(source_start=int(a), source_stop=int(b))
            for a, b in zip(start, stop)
            if a >= fine_start and b <= fine_stop and b - a < group_size
        ],
    )
    return grouped


def measure(grouped):
    low, high, weight = grouped["low"], grouped["high"], grouped["weight"]
    centers = (low + high) / 2
    density = weight / (high - low)
    fine = grouped["fine"]
    n = len(low)
    fine_indices = np.flatnonzero(fine)
    maximum = float(np.max(density[fine]))
    result = dict(
        schema=SCHEMA,
        representation=grouped["representation"],
        available=False,
        fwhm_GeV=None,
        fwhm_MeV=None,
        flags=[],
        peak=None,
        half_height_density=None,
        left_crossing=None,
        right_crossing=None,
        fine_above_half_components=[],
        additional_crossings=[],
        exact_halfheight_plateaus=[],
        interpretation=POLICY["relation_to_sigma"],
    )
    if maximum <= 0:
        result["flags"] = ["NO_POSITIVE_FINE_SEGMENT_DENSITY"]
        result["status"] = "NO_POSITIVE_FINE_SEGMENT_DENSITY"
        return result
    tied = np.flatnonzero(fine & (density == maximum))
    peak = int(tied[0])
    half = maximum / 2

    def sample(index):
        return dict(
            represented_index=int(index),
            source_start=int(grouped["source_start"][index]),
            source_stop=int(grouped["source_stop"][index]),
            center_GeV=float(centers[index]),
            low_GeV=float(low[index]),
            high_GeV=float(high[index]),
            density=float(density[index]),
            inside_fine_segment=bool(fine[index]),
        )

    plateau = next((a, b) for a, b in runs(density == maximum) if a <= peak < b)
    result["peak"] = dict(
        selected=sample(peak),
        height_density=maximum,
        tied_maximum_bins=[sample(i) for i in tied],
        plateau_center_extent_GeV=[
            float(centers[plateau[0]]),
            float(centers[plateau[1] - 1]),
        ],
        plateau_midpoint_GeV=float((centers[plateau[0]] + centers[plateau[1] - 1]) / 2),
    )
    result["half_height_density"] = half
    if len(tied) > 1:
        result["flags"].append("EXACT_PEAK_HEIGHT_TIES")
    if np.any(np.diff(tied) > 1):
        result["flags"].append("DISCONNECTED_EQUAL_MAXIMA")
    if peak in (int(fine_indices[0]), int(fine_indices[-1])):
        result["flags"].append("PEAK_AT_FINE_SEGMENT_EDGE")

    def crossing(first, second):
        x0, x1 = centers[first], centers[second]
        y0, y1 = density[first], density[second]
        require(
            (y0 - half) * (y1 - half) <= 0 and y0 != y1,
            "Crossing lacks a half-height bracket",
        )
        x = x0 + (half - y0) / (y1 - y0) * (x1 - x0)
        return dict(
            location_GeV=float(x),
            center_bracket_GeV=[float(x0), float(x1)],
            adjacent_bins=[sample(first), sample(second)],
            interpolation="exact threshold sample"
            if y0 == half or y1 == half
            else "linear at bin centers",
            entirely_inside_fine_segment=bool(fine[first] and fine[second]),
        )

    components = []
    for first, stop in runs(density > half):
        left = crossing(first - 1, first) if first > 0 else None
        right = crossing(stop - 1, stop) if stop < n else None
        intersects = bool(np.any(fine[first:stop]))
        components.append(
            dict(
                represented_start=int(first),
                represented_stop=int(stop),
                source_start=int(grouped["source_start"][first]),
                source_stop=int(grouped["source_stop"][stop - 1]),
                left_crossing=left,
                right_crossing=right,
                intersects_fine_segment=intersects,
                maximum_density=float(np.max(density[first:stop])),
            )
        )
    selected = next(
        component
        for component in components
        if component["represented_start"] <= peak < component["represented_stop"]
    )
    fine_components = [
        component for component in components if component["intersects_fine_segment"]
    ]
    result["fine_above_half_components"] = fine_components
    if len(fine_components) > 1:
        result["flags"].append("MULTIPLE_ABOVE_HALF_COMPONENTS_IN_FINE_SEGMENT")
    if any(
        component["represented_stop"] - component["represented_start"] == 1
        for component in fine_components
    ):
        result["flags"].append("SINGLE_BIN_ABOVE_HALF_COMPONENT")
    for first, stop in runs(density == half):
        if np.any(fine[first:stop]):
            result["exact_halfheight_plateaus"].append(
                dict(
                    first=sample(first),
                    last=sample(stop - 1),
                    center_extent_GeV=[float(centers[first]), float(centers[stop - 1])],
                )
            )
    if result["exact_halfheight_plateaus"]:
        result["flags"].append("EXACT_HALF_HEIGHT_SAMPLES_OR_PLATEAUS")
    result["left_crossing"] = left = selected["left_crossing"]
    result["right_crossing"] = right = selected["right_crossing"]
    for component in components:
        if component is selected:
            continue
        for side in ("left_crossing", "right_crossing"):
            if component[side] is not None:
                result["additional_crossings"].append(
                    dict(
                        component_source_start=component["source_start"],
                        side=side,
                        **component[side]
                    )
                )
    if left is None:
        result["flags"].append("LEFT_HALF_HEIGHT_CROSSING_OUTSIDE_RECORDED_SUPPORT")
    if right is None:
        result["flags"].append("RIGHT_HALF_HEIGHT_CROSSING_OUTSIDE_RECORDED_SUPPORT")
    if any(
        item is not None and not item["entirely_inside_fine_segment"]
        for item in (left, right)
    ):
        result["flags"].append("HALF_HEIGHT_CROSSING_USES_COARSE_OR_FINE_EDGE_BIN")
    if left is not None and right is not None:
        width = right["location_GeV"] - left["location_GeV"]
        require(
            width > 0, "Enclosing half-height crossings have nonpositive separation"
        )
        result.update(
            available=True,
            fwhm_GeV=width,
            fwhm_MeV=1000 * width,
            crossing_center_bracket_width_range_GeV=[
                max(
                    0.0, right["center_bracket_GeV"][0] - left["center_bracket_GeV"][1]
                ),
                right["center_bracket_GeV"][1] - left["center_bracket_GeV"][0],
            ],
            bracket_interpretation="Range from adjacent crossing-bin centers only; not a confidence or coverage interval",
        )
    result["status"] = (
        "MEASURED_WITH_FLAGS"
        if result["available"] and result["flags"]
        else "MEASURED"
        if result["available"]
        else "MISSING_ENCLOSING_CROSSING"
    )
    return result


def representations(low, high, weight, fine_start, fine_stop):
    rows = []
    for spec in POLICY["representations"]:
        grouped = group_histogram(low, high, weight, fine_start, fine_stop, **spec)
        rows.append(measure(grouped))
    primary = rows[0]["fwhm_MeV"]
    for row in rows:
        value = row["fwhm_MeV"]
        row["fwhm_relative_change_from_raw"] = (
            value / primary - 1 if value is not None and primary is not None else None
        )
    return rows
