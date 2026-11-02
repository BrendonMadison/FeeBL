"""Estimate reference-curve uncertainty from local three/five-point trends.

Only the published curve enters this estimator. It preserves normalized-bin
covariance and interpolation covariance; its assumptions do not recover the
original authors' unknown event sample or systematic uncertainty. The spacing-
aware contrast follows Gasser, Sroka and Jennen-Steinmetz, Biometrika 73 (1986)
625, doi:10.1093/biomet/73.3.625. Robust pooling and graphical floors are explicit
analysis choices. Independent-seed FeeBL covariance is calculated separately.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "reference"
NORM_MAD = 0.6744897501960817
WINDOWS = (9, 15, 23)
METHOD = "REFERENCE_ONLY_LOCAL_CONTRAST_AND_GRAPHICAL_FLOOR_ESTIMATE"


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polynomial_contrast(x, degree):
    """Unit-norm contrast that annihilates polynomials through degree d.

    For d=1 this is the spacing-aware local-linear second difference of the
    Gasser--Sroka--Jennen-Steinmetz construction. d=2 and 3 use its explicit
    polynomial-annihilation extension, not a claim from that source.
    """
    x = np.asarray(x, float)
    u = (x - x.mean()) / max(np.ptp(x), np.finfo(float).tiny)
    v = np.vstack([u ** k for k in range(degree + 1)])
    _, _, vh = np.linalg.svd(v, full_matrices=True)
    a = vh[-1]
    if a[0] < 0:
        a = -a
    return a / np.linalg.norm(a)


def robust_sigma(values):
    a = np.asarray(values, float)
    return float(np.median(np.abs(a - np.median(a))) / NORM_MAD) if len(a) else 0.0


def median_chi2_per_dof(dof):
    if dof == 1:
        return NORM_MAD ** 2
    if dof != 3:
        raise ValueError("Only explicit 1- and 3-dof calibrations are used")
    lo, hi = 0.0, 20.0
    for _ in range(100):
        m = (lo + hi) / 2
        cdf = math.erf(math.sqrt(m / 2)) - math.sqrt(2 * m / math.pi) * math.exp(-m / 2)
        if cdf < 0.5:
            lo = m
        else:
            hi = m
    return (lo + hi) / 6


def segments(x, y, low=None, high=None):
    """Retain censored gaps and detect isolated persistent jumps, not smooth tails.

    A declared bin gap is >5% of the smaller neighboring bin width. The
    optional abrupt-edge guard needs a >8-fold median change and a jump >4
    times BOTH sides' internal log ranges. A rapidly sloping Gaussian tail
    therefore does not qualify merely because adjacent ordinates differ.
    """
    n = len(x)
    cuts = []
    reasons = {}
    if low is not None:
        for i in range(n - 1):
            if low[i + 1] - high[i] > max(
                1e-9, 0.05 * min(high[i] - low[i], high[i + 1] - low[i + 1])
            ):
                cuts.append(i + 1)
                reasons[i + 1] = "retained-support gap"
    if np.all(y > 0):
        z = np.log(y)
        for i in range(2, n - 3):
            if i + 1 in reasons:
                continue
            left, right = z[i - 2 : i + 1], z[i + 1 : i + 4]
            jump = abs(np.median(right) - np.median(left))
            if jump > math.log(8) and jump > 4 * max(
                np.ptp(left), np.ptp(right), 1e-12
            ):
                cuts.append(i + 1)
                reasons[i + 1] = "persistent abrupt-edge guard"
    labels = np.searchsorted(sorted(set(cuts)), np.arange(n), side="right")
    return labels, [{"before_index": k, "reason": reasons[k]} for k in sorted(reasons)]


def local_noise(x, y, mode="log", low=None, high=None):
    """Density-level unresolved fluctuation proxy, before widths/normalization.

    The pilot is a three-point median within a contiguous segment. Linear
    histogram residuals use sqrt(pilot density) scaling; log-density residuals
    use its inverse. This is a local count-like variance-stabilizing weight,
    NOT an inferred exposure or source event count. A local robust scale then
    allows further heteroscedasticity. LEP azimuth means use constant weights.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    if not np.array_equal(order, np.arange(len(x))):
        raise ValueError("x must increase")
    if np.any(np.diff(x) <= 0):
        raise ValueError("distinct x required")
    if mode == "log" and np.any(y <= 0):
        raise ValueError("log requires positive density")
    labs, breaks = segments(x, y, low, high)
    positive = y[y > 0]
    tiny = max(float(np.median(positive)) * 1e-6 if len(positive) else 1e-12, 1e-30)
    pilot = np.array(
        [
            np.median(
                y[
                    np.flatnonzero(labs == labs[i])[
                        np.argsort(np.abs(x[labs == labs[i]] - x[i]))[:3]
                    ]
                ]
            )
            for i in range(len(x))
        ]
    )
    pilot = np.maximum(pilot, tiny)
    z = np.log(y) if mode == "log" else y.copy()
    scale = (
        (1 / np.sqrt(pilot) if mode == "log" else np.sqrt(pilot))
        if mode != "linear_mean"
        else np.ones(len(x))
    )
    result = {}
    contrasts = []
    details = []
    for degree in (1, 2, 3):
        records = []
        for label in np.unique(labs):
            ids = np.flatnonzero(labs == label)
            for j in range(len(ids) - degree - 1):
                q = ids[j : j + degree + 2]
                a = polynomial_contrast(x[q], degree)
                standardizer = math.sqrt(float(np.sum(a * a * scale[q] ** 2)))
                value = float(a @ z[q]) / standardizer
                records.append(
                    {
                        "label": int(label),
                        "x": float(x[q].mean()),
                        "value": value,
                        "index_first": int(q[0]),
                        "index_last": int(q[-1]),
                        "degree": degree,
                    }
                )
        if not records:
            raise ValueError(
                "No contrasts: must use explicitly specified non-curve model"
            )
        rx = np.array([r["x"] for r in records])
        rv = np.array([r["value"] for r in records])
        rl = np.array([r["label"] for r in records])
        contrasts.extend(records)
        for window in WINDOWS:
            sigmas = []
            fallback = []
            biases = []
            for i in range(len(x)):
                eligible = np.flatnonzero(rl == labs[i])
                # Small isolated support islands borrow a curve-wide scale, not
                # a curve across a physical/censor gap. This assumption is flagged.
                if len(eligible) < 5:
                    eligible = np.arange(len(rv))
                    fallback.append(i)
                near = eligible[np.argsort(np.abs(rx[eligible] - x[i]))[:window]]
                s = robust_sigma(rv[near])
                sigmas.append(s * scale[i] * (y[i] if mode == "log" else 1.0))
                biases.append(float(abs(np.median(rv[near])) / max(s, 1e-30)))
            key = f"degree{degree}_window{window}"
            result[key] = np.square(sigmas)
            details.append(
                {
                    "variant": key,
                    "short_segment_global_scale_indices": fallback,
                    "median_contrast_offset_over_scale": float(np.median(biases)),
                }
            )
    # Quadratic annihilation, 15 neighboring contrasts is the central estimator;
    # all nine choices are retained to expose curvature/window sensitivity.
    central = result["degree2_window15"]
    stack = np.stack(list(result.values()))
    return {
        "central": central,
        "low": np.min(stack, axis=0),
        "high": np.max(stack, axis=0),
        "variants": result,
        "contrasts": contrasts,
        "segment_id": labs,
        "pilot": pilot,
        "breaks": breaks,
        "variant_details": details,
    }


def linear_three_five_noise(x, y, mode="linear_histogram", low=None, high=None):
    """Local linear trend model on original ordinates.

    Three points: a unit-norm contrast orthogonal to (1,x). At equal spacing
    r=(y[i-1]-2*y[i]+y[i+1])/sqrt(6), so the single variance probe is r^2.
    Five points: heteroscedastically weighted line fit, whitened RSS/(5-2).
    The local median of 9/15/23 such probes is divided by the corresponding
    median chi-square/dof, a declared Gaussian local-noise calibration.
    The trend fit uses exactly 3 or 5 points; pooling is a separate operation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    labs, breaks = segments(x, y, low, high)
    positive = y[y > 0]
    tiny = max(float(np.median(positive)) * 1e-6 if len(positive) else 1e-12, 1e-30)
    pilot = []
    for i in range(len(x)):
        eligible = np.flatnonzero(labs == labs[i])
        near = eligible[np.argsort(abs(x[eligible] - x[i]))[:3]]
        pilot.append(max(float(np.median(y[near])), tiny))
    pilot = np.array(pilot)
    scale = np.ones(len(x)) if mode == "linear_mean" else np.sqrt(pilot)
    variants = {}
    all_records = []
    details = []
    for points in (3, 5):
        records = []
        for label in np.unique(labs):
            ids = np.flatnonzero(labs == label)
            for j in range(len(ids) - points + 1):
                q = ids[j : j + points]
                u = (x[q] - x[q].mean()) / np.ptp(x[q])
                if points == 3:
                    a = polynomial_contrast(x[q], 1)
                    denominator = float(np.sum(a * a * scale[q] ** 2))
                    probe = float(a @ y[q]) ** 2 / denominator
                else:
                    design = np.column_stack([np.ones(points), u]) / scale[q, None]
                    target = y[q] / scale[q]
                    coeff = np.linalg.lstsq(design, target, rcond=None)[0]
                    probe = float(np.sum((target - design @ coeff) ** 2) / (points - 2))
                records.append(
                    {
                        "points": points,
                        "label": int(label),
                        "x": float(x[q].mean()),
                        "variance_probe": probe,
                        "index_first": int(q[0]),
                        "index_last": int(q[-1]),
                    }
                )
        if not records:
            raise ValueError("No local linear probes")
        rx = np.array([r["x"] for r in records])
        rv = np.array([r["variance_probe"] for r in records])
        rl = np.array([r["label"] for r in records])
        correction = median_chi2_per_dof(points - 2)
        for window in WINDOWS:
            variance = []
            fallback = []
            for i in range(len(x)):
                eligible = np.flatnonzero(rl == labs[i])
                if len(eligible) < 5:
                    eligible = np.arange(len(rv))
                    fallback.append(i)
                near = eligible[np.argsort(np.abs(rx[eligible] - x[i]))[:window]]
                variance.append(float(np.median(rv[near])) / correction * scale[i] ** 2)
            key = f"linear{points}points_pool{window}"
            variants[key] = np.array(variance)
            details.append(
                {
                    "variant": key,
                    "trend_points": points,
                    "pooled_variance_probes": window,
                    "Gaussian_median_probe_calibration": correction,
                    "short_segment_global_scale_indices": fallback,
                }
            )
        all_records.extend(records)
    return {
        "central": variants["linear3points_pool15"],
        "five_point_companion": variants["linear5points_pool15"],
        "variants": variants,
        "probes": all_records,
        "pilot": pilot,
        "segment_id": labs,
        "breaks": breaks,
        "variant_details": details,
    }


def normalize_covariance(mass, covariance):
    mass = np.asarray(mass, float)
    s = mass.sum()
    p = mass / s
    jac = (np.eye(len(mass)) - p[:, None]) / s
    cov = jac @ covariance @ jac.T
    return (cov + cov.T) / 2


def interpolation_matrix(x, target):
    """The exact np.interp endpoint policy used by retained arrays."""
    a = np.zeros((len(target), len(x)))
    for k, value in enumerate(target):
        if value <= x[0]:
            a[k, 0] = 1
        elif value >= x[-1]:
            a[k, -1] = 1
        else:
            j = np.searchsorted(x, value) - 1
            f = (value - x[j]) / (x[j + 1] - x[j])
            a[k, j] = 1 - f
            a[k, j + 1] = f
    return a


def reference_key(row):
    return next(
        k
        for k in (
            "reference_mass",
            "reference_mean_urad",
            "reference_count",
            "reference_gaussian_density_per_GeV",
        )
        if k in row
    )


def original_coordinates(row):
    return {
        k: row[k]
        for k in ("case", "process", "phi_rad", "bin_low", "bin_high")
        if k in row
    }


def curve_panel(name):
    """Return original reference observations and their retained-bin map."""
    if name.startswith("lep_final_focusing_"):
        path = DATA / (name.removesuffix("_pull") + ".csv")
        rr = sorted(read(path), key=lambda r: float(r["phi_rad"]))
        x = np.array([float(r["phi_rad"]) for r in rr])
        y = np.array([float(r["theta_before_minus_after_urad"]) for r in rr])
        return dict(
            path=path,
            rows=rr,
            x=x,
            y=y,
            mode="linear_mean",
            floor=np.full(len(y), 0.1 / math.sqrt(3)),
            floor_description="0.1 microradian graphical ordinate half-width, assumed uniform readout",
            source_url="https://arxiv.org/abs/1908.01704v4",
            source_locator="Fig. 3 left",
            transform="linear interpolation of mean ordinates",
            widths=None,
            low=None,
            high=None,
        )
    if name == "lep_focusing_distribution_pull":
        path = DATA / "lep_final_focusing_distribution.csv"
        rr = [r for r in read(path) if r["series"] == "all"]
        low = np.array([float(r["theta_lo_urad"]) for r in rr])
        high = np.array([float(r["theta_hi_urad"]) for r in rr])
        y = np.array([float(r["ordinate_au"]) for r in rr])
        floor = np.full(len(y), 0.01 * max(y) / math.sqrt(3))
        return dict(
            path=path,
            rows=rr,
            x=(low + high) / 2,
            y=y,
            low=low,
            high=high,
            widths=high - low,
            mode="linear_histogram",
            floor=floor,
            floor_description="1% of peak ordinate graphical half-width, assumed uniform readout",
            source_url="https://arxiv.org/abs/1908.01704v4",
            source_locator="Fig. 2 left",
            transform="density times exact graphical bin width, then conditional normalization",
        )
    if name.startswith("halhf_"):
        _, energy, scope, _ = name.split("_")
        path = DATA / f"halhf_lumi_{scope}_halhf{energy}.csv"
        rr = [
            r
            for r in read(path)
            if r["source_ordinate"]
            and float(r["energy_high_GeV"]) <= 1.02 * int(energy) + 1e-3
        ]
        low = np.array([float(r["energy_low_GeV"]) for r in rr])
        high = np.array([float(r["energy_high_GeV"]) for r in rr])
        mass = np.array([float(r["source_ordinate"]) for r in rr])
        y = mass / (high - low)
        return dict(
            path=path,
            rows=rr,
            x=(low + high) / 2,
            y=y,
            low=low,
            high=high,
            widths=high - low,
            original_mass=mass,
            mode="log",
            floor=0.03 * y / math.sqrt(3),
            floor_description="3% relative graphical ordinate half-width, assumed uniform readout",
            source_url="https://indico.desy.de/event/55312/contributions/212416/attachments/106566/149673/PreCDR_final_LDG.pdf",
            source_locator=rr[0]["source_figure"],
            transform="source fractional-bin ordinates divided by widths for noise estimation, multiplied back and conditionally normalized",
        )
    if name.startswith("ilc_1000_"):
        scope = name.split("_")[2]
        path = DATA / f"ilc_lumi_{scope}_steps.csv"
        rr = [
            r
            for r in read(path)
            if r["collider"] == "ILC" and r["display_floor_or_baseline"] == "0"
        ]
        low = np.array([float(r["sqrt_s_low_GeV"]) for r in rr])
        high = np.array([float(r["sqrt_s_high_GeV"]) for r in rr])
        y = np.array([float(r["density_per_GeV"]) for r in rr])
        floor = []
        for r, yi in zip(rr, y):
            half = float(r["approx_y_halfwidth"])
            floor.append(
                yi * math.log(10) * half / math.sqrt(3)
                if r["approx_y_halfwidth_units"] == "dex"
                else half / float(r["nominal_sqrt_s_GeV"]) / math.sqrt(3)
            )
        return dict(
            path=path,
            rows=rr,
            x=(low + high) / 2,
            y=y,
            low=low,
            high=high,
            widths=high - low,
            mode="log",
            floor=np.array(floor),
            floor_description="Source extraction ordinate half-width: full 0.008 dex, peak 0.6 in dN/dx; uniform-readout variance and delta-method unit conversion",
            source_url="https://arxiv.org/abs/1411.4924v1",
            source_locator="Fig. 49",
            transform="original density times graphical segment width, then conditional normalization; coalesced equal segments are not interpreted as event counts",
        )
    raise KeyError(name)
