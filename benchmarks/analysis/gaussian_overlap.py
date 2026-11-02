"""Rigid Gaussian collision overlap with one common spatial-compensation fraction.

Lengths/emittances/betas are in metres; angles are radians; energies explicitly
use GeV. Beam emittances are geometric, not normalized. No native simulation,
fit, RF transport, crab-waist/hourglass weighting, radiation or ISR is included.

For alpha=theta/2 take mean velocity unit vectors n1=(s,0,c), n2=(s,0,-c)
and horizontal axes h1=(c,0,-s), h2=(c,0,s). In EACH beam's own axes the
rigid spatial shear is x = x_beta + D*delta - q*tan(alpha)*zeta. Thus the
lab horizontal coordinate is c*(x_beta+D*delta)+(1-q)*s*zeta. q=1 makes
the spatial bunch axes parallel; it does not rotate either particle momentum.
Actual theta, never (1-q)*theta, enters invariant energy and angular variance.

All distributions are Gaussian at a horizontal/vertical waist, with independent
energy, betatron position, longitudinal position and particle angles. In
particular the canonical zeta += D*px map and energy-angle correlations are
excluded. Individual angular errors enter kinematics but do not change the
rigid mean-velocity overlap/flux. This is an input model, not a machine design.
"""
from dataclasses import dataclass, asdict, replace
import math
import numpy as np

MODEL_VERSION = "feebl.common-rigid-gaussian-overlap/1"
FWHM_OVER_SIGMA = math.sqrt(8 * math.log(2))


@dataclass(frozen=True)
class Beam:
    energy_GeV: float
    population: float
    epsilon_x_m: float
    epsilon_y_m: float
    beta_x_m: float
    beta_y_m: float
    sigma_z_m: float
    sigma_delta: float

    def validate(self):
        for key, value in asdict(self).items():
            if (
                not math.isfinite(value)
                or value < 0
                or (key != "sigma_delta" and value == 0)
            ):
                raise ValueError("Invalid beam " + key)
        return self

    @property
    def x2(self):
        return self.epsilon_x_m * self.beta_x_m

    @property
    def y2(self):
        return self.epsilon_y_m * self.beta_y_m

    @property
    def ax2(self):
        return self.epsilon_x_m / self.beta_x_m

    @property
    def ay2(self):
        return self.epsilon_y_m / self.beta_y_m


def _inputs(beams, full_crossing_angle_rad, compensation, collision_rate_hz):
    beams = tuple(
        b.validate() if isinstance(b, Beam) else Beam(**b).validate() for b in beams
    )
    if len(beams) != 2:
        raise ValueError("Exactly two beams are required")
    if (
        not math.isfinite(full_crossing_angle_rad)
        or not 0 <= full_crossing_angle_rad < math.pi
    ):
        raise ValueError("Full crossing angle must be finite in [0, pi)")
    if not math.isfinite(compensation) or not 0 <= compensation <= 1:
        raise ValueError("Common spatial compensation must be in [0, 1]")
    if not math.isfinite(collision_rate_hz) or collision_rate_hz <= 0:
        raise ValueError("Positive finite per-IP collision rate required")
    return beams


def retune_to_W(beams, full_crossing_angle_rad, target_W_GeV=125.0):
    """Scale both mean energies equally; preserve ratio and geometric optics."""
    beams = _inputs(beams, full_crossing_angle_rad, 0.0, 1.0)
    if not math.isfinite(target_W_GeV) or target_W_GeV <= 0:
        raise ValueError("Positive finite target invariant energy required")
    current = (
        2
        * math.sqrt(beams[0].energy_GeV * beams[1].energy_GeV)
        * math.cos(full_crossing_angle_rad / 2)
    )
    scale = target_W_GeV / current
    return tuple(replace(b, energy_GeV=b.energy_GeV * scale) for b in beams)


def _dispersion(value):
    if np.isscalar(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Scalar dispersion is a finite nonnegative opposite-sign magnitude"
            )
        d = np.array([value, -value], dtype=float)
    else:
        d = np.asarray(value, dtype=float)
        if d.shape != (2,) or not np.all(np.isfinite(d)):
            raise ValueError("Dispersion pair must have two finite entries")
    return d


def conditional_energy_factor(beams, dispersion_pair_m, B_m2):
    """Return F with collision-conditioned C_delta=F F^T, stably at large D.

    Project in standardized energy coordinates onto the constrained direction
    and its perpendicular. This avoids subtracting nearly equal variances.
    """
    sd = np.array([b.sigma_delta for b in beams])
    a = np.array([dispersion_pair_m[0], -dispersion_pair_m[1]])
    w = sd * a
    norm = math.hypot(*w)
    if norm == 0:
        return np.diag(sd)
    unit = w / norm
    orthogonal = np.array([-unit[1], unit[0]])
    reduction = math.sqrt(B_m2) / math.hypot(math.sqrt(B_m2), norm)
    return sd[:, None] * np.column_stack((unit * reduction, orthogonal))


def forward(
    beams,
    dispersion_m,
    full_crossing_angle_rad,
    compensation,
    collision_rate_hz,
    *,
    nonlinear_order=None
):
    """Return absolute L, conditional energy covariance and linear width.

    Scalar D means (D1,D2)=(+D,-D); a pair permits unequal or same-sign D.
    The reported Gaussian FWHM is 2*sqrt(2*ln2)*linear_sigma, not an observed
    or fitted native FWHM. Optional nonlinear moments are diagnostics only.
    """
    beams = _inputs(beams, full_crossing_angle_rad, compensation, collision_rate_hz)
    d = _dispersion(dispersion_m)
    alpha = full_crossing_angle_rad / 2
    tan_a = math.tan(alpha)
    B = sum(b.x2 for b in beams) + ((1 - compensation) * tan_a) ** 2 * sum(
        b.sigma_z_m ** 2 for b in beams
    )
    Y = sum(b.y2 for b in beams)
    dispersive = sum((di * b.sigma_delta) ** 2 for di, b in zip(d, beams))
    denom = B + dispersive
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("Nonfinite or singular overlap variance")
    F = conditional_energy_factor(beams, d, B)
    # F^T u keeps the very small constrained variance stable.
    energy_relative_variance = float(np.dot(F.T @ np.ones(2), F.T @ np.ones(2)))
    W0 = 2 * math.sqrt(beams[0].energy_GeV * beams[1].energy_GeV) * math.cos(alpha)
    energy2 = (W0 / 2) ** 2 * energy_relative_variance
    angular2 = (W0 / 2 * tan_a) ** 2 * sum(b.ax2 for b in beams)
    L_BX = math.prod(b.population for b in beams) / (2 * math.pi * math.sqrt(denom * Y))
    sigma = math.sqrt(energy2 + angular2)
    out = dict(
        schema=MODEL_VERSION,
        status="INPUT_ONLY_RIGID_GAUSSIAN_MODEL",
        beams=[asdict(b) for b in beams],
        dispersion_m=d.tolist(),
        full_crossing_angle_rad=full_crossing_angle_rad,
        compensation=compensation,
        collision_rate_hz=collision_rate_hz,
        nominal_W_GeV=W0,
        betatron_horizontal_sum_m2=sum(b.x2 for b in beams),
        residual_geometric_horizontal_sum_m2=B,
        dispersive_horizontal_sum_m2=dispersive,
        conditional_delta_covariance=(F @ F.T).tolist(),
        conditional_delta_factor=F.tolist(),
        luminosity_m2_per_crossing=L_BX,
        luminosity_cm2_s=L_BX * collision_rate_hz / 1e4,
        energy_sigma_MeV=1000 * math.sqrt(energy2),
        angular_sigma_MeV=1000 * math.sqrt(angular2),
        linear_sigma_MeV=1000 * sigma,
        linear_gaussian_FWHM_MeV=1000 * FWHM_OVER_SIGMA * sigma,
        linear_angular_floor_FWHM_MeV=1000 * FWHM_OVER_SIGMA * math.sqrt(angular2),
        angular_floor_interpretation="Floor only of the uncorrelated first-order model; not a universal physical bound.",
        nonlinear_validity="At extreme narrowing the energy-product and second-order angular terms can dominate. A linear Gaussian FWHM then does not describe the exact invariant distribution.",
        geometric_compensation_is_RF_transport=False,
        actual_angle_retained_in_kinematics=True,
    )
    if nonlinear_order is not None:
        out["nonlinear_moment_diagnostic"] = nonlinear_moments(
            beams, F, full_crossing_angle_rad, nonlinear_order
        )
    return out


def inverse_target(
    beams,
    target_MeV,
    full_crossing_angle_rad,
    compensation,
    collision_rate_hz,
    *,
    quantity="FWHM"
):
    """Invert the opposite-equal-D LINEAR model, never a physical feasibility claim."""
    if (
        quantity not in ("FWHM", "sigma")
        or not math.isfinite(target_MeV)
        or target_MeV <= 0
    ):
        raise ValueError("Positive finite target and quantity FWHM or sigma required")
    zero = forward(beams, 0.0, full_crossing_angle_rad, compensation, collision_rate_hz)
    target_sigma = target_MeV / (FWHM_OVER_SIGMA if quantity == "FWHM" else 1)
    floor = zero["angular_sigma_MeV"]
    out = dict(
        target_MeV=target_MeV,
        quantity=quantity,
        model_linear_angular_sigma_floor_MeV=floor,
        interpretation="Inverse of the input-only linear Gaussian model; no RF, lattice, native or nonlinear feasibility claim.",
    )
    if target_sigma <= floor:
        return dict(
            out,
            status="UNATTAINABLE_AT_OR_BELOW_LINEAR_ANGULAR_FLOOR",
            dispersion_m=None,
        )
    if target_sigma >= zero["linear_sigma_MeV"]:
        return dict(
            out,
            status="ZERO_DISPERSION_ALREADY_AT_OR_BELOW_TARGET",
            dispersion_m=0.0,
            prediction=zero,
        )
    variance_sum = sum(b["sigma_delta"] ** 2 for b in zero["beams"])
    desired_energy2 = target_sigma ** 2 - floor ** 2
    D2 = (
        zero["residual_geometric_horizontal_sum_m2"]
        / variance_sum
        * (zero["energy_sigma_MeV"] ** 2 / desired_energy2 - 1)
    )
    if D2 < 0 or not math.isfinite(D2):
        raise ArithmeticError("Invalid inverse variance after explicit target checks")
    D = math.sqrt(D2)
    prediction = forward(
        beams, D, full_crossing_angle_rad, compensation, collision_rate_hz
    )
    return dict(
        out,
        status="FINITE_DISPERSION_IN_LINEAR_MODEL_ONLY",
        dispersion_m=D,
        prediction=prediction,
    )


def spacetime_overlap_oracle(
    beams, dispersion_m, full_crossing_angle_rad, compensation, collision_rate_hz
):
    """Independent 3D convolution followed by exact time Gaussian integral.

    Integral rho1(r-v1*t)rho2(r-v2*t) d^3r d(ct)
      = 1/[2*pi*sqrt(det(S1+S2))*sqrt(dn^T(S1+S2)^-1 dn)].
    Multiply by massless Moller flux 1-n1.dot(n2), not |n1-n2|.
    """
    beams = _inputs(beams, full_crossing_angle_rad, compensation, collision_rate_hz)
    ds = _dispersion(dispersion_m)
    a = full_crossing_angle_rad / 2
    s, c = math.sin(a), math.cos(a)
    n = (np.array([s, 0, c]), np.array([s, 0, -c]))
    h = (np.array([c, 0, -s]), np.array([c, 0, s]))
    cov = np.zeros((3, 3))
    for b, d, ni, hi in zip(beams, ds, n, h):
        longitudinal = ni - compensation * math.tan(a) * hi
        cov += (b.x2 + (d * b.sigma_delta) ** 2) * np.outer(hi, hi)
        cov += b.y2 * np.diag([0.0, 1.0, 0.0])
        cov += b.sigma_z_m ** 2 * np.outer(longitudinal, longitudinal)
    # Scale each spatial coordinate before determinant/inverse calculation.
    scale = np.sqrt(np.diag(cov))
    scaled = cov / np.outer(scale, scale)
    sign, logdet = np.linalg.slogdet(scaled)
    if sign <= 0:
        raise ArithmeticError("Nonpositive spatial covariance")
    dn = (n[0] - n[1]) / scale
    time_precision = float(dn @ np.linalg.solve(scaled, dn))
    logden = (
        math.log(2 * math.pi)
        + np.log(scale).sum()
        + 0.5 * logdet
        + 0.5 * math.log(time_precision)
    )
    flux = 1 - float(n[0] @ n[1])
    return (
        math.prod(b.population for b in beams)
        * flux
        * math.exp(-logden)
        * collision_rate_hz
        / 1e4
    )


def nonlinear_moments(beams, delta_factor, full_crossing_angle_rad, order=12):
    """Deterministic exact massless-invariant moments under this factorized model.

    Independent 2D energy and 4D slope Gauss-Hermite rules avoid 6D allocation.
    Stable centered moments avoid subtracting nearly equal E[W^2] and E[W]^2.
    This is an invariant-distribution RMS, NEVER a Gaussian peak fit or FWHM.
    """
    if not isinstance(order, int) or not 4 <= order <= 24:
        raise ValueError("Quadrature order must be integer 4..24")
    beams = _inputs(beams, full_crossing_angle_rad, 0.0, 1.0)
    F = np.asarray(delta_factor, dtype=float)
    if F.shape != (2, 2) or not np.all(np.isfinite(F)):
        raise ValueError("Finite 2x2 delta factor required")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = nodes * math.sqrt(2)
    weights = weights / math.sqrt(math.pi)
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    pw = np.outer(weights, weights).ravel()
    delta = F @ np.vstack((u.ravel(), v.ravel()))
    if np.any(delta <= -1):
        raise ValueError("Energy Gaussian reaches nonphysical quadrature nodes")
    ea = np.sqrt((1 + delta[0]) * (1 + delta[1]))
    da = ea - 1
    ma = float(pw @ da)
    va = float(pw @ (da - ma) ** 2)
    a = full_crossing_angle_rad / 2
    s, c = math.sin(a), math.cos(a)
    n1 = np.array([s, 0, c])
    n2 = np.array([s, 0, -c])
    h1 = np.array([c, 0, -s])
    h2 = np.array([c, 0, s])
    ey = np.array([0.0, 1.0, 0.0])
    directions = []
    for b, n, h in zip(beams, (n1, n2), (h1, h2)):
        ax = u.ravel() * math.sqrt(b.ax2)
        ay = v.ravel() * math.sqrt(b.ay2)
        raw = n + ax[:, None] * h + ay[:, None] * ey
        directions.append(raw / np.sqrt(1 + ax * ax + ay * ay)[:, None])
    dot = directions[0] @ directions[1].T
    ab = np.sqrt(np.maximum(0.0, (1 - dot) / (2 * c * c)))
    db = ab - 1
    angular_weights = np.outer(pw, pw)
    mb = float(np.sum(angular_weights * db))
    vb = float(np.sum(angular_weights * (db - mb) ** 2))
    W0 = 2 * math.sqrt(beams[0].energy_GeV * beams[1].energy_GeV) * c
    variance = va * ((1 + mb) ** 2 + vb) + (1 + ma) ** 2 * vb
    variance_diff = float(
        np.dot(F.T @ np.array([1.0, -1.0]), F.T @ np.array([1.0, -1.0]))
    )
    return dict(
        order_per_dimension=order,
        energy_nodes=order ** 2,
        angular_nodes=order ** 4,
        mean_shift_MeV=1000 * W0 * (ma + mb + ma * mb),
        exact_invariant_RMS_MeV=1000 * W0 * math.sqrt(variance),
        energy_only_RMS_MeV=1000 * W0 * math.sqrt(va),
        angular_only_RMS_MeV=1000 * W0 * math.sqrt(vb),
        head_on_second_order_energy_sigma_MeV=1000 * W0 * variance_diff / math.sqrt(32),
        head_on_second_order_angular_sigma_MeV=1000
        * W0
        * math.sqrt(sum(b.ax2 for b in beams) ** 2 + sum(b.ay2 for b in beams) ** 2)
        / math.sqrt(32),
        interpretation="Factorized Gaussian-model exact massless-invariant RMS; not a fitted response, observed FWHM, or universally physical floor.",
    )
