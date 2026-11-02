"""Head-on Gaussian IP-dispersion toy; exact four-vector invariant mass.

It checks a derivation, not a lattice, beam-beam field, crab-waist or RF design.
"""

from __future__ import annotations

import math
import random

from .common import finite, integer, keys, require

ELECTRON_MASS_GEV = 0.00051099895069


def conditional_covariance(s1, s2, d1, d2, sx1, sx2):
    """Cov(delta | collision overlap) = S - S a a^T S/(v+a^T S a).

    a=(D1,-D2), v=sigma_xbeta1^2+sigma_xbeta2^2. Overlap
    weight exp[-(D1 delta1-D2 delta2)^2/(2v)] is analytic.
    """
    require(
        s1 > 0 and s2 > 0 and sx1 > 0 and sx2 > 0,
        "All spreads and betatron widths must be positive",
    )
    v = sx1 * sx1 + sx2 * sx2
    a1, a2 = d1, -d2
    denom = v + a1 * a1 * s1 * s1 + a2 * a2 * s2 * s2
    q1, q2 = s1 * s1 * a1, s2 * s2 * a2
    covariance = [
        [s1 * s1 - q1 * q1 / denom, -q1 * q2 / denom],
        [-q1 * q2 / denom, s2 * s2 - q2 * q2 / denom],
    ]
    variance_sum = covariance[0][0] + 2 * covariance[0][1] + covariance[1][1]
    overlap = math.sqrt(v / denom)
    return covariance, variance_sum, overlap


def invariant_mass(e1, e2):
    require(e1 >= ELECTRON_MASS_GEV and e2 >= ELECTRON_MASS_GEV, "Energy below electron rest mass")
    p1 = math.sqrt((e1 - ELECTRON_MASS_GEV) * (e1 + ELECTRON_MASS_GEV))
    p2 = math.sqrt((e2 - ELECTRON_MASS_GEV) * (e2 + ELECTRON_MASS_GEV))
    return math.sqrt(2 * ELECTRON_MASS_GEV**2 + 2 * (e1 * e2 + p1 * p2))


def simulate(config):
    keys(
        config,
        (
            "schema",
            "seed",
            "samples",
            "energy1_GeV",
            "energy2_GeV",
            "sigma_delta1",
            "sigma_delta2",
            "dispersion1_m",
            "dispersion2_m",
            "sigma_xbeta1_m",
            "sigma_xbeta2_m",
        ),
    )
    require(config["schema"] == "feebl.gaussian_dispersion_toy.v1", "Unknown toy schema")
    n = integer(config["samples"], "samples", 1000)
    require(n <= 2000000, "Toy is limited to two million samples")
    rng = random.Random(integer(config["seed"], "seed", 0))
    names = (
        "energy1_GeV",
        "energy2_GeV",
        "sigma_delta1",
        "sigma_delta2",
        "dispersion1_m",
        "dispersion2_m",
        "sigma_xbeta1_m",
        "sigma_xbeta2_m",
    )
    e1, e2, s1, s2, d1, d2, sx1, sx2 = [finite(config[name], name) for name in names]
    require(e1 > ELECTRON_MASS_GEV and e2 > ELECTRON_MASS_GEV, "Unphysical energy")
    cov, variance_sum, overlap = conditional_covariance(s1, s2, d1, d2, sx1, sx2)
    nominal = invariant_mass(e1, e2)
    # Sample the analytic collision-conditioned Gaussian by Cholesky. This is
    # an equation implementation check, not an independent full-code oracle.
    l11 = math.sqrt(cov[0][0])
    l21 = cov[1][0] / l11
    l22 = math.sqrt(max(0, cov[1][1] - l21 * l21))
    w = []
    delta_sums = []
    x1_values, x2_values = [], []
    for _ in range(n):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        q1, q2 = l11 * a, l21 * a + l22 * b
        w.append(invariant_mass(e1 * (1 + q1), e2 * (1 + q2)))
        delta_sums.append(q1 + q2)
        x1_values.append(q1)
        x2_values.append(q2)
    mean = sum(w) / n
    var = sum((value - mean) ** 2 for value in w) / (n - 1)
    mean_delta = sum(delta_sums) / n
    measured_variance_sum = sum((x - mean_delta) ** 2 for x in delta_sums) / (n - 1)
    means = (sum(x1_values) / n, sum(x2_values) / n)
    measured_cov = [
        [
            sum(
                (a - means[i]) * (b - means[j])
                for a, b in zip((x1_values, x2_values)[i], (x1_values, x2_values)[j])
            )
            / (n - 1)
            for j in range(2)
        ]
        for i in range(2)
    ]
    return {
        "schema": "feebl.toy_result.v1",
        "classification": "SYNTHETIC_ONLY",
        "status": "DERIVATION_WORKSHOP_NOT_MACHINE_FEASIBILITY",
        "seed": config["seed"],
        "samples": n,
        "nominal_invariant_ecm_GeV": nominal,
        "analytic_conditional_covariance": cov,
        "measured_conditional_covariance": measured_cov,
        "analytic_variance_delta_sum": variance_sum,
        "measured_variance_delta_sum": measured_variance_sum,
        "analytic_linearized_sigma_W_GeV": nominal * math.sqrt(variance_sum) / 2,
        "exact_fourvector_mean_W_GeV": mean,
        "exact_fourvector_sigma_W_GeV": math.sqrt(var),
        "analytic_relative_geometric_luminosity_overlap": overlap,
        "limitations": [
            "Head-on, collinear, centered Gaussian bunches with linear dispersion and no angular correlations",
            "No beamstrahlung, disruption, radiation, RF, lattice tolerances, spin, crab crossing or crab waist",
            "Sampling uses analytic conditioned covariance; this is not independent beam-beam validation",
            "General asymmetric nominal ECM uses the exact four-vector invariant, not E1+E2",
            "Linearized sigma_W formula is ultrarelativistic; exact invariant is evaluated sample by sample",
        ],
    }
