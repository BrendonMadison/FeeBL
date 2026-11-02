"""Independent light-cone and rest-frame Klein-Nishina checks; no simulation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp

mp.mp.dps = 60
MASS = 0.00051099895


def stable_cdf(x, t):
    y = -math.expm1(-t)
    if t < 0.01:
        y_minus_t = math.fsum(
            (-1) ** (n + 1) * t ** n / math.factorial(n) for n in range(2, 20)
        )
        sinh_minus_t = math.fsum(t ** n / math.factorial(n) for n in range(3, 22, 2))
    else:
        y_minus_t, sinh_minus_t = y - t, math.sinh(t) - t
    return t + y - 0.5 * y * y + 4 * y_minus_t / x + 8 * sinh_minus_t / x ** 2


def angular_integral(x, t):
    """Integrate qÂ²(q+1/qâˆ’sinÂ²Ïˆ) dcosÏˆ in the electron rest frame."""
    x, t = mp.mpf(x), mp.mpf(t)
    rmax = mp.expm1(t) / x

    def kernel(r):
        q = 1 / (1 + x * r)
        return q ** 2 * (q + 1 / q - 4 * r * (1 - r))

    # r=(1-cosÏˆ)/2. Resolve the forward peak without using the candidate CDF.
    breaks = [mp.mpf(0), rmax]
    for exponent in range(-24, 1, 2):
        point = mp.power(10, exponent)
        if 0 < point < rmax:
            breaks.append(point)
    return x * mp.quad(kernel, sorted(set(breaks)))


def evaluate():
    started = time.monotonic()
    kinematics, quadrature = [], []
    for energy in (MASS, MASS * (1 + 1e-12), 0.001, 1.0, 125.0, 1e6):
        momentum = math.sqrt((energy - MASS) * (energy + MASS))
        a = energy + momentum
        b = MASS * MASS / a
        for photon_energy in (1e-18, 1e-12, 1e-6, 0.001, 1.0, 125.0, 1e8):
            x = 2 * photon_energy / b
            tmax = math.log1p(x)
            for fraction in (0.0, 1e-12, 0.1, 0.5, 1 - 1e-12, 1.0):
                t = fraction * tmax
                u, y = math.exp(-t), -math.expm1(-t)
                d = math.expm1(tmax - t)
                recoil_plus = a * u
                recoil_minus = b * (1 + y) + 2 * photon_energy * y
                photon_plus, photon_minus = a * y, b * d
                pt2 = MASS * MASS * y * d
                recoil_energy = (recoil_plus + recoil_minus) / 2
                recoil_pz = (recoil_plus - recoil_minus) / 2
                velocity_squared = (pt2 + recoil_pz ** 2) / recoil_energy ** 2
                kinematics.append(
                    {
                        "energy_GeV": energy,
                        "photon_energy_GeV": photon_energy,
                        "x": x,
                        "t_fraction": fraction,
                        "plus_closure": abs(recoil_plus + photon_plus - a) / a,
                        "minus_closure": abs(
                            recoil_minus + photon_minus - b - 2 * photon_energy
                        )
                        / (b + 2 * photon_energy),
                        "recoil_shell_scaled": abs(
                            recoil_plus * recoil_minus - pt2 - MASS ** 2
                        )
                        / max(MASS ** 2, recoil_plus * recoil_minus, pt2),
                        "photon_shell_scaled": abs(photon_plus * photon_minus - pt2)
                        / max(MASS ** 2, photon_plus * photon_minus, pt2),
                        "recoil_energy_minus_mass_GeV": recoil_energy - MASS,
                        "beta_squared": velocity_squared,
                        "d": d,
                        "direct_endpoint_d": x * (1 - y) - y,
                    }
                )
    for x in (1e-12, 1e-8, 1e-4, 0.1, 1.0, 10.0, 1e4, 1e8, 1e12):
        for fraction in (0.001, 0.1, 0.5, 1.0):
            t = fraction * math.log1p(x)
            numeric = angular_integral(x, t)
            analytic = stable_cdf(x, t)
            error = abs(mp.mpf(analytic) - numeric) / numeric
            quadrature.append(
                {
                    "x": x,
                    "t_fraction": fraction,
                    "cdf": analytic,
                    "angular_integral": float(numeric),
                    "relative_error": float(error),
                }
            )
    summary = {
        "kinematic_cases": len(kinematics),
        "quadrature_cases": len(quadrature),
        "maximum_residuals": {
            key: max(row[key] for row in kinematics)
            for key in (
                "plus_closure",
                "minus_closure",
                "recoil_shell_scaled",
                "photon_shell_scaled",
            )
        },
        "maximum_beta_squared": max(row["beta_squared"] for row in kinematics),
        "minimum_recoil_energy_minus_mass_GeV": min(
            row["recoil_energy_minus_mass_GeV"] for row in kinematics
        ),
        "maximum_cdf_relative_error": max(row["relative_error"] for row in quadrature),
        "negative_naive_d_cases": sum(
            row["direct_endpoint_d"] < 0 for row in kinematics
        ),
        "negative_stable_d_cases": sum(row["d"] < 0 for row in kinematics),
        "elapsed_seconds": time.monotonic() - started,
    }
    result = {
        "summary": summary,
        "kinematics": kinematics,
        "quadrature": quadrature,
        "normalization": "F=(x/2) times the polar-angle integral of the unpolarized rest-frame KN angular factor, omitting its common r_eÂ²/2 and azimuth factors.",
        "scope": "Independent equations and double-precision reconstruction audit; native code not executed.",
    }
    result["electron_mass_GeV"] = MASS
    result["quadrature_decimal_precision"] = mp.mp.dps
    result["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["sources"] = [
        {
            "title": "Geant4 Physics Reference Manual: Compton scattering",
            "url": "https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html",
            "locator": "Sampling the Final State",
            "supports": "Free-electron Klein-Nishina energy distribution and rest-frame Compton kinematics.",
        },
        {
            "title": "Geant4 Physics Reference Manual: Compton scattering, Penelope model",
            "url": "https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/penelope_compton.html",
            "locator": "Equation (5), Klein-Nishina differential cross section",
            "supports": "The free-electron angular kernel used for the independent quadrature; atomic binding terms are omitted.",
        },
    ]
    result["assumptions"] = [
        "Head-on incoming photon and free electron; unpolarized Klein-Nishina scattering.",
        "Natural units c=1; energies and momenta in GeV; plus/minus denote E +/- p_z.",
        "The angular quadrature uses r=(1-cos(psi))/2 independently of the candidate CDF primitive.",
        "Mass-shell residuals use light-cone products and are scaled by the largest shell term or m^2.",
        "Kinematic and CDF ranges are the explicit test points; no polarized or bound-electron cross section is checked.",
        "This equation audit does not test the native random generator or its sampled distribution.",
    ]
    result["checks"] = {
        "light_cone_and_shell_residuals_below_1e_minus_12": all(
            value < 1e-12 for value in summary["maximum_residuals"].values()
        ),
        "cdf_relative_error_below_2e_minus_12": summary["maximum_cdf_relative_error"]
        < 2e-12,
        "nonnegative_stable_transverse_squared": summary["negative_stable_d_cases"]
        == 0,
        "recoil_energy_above_mass": summary["minimum_recoil_energy_minus_mass_GeV"]
        >= -8 * math.ulp(MASS),
        "velocity_squared_within_roundoff": summary["maximum_beta_squared"]
        <= 1 + 8 * math.ulp(1.0),
    }
    result["status"] = (
        "PASS_FOR_STATED_EQUATION_CHECKS" if all(result["checks"].values()) else "FAIL"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], **result["summary"]}, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
