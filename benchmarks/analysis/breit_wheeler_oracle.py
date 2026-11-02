#!/usr/bin/env python3
"""Tabulate the unpolarized real-photon Breit-Wheeler cross section.

Here E_star=sqrt(s)/2 is one photon's centre-of-mass energy in GeV.
For head-on photons s=4 E_gamma1 E_gamma2. Cross sections are in m^2.
The helicity-averaged expression is independently implemented in WarpX's
LinearBreitWheelerCrossSection.H, citing Gould and Schreder, Phys. Rev. 155
(1967) 1404. This deterministic check does not validate a collider photon flux.
"""

import argparse
import json
import math
from pathlib import Path

ELECTRON_MASS_GEV = 0.51099906e-3
CLASSICAL_RADIUS_M = 2.81794092e-15
SOURCE = "https://warpx.readthedocs.io/en/26.08/_static/doxyhtml/_linear_breit_wheeler_cross_section_8_h_source.html"


def cross_section(energy_star_gev):
    """Return the helicity-averaged cross section, including its threshold."""
    if not math.isfinite(energy_star_gev) or energy_star_gev < 0:
        raise ValueError("Photon CM energy must be finite and nonnegative")
    if energy_star_gev <= ELECTRON_MASS_GEV:
        return 0.0
    inverse_gamma2 = (ELECTRON_MASS_GEV / energy_star_gev) ** 2
    beta = math.sqrt(1 - inverse_gamma2)
    # (1-beta) = g/(1+beta) avoids subtracting nearly equal numbers.
    log_ratio = 2 * math.log1p(beta) - math.log(inverse_gamma2)
    bracket = (2 + 2 * inverse_gamma2 - inverse_gamma2 ** 2) * log_ratio - 2 * beta * (
        1 + inverse_gamma2
    )
    return 0.5 * math.pi * CLASSICAL_RADIUS_M ** 2 * inverse_gamma2 * bracket


def audit():
    """Separate the former normalization defect from unresolved collider yields."""
    prefactor = 0.5 * math.pi * CLASSICAL_RADIUS_M ** 2
    ratios = [
        0.0,
        0.5,
        1.0,
        math.nextafter(1.0, math.inf),
        1.000001,
        1.01,
        1.1,
        2.0,
        10.0,
        100.0,
        10000.0,
        500000.0,
    ]
    rows = [
        {
            "energy_star_over_mass": ratio,
            "energy_star_GeV": ratio * ELECTRON_MASS_GEV,
            "cross_section_m2": cross_section(ratio * ELECTRON_MASS_GEV),
        }
        for ratio in ratios
    ]
    return {
        "formula": "pi*r_e^2/2 * (1-beta^2) * [(3-beta^4)*log((1+beta)/(1-beta)) - 2*beta*(2-beta^2)]",
        "invariant": "s=4*E_star^2; beta=sqrt(1-4*m_e^2/s)",
        "photon_polarization": "unpolarized; averaged over incoming helicities",
        "source": SOURCE,
        "prefactor_m2": prefactor,
        "previous_literal_m2": 1.23088e-29,
        "previous_normalization_relative_error": 1.23088e-29 / prefactor - 1,
        "expected_increase_at_fixed_photon_luminosity": prefactor / 1.23088e-29 - 1,
        "interpretation": "The old prefactor lowers the rate. Its correction cannot explain an already positive 16 percent excess.",
        "remaining_causes_to_test": [
            "independent collision scatter",
            "real-photon energy/overlap spectrum",
            "transverse domain and cell spacing",
            "energy cuts and recorded-lepton versus produced-pair definitions",
        ],
        "excluded_explanations": [
            "virtual-photon Qmax choice for a real-real subprocess",
            "secondary orbit changes as an explanation of the production count",
        ],
        "points": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(audit(), stream, indent=2, allow_nan=False)
        stream.write("\n")
