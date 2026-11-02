#!/usr/bin/env python3
"""Recompute the three baseline rigid-Gaussian curves from beam parameters."""

import argparse
import json
from pathlib import Path

import numpy as np

from comparison_statistics import INPUTS, write_rows
from gaussian_overlap import Beam, forward, retune_to_W, spacetime_overlap_oracle


def machines():
    """Bind SI beam parameters and documented crossing/tilt assumptions."""
    inputs = json.loads((INPUTS / "gaussian_overlap_inputs.json").read_text())
    result = {}
    for machine in ("ilc", "halhf"):
        source = inputs[machine]
        beams = tuple(
            Beam(
                b["energy_GeV"],
                b["population"],
                b["geometric_emittance_x_m_rad"],
                b["geometric_emittance_y_m_rad"],
                b["beta_x_m"],
                b["beta_y_m"],
                b["sigma_z_m"],
                b["sigma_delta_fraction"],
            )
            for b in (source["beam_minus"], source["beam_plus"])
        )
        angle, tilt = (0.014, 1.0) if machine == "ilc" else (0.0, 0.0)
        result[machine] = (
            retune_to_W(beams, angle),
            angle,
            tilt,
            source["collisions_per_second"],
        )
    b = inputs["fcc"]
    beam = Beam(
        b["beam_energy_GeV"],
        b["population_per_beam"],
        b["geometric_emittance_x_m"],
        b["geometric_emittance_y_m"],
        b["beta_x_m"],
        b["beta_y_m"],
        b["sigma_z_m"],
        b["relative_energy_spread"],
    )
    result["fcc"] = (
        retune_to_W((beam, beam), 0.03),
        0.03,
        0.0,
        b["collision_rate_per_IP_Hz"],
    )
    return result


def generate():
    """Evaluate fixed baselines; nonlinear RMS only marks linear-model validity."""
    rows = []
    for machine, (beams, angle, tilt, rate) in machines().items():
        for dispersion_mm in np.geomspace(0.1, 30000, 401):
            value = forward(
                beams, dispersion_mm / 1000, angle, tilt, rate, nonlinear_order=8
            )
            oracle = spacetime_overlap_oracle(
                beams, dispersion_mm / 1000, angle, tilt, rate
            )
            if not np.isclose(value["luminosity_cm2_s"], oracle, rtol=1e-10):
                raise ArithmeticError(
                    "Independent spacetime integral disagrees with overlap"
                )
            nonlinear = value["nonlinear_moment_diagnostic"]["exact_invariant_RMS_MeV"]
            relative = nonlinear / value["linear_sigma_MeV"] - 1
            rows.append(
                {
                    "curve_role": "baseline",
                    "machine": machine,
                    "dispersion_mm": dispersion_mm,
                    "full_crossing_angle_mrad": 1000 * angle,
                    "compensation_fraction": tilt,
                    "horizontal_beta_multiplier": 1.0,
                    "luminosity_cm2_s": value["luminosity_cm2_s"],
                    "linear_sigma_MeV": value["linear_sigma_MeV"],
                    "linear_gaussian_FWHM_MeV": value["linear_gaussian_FWHM_MeV"],
                    "angular_floor_FWHM_MeV": value["linear_angular_floor_FWHM_MeV"],
                    "nonlinear_RMS_diagnostic_MeV": nonlinear,
                    "nonlinear_relative_difference": relative,
                    "first_order_diagnostic_within_5percent": abs(relative) <= 0.05,
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_rows(args.output, generate())
