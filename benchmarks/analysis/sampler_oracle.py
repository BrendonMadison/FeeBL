#!/usr/bin/env python3
"""Compare native fixed-field radiation trials with exact-Bessel quadrature.

The transition law is CAIN 2.1e, Sec. 5.8, Eqs. (92), (99)--(104).
The independent oracle uses SciPy's modified Bessel function and quadrature,
not FeeBL's rational Bessel approximations. Exposure denotes the native
proposal probability p0, before quantum and spin-dependent rejection.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import kv

from comparison_statistics import INPUTS


@lru_cache(maxsize=16)
def bessel_moments(chi):
    """Integrate probability and energy kernels for zero/unit transverse spin."""
    scale = 1.5 * chi

    def integrand(z, component):
        fraction = scale * z / (1 + scale * z)
        jacobian = scale / (1 + scale * z) ** 2
        k13, k23 = kv(1 / 3, z), kv(2 / 3, z)
        integral13 = quad(
            lambda t: kv(1 / 3, t), z, np.inf, epsabs=2e-12, epsrel=2e-10
        )[0]
        unpolarized = 2 * k23 - integral13 + fraction ** 2 / (1 - fraction) * k23
        spin = fraction * k13
        return (
            jacobian
            * (unpolarized, spin, fraction * unpolarized, fraction * spin)[component]
        )

    return np.array(
        [
            quad(
                lambda z: integrand(z, component),
                0,
                100,
                epsabs=1e-13,
                epsrel=3e-8,
                points=[0.001, 0.01, 0.1, 1, 10],
            )[0]
            for component in range(4)
        ]
    )


def evaluate(directory):
    """Derive sampling SEMs and capped-unit-bar pulls from trial sufficient statistics."""
    rows = []
    seeds = set()
    for path in sorted(directory.glob("*.txt")):
        lines = [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        values = [float(value) for value in lines[-1].split()]
        if len(values) != 9:
            raise ValueError(f"Bad sampler record {path.name}")
        chi, exposure, spin, seed, count, emitted, sum_x, sum_x2, p0 = values
        if int(seed) in seeds:
            raise ValueError("Reused seed in sampler record")
        seeds.add(int(seed))
        probability, spin_probability, energy, spin_energy = bessel_moments(chi)
        factor = p0 * (1 + 0.75 * chi) ** (1 / 3) / (chi * 9.67287708690521)
        probability_ref = factor * (probability - spin * spin_probability)
        energy_ref = factor * (energy - spin * spin_energy)
        probability_mean, energy_mean = emitted / count, sum_x / count
        probability_sem = math.sqrt(probability_ref * (1 - probability_ref) / count)
        energy_sem = math.sqrt((sum_x2 - sum_x ** 2 / count) / (count * (count - 1)))
        rows.append(
            {
                "chi": chi,
                "exposure": exposure,
                "s2": spin,
                "seed": int(seed),
                "trials": int(count),
                "emitted": int(emitted),
                "sum_x": sum_x,
                "sum_x2": sum_x2,
                "p0": p0,
                "p_ref": probability_ref,
                "p_hat": probability_mean,
                "pull_p": (probability_mean - probability_ref) / probability_sem,
                "x_ref": energy_ref,
                "x_hat": energy_mean,
                "pull_x": (energy_mean - energy_ref) / energy_sem,
                "probability_sem": probability_sem,
                "energy_sem": energy_sem,
            }
        )
    return {
        "runs": rows,
        "summary": {
            "runs": len(rows),
            "independent_seeds": len(seeds),
            "probability_pull_rms": float(
                np.sqrt(np.mean([r["pull_p"] ** 2 for r in rows]))
            ),
            "energy_pull_rms": float(
                np.sqrt(np.mean([r["pull_x"] ** 2 for r in rows]))
            ),
            "sampling_unit": "one independent fixed-field trial",
            "reference_sampling_variance": 0,
            "joint_covariance": "Probability and energy within a run share trials; no global chi-square inferred.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=INPUTS / "sampler_records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.records)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
