"""Apply fixed simultaneous rules to the original 240 sampler checks only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import norm


HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "example_output/analysis_inputs/historical_sampler"
ALPHA = 0.05
FAMILY_SIZE = 240


def digest(path):
    """Identify a retained input or producer by its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Reduce all original checks under a fixed simultaneous sampling rule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.inputs, args.output
    checks = []
    samples = {}
    input_names = [
        "independent_physics_checks.json",
        "independent_physics_checks_chi3.json",
        "mc_after.txt",
        "mc_chi3.txt",
        "spin_physics_probe.cc",
        "compare_spin_physics.py",
        "probe_receipt_after.json",
        "chi3_validation_receipt.json",
    ]
    for name in input_names[:2]:
        checks.extend(json.loads((source / name).read_text())["sampler_checks"])
    for name in input_names[2:4]:
        for line in (source / name).read_text().splitlines():
            if not line.startswith("mc "):
                continue
            values = list(map(float, line.split()[1:]))
            key = (values[0], values[1], int(values[2]))
            if key in samples:
                raise ValueError("Duplicate historical setting")
            samples[key] = values
    assert len(checks) == FAMILY_SIZE and len(samples) == 40
    assert (
        len({(c["chi"], c["p0_unsuppressed"], c["initial"], c["test"]) for c in checks})
        == 240
    )
    assert {v[3] for v in samples.values()} == {2000000}

    # Two one-sided applications per marginal, followed by a union bound.
    logarithm = math.log(4 * FAMILY_SIZE / ALPHA)
    gaussian_threshold = float(norm.isf(ALPHA / (2 * FAMILY_SIZE)))
    rows = []
    for check in checks:
        key = (check["chi"], check["p0_unsuppressed"], check["initial"])
        sample = samples[key]
        n = int(sample[3])
        row = {
            "chi": key[0],
            "exposure": key[1],
            "initial": key[2],
            "test": check["test"],
            "trials": n,
        }
        if "invalid" in check:
            row.update(
                kind="deterministic_bounds",
                invalid=int(check["invalid"]),
                max_spin_norm_squared=check["result"],
                accepted=check["invalid"] == 0,
            )
            rows.append(row)
            continue

        spin = check["test"].startswith("ST ensemble")
        probability = check["test"] == "ST photon probability"
        mean = float(check["result"])
        target = float(check["reference"])
        if probability:
            assert abs(mean - sample[5] / n) < 1e-15
            variance = n / (n - 1) * mean * (1 - mean)
        else:
            # Probe SEM = sqrt((mean(X^2)-mean(X)^2)/n).
            component = {
                "ST ensemble Sx": 6,
                "ST ensemble Sy": 7,
                "ST ensemble Sz": 8,
            }.get(check["test"], 12)
            sem_component = component + 3 if spin else 13
            if (
                mean != sample[component]
                or float(check["sem"]) != sample[sem_component]
            ):
                raise ValueError("Oracle record differs from retained native sample")
            variance = n * n / (n - 1) * float(check["sem"]) ** 2
        width = 2.0 if spin else 1.0
        # Standard bounded double-precision accumulation model: preserve a
        # conservative variance allowance for the retained rounded summaries.
        unit_roundoff = 2.0 ** -53
        gamma_n = n * unit_roundoff / (1 - n * unit_roundoff)
        variance_roundoff_allowance = 8 * gamma_n * n / (n - 1)
        upper_variance = variance + variance_roundoff_allowance
        sampling_radius = math.sqrt(
            2 * upper_variance * logarithm / n
        ) + 7 * width * logarithm / (3 * (n - 1))
        finite_step_bound = float(check.get("finite_step_bound", 0))
        numerical_allowance = 2e-8 if spin else 2e-7 * max(target, 1e-5)
        error = abs(mean - target)
        excess = max(0.0, error - finite_step_bound - numerical_allowance)
        historical_sem = float(check["sem"])
        row.update(
            kind="sampling_mean",
            result=mean,
            reference=target,
            absolute_error=error,
            historical_sem=historical_sem,
            unbiased_sample_variance=variance,
            variance_roundoff_allowance=variance_roundoff_allowance,
            range_width=width,
            finite_step_bound=finite_step_bound,
            numerical_allowance=numerical_allowance,
            simultaneous_sampling_radius=sampling_radius,
            bound_adjusted_error_to_radius=excess / sampling_radius,
            raw_error_to_sampling_radius=error / sampling_radius,
            historical_raw_z=error / historical_sem if historical_sem else None,
            historical_bound_adjusted_z=excess / historical_sem
            if historical_sem
            else None,
            normal_bonferroni_diagnostic_pass=(
                excess <= gaussian_threshold * historical_sem
            ),
            accepted=excess <= sampling_radius,
        )
        rows.append(row)

    means = [r for r in rows if r["kind"] == "sampling_mean"]
    bounds = [r for r in rows if r["kind"] == "deterministic_bounds"]
    worst = max(means, key=lambda r: r["bound_adjusted_error_to_radius"])
    summary = {
        "schema": "feebl.historical-sampler-simultaneous/1",
        "family_size": FAMILY_SIZE,
        "familywise_alpha": ALPHA,
        "sampling_comparisons": len(means),
        "deterministic_checks": len(bounds),
        "settings": len(samples),
        "trials_per_setting": 2000000,
        "mean_checks_accepted": sum(r["accepted"] for r in means),
        "bounds_checks_accepted": sum(r["accepted"] for r in bounds),
        "zero_sem_means": sum(r["historical_sem"] == 0 for r in means),
        "max_adjusted_error_to_radius": worst["bound_adjusted_error_to_radius"],
        "max_raw_error_to_radius": max(
            r["raw_error_to_sampling_radius"] for r in means
        ),
        "worst_adjusted_comparison": worst,
        "normal_bonferroni_threshold_diagnostic_only": gaussian_threshold,
        "normal_diagnostic_mean_passes": sum(
            r["normal_bonferroni_diagnostic_pass"] for r in means
        ),
        "max_finite_raw_z": max(
            r["historical_raw_z"] for r in means if r["historical_raw_z"] is not None
        ),
        "max_finite_adjusted_z": max(
            r["historical_bound_adjusted_z"]
            for r in means
            if r["historical_bound_adjusted_z"] is not None
        ),
        "sampling_radius_min_max": [
            min(r["simultaneous_sampling_radius"] for r in means),
            max(r["simultaneous_sampling_radius"] for r in means),
        ],
        "rule": "abs(mean-reference) <= sqrt(2*(s2+variance_roundoff)*log(4*240/0.05)/n) + 7*range*log(4*240/0.05)/(3*(n-1)) + finite_step_bound + historical_numerical_allowance",
        "primary_reference": "Maurer and Pontil (2009), Theorem 4, https://arxiv.org/abs/0907.3740",
        "scope": [
            "Retrospective fixed-rule assessment of the full original 240-check family; no threshold fitted to residuals.",
            "There are 200 sampling means and 40 deterministic bounds checks, not 240 independent Gaussian pulls.",
            "Finite-sample sampling statement assumes independent identically distributed bounded trials within each setting, the intended PRNG model; shared seeds across settings are permitted by the union bound.",
            "The spin range is [-1,1], photon indicator and fractional photon energy [0,1], as intended by the normalized-state sampler; observed bounds are checked separately.",
            "Original finite-step model bound and numerical allowances remain deterministic, not Gaussian variances.",
            "Retained SEMs use a biased population variance; the reconstruction corrects to n-1 and adds a bounded double-accumulation roundoff allowance.",
            "The original 120-stream independent illustration is not read or used.",
            "No Gaussian marginal assumption is needed for the primary empirical-Bernstein rule; the normal calculation is diagnostic only.",
            "Historical source only; no new native execution, no collision convergence or general kernel accuracy claim.",
        ],
        "inputs": {name: digest(source / name) for name in input_names},
        "producer_sha256": digest(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=False)
    fields = sorted({field for row in rows for field in row})
    with (output / "all_240_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    tex = (
        "The historical single-step study contains "
        f"{len(means)} mean comparisons and {len(bounds)} norm/energy checks "
        "over forty settings of two million trials. A retrospective "
        r"5\% simultaneous empirical--Bernstein criterion accepts all "
        "mean comparisons; the maximum absolute discrepancy divided by "
        "its simultaneous sampling radius is "
        f"{summary['max_raw_error_to_radius']:.6f}. "
        "The existing deterministic finite-step and numerical allowances "
        "remain separate. All bounds checks report zero invalid trials. "
        "The criterion assumes independent bounded trials within a setting, "
        "but requires no independence between settings. It does not "
        "establish collision-level numerical accuracy.\n"
    )
    (output / "summary.tex").write_text(tex)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("inputs", "scope", "worst_adjusted_comparison")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
