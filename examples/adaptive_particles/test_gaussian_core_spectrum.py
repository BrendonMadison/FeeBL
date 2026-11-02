#!/usr/bin/env python3
"""Tests for the frozen FeeBL Gaussian-core spectrum extractor."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "adaptive_particles" / "gaussian_core_spectrum.py"
SPEC = importlib.util.spec_from_file_location("gaussian_core_spectrum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def toy_spectrum(
    *,
    mu: float = 91.18,
    sigma: float = 0.168,
    amplitude: float = 8.0e34,
    tail_strength: float = 0.2,
    tail_scale: float = 0.35,
    shape: str = "gaussian",
    bins: int = 4000,
    underflow: float = 0.0,
    overflow: float = 0.0,
    seed: int = 17,
) -> str:
    low_edge = 84.0
    high_edge = 92.0
    width = (high_edge - low_edge) / bins
    rows: list[str] = []
    luminosities: list[tuple[float, float]] = []
    for index in range(bins):
        low = low_edge + index * width
        high = low + width
        center = 0.5 * (low + high)
        x = (center - mu) / sigma
        if shape == "gaussian":
            core = amplitude * math.exp(-0.5 * x * x)
        elif shape == "super_gaussian":
            core = amplitude * math.exp(-0.5 * x**4)
        else:
            raise AssertionError(shape)
        tail = 0.0
        if center < mu and tail_strength:
            u = (mu - center) / tail_scale
            tail = amplitude * tail_strength * u * math.exp(-u)
        density = core + tail
        weight = density * width
        luminosities.append((center, weight))
        rows.append(
            f"{low:.12g} {high:.12g} {density:.17g} {weight:.17g} "
            "0 0 0 0 0 0 0"
        )
    total = math.fsum(weight for _, weight in luminosities)
    mean = math.fsum(center * weight for center, weight in luminosities) / total
    rms = math.sqrt(
        math.fsum(weight * (center - mean) ** 2 for center, weight in luminosities) / total
    )
    return "\n".join(
        [
            "#FEEBL-NATIVE 1.0",
            "#feebl_version 0.6.0-dev",
            "#accelerator ",
            "#parameters ",
            f"#rndm_seed {seed}",
            "#",
            "# units",
            "#length_unit nm",
            "#z_unit um",
            "#energy_unit GeV",
            "#angle_unit rad",
            "#",
            "#BLOCK pair_spectrum",
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff",
            *rows,
            f"#underflow {underflow:.17g}",
            f"#overflow {overflow:.17g}",
            f"#mean_W {mean:.17g} +- 1e-6",
            f"#rms_W {rms:.17g}",
            "#n_eff_total 1000000000",
            f"#END pair_spectrum records {bins}",
            "",
        ]
    )


def native_total_luminosity(text: str) -> float:
    lines = text.splitlines()
    start = lines.index(
        "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
    ) + 1
    end = next(index for index, line in enumerate(lines) if line.startswith("#underflow "))
    stored = math.fsum(float(line.split()[3]) for line in lines[start:end])
    underflow = float(lines[end].split()[1])
    overflow = float(lines[end + 1].split()[1])
    return stored + underflow + overflow


class GaussianCoreSpectrumTest(unittest.TestCase):
    def write(self, directory: Path, text: str, name: str = "lumi.ee.out.dLdW.fne") -> Path:
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def expectation(
        self,
        *,
        bins: int = 4000,
        seed: int = 17,
        total_luminosity: float,
        run_id: str = "unit-test-run",
    ) -> MODULE.SpectrumExpectation:
        return MODULE.SpectrumExpectation(
            seed, 84.0, 92.0, bins, total_luminosity, run_id
        )

    def read_bound(
        self, path: Path, *, bins: int = 4000, seed: int = 17
    ) -> MODULE.Spectrum:
        try:
            total = native_total_luminosity(path.read_text(encoding="utf-8"))
        except (IndexError, ValueError):
            # Malformed-format tests must reach the production parser; the
            # placeholder cannot make malformed input acceptable.
            total = 1.0
        return MODULE.read_spectrum(
            path,
            self.expectation(bins=bins, seed=seed, total_luminosity=total),
        )

    def test_gaussian_plus_one_sided_tail_recovers_core(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), toy_spectrum())
            spectrum = self.read_bound(path)
            result = MODULE.fit_spectrum(spectrum)
        fit = result["core_fit"]
        self.assertLess(abs(float(fit["mu_core_GeV"]) - 91.18), 2.0e-4)
        self.assertLess(abs(float(fit["sigma_core_GeV"]) / 0.168 - 1.0), 1.0e-3)
        self.assertGreater(float(result["spectrum"]["global_rms_W_GeV_diagnostic_only"]), 0.25)
        self.assertEqual(result["status"], "accepted_core_fit_not_physics_qualification")

    def test_deterministic_json_and_csv_cli(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_input = self.write(first_root, toy_spectrum())
            second_input = self.write(second_root, toy_spectrum())
            first_json, first_csv = first_root / "core.json", first_root / "core.csv"
            second_json, second_csv = second_root / "core.json", second_root / "core.csv"
            for source, json_path, csv_path in (
                (first_input, first_json, first_csv),
                (second_input, second_json, second_csv),
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(SCRIPT),
                        str(source),
                        "--json",
                        str(json_path),
                        "--csv",
                        str(csv_path),
                        "--expected-seed",
                        "17",
                        "--expected-w-min",
                        "84",
                        "--expected-w-max",
                        "92",
                        "--expected-bins",
                        "4000",
                        "--expected-total-luminosity",
                        format(native_total_luminosity(source.read_text(encoding="utf-8")), ".17g"),
                        "--expected-run-id",
                        "unit-test-run",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
            parsed = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema"], MODULE.SCHEMA)
            with first_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema"], MODULE.SCHEMA)

            repeated = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    str(first_input),
                    "--json",
                    str(first_json),
                    "--csv",
                    str(first_csv),
                    "--expected-seed",
                    "17",
                    "--expected-w-min",
                    "84",
                    "--expected-w-max",
                    "92",
                    "--expected-bins",
                    "4000",
                    "--expected-total-luminosity",
                    format(native_total_luminosity(first_input.read_text(encoding="utf-8")), ".17g"),
                    "--expected-run-id",
                    "unit-test-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing to overwrite", repeated.stderr)

    def test_malformed_gap_is_rejected(self) -> None:
        text = toy_spectrum(bins=400)
        lines = text.splitlines()
        first_record = lines.index(
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
        ) + 1
        fields = lines[first_record + 7].split()
        fields[0] = format(float(fields[0]) + 0.001, ".12g")
        fields[1] = format(float(fields[1]) + 0.001, ".12g")
        lines[first_record + 7] = " ".join(fields)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), "\n".join(lines) + "\n")
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "not aligned"):
                self.read_bound(path, bins=400)

    def test_aligned_omitted_empty_record_is_reconstructed(self) -> None:
        lines = toy_spectrum(bins=1600).splitlines()
        # Remove one far-low-side record exactly on the histogram lattice,
        # matching FeeBL's sparse omission of an empty bin.
        first_record = lines.index(
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
        ) + 1
        del lines[first_record + 5]
        lines[-1] = "#END pair_spectrum records 1599"
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), "\n".join(lines) + "\n")
            spectrum = self.read_bound(path, bins=1600)
            self.assertEqual(len(spectrum.bins), 1600)
            self.assertEqual(spectrum.bins[5].density, 0.0)
            result = MODULE.fit_spectrum(spectrum)
        self.assertEqual(result["schema"], MODULE.SCHEMA)

    def test_excessive_underflow_is_rejected(self) -> None:
        text = toy_spectrum(underflow=1.0e35)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "0.1%"):
                self.read_bound(path)

    def test_native_weight_identity_is_format_tolerant_but_fail_closed(self) -> None:
        base = toy_spectrum(bins=400).splitlines()
        columns = (
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
        )
        first = base.index(columns) + 1
        for relative_shift, accepted in ((1.0e-5, True), (3.0e-5, False)):
            with self.subTest(relative_shift=relative_shift):
                lines = list(base)
                fields = lines[first + 200].split()
                fields[3] = format(float(fields[3]) * (1.0 + relative_shift), ".17g")
                lines[first + 200] = " ".join(fields)
                text = "\n".join(lines) + "\n"
                with tempfile.TemporaryDirectory() as temporary:
                    path = self.write(Path(temporary), text)
                    if accepted:
                        self.read_bound(path, bins=400)
                    else:
                        with self.assertRaisesRegex(
                            MODULE.SpectrumFitError, "weight disagrees"
                        ):
                            self.read_bound(path, bins=400)

    def test_total_luminosity_identity_uses_frozen_format_tolerance(self) -> None:
        text = toy_spectrum(bins=400)
        actual = native_total_luminosity(text)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            accepted = MODULE.SpectrumExpectation(
                17, 84.0, 92.0, 400, actual * (1.0 + 1.0e-5), "accepted"
            )
            MODULE.read_spectrum(path, accepted)
            rejected = MODULE.SpectrumExpectation(
                17, 84.0, 92.0, 400, actual * (1.0 + 3.0e-5), "rejected"
            )
            with self.assertRaisesRegex(
                MODULE.SpectrumFitError, "disagrees with executed out.dat"
            ):
                MODULE.read_spectrum(path, rejected)

    def test_missing_footer_is_rejected(self) -> None:
        text = toy_spectrum().replace("#overflow 0\n", "")
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "#overflow footer"):
                self.read_bound(path)

    def test_strict_native_contract_rejects_adversarial_mutations(self) -> None:
        base = toy_spectrum(bins=400).splitlines()
        columns = (
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
        )
        first_record = base.index(columns) + 1

        def remove_line(lines: list[str], exact: str) -> None:
            lines.remove(exact)

        cases: dict[str, list[str]] = {}
        mutated = list(base)
        remove_line(mutated, "#FEEBL-NATIVE 1.0")
        cases["missing native header"] = mutated
        mutated = list(base)
        mutated[0] = "#FEEBL-NATIVE 9.9"
        cases["unsupported native version"] = mutated
        mutated = list(base)
        remove_line(mutated, "#energy_unit GeV")
        cases["missing energy unit"] = mutated
        mutated = list(base)
        mutated[mutated.index("#energy_unit GeV")] = "#energy_unit MeV"
        cases["wrong energy unit"] = mutated
        mutated = list(base)
        remove_line(mutated, "#rndm_seed 17")
        cases["missing seed"] = mutated
        mutated = list(base)
        mutated[-1] = "#END pair_spectrum records 999999"
        cases["false footer count"] = mutated
        mutated = list(base)
        del mutated[first_record + 5]
        cases["deleted row with stale count"] = mutated
        mutated = list(base)
        mutated[-1] = "#END pair_spectrum arbitrary"
        cases["malformed end grammar"] = mutated
        mutated = list(base)
        remove_line(mutated, columns)
        mutated.insert(mutated.index("#BLOCK pair_spectrum") - 1, columns)
        cases["columns outside block"] = mutated

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for label, lines in cases.items():
                with self.subTest(label=label):
                    path = self.write(directory, "\n".join(lines) + "\n", f"{label}.fne")
                    with self.assertRaises(MODULE.SpectrumFitError):
                        self.read_bound(path, bins=400)

    def test_expected_seed_is_an_authenticated_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), toy_spectrum(bins=400, seed=29))
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "does not match expected"):
                self.read_bound(path, bins=400, seed=17)

    def test_malformed_footer_uncertainty_is_rejected(self) -> None:
        text = re.sub(r"(?m)^#mean_W .+$", "#mean_W 91 +-", toy_spectrum(bins=400))
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "#mean_W footer"):
                self.read_bound(path, bins=400)

    def test_native_mean_footer_requires_unsigned_uncertainty(self) -> None:
        native = toy_spectrum(bins=400)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            malformed = {
                "missing": "#mean_W 91",
                "negative": "#mean_W 91 +- -1e-6",
            }
            for label, footer in malformed.items():
                with self.subTest(label=label):
                    text = re.sub(r"(?m)^#mean_W .+$", footer, native)
                    path = self.write(directory, text, f"mean-{label}.fne")
                    with self.assertRaisesRegex(MODULE.SpectrumFitError, "#mean_W footer"):
                        self.read_bound(path, bins=400)

    def test_native_scalar_footers_forbid_uncertainty_suffix(self) -> None:
        native = toy_spectrum(bins=400)
        for key in ("underflow", "overflow", "rms_W", "n_eff_total"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                text, replacements = re.subn(
                    rf"(?m)^(#{re.escape(key)}[ \t]+[^\r\n]+)$",
                    r"\1 +- 1e-6",
                    native,
                )
                self.assertEqual(replacements, 1)
                path = self.write(Path(temporary), text)
                with self.assertRaisesRegex(MODULE.SpectrumFitError, rf"#{key} footer"):
                    self.read_bound(path, bins=400)

    def test_full_lattice_reconstruction_makes_pair_rebin_phase_invariant(self) -> None:
        lines = toy_spectrum(bins=1600, tail_strength=0.15).splitlines()
        columns = (
            "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff"
        )
        first_record = lines.index(columns) + 1
        last_record = next(i for i, line in enumerate(lines) if line.startswith("#underflow"))
        outer_indices = list(range(first_record, first_record + 5)) + list(
            range(last_record - 4, last_record)
        )
        full = list(lines)
        for index in outer_indices:
            fields = full[index].split()
            fields[2] = "0"
            fields[3] = "0"
            full[index] = " ".join(fields)
        sparse = [line for index, line in enumerate(full) if index not in set(outer_indices)]
        sparse[-1] = "#END pair_spectrum records 1591"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            full_path = self.write(directory, "\n".join(full) + "\n", "full.fne")
            sparse_path = self.write(directory, "\n".join(sparse) + "\n", "sparse.fne")
            full_spectrum = self.read_bound(full_path, bins=1600)
            sparse_spectrum = self.read_bound(sparse_path, bins=1600)
            self.assertEqual(full_spectrum.bins, sparse_spectrum.bins)
            full_fit = MODULE.fit_spectrum(full_spectrum)
            sparse_fit = MODULE.fit_spectrum(sparse_spectrum)
            self.assertEqual(full_fit["core_fit"], sparse_fit["core_fit"])
            self.assertEqual(full_fit["stability"], sparse_fit["stability"])
            unbound = MODULE.read_spectrum(sparse_path, allow_unbound_diagnostic=True)
            self.assertFalse(unbound.lattice_phase_proven)
            unbound_fit = MODULE.fit_spectrum(unbound)
            self.assertEqual(
                unbound_fit["stability"]["pair_rebin_status"],
                "not_performed_unbound_lattice_phase",
            )

    def test_cli_requires_complete_binding_or_explicit_unbound_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = self.write(directory, toy_spectrum())
            command = [
                sys.executable,
                "-B",
                str(SCRIPT),
                str(source),
                "--json",
                str(directory / "core.json"),
                "--csv",
                str(directory / "core.csv"),
            ]
            refused = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("qualification-bound expectations are required", refused.stderr)
            accepted = subprocess.run(
                command + ["--allow-unbound-diagnostic"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            result = json.loads((directory / "core.json").read_text(encoding="utf-8"))
            self.assertEqual(result["qualification_binding"]["status"], "unbound_diagnostic_only")
            self.assertEqual(
                result["stability"]["pair_rebin_status"],
                "not_performed_unbound_lattice_phase",
            )

    def test_final_publish_fit_must_remain_a_fixed_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), toy_spectrum())
            spectrum = self.read_bound(path)
        baseline = MODULE._fit_once(spectrum, MODULE.MAIN_UPPER_SIGMA)
        publish_call = baseline.iteration_count + 1
        original = MODULE._mapped_core_parameters
        calls = 0

        def moved_publish(*args: object, **kwargs: object) -> tuple[object, ...]:
            nonlocal calls
            calls += 1
            result = list(original(*args, **kwargs))
            if calls == publish_call:
                result[5] = float(result[5]) + 1.0e-3
            return tuple(result)

        with mock.patch.object(MODULE, "_mapped_core_parameters", side_effect=moved_publish):
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "publish fit.*fixed point"):
                MODULE._fit_once(spectrum, MODULE.MAIN_UPPER_SIGMA)

    def test_scale_aware_alternate_window_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), toy_spectrum())
            spectrum = self.read_bound(path)
        baseline = MODULE._fit_once(spectrum, MODULE.MAIN_UPPER_SIGMA)

        with mock.patch.object(
            MODULE,
            "_fixed_window_parameters",
            side_effect=lambda candidate, *_: (
                (baseline.mu_gev, baseline.sigma_core_gev)
                if candidate.bin_width > 1.5 * spectrum.bin_width
                else (
                    baseline.mu_gev + 0.04 * baseline.sigma_core_gev,
                    1.02 * baseline.sigma_core_gev,
                )
            ),
        ):
            accepted = MODULE.fit_spectrum(spectrum)
        self.assertEqual(accepted["method"]["id"], "upper-flank-log-quadratic-v5")
        self.assertAlmostEqual(
            float(accepted["stability"]["maximum_window_mu_shift_sigma_fraction"]),
            0.04,
        )
        self.assertEqual(
            accepted["stability"]["window_mu_limit_sigma_fraction"], "0.05"
        )
        self.assertEqual(accepted["stability"]["window_sigma_relative_limit"], "0.03")

        with mock.patch.object(
            MODULE,
            "_fixed_window_parameters",
            side_effect=lambda candidate, *_: (
                (baseline.mu_gev, baseline.sigma_core_gev)
                if candidate.bin_width > 1.5 * spectrum.bin_width
                else (
                    baseline.mu_gev + 0.051 * baseline.sigma_core_gev,
                    baseline.sigma_core_gev,
                )
            ),
        ):
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "exceeds 0.05 core sigma"):
                MODULE.fit_spectrum(spectrum)

        with mock.patch.object(
            MODULE,
            "_fixed_window_parameters",
            side_effect=lambda candidate, *_: (
                (baseline.mu_gev, baseline.sigma_core_gev)
                if candidate.bin_width > 1.5 * spectrum.bin_width
                else (
                    baseline.mu_gev,
                    1.031 * baseline.sigma_core_gev,
                )
            ),
        ):
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "exceeds 0.03"):
                MODULE.fit_spectrum(spectrum)

    def test_pair_rebin_uses_fixed_primary_physical_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), toy_spectrum())
            spectrum = self.read_bound(path)
        original_fit_once = MODULE._fit_once

        def reject_independent_rebinned_iteration(
            candidate: MODULE.Spectrum, upper_sigma: float
        ) -> MODULE.RawFit:
            if candidate.bin_width > 1.5 * spectrum.bin_width:
                raise AssertionError("rebinned diagnostic must not iterate its own window")
            return original_fit_once(candidate, upper_sigma)

        with mock.patch.object(
            MODULE,
            "_fit_once",
            side_effect=reject_independent_rebinned_iteration,
        ):
            result = MODULE.fit_spectrum(spectrum)
        self.assertEqual(
            result["method"]["pair_rebin_window_policy"],
            "fixed_primary_physical_window",
        )
        self.assertEqual(
            result["stability"]["pair_rebin_status"],
            "accepted_bound_lattice_phase",
        )

    def test_non_gaussian_flat_top_is_rejected(self) -> None:
        text = toy_spectrum(shape="super_gaussian", tail_strength=0.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            spectrum = self.read_bound(path)
            with self.assertRaisesRegex(
                MODULE.SpectrumFitError,
                "poor Gaussian-core fit|inconsistent with the observed mode",
            ):
                MODULE.fit_spectrum(spectrum)

    def test_upper_boundary_too_close_is_rejected(self) -> None:
        # Moving the peak close to 92 GeV leaves less than three core sigmas on
        # the protected high side.
        text = toy_spectrum(mu=91.65, sigma=0.14, tail_strength=0.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), text)
            spectrum = self.read_bound(path)
            with self.assertRaisesRegex(MODULE.SpectrumFitError, "upper boundary"):
                MODULE.fit_spectrum(spectrum)


if __name__ == "__main__":
    unittest.main()
