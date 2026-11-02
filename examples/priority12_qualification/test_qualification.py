#!/usr/bin/env python3
"""Focused tests for the P1/P2 qualification harness itself."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import lumi_spectrum_equivalence as spectrum
import run_qualification as runner


def write_spectrum(path: Path, weights: list[float], scale: float = 1.0) -> None:
    rows = ["#FEEBL-NATIVE 1.0", "#COLUMNS W_low W_high dL_dW weight"]
    for index, weight in enumerate(weights):
        value = weight * scale
        rows.append(f"{index} {index + 1} {value} {value}")
    rows.extend(("#underflow 0", "#overflow 0", f"#END pair_spectrum records {len(weights)}"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class SpectrumEquivalenceTest(unittest.TestCase):
    def test_identical_paired_spectra_pass_every_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references = []
            candidates = []
            for seed in range(8):
                left = root / f"left-{seed}.fne"
                right = root / f"right-{seed}.fne"
                write_spectrum(left, [10.0 + seed, 80.0, 10.0])
                write_spectrum(right, [10.0 + seed, 80.0, 10.0])
                references.append(spectrum.read_spectrum(left))
                candidates.append(spectrum.read_spectrum(right))
            result = spectrum.compare_ensemble(references, candidates)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["pair_count"], 8)
            self.assertTrue(all(result["aggregate_pass"].values()))

    def test_normalization_and_peak_shape_fail_independent_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.fne"
            scaled = root / "scaled.fne"
            shifted = root / "shifted.fne"
            write_spectrum(base, [20.0, 60.0, 20.0])
            write_spectrum(scaled, [20.0, 60.0, 20.0], 1.02)
            write_spectrum(shifted, [10.0, 60.0, 30.0])
            normalization = spectrum.compare_ensemble(
                [spectrum.read_spectrum(base)], [spectrum.read_spectrum(scaled)]
            )
            shape = spectrum.compare_ensemble(
                [spectrum.read_spectrum(base)], [spectrum.read_spectrum(shifted)]
            )
            self.assertFalse(normalization["aggregate_pass"]["total_relative_effect"])
            self.assertAlmostEqual(
                normalization["aggregate_metrics"]["coarse_total_variation"], 0.0
            )
            self.assertFalse(shape["aggregate_pass"]["mean_shift_reference_sigma"])
            self.assertGreater(shape["aggregate_metrics"]["ks_distance"], 0.005)

    def test_parser_rejects_negative_weight_and_changed_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.fne"
            bad.write_text("0 1 -1 -1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "negative weight"):
                spectrum.read_spectrum(bad)
            left = root / "left.fne"
            right = root / "right.fne"
            write_spectrum(left, [1.0, 2.0])
            right.write_text("0 1 1 1\n1 2.1 2 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bin-edge mismatch"):
                spectrum.compare_ensemble(
                    [spectrum.read_spectrum(left)], [spectrum.read_spectrum(right)]
                )


class QualificationRunnerTest(unittest.TestCase):
    def test_parameter_injection_is_scoped_and_rng_is_owned_by_harness(self) -> None:
        source = (
            "$PARAMETERS:: other\n{x=1;}\n"
            "$PARAMETERS:: selected\n{rndm_seed=99;x=2;}\n"
        )
        result = runner.inject_parameters(source, "selected", 7, "candidate=1;")
        other, selected = result.split("$PARAMETERS:: selected", 1)
        self.assertNotIn("candidate", other)
        self.assertIn("candidate=1;rndm_seed=7;rndm_load=0;rndm_save=0;", selected)
        with self.assertRaisesRegex(ValueError, "RNG controls"):
            runner.inject_parameters(source, "selected", 7, "rndm_seed=3;")

    def test_manifest_requires_receipt_and_fail_closed_case(self) -> None:
        manifest = {
            "schema": "feebl.priority12.qualification.v1",
            "accelerator": "a", "parameters": "p",
            "baseline": {"card": "a.dat"},
            "candidate": {"card": "b.dat"},
        }
        with self.assertRaisesRegex(ValueError, "receipt_regex"):
            runner.validate_manifest(manifest, False)
        manifest["candidate"]["receipt_regex"] = ["actual=candidate"]
        manifest["candidate"]["success_only_regex"] = ["committed=1"]
        with self.assertRaisesRegex(ValueError, "fail-closed"):
            runner.validate_manifest(manifest, False)

    def test_numeric_comparison_uses_absolute_tolerance_near_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.write_text("x = 0\n", encoding="utf-8")
            right.write_text("x = 5e-13\n", encoding="utf-8")
            result = runner.compare_numeric_text(left, right, 1.0e-8, 1.0e-12)
            self.assertTrue(result["pass"])

    def test_card_gate_ignores_only_declared_selectors(self) -> None:
        baseline = (
            "$ACCELERATOR:: a\n{energy=45.6;}\n"
            "$PARAMETERS:: p\n{n_x=32;gpu_mode=1;rndm_seed=1;}\n"
        )
        candidate = (
            "$ACCELERATOR:: a\n{energy=45.6;}\n"
            "$PARAMETERS:: p\n{n_x=64;gpu_mode=5;rndm_seed=99;}\n"
        )
        failed = runner.compare_card_physics(
            baseline, candidate, "a", "p", ["gpu_mode"]
        )
        self.assertFalse(failed["equivalent"])
        self.assertEqual(failed["unexpected_differences"], ["parameters.n_x"])
        candidate = candidate.replace("n_x=64", "n_x=32")
        passed = runner.compare_card_physics(
            baseline, candidate, "a", "p", ["gpu_mode"]
        )
        self.assertTrue(passed["equivalent"])

    def test_end_to_end_fake_proves_rejection_hashes_spectra_and_timings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake-feebl"
            fake.write_text(textwrap.dedent(r'''
                #!/usr/bin/env python3
                import pathlib, re, sys
                card = pathlib.Path("acc.dat").read_text()
                if re.findall(r"invalid_mode\s*=\s*(\d+)", card)[-1:] == ["1"]:
                    print("preflight: unsupported selector", file=sys.stderr)
                    raise SystemExit(7)
                candidate = re.findall(r"candidate\s*=\s*(\d+)", card)[-1:] == ["1"]
                seed = int(re.findall(r"rndm_seed\s*=\s*(\d+)", card)[-1])
                if candidate:
                    print("actual=candidate wave_concurrency=same_wave field_ms=0.5 commit_ms=1.0", file=sys.stderr)
                    selector = "candidate"
                else:
                    print("actual=baseline field_ms=1.0 commit_ms=2.0", file=sys.stderr)
                    selector = "baseline"
                pathlib.Path("result.out").write_text(
                    f"selector = {selector}\nphysics = {1.0 + seed * 0.0}\n")
                pathlib.Path("lumi.ee.out").write_text("1 1 0 0 0 0 0\n")
                pathlib.Path("lumi.ee.out.fne").write_text("#COLUMNS e1 e2\n1 1\n")
                scale = 1.0 + seed * 1e-4
                rows = ["#COLUMNS W_low W_high dL_dW weight"]
                for i, w in enumerate((10.0, 80.0, 10.0)):
                    rows.append(f"{i} {i+1} {w*scale} {w*scale}")
                rows.extend(("#underflow 0", "#overflow 0"))
                pathlib.Path("lumi.ee.out.dLdW.fne").write_text("\n".join(rows) + "\n")
            ''').lstrip(), encoding="utf-8")
            fake.chmod(0o755)
            baseline = root / "baseline.dat"
            candidate = root / "candidate.dat"
            baseline.write_text(
                "$ACCELERATOR:: a\n{energy=1;}\n$PARAMETERS:: p\n{candidate=0;}\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "$ACCELERATOR:: a\n{energy=1;}\n$PARAMETERS:: p\n{candidate=1;}\n",
                encoding="utf-8",
            )
            manifest = {
                "schema": "feebl.priority12.qualification.v1",
                "accelerator": "a", "parameters": "p",
                "selector_assignment_keys": ["candidate"],
                "baseline": {"card": "baseline.dat", "receipt_regex": ["actual=baseline"]},
                "candidate": {
                    "card": "candidate.dat",
                    "receipt_regex": ["actual=candidate", "wave_concurrency=same_wave"],
                    "success_only_regex": ["wave_concurrency=same_wave"],
                },
                "rejections": [{
                    "name": "unsupported", "base": "candidate",
                    "fragment": "invalid_mode=1;",
                    "stderr_regex": "preflight: unsupported selector",
                }],
                "result_normalize_regex": ["^selector"],
                "deterministic": {
                    "seed": 2, "timing_repeats": 7, "warmups_per_variant": 1,
                    "cross_mode_exact_artifacts": [
                        "result.normalized", "lumi.ee.out", "lumi.ee.out.fne",
                        "lumi.ee.out.dLdW.fne",
                    ],
                },
                "stage_regex": {
                    "baseline": {
                        "field_ms": r"field_ms=([0-9.]+)",
                        "commit_ms": r"commit_ms=([0-9.]+)",
                    },
                    "candidate": {
                        "field_ms": r"field_ms=([0-9.]+)",
                        "commit_ms": r"commit_ms=([0-9.]+)",
                    },
                },
                "nonoverlap_stage_names": {
                    "baseline": ["field_ms", "commit_ms"],
                    "candidate": ["field_ms", "commit_ms"],
                },
                "statistical": {"first_seed": 10, "seed_count": 8},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "evidence"
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("run_qualification.py")),
                 "--binary", str(fake), "--manifest", str(manifest_path),
                 "--output", str(output), "--timeout-seconds", "10"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "qualification-result.json").read_text())
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["deterministic_and_timing"]
                            ["within_selector_byte_identity"])
            self.assertEqual(result["statistical_luminosity_spectrum"]
                             ["pair_count"], 8)
            self.assertEqual(result["fail_closed_rejections"][0]
                             ["produced_science_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
