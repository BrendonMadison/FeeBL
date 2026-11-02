#!/usr/bin/env python3
"""Deterministic tests for the standalone adaptive-particle comparator."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import compare_algorithms as comparator


class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = comparator.Grid(40, 32, -10.0, 10.0, -7.5, 7.5)
        cls.merge_grid = comparator.Grid(8, 8, -10.0, 10.0, -7.5, 7.5)
        cls.roots = comparator.generate_scenario("correlated_core_halo", 96, 2026090801)

    def assertRelativeLess(self, first: float, second: float, tolerance: float) -> None:
        scale = max(abs(second), 1.0)
        self.assertLessEqual(abs(first - second) / scale, tolerance)

    def test_synthetic_input_and_seed_are_reproducible(self) -> None:
        repeated = comparator.generate_scenario("correlated_core_halo", 96, 2026090801)
        changed = comparator.generate_scenario("correlated_core_halo", 96, 2026090802)
        self.assertEqual(comparator._input_digest(self.roots), comparator._input_digest(repeated))
        self.assertNotEqual(comparator._input_digest(self.roots), comparator._input_digest(changed))

    def test_A_exact_CIC_assignment_and_root_support(self) -> None:
        proposal = comparator.exact_assignment_split(self.roots, self.grid)
        metrics = comparator.compare_representation(self.roots, proposal, self.grid)
        self.assertLessEqual(metrics["charge_relative_error"], 2.0e-14)
        self.assertLessEqual(metrics["grid_projection"]["relative_l1"], 2.0e-14)
        self.assertLessEqual(metrics["grid_projection"]["relative_linf"], 2.0e-14)
        self.assertLessEqual(metrics["momentum_relative_error"], 2.0e-14)
        self.assertLessEqual(metrics["energy_relative_error"], 2.0e-14)
        self.assertTrue(metrics["ancestry"]["exact_ancestry"])
        self.assertFalse(metrics["ancestry"]["manufactured_support"])
        self.assertEqual(metrics["output_count"], 4 * len(self.roots))

    def test_B_nonnegative_moment_closure(self) -> None:
        proposal = comparator.constrained_moment_resample(self.roots, 16)
        metrics = comparator.compare_representation(self.roots, proposal, self.grid)
        self.assertEqual(len(proposal.particles), 16)
        self.assertTrue(all(particle.weight > 0.0 for particle in proposal.particles))
        self.assertLessEqual(metrics["charge_relative_error"], 3.0e-10)
        self.assertLessEqual(metrics["centroid_normalized_error"], 3.0e-9)
        self.assertLessEqual(metrics["covariance_relative_frobenius_error"], 3.0e-9)
        self.assertLessEqual(metrics["momentum_relative_error"], 3.0e-9)
        self.assertLessEqual(metrics["energy_relative_error"], 3.0e-9)
        self.assertTrue(all(particle.weight >= 0.0 for particle in proposal.particles))
        self.assertFalse(metrics["ancestry"]["exact_ancestry"])

    def test_C_two_output_merge_conserves_declared_invariants(self) -> None:
        proposal = comparator.vranic_two_output_merge(self.roots, self.merge_grid)
        metrics = comparator.compare_representation(self.roots, proposal, self.grid)
        self.assertLess(len(proposal.particles), len(self.roots))
        self.assertLessEqual(metrics["charge_relative_error"], 2.0e-14)
        self.assertLessEqual(metrics["centroid_normalized_error"], 2.0e-12)
        self.assertLessEqual(metrics["momentum_relative_error"], 2.0e-12)
        self.assertLessEqual(metrics["energy_relative_error"], 2.0e-12)
        self.assertLessEqual(metrics["maximum_relative_mass_shell_error"], 2.0e-12)
        self.assertTrue(metrics["ancestry"]["exact_ancestry"])
        self.assertFalse(metrics["ancestry"]["manufactured_support"])

    def test_D_fixed_seed_is_reproducible_but_is_not_exact_lineage(self) -> None:
        first = comparator.stratified_residual_thinning(self.roots, 64, 271828)
        second = comparator.stratified_residual_thinning(self.roots, 64, 271828)
        third = comparator.stratified_residual_thinning(self.roots, 64, 271829)
        self.assertEqual(
            comparator._representation_digest(first),
            comparator._representation_digest(second),
        )
        self.assertNotEqual(
            comparator._representation_digest(first),
            comparator._representation_digest(third),
        )
        metrics = comparator.compare_representation(self.roots, first, self.grid)
        self.assertLessEqual(metrics["charge_relative_error"], 2.0e-14)
        self.assertLessEqual(len(first.particles), 64)
        self.assertFalse(metrics["ancestry"]["exact_ancestry"])
        self.assertEqual(first.details["status"], "explicitly_experimental")

    def test_failure_cases_fail_closed_without_mutating_input(self) -> None:
        before = comparator._input_digest(self.roots)
        failures = comparator.run_failure_cases(self.roots, self.grid)
        self.assertEqual(len(failures), 5)
        self.assertTrue(all(case["pass"] for case in failures))
        self.assertTrue(all(case["observed"] == "rollback" for case in failures))
        self.assertEqual(before, comparator._input_digest(self.roots))

    def test_full_report_and_machine_readable_outputs(self) -> None:
        report = comparator.run_comparison(count=96, repeats=2)
        self.assertEqual(report["schema"], comparator.SCHEMA)
        self.assertTrue(report["standalone_not_production_integration"])
        self.assertTrue(all(case["pass"] for case in report["failure_cases"]))
        for scenario in report["scenarios"].values():
            self.assertEqual(len(scenario["results"]), 4)
            self.assertTrue(
                all(result["performance"]["repeat_deterministic"] for result in scenario["results"])
            )
        with tempfile.TemporaryDirectory(prefix="feebl-prototype-comparison-") as path:
            output = Path(path) / "evidence"
            comparator.write_outputs(report, output)
            loaded = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], comparator.SCHEMA)
            with (output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            with (output / "failure_cases.csv").open(newline="", encoding="utf-8") as stream:
                failure_rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(failure_rows), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
