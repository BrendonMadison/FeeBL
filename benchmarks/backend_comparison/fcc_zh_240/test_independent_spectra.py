"""Verify independent-ensemble uncertainty and fixed disjoint seed selection."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from independent_spectra import independent_comparison, select_independent_rows


class IndependentStatistics(unittest.TestCase):
    def test_covariance_of_means_adds_without_cross_subtraction(self):
        cpu = [[1, 2], [2, 5], [3, 8], [4, 11]]
        gpu = [[2, 0], [4, 1], [6, 2], [8, 3]]
        result = independent_comparison(cpu, gpu)
        np.testing.assert_allclose(result["difference_gpu_minus_cpu"], [2.5, -5])
        np.testing.assert_allclose(result["difference_covariance_of_mean"],
                                   [[25/12, 25/12], [25/12, 25/6]])
        np.testing.assert_allclose(result["pull"], [np.sqrt(3), -np.sqrt(6)])

    def test_equal_means_still_have_independent_uncertainty(self):
        values = [[1], [2], [3], [4]]
        result = independent_comparison(values, values)
        self.assertGreater(result["combined_sem"][0], 0)
        self.assertEqual(result["pull_state"], ["estimated_independent_means"])
        self.assertEqual(result["pull_error"], [1])

    def test_zero_observed_support_is_explicit(self):
        result = independent_comparison(np.zeros((4, 1)), np.zeros((4, 1)))
        self.assertEqual(result["pull"], [0])
        self.assertEqual(result["pull_state"], ["zero_variance_identity_convention"])

    def test_actual_inventory_selects_four_disjoint_seeds_per_arm(self):
        document = json.loads(Path(__file__).with_name("analysis_manifest.json").read_text())
        rows = select_independent_rows(document)
        self.assertEqual([r["seed"] for r in rows["cpu"]], [72942001,72942003,72942005,72942007])
        self.assertEqual([r["seed"] for r in rows["gpu"]], [72942002,72942004,72942006,72942008])
        changed = copy.deepcopy(document)
        changed["runs"] = [r for r in changed["runs"] if r["seed"] != 72942008]
        with self.assertRaises(ValueError):
            select_independent_rows(changed)


if __name__ == "__main__":
    unittest.main()
