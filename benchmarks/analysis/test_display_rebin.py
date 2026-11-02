"""Check uncertainty transfer and preservation of unsupported intervals."""

import unittest
import numpy as np
from display_rebin import display_edges, remap


class DisplayRebinTests(unittest.TestCase):
    def test_off_diagonal_covariance_is_retained(self):
        rows = [
            {
                "bin_low": i,
                "bin_high": i + 1,
                "simulation_mass": 0.5,
                "reference_mass": 0.4,
            }
            for i in range(2)
        ]
        covariance = {
            "simulation": np.array([[0.04, 0.03], [0.03, 0.04]]),
            "reference": np.eye(2) * 0.01,
            "simulation_reference": np.zeros((2, 2)),
        }
        result, matrices, _ = remap(rows, covariance, count=1)
        self.assertAlmostEqual(matrices["simulation"][0, 0], 0.14)
        self.assertAlmostEqual(result[0]["pull"], 0.2 / np.sqrt(0.16))

    def test_gaps_are_not_interpolated(self):
        low, high = display_edges(
            np.array([0.0, 1.0, 4.0]), np.array([1.0, 2.0, 5.0]), 40
        )
        self.assertEqual(len(low), 40)
        self.assertTrue(np.all((high <= 2) | (low >= 4)))

    def test_regularizer_flags_survive_nonempty_and_empty_rebinning(self):
        rows = [
            {
                "bin_low": i,
                "bin_high": i + 1,
                "simulation_mass": value,
                "reference_mass": 0.2,
                "empty_bin_regularized": i == 0,
            }
            for i, value in enumerate((0.0, 0.5))
        ]
        covariance = {
            "simulation": np.eye(2) * 0.04,
            "reference": np.eye(2) * 0.01,
            "simulation_reference": np.zeros((2, 2)),
        }
        combined, _, _ = remap(rows, covariance, count=1)
        self.assertTrue(combined[0]["contains_regularized_variance"])
        self.assertFalse(combined[0]["empty_bin_regularized"])
        separate, _, _ = remap(rows, covariance, count=2)
        self.assertTrue(separate[0]["empty_bin_regularized"])
        self.assertFalse(separate[1]["contains_regularized_variance"])

    def test_crop_preserves_original_normalization(self):
        rows = [
            {
                "bin_low": 0.0,
                "bin_high": 2.0,
                "simulation_mass": 1.0,
                "reference_mass": 0.8,
            }
        ]
        covariance = {
            "simulation": np.array([[0.04]]),
            "reference": np.array([[0.01]]),
            "simulation_reference": np.zeros((1, 1)),
        }
        result, matrices, _ = remap(rows, covariance, count=1, limits=(1.0, 3.0))
        self.assertEqual(result[0]["bin_high"], 2.0)
        self.assertEqual(result[0]["simulation_mass"], 0.5)
        self.assertAlmostEqual(matrices["simulation"][0, 0], 0.01)


if __name__ == "__main__":
    unittest.main()
