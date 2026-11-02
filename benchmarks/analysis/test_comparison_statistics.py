"""Check the sampling-unit, normalization and pull propagation contracts."""
import math
import unittest

import numpy as np

import comparison_statistics as statistics
import reference_uncertainty as reference


class ComparisonStatisticsTests(unittest.TestCase):
    def test_empirical_covariance_is_covariance_of_mean(self):
        samples = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        mean, covariance = statistics.sample_covariance(samples)
        np.testing.assert_array_equal(mean, [2.0, 2.0])
        np.testing.assert_allclose(covariance, [[1 / 3, -1 / 3], [-1 / 3, 1 / 3]])
        np.testing.assert_allclose(covariance.sum(axis=0), 0, atol=1e-15)

    def test_centering_handles_constant_nonbinary_values(self):
        _, covariance = statistics.sample_covariance(np.full((12, 3), 45.6))
        np.testing.assert_array_equal(covariance, np.zeros((3, 3)))

    def test_rebinning_preserves_mass_and_missing_support(self):
        bins = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
        result = statistics.rebin_mass(bins, [0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, [1 / 6, 1 / 6, 0.0, 4 / 6])

    def test_invalid_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            statistics.sample_covariance([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            statistics.sample_covariance([[1.0, 2.0], [math.nan, 3.0]])

    def test_local_linear_contrast(self):
        contrast = reference.polynomial_contrast(np.array([0.0, 1.0, 2.0]), 1)
        np.testing.assert_allclose(contrast, np.array([1.0, -2.0, 1.0]) / math.sqrt(6))
        self.assertAlmostEqual(float(contrast @ np.array([2.0, 5.0, 8.0])), 0.0)

    def test_normalized_reference_covariance(self):
        covariance = reference.normalize_covariance(
            [2.0, 4.0, 6.0], np.diag([1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(covariance.sum(axis=0), 0.0, atol=1e-16)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-16)

    def test_reference_raw_values_reconstruct_published_bins(self):
        for path in statistics.PANELS.glob("*_pull.csv"):
            rows = statistics.read_rows(path)
            values, covariance, _ = statistics.reference_estimate(path.stem, rows)
            key = reference.reference_key(rows[0])
            np.testing.assert_allclose(
                values, statistics.column(rows, key), rtol=1e-9, atol=2e-12
            )
            self.assertTrue(np.isfinite(covariance).all())
            self.assertTrue(np.all(np.diag(covariance) >= 0))


if __name__ == "__main__":
    unittest.main()
