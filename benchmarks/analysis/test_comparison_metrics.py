"""Check exact CDF integration for sign changes and separated supports."""
import unittest
from comparison_metrics import distances


class ComparisonMetricTests(unittest.TestCase):
    def test_shifted_uniform_distributions(self):
        result = distances([0, 1], [1, 2], [1.0, 0.0], [0.0, 1.0])
        self.assertEqual(result["conditional_cdf_distance"], 1.0)
        self.assertEqual(result["wasserstein1_GeV"], 1.0)

    def test_unobserved_gap_does_not_disappear_from_energy_distance(self):
        result = distances([0, 3], [1, 4], [1.0, 0.0], [0.0, 1.0])
        self.assertEqual(result["wasserstein1_GeV"], 3.0)

    def test_crossing_cdf_has_two_triangles_in_middle_bin(self):
        result = distances([0, 1, 2], [1, 2, 3], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0])
        self.assertEqual(result["conditional_cdf_distance"], 0.5)
        self.assertEqual(result["wasserstein1_GeV"], 0.75)


if __name__ == "__main__":
    unittest.main()
