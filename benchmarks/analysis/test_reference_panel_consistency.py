"""Keep censored support and paired histogram transformations explicit."""
import unittest
import numpy as np

from reference_panel_consistency import common_bins, overlap_matrix, conditioned


class CommonSupportTests(unittest.TestCase):
    def test_clipped_support_preserves_gap(self):
        low, high = common_bins([0, 2], [2, 4], [0.5, 2.5], [1.5, 3.5], False)
        np.testing.assert_array_equal(low, [0.5, 2.5])
        np.testing.assert_array_equal(high, [1.5, 3.5])

    def test_interior_selection_does_not_cut_coarse_bin(self):
        low, high = common_bins([0, 1, 2], [1, 2, 3], [0.5], [2.5], True)
        np.testing.assert_array_equal(low, [1])
        np.testing.assert_array_equal(high, [2])

    def test_different_representations_of_constant_density_agree(self):
        target_low, target_high = np.array([0.5, 1]), np.array([1, 1.5])
        coarse = overlap_matrix(np.array([0]), np.array([2]), target_low, target_high)
        fine = overlap_matrix(
            np.array([0, 1]), np.array([1, 2]), target_low, target_high
        )
        a, ac = conditioned(np.array([1.0]), np.array([[0.0]]), coarse)
        b, bc = conditioned(np.array([0.5, 0.5]), np.zeros((2, 2)), fine)
        np.testing.assert_allclose(a, b)
        np.testing.assert_array_equal(ac, bc)


if __name__ == "__main__":
    unittest.main()
