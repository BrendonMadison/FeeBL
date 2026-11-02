"""Independent smooth-peak and noisy-maximum checks for the width analysis."""

import math
import unittest

import numpy as np
from scipy.special import ndtr

from empirical_fwhm import group_histogram, measure
from peak_resolution import analyze, group_spectrum, phase_measurements, resolution_class


def gaussian_spectrum(sigma_MeV):
    """Exact bin integrals on the stored 125 GeV companion-spectrum grid."""
    edges = np.r_[np.linspace(0, 120, 2049), np.linspace(120, 130, 50001)[1:],
                  np.linspace(130, 137.5, 513)[1:]]
    low, high = edges[:-1], edges[1:]
    a, b = (low - 125) / (sigma_MeV / 1000), (high - 125) / (sigma_MeV / 1000)
    weight = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    return dict(low=low, high=high, weight=weight,
        metadata={"peak_bins": "50000"}, numeric={"nominal_W_GeV": 125.0})


class PeakResolutionTests(unittest.TestCase):
    def test_fixed_classes_do_not_follow_histogram_fluctuations(self):
        self.assertEqual(resolution_class("ilc_sig1_scan"), "sigma1")
        self.assertEqual(resolution_class("halhf_sig4_scan"), "sigma4")
        self.assertEqual(resolution_class("fcc_cw_d100_pilot_s1"), "broad")
        with self.assertRaises(ValueError):
            resolution_class("unknown")

    def test_grouping_matches_previous_implementation(self):
        spectrum = gaussian_spectrum(1)
        for size in (1, 2, 4):
            for phase in range(size):
                old = group_histogram(spectrum["low"], spectrum["high"], spectrum["weight"], 2048, 52048, size, phase)
                new = group_spectrum(spectrum, size, phase)
                for key in ("low", "high", "weight", "fine", "source_start", "source_stop"):
                    np.testing.assert_array_equal(old[key], new[key])

    def test_partial_groups_conserve_mass(self):
        spectrum = gaussian_spectrum(60)
        for size in (32, 64):
            for phase in range(size):
                grouped = group_spectrum(spectrum, size, phase)
                self.assertAlmostEqual(math.fsum(grouped["weight"]), 1, places=14)

    def test_broad_primary_bias_is_below_one_percent(self):
        for fwhm in (80, 140, 350, 420):
            spectrum = gaussian_spectrum(fwhm / math.sqrt(8 * math.log(2)))
            values = phase_measurements(spectrum, 32)
            errors = [abs(r["fwhm_MeV"] / fwhm - 1) for r in values if r["group_size"] == 32]
            self.assertLess(max(errors), 0.01)
            self.assertTrue(all(r["components"] == 1 for r in values))

    def test_narrow_primary_response_and_double_resolution(self):
        for sigma, size in ((1, 1), (2, 2), (4, 4)):
            values = phase_measurements(gaussian_spectrum(sigma), size)
            truth = sigma * math.sqrt(8 * math.log(2))
            primary = [abs(r["fwhm_MeV"] / truth - 1) for r in values if r["group_size"] == size]
            self.assertLess(max(primary), 0.01)
            self.assertLess(max(abs(r["fwhm_MeV"] / truth - 1) for r in values), 0.03)

    def test_old_fourfold_probe_can_fail_without_noise(self):
        spectrum = gaussian_spectrum(1)
        native = measure(group_spectrum(spectrum, 1, 0))["fwhm_MeV"]
        old = [measure(group_spectrum(spectrum, 4, phase))["fwhm_MeV"] for phase in range(4)]
        self.assertGreater(max(abs(v / native - 1) for v in old), 0.05)
        current = analyze(spectrum, "ilc_sig1_scan")
        self.assertEqual(current["status"], "BINNING_STABLE_AT_FIVE_PERCENT_TARGET")

    def test_noisy_maximum_is_not_mistaken_for_a_narrow_physical_peak(self):
        spectrum = gaussian_spectrum(56.56854249)
        center = np.argmax(spectrum["weight"])
        spectrum["weight"][center] *= 2.5
        raw = measure(group_spectrum(spectrum, 1, 0))["fwhm_MeV"]
        current = analyze(spectrum, "fcc_d0_control")
        self.assertLess(raw, 2)
        self.assertGreater(current["fwhm_MeV"], 100)

    def test_disconnected_peaks_remain_flagged(self):
        spectrum = gaussian_spectrum(1)
        shift = 100
        spectrum["weight"] += np.roll(spectrum["weight"], shift)
        current = analyze(spectrum, "ilc_sig1_scan")
        self.assertIn("UNRESOLVED_HALF_HEIGHT_COMPONENTS", current["flags"])

    def test_peak_at_fine_segment_boundary_remains_flagged(self):
        spectrum = gaussian_spectrum(1)
        spectrum["weight"][:] = 0
        spectrum["weight"][2048:2050] = [1.0, 0.5]
        current = analyze(spectrum, "ilc_sig1_scan")
        self.assertIn("UNRESOLVED_FINE_SEGMENT_SUPPORT", current["flags"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
