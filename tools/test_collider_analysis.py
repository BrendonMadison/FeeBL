import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "analysis", Path(__file__).with_name("collider_analysis.py")
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

SAMPLE = """#FEEBL-NATIVE 1.0
#energy_unit GeV
#half_crossing_angle_rad 0
#frame lab
#momenta_carry_crossing_angle no
#BLOCK pair_spectrum
#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff
249 249.5 4 2 0 0 0 0 0 0 0
250 250.5 6 3 0 0 0 0 0 0 0
#underflow 2
#overflow 3
#mean_W 249.9 +- 0.1
#rms_W 0.2
#n_eff_total 100
#END pair_spectrum records 2
"""
SUMMARY = "lumi_ee = 10 m**(-2) (from beam particles collisions)\n"


class AnalysisTests(unittest.TestCase):
    def result(self, **kw):
        return a.analyze_native(SAMPLE, SUMMARY, [125, 125], kw.pop("angle", 0), **kw)

    def reject(self, text):
        with self.assertRaises(ValueError):
            a.parse_spectrum(text)

    def test_native_prose_heading(self):
        m, rows = a.parse_spectrum(
            SAMPLE.replace("#frame lab", "# frame and conventions -- read these\n#frame lab")
        )
        self.assertEqual(m["frame"], "lab")
        self.assertEqual(len(rows), 2)

    def test_empty_missing_columns(self):
        self.reject(
            SAMPLE.replace(
                "#COLUMNS W_low W_high dL_dW weight h1 h1_err h2 h2_err h1h2 cov n_eff", ""
            )
        )

    def test_exact_native_bins(self):
        result = self.result()
        self.assertEqual(
            result["bins"][0]["native_columns"], [249, 249.5, 4, 2, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_full_total(self):
        self.assertEqual(self.result()["normalization"]["total_spectrum_m2_per_crossing"], 10)

    def test_flow_in_shape_denominator(self):
        self.assertEqual(self.result()["bins"][0]["shape_density_per_GeV"], 0.4)

    def test_machine_total_scope(self):
        result = self.result(
            crossing_frequency_hz=100, normalization_scope="machine_total_unallocated"
        )
        self.assertEqual(result["normalization"]["allocation_scope"], "machine_total_unallocated")
        self.assertIn("never quote as per-IP", result["normalization"]["scope"])

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):
            self.result(normalization_scope="guess")

    def test_explicit_frequency(self):
        r = self.result(crossing_frequency_hz=100)
        self.assertAlmostEqual(r["normalization"]["total_L_cm2_s"], 0.1)
        self.assertAlmostEqual(r["bins"][0]["dL_dW_cm2_s_GeV"], 0.04)

    def test_unknown_frequency_is_null(self):
        self.assertIsNone(self.result()["normalization"]["total_L_cm2_s"])

    def test_asymmetric_nominal(self):
        self.assertAlmostEqual(a.nominal_w([250, 50], 0), math.sqrt(50000), places=7)
        self.assertNotEqual(a.nominal_w([250, 50], 0), 300)

    def test_angle_qualification(self):
        r = self.result(angle=0.03)
        self.assertTrue(r["definition"]["finite_angle_approximation"])
        self.assertAlmostEqual(r["definition"]["nominal_W_GeV"], 250 * math.cos(0.015), places=7)
        self.assertEqual(r["bins"][0]["W_low_GeV"], 249)

    def test_peak_lower_and_high_side(self):
        r = self.result()
        self.assertEqual(r["plot"]["peak"]["x_min_GeV"], 245)
        self.assertGreater(r["plot"]["x_max_GeV"], 250)

    def test_gap_not_zero_bin(self):
        r = self.result()
        self.assertEqual(len(r["bins"]), 2)
        self.assertEqual(
            r["missing_record_gaps"],
            [dict(low_GeV=249.5, high_GeV=250, status="NO_SAMPLED_RECORD")],
        )

    def test_no_error_invention(self):
        r = self.result()
        self.assertIsNone(r["bins"][0]["uncertainty"])
        self.assertIsNone(r["native_statistics"]["gaussian_core_fit"])

    def test_integral_not_renormalized(self):
        interval = self.result()["high_energy_integral_bounds"]["0.98"]
        self.assertEqual(interval["weight_m2_per_crossing_bounds"], [8, 10])
        self.assertEqual(interval["fraction_of_total_bounds"], [0.8, 1])

    def test_missing_footer(self):
        self.reject(SAMPLE.replace("#END pair_spectrum records 2", ""))

    def test_footer_count(self):
        self.reject(SAMPLE.replace("records 2", "records 3"))

    def test_truncated_row(self):
        self.reject(SAMPLE.replace("249 249.5 4 2 0 0 0 0 0 0 0", "249 249.5"))

    def test_nonfinite(self):
        self.reject(SAMPLE.replace("249 249.5 4 2", "249 249.5 nan 2"))

    def test_zero_weight(self):
        self.reject(SAMPLE.replace("249 249.5 4 2", "249 249.5 4 0"))

    def test_wrong_columns(self):
        self.reject(SAMPLE.replace("W_low W_high", "low high"))

    def test_duplicate_metadata(self):
        self.reject(SAMPLE.replace("#energy_unit GeV", "#energy_unit GeV\n#energy_unit MeV"))

    def test_wrong_units(self):
        self.reject(SAMPLE.replace("#energy_unit GeV", "#energy_unit MeV"))

    def test_overlap(self):
        self.reject(SAMPLE.replace("250 250.5", "249 249.8"))

    def test_tail_after_footer(self):
        self.reject(SAMPLE + "extra\n")

    def test_summary_mismatch_is_flagged(self):
        r = a.analyze_native(SAMPLE, SUMMARY.replace("10 ", "12 "), [125, 125], 0)
        self.assertEqual(r["normalization"]["comparison_status"], "MISMATCH_REQUIRES_INVESTIGATION")

    def test_bad_frequency(self):
        with self.assertRaises(ValueError):
            self.result(crossing_frequency_hz=-1)

    def test_file_exports_match_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "lumi.ee.out.dLdW.fne").write_text(SAMPLE)
            (p / "out.dat").write_text(SUMMARY)
            r = a.analyze_run(p, [125, 125], 0)
            self.assertTrue((p / "collider_analysis.json").is_file())
            self.assertIn("249.0,249.5,4.0,2.0,,0.4", (p / "ee_spectrum.csv").read_text())
            self.assertEqual(
                r["sources"]["spectrum"]["sha256"], a.sha256(p / "lumi.ee.out.dLdW.fne")
            )


if __name__ == "__main__":
    unittest.main()
