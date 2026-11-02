"""Small numerical/selection fixtures; no simulated publication results."""
import math
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import lep_support as lep
import reduce_figures as analysis


class StatisticsTests(unittest.TestCase):
    def test_covariance_of_mean_preserves_correlations(self):
        values = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
        mean, factor, variance = analysis.low_rank_covariance(values)
        np.testing.assert_allclose(mean, [2, 2])
        expected = np.array([[1, -1], [-1, 1]])/3
        np.testing.assert_allclose(factor.T@factor, expected)
        np.testing.assert_allclose(variance, np.diag(expected))

    def test_table_total_preserves_channel_covariance(self):
        import render_tables
        values = np.array([[1, 3], [2, 2], [3, 1]], dtype=float)
        mean, sem = render_tables.sum_mean_and_sem(values)
        self.assertEqual(mean, 4)
        self.assertEqual(sem, 0)
        individual_sem = [render_tables.mean_and_sem(values[:, i])[1] for i in range(2)]
        self.assertGreater(np.linalg.norm(individual_sem), 0)
        with self.assertRaises(ValueError):
            render_tables.mean_and_sem([1, float("nan")])
        reference_rows = [dict(process=p, reference=10) for p in
                          ("Breit-Wheeler", "Bethe-Heitler", "Landau-Lifshitz")]
        summary = dict(comparisons={f"model{m}_luminosity": dict(columns=["lumi_ee"],
            mean=[2e34], standard_uncertainty=[1e32]) for m in (1, 2)})
        for model in (1, 2):
            summary["comparisons"][f"model{model}_effective_cross_section"] = dict(
                mean_mb=[1, 2, 3], covariance_of_mean_mb2=[[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        with patch.object(render_tables, "read_json", return_value=summary), \
                patch.object(render_tables, "read_csv", return_value=reference_rows), \
                patch.object(render_tables.np, "load") as loader:
            loader.return_value.__enter__.return_value = {"vectors": np.tile([1, 2, 3], (12, 1))}
            text, records = render_tables.ilc_table(Path("synthetic_reduction"))
            self.assertIn(r"\begin{tabular}", text)
            self.assertEqual(records[3]["model1"]["mean"], 6)
            self.assertEqual(records[3]["model1"]["standard_uncertainty"], 0)
            self.assertAlmostEqual(records[-1]["model1"]["mean"], 2)
            self.assertAlmostEqual(records[-1]["model1"]["standard_uncertainty"], .01)

    def test_luminosity_covariance_cancels_a_common_exposure_fluctuation(self):
        luminosity = np.array([1., 2., 4., 8.])*1e31
        counts = luminosity[:, None]*np.array([[2., 7.]])*1e-31
        estimate = analysis.luminosity_normalized_estimate(counts, luminosity)
        np.testing.assert_allclose(estimate["mean_mb"], [2, 7])
        np.testing.assert_allclose(estimate["covariance_of_mean_mb2"], 0, atol=1e-14)
        np.testing.assert_allclose(estimate["jackknife_covariance_mb2"], 0, atol=1e-14)
        # Removing the correlation must fail this cancellation, not look more precise.
        self.assertGreater(counts[:, 0].var(ddof=1)/len(counts), 0)

    def test_luminosity_ratio_uses_total_exposure_not_mean_of_run_ratios(self):
        counts = np.array([[1.], [3.], [8.], [20.]])
        luminosity = np.array([1., 2., 4., 10.])*1e31
        estimate = analysis.luminosity_normalized_estimate(counts, luminosity)
        self.assertAlmostEqual(estimate["mean_mb"][0], 32/17)
        self.assertNotAlmostEqual(estimate["mean_mb"][0], np.mean(counts[:, 0]/(luminosity*1e-31)))
        with self.assertRaises(ValueError):
            analysis.luminosity_normalized_estimate(counts, luminosity*0)

    def test_pull_uses_both_uncertainties_and_unit_bar(self):
        rows = analysis.combined_rows([dict(bin=0)], np.array([4.]), np.array([[3.]]),
                                      np.array([2.]), np.array([[1.]]), 10)
        self.assertEqual(rows[0]["pull"], 1)
        self.assertEqual(rows[0]["pull_standard_uncertainty"], 1)
        self.assertEqual(rows[0]["variance_degrees_of_freedom"], 9)

    def test_equal_strata_keep_event_legs_and_boundaries_paired(self):
        # Stratum zero has two samples; all others have one. Equal-stratum
        # averaging must differ from treating the 49 vertices as unit rows.
        strata = [0, 0, *range(1, 48)]
        rows = []
        for sample, index in enumerate(strata):
            phi = -math.pi+(index+.5)*2*math.pi/48
            for cohort, role in lep.POPULATIONS:
                value = index+(100 if role == "positron" else 0)+(10 if cohort == "outer" else 0)
                rows.append(dict(chunk_index=0, sample=sample, cohort=cohort, role=role,
                    phi_rad=phi, planned_phi_rad=phi, t_s=0, focus_urad=value))
        estimate = lep.estimate_field(rows, [-1], [60])
        self.assertAlmostEqual(estimate["vector"][-6], 23.5)
        self.assertAlmostEqual(estimate["vector"][-5], 123.5)
        self.assertNotAlmostEqual(estimate["raw_scalars"][0], 23.5)
        self.assertEqual(estimate["paired_vertices"], 49)
        with self.assertRaises(ValueError):
            lep.estimate_field(rows[:-1], [-1], [60])

    def test_admission_allows_one_float_ulp_not_two(self):
        expected = lep.admitted_velocity(45.6, [.01, .02, .99975])
        observed = np.nextafter(expected.astype(np.float32), np.float32(np.inf)).astype(float)
        self.assertTrue(lep.admitted_matches(observed, expected))
        observed = np.nextafter(observed.astype(np.float32), np.float32(np.inf)).astype(float)
        self.assertFalse(lep.admitted_matches(observed, expected))

    def test_hemisphere_is_position_and_pt_is_energy_times_beta(self):
        fixture = "1 .001 0 -1 0 0 1 0 0 0\n-1 .001 0 1 0 0 -1 1 0 0\n"
        with patch("pathlib.Path.open", return_value=io.StringIO(fixture)):
            path = Path("synthetic_secondary_axis_fixture.dat")
            grid = (np.array([-4., -2.]), np.array([-4., -2.]),
                    np.ones((1, 1), dtype=bool), {})
            counts, note = analysis.reduce_secondary(path, dict(forward=grid, backward=grid))
            self.assertEqual(counts["forward_electron"].sum(), 1)
            self.assertEqual(counts["backward_positron"].sum(), 1)
            self.assertEqual(note["position_momentum_sign_disagree"], 2)
            self.assertEqual(sum(note["under_or_overflow"].values()), 0)

    def test_rare_seed_is_retained_and_its_influence_exposed(self):
        values = np.array([[1.], [1.], [1.], [9.]])
        note = analysis.leave_one_out(values)
        self.assertEqual(note["maximum_seed_value"], [9.])
        self.assertEqual(note["leave_one_out_mean_min"], [1.])
        self.assertEqual(note["leave_one_out_sem_min"], [0.])
        self.assertGreater(note["leave_one_out_sem_max"][0], note["full_sample_sem"][0])
        self.assertEqual(note["first_half"]["sem"], [0.])
        self.assertEqual(note["second_half"]["sem"], [4.])
        np.testing.assert_allclose(values.ravel(), [1, 1, 1, 9])

    def test_luminosity_accepts_consistent_native_duplicate(self):
        text = ("lumi_ee = 1.23456e+31 m**(-2) (from beam particles collisions)\n"
                "lumi_fine = 1.25e+31 m**(-2)\n"
                "lumi_eg = 2e+29 m**(-2)\n"
                "lumi_ge = 3e+29 m**(-2)\n"
                "lumi_gg = 4e+28 m**(-2)\n"
                "lumi_ee=1.23456e+31;\n")
        with patch("pathlib.Path.read_text", return_value=text):
            self.assertEqual(analysis.collision_luminosity(Path("native_fixture"))["lumi_ee"], 1.23456e31)
        with patch("pathlib.Path.read_text", return_value=text.replace("lumi_ee=1.23456e+31;", "lumi_ee=9e+31;")):
            with self.assertRaises(ValueError):
                analysis.collision_luminosity(Path("native_fixture"))

    def test_replot_rejects_changed_output_and_producer(self):
        import render_figures
        receipt = dict(status="REDUCED_NOT_PHYSICS_QUALIFIED", source_native_identity="source-id",
            output_sha256={"comparison.csv": "table-hash"},
            reused_analysis_sha256={"comparison_statistics.py": "helper-hash"},
            producer_sha256={"reduce_figures.py": "producer-hash"})
        hashes = {"comparison.csv": "table-hash", "comparison_statistics.py": "helper-hash", "reduce_figures.py": "producer-hash"}
        with patch.object(render_figures, "read_json", return_value=receipt), \
                patch("pathlib.Path.read_text", return_value="source-id"), \
                patch.object(render_figures, "digest", side_effect=lambda p: hashes[p.name]):
            self.assertIs(render_figures.verify_reduction(Path("reduction"), Path("source")), receipt)
            hashes["comparison.csv"] = "changed"
            with self.assertRaises(ValueError):
                render_figures.verify_reduction(Path("reduction"), Path("source"))
            hashes["comparison.csv"] = "table-hash"
            hashes["reduce_figures.py"] = "changed"
            with self.assertRaises(ValueError):
                render_figures.verify_reduction(Path("reduction"), Path("source"))


if __name__ == "__main__":
    unittest.main()
