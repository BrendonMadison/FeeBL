"""Small synthetic tests for units, paired covariance and incomplete admission."""
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from contracts import BINARY_HASH, SOURCE_ID, load_campaign, populations
from definitions import ARMS
from ensemble_statistics import ratio_statistics, summarize


class StatisticalTests(unittest.TestCase):
    def test_microbarn_to_millibarn_conversion(self):
        vectors = [[1000, 2000, 3000, 2, 1, 1, 1]]*4
        mean, covariance = ratio_statistics(vectors, ["example"])
        np.testing.assert_allclose(mean, [.5, 1, 1.5])
        np.testing.assert_array_equal(covariance, np.zeros((3, 3)))

    def test_shared_luminosity_fluctuation_cancels(self):
        vectors = [[2000*lumi, 3000*lumi, 4000*lumi, lumi, 1, 1, 1] for lumi in (1, 2, 3, 4)]
        mean, covariance = ratio_statistics(vectors, ["example"])
        np.testing.assert_allclose(mean, [2, 3, 4])
        np.testing.assert_allclose(covariance, 0, atol=1e-25)

    def test_paired_change_keeps_cross_arm_cancellation(self):
        names = [arm for arm, *_ in ARMS]
        vectors = []
        for fluctuation in (-150, -50, 50, 150):
            row = []
            for arm in range(9):
                row.extend([1000+arm*100+fluctuation, 2000+arm*100+fluctuation,
                            3000+arm*100+fluctuation, 2, 1, 1, 1])
            vectors.append(row)
        result = summarize(vectors, names)
        self.assertGreater(result["records"][0]["sem_mb"], 0)
        change = result["changes"][0]
        np.testing.assert_allclose(change["change_mb"], [.05, .05, .05])
        np.testing.assert_allclose(change["covariance_mb2"], 0, atol=1e-18)
        np.testing.assert_allclose(change["jackknife_covariance_mb2"], 0, atol=1e-18)

    def test_only_complete_four_seed_summary(self):
        with self.assertRaises(ValueError):
            summarize(np.zeros((3, 63)), [arm for arm, *_ in ARMS])


class AdmissionTests(unittest.TestCase):
    def test_incomplete_campaign_cannot_be_reduced(self):
        progress = dict(status="RUNNING", not_started=[], source_native_identity=SOURCE_ID,
                        binary_sha256=BINARY_HASH, manifest_sha256="fixture")
        with patch("contracts.read_json", side_effect=[{}, progress]), patch("contracts.digest", return_value="fixture"):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_campaign(Path("synthetic_campaign"), Path("synthetic_inputs"))

    def test_birth_cut_is_not_a_final_energy_recut(self):
        path = Mock()
        path.open.side_effect = lambda **_: StringIO("0.004 0 0 0.9 0 0 1 2 -1 -1\n")
        np.testing.assert_array_equal(populations(path), [[0, 0], [0, 0], [1, 0]])
        with self.assertRaisesRegex(ValueError, "below"):
            populations(path, birth=True)

    def test_unexpected_population_fails(self):
        path = Mock()
        path.open.return_value = StringIO("0.1 0 0 0.9 0 0 1 3 -1 -1\n")
        with self.assertRaisesRegex(ValueError, "population"):
            populations(path)


if __name__ == "__main__":
    unittest.main()
