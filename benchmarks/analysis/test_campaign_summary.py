"""Check paired-seed aggregation and native-binary comparability."""

from pathlib import Path
import json
import unittest
import uuid

import numpy as np

from campaign_summary import summarize_domains
from comparison_statistics import campaign_inventory, reduce_campaign, write_rows


class DomainSummaryTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent
        self.directory = self.test_root / ("test_summary_" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.baseline, self.expanded = [], []
        for seed, value in enumerate((1.0, 2.0, 3.0)):
            row = {
                "case": "halhf380",
                "seed": str(seed),
                "binary_sha256": "same-binary",
                "outside_fraction_beam1": 0.1,
                "outside_fraction_beam2": 0.05,
            }
            for metric in (
                "lumi_ee",
                "lumi_ee_high",
                "lumi_gg",
                "n_phot1",
                "n_phot2",
                "de1",
                "de2",
            ):
                row[metric] = value
            self.baseline.append(row)
            self.expanded.append(
                {
                    **row,
                    "case": "halhf380_domain_x2",
                    "baseline_case": "halhf380",
                    "domain_factor": 2,
                    "lumi_ee": value * (1.1 + 0.1 * seed),
                    "outside_fraction_beam1": 0.01 * seed,
                }
            )

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.test_root))
        for path in self.directory.iterdir():
            self.assertTrue(path.is_file())
            path.unlink()
        self.directory.rmdir()

    def summarize(self):
        baseline_path, domain_path = (
            self.directory / "baseline.csv",
            self.directory / "domain.csv",
        )
        write_rows(baseline_path, self.baseline)
        write_rows(domain_path, self.expanded)
        return summarize_domains(baseline_path, domain_path, self.directory)

    def test_ratios_are_paired_before_averaging(self):
        result = self.summarize()[0]
        self.assertAlmostEqual(result["paired_luminosity_ratio"], 1.2)
        self.assertAlmostEqual(result["paired_luminosity_ratio_sem"], 0.1 / np.sqrt(3))
        self.assertEqual(
            result["domain_maximum_recorded_rejection_fraction_beam1"], 0.02
        )

    def test_mixed_binaries_are_rejected(self):
        self.expanded[1]["binary_sha256"] = "other-binary"
        with self.assertRaisesRegex(ValueError, "same native binary"):
            self.summarize()

    def test_duplicate_seeds_are_rejected(self):
        self.expanded[1]["seed"] = self.expanded[0]["seed"]
        with self.assertRaisesRegex(ValueError, "distinct matched seeds"):
            self.summarize()


class CampaignInventoryTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent
        self.directory = self.test_root / ("test_inventory_" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.folders, jobs = [], []
        for case in ("complete", "incomplete"):
            for seed in (1, 2):
                directory = self.directory / f"{case}_{seed}"
                directory.mkdir()
                self.folders.append(directory)
                status = "FAILED" if (case, seed) == ("incomplete", 2) else "EXECUTED"
                receipt = {
                    "status": status,
                    "binary_sha256": "one-binary",
                    "error": "Numerical diagnostic" if status == "FAILED" else None,
                }
                (directory / "execution.json").write_text(json.dumps(receipt))
                jobs.append({"case": case, "seed": seed, "directory": directory.name})
        (self.directory / "campaign.json").write_text(
            json.dumps({"seed_count": 2, "jobs": jobs})
        )

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.test_root))
        for directory in self.folders:
            (directory / "execution.json").unlink()
            directory.rmdir()
        (self.directory / "campaign.json").unlink()
        self.directory.rmdir()

    def test_failed_seed_remains_in_inventory(self):
        inventory = campaign_inventory(self.directory)
        cases = {case["case"]: case for case in inventory["cases"]}
        self.assertTrue(cases["complete"]["complete_requested_ensemble"])
        self.assertFalse(cases["incomplete"]["complete_requested_ensemble"])
        self.assertEqual(cases["incomplete"]["records"][1]["status"], "FAILED")
        self.assertFalse(inventory["all_requested_cases_complete"])

    def test_full_reduction_refuses_incomplete_campaign(self):
        output = self.directory / "must_not_exist"
        with self.assertRaisesRegex(ValueError, "Campaign incomplete"):
            reduce_campaign(self.directory, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
