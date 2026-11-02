"""Check the combined-uncertainty definition and malformed comparison handling."""

import csv
import math
from pathlib import Path
import unittest
import uuid

from replot import plot_comparison


class PullComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = Path(__file__).resolve().parent
        self.output = self.test_root / ("test_output_" + uuid.uuid4().hex)
        self.output.mkdir()

    def tearDown(self) -> None:
        self.assertTrue(self.output.resolve().is_relative_to(self.test_root))
        for path in self.output.iterdir():
            self.assertTrue(path.is_file())
            path.unlink()
        self.output.rmdir()

    def write_case(
        self, simulation_variance=3.0, reference_variance=1.0, covariance=0.0, pull=1.0
    ) -> Path:
        row = {
            "process": "test",
            "simulation_count": 4.0,
            "reference_count": 2.0,
            "simulation_variance": simulation_variance,
            "reference_variance": reference_variance,
            "simulation_reference_covariance": covariance,
            "pull": pull,
        }
        path = self.output / "comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_both_variance_contributions_enter(self) -> None:
        result = plot_comparison(self.write_case(), self.output)
        self.assertAlmostEqual(result["maximum_absolute_pull"], 1.0)

    def test_covariance_enters_with_negative_twice_sign(self) -> None:
        path = self.write_case(covariance=1.0, pull=math.sqrt(2))
        result = plot_comparison(path, self.output)
        self.assertAlmostEqual(result["maximum_absolute_pull"], math.sqrt(2))

    def test_pull_omitting_simulation_variance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Stored pulls disagree"):
            plot_comparison(self.write_case(pull=2.0), self.output)

    def test_nonpositive_combined_variance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid uncertainty model"):
            plot_comparison(self.write_case(covariance=2.0), self.output)

    def test_nonfinite_data_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            plot_comparison(
                self.write_case(reference_variance=float("nan")), self.output
            )


if __name__ == "__main__":
    unittest.main()
