"""Check momentum units in the native secondary-record reduction."""
from pathlib import Path
import unittest
import uuid

from reduce_pair_support import reduce_population


class PairSupportTests(unittest.TestCase):
    def test_beta_components_are_multiplied_by_total_energy(self):
        path = Path(__file__).resolve().parent / (
            "test_pair_" + uuid.uuid4().hex + ".dat"
        )
        references = {
            side: [
                {
                    "log10_theta_low": -2,
                    "log10_theta_high": 0,
                    "log10_pT_low": -4,
                    "log10_pT_high": -3.25,
                },
                {
                    "log10_theta_low": -2,
                    "log10_theta_high": 0,
                    "log10_pT_low": -3.25,
                    "log10_pT_high": -3.2,
                },
            ]
            for side in ("forward", "backward")
        }
        try:
            # pT=0.001*0.6=0.0006 GeV belongs above 10^-3.25 GeV.
            path.write_text("0.001 0.6 0 0.6155 0 0 1 0\n")
            arrays, metadata = reduce_population(path, references)
            self.assertEqual(arrays["forward_electron"][0, 1], 1)
            self.assertEqual(arrays["forward_electron"][0, 0], 0)
            self.assertEqual(metadata["native_rows"], 1)
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
