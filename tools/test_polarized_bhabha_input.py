"""Physical invariants of the manufactured polarized transport inputs."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from prepare_imported_spin_study import prepare
from generate_polarized_bhabha_events import generate


class PolarizedInputTest(unittest.TestCase):
    def test_back_to_back_spin_and_refinement_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "campaign"
            prepare(destination, events=64)
            rows = [list(map(float, row.split())) for row in
                    (destination / "field_on/event.ini").read_text().splitlines()]
            for electron, positron in zip(rows[::2], rows[1::2]):
                self.assertEqual(electron[1], -positron[1])
                self.assertEqual(electron[5:8], [-v for v in positron[5:8]])
                for track in (electron, positron):
                    self.assertAlmostEqual(sum(s*s for s in track[8:11]), 1, places=14)
                    speed = math.sqrt(sum(v*v for v in track[5:8]))
                    helicity = sum(s*v for s, v in zip(track[8:11], track[5:8])) / speed
                    self.assertAlmostEqual(abs(helicity), 1, places=14)
            self.assertEqual((destination / "field_on/event.ini").read_bytes(),
                             (destination / "field_off/event.ini").read_bytes())
            identity = json.loads((destination / "time_refined/polarized_event_identity.json").read_text())
            self.assertEqual(identity["admission_counter"], 480)
            self.assertEqual(identity["tracks"], 128)

    def test_invalid_physical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            for energy, theta, counter in ((float("nan"), 1, 240),
                                           (125, float("inf"), 240), (125, 1, -1)):
                with self.assertRaises(ValueError):
                    generate(Path(temporary) / "bad", 2, energy, theta, (.8, .3), 10, counter)


if __name__ == "__main__":
    unittest.main()
