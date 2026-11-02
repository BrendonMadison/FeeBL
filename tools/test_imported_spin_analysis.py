"""Manufactured rotation, identity and uncertainty checks without native runs."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from analyze_imported_spin import analyze, estimate, read_input, relative_change


def fixture(angle=0):
    tracks = []
    for source, (pdg, helicity) in enumerate([(11, 1), (-11, 1), (11, -1), (-11, 1)], 1):
        direction = 1 if pdg == 11 else -1
        initial = dict(signed_energy_GeV=direction*125, position_nm=[0, 0, 0],
                       beta=[0, 0, direction], spin=[0, 0, direction*helicity], helicity=helicity)
        final = copy.deepcopy(initial)
        final["spin"] = [direction*helicity*math.sin(angle), 0, direction*helicity*math.cos(angle)]
        final["helicity"] = helicity*math.cos(angle)
        tracks.append(dict(label=source, source_record=source, pdg_id=pdg,
                           initial=initial, final=final, max_chi=0))
    return dict(schema_version="feebl.imported-spin/1", mode=1, radiative_spin_flips=False, tracks=tracks)


class ImportedSpinAnalysisTests(unittest.TestCase):
    def test_gzip_replay_preserves_native_identity(self):
        native = json.dumps(fixture(), indent=2).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            compressed = folder / "imported_spin.json.gz"
            compressed.write_bytes(gzip.compress(native, mtime=0))
            payload, identity = read_input(folder)
            self.assertEqual(payload, fixture())
            self.assertEqual(identity["native_sha256"], hashlib.sha256(native).hexdigest())
            self.assertEqual(identity["native_bytes"], len(native))
            self.assertEqual(identity["compression"], "gzip")
            original = folder / "imported_spin.json"
            original.write_bytes(native)
            plain, plain_identity = read_input(folder)
            self.assertEqual(plain, payload)
            self.assertEqual(plain_identity["native_sha256"], identity["native_sha256"])
            self.assertEqual(plain_identity["compression"], "none")

    def test_corrupt_gzip_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imported_spin.json.gz"
            path.write_bytes(b"not a gzip stream")
            with self.assertRaises(gzip.BadGzipFile):
                read_input(path)

    def test_known_rotation_and_control(self):
        theta = 1e-4
        result, arrays = analyze(dict(field_on=fixture(theta), field_off=fixture(), refined=fixture(theta/2)))
        electron = result["species"]["electron"]
        self.assertAlmostEqual(electron["runs"]["field_on"]["retention_loss"]["mean"], 2*math.sin(theta/2)**2, places=20)
        self.assertEqual(electron["runs"]["field_off"]["delta_P_pp"]["mean"], 0)
        self.assertEqual(electron["runs"]["field_on"]["delta_P_pp"]["mean"], 0)
        self.assertGreater(electron["runs"]["field_on"]["delta_P_pp"]["sem"], 0)
        self.assertIsNone(electron["runs"]["field_on"]["relative_delta_percent"]["mean"])
        positron = result["species"]["positron"]
        self.assertAlmostEqual(positron["paired_refinement_delta_P_pp"]["mean"],
                               100*(math.cos(theta/2)-math.cos(theta)), places=13)
        self.assertEqual(len(arrays["electron"]["refinement"]), 2)

    def test_small_rotation_retention_avoids_subtraction(self):
        result, _ = analyze(dict(field_on=fixture(1e-10), field_off=fixture(), refined=fixture(1e-10)))
        self.assertAlmostEqual(result["species"]["electron"]["runs"]["field_on"]["retention_loss"]["mean"], 5e-21, places=35)

    def test_source_order_is_not_identity_and_counters_may_change(self):
        off = fixture()
        off["tracks"].reverse()
        off["tracks"][0]["admitted_time_counter"] = 987
        result, _ = analyze(dict(field_on=fixture(), field_off=off, refined=fixture()))
        self.assertEqual(result["species"]["positron"]["matched_events"], 2)

    def test_missing_and_changed_initial_identity_fail(self):
        off = fixture()
        off["tracks"].pop()
        with self.assertRaisesRegex(ValueError, "population"):
            analyze(dict(field_on=fixture(), field_off=off, refined=fixture()))
        off = fixture()
        off["tracks"][0]["initial"]["position_nm"][0] = 1
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            analyze(dict(field_on=fixture(), field_off=off, refined=fixture()))

    def test_nonfinite_and_wrong_helicity_fail(self):
        for value in (float("nan"), .4):
            bad = fixture()
            bad["tracks"][0]["final"]["helicity"] = value
            with self.assertRaises(ValueError):
                analyze(dict(field_on=bad, field_off=fixture(), refined=fixture()))

    def test_sem_and_paired_ratio(self):
        self.assertEqual(estimate([1, 3]), dict(mean=2, sem=1, n=2))
        result = relative_change([1, 1, -1], [-.1, -.1, .1])
        self.assertAlmostEqual(result["mean"], -10)
        self.assertAlmostEqual(result["sem"], 0)


if __name__ == "__main__":
    unittest.main()
