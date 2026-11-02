#!/usr/bin/env python3
"""Focused semantic and corruption tests for the native secondary adapter."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "tools" / "collider_secondary_kinematics.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("secondary", SOURCE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def summary(n=2, **overrides):
    flags = dict(do_pairs=1, load_events=0, track_pairs=1, store_pairs=1, pair_ratio=1,
                 pair_ecut=0.0005, beam_pair=0, do_comp=0, do_prod=0, do_bhabhas=0)
    flags.update(overrides)
    return "\n".join(f"{k} = {v}" for k, v in flags.items()) + f"\nn_pairs = {n}\nn_pairs={n};e_pairs=4;\n"


ROWS = "2 .3 .4 .866025403784 0 0 0 0 -1 -1\n-2 .3 .4 -.866025403784 0 0 0 1 -1 -1\n"


class SecondaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_case(self, rows=ROWS, text=None, **flags):
        (self.root / "out.dat").write_text(text if text is not None else summary(**flags))
        (self.root / "pairs.dat").write_text(rows)
        return adapter.analyze_run(self.root)

    def test_analytic_momentum_and_global_folded_angle(self):
        data = self.run_case()
        p = data["populations"]["pairs"]
        self.assertTrue(p["available"])
        self.assertAlmostEqual(p["distributions"]["pT"]["statistics"]["mean"], 1)
        self.assertAlmostEqual(p["distributions"]["theta"]["statistics"]["mean"], math.pi / 2)
        self.assertAlmostEqual(p["distributions"]["theta_folded"]["statistics"]["mean"], math.pi / 6, places=10)
        self.assertEqual(p["joint"]["in_range_records"], 2)
        self.assertEqual(p["joint_folded"]["in_range_records"], 2)
        self.assertEqual(sum(x != 0 for x in p["joint"]["counts"]), 2)
        self.assertEqual(sum(x != 0 for x in p["joint_folded"]["counts"]), 1)
        self.assertTrue(data["record_count_verification"]["available"])

    def test_zeros_accounted_without_logarithm_or_epsilon(self):
        data = self.run_case(rows="2 0 0 1 0 0 0 0 -1 -1\n-2 0 0 -1 0 0 0 1 -1 -1\n")
        p = data["populations"]["pairs"]
        ex = p["joint"]["excluded"]
        self.assertEqual(ex["theta_nonpositive"], 1)
        self.assertEqual(ex["pT_nonpositive"], 2)
        self.assertEqual(ex["both_nonpositive"], 1)
        self.assertEqual(ex["nonpositive_records"], 2)
        self.assertEqual(p["joint"]["in_range_records"], 0)
        self.assertEqual(p["joint_folded"]["excluded"]["theta_nonpositive"], 2)
        self.assertEqual(p["distributions"]["pT"]["total_weight"], 2)

    def test_beam_pair_records_are_combined_not_separated(self):
        data = self.run_case(beam_pair=1)
        self.assertTrue(data["populations"]["combined"]["available"])
        self.assertFalse(data["populations"]["pairs"]["available"])
        self.assertFalse(data["populations"]["deflected"]["available"])
        self.assertFalse(data["record_count_verification"]["available"])

    def test_imported_records_identified_as_deflected(self):
        data = self.run_case(rows=ROWS.replace(" 0 -1 -1", " -99 -1 -1").replace(" 1 -1 -1", " -99 -1 -1"),
                             do_pairs=0, load_events=1, do_comp=1)
        self.assertTrue(data["populations"]["deflected"]["available"])
        self.assertEqual(data["populations"]["deflected"]["selection"]["process_counts"], {"-99": 2})

    def test_imported_plus_pairs_is_combined(self):
        data = self.run_case(load_events=1)
        self.assertTrue(data["populations"]["combined"]["available"])

    def test_imported_input_bytes_are_authenticated_without_interpreting_payload(self):
        raw = b"deliberately incomplete input\xff\n"
        (self.root / "event.ini").write_bytes(raw)
        data = self.run_case(do_pairs=0, load_events=1)
        ref = data["source_artifacts"]["imported_input"]
        self.assertEqual(ref, dict(path="event.ini", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        self.assertTrue(data["populations"]["deflected"]["available"])
        self.assertIn("not parsed", data["definitions"]["imported_input"])
        (self.root / "event.ini").write_bytes(raw[:-3])
        changed = self.run_case(do_pairs=0, load_events=1)
        self.assertNotEqual(changed["source_artifacts"]["imported_input"]["sha256"], ref["sha256"])
        self.assertEqual(changed["populations"], data["populations"])

    def test_missing_imported_input_is_explicit_and_unused_input_is_not_bound(self):
        data = self.run_case(do_pairs=0, load_events=1)
        self.assertIsNone(data["source_artifacts"]["imported_input"])
        self.assertIn("not retained", data["definitions"]["imported_input"])
        (self.root / "event.ini").write_text("unused")
        data = self.run_case()
        self.assertIsNone(data["source_artifacts"]["imported_input"])

    def test_imported_input_symlink_rejected(self):
        target = self.root / "external.dat"
        target.write_text("input")
        (self.root / "event.ini").symlink_to(target)
        with self.assertRaises(ValueError):
            self.run_case(do_pairs=0, load_events=1)

    def test_bhabha_requires_evaluated_flag_and_event_index(self):
        rows = "\n".join("42 " + row for row in ROWS.splitlines()) + "\n"
        data = self.run_case(rows=rows, do_pairs=0, do_bhabhas=1)
        self.assertTrue(data["populations"]["deflected"]["available"])
        with self.assertRaisesRegex(ValueError, "column count"):
            self.run_case(rows=rows)
        with self.assertRaisesRegex(ValueError, "column count"):
            self.run_case(do_pairs=0, do_bhabhas=1)

    def test_legacy_pair_summary_admitted_but_unknown_nonpair_layout_unavailable(self):
        data = self.run_case(text=summary().replace("do_bhabhas = 0\n", ""))
        self.assertTrue(data["populations"]["pairs"]["available"])
        data = self.run_case(text=summary(do_pairs=0, load_events=1).replace("do_bhabhas = 0\n", ""))
        self.assertFalse(any(p["available"] for p in data["populations"].values()))

    def test_thinning_is_not_reweighted(self):
        data = self.run_case(pair_ratio=.25)
        self.assertEqual(data["populations"]["pairs"]["joint"]["samples"], 2)
        self.assertFalse(data["record_count_verification"]["available"])
        self.assertFalse(data["populations"]["pairs"]["normalization"]["physical_weight_available"])

    def test_whole_row_truncation_rejected_with_native_count(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.run_case(rows=ROWS.splitlines()[0] + "\n")

    def test_wrong_columns_and_nonfinite_rejected(self):
        for rows in (ROWS.replace("0 0 0", "0 0", 1), ROWS.replace(".3", "nan", 1), ROWS.replace(".3", "inf", 1)):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.run_case(rows=rows)

    def test_bad_velocity_or_undefined_direction_rejected(self):
        for velocities in ("0 0 0", "2 0 0"):
            with self.assertRaises(ValueError):
                self.run_case(rows="2 " + velocities + " 0 0 0 0 -1 -1\n" + ROWS.splitlines()[1] + "\n")

    def test_missing_duplicate_and_nonfinite_flags_rejected(self):
        for text in (summary().replace("beam_pair = 0\n", ""), summary() + "\nbeam_pair = 1\n", summary().replace("pair_ratio = 1", "pair_ratio = nan")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.run_case(text=text)

    def test_process_contamination_rejected_for_pure_pairs(self):
        with self.assertRaisesRegex(ValueError, "unexpected process"):
            self.run_case(rows=ROWS.replace(" 0 -1 -1", " -99 -1 -1"))

    def test_disabled_tracking_and_empty_output_have_no_plot(self):
        disabled = self.run_case(track_pairs=0)
        self.assertFalse(any(p["available"] for p in disabled["populations"].values()))
        empty = self.run_case(rows="", text=summary(n=0))
        self.assertFalse(any(p["available"] for p in empty["populations"].values()))
        self.assertTrue(empty["record_count_verification"]["available"])

    def test_histogram_counts_and_source_signs_close(self):
        data = self.run_case()
        p = data["populations"]["pairs"]
        for h in p["distributions"].values():
            self.assertEqual(sum(b["weight"] for b in h["bins"]) + h["underflow"] + h["overflow"], 2)
        self.assertEqual(p["selection"]["signed_energy_counts"], {"positive_energy": 1, "negative_energy": 1})

    def test_joint_corruption_and_duplicate_population_rejected(self):
        data = self.run_case()
        mutated = copy.deepcopy(data)
        mutated["populations"]["pairs"]["joint"]["counts"][0] += 1
        with self.assertRaisesRegex(ValueError, "count does not close"):
            adapter.validate(mutated)
        mutated = copy.deepcopy(data)
        mutated["populations"]["deflected"] = copy.deepcopy(mutated["populations"]["pairs"])
        with self.assertRaises(ValueError):
            adapter.validate(mutated)

    def test_forged_mode_normalization_and_completeness_rejected(self):
        data = self.run_case()
        cases = []
        altered = copy.deepcopy(data)
        altered["evaluated_flags"]["beam_pair"] = 1
        cases.append(altered)
        altered = copy.deepcopy(data)
        altered["populations"]["pairs"]["normalization"]["physical_weight_available"] = True
        cases.append(altered)
        altered = copy.deepcopy(data)
        altered["record_count_verification"]["expected_records"] = 3
        cases.append(altered)
        for altered in cases:
            with self.assertRaises(ValueError):
                adapter.validate(altered)

    def test_exact_top_edge_is_included(self):
        self.assertEqual(adapter._bin([0, 1, 2], 2), 1)
        self.assertEqual(adapter._bin([0, 1, 2], 2.1), 2)

    def test_written_sidecar_round_trip_and_source_hash(self):
        data = self.run_case()
        path = adapter.write_analysis(self.root, data)
        self.assertEqual(adapter.validate(json.loads(path.read_text())), data)
        self.assertEqual(data["source_artifacts"]["pairs"]["bytes"], len(ROWS.encode()))

    def test_unsafe_paths_and_symlinks_rejected(self):
        self.run_case()
        with self.assertRaises(ValueError):
            adapter.analyze_run(self.root, "../out.dat")
        path = self.root / "pairs.dat"
        target = self.root / "real.dat"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            adapter.analyze_run(self.root)


if __name__ == "__main__":
    unittest.main()
