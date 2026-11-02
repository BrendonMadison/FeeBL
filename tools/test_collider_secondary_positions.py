"""Stored spatial z selection, compact native binding and legacy compatibility."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collider_secondary_positions as positions
import collider_secondary_frames as frames
import collider_secondary_kinematics as adapter
import finalize_gui_dataset as finalizer
from test_collider_secondary_frames import FLAGS, install, raw_fixture, native_frame
from test_diagnostic_analysis_binding import ref


def add_positions(raw, zframes):
    raw["position_z_definition"] = dict(positions.DEFINITION)
    for frame, zs in zip(raw["frames"], zframes):
        if not frame["available"]:
            frame["position_z"] = None
            continue
        # Synthetic fixtures use theta/pT pairs supplied alongside z. Build
        # subgroup maps independently through the pre-existing fixture builder.
        values = zs
        result = dict(zero_records=sum(v[3] == 0 for v in values))
        for name, sign in [("positive", 1), ("negative", -1)]:
            subset = native_frame(0, [v[:3] for v in values if v[3] * sign > 0], raw["axes"])
            subset = {
                k: subset[k]
                for k in ("nrecords", "joint", "joint_folded", "marginals", "selection")
            }
            for key in ("joint", "joint_folded"):
                counts = subset[key].pop("counts")
                subset[key]["counts_sparse"] = [[i, n] for i, n in enumerate(counts) if n]
            for key in ("admitted_tracker_count", "authoritative_reserve_count"):
                subset["selection"].pop(key)
            result[name] = subset
        frame["position_z"] = result
    return raw


def position_fixture():
    values = [
        [(0.2, 1.0, 1, 2.0), (0.9, 2.0, -1, -3.0), (0.4, 0.5, 1, 0.0), (0.4, 0.5, -1, -0.0)],
        [(0.2, 1.0, 1, -2.0), (0.9, 2.0, -1, -3.0), (0.4, 0.5, 1, 0.0), (0.4, 0.5, -1, -0.0)],
        [],
        None,
    ]
    raw = raw_fixture([None if v is None else [p[:3] for p in v] for v in values])
    return add_positions(raw, values)


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def final(self, rows, **overrides):
        flags = dict(FLAGS, **overrides)
        (self.root / "out.dat").write_text(
            "\n".join(f"{k} = {v}" for k, v in flags.items()) + f"\nn_pairs = {len(rows)}\n"
        )
        (self.root / "pairs.dat").write_text("\n".join(rows) + "\n")
        return adapter.analyze_run(self.root)

    def test_position_not_velocity_charge_or_source_slice(self):
        data = self.final(["2 .3 .4 -.8660254 8 9 2 0 -1 -1", "-4 .3 .4 .8660254 8 9 -3 1 1 1"])
        p = data["populations"]["pairs"]["position_z"]
        self.assertEqual(
            (p["positive"]["nrecords"], p["negative"]["nrecords"], p["zero_records"]), (1, 1, 0)
        )
        self.assertEqual(p["positive"]["marginals"]["pT"]["statistics"]["mean"], 1.0)
        self.assertEqual(p["negative"]["marginals"]["pT"]["statistics"]["mean"], 2.0)
        self.assertGreater(p["positive"]["marginals"]["theta"]["statistics"]["mean"], math.pi / 2)
        self.assertLess(p["negative"]["marginals"]["theta"]["statistics"]["mean"], math.pi / 2)
        self.assertEqual(p["negative"]["selection"]["signed_energy_counts"]["negative_energy"], 1)

    def test_zero_both_signs_and_tiny_positions(self):
        data = self.final(
            [f"2 .3 .4 .8660254 0 0 {z} 0 -1 -1" for z in ("0", "-0", "1e-300", "-1e-300")]
        )
        p = data["populations"]["pairs"]["position_z"]
        self.assertEqual(p["zero_records"], 2)
        self.assertEqual(p["positive"]["nrecords"], 1)
        self.assertEqual(p["negative"]["nrecords"], 1)

    def test_empty_selected_subset_has_null_statistics(self):
        data = self.final(["2 .3 .4 .8660254 0 0 2 0 -1 -1"])
        p = data["populations"]["pairs"]["position_z"]["negative"]
        self.assertEqual(p["nrecords"], 0)
        self.assertEqual(p["joint"]["counts_sparse"], [])
        self.assertTrue(all(h["statistics"] is None for h in p["marginals"].values()))

    def test_bhabha_prefix_and_mixed_population(self):
        rows = ["47 2 .3 .4 -.8660254 0 0 2 -99 -1 -1", "48 -2 .3 .4 .8660254 0 0 -2 -99 -1 -1"]
        data = self.final(rows, do_pairs=0, do_bhabhas=1)
        self.assertEqual(
            data["populations"]["deflected"]["position_z"]["positive"]["selection"][
                "process_counts"
            ],
            {"-99": 1},
        )
        data = self.final([r.split(" ", 1)[1].replace("-99", "0") for r in rows], beam_pair=1)
        self.assertTrue(data["populations"]["combined"]["available"])
        self.assertEqual(data["populations"]["combined"]["position_z"]["negative"]["nrecords"], 1)

    def test_nonpositive_joint_exclusions_and_all_axes_retained(self):
        data = self.final(["2 0 0 1 0 0 2 0 -1 -1", "-2 0 0 -1 0 0 -2 1 -1 -1"])
        p = data["populations"]["pairs"]
        self.assertEqual(p["position_z"]["positive"]["joint"]["excluded"]["both_nonpositive"], 1)
        self.assertEqual(p["position_z"]["negative"]["joint"]["excluded"]["pT_nonpositive"], 1)
        self.assertNotIn("theta_edges_rad", p["position_z"]["positive"]["joint"])

    def test_frame_crosses_plane_without_reusing_final_state(self):
        man, raw = install(self.root, position_fixture())
        data = adapter.analyze_run(self.root, manifest=man)
        pp = [f["populations"]["pairs"] for f in data["frames"]]
        self.assertEqual([p["position_z"]["positive"]["nrecords"] for p in pp[:3]], [1, 0, 0])
        self.assertEqual(pp[0]["joint"]["counts"], pp[1]["joint"]["counts"])
        self.assertEqual(pp[0]["position_z"]["zero_records"], 2)
        self.assertTrue(pp[2]["available"])
        self.assertIsNone(pp[2]["position_z"]["positive"]["marginals"]["pT"]["statistics"])
        self.assertFalse(pp[3]["available"])
        self.assertIsNone(pp[3].get("position_z"))
        self.assertEqual(pp[0]["position_z"], raw["frames"][0]["position_z"])

    def test_frames_available_without_final_file(self):
        man, raw = install(self.root, position_fixture(), store_pairs=0)
        (self.root / "pairs.dat").unlink()
        data = adapter.analyze_run(self.root, manifest=man)
        self.assertFalse(data["populations"]["pairs"]["available"])
        self.assertTrue(data["frames"][0]["populations"]["pairs"]["available"])
        self.assertEqual(data["position_z_definition"], positions.DEFINITION)

    def test_new_final_with_old_frames_and_old_data(self):
        man, raw = install(self.root)
        data = adapter.analyze_run(self.root, manifest=man)
        self.assertIn("position_z", data["populations"]["pairs"])
        self.assertNotIn("position_z_definition", data["frame_definition"])
        self.assertNotIn("position_z", data["frames"][1]["populations"]["pairs"])
        old = copy.deepcopy(data)
        old.pop("position_z_definition")
        old["populations"]["pairs"].pop("position_z")
        adapter.validate(old, man, raw)

    def test_sparse_index_and_count_corruptions(self):
        raw = position_fixture()
        cases = [
            [[0, 1], [0, 1]],
            [[1, 1], [0, 1]],
            [[-1, 1]],
            [[16384, 1]],
            [[True, 1]],
            [[1, 0]],
            [[1, -1]],
            [[1, 1.5]],
            [[1, True]],
            [[1, 1, 2]],
        ]
        for pairs in cases:
            bad = copy.deepcopy(raw)
            bad["frames"][0]["position_z"]["positive"]["joint"]["counts_sparse"] = pairs
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                frames.validate_raw(bad)

    def test_subsets_cannot_exceed_all_or_forge_zero_receipt(self):
        raw = position_fixture()
        edits = [
            lambda p: p.__setitem__("zero_records", 0),
            lambda p: p["positive"]["joint"]["counts_sparse"][0].__setitem__(0, 0),
            lambda p: p["positive"]["selection"]["signed_energy_counts"].update(
                positive_energy=0, negative_energy=1
            ),
            lambda p: p["positive"]["selection"].__setitem__("admitted_tracker_count", 1),
        ]
        # Sign swap requires a zero-free fixture to prevent a valid residual-zero alternative.
        raw = add_positions(
            raw_fixture([[(0.2, 1.0, 1), (0.9, 2.0, -1)]]), [[(0.2, 1.0, 1, 2), (0.9, 2.0, -1, -2)]]
        )
        edits[0] = lambda p: p.__setitem__("zero_records", 1)
        for edit in edits:
            bad = copy.deepcopy(raw)
            edit(bad["frames"][0]["position_z"])
            with self.assertRaises(ValueError):
                frames.validate_raw(bad)

    def test_moment_closure_and_raw_binding(self):
        raw = add_positions(
            raw_fixture([[(0.2, 1.0, 1), (0.9, 2.0, -1)]]), [[(0.2, 1.0, 1, 2), (0.9, 2.0, -1, -2)]]
        )
        bad = copy.deepcopy(raw)
        bad["frames"][0]["position_z"]["positive"]["marginals"]["pT"]["statistics"][
            "variance"
        ] = 0.1
        with self.assertRaisesRegex(ValueError, "moment closure"):
            frames.validate_raw(bad)
        man, raw = install(self.root, position_fixture())
        data = adapter.analyze_run(self.root, manifest=man)
        data["frames"][0]["populations"]["pairs"]["position_z"]["positive"]["marginals"]["pT"][
            "statistics"
        ]["variance"] = 0.1
        with self.assertRaisesRegex(ValueError, "differs from authenticated"):
            adapter.validate(data, man, raw)

    def test_definition_or_partition_omission_rejected(self):
        raw = position_fixture()
        for edit in [
            lambda r: r.pop("position_z_definition"),
            lambda r: r["frames"][0].pop("position_z"),
            lambda r: r["frames"][3].pop("position_z"),
            lambda r: r["position_z_definition"].__setitem__("coordinate", "longitudinal momentum"),
            lambda r: r["frames"][3].__setitem__("position_z", r["frames"][0]["position_z"]),
        ]:
            bad = copy.deepcopy(raw)
            edit(bad)
            with self.assertRaises(ValueError):
                frames.validate_raw(bad)

    def test_all_flow_subset_has_null_mode_but_real_moments(self):
        raw = add_positions(raw_fixture([[(0.3, 2e6, -1)]]), [[(0.3, 2e6, -1, -2)]])
        frames.validate_raw(raw)
        stat = raw["frames"][0]["position_z"]["negative"]["marginals"]["pT"]["statistics"]
        self.assertIsNone(stat["mode"])
        self.assertEqual(stat["mean"], 2e6)

    def test_finalizer_binds_source_helper_and_preserves_bytes(self):
        man, raw = install(self.root, position_fixture())
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        finalizer.bind_diagnostic_analysis(self.root, man)
        finalizer.verify(self.root, man)
        self.assertIn(
            "collider_secondary_positions", man["diagnostic_analysis_provenance"]["adapter_sha256"]
        )
        self.assertTrue(all((self.root / name).read_bytes() == b for name, b in before.items()))

    def test_finalizer_rejects_raw_or_expanded_partition_tamper(self):
        man, raw = install(self.root, position_fixture())
        finalizer.bind_diagnostic_analysis(self.root, man)
        p = self.root / man["secondary_kinematics"]["path"]
        data = json.loads(p.read_text())
        data["frames"][0]["populations"]["pairs"]["position_z"]["positive"]["marginals"]["pT"][
            "statistics"
        ]["variance"] = 0.1
        p.write_text(json.dumps(data, separators=(",", ":")))
        man["secondary_kinematics"] = ref(p)
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, man)
        finalizer.bind_diagnostic_analysis(self.root, man)
        (self.root / "gui_secondary_frames.json").write_text(json.dumps(raw) + " ")
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, man)

    def test_bound_checks_unchanged(self):
        man, raw = install(self.root, position_fixture())
        data = adapter.analyze_run(self.root, manifest=man)
        with patch.object(adapter, "MAX_OUTPUT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "byte bound"):
                adapter.validate(data)
        with patch.object(frames, "MAX_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "byte bound"):
                adapter.analyze_run(self.root, manifest=man)


if __name__ == "__main__":
    unittest.main()
