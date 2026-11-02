"""Full-source snapshot schema/geometry tests; synthetic data only."""

import copy, json, unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import collider_fields as c
import finalize_gui_dataset as finalize

FIXTURE = json.loads((Path(__file__).parent / "testdata/collider_snapshot_fields.json").read_text())


class SnapshotFieldTests(unittest.TestCase):
    def setUp(self):
        self.d = copy.deepcopy(FIXTURE["fields"])
        self.m = copy.deepcopy(FIXTURE["manifest"]["frames"])

    def test_all_source_slices_each_frame(self):
        r = c.validate(self.d, self.m)["snapshot"]
        self.assertEqual(r["available_frames"], 2)
        self.assertEqual(r["available_strips"], 12)
        self.assertEqual(r["scalar_count"], 48)
        self.assertEqual(r["source_maxima"]["beam1"]["zx"], 113)
        self.assertEqual(r["source_maxima"]["beam2"]["zy"], 213)

    def test_old_encounter_data_accepted(self):
        self.d.pop("snapshot")
        for f in self.d["frames"]:
            f.pop("snapshot")
        self.assertIsNone(c.validate(self.d, self.m)["snapshot"])

    def test_whole_frame_unavailable(self):
        self.d["frames"][0]["snapshot"].update(
            available=False, reason="unavailable reconstruction", sources=None
        )
        self.assertEqual(c.validate(self.d, self.m)["snapshot"]["available_frames"], 1)

    def test_zero_deposited_slice_is_computed_zero(self):
        self.assertEqual(
            self.d["frames"][0]["snapshot"]["sources"]["beam1"]["strips"][0]["maps"]["zx"], [0, 0]
        )
        c.validate(self.d, self.m)

    def test_empty_source_solver_roundoff_is_preserved(self):
        self.d["frames"][0]["snapshot"]["sources"]["beam1"]["strips"][0]["maps"]["zx"][0] = 1e-25
        c.validate(self.d, self.m)
        self.assertEqual(
            self.d["frames"][0]["snapshot"]["sources"]["beam1"]["strips"][0]["maps"]["zx"][0], 1e-25
        )

    def test_unsupported_nz_explicit(self):
        self.d.pop("longitudinal")
        self.d["conventions"]["n_z"] = 129
        self.d["snapshot"]["n_z"] = 129
        for f in self.d["frames"]:
            f.pop("longitudinal")
            f["snapshot"].update(
                available=False, reason="slice bound", expected_slices=129, sources=None
            )
        self.assertEqual(c.validate(self.d, self.m)["snapshot"]["available_frames"], 0)

    def test_combined_scalar_bound(self):
        with patch.object(c, "MAX_VALUES", 80):
            with self.assertRaisesRegex(ValueError, "combined"):
                c.validate(self.d, self.m)

    def test_invalid_metadata_and_frames(self):
        cases = [
            lambda d: d.pop("snapshot"),
            lambda d: d["snapshot"].update(schema_version="old"),
            lambda d: d["snapshot"].update(frame="lab"),
            lambda d: d["snapshot"].update(n_z=2),
            lambda d: d["snapshot"].update(delta_z_nm=0),
            lambda d: d["snapshot"].update(model="full physical E"),
            lambda d: d["snapshot"].update(timing="pre-kick"),
            lambda d: d["snapshot"]["planes"]["zx"].update(samples=65),
            lambda d: d["snapshot"]["planes"]["zx"].update(fixed_coordinate_nm=1),
            lambda d: d["snapshot"]["planes"]["zx"].update(min_nm=-999),
            lambda d: d["frames"][0].pop("snapshot"),
            lambda d: d["frames"][0]["snapshot"].update(diagonal=1),
            lambda d: d["frames"][0]["snapshot"].update(phase="pre-kick"),
            lambda d: d["frames"][0]["snapshot"].update(expected_slices=1),
            lambda d: d["frames"][0]["snapshot"].update(available=False, reason="no data"),
            lambda d: d["frames"][0]["snapshot"].update(sources=None),
            lambda d: d["frames"][0]["snapshot"]["sources"]["beam1"]["strips"].pop(),
        ]
        for i, mutate in enumerate(cases):
            with self.subTest(i=i):
                d = copy.deepcopy(self.d)
                mutate(d)
                with self.assertRaises((ValueError, KeyError)):
                    c.validate(d, self.m)

    def test_invalid_strip_values(self):
        cases = [
            lambda s: s.update(slice=0),
            lambda s: s.update(display_z_nm=0),
            lambda s: s.update(display_z_bounds_nm=[-1, 1]),
            lambda s: s.update(source_particles=12),
            lambda s: s.update(outside_particles=-1),
            lambda s: s.update(deposited_particles=True),
            lambda s: s["maps"]["zx"].pop(),
            lambda s: s["maps"]["zx"].__setitem__(0, -1),
            lambda s: s["maps"]["zx"].__setitem__(0, float("nan")),
            lambda s: s["maps"]["zx"].__setitem__(0, True),
            lambda s: s["maps"].pop("zy"),
        ]
        for i, mutate in enumerate(cases):
            with self.subTest(i=i):
                d = copy.deepcopy(self.d)
                mutate(d["frames"][0]["snapshot"]["sources"]["beam1"]["strips"][1])
                with self.assertRaises((ValueError, KeyError)):
                    c.validate(d, self.m)

    def test_field_specific_file_and_aggregate_bounds(self):
        # File-content reads and hashes are mocked; byte budget remains the real verifier.
        for field_bytes, analysis_bytes, accepted in [
            (40 * 1024**2, 0, True),
            (65 * 1024**2, 0, False),
            (40 * 1024**2, 33 * 1024**2, False),
            (64 * 1024**2, 32 * 1024**2, False),
        ]:
            with self.subTest(field_bytes=field_bytes, analysis_bytes=analysis_bytes):
                arts = {}
                frames = copy.deepcopy(self.m)
                for i, f in enumerate(frames):
                    for b in (1, 2):
                        name = f"b{b}.{i}"
                        a = dict(
                            path=name, bytes=10 * 1024**2 if analysis_bytes == 32 * 1024**2 else 1
                        )
                        arts[name] = a
                        f["beam" + str(b)] = a
                field = dict(path="gui_fields.json", bytes=field_bytes, sha256="verified")
                man = dict(
                    schema_version=finalize.SCHEMA,
                    units=dict(energy="GeV", position="um", angle="urad", time="s"),
                    coordinates=dict(frame="legacy_collider_dump"),
                    frames=frames,
                    fields=field,
                )
                if analysis_bytes:
                    man["analysis"] = dict(
                        path="analysis.json", bytes=analysis_bytes, sha256="verified"
                    )

                def safe(root, name):
                    return SimpleNamespace(
                        stat=lambda: SimpleNamespace(
                            st_size=field_bytes if name == "gui_fields.json" else analysis_bytes
                        ),
                        read_text=lambda: json.dumps(self.d),
                    )

                with (
                    patch.object(
                        finalize, "describe_frame", side_effect=lambda root, name: arts[name]
                    ),
                    patch.object(finalize, "safe_path", side_effect=safe),
                    patch.object(finalize, "digest", return_value="verified"),
                ):
                    if accepted:
                        self.assertEqual(
                            finalize.verify(Path("."), man)["status"], "PASS_FOR_STATED_TARGET"
                        )
                    else:
                        with self.assertRaises(ValueError):
                            finalize.verify(Path("."), man)


if __name__ == "__main__":
    unittest.main()
