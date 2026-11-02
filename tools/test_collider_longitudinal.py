"""Longitudinal field geometry/coverage tests; synthetic fixtures, no physics claim."""

import copy
import json
from pathlib import Path
import unittest
import collider_fields as c

FIXTURE = json.loads((Path(__file__).parent / "testdata/collider_longitudinal.json").read_text())


class LongitudinalTests(unittest.TestCase):
    def setUp(self):
        self.d = copy.deepcopy(FIXTURE["fields"])
        self.m = copy.deepcopy(FIXTURE["manifest"]["frames"])

    def test_valid(self):
        result = c.validate(self.d, self.m)["longitudinal"]
        self.assertEqual(result["available_frames"], 2)
        self.assertEqual(result["available_strips"], 4)
        self.assertEqual(result["scalar_count"], 32)
        self.assertEqual(result["source_maxima"]["beam2"]["zy"], 15)

    def test_uncaptured_gap(self):
        s = self.d["frames"][1]["longitudinal"]["strips"][1]
        s.update(
            available=False,
            reason="uncaptured",
            maps=None,
            substep_index=None,
            encounter_counter=None,
        )
        self.assertEqual(c.validate(self.d, self.m)["longitudinal"]["available_strips"], 3)

    def test_visited_unsupported_gap(self):
        self.d["frames"][1]["longitudinal"]["strips"][1].update(
            available=False, reason="unsupported dispersion", maps=None
        )
        self.assertEqual(c.validate(self.d, self.m)["longitudinal"]["available_strips"], 3)

    def test_all_unavailable(self):
        for f in self.d["frames"]:
            f["longitudinal"].update(available=False, reason="unsupported")
            for s in f["longitudinal"]["strips"]:
                s.update(available=False, reason="unsupported", maps=None)
        self.assertEqual(c.validate(self.d, self.m)["longitudinal"]["available_frames"], 0)

    def test_unsupported_slice_count(self):
        self.d["conventions"]["n_z"] = 129
        self.d["longitudinal"]["n_z"] = 129
        for f in self.d["frames"]:
            f["longitudinal"].update(available=False, reason="slice bound", strips=[])
        self.assertEqual(c.validate(self.d, self.m)["longitudinal"]["scalar_count"], 0)

    def test_zero_is_valid_data(self):
        for f in self.d["frames"]:
            for s in f["longitudinal"]["strips"]:
                for views in s["maps"].values():
                    for values in views.values():
                        values[:] = [0] * len(values)
        self.assertEqual(
            c.validate(self.d, self.m)["longitudinal"]["source_maxima"]["beam1"]["zx"], 0
        )

    def test_bad_metadata(self):
        mutations = [
            lambda d: d["longitudinal"].update(schema_version="unknown"),
            lambda d: d["longitudinal"].update(frame="lab"),
            lambda d: d["longitudinal"].update(coordinate_unit="um"),
            lambda d: d["longitudinal"].update(n_z=4),
            lambda d: d["longitudinal"].update(delta_z_nm=0),
            lambda d: d["longitudinal"].update(delta_z_nm=float("inf")),
            lambda d: d["longitudinal"].update(selection="average"),
            lambda d: d["longitudinal"].update(timing="post-kick"),
            lambda d: d["longitudinal"].update(z_mapping="common z"),
            lambda d: d["longitudinal"]["planes"]["zx"].update(samples=65),
            lambda d: d["longitudinal"]["planes"]["zx"].update(samples=True),
            lambda d: d["longitudinal"]["planes"]["zx"].update(min_nm=-999),
            lambda d: d["longitudinal"]["planes"]["zx"].update(axes=["x", "z"]),
            lambda d: d["longitudinal"]["planes"]["zx"].update(fixed_coordinate_nm=False),
            lambda d: d["longitudinal"]["planes"].pop("zy"),
            lambda d: d.pop("longitudinal"),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                d = copy.deepcopy(self.d)
                mutate(d)
                with self.assertRaises(ValueError):
                    c.validate(d, self.m)

    def test_bad_strips(self):
        mutations = [
            lambda s: s.update(beam1_slice=1),
            lambda s: s.update(nominal_z_nm=1),
            lambda s: s.update(nominal_z_bounds_nm=[-500, 501]),
            lambda s: s["display_z_nm"].update(beam1=0),
            lambda s: s["display_z_bounds_nm"].update(beam2=[-1500, 500]),
            lambda s: s.update(substep_index=0),
            lambda s: s.update(encounter_counter=-1),
            lambda s: s.update(available=1),
            lambda s: s.update(reason="unexpected"),
            lambda s: s["maps"]["beam1"]["zx"].pop(),
            lambda s: s["maps"]["beam1"]["zx"].__setitem__(0, float("nan")),
            lambda s: s["maps"]["beam1"]["zx"].__setitem__(0, -1),
            lambda s: s["maps"]["beam1"]["zx"].__setitem__(0, True),
            lambda s: s["maps"].pop("beam2"),
            lambda s: s["maps"]["beam1"].pop("zy"),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                d = copy.deepcopy(self.d)
                mutate(d["frames"][0]["longitudinal"]["strips"][0])
                with self.assertRaises(ValueError):
                    c.validate(d, self.m)

    def test_bad_coverage(self):
        changes = [
            lambda r: r["strips"].pop(),
            lambda r: r.update(expected_encounters=2),
            lambda r: r.update(available=False),
            lambda r: r["strips"][1].update(encounter_counter=r["strips"][0]["encounter_counter"]),
            lambda r: r["strips"][1].update(
                available=False, reason="missing", maps=None, substep_index=None
            ),
            lambda r: r["strips"][1].update(
                available=False, reason="missing", maps=None, encounter_counter=None
            ),
        ]
        for i, change in enumerate(changes):
            with self.subTest(i=i):
                d = copy.deepcopy(self.d)
                change(d["frames"][1]["longitudinal"])
                with self.assertRaises(ValueError):
                    c.validate(d, self.m)

    def test_budget(self):
        self.d["conventions"]["n_z"] = 128
        self.d["longitudinal"]["n_z"] = 128
        for p in self.d["longitudinal"]["planes"].values():
            p["samples"] = 32
        self.d["frames"] = [dict(diagonal=i) for i in range(96, 160)]
        with self.assertRaisesRegex(ValueError, "budget"):
            c.validate_longitudinal(self.d)


if __name__ == "__main__":
    unittest.main()
