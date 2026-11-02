import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import collider_monochromatization as mono


SUMMARY = """------------- beam parameter 1 ------------
energy : 125 GeV ; particles : 2e10
sigma_x : 500 nm ; sigma_y : 5 nm
angle_x : 0 angle_y : 0 angle_phi : 0
dist_x : 0 dist_z : 0
------------- beam parameter 2 ------------
energy : 125 GeV ; particles : 2e10
sigma_x : 500 nm ; sigma_y : 5 nm
angle_x : 0 angle_y : 0 angle_phi : 0
dist_x : 0 dist_z : 0
------------- SWITCHES : ------------
do_espread = 1
espread.1 = .001 which_espread.1 = 3 espread.2 = .001 which_espread.2 = 3
------------- grid parameters ------------
"""
MANIFEST = dict(
    energy_GeV=[125, 125],
    espread=[0.001, 0.001],
    ip_dispersion_m=[0.001, -0.001],
    native_beam_angles_x_rad=[0, 0],
    native_beam_angles_y_rad=[0, 0],
    crab_mode=[0, 0],
    waist_mode=[0, 0],
    frames=[dict(id=0, diagonal=0), dict(id=1, diagonal=2)],
)


def fixture(root):
    (root / "out.dat").write_text(SUMMARY)
    (root / "collider_analysis.json").write_text(
        json.dumps(
            dict(
                schema_version="feebl.collider.analysis/1",
                native_statistics=dict(full_RMS_W_GeV=0.25),
            )
        )
    )
    frames = []
    for i, (first, last) in enumerate([(0, 0), (1, 2)]):
        frames.append(
            dict(
                id=i,
                diagonal_start=first,
                diagonal_end=last,
                luminosity_spectrum=dict(
                    bins=[dict(low=249, high=249.1, weight=1), dict(low=249.9, high=250, weight=1)],
                    underflow=0,
                    overflow=0,
                    total_weight=2,
                ),
            )
        )
    (root / "gui_diagnostics.json").write_text(
        json.dumps(dict(schema_version="feebl.collider.diagnostics/1", frames=frames))
    )


class MonochromatizationTests(unittest.TestCase):
    def inputs(self):
        return mono.parse_inputs(SUMMARY, MANIFEST)[0]

    def test_equal_opposite_gaussian_oracle(self):
        p = self.inputs()
        r = mono.ideal_reference(p)
        self.assertTrue(r["available"])
        self.assertAlmostEqual(r["value"], math.sqrt(5), 14)

    def test_equal_same_sign_null(self):
        p = self.inputs()
        p["ip_dispersion_m"] = [0.001, 0.001]
        self.assertAlmostEqual(mono.ideal_reference(p)["value"], 1, 14)

    def test_zero_dispersion_is_reference_even_for_actual_crossing(self):
        p = self.inputs()
        p["ip_dispersion_m"] = [0, 0]
        p["native_beam_angles_x_rad"] = [0.007, 0.007]
        p["crab_mode"] = [1, 1]
        self.assertEqual(mono.ideal_reference(p)["value"], 1)

    def test_unequal_independent_covariance_oracle(self):
        p = self.inputs()
        p["requested_espread"] = [0.001, 0.002]
        p["sigma_delta"] = [0.001, 0.002]
        p["sigma_beta_x_m"] = [4e-7, 7e-7]
        p["ip_dispersion_m"] = [0.0008, 0.0013]
        s1, s2 = [x * x for x in p["sigma_delta"]]
        b = sum(x * x for x in p["sigma_beta_x_m"])
        d1, d2 = p["ip_dispersion_m"]
        conditioned = s1 + s2 - (d1 * s1 - d2 * s2) ** 2 / (b + d1 * d1 * s1 + d2 * d2 * s2)
        self.assertAlmostEqual(
            mono.ideal_reference(p)["value"], math.sqrt((s1 + s2) / conditioned), 13
        )
        base = mono.ideal_reference(p)["value"]
        p["ip_dispersion_m"] = [-d1, -d2]
        self.assertEqual(mono.ideal_reference(p)["value"], base)
        for k in p:
            if isinstance(p[k], list):
                p[k].reverse()
        self.assertAlmostEqual(mono.ideal_reference(p)["value"], base, 14)

    def test_geometry_and_distribution_admission(self):
        for key in [
            "native_beam_angles_x_rad",
            "native_beam_angles_y_rad",
            "crab_mode",
            "waist_mode",
            "dist_x",
        ]:
            p = self.inputs()
            p[key][0] = 1
            self.assertFalse(mono.ideal_reference(p)["available"])
        for which in (1, 2):
            p = self.inputs()
            p["which_espread"][0] = which
            if which == 1:
                p["sigma_delta"][0] *= 1.7 / math.sqrt(3)
            self.assertFalse(mono.ideal_reference(p)["available"])

    def test_spread_is_actual_distribution_reference(self):
        text = SUMMARY.replace("which_espread.1 = 3", "which_espread.1 = 1")
        p = mono.parse_inputs(text, MANIFEST)[0]
        self.assertAlmostEqual(p["sigma_delta"][0], 0.001 * 1.7 / math.sqrt(3))
        p = mono.parse_inputs(SUMMARY.replace("do_espread = 1", "do_espread = 0"), MANIFEST)[0]
        self.assertEqual(p["sigma_delta"], [0, 0])
        self.assertEqual(mono.ideal_reference(p)["value"], 1)

    def test_invalid_or_ambiguous_native_inputs(self):
        for text in [
            SUMMARY.replace("500 nm", "500*2 nm", 1),
            SUMMARY.replace("sigma_x : 500 nm", "sigma_x : NaN nm", 1),
            SUMMARY.replace("grid parameters", "espread.1=.2\ngrid parameters"),
            SUMMARY.replace("beam parameter 2", "beam parameter 1"),
        ]:
            with self.assertRaises(ValueError):
                mono.parse_inputs(text, MANIFEST)
        bad = copy.deepcopy(MANIFEST)
        bad["energy_GeV"][0] = 100
        with self.assertRaises(ValueError):
            mono.parse_inputs(SUMMARY, bad)
        p = self.inputs()
        p["sigma_delta"][0] = math.nan
        with self.assertRaises(ValueError):
            mono.ideal_reference(p)

    def test_frame_estimator_weighting_and_resolution(self):
        r = dict(
            id=0,
            diagonal_start=0,
            diagonal_end=0,
            luminosity_spectrum=dict(
                bins=[dict(low=10, high=10.1, weight=1), dict(low=12, high=12.1, weight=3)],
                underflow=0,
                overflow=0,
                total_weight=4,
            ),
        )
        out = mono.frame_width(r, 1)
        self.assertTrue(out["available"])
        self.assertAlmostEqual(out["sigma_W_GeV"], math.sqrt(0.75), 14)
        for change in ("flow", "zero", "single", "unresolved"):
            q = copy.deepcopy(r)
            h = q["luminosity_spectrum"]
            if change == "flow":
                h["underflow"] = 1
                h["total_weight"] = 5
            if change == "zero":
                h["bins"] = []
                h["total_weight"] = 0
            if change == "single":
                h["bins"] = h["bins"][:1]
                h["total_weight"] = 1
            if change == "unresolved":
                h["bins"] = [dict(low=10, high=11, weight=1), dict(low=11, high=12, weight=1)]
                h["total_weight"] = 2
            self.assertFalse(mono.frame_width(q, 1)["available"], change)

    def test_ratio_not_mean_of_frame_ratios_and_missing_control(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            fixture(p)
            a = mono.analyze_run(p, MANIFEST)
            self.assertAlmostEqual(
                a["measured"]["reference_sigma_W_GeV"], 125 * math.sqrt(2) * 0.001
            )
            self.assertAlmostEqual(
                a["measured"]["total"]["value"], 125 * math.sqrt(2) * 0.001 / 0.25
            )
            self.assertNotEqual(
                a["measured"]["total"]["value"], a["measured"]["frames"][0]["value"]
            )
            self.assertFalse(a["matched_control"]["available"])
            self.assertIsNone(a["matched_control"]["value"])
            before = {x.name: x.read_bytes() for x in p.iterdir()}
            mono.write_analysis(p, a)
            self.assertTrue(all((p / n).read_bytes() == v for n, v in before.items()))
            self.assertEqual(json.loads((p / "collider_monochromatization.json").read_text()), a)

    def test_authentication_and_schema_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            fixture(p)
            a = mono.analyze_run(p, MANIFEST)
            for mutate in [
                lambda q: q["ideal"].update(value=99),
                lambda q: q["measured"]["total"].update(value=99),
                lambda q: q["measured"]["total"].update(value=float("nan")),
                lambda q: q["matched_control"].update(value=1),
                lambda q: q["inputs"]["sigma_delta"].__setitem__(0, 0.2),
                lambda q: q["source_artifacts"]["native_summary"].update(path="../out.dat"),
            ]:
                q = copy.deepcopy(a)
                mutate(q)
                with self.assertRaises(ValueError):
                    mono.validate(q)
            bad = copy.deepcopy(MANIFEST)
            bad["analysis"] = dict(path="collider_analysis.json", bytes=1, sha256="0" * 64)
            with self.assertRaises(ValueError):
                mono.analyze_run(p, bad)

    def test_no_width_never_produces_infinity_or_fake_gain(self):
        for reference, sigma in [(0, 1), (1, 0), (1, None)]:
            r = mono.width_ratio(reference, sigma, mono.TOTAL_ESTIMATOR)
            self.assertFalse(r["available"])
            self.assertIsNone(r["value"])

    def test_symlinked_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            fixture(p)
            a = mono.analyze_run(p, MANIFEST)
            target = p / "preserve.txt"
            target.write_text("preserve")
            try:
                (p / "collider_monochromatization.json").symlink_to(target)
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    self.skipTest("Windows account lacks symbolic-link privilege")
                raise
            with self.assertRaises(ValueError):
                mono.write_analysis(p, a)
            self.assertEqual(target.read_text(), "preserve")

    def test_coupled_spread_ratio_change_cannot_replace_native_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            fixture(p)
            a = mono.analyze_run(p, MANIFEST)
            a["inputs"]["sigma_delta"] = [v * 2 for v in a["inputs"]["sigma_delta"]]
            a["inputs"]["requested_espread"] = [v * 2 for v in a["inputs"]["requested_espread"]]
            a["ideal"].update(mono.ideal_reference(a["inputs"]))
            a["measured"]["reference_sigma_W_GeV"] *= 2
            for record in [a["measured"]["total"], *a["measured"]["frames"]]:
                if record["available"]:
                    record["value"] *= 2
            with self.assertRaisesRegex(ValueError, "evaluated/native input mismatch: espread"):
                mono.validate(a)

    def test_all_evaluated_native_receipt_keys_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            fixture(p)
            a = mono.analyze_run(p, MANIFEST)
            for key in mono.KEYS:
                changed = json.loads(json.dumps(a))
                changed["native_manifest_parameters"][key][0] += 0.1
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, "native.*mismatch"):
                        mono.validate(changed)


if __name__ == "__main__":
    unittest.main()
