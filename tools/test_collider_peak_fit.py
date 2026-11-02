"""Deterministic peak-fit recovery, model admission, and native source binding."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import collider_peak_fit as fit
import collider_peak_diagnostics as peak
import collider_monochromatization as mono
import finalize_gui_dataset as finalizer
from test_diagnostic_analysis_binding import fixture, ref


def histogram(mu=249.9, sigma=0.18, n=512, area=1.0, components=None, low=245.0, high=255.0):
    parts = components if components is not None else [(area, mu, sigma)]
    edges = [low + (high - low) * i / n for i in range(n + 1)]
    bins = [
        dict(
            low=a,
            high=b,
            weight=math.fsum(w * fit.gaussian_bin_probability(a, b, m, s) for w, m, s in parts),
        )
        for a, b in zip(edges, edges[1:])
    ]
    return dict(
        quantity="W",
        unit="GeV",
        weight_unit="m^-2 per BX",
        weighting="native collision pair luminosity",
        normalization="bin-integrated",
        bins=bins,
        samples=1000,
        underflow=0,
        overflow=0,
        total_weight=math.fsum(b["weight"] for b in bins),
    )


def selection(angle=0.0):
    p = math.sqrt((125.0 - peak.MASS) * (125.0 + peak.MASS))
    w = math.sqrt(2 * peak.MASS**2 + 2 * (125.0**2 + p * p * math.cos(angle)))
    return dict(
        threshold_fraction=0.98,
        nominal_W_GeV=w,
        minimum_W_GeV=0.98 * w,
        upper_cut_GeV=None,
        comparison="W >= minimum_W_GeV",
        nominal_reference=dict(
            energy_GeV=[125.0, 125.0],
            full_crossing_angle_rad=angle,
            electron_mass_GeV=peak.MASS,
            angle_source=peak.ANGLE_SOURCES[0 if angle == 0 else 1],
            definition=peak.DEFINITION,
            native_W=peak.NATIVE_W,
        ),
    )


def native_peak(h=None, polarized=True):
    h = histogram() if h is None else h
    w = h["total_weight"]
    mean = math.fsum((b["low"] + b["high"]) / 2 * b["weight"] for b in h["bins"]) / w if w else None
    variance = (
        math.fsum(((b["low"] + b["high"]) / 2 - mean) ** 2 * b["weight"] for b in h["bins"]) / w
        if w
        else None
    )
    return dict(
        available=True,
        reason=None,
        selection=selection(),
        weight_sum=w,
        contributions=h["samples"],
        spin_weight_sum=w if polarized else 0,
        spin_contributions=h["samples"] if polarized else 0,
        beam_energy_GeV={k: dict(mean=125.0, rms=0.1) for k in ("beam1", "beam2")} if w else None,
        W_GeV=dict(mean=mean, rms=math.sqrt(variance)) if w else None,
        polarization=(
            dict(weighting="native collision pair luminosity", weight_sum=w, beam1=0.8, beam2=-0.3)
            if w and polarized
            else None
        ),
        polarization_unavailable_reason=(
            None if w and polarized else "No selected helicity coverage"
        ),
        luminosity_spectrum=h,
    )


def diagnostic(p=None):
    p = native_peak() if p is None else p
    return dict(
        total=dict(diagonal_start=0, diagonal_end=0, peak=copy.deepcopy(p)),
        frames=[dict(id=0, diagonal_start=0, diagonal_end=0, peak=copy.deepcopy(p))],
    )


class FitTests(unittest.TestCase):
    def test_exact_gaussian_recovery_and_area(self):
        r = fit.fit_histogram(histogram(area=1e34), 250.0, 0.1768)
        self.assertTrue(r["available"])
        self.assertAlmostEqual(r["mean_W_GeV"], 249.9, places=7)
        self.assertAlmostEqual(r["sigma_W_GeV"], 0.18, places=7)
        self.assertLess(r["relative_l2_residual"], 1e-6)
        self.assertAlmostEqual(r["gaussian_area_native"] / 1e34, 1.0, places=7)
        self.assertAlmostEqual(r["input_reference_ratio"], 0.1768 / 0.18, places=6)

    def test_scale_and_binning_invariance(self):
        results = [
            fit.fit_histogram(histogram(n=n, area=a), 250.0)
            for n, a in [(256, 1.0), (512, 1e34), (1024, 1e-20)]
        ]
        self.assertTrue(all(r["available"] for r in results))
        self.assertLess(
            max(r["sigma_W_GeV"] for r in results) - min(r["sigma_W_GeV"] for r in results), 1e-7
        )

    def test_radiative_tail_does_not_become_full_rms(self):
        h = histogram(components=[(0.8, 249.9, 0.18), (0.2, 246.5, 1.0)])
        r = fit.fit_histogram(h, 250.0)
        self.assertTrue(r["available"])
        self.assertLess(abs(r["sigma_W_GeV"] - 0.18), 0.001)
        mean = (
            math.fsum((b["low"] + b["high"]) / 2 * b["weight"] for b in h["bins"])
            / h["total_weight"]
        )
        rms = math.sqrt(
            math.fsum(((b["low"] + b["high"]) / 2 - mean) ** 2 * b["weight"] for b in h["bins"])
            / h["total_weight"]
        )
        self.assertGreater(rms, 5 * r["sigma_W_GeV"])

    def test_two_distinct_peaks_fail_shape_admission_with_resolved_candidate(self):
        h = histogram(components=[(0.5, 248.5, 0.18), (0.5, 250.0, 0.18)])
        r = fit.fit_histogram(h, 250.0, 0.1768)
        self.assertFalse(r["available"])
        self.assertEqual(r["candidate_status"], "resolved_shape_mismatch")
        self.assertIsNone(r["input_reference_ratio"])
        self.assertGreater(r["candidate"]["input_reference_ratio"], 0)

    def test_coarse_sparse_and_empty_do_not_publish_candidate(self):
        coarse = histogram(n=32, sigma=0.03)
        sparse = histogram()
        sparse["bins"] = [
            dict(b, weight=1.0 if i in (240, 241) else 0.0) for i, b in enumerate(sparse["bins"])
        ]
        sparse["total_weight"] = 2.0
        empty = histogram(area=0.0)
        empty["samples"] = 0
        for h in (coarse, sparse, empty):
            r = fit.fit_histogram(h, 250.0)
            self.assertFalse(r["available"])
            self.assertIsNone(r["candidate"])
            self.assertFalse(r["candidate_display_allowed"])

    def test_unbracketed_and_broad_shapes_do_not_publish_candidate(self):
        for h in (histogram(mu=245.01, sigma=0.1), histogram(mu=250.0, sigma=4.0)):
            r = fit.fit_histogram(h, 250.0)
            self.assertFalse(r["available"])
            self.assertIsNone(r["candidate"])

    def test_upper_flow_rejected_but_recorded(self):
        h = histogram()
        h["overflow"] = 0.01
        h["total_weight"] += 0.01
        r = fit.fit_histogram(h, 250.0)
        self.assertFalse(r["available"])
        self.assertIn("upper-flow", r["reason"])
        self.assertIsNone(r["candidate"])

    def test_whole_bin_selection_records_effective_lower_edge(self):
        h = histogram(low=244.99, high=255.0)
        r = fit.fit_histogram(h, 250.0)
        self.assertTrue(r["available"])
        self.assertEqual(r["requested_lower_W_GeV"], 245.0)
        self.assertGreater(r["fit_lower_W_GeV"], 245.0)

    def test_one_percent_window_synthetic_stability(self):
        h = histogram()
        a = fit.fit_histogram(h, 250.0, lower_fraction=0.98)
        b = fit.fit_histogram(h, 250.0, lower_fraction=0.99)
        self.assertTrue(a["available"] and b["available"])
        self.assertAlmostEqual(a["sigma_W_GeV"], b["sigma_W_GeV"], places=7)

    def test_bad_histogram_rejected(self):
        mutations = [
            lambda h: h["bins"][0].__setitem__("weight", -1.0),
            lambda h: h.__setitem__("total_weight", 2.0),
            lambda h: h["bins"][4].__setitem__("low", h["bins"][4]["low"] + 0.001),
        ]
        for mutate in mutations:
            h = histogram()
            mutate(h)
            with self.assertRaises(ValueError):
                fit.fit_histogram(h, 250.0)


class NativePeakTests(unittest.TestCase):
    def test_actual_cpp_serializer_populated_empty_partial_spin_and_unsupported(self):
        data = json.loads(
            (Path(__file__).with_name("testdata") / "collider_peak_native_fixture.json").read_text()
        )
        for p in data.values():
            peak.record(p)
        p = data["populated"]
        r = fit.fit_histogram(p["luminosity_spectrum"], p["selection"]["nominal_W_GeV"])
        self.assertFalse(r["available"])
        self.assertIn("upper-flow", r["reason"])
        self.assertIsNone(data["partial_spin"]["polarization"])
        self.assertTrue(data["empty"]["available"])

    def test_exact_selection_moments_and_zero_weight_availability(self):
        peak.validate_all(diagnostic())
        h = histogram(area=0.0)
        h["samples"] = 0
        d = diagnostic(native_peak(h))
        peak.validate_all(d)
        built = fit.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        self.assertFalse(built["total"]["available"])
        self.assertTrue(d["total"]["peak"]["available"])

    def test_native_metadata_and_selection_corruption_rejected(self):
        for change in [
            lambda p: p["selection"].__setitem__("minimum_W_GeV", 244.0),
            lambda p: p["selection"]["nominal_reference"].__setitem__(
                "full_crossing_angle_rad", 0.01
            ),
            lambda p: p.__setitem__("contributions", 999),
            lambda p: p.__setitem__("spin_contributions", 1001),
            lambda p: p["W_GeV"].__setitem__("mean", 240.0),
        ]:
            p = native_peak()
            change(p)
            with self.assertRaises(ValueError):
                peak.record(p)

    def test_weighted_means_and_polarization_total_closure(self):
        d = diagnostic()
        d["total"]["peak"]["beam_energy_GeV"]["beam1"]["mean"] = 124.0
        with self.assertRaises(ValueError):
            peak.validate_all(d)
        d = diagnostic()
        d["total"]["peak"]["polarization"]["beam1"] = 0.5
        with self.assertRaises(ValueError):
            peak.validate_all(d)

    def test_reference_binds_external_design_angle(self):
        d = diagnostic()
        man = dict(
            energy_GeV=[125.0, 125.0],
            native_beam_angles_x_rad=[0.0, 0.0],
            native_beam_angles_y_rad=[0.0, 0.0],
        )
        analysis = dict(
            definition=dict(
                energy_GeV=[125.0, 125.0],
                full_crossing_angle_rad=0.0,
                nominal_W_GeV=250.0,
                angle_receipt_matches_design=True,
                design_W=peak.DEFINITION,
            )
        )
        peak.bind_reference(d, man, analysis)
        analysis["definition"]["full_crossing_angle_rad"] = 0.014
        with self.assertRaises(ValueError):
            peak.bind_reference(d, man, analysis)

    def test_fit_exact_native_source_binding(self):
        d = diagnostic()
        data = fit.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        fit.validate(data, 0.1768, d)
        data["total"]["candidate"]["sigma_W_GeV"] *= 1.2
        with self.assertRaises(ValueError):
            fit.validate(data, 0.1768, d)

    def test_refit_comparison_allows_roundoff_not_coupled_width_replacement(self):
        d = diagnostic()
        data = fit.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        for r in [data["total"], *data["frames"]]:
            r["sigma_W_GeV"] += 1e-9
            r["candidate"]["sigma_W_GeV"] = r["sigma_W_GeV"]
            r["input_reference_ratio"] = 0.1768 / r["sigma_W_GeV"]
            r["candidate"]["input_reference_ratio"] = r["input_reference_ratio"]
            r["solver"]["iterations"] += 1
        fit.validate(data, 0.1768, d)
        for r in [data["total"], *data["frames"]]:
            r["sigma_W_GeV"] *= 1.01
            r["candidate"]["sigma_W_GeV"] = r["sigma_W_GeV"]
            r["input_reference_ratio"] = 0.1768 / r["sigma_W_GeV"]
            r["candidate"]["input_reference_ratio"] = r["input_reference_ratio"]
        with self.assertRaisesRegex(ValueError, "deterministic analysis"):
            fit.validate(data, 0.1768, d)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.man = fixture(self.root)
        d = json.loads((self.root / "gui_diagnostics.json").read_text())
        p = native_peak()
        d["total"]["peak"] = copy.deepcopy(p)
        d["frames"][0]["peak"] = p
        # The new Appendix A.1 model uses the full exported histogram. Supply
        # a contiguous Gaussian-limit fixture, not the old two disjoint bins.
        for record in [d["total"], *d["frames"]]:
            record["luminosity_spectrum"] = copy.deepcopy(p["luminosity_spectrum"])
        (self.root / "gui_diagnostics.json").write_text(json.dumps(d))
        self.man["diagnostics"] = ref(self.root / "gui_diagnostics.json")
        a = json.loads((self.root / "collider_analysis.json").read_text())
        a["definition"] = dict(
            energy_GeV=[125.0, 125.0],
            full_crossing_angle_rad=0.0,
            nominal_W_GeV=250.0,
            angle_receipt_matches_design=True,
            design_W=peak.DEFINITION,
        )
        (self.root / "collider_analysis.json").write_text(json.dumps(a))
        self.man["analysis"] = ref(self.root / "collider_analysis.json")

    def test_finalizer_authentication_provenance_and_full_rms_preserved(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        finalizer.verify(self.root, self.man)
        d = json.loads((self.root / "collider_monochromatization.json").read_text())
        self.assertEqual(d["measured"]["total"]["sigma_W_GeV"], 0.25)
        self.assertAlmostEqual(d["peak_fit"]["total"]["sigma_W_GeV"], 0.18, places=7)
        self.assertIn(
            "collider_peak_fit", self.man["diagnostic_analysis_provenance"]["adapter_sha256"]
        )
        self.assertIn(
            "collider_peak_diagnostics",
            self.man["diagnostic_analysis_provenance"]["adapter_sha256"],
        )
        self.assertIn(
            "collider_beta_peak_fit", self.man["diagnostic_analysis_provenance"]["adapter_sha256"]
        )

    def test_finalizer_rejects_fit_and_design_source_corruption(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        path = self.root / "collider_monochromatization.json"
        d = json.loads(path.read_text())
        d["peak_fit"]["source_reference"]["sha256"] = "1" * 64
        path.write_text(json.dumps(d))
        self.man["monochromatization"] = ref(path)
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_finalizer_requires_derived_peak_fit_when_native_peak_exported(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        path = self.root / "collider_monochromatization.json"
        d = json.loads(path.read_text())
        d.pop("peak_fit")
        path.write_text(json.dumps(d))
        self.man["monochromatization"] = ref(path)
        with self.assertRaisesRegex(ValueError, "requires its bound fit"):
            finalizer.verify(self.root, self.man)


if __name__ == "__main__":
    unittest.main()
