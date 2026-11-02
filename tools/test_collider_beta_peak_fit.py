"""Appendix A.1 normalization, bin integration, recovery and guarded source binding."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln

import collider_beta_peak_fit as beta
import collider_peak_fit as legacy
import collider_peak_diagnostics as peak
import collider_monochromatization as mono
import finalize_gui_dataset as finalizer
from test_collider_peak_fit import native_peak, diagnostic, histogram as gaussian_histogram
from test_diagnostic_analysis_binding import fixture, ref


def histogram(
    endpoint=249.95, sigma=0.18, alpha=25.0, shape_beta=0.4, fraction=0.25, area=1.0, scale=1
):
    edges = np.r_[np.linspace(0, 240, 120 * scale + 1), np.linspace(240, 255, 300 * scale + 1)[1:]]
    g = beta.gaussian_bins(edges[:-1], edges[1:], endpoint, sigma)
    t = beta.beta_bins(edges[:-1], edges[1:], endpoint, sigma, alpha, shape_beta, 384)
    weights = area * (fraction * g + (1 - fraction) * t)
    return dict(
        quantity="W",
        unit="GeV",
        weight_unit="m^-2 per BX",
        weighting="native collision pair luminosity",
        normalization="bin-integrated",
        bins=[
            dict(low=float(a), high=float(b), weight=float(w))
            for a, b, w in zip(edges, edges[1:], weights)
        ],
        samples=1000,
        underflow=0.0,
        overflow=0.0,
        total_weight=float(math.fsum(weights)),
    )


def diagnostic_fixture(h=None):
    h = histogram() if h is None else h
    edges = np.linspace(245, 255, 513)
    g = beta.gaussian_bins(edges[:-1], edges[1:], 249.95, 0.18)
    t = beta.beta_bins(edges[:-1], edges[1:], 249.95, 0.18, 25.0, 0.4, 384)
    ph = gaussian_histogram()
    for b, w in zip(ph["bins"], 0.25 * g + 0.75 * t):
        b["weight"] = float(w)
    ph["total_weight"] = math.fsum(b["weight"] for b in ph["bins"])
    p = native_peak(ph)
    d = diagnostic(p)
    d["total"]["luminosity_spectrum"] = copy.deepcopy(h)
    d["frames"][0]["luminosity_spectrum"] = copy.deepcopy(h)
    return d


class NumericalTests(unittest.TestCase):
    def test_normalized_model_including_endpoint_singularities(self):
        edges = np.linspace(-3, 253, 1025)
        for a, b in [(25.0, 0.4), (0.2, 0.15), (1.0, 1.0), (100.0, 3.0)]:
            with self.subTest(alpha=a, beta=b):
                q = beta.beta_bins(edges[:-1], edges[1:], 250.0, 0.18, a, b, 384)
                self.assertAlmostEqual(float(sum(q)), 1.0, delta=2e-8)
                self.assertTrue(np.all(q >= 0))

    def test_independent_algebraic_weight_quadrature(self):
        e, s, a, b = 250.0, 0.18, 27.0, 0.38
        for lo, hi in [(0, 240), (244, 245), (249.9, 249.925), (250.0, 250.025), (250.5, 250.7)]:
            # QUADPACK's algebraic endpoint weights differ from the production
            # substitutions and directly implement the paper's beta density.
            fun = lambda x: float(
                beta.gaussian_bins(np.array([lo]), np.array([hi]), e * x, s)[0]
            ) * math.exp(-betaln(a, b))
            expected, error = quad(
                fun, 0, 1, weight="alg", wvar=(a - 1, b - 1), epsabs=1e-11, epsrel=2e-9, limit=400
            )
            actual = beta.beta_bins(np.array([lo]), np.array([hi]), e, s, a, b, 192)[0]
            self.assertAlmostEqual(actual, expected, delta=max(2e-9, 5 * error))

    def test_uniform_latent_distribution_symmetry_and_delta_limit(self):
        a = np.array([20.0, 100.0, 249.0])
        b = a + 0.1
        q = beta.beta_bins(a, b, 250.0, 0.2, 1.0, 1.0, 192)
        mirror = beta.beta_bins(250 - b, 250 - a, 250.0, 0.2, 1.0, 1.0, 192)
        self.assertTrue(np.allclose(q, mirror, rtol=1e-9, atol=1e-11))
        self.assertTrue(np.allclose(q[:2], 0.1 / 250, rtol=1e-9))
        g = beta.gaussian_bins(np.array([-5]), np.array([255]), 250.0, 0.18)
        self.assertAlmostEqual(float(g[0]), 1.0, places=12)

    def test_exact_beta_mixture_recovery(self):
        r = beta.fit_histogram(histogram(area=1e34), 250.0, 0.1768)
        self.assertTrue(r["available"], r["reason"])
        self.assertAlmostEqual(r["endpoint_W_GeV"], 249.95, delta=2e-5)
        self.assertAlmostEqual(r["sigma_W_GeV"], 0.18, delta=2e-5)
        self.assertAlmostEqual(r["alpha"], 25.0, delta=0.01)
        self.assertAlmostEqual(r["beta"], 0.4, delta=0.0001)
        self.assertAlmostEqual(r["peak_fraction"], 0.25, delta=0.0001)
        self.assertLess(r["relative_l2_residual"], 1e-6)
        self.assertAlmostEqual(r["model_area_native"] / 1e34, 1.0, delta=2e-6)
        self.assertAlmostEqual(r["input_reference_ratio"], 0.1768 / 0.18, delta=1e-5)

    def test_weight_scaling_and_bin_resolution_invariance(self):
        a = beta.fit_histogram(histogram(area=1e-20), 250.0)
        b = beta.fit_histogram(histogram(scale=2), 250.0)
        self.assertTrue(a["available"] and b["available"])
        self.assertAlmostEqual(a["sigma_W_GeV"], b["sigma_W_GeV"], delta=2e-5)

    def test_pure_gaussian_limit_retains_resolved_core(self):
        r = beta.fit_histogram(histogram(fraction=1.0), 250.0, 0.1768)
        self.assertTrue(r["available"], r["reason"])
        self.assertEqual(r["model_regime"], "gaussian_delta_limit")
        self.assertIsNone(r["alpha"])
        self.assertIsNone(r["beta"])
        self.assertEqual(r["peak_fraction"], 1.0)
        self.assertAlmostEqual(r["sigma_W_GeV"], 0.18, delta=1e-6)

    def test_sparse_empty_and_flow_never_publish_width(self):
        h = histogram()
        h["bins"] = [dict(b, weight=1.0 if i == 300 else 0.0) for i, b in enumerate(h["bins"])]
        h["total_weight"] = 1.0
        for source in (h, histogram(area=0.0)):
            r = beta.fit_histogram(source, 250.0)
            self.assertFalse(r["available"])
            self.assertIsNone(r["candidate"])
        h = histogram()
        h["overflow"] = 0.01
        h["total_weight"] += 0.01
        r = beta.fit_histogram(h, 250.0)
        self.assertFalse(r["available"])
        self.assertIn("flow", r["reason"])

    def test_coarse_width_is_unresolved(self):
        r = beta.fit_histogram(histogram(sigma=0.025), 250.0)
        self.assertFalse(r["available"])
        self.assertIsNone(r["candidate"])

    def test_shape_width_is_separate_from_full_rms(self):
        h = histogram()
        r = beta.fit_histogram(h, 250.0)
        y = np.array([b["weight"] for b in h["bins"]])
        y /= y.sum()
        x = np.array([(b["low"] + b["high"]) / 2 for b in h["bins"]])
        rms = math.sqrt(float(np.sum((x - float(x @ y)) ** 2 * y)))
        self.assertGreater(rms, 10 * r["sigma_W_GeV"])

    def test_invalid_source_rejected(self):
        h = histogram()
        h["bins"][0]["weight"] = -1
        with self.assertRaises(ValueError):
            beta.fit_histogram(h, 250.0)


class ContractTests(unittest.TestCase):
    def test_exact_source_binding_and_peak_overlay_integrals(self):
        d = diagnostic_fixture()
        data = beta.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        beta.validate(data, 0.1768, d)
        c = data["total"]["candidate"]
        self.assertEqual(len(c["curve"]["tail_bin_probabilities"]), 420)
        self.assertEqual(len(c["peak_curve"]["tail_bin_probabilities"]), 512)
        self.assertEqual(data["fit_source"]["histogram"], "luminosity_spectrum")
        self.assertAlmostEqual(c["sigma_fractional"], c["sigma_W_GeV"] / c["endpoint_W_GeV"])

    def test_curve_and_coupled_parameter_tamper_rejected(self):
        d = diagnostic_fixture()
        data = beta.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        for edit in [
            lambda c: c["curve"]["tail_bin_probabilities"].__setitem__(300, 0.1),
            lambda c: c["peak_curve"]["tail_bin_probabilities"].__setitem__(300, 0.1),
        ]:
            bad = copy.deepcopy(data)
            edit(bad["total"]["candidate"])
            with self.assertRaises(ValueError):
                beta.validate(bad, 0.1768, d)
        bad = copy.deepcopy(data)
        for r in [bad["total"], *bad["frames"]]:
            c = r["candidate"]
            c["sigma_W_GeV"] *= 1.01
            c["sigma_fractional"] = c["sigma_W_GeV"] / c["endpoint_W_GeV"]
            c["input_reference_ratio"] = 0.1768 / c["sigma_W_GeV"]
            for k in ("sigma_W_GeV", "sigma_fractional", "input_reference_ratio"):
                r[k] = c[k]
        with self.assertRaises(ValueError):
            beta.validate(bad, 0.1768, d)

    def test_quality_admission_cannot_be_bypassed(self):
        d = diagnostic_fixture()
        data = beta.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        for edit in [
            lambda q: q.__setitem__("quadrature_relative_l2_error", 0.01),
            lambda q: q["sigma_profile_probes"][0].__setitem__("objective_increase", 0),
            lambda q: q.__setitem__("interior_parameters", False),
        ]:
            bad = copy.deepcopy(data)
            edit(bad["total"]["quality"])
            with self.assertRaises(ValueError):
                beta.validate(bad, 0.1768)

    def test_legacy_gaussian_stays_readable(self):
        d = diagnostic()
        old = legacy.build(d, 0.1768, dict(path="gui_diagnostics.json", bytes=1, sha256="0" * 64))
        legacy.validate(old, 0.1768, d)
        self.assertEqual(old["schema_version"], "feebl.collider.gaussian-peak-fit/1")

    def test_cache_does_not_trust_supplied_results(self):
        h = histogram()
        r = beta.fit_histogram(h, 250.0)
        r["sigma_W_GeV"] = 999.0
        again = beta.fit_histogram(h, 250.0)
        self.assertNotEqual(again["sigma_W_GeV"], 999.0)


if __name__ == "__main__":
    unittest.main()
