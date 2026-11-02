"""Schema/closure regression fixtures; no physics-validation claim."""

import copy
import json
from pathlib import Path
import unittest
from collider_diagnostics import validate

FIXTURE = json.loads((Path(__file__).parent / "testdata/collider_species_chi.json").read_text())
KEYS = ("chi_e", "chi_e_electron", "chi_e_positron")


def records(d):
    return d["frames"] + [d["total"]]


def legacy(d):
    d.pop("primary_species")
    for r in records(d):
        for k in KEYS[1:]:
            r["distributions"].pop(k)


def unsupported(d):
    d["primary_species"].update(supported=False, charge_sign=0, beam1_pdg=None, beam2_pdg=None)
    for r in records(d):
        for k in KEYS[1:]:
            r["distributions"][k] = None


class SpeciesTests(unittest.TestCase):
    def setUp(self):
        self.d = copy.deepcopy(FIXTURE["diagnostics"])
        self.m = FIXTURE["manifest"]["frames"]

    def valid(self):
        self.assertEqual(validate(self.d, self.m)["frame_total_closure"], "PASS")

    def test_split_counts_with_flows(self):
        self.valid()

    def test_legacy_combined_remains_valid(self):
        legacy(self.d)
        self.valid()

    def test_unsupported_mapping_keeps_combined(self):
        unsupported(self.d)
        self.valid()

    def test_mapping_independent_of_hook_coverage(self):
        for r in records(self.d):
            for k in KEYS:
                r["distributions"][k] = None
        self.valid()

    def test_computed_zero_counts(self):
        for r in records(self.d):
            for k in KEYS:
                h = r["distributions"][k]
                for b in h["bins"]:
                    b["weight"] = 0
                for f in ("underflow", "overflow", "total_weight", "samples"):
                    h[f] = 0
        self.valid()

    def test_rejections(self):
        cases = [
            ("no mapping", lambda d: d.pop("primary_species")),
            ("wrong beam identity", lambda d: d["primary_species"].update(beam1_pdg=-11)),
            ("wrong charge", lambda d: d["primary_species"].update(charge_sign=0)),
            ("wrong basis", lambda d: d["primary_species"].update(basis="energy sign")),
            ("missing metadata value", lambda d: d["primary_species"].pop("supported")),
            (
                "missing species key",
                lambda d: d["frames"][0]["distributions"].pop("chi_e_electron"),
            ),
            (
                "partial availability",
                lambda d: d["frames"][0]["distributions"].update(chi_e_electron=None),
            ),
            (
                "wrong species label",
                lambda d: d["total"]["distributions"]["chi_e_electron"].update(
                    quantity="native primary positron kick chi_e"
                ),
            ),
            (
                "wrong weighting",
                lambda d: d["total"]["distributions"]["chi_e_electron"].update(
                    weighting="display particles"
                ),
            ),
            (
                "fractional counts",
                lambda d: d["frames"][0]["distributions"]["chi_e_electron"]["bins"][0].update(
                    weight=1.5
                ),
            ),
            (
                "wrong samples",
                lambda d: d["total"]["distributions"]["chi_e_electron"].update(samples=9),
            ),
            (
                "mismatched edges",
                lambda d: d["frames"][0]["distributions"]["chi_e_electron"]["bins"][0].update(
                    low=0.5
                ),
            ),
            (
                "missing split reason",
                lambda d: (unsupported(d), d["unavailable_reasons"].pop("chi_e_electron")),
            ),
            (
                "unsupported nonnull split",
                lambda d: d["primary_species"].update(
                    supported=False, charge_sign=0, beam1_pdg=None, beam2_pdg=None
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                d = copy.deepcopy(self.d)
                mutate(d)
                with self.assertRaises((ValueError, KeyError)):
                    validate(d, self.m)

    def test_split_bin_closure(self):
        for r in records(self.d):
            h = r["distributions"]["chi_e_electron"]
            h["bins"][0]["weight"] += 1
            h["total_weight"] += 1
            h["samples"] += 1
        with self.assertRaisesRegex(ValueError, "species/combined"):
            self.valid()

    def test_exact_large_integer_frame_closure(self):
        # A one-count bin shift is below generic float tolerances but is invalid for count histograms.
        for r in records(self.d):
            for k in KEYS:
                h = r["distributions"][k]
                for b in h["bins"]:
                    b["weight"] *= 1000000000
                for f in ("underflow", "overflow", "total_weight", "samples"):
                    h[f] *= 1000000000
        for k in ("chi_e", "chi_e_electron"):
            h = self.d["total"]["distributions"][k]
            h["bins"][0]["weight"] += 1
            h["bins"][1]["weight"] -= 1
        with self.assertRaisesRegex(ValueError, "exact frame/total"):
            self.valid()


if __name__ == "__main__":
    unittest.main()
