"""Reject unproven version reuse, partial families and overlapping reductions."""
import json
from pathlib import Path
import unittest
import uuid

import numpy as np

import merge_seed_ensembles as merger
from comparison_statistics import PANELS, panel_case, read_rows
from seed_campaign import DEFAULT_CASES, seed_for_case


class MergeSeedTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / ("test_merge_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.old_binary, self.new_binary = "a" * 64, "b" * 64
        self.controls = {
            "schema": "feebl.cross-version-controls/1",
            "status": "PASS",
            "controls": [
                {
                    "case": case,
                    "seed": seed_for_case(case, 0),
                    "old_binary_sha256": self.old_binary,
                    "new_binary_sha256": self.new_binary,
                    "old_spectrum_sha256": "c" * 64,
                    "new_spectrum_sha256": "c" * 64,
                    "scientific_summary_equal": True,
                    "ignored_summary_metadata": {},
                }
                for case in DEFAULT_CASES[:4]
            ],
        }
        self.control_path = self.root / "controls.json"
        self.save_controls()

    def save_controls(self):
        self.control_path.write_text(json.dumps(self.controls))

    def tearDown(self):
        self.assertEqual(self.root.parent.resolve(), Path(__file__).resolve().parent)
        for path in sorted(
            self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            path.unlink() if path.is_file() else path.rmdir()
        self.root.rmdir()

    def make_fixture(self, name, cases, binary):
        """Write synthetic matrix contracts only; these are never scientific outputs."""
        directory = self.root / name
        directory.mkdir()
        arrays, panels, inventories = {}, [], []
        for case in cases:
            canonical = [seed_for_case(case, replica) for replica in range(12)]
            inventories.append(
                {
                    "case": case,
                    "complete_requested_ensemble": True,
                    "records": [
                        {"seed": seed, "status": "EXECUTED", "binary_sha256": binary}
                        for seed in canonical
                    ],
                }
            )
            for source in PANELS.glob("*_pull.csv"):
                if source.stem.startswith("lep_") or panel_case(source.stem) != case:
                    continue
                name = source.stem
                arrays[name] = np.ones((12, len(read_rows(source))))
                arrays[name + "__seeds"] = np.asarray(canonical)
                arrays[name + "__encounter_neff"] = np.ones(12)
                panels.append(
                    {
                        "panel": name,
                        "case": case,
                        "seeds": canonical,
                        "binary_sha256": [binary],
                        "native_runs": [
                            {
                                "seed": seed,
                                "native_sha256": "d" * 64,
                                "execution_receipt_sha256": "e" * 64,
                            }
                            for seed in canonical
                        ],
                    }
                )
        np.savez_compressed(directory / "seed_vectors.npz", **arrays)
        (directory / "seed_vectors.json").write_text(json.dumps({"panels": panels}))
        (directory / "campaign_inventory.json").write_text(
            json.dumps({"cases": inventories})
        )
        return directory / "seed_vectors.npz"

    def test_disjoint_complete_families_keep_source_provenance(self):
        paths = [
            self.make_fixture("old", DEFAULT_CASES[:4], self.old_binary),
            self.make_fixture("new", DEFAULT_CASES[4:], self.new_binary),
        ]
        result = merger.merge(
            paths, ["1" * 64, "2" * 64], self.control_path, self.root / "merged"
        )
        self.assertEqual({row["case"] for row in result["panels"]}, set(DEFAULT_CASES))
        self.assertEqual(
            {row["native_source_archive_sha256"] for row in result["panels"]},
            {"1" * 64, "2" * 64},
        )
        self.assertTrue(result["all_requested_cases_complete"])

    def test_overlapping_families_are_rejected(self):
        paths = [
            self.make_fixture("old", DEFAULT_CASES[:4], self.old_binary),
            self.make_fixture("new", DEFAULT_CASES[3:], self.new_binary),
        ]
        with self.assertRaisesRegex(ValueError, "pool, repeat or replace"):
            merger.merge(
                paths, ["1" * 64, "2" * 64], self.control_path, self.root / "merged"
            )
        self.assertFalse((self.root / "merged").exists())

    def test_controls_reject_changed_spectra_or_hidden_scientific_fields(self):
        old = {case: self.old_binary for case in DEFAULT_CASES[:4]}
        self.controls["controls"][0]["new_spectrum_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Spectrum changed"):
            merger.check_controls(self.controls, old, self.new_binary)
        self.controls["controls"][0].update(
            {
                "old_spectrum_payload_sha256": "9" * 64,
                "new_spectrum_payload_sha256": "9" * 64,
                "ignored_spectrum_metadata": [
                    {
                        "name": "source_revision",
                        "old": "old_revision",
                        "new": "new_revision",
                    }
                ],
            }
        )
        merger.check_controls(self.controls, old, self.new_binary)
        self.controls["controls"][0]["ignored_spectrum_metadata"][0][
            "name"
        ] = "n_eff_total"
        with self.assertRaisesRegex(ValueError, "Unapproved spectrum"):
            merger.check_controls(self.controls, old, self.new_binary)
        self.controls["controls"][0]["new_spectrum_sha256"] = "c" * 64
        self.controls["controls"][0]["ignored_summary_metadata"] = {"n_pairs": 0}
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            merger.check_controls(self.controls, old, self.new_binary)
        self.controls["controls"][0]["ignored_summary_metadata"] = {
            "imported_spin_mode": 0
        }
        merger.check_controls(self.controls, old, self.new_binary)
        self.controls["controls"][0]["ignored_summary_metadata"] = [
            {"name": "imported_spin_mode", "value": 0}
        ]
        merger.check_controls(self.controls, old, self.new_binary)


if __name__ == "__main__":
    unittest.main()
