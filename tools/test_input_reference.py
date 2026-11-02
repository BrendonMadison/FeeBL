"""Guard the public card reference against missing and misleading metadata."""
import json
import unittest

import generate_input_reference as reference
from input_reference_metadata import metadata


class InputReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {row["name"]: row for row in reference.registry()}

    def test_every_registered_key_has_description_and_real_sources(self):
        self.assertGreater(len(self.rows), 300)
        self.assertEqual(set(self.rows), set(metadata(reference.ROOT)))
        for name, row in self.rows.items():
            with self.subTest(name=name):
                self.assertGreater(len(row["meaning"].strip()), 15)
                self.assertTrue(row["conditions"].strip())
                self.assertTrue(row["unit"].strip())
                self.assertTrue(row["semantic_sources"])
                for path in row["semantic_sources"]:
                    self.assertTrue((reference.ROOT / path).is_file())

    def test_registry_links_point_to_actual_declarations(self):
        lines = (reference.ROOT / "src/readData.cc").read_text(encoding="utf-8").splitlines()
        for name, row in self.rows.items():
            with self.subTest(name=name):
                self.assertIn(f'def_named_variable("{name}"', lines[row["registry_line"]-1])

    def test_misleading_legacy_controls_are_explicit(self):
        self.assertIn("no active control", self.rows["hist_ee_bins"]["meaning"])
        self.assertIn("no implemented physics effect", self.rows["couple_xy"]["meaning"])
        self.assertIn("not import an external laser", self.rows["ext_field"]["conditions"])
        self.assertIn("3 load both beams", self.rows["load_beam"]["conditions"])
        self.assertIn("which_espread", self.rows["espread"]["conditions"])
        self.assertNotIn("interpolation", self.rows)

    def test_units_suffixes_and_checkpoint_defaults(self):
        self.assertEqual(self.rows["offset_x"]["suffix"], "mirrored .1/.2")
        self.assertEqual(self.rows["ip_dispersion"]["unit"], "m")
        self.assertEqual(self.rows["sigma_z"]["unit"], "um")
        self.assertEqual(self.rows["pair_ecut"]["unit"], "GeV")
        self.assertEqual(self.rows["compt_x_min"]["unit"], "dimensionless")
        self.assertEqual(self.rows["rndm_load"]["default"], "0")
        self.assertEqual(self.rows["rndm_save"]["default"], "0")

    def test_contract_table_headers_never_become_descriptions(self):
        self.assertIn("reference_cic_v1", self.rows["adaptive_grid_source_model"]["meaning"])
        self.assertIn("fixed_width_bspline1_v1", self.rows["adaptive_grid_source_model"]["meaning"])
        for row in self.rows.values():
            self.assertNotEqual(row["meaning"], "Qualification status")

    def test_generated_products_are_current_and_complete(self):
        for path, expected in reference.products().items():
            self.assertEqual(path.read_text(encoding="utf-8"), expected)
        encoded = reference.products()[reference.ROOT / "docs/input_reference.json"]
        self.assertEqual(len(json.loads(encoded)), len(self.rows))


if __name__ == "__main__":
    unittest.main()
