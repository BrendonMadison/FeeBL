import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "contract", Path(__file__).with_name("finalize_gui_dataset.py")
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        frames = []
        for i in range(2):
            arts = {}
            for b in (1, 2):
                name = f"gui_b{b}.{i}"
                (self.root / name).write_text("125 0.2 -0.3 1 0.04 0.05\n")
                arts[f"beam{b}"] = c.describe_frame(self.root, name)
            frames.append(dict(id=i, elapsed_s=i * 1e-12, **arts))
        self.man = dict(
            schema_version=c.SCHEMA,
            units=dict(energy="GeV", position="um", angle="urad", time="s"),
            coordinates=dict(frame="legacy_collider_dump"),
            frames=frames,
        )

    def tearDown(self):
        self.temp.cleanup()

    def reject(self, mutate):
        m = copy.deepcopy(self.man)
        mutate(m)
        with self.assertRaises((ValueError, KeyError)):
            c.verify(self.root, m)

    def test_valid(self):
        self.assertEqual(c.verify(self.root, self.man)["frames"], 2)

    def test_schema(self):
        self.reject(lambda m: m.update(schema_version="future/3"))

    def test_units(self):
        self.reject(lambda m: m["units"].update(position="m"))

    def test_coordinates(self):
        self.reject(lambda m: m["coordinates"].update(frame="lab"))

    def test_time(self):
        self.reject(lambda m: m["frames"][1].update(elapsed_s=0))

    def test_id(self):
        self.reject(lambda m: m["frames"][1].update(id=0))

    def test_path(self):
        self.reject(lambda m: m["frames"][0]["beam1"].update(path="../outside"))

    def test_remote(self):
        self.reject(lambda m: m["frames"][0]["beam1"].update(path="https://remote"))

    def test_duplicate(self):
        self.reject(lambda m: m["frames"][1].update(beam1=m["frames"][0]["beam1"]))

    def test_hash(self):
        self.reject(lambda m: m["frames"][0]["beam1"].update(sha256="0" * 64))

    def test_count(self):
        self.reject(lambda m: m["frames"][0]["beam1"].update(rows=12))

    def test_bounds(self):
        self.reject(lambda m: m["frames"][0]["beam1"].update(bounds=[[0, 0]] * 6))

    def test_missing(self):
        (self.root / "gui_b2.1").unlink()
        with self.assertRaises(ValueError):
            c.verify(self.root, self.man)

    def test_truncated(self):
        (self.root / "gui_b2.1").write_text("125 0.1")
        with self.assertRaises(ValueError):
            c.verify(self.root, self.man)

    def test_nonfinite(self):
        (self.root / "gui_b2.1").write_text("125 nan 0 0 0 0")
        with self.assertRaises(ValueError):
            c.verify(self.root, self.man)

    def test_row_limit(self):
        (self.root / "gui_b2.1").write_text("125 0 0 0 0 0\n" * 4097)
        with self.assertRaises(ValueError):
            c.verify(self.root, self.man)

    def test_spin_columns(self):
        (self.root / "spin").write_text("125 0 0 0 0 0 0 0 1\n")
        self.assertEqual(c.describe_frame(self.root, "spin")["columns"], 9)

    def test_mixed_columns(self):
        (self.root / "mixed").write_text("125 0 0 0 0 0\n125 0 0 0 0 0 0 0 1\n")
        with self.assertRaises(ValueError):
            c.describe_frame(self.root, "mixed")


class NormalizationTests(unittest.TestCase):
    def test_evaluated_rate_and_horizontal_angle(self):
        angle, rate, receipt = c.resolve_collision_normalization(
            {"native_beam_angles_x_rad": [.015, .015], "native_beam_angles_y_rad": [0, 0]},
            "f_rep = 5 n_b = 2820", None, None)
        self.assertEqual(angle, .03)
        self.assertEqual(rate, 14100)
        self.assertEqual(receipt["crossing_frequency_hz"]["source"], "evaluated native summary")

    def test_explicit_override_is_identified(self):
        angle, rate, receipt = c.resolve_collision_normalization({}, "", .01, 100)
        self.assertEqual((angle, rate), (.01, 100))
        self.assertEqual(receipt["full_crossing_angle_rad"]["source"], "explicit CLI override")

    def test_nonhorizontal_angle_needs_override(self):
        angle, _, _ = c.resolve_collision_normalization(
            {"native_beam_angles_x_rad": [0, 0], "native_beam_angles_y_rad": [.01, .01]},
            "", None, None)
        self.assertIsNone(angle)

    def test_nonfinite_and_zero_rate_are_rejected(self):
        for angle, rate in [(float("nan"), 1), (0, 0), (0, float("inf"))]:
            with self.assertRaises(ValueError):
                c.resolve_collision_normalization({}, "", angle, rate)


class CompletionReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.man = dict(backend_actual_receipt="initial", receipts={})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, text):
        (self.root / "log.txt").write_text(text)

    def test_completed_cpu(self):
        self.write("FEEBL: final compute_backend requested=cpu actual=cpu\n")
        c.bind_completion_receipt(self.root, self.man)
        self.assertEqual(self.man["backend_initialization_receipt"], "initial")
        self.assertIn("actual=cpu", self.man["backend_actual_receipt"])
        self.assertEqual(
            self.man["receipts"]["runtime_completion_log"]["sha256"],
            c.digest(self.root / "log.txt"),
        )

    def test_missing_completion(self):
        self.write("FEEBL: compute_backend requested=cpu actual=cpu\n")
        with self.assertRaises(ValueError):
            c.bind_completion_receipt(self.root, self.man)

    def test_duplicate_completion(self):
        self.write("FEEBL: final compute_backend requested=cpu actual=cpu\n" * 2)
        with self.assertRaises(ValueError):
            c.bind_completion_receipt(self.root, self.man)

    def test_fallback_rejected(self):
        self.write("FEEBL: final compute_backend requested=gpu actual=cpu\n")
        with self.assertRaises(ValueError):
            c.bind_completion_receipt(self.root, self.man)

    def test_uninitialized_gpu(self):
        self.write(
            'FEEBL: final compute_backend requested=gpu actual=gpu gpu_calls=0 gpu_device_description="not-initialized"\n'
        )
        with self.assertRaises(ValueError):
            c.bind_completion_receipt(self.root, self.man)

    def test_actual_gpu(self):
        self.write(
            'FEEBL: final compute_backend requested=gpu actual=gpu gpu_calls=15 gpu_device_description="NVIDIA RTX"\n'
        )
        c.bind_completion_receipt(self.root, self.man)
        self.assertIn("gpu_calls=15", self.man["backend_actual_receipt"])


if __name__ == "__main__":
    unittest.main()
