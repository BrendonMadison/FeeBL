"""Check that the viewer server exposes only its assets and selected run."""
from pathlib import Path
import tempfile
import unittest

from serve_viewer import resolve_file


class ViewerPathTests(unittest.TestCase):
    def test_routes_and_escaped_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            manifest = run / "gui_manifest.json"
            manifest.write_text("{}")
            (root / "private.json").write_text("{}")
            self.assertEqual(resolve_file("/data/gui_manifest.json?x=1", run), manifest)
            self.assertIsNotNone(resolve_file("/feebl_gui.html", run))
            self.assertIsNotNone(resolve_file("/collider_fields.js", run))
            for path in ("/data/../private.json", "/data/%2e%2e%2fprivate.json",
                         "/data/..%5cprivate.json", "/data/", "/test_collider_fields.js",
                         "/finalize_gui_dataset.py", "/data/missing.json"):
                self.assertIsNone(resolve_file(path, run), path)


if __name__ == "__main__":
    unittest.main()
