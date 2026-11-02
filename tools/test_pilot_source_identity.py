"""Campaign requests must use the revision of their actual source tree."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "examples/adaptive_particles/generate_coupled_pilot_request.py"
)
spec = importlib.util.spec_from_file_location("pilot_request_generator", GENERATOR)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class PilotSourceIdentityTests(unittest.TestCase):
    def test_export_rejects_enclosing_git_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            (root / "SOURCE_REVISION").write_text("b" * 40 + "-dirty\n")
            outer = subprocess.CompletedProcess([], 0, str(root.parent), "")
            with (
                patch.object(generator, "SOURCE_ROOT", root),
                patch.object(generator.subprocess, "run", return_value=outer) as run,
            ):
                self.assertEqual(generator.git_head(), "b" * 40)
                self.assertEqual(run.call_count, 1)

    def test_no_git_export_requires_valid_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(generator, "SOURCE_ROOT", root),
                patch.object(generator.subprocess, "run", side_effect=FileNotFoundError),
            ):
                with self.assertRaises(RuntimeError):
                    generator.git_head()
                (root / "SOURCE_REVISION").write_text("a" * 39)
                with self.assertRaises(RuntimeError):
                    generator.git_head()
                (root / "SOURCE_REVISION").write_text("a" * 40)
                self.assertEqual(generator.git_head(), "a" * 40)

    def test_content_export_retains_digest_without_claiming_a_git_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = "sha256:" + "c" * 64
            (root / "SOURCE_REVISION").write_text(marker + "\n")
            with (
                patch.object(generator, "SOURCE_ROOT", root),
                patch.object(generator.subprocess, "run", side_effect=FileNotFoundError),
            ):
                self.assertEqual(generator.source_revision(), marker)
                self.assertIsNone(generator.git_head())
                for invalid in ("sha256:" + "c" * 63, "sha256:" + "g" * 64):
                    (root / "SOURCE_REVISION").write_text(invalid)
                    with self.assertRaises(RuntimeError):
                        generator.source_revision()


if __name__ == "__main__":
    unittest.main()
