"""Source exports must retain provenance without inheriting an outer Git tree."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from finalize_gui_dataset import source_identity


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "export"
        self.root.mkdir()
        (self.root / "SOURCE_REVISION").write_text("a" * 40 + "-dirty\n")
        (self.root / "src").mkdir()
        self.kernel = self.root / "src" / "kernel.cc"
        self.kernel.write_text("double field = 0.;\n")

    def test_export_without_git_tracks_changed_kernel(self):
        with patch("finalize_gui_dataset.subprocess.run", side_effect=FileNotFoundError):
            before, diff = source_identity(self.root)
            self.kernel.write_text("double field = 1.;\n")
            after, _ = source_identity(self.root)
        self.assertEqual(before["identity_method"], "source_export")
        self.assertEqual(before["commit"], "a" * 40)
        self.assertTrue(before["dirty"])
        self.assertEqual(diff, b"")
        self.assertNotEqual(
            before["source_files_sha256"]["src/kernel.cc"],
            after["source_files_sha256"]["src/kernel.cc"],
        )

    def test_parent_repository_is_not_the_source_revision(self):
        parent = subprocess.CompletedProcess([], 0, str(self.root.parent), "")
        with (
            patch("finalize_gui_dataset.subprocess.run", return_value=parent),
            patch("finalize_gui_dataset.subprocess.check_output") as git_read,
        ):
            identity, _ = source_identity(self.root)
        git_read.assert_not_called()
        self.assertEqual(identity["identity_method"], "source_export")

    def test_missing_or_malformed_revision_fails(self):
        revision = self.root / "SOURCE_REVISION"
        for value in ("unknown", "$Format:%H$", "a" * 39,
                      "sha256:" + "b" * 63, "sha256:" + "g" * 64):
            revision.write_text(value)
            with patch("finalize_gui_dataset.subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaises(ValueError):
                    source_identity(self.root)
        revision.unlink()
        with patch("finalize_gui_dataset.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(ValueError):
                source_identity(self.root)

    def test_content_digest_export_is_not_a_git_commit(self):
        marker = "sha256:" + "b" * 64
        (self.root / "SOURCE_REVISION").write_text(marker + "\n")
        with patch("finalize_gui_dataset.subprocess.run", side_effect=FileNotFoundError):
            identity, diff = source_identity(self.root)
        self.assertIsNone(identity["commit"])
        self.assertEqual(identity["source_revision"], marker)
        self.assertEqual(identity["source_export_sha256"], "b" * 64)
        self.assertEqual(identity["identity_method"], "source_export")
        self.assertIn("src/kernel.cc", identity["source_files_sha256"])
        self.assertEqual(diff, b"")

    def test_tar_export_retains_content_identity(self):
        marker = "sha256:" + "d" * 64
        (self.root / "SOURCE_REVISION").write_text(marker + "\n")
        archive = self.root.parent / "source.tar"
        unpacked = self.root.parent / "unpacked"
        with tarfile.open(archive, "w") as stream:
            stream.add(self.root, arcname="export")
        with tarfile.open(archive) as stream:
            stream.extractall(unpacked, filter="data")
        with patch("finalize_gui_dataset.subprocess.run", side_effect=FileNotFoundError):
            original, _ = source_identity(self.root)
            restored, _ = source_identity(unpacked / "export")
        self.assertEqual(original, restored)

    @unittest.skipIf(sys.platform == "win32", "benchmark driver requires POSIX resource module")
    def test_benchmark_manifest_distinguishes_content_and_git_identity(self):
        import benchmark_backends
        marker = "sha256:" + "c" * 64
        (self.root / "SOURCE_REVISION").write_text(marker + "\n")
        with (patch.object(benchmark_backends, "source_git_root", return_value=None),
              patch.object(benchmark_backends, "run_text", return_value="test inventory")):
            manifest = benchmark_backends.environment_manifest(
                self.kernel, self.kernel, self.root, [], [])
        self.assertEqual(manifest["source_revision"], marker)
        self.assertEqual(manifest["source_export_sha256"], "c" * 64)
        self.assertIsNone(manifest["source_git_commit"])


if __name__ == "__main__":
    unittest.main()
