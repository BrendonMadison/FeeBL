"""Small integrity and filesystem tests; no scientific datasets are required."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import unittest
from unittest.mock import patch
import uuid
import warnings
import zipfile

import reproduction_data as data


class OptionalDataTests(unittest.TestCase):
    """Exercise complete preflight and non-overwriting restoration with tiny ZIPs."""

    def setUp(self):
        self.work = Path(__file__).resolve().parent / (".data_test_" + uuid.uuid4().hex)
        self.work.mkdir()
        self.root = self.work / "repo"
        self.root.mkdir()
        self.archive = self.work / "data.zip"
        self.files = {"data/a.txt": b"first record\n", "data/nested/b.bin": b"\x00\x01\x02"}
        self.prepare()

    def tearDown(self):
        self.assertEqual(self.work.resolve().parent, Path(__file__).resolve().parent)
        shutil.rmtree(self.work)

    def prepare(self, members=None):
        """Bind the fixture manifest to a ZIP, even when testing invalid members."""
        if members is None:
            members = list(self.files.items())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, content in members:
                    archive.writestr(name, content)
        self.raw = {
            "schema": data.SCHEMA,
            "archive_name": self.archive.name,
            "archive_sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "files": [{"path": name, "sha256": hashlib.sha256(content).hexdigest(),
                       "bytes": len(content)} for name, content in self.files.items()],
        }
        self.save_manifest()

    def save_manifest(self):
        (self.root / "data_manifest.json").write_text(json.dumps(self.raw), encoding="utf-8")

    def restore(self, dry_run=False):
        return data.restore_data(self.root, self.archive, data.load_manifest(self.root), dry_run)

    def assert_preflight_rejects(self):
        with self.assertRaises(data.DataError):
            self.restore()
        self.assertFalse((self.root / "data").exists())

    def test_round_trip_retains_existing_matching_files(self):
        self.assertEqual(self.restore()["restored"], 2)
        for name, content in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertEqual(self.restore()["restored"], 0)
        self.assertEqual(data.verify_data(self.root, data.load_manifest(self.root))["missing"], [])

    def test_dry_run_and_missing_cli_do_not_write(self):
        self.assertEqual(len(self.restore(dry_run=True)["missing"]), 2)
        self.assertFalse((self.root / "data").exists())
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(data.main(["verify"], self.root), 2)
        self.assertIn("Missing optional data: 2 files", output.getvalue())

    def test_bad_archive_hash_rejected_before_writes(self):
        with self.archive.open("ab") as stream:
            stream.write(b"changed")
        self.assert_preflight_rejects()

    def test_late_bad_member_hash_rejected_before_any_writes(self):
        self.raw["files"][1]["sha256"] = "0" * 64
        self.save_manifest()
        self.assert_preflight_rejects()

    def test_wrong_member_size_rejected(self):
        self.raw["files"][1]["bytes"] += 1
        self.save_manifest()
        self.assert_preflight_rejects()

    def test_missing_and_extra_zip_members_rejected(self):
        for members in (list(self.files.items())[:1], list(self.files.items()) + [("README", b"extra")]):
            with self.subTest(members=members):
                self.prepare(members)
                self.assert_preflight_rejects()

    def test_unsafe_and_duplicate_zip_names_rejected(self):
        for name in ("../outside", "/absolute", "C:/absolute", "data\\a.txt", "data/a.txt", "DATA/A.TXT"):
            with self.subTest(name=name):
                self.prepare(list(self.files.items()) + [(name, b"unsafe")])
                self.assert_preflight_rejects()
        self.assertFalse((self.work / "outside").exists())

    def test_zip_symlink_rejected(self):
        link = zipfile.ZipInfo("data/a.txt")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.prepare([(link, self.files["data/a.txt"]), ("data/nested/b.bin", b"\x00\x01\x02")])
        self.assert_preflight_rejects()

    def test_unsafe_manifest_paths_rejected(self):
        for name in ("../outside", "data/../outside", "data//a", "data/CON.txt", "data/a.", "data/a:b"):
            with self.subTest(name=name):
                self.raw["files"][0]["path"] = name
                self.save_manifest()
                with self.assertRaises(data.DataError):
                    data.load_manifest(self.root)

    def test_manifest_alias_and_parent_collisions_rejected(self):
        for name in ("DATA/A.TXT", "data/a.txt/child"):
            with self.subTest(name=name):
                self.raw["files"][1]["path"] = name
                self.save_manifest()
                with self.assertRaises(data.DataError):
                    data.load_manifest(self.root)

    def test_existing_conflict_rejects_entire_restore(self):
        existing = self.root / "data/nested/b.bin"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"my existing data")
        with self.assertRaisesRegex(data.DataError, "Existing files differ"):
            self.restore()
        self.assertEqual(existing.read_bytes(), b"my existing data")
        self.assertFalse((self.root / "data/a.txt").exists())

    def test_parent_file_is_not_replaced_by_directory(self):
        (self.root / "data").write_bytes(b"keep")
        with self.assertRaisesRegex(data.DataError, "not a directory"):
            self.restore()
        self.assertEqual((self.root / "data").read_bytes(), b"keep")

    def test_stream_size_is_bounded_and_checked(self):
        with self.assertRaises(data.DataError):
            data.stream_digest(io.BytesIO(b"too long"), expected_bytes=2)
        with self.assertRaises(data.DataError):
            data.stream_digest(io.BytesIO(b"short"), expected_bytes=8)

    def test_interrupted_write_is_removed_and_can_be_retried(self):
        original_digest = data.stream_digest

        def interrupt_copy(stream, expected_bytes=None, destination=None):
            if destination is not None:
                destination.write(b"partial")
                raise KeyboardInterrupt()
            return original_digest(stream, expected_bytes)

        with patch.object(data, "stream_digest", side_effect=interrupt_copy):
            with self.assertRaises(KeyboardInterrupt):
                self.restore()
        self.assertFalse((self.root / "data/a.txt").exists())
        self.assertEqual(self.restore()["restored"], 2)


if __name__ == "__main__":
    unittest.main()
