"""Check export identity stability, changed-source detection and scope."""
from pathlib import Path
import tempfile
import unittest

import stamp_source_export as source


class SourceExportTests(unittest.TestCase):
    def test_source_changes_are_detected_and_paper_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in source.ROOT_FILES:
                (root / name).write_text("fixture\n")
            for name in source.DIRECTORIES:
                (root / name).mkdir()
                (root / name / "fixture.cc").write_text("initial source\n")
            original, _ = source.stamp(root, write=True)
            self.assertEqual(original, source.stamp(root)[0])
            (root / "paper.tex").write_text("A separate document\n")
            self.assertEqual(original, source.stamp(root)[0])
            (root / "src/fixture.cc").write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                source.stamp(root)
            changed, _ = source.stamp(root, write=True)
            self.assertNotEqual(original, changed)

    def test_canonical_mapping_order(self):
        self.assertEqual(source.identity({"a": "1", "b": "2"}),
                         source.identity({"b": "2", "a": "1"}))

    def test_missing_build_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                source.stamp(Path(directory), write=True)


if __name__ == "__main__":
    unittest.main()
