from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_axe_mapping import finalize_mapping


class MappingMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {
            "status": "verified",
            "mappings": [{"legacy_class_name": "fixture"}],
            "issues": [],
        }

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "mapping.json"
            report = finalize_mapping(self.mapping, output, dry_run=True)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())
            self.assertFalse(report["output_written"])
            self.assertIsNone(report["existing_output_matches"])

    def test_dry_run_compares_existing_without_changing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mapping.json"
            original = json.dumps(self.mapping)
            output.write_text(original, encoding="utf-8")
            report = finalize_mapping(self.mapping, output, dry_run=True)
            self.assertTrue(report["existing_output_matches"])
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_write_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mapping.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                finalize_mapping(self.mapping, output, dry_run=False)


if __name__ == "__main__":
    unittest.main()
