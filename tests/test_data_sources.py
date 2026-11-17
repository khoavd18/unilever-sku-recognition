from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.annotations import AnnotationFormat
from src.data.sources import DatasetSourceRegistry, load_dataset_sources


class DatasetSourceRegistryTests(unittest.TestCase):
    def test_active_brand_exports_declare_obb_and_yaml_class_mapping(self) -> None:
        registry = load_dataset_sources()
        brand_sources = [row for row in registry.sources if row.name.startswith("brand_")]
        self.assertEqual(len(brand_sources), 4)
        self.assertTrue(
            all(row.annotation_format is AnnotationFormat.YOLO_OBB for row in brand_sources)
        )
        self.assertTrue(
            all(row.class_mapping_source == "archive_data_yaml" for row in brand_sources)
        )

    def test_brand21_directory_declares_bbox_and_data_yaml_mapping(self) -> None:
        source = load_dataset_sources().get("unilever_brand21_20261019")
        self.assertEqual(source.annotation_format, AnnotationFormat.YOLO_BBOX)
        self.assertEqual(source.class_mapping_source, "data_yaml")
        self.assertEqual(source.path, (Path("data/raw/brand21_20261019")).resolve())

    def test_match_rejects_undeclared_source_and_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            source.write_bytes(b"fixture")
            payload = {
                "schema_version": 1,
                "sources": [
                    {
                        "name": "fixture",
                        "path": str(source),
                        "annotation_format": "yolo_bbox",
                        "class_mapping_source": "archive_data_yaml",
                        "sha256": "a" * 64,
                    }
                ],
            }
            registry = DatasetSourceRegistry.from_mapping(payload)
            self.assertEqual(
                registry.match(source).annotation_format, AnnotationFormat.YOLO_BBOX
            )
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                registry.match(source, sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "not declared"):
                registry.match(root / "other.zip")

    def test_duplicate_paths_and_ambiguous_formats_are_rejected(self) -> None:
        row = {
            "name": "first",
            "path": "data/source.zip",
            "annotation_format": "coordinate_count_guess",
            "class_mapping_source": "archive_data_yaml",
        }
        with self.assertRaisesRegex(ValueError, "Unsupported annotation format"):
            DatasetSourceRegistry.from_mapping(
                {"schema_version": 1, "sources": [row]}
            )
        second = {**row, "name": "second", "annotation_format": "yolo_obb"}
        first = {**row, "annotation_format": "yolo_obb"}
        with self.assertRaisesRegex(ValueError, "Duplicate dataset source path"):
            DatasetSourceRegistry.from_mapping(
                {"schema_version": 1, "sources": [first, second]}
            )


if __name__ == "__main__":
    unittest.main()
