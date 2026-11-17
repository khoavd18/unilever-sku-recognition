from __future__ import annotations

import tempfile
import unittest
import warnings
import copy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml
from PIL import Image

from scripts.prepare_brand_5brands_yolo import (
    TARGET_BRANDS,
    annotation_format,
    assign_splits,
    audit_zip,
    coordinates_to_box,
    infer_target_brand,
)
from src.data.annotations import AnnotationFormat


class FiveBrandPreparationTests(unittest.TestCase):
    def test_mapping_uses_verified_class_name_not_source_path(self) -> None:
        self.assertEqual(infer_target_brand("VIM_VienTreoBonCau"), "VIM")
        self.assertEqual(infer_target_brand("Lifebuoy_xabong"), "LIFEBUOY")
        self.assertEqual(infer_target_brand("REXONA"), "REXONA")
        self.assertIsNone(infer_target_brand("DOVE_REXONA-looking-name"))
        self.assertIsNone(infer_target_brand("CLEAR"))

    def test_ordinary_bbox_is_preserved(self) -> None:
        box = coordinates_to_box(
            [0.4, 0.5, 0.2, 0.6],
            "AXE",
            7,
            "AXE_Product",
            1,
            AnnotationFormat.YOLO_BBOX,
        )
        self.assertEqual(box.source_format, "ordinary_bbox")
        self.assertAlmostEqual(box.x_center, 0.4)
        self.assertAlmostEqual(box.y_center, 0.5)
        self.assertAlmostEqual(box.width, 0.2)
        self.assertAlmostEqual(box.height, 0.6)

    def test_obb_is_converted_with_minmax_xy(self) -> None:
        box = coordinates_to_box(
            [0.2, 0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.7],
            "HAZELINE",
            18,
            "HAZELINE_Product",
            1,
            AnnotationFormat.YOLO_OBB,
        )
        self.assertEqual(box.source_format, "obb_quadrilateral")
        self.assertAlmostEqual(box.x_center, 0.5)
        self.assertAlmostEqual(box.y_center, 0.5)
        self.assertAlmostEqual(box.width, 0.6)
        self.assertAlmostEqual(box.height, 0.8)

    def test_segmentation_polygon_is_converted_to_enclosing_box(self) -> None:
        values = [0.1, 0.2, 0.3, 0.1, 0.7, 0.2, 0.8, 0.8, 0.4, 0.9]
        self.assertEqual(
            annotation_format(AnnotationFormat.YOLO_SEGMENTATION),
            "segmentation_polygon",
        )
        box = coordinates_to_box(
            values,
            "VIM",
            3,
            "VIM_Product",
            1,
            AnnotationFormat.YOLO_SEGMENTATION,
        )
        self.assertAlmostEqual(box.x_center, 0.45)
        self.assertAlmostEqual(box.y_center, 0.5)
        self.assertAlmostEqual(box.width, 0.7)
        self.assertAlmostEqual(box.height, 0.8)

    def test_duplicate_yaml_entries_are_read_by_zipinfo_and_saved_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "source.zip"
            yaml_payload = yaml.safe_dump(
                {
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images",
                    "names": list(TARGET_BRANDS),
                    "annotation_format": "yolo_obb",
                },
                sort_keys=False,
            )
            image_path = root / "image.jpg"
            Image.new("RGB", (20, 10), "white").save(image_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                    archive.writestr("data.yaml", yaml_payload)
                    archive.writestr("train/images/example.jpg", image_path.read_bytes())
                    archive.writestr(
                        "train/labels/example.txt",
                        "1 0.1 0.1 0.8 0.1 0.8 0.9 0.1 0.9\n",
                    )
                    archive.writestr("data.yaml", yaml_payload)
            report, records = audit_zip(archive_path, root / "yaml_copies")
            self.assertFalse(report["blocking_issues"])
            self.assertTrue(report["yaml_consistent_within_archive"])
            self.assertEqual(len(report["yaml_entries"]), 2)
            self.assertEqual(len(list((root / "yaml_copies").glob("*.yaml"))), 2)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].boxes[0].brand, "VIM")

    def test_conflicting_duplicate_yaml_entries_block_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "conflict.zip"
            first = yaml.safe_dump(
                {"train": "train/images", "val": "valid/images", "names": ["AXE"], "annotation_format": "yolo_obb"}
            )
            second = yaml.safe_dump(
                {"train": "train/images", "val": "valid/images", "names": ["VIM"], "annotation_format": "yolo_obb"}
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                    archive.writestr("data.yaml", first)
                    archive.writestr("data.yaml", second)
            report, records = audit_zip(archive_path, root / "yaml_copies")
            self.assertFalse(records)
            self.assertFalse(report["yaml_consistent_within_archive"])
            self.assertTrue(
                any("disagree" in issue for issue in report["blocking_issues"]),
                report["blocking_issues"],
            )

    def test_conflicting_duplicate_label_entries_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "labels.zip"
            image_path = root / "image.jpg"
            Image.new("RGB", (20, 10), "white").save(image_path)
            yaml_payload = yaml.safe_dump(
                {"train": "train/images", "val": "valid/images", "names": ["AXE"], "annotation_format": "yolo_obb"}
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                    archive.writestr("data.yaml", yaml_payload)
                    archive.writestr("train/images/example.jpg", image_path.read_bytes())
                    archive.writestr(
                        "train/labels/example.txt",
                        "0 0.1 0.1 0.8 0.1 0.8 0.9 0.1 0.9\n",
                    )
                    archive.writestr(
                        "train/labels/example.txt",
                        "0 0.2 0.2 0.7 0.2 0.7 0.8 0.2 0.8\n",
                    )
            report, records = audit_zip(archive_path, root / "yaml_copies")
            self.assertFalse(records)
            self.assertTrue(
                any("conflicting duplicate labels" in issue for issue in report["blocking_issues"]),
                report["blocking_issues"],
            )

    def test_grouped_split_balancing_is_deterministic_and_keeps_all_brands(self) -> None:
        groups = [
            {
                "group_id": f"family-{index:03d}",
                "units": [],
                "image_count": 1,
                "family_ids": [f"family-{index:03d}"],
                "brand_box_counts": {
                    brand: 1 + ((index + brand_index) % 3)
                    for brand_index, brand in enumerate(TARGET_BRANDS)
                },
                "forced_split": "val" if index == 0 else ("test" if index == 1 else None),
                "force_reasons": [],
            }
            for index in range(100)
        ]
        first, second = copy.deepcopy(groups), copy.deepcopy(groups)
        report_first = assign_splits(first, seed=42)
        report_second = assign_splits(second, seed=42)
        self.assertEqual(
            [group["output_split"] for group in first],
            [group["output_split"] for group in second],
        )
        self.assertEqual(report_first["actual_image_counts"], report_second["actual_image_counts"])
        self.assertEqual(first[0]["output_split"], "val")
        self.assertEqual(first[1]["output_split"], "test")
        for split in ("train", "val", "test"):
            for brand in TARGET_BRANDS:
                self.assertGreater(report_first["actual_brand_box_counts"][split][brand], 0)
        self.assertLessEqual(abs(report_first["actual_image_counts"]["train"] - 70), 2)


if __name__ == "__main__":
    unittest.main()
