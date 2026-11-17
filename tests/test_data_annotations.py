from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.annotations import (
    AnnotationFormat,
    load_yolo_class_names,
    parse_yolo_annotation,
    validate_coco_document,
)
from src.data.crops import bbox_to_crop_box, make_crop_filename


class ExplicitAnnotationParserTests(unittest.TestCase):
    def test_same_eight_coordinates_require_declared_geometry(self) -> None:
        row = "0 0.1 0.2 0.8 0.2 0.8 0.9 0.1 0.9"
        obb = parse_yolo_annotation(row, ["AXE"], AnnotationFormat.YOLO_OBB)
        polygon = parse_yolo_annotation(
            row, ["AXE"], AnnotationFormat.YOLO_SEGMENTATION
        )
        self.assertEqual(obb.box, polygon.box)
        self.assertEqual(obb.annotation_format, AnnotationFormat.YOLO_OBB)
        self.assertEqual(
            polygon.annotation_format, AnnotationFormat.YOLO_SEGMENTATION
        )
        with self.assertRaisesRegex(ValueError, "exactly x_center"):
            parse_yolo_annotation(row, ["AXE"], AnnotationFormat.YOLO_BBOX)

    def test_obb_and_segmentation_have_distinct_shape_validation(self) -> None:
        six_coordinates = "0 0.1 0.1 0.8 0.1 0.5 0.9"
        parsed = parse_yolo_annotation(
            six_coordinates, ["AXE"], AnnotationFormat.YOLO_SEGMENTATION
        )
        self.assertAlmostEqual(parsed.box.width, 0.7)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            parse_yolo_annotation(
                six_coordinates, ["AXE"], AnnotationFormat.YOLO_OBB
            )

    def test_class_order_comes_from_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.yaml"
            path.write_text(
                yaml.safe_dump({"nc": 2, "names": {0: "AXE", 1: "VIM"}}),
                encoding="utf-8",
            )
            self.assertEqual(load_yolo_class_names(path), ("AXE", "VIM"))
            path.write_text(
                yaml.safe_dump({"nc": 3, "names": ["AXE", "VIM"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "declares nc"):
                load_yolo_class_names(path)

    def test_coco_schema_and_crop_clipping(self) -> None:
        payload = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 4, "name": "AXE"}],
        }
        self.assertIs(validate_coco_document(payload), payload)
        self.assertEqual(bbox_to_crop_box([-2, 1, 6, 5], 10, 10, 0), (0, 1, 4, 6))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_coco_document(
                {**payload, "categories": [{"id": 1}, {"id": 1}]}
            )

    def test_crop_names_are_short_deterministic_and_traceable_elsewhere(self) -> None:
        first = make_crop_filename(
            "a" * 300 + ".jpg",
            123,
            456,
            7,
            source_zip="source.zip",
            original_split="train",
        )
        second = make_crop_filename(
            "a" * 300 + ".jpg",
            123,
            456,
            7,
            source_zip="source.zip",
            original_split="train",
        )
        self.assertEqual(first, second)
        self.assertLess(len(first), 64)


if __name__ == "__main__":
    unittest.main()
