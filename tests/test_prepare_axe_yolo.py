from __future__ import annotations

import copy
import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from scripts.prepare_axe_yolo import (
    QUARANTINE_REASONS,
    assign_grouped_splits,
    convert_label_text,
    polygon_to_box,
    write_quarantine,
)


class AxeYoloPreparationTests(unittest.TestCase):
    def test_polygon_is_converted_to_enclosing_detection_box(self) -> None:
        box = polygon_to_box(
            [0.2, 0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.7],
            source_line=1,
            source_class_id=0,
            source_class_name="AXE_Product",
        )
        self.assertAlmostEqual(box.x_center, 0.5)
        self.assertAlmostEqual(box.y_center, 0.5)
        self.assertAlmostEqual(box.width, 0.6)
        self.assertAlmostEqual(box.height, 0.8)
        self.assertTrue(box.yolo_row().startswith("0 "))

    def test_only_axe_prefix_is_mapped_and_non_axe_is_removed(self) -> None:
        names = ["AXE_Product", "LIFEBUOY_Product"]
        text = "\n".join(
            [
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9",
                "1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            ]
        )
        boxes, removed, rejected, counts = convert_label_text(text, names, "synthetic.txt")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].source_class_name, "AXE_Product")
        self.assertEqual(removed, 1)
        self.assertFalse(rejected)
        self.assertEqual(sum(counts.values()), 2)

    def test_degenerate_and_out_of_range_axe_polygons_are_rejected(self) -> None:
        text = "\n".join(
            [
                "0 0.5 0.1 0.5 0.2 0.5 0.3",
                "0 -0.1 0.1 0.8 0.2 0.7 0.9",
            ]
        )
        boxes, _, rejected, _ = convert_label_text(text, ["AXE_Product"], "bad.txt")
        self.assertFalse(boxes)
        self.assertEqual(len(rejected), 2)
        self.assertIn("degenerate", rejected[0]["reason"])
        self.assertIn("outside", rejected[1]["reason"])

    def test_grouped_split_is_deterministic_and_respects_model2_holdouts(self) -> None:
        groups = [
            {
                "group_id": f"family_{index}",
                "family_ids": [f"family_{index}"],
                "image_count": 2 if index < 2 else 1,
                "units": [],
                "forced_split": None,
            }
            for index in range(20)
        ]
        holds = {"family_0": "val", "family_1": "test", "family_2": "train"}
        first = copy.deepcopy(groups)
        second = copy.deepcopy(groups)
        report_first = assign_grouped_splits(first, holds, seed=42)
        report_second = assign_grouped_splits(second, holds, seed=42)
        self.assertEqual(
            [group["output_split"] for group in first],
            [group["output_split"] for group in second],
        )
        self.assertEqual(first[0]["output_split"], "val")
        self.assertEqual(first[1]["output_split"], "test")
        self.assertEqual(report_first["actual_image_counts"], report_second["actual_image_counts"])

    def test_model2_val_test_conflict_is_blocked(self) -> None:
        groups = [
            {
                "group_id": "linked",
                "family_ids": ["a", "b"],
                "image_count": 1,
                "units": [],
                "forced_split": None,
            }
        ]
        with self.assertRaisesRegex(ValueError, "conflicting Model 2"):
            assign_grouped_splits(groups, {"a": "val", "b": "test"}, seed=42)

    def test_quarantine_writer_preserves_raw_image_and_label(self) -> None:
        from scripts.prepare_axe_yolo import SourceRecord

        prefix = next(iter(QUARANTINE_REASONS))
        image_buffer = io.BytesIO()
        Image.new("RGB", (12, 10), "red").save(image_buffer, format="JPEG")
        label_text = "25 0.1 0.1 0.9 0.1 0.9 0.9\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "source.zip"
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
                image_member = f"train/images/{prefix}_jpg.rf.hash.jpg"
                label_member = f"train/labels/{prefix}_jpg.rf.hash.txt"
                archive.writestr(image_member, image_buffer.getvalue())
                archive.writestr(label_member, label_text)
            record = SourceRecord(
                image_member=image_member,
                label_member=label_member,
                source_split="train",
                source_filename=Path(image_member).name,
                family_id=prefix,
                image_sha256="hash",
                width=12,
                height=10,
                quarantine_prefix=prefix,
            )
            output = root / "output"
            with ZipFile(archive_path) as archive:
                rows = write_quarantine(output, [record], archive)
            self.assertEqual(len(rows), 1)
            self.assertEqual((output / "quarantine" / "labels" / "q001.txt").read_text(), label_text)
            self.assertTrue((output / "quarantine" / "images" / "q001.jpg").is_file())
            self.assertTrue((output / "quarantine" / "contact_sheet.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
