from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts.prepare_coco_crops import (
    WINDOWS_SAFE_PATH_LIMIT,
    bbox_to_crop_box,
    build_category_mapping,
    make_crop_filename,
    prepare_coco_crops,
    sanitize_class_name,
    validate_dataset_root,
    windows_safe_crop_path,
)
from src.datasets.product_dataset import ProductDataset, build_transforms


def write_fixture_split(root: Path, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (40, 100, 180)).save(split_dir / "source.jpg")
    coco = {
        "images": [
            {"id": 10, "file_name": "source.jpg", "width": 40, "height": 30},
            {"id": 11, "file_name": "missing.jpg", "width": 20, "height": 20},
        ],
        "categories": [
            {"id": 0, "name": "Stock"},
            {"id": 1, "name": "0"},
            {"id": 2, "name": "SKU/Blue"},
        ],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": 2, "bbox": [2, 3, 20, 10], "segmentation": []},
            {"id": 101, "image_id": 10, "category_id": 2, "bbox": [35, 20, 20, 20], "segmentation": []},
            {"id": 102, "image_id": 10, "category_id": 0, "bbox": [2, 3, 20, 10], "segmentation": []},
            {"id": 103, "image_id": 10, "category_id": 1, "bbox": [2, 3, 20, 10], "segmentation": []},
            {"id": 104, "image_id": 10, "category_id": 2, "bbox": [2, 3, -1, 10], "segmentation": []},
            {"id": 105, "image_id": 11, "category_id": 2, "bbox": [2, 3, 10, 10], "segmentation": []},
        ],
    }
    with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)


def write_fixture_dataset(root: Path) -> None:
    for split in ("train", "valid", "test"):
        write_fixture_split(root, split)


def write_long_path_fixture_dataset(root: Path) -> tuple[str, dict[str, str]]:
    long_sku = "AXE_" + "Extremely_Long_SKU_Name_" * 18
    original_filenames: dict[str, str] = {}
    source_families = []
    for split_index, split in enumerate(("train", "valid", "test")):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"roboflow_{split}_" + "source-identity-" * 7 + "_aug2_jpg.rf.MixedHashValue.jpg"
        )
        original_filenames[split] = filename
        Image.new("RGB", (40, 30), (80, 120, 160)).save(split_dir / filename)
        image_id = 100 + split_index
        annotation_id = 200 + split_index
        coco = {
            "images": [
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": 40,
                    "height": 30,
                }
            ],
            "categories": [
                {"id": 0, "name": "Stock"},
                {"id": 2, "name": long_sku, "supercategory": "Stock"},
            ],
            "annotations": [
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 2,
                    "bbox": [2, 3, 20, 10],
                    "segmentation": [[2, 3, 22, 3, 22, 13, 2, 13]],
                }
            ],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )
        source_families.append(
            {
                "family_id": f"family-{split_index}",
                "images": [
                    {
                        "output_image_id": image_id,
                        "destination_file_name": filename,
                        "source_archive": f"axe-source-{split_index}.coco.zip",
                        "source_split": split,
                        "source_file_name": filename,
                        "source_image_id": 900 + split_index,
                        "sku": long_sku,
                    }
                ],
            }
        )
    (root / "source_family_manifest.json").write_text(
        json.dumps({"families": source_families}), encoding="utf-8"
    )
    return long_sku, original_filenames


class PrepareCocoCropsTests(unittest.TestCase):
    def test_bbox_padding_and_boundary_clamping(self) -> None:
        self.assertEqual(bbox_to_crop_box([10, 10, 20, 10], 100, 50, 0.10), (8, 9, 32, 21))
        self.assertEqual(bbox_to_crop_box([90, 90, 20, 20], 100, 100, 0.05), (89, 89, 100, 100))
        self.assertIsNone(bbox_to_crop_box([1, 1, -2, 5], 100, 100, 0.05))
        self.assertIsNone(bbox_to_crop_box([1, 1, 2], 100, 100, 0.05))

    def test_category_mapping_skips_stock_and_zero(self) -> None:
        mapping, skipped = build_category_mapping(
            {
                "train": {
                    "categories": [
                        {"id": 0, "name": "Stock"},
                        {"id": 1, "name": "0"},
                        {"id": 2, "name": "SKU/Blue"},
                    ]
                }
            }
        )
        self.assertEqual(list(mapping), [2])
        self.assertEqual(mapping[2]["directory_name"], "SKU_Blue")
        self.assertEqual({item["category_name"] for item in skipped}, {"Stock", "0"})
        self.assertEqual(sanitize_class_name(' x<>:"/\\|?* '), "x_________")

    def test_unique_traceable_filenames(self) -> None:
        kwargs = {
            "class_index": 7,
            "source_zip": "source-export.coco.zip",
            "original_split": "train",
            "original_image_id": 12,
        }
        first = make_crop_filename("a-very-long-roboflow-source-name.jpg", 42, 100, **kwargs)
        repeated = make_crop_filename("a-very-long-roboflow-source-name.jpg", 42, 100, **kwargs)
        second = make_crop_filename("a-very-long-roboflow-source-name.jpg", 42, 101, **kwargs)
        other_source = make_crop_filename("another-source-name.jpg", 42, 100, **kwargs)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, other_source)
        self.assertRegex(first, r"^c007_i000042_a000100_[0-9a-f]{10}\.jpg$")
        self.assertNotIn("roboflow", first)

    def test_windows_path_guard_uses_compact_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            normal_name = make_crop_filename(
                "source.jpg", 1, 2, 0, source_zip="source.zip", original_split="train"
            )
            prefix_length = len(str((output / "train").resolve()))
            directory_length = 242 - prefix_length - len(normal_name) - 2
            self.assertGreater(directory_length, 0)
            path = windows_safe_crop_path(
                output,
                "train",
                "d" * directory_length,
                source_file_name="source.jpg",
                image_id=1,
                annotation_id=2,
                class_index=0,
                source_zip="source.zip",
                original_split="train",
                original_image_id=1,
            )
            self.assertLess(len(str(path.resolve())), WINDOWS_SAFE_PATH_LIMIT)
            self.assertRegex(path.name, r"^c0_i1_a2_[0-9a-f]{8}\.jpg$")

    def test_dry_run_writes_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "crops"
            write_fixture_dataset(source)
            report = prepare_coco_crops(source, output, padding=0.05, min_size=5, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["total_successful_crops"], 6)
            self.assertFalse(output.exists())

    def test_double_nested_source_is_reported_without_moving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "Stock_Segmentation_export"
            write_fixture_dataset(nested)
            with self.assertRaisesRegex(ValueError, "double-nested"):
                validate_dataset_root(root)
            self.assertTrue((nested / "train" / "_annotations.coco.json").is_file())

    def test_crops_preserve_splits_and_load_with_dataloader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "crops"
            write_fixture_dataset(source)
            preview_dir = root / "preview"
            report = prepare_coco_crops(
                source, output, padding=0.05, min_size=5, preview_count=1, preview_dir=preview_dir
            )
            self.assertEqual(report["total_successful_crops"], 6)
            self.assertTrue((output / "train" / "SKU_Blue").is_dir())
            self.assertTrue((output / "valid" / "SKU_Blue").is_dir())
            self.assertTrue((output / "test" / "SKU_Blue").is_dir())
            self.assertTrue((output / "class_mapping.json").is_file())
            self.assertTrue((output / "crop_manifest.csv").is_file())
            self.assertTrue((output / "crop_manifest.json").is_file())
            self.assertEqual(report["preview_images_written"], 1)
            self.assertEqual(len(list(preview_dir.glob("*.jpg"))), 1)
            with (output / "crop_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                crop_rows = list(csv.DictReader(handle))
            self.assertEqual(len(crop_rows), 6)
            self.assertEqual({row["source_annotation_id"] for row in crop_rows}, {"100", "101"})
            self.assertEqual({row["split"] for row in crop_rows}, {"train", "val", "test"})
            self.assertEqual({row["class_index"] for row in crop_rows}, {"0"})
            self.assertTrue(all(row["generated_crop_path"] == row["image_path"] for row in crop_rows))
            with (output / "manifests" / "val.csv").open("r", encoding="utf-8", newline="") as handle:
                val_rows = list(csv.DictReader(handle))
            self.assertEqual(len(val_rows), 2)
            self.assertTrue(all(f"{os.sep}valid{os.sep}" in row["image_path"] for row in val_rows))
            dataset = ProductDataset(
                output / "manifests" / "train.csv", {"SKU/Blue": 0}, build_transforms(64, train=False)
            )
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
            self.assertEqual(tuple(batch[0].shape), (2, 3, 64, 64))
            self.assertTrue(torch.isfinite(batch[0]).all())

    def test_long_names_are_short_deterministic_and_fully_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            long_sku, original_filenames = write_long_path_fixture_dataset(source)
            output_a = root / ("output_a_" + "x" * 72)
            output_b = root / ("output_b_" + "x" * 72)
            source_mtimes = {
                path: path.stat().st_mtime_ns
                for split in ("train", "valid", "test")
                for path in (source / split).glob("*.jpg")
            }

            prepare_coco_crops(source, output_a, min_size=5, preview_count=0)
            prepare_coco_crops(source, output_b, min_size=5, preview_count=0)

            crops_a = sorted(path for path in output_a.rglob("*.jpg"))
            crops_b = sorted(path for path in output_b.rglob("*.jpg"))
            self.assertEqual(len(crops_a), 3)
            self.assertEqual([path.name for path in crops_a], [path.name for path in crops_b])
            for crop_path in crops_a:
                self.assertTrue(crop_path.parent.is_dir())
                self.assertLess(len(str(crop_path.resolve())), WINDOWS_SAFE_PATH_LIMIT)
                self.assertRegex(
                    crop_path.name, r"^c000_i00010[0-2]_a00020[0-2]_[0-9a-f]{10}\.jpg$"
                )
                self.assertNotIn("roboflow", crop_path.name)
                self.assertNotIn("Extremely", crop_path.name)

            mapping = json.loads((output_a / "class_mapping.json").read_text(encoding="utf-8"))
            category = mapping["categories"][0]
            self.assertEqual(category["category_name"], long_sku)
            self.assertEqual(mapping["class_to_idx"], {long_sku: 0})
            self.assertTrue(category["directory_name_was_shortened"])
            self.assertLess(len(category["directory_name"]), len(category["original_directory_name"]))

            with (output_a / "crop_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            json_rows = json.loads((output_a / "crop_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(len(json_rows), 3)
            for row in csv_rows:
                source_split = row["original_split"]
                split_index = ("train", "valid", "test").index(source_split)
                self.assertEqual(row["original_sku_name"], long_sku)
                self.assertEqual(row["class_index"], "0")
                self.assertEqual(row["source_zip"], f"axe-source-{split_index}.coco.zip")
                self.assertEqual(row["original_filename"], original_filenames[source_split])
                self.assertEqual(row["family_id"], f"family-{split_index}")
                self.assertEqual(row["original_source_image_id"], str(900 + split_index))
                self.assertEqual(row["generated_crop_path"], row["image_path"])
            self.assertEqual(
                source_mtimes,
                {path: path.stat().st_mtime_ns for path in source_mtimes},
            )

    def test_duplicate_generated_filename_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "crops"
            write_fixture_dataset(source)
            annotation_path = source / "train" / "_annotations.coco.json"
            coco = json.loads(annotation_path.read_text(encoding="utf-8"))
            coco["annotations"].append(dict(coco["annotations"][0]))
            annotation_path.write_text(json.dumps(coco), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "filename collision"):
                prepare_coco_crops(
                    source, output, padding=0.05, min_size=5, dry_run=True, preview_count=0
                )
            self.assertFalse(output.exists())

    def test_existing_crops_are_not_overwritten_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "crops"
            write_fixture_dataset(source)
            prepare_coco_crops(source, output, padding=0.05, min_size=5, preview_count=0)
            crop = next((output / "train" / "SKU_Blue").glob("*.jpg"))
            initial_mtime = crop.stat().st_mtime_ns
            report = prepare_coco_crops(source, output, padding=0.05, min_size=5, preview_count=0)
            self.assertEqual(crop.stat().st_mtime_ns, initial_mtime)
            self.assertEqual(report["splits"]["train"]["created_crops"], 0)
            self.assertEqual(report["splits"]["train"]["existing_crops_not_overwritten"], 2)


if __name__ == "__main__":
    unittest.main()
