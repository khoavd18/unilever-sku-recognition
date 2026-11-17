from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_clean_axe_dataset import (
    EXPECTED_SKUS,
    SPLITS,
    build_clean_axe_dataset,
    normalize_source_family,
    validate_ratios,
)


def _write_fixture_archives(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for zip_offset, sku in enumerate(EXPECTED_SKUS):
        zip_path = root / f"axe_{zip_offset + 1:02d}.coco.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for split_index, split in enumerate(SPLITS):
                images = []
                annotations = []
                for local_index in range(2):
                    sequence = split_index * 2 + local_index
                    if sequence == 0 and zip_offset in (0, 1):
                        source = f"shared-family_aug{zip_offset + 1}_jpg.rf.Hash{zip_offset}.jpg"
                    else:
                        source = (
                            f"zip{zip_offset}_source{sequence}_aug1_jpg."
                            f"rf.Hash{split_index}{local_index}.jpg"
                        )
                    image_id = split_index * 10 + local_index + 1
                    images.append(
                        {"id": image_id, "file_name": source, "width": 32, "height": 24}
                    )
                    annotations.append(
                        {
                            "id": 100 + local_index,
                            "image_id": image_id,
                            "category_id": 2,
                            "bbox": [1, 2, 20, 18],
                            "area": 360,
                            "segmentation": [[1, 2, 21, 2, 21, 20, 1, 20]],
                            "iscrowd": 0,
                        }
                    )
                    archive.writestr(f"{split}/{source}", f"fixture-{zip_offset}-{sequence}".encode())
                # Both excluded category forms are present and must not leak into output.
                annotations.extend(
                    [
                        {
                            "id": 200,
                            "image_id": images[0]["id"],
                            "category_id": 0,
                            "bbox": [0, 0, 1, 1],
                            "segmentation": [],
                        },
                        {
                            "id": 201,
                            "image_id": images[0]["id"],
                            "category_id": 1,
                            "bbox": [0, 0, 1, 1],
                            "segmentation": [],
                        },
                    ]
                )
                coco = {
                    "images": images,
                    "annotations": annotations,
                    "categories": [
                        {"id": 0, "name": "Stock", "supercategory": "none"},
                        {"id": 1, "name": "0", "supercategory": "Stock"},
                        {"id": 2, "name": sku, "supercategory": "Stock"},
                    ],
                }
                archive.writestr(
                    f"{split}/_annotations.coco.json",
                    json.dumps(coco).encode("utf-8"),
                )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildCleanAxeDatasetTests(unittest.TestCase):
    def test_conservative_family_normalization(self) -> None:
        self.assertEqual(
            normalize_source_family("folder/ABC_aug12_jpg.rf.MixedCaseHash.jpg"), "abc"
        )
        self.assertEqual(normalize_source_family("abc_aug2.jpg"), "abc")
        self.assertEqual(normalize_source_family("abc_jpg.jpg"), "abc")
        self.assertEqual(normalize_source_family("product_august.jpg"), "product_august")
        self.assertEqual(normalize_source_family("not_augmented_2024.jpg"), "not_augmented_2024")

    def test_ratio_validation(self) -> None:
        self.assertEqual(validate_ratios(0.7, 0.15, 0.15)["valid"], 0.15)
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_ratios(0.7, 0.2, 0.2)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            validate_ratios(0.8, 0.2, 0.0)

    def test_dry_run_is_deterministic_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "zips"
            output_dir = root / "clean"
            _write_fixture_archives(input_dir)
            before = {path.name: _sha256(path) for path in input_dir.glob("*.zip")}

            first = build_clean_axe_dataset(
                input_dir,
                output_dir,
                seed=19,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                dry_run=True,
            )
            second = build_clean_axe_dataset(
                input_dir,
                output_dir,
                seed=19,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                dry_run=True,
            )

            self.assertFalse(output_dir.exists())
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["source"]["images"], 30)
            self.assertEqual(first["source"]["annotations"], 30)
            self.assertEqual(first["source"]["families"], 29)
            self.assertEqual(first["source"]["cross_zip_families"], 1)
            self.assertEqual(first["assertions"]["cross_split_family_leakage"], 0)
            self.assertEqual(first["files_written"], 0)
            self.assertEqual(before, {path.name: _sha256(path) for path in input_dir.glob("*.zip")})
            for split in SPLITS:
                for sku in EXPECTED_SKUS:
                    self.assertGreater(first["splits"][split]["per_sku"][sku]["images"], 0)

    def test_build_preserves_coco_data_and_global_uniqueness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "zips"
            output_dir = root / "clean"
            _write_fixture_archives(input_dir)
            before = {path.name: _sha256(path) for path in input_dir.glob("*.zip")}

            report = build_clean_axe_dataset(
                input_dir,
                output_dir,
                seed=7,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )

            self.assertEqual(report["files_written"], 36)
            self.assertTrue((output_dir / "class_mapping.json").is_file())
            self.assertTrue((output_dir / "source_family_manifest.json").is_file())
            self.assertTrue((output_dir / "build_report.json").is_file())
            all_images = []
            all_annotations = []
            for split in SPLITS:
                annotation_path = output_dir / split / "_annotations.coco.json"
                coco = json.loads(annotation_path.read_text(encoding="utf-8"))
                self.assertEqual([item["name"] for item in coco["categories"]], list(EXPECTED_SKUS))
                all_images.extend(coco["images"])
                all_annotations.extend(coco["annotations"])
                for image in coco["images"]:
                    self.assertTrue((output_dir / split / image["file_name"]).is_file())
                    self.assertIn("__zip", image["file_name"])

            self.assertEqual(len(all_images), 30)
            self.assertEqual(len({image["id"] for image in all_images}), 30)
            self.assertEqual(len({image["file_name"] for image in all_images}), 30)
            self.assertEqual(len(all_annotations), 30)
            self.assertEqual(len({annotation["id"] for annotation in all_annotations}), 30)
            self.assertEqual(
                {annotation["category_id"] for annotation in all_annotations}, set(range(1, 6))
            )
            self.assertTrue(all(annotation["bbox"] == [1, 2, 20, 18] for annotation in all_annotations))
            self.assertTrue(
                all(
                    annotation["segmentation"] == [[1, 2, 21, 2, 21, 20, 1, 20]]
                    for annotation in all_annotations
                )
            )

            mapping = json.loads((output_dir / "class_mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["name_to_id"], {sku: index for index, sku in enumerate(EXPECTED_SKUS, 1)})
            for category in mapping["categories"]:
                self.assertTrue(
                    any(
                        image["file_name"].startswith(category["slug"] + "__zip")
                        for image in all_images
                    )
                )

            manifest = json.loads(
                (output_dir / "source_family_manifest.json").read_text(encoding="utf-8")
            )
            shared = next(
                family for family in manifest["families"] if family["family_id"] == "shared-family"
            )
            self.assertEqual(shared["occurrence_count"], 2)
            self.assertEqual(shared["zip_indices"], [1, 2])
            self.assertEqual(len({image["output_image_id"] for image in shared["images"]}), 2)
            self.assertEqual(before, {path.name: _sha256(path) for path in input_dir.glob("*.zip")})

            with self.assertRaises(FileExistsError):
                build_clean_axe_dataset(
                    input_dir,
                    output_dir,
                    seed=7,
                    train_ratio=0.5,
                    val_ratio=0.25,
                    test_ratio=0.25,
                )
            sentinel = output_dir / "keep-during-dry-run.txt"
            sentinel.write_text("untouched", encoding="utf-8")
            build_clean_axe_dataset(
                input_dir,
                output_dir,
                seed=7,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                dry_run=True,
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
            build_clean_axe_dataset(
                input_dir,
                output_dir,
                seed=7,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                overwrite=True,
            )
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
