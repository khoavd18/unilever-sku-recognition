from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_dataset_leakage import (
    class_distribution,
    exact_duplicate_pairs,
    hamming_distance,
    normalize_source_family,
    sha256_file,
)


class LeakageAuditUtilityTests(unittest.TestCase):
    def test_conservative_roboflow_family_normalization(self) -> None:
        self.assertEqual(normalize_source_family("abc_aug1_jpg.rf.0123456789abcdef.jpg"), "abc")
        self.assertEqual(normalize_source_family("abc-augmentation_2.rf.abcdef0123456789.png"), "abc")
        self.assertEqual(normalize_source_family("product_august.jpg"), "product_august")
        self.assertEqual(normalize_source_family("not_augmented_2024.jpg"), "not_augmented_2024")

    def test_exact_duplicate_cross_split_and_conflicting_label(self) -> None:
        records = [
            {"asset_type": "crop", "path": "train/a.jpg", "split": "train", "label": "SKU_A", "sha256": "same"},
            {"asset_type": "crop", "path": "val/b.jpg", "split": "val", "label": "SKU_A", "sha256": "same"},
            {"asset_type": "crop", "path": "test/c.jpg", "split": "test", "label": "SKU_B", "sha256": "same"},
        ]
        pairs, summary = exact_duplicate_pairs(records)
        self.assertEqual(summary["cross_split_pairs"], 3)
        self.assertEqual(summary["conflicting_label_pairs"], 2)
        self.assertEqual({pair["pair_scope"] for pair in pairs}, {"cross_split"})

    def test_hamming_distance(self) -> None:
        self.assertEqual(hamming_distance(0b0000, 0b1111), 4)
        self.assertEqual(hamming_distance(0xFF00, 0xF000), 4)

    def test_crop_to_source_cross_split_mapping(self) -> None:
        rows = [
            {"split": "train", "class_name": "SKU_A", "source_image_path": "D:/source/train/a.jpg", "source_annotation_id": "1"},
            {"split": "val", "class_name": "SKU_A", "source_image_path": "D:/source/valid/a.jpg", "source_annotation_id": "1"},
        ]
        distribution, summary = class_distribution(rows, ["SKU_A"])
        self.assertEqual(distribution[0]["train_crops"], 1)
        self.assertEqual(distribution[0]["val_crops"], 1)
        self.assertEqual(summary["duplicate_annotation_records"], 0)

    def test_hash_audit_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            Image.new("RGB", (16, 16), "red").save(path)
            before = path.stat().st_mtime_ns
            time.sleep(0.01)
            digest = sha256_file(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()

