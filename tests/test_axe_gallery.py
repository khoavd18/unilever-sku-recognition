from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts.build_clean_axe_dataset import EXPECTED_SKUS
from scripts.test_axe_gallery import (
    audit_gallery_query_inputs,
    choose_visual_queries,
    protocol_metrics,
    search_full_gallery,
)


CLASS_TO_IDX = {name: index for index, name in enumerate(EXPECTED_SKUS)}


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AxeGallerySyntheticTests(unittest.TestCase):
    def test_full_gallery_and_max_per_sku_ranking_agree(self) -> None:
        gallery = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        query = torch.tensor([[0.9, 0.1]])
        labels = list(EXPECTED_SKUS)
        result = search_full_gallery(query, gallery, labels, labels)
        self.assertEqual(result["nearest_gallery_indices"], [0])
        self.assertEqual(result["predicted_labels"], [labels[0]])
        self.assertEqual(result["ranked_sku_candidates"][0][0]["sku"], labels[0])
        self.assertAlmostEqual(result["nearest_scores"][0], 0.9938837, places=6)

    def test_gallery_metrics_are_computed_from_supplied_search_predictions(self) -> None:
        targets = list(EXPECTED_SKUS)
        predictions = list(EXPECTED_SKUS)
        predictions[-1] = EXPECTED_SKUS[0]
        metrics = protocol_metrics(targets, predictions, EXPECTED_SKUS)
        self.assertEqual(metrics["top1_sku_accuracy"], 0.8)
        self.assertEqual(metrics["macro_recall"], 0.8)
        self.assertEqual(metrics["per_sku"][EXPECTED_SKUS[-1]]["correct"], 0)
        self.assertEqual(metrics["per_sku"][EXPECTED_SKUS[-1]]["total"], 1)

    def test_visual_sampling_is_reproducible_and_keeps_all_errors(self) -> None:
        correctness = [True, False, True, True, False, True]
        first = choose_visual_queries(correctness, 2, seed=42)
        second = choose_visual_queries(correctness, 2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first[0], [1, 4])
        self.assertEqual(len(first[1]), 2)

    def _make_dataset(self, root: Path, overlapping_family: bool = False) -> None:
        (root / "manifests").mkdir(parents=True)
        (root / "class_mapping.json").write_text(
            json.dumps({"class_to_idx": CLASS_TO_IDX}), encoding="utf-8"
        )
        manifest_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
        crop_rows: list[dict[str, str]] = []
        for split in ("train", "val"):
            for index, sku in enumerate(EXPECTED_SKUS):
                image_path = root / split / sku / f"{split}_{index}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                color = (index * 30, 10 if split == "train" else 20, 90)
                Image.new("RGB", (8, 8), color).save(image_path)
                manifest_rows[split].append(
                    {"image_path": str(image_path), "class_name": sku, "split": split}
                )
                family = f"family_{index}_{split}"
                if overlapping_family and index == 0:
                    family = "confirmed_overlap"
                crop_rows.append(
                    {
                        "generated_crop_path": str(image_path),
                        "class_name": sku,
                        "class_index": str(index),
                        "split": split,
                        "source_zip": f"source_{index}.zip",
                        "family_id": family,
                        "source_image_id": f"{split}_{index}",
                        "source_annotation_id": f"annotation_{split}_{index}",
                    }
                )
        fields = ("image_path", "class_name", "split")
        _write_csv(root / "manifests" / "train.csv", fields, manifest_rows["train"])
        _write_csv(root / "manifests" / "val.csv", fields, manifest_rows["val"])
        _write_csv(
            root / "crop_manifest.csv",
            (
                "generated_crop_path",
                "class_name",
                "class_index",
                "split",
                "source_zip",
                "family_id",
                "source_image_id",
                "source_annotation_id",
            ),
            crop_rows,
        )

    def test_synthetic_manifest_audit_reports_counts_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_dataset(root)
            audit, gallery_rows, query_rows = audit_gallery_query_inputs(
                root,
                CLASS_TO_IDX,
                expected_gallery_count=None,
                expected_query_count=None,
            )
        self.assertEqual(len(gallery_rows), 5)
        self.assertEqual(len(query_rows), 5)
        self.assertEqual(audit["train_val_source_family_overlap"], 0)
        self.assertEqual(audit["train_val_exact_duplicate_groups"], 0)
        self.assertFalse(audit["test_data_loaded"])

    def test_synthetic_manifest_audit_blocks_family_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_dataset(root, overlapping_family=True)
            with self.assertRaisesRegex(ValueError, "source-family overlap"):
                audit_gallery_query_inputs(
                    root,
                    CLASS_TO_IDX,
                    expected_gallery_count=None,
                    expected_query_count=None,
                )

    def test_synthetic_manifest_audit_blocks_exact_cross_split_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_dataset(root)
            train_path = root / "train" / EXPECTED_SKUS[0] / "train_0.png"
            val_path = root / "val" / EXPECTED_SKUS[0] / "val_0.png"
            shutil.copyfile(train_path, val_path)
            with self.assertRaisesRegex(ValueError, "byte-identical crops"):
                audit_gallery_query_inputs(
                    root,
                    CLASS_TO_IDX,
                    expected_gallery_count=None,
                    expected_query_count=None,
                )


if __name__ == "__main__":
    unittest.main()
