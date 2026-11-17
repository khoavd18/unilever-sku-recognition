from __future__ import annotations

import unittest

import numpy as np

from src.retrieval.item_code import (
    evaluate_item_code_retrieval,
    validate_gallery,
    validate_gallery_runtime_compatibility,
)


def gallery_row(code: str, path: str, family: str, digest: str) -> dict[str, str]:
    return {
        "item_code": code, "item_name": code, "brand": "AXE", "image_path": path,
        "family_id": family, "image_sha256": digest, "model_version": "m",
        "catalog_version": "c", "preprocessing_version": "p",
    }


class ItemCodeRetrievalTests(unittest.TestCase):
    def test_known_answer_missing_gallery_and_macro_micro(self) -> None:
        gallery = np.array([[1, 0], [0, 1]], np.float32)
        gallery_meta = [gallery_row("A", "ga.jpg", "fa", "ha"), gallery_row("B", "gb.jpg", "fb", "hb")]
        queries = np.array([[1, 0], [0.8, 0.2], [0.5, 0.5]], np.float32)
        query_meta = [
            {"item_code": "A", "image_path": "qa.jpg", "family_id": "qa", "image_sha256": "qa"},
            {"item_code": "B", "image_path": "qb.jpg", "family_id": "qb", "image_sha256": "qb"},
            {"item_code": "C", "image_path": "qc.jpg", "family_id": "qc", "image_sha256": "qc"},
        ]
        report = evaluate_item_code_retrieval(queries, query_meta, gallery, gallery_meta)
        self.assertEqual(report["eligible_query_count"], 2)
        self.assertEqual(report["missing_gallery_query_count"], 1)
        self.assertAlmostEqual(report["micro_recall@1"], 0.5)
        self.assertAlmostEqual(report["macro_recall@1"], 0.5)
        self.assertEqual(report["per_item"]["C"]["recall@1"], None)

    def test_self_exact_duplicate_and_family_are_excluded(self) -> None:
        gallery = np.array([[1, 0], [0.9, 0.1], [0, 1]], np.float32)
        gallery_meta = [
            gallery_row("A", "same.jpg", "family_self", "samehash"),
            gallery_row("A", "alternative.jpg", "family_alt", "althash"),
            gallery_row("B", "b.jpg", "family_b", "bhash"),
        ]
        query_meta = [{"item_code": "A", "image_path": "same.jpg", "family_id": "family_self", "image_sha256": "samehash"}]
        report = evaluate_item_code_retrieval(np.array([[1, 0]], np.float32), query_meta, gallery, gallery_meta)
        self.assertEqual(report["eligible_query_count"], 1)
        self.assertEqual(report["micro_recall@1"], 1.0)
        self.assertEqual(report["candidate_exclusion_counts"]["self_match"], 1)
        self.assertEqual(report["query_results"][0]["matched_gallery_path"], "alternative.jpg")

    def test_gallery_manifest_incompatibility(self) -> None:
        embedding = np.array([[1, 0]], np.float32)
        metadata = [gallery_row("A", "a.jpg", "a", "a")]
        manifest = {
            "entry_count": 1, "embedding_dimension": 3, "l2_normalized": True,
            "model_version": "m", "catalog_version": "c", "preprocessing_version": "p",
        }
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_gallery(embedding, metadata, manifest)

    def test_every_gallery_row_has_required_fields_and_string_item_code(self) -> None:
        embedding = np.array([[1, 0], [0, 1]], np.float32)
        metadata = [
            gallery_row("00123", "a.jpg", "a", "a"),
            gallery_row("B", "b.jpg", "b", "b"),
        ]
        del metadata[1]["brand"]
        with self.assertRaisesRegex(ValueError, "row 1"):
            validate_gallery(embedding, metadata)
        metadata[1]["brand"] = "AXE"
        metadata[1]["item_code"] = 123  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            validate_gallery(embedding, metadata)

    def test_gallery_catalog_checkpoint_mapping_incompatibility(self) -> None:
        manifest = {
            "model_version": "checkpoint-a", "catalog_version": "catalog-a",
            "mapping_version": "mapping-a", "preprocessing_version": "prep-a",
            "embedding_dimension": 256,
        }
        with self.assertRaisesRegex(ValueError, "model_version"):
            validate_gallery_runtime_compatibility(
                manifest, model_version="checkpoint-b", catalog_version="catalog-a",
                mapping_version="mapping-a", preprocessing_version="prep-a", embedding_dimension=256,
            )
        with self.assertRaisesRegex(ValueError, "catalog_version"):
            validate_gallery_runtime_compatibility(
                manifest, model_version="checkpoint-a", catalog_version="catalog-b",
                mapping_version="mapping-a", preprocessing_version="prep-a", embedding_dimension=256,
            )


if __name__ == "__main__":
    unittest.main()
