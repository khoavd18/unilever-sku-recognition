from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.osa.results import build_osa_summary
from src.recognition.pipeline import (
    EXPECTED_DETECTOR_CLASS_NAMES,
    MISSING_GALLERY_REASON,
    PanoramaRecognizer,
    RecognitionConfig,
    gallery_supported_brands,
    validate_checkpoint_mapping_fingerprint,
    validate_detector_contract,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def extract_embedding(self, batch: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(batch.shape[0]))
        return torch.tensor([[1.0, 0.0]] * int(batch.shape[0]), dtype=torch.float32)


class _FakeGallery:
    def __init__(self) -> None:
        self.query_counts: list[int] = []

    def search(self, embeddings: np.ndarray) -> list[dict[str, object]]:
        self.query_counts.append(len(embeddings))
        return [
            {
                "item_code": "69566283",
                "item_name": "AXE Black",
                "brand": "AXE",
                "image_path": "gallery/axe-black.jpg",
                "cosine_similarity": 0.97,
            }
            for _ in embeddings
        ]


class RecognitionBrandCoverageTests(unittest.TestCase):
    def test_configured_checkpoint_is_detection_with_exact_21_brand_mapping(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "panorama" / "backend.yaml").read_text(encoding="utf-8")
        )
        recognition = config["recognition"]
        relative = recognition["yolo_weights"]
        self.assertEqual(
            relative,
            "deployment/assets/models/yolov8_brand21_int8.onnx",
        )
        names = {int(index): name for index, name in recognition["detector_class_names"].items()}
        metadata = validate_detector_contract(SimpleNamespace(task="detect", names=names), names)
        self.assertEqual(metadata["task"], "detect")
        self.assertEqual(metadata["names"], names)
        self.assertEqual(metadata["class_count"], 21)

    def test_detector_contract_rejects_wrong_task_or_class_order(self) -> None:
        valid = SimpleNamespace(task="detect", names=dict(EXPECTED_DETECTOR_CLASS_NAMES))
        self.assertEqual(validate_detector_contract(valid)["class_count"], 9)
        with self.assertRaisesRegex(ValueError, "task must be 'detect'"):
            validate_detector_contract(SimpleNamespace(task="segment", names=valid.names))
        swapped = dict(EXPECTED_DETECTOR_CLASS_NAMES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "class mapping differs"):
            validate_detector_contract(SimpleNamespace(task="detect", names=swapped))

    def test_gallery_coverage_is_derived_from_metadata_and_catalog(self) -> None:
        catalog = {"records": [{"item_code": "69566283", "brand": "AXE"}]}
        metadata = [{"item_code": "69566283", "brand": "AXE"}]
        self.assertEqual(gallery_supported_brands(catalog, metadata), {"axe": "AXE"})
        with self.assertRaisesRegex(ValueError, "not present in the scoped catalog"):
            gallery_supported_brands(catalog, [{"item_code": "x", "brand": "VIM"}])

    def test_mapping_checkpoint_fingerprint_is_enforced(self) -> None:
        validate_checkpoint_mapping_fingerprint(
            {"checkpoint_fingerprint": "sha256:one"}, "sha256:one"
        )
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            validate_checkpoint_mapping_fingerprint(
                {"checkpoint_fingerprint": "sha256:one"}, "sha256:two"
            )

    def test_recognition_yaml_paths_are_repo_root_relative(self) -> None:
        config = RecognitionConfig.from_yaml("configs/panorama/backend.yaml")
        self.assertTrue(Path(config.yolo_weights).is_absolute())
        self.assertEqual(
            Path(config.catalog),
            ROOT / "deployment" / "assets" / "catalog" / "product_catalog.json",
        )

    def test_axe_uses_full_gallery_while_unsupported_brand_is_retained(self) -> None:
        recognizer = PanoramaRecognizer.__new__(PanoramaRecognizer)
        recognizer.config = RecognitionConfig(
            yolo_weights="unused",
            embedding_checkpoint="unused",
            catalog="unused",
            gallery_dir="unused",
            mapping="unused",
            device="cpu",
            minimum_similarity=0.5,
            similarity_threshold_calibrated=True,
        )
        recognizer.device = torch.device("cpu")
        recognizer.detector_metadata = {
            "task": "detect",
            "names": dict(EXPECTED_DETECTOR_CLASS_NAMES),
            "class_count": 9,
        }
        recognizer.gallery_brand_coverage = {"axe": "AXE"}
        recognizer.versions = {"detector": "sha256:test"}
        recognizer.transform = lambda _image: torch.ones((3, 4, 4), dtype=torch.float32)
        recognizer.embedding_model = _FakeEmbeddingModel()
        recognizer.gallery = _FakeGallery()
        recognizer._detect = lambda _panorama, _mask: [
            {
                "bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
                "detector_score": 0.91,
                "detector_class_id": 1,
                "brand": "VIM",
            },
            {
                "bbox_xyxy": [35.0, 5.0, 75.0, 55.0],
                "detector_score": 0.88,
                "detector_class_id": 0,
                "brand": "AXE",
            },
        ]

        result = recognizer.recognize(np.zeros((64, 96, 3), dtype=np.uint8))
        vim, axe = result["detections"]

        self.assertEqual(vim["brand"], "VIM")
        self.assertEqual(vim["bbox"], [1, 2, 30, 40])
        self.assertEqual(vim["detector_score"], 0.91)
        self.assertFalse(vim["resolved"])
        self.assertEqual(vim["reason"], MISSING_GALLERY_REASON)
        self.assertIsNone(vim["item_code"])
        self.assertIsNone(vim["candidate"])
        self.assertIsNone(vim["similarity_score"])

        self.assertTrue(axe["resolved"])
        self.assertEqual(axe["item_code"], "69566283")
        self.assertEqual(axe["brand"], "AXE")
        self.assertEqual(axe["similarity_score"], 0.97)
        self.assertEqual(recognizer.embedding_model.batch_sizes, [1])
        self.assertEqual(recognizer.gallery.query_counts, [1])
        self.assertTrue(result["recognition_policy"]["full_gallery_search"])
        self.assertFalse(result["recognition_policy"]["brand_filter"])

        osa = build_osa_summary(result["detections"])
        self.assertEqual(osa["detected_visible_count"], 2)
        self.assertEqual(osa["brand_visible_counts"], {"AXE": 1, "VIM": 1})
        self.assertEqual(osa["resolved_visible_count"], 1)
        self.assertEqual(osa["model_visible_counts"], {"69566283": 1})
        self.assertEqual(osa["unresolved_count"], 1)
        self.assertEqual(
            osa["unresolved_detections"][0]["reason"], MISSING_GALLERY_REASON
        )


if __name__ == "__main__":
    unittest.main()
