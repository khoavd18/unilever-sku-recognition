"""Reuse YOLO and MobileNet components for panorama product recognition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from src.catalog.catalog import load_catalog, require_usable_catalog
from src.catalog.mapping import load_mapping
from src.config.paths import resolve_config_path, resolve_repo_path
from src.config.registry import load_brand_registry, normalize_brand_token
from src.datasets.product_dataset import build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.recognition.onnx_detector import OnnxYoloDetector
from src.recognition.tiling import (
    generate_tiles, map_box_to_panorama, suppress_duplicate_boxes, valid_region_ratio,
)
from src.retrieval.item_code import GalleryIndex, validate_gallery_runtime_compatibility
from src.retrieval.compatibility import embedding_preprocessing_version
from src.utils.config import resolve_device


EXPECTED_DETECTOR_CLASS_NAMES = load_brand_registry().class_names("brand_9brands")
MISSING_GALLERY_REASON = "missing_gallery_for_detected_brand"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordered_detector_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        try:
            return {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Detector class mapping keys must be integer-like") from exc
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    raise ValueError("Detector class mapping must be a dict or sequence")


def validate_detector_contract(
    detector: Any, expected_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the configured checkpoint is the reviewed detector."""

    task = getattr(detector, "task", None)
    if task is None and getattr(detector, "model", None) is not None:
        task = getattr(detector.model, "task", None)
    if str(task) != "detect":
        raise ValueError(f"Detector checkpoint task must be 'detect', found {task!r}")
    names = _ordered_detector_names(getattr(detector, "names", None))
    expected = expected_names or EXPECTED_DETECTOR_CLASS_NAMES
    if names != expected:
        raise ValueError(
            "Detector checkpoint class mapping differs from the configured deployment contract: "
            f"expected={expected}, actual={names}"
        )
    return {"task": "detect", "names": names, "class_count": len(names)}


@lru_cache(maxsize=None)
def canonical_brand_name(value: str) -> str:
    """Normalize catalog spelling to the detector's canonical brand token."""

    normalized = normalize_brand_token(value)
    registry = load_brand_registry()
    resolved = registry.infer_brand(normalized, "brand_21brands")
    return resolved or normalized


def validate_checkpoint_mapping_fingerprint(
    mapping: dict[str, Any], checkpoint_version: str
) -> None:
    """Validate the mapping's optional checkpoint identity without loading a model."""

    mapped_checkpoint = mapping.get("checkpoint_fingerprint")
    if mapped_checkpoint and mapped_checkpoint != checkpoint_version:
        raise ValueError(
            "Embedding checkpoint fingerprint differs from the verified mapping"
        )


def gallery_supported_brands(
    catalog: dict[str, Any], gallery_metadata: list[dict[str, Any]]
) -> dict[str, str]:
    """Return case-insensitive gallery coverage after catalog cross-checking."""

    catalog_records = {
        str(row.get("item_code", "")): row
        for row in catalog.get("records", [])
        if isinstance(row.get("item_code"), str) and row["item_code"]
    }
    catalog_brands = {
        canonical_brand_name(str(row.get("brand", ""))).casefold(): canonical_brand_name(str(row.get("brand", "")))
        for row in catalog.get("records", [])
        if str(row.get("brand", "")).strip()
    }
    gallery_brands = {
        canonical_brand_name(str(row.get("brand", ""))).casefold(): canonical_brand_name(str(row.get("brand", "")))
        for row in gallery_metadata
        if str(row.get("brand", "")).strip()
    }
    if not gallery_brands:
        raise ValueError("Gallery metadata contains no usable brand coverage")
    missing_from_catalog = sorted(set(gallery_brands) - set(catalog_brands))
    if missing_from_catalog:
        raise ValueError(
            "Gallery brand coverage is not present in the scoped catalog: "
            f"{missing_from_catalog}"
        )
    incompatible_items = []
    for index, row in enumerate(gallery_metadata):
        item_code = row.get("item_code")
        if not isinstance(item_code, str) or item_code not in catalog_records:
            incompatible_items.append(
                {"row": index, "item_code": item_code, "reason": "missing_catalog_item"}
            )
            continue
        gallery_brand = canonical_brand_name(str(row.get("brand", "")))
        catalog_brand = canonical_brand_name(str(catalog_records[item_code].get("brand", "")))
        if gallery_brand != catalog_brand:
            incompatible_items.append(
                {"row": index, "item_code": item_code, "reason": "brand_mismatch"}
            )
    if incompatible_items:
        raise ValueError(
            "Gallery item identities are incompatible with the scoped catalog: "
            f"{incompatible_items[:10]}"
        )
    return dict(sorted(gallery_brands.items()))


@dataclass(frozen=True)
class RecognitionConfig:
    yolo_weights: str
    embedding_checkpoint: str
    catalog: str
    gallery_dir: str
    mapping: str
    detector_class_names: dict[int, str] | None = None
    device: str = "auto"
    detector_confidence: float = 0.25
    detector_image_size: int = 960
    tile_size: int = 1280
    tile_overlap: float = 0.20
    tile_nms_iou: float = 0.50
    min_valid_tile_ratio: float = 0.15
    min_valid_crop_ratio: float = 0.60
    minimum_similarity: float | None = None
    similarity_threshold_calibrated: bool = False

    def __post_init__(self) -> None:
        if self.detector_class_names is not None:
            names = _ordered_detector_names(self.detector_class_names)
            if names != {index: names[index] for index in range(len(names))}:
                raise ValueError("detector_class_names must use contiguous IDs from zero")
        if not 0.0 <= float(self.detector_confidence) <= 1.0:
            raise ValueError("detector_confidence must be in [0, 1]")
        if int(self.detector_image_size) <= 0 or int(self.tile_size) <= 0:
            raise ValueError("detector_image_size and tile_size must be positive")
        for name in ("tile_overlap", "tile_nms_iou", "min_valid_tile_ratio", "min_valid_crop_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_similarity is not None and not -1.0 <= float(self.minimum_similarity) <= 1.0:
            raise ValueError("minimum_similarity must be in [-1, 1]")
        if self.similarity_threshold_calibrated and self.minimum_similarity is None:
            raise ValueError(
                "similarity_threshold_calibrated=true requires minimum_similarity"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RecognitionConfig":
        config_path = resolve_config_path(path)
        value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        recognition = dict(value["recognition"])
        for key in ("yolo_weights", "embedding_checkpoint", "catalog", "gallery_dir", "mapping"):
            recognition[key] = str(resolve_repo_path(recognition[key]))
        return cls(**recognition)


class PanoramaRecognizer:
    """Panorama -> tiled YOLO detections -> crops -> embeddings -> full gallery."""

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self.device = resolve_device(config.device)
        yolo_path = Path(config.yolo_weights).resolve()
        checkpoint_path = Path(config.embedding_checkpoint).resolve()
        for path in (yolo_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.catalog = load_catalog(config.catalog)
        require_usable_catalog(self.catalog, require_scope=True)
        self.mapping = load_mapping(config.mapping)
        if self.mapping["catalog_version"] != self.catalog["catalog_version"]:
            raise ValueError("Runtime catalog version differs from the verified mapping")
        catalog_codes = [row["item_code"] for row in self.catalog["records"]]
        mapping_codes = [row["item_code"] for row in self.mapping["mappings"]]
        if catalog_codes != mapping_codes:
            raise ValueError("Scoped catalog item order/content differs from checkpoint mapping")
        expected_names = self.config.detector_class_names or EXPECTED_DETECTOR_CLASS_NAMES
        self._onnx_detector = yolo_path.suffix.casefold() == ".onnx"
        if self._onnx_detector:
            self.detector = OnnxYoloDetector(yolo_path, expected_names)
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("PyTorch detector requires ultralytics from requirements.txt") from exc
            self.detector = YOLO(str(yolo_path), task="detect")
        self.detector_metadata = validate_detector_contract(
            self.detector, self.config.detector_class_names
        )
        self.embedding_model, checkpoint = model_from_checkpoint(checkpoint_path, self.device)
        checkpoint_version = _sha256(checkpoint_path)
        validate_checkpoint_mapping_fingerprint(self.mapping, checkpoint_version)
        checkpoint_order = [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda pair: pair[1])]
        mapping_order = [row["legacy_class_name"] for row in self.mapping["mappings"]]
        if checkpoint_order != mapping_order:
            raise ValueError("Embedding checkpoint class order differs from verified item-code mapping")
        self.gallery = GalleryIndex.load(config.gallery_dir)
        self.gallery_brand_coverage = gallery_supported_brands(
            self.catalog, self.gallery.metadata
        )
        image_size = int(checkpoint["config"]["dataset"]["image_size"])
        runtime_preprocessing_version = embedding_preprocessing_version(image_size)
        validate_gallery_runtime_compatibility(
            self.gallery.manifest,
            model_version=checkpoint_version,
            catalog_version=self.mapping["catalog_version"],
            mapping_version=self.mapping["mapping_version"],
            preprocessing_version=runtime_preprocessing_version,
            embedding_dimension=int(checkpoint["config"]["model"]["embedding_dim"]),
        )
        self.image_size = image_size
        self.transform = build_transforms(self.image_size, train=False)
        self.versions = {
            "detector": _sha256(yolo_path),
            "embedding_model": checkpoint_version,
            "catalog": self.gallery.manifest["catalog_version"],
            "gallery": "sha256:" + hashlib.sha256(
                json.dumps(self.gallery.manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "mapping": self.mapping["mapping_version"],
            "preprocessing": self.gallery.manifest["preprocessing_version"],
        }

    def _detect(self, panorama: np.ndarray, valid_mask: np.ndarray | None) -> list[dict[str, Any]]:
        height, width = panorama.shape[:2]
        detections: list[dict[str, Any]] = []
        tiles = generate_tiles(width, height, self.config.tile_size, self.config.tile_overlap)
        for tile in tiles:
            tile_valid_ratio = valid_region_ratio(valid_mask, tile)
            if tile_valid_ratio < self.config.min_valid_tile_ratio:
                continue
            image = panorama[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            if not image.size:
                continue
            if self._onnx_detector:
                rows = self.detector.predict(image, self.config.detector_confidence)
                names = self.detector.names
                predictions = ((row["xyxy"], row["score"], row["class_id"]) for row in rows)
            else:
                results = self.detector.predict(
                    source=image,
                    conf=self.config.detector_confidence,
                    imgsz=self.config.detector_image_size,
                    device=(self.device.index if self.device.index is not None else 0) if self.device.type == "cuda" else "cpu",
                    verbose=False,
                )
                predictions = (
                    (box.tolist(), float(score), int(class_id))
                    for result in results if result.boxes is not None
                    for box, score, class_id in zip(
                        result.boxes.xyxy.detach().cpu().numpy(),
                        result.boxes.conf.detach().cpu().numpy(),
                        result.boxes.cls.detach().cpu().numpy().astype(int),
                    )
                )
                names = self.detector.names
            for box, score, class_id in predictions:
                    try:
                        mapped_box = map_box_to_panorama(box, tile, width, height)
                    except ValueError:
                        continue
                    detections.append({
                        "bbox_xyxy": mapped_box,
                        "detector_score": float(score),
                        "detector_class_id": int(class_id),
                        "brand": str(names.get(int(class_id), class_id) if isinstance(names, dict) else names[int(class_id)]),
                        "source_tile_id": tile.tile_id,
                        "source_tile_valid_ratio": tile_valid_ratio,
                    })
        return suppress_duplicate_boxes(detections, self.config.tile_nms_iou, class_aware=True)

    def recognize(self, panorama: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, Any]:
        if panorama.ndim != 3 or panorama.shape[2] != 3 or panorama.size == 0:
            raise ValueError("Panorama must be a non-empty BGR image")
        if valid_mask is not None and valid_mask.shape[:2] != panorama.shape[:2]:
            raise ValueError("Valid mask dimensions differ from panorama")
        height, width = panorama.shape[:2]
        raw = self._detect(panorama, valid_mask)
        output: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        tensor_output_indexes: list[int] = []
        for detection_index, detection in enumerate(raw):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = panorama[y1:y2, x1:x2]
            mask_ratio = 1.0
            if valid_mask is not None and crop.size:
                mask_crop = valid_mask[y1:y2, x1:x2]
                mask_ratio = float((mask_crop > 0).mean()) if mask_crop.size else 0.0
            base = {
                "detection_id": f"det_{detection_index:04d}",
                "bbox": [x1, y1, x2, y2],
                "bbox_coordinate_system": "panorama_pixels_xyxy",
                "panorama_size": {"width": width, "height": height},
                "brand": detection["brand"],
                "detector_class_id": detection["detector_class_id"],
                "detector_score": detection["detector_score"],
                "item_code": None,
                "item_name": None,
                "similarity_score": None,
                "similarity_is_probability": False,
                "resolved": False,
                "status": "unresolved",
                "reason": None,
                "candidate": None,
                "valid_crop_ratio": mask_ratio,
                "model_versions": self.versions,
                "model_prediction": None,
                "manual_correction": None,
                "gallery_coverage": {
                    "supported": canonical_brand_name(detection["brand"]).casefold()
                    in self.gallery_brand_coverage,
                    "detected_brand": detection["brand"],
                    "supported_brands": sorted(self.gallery_brand_coverage.values()),
                },
            }
            output.append(base)
            if not base["gallery_coverage"]["supported"]:
                base["reason"] = MISSING_GALLERY_REASON
                continue
            if crop.size == 0:
                base["reason"] = "empty_crop"
                continue
            if mask_ratio < self.config.min_valid_crop_ratio:
                base["reason"] = "crop_overlaps_invalid_stitched_region"
                continue
            rgb = crop[:, :, ::-1]
            tensors.append(self.transform(Image.fromarray(rgb)))
            tensor_output_indexes.append(len(output) - 1)

        if tensors:
            with torch.inference_mode():
                embeddings = self.embedding_model.extract_embedding(torch.stack(tensors).to(self.device)).float().cpu().numpy()
            matches = self.gallery.search(embeddings)
            for output_index, match in zip(tensor_output_indexes, matches):
                row = output[output_index]
                score = float(match["cosine_similarity"])
                candidate = {
                    "item_code": match["item_code"], "item_name": match["item_name"],
                    "brand": match["brand"], "matched_gallery_path": match["image_path"],
                }
                row["similarity_score"] = score
                row["candidate"] = candidate
                can_resolve = (
                    self.config.minimum_similarity is not None
                    and self.config.similarity_threshold_calibrated
                    and score >= self.config.minimum_similarity
                )
                if can_resolve:
                    row.update({
                        "item_code": match["item_code"], "item_name": match["item_name"],
                        "resolved": True, "status": "resolved", "reason": None,
                        "model_prediction": {
                            "item_code": match["item_code"], "item_name": match["item_name"],
                            "gallery_brand": match["brand"],
                            "similarity_score": score,
                        },
                    })
                elif self.config.minimum_similarity is None or not self.config.similarity_threshold_calibrated:
                    row["reason"] = "similarity_threshold_not_calibrated"
                else:
                    row["reason"] = "below_similarity_threshold"
        return {
            "schema_version": 1,
            "panorama": {"width": width, "height": height, "coordinate_system": "pixels_top_left_origin"},
            "detection_count": len(output),
            "resolved_count": sum(bool(row["resolved"]) for row in output),
            "unresolved_count": sum(not bool(row["resolved"]) for row in output),
            "detections": output,
            "versions": self.versions,
            "recognition_policy": {
                "full_gallery_search": True,
                "brand_filter": False,
                "brand_filter_recall_risk": "A wrong detector brand can hide the correct gallery item; filtering is disabled.",
                "minimum_similarity": self.config.minimum_similarity,
                "threshold_calibrated": self.config.similarity_threshold_calibrated,
                "cosine_is_calibrated_probability": False,
                "gallery_supported_brands": sorted(self.gallery_brand_coverage.values()),
                "coverage_guard": (
                    "Detections whose brand has no gallery coverage remain unresolved and are "
                    "not embedded or searched. Supported detections still search the full gallery."
                ),
            },
            "detector": self.detector_metadata,
        }
