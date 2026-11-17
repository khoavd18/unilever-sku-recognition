"""Read-only readiness checks and DataLoaders for embedding workflows."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.catalog.profiles import load_product_profile
from src.datasets.product_dataset import ProductDataset, build_transforms


# Backward-compatible default. New callers pass ``expected_classes`` explicitly.
EXPECTED_SKUS = load_product_profile("axe5").legacy_class_names


def path_key(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def training_manifest_paths(split_dir: Path) -> dict[str, Path]:
    """Expose only gallery/train and validation membership to manual training."""

    return {"train": split_dir / "train.csv", "validation": split_dir / "val.csv"}


def training_dataset_fingerprint(crops_root: Path) -> str:
    """Fingerprint mapping/provenance plus train and validation membership."""

    digest = hashlib.sha256()
    for path in (
        crops_root / "class_mapping.json",
        crops_root / "crop_manifest.csv",
        crops_root / "manifests" / "train.csv",
        crops_root / "manifests" / "val.csv",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_training_readiness(
    crops_root: Path,
    clean_root: Path,
    *,
    expected_classes: Sequence[str] | None = None,
    dataset_label: str = "embedding",
) -> dict[str, Any]:
    """Validate training/validation without opening held-out test queries."""

    crops_root = crops_root.resolve()
    clean_root = clean_root.resolve()
    expected = tuple(str(name) for name in (expected_classes or EXPECTED_SKUS))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_classes must be a non-empty unique sequence")
    mapping_path = crops_root / "class_mapping.json"
    crop_manifest_path = crops_root / "crop_manifest.csv"
    train_path = crops_root / "manifests" / "train.csv"
    val_path = crops_root / "manifests" / "val.csv"
    required = (mapping_path, crop_manifest_path, train_path, val_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_label} crop preparation is incomplete; missing: {missing}"
        )

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_to_idx = {str(name): int(index) for name, index in mapping["class_to_idx"].items()}
    if set(class_to_idx) != set(expected) or sorted(class_to_idx.values()) != list(
        range(len(expected))
    ):
        raise ValueError(
            f"Expected one consistent configured class mapping for {list(expected)}, "
            f"found: {class_to_idx}"
        )

    rows_by_split = {"train": read_csv(train_path), "val": read_csv(val_path)}
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"{split}.csv is empty")
        if {row["class_name"] for row in rows} != set(expected):
            raise ValueError(
                f"{split}.csv does not contain exactly the configured classes"
            )
        for row in rows:
            image_path = Path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {split} crop: {image_path}")
            with Image.open(image_path) as image:
                image.verify()

    crop_rows = read_csv(crop_manifest_path)
    required_fields = {
        "generated_crop_path",
        "class_name",
        "source_zip",
        "family_id",
        "source_image_id",
        "source_annotation_id",
    }
    if not crop_rows or not required_fields.issubset(crop_rows[0]):
        raise ValueError("crop_manifest.csv does not contain complete source traceability")
    crop_by_path = {path_key(row["generated_crop_path"]): row for row in crop_rows}
    if len(crop_by_path) != len(crop_rows):
        raise ValueError("crop_manifest.csv contains duplicate generated crop paths")

    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in crop_rows:
        family_splits[row["family_id"]].add(row["split"])
    leaks = [family for family, splits in family_splits.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f"Source families cross clean split assignments: {leaks[:10]}")

    split_families: dict[str, set[str]] = {}
    split_paths: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        keys = {path_key(row["image_path"]) for row in rows}
        if not keys.issubset(crop_by_path):
            raise ValueError(f"{split}.csv contains crops absent from crop_manifest.csv")
        split_paths[split] = keys
        split_families[split] = {crop_by_path[key]["family_id"] for key in keys}
    if split_paths["train"] & split_paths["val"]:
        raise ValueError("A query crop occurs in the training gallery")
    family_overlap = split_families["train"] & split_families["val"]
    if family_overlap:
        raise ValueError(f"Validation families occur in the training gallery: {sorted(family_overlap)[:10]}")

    hash_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
            hash_splits[digest].add(split)
    exact_overlap = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if exact_overlap:
        raise ValueError(f"Exact crop duplicates cross train/validation: {exact_overlap[:10]}")

    clean_mapping = json.loads((clean_root / "class_mapping.json").read_text(encoding="utf-8"))
    if {item["name"] for item in clean_mapping["categories"]} != set(expected):
        raise ValueError("Clean COCO class mapping differs from the crop class mapping")
    clean_report = json.loads((clean_root / "build_report.json").read_text(encoding="utf-8"))
    if clean_report.get("assertions", {}).get("cross_split_family_leakage") != 0:
        raise ValueError("Clean dataset report does not assert zero source-family leakage")

    return {
        "status": "ready",
        "dataset_label": dataset_label,
        "crops_root": str(crops_root),
        "clean_root": str(clean_root),
        "class_to_idx": class_to_idx,
        "crop_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "per_sku_counts": {
            split: dict(sorted(Counter(row["class_name"] for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "source_family_counts": {
            split: len(split_families[split]) for split in ("train", "val")
        },
        "train_validation_family_overlap": 0,
        "train_validation_exact_duplicate_groups": 0,
        "dataset_fingerprint": training_dataset_fingerprint(crops_root),
        "test_queries_loaded": False,
        "remaining_uncertainty": (
            "Filename-family and exact-hash checks cannot prove capture-session independence."
        ),
    }


def make_loader(
    dataset: ProductDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=False,
    )


def build_training_loaders(
    split_dir: Path,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, deterministic train-gallery, and validation loaders only."""

    manifests = training_manifest_paths(split_dir)
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["dataset"].get("num_workers", 0))
    seed = int(config["dataset"].get("seed", 42))
    augmentation = config.get("augmentation", {})
    train_dataset = ProductDataset(
        manifests["train"],
        class_to_idx,
        build_transforms(
            image_size,
            train=True,
            horizontal_flip=bool(augmentation.get("horizontal_flip", False)),
            rotation_degrees=float(augmentation.get("rotation_degrees", 5.0)),
            brightness=float(augmentation.get("brightness", 0.12)),
            contrast=float(augmentation.get("contrast", 0.12)),
            saturation=float(augmentation.get("saturation", 0.08)),
            hue=float(augmentation.get("hue", 0.02)),
        ),
    )
    gallery_dataset = ProductDataset(
        manifests["train"], class_to_idx, build_transforms(image_size, train=False)
    )
    validation_dataset = ProductDataset(
        manifests["validation"], class_to_idx, build_transforms(image_size, train=False)
    )
    return (
        make_loader(train_dataset, batch_size, workers, True, device, seed),
        make_loader(gallery_dataset, 64, 0, False, device, seed),
        make_loader(validation_dataset, 64, 0, False, device, seed),
    )


def build_classification_loaders(
    split_dir: Path,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Construct augmented train and deterministic validation loaders; never test/gallery."""

    manifests = training_manifest_paths(split_dir)
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["dataset"].get("num_workers", 0))
    seed = int(config["dataset"].get("seed", 42))
    augmentation = config.get("augmentation", {})
    train_dataset = ProductDataset(
        manifests["train"],
        class_to_idx,
        build_transforms(
            image_size,
            train=True,
            horizontal_flip=bool(augmentation.get("horizontal_flip", False)),
            rotation_degrees=float(augmentation.get("rotation_degrees", 5.0)),
            brightness=float(augmentation.get("brightness", 0.12)),
            contrast=float(augmentation.get("contrast", 0.12)),
            saturation=float(augmentation.get("saturation", 0.08)),
            hue=float(augmentation.get("hue", 0.02)),
        ),
    )
    validation_dataset = ProductDataset(
        manifests["validation"], class_to_idx, build_transforms(image_size, train=False)
    )
    return (
        make_loader(train_dataset, batch_size, workers, True, device, seed),
        make_loader(validation_dataset, 64, 0, False, device, seed),
    )
