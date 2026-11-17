"""Create non-destructive CSV train/validation/test manifests from product folders."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.product_dataset import is_image_file
from src.utils.io import ensure_dir, save_json


SPLITS = ("train", "val", "test")
SOURCE_SPLIT_NAMES = {"train": "train", "val": "val", "test": "test"}


def find_images(class_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in class_dir.rglob("*") if is_image_file(path))


def detect_layout(dataset_path: Path) -> str:
    names = {path.name for path in dataset_path.iterdir() if path.is_dir()}
    # Roboflow exports commonly call validation "valid". Keep source folders
    # untouched and normalize only the generated manifest name to val.csv.
    if "valid" in names and "val" in names:
        raise ValueError("Both 'val' and 'valid' folders exist; keep only one to avoid an ambiguous validation split.")
    source_names = {"train", "test", "valid" if "valid" in names else "val"}
    if source_names.issubset(names):
        SOURCE_SPLIT_NAMES["val"] = "valid" if "valid" in names else "val"
        return "pre_split"
    found_split_names = names & {"train", "val", "valid", "test"}
    if found_split_names:
        raise ValueError(
            "Found only part of a train/val/test layout: "
            f"{sorted(found_split_names)}. Provide all three split folders or use direct product-class folders."
        )
    return "class_folders"


def split_class_images(images: list[Path], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, list[Path]]:
    images = images.copy()
    rng.shuffle(images)
    count = len(images)
    if count == 0:
        return {split: [] for split in SPLITS}
    if count == 1:
        return {"train": images, "val": [], "test": []}
    if count == 2:
        return {"train": images[:1], "val": images[1:], "test": []}
    val_count = max(1, round(count * ratios[1]))
    test_count = max(1, round(count * ratios[2]))
    train_count = count - val_count - test_count
    while train_count < 1:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
        train_count = count - val_count - test_count
    return {
        "train": images[:train_count],
        "val": images[train_count : train_count + val_count],
        "test": images[train_count + val_count :],
    }


def collect_from_class_folders(dataset_path: Path, ratios: tuple[float, float, float], seed: int):
    class_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class directories were found directly inside {dataset_path}. Expected dataset/product_id/image.jpg."
        )
    all_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    warnings: list[str] = []
    for class_index, class_dir in enumerate(class_dirs):
        images = find_images(class_dir)
        if not images:
            warnings.append(f"Class '{class_dir.name}' has no supported image files.")
            continue
        if len(images) < 3:
            warnings.append(
                f"Class '{class_dir.name}' has only {len(images)} image(s); it cannot appear in all train/val/test splits."
            )
        parts = split_class_images(images, ratios, random.Random(seed + class_index))
        for split, split_images in parts.items():
            all_rows[split].extend(
                {"image_path": str(image), "class_name": class_dir.name, "split": split} for image in split_images
            )
    return all_rows, warnings


def collect_pre_split(dataset_path: Path):
    all_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    warnings: list[str] = []
    for split in SPLITS:
        split_dir = dataset_path / SOURCE_SPLIT_NAMES[split]
        if (split_dir / "_annotations.coco.json").is_file():
            raise ValueError(
                f"{split_dir} is a COCO export, not a product-folder dataset. "
                "Run scripts/prepare_coco_crops.py; it preserves the Roboflow splits and creates compatible manifests."
            )
        if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
            raise ValueError(
                f"{split_dir} looks like a YOLO detection export (images/ and labels/). "
                "This project needs product-level classes: train/product_id/image.jpg. "
                "Do not use YOLO brand labels as product IDs; prepare cropped product images grouped by product ID."
            )
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            warnings.append(f"Split '{split}' has no class directories.")
        for class_dir in class_dirs:
            images = find_images(class_dir)
            if not images:
                warnings.append(f"Class '{class_dir.name}' in '{split}' has no supported images.")
            all_rows[split].extend(
                {"image_path": str(image), "class_name": class_dir.name, "split": split} for image in images
            )
    return all_rows, warnings


def write_manifests(rows_by_split: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    for split in SPLITS:
        rows = sorted(rows_by_split[split], key=lambda row: (row["class_name"], row["image_path"]))
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_path", "class_name", "split"))
            writer.writeheader()
            writer.writerows(rows)


def make_report(rows_by_split: dict[str, list[dict[str, str]]], layout: str, warnings: list[str], dataset_path: Path):
    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    class_counts = Counter(row["class_name"] for row in all_rows)
    split_counts = {split: len(rows_by_split[split]) for split in SPLITS}
    per_split_classes = {split: len({row["class_name"] for row in rows_by_split[split]}) for split in SPLITS}
    counts = list(class_counts.values())
    missing_by_class = {
        class_name: [split for split in SPLITS if not any(row["class_name"] == class_name for row in rows_by_split[split])]
        for class_name in sorted(class_counts)
    }
    missing_by_class = {name: missing for name, missing in missing_by_class.items() if missing}
    return {
        "dataset_path": str(dataset_path.resolve()),
        "detected_layout": layout,
        "total_classes": len(class_counts),
        "total_images": len(all_rows),
        "images_per_split": split_counts,
        "classes_per_split": per_split_classes,
        "images_per_class": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0,
        },
        "class_imbalance_ratio_max_to_min": (max(counts) / min(counts)) if counts and min(counts) else None,
        "classes_missing_a_split": missing_by_class,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path, help="Root of product folders or existing train/val/test folders.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must be positive and sum to 1.0")
    layout = detect_layout(dataset_path)
    if layout == "pre_split":
        rows_by_split, warnings = collect_pre_split(dataset_path)
    else:
        rows_by_split, warnings = collect_from_class_folders(dataset_path, ratios, args.seed)
    output_dir = ensure_dir(args.output_dir)
    write_manifests(rows_by_split, output_dir)
    report = make_report(rows_by_split, layout, warnings, dataset_path)
    save_json(report, output_dir / "split_report.json")
    print(f"Detected layout: {layout}")
    print(f"Classes: {report['total_classes']} | Images: {report['total_images']} | Splits: {report['images_per_split']}")
    print(f"Wrote read-only manifests and report to: {output_dir.resolve()}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
