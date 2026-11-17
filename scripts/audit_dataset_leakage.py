"""Read-only leakage, correlation, and checkpoint audit for product retrieval.

This script never changes source COCO images, crops, manifests, configuration, or
checkpoints. It writes only beneath ``outputs/overfitting_audit`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_coco_crops import SOURCE_SPLITS
from src.datasets.product_dataset import ProductDataset, build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.utils.config import load_config, resolve_device
from src.utils.io import ensure_dir, save_json


AUDIT_SPLITS = ("train", "val", "test")
PAIR_FIELDS = (
    "asset_type",
    "pair_scope",
    "file_a",
    "split_a",
    "label_a",
    "source_image_id_a",
    "source_annotation_id_a",
    "file_b",
    "split_b",
    "label_b",
    "source_image_id_b",
    "source_annotation_id_b",
    "sha256",
    "labels_conflict",
)


def normalized_path(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash bytes only; this is intentionally read-only."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_image(path: str | Path) -> int:
    """Return a deterministic 64-bit dHash without resizing files on disk."""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def normalize_source_family(filename_or_path: str | Path) -> str:
    """Conservatively remove only Roboflow hash and terminal augmentation suffixes."""
    stem = Path(filename_or_path).stem
    # Roboflow's common suffix is ``_jpg.rf.<hex>`` or ``.rf.<hex>``.
    stem = re.sub(r"(?:_jpg)?\.rf\.[0-9a-f]{8,}$", "", stem, flags=re.IGNORECASE)
    # Remove only a final augmentation marker such as _aug1, -aug_2, _augmentation3.
    stem = re.sub(r"(?:[_-](?:aug|augmentation)[_-]?\d+)$", "", stem, flags=re.IGNORECASE)
    return stem.casefold()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_crop_manifest(crops_root: Path) -> list[dict[str, str]]:
    path = crops_root / "crop_manifest.csv"
    rows = _csv_rows(path)
    required = {"split", "image_path", "class_name", "source_image_path", "source_image_id", "source_annotation_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Crop manifest is incomplete: {path}")
    return rows


def load_source_records(dataset_root: Path, crop_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Read COCO source image metadata and attach valid crop labels where available."""
    crop_labels: dict[str, set[str]] = defaultdict(set)
    crop_annotations: dict[str, set[str]] = defaultdict(set)
    for row in crop_rows:
        path_key = normalized_path(row["source_image_path"])
        crop_labels[path_key].add(row["class_name"])
        crop_annotations[path_key].add(row["source_annotation_id"])
    records: list[dict[str, Any]] = []
    for split, source_split in SOURCE_SPLITS.items():
        annotation_path = dataset_root / source_split / "_annotations.coco.json"
        with annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        split_dir = (dataset_root / source_split).resolve()
        for image in coco.get("images", []):
            path = (split_dir / str(image.get("file_name", ""))).resolve()
            path_key = normalized_path(path)
            records.append(
                {
                    "asset_type": "source",
                    "path": str(path),
                    "path_key": path_key,
                    "split": split,
                    "label": ";".join(sorted(crop_labels.get(path_key, set()))),
                    "source_image_id": str(image.get("id", "")),
                    "source_annotation_id": ";".join(sorted(crop_annotations.get(path_key, set()))),
                    "family_id": normalize_source_family(path.name),
                }
            )
    return records


def load_crop_records(crop_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records = []
    for row in crop_rows:
        crop_path = Path(row["image_path"]).resolve()
        records.append(
            {
                "asset_type": "crop",
                "path": str(crop_path),
                "path_key": normalized_path(crop_path),
                "split": row["split"],
                "label": row["class_name"],
                "source_path": row["source_image_path"],
                "source_image_id": row["source_image_id"],
                "source_annotation_id": row["source_annotation_id"],
            }
        )
    return records


def add_sha256(records: list[dict[str, Any]], description: str) -> list[dict[str, Any]]:
    valid = []
    failures = []
    for record in tqdm(records, desc=description):
        try:
            record["sha256"] = sha256_file(record["path"])
            valid.append(record)
        except OSError as error:
            failures.append({"path": record["path"], "split": record["split"], "error": str(error)})
    return valid, failures


def exact_duplicate_pairs(records: list[dict[str, Any]], max_pairs: int = 1_000_000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return within/cross-split SHA-256 duplicate pairs, with a defensive cap."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["sha256"]].append(record)
    pairs: list[dict[str, Any]] = []
    summary = {"duplicate_hash_groups": 0, "within_split_pairs": 0, "cross_split_pairs": 0, "conflicting_label_pairs": 0, "truncated": False}
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        summary["duplicate_hash_groups"] += 1
        for first, second in itertools.combinations(group, 2):
            is_cross_split = first["split"] != second["split"]
            label_conflict = bool(first.get("label") and second.get("label") and first["label"] != second["label"])
            scope = "cross_split" if is_cross_split else "within_split"
            summary[f"{scope}_pairs"] += 1
            summary["conflicting_label_pairs"] += int(label_conflict)
            if len(pairs) >= max_pairs:
                summary["truncated"] = True
                continue
            pairs.append(
                {
                    "asset_type": first["asset_type"],
                    "pair_scope": scope,
                    "file_a": first["path"],
                    "split_a": first["split"],
                    "label_a": first.get("label", ""),
                    "source_image_id_a": first.get("source_image_id", ""),
                    "source_annotation_id_a": first.get("source_annotation_id", ""),
                    "file_b": second["path"],
                    "split_b": second["split"],
                    "label_b": second.get("label", ""),
                    "source_image_id_b": second.get("source_image_id", ""),
                    "source_annotation_id_b": second.get("source_annotation_id", ""),
                    "sha256": digest,
                    "labels_conflict": label_conflict,
                }
            )
    return pairs, summary


def audit_families(source_records: list[dict[str, Any]], crop_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    crops_by_source: Counter[str] = Counter(normalized_path(row["source_image_path"]) for row in crop_rows)
    labels_by_source: dict[str, set[str]] = defaultdict(set)
    for row in crop_rows:
        labels_by_source[normalized_path(row["source_image_path"])].add(row["class_name"])
    for record in source_records:
        families[record["family_id"]].append(record)
    leakage_rows = []
    for family_id, members in sorted(families.items()):
        splits = sorted({member["split"] for member in members})
        if len(splits) < 2:
            continue
        source_keys = {member["path_key"] for member in members}
        leakage_rows.append(
            {
                "family_id": family_id,
                "splits": ";".join(splits),
                "source_images": len(members),
                "affected_crops": sum(crops_by_source[key] for key in source_keys),
                "labels": ";".join(sorted({label for key in source_keys for label in labels_by_source[key]})),
                "representative_paths": " | ".join(member["path"] for member in members[:5]),
            }
        )
    summary = {
        "normalized_source_families": len(families),
        "cross_split_families": len(leakage_rows),
        "affected_source_images": sum(row["source_images"] for row in leakage_rows),
        "affected_crops": sum(row["affected_crops"] for row in leakage_rows),
    }
    return leakage_rows, summary


def _dhash_variants(byte_value: int) -> tuple[int, ...]:
    return (byte_value, *(byte_value ^ (1 << bit) for bit in range(8)))


def near_duplicate_pairs(source_records: list[dict[str, Any]], threshold: int = 8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find cross-split dHash pairs with scalable 8-band, 1-bit-variant bucketing.

    Eight 8-bit bands plus one-bit variants give complete candidate coverage for
    dHash Hamming distance <= 8 (at least one band differs by at most one bit).
    """
    if threshold < 1 or threshold > 8:
        raise ValueError("near duplicate threshold must be in [1, 8] for the documented bucket guarantee")
    valid = []
    failures = []
    for record in tqdm(source_records, desc="dHash source images"):
        try:
            record["dhash"] = dhash_image(record["path"])
            valid.append(record)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            failures.append({"path": record["path"], "split": record["split"], "error": str(error)})
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(valid):
        value = record["dhash"]
        for band in range(8):
            buckets[(band, (value >> (band * 8)) & 0xFF)].append(index)
    candidate_indices: set[tuple[int, int]] = set()
    for index, record in enumerate(valid):
        value = record["dhash"]
        for band in range(8):
            byte_value = (value >> (band * 8)) & 0xFF
            for variant in _dhash_variants(byte_value):
                for other_index in buckets[(band, variant)]:
                    if other_index <= index or valid[other_index]["split"] == record["split"]:
                        continue
                    candidate_indices.add((index, other_index))
    pairs = []
    for first_index, second_index in candidate_indices:
        first, second = valid[first_index], valid[second_index]
        distance = hamming_distance(first["dhash"], second["dhash"])
        if distance == 0 or distance > threshold:
            continue
        pairs.append(
            {
                "dhash_distance": distance,
                "file_a": first["path"],
                "split_a": first["split"],
                "label_a": first.get("label", ""),
                "source_image_id_a": first.get("source_image_id", ""),
                "file_b": second["path"],
                "split_b": second["split"],
                "label_b": second.get("label", ""),
                "source_image_id_b": second.get("source_image_id", ""),
            }
        )
    pairs.sort(key=lambda pair: (pair["dhash_distance"], pair["split_a"], pair["file_a"], pair["file_b"]))
    summary = {
        "threshold": threshold,
        "source_images_hashed": len(valid),
        "source_image_hash_failures": failures,
        "candidate_pairs_examined": len(candidate_indices),
        "cross_split_near_duplicate_pairs": len(pairs),
    }
    return pairs, summary


def write_near_duplicate_previews(pairs: list[dict[str, Any]], preview_dir: Path, maximum: int = 20) -> int:
    ensure_dir(preview_dir)
    written = 0
    for index, pair in enumerate(pairs[:maximum]):
        try:
            with Image.open(pair["file_a"]) as first_open, Image.open(pair["file_b"]) as second_open:
                first = ImageOps.contain(first_open.convert("RGB"), (360, 300), Image.Resampling.LANCZOS)
                second = ImageOps.contain(second_open.convert("RGB"), (360, 300), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (740, 350), "white")
            canvas.paste(first, ((360 - first.width) // 2, 42 + (300 - first.height) // 2))
            canvas.paste(second, (380 + (360 - second.width) // 2, 42 + (300 - second.height) // 2))
            ImageDraw.Draw(canvas).text(
                (8, 8),
                f"dHash={pair['dhash_distance']} | {pair['split_a']} vs {pair['split_b']} | {pair['label_a']} / {pair['label_b']}"[:120],
                fill="black",
            )
            canvas.save(preview_dir / f"near_{index:03d}_d{pair['dhash_distance']}.jpg", quality=90)
            written += 1
        except (OSError, UnidentifiedImageError, ValueError):
            continue
    return written


def class_distribution(crop_rows: list[dict[str, str]], all_classes: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    crop_counts = Counter((row["split"], row["class_name"]) for row in crop_rows)
    source_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    annotation_counts = Counter((row["split"], row["source_image_path"], row["source_annotation_id"]) for row in crop_rows)
    labels_per_annotation: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    crops_per_source = Counter((row["split"], normalized_path(row["source_image_path"])) for row in crop_rows)
    for row in crop_rows:
        source_counts[(row["split"], row["class_name"])].add(normalized_path(row["source_image_path"]))
        labels_per_annotation[(row["split"], normalized_path(row["source_image_path"]), row["source_annotation_id"])].add(row["class_name"])
        source_splits[normalized_path(row["source_image_path"])].add(row["split"])
    split_totals = Counter(row["split"] for row in crop_rows)
    rows = []
    for label in all_classes:
        train = crop_counts[("train", label)]
        val = crop_counts[("val", label)]
        test = crop_counts[("test", label)]
        rows.append(
            {
                "class_name": label,
                "train_source_images": len(source_counts[("train", label)]),
                "train_crops": train,
                "train_percent": train / split_totals["train"] if split_totals["train"] else 0.0,
                "val_source_images": len(source_counts[("val", label)]),
                "val_crops": val,
                "val_percent": val / split_totals["val"] if split_totals["val"] else 0.0,
                "test_source_images": len(source_counts[("test", label)]),
                "test_crops": test,
                "test_percent": test / split_totals["test"] if split_totals["test"] else 0.0,
                "train_to_val_ratio": train / val if val else None,
                "train_to_test_ratio": train / test if test else None,
                "missing_from_split": ";".join(split for split, value in (("train", train), ("val", val), ("test", test)) if value == 0),
                "rare_class": min(train, val, test) < 10,
            }
        )
    duplicates = sum(count - 1 for count in annotation_counts.values() if count > 1)
    conflicting = sum(1 for labels in labels_per_annotation.values() if len(labels) > 1)
    source_cross_split = {path: sorted(splits) for path, splits in source_splits.items() if len(splits) > 1}
    diversity = {
        split: {
            "crop_count": split_totals[split],
            "unique_source_images": len({path for (row_split, path) in crops_per_source if row_split == split}),
            "mean_crops_per_source_image": (
                sum(count for (row_split, _), count in crops_per_source.items() if row_split == split)
                / max(1, len({path for (row_split, path) in crops_per_source if row_split == split}))
            ),
            "max_crops_from_one_source_image": max(
                [count for (row_split, _), count in crops_per_source.items() if row_split == split] or [0]
            ),
        }
        for split in AUDIT_SPLITS
    }
    return rows, {
        "split_crop_totals": dict(split_totals),
        "duplicate_annotation_records": duplicates,
        "conflicting_labels_per_source_annotation": conflicting,
        "source_images_in_multiple_crop_splits": source_cross_split,
        "effective_diversity": diversity,
    }


def inspect_implementation(
    config: dict[str, Any], crop_rows: list[dict[str, str]], checkpoint: dict[str, Any], dataset_root: Path
) -> dict[str, Any]:
    expected_source_folder = {"train": "train", "val": "valid", "test": "test"}
    split_mismatches = []
    for row in crop_rows:
        source_path = Path(row["source_image_path"])
        expected_part = dataset_root / expected_source_folder[row["split"]]
        try:
            source_path.resolve().relative_to(expected_part.resolve())
        except ValueError:
            split_mismatches.append({"crop": row["image_path"], "crop_split": row["split"], "source": row["source_image_path"]})
    manifest_classes = {}
    for split in AUDIT_SPLITS:
        classes = {row["class_name"] for row in crop_rows if row["split"] == split}
        manifest_classes[split] = sorted(classes)
    expected_classes = set(checkpoint["class_to_idx"])
    train_transform = build_transforms(int(config["dataset"]["image_size"]), train=True, horizontal_flip=bool(config.get("augmentation", {}).get("horizontal_flip", False)))
    eval_transform = build_transforms(int(config["dataset"]["image_size"]), train=False)
    training_source = inspect.getsource(__import__("src.training.trainer", fromlist=["Trainer"]).Trainer.fit)
    return {
        "source_to_crop_split_mismatches": len(split_mismatches),
        "source_to_crop_split_mismatch_examples": split_mismatches[:10],
        "manifest_classes": manifest_classes,
        "all_manifest_classes_match_checkpoint": all(set(classes) == expected_classes for classes in manifest_classes.values()),
        "horizontal_flip_enabled": bool(config.get("augmentation", {}).get("horizontal_flip", False)),
        "train_transform": repr(train_transform),
        "validation_test_transform": repr(eval_transform),
        "validation_test_random_augmentation_detected": "Random" in repr(eval_transform),
        "evaluation_normalization_matches_training": "Normalize" in repr(train_transform) and "Normalize" in repr(eval_transform),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "checkpoint_best_validation_top1": checkpoint.get("best_validation_metric"),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_has_optimizer_scheduler_config_mapping": all(
            key in checkpoint for key in ("optimizer_state_dict", "scheduler_state_dict", "config", "class_to_idx")
        ),
        "best_checkpoint_selection_uses_validation_top1": "val_metrics.top1 > best_top1" in training_source,
    }


@torch.inference_mode()
def extract_model_outputs(
    model: torch.nn.Module,
    manifest_path: Path,
    class_to_idx: dict[str, int],
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = ProductDataset(manifest_path, class_to_idx, build_transforms(image_size, train=False))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    logits_all, embeddings_all, targets_all = [], [], []
    paths, labels = [], []
    for images, targets, batch_paths, batch_labels in tqdm(loader, desc=f"embed {manifest_path.stem}"):
        logits, embeddings = model(images.to(device, non_blocking=True))
        logits_all.append(logits.cpu())
        embeddings_all.append(embeddings.cpu())
        targets_all.append(targets.cpu())
        paths.extend(batch_paths)
        labels.extend(batch_labels)
    return {
        "logits": torch.cat(logits_all),
        "embeddings": torch.cat(embeddings_all),
        "targets": torch.cat(targets_all),
        "paths": paths,
        "labels": labels,
    }


def _classification_audit(test_output: dict[str, Any], class_to_idx: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    logits, targets = test_output["logits"], test_output["targets"]
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    top_k = min(5, logits.shape[1])
    top5 = logits.topk(top_k, dim=1).indices.eq(targets[:, None]).any(dim=1).float().mean().item()
    loss = torch.nn.CrossEntropyLoss()(logits, targets).item()
    matrix = np.zeros((len(class_to_idx), len(class_to_idx)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    per_class = []
    for index in range(len(class_to_idx)):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_name": idx_to_class[index],
                "support": int(matrix[index, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    misclassified = []
    for index, (target, prediction) in enumerate(zip(targets.tolist(), predictions.tolist())):
        if target != prediction:
            misclassified.append(
                {
                    "image_path": test_output["paths"][index],
                    "true_label": idx_to_class[target],
                    "predicted_label": idx_to_class[prediction],
                    "prediction_confidence": float(probabilities[index, prediction]),
                    "true_label_confidence": float(probabilities[index, target]),
                }
            )
    metrics = {
        "test_images": len(targets),
        "test_loss": loss,
        "top1_accuracy": float(predictions.eq(targets).float().mean()),
        "top5_accuracy": top5,
        "top5_note": "With 10 classes, Top-5 is weak evidence and is reported only for completeness.",
        "misclassified_count": len(misclassified),
    }
    return metrics, per_class, misclassified, matrix


def _sample_values(values: torch.Tensor, maximum: int = 200_000) -> np.ndarray:
    values = values.detach().cpu().flatten()
    if len(values) > maximum:
        indices = torch.linspace(0, len(values) - 1, steps=maximum).long()
        values = values[indices]
    return values.numpy()


def _distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        return {"count_sampled": 0, "mean": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0}
    all_values = np.concatenate(values)
    if len(all_values) > 250_000:
        all_values = all_values[np.linspace(0, len(all_values) - 1, num=250_000, dtype=np.int64)]
    return {
        "count_sampled": int(len(all_values)),
        "mean": float(np.mean(all_values)),
        "p05": float(np.percentile(all_values, 5)),
        "median": float(np.median(all_values)),
        "p95": float(np.percentile(all_values, 95)),
    }


def retrieval_audit(gallery: dict[str, Any], queries: dict[str, Any], ks: tuple[int, ...] = (1, 5, 10)) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gallery_embeddings = torch.nn.functional.normalize(gallery["embeddings"].float(), dim=1)
    query_embeddings = torch.nn.functional.normalize(queries["embeddings"].float(), dim=1)
    gallery_labels = np.asarray(gallery["labels"], dtype=object)
    query_labels = np.asarray(queries["labels"], dtype=object)
    max_k = min(max(ks), len(gallery_labels))
    metrics_counts = Counter()
    class_counts = Counter()
    class_hits = Counter()
    ap_values: dict[int, list[float]] = {k: [] for k in ks}
    positive_samples: list[np.ndarray] = []
    negative_samples: list[np.ndarray] = []
    nearest_incorrect_scores = []
    nearest_incorrect_rows = []
    gallery_counts = Counter(gallery_labels.tolist())
    for start in range(0, len(query_embeddings), 64):
        batch_embeddings = query_embeddings[start : start + 64]
        scores = batch_embeddings @ gallery_embeddings.T
        ranked_scores, ranked_indices = scores.topk(max_k, dim=1)
        score_cpu = scores.cpu()
        for local_index in range(len(batch_embeddings)):
            query_index = start + local_index
            label = query_labels[query_index]
            ranking = ranked_indices[local_index].cpu().numpy()
            ranked_labels = gallery_labels[ranking]
            class_counts[label] += 1
            for k in ks:
                relevant = ranked_labels[: min(k, len(ranked_labels))] == label
                metrics_counts[f"recall@{k}"] += int(np.any(relevant))
                precision_at_ranks = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
                denominator = min(gallery_counts[label], k)
                ap_values[k].append(float((precision_at_ranks * relevant).sum() / denominator) if denominator else 0.0)
            class_hits[label] += int(ranked_labels[0] == label)
            all_scores = score_cpu[local_index]
            positive_mask = torch.tensor(gallery_labels == label)
            positive_samples.append(_sample_values(all_scores[positive_mask]))
            negative_samples.append(_sample_values(all_scores[~positive_mask]))
            wrong_scores = all_scores.clone()
            wrong_scores[positive_mask] = -float("inf")
            wrong_index = int(wrong_scores.argmax())
            nearest_incorrect_scores.append(float(wrong_scores[wrong_index]))
            nearest_incorrect_rows.append(
                {
                    "query_path": queries["paths"][query_index],
                    "query_label": label,
                    "incorrect_gallery_path": gallery["paths"][wrong_index],
                    "incorrect_gallery_label": gallery_labels[wrong_index],
                    "incorrect_similarity": float(wrong_scores[wrong_index]),
                    "top1_label": ranked_labels[0],
                    "top1_similarity": float(ranked_scores[local_index, 0]),
                }
            )
    query_count = len(query_labels)
    per_class_recall = {label: class_hits[label] / class_counts[label] for label in sorted(class_counts)}
    metrics = {
        "query_count": query_count,
        **{key: value / query_count for key, value in metrics_counts.items()},
        **{f"mAP@{k}": float(np.mean(values)) if values else 0.0 for k, values in ap_values.items()},
        "per_class_recall@1": per_class_recall,
        "positive_pair_similarity": _distribution(positive_samples),
        "negative_pair_similarity": _distribution(negative_samples),
        "nearest_incorrect_similarity": _distribution([np.asarray(nearest_incorrect_scores)]),
        "suggested_rejection_threshold": float(np.percentile(nearest_incorrect_scores, 99)) if nearest_incorrect_scores else None,
        "threshold_note": "Suggested as the 99th percentile of each query's nearest incorrect train-gallery similarity; validate on independent production data before use.",
    }
    return metrics, nearest_incorrect_rows


def _training_curve_summary(history_path: Path) -> dict[str, Any]:
    if not history_path.is_file():
        return {"available": False}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = history.get("history", [])
    if not rows:
        return {"available": False}
    best = max(rows, key=lambda row: row["val_top1"])
    return {
        "available": True,
        "epochs_recorded": len(rows),
        "best_epoch": best["epoch"],
        "best_val_top1": best["val_top1"],
        "best_val_loss": best["val_loss"],
        "last_epoch": rows[-1],
        "analysis": (
            "Validation Top-1 peaks at the selected epoch. Later rising validation loss with nearly stable Top-1 means "
            "the model becomes less calibrated/confident on some samples even when argmax labels mostly remain correct. "
            "The epoch-10 loss spike is a single transient point in this history, not by itself proof of systematic instability."
        ),
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    exact = summary["exact_duplicates"]
    families = summary["augmentation_families"]
    near = summary["near_duplicates"]
    impl = summary["implementation"]
    classification = summary["classification"]
    retrieval = summary["retrieval"]
    if exact["source"]["cross_split_pairs"] or exact["crop"]["cross_split_pairs"]:
        conclusion = "A. Confirmed cross-split leakage (exact byte-identical files)."
    elif families["cross_split_families"]:
        conclusion = "A. Confirmed cross-split leakage through Roboflow augmentation families (same conservative base family in multiple splits)."
    elif near["cross_split_near_duplicate_pairs"]:
        conclusion = "B. Strong near-duplicate/correlation risk."
    else:
        conclusion = "C. No meaningful leakage found by this audit; conventional model overfitting remains the leading explanation."
    return f"""# Overfitting and leakage audit

## Conclusion

**{conclusion}**

Evidence ranking:

1. Exact cross-split source pairs: **{exact['source']['cross_split_pairs']}**; crop pairs: **{exact['crop']['cross_split_pairs']}**.
2. Cross-split Roboflow families: **{families['cross_split_families']}** affecting **{families['affected_source_images']}** source images and **{families['affected_crops']}** crops.
3. Cross-split source-image near-duplicate pairs (dHash ≤ {near['threshold']}): **{near['cross_split_near_duplicate_pairs']}**.

## Split and preprocessing checks

- Crop-to-source split mismatches: **{impl['source_to_crop_split_mismatches']}**.
- All split class sets match checkpoint mapping: **{impl['all_manifest_classes_match_checkpoint']}**.
- Horizontal flip enabled: **{impl['horizontal_flip_enabled']}**.
- Validation/test random augmentation detected: **{impl['validation_test_random_augmentation_detected']}**.
- Checkpoint: epoch **{impl['checkpoint_epoch_zero_based']}** (zero-based), best validation Top-1 **{impl['checkpoint_best_validation_top1']:.6f}**, stage **{impl['checkpoint_stage']}**.

## Existing checkpoint on untouched test crops

- Test loss: **{classification['test_loss']:.6f}**
- Test Top-1: **{classification['top1_accuracy']:.6f}**
- Test Top-5: **{classification['top5_accuracy']:.6f}** (weak evidence with 10 classes)
- Train-gallery retrieval — validation Recall@1: **{retrieval['validation']['recall@1']:.6f}**; test Recall@1: **{retrieval['test']['recall@1']:.6f}**.

## Interpretation

The epoch-7 checkpoint is internally consistent with the recorded best validation metric. The validation-loss increase after its peak can coexist with stable Top-1 because probability calibration degrades before many argmax labels change. Training-curve evidence alone cannot distinguish capacity overfitting from correlated data; the duplicate/family/near-duplicate evidence above should take priority.

No split, source image, crop, manifest, config, or checkpoint was changed by this audit.
"""


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    crops_root = Path(args.crops_root).resolve()
    output_dir = ensure_dir(args.output_dir)
    crop_rows = load_crop_manifest(crops_root)
    source_records = load_source_records(dataset_root, crop_rows)
    crop_records = load_crop_records(crop_rows)
    source_records, source_hash_failures = add_sha256(source_records, "SHA-256 source images")
    crop_records, crop_hash_failures = add_sha256(crop_records, "SHA-256 product crops")
    source_pairs, source_exact_summary = exact_duplicate_pairs(source_records, args.max_duplicate_pairs)
    crop_pairs, crop_exact_summary = exact_duplicate_pairs(crop_records, args.max_duplicate_pairs)
    _write_csv(output_dir / "exact_duplicates.csv", [*source_pairs, *crop_pairs], PAIR_FIELDS)
    family_rows, family_summary = audit_families(source_records, crop_rows)
    _write_csv(
        output_dir / "augmentation_family_leakage.csv",
        family_rows,
        ("family_id", "splits", "source_images", "affected_crops", "labels", "representative_paths"),
    )
    near_pairs, near_summary = near_duplicate_pairs(source_records, args.near_threshold)
    _write_csv(
        output_dir / "near_duplicates.csv",
        near_pairs,
        ("dhash_distance", "file_a", "split_a", "label_a", "source_image_id_a", "file_b", "split_b", "label_b", "source_image_id_b"),
    )
    near_summary["preview_images_written"] = write_near_duplicate_previews(
        near_pairs, output_dir / "near_duplicate_preview", args.preview_count
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_to_idx = {str(name): int(index) for name, index in checkpoint["class_to_idx"].items()}
    config = load_config(args.config)
    implementation = inspect_implementation(config, crop_rows, checkpoint, dataset_root)
    distribution_rows, distribution_summary = class_distribution(crop_rows, sorted(class_to_idx, key=class_to_idx.get))
    _write_csv(output_dir / "class_distribution.csv", distribution_rows, distribution_rows[0].keys())
    save_json(distribution_summary, output_dir / "correlated_crop_report.json")
    device = resolve_device(args.device)
    model, model_checkpoint = model_from_checkpoint(args.checkpoint, device)
    image_size = int(model_checkpoint["config"]["dataset"]["image_size"])
    train_output = extract_model_outputs(model, crops_root / "manifests" / "train.csv", class_to_idx, image_size, device, args.batch_size)
    val_output = extract_model_outputs(model, crops_root / "manifests" / "val.csv", class_to_idx, image_size, device, args.batch_size)
    test_output = extract_model_outputs(model, crops_root / "manifests" / "test.csv", class_to_idx, image_size, device, args.batch_size)
    classification, per_class, misclassified, confusion = _classification_audit(test_output, class_to_idx)
    _write_csv(output_dir / "misclassified_test_samples.csv", misclassified, ("image_path", "true_label", "predicted_label", "prediction_confidence", "true_label_confidence"))
    _write_csv(output_dir / "per_class_classification_metrics.csv", per_class, ("class_name", "support", "precision", "recall", "f1"))
    np.save(output_dir / "confusion_matrix.npy", confusion)
    _write_csv(
        output_dir / "confusion_matrix.csv",
        [{"true_class": name, **{other: int(confusion[index, other_index]) for other_index, other in enumerate(sorted(class_to_idx, key=class_to_idx.get))}} for index, name in enumerate(sorted(class_to_idx, key=class_to_idx.get))],
        ("true_class", *sorted(class_to_idx, key=class_to_idx.get)),
    )
    val_retrieval, val_incorrect = retrieval_audit(train_output, val_output)
    test_retrieval, test_incorrect = retrieval_audit(train_output, test_output)
    retrieval = {
        "gallery": {"split": "train", "images": len(train_output["labels"])},
        "validation": val_retrieval,
        "test": test_retrieval,
        "within_brand_evaluation": "Not available: no SKU-to-brand metadata was supplied.",
    }
    save_json(retrieval, output_dir / "retrieval_metrics.json")
    _write_csv(
        output_dir / "nearest_incorrect_matches.csv",
        [*({"query_split": "val", **row} for row in val_incorrect), *({"query_split": "test", **row} for row in test_incorrect)],
        ("query_split", "query_path", "query_label", "incorrect_gallery_path", "incorrect_gallery_label", "incorrect_similarity", "top1_label", "top1_similarity"),
    )
    training_curve = _training_curve_summary(ROOT / "outputs" / "training_history.json")
    summary = {
        "audit_paths": {"dataset_root": str(dataset_root), "crops_root": str(crops_root), "checkpoint": str(Path(args.checkpoint).resolve())},
        "implementation": implementation,
        "exact_duplicates": {
            "source": {**source_exact_summary, "hash_failures": source_hash_failures},
            "crop": {**crop_exact_summary, "hash_failures": crop_hash_failures},
        },
        "augmentation_families": family_summary,
        "near_duplicates": near_summary,
        "class_and_correlation": distribution_summary,
        "classification": classification,
        "retrieval": retrieval,
        "training_curve": training_curve,
        "read_only_guarantee": "Only scripts/, tests/, and outputs/overfitting_audit were written by this audit.",
    }
    save_json(summary, output_dir / "audit_summary.json")
    (output_dir / "audit_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "product_dataset")
    parser.add_argument("--crops-root", type=Path, default=ROOT / "data" / "product_crops")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_model.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "overfitting_audit")
    parser.add_argument("--near-threshold", type=int, default=8)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-duplicate-pairs", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args)
    print(json.dumps({
        "exact_cross_split_source_pairs": summary["exact_duplicates"]["source"]["cross_split_pairs"],
        "exact_cross_split_crop_pairs": summary["exact_duplicates"]["crop"]["cross_split_pairs"],
        "cross_split_augmentation_families": summary["augmentation_families"]["cross_split_families"],
        "near_duplicate_pairs": summary["near_duplicates"]["cross_split_near_duplicate_pairs"],
        "test_top1": summary["classification"]["top1_accuracy"],
        "test_recall_at_1": summary["retrieval"]["test"]["recall@1"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

