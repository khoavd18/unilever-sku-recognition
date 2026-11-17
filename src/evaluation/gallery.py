"""Manual train-gallery/validation-query evaluation for the five-SKU AXE model.

The gallery prediction is the SKU of the single most similar training crop.
Equivalently, every distinct SKU is assigned the maximum cosine similarity of
any of its gallery images, and SKUs are ranked by that maximum.  Cosine scores
are similarities, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageOps
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.catalog.profiles import load_product_profile
from src.datasets.product_dataset import ProductDataset, build_transforms
from src.evaluation.metrics import classification_metrics_from_predictions
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.training.embedding_data import path_key, training_dataset_fingerprint
from src.utils.config import resolve_device
from src.utils.io import save_json


EXPECTED_SKUS = load_product_profile("axe5").legacy_class_names
EXPECTED_GALLERY_COUNT = 507
EXPECTED_QUERY_COUNT = 96
EXPECTED_CLASS_COUNT = 5
AGGREGATION_RULE = (
    "For each SKU, take the highest cosine similarity among all of its training-gallery "
    "images; rank SKUs by that maximum. The winning SKU is therefore the SKU of the "
    "single nearest gallery image."
)
PREDICTION_FIELDS = (
    "query_path",
    "true_sku",
    "predicted_sku",
    "correct",
    "cosine_score",
    "matched_gallery_path",
)
SOURCE_FIELDS = ("source_zip", "family_id", "source_image_id", "source_annotation_id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Selected AXE classification checkpoint.")
    parser.add_argument("--dataset-dir", default="data/axe_crops", help="Prepared AXE crop root.")
    parser.add_argument(
        "--query-split",
        choices=("val",),
        default="val",
        help="Only validation queries are supported; test data is deliberately excluded.",
    )
    parser.add_argument("--output-dir", required=True, help="A new, non-existing report directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device.")
    parser.add_argument(
        "--correct-sample-count",
        type=int,
        default=12,
        help="Reproducible number of correct queries to visualize (all errors are always shown).",
    )
    parser.add_argument(
        "--visual-candidates",
        type=int,
        default=5,
        help="Number of distinct-SKU nearest candidates shown beside each selected query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Visualization sampling seed; defaults to the checkpoint dataset seed.",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest was not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required_fields - fields)
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_dataset_mapping(dataset_dir: Path) -> dict[str, int]:
    mapping_path = dataset_dir / "class_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping was not found: {mapping_path}")
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("class_to_idx"), dict):
        raise ValueError(f"class_mapping.json has no class_to_idx object: {mapping_path}")
    mapping = {str(name): int(index) for name, index in payload["class_to_idx"].items()}
    validate_class_mapping(mapping, "dataset")
    return mapping


def validate_class_mapping(mapping: dict[str, int], source: str) -> None:
    if len(mapping) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"{source} must contain exactly {EXPECTED_CLASS_COUNT} AXE SKUs: {mapping}")
    if set(mapping) != set(EXPECTED_SKUS):
        raise ValueError(f"{source} SKU names differ from the verified AXE mapping: {mapping}")
    if sorted(mapping.values()) != list(range(EXPECTED_CLASS_COUNT)):
        raise ValueError(f"{source} class indices must be contiguous 0..4: {mapping}")


def _validate_split_rows(
    rows: list[dict[str, str]],
    split: str,
    dataset_dir: Path,
    class_to_idx: dict[str, int],
    expected_count: int | None,
) -> dict[str, Path]:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} crops, found {len(rows)}")
    labels = {row["class_name"] for row in rows}
    if labels != set(class_to_idx):
        raise ValueError(f"{split} manifest does not contain exactly the checkpoint SKUs: {sorted(labels)}")

    paths: dict[str, Path] = {}
    for row_number, row in enumerate(rows, start=2):
        if row["split"] != split:
            raise ValueError(
                f"{split} manifest row {row_number} declares split={row['split']!r}"
            )
        path = Path(row["image_path"]).resolve()
        if not _is_within(path, dataset_dir):
            raise ValueError(f"Manifest crop is outside --dataset-dir: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} crop at row {row_number}: {path}")
        key = path_key(path)
        if key in paths:
            raise ValueError(f"Duplicate crop path in {split} manifest: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"Unreadable {split} crop: {path}") from error
        paths[key] = path
    return paths


def audit_gallery_query_inputs(
    dataset_dir: Path,
    class_to_idx: dict[str, int],
    query_split: str = "val",
    expected_gallery_count: int | None = EXPECTED_GALLERY_COUNT,
    expected_query_count: int | None = EXPECTED_QUERY_COUNT,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Audit only train gallery and validation queries; raise on confirmed overlap."""

    dataset_dir = dataset_dir.resolve()
    if query_split != "val":
        raise ValueError("Only query_split='val' is allowed; test data must not be evaluated here")
    dataset_mapping = load_dataset_mapping(dataset_dir)
    if dataset_mapping != class_to_idx:
        raise ValueError(
            "Checkpoint class_to_idx does not exactly match data/axe_crops/class_mapping.json"
        )

    manifest_fields = {"image_path", "class_name", "split"}
    gallery_rows = _read_csv(dataset_dir / "manifests" / "train.csv", manifest_fields)
    query_rows = _read_csv(dataset_dir / "manifests" / "val.csv", manifest_fields)
    gallery_paths = _validate_split_rows(
        gallery_rows, "train", dataset_dir, class_to_idx, expected_gallery_count
    )
    query_paths = _validate_split_rows(
        query_rows, "val", dataset_dir, class_to_idx, expected_query_count
    )
    path_overlap = set(gallery_paths) & set(query_paths)
    if path_overlap:
        examples = [str(gallery_paths[key]) for key in sorted(path_overlap)[:5]]
        raise ValueError(f"Query crops occur directly in the training gallery: {examples}")

    crop_fields = {
        "generated_crop_path",
        "class_name",
        "class_index",
        "split",
        *SOURCE_FIELDS,
    }
    crop_rows = _read_csv(dataset_dir / "crop_manifest.csv", crop_fields)
    relevant_crop_rows = [row for row in crop_rows if row["split"] in {"train", "val"}]
    crop_by_path: dict[str, dict[str, str]] = {}
    for row in relevant_crop_rows:
        crop_path = Path(row["generated_crop_path"]).resolve()
        key = path_key(crop_path)
        if key in crop_by_path:
            raise ValueError(f"crop_manifest.csv has duplicate generated_crop_path: {crop_path}")
        crop_by_path[key] = row

    selected: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    for split, manifest_rows, manifest_paths in (
        ("train", gallery_rows, gallery_paths),
        ("val", query_rows, query_paths),
    ):
        for manifest_row in manifest_rows:
            key = path_key(Path(manifest_row["image_path"]))
            provenance = crop_by_path.get(key)
            if provenance is None:
                raise ValueError(
                    f"{split} manifest crop is absent from crop_manifest.csv: {manifest_paths[key]}"
                )
            if provenance["split"] != split or provenance["class_name"] != manifest_row["class_name"]:
                raise ValueError(f"Manifest/provenance mismatch for crop: {manifest_paths[key]}")
            expected_index = class_to_idx[manifest_row["class_name"]]
            try:
                provenance_index = int(provenance["class_index"])
            except ValueError as error:
                raise ValueError(f"Invalid class_index for crop: {manifest_paths[key]}") from error
            if provenance_index != expected_index:
                raise ValueError(f"Class-index mismatch for crop: {manifest_paths[key]}")
            missing_identifiers = [field for field in SOURCE_FIELDS if not provenance[field].strip()]
            if missing_identifiers:
                raise ValueError(
                    f"Missing source identifiers {missing_identifiers} for crop: {manifest_paths[key]}"
                )
            selected[split].append(provenance)

    families = {
        split: {row["family_id"] for row in rows} for split, rows in selected.items()
    }
    family_overlap = families["train"] & families["val"]
    if family_overlap:
        raise ValueError(
            "Confirmed source-family overlap between training gallery and validation queries: "
            f"{sorted(family_overlap)[:10]}"
        )

    hash_members: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"train": [], "val": []}
    )
    for split, paths in (("train", gallery_paths), ("val", query_paths)):
        for path in paths.values():
            hash_members[_sha256(path)][split].append(str(path))
    exact_overlap = {
        digest: members
        for digest, members in hash_members.items()
        if members["train"] and members["val"]
    }
    if exact_overlap:
        examples = list(exact_overlap.items())[:5]
        raise ValueError(
            "Confirmed byte-identical crops cross the train/validation boundary: "
            f"{examples}"
        )

    report: dict[str, Any] = {
        "status": "passed",
        "gallery_split": "train",
        "query_split": "val",
        "gallery_count": len(gallery_rows),
        "query_count": len(query_rows),
        "class_count": len(class_to_idx),
        "per_sku_counts": {
            "train": dict(sorted(Counter(row["class_name"] for row in gallery_rows).items())),
            "val": dict(sorted(Counter(row["class_name"] for row in query_rows).items())),
        },
        "source_identifier_fields_validated": list(SOURCE_FIELDS),
        "source_family_counts": {
            "train": len(families["train"]),
            "val": len(families["val"]),
        },
        "train_val_path_overlap": 0,
        "train_val_source_family_overlap": 0,
        "train_val_exact_duplicate_groups": 0,
        "test_data_loaded": False,
        "limitations": (
            "Manifest family identifiers and byte hashes can confirm listed overlaps, but cannot "
            "prove that separate identifiers came from independent capture sessions."
        ),
    }
    return report, gallery_rows, query_rows


def set_deterministic_evaluation(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def extract_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], list[str]]:
    """Return learned embeddings and logits without changing dataset order."""

    model.eval()
    embeddings: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    targets_all: list[int] = []
    paths_all: list[str] = []
    labels_all: list[str] = []
    for images, targets, paths, labels in tqdm(loader, desc=description, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits, learned_embeddings = model(images)
        embeddings.append(learned_embeddings.float().cpu())
        logits_all.append(logits.float().cpu())
        targets_all.extend(int(value) for value in targets.tolist())
        paths_all.extend(str(value) for value in paths)
        labels_all.extend(str(value) for value in labels)
    if not embeddings:
        raise ValueError(f"No samples were produced by {description}")
    # Normalize explicitly at evaluation time even though the current model also
    # normalizes its learned projection in forward().
    normalized = F.normalize(torch.cat(embeddings, dim=0), p=2, dim=1)
    return normalized, torch.cat(logits_all, dim=0), targets_all, paths_all, labels_all


def search_full_gallery(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: Sequence[str],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Search every image and rank distinct SKUs by their best image score."""

    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-two tensors")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ")
    if len(gallery_embeddings) != len(gallery_labels) or not len(gallery_labels):
        raise ValueError("Gallery embeddings and labels must have the same non-zero length")
    if not len(query_embeddings):
        raise ValueError("Query embeddings are empty")
    unknown = set(gallery_labels) - set(class_names)
    if unknown:
        raise ValueError(f"Gallery labels missing from class_names: {sorted(unknown)}")

    queries = F.normalize(query_embeddings.float(), p=2, dim=1)
    gallery = F.normalize(gallery_embeddings.float(), p=2, dim=1)
    similarities = queries @ gallery.T
    if not torch.isfinite(similarities).all():
        raise ValueError("Non-finite cosine similarity encountered")

    ranked_gallery_indices: list[list[int]] = []
    ranked_sku_candidates: list[list[dict[str, Any]]] = []
    nearest_indices: list[int] = []
    nearest_scores: list[float] = []
    predicted_labels: list[str] = []
    label_indices = {
        sku: [index for index, label in enumerate(gallery_labels) if label == sku]
        for sku in class_names
    }
    empty_skus = [sku for sku, indices in label_indices.items() if not indices]
    if empty_skus:
        raise ValueError(f"Gallery contains no images for SKUs: {empty_skus}")

    for query_index in range(len(queries)):
        scores = similarities[query_index].tolist()
        image_ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        nearest_index = image_ranking[0]
        candidates: list[dict[str, Any]] = []
        for class_index, sku in enumerate(class_names):
            sku_gallery_index = min(
                label_indices[sku], key=lambda index: (-scores[index], index)
            )
            candidates.append(
                {
                    "sku": sku,
                    "score": float(scores[sku_gallery_index]),
                    "gallery_index": sku_gallery_index,
                    "class_index": class_index,
                }
            )
        candidates.sort(
            key=lambda item: (-item["score"], item["gallery_index"], item["class_index"])
        )
        if candidates[0]["sku"] != gallery_labels[nearest_index]:
            raise AssertionError("Distinct-SKU ranking disagrees with nearest-image prediction")
        ranked_gallery_indices.append(image_ranking)
        ranked_sku_candidates.append(candidates)
        nearest_indices.append(nearest_index)
        nearest_scores.append(float(scores[nearest_index]))
        predicted_labels.append(gallery_labels[nearest_index])

    return {
        "similarities": similarities,
        "ranked_gallery_indices": ranked_gallery_indices,
        "ranked_sku_candidates": ranked_sku_candidates,
        "nearest_gallery_indices": nearest_indices,
        "nearest_scores": nearest_scores,
        "predicted_labels": predicted_labels,
    }


def protocol_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> dict[str, Any]:
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    try:
        targets = [class_to_idx[label] for label in true_labels]
        predictions = [class_to_idx[label] for label in predicted_labels]
    except KeyError as error:
        raise ValueError(f"Unknown metric label: {error.args[0]}") from error
    raw = classification_metrics_from_predictions(targets, predictions, class_names)
    return {
        "sample_count": raw["sample_count"],
        "top1_sku_accuracy": raw["accuracy"],
        "macro_recall": raw["macro_recall"],
        "per_sku": {
            sku: {
                "correct": values["correct"],
                "total": values["support"],
                "recall": values["recall"],
            }
            for sku, values in raw["per_class"].items()
        },
        "class_names": list(class_names),
        "confusion_matrix": raw["confusion_matrix"],
    }


def choose_visual_queries(
    correctness: Sequence[bool],
    correct_sample_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if correct_sample_count < 0:
        raise ValueError("--correct-sample-count must be non-negative")
    incorrect = [index for index, correct in enumerate(correctness) if not correct]
    correct = [index for index, is_correct in enumerate(correctness) if is_correct]
    rng = random.Random(seed)
    sampled = sorted(rng.sample(correct, min(correct_sample_count, len(correct))))
    return incorrect, sampled


def _fit_image(path: str, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (24, 24, 24))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def _draw_wrapped(draw: ImageDraw.ImageDraw, position: tuple[int, int], value: str) -> None:
    draw.multiline_text(position, textwrap.fill(value, width=31), fill="black", spacing=3)


def save_visual_examples(
    output_dir: Path,
    selected: Sequence[tuple[int, str]],
    query_paths: Sequence[str],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    ranked_candidates: Sequence[Sequence[dict[str, Any]]],
    gallery_paths: Sequence[str],
    candidate_count: int,
) -> list[dict[str, Any]]:
    if candidate_count < 1:
        raise ValueError("--visual-candidates must be at least 1")
    manifest: list[dict[str, Any]] = []
    cell_size, caption_height = 220, 126
    for query_index, category in selected:
        candidates = list(ranked_candidates[query_index][:candidate_count])
        columns = 1 + len(candidates)
        canvas = Image.new("RGB", (columns * cell_size, cell_size + caption_height), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(_fit_image(query_paths[query_index], cell_size), (0, 0))
        _draw_wrapped(
            draw,
            (7, cell_size + 5),
            f"QUERY\ntrue: {true_labels[query_index]}\npred: {predicted_labels[query_index]}",
        )
        for rank, candidate in enumerate(candidates, start=1):
            x = rank * cell_size
            gallery_index = int(candidate["gallery_index"])
            canvas.paste(_fit_image(gallery_paths[gallery_index], cell_size), (x, 0))
            _draw_wrapped(
                draw,
                (x + 7, cell_size + 5),
                f"SKU rank {rank}\n{candidate['sku']}\ncosine: {candidate['score']:.6f}",
            )
        category_dir = output_dir / "examples" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in Path(query_paths[query_index]).stem[:60]
        )
        target = category_dir / f"q{query_index:03d}_{safe_stem}.jpg"
        canvas.save(target, quality=92, subsampling=0)
        manifest.append(
            {
                "category": category,
                "query_index": query_index,
                "query_path": query_paths[query_index],
                "true_sku": true_labels[query_index],
                "predicted_sku": predicted_labels[query_index],
                "visual_path": str(target),
            }
        )
    return manifest


def _write_dict_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        class_names = metrics["class_names"]
        writer.writerow(["true_sku", *class_names])
        for sku, row in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([sku, *row])


def _metric_lines(title: str, metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Top-1 SKU accuracy: {metrics['top1_sku_accuracy']:.4f} "
        f"({sum(item['correct'] for item in metrics['per_sku'].values())}/{metrics['sample_count']})",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        "- Theo từng SKU:",
        "",
    ]
    for sku, values in metrics["per_sku"].items():
        lines.append(
            f"  - `{sku}`: {values['correct']}/{values['total']} đúng "
            f"(recall {values['recall']:.4f})"
        )
    lines.append("")
    lines.append("Ma trận nhầm lẫn (hàng = nhãn thật, cột = nhãn dự đoán):")
    lines.append("")
    header = "| true \\ pred | " + " | ".join(metrics["class_names"]) + " |"
    lines.extend([header, "|---|" + "---|" * len(metrics["class_names"])])
    for sku, row in zip(metrics["class_names"], metrics["confusion_matrix"]):
        lines.append(f"| {sku} | " + " | ".join(str(value) for value in row) + " |")
    lines.append("")
    return lines


def write_vietnamese_summary(
    path: Path,
    audit: dict[str, Any],
    gallery_metrics: dict[str, Any],
    classifier_metrics: dict[str, Any],
    incorrect_count: int,
    sampled_correct_count: int,
) -> None:
    lines = [
        "# Tóm tắt đánh giá gallery AXE Model 2",
        "",
        f"- Gallery: {audit['gallery_count']} crop train; query: {audit['query_count']} crop val; "
        f"số SKU: {audit['class_count']}.",
        "- Không dùng dữ liệu test và không lọc gallery theo brand.",
        f"- Kiểm tra chồng lấn: path = 0, family = {audit['train_val_source_family_overlap']}, "
        f"ảnh trùng byte = {audit['train_val_exact_duplicate_groups']}.",
        f"- Quy tắc gộp theo SKU: {AGGREGATION_RULE}",
        "",
    ]
    lines.extend(_metric_lines("Kết quả tìm kiếm bằng embedding", gallery_metrics))
    lines.extend(_metric_lines("Kết quả classifier (báo cáo riêng)", classifier_metrics))
    lines.extend(
        [
            "## Ảnh minh họa và giới hạn",
            "",
            f"Đã lưu toàn bộ {incorrect_count} query sai và mẫu cố định {sampled_correct_count} query đúng.",
            "",
            "Điểm cosine chỉ đo độ tương tự giữa embedding đã chuẩn hóa L2; đây không phải xác suất "
            "đã hiệu chuẩn. Báo cáo không đặt ngưỡng chấp nhận.",
            "",
            "Thử nghiệm closed-set với 5 SKU này không xác nhận hiệu năng hoặc khả năng mở rộng cho "
            "gallery 400 SKU. Cần một đánh giá riêng trên dữ liệu và quy mô mục tiêu.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_cli_values(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.correct_sample_count < 0:
        raise ValueError("--correct-sample-count must be non-negative")
    if not 1 <= args.visual_candidates <= EXPECTED_CLASS_COUNT:
        raise ValueError(f"--visual-candidates must be between 1 and {EXPECTED_CLASS_COUNT}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_cli_values(args)
    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory was not found: {dataset_dir}")

    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    validate_class_mapping(class_to_idx, "checkpoint")
    model_config = checkpoint.get("config", {}).get("model", {})
    if model_config.get("name") != "mobilenet_v3_large":
        raise ValueError(f"Unsupported checkpoint architecture: {model_config.get('name')!r}")
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    if image_size < 1:
        raise ValueError(f"Invalid checkpoint image_size: {image_size}")

    audit, _, _ = audit_gallery_query_inputs(dataset_dir, class_to_idx, args.query_split)
    current_fingerprint = training_dataset_fingerprint(dataset_dir)
    checkpoint_fingerprint = checkpoint.get("dataset_fingerprint")
    if not checkpoint_fingerprint:
        raise ValueError("Checkpoint has no dataset_fingerprint")
    if checkpoint_fingerprint != current_fingerprint:
        raise ValueError(
            "Checkpoint dataset_fingerprint does not match current mapping/manifests/provenance"
        )
    audit["dataset_fingerprint"] = current_fingerprint
    audit["checkpoint_dataset_fingerprint_match"] = True

    seed = int(
        args.seed
        if args.seed is not None
        else checkpoint["config"].get("dataset", {}).get("seed", 42)
    )
    set_deterministic_evaluation(seed)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    transform = build_transforms(image_size, train=False)
    gallery_dataset = ProductDataset(
        dataset_dir / "manifests" / "train.csv", class_to_idx, transform
    )
    query_dataset = ProductDataset(
        dataset_dir / "manifests" / "val.csv", class_to_idx, transform
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=min(args.batch_size, len(gallery_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=min(args.batch_size, len(query_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    amp_enabled = bool(checkpoint["config"]["training"].get("amp", True)) and device.type == "cuda"

    # Atomic mkdir is also the final race-safe overwrite guard.  Nothing has
    # been written before all metadata/provenance checks pass.
    output_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    with torch.inference_mode():
        gallery_embeddings, _, _, gallery_paths, gallery_labels = extract_outputs(
            model, gallery_loader, device, amp_enabled, "train gallery embeddings"
        )
        query_embeddings, query_logits, _, query_paths, true_labels = extract_outputs(
            model, query_loader, device, amp_enabled, "validation query embeddings"
        )

    search = search_full_gallery(
        query_embeddings, gallery_embeddings, gallery_labels, class_names
    )
    retrieval_predictions = search["predicted_labels"]
    gallery_metrics = protocol_metrics(true_labels, retrieval_predictions, class_names)
    gallery_metrics.update(
        {
            "protocol": "full_train_gallery_cosine_nearest_image",
            "gallery_count": len(gallery_paths),
            "query_count": len(query_paths),
            "embedding_dimension": int(query_embeddings.shape[1]),
            "embedding_l2_normalized": True,
            "brand_filter": False,
            "sku_aggregation_rule": AGGREGATION_RULE,
            "cosine_scores_are_calibrated_probabilities": False,
            "acceptance_threshold": None,
        }
    )

    classifier_indices = query_logits.argmax(dim=1).tolist()
    classifier_predictions = [class_names[index] for index in classifier_indices]
    classifier_metrics = protocol_metrics(true_labels, classifier_predictions, class_names)
    classifier_metrics.update(
        {
            "protocol": "checkpoint_classifier_logits_argmax",
            "reported_separately_from_gallery_search": True,
        }
    )

    correctness = [
        prediction == target
        for prediction, target in zip(retrieval_predictions, true_labels)
    ]
    prediction_rows = []
    candidate_rows = []
    classifier_rows = []
    for query_index, (query_path, true_sku) in enumerate(zip(query_paths, true_labels)):
        nearest_index = search["nearest_gallery_indices"][query_index]
        prediction_rows.append(
            {
                "query_path": query_path,
                "true_sku": true_sku,
                "predicted_sku": retrieval_predictions[query_index],
                "correct": correctness[query_index],
                "cosine_score": f"{search['nearest_scores'][query_index]:.8f}",
                "matched_gallery_path": gallery_paths[nearest_index],
            }
        )
        for rank, candidate in enumerate(search["ranked_sku_candidates"][query_index], start=1):
            candidate_rows.append(
                {
                    "query_path": query_path,
                    "true_sku": true_sku,
                    "sku_rank": rank,
                    "candidate_sku": candidate["sku"],
                    "cosine_score": f"{candidate['score']:.8f}",
                    "matched_gallery_path": gallery_paths[candidate["gallery_index"]],
                }
            )
        classifier_rows.append(
            {
                "query_path": query_path,
                "true_sku": true_sku,
                "predicted_sku": classifier_predictions[query_index],
                "correct": classifier_predictions[query_index] == true_sku,
            }
        )

    _write_dict_rows(output_dir / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
    _write_dict_rows(
        output_dir / "sku_candidates.csv",
        (
            "query_path",
            "true_sku",
            "sku_rank",
            "candidate_sku",
            "cosine_score",
            "matched_gallery_path",
        ),
        candidate_rows,
    )
    _write_dict_rows(
        output_dir / "classifier_predictions.csv",
        ("query_path", "true_sku", "predicted_sku", "correct"),
        classifier_rows,
    )
    _write_confusion_matrix(output_dir / "gallery_confusion_matrix.csv", gallery_metrics)
    _write_confusion_matrix(output_dir / "classifier_confusion_matrix.csv", classifier_metrics)
    per_sku_rows = []
    for protocol, metrics in (
        ("gallery_search", gallery_metrics),
        ("classifier", classifier_metrics),
    ):
        for sku, values in metrics["per_sku"].items():
            per_sku_rows.append({"protocol": protocol, "sku": sku, **values})
    _write_dict_rows(
        output_dir / "per_sku_metrics.csv",
        ("protocol", "sku", "correct", "total", "recall"),
        per_sku_rows,
    )

    incorrect, sampled_correct = choose_visual_queries(
        correctness, args.correct_sample_count, seed
    )
    selected = [(index, "incorrect") for index in incorrect] + [
        (index, "correct_sample") for index in sampled_correct
    ]
    visual_manifest = save_visual_examples(
        output_dir,
        selected,
        query_paths,
        true_labels,
        retrieval_predictions,
        search["ranked_sku_candidates"],
        gallery_paths,
        args.visual_candidates,
    )
    _write_dict_rows(
        output_dir / "visual_examples.csv",
        ("category", "query_index", "query_path", "true_sku", "predicted_sku", "visual_path"),
        visual_manifest,
    )

    run_metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": seed,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_architecture": model_config.get("name"),
        "embedding_dimension": model_config.get("embedding_dim"),
        "preprocessing": {
            "image_size": image_size,
            "training_augmentation": False,
            "transform": repr(transform),
        },
        "gallery_split": "train",
        "query_split": "val",
        "test_data_loaded": False,
        "sku_aggregation_rule": AGGREGATION_RULE,
        "correct_visual_sample_count": len(sampled_correct),
        "incorrect_visual_count": len(incorrect),
    }
    save_json(audit, output_dir / "input_audit.json")
    save_json(gallery_metrics, output_dir / "gallery_metrics.json")
    save_json(classifier_metrics, output_dir / "classifier_metrics.json")
    save_json(run_metadata, output_dir / "run_metadata.json")
    write_vietnamese_summary(
        output_dir / "summary_vi.md",
        audit,
        gallery_metrics,
        classifier_metrics,
        len(incorrect),
        len(sampled_correct),
    )

    console_summary = {
        "output_dir": str(output_dir),
        "gallery_count": len(gallery_paths),
        "query_count": len(query_paths),
        "gallery_top1_sku_accuracy": gallery_metrics["top1_sku_accuracy"],
        "gallery_macro_recall": gallery_metrics["macro_recall"],
        "classifier_top1_sku_accuracy": classifier_metrics["top1_sku_accuracy"],
        "classifier_macro_recall": classifier_metrics["macro_recall"],
        "incorrect_visuals": len(incorrect),
        "sampled_correct_visuals": len(sampled_correct),
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
