"""Create immutable-source COCO instance crops and retrieval-ready manifests.

The source COCO export is never modified. Each crop is traced to one source image
and COCO annotation in ``crop_manifest.csv`` and ``crop_manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_json
from src.data.annotations import validate_coco_document
from src.data.crops import (
    bbox_to_crop_box,
    make_crop_filename,
    sanitize_class_name,
    stable_identity_hash,
)


SOURCE_SPLITS = {"train": "train", "val": "valid", "test": "test"}
WINDOWS_SAFE_PATH_LIMIT = 240
SOURCE_FAMILY_MANIFEST_NAME = "source_family_manifest.json"
CROP_MANIFEST_FIELDS = (
    "split",
    "image_path",
    "generated_crop_path",
    "class_name",
    "original_sku_name",
    "class_index",
    "class_directory",
    "category_id",
    "category_name",
    "source_zip",
    "original_split",
    "original_filename",
    "family_id",
    "source_image_path",
    "source_image_id",
    "original_source_image_id",
    "source_annotation_id",
    "source_bbox_x",
    "source_bbox_y",
    "source_bbox_width",
    "source_bbox_height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_width",
    "crop_height",
)
TRAINING_MANIFEST_FIELDS = ("image_path", "class_name", "split")


def normalize_source_family(filename_or_path: str | Path) -> str:
    """Remove only Roboflow's terminal hash, _jpg, and _augN markers."""

    normalized = str(filename_or_path).replace("\\", "/")
    stem = Path(normalized.rsplit("/", 1)[-1]).stem
    stem = re.sub(r"\.rf\.[^.]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_aug\d+$", "", stem, flags=re.IGNORECASE)
    return stem.casefold()


def _annotation_id(annotation: dict[str, Any], fallback: int) -> int:
    try:
        return int(annotation.get("id", fallback))
    except (TypeError, ValueError):
        return fallback


def _category_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_excluded_category(name: str) -> str | None:
    normalized = str(name).strip().casefold()
    if normalized == "stock":
        return "excluded_stock_supercategory"
    if normalized == "0":
        return "excluded_category_name_0"
    return None


def _source_annotation_paths(dataset_root: Path) -> dict[str, Path]:
    return {split: dataset_root / source_split / "_annotations.coco.json" for split, source_split in SOURCE_SPLITS.items()}


def validate_dataset_root(dataset_path: str | Path) -> Path:
    """Validate direct COCO root and report accidental archive double nesting."""
    root = Path(dataset_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO dataset directory does not exist: {root}")
    expected = _source_annotation_paths(root)
    if all(path.is_file() for path in expected.values()):
        return root
    nested_roots = []
    for child in root.iterdir():
        if child.is_dir() and all(path.is_file() for path in _source_annotation_paths(child).values()):
            nested_roots.append(child)
    if nested_roots:
        locations = ", ".join(str(path) for path in nested_roots)
        raise ValueError(
            "Detected double-nested COCO dataset root. Source files were not changed. "
            f"Use the nested directory explicitly with --dataset-path: {locations}"
        )
    missing = ", ".join(str(path) for path in expected.values() if not path.is_file())
    raise FileNotFoundError(f"COCO annotation JSON files are missing under {root}: {missing}")


def load_coco_split(annotation_path: Path) -> dict[str, Any]:
    """Load only standard COCO lists and validate the minimum required schema."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    return validate_coco_document(coco, annotation_path)


def build_category_mapping(coco_splits: dict[str, dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Build one consistent category-id mapping for train/valid/test."""
    original_names: dict[int, str] = {}
    for split, coco in coco_splits.items():
        for category in coco["categories"]:
            category_id = _category_id(category.get("id"))
            if category_id is None:
                continue
            name = str(category.get("name", category_id))
            if category_id in original_names and original_names[category_id] != name:
                raise ValueError(
                    f"Category ID {category_id} has inconsistent names: '{original_names[category_id]}' and '{name}' ({split})."
                )
            original_names[category_id] = name
    mapping: dict[int, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    used_directories: set[str] = set()
    class_index = 0
    for category_id, original_name in sorted(original_names.items()):
        skip_reason = _is_excluded_category(original_name)
        if skip_reason:
            skipped.append({"category_id": category_id, "category_name": original_name, "reason": skip_reason})
            continue
        directory_name = sanitize_class_name(original_name, fallback=f"category_{category_id}")
        if directory_name.casefold() in used_directories:
            directory_name = f"{directory_name}__category_{category_id}"
        used_directories.add(directory_name.casefold())
        mapping[category_id] = {
            "category_id": category_id,
            "category_name": original_name,
            "directory_name": directory_name,
            "class_index": class_index,
        }
        class_index += 1
    return mapping, skipped


def _shorten_class_directory(
    directory_name: str,
    category_name: str,
    category_id: int,
    class_index: int,
    max_length: int,
) -> str:
    digest = stable_identity_hash(category_id, category_name, length=8)
    suffix = f"_{digest}"
    minimum = f"c{class_index:03d}{suffix}"
    if max_length < len(minimum):
        return minimum
    prefix = directory_name[: max_length - len(suffix)].rstrip(" ._")
    return f"{prefix or f'c{class_index:03d}'}{suffix}"


def fit_class_directories_for_windows(
    category_mapping: dict[int, dict[str, Any]],
    coco_splits: dict[str, dict[str, Any]],
    output_dir: Path,
    path_limit: int = WINDOWS_SAFE_PATH_LIMIT,
) -> None:
    """Shorten only class directory components that could exceed path_limit."""

    maximum_filename_lengths: Counter[int] = Counter()
    for split, coco in coco_splits.items():
        images_by_id = {
            _category_id(image.get("id")): image
            for image in coco["images"]
            if _category_id(image.get("id")) is not None
        }
        for fallback_id, annotation in enumerate(coco["annotations"], start=1):
            category_id = _category_id(annotation.get("category_id"))
            image_id = _category_id(annotation.get("image_id"))
            if category_id not in category_mapping or image_id not in images_by_id:
                continue
            category = category_mapping[category_id]
            filename = make_crop_filename(
                str(images_by_id[image_id].get("file_name", "image")),
                image_id,
                _annotation_id(annotation, fallback_id),
                category["class_index"],
                original_split=SOURCE_SPLITS[split],
            )
            maximum_filename_lengths[category_id] = max(
                maximum_filename_lengths[category_id], len(filename)
            )

    output_dir = output_dir.resolve()
    longest_split_prefix = max(
        len(str((output_dir / source_split).resolve())) for source_split in SOURCE_SPLITS.values()
    )
    used_directories: set[str] = set()
    for category_id in sorted(category_mapping):
        category = category_mapping[category_id]
        directory_name = category["directory_name"]
        filename_length = maximum_filename_lengths[category_id] or len(
            make_crop_filename("image.jpg", 0, 0, category["class_index"])
        )
        # prefix + separator + directory + separator + filename must be < path_limit.
        max_directory_length = path_limit - longest_split_prefix - filename_length - 3
        candidate_length = longest_split_prefix + 1 + len(directory_name) + 1 + filename_length
        shortened = candidate_length >= path_limit
        if shortened:
            directory_name = _shorten_class_directory(
                directory_name,
                category["category_name"],
                category_id,
                category["class_index"],
                max_directory_length,
            )
        if directory_name.casefold() in used_directories:
            raise ValueError(
                f"Windows-safe class directory collision for category {category['category_name']!r}: "
                f"{directory_name!r}"
            )
        used_directories.add(directory_name.casefold())
        category["original_directory_name"] = category["directory_name"]
        category["directory_name"] = directory_name
        category["directory_name_was_shortened"] = shortened


def load_source_provenance(dataset_root: Path) -> dict[str, dict[Any, dict[str, Any]]]:
    """Index optional clean-dataset family metadata by image ID and filename."""

    indexes: dict[str, dict[Any, dict[str, Any]]] = {"by_key": {}, "by_filename": {}}
    manifest_path = dataset_root / SOURCE_FAMILY_MANIFEST_NAME
    if not manifest_path.is_file():
        return indexes
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    families = manifest.get("families", []) if isinstance(manifest, dict) else []
    if not isinstance(families, list):
        raise ValueError(f"Invalid source-family manifest: {manifest_path}")
    for family in families:
        if not isinstance(family, dict) or not isinstance(family.get("images", []), list):
            raise ValueError(f"Invalid family record in {manifest_path}")
        for image in family.get("images", []):
            if not isinstance(image, dict):
                raise ValueError(f"Invalid image provenance record in {manifest_path}")
            provenance = dict(image)
            provenance["family_id"] = str(family.get("family_id", ""))
            output_image_id = _category_id(image.get("output_image_id"))
            destination = str(image.get("destination_file_name", ""))
            key = (output_image_id, destination)
            if key in indexes["by_key"] and indexes["by_key"][key] != provenance:
                raise ValueError(f"Conflicting source provenance for image {key!r}")
            indexes["by_key"][key] = provenance
            if destination:
                if (
                    destination in indexes["by_filename"]
                    and indexes["by_filename"][destination] != provenance
                ):
                    raise ValueError(f"Conflicting source provenance for filename {destination!r}")
                indexes["by_filename"][destination] = provenance
    return indexes


def source_traceability(
    image_info: dict[str, Any],
    split: str,
    provenance_indexes: dict[str, dict[Any, dict[str, Any]]],
) -> dict[str, Any]:
    """Resolve original ZIP identity, falling back to standard COCO image fields."""

    image_id = _category_id(image_info.get("id"))
    dataset_filename = str(image_info.get("file_name", ""))
    provenance = provenance_indexes["by_key"].get((image_id, dataset_filename))
    if provenance is None:
        provenance = provenance_indexes["by_filename"].get(dataset_filename, {})
    source_zip = str(
        provenance.get(
            "source_archive", image_info.get("source_zip", image_info.get("source_archive", ""))
        )
    )
    original_split = str(
        provenance.get("source_split", image_info.get("source_split", SOURCE_SPLITS[split]))
    )
    original_filename = str(
        provenance.get(
            "source_file_name",
            image_info.get("original_filename", image_info.get("source_file_name", dataset_filename)),
        )
    )
    original_source_image_id = provenance.get(
        "source_image_id", image_info.get("original_source_image_id", image_id)
    )
    family_id = str(
        provenance.get(
            "family_id",
            image_info.get("family_id", image_info.get("source_family_id", "")),
        )
    ) or normalize_source_family(original_filename)
    return {
        "source_zip": source_zip,
        "original_split": original_split,
        "original_filename": original_filename,
        "original_source_image_id": original_source_image_id,
        "family_id": family_id,
    }


def windows_safe_crop_path(
    output_dir: Path,
    output_split: str,
    class_directory: str,
    *,
    source_file_name: str,
    image_id: int,
    annotation_id: int,
    class_index: int,
    source_zip: str,
    original_split: str,
    original_image_id: Any,
    path_limit: int = WINDOWS_SAFE_PATH_LIMIT,
) -> Path:
    """Build a path below the Windows-safe limit, or fail with a clear error."""

    common = {
        "source_file_name": source_file_name,
        "image_id": image_id,
        "annotation_id": annotation_id,
        "class_index": class_index,
        "source_zip": source_zip,
        "original_split": original_split,
        "original_image_id": original_image_id,
    }
    filename = make_crop_filename(**common)
    crop_path = output_dir / output_split / class_directory / filename
    if len(str(crop_path.resolve())) < path_limit:
        return crop_path
    fallback = make_crop_filename(**common, compact=True)
    fallback_path = output_dir / output_split / class_directory / fallback
    if len(str(fallback_path.resolve())) < path_limit:
        return fallback_path
    raise OSError(
        f"Cannot create a Windows-safe crop path below {path_limit} characters. "
        f"Shortest candidate is {len(str(fallback_path.resolve()))} characters: {fallback_path}. "
        "Use a shorter --output-dir."
    )


def _resolve_source_image(split_directory: Path, file_name: Any) -> Path | None:
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    source_root = split_directory.resolve()
    candidate = (split_directory / file_name).resolve()
    if candidate != source_root and source_root not in candidate.parents:
        return None
    return candidate


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _preview_image(source: Image.Image, crop: Image.Image, box: tuple[int, int, int, int], label: str) -> Image.Image:
    """Render one compact source-with-box / crop pair without altering source imagery."""
    annotated = source.copy()
    ImageDraw.Draw(annotated).rectangle(box, outline=(255, 40, 40), width=max(2, min(6, source.width // 120)))
    source_thumb = ImageOps.contain(annotated, (360, 300), Image.Resampling.LANCZOS)
    crop_thumb = ImageOps.contain(crop, (300, 300), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (700, 350), "white")
    canvas.paste(source_thumb, ((360 - source_thumb.width) // 2, 34 + (300 - source_thumb.height) // 2))
    canvas.paste(crop_thumb, (390 + (300 - crop_thumb.width) // 2, 34 + (300 - crop_thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Source + COCO bbox | Crop: {label}"[:110], fill="black")
    return canvas


def _prepare_split(
    split: str,
    coco: dict[str, Any],
    source_split_dir: Path,
    output_dir: Path,
    category_mapping: dict[int, dict[str, Any]],
    padding: float,
    min_size: int,
    limit_images: int | None,
    dry_run: bool,
    overwrite: bool,
    preview_count_remaining: int,
    provenance_indexes: dict[str, dict[Any, dict[str, Any]]],
    generated_filenames_in_run: dict[str, str],
    generated_paths_in_run: dict[Path, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, int, int, Image.Image]]]:
    images = [image for image in coco["images"] if _category_id(image.get("id")) is not None]
    images.sort(key=lambda image: (str(image.get("file_name", "")), int(image["id"])))
    selected_images = images[:limit_images] if limit_images is not None else images
    selected_ids = {int(image["id"]) for image in selected_images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    annotations_selected = 0
    for annotation in coco["annotations"]:
        image_id = _category_id(annotation.get("image_id"))
        if image_id is None or image_id not in {int(image["id"]) for image in images}:
            skipped["missing_image_record"] += 1
            continue
        if image_id not in selected_ids:
            continue
        annotations_by_image[image_id].append(annotation)
        annotations_selected += 1
    crop_rows: list[dict[str, Any]] = []
    preview_items: list[tuple[str, int, int, Image.Image]] = []
    crop_widths: list[int] = []
    crop_heights: list[int] = []
    crops_by_sku = Counter()
    created_crops = 0
    existing_crops = 0
    invalid_or_missing_images = 0
    for image_info in selected_images:
        image_id = int(image_info["id"])
        image_path = _resolve_source_image(source_split_dir, image_info.get("file_name"))
        image_annotations = annotations_by_image.get(image_id, [])
        traceability = source_traceability(image_info, split, provenance_indexes)
        if image_path is None or not image_path.is_file():
            invalid_or_missing_images += 1
            skipped["missing_source_image"] += len(image_annotations)
            continue
        try:
            with Image.open(image_path) as opened:
                opened.load()
                source_image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            invalid_or_missing_images += 1
            skipped["corrupt_source_image"] += len(image_annotations)
            continue
        for fallback_id, annotation in enumerate(image_annotations, start=1):
            annotation_id = _annotation_id(annotation, fallback_id)
            category_id = _category_id(annotation.get("category_id"))
            if category_id not in category_mapping:
                skipped["excluded_category" if category_id is not None else "invalid_category_id"] += 1
                continue
            crop_box = bbox_to_crop_box(annotation.get("bbox"), source_image.width, source_image.height, padding)
            if crop_box is None:
                skipped["invalid_or_outside_bbox"] += 1
                continue
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            if crop_width < min_size or crop_height < min_size:
                skipped["crop_too_small"] += 1
                continue
            category = category_mapping[category_id]
            crop_path = windows_safe_crop_path(
                output_dir,
                SOURCE_SPLITS[split],
                category["directory_name"],
                source_file_name=traceability["original_filename"],
                image_id=image_id,
                annotation_id=annotation_id,
                class_index=category["class_index"],
                source_zip=traceability["source_zip"],
                original_split=traceability["original_split"],
                original_image_id=traceability["original_source_image_id"],
            )
            identity = (
                f"sku={category['category_name']!r}, source_zip={traceability['source_zip']!r}, "
                f"split={traceability['original_split']!r}, "
                f"filename={traceability['original_filename']!r}, image_id={image_id}, "
                f"annotation_id={annotation_id}"
            )
            if crop_path.name in generated_filenames_in_run:
                previous = generated_filenames_in_run[crop_path.name]
                raise RuntimeError(
                    f"Generated crop filename collision for {crop_path.name!r}. "
                    f"First identity: {previous}. Conflicting identity: {identity}."
                )
            if crop_path in generated_paths_in_run:
                previous = generated_paths_in_run[crop_path]
                raise RuntimeError(
                    f"Generated crop path collision for {crop_path}. "
                    f"First identity: {previous}. Conflicting identity: {identity}."
                )
            generated_filenames_in_run[crop_path.name] = identity
            generated_paths_in_run[crop_path] = identity
            crop = source_image.crop(crop_box)
            if crop.width <= 0 or crop.height <= 0:
                skipped["empty_crop"] += 1
                continue
            exists = crop_path.exists()
            if not dry_run and (overwrite or not exists):
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, format="JPEG", quality=95)
                created_crops += 1
            elif exists:
                existing_crops += 1
            crop_widths.append(crop_width)
            crop_heights.append(crop_height)
            crops_by_sku[category["category_name"]] += 1
            crop_rows.append(
                {
                    "split": split,
                    "image_path": str(crop_path.resolve()),
                    "generated_crop_path": str(crop_path.resolve()),
                    "class_name": category["category_name"],
                    "original_sku_name": category["category_name"],
                    "class_index": category["class_index"],
                    "class_directory": category["directory_name"],
                    "category_id": category_id,
                    "category_name": category["category_name"],
                    "source_zip": traceability["source_zip"],
                    "original_split": traceability["original_split"],
                    "original_filename": traceability["original_filename"],
                    "family_id": traceability["family_id"],
                    "source_image_path": str(image_path.resolve()),
                    "source_image_id": image_id,
                    "original_source_image_id": traceability["original_source_image_id"],
                    "source_annotation_id": annotation_id,
                    "source_bbox_x": annotation["bbox"][0],
                    "source_bbox_y": annotation["bbox"][1],
                    "source_bbox_width": annotation["bbox"][2],
                    "source_bbox_height": annotation["bbox"][3],
                    "crop_left": crop_box[0],
                    "crop_top": crop_box[1],
                    "crop_right": crop_box[2],
                    "crop_bottom": crop_box[3],
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                }
            )
            if not dry_run and len(preview_items) < preview_count_remaining:
                preview_items.append((split, image_id, annotation_id, _preview_image(source_image, crop, crop_box, category["category_name"])))
    report = {
        "source_images_total": len(images),
        "source_images_selected": len(selected_images),
        "source_annotations_total": len(coco["annotations"]),
        "source_annotations_selected": annotations_selected,
        "successful_crops": len(crop_rows),
        "created_crops": created_crops,
        "existing_crops_not_overwritten": existing_crops,
        "invalid_or_missing_images": invalid_or_missing_images,
        "crops_by_sku": dict(sorted(crops_by_sku.items())),
        "skipped_annotations_by_reason": dict(sorted(skipped.items())),
        "crop_width": _summary(crop_widths),
        "crop_height": _summary(crop_heights),
    }
    return report, crop_rows, preview_items


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_previews(preview_items: list[tuple[str, int, int, Image.Image]], preview_dir: Path, overwrite: bool) -> int:
    if not preview_items:
        return 0
    ensure_dir(preview_dir)
    written = 0
    for split, image_id, annotation_id, preview in preview_items:
        path = preview_dir / f"{split}__image_{image_id}__annotation_{annotation_id}.jpg"
        if path.exists() and not overwrite:
            continue
        preview.save(path, format="JPEG", quality=90)
        written += 1
    return written


def prepare_coco_crops(
    dataset_path: str | Path,
    output_dir: str | Path,
    padding: float = 0.05,
    min_size: int = 20,
    limit_images: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    preview_dir: str | Path = ROOT / "outputs" / "crop_preview",
    preview_count: int = 6,
) -> dict[str, Any]:
    """Prepare all requested COCO splits and return a report; dry runs write nothing."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if min_size < 1:
        raise ValueError("min_size must be at least one pixel")
    if limit_images is not None and limit_images < 1:
        raise ValueError("limit_images must be positive when supplied")
    if preview_count < 0:
        raise ValueError("preview_count must be non-negative")
    dataset_root = validate_dataset_root(dataset_path)
    target_root = Path(output_dir).resolve()
    coco_splits = {split: load_coco_split(path) for split, path in _source_annotation_paths(dataset_root).items()}
    category_mapping, skipped_categories = build_category_mapping(coco_splits)
    if not category_mapping:
        raise ValueError("No trainable categories remain after excluding Stock and category name '0'.")
    fit_class_directories_for_windows(category_mapping, coco_splits, target_root)
    provenance_indexes = load_source_provenance(dataset_root)
    all_crop_rows: list[dict[str, Any]] = []
    preview_items: list[tuple[str, int, int, Image.Image]] = []
    split_reports: dict[str, Any] = {}
    generated_filenames_in_run: dict[str, str] = {}
    generated_paths_in_run: dict[Path, str] = {}
    for split in SOURCE_SPLITS:
        remaining = max(0, preview_count - len(preview_items))
        split_report, rows, split_preview_items = _prepare_split(
            split=split,
            coco=coco_splits[split],
            source_split_dir=dataset_root / SOURCE_SPLITS[split],
            output_dir=target_root,
            category_mapping=category_mapping,
            padding=padding,
            min_size=min_size,
            limit_images=limit_images,
            dry_run=dry_run,
            overwrite=overwrite,
            preview_count_remaining=remaining,
            provenance_indexes=provenance_indexes,
            generated_filenames_in_run=generated_filenames_in_run,
            generated_paths_in_run=generated_paths_in_run,
        )
        split_reports[split] = split_report
        all_crop_rows.extend(rows)
        preview_items.extend(split_preview_items)
    class_mapping = {
        "source_dataset_path": str(dataset_root),
        "output_directory": str(target_root),
        "split_mapping": {"train": "train", "valid": "val", "test": "test"},
        "categories": [category_mapping[category_id] for category_id in sorted(category_mapping)],
        "skipped_categories": skipped_categories,
        "class_to_idx": {item["category_name"]: item["class_index"] for item in category_mapping.values()},
    }
    report = {
        "dry_run": dry_run,
        "dataset_root": str(dataset_root),
        "output_directory": str(target_root),
        "options": {"padding": padding, "min_size": min_size, "limit_images": limit_images, "overwrite": overwrite},
        "category_to_directory_mapping": class_mapping["categories"],
        "skipped_categories": skipped_categories,
        "splits": split_reports,
        "total_successful_crops": sum(split_report["successful_crops"] for split_report in split_reports.values()),
    }
    if dry_run:
        return report
    _write_csv(target_root / "crop_manifest.csv", all_crop_rows, CROP_MANIFEST_FIELDS)
    save_json(all_crop_rows, target_root / "crop_manifest.json")
    for split in SOURCE_SPLITS:
        training_rows = [
            {"image_path": row["image_path"], "class_name": row["class_name"], "split": split}
            for row in all_crop_rows
            if row["split"] == split
        ]
        _write_csv(target_root / "manifests" / f"{split}.csv", training_rows, TRAINING_MANIFEST_FIELDS)
    preview_written = _write_previews(preview_items, Path(preview_dir).resolve(), overwrite)
    report["preview_directory"] = str(Path(preview_dir).resolve())
    report["preview_images_written"] = preview_written
    save_json(class_mapping, target_root / "class_mapping.json")
    save_json(report, target_root / "crop_report.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "data" / "product_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "product_crops")
    parser.add_argument("--padding", type=float, default=0.05, help="Proportional padding added to each bbox side.")
    parser.add_argument("--min-size", type=int, default=20, help="Minimum crop width and height in pixels.")
    parser.add_argument("--limit-images", type=int, default=None, help="Maximum source images to process per split.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report only; do not create any file or directory.")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing an existing generated crop or preview.")
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "outputs" / "crop_preview")
    parser.add_argument("--preview-count", type=int, default=6, help="Maximum total preview pairs across all splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_coco_crops(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        padding=args.padding,
        min_size=args.min_size,
        limit_images=args.limit_images,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        preview_dir=args.preview_dir,
        preview_count=args.preview_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete: no crop, manifest, report, or preview file was written.")
    else:
        print(f"Created/verified product crops under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
