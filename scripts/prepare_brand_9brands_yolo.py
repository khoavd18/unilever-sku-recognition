"""Prepare a conservative configured multi-brand YOLO dataset from source ZIPs.

Exact-image duplicates are reconciled only when every archive occurrence has
the same brand counts and a one-to-one, same-brand geometric match after
OBB-to-AABB conversion. Unresolved images and their connected filename-family
components are quarantined; snapshot order and box union are never used.

This script does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from zipfile import ZipFile, ZipInfo

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_clean_axe_dataset import normalize_source_family
from scripts.prepare_axe_yolo import load_model2_family_assignments
from src.config.registry import load_brand_registry
from src.data.annotations import AnnotationFormat, normalized_box_from_values
from src.data.sources import load_dataset_sources
from src.data.yolo_archive import (
    label_member_for,
    normalize_yaml_names,
    read_info,
    sha256_bytes,
    sha256_file,
    write_csv,
    write_json,
)


_BRAND_REGISTRY = load_brand_registry()
_BRAND_PROFILE = "brand_9brands"
TARGET_BRANDS = _BRAND_REGISTRY.names(_BRAND_PROFILE)
OUTPUT_CLASS_IDS = {brand: index for index, brand in enumerate(TARGET_BRANDS)}
SOURCE_SPLITS = ("train", "valid", "test", "export")
OUTPUT_SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EDGE_TOLERANCE = 1e-6
IOU_TOLERANCE = 0.999
PREVIEW_COLORS = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
)


@dataclass(frozen=True)
class Box:
    brand: str
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    source_class_id: int
    source_class_name: str
    source_line: int
    raw_x_min: float
    raw_y_min: float
    raw_x_max: float
    raw_y_max: float
    was_clipped: bool

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def yolo_row(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )

    def signature(self, digits: int = 12) -> tuple[Any, ...]:
        return (
            self.class_id,
            round(self.x_min, digits),
            round(self.y_min, digits),
            round(self.x_max, digits),
            round(self.y_max, digits),
        )


@dataclass
class Occurrence:
    archive_path: Path
    archive_name: str
    source_split: str
    image_index: int
    image_member: str
    label_indices: tuple[int, ...]
    label_member: str
    source_filename: str
    family_id: str
    image_sha256: str
    width: int
    height: int
    boxes: list[Box]
    source_annotation_count: int
    non_target_annotation_count: int
    invalid_target_rows: list[dict[str, Any]]

    def exact_signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(box.signature() for box in self.boxes))


@dataclass
class ImageUnit:
    image_sha256: str
    canonical: Occurrence
    sources: list[Occurrence]
    family_ids: set[str]
    boxes: list[Box]
    resolution: str
    component_id: str = ""
    forced_split: str | None = None
    force_reasons: set[str] = field(default_factory=set)
    output_split: str = ""
    output_name: str = ""


@dataclass
class HashAudit:
    image_sha256: str
    sources: list[Occurrence]
    status: str
    reason: str
    candidate_sets: list[tuple[Box, ...]]
    family_ids: set[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--zip-dir", default="data/brand_zips")
    parser.add_argument(
        "--brand-profile",
        default="brand_9brands",
        help="Target class-order profile from configs/brands.yaml.",
    )
    parser.add_argument("--inventory-dir", default="outputs/brand9_inventory")
    parser.add_argument("--five-brand-dir", default="data/brand_5brands_yolo")
    parser.add_argument("--output-dir", default="data/brand_9brands_yolo")
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=45)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def configure_brand_profile(
    profile_name: str,
    *,
    registry_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Select a registry profile before audit/preparation begins.

    Extended profiles must append classes after the historical first five so
    Model 1/Model 2 holdout alignment remains meaningful.
    """

    global _BRAND_REGISTRY, _BRAND_PROFILE, TARGET_BRANDS, OUTPUT_CLASS_IDS
    registry = (
        load_brand_registry(registry_path)
        if registry_path is not None
        else load_brand_registry()
    )
    names = registry.names(profile_name)
    historical = registry.names("brand_5brands")
    if names[: len(historical)] != historical:
        raise ValueError(
            "Detector preparation profile must preserve the historical five-brand prefix"
        )
    _BRAND_REGISTRY = registry
    _BRAND_PROFILE = profile_name
    TARGET_BRANDS = names
    OUTPUT_CLASS_IDS = {brand: index for index, brand in enumerate(names)}
    return TARGET_BRANDS


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def infer_target_brand(class_name: str) -> str | None:
    return _BRAND_REGISTRY.infer_brand(class_name, _BRAND_PROFILE)


def box_iou(first: Box, second: Box) -> float:
    intersection_width = max(0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min))
    intersection_height = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    intersection = intersection_width * intersection_height
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def boxes_geometrically_equivalent(first: Box, second: Box) -> bool:
    if first.class_id != second.class_id:
        return False
    edge_delta = max(
        abs(first.x_min - second.x_min),
        abs(first.y_min - second.y_min),
        abs(first.x_max - second.x_max),
        abs(first.y_max - second.y_max),
    )
    return edge_delta <= EDGE_TOLERANCE and box_iou(first, second) >= IOU_TOLERANCE


def _perfect_match(first: Sequence[Box], second: Sequence[Box]) -> bool:
    if len(first) != len(second):
        return False
    edges = [
        [index for index, candidate in enumerate(second) if boxes_geometrically_equivalent(box, candidate)]
        for box in first
    ]
    matched_to = [-1] * len(second)

    def augment(left: int, seen: set[int]) -> bool:
        for right in edges[left]:
            if right in seen:
                continue
            seen.add(right)
            if matched_to[right] == -1 or augment(matched_to[right], seen):
                matched_to[right] = left
                return True
        return False

    return all(augment(index, set()) for index in range(len(first)))


def annotation_sets_equivalent(first: Sequence[Box], second: Sequence[Box]) -> bool:
    first_counts = Counter(box.class_id for box in first)
    second_counts = Counter(box.class_id for box in second)
    if first_counts != second_counts:
        return False
    for class_id in sorted(first_counts):
        left = [box for box in first if box.class_id == class_id]
        right = [box for box in second if box.class_id == class_id]
        if not _perfect_match(left, right):
            return False
    return True


def coordinates_to_box(
    values: Sequence[float],
    brand: str,
    source_class_id: int,
    source_class_name: str,
    source_line: int,
) -> Box:
    parsed = normalized_box_from_values(values, AnnotationFormat.YOLO_OBB)
    xs = values[0::2]
    ys = values[1::2]
    raw_x_min, raw_x_max = min(xs), max(xs)
    raw_y_min, raw_y_max = min(ys), max(ys)
    return Box(
        brand=brand,
        class_id=OUTPUT_CLASS_IDS[brand],
        x_min=parsed.x_min,
        y_min=parsed.y_min,
        x_max=parsed.x_max,
        y_max=parsed.y_max,
        source_class_id=source_class_id,
        source_class_name=source_class_name,
        source_line=source_line,
        raw_x_min=raw_x_min,
        raw_y_min=raw_y_min,
        raw_x_max=raw_x_max,
        raw_y_max=raw_y_max,
        was_clipped=parsed.was_clipped,
    )


def parse_label(
    text: str,
    class_names: Sequence[str],
    archive_name: str,
    label_member: str,
) -> tuple[list[Box], int, list[dict[str, Any]], list[dict[str, Any]]]:
    boxes: list[Box] = []
    non_target = 0
    invalid_target: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.split()
        if not fields:
            continue
        try:
            class_float = float(fields[0])
            class_id = int(class_float)
            if class_float != class_id or not 0 <= class_id < len(class_names):
                raise ValueError("invalid or out-of-range class id")
        except (ValueError, IndexError) as error:
            invalid_target.append(
                {
                    "archive": archive_name,
                    "label_member": label_member,
                    "line_number": line_number,
                    "reason": str(error),
                    "raw_line": raw_line,
                }
            )
            continue
        class_name = class_names[class_id]
        brand = infer_target_brand(class_name)
        if brand is None:
            non_target += 1
            continue
        try:
            values = [float(value) for value in fields[1:]]
            box = coordinates_to_box(values, brand, class_id, class_name, line_number)
            boxes.append(box)
            if box.was_clipped:
                corrections.append(
                    {
                        "archive": archive_name,
                        "label_member": label_member,
                        "line_number": line_number,
                        "brand": brand,
                        "source_class_id": class_id,
                        "source_class_name": class_name,
                        "raw_x_min": box.raw_x_min,
                        "raw_y_min": box.raw_y_min,
                        "raw_x_max": box.raw_x_max,
                        "raw_y_max": box.raw_y_max,
                        "corrected_x_min": box.x_min,
                        "corrected_y_min": box.y_min,
                        "corrected_x_max": box.x_max,
                        "corrected_y_max": box.y_max,
                    }
                )
        except ValueError as error:
            invalid_target.append(
                {
                    "archive": archive_name,
                    "label_member": label_member,
                    "line_number": line_number,
                    "brand": brand,
                    "source_class_id": class_id,
                    "source_class_name": class_name,
                    "reason": str(error),
                    "raw_line": raw_line,
                }
            )
    return boxes, non_target, invalid_target, corrections


def load_inventory(inventory_dir: Path) -> dict[str, Any]:
    required = (
        "summary_vi.md",
        "counts_by_zip_split_brand.csv",
        "class_mapping_candidates.csv",
        "duplicate_and_conflict_summary.csv",
    )
    missing = [name for name in required if not (inventory_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Brand-9 inventory is incomplete: {missing}")
    with (inventory_dir / "duplicate_and_conflict_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        duplicate_rows = list(csv.DictReader(handle))
    overall = {
        row["metric"]: int(row["value"])
        for row in duplicate_rows
        if row["record_type"] == "overall_metric"
    }
    with (inventory_dir / "class_mapping_candidates.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        mapping_rows = list(csv.DictReader(handle))
    return {
        "path": str(inventory_dir),
        "files": {
            name: {"sha256": sha256_file(inventory_dir / name), "size": (inventory_dir / name).stat().st_size}
            for name in required
        },
        "expected_raw_images": overall.get("raw_image_occurrences"),
        "expected_unique_images": overall.get("unique_image_content_sha256"),
        "mapping_rows": mapping_rows,
    }


def _inventory_mapping_check(
    inventory: dict[str, Any], archive_mappings: dict[str, list[str]]
) -> dict[str, Any]:
    added_brands = set(TARGET_BRANDS[5:])
    expected: set[tuple[str, int, str, str]] = set()
    for row in inventory["mapping_rows"]:
        if row["status"] == "ambiguous_unassigned":
            continue
        if row["proposed_brand"] not in added_brands:
            continue
        expected.add(
            (
                row["zip"],
                int(row["class_id"]),
                row["source_class_name"],
                row["proposed_brand"],
            )
        )
    actual: set[tuple[str, int, str, str]] = set()
    for archive_name, names in archive_mappings.items():
        for class_id, class_name in enumerate(names):
            brand = infer_target_brand(class_name)
            if brand in added_brands:
                actual.add((archive_name, class_id, class_name, brand))
    if actual != expected:
        raise ValueError(
            "Current ZIP mappings disagree with the reviewed inventory for added brands: "
            f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
        )
    return {
        "inventory_candidate_rows_verified": len(expected),
        "verified_added_brands": list(TARGET_BRANDS[5:]),
    }


def audit_archives(
    zip_dir: Path, inventory: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[Occurrence], list[dict[str, Any]], list[dict[str, Any]]]:
    zip_paths = sorted(zip_dir.glob("*.zip"), key=lambda path: path.name.casefold())
    if not zip_paths:
        raise FileNotFoundError(f"No source ZIPs found in {zip_dir}")
    reports: list[dict[str, Any]] = []
    records: list[Occurrence] = []
    corrections: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    mappings: dict[str, list[str]] = {}
    total_zip_images = 0
    started = time.time()

    for zip_path in zip_paths:
        print(f"[audit] {zip_path.name}: reading ZipInfo entries", flush=True)
        configured_source = load_dataset_sources().match(zip_path)
        if configured_source.annotation_format is not AnnotationFormat.YOLO_OBB:
            raise ValueError(
                f"Detector source {configured_source.name!r} must declare yolo_obb, "
                f"found {configured_source.annotation_format.value}"
            )
        archive_sha256 = sha256_file(zip_path)
        if (
            configured_source.sha256
            and configured_source.sha256.casefold() != archive_sha256.casefold()
        ):
            raise ValueError(
                f"Source checksum differs for {configured_source.name}: expected "
                f"{configured_source.sha256}, found {archive_sha256}"
            )
        with ZipFile(zip_path) as archive:
            infos = archive.infolist()
            indexed = list(enumerate(infos))
            by_name: dict[str, list[tuple[int, ZipInfo]]] = defaultdict(list)
            for index, info in indexed:
                by_name[info.filename].append((index, info))
            yaml_entries: list[dict[str, Any]] = []
            yaml_signatures: set[str] = set()
            yaml_payload_hashes: set[str] = set()
            parsed_names: list[list[str]] = []
            for index, info in indexed:
                if PurePosixPath(info.filename).name.lower() not in {"data.yaml", "data.yml"}:
                    continue
                payload = read_info(archive, info)
                parsed = yaml.safe_load(payload.decode("utf-8-sig")) or {}
                names = normalize_yaml_names(parsed.get("names"))
                signature = json.dumps(
                    {
                        "names": names,
                        "train": parsed.get("train"),
                        "val": parsed.get("val"),
                        "valid": parsed.get("valid"),
                        "test": parsed.get("test"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                yaml_signatures.add(signature)
                yaml_payload_hashes.add(sha256_bytes(payload))
                parsed_names.append(names)
                yaml_entries.append(
                    {
                        "member_index": index,
                        "member_path": info.filename,
                        "sha256": sha256_bytes(payload),
                        "crc32": f"{info.CRC:08x}",
                    }
                )
            if not yaml_entries:
                raise ValueError(f"{zip_path.name} has no data.yaml")
            if len(yaml_signatures) != 1:
                raise ValueError(f"{zip_path.name} has conflicting duplicate data.yaml mappings")
            class_names = parsed_names[0]
            mappings[zip_path.name] = class_names

            image_infos: list[tuple[int, ZipInfo]] = []
            label_infos: list[tuple[int, ZipInfo]] = []
            split_inventory: Counter[tuple[str, str]] = Counter()
            for index, info in indexed:
                path = PurePosixPath(info.filename)
                split = path.parts[0].lower() if path.parts else "other"
                if path.suffix.lower() in IMAGE_SUFFIXES and split in SOURCE_SPLITS:
                    image_infos.append((index, info))
                    split_inventory[(split, "images")] += 1
                elif (
                    path.suffix.lower() == ".txt"
                    and "labels" in [part.lower() for part in path.parts]
                    and split in SOURCE_SPLITS
                ):
                    label_infos.append((index, info))
                    split_inventory[(split, "labels")] += 1
            total_zip_images += len(image_infos)
            label_names = {info.filename for _, info in label_infos}
            paired_label_names: set[str] = set()
            archive_records_start = len(records)
            archive_corrections_start = len(corrections)
            archive_rejected_start = len(rejected)

            for local_index, (image_index, image_info) in enumerate(image_infos, start=1):
                payload = read_info(archive, image_info)
                image_hash = sha256_bytes(payload)
                with Image.open(io.BytesIO(payload)) as image:
                    width, height = image.size
                    image.verify()
                source_split = PurePosixPath(image_info.filename).parts[0].lower()
                label_member = label_member_for(image_info.filename)
                label_occurrences = by_name.get(label_member, [])
                invalid_rows: list[dict[str, Any]] = []
                boxes: list[Box] = []
                non_target = 0
                label_indices: tuple[int, ...] = ()
                if not label_occurrences:
                    invalid_rows.append(
                        {
                            "archive": zip_path.name,
                            "image_member": image_info.filename,
                            "label_member": label_member,
                            "line_number": "",
                            "reason": "missing label for image",
                            "raw_line": "",
                        }
                    )
                else:
                    paired_label_names.add(label_member)
                    label_indices = tuple(index for index, _ in label_occurrences)
                    label_payloads = [read_info(archive, info) for _, info in label_occurrences]
                    if len({sha256_bytes(value) for value in label_payloads}) != 1:
                        invalid_rows.append(
                            {
                                "archive": zip_path.name,
                                "image_member": image_info.filename,
                                "label_member": label_member,
                                "line_number": "",
                                "reason": "duplicate label members disagree",
                                "raw_line": "",
                            }
                        )
                    else:
                        boxes, non_target, parsed_invalid, parsed_corrections = parse_label(
                            label_payloads[0].decode("utf-8-sig", errors="replace"),
                            class_names,
                            zip_path.name,
                            label_member,
                        )
                        invalid_rows.extend(parsed_invalid)
                        for row in parsed_corrections:
                            row.update(
                                {
                                    "image_sha256": image_hash,
                                    "image_member": image_info.filename,
                                    "source_split": source_split,
                                }
                            )
                        corrections.extend(parsed_corrections)
                for row in invalid_rows:
                    row.setdefault("image_sha256", image_hash)
                    row.setdefault("image_member", image_info.filename)
                    row.setdefault("source_split", source_split)
                rejected.extend(invalid_rows)
                records.append(
                    Occurrence(
                        archive_path=zip_path,
                        archive_name=zip_path.name,
                        source_split=source_split,
                        image_index=image_index,
                        image_member=image_info.filename,
                        label_indices=label_indices,
                        label_member=label_member,
                        source_filename=PurePosixPath(image_info.filename).name,
                        family_id=normalize_source_family(image_info.filename),
                        image_sha256=image_hash,
                        width=width,
                        height=height,
                        boxes=boxes,
                        source_annotation_count=len(boxes) + non_target,
                        non_target_annotation_count=non_target,
                        invalid_target_rows=invalid_rows,
                    )
                )
                completed = len(records)
                if completed % 250 == 0:
                    elapsed = max(time.time() - started, 0.001)
                    expected = inventory.get("expected_raw_images") or total_zip_images
                    print(
                        f"[hash] {completed}/{expected} ({100 * completed / expected:.1f}%) "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            orphan_labels = sorted(label_names - paired_label_names)
            if orphan_labels:
                raise ValueError(
                    f"{zip_path.name} has {len(orphan_labels)} labels without images: {orphan_labels[:5]}"
                )
            reports.append(
                {
                    "archive": str(zip_path),
                    "archive_name": zip_path.name,
                    "archive_size": zip_path.stat().st_size,
                    "archive_sha256": archive_sha256,
                    "annotation_format": configured_source.annotation_format.value,
                    "zipinfo_entry_count": len(infos),
                    "yaml_entries": yaml_entries,
                    "yaml_payloads_byte_identical": len(yaml_payload_hashes) == 1,
                    "yaml_mapping_consistent": True,
                    "effective_class_count": len(class_names),
                    "image_occurrences": len(image_infos),
                    "source_records": len(records) - archive_records_start,
                    "out_of_bounds_corrections": len(corrections) - archive_corrections_start,
                    "rejected_rows": len(rejected) - archive_rejected_start,
                    "split_inventory": {
                        split: {
                            "images": split_inventory[(split, "images")],
                            "labels": split_inventory[(split, "labels")],
                        }
                        for split in SOURCE_SPLITS
                    },
                }
            )
        print(f"[audit] {zip_path.name}: {len(image_infos)} images complete", flush=True)

    if inventory.get("expected_raw_images") != len(records):
        raise ValueError(
            f"ZIP raw image count {len(records)} differs from inventory "
            f"{inventory.get('expected_raw_images')}"
        )
    mapping_check = _inventory_mapping_check(inventory, mappings)
    unique_hashes = len({record.image_sha256 for record in records})
    if inventory.get("expected_unique_images") != unique_hashes:
        raise ValueError(
            f"ZIP unique image count {unique_hashes} differs from inventory "
            f"{inventory.get('expected_unique_images')}"
        )
    inventory["cross_check"] = {
        **mapping_check,
        "raw_image_occurrences_verified": len(records),
        "unique_image_sha256_verified": unique_hashes,
    }
    print(f"[hash] complete: {len(records)} occurrences, {unique_hashes} unique", flush=True)
    return reports, records, corrections, rejected


def load_decisions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Review decisions file does not exist: {path}")
    decisions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_hash = row.get("image_sha256", "").strip()
            candidate_id = row.get("selected_candidate_id", "").strip()
            if not image_hash or not candidate_id:
                continue
            previous = decisions.setdefault(image_hash, candidate_id)
            if previous != candidate_id:
                raise ValueError(f"Conflicting review decisions for {image_hash}")
    return decisions


def reconcile_exact_duplicates(
    records: Sequence[Occurrence], decisions: dict[str, str]
) -> tuple[
    list[ImageUnit],
    list[HashAudit],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    by_hash: dict[str, list[Occurrence]] = defaultdict(list)
    for record in records:
        by_hash[record.image_sha256].append(record)
    units: list[ImageUnit] = []
    audits: list[HashAudit] = []
    conflict_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    used_decisions = 0

    for image_hash, sources in sorted(by_hash.items()):
        ordered_sources = sorted(
            sources,
            key=lambda source: (
                source.archive_name.casefold(),
                source.image_member.casefold(),
                source.image_index,
            ),
        )
        candidate_by_signature: dict[tuple[tuple[Any, ...], ...], tuple[Box, ...]] = {}
        source_signature: dict[tuple[str, int, str], tuple[tuple[Any, ...], ...]] = {}
        for source in ordered_sources:
            signature = source.exact_signature()
            candidate_by_signature.setdefault(signature, tuple(sorted(source.boxes, key=lambda box: box.signature())))
            source_signature[(source.archive_name, source.image_index, source.image_member)] = signature
        candidate_signatures = sorted(candidate_by_signature)
        candidate_sets = [candidate_by_signature[signature] for signature in candidate_signatures]
        family_ids = {source.family_id for source in ordered_sources}
        invalid = [row for source in ordered_sources for row in source.invalid_target_rows]
        nonempty = [candidate for candidate in candidate_sets if candidate]
        status = ""
        reason = ""
        selected_boxes: list[Box] | None = None
        resolution = ""

        if invalid:
            status = "unresolved"
            reason = "one or more source occurrences contain invalid/rejected target annotation rows"
        elif not nonempty:
            status = "non_target"
            reason = "no configured target-brand annotation in any occurrence"
        else:
            reference = ordered_sources[0].boxes
            equivalent = all(
                annotation_sets_equivalent(reference, source.boxes)
                for source in ordered_sources[1:]
            )
            if equivalent:
                status = "accepted_equivalent"
                selected_signature = min(candidate_signatures)
                selected_boxes = list(candidate_by_signature[selected_signature])
                resolution = (
                    "all occurrences have identical brand counts and a one-to-one same-brand match "
                    f"with max edge delta <= {EDGE_TOLERANCE} and IoU >= {IOU_TOLERANCE}; "
                    "lexicographically smallest equivalent geometry signature selected"
                )
            else:
                status = "unresolved"
                reason = (
                    "duplicate occurrences fail same-brand count or one-to-one geometric agreement; "
                    "no snapshot precedence and no box union applied"
                )

        if status == "unresolved" and image_hash in decisions:
            selected_id = decisions[image_hash]
            match = re.fullmatch(r"candidate_(\d{2})", selected_id)
            if not match:
                raise ValueError(f"Invalid candidate id in review decision for {image_hash}: {selected_id}")
            candidate_index = int(match.group(1))
            if not 0 <= candidate_index < len(candidate_sets) or not candidate_sets[candidate_index]:
                raise ValueError(
                    f"Review decision {selected_id} is unavailable/empty for {image_hash}"
                )
            status = "accepted_manual"
            selected_boxes = list(candidate_sets[candidate_index])
            resolution = f"manual review decision selected {selected_id}"
            reason = ""
            used_decisions += 1

        audit = HashAudit(
            image_sha256=image_hash,
            sources=ordered_sources,
            status=status,
            reason=reason,
            candidate_sets=candidate_sets,
            family_ids=family_ids,
        )
        audits.append(audit)
        status_counts[status] += 1

        for candidate_index, candidate_set in enumerate(candidate_sets):
            candidate_id = f"candidate_{candidate_index:02d}"
            matching_sources = [
                {
                    "archive": source.archive_name,
                    "source_split": source.source_split,
                    "image_member": source.image_member,
                    "label_member": source.label_member,
                }
                for source in ordered_sources
                if source_signature[(source.archive_name, source.image_index, source.image_member)]
                == candidate_signatures[candidate_index]
            ]
            candidate_rows.append(
                {
                    "image_sha256": image_hash,
                    "candidate_id": candidate_id,
                    "box_count": len(candidate_set),
                    "brand_counts": json.dumps(
                        dict(sorted(Counter(box.brand for box in candidate_set).items())),
                        ensure_ascii=False,
                    ),
                    "yolo_rows": json.dumps([box.yolo_row() for box in candidate_set]),
                    "source_occurrences": json.dumps(matching_sources, ensure_ascii=False),
                }
            )

        if status == "unresolved":
            conflict_rows.append(
                {
                    "image_sha256": image_hash,
                    "reason": reason,
                    "source_occurrence_count": len(ordered_sources),
                    "candidate_set_count": len(candidate_sets),
                    "candidate_box_counts": json.dumps([len(candidate) for candidate in candidate_sets]),
                    "candidate_brand_counts": json.dumps(
                        [
                            dict(sorted(Counter(box.brand for box in candidate).items()))
                            for candidate in candidate_sets
                        ],
                        ensure_ascii=False,
                    ),
                    "family_ids": json.dumps(sorted(family_ids), ensure_ascii=False),
                    "sources": json.dumps(
                        [
                            {
                                "archive": source.archive_name,
                                "source_split": source.source_split,
                                "image_member": source.image_member,
                                "label_member": source.label_member,
                                "box_count": len(source.boxes),
                                "invalid_rows": len(source.invalid_target_rows),
                            }
                            for source in ordered_sources
                        ],
                        ensure_ascii=False,
                    ),
                }
            )

        if selected_boxes:
            canonical = min(
                ordered_sources,
                key=lambda source: (
                    source.image_member.casefold(),
                    source.archive_name.casefold(),
                    source.image_index,
                ),
            )
            units.append(
                ImageUnit(
                    image_sha256=image_hash,
                    canonical=canonical,
                    sources=ordered_sources,
                    family_ids=family_ids,
                    boxes=selected_boxes,
                    resolution=resolution,
                )
            )

    unused_decisions = sorted(set(decisions) - {audit.image_sha256 for audit in audits})
    if unused_decisions:
        raise ValueError(f"Review decisions refer to unknown image hashes: {unused_decisions[:10]}")
    return units, audits, conflict_rows, candidate_rows, {
        "raw_image_occurrences": len(records),
        "unique_image_hashes": len(by_hash),
        "duplicate_occurrences_removed": len(records) - len(by_hash),
        "status_counts": dict(sorted(status_counts.items())),
        "equivalence_rule": {
            "class_agreement": "identical per-brand instance counts and same output class for each match",
            "geometry_agreement": (
                f"perfect one-to-one matching with normalized AABB max edge delta <= "
                f"{EDGE_TOLERANCE} and IoU >= {IOU_TOLERANCE}"
            ),
            "conversion_before_comparison": "OBB min/max AABB conversion followed by [0,1] clipping",
            "prohibited_rules": [
                "snapshot order",
                "generic-brand ratio",
                "blind union of boxes",
            ],
        },
        "manual_decisions_loaded": len(decisions),
        "manual_decisions_used": used_decisions,
    }


def load_five_brand_holdouts(
    dataset_dir: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    required = (
        dataset_dir / "mapping.json",
        dataset_dir / "dataset_checks.json",
        dataset_dir / "provenance.csv",
        dataset_dir / "source_family_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Five-brand provenance is incomplete: {missing}")
    mapping = json.loads((dataset_dir / "mapping.json").read_text(encoding="utf-8"))
    checks = json.loads((dataset_dir / "dataset_checks.json").read_text(encoding="utf-8"))
    expected_mapping = {str(index): brand for index, brand in enumerate(TARGET_BRANDS[:5])}
    if mapping.get("output_names") != expected_mapping:
        raise ValueError("Historical five-brand mapping is incompatible")
    if (
        checks.get("status") != "passed"
        or checks.get("model2_holdout_violations") != 0
        or checks.get("cross_split_family_overlap") != 0
    ):
        raise ValueError(f"Historical five-brand checks are not trustworthy: {checks}")

    hash_locks: dict[str, set[str]] = defaultdict(set)
    family_locks: dict[str, set[str]] = defaultdict(set)
    verified_rows = 0
    locked_rows = Counter()
    with (dataset_dir / "provenance.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = dataset_dir / Path(row["output_image_path"].replace("/", "\\"))
        if not image_path.is_file():
            raise FileNotFoundError(f"Historical five-brand image is missing: {image_path}")
        actual_hash = sha256_file(image_path)
        if actual_hash != row["image_sha256"]:
            raise ValueError(f"Historical five-brand provenance hash mismatch: {image_path}")
        verified_rows += 1
        split = row["output_split"]
        if split not in {"val", "test"}:
            continue
        hash_locks[row["image_sha256"]].add(split)
        locked_rows[split] += 1
        for family in json.loads(row["family_ids"]):
            family_locks[str(family).casefold()].add(split)
    return hash_locks, family_locks, {
        "path": str(dataset_dir),
        "mapping_verified": True,
        "checks_verified": True,
        "provenance_rows_hash_verified": verified_rows,
        "locked_holdout_rows": dict(sorted(locked_rows.items())),
        "locked_hashes": len(hash_locks),
        "locked_families": len(family_locks),
    }


def build_family_components(
    units: Sequence[ImageUnit],
    audits: Sequence[HashAudit],
    five_hash_locks: dict[str, set[str]],
    five_family_locks: dict[str, set[str]],
    model2_assignments: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    relevant = [audit for audit in audits if audit.status != "non_target"]
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            lower, upper = sorted((first_root, second_root))
            parent[upper] = lower

    for audit in relevant:
        families = sorted(audit.family_ids)
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)
    component_audits: dict[str, list[HashAudit]] = defaultdict(list)
    for audit in relevant:
        component = min(find(family) for family in audit.family_ids)
        component_audits[component].append(audit)
    unit_by_hash = {unit.image_sha256: unit for unit in units}
    clean_groups: list[dict[str, Any]] = []
    quarantined_groups: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for component, members in sorted(component_audits.items()):
        family_ids = sorted({family for member in members for family in member.family_ids})
        member_units = [unit_by_hash[member.image_sha256] for member in members if member.image_sha256 in unit_by_hash]
        unresolved_hashes = sorted(
            member.image_sha256 for member in members if member.status == "unresolved"
        )
        holdout_sources: dict[str, set[str]] = defaultdict(set)
        for member in members:
            for split in five_hash_locks.get(member.image_sha256, set()):
                holdout_sources[split].add(f"five-brand hash:{member.image_sha256}")
        for family in family_ids:
            for split in five_family_locks.get(family, set()):
                holdout_sources[split].add(f"five-brand family:{family}")
            model2_split = model2_assignments.get(family)
            if model2_split in {"val", "test"}:
                holdout_sources[model2_split].add(f"Model 2 family:{family}")
        holdout_splits = set(holdout_sources)
        quarantine_reasons: list[str] = []
        if unresolved_hashes:
            quarantine_reasons.append("connected family contains unresolved exact-image annotations")
            reason_counts["annotation_conflict_family"] += 1
        if len(holdout_splits) > 1:
            quarantine_reasons.append("family component links conflicting locked val/test holdouts")
            reason_counts["holdout_conflict_family"] += 1
        group = {
            "group_id": component,
            "audits": sorted(members, key=lambda item: item.image_sha256),
            "units": sorted(member_units, key=lambda item: item.image_sha256),
            "family_ids": family_ids,
            "image_count": len(member_units),
            "brand_instance_counts": dict(
                Counter(box.brand for unit in member_units for box in unit.boxes)
            ),
            "brand_image_counts": {
                brand: sum(any(box.brand == brand for box in unit.boxes) for unit in member_units)
                for brand in TARGET_BRANDS
            },
            "forced_split": next(iter(holdout_splits)) if len(holdout_splits) == 1 else None,
            "force_reasons": sorted(
                reason for reasons in holdout_sources.values() for reason in reasons
            ),
            "unresolved_hashes": unresolved_hashes,
            "quarantine_reasons": quarantine_reasons,
        }
        if quarantine_reasons:
            quarantined_groups.append(group)
        elif member_units:
            for unit in member_units:
                unit.component_id = component
                unit.forced_split = group["forced_split"]
                unit.force_reasons.update(group["force_reasons"])
            clean_groups.append(group)

    return clean_groups, quarantined_groups, {
        "relevant_family_components": len(component_audits),
        "clean_family_components": len(clean_groups),
        "quarantined_family_components": len(quarantined_groups),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "quarantine_policy": (
            "quarantine the entire connected filename-family component when any exact-image "
            "annotation is unresolved or when locked val/test assignments conflict"
        ),
    }


def _split_objective(
    image_counts: Counter[str],
    instance_counts: dict[str, Counter[str]],
    brand_image_counts: dict[str, Counter[str]],
    targets: dict[str, Any],
) -> float:
    total_images = max(targets["total_images"], 1)
    value = sum(
        ((image_counts[split] - targets["images"][split]) / total_images) ** 2
        for split in OUTPUT_SPLITS
    )
    for split in OUTPUT_SPLITS:
        for brand in TARGET_BRANDS:
            total_instances = max(targets["total_instances"][brand], 1)
            total_brand_images = max(targets["total_brand_images"][brand], 1)
            value += (
                (instance_counts[split][brand] - targets["instances"][split][brand])
                / total_instances
            ) ** 2
            value += (
                (brand_image_counts[split][brand] - targets["brand_images"][split][brand])
                / total_brand_images
            ) ** 2
    return value


def assign_splits(groups: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    total_images = sum(group["image_count"] for group in groups)
    total_instances: Counter[str] = Counter()
    total_brand_images: Counter[str] = Counter()
    for group in groups:
        total_instances.update(group["brand_instance_counts"])
        total_brand_images.update(group["brand_image_counts"])
    targets = {
        "total_images": total_images,
        "total_instances": total_instances,
        "total_brand_images": total_brand_images,
        "images": {split: total_images * SPLIT_RATIOS[split] for split in OUTPUT_SPLITS},
        "instances": {
            split: {
                brand: total_instances[brand] * SPLIT_RATIOS[split]
                for brand in TARGET_BRANDS
            }
            for split in OUTPUT_SPLITS
        },
        "brand_images": {
            split: {
                brand: total_brand_images[brand] * SPLIT_RATIOS[split]
                for brand in TARGET_BRANDS
            }
            for split in OUTPUT_SPLITS
        },
    }
    image_counts: Counter[str] = Counter()
    instance_counts = {split: Counter() for split in OUTPUT_SPLITS}
    brand_image_counts = {split: Counter() for split in OUTPUT_SPLITS}

    def apply(group: dict[str, Any], split: str, direction: int) -> None:
        image_counts[split] += direction * group["image_count"]
        for brand, count in group["brand_instance_counts"].items():
            instance_counts[split][brand] += direction * count
        for brand, count in group["brand_image_counts"].items():
            brand_image_counts[split][brand] += direction * count

    for group in groups:
        if group["forced_split"]:
            group["output_split"] = group["forced_split"]
            apply(group, group["output_split"], 1)

    rng = random.Random(seed)
    tie_breakers = {
        (group["group_id"], split): rng.random()
        for group in groups
        for split in OUTPUT_SPLITS
    }
    unassigned = [group for group in groups if not group["forced_split"]]
    unassigned.sort(
        key=lambda group: (
            -sum(group["brand_instance_counts"].values()),
            -group["image_count"],
            group["group_id"],
        )
    )
    for group in unassigned:
        candidates = []
        for split in OUTPUT_SPLITS:
            apply(group, split, 1)
            objective = _split_objective(
                image_counts, instance_counts, brand_image_counts, targets
            )
            apply(group, split, -1)
            candidates.append((objective, tie_breakers[(group["group_id"], split)], split))
        group["output_split"] = min(candidates)[2]
        apply(group, group["output_split"], 1)

    improvement_moves = 0
    passes = 0
    for pass_index in range(50):
        moved = 0
        for group in unassigned:
            old_split = group["output_split"]
            current = _split_objective(
                image_counts, instance_counts, brand_image_counts, targets
            )
            best = (current, old_split)
            apply(group, old_split, -1)
            for split in OUTPUT_SPLITS:
                if split == old_split:
                    continue
                apply(group, split, 1)
                objective = _split_objective(
                    image_counts, instance_counts, brand_image_counts, targets
                )
                apply(group, split, -1)
                if objective + 1e-15 < best[0]:
                    best = (objective, split)
            apply(group, best[1], 1)
            if best[1] != old_split:
                group["output_split"] = best[1]
                improvement_moves += 1
                moved += 1
        passes = pass_index + 1
        if not moved:
            break

    for group in groups:
        for unit in group["units"]:
            unit.output_split = group["output_split"]
    missing = {
        split: [brand for brand in TARGET_BRANDS if brand_image_counts[split][brand] == 0]
        for split in OUTPUT_SPLITS
    }
    if any(missing.values()):
        raise ValueError(f"Split assignment leaves brands absent: {missing}")
    return {
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "actual_image_counts": dict(image_counts),
        "actual_instance_counts": {
            split: {brand: instance_counts[split][brand] for brand in TARGET_BRANDS}
            for split in OUTPUT_SPLITS
        },
        "actual_brand_image_counts": {
            split: {brand: brand_image_counts[split][brand] for brand in TARGET_BRANDS}
            for split in OUTPUT_SPLITS
        },
        "forced_holdout_images": {
            split: sum(
                group["image_count"]
                for group in groups
                if group["forced_split"] == split
            )
            for split in ("val", "test")
        },
        "local_optimization_passes": passes,
        "local_improvement_moves": improvement_moves,
        "objective": _split_objective(
            image_counts, instance_counts, brand_image_counts, targets
        ),
    }


def read_occurrence_image(
    occurrence: Occurrence, archives: dict[Path, ZipFile]
) -> bytes:
    archive = archives[occurrence.archive_path]
    infos = archive.infolist()
    info = infos[occurrence.image_index]
    if info.filename != occurrence.image_member:
        raise ValueError(
            f"ZipInfo index drift for {occurrence.archive_name}: "
            f"{occurrence.image_index} -> {info.filename}, expected {occurrence.image_member}"
        )
    payload = read_info(archive, info)
    if sha256_bytes(payload) != occurrence.image_sha256:
        raise ValueError(f"Source image hash changed: {occurrence.image_member}")
    return payload


def draw_boxes(image: Image.Image, boxes: Sequence[Box], title: str) -> Image.Image:
    canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for box in boxes:
        color = PREVIEW_COLORS[box.class_id % len(PREVIEW_COLORS)]
        coords = (
            int(round(box.x_min * canvas.width)),
            int(round(box.y_min * canvas.height)),
            int(round(box.x_max * canvas.width)),
            int(round(box.y_max * canvas.height)),
        )
        draw.rectangle(coords, outline=color, width=max(2, min(canvas.size) // 300))
        label = box.brand
        text_bbox = draw.textbbox((coords[0], coords[1]), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((coords[0], coords[1]), label, fill="black", font=font)
    banner_height = 24
    output = Image.new("RGB", (canvas.width, canvas.height + banner_height), "white")
    output.paste(canvas, (0, banner_height))
    ImageDraw.Draw(output).text((5, 5), title, fill="black", font=font)
    return output


def make_review_sheet(
    image_payload: bytes,
    audit: HashAudit,
    output_path: Path,
) -> None:
    with Image.open(io.BytesIO(image_payload)) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    panels = [draw_boxes(original.copy(), (), "original")]
    for index, candidate in enumerate(audit.candidate_sets):
        counts = dict(sorted(Counter(box.brand for box in candidate).items()))
        panels.append(
            draw_boxes(
                original.copy(),
                candidate,
                f"candidate_{index:02d}: {json.dumps(counts, ensure_ascii=False)}",
            )
        )
    fitted = [ImageOps.contain(panel, (480, 420), Image.Resampling.LANCZOS) for panel in panels]
    width = sum(panel.width for panel in fitted)
    height = max(panel.height for panel in fitted)
    sheet = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for panel in fitted:
        sheet.paste(panel, (x_offset, 0))
        x_offset += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def make_contact_sheet(preview_paths: Sequence[Path], output_path: Path) -> None:
    if not preview_paths:
        return
    thumbs = []
    for path in preview_paths:
        with Image.open(path) as image:
            thumbs.append(ImageOps.contain(image.convert("RGB"), (320, 240), Image.Resampling.LANCZOS))
    columns = 4
    cell_width, cell_height = 330, 250
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * cell_width + 5
        y = (index // columns) * cell_height + 5
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def assign_output_names(units: Sequence[ImageUnit]) -> None:
    for index, unit in enumerate(sorted(units, key=lambda item: item.image_sha256), start=1):
        suffix = PurePosixPath(unit.canonical.image_member).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        unit.output_name = f"n{index:05d}{suffix}"


def _source_refs(sources: Sequence[Occurrence]) -> str:
    return json.dumps(
        [
            {
                "archive": source.archive_name,
                "source_split": source.source_split,
                "image_index": source.image_index,
                "image_member": source.image_member,
                "label_indices": list(source.label_indices),
                "label_member": source.label_member,
            }
            for source in sources
        ],
        ensure_ascii=False,
    )


def select_preview_units(units: Sequence[ImageUnit], limit: int) -> list[ImageUnit]:
    remaining = sorted(units, key=lambda unit: unit.image_sha256)
    selected: list[ImageUnit] = []
    uncovered = {
        (split, brand) for split in OUTPUT_SPLITS for brand in TARGET_BRANDS
    }
    while remaining and len(selected) < limit and uncovered:
        best = max(
            remaining,
            key=lambda unit: (
                sum(
                    (unit.output_split, brand) in uncovered
                    for brand in {box.brand for box in unit.boxes}
                ),
                len({box.brand for box in unit.boxes}),
                -len(unit.boxes),
                unit.image_sha256,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for brand in {box.brand for box in best.boxes}:
            uncovered.discard((best.output_split, brand))
    multi = [
        unit
        for unit in remaining
        if len({box.brand for box in unit.boxes}) > 1
    ]
    for unit in multi + remaining:
        if len(selected) >= limit:
            break
        if unit not in selected:
            selected.append(unit)
    return selected


def summarize_conflicts_by_brand(
    conflict_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    conflict_hashes = {row["image_sha256"] for row in conflict_rows}
    candidates_by_hash: dict[str, list[list[str]]] = defaultdict(list)
    for row in candidate_rows:
        if row["image_sha256"] in conflict_hashes:
            candidates_by_hash[row["image_sha256"]].append(json.loads(row["yolo_rows"]))
    summary = []
    for class_id, brand in enumerate(TARGET_BRANDS):
        affected = 0
        count_or_membership = 0
        geometry_only = 0
        for image_hash in sorted(conflict_hashes):
            per_candidate = [
                tuple(sorted(line for line in rows if int(line.split()[0]) == class_id))
                for rows in candidates_by_hash[image_hash]
            ]
            if len(set(per_candidate)) <= 1:
                continue
            affected += 1
            if len({len(rows) for rows in per_candidate}) > 1:
                count_or_membership += 1
            else:
                geometry_only += 1
        summary.append(
            {
                "class_id": class_id,
                "brand": brand,
                "unresolved_image_hashes": affected,
                "count_or_membership_conflicts": count_or_membership,
                "geometry_only_conflicts": geometry_only,
            }
        )
    return summary


def write_dataset(
    staging: Path,
    final_output_dir: Path,
    units: Sequence[ImageUnit],
    clean_groups: Sequence[dict[str, Any]],
    quarantined_groups: Sequence[dict[str, Any]],
    audits: Sequence[HashAudit],
    conflict_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    corrections: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
    preview_count: int,
) -> dict[str, Any]:
    assign_output_names(units)
    provenance_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    review_paths: list[Path] = []
    preview_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    family_variation_rows: list[dict[str, Any]] = []
    conflict_brand_rows = summarize_conflicts_by_brand(conflict_rows, candidate_rows)

    for split in OUTPUT_SPLITS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
    (staging / "quarantine" / "images").mkdir(parents=True, exist_ok=True)
    (staging / "quarantine" / "candidate_labels").mkdir(parents=True, exist_ok=True)
    (staging / "review_sheets").mkdir(parents=True, exist_ok=True)
    (staging / "previews").mkdir(parents=True, exist_ok=True)

    archive_paths = sorted({source.archive_path for audit in audits for source in audit.sources})
    with ExitStack() as stack:
        archives = {
            path: stack.enter_context(ZipFile(path, "r"))
            for path in archive_paths
        }
        for unit in sorted(units, key=lambda item: item.image_sha256):
            payload = read_occurrence_image(unit.canonical, archives)
            image_path = staging / "images" / unit.output_split / unit.output_name
            label_path = (
                staging / "labels" / unit.output_split / Path(unit.output_name).with_suffix(".txt")
            )
            image_path.write_bytes(payload)
            label_path.write_text(
                "\n".join(box.yolo_row() for box in sorted(unit.boxes, key=lambda box: box.signature()))
                + "\n",
                encoding="utf-8",
            )
            brand_counts = Counter(box.brand for box in unit.boxes)
            provenance_rows.append(
                {
                    "output_split": unit.output_split,
                    "output_image_path": image_path.relative_to(staging).as_posix(),
                    "output_label_path": label_path.relative_to(staging).as_posix(),
                    "image_sha256": unit.image_sha256,
                    "family_component_id": unit.component_id,
                    "family_ids": json.dumps(sorted(unit.family_ids), ensure_ascii=False),
                    "forced_holdout": unit.forced_split or "",
                    "force_reasons": json.dumps(sorted(unit.force_reasons), ensure_ascii=False),
                    "resolution": unit.resolution,
                    "selected_source_archive": unit.canonical.archive_name,
                    "selected_source_split": unit.canonical.source_split,
                    "selected_image_index": unit.canonical.image_index,
                    "selected_image_member": unit.canonical.image_member,
                    "selected_label_member": unit.canonical.label_member,
                    "duplicate_source_occurrences": len(unit.sources),
                    "source_occurrences": _source_refs(unit.sources),
                    "width": unit.canonical.width,
                    "height": unit.canonical.height,
                    "box_count": len(unit.boxes),
                    "brand_box_counts": json.dumps(dict(sorted(brand_counts.items())), ensure_ascii=False),
                }
            )
            if len(brand_counts) > 1:
                mixed_rows.append(
                    {
                        "status": "accepted",
                        "image_sha256": unit.image_sha256,
                        "output_split": unit.output_split,
                        "family_component_id": unit.component_id,
                        "brand_count": len(brand_counts),
                        "brands": ";".join(sorted(brand_counts)),
                        "brand_instance_counts": json.dumps(dict(sorted(brand_counts.items()))),
                        "duplicate_occurrence_agreement": "passed",
                        "completeness_limit": (
                            "snapshot agreement passed; visual annotation completeness is not proven"
                        ),
                    }
                )

        copied_quarantine_hashes: set[str] = set()
        for group in quarantined_groups:
            group_brand_sets = set()
            for audit in group["audits"]:
                union_brands = sorted(
                    {
                        box.brand
                        for candidate in audit.candidate_sets
                        for box in candidate
                    }
                )
                group_brand_sets.add(tuple(union_brands))
                canonical = audit.sources[0]
                suffix = PurePosixPath(canonical.image_member).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    suffix = ".jpg"
                quarantine_name = f"q_{audit.image_sha256[:20]}{suffix}"
                quarantine_path = staging / "quarantine" / "images" / quarantine_name
                if audit.image_sha256 not in copied_quarantine_hashes:
                    quarantine_path.write_bytes(read_occurrence_image(canonical, archives))
                    copied_quarantine_hashes.add(audit.image_sha256)
                candidate_ids = []
                for index, candidate in enumerate(audit.candidate_sets):
                    candidate_id = f"candidate_{index:02d}"
                    candidate_ids.append(candidate_id)
                    candidate_path = (
                        staging
                        / "quarantine"
                        / "candidate_labels"
                        / f"{audit.image_sha256}__{candidate_id}.txt"
                    )
                    text = "\n".join(box.yolo_row() for box in candidate)
                    candidate_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
                sheet_path = staging / "review_sheets" / f"{audit.image_sha256}.jpg"
                make_review_sheet(
                    read_occurrence_image(canonical, archives),
                    audit,
                    sheet_path,
                )
                review_paths.append(sheet_path)
                quarantine_rows.append(
                    {
                        "family_component_id": group["group_id"],
                        "image_sha256": audit.image_sha256,
                        "audit_status": audit.status,
                        "audit_reason": audit.reason,
                        "quarantine_reasons": json.dumps(group["quarantine_reasons"], ensure_ascii=False),
                        "family_ids": json.dumps(sorted(audit.family_ids), ensure_ascii=False),
                        "holdout_force_reasons": json.dumps(group["force_reasons"], ensure_ascii=False),
                        "candidate_ids": ";".join(candidate_ids),
                        "candidate_box_counts": json.dumps(
                            [len(candidate) for candidate in audit.candidate_sets]
                        ),
                        "candidate_brand_counts": json.dumps(
                            [
                                dict(sorted(Counter(box.brand for box in candidate).items()))
                                for candidate in audit.candidate_sets
                            ],
                            ensure_ascii=False,
                        ),
                        "quarantine_image": quarantine_path.relative_to(staging).as_posix(),
                        "review_sheet": sheet_path.relative_to(staging).as_posix(),
                        "source_occurrences": _source_refs(audit.sources),
                    }
                )
                if len(union_brands) > 1:
                    mixed_rows.append(
                        {
                            "status": "quarantined",
                            "image_sha256": audit.image_sha256,
                            "output_split": "",
                            "family_component_id": group["group_id"],
                            "brand_count": len(union_brands),
                            "brands": ";".join(union_brands),
                            "brand_instance_counts": "candidate-dependent",
                            "duplicate_occurrence_agreement": "failed_or_family_quarantine",
                            "completeness_limit": audit.reason or "; ".join(group["quarantine_reasons"]),
                        }
                    )
            if len(group_brand_sets) > 1:
                family_variation_rows.append(
                    {
                        "family_component_id": group["group_id"],
                        "status": "quarantined",
                        "image_count": len(group["audits"]),
                        "observed_brand_sets": json.dumps(sorted(group_brand_sets)),
                        "note": "filename-family brand-set variation is heuristic evidence, not proof of omission",
                    }
                )

        preview_units = select_preview_units(units, preview_count)
        preview_paths: list[Path] = []
        for index, unit in enumerate(preview_units, start=1):
            payload = read_occurrence_image(unit.canonical, archives)
            with Image.open(io.BytesIO(payload)) as image:
                rendered = draw_boxes(
                    image,
                    unit.boxes,
                    f"{unit.output_split} {unit.image_sha256[:12]}",
                )
            rendered = ImageOps.contain(rendered, (960, 720), Image.Resampling.LANCZOS)
            preview_path = staging / "previews" / f"preview_{index:03d}.jpg"
            rendered.save(preview_path, format="JPEG", quality=90, optimize=True)
            preview_paths.append(preview_path)
            preview_rows.append(
                {
                    "preview": preview_path.relative_to(staging).as_posix(),
                    "image_sha256": unit.image_sha256,
                    "split": unit.output_split,
                    "brands": ";".join(sorted({box.brand for box in unit.boxes})),
                    "instances": len(unit.boxes),
                }
            )
        make_contact_sheet(preview_paths, staging / "previews" / "contact_sheet.jpg")
        for batch_index in range(0, len(review_paths), 20):
            make_contact_sheet(
                review_paths[batch_index : batch_index + 20],
                staging / "review_sheets" / f"contact_sheet_{batch_index // 20 + 1:03d}.jpg",
            )

    for group in clean_groups:
        brand_sets = {
            tuple(sorted({box.brand for box in unit.boxes}))
            for unit in group["units"]
        }
        if len(brand_sets) > 1:
            family_variation_rows.append(
                {
                    "family_component_id": group["group_id"],
                    "status": "accepted",
                    "image_count": len(group["units"]),
                    "observed_brand_sets": json.dumps(sorted(brand_sets)),
                    "note": "filename-family brand-set variation is heuristic evidence, not proof of omission",
                }
            )

    yaml_payload = {
        "path": final_output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: brand for index, brand in enumerate(TARGET_BRANDS)},
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(
        staging / "mapping.json",
        {
            "output_names": {str(index): brand for index, brand in enumerate(TARGET_BRANDS)},
            "source_mapping_rule": (
                "Use each archive-local verified YAML; exact brand or BRAND_ prefix only, "
                "plus explicit aliases from configs/brands.yaml"
            ),
            "obb_to_aabb": (
                "xmin=min(x_i), ymin=min(y_i), xmax=max(x_i), ymax=max(y_i), then clip to [0,1]"
            ),
            "equivalence": {
                "edge_tolerance": EDGE_TOLERANCE,
                "minimum_iou": IOU_TOLERANCE,
                "same_brand_one_to_one_required": True,
            },
        },
    )
    write_csv(
        staging / "provenance.csv",
        (
            "output_split",
            "output_image_path",
            "output_label_path",
            "image_sha256",
            "family_component_id",
            "family_ids",
            "forced_holdout",
            "force_reasons",
            "resolution",
            "selected_source_archive",
            "selected_source_split",
            "selected_image_index",
            "selected_image_member",
            "selected_label_member",
            "duplicate_source_occurrences",
            "source_occurrences",
            "width",
            "height",
            "box_count",
            "brand_box_counts",
        ),
        provenance_rows,
    )
    write_csv(
        staging / "annotation_conflicts.csv",
        (
            "image_sha256",
            "reason",
            "source_occurrence_count",
            "candidate_set_count",
            "candidate_box_counts",
            "candidate_brand_counts",
            "family_ids",
            "sources",
        ),
        conflict_rows,
    )
    write_csv(
        staging / "annotation_candidates.csv",
        (
            "image_sha256",
            "candidate_id",
            "box_count",
            "brand_counts",
            "yolo_rows",
            "source_occurrences",
        ),
        candidate_rows,
    )
    write_csv(
        staging / "conflict_summary_by_brand.csv",
        (
            "class_id",
            "brand",
            "unresolved_image_hashes",
            "count_or_membership_conflicts",
            "geometry_only_conflicts",
        ),
        conflict_brand_rows,
    )
    write_csv(
        staging / "review_decisions.csv",
        ("image_sha256", "candidate_options", "selected_candidate_id", "reviewer", "notes"),
        (
            {
                "image_sha256": row["image_sha256"],
                "candidate_options": ";".join(
                    f"candidate_{index:02d}"
                    for index in range(int(row["candidate_set_count"]))
                ),
                "selected_candidate_id": "",
                "reviewer": "",
                "notes": "",
            }
            for row in conflict_rows
        ),
    )
    write_csv(
        staging / "out_of_bounds_corrections.csv",
        (
            "archive",
            "source_split",
            "image_sha256",
            "image_member",
            "label_member",
            "line_number",
            "brand",
            "source_class_id",
            "source_class_name",
            "raw_x_min",
            "raw_y_min",
            "raw_x_max",
            "raw_y_max",
            "corrected_x_min",
            "corrected_y_min",
            "corrected_x_max",
            "corrected_y_max",
        ),
        corrections,
    )
    write_csv(
        staging / "rejected_annotations.csv",
        (
            "archive",
            "source_split",
            "image_sha256",
            "image_member",
            "label_member",
            "line_number",
            "brand",
            "source_class_id",
            "source_class_name",
            "reason",
            "raw_line",
        ),
        rejected,
    )
    write_csv(
        staging / "quarantine_manifest.csv",
        (
            "family_component_id",
            "image_sha256",
            "audit_status",
            "audit_reason",
            "quarantine_reasons",
            "family_ids",
            "holdout_force_reasons",
            "candidate_ids",
            "candidate_box_counts",
            "candidate_brand_counts",
            "quarantine_image",
            "review_sheet",
            "source_occurrences",
        ),
        quarantine_rows,
    )
    write_csv(
        staging / "mixed_brand_audit.csv",
        (
            "status",
            "image_sha256",
            "output_split",
            "family_component_id",
            "brand_count",
            "brands",
            "brand_instance_counts",
            "duplicate_occurrence_agreement",
            "completeness_limit",
        ),
        mixed_rows,
    )
    write_csv(
        staging / "family_brand_variation.csv",
        ("family_component_id", "status", "image_count", "observed_brand_sets", "note"),
        family_variation_rows,
    )
    write_csv(
        staging / "previews" / "preview_manifest.csv",
        ("preview", "image_sha256", "split", "brands", "instances"),
        preview_rows,
    )
    write_json(
        staging / "source_family_manifest.json",
        {
            "normalization": (
                "case-fold filename stem; remove terminal .rf hash, _jpg and _augN; "
                "exact duplicate hashes connect every observed family id"
            ),
            "heuristic": True,
            "clean_groups": [
                {
                    "group_id": group["group_id"],
                    "family_ids": group["family_ids"],
                    "output_split": group["output_split"],
                    "forced_split": group["forced_split"],
                    "force_reasons": group["force_reasons"],
                    "image_count": group["image_count"],
                    "brand_instance_counts": group["brand_instance_counts"],
                    "brand_image_counts": group["brand_image_counts"],
                }
                for group in clean_groups
            ],
            "quarantined_groups": [
                {
                    "group_id": group["group_id"],
                    "family_ids": group["family_ids"],
                    "image_count": len(group["audits"]),
                    "included_candidate_image_count": group["image_count"],
                    "quarantine_reasons": group["quarantine_reasons"],
                    "force_reasons": group["force_reasons"],
                    "unresolved_hashes": group["unresolved_hashes"],
                }
                for group in quarantined_groups
            ],
        },
    )
    return {
        "provenance_rows": len(provenance_rows),
        "quarantine_rows": len(quarantine_rows),
        "review_sheets": len(review_paths),
        "bbox_previews": len(preview_rows),
        "mixed_brand_rows": len(mixed_rows),
        "family_brand_variation_rows": len(family_variation_rows),
        "conflict_brand_summary_rows": len(conflict_brand_rows),
    }


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    required = (
        output_dir / "data.yaml",
        output_dir / "mapping.json",
        output_dir / "provenance.csv",
        output_dir / "source_family_manifest.json",
        output_dir / "annotation_conflicts.csv",
        output_dir / "quarantine_manifest.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Detector dataset metadata is incomplete: {missing}")
    data = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    names = normalize_yaml_names(data.get("names"))
    if names != list(TARGET_BRANDS):
        raise ValueError(f"Detector class order mismatch: {names}")
    with (output_dir / "provenance.csv").open("r", encoding="utf-8", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    provenance_by_path = {row["output_image_path"]: row for row in provenance}
    split_stats: dict[str, Any] = {}
    hash_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    holdout_violations: list[dict[str, Any]] = []
    output_hashes: set[str] = set()

    for split in OUTPUT_SPLITS:
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        images = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(path for path in label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Image/label stems differ in split {split}")
        brand_instances: Counter[str] = Counter()
        brand_images: Counter[str] = Counter()
        brand_families: dict[str, set[str]] = defaultdict(set)
        split_hashes: set[str] = set()
        for image_path in images:
            relative_image = image_path.relative_to(output_dir).as_posix()
            row = provenance_by_path.get(relative_image)
            if row is None:
                raise ValueError(f"Image is absent from provenance: {relative_image}")
            digest = sha256_file(image_path)
            if digest != row["image_sha256"]:
                raise ValueError(f"Output image hash mismatch: {image_path}")
            if digest in split_hashes:
                raise ValueError(f"Duplicate exact image content within split {split}: {digest}")
            split_hashes.add(digest)
            output_hashes.add(digest)
            hash_splits[digest].add(split)
            families = [str(item).casefold() for item in json.loads(row["family_ids"])]
            for family in families:
                family_splits[family].add(split)
            if row["forced_holdout"] and row["forced_holdout"] != split:
                holdout_violations.append(row)
            label_path = label_dir / f"{image_path.stem}.txt"
            image_brands: set[str] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Non-detection label row: {label_path}:{line_number}")
                class_float = float(fields[0])
                class_id = int(class_float)
                if class_float != class_id or not 0 <= class_id < len(TARGET_BRANDS):
                    raise ValueError(f"Invalid class id: {label_path}:{line_number}")
                x_center, y_center, width, height = map(float, fields[1:])
                if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
                    raise ValueError(f"Non-finite box: {label_path}:{line_number}")
                if width <= 0 or height <= 0:
                    raise ValueError(f"Degenerate box: {label_path}:{line_number}")
                x_min, x_max = x_center - width / 2, x_center + width / 2
                y_min, y_max = y_center - height / 2, y_center + height / 2
                if min(x_min, y_min) < -1e-7 or max(x_max, y_max) > 1 + 1e-7:
                    raise ValueError(f"Output box exceeds normalized image bounds: {label_path}:{line_number}")
                brand = TARGET_BRANDS[class_id]
                brand_instances[brand] += 1
                image_brands.add(brand)
            if not image_brands:
                raise ValueError(f"Output image has no target annotation: {image_path}")
            component = row["family_component_id"]
            for brand in image_brands:
                brand_images[brand] += 1
                brand_families[brand].add(component)
        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": sum(brand_instances.values()),
            "per_brand": {
                brand: {
                    "images": brand_images[brand],
                    "instances": brand_instances[brand],
                    "families": len(brand_families[brand]),
                }
                for brand in TARGET_BRANDS
            },
        }

    hash_overlap = {
        f"{first}-{second}": len(
            {digest for digest, splits in hash_splits.items() if {first, second} <= splits}
        )
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    family_overlap = {
        family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1
    }
    if any(hash_overlap.values()):
        raise ValueError(f"Exact image hashes cross split boundaries: {hash_overlap}")
    if family_overlap:
        raise ValueError(f"Filename families cross split boundaries: {dict(list(family_overlap.items())[:10])}")
    if holdout_violations:
        raise ValueError(f"Known holdouts were moved: {holdout_violations[:5]}")
    if len(provenance) != sum(stats["images"] for stats in split_stats.values()):
        raise ValueError("Provenance row count differs from dataset image count")
    missing_brands = {
        split: [brand for brand in TARGET_BRANDS if stats["per_brand"][brand]["images"] == 0]
        for split, stats in split_stats.items()
    }
    if any(missing_brands.values()):
        raise ValueError(f"One or more splits lack a target brand: {missing_brands}")
    with (output_dir / "quarantine_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        quarantine_rows = list(csv.DictReader(handle))
    quarantined_hashes = {row["image_sha256"] for row in quarantine_rows}
    overlap_with_quarantine = output_hashes & quarantined_hashes
    if overlap_with_quarantine:
        raise ValueError(
            f"Quarantined images leaked into train/val/test: {sorted(overlap_with_quarantine)[:10]}"
        )
    with (output_dir / "annotation_conflicts.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        conflict_rows = list(csv.DictReader(handle))
    training_ready = not conflict_rows and not quarantine_rows
    return {
        "status": "structural_checks_passed",
        "training_ready": training_ready,
        "mapping_verified": {str(index): brand for index, brand in enumerate(TARGET_BRANDS)},
        "ordinary_yolo_detection_rows_only": True,
        "splits": split_stats,
        "exact_image_cross_split_overlap": hash_overlap,
        "cross_split_family_overlap": 0,
        "known_holdout_violations": 0,
        "quarantined_hashes_in_splits": 0,
        "unresolved_annotation_hashes": len(conflict_rows),
        "quarantined_image_hashes": len(quarantine_rows),
        "all_nine_brands_present_in_every_split": True,
        "annotation_completeness_visually_proven": False,
        "test_split_evaluated": False,
    }


def write_readme(
    path: Path,
    checks: dict[str, Any],
    report: dict[str, Any],
) -> None:
    unresolved = checks["unresolved_annotation_hashes"]
    quarantine = checks["quarantined_image_hashes"]
    lines = [
        f"# {len(TARGET_BRANDS)}-brand YOLO detection dataset",
        "",
        "Class order: "
        + ", ".join(f"{index} {brand}" for index, brand in enumerate(TARGET_BRANDS))
        + ".",
        "",
        f"Status: structural checks passed; training_ready={str(checks['training_ready']).lower()}.",
        f"Unresolved exact-image annotation hashes: {unresolved}.",
        f"Quarantined image hashes (including connected family members): {quarantine}.",
        "",
        "No source snapshot order, generic-class ratio, or box union was used to resolve conflicts. "
        "Equivalent sets require identical brand counts and a same-brand one-to-one AABB match "
        f"with edge delta <= {EDGE_TOLERANCE} and IoU >= {IOU_TOLERANCE}.",
        "",
        "Manual review workflow:",
        "",
        "1. Inspect review_sheets and annotation_candidates.csv.",
        "2. Fill selected_candidate_id in review_decisions.csv; keep reviewer and notes.",
        "3. Re-run the preparation script into a new/empty output path with --decisions.",
        "4. Run the training preflight. It refuses to train while training_ready is false.",
        "",
        "Manual commands from repository root:",
        "",
        "    python scripts/prepare_brand_9brands_yolo.py --check-only",
        "    python scripts/train_brand_9brands_yolo.py --preflight",
        "    python scripts/train_brand_9brands_yolo.py",
        "    python scripts/report_brand_9brands_validation.py --weights runs/model1_brand9/yolov8n_100ep_s42/weights/best.pt",
        "",
        "The last command evaluates validation only. Test evaluation remains disabled.",
        "",
        "## Prepared clean-subset counts",
        "",
        "| Split | Images | Instances |",
        "|---|---:|---:|",
    ]
    for split in OUTPUT_SPLITS:
        stats = checks["splits"][split]
        lines.append(f"| {split} | {stats['images']} | {stats['instances']} |")
    lines.extend(
        [
            "",
            "| Split | Brand | Images | Instances | Families |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split in OUTPUT_SPLITS:
        for brand in TARGET_BRANDS:
            stats = checks["splits"][split]["per_brand"][brand]
            lines.append(
                f"| {split} | {brand} | {stats['images']} | "
                f"{stats['instances']} | {stats['families']} |"
            )
    conversion = report["conversion"]
    lines.extend(
        [
            "",
            "## Audit safeguards",
            "",
            f"- OBB boxes clipped to image bounds: {conversion['out_of_bounds_source_boxes_corrected']}.",
            f"- Rejected source annotation rows: {conversion['rejected_source_annotation_rows']}.",
            f"- Rejected degenerate boxes: {conversion['degenerate_boxes_rejected']}.",
            "- Exact image cross-split overlap: 0.",
            "- Filename-family cross-split overlap: 0 (filename grouping remains heuristic).",
            "- Known holdout violations: 0.",
            "- Quarantined hashes present in train/val/test: 0.",
            "",
        ]
    )
    if report.get("annotation_conflicts_by_brand"):
        lines.extend(
            [
                "## Unresolved conflicts by brand",
                "",
                "| Brand | Affected hashes | Count/membership | Geometry-only |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in report["annotation_conflicts_by_brand"]:
            lines.append(
                f"| {row['brand']} | {row['unresolved_image_hashes']} | "
                f"{row['count_or_membership_conflicts']} | "
                f"{row['geometry_only_conflicts']} |"
            )
        lines.append("")
    lines.extend(
        [
        "Important limitation: agreement among source snapshots and structurally valid labels does "
        "not prove every visible product is annotated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(
    zip_dir: Path,
    inventory_dir: Path,
    five_brand_dir: Path,
    output_dir: Path,
    decisions_path: Path | None,
    seed: int,
    preview_count: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    inventory = load_inventory(inventory_dir)
    source_reports, records, corrections, rejected = audit_archives(zip_dir, inventory)
    decisions = load_decisions(decisions_path)
    units, audits, conflicts, candidates, dedup_report = reconcile_exact_duplicates(
        records, decisions
    )
    five_hash_locks, five_family_locks, five_report = load_five_brand_holdouts(
        five_brand_dir
    )
    model2_assignments, model2_report = load_model2_family_assignments(ROOT)
    clean_groups, quarantined_groups, family_report = build_family_components(
        units,
        audits,
        five_hash_locks,
        five_family_locks,
        model2_assignments,
    )
    clean_hashes = {
        unit.image_sha256 for group in clean_groups for unit in group["units"]
    }
    clean_units = sorted(
        [unit for unit in units if unit.image_sha256 in clean_hashes],
        key=lambda unit: unit.image_sha256,
    )
    split_report = assign_splits(clean_groups, seed)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    ).resolve()
    try:
        artifact_report = write_dataset(
            staging,
            output_dir.resolve(),
            clean_units,
            clean_groups,
            quarantined_groups,
            audits,
            conflicts,
            candidates,
            corrections,
            rejected,
            preview_count,
        )
        preliminary = {
            "status": "prepared_with_quarantine" if conflicts or quarantined_groups else "prepared",
            "training_ready": False,
            "target_brands": list(TARGET_BRANDS),
            "output_class_mapping": {
                str(index): brand for index, brand in enumerate(TARGET_BRANDS)
            },
            "inventory_cross_check": inventory,
            "source_archives": source_reports,
            "deduplication_and_annotation_agreement": dedup_report,
            "five_brand_holdouts": five_report,
            "model2_holdouts": model2_report,
            "family_components": family_report,
            "split_assignment": split_report,
            "conversion": {
                "source_format": "YOLOv8 OBB: class_id plus four x/y points",
                "output_format": "ordinary axis-aligned YOLO detection",
                "out_of_bounds_source_boxes_corrected": len(corrections),
                "rejected_source_annotation_rows": len(rejected),
                "degenerate_boxes_rejected": sum(
                    "degenerate" in row.get("reason", "").lower() for row in rejected
                ),
            },
            "artifacts": artifact_report,
            "manual_decisions_remaining": {
                "unresolved_annotation_hashes": len(conflicts),
                "quarantined_family_components": len(quarantined_groups),
                "review_template": "review_decisions.csv",
                "review_sheets": "review_sheets/",
                "conflict_summary": "conflict_summary_by_brand.csv",
            },
            "annotation_conflicts_by_brand": summarize_conflicts_by_brand(
                conflicts, candidates
            ),
            "annotation_completeness": {
                "duplicate_snapshot_agreement_checked_for_all_target_brands": True,
                "mixed_brand_report": "mixed_brand_audit.csv",
                "family_brand_variation_report": "family_brand_variation.csv",
                "visually_complete_annotations_proven": False,
            },
            "test_evaluation_performed": False,
            "training_started": False,
        }
        write_json(staging / "dataset_report.json", preliminary)
        checks = validate_dataset(staging)
        write_json(staging / "dataset_checks.json", checks)
        preliminary["training_ready"] = checks["training_ready"]
        preliminary["checks"] = checks
        write_json(staging / "dataset_report.json", preliminary)
        write_readme(staging / "README.md", checks, preliminary)
        staging.rename(output_dir)
        return preliminary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_brand_profile(args.brand_profile)
    output_dir = resolve_path(args.output_dir)
    if args.check_only:
        print(json.dumps(validate_dataset(output_dir), ensure_ascii=False, indent=2))
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report = prepare_dataset(
        resolve_path(args.zip_dir),
        resolve_path(args.inventory_dir),
        resolve_path(args.five_brand_dir),
        output_dir,
        resolve_path(args.decisions) if args.decisions else None,
        args.seed,
        args.preview_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





