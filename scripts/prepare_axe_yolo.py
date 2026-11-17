"""Prepare a leakage-aware, single-class AXE YOLO detection dataset from export v8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from zipfile import ZipFile, ZipInfo

import yaml
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_clean_axe_dataset import normalize_source_family


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLITS = ("train", "valid", "test", "export")
OUTPUT_SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
QUARANTINE_REASONS = {
    "z7273606252494_cb4e5850aa00c22b5dc6bed2bc624453_aug1": (
        "Visible AXE Dark product is incorrectly annotated as Lifebuoy."
    ),
    "c71fc10190441b1a425548": "Visible AXE products but annotations are empty.",
    "8f1749651820937eca3124": "Visible AXE products but annotations are empty.",
}
PROVENANCE_FIELDS = (
    "output_split",
    "output_image_path",
    "output_label_path",
    "image_sha256",
    "family_id",
    "family_ids",
    "model2_holdout",
    "source_split",
    "source_member",
    "source_filename",
    "source_label_member",
    "duplicate_source_count",
    "duplicate_source_members",
    "width",
    "height",
    "axe_box_count",
    "source_axe_classes",
    "non_axe_annotations_removed",
)


@dataclass(frozen=True)
class Box:
    source_line: int
    source_class_id: int
    source_class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def yolo_row(self) -> str:
        return (
            f"0 {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


@dataclass
class SourceRecord:
    image_member: str
    label_member: str
    source_split: str
    source_filename: str
    family_id: str
    image_sha256: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    non_axe_removed: int = 0
    quarantine_prefix: str | None = None


@dataclass
class ImageUnit:
    image_sha256: str
    canonical: SourceRecord
    sources: list[SourceRecord]
    family_ids: set[str]
    component_id: str = ""
    forced_split: str | None = None
    output_split: str = ""
    output_name: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-zip-glob",
        default="data/brand_zips/BRAND_CLASSIFICATION_YOLO.v8-*.yolov8.zip",
        help="Glob restricted to the Roboflow version-8 archive.",
    )
    parser.add_argument("--output-dir", default="data/axe_yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only validation of an already prepared --output-dir.",
    )
    return parser.parse_args(argv)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_zip_entry(archive: ZipFile, info_or_name: ZipInfo | str) -> bytes:
    with archive.open(info_or_name, "r") as handle:
        return handle.read()


def read_source_yaml(archive: ZipFile) -> tuple[dict[str, Any], dict[str, Any]]:
    yaml_infos = [
        info
        for info in archive.infolist()
        if PurePosixPath(info.filename).name.lower() in {"data.yaml", "data.yml"}
    ]
    if not yaml_infos:
        raise FileNotFoundError("The v8 archive contains no data.yaml")
    payloads = [_read_zip_entry(archive, info) for info in yaml_infos]
    hashes = [sha256_bytes(payload) for payload in payloads]
    if len(set(payloads)) != 1:
        raise ValueError("Repeated data.yaml ZIP entries do not have identical contents")
    data = yaml.safe_load(payloads[0].decode("utf-8-sig")) or {}
    names_value = data.get("names")
    if isinstance(names_value, dict):
        names = [str(names_value[index]) for index in sorted(names_value, key=int)]
    elif isinstance(names_value, list):
        names = [str(value) for value in names_value]
    else:
        raise ValueError("Source data.yaml names must be a list or integer-keyed mapping")
    if int(data.get("nc", len(names))) != len(names):
        raise ValueError("Source data.yaml nc does not match names")
    data["names"] = names
    return data, {
        "entry_count": len(yaml_infos),
        "identical_contents": True,
        "sha256": hashes[0],
    }


def polygon_to_box(
    values: Sequence[float],
    source_line: int,
    source_class_id: int,
    source_class_name: str,
) -> Box:
    if len(values) < 6 or len(values) % 2:
        raise ValueError("AXE annotation is not a polygon with at least three x/y points")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("polygon contains a non-finite coordinate")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("normalized polygon coordinate is outside [0, 1]")
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width, height = x_max - x_min, y_max - y_min
    if width <= 0.0 or height <= 0.0:
        raise ValueError("polygon has a degenerate enclosing bounding box")
    box = Box(
        source_line=source_line,
        source_class_id=source_class_id,
        source_class_name=source_class_name,
        x_center=(x_min + x_max) / 2.0,
        y_center=(y_min + y_max) / 2.0,
        width=width,
        height=height,
    )
    if any(
        value < 0.0 or value > 1.0
        for value in (box.x_center, box.y_center, box.width, box.height)
    ):
        raise ValueError("derived bounding box is outside normalized bounds")
    return box


def convert_label_text(
    text: str,
    class_names: Sequence[str],
    source_label: str,
) -> tuple[list[Box], int, list[dict[str, Any]], Counter[int]]:
    boxes: list[Box] = []
    non_axe_removed = 0
    rejected: list[dict[str, Any]] = []
    source_counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
        except (ValueError, IndexError):
            rejected.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "source_class_id": "",
                    "reason": "invalid class id",
                    "raw_line": raw_line,
                }
            )
            continue
        if not 0 <= class_id < len(class_names):
            rejected.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "source_class_id": class_id,
                    "reason": "class id is outside source YAML names",
                    "raw_line": raw_line,
                }
            )
            continue
        source_counts[class_id] += 1
        class_name = class_names[class_id]
        if not class_name.startswith("AXE_"):
            non_axe_removed += 1
            continue
        try:
            values = [float(value) for value in fields[1:]]
            boxes.append(
                polygon_to_box(values, line_number, class_id, class_name)
            )
        except ValueError as error:
            rejected.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "source_class_id": class_id,
                    "reason": str(error),
                    "raw_line": raw_line,
                }
            )
    return boxes, non_axe_removed, rejected, source_counts


def _label_member_for(image_member: str) -> str:
    image_path = PurePosixPath(image_member)
    parts = list(image_path.parts)
    try:
        image_directory = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Image is not under an images directory: {image_member}") from error
    parts[image_directory] = "labels"
    return str(PurePosixPath(*parts).with_suffix(".txt"))


def inspect_source_records(
    archive: ZipFile,
    class_names: Sequence[str],
) -> tuple[list[SourceRecord], list[dict[str, Any]], dict[str, Any]]:
    image_members = sorted(
        info.filename
        for info in archive.infolist()
        if PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
        and PurePosixPath(info.filename).parts
        and PurePosixPath(info.filename).parts[0] in SOURCE_SPLITS
    )
    available = {info.filename for info in archive.infolist()}
    if len(image_members) != len(set(image_members)):
        raise ValueError("The source ZIP contains duplicate image member names")

    records: list[SourceRecord] = []
    rejected: list[dict[str, Any]] = []
    source_class_counts: Counter[int] = Counter()
    for image_member in image_members:
        label_member = _label_member_for(image_member)
        if label_member not in available:
            raise FileNotFoundError(f"No matching label for ZIP image: {image_member}")
        image_bytes = _read_zip_entry(archive, image_member)
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception as error:
            raise ValueError(f"Unreadable source image: {image_member}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"Source image has an empty dimension: {image_member}")
        label_text = _read_zip_entry(archive, label_member).decode("utf-8-sig")
        boxes, removed, row_rejections, row_counts = convert_label_text(
            label_text, class_names, label_member
        )
        rejected.extend(row_rejections)
        source_class_counts.update(row_counts)
        source_filename = PurePosixPath(image_member).name
        quarantine_prefix = next(
            (prefix for prefix in QUARANTINE_REASONS if source_filename.startswith(prefix)),
            None,
        )
        records.append(
            SourceRecord(
                image_member=image_member,
                label_member=label_member,
                source_split=PurePosixPath(image_member).parts[0],
                source_filename=source_filename,
                family_id=normalize_source_family(source_filename),
                image_sha256=sha256_bytes(image_bytes),
                width=width,
                height=height,
                boxes=boxes,
                non_axe_removed=removed,
                quarantine_prefix=quarantine_prefix,
            )
        )

    quarantine_matches = Counter(
        record.quarantine_prefix for record in records if record.quarantine_prefix
    )
    missing_or_ambiguous = {
        prefix: quarantine_matches[prefix]
        for prefix in QUARANTINE_REASONS
        if quarantine_matches[prefix] != 1
    }
    if missing_or_ambiguous:
        raise ValueError(
            "Each configured quarantine prefix must match exactly one image: "
            f"{missing_or_ambiguous}"
        )

    family_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        family_splits[record.family_id].add(record.source_split)
    overlapping_families = {
        family: sorted(splits)
        for family, splits in sorted(family_splits.items())
        if len(splits) > 1
    }
    inspection = {
        "image_count": len(records),
        "source_split_image_counts": dict(sorted(Counter(r.source_split for r in records).items())),
        "family_count": len(family_splits),
        "cross_source_split_family_count": len(overlapping_families),
        "cross_source_split_families": overlapping_families,
        "source_annotation_count": sum(source_class_counts.values()),
        "source_axe_annotation_count": sum(
            count
            for class_id, count in source_class_counts.items()
            if class_names[class_id].startswith("AXE_")
        ),
        "source_non_axe_annotation_count": sum(
            count
            for class_id, count in source_class_counts.items()
            if not class_names[class_id].startswith("AXE_")
        ),
        "source_class_annotation_counts": {
            class_names[class_id]: source_class_counts[class_id]
            for class_id in sorted(source_class_counts)
        },
    }
    return records, rejected, inspection


def load_model2_family_assignments(repo_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    sources: list[str] = []
    family_manifest = repo_root / "data" / "axe_dataset_clean" / "source_family_manifest.json"
    if family_manifest.is_file():
        payload = json.loads(family_manifest.read_text(encoding="utf-8"))
        for row in payload.get("families", []):
            assignments[str(row["family_id"]).casefold()].add(str(row["output_split"]))
        sources.append(str(family_manifest))
    crop_manifest = repo_root / "data" / "axe_crops" / "crop_manifest.csv"
    if crop_manifest.is_file():
        with crop_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                assignments[str(row["family_id"]).casefold()].add(str(row["split"]))
        sources.append(str(crop_manifest))

    normalized: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    aliases = {"valid": "val", "validation": "val", "val": "val", "train": "train", "test": "test"}
    for family, raw_splits in assignments.items():
        try:
            splits = {aliases[value] for value in raw_splits}
        except KeyError as error:
            raise ValueError(f"Unknown Model 2 split {error.args[0]!r} for family {family}") from error
        if len(splits) > 1:
            conflicts[family] = sorted(splits)
        else:
            normalized[family] = next(iter(splits))
    if conflicts:
        raise ValueError(f"Conflicting Model 2 family assignments: {dict(list(conflicts.items())[:10])}")
    return normalized, {
        "available": bool(sources),
        "sources": sources,
        "family_count": len(normalized),
        "conflict_count": 0,
    }


def deduplicate_records(records: Sequence[SourceRecord]) -> tuple[list[ImageUnit], dict[str, Any]]:
    by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.image_sha256].append(record)
    units: list[ImageUnit] = []
    conflicting_labels: list[dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        signatures = {
            tuple(sorted(box.yolo_row() for box in member.boxes)) for member in members
        }
        if len(signatures) > 1:
            conflicting_labels.append(
                {
                    "image_sha256": digest,
                    "source_members": [member.image_member for member in members],
                }
            )
            continue
        ordered = sorted(members, key=lambda item: item.image_member.casefold())
        units.append(
            ImageUnit(
                image_sha256=digest,
                canonical=ordered[0],
                sources=ordered,
                family_ids={member.family_id for member in members},
            )
        )
    if conflicting_labels:
        raise ValueError(
            "Exact duplicate images have conflicting AXE labels; refusing to choose silently: "
            f"{conflicting_labels[:5]}"
        )
    return units, {
        "exact_duplicate_groups": sum(1 for members in by_hash.values() if len(members) > 1),
        "exact_duplicate_images_removed": sum(len(members) - 1 for members in by_hash.values()),
        "conflicting_duplicate_label_groups": 0,
    }


def group_units_by_family(units: Sequence[ImageUnit]) -> list[dict[str, Any]]:
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            lower, upper = sorted((first_root, second_root))
            parents[upper] = lower

    for unit in units:
        families = sorted(unit.family_ids)
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)
    grouped: dict[str, list[ImageUnit]] = defaultdict(list)
    for unit in units:
        component = min(find(family) for family in unit.family_ids)
        unit.component_id = component
        grouped[component].append(unit)
    return [
        {
            "group_id": component,
            "units": sorted(members, key=lambda item: item.canonical.image_member.casefold()),
            "image_count": len(members),
            "family_ids": sorted({family for unit in members for family in unit.family_ids}),
            "forced_split": None,
        }
        for component, members in sorted(grouped.items())
    ]


def assign_grouped_splits(
    groups: list[dict[str, Any]],
    model2_assignments: dict[str, str],
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    ratios = dict(ratios or SPLIT_RATIOS)
    if set(ratios) != set(OUTPUT_SPLITS) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must define train/val/test and sum to one")
    forced_counts: Counter[str] = Counter()
    matched_model2: Counter[str] = Counter()
    for group in groups:
        known = {
            model2_assignments[family]
            for family in group["family_ids"]
            if family in model2_assignments
        }
        holdouts = known & {"val", "test"}
        if len(holdouts) > 1:
            raise ValueError(
                f"Grouped source families have conflicting Model 2 val/test holdouts: {group['family_ids']}"
            )
        group["forced_split"] = next(iter(holdouts)) if holdouts else None
        for split in known:
            matched_model2[split] += 1
        if group["forced_split"]:
            forced_counts[group["forced_split"]] += group["image_count"]

    total = sum(int(group["image_count"]) for group in groups)
    targets = {split: total * ratios[split] for split in OUTPUT_SPLITS}
    counts: Counter[str] = Counter()
    for group in groups:
        if group["forced_split"]:
            group["output_split"] = group["forced_split"]
            counts[group["output_split"]] += group["image_count"]

    rng = random.Random(seed)
    tie = {group["group_id"]: rng.random() for group in groups}
    unassigned = sorted(
        (group for group in groups if not group["forced_split"]),
        key=lambda group: (-group["image_count"], tie[group["group_id"]], group["group_id"]),
    )
    split_ties = {
        (group["group_id"], split): rng.random()
        for group in unassigned
        for split in OUTPUT_SPLITS
    }
    for group in unassigned:
        size = group["image_count"]

        def score(split: str) -> tuple[float, float, int]:
            proposed = dict(counts)
            proposed[split] = proposed.get(split, 0) + size
            squared_error = sum(
                ((proposed.get(name, 0) - targets[name]) ** 2) / max(targets[name], 1.0)
                for name in OUTPUT_SPLITS
            )
            return squared_error, split_ties[(group["group_id"], split)], OUTPUT_SPLITS.index(split)

        selected = min(OUTPUT_SPLITS, key=score)
        group["output_split"] = selected
        counts[selected] += size

    for group in groups:
        for unit in group["units"]:
            unit.output_split = group["output_split"]
            unit.forced_split = group["forced_split"]
    return {
        "seed": seed,
        "target_ratios": ratios,
        "target_image_counts": targets,
        "actual_image_counts": {split: counts[split] for split in OUTPUT_SPLITS},
        "actual_ratios": {
            split: counts[split] / total if total else 0.0 for split in OUTPUT_SPLITS
        },
        "forced_holdout_image_counts": {
            split: forced_counts[split] for split in ("val", "test")
        },
        "matched_model2_family_references": dict(sorted(matched_model2.items())),
        "group_count": len(groups),
    }


def _draw_boxes(image: Image.Image, boxes: Sequence[Box], title: str) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 180))
    for box in boxes:
        left = (box.x_center - box.width / 2.0) * width
        top = (box.y_center - box.height / 2.0) * height
        right = (box.x_center + box.width / 2.0) * width
        bottom = (box.y_center + box.height / 2.0) * height
        draw.rectangle((left, top, right, bottom), outline=(255, 45, 45), width=line_width)
        draw.text((left + 3, max(0, top - 12)), "AXE", fill=(255, 45, 45))
    draw.rectangle((0, 0, min(width, 460), 24), fill=(255, 255, 255))
    draw.text((5, 5), title[:72], fill=(0, 0, 0))
    return canvas


def write_previews(
    output_dir: Path,
    units: Sequence[ImageUnit],
    archive: ZipFile,
    preview_count: int,
) -> list[dict[str, Any]]:
    if preview_count < 1:
        raise ValueError("--preview-count must be at least 1")
    selected: list[ImageUnit] = []
    per_split = max(1, preview_count // len(OUTPUT_SPLITS))
    for split in OUTPUT_SPLITS:
        candidates = sorted(
            (unit for unit in units if unit.output_split == split),
            key=lambda unit: unit.output_name,
        )
        selected.extend(candidates[:per_split])
    if len(selected) < preview_count:
        already = {unit.image_sha256 for unit in selected}
        selected.extend(
            unit
            for unit in sorted(units, key=lambda item: item.output_name)
            if unit.image_sha256 not in already
        )
    selected = selected[:preview_count]

    rows: list[dict[str, Any]] = []
    thumbnails: list[tuple[Image.Image, str]] = []
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for unit in selected:
        with Image.open(io.BytesIO(_read_zip_entry(archive, unit.canonical.image_member))) as source:
            annotated = _draw_boxes(
                source,
                unit.canonical.boxes,
                f"{unit.output_split} | {unit.output_name} | boxes={len(unit.canonical.boxes)}",
            )
        target = preview_dir / f"{unit.output_split}_{Path(unit.output_name).stem}.jpg"
        annotated.save(target, quality=92)
        thumb = ImageOps.contain(annotated, (320, 240), Image.Resampling.LANCZOS)
        thumbnails.append((thumb, f"{unit.output_split}/{unit.output_name}"))
        rows.append(
            {
                "output_split": unit.output_split,
                "output_image": f"images/{unit.output_split}/{unit.output_name}",
                "source_member": unit.canonical.image_member,
                "box_count": len(unit.canonical.boxes),
                "preview_path": str(target.relative_to(output_dir)).replace("\\", "/"),
            }
        )

    columns, cell_width, cell_height = 3, 340, 275
    rows_count = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, title) in enumerate(thumbnails):
        column, row = index % columns, index // columns
        x, y = column * cell_width, row * cell_height
        sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 24 + (240 - thumb.height) // 2))
        draw.text((x + 5, y + 5), title[:48], fill="black")
    sheet.save(preview_dir / "contact_sheet.jpg", quality=92)
    _write_csv(
        preview_dir / "preview_manifest.csv",
        ("output_split", "output_image", "source_member", "box_count", "preview_path"),
        rows,
    )
    return rows


def write_quarantine(
    output_dir: Path,
    records: Sequence[SourceRecord],
    archive: ZipFile,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    review_images: list[tuple[Image.Image, str]] = []
    for index, record in enumerate(sorted(records, key=lambda item: item.image_member), start=1):
        suffix = PurePosixPath(record.image_member).suffix.lower()
        output_stem = f"q{index:03d}"
        image_target = output_dir / "quarantine" / "images" / f"{output_stem}{suffix}"
        label_target = output_dir / "quarantine" / "labels" / f"{output_stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = _read_zip_entry(archive, record.image_member)
        image_target.write_bytes(image_bytes)
        label_target.write_bytes(_read_zip_entry(archive, record.label_member))
        reason = QUARANTINE_REASONS[record.quarantine_prefix or ""]
        with Image.open(io.BytesIO(image_bytes)) as image:
            review_images.append((ImageOps.contain(image.convert("RGB"), (320, 260)), reason))
        rows.append(
            {
                "quarantine_id": output_stem,
                "source_filename_prefix": record.quarantine_prefix,
                "reason": reason,
                "source_split": record.source_split,
                "source_image_member": record.image_member,
                "source_label_member": record.label_member,
                "review_image_path": str(image_target.relative_to(output_dir)).replace("\\", "/"),
                "review_label_path": str(label_target.relative_to(output_dir)).replace("\\", "/"),
                "excluded_from_all_splits": True,
            }
        )
    sheet = Image.new("RGB", (360 * len(review_images), 340), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, reason) in enumerate(review_images):
        x = index * 360
        sheet.paste(image, (x + (320 - image.width) // 2, 5 + (260 - image.height) // 2))
        draw.multiline_text((x + 5, 272), "\n".join([reason[i : i + 48] for i in range(0, len(reason), 48)]), fill="black")
    sheet_target = output_dir / "quarantine" / "contact_sheet.jpg"
    sheet_target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_target, quality=92)
    _write_csv(
        output_dir / "quarantine_report.csv",
        (
            "quarantine_id",
            "source_filename_prefix",
            "reason",
            "source_split",
            "source_image_member",
            "source_label_member",
            "review_image_path",
            "review_label_path",
            "excluded_from_all_splits",
        ),
        rows,
    )
    _write_json(output_dir / "quarantine_report.json", rows)
    return rows


def validate_derived_dataset(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    required = (
        output_dir / "data.yaml",
        output_dir / "mapping.json",
        output_dir / "dataset_report.json",
        output_dir / "provenance.csv",
        output_dir / "quarantine_report.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset is incomplete: {missing}")
    mapping = json.loads((output_dir / "mapping.json").read_text(encoding="utf-8"))
    if mapping.get("output_names") != {"0": "AXE"}:
        raise ValueError("Derived mapping must contain only class 0 = AXE")
    yaml_payload = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    if yaml_payload.get("names") != {0: "AXE"}:
        raise ValueError("Derived data.yaml must contain only class 0 = AXE")

    counts: dict[str, dict[str, int]] = {}
    all_stems: dict[str, set[str]] = {}
    for split in OUTPUT_SPLITS:
        images = sorted((output_dir / "images" / split).glob("*"))
        labels = sorted((output_dir / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in images if path.suffix.lower() in IMAGE_SUFFIXES}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(f"Image/label stem mismatch in derived {split} split")
        box_count = 0
        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(f"Invalid YOLO detection row: {label_path}:{line_number}")
                values = [float(value) for value in fields[1:]]
                if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
                    raise ValueError(f"Invalid normalized box: {label_path}:{line_number}")
                if values[2] <= 0.0 or values[3] <= 0.0:
                    raise ValueError(f"Degenerate derived box: {label_path}:{line_number}")
                box_count += 1
        counts[split] = {"images": len(image_stems), "labels": len(label_stems), "boxes": box_count}
        all_stems[split] = image_stems
    if any(all_stems[first] & all_stems[second] for first in OUTPUT_SPLITS for second in OUTPUT_SPLITS if first < second):
        raise ValueError("Derived output filename occurs in more than one split")

    with (output_dir / "provenance.csv").open("r", encoding="utf-8", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    family_splits: dict[str, set[str]] = defaultdict(set)
    holdout_violations: list[dict[str, str]] = []
    for row in provenance:
        for family in json.loads(row["family_ids"]):
            family_splits[family].add(row["output_split"])
        if row["model2_holdout"] and row["model2_holdout"] != row["output_split"]:
            holdout_violations.append(row)
    leaking = {family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"Derived source families cross output splits: {dict(list(leaking.items())[:10])}")
    if holdout_violations:
        raise ValueError(f"Model 2 holdout violations: {holdout_violations[:5]}")
    return {
        "status": "passed",
        "mapping_is_single_class_axe": True,
        "splits": counts,
        "cross_split_family_overlap": 0,
        "model2_holdout_violations": 0,
        "test_split_evaluated": False,
    }


def prepare_dataset(source_zip: Path, output_dir: Path, seed: int, preview_count: int) -> dict[str, Any]:
    source_zip = source_zip.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_staging_", dir=output_dir.parent)).resolve()
    try:
        with ZipFile(source_zip) as archive:
            source_yaml, yaml_inspection = read_source_yaml(archive)
            class_names = source_yaml["names"]
            axe_classes = [
                {"source_class_id": index, "source_class_name": name, "output_class_id": 0, "output_class_name": "AXE"}
                for index, name in enumerate(class_names)
                if name.startswith("AXE_")
            ]
            if not axe_classes:
                raise ValueError("Source YAML contains no classes starting with AXE_")
            records, rejected, source_inspection = inspect_source_records(archive, class_names)
            quarantined = [record for record in records if record.quarantine_prefix]
            active = [record for record in records if not record.quarantine_prefix]
            units, deduplication = deduplicate_records(active)
            groups = group_units_by_family(units)
            model2_assignments, model2_report = load_model2_family_assignments(ROOT)
            split_report = assign_grouped_splits(groups, model2_assignments, seed)

            for index, unit in enumerate(
                sorted(units, key=lambda item: item.canonical.image_member.casefold()), start=1
            ):
                suffix = PurePosixPath(unit.canonical.image_member).suffix.lower()
                unit.output_name = f"a{index:04d}{suffix}"

            provenance_rows: list[dict[str, Any]] = []
            for unit in units:
                image_target = staging / "images" / unit.output_split / unit.output_name
                label_target = staging / "labels" / unit.output_split / f"{Path(unit.output_name).stem}.txt"
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                image_target.write_bytes(_read_zip_entry(archive, unit.canonical.image_member))
                rows = [box.yolo_row() for box in unit.canonical.boxes]
                label_target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                source_classes = sorted({box.source_class_name for box in unit.canonical.boxes})
                provenance_rows.append(
                    {
                        "output_split": unit.output_split,
                        "output_image_path": f"images/{unit.output_split}/{unit.output_name}",
                        "output_label_path": f"labels/{unit.output_split}/{Path(unit.output_name).stem}.txt",
                        "image_sha256": unit.image_sha256,
                        "family_id": unit.component_id,
                        "family_ids": json.dumps(sorted(unit.family_ids), ensure_ascii=False),
                        "model2_holdout": unit.forced_split or "",
                        "source_split": unit.canonical.source_split,
                        "source_member": unit.canonical.image_member,
                        "source_filename": unit.canonical.source_filename,
                        "source_label_member": unit.canonical.label_member,
                        "duplicate_source_count": len(unit.sources),
                        "duplicate_source_members": json.dumps(
                            [source.image_member for source in unit.sources], ensure_ascii=False
                        ),
                        "width": unit.canonical.width,
                        "height": unit.canonical.height,
                        "axe_box_count": len(unit.canonical.boxes),
                        "source_axe_classes": json.dumps(source_classes, ensure_ascii=False),
                        "non_axe_annotations_removed": unit.canonical.non_axe_removed,
                    }
                )

            _write_csv(staging / "provenance.csv", PROVENANCE_FIELDS, provenance_rows)
            _write_csv(
                staging / "rejected_annotations.csv",
                ("source_label", "line_number", "source_class_id", "reason", "raw_line"),
                rejected,
            )
            quarantine_rows = write_quarantine(staging, quarantined, archive)
            preview_rows = write_previews(staging, units, archive, preview_count)

            mapping = {
                "output_names": {"0": "AXE"},
                "selection_rule": "Only source class names beginning exactly with 'AXE_' are mapped to output class 0.",
                "single_cls_training_option": False,
                "source_class_count": len(class_names),
                "source_axe_class_count": len(axe_classes),
                "source_classes": [
                    {
                        "source_class_id": index,
                        "source_class_name": name,
                        "mapped_to_output": 0 if name.startswith("AXE_") else None,
                    }
                    for index, name in enumerate(class_names)
                ],
                "axe_mappings": axe_classes,
            }
            _write_json(staging / "mapping.json", mapping)
            data_yaml = {
                "path": output_dir.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "AXE"},
            }
            (staging / "data.yaml").write_text(
                yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )

            split_counts: dict[str, Any] = {}
            for split in OUTPUT_SPLITS:
                split_units = [unit for unit in units if unit.output_split == split]
                split_counts[split] = {
                    "images": len(split_units),
                    "axe_boxes": sum(len(unit.canonical.boxes) for unit in split_units),
                    "families": len({unit.component_id for unit in split_units}),
                    "forced_model2_holdout_images": sum(
                        1 for unit in split_units if unit.forced_split == split
                    ),
                    "source_axe_class_box_counts": dict(
                        sorted(
                            Counter(
                                box.source_class_name
                                for unit in split_units
                                for box in unit.canonical.boxes
                            ).items()
                        )
                    ),
                }
            active_family_ids = {record.family_id for record in active}
            matched_assignments = {
                family: model2_assignments[family]
                for family in sorted(active_family_ids & set(model2_assignments))
            }
            report = {
                "status": "prepared",
                "source_archive": str(source_zip),
                "source_archive_sha256": sha256_bytes(source_zip.read_bytes()),
                "source_version": source_yaml.get("roboflow", {}).get("version"),
                "source_yaml_entries": yaml_inspection,
                "source": {
                    **source_inspection,
                    "class_count": len(class_names),
                    "axe_class_count": len(axe_classes),
                },
                "conversion": {
                    "active_source_images": len(active),
                    "output_images_after_exact_deduplication": len(units),
                    "output_axe_boxes": sum(len(unit.canonical.boxes) for unit in units),
                    "non_axe_annotations_removed_from_active_images": sum(
                        record.non_axe_removed for record in active
                    ),
                    "rejected_annotation_count": len(rejected),
                    "axis_aligned_rule": "min/max x/y over each normalized AXE polygon",
                    **deduplication,
                },
                "quarantine": {
                    "image_count": len(quarantine_rows),
                    "excluded_from_training_validation_test": True,
                    "raw_images_and_labels_preserved": True,
                },
                "family_heuristic": {
                    "description": (
                        "Case-fold the basename stem; remove only a terminal .rf.<hash>, then "
                        "terminal _jpg, then terminal _augN. Exact-image duplicates also link "
                        "their family groups before splitting."
                    ),
                    "remaining_uncertainty": (
                        "Filename families and exact byte hashes cannot prove independent capture "
                        "sessions or detect every near-duplicate/relabeled scene."
                    ),
                },
                "model2_provenance": {
                    **model2_report,
                    "matched_active_family_count": len(matched_assignments),
                    "matched_active_families_by_split": dict(
                        sorted(Counter(matched_assignments.values()).items())
                    ),
                    "validation_test_holdouts_forced": True,
                    "artifacts_rewritten": False,
                },
                "split_assignment": split_report,
                "splits": split_counts,
                "preview_count": len(preview_rows),
                "test_split_evaluated": False,
                "label_quality_limitation": (
                    "Metadata and geometry checks do not establish that every remaining visual "
                    "object is labeled correctly."
                ),
            }
            _write_json(staging / "dataset_report.json", report)
            _write_json(
                staging / "source_family_manifest.json",
                {
                    "seed": seed,
                    "ratios": SPLIT_RATIOS,
                    "groups": [
                        {
                            "group_id": group["group_id"],
                            "family_ids": group["family_ids"],
                            "output_split": group["output_split"],
                            "forced_split": group["forced_split"],
                            "image_count": group["image_count"],
                        }
                        for group in sorted(groups, key=lambda item: item["group_id"])
                    ],
                },
            )

        checks = validate_derived_dataset(staging)
        _write_json(staging / "dataset_checks.json", checks)
        staging.rename(output_dir)
        return report
    except Exception:
        if staging.exists() and staging.parent == output_dir.parent.resolve() and staging.name.startswith(
            f".{output_dir.name}_staging_"
        ):
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    if args.check_only:
        print(json.dumps(validate_derived_dataset(output_dir), ensure_ascii=False, indent=2))
        return
    glob_path = Path(args.source_zip_glob)
    if glob_path.is_absolute():
        matches = sorted(glob_path.parent.glob(glob_path.name))
    else:
        matches = sorted(ROOT.glob(args.source_zip_glob))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one version-8 source archive; refusing to combine versions: "
            f"{[str(path) for path in matches]}"
        )
    source_zip = matches[0]
    if ".v8-" not in source_zip.name or not source_zip.name.endswith(".yolov8.zip"):
        raise ValueError(f"Source archive is not the required v8 YOLO export: {source_zip}")
    report = prepare_dataset(source_zip, output_dir, args.seed, args.preview_count)
    concise = {
        "output_dir": str(output_dir),
        "source_images": report["source"]["image_count"],
        "source_families": report["source"]["family_count"],
        "source_axe_classes": report["source"]["axe_class_count"],
        "source_axe_polygons": report["source"]["source_axe_annotation_count"],
        "quarantined_images": report["quarantine"]["image_count"],
        "output_images": report["conversion"]["output_images_after_exact_deduplication"],
        "output_boxes": report["conversion"]["output_axe_boxes"],
        "rejected_annotations": report["conversion"]["rejected_annotation_count"],
        "splits": report["splits"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
