"""Audit overlapping brand ZIP snapshots and prepare a five-brand YOLO dataset.

The audit deliberately addresses ZIP members by ``ZipInfo`` rather than by
filename.  This matters because the supplied Roboflow archives contain repeated
``data.yaml`` paths, for which ``ZipFile.read(name)`` would silently select one
occurrence.

This script never trains or evaluates a model.  It writes an ordinary
axis-aligned YOLO detection dataset only after every archive-local YAML mapping
and every selected annotation has passed validation.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Sequence
from zipfile import ZipFile, ZipInfo

import yaml
from PIL import Image, ImageDraw, ImageOps

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
_BRAND_PROFILE = "brand_5brands"
TARGET_BRANDS = _BRAND_REGISTRY.names(_BRAND_PROFILE)
OUTPUT_CLASS_IDS = {name: index for index, name in enumerate(TARGET_BRANDS)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLITS = ("train", "valid", "test", "export")
OUTPUT_SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class Box:
    brand: str
    output_class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    source_class_id: int
    source_class_name: str
    source_line: int
    source_format: str
    was_clipped: bool = False

    def yolo_row(self) -> str:
        return (
            f"{self.output_class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )

    def signature(self) -> tuple[Any, ...]:
        return (
            self.output_class_id,
            round(self.x_center, 8),
            round(self.y_center, 8),
            round(self.width, 8),
            round(self.height, 8),
        )


@dataclass
class SourceRecord:
    source_kind: str
    source_id: str
    source_path: Path
    source_split: str
    image_member_index: int | None
    image_member: str
    label_member_indices: tuple[int, ...]
    label_member: str
    source_filename: str
    family_id: str
    image_sha256: str
    width: int
    height: int
    boxes: list[Box]
    source_annotation_count: int
    non_target_annotation_count: int
    archive_direct_brand_fraction: float

    def annotation_signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(box.signature() for box in self.boxes))


@dataclass
class ImageUnit:
    image_sha256: str
    canonical: SourceRecord
    sources: list[SourceRecord]
    family_ids: set[str]
    boxes: list[Box]
    component_id: str = ""
    forced_split: str | None = None
    force_reasons: set[str] = field(default_factory=set)
    output_split: str = ""
    output_name: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--zip-dir", default="data/brand_zips")
    parser.add_argument("--audit-dir", default="outputs/brand_zip_audit")
    parser.add_argument("--axe-dir", default="data/axe_yolo")
    parser.add_argument("--output-dir", default="data/brand_5brands_yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write the direct-ZIP audit but do not prepare the derived dataset.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing --output-dir without reading or changing ZIPs.",
    )
    return parser.parse_args(argv)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def infer_target_brand(class_name: str) -> str | None:
    """Map only an explicit generic brand or BRAND_* class prefix.

    Archive/folder names never participate in this decision.
    """

    return _BRAND_REGISTRY.infer_brand(class_name, _BRAND_PROFILE)


def annotation_format(declared_format: str | AnnotationFormat) -> str:
    """Return the legacy report label for an explicitly declared format."""

    selected = AnnotationFormat.parse(declared_format)
    labels = {
        AnnotationFormat.YOLO_BBOX: "ordinary_bbox",
        AnnotationFormat.YOLO_OBB: "obb_quadrilateral",
        AnnotationFormat.YOLO_SEGMENTATION: "segmentation_polygon",
    }
    if selected not in labels:
        raise ValueError(f"{selected.value} is not a YOLO text-row format")
    return labels[selected]


def coordinates_to_box(
    values: Sequence[float],
    brand: str,
    source_class_id: int,
    source_class_name: str,
    source_line: int,
    source_format: str | AnnotationFormat,
) -> Box:
    selected = AnnotationFormat.parse(source_format)
    fmt = annotation_format(selected)
    parsed = normalized_box_from_values(values, selected)
    return Box(
        brand=brand,
        output_class_id=OUTPUT_CLASS_IDS[brand],
        x_center=parsed.x_center,
        y_center=parsed.y_center,
        width=parsed.width,
        height=parsed.height,
        source_class_id=source_class_id,
        source_class_name=source_class_name,
        source_line=source_line,
        source_format=fmt,
        was_clipped=parsed.was_clipped,
    )


def parse_label_text(
    text: str,
    class_names: Sequence[str],
    source_label: str,
    source_format: str | AnnotationFormat,
) -> tuple[
    list[Box], Counter[int], Counter[str], Counter[str], list[dict[str, Any]]
]:
    boxes: list[Box] = []
    class_counts: Counter[int] = Counter()
    format_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    selected_format = AnnotationFormat.parse(source_format)
    format_label = annotation_format(selected_format)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except (ValueError, IndexError):
            errors.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "reason": "invalid class id",
                    "raw_line": raw_line,
                }
            )
            continue
        if not 0 <= class_id < len(class_names):
            errors.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "reason": f"class id {class_id} is outside YAML names",
                    "raw_line": raw_line,
                }
            )
            continue
        class_counts[class_id] += 1
        format_counts[format_label] += 1
        try:
            values = [float(value) for value in fields[1:]]
            if any(value < 0.0 or value > 1.0 for value in values):
                boundary_counts["source_rows_with_coordinate_outside_0_1"] += 1
            brand = infer_target_brand(class_names[class_id])
            if brand is not None:
                box = coordinates_to_box(
                        values,
                        brand,
                        class_id,
                        class_names[class_id],
                        line_number,
                        selected_format,
                )
                boxes.append(box)
                if box.was_clipped:
                    boundary_counts["target_boxes_clipped_to_image"] += 1
            else:
                # Validate non-target geometry too; excluded classes must not
                # hide a corrupt or format-incompatible source export.
                normalized_box_from_values(values, selected_format)
        except ValueError as error:
            errors.append(
                {
                    "source_label": source_label,
                    "line_number": line_number,
                    "reason": str(error),
                    "raw_line": raw_line,
                }
            )
    return boxes, class_counts, format_counts, boundary_counts, errors


def _member_kind(path: PurePosixPath) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".txt" and "labels" in path.parts:
        return "label"
    if path.name.lower() in {"data.yaml", "data.yml"}:
        return "yaml"
    return "other"


def audit_zip(
    zip_path: Path,
    yaml_copy_dir: Path,
    source_format: str | AnnotationFormat | None = None,
) -> tuple[dict[str, Any], list[SourceRecord]]:
    report: dict[str, Any] = {
        "archive": str(zip_path),
        "archive_name": zip_path.name,
        "archive_size": zip_path.stat().st_size,
        "archive_sha256": sha256_file(zip_path),
        "blocking_issues": [],
        "warnings": [],
    }
    records: list[SourceRecord] = []
    with ZipFile(zip_path) as archive:
        infos = archive.infolist()
        indexed_infos = list(enumerate(infos))
        by_name: dict[str, list[tuple[int, ZipInfo]]] = defaultdict(list)
        for index, info in indexed_infos:
            by_name[info.filename].append((index, info))

        yaml_entries: list[dict[str, Any]] = []
        parsed_yamls: list[tuple[dict[str, Any], list[str]]] = []
        for index, info in indexed_infos:
            if PurePosixPath(info.filename).name.lower() not in {"data.yaml", "data.yml"}:
                continue
            payload = read_info(archive, info)
            copy_name = (
                f"{zip_path.stem}__member_{index:05d}__"
                f"{PurePosixPath(info.filename).name}"
            )
            copy_path = yaml_copy_dir / copy_name
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            copy_path.write_bytes(payload)
            entry: dict[str, Any] = {
                "member_index": index,
                "member_path": info.filename,
                "sha256": sha256_bytes(payload),
                "saved_copy": str(copy_path),
            }
            try:
                data = yaml.safe_load(payload.decode("utf-8-sig")) or {}
                names = normalize_yaml_names(data.get("names"))
                declared_nc = data.get("nc")
                entry.update(
                    {
                        "declared_nc": declared_nc,
                        "effective_nc": len(names),
                        "names": names,
                        "split_paths": {
                            "train": data.get("train"),
                            "val": data.get("val"),
                            "valid": data.get("valid"),
                            "test": data.get("test"),
                        },
                    }
                )
                if declared_nc is not None and int(declared_nc) != len(names):
                    raise ValueError(
                        f"declared nc={declared_nc} differs from {len(names)} names"
                    )
                parsed_yamls.append((data, names))
            except Exception as error:
                entry["parse_error"] = str(error)
                report["blocking_issues"].append(
                    f"YAML member {index} ({info.filename}) is invalid: {error}"
                )
            yaml_entries.append(entry)
        report["yaml_entries"] = yaml_entries
        if not yaml_entries:
            report["blocking_issues"].append("archive contains no data.yaml/data.yml")
        if len(parsed_yamls) != len(yaml_entries) or not parsed_yamls:
            report["yaml_consistent_within_archive"] = False
            return report, records

        critical_signatures = {
            json.dumps(
                {
                    "declared_nc": data.get("nc"),
                    "names": names,
                    "train": data.get("train"),
                    "val": data.get("val"),
                    "valid": data.get("valid"),
                    "test": data.get("test"),
                    "annotation_format": data.get("annotation_format"),
                    "task": data.get("task"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for data, names in parsed_yamls
        }
        report["yaml_consistent_within_archive"] = len(critical_signatures) == 1
        report["yaml_payloads_byte_identical"] = len(
            {entry["sha256"] for entry in yaml_entries}
        ) == 1
        if len(critical_signatures) != 1:
            report["blocking_issues"].append(
                "repeated YAML entries disagree on nc, names/order, or split paths"
            )
            return report, records
        source_yaml, class_names = parsed_yamls[0]
        declared_formats = {
            AnnotationFormat.parse(raw)
            for data, _ in parsed_yamls
            for raw in (data.get("annotation_format") or data.get("task"),)
            if raw is not None
        }
        configured_format: AnnotationFormat | None = None
        configured_source = None
        try:
            configured_source = load_dataset_sources().match(zip_path)
        except ValueError:
            # Synthetic/custom sources may declare their format inside data.yaml
            # or pass it explicitly; undeclared real sources fail below.
            pass
        if configured_source is not None:
            if (
                configured_source.sha256
                and configured_source.sha256.casefold()
                != report["archive_sha256"].casefold()
            ):
                report["blocking_issues"].append(
                    "archive SHA-256 differs from configured source identity: "
                    f"expected {configured_source.sha256}, found {report['archive_sha256']}"
                )
                return report, records
            configured_format = configured_source.annotation_format
        explicit_format = (
            AnnotationFormat.parse(source_format) if source_format is not None else None
        )
        candidates = declared_formats | {
            value for value in (configured_format, explicit_format) if value is not None
        }
        if not candidates:
            report["blocking_issues"].append(
                "annotation format is undeclared; configure the source or add "
                "annotation_format/task to data.yaml"
            )
            return report, records
        if len(candidates) != 1:
            report["blocking_issues"].append(
                "annotation format declarations conflict: "
                + ", ".join(sorted(value.value for value in candidates))
            )
            return report, records
        selected_format = next(iter(candidates))
        if selected_format not in {
            AnnotationFormat.YOLO_BBOX,
            AnnotationFormat.YOLO_OBB,
            AnnotationFormat.YOLO_SEGMENTATION,
        }:
            report["blocking_issues"].append(
                f"unsupported YOLO source annotation format: {selected_format.value}"
            )
            return report, records
        report["annotation_format"] = selected_format.value
        if source_yaml.get("nc") is None:
            report["warnings"].append(
                "data.yaml omits nc; effective_nc is derived from the verified names list"
            )

        inventory: Counter[tuple[str, str]] = Counter()
        image_infos: list[tuple[int, ZipInfo]] = []
        label_infos: list[tuple[int, ZipInfo]] = []
        for index, info in indexed_infos:
            path = PurePosixPath(info.filename)
            split = path.parts[0] if path.parts and path.parts[0] in SOURCE_SPLITS else "other"
            kind = _member_kind(path)
            inventory[(split, kind)] += 1
            if kind == "image" and split in SOURCE_SPLITS:
                image_infos.append((index, info))
            elif kind == "label" and split in SOURCE_SPLITS:
                label_infos.append((index, info))

        duplicate_rows: list[dict[str, Any]] = []
        for member_path, occurrences in sorted(by_name.items()):
            if len(occurrences) < 2:
                continue
            kind = _member_kind(PurePosixPath(member_path))
            if kind not in {"image", "label", "yaml"}:
                continue
            hashes = [sha256_bytes(read_info(archive, info)) for _, info in occurrences]
            duplicate_rows.append(
                {
                    "member_path": member_path,
                    "kind": kind,
                    "occurrence_count": len(occurrences),
                    "member_indices": [index for index, _ in occurrences],
                    "payload_sha256": hashes,
                    "payloads_identical": len(set(hashes)) == 1,
                }
            )
        report["duplicate_member_paths"] = duplicate_rows

        labels_by_name: dict[str, list[tuple[int, ZipInfo]]] = defaultdict(list)
        for index, info in label_infos:
            labels_by_name[info.filename].append((index, info))

        all_class_counts: Counter[int] = Counter()
        all_format_counts: Counter[str] = Counter()
        all_boundary_counts: Counter[str] = Counter()
        format_samples: dict[str, dict[str, Any]] = {}
        invalid_rows: list[dict[str, Any]] = []
        missing_labels: list[dict[str, Any]] = []
        conflicting_duplicate_labels: list[dict[str, Any]] = []
        empty_label_count = 0
        selected_image_count = 0
        non_target_image_count = 0
        paired_label_paths: set[str] = set()

        parsed_label_cache: dict[
            tuple[int, ...],
            tuple[
                list[Box],
                Counter[int],
                Counter[str],
                Counter[str],
                list[dict[str, Any]],
                str,
            ],
        ] = {}
        preliminary: list[
            tuple[int, ZipInfo, str, tuple[int, ...], list[Box], int, int]
        ] = []
        for image_index, image_info in image_infos:
            expected_label = label_member_for(image_info.filename)
            occurrences = labels_by_name.get(expected_label, [])
            if not occurrences:
                missing_labels.append(
                    {
                        "image_member_index": image_index,
                        "image_member": image_info.filename,
                        "expected_label": expected_label,
                    }
                )
                continue
            paired_label_paths.add(expected_label)
            label_payloads = [read_info(archive, info) for _, info in occurrences]
            hashes = [sha256_bytes(payload) for payload in label_payloads]
            if len(set(hashes)) != 1:
                conflicting_duplicate_labels.append(
                    {
                        "image_member_index": image_index,
                        "image_member": image_info.filename,
                        "label_member": expected_label,
                        "label_member_indices": [index for index, _ in occurrences],
                        "label_sha256": hashes,
                    }
                )
                continue
            occurrence_indices = tuple(index for index, _ in occurrences)
            if occurrence_indices not in parsed_label_cache:
                text = label_payloads[0].decode("utf-8-sig")
                boxes, counts, formats, boundaries, errors = parse_label_text(
                    text, class_names, expected_label, selected_format
                )
                parsed_label_cache[occurrence_indices] = (
                    boxes,
                    counts,
                    formats,
                    boundaries,
                    errors,
                    text,
                )
            boxes, counts, formats, boundaries, errors, text = parsed_label_cache[
                occurrence_indices
            ]
            all_class_counts.update(counts)
            all_format_counts.update(formats)
            all_boundary_counts.update(boundaries)
            invalid_rows.extend(errors)
            if not counts:
                empty_label_count += 1
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                fields = raw_line.split()
                if not fields:
                    continue
                fmt = annotation_format(selected_format)
                format_samples.setdefault(
                    fmt,
                    {
                        "label_member_indices": list(occurrence_indices),
                        "label_member": expected_label,
                        "line_number": line_number,
                        "raw_line": raw_line,
                    },
                )
            annotation_count = sum(counts.values())
            non_target = annotation_count - len(boxes)
            preliminary.append(
                (
                    image_index,
                    image_info,
                    expected_label,
                    occurrence_indices,
                    list(boxes),
                    annotation_count,
                    non_target,
                )
            )
            if boxes:
                selected_image_count += 1
            else:
                non_target_image_count += 1

        orphan_labels = [
            {"member_index": index, "member_path": info.filename}
            for index, info in label_infos
            if info.filename not in paired_label_paths
        ]
        if missing_labels:
            report["blocking_issues"].append(
                f"{len(missing_labels)} image entries have no matching label entry"
            )
        if conflicting_duplicate_labels:
            report["blocking_issues"].append(
                f"{len(conflicting_duplicate_labels)} image entries have conflicting duplicate labels"
            )
        if invalid_rows:
            report["blocking_issues"].append(
                f"{len(invalid_rows)} annotation rows are malformed or invalid"
            )
        if all_boundary_counts["source_rows_with_coordinate_outside_0_1"]:
            report["warnings"].append(
                f"{all_boundary_counts['source_rows_with_coordinate_outside_0_1']} source rows "
                "cross an image boundary; selected target boxes are clipped explicitly"
            )

        target_annotation_count = sum(
            count
            for class_id, count in all_class_counts.items()
            if infer_target_brand(class_names[class_id]) is not None
        )
        direct_brand_annotation_count = sum(
            count
            for class_id, count in all_class_counts.items()
            if class_names[class_id].strip().upper() in OUTPUT_CLASS_IDS
        )
        direct_fraction = (
            direct_brand_annotation_count / target_annotation_count
            if target_annotation_count
            else 0.0
        )

        if not report["blocking_issues"]:
            for (
                image_index,
                image_info,
                expected_label,
                occurrence_indices,
                boxes,
                annotation_count,
                non_target,
            ) in preliminary:
                if not boxes:
                    continue
                image_bytes = read_info(archive, image_info)
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        width, height = image.size
                except Exception as error:
                    report["blocking_issues"].append(
                        f"unreadable image member {image_index} ({image_info.filename}): {error}"
                    )
                    continue
                if width <= 0 or height <= 0:
                    report["blocking_issues"].append(
                        f"image member {image_index} has an empty dimension"
                    )
                    continue
                records.append(
                    SourceRecord(
                        source_kind="zip",
                        source_id=zip_path.name,
                        source_path=zip_path,
                        source_split=PurePosixPath(image_info.filename).parts[0],
                        image_member_index=image_index,
                        image_member=image_info.filename,
                        label_member_indices=occurrence_indices,
                        label_member=expected_label,
                        source_filename=PurePosixPath(image_info.filename).name,
                        family_id=normalize_source_family(image_info.filename),
                        image_sha256=sha256_bytes(image_bytes),
                        width=width,
                        height=height,
                        boxes=boxes,
                        source_annotation_count=annotation_count,
                        non_target_annotation_count=non_target,
                        archive_direct_brand_fraction=direct_fraction,
                    )
                )

        used_classes = []
        for class_id, count in sorted(all_class_counts.items()):
            class_name = class_names[class_id]
            used_classes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "annotation_count": count,
                    "target_brand": infer_target_brand(class_name),
                    "output_class_id": (
                        OUTPUT_CLASS_IDS[infer_target_brand(class_name)]
                        if infer_target_brand(class_name) is not None
                        else None
                    ),
                    "mapping_basis": "verified YAML class name/prefix",
                }
            )

        report.update(
            {
                "member_count": len(infos),
                "inventory": {
                    split: {
                        kind: inventory[(split, kind)]
                        for kind in ("image", "label", "yaml", "other")
                    }
                    for split in (*SOURCE_SPLITS, "other")
                },
                "image_count": len(image_infos),
                "label_count": len(label_infos),
                "target_positive_image_count": selected_image_count,
                "zero_target_image_count": non_target_image_count,
                "empty_label_count": empty_label_count,
                "missing_labels": missing_labels,
                "orphan_labels": orphan_labels,
                "conflicting_duplicate_labels": conflicting_duplicate_labels,
                "invalid_annotation_rows": invalid_rows,
                "annotation_format_counts": dict(sorted(all_format_counts.items())),
                "annotation_boundary_counts": dict(sorted(all_boundary_counts.items())),
                "annotation_format_samples": format_samples,
                "used_classes": used_classes,
                "target_annotation_count": target_annotation_count,
                "direct_generic_brand_annotation_count": direct_brand_annotation_count,
                "direct_generic_brand_fraction": direct_fraction,
                "target_brand_annotation_counts": dict(
                    sorted(Counter(box.brand for record in records for box in record.boxes).items())
                ),
                "record_count_available_for_preparation": len(records),
            }
        )
    return report, records


def write_audit_reports(audit_dir: Path, reports: list[dict[str, Any]]) -> None:
    write_json(
        audit_dir / "zip_audit.json",
        {
            "status": (
                "blocked" if any(report["blocking_issues"] for report in reports) else "passed"
            ),
            "target_brands": list(TARGET_BRANDS),
            "mapping_rule": (
                "A class maps only when its verified YAML name is exactly a target brand or "
                "starts with that brand plus an underscore; ZIP/folder names are ignored."
            ),
            "archives": reports,
        },
    )
    yaml_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for report in reports:
        for entry in report["yaml_entries"]:
            yaml_rows.append(
                {
                    "archive_name": report["archive_name"],
                    "member_index": entry["member_index"],
                    "member_path": entry["member_path"],
                    "sha256": entry["sha256"],
                    "declared_nc": entry.get("declared_nc"),
                    "effective_nc": entry.get("effective_nc"),
                    "train": entry.get("split_paths", {}).get("train"),
                    "val": entry.get("split_paths", {}).get("val"),
                    "valid": entry.get("split_paths", {}).get("valid"),
                    "test": entry.get("split_paths", {}).get("test"),
                    "saved_copy": entry["saved_copy"],
                }
            )
        for row in report.get("used_classes", []):
            mapping_rows.append({"archive_name": report["archive_name"], **row})
        for row in report.get("duplicate_member_paths", []):
            duplicate_rows.append(
                {
                    "archive_name": report["archive_name"],
                    "member_path": row["member_path"],
                    "kind": row["kind"],
                    "occurrence_count": row["occurrence_count"],
                    "member_indices": json.dumps(row["member_indices"]),
                    "payload_sha256": json.dumps(row["payload_sha256"]),
                    "payloads_identical": row["payloads_identical"],
                }
            )
    write_csv(
        audit_dir / "yaml_entries.csv",
        (
            "archive_name",
            "member_index",
            "member_path",
            "sha256",
            "declared_nc",
            "effective_nc",
            "train",
            "val",
            "valid",
            "test",
            "saved_copy",
        ),
        yaml_rows,
    )
    write_csv(
        audit_dir / "class_mapping.csv",
        (
            "archive_name",
            "class_id",
            "class_name",
            "annotation_count",
            "target_brand",
            "output_class_id",
            "mapping_basis",
        ),
        mapping_rows,
    )
    write_csv(
        audit_dir / "duplicate_member_paths.csv",
        (
            "archive_name",
            "member_path",
            "kind",
            "occurrence_count",
            "member_indices",
            "payload_sha256",
            "payloads_identical",
        ),
        duplicate_rows,
    )
    lines = [
        "# Brand ZIP audit",
        "",
        f"Status: **{'BLOCKED' if any(r['blocking_issues'] for r in reports) else 'PASSED'}**",
        "",
        "Class mapping is derived from each archive-local YAML entry, never from the archive name.",
        "",
        "| Archive | SHA-256 | YAML entries | YAML local consistency | Images | Labels | Formats | Target-positive images |",
        "|---|---|---:|---|---:|---:|---|---:|",
    ]
    for report in reports:
        formats = ", ".join(
            f"{name}:{count}" for name, count in report.get("annotation_format_counts", {}).items()
        )
        lines.append(
            f"| {report['archive_name']} | `{report['archive_sha256']}` | "
            f"{len(report['yaml_entries'])} | {report.get('yaml_consistent_within_archive')} | "
            f"{report.get('image_count', 0)} | {report.get('label_count', 0)} | "
            f"{formats} | {report.get('target_positive_image_count', 0)} |"
        )
    lines.extend(["", "## Blocking issues", ""])
    issues = [
        f"- `{report['archive_name']}`: {issue}"
        for report in reports
        for issue in report["blocking_issues"]
    ]
    lines.extend(issues or ["- None."])
    lines.extend(
        [
            "",
            "## Important interpretation",
            "",
            "The archives are overlapping project snapshots with different class lists/orders. "
            "That cross-archive difference is not resolved by reusing one global index: every "
            "label is bound to its own verified archive YAML before remapping.",
            "",
        ]
    )
    (audit_dir / "zip_audit.md").write_text("\n".join(lines), encoding="utf-8")


def audit_archives(zip_dir: Path, audit_dir: Path) -> tuple[list[dict[str, Any]], list[SourceRecord]]:
    if audit_dir.exists():
        raise FileExistsError(f"Audit output already exists; refusing to overwrite: {audit_dir}")
    zip_paths = sorted(zip_dir.glob("*.zip"))
    if len(zip_paths) != 4:
        raise ValueError(f"Expected exactly four source ZIPs, found {len(zip_paths)}: {zip_paths}")
    audit_dir.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    records: list[SourceRecord] = []
    try:
        for zip_path in zip_paths:
            report, archive_records = audit_zip(zip_path, audit_dir / "yaml_copies")
            reports.append(report)
            records.extend(archive_records)
        write_audit_reports(audit_dir, reports)
    except Exception:
        # Keep a complete audit when validation itself completes with issues.  Only
        # unexpected execution failures remove the partial directory.
        if not (audit_dir / "zip_audit.json").is_file():
            shutil.rmtree(audit_dir, ignore_errors=True)
        raise
    return reports, records


def choose_archive_precedence(reports: Sequence[dict[str, Any]]) -> list[str]:
    eligible = [
        report
        for report in reports
        if not report["blocking_issues"]
        and all(
            report.get("target_brand_annotation_counts", {}).get(brand, 0) > 0
            for brand in TARGET_BRANDS
        )
    ]
    if not eligible:
        raise ValueError("No audited archive has verified annotation coverage for all five brands")
    ordered = sorted(
        eligible,
        key=lambda report: (
            -float(report["direct_generic_brand_fraction"]),
            -int(report["target_positive_image_count"]),
            report["archive_sha256"],
        ),
    )
    if len(ordered) > 1 and math.isclose(
        float(ordered[0]["direct_generic_brand_fraction"]),
        float(ordered[1]["direct_generic_brand_fraction"]),
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Archive precedence is ambiguous: top archives have the same direct-brand-label fraction"
        )
    if float(ordered[0]["direct_generic_brand_fraction"]) < 0.95:
        raise ValueError(
            "No archive has a sufficiently direct brand-level taxonomy (required >= 0.95)"
        )
    return [report["archive_name"] for report in ordered]


def deduplicate_zip_records(
    records: Sequence[SourceRecord],
    precedence: Sequence[str],
) -> tuple[list[ImageUnit], list[dict[str, Any]], dict[str, Any]]:
    rank = {archive: index for index, archive in enumerate(precedence)}
    by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.image_sha256].append(record)
    units: list[ImageUnit] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for image_hash, members in sorted(by_hash.items()):
        signatures = {member.annotation_signature() for member in members}
        ordered = sorted(
            members,
            key=lambda member: (
                rank.get(member.source_id, len(rank)),
                member.source_id.casefold(),
                member.image_member_index if member.image_member_index is not None else -1,
            ),
        )
        selected = ordered[0]
        same_rank = [
            member
            for member in ordered
            if rank.get(member.source_id, len(rank))
            == rank.get(selected.source_id, len(rank))
        ]
        if len({member.annotation_signature() for member in same_rank}) > 1:
            unresolved.append(
                {
                    "image_sha256": image_hash,
                    "reason": "conflicting annotations inside the same highest-ranked archive",
                    "sources": [
                        f"{member.source_id}#{member.image_member_index}:{member.image_member}"
                        for member in same_rank
                    ],
                }
            )
            continue
        if len(signatures) > 1:
            conflicts.append(
                {
                    "image_sha256": image_hash,
                    "selected_archive": selected.source_id,
                    "selected_member_index": selected.image_member_index,
                    "selected_member": selected.image_member,
                    "selected_box_count": len(selected.boxes),
                    "resolution": (
                        "selected the audited archive with the highest global fraction of direct "
                        "generic-brand annotations; archive name/version was not used"
                    ),
                    "discarded_sources": json.dumps(
                        [
                            {
                                "archive": member.source_id,
                                "member_index": member.image_member_index,
                                "member": member.image_member,
                                "box_count": len(member.boxes),
                            }
                            for member in ordered[1:]
                            if member.annotation_signature() != selected.annotation_signature()
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
        units.append(
            ImageUnit(
                image_sha256=image_hash,
                canonical=selected,
                sources=ordered,
                family_ids={member.family_id for member in members},
                boxes=list(selected.boxes),
            )
        )
    if unresolved:
        raise ValueError(
            "Conflicting exact-duplicate annotations could not be resolved without guessing: "
            f"{unresolved[:5]}"
        )
    return units, conflicts, {
        "source_positive_records": len(records),
        "unique_image_hashes": len(by_hash),
        "exact_duplicate_hash_groups": sum(len(group) > 1 for group in by_hash.values()),
        "exact_duplicate_records_removed": sum(len(group) - 1 for group in by_hash.values()),
        "conflicting_annotation_groups_resolved": len(conflicts),
        "unresolved_conflicting_annotation_groups": 0,
        "archive_precedence": list(precedence),
    }


def _local_image_path(dataset_dir: Path, row: dict[str, str]) -> Path:
    value = row.get("output_image_path")
    if value:
        return dataset_dir / Path(value.replace("/", "\\"))
    raise ValueError("AXE provenance row has no output_image_path")


def load_existing_axe_records(axe_dir: Path) -> tuple[list[SourceRecord], dict[str, str], dict[str, Any]]:
    required = (
        axe_dir / "mapping.json",
        axe_dir / "dataset_checks.json",
        axe_dir / "provenance.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Existing AXE dataset metadata is incomplete: {missing}")
    mapping = json.loads((axe_dir / "mapping.json").read_text(encoding="utf-8"))
    checks = json.loads((axe_dir / "dataset_checks.json").read_text(encoding="utf-8"))
    if mapping.get("output_names") != {"0": "AXE"}:
        raise ValueError("Existing AXE dataset mapping is not verified class 0 = AXE")
    if checks.get("status") != "passed" or checks.get("model2_holdout_violations") != 0:
        raise ValueError(f"Existing AXE dataset checks did not pass: {checks}")

    records: list[SourceRecord] = []
    holdout_by_hash: dict[str, str] = {}
    with (axe_dir / "provenance.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = _local_image_path(axe_dir, row)
        label_path = axe_dir / Path(row["output_label_path"].replace("/", "\\"))
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Existing AXE image/label is missing: {image_path}, {label_path}")
        image_bytes = image_path.read_bytes()
        image_hash = sha256_bytes(image_bytes)
        if image_hash != row["image_sha256"]:
            raise ValueError(f"Existing AXE provenance hash mismatch: {image_path}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        boxes: list[Box] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Invalid existing AXE detection row: {label_path}:{line_number}")
            values = [float(value) for value in fields[1:]]
            boxes.append(
                coordinates_to_box(
                    values,
                    "AXE",
                    0,
                    "AXE",
                    line_number,
                    AnnotationFormat.YOLO_BBOX,
                )
            )
        if not boxes:
            raise ValueError(f"Existing AXE record has no verified boxes: {label_path}")
        family_ids = json.loads(row["family_ids"])
        family_id = str(row.get("family_id") or family_ids[0]).casefold()
        split = row["output_split"]
        if split in {"val", "test"}:
            previous = holdout_by_hash.setdefault(image_hash, split)
            if previous != split:
                raise ValueError(f"Conflicting AXE holdout split for image {image_hash}")
        records.append(
            SourceRecord(
                source_kind="existing_axe",
                source_id="data/axe_yolo",
                source_path=image_path,
                source_split=split,
                image_member_index=None,
                image_member=str(image_path.relative_to(axe_dir)).replace("\\", "/"),
                label_member_indices=(),
                label_member=str(label_path.relative_to(axe_dir)).replace("\\", "/"),
                source_filename=image_path.name,
                family_id=family_id,
                image_sha256=image_hash,
                width=width,
                height=height,
                boxes=boxes,
                source_annotation_count=len(boxes),
                non_target_annotation_count=0,
                archive_direct_brand_fraction=1.0,
            )
        )
    return records, holdout_by_hash, {
        "path": str(axe_dir),
        "record_count": len(records),
        "box_count": sum(len(record.boxes) for record in records),
        "validation_holdout_count": sum(value == "val" for value in holdout_by_hash.values()),
        "test_holdout_count": sum(value == "test" for value in holdout_by_hash.values()),
        "mapping_verified": True,
        "checks_verified": True,
    }


def merge_existing_axe(
    units: list[ImageUnit],
    axe_records: Sequence[SourceRecord],
    holdout_by_hash: dict[str, str],
) -> dict[str, Any]:
    by_hash = {unit.image_sha256: unit for unit in units}
    replacements = 0
    identical = 0
    added = 0
    for axe in axe_records:
        unit = by_hash.get(axe.image_sha256)
        if unit is None:
            unit = ImageUnit(
                image_sha256=axe.image_sha256,
                canonical=axe,
                sources=[axe],
                family_ids={axe.family_id},
                boxes=list(axe.boxes),
            )
            units.append(unit)
            by_hash[axe.image_sha256] = unit
            added += 1
        else:
            old_axe = sorted(box.signature() for box in unit.boxes if box.brand == "AXE")
            new_axe = sorted(box.signature() for box in axe.boxes)
            if old_axe == new_axe:
                identical += 1
            else:
                replacements += 1
            unit.boxes = [box for box in unit.boxes if box.brand != "AXE"] + list(axe.boxes)
            unit.sources.append(axe)
            unit.family_ids.add(axe.family_id)
        if axe.image_sha256 in holdout_by_hash:
            unit.forced_split = holdout_by_hash[axe.image_sha256]
            unit.force_reasons.add("existing AXE validation/test holdout by exact image hash")
    units.sort(key=lambda unit: unit.image_sha256)
    return {
        "unique_existing_axe_images_added": added,
        "overlapping_images_with_identical_axe_boxes": identical,
        "overlapping_images_where_verified_existing_axe_boxes_replaced_zip_axe_boxes": replacements,
        "replacement_rule": (
            "For an exact image already present in verified data/axe_yolo, preserve its reviewed AXE "
            "boxes and retain selected ZIP boxes for the other four target brands."
        ),
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
        if not families:
            raise ValueError(f"Image unit has no source family: {unit.image_sha256}")
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)
    grouped: dict[str, list[ImageUnit]] = defaultdict(list)
    for unit in units:
        component = min(find(family) for family in unit.family_ids)
        unit.component_id = component
        grouped[component].append(unit)
    groups: list[dict[str, Any]] = []
    for component, members in sorted(grouped.items()):
        forced = {unit.forced_split for unit in members if unit.forced_split}
        if len(forced) > 1:
            raise ValueError(
                f"One family component links both validation and test holdouts: {component}: {forced}"
            )
        groups.append(
            {
                "group_id": component,
                "units": sorted(members, key=lambda unit: unit.image_sha256),
                "image_count": len(members),
                "family_ids": sorted(
                    {family for unit in members for family in unit.family_ids}
                ),
                "brand_box_counts": dict(
                    Counter(box.brand for unit in members for box in unit.boxes)
                ),
                "forced_split": next(iter(forced)) if forced else None,
                "force_reasons": sorted(
                    {reason for unit in members for reason in unit.force_reasons}
                ),
            }
        )
    return groups


def apply_model2_holdouts(
    groups: list[dict[str, Any]], model2_assignments: dict[str, str]
) -> dict[str, Any]:
    matched_families: Counter[str] = Counter()
    newly_forced = 0
    for group in groups:
        known = {
            model2_assignments[family]
            for family in group["family_ids"]
            if family in model2_assignments
        }
        holdouts = known & {"val", "test"}
        if len(holdouts) > 1:
            raise ValueError(
                "Grouped families link conflicting Model 2 validation/test holdouts: "
                f"{group['family_ids']}"
            )
        if holdouts:
            holdout = next(iter(holdouts))
            if group["forced_split"] and group["forced_split"] != holdout:
                raise ValueError(
                    f"Exact AXE holdout conflicts with Model 2 family holdout for {group['group_id']}"
                )
            if not group["forced_split"]:
                newly_forced += group["image_count"]
            group["forced_split"] = holdout
            group["force_reasons"] = sorted(
                set(group["force_reasons"]) | {"Model 2 family validation/test holdout"}
            )
        for split in known:
            matched_families[split] += 1
    return {
        "matched_family_assignments": dict(sorted(matched_families.items())),
        "newly_forced_image_count": newly_forced,
    }


def _split_objective(
    image_counts: Counter[str],
    brand_counts: dict[str, Counter[str]],
    image_targets: dict[str, float],
    brand_targets: dict[str, dict[str, float]],
    total_images: int,
    total_brand_counts: Counter[str],
) -> float:
    value = sum(
        ((image_counts[split] - image_targets[split]) / max(total_images, 1)) ** 2
        for split in OUTPUT_SPLITS
    )
    value += sum(
        (
            (brand_counts[split][brand] - brand_targets[split][brand])
            / max(total_brand_counts[brand], 1)
        )
        ** 2
        for split in OUTPUT_SPLITS
        for brand in TARGET_BRANDS
    )
    return value


def assign_splits(groups: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    total_images = sum(group["image_count"] for group in groups)
    total_brand_counts: Counter[str] = Counter()
    for group in groups:
        total_brand_counts.update(group["brand_box_counts"])
    image_targets = {
        split: total_images * SPLIT_RATIOS[split] for split in OUTPUT_SPLITS
    }
    brand_targets = {
        split: {
            brand: total_brand_counts[brand] * SPLIT_RATIOS[split]
            for brand in TARGET_BRANDS
        }
        for split in OUTPUT_SPLITS
    }
    image_counts: Counter[str] = Counter()
    brand_counts = {split: Counter() for split in OUTPUT_SPLITS}
    for group in groups:
        if group["forced_split"]:
            split = group["forced_split"]
            group["output_split"] = split
            image_counts[split] += group["image_count"]
            brand_counts[split].update(group["brand_box_counts"])

    rng = random.Random(seed)
    ties = {
        (group["group_id"], split): rng.random()
        for group in groups
        for split in OUTPUT_SPLITS
    }
    unassigned = [group for group in groups if not group["forced_split"]]
    unassigned.sort(
        key=lambda group: (
            -sum(group["brand_box_counts"].values()),
            -group["image_count"],
            group["group_id"],
        )
    )
    for group in unassigned:
        candidates: list[tuple[float, float, str]] = []
        for split in OUTPUT_SPLITS:
            image_counts[split] += group["image_count"]
            brand_counts[split].update(group["brand_box_counts"])
            objective = _split_objective(
                image_counts,
                brand_counts,
                image_targets,
                brand_targets,
                total_images,
                total_brand_counts,
            )
            candidates.append((objective, ties[(group["group_id"], split)], split))
            image_counts[split] -= group["image_count"]
            brand_counts[split].subtract(group["brand_box_counts"])
        split = min(candidates)[2]
        group["output_split"] = split
        image_counts[split] += group["image_count"]
        brand_counts[split].update(group["brand_box_counts"])

    # Greedy placement can front-load dense shelf images into train.  Improve
    # the same objective by moving whole, non-holdout family components only.
    # No individual image or augmentation relative can cross a split boundary.
    improvement_moves = 0
    optimization_passes = 0
    for pass_index in range(50):
        moved_this_pass = 0
        for group in unassigned:
            old_split = group["output_split"]
            current = _split_objective(
                image_counts,
                brand_counts,
                image_targets,
                brand_targets,
                total_images,
                total_brand_counts,
            )
            best = (current, old_split)
            for new_split in OUTPUT_SPLITS:
                if new_split == old_split:
                    continue
                if image_counts[old_split] - group["image_count"] <= 0:
                    continue
                if any(
                    brand_counts[old_split][brand]
                    - int(group["brand_box_counts"].get(brand, 0))
                    <= 0
                    for brand in TARGET_BRANDS
                    if int(group["brand_box_counts"].get(brand, 0)) > 0
                ):
                    continue
                image_counts[old_split] -= group["image_count"]
                brand_counts[old_split].subtract(group["brand_box_counts"])
                image_counts[new_split] += group["image_count"]
                brand_counts[new_split].update(group["brand_box_counts"])
                candidate = _split_objective(
                    image_counts,
                    brand_counts,
                    image_targets,
                    brand_targets,
                    total_images,
                    total_brand_counts,
                )
                image_counts[new_split] -= group["image_count"]
                brand_counts[new_split].subtract(group["brand_box_counts"])
                image_counts[old_split] += group["image_count"]
                brand_counts[old_split].update(group["brand_box_counts"])
                if candidate < best[0] - 1e-15:
                    best = (candidate, new_split)
            if best[1] != old_split:
                new_split = best[1]
                image_counts[old_split] -= group["image_count"]
                brand_counts[old_split].subtract(group["brand_box_counts"])
                image_counts[new_split] += group["image_count"]
                brand_counts[new_split].update(group["brand_box_counts"])
                group["output_split"] = new_split
                improvement_moves += 1
                moved_this_pass += 1
        optimization_passes = pass_index + 1
        if moved_this_pass == 0:
            break

    missing = [
        (split, brand)
        for split in OUTPUT_SPLITS
        for brand in TARGET_BRANDS
        if brand_counts[split][brand] <= 0
    ]
    if missing:
        raise ValueError(f"Split assignment lacks target brand coverage: {missing}")
    for group in groups:
        for unit in group["units"]:
            unit.output_split = group["output_split"]
            unit.forced_split = group["forced_split"]
            unit.force_reasons.update(group["force_reasons"])
    return {
        "seed": seed,
        "target_ratios": SPLIT_RATIOS,
        "target_image_counts": image_targets,
        "actual_image_counts": dict(image_counts),
        "actual_brand_box_counts": {
            split: {brand: brand_counts[split][brand] for brand in TARGET_BRANDS}
            for split in OUTPUT_SPLITS
        },
        "local_optimization_passes": optimization_passes,
        "local_improvement_moves": improvement_moves,
        "final_split_objective": _split_objective(
            image_counts,
            brand_counts,
            image_targets,
            brand_targets,
            total_images,
            total_brand_counts,
        ),
        "forced_holdout_image_counts": {
            split: sum(
                group["image_count"]
                for group in groups
                if group["forced_split"] == split
            )
            for split in ("val", "test")
        },
        "forced_val_images": sum(
            group["image_count"] for group in groups if group["forced_split"] == "val"
        ),
        "forced_test_images": sum(
            group["image_count"] for group in groups if group["forced_split"] == "test"
        ),
    }


def read_record_image(record: SourceRecord, archives: dict[Path, ZipFile]) -> bytes:
    if record.source_kind == "existing_axe":
        return record.source_path.read_bytes()
    archive = archives[record.source_path]
    if record.image_member_index is None:
        raise ValueError("ZIP record is missing its member index")
    infos = archive.infolist()
    if not 0 <= record.image_member_index < len(infos):
        raise IndexError(f"ZIP member index is outside archive: {record.image_member_index}")
    info = infos[record.image_member_index]
    if info.filename != record.image_member:
        raise ValueError(
            "ZIP member index/path changed since audit: "
            f"{record.image_member_index}: {info.filename} != {record.image_member}"
        )
    return read_info(archive, info)


def assign_output_names(units: Sequence[ImageUnit]) -> None:
    for index, unit in enumerate(sorted(units, key=lambda item: item.image_sha256), start=1):
        suffix = PurePosixPath(unit.canonical.image_member).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        unit.output_name = f"b{index:05d}{suffix}"


def write_dataset_files(
    staging: Path,
    output_dir: Path,
    units: Sequence[ImageUnit],
) -> list[dict[str, Any]]:
    archives: dict[Path, ZipFile] = {}
    provenance: list[dict[str, Any]] = []
    try:
        for unit in units:
            if unit.canonical.source_kind == "zip":
                archives.setdefault(unit.canonical.source_path, ZipFile(unit.canonical.source_path))
            image_bytes = read_record_image(unit.canonical, archives)
            if sha256_bytes(image_bytes) != unit.image_sha256:
                raise ValueError(f"Source image changed after audit: {unit.canonical.image_member}")
            image_target = staging / "images" / unit.output_split / unit.output_name
            label_target = (
                staging
                / "labels"
                / unit.output_split
                / f"{Path(unit.output_name).stem}.txt"
            )
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            image_target.write_bytes(image_bytes)
            label_target.write_text(
                "\n".join(box.yolo_row() for box in sorted(unit.boxes, key=Box.signature)) + "\n",
                encoding="utf-8",
            )
            provenance.append(
                {
                    "output_split": unit.output_split,
                    "output_image_path": str(image_target.relative_to(staging)).replace("\\", "/"),
                    "output_label_path": str(label_target.relative_to(staging)).replace("\\", "/"),
                    "image_sha256": unit.image_sha256,
                    "family_component_id": unit.component_id,
                    "family_ids": json.dumps(sorted(unit.family_ids), ensure_ascii=False),
                    "forced_holdout": unit.forced_split or "",
                    "force_reasons": json.dumps(sorted(unit.force_reasons), ensure_ascii=False),
                    "selected_source_kind": unit.canonical.source_kind,
                    "selected_source_id": unit.canonical.source_id,
                    "selected_source_split": unit.canonical.source_split,
                    "selected_image_member_index": (
                        unit.canonical.image_member_index
                        if unit.canonical.image_member_index is not None
                        else ""
                    ),
                    "selected_image_member": unit.canonical.image_member,
                    "selected_label_member_indices": json.dumps(
                        unit.canonical.label_member_indices
                    ),
                    "selected_label_member": unit.canonical.label_member,
                    "duplicate_source_record_count": len(unit.sources),
                    "source_records": json.dumps(
                        [
                            {
                                "source_kind": source.source_kind,
                                "source_id": source.source_id,
                                "image_member_index": source.image_member_index,
                                "image_member": source.image_member,
                            }
                            for source in unit.sources
                        ],
                        ensure_ascii=False,
                    ),
                    "width": unit.canonical.width,
                    "height": unit.canonical.height,
                    "box_count": len(unit.boxes),
                    "brand_box_counts": json.dumps(
                        dict(sorted(Counter(box.brand for box in unit.boxes).items())),
                        ensure_ascii=False,
                    ),
                    "non_target_annotations_removed_from_selected_source": (
                        unit.canonical.non_target_annotation_count
                    ),
                }
            )
    finally:
        for archive in archives.values():
            archive.close()
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
            "selected_source_kind",
            "selected_source_id",
            "selected_source_split",
            "selected_image_member_index",
            "selected_image_member",
            "selected_label_member_indices",
            "selected_label_member",
            "duplicate_source_record_count",
            "source_records",
            "width",
            "height",
            "box_count",
            "brand_box_counts",
            "non_target_annotations_removed_from_selected_source",
        ),
        provenance,
    )
    data_yaml = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: brand for index, brand in enumerate(TARGET_BRANDS)},
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return provenance


def write_previews(output_dir: Path, preview_count: int) -> dict[str, Any]:
    candidates: list[tuple[str, Path, Path]] = []
    for split in OUTPUT_SPLITS:
        for label_path in sorted((output_dir / "labels" / split).glob("*.txt")):
            image_matches = [
                path
                for path in (output_dir / "images" / split).glob(f"{label_path.stem}.*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(image_matches) == 1:
                candidates.append((split, image_matches[0], label_path))
    selected: list[tuple[str, Path, Path]] = []
    covered: set[str] = set()
    for candidate in candidates:
        rows = candidate[2].read_text(encoding="utf-8").splitlines()
        brands = {TARGET_BRANDS[int(row.split()[0])] for row in rows if row.strip()}
        if brands - covered:
            selected.append(candidate)
            covered.update(brands)
        if len(selected) >= preview_count:
            break
    for candidate in candidates:
        if candidate not in selected and len(selected) < preview_count:
            selected.append(candidate)
    colors = {
        "AXE": "red",
        "VIM": "blue",
        "LIFEBUOY": "green",
        "HAZELINE": "orange",
        "REXONA": "purple",
    }
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[tuple[Image.Image, str]] = []
    rows_out: list[dict[str, Any]] = []
    for index, (split, image_path, label_path) in enumerate(selected, start=1):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        brand_counts: Counter[str] = Counter()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            class_id = int(fields[0])
            x_center, y_center, box_width, box_height = map(float, fields[1:])
            x0 = (x_center - box_width / 2.0) * width
            y0 = (y_center - box_height / 2.0) * height
            x1 = (x_center + box_width / 2.0) * width
            y1 = (y_center + box_height / 2.0) * height
            brand = TARGET_BRANDS[class_id]
            brand_counts[brand] += 1
            draw.rectangle((x0, y0, x1, y1), outline=colors[brand], width=max(2, width // 400))
            draw.text((x0 + 2, y0 + 2), brand, fill=colors[brand])
        target = preview_dir / f"preview_{index:03d}.jpg"
        image.save(target, quality=92)
        thumbs.append((ImageOps.contain(image, (320, 240)), f"{split}/{image_path.name}"))
        rows_out.append(
            {
                "split": split,
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "preview": str(target.relative_to(output_dir)).replace("\\", "/"),
                "brand_box_counts": json.dumps(dict(brand_counts), ensure_ascii=False),
            }
        )
    columns = 4
    cell_width, cell_height = 340, 280
    sheet = Image.new(
        "RGB",
        (columns * cell_width, max(1, math.ceil(len(thumbs) / columns)) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, title) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 25))
        draw.text((x + 4, y + 4), title[:52], fill="black")
    sheet.save(preview_dir / "contact_sheet.jpg", quality=92)
    write_csv(
        preview_dir / "preview_manifest.csv",
        ("split", "image", "preview", "brand_box_counts"),
        rows_out,
    )
    return {"preview_count": len(rows_out), "brands_covered": sorted(covered)}


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    required = (
        output_dir / "data.yaml",
        output_dir / "mapping.json",
        output_dir / "dataset_report.json",
        output_dir / "provenance.csv",
        output_dir / "source_family_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset is incomplete: {missing}")
    mapping = json.loads((output_dir / "mapping.json").read_text(encoding="utf-8"))
    expected_mapping = {str(index): brand for index, brand in enumerate(TARGET_BRANDS)}
    if mapping.get("output_names") != expected_mapping:
        raise ValueError(f"Unexpected five-brand mapping: {mapping.get('output_names')}")
    data = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    normalized_names = normalize_yaml_names(data.get("names"))
    if normalized_names != list(TARGET_BRANDS):
        raise ValueError(f"data.yaml class order differs from mapping: {normalized_names}")

    split_counts: dict[str, Any] = {}
    hashes_by_split: dict[str, set[str]] = {}
    brand_counts_by_split: dict[str, Counter[str]] = {}
    for split in OUTPUT_SPLITS:
        images = [
            path
            for path in (output_dir / "images" / split).glob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        labels = list((output_dir / "labels" / split).glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Image/label stems differ in {split}")
        hashes: set[str] = set()
        brand_counts: Counter[str] = Counter()
        for image_path in images:
            image_hash = sha256_file(image_path)
            if image_hash in hashes:
                raise ValueError(f"Duplicate image content inside {split}: {image_path}")
            hashes.add(image_hash)
        for label_path in labels:
            rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
            if not rows:
                raise ValueError(f"Prepared target-positive image has an empty label: {label_path}")
            signatures: set[tuple[Any, ...]] = set()
            for line_number, line in enumerate(rows, start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Non-detection row: {label_path}:{line_number}")
                class_id = int(fields[0])
                if not 0 <= class_id < len(TARGET_BRANDS):
                    raise ValueError(f"Class outside mapping: {label_path}:{line_number}")
                values = [float(value) for value in fields[1:]]
                if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                    raise ValueError(f"Invalid normalized box: {label_path}:{line_number}")
                if values[2] <= 0.0 or values[3] <= 0.0:
                    raise ValueError(f"Degenerate box: {label_path}:{line_number}")
                signature = (class_id, *[round(value, 8) for value in values])
                if signature in signatures:
                    raise ValueError(f"Duplicate annotation row: {label_path}:{line_number}")
                signatures.add(signature)
                brand_counts[TARGET_BRANDS[class_id]] += 1
        hashes_by_split[split] = hashes
        brand_counts_by_split[split] = brand_counts
        split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": sum(brand_counts.values()),
            "brand_boxes": {brand: brand_counts[brand] for brand in TARGET_BRANDS},
        }
    overlaps = {
        f"{first}-{second}": len(hashes_by_split[first] & hashes_by_split[second])
        for first_index, first in enumerate(OUTPUT_SPLITS)
        for second in OUTPUT_SPLITS[first_index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"Exact image content crosses output splits: {overlaps}")
    missing_brand_coverage = [
        (split, brand)
        for split in OUTPUT_SPLITS
        for brand in TARGET_BRANDS
        if brand_counts_by_split[split][brand] <= 0
    ]
    if missing_brand_coverage:
        raise ValueError(f"Brand is absent from output split: {missing_brand_coverage}")

    family_splits: dict[str, set[str]] = defaultdict(set)
    holdout_violations: list[dict[str, str]] = []
    with (output_dir / "provenance.csv").open("r", encoding="utf-8", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    for row in provenance:
        for family in json.loads(row["family_ids"]):
            family_splits[family].add(row["output_split"])
        if row["forced_holdout"] and row["forced_holdout"] != row["output_split"]:
            holdout_violations.append(row)
    leaking = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    if leaking:
        raise ValueError(f"Source families cross output splits: {dict(list(leaking.items())[:10])}")
    if holdout_violations:
        raise ValueError(f"Holdout assignments were violated: {holdout_violations[:5]}")
    if len(provenance) != sum(item["images"] for item in split_counts.values()):
        raise ValueError("Provenance row count differs from output image count")
    return {
        "status": "passed",
        "ordinary_yolo_detection_rows_only": True,
        "mapping_verified": expected_mapping,
        "splits": split_counts,
        "exact_image_cross_split_overlap": overlaps,
        "cross_split_family_overlap": 0,
        "model2_holdout_violations": 0,
        "all_five_brands_present_in_every_split": True,
        "test_split_evaluated": False,
    }


def prepare_dataset(
    audit_dir: Path,
    output_dir: Path,
    axe_dir: Path,
    reports: Sequence[dict[str, Any]],
    zip_records: Sequence[SourceRecord],
    seed: int,
    preview_count: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    blockers = [
        (report["archive_name"], issue)
        for report in reports
        for issue in report["blocking_issues"]
    ]
    if blockers:
        raise ValueError(f"ZIP audit is blocked; dataset will not be prepared: {blockers[:10]}")

    precedence = choose_archive_precedence(reports)
    units, annotation_conflicts, dedup_report = deduplicate_zip_records(
        zip_records, precedence
    )
    axe_records, axe_holdouts, axe_report = load_existing_axe_records(axe_dir)
    axe_merge_report = merge_existing_axe(units, axe_records, axe_holdouts)
    if any(not unit.boxes for unit in units):
        raise ValueError("A selected output image has no target-brand annotations")

    groups = group_units_by_family(units)
    model2_assignments, model2_source_report = load_model2_family_assignments(ROOT)
    model2_match_report = apply_model2_holdouts(groups, model2_assignments)
    split_report = assign_splits(groups, seed)
    assign_output_names(units)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_staging_", dir=output_dir.parent)
    ).resolve()
    try:
        provenance = write_dataset_files(staging, output_dir, units)
        write_csv(
            staging / "conflicting_annotations_resolved.csv",
            (
                "image_sha256",
                "selected_archive",
                "selected_member_index",
                "selected_member",
                "selected_box_count",
                "resolution",
                "discarded_sources",
            ),
            annotation_conflicts,
        )
        uncertain_groups = [
            {
                "group_id": group["group_id"],
                "image_count": group["image_count"],
                "family_ids": json.dumps(group["family_ids"], ensure_ascii=False),
                "reason": (
                    "grouped by conservative filename normalization; exact hashes alone do not "
                    "prove that all augmentation/capture relatives were found"
                ),
            }
            for group in groups
            if group["image_count"] > 1
        ]
        write_csv(
            staging / "uncertain_family_review.csv",
            ("group_id", "image_count", "family_ids", "reason"),
            uncertain_groups,
        )
        mapping = {
            "output_names": {str(index): brand for index, brand in enumerate(TARGET_BRANDS)},
            "class_order_source": "explicit order requested by the user",
            "source_mapping_rule": (
                "Use each archive-local verified YAML. Map an exact generic target brand or a "
                "case-insensitive BRAND_* prefix to that brand; do not use ZIP/folder names."
            ),
            "archive_precedence": dedup_report["archive_precedence"],
            "archive_precedence_rule": (
                "For an exact duplicate image with different annotations, prefer the audited "
                "snapshot with the highest global fraction of direct generic-brand annotations; "
                "ties are blocked."
            ),
            "axis_aligned_conversion": {
                "ordinary_bbox": "preserved as x_center, y_center, width, height",
                "obb_or_polygon": "xmin=min(x_i), xmax=max(x_i), ymin=min(y_i), ymax=max(y_i)",
                "image_boundary": (
                    "clip xmin/ymin to >= 0 and xmax/ymax to <= 1; block a degenerate "
                    "post-clip box"
                ),
            },
            "single_cls_training_option": False,
        }
        write_json(staging / "mapping.json", mapping)
        family_manifest = {
            "seed": seed,
            "ratios": SPLIT_RATIOS,
            "normalization": (
                "Case-fold basename stem; remove terminal .rf.<hash>, then _jpg, then _augN. "
                "Exact-image duplicates link all observed family IDs before splitting."
            ),
            "groups": [
                {
                    "group_id": group["group_id"],
                    "family_ids": group["family_ids"],
                    "output_split": group["output_split"],
                    "forced_split": group["forced_split"],
                    "force_reasons": group["force_reasons"],
                    "image_count": group["image_count"],
                    "brand_box_counts": group["brand_box_counts"],
                }
                for group in groups
            ],
        }
        write_json(staging / "source_family_manifest.json", family_manifest)

        split_stats: dict[str, Any] = {}
        for split in OUTPUT_SPLITS:
            selected = [unit for unit in units if unit.output_split == split]
            counts = Counter(box.brand for unit in selected for box in unit.boxes)
            split_stats[split] = {
                "images": len(selected),
                "boxes": sum(counts.values()),
                "brand_boxes": {brand: counts[brand] for brand in TARGET_BRANDS},
                "family_components": len({unit.component_id for unit in selected}),
                "forced_holdout_images": sum(bool(unit.forced_split) for unit in selected),
            }
        report = {
            "status": "prepared",
            "target_brands": list(TARGET_BRANDS),
            "output_class_mapping": {
                str(index): brand for index, brand in enumerate(TARGET_BRANDS)
            },
            "audit_report": str(audit_dir / "zip_audit.json"),
            "source_archives": [
                {
                    "path": report_item["archive"],
                    "sha256": report_item["archive_sha256"],
                    "yaml_entry_count": len(report_item["yaml_entries"]),
                    "yaml_consistent_within_archive": report_item[
                        "yaml_consistent_within_archive"
                    ],
                    "declared_nc": report_item["yaml_entries"][0].get("declared_nc"),
                    "effective_nc": report_item["yaml_entries"][0].get("effective_nc"),
                    "direct_generic_brand_fraction": report_item[
                        "direct_generic_brand_fraction"
                    ],
                }
                for report_item in reports
            ],
            "selection_and_deduplication": dedup_report,
            "annotation_conflicts": {
                "resolved_group_count": len(annotation_conflicts),
                "report": "conflicting_annotations_resolved.csv",
                "silently_selected": False,
            },
            "existing_axe": {**axe_report, **axe_merge_report},
            "model2_holdouts": {
                **model2_source_report,
                **model2_match_report,
                "validation_and_test_preserved": True,
                "historical_artifacts_modified": False,
            },
            "conversion": {
                "ordinary_detection_output": True,
                "obb_and_polygon_rule": (
                    "xmin=min(x_i), xmax=max(x_i), ymin=min(y_i), ymax=max(y_i)"
                ),
                "target_boxes_clipped_to_image_boundary": sum(
                    box.was_clipped for unit in units for box in unit.boxes
                ),
                "selected_images": len(units),
                "output_boxes": sum(len(unit.boxes) for unit in units),
                "selected_source_non_target_annotations_removed": sum(
                    unit.canonical.non_target_annotation_count for unit in units
                ),
            },
            "split_assignment": split_report,
            "splits": split_stats,
            "family_matching": {
                "uncertain_multi_image_family_group_count": len(uncertain_groups),
                "review_report": "uncertain_family_review.csv",
                "limitation": (
                    "Filename normalization and exact byte hashes cannot prove capture-session "
                    "independence or find every near-duplicate."
                ),
            },
            "label_completeness": {
                "selected_images_have_at_least_one_target_box": True,
                "invalid_selected_rows": 0,
                "limitation": (
                    "Geometry/YAML checks do not prove that every visible product was annotated "
                    "or that every historical annotation is visually correct."
                ),
            },
            "source_records_in_provenance": len(provenance),
            "test_split_evaluated": False,
        }
        write_json(staging / "dataset_report.json", report)
        checks = validate_dataset(staging)
        write_json(staging / "dataset_checks.json", checks)
        preview_report = write_previews(staging, preview_count)
        report["previews"] = preview_report
        write_json(staging / "dataset_report.json", report)
        staging.rename(output_dir)
        return report
    except Exception:
        if staging.exists() and staging.parent == output_dir.parent.resolve():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir)
    if args.check_only:
        print(json.dumps(validate_dataset(output_dir), ensure_ascii=False, indent=2))
        return
    zip_dir = resolve_path(args.zip_dir)
    audit_dir = resolve_path(args.audit_dir)
    axe_dir = resolve_path(args.axe_dir)
    reports, records = audit_archives(zip_dir, audit_dir)
    audit_status = "blocked" if any(report["blocking_issues"] for report in reports) else "passed"
    if args.audit_only or audit_status == "blocked":
        print(
            json.dumps(
                {
                    "audit_status": audit_status,
                    "audit_dir": str(audit_dir),
                    "dataset_prepared": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if audit_status == "blocked" and not args.audit_only:
            raise SystemExit(2)
        return
    report = prepare_dataset(
        audit_dir,
        output_dir,
        axe_dir,
        reports,
        records,
        args.seed,
        args.preview_count,
    )
    print(
        json.dumps(
            {
                "audit_status": audit_status,
                "audit_dir": str(audit_dir),
                "output_dir": str(output_dir),
                "mapping": report["output_class_mapping"],
                "images": report["conversion"]["selected_images"],
                "boxes": report["conversion"]["output_boxes"],
                "splits": report["splits"],
                "test_split_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
