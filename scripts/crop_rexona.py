#!/usr/bin/env python3
"""Crop annotated Rexona products; no detector, training, or GPU required.

Requirements: Pillow, PyYAML (Python 3.10+).
Example (run from the project root):
  python scripts/crop_rexona.py --dataset-path data/rexona_raw --output-dir outputs/rexona_crops

Supported layouts inside ONE dataset root with ONE data.yaml/data.yml:
  images/... + labels/...
  train/images/... + train/labels/... (also valid, val, test, export)
  images/train/... + labels/train/... (also valid, val, test)

The script scans actual images/labels folders under --dataset-path. It does not
follow YAML paths outside this root (Roboflow often exports ../train/images).
Separate exports with different class mappings must be processed separately.

Outputs: crops/<source-class>/, manifest.csv, issues.csv, report.json.
Generic Rexona labels go into rexona_unassigned. Other folder names come from
the annotation, not inferred SKU predictions. All crops still require review;
this script does NOT cluster images or infer a SKU from cap/body colors.
Original files are never modified. Existing output directories are refused.
EXIF orientation is not applied: coordinates refer to stored image pixels.
--dry-run validates/decodes images but writes nothing. One crop is retained for
each valid annotation; duplicates are not silently deleted. source_sha256 and
source_family help trace duplicates and preserve later split integrity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit("Missing dependency. Install into this Python: python -m pip install Pillow PyYAML") from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_FIELDS = [
    "crop_id", "crop_path", "source_image", "source_label", "source_split",
    "source_family", "source_sha256", "annotation_line", "source_class_id",
    "source_class_name", "annotation_format", "annotation_coordinates_json",
    "box_xyxy_pixels_json", "crop_xyxy_pixels_json", "crop_width", "crop_height",
    "padding", "small_object", "review_status",
]


def load_mapping(root: Path) -> tuple[Path, dict[int, str]]:
    configs = sorted(p for p in root.rglob("*")
                     if p.is_file() and p.name.lower() in {"data.yaml", "data.yml"})
    if len(configs) != 1:
        raise ValueError(f"Expected ONE data.yaml/data.yml under {root}; found {len(configs)}. "
                         "Point --dataset-path at a single extracted dataset, not multiple versions.")
    document = yaml.safe_load(configs[0].read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("data.yaml must contain a mapping with 'names'.")
    raw = document.get("names")
    if isinstance(raw, list):
        raw = dict(enumerate(raw))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("data.yaml: 'names' must be a nonempty list or dictionary.")
    names = {}
    for key, value in raw.items():
        cid = int(key)
        if str(cid) != str(key) or cid < 0 or cid in names or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid/duplicate class mapping entry: {key!r}: {value!r}")
        names[cid] = value.strip()
    return configs[0], names


def discover_images(root: Path) -> list[tuple[Path, Path]]:
    folders = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "images")
    if not folders:
        raise ValueError("No images/ folder found under --dataset-path.")
    pairs = {}
    for folder in folders:
        matches = [p for p in folder.parent.iterdir() if p.is_dir() and p.name.lower() == "labels"]
        if len(matches) != 1:
            raise ValueError(f"Expected a labels/ folder next to {folder}.")
        for image_path in sorted(folder.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                label = matches[0] / image_path.relative_to(folder).with_suffix(".txt")
                if image_path in pairs and pairs[image_path] != label:
                    raise ValueError(f"Ambiguous image/label pairing: {image_path}")
                pairs[image_path] = label
    if not pairs:
        raise ValueError("No supported image files found.")
    return sorted(pairs.items())


def parse_row(line: str, names: dict[int, str]):
    parts = line.split()
    if not parts:
        return None
    cid = int(parts[0])
    if cid not in names:
        raise ValueError(f"Class ID {cid} is absent from data.yaml.")
    values = [float(v) for v in parts[1:]]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Coordinates must be finite, normalized numbers in [0, 1].")
    if len(values) == 4:
        xc, yc, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError("Box width/height must be positive.")
        bounds = (xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2)
        kind = "bbox"
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[::2], values[1::2]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        kind = "polygon"
    else:
        raise ValueError("Expected 4 box coordinates or at least 3 polygon vertices.")
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in bounds)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate box after clipping to image bounds.")
    return cid, kind, values, (x1, y1, x2, y2)


def pixel_boxes(bounds, width: int, height: int, padding: float):
    x1, y1, x2, y2 = bounds
    base = (x1 * width, y1 * height, x2 * width, y2 * height)
    dx, dy = (base[2] - base[0]) * padding, (base[3] - base[1]) * padding
    crop = (max(0, math.floor(base[0] - dx)), max(0, math.floor(base[1] - dy)),
            min(width, math.ceil(base[2] + dx)), min(height, math.ceil(base[3] + dy)))
    return base, crop


def source_family(stem: str) -> str:
    stem = re.sub(r"\.rf\.[^.]+$", "", stem, flags=re.I)
    stem = re.sub(r"_(?:jpg|jpeg|png|webp)$", "", stem, flags=re.I)
    return re.sub(r"(?:_aug\d+)+$", "", stem, flags=re.I)


def group_folder(cid: int, name: str, brand: str) -> str:
    if name.casefold() == brand.casefold():
        return "rexona_unassigned" if brand.casefold() == "rexona" else "brand_unassigned"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:52] or "source_class"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"class_{cid}_{safe}_{digest}"


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path, output: Path, brand="Rexona", padding=0.03, min_size=1, dry_run=False):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Input and output directories must be separate; neither may contain the other.")
    if not brand.strip() or not math.isfinite(padding) or not 0 <= padding <= 0.5 or min_size < 1:
        raise ValueError("Use a nonempty brand, padding in [0, 0.5], and min-size >= 1.")
    if output.exists() and not dry_run:
        raise ValueError(f"Output already exists: {output}. Choose a new --output-dir; nothing was overwritten.")
    yaml_path, names = load_mapping(root)
    selected = {cid: name for cid, name in names.items() if brand.casefold() in name.casefold()}
    if not selected:
        raise ValueError(f"No class name contains {brand!r}. Available names: {list(names.values())}")
    pairs = discover_images(root)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    rows, issues = [], []
    label_counts, format_counts, crop_counts = Counter(), Counter(), Counter()
    image_hashes = Counter()

    def issue(image_path, line_no, reason, detail):
        issues.append({"source_image": image_path.relative_to(root).as_posix(),
                       "annotation_line": line_no, "reason": reason, "detail": str(detail)})

    for image_no, (image_path, label_path) in enumerate(pairs, 1):
        if not label_path.is_file():
            issue(image_path, "", "missing_label", "Not treated as a verified negative image.")
            continue
        try:
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            issue(image_path, "", "unreadable_label", exc)
            continue
        selected_rows = []
        for line_no, line in enumerate(lines, 1):
            try:
                parsed = parse_row(line, names)
                if parsed and parsed[0] in selected:
                    selected_rows.append((line_no, *parsed))
                    label_counts[names[parsed[0]]] += 1
            except (ValueError, OverflowError) as exc:
                issue(image_path, line_no, "invalid_annotation", exc)
        if not selected_rows:
            continue
        try:
            source_bytes = image_path.read_bytes()
            with Image.open(io.BytesIO(source_bytes)) as original:
                image = original.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            issue(image_path, "", "unreadable_image", exc)
            continue
        image_hash = hashlib.sha256(source_bytes).hexdigest()
        image_hashes[image_hash] += 1
        rel_image = image_path.relative_to(root).as_posix()
        split = next((p for p in image_path.relative_to(root).parts[:-1]
                      if p.casefold() in {"train", "valid", "val", "test", "export"}), "unspecified")
        for line_no, cid, kind, values, bounds in selected_rows:
            base, crop_box = pixel_boxes(bounds, image.width, image.height, padding)
            object_width, object_height = base[2] - base[0], base[3] - base[1]
            if min(object_width, object_height) + 1e-9 < min_size:
                issue(image_path, line_no, "object_too_small", f"{object_width:.2f} x {object_height:.2f} before padding")
                continue
            identity = json.dumps([rel_image, image_hash, line_no, names[cid], values, padding], ensure_ascii=False)
            crop_id = "rc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            crop_rel = Path("crops") / group_folder(cid, names[cid], brand) / f"{crop_id}.jpg"
            crop_path = output / crop_rel
            if sys.platform == "win32" and len(str(crop_path)) >= 240:
                raise ValueError("Output path is too long for portable Windows use. Choose a shorter --output-dir.")
            if not dry_run:
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                with crop_path.open("xb") as handle:
                    image.crop(crop_box).save(handle, format="JPEG", quality=95)
            rows.append({
                "crop_id": crop_id, "crop_path": crop_rel.as_posix(), "source_image": rel_image,
                "source_label": label_path.relative_to(root).as_posix(), "source_split": split,
                "source_family": source_family(image_path.stem), "source_sha256": image_hash,
                "annotation_line": line_no, "source_class_id": cid, "source_class_name": names[cid],
                "annotation_format": kind, "annotation_coordinates_json": json.dumps(values),
                "box_xyxy_pixels_json": json.dumps(base), "crop_xyxy_pixels_json": json.dumps(crop_box),
                "crop_width": crop_box[2] - crop_box[0], "crop_height": crop_box[3] - crop_box[1],
                "padding": padding, "small_object": min(object_width, object_height) < 20,
                "review_status": "needs_sku_assignment" if names[cid].casefold() == brand.casefold() else "source_label_unverified",
            })
            crop_counts[names[cid]] += 1
            format_counts[kind] += 1
        image.close()
        if image_no % 50 == 0 or image_no == len(pairs):
            print(f"Scanned {image_no}/{len(pairs)} images; {len(rows)} valid crops", file=sys.stderr, flush=True)
    report = {
        "dry_run": dry_run, "dataset_root": str(root), "data_yaml": str(yaml_path), "output_dir": str(output),
        "brand_substring": brand, "selected_classes": selected, "images_discovered": len(pairs),
        "images_decoded_with_brand_annotations": sum(image_hashes.values()),
        "exact_duplicate_source_occurrences": sum(n - 1 for n in image_hashes.values()),
        "valid_brand_annotations": sum(label_counts.values()), "successful_crops": len(rows),
        "created_crops": 0 if dry_run else len(rows), "crops_by_source_class": dict(crop_counts),
        "formats": dict(format_counts), "padding": padding, "min_size_before_padding": min_size,
        "issue_count": len(issues), "issues_by_reason": dict(Counter(x["reason"] for x in issues)),
        "small_object_crops": sum(r["small_object"] for r in rows),
        "notes": ["Folder names come from source labels, not visual clustering or verified SKU identities.",
                  "Rexona-only annotations require later SKU grouping and review.",
                  "Coordinates use stored pixels without EXIF transposition.",
                  "Source-family names are a filename heuristic, not proof of independent captures.",
                  "All valid annotation instances are retained; duplicates are reported, not removed."],
    }
    if not dry_run:
        write_csv(output / "manifest.csv", rows, MANIFEST_FIELDS)
        write_csv(output / "issues.csv", issues, ["source_image", "annotation_line", "reason", "detail"])
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brand", default="Rexona", help="Case-insensitive substring in YAML class names.")
    parser.add_argument("--padding", type=float, default=0.03, help="Padding on each side as fraction of object dimensions.")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum object side in pixels BEFORE padding.")
    parser.add_argument("--dry-run", action="store_true", help="Check labels and decode relevant images; write nothing.")
    args = parser.parse_args()
    try:
        report = prepare(args.dataset_path, args.output_dir, args.brand, args.padding, args.min_size, args.dry_run)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["successful_crops"] == 0:
        print("No crops available. Check class names, annotations and issue counts.", file=sys.stderr)
        return 2
    if report["issue_count"]:
        print("Some records were skipped. Review report.json and issues.csv before using the dataset.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
