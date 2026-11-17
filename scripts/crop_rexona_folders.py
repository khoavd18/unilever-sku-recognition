#!/usr/bin/env python3
"""Crop COCO ZIPs or extracted dataset folders into one output. Python 3.10+."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageOps


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    if path.is_dir():
        # A folder fingerprint includes relative filenames and file contents.
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(item).encode("ascii"))
        return digest.hexdigest()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DirectorySource:
    """Read-only adapter; each annotation JSON retains its own ID namespace."""
    def __init__(self, root):
        self.root = root.resolve()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def namelist(self):
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def read(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError("Image path escapes source folder")
        return path.read_bytes()


def open_source(path):
    return DirectorySource(path) if path.is_dir() else zipfile.ZipFile(path)


def family_hint(image):
    """Filename heuristic only; does NOT establish capture/session independence."""
    extra = image.get("extra") or {}
    name = extra.get("name") or image["file_name"]
    name = PurePosixPath(name.replace("\\", "/")).name.casefold()
    name = name.split(".rf.", 1)[0]
    name = re.sub(r"(?:[._](?:jpg|jpeg|png|webp))$", "", name)
    return re.sub(r"(?:[_-]aug\d+)$", "", name)


def crop_bounds(bbox, size, padding):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be COCO [x, y, width, height]")
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("bbox is non-finite or has non-positive width/height")
    width, height = size
    # Reject boxes entirely outside the image before applying padding.
    if x >= width or y >= height or x + w <= 0 or y + h <= 0:
        raise ValueError("bbox does not intersect the image")
    left = max(0, math.floor(x - w * padding))
    top = max(0, math.floor(y - h * padding))
    right = min(width, math.ceil(x + w * (1 + padding)))
    bottom = min(height, math.ceil(y + h * (1 + padding)))
    if right <= left or bottom <= top:
        raise ValueError("empty crop after clamping")
    return left, top, right, bottom


def image_member(annotation_member, filename, members):
    filename = filename.replace("\\", "/")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe image filename")
    candidates = [str(PurePosixPath(annotation_member).parent / relative), str(relative)]
    for candidate in dict.fromkeys(candidates):
        if candidate in members:
            return candidate
    raise ValueError("source image missing from archive")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheets(output, rows):
    directory = output / "contact_sheets"
    directory.mkdir()
    per_page, columns, cell_w, cell_h = 48, 6, 150, 195
    for page_start in range(0, len(rows), per_page):
        page = rows[page_start:page_start + per_page]
        sheet = Image.new("RGB", (columns * cell_w, math.ceil(len(page) / columns) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            x, y = (index % columns) * cell_w, (index // columns) * cell_h
            with Image.open(output / row["initial_crop_path"]) as crop:
                thumb = ImageOps.contain(crop, (cell_w - 12, cell_h - 40))
                sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
            draw.text((x + 5, y + cell_h - 33), row["crop_id"], fill="black")
            draw.text((x + 5, y + cell_h - 17), f'{row["crop_width"]}x{row["crop_height"]} px', fill="black")
        sheet.save(directory / f"page_{page_start // per_page + 1:03d}.jpg", quality=90)


MANIFEST_FIELDS = [
    "crop_id", "crop_filename", "initial_crop_path", "archive", "archive_sha256",
    "annotation_member", "source_split", "source_image_id", "source_annotation_id",
    "source_filename", "source_member", "source_sha256", "family_hint_unverified",
    "category_id", "category_name", "source_width", "source_height", "bbox_xywh",
    "crop_xyxy", "padding_fraction", "crop_width", "crop_height", "small_crop",
    "iscrowd", "crop_sha256",
]
ISSUE_FIELDS = ["archive", "annotation_member", "image_id", "annotation_id", "reason"]


def crop_archives(archives, output, brand="REXONA", padding=0.03, min_size=1, make_sheets=True):
    archives = [Path(p).resolve() for p in archives]
    output = Path(output).resolve()
    if not archives or len(set(archives)) != len(archives):
        raise ValueError("Provide distinct input ZIP paths")
    for source in archives:
        if source.is_dir() and (source == output or source in output.parents or output in source.parents):
            raise ValueError("Input and output directories must be separate, not nested")
    if not math.isfinite(padding) or not 0 <= padding <= 0.5 or min_size < 1:
        raise ValueError("padding must be 0..0.5 and min-size must be >= 1")
    if output.exists():
        raise ValueError("Output already exists. Choose a NEW directory; existing work is never overwritten.")

    # Preflight structural errors before creating any output.
    datasets, sources = [], []
    for archive in archives:
        digest = file_sha256(archive)
        with open_source(archive) as z:
            members = z.namelist()
            if len(set(members)) != len(members):
                raise ValueError(f"Duplicate ZIP member names: {archive.name}")
            json_members = sorted(n for n in members if PurePosixPath(n).name == "_annotations.coco.json")
            if not json_members:
                raise ValueError(f"No _annotations.coco.json found: {archive.name}")
            source = {"archive": archive.name, "sha256": digest, "source_images": 0,
                      "rexona_annotations": 0, "splits": {}}
            for member in json_members:
                data = json.loads(z.read(member))
                cats = {c["id"]: c["name"] for c in data["categories"]}
                images = {im["id"]: im for im in data["images"]}
                annotations = data["annotations"]
                if len(cats) != len(data["categories"]) or len(images) != len(data["images"]):
                    raise ValueError(f"Duplicate category or image IDs in {member}")
                if len({a["id"] for a in annotations}) != len(annotations):
                    raise ValueError(f"Duplicate annotation IDs in {member}")
                if any(a["category_id"] not in cats for a in annotations):
                    raise ValueError(f"Undefined category ID in {member}")
                chosen = [a for a in annotations if cats[a["category_id"]].strip().casefold() == brand.casefold()]
                source["source_images"] += len(images)
                source["rexona_annotations"] += len(chosen)
                split = PurePosixPath(member).parent.name
                source["splits"][member] = {"split": split, "images": len(images), "rexona_annotations": len(chosen)}
                datasets.append((archive, digest, member, images, cats, chosen))
            sources.append(source)
    if not sum(s["rexona_annotations"] for s in sources):
        raise ValueError(f"No exact category name {brand!r}; inspect category names first")

    output.mkdir(parents=True)
    for dirname in ("unassigned", "sku_folders", "needs_review", "excluded"):
        (output / dirname).mkdir()
    rows, issues, sequence, processed_images = [], [], 0, 0
    for archive, digest, member, images, cats, annotations in datasets:
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann["image_id"]].append(ann)
        print(f"Processing {archive.name} / {member}: {len(annotations)} {brand} annotations", flush=True)
        with open_source(archive) as z:
            members = set(z.namelist())
            for image_id, image_annotations in grouped.items():
                context = {"archive": archive.name, "annotation_member": member, "image_id": image_id}
                try:
                    im = images[image_id]
                    source_member = image_member(member, im["file_name"], members)
                    image_bytes = z.read(source_member)
                    source_hash = sha256_bytes(image_bytes)
                    with Image.open(io.BytesIO(image_bytes)) as original:
                        original.load()
                        if original.size != (int(im["width"]), int(im["height"])):
                            raise ValueError("decoded image dimensions differ from COCO metadata")
                        # Keep stored pixel orientation: labels use this coordinate system.
                        rgb = original.convert("RGB")
                except (KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
                    for ann in image_annotations:
                        issues.append({**context, "annotation_id": ann["id"], "reason": f"image: {exc}"})
                    continue
                for ann in image_annotations:
                    sequence += 1
                    try:
                        bounds = crop_bounds(ann.get("bbox"), rgb.size, padding)
                        cropped = rgb.crop(bounds)
                        if min(cropped.size) < min_size:
                            raise ValueError("crop smaller than min-size")
                    except (ValueError, TypeError, OverflowError) as exc:
                        issues.append({**context, "annotation_id": ann["id"], "reason": f"bbox: {exc}"})
                        continue
                    crop_id = f"rx_{sequence:06d}"
                    identity = json.dumps([digest, member, image_id, ann["id"]], ensure_ascii=False).encode()
                    filename = f"{crop_id}_{sha256_bytes(identity)[:10]}.jpg"
                    relative_path = "unassigned/" + filename
                    cropped.save(output / relative_path, format="JPEG", quality=95)
                    rows.append(dict(zip(MANIFEST_FIELDS, [
                        crop_id, filename, relative_path, archive.name, digest, member,
                        PurePosixPath(member).parent.name, image_id, ann["id"], im["file_name"],
                        source_member, source_hash, family_hint(im), ann["category_id"], cats[ann["category_id"]],
                        rgb.width, rgb.height, json.dumps(ann["bbox"]), json.dumps(bounds), padding,
                        cropped.width, cropped.height, int(min(cropped.size) < 30), ann.get("iscrowd", 0),
                        file_sha256(output / relative_path),
                    ])))
                rgb.close()
                processed_images += 1
                if processed_images % 20 == 0:
                    print(f"  {processed_images} source images decoded; {len(rows)} crops written", flush=True)
    write_csv(output / "manifest.csv", rows, MANIFEST_FIELDS)
    write_csv(output / "issues.csv", issues, ISSUE_FIELDS)
    crop_hashes = defaultdict(list)
    families = defaultdict(set)
    for row in rows:
        crop_hashes[row["crop_sha256"]].append(row["crop_id"])
        families[row["family_hint_unverified"]].add(row["source_split"])
    report = {
        "brand": brand, "sources": sources, "source_image_records": sum(s["source_images"] for s in sources),
        "target_annotations": sum(s["rexona_annotations"] for s in sources), "successful_crops": len(rows),
        "skipped_annotations": len(issues), "source_images_decoded": processed_images,
        "crop_counts_by_archive": dict(Counter(r["archive"] for r in rows)),
        "small_crops_min_dimension_under_30px": sum(r["small_crop"] for r in rows),
        "exact_duplicate_crop_groups_retained": [v for v in crop_hashes.values() if len(v) > 1],
        "filename_family_hints": len(families),
        "cross_split_family_hints": {k: sorted(v) for k, v in families.items() if len(v) > 1},
        "padding_fraction_per_side": padding, "min_size": min_size,
        "sku_classification": "NOT performed; all crops are unassigned",
        "notes": ["Sources unchanged. No training, embeddings or re-splitting performed.",
                  "In folder mode, archive denotes the input root; annotation_member identifies each dataset/split.",
                  "In folder mode, archive_sha256 fingerprints the complete input directory tree.",
                  "Crops preserve annotation bbox plus padding; background is not removed.",
                  "All duplicate crops are retained. family_hint is only a filename heuristic.",
                  "Source splits are provenance, NOT a certified independent training split.",
                  "Annotation labels and SKU identities have not been manually verified."],
    }
    if len(rows) + len(issues) != report["target_annotations"]:
        raise RuntimeError("Accounting mismatch")
    if make_sheets:
        contact_sheets(output, rows)
    report["source_archives_unchanged"] = all(file_sha256(p) == s["sha256"] for p, s in zip(archives, sources))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(rows)} crops; {len(issues)} skipped. Output: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", nargs="+", type=Path, help="One or more COCO ZIPs")
    source.add_argument("--zip-dir", type=Path, help="Folder containing only the input COCO ZIPs")
    source.add_argument("--dataset-root", type=Path, help="Parent folder containing the two extracted COCO datasets")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--brand", default="REXONA", help="Exact category name; case insensitive")
    parser.add_argument("--padding", type=float, default=0.03, help="Fraction of box width/height per side")
    parser.add_argument("--min-size", type=int, default=1, help="Keep tiny crops by default for human review")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    if args.dataset_root is not None:
        if not args.dataset_root.is_dir():
            parser.error("--dataset-root must be an existing directory")
        archives = [args.dataset_root]
    else:
        archives = args.zip if args.zip is not None else sorted(args.zip_dir.glob("*.zip"))
    try:
        crop_archives(archives, args.output_dir, args.brand, args.padding, args.min_size, not args.no_contact_sheets)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
