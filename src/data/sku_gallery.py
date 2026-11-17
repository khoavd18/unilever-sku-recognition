"""Prepare the Roboflow 469-SKU segmentation export for Model 1 and Model 2.

* Polygon rows (YOLO segmentation) and bbox rows are both converted to
  axis-aligned boxes with :mod:`src.data.annotations` (min/max of the polygon
  points, clipped to the image).  Raw archives are never modified.
* Model 1 labels: brand-level boxes using the Model 1 *training* class order.
* Model 2 gallery: per-SKU crops grouped by item code, one representative per
  source group, preferring unmarked images and falling back to augmented ones.
  The manifest records source provenance and split.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

import yaml
from PIL import Image

from src.data.annotations import AnnotationFormat, NormalizedBox, normalize_yolo_names, parse_yolo_annotation
from src.data.crops import bbox_to_crop_box, make_crop_filename, sanitize_class_name
from src.data.yolo_archive import label_member_for, write_csv, write_json

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "valid", "test")
_RF_NAME = re.compile(r"^(?P<base>.+?)(?:_aug\d+)?_(?:jpg|jpeg|png)\.rf\.[0-9a-fA-F]+$")
_AUG_NAME = re.compile(r"_aug\d+_(?:jpg|jpeg|png)\.rf\.", re.I)


def source_group(stem: str) -> str:
    """Group key shared by an original image and its ``_augN`` variants."""
    match = _RF_NAME.match(stem)
    return match.group("base") if match else stem


def is_augmented(stem: str) -> bool:
    return bool(_AUG_NAME.search(stem))


def norm_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z]+", "", text)


def gallery_split(group: str, train_pct: int = 80, valid_pct: int = 10) -> str:
    """Deterministic GROUP-level split for Model 2 (the Roboflow splits share source groups)."""
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < train_pct else "valid" if bucket < train_pct + valid_pct else "test"


def row_format(fields: list[str]) -> AnnotationFormat:
    """5 fields = bbox, otherwise segmentation polygon (this export is mixed)."""
    return AnnotationFormat.YOLO_BBOX if len(fields) == 5 else AnnotationFormat.YOLO_SEGMENTATION


# --------------------------------------------------------------------- class map
def load_item_table(xlsx: str | Path, sheet: str = "TIẾN ĐỘ") -> list[dict[str, str]]:
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(workbook[sheet].iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    need = {"BRAND": "BRAND", "ITEM_CODE": "ITEM_CODE", "PRODUCT_NAME_VN": "PRODUCT_NAME_VN", "CLASS LABEL": "CLASS LABEL"}
    index = {key: header.index(col) for key, col in need.items()}
    items = []
    for row in rows[1:]:
        label, code = row[index["CLASS LABEL"]], row[index["ITEM_CODE"]]
        if not label or code in (None, ""):
            continue
        code = str(int(code)) if isinstance(code, (int, float)) else str(code).strip()
        items.append({
            "class_label": str(label).strip(), "item_code": code,
            "brand": str(row[index["BRAND"]] or "").strip(),
            "product_name": str(row[index["PRODUCT_NAME_VN"]] or "").strip(),
        })
    return items


def load_model1_brands(data_yaml: str | Path, registry_yaml: str | Path | None = None):
    """Return (training_names, lookup norm_key -> training id) incl. registry aliases."""
    names = normalize_yolo_names((yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {})["names"])
    lookup = {norm_key(n): i for i, n in enumerate(names)}
    if registry_yaml and Path(registry_yaml).exists():
        for brand in yaml.safe_load(Path(registry_yaml).read_text(encoding="utf-8"))["brands"]:
            target = lookup.get(norm_key(brand["name"]))
            if target is not None:
                for alias in brand.get("aliases") or []:
                    lookup.setdefault(norm_key(alias), target)
    return names, lookup


def build_class_map(class_names, items, brand_lookup, brand_names) -> list[dict[str, Any]]:
    exact: dict[str, list[dict]] = defaultdict(list)
    fuzzy: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        exact[item["class_label"]].append(item)
        fuzzy[norm_key(item["class_label"])].append(item)
    result = []
    for class_id, name in enumerate(class_names):
        method, item = "unresolved", None
        for table, key, label in ((exact, name.strip(), "exact"), (fuzzy, norm_key(name), "normalized")):
            candidates = table.get(key, [])
            if len({c["item_code"] for c in candidates}) == 1:
                method, item = label, candidates[0]
                break
        brand_text = item["brand"] if item else name.replace(" ", "_").split("_")[0]
        brand_id = brand_lookup.get(norm_key(brand_text), -1)
        result.append({
            "sku_class_id": class_id, "sku_class_name": name,
            "item_code": item["item_code"] if item else "",
            "product_name": item["product_name"] if item else "",
            "brand_text": brand_text,
            "brand_id": brand_id, "brand_name": brand_names[brand_id] if brand_id >= 0 else "",
            "match_method": method,
            "needs_review": method != "exact" or brand_id < 0,
        })
    return result


# ------------------------------------------------------------------------ prepare
def prepare(
    zip_path: Path, class_map: list[dict], brand_names: list[str], out_dir: Path, *,
    padding: float = 0.05, min_crop_px: int = 24, write_detector_images: bool = False,
    detector_originals_only: bool = False, strict_class_map: bool = False,
    limit_images: int | None = None, jpeg_quality: int = 95,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Output exists; choose a new version directory: {out_dir}")
    cmap = {row["sku_class_id"]: row for row in class_map}
    sku_names = [row["sku_class_name"] for row in class_map]
    audit: dict[str, Any] = {"splits": {}, "invalid_reasons": Counter(), "crops": Counter()}
    group_splits: dict[str, set[str]] = defaultdict(set)
    sku_instances, sku_crops, brand_instances = Counter(), Counter(), Counter()
    crop_rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True)

    with ZipFile(zip_path) as archive:
        infos = {i.filename: i for i in archive.infolist() if not i.is_dir()}
        archive_names_hash = hashlib.sha256("\n".join(sorted(infos)).encode()).hexdigest()
        # One representative image per source group: Roboflow stores ~3 near-identical
        # versions of every source image (some without an "_augN" marker).
        order = {"train": 0, "valid": 1, "test": 2}
        representative: dict[str, tuple] = {}
        for name in infos:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] in order and parts[1] == "images" and name.lower().endswith(IMAGE_EXTENSIONS):
                key = (is_augmented(Path(name).stem), order[parts[0]], name)
                group = source_group(Path(name).stem)
                if group not in representative or key < representative[group]:
                    representative[group] = key
        for split in SPLITS:
            members = sorted(n for n in infos if n.startswith(f"{split}/images/") and n.lower().endswith(IMAGE_EXTENSIONS))
            if limit_images:
                members = members[:limit_images]
            st = Counter()
            for image_idx, member in enumerate(members):
                stem = Path(member).stem
                group, aug = source_group(stem), is_augmented(stem)
                group_splits[group].add(split)
                st["images"] += 1
                st["augmented" if aug else "original"] += 1
                label_member = label_member_for(member)
                if label_member not in infos:
                    st["label_missing"] += 1
                    continue
                lines = [l for l in archive.read(infos[label_member]).decode("utf-8").splitlines() if l.strip()]
                if not lines:
                    st["label_empty"] += 1
                    continue
                anns = []
                for ann_idx, line in enumerate(lines):
                    fields = line.split()
                    try:
                        parsed = parse_yolo_annotation(line, sku_names, row_format(fields))
                    except ValueError as exc:
                        st["rows_invalid"] += 1
                        audit["invalid_reasons"][str(exc)[:80]] += 1
                        continue
                    st["rows_polygon" if len(fields) > 5 else "rows_bbox"] += 1
                    st["rows_clipped"] += int(parsed.box.was_clipped)
                    anns.append((ann_idx, parsed, (len(fields) - 1) // 2 if len(fields) > 5 else 0))
                    sku_instances[parsed.class_id] += 1

                def yolo_row(cid: int, b: NormalizedBox) -> str:
                    return f"{cid} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"

                (out_dir / "labels_sku" / split).mkdir(parents=True, exist_ok=True)
                (out_dir / "labels_sku" / split / f"{stem}.txt").write_text(
                    "\n".join(yolo_row(a.class_id, a.box) for _, a, _ in anns) + "\n", encoding="utf-8")

                brand_rows = [yolo_row(cmap[a.class_id]["brand_id"], a.box) for _, a, _ in anns if cmap[a.class_id]["brand_id"] >= 0]
                for _, a, _ in anns:
                    brand_instances[cmap[a.class_id]["brand_id"]] += 1
                st["rows_brand_unresolved"] += len(anns) - len(brand_rows)
                keep_detector = brand_rows and not (detector_originals_only and aug)
                if not brand_rows:
                    st["images_without_brand_boxes"] += 1
                if keep_detector:
                    lbl = out_dir / "detector_brand21" / "labels" / split
                    lbl.mkdir(parents=True, exist_ok=True)
                    (lbl / f"{stem}.txt").write_text("\n".join(brand_rows) + "\n", encoding="utf-8")
                    st["detector_images"] += 1

                need_crops = representative[group][2] == member
                if not (need_crops or (write_detector_images and keep_detector)):
                    continue
                raw = archive.read(infos[member])
                if write_detector_images and keep_detector:
                    img_dir = out_dir / "detector_brand21" / "images" / split
                    img_dir.mkdir(parents=True, exist_ok=True)
                    (img_dir / Path(member).name).write_bytes(raw)
                if not need_crops:
                    continue
                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    width, height = im.size
                    for ann_idx, a, npts in anns:
                        row = cmap[a.class_id]
                        if strict_class_map and row["match_method"] != "exact":
                            audit["crops"]["skipped_non_exact_class_map"] += 1
                            continue
                        b = a.box
                        box = bbox_to_crop_box([b.x_min * width, b.y_min * height, b.width * width, b.height * height], width, height, padding)
                        if box is None or min(box[2] - box[0], box[3] - box[1]) < min_crop_px:
                            audit["crops"]["skipped_too_small"] += 1
                            continue
                        folder = row["item_code"] or f"UNMAPPED_{sanitize_class_name(row['sku_class_name'])}"
                        fname = make_crop_filename(stem, image_idx, ann_idx, a.class_id, original_split=split, compact=True)
                        target = out_dir / "gallery" / folder
                        target.mkdir(parents=True, exist_ok=True)
                        im.crop(box).save(target / fname, quality=jpeg_quality)
                        sku_crops[a.class_id] += 1
                        crop_rows.append({
                            "crop_path": f"gallery/{folder}/{fname}", "roboflow_split": split, "gallery_split": gallery_split(group), "group_id": group,
                            "source_image": member, "annotation_index": ann_idx,
                            "sku_class_id": a.class_id, "sku_class_name": row["sku_class_name"],
                            "item_code": row["item_code"], "brand_name": row["brand_name"],
                            "match_method": row["match_method"], "geometry": "polygon" if npts else "bbox",
                            "polygon_points": npts, "box_clipped": a.box.was_clipped,
                            "crop_box_px": f"{box[0]},{box[1]},{box[2]},{box[3]}", "image_size_px": f"{width}x{height}",
                        })
            audit["splits"][split] = dict(st)

    cross = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    audit.update({
        "zip": {"path": str(zip_path), "size_bytes": zip_path.stat().st_size, "member_names_sha256": archive_names_hash},
        "settings": {"padding": padding, "min_crop_px": min_crop_px, "strict_class_map": strict_class_map,
                     "detector_originals_only": detector_originals_only, "limit_images": limit_images},
        "groups_total": len(group_splits), "groups_in_multiple_splits": len(cross),
        "group_leakage_examples": dict(list(cross.items())[:10]),
        "classes_total": len(class_map),
        "classes_without_item_code": [r["sku_class_name"] for r in class_map if not r["item_code"]],
        "classes_without_brand": [r["sku_class_name"] for r in class_map if r["brand_id"] < 0],
        "classes_non_exact_match": [r["sku_class_name"] for r in class_map if r["match_method"] == "normalized"],
        "classes_with_zero_instances": [sku_names[i] for i in range(len(sku_names)) if sku_instances[i] == 0],
        "classes_with_zero_crops": [sku_names[i] for i in range(len(sku_names)) if sku_crops[i] == 0],
        "sku_instances": {sku_names[i]: n for i, n in sorted(sku_instances.items())},
        "sku_crops": {sku_names[i]: n for i, n in sorted(sku_crops.items())},
        "brand_instances": {(brand_names[i] if i >= 0 else "UNRESOLVED"): n for i, n in sorted(brand_instances.items())},
        "invalid_reasons": dict(audit["invalid_reasons"]), "crops": dict(audit["crops"]), "crops_written": len(crop_rows),
    })
    per_sku: dict[tuple, int] = Counter((r["item_code"] or r["sku_class_name"], r["gallery_split"]) for r in crop_rows)
    codes = sorted({k[0] for k in per_sku})
    write_csv(out_dir / "sku_split_counts.csv", ["item_code", "train", "valid", "test"],
              [{"item_code": c, **{sp: per_sku.get((c, sp), 0) for sp in SPLITS}} for c in codes])
    audit["crop_source_groups"] = len(representative)
    audit["gallery_split_crops"] = dict(Counter(r["gallery_split"] for r in crop_rows))
    audit["skus_without_train_crops"] = [c for c in codes if per_sku.get((c, "train"), 0) == 0]
    write_csv(out_dir / "crops_manifest.csv", list(crop_rows[0]) if crop_rows else ["crop_path"], crop_rows)
    write_csv(out_dir / "class_map.csv", list(class_map[0]), class_map)
    write_csv(out_dir / "class_map_review.csv", list(class_map[0]), [r for r in class_map if r["needs_review"]])
    write_json(out_dir / "audit_report.json", audit)
    detector_dir = out_dir / "detector_brand21"
    detector_dir.mkdir(exist_ok=True)
    (detector_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/valid", "test": "images/test",
        "nc": len(brand_names), "names": {i: n for i, n in enumerate(brand_names)}}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return audit
