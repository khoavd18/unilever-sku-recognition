"""Polygon->bbox conversion + SKU gallery crops from the Roboflow 469-SKU export.

Never edits the source ZIP.  Writes a NEW versioned directory:
  labels_sku/<split>/            469-class bbox labels (converted from polygons)
  detector_brand21/              Model 1 labels (brand, training class order) + data.yaml
  gallery/<item_code>/*.jpg      Model 2 crops (one representative/group; prefers unmarked images)
  crops_manifest.csv (gallery_split = group-level 80/10/10), sku_split_counts.csv, class_map.csv, class_map_review.csv, audit_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data.sku_gallery import build_class_map, load_item_table, load_model1_brands, prepare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="Stock_Segmentation.v33-469_sku.yolov8.zip")
    p.add_argument("--xlsx", default="ITEMCODE_SKU.xlsx")
    p.add_argument("--model1-data-yaml", default="configs/datasets/model1_brand21_data.yaml")
    p.add_argument("--registry", default="configs/brands.yaml")
    p.add_argument("--output-dir", default="data/sku469_v33")
    p.add_argument("--padding", type=float, default=0.05)
    p.add_argument("--min-crop-px", type=int, default=24)
    p.add_argument("--write-detector-images", action="store_true", help="also copy images into detector_brand21/images")
    p.add_argument("--detector-originals-only", action="store_true")
    p.add_argument("--strict-class-map", action="store_true", help="only exact class-label matches go to the gallery")
    p.add_argument("--limit-images", type=int, default=None, help="per split, for smoke tests")
    a = p.parse_args()
    r = lambda s: (ROOT / s) if not Path(s).is_absolute() else Path(s)

    import zipfile
    with zipfile.ZipFile(r(a.zip)) as z:
        class_names = yaml.safe_load(z.read("data.yaml"))["names"]
    brand_names, lookup = load_model1_brands(r(a.model1_data_yaml), r(a.registry))
    class_map = build_class_map(class_names, load_item_table(r(a.xlsx)), lookup, brand_names)
    audit = prepare(r(a.zip), class_map, brand_names, r(a.output_dir), padding=a.padding, min_crop_px=a.min_crop_px,
                    write_detector_images=a.write_detector_images, detector_originals_only=a.detector_originals_only,
                    strict_class_map=a.strict_class_map, limit_images=a.limit_images)
    keys = ("splits", "groups_total", "groups_in_multiple_splits", "classes_without_item_code", "classes_without_brand",
            "classes_with_zero_crops", "crops_written", "crop_source_groups", "gallery_split_crops", "crops", "invalid_reasons", "brand_instances")
    print(json.dumps({k: audit[k] for k in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
