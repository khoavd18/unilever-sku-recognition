"""Build the versioned 459-SKU catalog, mapping, and embedding gallery for runtime.

This command writes new artifacts only.  It derives the deployment catalog from
the selected checkpoint's exact class labels, so a catalog row cannot silently
be substituted for a different SKU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_item_gallery import main as build_gallery
from src.catalog.catalog import create_scoped_catalog, load_catalog, write_catalog
from src.catalog.mapping import build_exact_catalog_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-catalog", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-images-per-item", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(f"Runtime asset output exists; choose a new directory: {root}")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Checkpoint has no class_to_idx metadata")

    full_catalog = load_catalog(args.full_catalog)
    by_legacy = {row.get("legacy_class_name"): row for row in full_catalog["records"]}
    missing = [name for name in class_to_idx if name not in by_legacy]
    if missing:
        raise ValueError(f"Checkpoint classes are absent from full catalog: {missing[:10]}")
    item_codes = [by_legacy[name]["item_code"] for name, _ in sorted(class_to_idx.items(), key=lambda pair: pair[1])]
    scoped_catalog = create_scoped_catalog(full_catalog, item_codes, "sku459_runtime")
    if scoped_catalog["summary"]["errors"]:
        raise ValueError(f"Scoped runtime catalog is invalid: {scoped_catalog['summary']}")
    mapping = build_exact_catalog_mapping(scoped_catalog, class_to_idx, checkpoint_path)
    if mapping["status"] != "verified":
        raise ValueError(f"Runtime mapping is invalid: {mapping['issues'][:5]}")

    catalog_dir = root / "catalog"
    mapping_path = root / "mapping.json"
    gallery_dir = root / "gallery"
    write_catalog(scoped_catalog, catalog_dir)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    original_argv = sys.argv
    try:
        sys.argv = [
            "build_item_gallery.py",
            "--checkpoint", str(checkpoint_path),
            "--catalog", str(catalog_dir / "product_catalog.json"),
            "--mapping", str(mapping_path),
            "--dataset-dir", str(Path(args.dataset_dir).resolve()),
            "--manifest", str(Path(args.manifest).resolve()),
            "--output-dir", str(gallery_dir),
            "--batch-size", str(args.batch_size),
            "--max-images-per-item", str(args.max_images_per_item),
            "--device", args.device,
        ]
        build_gallery()
    finally:
        sys.argv = original_argv

    print(json.dumps({
        "runtime_assets": str(root),
        "sku_count": len(item_codes),
        "catalog": str(catalog_dir / "product_catalog.json"),
        "mapping": str(mapping_path),
        "gallery": str(gallery_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
