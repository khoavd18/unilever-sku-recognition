"""Build a traceable item-code embedding gallery from a selected checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.catalog import load_catalog, require_usable_catalog
from src.catalog.mapping import load_mapping
from src.datasets.product_dataset import build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.retrieval.item_code import validate_gallery
from src.retrieval.compatibility import embedding_preprocessing, preprocessing_version
from src.training.embedding_data import path_key, read_csv
from src.utils.config import resolve_device


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def select_gallery_rows(rows: list[dict[str, str]], max_per_item: int | None) -> list[dict[str, str]]:
    """Keep diverse, deterministic gallery representatives per SKU.

    An unbounded gallery preserves every train crop.  A bounded deployment
    gallery first takes one image from each source family, prefers original
    images over augmented ones, then fills remaining slots deterministically.
    """

    if max_per_item is None:
        return rows
    if max_per_item < 1:
        raise ValueError("max_per_item must be at least 1")
    selected: list[dict[str, str]] = []
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["class_name"], []).append(row)
    for class_name in sorted(by_class):
        candidates = sorted(
            by_class[class_name],
            key=lambda row: (
                str(row.get("source_is_augmented", "")).strip().casefold() in {"true", "1", "yes"},
                row.get("family_id", ""),
                row["image_path"],
            ),
        )
        picked: list[dict[str, str]] = []
        seen_families: set[str] = set()
        for row in candidates:
            family = row.get("family_id", "")
            if family not in seen_families:
                picked.append(row)
                seen_families.add(family)
            if len(picked) == max_per_item:
                break
        if len(picked) < max_per_item:
            already = {row["image_path"] for row in picked}
            picked.extend(row for row in candidates if row["image_path"] not in already)
        selected.extend(picked[:max_per_item])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog", default="artifacts/catalog/axe5_demo_v3/product_catalog.json")
    parser.add_argument(
        "--mapping",
        default="artifacts/mappings/axe5_classification_100ep_s42_axe5_demo_mapping.v3.json",
    )
    parser.add_argument("--dataset-dir", default="data/axe_crops")
    parser.add_argument("--manifest", default=None, help="Defaults to <dataset-dir>/manifests/train.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-images-per-item", type=int, default=None,
        help="Bound a deployment gallery while preserving diverse source families.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Gallery output exists; choose a new version directory: {output}")
    checkpoint_path = Path(args.checkpoint).resolve()
    catalog = load_catalog(args.catalog)
    require_usable_catalog(catalog, require_scope=True)
    mapping = load_mapping(args.mapping)
    if mapping["catalog_version"] != catalog["catalog_version"]:
        raise ValueError("Mapping/catalog versions differ")
    mapping_by_legacy = {row["legacy_class_name"]: row for row in mapping["mappings"]}
    catalog_by_code = {row["item_code"]: row for row in catalog["records"]}
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    checkpoint_order = [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda p: p[1])]
    mapping_order = [row["legacy_class_name"] for row in mapping["mappings"]]
    if checkpoint_order != mapping_order:
        raise ValueError("Checkpoint class order differs from the verified mapping")

    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else dataset_dir / "manifests" / "train.csv"
    manifest_rows = select_gallery_rows(read_csv(manifest_path), args.max_images_per_item)
    crop_rows = read_csv(dataset_dir / "crop_manifest.csv")
    crops = {path_key(row["generated_crop_path"]): row for row in crop_rows}
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    transform = build_transforms(image_size, train=False)
    checkpoint_version = sha256_file(checkpoint_path)
    preprocessing_payload = embedding_preprocessing(image_size)
    preprocessing_fingerprint = preprocessing_version(preprocessing_payload)
    all_embeddings: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    model.eval()
    for start in range(0, len(manifest_rows), args.batch_size):
        batch_rows = manifest_rows[start : start + args.batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            embeddings = model.extract_embedding(torch.stack(tensors).to(device)).float().cpu().numpy()
        all_embeddings.append(embeddings)
        for row in batch_rows:
            legacy = row["class_name"]
            if legacy not in mapping_by_legacy:
                raise ValueError(f"Gallery class absent from verified mapping: {legacy}")
            mapped = mapping_by_legacy[legacy]
            product = catalog_by_code[mapped["item_code"]]
            provenance = crops.get(path_key(row["image_path"]))
            if provenance is None:
                raise ValueError(f"Gallery image missing crop provenance: {row['image_path']}")
            metadata.append({
                "gallery_index": len(metadata),
                "item_code": product["item_code"],
                "item_name": product["item_name"],
                "brand": product["brand"],
                "category": product["category"],
                "class_name": product["class_name"],
                "legacy_class_name": legacy,
                "image_path": str(Path(row["image_path"]).resolve()),
                "image_sha256": sha256_file(row["image_path"]),
                "family_id": provenance["family_id"],
                "source_image_path": provenance.get("source_image_path", provenance["source_image_id"]),
                "source_image_id": provenance["source_image_id"],
                "source_annotation_id": provenance["source_annotation_id"],
                "source_zip": provenance["source_zip"],
                "model_version": checkpoint_version,
                "catalog_version": catalog["catalog_version"],
                "mapping_version": mapping["mapping_version"],
                "preprocessing_version": preprocessing_fingerprint,
            })
    matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    gallery_manifest = {
        "schema_version": 1,
        "entry_count": len(metadata),
        "item_count": len({row["item_code"] for row in metadata}),
        "embedding_dimension": int(matrix.shape[1]),
        "l2_normalized": True,
        "similarity": "cosine",
        "model_version": checkpoint_version,
        "catalog_version": catalog["catalog_version"],
        "catalog_validation_status": catalog.get("status"),
        "catalog_validation_summary": catalog.get("summary"),
        "mapping_version": mapping["mapping_version"],
        "preprocessing_version": preprocessing_fingerprint,
        "preprocessing": preprocessing_payload,
        "source_manifest": str(manifest_path),
        "max_images_per_item": args.max_images_per_item,
        "embeddings_file": "embeddings.npy",
        "metadata_file": "metadata.jsonl",
        "full_gallery_search": True,
        "brand_filter": False,
        "cosine_is_calibrated_probability": False,
    }
    validate_gallery(matrix, metadata, gallery_manifest)
    output.mkdir(parents=True)
    np.save(output / "embeddings.npy", matrix, allow_pickle=False)
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "gallery_manifest.json").write_text(
        json.dumps(gallery_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **gallery_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
