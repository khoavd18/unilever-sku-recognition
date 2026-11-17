"""Manually evaluate item-code Recall@1 on validation queries."""

from __future__ import annotations

import argparse
import csv
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

from scripts.build_item_gallery import sha256_file
from src.catalog.mapping import load_mapping
from src.datasets.product_dataset import build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.retrieval.item_code import (
    GalleryIndex,
    evaluate_item_code_retrieval,
    validate_gallery_runtime_compatibility,
)
from src.training.embedding_data import path_key, read_csv
from src.utils.config import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument(
        "--mapping",
        default="artifacts/mappings/axe5_classification_100ep_s42_axe5_demo_mapping.v3.json",
    )
    parser.add_argument("--dataset-dir", default="data/axe_crops")
    parser.add_argument("--query-manifest", default=None, help="Defaults to validation; pass test only for final manual evaluation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output exists: {output}")
    gallery = GalleryIndex.load(args.gallery_dir)
    mapping = load_mapping(args.mapping)
    checkpoint_path = Path(args.checkpoint).resolve()
    mapping_by_legacy = {row["legacy_class_name"]: row for row in mapping["mappings"]}
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    preprocessing_payload = {
        "image_size": image_size,
        "resize": "fit_with_black_pad_no_stretch",
        "normalization": "ImageNet mean/std",
        "training_augmentation": False,
    }
    preprocessing_version = "sha256:" + hashlib.sha256(
        json.dumps(preprocessing_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    validate_gallery_runtime_compatibility(
        gallery.manifest,
        model_version=sha256_file(checkpoint_path),
        catalog_version=mapping["catalog_version"],
        mapping_version=mapping["mapping_version"],
        preprocessing_version=preprocessing_version,
        embedding_dimension=int(checkpoint["config"]["model"]["embedding_dim"]),
    )
    transform = build_transforms(image_size, train=False)
    dataset = Path(args.dataset_dir).resolve()
    query_manifest = Path(args.query_manifest).resolve() if args.query_manifest else dataset / "manifests" / "val.csv"
    query_rows = read_csv(query_manifest)
    crops = {path_key(row["generated_crop_path"]): row for row in read_csv(dataset / "crop_manifest.csv")}
    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for start in range(0, len(query_rows), args.batch_size):
        batch = query_rows[start : start + args.batch_size]
        tensors = []
        for row in batch:
            with Image.open(row["image_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            embeddings.append(model.extract_embedding(torch.stack(tensors).to(device)).float().cpu().numpy())
        for row in batch:
            mapped = mapping_by_legacy.get(row["class_name"])
            provenance = crops.get(path_key(row["image_path"]))
            if mapped is None or provenance is None:
                raise ValueError(f"Incomplete query mapping/provenance: {row['image_path']}")
            metadata.append({
                "item_code": mapped["item_code"],
                "image_path": str(Path(row["image_path"]).resolve()),
                "image_sha256": sha256_file(row["image_path"]),
                "family_id": provenance["family_id"],
            })
    report = evaluate_item_code_retrieval(
        np.concatenate(embeddings).astype(np.float32), metadata,
        gallery.embeddings, gallery.metadata,
    )
    report["protocol"] = {
        "query_manifest": str(query_manifest),
        "gallery_dir": str(Path(args.gallery_dir).resolve()),
        "checkpoint": str(checkpoint_path),
        "preprocessing_version": gallery.manifest["preprocessing_version"],
        "test_data_loaded": query_manifest.name.casefold() == "test.csv",
        "brand_filter": False,
    }
    output.mkdir(parents=True)
    (output / "retrieval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tables = {
        "per_item_recall.csv": [
            {"item_code": item_code, **values} for item_code, values in report["per_item"].items()
        ],
        "item_code_confusion.csv": report["confusion_table"],
        "incorrect_matches.csv": report["incorrect_matches"],
        "missing_gallery_queries.csv": report["missing_gallery_queries"],
    }
    for filename, rows in tables.items():
        if not rows:
            (output / filename).write_text("", encoding="utf-8")
            continue
        fields = list(rows[0])
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "output_dir": str(output), "query_count": report["query_count"],
        "eligible_query_count": report["eligible_query_count"],
        "micro_recall@1": report["micro_recall@1"], "macro_recall@1": report["macro_recall@1"],
        "missing_gallery_query_count": report["missing_gallery_query_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
