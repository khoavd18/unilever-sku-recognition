"""Manually evaluate one selected AXE checkpoint on test queries with a train-only gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_clean_axe_dataset import EXPECTED_SKUS
from src.datasets.product_dataset import ProductDataset, build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.training.axe_data import make_loader, path_key, read_csv
from src.training.retrieval_trainer import evaluate_retrieval_protocol, write_retrieval_artifacts
from src.utils.config import resolve_device
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="data/axe_crops")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    crops_root = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Test output already exists; choose a new --output-dir: {output_dir}")
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    if set(class_to_idx) != set(EXPECTED_SKUS):
        raise ValueError("Selected checkpoint is not the verified five-SKU AXE model")
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    image_size = int(checkpoint["config"]["dataset"]["image_size"])

    gallery_dataset = ProductDataset(
        crops_root / "manifests" / "train.csv",
        class_to_idx,
        build_transforms(image_size, train=False),
    )
    test_dataset = ProductDataset(
        crops_root / "manifests" / "test.csv",
        class_to_idx,
        build_transforms(image_size, train=False),
    )
    crop_rows = read_csv(crops_root / "crop_manifest.csv")
    crop_by_path = {path_key(row["generated_crop_path"]): row for row in crop_rows}
    gallery_families = {
        crop_by_path[path_key(row["image_path"])]["family_id"] for row in gallery_dataset.samples
    }
    test_families = {
        crop_by_path[path_key(row["image_path"])]["family_id"] for row in test_dataset.samples
    }
    overlap = gallery_families & test_families
    if overlap:
        raise ValueError(f"Test source families occur in the training gallery: {sorted(overlap)[:10]}")

    gallery_loader = make_loader(gallery_dataset, 64, 0, False, device, 42)
    test_loader = make_loader(test_dataset, 64, 0, False, device, 42)
    metrics = evaluate_retrieval_protocol(
        model,
        gallery_loader,
        test_loader,
        class_names,
        device,
        bool(checkpoint["config"]["training"].get("amp", True)) and device.type == "cuda",
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "selection_metric_recorded": checkpoint.get("selection_metric"),
            "gallery_split": "train",
            "query_split": "test",
            "gallery_source_family_count": len(gallery_families),
            "test_source_family_count": len(test_families),
            "source_family_overlap": 0,
        }
    )
    ensure_dir(output_dir)
    save_json(metrics, output_dir / "test_metrics.json")
    write_retrieval_artifacts(metrics, output_dir, prefix="test")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
