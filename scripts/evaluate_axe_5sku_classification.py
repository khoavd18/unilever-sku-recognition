"""Manually evaluate a selected five-SKU AXE checkpoint using classification metrics only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_clean_axe_dataset import EXPECTED_SKUS
from src.datasets.product_dataset import ProductDataset, build_transforms
from src.evaluation.metrics import classification_metrics_from_predictions
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.utils.config import resolve_device
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="data/axe_crops")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    if set(class_to_idx) != set(EXPECTED_SKUS):
        raise ValueError("Checkpoint class mapping is not the verified five-SKU AXE mapping")
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    dataset = ProductDataset(
        Path(args.dataset_dir).resolve() / "manifests" / "test.csv",
        class_to_idx,
        build_transforms(image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(args.batch_size), len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss()
    amp_enabled = bool(checkpoint["config"]["training"].get("amp", True)) and device.type == "cuda"
    total_loss = 0.0
    sample_count = 0
    predictions: list[int] = []
    targets_all: list[int] = []
    model.eval()
    for images, targets, _, _ in tqdm(loader, desc="classification test", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits, _ = model(images)
            loss = criterion(logits, targets)
        total_loss += float(loss.item()) * len(targets)
        sample_count += len(targets)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    metrics = classification_metrics_from_predictions(targets_all, predictions, class_names)
    metrics.update(
        {
            "ce_loss": total_loss / sample_count,
            "checkpoint": str(checkpoint_path),
            "query_split": "test",
            "classification_only": True,
        }
    )
    ensure_dir(output_dir)
    save_json(metrics, output_dir / "test_classification_metrics.json")
    with (output_dir / "test_per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ("sku", "precision", "recall", "f1", "support", "correct")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sku, values in metrics["per_class"].items():
            writer.writerow({"sku": sku, **values})
    with (output_dir / "test_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_sku", *class_names])
        for sku, row in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([sku, *row])
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
