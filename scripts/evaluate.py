"""Evaluate classification and image-to-image retrieval on test CSV manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.product_dataset import ProductDataset, build_transforms
from src.evaluation.metrics import classification_accuracy, confusion_matrix, retrieval_recall_at_k
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.utils.config import resolve_device
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    split_dir = Path(args.split_dir)
    test_manifest = split_dir / "test.csv"
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    dataset = ProductDataset(test_manifest, checkpoint["class_to_idx"], build_transforms(image_size, train=False))
    if not dataset:
        raise ValueError("Test manifest is empty.")
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0)
    logits_list, targets_list, embeddings, paths, names = [], [], [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for images, targets, image_paths, class_names in tqdm(loader, desc="evaluate"):
        images, targets = images.to(device), targets.to(device)
        logits, batch_embeddings = model(images)
        total_loss += criterion(logits, targets).item() * len(targets)
        logits_list.append(logits.cpu())
        targets_list.append(targets.cpu())
        embeddings.append(batch_embeddings.cpu())
        paths.extend(image_paths)
        names.extend(class_names)
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    embedding_matrix = torch.cat(embeddings)
    accuracy = classification_accuracy(logits, targets, topk=(1, 5))
    predicted = logits.argmax(dim=1).numpy()
    matrix = confusion_matrix(targets.numpy(), predicted, len(checkpoint["class_to_idx"]))
    retrieval = retrieval_recall_at_k(embedding_matrix, names, embedding_matrix, names, (1, 5, 10), paths, paths)
    output_dir = ensure_dir(args.output_dir)
    np.save(output_dir / "confusion_matrix.npy", matrix)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_images": len(dataset),
        "test_loss": total_loss / len(dataset),
        "top1_accuracy": accuracy["top1"],
        "top5_accuracy": accuracy["top5"],
        "retrieval": retrieval,
        "confusion_matrix": str((output_dir / "confusion_matrix.npy").resolve()),
    }
    if retrieval["eligible_queries"] == 0:
        report["retrieval_warning"] = (
            "No test query has another image of the same class in the test gallery; Recall@K is not meaningful. "
            "Use more test images per class or evaluate against a separate gallery."
        )
    save_json(report, output_dir / "evaluation_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
