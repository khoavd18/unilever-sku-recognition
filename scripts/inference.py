"""Retrieve the closest product images using a saved NumPy embedding gallery."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.product_dataset import build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.utils.config import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Cropped product image to search for.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--embeddings", default="outputs/embeddings/embeddings.npy")
    parser.add_argument("--metadata", default="outputs/embeddings/metadata.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Query image does not exist: {image_path}")
    embeddings = np.load(args.embeddings).astype(np.float32)
    with Path(args.metadata).open("r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if len(metadata) != len(embeddings):
        raise ValueError("metadata.csv row count does not match embeddings.npy")
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    transform = build_transforms(int(checkpoint["config"]["dataset"]["image_size"]), train=False)
    with Image.open(image_path) as image:
        query = transform(image.convert("RGB")).unsqueeze(0).to(device)
    query_embedding = model.extract_embedding(query).cpu().numpy()[0]
    query_embedding /= max(np.linalg.norm(query_embedding), 1e-12)
    gallery = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    scores = gallery @ query_embedding
    query_resolved = str(image_path.resolve())
    for index, row in enumerate(metadata):
        if str(Path(row["image_path"]).resolve()) == query_resolved:
            scores[index] = -np.inf
    top_k = min(max(1, args.top_k), len(metadata))
    top_indices = np.argsort(-scores)[:top_k]
    print(f"Query: {image_path}")
    print(f"Top {top_k}:")
    for rank, index in enumerate(top_indices, start=1):
        row = metadata[int(index)]
        print(f"{rank}. Product: {row['class_name']}\n   Similarity: {scores[index]:.3f}\n   Image: {row['image_path']}")


if __name__ == "__main__":
    main()
