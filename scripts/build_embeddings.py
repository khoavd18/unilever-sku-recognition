"""Build an inspectable NumPy cosine-similarity gallery from split manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.product_dataset import ProductDataset, build_transforms
from src.models.mobilenetv3_embedding import model_from_checkpoint
from src.utils.config import resolve_device
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--split-dir", default="outputs/splits")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/embeddings")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    split_dir = Path(args.split_dir)
    image_size = int(checkpoint["config"]["dataset"]["image_size"])
    datasets = []
    for split in args.splits:
        manifest = split_dir / f"{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        datasets.append(ProductDataset(manifest, checkpoint["class_to_idx"], build_transforms(image_size, train=False)))
    gallery = ConcatDataset(datasets)
    if not len(gallery):
        raise ValueError("Selected gallery splits contain no images.")
    loader = DataLoader(gallery, batch_size=min(args.batch_size, len(gallery)), shuffle=False, num_workers=0)
    vectors, metadata = [], []
    for images, _, paths, class_names in tqdm(loader, desc="embedding gallery"):
        vectors.append(model.extract_embedding(images.to(device)).cpu().numpy().astype(np.float32))
        start_index = len(metadata)
        metadata.extend(
            {"embedding_index": start_index + offset, "image_path": path, "class_name": name}
            for offset, (path, name) in enumerate(zip(paths, class_names))
        )
    embeddings = np.concatenate(vectors, axis=0)
    output_dir = ensure_dir(args.output_dir)
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("embedding_index", "image_path", "class_name"))
        writer.writeheader()
        writer.writerows(metadata)
    save_json(
        {
            "embedding_path": str((output_dir / "embeddings.npy").resolve()),
            "metadata_path": str((output_dir / "metadata.csv").resolve()),
            "num_images": len(metadata),
            "embedding_dim": int(embeddings.shape[1]),
            "splits": list(args.splits),
        },
        output_dir / "embedding_report.json",
    )
    print(f"Saved {len(metadata)} normalized {embeddings.shape[1]}-D embeddings to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
