"""End-to-end non-destructive smoke test using temporary images under outputs/."""

from __future__ import annotations

import json
import copy
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.product_dataset import ProductDataset, build_transforms
from src.evaluation.metrics import retrieval_recall_at_k
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier, model_from_checkpoint
from src.training.trainer import Trainer
from src.utils.config import resolve_device, set_seed
from src.utils.io import ensure_dir, save_json


def create_temporary_product_images(source_dir: Path) -> None:
    """Create four visually distinct mini product classes only for this smoke test."""
    palette = [(210, 50, 45), (35, 150, 70), (35, 100, 210), (170, 85, 185)]
    for class_index, color in enumerate(palette):
        class_dir = ensure_dir(source_dir / f"smoke_product_{class_index:02d}")
        for image_index in range(4):
            image = Image.new("RGB", (96, 96), tuple(min(255, value + image_index * 4) for value in color))
            draw = ImageDraw.Draw(image)
            offset = 12 + image_index * 2
            draw.rectangle((offset, offset, 96 - offset, 96 - offset), outline=(255, 255, 255), width=4)
            draw.text((34, 40), str(class_index), fill=(255, 255, 0))
            image.save(class_dir / f"image_{image_index:02d}.png")


@torch.inference_mode()
def verify_embeddings(model, datasets, device: torch.device) -> dict[str, object]:
    loader = DataLoader(ConcatDataset(datasets), batch_size=8, shuffle=False, num_workers=0)
    vectors, labels, paths = [], [], []
    for images, _, image_paths, class_names in loader:
        embeddings = model.extract_embedding(images.to(device)).cpu()
        vectors.append(embeddings)
        labels.extend(class_names)
        paths.extend(image_paths)
    matrix = torch.cat(vectors)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        raise AssertionError("Embeddings are not L2 normalized")
    retrieval = retrieval_recall_at_k(matrix, labels, matrix, labels, (1, 5, 10), paths, paths)
    return {"embedding_shape": list(matrix.shape), "min_l2_norm": float(norms.min()), "max_l2_norm": float(norms.max()), "retrieval": retrieval}


def main() -> None:
    work_dir = ensure_dir(ROOT / "outputs" / "smoke_test")
    source_dir = ensure_dir(work_dir / "source_dataset")
    split_dir = ensure_dir(work_dir / "splits")
    checkpoint_dir = ensure_dir(work_dir / "checkpoints")
    create_temporary_product_images(source_dir)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            "--dataset-path",
            str(source_dir),
            "--output-dir",
            str(split_dir),
            "--seed",
            "42",
        ],
        check=True,
    )
    set_seed(42)
    device = resolve_device("auto")
    class_to_idx = {f"smoke_product_{index:02d}": index for index in range(4)}
    train_data = ProductDataset(split_dir / "train.csv", class_to_idx, build_transforms(64, train=True))
    val_data = ProductDataset(split_dir / "val.csv", class_to_idx, build_transforms(64, train=False))
    test_data = ProductDataset(split_dir / "test.csv", class_to_idx, build_transforms(64, train=False))
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)
    config = {
        "dataset": {"image_size": 64, "seed": 42},
        "model": {"embedding_dim": 32, "dropout": 0.1, "pretrained": False},
        "training": {
            "batch_size": 4,
            "num_epochs": 2,
            "head_epochs": 1,
            "learning_rate": 1e-3,
            "backbone_learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "early_stopping_patience": 2,
            "amp": True,
        },
        "metric_learning": {"enabled": False, "loss_weight": 0.1, "margin": 0.2},
        "output": {"checkpoint_dir": str(checkpoint_dir), "output_dir": str(work_dir)},
    }
    model = MobileNetV3EmbeddingClassifier(num_classes=len(class_to_idx), embedding_dim=32, dropout=0.1, pretrained=False).to(device)
    result = Trainer(model, config, class_to_idx, device).fit(train_loader, val_loader)
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if not best_checkpoint.is_file():
        raise AssertionError("Training did not create best_model.pth")
    resume_config = copy.deepcopy(config)
    resume_config["training"]["num_epochs"] = 3
    resumed_model = MobileNetV3EmbeddingClassifier(
        num_classes=len(class_to_idx), embedding_dim=32, dropout=0.1, pretrained=False
    ).to(device)
    resumed_result = Trainer(resumed_model, resume_config, class_to_idx, device).fit(
        train_loader, val_loader, resume_path=checkpoint_dir / "last_model.pth"
    )
    restored_model, _ = model_from_checkpoint(best_checkpoint, device)
    embedding_result = verify_embeddings(restored_model, [train_data, val_data, test_data], device)
    report = {
        "status": "pass",
        "device": str(device),
        "train_images": len(train_data),
        "val_images": len(val_data),
        "test_images": len(test_data),
        "checkpoint": str(best_checkpoint),
        "training": result,
        "resumed_training": resumed_result,
        **embedding_result,
    }
    save_json(report, work_dir / "smoke_test_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
