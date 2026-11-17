"""Manually launch one isolated five-SKU AXE training run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.training.axe_data import (
    audit_training_readiness,
    build_classification_loaders,
    build_training_loaders,
)
from src.training.classification_trainer import ClassificationTrainer
from src.training.retrieval_trainer import RetrievalTrainer, state_dict_fingerprint
from src.utils.config import load_config, resolve_device, resolve_path, set_seed
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/axe_5sku.yaml")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique short run name; defaults to a timestamp. Existing runs are never overwritten.",
    )
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cpu.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Explicit classification-only last_model.pth to resume; never enabled automatically.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate config and train/validation readiness without creating a run or training.",
    )
    return parser.parse_args()


def validate_manual_config(config: dict) -> None:
    training = config["training"]
    model = config["model"]
    metric = config["metric_learning"]
    mode = str(training.get("mode", "retrieval"))
    if mode == "classification_only":
        if int(training["num_epochs"]) != 100:
            raise ValueError("The classification-only experiment requires exactly 100 total epochs")
        if bool(training.get("early_stopping_enabled", True)):
            raise ValueError("Early stopping must be explicitly disabled for the 100-epoch experiment")
        if training.get("early_stopping_patience") is not None:
            raise ValueError("early_stopping_patience must be null when early stopping is disabled")
        if str(training.get("checkpoint_metric")) != "val_macro_f1_then_val_ce_loss":
            raise ValueError("Classification checkpoints must use Macro F1 then validation CE loss")
    else:
        if int(training["num_epochs"]) > 15:
            raise ValueError("AXE retrieval runs are capped at 15 total epochs")
        if str(training.get("checkpoint_metric")) != "val_retrieval_r1_macro":
            raise ValueError("Best retrieval checkpoints must use val_retrieval_r1_macro")
    if int(training["head_epochs"]) != 3:
        raise ValueError("AXE baseline requires exactly 3 head-only epochs")
    if training.get("resume"):
        raise ValueError("Config-based automatic resume is forbidden; use explicit --resume")
    if not bool(model.get("pretrained", True)):
        raise ValueError("AXE manual runs must start from ImageNet pretrained weights")
    if bool(metric.get("enabled", False)) or bool(metric.get("triplet_loss", False)):
        raise ValueError("The controlled AXE baseline uses CrossEntropy only")
    if str(training.get("checkpoint_tie")) != "keep_earlier":
        raise ValueError("Equal checkpoint scores must keep the earlier checkpoint")


def safe_run_name(value: str | None) -> str:
    value = value or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", value):
        raise ValueError("--run-name must be 1-48 characters using letters, digits, '_' or '-'")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_manual_config(config)
    run_name = safe_run_name(args.run_name)
    crops_root = resolve_path(config["dataset"]["path"])
    clean_root = resolve_path(config["dataset"]["source_coco_path"])
    split_dir = resolve_path(config["dataset"]["split_dir"])

    # This read-only preflight deliberately opens only train/validation images.
    readiness = audit_training_readiness(crops_root, clean_root)
    class_to_idx = readiness["class_to_idx"]
    output_dir = resolve_path(config["output"]["output_root"]) / run_name
    checkpoint_dir = resolve_path(config["output"]["checkpoint_root"]) / run_name
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is None:
        if output_dir.exists() or checkpoint_dir.exists():
            raise FileExistsError(
                f"Run name {run_name!r} already exists. Choose another --run-name; nothing was overwritten."
            )
    else:
        if str(config["training"].get("mode", "retrieval")) != "classification_only":
            raise ValueError("Explicit resume is currently supported only for classification-only runs")
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint was not found: {resume_path}")
        if not output_dir.is_dir() or not checkpoint_dir.is_dir():
            raise FileNotFoundError("Resume requires the existing matching output and checkpoint run directories")

    requested_device = args.device or config["training"].get("device", "auto")
    device = resolve_device(requested_device)
    if args.preflight:
        print(
            json.dumps(
                {
                    "config": config["_config_path"],
                    "run_name": run_name,
                    "mode": config["training"].get("mode", "retrieval"),
                    "device": str(device),
                    "class_to_idx": class_to_idx,
                    "crop_counts": readiness["crop_counts"],
                    "source_family_counts": readiness["source_family_counts"],
                    "dataset_fingerprint": readiness["dataset_fingerprint"],
                    "output_dir": str(output_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                    "test_queries_loaded": False,
                    "training_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    seed = int(config["dataset"].get("seed", 42))
    set_seed(seed)
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    config["output"]["output_dir"] = str(output_dir.resolve())
    config["output"]["checkpoint_dir"] = str(checkpoint_dir.resolve())
    config["runtime"] = {
        "run_name": run_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "class_to_idx": class_to_idx,
        "train_crops": readiness["crop_counts"]["train"],
        "validation_crops": readiness["crop_counts"]["val"],
        "validation_source_families": readiness["source_family_counts"]["val"],
        "dataset_fingerprint": readiness["dataset_fingerprint"],
        "test_queries_loaded": False,
    }
    if resume_path is None:
        with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        save_json(readiness, output_dir / "readiness.json")
        save_json(config["runtime"], output_dir / "run_identity.json")
        save_json(config["runtime"], checkpoint_dir / "run_identity.json")

    print(f"Device: {device} | GPU: {config['runtime']['gpu_name']}", flush=True)
    print(f"Class mapping: {json.dumps(class_to_idx, ensure_ascii=False)}", flush=True)
    print(
        f"Crops: train={readiness['crop_counts']['train']} | "
        f"validation={readiness['crop_counts']['val']} | "
        f"validation families={readiness['source_family_counts']['val']}",
        flush=True,
    )
    print(f"Dataset: {crops_root}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print("Effective config:", flush=True)
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), flush=True)
    print(
        "Metric note: train_ce_loss/train_cls_top1 use training augmentation and training-mode "
        "behavior; they are not directly identical to deterministic evaluation-mode measurements.",
        flush=True,
    )

    model = MobileNetV3EmbeddingClassifier(
        num_classes=len(class_to_idx),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.2)),
        pretrained=resume_path is None,
    )
    initialization_fingerprint = state_dict_fingerprint(model)
    if str(config["training"].get("mode", "retrieval")) == "classification_only":
        train_loader, validation_loader = build_classification_loaders(
            split_dir, class_to_idx, config, device
        )
        trainer = ClassificationTrainer(
            model,
            config,
            class_to_idx,
            device,
            output_dir,
            checkpoint_dir,
            readiness["dataset_fingerprint"],
            initialization_fingerprint,
        )
        result = trainer.fit(train_loader, validation_loader, resume_path=resume_path)
        print(
            f"Manual classification run {result['status']}. Best epoch="
            f"{result['best_checkpoint']['epoch']} | validation Macro F1="
            f"{100.0 * result['best_checkpoint']['macro_f1']:.2f}%",
            flush=True,
        )
    else:
        train_loader, gallery_loader, validation_loader = build_training_loaders(
            split_dir, class_to_idx, config, device
        )
        trainer = RetrievalTrainer(
            model,
            config,
            class_to_idx,
            device,
            output_dir,
            initialization_fingerprint,
            readiness["dataset_fingerprint"],
            checkpoint_dir=checkpoint_dir,
        )
        result = trainer.fit(train_loader, gallery_loader, validation_loader)
        print(
            f"Manual run complete. Best epoch={result['best_epoch']} | "
            f"val_retrieval_r1_macro={100.0 * result['best_validation_macro_recall@1']:.2f}%",
            flush=True,
        )
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pth'}", flush=True)
    print(f"Metrics: {output_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
