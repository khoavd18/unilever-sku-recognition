"""Plan and manually run MobileNetV3 classification/embedding training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader

from src.datasets.product_dataset import ClassBalancedBatchSampler, ProductDataset, build_transforms, read_manifest
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.training.trainer import Trainer
from src.utils.config import load_config, resolve_device, resolve_path, set_seed
from src.utils.io import ensure_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="YAML configuration path.")
    parser.add_argument("--split-dir", default=None, help="Directory containing train.csv and val.csv.")
    parser.add_argument("--resume", default=None, help="Continue from a last_model.pth checkpoint.")
    parser.add_argument("--run-name", default=None, help="Unique run name under the configured output roots.")
    parser.add_argument("--device", default=None, help="Override the configured device.")
    parser.add_argument("--preflight", action="store_true", help="Check train/validation without training or downloading weights.")
    return parser.parse_args(argv)


def classes_from_manifests(split_dir: Path) -> dict[str, int]:
    """Derive training classes without opening the held-out test manifest."""

    names: set[str] = set()
    for split in ("train", "val"):
        manifest = split_dir / f"{split}.csv"
        if manifest.exists():
            names.update(row["class_name"] for row in read_manifest(manifest))
    if not names:
        raise ValueError(f"No classes found in CSV manifests under {split_dir}")
    return {name: index for index, name in enumerate(sorted(names))}


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device_type: str, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device_type == "cuda",
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


def build_training_plan(
    config_path: str | Path = "configs/config.yaml",
    split_dir_override: str | Path | None = None,
    *,
    run_name: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Read and validate a training plan without creating outputs or training."""

    config = load_config(config_path)
    dataset_config = config["dataset"]
    split_dir = resolve_path(split_dir_override or dataset_config["split_dir"])
    manifests = {name: split_dir / f"{name}.csv" for name in ("train", "val")}
    for path in manifests.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. For a COCO export, run scripts/prepare_coco_crops.py "
                "first; for a product-level dataset, use scripts/split_dataset.py."
            )
    rows = {name: read_manifest(path) for name, path in manifests.items()}
    class_to_idx = classes_from_manifests(split_dir)
    if dataset_config.get("class_mapping"):
        mapping = json.loads(resolve_path(dataset_config["class_mapping"]).read_text(encoding="utf-8"))["class_to_idx"]
        if (set(mapping) != set(class_to_idx) or sorted(mapping.values()) != list(range(len(mapping)))):
            raise ValueError("Configured class mapping differs from train/validation classes")
        class_to_idx = mapping
    if dataset_config.get("require_all_classes", False):
        for split, members in rows.items():
            missing = set(class_to_idx) - {row["class_name"] for row in members}
            if missing:
                raise ValueError(f"{split} is missing configured SKU classes: {sorted(missing)}")
    if dataset_config.get("require_group_disjoint", False):
        paths, groups, hashes = {}, {}, {}
        crop_root = resolve_path(dataset_config["crop_root"])
        for split, members in rows.items():
            required = {"group_id", "family_id", "image_sha256"}
            if not required.issubset(members[0]):
                raise ValueError(f"{split} lacks group/hash provenance")
            for row in members:
                if row["split"] != split or any(not row[key] for key in required):
                    raise ValueError(f"Invalid split/provenance in {manifests[split]}")
                image = Path(row["image_path"]).resolve()
                if not image.is_relative_to(crop_root) or not image.is_file():
                    raise ValueError(f"Crop missing or outside configured crop_root: {image}")
                key = str(image).casefold()
                if key in paths:
                    raise ValueError(f"Duplicate crop path: {image}")
                paths[key] = split
                # Both original groups and duplicate-linked components must be disjoint.
                for group in ("group:" + row["group_id"], "family:" + row["family_id"]):
                    if group in groups and groups[group] != split:
                        raise ValueError(f"Source group crosses training/validation: {group}")
                    groups[group] = split
                digest = row["image_sha256"]
                if digest in hashes and hashes[digest] != split:
                    raise ValueError("Identical crop hash crosses training/validation")
                hashes[digest] = split
    dataset_fingerprint = None
    if dataset_config.get("preparation_report"):
        report = json.loads(resolve_path(dataset_config["preparation_report"]).read_text(encoding="utf-8"))
        if report.get("status") != "ready":
            raise ValueError("Prepared SKU dataset is not ready")
        checked_hashes = []
        for name, path in (("class_mapping.json", resolve_path(dataset_config["class_mapping"])),
                           ("manifests/train.csv", manifests["train"]), ("manifests/val.csv", manifests["val"])):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != report["manifest_sha256"].get(name):
                raise ValueError(f"Prepared training metadata changed: {path}")
            checked_hashes.append(digest)
        dataset_fingerprint = hashlib.sha256("\n".join(checked_hashes).encode()).hexdigest()
        if dataset_fingerprint != report["training_fingerprint"]:
            raise ValueError("Training dataset fingerprint differs from preparation report")
    if config["output"].get("require_run_name", False) and run_name is None:
        raise ValueError("This config requires an explicit unique --run-name")
    if run_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
            raise ValueError("Run name must use 1-64 letters, digits, '_' or '-'")
        for key in ("checkpoint_dir", "output_dir"):
            config["output"][key] = str(resolve_path(config["output"][key]) / run_name)
    if device_override is not None:
        config["training"]["device"] = device_override
    counts = {name: len(members) for name, members in rows.items()}
    return {
        "config": config,
        "config_path": config["_config_path"],
        "split_dir": split_dir,
        "manifests": manifests,
        "class_to_idx": class_to_idx,
        "counts": counts,
        "run_name": run_name,
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_split": "val",
        "test_manifest_loaded": False,
        "training_started": False,
    }


def validate_run_destination(plan: dict[str, Any], resume_path: str | Path | None) -> str | Path | None:
    """Refuse accidental overwrite and require resume to match this run/dataset."""
    if plan["run_name"] is None:
        return resume_path
    config = plan["config"]
    output_paths = [resolve_path(config["output"][key]) for key in ("checkpoint_dir", "output_dir")]
    if resume_path:
        resume_file = resolve_path(resume_path)
        if resume_file != output_paths[0] / "last_model.pth" or not all(path.is_dir() for path in output_paths):
            raise ValueError("Resume must use last_model.pth from the same existing run")
        import torch

        previous = torch.load(resume_file, map_location="cpu", weights_only=False)
        if previous.get("config", {}).get("runtime", {}).get("dataset_fingerprint") != plan["dataset_fingerprint"]:
            raise ValueError("Resume checkpoint belongs to a different prepared dataset")
        return resume_file
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"Run {plan['run_name']!r} already exists; choose a new --run-name")
    return None


def run_training(
    config_path: str | Path,
    *,
    split_dir_override: str | Path | None = None,
    resume_override: str | Path | None = None,
    run_name: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Start training explicitly; importing/planning this module never trains."""

    plan = build_training_plan(config_path, split_dir_override, run_name=run_name, device_override=device_override)
    config = plan["config"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    resume_path = resume_override or training_config.get("resume")
    resume_path = validate_run_destination(plan, resume_path)
    split_dir = plan["split_dir"]
    set_seed(int(dataset_config.get("seed", 42)))
    device = resolve_device(training_config.get("device", "auto"))
    class_to_idx = plan["class_to_idx"]
    image_size = int(dataset_config["image_size"])
    augmentation_config = config.get("augmentation", {})
    train_dataset = ProductDataset(
        split_dir / "train.csv",
        class_to_idx,
        build_transforms(
            image_size,
            train=True,
            horizontal_flip=bool(augmentation_config.get("horizontal_flip", False)),
            rotation_degrees=float(augmentation_config.get("rotation_degrees", 5.0)),
        ),
    )
    val_dataset = ProductDataset(split_dir / "val.csv", class_to_idx, build_transforms(image_size, train=False))
    if len(train_dataset) < 2:
        raise ValueError("Training needs at least two training images because the embedding head uses BatchNorm.")
    if not len(val_dataset):
        raise ValueError("Validation manifest is empty. Add images or adjust the source dataset/split.")
    configured_batch_size = int(training_config["batch_size"])
    batch_size = min(configured_batch_size, len(train_dataset))
    if batch_size == 1:
        raise ValueError("Effective batch size is one; add data or lower the risk by using a larger dataset.")
    workers = int(dataset_config.get("num_workers", 0))
    metric_config = config["metric_learning"]
    if metric_config.get("enabled", False):
        sampler = ClassBalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            samples_per_class=int(metric_config.get("samples_per_class", 2)),
            seed=int(dataset_config.get("seed", 42)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
    else:
        drop_last = len(train_dataset) % batch_size == 1
        train_loader = make_loader(train_dataset, batch_size, workers, True, device.type, drop_last=drop_last)
    val_loader = make_loader(val_dataset, min(batch_size, len(val_dataset)), workers, False, device.type)
    model_config = config["model"]
    model = MobileNetV3EmbeddingClassifier(
        num_classes=len(class_to_idx),
        embedding_dim=int(model_config.get("embedding_dim", 256)),
        dropout=float(model_config.get("dropout", 0.2)),
        pretrained=bool(model_config.get("pretrained", True)) and not bool(resume_path),
    ).to(device)
    checkpoint_dir = ensure_dir(resolve_path(config["output"]["checkpoint_dir"]))
    output_dir = ensure_dir(resolve_path(config["output"]["output_dir"]))
    config["output"]["checkpoint_dir"] = str(checkpoint_dir)
    config["output"]["output_dir"] = str(output_dir)
    config["runtime"] = {
        "device": str(device), "num_classes": len(class_to_idx), "split_dir": str(split_dir),
        "run_name": run_name, "dataset_fingerprint": plan["dataset_fingerprint"],
        "test_manifest_loaded": False,
    }
    with (output_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, ensure_ascii=False, indent=2)
    print(f"Device: {device} | classes: {len(class_to_idx)} | train/val: {len(train_dataset)}/{len(val_dataset)}")
    result = Trainer(model, config, class_to_idx, device).fit(train_loader, val_loader, resume_path=resume_path)
    print(f"Training complete. Best validation top-1: {result['best_val_top1']:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pth'}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        plan = build_training_plan(args.config, args.split_dir, run_name=args.run_name, device_override=args.device)
        output = plan["config"]["output"]
        validate_run_destination(plan, args.resume or plan["config"]["training"].get("resume"))
        print(json.dumps({
            "config": plan["config_path"], "run_name": args.run_name,
            "device": str(resolve_device(plan["config"]["training"].get("device", "auto"))),
            "class_count": len(plan["class_to_idx"]), "crop_counts": plan["counts"],
            "dataset_fingerprint": plan["dataset_fingerprint"],
            "checkpoint_dir": output.get("checkpoint_dir"), "output_dir": output.get("output_dir"),
            "test_manifest_loaded": False, "training_started": False,
        }, ensure_ascii=False, indent=2))
        return
    run_training(
        args.config,
        split_dir_override=args.split_dir,
        resume_override=args.resume,
        run_name=args.run_name,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
