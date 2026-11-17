"""Classification-only two-stage trainer for the manual 100-epoch AXE experiment."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics_from_predictions
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.utils.io import ensure_dir, save_json


METRICS_FIELDS = (
    "epoch",
    "stage",
    "head_lr",
    "backbone_lr",
    "train_ce_loss",
    "train_cls_top1",
    "train_macro_precision",
    "train_macro_recall",
    "train_macro_f1",
    "val_ce_loss",
    "val_cls_top1",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "epoch_seconds",
    "elapsed_seconds",
    "eta_seconds",
)


def is_better_classification_checkpoint(
    candidate_macro_f1: float,
    candidate_loss: float,
    best_macro_f1: float,
    best_loss: float,
) -> bool:
    """Prefer macro F1, then lower CE loss; exact ties retain the earlier epoch."""

    if not math.isfinite(candidate_macro_f1) or not math.isfinite(candidate_loss):
        return False
    if candidate_macro_f1 > best_macro_f1 + 1e-12:
        return True
    return abs(candidate_macro_f1 - best_macro_f1) <= 1e-12 and candidate_loss < best_loss - 1e-12


def classification_stage(epoch_index: int, head_epochs: int = 3) -> str:
    """Map a zero-based epoch to the configured two-stage schedule."""

    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    return "head" if epoch_index < head_epochs else "fine_tune"


def _write_metrics_csv(history: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _plot(
    history: list[dict[str, Any]],
    path: Path,
    title: str,
    ylabel: str,
    series: tuple[tuple[str, str], ...],
    fixed_unit_range: bool = False,
) -> None:
    colors = ((36, 99, 181), (220, 79, 54), (60, 160, 90))
    canvas = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 80, 60, 870, 450
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    values = [float(row[key]) for key, _ in series for row in history]
    y_min = 0.0
    y_max = 1.0 if fixed_unit_range else max(values) * 1.08
    if y_max <= y_min:
        y_max = 1.0
    for tick in range(6):
        y = bottom - round((bottom - top) * tick / 5)
        value = y_min + (y_max - y_min) * tick / 5
        draw.line((left, y, right, y), fill=(225, 225, 225))
        draw.text((12, y - 7), f"{value:.4f}", fill="black")
    denominator = max(1, len(history) - 1)
    for series_index, (key, label) in enumerate(series):
        points = []
        for index, row in enumerate(history):
            x = left + round((right - left) * index / denominator)
            y = bottom - round((bottom - top) * float(row[key]) / y_max)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=colors[series_index], width=3)
        legend_y = 18 + 18 * series_index
        draw.line((610, legend_y + 6, 642, legend_y + 6), fill=colors[series_index], width=3)
        draw.text((650, legend_y), label, fill="black")
    draw.text((left, 20), title, fill="black")
    draw.text((left, 480), "Epoch", fill="black")
    draw.text((10, 40), ylabel, fill="black")
    ensure_dir(path.parent)
    canvas.save(path)


def write_classification_plots(history: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)
    _plot(
        history,
        output_dir / "ce_loss.png",
        "Train/validation CrossEntropy loss",
        "Mean CE loss",
        (("train_ce_loss", "Train CE"), ("val_ce_loss", "Validation CE")),
    )
    _plot(
        history,
        output_dir / "classification_accuracy.png",
        "Train/validation classification Top-1",
        "Accuracy [0, 1]",
        (("train_cls_top1", "Train Top-1"), ("val_cls_top1", "Validation Top-1")),
        fixed_unit_range=True,
    )
    _plot(
        history,
        output_dir / "macro_f1.png",
        "Train/validation Macro F1",
        "Macro F1 [0, 1]",
        (("train_macro_f1", "Train Macro F1"), ("val_macro_f1", "Validation Macro F1")),
        fixed_unit_range=True,
    )
    _plot(
        history,
        output_dir / "learning_rates.png",
        "Learning rates",
        "Learning rate",
        (("head_lr", "Head LR"), ("backbone_lr", "Backbone LR")),
    )


class ClassificationTrainer:
    def __init__(
        self,
        model: MobileNetV3EmbeddingClassifier,
        config: dict[str, Any],
        class_to_idx: dict[str, int],
        device: torch.device,
        output_dir: Path,
        checkpoint_dir: Path,
        dataset_fingerprint: str,
        initialization_fingerprint: str,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.class_to_idx = class_to_idx
        self.class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.dataset_fingerprint = dataset_fingerprint
        self.initialization_fingerprint = initialization_fingerprint
        training = config["training"]
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
        self.amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingLR | None = None

    def _start_stage(self, fine_tuning: bool, stage_epochs: int) -> None:
        training = self.config["training"]
        head_lr = float(training["learning_rate"])
        if fine_tuning:
            self.model.unfreeze_last_blocks(None)
            groups = [
                {
                    "name": "backbone",
                    "params": [parameter for parameter in self.model.backbone.parameters() if parameter.requires_grad],
                    "lr": float(training["backbone_learning_rate"]),
                },
                {"name": "embedding_head", "params": self.model.embedding_head.parameters(), "lr": head_lr},
                {"name": "classifier", "params": self.model.classifier.parameters(), "lr": head_lr},
            ]
        else:
            self.model.freeze_backbone()
            groups = [
                {"name": "embedding_head", "params": self.model.embedding_head.parameters(), "lr": head_lr},
                {"name": "classifier", "params": self.model.classifier.parameters(), "lr": head_lr},
            ]
        self.optimizer = AdamW(groups, weight_decay=float(training["weight_decay"]))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, stage_epochs), eta_min=0.0)

    def _run_epoch(self, loader, train: bool) -> dict[str, Any]:
        self.model.train(train)
        total_loss = 0.0
        sample_count = 0
        predictions: list[int] = []
        targets_all: list[int] = []
        context = torch.enable_grad if train else torch.no_grad
        with context():
            progress = tqdm(
                loader,
                desc="train batches" if train else "validation batches",
                leave=False,
                dynamic_ncols=True,
            )
            for images, targets, _, _ in progress:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if train:
                    assert self.optimizer is not None
                    self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits, _ = self.model(images)
                    loss = self.criterion(logits, targets)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = len(targets)
                total_loss += float(loss.detach().item()) * batch_size
                sample_count += batch_size
                predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
                targets_all.extend(targets.detach().cpu().tolist())
                progress.set_postfix(ce=f"{total_loss / sample_count:.4f}")
        if not sample_count:
            raise ValueError("DataLoader produced no samples")
        metrics = classification_metrics_from_predictions(targets_all, predictions, self.class_names)
        metrics["ce_loss"] = total_loss / sample_count
        return metrics

    def _checkpoint(
        self,
        path: Path,
        epoch_index: int,
        stage: str,
        best_epoch: int,
        best_macro_f1: float,
        best_val_loss: float,
        validation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch_index,
                "stage": stage,
                "best_epoch": best_epoch,
                "best_validation_metric": best_macro_f1,
                "best_validation_loss": best_val_loss,
                "selection_metric": "val_macro_f1_then_val_ce_loss",
                "validation_metrics": validation,
                "history": history,
                "class_to_idx": self.class_to_idx,
                "config": self.config,
                "dataset_fingerprint": self.dataset_fingerprint,
                "initialization_fingerprint": self.initialization_fingerprint,
            },
            path,
        )

    def _evaluate_checkpoint(self, path: Path, validation_loader) -> tuple[int, dict[str, Any]]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        metrics = self._run_epoch(validation_loader, train=False)
        return int(payload["epoch"]) + 1, metrics

    def _write_confusion(self, metrics: dict[str, Any], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_sku", *self.class_names])
            for name, row in zip(self.class_names, metrics["confusion_matrix"]):
                writer.writerow([name, *row])

    def _finalize(
        self,
        validation_loader,
        history: list[dict[str, Any]],
        best_epoch: int,
        runtime_seconds: float,
        interrupted: bool,
    ) -> dict[str, Any]:
        best_checkpoint = self.checkpoint_dir / "best_model.pth"
        last_checkpoint = self.checkpoint_dir / "last_model.pth"
        best_epoch_reloaded, best_metrics = self._evaluate_checkpoint(best_checkpoint, validation_loader)
        last_epoch_reloaded, last_metrics = self._evaluate_checkpoint(last_checkpoint, validation_loader)
        self._write_confusion(best_metrics, self.output_dir / "validation_confusion_best.csv")
        self._write_confusion(last_metrics, self.output_dir / "validation_confusion_last.csv")
        with (self.output_dir / "per_class_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = ("checkpoint", "epoch", "sku", "precision", "recall", "f1", "support", "correct")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for checkpoint_name, epoch, metrics in (
                ("best", best_epoch_reloaded, best_metrics),
                ("last", last_epoch_reloaded, last_metrics),
            ):
                for sku, values in metrics["per_class"].items():
                    writer.writerow({"checkpoint": checkpoint_name, "epoch": epoch, "sku": sku, **values})
        write_classification_plots(history, self.output_dir / "plots")
        summary = {
            "status": "partial_interrupted" if interrupted else "complete",
            "classification_only": True,
            "completed_epochs": len(history),
            "configured_epochs": int(self.config["training"]["num_epochs"]),
            "runtime_seconds": runtime_seconds,
            "dataset_counts": {
                "train": self.config.get("runtime", {}).get("train_crops"),
                "validation": self.config.get("runtime", {}).get("validation_crops"),
            },
            "best_checkpoint": {
                "path": str(best_checkpoint.resolve()),
                "epoch": best_epoch_reloaded,
                **{key: best_metrics[key] for key in ("ce_loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
            },
            "last_checkpoint": {
                "path": str(last_checkpoint.resolve()),
                "epoch": last_epoch_reloaded,
                **{key: last_metrics[key] for key in ("ce_loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
            },
            "best_checkpoint_per_sku": best_metrics["per_class"],
            "zero_division_policy": 0,
            "saved_files": {
                "metrics_csv": str((self.output_dir / "metrics.csv").resolve()),
                "per_class_metrics_csv": str((self.output_dir / "per_class_metrics.csv").resolve()),
                "best_confusion": str((self.output_dir / "validation_confusion_best.csv").resolve()),
                "last_confusion": str((self.output_dir / "validation_confusion_last.csv").resolve()),
                "plots": str((self.output_dir / "plots").resolve()),
                "summary_json": str((self.output_dir / "summary.json").resolve()),
                "summary_markdown": str((self.output_dir / "summary.md").resolve()),
                "best_checkpoint": str(best_checkpoint.resolve()),
                "last_checkpoint": str(last_checkpoint.resolve()),
            },
        }
        save_json(summary, self.output_dir / "summary.json")
        lines = [
            "# Báo cáo validation phân loại AXE 5 SKU",
            "",
            f"Trạng thái: **{summary['status']}**. Chế độ classification-only không đọc test trong training.",
            "",
            "## Best checkpoint so với last checkpoint",
            "",
            "| Checkpoint | Epoch | Val CE | Accuracy | Macro precision | Macro recall | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, values in (("Best", summary["best_checkpoint"]), ("Last", summary["last_checkpoint"])):
            lines.append(
                f"| {label} | {values['epoch']} | {values['ce_loss']:.6f} | "
                f"{values['accuracy']:.6f} | {values['macro_precision']:.6f} | "
                f"{values['macro_recall']:.6f} | {values['macro_f1']:.6f} |"
            )
        lines.extend(["", "## Per-SKU tại best checkpoint", "", "| SKU | Precision | Recall | F1 | Đúng/Tổng |", "|---|---:|---:|---:|---:|"])
        for sku, values in best_metrics["per_class"].items():
            lines.append(
                f"| {sku} | {values['precision']:.6f} | {values['recall']:.6f} | "
                f"{values['f1']:.6f} | {values['correct']}/{values['support']} |"
            )
        lines.extend(
            [
                "",
                f"Runtime: {runtime_seconds:.1f} giây; hoàn tất {len(history)}/{self.config['training']['num_epochs']} epoch.",
                "Zero denominator cho precision/recall/F1 được gán bằng 0; macro là trung bình không trọng số của đủ năm SKU.",
                "Kết quả ở đây chỉ là validation; test phải chạy riêng sau khi người dùng chọn checkpoint.",
            ]
        )
        (self.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary

    def fit(self, train_loader, validation_loader, resume_path: Path | None = None) -> dict[str, Any]:
        training = self.config["training"]
        total_epochs = int(training["num_epochs"])
        head_epochs = int(training["head_epochs"])
        best_macro_f1 = float("-inf")
        best_val_loss = float("inf")
        best_epoch = 0
        start_epoch = 0
        history: list[dict[str, Any]] = []
        resume_payload: dict[str, Any] | None = None
        if resume_path is not None:
            resume_payload = torch.load(resume_path, map_location=self.device, weights_only=False)
            if resume_payload.get("class_to_idx") != self.class_to_idx:
                raise ValueError("Resume checkpoint class mapping differs from this run")
            if resume_payload.get("dataset_fingerprint") != self.dataset_fingerprint:
                raise ValueError("Resume checkpoint dataset fingerprint differs from this run")
            self.model.load_state_dict(resume_payload["model_state_dict"])
            self.initialization_fingerprint = str(
                resume_payload.get("initialization_fingerprint", self.initialization_fingerprint)
            )
            start_epoch = int(resume_payload["epoch"]) + 1
            best_epoch = int(resume_payload["best_epoch"])
            best_macro_f1 = float(resume_payload["best_validation_metric"])
            best_val_loss = float(resume_payload["best_validation_loss"])
            history = list(resume_payload.get("history", []))
        prior_elapsed = float(history[-1]["elapsed_seconds"]) if history else 0.0
        training_started = time.monotonic()
        interrupted = False
        try:
            for epoch_index in range(start_epoch, total_epochs):
                stage = classification_stage(epoch_index, head_epochs)
                fine_tuning = stage == "fine_tune"
                if epoch_index == start_epoch or epoch_index == 0 or epoch_index == head_epochs:
                    stage_epochs = total_epochs - head_epochs if fine_tuning else head_epochs
                    self._start_stage(fine_tuning, stage_epochs)
                    if resume_payload is not None and epoch_index == start_epoch and resume_payload.get("stage") == stage:
                        assert self.optimizer is not None and self.scheduler is not None
                        self.optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                        self.scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
                        self.scaler.load_state_dict(resume_payload["scaler_state_dict"])
                assert self.optimizer is not None and self.scheduler is not None
                epoch_started = time.monotonic()
                lrs = {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(self.optimizer.param_groups)}
                frozen = not any(parameter.requires_grad for parameter in self.model.backbone.parameters())
                print(
                    f"Epoch {epoch_index + 1}/100 | stage={stage} | backbone={'frozen' if frozen else 'unfrozen'} | LRs={lrs}",
                    flush=True,
                )
                train_metrics = self._run_epoch(train_loader, train=True)
                validation_metrics = self._run_epoch(validation_loader, train=False)
                epoch_seconds = time.monotonic() - epoch_started
                elapsed = prior_elapsed + time.monotonic() - training_started
                completed_this_session = epoch_index - start_epoch + 1
                eta = elapsed / completed_this_session * (total_epochs - epoch_index - 1)
                improved = is_better_classification_checkpoint(
                    float(validation_metrics["macro_f1"]),
                    float(validation_metrics["ce_loss"]),
                    best_macro_f1,
                    best_val_loss,
                )
                if improved:
                    best_macro_f1 = float(validation_metrics["macro_f1"])
                    best_val_loss = float(validation_metrics["ce_loss"])
                    best_epoch = epoch_index + 1
                record = {
                    "epoch": epoch_index + 1,
                    "stage": stage,
                    "head_lr": lrs.get("embedding_head", lrs.get("classifier", 0.0)),
                    "backbone_lr": lrs.get("backbone", 0.0),
                    "train_ce_loss": train_metrics["ce_loss"],
                    "train_cls_top1": train_metrics["accuracy"],
                    "train_macro_precision": train_metrics["macro_precision"],
                    "train_macro_recall": train_metrics["macro_recall"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_ce_loss": validation_metrics["ce_loss"],
                    "val_cls_top1": validation_metrics["accuracy"],
                    "val_macro_precision": validation_metrics["macro_precision"],
                    "val_macro_recall": validation_metrics["macro_recall"],
                    "val_macro_f1": validation_metrics["macro_f1"],
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                }
                history.append(record)
                self.scheduler.step()
                if improved:
                    self._checkpoint(
                        self.checkpoint_dir / "best_model.pth",
                        epoch_index,
                        stage,
                        best_epoch,
                        best_macro_f1,
                        best_val_loss,
                        validation_metrics,
                        history,
                    )
                self._checkpoint(
                    self.checkpoint_dir / "last_model.pth",
                    epoch_index,
                    stage,
                    best_epoch,
                    best_macro_f1,
                    best_val_loss,
                    validation_metrics,
                    history,
                )
                _write_metrics_csv(history, self.output_dir / "metrics.csv")
                save_json({"history": history}, self.output_dir / "history.json")
                print(
                    f"Train: CE={train_metrics['ce_loss']:.4f}, Top-1={100*train_metrics['accuracy']:.2f}%, "
                    f"Macro P/R/F1={100*train_metrics['macro_precision']:.2f}%/"
                    f"{100*train_metrics['macro_recall']:.2f}%/{100*train_metrics['macro_f1']:.2f}%",
                    flush=True,
                )
                print(
                    f"Validation: CE={validation_metrics['ce_loss']:.4f}, Top-1={100*validation_metrics['accuracy']:.2f}%, "
                    f"Macro P/R/F1={100*validation_metrics['macro_precision']:.2f}%/"
                    f"{100*validation_metrics['macro_recall']:.2f}%/{100*validation_metrics['macro_f1']:.2f}% | "
                    f"epoch={epoch_seconds:.1f}s elapsed={elapsed:.1f}s ETA={eta:.1f}s | "
                    f"best_epoch={best_epoch} best_val_macro_f1={100*best_macro_f1:.2f}%",
                    flush=True,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted; preserving all completed epochs and writing a partial summary.", flush=True)
        if not history:
            raise RuntimeError("No epoch completed; no summary/checkpoint is available")
        summary = self._finalize(
            validation_loader,
            history,
            best_epoch,
            prior_elapsed + time.monotonic() - training_started,
            interrupted,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
