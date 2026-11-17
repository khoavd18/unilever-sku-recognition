"""Two-stage classification trainer with optional metric-loss augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.evaluation.metrics import classification_accuracy
from src.losses.metric_loss import BatchHardTripletLoss
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.utils.io import ensure_dir, save_json


@dataclass
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    samples: int


class Trainer:
    """Train head first, then fine-tune MobileNetV3 with a smaller backbone LR."""

    def __init__(
        self,
        model: MobileNetV3EmbeddingClassifier,
        config: dict[str, Any],
        class_to_idx: dict[str, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.class_to_idx = class_to_idx
        self.device = device
        training = config["training"]
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
        metric_config = config["metric_learning"]
        self.metric_enabled = bool(metric_config.get("enabled", False))
        self.metric_weight = float(metric_config.get("loss_weight", 0.1))
        self.metric_criterion = BatchHardTripletLoss(float(metric_config.get("margin", 0.2)))
        self.amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.checkpoint_dir = ensure_dir(config["output"]["checkpoint_dir"])
        self.output_dir = ensure_dir(config["output"]["output_dir"])
        self.optimizer: AdamW | None = None
        self.scheduler: ReduceLROnPlateau | None = None

    def _make_optimizer(self, fine_tuning: bool, stage_epochs: int) -> None:
        training = self.config["training"]
        if not fine_tuning:
            self.model.freeze_backbone()
            parameter_groups = [
                {"params": self.model.embedding_head.parameters(), "lr": float(training["learning_rate"])},
                {"params": self.model.classifier.parameters(), "lr": float(training["learning_rate"])},
            ]
        else:
            self.model.unfreeze_last_blocks(None)
            parameter_groups = [
                {
                    "params": [p for p in self.model.backbone.parameters() if p.requires_grad],
                    "lr": float(training["backbone_learning_rate"]),
                },
                {"params": self.model.embedding_head.parameters(), "lr": float(training["learning_rate"])},
                {"params": self.model.classifier.parameters(), "lr": float(training["learning_rate"])},
            ]
        self.optimizer = AdamW(parameter_groups, weight_decay=float(training["weight_decay"]))
        # Validation loss is stable across both transfer-learning stages, unlike a
        # per-stage epoch counter, so it is a useful signal for LR reduction.
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

    def _run_epoch(self, loader, train: bool) -> EpochMetrics:
        self.model.train(train)
        total_loss = total_top1 = total_top5 = 0.0
        total_samples = 0
        context = torch.enable_grad if train else torch.no_grad
        with context():
            progress = tqdm(loader, desc="train" if train else "validation", leave=False)
            for images, targets, _, _ in progress:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if train:
                    assert self.optimizer is not None
                    self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits, embeddings = self.model(images)
                    loss = self.criterion(logits, targets)
                    if train and self.metric_enabled:
                        loss = loss + self.metric_weight * self.metric_criterion(embeddings, targets)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = targets.size(0)
                accuracy = classification_accuracy(logits.detach(), targets, topk=(1, 5))
                total_loss += loss.detach().item() * batch_size
                total_top1 += accuracy["top1"] * batch_size
                total_top5 += accuracy["top5"] * batch_size
                total_samples += batch_size
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
        if total_samples == 0:
            raise ValueError("DataLoader produced zero samples")
        return EpochMetrics(total_loss / total_samples, total_top1 / total_samples, total_top5 / total_samples, total_samples)

    def _checkpoint(self, path: Path, epoch: int, best_metric: float, stage: str, bad_epochs: int) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch,
                "best_validation_metric": best_metric,
                "stage": stage,
                "bad_epochs": bad_epochs,
                "class_to_idx": self.class_to_idx,
                "config": self.config,
            },
            path,
        )

    def fit(self, train_loader, val_loader, resume_path: str | Path | None = None) -> dict[str, Any]:
        training = self.config["training"]
        total_epochs = int(training["num_epochs"])
        head_epochs = min(max(0, int(training.get("head_epochs", 0))), total_epochs)
        patience = int(training.get("early_stopping_patience", 6))
        best_top1 = float("-inf")
        bad_epochs = 0
        start_epoch = 0
        resume_payload: dict[str, Any] | None = None
        if resume_path:
            resume_path = Path(resume_path)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint was not found: {resume_path}")
            resume_payload = torch.load(resume_path, map_location=self.device, weights_only=False)
            if resume_payload.get("class_to_idx") != self.class_to_idx:
                raise ValueError("class_to_idx in resume checkpoint differs from current split manifests.")
            self.model.load_state_dict(resume_payload["model_state_dict"])
            start_epoch = int(resume_payload["epoch"]) + 1
            best_top1 = float(resume_payload["best_validation_metric"])
            bad_epochs = int(resume_payload.get("bad_epochs", 0))
            print(f"Resuming from {resume_path} at epoch {start_epoch + 1}/{total_epochs}.")
        if start_epoch >= total_epochs:
            print("Resume checkpoint already completed the configured number of epochs; no new epoch was run.")
            return {"best_val_top1": best_top1, "epochs_completed": 0, "history": []}
        history: list[dict[str, float | int | str]] = []
        for epoch in range(start_epoch, total_epochs):
            fine_tuning = epoch >= head_epochs
            stage = "fine_tune" if fine_tuning else "head"
            stage_start = epoch == start_epoch or epoch == 0 or epoch == head_epochs
            if stage_start:
                stage_epochs = (total_epochs - head_epochs) if fine_tuning else head_epochs
                self._make_optimizer(fine_tuning=fine_tuning, stage_epochs=stage_epochs)
                if resume_payload is not None and epoch == start_epoch and resume_payload.get("stage") == stage:
                    assert self.optimizer is not None and self.scheduler is not None
                    self.optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                    self.scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
                    if "scaler_state_dict" in resume_payload:
                        self.scaler.load_state_dict(resume_payload["scaler_state_dict"])
                print("Stage 2: fine-tuning the full backbone." if fine_tuning else "Stage 1: training projection and classifier heads.")
            sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            assert self.scheduler is not None
            self.scheduler.step(val_metrics.loss)
            record = {
                "epoch": epoch + 1,
                "stage": stage,
                "train_loss": train_metrics.loss,
                "train_top1": train_metrics.top1,
                "val_loss": val_metrics.loss,
                "val_top1": val_metrics.top1,
                "val_top5": val_metrics.top5,
            }
            history.append(record)
            print(
                f"Epoch {epoch + 1}/{total_epochs} | train loss {train_metrics.loss:.4f}, "
                f"val loss {val_metrics.loss:.4f}, val top-1 {val_metrics.top1:.4f}, val top-5 {val_metrics.top5:.4f}"
            )
            if val_metrics.top1 > best_top1:
                best_top1 = val_metrics.top1
                bad_epochs = 0
                self._checkpoint(self.checkpoint_dir / "best_model.pth", epoch, best_top1, stage, bad_epochs)
            else:
                bad_epochs += 1
            self._checkpoint(self.checkpoint_dir / "last_model.pth", epoch, best_top1, stage, bad_epochs)
            if val_metrics.top1 <= best_top1:
                if bad_epochs >= patience:
                    print(f"Early stopping after {patience} epochs without validation top-1 improvement.")
                    break
        result = {"best_val_top1": best_top1, "epochs_completed": len(history), "history": history}
        save_json(result, self.output_dir / "training_history.json")
        return result
