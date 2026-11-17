"""Retrieval-selected two-stage training used by the bounded AXE experiment."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps

from src.evaluation.metrics import nearest_neighbor_retrieval
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.utils.io import ensure_dir, save_json


def state_dict_fingerprint(model: nn.Module) -> str:
    """Return a stable full-state fingerprint for trial-initialization audits."""

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def is_better_retrieval_checkpoint(candidate_macro: float, best_macro: float) -> bool:
    """Checkpoint selection intentionally uses only macro validation Recall@1."""

    return math.isfinite(candidate_macro) and candidate_macro > best_macro + 1e-12


@torch.inference_mode()
def _extract_outputs(
    model: MobileNetV3EmbeddingClassifier,
    loader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    embeddings: list[Tensor] = []
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    paths: list[str] = []
    labels: list[str] = []
    for images, batch_targets, batch_paths, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            batch_logits, batch_embeddings = model(images)
        logits.append(batch_logits.float().cpu())
        embeddings.append(batch_embeddings.float().cpu())
        targets.append(batch_targets.cpu())
        paths.extend(str(path) for path in batch_paths)
        labels.extend(str(label) for label in batch_labels)
    if not embeddings:
        raise ValueError("Evaluation DataLoader produced no samples")
    return {
        "embeddings": torch.cat(embeddings),
        "logits": torch.cat(logits),
        "targets": torch.cat(targets),
        "paths": paths,
        "labels": labels,
    }


def evaluate_retrieval_protocol(
    model: MobileNetV3EmbeddingClassifier,
    gallery_loader,
    query_loader,
    class_names: list[str],
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Regenerate the train gallery and evaluate deterministic query transforms."""

    gallery = _extract_outputs(model, gallery_loader, device, amp_enabled)
    query = _extract_outputs(model, query_loader, device, amp_enabled)
    metrics = nearest_neighbor_retrieval(
        query["embeddings"],
        query["labels"],
        gallery["embeddings"],
        gallery["labels"],
        class_names,
        query_paths=query["paths"],
        gallery_paths=gallery["paths"],
    )
    query_predictions = query["logits"].argmax(dim=1)
    metrics["classification_top1"] = float(
        query_predictions.eq(query["targets"]).float().mean().item()
    )
    metrics["classification_loss"] = float(
        nn.functional.cross_entropy(query["logits"], query["targets"]).item()
    )
    return metrics


METRICS_CSV_FIELDS = (
    "epoch",
    "stage",
    "head_lr",
    "backbone_lr",
    "train_ce_loss",
    "train_cls_top1",
    "val_ce_loss",
    "val_cls_top1",
    "val_retrieval_r1_macro",
    "val_retrieval_r1_micro",
    "elapsed_seconds",
)


def write_metrics_csv(history: list[dict[str, Any]], path: Path) -> None:
    """Write losses as means and accuracies/recalls on the numeric [0, 1] scale."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _error_preview(error: dict[str, Any]) -> Image.Image:
    panels = []
    for path in (error["query_path"], error["gallery_path"]):
        with Image.open(path) as image:
            panels.append(ImageOps.contain(image.convert("RGB"), (300, 300)))
    canvas = Image.new("RGB", (640, 360), "white")
    canvas.paste(panels[0], ((310 - panels[0].width) // 2, 45 + (300 - panels[0].height) // 2))
    canvas.paste(panels[1], (330 + (300 - panels[1].width) // 2, 45 + (300 - panels[1].height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Query: {error['query_label']}"[:75], fill="black")
    draw.text((330, 8), f"Nearest: {error['predicted_label']}"[:75], fill="black")
    draw.text((8, 338), f"cosine={float(error['cosine_similarity']):.4f}", fill="black")
    return canvas


def write_retrieval_artifacts(
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str,
    max_error_previews: int = 8,
) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(metrics["per_sku"], output_dir / f"{prefix}_per_sku.json")
    class_names = list(metrics["class_names"])
    matrix = metrics["confusion_matrix"]
    with (output_dir / f"{prefix}_retrieval_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_sku", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])
    errors = list(metrics["incorrect_nearest_neighbors"])
    error_fields = (
        "query_index",
        "query_path",
        "query_label",
        "gallery_index",
        "gallery_path",
        "predicted_label",
        "cosine_similarity",
    )
    with (output_dir / f"{prefix}_incorrect_nearest_neighbors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(errors)
    preview_dir = output_dir / f"{prefix}_incorrect_pairs"
    for index, error in enumerate(errors[:max_error_previews]):
        ensure_dir(preview_dir)
        _error_preview(error).save(preview_dir / f"error_{index:03d}.jpg", quality=92)


def write_training_plots(history: list[dict[str, Any]], output_dir: Path) -> None:
    """Create the three requested plots after a manual run completes."""
    output_dir = ensure_dir(output_dir)
    epochs = [int(row["epoch"]) for row in history]
    plot_specs = (
        (
            "ce_loss.png",
            "CrossEntropy loss",
            (("train_ce_loss", "Train CE"), ("val_ce_loss", "Validation CE")),
            "Mean CrossEntropy",
        ),
        (
            "classification_top1.png",
            "Classification Top-1",
            (("train_cls_top1", "Train classification Top-1"), ("val_cls_top1", "Validation classification Top-1")),
            "Accuracy [0, 1]",
        ),
        (
            "validation_retrieval_recall_at_1.png",
            "Validation retrieval Recall@1",
            (("val_retrieval_r1_macro", "Macro R@1"), ("val_retrieval_r1_micro", "Micro R@1")),
            "Recall [0, 1]",
        ),
    )
    colors = ((36, 99, 181), (220, 79, 54))
    for filename, title, series, ylabel in plot_specs:
        canvas = Image.new("RGB", (840, 500), "white")
        draw = ImageDraw.Draw(canvas)
        left, top, right, bottom = 75, 55, 815, 430
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        all_values = [float(row[key]) for key, _ in series for row in history]
        y_min = 0.0
        y_max = 1.0 if "[0, 1]" in ylabel else max(all_values) * 1.08
        if y_max <= y_min:
            y_max = y_min + 1.0
        for tick in range(6):
            y = bottom - round((bottom - top) * tick / 5)
            value = y_min + (y_max - y_min) * tick / 5
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((8, y - 7), f"{value:.2f}", fill="black")
        x_denominator = max(1, len(epochs) - 1)
        for series_index, (key, label) in enumerate(series):
            points = []
            for index, row in enumerate(history):
                x = left + round((right - left) * index / x_denominator)
                value = float(row[key])
                y = bottom - round((bottom - top) * (value - y_min) / (y_max - y_min))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[series_index], width=3)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=colors[series_index])
            legend_y = 18 + series_index * 18
            draw.line((560, legend_y + 6, 590, legend_y + 6), fill=colors[series_index], width=3)
            draw.text((598, legend_y), label, fill="black")
        for index, epoch in enumerate(epochs):
            x = left + round((right - left) * index / x_denominator)
            draw.text((x - 4, bottom + 8), str(epoch), fill="black")
        draw.text((left, 18), title, fill="black")
        draw.text((left, 462), "Epoch", fill="black")
        draw.text((8, 36), ylabel, fill="black")
        canvas.save(output_dir / filename)


class RetrievalTrainer:
    """Train with CE logits and select checkpoints by macro validation Recall@1."""

    def __init__(
        self,
        model: MobileNetV3EmbeddingClassifier,
        config: dict[str, Any],
        class_to_idx: dict[str, int],
        device: torch.device,
        trial_dir: Path,
        initialization_fingerprint: str,
        dataset_fingerprint: str,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.class_to_idx = class_to_idx
        self.class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.device = device
        self.trial_dir = ensure_dir(trial_dir)
        self.checkpoint_dir = ensure_dir(checkpoint_dir or self.trial_dir / "checkpoints")
        self.initialization_fingerprint = initialization_fingerprint
        self.dataset_fingerprint = dataset_fingerprint
        training = config["training"]
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(training.get("label_smoothing", 0.0))
        )
        self.amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingLR | None = None
        self.optimizer_was_fresh = False
        self.embedding_head_gradient_seen = False

    def _start_stage(self, fine_tuning: bool, stage_epochs: int) -> None:
        training = self.config["training"]
        head_lr = float(training["learning_rate"])
        if fine_tuning:
            self.model.unfreeze_last_blocks(None)
            parameter_groups = [
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
            parameter_groups = [
                {"name": "embedding_head", "params": self.model.embedding_head.parameters(), "lr": head_lr},
                {"name": "classifier", "params": self.model.classifier.parameters(), "lr": head_lr},
            ]
        self.optimizer = AdamW(
            parameter_groups, weight_decay=float(training["weight_decay"])
        )
        self.optimizer_was_fresh = self.optimizer_was_fresh or not self.optimizer.state
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, stage_epochs),
            eta_min=0.0,
        )

    def _train_epoch(self, loader) -> dict[str, float | int]:
        assert self.optimizer is not None
        self.model.train(True)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress = tqdm(loader, desc="training batches", leave=False, dynamic_ncols=True)
        for images, targets, _, _ in progress:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits, _ = self.model(images)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            if any(
                parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in self.model.embedding_head.parameters()
            ):
                self.embedding_head_gradient_seen = True
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.detach().item()) * len(targets)
            total_correct += int(logits.detach().argmax(dim=1).eq(targets).sum().item())
            total_samples += len(targets)
            progress.set_postfix(
                train_ce=f"{total_loss / total_samples:.4f}",
                train_cls_top1=f"{100.0 * total_correct / total_samples:.2f}%",
            )
        if not total_samples:
            raise ValueError("Training DataLoader produced no samples")
        return {
            "loss": total_loss / total_samples,
            "top1": total_correct / total_samples,
            "samples": total_samples,
        }

    def _checkpoint(
        self,
        path: Path,
        epoch: int,
        stage: str,
        best_macro: float,
        validation_metrics: dict[str, Any],
    ) -> None:
        assert self.optimizer is not None and self.scheduler is not None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "best_validation_metric": best_macro,
                "selection_metric": "macro_validation_recall@1",
                "validation_metrics": validation_metrics,
                "class_to_idx": self.class_to_idx,
                "config": self.config,
                "initialization_fingerprint": self.initialization_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
            },
            path,
        )

    def fit(
        self,
        train_loader,
        gallery_loader,
        validation_loader,
        *,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        training = self.config["training"]
        total_epochs = min(15, int(training["num_epochs"]))
        head_epochs = min(max(1, int(training.get("head_epochs", 3))), total_epochs)
        patience = int(training.get("early_stopping_patience", 5))
        freeze_throughout = bool(training.get("freeze_backbone_throughout", False))
        best_macro = float("-inf")
        best_epoch = 0
        best_metrics: dict[str, Any] | None = None
        monitored_bad_epochs = 0
        history: list[dict[str, Any]] = []
        stopped_for_budget = False
        training_started = time.monotonic()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if freeze_throughout:
            self._start_stage(fine_tuning=False, stage_epochs=total_epochs)

        for epoch_index in range(total_epochs):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                stopped_for_budget = True
                break
            fine_tuning = not freeze_throughout and epoch_index >= head_epochs
            stage = "fine_tune" if fine_tuning else (
                "frozen_continuation" if freeze_throughout and epoch_index >= head_epochs else "head"
            )
            if not freeze_throughout and (epoch_index == 0 or epoch_index == head_epochs):
                stage_epochs = head_epochs if epoch_index == 0 else total_epochs - head_epochs
                self._start_stage(fine_tuning=fine_tuning, stage_epochs=stage_epochs)
                if fine_tuning:
                    monitored_bad_epochs = 0

            epoch_started = time.monotonic()
            assert self.optimizer is not None and self.scheduler is not None
            learning_rates = {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(self.optimizer.param_groups)
            }
            backbone_frozen = not any(
                parameter.requires_grad for parameter in self.model.backbone.parameters()
            )
            print(
                f"Epoch {epoch_index + 1:02d}/{total_epochs} | stage={stage} | "
                f"backbone={'frozen' if backbone_frozen else 'unfrozen'} | "
                f"learning_rates={learning_rates}",
                flush=True,
            )
            train_metrics = self._train_epoch(train_loader)
            validation = evaluate_retrieval_protocol(
                self.model,
                gallery_loader,
                validation_loader,
                self.class_names,
                self.device,
                self.amp_enabled,
            )
            macro = float(validation["macro_recall@1"])
            improved = is_better_retrieval_checkpoint(macro, best_macro)
            if improved:
                best_macro = macro
                best_epoch = epoch_index + 1
                best_metrics = validation
                monitored_bad_epochs = 0
                self._checkpoint(
                    self.checkpoint_dir / "best_model.pth",
                    epoch_index,
                    stage,
                    best_macro,
                    validation,
                )
                save_json(validation, self.trial_dir / "best_validation_metrics.json")
            elif epoch_index + 1 > head_epochs:
                monitored_bad_epochs += 1
            self.scheduler.step()
            self._checkpoint(
                self.checkpoint_dir / "last_model.pth",
                epoch_index,
                stage,
                best_macro,
                validation,
            )
            record = {
                "epoch": epoch_index + 1,
                "stage": stage,
                "head_lr": learning_rates.get("embedding_head", learning_rates.get("classifier", 0.0)),
                "backbone_lr": learning_rates.get("backbone", 0.0),
                "train_ce_loss": train_metrics["loss"],
                "train_cls_top1": train_metrics["top1"],
                "val_ce_loss": validation["classification_loss"],
                "val_cls_top1": validation["classification_top1"],
                "val_retrieval_r1_macro": macro,
                "val_retrieval_r1_micro": validation["micro_recall@1"],
                "elapsed_seconds": time.monotonic() - training_started,
                "per_sku_recall@1": {
                    name: values["recall@1"] for name, values in validation["per_sku"].items()
                },
                "learning_rates": learning_rates,
                "improved": improved,
                "epoch_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            write_metrics_csv(history, self.trial_dir / "metrics.csv")
            save_json({"history": history}, self.trial_dir / "history.json")
            with (self.trial_dir / "epoch_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(
                f"Epoch {epoch_index + 1:02d}/{total_epochs} complete | "
                f"train_ce_loss={train_metrics['loss']:.4f} | "
                f"train_cls_top1={100.0 * float(train_metrics['top1']):.2f}% | "
                f"val_ce_loss={validation['classification_loss']:.4f} | "
                f"val_cls_top1={100.0 * float(validation['classification_top1']):.2f}% | "
                f"val_retrieval_r1_macro={100.0 * macro:.2f}% | "
                f"val_retrieval_r1_micro={100.0 * float(validation['micro_recall@1']):.2f}% | "
                f"elapsed={record['elapsed_seconds']:.1f}s | best_epoch={best_epoch} | "
                f"best_val_retrieval_r1_macro={100.0 * best_macro:.2f}% | "
                f"early_stopping_counter={monitored_bad_epochs}/{patience}",
                flush=True,
            )
            for sku_name, sku_metrics in validation["per_sku"].items():
                print(
                    f"  {sku_name}: retrieval R@1={100.0 * float(sku_metrics['recall@1']):.2f}% "
                    f"({sku_metrics['correct']}/{sku_metrics['queries']})",
                    flush=True,
                )
            monitoring_active = epoch_index + 1 > head_epochs
            if monitoring_active and monitored_bad_epochs >= patience:
                print(
                    f"Early stopping: {patience} monitored epochs without macro validation Recall@1 improvement.",
                    flush=True,
                )
                break

        if best_metrics is None:
            raise RuntimeError("No epoch completed, so no best checkpoint was produced")
        peak_vram_mib = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )
        result = {
            "best_epoch": best_epoch,
            "best_validation_macro_recall@1": best_macro,
            "best_validation_micro_recall@1": best_metrics["micro_recall@1"],
            "best_validation_classification_top1": best_metrics["classification_top1"],
            "best_validation_per_sku": best_metrics["per_sku"],
            "epochs_completed": len(history),
            "stopped_for_budget": stopped_for_budget,
            "peak_vram_mib": peak_vram_mib,
            "initialization_fingerprint": self.initialization_fingerprint,
            "optimizer_started_fresh": self.optimizer_was_fresh,
            "embedding_head_gradient_seen": self.embedding_head_gradient_seen,
            "scheduler": "stage-local CosineAnnealingLR; reset only when Stage 2 creates its optimizer",
            "metric_scale": "losses are sample-weighted means; accuracy and recall values are in [0, 1]",
            "last_epoch_metrics": history[-1],
            "best_checkpoint_metrics": {
                "epoch": best_epoch,
                "val_retrieval_r1_macro": best_macro,
                "val_retrieval_r1_micro": best_metrics["micro_recall@1"],
                "val_cls_top1": best_metrics["classification_top1"],
                "val_ce_loss": best_metrics["classification_loss"],
            },
            "history": history,
        }
        write_training_plots(history, self.trial_dir / "plots")
        write_retrieval_artifacts(
            best_metrics,
            self.trial_dir / "best_checkpoint_validation",
            prefix="validation",
        )
        save_json(result, self.trial_dir / "result.json")
        return result
