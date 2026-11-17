"""Metrics for closed-set classification and cosine-similarity retrieval."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def classification_accuracy(logits: Tensor, targets: Tensor, topk: Sequence[int] = (1, 5)) -> dict[str, float]:
    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices
    result: dict[str, float] = {}
    for requested_k in topk:
        k = min(requested_k, logits.shape[1])
        correct = predictions[:, :k].eq(targets[:, None]).any(dim=1).float().mean().item()
        result[f"top{requested_k}"] = correct
    return result


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics_from_predictions(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute all classification metrics from full-epoch predictions.

    Precision, recall, and F1 use zero when their denominator is zero. Macro
    values are the unweighted mean across every explicitly supplied class.
    """

    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    class_names = list(class_names)
    if targets_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional")
    if len(targets_array) != len(predictions_array) or not len(targets_array):
        raise ValueError("targets and predictions must have the same non-zero length")
    if len(class_names) != len(set(class_names)) or not class_names:
        raise ValueError("class_names must be non-empty and unique")
    if (
        targets_array.min() < 0
        or predictions_array.min() < 0
        or targets_array.max() >= len(class_names)
        or predictions_array.max() >= len(class_names)
    ):
        raise ValueError("targets/predictions contain an index outside class_names")

    matrix = confusion_matrix(targets_array, predictions_array, len(class_names))
    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "correct": true_positive,
        }
    return {
        "sample_count": len(targets_array),
        "accuracy": float((targets_array == predictions_array).mean()),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "zero_division_policy": 0,
    }


def retrieval_recall_at_k(
    query_embeddings: Tensor,
    query_labels: Sequence[str],
    gallery_embeddings: Tensor,
    gallery_labels: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
    query_paths: Sequence[str] | None = None,
    gallery_paths: Sequence[str] | None = None,
) -> dict[str, float | int]:
    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        raise ValueError("Query and gallery embeddings must both be non-empty")
    query_embeddings = F.normalize(query_embeddings.float(), dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings.float(), dim=1)
    similarities = query_embeddings @ gallery_embeddings.T
    if query_paths is not None and gallery_paths is not None:
        for query_index, query_path in enumerate(query_paths):
            for gallery_index, gallery_path in enumerate(gallery_paths):
                if query_path == gallery_path:
                    similarities[query_index, gallery_index] = float("-inf")
    max_k = min(max(ks), len(gallery_labels))
    ranked = similarities.topk(max_k, dim=1).indices.cpu().numpy()
    recalls = {k: 0 for k in ks}
    eligible = 0
    for query_index, label in enumerate(query_labels):
        candidates = [
            gallery_labels[index]
            for index in range(len(gallery_labels))
            if not (query_paths is not None and gallery_paths is not None and query_paths[query_index] == gallery_paths[index])
        ]
        if label not in candidates:
            continue
        eligible += 1
        ranked_labels = [gallery_labels[index] for index in ranked[query_index]]
        for k in ks:
            if label in ranked_labels[: min(k, len(ranked_labels))]:
                recalls[k] += 1
    result: dict[str, float | int] = {"eligible_queries": eligible}
    for k in ks:
        result[f"recall@{k}"] = recalls[k] / eligible if eligible else 0.0
    return result


def nearest_neighbor_retrieval(
    query_embeddings: Tensor,
    query_labels: Sequence[str],
    gallery_embeddings: Tensor,
    gallery_labels: Sequence[str],
    class_names: Sequence[str],
    query_paths: Sequence[str] | None = None,
    gallery_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate closed-set nearest-neighbor SKU retrieval with macro Recall@1."""

    if len(query_embeddings) != len(query_labels) or len(gallery_embeddings) != len(gallery_labels):
        raise ValueError("Embedding and label counts must match")
    if not len(query_embeddings) or not len(gallery_embeddings):
        raise ValueError("Query and gallery embeddings must both be non-empty")
    if query_paths is not None and len(query_paths) != len(query_labels):
        raise ValueError("query_paths must match the query count")
    if gallery_paths is not None and len(gallery_paths) != len(gallery_labels):
        raise ValueError("gallery_paths must match the gallery count")
    class_names = list(class_names)
    if len(class_names) != len(set(class_names)):
        raise ValueError("class_names must be unique")
    unknown = (set(query_labels) | set(gallery_labels)) - set(class_names)
    if unknown:
        raise ValueError(f"Labels missing from class_names: {sorted(unknown)}")

    query_embeddings = F.normalize(query_embeddings.float(), dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings.float(), dim=1)
    similarities = query_embeddings @ gallery_embeddings.T
    if query_paths is not None and gallery_paths is not None:
        gallery_path_to_indices: dict[str, list[int]] = {}
        for gallery_index, gallery_path in enumerate(gallery_paths):
            gallery_path_to_indices.setdefault(gallery_path, []).append(gallery_index)
        for query_index, query_path in enumerate(query_paths):
            for gallery_index in gallery_path_to_indices.get(query_path, []):
                similarities[query_index, gallery_index] = float("-inf")
    if torch.isneginf(similarities).all(dim=1).any():
        raise ValueError("At least one query has no eligible gallery image")

    scores, nearest_indices = similarities.max(dim=1)
    nearest_indices_list = nearest_indices.cpu().tolist()
    predicted_labels = [gallery_labels[index] for index in nearest_indices_list]
    correctness = np.asarray(
        [prediction == target for prediction, target in zip(predicted_labels, query_labels)],
        dtype=np.bool_,
    )
    per_sku: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for class_name in class_names:
        class_indices = [index for index, label in enumerate(query_labels) if label == class_name]
        if not class_indices:
            raise ValueError(f"Validation/test queries do not contain class {class_name!r}")
        correct = int(correctness[class_indices].sum())
        recall = correct / len(class_indices)
        recalls.append(recall)
        per_sku[class_name] = {"queries": len(class_indices), "correct": correct, "recall@1": recall}
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, prediction in zip(query_labels, predicted_labels):
        matrix[class_to_idx[target], class_to_idx[prediction]] += 1
    incorrect = []
    for query_index, is_correct in enumerate(correctness):
        if is_correct:
            continue
        gallery_index = nearest_indices_list[query_index]
        incorrect.append(
            {
                "query_index": query_index,
                "query_path": query_paths[query_index] if query_paths is not None else "",
                "query_label": query_labels[query_index],
                "gallery_index": gallery_index,
                "gallery_path": gallery_paths[gallery_index] if gallery_paths is not None else "",
                "predicted_label": predicted_labels[query_index],
                "cosine_similarity": float(scores[query_index].item()),
            }
        )
    return {
        "query_count": len(query_labels),
        "gallery_count": len(gallery_labels),
        "macro_recall@1": float(np.mean(recalls)),
        "micro_recall@1": float(correctness.mean()),
        "per_sku": per_sku,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "predicted_labels": predicted_labels,
        "correct": correctness.tolist(),
        "nearest_gallery_indices": nearest_indices_list,
        "incorrect_nearest_neighbors": incorrect,
    }
