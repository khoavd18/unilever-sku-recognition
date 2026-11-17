"""Leakage-aware item-code retrieval independent of model inference."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REQUIRED_GALLERY_FIELDS = {
    "item_code", "item_name", "brand", "image_path", "family_id",
    "model_version", "catalog_version", "preprocessing_version",
}


def validate_gallery_runtime_compatibility(
    manifest: dict[str, Any],
    *,
    model_version: str,
    catalog_version: str,
    mapping_version: str,
    preprocessing_version: str | None = None,
    embedding_dimension: int | None = None,
) -> None:
    """Fail closed when gallery/catalog/checkpoint/preprocessing identities differ."""

    expected: dict[str, Any] = {
        "model_version": model_version,
        "catalog_version": catalog_version,
        "mapping_version": mapping_version,
    }
    if preprocessing_version is not None:
        expected["preprocessing_version"] = preprocessing_version
    if embedding_dimension is not None:
        expected["embedding_dimension"] = int(embedding_dimension)
    mismatches = {
        key: {"gallery": manifest.get(key), "runtime": value}
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Gallery runtime compatibility check failed: {mismatches}")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("Embeddings must be a non-empty 2D matrix")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if (norms <= 1e-12).any() or not np.isfinite(value).all():
        raise ValueError("Embeddings contain zero-norm or non-finite rows")
    return value / norms


def validate_gallery(
    embeddings: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> None:
    if len(embeddings) != len(metadata):
        raise ValueError("Gallery embedding and metadata row counts differ")
    _normalize(embeddings)
    if not metadata:
        raise ValueError("Gallery metadata is empty")
    missing_rows = {
        index: sorted(REQUIRED_GALLERY_FIELDS - set(row))
        for index, row in enumerate(metadata)
        if REQUIRED_GALLERY_FIELDS - set(row)
    }
    if missing_rows:
        first_index = min(missing_rows)
        raise ValueError(
            "Gallery metadata is missing fields at row "
            f"{first_index}: {missing_rows[first_index]}"
        )
    invalid_item_codes = [
        index
        for index, row in enumerate(metadata)
        if not isinstance(row.get("item_code"), str) or not row["item_code"].strip()
    ]
    if invalid_item_codes:
        raise ValueError(
            "Gallery item_code values must be non-empty strings; invalid rows: "
            f"{invalid_item_codes[:10]}"
        )
    versions = {
        key: {str(row.get(key)) for row in metadata}
        for key in ("model_version", "catalog_version", "preprocessing_version")
    }
    inconsistent = {key: sorted(values) for key, values in versions.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Gallery has inconsistent version metadata: {inconsistent}")
    if manifest:
        if int(manifest["embedding_dimension"]) != int(embeddings.shape[1]):
            raise ValueError("Gallery manifest embedding dimension is incompatible")
        if int(manifest["entry_count"]) != len(metadata):
            raise ValueError("Gallery manifest entry count is incompatible")
        if not bool(manifest.get("l2_normalized")):
            raise ValueError("Gallery manifest does not declare L2-normalized embeddings")
        if manifest.get("similarity") != "cosine":
            raise ValueError("Gallery manifest similarity must be cosine")
        if manifest.get("cosine_is_calibrated_probability") is not False:
            raise ValueError(
                "Gallery manifest must explicitly state that cosine is not a calibrated probability"
            )
        for key in ("model_version", "catalog_version", "preprocessing_version"):
            if str(manifest[key]) not in versions[key]:
                raise ValueError(f"Gallery manifest {key} is incompatible with metadata")


def _same_path(first: Any, second: Any) -> bool:
    if not first or not second:
        return False
    try:
        return str(Path(str(first)).resolve()).casefold() == str(Path(str(second)).resolve()).casefold()
    except OSError:
        return str(first).casefold() == str(second).casefold()


def _leak_reason(query: dict[str, Any], gallery: dict[str, Any]) -> str | None:
    if _same_path(query.get("image_path"), gallery.get("image_path")):
        return "self_match"
    query_hash, gallery_hash = query.get("image_sha256"), gallery.get("image_sha256")
    if query_hash and gallery_hash and query_hash == gallery_hash:
        return "exact_duplicate"
    query_family, gallery_family = query.get("family_id"), gallery.get("family_id")
    if query_family and gallery_family and query_family == gallery_family:
        return "source_family_overlap"
    return None


def evaluate_item_code_retrieval(
    query_embeddings: np.ndarray,
    query_metadata: Sequence[dict[str, Any]],
    gallery_embeddings: np.ndarray,
    gallery_metadata: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate nearest-image Recall@1 after excluding known leakage.

    Micro Recall@1 denominator = queries whose true item code still has at
    least one eligible gallery image after self/hash/family exclusions.
    Missing-gallery and no-eligible-positive queries are reported separately.
    """

    queries = _normalize(query_embeddings)
    gallery = _normalize(gallery_embeddings)
    if len(queries) != len(query_metadata):
        raise ValueError("Query embedding and metadata row counts differ")
    validate_gallery(gallery, gallery_metadata)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("Query/gallery embedding dimensions differ")
    if any(not row.get("item_code") for row in query_metadata):
        raise ValueError("Every query needs a true item_code")

    similarities = queries @ gallery.T
    all_gallery_codes = {str(row["item_code"]) for row in gallery_metadata}
    per_item: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"query_count": 0, "eligible_query_count": 0, "correct": 0}
    )
    missing_gallery: list[dict[str, Any]] = []
    excluded_no_positive: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    incorrect: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    excluded_counts: Counter[str] = Counter()

    for query_index, query in enumerate(query_metadata):
        true_code = str(query["item_code"])
        per_item[true_code]["query_count"] += 1
        if true_code not in all_gallery_codes:
            row = {
                "query_index": query_index, "query_path": query.get("image_path"),
                "item_code": true_code, "reason": "true_item_code_missing_from_gallery",
            }
            missing_gallery.append(row)
            details.append({**row, "eligible": False, "predicted_item_code": None})
            continue
        eligible_indices: list[int] = []
        eligible_positive = False
        query_excluded = Counter()
        for gallery_index, gallery_row in enumerate(gallery_metadata):
            reason = _leak_reason(query, gallery_row)
            if reason:
                query_excluded[reason] += 1
                excluded_counts[reason] += 1
                continue
            eligible_indices.append(gallery_index)
            if str(gallery_row["item_code"]) == true_code:
                eligible_positive = True
        if not eligible_indices or not eligible_positive:
            reason = "no_gallery_candidates_after_exclusions" if not eligible_indices else "no_positive_after_exclusions"
            row = {
                "query_index": query_index, "query_path": query.get("image_path"),
                "item_code": true_code, "reason": reason,
                "excluded_candidates": dict(query_excluded),
            }
            excluded_no_positive.append(row)
            details.append({**row, "eligible": False, "predicted_item_code": None})
            continue
        local_scores = similarities[query_index, eligible_indices]
        nearest_index = eligible_indices[int(np.argmax(local_scores))]
        nearest = gallery_metadata[nearest_index]
        predicted = str(nearest["item_code"])
        score = float(similarities[query_index, nearest_index])
        correct = predicted == true_code
        per_item[true_code]["eligible_query_count"] += 1
        per_item[true_code]["correct"] += int(correct)
        confusion[(true_code, predicted)] += 1
        detail = {
            "query_index": query_index,
            "query_path": query.get("image_path"),
            "true_item_code": true_code,
            "predicted_item_code": predicted,
            "matched_gallery_index": nearest_index,
            "matched_gallery_path": nearest.get("image_path"),
            "cosine_similarity": score,
            "correct": correct,
            "eligible": True,
            "excluded_candidates": dict(query_excluded),
        }
        details.append(detail)
        if not correct:
            incorrect.append(detail)

    eligible_count = sum(row["eligible_query_count"] for row in per_item.values())
    correct_count = sum(row["correct"] for row in per_item.values())
    recalls: list[float] = []
    per_item_output: dict[str, dict[str, Any]] = {}
    for item_code in sorted(per_item):
        row = dict(per_item[item_code])
        denominator = int(row["eligible_query_count"])
        row["recall@1"] = row["correct"] / denominator if denominator else None
        if denominator:
            recalls.append(float(row["recall@1"]))
        per_item_output[item_code] = row
    confusion_rows = [
        {"true_item_code": true, "predicted_item_code": predicted, "count": count}
        for (true, predicted), count in sorted(confusion.items())
    ]
    return {
        "metric_label": "item_code",
        "query_count": len(query_metadata),
        "gallery_count": len(gallery_metadata),
        "gallery_item_count": len(all_gallery_codes),
        "eligible_query_count": eligible_count,
        "correct_query_count": correct_count,
        "evaluation_denominator": (
            "Queries whose true item_code has at least one gallery image remaining after "
            "self-match, exact-duplicate, and known source-family exclusions."
        ),
        "micro_recall@1": correct_count / eligible_count if eligible_count else None,
        "macro_recall@1": float(np.mean(recalls)) if recalls else None,
        "macro_denominator_item_count": len(recalls),
        "gallery_query_coverage": (len(query_metadata) - len(missing_gallery)) / len(query_metadata)
        if query_metadata else 0.0,
        "missing_gallery_query_count": len(missing_gallery),
        "missing_gallery_queries": missing_gallery,
        "excluded_no_positive_query_count": len(excluded_no_positive),
        "excluded_no_positive_queries": excluded_no_positive,
        "candidate_exclusion_counts": dict(excluded_counts),
        "per_item": per_item_output,
        "confusion_table": confusion_rows,
        "incorrect_matches": incorrect,
        "query_results": details,
        "limitations": [
            "Family IDs derived from filenames cannot prove capture-session independence.",
            "Cosine similarity is not a calibrated probability.",
        ],
    }


class GalleryIndex:
    """Validated full-gallery cosine search used by panorama recognition."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict[str, Any]], manifest: dict[str, Any]):
        validate_gallery(embeddings, metadata, manifest)
        self.embeddings = _normalize(embeddings)
        self.metadata = metadata
        self.manifest = manifest

    @classmethod
    def load(cls, directory: str | Path) -> "GalleryIndex":
        root = Path(directory)
        manifest = json.loads((root / "gallery_manifest.json").read_text(encoding="utf-8"))
        embeddings = np.load(root / manifest.get("embeddings_file", "embeddings.npy"), allow_pickle=False)
        with (root / manifest.get("metadata_file", "metadata.jsonl")).open(encoding="utf-8") as handle:
            metadata = [json.loads(line) for line in handle if line.strip()]
        return cls(embeddings, metadata, manifest)

    def search(self, query_embeddings: np.ndarray) -> list[dict[str, Any]]:
        queries = _normalize(query_embeddings)
        if queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match gallery")
        scores = queries @ self.embeddings.T
        indexes = scores.argmax(axis=1)
        return [
            {
                "gallery_index": int(index),
                "cosine_similarity": float(scores[row_index, index]),
                **self.metadata[int(index)],
            }
            for row_index, index in enumerate(indexes)
        ]
