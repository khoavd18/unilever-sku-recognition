"""OSA-ready observations that preserve prediction/correction provenance."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable


def build_osa_summary(
    detections: Iterable[dict[str, Any]],
    expected_item_codes: Iterable[str] | None = None,
    capture_coverage_status: str = "unknown",
) -> dict[str, Any]:
    detections = list(detections)
    brand_counts = Counter(
        str(row["brand"]).strip() for row in detections
        if str(row.get("brand", "")).strip()
    )
    counts = Counter(
        str(row["item_code"]) for row in detections
        if row.get("resolved") and row.get("item_code")
    )
    unresolved = [
        {"detection_id": row.get("detection_id"), "reason": row.get("reason"), "candidate": row.get("candidate")}
        for row in detections if not row.get("resolved")
    ]
    expected = sorted({str(code) for code in expected_item_codes}) if expected_item_codes is not None else None
    comparison_available = expected is not None and capture_coverage_status == "complete"
    return {
        "detected_visible_count": len(detections),
        "brand_visible_counts": dict(sorted(brand_counts.items())),
        "resolved_visible_count": sum(counts.values()),
        "observed_item_codes": sorted(counts),
        "model_visible_counts": dict(sorted(counts.items())),
        "unresolved_detections": unresolved,
        "unresolved_count": len(unresolved),
        "capture_coverage_status": capture_coverage_status,
        "expected_assortment_supplied": expected is not None,
        "osa_status": "observation_complete" if comparison_available else "incomplete_or_unavailable",
        "expected_comparison": {
            "expected_item_codes": expected,
            "not_observed_item_codes": sorted(set(expected or []) - set(counts)) if comparison_available else None,
            "interpretation": (
                "not_observed; do not label out_of_stock without an approved OSA rule"
                if comparison_available else "unavailable until expected assortment and sufficient capture coverage are supplied"
            ),
        },
        "limitations": [
            "Not detected does not mean out of stock.",
            "Visible counts do not represent hidden or total inventory.",
        ],
        "manual_corrections": [],
        "corrected_visible_counts": None,
    }


def apply_count_corrections(
    summary: dict[str, Any],
    corrections: Iterable[dict[str, Any]],
    actor: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Add correction provenance without mutating original model predictions."""

    result = deepcopy(summary)
    corrected = dict(result.get("model_visible_counts", {}))
    events = list(result.get("manual_corrections", []))
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    for correction in corrections:
        item_code = str(correction["item_code"])
        count = int(correction["visible_count"])
        if count < 0:
            raise ValueError("Visible count cannot be negative")
        events.append({
            "item_code": item_code,
            "previous_visible_count": int(corrected.get(item_code, 0)),
            "corrected_visible_count": count,
            "actor": actor,
            "timestamp": timestamp,
            "source": "manual_user_correction",
        })
        corrected[item_code] = count
    result["manual_corrections"] = events
    result["corrected_visible_counts"] = dict(sorted(corrected.items()))
    return result
