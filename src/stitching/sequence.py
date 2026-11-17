"""Ordered multi-image stitching built on the existing quality-gated pair stitcher."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .two_image import (
    StitchConfig,
    StitchResult,
    _preview,
    load_oriented_bgr,
    stitch_arrays,
)


def stitch_image_sequence(
    image_paths: Sequence[str | Path],
    config: StitchConfig | None = None,
    capture_metadata: dict[str, Any] | None = None,
) -> StitchResult:
    """Stitch an ordered sweep, stopping at the first failed adjacent stage.

    This deliberately reuses the pair stitcher's quality, overlap, homography,
    reprojection, spatial-coverage, and canvas gates. It does not substitute a
    guessed translation when a pair fails.
    """

    config = config or StitchConfig()
    paths = [Path(path) for path in image_paths]
    if len(paths) < 2:
        raise ValueError("At least two ordered images are required")

    loaded = [load_oriented_bgr(path) for path in paths]
    panorama, first_metadata = loaded[0]
    source_metadata = [first_metadata]
    stages: list[dict[str, Any]] = []
    valid_mask = np.full(panorama.shape[:2], 255, dtype=np.uint8)
    last_match_preview = None

    for next_index, (next_image, next_metadata) in enumerate(loaded[1:], start=1):
        stage = stitch_arrays(
            panorama,
            next_image,
            config=config,
            capture_metadata={
                "sequence_stage": next_index,
                "source_capture_metadata": capture_metadata,
            },
            source_metadata=[
                {
                    "kind": "accumulated_panorama",
                    "source_count": next_index,
                    "width": int(panorama.shape[1]),
                    "height": int(panorama.shape[0]),
                },
                next_metadata,
            ],
        )
        stages.append({
            "stage": next_index,
            "added_capture_index": next_index,
            "success": stage.success,
            "reason_code": stage.reason_code,
            "message": stage.message,
            "report": stage.report,
        })
        last_match_preview = stage.match_preview
        if not stage.success:
            report = {
                "schema_version": 1,
                "mode": "ordered_sequence",
                "success": False,
                "status": "retake_required",
                "reason_code": stage.reason_code,
                "message": f"capture pair {next_index}/{next_index + 1}: {stage.message}",
                "input_count": len(paths),
                "completed_capture_count": next_index,
                "failed_capture_index": next_index,
                "capture_order": [path.name for path in paths],
                "capture_metadata": capture_metadata,
                "config": asdict(config),
                "stages": stages,
                "limitations": [
                    "Sequential homographies do not provide global bundle adjustment.",
                    "Repeated packaging and shelf parallax can still create false matches or ghosting.",
                ],
            }
            return StitchResult(
                False,
                "retake_required",
                stage.reason_code,
                report["message"],
                report,
                match_preview=last_match_preview,
            )

        assert stage.panorama is not None and stage.valid_mask is not None
        panorama = stage.panorama
        valid_mask = stage.valid_mask
        source_metadata.append(next_metadata)

    report = {
        "schema_version": 1,
        "mode": "ordered_sequence",
        "success": True,
        "status": "suitable_for_stitching",
        "reason_code": None,
        "message": "all ordered captures passed pairwise quality gates",
        "input_count": len(paths),
        "completed_capture_count": len(paths),
        "capture_order": [path.name for path in paths],
        "capture_metadata": capture_metadata,
        "source_images": source_metadata,
        "config": asdict(config),
        "stages": stages,
        "panorama_width": int(panorama.shape[1]),
        "panorama_height": int(panorama.shape[0]),
        "valid_pixel_ratio": float((valid_mask > 0).mean()),
        "limitations": [
            "Sequential homographies do not provide global bundle adjustment.",
            "A successful output does not prove absence of parallax, ghosting, or seam duplication.",
        ],
    }
    return StitchResult(
        True,
        "suitable_for_stitching",
        None,
        report["message"],
        report,
        panorama=panorama,
        valid_mask=valid_mask,
        match_preview=last_match_preview,
        alignment_preview=_preview(panorama),
    )
