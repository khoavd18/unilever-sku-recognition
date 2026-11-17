"""Overlapping panorama tiles and duplicate-box suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x: int
    y: int
    width: int
    height: int


def _starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_tiles(width: int, height: int, tile_size: int, overlap: float = 0.2) -> list[Tile]:
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("Image and tile dimensions must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Tile overlap must be in [0, 1)")
    tiles = []
    for row, y in enumerate(_starts(height, tile_size, overlap)):
        for column, x in enumerate(_starts(width, tile_size, overlap)):
            tiles.append(Tile(
                f"tile_{row:03d}_{column:03d}", x, y,
                min(tile_size, width - x), min(tile_size, height - y),
            ))
    return tiles


def map_box_to_panorama(box_xyxy: list[float] | tuple[float, ...], tile: Tile, width: int, height: int) -> list[float]:
    if len(box_xyxy) != 4:
        raise ValueError("Box must use xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    mapped = [
        min(max(x1 + tile.x, 0.0), float(width)),
        min(max(y1 + tile.y, 0.0), float(height)),
        min(max(x2 + tile.x, 0.0), float(width)),
        min(max(y2 + tile.y, 0.0), float(height)),
    ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("Mapped box is empty")
    return mapped


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def suppress_duplicate_boxes(
    detections: list[dict[str, Any]], iou_threshold: float = 0.5, class_aware: bool = True,
) -> list[dict[str, Any]]:
    """Greedy NMS after tile boxes have been mapped to panorama pixels."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be in [0, 1]")
    ordered = sorted(detections, key=lambda row: float(row["detector_score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            same_class = candidate.get("detector_class_id") == existing.get("detector_class_id")
            if (same_class or not class_aware) and box_iou(candidate["bbox_xyxy"], existing["bbox_xyxy"]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def valid_region_ratio(mask: np.ndarray | None, tile: Tile) -> float:
    if mask is None:
        return 1.0
    region = mask[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
    return float((region > 0).mean()) if region.size else 0.0
