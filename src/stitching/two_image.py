"""Quality-gated two-image panorama stitching with explicit retake reasons."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class StitchConfig:
    max_feature_side: int = 1800
    max_features: int = 5000
    ratio_test: float = 0.75
    ransac_reprojection_threshold: float = 4.0
    min_keypoints: int = 40
    min_good_matches: int = 20
    min_inliers: int = 14
    min_inlier_ratio: float = 0.35
    min_spatial_coverage: float = 0.015
    min_overlap_ratio: float = 0.08
    max_median_reprojection_error: float = 3.5
    min_sharpness: float = 12.0
    min_mean_brightness: float = 8.0
    max_mean_brightness: float = 247.0
    min_dimension: int = 120
    max_canvas_megapixels: float = 120.0


@dataclass
class StitchResult:
    success: bool
    status: str
    reason_code: str | None
    message: str
    report: dict[str, Any]
    panorama: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    match_preview: np.ndarray | None = None
    alignment_preview: np.ndarray | None = None


def load_oriented_bgr(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply EXIF orientation once and return an OpenCV BGR image."""

    source = Path(path)
    with Image.open(source) as image:
        exif_orientation = image.getexif().get(274)
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(oriented)
    return rgb[:, :, ::-1].copy(), {
        "path": str(source.resolve()),
        "exif_orientation": exif_orientation,
        "orientation_applied": exif_orientation not in (None, 1),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }


def _resize_for_features(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, scale
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def _quality(image: np.ndarray, config: StitchConfig) -> dict[str, Any]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons = []
    if min(height, width) < config.min_dimension:
        reasons.append("image_too_small")
    if sharpness < config.min_sharpness:
        reasons.append("image_too_blurry")
    if brightness < config.min_mean_brightness:
        reasons.append("image_too_dark")
    if brightness > config.max_mean_brightness:
        reasons.append("image_overexposed")
    return {
        "width": width, "height": height, "sharpness_laplacian_variance": sharpness,
        "mean_brightness": brightness, "valid": not reasons, "reasons": reasons,
    }


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    import cv2

    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (shape[0] * shape[1]))


def _fail(
    code: str,
    message: str,
    report: dict[str, Any],
    match_preview: np.ndarray | None = None,
) -> StitchResult:
    report.update({"success": False, "status": "retake_required", "reason_code": code, "message": message})
    return StitchResult(False, "retake_required", code, message, report, match_preview=match_preview)


def _overlap_ratio(homography: np.ndarray, left_shape: tuple[int, int], right_shape: tuple[int, int]) -> tuple[float, np.ndarray]:
    import cv2

    lh, lw = left_shape
    rh, rw = right_shape
    left_quad = np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    right_quad = np.float32([[[0, 0], [rw, 0], [rw, rh], [0, rh]]])
    transformed = cv2.perspectiveTransform(right_quad, homography)[0]
    if not np.isfinite(transformed).all() or not cv2.isContourConvex(transformed.astype(np.float32)):
        return 0.0, transformed
    transformed_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
    try:
        overlap_area, _ = cv2.intersectConvexConvex(left_quad, transformed.astype(np.float32))
    except cv2.error:
        overlap_area = 0.0
    denominator = min(float(lw * lh), transformed_area)
    return (float(overlap_area) / denominator if denominator > 0 else 0.0), transformed


def _compose_panorama(
    left: np.ndarray,
    right: np.ndarray,
    homography_right_to_left: np.ndarray,
    config: StitchConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2

    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    left_corners = np.float64([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    right_corners = cv2.perspectiveTransform(
        np.float64([[[0, 0], [rw, 0], [rw, rh], [0, rh]]]), homography_right_to_left
    )[0]
    all_corners = np.vstack([left_corners, right_corners])
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = int(max_xy[0] - min_xy[0]), int(max_xy[1] - min_xy[1])
    if width <= 0 or height <= 0:
        raise ValueError("implausible_warp_bounds")
    if width * height > config.max_canvas_megapixels * 1_000_000:
        raise ValueError("implausible_warp_canvas_too_large")
    translation = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]])
    left_transform = translation
    right_transform = translation @ homography_right_to_left
    left_warp = cv2.warpPerspective(left, left_transform, (width, height))
    right_warp = cv2.warpPerspective(right, right_transform, (width, height))
    left_mask = cv2.warpPerspective(np.full((lh, lw), 255, np.uint8), left_transform, (width, height)) > 0
    right_mask = cv2.warpPerspective(np.full((rh, rw), 255, np.uint8), right_transform, (width, height)) > 0
    valid = left_mask | right_mask
    # Distance-weighted feathering keeps full-resolution source pixels and avoids a hard seam.
    left_distance = cv2.distanceTransform(left_mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    right_distance = cv2.distanceTransform(right_mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    denominator = left_distance + right_distance
    left_weight = np.divide(left_distance, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    right_weight = np.divide(right_distance, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    panorama = (
        left_warp.astype(np.float32) * left_weight[:, :, None]
        + right_warp.astype(np.float32) * right_weight[:, :, None]
    ).clip(0, 255).astype(np.uint8)
    return panorama, (valid.astype(np.uint8) * 255), {
        "canvas_width": width,
        "canvas_height": height,
        "left_to_panorama": left_transform.tolist(),
        "right_to_panorama": right_transform.tolist(),
        "right_to_left": homography_right_to_left.tolist(),
    }


def _preview(image: np.ndarray, max_side: int = 1400) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    return image if scale == 1.0 else cv2.resize(image, (round(width * scale), round(height * scale)))


def stitch_arrays(
    left: np.ndarray,
    right: np.ndarray,
    config: StitchConfig | None = None,
    capture_metadata: dict[str, Any] | None = None,
    source_metadata: list[dict[str, Any]] | None = None,
) -> StitchResult:
    import cv2

    config = config or StitchConfig()
    report: dict[str, Any] = {
        "schema_version": 1,
        "capture_order": ["left_or_first", "right_or_second"],
        "capture_metadata": capture_metadata,
        "source_images": source_metadata,
        "config": asdict(config),
        "geometry_note": (
            "A robust homography only indicates a usable alignment model for these views; "
            "it does not prove exact scene coplanarity or absence of shelf/product parallax."
        ),
    }
    qualities = [_quality(left, config), _quality(right, config)]
    report["image_quality"] = qualities
    if not all(item["valid"] for item in qualities):
        return _fail("invalid_image_quality", "retake required: image quality check failed", report)

    left_work, left_scale = _resize_for_features(left, config.max_feature_side)
    right_work, right_scale = _resize_for_features(right, config.max_feature_side)
    left_gray = cv2.cvtColor(left_work, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_work, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=config.max_features)
        norm = cv2.NORM_L2
        feature_name = "SIFT"
    else:  # pragma: no cover - modern OpenCV distributions include SIFT
        detector = cv2.ORB_create(nfeatures=config.max_features)
        norm = cv2.NORM_HAMMING
        feature_name = "ORB"
    keypoints_left, descriptors_left = detector.detectAndCompute(left_gray, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(right_gray, None)
    report["features"] = {
        "method": feature_name,
        "left_keypoints": len(keypoints_left), "right_keypoints": len(keypoints_right),
        "left_work_scale": left_scale, "right_work_scale": right_scale,
    }
    if (
        descriptors_left is None or descriptors_right is None
        or min(len(keypoints_left), len(keypoints_right)) < config.min_keypoints
    ):
        return _fail("insufficient_features", "retake required: insufficient visual features", report)
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(descriptors_right, descriptors_left, k=2)
    good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < config.ratio_test * second.distance]
    report["features"]["ratio_test_matches"] = len(good)
    raw_preview = cv2.drawMatches(
        right_work, keypoints_right, left_work, keypoints_left, good[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    ) if good else None
    if len(good) < config.min_good_matches:
        return _fail("insufficient_overlap", "retake required: insufficient overlap", report, raw_preview)

    points_right = np.float32([keypoints_right[match.queryIdx].pt for match in good])
    points_left = np.float32([keypoints_left[match.trainIdx].pt for match in good])
    homography_work, inlier_mask = cv2.findHomography(
        points_right, points_left, cv2.RANSAC, config.ransac_reprojection_threshold
    )
    if homography_work is None or inlier_mask is None or not np.isfinite(homography_work).all():
        return _fail("unstable_alignment", "retake required: unstable alignment", report, raw_preview)
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(good)
    projected = cv2.perspectiveTransform(points_right.reshape(-1, 1, 2), homography_work).reshape(-1, 2)
    errors = np.linalg.norm(projected - points_left, axis=1)
    median_error = float(np.median(errors[inliers])) if inlier_count else math.inf
    left_coverage = _coverage(points_left[inliers], left_gray.shape)
    right_coverage = _coverage(points_right[inliers], right_gray.shape)
    report["alignment"] = {
        "good_matches": len(good), "inliers": inlier_count, "inlier_ratio": inlier_ratio,
        "median_reprojection_error_work_pixels": median_error,
        "left_inlier_spatial_coverage": left_coverage,
        "right_inlier_spatial_coverage": right_coverage,
    }
    inlier_matches = [match for match, keep in zip(good, inliers) if keep]
    match_preview = cv2.drawMatches(
        right_work, keypoints_right, left_work, keypoints_left, inlier_matches[:150], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if inlier_count < config.min_inliers or inlier_ratio < config.min_inlier_ratio:
        return _fail("unstable_alignment", "retake required: unstable alignment", report, match_preview)
    if min(left_coverage, right_coverage) < config.min_spatial_coverage:
        return _fail("poor_match_coverage", "retake required: matches are too spatially concentrated", report, match_preview)
    if median_error > config.max_median_reprojection_error:
        return _fail("high_reprojection_error", "retake required: unstable alignment", report, match_preview)

    scale_left = np.diag([left_scale, left_scale, 1.0])
    scale_right = np.diag([right_scale, right_scale, 1.0])
    homography_full = np.linalg.inv(scale_left) @ homography_work @ scale_right
    if abs(homography_full[2, 2]) < 1e-12:
        return _fail("implausible_warp", "retake required: implausible warp", report, match_preview)
    homography_full /= homography_full[2, 2]
    overlap_ratio, transformed_quad = _overlap_ratio(
        homography_full, left.shape[:2], right.shape[:2]
    )
    report["alignment"].update({
        "estimated_overlap_ratio": overlap_ratio,
        "transformed_right_corners_in_left": transformed_quad.tolist(),
        "right_to_left_homography": homography_full.tolist(),
    })
    if overlap_ratio < config.min_overlap_ratio:
        return _fail("insufficient_overlap", "retake required: insufficient overlap", report, match_preview)
    try:
        panorama, valid_mask, transforms = _compose_panorama(left, right, homography_full, config)
    except ValueError as exc:
        return _fail("implausible_warp", f"retake required: {exc}", report, match_preview)
    report["transforms"] = transforms
    report.update({
        "success": True, "status": "suitable_for_stitching", "reason_code": None,
        "message": "suitable for stitching",
        "valid_pixel_ratio": float((valid_mask > 0).mean()),
    })
    return StitchResult(
        True, "suitable_for_stitching", None, "suitable for stitching", report,
        panorama=panorama, valid_mask=valid_mask, match_preview=match_preview,
        alignment_preview=_preview(panorama),
    )


def stitch_two_images(
    first_path: str | Path,
    second_path: str | Path,
    config: StitchConfig | None = None,
    capture_metadata: dict[str, Any] | None = None,
) -> StitchResult:
    first, first_metadata = load_oriented_bgr(first_path)
    second, second_metadata = load_oriented_bgr(second_path)
    return stitch_arrays(first, second, config, capture_metadata, [first_metadata, second_metadata])
