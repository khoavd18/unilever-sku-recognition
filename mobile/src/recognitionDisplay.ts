import type { Detection } from "./types";

type DetectionDisplay = Pick<
  Detection,
  "bbox" | "brand" | "detector_score" | "item_code" | "item_name" | "reason" | "resolved"
>;

export function formatDetectionBBox(bbox: Detection["bbox"]): string {
  return `[${bbox.map((value) => Math.round(value)).join(", ")}] px (x1, y1, x2, y2)`;
}

export function formatDetectionReason(detection: DetectionDisplay): string {
  if (detection.reason === "missing_gallery_for_detected_brand") {
    return `Chưa có gallery cho brand ${detection.brand}; không gán item_code.`;
  }
  if (detection.reason === "similarity_threshold_not_calibrated") {
    return "Có ứng viên AXE nhưng ngưỡng cosine chưa được hiệu chỉnh.";
  }
  if (detection.reason === "below_similarity_threshold") {
    return "Cosine similarity thấp hơn ngưỡng đã hiệu chỉnh.";
  }
  if (detection.reason === "empty_crop") {
    return "Bounding box không tạo được crop hợp lệ.";
  }
  if (detection.reason === "crop_overlaps_invalid_stitched_region") {
    return "Crop chồng lấp quá nhiều vùng panorama không hợp lệ.";
  }
  return detection.reason ? `Lý do: ${detection.reason}` : "Chưa xác định item_code.";
}

export function formatDetectionOverlayLabel(detection: DetectionDisplay): string {
  const detector = `${detection.brand} ${detection.detector_score.toFixed(2)}`;
  if (detection.resolved && detection.item_code) {
    return `${detector}\n${detection.item_code}`;
  }
  if (detection.reason === "missing_gallery_for_detected_brand") {
    return `${detector}\nThiếu gallery`;
  }
  return `${detector}\nChưa có item_code`;
}

export type DetectionCountSummary = {
  total: number;
  resolved: number;
  unresolved: number;
  byBrand: Array<{ brand: string; count: number }>;
};

export function summarizeDetections(
  detections: readonly Pick<Detection, "brand" | "resolved">[],
): DetectionCountSummary {
  const counts = new Map<string, number>();
  let resolved = 0;
  for (const detection of detections) {
    const brand = detection.brand.trim() || "Không rõ brand";
    counts.set(brand, (counts.get(brand) ?? 0) + 1);
    if (detection.resolved) resolved += 1;
  }
  return {
    total: detections.length,
    resolved,
    unresolved: detections.length - resolved,
    byBrand: [...counts.entries()]
      .map(([brand, count]) => ({ brand, count }))
      .sort((left, right) => right.count - left.count || left.brand.localeCompare(right.brand)),
  };
}
