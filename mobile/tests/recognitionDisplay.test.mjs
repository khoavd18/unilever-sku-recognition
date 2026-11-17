import assert from "node:assert/strict";
import test from "node:test";

import {
  formatDetectionBBox,
  formatDetectionOverlayLabel,
  formatDetectionReason,
  summarizeDetections,
} from "../src/recognitionDisplay.ts";

const unsupported = {
  bbox: [10.2, 20.4, 110.7, 220.8],
  brand: "VIM",
  detector_score: 0.91234,
  item_code: null,
  item_name: null,
  reason: "missing_gallery_for_detected_brand",
  resolved: false,
};

test("unsupported brand display retains brand, confidence, bbox and missing-gallery reason", () => {
  assert.equal(formatDetectionOverlayLabel(unsupported), "VIM 0.91\nThiếu gallery");
  assert.equal(
    formatDetectionReason(unsupported),
    "Chưa có gallery cho brand VIM; không gán item_code.",
  );
  assert.equal(formatDetectionBBox(unsupported.bbox), "[10, 20, 111, 221] px (x1, y1, x2, y2)");
});

test("resolved AXE overlay distinguishes detector confidence from item identity", () => {
  const resolved = {
    ...unsupported,
    brand: "AXE",
    detector_score: 0.876,
    item_code: "69566283",
    item_name: "AXE Black",
    reason: null,
    resolved: true,
  };
  assert.equal(formatDetectionOverlayLabel(resolved), "AXE 0.88\n69566283");
});

test("summarizes visible detections by brand without treating unresolved SKU as resolved", () => {
  assert.deepEqual(
    summarizeDetections([
      { brand: "CLEAR", resolved: false },
      { brand: "AXE", resolved: true },
      { brand: "CLEAR", resolved: false },
      { brand: "DOVE", resolved: false },
    ]),
    {
      total: 4,
      resolved: 1,
      unresolved: 3,
      byBrand: [
        { brand: "CLEAR", count: 2 },
        { brand: "AXE", count: 1 },
        { brand: "DOVE", count: 1 },
      ],
    },
  );
});
