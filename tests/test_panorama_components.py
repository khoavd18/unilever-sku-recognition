from __future__ import annotations

import json
import unittest

import numpy as np

from src.osa.results import apply_count_corrections, build_osa_summary
from src.recognition.tiling import Tile, generate_tiles, map_box_to_panorama, suppress_duplicate_boxes
from src.stitching.two_image import StitchConfig, stitch_arrays
from src.stitching.sequence import stitch_image_sequence


class PanoramaComponentTests(unittest.TestCase):
    def test_tile_coordinate_mapping_and_boundary_coverage(self) -> None:
        tiles = generate_tiles(2500, 900, 1000, 0.2)
        self.assertEqual(tiles[0].x, 0)
        self.assertEqual(max(tile.x + tile.width for tile in tiles), 2500)
        mapped = map_box_to_panorama([10, 20, 100, 200], Tile("t", 800, 50, 1000, 850), 2500, 900)
        self.assertEqual(mapped, [810.0, 70.0, 900.0, 250.0])

    def test_duplicate_suppression_across_tiles(self) -> None:
        detections = [
            {"bbox_xyxy": [100, 100, 300, 400], "detector_score": 0.9, "detector_class_id": 0},
            {"bbox_xyxy": [105, 105, 298, 398], "detector_score": 0.8, "detector_class_id": 0},
            {"bbox_xyxy": [500, 100, 600, 300], "detector_score": 0.7, "detector_class_id": 0},
        ]
        kept = suppress_duplicate_boxes(detections, 0.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["detector_score"], 0.9)

    def test_synthetic_stitch_success_and_rejection(self) -> None:
        import cv2

        rng = np.random.default_rng(42)
        base = rng.integers(0, 256, size=(420, 1000, 3), dtype=np.uint8)
        for x in range(40, 960, 80):
            cv2.circle(base, (x, 80 + (x % 240)), 18, (255, 255, 255), 3)
            cv2.putText(base, str(x), (x - 20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        result = stitch_arrays(base[:, :650], base[:, 350:])
        self.assertTrue(result.success, result.report)
        self.assertGreater(result.panorama.shape[1], 900)
        blank = np.full((300, 500, 3), 127, np.uint8)
        rejected = stitch_arrays(blank, blank, StitchConfig(min_sharpness=0.0))
        self.assertFalse(rejected.success)
        self.assertIn(rejected.reason_code, {"insufficient_features", "insufficient_overlap"})
        resource_limited = stitch_arrays(
            base[:, :650],
            base[:, 350:],
            StitchConfig(max_canvas_megapixels=0.1),
        )
        self.assertFalse(resource_limited.success)
        self.assertEqual(resource_limited.reason_code, "implausible_warp")

    def test_ordered_three_image_stitch(self) -> None:
        import cv2
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, size=(420, 1400, 3), dtype=np.uint8)
        for x in range(30, 1370, 70):
            cv2.circle(base, (x, 60 + (x % 280)), 14, (255, 255, 255), 3)
            cv2.putText(base, str(x), (x - 15, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        crops = [base[:, 0:700], base[:, 350:1050], base[:, 700:1400]]
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = []
            for index, crop in enumerate(crops):
                path = Path(temporary_directory) / f"capture_{index}.jpg"
                self.assertTrue(cv2.imwrite(str(path), crop))
                paths.append(path)
            result = stitch_image_sequence(paths)
        self.assertTrue(result.success, result.report)
        self.assertEqual(result.report["completed_capture_count"], 3)
        self.assertEqual(len(result.report["stages"]), 2)
        self.assertGreater(result.panorama.shape[1], 1200)

    def test_unresolved_serialization_and_manual_correction_separation(self) -> None:
        detections = [{
            "detection_id": "det_0001", "resolved": False, "item_code": None,
            "brand": "AXE",
            "reason": "similarity_threshold_not_calibrated",
            "candidate": {"item_code": "69566283"},
        }, {
            "detection_id": "det_0002", "resolved": True, "item_code": "69566283",
            "brand": "AXE",
            "reason": None, "candidate": None,
        }]
        summary = build_osa_summary(detections)
        serialized = json.loads(json.dumps(summary))
        self.assertEqual(serialized["unresolved_count"], 1)
        self.assertEqual(serialized["detected_visible_count"], 2)
        self.assertEqual(serialized["brand_visible_counts"], {"AXE": 2})
        self.assertEqual(serialized["resolved_visible_count"], 1)
        corrected = apply_count_corrections(summary, [{"item_code": "69566283", "visible_count": 3}], "tester", "2026-01-01T00:00:00Z")
        self.assertEqual(summary["model_visible_counts"]["69566283"], 1)
        self.assertEqual(corrected["model_visible_counts"]["69566283"], 1)
        self.assertEqual(corrected["corrected_visible_counts"]["69566283"], 3)
        self.assertEqual(corrected["manual_corrections"][0]["source"], "manual_user_correction")
        self.assertEqual(summary["osa_status"], "incomplete_or_unavailable")


if __name__ == "__main__":
    unittest.main()
