"""Stitch two ordered full-resolution photos or return an explicit retake reason."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stitching.two_image import StitchConfig, stitch_two_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="First photo in capture order")
    parser.add_argument("--second", required=True, help="Adjacent second photo in capture order")
    parser.add_argument("--capture-metadata", default=None, help="Optional JSON file from the phone")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Stitch output exists; choose a new directory: {output}")
    capture_metadata = None
    if args.capture_metadata:
        capture_metadata = json.loads(Path(args.capture_metadata).read_text(encoding="utf-8"))
    result = stitch_two_images(args.first, args.second, StitchConfig(), capture_metadata)
    output.mkdir(parents=True)
    (output / "stitch_report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import cv2

    if result.match_preview is not None:
        cv2.imwrite(str(output / "match_preview.jpg"), result.match_preview)
    if result.alignment_preview is not None:
        cv2.imwrite(str(output / "alignment_preview.jpg"), result.alignment_preview)
    if result.success:
        cv2.imwrite(str(output / "panorama.jpg"), result.panorama)
        cv2.imwrite(str(output / "valid_mask.png"), result.valid_mask)
    print(json.dumps({
        "success": result.success, "status": result.status, "reason_code": result.reason_code,
        "message": result.message, "output_dir": str(output),
    }, ensure_ascii=False, indent=2))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
