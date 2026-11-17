"""Run validation-only reporting for a manually trained nine-brand detector."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_brand_9brands_yolo import TARGET_BRANDS, validate_dataset


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/brand_9brands_yolo/data.yaml")
    parser.add_argument("--output-dir", default="outputs/brand9_validation")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    weights = resolve_path(args.weights)
    data_yaml = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights do not exist: {weights}")
    checks = validate_dataset(data_yaml.parent)
    if not checks["training_ready"]:
        raise RuntimeError("Dataset review is incomplete; validation reporting is blocked")
    if output_dir.exists():
        raise FileExistsError(f"Validation output already exists: {output_dir}")
    if importlib.util.find_spec("ultralytics") is None:
        raise ModuleNotFoundError("Ultralytics is not installed in this interpreter")
    from ultralytics import YOLO, __version__ as ultralytics_version

    if str(ultralytics_version) != "8.3.155":
        raise RuntimeError(f"Expected Ultralytics 8.3.155, found {ultralytics_version}")
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=False,
        plots=True,
    )
    rows = []
    for class_id, brand in enumerate(TARGET_BRANDS):
        precision, recall, map50, map50_95 = metrics.box.class_result(class_id)
        rows.append(
            {
                "class_id": class_id,
                "brand": brand,
                "validation_images": checks["splits"]["val"]["per_brand"][brand]["images"],
                "validation_instances": checks["splits"]["val"]["per_brand"][brand]["instances"],
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map50_95),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_brand_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": "val",
        "test_evaluated": False,
        "ultralytics": str(ultralytics_version),
        "per_brand": rows,
        "macro": {
            "precision": sum(row["precision"] for row in rows) / len(rows),
            "recall": sum(row["recall"] for row in rows) / len(rows),
            "map50": sum(row["map50"] for row in rows) / len(rows),
            "map50_95": sum(row["map50_95"] for row in rows) / len(rows),
        },
    }
    (output_dir / "per_brand_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
