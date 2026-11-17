"""Manually train Model 1 (YOLOv8n) after AXE dataset preparation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import yolo as yolo_training  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/axe_yolo.yaml")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate the Vietnamese summary from an existing completed run without training.",
    )
    return parser.parse_args(argv)


def _resolve(path_value: str | Path) -> Path:
    return yolo_training.resolve_path(path_value, ROOT)


def load_and_validate_config(path: Path) -> dict[str, Any]:
    expected = {
        "model": "yolov8n.pt",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": 0,
        "amp": True,
        "seed": 42,
        "patience": 0,
        "single_cls": False,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    return yolo_training.load_reviewed_yolo_config(
        path,
        required_settings=expected,
        mismatch_message="Model 1 baseline config differs from required settings",
        heldout_message="Model 1 config must explicitly disable held-out test evaluation",
    )


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {data_yaml}. Run scripts/prepare_axe_yolo.py first."
        )
    dataset_root = data_yaml.parent
    checks_path = dataset_root / "dataset_checks.json"
    report_path = dataset_root / "dataset_report.json"
    mapping_path = dataset_root / "mapping.json"
    for path in (checks_path, report_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset metadata is missing: {path}")
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if checks.get("status") != "passed" or checks.get("cross_split_family_overlap") != 0:
        raise ValueError(f"Prepared dataset checks did not pass: {checks}")
    if mapping.get("output_names") != {"0": "AXE"}:
        raise ValueError("Prepared dataset is not the reviewed single-class AXE mapping")
    if report.get("source_version") != 8:
        raise ValueError("Prepared dataset was not built from Roboflow source version 8")
    if report.get("conversion", {}).get("rejected_annotation_count") != 0:
        raise ValueError("Prepared dataset has rejected annotations; review before training")
    return report


def _float(row: dict[str, str], name: str) -> float:
    value = row.get(name)
    if value is None:
        raise KeyError(f"Ultralytics results.csv has no column {name!r}")
    return float(value)


def write_validation_summary(run_dir: Path, dataset_report: dict[str, Any]) -> Path:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Training results.csv was not found: {results_path}")
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{str(key).strip(): str(value).strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"Training results.csv has no epochs: {results_path}")
    metric_names = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    best = max(rows, key=lambda row: _float(row, metric_names["map50_95"]))
    last = rows[-1]
    best_epoch = int(float(best.get("epoch", "0"))) + 1
    last_epoch = int(float(last.get("epoch", "0"))) + 1
    lines = [
        "# Tóm tắt validation Model 1 AXE",
        "",
        f"- Run: `{run_dir}`",
        f"- Dataset: {sum(v['images'] for v in dataset_report['splits'].values())} ảnh, "
        f"{dataset_report['conversion']['output_axe_boxes']} hộp AXE.",
        f"- Epoch tốt nhất theo mAP50-95 validation: {best_epoch}.",
        f"- Precision: {_float(best, metric_names['precision']):.6f}",
        f"- Recall: {_float(best, metric_names['recall']):.6f}",
        f"- mAP50: {_float(best, metric_names['map50']):.6f}",
        f"- mAP50-95: {_float(best, metric_names['map50_95']):.6f}",
        f"- Epoch cuối đã hoàn thành: {last_epoch}.",
        "- Không chạy đánh giá trên test split.",
        "",
        "Đây là bộ detector AXE nhỏ; kiểm tra metadata/hình học không chứng minh mọi nhãn ảnh "
        "đều đúng. Model 2 hiện chỉ nhận dạng 5 SKU AXE, ít hơn 9 lớp AXE nguồn của detector.",
        "",
    ]
    target = run_dir / "validation_summary_vi.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    config = load_and_validate_config(config_path)
    data_yaml = _resolve(config["data"])
    dataset_report = validate_dataset(data_yaml)
    plan = yolo_training.build_yolo_run_plan(
        config,
        config_path,
        root=ROOT,
        resolve_model_argument=False,
    )
    run_dir = plan.run_dir
    if args.summary_only:
        print(write_validation_summary(run_dir, dataset_report))
        return
    yolo_training.ensure_new_run(run_dir)
    yolo_training.execute_yolo_training(
        plan,
        expected_version="8.",
        version_mode="prefix",
        missing_dependency_message=(
            "Ultralytics is not installed in this Python interpreter. Use the existing Python 3.10 "
            "environment shown in the readiness summary; this script never installs or upgrades packages."
        ),
        version_error_factory=lambda _expected, actual: (
            f"Expected an installed Ultralytics 8.x API, found {actual}"
        ),
        payload_factory=lambda version: {
            "ultralytics": version,
            "model": config["model"],
            "data": str(data_yaml),
            "run_dir": str(run_dir),
            "test_evaluation": False,
        },
    )
    summary = write_validation_summary(run_dir, dataset_report)
    print(f"Validation summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
