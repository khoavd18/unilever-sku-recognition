"""Manually train the prepared five-brand Model 1 dataset.

The ``--preflight`` mode is read-only and must be used to verify the exact
dataset/config/interpreter before manually starting a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_brand_5brands_yolo import TARGET_BRANDS, validate_dataset
from src.training import yolo as yolo_training


def resolve_path(value: str | Path) -> Path:
    return yolo_training.resolve_path(value, ROOT)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/brand_5brands_yolo.yaml")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate inputs and print the resolved run without training.",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    required = {
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
        "resume": False,
        "exist_ok": False,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    return yolo_training.load_reviewed_yolo_config(
        path,
        required_settings=required,
        mismatch_message="Five-brand baseline config differs from reviewed settings",
        heldout_message="Config must explicitly keep held-out test evaluation disabled",
    )


def preflight(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    data_yaml = resolve_path(config["data"])
    checks = validate_dataset(data_yaml.parent)
    expected = {str(index): brand for index, brand in enumerate(TARGET_BRANDS)}
    if checks.get("mapping_verified") != expected:
        raise ValueError(f"Prepared dataset class order is incompatible: {checks}")
    if checks.get("test_split_evaluated") is not False:
        raise ValueError("Prepared dataset unexpectedly records test evaluation")
    plan = yolo_training.build_yolo_run_plan(config, config_path, root=ROOT)
    yolo_training.ensure_new_run(plan.run_dir)
    summary = {
        "python": sys.executable,
        "config": str(config_path),
        "model": plan.model_argument,
        "data": str(data_yaml),
        "mapping": expected,
        "run_dir": str(plan.run_dir),
        "epochs": config["train"]["epochs"],
        "test_evaluation": False,
        "training_started": False,
    }
    return config, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_path(args.config)
    config, summary = preflight(config_path)
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    expected_version = str(config.get("compatibility", {}).get("verified_ultralytics"))
    plan = yolo_training.build_yolo_run_plan(config, config_path, root=ROOT)
    yolo_training.execute_yolo_training(
        plan,
        expected_version=expected_version,
        missing_dependency_message="Ultralytics is not installed in this interpreter",
        version_error_factory=lambda expected, actual: (
            f"Expected reviewed Ultralytics {expected}, found {actual}"
        ),
        payload_factory=lambda version: {**summary, "ultralytics": version},
    )


if __name__ == "__main__":
    main()
