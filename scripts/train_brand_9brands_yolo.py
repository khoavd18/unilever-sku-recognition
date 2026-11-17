"""Manually preflight or train the guarded nine-brand YOLOv8n baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_brand_9brands_yolo import TARGET_BRANDS, validate_dataset
from src.training import yolo as yolo_training
from src.training.yolo_dataset_policy import assess_nine_brand_training_policy


def resolve_path(value: str | Path) -> Path:
    return yolo_training.resolve_path(value, ROOT)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brand_9brands_yolo.yaml")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and print the resolved run without starting training.",
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
        "deterministic": True,
        "patience": 0,
        "single_cls": False,
        "pretrained": True,
        "resume": False,
        "exist_ok": False,
    }
    return yolo_training.load_reviewed_yolo_config(
        path,
        required_settings=required,
        mismatch_message="Nine-brand baseline differs from reviewed settings",
        heldout_message="Config must explicitly keep test evaluation disabled",
    )


def preflight(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    data_yaml = resolve_path(config["data"])
    checks = validate_dataset(data_yaml.parent)
    expected = {str(index): brand for index, brand in enumerate(TARGET_BRANDS)}
    if checks["mapping_verified"] != expected:
        raise ValueError("Prepared dataset class order is incompatible")
    policy = assess_nine_brand_training_policy(
        checks,
        allow_quarantined_exclusions=bool(
            config.get("dataset_policy", {}).get("allow_quarantined_exclusions", False)
        ),
    )
    summary = {
        "python": sys.executable,
        "config": str(config_path),
        "model": str(resolve_path(config["model"])),
        "data": str(data_yaml),
        "mapping": expected,
        "run_dir": str(resolve_path(config["project"]) / str(config["name"])),
        "epochs": config["train"]["epochs"],
        "training_ready": policy["effective_training_ready"],
        "dataset_policy": policy,
        "unresolved_annotation_hashes": checks["unresolved_annotation_hashes"],
        "quarantined_image_hashes": checks["quarantined_image_hashes"],
        "test_evaluation": False,
        "training_started": False,
    }
    if not policy["effective_training_ready"]:
        raise RuntimeError(
            "Nine-brand dataset is not approved by the configured policy: "
            f"{checks['unresolved_annotation_hashes']} unresolved annotation hashes and "
            f"{checks['quarantined_image_hashes']} quarantined hashes remain. "
            f"Failed safety requirements: {policy['failed_safety_requirements']}"
        )
    yolo_training.ensure_new_run(Path(summary["run_dir"]))
    return config, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, summary = preflight(resolve_path(args.config))
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    expected_version = str(config.get("compatibility", {}).get("verified_ultralytics"))
    config_path = resolve_path(args.config)
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
