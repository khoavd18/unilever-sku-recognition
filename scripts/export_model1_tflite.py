"""Export a reviewed nine-brand detector to TFLite with Ultralytics 8.3.155.

The exporter uses the supported ``format='tflite'``, ``half`` and ``int8``
arguments.  Every precision is exported from an isolated staging directory so
Ultralytics cannot silently reuse or overwrite another variant's SavedModel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_brand_9brands_yolo import TARGET_BRANDS, validate_dataset
from src.utils.io import ensure_dir, save_json
from src.training.yolo_dataset_policy import assess_nine_brand_training_policy


VERIFIED_ULTRALYTICS = "8.3.155"
SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/brand_9brands_yolo/data.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/model1_brand9/android"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration-fraction", type=float, default=1.0)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=SUPPORTED_PRECISIONS,
        default=SUPPORTED_PRECISIONS,
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--allow-quarantined-exclusions",
        action="store_true",
        help="Allow calibration/validation on the structurally clean subset only.",
    )
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_tflite_export_args(
    precision: str,
    *,
    image_size: int,
    data_path: str | Path,
    calibration_fraction: float,
) -> dict[str, Any]:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported TFLite precision {precision!r}; choose from {SUPPORTED_PRECISIONS}"
        )
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0.0 < calibration_fraction <= 1.0:
        raise ValueError("calibration_fraction must be in (0, 1]")
    args: dict[str, Any] = {
        "format": "tflite",
        "imgsz": image_size,
        "nms": False,
        "half": precision == "fp16",
        "int8": precision == "int8",
    }
    if precision == "int8":
        args.update(data=str(Path(data_path).resolve()), fraction=calibration_fraction)
    return args


def _expected_names() -> dict[int, str]:
    return {index: name for index, name in enumerate(TARGET_BRANDS)}


def validate_export_inputs(
    model_path: str | Path,
    data_path: str | Path,
    *,
    image_size: int,
    precisions: Sequence[str],
    calibration_fraction: float,
    allow_quarantined_exclusions: bool = False,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    from ultralytics import YOLO, __version__

    if str(__version__) != VERIFIED_ULTRALYTICS:
        raise RuntimeError(
            f"Expected reviewed Ultralytics {VERIFIED_ULTRALYTICS}, found {__version__}"
        )
    model_path = _resolve(model_path)
    data_path = _resolve(data_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_path}")
    checks = validate_dataset(data_path.parent)
    policy = assess_nine_brand_training_policy(
        checks,
        allow_quarantined_exclusions=allow_quarantined_exclusions,
    )
    if not policy["effective_training_ready"]:
        raise RuntimeError(
            "Nine-brand export data is blocked by the configured policy; failed safety "
            f"requirements: {policy['failed_safety_requirements']}"
        )
    if len(set(precisions)) != len(precisions):
        raise ValueError("Each export precision may be requested only once")
    export_args = {
        precision: build_tflite_export_args(
            precision,
            image_size=image_size,
            data_path=data_path,
            calibration_fraction=calibration_fraction,
        )
        for precision in precisions
    }
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"Model 1 export requires task='detect', found {model.task!r}")
    names = {int(index): str(name) for index, name in model.names.items()}
    if names != _expected_names():
        raise ValueError(
            f"Model 1 class mapping differs from the reviewed order: {names}"
        )
    return model_path, data_path, export_args, policy


def _find_exported_tflite(exported: Any, staging_dir: Path) -> Path:
    candidates: list[Path] = []
    if exported is not None:
        exported_path = Path(str(exported))
        if exported_path.is_file() and exported_path.suffix == ".tflite":
            candidates.append(exported_path)
        elif exported_path.is_dir():
            candidates.extend(exported_path.rglob("*.tflite"))
    candidates.extend(staging_dir.rglob("*.tflite"))
    unique = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if not unique:
        raise FileNotFoundError("Ultralytics reported export success but produced no .tflite file")
    return max(unique, key=lambda candidate: candidate.stat().st_mtime_ns)


def _evaluate_tflite(model_path: Path, data_path: Path, image_size: int) -> dict[str, float]:
    from ultralytics import YOLO

    result = YOLO(str(model_path), task="detect").val(
        data=str(data_path),
        split="val",
        imgsz=image_size,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
    )
    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }


def export_tflite_variants(
    *,
    model_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    image_size: int = 640,
    precisions: Sequence[str] = SUPPORTED_PRECISIONS,
    calibration_fraction: float = 1.0,
    evaluate: bool = True,
    allow_quarantined_exclusions: bool = False,
) -> Path:
    from ultralytics import YOLO

    model_path, data_path, all_export_args, dataset_policy = validate_export_inputs(
        model_path,
        data_path,
        image_size=image_size,
        precisions=precisions,
        calibration_fraction=calibration_fraction,
        allow_quarantined_exclusions=allow_quarantined_exclusions,
    )
    output_dir = ensure_dir(_resolve(output_dir))
    report_path = output_dir / "report.json"
    targets = {
        precision: output_dir / f"model1_9brands_{image_size}_{precision}.tflite"
        for precision in precisions
    }
    collisions = [str(path) for path in (*targets.values(), report_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"Export refuses to overwrite existing artifacts: {collisions}")

    report: dict[str, Any] = {
        "source_model": str(model_path),
        "data": str(data_path),
        "task": "detect",
        "class_mapping": _expected_names(),
        "image_size": image_size,
        "ultralytics": VERIFIED_ULTRALYTICS,
        "dataset_policy": dataset_policy,
        "artifacts": [],
    }
    save_json(report, report_path)
    for precision in precisions:
        try:
            with tempfile.TemporaryDirectory(prefix=f"model1_{precision}_", dir=output_dir) as temporary:
                staging_dir = Path(temporary)
                staged_checkpoint = staging_dir / f"model1_{precision}.pt"
                shutil.copy2(model_path, staged_checkpoint)
                exported = YOLO(str(staged_checkpoint)).export(**all_export_args[precision])
                exported_tflite = _find_exported_tflite(exported, staging_dir)
                shutil.copy2(exported_tflite, targets[precision])
            artifact: dict[str, Any] = {
                "precision": precision,
                "file": str(targets[precision]),
                "size_mb": targets[precision].stat().st_size / 1_000_000,
                "export_args": all_export_args[precision],
            }
            if evaluate:
                try:
                    artifact["validation"] = _evaluate_tflite(
                        targets[precision], data_path, image_size
                    )
                except Exception as exc:  # export remains valid; evaluation failure is explicit
                    artifact["validation_error"] = f"{type(exc).__name__}: {exc}"
            report["artifacts"].append(artifact)
            save_json(report, report_path)
        except Exception as exc:
            report["failed_precision"] = precision
            report["export_error"] = f"{type(exc).__name__}: {exc}"
            save_json(report, report_path)
            raise RuntimeError(
                f"TFLite export failed for {precision}; partial report: {report_path}"
            ) from exc
    report["status"] = "complete"
    save_json(report, report_path)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_path, data_path, export_args, dataset_policy = validate_export_inputs(
        args.model,
        args.data,
        image_size=args.imgsz,
        precisions=args.precisions,
        calibration_fraction=args.calibration_fraction,
        allow_quarantined_exclusions=args.allow_quarantined_exclusions,
    )
    plan = {
        "model": str(model_path),
        "data": str(data_path),
        "output_dir": str(_resolve(args.output_dir)),
        "evaluate": not args.skip_evaluation,
        "dataset_policy": dataset_policy,
        "variants": export_args,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    report = export_tflite_variants(
        model_path=model_path,
        data_path=data_path,
        output_dir=args.output_dir,
        image_size=args.imgsz,
        precisions=args.precisions,
        calibration_fraction=args.calibration_fraction,
        evaluate=not args.skip_evaluation,
        allow_quarantined_exclusions=args.allow_quarantined_exclusions,
    )
    print(f"Export report: {report}", flush=True)


if __name__ == "__main__":
    main()
