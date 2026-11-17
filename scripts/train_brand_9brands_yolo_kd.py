"""Run a guarded nine-brand YOLOv8 teacher/student detection experiment.

This applies class-logit and DFL knowledge distillation while preserving the
reviewed nine-brand ``detect`` contract.  A non-KD student is trained with the
same settings so any KD gain can be measured instead of assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_brand_9brands_yolo import TARGET_BRANDS, validate_dataset
from scripts.train_brand_9brands_yolo import load_config as load_base_config
from scripts.train_brand_9brands_yolo import resolve_path
from src.training.yolo_detection_kd import (
    DetectionKDConfig,
    KDDetectionTrainer,
    make_portable_detection_checkpoint,
)
from src.training.yolo_dataset_policy import assess_nine_brand_training_policy
from src.utils.io import ensure_dir, save_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=("preflight", "teacher", "baseline", "student-kd", "resume-kd", "all"),
    )
    parser.add_argument("--config", default="configs/brand_9brands_yolo_kd.yaml")
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="Existing nine-brand detect checkpoint for the student-kd command.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="KD last.pt checkpoint. Used by command=resume-kd.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="After command=all, explicitly export the KD student to TFLite.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=("fp32", "fp16", "int8"),
        choices=("fp32", "fp16", "int8"),
    )
    return parser.parse_args(argv)


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KD experiment config was not found: {path}")
    experiment = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required_sections = ("teacher", "student", "kd", "evaluation", "export", "compatibility")
    missing = [section for section in required_sections if not isinstance(experiment.get(section), dict)]
    if missing:
        raise ValueError(f"KD experiment config is missing sections: {missing}")
    base_path = resolve_path(experiment.get("base_config", "configs/brand_9brands_yolo.yaml"))
    base = load_base_config(base_path)
    expected_version = str(experiment["compatibility"].get("verified_ultralytics"))
    base_version = str(base.get("compatibility", {}).get("verified_ultralytics"))
    if expected_version != base_version:
        raise ValueError(
            f"KD config expects Ultralytics {expected_version}, base config expects {base_version}"
        )
    data_yaml = resolve_path(base["data"])
    checks = validate_dataset(data_yaml.parent)
    expected_mapping = {str(index): name for index, name in enumerate(TARGET_BRANDS)}
    if checks.get("mapping_verified") != expected_mapping:
        raise ValueError("Prepared nine-brand class mapping differs from the reviewed contract")
    if checks.get("cross_split_family_overlap") != 0:
        raise ValueError("Prepared nine-brand dataset contains cross-split family leakage")
    if checks.get("test_split_evaluated") is not False:
        raise ValueError("Held-out test must remain untouched during teacher/student selection")
    experiment["_dataset_checks"] = checks
    experiment["_dataset_policy"] = assess_nine_brand_training_policy(
        checks,
        allow_quarantined_exclusions=bool(
            base.get("dataset_policy", {}).get("allow_quarantined_exclusions", False)
        ),
    )
    return experiment, base


def kd_config_from_dict(value: dict[str, Any]) -> DetectionKDConfig:
    config = DetectionKDConfig(
        class_weight=float(value.get("class_weight", 1.0)),
        dfl_weight=float(value.get("dfl_weight", 1.0)),
        temperature=float(value.get("temperature", 2.0)),
        topk=int(value.get("topk", 64)),
        min_confidence=float(value.get("min_confidence", 0.05)),
    )
    config.validate()
    return config


def _expected_names() -> dict[int, str]:
    return {index: name for index, name in enumerate(TARGET_BRANDS)}


def require_training_ready(experiment: dict[str, Any]) -> None:
    checks = experiment.get("_dataset_checks", {})
    policy = experiment.get("_dataset_policy", {})
    if not policy.get("effective_training_ready", False):
        raise RuntimeError(
            "Nine-brand dataset is blocked by the configured dataset policy: "
            f"{checks.get('unresolved_annotation_hashes')} unresolved annotation hashes and "
            f"{checks.get('quarantined_image_hashes')} quarantined image hashes remain. "
            f"Failed safety requirements: {policy.get('failed_safety_requirements', [])}"
        )


def _ordered_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    raise ValueError("Checkpoint has no usable class mapping")


def validate_teacher_checkpoint(path: str | Path) -> Path:
    from ultralytics import YOLO

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint was not found: {path}")
    model = YOLO(str(path))
    if model.task != "detect":
        raise ValueError(f"Teacher task must be 'detect', found {model.task!r}")
    names = _ordered_names(model.names)
    if names != _expected_names():
        raise ValueError(
            f"Teacher class order is incompatible: expected={_expected_names()}, actual={names}"
        )
    return path


def validate_kd_resume_checkpoint(
    path: str | Path,
    *,
    expected_run_dir: Path,
    expected_data: Path,
) -> tuple[Path, int]:
    """Validate a trusted, resumable KD ``last.pt`` checkpoint."""

    path = Path(path).resolve()
    expected_run_dir = expected_run_dir.resolve()
    expected_data = expected_data.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KD resume checkpoint was not found: {path}")
    if path.parent.parent != expected_run_dir:
        raise ValueError(
            f"Resume checkpoint belongs to {path.parent.parent}, expected {expected_run_dir}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed_epoch = int(checkpoint.get("epoch", -1)) + 1
    if completed_epoch <= 0:
        raise ValueError("KD checkpoint has no completed epoch to resume")
    if checkpoint.get("optimizer") is None:
        raise ValueError("KD checkpoint optimizer was stripped; use an interrupted run's last.pt")
    train_args = checkpoint.get("train_args", {})
    checkpoint_data = Path(str(train_args.get("data", ""))).resolve()
    if checkpoint_data != expected_data:
        raise ValueError(
            f"Resume checkpoint dataset differs: checkpoint={checkpoint_data}, expected={expected_data}"
        )
    model = checkpoint.get("ema") or checkpoint.get("model")
    names = _ordered_names(getattr(model, "names", None))
    if names != _expected_names():
        raise ValueError(
            f"Resume checkpoint class order is incompatible: expected={_expected_names()}, actual={names}"
        )
    return path, completed_epoch


def build_train_args(
    base: dict[str, Any],
    experiment: dict[str, Any],
    role: str,
    name: str,
) -> dict[str, Any]:
    if role not in {"teacher", "student"}:
        raise ValueError(f"Unknown training role: {role}")
    role_config = experiment[role]
    args = dict(base["train"])
    args.update(
        {
            "data": str(resolve_path(base["data"])),
            "project": str(resolve_path(experiment["project"])),
            "name": name,
            "epochs": int(role_config["epochs"]),
            "batch": int(role_config["batch"]),
            "patience": int(role_config["patience"]),
            "exist_ok": False,
            "resume": False,
        }
    )
    return args


def _assert_new_run(project: Path, name: str) -> Path:
    run_dir = project / name
    if run_dir.exists():
        raise FileExistsError(f"Training run already exists; refusing to overwrite: {run_dir}")
    return run_dir


def _check_ultralytics_version(experiment: dict[str, Any]) -> str:
    from ultralytics import __version__

    expected = str(experiment["compatibility"]["verified_ultralytics"])
    if str(__version__) != expected:
        raise RuntimeError(f"Expected reviewed Ultralytics {expected}, found {__version__}")
    return str(__version__)


def preflight_summary(experiment: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    project = resolve_path(experiment["project"])
    teacher_name = str(experiment["teacher"]["name"])
    baseline_name = str(experiment["student"]["baseline_name"])
    kd_name = str(experiment["student"]["kd_name"])
    checks = experiment["_dataset_checks"]
    policy = experiment["_dataset_policy"]
    return {
        "ultralytics": _check_ultralytics_version(experiment),
        "task": "detect",
        "data": str(resolve_path(base["data"])),
        "class_mapping": _expected_names(),
        "test_evaluation": False,
        "dataset_review": {
            "source_training_ready": bool(checks["training_ready"]),
            "effective_training_ready": bool(policy["effective_training_ready"]),
            "mode": policy["mode"],
            "unresolved_annotation_hashes": int(checks["unresolved_annotation_hashes"]),
            "quarantined_image_hashes": int(checks["quarantined_image_hashes"]),
            "quarantined_hashes_in_splits": int(checks["quarantined_hashes_in_splits"]),
            "cross_split_family_overlap": int(checks["cross_split_family_overlap"]),
            "failed_safety_requirements": policy["failed_safety_requirements"],
            "limitation": policy["limitation"],
        },
        "teacher": {
            "initial_weights": str(experiment["teacher"]["model"]),
            "run_dir": str(project / teacher_name),
            "exists": (project / teacher_name).exists(),
        },
        "student_baseline": {
            "initial_weights": str(experiment["student"]["model"]),
            "run_dir": str(project / baseline_name),
            "exists": (project / baseline_name).exists(),
        },
        "student_kd": {
            "initial_weights": str(experiment["student"]["model"]),
            "run_dir": str(project / kd_name),
            "exists": (project / kd_name).exists(),
            "kd": vars(kd_config_from_dict(experiment["kd"])),
        },
    }


def train_teacher(experiment: dict[str, Any], base: dict[str, Any]) -> Path:
    from ultralytics import YOLO

    require_training_ready(experiment)
    project = resolve_path(experiment["project"])
    name = str(experiment["teacher"]["name"])
    _assert_new_run(project, name)
    model = YOLO(str(experiment["teacher"]["model"]))
    model.train(**build_train_args(base, experiment, "teacher", name))
    return validate_teacher_checkpoint(Path(model.trainer.best))


def train_student_baseline(experiment: dict[str, Any], base: dict[str, Any]) -> Path:
    from ultralytics import YOLO

    require_training_ready(experiment)
    project = resolve_path(experiment["project"])
    name = str(experiment["student"]["baseline_name"])
    _assert_new_run(project, name)
    model = YOLO(str(experiment["student"]["model"]))
    model.train(**build_train_args(base, experiment, "student", name))
    return Path(model.trainer.best).resolve()


def train_student_kd(
    experiment: dict[str, Any],
    base: dict[str, Any],
    teacher_path: str | Path,
    resume_path: str | Path | None = None,
) -> Path:
    require_training_ready(experiment)
    teacher_path = validate_teacher_checkpoint(teacher_path)
    project = resolve_path(experiment["project"])
    name = str(experiment["student"]["kd_name"])
    run_dir = project / name
    overrides = build_train_args(base, experiment, "student", name)
    if resume_path is None:
        _assert_new_run(project, name)
    else:
        resume_path, completed_epoch = validate_kd_resume_checkpoint(
            resume_path,
            expected_run_dir=run_dir,
            expected_data=resolve_path(base["data"]),
        )
        print(
            f"Resuming KD student after completed epoch {completed_epoch}; "
            f"partially processed next-epoch batches will be repeated.",
            flush=True,
        )
        overrides["resume"] = str(resume_path)
    overrides["model"] = str(experiment["student"]["model"])
    trainer = KDDetectionTrainer(
        overrides=overrides,
        teacher_path=teacher_path,
        kd_config=kd_config_from_dict(experiment["kd"]),
    )
    trainer.train()
    best = Path(trainer.best).resolve()
    last = Path(trainer.last).resolve()
    portable_best = make_portable_detection_checkpoint(
        best, run_dir / "weights" / "best_portable.pt"
    )
    if last.is_file():
        make_portable_detection_checkpoint(last, run_dir / "weights" / "last_portable.pt")
    return portable_best


def evaluate_checkpoint(
    checkpoint: str | Path,
    experiment: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).resolve()
    evaluation = experiment["evaluation"]
    result = YOLO(str(checkpoint)).val(
        data=str(resolve_path(base["data"])),
        split=str(evaluation.get("split", "val")),
        imgsz=int(base["train"]["imgsz"]),
        batch=int(evaluation.get("batch", base["train"]["batch"])),
        device=base["train"]["device"],
        plots=False,
        verbose=False,
    )
    return {
        "checkpoint": str(checkpoint),
        "size_mb": checkpoint.stat().st_size / 1_000_000,
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class_map50_95": {
            _expected_names()[index]: float(value)
            for index, value in enumerate(result.box.maps.tolist())
        },
    }


def write_comparison_report(
    experiment: dict[str, Any],
    base: dict[str, Any],
    teacher: Path,
    baseline: Path,
    student_kd: Path,
) -> Path:
    output = resolve_path(experiment["project"]) / "comparison_report.json"
    if output.exists():
        raise FileExistsError(f"Comparison report already exists; refusing to overwrite: {output}")
    report = {
        "protocol": {
            "task": "detect",
            "selection_split": "validation",
            "held_out_test_used": False,
            "student_architecture": str(experiment["student"]["model"]),
            "kd": vars(kd_config_from_dict(experiment["kd"])),
            "dataset_policy": experiment["_dataset_policy"],
        },
        "teacher": evaluate_checkpoint(teacher, experiment, base),
        "student_baseline": evaluate_checkpoint(baseline, experiment, base),
        "student_kd": evaluate_checkpoint(student_kd, experiment, base),
    }
    report["kd_minus_baseline"] = {
        metric: report["student_kd"][metric] - report["student_baseline"][metric]
        for metric in ("precision", "recall", "map50", "map50_95")
    }
    ensure_dir(output.parent)
    save_json(report, output)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment, base = load_experiment_config(args.config)
    summary = preflight_summary(experiment, base)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.command == "preflight":
        return

    if args.command == "teacher":
        print(f"Teacher checkpoint: {train_teacher(experiment, base)}")
        return
    if args.command == "baseline":
        print(f"Baseline checkpoint: {train_student_baseline(experiment, base)}")
        return
    if args.command == "student-kd":
        if args.teacher is None:
            raise SystemExit("student-kd requires --teacher path/to/best.pt")
        print(f"KD student checkpoint: {train_student_kd(experiment, base, args.teacher)}")
        return
    if args.command == "resume-kd":
        if args.resume is None:
            raise SystemExit("resume-kd requires --resume path/to/last.pt")
        project = resolve_path(experiment["project"])
        teacher = (
            args.teacher.resolve()
            if args.teacher is not None
            else project / str(experiment["teacher"]["name"]) / "weights" / "best.pt"
        )
        baseline = (
            project
            / str(experiment["student"]["baseline_name"])
            / "weights"
            / "best.pt"
        )
        if not baseline.is_file():
            raise FileNotFoundError(f"Completed student baseline checkpoint was not found: {baseline}")
        student_kd = train_student_kd(
            experiment,
            base,
            teacher,
            resume_path=args.resume,
        )
        report = write_comparison_report(experiment, base, teacher, baseline, student_kd)
        print(f"KD student checkpoint: {student_kd}", flush=True)
        print(f"Comparison report: {report}", flush=True)
        return

    project = resolve_path(experiment["project"])
    for name in (
        experiment["teacher"]["name"],
        experiment["student"]["baseline_name"],
        experiment["student"]["kd_name"],
    ):
        _assert_new_run(project, str(name))
    teacher = train_teacher(experiment, base)
    baseline = train_student_baseline(experiment, base)
    student_kd = train_student_kd(experiment, base, teacher)
    report = write_comparison_report(experiment, base, teacher, baseline, student_kd)
    print(f"Comparison report: {report}", flush=True)

    if args.export:
        from scripts.export_model1_tflite import export_tflite_variants

        export_report = export_tflite_variants(
            model_path=student_kd,
            data_path=resolve_path(base["data"]),
            output_dir=resolve_path(experiment["export"]["output_dir"]),
            image_size=int(experiment["export"]["image_size"]),
            precisions=tuple(args.precisions),
            calibration_fraction=float(experiment["export"]["calibration_fraction"]),
            allow_quarantined_exclusions=bool(
                base.get("dataset_policy", {}).get("allow_quarantined_exclusions", False)
            ),
        )
        print(f"Export report: {export_report}", flush=True)


if __name__ == "__main__":
    main()
