from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from scripts.export_model1_tflite import _expected_names as export_expected_names
from scripts.export_model1_tflite import build_tflite_export_args
from scripts.prepare_brand_9brands_yolo import TARGET_BRANDS
from scripts.train_brand_9brands_yolo_kd import (
    _expected_names as training_expected_names,
)
from scripts.train_brand_9brands_yolo_kd import (
    build_train_args,
    kd_config_from_dict,
    require_training_ready,
    validate_kd_resume_checkpoint,
)
from src.training.yolo_detection_kd import (
    DetectionKDConfig,
    KDDetectionModel,
    combine_detection_and_kd_loss,
    detection_kd_loss,
    make_portable_detection_checkpoint,
    parse_detection_predictions,
)
from src.training.yolo_dataset_policy import assess_nine_brand_training_policy


class DetectionKnowledgeDistillationTests(unittest.TestCase):
    def _safe_nine_brand_checks(self) -> dict:
        return {
            "status": "structural_checks_passed",
            "training_ready": False,
            "ordinary_yolo_detection_rows_only": True,
            "all_nine_brands_present_in_every_split": True,
            "exact_image_cross_split_overlap": {
                "train-val": 0,
                "train-test": 0,
                "val-test": 0,
            },
            "cross_split_family_overlap": 0,
            "known_holdout_violations": 0,
            "quarantined_hashes_in_splits": 0,
            "test_split_evaluated": False,
            "unresolved_annotation_hashes": 835,
            "quarantined_image_hashes": 929,
        }

    def _predictions(self, *, requires_grad: bool) -> list[torch.Tensor]:
        channels = 4 * 16 + 9
        return [
            torch.randn(2, channels, 4, 4, requires_grad=requires_grad),
            torch.randn(2, channels, 2, 2, requires_grad=requires_grad),
        ]

    def test_raw_detection_predictions_and_kd_backward(self) -> None:
        student = self._predictions(requires_grad=True)
        teacher = self._predictions(requires_grad=False)
        box, classes = parse_detection_predictions(student, num_classes=9, reg_max=16)
        self.assertEqual(tuple(box.shape), (2, 64, 20))
        self.assertEqual(tuple(classes.shape), (2, 9, 20))

        class_loss, dfl_loss, total = detection_kd_loss(
            student,
            teacher,
            num_classes=9,
            reg_max=16,
            config=DetectionKDConfig(topk=8, min_confidence=0.0),
        )
        self.assertTrue(torch.isfinite(torch.stack((class_loss, dfl_loss, total))).all())
        total.backward()
        self.assertTrue(all(feature.grad is not None for feature in student))
        self.assertTrue(all(feature.grad is None for feature in teacher))

    def test_incompatible_teacher_grid_fails_explicitly(self) -> None:
        student = self._predictions(requires_grad=True)
        channels = 4 * 16 + 9
        teacher = [torch.randn(2, channels, 3, 3)]
        with self.assertRaisesRegex(ValueError, "prediction shapes differ"):
            detection_kd_loss(
                student,
                teacher,
                num_classes=9,
                reg_max=16,
                config=DetectionKDConfig(min_confidence=0.0),
            )

    def test_kd_scalar_is_added_once_not_broadcast_per_component(self) -> None:
        base = torch.tensor([1.0, 2.0, 3.0])
        combined = combine_detection_and_kd_loss(base, torch.tensor(4.0), batch_size=2)
        self.assertEqual(float(combined), 14.0)

    def test_invalid_kd_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            kd_config_from_dict({"class_weight": 0, "dfl_weight": 0})
        with self.assertRaisesRegex(ValueError, "temperature"):
            DetectionKDConfig(temperature=0).validate()

    def test_training_and_export_contracts_use_all_nine_brands(self) -> None:
        expected = {index: brand for index, brand in enumerate(TARGET_BRANDS)}
        self.assertEqual(len(expected), 9)
        self.assertEqual(training_expected_names(), expected)
        self.assertEqual(export_expected_names(), expected)
        self.assertEqual(expected[8], "TRESEMME")

    def test_incomplete_nine_brand_review_blocks_training(self) -> None:
        checks = self._safe_nine_brand_checks()
        policy = assess_nine_brand_training_policy(
            checks, allow_quarantined_exclusions=False
        )
        experiment = {
            "_dataset_checks": checks,
            "_dataset_policy": policy,
        }
        with self.assertRaisesRegex(RuntimeError, "835 unresolved.*929 quarantined"):
            require_training_ready(experiment)

    def test_clean_subset_policy_allows_only_fully_excluded_quarantine(self) -> None:
        checks = self._safe_nine_brand_checks()
        policy = assess_nine_brand_training_policy(
            checks, allow_quarantined_exclusions=True
        )
        self.assertTrue(policy["effective_training_ready"])
        self.assertEqual(policy["mode"], "clean_subset_excluding_quarantine")
        require_training_ready({"_dataset_checks": checks, "_dataset_policy": policy})

        checks["quarantined_hashes_in_splits"] = 1
        unsafe = assess_nine_brand_training_policy(
            checks, allow_quarantined_exclusions=True
        )
        self.assertFalse(unsafe["effective_training_ready"])
        self.assertIn("quarantined_hashes_in_splits_zero", unsafe["failed_safety_requirements"])

    def test_student_baseline_and_kd_use_same_training_settings(self) -> None:
        base = {
            "data": "data/example.yaml",
            "train": {
                "epochs": 100,
                "batch": 8,
                "patience": 0,
                "exist_ok": False,
                "resume": False,
                "imgsz": 640,
            },
        }
        experiment = {
            "project": "runs/kd",
            "student": {"epochs": 25, "batch": 4, "patience": 3},
        }
        args = build_train_args(base, experiment, "student", "student")
        self.assertEqual(args["epochs"], 25)
        self.assertEqual(args["batch"], 4)
        self.assertEqual(args["patience"], 3)
        self.assertFalse(args["exist_ok"])
        self.assertFalse(args["resume"])

    def test_tflite_arguments_match_pinned_ultralytics_api(self) -> None:
        fp32 = build_tflite_export_args(
            "fp32", image_size=640, data_path="data.yaml", calibration_fraction=1.0
        )
        fp16 = build_tflite_export_args(
            "fp16", image_size=640, data_path="data.yaml", calibration_fraction=1.0
        )
        int8 = build_tflite_export_args(
            "int8", image_size=640, data_path="data.yaml", calibration_fraction=0.5
        )
        self.assertEqual(fp32["format"], "tflite")
        self.assertFalse(fp32["half"])
        self.assertFalse(fp32["int8"])
        self.assertTrue(fp16["half"])
        self.assertTrue(int8["int8"])
        self.assertEqual(int8["fraction"], 0.5)
        for args in (fp32, fp16, int8):
            get_cfg(overrides=args)  # raises if an argument is unsupported by 8.3.155
            self.assertNotIn("quantize", args)

    def test_kd_resume_requires_optimizer_and_matching_nine_brand_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "student_kd"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            data = root / "data.yaml"
            data.write_text("names: {}\n", encoding="utf-8")
            checkpoint = weights_dir / "last.pt"
            model = SimpleNamespace(names=training_expected_names())
            payload = {
                "epoch": 0,
                "optimizer": {"state": {}, "param_groups": []},
                "train_args": {"data": str(data)},
                "ema": model,
                "model": None,
            }
            torch.save(payload, checkpoint)
            validated, completed = validate_kd_resume_checkpoint(
                checkpoint,
                expected_run_dir=run_dir,
                expected_data=data,
            )
            self.assertEqual(validated, checkpoint.resolve())
            self.assertEqual(completed, 1)

            payload["optimizer"] = None
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "optimizer was stripped"):
                validate_kd_resume_checkpoint(
                    checkpoint,
                    expected_run_dir=run_dir,
                    expected_data=data,
                )

    def test_kd_checkpoint_can_be_copied_as_standard_detection_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "best.pt"
            destination = root / "best_portable.pt"
            model = KDDetectionModel("yolov8n.yaml", nc=9, verbose=False)
            torch.save({"model": model, "ema": None}, source)
            make_portable_detection_checkpoint(source, destination)
            checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertIs(type(checkpoint["model"]), DetectionModel)


if __name__ == "__main__":
    unittest.main()
