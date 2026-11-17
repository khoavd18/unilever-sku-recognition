from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import train_axe_yolo
from scripts import train_brand_5brands_yolo
from scripts import train_brand_9brands_yolo
from scripts import train_detector
from src.training import yolo as yolo_training


def _generic_fixture(root: Path) -> Path:
    dataset_dir = root / "data" / "synthetic_detector"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "SYNTH_ALPHA", 1: "SYNTH_BETA"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset_dir / "dataset_checks.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "ordinary_yolo_detection_rows_only": True,
                "mapping_verified": {"0": "SYNTH_ALPHA", "1": "SYNTH_BETA"},
                "exact_image_cross_split_overlap": {
                    "train_validation": 0,
                    "train_test": 0,
                    "validation_test": 0,
                },
                "cross_split_family_overlap": 0,
                "test_split_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "configs" / "synthetic_detector.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "yolov8n.pt",
                "data": "data/synthetic_detector/data.yaml",
                "project": "runs/synthetic_detector",
                "name": "baseline_s42",
                "train": {
                    "epochs": 3,
                    "imgsz": 320,
                    "batch": 2,
                    "workers": 0,
                    "device": "cpu",
                    "seed": 42,
                    "val": True,
                    "resume": False,
                    "exist_ok": False,
                },
                "artifacts": {"evaluate_test_split": False},
                "compatibility": {
                    "verified_ultralytics": "8.3.155",
                    "install_or_upgrade_automatically": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _legacy_config(required: dict[str, object]) -> dict[str, object]:
    train = {key: value for key, value in required.items() if key != "model"}
    return {
        "model": required["model"],
        "data": "data/synthetic/data.yaml",
        "project": "runs/synthetic",
        "name": "baseline",
        "train": train,
        "artifacts": {"evaluate_test_split": False},
        "compatibility": {"verified_ultralytics": "8.3.155"},
    }


def _source_order_profile_fixture(root: Path) -> tuple[Path, Path]:
    config_path = _generic_fixture(root)
    dataset_dir = root / "data" / "synthetic_detector"
    source_order = (
        "VIM",
        "AXE",
        "LIFEBUOY",
        "HAZELINE",
        "REXONA",
        "CLEAR",
        "DOVE",
        "SUNSILK",
        "TRESEMME",
    )
    stable_ids = {
        "AXE": 0,
        "VIM": 1,
        "LIFEBUOY": 2,
        "HAZELINE": 3,
        "REXONA": 4,
        "CLEAR": 5,
        "DOVE": 6,
        "SUNSILK": 7,
        "TRESEMME": 8,
    }
    data = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    data["names"] = {index: name for index, name in enumerate(source_order)}
    (dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    checks = json.loads((dataset_dir / "dataset_checks.json").read_text(encoding="utf-8"))
    checks["mapping_verified"] = {
        str(index): name for index, name in enumerate(source_order)
    }
    checks["class_order"] = "source_data_yaml"
    checks["stable_registry_ids"] = {
        str(index): stable_ids[name] for index, name in enumerate(source_order)
    }
    (dataset_dir / "dataset_checks.json").write_text(
        json.dumps(checks), encoding="utf-8"
    )
    mapping_path = dataset_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "target_profile": "brand_9brands",
                "class_order": "source_data_yaml",
                "target_names": {
                    str(index): name for index, name in enumerate(source_order)
                },
                "output_names": {
                    str(index): name for index, name in enumerate(source_order)
                },
                "source_to_target": [
                    {
                        "source_id": index,
                        "source_name": name,
                        "target_id": index,
                        "target_name": name,
                        "registry_id": stable_ids[name],
                    }
                    for index, name in enumerate(source_order)
                ],
            }
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["brand_profile"] = "brand_9brands"
    config["class_order"] = "source_data_yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, mapping_path


class SharedYoloTrainingWorkflowTests(unittest.TestCase):
    def test_generic_preflight_supports_synthetic_class_mapping_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = _generic_fixture(root)
            with patch.object(
                yolo_training,
                "_load_ultralytics_backend",
                side_effect=AssertionError("preflight must not load Ultralytics"),
            ):
                plan, summary = train_detector.preflight(config_path, root=root)

        self.assertEqual(
            summary["mapping"], {"0": "SYNTH_ALPHA", "1": "SYNTH_BETA"}
        )
        self.assertEqual(summary["dataset_check_status"], "passed")
        self.assertFalse(summary["test_evaluation"])
        self.assertFalse(summary["model_inference"])
        self.assertFalse(summary["training_started"])
        self.assertTrue(summary["training_arguments_supported"])
        self.assertEqual(summary["ultralytics"]["installed"], "8.3.155")
        self.assertEqual(summary["device"]["selected"], "cpu")
        self.assertFalse(summary["model_constructed"])
        self.assertEqual(plan.train_args["epochs"], 3)
        self.assertEqual(plan.train_args["data"], str(plan.data_yaml))
        self.assertEqual(plan.train_args["project"], str(plan.project_dir))
        self.assertEqual(plan.train_args["name"], "baseline_s42")

    def test_generic_preflight_rejects_mapping_drift_and_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = _generic_fixture(root)
            checks_path = root / "data" / "synthetic_detector" / "dataset_checks.json"
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
            checks["mapping_verified"]["1"] = "WRONG"
            checks_path.write_text(json.dumps(checks), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class order is incompatible"):
                train_detector.preflight(config_path, root=root)

            checks["mapping_verified"]["1"] = "SYNTH_BETA"
            checks_path.write_text(json.dumps(checks), encoding="utf-8")
            (root / "runs" / "synthetic_detector" / "baseline_s42").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                train_detector.preflight(config_path, root=root)

    def test_brand_profile_requires_exact_registry_class_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = _generic_fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["brand_profile"] = "brand_9brands"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "one-to-one permutation|does not match brand profile"
            ):
                yolo_training.build_generic_preflight(config_path, root=root)

    def test_source_yaml_order_preserves_separate_stable_registry_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, _mapping_path = _source_order_profile_fixture(root)
            _plan, summary = yolo_training.build_generic_preflight(
                config_path, root=root
            )

        self.assertEqual(summary["class_order"], "source_data_yaml")
        self.assertTrue(summary["source_to_training_mapping_verified"])
        first, second = summary["registry_stable_mapping"][:2]
        self.assertEqual(
            (first["training_id"], first["training_name"], first["registry_id"]),
            (0, "VIM", 1),
        )
        self.assertEqual(
            (second["training_id"], second["training_name"], second["registry_id"]),
            (1, "AXE", 0),
        )

    def test_source_yaml_order_rejects_mapping_training_or_registry_id_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, mapping_path = _source_order_profile_fixture(root)
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping["source_to_target"][0]["target_id"] = 1
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_id to equal"):
                yolo_training.build_generic_preflight(config_path, root=root)

            mapping["source_to_target"][0]["target_id"] = 0
            mapping["source_to_target"][0]["registry_id"] = 999
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registry_id differs"):
                yolo_training.build_generic_preflight(config_path, root=root)

            mapping["source_to_target"][0]["registry_id"] = 1
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            checks_path = mapping_path.parent / "dataset_checks.json"
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
            checks["stable_registry_ids"]["0"] = 999
            checks_path.write_text(json.dumps(checks), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable_registry_ids differ"):
                yolo_training.build_generic_preflight(config_path, root=root)

    def test_default_profile_order_remains_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, _mapping_path = _source_order_profile_fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config.pop("class_order")
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match brand profile"):
                yolo_training.build_generic_preflight(config_path, root=root)

    def test_explicit_optimizer_and_augmentation_reach_train_unchanged(self) -> None:
        requested = {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "hsv_h": 0.01,
            "hsv_s": 0.2,
            "hsv_v": 0.2,
            "translate": 0.05,
            "scale": 0.2,
            "degrees": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "mosaic": 0.3,
            "close_mosaic": 10,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = _generic_fixture(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["train"].update(requested)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            plan, summary = train_detector.preflight(config_path, root=root)
            received: dict[str, object] = {}

            class FakeModel:
                def train(self, **kwargs: object) -> str:
                    received.update(kwargs)
                    return "ok"

            with contextlib.redirect_stdout(io.StringIO()):
                result = yolo_training.execute_yolo_training(
                    plan,
                    expected_version="8.3.155",
                    backend_loader=lambda: (lambda _model: FakeModel(), "8.3.155"),
                )

        self.assertEqual(result, "ok")
        self.assertEqual({key: received[key] for key in requested}, requested)
        self.assertEqual(summary["effective_optimizer"]["optimizer"], "AdamW")
        self.assertEqual(summary["effective_augmentation"]["mosaic"], 0.3)

    def test_injected_backend_receives_one_exact_training_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, _summary = train_detector.preflight(_generic_fixture(root), root=root)
            calls: list[tuple[str, object]] = []

            class FakeModel:
                def train(self, **kwargs: object) -> str:
                    calls.append(("train", kwargs))
                    return "synthetic-result"

            def factory(model_argument: str) -> FakeModel:
                calls.append(("model", model_argument))
                return FakeModel()

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = yolo_training.execute_yolo_training(
                    plan,
                    expected_version="8.3.155",
                    payload_factory=lambda version: {"ultralytics": version},
                    backend_loader=lambda: (factory, "8.3.155"),
                )

        self.assertEqual(result, "synthetic-result")
        self.assertEqual(calls[0], ("model", plan.model_argument))
        self.assertEqual(calls[1], ("train", plan.train_args))
        self.assertEqual(json.loads(output.getvalue()), {"ultralytics": "8.3.155"})

    def test_version_mismatch_fails_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, _summary = train_detector.preflight(_generic_fixture(root), root=root)

            def forbidden_factory(_model: str) -> object:
                raise AssertionError("model must not be constructed after version mismatch")

            with self.assertRaisesRegex(RuntimeError, "found 9.0.0"):
                yolo_training.execute_yolo_training(
                    plan,
                    expected_version="8.3.155",
                    backend_loader=lambda: (forbidden_factory, "9.0.0"),
                )

    def test_executor_rechecks_collision_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, _summary = train_detector.preflight(_generic_fixture(root), root=root)
            plan.run_dir.mkdir(parents=True)

            def forbidden_factory(_model: str) -> object:
                raise AssertionError("model must not be constructed after a run collision")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    yolo_training.execute_yolo_training(
                        plan,
                        expected_version="8.3.155",
                        backend_loader=lambda: (forbidden_factory, "8.3.155"),
                    )

    def test_executor_attaches_capture_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, _summary = train_detector.preflight(_generic_fixture(root), root=root)
            events: list[str] = []

            class Capture:
                def attach(self, _model: object) -> None:
                    events.append("attach")

            class FakeModel:
                def train(self, **_kwargs: object) -> str:
                    events.append("train")
                    return "ok"

            with contextlib.redirect_stdout(io.StringIO()):
                yolo_training.execute_yolo_training(
                    plan,
                    expected_version="8.3.155",
                    backend_loader=lambda: (lambda _model: FakeModel(), "8.3.155"),
                    training_capture=Capture(),
                )

        self.assertEqual(events, ["attach", "train"])

    def test_legacy_config_contracts_are_preserved_by_shared_loader(self) -> None:
        contracts = [
            (
                train_axe_yolo.load_and_validate_config,
                {
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
                },
                "Model 1 baseline config differs",
            ),
            (
                train_brand_5brands_yolo.load_config,
                {
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
                },
                "Five-brand baseline config differs",
            ),
            (
                train_brand_9brands_yolo.load_config,
                {
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
                },
                "Nine-brand baseline differs",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, (loader, required, error) in enumerate(contracts):
                with self.subTest(loader=loader.__module__):
                    config = _legacy_config(required)
                    path = root / f"legacy_{index}.yaml"
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    self.assertEqual(loader(path)["train"]["epochs"], 100)
                    invalid = copy.deepcopy(config)
                    invalid["train"]["epochs"] = 99
                    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        loader(path)

    def test_legacy_cli_defaults_and_callable_symbols_remain_available(self) -> None:
        self.assertEqual(
            train_axe_yolo.parse_args([]).config, "configs/axe_yolo.yaml"
        )
        self.assertFalse(train_axe_yolo.parse_args([]).summary_only)
        self.assertEqual(
            train_brand_5brands_yolo.parse_args([]).config,
            "configs/brand_5brands_yolo.yaml",
        )
        self.assertFalse(train_brand_5brands_yolo.parse_args([]).preflight)
        self.assertEqual(
            train_brand_9brands_yolo.parse_args([]).config,
            "configs/brand_9brands_yolo.yaml",
        )
        self.assertFalse(train_brand_9brands_yolo.parse_args([]).preflight)
        for module, names in (
            (
                train_axe_yolo,
                ("_resolve", "load_and_validate_config", "validate_dataset", "write_validation_summary"),
            ),
            (train_brand_5brands_yolo, ("resolve_path", "load_config", "preflight")),
            (train_brand_9brands_yolo, ("resolve_path", "load_config", "preflight")),
        ):
            for name in names:
                self.assertTrue(callable(getattr(module, name)))

    def test_legacy_no_training_modes_return_before_shared_executor(self) -> None:
        five_summary = {"training_started": False}
        nine_summary = {"training_started": False}
        config = {
            "model": "yolov8n.pt",
            "data": "unused.yaml",
            "project": "unused",
            "name": "unused",
            "train": {"epochs": 1},
            "compatibility": {"verified_ultralytics": "8.3.155"},
        }
        with patch.object(
            yolo_training,
            "execute_yolo_training",
            side_effect=AssertionError("preflight must not execute training"),
        ):
            with patch.object(
                train_brand_5brands_yolo,
                "preflight",
                return_value=(config, five_summary),
            ), contextlib.redirect_stdout(io.StringIO()):
                train_brand_5brands_yolo.main(["--preflight"])
            with patch.object(
                train_brand_9brands_yolo,
                "preflight",
                return_value=(config, nine_summary),
            ), contextlib.redirect_stdout(io.StringIO()):
                train_brand_9brands_yolo.main(["--preflight"])

            axe_config = {**config, "artifacts": {"evaluate_test_split": False}}
            with (
                patch.object(
                    train_axe_yolo,
                    "load_and_validate_config",
                    return_value=axe_config,
                ),
                patch.object(train_axe_yolo, "validate_dataset", return_value={}),
                patch.object(
                    train_axe_yolo,
                    "write_validation_summary",
                    return_value=Path("summary.md"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_axe_yolo.main(["--summary-only"])

    def test_five_brand_training_path_delegates_to_shared_executor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {
                "model": str(root / "yolov8n.pt"),
                "data": str(root / "data.yaml"),
                "project": str(root / "runs"),
                "name": "baseline",
                "train": {"epochs": 1, "batch": 2},
                "compatibility": {"verified_ultralytics": "8.3.155"},
            }
            summary = {"training_started": False}
            with (
                patch.object(
                    train_brand_5brands_yolo,
                    "preflight",
                    return_value=(config, summary),
                ),
                patch.object(
                    yolo_training, "execute_yolo_training", return_value=None
                ) as execute,
            ):
                train_brand_5brands_yolo.main([])

        execute.assert_called_once()
        plan = execute.call_args.args[0]
        self.assertEqual(plan.train_args["epochs"], 1)
        self.assertEqual(plan.train_args["batch"], 2)
        self.assertEqual(plan.train_args["data"], str((root / "data.yaml").resolve()))


if __name__ == "__main__":
    unittest.main()
