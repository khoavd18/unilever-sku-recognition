from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from src.training.embedding_workflow import build_training_plan, classes_from_manifests


def _manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "class_name", "split"))
        writer.writeheader()
        writer.writerows(rows)


class EmbeddingWorkflowTests(unittest.TestCase):
    def test_plan_uses_train_and_validation_without_opening_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "manifests"
            _manifest(
                split_dir / "train.csv",
                [{"image_path": "train.jpg", "class_name": "A", "split": "train"}],
            )
            _manifest(
                split_dir / "val.csv",
                [{"image_path": "val.jpg", "class_name": "A", "split": "val"}],
            )
            # An invalid held-out file must remain irrelevant to a training plan.
            (split_dir / "test.csv").write_text("not,a,valid,manifest\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"split_dir": str(split_dir), "image_size": 224},
                        "model": {},
                        "training": {},
                        "metric_learning": {},
                        "output": {},
                    }
                ),
                encoding="utf-8",
            )
            plan = build_training_plan(config)
            self.assertEqual(plan["class_to_idx"], {"A": 0})
            self.assertEqual(plan["counts"], {"train": 1, "val": 1})
            self.assertFalse(plan["test_manifest_loaded"])
            self.assertFalse(plan["training_started"])

    def test_class_mapping_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"image_path": "b.jpg", "class_name": "B", "split": "train"},
                {"image_path": "a.jpg", "class_name": "A", "split": "train"},
            ]
            _manifest(root / "train.csv", rows)
            _manifest(root / "val.csv", [rows[0]])
            self.assertEqual(classes_from_manifests(root), {"A": 0, "B": 1})


if __name__ == "__main__":
    unittest.main()
