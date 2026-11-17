from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from src.data.sku_training import prepare_training_dataset, repair_splits
from src.training.embedding_workflow import build_training_plan, main, run_training


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root: Path, groups: int = 4, duplicate: bool = False) -> Path:
    source = root / "source"
    source.mkdir()
    raw, identities = [], []
    for label_index, label in enumerate(("AXE_one", "DOVE_two")):
        code = str(100 + label_index)
        identities.append({"sku_class_name": label, "item_code": code, "brand_name": label.split("_")[0], "product_name": label, "match_method": "exact"})
        for group_index in range(groups):
            relative = f"gallery/{code}/group_{group_index}.jpg"
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40 + group_index, 60), (50 + label_index * 100, group_index * 30, 40)).save(path)
            raw.append({"crop_path": relative, "gallery_split": "train", "group_id": f"{label}_{group_index}",
                        "source_image": f"train/images/{label}_{group_index}_jpg.rf.abc.jpg", "annotation_index": "0",
                        "sku_class_name": label, "item_code": code, "brand_name": label.split("_")[0], "match_method": "exact"})
    if duplicate:
        copied = dict(raw[0], crop_path="gallery/100/copy.jpg", group_id="copy_group", gallery_split="valid")
        (source / copied["crop_path"]).write_bytes((source / raw[0]["crop_path"]).read_bytes())
        raw.append(copied)
    write_csv(source / "crops_manifest.csv", raw)
    write_csv(source / "class_map.csv", identities)
    return source


def training_config(root: Path, source: Path, prepared: Path) -> Path:
    config = {
        "dataset": {
            "path": str(prepared), "crop_root": str(source), "split_dir": str(prepared / "manifests"),
            "class_mapping": str(prepared / "class_mapping.json"), "preparation_report": str(prepared / "preparation_report.json"),
            "require_all_classes": True, "require_group_disjoint": True, "image_size": 224,
        },
        "training": {"device": "cpu"},
        "output": {"checkpoint_dir": str(root / "checkpoints"), "output_dir": str(root / "outputs"), "require_run_name": True},
    }
    path = root / "training.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class SkuPreparationTests(unittest.TestCase):
    def test_multi_sku_source_components_are_kept_whole(self):
        labels = {f"group_{index}": {"A", "B"} for index in range(3)}
        assigned = repair_splits(labels, {group: Counter(train=2) for group in labels})
        self.assertEqual(set(assigned.values()), {"train", "val", "test"})
        for label in ("A", "B"):
            self.assertEqual({assigned[group] for group in labels if label in labels[group]}, {"train", "val", "test"})

    def test_complete_deterministic_splits_merge_duplicate_groups_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, duplicate=True)
            original = (source / "crops_manifest.csv").read_bytes()
            prepared = root / "prepared"
            report = prepare_training_dataset(source, prepared)
            self.assertEqual(report["class_count"], 2)
            self.assertEqual(report["sku_counts"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(report["duplicate_crop_count"], 1)
            self.assertEqual(report["independent_component_count"], 8)
            with (prepared / "crop_manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for key in ("group_id", "family_id", "image_sha256"):
                memberships = defaultdict(set)
                for row in rows:
                    memberships[row[key]].add(row["split"])
                self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))
            self.assertTrue(all(Path(row["image_path"]).is_relative_to(source) for row in rows))
            self.assertFalse((prepared / "gallery").exists())
            self.assertEqual(original, (source / "crops_manifest.csv").read_bytes())
            second = root / "second"
            prepare_training_dataset(source, second)
            self.assertEqual((prepared / "group_assignments.csv").read_bytes(), (second / "group_assignments.csv").read_bytes())
            with self.assertRaises(FileExistsError):
                prepare_training_dataset(source, prepared)

    def test_too_few_independent_groups_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, groups=2)
            with self.assertRaisesRegex(ValueError, "3 independent"):
                prepare_training_dataset(source, root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_identical_crop_with_conflicting_item_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root)
            (source / "gallery/101/group_0.jpg").write_bytes((source / "gallery/100/group_0.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "conflicting item codes"):
                prepare_training_dataset(source, root / "prepared")
            self.assertFalse((root / "prepared").exists())

    def test_quarantine_removes_both_labels_without_editing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root)
            original_image = (source / "gallery/100/group_0.jpg").read_bytes()
            (source / "gallery/101/group_0.jpg").write_bytes(original_image)
            report = prepare_training_dataset(source, root / "prepared", quarantine_conflicts=True)
            self.assertEqual(report["quarantined_crop_count"], 2)
            self.assertEqual(report["conflicting_hash_count"], 1)
            self.assertEqual(report["sku_counts"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(sum(report["crop_counts"].values()), 6)
            self.assertTrue((root / "prepared/quarantined_crops.csv").is_file())
            self.assertEqual((source / "gallery/101/group_0.jpg").read_bytes(), original_image)

    def test_crop_path_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root)
            manifest = source / "crops_manifest.csv"
            manifest.write_text(manifest.read_text().replace("gallery/100/group_0.jpg", "../../outside.jpg"))
            with self.assertRaisesRegex(ValueError, "escapes source"):
                prepare_training_dataset(source, root / "prepared")


class SkuTrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = fixture(self.root)
        self.prepared = self.root / "prepared"
        prepare_training_dataset(self.source, self.prepared)
        self.config = training_config(self.root, self.source, self.prepared)

    def test_preflight_never_loads_test_or_constructs_model(self):
        (self.prepared / "manifests/test.csv").write_text("invalid held-out file", encoding="utf-8")
        with patch("src.training.embedding_workflow.MobileNetV3EmbeddingClassifier", side_effect=AssertionError("model constructed")):
            output = io.StringIO()
            with redirect_stdout(output):
                main(["--config", str(self.config), "--run-name", "baseline", "--preflight"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["class_count"], 2)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["test_manifest_loaded"])
        self.assertFalse((self.root / "checkpoints").exists())
        self.assertFalse((self.root / "outputs").exists())

    def test_unique_run_name_is_required(self):
        with self.assertRaisesRegex(ValueError, "--run-name"):
            build_training_plan(self.config)
        with self.assertRaisesRegex(ValueError, "Run name"):
            build_training_plan(self.config, run_name="../escape")

    def test_prepared_manifest_edit_is_detected(self):
        manifest = self.prepared / "manifests/train.csv"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            build_training_plan(self.config, run_name="baseline")

    def test_existing_run_is_rejected_before_model_or_training(self):
        (self.root / "checkpoints/baseline").mkdir(parents=True)
        with patch("src.training.embedding_workflow.MobileNetV3EmbeddingClassifier", side_effect=AssertionError("model constructed")):
            with self.assertRaises(FileExistsError):
                run_training(self.config, run_name="baseline")

    def test_preflight_rejects_resume_from_another_run(self):
        with self.assertRaisesRegex(ValueError, "same existing run"):
            main(["--config", str(self.config), "--run-name", "baseline", "--preflight", "--resume", str(self.root / "other/last_model.pth")])


if __name__ == "__main__":
    unittest.main()
