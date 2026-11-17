from __future__ import annotations

import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from scripts.train_axe_5sku import safe_run_name, validate_manual_config
from src.evaluation.metrics import nearest_neighbor_retrieval
from src.models.mobilenetv3_embedding import MobileNetV3EmbeddingClassifier
from src.training.axe_data import training_manifest_paths
from src.training.classification_trainer import (
    ClassificationTrainer,
    classification_stage,
    is_better_classification_checkpoint,
)
from src.training.retrieval_trainer import (
    METRICS_CSV_FIELDS,
    is_better_retrieval_checkpoint,
    write_metrics_csv,
)


ROOT = Path(__file__).resolve().parents[1]


class AxeFiveSkuManualTrainingTests(unittest.TestCase):
    def test_macro_retrieval_is_mean_of_per_sku_accuracy(self) -> None:
        gallery_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        result = nearest_neighbor_retrieval(
            query_embeddings,
            ["A", "A", "A", "A", "A", "B"],
            gallery_embeddings,
            ["A", "B"],
            ["A", "B"],
        )
        self.assertAlmostEqual(result["micro_recall@1"], 5 / 6)
        self.assertAlmostEqual(result["macro_recall@1"], 0.5)
        self.assertEqual(result["per_sku"]["A"], {"queries": 5, "correct": 5, "recall@1": 1.0})
        self.assertEqual(result["per_sku"]["B"], {"queries": 1, "correct": 0, "recall@1": 0.0})

    def test_equal_retrieval_score_keeps_earlier_checkpoint(self) -> None:
        self.assertTrue(is_better_retrieval_checkpoint(0.81, 0.80))
        self.assertFalse(is_better_retrieval_checkpoint(0.80, 0.80))
        self.assertFalse(is_better_retrieval_checkpoint(0.79, 0.80))

    def test_classification_metrics_use_all_predictions_and_zero_policy(self) -> None:
        from src.evaluation.metrics import classification_metrics_from_predictions

        metrics = classification_metrics_from_predictions(
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            ["A", "B", "C", "D", "E"],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_class"]["A"]["precision"], 0.5)
        self.assertEqual(metrics["per_class"]["A"]["recall"], 1.0)
        self.assertEqual(metrics["per_class"]["B"]["f1"], 0.0)
        self.assertEqual(metrics["per_class"]["C"]["support"], 0)
        self.assertEqual(metrics["zero_division_policy"], 0)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 5)

    def test_classification_checkpoint_tie_breaking(self) -> None:
        self.assertTrue(is_better_classification_checkpoint(0.91, 0.5, 0.90, 0.4))
        self.assertTrue(is_better_classification_checkpoint(0.90, 0.3, 0.90, 0.4))
        self.assertFalse(is_better_classification_checkpoint(0.90, 0.4, 0.90, 0.4))
        self.assertFalse(is_better_classification_checkpoint(0.89, 0.1, 0.90, 0.4))

    def test_100_epoch_schedule_and_early_stopping_disabled(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "axe_5sku_100ep_classification.yaml").read_text(encoding="utf-8")
        )
        validate_manual_config(config)
        stages = [classification_stage(epoch, config["training"]["head_epochs"]) for epoch in range(100)]
        self.assertEqual(stages[:3], ["head", "head", "head"])
        self.assertEqual(stages[3:], ["fine_tune"] * 97)
        self.assertEqual(config["training"]["num_epochs"], 100)
        self.assertFalse(config["training"]["early_stopping_enabled"])
        self.assertIsNone(config["training"]["early_stopping_patience"])
        self.assertEqual(config["training"]["mode"], "classification_only")

    def test_stage1_keeps_backbone_batchnorm_frozen(self) -> None:
        model = MobileNetV3EmbeddingClassifier(num_classes=5, embedding_dim=16, pretrained=False)
        model.freeze_backbone()
        model.train()
        backbone_batchnorms = [
            module for module in model.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(backbone_batchnorms)
        self.assertTrue(all(not module.training for module in backbone_batchnorms))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(model.embedding_head[1].training)
        model.unfreeze_last_blocks(None)
        model.train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(any(module.training for module in backbone_batchnorms))

    def test_manual_training_manifest_set_excludes_test(self) -> None:
        paths = training_manifest_paths(Path("manifests"))
        self.assertEqual(set(paths), {"train", "validation"})
        self.assertEqual(paths["train"].name, "train.csv")
        self.assertEqual(paths["validation"].name, "val.csv")

    def test_classification_trainer_has_no_retrieval_evaluation(self) -> None:
        fit_source = inspect.getsource(ClassificationTrainer.fit)
        self.assertNotIn("evaluate_retrieval_protocol", fit_source)
        self.assertNotIn("gallery_loader", fit_source)
        self.assertNotIn("retrieval_r1", fit_source)

    def test_three_configs_differ_only_by_backbone_lr(self) -> None:
        paths = [
            ROOT / "configs" / "axe_5sku.yaml",
            ROOT / "configs" / "axe_5sku_lr1e5.yaml",
            ROOT / "configs" / "axe_5sku_lr1e4.yaml",
        ]
        configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
        for config in configs:
            validate_manual_config(config)
        learning_rates = [config["training"]["backbone_learning_rate"] for config in configs]
        self.assertEqual(learning_rates, [3e-5, 1e-5, 1e-4])
        for config in configs:
            config["training"]["backbone_learning_rate"] = 0
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])

    def test_metrics_csv_schema_and_scale_are_explicit(self) -> None:
        row = {
            "epoch": 1,
            "stage": "head",
            "head_lr": 1e-3,
            "backbone_lr": 0.0,
            "train_ce_loss": 1.2,
            "train_cls_top1": 0.5,
            "val_ce_loss": 1.1,
            "val_cls_top1": 0.6,
            "val_retrieval_r1_macro": 0.7,
            "val_retrieval_r1_micro": 0.72,
            "elapsed_seconds": 10.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            write_metrics_csv([row], path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), METRICS_CSV_FIELDS)
        self.assertEqual(float(rows[0]["val_retrieval_r1_macro"]), 0.7)

    def test_run_names_are_short_and_safe(self) -> None:
        self.assertEqual(safe_run_name("baseline_s42"), "baseline_s42")
        with self.assertRaises(ValueError):
            safe_run_name("bad/run")


if __name__ == "__main__":
    unittest.main()
