from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_clean_axe_dataset import EXPECTED_SKUS as SCRIPT_EXPECTED_SKUS
from src.catalog.profiles import load_product_profile
from src.training import axe_data
from src.training.embedding_data import training_manifest_paths


class EmbeddingDataCompatibilityTests(unittest.TestCase):
    def test_legacy_axe_constants_come_from_product_profile(self) -> None:
        expected = load_product_profile("axe5").legacy_class_names
        self.assertEqual(axe_data.EXPECTED_SKUS, expected)
        self.assertEqual(SCRIPT_EXPECTED_SKUS, expected)

    def test_training_manifests_exclude_held_out_test(self) -> None:
        paths = training_manifest_paths(Path("manifests"))
        self.assertEqual(set(paths), {"train", "validation"})
        self.assertNotIn("test", paths)


if __name__ == "__main__":
    unittest.main()
