from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from src.catalog.mapping import EXPECTED_AXE5, build_mapping
from src.catalog.profiles import load_product_profile
from src.config.paths import (
    PROJECT_ROOT,
    resolve_config_path,
    resolve_config_relative_path,
    resolve_repo_path,
)
from src.config.registry import BrandRegistry, load_brand_registry
from src.utils.config import PROJECT_ROOT as LEGACY_PROJECT_ROOT
from src.utils.config import resolve_path as legacy_resolve_path


def registry_document(
    brands: list[dict[str, object]], profile_ids: list[int] | None = None
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "brands": brands,
        "profiles": {"test": {"brand_ids": profile_ids or list(range(len(brands))) }},
    }


class BrandRegistryTests(unittest.TestCase):
    def test_default_profiles_preserve_deployed_detector_order_and_alias(self) -> None:
        registry = load_brand_registry()
        self.assertEqual(
            registry.names("brand_9brands"),
            ("AXE", "VIM", "LIFEBUOY", "HAZELINE", "REXONA", "CLEAR", "DOVE", "SUNSILK", "TRESEMME"),
        )
        self.assertEqual(registry.names("brand_5brands"), registry.names("brand_9brands")[:5])
        self.assertEqual(registry.infer_brand("CLEARMEN_DauGoi", "brand_9brands"), "CLEAR")
        self.assertIsNone(registry.infer_brand("CLEARMEN_DauGoi", "brand_5brands"))
        self.assertIsNone(registry.infer_brand("UNCLEAR_PRODUCT", "brand_9brands"))

    def test_brand_21_profile_appends_ids_without_changing_deployed_nine(self) -> None:
        registry = load_brand_registry()
        expected = (
            "AXE",
            "VIM",
            "LIFEBUOY",
            "HAZELINE",
            "REXONA",
            "CLEAR",
            "DOVE",
            "SUNSILK",
            "TRESEMME",
            "BEST_FOODS",
            "CLOSE_UP",
            "COMFORT",
            "KNORR",
            "LUX",
            "OMO",
            "PONDS",
            "PS",
            "SIMPLE",
            "SUNLIGHT",
            "SURF",
            "VASELINE",
        )
        self.assertEqual(registry.names("brand_21brands"), expected)
        self.assertEqual(registry.names("brand_21brands")[:9], registry.names("brand_9brands"))
        source_names = (
            "AXE",
            "BEST FOODS",
            "CLEAR",
            "CLOSE_UP",
            "COMFORT",
            "DOVE",
            "HAZELINE",
            "KNORR",
            "LIFEBUOY",
            "LUX",
            "OMO",
            "PONDS",
            "PS",
            "REXONA",
            "SIMPLE",
            "SUNLIGHT",
            "SUNSILK",
            "SURF",
            "TRESEMME",
            "VASELINE",
            "VIM",
        )
        source_to_training = [
            registry.profile("brand_21brands")[expected.index(registry.infer_brand(name, "brand_21brands"))].id
            for name in source_names
        ]
        self.assertEqual(
            source_to_training,
            [0, 9, 5, 10, 11, 6, 3, 12, 2, 13, 14, 15, 16, 4, 17, 18, 7, 19, 8, 20, 1],
        )

    def test_synthetic_additional_brand_requires_only_configuration(self) -> None:
        value = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "brands.yaml").read_text(encoding="utf-8")
        )
        value["brands"].append({"id": 21, "name": "SYNTHETIC", "aliases": ["SYNTH"]})
        value["profiles"]["brand_22brands"] = {"brand_ids": list(range(22))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brands.yaml"
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            registry = BrandRegistry.from_file(path)

        self.assertEqual(len(registry.names("brand_9brands")), 9)
        self.assertEqual(registry.class_names("brand_22brands")[21], "SYNTHETIC")
        self.assertEqual(registry.infer_brand("SYNTH_demo_product", "brand_22brands"), "SYNTHETIC")

    def test_duplicate_ids_are_rejected(self) -> None:
        value = registry_document([
            {"id": 0, "name": "FIRST", "aliases": []},
            {"id": 0, "name": "SECOND", "aliases": []},
        ], [0])
        with self.assertRaisesRegex(ValueError, "Duplicate brand id"):
            BrandRegistry.from_mapping(value)

    def test_duplicate_and_prefix_ambiguous_aliases_are_rejected(self) -> None:
        collision = registry_document([
            {"id": 0, "name": "FIRST", "aliases": ["SHARED"]},
            {"id": 1, "name": "SECOND", "aliases": ["shared"]},
        ])
        with self.assertRaisesRegex(ValueError, "ambiguous brand alias"):
            BrandRegistry.from_mapping(collision)

        prefix = registry_document([
            {"id": 0, "name": "FIRST", "aliases": ["GROUP"]},
            {"id": 1, "name": "SECOND", "aliases": ["GROUP_CHILD"]},
        ])
        with self.assertRaisesRegex(ValueError, "Ambiguous brand alias prefixes"):
            BrandRegistry.from_mapping(prefix)

    def test_missing_fields_and_noncontiguous_profiles_are_rejected(self) -> None:
        missing_aliases = {
            "schema_version": 1,
            "brands": [{"id": 0, "name": "FIRST"}],
            "profiles": {"test": {"brand_ids": [0]}},
        }
        with self.assertRaisesRegex(ValueError, "missing fields"):
            BrandRegistry.from_mapping(missing_aliases)

        noncontiguous = registry_document([
            {"id": 0, "name": "FIRST", "aliases": []},
            {"id": 2, "name": "THIRD", "aliases": []},
        ], [0, 2])
        with self.assertRaisesRegex(ValueError, "contiguous detector ids"):
            BrandRegistry.from_mapping(noncontiguous)


class PathResolutionTests(unittest.TestCase):
    def test_repository_paths_are_cwd_independent_and_legacy_api_is_preserved(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                expected = (PROJECT_ROOT / "configs" / "brands.yaml").resolve()
                self.assertEqual(resolve_repo_path("configs/brands.yaml"), expected)
                self.assertEqual(resolve_config_path("configs/brands.yaml"), expected)
                self.assertEqual(legacy_resolve_path("configs/brands.yaml"), expected)
                self.assertEqual(LEGACY_PROJECT_ROOT, PROJECT_ROOT)
            finally:
                os.chdir(previous)

    def test_absolute_and_config_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "nested" / "example.yaml"
            absolute = root / "absolute.bin"
            self.assertEqual(resolve_repo_path(absolute), absolute)
            self.assertEqual(
                resolve_config_relative_path("assets/model.bin", config),
                root / "nested" / "assets" / "model.bin",
            )


class ProductIdentityProfileTests(unittest.TestCase):
    def test_axe_profile_is_exact_and_blue_lavender_identities_are_distinct(self) -> None:
        profile = load_product_profile("axe5")
        self.assertEqual(profile.expected_mapping, EXPECTED_AXE5)
        self.assertEqual(
            profile.item_codes,
            ("69566283", "69566284", "69566277", "64863893", "64863894"),
        )
        guard = profile.identity_guard("blue_lavender")
        self.assertEqual(guard["spray"], "64823658")
        self.assertEqual(guard["body_wash"], "64863894")
        self.assertNotEqual(guard["spray"], guard["body_wash"])
        self.assertNotIn(guard["spray"], profile.item_codes)
        self.assertEqual(profile.classes[4].item_code, guard["body_wash"])

    def test_mapping_default_uses_profile_and_preserves_blue_lavender_guard(self) -> None:
        profile = load_product_profile()
        records = [
            {
                "item_code": row.item_code,
                "item_name": row.legacy_class_name,
                "brand": profile.brand,
                "category": "fixture",
                "class_name": f"{row.item_code}_{row.legacy_class_name}",
                "legacy_class_name": row.legacy_class_name,
            }
            for row in profile.classes
        ]
        catalog = {
            "catalog_version": "fixture-v1",
            "status": "valid",
            "records": records,
            "scope": {"selected_item_codes": list(profile.item_codes)},
        }
        class_to_idx = {
            row.legacy_class_name: row.class_index for row in profile.classes
        }
        result = build_mapping(catalog, class_to_idx)

        self.assertEqual(result["status"], "verified")
        self.assertEqual(
            [row["item_code"] for row in result["mappings"]], list(profile.item_codes)
        )
        self.assertTrue(result["blue_lavender_guard"]["spray_explicitly_out_of_scope"])
        self.assertFalse(result["blue_lavender_guard"]["spray_is_trained"])
        self.assertTrue(result["blue_lavender_guard"]["body_wash_is_trained"])
        self.assertTrue(result["blue_lavender_guard"]["identity_guard_verified"])


if __name__ == "__main__":
    unittest.main()
