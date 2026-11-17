from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from src.catalog.catalog import (
    CatalogRecord,
    create_scoped_catalog,
    import_workbook,
    require_usable_catalog,
    validate_records,
)
from src.catalog.mapping import build_mapping, load_mapping
from src.catalog.profiles import load_product_profile


HEADERS = ["STT", "CATEGORY", "BRAND", "ITEM_CODE", "PRODUCT_NAME_VN", "CLASS LABEL"]


def workbook_with(rows: list[list[object]], number_format: str | None = None) -> Path:
    directory = Path(tempfile.mkdtemp())
    path = directory / "catalog.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    for row in rows:
        sheet.append(row)
    if number_format:
        sheet["D2"].number_format = number_format
    workbook.save(path)
    return path


class CatalogTests(unittest.TestCase):
    def test_loaded_mapping_requires_string_item_codes_and_identity_guard(self) -> None:
        active = Path(
            "artifacts/mappings/axe5_classification_100ep_s42_axe5_demo_mapping.v3.json"
        )
        payload = load_mapping(active)
        self.assertEqual(payload["mappings"][-1]["item_code"], "64863894")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            changed = json.loads(active.read_text(encoding="utf-8"))
            changed["mappings"][0]["item_code"] = 69566283
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                load_mapping(path)

    def test_leading_zero_text_is_preserved(self) -> None:
        bundle = import_workbook(workbook_with([[1, "CAT", "B", "00123", "Tên SP 10ml/6 chai", "Legacy"]]))
        self.assertEqual(bundle["records"][0]["item_code"], "00123")
        self.assertEqual(bundle["records"][0]["packaging"], "10ml/6 chai")
        self.assertEqual(bundle["records"][0]["class_name"], "00123_Legacy")
        self.assertFalse(any(issue["code"] == "ambiguous_numeric_item_code_format" for issue in bundle["issues"]))

    def test_fixed_width_numeric_format_restores_zeroes(self) -> None:
        bundle = import_workbook(workbook_with([[1, "CAT", "B", 123, "Tên SP", "Legacy"]], "00000"))
        self.assertEqual(bundle["records"][0]["item_code"], "00123")

    def test_general_numeric_format_is_reported(self) -> None:
        bundle = import_workbook(workbook_with([[1, "CAT", "B", 123, "Tên SP", "Legacy"]]))
        self.assertTrue(any(issue["code"] == "ambiguous_numeric_item_code_format" for issue in bundle["issues"]))

    def test_duplicate_and_conflicting_rows(self) -> None:
        duplicate = import_workbook(workbook_with([
            [1, "CAT", "B", "100", "Same", "Legacy"],
            [2, "CAT", "B", "100", "Same", "Legacy"],
        ]))
        self.assertEqual(duplicate["record_count"], 1)
        self.assertTrue(any(issue["code"] == "duplicate_identical_record" for issue in duplicate["issues"]))
        conflict = import_workbook(workbook_with([
            [1, "CAT", "B", "100", "First", "Legacy1"],
            [2, "CAT", "B", "100", "Second", "Legacy2"],
        ]))
        self.assertEqual(conflict["status"], "invalid")
        self.assertTrue(any(issue["code"] == "conflicting_item_code" for issue in conflict["issues"]))

    def test_folder_collision_and_ambiguous_legacy_are_detected(self) -> None:
        common = dict(item_name="Name", brand="B", category="C", class_name="X", workbook_reference={"cell": "D2"})
        records = [
            CatalogRecord(item_code="1", filesystem_name="same", legacy_class_name="Legacy", **common),
            CatalogRecord(item_code="2", filesystem_name="SAME", legacy_class_name="Legacy", **common),
        ]
        codes = {issue.code for issue in validate_records(records)}
        self.assertIn("filesystem_name_collision", codes)
        self.assertIn("ambiguous_legacy_class", codes)

    def test_checkpoint_order_is_preserved_and_ambiguity_blocks_mapping(self) -> None:
        records = [
            {"item_code": "69566283", "item_name": "Black", "brand": "AXE", "category": "D", "class_name": "69566283_Black", "legacy_class_name": "AXE_NamXitNganMuiToanThanBlack_135ml_chai"},
            {"item_code": "69566284", "item_name": "Gold", "brand": "AXE", "category": "D", "class_name": "69566284_Gold", "legacy_class_name": "AXE_NamXitNganMuiToanThanGold_135ml_chai"},
            {"item_code": "64823658", "item_name": "Blue spray", "brand": "AXE", "category": "D", "class_name": "64823658_Blue", "legacy_class_name": "BlueSpray"},
            {"item_code": "64863894", "item_name": "Blue wash", "brand": "AXE", "category": "W", "class_name": "64863894_Blue", "legacy_class_name": "BlueWash"},
        ]
        catalog = {"catalog_version": "v", "records": records}
        class_to_idx = {
            "AXE_NamXitNganMuiToanThanGold_135ml_chai": 1,
            "AXE_NamXitNganMuiToanThanBlack_135ml_chai": 0,
        }
        result = build_mapping(catalog, class_to_idx)
        self.assertEqual([row["item_code"] for row in result["mappings"]], ["69566283", "69566284"])
        ambiguous_catalog = {"catalog_version": "v", "records": records + [{**records[0], "item_code": "999"}]}
        invalid = build_mapping(ambiguous_catalog, {"AXE_NamXitNganMuiToanThanBlack_135ml_chai": 0})
        self.assertEqual(invalid["status"], "invalid")
        self.assertTrue(any(issue["code"] == "ambiguous_legacy_class_match" for issue in invalid["issues"]))

    def test_scoped_catalog_preserves_full_invalid_status_and_reports_out_of_scope_error(self) -> None:
        full = import_workbook(workbook_with([
            [1, "CAT", "B", "001", "First", "Legacy1"],
            [2, "CAT", "B", "002", "Second", "Legacy2"],
            [3, "CAT", "CLEAR", None, "Genuine product missing code", None],
        ]))
        self.assertEqual(full["status"], "invalid")
        scoped = create_scoped_catalog(full, ["001", "002"], "demo")
        require_usable_catalog(scoped, require_scope=True)
        self.assertEqual(scoped["status"], "valid")
        self.assertEqual(scoped["source_full_catalog"]["status"], "invalid")
        self.assertTrue(scoped["source_workbook_fingerprint"].startswith("sha256:"))
        self.assertEqual(scoped["scope"]["selected_item_codes"], ["001", "002"])
        self.assertEqual(scoped["known_out_of_scope_issues"][0]["code"], "missing_item_code")

    def test_scoped_catalog_reports_identical_duplicates_and_blocks_conflict(self) -> None:
        duplicate_full = import_workbook(workbook_with([
            [1, "CAT", "B", "001", "Same", "Legacy"],
            [2, "CAT", "B", "001", "Same", "Legacy"],
        ]))
        scoped_duplicate = create_scoped_catalog(duplicate_full, ["001"], "duplicate_demo")
        self.assertEqual(
            scoped_duplicate["scope"]["occurrence_validation"][0]["workbook_occurrence_count"], 2
        )
        self.assertEqual(scoped_duplicate["status"], "valid_with_warnings")
        conflict_full = import_workbook(workbook_with([
            [1, "CAT", "B", "001", "First", "Legacy1"],
            [2, "CAT", "B", "001", "Second", "Legacy2"],
        ]))
        conflict = create_scoped_catalog(conflict_full, ["001"], "conflict_demo")
        self.assertEqual(conflict["status"], "invalid")
        self.assertTrue(any(issue["code"] == "scoped_item_code_conflict" for issue in conflict["issues"]))

    def test_explicit_axe_scope_keeps_blue_lavender_spray_out_of_scope(self) -> None:
        records = []
        class_to_idx = {}
        for index, (legacy, code) in enumerate({
            "AXE_NamXitNganMuiToanThanBlack_135ml_chai": "69566283",
            "AXE_NamXitNganMuiToanThanGold_135ml_chai": "69566284",
            "AXE_XitNganMuiToanThanIceChill_135ml_chai": "69566277",
            "AXE_SuaTamNuocHoaGoldenMango_532ML_Chai": "64863893",
            "AXE_SuaTamNuocHoaBlueLavender_532ML_Chai": "64863894",
        }.items()):
            class_to_idx[legacy] = index
            records.append({
                "item_code": code, "item_name": legacy, "brand": "AXE", "category": "AXE",
                "class_name": f"{code}_{legacy}", "legacy_class_name": legacy,
            })
        scoped = {
            "catalog_version": "scoped", "status": "valid", "records": records,
            "scope": {"selected_item_codes": [row["item_code"] for row in records]},
        }
        result = build_mapping(scoped, class_to_idx)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["blue_lavender_guard"]["spray_explicitly_out_of_scope"])
        self.assertTrue(result["blue_lavender_guard"]["identity_guard_verified"])

    def test_synthetic_product_profile_builds_generic_mapping_and_rejects_partial_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "fixture_products.yaml"
            profile_path.write_text(
                """\
schema_version: 1
name: fixture_products
brand: FIXTURE
classes:
  - class_index: 0
    legacy_class_name: Fixture_Alpha
    item_code: "001"
  - class_index: 1
    legacy_class_name: Fixture_Beta
    item_code: "002"
identity_guards:
  similar_packaging:
    original: "001"
    refill: "009"
""",
                encoding="utf-8",
            )
            profile = load_product_profile(path=profile_path)

        catalog = {
            "catalog_version": "fixture-v1",
            "status": "valid",
            "scope": {"selected_item_codes": ["001", "002"]},
            "records": [
                {
                    "item_code": "001",
                    "item_name": "Alpha",
                    "brand": "FIXTURE",
                    "category": "Fixture",
                    "class_name": "001_Alpha",
                    "legacy_class_name": "Fixture_Alpha",
                },
                {
                    "item_code": "002",
                    "item_name": "Beta",
                    "brand": "FIXTURE",
                    "category": "Fixture",
                    "class_name": "002_Beta",
                    "legacy_class_name": "Fixture_Beta",
                },
            ],
        }
        result = build_mapping(
            catalog,
            {"Fixture_Alpha": 0, "Fixture_Beta": 1},
            product_profile=profile,
        )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["product_profile"], "fixture_products")
        self.assertNotIn("blue_lavender_guard", result)
        guard = result["identity_guards"]["similar_packaging"]
        self.assertTrue(guard["identity_guard_verified"])
        self.assertTrue(guard["identities"]["refill"]["explicitly_out_of_scope"])
        self.assertEqual(
            [row["item_code"] for row in result["mappings"]], ["001", "002"]
        )

        partial = build_mapping(
            catalog,
            {"Fixture_Alpha": 0},
            product_profile=profile,
        )
        self.assertEqual(partial["status"], "invalid")
        mismatch = next(
            issue
            for issue in partial["issues"]
            if issue["code"] == "checkpoint_product_profile_class_mismatch"
        )
        self.assertEqual(mismatch["missing_classes"], ["Fixture_Beta"])
        self.assertEqual(mismatch["unexpected_classes"], [])


if __name__ == "__main__":
    unittest.main()
