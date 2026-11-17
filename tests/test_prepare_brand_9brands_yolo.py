from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import yaml

from scripts.prepare_brand_9brands_yolo import (
    Box,
    HashAudit,
    ImageUnit,
    Occurrence,
    annotation_sets_equivalent,
    build_family_components,
    coordinates_to_box,
    configure_brand_profile,
    infer_target_brand,
)


def make_box(brand: str, x_min: float = 0.1, delta: float = 0.0) -> Box:
    class_id = {
        "AXE": 0,
        "VIM": 1,
        "LIFEBUOY": 2,
        "HAZELINE": 3,
        "REXONA": 4,
        "CLEAR": 5,
        "DOVE": 6,
        "SUNSILK": 7,
        "TRESEMME": 8,
    }[brand]
    return Box(
        brand=brand,
        class_id=class_id,
        x_min=x_min + delta,
        y_min=0.2 + delta,
        x_max=x_min + 0.3 + delta,
        y_max=0.6 + delta,
        source_class_id=class_id,
        source_class_name=brand,
        source_line=1,
        raw_x_min=x_min + delta,
        raw_y_min=0.2 + delta,
        raw_x_max=x_min + 0.3 + delta,
        raw_y_max=0.6 + delta,
        was_clipped=False,
    )


def make_occurrence(image_hash: str, family: str, boxes: list[Box]) -> Occurrence:
    return Occurrence(
        archive_path=Path("source.zip"),
        archive_name="source.zip",
        source_split="train",
        image_index=1,
        image_member=f"train/images/{family}_aug1_jpg.rf.hash.jpg",
        label_indices=(2,),
        label_member=f"train/labels/{family}_aug1_jpg.rf.hash.txt",
        source_filename="image.jpg",
        family_id=family,
        image_sha256=image_hash,
        width=100,
        height=100,
        boxes=boxes,
        source_annotation_count=len(boxes),
        non_target_annotation_count=0,
        invalid_target_rows=[],
    )


class BrandNinePreparationTests(unittest.TestCase):
    def test_synthetic_appended_profile_requires_no_preparer_code_change(self) -> None:
        root = Path(__file__).resolve().parents[1]
        value = yaml.safe_load((root / "configs" / "brands.yaml").read_text(encoding="utf-8"))
        synthetic_id = max(row["id"] for row in value["brands"]) + 1
        value["brands"].append(
            {"id": synthetic_id, "name": "SYNTHETIC", "aliases": ["SYNTH"]}
        )
        value["profiles"]["brand_22brands"] = {
            "brand_ids": list(range(len(value["brands"])))
        }
        try:
            with tempfile.TemporaryDirectory() as temporary:
                registry_path = Path(temporary) / "brands.yaml"
                registry_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
                names = configure_brand_profile(
                    "brand_22brands", registry_path=registry_path
                )
                self.assertEqual(names[-1], "SYNTHETIC")
                self.assertEqual(infer_target_brand("SYNTH_demo"), "SYNTHETIC")
        finally:
            configure_brand_profile("brand_9brands")

    def test_mapping_is_explicit_and_clear_men_is_verified_variant(self) -> None:
        self.assertEqual(infer_target_brand("CLEARMEN_DauGoi"), "CLEAR")
        self.assertEqual(infer_target_brand("DOVE_SuaTam"), "DOVE")
        self.assertIsNone(infer_target_brand("SUNLIGHT_NRC"))
        self.assertIsNone(infer_target_brand("UNCLEAR_PRODUCT"))

    def test_equivalence_requires_class_and_strict_geometry(self) -> None:
        first = [make_box("DOVE"), make_box("CLEAR", 0.5)]
        equivalent = [make_box("CLEAR", 0.5, 5e-7), make_box("DOVE", delta=5e-7)]
        wrong_class = [make_box("DOVE"), make_box("SUNSILK", 0.5)]
        wrong_geometry = [make_box("DOVE", delta=2e-6), make_box("CLEAR", 0.5)]
        self.assertTrue(annotation_sets_equivalent(first, equivalent))
        self.assertFalse(annotation_sets_equivalent(first, wrong_class))
        self.assertFalse(annotation_sets_equivalent(first, wrong_geometry))

    def test_obb_conversion_clips_and_rejects_degenerate_box(self) -> None:
        box = coordinates_to_box(
            (-0.1, 0.2, 0.4, 0.2, 0.4, 0.8, -0.1, 0.8),
            "AXE",
            0,
            "AXE",
            1,
        )
        self.assertTrue(box.was_clipped)
        self.assertEqual(box.x_min, 0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            coordinates_to_box(
                (-0.2, 0.1, -0.1, 0.1, -0.1, 0.5, -0.2, 0.5),
                "AXE",
                0,
                "AXE",
                1,
            )

    def test_conflicting_locked_holdouts_quarantine_family_component(self) -> None:
        first = make_occurrence("a" * 64, "family-a", [make_box("AXE")])
        second = make_occurrence("b" * 64, "family-a", [make_box("DOVE")])
        first_unit = ImageUnit(
            first.image_sha256,
            first,
            [first],
            {first.family_id},
            first.boxes,
            "equivalent",
        )
        second_unit = ImageUnit(
            second.image_sha256,
            second,
            [second],
            {second.family_id},
            second.boxes,
            "equivalent",
        )
        audits = [
            HashAudit(first.image_sha256, [first], "accepted_equivalent", "", [tuple(first.boxes)], {"family-a"}),
            HashAudit(second.image_sha256, [second], "accepted_equivalent", "", [tuple(second.boxes)], {"family-a"}),
        ]
        clean, quarantined, report = build_family_components(
            [first_unit, second_unit],
            audits,
            {first.image_sha256: {"val"}, second.image_sha256: {"test"}},
            {},
            {},
        )
        self.assertFalse(clean)
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(report["quarantine_reason_counts"]["holdout_conflict_family"], 1)


if __name__ == "__main__":
    unittest.main()
