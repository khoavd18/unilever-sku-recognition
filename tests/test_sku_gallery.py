import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from src.data.annotations import AnnotationFormat, parse_yolo_annotation
from src.data.sku_gallery import build_class_map, gallery_split, is_augmented, prepare, source_group


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buffer, "JPEG")
    return buffer.getvalue()


class PolygonToBboxTests(unittest.TestCase):
    def test_polygon_becomes_min_max_box(self):
        row = "0 0.2 0.3 0.6 0.3 0.5 0.7 0.1 0.6"
        box = parse_yolo_annotation(row, ["a"], AnnotationFormat.YOLO_SEGMENTATION).box
        self.assertAlmostEqual(box.x_min, 0.1)
        self.assertAlmostEqual(box.x_max, 0.6)
        self.assertAlmostEqual(box.y_min, 0.3)
        self.assertAlmostEqual(box.y_max, 0.7)

    def test_group_and_aug_detection(self):
        self.assertEqual(source_group("abc_aug2_jpg.rf.0a1b"), "abc")
        self.assertEqual(source_group("abc_jpg.rf.0a1b"), "abc")
        self.assertTrue(is_augmented("abc_aug1_jpg.rf.0a1b"))
        self.assertFalse(is_augmented("abc_jpg.rf.0a1b"))

    def test_gallery_split_is_deterministic(self):
        self.assertEqual(gallery_split("g1"), gallery_split("g1"))


class PrepareTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            zpath = tmp / "x.zip"
            with ZipFile(zpath, "w") as z:
                for name in ("s1_jpg.rf.aa", "s1_aug1_jpg.rf.bb"):
                    z.writestr(f"train/images/{name}.jpg", _jpeg())
                    z.writestr(f"train/labels/{name}.txt", "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n1 0.5 0.5 0.4 0.4\n")
            items = [{"class_label": "AXE_a", "item_code": "111", "brand": "AXE", "product_name": "p"}]
            cmap = build_class_map(["AXE_a", "ZZZ_b"], items, {"axe": 0}, ["AXE"])
            self.assertEqual(cmap[0]["item_code"], "111")
            self.assertTrue(cmap[1]["needs_review"])
            audit = prepare(zpath, cmap, ["AXE"], tmp / "out", min_crop_px=5)
            self.assertEqual(audit["crop_source_groups"], 1)  # one representative per group
            self.assertEqual(len(list((tmp / "out" / "gallery" / "111").glob("*.jpg"))), 1)
            label = (tmp / "out" / "detector_brand21" / "labels" / "train" / "s1_jpg.rf.aa.txt").read_text().split()
            self.assertEqual(label[0], "0")  # brand id; unresolved-brand SKU row dropped
            self.assertEqual(len(label), 5)
            with self.assertRaises(FileExistsError):
                prepare(zpath, cmap, ["AXE"], tmp / "out")


if __name__ == "__main__":
    unittest.main()
