# Dataset 469 SKU -> Model 1 (brand bbox) + Model 2 (gallery)

Nguồn: `Stock_Segmentation.v33-469_sku.yolov8.zip` (Roboflow, nhãn polygon, ảnh resize 640x640 stretch).
Script: `scripts/prepare_sku_gallery_dataset.py` (logic: `src/data/sku_gallery.py`, test: `tests/test_sku_gallery.py`).

```powershell
Set-Location D:\code\anhoa\train_embedding
python scripts\prepare_sku_gallery_dataset.py --output-dir data\sku469_v33
# thêm --write-detector-images để copy ảnh vào detector_brand21\images (Model 1 train được ngay)
```

Output (thư mục mới, không ghi đè):

| Đường dẫn | Dùng cho |
|---|---|
| `labels_sku/<split>/` | bbox 469 class (đã đổi từ polygon = min/max các điểm, clip vào ảnh) |
| `detector_brand21/` | Model 1: bbox theo brand, **thứ tự class training** của `configs/datasets/model1_brand21_data.yaml` + `data.yaml` |
| `gallery/<item_code>/*.jpg` | Model 2: crop theo SKU (1 ảnh đại diện / nhóm ảnh gốc) |
| `crops_manifest.csv` | provenance; cột `gallery_split` = chia 80/10/10 theo NHÓM; `match_method` |
| `class_map.csv`, `class_map_review.csv` | class -> item code -> brand; dòng cần người duyệt |
| `sku_split_counts.csv`, `audit_report.json` | số crop theo SKU/split, leakage, class thiếu |

Lưu ý: export có ~3 phiên bản gần giống mỗi ảnh gốc và cùng nhóm xuất hiện ở nhiều split Roboflow,
nên Model 2 phải dùng `gallery_split`, không dùng split Roboflow. Class không khớp chính xác với
`ITEMCODE_SKU.xlsx` được ghi trong `class_map_review.csv`; `--strict-class-map` loại chúng khỏi gallery.
Ảnh nguồn đã bị stretch về 640x640 nên crop méo tỉ lệ hơn crop từ ảnh kệ thật.

Nếu một nhóm chỉ có ảnh `_aug`, representative vẫn lấy một ảnh augmented;
gallery không bảo đảm chỉ chứa ảnh gốc chưa tăng cường.

Bước tiếp theo để tạo manifest train/val/test đủ SKU, kiểm tra crop trùng nội
dung và chuẩn bị MobileNetV3 nằm trong [SKU459_TRAINING.md](SKU459_TRAINING.md).
