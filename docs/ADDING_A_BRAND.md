# Thêm brand hoặc item bằng cấu hình

“Thêm brand” có thể chỉ ba việc khác nhau. Phải chọn đúng phạm vi trước khi sửa:

1. thêm brand vào registry/catalog;
2. thêm class vào Model 1 detector;
3. thêm item code vào Model 2 gallery.

Ba việc này độc lập. Một thay đổi YAML không làm checkpoint cũ học class mới.

## 1. Contract đang tồn tại

`configs/brands.yaml` giữ ID detector ổn định `0..8` cho AXE, VIM, LIFEBUOY, HAZELINE, REXONA, CLEAR, DOVE, SUNSILK và TRESEMME. Không đổi ID hoặc reorder profile `brand_9brands` vì checkpoint runtime đang dùng đúng thứ tự này.

`configs/products/axe5.yaml` giữ năm class/item code của checkpoint MobileNet hiện hành. Item code phải được quote để luôn là string và giữ được leading zero nếu có.

Gallery runtime hiện chỉ có năm item AXE. Detector có thể thấy brand khác nhưng runtime cố ý trả `missing_gallery_for_detected_brand` thay vì gán một item AXE sai.

## 2. Ví dụ thực tế: thêm class detector `PONDS`

Ví dụ này minh họa configuration-driven extension của registry và generic detector trainer. Nó không chứng minh checkpoint hiện tại nhận diện được PONDS.

### Bước 1 — thêm identity mới, không sửa ID cũ

Thêm entry sau vào cuối `brands` trong `configs/brands.yaml`:

```yaml
  - id: 9
    name: PONDS
    aliases:
      - POND_S
```

Thêm profile mới, không sửa profile runtime cũ:

```yaml
profiles:
  brand_5brands:
    brand_ids: [0, 1, 2, 3, 4]
  brand_9brands:
    brand_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  brand_10brands:
    brand_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Tên canonical phải uppercase/normalized. Alias phải tường minh và không được trùng hoặc tạo prefix mơ hồ với brand khác. Kiểm tra registry:

```powershell
Set-Location D:\code\anhoa\train_embedding
python -c "from src.config import load_brand_registry; r=load_brand_registry(); print(r.class_names('brand_10brands'))"
python -m unittest tests.test_config_registry -v
```

### Bước 2 — đọc mapping từ từng export

Với mỗi ZIP/export, đọc class ID từ chính `data.yaml` hoặc COCO `categories`. Không dùng tên ZIP để đoán class và không copy ID của export này sang export khác.

Khai báo source trước trong `configs/datasets/sources.yaml`. Ví dụ:

```yaml
  - name: brand_ponds_v1
    path: data/brand_zips/BRAND_PONDS.v1.yolov8-obb.zip
    annotation_format: yolo_obb
    class_mapping_source: archive_data_yaml
```

Khi đã review file, nên thêm SHA-256 thực tế:

```powershell
(Get-FileHash data\brand_zips\BRAND_PONDS.v1.yolov8-obb.zip -Algorithm SHA256).Hash.ToLowerInvariant()
```

Không đưa checksum placeholder vào config. Registry từ chối source path/name/hash trùng, checksum drift và source chưa khai báo.

Khai báo geometry nguồn một cách tường minh là một trong:

- `coco_bbox`;
- `yolo_bbox`;
- `yolo_obb`;
- `yolo_segmentation`.

Nếu source format/geometry chưa được parser hiện hành hỗ trợ, dừng với lỗi rõ ràng và thêm adapter có test. Không suy luận geometry chỉ từ số coordinate.

Entry point cấu hình-driven dùng brand profile mới:

```powershell
# Lệnh này tạo dataset mới; chỉ chạy sau khi đã review source/inventory/decisions.
python scripts\prepare_detector_dataset.py `
  --brand-profile brand_10brands `
  --zip-dir data\brand_zips `
  --inventory-dir outputs\brand10_inventory `
  --five-brand-dir data\brand_5brands_yolo `
  --output-dir data\brand_10brands_yolo

# Sau khi output tồn tại, kiểm tra read-only:
python scripts\prepare_detector_dataset.py `
  --brand-profile brand_10brands `
  --output-dir data\brand_10brands_yolo `
  --check-only
```

`prepare_brand_9brands_yolo.py` được giữ như compatibility entry point và gọi cùng implementation. `--brand-profile` quyết định class order; nó không thay thế việc đọc mapping của từng source export.

Prepared detector dataset phải có class order đích:

```yaml
# data/brand_10brands_yolo/data.yaml
path: D:/code/anhoa/train_embedding/data/brand_10brands_yolo
train: images/train
val: images/val
test: images/test
nc: 10
names:
  0: AXE
  1: VIM
  2: LIFEBUOY
  3: HAZELINE
  4: REXONA
  5: CLEAR
  6: DOVE
  7: SUNSILK
  8: TRESEMME
  9: PONDS
```

Mỗi source class phải được map qua canonical name/alias đã review. Annotation conflict phải được đưa vào review/quarantine; không tự union hoặc chọn nhãn. Giữ historical holdout, exact-image deduplication, augmentation-family grouping và provenance.

`dataset_checks.json` tối thiểu phải chứng minh mapping và split policy mà generic preflight kiểm tra:

```json
{
  "status": "passed",
  "ordinary_yolo_detection_rows_only": true,
  "mapping_verified": {
    "0": "AXE",
    "1": "VIM",
    "2": "LIFEBUOY",
    "3": "HAZELINE",
    "4": "REXONA",
    "5": "CLEAR",
    "6": "DOVE",
    "7": "SUNSILK",
    "8": "TRESEMME",
    "9": "PONDS"
  },
  "exact_image_cross_split_overlap": {
    "train_validation": 0,
    "train_test": 0,
    "validation_test": 0
  },
  "cross_split_family_overlap": 0,
  "test_split_evaluated": false
}
```

Ví dụ fixture trong test chỉ chứng minh registry/generic preflight chấp nhận mapping mới mà không sửa brand-specific Python. Nó không chứng minh data thực đã sạch hoặc model nhận diện được brand mới.

### Bước 3 — tạo detector config mới

Tạo `configs/brand_10brands_yolo.yaml` với output mới:

```yaml
model: yolov8n.pt
data: data/brand_10brands_yolo/data.yaml
project: runs/model1_brand10
name: yolov8n_100ep_s42

train:
  epochs: 100
  imgsz: 640
  batch: 8
  workers: 0
  device: 0
  amp: true
  seed: 42
  deterministic: true
  patience: 0
  single_cls: false
  pretrained: true
  resume: false
  exist_ok: false
  save: true
  plots: true
  val: true
  cache: false

artifacts:
  evaluate_test_split: false

compatibility:
  verified_ultralytics: 8.3.155
  install_or_upgrade_automatically: false
```

Các hyperparameter còn lại nên copy tường minh từ baseline được chọn; không để refactor vô tình đổi hành vi thí nghiệm.

Chạy preflight trước:

```powershell
python scripts\train_detector.py `
  --config configs\brand_10brands_yolo.yaml `
  --preflight
```

Chỉ khi preflight, review dataset và output path đều đúng mới chủ động train:

```powershell
python scripts\train_detector.py `
  --config configs\brand_10brands_yolo.yaml
```

Sau training, phải báo cáo validation precision, recall, mAP50, mAP50-95 và per-brand metrics. Chỉ thay `yolo_weights` trong backend config khi checkpoint mới có task `detect`, class order chính xác và contract tests đã qua. Test split vẫn dành cho đánh giá cuối, không dùng để tune.

## 3. Thêm item code/gallery coverage

Thêm detector brand không tự tạo item-level recognition. Muốn resolve item cho PONDS hoặc thêm SKU AXE, cần xử lý riêng:

1. xác minh item code/identity trong catalog; item code giữ dạng string;
2. chuẩn bị ảnh gallery và provenance theo item;
3. xác định checkpoint embedding nào sẽ dùng;
4. tạo mapping tương thích chính xác với checkpoint class order nếu workflow cần classifier mapping;
5. build gallery version mới, không ghi đè gallery lịch sử;
6. đánh giá trên validation query sau khi loại self/hash/family leakage;
7. chỉ calibration threshold bằng validation;
8. cập nhật backend config atomically sang catalog/mapping/gallery/checkpoint tương thích.

### Có cần retrain embedding không?

Không có câu trả lời mặc định cho mọi item:

- Có thể thử checkpoint hiện hành như feature extractor trên gallery/item mới.
- Chỉ được reuse khi validation theo item mới đạt yêu cầu đã thống nhất.
- Nếu item mới khó phân biệt, domain khác hoặc checkpoint class/order phải thay đổi, cần retrain/fine-tune và tạo checkpoint/mapping/gallery version mới.
- Không dùng test để quyết định có retrain hay chọn threshold.

Tạo product profile mới dưới `configs/products/`, ví dụ:

```yaml
schema_version: 1
name: ponds2
brand: PONDS
classes:
  - class_index: 0
    legacy_class_name: PONDS_Item_A
    item_code: "00123456"
  - class_index: 1
    legacy_class_name: PONDS_Item_B
    item_code: "00123457"
identity_guards: {}
```

Mapping generic chọn profile tường minh và dry-run trước:

```powershell
python scripts\build_product_mapping.py `
  --catalog artifacts\catalog\ponds2_v1\product_catalog.json `
  --checkpoint checkpoints\ponds2\run_name\best_model.pth `
  --output artifacts\mappings\ponds2_mapping.v1.json `
  --product-profile ponds2 `
  --dry-run
```

Chỉ bỏ `--dry-run` khi output path mới, catalog/checkpoint/profile đều đã review. CLI từ chối ghi đè mapping đã tồn tại. `migrate_axe_mapping.py` là compatibility entry point của cùng implementation; AXE v3 payload và Blue Lavender guard vẫn được giữ byte-compatible.

Product-profile selection làm mapping cấu hình-driven, nhưng gallery/item mới vẫn chỉ được gọi là hỗ trợ sau khi build artifact version mới và validation đạt yêu cầu.

## 4. Thêm SKU AXE vào catalog không đồng nghĩa thêm trained class

Nếu chỉ thêm record vào workbook/catalog:

- checkpoint AXE five-class vẫn chỉ có năm class cũ;
- mapping hiện hành vẫn phải giữ đúng `0..4`;
- gallery hiện hành không tự thay đổi;
- item mới không được force-map vào class gần giống;
- `64823658` spray tuyệt đối không được gộp với `64863894` body wash.

Catalog coverage, trained-class coverage và gallery coverage phải được báo cáo riêng.

## 5. Checklist review

- [ ] ID cũ và class order cũ không đổi.
- [ ] Alias mới không trùng/mơ hồ.
- [ ] Source mới đã khai báo path, format, mapping source và checksum đã review trong `configs/datasets/sources.yaml`.
- [ ] Class ID đọc từ từng export metadata.
- [ ] Geometry được khai báo tường minh.
- [ ] Conflict vẫn ở quarantine cho tới review.
- [ ] Exact-image và family cross-split overlap bằng zero.
- [ ] Historical holdout được giữ.
- [ ] Output run/artifact dùng version mới, không overwrite.
- [ ] Detector validation có per-brand metrics.
- [ ] Gallery validation có per-item coverage/Recall@1.
- [ ] Cosine không được hiển thị như probability.
- [ ] Test chỉ chạy opt-in sau khi model/threshold đã chốt bằng validation.
- [ ] Backend vẫn trả unresolved cho brand/item chưa có coverage.
