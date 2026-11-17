# Báo cáo refactor và cleanup

Ngày thực hiện: 2026-10-11  
Repository: `D:\code\anhoa\train_embedding`

## 1. Kết quả

Refactor đã tách các contract dùng chung cho configuration, path, annotation parsing, crop geometry, detector training, product identity và gallery preprocessing. CLI cũ được giữ để không làm vỡ lệnh lịch sử; notebook mới gọi lại cùng module/CLI thay vì chép training loop hoặc metric implementation.

Phạm vi được giữ đúng với hệ thống hiện có:

- Model 1 là detector chín brand với class order ổn định `0..8`;
- Model 2/runtime gallery hiện chỉ phủ năm item AXE;
- brand thiếu gallery tiếp tục ở trạng thái unresolved;
- backend/mobile response và capture workflow được giữ tương thích;
- không tuyên bố hỗ trợ mọi dataset/model tương lai.

Không có training, tuning, dataset rebuild, gallery generation, full-dataset inference hoặc test-split evaluation nào được khởi động trong đợt refactor này.

## 2. Backup và trạng thái version control

Workspace không có Git metadata. Trước khi xóa source/config/documentation, một backup ZIP đã được tạo ngoài repository:

```text
D:\code\anhoa\train_embedding_refactor_backup_20261011_101705.zip
```

Backup chỉ chứa phạm vi source/config/documentation bị ảnh hưởng; không dùng một archive folder trong project để giữ code obsolete.

## 3. Tổ chức sau refactor

| Khu vực | Vai trò |
|---|---|
| `configs/brands.yaml` | Brand registry, alias và detector profiles |
| `configs/datasets/sources.yaml` | Source path, annotation format, class-mapping source và checksum |
| `configs/products/axe5.yaml` | AXE checkpoint class/item identity và guard |
| `src/config/` | Validated registry và repository/config-relative paths |
| `src/data/` | Explicit annotation formats, geometry và deterministic crop names |
| `src/models/` | MobileNetV3 model/checkpoint loading |
| `src/training/` | `embedding_workflow`, embedding data/trainers và shared YOLO workflow |
| `src/evaluation/` | Metric primitives và shared gallery evaluation workflow |
| `src/catalog/` | Catalog import/validation, product profiles và mapping |
| `src/retrieval/` | Gallery validation, preprocessing compatibility và search |
| `src/stitching/` | Stitch quality gates và ordered sequence |
| `src/recognition/` | Detector contract, tiling, crop, embedding và retrieval |
| `src/osa/` | Visible-count/assortment/correction provenance |
| `scripts/` | Thin CLIs và compatibility wrappers |
| `notebooks/` | Audit, training, metrics, gallery và inference UI an toàn |
| `docs/` | Kiến trúc, adding-a-brand và handoff report |

## 4. Thay đổi chính

### Configuration và identity

- `BrandRegistry` đọc `configs/brands.yaml`, giữ ID ổn định, kiểm tra alias trùng/mơ hồ và profile liên tục từ zero.
- Alias `CLEARMEN -> CLEAR` được khai báo bằng config thay vì copy logic brand-specific.
- `configs/datasets/sources.yaml` khai báo source export, geometry, class-mapping source và checksum; undeclared source/checksum drift bị từ chối.
- `configs/products/axe5.yaml` giữ mapping class `0..4` sang item code dạng string.
- Identity guard phân biệt `64823658` Blue Lavender spray với `64863894` Blue Lavender body wash.
- Path helper resolve theo repository/config root, không theo current working directory ngẫu nhiên.
- Synthetic additional-brand fixture chứng minh registry và generic detector preflight mở rộng bằng config/data mà không sửa brand-specific Python. Fixture không chứng minh detector đã train nhận diện brand đó.

### Data parsing và crop

- Parser dùng enum format tường minh cho COCO bbox, YOLO bbox, YOLO OBB và YOLO segmentation.
- Không còn coi số coordinate là đủ để đoán geometry.
- YOLO class order đọc từ `data.yaml`; COCO category đọc từ document nguồn.
- Shared crop helper giữ clipping/padding và tên file ngắn, deterministic, collision-resistant cho Windows; provenance đầy đủ vẫn nằm trong manifest.
- Existing split, exact-image/family leakage, quarantine và historical holdout policies được giữ.

### Training và evaluation

- `src.training.yolo` gom config loading, path resolution, preflight, dataset mapping checks, new-run guard, Ultralytics version guard và final `YOLO.train` call.
- `scripts/train_detector.py` là generic CLI cho prepared detector dataset có metadata hợp lệ.
- `scripts/prepare_detector_dataset.py --brand-profile` chọn class-order profile và dùng source registry; tên `prepare_brand_9brands_yolo.py` tiếp tục tương thích.
- `src.training.embedding_workflow` chứa generic MobileNet planning/training; `scripts/train.py` forward vào module và import không train.
- `src.evaluation.gallery` chứa AXE train-gallery/validation-query implementation; `scripts/test_axe_gallery.py` forward vào module và chỉ hỗ trợ validation query.
- Import/preflight không import Ultralytics và không train.
- Classification, detector và retrieval metrics vẫn tách riêng; test không được dùng để chọn checkpoint/threshold.

### Catalog, mapping và gallery

- Product profile và preprocessing fingerprint có nguồn canonical.
- Gallery/runtime compatibility fail closed khi model/catalog/mapping/preprocessing/dimension khác nhau.
- Existing v3 mapping được rebuild trong validation và cho payload/mapping version giống hệt lịch sử: `sha256:c7d08f71c6022d70e928f779ce8ba8e3d13f8ca88edc29f0757bd89406148ccf`.
- `scripts/build_product_mapping.py --product-profile` là generic mapping CLI; `migrate_axe_mapping.py` giữ compatibility và cùng dry-run/no-overwrite guards.
- Full-gallery cosine search, leakage exclusions và “cosine không phải probability” được giữ.

### Documentation và notebooks

- README được viết lại theo kiến trúc runtime hiện tại; bỏ claim cũ rằng `data/product_dataset` chưa tồn tại.
- Nội dung panorama handoff hữu ích được hợp nhất vào `ARCHITECTURE.md`, `ADDING_A_BRAND.md`, README và báo cáo này.
- Năm notebook tiếng Việt dùng lại module/CLI hiện có, output được clear và mọi cờ thực thi mặc định `False`.
- Validation là evaluation split mặc định; test có cờ opt-in riêng.
- Dependency notebook nằm trong `requirements-notebooks.txt`, không có cell cài package.

## 5. Compatibility wrappers giữ lại

Toàn bộ `scripts/*.py` được giữ vì đây là các direct CLI entry point của repository và một số tên đã xuất hiện trong README cũ, automation hoặc tests. Đặc biệt:

- dataset/crop: `build_clean_axe_dataset.py`, `prepare_coco_crops.py`, `prepare_axe_yolo.py`, `prepare_brand_5brands_yolo.py`, `prepare_brand_9brands_yolo.py`;
- detector: `train_axe_yolo.py`, `train_brand_5brands_yolo.py`, `train_brand_9brands_yolo.py`, `train_brand_9brands_yolo_kd.py`;
- embedding: `train.py`, `train_axe_5sku.py`, `evaluate.py`, `evaluate_axe_5sku.py`, `evaluate_axe_5sku_classification.py`;
- gallery/runtime: `build_embeddings.py`, `inference.py`, `build_item_gallery.py`, `evaluate_item_retrieval.py`, `test_axe_gallery.py`;
- catalog/stitch/export: `import_product_catalog.py`, `migrate_axe_mapping.py`, `stitch_two_images.py`, `export_model1_tflite.py`.

Entry point generic mới là `prepare_detector_dataset.py`, `train_detector.py` và `build_product_mapping.py`. Chúng không xóa các tên lịch sử; wrapper cũ forward vào implementation dùng chung.

Các script Rexona xử lý hai input contract khác nhau (YOLO directory và COCO ZIP/folder), nên không bị xóa chỉ vì tên gần giống. Logic mới nên đi qua shared parser khi được migrate, nhưng external entry point chưa được chứng minh là bỏ được.

## 6. Cleanup

Không xóa dataset, source ZIP, annotation gốc, split, checkpoint, catalog, mapping, gallery, metrics, training report, runtime session, pretrained weights, workbook hoặc `mobile/node_modules`.

| Đối tượng | Lý do | Trạng thái bàn giao |
|---|---|---|
| Python `__pycache__`/`*.pyc` trong source/tests/scripts/backend | Bytecode tái tạo được; gồm orphan bytecode của source đã không còn | Đã xóa; kiểm tra cuối không còn cache/bytecode |
| `.ruff_cache/` | Cache cục bộ do static analysis tạo lại được; đã có ignore rule | Đã xóa |
| Bảy YOLO `labels/*.cache` | Label index cache do Ultralytics tái tạo từ labels/data.yaml | Đã xóa đúng bảy file đã xác minh |
| `mobile/.expo/` | Expo local state/logs, đã ignore | Đã xóa |
| `mobile/android/.gradle/`, `mobile/android/build/` | Gradle build/cache output, đã ignore | Đã xóa |
| `outputs/mobile_capture_ui_bundle_smoke/` | Hermes/Expo bundle chỉ sinh cho smoke validation, tái tạo được và không phải historical artifact | Đã xóa |
| File-empty `outputs/axe_crop_smoke/`, `runs/detect/` | Chỉ còn directory rỗng, không chứa artifact | Đã xóa |
| `docs/PANORAMA_MVP_VI.md` | Nội dung còn đúng đã hợp nhất; file cũ có claim detector năm brand mâu thuẫn contract chín brand | Đã xóa sau khi hợp nhất |

Historical `outputs/smoke_test/` dù tái tạo được vẫn được giữ vì chứa checkpoint/metrics lịch sử. `runtime/sessions/` được giữ như capture/stitch evidence.

## 7. `.gitignore` và report sharing

Ignore rules đã bao phủ Python/notebook caches, local environments/secrets, datasets/runtime sessions, generated outputs, runs/model binaries, mobile/native build output, coverage/packaging, IDE và OS files.

`outputs/share/` là đường dẫn opt-in để chia sẻ report nhẹ đã review với mentor. Chỉ Markdown, JSON, CSV và PNG dưới thư mục đó được phép re-include; không copy dataset, credential, checkpoint hoặc model binary vào đó. Ignore rule không xóa hoặc thay đổi historical artifact đang tồn tại.

## 8. Validation thực tế

### Baseline trước refactor

| Lệnh/check | Kết quả |
|---|---|
| `python -m unittest discover -s tests -v` | 92/92 pass |
| `pytest` | Không có trong environment; dùng `unittest` là test runner hiện hành |
| Bốn mobile Node test scripts | 17/17 pass |
| `npm run typecheck` | Pass |
| `npm test` | Chưa tồn tại ở baseline; aggregate script được thêm trong refactor |

Không có baseline unit-test failure. Trong quá trình refactor từng xuất hiện một regression Windows ở decoded-megapixel test: expected HTTP 413 nhưng nhận 400 do temporary upload bị unlink khi Pillow còn giữ handle. Lỗi được sửa bằng lifecycle đóng handle đúng cách; focused backend upload suite sau sửa đạt 3/3.

### Focused validation trong refactor

| Phạm vi | Kết quả |
|---|---|
| Registry/product profile tests | 9/9 pass |
| Affected configuration/compatibility suite | 49/49 pass |
| Existing v3 mapping payload/version comparison | Exact match |
| Focused backend upload contract | 3/3 pass |
| Bốn SHA-256 của brand source ZIP đã khai báo | 4/4 khớp file nguồn |

### Validation tích hợp cuối

| Lệnh/check | Kết quả |
|---|---|
| `python -m unittest discover -s tests -v` | 131/131 pass |
| Ruff trên Python source, scripts, backend và tests | Pass |
| Compile toàn bộ Python source, scripts, backend và tests | Pass |
| Cấu trúc/cú pháp của năm notebook | 5/5 pass; output đã clear, 18/18 cờ `RUN_*` là `False` |
| `Set-Location mobile; npm test` | 17/17 pass |
| `Set-Location mobile; npm run typecheck` | Pass |
| Detector/embedding preflight và khởi tạo runtime trên CPU | Pass; không train, không predict, không load test query và không tạo run directory |
| AXE mapping dry-run với `--product-profile axe5` | `verified`; existing output khớp; không ghi file |

Các check tích hợp trên không chạy training, gallery generation, full-dataset inference, checkpoint evaluation hay test-split evaluation.

## 9. Cách chạy tiếp

Các lệnh read-only nên chạy trước:

```powershell
Set-Location D:\code\anhoa\train_embedding
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python scripts\prepare_detector_dataset.py --brand-profile brand_9brands --output-dir data\brand_9brands_yolo_reviewed --check-only
python scripts\train_detector.py --config configs\brand_9brands_yolo.yaml --preflight
python scripts\import_product_catalog.py --workbook ITEMCODE_SKU.xlsx --validate-only
python scripts\build_product_mapping.py --product-profile axe5 --dry-run

Set-Location mobile
npm test
npm run typecheck
```

Mở notebook:

```powershell
Set-Location D:\code\anhoa\train_embedding
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

Training, gallery build và test evaluation vẫn là quyết định thủ công riêng; không phải “next command” mặc định của refactor.

## 10. Giới hạn còn lại

- Gallery/catalog đang chỉ phủ AXE five-item; tám detector brand còn lại unresolved.
- Similarity threshold chưa calibration, nên candidate không tự trở thành resolved item code.
- Dataset source registry/preparer chỉ hỗ trợ các annotation format và class-mapping source được khai báo; nó không tự động hiểu mọi export tương lai.
- Generic product-profile mapping đã có, nhưng non-AXE catalog/gallery/checkpoint vẫn cần artifact version mới và validation riêng trước khi gọi là end-to-end supported.
- Filename-family grouping không chứng minh capture-session independence.
- Stitching là chuỗi homography có quality gates, không phải global bundle adjustment; ảnh kệ thật/parallax vẫn cần manual field validation.
- Mobile physical-device/LAN/native-build behavior cần acceptance test trên thiết bị mục tiêu.
- Recognition hiện chạy trên backend; local native stitch không đồng nghĩa inference offline/on-device.
- OSA chỉ là model-visible observation khi chưa có expected assortment và coverage complete.
