# Unilever SKU Recognition

Ứng dụng nhận diện sản phẩm trên kệ theo hai tầng:

```text
2–6 ảnh kệ → ghép panorama → YOLOv8 INT8 (bbox + brand)
→ MobileNetV3 embedding → cosine search gallery → SKU candidate
```

Repository chỉ chứa source code và contract runtime. Các artifact inference được
giữ ở private storage, không được commit vào GitHub:

- YOLOv8 INT8 ONNX, 21 brand;
- MobileNetV3-Large embedding 256 chiều;
- gallery, catalog và mapping 459 SKU;
- FastAPI backend và app Expo để chụp, ghép ảnh và xem kết quả.

Candidate SKU chưa tự chuyển thành `resolved`: ngưỡng cosine cần được chọn từ
validation độc lập trước khi bật `similarity_threshold_calibrated`.

## Cấu trúc

```text
backend/                 FastAPI API
configs/                 contract runtime và training
deployment/assets/       model, catalog, mapping và gallery được versioned
mobile/                  Expo/React Native app
scripts/                 CLI build, evaluation và training
src/                     domain code dùng chung
tests/                   unit và contract tests
```

`data/`, `runs/`, `outputs/`, `checkpoints/`, `runtime/`, `test_images/` và
`deployment/assets/` là dữ liệu local/private, không nằm trong Git. Cần cấp
private deployment package vào `deployment/assets/` trước khi chạy inference.

## Chạy backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Kiểm tra `http://127.0.0.1:8000/api/v1/health`. Backend sẽ báo `degraded` nếu
chưa có private deployment assets; sau khi được provision, model được load ở
request nhận diện đầu tiên. ONNX detector luôn chạy CPU; embedding dùng CUDA nếu có.

## Chạy app điện thoại

```powershell
Set-Location mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL = "http://<IP-LAN-MAY-TINH>:8000"
npm start
```

Điện thoại phải gọi IP LAN của máy chạy backend, không dùng `localhost`.

## Docker

```powershell
docker build -t unilever-sku-recognition .
docker run --rm -p 8000:8000 unilever-sku-recognition
```

## Kiểm tra

```powershell
python -m unittest discover -s tests -v
Set-Location mobile
npm test
npm run typecheck
```

## Tái tạo deployment assets

Khi chọn checkpoint mới, dùng [deployment/README.md](deployment/README.md).
Builder tạo catalog scope theo checkpoint, exact mapping và gallery mới; không
ghi đè artifact cũ hoặc dữ liệu nguồn.
