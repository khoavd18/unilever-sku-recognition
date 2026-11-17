# Runtime assets

This folder is the runtime package location. Its `assets/` directory is private
and ignored by Git; provision it from approved private storage before running
the service.

- `assets/models/yolov8_brand21_int8.onnx`: 21-brand YOLOv8 INT8 detector.
- `assets/models/mobilenetv3_sku459.pth`: MobileNetV3 embedding checkpoint.
- `assets/catalog`: exact 459-SKU scoped catalog.
- `assets/mapping.json`: checkpoint-to-SKU mapping verified by exact legacy label.
- `assets/gallery`: 9,134 L2-normalized embeddings, up to 20 diverse train images per SKU.

Regenerate this package only from the selected checkpoint and the prepared source
dataset. The builder creates a new output directory and refuses to overwrite:

```powershell
python scripts\build_runtime_assets.py `
  --checkpoint checkpoints\sku459_mobilenetv3\baseline_s42\best_model.pth `
  --full-catalog artifacts\catalog\v1\product_catalog.json `
  --dataset-dir data\sku469_v33_training_v1 `
  --manifest data\sku469_v33_training_v1\manifests\train.csv `
  --output-root runtime_assets\sku459_yolov8int8_v4
```
