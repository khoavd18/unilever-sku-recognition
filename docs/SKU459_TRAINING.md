# 459-SKU embedding model

The runtime uses `MobileNetV3-Large` with a 256-dimensional L2-normalized
embedding and a 459-class classifier head. Its selected checkpoint is packaged
at `deployment/assets/models/mobilenetv3_sku459.pth`.

The prepared dataset contains 459 exactly mapped item codes across 21 brands.
Training/validation/test groups and exact file hashes are split independently;
the test split is never used to choose a checkpoint or retrieval threshold.

## Train a new run

```powershell
python scripts\train.py `
  --config configs\sku459_mobilenetv3.yaml `
  --run-name <new-run-name>
```

After selecting a checkpoint on validation data, build a new deployment package:

```powershell
python scripts\build_runtime_assets.py `
  --checkpoint checkpoints\sku459_mobilenetv3\<new-run-name>\best_model.pth `
  --full-catalog artifacts\catalog\v1\product_catalog.json `
  --dataset-dir data\sku469_v33_training_v1 `
  --manifest data\sku469_v33_training_v1\manifests\train.csv `
  --output-root runtime_assets\sku459_<new-run-name>
```

The builder scopes the catalog by exact legacy class label, writes a
checkpoint-fingerprinted mapping, and produces a bounded but source-family
diverse gallery. It refuses to overwrite prior artifact versions.

Before enabling automatic resolution, evaluate retrieval on `val.csv`, choose
a minimum similarity threshold, and set both `minimum_similarity` and
`similarity_threshold_calibrated: true` in `configs/panorama/backend.yaml`.
