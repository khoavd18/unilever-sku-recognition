# Runtime architecture

```text
2–6 ordered shelf images
  → stitch panorama and quality gate
  → tiled YOLOv8 INT8 ONNX Runtime detector
  → bbox + one of 21 brands
  → crop product
  → MobileNetV3-Large, 256-D L2-normalized embedding
  → full cosine search of the 459-SKU gallery
  → SKU candidate / resolved result after calibrated threshold
```

The backend loads every asset from private `deployment/assets`. It refuses to
start recognition if the checkpoint, scoped catalog, mapping, gallery dimension
or preprocessing fingerprints do not agree.

## Model 1: brand detector

`deployment/assets/models/yolov8_brand21_int8.onnx` is a static-shape YOLOv8
QDQ INT8 export. `src.recognition.onnx_detector` runs it directly through ONNX
Runtime's CPU provider, using square letterbox preprocessing and class-aware
NMS. The class order in `configs/panorama/backend.yaml` is part of the model
contract and must not be reordered.

## Model 2: SKU retrieval

`deployment/assets/models/mobilenetv3_sku459.pth` produces 256-dimensional
normalized embeddings. The deployment gallery contains 9,134 representative
training embeddings for 459 exact-mapped SKUs across 21 brands. Search is
full-gallery; the detector's brand is reported but is not used to filter
candidate SKUs.

The gallery is a retrieval index, so cosine similarity is not a probability.
The current backend returns the nearest SKU as a candidate until a threshold is
calibrated on validation data and explicitly enabled in config.

## Contracts and reproducibility

- `deployment/assets/catalog/product_catalog.json` scopes the catalog to the
  exact 459 classes in the checkpoint.
- `deployment/assets/mapping.json` is a one-to-one, exact-label mapping from
  checkpoint class index to `item_code`.
- `deployment/assets/gallery/gallery_manifest.json` fingerprints model,
  catalog, mapping, preprocessing and embedding dimension.
- `scripts/build_runtime_assets.py` creates a fresh versioned catalog, mapping
  and gallery. It does not modify source data, models or existing output.

Raw datasets, models, catalogs, gallery embeddings and training output remain
outside Git. Provision the approved deployment package through private storage
before running the API, mobile app or Docker image.
