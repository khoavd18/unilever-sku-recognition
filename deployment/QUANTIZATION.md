# YOLOv8 INT8 provenance

`assets/models/yolov8_brand21_int8.onnx` is the QDQ INT8 export of the
21-brand YOLOv8 baseline. It uses per-channel `QInt8` weights and `QUInt8`
activations; the Detect decode/DFL output operations remain FP32.

The checked export used 200 training images for calibration and 100 validation
images for parity. PyTorch versus ONNX FP32 passed the export-parity contract:
maximum box difference `0.004913 px` (limit `0.05 px`) and maximum score
difference `0.000011` (limit `0.0001`).

On the mapped SKU469 validation corpus, the ONNX INT8 model had `mAP50-95
0.2796` versus ONNX FP32 `0.2753`; its CPU p50 inference latency was `30.02 ms`
versus `49.35 ms` for ONNX FP32. These metrics describe the packaged detector
and should be remeasured whenever its model, calibration data or export options
change.
