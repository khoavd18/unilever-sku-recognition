"""Small CPU-only ONNX Runtime adapter for exported YOLOv8 detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OnnxYoloDetector:
    """Run the project's static-shape YOLOv8 ONNX export without Ultralytics auto-installation."""

    task = "detect"

    def __init__(self, model_path: str | Path, names: dict[int, str]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX detector requires onnxruntime from requirements.txt") from exc
        self.names = {int(index): str(name) for index, name in names.items()}
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(Path(model_path).resolve()), options, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError("Deployment ONNX detector must have a static NCHW input shape")
        if shape[0] not in (1, "batch", None) or shape[1] != 3:
            raise ValueError(f"Unsupported ONNX detector input shape: {shape}")
        self.input_name = model_input.name
        self.height, self.width = int(shape[2]), int(shape[3])

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.width - resized_width) // 2
        pad_y = (self.height - resized_height) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0, scale, pad_x, pad_y

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        left_top = np.maximum(first[:2], second[:, :2])
        right_bottom = np.minimum(first[2:], second[:, 2:])
        intersection = np.maximum(0, right_bottom[:, 0] - left_top[:, 0]) * np.maximum(
            0, right_bottom[:, 1] - left_top[:, 1]
        )
        first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
        second_area = np.maximum(0, second[:, 2] - second[:, 0]) * np.maximum(0, second[:, 3] - second[:, 1])
        return intersection / np.maximum(first_area + second_area - intersection, 1e-9)

    def predict(self, image: np.ndarray, confidence: float, iou_threshold: float = 0.7) -> list[dict[str, Any]]:
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4 + len(self.names):
            raise ValueError(f"Unexpected YOLOv8 ONNX output shape: {output.shape}")
        rows = output[0].T
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4 + class_ids]
        keep = scores >= confidence
        rows, class_ids, scores = rows[keep], class_ids[keep], scores[keep]
        if not len(rows):
            return []
        centers = rows[:, :2]
        sizes = rows[:, 2:4]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        image_height, image_width = image.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, class_ids, scores = boxes[valid], class_ids[valid], scores[valid]
        selected: list[int] = []
        # QDQ INT8 exports can yield thousands of candidates before NMS.  The
        # product API never returns more than a few hundred detections per tile;
        # bounding each class keeps latency predictable without changing the
        # final cross-tile NMS contract.
        max_candidates_per_class = 300
        for class_id in np.unique(class_ids):
            indexes = np.where(class_ids == class_id)[0]
            if len(indexes) > max_candidates_per_class:
                indexes = indexes[np.argsort(-scores[indexes])[:max_candidates_per_class]]
            class_boxes = boxes[indexes]
            xywh = [
                [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
                for box in class_boxes
            ]
            kept = cv2.dnn.NMSBoxes(
                xywh, [float(score) for score in scores[indexes]], float(confidence), float(iou_threshold)
            )
            if len(kept):
                selected.extend(indexes[np.asarray(kept).reshape(-1)].tolist())
        return [
            {"xyxy": boxes[index].tolist(), "score": float(scores[index]), "class_id": int(class_ids[index])}
            for index in sorted(selected, key=lambda index: float(scores[index]), reverse=True)
        ]
