"""Versioned preprocessing metadata shared by gallery build and recognition."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def embedding_preprocessing(image_size: int) -> dict[str, Any]:
    """Describe the exact MobileNet input preprocessing used by this project."""

    size = int(image_size)
    if size <= 0:
        raise ValueError("image_size must be positive")
    return {
        "image_size": size,
        "resize": "fit_with_black_pad_no_stretch",
        "normalization": "ImageNet mean/std",
        "training_augmentation": False,
    }


def preprocessing_version(payload: dict[str, Any]) -> str:
    """Return the stable version identifier stored in gallery manifests."""

    # Preserve the historical v1 serialization exactly; active galleries were
    # fingerprinted with json.dumps(..., sort_keys=True)'s default separators.
    canonical = json.dumps(payload, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedding_preprocessing_version(image_size: int) -> str:
    return preprocessing_version(embedding_preprocessing(image_size))
