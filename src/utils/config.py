"""Configuration, reproducibility, and path helpers."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.config.paths import PROJECT_ROOT, resolve_config_path, resolve_repo_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and make all required top-level sections available."""
    path = resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("dataset", "model", "training", "metric_learning", "output"):
        config.setdefault(section, {})
    config["_config_path"] = str(path.resolve())
    return config


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy config before putting it in a checkpoint or changing it for a run."""
    return copy.deepcopy(config)


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Backward-compatible facade over repository-root path resolution."""

    return resolve_repo_path(value, root)


def resolve_device(requested: str = "auto") -> torch.device:
    """Choose CUDA automatically, while retaining an explicit CPU override."""
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Set deterministic-enough seeds for repeatable dataset splits and runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

