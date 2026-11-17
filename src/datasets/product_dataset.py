"""Manifest-based dataset code. It never moves or modifies source images."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


class ResizeWithPad:
    """Fit an image into a square without stretching, using a constant background."""

    def __init__(self, image_size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.image_size = image_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Cannot resize an image with an empty dimension")
        scale = min(self.image_size / width, self.image_size / height)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.fill)
        canvas.paste(resized, ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_size={self.image_size}, fill={self.fill})"


def build_transforms(
    image_size: int,
    train: bool,
    horizontal_flip: bool = False,
    rotation_degrees: float = 5.0,
    brightness: float = 0.12,
    contrast: float = 0.12,
    saturation: float = 0.08,
    hue: float = 0.02,
) -> transforms.Compose:
    """Preserve crop aspect ratio; flips are opt-in because they reverse package text."""
    operations: list = [ResizeWithPad(image_size)]
    if train:
        if horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        operations.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )
        if rotation_degrees > 0:
            operations.append(transforms.RandomRotation(degrees=rotation_degrees, fill=(0, 0, 0)))
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest was not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "class_name", "split"}
    if not rows:
        raise ValueError(f"Manifest has no images: {path}")
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest {path} must contain columns: {sorted(required)}")
    return rows


class ProductDataset(Dataset):
    """Loads source images listed in a split CSV, rather than copying image files."""

    def __init__(
        self,
        manifest_path: str | Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.samples = read_manifest(self.manifest_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
        unknown = sorted({row["class_name"] for row in self.samples} - set(class_to_idx))
        if unknown:
            raise ValueError(f"Classes in {self.manifest_path} missing from class_to_idx: {unknown[:5]}")
        missing = [row["image_path"] for row in self.samples if not Path(row["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image paths in {self.manifest_path} do not exist. First: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image) if self.transform else image
        return image, self.class_to_idx[row["class_name"]], row["image_path"], row["class_name"]


class ClassBalancedBatchSampler(Sampler[list[int]]):
    """Sample at least two instances per class so batch-hard triplet loss is valid."""

    def __init__(self, dataset: ProductDataset, batch_size: int, samples_per_class: int, seed: int = 42) -> None:
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2 for triplet loss")
        if batch_size < samples_per_class or batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class")
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.samples):
            grouped[dataset.class_to_idx[row["class_name"]]].append(index)
        self.grouped = {label: indexes for label, indexes in grouped.items() if len(indexes) >= samples_per_class}
        if len(self.grouped) < 2:
            raise ValueError(
                "Metric learning needs at least two classes with two or more train images each. "
                "Disable metric_learning or add more images."
            )
        self.classes_per_batch = batch_size // samples_per_class
        self.batches_per_epoch = max(1, len(dataset) // batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        labels = list(self.grouped)
        for _ in range(self.batches_per_epoch):
            # Start with distinct labels: every anchor then has a negative class.
            chosen = rng.sample(labels, min(self.classes_per_batch, len(labels)))
            chosen.extend(rng.choice(labels) for _ in range(self.classes_per_batch - len(chosen)))
            batch: list[int] = []
            for label in chosen:
                batch.extend(rng.sample(self.grouped[label], self.samples_per_class))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
