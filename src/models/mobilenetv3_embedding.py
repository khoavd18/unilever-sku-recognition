"""MobileNetV3-Large classifier with a reusable normalized embedding output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetV3EmbeddingClassifier(nn.Module):
    """ImageNet-initialized MobileNetV3-Large, projection head, and classifier."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.backbone = base.features
        self.avgpool = base.avgpool
        self.feature_dim = base.classifier[0].in_features
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self._backbone_frozen = False

    def train(self, mode: bool = True) -> "MobileNetV3EmbeddingClassifier":
        """Keep frozen backbone BatchNorm statistics fixed during head training."""

        super().train(mode)
        if mode and self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward_features(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        return torch.flatten(self.avgpool(features), 1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.forward_features(images)
        projection = self.embedding_head(features)
        embedding = F.normalize(projection, p=2, dim=1)
        logits = self.classifier(projection)
        return logits, embedding

    @torch.inference_mode()
    def extract_embedding(self, images: Tensor) -> Tensor:
        _, embedding = self(images)
        return embedding

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self._backbone_frozen = True
        self.backbone.eval()

    def unfreeze_last_blocks(self, num_blocks: int | None = None) -> None:
        """Unfreeze final MobileNet feature blocks; None unfreezes the full backbone."""
        blocks = list(self.backbone.children())
        if num_blocks is None or num_blocks >= len(blocks):
            selected = blocks
        elif num_blocks <= 0:
            selected = []
        else:
            selected = blocks[-num_blocks:]
        for block in selected:
            for parameter in block.parameters():
                parameter.requires_grad = True
        self._backbone_frozen = not any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )


def model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[MobileNetV3EmbeddingClassifier, dict[str, Any]]:
    """Rebuild a model using checkpoint metadata, without needing the original YAML file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    model_config = config["model"]
    model = MobileNetV3EmbeddingClassifier(
        num_classes=len(class_to_idx),
        embedding_dim=int(model_config["embedding_dim"]),
        dropout=float(model_config.get("dropout", 0.2)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    checkpoint["class_to_idx"] = {str(name): int(index) for name, index in class_to_idx.items()}
    return model, checkpoint
