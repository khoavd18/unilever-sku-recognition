"""Optional batch-hard triplet loss for later metric-learning experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BatchHardTripletLoss(nn.Module):
    """Hardest positive / hardest negative triplet loss on normalized embeddings."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2 or labels.ndim != 1:
            raise ValueError("embeddings must be [B, D] and labels must be [B]")
        similarities = embeddings @ embeddings.T
        distances = 1.0 - similarities
        same_class = labels[:, None].eq(labels[None, :])
        identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = same_class & ~identity
        negative_mask = ~same_class
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0
        hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        return F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin).mean()

