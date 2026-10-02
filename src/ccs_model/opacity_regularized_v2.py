"""Regularized four-class anterior-capsule opacity specialist."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


class OpacityRegularizedV2(nn.Module):
    def __init__(self, *, dropout: float = 0.55, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = convnext_tiny(weights=None).features
        self.head = nn.Sequential(
            nn.LayerNorm(768 * 3),
            nn.Dropout(dropout),
            nn.Linear(768 * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def extract_features(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            feature = self.encoder(image)
            mean = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            rms = F.adaptive_avg_pool2d(feature.square(), 1).sqrt().flatten(1)
            height, width = feature.shape[-2:]
            center = feature[
                :, :, height // 4 : max(height // 4 + 1, 3 * height // 4),
                width // 4 : max(width // 4 + 1, 3 * width // 4),
            ]
            center_mean = F.adaptive_avg_pool2d(center, 1).flatten(1)
        return torch.cat((mean, rms, center_mean), dim=1)

    def forward_features(self, feature: Tensor) -> Tensor:
        return self.head(feature)

    def forward(self, image: Tensor) -> Tensor:
        return self.forward_features(self.extract_features(image))
