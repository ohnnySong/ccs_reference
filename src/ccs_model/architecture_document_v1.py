"""Document-aligned CCS model with separate opacity and contraction outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


@dataclass
class CcsDocumentOutput:
    opacity_logits: Tensor
    opacity_probabilities: Tensor
    contraction_logits: Tensor
    contraction_probabilities: Tensor
    opening_area_mm2: Tensor


class CcsDocumentMultiTaskV1(nn.Module):
    """Predict 4-grade opacity, 3-grade contraction, and WTW-scaled area."""

    def __init__(self, *, pretrained: bool = True, dropout: float = 0.25) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.encoder = convnext_tiny(weights=weights).features
        feature_width = 768 * 3 + 1
        self.opacity_head = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 4),
        )
        self.contraction_head = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
        self.area_fraction_head = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(self, image: Tensor, wtw_mm: Tensor) -> CcsDocumentOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        # Train the document-aligned heads on a frozen CCS image encoder.
        with torch.no_grad():
            feature = self.encoder(image)
        global_mean = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        global_rms = F.adaptive_avg_pool2d(feature.square(), 1).sqrt().flatten(1)
        height, width = feature.shape[-2:]
        center = feature[
            :, :, height // 4 : max(height // 4 + 1, 3 * height // 4),
            width // 4 : max(width // 4 + 1, 3 * width // 4),
        ]
        center_mean = F.adaptive_avg_pool2d(center, 1).flatten(1)
        physical_wtw = wtw_mm.to(image).reshape(image.shape[0]).clamp(8.0, 15.0)
        fused = torch.cat(
            (global_mean, global_rms, center_mean, (physical_wtw / 11.3).unsqueeze(1)),
            dim=1,
        )
        opacity_logits = self.opacity_head(fused)
        contraction_logits = self.contraction_head(fused)
        area_fraction = torch.sigmoid(self.area_fraction_head(fused).squeeze(1))
        return CcsDocumentOutput(
            opacity_logits=opacity_logits,
            opacity_probabilities=torch.softmax(opacity_logits, dim=1),
            contraction_logits=contraction_logits,
            contraction_probabilities=torch.softmax(contraction_logits, dim=1),
            opening_area_mm2=area_fraction * physical_wtw.square(),
        )
