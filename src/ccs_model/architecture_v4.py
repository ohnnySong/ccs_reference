"""CCS v4 architecture: severity grading plus physical opening measurement.

The network predicts the capsulorhexis opening mask and two horizontal WTW
endpoint heatmaps. Physical area is derived from those geometric outputs rather
than from an unconstrained scalar regression head. The fibrosis classifier uses
global and opening-boundary features and emits light/moderate/severe logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


SEVERITY_NAMES = ("light", "moderate", "severe")


def remap_source_grade(source_grade: Tensor) -> Tensor:
    """Map historical grade 0/1/2/3 to light/moderate/severe = 0/1/2."""

    if torch.any((source_grade < 0) | (source_grade > 3)):
        raise ValueError("historical CCS grade must be in [0, 3]")
    return torch.where(source_grade <= 1, 0, source_grade - 1).long()


@dataclass
class CcsOutput:
    severity_logits: Tensor
    severity_probabilities: Tensor
    collapsed_probability: Tensor
    binary_logit: Tensor
    binary_probability: Tensor
    opening_mask_logits: Tensor
    opening_mask_probability: Tensor
    wtw_endpoint_heatmaps: Tensor
    wtw_endpoints_xy: Tensor
    wtw_diameter_px: Tensor
    area_scale_diameter_px: Tensor
    opening_area_mm2: Tensor
    opening_area_regression_mm2: Tensor
    opening_area_geometric_mm2: Tensor
    measurement_confidence: Tensor


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class _FpnDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], width: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.ModuleList(_ConvNormAct(width, width) for _ in channels)

    def forward(self, features: list[Tensor]) -> Tensor:
        pyramid: Tensor | None = None
        for feature, lateral, smooth in zip(
            reversed(features), reversed(self.lateral), reversed(self.smooth), strict=True
        ):
            current = lateral(feature)
            if pyramid is not None:
                pyramid = F.interpolate(
                    pyramid, size=current.shape[-2:], mode="bilinear", align_corners=False
                )
                current = current + pyramid
            pyramid = smooth(current)
        assert pyramid is not None
        return pyramid


class CcsMultiTaskV4(nn.Module):
    """Shared encoder with geometry-aware measurement and severity heads."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        decoder_width: int = 128,
        head_dropout: float = 0.25,
        default_wtw_mm: float = 11.3,
        softargmax_temperature: float = 0.08,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        network = convnext_tiny(weights=weights)
        self.encoder = network.features
        self.feature_indices = (1, 3, 5, 7)
        self.decoder = _FpnDecoder((96, 192, 384, 768), decoder_width)
        self.opening_head = nn.Conv2d(decoder_width, 1, 1)
        self.wtw_endpoint_head = nn.Conv2d(decoder_width, 2, 1)
        self.severity_head = nn.Sequential(
            nn.LayerNorm(768 * 2 + 3),
            nn.Linear(768 * 2 + 3, 384),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(384, len(SEVERITY_NAMES)),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(768 * 2 + 3),
            nn.Linear(768 * 2 + 3, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 1),
        )
        self.area_fraction_head = nn.Sequential(
            nn.LayerNorm(768 + 1),
            nn.Linear(768 + 1, 192),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(192, 1),
        )
        self.default_wtw_mm = float(default_wtw_mm)
        self.softargmax_temperature = float(softargmax_temperature)

    def _encode(self, image: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        x = image
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index in self.feature_indices:
                features.append(x)
        return features

    def _softargmax_xy(self, heatmaps: Tensor) -> tuple[Tensor, Tensor]:
        batch, points, height, width = heatmaps.shape
        weights = torch.softmax(
            heatmaps.reshape(batch, points, -1) / self.softargmax_temperature,
            dim=-1,
        )
        ys, xs = torch.meshgrid(
            torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
            torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1)
        xy = weights @ coordinates
        peak_probability = weights.amax(dim=-1)
        return xy, peak_probability

    @staticmethod
    def _boundary_pool(feature: Tensor, opening_probability: Tensor) -> Tensor:
        mask = F.interpolate(
            opening_probability, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )
        dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp_min(0.0)
        numerator = (feature * boundary).sum(dim=(-2, -1))
        denominator = boundary.sum(dim=(-2, -1)).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        image: Tensor,
        wtw_mm: Tensor | None = None,
        wtw_diameter_px: Tensor | None = None,
        *,
        compute_geometry: bool = True,
    ) -> CcsOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        batch, _, image_height, image_width = image.shape
        features = self._encode(image)
        deepest = features[-1]
        if compute_geometry:
            decoded = self.decoder(features)
            opening_logits = F.interpolate(
                self.opening_head(decoded),
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )
            endpoint_heatmaps = F.interpolate(
                self.wtw_endpoint_head(decoded),
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )
            opening_probability = torch.sigmoid(opening_logits)
            endpoint_xy, endpoint_confidence = self._softargmax_xy(endpoint_heatmaps)
            predicted_diameter = torch.linalg.vector_norm(
                endpoint_xy[:, 0] - endpoint_xy[:, 1], dim=1
            )
            local_feature = self._boundary_pool(deepest, opening_probability)
        else:
            opening_logits = image.new_zeros((batch, 1, image_height, image_width))
            endpoint_heatmaps = image.new_zeros((batch, 2, image_height, image_width))
            opening_probability = image.new_zeros((batch, 1, image_height, image_width))
            endpoint_xy = image.new_zeros((batch, 2, 2))
            endpoint_confidence = image.new_zeros((batch, 2))
            predicted_diameter = image.new_full(
                (batch,), 0.8 * float(max(image_height, image_width))
            )
            height, width = deepest.shape[-2:]
            center = deepest[
                :,
                :,
                height // 4 : max(height // 4 + 1, 3 * height // 4),
                width // 4 : max(width // 4 + 1, 3 * width // 4),
            ]
            local_feature = F.adaptive_avg_pool2d(center, 1).flatten(1)

        if wtw_diameter_px is None:
            scale_diameter = predicted_diameter
        else:
            scale_diameter = wtw_diameter_px.to(image).reshape(batch)
        scale_diameter = scale_diameter.clamp_min(1.0)

        if wtw_mm is None:
            physical_wtw = image.new_full((batch,), self.default_wtw_mm)
        else:
            physical_wtw = wtw_mm.to(image).reshape(batch)
        opening_pixels = opening_probability.sum(dim=(-3, -2, -1))
        opening_area_geometric_mm2 = (
            opening_pixels / scale_diameter.square() * physical_wtw.square()
        )

        global_feature = F.adaptive_avg_pool2d(deepest, 1).flatten(1)
        geometry = torch.stack(
            (
                opening_pixels / float(image_height * image_width),
                scale_diameter / float(max(image_height, image_width)),
                physical_wtw / self.default_wtw_mm,
            ),
            dim=1,
        )
        fused_feature = torch.cat((global_feature, local_feature, geometry), dim=1)
        severity_logits = self.severity_head(fused_feature)
        severity_probabilities = torch.softmax(severity_logits, dim=1)
        collapsed_probability = severity_probabilities[:, 1:].sum(dim=1)
        binary_logit = self.binary_head(fused_feature).squeeze(1)
        binary_probability = torch.sigmoid(binary_logit)
        area_fraction = torch.sigmoid(
            self.area_fraction_head(
                torch.cat(
                    (global_feature, (physical_wtw / self.default_wtw_mm).unsqueeze(1)),
                    dim=1,
                )
            ).squeeze(1)
        )
        opening_area_regression_mm2 = area_fraction * physical_wtw.square()
        measurement_confidence = endpoint_confidence.mean(dim=1)

        return CcsOutput(
            severity_logits=severity_logits,
            severity_probabilities=severity_probabilities,
            collapsed_probability=collapsed_probability,
            binary_logit=binary_logit,
            binary_probability=binary_probability,
            opening_mask_logits=opening_logits,
            opening_mask_probability=opening_probability,
            wtw_endpoint_heatmaps=endpoint_heatmaps,
            wtw_endpoints_xy=endpoint_xy,
            wtw_diameter_px=predicted_diameter,
            area_scale_diameter_px=scale_diameter,
            opening_area_mm2=opening_area_regression_mm2,
            opening_area_regression_mm2=opening_area_regression_mm2,
            opening_area_geometric_mm2=opening_area_geometric_mm2,
            measurement_confidence=measurement_confidence,
        )
