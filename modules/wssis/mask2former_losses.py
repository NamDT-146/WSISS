"""Feature projector and semi-weak losses for Stage-2 SWSIS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """Align Mask2Former stride-16 features to SAM embedding space (64x64)."""

    def __init__(self, m2f_dim: int = 256, sam_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(m2f_dim, sam_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, sam_dim),
        )

    def forward(self, m2f_feat_stride16: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            m2f_feat_stride16,
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        return self.proj(upsampled)


def feature_distillation_loss(
    aligned_m2f_feat: torch.Tensor,
    sam_feat: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(aligned_m2f_feat, sam_feat.detach())


def agreement_rate(refined_logits: torch.Tensor) -> float:
    """Fraction of spatial locations with >=2/3 mask agreement."""
    if refined_logits.numel() == 0:
        return 0.0
    probs = torch.sigmoid(refined_logits)
    binary = (probs > 0.5).float()
    votes = binary.sum(dim=1)
    agreed = (votes >= 2).float().mean()
    return float(agreed.item())
