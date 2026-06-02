"""Feature aligner and semi-weak losses for Stage-2 SWSIS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFeatureAligner(nn.Module):
    """1x1 conv only — maps student stride-16 features into SAM embedding dim.

    SAM targets are downsampled to the student grid in the loss (no student upsample).
    """

    def __init__(self, m2f_dim: int = 384, sam_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(m2f_dim, sam_dim, kernel_size=1, bias=False)

    def forward(self, m2f_feat_stride16: torch.Tensor) -> torch.Tensor:
        return self.proj(m2f_feat_stride16)


# Backward-compatible alias (older checkpoints / docs).
FeatureProjector = LightweightFeatureAligner


def feature_distillation_loss(
    aligned_m2f_feat: torch.Tensor,
    sam_feat: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(aligned_m2f_feat, sam_feat.detach())


def agreement_rate(
    refined_logits: torch.Tensor,
    confidence_threshold: float | None = None,
) -> float:
    """Fraction of spatial locations with >=2/3 mask agreement above threshold."""
    from modules.wssis.pseudo_label_confidence import (
        DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
        agreement_rate as _agreement_rate,
    )

    thresh = (
        DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else confidence_threshold
    )
    return _agreement_rate(refined_logits, confidence_threshold=thresh)
