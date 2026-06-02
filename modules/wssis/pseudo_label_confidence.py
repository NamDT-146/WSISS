"""
Pseudo-label confidence gating for GNN-refined masks (Stage 1 logging + Stage 2 student).

Per-head logits are converted to probabilities, binarized with ``confidence_threshold``,
then combined with 2/3 voting in :func:`generate_pseudo_label_from_logits`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Default matches legacy ``probs > 0.5``; raise for stricter pseudo-labels (e.g. 0.7).
DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD = 0.5
PSEUDO_VOTE_MIN = 2


def resolve_pseudo_confidence_threshold(
    config: Optional[dict] = None,
    *,
    fallback: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> float:
    """Read ``pseudo_label.confidence_threshold`` from a run / checkpoint config dict."""
    if not config:
        return fallback
    pl = config.get("pseudo_label") or {}
    if isinstance(pl, dict) and "confidence_threshold" in pl:
        return float(pl["confidence_threshold"])
    if "pseudo_confidence_threshold" in config:
        return float(config["pseudo_confidence_threshold"])
    return fallback


def refined_probs_from_logits(refined_masks_logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid probabilities; accepts logits or probabilities in [0, 1]."""
    if refined_masks_logits.numel() == 0:
        return refined_masks_logits
    if refined_masks_logits.min() >= 0.0 and refined_masks_logits.max() <= 1.0:
        return refined_masks_logits.float()
    return torch.sigmoid(refined_masks_logits)


def binary_masks_from_probs(
    probs: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> torch.Tensor:
    """
    High-confidence binary mask per GNN head.

    Returns:
        float tensor matching ``probs`` shape, values in {0.0, 1.0}.
    """
    return (probs > confidence_threshold).float()


def binary_masks_from_logits(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> torch.Tensor:
    """Logits → per-head binary masks at ``confidence_threshold``."""
    return binary_masks_from_probs(
        refined_probs_from_logits(refined_masks_logits),
        confidence_threshold,
    )


def over_threshold_ratio(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> float:
    """
    Mean fraction of pixels with probability above ``confidence_threshold``.

    Averaged over batch, heads, height, and width (training diagnostic).
    """
    if refined_masks_logits.numel() == 0:
        return 0.0
    probs = refined_probs_from_logits(refined_masks_logits)
    return float((probs > confidence_threshold).float().mean().item())


def agreement_rate(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    vote_min: int = PSEUDO_VOTE_MIN,
) -> float:
    """Fraction of spatial locations where >= ``vote_min`` heads pass the threshold."""
    if refined_masks_logits.numel() == 0:
        return 0.0
    binary = binary_masks_from_logits(refined_masks_logits, confidence_threshold)
    if binary.dim() == 3:
        binary = binary.unsqueeze(0)
    votes = binary.sum(dim=1)
    agreed = (votes >= vote_min).float().mean()
    return float(agreed.item())


def generate_pseudo_label_from_logits(
    refined_masks_logits: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    *,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    vote_min: int = PSEUDO_VOTE_MIN,
) -> torch.Tensor:
    """
    2/3 vote on thresholded GNN heads → student pseudo-GT.

    Only pixels where at least ``vote_min`` heads exceed ``confidence_threshold``
    are kept in the pseudo mask.
    """
    if refined_masks_logits.dim() == 3:
        refined_masks_logits = refined_masks_logits.unsqueeze(0)
    binary = binary_masks_from_logits(refined_masks_logits, confidence_threshold)
    votes = binary.sum(dim=1, keepdim=True)
    agreed = (votes >= vote_min).float()
    if target_size is not None and agreed.shape[-2:] != target_size:
        agreed = F.interpolate(agreed, size=target_size, mode="nearest")
    return agreed
