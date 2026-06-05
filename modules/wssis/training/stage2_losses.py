"""
Stage-2 shared losses: PCE, symmetric SAM triplet, voting pseudo, student feedback, ramp-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from modules.wssis.pseudo_label_confidence import (
    DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    PSEUDO_VOTE_MIN,
    refined_probs_from_logits,
)
from modules.wssis.training.gnn_losses import soft_dice_symmetric


def _single_mask_logits(logits: torch.Tensor) -> torch.Tensor:
    """Collapse [B,C,H,W] to [B,1,H,W] for PCE (GNN v2 uses C=1; legacy may use C=3)."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if logits.shape[1] == 1:
        return logits
    return logits.mean(dim=1, keepdim=True)


def partial_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked BCE (PCE) on binary masks. logits/target [B,1,H,W] or [B,H,W]."""
    logits = _single_mask_logits(logits)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(probs, target.float(), reduction="none")
    denom = valid_mask.float().sum().clamp_min(eps)
    return (bce * valid_mask.float()).sum() / denom


def partial_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = _single_mask_logits(logits)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    probs = torch.sigmoid(logits)
    m = valid_mask.float()
    inter = (probs * target * m).sum()
    den = (probs * m).sum() + (target * m).sum() + eps
    dice = 1.0 - (2.0 * inter + eps) / den
    return dice


def build_pce_valid_mask(
    weak_signal: torch.Tensor,
    signal_type: str,
    *,
    box_outside_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (valid_mask, target) for partial supervision.

    point/scribble: valid = weak_map > 0, target = 1
    box: valid = outside box (weak ch1 uniform interior ignored), target = 0 outside
    """
    if weak_signal.dim() == 3:
        weak_signal = weak_signal.unsqueeze(0)
    b, c, h, w = weak_signal.shape
    device = weak_signal.device
    valid = torch.zeros((b, 1, h, w), device=device)
    target = torch.zeros((b, 1, h, w), device=device)

    sig = signal_type.replace("_only", "")
    if sig in ("point", "points", "scribble", "scribbles", "mixed"):
        valid = (weak_signal.max(dim=1, keepdim=True)[0] > 0.1).float()
        target = valid.clone()
    elif sig in ("box", "boxes"):
        box_ch = weak_signal[:, 1:2] if c >= 2 else weak_signal[:, :1]
        inside = (box_ch > 0.5).float()
        if box_outside_only:
            valid = (1.0 - inside)
            target = torch.zeros_like(valid)
        else:
            valid = torch.ones_like(inside)
            target = inside
    else:
        valid = (weak_signal.max(dim=1, keepdim=True)[0] > 0.1).float()
        target = valid.clone()
    return valid, target


def symmetric_sam_triplet_loss(masks_3: torch.Tensor) -> torch.Tensor:
    """Pairwise symmetric Dice on 3 SAM/GNN heads [B,3,H,W] or [3,H,W]."""
    if masks_3.dim() == 3:
        masks_3 = masks_3.unsqueeze(0)
    probs = refined_probs_from_logits(masks_3)
    b = probs.shape[0]
    total = probs.new_zeros(())
    n_pairs = 0
    for bi in range(b):
        for i in range(3):
            for j in range(i + 1, 3):
                total = total + soft_dice_symmetric(probs[bi, i], probs[bi, j])
                n_pairs += 1
    return total / max(n_pairs, 1)


def voting_pseudo_mask(
    probs_3: torch.Tensor,
    threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    vote_min: int = PSEUDO_VOTE_MIN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold each head, vote >= vote_min.

    Returns (pseudo_fg [B,1,H,W], valid_mask [B,1,H,W]).
    """
    if probs_3.dim() == 3:
        probs_3 = probs_3.unsqueeze(0)
    probs = refined_probs_from_logits(probs_3)
    binary = (probs > threshold).float()
    votes = binary.sum(dim=1, keepdim=True)
    valid = (votes >= vote_min).float()
    pseudo = valid.clone()
    return pseudo, valid


def student_feedback_loss(
    teacher_logits: torch.Tensor,
    student_probs: torch.Tensor,
    tau: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PCE from high-confidence student pixels onto teacher."""
    teacher_logits = _single_mask_logits(teacher_logits)
    if student_probs.dim() == 3:
        student_probs = student_probs.unsqueeze(1)
    high = (student_probs > tau).float()
    target = (student_probs > 0.5).float()
    return partial_bce_loss(teacher_logits, target.detach(), high)


@dataclass
class LossWeightSchedule:
    """Ramp unsup/feedback after warmup_frac of training."""

    warmup_frac: float = 0.2
    lambda_t_pce: float = 1.0
    lambda_t_sym: float = 0.1
    lambda_t_feedback: float = 0.05
    lambda_s_sup: float = 1.0
    lambda_s_unsup: float = 1.0
    lambda_s_semi: float = 0.5

    def progress(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0
        return min(1.0, step / max(1, total_steps))

    def ramp(self, step: int, total_steps: int) -> float:
        """Cosine ramp 0->1 after warmup."""
        p = self.progress(step, total_steps)
        if p <= self.warmup_frac:
            return 0.0
        t = (p - self.warmup_frac) / max(1e-8, 1.0 - self.warmup_frac)
        return 0.5 * (1.0 - math.cos(math.pi * min(1.0, t)))

    def weights(self, step: int, total_steps: int) -> dict[str, float]:
        r = self.ramp(step, total_steps)
        in_warmup = self.progress(step, total_steps) <= self.warmup_frac
        return {
            "lambda_t_pce": self.lambda_t_pce,
            "lambda_t_sym": self.lambda_t_sym if not in_warmup else min(self.lambda_t_sym, 0.1),
            "lambda_t_feedback": 0.0 if in_warmup else self.lambda_t_feedback * r,
            "lambda_s_sup": self.lambda_s_sup,
            "lambda_s_unsup": 0.0 if in_warmup else self.lambda_s_unsup * r,
            "lambda_s_semi": self.lambda_s_semi,
        }


def instance_mask_bce_dice(
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-instance CE + dice for unsup path."""
    vm = valid_mask if valid_mask is not None else torch.ones_like(target_mask)
    return (
        partial_bce_loss(pred_logits, target_mask, vm),
        partial_dice_loss(pred_logits, target_mask, vm),
    )
