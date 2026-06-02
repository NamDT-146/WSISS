"""
Stage-1 GNN v2 losses: triplet KL, matched symmetric Dice, intra-SAM-head consensus, anchors.

Main supervised loss remains BCE + Dice on refined [B,1,H,W] vs GT (CombinedSegLoss).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.wssis.weak_prompts import WEAK_SIGNAL_TYPES

# Weaker → stronger for distillation (student || teacher)
SIGNAL_ORDER = ("points_only", "scribbles_only", "boxes_only")
SIGNAL_RANK = {s: i for i, s in enumerate(SIGNAL_ORDER)}


def _ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _ensure_b3hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def soft_dice_symmetric(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric soft Dice loss between two probability maps [H,W] or [B,H,W]."""
    if a.dim() == 2:
        a, b = a.flatten(), b.flatten()
    else:
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    num = 2.0 * (a * b).sum(dim=-1)
    den = (a * a).sum(dim=-1) + (b * b).sum(dim=-1) + eps
    dice = num / den
    return (1.0 - dice).mean()


def bernoulli_kl(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(teacher || student) on Bernoulli probs; teacher should be detached."""
    s = student.clamp(eps, 1.0 - eps)
    t = teacher.clamp(eps, 1.0 - eps)
    return (
        t * (t.log() - s.log())
        + (1.0 - t) * ((1.0 - t).log() - (1.0 - s).log())
    ).mean()


def _pairwise_iou_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3x3 cost = 1 - IoU between binary masks [3,H,W] and [3,H,W]."""
    a = (a > 0.5).float().reshape(3, -1)
    b = (b > 0.5).float().reshape(3, -1)
    inter = a.unsqueeze(1) * b.unsqueeze(0)
    inter = inter.sum(dim=-1)
    union = a.sum(dim=-1, keepdim=True) + b.sum(dim=-1).unsqueeze(0) - inter
    iou = inter / (union + 1e-6)
    return 1.0 - iou


def hungarian_match_perm(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (row_ind, col_ind) permutation for 3x3 cost via scipy or greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.tensor(r, device=cost.device), torch.tensor(c, device=cost.device)
    except ImportError:
        # Greedy fallback for 3x3
        used_r, used_c = set(), set()
        pairs = []
        flat = cost.flatten()
        order = flat.argsort()
        for idx in order:
            r = int(idx // 3)
            c = int(idx % 3)
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
            if len(pairs) == 3:
                break
        if len(pairs) < 3:
            for i in range(3):
                if i not in used_r:
                    for j in range(3):
                        if j not in used_c:
                            pairs.append((i, j))
                            used_r.add(i)
                            used_c.add(j)
                            break
        r = torch.tensor([p[0] for p in pairs], device=cost.device)
        c = torch.tensor([p[1] for p in pairs], device=cost.device)
        return r, c


def align_sam_masks_3(
    anchor: torch.Tensor, other: torch.Tensor
) -> torch.Tensor:
    """
    Permute ``other`` [3,H,W] to align with anchor [3,H,W] using max-IoU matching.
    """
    cost = _pairwise_iou_cost(anchor, other)
    r, c = hungarian_match_perm(cost)
    perm = torch.zeros(3, dtype=torch.long, device=other.device)
    for i in range(len(r)):
        perm[c[i]] = r[i]
    return other[perm]


def intra_sam_symmetric_dice(sam_masks_3: torch.Tensor) -> torch.Tensor:
    """Symmetric soft Dice across SAM's 3 multimask proposals [B,3,H,W]."""
    p = sam_masks_3.float().clamp(0.0, 1.0)
    losses = []
    for b in range(p.shape[0]):
        m0, m1, m2 = p[b, 0], p[b, 1], p[b, 2]
        losses.append(
            (
                soft_dice_symmetric(m0, m1)
                + soft_dice_symmetric(m1, m2)
                + soft_dice_symmetric(m0, m2)
            )
            / 3.0
        )
    if not losses:
        return torch.tensor(0.0, device=sam_masks_3.device)
    return torch.stack(losses).mean()


def triplet_refined_symmetric_dice(
    refined: torch.Tensor,
    metas: Sequence[dict],
) -> torch.Tensor:
    """Soft Dice between matched refined masks (point, scribble, box) per instance."""
    groups = _group_triplet_indices(metas)
    if not groups:
        return torch.tensor(0.0, device=refined.device)
    p = torch.sigmoid(_ensure_b1hw(refined))
    losses = []
    for _key, idxs in groups.items():
        if len(idxs) != 3:
            continue
        by_type = {metas[i].get("weak_signal_type"): p[i, 0] for i in idxs}
        if not all(t in by_type for t in SIGNAL_ORDER):
            continue
        mp, ms, mb = by_type["points_only"], by_type["scribbles_only"], by_type["boxes_only"]
        losses.append(
            (soft_dice_symmetric(mp, ms) + soft_dice_symmetric(ms, mb) + soft_dice_symmetric(mp, mb))
            / 3.0
        )
    if not losses:
        return torch.tensor(0.0, device=refined.device)
    return torch.stack(losses).mean()


def triplet_hierarchical_kl(
    refined: torch.Tensor,
    metas: Sequence[dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KL(scribble || box) and KL(point || scribble) on refined masks.
    Returns (kl_p2s, kl_s2b) scalars.
    """
    groups = _group_triplet_indices(metas)
    device = refined.device
    if not groups:
        z = torch.tensor(0.0, device=device)
        return z, z
    p = torch.sigmoid(_ensure_b1hw(refined))
    kl_ps, kl_sb = [], []
    for _key, idxs in groups.items():
        if len(idxs) != 3:
            continue
        by_type = {metas[i].get("weak_signal_type"): p[i, 0] for i in idxs}
        if not all(t in by_type for t in SIGNAL_ORDER):
            continue
        mp, ms, mb = by_type["points_only"], by_type["scribbles_only"], by_type["boxes_only"]
        kl_sb.append(bernoulli_kl(ms, mb.detach()))
        kl_ps.append(bernoulli_kl(mp, ms.detach()))
    if not kl_ps:
        z = torch.tensor(0.0, device=device)
        return z, z
    return torch.stack(kl_ps).mean(), torch.stack(kl_sb).mean()


def triplet_matched_sam_kl_sym(
    refined: torch.Tensor,
    sam_masks_3: torch.Tensor,
    metas: Sequence[dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    After aligning SAM heads across types, apply KL and symmetric Dice on
    refined logits at matched head index (mean over 3 heads).
    """
    groups = _group_triplet_indices(metas)
    device = refined.device
    if not groups:
        z = torch.tensor(0.0, device=device)
        return z, z
    r = torch.sigmoid(_ensure_b1hw(refined))
    s = sam_masks_3.float().clamp(0.0, 1.0)
    kl_ps, kl_sb, sym = [], [], []
    for _key, idxs in groups.items():
        if len(idxs) != 3:
            continue
        by_type = {
            metas[i].get("weak_signal_type"): (r[i, 0], s[i]) for i in idxs
        }
        if not all(t in by_type for t in SIGNAL_ORDER):
            continue
        rp, sp = by_type["points_only"]
        rs, ss = by_type["scribbles_only"]
        rb, sb = by_type["boxes_only"]
        ss_a = align_sam_masks_3(sb, ss)
        sp_a = align_sam_masks_3(ss_a, sp)
        for hi in range(3):
            kl_sb.append(bernoulli_kl(rs, rb[:, hi].detach()))
            kl_ps.append(bernoulli_kl(rp, rs.detach()))
            sym.append(
                (
                    soft_dice_symmetric(rb[:, hi], rs)
                    + soft_dice_symmetric(rs, rp)
                    + soft_dice_symmetric(rb[:, hi], rp)
                )
                / 3.0
            )
    if not kl_ps:
        z = torch.tensor(0.0, device=device)
        return z, z
    return torch.stack(kl_ps).mean(), torch.stack(sym).mean()


def weak_anchor_loss(
    refined_logits: torch.Tensor,
    weak_signal: torch.Tensor,
    metas: Sequence[dict],
) -> torch.Tensor:
    """
    Anchor refined mask to weak prompt map (BCE on weak channel) + box exterior penalty.
    """
    pred = torch.sigmoid(_ensure_b1hw(refined_logits))
    weak = _ensure_b1hw(weak_signal)
    bce = F.binary_cross_entropy(pred, weak.clamp(0.0, 1.0), reduction="mean")
    return bce


def _group_triplet_indices(metas: Sequence[dict]) -> Dict[Tuple[int, int], List[int]]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(metas):
        key = (int(m.get("image_id", -1)), int(m.get("ann_id", -1)))
        groups.setdefault(key, []).append(i)
    return {k: v for k, v in groups.items() if len(v) >= 1}


class Stage1V2Loss(nn.Module):
  """Combined Stage-1 v2 loss with configurable weights."""

  def __init__(
      self,
      seg_criterion: nn.Module,
      *,
      kl_weight: float = 0.1,
      sym_triplet_weight: float = 0.1,
      sym_sam_weight: float = 0.1,
      anchor_weight: float = 0.05,
  ):
      super().__init__()
      self.seg_criterion = seg_criterion
      self.kl_weight = kl_weight
      self.sym_triplet_weight = sym_triplet_weight
      self.sym_sam_weight = sym_sam_weight
      self.anchor_weight = anchor_weight

  def forward(
      self,
      refined_logits: torch.Tensor,
      gt_masks: torch.Tensor,
      sam_masks_3: torch.Tensor,
      weak_signal: torch.Tensor,
      metas: Sequence[dict],
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
      gt = _ensure_b1hw(gt_masks)
      logits = _ensure_b1hw(refined_logits)
      seg = self.seg_criterion(logits, gt)

      kl_p2s, kl_s2b = triplet_hierarchical_kl(logits, metas)
      kl_total = kl_p2s + kl_s2b

      sym_triplet = triplet_refined_symmetric_dice(logits, metas)
      sym_sam = intra_sam_symmetric_dice(sam_masks_3)
      anchor = weak_anchor_loss(logits, weak_signal, metas)

      total = (
          seg
          + self.kl_weight * kl_total
          + self.sym_triplet_weight * sym_triplet
          + self.sym_sam_weight * sym_sam
          + self.anchor_weight * anchor
      )
      comps = {
          "seg": seg.item(),
          "kl_p2s": kl_p2s.item(),
          "kl_s2b": kl_s2b.item(),
          "kl_weighted": (self.kl_weight * kl_total).item(),
          "sym_triplet": sym_triplet.item(),
          "sym_triplet_weighted": (self.sym_triplet_weight * sym_triplet).item(),
          "sym_sam": sym_sam.item(),
          "sym_sam_weighted": (self.sym_sam_weight * sym_sam).item(),
          "anchor": anchor.item(),
          "anchor_weighted": (self.anchor_weight * anchor).item(),
          "total": total.item(),
      }
      return total, comps
