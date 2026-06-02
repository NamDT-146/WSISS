"""
Stage-1 GNN losses: 9 aligned proposals (3 SAM heads x 3 weak prompts), BCE+Dice vs GT, warmup.

Per instance:
  - Box / scribble / point each yield 3 SAM multimasks and 3 GNN refined heads.
  - Hungarian alignment (box anchor) matches heads across prompt types.
  - Hierarchical KL: point -> scribble -> box (teacher detached).
  - Symmetric soft Dice on matched triplets (all 3 pairs), averaged over 3 heads.
  - Supervised BCE + Dice on all refined heads vs GT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.wssis.weak_prompts import WEAK_SIGNAL_TYPES

SIGNAL_ORDER = ("points_only", "scribbles_only", "boxes_only")


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
    if a.dim() == 2:
        a, b = a.flatten(), b.flatten()
    else:
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    num = 2.0 * (a * b).sum(dim=-1)
    den = (a * a).sum(dim=-1) + (b * b).sum(dim=-1) + eps
    return (1.0 - num / den).mean()


def bernoulli_kl(student: torch.Tensor, teacher: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(teacher || student); teacher should be detached."""
    s = student.clamp(eps, 1.0 - eps)
    t = teacher.clamp(eps, 1.0 - eps)
    return (
        t * (t.log() - s.log())
        + (1.0 - t) * ((1.0 - t).log() - (1.0 - s).log())
    ).mean()


def _pairwise_iou_cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = (a > 0.5).float().reshape(3, -1)
    b = (b > 0.5).float().reshape(3, -1)
    inter = a.unsqueeze(1) * b.unsqueeze(0)
    inter = inter.sum(dim=-1)
    union = a.sum(dim=-1, keepdim=True) + b.sum(dim=-1).unsqueeze(0) - inter
    iou = inter / (union + 1e-6)
    return 1.0 - iou


def hungarian_match_perm(cost: torch.Tensor) -> torch.Tensor:
    """Permutation index ``perm`` such that ``other[perm]`` aligns rows to anchor."""
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = torch.zeros(3, dtype=torch.long, device=cost.device)
        for i in range(len(r)):
            perm[int(c[i])] = int(r[i])
        return perm
    except ImportError:
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
        perm = torch.zeros(3, dtype=torch.long, device=cost.device)
        for r, c in pairs:
            perm[c] = r
        return perm


def align_three_heads(
    anchor_sam: torch.Tensor,
    other_sam: torch.Tensor,
    other_refined: torch.Tensor,
) -> torch.Tensor:
    """
    Align ``other`` to ``anchor`` using IoU on SAM masks; return permuted refined [3,H,W].
    """
    cost = _pairwise_iou_cost(anchor_sam.detach(), other_sam.detach())
    perm = hungarian_match_perm(cost)
    return other_refined[perm]


def supervised_seg_loss(
    seg_criterion: nn.Module,
    refined_logits: torch.Tensor,
    gt_masks: torch.Tensor,
) -> torch.Tensor:
    """BCE + Dice on all 3 refined heads vs GT."""
    logits = _ensure_b3hw(refined_logits)
    gt = _ensure_b1hw(gt_masks)
    if gt.shape[1] == 1:
        gt = gt.expand(-1, logits.shape[1], -1, -1)
    return seg_criterion(logits, gt)


def nine_aligned_proposal_loss(
    refined_logits: torch.Tensor,
    sam_masks_3: torch.Tensor,
    metas: Sequence[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Hierarchical KL + symmetric Dice on 9 matched proposals (3 heads x 3 prompt types).

    Alignment uses frozen SAM IoU (box -> scribble -> point); losses run on GNN refined logits.
    """
    device = refined_logits.device
    z = torch.tensor(0.0, device=device)
    groups = _group_triplet_indices(metas)
    if not groups:
        return z, z, z, {}

    kl_ps_list, kl_sb_list, sym_list = [], [], []
    n_groups = 0

    for idxs in groups.values():
        if len(idxs) != 3:
            continue
        by_type = {metas[i].get("weak_signal_type"): i for i in idxs}
        if not all(t in by_type for t in SIGNAL_ORDER):
            continue

        ib, is_, ip = by_type["boxes_only"], by_type["scribbles_only"], by_type["points_only"]
        rb = _ensure_b3hw(refined_logits[ib : ib + 1])[0]
        rs = _ensure_b3hw(refined_logits[is_ : is_ + 1])[0]
        rp = _ensure_b3hw(refined_logits[ip : ip + 1])[0]
        sb = sam_masks_3[ib].float().clamp(0.0, 1.0)
        ss = sam_masks_3[is_].float().clamp(0.0, 1.0)
        sp = sam_masks_3[ip].float().clamp(0.0, 1.0)

        rs_a = align_three_heads(sb, ss, rs)
        rp_a = align_three_heads(rs_a, sp, rp)

        pb = torch.sigmoid(rb)
        ps = torch.sigmoid(rs_a)
        pp = torch.sigmoid(rp_a)

        for hi in range(3):
            kl_sb_list.append(bernoulli_kl(ps[hi], pb[hi].detach()))
            kl_ps_list.append(bernoulli_kl(pp[hi], ps[hi].detach()))
            sym_list.append(
                (
                    soft_dice_symmetric(pb[hi], ps[hi])
                    + soft_dice_symmetric(ps[hi], pp[hi])
                    + soft_dice_symmetric(pb[hi], pp[hi])
                )
                / 3.0
            )
        n_groups += 1

    if not kl_ps_list:
        return z, z, z, {"n_triplets": 0}

    kl_p2s = torch.stack(kl_ps_list).mean()
    kl_s2b = torch.stack(kl_sb_list).mean()
    sym = torch.stack(sym_list).mean()
    stats = {
        "n_triplets": n_groups,
        "kl_p2s": kl_p2s.item(),
        "kl_s2b": kl_s2b.item(),
        "sym_nine": sym.item(),
    }
    return kl_p2s, kl_s2b, sym, stats


def _group_triplet_indices(metas: Sequence[dict]) -> Dict[Tuple[int, int], List[int]]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(metas):
        key = (int(m.get("image_id", -1)), int(m.get("ann_id", -1)))
        groups.setdefault(key, []).append(i)
    return groups


@dataclass
class Stage1LossWarmup:
    """Ramp KL down and symmetric up over early epochs."""

    warmup_epochs: int = 5
    kl_start: float = 0.2
    kl_end: float = 0.05
    sym_start: float = 0.02
    sym_end: float = 0.15

    def weights_for_epoch(self, epoch: int) -> Dict[str, float]:
        if self.warmup_epochs <= 0:
            return {"kl": self.kl_end, "sym": self.sym_end}
        t = min(1.0, max(0.0, (epoch - 1) / float(self.warmup_epochs)))
        kl = self.kl_start * (1.0 - t) + self.kl_end * t
        sym = self.sym_start * (1.0 - t) + self.sym_end * t
        return {"kl": kl, "sym": sym, "warmup_t": t}


class Stage1V2Loss(nn.Module):
    """BCE+Dice vs GT + warmup-weighted 9-proposal KL and symmetric losses."""

    def __init__(
        self,
        seg_criterion: nn.Module,
        *,
        kl_weight: float = 0.1,
        sym_weight: float = 0.1,
        loss_warmup: Optional[Stage1LossWarmup] = None,
    ):
        super().__init__()
        self.seg_criterion = seg_criterion
        self.kl_weight = kl_weight
        self.sym_weight = sym_weight
        self.loss_warmup = loss_warmup or Stage1LossWarmup(
            kl_start=kl_weight * 2.0,
            kl_end=kl_weight,
            sym_start=sym_weight * 0.2,
            sym_end=sym_weight,
        )

    def forward(
        self,
        refined_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        sam_masks_3: torch.Tensor,
        metas: Sequence[dict],
        *,
        epoch: int = 1,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lw = loss_weights or self.loss_warmup.weights_for_epoch(epoch)
        kl_w = lw.get("kl", self.kl_weight)
        sym_w = lw.get("sym", self.sym_weight)

        seg = supervised_seg_loss(self.seg_criterion, refined_logits, gt_masks)
        kl_p2s, kl_s2b, sym, nine_stats = nine_aligned_proposal_loss(
            refined_logits, sam_masks_3, metas
        )
        kl_total = kl_p2s + kl_s2b

        total = seg + kl_w * kl_total + sym_w * sym
        comps = {
            "seg": seg.item(),
            "kl_p2s": kl_p2s.item(),
            "kl_s2b": kl_s2b.item(),
            "kl_total": kl_total.item(),
            "kl_weighted": (kl_w * kl_total).item(),
            "sym_nine": sym.item(),
            "sym_weighted": (sym_w * sym).item(),
            "sym_raw": sym.item(),
            "kl_w": kl_w,
            "sym_w": sym_w,
            "warmup_t": lw.get("warmup_t", 1.0),
            "total": total.item(),
            **nine_stats,
        }
        return total, comps
