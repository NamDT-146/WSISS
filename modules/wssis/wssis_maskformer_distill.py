"""
DEPRECATED: Feature distillation removed in Stage-2 joint loss (STAGE2_PROPOSAL).

Kept for backward compatibility only; ``attach_wssis_distillation`` is no longer called.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from modules.wssis.mask2former_losses import LightweightFeatureAligner, feature_distillation_loss
from modules.wssis.sam_cache import fetch_sam_embeddings_batch, load_sam_embedding_cache


def _backbone_feat_dim(model: nn.Module, feat_name: str) -> int:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 384
    channels = getattr(backbone, "_out_feature_channels", None)
    if isinstance(channels, dict) and feat_name in channels:
        return int(channels[feat_name])
    return 384


def _zero_module_loss(module: nn.Module) -> torch.Tensor:
    """Autograd-linked zero loss so DDP always sees module params as used."""
    loss: Optional[torch.Tensor] = None
    for param in module.parameters():
        if not param.requires_grad:
            continue
        term = param.reshape(-1)[0] * 0.0
        loss = term if loss is None else loss + term
    if loss is None:
        raise RuntimeError("Expected at least one trainable parameter in module")
    return loss


def _load_sam_embed_for_rec(
    rec: dict,
    device: torch.device,
    model: nn.Module,
) -> Optional[torch.Tensor]:
    image_id = int(rec.get("image_id", 0))
    split = rec.get("split", "train")
    emb = load_sam_embedding_cache(image_id, split, device)
    if emb is not None:
        return emb

    if "image" not in rec:
        return None

    if getattr(model, "_wssis_distill_sam", None) is None:
        from modules.vig_refinenet.sam_stage1_common import (
            get_sam_pixel_stats,
            load_sam_vit_b,
        )
        from modules.wssis.paths import sam_vit_b_checkpoint

        model._wssis_distill_sam = load_sam_vit_b(str(sam_vit_b_checkpoint()), device)
        mean, std = get_sam_pixel_stats(device)
        model._wssis_distill_sam_mean = mean
        model._wssis_distill_sam_std = std

    img = rec["image"].float().to(device)
    if img.max() > 1.5:
        img = img / 255.0
    meta = {"image_id": image_id, "split": split}
    batch, _ = fetch_sam_embeddings_batch(
        [meta],
        model._wssis_distill_sam,
        img.unsqueeze(0),
        model._wssis_distill_sam_mean,
        model._wssis_distill_sam_std,
        use_cache=False,
    )
    return batch[0]


def _compute_distill_loss(
    batched_inputs: List[dict],
    backbone_features: Optional[Dict[str, torch.Tensor]],
    aligner: LightweightFeatureAligner,
    feat_name: str,
    model: nn.Module,
) -> Optional[torch.Tensor]:
    if backbone_features is None or feat_name not in backbone_features:
        return None

    res4 = backbone_features[feat_name]
    device = res4.device

    weak_indices = [
        i for i, rec in enumerate(batched_inputs) if not rec.get("wssis_is_labeled", True)
    ]
    if not weak_indices:
        return None

    res4_weak = res4[weak_indices]
    sam_embeds: List[torch.Tensor] = []
    valid_rows: List[int] = []

    for local_i, batch_i in enumerate(weak_indices):
        emb = _load_sam_embed_for_rec(batched_inputs[batch_i], device, model)
        if emb is None:
            continue
        sam_embeds.append(emb)
        valid_rows.append(local_i)

    if not sam_embeds:
        return None

    if len(valid_rows) < len(weak_indices):
        res4_weak = res4_weak[valid_rows]

    sam_stack = torch.stack(sam_embeds, dim=0)
    target_hw = res4_weak.shape[-2:]
    if sam_stack.shape[-2:] != target_hw:
        sam_stack = torch.nn.functional.adaptive_avg_pool2d(sam_stack, output_size=target_hw)

    aligned = aligner(res4_weak)
    return feature_distillation_loss(aligned, sam_stack)


def attach_wssis_distillation(model: nn.Module, cfg) -> nn.Module:
    """Patch MaskFormer.forward to add loss_distill on weak images (stride-16 vs SAM cache)."""
    wssis = cfg.WSSIS
    if not getattr(wssis, "USE_SEMI_WEAK", False) or not getattr(wssis, "USE_DISTILL", False):
        return model

    if not hasattr(model, "backbone"):
        raise TypeError(f"Cannot attach distillation to {type(model).__name__} (no backbone).")

    feat_name = getattr(wssis, "DISTILL_BACKBONE_FEAT", "res4")
    feat_dim = int(getattr(wssis, "DISTILL_FEAT_DIM", 0)) or _backbone_feat_dim(model, feat_name)
    distill_weight = float(getattr(wssis, "DISTILL_WEIGHT", 1.0))

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    aligner = LightweightFeatureAligner(m2f_dim=feat_dim, sam_dim=256).to(device)
    model.add_module("wssis_aligner", aligner)
    model.wssis_distill_weight = distill_weight
    model.wssis_distill_feat = feat_name

    original_forward = model.forward

    def forward_with_distill(batched_inputs: List[dict]):
        if not model.training:
            return original_forward(batched_inputs)

        captured: Dict[str, Any] = {}

        def _hook(_module, _inp, output):
            if isinstance(output, dict):
                captured["features"] = output

        handle = model.backbone.register_forward_hook(_hook)
        try:
            losses = original_forward(batched_inputs)
        finally:
            handle.remove()

        distill = _compute_distill_loss(
            batched_inputs,
            captured.get("features"),
            aligner,
            feat_name,
            model,
        )
        if distill is None:
            distill = _zero_module_loss(aligner)
        losses["loss_distill"] = distill * distill_weight
        return losses

    model.forward = forward_with_distill  # type: ignore[method-assign]
    return model
