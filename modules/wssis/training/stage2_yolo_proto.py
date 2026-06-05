"""YOLOv8-seg mask prototype extraction (no Detectron2 dependency)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.wssis.training.stage2_augment import IMAGENET_MEAN, IMAGENET_STD


def denorm_yolo_images(images: torch.Tensor) -> torch.Tensor:
    """YOLO expects RGB [0,1]; undo ImageNet norm from strong augment."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def find_proto_tensor(out) -> Optional[torch.Tensor]:
    """Locate mask prototype tensor [B, C, H, W] from YOLOv8-seg forward output."""
    if isinstance(out, torch.Tensor) and out.dim() == 4:
        if 8 <= out.shape[1] <= 64:
            return out
        return None
    if isinstance(out, (list, tuple)):
        protos = []
        for item in out:
            t = find_proto_tensor(item)
            if t is not None:
                protos.append(t)
        if not protos:
            return None
        for t in reversed(protos):
            if t.shape[1] == 32:
                return t
        return protos[-1]
    if isinstance(out, dict):
        for key in ("proto", "protos", "mask_proto", "pred_masks"):
            val = out.get(key)
            if isinstance(val, torch.Tensor) and val.dim() == 4:
                return val
        for val in out.values():
            t = find_proto_tensor(val)
            if t is not None:
                return t
    return None


def forward_yolo_backbone(student_model, images: torch.Tensor):
    """Run predict path (not loss dict) with gradients."""
    if hasattr(student_model, "_predict_once"):
        return student_model._predict_once(images)
    if hasattr(student_model, "predict"):
        return student_model.predict(images)
    return student_model(images)


def yolo_mask_logits(
    student_model: nn.Module,
    mask_proj: nn.Module,
    images: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """
    Differentiable [B,1,out_size,out_size] logits from YOLOv8-seg mask prototypes.

    Segment head returns (det, mask_coeff, proto) in training; proto is [B,32,H',W'].
    """
    imgs = denorm_yolo_images(images)
    if imgs.shape[-2] != out_size or imgs.shape[-1] != out_size:
        imgs = F.interpolate(imgs, size=(out_size, out_size), mode="bilinear", align_corners=False)

    out = forward_yolo_backbone(student_model, imgs)
    proto = find_proto_tensor(out)
    if proto is None:
        raise RuntimeError(
            f"Could not extract YOLO seg prototypes from output type {type(out).__name__}"
        )

    logits = mask_proj(proto)
    if logits.shape[-2:] != (out_size, out_size):
        logits = F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return logits
