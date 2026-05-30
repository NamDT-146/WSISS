"""
Losses, metrics, and SAM preprocessing helpers for Stage-1 training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# #region agent log
def _agent_dbg_log(
    location: str,
    message: str,
    data: dict,
    hypothesis_id: str,
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": "5ff431",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": run_id,
    }
    line = json.dumps(payload) + "\n"
    candidates = [
        Path("debug-5ff431.log"),
        Path("/kaggle/working/debug-5ff431.log"),
    ]
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "debug-5ff431.log")
    except NameError:
        pass
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            break
        except OSError:
            continue


def probe_cuda_runtime() -> bool:
    """
    Test the same CUDA ops used in preprocess_sam_batch.
    Returns False when PyTorch has no kernels for the GPU (common on new Kaggle GPUs).
    """
    if not torch.cuda.is_available():
        _agent_dbg_log(
            "sam_stage1_common.probe_cuda_runtime",
            "cuda not available",
            {"torch": torch.__version__},
            "A",
        )
        return False

    info = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    try:
        x = torch.zeros(1, 3, 8, 8, device="cuda")
        x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
        mean = torch.tensor([123.675, 116.28, 103.53], device="cuda")
        std = torch.tensor([58.395, 57.12, 57.375], device="cuda")
        _ = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        torch.cuda.synchronize()
        _agent_dbg_log(
            "sam_stage1_common.probe_cuda_runtime",
            "cuda probe ok",
            info,
            "A",
        )
        return True
    except Exception as e:
        info["error"] = type(e).__name__
        info["error_msg"] = str(e)[:500]
        _agent_dbg_log(
            "sam_stage1_common.probe_cuda_runtime",
            "cuda probe failed",
            info,
            "A",
        )
        return False


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Pick cuda only if runtime probe succeeds (Hypothesis A/D)."""
    use_cuda = prefer_cuda and probe_cuda_runtime()
    dev = torch.device("cuda" if use_cuda else "cpu")
    _agent_dbg_log(
        "sam_stage1_common.resolve_device",
        "resolved device",
        {"device": str(dev), "prefer_cuda": prefer_cuda},
        "D",
    )
    return dev
# #endregion


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedSegLoss(nn.Module):
    """BCE + Dice (PLAN Stage-1 supervised loss)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )


def compute_iou(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    if pred.min() < 0:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    if pred.min() < 0:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum().item()
    total = pred_binary.sum().item() + target_binary.sum().item()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for i in range(pred.shape[0]):
            self.iou_sum += compute_iou(pred[i : i + 1], target[i : i + 1])
            self.dice_sum += compute_dice(pred[i : i + 1], target[i : i + 1])
            self.count += 1

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"iou": 0.0, "dice": 0.0}
        return {
            "iou": self.iou_sum / self.count,
            "dice": self.dice_sum / self.count,
        }


def preprocess_sam_batch(
    images: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    target_size: int = 1024,
) -> torch.Tensor:
    """
    Resize + pad to target_size and apply SAM normalization.

    Args:
        images: [B, 3, H, W] in [0, 1] RGB.

    Returns:
        [B, 3, target_size, target_size] normalized for SAM encoder.
    """
    B, C, H, W = images.shape
    scale = target_size / max(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    x = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    x = F.pad(x, (0, pad_w, 0, pad_h))
    mean = pixel_mean.view(1, 3, 1, 1).to(x.device, x.dtype)
    std = pixel_std.view(1, 3, 1, 1).to(x.device, x.dtype)
    return (x - mean) / std


@torch.no_grad()
def encode_sam_embeddings(
    sam_model: nn.Module,
    images: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    """Run frozen SAM image encoder -> [B, 256, 64, 64]."""
    # #region agent log
    _agent_dbg_log(
        "sam_stage1_common.encode_sam_embeddings",
        "encode entry",
        {
            "images_device": str(images.device),
            "sam_param_device": str(next(sam_model.parameters()).device),
            "pixel_mean_device": str(pixel_mean.device),
        },
        "B",
    )
    # #endregion
    sam_model.eval()
    x = preprocess_sam_batch(images, pixel_mean, pixel_std, target_size=1024)
    out = sam_model.image_encoder(x)
    # #region agent log
    _agent_dbg_log(
        "sam_stage1_common.encode_sam_embeddings",
        "encode ok",
        {"out_shape": list(out.shape), "out_device": str(out.device)},
        "B",
    )
    # #endregion
    return out


def freeze_sam(sam_model: nn.Module) -> None:
    sam_model.eval()
    for p in sam_model.parameters():
        p.requires_grad = False


def load_sam_vit_b(checkpoint_path: str, device: torch.device):
    """Load SAM ViT-B (requires segment_anything installed)."""
    from segment_anything import sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    freeze_sam(sam)
    return sam


def get_sam_pixel_stats(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """SAM default ImageNet-style pixel mean/std buffers."""
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device)
    return pixel_mean, pixel_std
