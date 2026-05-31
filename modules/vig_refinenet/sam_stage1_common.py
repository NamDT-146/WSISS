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
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
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


# COCO instance-seg IoU thresholds for mask AP (0.50:0.05:0.95)
COCO_MASK_IOU_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.5, 1.0, 0.05).tolist())


def mask_ap_from_iou(iou: float, thresholds=COCO_MASK_IOU_THRESHOLDS) -> float:
    """Per-instance mask AP contribution (fraction of thresholds met)."""
    return float(np.mean([iou >= t for t in thresholds]))


def symmetric_loss(logits: torch.Tensor) -> torch.Tensor:
    """Pairwise MSE between 3 refined mask logits [B, 3, H, W]."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    if logits.shape[1] < 2:
        return torch.tensor(0.0, device=logits.device)
    p = torch.sigmoid(logits)
    m1, m2, m3 = p[:, 0], p[:, 1], p[:, 2]
    return (F.mse_loss(m1, m2) + F.mse_loss(m2, m3) + F.mse_loss(m1, m3)) / 3.0


def _scale_prompt_to_image(
    prompts: dict, prompt_space: int, image_space: int
) -> dict:
    """Scale point/bbox from mask resolution to image resolution."""
    if prompt_space == image_space:
        return dict(prompts)
    scale = image_space / float(prompt_space)
    out = dict(prompts)
    if "point" in out:
        out["point"] = [out["point"][0] * scale, out["point"][1] * scale]
    if "bbox" in out:
        b = out["bbox"]
        out["bbox"] = [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
    return out


def build_batch_prompts_from_masks(
    masks: torch.Tensor,
    policy: str = "train_online",
    signal_type: str = "mixed",
    metas: Optional[list] = None,
) -> list:
    """Build weak prompt dicts from GT masks [B, 1, H, W]."""
    from modules.wssis.weak_prompts import build_instance_prompts

    prompts = []
    for i in range(masks.shape[0]):
        mask_np = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        ann_id = None
        if metas and i < len(metas):
            ann_id = metas[i].get("ann_id")
        prompts.append(
            build_instance_prompts(
                mask_np,
                policy=policy,
                signal_type=signal_type,
                ann_id=ann_id,
            )
        )
    return prompts


def generate_pseudo_label_from_logits(
    refined_masks_logits: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    2/3 vote agreement on 3 refined mask heads.

    Args:
        refined_masks_logits: [B, 3, H, W] or [3, H, W]
    Returns:
        pseudo_gt: [B, 1, H, W] or [1, H, W] binary float
    """
    if refined_masks_logits.dim() == 3:
        refined_masks_logits = refined_masks_logits.unsqueeze(0)
    probs = torch.sigmoid(refined_masks_logits)
    binary = (probs > 0.5).float()
    votes = binary.sum(dim=1, keepdim=True)
    agreed = (votes >= 2).float()
    if target_size is not None and agreed.shape[-2:] != target_size:
        agreed = F.interpolate(agreed, size=target_size, mode="nearest")
    return agreed


@torch.no_grad()
def forward_teacher_objects(
    sam_model: nn.Module,
    gnn_model: Optional[nn.Module],
    image: torch.Tensor,
    object_masks: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    mask_size: int,
    meta: dict,
    *,
    prompt_policy: str = "train_online",
    signal_type: str = "mixed",
    use_gnn: bool = True,
    use_sam_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Teacher forward for one image with N objects.

    Returns:
        pseudo_masks: [N, 1, mask_size, mask_size]
        raw_sam_best: [N, 1, mask_size, mask_size]
        refined_logits: [N, 3, mask_size, mask_size]
    """
    from modules.wssis.sam_cache import fetch_sam_embeddings_batch
    from modules.wssis.weak_prompts import build_image_prompts

    device = image.device
    n = object_masks.shape[0]
    mask_np_list = [
        (object_masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
        for i in range(n)
    ]
    prompts = build_image_prompts(
        mask_np_list,
        policy=prompt_policy,
        signal_type=signal_type,
        ann_ids=meta.get("ann_ids"),
    )
    image_batch = image.unsqueeze(0).expand(n, -1, -1, -1)
    metas = [
        {
            "image_id": meta["image_id"],
            "ann_id": meta["ann_ids"][i],
            "split": meta.get("split", "train"),
        }
        for i in range(n)
    ]
    masks_batch = object_masks.unsqueeze(1) if object_masks.dim() == 3 else object_masks

    sam_embed, _ = fetch_sam_embeddings_batch(
        metas,
        sam_model,
        image_batch,
        pixel_mean,
        pixel_std,
        use_cache=use_sam_cache,
    )
    sam_masks_3, sam_scores = decode_sam_masks_3_batch(
        sam_model,
        image_batch,
        prompts,
        mask_size=mask_size,
        prompt_space=mask_size,
        image_embeddings=sam_embed,
    )
    weak_signal = build_weak_signal_tensor(
        prompts,
        spatial_size=mask_size,
        device=device,
        mask_np_list=mask_np_list,
        policy=prompt_policy,
    )
    if use_gnn and gnn_model is not None:
        refined_logits = gnn_model(sam_embed, image_batch, sam_masks_3, weak_signal)
    else:
        refined_logits = sam_masks_3
    pseudo = generate_pseudo_label_from_logits(refined_logits, target_size=(mask_size, mask_size))
    raw_best = _best_mask_by_score(sam_masks_3, sam_scores)
    return pseudo, raw_best, refined_logits


def build_weak_signal_tensor(
    prompts: list,
    spatial_size: int,
    device: torch.device,
    mask_np_list: Optional[list] = None,
    active_signal: Optional[str] = None,
    policy: str = "val_fixed",
    gaussian_sigma: float = 4.0,
) -> torch.Tensor:
    """
    Rasterize weak prompts to [B, 3, H, W]:
      ch0 — point (Gaussian), ch1 — box (uniform fill), ch2 — scribble (Gaussian-widened).

    Unified training (``active_signal=None``): all three channels populated.
    Per-type eval: pass ``active_signal`` in {'boxes_only','points_only','scribbles_only'}.
    """
    from modules.wssis.weak_prompts import rasterize_weak_signal_maps

    batch = []
    for i, p in enumerate(prompts):
        mask_np = None
        if mask_np_list and i < len(mask_np_list):
            mask_np = mask_np_list[i]
        elif spatial_size and "bbox" in p:
            mask_np = np.zeros((spatial_size, spatial_size), dtype=np.uint8)
        else:
            mask_np = np.zeros((spatial_size, spatial_size), dtype=np.uint8)

        maps = rasterize_weak_signal_maps(
            mask_np,
            p,
            spatial_size=spatial_size,
            active_signal=active_signal,
            policy=policy,
            gaussian_sigma=gaussian_sigma,
        )
        batch.append(maps)

    return torch.from_numpy(np.stack(batch, axis=0)).to(device)


@torch.no_grad()
def decode_sam_masks_3_batch(
    sam_model: nn.Module,
    images: torch.Tensor,
    prompts: list,
    mask_size: int = 256,
    prompt_space: Optional[int] = None,
    image_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run SAM prompt decoder → 3 proposal masks per sample.

    Args:
        images: [B, 3, H, W] float RGB in [0, 1].
        prompts: list of prompt dicts (point/bbox in prompt_space coords).
        mask_size: output spatial size for GNN input.
        prompt_space: resolution prompts were built in (defaults to mask_size).
        image_embeddings: optional [B, 256, 64, 64] from P0.2 cache (skips encoder in set_image).

    Returns:
        sam_masks: [B, 3, mask_size, mask_size] float in {0, 1}
        sam_scores: [B, 3] SAM quality scores
    """
    from segment_anything import SamPredictor

    B, _, H, W = images.shape
    prompt_space = prompt_space or mask_size
    device = images.device
    all_masks = []
    all_scores = []

    predictor = SamPredictor(sam_model)
    for i in range(B):
        scaled = _scale_prompt_to_image(prompts[i], prompt_space, H)

        if image_embeddings is not None:
            predictor.reset_image()
            predictor.original_size = (H, W)
            predictor.input_size = (H, W)
            predictor.features = image_embeddings[i : i + 1].to(
                device=predictor.device, dtype=next(sam_model.parameters()).dtype
            )
            predictor.is_image_set = True
        else:
            img_np = (
                images[i].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255
            ).astype(np.uint8)
            predictor.set_image(img_np)

        point_coords = None
        point_labels = None
        box = None
        if "point" in scaled:
            point_coords = np.array([scaled["point"]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        if "bbox" in scaled:
            x, y, bw, bh = scaled["bbox"]
            box = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        # masks: [3, H, W] at original image resolution
        m = torch.from_numpy(masks.astype(np.float32)).to(device)
        if m.shape[-2:] != (mask_size, mask_size):
            m = F.interpolate(
                m.unsqueeze(0),
                size=(mask_size, mask_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        all_masks.append(m)
        all_scores.append(torch.from_numpy(scores.astype(np.float32)).to(device))

    sam_masks = torch.stack(all_masks, dim=0)
    sam_scores = torch.stack(all_scores, dim=0)
    return sam_masks, sam_scores


def _best_mask_by_score(masks_3: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Pick highest SAM-score mask per sample. Returns [B, 1, H, W]."""
    idx = scores.argmax(dim=1)
    B = masks_3.shape[0]
    chosen = masks_3[torch.arange(B, device=masks_3.device), idx]
    return chosen.unsqueeze(1)


def _best_mask_by_iou(masks_3: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pick mask with highest IoU vs GT (oracle, for analysis). Returns [B, 1, H, W]."""
    B = masks_3.shape[0]
    best = []
    for i in range(B):
        ious = [
            compute_iou(masks_3[i : i + 1, k : k + 1], target[i : i + 1])
            for k in range(masks_3.shape[1])
        ]
        k = int(np.argmax(ious))
        best.append(masks_3[i, k : k + 1])
    return torch.stack(best, dim=0)


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


class RefinementMetricTracker:
    """Raw SAM vs GNN-refined metrics with COCO-style mask AP."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_iou_sum = 0.0
        self.refined_iou_sum = 0.0
        self.raw_ap_sum = 0.0
        self.refined_ap_sum = 0.0
        self.raw_ap50_sum = 0.0
        self.refined_ap50_sum = 0.0
        self.count = 0

    def update(
        self,
        raw_masks_3: torch.Tensor,
        raw_scores: torch.Tensor,
        refined_logits: torch.Tensor,
        target: torch.Tensor,
    ):
        """Compare SAM best-score mask vs mean refined mask quality."""
        raw_best = _best_mask_by_score(raw_masks_3, raw_scores)
        refined_mean = torch.sigmoid(refined_logits).mean(dim=1, keepdim=True)

        for i in range(target.shape[0]):
            raw_iou = compute_iou(raw_best[i : i + 1], target[i : i + 1])
            ref_iou = compute_iou(refined_mean[i : i + 1], target[i : i + 1])
            self.raw_iou_sum += raw_iou
            self.refined_iou_sum += ref_iou
            self.raw_ap_sum += mask_ap_from_iou(raw_iou)
            self.refined_ap_sum += mask_ap_from_iou(ref_iou)
            self.raw_ap50_sum += 1.0 if raw_iou >= 0.5 else 0.0
            self.refined_ap50_sum += 1.0 if ref_iou >= 0.5 else 0.0
            self.count += 1

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "raw_sam_iou": 0.0,
                "refined_iou": 0.0,
                "delta_iou": 0.0,
                "raw_sam_ap": 0.0,
                "refined_ap": 0.0,
                "delta_ap": 0.0,
                "raw_sam_ap50": 0.0,
                "refined_ap50": 0.0,
                "delta_ap50": 0.0,
            }
        n = self.count
        raw_iou = self.raw_iou_sum / n
        ref_iou = self.refined_iou_sum / n
        raw_ap = self.raw_ap_sum / n
        ref_ap = self.refined_ap_sum / n
        raw_ap50 = self.raw_ap50_sum / n
        ref_ap50 = self.refined_ap50_sum / n
        return {
            "raw_sam_iou": raw_iou,
            "refined_iou": ref_iou,
            "delta_iou": ref_iou - raw_iou,
            "raw_sam_ap": raw_ap,
            "refined_ap": ref_ap,
            "delta_ap": ref_ap - raw_ap,
            "raw_sam_ap50": raw_ap50,
            "refined_ap50": ref_ap50,
            "delta_ap50": ref_ap50 - raw_ap50,
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
