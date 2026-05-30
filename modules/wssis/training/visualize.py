"""
Per-epoch refinement pipeline visualization (report/EXPERIMENT.md Phase 3).

Saves a grid per sample: Image | Weak signal | Raw SAM | GNN refined (pseudo) | GT
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from modules.wssis.paths import stage1_viz_dir
from modules.wssis.weak_prompts import build_instance_prompts

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize binary mask to (H, W)."""
    import cv2

    h, w = size
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (0.2, 0.8, 0.2),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend binary mask onto RGB uint8 image."""
    out = image_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    if not m.any():
        return image_rgb
    for c in range(3):
        out[..., c] = np.where(
            m,
            (1 - alpha) * out[..., c] + alpha * (color[c] * 255),
            out[..., c],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_weak_signal_cv2(image_rgb: np.ndarray, prompts: dict) -> np.ndarray:
    """Draw weak signal without matplotlib figure (faster)."""
    import cv2

    out = image_rgb.copy()
    if "bbox" in prompts:
        x, y, bw, bh = [int(v) for v in prompts["bbox"]]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
    if "point" in prompts:
        px, py = int(prompts["point"][0]), int(prompts["point"][1])
        cv2.circle(out, (px, py), 8, (0, 255, 0), -1)
        cv2.circle(out, (px, py), 8, (0, 0, 0), 1)
    return out


def sam_masks_from_prompts(
    sam_model,
    image_rgb: np.ndarray,
    prompts: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SAM prompt decoder via SamPredictor.

    Returns:
        best_mask: [H, W] bool — highest-scoring of 3 multimask outputs
        pseudo_mask: [H, W] bool — 2/3 vote agreement (pseudo-label)
    """
    from segment_anything import SamPredictor

    predictor = SamPredictor(sam_model)
    predictor.set_image(image_rgb)

    point_coords = None
    point_labels = None
    box = None

    if "point" in prompts:
        point_coords = np.array([prompts["point"]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
    if "bbox" in prompts:
        x, y, bw, bh = prompts["bbox"]
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    # masks: [C, H, W]
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx] > 0

    binary = (masks > 0).astype(np.float32)
    votes = binary.sum(axis=0)
    pseudo_mask = votes >= 2

    return best_mask, pseudo_mask


def gnn_mask_from_inputs(
    refiner: torch.nn.Module,
    sam_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    prompts: dict,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    target_hw: Tuple[int, int],
    mask_size: int = 256,
) -> np.ndarray:
    """GNN refined mask at original image resolution (PLAN §2 pipeline)."""
    from modules.vig_refinenet.sam_stage1_common import (
        build_weak_signal_tensor,
        decode_sam_masks_3_batch,
        encode_sam_embeddings,
    )

    device = image_tensor.device
    mask_np = np.zeros((mask_size, mask_size), dtype=np.uint8)
    with torch.no_grad():
        embed = encode_sam_embeddings(sam_model, image_tensor, pixel_mean, pixel_std)
        from modules.wssis.weak_prompts import sam_prompt_for_signal

        sam_prompt = sam_prompt_for_signal(prompts, "points_only")
        sam_masks_3, _ = decode_sam_masks_3_batch(
            sam_model,
            image_tensor,
            [sam_prompt],
            mask_size=mask_size,
            prompt_space=image_tensor.shape[-1],
        )
        weak_signal = build_weak_signal_tensor(
            [prompts],
            spatial_size=mask_size,
            device=device,
            mask_np_list=[mask_np],
            active_signal=None,
            policy="val_fixed",
        )
        logits = refiner(embed, image_tensor, sam_masks_3, weak_signal)
        prob = torch.sigmoid(logits).mean(dim=1, keepdim=True)
        if prob.shape[-2:] != target_hw:
            prob = F.interpolate(
                prob,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
    return (prob[0, 0].cpu().numpy() > 0.5)


def save_refinement_grid(
    panels: List[Tuple[str, np.ndarray]],
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Save horizontal 1×N panel grid."""
    if plt is None:
        raise ImportError("matplotlib required for visualization")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_stage1_epoch(
    epoch: int,
    dataset,
    sam_model: torch.nn.Module,
    refiner: torch.nn.Module,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    device: torch.device,
    run_name: str = "default",
    output_dir: Optional[Path] = None,
    sample_indices: Optional[Sequence[int]] = None,
    num_samples: int = 4,
    policy: str = "val_fixed",
) -> Path:
    """
    Save 1×5 refinement grids for fixed val samples after each epoch.

    Panels: Image | Weak signal | Raw SAM | GNN refined (pseudo) | GT
    """
    out_dir = Path(output_dir) if output_dir is not None else stage1_viz_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    if sample_indices is None:
        if n == 0:
            return out_dir
        step = max(1, n // num_samples)
        sample_indices = [min(i * step, n - 1) for i in range(num_samples)]

    refiner.eval()
    sam_model.eval()

    for row_idx, ds_idx in enumerate(sample_indices):
        image_rgb, mask_np, meta = dataset.get_raw_pair(ds_idx)
        h, w = image_rgb.shape[:2]

        import cv2

        img1024 = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask1024 = cv2.resize(
            mask_np.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST
        )
        prompts_orig = build_instance_prompts(mask_np > 0, policy=policy, signal_type="mixed")
        prompts_1024 = build_instance_prompts(mask1024 > 0, policy=policy, signal_type="mixed")
        weak_vis = _draw_weak_signal_cv2(image_rgb, prompts_orig)

        try:
            sam_mask, pseudo_sam = sam_masks_from_prompts(sam_model, image_rgb, prompts_orig)
        except Exception as e:
            print(f"[viz] SAM predict failed for idx {ds_idx}: {e}")
            sam_mask = np.zeros((h, w), dtype=bool)
            pseudo_sam = sam_mask.copy()

        image_t = torch.from_numpy(img1024).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        gnn_mask = gnn_mask_from_inputs(
            refiner,
            sam_model,
            image_t,
            prompts_1024,
            pixel_mean,
            pixel_std,
            target_hw=(h, w),
            mask_size=256,
        )

        gt_mask = mask_np > 0

        panels = [
            ("Image", image_rgb),
            ("Weak signal", weak_vis),
            ("Raw SAM", _overlay_mask(image_rgb, sam_mask, color=(1.0, 0.3, 0.3))),
            ("GNN refined (pseudo)", _overlay_mask(image_rgb, gnn_mask, color=(0.2, 0.5, 1.0))),
            ("GT", _overlay_mask(image_rgb, gt_mask, color=(0.2, 0.9, 0.2))),
        ]

        fname = out_dir / f"epoch_{epoch:03d}_sample_{row_idx:02d}_img{meta['image_id']}.png"
        save_refinement_grid(
            panels,
            fname,
            title=f"Epoch {epoch} | img {meta['image_id']} | ann {meta['ann_id']}",
        )

    # Combined montage: stack rows
    _save_epoch_montage(out_dir, epoch, len(sample_indices))
    print(f"[viz] Epoch {epoch} grids -> {out_dir}")
    return out_dir


def _save_epoch_montage(viz_dir: Path, epoch: int, num_rows: int) -> None:
    """Optional single PNG combining all samples for the epoch."""
    if plt is None or num_rows == 0:
        return
    paths = sorted(viz_dir.glob(f"epoch_{epoch:03d}_sample_*.png"))
    if not paths:
        return
    import matplotlib.image as mpimg

    imgs = [mpimg.imread(str(p)) for p in paths]
    fig, axes = plt.subplots(len(imgs), 1, figsize=(20, 4 * len(imgs)))
    if len(imgs) == 1:
        axes = [axes]
    for ax, p, im in zip(axes, paths, imgs):
        ax.imshow(im)
        ax.set_title(p.name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(viz_dir / f"epoch_{epoch:03d}_montage.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
